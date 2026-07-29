//! One recording, many documents.
//!
//! A recording used to be a single Markdown file that everything got appended
//! to: the raw transcript first, then the LLM cleanup, then a summary, each
//! separated by a `---`. That made the result impossible to use — you could
//! not copy "the corrected text" without hand-selecting it out of the middle
//! of the file, re-running the cleanup fed the previous cleanup back into the
//! model, and the history preview showed whichever section happened to be
//! first.
//!
//! Now the raw transcript stays untouched in `<stem>.md` and every derived
//! document is its own sibling file:
//!
//! ```text
//! 2026-07-26_14-30_meeting.md              ← transcript, never rewritten
//! 2026-07-26_14-30_meeting.cleanup.md      ← LLM cleanup
//! 2026-07-26_14-30_meeting.summary.md      ← summary
//! 2026-07-26_14-30_meeting.action-items.md ← action items
//! 2026-07-26_14-30_meeting.meta.json       ← which of the above exist
//! ```
//!
//! [`migrate_legacy`] splits already-appended files into that layout the
//! first time they are listed, so existing transcripts gain the same
//! structure without the user doing anything.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use chrono::{Datelike, Local, NaiveDateTime, TimeZone};

use crate::output::transcript_meta::TranscriptMeta;

/// The kinds of document a recording can have.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DocKind {
    /// The verbatim transcript written while recording.
    Transcript,
    /// The second pass over the saved audio: bigger windows, better models,
    /// the variants reconciled. See [`crate::transcription::post`].
    PostTranscript,
    /// Every model's second-pass transcript side by side.
    PostVariants,
    Cleanup,
    Summary,
    ActionItems,
    Custom,
}

/// Every derived kind, in the order they are shown as tabs.
///
/// The post-transcription comes first because it supersedes the live one: if
/// it exists, it is the transcript a reader wants.
pub const DERIVED_KINDS: [DocKind; 6] = [
    DocKind::PostTranscript,
    DocKind::PostVariants,
    DocKind::Cleanup,
    DocKind::Summary,
    DocKind::ActionItems,
    DocKind::Custom,
];

impl DocKind {
    /// Key this document is filed under in the sidecar. For the LLM presets
    /// this is also the preset key from the AI settings.
    pub fn preset_key(self) -> &'static str {
        match self {
            DocKind::Transcript => "",
            DocKind::PostTranscript => "post",
            DocKind::PostVariants => "post_variants",
            DocKind::Cleanup => "cleanup",
            DocKind::Summary => "summary",
            DocKind::ActionItems => "action_items",
            DocKind::Custom => "custom",
        }
    }

    pub fn from_preset(key: &str) -> Self {
        match key {
            "post" => DocKind::PostTranscript,
            "post_variants" => DocKind::PostVariants,
            "summary" => DocKind::Summary,
            "action_items" => DocKind::ActionItems,
            "custom" => DocKind::Custom,
            _ => DocKind::Cleanup,
        }
    }

    /// Is this document produced by running an LLM preset over a transcript?
    ///
    /// The post-transcription kinds are not: they need the recording's audio
    /// and a speech model, so the "generate this with AI" button that every
    /// other empty tab offers would be a lie.
    pub fn is_llm_preset(self) -> bool {
        !matches!(
            self,
            DocKind::Transcript | DocKind::PostTranscript | DocKind::PostVariants
        )
    }

    /// Filename infix, e.g. `cleanup` in `meeting.cleanup.md`.
    pub fn suffix(self) -> Option<&'static str> {
        match self {
            DocKind::Transcript => None,
            DocKind::PostTranscript => Some("post"),
            DocKind::PostVariants => Some("post-variants"),
            DocKind::Cleanup => Some("cleanup"),
            DocKind::Summary => Some("summary"),
            DocKind::ActionItems => Some("action-items"),
            DocKind::Custom => Some("custom"),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            DocKind::Transcript => "Transkript",
            DocKind::PostTranscript => "Nachbearbeitet",
            DocKind::PostVariants => "Modellvergleich",
            DocKind::Cleanup => "Bereinigung",
            DocKind::Summary => "Zusammenfassung",
            DocKind::ActionItems => "Action Items",
            DocKind::Custom => "Eigener Prompt",
        }
    }

    /// Shown when the document has not been generated yet.
    pub fn empty_hint(self) -> &'static str {
        match self {
            DocKind::Transcript => "Für diese Aufnahme gibt es kein Transkript.",
            DocKind::PostTranscript => {
                "Noch nicht nachbearbeitet. Der zweite Durchlauf transkribiert die \
                 gespeicherte Aufnahme erneut — mit größeren Abschnitten und mehreren \
                 Modellen."
            },
            DocKind::PostVariants => {
                "Kein Modellvergleich vorhanden. Er entsteht, wenn die Nachbearbeitung \
                 mit mehr als einem Modell läuft."
            },
            DocKind::Cleanup => {
                "Noch nicht bereinigt. Die Bereinigung entfernt Füllwörter und setzt Satzzeichen."
            },
            DocKind::Summary => "Noch keine Zusammenfassung erstellt.",
            DocKind::ActionItems => "Noch keine Action Items extrahiert.",
            DocKind::Custom => "Noch nichts mit dem eigenen Prompt erzeugt.",
        }
    }

    /// Every suffix that marks a file as derived rather than a transcript.
    fn all_suffixes() -> [&'static str; 6] {
        [
            "post",
            "post-variants",
            "cleanup",
            "summary",
            "action-items",
            "custom",
        ]
    }
}

/// Path of the document `kind` belonging to the transcript at `transcript`.
pub fn doc_path(transcript: &Path, kind: DocKind) -> PathBuf {
    match kind.suffix() {
        None => transcript.to_path_buf(),
        Some(suffix) => {
            let stem = transcript
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("transcript");
            transcript.with_file_name(format!("{stem}.{suffix}.md"))
        },
    }
}

/// `true` if `path` is a derived document rather than a transcript.
pub fn is_derived(path: &Path) -> bool {
    let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
        return false;
    };
    DocKind::all_suffixes()
        .iter()
        .any(|s| stem.ends_with(&format!(".{s}")))
}

/// A recording and everything that has been derived from it.
#[derive(Debug, Clone)]
pub struct Recording {
    pub path: PathBuf,
    /// Human title from the transcript header, falling back to the filename.
    pub title: String,
    /// When the recording started, parsed from the filename.
    pub started: Option<chrono::DateTime<Local>>,
    pub modified: SystemTime,
    /// Formatted duration from the header, e.g. `0h 02m 05s`.
    pub duration: Option<String>,
    /// First line of actual speech, for the list preview.
    pub preview: String,
    pub meta: TranscriptMeta,
    /// Derived documents that exist on disk.
    pub docs: Vec<DocKind>,
}

impl Recording {
    pub fn load(path: PathBuf) -> Self {
        let body = std::fs::read_to_string(&path).unwrap_or_default();
        let modified = std::fs::metadata(&path)
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_string();

        let docs = DERIVED_KINDS
            .iter()
            .copied()
            .filter(|k| doc_path(&path, *k).is_file())
            .collect();

        Self {
            title: header_title(&body).unwrap_or_else(|| stem.clone()),
            started: parse_started(&stem),
            duration: header_field(&body, "Duration"),
            preview: first_speech_line(&body),
            meta: TranscriptMeta::load(&path),
            docs,
            modified,
            path,
        }
    }

    pub fn has_doc(&self, kind: DocKind) -> bool {
        kind == DocKind::Transcript || self.docs.contains(&kind)
    }

    pub fn doc_path(&self, kind: DocKind) -> PathBuf {
        doc_path(&self.path, kind)
    }

    pub fn read_doc(&self, kind: DocKind) -> Option<String> {
        std::fs::read_to_string(self.doc_path(kind)).ok()
    }

    /// When the document was generated, if we recorded it.
    pub fn doc_created_at(&self, kind: DocKind) -> Option<String> {
        self.meta
            .docs
            .get(kind.preset_key())
            .map(|d| d.created_at.clone())
    }

    /// Date line for the list and the detail header.
    pub fn display_date(&self) -> String {
        match self.started {
            Some(dt) => format_de(dt),
            None => {
                let dt: chrono::DateTime<Local> = self.modified.into();
                format_de(dt)
            },
        }
    }

    /// Delete the transcript, every derived document, and the sidecar.
    pub fn delete(&self) -> std::io::Result<()> {
        for kind in DERIVED_KINDS {
            let p = self.doc_path(kind);
            if p.is_file() {
                let _ = std::fs::remove_file(p);
            }
        }
        TranscriptMeta::delete(&self.path);
        std::fs::remove_file(&self.path)
    }

    /// Does any document of this recording contain `needle` (lowercased)?
    pub fn matches(&self, needle: &str) -> bool {
        if needle.is_empty() {
            return true;
        }
        if self.title.to_lowercase().contains(needle)
            || self.preview.to_lowercase().contains(needle)
        {
            return true;
        }
        std::iter::once(DocKind::Transcript)
            .chain(self.docs.iter().copied())
            .any(|kind| {
                std::fs::read_to_string(self.doc_path(kind))
                    .map(|c| c.to_lowercase().contains(needle))
                    .unwrap_or(false)
            })
    }
}

/// List every recording in `dir`, newest first. Legacy single-file
/// transcripts are split up on the way through.
pub fn scan(dir: &Path) -> Vec<Recording> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };

    let mut out: Vec<Recording> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().map(|e| e == "md").unwrap_or(false))
        .filter(|p| !is_derived(p))
        .map(|p| {
            if let Err(e) = migrate_legacy(&p) {
                log::warn!("transcript migration failed for {}: {e}", p.display());
            }
            Recording::load(p)
        })
        .collect();

    out.sort_by(|a, b| match (a.started, b.started) {
        (Some(x), Some(y)) => y.cmp(&x),
        _ => b.modified.cmp(&a.modified),
    });
    out
}

// --- legacy migration ------------------------------------------------------

/// Headings older builds appended into the transcript file, mapped to the
/// document they should have been.
const LEGACY_SECTIONS: &[(&str, DocKind)] = &[
    ("## Bereinigung — ", DocKind::Cleanup),
    ("## Bereinigung —", DocKind::Cleanup),
    ("## Cleanup — ", DocKind::Cleanup),
    ("## Summary — ", DocKind::Summary),
    ("## Zusammenfassung — ", DocKind::Summary),
    ("## Action Items — ", DocKind::ActionItems),
];

/// Notes older builds appended when no summary could be produced. They are
/// status messages, not transcript content.
const LEGACY_NOTES: &[&str] = &[
    "_No speech was detected",
    "_Configure an AI provider",
    "_Summary skipped",
    "_Summary failed",
];

/// Split appended sections of `transcript` into sibling documents. No-op for
/// files that are already in the new layout.
pub fn migrate_legacy(transcript: &Path) -> std::io::Result<()> {
    let body = std::fs::read_to_string(transcript)?;
    let lines: Vec<&str> = body.lines().collect();

    // Index of the first appended section heading, if any.
    let mut cuts: Vec<(usize, DocKind)> = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if let Some((_, kind)) = LEGACY_SECTIONS
            .iter()
            .find(|(prefix, _)| trimmed.starts_with(prefix))
        {
            cuts.push((i, *kind));
        }
    }

    let has_notes = lines
        .iter()
        .any(|l| LEGACY_NOTES.iter().any(|n| l.trim_start().starts_with(n)));
    if cuts.is_empty() && !has_notes {
        return Ok(());
    }

    let mut meta = TranscriptMeta::load(transcript);

    // Write each appended section out to its own file, oldest section first.
    for (idx, (start, kind)) in cuts.iter().enumerate() {
        let end = cuts.get(idx + 1).map(|(i, _)| *i).unwrap_or(lines.len());
        let stamp = lines[*start]
            .rsplit('—')
            .next()
            .map(|s| s.trim().to_string())
            .unwrap_or_default();
        let content = lines[start + 1..end].join("\n");
        let content = trim_separators(&content);
        if content.is_empty() {
            continue;
        }
        let target = doc_path(transcript, *kind);
        if !target.is_file() {
            let title = header_title(&body).unwrap_or_else(|| "Meeting".into());
            let header = doc_header(kind.label(), &title, &stamp, "", transcript);
            std::fs::write(&target, format!("{header}{content}\n"))?;
            log::info!("migrated legacy section → {}", target.display());
        }
        meta.record_doc(kind.preset_key(), &stamp, "", "");
    }

    // Keep only the transcript itself in the original file. A file that
    // *starts* with an appended section is not a transcript we can split —
    // leave it exactly as it is rather than truncating it to nothing.
    let keep_until = cuts.first().map(|(i, _)| *i).unwrap_or(lines.len());
    if keep_until == 0 {
        let _ = meta.save(transcript);
        return Ok(());
    }
    let kept: Vec<&str> = lines[..keep_until]
        .iter()
        .copied()
        .filter(|l| !LEGACY_NOTES.iter().any(|n| l.trim_start().starts_with(n)))
        .collect();
    let rewritten = format!("{}\n", trim_separators(&kept.join("\n")));
    if rewritten != body {
        std::fs::write(transcript, rewritten)?;
    }
    let _ = meta.save(transcript);
    Ok(())
}

/// Drop trailing whitespace and dangling `---` separators.
fn trim_separators(s: &str) -> String {
    let mut lines: Vec<&str> = s.lines().collect();
    while let Some(last) = lines.last() {
        let t = last.trim();
        if t.is_empty() || t == "---" {
            lines.pop();
        } else {
            break;
        }
    }
    lines.join("\n")
}

// --- document contents -----------------------------------------------------

/// Header written at the top of every derived document.
pub fn doc_header(
    label: &str,
    title: &str,
    stamp: &str,
    model: &str,
    source: &Path,
) -> String {
    let source_name = source
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or_default();
    let mut meta_line = String::new();
    if !stamp.is_empty() {
        meta_line.push_str(stamp);
    }
    if !model.is_empty() {
        if !meta_line.is_empty() {
            meta_line.push_str(" · ");
        }
        meta_line.push_str(model);
    }
    format!(
        "# {label} — {title}\n\n_{meta_line}_\n_Quelle: {source_name}_\n\n---\n\n",
        meta_line = if meta_line.is_empty() {
            "erzeugt".to_string()
        } else {
            meta_line
        },
    )
}

/// The content of a document without its metadata header — this is what a
/// "Kopieren" button puts on the clipboard.
pub fn body_of(markdown: &str) -> String {
    let mut lines = markdown.lines().peekable();
    // Skip the leading metadata block: headings, italic meta lines, list
    // items from the transcript header, separators and blank lines.
    while let Some(line) = lines.peek() {
        let t = line.trim();
        let is_meta = t.is_empty()
            || t == "---"
            || t.starts_with('#')
            || (t.starts_with('_') && t.ends_with('_'))
            || t.starts_with("- **Start:**")
            || t.starts_with("- **Duration:**");
        if is_meta {
            lines.next();
        } else {
            break;
        }
    }
    lines.collect::<Vec<_>>().join("\n").trim_end().to_string()
}

/// Transcript body with the `**[MM:SS] Sprecher:**` markers removed — for
/// pasting the spoken text somewhere that does not want the scaffolding.
pub fn plain_text(markdown: &str) -> String {
    body_of(markdown)
        .lines()
        .map(strip_line_marker)
        .filter(|l| !l.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

/// `**[00:12] Du:** hallo` → `hallo`
pub fn strip_line_marker(line: &str) -> &str {
    let trimmed = line.trim();
    let Some(rest) = trimmed.strip_prefix("**[") else {
        return trimmed;
    };
    match rest.find(":**") {
        Some(end) => rest[end + 3..].trim_start(),
        None => match rest.find("]**") {
            Some(end) => rest[end + 3..].trim_start(),
            None => trimmed,
        },
    }
}

// --- header parsing --------------------------------------------------------

/// Title from a transcript's `# …` header, or a fallback.
pub fn title_of(body: &str) -> String {
    header_title(body).unwrap_or_else(|| "Meeting".to_string())
}

fn header_title(body: &str) -> Option<String> {
    body.lines()
        .find(|l| l.starts_with("# "))
        .map(|l| l[2..].trim().to_string())
        .filter(|s| !s.is_empty())
}

fn header_field(body: &str, field: &str) -> Option<String> {
    let needle = format!("- **{field}:**");
    body.lines()
        .find(|l| l.trim_start().starts_with(&needle))
        .map(|l| l.trim_start()[needle.len()..].trim().to_string())
        .filter(|s| !s.is_empty() && !s.starts_with("_("))
}

/// First line that is actual speech, for the list preview.
fn first_speech_line(body: &str) -> String {
    for line in body.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') || t.starts_with('-') || t == "---" {
            continue;
        }
        let cleaned = strip_line_marker(t);
        if cleaned.is_empty() {
            continue;
        }
        return cleaned.chars().take(90).collect();
    }
    "Keine Sprache erkannt".to_string()
}

/// Filenames look like `2026-07-26_14-30_meeting`.
fn parse_started(stem: &str) -> Option<chrono::DateTime<Local>> {
    if stem.len() < 16 {
        return None;
    }
    let naive = NaiveDateTime::parse_from_str(&stem[..16], "%Y-%m-%d_%H-%M").ok()?;
    Local.from_local_datetime(&naive).single()
}

const WEEKDAYS_DE: [&str; 7] = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"];
const MONTHS_DE: [&str; 12] = [
    "Januar",
    "Februar",
    "März",
    "April",
    "Mai",
    "Juni",
    "Juli",
    "August",
    "September",
    "Oktober",
    "November",
    "Dezember",
];

/// `Heute · 14:32`, `Gestern · 09:15`, `Fr, 24. Juli · 14:32`.
pub fn format_de(dt: chrono::DateTime<Local>) -> String {
    let today = Local::now().date_naive();
    let day = dt.date_naive();
    let time = dt.format("%H:%M");
    if day == today {
        return format!("Heute · {time}");
    }
    if today.signed_duration_since(day).num_days() == 1 {
        return format!("Gestern · {time}");
    }
    let weekday = WEEKDAYS_DE[dt.weekday().num_days_from_monday() as usize];
    let month = MONTHS_DE[(dt.month0()) as usize];
    if dt.year() == Local::now().year() {
        format!("{weekday}, {}. {month} · {time}", dt.day())
    } else {
        format!("{weekday}, {}. {month} {} · {time}", dt.day(), dt.year())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn derived_paths_are_siblings() {
        let p = Path::new("/tmp/2026-07-26_14-30_meeting.md");
        assert_eq!(
            doc_path(p, DocKind::Cleanup),
            PathBuf::from("/tmp/2026-07-26_14-30_meeting.cleanup.md")
        );
        assert_eq!(doc_path(p, DocKind::Transcript), p.to_path_buf());
        assert!(is_derived(Path::new("/tmp/x.summary.md")));
        assert!(!is_derived(p));
    }

    #[test]
    fn body_and_plain_text_strip_scaffolding() {
        let md = "# Meeting\n\n- **Start:** 2026-07-26 14:30\n- **Duration:** 0h 02m 05s\n\n---\n\n**[00:00] Du:** Guten Morgen.\n\n**[00:12] Andere:** Moin.\n";
        assert_eq!(
            body_of(md),
            "**[00:00] Du:** Guten Morgen.\n\n**[00:12] Andere:** Moin."
        );
        assert_eq!(plain_text(md), "Guten Morgen.\nMoin.");
    }

    #[test]
    fn legacy_append_is_split_into_files() {
        let dir = tempdir().unwrap();
        let md = dir.path().join("2026-07-26_14-30_meeting.md");
        std::fs::write(
            &md,
            "# Meeting\n\n- **Start:** 2026-07-26 14:30\n\n---\n\n\
             **[00:00]** Hallo zusammen.\n\n---\n\n\
             ## Bereinigung — 2026-07-26 15:01\n\nHallo zusammen!\n",
        )
        .unwrap();

        migrate_legacy(&md).unwrap();

        let cleanup = dir.path().join("2026-07-26_14-30_meeting.cleanup.md");
        assert!(cleanup.is_file());
        let cleaned = std::fs::read_to_string(&cleanup).unwrap();
        assert_eq!(body_of(&cleaned), "Hallo zusammen!");

        let raw = std::fs::read_to_string(&md).unwrap();
        assert!(!raw.contains("Bereinigung"));
        assert!(raw.contains("Hallo zusammen."));

        let rec = Recording::load(md);
        assert!(rec.has_doc(DocKind::Cleanup));
        assert!(!rec.has_doc(DocKind::Summary));
        assert_eq!(rec.preview, "Hallo zusammen.");
    }

    /// A file that is nothing but an appended section must not be emptied.
    #[test]
    fn migration_never_truncates_to_nothing() {
        let dir = tempdir().unwrap();
        let md = dir.path().join("2026-07-26_10-00_odd.md");
        let body = "## Bereinigung — 2026-07-26 10:30\n\nNur bereinigter Text.\n";
        std::fs::write(&md, body).unwrap();

        migrate_legacy(&md).unwrap();

        assert_eq!(std::fs::read_to_string(&md).unwrap(), body);
    }

    /// The post-transcription writes siblings of the transcript. If they were
    /// not recognised as derived, every recording would show up three times in
    /// the history list and `migrate_legacy` would run over them.
    #[test]
    fn post_documents_are_derived_not_recordings() {
        let p = Path::new("/tmp/2026-07-26_14-30_meeting.md");
        assert_eq!(
            doc_path(p, DocKind::PostTranscript),
            PathBuf::from("/tmp/2026-07-26_14-30_meeting.post.md")
        );
        assert_eq!(
            doc_path(p, DocKind::PostVariants),
            PathBuf::from("/tmp/2026-07-26_14-30_meeting.post-variants.md")
        );
        assert!(is_derived(&doc_path(p, DocKind::PostTranscript)));
        assert!(is_derived(&doc_path(p, DocKind::PostVariants)));
        assert!(!is_derived(p));
    }

    /// `.post` and `.post-variants` share a prefix; neither may be mistaken
    /// for the other, in either direction.
    #[test]
    fn post_kinds_round_trip_through_their_keys() {
        for kind in [DocKind::PostTranscript, DocKind::PostVariants] {
            assert_eq!(DocKind::from_preset(kind.preset_key()), kind);
            assert!(!kind.is_llm_preset(), "{kind:?} is not an LLM preset");
        }
        assert!(DocKind::Cleanup.is_llm_preset());
        assert!(DocKind::Summary.is_llm_preset());
    }

    /// A recording lists its post documents alongside the LLM ones, and
    /// deleting it takes all of them.
    #[test]
    fn post_documents_are_listed_and_deleted_with_the_recording() {
        let dir = tempdir().unwrap();
        let md = dir.path().join("2026-07-26_14-30_meeting.md");
        std::fs::write(&md, "# Meeting\n\n---\n\n**[00:00]** Hallo.\n").unwrap();
        let post = doc_path(&md, DocKind::PostTranscript);
        let variants = doc_path(&md, DocKind::PostVariants);
        std::fs::write(&post, "# Nachbearbeitet\n").unwrap();
        std::fs::write(&variants, "# Modellvergleich\n").unwrap();

        let found = scan(dir.path());
        assert_eq!(found.len(), 1, "the post files are not their own recordings");
        assert!(found[0].has_doc(DocKind::PostTranscript));
        assert!(found[0].has_doc(DocKind::PostVariants));
        assert!(!found[0].has_doc(DocKind::Summary));

        found[0].delete().unwrap();
        assert!(!post.exists());
        assert!(!variants.exists());
        assert!(!md.exists());
    }

    #[test]
    fn scan_skips_derived_files() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("2026-07-26_14-30_meeting.md"), "# Meeting\n").unwrap();
        std::fs::write(
            dir.path().join("2026-07-26_14-30_meeting.cleanup.md"),
            "# Bereinigt\n",
        )
        .unwrap();
        let found = scan(dir.path());
        assert_eq!(found.len(), 1);
        assert!(found[0].has_doc(DocKind::Cleanup));
    }
}
