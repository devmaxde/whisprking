//! Run an LLM preset over a transcript and store the result as its own
//! document next to it.
//!
//! Both entry points that post-process a recording — the "KI" action in the
//! history page and the automatic summary after an import — go through here,
//! so they cannot drift apart. Two rules matter:
//!
//! * the **transcript is the only input**. Feeding the file back in when it
//!   already contained a previous cleanup is what made repeated runs produce
//!   summaries of summaries.
//! * the **transcript is never written to**. The result goes to
//!   `<stem>.<kind>.md`; see [`crate::output::transcript_doc`].

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};

use crate::config::AiPostprocessConfig;
use crate::output::transcript_doc::{self, DocKind};
use crate::output::transcript_meta::TranscriptMeta;
use crate::postprocess::llm::{
    estimate_tokens, make_provider, provider_label, resolve_system_prompt, LlmConfig,
};

/// Everything needed to derive one document from one transcript.
#[derive(Debug, Clone)]
pub struct RefineJob {
    /// The transcript to read. Must be the raw `.md`, not a derived file.
    pub source: PathBuf,
    /// Preset key from the AI settings (`cleanup`, `summary`, …).
    pub preset: String,
    pub llm: LlmConfig,
    pub custom_prompt: String,
    pub overrides: HashMap<String, String>,
}

impl RefineJob {
    pub fn from_config(source: &Path, preset: &str, ai: &AiPostprocessConfig) -> Self {
        Self {
            source: source.to_path_buf(),
            preset: preset.to_string(),
            llm: LlmConfig::from_ai_section(ai),
            custom_prompt: ai.custom_prompt.clone(),
            overrides: ai
                .prompts
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        }
    }

    pub fn kind(&self) -> DocKind {
        DocKind::from_preset(&self.preset)
    }
}

/// Why a run failed, in the shape the UI needs.
///
/// This used to be a bare `String`, which is how a five-minute meeting could
/// fail with "Der Provider hat einen Fehler gemeldet: http error: operation
/// timed out" — true, useless, and silent about the two facts that explain it
/// (how long the transcript was and how long we waited). Everything needed to
/// tell that story is collected here instead.
#[derive(Debug, Clone, Default)]
pub struct RefineError {
    /// One line for the banner headline.
    pub summary: String,
    /// Label/value pairs: provider, model, transcript size, HTTP status …
    pub facts: Vec<(String, String)>,
    /// What to try next, when that is knowable.
    pub hint: Option<String>,
    /// The provider's answer, verbatim (clipped).
    pub raw: Option<String>,
}

impl RefineError {
    pub fn new(summary: impl Into<String>) -> Self {
        Self {
            summary: summary.into(),
            ..Self::default()
        }
    }

    pub fn with_facts(mut self, facts: Vec<(String, String)>) -> Self {
        self.facts = facts;
        self
    }

    pub fn with_hint(mut self, hint: impl Into<String>) -> Self {
        self.hint = Some(hint.into());
        self
    }

    pub fn with_raw(mut self, raw: impl Into<String>) -> Self {
        self.raw = Some(raw.into());
        self
    }

    /// Summary plus hint on one line — for callers that only have room for a
    /// sentence, like the import progress note.
    pub fn note(&self) -> String {
        match &self.hint {
            Some(hint) => format!("{} {hint}", self.summary),
            None => self.summary.clone(),
        }
    }

    /// The whole thing as plain text, for the clipboard and the log. This is
    /// what a user can paste into an issue.
    pub fn report(&self) -> String {
        let mut out = self.summary.clone();
        for (key, value) in &self.facts {
            out.push_str(&format!("\n{key}: {value}"));
        }
        if let Some(hint) = &self.hint {
            out.push_str(&format!("\n\n{hint}"));
        }
        if let Some(raw) = &self.raw {
            out.push_str(&format!("\n\nAntwort des Providers:\n{raw}"));
        }
        out
    }
}

impl fmt::Display for RefineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.summary)
    }
}

/// `12345` → `12.345`
fn de_int(n: usize) -> String {
    let digits = n.to_string();
    let mut groups: Vec<&str> = digits
        .as_bytes()
        .rchunks(3)
        // Digits only, so every chunk is valid UTF-8 by construction.
        .filter_map(|c| std::str::from_utf8(c).ok())
        .collect();
    groups.reverse();
    groups.join(".")
}

/// Run the job. Returns the path of the document that was written.
pub fn refine(job: &RefineJob) -> Result<PathBuf, RefineError> {
    let kind = job.kind();
    let file = job
        .source
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| job.source.display().to_string());

    let raw = std::fs::read_to_string(&job.source).map_err(|e| {
        RefineError::new("Das Transkript konnte nicht gelesen werden.")
            .with_facts(vec![("Datei".into(), job.source.display().to_string())])
            .with_raw(e.to_string())
    })?;
    let body = transcript_doc::body_of(&raw);
    if body.trim().is_empty() {
        return Err(RefineError::new("Das Transkript enthält keinen Text.")
            .with_facts(vec![("Datei".into(), file)]));
    }

    // Collected before the request so a failure can say what was attempted.
    let facts = |extra: Vec<(String, String)>| -> Vec<(String, String)> {
        let mut facts = vec![
            ("Aktion".into(), kind.label().to_string()),
            (
                "Provider".into(),
                provider_label(&job.llm.provider).to_string(),
            ),
            (
                "Modell".into(),
                if job.llm.model.is_empty() {
                    "—".into()
                } else {
                    job.llm.model.clone()
                },
            ),
            (
                "Transkript".into(),
                format!(
                    "{} Zeichen · ca. {} Tokens · {}",
                    de_int(body.chars().count()),
                    de_int(estimate_tokens(&body)),
                    file
                ),
            ),
        ];
        facts.extend(extra);
        facts
    };

    let provider = make_provider(&job.llm).map_err(|e| {
        let mut err = RefineError::new(format!("{} nicht möglich — {e}", kind.label()))
            .with_facts(facts(e.facts()));
        err.hint = e.hint();
        err
    })?;

    let system = resolve_system_prompt(&job.preset, &job.custom_prompt, &job.overrides);
    let result = provider.run(&system, &body).map_err(|e| {
        let err = RefineError {
            summary: format!("{} fehlgeschlagen — {e}", kind.label()),
            facts: facts(e.facts()),
            hint: e.hint(),
            raw: e.raw().map(str::to_string),
        };
        log::warn!("refine: {}", err.report().replace('\n', " | "));
        err
    })?;
    let result = result.trim();
    if result.is_empty() {
        return Err(
            RefineError::new(format!("{} fehlgeschlagen — leere Antwort.", kind.label()))
                .with_facts(facts(Vec::new()))
                .with_hint("Ein anderes Modell versuchen."),
        );
    }

    let now = chrono::Local::now();
    let stamp = now.format("%d.%m.%Y %H:%M").to_string();
    let header = transcript_doc::doc_header(
        kind.label(),
        &transcript_doc::title_of(&raw),
        &stamp,
        &job.llm.model,
        &job.source,
    );
    let target = transcript_doc::doc_path(&job.source, kind);
    std::fs::write(&target, format!("{header}{result}\n")).map_err(|e| {
        RefineError::new("Das Ergebnis konnte nicht gespeichert werden.")
            .with_facts(facts(vec![("Zieldatei".into(), target.display().to_string())]))
            .with_raw(e.to_string())
    })?;

    let mut meta = TranscriptMeta::load(&job.source);
    meta.record_doc(kind.preset_key(), &stamp, &job.llm.model, &job.llm.provider);
    if let Err(e) = meta.save(&job.source) {
        log::warn!("refine: could not update sidecar: {e}");
    }

    log::info!("refine: wrote {}", target.display());
    Ok(target)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thousands_separator() {
        assert_eq!(de_int(0), "0");
        assert_eq!(de_int(999), "999");
        assert_eq!(de_int(1000), "1.000");
        assert_eq!(de_int(1234567), "1.234.567");
    }

    #[test]
    fn report_contains_every_part() {
        let err = RefineError::new("Bereinigung fehlgeschlagen — Zeitüberschreitung.")
            .with_facts(vec![("Modell".into(), "x/y".into())])
            .with_hint("Kleineres Modell wählen.")
            .with_raw("{\"error\":1}");
        let text = err.report();
        assert!(text.contains("Zeitüberschreitung"));
        assert!(text.contains("Modell: x/y"));
        assert!(text.contains("Kleineres Modell"));
        assert!(text.contains("{\"error\":1}"));
        // The banner headline stays a single line.
        assert!(!err.to_string().contains('\n'));
    }
}
