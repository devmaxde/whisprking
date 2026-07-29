//! "Verlauf" — every recording, with its transcript and everything derived
//! from it.
//!
//! The page is a master/detail: recordings on the left, documents of the
//! selected recording on the right. Each document (transcript, Bereinigung,
//! Zusammenfassung, Action Items) is its own file and its own tab with its
//! own copy button — the previous version concatenated all of them into one
//! Markdown blob and rendered it as a single unselectable label, which is
//! why there was no way to get the corrected text out of it.

mod detail;
mod list;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::config::Config;
use crate::output::transcript_doc::{self, DocKind, Recording};
use crate::postprocess::refine::{refine, RefineError, RefineJob};
use crate::ui::pages::meeting::post::{existing_tracks, PostJob};
use crate::ui::icons::Icon;
use crate::ui::theme::{self, space};
use crate::ui::widgets::{self as w, Tone};

/// State of a running post-processing job, shared with its worker thread.
pub(super) enum JobPhase {
    Running,
    Done(PathBuf),
    Error(RefineError),
}

pub(super) struct Job {
    pub source: PathBuf,
    pub kind: DocKind,
    pub phase: JobPhase,
}

/// The document currently shown in the detail pane, kept in memory.
///
/// Both fields are `Arc` so the copy buttons can hold on to the text without
/// cloning a whole transcript every frame.
pub(super) struct DocCache {
    pub path: PathBuf,
    pub kind: DocKind,
    /// Document text without its Markdown header.
    pub body: Arc<str>,
    /// Transcript text with the `**[MM:SS] Sprecher:**` markers removed.
    pub plain: Arc<str>,
}

pub struct HistoryPage {
    search: String,
    recordings: Vec<Recording>,
    selected: Option<usize>,
    /// Indices of `recordings` matching `search`, and the query they were
    /// computed for — searching reads every document, so it must not happen
    /// per frame.
    visible: Vec<usize>,
    filtered_for: Option<String>,
    /// Contents of the open document; reading it per frame would mean
    /// re-reading megabytes off disk while the mouse moves.
    doc_cache: Option<DocCache>,
    /// Document tab shown in the detail pane.
    tab: DocKind,
    /// Preset the "Erzeugen" button runs.
    preset: String,
    job: Arc<Mutex<Option<Job>>>,
    /// Re-running the second pass over a recording whose audio is still on
    /// disk. Separate from `job`: it runs speech models, not an LLM preset,
    /// and it takes minutes rather than seconds.
    post: PostJob,
    /// Pending delete confirmation.
    confirm_delete: Option<PathBuf>,
    status: Option<(Tone, String)>,
    /// Last failed post-processing run. Kept separate from `status` because it
    /// carries a whole report, and it stays on screen until dismissed.
    error: Option<RefineError>,
    copy_doc: w::CopyButton,
    copy_plain: w::CopyButton,
    copy_error: w::CopyButton,
}

impl HistoryPage {
    pub fn new() -> Self {
        Self {
            search: String::new(),
            recordings: Vec::new(),
            selected: None,
            visible: Vec::new(),
            filtered_for: None,
            doc_cache: None,
            tab: DocKind::Transcript,
            preset: String::new(),
            job: Arc::new(Mutex::new(None)),
            post: PostJob::default(),
            confirm_delete: None,
            status: None,
            error: None,
            copy_doc: w::CopyButton::default(),
            copy_plain: w::CopyButton::default(),
            copy_error: w::CopyButton::default(),
        }
    }

    /// Re-scan the transcripts directory, keeping the current selection when
    /// the same recording is still there.
    pub fn refresh(&mut self, config: &Config) {
        let keep = self.selected_path();
        self.recordings = transcript_doc::scan(&Config::transcripts_dir(config));
        self.selected = match keep {
            Some(path) => self
                .recordings
                .iter()
                .position(|r| r.path == path)
                .or((!self.recordings.is_empty()).then_some(0)),
            None => (!self.recordings.is_empty()).then_some(0),
        };
        if self.preset.is_empty() {
            self.preset = config.ai_postprocess.default_prompt.clone();
        }
        self.filtered_for = None;
        self.doc_cache = None;
        self.clamp_tab();
    }

    /// Recompute the filtered index list when the query (or the list) changed.
    fn ensure_filter(&mut self) {
        if self.filtered_for.as_deref() == Some(self.search.as_str()) {
            return;
        }
        let needle = self.search.trim().to_lowercase();
        self.visible = (0..self.recordings.len())
            .filter(|i| self.recordings[*i].matches(&needle))
            .collect();
        self.filtered_for = Some(self.search.clone());
    }

    /// Load the open document if the cache does not already hold it.
    fn ensure_doc(&mut self, rec: &Recording, kind: DocKind) {
        let hit = self
            .doc_cache
            .as_ref()
            .is_some_and(|c| c.path == rec.path && c.kind == kind);
        if hit {
            return;
        }
        self.doc_cache = rec.read_doc(kind).map(|raw| DocCache {
            path: rec.path.clone(),
            kind,
            body: Arc::from(transcript_doc::body_of(&raw)),
            plain: Arc::from(transcript_doc::plain_text(&raw)),
        });
    }

    fn selected_path(&self) -> Option<PathBuf> {
        self.selected
            .and_then(|i| self.recordings.get(i))
            .map(|r| r.path.clone())
    }

    fn selected_recording(&self) -> Option<&Recording> {
        self.selected.and_then(|i| self.recordings.get(i))
    }

    /// Never leave the detail pane on a tab whose document does not exist.
    fn clamp_tab(&mut self) {
        let ok = self
            .selected_recording()
            .map(|r| r.has_doc(self.tab))
            .unwrap_or(false);
        if !ok {
            self.tab = DocKind::Transcript;
        }
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, config: &Config) {
        self.poll_job(config);

        w::page_header(ui, "Verlauf", "Aufnahmen, Transkripte und KI-Ergebnisse");
        ui.add_space(space::LG);

        self.toolbar(ui, config);
        self.ensure_filter();
        ui.add_space(space::MD);

        if let Some((tone, msg)) = self.status.clone() {
            w::banner(ui, tone, &msg);
            ui.add_space(space::MD);
        }

        // Taken out of `self` for the call: the panel needs `&mut` on the copy
        // button, which lives in the same struct.
        if let Some(err) = self.error.take() {
            let dismissed = w::error_panel(
                ui,
                &err.summary,
                &err.facts,
                err.hint.as_deref(),
                err.raw.as_deref(),
                &mut self.copy_error,
                || err.report(),
            );
            if !dismissed {
                self.error = Some(err);
            }
            ui.add_space(space::MD);
        }

        if self.recordings.is_empty() {
            w::empty_state(
                ui,
                Icon::Transcript,
                "Noch keine Aufnahmen",
                "Nimm ein Meeting auf oder importiere eine Datei — die Transkripte landen hier.",
            );
            return;
        }

        // Both column widths are computed up front: asking for
        // `available_width()` between the two columns does not account for
        // the item spacing egui inserts after the first one, and the detail
        // card would end up wider than the window.
        let height = ui.available_height();
        let gap = space::MD + ui.spacing().item_spacing.x;
        let list_width = theme::LIST_WIDTH.min(ui.available_width() * 0.4);
        let detail_width = (ui.available_width() - list_width - gap).max(280.0);

        ui.horizontal_top(|ui| {
            ui.allocate_ui_with_layout(
                egui::vec2(list_width, height),
                egui::Layout::top_down(egui::Align::Min),
                |ui| self.list_pane(ui),
            );
            ui.add_space(space::MD);
            ui.allocate_ui_with_layout(
                egui::vec2(detail_width, height),
                egui::Layout::top_down(egui::Align::Min),
                |ui| self.detail_pane(ui, config),
            );
        });
    }

    fn toolbar(&mut self, ui: &mut egui::Ui, config: &Config) {
        let count = self.recordings.len();
        let mut refresh = false;
        egui::Sides::new().height(theme::CONTROL_HEIGHT).show(
            ui,
            |ui| {
                ui.set_max_width(360.0);
                if w::search_field(
                    ui,
                    "history-search",
                    &mut self.search,
                    "In Transkripten suchen …",
                ) {
                    self.filtered_for = None;
                }
            },
            |ui| {
                refresh = w::icon_button(ui, Icon::Refresh, "Neu einlesen").clicked();
                w::text_line(
                    ui,
                    &format!("{count} Aufnahmen"),
                    theme::text::small(),
                    theme::TEXT_MUTED,
                );
            },
        );
        if refresh {
            self.refresh(config);
        }
    }

    // --- post-processing job ----------------------------------------------

    fn job_running(&self) -> bool {
        matches!(
            self.job.lock().expect("job").as_ref().map(|j| &j.phase),
            Some(JobPhase::Running)
        )
    }

    /// Pick up a finished job: reload from disk and jump to the new document.
    fn poll_job(&mut self, config: &Config) {
        let finished = {
            let mut guard = self.job.lock().expect("job");
            let settled = matches!(
                guard.as_ref().map(|j| &j.phase),
                Some(JobPhase::Done(_)) | Some(JobPhase::Error(_))
            );
            if settled {
                guard.take()
            } else {
                None
            }
        };
        let Some(job) = finished else { return };
        match job.phase {
            JobPhase::Done(path) => {
                let name = path.file_name().and_then(|s| s.to_str()).unwrap_or_default();
                self.status = Some((
                    Tone::Success,
                    format!("{} erzeugt · {name}", job.kind.label()),
                ));
                self.refresh(config);
                // Only jump to the new tab if the user is still looking at
                // the recording the job belonged to.
                let same = self.selected_path().as_deref() == Some(job.source.as_path());
                if same
                    && self
                        .selected_recording()
                        .map(|r| r.has_doc(job.kind))
                        .unwrap_or(false)
                {
                    self.tab = job.kind;
                }
            },
            JobPhase::Error(err) => {
                self.error = Some(err);
            },
            JobPhase::Running => {},
        }
    }

    /// Re-run the post-transcription for the selected recording.
    ///
    /// Only possible while its audio is still there, which is why
    /// `keep_audio` exists — the transcript alone cannot be re-decoded.
    fn start_post(&mut self, config: &Config, ctx: &egui::Context) {
        let Some(source) = self.selected_path() else {
            return;
        };
        let tracks = existing_tracks(&source, config);
        if tracks.is_empty() {
            self.status = Some((
                Tone::Warning,
                "Zu dieser Aufnahme liegt kein Audio mehr vor.".into(),
            ));
            return;
        }
        self.status = None;
        self.error = None;
        self.post.clear();
        self.post.start(source, tracks, config, ctx);
    }

    /// Start `preset` for the selected recording on a worker thread.
    fn start_job(&mut self, config: &Config, ctx: &egui::Context) {
        let Some(source) = self.selected_path() else {
            return;
        };
        let preset = self.preset.clone();
        let kind = DocKind::from_preset(&preset);
        let request = RefineJob::from_config(&source, &preset, &config.ai_postprocess);

        self.status = None;
        self.error = None;
        *self.job.lock().expect("job") = Some(Job {
            source: source.clone(),
            kind,
            phase: JobPhase::Running,
        });

        let slot = Arc::clone(&self.job);
        let worker_ctx = ctx.clone();
        std::thread::spawn(move || {
            let result = refine(&request);
            if let Some(job) = slot.lock().expect("job").as_mut() {
                job.phase = match result {
                    Ok(path) => JobPhase::Done(path),
                    Err(e) => JobPhase::Error(e),
                };
            }
            worker_ctx.request_repaint();
        });
        ctx.request_repaint();
    }
}

impl Default for HistoryPage {
    fn default() -> Self {
        Self::new()
    }
}

/// Open the containing folder and select the file.
pub(super) fn reveal(path: &std::path::Path) {
    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("open")
            .arg("-R")
            .arg(path)
            .spawn();
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = opener::open(path.parent().unwrap_or(path));
    }
}
