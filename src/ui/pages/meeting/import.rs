//! "Aufnahme importieren": drop a file (or pick one) and get a transcript
//! plus — if a provider is configured — a summary, both as their own files.
//!
//! The import shares the transcription engine with live recording, so the
//! two are mutually exclusive; the page disables whichever is not running.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::audio::import::{run_import, ImportPhase, SummarySpec};
use crate::config::Config;
use crate::postprocess::llm::LlmConfig;
use crate::transcription::engine::Transcriber;
use crate::transcription::model_manager::ModelManager;

/// Files the drop target and the file dialog accept.
pub const IMPORT_EXTENSIONS: &[&str] = &[
    "m4a", "mp3", "aac", "flac", "ogg", "oga", "wav", "wave", "opus", "aiff", "aif", // audio
    "mp4", "mov", "m4v", "mkv", "webm", "avi", "wmv", "flv", "mpg", "mpeg", // video
];

/// Progress of a "drop in a recording" job, shared with its worker thread.
#[derive(Clone, Default)]
pub struct ImportJob {
    phase: Arc<Mutex<Option<ImportPhase>>>,
}

impl ImportJob {
    pub fn phase(&self) -> Option<ImportPhase> {
        self.phase.lock().expect("import phase").clone()
    }

    pub fn running(&self) -> bool {
        matches!(
            self.phase(),
            Some(ImportPhase::Decoding)
                | Some(ImportPhase::Transcribing { .. })
                | Some(ImportPhase::Summarizing)
        )
    }

    pub fn clear(&self) {
        *self.phase.lock().expect("import phase") = None;
    }

    /// Transcribe `path` on a background thread.
    pub fn start(
        &self,
        path: PathBuf,
        config: &Config,
        engine: Arc<Mutex<Box<dyn Transcriber>>>,
        ctx: &egui::Context,
    ) {
        *self.phase.lock().expect("import phase") = Some(ImportPhase::Decoding);

        let vad_model = ModelManager::new(Config::models_dir(config)).vad_model_path();
        let chunk_seconds = config.meeting.chunk_duration_seconds.max(1);
        let language = config.meeting.language.clone();
        let transcripts_dir = Config::transcripts_dir(config);
        let summary = summary_spec(config);
        let phase = Arc::clone(&self.phase);
        let worker_ctx = ctx.clone();

        std::thread::spawn(move || {
            let result = run_import(
                &path,
                &engine,
                &vad_model,
                chunk_seconds,
                &language,
                &transcripts_dir,
                summary,
                &phase,
            );
            if let Err(e) = result {
                *phase.lock().expect("import phase") = Some(ImportPhase::Error(e));
            }
            worker_ctx.request_repaint();
        });
        ctx.request_repaint();
    }
}

/// Build the summary job from the AI section, or `None` when no provider is
/// configured — the importer still writes the transcript in that case.
fn summary_spec(config: &Config) -> Option<SummarySpec> {
    let ai = &config.ai_postprocess;
    if ai.provider == "none" {
        return None;
    }
    Some(SummarySpec {
        llm: LlmConfig::from_ai_section(ai),
        preset: "summary".into(),
        custom_prompt: ai.custom_prompt.clone(),
        overrides: ai
            .prompts
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect(),
    })
}

/// `true` while the user is dragging a file over the window.
pub fn hovering_file(ctx: &egui::Context) -> bool {
    ctx.input(|i| {
        i.raw
            .hovered_files
            .iter()
            .any(|f| f.path.as_deref().map(has_import_ext).unwrap_or(true))
    })
}

/// The first importable file dropped this frame.
pub fn first_dropped(ctx: &egui::Context) -> Option<PathBuf> {
    ctx.input(|i| {
        i.raw.dropped_files.iter().find_map(|f| {
            let p = f.path.clone()?;
            has_import_ext(&p).then_some(p)
        })
    })
}

pub fn has_import_ext(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
        .is_some_and(|e| IMPORT_EXTENSIONS.contains(&e.as_str()))
}

/// Native file picker filtered to importable audio/video files.
pub fn pick_file() -> Option<PathBuf> {
    rfd::FileDialog::new()
        .add_filter("Audio / Video", IMPORT_EXTENSIONS)
        .pick_file()
}
