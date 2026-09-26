//! The second pass, from the UI's side.
//!
//! Structurally the twin of [`super::import::ImportJob`]: a phase behind a
//! mutex, a worker thread that fills it, and a page that polls. The work
//! itself lives in [`crate::transcription::post`]; nothing here knows how a
//! model is run.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::audio::capture::Track;
use crate::config::Config;
use crate::transcription::diarize::DiarizeSpec;
use crate::transcription::model_manager::ModelManager;
use crate::transcription::post::{run_post, PostPhase, PostSpec, ReconcileSpec};

/// A post-transcription in flight, shared with its worker thread.
#[derive(Clone, Default)]
pub struct PostJob {
    phase: Arc<Mutex<Option<PostPhase>>>,
}

impl PostJob {
    pub fn phase(&self) -> Option<PostPhase> {
        self.phase.lock().expect("post phase").clone()
    }

    pub fn running(&self) -> bool {
        matches!(
            self.phase(),
            Some(PostPhase::Preparing)
                | Some(PostPhase::Diarizing)
                | Some(PostPhase::Transcribing(_))
                | Some(PostPhase::Reconciling)
        )
    }

    pub fn clear(&self) {
        *self.phase.lock().expect("post phase") = None;
    }

    /// Kick the pass off on a background thread.
    ///
    /// Deliberately not gated on the meeting page staying open: the pass takes
    /// minutes and its results are files, so navigating away must not cancel
    /// it. The history page picks them up on its next refresh either way.
    pub fn start(
        &self,
        transcript: PathBuf,
        tracks: Vec<(Track, PathBuf)>,
        config: &Config,
        ctx: &egui::Context,
    ) {
        if tracks.is_empty() {
            // Enabled but nothing to work with. Saying so beats a feature that
            // silently does nothing — the usual cause is a failed track write,
            // which is already in the log.
            *self.phase.lock().expect("post phase") = Some(PostPhase::Error(
                "Zu dieser Aufnahme wurde kein Audio gespeichert.".into(),
            ));
            ctx.request_repaint();
            return;
        }
        let spec = spec_from_config(transcript, tracks, config);
        *self.phase.lock().expect("post phase") = Some(PostPhase::Preparing);

        let phase = Arc::clone(&self.phase);
        let worker_ctx = ctx.clone();
        let keep_audio = config.meeting.post_transcribe.keep_audio;
        let models_dir = Config::models_dir(config);

        std::thread::spawn(move || {
            if spec.diarize.is_some() {
                ensure_diarize_models(&models_dir);
            }
            if let Err(e) = run_post(&spec, &phase) {
                log::warn!("post: {e}");
                *phase.lock().expect("post phase") = Some(PostPhase::Error(e));
            }
            // Only once the pass is over — deleting the audio before it ran
            // would leave a recording that can never be re-transcribed.
            if !keep_audio {
                crate::transcription::post::remove_tracks(&spec.tracks);
            }
            worker_ctx.request_repaint();
        });
        ctx.request_repaint();
    }
}

/// Assemble the job from the config. Shared with the history page so a re-run
/// cannot drift from what runs automatically after a meeting.
pub fn spec_from_config(
    transcript: PathBuf,
    tracks: Vec<(Track, PathBuf)>,
    config: &Config,
) -> PostSpec {
    let post = &config.meeting.post_transcribe;
    let ai = &config.ai_postprocess;
    // Merging needs a provider. Without one the pass still runs and the best
    // single variant becomes the result — worth saying, not worth blocking on.
    let reconcile = (post.reconcile && ai.provider != "none").then(|| ReconcileSpec {
        llm: crate::postprocess::llm::LlmConfig::from_ai_section(ai),
        custom_prompt: ai.custom_prompt.clone(),
        overrides: ai
            .prompts
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect(),
    });

    PostSpec {
        transcript,
        tracks,
        models: post.models.clone(),
        models_dir: Config::models_dir(config),
        vad_model: ModelManager::new(Config::models_dir(config)).vad_model_path(),
        chunk_seconds: config.meeting.chunk_duration_seconds.max(1),
        language: config.meeting.language.clone(),
        label_speakers: config.meeting.label_speakers,
        diarize: diarize_spec(config),
        reconcile,
    }
}

/// The diarization job, or `None` when it is switched off — or when the lines
/// carry no speaker at all, in which case working out who spoke would be work
/// nobody sees.
///
/// Not gated on the models being present: they are fetched by
/// [`ensure_diarize_models`] when the pass starts, and a pass that finds them
/// missing anyway says so instead of quietly dropping the names.
pub fn diarize_spec(config: &Config) -> Option<DiarizeSpec> {
    let diarize = &config.meeting.diarize;
    if !diarize.enabled || !config.meeting.label_speakers {
        return None;
    }
    let paths = ModelManager::new(Config::models_dir(config)).diarize_paths();
    Some(DiarizeSpec {
        segmentation: paths.segmentation,
        embedding: paths.embedding,
        speakers: diarize.speakers,
        threshold: diarize.threshold,
        threads: diarize_threads(),
    })
}

/// Diarization runs before the transcription models are loaded, so for those
/// few minutes it can have the machine. Two cores are left alone so the UI
/// keeps painting.
fn diarize_threads() -> i32 {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    cores.saturating_sub(2).clamp(2, 8) as i32
}

/// Fetch the two diarization models if they are missing. Blocking (~34 MB),
/// and called from a worker that is about to spend minutes transcribing
/// anyway. A failure is logged and left to the pass itself to report — the
/// transcript is worth producing either way.
pub fn ensure_diarize_models(models_dir: &std::path::Path) {
    if cfg!(not(feature = "sherpa")) {
        // Nothing in this build can run them, and the pass says so in its
        // note. Downloading 34 MB to prove it would be rude.
        log::info!("diarize: skipping model download — this build has no sherpa backend");
        return;
    }
    let manager = ModelManager::new(models_dir);
    if manager.is_diarize_downloaded() {
        return;
    }
    match manager.ensure_diarize_models(None) {
        Ok(paths) => log::info!(
            "diarize: models ready ({}, {})",
            paths.segmentation.display(),
            paths.embedding.display()
        ),
        Err(e) => log::warn!("diarize: model download failed: {e}"),
    }
}

/// The track files belonging to a transcript that are still on disk.
pub fn existing_tracks(transcript: &std::path::Path, config: &Config) -> Vec<(Track, PathBuf)> {
    let Some(stem) = transcript.file_stem().and_then(|s| s.to_str()) else {
        return Vec::new();
    };
    let dir = Config::audio_dir(config);
    [Track::Mic, Track::System]
        .into_iter()
        .map(|t| (t, crate::audio::track_writer::track_path(&dir, stem, t)))
        .filter(|(_, p)| p.is_file())
        .collect()
}
