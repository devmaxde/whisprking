//! Batch import: turn a recording on disk into a transcript + summary.
//!
//! This is the offline twin of the live meeting worker in
//! [`crate::ui::pages::meeting`]. Instead of audio streaming in from two
//! capture tracks, the whole file is decoded up front
//! ([`super::decode::decode_to_mono_16k`]) and pushed through the *same*
//! [`Segmenter`] → [`Transcriber`] → [`TranscriptWriter`] path. A file has
//! no mic/system split, so it is a single, unlabelled track.
//!
//! After transcription it optionally runs the LLM `summary` preset — the
//! "drop in audio, get a summary" flow. The summary is written as its own
//! document next to the transcript ([`crate::postprocess::refine`]); the
//! transcript file itself is never appended to.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::audio::decode::decode_to_mono_16k;
use crate::audio::resample::TARGET_RATE;
use crate::audio::vad::Segmenter;
use crate::output::transcript_writer::TranscriptWriter;
use crate::postprocess::llm::LlmConfig;
use crate::postprocess::refine::{refine, RefineJob};
use crate::transcription::context::{carry_context, language_override};
use crate::transcription::engine::{DecodeContext, Transcriber};

/// How much decoded audio to hand the segmenter per step. Only affects how
/// finely segmentation progresses; the transcription that follows dominates
/// wall-clock either way.
const FEED_SAMPLES: usize = TARGET_RATE as usize; // 1 s

/// Where a running import is. The UI polls this behind a mutex.
#[derive(Debug, Clone)]
pub enum ImportPhase {
    Decoding,
    /// Transcribing, with how far through the audio we are.
    Transcribing {
        done_secs: f64,
        total_secs: f64,
    },
    Summarizing,
    Done {
        transcript: PathBuf,
        /// Why no summary was produced, when that is the case.
        note: Option<String>,
    },
    Error(String),
}

/// Everything needed to auto-run a summary after transcription. Built from
/// the config's AI section by the caller.
#[derive(Debug, Clone)]
pub struct SummarySpec {
    pub llm: LlmConfig,
    /// Preset key to run — the importer passes `"summary"`.
    pub preset: String,
    pub custom_prompt: String,
    pub overrides: HashMap<String, String>,
}

impl SummarySpec {
    fn job(&self, source: &Path) -> RefineJob {
        RefineJob {
            source: source.to_path_buf(),
            preset: self.preset.clone(),
            llm: self.llm.clone(),
            custom_prompt: self.custom_prompt.clone(),
            overrides: self.overrides.clone(),
        }
    }
}

/// Decode → segment → transcribe → write → summarize. Blocking; call it on a
/// background thread. `progress` is updated as each phase advances.
///
/// Returns the transcript path on success. On failure the `Err` string is a
/// user-facing message (and also written into `progress` by the caller).
/// `language` follows the meeting setting: empty means "leave it to the
/// engine", otherwise it pins this import's decode (see
/// [`DecodeContext::language`]).
pub fn run_import(
    path: &Path,
    engine: &Arc<Mutex<Box<dyn Transcriber>>>,
    vad_model: &Path,
    chunk_seconds: u32,
    language: &str,
    transcripts_dir: &Path,
    summary: Option<SummarySpec>,
    progress: &Arc<Mutex<Option<ImportPhase>>>,
) -> Result<PathBuf, String> {
    set(progress, ImportPhase::Decoding);
    let samples = decode_to_mono_16k(path).map_err(|e| e.to_string())?;
    let total_secs = samples.len() as f64 / TARGET_RATE as f64;

    // Segment the whole file. Segmentation is cheap; do it up front so the
    // progress bar can track the (slow) transcription against a known total.
    let mut segmenter = Segmenter::new(Some(vad_model), chunk_seconds);
    let mut spans = Vec::new();
    for frame in samples.chunks(FEED_SAMPLES) {
        spans.extend(segmenter.push(frame));
    }
    spans.extend(segmenter.finish());

    // One transcript file, titled after the source recording.
    let title = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("recording");
    let mut writer = TranscriptWriter::new(transcripts_dir).map_err(|e| e.to_string())?;
    let out_path = writer
        .start_new(Some(title))
        .map_err(|e| e.to_string())?
        .to_path_buf();

    // Transcribe each span, carrying decode context across the boundaries.
    let lang = language_override(language);
    let mut prior = String::new();
    let mut wrote_any = false;
    for span in &spans {
        set(
            progress,
            ImportPhase::Transcribing {
                done_secs: span.elapsed,
                total_secs,
            },
        );
        let text = {
            let guard = engine.lock().expect("engine mutex");
            guard.transcribe_with_context(
                &span.audio,
                TARGET_RATE,
                DecodeContext {
                    prior: &prior,
                    language: lang,
                },
            )
        };
        match text {
            Ok(t) if !t.trim().is_empty() => {
                let t = t.trim().to_string();
                carry_context(&mut prior, &t);
                let _ = writer.add_segment(span.elapsed, &t);
                wrote_any = true;
            }
            Ok(_) => {}
            Err(e) => log::warn!("import transcribe failed: {e}"),
        }
    }

    writer.finalize(total_secs).map_err(|e| e.to_string())?;

    // Summarize (or explain why we couldn't). Either way the transcript is
    // already on disk and worth keeping.
    let note = maybe_summarize(&out_path, wrote_any, summary, progress);

    set(
        progress,
        ImportPhase::Done {
            transcript: out_path.clone(),
            note,
        },
    );
    Ok(out_path)
}

/// Run the summary preset over the finished transcript. Returns a note for
/// the UI when no summary was produced — never fails the whole import, a
/// good transcript with a failed summary is still worth keeping.
fn maybe_summarize(
    transcript: &Path,
    wrote_any: bool,
    summary: Option<SummarySpec>,
    progress: &Arc<Mutex<Option<ImportPhase>>>,
) -> Option<String> {
    if !wrote_any {
        return Some("In dieser Aufnahme wurde keine Sprache erkannt.".into());
    }

    let Some(spec) = summary else {
        return Some(
            "Kein KI-Provider aktiv — es wurde nur das Transkript erzeugt \
             (Einstellungen → KI)."
                .into(),
        );
    };

    set(progress, ImportPhase::Summarizing);
    match refine(&spec.job(transcript)) {
        Ok(path) => {
            log::info!("import: summary written to {}", path.display());
            None
        },
        Err(e) => {
            // The full report is in the log; the import panel has room for a
            // sentence, so it gets the headline plus what to try next.
            log::warn!("import: summary failed: {}", e.report().replace('\n', " | "));
            Some(e.note())
        },
    }
}

fn set(progress: &Arc<Mutex<Option<ImportPhase>>>, phase: ImportPhase) {
    *progress.lock().expect("import progress mutex") = Some(phase);
}
