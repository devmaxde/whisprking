//! Second-pass transcription: re-decode a finished recording, properly.
//!
//! The live meeting transcript is produced under a hard constraint — a span
//! cannot be transcribed before it has been spoken — and pays for it twice.
//! Every span is decoded alone, so the model never sees the sentence it is in
//! the middle of; and the model has to be fast enough to keep up, which rules
//! out the accurate ones.
//!
//! Once the meeting is over neither constraint applies. This module takes the
//! per-track WAVs [`crate::audio::track_writer`] kept during capture and runs
//! them again:
//!
//! * **Bigger windows.** Spans are glued back together by
//!   [`crate::transcription::pack`] into the longest buffer each backend can
//!   use, cutting only where the VAD found silence. Nothing is decoded
//!   mid-word, and a sentence spanning two utterances is decoded as one.
//! * **Several models at once.** Each configured model gets its own thread and
//!   its own engine, so whisper (Metal) and Parakeet (CPU ONNX) genuinely run
//!   in parallel rather than taking turns. Every model transcribes the whole
//!   recording.
//! * **Reconciliation.** The variants go to the LLM together, in batches
//!   aligned on the recording clock, and it produces one transcript from the
//!   places they agree and disagree.
//! * **Speaker names.** Each track is diarized once
//!   ([`crate::transcription::diarize`]) and every line is attributed to the
//!   voice that spoke over it, so a call with four people on the "Andere"
//!   track reads as Person 1 … Person 4 instead of one collective "Andere".
//!   That, too, is only possible here: the voices are found by clustering the
//!   whole recording, which cannot be done while it is still running.
//!
//! Nothing here ever writes to the live transcript. Results land in
//! `<stem>.post.md` and `<stem>.post-variants.md`, the same
//! one-document-per-file rule the rest of [`crate::output::transcript_doc`]
//! follows.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::audio::capture::Track;
use crate::audio::track_writer::read_track;
use crate::audio::vad::{Segmenter, SpeechSpan};
use crate::output::transcript_doc::{self, DocKind};
use crate::output::transcript_meta::TranscriptMeta;
use crate::postprocess::llm::{make_provider, resolve_system_prompt, LlmConfig};
use crate::transcription::context::{carry_context, language_override};
use crate::transcription::diarize::{self, DiarizeSpec, Diarizer, SpeakerNames, SpeakerSpan};
use crate::transcription::engine::{build_transcriber, DecodeContext, SAMPLE_RATE};
use crate::transcription::model_manager::ModelManager;
use crate::transcription::pack;

/// How much decoded audio to hand the segmenter per step.
const FEED_SAMPLES: usize = SAMPLE_RATE as usize;

/// Roughly how much variant text goes into one reconciliation request.
///
/// The whole recording in one call would be simpler and would fail on every
/// model with a modest context window — and a three-hour meeting fails on all
/// of them. Batching keeps each request small enough to succeed and to be
/// retried cheaply.
const RECONCILE_BATCH_CHARS: usize = 6_000;

/// Reconciled output shorter than this fraction of the variant it was built
/// from is treated as a failed batch, not as a concise one. A model that
/// answers "…" or summarises instead of merging must not be allowed to delete
/// minutes of transcript.
const RECONCILE_MIN_RATIO: f64 = 0.4;

// =====================================================================
// Inputs
// =====================================================================

/// Everything the pass needs. Assembled by the caller from the config.
#[derive(Debug, Clone)]
pub struct PostSpec {
    /// The live transcript. Read for its title, never written to.
    pub transcript: PathBuf,
    /// Captured audio, one entry per track that recorded anything.
    pub tracks: Vec<(Track, PathBuf)>,
    /// Model ids to run, in preference order. The first one that succeeds is
    /// the fallback if reconciliation cannot run.
    pub models: Vec<String>,
    pub models_dir: PathBuf,
    pub vad_model: PathBuf,
    /// Fixed-window size for the segmenter when VAD is unavailable.
    pub chunk_seconds: u32,
    pub language: String,
    pub label_speakers: bool,
    /// `None` keeps the track labels ("Du" / "Andere"). With a spec, each
    /// track is split into the individual voices in it and the lines are
    /// named after those instead.
    pub diarize: Option<DiarizeSpec>,
    /// `None` disables reconciliation; the best single variant is then
    /// promoted to the post transcript.
    pub reconcile: Option<ReconcileSpec>,
}

#[derive(Debug, Clone)]
pub struct ReconcileSpec {
    pub llm: LlmConfig,
    pub custom_prompt: String,
    pub overrides: HashMap<String, String>,
}

// =====================================================================
// Progress
// =====================================================================

#[derive(Debug, Clone, PartialEq)]
pub enum RunState {
    Loading,
    Running,
    Done { seconds: f64 },
    Failed(String),
}

/// What one model is doing, polled by the UI.
#[derive(Debug, Clone)]
pub struct ModelProgress {
    pub model: String,
    pub backend: &'static str,
    /// Longest single buffer this backend takes, in seconds — the number that
    /// makes the difference between the passes visible.
    pub window_seconds: f64,
    pub done_secs: f64,
    pub total_secs: f64,
    pub state: RunState,
}

impl ModelProgress {
    pub fn fraction(&self) -> f32 {
        if self.total_secs <= 0.0 {
            return 0.0;
        }
        (self.done_secs / self.total_secs).clamp(0.0, 1.0) as f32
    }
}

#[derive(Debug, Clone)]
pub enum PostPhase {
    /// Reading the tracks back and finding speech boundaries.
    Preparing,
    /// Working out how many people are on the recording, and where each of
    /// them speaks.
    Diarizing,
    Transcribing(Vec<ModelProgress>),
    Reconciling,
    Done {
        /// The reconciled (or best single) transcript.
        post: PathBuf,
        /// Every model's transcript side by side, when more than one ran.
        variants: Option<PathBuf>,
        /// Why the result is less than it could have been, when that applies.
        note: Option<String>,
    },
    Error(String),
}

type Progress = Arc<Mutex<Option<PostPhase>>>;

fn set(progress: &Progress, phase: PostPhase) {
    *progress.lock().expect("post progress") = Some(phase);
}

// =====================================================================
// One model's result
// =====================================================================

/// One transcript line on the recording clock.
#[derive(Debug, Clone)]
pub struct Line {
    pub elapsed: f64,
    pub track: Track,
    /// Which voice said it, once the recording has been diarized. `None`
    /// falls back to the track's own label, which is what every line had
    /// before diarization existed.
    pub speaker: Option<String>,
    pub text: String,
}

struct ModelRun {
    model: String,
    backend: &'static str,
    window_seconds: f64,
    lines: Vec<Line>,
    took: f64,
}

// =====================================================================
// Entry point
// =====================================================================

/// Run the pass. Blocking — call it on a background thread. `progress` is
/// updated as it goes and ends on [`PostPhase::Done`] or
/// [`PostPhase::Error`].
pub fn run_post(spec: &PostSpec, progress: &Progress) -> Result<PathBuf, String> {
    set(progress, PostPhase::Preparing);

    let models = usable_models(spec)?;

    // Neither segmentation nor diarization depends on the model, so both
    // happen once and every engine works from the same spans.
    let prepared = prepare_tracks(spec, progress)?;
    let spans = prepared.spans;
    let total_secs: f64 = spans
        .iter()
        .flat_map(|(_, s)| s.iter())
        .map(|s| s.audio.len() as f64 / SAMPLE_RATE as f64)
        .sum();
    if total_secs <= 0.0 {
        return Err("In der Aufnahme wurde keine Sprache gefunden.".into());
    }
    log::info!(
        "post: {:.1}s of speech across {} track(s), models: {}",
        total_secs,
        spans.len(),
        models.join(", ")
    );

    let mut runs = transcribe_all(spec, &models, Arc::new(spans), total_secs, progress);
    if runs.is_empty() {
        return Err("Kein Modell konnte die Aufnahme transkribieren.".into());
    }
    for run in &mut runs {
        label_lines(run, &prepared.voices, &prepared.names);
    }

    let title = read_title(&spec.transcript);
    let variants = (runs.len() > 1)
        .then(|| write_variants(spec, &title, &runs))
        .transpose()?;

    let (body, note, mut model_label) = reconcile_or_pick(spec, &runs, progress);
    if !prepared.names.is_empty() {
        model_label.push_str(" + Sprechererkennung");
    }
    let post = write_post(spec, &title, &body, &model_label)?;

    record_meta(spec, variants.is_some(), &model_label);

    set(
        progress,
        PostPhase::Done {
            post: post.clone(),
            variants,
            note: join_notes(prepared.note, note),
        },
    );
    Ok(post)
}

// =====================================================================
// Model selection
// =====================================================================

/// Keep the models that can actually run here: present on disk and, for
/// sherpa models, compiled in. Substituting silently would be worse than
/// saying so, so the caller surfaces what was dropped.
fn usable_models(spec: &PostSpec) -> Result<Vec<String>, String> {
    let manager = ModelManager::new(&spec.models_dir);
    let mut usable = Vec::new();
    for model in &spec.models {
        if !manager.is_downloaded(model) {
            log::warn!("post: skipping {model} — not downloaded");
            continue;
        }
        if cfg!(not(feature = "sherpa")) && is_sherpa(&manager, model) {
            log::warn!("post: skipping {model} — this build has no sherpa backend");
            continue;
        }
        if !usable.contains(model) {
            usable.push(model.clone());
        }
    }
    if usable.is_empty() {
        return Err(format!(
            "Keines der Nachbearbeitungs-Modelle ist verfügbar ({}). \
             Unter Einstellungen → Modelle herunterladen.",
            spec.models.join(", ")
        ));
    }
    Ok(usable)
}

fn is_sherpa(manager: &ModelManager, model: &str) -> bool {
    use crate::transcription::model_manager::Backend;
    manager
        .spec(model)
        .map(|s| matches!(s.backend, Backend::SherpaTransducer { .. }))
        .unwrap_or(false)
}

// =====================================================================
// Segmentation and diarization
// =====================================================================

/// Everything that is derived from the audio alone, before any model runs.
struct Prepared {
    /// Per track, the spans of speech to transcribe.
    spans: Vec<(Track, Vec<SpeechSpan>)>,
    /// Per track, who was speaking when. Empty unless the recording was
    /// diarized.
    voices: Vec<(Track, Vec<SpeakerSpan>)>,
    /// The name each voice gets in the transcript.
    names: SpeakerNames,
    /// Why there are no speaker names, when they were asked for.
    note: Option<String>,
}

/// Read every track back once and get both things out of it: the speech spans
/// to transcribe, and — when asked for — who is speaking in them.
///
/// One read per track, deliberately: the WAVs are around 115 MB per hour, and
/// segmenting and diarizing from the same buffer costs nothing extra.
fn prepare_tracks(spec: &PostSpec, progress: &Progress) -> Result<Prepared, String> {
    let (diarizer, mut note) = load_diarizer(spec);

    let mut spans_out = Vec::new();
    let mut voices = Vec::new();
    for (track, path) in &spec.tracks {
        let samples = read_track(path).map_err(|e| e.to_string())?;
        let mut segmenter = Segmenter::new(Some(&spec.vad_model), spec.chunk_seconds);
        let mut spans = Vec::new();
        for frame in samples.chunks(FEED_SAMPLES) {
            spans.extend(segmenter.push(frame));
        }
        spans.extend(segmenter.finish());
        log::info!(
            "post: {} → {} spans from {:.1}s ({})",
            path.display(),
            spans.len(),
            samples.len() as f64 / SAMPLE_RATE as f64,
            if segmenter.uses_vad() {
                "VAD"
            } else {
                "fixed windows"
            },
        );
        if spans.is_empty() {
            continue;
        }

        if let Some(diarizer) = diarizer.as_ref() {
            set(progress, PostPhase::Diarizing);
            // A fixed count is a count for the call, and the call is on the
            // system track. The microphone carries you plus whoever is in the
            // room, so forcing the same number onto it would invent people.
            diarizer.set_speakers(match track {
                Track::Mic if spec.tracks.len() > 1 => 0,
                _ => spec.diarize.as_ref().map(|d| d.speakers).unwrap_or(0),
            });
            match diarizer.run(&samples, SAMPLE_RATE) {
                Ok(found) if !found.is_empty() => voices.push((*track, found)),
                Ok(_) => log::info!("post: no voices found on {:?}", track),
                Err(e) => {
                    log::warn!("post: diarization of {track:?} failed: {e}");
                    note.get_or_insert_with(|| {
                        format!("Die Sprecher konnten nicht unterschieden werden ({e}).")
                    });
                },
            }
        }

        spans_out.push((*track, spans));
    }
    if spans_out.is_empty() {
        return Err("Die Aufnahme enthält keine Tonspur mit Sprache.".into());
    }

    let names = SpeakerNames::assign(&voices);
    if names.count() > 0 {
        log::info!("post: {} voice(s) across the recording", names.count());
    }
    Ok(Prepared {
        spans: spans_out,
        voices,
        names,
        note,
    })
}

/// Load the diarization models, or explain why the transcript will keep the
/// track labels. Never fails the pass: a transcript that says "Andere" is a
/// great deal better than no transcript.
fn load_diarizer(spec: &PostSpec) -> (Option<Diarizer>, Option<String>) {
    let Some(diarize) = spec.diarize.as_ref() else {
        return (None, None);
    };
    match Diarizer::load(diarize) {
        Ok(d) => (Some(d), None),
        Err(e) => {
            log::warn!("post: diarization unavailable: {e}");
            (
                None,
                Some(format!(
                    "Die Sprecher wurden nicht unterschieden ({e}) — die Zeilen \
                     tragen weiter „Du“ und „Andere“."
                )),
            )
        },
    }
}

/// Attribute every line to the voice that spoke over it.
///
/// A line has a start but no end — the transcriber reports when a piece began,
/// not how long it ran — so the line is taken to last until the next line on
/// the same track ([`diarize::line_end`]) and goes to whichever voice covers
/// most of that stretch.
fn label_lines(run: &mut ModelRun, voices: &[(Track, Vec<SpeakerSpan>)], names: &SpeakerNames) {
    if voices.is_empty() {
        return;
    }
    // Where the next line on the same track starts, per line. Walked
    // backwards so each track's successor is known in one pass; the lines are
    // interleaved across tracks and sorted by time.
    let mut next: Vec<Option<f64>> = vec![None; run.lines.len()];
    let mut following: Vec<(Track, f64)> = Vec::new();
    for i in (0..run.lines.len()).rev() {
        let (track, elapsed) = (run.lines[i].track, run.lines[i].elapsed);
        match following.iter_mut().find(|(t, _)| *t == track) {
            Some(entry) => {
                next[i] = Some(entry.1);
                entry.1 = elapsed;
            },
            None => following.push((track, elapsed)),
        }
    }

    for (i, line) in run.lines.iter_mut().enumerate() {
        let Some((_, spans)) = voices.iter().find(|(t, _)| *t == line.track) else {
            continue;
        };
        let end = diarize::line_end(line.elapsed, next[i]);
        line.speaker = diarize::speaker_at(spans, line.elapsed, end)
            .and_then(|s| names.label(line.track, s))
            .map(str::to_string);
    }
}

/// Both halves of "why is this less than you asked for", in one paragraph.
fn join_notes(a: Option<String>, b: Option<String>) -> Option<String> {
    match (a, b) {
        (Some(a), Some(b)) => Some(format!("{a} {b}")),
        (Some(one), None) | (None, Some(one)) => Some(one),
        (None, None) => None,
    }
}

// =====================================================================
// Transcription
// =====================================================================

/// Run every model over the whole recording, all at the same time.
fn transcribe_all(
    spec: &PostSpec,
    models: &[String],
    spans: Arc<Vec<(Track, Vec<SpeechSpan>)>>,
    total_secs: f64,
    progress: &Progress,
) -> Vec<ModelRun> {
    let threads = threads_per_model(models.len());
    let shared: Arc<Mutex<Vec<ModelProgress>>> = Arc::new(Mutex::new(
        models
            .iter()
            .map(|m| ModelProgress {
                model: m.clone(),
                backend: "…",
                window_seconds: 0.0,
                done_secs: 0.0,
                total_secs,
                state: RunState::Loading,
            })
            .collect(),
    ));
    publish(&shared, progress);

    let runs: Vec<Option<ModelRun>> = std::thread::scope(|scope| {
        let handles: Vec<_> = models
            .iter()
            .enumerate()
            .map(|(idx, model)| {
                let spans = Arc::clone(&spans);
                let shared = Arc::clone(&shared);
                scope.spawn(move || {
                    run_one_model(spec, model, idx, threads, &spans, &shared, progress)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap_or(None))
            .collect()
    });

    runs.into_iter().flatten().collect()
}

/// Both engines are loaded at once, so neither gets the whole machine. Whisper
/// leans on Metal and Parakeet on CPU ONNX, but oversubscribing still costs
/// more than it buys.
fn threads_per_model(models: usize) -> i32 {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    (cores.saturating_sub(1) / models.max(1)).clamp(2, 8) as i32
}

fn run_one_model(
    spec: &PostSpec,
    model: &str,
    idx: usize,
    threads: i32,
    spans: &[(Track, Vec<SpeechSpan>)],
    shared: &Arc<Mutex<Vec<ModelProgress>>>,
    progress: &Progress,
) -> Option<ModelRun> {
    let started = Instant::now();
    let engine = match build_transcriber(model, &spec.models_dir, &spec.language, threads) {
        Ok(e) => e,
        Err(e) => {
            log::error!("post: {model} failed to load: {e}");
            update(shared, idx, |p| p.state = RunState::Failed(e.to_string()));
            publish(shared, progress);
            return None;
        },
    };

    let window_seconds = engine.max_input_seconds();
    let backend = engine.backend_label();
    update(shared, idx, |p| {
        p.state = RunState::Running;
        p.backend = backend;
        p.window_seconds = window_seconds;
    });
    publish(shared, progress);
    log::info!("post: {model} ({backend}) running with {window_seconds:.0}s windows");

    let lang = language_override(&spec.language);
    let mut lines = Vec::new();
    let mut done = 0.0f64;

    for (track, track_spans) in spans {
        let windows = pack::pack(track_spans, window_seconds);
        log::info!(
            "post: {model} · {:?}: {} spans → {} windows",
            track,
            track_spans.len(),
            windows.len()
        );
        // Context carries within a track only — the two sides of a call are
        // separate conversations, exactly as in the live path.
        let mut prior = String::new();
        for window in windows {
            let result = engine.transcribe_timed(
                &window.audio,
                SAMPLE_RATE,
                DecodeContext {
                    prior: prior.as_str(),
                    language: lang,
                },
            );
            done += window.duration();
            match result {
                Ok(pieces) => {
                    for piece in &pieces {
                        let text = piece.text.trim();
                        if text.is_empty() {
                            continue;
                        }
                        lines.push(Line {
                            elapsed: window.absolute(piece.offset),
                            track: *track,
                            speaker: None,
                            text: text.to_string(),
                        });
                    }
                    let joined = crate::transcription::engine::join_pieces(&pieces);
                    carry_context(&mut prior, &joined);
                },
                Err(e) => log::warn!("post: {model} window at {:.1}s failed: {e}", window.elapsed),
            }
            update(shared, idx, |p| p.done_secs = done);
            publish(shared, progress);
        }
    }

    lines.sort_by(|a, b| a.elapsed.total_cmp(&b.elapsed));
    let took = started.elapsed().as_secs_f64();
    log::info!(
        "post: {model} produced {} lines in {:.1}s",
        lines.len(),
        took
    );

    update(shared, idx, |p| {
        p.done_secs = p.total_secs;
        p.state = RunState::Done { seconds: took };
    });
    publish(shared, progress);

    if lines.is_empty() {
        return None;
    }
    Some(ModelRun {
        model: model.to_string(),
        backend,
        window_seconds,
        lines,
        took,
    })
}

fn update(
    shared: &Arc<Mutex<Vec<ModelProgress>>>,
    idx: usize,
    edit: impl FnOnce(&mut ModelProgress),
) {
    if let Ok(mut guard) = shared.lock() {
        if let Some(entry) = guard.get_mut(idx) {
            edit(entry);
        }
    }
}

fn publish(shared: &Arc<Mutex<Vec<ModelProgress>>>, progress: &Progress) {
    let snapshot = shared.lock().map(|g| g.clone()).unwrap_or_default();
    set(progress, PostPhase::Transcribing(snapshot));
}

// =====================================================================
// Documents
// =====================================================================

pub fn fmt_mmss(seconds: f64) -> String {
    let s = seconds.max(0.0) as u64;
    format!("{:02}:{:02}", s / 60, s % 60)
}

/// One transcript line in the same shape the live writer uses.
///
/// The name is the diarized voice when there is one and the track's own label
/// otherwise — so a recording that was not (or could not be) diarized reads
/// exactly as it did before.
fn fmt_line(line: &Line, label_speakers: bool) -> String {
    if label_speakers {
        format!(
            "**[{}] {}:** {}",
            fmt_mmss(line.elapsed),
            line.speaker.as_deref().unwrap_or(line.track.label()),
            line.text
        )
    } else {
        format!("**[{}]** {}", fmt_mmss(line.elapsed), line.text)
    }
}

fn render(lines: &[Line], label_speakers: bool) -> String {
    lines
        .iter()
        .map(|l| fmt_line(l, label_speakers))
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn read_title(transcript: &Path) -> String {
    std::fs::read_to_string(transcript)
        .map(|raw| transcript_doc::title_of(&raw))
        .unwrap_or_else(|_| "Meeting".to_string())
}

fn stamp() -> String {
    chrono::Local::now().format("%d.%m.%Y %H:%M").to_string()
}

/// Every model's transcript in one document, so the two can be read against
/// each other without opening two files.
fn write_variants(spec: &PostSpec, title: &str, runs: &[ModelRun]) -> Result<PathBuf, String> {
    let mut body = String::new();
    for run in runs {
        body.push_str(&format!(
            "## {} · {} · {:.0}s-Fenster · {:.0}s Laufzeit\n\n",
            run.model, run.backend, run.window_seconds, run.took
        ));
        body.push_str(&render(&run.lines, spec.label_speakers));
        body.push_str("\n\n");
    }

    let models = runs
        .iter()
        .map(|r| r.model.as_str())
        .collect::<Vec<_>>()
        .join(" · ");
    let header = transcript_doc::doc_header(
        DocKind::PostVariants.label(),
        title,
        &stamp(),
        &models,
        &spec.transcript,
    );
    let path = transcript_doc::doc_path(&spec.transcript, DocKind::PostVariants);
    std::fs::write(&path, format!("{header}{body}"))
        .map_err(|e| format!("{}: {e}", path.display()))?;
    log::info!("post: wrote {}", path.display());
    Ok(path)
}

fn write_post(
    spec: &PostSpec,
    title: &str,
    body: &str,
    model_label: &str,
) -> Result<PathBuf, String> {
    let header = transcript_doc::doc_header(
        DocKind::PostTranscript.label(),
        title,
        &stamp(),
        model_label,
        &spec.transcript,
    );
    let path = transcript_doc::doc_path(&spec.transcript, DocKind::PostTranscript);
    std::fs::write(&path, format!("{header}{}\n", body.trim_end()))
        .map_err(|e| format!("{}: {e}", path.display()))?;
    log::info!("post: wrote {}", path.display());
    Ok(path)
}

fn record_meta(spec: &PostSpec, wrote_variants: bool, model_label: &str) {
    let at = stamp();
    let mut meta = TranscriptMeta::load(&spec.transcript);
    meta.record_doc(DocKind::PostTranscript.preset_key(), &at, model_label, "local");
    if wrote_variants {
        meta.record_doc(DocKind::PostVariants.preset_key(), &at, model_label, "local");
    }
    if let Err(e) = meta.save(&spec.transcript) {
        log::warn!("post: could not update sidecar: {e}");
    }
}

/// Delete a recording's kept audio. Used when the pass is over and the user
/// does not want it, and when the recording itself is deleted.
pub fn remove_tracks(tracks: &[(Track, PathBuf)]) {
    for (_, path) in tracks {
        if let Err(e) = std::fs::remove_file(path) {
            if e.kind() != std::io::ErrorKind::NotFound {
                log::warn!("post: could not remove {}: {e}", path.display());
            }
        }
    }
}

// =====================================================================
// Reconciliation
// =====================================================================

/// Merge the variants with the LLM, or fall back to the best single one.
///
/// Returns the transcript body, a note for the UI when the result is not what
/// was asked for, and the label recorded as the producing model.
fn reconcile_or_pick(
    spec: &PostSpec,
    runs: &[ModelRun],
    progress: &Progress,
) -> (String, Option<String>, String) {
    let primary = &runs[0];
    let fallback = || render(&primary.lines, spec.label_speakers);

    if runs.len() < 2 {
        let note = (spec.models.len() > 1).then(|| {
            format!(
                "Nur {} lief — die anderen Modelle waren nicht verfügbar, \
                 also gab es nichts zu vergleichen.",
                primary.model
            )
        });
        return (fallback(), note, primary.model.clone());
    }

    let Some(rc) = spec.reconcile.as_ref() else {
        return (
            fallback(),
            Some(format!(
                "Kein KI-Provider aktiv — die Varianten wurden nicht zusammengeführt. \
                 Übernommen wurde {}.",
                primary.model
            )),
            primary.model.clone(),
        );
    };

    set(progress, PostPhase::Reconciling);
    let label = format!(
        "{} + {}",
        runs.iter()
            .map(|r| r.model.as_str())
            .collect::<Vec<_>>()
            .join(" + "),
        rc.llm.model
    );

    match reconcile(spec, runs, rc) {
        Ok(body) => (body, None, label),
        Err(e) => {
            log::warn!("post: reconciliation failed: {e}");
            (
                fallback(),
                Some(format!(
                    "Zusammenführen fehlgeschlagen ({e}) — übernommen wurde {}. \
                     Beide Varianten stehen im Vergleich.",
                    primary.model
                )),
                primary.model.clone(),
            )
        },
    }
}

fn reconcile(spec: &PostSpec, runs: &[ModelRun], rc: &ReconcileSpec) -> Result<String, String> {
    let provider = make_provider(&rc.llm).map_err(|e| e.to_string())?;
    let system = resolve_system_prompt("reconcile", &rc.custom_prompt, &rc.overrides);
    let batches = batch(runs, spec.label_speakers);
    log::info!(
        "post: reconciling {} batch(es) with {}",
        batches.len(),
        rc.llm.model
    );

    let mut out: Vec<String> = Vec::new();
    let mut failures = 0usize;
    for (i, b) in batches.iter().enumerate() {
        match provider.run(&system, &b.prompt) {
            Ok(text) if keeps_enough(&text, &b.baseline) => out.push(text.trim().to_string()),
            Ok(text) => {
                // A merge that comes back a fraction of the size did not merge;
                // it summarised, or gave up. Keeping the variant is the honest
                // outcome — losing minutes of speech silently is not.
                log::warn!(
                    "post: batch {i} came back {} chars against {} — keeping the variant",
                    text.chars().count(),
                    b.baseline.chars().count()
                );
                out.push(b.baseline.clone());
                failures += 1;
            },
            Err(e) => {
                log::warn!("post: batch {i} failed: {e} — keeping the variant");
                out.push(b.baseline.clone());
                failures += 1;
            },
        }
    }

    if failures == batches.len() {
        return Err("kein Abschnitt konnte zusammengeführt werden".into());
    }
    if failures > 0 {
        log::warn!("post: {failures}/{} batches fell back", batches.len());
    }
    Ok(out.join("\n\n"))
}

fn keeps_enough(candidate: &str, baseline: &str) -> bool {
    let got = candidate.trim().chars().count() as f64;
    let want = baseline.trim().chars().count() as f64;
    got > 0.0 && (want == 0.0 || got / want >= RECONCILE_MIN_RATIO)
}

/// One reconciliation request.
struct Batch {
    prompt: String,
    /// What to keep if the request fails — the primary model's lines for the
    /// same stretch of the recording.
    baseline: String,
}

/// Cut the variants into aligned chunks of the recording.
///
/// Alignment is by timestamp, not by line count: the models disagree about how
/// many lines a stretch of speech is, which is exactly what makes a positional
/// pairing produce nonsense a few minutes in.
fn batch(runs: &[ModelRun], label_speakers: bool) -> Vec<Batch> {
    let end = runs
        .iter()
        .flat_map(|r| r.lines.last())
        .map(|l| l.elapsed)
        .fold(0.0f64, f64::max)
        + 1.0;

    // Step through the recording in fixed slices and grow a batch until it is
    // big enough to be worth a round trip.
    const SLICE: f64 = 30.0;
    let mut batches = Vec::new();
    let mut from = 0.0f64;
    let mut cursor = 0.0f64;

    while cursor < end {
        let to = (cursor + SLICE).min(end);
        let size: usize = runs
            .iter()
            .map(|r| slice_chars(r, from, to, label_speakers))
            .sum();
        cursor = to;
        if size >= RECONCILE_BATCH_CHARS || cursor >= end {
            if let Some(b) = build_batch(runs, from, to, label_speakers) {
                batches.push(b);
            }
            from = to;
        }
    }
    batches
}

fn lines_in(run: &ModelRun, from: f64, to: f64) -> Vec<&Line> {
    run.lines
        .iter()
        .filter(|l| l.elapsed >= from && l.elapsed < to)
        .collect()
}

fn slice_chars(run: &ModelRun, from: f64, to: f64, label_speakers: bool) -> usize {
    lines_in(run, from, to)
        .iter()
        .map(|l| fmt_line(l, label_speakers).chars().count())
        .sum()
}

fn build_batch(runs: &[ModelRun], from: f64, to: f64, label_speakers: bool) -> Option<Batch> {
    let mut prompt = String::new();
    let mut baseline = String::new();

    for (i, run) in runs.iter().enumerate() {
        let lines = lines_in(run, from, to);
        let body = lines
            .iter()
            .map(|l| fmt_line(l, label_speakers))
            .collect::<Vec<_>>()
            .join("\n");
        if i == 0 {
            baseline = body.clone();
        }
        prompt.push_str(&format!(
            "## Variante {} — {} ({})\n\n{}\n\n",
            (b'A' + i as u8) as char,
            run.model,
            run.backend,
            if body.trim().is_empty() {
                "(nichts erkannt)"
            } else {
                &body
            }
        ));
    }

    if baseline.trim().is_empty() && runs.iter().all(|r| lines_in(r, from, to).is_empty()) {
        return None;
    }
    Some(Batch { prompt, baseline })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(elapsed: f64, text: &str) -> Line {
        Line {
            elapsed,
            track: Track::Mic,
            speaker: None,
            text: text.to_string(),
        }
    }

    fn run(model: &str, lines: Vec<Line>) -> ModelRun {
        ModelRun {
            model: model.into(),
            backend: "test",
            window_seconds: 28.0,
            lines,
            took: 1.0,
        }
    }

    #[test]
    fn lines_render_with_and_without_speakers() {
        let l = line(72.0, "Guten Morgen.");
        assert_eq!(fmt_line(&l, true), "**[01:12] Du:** Guten Morgen.");
        assert_eq!(fmt_line(&l, false), "**[01:12]** Guten Morgen.");
    }

    /// The whole point of the diarization pass, end to end: a line that was
    /// spoken by the second voice on the "Andere" track has to come out as
    /// that person rather than as the track.
    #[test]
    fn speaker_names_replace_the_track_label() {
        let voice = |start: f64, end: f64, speaker: u32| SpeakerSpan {
            start,
            end,
            speaker,
        };
        let voices = vec![
            (Track::Mic, vec![voice(0.0, 5.0, 0)]),
            (Track::System, vec![voice(6.0, 9.0, 0), voice(10.0, 20.0, 1)]),
        ];
        let names = SpeakerNames::assign(&voices);

        let mut run = run(
            "whisper",
            vec![
                Line {
                    elapsed: 1.0,
                    track: Track::Mic,
                    speaker: None,
                    text: "Kurze Frage.".into(),
                },
                Line {
                    elapsed: 6.5,
                    track: Track::System,
                    speaker: None,
                    text: "Klar.".into(),
                },
                Line {
                    elapsed: 11.0,
                    track: Track::System,
                    speaker: None,
                    text: "Ich übernehme das.".into(),
                },
            ],
        );
        label_lines(&mut run, &voices, &names);

        assert_eq!(fmt_line(&run.lines[0], true), "**[00:01] Du:** Kurze Frage.");
        assert_eq!(fmt_line(&run.lines[1], true), "**[00:06] Person 1:** Klar.");
        assert_eq!(
            fmt_line(&run.lines[2], true),
            "**[00:11] Person 2:** Ich übernehme das."
        );
        // Names are for the reader; switching labels off still drops them.
        assert_eq!(fmt_line(&run.lines[1], false), "**[00:06]** Klar.");
    }

    /// A recording nobody could diarize has to read exactly as it did before
    /// the feature existed.
    #[test]
    fn without_voices_the_lines_keep_their_track() {
        let mut run = run("whisper", vec![line(3.0, "Guten Morgen.")]);
        label_lines(&mut run, &[], &SpeakerNames::default());
        assert_eq!(fmt_line(&run.lines[0], true), "**[00:03] Du:** Guten Morgen.");
    }

    /// Both variants of the same stretch have to end up in the same request,
    /// or the model has nothing to compare.
    #[test]
    fn a_batch_carries_every_variant_for_its_slice() {
        let runs = vec![
            run("whisper", vec![line(1.0, "hallo welt"), line(200.0, "spät")]),
            run("parakeet", vec![line(1.2, "hallo Welt"), line(201.0, "spaet")]),
        ];
        let batches = batch(&runs, true);
        assert!(!batches.is_empty());
        let all: String = batches.iter().map(|b| b.prompt.clone()).collect();
        assert!(all.contains("hallo welt"));
        assert!(all.contains("hallo Welt"));
        assert!(all.contains("Variante A — whisper"));
        assert!(all.contains("Variante B — parakeet"));
    }

    /// Long recordings must not go out as one request.
    #[test]
    fn long_recordings_are_split_into_several_requests() {
        let long: Vec<Line> = (0..400)
            .map(|i| line(i as f64 * 5.0, "eine ziemlich lange gesprochene Zeile über nichts"))
            .collect();
        let runs = vec![run("a", long.clone()), run("b", long)];
        let batches = batch(&runs, true);
        assert!(batches.len() > 1, "got {} batches", batches.len());
        for b in &batches {
            // The slice granularity means a batch overshoots by at most one
            // 30 s slice worth of text.
            assert!(
                b.prompt.chars().count() < RECONCILE_BATCH_CHARS * 3,
                "batch of {} chars",
                b.prompt.chars().count()
            );
        }
    }

    /// Every line of the primary variant has to appear in exactly one batch —
    /// a gap in the slicing would silently drop speech.
    #[test]
    fn batching_covers_every_line_exactly_once() {
        let lines: Vec<Line> = (0..50).map(|i| line(i as f64 * 7.0, &format!("zeile {i}"))).collect();
        let runs = vec![run("a", lines.clone()), run("b", lines)];
        let joined: String = batch(&runs, true)
            .iter()
            .map(|b| b.baseline.clone())
            .collect::<Vec<_>>()
            .join("\n");
        for i in 0..50 {
            assert_eq!(
                joined.matches(&format!("zeile {i} ")).count()
                    + joined.matches(&format!("zeile {i}\n")).count()
                    + usize::from(joined.ends_with(&format!("zeile {i}"))),
                1,
                "line {i} should appear exactly once"
            );
        }
    }

    #[test]
    fn a_collapsed_merge_is_rejected() {
        let baseline = "a".repeat(1000);
        assert!(keeps_enough(&"b".repeat(900), &baseline));
        assert!(keeps_enough(&"b".repeat(400), &baseline));
        assert!(!keeps_enough(&"b".repeat(100), &baseline), "too short");
        assert!(!keeps_enough("   ", &baseline), "empty is never enough");
    }

    #[test]
    fn threads_are_split_between_models_and_stay_sane() {
        assert!(threads_per_model(1) >= 2);
        assert!(threads_per_model(2) >= 2);
        assert!(threads_per_model(8) >= 2);
        assert!(threads_per_model(1) <= 8);
    }
}
