//! Recording session: capture → segmentation → transcription → file.
//!
//! This is the part of the meeting page that has nothing to do with pixels.
//! Audio arrives as [`Chunk`]s tagged with the [`Track`] they came from
//! (microphone vs. system output), a worker thread turns each into a
//! [`Segment`], and the UI thread appends segments to the live view and
//! streams them to a [`TranscriptWriter`].
//!
//! Because the two tracks are captured and transcribed independently, the
//! transcript gets speaker attribution — "Du" vs. "Andere" — without any
//! diarization pass over the audio.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use crate::audio::capture::{Chunk, MeetingCapture, SourceMode, Track};
use crate::audio::track_writer::{track_path, TrackWriter};
use crate::audio::vad::{Segmenter, SpeechSpan};
use crate::config::Config;
use crate::output::transcript_writer::TranscriptWriter;
use crate::transcription::context::{carry_context, language_override};
use crate::transcription::engine::{DecodeContext, Transcriber};
use crate::transcription::model_manager::ModelManager;

/// How far behind the newest segment we let the writer run before flushing.
/// The two tracks are transcribed concurrently, so a segment from one can
/// arrive after a later segment from the other; holding a little slack lets
/// us emit them in the order they were spoken.
const REORDER_SLACK_SECONDS: f64 = 12.0;

/// One transcribed line with its offset from the start of the meeting.
#[derive(Debug, Clone)]
pub struct Segment {
    pub elapsed: f64,
    pub track: Track,
    pub text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeetingState {
    Idle,
    Recording,
    Paused,
}

/// What a finished meeting left behind.
pub struct StopReport {
    pub transcript: PathBuf,
    /// The audio of each track that recorded something, kept for the
    /// post-transcription pass. Empty when it is switched off.
    pub tracks: Vec<(Track, PathBuf)>,
}

pub struct Session {
    pub state: MeetingState,
    /// Wall-clock length of the meeting, advanced from the audio clock.
    pub duration: f64,
    /// Segments already shown, in spoken order.
    pub transcript: Vec<Segment>,

    capture: MeetingCapture,
    writer: TranscriptWriter,
    engine: Arc<Mutex<Box<dyn Transcriber>>>,
    audio_dir: PathBuf,

    chunk_tx: Option<Sender<Chunk>>,
    segment_rx: Receiver<Segment>,
    segment_tx: Sender<Segment>,
    worker: Option<JoinHandle<()>>,

    /// Filled by the worker as it closes each track file. Read after the
    /// worker is joined, so no lock is held across the meeting.
    tracks: Arc<Mutex<Vec<(Track, PathBuf)>>>,

    /// Segments received but not yet flushed to disk, held back so the file
    /// ends up in spoken order too.
    pending: Vec<Segment>,
}

impl Session {
    pub fn new(
        transcripts_dir: PathBuf,
        audio_dir: PathBuf,
        engine: Arc<Mutex<Box<dyn Transcriber>>>,
    ) -> Result<Self, anyhow::Error> {
        let writer = TranscriptWriter::new(transcripts_dir)?;
        let (segment_tx, segment_rx) = mpsc::channel();
        Ok(Self {
            state: MeetingState::Idle,
            duration: 0.0,
            transcript: Vec::new(),
            capture: MeetingCapture::new(),
            writer,
            engine,
            audio_dir,
            chunk_tx: None,
            segment_rx,
            segment_tx,
            worker: None,
            tracks: Arc::new(Mutex::new(Vec::new())),
            pending: Vec::new(),
        })
    }

    pub fn is_idle(&self) -> bool {
        self.state == MeetingState::Idle
    }

    pub fn path(&self) -> Option<&Path> {
        self.writer.path()
    }

    /// `(mic_active, mic_level, system_active, system_level)`.
    pub fn levels(&self) -> Option<(bool, f32, bool, f32)> {
        let report = self.capture.report()?;
        Some((
            report.mic,
            self.capture.mic_level(),
            report.system,
            self.capture.system_level(),
        ))
    }

    /// Warning the capture backend reported when starting, if any.
    pub fn start_warning(&self) -> Option<String> {
        self.capture.report().and_then(|r| r.warning.clone())
    }

    /// Start recording. `Err` carries a user-facing message.
    pub fn start(&mut self, config: &Config) -> Result<(), String> {
        self.transcript.clear();
        self.pending.clear();
        self.duration = 0.0;
        self.tracks.lock().expect("tracks").clear();

        let transcript = self
            .writer
            .start_new(None)
            .map_err(|e| format!("Transkript-Datei konnte nicht angelegt werden: {e}"))?
            .to_path_buf();

        // Keeping the audio is what makes a second pass possible at all, so
        // the track files are named after the transcript they belong to.
        // `save_audio` is the older "keep a wav too" switch; it now means the
        // same thing, since the per-track files are the wav.
        let wants_audio =
            config.meeting.post_transcribe.enabled || config.meeting.save_audio;
        let keep_audio = wants_audio.then(|| {
            (
                self.audio_dir.clone(),
                transcript
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("meeting")
                    .to_string(),
            )
        });

        // The worker segments each track (VAD when the model is present and
        // this is a `sherpa` build, fixed windows otherwise) before it hands
        // audio to the engine.
        let vad_model = ModelManager::new(Config::models_dir(config)).vad_model_path();
        self.spawn_worker(
            vad_model,
            config.meeting.chunk_duration_seconds.max(1),
            config.meeting.language.clone(),
            keep_audio,
        );
        self.fetch_vad_model(config);
        self.fetch_diarize_models(config);
        let tx = self.chunk_tx.as_ref().expect("worker spawned").clone();

        let result = self.capture.start(
            SourceMode::from_config(&config.meeting.audio_source),
            config.meeting.input_device.clone(),
            // Capture's own mixed buffer stays off: it accumulates the whole
            // meeting in memory and nothing here ever read it back. The
            // per-track WAVs replaced it, and they stream to disk.
            false,
            move |chunk| {
                let _ = tx.send(chunk);
            },
        );

        match result {
            Ok(()) => {
                self.state = MeetingState::Recording;
                Ok(())
            },
            Err(e) => {
                log::error!("meeting start: {e}");
                self.shutdown_worker();
                Err(e.to_string())
            },
        }
    }

    pub fn toggle_pause(&mut self) {
        match self.state {
            MeetingState::Recording => {
                self.capture.set_paused(true);
                self.state = MeetingState::Paused;
            },
            MeetingState::Paused => {
                self.capture.set_paused(false);
                self.state = MeetingState::Recording;
            },
            MeetingState::Idle => {},
        }
    }

    /// Stop and finalize. Returns the transcript path and the audio kept for
    /// the post-transcription pass.
    pub fn stop(&mut self) -> Option<StopReport> {
        self.capture.stop();
        self.state = MeetingState::Idle;

        // The capture thread is gone, so no more chunks are coming; drain
        // whatever the worker still owes us before tearing it down. The worker
        // closes the track files on its way out, so this must come first.
        self.shutdown_worker();
        self.drain();
        self.flush_pending(true);

        let _ = self.writer.finalize(self.duration);
        let transcript = self.writer.path().map(|p| p.to_path_buf())?;
        let tracks = std::mem::take(&mut *self.tracks.lock().expect("tracks"));
        Some(StopReport { transcript, tracks })
    }

    /// Pull finished segments off the worker channel.
    pub fn drain(&mut self) {
        let mut got_any = false;
        while let Ok(seg) = self.segment_rx.try_recv() {
            self.duration = self.duration.max(seg.elapsed);
            self.pending.push(seg);
            got_any = true;
        }
        if got_any {
            self.flush_pending(false);
        }
    }

    /// The live transcript as Markdown, for the clipboard.
    pub fn transcript_markdown(&self, label_speakers: bool) -> String {
        self.transcript
            .iter()
            .map(|s| {
                if label_speakers {
                    format!(
                        "**[{}] {}:** {}",
                        fmt_mmss(s.elapsed),
                        s.track.label(),
                        s.text
                    )
                } else {
                    format!("**[{}]** {}", fmt_mmss(s.elapsed), s.text)
                }
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    fn spawn_worker(
        &mut self,
        vad_model: PathBuf,
        chunk_seconds: u32,
        language: String,
        keep_audio: Option<(PathBuf, String)>,
    ) {
        let (tx, rx) = mpsc::channel::<Chunk>();
        self.chunk_tx = Some(tx);
        let engine = Arc::clone(&self.engine);
        let segment_tx = self.segment_tx.clone();
        let tracks = Arc::clone(&self.tracks);

        self.worker = Some(
            thread::Builder::new()
                .name("whisprking-meeting-worker".into())
                .spawn(move || {
                    // One segmenter per track: the two sides are independent
                    // streams and must not share VAD state.
                    let mut mic_seg = Segmenter::new(Some(vad_model.as_path()), chunk_seconds);
                    let mut sys_seg = Segmenter::new(Some(vad_model.as_path()), chunk_seconds);
                    log::info!(
                        "meeting: segmentation = {}",
                        if mic_seg.uses_vad() {
                            "VAD (speech boundaries)"
                        } else {
                            "fixed windows"
                        }
                    );

                    // Per-track decoding context (whisper `initial_prompt`),
                    // kept independently: the two sides are separate sentences.
                    let mut mic_prior = String::new();
                    let mut sys_prior = String::new();

                    let lang = language_override(&language);

                    let run = |track: Track, span: SpeechSpan, prior: &mut String| {
                        let result = {
                            let guard = engine.lock().expect("engine");
                            guard.transcribe_with_context(
                                &span.audio,
                                crate::audio::SAMPLE_RATE,
                                DecodeContext {
                                    prior: prior.as_str(),
                                    language: lang,
                                },
                            )
                        };
                        match result {
                            Ok(text) if !text.trim().is_empty() => {
                                let text = text.trim().to_string();
                                carry_context(prior, &text);
                                let _ = segment_tx.send(Segment {
                                    elapsed: span.elapsed,
                                    track,
                                    text,
                                });
                            },
                            Ok(_) => {},
                            Err(e) => log::warn!("meeting transcribe failed: {e}"),
                        }
                    };

                    // Track files are opened on the first frame that actually
                    // arrives for a side, so a meeting recorded with only one
                    // source does not leave an empty file behind for the other.
                    let mut files: Vec<(Track, TrackWriter)> = Vec::new();
                    let mut open = |track: Track, audio: &[f32]| {
                        let Some((dir, stem)) = keep_audio.as_ref() else {
                            return;
                        };
                        if let Some((_, w)) = files.iter_mut().find(|(t, _)| *t == track) {
                            w.push(audio);
                            return;
                        }
                        match TrackWriter::create(track_path(dir, stem, track)) {
                            Ok(mut w) => {
                                w.push(audio);
                                files.push((track, w));
                            },
                            // Losing the second pass must not cost the meeting.
                            Err(e) => log::error!("meeting: cannot keep {track:?} audio: {e}"),
                        }
                    };

                    while let Ok(job) = rx.recv() {
                        open(job.track, &job.audio);
                        let (seg, prior) = match job.track {
                            Track::Mic => (&mut mic_seg, &mut mic_prior),
                            Track::System => (&mut sys_seg, &mut sys_prior),
                        };
                        for span in seg.push(&job.audio) {
                            run(job.track, span, prior);
                        }
                    }

                    // Capture has stopped; flush trailing speech both
                    // segmenters were still holding for a pause that never came.
                    for span in mic_seg.finish() {
                        run(Track::Mic, span, &mut mic_prior);
                    }
                    for span in sys_seg.finish() {
                        run(Track::System, span, &mut sys_prior);
                    }

                    let kept: Vec<(Track, PathBuf)> = files
                        .into_iter()
                        .filter_map(|(t, w)| w.finish().map(|p| (t, p)))
                        .collect();
                    if let Ok(mut guard) = tracks.lock() {
                        *guard = kept;
                    }
                })
                .expect("spawn meeting worker"),
        );
    }

    /// Kick off a one-time background download of the VAD model if it is
    /// missing. Fire-and-forget: this session falls back to fixed windows;
    /// the next meeting picks up VAD. Only worth doing on `sherpa` builds,
    /// where the detector can actually run.
    fn fetch_vad_model(&self, config: &Config) {
        #[cfg(feature = "sherpa")]
        {
            let mm = ModelManager::new(Config::models_dir(config));
            if mm.is_vad_downloaded() {
                return;
            }
            let _ = thread::Builder::new()
                .name("whisprking-vad-fetch".into())
                .spawn(move || match mm.ensure_vad_model(None) {
                    Ok(p) => log::info!("vad: model ready at {}", p.display()),
                    Err(e) => log::warn!("vad: model download failed: {e}"),
                });
        }
        #[cfg(not(feature = "sherpa"))]
        let _ = config;
    }

    /// Same idea for the two speaker-diarization models, which the second
    /// pass needs when the meeting is over. Fetching them now means the pass
    /// starts on a model load instead of on a 34 MB download; it would fetch
    /// them itself either way.
    fn fetch_diarize_models(&self, config: &Config) {
        #[cfg(feature = "sherpa")]
        {
            if !config.meeting.diarize.enabled || !config.meeting.label_speakers {
                return;
            }
            let models_dir = Config::models_dir(config);
            if ModelManager::new(&models_dir).is_diarize_downloaded() {
                return;
            }
            let _ = thread::Builder::new()
                .name("whisprking-diarize-fetch".into())
                .spawn(move || super::post::ensure_diarize_models(&models_dir));
        }
        #[cfg(not(feature = "sherpa"))]
        let _ = config;
    }

    fn shutdown_worker(&mut self) {
        // Dropping the sender makes the worker exit once it has drained.
        self.chunk_tx.take();
        if let Some(handle) = self.worker.take() {
            let _ = handle.join();
        }
    }

    /// Move settled segments out of `pending` into the visible transcript
    /// and the file, oldest first.
    fn flush_pending(&mut self, force: bool) {
        if self.pending.is_empty() {
            return;
        }
        self.pending.sort_by(|a, b| a.elapsed.total_cmp(&b.elapsed));

        let newest = self.pending.last().map(|s| s.elapsed).unwrap_or(0.0);
        let cutoff = newest - REORDER_SLACK_SECONDS;

        let split = if force {
            self.pending.len()
        } else {
            self.pending
                .iter()
                .position(|s| s.elapsed > cutoff)
                .unwrap_or(self.pending.len())
        };

        for seg in self.pending.drain(..split) {
            let _ = self
                .writer
                .add_labeled_segment(seg.elapsed, Some(seg.track.label()), &seg.text);
            self.transcript.push(seg);
        }
    }
}

pub fn fmt_mmss(seconds: f64) -> String {
    let s = seconds.max(0.0) as u64;
    format!("{:02}:{:02}", s / 60, s % 60)
}

pub fn fmt_hhmmss(seconds: f64) -> String {
    let s = seconds.max(0.0) as u64;
    format!("{:02}:{:02}:{:02}", s / 3600, (s % 3600) / 60, s % 60)
}
