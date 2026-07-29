//! Speech-boundary segmentation for the meeting pipeline.
//!
//! The capture side is deliberately dumb: it forwards small raw frames per
//! track. This module turns that continuous per-track stream into discrete
//! [`SpeechSpan`]s — the units actually handed to the transcriber.
//!
//! Two backends live behind one [`Segmenter`]:
//!
//! - **VAD** ([`sherpa_onnx::VoiceActivityDetector`], Silero) — cuts on real
//!   speech boundaries so a chunk never starts or ends mid-word, and force-
//!   splits a monologue at [`VadParams::max_speech_duration`] so we never
//!   blow past Whisper's 30 s window. Only available in `sherpa`-feature
//!   builds (the detector is in the native lib) and only when the VAD model
//!   has been downloaded.
//! - **Fixed windows** — the pre-VAD behaviour: cut every `chunk_seconds`,
//!   drop windows quieter than [`SILENCE_RMS`]. Always available; the
//!   fallback when VAD is unavailable. Pure logic, unit-tested below.
//!
//! Fixed-window chunking is what made meeting transcripts bad: a 10 s
//! boundary lands mid-word several times a minute, and a transducer fed a
//! fragment starting mid-syllable emits garbage. VAD is the fix; the fixed
//! path is kept only so whisper-only builds still work.

use std::path::Path;

use super::resample::rms;

/// Below this RMS a *fixed-window* chunk is treated as silence and dropped
/// before it reaches the engine — a silent chunk costs a full model
/// invocation and yields only hallucinated filler. The VAD path does its
/// own silence handling and ignores this.
pub const SILENCE_RMS: f32 = 0.004;

/// Sample rate every backend operates at.
const SAMPLE_RATE: u32 = 16_000;

/// One span of audio ready for transcription, tagged with its start time.
#[derive(Debug, Clone)]
pub struct SpeechSpan {
    /// Seconds since the start of this track's stream, at the *start* of the
    /// span. Feeds the transcript timestamp and the cross-track reorder
    /// buffer, so it must be the segment start, not its end.
    pub elapsed: f64,
    pub audio: Vec<f32>,
}

/// Tunables for the Silero VAD. Defaults follow the phase-2 plan.
#[derive(Debug, Clone, Copy)]
pub struct VadParams {
    pub threshold: f32,
    pub min_silence_duration: f32,
    pub min_speech_duration: f32,
    /// Hard cap: continuous speech with no qualifying pause is force-split at
    /// the deepest silence before this, so a segment can never outgrow the
    /// transcriber's window or build an unbounded buffer.
    pub max_speech_duration: f32,
    pub window_size: i32,
    /// Size of the detector's internal result buffer; must comfortably hold
    /// the largest possible segment (`max_speech_duration`).
    pub buffer_seconds: f32,
}

impl Default for VadParams {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            min_silence_duration: 0.4,
            min_speech_duration: 0.25,
            // A hair under 30 s so even the longest forced segment stays
            // inside Whisper's single-window budget.
            max_speech_duration: 28.0,
            window_size: 512,
            buffer_seconds: 30.0,
        }
    }
}

/// Per-track segmenter. One instance per [`super::capture::Track`] — the two
/// sides of a call are independent streams and must not share VAD state.
pub struct Segmenter {
    inner: Inner,
}

enum Inner {
    Fixed(FixedSegmenter),
    #[cfg(feature = "sherpa")]
    Vad(VadSegmenter),
}

impl Segmenter {
    /// Prefer VAD; fall back to fixed windows.
    ///
    /// Uses VAD when `vad_model` is `Some`, the file exists, and this is a
    /// `sherpa`-feature build; otherwise (or if the detector fails to load)
    /// returns a fixed-window segmenter cutting every `chunk_seconds`.
    pub fn new(vad_model: Option<&Path>, chunk_seconds: u32) -> Self {
        #[cfg(feature = "sherpa")]
        if let Some(model) = vad_model {
            if model.is_file() {
                match VadSegmenter::new(model, VadParams::default()) {
                    Ok(v) => {
                        return Self {
                            inner: Inner::Vad(v),
                        }
                    }
                    Err(e) => log::warn!(
                        "vad: could not load {}, falling back to fixed windows: {e}",
                        model.display()
                    ),
                }
            }
        }
        #[cfg(not(feature = "sherpa"))]
        let _ = vad_model;

        Self {
            inner: Inner::Fixed(FixedSegmenter::new(chunk_seconds)),
        }
    }

    /// `true` if this segmenter cuts on speech boundaries rather than a clock.
    pub fn uses_vad(&self) -> bool {
        match self.inner {
            #[cfg(feature = "sherpa")]
            Inner::Vad(_) => true,
            Inner::Fixed(_) => false,
        }
    }

    /// Feed the next raw frame; return any spans that completed.
    pub fn push(&mut self, samples: &[f32]) -> Vec<SpeechSpan> {
        match &mut self.inner {
            Inner::Fixed(f) => f.push(samples),
            #[cfg(feature = "sherpa")]
            Inner::Vad(v) => v.push(samples),
        }
    }

    /// End of stream: flush whatever speech is still buffered.
    pub fn finish(&mut self) -> Vec<SpeechSpan> {
        match &mut self.inner {
            Inner::Fixed(f) => f.finish(),
            #[cfg(feature = "sherpa")]
            Inner::Vad(v) => v.finish(),
        }
    }
}

// =====================================================================
// Fixed-window backend (pure logic — replicates the pre-VAD behaviour)
// =====================================================================

struct FixedSegmenter {
    chunk_samples: usize,
    pending: Vec<f32>,
    /// Samples already cut into spans, for deriving each span's start time
    /// from the audio clock rather than wall-clock. Advances even for spans
    /// dropped as silent, so timestamps stay aligned to real elapsed audio.
    emitted: usize,
}

impl FixedSegmenter {
    fn new(chunk_seconds: u32) -> Self {
        Self {
            chunk_samples: chunk_seconds.max(1) as usize * SAMPLE_RATE as usize,
            pending: Vec::new(),
            emitted: 0,
        }
    }

    fn cut(&mut self, n: usize) -> SpeechSpan {
        let audio: Vec<f32> = self.pending.drain(..n).collect();
        let elapsed = self.emitted as f64 / SAMPLE_RATE as f64;
        self.emitted += audio.len();
        SpeechSpan { elapsed, audio }
    }

    fn push(&mut self, samples: &[f32]) -> Vec<SpeechSpan> {
        self.pending.extend_from_slice(samples);
        let mut out = Vec::new();
        while self.pending.len() >= self.chunk_samples {
            let span = self.cut(self.chunk_samples);
            if rms(&span.audio) >= SILENCE_RMS {
                out.push(span);
            }
        }
        out
    }

    fn finish(&mut self) -> Vec<SpeechSpan> {
        if self.pending.is_empty() {
            return Vec::new();
        }
        let n = self.pending.len();
        let span = self.cut(n);
        if rms(&span.audio) >= SILENCE_RMS {
            vec![span]
        } else {
            Vec::new()
        }
    }
}

// =====================================================================
// VAD backend (sherpa-onnx Silero)
// =====================================================================

#[cfg(feature = "sherpa")]
struct VadSegmenter {
    vad: sherpa_onnx::VoiceActivityDetector,
}

#[cfg(feature = "sherpa")]
impl VadSegmenter {
    fn new(model: &Path, params: VadParams) -> Result<Self, String> {
        use sherpa_onnx::{SileroVadModelConfig, VadModelConfig, VoiceActivityDetector};

        let mut cfg = VadModelConfig::default();
        cfg.silero_vad = SileroVadModelConfig {
            model: Some(model.to_string_lossy().into_owned()),
            threshold: params.threshold,
            min_silence_duration: params.min_silence_duration,
            min_speech_duration: params.min_speech_duration,
            max_speech_duration: params.max_speech_duration,
            window_size: params.window_size,
            // Spread the remainder so a future field added to the crate's
            // struct doesn't break this literal.
            ..Default::default()
        };
        cfg.sample_rate = SAMPLE_RATE as i32;
        cfg.num_threads = 1;
        cfg.provider = Some("cpu".into());
        cfg.debug = false;

        let vad = VoiceActivityDetector::create(&cfg, params.buffer_seconds)
            .ok_or_else(|| "VoiceActivityDetector::create returned None".to_string())?;
        log::info!("vad: Silero segmentation active ({})", model.display());
        Ok(Self { vad })
    }

    /// Move every queued segment out of the detector into `out`.
    fn drain(&self, out: &mut Vec<SpeechSpan>) {
        while !self.vad.is_empty() {
            if let Some(seg) = self.vad.front() {
                // `start` is in samples, relative to all audio fed so far —
                // exactly this track's elapsed time at the segment start.
                out.push(SpeechSpan {
                    elapsed: seg.start() as f64 / SAMPLE_RATE as f64,
                    audio: seg.samples().to_vec(),
                });
            }
            self.vad.pop();
        }
    }

    fn push(&mut self, samples: &[f32]) -> Vec<SpeechSpan> {
        self.vad.accept_waveform(samples);
        let mut out = Vec::new();
        self.drain(&mut out);
        out
    }

    fn finish(&mut self) -> Vec<SpeechSpan> {
        // Emit any trailing speech the detector was still holding for a pause
        // that never came.
        self.vad.flush();
        let mut out = Vec::new();
        self.drain(&mut out);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: usize = SAMPLE_RATE as usize;

    fn loud(n: usize) -> Vec<f32> {
        vec![0.5; n]
    }

    fn quiet(n: usize) -> Vec<f32> {
        vec![0.0; n]
    }

    #[test]
    fn fixed_cuts_at_the_configured_window() {
        let mut seg = FixedSegmenter::new(1); // 1 s == SR samples
                                              // Two-and-a-half windows in one push.
        let spans = seg.push(&loud(SR * 2 + SR / 2)); // 2.5 s
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].audio.len(), SR);
        assert_eq!(spans[0].elapsed, 0.0);
        assert!((spans[1].elapsed - 1.0).abs() < 1e-9);

        // The trailing half-window only comes out on finish.
        let tail = seg.finish();
        assert_eq!(tail.len(), 1);
        assert_eq!(tail[0].audio.len(), SR / 2);
        assert!((tail[0].elapsed - 2.0).abs() < 1e-9);
    }

    #[test]
    fn fixed_accumulates_across_small_frames() {
        let mut seg = FixedSegmenter::new(1);
        // Feed one window in ten 0.1 s frames: nothing until it fills.
        let mut spans = Vec::new();
        for _ in 0..9 {
            spans.extend(seg.push(&loud(SR / 10)));
        }
        assert!(spans.is_empty(), "no full window yet");
        spans.extend(seg.push(&loud(SR / 10)));
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].audio.len(), SR);
    }

    #[test]
    fn fixed_drops_silent_windows_but_keeps_the_clock() {
        let mut seg = FixedSegmenter::new(1);
        // Silent window, then a loud one.
        let mut spans = seg.push(&quiet(SR));
        assert!(spans.is_empty(), "silent window dropped");
        spans = seg.push(&loud(SR));
        assert_eq!(spans.len(), 1);
        // The dropped silent second must still have advanced elapsed.
        assert!(
            (spans[0].elapsed - 1.0).abs() < 1e-9,
            "elapsed should count the dropped silence, got {}",
            spans[0].elapsed
        );
    }

    #[test]
    fn fixed_finish_on_empty_is_noop() {
        let mut seg = FixedSegmenter::new(1);
        assert!(seg.finish().is_empty());
    }

    #[test]
    fn fixed_finish_drops_a_silent_tail() {
        let mut seg = FixedSegmenter::new(10);
        assert!(seg.push(&quiet(SR)).is_empty());
        assert!(seg.finish().is_empty(), "silent tail should not be emitted");
    }
}
