//! Pack speech spans into the largest window a backend can actually use.
//!
//! The live meeting path hands the engine one [`SpeechSpan`] at a time,
//! because it has to: a segment cannot be transcribed before it has been
//! spoken. That costs accuracy in a way that is invisible until you read the
//! transcript — every span is decoded with no idea what came before or after
//! it, so a sentence split across two spans is decoded as two fragments, each
//! guessing at its own beginning.
//!
//! The post pass has the whole recording, so it does not have to accept that.
//! This module glues consecutive spans back together into windows as long as
//! the backend will take ([`Transcriber::max_input_seconds`]) — 28 s for
//! whisper's mel window, minutes for a transducer — and cuts *only* where the
//! VAD already found silence. No window boundary ever lands mid-word.
//!
//! The silence between two glued spans is put back, capped at
//! [`MAX_GAP_SECONDS`]: a real pause tells the model a sentence ended, but a
//! four-minute gap between two remarks would otherwise eat the whole window.
//! Because that cap makes window-relative time drift from recording time,
//! each window remembers where its source spans landed and re-times decoded
//! pieces through [`Window::absolute`].
//!
//! [`Transcriber::max_input_seconds`]: crate::transcription::engine::Transcriber::max_input_seconds

use crate::audio::vad::SpeechSpan;

/// Sample rate every backend operates at.
const SAMPLE_RATE: f64 = 16_000.0;

/// Longest silence inserted between two glued spans. Enough to read as a
/// sentence boundary, short enough that a long pause cannot consume the
/// window we are trying to fill with speech.
pub const MAX_GAP_SECONDS: f64 = 0.6;

/// Where one source span starts inside a packed window.
#[derive(Debug, Clone, Copy)]
struct Mark {
    /// Sample index within the window's audio.
    at: usize,
    /// Time in the recording that sample corresponds to.
    elapsed: f64,
}

/// One buffer to hand an engine, plus what it takes to map results back onto
/// the recording's clock.
#[derive(Debug, Clone)]
pub struct Window {
    /// Start of this window in the recording, in seconds.
    pub elapsed: f64,
    pub audio: Vec<f32>,
    marks: Vec<Mark>,
}

impl Window {
    pub fn duration(&self) -> f64 {
        self.audio.len() as f64 / SAMPLE_RATE
    }

    /// Recording time of a point `offset` seconds into this window.
    ///
    /// Inside a span the mapping is exact; across an inserted gap it resolves
    /// to the following span's real start, which is where the speech actually
    /// is. Without this every timestamp after the first capped pause would be
    /// wrong by the amount of silence we removed.
    pub fn absolute(&self, offset: f64) -> f64 {
        let sample = (offset.max(0.0) * SAMPLE_RATE) as usize;
        let mark = self
            .marks
            .iter()
            .rev()
            .find(|m| m.at <= sample)
            .or_else(|| self.marks.first());
        match mark {
            Some(m) => m.elapsed + (sample.saturating_sub(m.at) as f64) / SAMPLE_RATE,
            None => self.elapsed + offset,
        }
    }
}

/// Glue `spans` into windows of at most `max_seconds`.
///
/// Spans must be in spoken order — they come straight off one track's
/// segmenter, which produces them that way. A single span longer than the
/// budget is emitted alone rather than cut: splitting it would reintroduce
/// exactly the mid-word boundary this exists to remove, and every backend
/// degrades more gracefully on an over-long buffer than on a severed word.
pub fn pack(spans: &[SpeechSpan], max_seconds: f64) -> Vec<Window> {
    let budget = (max_seconds.max(1.0) * SAMPLE_RATE) as usize;
    let max_gap = (MAX_GAP_SECONDS * SAMPLE_RATE) as usize;

    let mut out: Vec<Window> = Vec::new();
    let mut current: Option<Window> = None;

    for span in spans {
        if span.audio.is_empty() {
            continue;
        }

        let Some(window) = current.as_mut() else {
            current = Some(start_window(span));
            continue;
        };

        // Silence between the end of what we have and the start of this span,
        // as it was actually recorded. Negative means the segmenter handed us
        // overlapping spans, which nothing downstream should paper over — fall
        // back to no gap at all.
        let window_end = window.elapsed + window.duration();
        let gap_samples = (((span.elapsed - window_end).max(0.0)) * SAMPLE_RATE) as usize;
        let gap = gap_samples.min(max_gap);

        if window.audio.len() + gap + span.audio.len() > budget {
            out.push(current.take().expect("window present"));
            current = Some(start_window(span));
            continue;
        }

        window.audio.resize(window.audio.len() + gap, 0.0);
        window.marks.push(Mark {
            at: window.audio.len(),
            elapsed: span.elapsed,
        });
        window.audio.extend_from_slice(&span.audio);
    }

    out.extend(current);
    out
}

fn start_window(span: &SpeechSpan) -> Window {
    Window {
        elapsed: span.elapsed,
        audio: span.audio.clone(),
        marks: vec![Mark {
            at: 0,
            elapsed: span.elapsed,
        }],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(elapsed: f64, seconds: f64) -> SpeechSpan {
        SpeechSpan {
            elapsed,
            audio: vec![0.5; (seconds * SAMPLE_RATE) as usize],
        }
    }

    #[test]
    fn short_spans_are_glued_into_one_window() {
        // Five 3 s utterances, half a second apart, into a 28 s budget.
        let spans: Vec<SpeechSpan> = (0..5).map(|i| span(i as f64 * 3.5, 3.0)).collect();
        let windows = pack(&spans, 28.0);
        assert_eq!(windows.len(), 1, "everything fits in one window");
        // 5 × 3 s of speech + 4 × 0.5 s of restored silence.
        assert!((windows[0].duration() - 17.0).abs() < 0.01);
        assert_eq!(windows[0].elapsed, 0.0);
    }

    #[test]
    fn the_budget_is_never_exceeded() {
        let spans: Vec<SpeechSpan> = (0..20).map(|i| span(i as f64 * 5.5, 5.0)).collect();
        let windows = pack(&spans, 28.0);
        assert!(windows.len() > 1, "20 × 5 s cannot be one 28 s window");
        for w in &windows {
            assert!(
                w.duration() <= 28.0 + 1e-6,
                "window ran to {:.2}s",
                w.duration()
            );
        }
    }

    /// The point of the exercise: a transducer's longer budget must produce
    /// fewer, longer windows from the very same spans.
    #[test]
    fn a_bigger_budget_means_fewer_cuts() {
        let spans: Vec<SpeechSpan> = (0..40).map(|i| span(i as f64 * 5.5, 5.0)).collect();
        let whisper = pack(&spans, 28.0);
        let transducer = pack(&spans, 120.0);
        assert!(
            transducer.len() < whisper.len(),
            "{} vs {} windows",
            transducer.len(),
            whisper.len()
        );
    }

    /// A long silence is not carried into the window, but the timestamps on
    /// the far side of it must still point at when things were said.
    #[test]
    fn timestamps_survive_a_capped_pause() {
        // Speech at 0–4 s, then nothing until 120 s, then speech again.
        let spans = vec![span(0.0, 4.0), span(120.0, 4.0)];
        let windows = pack(&spans, 28.0);
        assert_eq!(windows.len(), 1, "both fit — the pause is capped");
        let w = &windows[0];
        assert!(
            (w.duration() - 8.6).abs() < 0.01,
            "expected 4 + 0.6 + 4 s, got {:.2}",
            w.duration()
        );

        // Start of the first span.
        assert!((w.absolute(0.0) - 0.0).abs() < 0.01);
        // Two seconds into the first span is two seconds into the recording.
        assert!((w.absolute(2.0) - 2.0).abs() < 0.01);
        // The second span starts 4.6 s into the window but 120 s into the call.
        assert!(
            (w.absolute(4.6) - 120.0).abs() < 0.01,
            "got {:.2}",
            w.absolute(4.6)
        );
        assert!((w.absolute(6.6) - 122.0).abs() < 0.01);
    }

    #[test]
    fn an_oversized_span_is_passed_through_whole() {
        let spans = vec![span(0.0, 45.0)];
        let windows = pack(&spans, 28.0);
        assert_eq!(windows.len(), 1);
        assert!(
            (windows[0].duration() - 45.0).abs() < 0.01,
            "an over-long span must not be cut mid-word"
        );
    }

    #[test]
    fn empty_input_and_empty_spans_are_dropped() {
        assert!(pack(&[], 28.0).is_empty());
        assert!(pack(&[span(0.0, 0.0)], 28.0).is_empty());
    }

    /// Overlapping spans are a segmenter bug; packing must not panic or
    /// produce a negative gap.
    #[test]
    fn overlapping_spans_do_not_break_packing() {
        let spans = vec![span(0.0, 5.0), span(2.0, 5.0)];
        let windows = pack(&spans, 28.0);
        assert_eq!(windows.len(), 1);
        assert!((windows[0].duration() - 10.0).abs() < 0.01);
    }
}
