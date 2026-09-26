//! Speaker diarization — telling the voices inside one track apart.
//!
//! WhisprKing already separates the two *sides* of a call by capturing them as
//! two tracks ([`crate::audio::capture`]), which is why the live transcript can
//! say "Du" and "Andere" without any model at all. What that cannot do is split
//! the "Andere" track into the four people on the call, or an imported
//! recording — one file, one track — into anyone at all.
//!
//! This module does that, offline, with the pair of models sherpa-onnx uses for
//! it:
//!
//! - **pyannote segmentation 3.0** finds speech and marks where the voice
//!   changes, including overlaps.
//! - **A speaker-embedding model** (3D-Speaker CAM++) turns each of those
//!   stretches into a vector, and the vectors are clustered — one cluster per
//!   voice.
//!
//! Clustering is why this is an *offline* pass and never runs live: a cluster
//! only exists once every stretch of the recording has been embedded, so
//! "speaker 2" cannot be named before the recording is over. The live path is
//! deliberately left alone; the second pass
//! ([`crate::transcription::post`]) and the importer ([`crate::audio::import`])
//! are where diarized labels appear.
//!
//! Both models are ONNX and run through the sherpa-onnx native library, so
//! everything here needs the `sherpa` build feature. Without it
//! [`Diarizer::load`] fails with [`DiarizeError::FeatureDisabled`] and callers
//! fall back to the track labels they used before.

use std::path::PathBuf;
#[cfg(feature = "sherpa")]
use std::sync::atomic::{AtomicU32, Ordering};

use thiserror::Error;

use crate::audio::capture::Track;

/// Sample rate the segmentation model was trained at, and the rate every track
/// on disk is already stored at.
pub const SAMPLE_RATE: u32 = 16_000;

/// A line and a diarization segment that miss each other by less than this are
/// still considered the same speech. Timestamps come from two different models
/// (the transcriber's and the segmenter's), so they disagree by a fraction of a
/// second on where an utterance starts.
const NEAR_TOLERANCE: f64 = 1.0;

/// Assumed length of a transcript line when nothing bounds it — the last line
/// of a track has no successor to end it.
const DEFAULT_LINE_SECONDS: f64 = 4.0;

#[derive(Debug, Error)]
pub enum DiarizeError {
    #[error(
        "speaker diarization requires the `sherpa` feature — \
         rebuild with `cargo build --features sherpa`"
    )]
    FeatureDisabled,
    #[error("diarization model missing: {0}")]
    ModelMissing(PathBuf),
    #[error("sherpa-onnx error: {0}")]
    Sherpa(String),
    #[error("the segmentation model expects {expected} Hz audio, got {actual} Hz")]
    SampleRate { expected: u32, actual: u32 },
}

/// One stretch of audio attributed to one voice, in seconds from the start of
/// the track it was found in.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeakerSpan {
    pub start: f64,
    pub end: f64,
    /// Cluster index. Arbitrary until [`renumber_by_first_appearance`] puts it
    /// in the order the voices are heard.
    pub speaker: u32,
}

impl SpeakerSpan {
    fn duration(&self) -> f64 {
        (self.end - self.start).max(0.0)
    }
}

/// Everything the pass needs, assembled by the caller from the config and the
/// model manager.
#[derive(Debug, Clone)]
pub struct DiarizeSpec {
    /// pyannote segmentation `model.onnx`.
    pub segmentation: PathBuf,
    /// Speaker-embedding `.onnx`.
    pub embedding: PathBuf,
    /// Number of voices to cluster into. `0` estimates it from the audio,
    /// which is the only honest default — nobody knows in advance how many
    /// people join a call.
    pub speakers: u32,
    /// Cosine distance at which two stretches are still the same voice. Only
    /// consulted when `speakers == 0`; a fixed count overrides it.
    pub threshold: f32,
    pub threads: i32,
}

/// The two models, loaded. Loading costs a second or two, so one instance is
/// reused across every track of a recording.
pub struct Diarizer {
    #[cfg(feature = "sherpa")]
    inner: sherpa_onnx::OfflineSpeakerDiarization,
    #[cfg(feature = "sherpa")]
    spec: DiarizeSpec,
    /// What the clustering is configured for right now — [`set_speakers`] is
    /// called per track and has to compare against the live value, not the
    /// one the diarizer was loaded with.
    ///
    /// [`set_speakers`]: Diarizer::set_speakers
    #[cfg(feature = "sherpa")]
    speakers: AtomicU32,
    #[cfg(not(feature = "sherpa"))]
    _unused: (),
}

/// Translate a spec into the sherpa-onnx configuration.
#[cfg(feature = "sherpa")]
fn to_config(spec: &DiarizeSpec) -> sherpa_onnx::OfflineSpeakerDiarizationConfig {
    use sherpa_onnx::{
        OfflineSpeakerDiarizationConfig, OfflineSpeakerSegmentationPyannoteModelConfig,
    };

    let mut cfg = OfflineSpeakerDiarizationConfig::default();
    cfg.segmentation.pyannote = OfflineSpeakerSegmentationPyannoteModelConfig {
        model: Some(spec.segmentation.to_string_lossy().into_owned()),
    };
    cfg.segmentation.num_threads = spec.threads.max(1);
    cfg.embedding.model = Some(spec.embedding.to_string_lossy().into_owned());
    cfg.embedding.num_threads = spec.threads.max(1);
    // A fixed count and a distance threshold are alternatives, not a pair:
    // sherpa-onnx only consults the threshold when the count is negative.
    cfg.clustering.num_clusters = match spec.speakers {
        0 => -1,
        n => n as i32,
    };
    cfg.clustering.threshold = spec.threshold;
    cfg
}

impl Diarizer {
    /// Load both models. Fails rather than silently degrading — the caller
    /// decides whether a recording without speaker names is still worth
    /// producing (it is; every caller keeps going and says why).
    #[cfg(feature = "sherpa")]
    pub fn load(spec: &DiarizeSpec) -> Result<Self, DiarizeError> {
        use sherpa_onnx::OfflineSpeakerDiarization;

        for path in [&spec.segmentation, &spec.embedding] {
            if !path.is_file() {
                return Err(DiarizeError::ModelMissing(path.clone()));
            }
        }

        let cfg = to_config(spec);
        log::info!(
            "diarize: loading segmentation={} embedding={} ({})",
            spec.segmentation.display(),
            spec.embedding.display(),
            match spec.speakers {
                0 => format!("auto, threshold {:.2}", spec.threshold),
                n => format!("{n} speakers"),
            },
        );

        let inner = OfflineSpeakerDiarization::create(&cfg).ok_or_else(|| {
            DiarizeError::Sherpa("the diarizer could not be created from these models".into())
        })?;
        let rate = inner.sample_rate();
        if rate != SAMPLE_RATE as i32 {
            return Err(DiarizeError::SampleRate {
                expected: rate.max(0) as u32,
                actual: SAMPLE_RATE,
            });
        }
        Ok(Self {
            inner,
            speakers: AtomicU32::new(spec.speakers),
            spec: spec.clone(),
        })
    }

    /// Change how many voices to expect before the next [`run`](Self::run).
    ///
    /// A recording has one count but several tracks, and the count belongs to
    /// the track the people are on. Forcing "four speakers" onto a microphone
    /// track that only ever carried one would invent three, so the caller
    /// hands that track a `0` and lets it estimate. Only the clustering is
    /// reconfigured; the models stay loaded.
    #[cfg(feature = "sherpa")]
    pub fn set_speakers(&self, speakers: u32) {
        if self.speakers.swap(speakers, Ordering::Relaxed) == speakers {
            return;
        }
        let mut spec = self.spec.clone();
        spec.speakers = speakers;
        self.inner.set_config(&to_config(&spec));
    }

    #[cfg(not(feature = "sherpa"))]
    pub fn set_speakers(&self, _speakers: u32) {}

    #[cfg(not(feature = "sherpa"))]
    pub fn load(_spec: &DiarizeSpec) -> Result<Self, DiarizeError> {
        Err(DiarizeError::FeatureDisabled)
    }

    /// Diarize one whole track. Blocking, and roughly linear in the length of
    /// the recording — an hour of audio is minutes of work, which is why this
    /// only ever runs on a background thread next to the other offline passes.
    ///
    /// The returned spans are sorted by start time and renumbered so speaker 0
    /// is the first voice heard.
    #[cfg(feature = "sherpa")]
    pub fn run(&self, samples: &[f32], sample_rate: u32) -> Result<Vec<SpeakerSpan>, DiarizeError> {
        if sample_rate != SAMPLE_RATE {
            return Err(DiarizeError::SampleRate {
                expected: SAMPLE_RATE,
                actual: sample_rate,
            });
        }
        let result = self
            .inner
            .process(samples)
            .ok_or_else(|| DiarizeError::Sherpa("diarization returned no result".into()))?;

        let mut spans: Vec<SpeakerSpan> = result
            .sort_by_start_time()
            .into_iter()
            .map(|s| SpeakerSpan {
                start: s.start as f64,
                end: s.end as f64,
                speaker: s.speaker.max(0) as u32,
            })
            .collect();
        let voices = renumber_by_first_appearance(&mut spans);
        log::info!(
            "diarize: {} segment(s), {voices} voice(s) in {:.1}s of audio",
            spans.len(),
            samples.len() as f64 / SAMPLE_RATE as f64,
        );
        Ok(spans)
    }

    #[cfg(not(feature = "sherpa"))]
    pub fn run(
        &self,
        _samples: &[f32],
        _sample_rate: u32,
    ) -> Result<Vec<SpeakerSpan>, DiarizeError> {
        Err(DiarizeError::FeatureDisabled)
    }
}

/// Renumber the clusters so that speaker 0 is the first voice heard, 1 the
/// second, and so on. Returns how many voices there are.
///
/// The cluster indices sherpa-onnx hands back are an implementation detail of
/// the clustering, in no particular order. "Person 1" has to mean the first
/// person on the recording or the numbering is noise.
pub fn renumber_by_first_appearance(spans: &mut [SpeakerSpan]) -> u32 {
    let mut order: Vec<u32> = Vec::new();
    for span in spans.iter() {
        if !order.contains(&span.speaker) {
            order.push(span.speaker);
        }
    }
    for span in spans.iter_mut() {
        span.speaker = order
            .iter()
            .position(|s| *s == span.speaker)
            .unwrap_or(0) as u32;
    }
    order.len() as u32
}

/// Which voice was speaking over `[start, end)`.
///
/// Picks the voice with the most overlap, so a line that spans a handover goes
/// to whoever said most of it. With no overlap at all — the transcriber found
/// words where the segmenter found none — the nearest voice within
/// [`NEAR_TOLERANCE`] wins, and beyond that nobody does.
pub fn speaker_at(spans: &[SpeakerSpan], start: f64, end: f64) -> Option<u32> {
    let end = end.max(start);
    let mut best: Option<(f64, u32)> = None;
    for span in spans {
        let overlap = span.end.min(end) - span.start.max(start);
        if overlap > 0.0 && best.map(|(o, _)| overlap > o).unwrap_or(true) {
            best = Some((overlap, span.speaker));
        }
    }
    if let Some((_, speaker)) = best {
        return Some(speaker);
    }

    let mut nearest: Option<(f64, u32)> = None;
    for span in spans {
        let gap = if end <= span.start {
            span.start - end
        } else {
            start - span.end
        };
        if gap <= NEAR_TOLERANCE && nearest.map(|(g, _)| gap < g).unwrap_or(true) {
            nearest = Some((gap, span.speaker));
        }
    }
    nearest.map(|(_, speaker)| speaker)
}

/// End time to assume for a transcript line: where the next line on the same
/// track starts, or a few seconds when it is the last one.
pub fn line_end(start: f64, next_start: Option<f64>) -> f64 {
    match next_start {
        Some(next) if next > start => next,
        _ => start + DEFAULT_LINE_SECONDS,
    }
}

/// The name each voice gets in the transcript.
pub fn person_label(speaker: u32) -> String {
    format!("Person {}", speaker + 1)
}

/// The names for a whole recording — every voice on every track, numbered once
/// so that "Person 2" means the same person in both.
#[derive(Debug, Clone, Default)]
pub struct SpeakerNames {
    /// `(track, speaker) → label`. Two entries at most per voice, and a
    /// recording has a handful of voices, so a `Vec` beats a map.
    names: Vec<(Track, u32, String)>,
}

impl SpeakerNames {
    /// Number the voices in the order they are first heard, across all tracks.
    ///
    /// The microphone track is the one exception: it is *your* microphone, so
    /// the voice that does most of the talking on it is you and keeps the
    /// label it has always had. Anyone else picked up by the same microphone —
    /// someone sitting in the room with you — is numbered like everyone else.
    pub fn assign(tracks: &[(Track, Vec<SpeakerSpan>)]) -> Self {
        // (track, speaker) → (first start, total speech)
        let mut stats: Vec<(Track, u32, f64, f64)> = Vec::new();
        for (track, spans) in tracks {
            for span in spans {
                match stats
                    .iter_mut()
                    .find(|(t, s, _, _)| t == track && *s == span.speaker)
                {
                    Some((_, _, first, total)) => {
                        *first = first.min(span.start);
                        *total += span.duration();
                    },
                    None => stats.push((*track, span.speaker, span.start, span.duration())),
                }
            }
        }

        let you = stats
            .iter()
            .filter(|(t, _, _, _)| *t == Track::Mic)
            .max_by(|a, b| a.3.total_cmp(&b.3))
            .map(|(t, s, _, _)| (*t, *s));

        let mut rest: Vec<&(Track, u32, f64, f64)> = stats
            .iter()
            .filter(|(t, s, _, _)| Some((*t, *s)) != you)
            .collect();
        rest.sort_by(|a, b| a.2.total_cmp(&b.2));

        let mut names = Vec::new();
        if let Some((track, speaker)) = you {
            names.push((track, speaker, Track::Mic.label().to_string()));
        }
        for (i, (track, speaker, _, _)) in rest.into_iter().enumerate() {
            names.push((*track, *speaker, person_label(i as u32)));
        }
        Self { names }
    }

    pub fn label(&self, track: Track, speaker: u32) -> Option<&str> {
        self.names
            .iter()
            .find(|(t, s, _)| *t == track && *s == speaker)
            .map(|(_, _, name)| name.as_str())
    }

    /// How many voices were found in the whole recording.
    pub fn count(&self) -> usize {
        self.names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(start: f64, end: f64, speaker: u32) -> SpeakerSpan {
        SpeakerSpan {
            start,
            end,
            speaker,
        }
    }

    #[test]
    fn renumbering_follows_the_order_voices_are_heard() {
        let mut spans = vec![span(0.0, 2.0, 7), span(2.5, 4.0, 3), span(4.5, 5.0, 7)];
        assert_eq!(renumber_by_first_appearance(&mut spans), 2);
        assert_eq!(spans[0].speaker, 0);
        assert_eq!(spans[1].speaker, 1);
        assert_eq!(spans[2].speaker, 0);
    }

    #[test]
    fn a_line_goes_to_whoever_said_most_of_it() {
        let spans = vec![span(0.0, 5.0, 0), span(5.0, 12.0, 1)];
        // Starts inside speaker 0 but two thirds of it is speaker 1.
        assert_eq!(speaker_at(&spans, 4.0, 9.0), Some(1));
        assert_eq!(speaker_at(&spans, 0.5, 3.0), Some(0));
    }

    /// The transcriber and the segmenter are different models and disagree
    /// slightly about where an utterance starts. A line that lands in the gap
    /// still belongs to the voice next to it.
    #[test]
    fn a_line_just_outside_a_span_takes_the_nearest_voice() {
        let spans = vec![span(0.0, 5.0, 0), span(10.0, 20.0, 1)];
        assert_eq!(speaker_at(&spans, 5.4, 5.6), Some(0));
        assert_eq!(speaker_at(&spans, 9.5, 9.9), Some(1));
        // Four seconds of silence away from either: no honest answer.
        assert_eq!(speaker_at(&spans, 7.0, 7.5), None);
    }

    #[test]
    fn nothing_diarized_names_nobody() {
        assert_eq!(speaker_at(&[], 0.0, 10.0), None);
        assert!(SpeakerNames::assign(&[]).is_empty());
    }

    #[test]
    fn the_dominant_voice_on_the_microphone_is_you() {
        let names = SpeakerNames::assign(&[
            // You talk for 30 s; someone in the room with you says one thing.
            (
                Track::Mic,
                vec![span(0.0, 30.0, 1), span(31.0, 33.0, 0)],
            ),
            (Track::System, vec![span(40.0, 45.0, 0), span(50.0, 55.0, 1)]),
        ]);
        assert_eq!(names.label(Track::Mic, 1), Some("Du"));
        // Everyone else is numbered by when they were first heard, which is
        // the person in the room at 31 s, then the two on the call.
        assert_eq!(names.label(Track::Mic, 0), Some("Person 1"));
        assert_eq!(names.label(Track::System, 0), Some("Person 2"));
        assert_eq!(names.label(Track::System, 1), Some("Person 3"));
        assert_eq!(names.count(), 4);
    }

    /// An imported file has no microphone track, so nobody is "Du".
    #[test]
    fn a_single_track_recording_is_all_persons() {
        let names = SpeakerNames::assign(&[(
            Track::System,
            vec![span(0.0, 4.0, 0), span(4.0, 9.0, 1)],
        )]);
        assert_eq!(names.label(Track::System, 0), Some("Person 1"));
        assert_eq!(names.label(Track::System, 1), Some("Person 2"));
    }

    /// The real thing, against the real models — the only way to find out that
    /// a URL moved, a tarball changed shape, or the native call needs
    /// something we are not giving it. Ignored by default because it needs
    /// the `sherpa` feature, a network for the first run, and a recording:
    ///
    /// ```text
    /// WHISPRKING_DIARIZE_WAV=/path/two-speakers-16k-mono.wav \
    ///   cargo test --features sherpa -- --ignored --nocapture diarizes_a_real_recording
    /// ```
    ///
    /// The models land in `WHISPRKING_MODELS_DIR`, or the app's own models
    /// directory when that is unset.
    #[test]
    #[ignore = "needs the sherpa feature, the diarization models and a recording"]
    fn diarizes_a_real_recording() {
        use crate::audio::track_writer::read_track;
        use crate::config::Config;
        use crate::transcription::model_manager::ModelManager;

        let wav = std::env::var("WHISPRKING_DIARIZE_WAV")
            .expect("set WHISPRKING_DIARIZE_WAV to a 16 kHz mono recording");
        let models_dir = std::env::var("WHISPRKING_MODELS_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| Config::models_dir(&Config::default()));

        let paths = ModelManager::new(&models_dir)
            .ensure_diarize_models(None)
            .expect("diarization models");
        let samples = read_track(std::path::Path::new(&wav)).expect("16 kHz mono wav");

        let diarizer = Diarizer::load(&DiarizeSpec {
            segmentation: paths.segmentation,
            embedding: paths.embedding,
            speakers: 0,
            threshold: 0.5,
            threads: 4,
        })
        .expect("load");
        let spans = diarizer.run(&samples, SAMPLE_RATE).expect("run");

        assert!(!spans.is_empty(), "no speech found in {wav}");
        let voices = spans.iter().map(|s| s.speaker).max().unwrap_or(0) + 1;
        for s in &spans {
            println!(
                "{:>7.2}s – {:>7.2}s  {}",
                s.start,
                s.end,
                person_label(s.speaker)
            );
        }
        println!("{} segment(s), {voices} voice(s)", spans.len());
        assert!(voices >= 2, "expected more than one voice, got {voices}");
        // The renumbering contract the labels depend on.
        assert_eq!(spans[0].speaker, 0);
    }

    #[test]
    fn the_last_line_of_a_track_still_gets_an_end() {
        assert_eq!(line_end(10.0, Some(14.0)), 14.0);
        assert_eq!(line_end(10.0, None), 10.0 + DEFAULT_LINE_SECONDS);
        // A next line that is not actually later must not produce a backwards
        // interval — overlapping tracks and rounding both make that happen.
        assert_eq!(line_end(10.0, Some(9.0)), 10.0 + DEFAULT_LINE_SECONDS);
    }
}
