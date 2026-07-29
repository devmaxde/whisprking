//! Speech-to-text engines.
//!
//! Two backends share a single [`Transcriber`] trait:
//!
//! - [`WhisperEngine`] — whisper.cpp via `whisper-rs`. Multilingual; always
//!   compiled in.
//! - [`SherpaTransducerEngine`] — sherpa-onnx offline transducer (NeMo
//!   Parakeet TDT v3). Multilingual (25 European languages incl. German);
//!   gated behind the `sherpa` feature.
//!
//! Both pick the right backend automatically: call [`build_transcriber`]
//! with a model name and it returns the right `Box<dyn Transcriber>`.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use thiserror::Error;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use super::model_manager::{Backend, ModelError, ModelManager};

#[derive(Debug, Error)]
pub enum EngineError {
    #[error(transparent)]
    Model(#[from] ModelError),
    #[error("model not downloaded: {0}")]
    ModelMissing(String),
    #[error("whisper.cpp error: {0}")]
    Whisper(String),
    #[error("sherpa-onnx error: {0}")]
    Sherpa(String),
    #[error("model {model} requires the `sherpa` feature — rebuild with `cargo build --features sherpa`")]
    SherpaFeatureDisabled { model: String },
}

impl From<whisper_rs::WhisperError> for EngineError {
    fn from(err: whisper_rs::WhisperError) -> Self {
        EngineError::Whisper(err.to_string())
    }
}

/// Sample rate that all backends operate at.
pub const SAMPLE_RATE: u32 = 16_000;

/// Longest buffer a whisper.cpp call should be given, in seconds.
///
/// Whisper's encoder works on a fixed 30 s mel window. Handing it more than
/// that is not an error — whisper.cpp just loops over as many windows as it
/// needs — but each extra window is decoded with no acoustic context from the
/// one before it, which is the boundary artefact this whole path exists to
/// avoid. Two seconds of headroom below 30 keeps a packed window from
/// spilling a fragment into a second pass.
pub const WHISPER_MAX_INPUT_SECONDS: f64 = 28.0;

/// Longest buffer a sherpa offline transducer call should be given.
///
/// A transducer has no fixed window: it consumes the whole utterance in one
/// encoder pass, so the real limit is memory, not architecture. Two minutes
/// keeps peak RSS reasonable for the int8 Parakeet encoder while being long
/// enough that a packed window almost never has to cut a train of thought.
pub const SHERPA_MAX_INPUT_SECONDS: f64 = 120.0;

/// One timestamped piece of a decoded buffer.
///
/// The post-transcription pass hands an engine a window several times longer
/// than one utterance, so a single string back would collapse minutes of
/// speech onto one timestamp. Both backends can say *when* inside the window
/// each piece was spoken — whisper per decoded segment, sherpa per token —
/// so the transcript keeps utterance-level timing even though the decode was
/// done in large blocks.
#[derive(Debug, Clone, PartialEq)]
pub struct TimedPiece {
    /// Seconds from the start of the buffer that was passed in.
    pub offset: f64,
    pub text: String,
}

/// Per-call decoding hints for one segment of a longer recording.
///
/// Both fields exist because a meeting is decoded segment-by-segment, which
/// robs the model of the two things it would otherwise infer from a whole
/// recording: what was being said just before, and which language this is.
#[derive(Debug, Clone, Copy, Default)]
pub struct DecodeContext<'a> {
    /// Tail of the previous segment on the same track, fed to the model as
    /// decoding context (whisper's `initial_prompt`) so a chunk that continues
    /// a sentence decodes coherently instead of guessing at a mid-thought
    /// start. Empty for one-shot dictation, which has no prior.
    pub prior: &'a str,
    /// Language to pin this call to, overriding whatever the engine was built
    /// with. `None` keeps the engine's own setting.
    ///
    /// This matters most for meetings: the engine is built once at startup
    /// from the *dictation* settings, so without an override every meeting
    /// segment would decode under dictation's language — usually `auto`,
    /// which re-detects per segment and lets one English loanword in a German
    /// sentence flip whisper into *translating* the rest of the call.
    pub language: Option<&'a str>,
}

/// One-shot transcription. Implementations must be `Send + Sync` so the
/// engine can be moved into a worker thread.
pub trait Transcriber: Send + Sync {
    fn transcribe(&self, audio: &[f32], sample_rate: u32) -> Result<String, EngineError>;

    /// Like [`transcribe`](Self::transcribe), but decoded with the hints in
    /// `cx` — see [`DecodeContext`]. Backends with no prompt or language
    /// mechanism (transducers) ignore `cx`; the default implementation does
    /// exactly that.
    fn transcribe_with_context(
        &self,
        audio: &[f32],
        sample_rate: u32,
        _cx: DecodeContext<'_>,
    ) -> Result<String, EngineError> {
        self.transcribe(audio, sample_rate)
    }

    /// Decode `audio` into [`TimedPiece`]s, each offset from the start of the
    /// buffer. A backend that cannot report timing falls back to one piece
    /// covering everything, which is what the default does.
    fn transcribe_timed(
        &self,
        audio: &[f32],
        sample_rate: u32,
        cx: DecodeContext<'_>,
    ) -> Result<Vec<TimedPiece>, EngineError> {
        let text = self.transcribe_with_context(audio, sample_rate, cx)?;
        Ok(single_piece(text))
    }

    /// Longest buffer this backend should be handed in one call, in seconds.
    ///
    /// The post-transcription packer uses this to build the largest window a
    /// backend can actually use, which is the whole point of the second pass:
    /// whisper tops out at its 30 s mel window, a transducer does not.
    fn max_input_seconds(&self) -> f64 {
        WHISPER_MAX_INPUT_SECONDS
    }

    /// Backend name, for progress reports and the variants document.
    fn backend_label(&self) -> &'static str {
        "whisper.cpp"
    }
}

/// Wrap a whole-buffer decode as the single piece at offset zero. Empty text
/// yields no pieces at all, so callers never have to filter blanks.
pub fn single_piece(text: String) -> Vec<TimedPiece> {
    let text = text.trim();
    if text.is_empty() {
        return Vec::new();
    }
    vec![TimedPiece {
        offset: 0.0,
        text: text.to_string(),
    }]
}

/// Build the right engine for a given model id. Returns
/// [`EngineError::ModelMissing`] if files are not on disk and
/// [`EngineError::SherpaFeatureDisabled`] if the user picked a sherpa model
/// in a build that didn't include the `sherpa` feature.
pub fn build_transcriber(
    model_name: &str,
    models_dir: &Path,
    language: &str,
    num_threads: i32,
) -> Result<Box<dyn Transcriber>, EngineError> {
    let manager = ModelManager::new(models_dir);
    if !manager.is_downloaded(model_name) {
        return Err(EngineError::ModelMissing(model_name.into()));
    }
    let spec = manager.spec(model_name)?;
    match &spec.backend {
        Backend::WhisperCpp { .. } => {
            let engine =
                WhisperEngine::new(model_name, models_dir, language.to_string(), num_threads)?;
            Ok(Box::new(engine))
        }
        Backend::SherpaTransducer { .. } => build_sherpa(model_name, &manager, num_threads),
    }
}

#[cfg(feature = "sherpa")]
fn build_sherpa(
    model_name: &str,
    manager: &ModelManager,
    num_threads: i32,
) -> Result<Box<dyn Transcriber>, EngineError> {
    let paths = manager.sherpa_paths(model_name)?;
    let engine = SherpaTransducerEngine::new(&paths, num_threads)?;
    Ok(Box::new(engine))
}

#[cfg(not(feature = "sherpa"))]
fn build_sherpa(
    model_name: &str,
    _manager: &ModelManager,
    _num_threads: i32,
) -> Result<Box<dyn Transcriber>, EngineError> {
    Err(EngineError::SherpaFeatureDisabled {
        model: model_name.into(),
    })
}

// =====================================================================
// Whisper.cpp engine
// =====================================================================

pub struct EngineConfig {
    pub model_name: String,
    pub models_dir: PathBuf,
    pub language: String,
    pub num_threads: i32,
}

/// whisper.cpp keeps a mutable decoder state — the recognizer's `full()`
/// call takes `&mut self`. The rest of the app expects a `&self`
/// `Transcriber`, so we wrap state behind a `Mutex` and serialize calls.
/// Dictation is one-at-a-time so contention is fine.
pub struct WhisperEngine {
    config: EngineConfig,
    inner: Mutex<WhisperInner>,
}

struct WhisperInner {
    ctx: WhisperContext,
}

/// Back-compat alias — older code (and tests) call this `TranscriptionEngine`.
pub type TranscriptionEngine = WhisperEngine;

impl WhisperEngine {
    pub fn new(
        model_name: impl Into<String>,
        models_dir: impl Into<PathBuf>,
        language: impl Into<String>,
        num_threads: i32,
    ) -> Result<Self, EngineError> {
        let config = EngineConfig {
            model_name: model_name.into(),
            models_dir: models_dir.into(),
            language: language.into(),
            num_threads,
        };

        let manager = ModelManager::new(&config.models_dir);
        if !manager.is_downloaded(&config.model_name) {
            return Err(EngineError::ModelMissing(config.model_name.clone()));
        }
        let model_path = manager.whisper_file(&config.model_name)?;
        log::info!("whisper.cpp: loading {}", model_path.display());

        let ctx_params = WhisperContextParameters::default();
        let ctx =
            WhisperContext::new_with_params(model_path.to_string_lossy().as_ref(), ctx_params)?;

        Ok(Self {
            config,
            inner: Mutex::new(WhisperInner { ctx }),
        })
    }

    /// Construction helper used by tests + the file CLI. Defaults to 4 threads.
    pub fn with_threads_default(
        model_name: impl Into<String>,
        models_dir: impl Into<PathBuf>,
        language: impl Into<String>,
    ) -> Result<Self, EngineError> {
        Self::new(model_name, models_dir, language, 4)
    }
}

impl Transcriber for WhisperEngine {
    fn transcribe(&self, audio: &[f32], sample_rate: u32) -> Result<String, EngineError> {
        self.transcribe_with_context(audio, sample_rate, DecodeContext::default())
    }

    fn transcribe_with_context(
        &self,
        audio: &[f32],
        sample_rate: u32,
        cx: DecodeContext<'_>,
    ) -> Result<String, EngineError> {
        let pieces = self.decode(audio, sample_rate, cx)?;
        Ok(join_pieces(&pieces))
    }

    fn transcribe_timed(
        &self,
        audio: &[f32],
        sample_rate: u32,
        cx: DecodeContext<'_>,
    ) -> Result<Vec<TimedPiece>, EngineError> {
        self.decode(audio, sample_rate, cx)
    }

    fn max_input_seconds(&self) -> f64 {
        WHISPER_MAX_INPUT_SECONDS
    }

    fn backend_label(&self) -> &'static str {
        "whisper.cpp"
    }
}

impl WhisperEngine {
    /// The one decode path. Both [`Transcriber`] entry points go through it so
    /// the params can never drift between the live and the post pass.
    fn decode(
        &self,
        audio: &[f32],
        sample_rate: u32,
        cx: DecodeContext<'_>,
    ) -> Result<Vec<TimedPiece>, EngineError> {
        if audio.is_empty() {
            return Ok(Vec::new());
        }

        if sample_rate != SAMPLE_RATE {
            log::warn!(
                "whisper: got sample_rate={}, expected {}; audio quality may suffer",
                sample_rate,
                SAMPLE_RATE,
            );
        }

        let inner = self.inner.lock().expect("engine mutex");
        let mut state = inner.ctx.create_state()?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_n_threads(self.config.num_threads);
        params.set_translate(false);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        // Pin the language when one was chosen. Auto-detect (`None`) runs per
        // call, so a chunk with a stray loanword can flip the detected
        // language and make whisper *translate* instead of transcribe — pass
        // the language explicitly to stop that. The caller's override wins, so
        // a meeting can pin German while dictation stays on auto.
        let language = cx.language.unwrap_or(&self.config.language);
        if language != "auto" && !language.is_empty() {
            params.set_language(Some(language));
        } else {
            params.set_language(None);
        }

        // Temperature fallback. The previous config was a bare greedy decode
        // with no fallback — the most repetition/hallucination-prone setup.
        // With these (whisper.cpp's standard) thresholds a degenerate or
        // low-confidence decode is retried at a higher temperature instead of
        // being emitted as a loop.
        params.set_temperature(0.0);
        params.set_temperature_inc(0.2);
        params.set_entropy_thold(2.4);
        params.set_logprob_thold(-1.0);
        params.set_no_speech_thold(0.6);

        // Context carryover: prime the decoder with the previous segment's
        // tail so a chunk continuing a sentence stays coherent. Empty for
        // one-shot dictation, which has no prior.
        if !cx.prior.is_empty() {
            params.set_initial_prompt(cx.prior);
        }

        state.full(params, audio)?;

        // Whisper decides its own segment boundaries inside the window, and
        // reports where each one started in centiseconds. Keeping them apart
        // is what lets a 28 s window still produce utterance-level lines.
        let n = state.full_n_segments();
        let mut out = Vec::with_capacity(n as usize);
        for i in 0..n {
            let Some(seg) = state.get_segment(i) else {
                continue;
            };
            let text = seg.to_str_lossy()?;
            let text = text.trim();
            if text.is_empty() {
                continue;
            }
            out.push(TimedPiece {
                offset: (seg.start_timestamp().max(0) as f64) / 100.0,
                text: text.to_string(),
            });
        }
        Ok(out)
    }
}

/// Flatten pieces back into the one string the live path expects.
pub fn join_pieces(pieces: &[TimedPiece]) -> String {
    let mut out = String::new();
    for piece in pieces {
        let text = piece.text.trim();
        if text.is_empty() {
            continue;
        }
        if !out.is_empty() && !out.ends_with(' ') {
            out.push(' ');
        }
        out.push_str(text);
    }
    out.trim().to_string()
}

// =====================================================================
// Sherpa-onnx offline transducer engine
// =====================================================================

#[cfg(feature = "sherpa")]
pub struct SherpaTransducerEngine {
    inner: Mutex<sherpa_onnx::OfflineRecognizer>,
    num_threads: i32,
}

#[cfg(feature = "sherpa")]
impl SherpaTransducerEngine {
    pub fn new(
        paths: &super::model_manager::SherpaPaths,
        num_threads: i32,
    ) -> Result<Self, EngineError> {
        use sherpa_onnx::{
            OfflineRecognizer, OfflineRecognizerConfig, OfflineTransducerModelConfig,
        };

        log::info!(
            "sherpa-onnx: loading encoder={} decoder={} joiner={} tokens={}",
            paths.encoder.display(),
            paths.decoder.display(),
            paths.joiner.display(),
            paths.tokens.display(),
        );

        let mut cfg = OfflineRecognizerConfig::default();
        cfg.model_config.transducer = OfflineTransducerModelConfig {
            encoder: Some(paths.encoder.to_string_lossy().into_owned()),
            decoder: Some(paths.decoder.to_string_lossy().into_owned()),
            joiner: Some(paths.joiner.to_string_lossy().into_owned()),
        };
        cfg.model_config.tokens = Some(paths.tokens.to_string_lossy().into_owned());
        // NeMo transducers normalise features differently from the k2 ones
        // sherpa defaults to. The metadata in the exported model usually says
        // so, but the model type is what the decoder actually keys off, and
        // sherpa's own example for this exact Parakeet build sets it — so set
        // it rather than rely on the probe.
        cfg.model_config.model_type = Some("nemo_transducer".into());
        cfg.model_config.provider = Some("cpu".into());
        cfg.model_config.num_threads = num_threads;
        cfg.model_config.debug = false;

        let rec = OfflineRecognizer::create(&cfg)
            .ok_or_else(|| EngineError::Sherpa("OfflineRecognizer::create returned None".into()))?;
        Ok(Self {
            inner: Mutex::new(rec),
            num_threads,
        })
    }
}

#[cfg(feature = "sherpa")]
impl Transcriber for SherpaTransducerEngine {
    fn transcribe(&self, audio: &[f32], sample_rate: u32) -> Result<String, EngineError> {
        Ok(join_pieces(&self.decode(audio, sample_rate)?))
    }

    fn transcribe_timed(
        &self,
        audio: &[f32],
        sample_rate: u32,
        _cx: DecodeContext<'_>,
    ) -> Result<Vec<TimedPiece>, EngineError> {
        self.decode(audio, sample_rate)
    }

    fn max_input_seconds(&self) -> f64 {
        SHERPA_MAX_INPUT_SECONDS
    }

    fn backend_label(&self) -> &'static str {
        "sherpa-onnx"
    }
}

#[cfg(feature = "sherpa")]
impl SherpaTransducerEngine {
    fn decode(&self, audio: &[f32], sample_rate: u32) -> Result<Vec<TimedPiece>, EngineError> {
        if audio.is_empty() {
            return Ok(Vec::new());
        }
        let _ = self.num_threads;
        let result = {
            let rec = self.inner.lock().expect("engine mutex");
            let stream = rec.create_stream();
            stream.accept_waveform(sample_rate as i32, audio);
            rec.decode(&stream);
            stream.get_result()
        };
        let Some(result) = result else {
            return Ok(Vec::new());
        };
        Ok(split_tokens(
            &result.text,
            &result.tokens,
            result.timestamps.as_deref(),
        ))
    }
}

/// Longest silence inside one transducer piece before it is broken in two.
/// Below this a gap is just a breath, not a sentence boundary.
const TOKEN_SPLIT_GAP_SECONDS: f32 = 0.6;

/// Shortest piece the token splitter will emit, so a stutter mid-sentence
/// does not produce a transcript line holding two words.
const TOKEN_MIN_PIECE_SECONDS: f32 = 2.0;

/// Turn a transducer's per-token timestamps into timed pieces, cutting where
/// the speaker paused.
///
/// The tokens are sentencepiece, so `▁` marks a word start and detokenising is
/// "replace the marker with a space". That reconstruction has to be exact or
/// the transcript would silently differ from what the model actually said, so
/// the result is checked against the model's own `text` and the whole thing
/// falls back to a single untimed piece when it does not line up.
///
/// Not gated behind the `sherpa` feature even though only that backend feeds
/// it: it is plain string handling, and keeping it buildable everywhere is
/// what lets it be tested on a machine that cannot compile the native lib.
pub fn split_tokens(text: &str, tokens: &[String], timestamps: Option<&[f32]>) -> Vec<TimedPiece> {
    let Some(times) = timestamps else {
        return single_piece(text.to_string());
    };
    if tokens.is_empty() || times.len() != tokens.len() {
        return single_piece(text.to_string());
    }

    let mut pieces: Vec<TimedPiece> = Vec::new();
    let mut current = String::new();
    let mut start = times[0];
    let mut previous = times[0];

    for (token, &at) in tokens.iter().zip(times) {
        let long_enough = previous - start >= TOKEN_MIN_PIECE_SECONDS;
        if at - previous >= TOKEN_SPLIT_GAP_SECONDS && long_enough && !current.trim().is_empty() {
            pieces.push(TimedPiece {
                offset: start as f64,
                text: current.trim().to_string(),
            });
            current.clear();
            start = at;
        }
        match token.strip_prefix('\u{2581}') {
            Some(rest) => {
                if !current.is_empty() {
                    current.push(' ');
                }
                current.push_str(rest);
            },
            None => current.push_str(token),
        }
        previous = at;
    }
    if !current.trim().is_empty() {
        pieces.push(TimedPiece {
            offset: start as f64,
            text: current.trim().to_string(),
        });
    }

    // Detokenisation must round-trip, or we are inventing a transcript.
    if squash(&join_pieces(&pieces)) != squash(text) {
        log::debug!("sherpa: token split did not round-trip; using the flat result");
        return single_piece(text.to_string());
    }
    pieces
}

/// Whitespace-insensitive comparison key.
fn squash(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

// =====================================================================
// CLI helper for tests
// =====================================================================

/// CLI helper used by tests: load a WAV and transcribe it. Mono PCM16 or
/// PCM32 at any sample rate (downmixed to mono; whisper.cpp resamples
/// internally if needed).
pub fn transcribe_file(
    wav_path: &Path,
    model_name: &str,
    models_dir: &Path,
    language: &str,
) -> Result<String, EngineError> {
    let mut reader = hound::WavReader::open(wav_path)
        .map_err(|e| EngineError::ModelMissing(format!("read wav {}: {e}", wav_path.display())))?;
    let spec = reader.spec();
    let channels = spec.channels as usize;

    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .filter_map(Result::ok)
                .map(|s| s as f32 / max)
                .collect()
        }
        hound::SampleFormat::Float => reader.samples::<f32>().filter_map(Result::ok).collect(),
    };

    let mono = downmix(&interleaved, channels);
    let engine = build_transcriber(model_name, models_dir, language, 4)?;
    engine.transcribe(&mono, spec.sample_rate)
}

fn downmix(interleaved: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return interleaved.to_vec();
    }
    interleaved
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(words: &[&str]) -> Vec<String> {
        words.iter().map(|w| w.to_string()).collect()
    }

    #[test]
    fn joining_pieces_puts_exactly_one_space_between_them() {
        let pieces = vec![
            TimedPiece {
                offset: 0.0,
                text: " Guten Morgen. ".into(),
            },
            TimedPiece {
                offset: 3.0,
                text: "Wie geht es?".into(),
            },
            TimedPiece {
                offset: 5.0,
                text: "   ".into(),
            },
        ];
        assert_eq!(join_pieces(&pieces), "Guten Morgen. Wie geht es?");
        assert_eq!(join_pieces(&[]), "");
    }

    #[test]
    fn empty_text_yields_no_piece_at_all() {
        assert!(single_piece("   ".into()).is_empty());
        assert_eq!(single_piece("hallo".into()).len(), 1);
    }

    /// Without timestamps there is nothing to cut on, so the flat result has
    /// to survive intact.
    #[test]
    fn missing_timestamps_fall_back_to_one_piece() {
        let t = toks(&["\u{2581}hallo", "\u{2581}welt"]);
        let out = split_tokens("hallo welt", &t, None);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].text, "hallo welt");
    }

    /// A timestamp array that does not line up with the tokens means we
    /// cannot trust either — keep the model's own text rather than guess.
    #[test]
    fn mismatched_timestamps_fall_back_to_one_piece() {
        let t = toks(&["\u{2581}hallo", "\u{2581}welt"]);
        let out = split_tokens("hallo welt", &t, Some(&[0.0]));
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].text, "hallo welt");
    }

    #[test]
    fn a_long_pause_starts_a_new_piece() {
        // Continuous speech to 2.2 s (tokens land every ~0.5 s), then nothing
        // until 5.0 s. Only the real pause may cut.
        let t = toks(&[
            "\u{2581}das",
            "\u{2581}ist",
            "\u{2581}ein",
            "\u{2581}test",
            "\u{2581}und",
            "\u{2581}weiter",
        ]);
        let times = [0.0, 0.5, 1.2, 2.2, 5.0, 5.4];
        let out = split_tokens("das ist ein test und weiter", &t, Some(&times));
        assert_eq!(out.len(), 2, "{out:?}");
        assert_eq!(out[0].text, "das ist ein test");
        assert_eq!(out[1].text, "und weiter");
        assert!((out[0].offset - 0.0).abs() < 1e-6);
        assert!((out[1].offset - 5.0).abs() < 1e-6);
    }

    /// A pause in the first couple of seconds is a hesitation, not a sentence
    /// break — cutting there would leave a transcript line holding two words.
    #[test]
    fn a_pause_too_early_does_not_split() {
        let t = toks(&["\u{2581}äh", "\u{2581}also", "\u{2581}gut"]);
        let times = [0.0, 1.0, 1.6];
        let out = split_tokens("äh also gut", &t, Some(&times));
        assert_eq!(out.len(), 1, "{out:?}");
    }

    /// Normal speech has no gaps wide enough to cut on, so a whole window
    /// stays one piece.
    #[test]
    fn continuous_speech_is_not_chopped_up() {
        let t = toks(&["\u{2581}eins", "\u{2581}zwei", "\u{2581}drei", "\u{2581}vier"]);
        let times = [0.0, 0.4, 0.9, 1.5];
        let out = split_tokens("eins zwei drei vier", &t, Some(&times));
        assert_eq!(out.len(), 1, "{out:?}");
    }

    /// Word-continuation tokens carry no marker and must not gain a space.
    #[test]
    fn subword_tokens_are_joined_without_a_space() {
        let t = toks(&["\u{2581}Auf", "nahme", "\u{2581}läuft"]);
        let times = [0.0, 0.3, 0.6];
        let out = split_tokens("Aufnahme läuft", &t, Some(&times));
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].text, "Aufnahme läuft");
    }

    /// The safety net: if detokenising does not reproduce what the model
    /// said, the split is thrown away rather than shipped.
    #[test]
    fn a_split_that_does_not_round_trip_is_discarded() {
        let t = toks(&["\u{2581}hallo", "\u{2581}welt"]);
        let times = [0.0, 4.0];
        // The model's own text disagrees with the tokens.
        let out = split_tokens("etwas ganz anderes", &t, Some(&times));
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].text, "etwas ganz anderes");
    }

}
