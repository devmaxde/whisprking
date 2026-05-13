//! Speech-to-text engines.
//!
//! Two backends share a single [`Transcriber`] trait:
//!
//! - [`WhisperEngine`] — whisper.cpp via `whisper-rs`. Multilingual; always
//!   compiled in.
//! - [`SherpaTransducerEngine`] — sherpa-onnx offline transducer (NeMo
//!   Parakeet). English-only; gated behind the `sherpa` feature.
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

/// One-shot transcription. Implementations must be `Send + Sync` so the
/// engine can be moved into a worker thread.
pub trait Transcriber: Send + Sync {
    fn transcribe(&self, audio: &[f32], sample_rate: u32) -> Result<String, EngineError>;
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
        if audio.is_empty() {
            return Ok(String::new());
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
        if self.config.language != "auto" && !self.config.language.is_empty() {
            params.set_language(Some(&self.config.language));
        } else {
            params.set_language(None);
        }

        state.full(params, audio)?;

        let n = state.full_n_segments();
        let mut out = String::new();
        for i in 0..n {
            if let Some(seg) = state.get_segment(i) {
                let text = seg.to_str_lossy()?;
                out.push_str(&text);
            }
        }
        Ok(out.trim().to_string())
    }
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
        if audio.is_empty() {
            return Ok(String::new());
        }
        let _ = self.num_threads;
        let rec = self.inner.lock().expect("engine mutex");
        let stream = rec.create_stream();
        stream.accept_waveform(sample_rate as i32, audio);
        rec.decode(&stream);
        let text = stream.get_result().map(|r| r.text).unwrap_or_default();
        Ok(text.trim().to_string())
    }
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
