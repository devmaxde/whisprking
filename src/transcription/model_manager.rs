//! Catalog, download, and on-disk layout for STT models.
//!
//! Two backends are supported, selected per-model via [`Backend`]:
//!
//! - [`Backend::WhisperCpp`] — single `.bin` GGML file streamed straight to
//!   disk (whisper.cpp). Always built in.
//! - [`Backend::SherpaTransducer`] — sherpa-onnx prebuilt NeMo/Parakeet
//!   transducer model packaged as `.tar.bz2`. Downloaded and extracted into
//!   `<models_dir>/<model_name>/`. Requires the `sherpa` build feature to
//!   actually transcribe (download works without it).

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("unknown model: {0}")]
    Unknown(String),
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("io error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("model file missing: {0}")]
    FileMissing(PathBuf),
    #[error("archive extract error: {0}")]
    Extract(String),
    #[error("wrong backend for {model}: expected {expected}, got {actual}")]
    WrongBackend {
        model: String,
        expected: &'static str,
        actual: &'static str,
    },
}

/// Per-model storage + decoding strategy.
#[derive(Debug, Clone)]
pub enum Backend {
    /// Single `.bin` GGML file. `file_name` is the on-disk name.
    WhisperCpp { file_name: &'static str },
    /// sherpa-onnx prebuilt offline transducer: tar.bz2 archive with
    /// encoder/decoder/joiner/tokens. `inner_dir` is the directory inside
    /// the tarball; we strip it on extract so the four files land in
    /// `<models_dir>/<model_name>/` directly.
    SherpaTransducer {
        inner_dir: &'static str,
        encoder: &'static str,
        decoder: &'static str,
        joiner: &'static str,
        tokens: &'static str,
    },
}

impl Backend {
    pub fn kind(&self) -> &'static str {
        match self {
            Backend::WhisperCpp { .. } => "whisper.cpp",
            Backend::SherpaTransducer { .. } => "sherpa-transducer",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub display_name: &'static str,
    /// Either a single `.bin` (whisper) or a `.tar.bz2` archive (sherpa).
    pub url: &'static str,
    pub size_mb: u32,
    pub multilingual: bool,
    pub backend: Backend,
}

/// Paths inside a downloaded sherpa transducer model.
#[derive(Debug, Clone)]
pub struct SherpaPaths {
    pub encoder: PathBuf,
    pub decoder: PathBuf,
    pub joiner: PathBuf,
    pub tokens: PathBuf,
}

/// `(downloaded, total)` callback for download UI.
pub type ProgressCallback = Box<dyn FnMut(u64, u64) + Send>;

/// All built-in models, keyed by short id (e.g. `whisper-turbo`).
pub fn available_models() -> &'static HashMap<&'static str, ModelSpec> {
    static MODELS: OnceLock<HashMap<&'static str, ModelSpec>> = OnceLock::new();
    MODELS.get_or_init(|| {
        let mut m = HashMap::new();
        // Three sizes of the same large-v3 family, because the trade-off is
        // real in both directions: q5_0 is a 5-bit quantization of the turbo
        // weights — it loads faster and costs a third of the memory, and it
        // does lose accuracy, mostly on proper nouns and numbers. The f16
        // files are what whisper.cpp calls "full"; on Apple Silicon they run
        // straight through Metal at no meaningful penalty per audio second,
        // so the price is disk and RAM, not speed.
        m.insert(
            "whisper-turbo",
            ModelSpec {
                display_name: "Whisper large-v3 Turbo q5 — klein und schnell (multilingual, ~574MB)",
                url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin",
                size_mb: 574,
                multilingual: true,
                backend: Backend::WhisperCpp {
                    file_name: "ggml-large-v3-turbo-q5_0.bin",
                },
            },
        );
        m.insert(
            "whisper-turbo-f16",
            ModelSpec {
                display_name: "Whisper large-v3 Turbo f16 — unquantisiert (multilingual, ~1,6GB)",
                url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin",
                size_mb: 1624,
                multilingual: true,
                backend: Backend::WhisperCpp {
                    file_name: "ggml-large-v3-turbo.bin",
                },
            },
        );
        // Turbo is large-v3 with the decoder cut from 32 layers to 4. That is
        // where its speed comes from, and also the only thing this model does
        // better: the full large-v3 decodes several times slower, but nothing
        // in the whisper family transcribes German more accurately.
        m.insert(
            "whisper-large-v3",
            ModelSpec {
                display_name: "Whisper large-v3 f16 — höchste Genauigkeit, langsam (multilingual, ~3,1GB)",
                url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin",
                size_mb: 3095,
                multilingual: true,
                backend: Backend::WhisperCpp {
                    file_name: "ggml-large-v3.bin",
                },
            },
        );
        m.insert(
            "whisper-base",
            ModelSpec {
                display_name: "Whisper base (multilingual, ~142MB)",
                url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
                size_mb: 142,
                multilingual: true,
                backend: Backend::WhisperCpp {
                    file_name: "ggml-base.bin",
                },
            },
        );
        m.insert(
            "whisper-small",
            ModelSpec {
                display_name: "Whisper small (multilingual, ~466MB)",
                url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
                size_mb: 466,
                multilingual: true,
                backend: Backend::WhisperCpp {
                    file_name: "ggml-small.bin",
                },
            },
        );
        m.insert(
            "whisper-tiny",
            ModelSpec {
                display_name: "Whisper tiny (multilingual, ~75MB)",
                url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin",
                size_mb: 75,
                multilingual: true,
                backend: Backend::WhisperCpp {
                    file_name: "ggml-tiny.bin",
                },
            },
        );
        // NVIDIA NeMo Parakeet TDT 0.6B v3 int8 — multilingual: 25 European
        // languages including German (German 5.04% WER on FLEURS, better than
        // its own average). The v2 model this catalog entry was cloned from
        // was English-only; the "English" label was carried over by mistake.
        // sherpa-onnx tar.bz2 from k2-fsa GitHub releases.
        m.insert(
            "parakeet-v3",
            ModelSpec {
                display_name: "NVIDIA Parakeet TDT 0.6b v3 int8 — recommended for German (25 languages, ~464MB, needs a `sherpa` build)",
                url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2",
                size_mb: 464,
                multilingual: true,
                backend: Backend::SherpaTransducer {
                    inner_dir: "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
                    encoder: "encoder.int8.onnx",
                    decoder: "decoder.int8.onnx",
                    joiner: "joiner.int8.onnx",
                    tokens: "tokens.txt",
                },
            },
        );
        m
    })
}

/// Alias so callers can use the familiar name.
pub use available_models as AVAILABLE_MODELS;

/// Silero VAD (MIT) — a single ~2 MB `.onnx` used for speech-boundary
/// segmentation, orthogonal to the STT backend. It is deliberately *not* a
/// [`ModelSpec`]/[`Backend`] entry: it is never selected as a transcription
/// model, it is a small fixed asset the meeting pipeline fetches once and
/// reuses. Only usable at runtime in `sherpa`-feature builds (the VAD lives
/// in the sherpa-onnx native lib); the download itself needs no feature.
pub const VAD_MODEL_URL: &str =
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx";
const VAD_DIR: &str = "silero_vad";
const VAD_FILE: &str = "silero_vad.onnx";

/// Speaker diarization needs two models, and like the VAD they are fixed
/// assets rather than [`ModelSpec`] entries: neither is ever selected as a
/// transcription model, and one is useless without the other.
///
/// * **pyannote segmentation 3.0** — finds speech and marks where the voice
///   changes. Shipped as a `.tar.bz2` with the same layout as the transducer
///   packs, so it is extracted the same way.
/// * **3D-Speaker CAM++** — turns a stretch of speech into a vector that can
///   be clustered. Trained on Chinese *and* English data, which is the widest
///   coverage among the small embedding models sherpa-onnx publishes; the
///   embedding compares voices rather than words, so the language of the
///   meeting barely matters.
///
/// Both are only usable at runtime in `sherpa`-feature builds. Together they
/// are around [`DIARIZE_SIZE_MB`] on disk.
pub const DIARIZE_SEGMENTATION_URL: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2";
const DIARIZE_SEGMENTATION_DIR: &str = "pyannote_segmentation";
const DIARIZE_SEGMENTATION_INNER: &str = "sherpa-onnx-pyannote-segmentation-3-0";
const DIARIZE_SEGMENTATION_FILE: &str = "model.onnx";
// The release tag really is spelled "recongition" upstream.
pub const DIARIZE_EMBEDDING_URL: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx";
const DIARIZE_EMBEDDING_DIR: &str = "speaker_embedding";
const DIARIZE_EMBEDDING_FILE: &str = "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx";

/// Combined on-disk size of the two diarization models, for the UI.
pub const DIARIZE_SIZE_MB: u32 = 34;

/// The pair of models [`crate::transcription::diarize`] loads.
#[derive(Debug, Clone)]
pub struct DiarizePaths {
    pub segmentation: PathBuf,
    pub embedding: PathBuf,
}

pub struct ModelManager {
    models_dir: PathBuf,
}

impl ModelManager {
    pub fn new(models_dir: impl Into<PathBuf>) -> Self {
        let models_dir = models_dir.into();
        let _ = std::fs::create_dir_all(&models_dir);
        Self { models_dir }
    }

    pub fn model_path(&self, model_name: &str) -> PathBuf {
        self.models_dir.join(model_name)
    }

    pub fn spec(&self, model_name: &str) -> Result<&'static ModelSpec, ModelError> {
        available_models()
            .get(model_name)
            .ok_or_else(|| ModelError::Unknown(model_name.into()))
    }

    /// Path to the single whisper `.bin` file. Errors if the model is not a
    /// whisper.cpp model.
    pub fn whisper_file(&self, model_name: &str) -> Result<PathBuf, ModelError> {
        let spec = self.spec(model_name)?;
        match &spec.backend {
            Backend::WhisperCpp { file_name } => Ok(self.model_path(model_name).join(file_name)),
            other => Err(ModelError::WrongBackend {
                model: model_name.into(),
                expected: "whisper.cpp",
                actual: other.kind(),
            }),
        }
    }

    /// Paths to encoder/decoder/joiner/tokens for a sherpa transducer model.
    /// Errors if the model is not a sherpa transducer.
    pub fn sherpa_paths(&self, model_name: &str) -> Result<SherpaPaths, ModelError> {
        let spec = self.spec(model_name)?;
        match &spec.backend {
            Backend::SherpaTransducer {
                encoder,
                decoder,
                joiner,
                tokens,
                ..
            } => {
                let dir = self.model_path(model_name);
                Ok(SherpaPaths {
                    encoder: dir.join(encoder),
                    decoder: dir.join(decoder),
                    joiner: dir.join(joiner),
                    tokens: dir.join(tokens),
                })
            }
            other => Err(ModelError::WrongBackend {
                model: model_name.into(),
                expected: "sherpa-transducer",
                actual: other.kind(),
            }),
        }
    }

    pub fn is_downloaded(&self, model_name: &str) -> bool {
        let Ok(spec) = self.spec(model_name) else {
            return false;
        };
        match &spec.backend {
            Backend::WhisperCpp { .. } => self
                .whisper_file(model_name)
                .map(|p| p.is_file())
                .unwrap_or(false),
            Backend::SherpaTransducer { .. } => self
                .sherpa_paths(model_name)
                .map(|p| {
                    p.encoder.is_file()
                        && p.decoder.is_file()
                        && p.joiner.is_file()
                        && p.tokens.is_file()
                })
                .unwrap_or(false),
        }
    }

    pub fn list_downloaded(&self) -> Vec<&'static str> {
        available_models()
            .keys()
            .copied()
            .filter(|n| self.is_downloaded(n))
            .collect()
    }

    /// Download (and, for sherpa models, extract) the model. Idempotent.
    /// Returns the model directory.
    pub fn download(
        &self,
        model_name: &str,
        mut progress: Option<ProgressCallback>,
    ) -> Result<PathBuf, ModelError> {
        if self.is_downloaded(model_name) {
            return Ok(self.model_path(model_name));
        }
        let spec = self.spec(model_name)?;
        let dir = self.model_path(model_name);
        std::fs::create_dir_all(&dir).map_err(|source| ModelError::Io {
            path: dir.clone(),
            source,
        })?;

        match &spec.backend {
            Backend::WhisperCpp { file_name } => {
                let dest = dir.join(file_name);
                let tmp = dir.join(format!("{file_name}.partial"));
                download_file(spec.url, &tmp, progress.as_mut())?;
                std::fs::rename(&tmp, &dest)
                    .map_err(|source| ModelError::Io { path: dest, source })?;
            }
            Backend::SherpaTransducer { inner_dir, .. } => {
                let archive = dir.join("archive.tar.bz2.partial");
                download_file(spec.url, &archive, progress.as_mut())?;
                extract_tar_bz2_stripping(&archive, &dir, inner_dir)?;
                let _ = std::fs::remove_file(&archive);
            }
        }
        Ok(dir)
    }

    /// On-disk path to the Silero VAD model (may not exist yet).
    pub fn vad_model_path(&self) -> PathBuf {
        self.models_dir.join(VAD_DIR).join(VAD_FILE)
    }

    pub fn is_vad_downloaded(&self) -> bool {
        self.vad_model_path().is_file()
    }

    /// Download the Silero VAD model if it is not already present. Idempotent;
    /// returns the model path. Safe to call on any build — the download needs
    /// no native feature.
    pub fn ensure_vad_model(
        &self,
        mut progress: Option<ProgressCallback>,
    ) -> Result<PathBuf, ModelError> {
        let dest = self.vad_model_path();
        if dest.is_file() {
            return Ok(dest);
        }
        let dir = dest.parent().unwrap_or(&self.models_dir).to_path_buf();
        std::fs::create_dir_all(&dir).map_err(|source| ModelError::Io {
            path: dir.clone(),
            source,
        })?;
        let tmp = dir.join(format!("{VAD_FILE}.partial"));
        download_file(VAD_MODEL_URL, &tmp, progress.as_mut())?;
        std::fs::rename(&tmp, &dest).map_err(|source| ModelError::Io {
            path: dest.clone(),
            source,
        })?;
        Ok(dest)
    }

    /// On-disk paths of the two diarization models (they may not exist yet).
    pub fn diarize_paths(&self) -> DiarizePaths {
        DiarizePaths {
            segmentation: self
                .models_dir
                .join(DIARIZE_SEGMENTATION_DIR)
                .join(DIARIZE_SEGMENTATION_FILE),
            embedding: self
                .models_dir
                .join(DIARIZE_EMBEDDING_DIR)
                .join(DIARIZE_EMBEDDING_FILE),
        }
    }

    pub fn is_diarize_downloaded(&self) -> bool {
        let paths = self.diarize_paths();
        paths.segmentation.is_file() && paths.embedding.is_file()
    }

    /// Download whichever of the two diarization models is missing.
    /// Idempotent. `progress` is called for each file in turn, so it reports
    /// two downloads, not one combined bar.
    ///
    /// Extraction of the segmentation pack needs the `sherpa` feature (that is
    /// where the `.tar.bz2` support lives), which is also the only kind of
    /// build that can run diarization at all.
    pub fn ensure_diarize_models(
        &self,
        mut progress: Option<ProgressCallback>,
    ) -> Result<DiarizePaths, ModelError> {
        let paths = self.diarize_paths();

        if !paths.segmentation.is_file() {
            let dir = self.models_dir.join(DIARIZE_SEGMENTATION_DIR);
            std::fs::create_dir_all(&dir).map_err(|source| ModelError::Io {
                path: dir.clone(),
                source,
            })?;
            let archive = dir.join("archive.tar.bz2.partial");
            download_file(DIARIZE_SEGMENTATION_URL, &archive, progress.as_mut())?;
            extract_tar_bz2_stripping(&archive, &dir, DIARIZE_SEGMENTATION_INNER)?;
            let _ = std::fs::remove_file(&archive);
            if !paths.segmentation.is_file() {
                return Err(ModelError::FileMissing(paths.segmentation.clone()));
            }
        }

        if !paths.embedding.is_file() {
            let dir = self.models_dir.join(DIARIZE_EMBEDDING_DIR);
            std::fs::create_dir_all(&dir).map_err(|source| ModelError::Io {
                path: dir.clone(),
                source,
            })?;
            let tmp = dir.join(format!("{DIARIZE_EMBEDDING_FILE}.partial"));
            download_file(DIARIZE_EMBEDDING_URL, &tmp, progress.as_mut())?;
            std::fs::rename(&tmp, &paths.embedding).map_err(|source| ModelError::Io {
                path: paths.embedding.clone(),
                source,
            })?;
        }

        Ok(paths)
    }
}

fn download_file(
    url: &str,
    dest: &Path,
    mut progress: Option<&mut ProgressCallback>,
) -> Result<(), ModelError> {
    let mut response = reqwest::blocking::get(url)?.error_for_status()?;
    let total = response.content_length().unwrap_or(0);

    let mut file = File::create(dest).map_err(|source| ModelError::Io {
        path: dest.to_path_buf(),
        source,
    })?;

    let mut downloaded = 0u64;
    let mut buf = [0u8; 128 * 1024];
    loop {
        let n = response.read(&mut buf).map_err(|source| ModelError::Io {
            path: dest.to_path_buf(),
            source,
        })?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).map_err(|source| ModelError::Io {
            path: dest.to_path_buf(),
            source,
        })?;
        downloaded += n as u64;
        if let Some(cb) = progress.as_mut() {
            cb(downloaded, total);
        }
    }
    Ok(())
}

/// Extract a `.tar.bz2` archive into `dest_dir`, stripping the leading
/// `inner_dir/` path from each entry. Only available when the `sherpa`
/// feature is enabled (which pulls in `tar` + `bzip2`).
#[cfg(feature = "sherpa")]
fn extract_tar_bz2_stripping(
    archive_path: &Path,
    dest_dir: &Path,
    inner_dir: &str,
) -> Result<(), ModelError> {
    use bzip2::read::BzDecoder;
    use tar::Archive;

    let file = File::open(archive_path).map_err(|source| ModelError::Io {
        path: archive_path.to_path_buf(),
        source,
    })?;
    let bz = BzDecoder::new(file);
    let mut archive = Archive::new(bz);

    let strip_prefix = format!("{inner_dir}/");
    for entry in archive
        .entries()
        .map_err(|e| ModelError::Extract(e.to_string()))?
    {
        let mut entry = entry.map_err(|e| ModelError::Extract(e.to_string()))?;
        let path = entry
            .path()
            .map_err(|e| ModelError::Extract(e.to_string()))?
            .into_owned();
        let path_str = path.to_string_lossy();
        let rel = path_str
            .strip_prefix(&strip_prefix)
            .unwrap_or(path_str.as_ref());
        if rel.is_empty() {
            continue;
        }
        let out = dest_dir.join(rel);
        if entry.header().entry_type().is_dir() {
            std::fs::create_dir_all(&out).map_err(|source| ModelError::Io {
                path: out.clone(),
                source,
            })?;
            continue;
        }
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent).map_err(|source| ModelError::Io {
                path: parent.to_path_buf(),
                source,
            })?;
        }
        entry
            .unpack(&out)
            .map_err(|e| ModelError::Extract(e.to_string()))?;
    }
    Ok(())
}

#[cfg(not(feature = "sherpa"))]
fn extract_tar_bz2_stripping(
    _archive_path: &Path,
    _dest_dir: &Path,
    _inner_dir: &str,
) -> Result<(), ModelError> {
    Err(ModelError::Extract(
        "this build was compiled without the `sherpa` feature — \
         rebuild with `cargo build --features sherpa` to install Parakeet \
         (or other tar.bz2 sherpa-onnx models)"
            .into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A whisper entry whose `file_name` does not match the file its URL
    /// points at downloads fine and is then reported as "nicht installiert"
    /// forever, because [`ModelManager::is_downloaded`] looks for the name the
    /// catalog claims. Cheap to get wrong when adding a model, invisible until
    /// someone waits out a 3 GB download.
    #[test]
    fn whisper_file_names_match_their_urls() {
        for (id, spec) in available_models() {
            let Backend::WhisperCpp { file_name } = &spec.backend else {
                continue;
            };
            let from_url = spec.url.rsplit('/').next().unwrap_or_default();
            assert_eq!(&from_url, file_name, "{id}");
        }
    }

    /// Two models sharing a `file_name` would be fine on disk (one directory
    /// each) but means one of them is a copy-paste of the other.
    #[test]
    fn models_are_distinct() {
        let mut urls: Vec<&str> = available_models().values().map(|s| s.url).collect();
        let total = urls.len();
        urls.sort_unstable();
        urls.dedup();
        assert_eq!(urls.len(), total, "duplicate model url in the catalog");
    }
}
