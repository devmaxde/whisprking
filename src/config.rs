//! JSON-backed configuration.
//!
//! Config lives at `<data_dir>/config.json`. Missing fields are filled in
//! from the defaults via deep merge, so adding a new key in code does not
//! invalidate existing configs.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const DEFAULT_DATA_DIR: &str = "~/WhisprKing";
pub const CONFIG_FILENAME: &str = "config.json";
const SUBDIRS: &[&str] = &["models", "transcripts", "audio"];

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("io error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid json at {path}: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub data_dir: String,
    pub dictation: DictationConfig,
    pub meeting: MeetingConfig,
    pub ai_postprocess: AiPostprocessConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DictationConfig {
    pub hotkey: String,
    pub mode: String,
    pub model: String,
    pub language: String,
    pub insert_method: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingConfig {
    /// Which side(s) to record: `"mic"`, `"system"`, or `"both"`.
    ///
    /// The legacy value `"mix"` is accepted and treated as `"both"`.
    pub audio_source: String,
    /// cpal-side input device name. `None` → use the system default input.
    ///
    /// This is the user's choice and only the user changes it. Nothing in
    /// the capture path is allowed to write here — doing exactly that is
    /// what used to silently hijack the microphone selection.
    #[serde(default)]
    pub input_device: Option<String>,
    pub model: String,
    /// Language spoken in meetings. Three cases:
    ///
    /// - `""` (default) — follow [`DictationConfig::language`], i.e. whatever
    ///   the shared engine was built with. Preserves pre-existing behaviour.
    /// - `"auto"` — detect per segment, independently of dictation.
    /// - an ISO code like `"de"` — pin it.
    ///
    /// Worth its own setting because a meeting is decoded segment-by-segment
    /// and `auto` re-detects on every one of them, so a single English
    /// loanword in a German sentence can flip whisper into *translating* the
    /// rest of the call. Pinning here stops that without forcing the same
    /// choice on dictation.
    #[serde(default)]
    pub language: String,
    pub save_audio: bool,
    pub chunk_duration_seconds: u32,
    /// Label each transcript line with which side spoke.
    #[serde(default = "default_true")]
    pub label_speakers: bool,
    /// Second transcription pass over the saved audio once a meeting ends.
    #[serde(default)]
    pub post_transcribe: PostTranscribeConfig,
    /// Leftover state from the BlackHole era. Read once at startup so the
    /// routing it describes can be torn down, then cleared for good; see
    /// [`crate::audio::legacy`].
    #[serde(default, skip_serializing_if = "BlackHoleSetup::is_empty")]
    pub blackhole: BlackHoleSetup,
}

fn default_true() -> bool {
    true
}

/// What happens after the recording stops.
///
/// The live transcript is decoded under real-time pressure, one short span at
/// a time. This runs the same audio again with none of that: the largest
/// window each backend accepts, and as many models as the user wants, in
/// parallel. See [`crate::transcription::post`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostTranscribeConfig {
    pub enabled: bool,
    /// Models to run over the recording, in preference order. The first one
    /// that produces a transcript is what gets used if the variants cannot be
    /// merged. Anything not downloaded is skipped with a note.
    pub models: Vec<String>,
    /// Merge the variants into one transcript with the configured LLM. Off, or
    /// with no provider, the best single variant becomes the result.
    pub reconcile: bool,
    /// Keep the per-track WAVs after the pass. They are what makes a re-run
    /// possible, and they are large — roughly 115 MB per hour per track.
    pub keep_audio: bool,
}

impl Default for PostTranscribeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            // The two best German models in the catalog, one per backend, so
            // they genuinely run side by side rather than queueing behind each
            // other on the same accelerator.
            models: vec!["whisper-large-v3".into(), "parakeet-v3".into()],
            reconcile: true,
            keep_audio: true,
        }
    }
}

/// Deprecated. Retained only so the cleanup pass can find and undo what an
/// older version left behind on the user's machine.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct BlackHoleSetup {
    /// `true` while our created devices are alive in the HAL.
    #[serde(default)]
    pub configured: bool,
    /// CoreAudio UID of the mic we mixed in.
    #[serde(default)]
    pub mic_uid: String,
    /// CoreAudio UID of the speakers we routed system audio through.
    #[serde(default)]
    pub speaker_uid: String,
    /// UID of the Multi-Output Device we created (Speakers + BlackHole).
    #[serde(default)]
    pub output_uid: String,
    /// UID of the Aggregate Device we created (BlackHole + Mic).
    #[serde(default)]
    pub input_uid: String,
    /// UID of whatever the default output was before we switched it — the
    /// one piece of this struct that still matters, because it is how the
    /// user gets their speakers back.
    #[serde(default)]
    pub previous_default_output_uid: String,
    /// cpal device name of the aggregate input.
    #[serde(default)]
    pub input_device_name: String,
}

impl BlackHoleSetup {
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiPostprocessConfig {
    pub enabled: bool,
    pub provider: String,
    pub api_key: String,
    pub model: String,
    pub base_url: String,
    pub default_prompt: String,
    pub custom_prompt: String,
    pub dictation_autorun: bool,
    pub dictation_preset: String,
    #[serde(default)]
    pub prompts: BTreeMap<String, String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            data_dir: DEFAULT_DATA_DIR.into(),
            dictation: DictationConfig {
                hotkey: "right_cmd".into(),
                mode: "hold_to_talk".into(),
                model: "whisper-turbo".into(),
                language: "auto".into(),
                insert_method: "smart_paste".into(),
            },
            meeting: MeetingConfig {
                audio_source: "both".into(),
                input_device: None,
                model: "whisper-turbo".into(),
                language: String::new(),
                save_audio: false,
                chunk_duration_seconds: 10,
                label_speakers: true,
                post_transcribe: PostTranscribeConfig::default(),
                blackhole: BlackHoleSetup::default(),
            },
            ai_postprocess: AiPostprocessConfig {
                enabled: false,
                provider: "none".into(),
                api_key: String::new(),
                model: "anthropic/claude-sonnet-4.5".into(),
                base_url: String::new(),
                default_prompt: "cleanup".into(),
                custom_prompt: String::new(),
                dictation_autorun: false,
                dictation_preset: "cleanup".into(),
                prompts: BTreeMap::new(),
            },
        }
    }
}

impl Config {
    /// Resolve `~` and return an absolute path.
    pub fn resolve_data_dir(data_dir: &str) -> PathBuf {
        let expanded = if let Some(stripped) = data_dir.strip_prefix("~/") {
            dirs::home_dir()
                .map(|h| h.join(stripped))
                .unwrap_or_else(|| PathBuf::from(data_dir))
        } else if data_dir == "~" {
            dirs::home_dir().unwrap_or_else(|| PathBuf::from(data_dir))
        } else {
            PathBuf::from(data_dir)
        };
        // canonicalize() requires existence; fall back to expanded form.
        std::fs::canonicalize(&expanded).unwrap_or(expanded)
    }

    pub fn config_path(data_dir: &str) -> PathBuf {
        Self::resolve_data_dir(data_dir).join(CONFIG_FILENAME)
    }

    pub fn models_dir(cfg: &Config) -> PathBuf {
        Self::resolve_data_dir(&cfg.data_dir).join("models")
    }

    pub fn transcripts_dir(cfg: &Config) -> PathBuf {
        Self::resolve_data_dir(&cfg.data_dir).join("transcripts")
    }

    pub fn audio_dir(cfg: &Config) -> PathBuf {
        Self::resolve_data_dir(&cfg.data_dir).join("audio")
    }

    /// Ensure `<data_dir>/{models,transcripts,audio}` exist.
    pub fn ensure_dirs(data_dir: &str) -> Result<PathBuf, ConfigError> {
        let root = Self::resolve_data_dir(data_dir);
        mkdir_all(&root)?;
        for sub in SUBDIRS {
            mkdir_all(&root.join(sub))?;
        }
        Ok(root)
    }

    /// Load config from the default data dir (`~/WhisprKing`).
    pub fn load_default() -> Result<Config, ConfigError> {
        Self::load(DEFAULT_DATA_DIR)
    }

    /// Load config from `<data_dir>/config.json`. Creates a default file on
    /// first run. Missing keys fall through to defaults via JSON deep merge.
    pub fn load(data_dir: &str) -> Result<Config, ConfigError> {
        Self::ensure_dirs(data_dir)?;
        let path = Self::config_path(data_dir);
        if !path.exists() {
            let cfg = Config {
                data_dir: data_dir.to_string(),
                ..Config::default()
            };
            cfg.save_to(data_dir)?;
            return Ok(cfg);
        }
        let raw = std::fs::read_to_string(&path).map_err(|e| ConfigError::Io {
            path: path.clone(),
            source: e,
        })?;
        let stored: serde_json::Value =
            serde_json::from_str(&raw).map_err(|e| ConfigError::Json {
                path: path.clone(),
                source: e,
            })?;

        let defaults = serde_json::to_value(Config::default()).expect("Config::default serializes");
        let merged = deep_merge(defaults, stored);
        serde_json::from_value(merged).map_err(|e| ConfigError::Json { path, source: e })
    }

    /// Save back into `<data_dir>/config.json`.
    pub fn save(&self) -> Result<(), ConfigError> {
        self.save_to(&self.data_dir.clone())
    }

    pub fn save_to(&self, data_dir: &str) -> Result<(), ConfigError> {
        Self::ensure_dirs(data_dir)?;
        let path = Self::config_path(data_dir);
        let body = serde_json::to_string_pretty(self).map_err(|e| ConfigError::Json {
            path: path.clone(),
            source: e,
        })?;
        std::fs::write(&path, body).map_err(|e| ConfigError::Io { path, source: e })
    }
}

fn mkdir_all(p: &Path) -> Result<(), ConfigError> {
    std::fs::create_dir_all(p).map_err(|e| ConfigError::Io {
        path: p.to_path_buf(),
        source: e,
    })
}

fn deep_merge(base: serde_json::Value, override_: serde_json::Value) -> serde_json::Value {
    match (base, override_) {
        (serde_json::Value::Object(mut b), serde_json::Value::Object(o)) => {
            for (k, v) in o {
                let merged = match b.remove(&k) {
                    Some(existing) => deep_merge(existing, v),
                    None => v,
                };
                b.insert(k, merged);
            }
            serde_json::Value::Object(b)
        }
        (_, override_) => override_,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn round_trip_creates_default() {
        let dir = tempdir().unwrap();
        let path = dir.path().to_str().unwrap().to_string();
        let cfg = Config::load(&path).unwrap();
        assert_eq!(cfg.dictation.hotkey, "right_cmd");
        assert!(dir.path().join("models").is_dir());
        assert!(dir.path().join("config.json").is_file());
    }

    #[test]
    fn missing_keys_fill_from_defaults() {
        let dir = tempdir().unwrap();
        let path = dir.path().to_str().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"dictation": {"hotkey": "f19"}}"#,
        )
        .unwrap();
        let cfg = Config::load(path).unwrap();
        assert_eq!(cfg.dictation.hotkey, "f19");
        // defaults filled in
        assert_eq!(cfg.dictation.model, "whisper-turbo");
        assert_eq!(cfg.meeting.chunk_duration_seconds, 10);
    }

    /// A config written before meetings had their own language must keep
    /// behaving exactly as it did: empty means "follow dictation".
    #[test]
    fn meeting_language_defaults_to_following_dictation() {
        let dir = tempdir().unwrap();
        let path = dir.path().to_str().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"dictation": {"language": "de"}, "meeting": {"model": "whisper-turbo"}}"#,
        )
        .unwrap();
        let cfg = Config::load(path).unwrap();
        assert_eq!(cfg.dictation.language, "de");
        assert_eq!(cfg.meeting.language, "");
    }
}
