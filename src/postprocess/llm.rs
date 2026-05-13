//! LLM post-processing with pluggable providers. Only OpenRouter for now;
//! adding e.g. Ollama means another `impl LlmProvider`.

use std::collections::HashMap;
use std::time::Duration;

use serde::Deserialize;
use serde_json::{json, Value};
use thiserror::Error;

const OPENROUTER_MODELS_URL: &str = "https://openrouter.ai/api/v1/models";
const OPENROUTER_CHAT_URL: &str = "https://openrouter.ai/api/v1/chat/completions";

#[derive(Debug, Error)]
pub enum LlmError {
    #[error("openrouter api key missing")]
    MissingApiKey,
    #[error("openrouter model id missing")]
    MissingModel,
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("openrouter responded with HTTP {status}: {body}")]
    BadStatus { status: u16, body: String },
    #[error("unexpected response shape: {0}")]
    Shape(String),
}

#[derive(Debug, Clone)]
pub struct Preset {
    pub key: &'static str,
    pub label: &'static str,
    pub system: &'static str,
}

pub const PRESETS: &[Preset] = &[
    Preset {
        key: "cleanup",
        label: "Cleanup (Filler, Punctuation)",
        system: include_str!("./prompts/cleanup.txt"),
    },
    Preset {
        key: "summary",
        label: "Zusammenfassung",
        system: include_str!("./prompts/summary.txt"),
    },
    Preset {
        key: "action_items",
        label: "Action Items",
        system: include_str!("./prompts/action_items.txt"),
    },
    Preset {
        key: "custom",
        label: "Custom prompt",
        system: "",
    },
];

fn preset(key: &str) -> Option<&'static Preset> {
    PRESETS.iter().find(|p| p.key == key)
}

#[derive(Debug, Clone)]
pub struct LlmConfig {
    pub provider: String,
    pub api_key: String,
    pub model: String,
    pub base_url: String,
    pub default_prompt: String,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            provider: "none".into(),
            api_key: String::new(),
            model: String::new(),
            base_url: String::new(),
            default_prompt: "cleanup".into(),
        }
    }
}

impl LlmConfig {
    pub fn from_ai_section(ai: &crate::config::AiPostprocessConfig) -> Self {
        Self {
            provider: ai.provider.clone(),
            api_key: ai.api_key.clone(),
            model: ai.model.clone(),
            base_url: ai.base_url.clone(),
            default_prompt: ai.default_prompt.clone(),
        }
    }
}

pub trait LlmProvider: Send + Sync {
    fn run(&self, system_prompt: &str, user_text: &str) -> Result<String, LlmError>;
}

pub fn make_provider(cfg: &LlmConfig) -> Option<Box<dyn LlmProvider>> {
    match cfg.provider.as_str() {
        "openrouter" => match OpenRouterProvider::new(&cfg.api_key, &cfg.model, &cfg.base_url) {
            Ok(p) => Some(Box::new(p)),
            Err(e) => {
                log::warn!("openrouter provider unavailable: {e}");
                None
            }
        },
        "lm_studio" => local_provider(cfg, "http://localhost:1234/v1/chat/completions"),
        "ollama" => local_provider(cfg, "http://localhost:11434/v1/chat/completions"),
        _ => None,
    }
}

fn local_provider(cfg: &LlmConfig, default_url: &str) -> Option<Box<dyn LlmProvider>> {
    let url = if cfg.base_url.is_empty() {
        default_url.to_string()
    } else {
        cfg.base_url.clone()
    };
    match LocalChatProvider::new(&cfg.model, url) {
        Ok(p) => Some(Box::new(p)),
        Err(e) => {
            log::warn!("local provider unavailable: {e}");
            None
        }
    }
}

/// Provider for OpenAI-compatible chat endpoints exposed by local runners
/// like LM Studio (`/v1/chat/completions` on :1234) or Ollama
/// (`/v1/chat/completions` on :11434). No auth, model id is passed through
/// verbatim.
pub struct LocalChatProvider {
    client: reqwest::blocking::Client,
    model: String,
    url: String,
}

impl LocalChatProvider {
    pub fn new(model: &str, url: String) -> Result<Self, LlmError> {
        if model.is_empty() {
            return Err(LlmError::MissingModel);
        }
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()?;
        Ok(Self {
            client,
            model: model.to_string(),
            url,
        })
    }
}

impl LlmProvider for LocalChatProvider {
    fn run(&self, system_prompt: &str, user_text: &str) -> Result<String, LlmError> {
        let payload = json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            "stream": false,
        });
        let resp = self.client.post(&self.url).json(&payload).send()?;
        if !resp.status().is_success() {
            return Err(LlmError::BadStatus {
                status: resp.status().as_u16(),
                body: resp.text().unwrap_or_default().chars().take(400).collect(),
            });
        }
        let body: Value = resp.json()?;
        body.get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|t| t.as_str())
            .map(|s| s.trim().to_string())
            .ok_or_else(|| LlmError::Shape(body.to_string()))
    }
}

/// Resolve the system prompt for a preset, honouring per-preset overrides
/// from the config (an empty override means "use built-in").
pub fn resolve_system_prompt(
    preset_key: &str,
    custom_prompt: &str,
    overrides: &HashMap<String, String>,
) -> String {
    if preset_key == "custom" {
        let trimmed = custom_prompt.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
        return preset("cleanup")
            .map(|p| p.system.to_string())
            .unwrap_or_default();
    }

    if let Some(over) = overrides.get(preset_key) {
        let trimmed = over.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }

    preset(preset_key)
        .or_else(|| preset("cleanup"))
        .map(|p| p.system.to_string())
        .unwrap_or_default()
}

#[derive(Debug, Clone, Deserialize)]
pub struct OpenRouterModel {
    pub id: String,
    pub name: String,
    pub context_length: Option<u32>,
    #[serde(default)]
    pub pricing: HashMap<String, Value>,
}

#[derive(Deserialize)]
struct ModelsResponse {
    data: Vec<RawModel>,
}

#[derive(Deserialize)]
struct RawModel {
    #[serde(default)]
    id: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    context_length: Option<u32>,
    #[serde(default)]
    pricing: HashMap<String, Value>,
}

pub fn fetch_openrouter_models() -> Result<Vec<OpenRouterModel>, LlmError> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()?;
    let resp = client.get(OPENROUTER_MODELS_URL).send()?;
    if !resp.status().is_success() {
        return Err(LlmError::BadStatus {
            status: resp.status().as_u16(),
            body: resp.text().unwrap_or_default(),
        });
    }
    let parsed: ModelsResponse = resp.json()?;
    let mut models: Vec<OpenRouterModel> = parsed
        .data
        .into_iter()
        .map(|m| OpenRouterModel {
            name: m.name.clone().unwrap_or_else(|| m.id.clone()),
            id: m.id,
            context_length: m.context_length,
            pricing: m.pricing,
        })
        .collect();
    models.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(models)
}

pub struct OpenRouterProvider {
    client: reqwest::blocking::Client,
    api_key: String,
    model: String,
    url: String,
}

impl OpenRouterProvider {
    pub fn new(api_key: &str, model: &str, base_url: &str) -> Result<Self, LlmError> {
        if api_key.is_empty() {
            return Err(LlmError::MissingApiKey);
        }
        if model.is_empty() {
            return Err(LlmError::MissingModel);
        }
        let url = if base_url.is_empty() {
            OPENROUTER_CHAT_URL.to_string()
        } else {
            base_url.to_string()
        };
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()?;
        Ok(Self {
            client,
            api_key: api_key.to_string(),
            model: model.to_string(),
            url,
        })
    }
}

impl LlmProvider for OpenRouterProvider {
    fn run(&self, system_prompt: &str, user_text: &str) -> Result<String, LlmError> {
        let payload = json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
        });
        let resp = self
            .client
            .post(&self.url)
            .bearer_auth(&self.api_key)
            .header("HTTP-Referer", "https://github.com/moritz/whisprking")
            .header("X-Title", "WhisprKing")
            .json(&payload)
            .send()?;

        if !resp.status().is_success() {
            return Err(LlmError::BadStatus {
                status: resp.status().as_u16(),
                body: resp.text().unwrap_or_default().chars().take(400).collect(),
            });
        }
        let body: Value = resp.json()?;
        body.get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|t| t.as_str())
            .map(|s| s.trim().to_string())
            .ok_or_else(|| LlmError::Shape(body.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn custom_falls_back_to_cleanup() {
        let p = resolve_system_prompt("custom", "  ", &HashMap::new());
        assert!(p.starts_with("You clean up speech-to-text"));
    }

    #[test]
    fn override_replaces_builtin() {
        let mut o = HashMap::new();
        o.insert("summary".into(), "be brief".into());
        let p = resolve_system_prompt("summary", "", &o);
        assert_eq!(p, "be brief");
    }

    #[test]
    fn unknown_preset_falls_back() {
        let p = resolve_system_prompt("nope", "", &HashMap::new());
        assert!(p.starts_with("You clean up"));
    }
}
