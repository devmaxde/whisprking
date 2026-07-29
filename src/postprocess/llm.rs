//! LLM post-processing with pluggable providers. Only OpenRouter for now;
//! adding e.g. Ollama means another `impl LlmProvider`.

use std::collections::HashMap;
use std::time::Duration;

use serde::Deserialize;
use serde_json::{json, Value};
use thiserror::Error;

const OPENROUTER_MODELS_URL: &str = "https://openrouter.ai/api/v1/models";
const OPENROUTER_CHAT_URL: &str = "https://openrouter.ai/api/v1/chat/completions";
const LM_STUDIO_URL: &str = "http://localhost:1234/v1/chat/completions";
const OLLAMA_URL: &str = "http://localhost:11434/v1/chat/completions";

/// How long one chat request may take, scaled to the amount of text it
/// carries: a base allowance plus a minute per 10k characters.
///
/// A flat two-minute limit is what made long meeting transcripts fail. The
/// model was still writing when the client hung up, and because a client-side
/// timeout arrives as a generic transport error, the banner said "der Provider
/// hat einen Fehler gemeldet" for something the provider never did.
const BASE_TIMEOUT_SECS: u64 = 90;
const TIMEOUT_SECS_PER_10K_CHARS: u64 = 60;
const MAX_TIMEOUT_SECS: u64 = 20 * 60;

/// Longest excerpt of a provider payload kept for the error report. Enough to
/// hold a full error object, short enough to stay readable in a panel.
const RAW_EXCERPT_CHARS: usize = 1200;

pub fn chat_timeout(chars: usize) -> Duration {
    let extra = (chars / 10_000) as u64 * TIMEOUT_SECS_PER_10K_CHARS;
    Duration::from_secs((BASE_TIMEOUT_SECS + extra).min(MAX_TIMEOUT_SECS))
}

/// Very rough token count — four characters per token. Only ever used to
/// explain a failure ("your transcript is about this big"), never to decide
/// whether a request is made.
pub fn estimate_tokens(text: &str) -> usize {
    text.chars().count().div_ceil(4)
}

#[derive(Debug, Error)]
pub enum LlmError {
    #[error("Es ist kein KI-Provider ausgewählt.")]
    NoProvider,
    #[error("Unbekannter Provider „{0}“.")]
    UnknownProvider(String),
    #[error("Es ist kein API-Key hinterlegt.")]
    MissingApiKey,
    #[error("Es ist kein Modell ausgewählt.")]
    MissingModel,
    /// The client stopped waiting. Prime suspect for long transcripts.
    #[error("Zeitüberschreitung: nach {seconds} s kam keine vollständige Antwort.")]
    Timeout { seconds: u64 },
    /// Nothing listening at the other end — local runner not started, or a
    /// wrong base URL.
    #[error("{url} war nicht erreichbar.")]
    Unreachable { url: String, detail: String },
    #[error("Netzwerkfehler: {0}")]
    Http(#[from] reqwest::Error),
    /// The provider answered, and the answer was a refusal.
    #[error("Der Provider hat abgelehnt (HTTP {status}): {message}")]
    Api {
        status: u16,
        code: Option<String>,
        message: String,
        raw: String,
    },
    /// Text arrived, but the model stopped because it ran out of room.
    #[error("Die Antwort wurde nach {chars} Zeichen abgeschnitten (finish_reason = {reason}).")]
    Truncated { reason: String, chars: usize },
    #[error("Die Antwort hatte ein unerwartetes Format.")]
    Shape { raw: String },
}

impl LlmError {
    /// Extra facts for the error report, on top of what the caller knows.
    pub fn facts(&self) -> Vec<(String, String)> {
        match self {
            LlmError::Timeout { seconds } => {
                vec![("Zeitlimit".into(), format!("{seconds} s"))]
            },
            LlmError::Unreachable { url, detail } => vec![
                ("Adresse".into(), url.clone()),
                ("Grund".into(), detail.clone()),
            ],
            LlmError::Api { status, code, .. } => {
                let mut facts = vec![("HTTP-Status".into(), status.to_string())];
                if let Some(code) = code {
                    facts.push(("Fehlercode".into(), code.clone()));
                }
                facts
            },
            LlmError::Truncated { reason, chars } => vec![
                ("Abbruchgrund".into(), reason.clone()),
                ("Erhaltene Länge".into(), format!("{chars} Zeichen")),
            ],
            _ => Vec::new(),
        }
    }

    /// The provider's own words, verbatim, when there are any — so a 400 can
    /// be read without opening the log.
    pub fn raw(&self) -> Option<&str> {
        match self {
            LlmError::Api { raw, .. } | LlmError::Shape { raw } => Some(raw.as_str()),
            _ => None,
        }
    }

    /// What to try next. Keyed on what actually happened, never a guess.
    pub fn hint(&self) -> Option<String> {
        let hint = match self {
            LlmError::NoProvider | LlmError::UnknownProvider(_) => {
                "Einstellungen → KI: einen Provider auswählen."
            },
            LlmError::MissingApiKey => "Einstellungen → KI: API-Key eintragen.",
            LlmError::MissingModel => "Einstellungen → KI: ein Modell eintragen oder wählen.",
            LlmError::Timeout { .. } => {
                "Ein schnelleres Modell wählen, oder das Transkript in kürzere Aufnahmen \
                 aufteilen — die Wartezeit wächst mit der Transkriptlänge."
            },
            LlmError::Unreachable { .. } => {
                "Läuft der lokale Server (LM Studio bzw. Ollama), und stimmt die Adresse?"
            },
            LlmError::Http(_) => "Internetverbindung prüfen und erneut versuchen.",
            LlmError::Api {
                status, message, ..
            } => {
                if looks_like_context_overflow(message) {
                    "Das Transkript ist länger als das Kontextfenster des Modells. \
                     Ein Modell mit größerem Kontext wählen (bei OpenRouter steht die \
                     Kontextlänge in der Modelliste)."
                } else {
                    match status {
                        401 | 403 => "API-Key prüfen — er wurde abgelehnt.",
                        402 => "Beim Provider ist kein Guthaben mehr vorhanden.",
                        404 => "Die Modell-ID gibt es bei diesem Provider nicht.",
                        413 => {
                            "Die Anfrage war zu groß für den Provider — kürzeres Transkript \
                             oder Modell mit größerem Kontext."
                        },
                        429 => "Rate Limit erreicht — später erneut oder anderes Modell.",
                        500..=599 => "Störung beim Provider — später erneut versuchen.",
                        _ => return None,
                    }
                }
            },
            LlmError::Truncated { .. } => {
                "Das Modell hat sein Ausgabelimit erreicht, bevor es fertig war. Für lange \
                 Transkripte eignet sich „Zusammenfassung“ besser als „Bereinigung“, oder \
                 ein Modell mit größerem Ausgabelimit."
            },
            LlmError::Shape { .. } => {
                "Antwortet unter dieser Adresse wirklich eine OpenAI-kompatible \
                 /chat/completions-Schnittstelle?"
            },
        };
        Some(hint.to_string())
    }
}

/// Providers phrase it differently ("maximum context length", "too many
/// tokens", "context_length_exceeded"), so match on the vocabulary rather
/// than on one provider's exact string.
fn looks_like_context_overflow(message: &str) -> bool {
    let m = message.to_lowercase();
    let context_word = m.contains("context") || m.contains("kontext");
    let size_word = m.contains("length") || m.contains("window") || m.contains("exceed");
    (context_word && size_word)
        || m.contains("too many tokens")
        || (m.contains("maximum") && m.contains("token"))
}

/// Human name of a provider key, for error reports.
pub fn provider_label(key: &str) -> &'static str {
    match key {
        "openrouter" => "OpenRouter",
        "lm_studio" => "LM Studio (lokal)",
        "ollama" => "Ollama (lokal)",
        _ => "—",
    }
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
        label: "Bereinigung",
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
        label: "Eigener Prompt",
        system: "",
    },
];

/// Prompt for merging several acoustic models' transcripts of one recording.
///
/// Deliberately not in [`PRESETS`]: that list is the set of actions a user
/// picks to run against a finished transcript, and "merge the variants" is not
/// one — it is a step inside the post-transcription pass, which supplies its
/// own input. It still resolves through [`resolve_system_prompt`], so the
/// per-preset override in the config works for it like any other.
pub const RECONCILE_PRESET: Preset = Preset {
    key: "reconcile",
    label: "Modelle zusammenführen",
    system: include_str!("./prompts/reconcile.txt"),
};

/// Every prompt the user may edit: the pickable actions plus the internal
/// ones that are still worth tuning.
pub fn editable_presets() -> Vec<&'static Preset> {
    PRESETS.iter().chain(std::iter::once(&RECONCILE_PRESET)).collect()
}

fn preset(key: &str) -> Option<&'static Preset> {
    PRESETS
        .iter()
        .chain(std::iter::once(&RECONCILE_PRESET))
        .find(|p| p.key == key)
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

/// Build the configured provider, or say precisely what is missing — the
/// caller turns that into the banner, so "nothing happens and the log knows
/// why" is not an option here.
pub fn make_provider(cfg: &LlmConfig) -> Result<Box<dyn LlmProvider>, LlmError> {
    match cfg.provider.as_str() {
        "openrouter" => Ok(Box::new(OpenRouterProvider::new(
            &cfg.api_key,
            &cfg.model,
            &cfg.base_url,
        )?)),
        "lm_studio" => local_provider(cfg, LM_STUDIO_URL),
        "ollama" => local_provider(cfg, OLLAMA_URL),
        "none" | "" => Err(LlmError::NoProvider),
        other => Err(LlmError::UnknownProvider(other.to_string())),
    }
}

fn local_provider(cfg: &LlmConfig, default_url: &str) -> Result<Box<dyn LlmProvider>, LlmError> {
    let url = if cfg.base_url.is_empty() {
        default_url.to_string()
    } else {
        cfg.base_url.clone()
    };
    Ok(Box::new(LocalChatProvider::new(&cfg.model, url)?))
}

// --- shared request/response handling --------------------------------------

/// Client used for chat requests. The timeout is set per request because it
/// depends on how much text is being sent; see [`chat_timeout`]. The one here
/// is only a backstop.
fn chat_client(headers: reqwest::header::HeaderMap) -> Result<reqwest::blocking::Client, LlmError> {
    Ok(reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(MAX_TIMEOUT_SECS))
        .default_headers(headers)
        .build()?)
}

/// Classify a transport failure. `reqwest`'s own message for a client-side
/// timeout is indistinguishable from any other I/O problem in a banner, and
/// that is exactly the failure long transcripts hit.
fn transport_error(err: reqwest::Error, url: &str, timeout: Duration) -> LlmError {
    if err.is_timeout() {
        LlmError::Timeout {
            seconds: timeout.as_secs(),
        }
    } else if err.is_connect() {
        LlmError::Unreachable {
            url: url.to_string(),
            detail: err.to_string(),
        }
    } else {
        LlmError::Http(err)
    }
}

fn clip(s: &str) -> String {
    let mut out: String = s.trim().chars().take(RAW_EXCERPT_CHARS).collect();
    if s.trim().chars().count() > RAW_EXCERPT_CHARS {
        out.push('…');
    }
    out
}

fn first_line(s: &str) -> String {
    let line = s.trim().lines().next().unwrap_or_default().trim();
    if line.is_empty() {
        "kein Fehlertext".to_string()
    } else {
        line.chars().take(300).collect()
    }
}

/// Build an [`LlmError::Api`] from an OpenAI-style `{"error": …}` object.
fn api_error(status: u16, err: &Value, body: &str) -> LlmError {
    let message = err
        .get("message")
        .and_then(|m| m.as_str())
        .or_else(|| err.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| first_line(&err.to_string()));
    let code = err.get("code").map(|c| match c.as_str() {
        Some(s) => s.to_string(),
        None => c.to_string(),
    });
    LlmError::Api {
        status,
        code,
        message,
        raw: clip(body),
    }
}

/// Non-2xx response → error. The body is usually JSON, but a proxy in front
/// of a local runner will happily return HTML, so both are handled.
fn status_error(status: u16, body: String) -> LlmError {
    if let Ok(value) = serde_json::from_str::<Value>(&body) {
        if let Some(err) = value.get("error") {
            return api_error(status, err, &body);
        }
    }
    LlmError::Api {
        status,
        code: None,
        message: first_line(&body),
        raw: clip(&body),
    }
}

/// Pull the assistant message out of a chat completion.
///
/// Every OpenAI-compatible endpoint has the same three ways of failing while
/// still returning HTTP 200: an `error` object instead of choices, a decode
/// that stopped at the output limit, and an empty message. Handled once here
/// so no provider can forget one.
fn content_of(body: &Value, raw: &str) -> Result<String, LlmError> {
    if let Some(err) = body.get("error") {
        return Err(api_error(200, err, raw));
    }
    let choice = body
        .get("choices")
        .and_then(|c| c.get(0))
        .ok_or_else(|| LlmError::Shape { raw: clip(raw) })?;
    let text = choice
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(|t| t.as_str())
        .unwrap_or_default()
        .trim()
        .to_string();
    let finish = choice
        .get("finish_reason")
        .and_then(|f| f.as_str())
        .unwrap_or_default();
    if finish == "length" {
        return Err(LlmError::Truncated {
            reason: finish.to_string(),
            chars: text.chars().count(),
        });
    }
    if text.is_empty() {
        return Err(LlmError::Shape { raw: clip(raw) });
    }
    Ok(text)
}

/// One chat round trip against an OpenAI-compatible endpoint.
fn post_chat(
    client: &reqwest::blocking::Client,
    url: &str,
    api_key: Option<&str>,
    payload: Value,
    timeout: Duration,
) -> Result<String, LlmError> {
    let mut req = client.post(url).timeout(timeout).json(&payload);
    if let Some(key) = api_key {
        req = req.bearer_auth(key);
    }
    let resp = req
        .send()
        .map_err(|e| transport_error(e, url, timeout))?;

    let status = resp.status().as_u16();
    let raw = resp
        .text()
        .map_err(|e| transport_error(e, url, timeout))?;
    if !(200..300).contains(&status) {
        return Err(status_error(status, raw));
    }
    let body: Value =
        serde_json::from_str(&raw).map_err(|_| LlmError::Shape { raw: clip(&raw) })?;
    content_of(&body, &raw)
}

fn chat_payload(model: &str, system_prompt: &str, user_text: &str) -> Value {
    json!({
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
    })
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
        Ok(Self {
            client: chat_client(reqwest::header::HeaderMap::new())?,
            model: model.to_string(),
            url,
        })
    }
}

impl LlmProvider for LocalChatProvider {
    fn run(&self, system_prompt: &str, user_text: &str) -> Result<String, LlmError> {
        let mut payload = chat_payload(&self.model, system_prompt, user_text);
        payload["stream"] = Value::Bool(false);
        let timeout = chat_timeout(system_prompt.len() + user_text.len());
        log::info!(
            "llm: {} chars → {} (timeout {} s)",
            user_text.chars().count(),
            self.url,
            timeout.as_secs()
        );
        post_chat(&self.client, &self.url, None, payload, timeout)
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
    let status = resp.status().as_u16();
    if !resp.status().is_success() {
        return Err(status_error(status, resp.text().unwrap_or_default()));
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
        // App attribution for the OpenRouter dashboard; sent on every call.
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            "HTTP-Referer",
            reqwest::header::HeaderValue::from_static("https://github.com/moritz/whisprking"),
        );
        headers.insert(
            "X-Title",
            reqwest::header::HeaderValue::from_static("WhisprKing"),
        );
        Ok(Self {
            client: chat_client(headers)?,
            api_key: api_key.to_string(),
            model: model.to_string(),
            url,
        })
    }
}

impl LlmProvider for OpenRouterProvider {
    fn run(&self, system_prompt: &str, user_text: &str) -> Result<String, LlmError> {
        let payload = chat_payload(&self.model, system_prompt, user_text);
        let timeout = chat_timeout(system_prompt.len() + user_text.len());
        log::info!(
            "llm: {} chars → openrouter {} (timeout {} s)",
            user_text.chars().count(),
            self.model,
            timeout.as_secs()
        );
        post_chat(
            &self.client,
            &self.url,
            Some(&self.api_key),
            payload,
            timeout,
        )
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

    /// A long transcript must be given proportionally more time — the flat
    /// two-minute limit is what made them fail.
    #[test]
    fn timeout_grows_with_input_and_is_capped() {
        let short = chat_timeout(1_000);
        let long = chat_timeout(120_000);
        assert!(short.as_secs() >= BASE_TIMEOUT_SECS);
        assert!(long > short);
        assert_eq!(chat_timeout(usize::MAX).as_secs(), MAX_TIMEOUT_SECS);
    }

    /// OpenRouter answers HTTP 200 with an error object often enough that
    /// treating it as "unexpected shape" hides the actual reason.
    #[test]
    fn error_object_in_a_200_is_an_api_error() {
        let raw = r#"{"error":{"message":"maximum context length is 8192 tokens","code":400}}"#;
        let body: Value = serde_json::from_str(raw).unwrap();
        match content_of(&body, raw) {
            Err(LlmError::Api { message, code, .. }) => {
                assert!(message.contains("maximum context length"));
                assert_eq!(code.as_deref(), Some("400"));
            },
            other => panic!("expected an API error, got {other:?}"),
        }
    }

    #[test]
    fn a_cut_off_answer_is_not_silently_kept() {
        let raw = r#"{"choices":[{"finish_reason":"length","message":{"content":"halb"}}]}"#;
        let body: Value = serde_json::from_str(raw).unwrap();
        assert!(matches!(
            content_of(&body, raw),
            Err(LlmError::Truncated { .. })
        ));
    }

    #[test]
    fn a_complete_answer_comes_through_trimmed() {
        let raw = r#"{"choices":[{"finish_reason":"stop","message":{"content":"  hallo  "}}]}"#;
        let body: Value = serde_json::from_str(raw).unwrap();
        assert_eq!(content_of(&body, raw).unwrap(), "hallo");
    }

    /// Local runners behind a proxy answer with HTML, not JSON.
    #[test]
    fn non_json_error_body_still_yields_a_message() {
        let err = status_error(502, "<html>Bad Gateway</html>".into());
        match err {
            LlmError::Api {
                status, message, ..
            } => {
                assert_eq!(status, 502);
                assert!(message.contains("Bad Gateway"));
            },
            other => panic!("expected an API error, got {other:?}"),
        }
    }

    #[test]
    fn context_overflow_is_recognised_across_wordings() {
        assert!(looks_like_context_overflow(
            "This model's maximum context length is 8192 tokens"
        ));
        assert!(looks_like_context_overflow("context_length_exceeded"));
        assert!(!looks_like_context_overflow("invalid api key"));
    }

    /// Every failure has to offer a next step; a bare message in a banner is
    /// what this whole path exists to replace.
    #[test]
    fn every_error_has_a_hint() {
        let errors = [
            LlmError::NoProvider,
            LlmError::MissingApiKey,
            LlmError::MissingModel,
            LlmError::Timeout { seconds: 300 },
            LlmError::Truncated {
                reason: "length".into(),
                chars: 10,
            },
            LlmError::Shape { raw: "{}".into() },
        ];
        for err in errors {
            assert!(err.hint().is_some(), "no hint for {err:?}");
            assert!(!err.to_string().is_empty());
        }
    }
}
