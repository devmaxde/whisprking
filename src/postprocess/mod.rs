//! LLM-based post-processing of transcripts.

pub mod llm;

pub use llm::{
    fetch_openrouter_models, make_provider, resolve_system_prompt, LlmConfig, LlmError,
    LlmProvider, LocalChatProvider, OpenRouterModel, OpenRouterProvider, Preset, PRESETS,
};
