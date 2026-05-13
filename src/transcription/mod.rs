//! Speech-to-text engine and model catalog.

pub mod engine;
pub mod model_manager;

pub use engine::{EngineError, Transcriber, TranscriptionEngine};
pub use model_manager::{ModelManager, ModelSpec, AVAILABLE_MODELS};
