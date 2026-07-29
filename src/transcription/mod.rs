//! Speech-to-text engine and model catalog.

pub mod context;
pub mod engine;
pub mod model_manager;
pub mod pack;
pub mod post;

pub use context::{carry_context, language_override, looks_degenerate};
pub use engine::{DecodeContext, EngineError, TimedPiece, Transcriber, TranscriptionEngine};
pub use pack::{pack, Window};
pub use post::{run_post, PostPhase, PostSpec, ReconcileSpec};
pub use model_manager::{ModelManager, ModelSpec, AVAILABLE_MODELS};
