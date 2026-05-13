//! WhisprKing — local speech-to-text for macOS.
//!
//! Each submodule is independently testable
//! the binary entry point in `main.rs` is the only place that
//! wires them together.

pub mod audio;
pub mod config;
pub mod dictation;
pub mod hotkey;
pub mod output;
pub mod postprocess;
pub mod transcription;
pub mod ui;
pub mod utils;
