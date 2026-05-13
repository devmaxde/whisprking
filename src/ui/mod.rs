//! GUI layer: egui-based main window + dictation overlay viewport + a
//! menu-bar tray icon.

pub mod app;
pub mod overlay;
pub mod pages;
pub mod styles;
pub mod tray;

pub use app::{run as run_app, AppEvent};
pub use tray::{TrayEvent, TrayHandle, TrayState};
