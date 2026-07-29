//! GUI layer: egui-based main window + dictation overlay viewport + a
//! menu-bar tray icon.
//!
//! The look lives in three files and nowhere else: [`theme`] (design tokens
//! and the egui style), [`icons`] (vector glyphs) and [`widgets`] (the
//! components every page is assembled from). Pages compose those; they do
//! not paint raw colors. [`brand`] is the app's own mark — the crowned
//! microphone that appears in the rail, the menu bar and the bundle icon.

pub mod activation;
pub mod app;
pub mod brand;
pub mod icons;
pub mod overlay;
pub mod pages;
pub mod theme;
pub mod tray;
pub mod widgets;

pub use app::{run as run_app, AppEvent};
pub use tray::{TrayEvent, TrayHandle, TrayState};
