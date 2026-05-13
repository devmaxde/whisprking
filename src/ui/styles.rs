//! Colors, sizes, fonts. Mirrors `ui/styles.py`.

use egui::Color32;

pub const BG_WINDOW: Color32 = Color32::from_rgb(0x1e, 0x1e, 0x2e);
pub const BG_SURFACE: Color32 = Color32::from_rgb(0x2a, 0x2a, 0x3e);
pub const BG_INPUT: Color32 = Color32::from_rgb(0x25, 0x25, 0x38);
pub const BG_HOVER: Color32 = Color32::from_rgb(0x35, 0x35, 0x4d);
pub const BORDER: Color32 = Color32::from_rgb(0x3a, 0x3a, 0x50);

pub const TEXT_PRIMARY: Color32 = Color32::from_rgb(0xe0, 0xe0, 0xe0);
pub const TEXT_SECONDARY: Color32 = Color32::from_rgb(0x88, 0x88, 0xa0);
pub const TEXT_TIMESTAMP: Color32 = Color32::from_rgb(0x6a, 0x6a, 0x80);

pub const ACCENT_RED: Color32 = Color32::from_rgb(0xe7, 0x4c, 0x3c);
pub const ACCENT_ORANGE: Color32 = Color32::from_rgb(0xe6, 0x7e, 0x22);
pub const ACCENT_BLUE: Color32 = Color32::from_rgb(0x34, 0x98, 0xdb);
pub const ACCENT_GREEN: Color32 = Color32::from_rgb(0x2e, 0xcc, 0x71);

pub const FONT_SIZE_H1: f32 = 18.0;
pub const FONT_SIZE_BODY: f32 = 14.0;
pub const FONT_SIZE_SMALL: f32 = 12.0;

pub const OVERLAY_WIDTH: f32 = 260.0;
pub const OVERLAY_HEIGHT: f32 = 60.0;
pub const OVERLAY_BOTTOM_OFFSET: f32 = 60.0;

pub const WINDOW_DEFAULT_SIZE: [f32; 2] = [700.0, 500.0];
pub const WINDOW_MIN_SIZE: [f32; 2] = [500.0, 400.0];

/// Apply the dark theme defaults to an egui context. Call once at startup.
pub fn apply_theme(ctx: &egui::Context) {
    let mut visuals = egui::Visuals::dark();
    visuals.window_fill = BG_WINDOW;
    visuals.panel_fill = BG_WINDOW;
    visuals.extreme_bg_color = BG_INPUT;
    visuals.widgets.noninteractive.bg_fill = BG_SURFACE;
    visuals.widgets.inactive.bg_fill = BG_SURFACE;
    visuals.widgets.inactive.weak_bg_fill = BG_SURFACE;
    visuals.widgets.hovered.bg_fill = BG_HOVER;
    visuals.widgets.active.bg_fill = BORDER;
    visuals.selection.bg_fill = ACCENT_BLUE.linear_multiply(0.4);
    visuals.hyperlink_color = ACCENT_BLUE;
    ctx.set_visuals(visuals);
}
