//! Floating dictation overlay rendered as a borderless, always-on-top
//! egui viewport.
//!
//! The overlay shows a level meter while recording and a spinner while
//! transcribing. It is driven from the main app's `update()` loop — when
//! [`OverlayState::Hidden`], no viewport is requested and nothing renders.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use egui::{Color32, FontId, Rect, Sense, Stroke, Vec2, ViewportBuilder, ViewportId};

use super::styles;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum OverlayState {
    #[default]
    Hidden,
    Recording,
    Transcribing,
}

/// Lock-free shared `f32` for the audio level. The recorder writes from
/// its callback thread; the UI thread reads each frame.
#[derive(Clone, Default)]
pub struct LevelSource(Arc<AtomicU32>);

impl LevelSource {
    pub fn set(&self, value: f32) {
        self.0.store(value.to_bits(), Ordering::Relaxed);
    }
    pub fn get(&self) -> f32 {
        f32::from_bits(self.0.load(Ordering::Relaxed))
    }
}

/// Renders the overlay viewport. Call once per egui frame from the app's
/// `update`. State is owned by the caller so the overlay is just a view.
pub fn show_overlay(
    ctx: &egui::Context,
    state: OverlayState,
    level: &LevelSource,
    spinner_phase: &mut f32,
) {
    if state == OverlayState::Hidden {
        return;
    }

    let viewport = ViewportBuilder::default()
        .with_title("WhisprKing Overlay")
        .with_decorations(false)
        .with_always_on_top()
        .with_transparent(true)
        .with_resizable(false)
        .with_taskbar(false)
        .with_inner_size([styles::OVERLAY_WIDTH, styles::OVERLAY_HEIGHT])
        .with_position(rest_position(ctx));

    ctx.show_viewport_immediate(
        ViewportId::from_hash_of("whisprking-overlay"),
        viewport,
        |ui, _class| {
            // Repaint quickly so the level meter feels alive.
            ui.ctx()
                .request_repaint_after(std::time::Duration::from_millis(33));
            *spinner_phase += 0.12;

            let frame = egui::Frame::new()
                .fill(Color32::from_rgba_unmultiplied(20, 20, 28, 230))
                .inner_margin(egui::Margin::symmetric(14, 8))
                .corner_radius(egui::CornerRadius::same(14));

            egui::CentralPanel::default()
                .frame(frame)
                .show_inside(ui, |ui| {
                    ui.horizontal_centered(|ui| {
                        let (rect, _) = ui.allocate_exact_size(Vec2::splat(24.0), Sense::hover());
                        match state {
                            OverlayState::Recording => {
                                draw_level_bars(ui, rect, level.get(), *spinner_phase);
                            }
                            OverlayState::Transcribing => {
                                draw_spinner(ui, rect, *spinner_phase);
                            }
                            OverlayState::Hidden => {}
                        }
                        ui.add_space(10.0);
                        let (label, color) = match state {
                            OverlayState::Recording => ("Listening…", styles::TEXT_PRIMARY),
                            OverlayState::Transcribing => ("Transcribing…", styles::TEXT_PRIMARY),
                            OverlayState::Hidden => ("", styles::TEXT_PRIMARY),
                        };
                        ui.label(
                            egui::RichText::new(label)
                                .color(color)
                                .font(FontId::proportional(13.0)),
                        );
                    });
                });
        },
    );
}

fn rest_position(ctx: &egui::Context) -> [f32; 2] {
    // Center horizontally on the parent screen, offset above the bottom.
    let screen = ctx.content_rect();
    let x = screen.center().x - styles::OVERLAY_WIDTH / 2.0;
    let y = screen.bottom() - styles::OVERLAY_HEIGHT - styles::OVERLAY_BOTTOM_OFFSET;
    [x.max(0.0), y.max(0.0)]
}

fn draw_level_bars(ui: &mut egui::Ui, rect: Rect, raw: f32, phase: f32) {
    let bars = 3;
    let bar_w = 3.0;
    let gap = 4.0;
    let total = bars as f32 * bar_w + (bars - 1) as f32 * gap;
    let x0 = rect.center().x - total / 2.0;
    let base = 0.22 + 0.78 * raw.clamp(0.0, 1.0);

    for i in 0..bars {
        let shape = if i == 1 { 1.0 } else { 0.75 };
        let wobble = 0.15 * (phase + i as f32 * 1.7).sin();
        let lvl = (base * shape + wobble * base).clamp(0.18, 1.0);
        let h = (lvl * rect.height()).max(4.0);
        let x = x0 + i as f32 * (bar_w + gap);
        let y = rect.center().y - h / 2.0;
        let bar_rect = Rect::from_min_size(egui::pos2(x, y), Vec2::new(bar_w, h));
        ui.painter().rect_filled(
            bar_rect,
            egui::CornerRadius::same((bar_w / 2.0) as u8),
            styles::ACCENT_RED.linear_multiply(0.85),
        );
    }
}

fn draw_spinner(ui: &mut egui::Ui, rect: Rect, phase: f32) {
    let painter = ui.painter();
    let center = rect.center();
    let radius = (rect.width().min(rect.height()) / 2.0) - 3.5;
    // background ring
    painter.circle_stroke(
        center,
        radius,
        Stroke::new(2.0, styles::ACCENT_BLUE.linear_multiply(0.15)),
    );
    // moving arc
    let segments = 24;
    let arc_len = std::f32::consts::FRAC_PI_2; // quarter
    let start = phase * 1.5;
    for i in 0..segments {
        let t0 = i as f32 / segments as f32;
        let t1 = (i + 1) as f32 / segments as f32;
        let a0 = start + t0 * arc_len;
        let a1 = start + t1 * arc_len;
        let p0 = center + Vec2::angled(a0) * radius;
        let p1 = center + Vec2::angled(a1) * radius;
        painter.line_segment([p0, p1], Stroke::new(2.0, styles::ACCENT_BLUE));
    }
}
