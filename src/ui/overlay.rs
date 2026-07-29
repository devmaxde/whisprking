//! Floating dictation overlay rendered as a borderless, always-on-top
//! egui viewport.
//!
//! The overlay shows a live level meter while recording and a progress ring
//! while transcribing. It is driven from the main app's update loop — when
//! [`OverlayState::Hidden`], no viewport is requested and nothing renders.

use std::sync::atomic::{AtomicU32, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

use egui::{Color32, Rect, Sense, Stroke, Vec2, ViewportBuilder, ViewportId};

use super::theme::{self, radius, space, text};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum OverlayState {
	#[default]
	Hidden,
	Recording,
	Transcribing,
}

impl OverlayState {
	fn to_bits(self) -> u8 {
		match self {
			OverlayState::Hidden => 0,
			OverlayState::Recording => 1,
			OverlayState::Transcribing => 2,
		}
	}

	fn from_bits(bits: u8) -> Self {
		match bits {
			1 => OverlayState::Recording,
			2 => OverlayState::Transcribing,
			_ => OverlayState::Hidden,
		}
	}
}

/// What the hold-to-talk pipeline is doing, shared between the dictation
/// worker thread (writer) and the egui thread (reader).
///
/// Every write also wakes the UI thread. That is not an optimization: for
/// dictation the main window is normally hidden, and eframe only runs the
/// app again when something requests a repaint. Without the wake, pressing
/// the hotkey would change this state and nothing would ever draw it.
#[derive(Clone, Default)]
pub struct DictationStatus {
	state: Arc<AtomicU8>,
	level: LevelSource,
	/// Set once the egui context exists — the status object is created
	/// before eframe boots so it can be handed to the dictation worker.
	ctx: Arc<Mutex<Option<egui::Context>>>,
}

impl DictationStatus {
	/// Give the status the context to wake. Called from the app's
	/// constructor, i.e. the first moment a context exists.
	pub fn attach(
		&self,
		ctx: egui::Context,
	) {
		*self.ctx.lock().expect("overlay ctx") = Some(ctx);
	}

	pub fn set_state(
		&self,
		state: OverlayState,
	) {
		self.state.store(state.to_bits(), Ordering::Relaxed);
		self.wake();
	}

	pub fn state(&self) -> OverlayState {
		OverlayState::from_bits(self.state.load(Ordering::Relaxed))
	}

	pub fn set_level(
		&self,
		level: f32,
	) {
		self.level.set(level);
		self.wake();
	}

	pub fn level(&self) -> f32 {
		self.level.get()
	}

	fn wake(&self) {
		if let Some(ctx) = self.ctx.lock().expect("overlay ctx").as_ref() {
			ctx.request_repaint();
		}
	}
}

/// Lock-free shared `f32` for the audio level. The recorder writes from
/// its callback thread; the UI thread reads each frame.
#[derive(Clone, Default)]
pub struct LevelSource(Arc<AtomicU32>);

impl LevelSource {
	pub fn set(
		&self,
		value: f32,
	) {
		self.0.store(value.to_bits(), Ordering::Relaxed);
	}
	pub fn get(&self) -> f32 {
		f32::from_bits(self.0.load(Ordering::Relaxed))
	}
}

/// Renders the overlay viewport. Call once per egui frame from the app's
/// update. State is owned by the caller so the overlay is just a view.
pub fn show_overlay(
	ctx: &egui::Context,
	state: OverlayState,
	level: &LevelSource,
	spinner_phase: &mut f32,
) {
	if state == OverlayState::Hidden {
		return;
	}

	// Drive the parent: an immediate viewport only renders when the frame
	// that declares it runs, and the main window is usually hidden while
	// dictating, so nothing else is asking for repaints.
	ctx.request_repaint_after(std::time::Duration::from_millis(33));

	let viewport = ViewportBuilder::default()
		.with_title("WhisprKing Overlay")
		.with_decorations(false)
		.with_always_on_top()
		.with_transparent(true)
		.with_resizable(false)
		.with_taskbar(false)
		// The overlay must never become the key window: the Cmd+V we
		// synthesize after transcribing has to land in whatever the user
		// was typing into, and clicks have to reach the app underneath.
		.with_active(false)
		.with_mouse_passthrough(true)
		.with_inner_size([theme::OVERLAY_WIDTH, theme::OVERLAY_HEIGHT])
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
				.fill(Color32::from_rgba_unmultiplied(16, 17, 23, 235))
				.stroke(Stroke::new(1.0, theme::tint(Color32::WHITE, 0.08)))
				.inner_margin(egui::Margin::symmetric(space::LG as i8, space::MD as i8))
				.corner_radius(radius::LG);

			egui::CentralPanel::default().frame(frame).show(ui, |ui| {
				ui.horizontal_centered(|ui| {
					let (rect, _) = ui.allocate_exact_size(Vec2::splat(24.0), Sense::hover());
					match state {
						OverlayState::Recording => {
							draw_level_bars(ui, rect, level.get(), *spinner_phase);
						},
						OverlayState::Transcribing => {
							draw_spinner(ui, rect, *spinner_phase);
						},
						OverlayState::Hidden => {},
					}
					ui.add_space(space::MD);

					let (label, color) = match state {
						OverlayState::Recording => ("Ich höre zu …", theme::TEXT_PRIMARY),
						OverlayState::Transcribing => {
							("Wird transkribiert …", theme::TEXT_PRIMARY)
						},
						OverlayState::Hidden => ("", theme::TEXT_PRIMARY),
					};
					ui.add(egui::Label::new(
						egui::RichText::new(label).font(text::strong()).color(color),
					));
				});
			});
		},
	);
}

/// Bottom centre of the monitor, in the screen coordinates a viewport
/// position wants.
///
/// This deliberately uses the monitor size and not `content_rect()`: the
/// latter is the main window's own rect, so the overlay would follow the
/// window around — including off to wherever it was left before being
/// hidden — instead of sitting at the bottom of the display.
fn rest_position(ctx: &egui::Context) -> [f32; 2] {
	let screen = ctx
		.input(|i| i.viewport().monitor_size)
		.unwrap_or_else(|| ctx.content_rect().size());
	let x = (screen.x - theme::OVERLAY_WIDTH) / 2.0;
	let y = screen.y - theme::OVERLAY_HEIGHT - theme::OVERLAY_BOTTOM_OFFSET;
	[x.max(0.0), y.max(0.0)]
}

fn draw_level_bars(
	ui: &mut egui::Ui,
	rect: Rect,
	raw: f32,
	phase: f32,
) {
	let bars = 4;
	let bar_w = 3.0;
	let gap = 3.5;
	let total = bars as f32 * bar_w + (bars - 1) as f32 * gap;
	let x0 = rect.center().x - total / 2.0;
	let base = 0.22 + 0.78 * raw.clamp(0.0, 1.0);

	for i in 0..bars {
		let shape = if i == 1 || i == 2 { 1.0 } else { 0.7 };
		let wobble = 0.15 * (phase + i as f32 * 1.7).sin();
		let lvl = (base * shape + wobble * base).clamp(0.18, 1.0);
		let h = (lvl * rect.height()).max(4.0);
		let x = x0 + i as f32 * (bar_w + gap);
		let y = rect.center().y - h / 2.0;
		let bar_rect = Rect::from_min_size(egui::pos2(x, y), Vec2::new(bar_w, h));
		ui.painter()
			.rect_filled(bar_rect, radius::PILL, theme::DANGER);
	}
}

fn draw_spinner(
	ui: &mut egui::Ui,
	rect: Rect,
	phase: f32,
) {
	let painter = ui.painter();
	let center = rect.center();
	let radius = (rect.width().min(rect.height()) / 2.0) - 3.5;
	// background ring
	painter.circle_stroke(
		center,
		radius,
		Stroke::new(2.5, theme::tint(theme::ACCENT, 0.25)),
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
		painter.line_segment([p0, p1], Stroke::new(2.5, theme::ACCENT_HOVER));
	}
}
