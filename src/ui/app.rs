//! Top-level eframe app: a navigation rail plus the active page.
//!
//! Owns the egui-side state and forwards "I changed the hotkey" / "I
//! changed the data dir" signals back to the host via [`AppEvent`].
//!
//! The dictation overlay and the tray icon are shown / hidden from the
//! same update loop so we have exactly one source of truth for state.

use std::sync::mpsc::{self, Receiver, Sender};

use eframe::egui;

use crate::config::Config;
use crate::transcription::engine::Transcriber;
use crate::ui::activation;
// Aliased: `brand` is also the name of the function that paints the mark
// into the navigation rail.
use crate::ui::brand as brand_mark;
use crate::ui::icons::{self, Icon};
use crate::ui::overlay::{show_overlay, DictationStatus, LevelSource, OverlayState};
use crate::ui::pages::{HistoryPage, MeetingPage, SettingsPage};
use crate::ui::theme::{self, radius, space, text};
use crate::ui::tray::{TrayEvent, TrayHandle, TrayState};
use crate::ui::widgets as w;

#[derive(Debug, Clone)]
pub enum AppEvent {
	HotkeyChanged(String),
	Quit,
}

/// Inputs from the host into the app for the next frame.
///
/// The dictation overlay is *not* driven from here: it is written from the
/// dictation worker thread into a shared [`DictationStatus`], because it
/// has to keep working while the main window is hidden.
#[derive(Default)]
pub struct AppInputs {
	/// Set true the frame the user clicked "open meeting page" in the
	/// tray; the app resets it after focusing the tab.
	pub focus_meeting: bool,
	/// Same for settings.
	pub focus_settings: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tab {
	Meeting,
	History,
	Settings,
}

impl Tab {
	const ALL: [Tab; 3] = [Tab::Meeting, Tab::History, Tab::Settings];

	fn label(self) -> &'static str {
		match self {
			Tab::Meeting => "Meeting",
			Tab::History => "Verlauf",
			Tab::Settings => "Einstellungen",
		}
	}

	fn icon(self) -> Icon {
		match self {
			Tab::Meeting => Icon::Mic,
			Tab::History => Icon::Transcript,
			Tab::Settings => Icon::Settings,
		}
	}
}

pub struct WhisprKingApp {
	config: Config,
	tab: Tab,
	meeting: Option<MeetingPage>,
	history: HistoryPage,
	settings: SettingsPage,
	history_loaded: bool,
	level_source: LevelSource,
	overlay_state: OverlayState,
	spinner_phase: f32,
	status: DictationStatus,
	event_tx: Sender<AppEvent>,
	inputs_rx: Receiver<AppInputs>,
	tray: Option<TrayHandle>,
	quitting: bool,
}

impl WhisprKingApp {
	pub fn new(
		cc: &eframe::CreationContext<'_>,
		config: Config,
		engine: Option<Box<dyn Transcriber>>,
		status: DictationStatus,
		event_tx: Sender<AppEvent>,
		inputs_rx: Receiver<AppInputs>,
	) -> Self {
		theme::apply_theme(&cc.egui_ctx);

		// The window is on screen from the first frame, so the app starts as
		// a normal foreground app. It only becomes an accessory (menu-bar
		// only) app once that window is closed.
		activation::set_regular();

		// From here on the dictation worker can wake the UI thread, which
		// is what makes the overlay appear while the window is hidden.
		status.attach(cc.egui_ctx.clone());

		let meeting = match engine {
			Some(e) => match MeetingPage::new(&config, e) {
				Ok(p) => Some(p),
				Err(err) => {
					log::warn!("meeting page disabled: {err}");
					None
				},
			},
			None => None,
		};

		let tray = match TrayHandle::new(cc.egui_ctx.clone()) {
			Ok(t) => Some(t),
			Err(e) => {
				log::warn!("tray icon disabled: {e}");
				None
			},
		};

		Self {
			config,
			tab: Tab::Meeting,
			meeting,
			history: HistoryPage::new(),
			settings: SettingsPage::new(),
			history_loaded: false,
			level_source: LevelSource::default(),
			overlay_state: OverlayState::Hidden,
			spinner_phase: 0.0,
			status,
			event_tx,
			inputs_rx,
			tray,
			quitting: false,
		}
	}

	pub fn tray_set_state(
		&self,
		state: TrayState,
	) {
		if let Some(t) = &self.tray {
			t.set_state(state);
		}
	}

	fn drain_tray(
		&mut self,
		ctx: &egui::Context,
	) {
		let Some(tray) = &self.tray else { return };
		while let Some(evt) = tray.try_recv() {
			match evt {
				TrayEvent::ShowWindow => show_and_focus(ctx),
				TrayEvent::OpenMeeting => {
					self.tab = Tab::Meeting;
					show_and_focus(ctx);
				},
				TrayEvent::OpenSettings => {
					self.tab = Tab::Settings;
					show_and_focus(ctx);
				},
				TrayEvent::Quit => {
					self.quitting = true;
					ctx.send_viewport_cmd(egui::ViewportCommand::Close);
				},
			}
		}
	}

	pub fn level_source(&self) -> LevelSource {
		self.level_source.clone()
	}

	/// Mirror what dictation (or a running meeting) is doing onto the menu
	/// bar glyph. `TrayHandle::set_state` rasterizes an icon, so it is a
	/// no-op when the state has not actually changed.
	fn sync_tray(&self) {
		let meeting = self
			.meeting
			.as_ref()
			.map(|m| m.state() != crate::ui::pages::MeetingState::Idle)
			.unwrap_or(false);

		let state = if meeting {
			TrayState::Meeting
		} else {
			match self.overlay_state {
				OverlayState::Recording => TrayState::Recording,
				OverlayState::Transcribing => TrayState::Transcribing,
				OverlayState::Hidden => TrayState::Idle,
			}
		};
		self.tray_set_state(state);
	}

	fn drain_inputs(&mut self) {
		while let Ok(inp) = self.inputs_rx.try_recv() {
			if inp.focus_meeting {
				self.tab = Tab::Meeting;
			}
			if inp.focus_settings {
				self.tab = Tab::Settings;
			}
		}
	}

	/// Left navigation rail: brand, tabs, and what the app is currently
	/// listening for.
	fn nav(
		&mut self,
		ui: &mut egui::Ui,
	) {
		let frame = egui::Frame::new()
			.fill(theme::BG_SUNKEN)
			.inner_margin(egui::Margin::symmetric(space::MD as i8, space::LG as i8));

		egui::Panel::left("nav")
			.resizable(false)
			.exact_size(theme::NAV_WIDTH)
			.show_separator_line(false)
			.frame(frame)
			.show(ui, |ui| {
				brand(ui);
				ui.add_space(space::XL);

				for tab in Tab::ALL {
					if nav_item(ui, tab.icon(), tab.label(), self.tab == tab) {
						self.tab = tab;
					}
					ui.add_space(2.0);
				}

				// Push the status block to the bottom of the rail. 80pt is
				// the height of the three lines it renders.
				ui.add_space((ui.available_height() - 80.0).max(space::XL));
				self.nav_status(ui);
			});
	}

	fn nav_status(
		&self,
		ui: &mut egui::Ui,
	) {
		let recording = self
			.meeting
			.as_ref()
			.map(|m| m.state() != crate::ui::pages::MeetingState::Idle)
			.unwrap_or(false);

		let (tone, label) = if recording {
			(theme::DANGER, "Meeting läuft")
		} else if self.meeting.is_some() {
			(theme::SUCCESS, "Bereit")
		} else {
			(theme::WARNING, "Kein Modell geladen")
		};

		ui.horizontal(|ui| {
			let (dot, _) = ui.allocate_exact_size(egui::vec2(8.0, 8.0), egui::Sense::hover());
			ui.painter().circle_filled(dot.center(), 3.5, tone);
			ui.add_space(2.0);
			w::text_line(ui, label, text::caption(), theme::TEXT_SECONDARY);
		});
		ui.add_space(space::XS);
		w::text_line(
			ui,
			&format!("Modell · {}", self.config.dictation.model),
			text::caption(),
			theme::TEXT_MUTED,
		);
		w::text_line(
			ui,
			&format!("Hotkey · {}", self.config.dictation.hotkey.replace('_', " ")),
			text::caption(),
			theme::TEXT_MUTED,
		);
	}
}

fn show_and_focus(ctx: &egui::Context) {
	// Before showing: an accessory app's windows get no Dock tile, are
	// skipped by Cmd+Tab and are ignored by window managers. Become a
	// normal app first, then the window is a normal window.
	activation::set_regular();
	ctx.send_viewport_cmd(egui::ViewportCommand::Visible(true));
	ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
}

/// App mark + name at the top of the rail.
///
/// This is the app icon itself, not a stand-in glyph: what identifies
/// WhisprKing in the rail is the same artwork as in Finder and the menu bar.
fn brand(ui: &mut egui::Ui) {
	ui.horizontal(|ui| {
		let (rect, _) = ui.allocate_exact_size(egui::Vec2::splat(30.0), egui::Sense::hover());
		brand_mark::paint_badge(ui, rect);
		ui.add_space(space::XS);
		w::text_line(ui, "WhisprKing", text::title(), theme::TEXT_PRIMARY);
	});
}

/// One row in the navigation rail. Returns `true` when clicked.
fn nav_item(
	ui: &mut egui::Ui,
	icon: Icon,
	label: &str,
	active: bool,
) -> bool {
	let width = ui.available_width();
	let (rect, resp) = ui.allocate_exact_size(egui::vec2(width, 34.0), egui::Sense::click());
	if !ui.is_rect_visible(rect) {
		return resp.clicked();
	}

	let painter = ui.painter();
	if active {
		painter.rect_filled(rect, radius::SM, theme::tint(theme::ACCENT, 0.16));
	} else if resp.hovered() {
		painter.rect_filled(rect, radius::SM, theme::BG_SURFACE);
	}

	let color = if active {
		theme::ACCENT_HOVER
	} else if resp.hovered() {
		theme::TEXT_PRIMARY
	} else {
		theme::TEXT_SECONDARY
	};

	let icon_rect = egui::Rect::from_center_size(
		egui::pos2(rect.left() + 19.0, rect.center().y),
		egui::Vec2::splat(16.0),
	);
	icons::paint(painter, icon, icon_rect, color);

	let galley = painter.layout_no_wrap(
		label.to_owned(),
		if active { text::strong() } else { text::body() },
		color,
	);
	painter.galley(
		egui::pos2(rect.left() + 36.0, rect.center().y - galley.size().y / 2.0),
		galley,
		color,
	);
	resp.clicked()
}

impl eframe::App for WhisprKingApp {
	/// Runs every frame *and* whenever a repaint is requested while the
	/// main window is hidden — unlike [`Self::ui`], which eframe skips
	/// entirely for an invisible viewport.
	///
	/// Hold-to-talk dictation happens with the window hidden nearly every
	/// time, so the overlay has to be declared from here or it would only
	/// ever show up for users who left the main window open.
	fn logic(
		&mut self,
		ctx: &egui::Context,
		_frame: &mut eframe::Frame,
	) {
		self.overlay_state = self.status.state();
		self.level_source.set(self.status.level());
		self.sync_tray();

		show_overlay(
			ctx,
			self.overlay_state,
			&self.level_source,
			&mut self.spinner_phase,
		);
	}

	fn ui(
		&mut self,
		ui: &mut egui::Ui,
		_frame: &mut eframe::Frame,
	) {
		let ctx = ui.ctx().clone();
		self.drain_inputs();
		self.drain_tray(&ctx);

		// Close button → hide window, keep tray + hotkey alive. Real quit
		// comes from the tray "Beenden" entry which sets `quitting` first.
		// Dropping back to accessory is what removes the now-window-less
		// Dock tile and keeps the overlay from stealing focus while
		// dictating.
		if ctx.input(|i| i.viewport().close_requested()) && !self.quitting {
			ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
			ctx.send_viewport_cmd(egui::ViewportCommand::Visible(false));
			activation::set_accessory();
		}

		self.nav(ui);

		let content = egui::Frame::new()
			.fill(theme::BG_BASE)
			.inner_margin(egui::Margin::symmetric(space::XL as i8, space::LG as i8));

		egui::CentralPanel::default()
			.frame(content)
			.show(ui, |ui| match self.tab {
				Tab::Meeting => match self.meeting.as_mut() {
					Some(page) => page.ui(ui, &mut self.config),
					None => {
						w::page_header(ui, "Meeting", "");
						ui.add_space(space::XL);
						w::empty_state(
							ui,
							Icon::Download,
							"Kein Modell installiert",
							"Lade in den Einstellungen ein Modell herunter, dann kann \
							 WhisprKing transkribieren.",
						);
					},
				},
				Tab::History => {
					if !self.history_loaded {
						self.history.refresh(&self.config);
						self.history_loaded = true;
					}
					self.history.ui(ui, &self.config);
				},
				Tab::Settings => {
					self.settings.ui(ui, &mut self.config);
					if let Some(hk) = self.settings.hotkey_changed.take() {
						let _ = self.event_tx.send(AppEvent::HotkeyChanged(hk));
					}
					if self.settings.data_dir_changed {
						self.history_loaded = false;
						self.settings.clear_signals();
					}
				},
			});
	}

	fn on_exit(
		&mut self,
		_gl: Option<&eframe::glow::Context>,
	) {
		let _ = self.event_tx.send(AppEvent::Quit);
	}
}

/// Run the egui main loop. Blocks until the user closes the window.
pub fn run(
	config: Config,
	engine: Option<Box<dyn Transcriber>>,
	status: DictationStatus,
) -> Result<RunHandles, anyhow::Error> {
	let (event_tx, event_rx) = mpsc::channel::<AppEvent>();
	let (inputs_tx, inputs_rx) = mpsc::channel::<AppInputs>();

	let native_options = eframe::NativeOptions {
		viewport: egui::ViewportBuilder::default()
			.with_title("WhisprKing")
			// The bundle's .icns is what Finder and the Dock use; this is
			// what an unbundled `just run` and non-macOS builds show.
			.with_icon(brand_mark::window_icon())
			.with_inner_size(theme::WINDOW_DEFAULT_SIZE)
			.with_min_inner_size(theme::WINDOW_MIN_SIZE),
		..Default::default()
	};

	eframe::run_native(
		"WhisprKing",
		native_options,
		Box::new(move |cc| {
			Ok(Box::new(WhisprKingApp::new(
				cc, config, engine, status, event_tx, inputs_rx,
			)))
		}),
	)
	.map_err(|e| anyhow::anyhow!("eframe: {e}"))?;

	Ok(RunHandles {
		event_rx,
		inputs_tx,
	})
}

/// Returned from [`run`] for tests / out-of-loop interaction. The struct
/// is intentionally unused by `main.rs` today because `run_native` blocks;
/// it is here so the API does not need to change once we move the host
/// side onto a different thread.
pub struct RunHandles {
	pub event_rx: Receiver<AppEvent>,
	pub inputs_tx: Sender<AppInputs>,
}
