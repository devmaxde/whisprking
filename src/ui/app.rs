//! Top-level eframe app: side-nav with Meeting / History / Settings.
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
use crate::ui::overlay::{show_overlay, LevelSource, OverlayState};
use crate::ui::pages::{HistoryPage, MeetingPage, SettingsPage};
use crate::ui::styles;
use crate::ui::tray::{TrayEvent, TrayHandle, TrayState};

#[derive(Debug, Clone)]
pub enum AppEvent {
    HotkeyChanged(String),
    Quit,
}

/// Inputs from the host into the app for the next frame.
#[derive(Default)]
pub struct AppInputs {
    /// Most recent dictation overlay state computed by the host.
    pub overlay_state: OverlayState,
    /// Latest known live audio level for the overlay meter.
    pub audio_level: f32,
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
        event_tx: Sender<AppEvent>,
        inputs_rx: Receiver<AppInputs>,
    ) -> Self {
        styles::apply_theme(&cc.egui_ctx);

        let meeting = match engine {
            Some(e) => match MeetingPage::new(&config, e) {
                Ok(p) => Some(p),
                Err(err) => {
                    log::warn!("meeting page disabled: {err}");
                    None
                }
            },
            None => None,
        };

        let tray = match TrayHandle::new(cc.egui_ctx.clone()) {
            Ok(t) => Some(t),
            Err(e) => {
                log::warn!("tray icon disabled: {e}");
                None
            }
        };

        Self {
            config,
            tab: Tab::Settings,
            meeting,
            history: HistoryPage::new(),
            settings: SettingsPage::new(),
            history_loaded: false,
            level_source: LevelSource::default(),
            overlay_state: OverlayState::Hidden,
            spinner_phase: 0.0,
            event_tx,
            inputs_rx,
            tray,
            quitting: false,
        }
    }

    pub fn tray_set_state(&self, state: TrayState) {
        if let Some(t) = &self.tray {
            t.set_state(state);
        }
    }

    fn drain_tray(&mut self, ctx: &egui::Context) {
        let Some(tray) = &self.tray else { return };
        while let Some(evt) = tray.try_recv() {
            match evt {
                TrayEvent::ShowWindow => show_and_focus(ctx),
                TrayEvent::OpenMeeting => {
                    self.tab = Tab::Meeting;
                    show_and_focus(ctx);
                }
                TrayEvent::OpenSettings => {
                    self.tab = Tab::Settings;
                    show_and_focus(ctx);
                }
                TrayEvent::Quit => {
                    self.quitting = true;
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
            }
        }
    }

    pub fn level_source(&self) -> LevelSource {
        self.level_source.clone()
    }

    fn drain_inputs(&mut self) {
        while let Ok(inp) = self.inputs_rx.try_recv() {
            self.overlay_state = inp.overlay_state;
            self.level_source.set(inp.audio_level);
            if inp.focus_meeting {
                self.tab = Tab::Meeting;
            }
            if inp.focus_settings {
                self.tab = Tab::Settings;
            }
        }
    }
}

fn show_and_focus(ctx: &egui::Context) {
    ctx.send_viewport_cmd(egui::ViewportCommand::Visible(true));
    ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
}

impl eframe::App for WhisprKingApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.drain_inputs();
        self.drain_tray(&ctx);

        // Close button → hide window, keep tray + hotkey alive. Real quit
        // comes from the tray "Beenden" entry which sets `quitting` first.
        if ctx.input(|i| i.viewport().close_requested()) && !self.quitting {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            ctx.send_viewport_cmd(egui::ViewportCommand::Visible(false));
        }

        show_overlay(
            &ctx,
            self.overlay_state,
            &self.level_source,
            &mut self.spinner_phase,
        );

        egui::Panel::left("nav")
            .resizable(false)
            .exact_size(160.0)
            .show_inside(ui, |ui| {
                ui.add_space(12.0);
                if ui
                    .selectable_label(self.tab == Tab::Meeting, "🎙  Meeting")
                    .clicked()
                {
                    self.tab = Tab::Meeting;
                }
                if ui
                    .selectable_label(self.tab == Tab::History, "📄  Verlauf")
                    .clicked()
                {
                    if !self.history_loaded {
                        self.history.refresh(&self.config);
                        self.history_loaded = true;
                    }
                    self.tab = Tab::History;
                }
                if ui
                    .selectable_label(self.tab == Tab::Settings, "⚙  Einstellungen")
                    .clicked()
                {
                    self.tab = Tab::Settings;
                }
            });

        egui::CentralPanel::default().show_inside(ui, |ui| match self.tab {
            Tab::Meeting => match self.meeting.as_mut() {
                Some(page) => page.ui(ui, &mut self.config),
                None => {
                    ui.add_space(20.0);
                    ui.label("Engine not loaded — install a model first.");
                }
            },
            Tab::History => {
                if !self.history_loaded {
                    self.history.refresh(&self.config);
                    self.history_loaded = true;
                }
                self.history.ui(ui, &self.config);
            }
            Tab::Settings => {
                self.settings.ui(ui, &mut self.config);
                if let Some(hk) = self.settings.hotkey_changed.take() {
                    let _ = self.event_tx.send(AppEvent::HotkeyChanged(hk));
                }
                if self.settings.data_dir_changed {
                    self.history_loaded = false;
                    self.settings.clear_signals();
                }
            }
        });

        // Repaint continuously while the overlay is up so the level meter
        // stays smooth.
        if self.overlay_state != OverlayState::Hidden {
            ctx.request_repaint();
        }
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        let _ = self.event_tx.send(AppEvent::Quit);
    }
}

/// Run the egui main loop. Blocks until the user closes the window.
pub fn run(
    config: Config,
    engine: Option<Box<dyn Transcriber>>,
) -> Result<RunHandles, anyhow::Error> {
    let (event_tx, event_rx) = mpsc::channel::<AppEvent>();
    let (inputs_tx, inputs_rx) = mpsc::channel::<AppInputs>();

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("WhisprKing")
            .with_inner_size(styles::WINDOW_DEFAULT_SIZE)
            .with_min_inner_size(styles::WINDOW_MIN_SIZE),
        ..Default::default()
    };

    eframe::run_native(
        "WhisprKing",
        native_options,
        Box::new(move |cc| {
            Ok(Box::new(WhisprKingApp::new(
                cc, config, engine, event_tx, inputs_rx,
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
