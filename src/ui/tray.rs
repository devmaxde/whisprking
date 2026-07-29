//! Menu-bar tray icon.
//!
//! tray-icon ships its own event channel; we map the menu-event ids to
//! strongly-typed [`TrayEvent`]s and surface a single `try_recv` for the
//! main loop to poll.
//!
//! State changes (idle / recording / transcribing / meeting) recolor the
//! icon. It is the same crowned microphone as the app icon, rasterized from
//! [`super::brand`] — no asset files, and one place to change the artwork.

use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};

use tray_icon::menu::{Menu, MenuEvent, MenuItem, PredefinedMenuItem};
use tray_icon::{Icon, TrayIcon, TrayIconBuilder};

use super::{brand, theme};

/// Menu bar height on macOS is 22 pt; the icon is drawn at that size and
/// anti-aliased into it.
const ICON_SIZE: u32 = 22;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrayState {
    Idle,
    Recording,
    Transcribing,
    Meeting,
}

impl TrayState {
    fn label(self) -> &'static str {
        match self {
            TrayState::Idle => "Bereit",
            TrayState::Recording => "Aufnahme läuft…",
            TrayState::Transcribing => "Transkribiere…",
            TrayState::Meeting => "Meeting läuft",
        }
    }

    fn color(self) -> egui::Color32 {
        match self {
            TrayState::Idle => theme::TEXT_PRIMARY,
            TrayState::Recording => theme::DANGER,
            TrayState::Transcribing => theme::ACCENT_HOVER,
            TrayState::Meeting => theme::SUCCESS,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum TrayEvent {
    ShowWindow,
    OpenMeeting,
    OpenSettings,
    Quit,
}

/// Owning handle for the tray icon. Drop it to remove the icon from the
/// menu bar.
pub struct TrayHandle {
    icon: TrayIcon,
    status_item: MenuItem,
    state: Arc<Mutex<TrayState>>,
    rx: Receiver<TrayEvent>,
}

impl TrayHandle {
    /// `ctx` is the egui context used to wake the UI thread when a menu
    /// event fires — tray-icon delivers menu events on its own thread, and
    /// without a `request_repaint` the eframe loop stays idle (especially
    /// while the window is hidden) and never drains them.
    pub fn new(ctx: egui::Context) -> anyhow::Result<Self> {
        let menu = Menu::new();
        let title = MenuItem::new("WhisprKing", false, None);
        let status = MenuItem::new(format!("Status: {}", TrayState::Idle.label()), false, None);
        let show = MenuItem::new("Fenster zeigen", true, None);
        let meeting = MenuItem::new("Meeting starten", true, None);
        let settings = MenuItem::new("Einstellungen", true, None);
        let quit = MenuItem::new("Beenden", true, None);

        menu.append(&title)?;
        menu.append(&status)?;
        menu.append(&PredefinedMenuItem::separator())?;
        menu.append(&show)?;
        menu.append(&meeting)?;
        menu.append(&settings)?;
        menu.append(&PredefinedMenuItem::separator())?;
        menu.append(&quit)?;

        let show_id = show.id().clone();
        let meeting_id = meeting.id().clone();
        let settings_id = settings.id().clone();
        let quit_id = quit.id().clone();

        let icon = TrayIconBuilder::new()
            .with_menu(Box::new(menu))
            .with_tooltip("WhisprKing")
            .with_icon(make_mic_icon(TrayState::Idle.color()))
            .build()?;

        let (tx, rx) = mpsc::channel::<TrayEvent>();
        let tx = Mutex::new(tx);
        let wake = ctx.clone();
        MenuEvent::set_event_handler(Some(move |event: MenuEvent| {
            let mapped = if event.id == show_id {
                Some(TrayEvent::ShowWindow)
            } else if event.id == meeting_id {
                Some(TrayEvent::OpenMeeting)
            } else if event.id == settings_id {
                Some(TrayEvent::OpenSettings)
            } else if event.id == quit_id {
                Some(TrayEvent::Quit)
            } else {
                None
            };
            if let Some(evt) = mapped {
                if let Ok(guard) = tx.lock() {
                    let _ = guard.send(evt);
                }
                wake.request_repaint();
            }
        }));

        Ok(Self {
            icon,
            status_item: status,
            state: Arc::new(Mutex::new(TrayState::Idle)),
            rx,
        })
    }

    /// Idempotent: the UI calls this every frame, and each real change
    /// rasterizes a new icon.
    pub fn set_state(&self, state: TrayState) {
        {
            let mut guard = self.state.lock().expect("tray state");
            if *guard == state {
                return;
            }
            *guard = state;
        }
        let _ = self.icon.set_icon(Some(make_mic_icon(state.color())));
        self.status_item
            .set_text(format!("Status: {}", state.label()));
    }

    pub fn state(&self) -> TrayState {
        *self.state.lock().expect("tray state")
    }

    /// Poll the global menu-event channel and translate the next event to
    /// a [`TrayEvent`]. Returns `None` if no event is pending.
    pub fn try_recv(&self) -> Option<TrayEvent> {
        self.rx.try_recv().ok()
    }
}

/// Rasterize the mark for the menu bar in the color of the current state.
fn make_mic_icon(color: egui::Color32) -> Icon {
    let rgba = brand::mark_rgba(ICON_SIZE, color, brand::MENUBAR_MARGIN);
    Icon::from_rgba(rgba, ICON_SIZE, ICON_SIZE).expect("valid rgba")
}
