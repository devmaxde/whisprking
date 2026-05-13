//! Menu-bar tray icon.
//!
//! tray-icon ships its own event channel; we map the menu-event ids to
//! strongly-typed [`TrayEvent`]s and surface a single `try_recv` for the
//! main loop to poll.
//!
//! State changes (idle / recording / transcribing / meeting) recolor the
//! microphone icon. The icon is drawn programmatically into an RGBA buffer
//! so we ship no asset files.

use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};

use tray_icon::menu::{Menu, MenuEvent, MenuItem, PredefinedMenuItem};
use tray_icon::{Icon, TrayIcon, TrayIconBuilder};

use super::styles;

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

    fn color(self) -> [u8; 4] {
        let c = match self {
            TrayState::Idle => styles::TEXT_PRIMARY,
            TrayState::Recording => styles::ACCENT_RED,
            TrayState::Transcribing => styles::ACCENT_BLUE,
            TrayState::Meeting => styles::ACCENT_GREEN,
        };
        [c.r(), c.g(), c.b(), 255]
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
        let meeting = MenuItem::new("Meeting starten…", true, None);
        let settings = MenuItem::new("Einstellungen…", true, None);
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

    pub fn set_state(&self, state: TrayState) {
        {
            let mut guard = self.state.lock().expect("tray state");
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

/// Render the microphone glyph used in the menu bar. Programmatic — no
/// asset files in the repo.
fn make_mic_icon(rgba: [u8; 4]) -> Icon {
    let size = ICON_SIZE as i32;
    let mut buf = vec![0u8; (ICON_SIZE * ICON_SIZE * 4) as usize];

    let cx = size as f32 / 2.0;
    let cy = size as f32 * 0.42;
    let capsule_w = size as f32 * 0.42;
    let capsule_h = size as f32 * 0.55;
    let radius = capsule_w / 2.0;

    for y in 0..size {
        for x in 0..size {
            let fx = x as f32 + 0.5;
            let fy = y as f32 + 0.5;
            let mut alpha = 0u8;

            if in_capsule(fx, fy, cx, cy, capsule_w, capsule_h, radius) {
                alpha = 255;
            }

            // stem
            let stem_top = cy + capsule_h * 0.55;
            let stem_bot = size as f32 * 0.88;
            if fx >= cx - 0.9 && fx <= cx + 0.9 && fy >= stem_top && fy <= stem_bot {
                alpha = 255;
            }
            // base bar
            if fx >= cx - size as f32 * 0.18
                && fx <= cx + size as f32 * 0.18
                && (fy - stem_bot).abs() < 1.0
            {
                alpha = 255;
            }

            let idx = ((y * size + x) * 4) as usize;
            if alpha > 0 {
                buf[idx] = rgba[0];
                buf[idx + 1] = rgba[1];
                buf[idx + 2] = rgba[2];
                buf[idx + 3] = alpha;
            }
        }
    }
    Icon::from_rgba(buf, ICON_SIZE, ICON_SIZE).expect("valid rgba")
}

fn in_capsule(fx: f32, fy: f32, cx: f32, cy: f32, w: f32, h: f32, r: f32) -> bool {
    let left = cx - w / 2.0;
    let right = cx + w / 2.0;
    let top = cy - h / 2.0 + r;
    let bot = cy + h / 2.0 - r;
    // central rectangle
    if fx >= left && fx <= right && fy >= top && fy <= bot {
        return true;
    }
    // top cap
    let top_dy = (fy - top).abs();
    if fy < top && fx >= left && fx <= right && top_dy <= r {
        let dx = fx - cx;
        let dy = fy - top;
        if dx * dx + dy * dy <= r * r {
            return true;
        }
    }
    // bottom cap
    if fy > bot && fx >= left && fx <= right {
        let dx = fx - cx;
        let dy = fy - bot;
        if dx * dx + dy * dy <= r * r {
            return true;
        }
    }
    false
}
