//! Hold-to-talk global hotkey.
//!
//! macOS 14+ enforces main-thread-only access to Carbon TSM
//! (`TSMGetInputSourceProperty`). The `rdev` 0.5 macOS backend calls TSM
//! from inside its tap callback to compute a Unicode `name` for every
//! key, which crashes the process with `_dispatch_assert_queue_fail` on
//! the very first event. To work around this we install our own
//! `CGEventTap` on macOS — it only reads the integer keycode + event
//! type, no TSM, no Unicode lookup, so it is safe to run on a worker
//! thread.
//!
//! On other platforms we keep using `rdev` since the TSM issue is
//! macOS-specific.
//!
//! Callbacks fire on the tap runloop thread — keep them short and shunt
//! work onto your own queue.

use std::str::FromStr;
use std::sync::Arc;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum HotkeyError {
    #[error("unknown hotkey: {name}. options: {opts}", name = .0, opts = HotkeyName::all_names().join(", "))]
    Unknown(String),
    #[error("failed to install macOS event tap (Accessibility permission required)")]
    TapInstall,
    #[cfg(not(target_os = "macos"))]
    #[error("rdev listener error: {0:?}")]
    Listener(rdev::ListenError),
}

/// Named keys that the dictation hotkey may be bound to. The list mirrors
/// `HOTKEY_MAP` in `hotkey/listener.py`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HotkeyName {
    RightCmd,
    LeftCmd,
    RightOption,
    LeftOption,
    RightCtrl,
    RightShift,
    F18,
    F19,
}

impl HotkeyName {
    pub const fn all() -> &'static [HotkeyName] {
        use HotkeyName::*;
        &[
            RightCmd,
            LeftCmd,
            RightOption,
            LeftOption,
            RightCtrl,
            RightShift,
            F18,
            F19,
        ]
    }

    pub fn all_names() -> Vec<&'static str> {
        Self::all().iter().map(|h| h.as_str()).collect()
    }

    pub fn as_str(self) -> &'static str {
        match self {
            HotkeyName::RightCmd => "right_cmd",
            HotkeyName::LeftCmd => "left_cmd",
            HotkeyName::RightOption => "right_option",
            HotkeyName::LeftOption => "left_option",
            HotkeyName::RightCtrl => "right_ctrl",
            HotkeyName::RightShift => "right_shift",
            HotkeyName::F18 => "f18",
            HotkeyName::F19 => "f19",
        }
    }
}

impl FromStr for HotkeyName {
    type Err = HotkeyError;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        for h in Self::all() {
            if h.as_str() == name {
                return Ok(*h);
            }
        }
        Err(HotkeyError::Unknown(name.to_string()))
    }
}

/// Trait so the listener stays testable. The closures used in `new()`
/// implement this automatically.
pub trait HotkeyHandler: Send + Sync + 'static {
    fn on_activate(&self);
    fn on_deactivate(&self);
}

impl<A, D> HotkeyHandler for (A, D)
where
    A: Fn() + Send + Sync + 'static,
    D: Fn() + Send + Sync + 'static,
{
    fn on_activate(&self) {
        (self.0)()
    }
    fn on_deactivate(&self) {
        (self.1)()
    }
}

// ─── public facade ──────────────────────────────────────────────────────────

/// Hold-to-talk listener. `start()` spawns one OS-level event tap; the
/// tap runs until the process exits. `update_hotkey()` swaps the target
/// key without restarting the tap.
pub struct HotkeyListener {
    #[cfg(target_os = "macos")]
    backend: macos::Backend,
    #[cfg(not(target_os = "macos"))]
    backend: rdev_backend::Backend,
}

impl HotkeyListener {
    pub fn new(name: &str, handler: Arc<dyn HotkeyHandler>) -> Result<Self, HotkeyError> {
        let target = HotkeyName::from_str(name)?;
        #[cfg(target_os = "macos")]
        {
            Ok(Self {
                backend: macos::Backend::new(target, handler),
            })
        }
        #[cfg(not(target_os = "macos"))]
        {
            Ok(Self {
                backend: rdev_backend::Backend::new(target, handler),
            })
        }
    }

    pub fn start(&mut self) {
        self.backend.start()
    }

    /// Swap the target key. Safe to call while the listener is running.
    pub fn update_hotkey(&self, name: &str) -> Result<(), HotkeyError> {
        let target = HotkeyName::from_str(name)?;
        self.backend.update(target);
        Ok(())
    }
}

// ─── macOS backend: CGEventTap, no TSM ──────────────────────────────────────

#[cfg(target_os = "macos")]
mod macos {
    use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
    use std::sync::Arc;
    use std::thread;

    use core_foundation::runloop::{kCFRunLoopCommonModes, CFRunLoop};
    use core_graphics::event::{
        CGEventTap, CGEventTapLocation, CGEventTapOptions, CGEventTapPlacement, CGEventType,
        CallbackResult, EventField,
    };

    use super::{HotkeyHandler, HotkeyName};

    // macOS virtual keycodes from `<HIToolbox/Events.h>`.
    const VK_RIGHT_CMD: i64 = 54;
    const VK_LEFT_CMD: i64 = 55;
    const VK_LEFT_OPT: i64 = 58;
    const VK_RIGHT_OPT: i64 = 61;
    const VK_RIGHT_CTRL: i64 = 62;
    const VK_RIGHT_SHIFT: i64 = 60;
    const VK_F18: i64 = 79;
    const VK_F19: i64 = 80;

    fn keycode_of(name: HotkeyName) -> i64 {
        match name {
            HotkeyName::RightCmd => VK_RIGHT_CMD,
            HotkeyName::LeftCmd => VK_LEFT_CMD,
            HotkeyName::RightOption => VK_RIGHT_OPT,
            HotkeyName::LeftOption => VK_LEFT_OPT,
            HotkeyName::RightCtrl => VK_RIGHT_CTRL,
            HotkeyName::RightShift => VK_RIGHT_SHIFT,
            HotkeyName::F18 => VK_F18,
            HotkeyName::F19 => VK_F19,
        }
    }

    pub struct Backend {
        target_kc: Arc<AtomicI64>,
        pressed: Arc<AtomicBool>,
        handler: Arc<dyn HotkeyHandler>,
        started: bool,
    }

    impl Backend {
        pub fn new(name: HotkeyName, handler: Arc<dyn HotkeyHandler>) -> Self {
            Self {
                target_kc: Arc::new(AtomicI64::new(keycode_of(name))),
                pressed: Arc::new(AtomicBool::new(false)),
                handler,
                started: false,
            }
        }

        pub fn update(&self, name: HotkeyName) {
            self.target_kc.store(keycode_of(name), Ordering::SeqCst);
            self.pressed.store(false, Ordering::SeqCst);
        }

        pub fn start(&mut self) {
            if self.started {
                log::debug!("hotkey listener already running");
                return;
            }
            self.started = true;

            let target_kc = Arc::clone(&self.target_kc);
            let pressed = Arc::clone(&self.pressed);
            let handler = Arc::clone(&self.handler);

            log::info!(
                "hotkey listener spawning — CGEventTap (no rdev/TSM). \
                 Requires macOS Accessibility permission."
            );

            thread::Builder::new()
                .name("whisprking-hotkey".into())
                .spawn(move || run_tap_loop(target_kc, pressed, handler))
                .expect("spawn hotkey thread");
        }
    }

    fn run_tap_loop(
        target_kc: Arc<AtomicI64>,
        pressed: Arc<AtomicBool>,
        handler: Arc<dyn HotkeyHandler>,
    ) {
        log::info!("hotkey thread alive, installing CGEventTap");

        let tap = CGEventTap::new(
            CGEventTapLocation::HID,
            CGEventTapPlacement::HeadInsertEventTap,
            CGEventTapOptions::ListenOnly,
            vec![
                CGEventType::KeyDown,
                CGEventType::KeyUp,
                CGEventType::FlagsChanged,
            ],
            move |_proxy, etype, event| {
                let kc = event.get_integer_value_field(EventField::KEYBOARD_EVENT_KEYCODE);
                if kc != target_kc.load(Ordering::SeqCst) {
                    return CallbackResult::Keep;
                }
                let is_press = match etype {
                    CGEventType::KeyDown => true,
                    CGEventType::KeyUp => false,
                    // Modifier keys only fire FlagsChanged; we infer
                    // press vs. release from our own toggle since the
                    // CGEventFlags bitmask only encodes the merged
                    // L+R state per modifier (no per-side bit).
                    CGEventType::FlagsChanged => !pressed.load(Ordering::SeqCst),
                    _ => return CallbackResult::Keep,
                };
                if is_press {
                    if !pressed.swap(true, Ordering::SeqCst) {
                        log::info!("hotkey activate");
                        handler.on_activate();
                    }
                } else if pressed.swap(false, Ordering::SeqCst) {
                    log::info!("hotkey deactivate");
                    handler.on_deactivate();
                }
                CallbackResult::Keep
            },
        );
        let tap = match tap {
            Ok(t) => t,
            Err(()) => {
                log::error!(
                    "CGEventTapCreate failed — Accessibility likely not granted. \
                     Open System Settings → Privacy & Security → Accessibility, \
                     enable WhisprKing, then restart."
                );
                return;
            }
        };

        let source = match tap.mach_port().create_runloop_source(0) {
            Ok(s) => s,
            Err(()) => {
                log::error!("CFMachPortCreateRunLoopSource failed");
                return;
            }
        };

        let rl = CFRunLoop::get_current();
        // SAFETY: `kCFRunLoopCommonModes` is a constant CFString published
        // by CoreFoundation; reading it is unsafe only because it is an
        // extern static.
        let mode = unsafe { kCFRunLoopCommonModes };
        rl.add_source(&source, mode);
        tap.enable();

        log::info!("CGEventTap enabled; entering CFRunLoop");
        CFRunLoop::run_current();
        log::warn!("CFRunLoop exited (hotkey thread shutting down)");
        // Keep tap alive until run loop returns.
        drop(tap);
    }
}

// ─── non-macOS backend: rdev (unchanged) ────────────────────────────────────

#[cfg(not(target_os = "macos"))]
mod rdev_backend {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    use std::thread::{self, JoinHandle};

    use rdev::{listen, Event, EventType, Key};

    use super::{HotkeyHandler, HotkeyName};

    fn rdev_key(name: HotkeyName) -> Key {
        const F18_MAC: u32 = 79;
        const F19_MAC: u32 = 80;
        match name {
            HotkeyName::RightCmd => Key::MetaRight,
            HotkeyName::LeftCmd => Key::MetaLeft,
            HotkeyName::RightOption => Key::AltGr,
            HotkeyName::LeftOption => Key::Alt,
            HotkeyName::RightCtrl => Key::ControlRight,
            HotkeyName::RightShift => Key::ShiftRight,
            HotkeyName::F18 => Key::Unknown(F18_MAC),
            HotkeyName::F19 => Key::Unknown(F19_MAC),
        }
    }

    pub struct Backend {
        target: Arc<Mutex<Key>>,
        pressed: Arc<AtomicBool>,
        handler: Arc<dyn HotkeyHandler>,
        thread: Option<JoinHandle<()>>,
    }

    impl Backend {
        pub fn new(name: HotkeyName, handler: Arc<dyn HotkeyHandler>) -> Self {
            Self {
                target: Arc::new(Mutex::new(rdev_key(name))),
                pressed: Arc::new(AtomicBool::new(false)),
                handler,
                thread: None,
            }
        }

        pub fn update(&self, name: HotkeyName) {
            *self.target.lock().expect("hotkey target") = rdev_key(name);
            self.pressed.store(false, Ordering::SeqCst);
        }

        pub fn start(&mut self) {
            if self.thread.is_some() {
                return;
            }
            let target = Arc::clone(&self.target);
            let pressed = Arc::clone(&self.pressed);
            let handler = Arc::clone(&self.handler);
            let thread = thread::Builder::new()
                .name("whisprking-hotkey".into())
                .spawn(move || {
                    let cb = move |event: Event| {
                        let current = *target.lock().expect("hotkey target");
                        match event.event_type {
                            EventType::KeyPress(k) if k == current => {
                                if !pressed.swap(true, Ordering::SeqCst) {
                                    handler.on_activate();
                                }
                            }
                            EventType::KeyRelease(k) if k == current => {
                                if pressed.swap(false, Ordering::SeqCst) {
                                    handler.on_deactivate();
                                }
                            }
                            _ => {}
                        }
                    };
                    if let Err(err) = listen(cb) {
                        log::error!("rdev::listen failed: {err:?}");
                    }
                })
                .expect("spawn hotkey thread");
            self.thread = Some(thread);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_known_names() {
        assert_eq!(
            HotkeyName::from_str("right_cmd").unwrap(),
            HotkeyName::RightCmd
        );
        assert_eq!(HotkeyName::from_str("f19").unwrap(), HotkeyName::F19);
    }

    #[test]
    fn rejects_unknown_names() {
        assert!(matches!(
            HotkeyName::from_str("right_pinky"),
            Err(HotkeyError::Unknown(_))
        ));
    }
}
