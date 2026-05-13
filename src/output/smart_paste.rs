//! Smart paste: stash the current clipboard, copy `text`, synthesize Cmd+V
//! into the focused app, then restore the original clipboard.
//!
//! Cmd+V is dispatched via Quartz `CGEventPost`. The process must hold the
//! macOS Accessibility entitlement or the event is silently dropped.

use std::thread;
use std::time::Duration;

use arboard::Clipboard;
use thiserror::Error;

const PASTE_SETTLE: Duration = Duration::from_millis(150);

#[derive(Debug, Error)]
pub enum PasteError {
    #[error("clipboard error: {0}")]
    Clipboard(#[from] arboard::Error),
    #[error("keystroke synthesis only implemented on macOS")]
    PlatformUnsupported,
}

/// Set the clipboard to `text` without pasting.
pub fn copy_only(text: &str) -> Result<(), PasteError> {
    Clipboard::new()?.set_text(text.to_string())?;
    Ok(())
}

/// Stash → set → Cmd+V → restore. Returns `false` only when `text` is empty.
pub fn smart_paste(text: &str) -> Result<bool, PasteError> {
    if text.is_empty() {
        return Ok(false);
    }
    let mut cb = Clipboard::new()?;
    let previous = cb.get_text().unwrap_or_default();

    cb.set_text(text.to_string())?;
    send_cmd_v()?;
    thread::sleep(PASTE_SETTLE);

    // Best-effort restore. If the original was empty we still write empty
    // so the user doesn't see the dictated text lingering.
    cb.set_text(previous)?;
    Ok(true)
}

#[cfg(target_os = "macos")]
fn send_cmd_v() -> Result<(), PasteError> {
    use core_graphics::event::{CGEvent, CGEventFlags, CGEventTapLocation};
    use core_graphics::event_source::{CGEventSource, CGEventSourceStateID};

    const V_KEYCODE: u16 = 9;

    let source = CGEventSource::new(CGEventSourceStateID::HIDSystemState)
        .map_err(|_| PasteError::PlatformUnsupported)?;

    let down = CGEvent::new_keyboard_event(source.clone(), V_KEYCODE, true)
        .map_err(|_| PasteError::PlatformUnsupported)?;
    down.set_flags(CGEventFlags::CGEventFlagCommand);
    down.post(CGEventTapLocation::HID);

    let up = CGEvent::new_keyboard_event(source, V_KEYCODE, false)
        .map_err(|_| PasteError::PlatformUnsupported)?;
    up.set_flags(CGEventFlags::CGEventFlagCommand);
    up.post(CGEventTapLocation::HID);

    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn send_cmd_v() -> Result<(), PasteError> {
    Err(PasteError::PlatformUnsupported)
}
