//! System-audio capability reporting.
//!
//! WhisprKing no longer needs BlackHole, a Multi-Output Device, or any
//! change to the user's audio routing — system audio is captured through a
//! Core Audio process tap instead (see [`super::tap`]). What remains here
//! is the capability probe the UI uses to explain the state of things, and
//! a check for a now-redundant BlackHole install so we can tell the user
//! they are free to remove it.

use cpal::traits::{DeviceTrait, HostTrait};

const BLACKHOLE_HINTS: &[&str] = &["blackhole"];

/// Why system-audio capture is unavailable, when it is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SystemAudioStatus {
    /// Process taps are available; nothing to install or configure.
    Available,
    /// The OS is too old for process taps (introduced in macOS 14.4).
    NeedsNewerMacOs,
    /// Not macOS.
    Unsupported,
}

impl SystemAudioStatus {
    pub fn is_available(&self) -> bool {
        matches!(self, SystemAudioStatus::Available)
    }

    /// Sentence shown under the system-audio toggle when it is off.
    pub fn hint(&self) -> Option<&'static str> {
        match self {
            SystemAudioStatus::Available => None,
            SystemAudioStatus::NeedsNewerMacOs => Some(
                "Die andere Seite aufzunehmen braucht macOS 14.4 oder neuer. \
                 Auf dieser Version kann WhisprKing nur dein Mikrofon aufnehmen.",
            ),
            SystemAudioStatus::Unsupported => {
                Some("System-Audio aufzunehmen geht nur unter macOS.")
            }
        }
    }
}

pub fn system_audio_status() -> SystemAudioStatus {
    #[cfg(target_os = "macos")]
    {
        if super::tap::is_supported() {
            SystemAudioStatus::Available
        } else {
            SystemAudioStatus::NeedsNewerMacOs
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        SystemAudioStatus::Unsupported
    }
}

/// Name of the first BlackHole input device, if one is still installed.
pub fn find_blackhole_device_name() -> Option<String> {
    let host = cpal::default_host();
    let devices = host.input_devices().ok()?;
    for dev in devices {
        let Ok(name) = dev.description().map(|d| d.name().to_string()) else {
            continue;
        };
        let lower = name.to_lowercase();
        if BLACKHOLE_HINTS.iter().any(|h| lower.contains(h)) {
            return Some(name);
        }
    }
    None
}

pub fn detect_blackhole() -> bool {
    find_blackhole_device_name().is_some()
}

/// Shown once when BlackHole is installed but no longer needed, so users
/// who set it up for an older WhisprKing know they can reclaim it.
pub const BLACKHOLE_NO_LONGER_NEEDED: &str =
    "BlackHole ist noch installiert, WhisprKing benutzt es aber nicht mehr. \
     Du kannst es deinstallieren und ein selbst angelegtes Multi-Output-Gerät \
     im Audio-MIDI-Setup wieder entfernen.";
