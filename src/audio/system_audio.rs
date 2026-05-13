//! Detect the BlackHole virtual audio device on macOS.

use cpal::traits::{DeviceTrait, HostTrait};

const BLACKHOLE_HINTS: &[&str] = &["blackhole"];

pub const BLACKHOLE_INSTALL_URL: &str = "https://existential.audio/blackhole/";
pub const BLACKHOLE_INSTALL_HINT: &str =
    "Für System-Audio installiere BlackHole 2ch und richte einen \
     Multi-Output-Device in Audio-MIDI-Setup ein.\n\
     Download: https://existential.audio/blackhole/";

/// Name of the first BlackHole input device, if present.
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
