//! One-shot cleanup of the BlackHole routing older versions installed.
//!
//! Up to and including the aggregate-device era, WhisprKing did three
//! things that outlived the app:
//!
//! 1. Created a public Multi-Output Device (`…whisprking.output`) and made
//!    it the **system default output**.
//! 2. Created a public Aggregate Device (`…whisprking.input`) that other
//!    applications could — and did — select as a microphone, where it fed
//!    them system audio on channel 1 instead of the user's voice.
//! 3. Overwrote the user's own input-device choice in `config.json`.
//!
//! All three only got undone by pressing "Reset" in the UI. Quitting or
//! crashing while configured left the machine permanently routed through a
//! device belonging to an app that was no longer running.
//!
//! This module reverses all of it at startup, unconditionally, whether or
//! not the config still claims the routing is active — a crash between
//! creating the devices and saving the config would otherwise leave
//! orphans that nothing ever cleans up.

#![cfg(target_os = "macos")]

use crate::config::Config;

use super::coreaudio;

/// UIDs the old implementation used. Kept here verbatim so the cleanup
/// keeps working after the constants are gone from the meeting page.
const LEGACY_OUTPUT_UID: &str = "com.devmaxde.whisprking.output";
const LEGACY_INPUT_UID: &str = "com.devmaxde.whisprking.input";

/// What the cleanup actually did, for logging and for telling the user.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct RepairReport {
    pub restored_output: Option<String>,
    pub destroyed_devices: Vec<String>,
    pub cleared_input_override: bool,
}

impl RepairReport {
    pub fn did_anything(&self) -> bool {
        self.restored_output.is_some()
            || !self.destroyed_devices.is_empty()
            || self.cleared_input_override
    }

    /// One-line summary for the UI, or `None` if nothing needed fixing.
    pub fn summary(&self) -> Option<String> {
        if !self.did_anything() {
            return None;
        }
        let mut parts = Vec::new();
        if let Some(name) = &self.restored_output {
            parts.push(format!("restored audio output to {name}"));
        }
        if !self.destroyed_devices.is_empty() {
            parts.push(format!(
                "removed {} leftover virtual device(s)",
                self.destroyed_devices.len()
            ));
        }
        if self.cleared_input_override {
            parts.push("reset the microphone selection".into());
        }
        Some(format!(
            "Cleaned up the old BlackHole routing: {}.",
            parts.join(", ")
        ))
    }
}

/// Undo any legacy routing and clear it out of `config`.
///
/// Safe to call on every launch — it is a no-op once there is nothing left
/// to clean up. Saves `config` itself if it changed anything.
pub fn repair(config: &mut Config) -> RepairReport {
    let mut report = RepairReport::default();

    let devices = coreaudio::list_devices().unwrap_or_default();
    let ours: Vec<_> = devices
        .iter()
        .filter(|d| d.uid == LEGACY_OUTPUT_UID || d.uid == LEGACY_INPUT_UID)
        .collect();

    if ours.is_empty() && !config.meeting.blackhole.configured {
        return report;
    }

    // If our multi-output is currently the default, move the user back to
    // a real device *before* destroying it — otherwise the HAL picks a
    // replacement on its own and the user gets silence or the wrong one.
    let current_default = coreaudio::default_output_uid().unwrap_or_default();
    if current_default == LEGACY_OUTPUT_UID || current_default.is_empty() {
        let saved = &config.meeting.blackhole;
        let candidates = [
            saved.previous_default_output_uid.as_str(),
            saved.speaker_uid.as_str(),
        ];
        for uid in candidates.into_iter().filter(|u| !u.is_empty()) {
            if uid == LEGACY_OUTPUT_UID {
                continue;
            }
            if coreaudio::set_default_output_by_uid(uid).is_ok() {
                let name = devices
                    .iter()
                    .find(|d| d.uid == uid)
                    .map(|d| d.name.clone())
                    .unwrap_or_else(|| uid.to_string());
                log::info!("legacy cleanup: default output restored to {name}");
                report.restored_output = Some(name);
                break;
            }
        }
        if report.restored_output.is_none() {
            log::warn!(
                "legacy cleanup: could not restore a previous output device — \
                 pick your speakers again in System Settings → Sound"
            );
        }
    }

    for dev in ours {
        match coreaudio::destroy_aggregate_by_uid(&dev.uid) {
            Ok(()) => {
                log::info!("legacy cleanup: destroyed {} ({})", dev.name, dev.uid);
                report.destroyed_devices.push(dev.name.clone());
            }
            Err(e) => log::warn!("legacy cleanup: could not destroy {}: {e}", dev.uid),
        }
    }

    // Drop the input override if it pointed at the device we just removed.
    let stale_name = config.meeting.blackhole.input_device_name.clone();
    if !stale_name.is_empty() && config.meeting.input_device.as_deref() == Some(stale_name.as_str())
    {
        config.meeting.input_device = None;
        report.cleared_input_override = true;
    }

    if config.meeting.blackhole.configured || report.did_anything() {
        config.meeting.blackhole = Default::default();
        if let Err(e) = config.save() {
            log::warn!("legacy cleanup: could not save config: {e}");
        }
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quiet_report_summarises_to_nothing() {
        assert!(!RepairReport::default().did_anything());
        assert_eq!(RepairReport::default().summary(), None);
    }

    #[test]
    fn summary_mentions_each_action() {
        let r = RepairReport {
            restored_output: Some("MacBook Pro Speakers".into()),
            destroyed_devices: vec!["WhisprKing System Out".into()],
            cleared_input_override: true,
        };
        let s = r.summary().expect("summary");
        assert!(s.contains("MacBook Pro Speakers"), "{s}");
        assert!(s.contains("1 leftover"), "{s}");
        assert!(s.contains("microphone selection"), "{s}");
    }
}
