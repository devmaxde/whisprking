//! macOS permission probes and Settings deep-links.
//!
//! On other platforms every check returns `true` / `Unknown` so the rest
//! of the app stays buildable; the permission UI is only meaningful on
//! macOS anyway.

use std::process::Command;

/// Tri-state result of a permission probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionStatus {
    Granted,
    Denied,
    /// macOS has not asked the user yet — the next privileged call will
    /// trigger the system prompt.
    NotDetermined,
    /// We cannot determine the status (e.g. running on a non-macOS host).
    Unknown,
}

/// Does the process currently hold the macOS Accessibility entitlement?
/// Required for Cmd+V synthesis in [`crate::output::smart_paste`] and for
/// rdev to install its global keyboard tap.
pub fn check_accessibility() -> bool {
    #[cfg(target_os = "macos")]
    {
        // Tiny FFI surface — avoids a separate `accessibility-sys` crate.
        #[link(name = "ApplicationServices", kind = "framework")]
        extern "C" {
            fn AXIsProcessTrusted() -> bool;
        }
        // SAFETY: zero-arg C function with stable Apple ABI.
        unsafe { AXIsProcessTrusted() }
    }
    #[cfg(not(target_os = "macos"))]
    {
        true
    }
}

/// Same as [`check_accessibility`] but pops the macOS prompt if the
/// process is not yet trusted. Calling this also makes the binary show
/// up under *System Settings → Privacy & Security → Accessibility* —
/// macOS only lists an app once it has actually asked for the permission.
///
/// Returns the trust state observed at call time. The prompt is async:
/// the user will still need to flip the switch and restart the app.
pub fn prompt_for_accessibility() -> bool {
    #[cfg(target_os = "macos")]
    {
        use core_foundation::base::{CFType, TCFType};
        use core_foundation::boolean::CFBoolean;
        use core_foundation::dictionary::CFDictionary;
        use core_foundation::string::CFString;

        #[link(name = "ApplicationServices", kind = "framework")]
        extern "C" {
            fn AXIsProcessTrustedWithOptions(
                options: core_foundation::dictionary::CFDictionaryRef,
            ) -> bool;
        }

        // Key string is `AXTrustedCheckOptionPrompt`. We construct it
        // manually instead of pulling in `accessibility-sys` for one
        // constant.
        let key = CFString::new("AXTrustedCheckOptionPrompt");
        let value = CFBoolean::true_value();
        let opts =
            CFDictionary::from_CFType_pairs(&[(key.as_CFType(), value.as_CFType() as CFType)]);
        // SAFETY: dictionary lives for the duration of the call.
        let trusted = unsafe { AXIsProcessTrustedWithOptions(opts.as_concrete_TypeRef()) };
        log::info!("AXIsProcessTrustedWithOptions(prompt=true) → {}", trusted);
        trusted
    }
    #[cfg(not(target_os = "macos"))]
    {
        true
    }
}

/// Probe microphone authorization without prompting. On non-macOS we
/// return `Unknown` and let the OS handle it at first use.
pub fn check_microphone() -> PermissionStatus {
    #[cfg(target_os = "macos")]
    {
        macos::check_microphone()
    }
    #[cfg(not(target_os = "macos"))]
    {
        PermissionStatus::Unknown
    }
}

pub fn open_accessibility_settings() {
    let _ = Command::new("open")
        .arg("x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility")
        .spawn();
}

pub fn open_microphone_settings() {
    let _ = Command::new("open")
        .arg("x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone")
        .spawn();
}

#[cfg(target_os = "macos")]
mod macos {
    use super::PermissionStatus;

    // AVAuthorizationStatus enum values from AVFoundation.
    const STATUS_NOT_DETERMINED: i64 = 0;
    const STATUS_RESTRICTED: i64 = 1;
    const STATUS_DENIED: i64 = 2;
    const STATUS_AUTHORIZED: i64 = 3;

    /// We pipe through `swift` if available, falling back to `Unknown` if
    /// we cannot determine the status. A direct ObjC call here would
    /// require pulling in `objc2` + `objc2-av-foundation`, which is a lot
    /// of build surface for a single boolean check. We use the bundled
    /// `tccutil` is intentionally not invoked — it requires root.
    pub fn check_microphone() -> PermissionStatus {
        // First try a no-prompt query via osascript. Apple removed many
        // shell hooks for TCC; we use the AppleScript bridge which
        // surfaces the AVCaptureDevice authorization status without
        // requesting access.
        let script = r#"
            use framework "AVFoundation"
            return (current application's AVCaptureDevice's authorizationStatusForMediaType:"soun") as integer
        "#;
        let output = match std::process::Command::new("osascript")
            .arg("-e")
            .arg(script)
            .output()
        {
            Ok(o) if o.status.success() => o,
            _ => return PermissionStatus::Unknown,
        };
        let raw = String::from_utf8_lossy(&output.stdout);
        let n: i64 = match raw.trim().parse() {
            Ok(v) => v,
            Err(_) => return PermissionStatus::Unknown,
        };
        match n {
            STATUS_AUTHORIZED => PermissionStatus::Granted,
            STATUS_DENIED | STATUS_RESTRICTED => PermissionStatus::Denied,
            STATUS_NOT_DETERMINED => PermissionStatus::NotDetermined,
            _ => PermissionStatus::Unknown,
        }
    }
}
