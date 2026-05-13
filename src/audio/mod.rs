//! Audio capture and host-device discovery.

#[cfg(target_os = "macos")]
pub mod coreaudio;
pub mod recorder;
pub mod system_audio;

pub use recorder::{AudioRecorder, RecorderError, SAMPLE_RATE};
pub use system_audio::{detect_blackhole, BLACKHOLE_INSTALL_HINT, BLACKHOLE_INSTALL_URL};
