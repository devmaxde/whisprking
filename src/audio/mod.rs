//! Audio capture and host-device discovery.

pub mod capture;
#[cfg(target_os = "macos")]
pub mod coreaudio;
pub mod decode;
pub mod import;
#[cfg(target_os = "macos")]
pub mod legacy;
pub mod recorder;
pub mod resample;
pub mod system_audio;
#[cfg(target_os = "macos")]
pub mod tap;
pub mod track_writer;
pub mod vad;

pub use capture::{Chunk, MeetingCapture, SourceMode, Track};
pub use recorder::{AudioRecorder, MicSink, RecorderError, SAMPLE_RATE};
pub use resample::{Resampler, TARGET_RATE};
pub use track_writer::{read_track, track_path, TrackWriter};
pub use vad::{Segmenter, SpeechSpan};
