//! Microphone capture. 16 kHz mono f32 via cpal.
//!
//! Two operating modes share one buffer:
//!
//! 1. **One-shot** — `start()` then `stop()` returns everything captured.
//!    Used by hold-to-talk dictation.
//! 2. **Streaming** — `start()` then repeated `drain()` calls hand back
//!    whatever arrived since the previous call. Used by meetings, where
//!    [`super::capture::MeetingCapture`] owns the chunk timing so the
//!    microphone and the system-audio tap stay aligned.
//!
//! Format conversion (downmix + resample to 16 kHz) lives in
//! [`super::resample`] and is shared with the system-audio path.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, Stream, StreamConfig};
use thiserror::Error;

use super::resample::{self, Resampler};

pub use super::resample::TARGET_RATE as SAMPLE_RATE;

#[derive(Debug, Error)]
pub enum RecorderError {
    #[error("no input device named {0:?} — it may have been unplugged")]
    DeviceNotFound(String),
    #[error("no default input device")]
    NoInputDevice,
    // cpal 0.18 collapsed its per-operation error enums into one `cpal::Error`
    // carrying an `ErrorKind`, so device enumeration, config query, stream
    // build and stream play all funnel through this variant.
    #[error("audio device error: {0}")]
    Cpal(#[from] cpal::Error),
    #[error("unsupported sample format: {0:?}")]
    UnsupportedSampleFormat(SampleFormat),
}

struct Shared {
    /// Captured 16 kHz mono samples awaiting collection.
    buffer: Mutex<Vec<f32>>,
    /// Smoothed RMS in [0, 1] for the UI meter.
    level: Mutex<f32>,
    paused: AtomicBool,
}

/// Microphone recorder. The cpal `Stream` is `!Send` on macOS, so an
/// instance must stay on the thread that started it.
pub struct AudioRecorder {
    shared: Arc<Shared>,
    stream: Option<Stream>,
    recording: bool,
    /// cpal device name override. `None` → host default input.
    input_device_name: Option<String>,
}

impl AudioRecorder {
    pub fn new() -> Self {
        Self {
            shared: Arc::new(Shared {
                buffer: Mutex::new(Vec::with_capacity(SAMPLE_RATE as usize * 30)),
                level: Mutex::new(0.0),
                paused: AtomicBool::new(false),
            }),
            stream: None,
            recording: false,
            input_device_name: None,
        }
    }

    /// Pin the recorder to a specific cpal input device by name. `None`
    /// falls back to the host default.
    ///
    /// This is only ever set from the user's own choice in the UI. Nothing
    /// in the capture pipeline is allowed to overwrite it on the user's
    /// behalf.
    pub fn set_input_device(&mut self, name: Option<String>) {
        self.input_device_name = name;
    }

    pub fn input_device(&self) -> Option<&str> {
        self.input_device_name.as_deref()
    }

    pub fn is_recording(&self) -> bool {
        self.recording
    }

    pub fn is_paused(&self) -> bool {
        self.shared.paused.load(Ordering::SeqCst)
    }

    pub fn set_paused(&self, paused: bool) {
        self.shared.paused.store(paused, Ordering::SeqCst);
    }

    /// Smoothed signal level for UI meters, in [0, 1].
    pub fn level(&self) -> f32 {
        *self.shared.level.lock().expect("level")
    }

    /// Open the input stream and begin filling the buffer.
    pub fn start(&mut self) -> Result<(), RecorderError> {
        if self.recording {
            return Ok(());
        }
        self.clear();
        self.shared.paused.store(false, Ordering::SeqCst);
        self.open_stream()?;
        self.recording = true;
        Ok(())
    }

    /// Stop capture and return everything still buffered.
    pub fn stop(&mut self) -> Vec<f32> {
        if !self.recording {
            return Vec::new();
        }
        self.stream.take(); // dropping the Stream stops capture
        self.recording = false;
        self.drain()
    }

    /// Take everything captured since the previous call.
    pub fn drain(&self) -> Vec<f32> {
        self.sink().drain()
    }

    /// A `Send + Clone` handle onto this recorder's buffer.
    ///
    /// The cpal `Stream` itself is `!Send` and must stay on the thread that
    /// opened it, but the buffer behind it is just an `Arc<Mutex<…>>`. This
    /// hands that out so a dispatcher thread can collect audio without ever
    /// touching the stream.
    pub fn sink(&self) -> MicSink {
        MicSink(Arc::clone(&self.shared))
    }

    /// How many samples are waiting, without taking them.
    pub fn buffered(&self) -> usize {
        self.shared.buffer.lock().expect("audio buffer").len()
    }

    fn clear(&self) {
        self.shared.buffer.lock().expect("audio buffer").clear();
        *self.shared.level.lock().expect("level") = 0.0;
    }

    fn open_stream(&mut self) -> Result<(), RecorderError> {
        let host = cpal::default_host();
        let device = match self.input_device_name.as_deref() {
            Some(name) => host
                .input_devices()?
                .find(|d| {
                    d.description()
                        .map(|desc| desc.name() == name)
                        .unwrap_or(false)
                })
                .ok_or_else(|| RecorderError::DeviceNotFound(name.to_string()))?,
            None => host
                .default_input_device()
                .ok_or(RecorderError::NoInputDevice)?,
        };

        let device_label = device
            .description()
            .map(|d| d.name().to_string())
            .unwrap_or_else(|_| "<unnamed>".into());

        let supported = device.default_input_config()?;
        let sample_format = supported.sample_format();
        let in_rate = supported.sample_rate();
        let in_channels = supported.channels() as usize;
        let config: StreamConfig = supported.into();

        log::info!(
            "mic capture: {device_label} @ {in_rate} Hz, {in_channels} ch, {sample_format:?}"
        );

        let shared = Arc::clone(&self.shared);
        let stream = match sample_format {
            SampleFormat::F32 => {
                build_input::<f32>(&device, &config, in_rate, in_channels, shared)?
            }
            SampleFormat::I16 => {
                build_input::<i16>(&device, &config, in_rate, in_channels, shared)?
            }
            SampleFormat::U16 => {
                build_input::<u16>(&device, &config, in_rate, in_channels, shared)?
            }
            other => return Err(RecorderError::UnsupportedSampleFormat(other)),
        };

        stream.play()?;
        self.stream = Some(stream);
        Ok(())
    }
}

/// Thread-safe view onto a recorder's captured audio.
#[derive(Clone)]
pub struct MicSink(Arc<Shared>);

impl MicSink {
    /// Take everything captured since the previous call.
    pub fn drain(&self) -> Vec<f32> {
        let mut guard = self.0.buffer.lock().expect("audio buffer");
        std::mem::take(&mut *guard)
    }

    pub fn level(&self) -> f32 {
        *self.0.level.lock().expect("level")
    }

    pub fn set_paused(&self, paused: bool) {
        self.0.paused.store(paused, Ordering::SeqCst);
    }
}

impl Default for AudioRecorder {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for AudioRecorder {
    fn drop(&mut self) {
        self.stream.take();
    }
}

/// Sample types we can read from cpal — `i16`, `u16`, `f32`.
trait FromCpalSample: cpal::SizedSample + Send + 'static {
    fn to_f32(self) -> f32;
}

impl FromCpalSample for f32 {
    fn to_f32(self) -> f32 {
        self
    }
}
impl FromCpalSample for i16 {
    fn to_f32(self) -> f32 {
        (self as f32) / (i16::MAX as f32)
    }
}
impl FromCpalSample for u16 {
    fn to_f32(self) -> f32 {
        ((self as f32) - (u16::MAX as f32 / 2.0)) / (u16::MAX as f32 / 2.0)
    }
}

fn build_input<T>(
    device: &cpal::Device,
    config: &StreamConfig,
    in_rate: u32,
    in_channels: usize,
    shared: Arc<Shared>,
) -> Result<Stream, RecorderError>
where
    T: FromCpalSample + cpal::SizedSample,
{
    let mut resampler = Resampler::new(in_rate, in_channels);
    let mut widened = Vec::new();
    let mut converted = Vec::new();

    let err = |e| log::error!("audio stream error: {e:?}");

    let stream = device.build_input_stream(
        config.clone(),
        move |data: &[T], _: &_| {
            if shared.paused.load(Ordering::SeqCst) {
                return;
            }

            // Widen to f32 first — the resampler is generic over anything
            // that converts into f32, but cpal's sample types do not
            // implement `Into<f32>`. Still interleaved at this point.
            widened.clear();
            widened.extend(data.iter().copied().map(FromCpalSample::to_f32));

            converted.clear();
            resampler.push_interleaved(&widened, &mut converted);
            if converted.is_empty() {
                return;
            }

            let rms = resample::rms(&converted);
            if let Ok(mut level) = shared.level.lock() {
                *level = 0.75 * *level + 0.25 * (rms * 6.0).min(1.0);
            }
            if let Ok(mut buf) = shared.buffer.lock() {
                buf.extend_from_slice(&converted);
            }
        },
        err,
        None,
    )?;

    Ok(stream)
}
