//! Microphone recorder. 16 kHz mono f32 via cpal.
//!
//! Two operating modes:
//!
//! 1. **One-shot** — `start()` then `stop()` returns the full captured buffer.
//!    Used by hold-to-talk dictation.
//! 2. **Continuous** — `start_continuous(on_chunk, …)` invokes `on_chunk`
//!    on a worker thread every N seconds with one chunk; `pause` /
//!    `resume` / `stop_continuous` control the stream. Used by meetings.
//!
//! Input format from cpal varies per device. We resample / downmix /
//! convert to mono f32 at 16 kHz inside the audio callback.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, Stream, StreamConfig};
use thiserror::Error;

pub const SAMPLE_RATE: u32 = 16_000;

#[derive(Debug, Error)]
pub enum RecorderError {
    #[error("no default input device")]
    NoInputDevice,
    #[error("device error: {0}")]
    Device(#[from] cpal::DevicesError),
    #[error("default config error: {0}")]
    Config(#[from] cpal::DefaultStreamConfigError),
    #[error("build stream error: {0}")]
    BuildStream(#[from] cpal::BuildStreamError),
    #[error("play stream error: {0}")]
    PlayStream(#[from] cpal::PlayStreamError),
    #[error("unsupported sample format: {0:?}")]
    UnsupportedSampleFormat(SampleFormat),
}

/// Smoothed RMS level in [0, 1]. Updated on every callback; safe to read
/// from any thread.
#[derive(Clone, Default)]
struct Level(Arc<AtomicU32>);

impl Level {
    fn store(&self, value: f32) {
        self.0.store(value.to_bits(), Ordering::Relaxed);
    }
    fn load(&self) -> f32 {
        f32::from_bits(self.0.load(Ordering::Relaxed))
    }
}

type ChunkCallback = Arc<dyn Fn(Vec<f32>) + Send + Sync + 'static>;

struct Inner {
    /// Buffer of captured 16 kHz mono samples, drained on chunk dispatch
    /// or by `stop()`.
    buffer: Vec<f32>,
    /// Saved continuous-mode samples when `save_full=true`.
    full: Vec<f32>,
}

/// Audio recorder state shared between the cpal callback and the
/// dispatcher / control threads.
pub struct AudioRecorder {
    inner: Arc<Mutex<Inner>>,
    level: Level,
    is_recording: Arc<AtomicBool>,
    paused: Arc<AtomicBool>,
    continuous: Arc<AtomicBool>,
    save_full: Arc<AtomicBool>,

    stream: Option<Stream>,
    dispatcher: Option<DispatcherHandle>,
    chunk_samples: usize,

    /// cpal device name override. `None` → host default input.
    input_device_name: Option<String>,
}

struct DispatcherHandle {
    stop: Sender<()>,
    thread: JoinHandle<()>,
}

impl AudioRecorder {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner {
                buffer: Vec::with_capacity(SAMPLE_RATE as usize * 30),
                full: Vec::new(),
            })),
            level: Level::default(),
            is_recording: Arc::new(AtomicBool::new(false)),
            paused: Arc::new(AtomicBool::new(false)),
            continuous: Arc::new(AtomicBool::new(false)),
            save_full: Arc::new(AtomicBool::new(false)),
            stream: None,
            dispatcher: None,
            chunk_samples: 0,
            input_device_name: None,
        }
    }

    /// Pin the recorder to a specific cpal input device by name. Pass
    /// `None` to fall back to the host default.
    pub fn set_input_device(&mut self, name: Option<String>) {
        self.input_device_name = name;
    }

    pub fn is_recording(&self) -> bool {
        self.is_recording.load(Ordering::SeqCst)
    }

    pub fn is_paused(&self) -> bool {
        self.paused.load(Ordering::SeqCst)
    }

    /// Smoothed signal level for UI meters, in [0, 1].
    pub fn level(&self) -> f32 {
        self.level.load()
    }

    // --- one-shot mode --------------------------------------------------

    pub fn start(&mut self) -> Result<(), RecorderError> {
        if self.is_recording() {
            return Ok(());
        }
        self.clear_buffers();
        self.open_stream()?;
        self.is_recording.store(true, Ordering::SeqCst);
        Ok(())
    }

    pub fn stop(&mut self) -> Vec<f32> {
        if !self.is_recording() {
            return Vec::new();
        }
        self.close_stream();
        self.is_recording.store(false, Ordering::SeqCst);
        let mut inner = self.inner.lock().expect("audio buffer");
        std::mem::take(&mut inner.buffer)
    }

    // --- continuous mode ------------------------------------------------

    pub fn start_continuous<F>(
        &mut self,
        on_chunk: F,
        chunk_seconds: u32,
        save_full: bool,
    ) -> Result<(), RecorderError>
    where
        F: Fn(Vec<f32>) + Send + Sync + 'static,
    {
        if self.is_recording() {
            return Ok(());
        }

        self.clear_buffers();
        self.continuous.store(true, Ordering::SeqCst);
        self.paused.store(false, Ordering::SeqCst);
        self.save_full.store(save_full, Ordering::SeqCst);
        self.chunk_samples = chunk_seconds as usize * SAMPLE_RATE as usize;

        let cb: ChunkCallback = Arc::new(on_chunk);
        self.dispatcher = Some(self.spawn_dispatcher(cb));
        self.open_stream()?;
        self.is_recording.store(true, Ordering::SeqCst);
        Ok(())
    }

    pub fn pause_continuous(&self) {
        self.paused.store(true, Ordering::SeqCst);
    }

    pub fn resume_continuous(&self) {
        self.paused.store(false, Ordering::SeqCst);
    }

    /// Stop continuous capture. Returns the full waveform iff `save_full`
    /// was set on `start_continuous`. The dispatcher flushes any trailing
    /// audio as one final chunk before returning.
    pub fn stop_continuous(&mut self) -> Option<Vec<f32>> {
        if !(self.is_recording() && self.continuous.load(Ordering::SeqCst)) {
            return None;
        }
        self.close_stream();
        self.is_recording.store(false, Ordering::SeqCst);
        self.stop_dispatcher_flushing();

        let save_full = self.save_full.swap(false, Ordering::SeqCst);
        self.continuous.store(false, Ordering::SeqCst);

        let mut inner = self.inner.lock().expect("audio buffer");

        if save_full {
            Some(std::mem::take(&mut inner.full))
        } else {
            inner.full.clear();
            None
        }
    }

    // --- internals ------------------------------------------------------

    fn clear_buffers(&self) {
        let mut inner = self.inner.lock().expect("audio buffer");
        inner.buffer.clear();
        inner.full.clear();
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
                .ok_or(RecorderError::NoInputDevice)?,
            None => host
                .default_input_device()
                .ok_or(RecorderError::NoInputDevice)?,
        };
        let supported = device.default_input_config()?;
        let sample_format = supported.sample_format();
        let in_rate = supported.sample_rate();
        let in_channels = supported.channels() as usize;
        let config: StreamConfig = supported.into();

        let inner = Arc::clone(&self.inner);
        let level = self.level.clone();
        let paused = Arc::clone(&self.paused);
        let continuous = Arc::clone(&self.continuous);
        let save_full = Arc::clone(&self.save_full);

        let stream = match sample_format {
            SampleFormat::F32 => build_input::<f32>(
                &device,
                &config,
                in_rate,
                in_channels,
                inner,
                level,
                paused,
                continuous,
                save_full,
            )?,
            SampleFormat::I16 => build_input::<i16>(
                &device,
                &config,
                in_rate,
                in_channels,
                inner,
                level,
                paused,
                continuous,
                save_full,
            )?,
            SampleFormat::U16 => build_input::<u16>(
                &device,
                &config,
                in_rate,
                in_channels,
                inner,
                level,
                paused,
                continuous,
                save_full,
            )?,
            other => return Err(RecorderError::UnsupportedSampleFormat(other)),
        };

        stream.play()?;
        self.stream = Some(stream);
        Ok(())
    }

    fn close_stream(&mut self) {
        // Dropping the cpal Stream stops capture cleanly.
        self.stream.take();
    }

    fn spawn_dispatcher(&self, on_chunk: ChunkCallback) -> DispatcherHandle {
        let (tx, rx): (Sender<()>, Receiver<()>) = mpsc::channel();
        let inner = Arc::clone(&self.inner);
        let chunk_samples = self.chunk_samples;

        let thread = thread::Builder::new()
            .name("whisprking-audio-dispatch".into())
            .spawn(move || dispatch_loop(inner, on_chunk, chunk_samples, rx))
            .expect("spawn audio dispatcher");

        DispatcherHandle { stop: tx, thread }
    }

    fn stop_dispatcher_flushing(&mut self) {
        if let Some(handle) = self.dispatcher.take() {
            let _ = handle.stop.send(());
            // Best-effort join — dispatcher exits on next wake.
            let _ = handle.thread.join();
        }
    }
}

impl Default for AudioRecorder {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for AudioRecorder {
    fn drop(&mut self) {
        self.close_stream();
        self.stop_dispatcher_flushing();
    }
}

/// Drain at most `chunk_samples` from the shared buffer; on `force=true`
/// drain everything.
fn drain_chunk(inner: &Mutex<Inner>, chunk_samples: usize, force: bool) -> Option<Vec<f32>> {
    let mut guard = inner.lock().expect("audio buffer");
    if guard.buffer.is_empty() {
        return None;
    }
    if !force && guard.buffer.len() < chunk_samples {
        return None;
    }
    Some(std::mem::take(&mut guard.buffer))
}

fn dispatch_loop(
    inner: Arc<Mutex<Inner>>,
    on_chunk: ChunkCallback,
    chunk_samples: usize,
    stop: Receiver<()>,
) {
    let tick = Duration::from_millis(200);
    loop {
        match stop.recv_timeout(tick) {
            Ok(()) => break,
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
            Err(mpsc::RecvTimeoutError::Timeout) => {}
        }
        if let Some(chunk) = drain_chunk(&inner, chunk_samples, false) {
            on_chunk(chunk);
        }
    }
    // Flush a final chunk on shutdown so trailing audio is not lost.
    if let Some(chunk) = drain_chunk(&inner, chunk_samples, true) {
        on_chunk(chunk);
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

#[allow(clippy::too_many_arguments)]
fn build_input<T>(
    device: &cpal::Device,
    config: &StreamConfig,
    in_rate: u32,
    in_channels: usize,
    inner: Arc<Mutex<Inner>>,
    level: Level,
    paused: Arc<AtomicBool>,
    continuous: Arc<AtomicBool>,
    save_full: Arc<AtomicBool>,
) -> Result<Stream, RecorderError>
where
    T: FromCpalSample + cpal::SizedSample,
{
    // Simple linear-step resampler. cpal hands us blocks at the device
    // rate; we emit one f32 mono sample every `step` source frames.
    let step = (in_rate as f64) / (SAMPLE_RATE as f64);
    let mut phase = 0.0_f64;
    let mut frame_idx = 0_u64;

    let err = |e| log::error!("audio stream error: {e:?}");

    let stream = device.build_input_stream(
        config,
        move |data: &[T], _: &_| {
            if continuous.load(Ordering::SeqCst) && paused.load(Ordering::SeqCst) {
                return;
            }

            let frames = data.len() / in_channels;
            let mut mono = Vec::with_capacity((frames as f64 / step) as usize + 1);

            for f in 0..frames {
                let mut sum = 0.0f32;
                for c in 0..in_channels {
                    sum += data[f * in_channels + c].to_f32();
                }
                let mono_sample = sum / in_channels as f32;

                // Emit once per `step` source frames.
                let position = frame_idx as f64 + f as f64 - phase;
                if position >= 0.0 {
                    mono.push(mono_sample);
                    phase += step;
                }
            }
            frame_idx += frames as u64;
            phase = phase.max(frame_idx as f64);

            if mono.is_empty() {
                return;
            }

            let rms = rms(&mono);
            let smoothed = 0.75 * level.load() + 0.25 * rms.mul_add(6.0, 0.0).min(1.0);
            level.store(smoothed);

            let mut guard = inner.lock().expect("audio buffer");
            guard.buffer.extend_from_slice(&mono);
            if save_full.load(Ordering::SeqCst) {
                guard.full.extend_from_slice(&mono);
            }
        },
        err,
        None,
    )?;

    Ok(stream)
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    (sum_sq / samples.len() as f32 + 1e-12).sqrt()
}
