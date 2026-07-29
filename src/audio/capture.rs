//! Meeting capture: microphone and system audio as two independent tracks.
//!
//! The previous design fused the microphone and system audio into a single
//! Core Audio *aggregate device* and asked the user to select it as their
//! microphone in Zoom. That is what broke outgoing audio: an aggregate
//! built from `[BlackHole, Mic]` puts system audio on channels 1–2 and the
//! microphone on channels 3+, so a meeting app reading channel 1 sends the
//! far end *its own audio* played back instead of the user's voice.
//!
//! Nothing here is ever exposed to another application. The microphone is
//! opened read-only through cpal exactly as the user configured it, system
//! audio comes from a private process tap, and the two are kept apart:
//!
//! - The user's meeting app keeps using their real microphone, untouched.
//! - The default output device is never changed.
//! - No device we create is visible outside this process.
//!
//! Keeping the tracks separate all the way through transcription also
//! removes the worst input an acoustic model can get — two people talking
//! over each other in one mixed channel — and gives the transcript speaker
//! attribution for free.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use super::recorder::{AudioRecorder, MicSink, RecorderError, SAMPLE_RATE};
use super::resample;

#[cfg(target_os = "macos")]
use super::tap::{self, SystemAudioTap, TapError, TapSink};

/// On every other platform there is no system-audio path, but the rest of
/// this module stays `cfg`-free by talking to a sink that yields nothing.
#[cfg(not(target_os = "macos"))]
#[derive(Clone)]
pub struct TapSink;

#[cfg(not(target_os = "macos"))]
impl TapSink {
	pub fn drain(&self) -> Vec<f32> {
		Vec::new()
	}
	pub fn level(&self) -> f32 {
		0.0
	}
}

/// Which side of the conversation a chunk came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Track {
	/// The user's microphone.
	Mic,
	/// Everything the machine was playing — i.e. the other participants.
	System,
}

impl Track {
	/// Label used in the transcript and in the live view. Transcripts written
	/// by older builds say `You` / `Others`; readers accept both.
	pub fn label(self) -> &'static str {
		match self {
			Track::Mic => "Du",
			Track::System => "Andere",
		}
	}
}

/// What the user asked us to capture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceMode {
	MicOnly,
	SystemOnly,
	/// Both, transcribed as separate speaker-attributed tracks.
	Both,
}

impl SourceMode {
	pub fn from_config(s: &str) -> Self {
		match s {
			"mic" => SourceMode::MicOnly,
			"system" => SourceMode::SystemOnly,
			// "mix" is the legacy value from the BlackHole-era config.
			_ => SourceMode::Both,
		}
	}

	pub fn as_config(self) -> &'static str {
		match self {
			SourceMode::MicOnly => "mic",
			SourceMode::SystemOnly => "system",
			SourceMode::Both => "both",
		}
	}

	fn wants_mic(self) -> bool {
		matches!(self, SourceMode::MicOnly | SourceMode::Both)
	}

	fn wants_system(self) -> bool {
		matches!(self, SourceMode::SystemOnly | SourceMode::Both)
	}
}

/// One raw frame of captured audio for a single track.
///
/// Capture stays dumb: it forwards frames as they arrive and does no
/// silence-gating or window-cutting. Turning this continuous per-track
/// stream into transcription-sized [`SpeechSpan`](super::vad::SpeechSpan)s —
/// on speech boundaries via VAD, or fixed windows as a fallback — happens
/// downstream in the meeting worker's [`Segmenter`](super::vad::Segmenter).
pub struct Chunk {
	pub track: Track,
	/// Seconds since capture started, at the *start* of this frame.
	pub elapsed: f64,
	pub audio: Vec<f32>,
}

#[derive(Debug, thiserror::Error)]
pub enum CaptureError {
	#[error("microphone: {0}")]
	Mic(#[from] RecorderError),
	#[cfg(target_os = "macos")]
	#[error("system audio: {0}")]
	System(#[from] TapError),
	#[error("no capture source is available — enable the microphone or system audio")]
	NothingToCapture,
}

/// Outcome of [`MeetingCapture::start`]. Capture can succeed with one
/// track missing; the UI surfaces `warning` rather than failing outright,
/// so losing system audio does not cost the user their microphone too.
pub struct StartReport {
	pub mic: bool,
	pub system: bool,
	pub warning: Option<String>,
}

struct DispatcherHandle {
	stop: Sender<()>,
	thread: JoinHandle<()>,
	full: Arc<Mutex<Vec<f32>>>,
	paused: Arc<AtomicBool>,
}

/// Owns the microphone stream and the system-audio tap for one meeting.
///
/// Must stay on the thread that called [`start`](Self::start) — cpal's
/// `Stream` is `!Send` on macOS.
pub struct MeetingCapture {
	recorder: AudioRecorder,
	#[cfg(target_os = "macos")]
	tap: Option<SystemAudioTap>,
	dispatcher: Option<DispatcherHandle>,
	active: Option<StartReport>,
}

impl MeetingCapture {
	pub fn new() -> Self {
		Self {
			recorder: AudioRecorder::new(),
			#[cfg(target_os = "macos")]
			tap: None,
			dispatcher: None,
			active: None,
		}
	}

	/// Is system-audio capture possible on this machine?
	pub fn system_audio_supported() -> bool {
		#[cfg(target_os = "macos")]
		{
			tap::is_supported()
		}
		#[cfg(not(target_os = "macos"))]
		{
			false
		}
	}

	pub fn mic_level(&self) -> f32 {
		self.recorder.level()
	}

	pub fn system_level(&self) -> f32 {
		#[cfg(target_os = "macos")]
		{
			self.tap.as_ref().map(|t| t.level()).unwrap_or(0.0)
		}
		#[cfg(not(target_os = "macos"))]
		{
			0.0
		}
	}

	pub fn is_running(&self) -> bool {
		self.dispatcher.is_some()
	}

	pub fn report(&self) -> Option<&StartReport> {
		self.active.as_ref()
	}

	/// Begin capture. `on_chunk` is invoked from a dispatcher thread.
	///
	/// `input_device` is the user's own choice and is used verbatim; this
	/// function never rewrites it.
	pub fn start<F>(
		&mut self,
		mode: SourceMode,
		input_device: Option<String>,
		save_audio: bool,
		on_chunk: F,
	) -> Result<(), CaptureError>
	where
		F: Fn(Chunk) + Send + 'static,
	{
		if self.is_running() {
			return Ok(());
		}

		let mut warning: Option<String> = None;

		// --- microphone ------------------------------------------------
		let mic_sink = if mode.wants_mic() {
			self.recorder.set_input_device(input_device);
			match self.recorder.start() {
				Ok(()) => Some(self.recorder.sink()),
				Err(e) if mode == SourceMode::Both => {
					// Losing the mic should not also cost system audio.
					log::error!("meeting: microphone unavailable: {e}");
					warning = Some(format!("Microphone unavailable: {e}"));
					None
				},
				Err(e) => return Err(e.into()),
			}
		} else {
			None
		};

		// --- system audio ----------------------------------------------
		#[cfg(target_os = "macos")]
		let system_sink = if mode.wants_system() {
			match SystemAudioTap::start() {
				Ok(t) => {
					tap::warn_if_unexpected_format(&t);
					log::info!(
						"meeting: system audio tap running at {} Hz",
						t.source_rate()
					);
					let sink = t.sink();
					self.tap = Some(t);
					Some(sink)
				},
				Err(e) if mic_sink.is_some() => {
					log::error!("meeting: system audio unavailable: {e}");
					warning = Some(e.to_string());
					None
				},
				Err(e) => {
					// Nothing else is running — undo the mic and report.
					self.recorder.stop();
					return Err(e.into());
				},
			}
		} else {
			None
		};
		#[cfg(not(target_os = "macos"))]
		let system_sink: Option<TapSink> = {
			if mode.wants_system() {
				warning = Some("System audio capture is only available on macOS.".into());
			}
			None
		};

		if mic_sink.is_none() && system_sink.is_none() {
			return Err(CaptureError::NothingToCapture);
		}

		let report = StartReport {
			mic: mic_sink.is_some(),
			system: system_sink.is_some(),
			warning,
		};

		// --- dispatcher -------------------------------------------------
		let (stop_tx, stop_rx) = mpsc::channel();
		let full = Arc::new(Mutex::new(Vec::new()));
		let paused = Arc::new(AtomicBool::new(false));

		let thread = {
			let full = Arc::clone(&full);
			let paused = Arc::clone(&paused);
			thread::Builder::new()
				.name("whisprking-capture".into())
				.spawn(move || {
					dispatch_loop(
						mic_sink,
						system_sink,
						save_audio,
						full,
						paused,
						stop_rx,
						on_chunk,
					)
				})
				.expect("spawn capture dispatcher")
		};

		self.dispatcher = Some(DispatcherHandle {
			stop: stop_tx,
			thread,
			full,
			paused,
		});
		self.active = Some(report);
		Ok(())
	}

	pub fn set_paused(
		&self,
		paused: bool,
	) {
		if let Some(d) = &self.dispatcher {
			d.paused.store(paused, Ordering::SeqCst);
		}
		self.recorder.set_paused(paused);
	}

	/// Stop capture, flush trailing audio, and return the full mixed
	/// recording if `save_audio` was set.
	pub fn stop(&mut self) -> Option<Vec<f32>> {
		// Stop the sources first so the dispatcher's final drain sees
		// everything and nothing arrives after it exits.
		self.recorder.stop();
		#[cfg(target_os = "macos")]
		{
			self.tap.take();
		}
		self.active = None;

		let handle = self.dispatcher.take()?;
		let _ = handle.stop.send(());
		let _ = handle.thread.join();
		let mut guard = handle.full.lock().expect("full recording");
		let audio = std::mem::take(&mut *guard);
		(!audio.is_empty()).then_some(audio)
	}
}

impl Default for MeetingCapture {
	fn default() -> Self {
		Self::new()
	}
}

impl Drop for MeetingCapture {
	fn drop(&mut self) {
		self.stop();
	}
}

/// Per-track frame clock. Turns each drained buffer into a [`Chunk`] whose
/// `elapsed` is derived from the running sample count, not wall-clock, so
/// pauses and scheduling jitter cannot desynchronise transcript timestamps.
struct TrackClock {
	track: Track,
	emitted: usize,
}

impl TrackClock {
	fn new(track: Track) -> Self {
		Self { track, emitted: 0 }
	}

	/// Wrap `audio` (this tick's drained samples) as a frame, or `None` if
	/// there was nothing to send.
	fn frame(
		&mut self,
		audio: Vec<f32>,
	) -> Option<Chunk> {
		if audio.is_empty() {
			return None;
		}
		let elapsed = self.emitted as f64 / SAMPLE_RATE as f64;
		self.emitted += audio.len();
		Some(Chunk {
			track: self.track,
			elapsed,
			audio,
		})
	}
}

fn dispatch_loop<F>(
	mic: Option<MicSink>,
	system: Option<TapSink>,
	save_audio: bool,
	full: Arc<Mutex<Vec<f32>>>,
	paused: Arc<AtomicBool>,
	stop: Receiver<()>,
	on_chunk: F,
) where
	F: Fn(Chunk),
{
	let tick = Duration::from_millis(200);
	let mut mic_clock = TrackClock::new(Track::Mic);
	let mut sys_clock = TrackClock::new(Track::System);
	let started = Instant::now();

	// Drain both sources once and forward a frame per track. Silence is *not*
	// dropped here: the downstream VAD needs to see the pauses to find speech
	// boundaries, and the fixed-window fallback does its own silence-gating.
	let pump = |mic_clock: &mut TrackClock, sys_clock: &mut TrackClock| {
		let is_paused = paused.load(Ordering::SeqCst);

		let from_mic = mic.as_ref().map(|m| m.drain()).unwrap_or_default();
		let from_sys = system.as_ref().map(|s| s.drain()).unwrap_or_default();

		// While paused we still drain, so resuming does not dump a backlog
		// of stale audio into the transcript — we just throw it away.
		if is_paused {
			return;
		}

		if save_audio && !(from_mic.is_empty() && from_sys.is_empty()) {
			if let Ok(mut g) = full.lock() {
				g.extend_from_slice(&resample::mix(&from_mic, &from_sys));
			}
		}

		if let Some(c) = mic_clock.frame(from_mic) {
			on_chunk(c);
		}
		if let Some(c) = sys_clock.frame(from_sys) {
			on_chunk(c);
		}
	};

	loop {
		match stop.recv_timeout(tick) {
			Ok(()) => break,
			Err(mpsc::RecvTimeoutError::Disconnected) => break,
			Err(mpsc::RecvTimeoutError::Timeout) => {},
		}
		pump(&mut mic_clock, &mut sys_clock);
	}

	// Final drain so trailing speech is not lost.
	pump(&mut mic_clock, &mut sys_clock);
	log::info!(
		"capture: dispatcher finished after {:.1}s",
		started.elapsed().as_secs_f64()
	);
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn source_mode_round_trips_through_config() {
		for mode in [
			SourceMode::MicOnly,
			SourceMode::SystemOnly,
			SourceMode::Both,
		] {
			assert_eq!(SourceMode::from_config(mode.as_config()), mode);
		}
	}

	/// The BlackHole-era config wrote `"mix"`. It must land somewhere sane
	/// rather than silently disabling a track.
	#[test]
	fn legacy_mix_maps_to_both() {
		assert_eq!(SourceMode::from_config("mix"), SourceMode::Both);
	}

	/// A frame carries exactly what was drained, and an empty drain produces
	/// nothing (segmentation into transcription units is the worker's job).
	#[test]
	fn frame_wraps_drained_audio_and_skips_empty() {
		let mut clock = TrackClock::new(Track::Mic);
		assert!(clock.frame(Vec::new()).is_none());
		let f = clock.frame(vec![0.5; 1_000]).expect("frame");
		assert_eq!(f.audio.len(), 1_000);
		assert_eq!(f.track, Track::Mic);
		assert_eq!(f.elapsed, 0.0);
	}

	/// Frame start times come from the running sample count, so they stay
	/// correct no matter how the dispatcher was scheduled.
	#[test]
	fn elapsed_tracks_sample_count_not_wall_clock() {
		let mut clock = TrackClock::new(Track::System);
		let sec = SAMPLE_RATE as usize; // one second
		let times: Vec<f64> = (0..3)
			.map(|_| clock.frame(vec![0.1; sec]).unwrap().elapsed)
			.collect();
		assert_eq!(times, vec![0.0, 1.0, 2.0]);
	}

	#[test]
	fn track_labels_are_distinct() {
		assert_ne!(Track::Mic.label(), Track::System.label());
	}
}
