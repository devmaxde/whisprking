//! "Meeting" tab — start/stop chunked recording, show live transcript.
//!
//! Transcription runs on a worker thread. Audio chunks arrive over an
//! `mpsc` channel; we send them to the worker and receive `(elapsed, text)`
//! segments back, which are appended to the in-memory transcript view and
//! flushed to disk via [`TranscriptWriter`].

use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Instant;

use cpal::traits::{DeviceTrait, HostTrait};

use crate::audio::recorder::AudioRecorder;
use crate::audio::system_audio::BLACKHOLE_INSTALL_HINT;
use crate::config::{BlackHoleSetup, Config};
use crate::output::transcript_writer::TranscriptWriter;
use crate::transcription::engine::Transcriber;

#[cfg(target_os = "macos")]
use crate::audio::coreaudio;

use super::super::styles;

const BH_OUTPUT_NAME: &str = "WhisprKing System Out";
const BH_OUTPUT_UID: &str = "com.devmaxde.whisprking.output";
const BH_INPUT_NAME: &str = "WhisprKing Meeting Input";
const BH_INPUT_UID: &str = "com.devmaxde.whisprking.input";

/// One transcribed line with the wall-clock offset since start.
#[derive(Debug, Clone)]
struct Segment {
    elapsed: f64,
    text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeetingState {
    Idle,
    Recording,
    Paused,
}

pub struct MeetingPage {
    state: MeetingState,
    recorder: AudioRecorder,
    writer: TranscriptWriter,
    started_at: Option<Instant>,
    paused_for: f64,
    pause_started: Option<Instant>,

    /// cpal input device names, refreshed on demand.
    input_devices: Vec<String>,

    /// CoreAudio device list for the BlackHole-setup pickers. macOS-only.
    #[cfg(target_os = "macos")]
    ca_devices: Vec<coreaudio::DeviceInfo>,
    #[cfg(target_os = "macos")]
    bh_status: BhStatus,

    /// Engine guarded by a Mutex so we can hand it to the worker without
    /// cloning. The worker holds the lock only while transcribing.
    engine: Arc<Mutex<Box<dyn Transcriber>>>,

    /// Channel from the recorder dispatcher → worker.
    chunk_tx: Option<Sender<ChunkJob>>,
    /// Channel from worker → UI thread for segment delivery.
    segment_rx: Receiver<Segment>,
    segment_tx: Sender<Segment>,
    worker: Option<JoinHandle<()>>,

    transcript: Vec<Segment>,
}

#[cfg(target_os = "macos")]
#[derive(Debug, Clone, Default)]
struct BhStatus {
    /// `Ok` text (e.g. success), `Err` text (last error). `None` while idle.
    message: Option<Result<String, String>>,
}

struct ChunkJob {
    elapsed: f64,
    audio: Vec<f32>,
}

impl MeetingPage {
    pub fn new(config: &Config, engine: Box<dyn Transcriber>) -> Result<Self, anyhow::Error> {
        let writer = TranscriptWriter::new(Config::transcripts_dir(config))?;
        let (segment_tx, segment_rx) = mpsc::channel();
        Ok(Self {
            state: MeetingState::Idle,
            recorder: AudioRecorder::new(),
            writer,
            started_at: None,
            paused_for: 0.0,
            pause_started: None,
            input_devices: list_input_devices(),
            #[cfg(target_os = "macos")]
            ca_devices: coreaudio::list_devices().unwrap_or_default(),
            #[cfg(target_os = "macos")]
            bh_status: BhStatus::default(),
            engine: Arc::new(Mutex::new(engine)),
            chunk_tx: None,
            segment_rx,
            segment_tx,
            worker: None,
            transcript: Vec::new(),
        })
    }

    pub fn state(&self) -> MeetingState {
        self.state
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, config: &mut Config) {
        self.drain_worker();

        ui.vertical(|ui| {
            ui.add_space(8.0);
            ui.heading("Meeting");
            ui.add_space(8.0);

            self.transport_row(ui, config);
            ui.add_space(8.0);
            self.input_row(ui, config);
            ui.add_space(6.0);

            #[cfg(target_os = "macos")]
            self.blackhole_row(ui, config);

            ui.add_space(8.0);
            ui.separator();
            ui.add_space(4.0);

            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    for seg in &self.transcript {
                        ui.horizontal_wrapped(|ui| {
                            ui.label(
                                egui::RichText::new(format!("[{}]", format_mmss(seg.elapsed)))
                                    .color(styles::TEXT_TIMESTAMP)
                                    .monospace(),
                            );
                            ui.label(egui::RichText::new(&seg.text).color(styles::TEXT_PRIMARY));
                        });
                    }
                });
        });
    }

    fn transport_row(&mut self, ui: &mut egui::Ui, config: &mut Config) {
        ui.horizontal(|ui| {
            let recording = self.state != MeetingState::Idle;
            if ui
                .add_enabled(!recording, egui::Button::new("● Start"))
                .clicked()
            {
                if let Err(e) = self.start(config) {
                    log::error!("meeting start: {e}");
                }
            }
            let pause_label = if self.state == MeetingState::Paused {
                "▶ Fortsetzen"
            } else {
                "⏸ Pause"
            };
            if ui
                .add_enabled(recording, egui::Button::new(pause_label))
                .clicked()
            {
                self.toggle_pause();
            }
            if ui
                .add_enabled(recording, egui::Button::new("⏹ Stop"))
                .clicked()
            {
                self.stop(config);
            }
            ui.separator();
            ui.label(
                egui::RichText::new(format_elapsed(self.elapsed())).color(styles::TEXT_SECONDARY),
            );
        });
    }

    fn input_row(&mut self, ui: &mut egui::Ui, config: &mut Config) {
        ui.horizontal(|ui| {
            ui.label("Eingabegerät:");
            let mut current = config
                .meeting
                .input_device
                .clone()
                .unwrap_or_else(|| "(System-Standard)".into());
            let prev = current.clone();
            egui::ComboBox::from_id_salt("meeting-input-device")
                .selected_text(&current)
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut current,
                        "(System-Standard)".into(),
                        "(System-Standard)",
                    );
                    for name in &self.input_devices {
                        ui.selectable_value(&mut current, name.clone(), name);
                    }
                });
            if current != prev {
                config.meeting.input_device = if current == "(System-Standard)" {
                    None
                } else {
                    Some(current)
                };
                let _ = config.save();
            }
            if ui
                .small_button("↻")
                .on_hover_text("Geräteliste neu laden")
                .clicked()
            {
                self.input_devices = list_input_devices();
                #[cfg(target_os = "macos")]
                {
                    self.ca_devices = coreaudio::list_devices().unwrap_or_default();
                }
            }
        });
    }

    #[cfg(target_os = "macos")]
    fn blackhole_row(&mut self, ui: &mut egui::Ui, config: &mut Config) {
        let bh = coreaudio::find_blackhole();
        ui.group(|ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("BlackHole Routing").strong());
                ui.label(
                    egui::RichText::new(if config.meeting.blackhole.configured {
                        "● aktiv"
                    } else {
                        "○ inaktiv"
                    })
                    .color(if config.meeting.blackhole.configured {
                        styles::ACCENT_GREEN
                    } else {
                        styles::TEXT_SECONDARY
                    }),
                );
            });

            if bh.is_none() {
                ui.label(
                    egui::RichText::new(BLACKHOLE_INSTALL_HINT)
                        .color(styles::TEXT_SECONDARY)
                        .size(styles::FONT_SIZE_SMALL),
                );
                return;
            }

            let configured = config.meeting.blackhole.configured;
            let mics: Vec<_> = self
                .ca_devices
                .iter()
                .filter(|d| d.is_input() && !d.name.to_lowercase().contains("blackhole"))
                .filter(|d| d.uid != BH_INPUT_UID && d.uid != BH_OUTPUT_UID)
                .cloned()
                .collect();
            let speakers: Vec<_> = self
                .ca_devices
                .iter()
                .filter(|d| d.is_output() && !d.name.to_lowercase().contains("blackhole"))
                .filter(|d| d.uid != BH_INPUT_UID && d.uid != BH_OUTPUT_UID)
                .cloned()
                .collect();

            let mut mic_uid = config.meeting.blackhole.mic_uid.clone();
            let mut spk_uid = config.meeting.blackhole.speaker_uid.clone();
            if mic_uid.is_empty() {
                mic_uid = mics.first().map(|d| d.uid.clone()).unwrap_or_default();
            }
            if spk_uid.is_empty() {
                spk_uid = coreaudio::default_output_uid().unwrap_or_default();
            }

            ui.horizontal(|ui| {
                ui.label("Mikrofon:");
                ui.add_enabled_ui(!configured, |ui| {
                    egui::ComboBox::from_id_salt("bh-mic")
                        .selected_text(label_of(&mics, &mic_uid))
                        .show_ui(ui, |ui| {
                            for d in &mics {
                                ui.selectable_value(&mut mic_uid, d.uid.clone(), &d.name);
                            }
                        });
                });
            });
            ui.horizontal(|ui| {
                ui.label("Lautsprecher:");
                ui.add_enabled_ui(!configured, |ui| {
                    egui::ComboBox::from_id_salt("bh-spk")
                        .selected_text(label_of(&speakers, &spk_uid))
                        .show_ui(ui, |ui| {
                            for d in &speakers {
                                ui.selectable_value(&mut spk_uid, d.uid.clone(), &d.name);
                            }
                        });
                });
            });

            ui.horizontal(|ui| {
                let recording = self.state != MeetingState::Idle;
                if ui
                    .add_enabled(
                        !configured && !recording && !mic_uid.is_empty() && !spk_uid.is_empty(),
                        egui::Button::new("Konfigurieren"),
                    )
                    .clicked()
                {
                    match setup_blackhole(&mic_uid, &spk_uid) {
                        Ok(setup) => {
                            config.meeting.input_device = Some(setup.input_device_name.clone());
                            config.meeting.blackhole = setup;
                            let _ = config.save();
                            self.bh_status.message =
                                Some(Ok("Routing aktiv, System-Audio + Mikro vereint.".into()));
                            self.ca_devices = coreaudio::list_devices().unwrap_or_default();
                            self.input_devices = list_input_devices();
                        }
                        Err(e) => {
                            self.bh_status.message =
                                Some(Err(format!("Setup fehlgeschlagen: {e}")));
                        }
                    }
                }
                if ui
                    .add_enabled(configured && !recording, egui::Button::new("Zurücksetzen"))
                    .clicked()
                {
                    match teardown_blackhole(&config.meeting.blackhole) {
                        Ok(()) => {
                            if config.meeting.input_device.as_deref()
                                == Some(config.meeting.blackhole.input_device_name.as_str())
                            {
                                config.meeting.input_device = None;
                            }
                            config.meeting.blackhole = BlackHoleSetup::default();
                            let _ = config.save();
                            self.bh_status.message = Some(Ok("Routing entfernt.".into()));
                            self.ca_devices = coreaudio::list_devices().unwrap_or_default();
                            self.input_devices = list_input_devices();
                        }
                        Err(e) => {
                            self.bh_status.message =
                                Some(Err(format!("Reset fehlgeschlagen: {e}")));
                        }
                    }
                }
            });

            // Update saved UID choices even before the user clicks
            // Konfigurieren, so the next launch remembers them.
            if !configured
                && (mic_uid != config.meeting.blackhole.mic_uid
                    || spk_uid != config.meeting.blackhole.speaker_uid)
            {
                config.meeting.blackhole.mic_uid = mic_uid;
                config.meeting.blackhole.speaker_uid = spk_uid;
                let _ = config.save();
            }

            if let Some(msg) = &self.bh_status.message {
                ui.add_space(2.0);
                match msg {
                    Ok(t) => ui.label(egui::RichText::new(t).color(styles::ACCENT_GREEN)),
                    Err(t) => ui.label(egui::RichText::new(t).color(styles::ACCENT_RED)),
                };
            }
        });
    }

    // --- lifecycle ------------------------------------------------------

    fn start(&mut self, config: &Config) -> anyhow::Result<()> {
        self.transcript.clear();
        self.writer.start_new(None)?;
        self.started_at = Some(Instant::now());
        self.paused_for = 0.0;
        self.pause_started = None;
        self.state = MeetingState::Recording;

        self.recorder
            .set_input_device(config.meeting.input_device.clone());

        self.spawn_worker();

        let chunk_seconds = config.meeting.chunk_duration_seconds.max(1);
        let save_full = config.meeting.save_audio;
        let started_at = self.started_at.unwrap();
        let tx = self.chunk_tx.as_ref().expect("worker spawned").clone();

        self.recorder.start_continuous(
            move |chunk| {
                let elapsed = started_at.elapsed().as_secs_f64();
                let _ = tx.send(ChunkJob {
                    elapsed,
                    audio: chunk,
                });
            },
            chunk_seconds,
            save_full,
        )?;
        Ok(())
    }

    fn toggle_pause(&mut self) {
        match self.state {
            MeetingState::Recording => {
                self.recorder.pause_continuous();
                self.pause_started = Some(Instant::now());
                self.state = MeetingState::Paused;
            }
            MeetingState::Paused => {
                self.recorder.resume_continuous();
                if let Some(start) = self.pause_started.take() {
                    self.paused_for += start.elapsed().as_secs_f64();
                }
                self.state = MeetingState::Recording;
            }
            MeetingState::Idle => {}
        }
    }

    fn stop(&mut self, _config: &Config) {
        let _ = self.recorder.stop_continuous();
        let total = self.elapsed();
        let _ = self.writer.finalize(total);
        self.state = MeetingState::Idle;
        self.started_at = None;
        self.pause_started = None;
        self.paused_for = 0.0;

        // Closing the channel makes the worker thread exit on the next
        // recv. Drop our sender; the dispatcher's sender was already
        // dropped when the recorder stopped.
        self.chunk_tx.take();
        if let Some(handle) = self.worker.take() {
            let _ = handle.join();
        }
    }

    fn drain_worker(&mut self) {
        while let Ok(seg) = self.segment_rx.try_recv() {
            let _ = self.writer.add_segment(seg.elapsed, &seg.text);
            self.transcript.push(seg);
        }
    }

    fn spawn_worker(&mut self) {
        let (tx, rx) = mpsc::channel::<ChunkJob>();
        self.chunk_tx = Some(tx);
        let engine = Arc::clone(&self.engine);
        let segment_tx = self.segment_tx.clone();

        self.worker = Some(
            thread::Builder::new()
                .name("whisprking-meeting-worker".into())
                .spawn(move || {
                    while let Ok(job) = rx.recv() {
                        let result = {
                            let guard = engine.lock().expect("engine");
                            guard.transcribe(&job.audio, crate::audio::recorder::SAMPLE_RATE)
                        };
                        match result {
                            Ok(text) if !text.trim().is_empty() => {
                                let _ = segment_tx.send(Segment {
                                    elapsed: job.elapsed,
                                    text: text.trim().to_string(),
                                });
                            }
                            Ok(_) => {}
                            Err(e) => log::warn!("meeting transcribe failed: {e}"),
                        }
                    }
                })
                .expect("spawn meeting worker"),
        );
    }

    fn elapsed(&self) -> f64 {
        let Some(start) = self.started_at else {
            return 0.0;
        };
        let now = self
            .pause_started
            .map(|p| {
                p.elapsed().as_secs_f64() + start.elapsed().as_secs_f64()
                    - p.elapsed().as_secs_f64()
            })
            .unwrap_or_else(|| start.elapsed().as_secs_f64());
        (now - self.paused_for).max(0.0)
    }
}

fn list_input_devices() -> Vec<String> {
    let host = cpal::default_host();
    let Ok(devs) = host.input_devices() else {
        return Vec::new();
    };
    let mut out: Vec<String> = devs
        .filter_map(|d| d.description().ok().map(|desc| desc.name().to_string()))
        .collect();
    out.sort();
    out.dedup();
    out
}

#[cfg(target_os = "macos")]
fn label_of(list: &[coreaudio::DeviceInfo], uid: &str) -> String {
    list.iter()
        .find(|d| d.uid == uid)
        .map(|d| d.name.clone())
        .unwrap_or_else(|| "—".into())
}

/// Create the Multi-Output + Aggregate pair and switch the system default
/// output. On any failure we try to roll back already-created devices so
/// the user is not left in a half-configured state.
#[cfg(target_os = "macos")]
fn setup_blackhole(mic_uid: &str, speaker_uid: &str) -> Result<BlackHoleSetup, String> {
    let bh = coreaudio::find_blackhole().ok_or_else(|| "BlackHole not installed".to_string())?;
    let previous_default = coreaudio::default_output_uid().unwrap_or_default();

    // Multi-Output: speakers + BlackHole, stacked so audio plays to both.
    let multi = coreaudio::create_aggregate(
        BH_OUTPUT_NAME,
        BH_OUTPUT_UID,
        speaker_uid,
        &[speaker_uid, &bh.uid],
        true,
    )
    .map_err(|e| format!("create multi-output: {e}"))?;

    // Aggregate input: BlackHole + Mic, not stacked → multi-channel input.
    let agg = match coreaudio::create_aggregate(
        BH_INPUT_NAME,
        BH_INPUT_UID,
        &bh.uid,
        &[&bh.uid, mic_uid],
        false,
    ) {
        Ok(d) => d,
        Err(e) => {
            let _ = coreaudio::destroy_aggregate_by_uid(BH_OUTPUT_UID);
            return Err(format!("create aggregate input: {e}"));
        }
    };

    if let Err(e) = coreaudio::set_default_output_by_uid(BH_OUTPUT_UID) {
        let _ = coreaudio::destroy_aggregate_by_uid(BH_INPUT_UID);
        let _ = coreaudio::destroy_aggregate_by_uid(BH_OUTPUT_UID);
        return Err(format!("set default output: {e}"));
    }

    Ok(BlackHoleSetup {
        configured: true,
        mic_uid: mic_uid.into(),
        speaker_uid: speaker_uid.into(),
        output_uid: multi.uid,
        input_uid: agg.uid,
        previous_default_output_uid: previous_default,
        input_device_name: agg.name,
    })
}

#[cfg(target_os = "macos")]
fn teardown_blackhole(setup: &BlackHoleSetup) -> Result<(), String> {
    if !setup.previous_default_output_uid.is_empty() {
        let _ = coreaudio::set_default_output_by_uid(&setup.previous_default_output_uid);
    }
    if !setup.input_uid.is_empty() {
        coreaudio::destroy_aggregate_by_uid(&setup.input_uid)
            .map_err(|e| format!("destroy input: {e}"))?;
    }
    if !setup.output_uid.is_empty() {
        coreaudio::destroy_aggregate_by_uid(&setup.output_uid)
            .map_err(|e| format!("destroy output: {e}"))?;
    }
    Ok(())
}

fn format_mmss(seconds: f64) -> String {
    let s = seconds.max(0.0) as u64;
    format!("{:02}:{:02}", s / 60, s % 60)
}

fn format_elapsed(seconds: f64) -> String {
    let s = seconds.max(0.0) as u64;
    format!("{:02}:{:02}:{:02}", s / 3600, (s % 3600) / 60, s % 60)
}
