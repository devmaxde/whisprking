//! "Meeting" — record a call, watch it being transcribed, save it.
//!
//! The view is three blocks: a transport card (record / pause / clock /
//! levels), the source settings that belong to it, and the live transcript
//! filling everything below. Importing a file is the same pipeline without
//! the microphone, so it lives here too — as a button in the header and a
//! drop target over the whole page.

mod import;
pub mod post;
mod session;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait};

use crate::audio::capture::{SourceMode, Track};
use crate::audio::import::ImportPhase;
use crate::transcription::post::{ModelProgress, PostPhase, RunState};
use crate::audio::system_audio::{
    detect_blackhole, system_audio_status, SystemAudioStatus, BLACKHOLE_NO_LONGER_NEEDED,
};
use crate::config::Config;
use crate::transcription::engine::Transcriber;
use crate::ui::icons::{self, Icon};
use crate::ui::theme::{self, radius, space, text};
use crate::ui::widgets::{self as w, ButtonKind, Tone};

use import::ImportJob;
use post::PostJob;
use session::{fmt_hhmmss, fmt_mmss, Session};

pub use session::MeetingState;

const SYSTEM_DEFAULT: &str = "(Systemstandard)";

pub struct MeetingPage {
    session: Session,
    engine: Arc<Mutex<Box<dyn Transcriber>>>,
    import: ImportJob,
    post: PostJob,

    input_devices: Vec<String>,
    system_status: SystemAudioStatus,
    blackhole_still_installed: bool,

    status: Option<(Tone, String)>,
    copy: w::CopyButton,
}

impl MeetingPage {
    pub fn new(config: &Config, engine: Box<dyn Transcriber>) -> Result<Self, anyhow::Error> {
        let engine = Arc::new(Mutex::new(engine));
        let session = Session::new(
            Config::transcripts_dir(config),
            Config::audio_dir(config),
            Arc::clone(&engine),
        )?;
        Ok(Self {
            session,
            engine,
            import: ImportJob::default(),
            post: PostJob::default(),
            input_devices: list_input_devices(),
            system_status: system_audio_status(),
            blackhole_still_installed: detect_blackhole(),
            status: None,
            copy: w::CopyButton::default(),
        })
    }

    pub fn state(&self) -> MeetingState {
        self.session.state
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, config: &mut Config) {
        self.session.drain();

        // A file dropped anywhere on the page starts an import.
        if self.can_import() {
            if let Some(path) = import::first_dropped(ui.ctx()) {
                self.start_import(path, config, ui.ctx());
            }
        }

        self.header(ui, config);
        ui.add_space(space::LG);
        self.transport_card(ui, config);
        ui.add_space(space::MD);

        if let Some((tone, message)) = self.status.clone() {
            w::banner(ui, tone, &message);
            ui.add_space(space::MD);
        }
        self.import_status(ui);
        self.post_status(ui);

        self.transcript_card(ui, config);

        if import::hovering_file(ui.ctx()) && self.can_import() {
            drop_hint(ui);
        }

        // Keep the clock, the meters and any import progress moving.
        if !self.session.is_idle() || self.import.running() || self.post.running() {
            ui.ctx().request_repaint();
        }
    }

    fn can_import(&self) -> bool {
        self.session.is_idle() && !self.import.running() && !self.post.running()
    }

    fn header(&mut self, ui: &mut egui::Ui, config: &Config) {
        let can_import = self.can_import();
        let mut pick = false;
        egui::Sides::new().height(44.0).show(
            ui,
            |ui| {
                w::page_header(
                    ui,
                    "Meeting",
                    "Nimm beide Seiten des Gesprächs auf — getrennt transkribiert.",
                );
            },
            |ui| {
                pick = ui
                    .add_enabled_ui(can_import, |ui| {
                        w::button(
                            ui,
                            ButtonKind::Secondary,
                            Some(Icon::Import),
                            "Aufnahme importieren",
                        )
                    })
                    .inner
                    .on_hover_text("Audio- oder Videodatei transkribieren und zusammenfassen")
                    .on_disabled_hover_text("Erst die laufende Aufnahme beenden.")
                    .clicked();
            },
        );
        if pick {
            if let Some(path) = import::pick_file() {
                let ctx = ui.ctx().clone();
                self.start_import(path, config, &ctx);
            }
        }
    }

    // --- transport --------------------------------------------------------

    fn transport_card(&mut self, ui: &mut egui::Ui, config: &mut Config) {
        let ctx = ui.ctx().clone();
        let recording = !self.session.is_idle();
        let paused = self.session.state == MeetingState::Paused;
        let importing = self.import.running();

        let mut toggle_record = false;
        let mut toggle_pause = false;

        w::card(ui, |ui| {
            ui.set_width(ui.available_width());
            egui::Sides::new().height(theme::CONTROL_HEIGHT_LG).show(
                ui,
                |ui| {
                    let (kind, icon, label) = if recording {
                        (ButtonKind::RecordActive, Icon::Stop, "Aufnahme beenden")
                    } else {
                        (ButtonKind::Primary, Icon::Record, "Aufnahme starten")
                    };
                    toggle_record = ui
                        .add_enabled_ui(!importing, |ui| {
                            w::button_large(ui, kind, Some(icon), label)
                        })
                        .inner
                        .on_disabled_hover_text("Der Import läuft noch.")
                        .clicked();

                    if recording {
                        ui.add_space(space::XS);
                        let icon = if paused { Icon::Play } else { Icon::Pause };
                        let tip = if paused { "Fortsetzen" } else { "Pausieren" };
                        toggle_pause = w::icon_button(ui, icon, tip).clicked();
                    }

                    ui.add_space(space::MD);
                    clock(ui, self.session.duration, recording, paused);
                },
                |ui| {
                    if let Some((mic_on, mic, sys_on, system)) =
                        self.session.levels().filter(|_| recording)
                    {
                        if sys_on {
                            w::level_meter(ui, Track::System.label(), theme::TRACK_THEIRS, system);
                        }
                        if mic_on {
                            w::level_meter(ui, Track::Mic.label(), theme::TRACK_MINE, mic);
                        }
                    }
                },
            );

            ui.add_space(space::MD);
            w::divider(ui);
            self.source_row(ui, config, recording);
        });

        if toggle_record {
            if recording {
                self.stop(config, &ctx);
            } else {
                self.start(config);
            }
        }
        if toggle_pause {
            self.session.toggle_pause();
        }
    }

    /// Which sides to record, and from which microphone.
    fn source_row(&mut self, ui: &mut egui::Ui, config: &mut Config, locked: bool) {
        let system_ok = self.system_status.is_available();
        let mut mode = SourceMode::from_config(&config.meeting.audio_source);
        let mut device = config
            .meeting
            .input_device
            .clone()
            .unwrap_or_else(|| SYSTEM_DEFAULT.into());

        let mut mode_changed = false;
        let mut device_changed = false;
        let mut reload = false;

        ui.add_enabled_ui(!locked, |ui| {
            egui::Sides::new().height(theme::CONTROL_HEIGHT).show(
                ui,
                |ui| {
                    w::text_line(ui, "Aufnehmen", text::body(), theme::TEXT_SECONDARY);
                    ui.add_space(space::SM);
                    mode_changed = w::segmented(
                        ui,
                        "meeting-source",
                        &mut mode,
                        &[
                            (SourceMode::Both, "Beide Seiten", true),
                            (SourceMode::MicOnly, "Nur ich", true),
                            (SourceMode::SystemOnly, "Nur andere", system_ok),
                        ],
                    );
                },
                |ui| {
                    reload = w::icon_button(ui, Icon::Refresh, "Geräte neu einlesen").clicked();
                    let mut options: Vec<(String, String)> =
                        vec![(SYSTEM_DEFAULT.to_string(), SYSTEM_DEFAULT.to_string())];
                    options.extend(
                        self.input_devices
                            .iter()
                            .map(|d| (d.clone(), d.clone())),
                    );
                    let label = device.clone();
                    device_changed =
                        w::combo(ui, "meeting-device", 220.0, &mut device, &options, &label);
                    ui.add_space(space::XS);
                    w::text_line(ui, "Mikrofon", text::body(), theme::TEXT_SECONDARY);
                },
            );
        });

        ui.add_space(space::SM);
        match self.system_status.hint() {
            Some(hint) => w::banner(ui, Tone::Warning, hint),
            None => w::hint(
                ui,
                "Die anderen Teilnehmer werden direkt aus dem System-Audio mitgeschnitten. \
                 Lass in deiner Meeting-App weiterhin dein echtes Mikrofon eingestellt — \
                 WhisprKing übernimmt es nicht.",
            ),
        }
        if self.blackhole_still_installed {
            ui.add_space(space::XS);
            w::hint(ui, BLACKHOLE_NO_LONGER_NEEDED);
        }

        if mode_changed {
            config.meeting.audio_source = mode.as_config().into();
            let _ = config.save();
        }
        if device_changed {
            config.meeting.input_device = (device != SYSTEM_DEFAULT).then_some(device);
            let _ = config.save();
        }
        if reload {
            self.input_devices = list_input_devices();
            self.system_status = system_audio_status();
            self.blackhole_still_installed = detect_blackhole();
        }
    }

    // --- transcript -------------------------------------------------------

    fn transcript_card(&mut self, ui: &mut egui::Ui, config: &Config) {
        let label_speakers = config.meeting.label_speakers;
        let count = self.session.transcript.len();
        let path = self.session.path().map(|p| p.to_path_buf());
        let markdown = self.session.transcript_markdown(label_speakers);

        w::card(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.set_min_height(ui.available_height());

            egui::Sides::new().height(theme::CONTROL_HEIGHT).show(
                ui,
                |ui| {
                    w::section_title(ui, Some(Icon::Waveform), "Live-Transkript");
                    if count > 0 {
                        ui.add_space(space::SM);
                        w::badge(ui, &format!("{count} Zeilen"), Tone::Neutral);
                    }
                },
                |ui| {
                    if let Some(path) = &path {
                        if w::icon_button(ui, Icon::Reveal, "Im Finder zeigen").clicked() {
                            super::history::reveal(path);
                        }
                    }
                    if count > 0 {
                        let payload = markdown.clone();
                        self.copy
                            .show(ui, ButtonKind::Secondary, "Kopieren", move || payload);
                    }
                },
            );

            ui.add_space(space::SM);
            theme::inset_frame().show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.set_min_height(ui.available_height().max(140.0));
                egui::ScrollArea::vertical()
                    .id_salt("meeting-transcript")
                    .auto_shrink([false, false])
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        if self.session.transcript.is_empty() {
                            let (title, hint) = if self.session.is_idle() {
                                (
                                    "Noch nichts aufgenommen",
                                    "Starte die Aufnahme — der Text erscheint hier, während gesprochen wird.",
                                )
                            } else {
                                ("Hört zu …", "Sobald jemand spricht, erscheint der Text hier.")
                            };
                            w::empty_state(ui, Icon::Waveform, title, hint);
                            return;
                        }
                        ui.spacing_mut().item_spacing.y = space::SM;
                        for segment in &self.session.transcript {
                            w::transcript_line(
                                ui,
                                &fmt_mmss(segment.elapsed),
                                label_speakers.then(|| segment.track.label()),
                                &segment.text,
                            );
                        }
                    });
            });
        });
    }

    // --- import -----------------------------------------------------------

    fn start_import(&mut self, path: PathBuf, config: &Config, ctx: &egui::Context) {
        self.status = None;
        self.import
            .start(path, config, Arc::clone(&self.engine), ctx);
    }

    fn import_status(&mut self, ui: &mut egui::Ui) {
        let Some(phase) = self.import.phase() else {
            return;
        };
        let mut dismiss = false;
        match phase {
            ImportPhase::Decoding => w::busy_row(ui, "Audio wird dekodiert …"),
            ImportPhase::Diarizing => w::busy_row(ui, "Sprecher werden unterschieden …"),
            ImportPhase::Transcribing {
                done_secs,
                total_secs,
            } => {
                w::busy_row(
                    ui,
                    &format!(
                        "Wird transkribiert … {} / {}",
                        fmt_mmss(done_secs),
                        fmt_mmss(total_secs)
                    ),
                );
                let fraction = if total_secs > 0.0 {
                    (done_secs / total_secs).clamp(0.0, 1.0) as f32
                } else {
                    0.0
                };
                ui.add(
                    egui::ProgressBar::new(fraction)
                        .desired_height(6.0)
                        .corner_radius(radius::PILL)
                        .fill(theme::ACCENT),
                );
            },
            ImportPhase::Summarizing => w::busy_row(ui, "Zusammenfassung wird erstellt …"),
            ImportPhase::Done { transcript, note } => {
                let name = transcript
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or_default();
                w::banner(ui, Tone::Success, &format!("Import fertig: {name}"));
                if let Some(note) = note {
                    ui.add_space(space::XS);
                    w::banner(ui, Tone::Warning, &note);
                }
                ui.add_space(space::XS);
                dismiss = w::button(ui, ButtonKind::Ghost, None, "Ausblenden").clicked();
            },
            ImportPhase::Error(msg) => {
                w::banner(ui, Tone::Danger, &format!("Import fehlgeschlagen: {msg}"));
                ui.add_space(space::XS);
                dismiss = w::button(ui, ButtonKind::Ghost, None, "Ausblenden").clicked();
            },
        }
        ui.add_space(space::MD);
        if dismiss {
            self.import.clear();
        }
    }

    // --- lifecycle --------------------------------------------------------

    fn start(&mut self, config: &Config) {
        self.status = None;
        self.import.clear();
        match self.session.start(config) {
            Ok(()) => {
                if let Some(warning) = self.session.start_warning() {
                    self.status = Some((Tone::Warning, warning));
                }
            },
            Err(e) => self.status = Some((Tone::Danger, e)),
        }
    }

    fn stop(&mut self, config: &Config, ctx: &egui::Context) {
        let Some(report) = self.session.stop() else {
            self.status = None;
            return;
        };
        let name = report
            .transcript
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        self.status = Some((
            Tone::Success,
            format!("Gespeichert als {name} — im Verlauf zu finden."),
        ));

        if config.meeting.post_transcribe.enabled {
            self.post.clear();
            self.post.start(report.transcript, report.tracks, config, ctx);
        }
    }

    // --- post-transcription -----------------------------------------------

    /// Progress of the second pass. Shows one row per model, because the whole
    /// point is that they run at the same time — a single bar would hide which
    /// one is holding things up.
    fn post_status(&mut self, ui: &mut egui::Ui) {
        let Some(phase) = self.post.phase() else {
            return;
        };
        let mut dismiss = false;
        match phase {
            PostPhase::Preparing => {
                w::busy_row(ui, "Nachbearbeitung: Aufnahme wird gelesen und segmentiert …")
            },
            PostPhase::Diarizing => {
                w::busy_row(ui, "Nachbearbeitung: Sprecher werden unterschieden …")
            },
            PostPhase::Transcribing(models) => {
                w::busy_row(
                    ui,
                    &format!(
                        "Nachbearbeitung läuft — {} Modell(e) parallel",
                        models.len()
                    ),
                );
                ui.add_space(space::XS);
                for model in &models {
                    model_row(ui, model);
                }
            },
            PostPhase::Reconciling => {
                w::busy_row(ui, "Nachbearbeitung: Varianten werden zusammengeführt …")
            },
            PostPhase::Done {
                post,
                variants,
                note,
            } => {
                let name = post.file_name().and_then(|s| s.to_str()).unwrap_or_default();
                w::banner(ui, Tone::Success, &format!("Nachbearbeitung fertig: {name}"));
                if variants.is_some() {
                    ui.add_space(space::XS);
                    w::hint(
                        ui,
                        "Der Modellvergleich steht im Verlauf als eigener Tab.",
                    );
                }
                if let Some(note) = note {
                    ui.add_space(space::XS);
                    w::banner(ui, Tone::Warning, &note);
                }
                ui.add_space(space::XS);
                dismiss = w::button(ui, ButtonKind::Ghost, None, "Ausblenden").clicked();
            },
            PostPhase::Error(msg) => {
                w::banner(ui, Tone::Danger, &format!("Nachbearbeitung: {msg}"));
                ui.add_space(space::XS);
                dismiss = w::button(ui, ButtonKind::Ghost, None, "Ausblenden").clicked();
            },
        }
        ui.add_space(space::MD);
        if dismiss {
            self.post.clear();
        }
    }
}

/// One model's line in the post-transcription progress block.
fn model_row(ui: &mut egui::Ui, model: &ModelProgress) {
    egui::Sides::new().height(theme::CONTROL_HEIGHT).show(
        ui,
        |ui| {
            w::text_line(ui, &model.model, text::body(), theme::TEXT_PRIMARY);
            ui.add_space(space::SM);
            if model.window_seconds > 0.0 {
                w::badge(
                    ui,
                    &format!("{} · {:.0}s-Fenster", model.backend, model.window_seconds),
                    Tone::Neutral,
                );
            }
        },
        |ui| match &model.state {
            RunState::Loading => w::text_line(
                ui,
                "wird geladen …",
                text::small(),
                theme::TEXT_MUTED,
            ),
            RunState::Running => w::text_line(
                ui,
                &format!(
                    "{} / {}",
                    fmt_mmss(model.done_secs),
                    fmt_mmss(model.total_secs)
                ),
                text::small(),
                theme::TEXT_MUTED,
            ),
            RunState::Done { seconds } => {
                w::badge(ui, &format!("fertig in {seconds:.0}s"), Tone::Success)
            },
            RunState::Failed(e) => {
                // Engine errors carry paths and provider detail; the row has
                // room for a headline, the log has the rest.
                let short: String = e.chars().take(60).collect();
                w::badge(ui, &short, Tone::Danger).on_hover_text(e)
            },
        },
    );
    if matches!(model.state, RunState::Loading | RunState::Running) {
        ui.add(
            egui::ProgressBar::new(model.fraction())
                .desired_height(6.0)
                .corner_radius(radius::PILL)
                .fill(theme::ACCENT),
        );
    }
    ui.add_space(space::XS);
}

/// Elapsed time with a pulsing dot while recording.
fn clock(ui: &mut egui::Ui, seconds: f64, recording: bool, paused: bool) {
    let (dot, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
    if recording {
        let pulse = if paused {
            0.5
        } else {
            let t = ui.ctx().input(|i| i.time) as f32;
            0.55 + 0.45 * (t * 3.0).sin().abs()
        };
        ui.painter().circle_filled(
            dot.center(),
            4.0,
            theme::DANGER.gamma_multiply(pulse),
        );
    }
    ui.add_space(space::XS);
    w::text_line(
        ui,
        &fmt_hhmmss(seconds),
        text::mono_large(),
        if recording {
            theme::TEXT_PRIMARY
        } else {
            theme::TEXT_MUTED
        },
    );
    if paused {
        ui.add_space(space::SM);
        w::badge(ui, "Pausiert", Tone::Warning);
    }
}

/// Dashed overlay shown while a file is dragged over the window.
fn drop_hint(ui: &mut egui::Ui) {
    let rect = ui.ctx().content_rect().shrink(24.0);
    let painter = ui.ctx().layer_painter(egui::LayerId::new(
        egui::Order::Foreground,
        egui::Id::new("meeting-drop-hint"),
    ));
    painter.rect(
        rect,
        radius::LG,
        theme::tint(theme::ACCENT, 0.10),
        egui::Stroke::new(2.0, theme::ACCENT_HOVER),
        egui::StrokeKind::Inside,
    );
    let galley = painter.layout_no_wrap(
        "Loslassen, um zu importieren".to_owned(),
        text::title(),
        theme::TEXT_PRIMARY,
    );
    let icon_rect =
        egui::Rect::from_center_size(rect.center() - egui::vec2(0.0, 26.0), egui::Vec2::splat(32.0));
    icons::paint(&painter, Icon::Import, icon_rect, theme::ACCENT_HOVER);
    painter.galley(
        egui::pos2(
            rect.center().x - galley.size().x / 2.0,
            rect.center().y,
        ),
        galley,
        theme::TEXT_PRIMARY,
    );
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
