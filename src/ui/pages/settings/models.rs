//! Model picker and download progress inside the settings page.

use std::sync::{Arc, Mutex};

use crate::config::Config;
use crate::transcription::model_manager::{available_models, ModelManager};
use crate::ui::icons::Icon;
use crate::ui::theme::{self, radius, space};
use crate::ui::widgets::{self as w, ButtonKind, Tone};

use super::{status_line, SettingsPage};

#[derive(Clone)]
pub enum DownloadPhase {
    Downloading { downloaded: u64, total: u64 },
    Extracting,
    Done,
    Error(String),
}

/// A model download in flight, shared with its worker thread.
#[derive(Clone, Default)]
pub struct DownloadJob {
    slot: Arc<Mutex<Option<(String, DownloadPhase)>>>,
}

impl DownloadJob {
    fn snapshot(&self) -> Option<(String, DownloadPhase)> {
        self.slot.lock().expect("download job").clone()
    }

    fn busy(&self) -> bool {
        matches!(
            self.snapshot().map(|(_, p)| p),
            Some(DownloadPhase::Downloading { .. }) | Some(DownloadPhase::Extracting)
        )
    }

    fn clear(&self) {
        *self.slot.lock().expect("download job") = None;
    }

    fn start(&self, model: &str, models_dir: std::path::PathBuf, ctx: &egui::Context) {
        let Some(spec) = available_models().get(model) else {
            *self.slot.lock().expect("download job") = Some((
                model.to_string(),
                DownloadPhase::Error("Unbekanntes Modell".into()),
            ));
            return;
        };
        let est_total = (spec.size_mb as u64) * 1024 * 1024;
        *self.slot.lock().expect("download job") = Some((
            model.to_string(),
            DownloadPhase::Downloading {
                downloaded: 0,
                total: est_total,
            },
        ));

        let slot = Arc::clone(&self.slot);
        let worker_ctx = ctx.clone();
        let model = model.to_string();
        std::thread::spawn(move || {
            let manager = ModelManager::new(models_dir);
            let progress_slot = Arc::clone(&slot);
            let progress_ctx = worker_ctx.clone();
            let progress_model = model.clone();
            let progress = Box::new(move |downloaded: u64, total: u64| {
                let phase = if downloaded >= total && total > 0 {
                    // bytes streamed; for sherpa, extraction follows
                    DownloadPhase::Extracting
                } else {
                    DownloadPhase::Downloading { downloaded, total }
                };
                *progress_slot.lock().expect("download job") =
                    Some((progress_model.clone(), phase));
                progress_ctx.request_repaint();
            });

            let result = manager.download(&model, Some(progress));
            let phase = match result {
                Ok(_) => DownloadPhase::Done,
                Err(e) => DownloadPhase::Error(e.to_string()),
            };
            *slot.lock().expect("download job") = Some((model, phase));
            worker_ctx.request_repaint();
        });
        ctx.request_repaint();
    }
}

impl SettingsPage {
    /// Model combo, download button, and the progress that follows it.
    pub(super) fn model_row(&mut self, ui: &mut egui::Ui, config: &mut Config) {
        let manager = ModelManager::new(Config::models_dir(config));
        let installed: Vec<&'static str> = manager.list_downloaded();
        let busy = self.download.busy();

        // Smallest first, so the list reads the way the hint under it does
        // ("Größere Modelle sind genauer und langsamer") instead of putting
        // `whisper-tiny` between `small` and `turbo` alphabetically.
        let mut names: Vec<&'static str> = available_models().keys().copied().collect();
        names.sort_by_key(|name| {
            let size = available_models().get(name).map(|s| s.size_mb).unwrap_or(0);
            (size, *name)
        });

        let options: Vec<(String, String)> = names
            .iter()
            .map(|name| {
                let spec = available_models().get(name);
                let display = spec.map(|s| s.display_name).unwrap_or(*name);
                let size = spec.map(|s| s.size_mb).unwrap_or(0);
                let suffix = if installed.contains(name) {
                    "installiert".to_string()
                } else {
                    format!("{size} MB · nicht installiert")
                };
                ((*name).to_string(), format!("{display}  ·  {suffix}"))
            })
            .collect();

        let current = config.dictation.model.clone();
        let current_installed = installed.contains(&current.as_str());
        let label = options
            .iter()
            .find(|(key, _)| *key == current)
            .map(|(_, label)| label.clone())
            .unwrap_or_else(|| current.clone());

        let mut download = false;
        w::setting_row(
            ui,
            "Modell",
            "Größere Modelle sind genauer und langsamer.",
            |ui| {
                if !current_installed {
                    download = ui
                        .add_enabled_ui(!busy, |ui| {
                            w::button(
                                ui,
                                ButtonKind::Primary,
                                Some(Icon::Download),
                                "Herunterladen",
                            )
                        })
                        .inner
                        .clicked();
                }
                if w::combo(
                    ui,
                    "dictation-model",
                    280.0,
                    &mut config.dictation.model,
                    &options,
                    &label,
                ) {
                    let _ = config.save();
                }
            },
        );

        if current_installed {
            ui.add_space(space::XS);
            w::badge(ui, "Installiert", Tone::Success);
        }

        if download {
            let ctx = ui.ctx().clone();
            self.download
                .start(&config.dictation.model, Config::models_dir(config), &ctx);
        }

        self.download_progress(ui);
    }

    fn download_progress(&mut self, ui: &mut egui::Ui) {
        let Some((model, phase)) = self.download.snapshot() else {
            return;
        };
        ui.add_space(space::SM);
        match phase {
            DownloadPhase::Downloading { downloaded, total } => {
                let fraction = if total > 0 {
                    (downloaded as f32 / total as f32).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                status_line(
                    ui,
                    &format!(
                        "{model} wird geladen … {} / {}",
                        fmt_bytes(downloaded),
                        fmt_bytes(total)
                    ),
                );
                ui.add_space(space::XS);
                ui.add(
                    egui::ProgressBar::new(fraction)
                        .desired_height(6.0)
                        .corner_radius(radius::PILL)
                        .fill(theme::ACCENT),
                );
                ui.ctx().request_repaint();
            },
            DownloadPhase::Extracting => w::busy_row(ui, &format!("{model} wird entpackt …")),
            DownloadPhase::Done => {
                w::banner(ui, Tone::Success, &format!("{model} ist installiert."));
                ui.add_space(space::XS);
                if w::button(ui, ButtonKind::Ghost, None, "Ausblenden").clicked() {
                    self.download.clear();
                }
            },
            DownloadPhase::Error(message) => {
                w::banner(
                    ui,
                    Tone::Danger,
                    &format!("Download fehlgeschlagen ({model}): {message}"),
                );
                ui.add_space(space::XS);
                if w::button(ui, ButtonKind::Ghost, None, "Ausblenden").clicked() {
                    self.download.clear();
                }
            },
        }
    }
}

fn fmt_bytes(n: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;
    const GB: u64 = 1024 * MB;
    if n == 0 {
        "?".into()
    } else if n >= GB {
        format!("{:.2} GB", n as f64 / GB as f64)
    } else if n >= MB {
        format!("{:.1} MB", n as f64 / MB as f64)
    } else if n >= KB {
        format!("{:.0} KB", n as f64 / KB as f64)
    } else {
        format!("{n} B")
    }
}
