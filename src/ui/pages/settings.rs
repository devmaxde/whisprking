//! "Einstellungen" tab — edit config and trigger model downloads.
//!
//! All changes are flushed to disk via [`Config::save`] as soon as the
//! relevant widget reports a finished edit (combo selection, file dialog,
//! checkbox toggle, edit-finish on text inputs). The page never owns the
//! config; the caller passes a `&mut Config` each frame.

use std::sync::{Arc, Mutex};

use crate::config::Config;
use crate::hotkey::listener::HotkeyName;
use crate::postprocess::llm::PRESETS;
use crate::transcription::model_manager::{available_models, ModelManager};

use super::super::styles;

#[derive(Clone)]
enum DownloadPhase {
    Downloading { downloaded: u64, total: u64 },
    Extracting,
    Done,
    Error(String),
}

struct DownloadJob {
    model: String,
    phase: DownloadPhase,
}

type SharedJob = Arc<Mutex<Option<DownloadJob>>>;

const LANGUAGES: &[(&str, &str)] = &[
    ("auto", "Auto"),
    ("de", "Deutsch"),
    ("en", "English"),
    ("fr", "Français"),
    ("es", "Español"),
    ("it", "Italiano"),
];

const MODES: &[(&str, &str)] = &[("hold_to_talk", "Hold-to-Talk"), ("toggle", "Toggle")];

const PROVIDERS: &[(&str, &str)] = &[
    ("none", "Aus"),
    ("openrouter", "OpenRouter"),
    ("lm_studio", "LM Studio (lokal)"),
    ("ollama", "Ollama (lokal)"),
];

fn provider_uses_api_key(provider: &str) -> bool {
    matches!(provider, "openrouter")
}

pub struct SettingsPage {
    pub hotkey_changed: Option<String>,
    pub data_dir_changed: bool,
    download_message: Option<String>,
    download_job: SharedJob,
    fetch_message: Option<String>,
    available_openrouter_models: Vec<String>,
}

impl SettingsPage {
    pub fn new() -> Self {
        Self {
            hotkey_changed: None,
            data_dir_changed: false,
            download_message: None,
            download_job: Arc::new(Mutex::new(None)),
            fetch_message: None,
            available_openrouter_models: Vec::new(),
        }
    }

    /// Reset one-shot signal flags. Call at the top of each frame *after*
    /// the parent has consumed them.
    pub fn clear_signals(&mut self) {
        self.hotkey_changed = None;
        self.data_dir_changed = false;
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, config: &mut Config) {
        ui.vertical(|ui| {
            ui.add_space(8.0);
            ui.heading("Einstellungen");
            ui.add_space(8.0);

            egui::Grid::new("settings-main")
                .num_columns(2)
                .spacing([16.0, 10.0])
                .show(ui, |ui| {
                    self.row_data_dir(ui, config);
                    self.row_hotkey(ui, config);
                    self.row_mode(ui, config);
                    self.row_model(ui, config);
                    self.row_language(ui, config);
                    self.row_save_audio(ui, config);
                });

            ui.add_space(12.0);
            ui.separator();
            ui.add_space(8.0);
            ui.label(
                egui::RichText::new("AI-Nachbearbeitung")
                    .size(16.0)
                    .strong(),
            );
            ui.add_space(4.0);

            egui::Grid::new("settings-ai")
                .num_columns(2)
                .spacing([16.0, 10.0])
                .show(ui, |ui| {
                    self.row_provider(ui, config);
                    if provider_uses_api_key(&config.ai_postprocess.provider) {
                        self.row_api_key(ui, config);
                    }
                    self.row_ai_model(ui, config);
                    self.row_default_prompt(ui, config);
                    self.row_dictation_autorun(ui, config);
                    self.row_dictation_preset(ui, config);
                });

            self.render_download_status(ui);

            if let Some(msg) = &self.download_message {
                ui.add_space(8.0);
                ui.label(egui::RichText::new(msg).color(styles::TEXT_SECONDARY));
            }
            if let Some(msg) = &self.fetch_message {
                ui.add_space(4.0);
                ui.label(egui::RichText::new(msg).color(styles::TEXT_SECONDARY));
            }

            ui.add_space(8.0);
            ui.label(
                egui::RichText::new("Modell- oder Hotkey-Wechsel werden sofort übernommen.")
                    .color(styles::TEXT_SECONDARY)
                    .size(styles::FONT_SIZE_SMALL),
            );
        });
    }

    // --- rows -----------------------------------------------------------

    fn row_data_dir(&mut self, ui: &mut egui::Ui, config: &mut Config) {
        ui.label("Datenordner:");
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new(
                    Config::resolve_data_dir(&config.data_dir)
                        .to_string_lossy()
                        .into_owned(),
                )
                .color(styles::TEXT_SECONDARY),
            );
            if ui.button("Ändern…").clicked() {
                if let Some(dir) = rfd_pick_folder(&config.data_dir) {
                    config.data_dir = dir.to_string_lossy().into_owned();
                    let _ = Config::ensure_dirs(&config.data_dir);
                    let _ = config.save();
                    self.data_dir_changed = true;
                }
            }
        });
        ui.end_row();
    }

    fn row_hotkey(&mut self, ui: &mut egui::Ui, config: &mut Config) {
        ui.label("Hotkey:");
        let mut current = config.dictation.hotkey.clone();
        egui::ComboBox::from_id_salt("hotkey-combo")
            .selected_text(human_hotkey(&current))
            .show_ui(ui, |ui| {
                for name in HotkeyName::all_names() {
                    if ui
                        .selectable_value(&mut current, name.to_string(), human_hotkey(name))
                        .changed()
                        || ui.button("").clicked()
                    // no-op, satisfies clippy
                    {}
                }
            });
        if current != config.dictation.hotkey {
            config.dictation.hotkey = current.clone();
            let _ = config.save();
            self.hotkey_changed = Some(current);
        }
        ui.end_row();
    }

    fn row_mode(&self, ui: &mut egui::Ui, config: &mut Config) {
        ui.label("Modus:");
        let mut current = config.dictation.mode.clone();
        egui::ComboBox::from_id_salt("mode-combo")
            .selected_text(label_for(MODES, &current))
            .show_ui(ui, |ui| {
                for (value, label) in MODES {
                    ui.selectable_value(&mut current, value.to_string(), *label);
                }
            });
        if current != config.dictation.mode {
            config.dictation.mode = current;
            let _ = config.save();
        }
        ui.end_row();
    }

    fn row_model(&mut self, ui: &mut egui::Ui, config: &mut Config) {
        ui.label("Modell:");
        ui.horizontal(|ui| {
            let mm = ModelManager::new(Config::models_dir(config));
            let installed: std::collections::HashSet<&'static str> =
                mm.list_downloaded().into_iter().collect();
            let mut current = config.dictation.model.clone();

            egui::ComboBox::from_id_salt("dictation-model")
                .selected_text(label_for_model(&current, &installed))
                .show_ui(ui, |ui| {
                    for name in available_models().keys() {
                        let label = label_for_model(name, &installed);
                        ui.selectable_value(&mut current, (*name).to_string(), label);
                    }
                });
            if current != config.dictation.model {
                config.dictation.model = current;
                let _ = config.save();
            }

            let busy = self.is_download_busy();
            let btn = egui::Button::new("Herunterladen…");
            if ui.add_enabled(!busy, btn).clicked() {
                self.start_download(config, ui.ctx());
            }
        });
        ui.end_row();
    }

    fn is_download_busy(&self) -> bool {
        match &*self.download_job.lock().unwrap() {
            Some(j) => matches!(
                j.phase,
                DownloadPhase::Downloading { .. } | DownloadPhase::Extracting
            ),
            None => false,
        }
    }

    fn start_download(&mut self, config: &Config, ctx: &egui::Context) {
        let mm = ModelManager::new(Config::models_dir(config));
        let missing: Vec<&'static str> = available_models()
            .keys()
            .copied()
            .filter(|n| !mm.is_downloaded(n))
            .collect();
        let Some(name) = missing.first().copied() else {
            self.download_message = Some("Alle Modelle sind installiert.".into());
            return;
        };
        let est_total = available_models()
            .get(name)
            .map(|s| (s.size_mb as u64) * 1024 * 1024)
            .unwrap_or(0);

        self.download_message = None;
        {
            let mut g = self.download_job.lock().unwrap();
            *g = Some(DownloadJob {
                model: name.to_string(),
                phase: DownloadPhase::Downloading {
                    downloaded: 0,
                    total: est_total,
                },
            });
        }

        let job = Arc::clone(&self.download_job);
        let ctx_for_thread = ctx.clone();
        let models_dir = Config::models_dir(config);
        let model_name = name.to_string();

        std::thread::spawn(move || {
            let mm = ModelManager::new(models_dir);
            let job_cb = Arc::clone(&job);
            let ctx_cb = ctx_for_thread.clone();
            let model_for_cb = model_name.clone();
            let progress = Box::new(move |downloaded: u64, total: u64| {
                let mut g = job_cb.lock().unwrap();
                if let Some(j) = g.as_mut() {
                    if downloaded >= total && total > 0 {
                        // bytes streamed; for sherpa, extraction phase follows
                        j.phase = DownloadPhase::Extracting;
                    } else {
                        j.phase = DownloadPhase::Downloading { downloaded, total };
                    }
                } else {
                    *g = Some(DownloadJob {
                        model: model_for_cb.clone(),
                        phase: DownloadPhase::Downloading { downloaded, total },
                    });
                }
                drop(g);
                ctx_cb.request_repaint();
            });

            let result = mm.download(&model_name, Some(progress));
            let mut g = job.lock().unwrap();
            if let Some(j) = g.as_mut() {
                j.phase = match result {
                    Ok(_) => DownloadPhase::Done,
                    Err(e) => DownloadPhase::Error(e.to_string()),
                };
            }
            drop(g);
            ctx_for_thread.request_repaint();
        });

        ctx.request_repaint();
    }

    fn render_download_status(&mut self, ui: &mut egui::Ui) {
        let snapshot: Option<(String, DownloadPhase)> = self
            .download_job
            .lock()
            .unwrap()
            .as_ref()
            .map(|j| (j.model.clone(), j.phase.clone()));

        let Some((model, phase)) = snapshot else {
            return;
        };

        ui.add_space(8.0);
        match phase {
            DownloadPhase::Downloading { downloaded, total } => {
                let frac = if total > 0 {
                    (downloaded as f32 / total as f32).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                ui.label(
                    egui::RichText::new(format!(
                        "Lade {model} … {} / {}",
                        fmt_bytes(downloaded),
                        fmt_bytes(total),
                    ))
                    .color(styles::TEXT_SECONDARY),
                );
                ui.add(
                    egui::ProgressBar::new(frac)
                        .show_percentage()
                        .desired_width(360.0),
                );
                ui.ctx().request_repaint();
            }
            DownloadPhase::Extracting => {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(
                        egui::RichText::new(format!("Extrahiere {model} …"))
                            .color(styles::TEXT_SECONDARY),
                    );
                });
                ui.ctx().request_repaint();
            }
            DownloadPhase::Done => {
                ui.label(
                    egui::RichText::new(format!("Modell {model} installiert."))
                        .color(styles::ACCENT_GREEN),
                );
                if ui.button("OK").clicked() {
                    *self.download_job.lock().unwrap() = None;
                }
            }
            DownloadPhase::Error(msg) => {
                ui.label(
                    egui::RichText::new(format!("Download fehlgeschlagen ({model}): {msg}"))
                        .color(styles::ACCENT_RED),
                );
                if ui.button("Schließen").clicked() {
                    *self.download_job.lock().unwrap() = None;
                }
            }
        }
    }

    fn row_language(&self, ui: &mut egui::Ui, config: &mut Config) {
        ui.label("Sprache:");
        let mut current = config.dictation.language.clone();
        egui::ComboBox::from_id_salt("language-combo")
            .selected_text(label_for(LANGUAGES, &current))
            .show_ui(ui, |ui| {
                for (value, label) in LANGUAGES {
                    ui.selectable_value(&mut current, value.to_string(), *label);
                }
            });
        if current != config.dictation.language {
            config.dictation.language = current;
            let _ = config.save();
        }
        ui.end_row();
    }

    fn row_save_audio(&self, ui: &mut egui::Ui, config: &mut Config) {
        ui.label("");
        let mut v = config.meeting.save_audio;
        if ui
            .checkbox(&mut v, "Meeting-Audio als .wav speichern")
            .changed()
        {
            config.meeting.save_audio = v;
            let _ = config.save();
        }
        ui.end_row();
    }

    fn row_provider(&self, ui: &mut egui::Ui, config: &mut Config) {
        ui.label("Provider:");
        let mut current = config.ai_postprocess.provider.clone();
        egui::ComboBox::from_id_salt("provider-combo")
            .selected_text(label_for(PROVIDERS, &current))
            .show_ui(ui, |ui| {
                for (value, label) in PROVIDERS {
                    ui.selectable_value(&mut current, value.to_string(), *label);
                }
            });
        if current != config.ai_postprocess.provider {
            config.ai_postprocess.provider = current;
            let _ = config.save();
        }
        ui.end_row();
    }

    fn row_api_key(&self, ui: &mut egui::Ui, config: &mut Config) {
        ui.label("API-Key:");
        let mut text = config.ai_postprocess.api_key.clone();
        let resp = ui.add(
            egui::TextEdit::singleline(&mut text)
                .password(true)
                .hint_text("sk-or-…"),
        );
        if resp.changed() {
            config.ai_postprocess.api_key = text;
            let _ = config.save();
        }
        ui.end_row();
    }

    fn row_ai_model(&mut self, ui: &mut egui::Ui, config: &mut Config) {
        ui.label("Modell:");
        let provider = config.ai_postprocess.provider.clone();
        let hint = match provider.as_str() {
            "lm_studio" => "z. B. lmstudio-community/Llama-3.1-8B-Instruct-GGUF",
            "ollama" => "z. B. llama3.1:8b",
            _ => "anthropic/claude-sonnet-4.5",
        };
        ui.horizontal(|ui| {
            let mut text = config.ai_postprocess.model.clone();
            let resp = ui.add(
                egui::TextEdit::singleline(&mut text)
                    .hint_text(hint)
                    .desired_width(280.0),
            );
            if resp.changed() {
                config.ai_postprocess.model = text;
                let _ = config.save();
            }

            if provider == "openrouter" && ui.button("Modelle laden").clicked() {
                self.fetch_openrouter_models();
            }
        });
        if provider == "openrouter" && !self.available_openrouter_models.is_empty() {
            ui.end_row();
            ui.label("");
            egui::ComboBox::from_id_salt("openrouter-models")
                .selected_text("Aus Liste wählen…")
                .show_ui(ui, |ui| {
                    let mut chosen: Option<String> = None;
                    for id in &self.available_openrouter_models {
                        if ui.selectable_label(false, id).clicked() {
                            chosen = Some(id.clone());
                        }
                    }
                    if let Some(id) = chosen {
                        config.ai_postprocess.model = id;
                        let _ = config.save();
                    }
                });
        }
        ui.end_row();
    }

    fn row_default_prompt(&self, ui: &mut egui::Ui, config: &mut Config) {
        ui.label("Standard-Prompt:");
        let mut current = config.ai_postprocess.default_prompt.clone();
        egui::ComboBox::from_id_salt("default-prompt-combo")
            .selected_text(preset_label(&current))
            .show_ui(ui, |ui| {
                for preset in PRESETS {
                    ui.selectable_value(&mut current, preset.key.to_string(), preset.label);
                }
            });
        if current != config.ai_postprocess.default_prompt {
            config.ai_postprocess.default_prompt = current;
            let _ = config.save();
        }
        ui.end_row();
    }

    fn row_dictation_autorun(&self, ui: &mut egui::Ui, config: &mut Config) {
        ui.label("");
        let mut v = config.ai_postprocess.dictation_autorun;
        if ui
            .checkbox(&mut v, "Diktate automatisch nachbearbeiten (vor Paste)")
            .changed()
        {
            config.ai_postprocess.dictation_autorun = v;
            let _ = config.save();
        }
        ui.end_row();
    }

    fn row_dictation_preset(&self, ui: &mut egui::Ui, config: &mut Config) {
        ui.label("Diktat-Prompt:");
        let mut current = config.ai_postprocess.dictation_preset.clone();
        egui::ComboBox::from_id_salt("dictation-preset-combo")
            .selected_text(preset_label(&current))
            .show_ui(ui, |ui| {
                for preset in PRESETS {
                    if preset.key == "custom" {
                        continue;
                    }
                    ui.selectable_value(&mut current, preset.key.to_string(), preset.label);
                }
            });
        if current != config.ai_postprocess.dictation_preset {
            config.ai_postprocess.dictation_preset = current;
            let _ = config.save();
        }
        ui.end_row();
    }

    fn fetch_openrouter_models(&mut self) {
        match crate::postprocess::llm::fetch_openrouter_models() {
            Ok(models) => {
                self.available_openrouter_models = models.into_iter().map(|m| m.id).collect();
                self.fetch_message = Some(format!(
                    "{} Modelle geladen.",
                    self.available_openrouter_models.len()
                ));
            }
            Err(e) => {
                self.fetch_message = Some(format!("Modelle laden fehlgeschlagen: {e}"));
            }
        }
    }
}

impl Default for SettingsPage {
    fn default() -> Self {
        Self::new()
    }
}

fn label_for<'a>(table: &'a [(&'a str, &'a str)], value: &str) -> &'a str {
    for (k, v) in table {
        if *k == value {
            return v;
        }
    }
    "—"
}

fn label_for_model(name: &str, installed: &std::collections::HashSet<&'static str>) -> String {
    let spec = match available_models().get(name) {
        Some(s) => s,
        None => return name.to_string(),
    };
    if installed.contains(name) {
        spec.display_name.to_string()
    } else {
        format!("{}  (nicht installiert)", spec.display_name)
    }
}

fn human_hotkey(raw: &str) -> String {
    raw.replace('_', " ")
        .split_whitespace()
        .map(|w| {
            let mut chars = w.chars();
            match chars.next() {
                Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn preset_label(key: &str) -> &'static str {
    PRESETS
        .iter()
        .find(|p| p.key == key)
        .map(|p| p.label)
        .unwrap_or("—")
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

fn rfd_pick_folder(_current: &str) -> Option<std::path::PathBuf> {
    // We do not pull in `rfd` to keep the dep tree small. Returning None
    // means the data-dir picker is a no-op for now — the user can still
    // edit data_dir directly in config.json.
    None
}
