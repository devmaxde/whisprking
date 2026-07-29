//! The "KI-Nachbearbeitung" card: which provider runs the presets, and what
//! happens automatically.

use crate::config::Config;
use crate::postprocess::llm::{editable_presets, PRESETS};
use crate::ui::icons::Icon;
use crate::ui::theme::space;
use crate::ui::widgets::{self as w, ButtonKind, Tone};

use super::{label_for, status_line, SettingsPage};

const PROVIDERS: &[(&str, &str)] = &[
    ("none", "Aus"),
    ("openrouter", "OpenRouter"),
    ("lm_studio", "LM Studio (lokal)"),
    ("ollama", "Ollama (lokal)"),
];

fn provider_uses_api_key(provider: &str) -> bool {
    matches!(provider, "openrouter")
}

fn model_hint(provider: &str) -> &'static str {
    match provider {
        "lm_studio" => "z. B. lmstudio-community/Llama-3.1-8B-Instruct-GGUF",
        "ollama" => "z. B. llama3.1:8b",
        _ => "z. B. anthropic/claude-sonnet-4.5",
    }
}

impl SettingsPage {
    pub(super) fn ai_card(&mut self, ui: &mut egui::Ui, config: &mut Config) {
        let provider = config.ai_postprocess.provider.clone();
        let active = provider != "none";

        w::card_titled(
            ui,
            Some(Icon::Sparkle),
            "KI-Nachbearbeitung",
            |ui| {
                if active {
                    w::badge(ui, "Aktiv", Tone::Success);
                }
            },
            |ui| {
                w::setting_row(
                    ui,
                    "Provider",
                    "Bereinigung, Zusammenfassung und Action Items laufen hierüber.",
                    |ui| {
                        let options: Vec<(String, String)> = PROVIDERS
                            .iter()
                            .map(|(k, v)| (k.to_string(), v.to_string()))
                            .collect();
                        let label = label_for(PROVIDERS, &provider).to_string();
                        if w::combo(
                            ui,
                            "ai-provider",
                            190.0,
                            &mut config.ai_postprocess.provider,
                            &options,
                            &label,
                        ) {
                            let _ = config.save();
                        }
                    },
                );

                if !active {
                    ui.add_space(space::SM);
                    w::hint(
                        ui,
                        "Ohne Provider bleiben Transkripte unverändert — die KI-Dokumente \
                         im Verlauf lassen sich dann nicht erzeugen.",
                    );
                    return;
                }

                if provider_uses_api_key(&provider) {
                    w::divider(ui);
                    w::setting_row(ui, "API-Key", "Wird lokal in config.json abgelegt.", |ui| {
                        let mut key = config.ai_postprocess.api_key.clone();
                        if w::text_field(ui, "ai-key", &mut key, "sk-or-…", 260.0, true).changed() {
                            config.ai_postprocess.api_key = key;
                            let _ = config.save();
                        }
                    });
                }

                w::divider(ui);
                self.model_field(ui, config, &provider);

                w::divider(ui);
                w::setting_row(
                    ui,
                    "Standard-Prompt",
                    "Vorauswahl für die KI-Aktion im Verlauf.",
                    |ui| {
                        if preset_combo(
                            ui,
                            "ai-default-prompt",
                            &mut config.ai_postprocess.default_prompt,
                            true,
                        ) {
                            let _ = config.save();
                        }
                    },
                );

                w::divider(ui);
                let mut autorun = config.ai_postprocess.dictation_autorun;
                if w::toggle_row(
                    ui,
                    "Diktate automatisch nachbearbeiten",
                    "Vor dem Einfügen durch das Modell schicken — kostet ein paar Sekunden.",
                    &mut autorun,
                ) {
                    config.ai_postprocess.dictation_autorun = autorun;
                    let _ = config.save();
                }

                if autorun {
                    w::divider(ui);
                    w::setting_row(ui, "Diktat-Prompt", "", |ui| {
                        if preset_combo(
                            ui,
                            "ai-dictation-prompt",
                            &mut config.ai_postprocess.dictation_preset,
                            false,
                        ) {
                            let _ = config.save();
                        }
                    });
                }
            },
        );
    }

    /// The prompt editor. Every preset's system prompt is editable and each
    /// edit is stored per preset (`ai_postprocess.prompts`), so "Bereinigung"
    /// can be tuned to the user's own vocabulary without touching the others
    /// — and reset back to the shipped text at any time.
    pub(super) fn prompts_card(&mut self, ui: &mut egui::Ui, config: &mut Config) {
        // Load the draft on first show and whenever the preset changes. The
        // built-in text is only ever copied into the draft, never written to
        // the config, so an untouched preset stays on the shipped prompt.
        if self.prompt_draft_for.as_deref() != Some(self.prompt_key.as_str()) {
            self.prompt_draft = effective_prompt(config, &self.prompt_key);
            self.prompt_draft_for = Some(self.prompt_key.clone());
        }

        let customised = is_customised(config, &self.prompt_key);
        let is_custom = self.prompt_key == "custom";
        let mut reset = false;

        w::card_titled(
            ui,
            Some(Icon::Transcript),
            "Prompts",
            |ui| {
                if customised {
                    w::badge(ui, "Angepasst", Tone::Accent);
                }
            },
            |ui| {
                w::setting_row(
                    ui,
                    "Prompt",
                    "Legt fest, was die KI mit dem Transkript macht.",
                    |ui| {
                        let options: Vec<(String, String)> = editable_presets()
                            .iter()
                            .map(|p| (p.key.to_string(), p.label.to_string()))
                            .collect();
                        let label = editable_presets()
                            .iter()
                            .find(|p| p.key == self.prompt_key)
                            .map(|p| p.label)
                            .unwrap_or("Bereinigung")
                            .to_string();
                        w::combo(
                            ui,
                            "prompt-preset",
                            200.0,
                            &mut self.prompt_key,
                            &options,
                            &label,
                        );
                    },
                );

                ui.add_space(space::SM);
                if w::text_area(ui, "prompt-editor", &mut self.prompt_draft, 9).changed() {
                    store_prompt(config, &self.prompt_key, &self.prompt_draft);
                    let _ = config.save();
                }

                ui.add_space(space::SM);
                let mut clicked = false;
                egui::Sides::new().show(
                    ui,
                    |ui| {
                        let note = if is_custom && self.prompt_draft.trim().is_empty() {
                            "Solange dieser Prompt leer ist, wird die Bereinigung verwendet."
                        } else {
                            "Der Text ist die System-Anweisung; das Transkript wird als \
                             Nachricht darunter geschickt."
                        };
                        w::hint(ui, note);
                    },
                    |ui| {
                        clicked = ui
                            .add_enabled_ui(customised, |ui| {
                                w::button(
                                    ui,
                                    ButtonKind::Secondary,
                                    None,
                                    "Auf Standard zurücksetzen",
                                )
                            })
                            .inner
                            .clicked();
                    },
                );
                reset = clicked;
            },
        );

        if reset {
            clear_prompt(config, &self.prompt_key);
            let _ = config.save();
            // Force a reload of the draft from the built-in text next frame.
            self.prompt_draft_for = None;
        }
    }

    /// Model id, plus the OpenRouter catalog fetch.
    fn model_field(&mut self, ui: &mut egui::Ui, config: &mut Config, provider: &str) {
        let is_openrouter = provider == "openrouter";
        let mut fetch = false;

        w::setting_row(ui, "Modell", model_hint(provider), |ui| {
            if is_openrouter {
                fetch = w::button(ui, ButtonKind::Secondary, None, "Modelle laden").clicked();
            }
            let mut model = config.ai_postprocess.model.clone();
            if w::text_field(ui, "ai-model", &mut model, model_hint(provider), 300.0, false)
                .changed()
            {
                config.ai_postprocess.model = model;
                let _ = config.save();
            }
        });

        if is_openrouter && !self.openrouter_models.is_empty() {
            ui.add_space(space::SM);
            let options: Vec<(String, String)> = self
                .openrouter_models
                .iter()
                .map(|id| (id.clone(), id.clone()))
                .collect();
            w::setting_row(ui, "Aus Katalog wählen", "", |ui| {
                let mut chosen = config.ai_postprocess.model.clone();
                if w::combo(ui, "ai-model-list", 320.0, &mut chosen, &options, "Modell wählen …") {
                    config.ai_postprocess.model = chosen;
                    let _ = config.save();
                }
            });
        }

        if let Some(message) = &self.fetch_message {
            ui.add_space(space::XS);
            status_line(ui, message);
        }

        if fetch {
            self.fetch_openrouter_models();
        }
    }

    fn fetch_openrouter_models(&mut self) {
        // Blocking on the UI thread for up to the client's 10 s timeout. Rare
        // and explicit (the user pressed a button), so it stays simple.
        match crate::postprocess::llm::fetch_openrouter_models() {
            Ok(models) => {
                self.openrouter_models = models.into_iter().map(|m| m.id).collect();
                self.fetch_message = Some(format!(
                    "{} Modelle geladen.",
                    self.openrouter_models.len()
                ));
            },
            Err(e) => {
                self.fetch_message = Some(format!("Modelle laden fehlgeschlagen: {e}"));
            },
        }
    }
}

// --- prompt storage --------------------------------------------------------
//
// The free-form preset predates the per-preset overrides and keeps its own
// config field, so every access goes through these four functions instead of
// spreading the `"custom"` special case over the UI.

fn builtin_prompt(key: &str) -> &'static str {
    editable_presets()
        .iter()
        .find(|p| p.key == key)
        .map(|p| p.system)
        .unwrap_or_default()
}

fn stored_prompt(config: &Config, key: &str) -> String {
    if key == "custom" {
        config.ai_postprocess.custom_prompt.clone()
    } else {
        config
            .ai_postprocess
            .prompts
            .get(key)
            .cloned()
            .unwrap_or_default()
    }
}

fn store_prompt(config: &mut Config, key: &str, text: &str) {
    if key == "custom" {
        config.ai_postprocess.custom_prompt = text.to_string();
    } else {
        config
            .ai_postprocess
            .prompts
            .insert(key.to_string(), text.to_string());
    }
}

fn clear_prompt(config: &mut Config, key: &str) {
    if key == "custom" {
        config.ai_postprocess.custom_prompt.clear();
    } else {
        config.ai_postprocess.prompts.remove(key);
    }
}

/// The prompt actually in force: the user's version when there is one,
/// otherwise the shipped text. Mirrors
/// [`crate::postprocess::llm::resolve_system_prompt`], which is what the run
/// itself uses — an empty override means "built-in" on both sides.
fn effective_prompt(config: &Config, key: &str) -> String {
    let stored = stored_prompt(config, key);
    if stored.trim().is_empty() {
        builtin_prompt(key).to_string()
    } else {
        stored
    }
}

/// Customised means *different from the built-in*, so re-typing the shipped
/// text does not leave a badge behind.
fn is_customised(config: &Config, key: &str) -> bool {
    let stored = stored_prompt(config, key);
    !stored.trim().is_empty() && stored.trim() != builtin_prompt(key).trim()
}

/// Preset picker. `with_custom` includes the free-form prompt entry.
fn preset_combo(ui: &mut egui::Ui, id_salt: &str, value: &mut String, with_custom: bool) -> bool {
    let options: Vec<(String, String)> = PRESETS
        .iter()
        .filter(|p| with_custom || p.key != "custom")
        .map(|p| (p.key.to_string(), p.label.to_string()))
        .collect();
    let label = PRESETS
        .iter()
        .find(|p| p.key == value.as_str())
        .map(|p| p.label)
        .unwrap_or("—")
        .to_string();
    w::combo(ui, id_salt, 240.0, value, &options, &label)
}
