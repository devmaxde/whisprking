//! "Einstellungen" — one card per topic, one row per setting.
//!
//! Every change is written to `config.json` immediately (there is no Save
//! button), so each row hands the caller a `&mut Config` and persists when
//! the widget reports a finished edit.

mod ai;
mod models;

use crate::config::Config;
use crate::hotkey::listener::HotkeyName;
use crate::transcription::model_manager::{available_models, ModelManager, DIARIZE_SIZE_MB};
use crate::ui::icons::Icon;
use crate::ui::theme::{self, space, text};
use crate::ui::widgets::{self as w, ButtonKind, Tone};

use models::DownloadJob;

pub(super) const LANGUAGES: &[(&str, &str)] = &[
    ("auto", "Automatisch"),
    ("de", "Deutsch"),
    ("en", "English"),
    ("fr", "Français"),
    ("es", "Español"),
    ("it", "Italiano"),
];

/// Same list as [`LANGUAGES`] plus the meeting-only default: follow whatever
/// dictation is set to. See [`crate::config::MeetingConfig::language`].
const MEETING_LANGUAGES: &[(&str, &str)] = &[
    ("", "Wie Diktat"),
    ("auto", "Automatisch"),
    ("de", "Deutsch"),
    ("en", "English"),
    ("fr", "Français"),
    ("es", "Español"),
    ("it", "Italiano"),
];

pub struct SettingsPage {
    pub hotkey_changed: Option<String>,
    pub data_dir_changed: bool,
    pub(super) download: DownloadJob,
    pub(super) openrouter_models: Vec<String>,
    pub(super) fetch_message: Option<String>,
    /// Preset whose prompt is open in the editor.
    pub(super) prompt_key: String,
    /// The text being edited, and which preset it belongs to. Held here
    /// rather than read from the config every frame so switching presets can
    /// fall back to the built-in prompt without writing it to disk first.
    pub(super) prompt_draft: String,
    pub(super) prompt_draft_for: Option<String>,
}

impl SettingsPage {
    pub fn new() -> Self {
        Self {
            hotkey_changed: None,
            data_dir_changed: false,
            download: DownloadJob::default(),
            openrouter_models: Vec::new(),
            fetch_message: None,
            prompt_key: "cleanup".into(),
            prompt_draft: String::new(),
            prompt_draft_for: None,
        }
    }

    /// Reset one-shot signal flags. Call after the parent consumed them.
    pub fn clear_signals(&mut self) {
        self.hotkey_changed = None;
        self.data_dir_changed = false;
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, config: &mut Config) {
        w::page_header(
            ui,
            "Einstellungen",
            "Änderungen werden sofort gespeichert.",
        );
        ui.add_space(space::LG);

        egui::ScrollArea::vertical()
            .id_salt("settings-scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.set_width(ui.available_width() - space::MD);

                self.dictation_card(ui, config);
                ui.add_space(space::MD);
                self.meeting_card(ui, config);
                ui.add_space(space::MD);
                self.post_transcribe_card(ui, config);
                ui.add_space(space::MD);
                self.ai_card(ui, config);
                ui.add_space(space::MD);
                self.prompts_card(ui, config);
                ui.add_space(space::MD);
                self.data_card(ui, config);
                ui.add_space(space::XL);
            });
    }

    // --- Diktat -----------------------------------------------------------

    fn dictation_card(&mut self, ui: &mut egui::Ui, config: &mut Config) {
        w::card_titled(
            ui,
            Some(Icon::Mic),
            "Diktat",
            |_| {},
            |ui| {
                w::setting_row(
                    ui,
                    "Hotkey",
                    "Halten (oder drücken) zum Diktieren — wirkt sofort.",
                    |ui| {
                        let options: Vec<(String, String)> = HotkeyName::all_names()
                            .into_iter()
                            .map(|n| (n.to_string(), human_hotkey(n)))
                            .collect();
                        let label = human_hotkey(&config.dictation.hotkey);
                        if w::combo(
                            ui,
                            "hotkey",
                            170.0,
                            &mut config.dictation.hotkey,
                            &options,
                            &label,
                        ) {
                            let _ = config.save();
                            self.hotkey_changed = Some(config.dictation.hotkey.clone());
                        }
                    },
                );
                w::divider(ui);

                w::setting_row(ui, "Modus", "Gedrückt halten oder umschalten.", |ui| {
                    let mut mode = config.dictation.mode.clone();
                    let changed = w::segmented(
                        ui,
                        "dictation-mode",
                        &mut mode,
                        &[
                            ("hold_to_talk".to_string(), "Halten", true),
                            ("toggle".to_string(), "Umschalten", true),
                        ],
                    );
                    if changed {
                        config.dictation.mode = mode;
                        let _ = config.save();
                    }
                });
                w::divider(ui);

                w::setting_row(
                    ui,
                    "Sprache",
                    "Feste Sprache erkennt zuverlässiger als „Automatisch“.",
                    |ui| {
                        if language_combo(
                            ui,
                            "dictation-language",
                            &mut config.dictation.language,
                            LANGUAGES,
                        ) {
                            let _ = config.save();
                        }
                    },
                );
                w::divider(ui);

                self.model_row(ui, config);
            },
        );
    }

    // --- Meeting ----------------------------------------------------------

    fn meeting_card(&mut self, ui: &mut egui::Ui, config: &mut Config) {
        w::card_titled(
            ui,
            Some(Icon::Waveform),
            "Meeting",
            |_| {},
            |ui| {
                w::setting_row(
                    ui,
                    "Meeting-Sprache",
                    "Eine feste Sprache verhindert, dass Whisper mitten im Gespräch \
                     die Sprache wechselt oder übersetzt.",
                    |ui| {
                        if language_combo(
                            ui,
                            "meeting-language",
                            &mut config.meeting.language,
                            MEETING_LANGUAGES,
                        ) {
                            let _ = config.save();
                        }
                    },
                );
                w::divider(ui);

                let mut label_speakers = config.meeting.label_speakers;
                if w::toggle_row(
                    ui,
                    "Sprecher benennen",
                    "Jede Zeile mit „Du“ oder „Andere“ kennzeichnen.",
                    &mut label_speakers,
                ) {
                    config.meeting.label_speakers = label_speakers;
                    let _ = config.save();
                }
                w::divider(ui);

                self.diarize_rows(ui, config);
                w::divider(ui);

                let mut save_audio = config.meeting.save_audio;
                if w::toggle_row(
                    ui,
                    "Audio mitschneiden",
                    "Beide Tonspuren zusätzlich als .wav speichern. Die Nachbearbeitung \
                     schaltet das ohnehin ein — sie braucht die Aufnahme.",
                    &mut save_audio,
                ) {
                    config.meeting.save_audio = save_audio;
                    let _ = config.save();
                }
            },
        );
    }

    /// Telling the individual voices inside a track apart.
    ///
    /// Sits under "Sprecher benennen" because it is the finer version of the
    /// same thing: without it a line says which *side* spoke, with it who.
    /// Switched off when the lines carry no name at all, which is what the
    /// row above decides.
    fn diarize_rows(&mut self, ui: &mut egui::Ui, config: &mut Config) {
        let installed = ModelManager::new(Config::models_dir(config)).is_diarize_downloaded();

        let mut enabled = config.meeting.diarize.enabled;
        if w::toggle_row(
            ui,
            "Sprecher unterscheiden",
            "Nach dem Meeting wird die Aufnahme in einzelne Stimmen zerlegt: \
             statt „Andere“ steht dann „Person 1“, „Person 2“ … in jeder Zeile. \
             Läuft nur nachträglich — wer „Person 2“ ist, steht erst fest, wenn \
             die ganze Aufnahme gehört wurde.",
            &mut enabled,
        ) {
            config.meeting.diarize.enabled = enabled;
            let _ = config.save();
        }

        if !enabled || !config.meeting.label_speakers {
            if enabled {
                ui.add_space(space::XS);
                w::hint(
                    ui,
                    "Ohne „Sprecher benennen“ trägt keine Zeile einen Namen — \
                     die Unterscheidung bliebe unsichtbar und läuft deshalb nicht.",
                );
            }
            return;
        }

        ui.add_space(space::SM);
        w::setting_row(
            ui,
            "Anzahl Personen",
            "„Automatisch“ schätzt sie aus der Aufnahme. Eine feste Zahl ist \
             zuverlässiger, wenn man sie kennt.",
            |ui| {
                let options: Vec<(u32, String)> = std::iter::once((0, "Automatisch".to_string()))
                    .chain((2..=8).map(|n| (n, format!("{n} Personen"))))
                    .collect();
                let current = config.meeting.diarize.speakers;
                let label = options
                    .iter()
                    .find(|(n, _)| *n == current)
                    .map(|(_, l)| l.clone())
                    .unwrap_or_else(|| format!("{current} Personen"));
                if w::combo(
                    ui,
                    "diarize-speakers",
                    170.0,
                    &mut config.meeting.diarize.speakers,
                    &options,
                    &label,
                ) {
                    let _ = config.save();
                }
            },
        );

        if !config.meeting.post_transcribe.enabled {
            ui.add_space(space::XS);
            w::hint(
                ui,
                "Für Meetings passiert das in der Nachbearbeitung, und die ist \
                 gerade aus — importierte Aufnahmen bekommen trotzdem Namen.",
            );
        }

        ui.add_space(space::XS);
        if installed {
            w::badge(ui, "Sprechermodelle installiert", Tone::Success);
        } else {
            w::hint(
                ui,
                &format!(
                    "Die beiden Sprechermodelle (~{DIARIZE_SIZE_MB} MB) werden beim \
                     ersten Lauf automatisch geladen.",
                ),
            );
        }
    }

    // --- Nachbearbeitung ---------------------------------------------------

    /// The second pass over a finished recording.
    ///
    /// Its own card rather than a row in "Meeting", because the trade-off it
    /// offers is the opposite one: the live transcript is bounded by having to
    /// keep up, this is bounded by nothing at all.
    fn post_transcribe_card(&mut self, ui: &mut egui::Ui, config: &mut Config) {
        let manager = ModelManager::new(Config::models_dir(config));
        let installed = manager.list_downloaded();

        w::card_titled(
            ui,
            Some(Icon::Sparkle),
            "Nachbearbeitung",
            |_| {},
            |ui| {
                let mut enabled = config.meeting.post_transcribe.enabled;
                if w::toggle_row(
                    ui,
                    "Nach dem Meeting erneut transkribieren",
                    "Die Aufnahme wird gespeichert und danach noch einmal transkribiert — \
                     in möglichst großen Abschnitten statt in Echtzeit-Häppchen, damit kein \
                     Satz an einer Fenstergrenze zerfällt.",
                    &mut enabled,
                ) {
                    config.meeting.post_transcribe.enabled = enabled;
                    let _ = config.save();
                }

                if !enabled {
                    return;
                }
                w::divider(ui);

                w::text_line(ui, "Modelle", text::body(), theme::TEXT_PRIMARY);
                ui.add_space(space::XS);
                w::hint(
                    ui,
                    "Alle ausgewählten Modelle laufen gleichzeitig über die ganze Aufnahme. \
                     Das erste in der Liste gilt als Vorgabe, falls sich die Ergebnisse \
                     nicht zusammenführen lassen.",
                );
                ui.add_space(space::SM);

                // Biggest first: the point of this pass is accuracy, and the
                // order here is also the preference order.
                let mut names: Vec<&'static str> = available_models().keys().copied().collect();
                names.sort_by_key(|n| {
                    let size = available_models().get(n).map(|s| s.size_mb).unwrap_or(0);
                    (std::cmp::Reverse(size), *n)
                });

                for name in names {
                    let mut on = config
                        .meeting
                        .post_transcribe
                        .models
                        .iter()
                        .any(|m| m == name);
                    let ready = installed.contains(&name);
                    let size = available_models().get(name).map(|s| s.size_mb).unwrap_or(0);
                    let hint = if ready {
                        "installiert".to_string()
                    } else {
                        format!("{size} MB · nicht installiert — wird übersprungen")
                    };
                    if w::toggle_row(ui, name, &hint, &mut on) {
                        let models = &mut config.meeting.post_transcribe.models;
                        match on {
                            true => models.push(name.to_string()),
                            false => models.retain(|m| m != name),
                        }
                        let _ = config.save();
                    }
                }

                w::divider(ui);
                let mut reconcile = config.meeting.post_transcribe.reconcile;
                if w::toggle_row(
                    ui,
                    "Varianten zusammenführen",
                    "Die Ergebnisse der Modelle gehen gemeinsam an die KI, die daraus \
                     ein Transkript macht. Ohne KI-Provider wird die beste Einzelvariante \
                     übernommen.",
                    &mut reconcile,
                ) {
                    config.meeting.post_transcribe.reconcile = reconcile;
                    let _ = config.save();
                }

                w::divider(ui);
                let mut keep_audio = config.meeting.post_transcribe.keep_audio;
                if w::toggle_row(
                    ui,
                    "Aufnahme behalten",
                    "Rund 115 MB pro Stunde und Tonspur. Nur mit behaltener Aufnahme lässt \
                     sich die Nachbearbeitung später erneut starten.",
                    &mut keep_audio,
                ) {
                    config.meeting.post_transcribe.keep_audio = keep_audio;
                    let _ = config.save();
                }
            },
        );
    }

    // --- Daten ------------------------------------------------------------

    fn data_card(&mut self, ui: &mut egui::Ui, config: &mut Config) {
        let dir = Config::resolve_data_dir(&config.data_dir);
        let mut pick = false;
        let mut open = false;

        w::card_titled(
            ui,
            Some(Icon::Folder),
            "Daten",
            |_| {},
            |ui| {
                w::setting_row(
                    ui,
                    "Datenordner",
                    &dir.to_string_lossy(),
                    |ui| {
                        open = w::icon_button(ui, Icon::Reveal, "Ordner öffnen").clicked();
                        pick = w::button(ui, ButtonKind::Secondary, None, "Ändern …").clicked();
                    },
                );
                ui.add_space(space::SM);
                w::hint(
                    ui,
                    "Transkripte, KI-Dokumente, Modelle und die Konfiguration liegen hier.",
                );
            },
        );

        if open {
            let _ = opener::open(&dir);
        }
        if pick {
            if let Some(new_dir) = rfd::FileDialog::new()
                .set_directory(&dir)
                .set_title("Datenordner wählen")
                .pick_folder()
            {
                config.data_dir = new_dir.to_string_lossy().into_owned();
                let _ = Config::ensure_dirs(&config.data_dir);
                let _ = config.save();
                self.data_dir_changed = true;
            }
        }
    }
}

impl Default for SettingsPage {
    fn default() -> Self {
        Self::new()
    }
}

/// Language picker shared by the dictation and meeting rows.
fn language_combo(
    ui: &mut egui::Ui,
    id_salt: &str,
    value: &mut String,
    table: &[(&str, &str)],
) -> bool {
    let options: Vec<(String, String)> = table
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    let label = label_for(table, value).to_string();
    w::combo(ui, id_salt, 170.0, value, &options, &label)
}

pub(super) fn label_for<'a>(table: &'a [(&'a str, &'a str)], value: &str) -> &'a str {
    table
        .iter()
        .find(|(k, _)| *k == value)
        .map(|(_, v)| *v)
        .unwrap_or("—")
}

/// `right_cmd` → `Right Cmd`
fn human_hotkey(raw: &str) -> String {
    raw.replace('_', " ")
        .split_whitespace()
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Small status line used by the model and provider sections.
pub(super) fn status_line(ui: &mut egui::Ui, message: &str) {
    w::text_line(ui, message, text::small(), theme::TEXT_MUTED);
}
