//! Detail pane of the history page: the documents of one recording.
//!
//! Every document is a tab, every tab has its own copy button, and the body
//! is selectable text — "how do I copy the correction" has exactly one
//! answer here, and it is one click.

use std::sync::Arc;

use egui::{vec2, Sense};

use crate::config::Config;
use crate::output::transcript_doc::{DocKind, Recording, DERIVED_KINDS};
use crate::postprocess::llm::PRESETS;
use crate::transcription::post::remove_tracks;
use crate::ui::pages::meeting::post::existing_tracks;
use crate::ui::icons::Icon;
use crate::ui::theme::{self, radius, space, text};
use crate::ui::widgets::{self as w, ButtonKind, Tone};

use super::{reveal, HistoryPage};

impl HistoryPage {
    pub(super) fn detail_pane(&mut self, ui: &mut egui::Ui, config: &Config) {
        let Some(rec) = self.selected_recording().cloned() else {
            theme::card_frame().show(ui, |ui| {
                ui.set_min_height(ui.available_height());
                w::empty_state(
                    ui,
                    Icon::Transcript,
                    "Keine Aufnahme ausgewählt",
                    "Links eine Aufnahme anklicken.",
                );
            });
            return;
        };

        theme::card_frame().show(ui, |ui| {
            ui.set_min_height(ui.available_height());
            ui.set_width(ui.available_width());

            self.detail_header(ui, &rec);
            ui.add_space(space::MD);
            self.ai_row(ui, &rec, config);

            if let Some(path) = self.confirm_delete.clone() {
                if path == rec.path {
                    ui.add_space(space::MD);
                    self.delete_confirmation(ui, &rec, config);
                }
            }

            ui.add_space(space::MD);
            self.tabs(ui, &rec);
            ui.add_space(space::SM);
            self.document(ui, &rec, config);
        });
    }

    /// Title, date/duration, and the file-level actions.
    fn detail_header(&mut self, ui: &mut egui::Ui, rec: &Recording) {
        let named = rec.title != "Meeting";
        let title = if named {
            rec.title.clone()
        } else {
            rec.display_date()
        };
        let mut meta = Vec::new();
        if named {
            meta.push(rec.display_date());
        }
        if let Some(d) = &rec.duration {
            meta.push(format!("Dauer {d}"));
        }
        meta.push(
            rec.path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or_default()
                .to_string(),
        );

        let mut do_reveal = false;
        let mut do_delete = false;
        egui::Sides::new().height(38.0).show(
            ui,
            |ui| {
                ui.vertical(|ui| {
                    w::text_line(ui, &title, text::title(), theme::TEXT_PRIMARY);
                    ui.add_space(space::XXS);
                    w::text_line(
                        ui,
                        &meta.join(" · "),
                        text::small(),
                        theme::TEXT_MUTED,
                    );
                });
            },
            |ui| {
                do_delete = w::icon_button_danger(ui, Icon::Trash, "Aufnahme löschen").clicked();
                do_reveal = w::icon_button(ui, Icon::Reveal, "Im Finder zeigen").clicked();
            },
        );

        if do_reveal {
            reveal(&rec.path);
        }
        if do_delete {
            self.confirm_delete = Some(rec.path.clone());
        }
    }

    /// Preset picker + the button that produces a document.
    fn ai_row(&mut self, ui: &mut egui::Ui, rec: &Recording, config: &Config) {
        let provider_on = config.ai_postprocess.provider != "none";
        let running = self.job_running();
        let kind = DocKind::from_preset(&self.preset);
        let exists = rec.has_doc(kind);

        let mut start = false;
        theme::inset_frame()
            .inner_margin(egui::Margin::symmetric(space::MD as i8, space::SM as i8))
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                egui::Sides::new().height(theme::CONTROL_HEIGHT).show(
                    ui,
                    |ui| {
                        let (r, _) = ui.allocate_exact_size(vec2(15.0, 15.0), Sense::hover());
                        crate::ui::icons::paint(
                            ui.painter(),
                            Icon::Sparkle,
                            r,
                            theme::ACCENT_HOVER,
                        );
                        ui.add_space(space::XS);
                        w::text_line(
                            ui,
                            "KI-Nachbearbeitung",
                            text::strong(),
                            theme::TEXT_PRIMARY,
                        );
                        ui.add_space(space::SM);
                        let options: Vec<(String, String)> = PRESETS
                            .iter()
                            .map(|p| (p.key.to_string(), p.label.to_string()))
                            .collect();
                        let label = PRESETS
                            .iter()
                            .find(|p| p.key == self.preset)
                            .map(|p| p.label)
                            .unwrap_or("Bereinigung");
                        w::combo(ui, "history-preset", 180.0, &mut self.preset, &options, label);
                    },
                    |ui| {
                        if running {
                            w::busy_row(ui, "Läuft …");
                            return;
                        }
                        let label = if exists { "Neu erzeugen" } else { "Erzeugen" };
                        let resp = ui
                            .add_enabled_ui(provider_on, |ui| {
                                w::button(ui, ButtonKind::Primary, Some(Icon::Sparkle), label)
                            })
                            .inner;
                        let resp = if provider_on {
                            resp.on_hover_text(format!(
                                "Erzeugt „{}“ aus dem Transkript — als eigene Datei.",
                                kind.label()
                            ))
                        } else {
                            resp.on_disabled_hover_text(
                                "Erst in den Einstellungen einen KI-Provider aktivieren.",
                            )
                        };
                        start = resp.clicked();
                    },
                );
            });

        if start {
            let ctx = ui.ctx().clone();
            self.start_job(config, &ctx);
        }
    }

    fn delete_confirmation(&mut self, ui: &mut egui::Ui, rec: &Recording, config: &Config) {
        let mut confirm = false;
        let mut cancel = false;
        egui::Frame::new()
            .fill(theme::tint(theme::DANGER, 0.10))
            .stroke(egui::Stroke::new(1.0, theme::tint(theme::DANGER, 0.30)))
            .corner_radius(radius::MD)
            .inner_margin(egui::Margin::symmetric(space::MD as i8, space::SM as i8))
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                egui::Sides::new().height(theme::CONTROL_HEIGHT).show(
                    ui,
                    |ui| {
                        w::text_line(
                            ui,
                            "Aufnahme mit allen KI-Dokumenten löschen?",
                            text::body(),
                            theme::TEXT_PRIMARY,
                        );
                    },
                    |ui| {
                        confirm =
                            w::button(ui, ButtonKind::Danger, Some(Icon::Trash), "Löschen").clicked();
                        cancel = w::button(ui, ButtonKind::Ghost, None, "Abbrechen").clicked();
                    },
                );
            });

        if cancel {
            self.confirm_delete = None;
        }
        if confirm {
            // The kept audio lives in a different directory, so `delete()`
            // cannot reach it. Leaving it behind would quietly cost hundreds of
            // megabytes per deleted meeting.
            remove_tracks(&existing_tracks(&rec.path, config));
            match rec.delete() {
                Ok(()) => {
                    self.status = Some((Tone::Success, "Aufnahme gelöscht.".into()));
                    self.selected = None;
                },
                Err(e) => {
                    self.status = Some((Tone::Danger, format!("Löschen fehlgeschlagen: {e}")));
                },
            }
            self.confirm_delete = None;
            self.refresh(config);
        }
    }

    /// One tab per document; tabs without a document stay visible but muted
    /// so it is obvious what could exist.
    fn tabs(&mut self, ui: &mut egui::Ui, rec: &Recording) {
        let mut kinds = vec![DocKind::Transcript];
        for kind in DERIVED_KINDS {
            // Show every standard document; the custom one only once used.
            if kind != DocKind::Custom || rec.has_doc(kind) {
                kinds.push(kind);
            }
        }
        let labels: Vec<(&str, bool)> = kinds
            .iter()
            .map(|k| (k.label(), rec.has_doc(*k)))
            .collect();
        let selected = kinds.iter().position(|k| *k == self.tab).unwrap_or(0);
        if let Some(clicked) = w::tab_bar(ui, selected, &labels) {
            self.tab = kinds[clicked];
        }
    }

    /// Body of the active tab.
    fn document(&mut self, ui: &mut egui::Ui, rec: &Recording, config: &Config) {
        let kind = self.tab;
        self.ensure_doc(rec, kind);
        let Some((body, plain)) = self
            .doc_cache
            .as_ref()
            .map(|c| (Arc::clone(&c.body), Arc::clone(&c.plain)))
        else {
            self.missing_document(ui, kind, config);
            return;
        };

        // Toolbar: provenance on the left, copy actions on the right.
        egui::Sides::new().height(theme::CONTROL_HEIGHT).show(
            ui,
            |ui| {
                let info = match kind {
                    DocKind::Transcript => rec
                        .duration
                        .as_ref()
                        .map(|d| format!("Aufgenommen · {d}"))
                        .unwrap_or_else(|| "Transkript".into()),
                    _ => rec
                        .doc_created_at(kind)
                        .map(|t| format!("Erzeugt am {t}"))
                        .unwrap_or_else(|| kind.label().to_string()),
                };
                w::text_line(ui, &info, text::small(), theme::TEXT_MUTED);
            },
            |ui| {
                let for_copy = Arc::clone(&body);
                self.copy_doc
                    .show(ui, ButtonKind::Secondary, "Kopieren", move || {
                        for_copy.to_string()
                    });
                if kind == DocKind::Transcript {
                    // Same text without `**[MM:SS] Sprecher:**` scaffolding.
                    let for_copy = Arc::clone(&plain);
                    let _ = self.copy_plain.show(ui, ButtonKind::Ghost, "Nur Text", move || {
                        for_copy.to_string()
                    });
                }
            },
        );

        ui.add_space(space::SM);
        theme::inset_frame().show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.set_min_height(ui.available_height().max(120.0));
            egui::ScrollArea::vertical()
                .id_salt(kind.label())
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    if body.trim().is_empty() {
                        w::text_line(
                            ui,
                            "Dieses Dokument ist leer.",
                            text::body(),
                            theme::TEXT_MUTED,
                        );
                    } else if kind == DocKind::Transcript {
                        transcript_body(ui, &body);
                    } else {
                        w::selectable_body(ui, &body);
                    }
                });
        });
    }

    /// Placeholder for a document that has not been generated yet, with the
    /// action that would create it.
    ///
    /// Which action that is depends on the kind. Most documents come from
    /// running an LLM preset over the transcript; the post-transcription ones
    /// come from decoding the recording's audio again, which is only possible
    /// while that audio is still on disk.
    fn missing_document(&mut self, ui: &mut egui::Ui, kind: DocKind, config: &Config) {
        let provider_on = config.ai_postprocess.provider != "none";
        let by_llm = kind.is_llm_preset();
        let tracks = self
            .selected_path()
            .map(|p| existing_tracks(&p, config))
            .unwrap_or_default();
        let running = if by_llm {
            self.job_running()
        } else {
            self.post.running()
        };
        let mut generate = false;
        let mut rerun = false;

        theme::inset_frame().show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.set_min_height(ui.available_height().max(120.0));
            ui.vertical_centered(|ui| {
                ui.add_space(space::XL);
                w::text_line(ui, kind.label(), text::title(), theme::TEXT_SECONDARY);
                ui.add_space(space::XS);
                w::text_line(ui, kind.empty_hint(), text::small(), theme::TEXT_MUTED);
                ui.add_space(space::MD);
                if running {
                    w::busy_row(ui, "Wird erzeugt …");
                    return;
                }
                if by_llm {
                    generate = ui
                        .add_enabled_ui(provider_on, |ui| {
                            w::button(
                                ui,
                                ButtonKind::Primary,
                                Some(Icon::Sparkle),
                                &format!("{} erzeugen", kind.label()),
                            )
                        })
                        .inner
                        .on_disabled_hover_text(
                            "Erst in den Einstellungen einen KI-Provider aktivieren.",
                        )
                        .clicked();
                    return;
                }
                rerun = ui
                    .add_enabled_ui(!tracks.is_empty(), |ui| {
                        w::button(
                            ui,
                            ButtonKind::Primary,
                            Some(Icon::Waveform),
                            "Nachbearbeitung starten",
                        )
                    })
                    .inner
                    .on_hover_text(
                        "Transkribiert die gespeicherte Aufnahme erneut — \
                         größere Abschnitte, alle ausgewählten Modelle.",
                    )
                    .on_disabled_hover_text(
                        "Zu dieser Aufnahme ist kein Audio mehr gespeichert.",
                    )
                    .clicked();
            });
        });

        let ctx = ui.ctx().clone();
        if generate {
            self.preset = kind.preset_key().to_string();
            self.start_job(config, &ctx);
        }
        if rerun {
            self.start_post(config, &ctx);
        }
    }
}

/// Render transcript lines as timestamp + speaker + text instead of raw
/// Markdown, and keep the text selectable.
fn transcript_body(ui: &mut egui::Ui, body: &str) {
    ui.spacing_mut().item_spacing.y = space::SM;
    for line in body.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let (stamp, speaker, spoken) = parse_line(line);
        w::transcript_line(ui, stamp.unwrap_or(""), speaker, spoken);
    }
}

/// `**[00:12] Du:** hallo` → `("00:12", Some("Du"), "hallo")`
fn parse_line(line: &str) -> (Option<&str>, Option<&str>, &str) {
    let Some(rest) = line.strip_prefix("**[") else {
        return (None, None, line);
    };
    let Some(close) = rest.find(']') else {
        return (None, None, line);
    };
    let stamp = &rest[..close];
    let after = &rest[close + 1..];
    if let Some(end) = after.find(":**") {
        let speaker = after[..end].trim();
        let spoken = after[end + 3..].trim_start();
        let speaker = (!speaker.is_empty()).then_some(speaker);
        return (Some(stamp), speaker, spoken);
    }
    let spoken = after.strip_prefix("**").unwrap_or(after).trim_start();
    (Some(stamp), None, spoken)
}
