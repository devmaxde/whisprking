//! "Verlauf" tab — list previous meeting transcripts with preview, search,
//! reveal-in-Finder, delete. Per-file state (whether the LLM cleanup has
//! already run) lives in a `<stem>.meta.json` sidecar.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use crate::config::Config;
use crate::output::transcript_meta::TranscriptMeta;
use crate::postprocess::llm::{make_provider, resolve_system_prompt, LlmConfig};
use crate::ui::styles;

pub struct HistoryPage {
    search: String,
    files: Vec<TranscriptFile>,
    selected: Option<usize>,
    preview: String,
    last_data_dir: PathBuf,
    cleanup_job: Arc<Mutex<Option<CleanupJob>>>,
}

struct TranscriptFile {
    path: PathBuf,
    name: String,
    preview_line: String,
    modified: SystemTime,
    meta: TranscriptMeta,
}

#[derive(Clone)]
enum CleanupPhase {
    Running,
    Done,
    Error(String),
}

struct CleanupJob {
    file: PathBuf,
    phase: CleanupPhase,
}

impl HistoryPage {
    pub fn new() -> Self {
        Self {
            search: String::new(),
            files: Vec::new(),
            selected: None,
            preview: String::new(),
            last_data_dir: PathBuf::new(),
            cleanup_job: Arc::new(Mutex::new(None)),
        }
    }

    pub fn refresh(&mut self, config: &Config) {
        let dir = Config::transcripts_dir(config);
        self.last_data_dir = dir.clone();
        let entries = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            Err(_) => {
                self.files.clear();
                self.selected = None;
                self.preview.clear();
                return;
            }
        };

        let mut files: Vec<TranscriptFile> = entries
            .flatten()
            .filter(|e| e.path().extension().map(|ext| ext == "md").unwrap_or(false))
            .map(|e| {
                let path = e.path();
                let modified = e
                    .metadata()
                    .and_then(|m| m.modified())
                    .unwrap_or(SystemTime::UNIX_EPOCH);
                let name = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or_default()
                    .to_string();
                let preview_line = preview_line(&path);
                let meta = TranscriptMeta::load(&path);
                TranscriptFile {
                    path,
                    name,
                    preview_line,
                    modified,
                    meta,
                }
            })
            .collect();
        files.sort_by_key(|b| std::cmp::Reverse(b.modified));

        self.files = files;
        self.selected = (!self.files.is_empty()).then_some(0);
        self.reload_preview();
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, config: &Config) {
        // If a background cleanup just finished, pick up the new content +
        // meta on the next paint.
        let job_done = {
            let g = self.cleanup_job.lock().unwrap();
            matches!(g.as_ref().map(|j| &j.phase), Some(CleanupPhase::Done))
        };
        if job_done {
            self.refresh(config);
        }

        ui.vertical(|ui| {
            ui.add_space(8.0);
            ui.heading("Verlauf");
            ui.add_space(8.0);

            if ui
                .text_edit_singleline(&mut self.search)
                .on_hover_text("Suche im Dateinamen oder Inhalt")
                .changed()
            {
                // search is applied in render below; nothing else to do.
            }

            ui.add_space(8.0);

            ui.horizontal_top(|ui| {
                let list_width = (ui.available_width() * 0.35).max(180.0);
                ui.vertical(|ui| {
                    ui.set_width(list_width);
                    egui::ScrollArea::vertical()
                        .id_salt("history-list")
                        .show(ui, |ui| {
                            let needle = self.search.trim().to_lowercase();
                            let mut to_select: Option<usize> = None;
                            for (i, file) in self.files.iter().enumerate() {
                                if !matches_filter(file, &needle) {
                                    continue;
                                }
                                let selected = self.selected == Some(i);
                                let badge = if file.meta.cleanup_ran { "✓ " } else { "" };
                                let label = format!("{badge}{}\n{}", file.name, file.preview_line);
                                let resp = ui.selectable_label(selected, label);
                                if resp.clicked() {
                                    to_select = Some(i);
                                }
                                resp.context_menu(|ui| {
                                    if ui.button("Im Finder zeigen").clicked() {
                                        reveal_in_finder(&file.path);
                                        ui.close();
                                    }
                                    if ui.button("Löschen").clicked() {
                                        let _ = std::fs::remove_file(&file.path);
                                        TranscriptMeta::delete(&file.path);
                                        ui.close();
                                    }
                                });
                            }
                            if let Some(i) = to_select {
                                self.selected = Some(i);
                                self.reload_preview();
                            }
                        });
                });

                ui.separator();

                ui.vertical(|ui| {
                    self.preview_pane(ui, config);
                });
            });
        });
    }

    fn preview_pane(&mut self, ui: &mut egui::Ui, config: &Config) {
        let selected_path = self
            .selected
            .and_then(|i| self.files.get(i))
            .map(|f| f.path.clone());

        let (cleanup_ran, cleanup_at) = self
            .selected
            .and_then(|i| self.files.get(i))
            .map(|f| (f.meta.cleanup_ran, f.meta.cleanup_at.clone()))
            .unwrap_or((false, None));

        ui.horizontal(|ui| {
            let provider_ok = config.ai_postprocess.provider != "none";
            let job_running = {
                let g = self.cleanup_job.lock().unwrap();
                matches!(g.as_ref().map(|j| &j.phase), Some(CleanupPhase::Running))
            };

            let label = if cleanup_ran {
                "Cleanup erneut ausführen"
            } else {
                "Cleanup ausführen"
            };
            let enabled = selected_path.is_some() && provider_ok && !job_running;
            if ui
                .add_enabled(enabled, egui::Button::new(label))
                .on_hover_text(if provider_ok {
                    "LLM-Nachbearbeitung auf dieses Transkript anwenden."
                } else {
                    "Provider in den Einstellungen aktivieren."
                })
                .clicked()
            {
                if let Some(path) = selected_path.clone() {
                    self.start_cleanup(&path, config, ui.ctx());
                }
            }

            if cleanup_ran {
                let stamp = cleanup_at.as_deref().unwrap_or("?");
                ui.label(
                    egui::RichText::new(format!("Bereinigt: {stamp}")).color(styles::ACCENT_GREEN),
                );
            }
        });

        // Status / error row.
        let snapshot: Option<(PathBuf, CleanupPhase)> = self
            .cleanup_job
            .lock()
            .unwrap()
            .as_ref()
            .map(|j| (j.file.clone(), j.phase.clone()));
        if let Some((file, phase)) = snapshot {
            ui.add_space(4.0);
            match phase {
                CleanupPhase::Running => {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label(
                            egui::RichText::new(format!(
                                "Bereinige {} …",
                                file.file_name().and_then(|s| s.to_str()).unwrap_or("")
                            ))
                            .color(styles::TEXT_SECONDARY),
                        );
                    });
                    ui.ctx().request_repaint();
                }
                CleanupPhase::Done => {
                    // refresh() at the top of ui() will have picked up the
                    // new file/meta; clear the job so the row disappears.
                    *self.cleanup_job.lock().unwrap() = None;
                }
                CleanupPhase::Error(msg) => {
                    ui.label(
                        egui::RichText::new(format!("Cleanup fehlgeschlagen: {msg}"))
                            .color(styles::ACCENT_RED),
                    );
                    if ui.button("Schließen").clicked() {
                        *self.cleanup_job.lock().unwrap() = None;
                    }
                }
            }
        }

        ui.add_space(6.0);
        egui::ScrollArea::vertical()
            .id_salt("history-preview")
            .show(ui, |ui| {
                ui.label(egui::RichText::new(&self.preview).color(styles::TEXT_PRIMARY));
            });
    }

    fn start_cleanup(&self, file: &Path, config: &Config, ctx: &egui::Context) {
        let ai = &config.ai_postprocess;
        let llm_cfg = LlmConfig::from_ai_section(ai);
        let preset = ai.default_prompt.clone();
        let custom_prompt = ai.custom_prompt.clone();
        let overrides: std::collections::HashMap<String, String> = ai
            .prompts
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let file = file.to_path_buf();

        {
            let mut g = self.cleanup_job.lock().unwrap();
            *g = Some(CleanupJob {
                file: file.clone(),
                phase: CleanupPhase::Running,
            });
        }

        let job = Arc::clone(&self.cleanup_job);
        let ctx_thread = ctx.clone();
        std::thread::spawn(move || {
            let result = run_cleanup(&file, &llm_cfg, &preset, &custom_prompt, &overrides);
            let mut g = job.lock().unwrap();
            if let Some(j) = g.as_mut() {
                j.phase = match result {
                    Ok(()) => CleanupPhase::Done,
                    Err(e) => CleanupPhase::Error(e),
                };
            }
            drop(g);
            ctx_thread.request_repaint();
        });
        ctx.request_repaint();
    }

    fn reload_preview(&mut self) {
        self.preview = match self.selected.and_then(|i| self.files.get(i)) {
            Some(file) => std::fs::read_to_string(&file.path).unwrap_or_default(),
            None => String::new(),
        };
    }
}

impl Default for HistoryPage {
    fn default() -> Self {
        Self::new()
    }
}

fn run_cleanup(
    file: &Path,
    llm_cfg: &LlmConfig,
    preset: &str,
    custom_prompt: &str,
    overrides: &std::collections::HashMap<String, String>,
) -> Result<(), String> {
    let body = std::fs::read_to_string(file).map_err(|e| e.to_string())?;
    let provider =
        make_provider(llm_cfg).ok_or_else(|| "kein Provider konfiguriert".to_string())?;
    let system = resolve_system_prompt(preset, custom_prompt, overrides);
    let cleaned = provider.run(&system, &body).map_err(|e| e.to_string())?;
    let now = chrono::Local::now();
    let stamp = now.format("%Y-%m-%d %H:%M").to_string();
    let appended = format!("{body}\n\n---\n\n## Bereinigung — {stamp}\n\n{cleaned}\n");
    std::fs::write(file, appended).map_err(|e| e.to_string())?;
    let meta = TranscriptMeta {
        cleanup_ran: true,
        cleanup_at: Some(now.format("%Y-%m-%d %H:%M:%S").to_string()),
        cleanup_preset: Some(preset.to_string()),
    };
    meta.save(file).map_err(|e| e.to_string())?;
    Ok(())
}

fn matches_filter(file: &TranscriptFile, needle: &str) -> bool {
    if needle.is_empty() {
        return true;
    }
    if file.name.to_lowercase().contains(needle) {
        return true;
    }
    std::fs::read_to_string(&file.path)
        .map(|c| c.to_lowercase().contains(needle))
        .unwrap_or(false)
}

fn preview_line(path: &std::path::Path) -> String {
    let body = match std::fs::read_to_string(path) {
        Ok(b) => b,
        Err(_) => return "(leer)".into(),
    };
    for line in body.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty()
            || trimmed.starts_with('#')
            || trimmed.starts_with('-')
            || trimmed == "---"
        {
            continue;
        }
        let cleaned = strip_timestamp(trimmed);
        return cleaned.chars().take(80).collect();
    }
    "(leer)".into()
}

fn reveal_in_finder(path: &std::path::Path) {
    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("open")
            .arg("-R")
            .arg(path)
            .spawn();
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = opener::open(path);
    }
}

fn strip_timestamp(line: &str) -> String {
    // Lines look like `**[MM:SS]** the rest…`
    if let Some(rest) = line.strip_prefix("**[") {
        if let Some(end) = rest.find("]**") {
            return rest[end + 3..].trim().to_string();
        }
    }
    line.to_string()
}
