//! The recording list on the left of the history page.

use egui::{pos2, vec2, Rect, Sense, Stroke, StrokeKind};

use crate::output::transcript_doc::Recording;
use crate::ui::icons::{self, Icon};
use crate::ui::theme::{self, radius, space, text};
use crate::ui::widgets as w;

use super::{reveal, HistoryPage};

/// Height of one row: two lines of text plus padding.
const ROW_HEIGHT: f32 = 56.0;

impl HistoryPage {
    pub(super) fn list_pane(&mut self, ui: &mut egui::Ui) {
        // Filtering happens in `ensure_filter`, not here: matching reads
        // every document of every recording. The index list is small.
        let visible = self.visible.clone();

        theme::card_frame()
            .inner_margin(egui::Margin::same(space::SM as i8))
            .show(ui, |ui| {
                ui.set_min_height(ui.available_height());
                if visible.is_empty() {
                    ui.add_space(space::XL);
                    ui.vertical_centered(|ui| {
                        w::text_line(
                            ui,
                            "Nichts gefunden",
                            text::body(),
                            theme::TEXT_SECONDARY,
                        );
                        ui.add_space(space::XS);
                        w::text_line(
                            ui,
                            "Andere Suchbegriffe versuchen.",
                            text::small(),
                            theme::TEXT_MUTED,
                        );
                    });
                    return;
                }

                let mut select: Option<usize> = None;
                let mut delete: Option<std::path::PathBuf> = None;
                let mut show_in_finder: Option<std::path::PathBuf> = None;

                egui::ScrollArea::vertical()
                    .id_salt("history-list")
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.spacing_mut().item_spacing.y = 2.0;
                        for index in visible {
                            let selected = self.selected == Some(index);
                            let rec = &self.recordings[index];
                            let resp = row(ui, rec, selected);
                            if resp.clicked() {
                                select = Some(index);
                            }
                            resp.context_menu(|ui| {
                                if w::button(
                                    ui,
                                    w::ButtonKind::Ghost,
                                    Some(Icon::Reveal),
                                    "Im Finder zeigen",
                                )
                                .clicked()
                                {
                                    show_in_finder = Some(rec.path.clone());
                                    ui.close();
                                }
                                if w::button(
                                    ui,
                                    w::ButtonKind::Ghost,
                                    Some(Icon::Trash),
                                    "Löschen …",
                                )
                                .clicked()
                                {
                                    delete = Some(rec.path.clone());
                                    ui.close();
                                }
                            });
                        }
                    });

                if let Some(index) = select {
                    self.selected = Some(index);
                    self.confirm_delete = None;
                    self.status = None;
                    self.clamp_tab();
                }
                if let Some(path) = show_in_finder {
                    reveal(&path);
                }
                if let Some(path) = delete {
                    self.confirm_delete = Some(path);
                }
            });
    }
}

/// One list row: date, duration, preview, and a marker when the recording
/// already has KI documents.
fn row(ui: &mut egui::Ui, rec: &Recording, selected: bool) -> egui::Response {
    let width = ui.available_width();
    let (rect, resp) = ui.allocate_exact_size(vec2(width, ROW_HEIGHT), Sense::click());
    if !ui.is_rect_visible(rect) {
        return resp;
    }

    let painter = ui.painter();
    if selected {
        painter.rect(
            rect,
            radius::MD,
            theme::mix(theme::BG_SURFACE, theme::ACCENT, 0.20),
            Stroke::new(1.0, theme::tint(theme::ACCENT, 0.45)),
            StrokeKind::Inside,
        );
        // Accent bar on the leading edge.
        painter.rect_filled(
            Rect::from_min_size(
                pos2(rect.left() + 1.0, rect.top() + 10.0),
                vec2(3.0, rect.height() - 20.0),
            ),
            radius::PILL,
            theme::ACCENT_HOVER,
        );
    } else if resp.hovered() {
        painter.rect_filled(rect, radius::MD, theme::BG_ELEVATED);
    }

    let named = rec.title != "Meeting";
    let primary = if named {
        rec.title.clone()
    } else {
        rec.display_date()
    };
    let secondary = if named {
        format!("{} · {}", rec.display_date(), rec.preview)
    } else {
        rec.preview.clone()
    };

    let has_ai = !rec.docs.is_empty();
    let left = rect.left() + 12.0;
    let mut right = rect.right() - 10.0;

    // Duration, right aligned on the first line.
    if let Some(duration) = &rec.duration {
        let short = shorten_duration(duration);
        let galley = w::truncated(ui, &short, text::mono(), 70.0);
        right -= galley.size().x;
        ui.painter().galley(
            pos2(right, rect.top() + 11.0),
            galley,
            theme::TEXT_TIMESTAMP,
        );
        right -= 6.0;
    }
    if has_ai {
        let icon_rect = Rect::from_center_size(pos2(right - 7.0, rect.top() + 17.0), vec2(14.0, 14.0));
        icons::paint(ui.painter(), Icon::Sparkle, icon_rect, theme::ACCENT_HOVER);
        right -= 20.0;
    }

    let text_width = (right - left - 6.0).max(40.0);
    let title_galley = w::truncated(ui, &primary, text::strong(), text_width);
    ui.painter()
        .galley(pos2(left, rect.top() + 10.0), title_galley, theme::TEXT_PRIMARY);

    let preview_galley = w::truncated(ui, &secondary, text::small(), rect.width() - 22.0);
    ui.painter().galley(
        pos2(left, rect.top() + 31.0),
        preview_galley,
        theme::TEXT_MUTED,
    );

    resp
}

/// `0h 12m 30s` → `12:30`, `1h 02m 03s` → `1:02:03`.
fn shorten_duration(raw: &str) -> String {
    let nums: Vec<u32> = raw
        .split_whitespace()
        .filter_map(|part| {
            part.trim_end_matches(['h', 'm', 's'])
                .parse::<u32>()
                .ok()
        })
        .collect();
    match nums.as_slice() {
        [h, m, s] if *h > 0 => format!("{h}:{m:02}:{s:02}"),
        [_, m, s] => format!("{m}:{s:02}"),
        _ => raw.to_string(),
    }
}
