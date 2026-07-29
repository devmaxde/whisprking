//! The component library every page is built from.
//!
//! egui's stock widgets are functional but generic; a modern-looking app
//! needs a small set of opinionated pieces used consistently — buttons with
//! one padding, cards with one radius, one way to show a status, one way to
//! show "nothing here yet". Everything in this module paints itself from the
//! tokens in [`crate::ui::theme`], so the whole app restyles from one place.

use std::sync::Arc;

use egui::{
    pos2, vec2, Align, Color32, CornerRadius, FontId, Galley, Layout, Rect, Response, Sense,
    Stroke, StrokeKind, Ui, Vec2,
};

use crate::ui::icons::{self, Icon};
use crate::ui::theme::{self, radius, space, text};

// --- text helpers ----------------------------------------------------------

fn layout(ui: &Ui, s: &str, font: FontId) -> Arc<Galley> {
    ui.painter()
        .layout_no_wrap(s.to_owned(), font, Color32::PLACEHOLDER)
}

/// One line, cut off with an ellipsis instead of wrapping — list rows stay
/// the same height no matter how long the content is.
pub fn truncated(ui: &Ui, s: &str, font: FontId, width: f32) -> Arc<Galley> {
    let mut job = egui::text::LayoutJob::single_section(
        s.to_owned(),
        egui::TextFormat {
            font_id: font,
            color: Color32::PLACEHOLDER,
            ..Default::default()
        },
    );
    job.wrap = egui::text::TextWrapping {
        max_width: width.max(16.0),
        max_rows: 1,
        break_anywhere: true,
        overflow_character: Some('…'),
    };
    ui.painter().layout_job(job)
}

/// A line of text in an explicit font + color, no widget frame.
pub fn text_line(ui: &mut Ui, s: &str, font: FontId, color: Color32) -> Response {
    ui.add(egui::Label::new(
        egui::RichText::new(s).font(font).color(color),
    ))
}

/// Small, muted explanatory text under a control.
pub fn hint(ui: &mut Ui, s: &str) {
    ui.add(
        egui::Label::new(
            egui::RichText::new(s)
                .font(text::small())
                .color(theme::TEXT_MUTED),
        )
        .wrap(),
    );
}

/// Section heading inside a card.
pub fn section_title(ui: &mut Ui, icon: Option<Icon>, title: &str) {
    ui.horizontal(|ui| {
        if let Some(icon) = icon {
            let (r, _) = ui.allocate_exact_size(Vec2::splat(15.0), Sense::hover());
            icons::paint(ui.painter(), icon, r, theme::TEXT_SECONDARY);
        }
        text_line(ui, title, text::title(), theme::TEXT_PRIMARY);
    });
}

/// The title block at the top of a page.
pub fn page_header(ui: &mut Ui, title: &str, subtitle: &str) {
    ui.vertical(|ui| {
        text_line(ui, title, text::display(), theme::TEXT_PRIMARY);
        if !subtitle.is_empty() {
            ui.add_space(space::XXS);
            text_line(ui, subtitle, text::small(), theme::TEXT_MUTED);
        }
    });
}

// --- containers ------------------------------------------------------------

/// A content card: surface fill, hairline border, generous padding.
pub fn card<R>(ui: &mut Ui, add: impl FnOnce(&mut Ui) -> R) -> R {
    theme::card_frame().show(ui, add).inner
}

/// A card with a title row on the left and optional actions on the right.
pub fn card_titled<R>(
    ui: &mut Ui,
    icon: Option<Icon>,
    title: &str,
    actions: impl FnOnce(&mut Ui),
    body: impl FnOnce(&mut Ui) -> R,
) -> R {
    card(ui, |ui| {
        egui::Sides::new().height(22.0).show(
            ui,
            |ui| section_title(ui, icon, title),
            |ui| actions(ui),
        );
        ui.add_space(space::MD);
        body(ui)
    })
}

/// Hairline divider with breathing room around it.
pub fn divider(ui: &mut Ui) {
    ui.add_space(space::MD);
    let width = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(vec2(width, 1.0), Sense::hover());
    ui.painter().rect_filled(rect, 0.0, theme::BORDER);
    ui.add_space(space::MD);
}

// --- buttons ---------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ButtonKind {
    /// Filled accent — the one primary action of a view.
    Primary,
    /// Filled neutral surface with a border.
    Secondary,
    /// No fill until hovered.
    Ghost,
    /// Destructive, tinted red.
    Danger,
    /// Filled red — used by the running-recording stop button.
    RecordActive,
}

struct ButtonColors {
    fill: Color32,
    stroke: Color32,
    fg: Color32,
}

fn button_colors(kind: ButtonKind, hovered: bool, pressed: bool) -> ButtonColors {
    match kind {
        ButtonKind::Primary => ButtonColors {
            fill: if pressed {
                theme::ACCENT_PRESSED
            } else if hovered {
                theme::ACCENT_HOVER
            } else {
                theme::ACCENT
            },
            stroke: Color32::TRANSPARENT,
            fg: theme::TEXT_ON_ACCENT,
        },
        ButtonKind::Secondary => ButtonColors {
            fill: if pressed {
                theme::mix(theme::BG_ACTIVE, Color32::WHITE, 0.06)
            } else if hovered {
                theme::BG_ACTIVE
            } else {
                theme::BG_ELEVATED
            },
            stroke: if hovered {
                theme::BORDER_STRONG
            } else {
                theme::BORDER
            },
            fg: theme::TEXT_PRIMARY,
        },
        ButtonKind::Ghost => ButtonColors {
            fill: if pressed {
                theme::BG_ACTIVE
            } else if hovered {
                theme::BG_ELEVATED
            } else {
                Color32::TRANSPARENT
            },
            stroke: Color32::TRANSPARENT,
            fg: if hovered {
                theme::TEXT_PRIMARY
            } else {
                theme::TEXT_SECONDARY
            },
        },
        ButtonKind::Danger => ButtonColors {
            fill: if hovered {
                theme::tint(theme::DANGER, 0.18)
            } else {
                theme::tint(theme::DANGER, 0.10)
            },
            stroke: theme::tint(theme::DANGER, 0.35),
            fg: theme::DANGER,
        },
        ButtonKind::RecordActive => ButtonColors {
            fill: if pressed {
                theme::mix(theme::DANGER, Color32::BLACK, 0.15)
            } else if hovered {
                theme::mix(theme::DANGER, Color32::WHITE, 0.10)
            } else {
                theme::DANGER
            },
            stroke: Color32::TRANSPARENT,
            fg: theme::TEXT_ON_ACCENT,
        },
    }
}

/// Standard-height button with an optional leading icon.
pub fn button(ui: &mut Ui, kind: ButtonKind, icon: Option<Icon>, label: &str) -> Response {
    button_sized(ui, kind, icon, label, theme::CONTROL_HEIGHT, radius::SM)
}

/// Prominent pill button (record / primary call to action).
pub fn button_large(ui: &mut Ui, kind: ButtonKind, icon: Option<Icon>, label: &str) -> Response {
    button_sized(
        ui,
        kind,
        icon,
        label,
        theme::CONTROL_HEIGHT_LG,
        radius::PILL,
    )
}

pub fn button_sized(
    ui: &mut Ui,
    kind: ButtonKind,
    icon: Option<Icon>,
    label: &str,
    height: f32,
    corner: CornerRadius,
) -> Response {
    let galley = layout(ui, label, text::strong());
    let icon_size = 15.0_f32;
    let has_icon = icon.is_some();
    let has_text = !label.is_empty();
    let pad = if height >= theme::CONTROL_HEIGHT_LG {
        18.0
    } else {
        13.0
    };
    let gap = if has_icon && has_text { 7.0 } else { 0.0 };
    let width = pad * 2.0
        + if has_icon { icon_size } else { 0.0 }
        + gap
        + if has_text { galley.size().x } else { 0.0 };

    let (rect, resp) = ui.allocate_exact_size(vec2(width, height), Sense::click());
    if !ui.is_rect_visible(rect) {
        return resp;
    }
    let c = button_colors(kind, resp.hovered(), resp.is_pointer_button_down_on());
    let painter = ui.painter();
    painter.rect(
        rect,
        corner,
        c.fill,
        Stroke::new(1.0, c.stroke),
        StrokeKind::Inside,
    );

    let content_w = width - pad * 2.0;
    let mut x = rect.center().x - content_w / 2.0;
    if let Some(icon) = icon {
        let ir = Rect::from_center_size(
            pos2(x + icon_size / 2.0, rect.center().y),
            Vec2::splat(icon_size),
        );
        icons::paint(painter, icon, ir, c.fg);
        x += icon_size + gap;
    }
    if has_text {
        painter.galley(
            pos2(x, rect.center().y - galley.size().y / 2.0),
            galley,
            c.fg,
        );
    }
    resp
}

/// Square, icon-only button — toolbars and row actions.
pub fn icon_button(ui: &mut Ui, icon: Icon, tooltip: &str) -> Response {
    let size = 28.0;
    let (rect, resp) = ui.allocate_exact_size(Vec2::splat(size), Sense::click());
    paint_icon_button(ui, rect, icon, &resp, theme::TEXT_SECONDARY);
    if tooltip.is_empty() {
        resp
    } else {
        resp.on_hover_text(tooltip)
    }
}

/// Icon button placed in an explicit rect (inside a composed row).
pub fn icon_button_at(ui: &mut Ui, rect: Rect, icon: Icon, tooltip: &str) -> Response {
    let resp = ui.interact(rect, ui.auto_id_with(tooltip), Sense::click());
    paint_icon_button(ui, rect, icon, &resp, theme::TEXT_SECONDARY);
    if tooltip.is_empty() {
        resp
    } else {
        resp.on_hover_text(tooltip)
    }
}

/// Danger-tinted icon button (delete).
pub fn icon_button_danger(ui: &mut Ui, icon: Icon, tooltip: &str) -> Response {
    let size = 28.0;
    let (rect, resp) = ui.allocate_exact_size(Vec2::splat(size), Sense::click());
    paint_icon_button(ui, rect, icon, &resp, theme::DANGER);
    if tooltip.is_empty() {
        resp
    } else {
        resp.on_hover_text(tooltip)
    }
}

fn paint_icon_button(ui: &Ui, rect: Rect, icon: Icon, resp: &Response, base: Color32) {
    if !ui.is_rect_visible(rect) {
        return;
    }
    let painter = ui.painter();
    if resp.hovered() || resp.is_pointer_button_down_on() {
        let fill = if resp.is_pointer_button_down_on() {
            theme::BG_ACTIVE
        } else {
            theme::BG_ELEVATED
        };
        painter.rect_filled(rect, radius::SM, fill);
    }
    let color = if resp.hovered() {
        if base == theme::DANGER {
            theme::DANGER
        } else {
            theme::TEXT_PRIMARY
        }
    } else {
        base
    };
    icons::paint(painter, icon, rect.shrink(6.0), color);
}

// --- selection controls ----------------------------------------------------

/// iOS-style switch. Returns `true` when toggled this frame.
pub fn toggle(ui: &mut Ui, on: &mut bool) -> bool {
    let size = vec2(38.0, 22.0);
    let (rect, resp) = ui.allocate_exact_size(size, Sense::click());
    let changed = resp.clicked();
    if changed {
        *on = !*on;
    }
    if ui.is_rect_visible(rect) {
        let t = ui.ctx().animate_bool(resp.id, *on);
        let track = theme::mix(theme::BG_ACTIVE, theme::ACCENT, t);
        let painter = ui.painter();
        painter.rect(
            rect,
            radius::PILL,
            track,
            Stroke::new(1.0, if *on { Color32::TRANSPARENT } else { theme::BORDER_STRONG }),
            StrokeKind::Inside,
        );
        let knob_r = rect.height() / 2.0 - 3.0;
        let x = egui::lerp((rect.left() + knob_r + 3.0)..=(rect.right() - knob_r - 3.0), t);
        painter.circle_filled(pos2(x, rect.center().y), knob_r, Color32::WHITE);
    }
    changed
}

/// A labelled row with a switch on the right.
pub fn toggle_row(ui: &mut Ui, label: &str, hint_text: &str, on: &mut bool) -> bool {
    let mut changed = false;
    egui::Sides::new().show(
        ui,
        |ui| {
            ui.vertical(|ui| {
                text_line(ui, label, text::body(), theme::TEXT_PRIMARY);
                if !hint_text.is_empty() {
                    ui.add_space(space::XXS);
                    hint(ui, hint_text);
                }
            });
        },
        |ui| {
            changed = toggle(ui, on);
        },
    );
    changed
}

/// Segmented control — the modern replacement for a row of radio buttons.
/// Returns `true` if the selection changed.
pub fn segmented<T: Clone + PartialEq>(
    ui: &mut Ui,
    id_salt: &str,
    current: &mut T,
    options: &[(T, &str, bool)],
) -> bool {
    let font = text::strong();
    let pad = 14.0;
    let height = theme::CONTROL_HEIGHT;
    let galleys: Vec<Arc<Galley>> = options
        .iter()
        .map(|(_, label, _)| layout(ui, label, font.clone()))
        .collect();
    let widths: Vec<f32> = galleys.iter().map(|g| g.size().x + pad * 2.0).collect();
    let total: f32 = widths.iter().sum::<f32>() + 6.0;

    let (rect, _) = ui.allocate_exact_size(vec2(total, height), Sense::hover());
    let mut changed = false;
    if !ui.is_rect_visible(rect) {
        return changed;
    }
    ui.painter().rect(
        rect,
        radius::SM,
        theme::BG_INPUT,
        Stroke::new(1.0, theme::BORDER),
        StrokeKind::Inside,
    );

    let base_id = ui.auto_id_with(id_salt);
    let mut x = rect.left() + 3.0;
    for (i, ((value, _, enabled), galley)) in options.iter().zip(galleys).enumerate() {
        let seg = Rect::from_min_size(pos2(x, rect.top() + 3.0), vec2(widths[i], height - 6.0));
        x += widths[i];
        let selected = *current == *value;
        let resp = if *enabled {
            ui.interact(seg, base_id.with(i), Sense::click())
        } else {
            ui.interact(seg, base_id.with(i), Sense::hover())
        };
        if resp.clicked() && !selected {
            *current = value.clone();
            changed = true;
        }
        let painter = ui.painter();
        if selected {
            painter.rect(
                seg,
                radius::SM,
                theme::mix(theme::BG_SURFACE, theme::ACCENT, 0.22),
                Stroke::new(1.0, theme::tint(theme::ACCENT, 0.55)),
                StrokeKind::Inside,
            );
        } else if resp.hovered() && *enabled {
            painter.rect_filled(seg, radius::SM, theme::BG_ELEVATED);
        }
        let color = if !*enabled {
            theme::TEXT_MUTED
        } else if selected {
            theme::TEXT_PRIMARY
        } else {
            theme::TEXT_SECONDARY
        };
        painter.galley(
            pos2(
                seg.center().x - galley.size().x / 2.0,
                seg.center().y - galley.size().y / 2.0,
            ),
            galley,
            color,
        );
    }
    changed
}

/// Tab bar for switching between documents of one recording.
/// Returns the index that was clicked, if any.
pub fn tab_bar(ui: &mut Ui, selected: usize, tabs: &[(&str, bool)]) -> Option<usize> {
    let font = text::strong();
    let height = 32.0;
    let mut clicked = None;

    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = space::XS;
        for (i, (label, filled)) in tabs.iter().enumerate() {
            let galley = layout(ui, label, font.clone());
            let dot_w = if *filled { 12.0 } else { 0.0 };
            let w = galley.size().x + 24.0 + dot_w;
            let (rect, resp) = ui.allocate_exact_size(vec2(w, height), Sense::click());
            if resp.clicked() {
                clicked = Some(i);
            }
            let is_sel = i == selected;
            let painter = ui.painter();
            if is_sel {
                painter.rect_filled(rect, radius::SM, theme::mix(theme::BG_SURFACE, theme::ACCENT, 0.20));
            } else if resp.hovered() {
                painter.rect_filled(rect, radius::SM, theme::BG_ELEVATED);
            }
            let color = if is_sel {
                theme::TEXT_PRIMARY
            } else if *filled {
                theme::TEXT_SECONDARY
            } else {
                theme::TEXT_MUTED
            };
            let mut tx = rect.center().x - (galley.size().x + dot_w) / 2.0;
            if *filled {
                painter.circle_filled(
                    pos2(tx + 3.0, rect.center().y),
                    3.0,
                    if is_sel { theme::ACCENT_HOVER } else { theme::TEXT_MUTED },
                );
                tx += dot_w;
            }
            painter.galley(
                pos2(tx, rect.center().y - galley.size().y / 2.0),
                galley,
                color,
            );
        }
    });
    clicked
}

// --- feedback --------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tone {
    Neutral,
    Success,
    Warning,
    Danger,
    Accent,
}

impl Tone {
    fn color(self) -> Color32 {
        match self {
            Tone::Neutral => theme::TEXT_SECONDARY,
            Tone::Success => theme::SUCCESS,
            Tone::Warning => theme::WARNING,
            Tone::Danger => theme::DANGER,
            Tone::Accent => theme::ACCENT_HOVER,
        }
    }

    fn icon(self) -> Icon {
        match self {
            Tone::Success => Icon::Check,
            Tone::Warning | Tone::Danger => Icon::Warning,
            _ => Icon::Info,
        }
    }
}

/// Small pill with a label — "Bereinigt", "Nicht installiert", …
pub fn badge(ui: &mut Ui, label: &str, tone: Tone) -> Response {
    let color = tone.color();
    let galley = layout(ui, label, text::caption());
    let size = vec2(galley.size().x + 16.0, 19.0);
    let (rect, resp) = ui.allocate_exact_size(size, Sense::hover());
    if ui.is_rect_visible(rect) {
        let painter = ui.painter();
        painter.rect(
            rect,
            radius::PILL,
            theme::tint(color, 0.14),
            Stroke::new(1.0, theme::tint(color, 0.30)),
            StrokeKind::Inside,
        );
        painter.galley(
            pos2(
                rect.center().x - galley.size().x / 2.0,
                rect.center().y - galley.size().y / 2.0,
            ),
            galley,
            color,
        );
    }
    resp
}

/// Full-width message strip: tinted background, icon, wrapped text.
pub fn banner(ui: &mut Ui, tone: Tone, message: &str) {
    let color = tone.color();
    egui::Frame::new()
        .fill(theme::tint(color, 0.10))
        .stroke(Stroke::new(1.0, theme::tint(color, 0.28)))
        .corner_radius(radius::MD)
        .inner_margin(egui::Margin::symmetric(space::MD as i8, space::SM as i8))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal_top(|ui| {
                let (r, _) = ui.allocate_exact_size(Vec2::splat(15.0), Sense::hover());
                icons::paint(
                    ui.painter(),
                    tone.icon(),
                    r.translate(vec2(0.0, 1.0)),
                    color,
                );
                ui.add_space(space::XS);
                ui.add(
                    egui::Label::new(
                        egui::RichText::new(message)
                            .font(text::small())
                            .color(theme::TEXT_PRIMARY),
                    )
                    .wrap(),
                );
            });
        });
}

/// Failure panel: headline, the facts behind it, what to try next, and the
/// other side's answer verbatim — with one button that copies all of it.
///
/// A one-line [`banner`] is the right shape for "Aufnahme gelöscht". It is
/// not enough for a failed KI run, where everything that makes the failure
/// diagnosable — how long the transcript was, which model, which HTTP status
/// — used to be visible only in the log.
///
/// Returns `true` when the user dismissed it.
#[allow(clippy::too_many_arguments)]
pub fn error_panel(
    ui: &mut Ui,
    title: &str,
    facts: &[(String, String)],
    hint: Option<&str>,
    raw: Option<&str>,
    copy: &mut CopyButton,
    report: impl FnOnce() -> String,
) -> bool {
    const LABEL_WIDTH: f32 = 104.0;
    let color = Tone::Danger.color();
    let mut dismiss = false;

    egui::Frame::new()
        .fill(theme::tint(color, 0.10))
        .stroke(Stroke::new(1.0, theme::tint(color, 0.28)))
        .corner_radius(radius::MD)
        .inner_margin(egui::Margin::symmetric(space::MD as i8, space::SM as i8))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            // The headline is a whole sentence and wraps. The actions are laid
            // out first, right to left, so the text wraps inside what is left
            // over instead of running underneath the buttons.
            ui.horizontal_top(|ui| {
                let (r, _) = ui.allocate_exact_size(Vec2::splat(15.0), Sense::hover());
                icons::paint(
                    ui.painter(),
                    Tone::Danger.icon(),
                    r.translate(vec2(0.0, 3.0)),
                    color,
                );
                ui.add_space(space::XS);
                ui.with_layout(Layout::right_to_left(Align::Min), |ui| {
                    dismiss = icon_button(ui, Icon::Close, "Meldung schließen").clicked();
                    copy.show(ui, ButtonKind::Secondary, "Details kopieren", report);
                    ui.with_layout(Layout::left_to_right(Align::Min), |ui| {
                        ui.add(
                            egui::Label::new(
                                egui::RichText::new(title)
                                    .font(text::strong())
                                    .color(theme::TEXT_PRIMARY),
                            )
                            .wrap(),
                        );
                    });
                });
            });

            for (key, value) in facts {
                ui.add_space(space::XXS);
                ui.horizontal_top(|ui| {
                    let (slot, _) = ui.allocate_exact_size(vec2(LABEL_WIDTH, 16.0), Sense::hover());
                    let galley = truncated(ui, key, text::small(), LABEL_WIDTH);
                    ui.painter()
                        .galley(slot.left_top(), galley, theme::TEXT_MUTED);
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new(value)
                                .font(text::small())
                                .color(theme::TEXT_SECONDARY),
                        )
                        .selectable(true)
                        .wrap(),
                    );
                });
            }

            if let Some(hint) = hint {
                ui.add_space(space::SM);
                ui.add(
                    egui::Label::new(
                        egui::RichText::new(hint)
                            .font(text::small())
                            .color(theme::TEXT_PRIMARY),
                    )
                    .wrap(),
                );
            }

            if let Some(raw) = raw {
                ui.add_space(space::SM);
                theme::inset_frame().show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    egui::ScrollArea::vertical()
                        .id_salt("error-panel-raw")
                        .max_height(120.0)
                        .auto_shrink([false, true])
                        .show(ui, |ui| {
                            ui.add(
                                egui::Label::new(
                                    egui::RichText::new(raw)
                                        .font(text::mono())
                                        .color(theme::TEXT_SECONDARY),
                                )
                                .selectable(true)
                                .wrap(),
                            );
                        });
                });
            }
        });
    dismiss
}

/// Spinner + label, for anything in flight.
pub fn busy_row(ui: &mut Ui, message: &str) {
    ui.horizontal(|ui| {
        ui.add(egui::Spinner::new().size(14.0).color(theme::ACCENT_HOVER));
        ui.add_space(space::XS);
        text_line(ui, message, text::small(), theme::TEXT_SECONDARY);
    });
    ui.ctx().request_repaint();
}

/// Centered "nothing here" state with an icon medallion.
pub fn empty_state(ui: &mut Ui, icon: Icon, title: &str, hint_text: &str) {
    ui.vertical_centered(|ui| {
        ui.add_space(ui.available_height() * 0.22);
        let (rect, _) = ui.allocate_exact_size(Vec2::splat(52.0), Sense::hover());
        ui.painter()
            .rect_filled(rect, radius::LG, theme::BG_ELEVATED);
        icons::paint(ui.painter(), icon, rect.shrink(15.0), theme::TEXT_MUTED);
        ui.add_space(space::MD);
        text_line(ui, title, text::title(), theme::TEXT_SECONDARY);
        if !hint_text.is_empty() {
            ui.add_space(space::XS);
            text_line(ui, hint_text, text::small(), theme::TEXT_MUTED);
        }
    });
}

/// Horizontal level meter with a caption.
pub fn level_meter(ui: &mut Ui, label: &str, color: Color32, level: f32) {
    ui.horizontal(|ui| {
        let galley = layout(ui, label, text::caption());
        let (lr, _) = ui.allocate_exact_size(vec2(galley.size().x.max(44.0), 12.0), Sense::hover());
        ui.painter().galley(
            pos2(lr.left(), lr.center().y - galley.size().y / 2.0),
            galley,
            color,
        );
        let (rect, _) = ui.allocate_exact_size(vec2(96.0, 6.0), Sense::hover());
        let painter = ui.painter();
        painter.rect_filled(rect, radius::PILL, theme::BG_INPUT);
        let w = rect.width() * level.clamp(0.0, 1.0);
        if w > 1.0 {
            painter.rect_filled(
                Rect::from_min_size(rect.min, vec2(w, rect.height())),
                radius::PILL,
                color,
            );
        }
    });
}

// --- inputs ----------------------------------------------------------------

/// Frame used by every text input so they all match.
pub fn input_frame() -> egui::Frame {
    egui::Frame::new()
        .fill(theme::BG_INPUT)
        .stroke(Stroke::new(1.0, theme::BORDER))
        .corner_radius(radius::SM)
        .inner_margin(egui::Margin::symmetric(space::SM as i8, 6))
}

/// Single-line text field with our frame.
pub fn text_field(
    ui: &mut Ui,
    id_salt: &str,
    value: &mut String,
    placeholder: &str,
    width: f32,
    password: bool,
) -> Response {
    ui.add_sized(
        vec2(width, theme::CONTROL_HEIGHT),
        egui::TextEdit::singleline(value)
            .id_salt(id_salt)
            .hint_text(placeholder)
            .password(password)
            .font(text::body())
            .frame(input_frame())
            .margin(egui::Margin::ZERO)
            .desired_width(width),
    )
}

/// Multi-line input with our frame — for prompts, which are paragraphs, not
/// single-line values. `rows` is the visible height in text rows; it grows
/// with the content.
pub fn text_area(ui: &mut Ui, id_salt: &str, value: &mut String, rows: usize) -> Response {
    let width = ui.available_width();
    ui.add(
        egui::TextEdit::multiline(value)
            .id_salt(id_salt)
            .font(text::body())
            .frame(input_frame())
            .desired_rows(rows)
            .desired_width(width),
    )
}

/// Search box with a leading magnifier and a clear button.
/// Returns `true` if the query changed.
pub fn search_field(ui: &mut Ui, id_salt: &str, query: &mut String, placeholder: &str) -> bool {
    let width = ui.available_width();
    let height = theme::CONTROL_HEIGHT;
    let (rect, _) = ui.allocate_exact_size(vec2(width, height), Sense::hover());
    ui.painter().rect(
        rect,
        radius::SM,
        theme::BG_INPUT,
        Stroke::new(1.0, theme::BORDER),
        StrokeKind::Inside,
    );
    icons::paint(
        ui.painter(),
        Icon::Search,
        Rect::from_center_size(pos2(rect.left() + 17.0, rect.center().y), Vec2::splat(14.0)),
        theme::TEXT_MUTED,
    );

    let has_text = !query.is_empty();
    let right_pad = if has_text { 32.0 } else { 10.0 };
    // `Ui::put` justifies the edit over the full row height, and a `TextEdit`
    // lays its text out at the top of whatever it is given — the hint text
    // unconditionally so, whatever `vertical_align` says. With no frame margin
    // to fake the padding (which is what carries `text_field`, see
    // `input_frame`) that pinned the placeholder to the top edge of the box, a
    // few pixels above the magnifier beside it. Hand it a slot one text row
    // tall, centred on the row, so text and placeholder sit on the same line
    // whatever `CONTROL_HEIGHT` is.
    let row = layout(ui, "Ag", text::body()).size().y;
    let field = Rect::from_min_max(
        pos2(rect.left() + 30.0, rect.center().y - row / 2.0),
        pos2(rect.right() - right_pad, rect.center().y + row / 2.0),
    );
    let resp = ui.put(
        field,
        egui::TextEdit::singleline(query)
            .id_salt(id_salt)
            .hint_text(placeholder)
            .font(text::body())
            .vertical_align(Align::Center)
            .frame(egui::Frame::NONE)
            .margin(egui::Margin::ZERO)
            .desired_width(field.width()),
    );
    let mut changed = resp.changed();

    if has_text {
        let clear = Rect::from_center_size(
            pos2(rect.right() - 17.0, rect.center().y),
            Vec2::splat(20.0),
        );
        if icon_button_at(ui, clear, Icon::Close, "Suche löschen").clicked() {
            query.clear();
            changed = true;
        }
    }
    changed
}

/// Combo box styled like the rest of the controls.
pub fn combo<T: Clone + PartialEq>(
    ui: &mut Ui,
    id_salt: &str,
    width: f32,
    current: &mut T,
    options: &[(T, String)],
    selected_label: &str,
) -> bool {
    let mut changed = false;
    // The combo is boxed into an exactly `width` wide slot. `ComboBox::width`
    // alone is only a minimum: a long entry widens the box past its
    // container, egui then expands the parent's max rect, and every row
    // below it draws outside the card.
    ui.allocate_ui_with_layout(
        vec2(width, theme::CONTROL_HEIGHT),
        Layout::left_to_right(Align::Center),
        |ui| {
            ui.set_max_width(width);
            egui::ComboBox::from_id_salt(id_salt)
                .width(width)
                .height(320.0)
                .truncate()
                .selected_text(
                    egui::RichText::new(selected_label)
                        .font(text::body())
                        .color(theme::TEXT_PRIMARY),
                )
                .show_ui(ui, |ui| {
                    ui.set_min_width(width.max(220.0));
                    for (value, label) in options {
                        let selected = *current == *value;
                        if ui
                            .selectable_label(
                                selected,
                                egui::RichText::new(label).font(text::body()).color(
                                    if selected {
                                        theme::TEXT_PRIMARY
                                    } else {
                                        theme::TEXT_SECONDARY
                                    },
                                ),
                            )
                            .clicked()
                        {
                            *current = value.clone();
                            changed = true;
                        }
                    }
                });
        },
    );
    changed
}

/// Label + optional hint on the left, control on the right.
pub fn setting_row<R>(
    ui: &mut Ui,
    label: &str,
    hint_text: &str,
    control: impl FnOnce(&mut Ui) -> R,
) -> R {
    let mut out = None;
    egui::Sides::new().shrink_left().show(
        ui,
        |ui| {
            ui.vertical(|ui| {
                ui.set_max_width(ui.available_width().min(320.0));
                text_line(ui, label, text::body(), theme::TEXT_PRIMARY);
                if !hint_text.is_empty() {
                    ui.add_space(space::XXS);
                    hint(ui, hint_text);
                }
            });
        },
        |ui| {
            out = Some(control(ui));
        },
    );
    out.expect("setting_row control ran")
}

// --- copy ------------------------------------------------------------------

/// A copy button that confirms itself for a moment after being pressed —
/// without feedback nobody trusts that the click did anything.
#[derive(Default)]
pub struct CopyButton {
    copied_at: Option<f64>,
}

const COPY_FEEDBACK_SECONDS: f64 = 1.6;

impl CopyButton {
    fn is_fresh(&self, now: f64) -> bool {
        self.copied_at
            .is_some_and(|t| now - t < COPY_FEEDBACK_SECONDS)
    }

    /// Full button with a label. `payload` is only called on click.
    pub fn show(
        &mut self,
        ui: &mut Ui,
        kind: ButtonKind,
        label: &str,
        payload: impl FnOnce() -> String,
    ) -> bool {
        let now = ui.ctx().input(|i| i.time);
        let fresh = self.is_fresh(now);
        let (icon, text_label) = if fresh {
            (Icon::Check, "Kopiert")
        } else {
            (Icon::Copy, label)
        };
        let clicked = button(ui, kind, Some(icon), text_label).clicked();
        if clicked {
            ui.ctx().copy_text(payload());
            self.copied_at = Some(now);
        }
        if fresh {
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(200));
        }
        clicked
    }

    /// Icon-only variant for dense rows.
    pub fn show_icon(&mut self, ui: &mut Ui, tooltip: &str, payload: impl FnOnce() -> String) -> bool {
        let now = ui.ctx().input(|i| i.time);
        let fresh = self.is_fresh(now);
        let icon = if fresh { Icon::Check } else { Icon::Copy };
        let clicked = icon_button(ui, icon, tooltip).clicked();
        if clicked {
            ui.ctx().copy_text(payload());
            self.copied_at = Some(now);
        }
        if fresh {
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(200));
        }
        clicked
    }
}

/// Selectable, wrapped body text — so the user can also grab a single
/// sentence with the mouse instead of copying everything.
pub fn selectable_body(ui: &mut Ui, body: &str) {
    ui.add(
        egui::Label::new(
            egui::RichText::new(body)
                .font(text::body())
                .color(theme::TEXT_PRIMARY)
                .line_height(Some(21.0)),
        )
        .selectable(true)
        .wrap(),
    );
}

/// One line of a transcript: timestamp, speaker pill, selectable text.
/// Shared by the live meeting view and the history detail pane so a
/// transcript looks the same wherever it is shown.
pub fn transcript_line(ui: &mut Ui, stamp: &str, speaker: Option<&str>, body: &str) {
    ui.horizontal_top(|ui| {
        let (slot, _) = ui.allocate_exact_size(vec2(38.0, 18.0), Sense::hover());
        if !stamp.is_empty() {
            let galley = truncated(ui, stamp, text::mono(), 38.0);
            ui.painter()
                .galley(slot.left_top() + vec2(0.0, 2.0), galley, theme::TEXT_TIMESTAMP);
        }
        if let Some(name) = speaker {
            let color = speaker_color(name);
            let galley = truncated(ui, name, text::caption(), 90.0);
            let (pill, _) =
                ui.allocate_exact_size(vec2(galley.size().x + 14.0, 18.0), Sense::hover());
            ui.painter()
                .rect_filled(pill, radius::PILL, theme::tint(color, 0.16));
            ui.painter().galley(
                pos2(
                    pill.center().x - galley.size().x / 2.0,
                    pill.center().y - galley.size().y / 2.0,
                ),
                galley,
                color,
            );
        }
        selectable_body(ui, body);
    });
}

/// Color of a speaker pill. Transcripts from older builds say
/// `You` / `Others`, current ones `Du` / `Andere`.
pub fn speaker_color(speaker: &str) -> Color32 {
    match speaker {
        "Du" | "You" | "Ich" => theme::TRACK_MINE,
        _ => theme::TRACK_THEIRS,
    }
}
