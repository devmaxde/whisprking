//! Vector icons, drawn with the painter.
//!
//! The old UI used emoji ("🎙", "⚙", "⏹") for its controls, which is one of
//! the fastest ways to make an app look unfinished: the glyphs come from a
//! different font, have their own color and baseline, and change with the
//! platform. These are simple stroked paths — one color, consistent weight,
//! crisp at any size.
//!
//! Every icon is drawn inside a centered square derived from the given rect,
//! in normalized 0…1 coordinates, so a call site only has to pick a box.

use egui::{Color32, Painter, Pos2, Rect, Shape, Stroke, StrokeKind, Vec2};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Icon {
    Mic,
    Waveform,
    Transcript,
    Settings,
    Record,
    Stop,
    Pause,
    Play,
    Copy,
    Check,
    Search,
    Trash,
    Reveal,
    Sparkle,
    Refresh,
    Download,
    Import,
    Close,
    ChevronDown,
    ChevronRight,
    Clock,
    Person,
    Speaker,
    Warning,
    Info,
    Folder,
}

/// Paint `icon` centered in `rect`.
pub fn paint(painter: &Painter, icon: Icon, rect: Rect, color: Color32) {
    let side = rect.width().min(rect.height());
    let b = Rect::from_center_size(rect.center(), Vec2::splat(side));
    let w = (side * 0.085).clamp(1.1, 2.0);
    let stroke = Stroke::new(w, color);

    // Normalized point inside the icon box.
    let p = |x: f32, y: f32| -> Pos2 { b.lerp_inside(egui::vec2(x, y)) };
    let line = |pts: Vec<Pos2>| painter.add(Shape::line(pts, stroke));
    let seg = |a: Pos2, c: Pos2| painter.line_segment([a, c], stroke);

    match icon {
        Icon::Mic => {
            // capsule
            let cap = Rect::from_min_max(p(0.36, 0.14), p(0.64, 0.56));
            painter.rect_stroke(cap, egui::CornerRadius::same(99), stroke, StrokeKind::Middle);
            // cradle
            line(vec![
                p(0.24, 0.48),
                p(0.24, 0.55),
                p(0.5, 0.74),
                p(0.76, 0.55),
                p(0.76, 0.48),
            ]);
            seg(p(0.5, 0.74), p(0.5, 0.88));
        },
        Icon::Waveform => {
            for (x, h) in [
                (0.16, 0.16),
                (0.32, 0.34),
                (0.48, 0.24),
                (0.64, 0.40),
                (0.82, 0.20),
            ] {
                seg(p(x, 0.5 - h), p(x, 0.5 + h));
            }
        },
        Icon::Transcript => {
            let page = Rect::from_min_max(p(0.22, 0.12), p(0.78, 0.88));
            painter.rect_stroke(
                page,
                egui::CornerRadius::same((side * 0.12) as u8),
                stroke,
                StrokeKind::Middle,
            );
            for y in [0.34, 0.5, 0.66] {
                seg(p(0.34, y), p(0.66, y));
            }
        },
        Icon::Settings => {
            painter.circle_stroke(b.center(), side * 0.20, stroke);
            for i in 0..8 {
                let a = std::f32::consts::TAU * (i as f32) / 8.0;
                let dir = Vec2::angled(a);
                seg(
                    b.center() + dir * side * 0.30,
                    b.center() + dir * side * 0.42,
                );
            }
        },
        Icon::Record => {
            painter.circle_filled(b.center(), side * 0.30, color);
        },
        Icon::Stop => {
            let r = Rect::from_center_size(b.center(), Vec2::splat(side * 0.52));
            painter.rect_filled(r, egui::CornerRadius::same((side * 0.10) as u8), color);
        },
        Icon::Pause => {
            let cr = egui::CornerRadius::same((side * 0.06) as u8);
            painter.rect_filled(Rect::from_min_max(p(0.30, 0.24), p(0.43, 0.76)), cr, color);
            painter.rect_filled(Rect::from_min_max(p(0.57, 0.24), p(0.70, 0.76)), cr, color);
        },
        Icon::Play => {
            painter.add(Shape::convex_polygon(
                vec![p(0.34, 0.22), p(0.34, 0.78), p(0.78, 0.5)],
                color,
                Stroke::NONE,
            ));
        },
        Icon::Copy => {
            // Front sheet, plus the visible corner of the sheet behind it —
            // drawn as an open path so nothing has to be painted over.
            let front = Rect::from_min_max(p(0.36, 0.36), p(0.86, 0.86));
            painter.rect_stroke(
                front,
                egui::CornerRadius::same((side * 0.10) as u8),
                stroke,
                StrokeKind::Middle,
            );
            line(vec![
                p(0.28, 0.64),
                p(0.16, 0.64),
                p(0.16, 0.16),
                p(0.64, 0.16),
                p(0.64, 0.28),
            ]);
        },
        Icon::Check => {
            line(vec![p(0.20, 0.53), p(0.42, 0.74), p(0.80, 0.28)]);
        },
        Icon::Search => {
            painter.circle_stroke(p(0.44, 0.44), side * 0.24, stroke);
            seg(p(0.62, 0.62), p(0.84, 0.84));
        },
        Icon::Trash => {
            seg(p(0.16, 0.26), p(0.84, 0.26));
            seg(p(0.40, 0.26), p(0.40, 0.16));
            seg(p(0.60, 0.26), p(0.60, 0.16));
            seg(p(0.40, 0.16), p(0.60, 0.16));
            line(vec![
                p(0.26, 0.26),
                p(0.31, 0.86),
                p(0.69, 0.86),
                p(0.74, 0.26),
            ]);
        },
        Icon::Reveal => {
            line(vec![
                p(0.52, 0.18),
                p(0.18, 0.18),
                p(0.18, 0.82),
                p(0.82, 0.82),
                p(0.82, 0.48),
            ]);
            seg(p(0.50, 0.50), p(0.84, 0.16));
            line(vec![p(0.60, 0.16), p(0.84, 0.16), p(0.84, 0.40)]);
        },
        Icon::Sparkle => {
            let star = |cx: f32, cy: f32, r: f32| {
                painter.add(Shape::convex_polygon(
                    vec![
                        p(cx, cy - r),
                        p(cx + r * 0.32, cy - r * 0.32),
                        p(cx + r, cy),
                        p(cx + r * 0.32, cy + r * 0.32),
                        p(cx, cy + r),
                        p(cx - r * 0.32, cy + r * 0.32),
                        p(cx - r, cy),
                        p(cx - r * 0.32, cy - r * 0.32),
                    ],
                    color,
                    Stroke::NONE,
                ));
            };
            star(0.42, 0.42, 0.32);
            star(0.76, 0.74, 0.18);
        },
        Icon::Refresh => {
            let c = b.center();
            let r = side * 0.30;
            let mut pts = Vec::new();
            let start = -0.35 * std::f32::consts::TAU;
            let sweep = 0.78 * std::f32::consts::TAU;
            for i in 0..=24 {
                let a = start + sweep * (i as f32 / 24.0);
                pts.push(c + Vec2::angled(a) * r);
            }
            painter.add(Shape::line(pts, stroke));
            // arrow head at the start of the arc
            let head = c + Vec2::angled(start) * r;
            painter.add(Shape::convex_polygon(
                vec![
                    head + egui::vec2(-side * 0.10, -side * 0.02),
                    head + egui::vec2(side * 0.06, -side * 0.10),
                    head + egui::vec2(side * 0.04, side * 0.08),
                ],
                color,
                Stroke::NONE,
            ));
        },
        Icon::Download => {
            seg(p(0.5, 0.14), p(0.5, 0.62));
            line(vec![p(0.32, 0.44), p(0.5, 0.62), p(0.68, 0.44)]);
            line(vec![p(0.18, 0.72), p(0.18, 0.86), p(0.82, 0.86), p(0.82, 0.72)]);
        },
        Icon::Import => {
            seg(p(0.5, 0.62), p(0.5, 0.14));
            line(vec![p(0.32, 0.32), p(0.5, 0.14), p(0.68, 0.32)]);
            line(vec![p(0.18, 0.72), p(0.18, 0.86), p(0.82, 0.86), p(0.82, 0.72)]);
        },
        Icon::Close => {
            seg(p(0.26, 0.26), p(0.74, 0.74));
            seg(p(0.74, 0.26), p(0.26, 0.74));
        },
        Icon::ChevronDown => {
            line(vec![p(0.28, 0.40), p(0.5, 0.62), p(0.72, 0.40)]);
        },
        Icon::ChevronRight => {
            line(vec![p(0.40, 0.28), p(0.62, 0.5), p(0.40, 0.72)]);
        },
        Icon::Clock => {
            painter.circle_stroke(b.center(), side * 0.34, stroke);
            line(vec![p(0.5, 0.28), p(0.5, 0.52), p(0.68, 0.62)]);
        },
        Icon::Person => {
            painter.circle_stroke(p(0.5, 0.34), side * 0.16, stroke);
            line(vec![
                p(0.22, 0.84),
                p(0.22, 0.72),
                p(0.5, 0.58),
                p(0.78, 0.72),
                p(0.78, 0.84),
            ]);
        },
        Icon::Speaker => {
            line(vec![
                p(0.20, 0.38),
                p(0.34, 0.38),
                p(0.52, 0.20),
                p(0.52, 0.80),
                p(0.34, 0.62),
                p(0.20, 0.62),
                p(0.20, 0.38),
            ]);
            seg(p(0.64, 0.36), p(0.64, 0.64));
            seg(p(0.78, 0.28), p(0.78, 0.72));
        },
        Icon::Warning => {
            line(vec![
                p(0.5, 0.14),
                p(0.9, 0.84),
                p(0.1, 0.84),
                p(0.5, 0.14),
            ]);
            seg(p(0.5, 0.40), p(0.5, 0.60));
            painter.circle_filled(p(0.5, 0.72), w * 0.6, color);
        },
        Icon::Info => {
            painter.circle_stroke(b.center(), side * 0.36, stroke);
            seg(p(0.5, 0.46), p(0.5, 0.70));
            painter.circle_filled(p(0.5, 0.32), w * 0.6, color);
        },
        Icon::Folder => {
            line(vec![
                p(0.14, 0.78),
                p(0.14, 0.24),
                p(0.42, 0.24),
                p(0.52, 0.36),
                p(0.86, 0.36),
                p(0.86, 0.78),
                p(0.14, 0.78),
            ]);
        },
    }
}
