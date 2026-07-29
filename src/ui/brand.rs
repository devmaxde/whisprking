//! The WhisprKing mark: a microphone wearing a crown.
//!
//! The mark has to work in four places at wildly different sizes — the
//! navigation rail (30 pt), the macOS menu bar (22 px, one flat color), the
//! app bundle icon (16 … 1024 px) and the window/dock icon. Shipping four
//! hand-drawn assets means four things to keep in sync, so the geometry is
//! defined exactly once here, as signed distance fields, and rasterized on
//! demand. Consequences: no binary assets in the repo, crisp anti-aliased
//! edges at every size, and a restyle is a constant away.
//!
//! Coordinates are normalized: the mark is drawn inside the unit square with
//! y pointing down, matching egui.

use egui::Color32;

use super::theme;

/// A point in mark space.
type P = [f32; 2];

fn sub(a: P, b: P) -> P {
    [a[0] - b[0], a[1] - b[1]]
}

fn dot(a: P, b: P) -> f32 {
    a[0] * b[0] + a[1] * b[1]
}

fn len(a: P) -> f32 {
    dot(a, a).sqrt()
}

/// Which drawing of the mark to use.
///
/// Below ~48 px the cradle, the stem and the base bar of the full mark land
/// on the same pixel row and turn into a grey smudge. `Compact` is the same
/// mark redrawn for that size: fewer parts, thicker strokes, a wider crown
/// and a bigger gap under it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    Full,
    Compact,
}

impl Variant {
    /// What to use at a given pixel size.
    pub fn for_size(px: u32) -> Self {
        if px < 48 {
            Variant::Compact
        } else {
            Variant::Full
        }
    }
}

/// One primitive of the mark. Every variant is a union of these, so the
/// distance to the mark is the minimum over all of them.
enum Shape {
    /// Segment `a`–`b` thickened by `r`, with round caps.
    Capsule { a: P, b: P, r: f32 },
    /// Axis-aligned box with corner radius `r`.
    RoundRect { min: P, max: P, r: f32 },
    /// Circular band around `c` of radius `ra` and half-thickness `rb`,
    /// symmetric about straight down, spanning `ap` radians to either side.
    Arc { c: P, ra: f32, rb: f32, ap: f32 },
    /// Simple polygon, grown by `round` — which also rounds its corners, so
    /// the crown's spikes stay solid instead of aliasing away at small sizes.
    Poly { pts: Vec<P>, round: f32 },
}

impl Shape {
    /// Signed distance from `p`: negative inside, in mark-space units.
    fn distance(&self, p: P) -> f32 {
        match self {
            Shape::Capsule { a, b, r } => {
                let pa = sub(p, *a);
                let ba = sub(*b, *a);
                let h = (dot(pa, ba) / dot(ba, ba)).clamp(0.0, 1.0);
                len([pa[0] - ba[0] * h, pa[1] - ba[1] * h]) - r
            },
            Shape::RoundRect { min, max, r } => {
                let c = [(min[0] + max[0]) * 0.5, (min[1] + max[1]) * 0.5];
                let half = [
                    (max[0] - min[0]) * 0.5 - r,
                    (max[1] - min[1]) * 0.5 - r,
                ];
                let q = [
                    (p[0] - c[0]).abs() - half[0],
                    (p[1] - c[1]).abs() - half[1],
                ];
                len([q[0].max(0.0), q[1].max(0.0)]) + q[0].max(q[1]).min(0.0) - r
            },
            Shape::Arc { c, ra, rb, ap } => {
                // Mirror into the right half so one branch covers both arms.
                let q = [(p[0] - c[0]).abs(), p[1] - c[1]];
                let (sin, cos) = (ap.sin(), ap.cos());
                let d = if cos * q[0] > sin * q[1] {
                    // Past the open end: distance to the arc's end cap.
                    len(sub(q, [sin * ra, cos * ra]))
                } else {
                    (len(q) - ra).abs()
                };
                d - rb
            },
            Shape::Poly { pts, round } => sd_poly(p, pts) - round,
        }
    }
}

/// Exact signed distance to a simple (possibly concave) polygon.
fn sd_poly(p: P, v: &[P]) -> f32 {
    let n = v.len();
    let mut d = dot(sub(p, v[0]), sub(p, v[0]));
    let mut sign = 1.0f32;
    let mut j = n - 1;
    for i in 0..n {
        let e = sub(v[j], v[i]);
        let w = sub(p, v[i]);
        let t = (dot(w, e) / dot(e, e)).clamp(0.0, 1.0);
        let b = [w[0] - e[0] * t, w[1] - e[1] * t];
        d = d.min(dot(b, b));
        // Flip the sign on every edge the downward ray from `p` crosses.
        let cond = [
            p[1] >= v[i][1],
            p[1] < v[j][1],
            e[0] * w[1] > e[1] * w[0],
        ];
        if cond.iter().all(|c| *c) || !cond.iter().any(|c| *c) {
            sign = -sign;
        }
        j = i;
    }
    sign * d.sqrt()
}

/// Distance to a superellipse — Apple's icon outline is much closer to
/// `|x|^5 + |y|^5 = 1` than to a rounded rectangle, and the difference is
/// visible next to other icons in the Dock.
fn sd_squircle(p: P, c: P, half: f32) -> f32 {
    const N: f32 = 5.0;
    let q = [(p[0] - c[0]).abs() / half, (p[1] - c[1]).abs() / half];
    (q[0].powf(N) + q[1].powf(N)).powf(1.0 / N) * half - half
}

// --- geometry ---------------------------------------------------------------

fn crown(pts: [P; 7], round: f32) -> Shape {
    Shape::Poly {
        pts: pts.to_vec(),
        round,
    }
}

fn shapes(variant: Variant) -> Vec<Shape> {
    match variant {
        // Bottom corners, left spike, valley, center spike, valley, right
        // spike — the crown is wider than the capsule so it reads as sitting
        // *on* the microphone rather than being part of it.
        Variant::Full => vec![
            crown(
                [
                    [0.268, 0.242],
                    [0.268, 0.112],
                    [0.385, 0.198],
                    [0.500, 0.050],
                    [0.615, 0.198],
                    [0.732, 0.112],
                    [0.732, 0.242],
                ],
                0.020,
            ),
            // capsule
            Shape::RoundRect {
                min: [0.380, 0.312],
                max: [0.620, 0.660],
                r: 0.120,
            },
            // cradle
            Shape::Arc {
                c: [0.500, 0.560],
                ra: 0.222,
                rb: 0.046,
                ap: 105f32.to_radians(),
            },
            // Stem. It starts far enough down that its round cap stays
            // buried inside the cradle — a cap poking through the gap under
            // the capsule reads as a stray nub at large sizes.
            Shape::Capsule {
                a: [0.500, 0.800],
                b: [0.500, 0.878],
                r: 0.040,
            },
            // base bar
            Shape::Capsule {
                a: [0.362, 0.912],
                b: [0.638, 0.912],
                r: 0.042,
            },
        ],
        // No base bar: at 22 px it is a single row of grey pixels under the
        // stem and reads as blur, not as a microphone stand.
        Variant::Compact => vec![
            crown(
                [
                    [0.262, 0.212],
                    [0.262, 0.098],
                    [0.381, 0.176],
                    [0.500, 0.042],
                    [0.619, 0.176],
                    [0.738, 0.098],
                    [0.738, 0.212],
                ],
                0.026,
            ),
            Shape::RoundRect {
                min: [0.378, 0.348],
                max: [0.622, 0.678],
                r: 0.122,
            },
            Shape::Arc {
                c: [0.500, 0.585],
                ra: 0.228,
                rb: 0.055,
                ap: 100f32.to_radians(),
            },
            Shape::Capsule {
                a: [0.500, 0.810],
                b: [0.500, 0.900],
                r: 0.047,
            },
        ],
    }
}

// --- rasterizing ------------------------------------------------------------

/// Coverage of the mark, one byte per pixel, row-major.
///
/// `margin` is the fraction of `px` left empty on each side; the mark is
/// drawn into what remains. Anti-aliasing is analytic — the distance field
/// gives the exact distance to the edge, so one sample per pixel is enough.
pub fn mark_alpha(px: u32, variant: Variant, margin: f32) -> Vec<u8> {
    let shapes = shapes(variant);
    let side = px as f32;
    let scale = 1.0 - 2.0 * margin;
    // Width of one pixel in mark-space units.
    let unit = 1.0 / (side * scale);

    let mut out = vec![0u8; (px * px) as usize];
    for y in 0..px {
        for x in 0..px {
            let p = [
                ((x as f32 + 0.5) / side - margin) / scale,
                ((y as f32 + 0.5) / side - margin) / scale,
            ];
            let d = shapes
                .iter()
                .map(|s| s.distance(p))
                .fold(f32::MAX, f32::min);
            let cover = (0.5 - d / unit).clamp(0.0, 1.0);
            out[(y * px + x) as usize] = (cover * 255.0).round() as u8;
        }
    }
    out
}

/// The mark in a single color on a transparent background — menu bar icon.
pub fn mark_rgba(px: u32, color: Color32, margin: f32) -> Vec<u8> {
    let alpha = mark_alpha(px, Variant::for_size(px), margin);
    let mut out = vec![0u8; alpha.len() * 4];
    for (i, a) in alpha.iter().enumerate() {
        out[i * 4] = color.r();
        out[i * 4 + 1] = color.g();
        out[i * 4 + 2] = color.b();
        out[i * 4 + 3] = *a;
    }
    out
}

/// Gradient stops of the icon background, top-left to bottom-right.
const ICON_TOP: Color32 = Color32::from_rgb(0x93, 0x9C, 0xFF);
const ICON_BOTTOM: Color32 = Color32::from_rgb(0x44, 0x4C, 0xD6);

/// How much of the canvas the rounded square occupies. macOS reserves the
/// rest for the shadow it draws itself; an icon that fills its canvas looks
/// oversized next to every other icon in the Dock.
const SQUIRCLE_SCALE: f32 = 824.0 / 1024.0;

/// How much of the canvas the mark itself occupies. The mark is a tall,
/// narrow glyph, so it needs a taller box than a square logo would to carry
/// the same visual weight inside the squircle.
///
/// Below 48 px the glyph gets more of the tile: at 16 px the normal
/// proportion leaves nine pixels of microphone, which is a smudge. Detail is
/// lost at that size either way, so trade the breathing room for legibility.
fn glyph_scale(px: u32) -> f32 {
    if px < 48 {
        0.70
    } else {
        0.58
    }
}

/// The full app icon: the mark in white on the indigo squircle.
/// Straight (non-premultiplied) RGBA, row-major.
pub fn app_icon_rgba(px: u32) -> Vec<u8> {
    icon_rgba(px, glyph_scale(px), Variant::for_size(px))
}

/// The icon with the proportions chosen explicitly — the in-app badge is
/// rendered large but *displayed* at 30 pt, so it wants the proportions of a
/// small icon even though it is rasterized at 128 px.
pub fn icon_rgba(
    px: u32,
    glyph: f32,
    variant: Variant,
) -> Vec<u8> {
    let margin = (1.0 - glyph) / 2.0;
    let alpha = mark_alpha(px, variant, margin);

    let side = px as f32;
    let unit = 1.0 / side;
    let half = SQUIRCLE_SCALE / 2.0;

    let mut out = vec![0u8; (px * px * 4) as usize];
    for y in 0..px {
        for x in 0..px {
            let i = ((y * px + x) * 4) as usize;
            let p = [(x as f32 + 0.5) / side, (y as f32 + 0.5) / side];

            let d = sd_squircle(p, [0.5, 0.5], half);
            let bg_a = (0.5 - d / unit).clamp(0.0, 1.0);
            let t = ((p[0] + p[1]) * 0.5).clamp(0.0, 1.0);
            let bg = theme::mix(ICON_TOP, ICON_BOTTOM, t);

            // Mark over background, both straight alpha.
            let mark_a = alpha[(y * px + x) as usize] as f32 / 255.0;
            let a = bg_a + mark_a * (1.0 - bg_a);
            if a <= 0.0 {
                continue;
            }
            let blend = |bg_c: u8| -> u8 {
                let v = (255.0 * mark_a + bg_c as f32 * bg_a * (1.0 - mark_a)) / a;
                v.round().clamp(0.0, 255.0) as u8
            };
            out[i] = blend(bg.r());
            out[i + 1] = blend(bg.g());
            out[i + 2] = blend(bg.b());
            out[i + 3] = (a * 255.0).round() as u8;
        }
    }
    out
}

// --- egui -------------------------------------------------------------------

/// Texture of the mark alone, tintable by the caller. Built once per context.
fn mark_texture(ctx: &egui::Context) -> egui::TextureHandle {
    texture(ctx, "wk-mark", || {
        const PX: u32 = 128;
        let rgba = mark_rgba(PX, Color32::WHITE, 0.04);
        egui::ColorImage::from_rgba_unmultiplied([PX as usize; 2], &rgba)
    })
}

/// Texture of the complete app icon — used wherever the app identifies
/// itself, so what the user sees in the rail is what they see in Finder.
fn badge_texture(ctx: &egui::Context) -> egui::TextureHandle {
    texture(ctx, "wk-badge", || {
        const PX: u32 = 128;
        let rgba = icon_rgba(PX, 0.68, Variant::Full);
        egui::ColorImage::from_rgba_unmultiplied([PX as usize; 2], &rgba)
    })
}

fn texture(
    ctx: &egui::Context,
    name: &'static str,
    build: impl FnOnce() -> egui::ColorImage,
) -> egui::TextureHandle {
    let id = egui::Id::new(name);
    if let Some(handle) = ctx.data(|d| d.get_temp::<egui::TextureHandle>(id)) {
        return handle;
    }
    let handle = ctx.load_texture(name, build(), egui::TextureOptions::LINEAR);
    ctx.data_mut(|d| d.insert_temp(id, handle.clone()));
    handle
}

/// Paint the mark inside `rect`, in `color`.
pub fn paint(
    ui: &egui::Ui,
    rect: egui::Rect,
    color: Color32,
) {
    let tex = mark_texture(ui.ctx());
    ui.painter().image(
        tex.id(),
        square(rect),
        egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)),
        color,
    );
}

/// Paint the app icon itself inside `rect`.
pub fn paint_badge(
    ui: &egui::Ui,
    rect: egui::Rect,
) {
    let tex = badge_texture(ui.ctx());
    ui.painter().image(
        tex.id(),
        square(rect),
        egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)),
        Color32::WHITE,
    );
}

/// The icon is square; a non-square rect would stretch it.
fn square(rect: egui::Rect) -> egui::Rect {
    let side = rect.width().min(rect.height());
    egui::Rect::from_center_size(rect.center(), egui::Vec2::splat(side))
}

/// Icon for the window and (on platforms that use it) the task switcher.
pub fn window_icon() -> egui::IconData {
    const PX: u32 = 256;
    egui::IconData {
        rgba: app_icon_rgba(PX),
        width: PX,
        height: PX,
    }
}

// --- .icns ------------------------------------------------------------------

/// The icon sizes an `.icns` carries, paired with the four-byte type macOS
/// looks them up by. Both members of a `@1x`/`@2x` pair hold the same pixel
/// data — that is how `iconutil` writes them too.
const ICNS_ENTRIES: [(&[u8; 4], u32); 10] = [
    (b"icp4", 16),
    (b"icp5", 32),
    (b"ic11", 32),
    (b"ic12", 64),
    (b"ic07", 128),
    (b"ic13", 256),
    (b"ic08", 256),
    (b"ic14", 512),
    (b"ic09", 512),
    (b"ic10", 1024),
];

fn png(px: u32) -> Result<Vec<u8>, String> {
    use image::ImageEncoder as _;

    let rgba = app_icon_rgba(px);
    let mut out = Vec::new();
    image::codecs::png::PngEncoder::new(&mut out)
        .write_image(&rgba, px, px, image::ExtendedColorType::Rgba8)
        .map_err(|e| format!("PNG encode ({px}px): {e}"))?;
    Ok(out)
}

/// Assemble an `.icns` archive: a header, then one length-prefixed PNG per
/// entry. Written by hand so the icon can be produced on any machine — the
/// alternative, `iconutil`, only exists on macOS.
pub fn icns() -> Result<Vec<u8>, String> {
    let mut body = Vec::new();
    let mut cache: Vec<(u32, Vec<u8>)> = Vec::new();
    for (kind, px) in ICNS_ENTRIES {
        let data = match cache.iter().find(|(size, _)| *size == px) {
            Some((_, data)) => data.clone(),
            None => {
                let data = png(px)?;
                cache.push((px, data.clone()));
                data
            },
        };
        body.extend_from_slice(kind);
        body.extend_from_slice(&(data.len() as u32 + 8).to_be_bytes());
        body.extend_from_slice(&data);
    }

    let mut out = Vec::with_capacity(body.len() + 8);
    out.extend_from_slice(b"icns");
    out.extend_from_slice(&(body.len() as u32 + 8).to_be_bytes());
    out.extend_from_slice(&body);
    Ok(out)
}

/// Write the app icon to `path` as an `.icns`. Called by `just bundle`.
pub fn write_icns(path: &std::path::Path) -> Result<(), String> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    std::fs::write(path, icns()?).map_err(|e| format!("{}: {e}", path.display()))
}

/// Write every icon size to `dir` as separate PNGs, plus the flat mark used
/// in the menu bar. For inspecting the artwork without building a bundle.
pub fn write_pngs(dir: &std::path::Path) -> Result<Vec<std::path::PathBuf>, String> {
    use image::ImageEncoder as _;

    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut written = Vec::new();

    let mut sizes: Vec<u32> = ICNS_ENTRIES.iter().map(|(_, px)| *px).collect();
    sizes.sort_unstable();
    sizes.dedup();
    for px in sizes {
        let path = dir.join(format!("icon_{px}.png"));
        std::fs::write(&path, png(px)?).map_err(|e| format!("{}: {e}", path.display()))?;
        written.push(path);
    }

    for px in [22u32, 44] {
        let path = dir.join(format!("menubar_{px}.png"));
        let rgba = mark_rgba(px, Color32::WHITE, MENUBAR_MARGIN);
        let mut out = Vec::new();
        image::codecs::png::PngEncoder::new(&mut out)
            .write_image(&rgba, px, px, image::ExtendedColorType::Rgba8)
            .map_err(|e| format!("PNG encode (menu bar {px}px): {e}"))?;
        std::fs::write(&path, out).map_err(|e| format!("{}: {e}", path.display()))?;
        written.push(path);
    }
    Ok(written)
}

/// Padding around the menu bar glyph. macOS menu bar icons do not touch the
/// edges of their box.
pub const MENUBAR_MARGIN: f32 = 0.07;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mark_is_drawn_and_bounded() {
        let px = 64;
        let a = mark_alpha(px, Variant::Full, 0.05);
        assert_eq!(a.len(), (px * px) as usize);
        assert!(a.iter().any(|v| *v == 255), "mark has solid pixels");
        assert!(a.iter().any(|v| *v == 0), "mark does not fill its box");

        // The margin has to stay empty, otherwise the glyph is clipped in
        // the menu bar and against the icon's rounded corners.
        let row = |y: u32| (0..px).all(|x| a[(y * px + x) as usize] == 0);
        let col = |x: u32| (0..px).all(|y| a[(y * px + x) as usize] == 0);
        assert!(row(0) && row(px - 1), "top/bottom row clear");
        assert!(col(0) && col(px - 1), "left/right column clear");
    }

    #[test]
    fn crown_sits_above_the_capsule() {
        // A horizontal band between the crown and the microphone must be
        // empty, or the two merge into one blob at small sizes.
        let px = 256;
        let a = mark_alpha(px, Variant::Compact, 0.05);
        let filled = |y: u32| (0..px).any(|x| a[(y * px + x) as usize] > 0);
        let gap = (0..px).filter(|y| !filled(*y)).collect::<Vec<_>>();
        assert!(
            gap.iter().any(|y| *y > px / 8 && *y < px / 2),
            "expected a clear gap under the crown, rows: {gap:?}"
        );
    }

    #[test]
    fn icon_corners_are_transparent_and_center_is_opaque() {
        let px = 64;
        let rgba = app_icon_rgba(px);
        assert_eq!(rgba.len(), (px * px * 4) as usize);
        let alpha = |x: u32, y: u32| rgba[((y * px + x) * 4 + 3) as usize];
        assert_eq!(alpha(0, 0), 0, "canvas corner is outside the squircle");
        assert_eq!(alpha(px / 2, px / 2), 255, "icon body is opaque");
    }

    #[test]
    fn icns_has_a_header_and_every_entry() {
        let data = icns().expect("icns");
        assert_eq!(&data[0..4], b"icns");
        let declared = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);
        assert_eq!(declared as usize, data.len(), "length field covers the file");

        // Walk the entries the way macOS does.
        let mut at = 8usize;
        let mut seen = 0;
        while at < data.len() {
            let size = u32::from_be_bytes([
                data[at + 4],
                data[at + 5],
                data[at + 6],
                data[at + 7],
            ]) as usize;
            assert!(size >= 8 && at + size <= data.len(), "entry fits");
            assert_eq!(&data[at + 8..at + 16], b"\x89PNG\r\n\x1a\n", "PNG payload");
            at += size;
            seen += 1;
        }
        assert_eq!(seen, ICNS_ENTRIES.len());
    }
}
