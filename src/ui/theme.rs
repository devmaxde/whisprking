//! Design tokens and the egui style built from them.
//!
//! Everything visual funnels through here: colors, spacing, corner radii,
//! type scale, fonts. Pages never hardcode a color or a pixel value — they
//! ask this module — which is what keeps the surfaces consistent and makes a
//! restyle a one-file change.
//!
//! Two things are worth knowing:
//!
//! * **Weights are real.** egui's bundled font ships a single (light) weight,
//!   which is why the old UI had no typographic hierarchy — `strong()` only
//!   brightened the text color. On macOS we load the system UI font (a
//!   variable font) three times at different `wght` coordinates and register
//!   them as separate families, so headings are genuinely heavier than body
//!   text. If no system font can be read we fall back to egui's default and
//!   the app still runs, just flatter.
//! * **Surfaces are layered.** `BG_SUNKEN` < `BG_BASE` < `BG_SURFACE` <
//!   `BG_ELEVATED`. Depth comes from that ladder plus hairline borders, not
//!   from heavy strokes and gray boxes.

use egui::{Color32, FontFamily, FontId, Margin, Stroke};

// --- palette ---------------------------------------------------------------

/// Nav rail / chrome: the darkest surface.
pub const BG_SUNKEN: Color32 = Color32::from_rgb(0x0b, 0x0c, 0x10);
/// Window and page background.
pub const BG_BASE: Color32 = Color32::from_rgb(0x11, 0x12, 0x18);
/// Cards sitting on the page.
pub const BG_SURFACE: Color32 = Color32::from_rgb(0x18, 0x1a, 0x22);
/// Rows/controls sitting on a card, and hover states.
pub const BG_ELEVATED: Color32 = Color32::from_rgb(0x20, 0x22, 0x2c);
/// Pressed / strongly hovered.
pub const BG_ACTIVE: Color32 = Color32::from_rgb(0x2a, 0x2d, 0x3a);
/// Inset fields (text edits, meters, code blocks).
pub const BG_INPUT: Color32 = Color32::from_rgb(0x0d, 0x0e, 0x13);

/// Hairline between surfaces.
pub const BORDER: Color32 = Color32::from_rgb(0x25, 0x27, 0x31);
/// Border of an interactive thing at rest.
pub const BORDER_STRONG: Color32 = Color32::from_rgb(0x33, 0x36, 0x44);

pub const TEXT_PRIMARY: Color32 = Color32::from_rgb(0xea, 0xeb, 0xf1);
pub const TEXT_SECONDARY: Color32 = Color32::from_rgb(0x9b, 0xa0, 0xb0);
pub const TEXT_MUTED: Color32 = Color32::from_rgb(0x6b, 0x70, 0x82);
/// Timestamps and other tabular metadata.
pub const TEXT_TIMESTAMP: Color32 = Color32::from_rgb(0x64, 0x69, 0x7c);
/// Text on top of a filled accent surface.
pub const TEXT_ON_ACCENT: Color32 = Color32::from_rgb(0xff, 0xff, 0xff);

pub const ACCENT: Color32 = Color32::from_rgb(0x6e, 0x7b, 0xff);
pub const ACCENT_HOVER: Color32 = Color32::from_rgb(0x8b, 0x95, 0xff);
pub const ACCENT_PRESSED: Color32 = Color32::from_rgb(0x5a, 0x67, 0xe8);

pub const SUCCESS: Color32 = Color32::from_rgb(0x3e, 0xcf, 0x8e);
pub const WARNING: Color32 = Color32::from_rgb(0xf0, 0xa9, 0x3b);
pub const DANGER: Color32 = Color32::from_rgb(0xff, 0x5a, 0x52);
pub const INFO: Color32 = Color32::from_rgb(0x55, 0xa9, 0xff);

/// "You" — the microphone track.
pub const TRACK_MINE: Color32 = ACCENT_HOVER;
/// "Others" — the system-audio track.
pub const TRACK_THEIRS: Color32 = Color32::from_rgb(0x2f, 0xc4, 0xb2);

/// Blend `top` over `bottom` at `t` (0 = bottom, 1 = top).
pub fn mix(bottom: Color32, top: Color32, t: f32) -> Color32 {
    bottom.lerp_to_gamma(top, t.clamp(0.0, 1.0))
}

/// A translucent wash of `color` — used for selected rows, badges, and the
/// tinted background of an active nav item.
pub fn tint(color: Color32, alpha: f32) -> Color32 {
    Color32::from_rgba_unmultiplied(
        color.r(),
        color.g(),
        color.b(),
        (alpha.clamp(0.0, 1.0) * 255.0) as u8,
    )
}

// --- metrics ---------------------------------------------------------------

/// 4-point spacing scale. Use these instead of ad-hoc numbers.
pub mod space {
    pub const XXS: f32 = 2.0;
    pub const XS: f32 = 4.0;
    pub const SM: f32 = 8.0;
    pub const MD: f32 = 12.0;
    pub const LG: f32 = 16.0;
    pub const XL: f32 = 24.0;
    pub const XXL: f32 = 32.0;
}

pub mod radius {
    use egui::CornerRadius;
    pub const SM: CornerRadius = CornerRadius::same(6);
    pub const MD: CornerRadius = CornerRadius::same(10);
    pub const LG: CornerRadius = CornerRadius::same(14);
    /// Fully rounded — `u8` caps at 255, which is plenty for a pill.
    pub const PILL: CornerRadius = CornerRadius::same(255);
}

/// Height of a standard control (button, combo, field).
pub const CONTROL_HEIGHT: f32 = 30.0;
/// Height of a prominent control (the record button).
pub const CONTROL_HEIGHT_LG: f32 = 38.0;
/// Width of the left navigation rail.
pub const NAV_WIDTH: f32 = 212.0;
/// Width of the transcript list in the history page.
pub const LIST_WIDTH: f32 = 300.0;

pub const OVERLAY_WIDTH: f32 = 268.0;
pub const OVERLAY_HEIGHT: f32 = 62.0;
pub const OVERLAY_BOTTOM_OFFSET: f32 = 72.0;

pub const WINDOW_DEFAULT_SIZE: [f32; 2] = [1020.0, 680.0];
pub const WINDOW_MIN_SIZE: [f32; 2] = [780.0, 520.0];

// --- typography ------------------------------------------------------------

/// Family name of the medium-weight face.
const FAMILY_MEDIUM: &str = "wk-medium";
/// Family name of the semibold face.
const FAMILY_SEMIBOLD: &str = "wk-semibold";

fn family(name: &str) -> FontFamily {
    FontFamily::Name(name.into())
}

pub fn font_medium(size: f32) -> FontId {
    FontId::new(size, family(FAMILY_MEDIUM))
}

pub fn font_semibold(size: f32) -> FontId {
    FontId::new(size, family(FAMILY_SEMIBOLD))
}

pub fn font_regular(size: f32) -> FontId {
    FontId::new(size, FontFamily::Proportional)
}

pub fn font_mono(size: f32) -> FontId {
    FontId::new(size, FontFamily::Monospace)
}

/// The type scale. Sizes are paired with a weight on purpose — call these
/// rather than picking a size and hoping.
pub mod text {
    use super::{font_medium, font_mono, font_regular, font_semibold};
    use egui::FontId;

    /// Page title.
    pub fn display() -> FontId {
        font_semibold(21.0)
    }
    /// Card / section title.
    pub fn title() -> FontId {
        font_semibold(15.0)
    }
    /// Emphasised body, nav items, buttons.
    pub fn strong() -> FontId {
        font_medium(13.5)
    }
    pub fn body() -> FontId {
        font_regular(13.5)
    }
    /// Secondary explanatory text.
    pub fn small() -> FontId {
        font_regular(12.0)
    }
    /// Labels above fields, badge text.
    pub fn caption() -> FontId {
        font_medium(11.0)
    }
    /// Timestamps, durations, byte counts.
    pub fn mono() -> FontId {
        font_mono(12.0)
    }
    pub fn mono_large() -> FontId {
        font_mono(19.0)
    }
}

/// Candidate UI font files, best first. macOS ships SF Pro as a variable
/// font; the rest are fallbacks for other machines (mostly so a dev build on
/// Linux does not look broken). `WHISPRKING_FONT` / `WHISPRKING_FONT_MONO`
/// override the search with an explicit file.
const UI_FONT_CANDIDATES: &[&str] = &[
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/SFNSDisplay.ttf",
    "/System/Library/Fonts/SFNSText.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/inter/InterVariable.ttf",
    "/usr/share/fonts/truetype/inter/Inter-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
];

const MONO_FONT_CANDIDATES: &[&str] = &[
    "/System/Library/Fonts/SFNSMono.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
];

/// Read the first readable file from `env_override` (if set) or `paths`.
fn read_first(env_override: &str, paths: &[&'static str]) -> Option<(String, Vec<u8>)> {
    let from_env = std::env::var(env_override).ok().filter(|p| !p.is_empty());
    let candidates = from_env
        .iter()
        .map(String::as_str)
        .chain(paths.iter().copied());

    for path in candidates {
        match std::fs::read(path) {
            Ok(bytes) if !bytes.is_empty() => {
                log::info!("theme: using font {path}");
                return Some((path.to_owned(), bytes));
            },
            _ => {},
        }
    }
    log::info!("theme: no system font found — falling back to egui's bundled font");
    None
}

/// Register the UI fonts. Always registers the `wk-medium` / `wk-semibold`
/// families — pointing them at the fallback font if we could not read a
/// system one — so [`font_semibold`] can never reference a missing family.
fn install_fonts(ctx: &egui::Context) {
    use egui::epaint::text::VariationCoords;
    use egui::{FontData, FontDefinitions, FontTweak};

    let mut fonts = FontDefinitions::default();

    let ui_font = read_first("WHISPRKING_FONT", UI_FONT_CANDIDATES);
    let mono_font = read_first("WHISPRKING_FONT_MONO", MONO_FONT_CANDIDATES);

    // Register the same file three times at different weights. On a static
    // font the coords are simply ignored and every family looks the same,
    // which is exactly the graceful degradation we want.
    if let Some((path, bytes)) = &ui_font {
        if FontData::from_owned(bytes.clone()).variation_axes().is_empty() {
            log::info!("theme: {path} is not a variable font — weights will look flat");
        }
        for (name, weight) in [
            ("wk-regular", 400.0_f32),
            ("wk-medium", 510.0),
            ("wk-semibold", 620.0),
        ] {
            let data = FontData::from_owned(bytes.clone()).tweak(FontTweak {
                coords: VariationCoords::new([("wght", weight)]),
                ..Default::default()
            });
            fonts.font_data.insert(name.to_owned(), std::sync::Arc::new(data));
        }
    }

    if let Some((_, bytes)) = &mono_font {
        fonts.font_data.insert(
            "wk-mono".to_owned(),
            std::sync::Arc::new(FontData::from_owned(bytes.clone())),
        );
    }

    // Proportional: our regular first, egui's bundled font behind it as a
    // glyph fallback, then the emoji fonts that were already there.
    if ui_font.is_some() {
        if let Some(list) = fonts.families.get_mut(&FontFamily::Proportional) {
            list.insert(0, "wk-regular".to_owned());
        }
    }
    if mono_font.is_some() {
        if let Some(list) = fonts.families.get_mut(&FontFamily::Monospace) {
            list.insert(0, "wk-mono".to_owned());
        }
    }

    let proportional = fonts
        .families
        .get(&FontFamily::Proportional)
        .cloned()
        .unwrap_or_default();

    for (family_name, font_name) in [
        (FAMILY_MEDIUM, "wk-medium"),
        (FAMILY_SEMIBOLD, "wk-semibold"),
    ] {
        let mut list = Vec::new();
        if ui_font.is_some() {
            list.push(font_name.to_owned());
        }
        list.extend(proportional.iter().cloned());
        fonts.families.insert(family(family_name), list);
    }

    ctx.set_fonts(fonts);
}

// --- style -----------------------------------------------------------------

/// Apply fonts, colors, spacing and widget geometry. Call once at startup.
///
/// The app is dark-only, so the same overrides go into both the dark and the
/// light [`egui::Style`] and the theme preference is pinned to dark — that way
/// a popup created before the first frame cannot come up in default egui gray.
pub fn apply_theme(ctx: &egui::Context) {
    install_fonts(ctx);
    ctx.set_theme(egui::ThemePreference::Dark);
    ctx.all_styles_mut(style_overrides);
}

fn style_overrides(style: &mut egui::Style) {
    style.text_styles = [
        (egui::TextStyle::Heading, text::display()),
        (egui::TextStyle::Body, text::body()),
        (egui::TextStyle::Button, text::strong()),
        (egui::TextStyle::Small, text::small()),
        (egui::TextStyle::Monospace, text::mono()),
    ]
    .into();

    let s = &mut style.spacing;
    s.item_spacing = egui::vec2(space::SM, space::SM);
    s.button_padding = egui::vec2(space::MD, 6.0);
    s.interact_size = egui::vec2(28.0, CONTROL_HEIGHT);
    s.icon_width = 16.0;
    s.icon_width_inner = 9.0;
    s.icon_spacing = space::SM;
    s.indent = 18.0;
    s.combo_width = 140.0;
    s.text_edit_width = 260.0;
    s.window_margin = Margin::same(space::LG as i8);
    s.menu_margin = Margin::symmetric(space::XS as i8, space::XS as i8);
    s.tooltip_width = 320.0;

    // Thin, floating, unobtrusive scrollbars.
    s.scroll.floating = true;
    s.scroll.bar_width = 8.0;
    s.scroll.floating_width = 6.0;
    s.scroll.floating_allocated_width = 0.0;
    s.scroll.bar_inner_margin = 2.0;
    s.scroll.bar_outer_margin = 0.0;
    s.scroll.handle_min_length = 24.0;
    s.scroll.dormant_handle_opacity = 0.0;
    s.scroll.active_handle_opacity = 0.5;
    s.scroll.interact_handle_opacity = 0.8;
    s.scroll.dormant_background_opacity = 0.0;
    s.scroll.active_background_opacity = 0.0;
    s.scroll.interact_background_opacity = 0.2;

    style.interaction.tooltip_delay = 0.35;
    style.animation_time = 0.10;

    let v = &mut style.visuals;
    v.dark_mode = true;
    v.panel_fill = BG_BASE;
    v.window_fill = BG_SURFACE;
    v.extreme_bg_color = BG_INPUT;
    v.text_edit_bg_color = Some(BG_INPUT);
    v.faint_bg_color = BG_ELEVATED;
    v.code_bg_color = BG_INPUT;
    v.override_text_color = None;
    v.weak_text_color = Some(TEXT_SECONDARY);
    v.hyperlink_color = ACCENT_HOVER;
    v.warn_fg_color = WARNING;
    v.error_fg_color = DANGER;
    v.window_corner_radius = radius::LG;
    v.menu_corner_radius = radius::MD;
    v.window_stroke = Stroke::new(1.0, BORDER);
    v.window_shadow = egui::epaint::Shadow {
        offset: [0, 8],
        blur: 24,
        spread: 0,
        color: Color32::from_black_alpha(120),
    };
    v.popup_shadow = egui::epaint::Shadow {
        offset: [0, 6],
        blur: 18,
        spread: 0,
        color: Color32::from_black_alpha(110),
    };
    v.selection.bg_fill = tint(ACCENT, 0.35);
    v.selection.stroke = Stroke::new(1.0, TEXT_PRIMARY);
    v.text_cursor.stroke = Stroke::new(2.0, ACCENT_HOVER);
    v.interact_cursor = Some(egui::CursorIcon::PointingHand);
    v.button_frame = true;
    v.collapsing_header_frame = false;
    v.indent_has_left_vline = false;
    v.striped = false;
    v.disabled_alpha = 0.45;
    v.clip_rect_margin = 0.0;
    v.resize_corner_size = 12.0;

    let w = &mut v.widgets;
    w.noninteractive.bg_fill = BG_SURFACE;
    w.noninteractive.weak_bg_fill = BG_SURFACE;
    w.noninteractive.bg_stroke = Stroke::new(1.0, BORDER);
    w.noninteractive.fg_stroke = Stroke::new(1.0, TEXT_PRIMARY);
    w.noninteractive.corner_radius = radius::MD;
    w.noninteractive.expansion = 0.0;

    w.inactive.bg_fill = BG_ELEVATED;
    w.inactive.weak_bg_fill = BG_ELEVATED;
    w.inactive.bg_stroke = Stroke::new(1.0, BORDER);
    w.inactive.fg_stroke = Stroke::new(1.0, TEXT_PRIMARY);
    w.inactive.corner_radius = radius::SM;
    w.inactive.expansion = 0.0;

    w.hovered.bg_fill = BG_ACTIVE;
    w.hovered.weak_bg_fill = BG_ACTIVE;
    w.hovered.bg_stroke = Stroke::new(1.0, BORDER_STRONG);
    w.hovered.fg_stroke = Stroke::new(1.0, TEXT_PRIMARY);
    w.hovered.corner_radius = radius::SM;
    w.hovered.expansion = 0.0;

    w.active.bg_fill = mix(BG_ACTIVE, ACCENT, 0.18);
    w.active.weak_bg_fill = mix(BG_ACTIVE, ACCENT, 0.18);
    w.active.bg_stroke = Stroke::new(1.0, tint(ACCENT, 0.7));
    w.active.fg_stroke = Stroke::new(1.0, TEXT_PRIMARY);
    w.active.corner_radius = radius::SM;
    w.active.expansion = 0.0;

    w.open.bg_fill = BG_ACTIVE;
    w.open.weak_bg_fill = BG_ACTIVE;
    w.open.bg_stroke = Stroke::new(1.0, tint(ACCENT, 0.6));
    w.open.fg_stroke = Stroke::new(1.0, TEXT_PRIMARY);
    w.open.corner_radius = radius::SM;
}

/// Frame for a content card.
pub fn card_frame() -> egui::Frame {
    egui::Frame::new()
        .fill(BG_SURFACE)
        .stroke(Stroke::new(1.0, BORDER))
        .corner_radius(radius::LG)
        .inner_margin(Margin::same(space::LG as i8))
}

/// Frame for a card that only holds a single row of controls.
pub fn card_frame_tight() -> egui::Frame {
    card_frame().inner_margin(Margin::symmetric(space::MD as i8, space::MD as i8))
}

/// Frame for an inset region inside a card (transcript body, code, meters).
pub fn inset_frame() -> egui::Frame {
    egui::Frame::new()
        .fill(BG_INPUT)
        .stroke(Stroke::new(1.0, BORDER))
        .corner_radius(radius::MD)
        .inner_margin(Margin::same(space::MD as i8))
}
