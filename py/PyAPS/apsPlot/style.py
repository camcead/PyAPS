"""Shared colour palette and layout defaults for PyAPS Plotly figures."""

import numpy as np

# Deliberately back to the original literal near-black — a brief detour
# through a fixed medium-slate (see git history if curious) was reverted:
# this constant isn't only a live-app concern. `apsPlot.redrock.
# make_rrplot` (imported by `aps_rr.py`) and the analogous RVS/FERRE/
# PPXF/EMI static-diagnostic-PNG builders all funnel through the same
# `spectrum_overlay_figure()` this powers, and those are real pipeline
# outputs (kaleido-rendered, always on a plain white background,
# consumed by scientists outside this Dash app entirely) — explicit
# follow-up request: "make sure if you touch any plot, it does not
# disturb the basic plotting function use[d] by aps modules like
# aps_exGal or aps_Gal or aps_rr or so." Near-black is genuinely the
# right choice there and always was; it only became a problem inside the
# *live app's* new dark-mode plot backgrounds. Fixed at the correct
# layer instead: assets/explorer_ui.js's themePlotlyFigures() now
# detects near-black trace/marker colours itself (a generic luminance
# check, not a name/module-specific one) and restyles just those, live,
# client-side-only, only inside the browser — so this constant (and
# every static/offline consumer of it) never changes at all.
DATA_COLOR = "#1a1a1a"

# Colour per fit rank (rank 0 = best fit). Reused for any module that
# overlays more than one candidate model (currently only Redrock).
RANK_COLORS = ["#d62728", "#ff7f0e", "#9467bd", "#2ca02c", "#17becf"]

IVAR_COLOR = "#4c78a8"
MASK_COLOR = "#d62728"
RESIDUAL_COLOR = "#2ca02c"

FONT_FAMILY = "Helvetica, Arial, sans-serif"

BASE_LAYOUT = dict(
    template="plotly_white",
    font=dict(family=FONT_FAMILY, size=12),
    margin=dict(l=60, r=30, t=60, b=40),
    hovermode="x unified",
)

# The astronomical-image-viewing "stretch" options (per explicit user
# request: "is it possible to have linear, log and power and most popular
# options we have in Astronomy for the colour viewing?") — the same
# families ds9/astropy.visualization offer (LinearStretch/LogStretch/
# SqrtStretch/PowerStretch/AsinhStretch). Lives here (not in
# aps_explorer.py, where it originated) so aps_l1_preview.py's own plots
# (e.g. the Slit Explorer) can reuse the exact same stretch a user has
# picked for the Aladin overlay without aps_l1_preview.py needing to
# import aps_explorer.py (which would be backwards — aps_explorer.py
# imports the three viewer modules, not the other way around).
# aps_explorer.COLOR_SCALE_OPTIONS/_apply_color_scale are thin aliases
# for these, kept for the tests/call sites that already reference them
# under those names.
COLOR_SCALE_OPTIONS = {
    "linear": "Linear",
    "log": "Log",
    "sqrt": "Sqrt",
    "power": "Power (^2)",
    "asinh": "Asinh",
}


def apply_color_scale(frac, scale):
    """Re-maps an already-vmin/vmax-normalized, [0,1]-clipped fraction
    array through the chosen stretch — applied to the *normalized*
    fraction rather than the raw data so it works identically regardless
    of the quantity's own units/range, and needs no extra parameters per
    quantity. Matches astropy.visualization's stretch conventions
    (LogStretch/PowerStretch(2)/AsinhStretch's default linear_width=0.1)
    closely enough for a display-only colour mapping — this does not
    claim exact numerical equivalence, just the same visual family."""
    if scale == "log":
        a = 1000.0
        return np.log(a * frac + 1.0) / np.log(a + 1.0)
    if scale == "sqrt":
        return np.sqrt(frac)
    if scale == "power":
        return frac ** 2
    if scale == "asinh":
        a = 0.1
        return np.arcsinh(frac / a) / np.arcsinh(1.0 / a)
    return frac  # "linear" (or anything unrecognized — never fail closed)


def stretch_for_marker_color(data, vmin, vmax, scale):
    """`(color_values, cmin, cmax)` for a Plotly `marker=dict(color=...,
    cmin=..., cmax=...)` continuous colour mapping, with `scale`'s own
    stretch (see `apply_color_scale` above) applied — the multi-panel L2
    map feature's own per-panel "Scale" control, extending
    `apsPlot/spaxel_map.py`'s `spaxel_map_figure` and its MOS counterpart
    `apsPlot/source_map.py`'s `source_map_figure`, both of which only
    ever supported a plain linear `vmin`/`vmax` range before.

    For `scale == "linear"` this is a no-op passthrough (`data, vmin,
    vmax` unchanged) — exactly `spaxel_map_figure`'s own pre-existing
    behaviour, so every already-working linear call site is untouched.
    For any other scale, normalizes `data` to `[vmin, vmax] -> [0, 1]`,
    applies `apply_color_scale`'s stretch (which is itself defined on
    exactly that `[0, 1]` domain), and returns the stretched values with
    `cmin=0, cmax=1` — Plotly's own colour mapping then just linearly
    interpolates the chosen colorscale across that stretched range,
    matching how the Aladin/flux-cube colour bucketing elsewhere in this
    app already treats a "stretch" (a display-only remap of already
    -normalized [0,1] fractions, per `apply_color_scale`'s own
    docstring — never claimed to be exact numerically, same family).

    Callers **must not** show `color_values` itself in hover text once
    `scale != "linear"` — it's in stretched [0,1] space, not the
    quantity's real units. Pass the original `data` separately (Plotly's
    `text=`/`customdata`, not `marker.color`) for hover display, and
    expect the colorbar's own native tick labels to read 0-1 rather than
    real units for a non-linear scale — a known, accepted simplification
    consistent with `apply_color_scale`'s own "visual family, not exact
    equivalence" scope.
    """
    if scale == "linear" or scale is None:
        return data, vmin, vmax
    span = (vmax - vmin) or 1.0
    frac = np.clip((np.asarray(data, dtype=np.float64) - vmin) / span, 0.0, 1.0)
    return apply_color_scale(frac, scale), 0.0, 1.0

# Plotly dcc.Graph `config` prop, shared by every spectra/fit/map plot
# across L1/IFU/MOS (aps_l1_preview.py, aps_IFUviewer.py, aps_MOSviewer.py)
# — per explicit user request ("we had many options to deal with spectra
# plots like right click to set range and so, they are mostly disabled...
# add them all"). Nothing in this codebase was actually found disabling
# Plotly's interactivity (no dragmode/modeBarButtonsToRemove/staticPlot
# anywhere — confirmed by grep before writing this), but the old bare
# {"scrollZoom": True} left everything else at Plotly's own conservative
# default: a toolbar that only appears on hover, with just the basic six
# buttons. This turns on the fuller set Plotly actually ships: an
# always-visible toolbar, shape-drawing tools (draw/erase a line or a box
# directly on a spectrum — the closest built-in equivalent to "set a
# range" — draggable/resizable afterwards via `edits`), hover-compare
# mode, per-trace spike lines for precise cross-trace reading, and a
# cleaner (2x-scale) PNG export, on top of the scroll-to-zoom already on.
GRAPH_CONFIG = {
    "scrollZoom": True,
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToAdd": [
        "drawline", "drawopenpath", "drawrect", "eraseshape",
        "hoverclosest", "hovercompare", "toggleSpikelines",
    ],
    "doubleClick": "reset+autosize",
    "edits": {"shapePosition": True},
    "toImageButtonOptions": {"format": "png", "scale": 2},
    # Lets Plotly re-fit the plot to its container on any later resize
    # (sidebar collapse/expand, browser resize) — the *initial* fill
    # (see fill_container_width below) already happens without this, but
    # this keeps it correct afterwards too.
    "responsive": True,
}


def rank_color(rank: int) -> str:
    return RANK_COLORS[rank % len(RANK_COLORS)]


def fill_container_width(fig):
    """Clear a figure's fixed pixel `layout.width`, so the *interactive*
    Dash view stretches to fill its container instead of leaving empty
    space beside a fixed-width plot (explicit user report: "the right
    side of the flux plot is empty within the spectra tab... the same
    with FWHM and the same with L2 plots").

    `apsPlot.spectra.spectrum_overlay_figure` and `apsPlot.fwhm.*` both
    default to a fixed `width` (1100/900px) because their figures are
    *also* used for static pipeline-diagnostic PNG export via
    `fig.write_image()` (aps_rr.make_rrplot and the analogous RVS/FERRE/
    PPXF/EMI/redrock/line-strength exporters) — kaleido needs a concrete
    pixel size, and that size should stay stable regardless of whatever
    this app's Dash view happens to do. So the fixed default is kept
    as-is at the source; call sites that build a figure purely for this
    app's own live display call this afterwards instead of threading a
    "make it responsive" flag back through every builder's signature.

    Setting `width=None` (rather than just leaving it out of this call)
    is what actually clears a previously-assigned value — Plotly
    property semantics treat `None` as "unset", not literally "None" —
    and `autosize=True` is what tells Plotly.js to compute the now-unset
    width from the container on render (`height` is left untouched: each
    builder already sizes it deliberately, from its own row/panel count).
    """
    fig.update_layout(width=None, autosize=True)
    return fig


# --------------------------------------------------------------------------- #
# Shared dash_table.DataTable / dcc.Tabs styling — a single "modern" look
# reused everywhere a DataTable or Tabs shows up (aps_explorer.py,
# aps_IFUviewer.py, aps_MOSviewer.py, aps_l1_preview.py) rather than each
# module's own plain-text-on-white default, per explicit user request for
# a more graphical/modern design instead of "just text."
# --------------------------------------------------------------------------- #

# var(--pyaps-accent) etc. below are CSS custom properties defined in the new
# py/PyAPS/assets/style.css (light + dark values, see that file's own
# header comment) — Dash's inline style={} dicts render straight to the
# DOM's style attribute, so var() resolves reactively in the browser
# exactly like real CSS. Confirmed before making this change: every
# consumer of these constants (grepped across the whole py/PyAPS/ tree)
# feeds a genuine Dash HTML/component style — dcc.Tab, dash_table.
# DataTable, html.Button/Div — never a Plotly figure spec, which
# wouldn't be able to resolve a CSS variable at all (FONT_FAMILY above,
# which *is* also used inside a Plotly layout font, is deliberately left
# as a plain string, not tokenized, for exactly this reason).
TABLE_HEADER_COLOR = "var(--pyaps-accent)"

# DataTable headers deliberately do NOT use TABLE_HEADER_COLOR (accent)
# for their own background any more, even though TAB_SELECTED_STYLE
# below still does — explicit design critique: "Choose one semantic
# accent... reserved for selected tabs, focus, and interactive
# controls... Avoid using it decoratively elsewhere." A table header is
# neither selected, focused, nor a control; filling every single table's
# header row with a full accent-coloured background was exactly the kind
# of "elsewhere" that critique named, confirmed by grep (this was the
# accent's only real *decorative* use anywhere in the app — every other
# call site is a genuine selection/focus/active-control state). Now a
# raised-surface background with a strong bottom border and bold primary
# ink text instead — still clearly a header, no colour hierarchy competing
# with the one legitimate accent use (tabs) any more.
DATATABLE_STYLE_HEADER = {
    "backgroundColor": "var(--pyaps-paper-raised)", "color": "var(--pyaps-ink)", "fontWeight": "700",
    "textTransform": "uppercase", "letterSpacing": "0.04em", "fontSize": "11px",
    "border": "none", "borderBottom": "2px solid var(--pyaps-line-strong)", "padding": "8px 10px",
}
DATATABLE_STYLE_CELL = {
    "fontFamily": FONT_FAMILY, "fontSize": "12px", "padding": "6px 10px",
    "border": "none", "borderBottom": "1px solid var(--pyaps-line)", "textAlign": "left",
    "backgroundColor": "var(--pyaps-paper-raised)", "color": "var(--pyaps-ink)",
}
DATATABLE_STYLE_DATA_CONDITIONAL = [
    {"if": {"row_index": "odd"}, "backgroundColor": "var(--pyaps-paper-sunken)"},
    {"if": {"state": "active"}, "backgroundColor": "var(--pyaps-accent-soft)", "border": "none"},
]
DATATABLE_STYLE_TABLE = {"overflowX": "auto"}

#: Convenience bundle — `dash_table.DataTable(..., **DATATABLE_KWARGS)`
#: applies the full look in one splat; individual pieces above stay
#: available for call sites that need to merge in their own extras (e.g. a
#: fixed height + `overflowY: "auto"` for a specific panel).
#:
#: Does NOT carry DataTable's own native `export_format`/`export_columns`/
#: `export_headers` props — tried first (a one-line central change here
#: would have given every real table in the app a CSV export button for
#: free) and reverted: confirmed via live Playwright reproduction against
#: this app's actual running server that clicking the resulting button
#: deterministically fails with a real ChunkLoadError on dash_table's own
#: async-export.js chunk. See `apsPlot/datatable_export.py`'s module
#: docstring for the full investigation and the custom dcc.Download-based
#: replacement now used instead (`csv_export_row()`/`register_csv_export()`
#: — wraps a table with its own button rather than a DataTable kwarg, so
#: nothing belongs here).
DATATABLE_KWARGS = dict(
    style_header=DATATABLE_STYLE_HEADER,
    style_cell=DATATABLE_STYLE_CELL,
    style_data_conditional=DATATABLE_STYLE_DATA_CONDITIONAL,
    style_table=DATATABLE_STYLE_TABLE,
)

#: Explicit request: dcc.Loading's default (non-fullscreen) overlay is
#: sized/positioned to its own children's bounding box, so a spinner
#: anchored near the top of a tall, scrollable page (e.g. the load-panel
#: sidebar) disappears once the user scrolls past it - "a rotating
#: progress bar only appears on the very top part of the page and if
#: you scrolled down, you do not see it... I want the whole page get
#: white or faded with a transparent [background] with the progress bar
#: at the centre of the page."
#:
#: `fullscreen=True` (dcc.Loading's own built-in mechanism for exactly
#: this) was tried first and REVERTED — confirmed live to cause a real
#: regression, not just a style quirk: with all 4 of this app's
#: dcc.Loading call sites switched to fullscreen simultaneously, real
#: usage against a real L1 dataset became "super slow and laggy," 3D-map
#: and FWHM-tab clicks stopped producing any content at all, and the
#: fullscreen overlay itself got stuck showing "usually nothing
#: appears" afterward — confirmed via `docker stats` that the *server*
#: was completely idle (0.03% CPU) while this was happening, meaning
#: the breakage is in the client-side rendering of multiple simultaneous
#: fullscreen Loading portals, not a backend slowdown. Back to the
#: known-good, non-fullscreen, per-panel `type="circle"` spinner for
#: now — the scroll-visibility request itself is still open, to be
#: revisited with a hand-built single overlay (one plain `position:
#: fixed` div toggled by a clientside callback) rather than relying on
#: dcc.Loading's own fullscreen prop again, once that's been verified
#: safe under real multi-click usage before shipping, not just a quick
#: smoke test.
LOADING_KWARGS = dict(type="circle")

TABS_STYLE = {"height": "38px"}
TAB_STYLE = {
    "padding": "8px 16px", "fontSize": "12px", "fontWeight": "500", "color": "var(--pyaps-ink-muted)",
    "backgroundColor": "var(--pyaps-paper-sunken)", "border": "none", "borderBottom": "2px solid transparent",
}
TAB_SELECTED_STYLE = {
    "padding": "8px 16px", "fontSize": "12px", "fontWeight": "700", "color": TABLE_HEADER_COLOR,
    "backgroundColor": "var(--pyaps-paper-raised)", "border": "none", "borderBottom": f"3px solid {TABLE_HEADER_COLOR}",
}

# A real, reusable "soft button" look — explicit follow-up report:
# "the bottons are rectangular bit I see other bottons like expand and
# so are some hwo soft edge rounish edge" and "the Go botton... has very
# sharp edges not lile expand or others with roundish edge." Most
# `html.Button`s in this app (Go/Apply/Reset/etc) never got explicit
# styling at all before this, so they fell back to plain, theme
# -unaware browser-default button chrome — not one specific button's
# own oversight, a gap in the whole app's own button styling. Uses the
# same tokens/radius `input[type=...]` elements already get from
# assets/style.css's own global rule (see that file) so a button and an
# adjacent input (e.g. "Go to bin ID") read as one deliberately
# -designed control group, not two mismatched pieces. Explicit fixed
# `height` (not just padding) is what actually fixes the "Go button's
# own height doesn't match the text box next to it" report — browser
# -default `<button>`/`<input>` heights differ slightly even with
# identical padding, so height needs to be pinned on *both* sides (see
# SOFT_INPUT_HEIGHT_STYLE below) rather than trusted to match on their
# own.
SOFT_BUTTON_STYLE = {
    "backgroundColor": "var(--pyaps-paper-raised)",
    "color": "var(--pyaps-ink)",
    "border": "1px solid var(--pyaps-line-strong)",
    "borderRadius": "var(--pyaps-radius-s)",
    "padding": "0 12px",
    "height": "30px",
    "boxSizing": "border-box",
    "cursor": "pointer",
    "fontSize": "12px",
}
# Paired with an `dcc.Input` sitting directly beside a SOFT_BUTTON_STYLE
# button (e.g. "Go to bin ID") — same height/box-sizing, so the two
# genuinely line up instead of each following its own browser default.
SOFT_INPUT_HEIGHT_STYLE = {"height": "30px", "boxSizing": "border-box"}
