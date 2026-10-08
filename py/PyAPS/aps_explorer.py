"""
aps_explorer.py - Central WEAVE L1/L2 explorer (Dash)
========================================================

The single interactive entry point for viewing WEAVE L1 (raw/reduced
per-fibre spectra) and L2 (`_APS.fits` analysis products) data. Give it
a dataset — via CLI args or the in-browser "Load dataset" form — and it
routes to whichever of three library modules understands it:

- **L1** (`--infiles`/`--infiles_list`, a list of stack/stackcube FITS
  files): routed to `aps_l1_preview.py`'s state/figures — click-to-select
  fibre map, flux/IVAR spectra, FWHM diagnostics, primary header.
- **L2 MOS/fibre-level** (`CLASS_TABLE` present in the merged `_APS.fits`
  — covers true MOS observations *and* LIFU/MIFU data processed at
  individual-fibre level): routed to `aps_MOSviewer.py`'s state/figures —
  click-to-select source map, Redrock (all ranks) always, Stellar
  (RVS/FERRE) and Galaxy (PPXF/EMI) tabs shown per-target since a single
  target commonly has both at once.
- **L2 IFU Voronoi-patch**, ExGal (`PATCH_TABLE`+`GALAXY_TABLE`) or Gal
  (`PATCH_TABLE`+`STAR_TABLE`): routed to `aps_IFUviewer.py`'s state/
  figures — click-to-select spatial map of any fitted quantity, PPXF/
  Emission/Line-Strength or RVS/FERRE fit tabs.

L2 schema detection uses `aps_utils.aps_file_info()`'s minimal-I/O,
headers-only inspection — never the filename/headname.

The spatial "click-to-select" view for all three kinds is an embedded
Aladin Lite panel (DSS imagery, on by default (toggleable via its own
checkbox) — plus a clickable catalog overlay coloured the same way a
plain Plotly coordinate/map plot would be) rather than a Plotly figure;
see `_aladin_catalog_points`/`_aladin_target` below.

This file owns the one real Dash `app` and every callback; it holds no
data-loading or figure-building logic of its own — `aps_l1_preview.py`,
`aps_IFUviewer.py`, and `aps_MOSviewer.py` are imported purely as
libraries (their own `AppState`/`STATE` singletons, loaders, figure
builders, and layout-piece/callback-logic functions are reused directly,
undecorated). Where a concern is genuinely shared across all three kinds
(the map, the tabs, ID-driven selection) there is one set of IDs and one
callback here that branches on which kind is loaded; where a kind
produces structurally different panels (settings row, value tables, the
load form itself) the shared container's *children* are swapped
wholesale, reusing each library's own layout-piece function unchanged.

USAGE
-----
    explorer_worker(["--outpath", "/path/to/results", "--headname", "stack_<runid>__stack_<runid>"])
    explorer_worker(["--infiles", "stackcube_<runid>.fit", "stackcube_<runid>.fit"])
    python aps_explorer.py --outpath /path/to/results --headname stack_<runid>__stack_<runid>
    python aps_explorer.py --infiles stackcube_<runid>.fit stackcube_<runid>.fit

Then open the printed URL in a browser. Launching with neither
--outpath/--headname nor --infiles opens straight into the "Load
dataset" form (both an L1 and an L2 section).
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import sys
import threading
import time
import traceback
from collections import deque
from pathlib import Path
from urllib.parse import parse_qs

import numpy as np
import pandas as pd

import PyAPS

# Flask's dev server logs an INFO-level line per HTTP request (every
# _dash-update-component POST, every Interval poll) via the "werkzeug"
# logger straight to stderr — with the in-browser log panel below tee-ing
# stderr too, that would drown out the pipeline's own prints in a wall of
# "POST /_dash-update-component ... 200" noise. Silence it to ERROR so
# actual werkzeug problems still surface but routine request logging
# doesn't.
logging.getLogger("werkzeug").setLevel(logging.ERROR)

from dash import ALL, MATCH, Dash, Input, Output, State, ctx, dash_table, dcc, html, no_update
from flask import request as flask_request, g as flask_g
from werkzeug.local import LocalProxy

from PyAPS import aps_constants
from PyAPS.aps_utils import none_or_str, aps_file_info, l1_fileinfo
from PyAPS import aps_l1_preview as l1_mod
from PyAPS import aps_IFUviewer as ifu_mod
from PyAPS import aps_MOSviewer as mos_mod
from PyAPS import aps_explorer_session as _sess
from PyAPS import aps_explorer_auth as _auth
from PyAPS.apsPlot import style as aps_style
from PyAPS.apsPlot.flux_cube import flux_cube_figure, DEFAULT_WAVE_BIN_WIDTH_ANGSTROM
from PyAPS.apsPlot.datatable_export import csv_export_row
from PyAPS.apsPlot.spaxel_map import spaxel_map_figure, select_colorscale
from PyAPS.apsPlot.source_map import source_map_figure


def _warm_up_plotly():
    """Pays Plotly's own one-time lazy-import/template-resolution cost
    here, at process startup, instead of on whichever real user's
    request happens to build the first figure. Explicit live report:
    "in MOS L2 view, when I click on a spaxel, I can quickly see the
    RVS or FERRE output spectra plots but REDROCK always take[s]
    tooooo long" — profiled directly (cProfile against a real dataset,
    not guessed) and confirmed this had *nothing* to do with Redrock
    itself: calling RVS *first* in a fresh process made *RVS* take the
    same ~0.3-0.4s (`io.open_code`/`json.decoder`/layout-template
    `__init__` dominating), with every subsequent call of *any* kind
    dropping to ~0.04-0.1s. Redrock only ever looked uniquely slow
    because it's always this app's own default/first-shown MOS tab
    (`aps_MOSviewer._available_tabs` always puts `_tab("Redrock",
    "redrock")` first, unconditionally), so it was always the one
    absorbing this one-time cost in a freshly-started worker. `plotly.
    graph_objs`'s own lazy
    submodule imports and its "plotly_white" named template's
    deep-copy-on-first-use both get resolved by this one throwaway
    figure — mirrors real usage shape (a small subplot grid, a trace, a
    `shapes=` list, the same `template="plotly_white"` every real
    figure here sets via `apsPlot.style.BASE_LAYOUT`) since narrower
    warm-ups (tried directly, not assumed) left some of the cost behind.
    Wrapped in a bare `except Exception` — this is a pure, best-effort
    performance optimization; a failure here (e.g. a future Plotly
    version restructuring these internals) must never block the app
    from starting."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Scattergl(x=[0], y=[0]))
        fig.update_layout(
            template="plotly_white",
            shapes=[dict(type="rect", x0=0, x1=1, y0=0, y1=1,
                          xref="x", yref="y domain", fillcolor="red", opacity=0.1)],
        )
    except Exception:
        pass


_warm_up_plotly()


# --------------------------------------------------------------------------- #
# Live log — the loading pipeline (APSOB, aps_calib, etc.) is all plain
# `print()`, previously visible only in the terminal running the server; the
# browser gave zero feedback while a Load click was working (long loads can
# take well over a minute), no visible spinner, and errors only ever reached
# a small `lf-status` div. This tees process-wide stdout/stderr into a ring
# buffer that a `dcc.Interval`-polled panel at the bottom of the page reads,
# so any print/traceback anywhere in the pipeline shows up live in-browser —
# not just the ones this file explicitly surfaces via lf-status.
# --------------------------------------------------------------------------- #

# Server mode only: collapses any absolute filesystem path down to just
# its basename before a line ever reaches the browser-visible log —
# explicit request: "the log... shows where we are reading the data
# from the server data directory and also... reveal many secure information... fine
# for standalone but not here." The pipeline (APSOB, aps_calib, etc.) is
# all plain print() with real paths baked directly into the text (e.g.
# "Analysing <PYAPS_DATA>/L1/.../single_3039506.fit file..."), so
# hiding this at the *source* would mean auditing/gating hundreds of
# scattered print() call sites across the whole codebase, with no single
# place to verify none were missed. Fixed centrally here instead — every
# print()/traceback anywhere in the pipeline already funnels through
# _StreamTee -> LOG.add() -> here, so this one place covers all of them
# uniformly, standalone mode included in the regex's reach but simply
# never invoked there (MULTI_SESSION off). Deliberately keeps the
# basename (not a blanket placeholder) rather than deleting it outright
# -- filenames aren't sensitive on their own and stay useful for
# following what's happening (they're exactly what a weaveOR user
# already saw in their own query results); it's the server's own
# directory/mount structure that's the actual concern.
_ABS_PATH_RE = re.compile(r"(?:/[\w.\-+]+)+/([\w.\-+]+)")


def _redact_paths(text):
    if not _sess.MULTI_SESSION:
        return text
    return _ABS_PATH_RE.sub(lambda m: m.group(1), text)


class _LogBuffer:
    def __init__(self, maxlen=1000):
        self.lines = deque(maxlen=maxlen)
        self.lock = threading.Lock()

    def add(self, tag, text):
        ts = time.strftime("%H:%M:%S")
        text = _redact_paths(text)
        with self.lock:
            for line in text.splitlines():
                if line.strip():
                    self.lines.append(f"[{ts}]{tag} {line}")

    def snapshot(self):
        with self.lock:
            return list(self.lines)

    def clear(self):
        with self.lock:
            self.lines.clear()


# A LocalProxy, not a bare instance: in --multi-session (server/Docker)
# mode this resolves to the current request's own SessionBundle slot, so
# each browser's log panel only ever shows that browser's own activity;
# with --multi-session off (the default — standalone usage) or outside
# any Flask request at all (the test suite's calling convention), it
# resolves to one persistent shared buffer — exactly today's behaviour.
# See aps_explorer_session.py's own module docstring for the full design.
LOG = LocalProxy(lambda: _sess.current_bundle().get_or_create("log", _LogBuffer))


class _StreamTee:
    """Writes through to the real stream (terminal behaviour unchanged) and
    also appends into LOG."""

    def __init__(self, real_stream, tag):
        self._real = real_stream
        self._tag = tag

    def write(self, s):
        self._real.write(s)
        if s.strip():
            LOG.add(self._tag, s)

    def flush(self):
        self._real.flush()

    def isatty(self):
        return False


sys.stdout = _StreamTee(sys.stdout, "")
sys.stderr = _StreamTee(sys.stderr, " [stderr]")


# --------------------------------------------------------------------------- #
# Explorer state — just tracks *which* library's STATE is currently active.
# The actual loaded dataset lives in {l1,ifu,mos}_mod.STATE.
# --------------------------------------------------------------------------- #

class ExplorerState:
    def __init__(self):
        self.kind = None          # "l1", "ifu", or "mos"
        self.file_info = None     # aps_file_info() result — L2 kinds only
        # Not reset on load — deliberately mirrors aladin-show-dss (a
        # plain checkbox whose own value isn't tied to EXPLORER state
        # either), so toggling "true fibre/spaxel size" on sticks across
        # a "Load different dataset" the same way that one does.
        self.aladin_true_size = False
        # Same "sticks across a reload" convention as aladin_true_size
        # above — see _contrib_exposures_payload(). Per-file selection
        # within the overlay (which contributing exposure is shown/
        # hidden) is a pure client-side show()/hide() toggle now, tracked
        # only in the browser's own DOM state, not here — see
        # _contrib_exposures_payload's own docstring.
        self.contrib_exposures_on = False
        # 3D flux cube only — explicit report: "when I look at the cross
        # section of 3d data cubes, I do not see what I usually see in 2d
        # maps... probably due to transparency issue." Off by default (a
        # 2D map's own points are always fully opaque) so the two views
        # match unless the user deliberately opts back into the low-
        # signal-fades gradient — see flux_cube_figure's own `transparent`
        # docstring. Same "sticks across a reload" convention as the two
        # fields above (a display preference, not dataset-specific).
        self.cube_transparency_on = False
        # Shared colour PALETTE override — explicit request: "is it
        # possible to instead of using the current color map, we use
        # something that is black for low signal and white for high
        # signal... Can I have it in all maps either 2D or 3D maps so we
        # can switch between this blue to red to black to white?" One
        # setting, not per-kind, applied uniformly to whatever colourscale
        # a given map/kind would otherwise use — see
        # _resolve_palette_override(). "default" = leave every kind's own
        # existing per-quantity colourscale choice (Jet for flux, Plasma
        # for S/N, etc.) untouched, same "sticks across a reload"
        # convention as the fields above.
        self.color_palette = "default"
        # Which `dataset-version` value the 3D flux cube's own current
        # camera (read back from flux-cube-graph's own live `figure` prop
        # — see update_map_mode's own docstring for why) was last built
        # for. `None` initially (nothing built yet). update_map_mode
        # compares this against the incoming `version` on every rebuild:
        # equal → the cube is being rebuilt for the *same* dataset the
        # user was already looking at (a click/colour-range/scale/
        # palette/transparency/2D<->3D-toggle change), so the read-back
        # camera is reused, keeping the user's current view stable;
        # different (or None) → this is the *first* rebuild for a
        # genuinely different dataset, so the camera is deliberately
        # reset to flux_cube_figure's own default top-down view instead
        # of silently inheriting wherever the previous dataset's cube
        # happened to be left rotated to. NOT a "sticks across a reload"
        # preference like the fields above (it's reset, on purpose, on
        # every genuinely new dataset).
        self.flux_cube_camera_version = None

    def loaded(self):
        return self.kind is not None


# Same LocalProxy pattern as LOG above — per-session in --multi-session
# mode, one shared instance otherwise (standalone usage, or outside any
# Flask request at all, e.g. this repo's own test suite).
EXPLORER = LocalProxy(lambda: _sess.current_bundle().get_or_create("explorer", ExplorerState))
_MAP_HEIGHT = "600px"

# The Plotly coordinate/spatial map (main-map) was dropped entirely — the
# Aladin panel (DSS/no-background sky view + the same click-to-select
# catalog overlay, coloured by whatever quantity is currently shown) does
# everything it did, and then some, per explicit user request; kept as the
# one and only spatial view now, always visible (no more Off/Separate-panel
# toggle — see the removed aladin-mode RadioItems). Style is set entirely
# client-side (see the aladin-target clientside_callback below) rather than
# via a server round-trip — kept here only for the very first
# server-rendered layout, before any client-side JS has run.
#
# position:relative (not "static") is load-bearing, not cosmetic: Aladin
# Lite draws its own UI controls (.aladin-fullscreen, .aladin-zoomControl,
# .aladin-gotoControl, .aladin-layersControl, confirmed against the v3 API
# docs) as CSS position:absolute children of this div. An
# absolutely-positioned element with no positioned ancestor positions
# itself against the viewport instead — exactly what "Aladin spans the
# whole screen" looked like in an earlier round. zIndex on top of
# position:relative establishes a real stacking context, so nothing inside
# this div can escape above it regardless of whatever z-index Aladin's own
# internal CSS uses.
_ALADIN_DIV_STYLE = {
    "display": "block", "position": "relative", "zIndex": 0,
    "width": "100%", "height": _MAP_HEIGHT,
    "border": "1px solid var(--pyaps-line-strong)", "borderRadius": "var(--pyaps-radius-m)", "overflow": "hidden",
}

#: A visually-distinct, rounded, theme-adapted boundary around a major
#: page cluster — explicit request: "a border around the coordinate (2d
#: or 3d) plots with a roundish edge and color adopt for bright and dark
#: mode? the same for the tabs group next to it...so the boarder of each
#: element on the page be more clear." Reused for both the coordinate-map
#: cluster (map-mode switch + 2D/3D map + its legend/checkboxes/colour
#: range/info-box — everything that already read as "the Aladin
#: coordinate map" as one cluster, per the comment above Slit Explorer
#: below, which stays outside this card as its own separate panel) and
#: the tabs+tab-content cluster on the other side. boxSizing:border-box
#: so the added border/padding don't push either column past its
#: declared 44%/54% width (no box-sizing:border-box is set globally in
#: this app — confirmed before adding padding here).
_SECTION_CARD_STYLE = {
    "border": "1px solid var(--pyaps-line-strong)", "borderRadius": "var(--pyaps-radius-m)",
    "padding": "12px", "boxSizing": "border-box",
}


def _route_and_load_l2(outpath, headname):
    """Build the `_APS.fits` path, classify it with `aps_file_info()`
    (extensions + header content only — never the filename/headname), and
    load it through whichever library actually understands that schema.
    Raises on an unreadable or unrecognized file; caller decides how to
    surface that.
    """
    aps_file = str(Path(outpath) / headname) + "_APS.fits"
    print(f">>> Loading L2 product: {aps_file}")
    info = aps_file_info(aps_file)
    schema = info["schema"]
    print(f">>> Detected schema={schema} obsmode={info['obsmode']}")
    if schema == "mos":
        mos_mod.STATE.load(outpath, headname)
        EXPLORER.kind = "mos"
        initial_item = _central_item("mos")
        mos_mod.STATE.selected_aps_id = initial_item
    elif schema in ("ifu_exgal", "ifu_gal"):
        ifu_mod.STATE.load(outpath, headname)
        EXPLORER.kind = "ifu"
        initial_item = _central_item("ifu")
    else:
        raise ValueError(
            f"Unrecognized L2 schema for {aps_file} (extensions: {info['extensions']}) — "
            "expected a mosL2merge, ifuExGalL2merge, or ifuGalL2merge product."
        )
    EXPLORER.file_info = info
    print(">>> Load complete.")
    return initial_item


def _load_l1(args):
    """Load an L1 dataset from an argparse-style Namespace (see
    `aps_l1_preview._build_arg_parser`/`load_from_form_fields`). Raises
    on a load error; caller decides how to surface that."""
    print(f">>> Loading L1 dataset: {args.infiles or args.infiles_list}")
    l1_mod.STATE.load(args)
    EXPLORER.kind = "l1"
    EXPLORER.file_info = None
    print(">>> Load complete.")
    return _central_item("l1")


# --------------------------------------------------------------------------- #
# Layout pieces
# --------------------------------------------------------------------------- #

# load-panel is an overlay sidebar, not part of the normal document flow —
# previously it sat inline above main-panel, so reopening it (to tweak a
# parameter and reload) pushed the still-visible current view far down the
# page, making it look like two disconnected things stacked rather than
# "here are the settings for what you're already looking at". Every place
# that sets this style (here and every callback Output("load-panel","style"))
# must go through this helper — Dash replaces the whole style dict on each
# Output, so a callback returning a bare {"display": ...} would silently
# wipe out the positioning/sizing below.
_LOAD_PANEL_BASE_STYLE = {
    "position": "fixed", "top": 0, "left": 0, "height": "100vh", "width": "920px",
    "maxWidth": "95vw", "overflowY": "auto", "backgroundColor": "var(--pyaps-paper-raised)",
    "boxShadow": "var(--pyaps-shadow-drawer)",
    # Real animated slide (explicit request: it "appears instead of
    # sliding in"), not display:none/block — see _load_panel_style()
    # below for why this is a single shared transition string rather
    # than a CSS class.
    "transition": "transform .3s cubic-bezier(.22,.9,.32,1)",
    # Was 1000 (still the design intent: "the sidebar, when open, covers
    # everything else" — see _BANNER_Z's own comment). Raised to clear
    # #global-load-overlay (9500) and _log_panel's 9600, both added
    # later — otherwise, any time the sidebar happened to visually
    # overlap the log panel on screen (routine: both render near the
    # top of the page whenever main-panel is hidden), the log panel's
    # own elevated stacking silently ate clicks meant for the sidebar's
    # "Load / Reload" button, confirmed live via Playwright. Must stay
    # above _PATH_BROWSER_MODAL_STYLE_BASE too, which is deliberately
    # kept one level above this one (opened from inside the sidebar).
    # #load-panel-backdrop sits one level below, at 9690.
    "zIndex": 9700,
}


def _load_panel_style(visible):
    """`display: none/block` (the old behavior) can't be transitioned by
    CSS at all, which is exactly why the panel used to just pop in/out
    instead of sliding — swapped for `transform` here, which does
    animate via the `transition` already sitting in
    _LOAD_PANEL_BASE_STYLE. `visibility` is delayed on the *closing*
    transition only (matches the .3s slide duration) so the panel stays
    paintable while it's still visibly sliding off-screen, then actually
    leaves the accessibility tree/tab order once fully hidden — opening
    restores `visibility` immediately so there's no ghost delay before
    the slide-in becomes visible. `pointerEvents` is a second, redundant
    safety net (visibility:hidden already blocks all interaction) kept
    for defense in depth, since it costs nothing.
    """
    style = dict(_LOAD_PANEL_BASE_STYLE)
    if visible:
        style["transform"] = "translateX(0)"
        style["visibility"] = "visible"
        style["pointerEvents"] = "auto"
    else:
        style["transform"] = "translateX(-100%)"
        style["visibility"] = "hidden"
        style["pointerEvents"] = "none"
        style["transition"] = _LOAD_PANEL_BASE_STYLE["transition"] + ", visibility 0s linear .3s"
    return style


def _load_panel_backdrop_style(visible):
    """Dimming scrim behind the drawer — never existed before (load-panel
    previously had no backdrop of any kind, confirmed against the whole
    file before adding this). Used only for the very first (server-
    rendered) page load; every update after that comes from a
    clientside_callback (near on_open_load_panel below) reacting to
    load-panel's own style, so the backdrop always mirrors the drawer's
    real state — including the several *business-logic*-driven opens/
    closes (e.g. auto-close on a successful load) that never go through
    the open/close buttons at all, without needing to touch any of
    those existing callbacks. Plain black, not a theme token — a dimming
    scrim reads correctly as "a dimming scrim" in both app themes.

    `top` deliberately starts below the banner+toolbar rather than
    `inset: 0` — confirmed live via Playwright that a full-viewport
    backdrop sits (by z-index) on top of the *entire* #app-banner
    (z-index 501) including the new theme toggle, silently making it
    unclickable any time the drawer is open — which, since load-panel
    opens automatically whenever no dataset is loaded, is effectively
    "on every fresh visit." `_BANNER_HEIGHT_PX` is exact; the toolbar has
    no fixed height constant of its own (content-sized), so this uses a
    safe overestimate for this one SSR-only initial paint — the
    clientside callback below immediately replaces it with the actual
    measured height once the page has rendered, so the imprecision here
    lasts at most one frame."""
    return {
        "position": "fixed", "top": f"{_BANNER_HEIGHT_PX + 60}px", "left": "0", "right": "0", "bottom": "0",
        "backgroundColor": "rgba(0,0,0,0.35)",
        "zIndex": 9690,  # one below load-panel's 9700, above log-panel's 9600
        "opacity": "1" if visible else "0",
        "pointerEvents": "auto" if visible else "none",
        "transition": "opacity .28s ease",
    }


# Same "Dash replaces the whole style dict on each Output" gotcha as
# _load_panel_style above — main-panel's own padding (added alongside the
# banner, once body's default margin could no longer be relied on for edge
# spacing — see _banner()) would otherwise vanish the moment either load
# callback set a bare {"display": "block"}.
_MAIN_PANEL_BASE_STYLE = {"padding": "0 16px 16px 16px", "boxSizing": "border-box"}


def _main_panel_style(visible):
    style = dict(_MAIN_PANEL_BASE_STYLE)
    style["display"] = "block" if visible else "none"
    return style


def _current_l2_outpath_headname():
    """Best-effort (outpath, headname) the currently-loaded L2 dataset was
    opened with, derived from its own merged filepath (there's no separate
    "load_args" for L2 the way L1 has — outpath+headname *is* the whole
    input) — used only to prefill the L2 form when it's reopened. None,
    None if nothing L2 is loaded."""
    if EXPLORER.kind not in ("mos", "ifu") or not EXPLORER.file_info:
        return None, None
    filepath = EXPLORER.file_info.get("filepath")
    if not filepath or not filepath.endswith("_APS.fits"):
        return None, None
    return os.path.split(filepath[: -len("_APS.fits")])


_CLOSE_LOAD_PANEL_BTN = html.Div([
    html.Button("×", id="close-load-panel", title="Hide this panel", style={
        "position": "absolute", "top": "10px", "right": "16px", "fontSize": "20px",
        "lineHeight": "1", "border": "none", "background": "transparent",
        "cursor": "pointer", "color": "var(--pyaps-ink-faint)", "padding": "4px 8px",
    }),
], style={"position": "relative"})


def _load_form():
    """L1 and L2 are two dcc.Tabs, not two stacked sections sharing one
    scroll area — previously both "Load / Reload" buttons were visible on
    screen at once regardless of which kind you were actually editing,
    an easy way to press the wrong one; tabs make only one section (and
    one Load button) visible/clickable at a time. Whichever section
    matches the currently-loaded kind (if any) opens active, and every
    field in both sections prefills with the exact values that dataset was
    loaded with (L1: STATE.load_args; L2: derived from its own filepath) —
    reopening this panel after a load shows what's actually running, not a
    blank form, per explicit user request.

    In server mode (`aps_explorer_session.MULTI_SESSION`), this panel has
    two further states on top of the above, gated on whether this
    session already has a dataset recorded (see
    `aps_explorer_auth.check_load_authorized`'s own docstring for why —
    only a URL, token or plain deep link, may ever set *which* files a
    server-mode session shows): "locked" (something's loaded — the full
    form renders, but the file-selection fields are replaced with a
    non-revealing notice; every other param stays editable, matching the
    "so they can tick/untick sky subtraction, switch air/vacuum, etc."
    ask directly) or "blocked" (nothing loaded yet — a bare page load
    with no valid deep link; a plain explanatory message instead of a
    form that could only ever reject whatever got typed into it)."""
    bundle = _sess.current_bundle()
    auth = _auth.current_auth(bundle) if _sess.MULTI_SESSION else None

    if _sess.MULTI_SESSION and auth is None:
        return html.Div([
            _CLOSE_LOAD_PANEL_BTN,
            html.Div([
                html.H3("No dataset loaded", style={"marginTop": 0}),
                html.P("This server only opens datasets via a direct link — "
                       "please use the link you were given (e.g. from WeaveOR)."),
            ], style={"padding": "20px", "fontSize": "13px"}),
        ])

    locked = auth is not None
    l1_current = l1_mod.STATE.load_args if EXPLORER.kind == "l1" else None
    outpath, headname = _current_l2_outpath_headname()
    default_tab = "l2" if EXPLORER.kind in ("mos", "ifu") else "l1"

    l1_tab_content = html.Div(l1_mod._load_form(current=l1_current, locked=locked),
                               style={"padding": "0 20px"})

    if locked:
        l2_file_section = html.Div([
            html.H3("Load an L2 product (IFU or MOS)", style={"marginTop": 0}),
            html.Div("🔒 Dataset fixed for this session — opened via a direct link.",
                      style={"fontSize": "12px", "color": "var(--pyaps-ink-muted)", "fontStyle": "italic",
                             "marginBottom": "8px"}),
            dcc.Input(id="lf-l2-fullfile", type="text", style={"display": "none"}, value=""),
            dcc.Input(id="lf-outpath", type="text", style={"display": "none"}, value=""),
            dcc.Input(id="lf-headname", type="text", style={"display": "none"}, value=""),
        ])
    else:
        l2_file_section = html.Div([
            html.H3("Load an L2 product (IFU or MOS)", style={"marginTop": 0}),
            html.Label("Full L2 filename (_APS.fits) — alternative to outpath+headname below",
                       style=l1_mod._LABEL_STYLE),
            html.Div([
                dcc.Input(id="lf-l2-fullfile", type="text", style={**l1_mod._INPUT_STYLE, "flex": "1"}),
                l1_mod._browse_button("lf-l2-fullfile", mode="file"),
            ], style={"display": "flex", "alignItems": "flex-start"}),
            html.Div([
                l1_mod._field_browsable("Output/results directory (--outpath)", "lf-outpath",
                                         "", mode="dir", value=outpath),
                l1_mod._field("Head name (--headname)",
                               dcc.Input(id="lf-headname", type="text", style=l1_mod._INPUT_STYLE,
                                         value=headname)),
            ], style=l1_mod._GRID_STYLE),
        ])

    l2_tab_content = html.Div([
        l2_file_section,
        html.Button("Load / Reload", id="lf-run", n_clicks=0,
                    style={"marginTop": "14px", "fontSize": "13px", "padding": "6px 16px"}),
        html.Div(id="lf-status", style={"color": "crimson", "marginTop": "6px", "fontSize": "12px"}),
    ], style={"padding": "20px", "fontSize": "13px"})

    return html.Div([
        _CLOSE_LOAD_PANEL_BTN,
        dcc.Tabs(id="load-form-tabs", value=default_tab, style=aps_style.TABS_STYLE, children=[
            dcc.Tab(label="L1", value="l1", style=aps_style.TAB_STYLE,
                    selected_style=aps_style.TAB_SELECTED_STYLE, children=l1_tab_content),
            dcc.Tab(label="L2 (IFU / MOS)", value="l2", style=aps_style.TAB_STYLE,
                    selected_style=aps_style.TAB_SELECTED_STYLE, children=l2_tab_content),
        ]),
    ])


# --------------------------------------------------------------------------- #
# Server-side path browser — every path field in the load form is a path on
# *this server*, not the browser's own machine, so a native
# OS file-picker would pick from entirely the wrong filesystem. This is a
# small in-app directory/file browser instead: click a field's 📁 button,
# navigate server-side folders, click to select — the field stays freely
# editable by hand either way, this is just a shortcut.
# --------------------------------------------------------------------------- #

# PYAPS_DATA_DIR (env) points at the folder holding your L1/L2/CAL/CAT data;
# falls back to PYAPS_HOME, then the user's home directory.
_PATH_BROWSER_DEFAULT_ROOT = next(
    (p for p in (os.environ.get("PYAPS_DATA_DIR"), os.environ.get("PYAPS_HOME"),
                 os.path.expanduser("~"), "/") if p and os.path.isdir(p)), "/"
)

_PATH_BROWSER_MODAL_STYLE_BASE = {
    "position": "fixed", "top": 0, "left": 0, "right": 0, "bottom": 0,
    "backgroundColor": "rgba(0,0,0,0.35)",
    # Kept one level above _LOAD_PANEL_BASE_STYLE's own zIndex (see its
    # comment) since this modal is opened from a button inside that
    # sidebar and must render on top of it.
    "zIndex": 9800,
    "alignItems": "center", "justifyContent": "center",
}


def _path_browser_modal_style(visible):
    return {**_PATH_BROWSER_MODAL_STYLE_BASE, "display": "flex" if visible else "none"}


def _path_browser_modal():
    return html.Div(
        id="path-browser-modal",
        style=_path_browser_modal_style(False),
        children=[
            html.Div([
                html.Div([
                    html.Span("Browse server files", style={"fontWeight": "700", "fontSize": "14px"}),
                    html.Button("×", id="path-browser-close", n_clicks=0, style={
                        "float": "right", "border": "none", "background": "transparent",
                        "fontSize": "18px", "cursor": "pointer", "color": "var(--pyaps-ink-faint)", "lineHeight": "1",
                    }),
                ], style={"marginBottom": "8px"}),
                html.Div(id="path-browser-hint", style={"fontSize": "11px", "color": "var(--pyaps-ink-muted)", "marginBottom": "6px"}),
                html.Div([
                    html.Button("⬆ Up", id="path-browser-up", n_clicks=0,
                                style={"fontSize": "12px", "padding": "3px 10px", "marginRight": "6px"}),
                    dcc.Input(id="path-browser-path-input", type="text", debounce=True,
                              style={"width": "70%", "fontSize": "12px", "padding": "4px 6px"}),
                ], style={"marginBottom": "8px", "display": "flex", "alignItems": "center"}),
                html.Div(id="path-browser-listing", style={
                    "height": "320px", "overflowY": "auto", "border": "1px solid #e0e0e0",
                    "borderRadius": "4px", "padding": "4px",
                }),
                html.Div([
                    html.Button("Use this directory", id="path-browser-use-dir", n_clicks=0, style={
                        "fontSize": "12px", "padding": "5px 12px", "marginTop": "10px",
                        "backgroundColor": aps_style.TABLE_HEADER_COLOR, "color": "white",
                        "border": "none", "borderRadius": "4px", "cursor": "pointer",
                    }),
                    html.Button("Cancel", id="path-browser-cancel", n_clicks=0, style={
                        "fontSize": "12px", "padding": "5px 12px", "marginTop": "10px", "marginLeft": "8px",
                    }),
                ]),
            ], style={"backgroundColor": "white", "borderRadius": "6px", "padding": "16px 18px",
                       "width": "560px", "maxWidth": "92vw", "boxShadow": "0 8px 30px rgba(0,0,0,0.3)"}),
        ],
    )


def _path_browser_listing(cwd, mode):
    """Row list for the given directory — folders always navigable, files
    only shown (and selectable) when mode isn't "dir"-only."""
    try:
        entries = sorted(os.scandir(cwd), key=lambda e: (not e.is_dir(), e.name.lower()))
    except OSError as e:
        return [html.Div(f"Can't read this directory: {e}", style={"color": "crimson", "padding": "8px"})]

    rows = []
    for entry in entries:
        if entry.name.startswith("."):
            continue
        is_dir = entry.is_dir()
        if not is_dir and mode == "dir":
            continue  # dir-only pickers (caldir/catdir/configdir) don't show files at all
        icon = "📁" if is_dir else "📄"
        rows.append(html.Div(
            f"{icon} {entry.name}",
            id={"type": "path-browser-entry", "path": entry.path, "kind": "dir" if is_dir else "file"},
            n_clicks=0,
            style={"padding": "5px 8px", "cursor": "pointer", "fontSize": "12px", "borderRadius": "3px"},
            className="path-browser-row",
        ))
    if not rows:
        rows = [html.Div("(empty)", style={"padding": "8px", "color": "var(--pyaps-ink-faint)", "fontSize": "12px"})]
    return rows


_INFO_PANEL_STYLE = {
    "fontSize": "12px", "color": "var(--pyaps-ink)", "backgroundColor": "var(--pyaps-paper-sunken)",
    "border": "1px solid var(--pyaps-line-strong)", "borderRadius": "4px",
    "padding": "8px 12px", "marginBottom": "10px", "lineHeight": "1.6",
}


def _is_locked():
    """Same server-mode-with-a-fixed-dataset check `serve_load_panel`
    already computes for `_load_form`'s own `locked` argument (see that
    function's own docstring) — factored out so `_file_info_panel` can
    reuse the exact same condition for masking real server-side paths
    in the always-visible summary panel too, not just the load form."""
    bundle = _sess.current_bundle()
    auth = _auth.current_auth(bundle) if _sess.MULTI_SESSION else None
    return auth is not None


#: Shared chevron used by every collapsible "card" in this panel — points
#: down (expanded) by default; .card.collapsed .card-body (style.css)
#: hides the body, purely in CSS so the *initial* (server-rendered)
#: collapsed state below needs no clientside_callback to take effect on
#: first paint.
#:
#: A real, visibly-a-button pill (border/background/padding, like every
#: other small control in this app — the theme toggle, "Export CSV",
#: etc.) with an actual text label, not just a bare glyph — explicit
#: follow-up report: "the collapase/uncollapse bottton for the input
#: info is very small and not visible at all..Make a proper bottom with
#: text for it." Two label spans, one per collapsed state, swapped
#: purely via style.css's `.card.collapsed`-scoped display rules (same
#: "no JS needed for the initial/CSS-driven part" reasoning as the card
#: mechanism itself) — a single rotating chevron can't carry different
#: *text* the way a rotating icon can carry a different *angle*.
def _card_toggle_glyph():
    return html.Span(
        [
            html.Span("▾  Collapse", className="card-toggle-label-open"),
            html.Span("▸  Expand", className="card-toggle-label-collapsed"),
        ],
        className="card-toggle",
        style={
            "display": "inline-flex", "alignItems": "center", "fontSize": "11px", "fontWeight": "600",
            "color": "var(--pyaps-ink-muted)", "padding": "4px 12px", "borderRadius": "var(--pyaps-radius-pill)",
            "border": "1px solid var(--pyaps-line-strong)", "backgroundColor": "var(--pyaps-paper-raised)",
            "flexShrink": "0",
        },
    )


def _collapsible_card(title, body_children, collapsed=True):
    """Wraps `body_children` in the .card/.card-head/.card-body structure
    explorer_ui.js's click-delegated toggle (and style.css's own
    .card.collapsed rules) were built for but, until now, had no real
    content to apply to (see that JS module's own docstring — "Deferred
    to Phase 2"). First real use: the dataset-info panel — explicit
    request: "I want for both L1 and L2 that panel be minimise by
    default and have a bottoon to collapse or uncollapse it."
    `collapsed=True` renders the initial "collapsed" class server-side
    (so it starts minimised on first paint, no JS round-trip needed for
    that); clicking the header toggles it from then on, entirely
    client-side."""
    return html.Div([
        html.Div([
            html.B(title),
            _card_toggle_glyph(),
        ], className="card-head", style={
            "display": "flex", "alignItems": "center", "justifyContent": "space-between", "cursor": "pointer",
        }),
        html.Div(body_children, className="card-body", style={"marginTop": "8px"}),
    ], className="card collapsed" if collapsed else "card", style={**_INFO_PANEL_STYLE, "padding": "8px"})


def _file_info_panel():
    """Always-visible "what am I looking at" panel — directory/filename(s),
    file/pipeline version, and every load option actually used. Previously
    this either didn't exist (L1: just a fibre count) or omitted the parts
    that actually answer "which file/options" (L2: schema/obsmode summary
    only, no explicit path) — both confusing once more than one dataset had
    been loaded in the same session. Minimised by default with a
    collapse/expand toggle (see _collapsible_card) — explicit follow-up
    request, since a several-row table (L1) or a now-tabular L2 summary
    (below) permanently taking up space at the top of the page competed
    with the actual content for attention."""
    if not EXPLORER.loaded():
        return None
    locked = _is_locked()
    if EXPLORER.kind == "l1":
        st = l1_mod.STATE
        rows = st.load_summary(locked=locked)
        return _collapsible_card("Current dataset (L1)", [
            csv_export_row("l1-info-table", dash_table.DataTable(
                id="l1-info-table",
                data=rows,
                columns=[{"name": "Parameter", "id": "Parameter"}, {"name": "Value", "id": "Value"}],
                **{**aps_style.DATATABLE_KWARGS,
                   "style_table": {**aps_style.DATATABLE_STYLE_TABLE, "height": "220px", "overflowY": "auto"}},
                style_cell_conditional=[{"if": {"column_id": "Parameter"}, "width": "45%", "fontWeight": "600"}],
                fixed_rows={"headers": True},
            )),
        ])

    fi = EXPLORER.file_info
    if EXPLORER.kind == "mos":
        n_targets, label = len(mos_mod.STATE.data["class_table"]), "targets"
    else:
        n_targets, label = len(ifu_mod.STATE.data["table"]["BIN_ID"]), "spaxels"
    # The real server-side path is masked when locked, same as L1's own
    # load_summary() above — explicit report that this panel was still
    # revealing it "in all modes L1 or L2."
    file_display = ("🔒 Dataset fixed for this session — opened via a direct link."
                     if locked else fi["filepath"])
    # Same {"Parameter", "Value"} row-table shape as L1's own load_summary()
    # above, rather than the several lines of plain text this used to be —
    # explicit report: "in L2 mode, the top panel show the basic info of
    # the input files is still showing plai text not tabular mode...I want
    # somethinmg similar to L1 one." Fewer rows than L1 (there's no
    # equivalent "every load option used" for an already-merged _APS.fits
    # the way there is for a fresh L1 load), but the same table look/feel.
    rows = [
        {"Parameter": "File", "Value": file_display},
        {"Parameter": "Schema", "Value": fi["schema"]},
        {"Parameter": "Obsmode", "Value": fi["obsmode"]},
        {"Parameter": "Resolution", "Value": fi.get("resolution") or "(none)"},
        {"Parameter": "Binning (x, y)", "Value": f"{fi.get('xbin', '?')}, {fi.get('ybin', '?')}"},
        {"Parameter": "Arms", "Value": ", ".join(fi["arms"])},
        {"Parameter": "PyAPS version", "Value": f"v{fi['pyaps_version']}"},
        {"Parameter": "CPS version", "Value": f"v{fi['cps_version']}"},
        {"Parameter": label.capitalize(), "Value": str(n_targets)},
        {"Parameter": "Observed", "Value": fi["date_obs"]},
        {"Parameter": "APS processed", "Value": fi["aps_date"]},
    ]
    table_id = "l2-info-table"
    return _collapsible_card(f"Current dataset ({'MOS/fibre-level' if EXPLORER.kind == 'mos' else 'IFU'})", [
        csv_export_row(table_id, dash_table.DataTable(
            id=table_id,
            data=rows,
            columns=[{"name": "Parameter", "id": "Parameter"}, {"name": "Value", "id": "Value"}],
            **{**aps_style.DATATABLE_KWARGS,
               "style_table": {**aps_style.DATATABLE_STYLE_TABLE, "height": "220px", "overflowY": "auto"}},
            style_cell_conditional=[{"if": {"column_id": "Parameter"}, "width": "45%", "fontWeight": "600"}],
            fixed_rows={"headers": True},
        )),
    ])


def _settings_panel_children():
    if EXPLORER.kind == "ifu":
        return ifu_mod._settings_panel()
    if EXPLORER.kind == "mos":
        return mos_mod._settings_row()
    if EXPLORER.kind == "l1":
        return l1_mod._settings_row()
    return None


def _value_tables_children():
    if EXPLORER.kind == "ifu":
        return ifu_mod._value_tables_panel()
    if EXPLORER.kind == "mos":
        return mos_mod._value_tables_panel()
    # L1 has no below-the-fold value-tables panel; all of its info lives
    # in its tabs (Spectra/FWHM/Header).
    return None


# --------------------------------------------------------------------------- #
# Multi-panel L2 map feature (IFU/MOS only, 2D only) — explicit request:
# "for L2 preview, no matter MOS or IFU... I want to be able to add a
# number of panels exactly like the coordinate maps, and for each I want
# to be able to select what maps should be plot[ted] and it should has
# its own color bar and scale and range selection... right below the
# first row where shows the master map and the spectrum... Clearly no
# need for 3d (Only 2d)... no need to have transparency by signal or BIN
# id info panel... save that preview configuration into a file and load
# it again from file."
#
# Deliberately plain Plotly (no Aladin) per panel, reusing IFU's own
# already-built (but, until now, dead-code) spaxel_map_figure — see
# apsPlot/spaxel_map.py — and a small new MOS counterpart,
# apsPlot/source_map.py, built the same way. Neither one touches a FITS
# file or calls a viewer's own load() — both read purely from the
# already-loaded, already-in-memory STATE.data any other part of this
# app is already using, addressing the explicit follow-up: "avoid to
# load files every time for each [panel]... I do not want that it break
# the system if more users use the server type mode at the same time."
# --------------------------------------------------------------------------- #

# Height only — width is deliberately never passed fixed to the panel
# figure builders (see _build_extra_map_figure's own comment): a 1-row
# grid used to render its one panel at this literal pixel width, visibly
# left-aligned with empty space beside it while the controls row above
# stretched to the panel's full actual width — explicit follow-up
# report: "when I select row 1 and column 1 the plot is left aligned
# but the map selection bar fill[s] the whole width of that panel...
# limited to hight not width... I prefer better mode."
_EXTRA_MAP_PANEL_HEIGHT = 330
# Explicit follow-up: "I want to have max 6 column and max 3 rows" —
# deliberately asymmetric, not a single square cap.
_EXTRA_MAP_MAX_ROWS = 3
_EXTRA_MAP_MAX_COLS = 6

# A real, independent per-panel palette choice — deliberately *not*
# PALETTE_OPTIONS above (that's a single app-wide override toggle
# applied uniformly everywhere, "default"/"jet"/"greyscale"/"white_red";
# this is genuine per-quantity Plotly colorscale names, matching what
# select_colorscale/COLOR_BY_OPTIONS already use elsewhere in the app).
_EXTRA_MAP_PALETTE_OPTIONS = ["Jet", "Viridis", "Plasma", "Inferno", "Turbo",
                              "RdBu_r", "RdYlBu_r", "Greys"]

# marginTop: "16px" -- a real, visible gap above this section, not just
# the two cards' own bottom borders touching it directly. A single
# shared constant, not a one-off inline style on _extra_map_panels_
# section() alone, because *every* callback that (re)writes this
# section's own "style" Output (update_extra_map_panels_grid,
# on_extra_map_load — both below) needs to preserve it too: returning
# plain _SECTION_CARD_STYLE from those was a real bug, confirmed live
# via getBoundingClientRect() (0px measured, not 16px) — the very first
# Rows/Cols change silently wiped the margin straight back out again.
_EXTRA_MAP_SECTION_STYLE = {**_SECTION_CARD_STYLE, "marginTop": "16px"}
_EXTRA_MAP_SECTION_STYLE_HIDDEN = {**_EXTRA_MAP_SECTION_STYLE, "display": "none"}


def _extra_map_visible():
    return EXPLORER.kind in ("ifu", "mos") and EXPLORER.loaded()


def _extra_map_options():
    """[(value, label), ...] of every quantity selectable in one of this
    feature's own per-panel Map dropdowns, for whatever EXPLORER.kind
    currently is — a single flat, searchable list for both kinds
    (dcc.Dropdown is searchable by default), deliberately *not* IFU's
    own Category+Map two-level split from ifu_mod._settings_panel()
    (that split exists because the *main* map's single list can reach
    ~787 entries on a real dataset; keeping each of these smaller,
    secondary panels to one control matches the explicit "no need [for]
    extra features" request for this specific feature). MOS's
    categorical "availability" option is excluded — colour bar/scale/
    range are all continuous-only concepts, and availability has none of
    them (see mos_mod._build_source_map_figure, the categorical map this
    feature's own MOS builder deliberately doesn't reuse)."""
    if EXPLORER.kind == "ifu" and ifu_mod.STATE.data is not None:
        groups = ifu_mod._list_maptypes(ifu_mod.STATE.data)
        return [(value, label) for _cat, opts in groups for value, label in opts]
    if EXPLORER.kind == "mos" and mos_mod.STATE.data is not None:
        return [(key, label) for key, (label, *_rest) in mos_mod.COLOR_BY_OPTIONS.items()
                if key != "availability"]
    return []


def _extra_map_default_palette(maptype):
    if EXPLORER.kind == "ifu":
        return select_colorscale(maptype or "")
    if EXPLORER.kind == "mos" and maptype in mos_mod.COLOR_BY_OPTIONS:
        return mos_mod.COLOR_BY_OPTIONS[maptype][2] or "Viridis"
    return "Viridis"


def _build_extra_map_figure(maptype, scale, palette, vmin, vmax, selected_item):
    """One panel's own figure — dispatches by EXPLORER.kind, reading
    only STATE.data/_resolve_maptype/_color_by_values (all already
    -loaded, in-memory, no file I/O — see this section's own module
    -level comment). `selected_item` seeds the initial "Selected" marker
    trace only; *live* selection changes afterward are handled by a
    separate clientside restyle (extra-map-panels-selection-sync below),
    not by rebuilding this figure — see that callback's own comment."""
    import plotly.graph_objects as go

    if EXPLORER.kind == "ifu":
        st = ifu_mod.STATE.data
        if st is None or maptype is None:
            return go.Figure()
        data, title, cbar_label = ifu_mod._resolve_maptype(st, maptype)
        if data is None:
            fig = go.Figure()
            fig.update_layout(title="No data", autosize=True, height=_EXTRA_MAP_PANEL_HEIGHT)
            return fig
        # BIN_ID itself (was np.arange(...), a plain positional index) --
        # real bug, confirmed live: "selected-item" (written by both the
        # main Aladin map and this panel's own on_extra_map_panel_click)
        # is always a BIN_ID, but the cross-panel restyle sync (see
        # extra-map-panels-selection-sync below) can only compare against
        # whatever this figure's own customdata[0] actually holds --
        # comparing a BIN_ID to an unrelated row-position number matched
        # wherever the two numbers happened to coincide (or nowhere),
        # landing the highlight on a effectively arbitrary point instead
        # of the one actually clicked ("it shows the cross sign on the
        # lower right part of the additional map panels" for a centre
        # click). ifu_mod.on_map_click below is updated to match --
        # customdata[0] no longer needs its own separate BIN_ID lookup.
        indices = st["table"]["BIN_ID"]
        selected_xy = None
        if selected_item is not None:
            match = np.flatnonzero(st["table"]["BIN_ID"] == selected_item)
            if match.size:
                i = match[0]
                selected_xy = (st["table"]["XBIN"][i], st["table"]["YBIN"][i])
        return spaxel_map_figure(
            st["table"]["X"], st["table"]["Y"], data, indices,
            title=title, colorbar_label=cbar_label, selected_xy=selected_xy,
            colorscale=palette or select_colorscale(maptype), vmin=vmin, vmax=vmax,
            scale=scale or "linear", width=None, height=_EXTRA_MAP_PANEL_HEIGHT,
        )

    if EXPLORER.kind == "mos":
        st = mos_mod.STATE
        data_dict = st.data
        if data_dict is None or maptype not in mos_mod.COLOR_BY_OPTIONS:
            return go.Figure()
        _, column, default_colorscale, legend_label = mos_mod.COLOR_BY_OPTIONS[maptype]
        values = mos_mod._color_by_values(st, column)
        ra, dec = np.asarray(data_dict["ra"], dtype=np.float64), np.asarray(data_dict["dec"], dtype=np.float64)
        aps_ids = np.asarray(data_dict["class_table"]["APS_ID"])
        selected_radec = None
        if selected_item is not None:
            match = np.flatnonzero(aps_ids == selected_item)
            if match.size and np.isfinite(ra[match[0]]) and np.isfinite(dec[match[0]]):
                i = match[0]
                selected_radec = (float(ra[i]), float(dec[i]))
        return source_map_figure(
            ra, dec, values, aps_ids,
            title=legend_label, colorbar_label=legend_label, selected_radec=selected_radec,
            colorscale=palette or default_colorscale or "Viridis", vmin=vmin, vmax=vmax,
            scale=scale or "linear", width=None, height=_EXTRA_MAP_PANEL_HEIGHT,
        )

    return go.Figure()


def _extra_map_panel_div(index, cfg=None, selected_item=None):
    """One panel's own controls (Map/Scale/Palette/Min/Max) + its
    dcc.Graph, sized for a grid cell. `cfg` (optional) is one entry of a
    loaded config's own "panels" list — {"maptype","scale","palette",
    "vmin","vmax"} — used to seed this panel's initial control values on
    a fresh Load; omitted (the ordinary "just resized the grid") falls
    back to sensible per-kind defaults. A saved maptype absent from
    whatever's actually loaded right now (a config saved against a
    different dataset) falls back to the first available option instead
    of erroring — configs describe a *view*, not a specific dataset."""
    cfg = cfg or {}
    options = [{"label": label, "value": value} for value, label in _extra_map_options()]
    valid_values = {opt["value"] for opt in options}
    maptype = cfg.get("maptype")
    if maptype not in valid_values:
        maptype = options[0]["value"] if options else None
    scale = cfg.get("scale") if cfg.get("scale") in aps_style.COLOR_SCALE_OPTIONS else "linear"
    palette = cfg.get("palette") or _extra_map_default_palette(maptype)
    vmin = cfg.get("vmin")
    vmax = cfg.get("vmax")
    return html.Div([
        dcc.Dropdown(
            id={"type": "extra-map-type-dd", "index": index}, options=options, value=maptype,
            clearable=False, searchable=True, style={"fontSize": "11px", "marginBottom": "4px"},
        ),
        html.Div([
            dcc.Dropdown(
                id={"type": "extra-map-scale-dd", "index": index},
                options=[{"label": v, "value": k} for k, v in aps_style.COLOR_SCALE_OPTIONS.items()],
                value=scale, clearable=False,
                style={"width": "88px", "fontSize": "11px", "display": "inline-block"},
            ),
            dcc.Dropdown(
                id={"type": "extra-map-palette-dd", "index": index},
                options=[{"label": p, "value": p} for p in _EXTRA_MAP_PALETTE_OPTIONS],
                value=palette, clearable=False,
                style={"width": "104px", "fontSize": "11px", "display": "inline-block", "marginLeft": "4px"},
            ),
            dcc.Input(
                id={"type": "extra-map-vmin", "index": index}, type="number", placeholder="Min", value=vmin,
                style={"width": "56px", "marginLeft": "4px", "fontSize": "11px"},
            ),
            dcc.Input(
                id={"type": "extra-map-vmax", "index": index}, type="number", placeholder="Max", value=vmax,
                style={"width": "56px", "marginLeft": "4px", "fontSize": "11px"},
            ),
        ], style={"marginBottom": "4px", "display": "flex", "alignItems": "center"}),
        dcc.Graph(
            id={"type": "extra-map-graph", "index": index}, config=aps_style.GRAPH_CONFIG,
            figure=_build_extra_map_figure(maptype, scale, palette, vmin, vmax, selected_item),
            # Explicit CSS height on the *wrapper* div — real bug, found
            # only by scrolling a live page, not visible from the figure
            # JSON alone: the figure's own `autosize=True` (see
            # spaxel_map_figure's own comment for why it's there — fills
            # available *width*) makes Plotly measure its actual DOM
            # container for *both* dimensions, not just width. Without
            # this, that container had no CSS height of its own at all,
            # so Plotly collapsed the whole plot down to ~76px tall
            # regardless of this figure's own `height=330` — confirmed
            # live via getBoundingClientRect(), not guessed — and the
            # panel's outer card shrank to match, while the *actual*
            # rendered SVG/modebar still occupied their normal on-screen
            # size, visibly bleeding into whatever content came next on
            # the page (explicit report: "the BIN and Spaxel data tables
            # overlap that additional maps panel completely... while you
            # scroll down the page, the bin and spaxel overlay them").
            # Giving the wrapper a real, explicit height fixes both the
            # collapse and the overlap in one place — width is still
            # left unstyled here, so it keeps filling the grid cell.
            style={"height": f"{_EXTRA_MAP_PANEL_HEIGHT}px"},
        ),
    ], style={"border": "1px solid var(--pyaps-line)", "borderRadius": "var(--pyaps-radius-m)", "padding": "6px"})


def _extra_map_panels_grid(rows, cols, panels_cfg, selected_item):
    rows = int(rows or 0)
    cols = int(cols or 0)
    # Explicit follow-up: "when I load the page I do not want to see
    # even one additional map is loaded... Just want to see the panel
    # show number of rows and columns." 0 (either dimension) is now a
    # real, deliberate "show nothing yet" state, not clamped up to 1 the
    # way this used to unconditionally do — a fresh page genuinely shows
    # zero panels until the user picks a real grid size.
    if rows <= 0 or cols <= 0:
        return html.Div()
    n = rows * cols
    panels_cfg = panels_cfg or []
    return html.Div(
        [_extra_map_panel_div(i, panels_cfg[i] if i < len(panels_cfg) else None, selected_item)
         for i in range(n)],
        style={"display": "grid", "gridTemplateColumns": f"repeat({cols}, 1fr)",
               "gap": "10px"},
    )


def _extra_map_panels_section():
    """Always present in the base layout (not conditional on
    EXPLORER.kind) — same "avoid the missing-component gotcha" reason
    color-range-version/flux-cube-camera-store etc. already are (see
    update_map_mode's own docstring) — hidden via its own style Output
    instead for L1/not-loaded, driven by the same callback that
    (re)builds the grid.

    Rows/Cols both default to 0 ("show nothing yet" — see
    _extra_map_panels_grid's own comment) with an explicit "0" option,
    per the explicit follow-up request. Save/Load buttons and the
    filename field use style.SOFT_BUTTON_STYLE/SOFT_INPUT_HEIGHT_STYLE
    (see that module's own comment) instead of unstyled `html.Button`s —
    explicit follow-up: "the color for ba[ck]ground of save layout and
    load layout is not well fitted to the dark mode... the bottons are
    re[c]tangular b[u]t I see other bottons like expand... some[how]
    soft edge round[ish]." The panel grid itself is wrapped in
    dcc.Loading — explicit follow-up: "while we update the row and
    columns... the page show[s] no activit[y] like nothing has
    happened... I need some sign of loading.\""""
    return html.Div([
        html.Div([
            html.Label("Additional map panels:", style={"fontWeight": "600", "marginRight": "14px"}),
            html.Div([
                html.Label("Rows:", style={"marginRight": "4px", "fontSize": "12px"}),
                dcc.Dropdown(id="extra-map-rows-dd",
                             options=[{"label": str(i), "value": i} for i in range(0, _EXTRA_MAP_MAX_ROWS + 1)],
                             value=0, clearable=False,
                             style={"width": "64px", "display": "inline-block", "fontSize": "12px"}),
            ], style={"display": "inline-block", "marginRight": "14px"}),
            html.Div([
                html.Label("Columns:", style={"marginRight": "4px", "fontSize": "12px"}),
                dcc.Dropdown(id="extra-map-cols-dd",
                             options=[{"label": str(i), "value": i} for i in range(0, _EXTRA_MAP_MAX_COLS + 1)],
                             value=0, clearable=False,
                             style={"width": "64px", "display": "inline-block", "fontSize": "12px"}),
            ], style={"display": "inline-block", "marginRight": "18px"}),
            # Explicit follow-up: "I want when we hit save[] or load
            # layout we see the option to select a name; it is not good
            # that the default name is [on] the main page." A persistent
            # text field on the page (the first attempt at "save to a
            # specific filename") was itself the complaint, not the
            # solution — replaced with a real prompt at the moment of
            # saving instead (see the clientside callback below),
            # exactly like a native "Save As" dialog's own filename
            # field, not a permanent page element. dcc.Download still
            # can't open a genuine filesystem-path picker (no web page
            # can), so the prompt's own text is the filename the
            # download actually uses — the closest achievable version.
            html.Button("Save layout", id="extra-map-save-btn", n_clicks=0,
                        style={**aps_style.SOFT_BUTTON_STYLE, "marginRight": "8px"}),
            dcc.Upload(
                id="extra-map-upload",
                children=html.Button("Load layout", style=aps_style.SOFT_BUTTON_STYLE),
                multiple=False,
            ),
            dcc.Download(id="extra-map-download"),
            dcc.Store(id="extra-map-save-filename-store"),
            html.Div(id="extra-map-load-error", style={"color": "var(--pyaps-error)", "fontSize": "11px",
                                                         "marginLeft": "10px"}),
        ], style={"display": "flex", "alignItems": "flex-end", "flexWrap": "wrap", "gap": "6px",
                  "padding": "10px 0"}),
        # Dummy Output for the cross-panel selection-sync clientside
        # callback — that callback's real work is a Plotly.restyle() side
        # effect, not a Dash prop write (same established pattern as
        # flux-cube-wave-filter-dummy elsewhere in this file).
        html.Div(id="extra-map-selection-sync-dummy", style={"display": "none"}),
        dcc.Loading(
            **aps_style.LOADING_KWARGS,
            # target_components scopes the spinner to *only* the grid
            # -rebuild output (Rows/Cols change or a Load) -- explicit
            # follow-up bug report: with no target_components, dcc.
            # Loading shows (and, worse, remounts) its *entire* subtree
            # whenever *any* descendant callback is in flight, including
            # each individual panel's own MATCH-scoped figure update --
            # "when I change the content of each of the additional
            # panels, all other additional panels get reloaded and
            # disappear and appear again" and "when I change the scale
            # on each of the additional panels, it reload[s] all other
            # additional panels" were this, exactly: a single panel's own
            # figure recompute was blanket-flagging the whole grid as
            # loading, remounting every *other* panel's Graph along with
            # it and wiping out whatever cross-panel-selection restyle
            # had been applied to them client-side (see this feature's
            # own selection-sync callback) -- "after a few clicks the
            # connection... is lost" was the direct, cumulative result.
            # Scoping to just this one Output means a per-panel edit no
            # longer touches the grid's own loading state at all.
            target_components={"extra-map-panels-grid": "children"},
            children=[html.Div(id="extra-map-panels-grid")],
        ),
    ], id="extra-map-panels-section", style=_EXTRA_MAP_SECTION_STYLE)


def _tabs_for_current(selected_item):
    if EXPLORER.kind == "ifu":
        return ifu_mod._available_tabs(ifu_mod.STATE.data)
    if EXPLORER.kind == "mos":
        return mos_mod._available_tabs(mos_mod.STATE.data, selected_item)
    if EXPLORER.kind == "l1":
        return [dcc.Tab(label="Spectra", value="spectra", style=aps_style.TAB_STYLE,
                         selected_style=aps_style.TAB_SELECTED_STYLE),
                dcc.Tab(label="FWHM", value="fwhm", style=aps_style.TAB_STYLE,
                         selected_style=aps_style.TAB_SELECTED_STYLE),
                # Explicit request: "another tab to display the primary
                # header... preferably after fwhm tab".
                dcc.Tab(label="Header", value="header", style=aps_style.TAB_STYLE,
                         selected_style=aps_style.TAB_SELECTED_STYLE)]
    return [dcc.Tab(label="No dataset loaded", value="none", style=aps_style.TAB_STYLE,
                     selected_style=aps_style.TAB_SELECTED_STYLE)]


def _sky_fov_deg(ra_arr, dec_arr):
    """A reasonable Aladin field-of-view (degrees) sized to the actual
    RA/Dec spread of the loaded dataset, padded 30%, floored/capped to
    sane bounds — same "field-of-view sized to the data" idea as
    aps_l1_preview.compute_ra_dec_range, but returning a plain float
    since Aladin Lite's setFov()/init `fov` option wants degrees, not a
    "45arcsec"-style string."""
    ra_arr = np.asarray(ra_arr, dtype=np.float64)
    dec_arr = np.asarray(dec_arr, dtype=np.float64)
    finite = np.isfinite(ra_arr) & np.isfinite(dec_arr)
    if finite.sum() < 2:
        return 0.05
    cos_dec = np.cos(np.radians(np.nanmean(dec_arr[finite])))
    ra_span = float(np.nanmax(ra_arr[finite]) - np.nanmin(ra_arr[finite])) * cos_dec
    dec_span = float(np.nanmax(dec_arr[finite]) - np.nanmin(dec_arr[finite]))
    span = max(ra_span, dec_span, 0.01)
    return min(span * 1.3, 3.0)


def _aladin_target(selected_item):
    """(ra, dec, fov) in degrees to recenter the Aladin panel on, or None
    if not computable — dispatches by kind, reused by both the panel's own
    recentring and the info-box readout below it."""
    if EXPLORER.kind == "l1":
        st = l1_mod.STATE
        if not st.loaded():
            return None
        idx = st.targs_apstoid.get(selected_item)
        ra = dec = None
        if idx is not None:
            targ = st.targs[idx]
            ra, dec = float(targ.targra), float(targ.targdec)
        # Many fibres in a real L1 file are unused sky/calib slots with no
        # target coordinates at all (confirmed on real data — TARGRA/DEC
        # NaN, empty TARGUSE); fall back to the dataset's mean position
        # rather than sending Aladin a NaN target.
        if ra is None or not np.isfinite(ra) or not np.isfinite(dec):
            ra = float(np.nanmean(st.coord_arr[:, 0]))
            dec = float(np.nanmean(st.coord_arr[:, 1]))
        return {"ra": ra, "dec": dec, "fov": _sky_fov_deg(st.coord_arr[:, 0], st.coord_arr[:, 1])}

    if EXPLORER.kind == "ifu":
        data = ifu_mod.STATE.data
        if data is None:
            return None
        # PATCH_TABLE.X_0/Y_0 are the absolute sky RA/Dec (deg) of the
        # field centre — confirmed constant across every row on real data
        # (e.g. X_0=183.576/Y_0=59.615 for CNAME WVE_12141807+5936554,
        # matching that CNAME's decoded RA/Dec exactly). X/Y are
        # cos(dec)-corrected arcsec offsets from that centre, so their
        # span converts straight to a sky-degree FoV.
        rec = data["patch_table_rec"]
        ra0 = float(np.nanmean(rec.X_0))
        dec0 = float(np.nanmean(rec.Y_0))
        cos_dec = np.cos(np.radians(dec0)) or 1.0
        table = data["table"]
        x_span = float(np.nanmax(table["X"]) - np.nanmin(table["X"])) / 3600.0
        y_span = float(np.nanmax(table["Y"]) - np.nanmin(table["Y"])) / 3600.0
        fov = min(max(x_span, y_span, 0.01) * 1.3, 3.0)
        ra, dec = ra0, dec0
        if selected_item is not None:
            bin_ids = np.asarray(table["BIN_ID"])
            match = np.flatnonzero(bin_ids == selected_item)
            if match.size:
                # XBIN/YBIN: the *bin's* centroid (same value repeated
                # across every spaxel belonging to that bin), not an
                # arbitrary member spaxel's raw X/Y — matches the
                # "Selected" marker convention _build_map_figure uses (see
                # its own comment). Without this, every recentre landed on
                # the constant field centre regardless of which spaxel/bin
                # was actually clicked, since X_0/Y_0 never vary.
                i = match[0]
                ra = ra0 - float(table["XBIN"][i]) / (3600.0 * cos_dec)
                dec = dec0 + float(table["YBIN"][i]) / 3600.0
        return {"ra": ra, "dec": dec, "fov": fov}

    if EXPLORER.kind == "mos":
        data = mos_mod.STATE.data
        if data is None:
            return None
        ra_arr, dec_arr = np.asarray(data["ra"]), np.asarray(data["dec"])
        aps_ids = np.asarray(data["class_table"]["APS_ID"])
        match = np.flatnonzero(aps_ids == selected_item)
        if match.size:
            i = match[0]
            ra, dec = float(ra_arr[i]), float(dec_arr[i])
        else:
            ra, dec = float(np.nanmean(ra_arr)), float(np.nanmean(dec_arr))
        return {"ra": ra, "dec": dec, "fov": _sky_fov_deg(ra_arr, dec_arr)}

    return None


_ALADIN_N_COLOR_BUCKETS = 16

# Colour-scale ("stretch") options — canonical definitions now live in
# apsPlot/style.py (see its own comment for why: aps_l1_preview.py's Slit
# Explorer needs the exact same stretch a user picked here, and can't
# import this module without a circular dependency). Kept as aliases under
# these names since existing call sites/tests in this file already use
# them directly.
COLOR_SCALE_OPTIONS = aps_style.COLOR_SCALE_OPTIONS
_apply_color_scale = aps_style.apply_color_scale

# Shared colour-PALETTE override — explicit request: "is it possible to
# instead of using the current color map, we use something that is black
# for low signal and white for high signal... so make it more realistic.
# Can I have it in all maps either 2D or 3D so we can switch between this
# blue to red to black to white?" Deliberately a plain 2-stop
# black-to-white colourscale (not Plotly's built-in "Greys", which by
# default runs the *opposite* direction — light/white at the low end,
# dark at the high end — confirmed by checking its own documented stop
# colours before use, not assumed) so "dark means low signal" is
# guaranteed regardless of any particular named colourscale's own
# convention.
_GREYSCALE_COLORSCALE = [[0.0, "rgb(0,0,0)"], [1.0, "rgb(255,255,255)"]]
# Follow-up request: "add a pallet that low signal is white and high
# signal is [get]ting red or something... because the background is
# always white for 3d and if I have a palette that shows low signal
# with white[,] it is more beaut[if]ul." Plotly's own built-in "Reds"
# colourscale already runs exactly this direction — confirmed directly
# (`rgb(255,245,240)` i.e. near-white at fraction 0, `rgb(103,0,13)` a
# deep red at fraction 1) — unlike "Greys" above, no custom colourscale
# needed here.
PALETTE_OPTIONS = {
    "default": "Default (per-quantity)",
    "jet": "Blue → Red (Jet)",
    "greyscale": "Black → White (mono)",
    "white_red": "White → Red (hot)",
}


def _resolve_palette_override(default_colorscale):
    """Every per-kind colour-by/maptype path already resolves its own
    "natural" colourscale (Jet for L1 flux, Plasma for S/N, whatever an
    IFU maptype's own `select_colorscale` returns, ...) — this is the one
    shared place all of them route back through afterward, so
    EXPLORER.color_palette applies uniformly to every map (2D Aladin
    catalog buckets, the Aladin legend bar, and the 3D flux cube) without
    each one needing its own copy of this same three-way branch."""
    palette = EXPLORER.color_palette
    if palette == "jet":
        return "Jet"
    if palette == "greyscale":
        return _GREYSCALE_COLORSCALE
    if palette == "white_red":
        return "Reds"
    return default_colorscale  # "default" (or anything unrecognized): unchanged per-kind behaviour


def _bucket_by_color(items, ras, decs, values, vmin, vmax, colorscale, scale="linear"):
    """Groups points into a small number of discrete colour buckets, one
    Aladin catalog per bucket. Aladin Lite v3's *confirmed* API only
    documents catalog-wide colour (`Catalog.setColor()`); per-source colour
    isn't clearly documented either way, so rather than guess at an
    unconfirmed `A.source(..., {color: ...})` option, this reproduces a
    continuous-looking colourmap using only the mechanism that's certain
    to work — visually a very close approximation at 16 buckets, exactly
    matching the Plotly map's colourscale/vmin/vmax."""
    import plotly.colors as pc

    values = np.asarray(values, dtype=np.float64)
    if vmin is None or vmax is None:
        finite = values[np.isfinite(values)]
        lo, hi = np.percentile(finite, [1, 99]) if finite.size else (0.0, 1.0)
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax
    span = (vmax - vmin) or 1.0
    frac = np.clip((values - vmin) / span, 0.0, 1.0)
    frac = _apply_color_scale(frac, scale)
    frac = np.nan_to_num(frac, nan=0.0)
    bin_idx = np.clip((frac * _ALADIN_N_COLOR_BUCKETS).astype(int), 0, _ALADIN_N_COLOR_BUCKETS - 1)
    bucket_fracs = [(i + 0.5) / _ALADIN_N_COLOR_BUCKETS for i in range(_ALADIN_N_COLOR_BUCKETS)]
    bucket_colors = pc.sample_colorscale(colorscale, bucket_fracs)

    buckets = {}
    for i in range(len(values)):
        if not (np.isfinite(ras[i]) and np.isfinite(decs[i])):
            continue
        b = int(bin_idx[i])
        buckets.setdefault(b, []).append({"ra": float(ras[i]), "dec": float(decs[i]), "item": int(items[i])})
    return [{"color": bucket_colors[b], "points": pts} for b, pts in buckets.items()]


def _l1_active_color_params(st):
    """(values, vmin, vmax, colorscale) for whatever L1's own "Colour
    Aladin points by"/colour-range/scale controls currently say — the
    exact numbers the main L1 catalog overlay itself is coloured by.
    Factored out of _aladin_catalog_points's own L1 branch so
    _contrib_exposures_payload (the "Show contributing exposures"
    overlay) can colour its own points by the *identical* values —
    explicit request: "I want the color code [to] come again from the
    total flux or S/N or whatever the main plot is" — without either
    call site's own vmin/vmax-default logic being able to drift out of
    sync with the other's over time."""
    _, colorscale, default_vmin = l1_mod.COLOR_BY_OPTIONS[st.color_by]
    values = l1_mod._color_by_values(st, st.color_by)
    # default_vmin (12.0 for flux, matching the historical
    # fiber_map_figure default; 0.0 for S/N) rather than
    # _bucket_by_color's generic percentile fallback, so the catalog
    # overlay's colours agree with the legend below it.
    finite = values[np.isfinite(values)]
    vmin = st.color_vmin if st.color_vmin is not None else default_vmin
    vmax = st.color_vmax if st.color_vmax is not None else (float(np.nanmax(finite)) if finite.size else 1.0)
    return values, vmin, vmax, colorscale


def _aladin_catalog_points():
    """{"buckets": [{"color", "points": [{"ra","dec","item"}, ...]}, ...]}
    for the Aladin overlay — dispatched by kind like _aladin_target,
    reusing each kind's own colour logic (the same one the now-removed
    Plotly map used to use) so the overlay is coloured consistently
    (bucketed, not per-point — see _bucket_by_color for why)."""
    if EXPLORER.kind == "ifu":
        data = ifu_mod.STATE.data
        if data is None:
            return []
        table = data["table"]
        values, _, _ = ifu_mod._resolve_maptype(data, ifu_mod.STATE.current_maptype)
        if values is None:
            return []
        colorscale = _resolve_palette_override(ifu_mod.select_colorscale(ifu_mod.STATE.current_maptype))
        # RA/Dec per spaxel: PATCH_TABLE.X_0/Y_0 are the *constant* field
        # centre (same value every row, confirmed on real data), and X/Y
        # are already cos(dec)-corrected arcsec offsets from that centre
        # (aps_ifu_v0.py: `cube['x'] = -dx_deg*3600*cos(dec)`, Y
        # uncorrected) — see _load_aps_fits's own comment for the same
        # convention. Inverting: RA = X_0 - X/(3600*cos(dec)), Dec = Y_0 + Y/3600.
        ra0 = float(np.nanmean(data["patch_table_rec"].X_0))
        dec0 = float(np.nanmean(data["patch_table_rec"].Y_0))
        cos_dec = np.cos(np.radians(dec0)) or 1.0
        ra = ra0 - np.asarray(table["X"], dtype=np.float64) / (3600.0 * cos_dec)
        dec = dec0 + np.asarray(table["Y"], dtype=np.float64) / 3600.0
        return _bucket_by_color(table["BIN_ID"], ra, dec, values,
                                 ifu_mod.STATE.color_vmin, ifu_mod.STATE.color_vmax, colorscale,
                                 scale=ifu_mod.STATE.color_scale)

    if EXPLORER.kind == "l1":
        st = l1_mod.STATE
        if not st.loaded():
            return []
        values, vmin, vmax, colorscale = _l1_active_color_params(st)
        return _bucket_by_color(st.aps_id_arr, st.coord_arr[:, 0], st.coord_arr[:, 1], values,
                                 vmin, vmax, _resolve_palette_override(colorscale), scale=st.color_scale)

    if EXPLORER.kind == "mos":
        data = mos_mod.STATE.data
        if data is None:
            return []
        ra_arr, dec_arr = np.asarray(data["ra"]), np.asarray(data["dec"])
        aps_ids = np.asarray(data["class_table"]["APS_ID"])
        color_by = mos_mod.STATE.color_by

        if color_by == "availability" or color_by not in mos_mod.COLOR_BY_OPTIONS:
            # Categorical (Gal/ExGal/Both/None) — one exact bucket per
            # category via the same _AVAIL_STYLE colours the Plotly map
            # itself used to use, no continuous-colourmap approximation.
            buckets = {}
            for i, aid in enumerate(aps_ids):
                if not (np.isfinite(ra_arr[i]) and np.isfinite(dec_arr[i])):
                    continue
                av = mos_mod._target_availability(data, int(aid))
                has_gal = av["rvs"] or av["ferre"]
                has_exgal = av["ppxf"] or av["emi"]
                key = "both" if (has_gal and has_exgal) else "gal" if has_gal else "exgal" if has_exgal else "none"
                buckets.setdefault(key, []).append(
                    {"ra": float(ra_arr[i]), "dec": float(dec_arr[i]), "item": int(aid)}
                )
            return [{"color": mos_mod._AVAIL_STYLE[key][1], "points": pts} for key, pts in buckets.items()]

        # Continuous alternative (redshift/S/N) — reuses the same bucketed
        # approach as IFU/L1, per explicit user request for more than just
        # the one categorical colouring option.
        _, column, colorscale, _ = mos_mod.COLOR_BY_OPTIONS[color_by]
        values = mos_mod._color_by_values(mos_mod.STATE, column)
        return _bucket_by_color(aps_ids, ra_arr, dec_arr, values,
                                 mos_mod.STATE.color_vmin, mos_mod.STATE.color_vmax,
                                 _resolve_palette_override(colorscale),
                                 scale=mos_mod.STATE.color_scale)

    return []


def _l1_flux_cube_data():
    """(ra, dec, wave, flux_matrix, items, fibre_radius_deg, shape) for
    the 3D flux-cube view — L1's own `STATE.targs` already carries a real
    per-target spectrum (one `APSSPEC` per arm, `.wave`/`.flux`), so no
    new data needs loading; arms are concatenated end-to-end per target
    (same convention `_build_spectra_figure`'s own flux/ivar dicts use)
    to give one composite spectrum spanning every loaded arm. Confirmed
    on real data that every target shares byte-identical wave grids per
    arm (same instrument setup within one load) — targets are still
    checked individually and dropped (not just assumed) if any turns out
    not to match, rather than letting a ragged-array error surface from
    deep inside numpy.

    `shape` follows `is_fibre_level()` exactly — a real "circle" fibre
    aperture for single-exposure MOS/MOSLIFU/MOSMIFU, a "square" WCS
    grid cell for a stacked/co-added cube (no single real fibre per
    spatial position there at all — same distinction the Slit Explorer
    and "True fibre/spaxel size" overlay already gate on) — explicit
    request: "is it possible to have square spaxels instead of circle in
    case of no fibre mode.\""""
    st = l1_mod.STATE
    if not st.loaded():
        return None
    n_arms = len(st.targs[0].spectra)
    ref_wave = np.concatenate([st.targs[0].spectra[i].wave for i in range(n_arms)])
    n_wave = len(ref_wave)

    ra_list, dec_list, item_list, flux_rows = [], [], [], []
    for t in st.targs:
        try:
            if len(t.spectra) != n_arms:
                continue
            row = np.concatenate([t.spectra[i].flux for i in range(n_arms)])
            if len(row) != n_wave:
                continue
        except Exception:
            continue
        ra_list.append(t.targra)
        dec_list.append(t.targdec)
        item_list.append(t.aps_id)
        flux_rows.append(row)
    if not flux_rows:
        return None

    shape = "circle" if st.is_fibre_level() else "square"
    # The real WEAVE fibre *aperture* diameter only means something for
    # `shape="circle"` (a genuine single fibre per position). For a
    # stacked/co-added cube (`shape="square"`), the relevant size is the
    # *reconstructed cube's* own WCS pixel spacing — a different, usually
    # much smaller, number that this fibre-diameter lookup was never
    # actually measuring. Passing it through unconditionally used to make
    # every square-mode voxel several times too large, heavily
    # overlapping its neighbours — confirmed live: "I guess if you select
    # the right spaxel size these should not have overlapped at all as
    # they are blocks." `flux_cube_figure`'s own square branch now always
    # measures the real spacing directly from the data instead (see
    # `_auto_square_radius_deg`), so `None` here just means "no
    # meaningful fallback available if that measurement can't run" —
    # never the wrong physical size.
    if shape == "circle":
        mode = st.targs[0].meta[0].get("mode") if st.targs[0].meta else None
        diam_arcsec = aps_constants.WEAVE_FIBRE_DIAMETER_ARCSEC.get(mode)
        fibre_radius_deg = (diam_arcsec / 2.0 / 3600.0) if diam_arcsec else None
    else:
        fibre_radius_deg = None

    return (np.array(ra_list), np.array(dec_list), ref_wave, np.array(flux_rows),
            np.array(item_list), fibre_radius_deg, shape)


def _mos_flux_cube_data():
    """Same shape as _l1_flux_cube_data, sourced from MOS's own Redrock
    fit inputs (`class_spec`'s `LAMBDA_RR_<arm>`/`FLUX_RR_<arm>`
    columns — the one spectral source essentially every MOS target
    actually has, unlike STAR_SPEC/GALAXY_SPEC which only cover
    whichever subset turned out Galactic/ExGal) — arms concatenated the
    same way `_redrock_figure` itself builds its own wave/flux dicts.
    `st["ra"]`/`st["dec"]` are already aligned row-for-row with
    `class_table`'s own `APS_ID` order (the same array `_build_source_
    map_figure` indexes directly), so no separate position lookup is
    needed beyond that existing convention. MOS mode is real fibres by
    definition (see aps_l1_preview.AppState.is_fibre_level's own
    docstring — MOS/MOSLIFU/MOSMIFU all count), so always "circle."""
    st = mos_mod.STATE.data
    if not st:
        return None
    arms = st["arms"]
    cs = st["class_spec"]
    pos_map = st["class_spec_pos"]
    aps_ids = np.asarray(st["class_table"]["APS_ID"])
    ra_arr, dec_arr = st["ra"], st["dec"]

    ref_wave = None
    ra_list, dec_list, item_list, flux_rows = [], [], [], []
    for i, aid in enumerate(aps_ids):
        aid = int(aid)
        spos = pos_map.get(aid)
        if spos is None or not (np.isfinite(ra_arr[i]) and np.isfinite(dec_arr[i])):
            continue
        try:
            wave = np.concatenate([np.asarray(cs[f"LAMBDA_RR_{a}"][spos]) for a in arms])
            flux = np.concatenate([np.asarray(cs[f"FLUX_RR_{a}"][spos]) for a in arms])
        except Exception:
            continue
        if ref_wave is None:
            ref_wave = wave
        elif len(wave) != len(ref_wave):
            continue
        ra_list.append(ra_arr[i])
        dec_list.append(dec_arr[i])
        item_list.append(aid)
        flux_rows.append(flux)
    if not flux_rows:
        return None

    diam_arcsec = aps_constants.WEAVE_FIBRE_DIAMETER_ARCSEC.get("MOS")
    fibre_radius_deg = diam_arcsec / 2.0 / 3600.0

    return (np.array(ra_list), np.array(dec_list), ref_wave, np.array(flux_rows),
            np.array(item_list), fibre_radius_deg, "circle")


def _ifu_flux_cube_data():
    """Same shape again, sourced from IFU's own `PATCH_BINSPEC` extension
    — already loaded as a full (n_bins, n_wave) matrix (`st["spectra"]`)
    plus its shared *log*-wavelength grid (`st["lambda_"]`, `np.exp()`'d
    here — the established `PATCH_BINSPEC`-is-log-rebinned convention
    already used by every PPXF/EMI figure builder in this module).
    Per-*spaxel* resolution (one x/y point per `PATCH_TABLE` row, several
    spaxels legitimately sharing one bin's own spectrum), not deduped to
    one point per unique bin — matching the main 2D catalog's own
    per-spaxel `_aladin_catalog_points` convention exactly, since the
    explicit requirement is "x and y... as it is now." A PATCH_TABLE row
    is always a WCS spaxel, never a real fibre — always "square"."""
    st = ifu_mod.STATE.data
    if not st or not st.get("binspec_pos") or st.get("spectra") is None:
        return None
    table = st["table"]
    ra0 = float(np.nanmean(st["patch_table_rec"].X_0))
    dec0 = float(np.nanmean(st["patch_table_rec"].Y_0))
    cos_dec = np.cos(np.radians(dec0)) or 1.0
    ra_all = ra0 - np.asarray(table["X"], dtype=np.float64) / (3600.0 * cos_dec)
    dec_all = dec0 + np.asarray(table["Y"], dtype=np.float64) / 3600.0
    bin_ids_all = np.asarray(table["BIN_ID"])

    pos_map = st["binspec_pos"]
    rows = np.array([pos_map.get(int(b), -1) for b in bin_ids_all])
    valid = (rows >= 0) & np.isfinite(ra_all) & np.isfinite(dec_all)
    if not valid.any():
        return None

    wave = np.exp(np.asarray(st["lambda_"], dtype=np.float64))
    flux_matrix = np.asarray(st["spectra"])[rows[valid]]
    pixelsize_arcsec = st.get("pixelsize")
    fibre_radius_deg = (pixelsize_arcsec / 2.0 / 3600.0) if pixelsize_arcsec else None

    return (ra_all[valid], dec_all[valid], wave, flux_matrix, bin_ids_all[valid],
            fibre_radius_deg, "square")


def _flux_cube_data():
    """Dispatched by kind, mirroring _aladin_catalog_points's own
    dispatch pattern — returns `None` whenever there's nothing to build a
    cube from (nothing loaded yet, or the per-kind helper found no usable
    spectra)."""
    if EXPLORER.kind == "l1":
        return _l1_flux_cube_data()
    if EXPLORER.kind == "mos":
        return _mos_flux_cube_data()
    if EXPLORER.kind == "ifu":
        return _ifu_flux_cube_data()
    return None


def _true_size_radius_deg():
    """Real angular radius (degrees) of one fibre/spaxel on the sky, for
    the optional "true fibre/spaxel size" Aladin overlay mode — explicit
    user request: "instead of a fixed-size symbol, a circle with the size
    of the fibre width, so I can see exactly how [each] fibre [maps] on
    the sky." None wherever there's no single well-defined size to show:
    a stacked L1 cube has no one fibre per spatial position (see
    l1_mod.AppState.is_fibre_level's own docstring), and an unrecognized
    MOS obsmode has no entry in aps_constants.WEAVE_FIBRE_DIAMETER_ARCSEC
    to fall back on (never guessed). Also None whenever the toggle itself
    (EXPLORER.aladin_true_size) is off — the normal, default state."""
    if not EXPLORER.aladin_true_size:
        return None

    if EXPLORER.kind == "mos":
        obsmode = (EXPLORER.file_info or {}).get("obsmode")
        diam_arcsec = aps_constants.WEAVE_FIBRE_DIAMETER_ARCSEC.get(obsmode)
        return (diam_arcsec / 2.0 / 3600.0) if diam_arcsec else None

    if EXPLORER.kind == "l1":
        if not l1_mod.STATE.is_fibre_level():
            return None
        mode = l1_mod.STATE.targs[0].meta[0].get("mode")
        diam_arcsec = aps_constants.WEAVE_FIBRE_DIAMETER_ARCSEC.get(mode)
        return (diam_arcsec / 2.0 / 3600.0) if diam_arcsec else None

    if EXPLORER.kind == "ifu":
        data = ifu_mod.STATE.data
        pixsize_arcsec = data.get("pixelsize") if data else None
        return (pixsize_arcsec / 2.0 / 3600.0) if pixsize_arcsec else None

    return None


def _aladin_catalog_payload():
    """The one and only place that builds Output("aladin-catalog-data")'s
    value — every callback that touches this Store (maptype/colour-by/
    colour-range/colour-scale/true-size-toggle changes, the initial load)
    must go through this, not hand-roll `{"buckets": ...}` itself. Same
    "Dash replaces the whole prop on every Output, never merges it" trap
    this codebase has already been bitten by more than once (see
    _load_panel_style/_main_panel_style's own comments) — with `buckets`
    and `radius_deg` both living in this one Store, a callback that only
    remembered to set one would silently blank out the other on its next
    fire."""
    return {"buckets": _aladin_catalog_points(), "radius_deg": _true_size_radius_deg()}


# Real angular radius (degrees), same convention/reasoning as
# _true_size_radius_deg — a contributing-exposure fibre is always a real
# single-exposure LIFU/MIFU fibre (the only case this feature applies to
# at all), so LIFU's own real fibre radius is what's drawn. This feature
# went through two earlier designs before landing here, both dropped
# after direct live-testing reports, not guesses: real-size circles
# coloured *per file* (indistinguishable from this same map's own flux/
# S-N colour-coding — "blue red and green... not distinguishable at
# all"), then fixed-pixel A.catalog markers varying *shape* per file
# instead of colour (rejected outright — "they cannot reflect the real
# shape of the fibre and its diameter... so they are useless"). This
# version colours every point by the *same* flux/S-N value the main
# catalog itself is coloured by (see _l1_active_color_params/
# _contrib_exposures_payload) and distinguishes *files* only via the
# separate checklist below the map, never via the points' own
# appearance — real angular size, real data colour, nothing invented.
_CONTRIB_MARKER_RADIUS_DEG = aps_constants.WEAVE_FIBRE_DIAMETER_ARCSEC["MOSLIFU"] / 2.0 / 3600.0


def _contrib_exposures_row_style():
    """Show/hide the "Show contributing exposures" checkbox row — only
    ever meaningful for an L1 stacked/superstacked/cube load (see
    l1_mod.AppState.is_fibre_level's own docstring for why fibre-level
    data has nothing to resolve here). Same "toggle visibility via a
    dedicated style-helper, never destroy/recreate the component itself"
    pattern as _load_panel_style/_main_panel_style — keeps the checkbox's
    own on/off state intact across a dataset reload, matching how
    aladin-true-size/aladin-show-dss already behave (both of those
    just never need hiding at all, since they apply to every kind —
    this is the first control here that does)."""
    applicable = (
        EXPLORER.kind == "l1" and l1_mod.STATE.loaded() and not l1_mod.STATE.is_fibre_level()
    )
    return {"display": "block" if applicable else "none", "marginTop": "8px"}


def _contrib_exposures_payload():
    """Per-contributing-single-exposure-file fibre positions for the
    optional "Show contributing exposures" Aladin overlay — explicit user
    request: on a stacked/superstacked/cube L1 load, see where each
    individual exposure's own fibres actually were, not just the final
    combined positions. `None` whenever not applicable (not L1, nothing
    loaded, fibre-level data with nothing to resolve, the checkbox is
    off, or no provenance could be found) so the clientside callback can
    cheaply no-op/clear its layers.

    This feature's *fourth* rendering design, the previous three each
    dropped after direct, correct live-testing reports rather than
    guesses: (1) `A.circle`/`A.graphicOverlay`, real angular size, one
    fixed colour per file — "not distinguishable at all... because color
    has been us[ed] for the flux and SNR"; (2) `A.catalog` markers
    varying *shape* per file — "they cannot reflect the real shape of the
    fibre and its diameter... so they are useless"; (3) real angular
    size + colour-by-bucket (matching the main catalog), but *merged*
    across every currently-selected file into one set of bucket overlays,
    rebuilt wholesale (`removeLayer` every existing shape, `A.circle` every
    point again) on every single per-file checkbox tick/untick — a real
    server round-trip each time, but the actual complaint ("really slow
    and laggy to plot provinces or even remove them from the plot") turned
    out to be dominated by the *client-side* rebuild, not the server: a
    live JS-timing profile (`console.time` bracketing the draw loop, not
    guessed) showed the per-toggle Aladin redraw alone costing 1-2 seconds
    and, worse, *growing* on each successive toggle (0.4s -> 0.6s -> 0.9s
    just to `removeLayer` the previous shapes) — repeatedly tearing down
    and rebuilding thousands of real `A.circle` footprints is a genuine,
    apparently cumulative, cost in Aladin Lite itself, not something more
    server-side caching (see _color_by_values's own cache, which fixed the
    *server*-side half of this same complaint one round earlier) could
    reach.

    This version's actual fix: colour-bucket boundaries never depended on
    which files were selected in the first place (`_l1_active_color_params`
    derives vmin/vmax from the *whole* stacked target list, verified
    directly in the code, not merely assumed) — so every file's own points
    can be coloured once, independently, and the payload below always
    includes *every* file (not just currently-selected ones). Showing/
    hiding a file is then a purely client-side `overlay.show()`/`.hide()`
    toggle on that file's own already-built, never-rebuilt overlay object
    (confirmed live that Aladin Lite's `graphicOverlay` genuinely exposes
    `show()`/`hide()`/`toggle()` methods, not assumed) — no server
    round-trip, no `removeLayer`/`A.circle`-rebuild churn, at all. Real
    angular size and colour-matches-the-main-plot are both unchanged from
    design (3); only *when*/*how often* the expensive part runs changed.

    One `A.graphicOverlay` per *file* (not per file-and-colour-bucket) —
    each point carries its own already-resolved colour string instead of
    being grouped into a shared-colour bucket, confirmed live (empirically,
    in a real browser, not assumed from the docs) that `A.circle(ra, dec,
    radius, {color: ...})` genuinely accepts a *per-shape* colour, a
    limitation that only ever applied to `A.source`/`A.catalog` markers
    (see `_bucket_by_color`'s own docstring), never to `A.circle`/
    `A.graphicOverlay` shapes. One overlay per file (not up to 16 files x
    13 buckets = 208 separate overlay objects, an earlier version of this
    exact redesign) is what keeps the very first "turn the overlay on"
    build fast too — each `A.graphicOverlay`/`addOverlay()` call has its
    own real per-object construction/registration cost in Aladin Lite,
    confirmed live: the 208-overlay version measured ~12s for that first
    build (up from a few seconds before this redesign), cut back down by
    this flattening."""
    if EXPLORER.kind != "l1" or not EXPLORER.contrib_exposures_on:
        return None
    result = l1_mod.STATE.contributing_fibre_positions()
    if not result or not result.get("files"):
        return None

    valid_files = [f for f in result["files"] if "error" not in f]
    if not valid_files:
        return None

    st = l1_mod.STATE
    target_values, vmin, vmax, colorscale = _l1_active_color_params(st)

    files_payload = []
    for f in valid_files:
        ra, dec = np.asarray(f["ra"]), np.asarray(f["dec"])
        nearest = np.asarray(f["nearest_idx"])
        finite = np.isfinite(ra) & np.isfinite(dec) & (nearest >= 0)
        ra_f, dec_f, nearest_f = ra[finite], dec[finite], nearest[finite]
        # Flattened to one point-list per *file*, each point carrying its
        # own already-resolved colour string, rather than one nested list
        # of points per (file, colour-bucket) pair — confirmed directly
        # (empirically, in a real browser, not assumed from the docs)
        # that `A.circle(ra, dec, radius, {color: ...})` genuinely accepts
        # a *per-shape* colour, not just the catalog-wide colour
        # `_bucket_by_color`'s own docstring says is the only *confirmed*
        # option for `A.source`/`A.catalog` markers — that limitation
        # never applied to `A.circle`/`A.graphicOverlay` shapes at all.
        # This lets the clientside drawing callback build exactly one
        # A.graphicOverlay per *file* (16 here) instead of one per
        # (file, bucket) pair (up to 16*13=208) — the same real angular-
        # size, colour-matches-the-main-plot points either way, just far
        # fewer overlay *objects* for Aladin to construct/register, which
        # is what made the very first "turn the overlay on" build slower
        # than intended in an earlier version of this per-file-overlay
        # redesign (live-profiled at ~12s, up from a few seconds) before
        # being cut back down by this flattening.
        points = []
        if len(ra_f):
            point_values = target_values[nearest_f]
            buckets = _bucket_by_color(nearest_f, ra_f, dec_f, point_values,
                                        vmin, vmax, colorscale, scale=st.color_scale)
            for bucket in buckets:
                color = bucket["color"]
                for p in bucket["points"]:
                    points.append({"ra": p["ra"], "dec": p["dec"], "color": color})
        files_payload.append({
            "file": f["file"],
            "label": " + ".join(
                [f["file"] + (f" ({f['camera']})" if f.get("camera") else "")]
                + [o["file"] + (f" ({o['camera']})" if o.get("camera") else "")
                   for o in f.get("other_files", [])]
            ),
            "points": points,
        })

    return {
        "files": files_payload,
        "radius_deg": _CONTRIB_MARKER_RADIUS_DEG,
        "arm_used": result["arm_used"],
        "n_arms_loaded": result["n_arms_loaded"],
    }


def _contrib_exposures_note_text(payload):
    """The always-visible note accompanying the overlay — explicit
    request: "if you have multiple arm data... you can only use one of
    them [for positions]... however, add a note saying that you are
    using [one of N] arms." Also doubles as the per-file legend — a real
    `dcc.Checklist` (explicit request: "I want to have a[n option] to
    select and deselect each contributed single file... a[checkbox] next
    to the name of each single one... so I can manage it by myself"),
    one entry per contributing exposure, its label combining every loaded
    arm's own name for that same exposure epoch (explicit request: "no
    need to load the fibre position for each arm of the single but not
    bad that you write the name of both single blue and red next to the
    [checkbox]" — see `AppState.contributing_fibre_positions`'s own
    `other_files` for how those names are paired up). No colour swatch or
    shape glyph in the label any more — colour is now data-driven (the
    same value the main catalog uses), not a per-file identifier, so a
    legend swatch showing a fixed per-file colour would just be wrong.

    Checking/unchecking an entry here is now a **pure client-side**
    show()/hide() toggle on that file's own already-built overlay objects
    (see _contrib_exposures_payload's own docstring for why this is safe
    — colour-bucket boundaries never actually depended on file selection
    in the first place) — no server round-trip at all, so this checklist
    is only ever rebuilt (and its own `value` reset to "all checked") on
    a genuine data change (fresh load, or the main on/off checkbox),
    never merely because the user ticked/unticked one file."""
    if not payload:
        return None
    n_files = len(payload["files"])
    arm_note = (
        f"Fibre positions resolved from {n_files} contributing exposure(s) via "
        f"\"{payload['arm_used']}\"'s own provenance"
        + (
            f" — only 1 of {payload['n_arms_loaded']} loaded arms was used for positions "
            f"(arms share identical fibre positions, so reading more would be redundant)."
            if payload["n_arms_loaded"] > 1 else "."
        )
    )
    all_files = [f["file"] for f in payload["files"]]
    legend = dcc.Checklist(
        id="contrib-legend-checklist",
        options=[{"label": f["label"], "value": f["file"]} for f in payload["files"]],
        value=all_files,  # always "everything checked" on a fresh build — see docstring above
        style={"display": "flex", "flexDirection": "column", "rowGap": "2px"},
        inputStyle={"marginRight": "5px"},
    )
    return html.Div([
        html.Div(arm_note, style={"fontStyle": "italic", "marginBottom": "3px"}),
        legend,
    ], style={"fontSize": "11px", "color": "var(--pyaps-ink-muted)", "marginTop": "4px"})


def _aladin_legend_info():
    """Colour-legend metadata for the bar under the Aladin panel. Kept as
    its own function rather than a return value of _bucket_by_color (which
    would mean threading it through both call sites) — but it deliberately
    re-derives the *exact same* vmin/vmax percentile fallback
    _bucket_by_color applies internally, so the legend can never show a
    range that disagrees with what's actually plotted. Returns None if
    nothing is loaded; otherwise either
    {"kind": "continuous", "colorscale", "vmin", "vmax", "label"} (IFU/L1)
    or {"kind": "categorical", "entries": [{"color", "label"}, ...]} (MOS)."""
    if EXPLORER.kind == "ifu":
        data = ifu_mod.STATE.data
        if data is None:
            return None
        values, _, cbar_label = ifu_mod._resolve_maptype(data, ifu_mod.STATE.current_maptype)
        if values is None:
            return None
        colorscale = _resolve_palette_override(ifu_mod.select_colorscale(ifu_mod.STATE.current_maptype))
        vmin, vmax = ifu_mod.STATE.color_vmin, ifu_mod.STATE.color_vmax
        if vmin is None or vmax is None:
            finite = np.asarray(values, dtype=np.float64)
            finite = finite[np.isfinite(finite)]
            lo, hi = np.percentile(finite, [1, 99]) if finite.size else (0.0, 1.0)
            vmin = lo if vmin is None else vmin
            vmax = hi if vmax is None else vmax
        return {"kind": "continuous", "colorscale": colorscale,
                "vmin": float(vmin), "vmax": float(vmax), "label": cbar_label or "Value"}

    if EXPLORER.kind == "l1":
        st = l1_mod.STATE
        if not st.loaded():
            return None
        label, colorscale, default_vmin = l1_mod.COLOR_BY_OPTIONS[st.color_by]
        values = l1_mod._color_by_values(st, st.color_by)
        # Same defaults as _aladin_catalog_points's L1 branch so the legend
        # agrees with the plotted colours.
        finite = values[np.isfinite(values)]
        vmin = st.color_vmin if st.color_vmin is not None else default_vmin
        vmax = st.color_vmax if st.color_vmax is not None else (float(np.nanmax(finite)) if finite.size else 1.0)
        return {"kind": "continuous", "colorscale": _resolve_palette_override(colorscale),
                "vmin": float(vmin), "vmax": float(vmax), "label": label}

    if EXPLORER.kind == "mos":
        data = mos_mod.STATE.data
        if data is None:
            return None
        color_by = mos_mod.STATE.color_by
        if color_by == "availability" or color_by not in mos_mod.COLOR_BY_OPTIONS:
            # Categorical (Gal/ExGal/Both/None) — exact swatches, no
            # continuous bar, matching _aladin_catalog_points's MOS branch.
            return {"kind": "categorical",
                    "entries": [{"color": color, "label": label}
                                for label, color in mos_mod._AVAIL_STYLE.values()]}

        _, column, colorscale, legend_label = mos_mod.COLOR_BY_OPTIONS[color_by]
        values = mos_mod._color_by_values(mos_mod.STATE, column)
        vmin, vmax = mos_mod.STATE.color_vmin, mos_mod.STATE.color_vmax
        if vmin is None or vmax is None:
            finite = values[np.isfinite(values)]
            lo, hi = np.percentile(finite, [1, 99]) if finite.size else (0.0, 1.0)
            vmin = lo if vmin is None else vmin
            vmax = hi if vmax is None else vmax
        return {"kind": "continuous", "colorscale": _resolve_palette_override(colorscale),
                "vmin": float(vmin), "vmax": float(vmax), "label": legend_label}

    return None


def _aladin_legend_children():
    """Builds the Dash children for the #aladin-legend Div: a horizontal
    CSS-gradient colour bar with min/max labels for continuous quantities
    (IFU/L1), or a row of coloured swatches with labels for MOS's
    categorical availability classes. Returns None (renders nothing) if
    nothing is loaded."""
    info = _aladin_legend_info()
    if info is None:
        return None

    # Font sizes throughout this legend were bumped from the original
    # 11px (categorical labels had no explicit size at all, inheriting
    # whatever the page default was) per explicit user request — the
    # colour bar and its range numbers are an important feature and
    # deserve to be more eye-catching, not blend in as fine print.
    if info["kind"] == "categorical":
        return [
            html.Div([
                html.Span(style={
                    "display": "inline-block", "width": "13px", "height": "13px",
                    "backgroundColor": e["color"], "borderRadius": "2px",
                    "marginRight": "6px", "verticalAlign": "middle",
                }),
                html.Span(e["label"], style={"verticalAlign": "middle", "fontSize": "14px",
                                              "fontWeight": "600"}),
            ], style={"display": "inline-block", "marginRight": "18px", "whiteSpace": "nowrap"})
            for e in info["entries"]
        ]

    import plotly.colors as pc
    n_stops = 32
    stop_colors = pc.sample_colorscale(info["colorscale"], [i / (n_stops - 1) for i in range(n_stops)])
    gradient = "linear-gradient(to right, " + ", ".join(stop_colors) + ")"
    return [
        html.Div(info["label"], style={"fontSize": "14px", "fontWeight": "700", "marginBottom": "3px"}),
        html.Div(style={
            "height": "16px", "borderRadius": "3px", "background": gradient,
            "border": "1px solid var(--pyaps-line)",
        }),
        html.Div([
            html.Span(f"{info['vmin']:.4g}", style={"float": "left"}),
            html.Span(f"{info['vmax']:.4g}", style={"float": "right"}),
        ], style={"fontSize": "14px", "fontWeight": "700", "marginTop": "3px", "overflow": "hidden"}),
    ]


def _aladin_color_range_control():
    """The Colour range Min/Max/Reset/Scale/Palette control, centred
    directly under the Aladin legend bar it actually controls — moved
    here (out of the settings panel up near the top of the page, where it
    had no visual connection to the legend at all) per explicit user
    request: "put the colour range option below the colorbar... so it's
    made clear it's about this." None (renders nothing at all) only when
    nothing's loaded — the Min/Max/Scale row itself is separately omitted
    for MOS while it's coloured by the fixed categorical "availability"
    swatches (nothing continuous to set a range on), but the "Transparency
    by signal" checkbox below still renders even then, since it's a
    3D-flux-cube-only setting, independent of whatever the 2D catalog's
    own colour-by mode happens to be."""
    if not EXPLORER.loaded():
        return None
    state = _color_range_state()
    row_children = []
    if state is not None:
        row_children += [
            html.Label("Colour range:", style={"fontSize": "13px", "fontWeight": "600", "marginRight": "6px"}),
            # debounce=True: without it, dcc.Input fires its value on every
            # keystroke, and since that value change bumps aladin-catalog-data
            # (which this control feeds directly, not dataset-version — no
            # need to rebuild the whole shell just to recolour the catalog),
            # typing e.g. "200" would otherwise recompute the catalog after
            # every single digit.
            dcc.Input(id="color-vmin", type="number", value=state.color_vmin, debounce=True,
                      placeholder="min", style={"width": "70px"}),
            dcc.Input(id="color-vmax", type="number", value=state.color_vmax, debounce=True,
                      placeholder="max", style={"width": "70px", "marginLeft": "4px"}),
            html.Button("Reset", id="color-range-reset", n_clicks=0, style={"marginLeft": "4px"}),
            html.Label("Scale:", style={"fontSize": "13px", "fontWeight": "600",
                                         "marginLeft": "14px", "marginRight": "6px"}),
            dcc.Dropdown(
                id="color-scale-dd",
                options=[{"label": label, "value": key} for key, label in COLOR_SCALE_OPTIONS.items()],
                value=state.color_scale, clearable=False, searchable=False,
                style={"width": "120px", "display": "inline-block", "verticalAlign": "middle"},
            ),
            html.Label("Palette:", style={"fontSize": "13px", "fontWeight": "600",
                                           "marginLeft": "14px", "marginRight": "6px"}),
            # Explicit request: "is it possible to instead of using the
            # current color map, we use something that is black for low
            # signal and white for high signal... Can I have it in all maps
            # either 2D or 3D so we can switch between this blue to red to
            # black to white?" One shared EXPLORER-level setting (not
            # per-kind — see _resolve_palette_override), so it's the same
            # dropdown/value regardless of which kind is loaded.
            dcc.Dropdown(
                id="color-palette-dd",
                options=[{"label": label, "value": key} for key, label in PALETTE_OPTIONS.items()],
                value=EXPLORER.color_palette, clearable=False, searchable=False,
                style={"width": "170px", "display": "inline-block", "verticalAlign": "middle"},
            ),
        ]
    return html.Div([
        html.Div(row_children, style={"textAlign": "center", "display": "flex",
                                       "alignItems": "center", "justifyContent": "center"}) if row_children else None,
        # "Transparency by signal (low signal fades)" — explicit report:
        # "when I look at the cross section of 3d data cubes, I do not
        # see what I usually see in 2d maps... probably due to
        # transparency issue." Moved here this round, next to Scale/
        # Palette, per explicit follow-up: "Add the transparency by
        # signal o[p]tion next to the col[o]r range setting below the
        # col[o]r bar." Only meaningful in 3D mode (see flux_cube_
        # figure's own `transparent` docstring) but rendered regardless
        # of map-mode — same "always present, cheap to render" approach
        # every other 3D-only control here already uses (matching how
        # Scale/Palette are shown even while the 2D panel is the one
        # currently visible). `value=` reflects `EXPLORER.
        # cube_transparency_on` directly (the same "sticks across a
        # reload" persisted-preference convention `aladin-true-size`/
        # `color-palette-dd` already use), so reopening/reloading shows
        # whatever was last chosen rather than always resetting to off.
        html.Div(
            dcc.Checklist(
                id="cube-transparency-toggle",
                options=[{"label": " Transparency by signal (low signal fades)", "value": "on"}],
                value=["on"] if EXPLORER.cube_transparency_on else [],
                style={"fontSize": "12px"},
            ),
            style={"textAlign": "center", "marginTop": "6px"},
        ),
    ], style={"marginTop": "6px"})


def _central_item(kind):
    """The bin/target/fibre spatially closest to the field's own centre
    (mean position across every point with a finite coordinate) — used as
    the default selection on load instead of an arbitrary "first row"
    pick (Voronoi-binning/target-list order has no relation to spatial
    position, so "row 0" routinely landed on a corner/edge point) per
    explicit user request: "is it possible to always select the central
    bin or spaxel... right now it usually shows a point at the lower part
    and not good." Falls back to row 0 only if every coordinate is
    somehow non-finite (never raises)."""
    if kind == "ifu":
        table = ifu_mod.STATE.data["table"]
        # X/Y are already field-centre-relative arcsec offsets (see
        # _aladin_catalog_points's own comment on the same convention) —
        # no separate "compute the centre" step needed, the origin *is*
        # the centre.
        x = np.asarray(table["X"], dtype=np.float64)
        y = np.asarray(table["Y"], dtype=np.float64)
        finite = np.isfinite(x) & np.isfinite(y)
        if not finite.any():
            return int(table["BIN_ID"][0])
        d2 = np.where(finite, x ** 2 + y ** 2, np.inf)
        return int(table["BIN_ID"][int(np.argmin(d2))])

    if kind in ("l1", "mos"):
        if kind == "l1":
            st = l1_mod.STATE
            ra, dec, ids = st.coord_arr[:, 0], st.coord_arr[:, 1], st.aps_id_arr
        else:
            data = mos_mod.STATE.data
            ra = np.asarray(data["ra"], dtype=np.float64)
            dec = np.asarray(data["dec"], dtype=np.float64)
            ids = np.asarray(data["class_table"]["APS_ID"])
        finite = np.isfinite(ra) & np.isfinite(dec)
        if not finite.any():
            return int(ids[0])
        ra0 = float(np.nanmean(ra[finite]))
        dec0 = float(np.nanmean(dec[finite]))
        cos_dec = np.cos(np.radians(dec0)) or 1.0
        d2 = np.where(finite, ((ra - ra0) * cos_dec) ** 2 + (dec - dec0) ** 2, np.inf)
        return int(ids[int(np.argmin(d2))])

    return None


def _initial_selected_item():
    if EXPLORER.kind in ("ifu", "l1", "mos"):
        return _central_item(EXPLORER.kind)
    return None


def _aladin_extra_fields(selected_item):
    """Kind-dispatched extra identification fields beyond bare ID/RA/Dec —
    CNAME/target survey/class where they actually exist for the current
    kind, added per explicit user request ("if possible add CNAME or
    CCNAME and targsrvy and other info that might be useful"). Returns a
    list of (label, value) pairs, skipping any field that isn't actually
    available rather than showing a misleading blank."""
    fields = []
    if EXPLORER.kind == "l1":
        st = l1_mod.STATE
        idx = st.targs_apstoid.get(selected_item)
        if idx is not None:
            t = st.targs[idx]
            if getattr(t, "cname", None):
                fields.append(("CNAME", str(t.cname).strip()))
            if getattr(t, "targsrvy", None):
                fields.append(("Survey", str(t.targsrvy).strip()))
            if getattr(t, "targclass", None):
                fields.append(("Class", str(t.targclass).strip()))
    elif EXPLORER.kind == "mos":
        data = mos_mod.STATE.data
        if data is not None:
            ct = data["class_table"]
            match = np.flatnonzero(np.asarray(ct["APS_ID"]) == selected_item)
            if match.size:
                i = int(match[0])

                def _scalarize(val):
                    # TARGSRVY/TARGCLASS (like several CLASS_TABLE columns)
                    # are stored per-Redrock-rank, i.e. one *array* per
                    # target (all ranks holding the identical value in
                    # practice) — confirmed on real data: `str(array)`
                    # alone would show "[b'X' b'X' b'X']", not "X". Take
                    # rank 0, matching the same convention
                    # _class_table_rows already uses for these columns.
                    arr = np.asarray(val)
                    v = arr[0] if arr.ndim >= 1 else arr
                    return v.decode(errors="replace") if isinstance(v, (bytes, np.bytes_)) else str(v)

                if "CNAME" in ct.colnames:
                    fields.append(("CNAME", _scalarize(ct["CNAME"][i]).strip()))
                if "TARGSRVY" in ct.colnames:
                    fields.append(("Survey", _scalarize(ct["TARGSRVY"][i]).strip()))
                if "TARGCLASS" in ct.colnames:
                    fields.append(("Class", _scalarize(ct["TARGCLASS"][i]).strip()))
    elif EXPLORER.kind == "ifu":
        cname = (ifu_mod.STATE.data or {}).get("field_cname")
        if cname:
            fields.append(("Field CNAME", str(cname).strip()))
    return fields


def _aladin_info_text(selected_item):
    """"What am I looking at" readout for the small box under the Aladin
    panel — the coordinate/ID info the Plotly map's hover tooltip used to
    show, now that hover is gone and the map itself is gone (Aladin's own
    `objectClicked` doesn't carry a tooltip either). BIN_ID for IFU,
    APS_ID for L1/MOS — whichever `selected_item` actually is for the
    currently-loaded kind — plus CNAME/survey/class where available (see
    _aladin_extra_fields). Rendered bold by the info box's own style
    (aladin-info-box), not here, so a changed selection is visually
    dominant/eye-catching at a glance, per explicit user request."""
    if not EXPLORER.loaded() or selected_item is None:
        return "No selection."
    target = _aladin_target(selected_item)
    if target is None:
        return "No selection."
    id_label = "BIN_ID" if EXPLORER.kind == "ifu" else "APS_ID"
    parts = [f"{id_label}: {selected_item}", f"RA: {target['ra']:.5f}°", f"Dec: {target['dec']:.5f}°"]
    parts += [f"{label}: {value}" for label, value in _aladin_extra_fields(selected_item)]
    return "   ".join(parts)


# Deliberately its own config, not the shared aps_style.GRAPH_CONFIG every
# other plot in the app uses — explicit user request: "probably better if
# I cannot zoom in zoom out... so a fixed zoom is fine" (the strip's own
# default view is already a fixed-width window around the current
# selection, see slit_explorer.WINDOW_HALF_WIDTH; zoom would just let that
# width drift). scrollZoom off + the box/lasso-zoom and autoscale/reset
# modebar buttons removed; panning (drag, or the still-present "pan"
# button) stays on so the rest of the slit remains reachable.
_SLIT_EXPLORER_GRAPH_CONFIG = {
    "scrollZoom": False,
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["zoomIn2d", "zoomOut2d", "zoom2d", "autoScale2d",
                                "resetScale2d", "select2d", "lasso2d"],
    "doubleClick": False,
}

# Also its own config, not the shared aps_style.GRAPH_CONFIG — that shared
# one sets `doubleClick: "reset+autosize"` deliberately, a genuinely wanted
# feature on every 2D spectra/fit plot ("Double-clicking a plot resets it
# to the full/auto view"). For a 3D scene specifically, Plotly's
# documented `doubleClick` config maps double-click to resetting
# `scene.camera` back to the figure's own initial value — explicit report:
# "in all coordinate in 2d and 3d plots, can you disable the click to
# centre behaviour... in 3d when I double click on a point, it make[s] it
# cent[]re of the field which is annoying." `False` disables it outright,
# the same already-established pattern _SLIT_EXPLORER_GRAPH_CONFIG uses
# for the identical reason on a different plot.
#
# `modeBarButtonsToRemove` — explicit report: "Even reset camera to last
# save also always reset to the view along the Z axis which is
# wrong... Make sure all default plotly 3d options are not distracting
# you and not making problems." Confirmed live (Playwright, real camera
# rotation then click): Plotly's own "Reset camera to last save" button
# (`resetCameraLastSave3d`) does not track this app's live/rotated camera
# at all — it always snaps back to whatever `layout.scene.camera` the
# figure was originally built with (this app's own `flux-cube-camera
# -store` mechanism, which *does* track the live camera across rebuilds,
# is a completely separate system Plotly's own "last save" bookkeeping
# has no knowledge of). There's no supported hook to make Plotly's
# built-in "last save" state track a later client-side relayout, so
# removed rather than "fixed" — a button whose name promises one thing
# and reliably does another is worse than no button. `hoverClosest3d`
# ("Toggle show closest data on hover") removed too, per the same
# explicit request — it ships on by default and adds a second, redundant
# hover mode on top of Plotly's already-on-by-default plain hover, purely
# extra modebar clutter for this app's own click-to-select-driven
# workflow. `resetCameraDefault3d` ("Reset camera to default") is kept —
# unlike "last save" it does exactly what its label says (goes to the
# figure's genuine build-time default), so it's not misleading.
_FLUX_CUBE_GRAPH_CONFIG = {
    **aps_style.GRAPH_CONFIG,
    "doubleClick": False,
    "modeBarButtonsToRemove": ["resetCameraLastSave3d", "hoverClosest3d"],
}


def _slit_explorer_container(selected_item):
    """"Slit Explorer" — a very-narrow-height strip just below the Aladin
    panel showing every fibre's position on the physical spectrograph
    slit (NSPEC), not its position on the sky (APS_ID) — two independent
    numbering schemes (confirmed on real data: uncorrelated permutations
    of the same 1..N). Explicit user request/rationale: a fibre can sit on
    the slit right next to an unusually bright one and pick up scattered-
    light contamination that's completely invisible from its APS_ID/sky
    position alone, so seeing slit-adjacency at a glance matters. Only for
    L1, and only when the loaded file is genuinely fibre-level (single-
    exposure MOS/MOSLIFU/MOSMIFU, not a stacked/co-added cube — see
    l1_mod.AppState.is_fibre_level's own docstring for exactly why a
    stacked cube has no real per-fibre NSPEC to show)."""
    if EXPLORER.kind != "l1":
        return None
    if not l1_mod.STATE.is_fibre_level():
        return html.Div(
            "Slit Explorer: not available for this dataset — only single-exposure, "
            "fibre-level L1 data (not a stacked/co-added cube) has a real per-fibre "
            "slit position to show.",
            style={"fontSize": "11px", "color": "var(--pyaps-ink-faint)", "fontStyle": "italic",
                   "padding": "4px 0", "marginTop": "4px"},
        )
    fig = l1_mod._build_slit_explorer_figure(selected_item)
    if fig is None:
        return None
    info_text = l1_mod.slit_info_text(selected_item)
    return html.Div([
        # "Slit Explorer" itself bold (like the Log panel's own H4 title —
        # explicit request), the rest of the sentence (what it actually
        # shows/how to use it) left at normal weight — explicit
        # counter-request: "no need to make the description ... bold".
        html.Div([
            html.Span("Slit Explorer", style={"fontWeight": "700"}),
            html.Span(" — fibre position on the spectrograph slit (NSPEC); "
                      "click a point here to select that fibre, or drag to scroll along the slit",
                      style={"fontWeight": "400"}),
        ], style={"fontSize": "11px", "color": "var(--pyaps-ink-muted)", "marginBottom": "2px"}),
        dcc.Graph(id="slit-explorer-graph", figure=fig, config=_SLIT_EXPLORER_GRAPH_CONFIG,
                  style={"height": "130px"}),
        # Always-visible readout for the current selection — explicit user
        # request: the plot's own hover tooltip only shows info for
        # whichever point the mouse happens to be over (and vanishes the
        # instant it isn't), which isn't the same thing as "tell me about
        # the fibre I actually selected." Same bold/eye-catching styling
        # as aladin-info-box, for the same reason and right next to the
        # marker/arrow it describes.
        html.Div(
            id="slit-explorer-info",
            children=info_text or "No selection.",
            style={
                "marginTop": "4px", "padding": "6px 10px", "backgroundColor": "var(--pyaps-paper-sunken)",
                "border": "1px solid var(--pyaps-line-strong)", "borderRadius": "4px",
                "fontSize": "12px", "fontFamily": "monospace", "fontWeight": "700",
            },
        ),
    ], style={"marginTop": "6px"})


# Logos, base64-encoded once at import time (not per-request) — WEAVE's own
# mark (weave_logo.png, already alpha-transparent) on the left, camCEAD's
# on the right. The first two camCEAD source files tried here both
# needed one-time de-matte pre-processing (their `*_transparent.png`
# outputs, still checked into doc/ even though no longer referenced
# below, since they document the two formulas used and cost nothing to
# keep): `camcead_logo.png` had a fully opaque *white* background (every
# corner pixel (255,255,255,255)), de-matted via alpha = 255 -
# min(R,G,B); `camcead_logo_dark.png` was the mirror image, a flat white
# mark on a near-black background, de-matted via alpha = (pixel-bg)/
# (white-bg). The current `camCEAD_main_logo.png` (swapped in per
# explicit request) needs none of that — confirmed via direct pixel
# inspection to already be genuine RGBA with real per-pixel alpha (a
# light-grey, not pure-white, fill, but visually indistinguishable from
# white once composited over this banner's dark blue — checked by
# rendering the composite directly, not assumed) — so it's used as-is.
#
# The two files actually used live in py/PyAPS/data/ (a real package
# subdirectory, listed in pyproject.toml's package-data), NOT doc/ at
# the repo root — this was the real cause of "I do not see the camcead
# logo when loading aps_explorer from the server" (nothing to do with
# screen resolution): the old path here was
# Path(__file__).resolve().parent.parent.parent / "doc", which only
# ever resolved correctly for a source checkout run in place. The
# Docker image (see Dockerfile: only `py/` is COPYed, then pip-installed
# into site-packages) never had a doc/ directory at all, and even a
# plain `pip install PyAPS` puts aps_explorer.py under .../site-packages/
# PyAPS/, three parents up from which is nowhere near this repo's doc/
# either — confirmed live in the running server container: both
# _WEAVE_LOGO_URI and _CAMCEAD_LOGO_URI were None, not just camCEAD's
# (the user likely just didn't notice/mention WEAVE's own absence).
# A path relative to this file's own package directory resolves
# correctly either way, exactly the same reasoning configs/templates
# data already relies on package-data for.
_LOGO_DIR = Path(__file__).resolve().parent / "data"


def _logo_data_uri(filename):
    try:
        data = (_LOGO_DIR / filename).read_bytes()
    except Exception:
        return None
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


_WEAVE_LOGO_URI = _logo_data_uri("weave_logo.png")
_CAMCEAD_LOGO_URI = _logo_data_uri("camCEAD_main_logo.png")

# Both the banner and the toolbar beneath it are `position: sticky` (not
# `fixed`) — the earlier design used a `fixed` floating "☰ Load dataset"
# button with a hand-tuned pixel offset meant to sit just below the
# banner, which per direct user report could still end up overlapping it
# ("it can even cover the banner"). `sticky` sidesteps the whole class of
# bug: the element is a completely normal, in-flow flexbox child (so
# nothing can ever overlap it) right up until the page scrolls past its
# position, at which point it sticks to the top of the viewport exactly
# like a fixed element would — same "always reachable" behaviour, none of
# the manual-offset fragility. Both sit below load-panel's own z-index
# (1000) so the sidebar, when open, still covers them the same way it
# covers everything else.
_BANNER_Z = 501
_TOOLBAR_Z = 500
# Shared between _banner()'s own height and _toolbar()'s "top" (the
# offset at which it starts sticking, i.e. right where the banner ends)
# so the two can't drift apart if one is ever changed without the other.
# 130px (was 64px) — grown to fit the 2.5x-larger logos below (explicit
# request), keeping roughly the same ~20px breathing room above/below the
# tallest element (110px WEAVE logo) that the original 64px/44px pairing
# had (64-44=20).
_BANNER_HEIGHT_PX = 130


_THEME_CHOICES = [("light", "Light"), ("auto", "Auto"), ("dark", "Dark")]


def _theme_toggle():
    """Light/Auto/Dark switch. Pure client state (see
    assets/explorer_ui.js) — no server callback, no dcc.Store: the
    app's own Python code never needs to know which theme is active,
    only the browser does (it's read/written straight to
    localStorage). `data-theme-choice` is what explorer_ui.js's
    delegated click listener keys off; the `.theme-toggle-btn`/
    `.active` classes are what assets/style.css uses for the states
    Dash's inline style={} can't express (:hover, and "whichever one
    is currently picked," which is only knowable client-side).

    Lives in `_toolbar()` now, not `_banner()` — explicit request: "Cna
    you put the theme selection in the next bar where you said loggend
    in as ... (WEAVEOR or so)...I mean it is not nice that it is in the
    banner." Styling switched from literal white-on-transparent (which
    only ever worked because the banner is *always* a dark blue
    gradient in both app themes, see `_banner()`'s own comment on that)
    to the normal `--pyaps-*` tokens every other toolbar control already
    uses, since the toolbar's own background genuinely does flip with
    the theme this control switches."""
    return html.Div(
        [
            html.Button(
                label, className="theme-toggle-btn", title=label,
                **{"data-theme-choice": value},
                style={
                    "border": "none", "background": "transparent", "color": "var(--pyaps-ink-muted)",
                    "cursor": "pointer", "fontSize": "11px", "fontWeight": "600",
                    "padding": "5px 10px", "borderRadius": "16px", "lineHeight": "1",
                },
            )
            for value, label in _THEME_CHOICES
        ],
        id="theme-toggle",
        style={
            "display": "flex", "border": "1px solid var(--pyaps-line-strong)",
            "borderRadius": "20px", "padding": "2px", "background": "var(--pyaps-paper-raised)",
            "flexShrink": "0",
        },
    )


def _banner():
    """Top branding strip: WEAVE logo + title/version grouped together on
    the left, camCEAD logo on the right, per explicit request (title used
    to be centred across the whole banner — moved to sit directly beside
    the WEAVE logo instead: "left align next to the WEAVE logo"). The
    version shown is PyAPS's own running package version
    (`PyAPS.__version__`, sourced from version.txt — the same value every
    `APSVERS = PyAPS.__version__` in the codebase uses), not a loaded
    file's processing version."""
    # Logo sizes (110px/95px) are 2.5x their original values, kept as-is
    # on the latest logo-file swap per explicit request ("keep the size as
    # the previous one"). Title/version text is 30% *smaller* than an
    # earlier round's 2x-enlarged size (40px/24px), i.e. 0.7x those:
    # 28px/17px — explicit follow-up request ("now it is too big").
    title_block = html.Div([
        html.Div("WEAVE Data Explorer", style={"fontSize": "28px", "fontWeight": "700", "lineHeight": "1.25"}),
        html.Div(f"PyAPS v{PyAPS.__version__}", style={"fontSize": "17px", "opacity": "0.85",
                                                         "letterSpacing": "0.03em"}),
    ], style={"textAlign": "left", "marginLeft": "18px"})
    left = html.Div(
        [
            html.Img(src=_WEAVE_LOGO_URI, style={"height": "110px", "display": "block"})
            if _WEAVE_LOGO_URI else None,
            title_block,
        ],
        style={"display": "flex", "alignItems": "center", "flexShrink": "0"},
    )
    right = html.Div(
        [
            html.Img(src=_CAMCEAD_LOGO_URI, style={"height": "95px", "display": "block"})
            if _CAMCEAD_LOGO_URI else None,
        ],
        style={"display": "flex", "alignItems": "center", "gap": "20px",
               "justifyContent": "flex-end", "flexShrink": "0"},
    )
    # A plain flexible spacer (no content of its own) between the left
    # group and the camCEAD logo — replaces the old `center` div's job of
    # soaking up the banner's leftover width, now that the title text has
    # moved out of the middle and into the left group itself.
    spacer = html.Div(style={"flex": "1 1 auto"})
    return html.Div([left, spacer, right], id="app-banner", style={
        "position": "sticky", "top": "0", "zIndex": _BANNER_Z,
        "height": f"{_BANNER_HEIGHT_PX}px", "width": "100%", "boxSizing": "border-box",
        # A subtle top-to-bottom gradient within the app's existing accent
        # blue — varying top-to-bottom rather than left-to-right means both
        # logos, sitting at the same vertical band, always see identical
        # shading/contrast regardless of which side they're on (confirmed
        # against both logos directly before picking this: the mid-tone
        # blue is the one background both a colourful mark with a white
        # halo (WEAVE) and a dark charcoal mark (camCEAD) read clearly
        # against — a darker navy muddies camCEAD's contrast, a lighter one
        # washes out on WEAVE). var(--pyaps-banner-grad) (py/PyAPS/assets/
        # style.css) carries the exact same two stops in light mode
        # (#1f4864 -> aps_style.TABLE_HEADER_COLOR's original #2c5f8a) and
        # a darker pair in dark mode — the banner is deliberately always a
        # dark gradient in *both* app themes (this is why "color": "white"
        # below is a plain literal, not a token: white reads correctly on
        # both variants of this gradient, unlike the page's ink/paper
        # tokens which do need to flip).
        "backgroundImage": "var(--pyaps-banner-grad)",
        "color": "white", "display": "flex", "alignItems": "center", "padding": "0 20px",
        "boxShadow": "var(--pyaps-shadow-banner)", "flexShrink": "0",
    })


def _identity_text(bundle=None):
    """"Logged in as ..." text for the toolbar indicator, or "" if this
    session has no verified weaveOR identity. A plain function (not
    baked directly into _toolbar()) specifically so handle_url_handoff
    can also produce this same text as an Output — _toolbar() only runs
    once, server-side, at the start of a page load, *before* a token in
    that same page's own URL has had any chance to be verified (that
    happens via a later, separate client-driven callback round-trip,
    not a fresh page render) — without a dedicated Output writing into
    "toolbar-identity" directly after a successful handoff, this text
    would only ever reflect what was true one full page load ago."""
    auth = _auth.current_auth(bundle if bundle is not None else _sess.current_bundle())
    return f"Logged in as {auth['user']} (via WeaveOR)" if auth else ""


def _toolbar():
    """Slim sticky strip directly under the banner — the "☰ Load dataset"
    button's real home now (see _banner()'s docstring for why `sticky`,
    not `fixed`). A normal flexbox child, so it can never overlap the
    banner above it or main-panel below it; kept as its own row rather
    than folded into the banner itself so the branding row stays exactly
    the clean 3-part WEAVE/title/camCEAD layout that was asked for, with
    room for other page-global controls here later if needed — the
    "Logged in as" indicator was the first one to use that room, the
    theme toggle (moved here from _banner(), see _theme_toggle()'s own
    docstring) is the second."""
    identity = html.Span(
        _identity_text(), id="toolbar-identity",
        style={"fontSize": "12px", "color": "var(--pyaps-ink-muted)"},
    )
    right = html.Div(
        [identity, _theme_toggle()],
        style={"display": "flex", "alignItems": "center", "gap": "16px"},
    )
    return html.Div(
        [
            html.Button("☰  Dataset & Settings", id="open-load-panel-btn", style={
                "fontSize": "13px", "fontWeight": "600", "padding": "7px 16px",
                "backgroundColor": "var(--pyaps-paper-raised)", "color": aps_style.TABLE_HEADER_COLOR,
                "border": f"1.5px solid {aps_style.TABLE_HEADER_COLOR}", "borderRadius": "20px",
                "cursor": "pointer", "boxShadow": "var(--pyaps-shadow-btn)",
            }),
            right,
        ],
        id="app-toolbar", style={
            "position": "sticky", "top": f"{_BANNER_HEIGHT_PX}px", "zIndex": _TOOLBAR_Z,
            "width": "100%", "boxSizing": "border-box", "backgroundColor": "var(--pyaps-paper-sunken)",
            "borderBottom": "1px solid var(--pyaps-line)", "padding": "8px 20px", "flexShrink": "0",
            "display": "flex", "justifyContent": "space-between", "alignItems": "center",
        },
    )


def _log_panel():
    return html.Div([
        html.Div([
            html.H4("Log", style={"margin": "4px 0"}),
            html.Button("Clear log", id="log-clear", n_clicks=0, style={"fontSize": "11px"}),
        ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"}),
        html.Pre(id="log-panel", children="\n".join(LOG.snapshot()), style={
            "height": "220px", "overflowY": "scroll", "backgroundColor": "#1e1e1e",
            "color": "#d4d4d4", "padding": "10px", "fontSize": "11px",
            "fontFamily": "monospace", "whiteSpace": "pre-wrap", "margin": 0,
            "borderRadius": "4px",
        }),
    ], style={
        "marginTop": "20px", "paddingTop": "10px", "borderTop": "2px solid var(--pyaps-line-strong)",
        # Higher than #global-load-overlay's own z-index (9500, see
        # index_string) so the Log panel stays visible and usable
        # through the white overlay while a dataset is loading —
        # explicit request ("except the log section where I can follow
        # how the loading is going"). position:relative is what makes
        # zIndex apply at all on a normal-flow element like this.
        # The console itself (html.Pre#log-panel above) deliberately
        # keeps its literal #1e1e1e/#d4d4d4 dark-terminal colors in
        # *both* app themes, not tokenized — matches the approved
        # mockup's own console styling and reads correctly as "a log
        # console" regardless of the surrounding page theme.
        "position": "relative", "zIndex": 9600, "backgroundColor": "var(--pyaps-paper)",
    })


def _incoming_url_wants_different_dataset(bundle):
    """True if the request that's about to render `serve_layout()` carries
    a token/deep-link URL naming a *different* dataset than whatever this
    session already has loaded — read-only, no side effects (never
    verifies/applies/records anything; `handle_url_handoff` still does
    all of that moments later, exactly as before this existed).

    Explicit report: "when I use aps_explorer to explore an L1 or L2
    file and then close it and select another file(s) from WEAVEOR to
    display, it still load[s] the previous one on startup and after a
    few seconds it loads the new one... which is really confusing." Root
    cause: `serve_layout()` doesn't run as part of handling the browser's
    real page-navigation request at all in this Dash version (confirmed
    directly against this app's real Flask url_map) — dash-renderer's own
    JS fetches the actual layout afterwards, via a separate `/_dash
    -layout` XHR that carries no query string of its own, only once page
    bootstrap finishes. So by the time `serve_layout()` actually runs,
    reading the *live* request's own query string (tried first, then
    found to always read empty here) can't tell it a new dataset was even
    requested — and reading `EXPLORER.loaded()`/the coordinates-plots
    -tables built from it unconditionally means, whenever WeaveOR reuses
    the same browser tab for a new "Explore" click (same session cookie,
    same `SessionBundle`, same `EXPLORER`), that render baked the
    *previous* dataset into the very first thing the browser painted, so
    the user saw the old dataset "flash" for however long the real
    reload then took. Fixed by capturing the query string one request
    earlier instead — `_bind_session`'s own `before_request` hook *does*
    see the real navigation request, and stashes it on the bundle
    (`_pending_url_search`) for this function to read back moments later.

    Used by `serve_layout()` to render the "nothing loaded yet"/loading
    skeleton immediately instead, for this one render, whenever the
    incoming URL itself already promises a genuinely different dataset
    is on the way — the actual load (and the real, correct content
    replacing this skeleton) still happens exactly as before, just
    without a misleading stale frame in between.

    Deliberately fails *safe* toward the old behaviour (returns False,
    i.e. "show whatever's currently loaded") on any error decoding the
    URL — this only ever suppresses one frame's worth of stale content,
    never the actual load, so there's nothing to lose by being
    conservative here."""
    try:
        search = bundle.slots.get("_pending_url_search") or ""
        if not search:
            return False
        params = parse_qs(search)
        token = (params.get("token") or [None])[0]
        current = _auth.current_auth(bundle)
        current_identity = current["authorized_dataset"] if current else None
        if token:
            claims = _auth.verify_token(token)
            requested = _auth.dataset_identity(
                claims["kind"],
                **{k: v for k, v in claims.items() if k not in ("user", "allowed_surveys", "kind")},
            )
            return requested != current_identity
        kind = (params.get("kind") or [None])[0]
        if kind == "l2":
            outpath = (params.get("outpath") or [None])[0]
            headname = (params.get("headname") or [None])[0]
            if not outpath or not headname:
                return False
            requested = _auth.dataset_identity("l2", outpath=outpath, headname=headname)
            return requested != current_identity
        if kind == "l1":
            infiles = params.get("infiles")
            if not infiles:
                return False
            requested = _auth.dataset_identity("l1", infiles=infiles)
            return requested != current_identity
    except Exception:
        pass
    return False


def serve_layout():
    bundle = _sess.current_bundle() if _sess.MULTI_SESSION else None
    loaded = EXPLORER.loaded() and not (
        bundle is not None and _incoming_url_wants_different_dataset(bundle)
    )
    initial_item = _initial_selected_item() if loaded else None
    tabs = _tabs_for_current(initial_item) if loaded else []
    return html.Div([
        _banner(),
        _toolbar(),
        # Dimming scrim behind the settings drawer — see
        # _load_panel_backdrop_style's own docstring for why every
        # update after this initial render comes from a
        # clientside_callback rather than a Python helper call. Matches
        # load-panel's own new closed-by-default initial style just
        # below (a visible backdrop behind an already-closed drawer
        # would be a dim-screen-with-nothing-open glitch on first paint).
        html.Div(id="load-panel-backdrop", style=_load_panel_backdrop_style(False)),
        # See its own CSS (in index_string) and the show/hide
        # clientside callbacks near handle_l1_load/handle_l2_load for
        # the full design — no children here on purpose, the spinner
        # is pure CSS.
        html.Div(id="global-load-overlay"),
        html.Div(id="global-load-overlay-show-l2-dummy", style={"display": "none"}),
        html.Div(id="global-load-overlay-show-l1-dummy", style={"display": "none"}),
        html.Div(id="global-load-overlay-hide-dummy", style={"display": "none"}),
        # Read (and, once a token handoff is consumed, cleared — see
        # handle_url_handoff) for the weaveOR handoff / plain deep-link
        # mechanism — see aps_explorer_auth.py's own module docstring.
        # refresh=False: this app has no client-side routing, so a
        # programmatic write to "search" must not trigger a real page
        # navigation (which would re-run this whole layout function and
        # discard whatever was just loaded).
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="selected-item", data=initial_item),
        dcc.Store(id="dataset-version", data=1 if loaded else 0),
        # Bumped by any Colour range/Scale change (on_color_range_typed/
        # _reset/on_color_scale_change below) — a narrower-scoped sibling
        # to dataset-version, deliberately always present in the base
        # layout (unlike color-scale-dd/color-vmin/color-vmax themselves,
        # which only exist once a dataset is loaded) so update_map_mode
        # can take it as a direct Input without risking the documented
        # "Input on a component that doesn't exist yet silently kills the
        # *whole* callback" gotcha (see update_map_mode's own docstring
        # and this project's own established "Dash gotcha #1").
        dcc.Store(id="color-range-version", data=0),
        # The 3D flux cube's own current camera (eye/up/center), written
        # directly by a clientside `plotly_relayout` listener on every
        # rotate/zoom — see that callback's own comment for why this
        # exists instead of `update_map_mode` reading `flux-cube-graph`'s
        # own `figure` prop directly. Always present in the base layout
        # (not conditional on 3D mode/a dataset being loaded) for the
        # same "avoid the documented missing-component gotcha" reason
        # `color-range-version` above already is.
        dcc.Store(id="flux-cube-camera-store", data=None),
        # Bumped once by update_map_mode every time it *genuinely* rebuilds
        # flux-cube-graph's figure (never on its own early no_update
        # returns) — the wave-range clientside filter callback's own
        # signal for "a real new mesh just landed, refresh the pristine
        # cache", replacing an earlier attempt that watched
        # flux-cube-graph's own "figure" prop directly. Confirmed live
        # that watching the figure prop was itself unreliable: dcc.Graph
        # mirrors *any* Plotly.js relayout back into that same prop,
        # including a pure camera rotation/pan/zoom that never touches a
        # single vertex — every ordinary orbit of the cube was silently
        # re-caching whatever was *currently displayed* (already NaN
        # -filtered down to the live Start/End window) as if it were the
        # dataset's full pristine coverage, permanently discarding
        # everything outside that window until the next genuine server
        # rebuild. Exactly the "after a bit of playing with the
        # wavelength bin and range it lost the rest of data" report — the
        # actual trigger turned out to be ordinary camera dragging
        # interleaved with range changes, not the range controls alone.
        # A dedicated version counter that only update_map_mode's own
        # genuine-rebuild return path ever touches has no such ambiguity.
        dcc.Store(id="flux-cube-rebuild-version", data=0),
        # Debounced echoes of flux-cube-depth-mode/-bin-width's own live
        # values — explicit report: "on 3d view, when I press [a] few
        # options one after each other, the page try[s] to load
        # everything... step by step and will keep the system busy for a
        # long while because of tasks in buffer." Both fields are direct
        # `update_map_mode` Inputs with nothing debouncing them at the
        # Dash level (depth-mode's own RadioItems click is immediate by
        # nature; bin-width's own reliable-commit fix, a few rounds back,
        # made its blur/Enter commit *more* consistently immediate too) —
        # so several quick changes in a row each independently queue their
        # own full, genuinely expensive server-side flux-cube rebuild
        # (confirmed live: a single depth-mode toggle alone already costs
        # several seconds end to end), and Dash has no built-in mechanism
        # to cancel a stale one once a newer request has superseded it —
        # they all run to completion, one after another, even though only
        # the *last* one's result still matters by the time it lands.
        # `update_map_mode` itself takes these debounced Stores as its
        # real Input instead of the live fields directly (see the
        # clientside_callback pair below, right after the reliable-commit
        # one, for the actual debounce mechanism) — a burst of rapid
        # changes now collapses into a single rebuild for whatever the
        # *final* settled value was, the same principle a search box's
        # own "wait for the user to stop typing" debounce uses, just
        # applied to a button/radio click instead of a keystroke.
        dcc.Store(id="flux-cube-depth-mode-debounced", data="slice"),
        dcc.Store(id="flux-cube-bin-width-debounced",
                  data=str(int(DEFAULT_WAVE_BIN_WIDTH_ANGSTROM))
                  if DEFAULT_WAVE_BIN_WIDTH_ANGSTROM == int(DEFAULT_WAVE_BIN_WIDTH_ANGSTROM)
                  else str(DEFAULT_WAVE_BIN_WIDTH_ANGSTROM)),
        html.Div(id="flux-cube-depth-mode-debounce-dummy", style={"display": "none"}),
        html.Div(id="flux-cube-bin-width-debounce-dummy", style={"display": "none"}),
        # Written by the console.error/window.onerror interceptor installed
        # in index_string's <head> (see below) — the only way any purely
        # client-side failure (a ChunkLoadError, an Aladin JS exception,
        # etc.) ever reaches this file's Log panel at all, since the Log
        # panel is otherwise just a tee of *server-side* Python stdout/
        # stderr (see the _LogBuffer/_StreamTee comment above) and has zero
        # visibility into the browser by default.
        dcc.Store(id="js-error-sink"),
        # Written by index_string's own client-side inactivity timer
        # (MULTI_SESSION-only — see aps_explorer_session.IDLE_TIMEOUT_MINUTES)
        # once a tab has gone genuinely unused (no mouse/keyboard/click/
        # scroll) for that long; handle_idle_kill below evicts this
        # session's own SessionBundle immediately rather than waiting for
        # the much longer SESSION_TTL_SECONDS passive sweep.
        dcc.Store(id="idle-kill-trigger"),
        html.Div(id="idle-kill-status", style={"display": "none"}),
        # aladin-target/aladin-div below are the embedded Aladin Lite (JS)
        # sky-view panel — clicking a point on the map recenters it, and
        # clicking a point in it selects that target on the map (see the
        # aladin-target clientside_callback and the 'objectClicked'
        # handler inside it).
        dcc.Store(id="aladin-target"),
        dcc.Store(id="aladin-catalog-data"),
        # "Show contributing exposures" overlay data (see
        # _contrib_exposures_payload) — a separate Store/dummy-div pair
        # from aladin-catalog-data/aladin-catalog-dummy above rather than
        # folded into that same payload, since this overlay is drawn
        # entirely differently (A.circle/A.graphicOverlay shapes with no
        # click handler attached, one overlay per colour bucket like the
        # true-size overlay, not tied to the main catalog's own colour-
        # by/range/scale triggers directly — it's rebuilt via this
        # dedicated Store instead, on the checkbox, the per-file legend
        # checklist, or a fresh dataset load).
        dcc.Store(id="contrib-fibre-data"),
        html.Div(id="contrib-fibre-dummy", style={"display": "none"}),
        html.Div(id="contrib-legend-toggle-dummy", style={"display": "none"}),
        dcc.Interval(id="log-poll", interval=700, n_intervals=0),
        # Permanently present, invisible, never touched by any callback —
        # exists purely so Dash's client requests dcc.Graph's own lazily-
        # loaded JS chunk (async-graph.js) once, at initial page load. With
        # the old main-map dcc.Graph gone, every dcc.Graph in this app now
        # first appears from *inside* a tab-content callback's return value
        # rather than the initial layout — a well-documented Dash gotcha
        # (components used only inside callbacks, never in the initial
        # layout, can fail their first dynamic use with "ChunkLoadError:
        # Loading chunk ... failed (missing: .../async-graph.js)", even
        # though the file itself is served correctly) that broke every
        # Spectra/PPXF/Emission/etc. tab's plot the moment main-map was
        # removed. This is the standard workaround.
        dcc.Graph(id="_graph_preload_dummy", figure={}, style={"display": "none"}),
        # Same fix, same reason, for dash_table.DataTable's own lazily-loaded
        # chunk (async-highlight.js) — a *second*, independently-lazy-loaded
        # component type, not covered by the dcc.Graph dummy above. Only
        # IFU/MOS kinds have a DataTable anywhere in their initial
        # value-tables-panel; L1 has none at all (its "Header" DataTable
        # only ever appears from inside a tab-content callback). A session
        # that loads an L1 dataset first (never having loaded IFU/MOS
        # before) hit exactly this on the very first such tab click —
        # confirmed live via Playwright: repeated
        # "ChunkLoadError: Loading chunk ... failed (missing:
        # .../dash_table/async-highlight.js)" in the browser console, and
        # the tab silently never switching away from whatever was showing
        # (originally reported against the now-removed Metadata tab as
        # "cannot open Metadata, it jumps back to Spectra"; the same dummy
        # still guards the still-present Header tab's own DataTable).
        dash_table.DataTable(id="_datatable_preload_dummy", data=[], columns=[],
                              style_table={"display": "none"}),
        html.Div(id="aladin-lite-dummy", style={"display": "none"}),
        html.Div(id="aladin-catalog-dummy", style={"display": "none"}),
        html.Div(id="log-scroll-dummy", style={"display": "none"}),
        html.Div(id="js-error-dummy", style={"display": "none"}),
        dcc.Store(id="path-browser-state", data={"target": None, "mode": "dir", "cwd": _PATH_BROWSER_DEFAULT_ROOT}),
        dcc.Store(id="path-browser-result"),
        html.Div(id="path-browser-commit-dummy", style={"display": "none"}),
        _path_browser_modal(),
        # Wrapping just load-panel (not the whole main-panel) is what
        # decides WHEN this fires: only for as long as a genuine
        # Load/Reload callback touching load-panel's own outputs is
        # running (the thing the user could not previously tell was even
        # happening), not on routine, normally-fast interactions like a
        # spaxel click or tab switch. aps_style.LOADING_KWARGS is a plain
        # non-fullscreen circle spinner confined to this wrapper — the
        # separate #global-load-overlay (see index_string / serve_layout)
        # is what now covers the rest of the page while a load is in
        # flight; dcc.Loading's own fullscreen prop was tried and reverted
        # earlier as unsafe with multiple simultaneous instances (see that
        # overlay's own comments) and is deliberately not used here.
        dcc.Loading(**aps_style.LOADING_KWARGS, children=[
            # Closed on every initial render, not _load_panel_style(not
            # loaded) (the pre-existing "open it if nothing is loaded
            # yet" behaviour) -- explicit user complaint: "when I click
            # on explore in weave_or to open aps_explorer it first open[s]
            # the load L1... window... covering the log... I want [it] be
            # closed by default and only open if I click on that
            # b[u]tton." That old default meant a fresh weaveOR handoff
            # (nothing loaded yet at this exact instant, since
            # handle_url_handoff hasn't run yet -- it fires on the same
            # initial page load, not before it) opened the drawer right
            # over the log panel for the whole duration of the automatic
            # token-triggered load, the same "can't watch progress"
            # problem the Load/Reload-click fix (see the two clientside
            # callbacks near handle_l1_load/handle_l2_load) already
            # solved for the *manual* load path. `open-load-panel-btn`
            # (the "☰ Dataset & Settings" toolbar button) is still the
            # only way to open it from here.
            html.Div(id="load-panel", children=_load_form(),
                      style=_load_panel_style(False)),
        ]),
        # The one and only way to open the sidebar (before or after a
        # dataset is loaded) — always present (now living in _toolbar()
        # above, not a separately floating button here), so closing it via
        # its own × button never leaves the page with no way back. A
        # second "Load different dataset…" button inside main-panel used
        # to duplicate this; removed as redundant.
        html.Div(id="main-panel", children=[
            html.Div(id="file-info-panel", children=_file_info_panel()),
            html.Div(id="settings-panel", children=_settings_panel_children()),
            # display:flex + gap (rather than the two columns' own
            # inline-block width arithmetic alone) is what actually
            # produces visible breathing room between the coordinate-map
            # column and the tabs column — explicit report: "I want a gap
            # between the tabs for spectra plots and the aladin coordnate
            # or 3d map[.] Right now they are stick together." Flexbox
            # shrinks each child's declared 44%/54% width to fit
            # alongside this gap automatically (default flex-shrink:1),
            # so neither column needed its own width changed.
            html.Div([
                html.Div([
                    # Coordinate-map card (see _SECTION_CARD_STYLE's own
                    # docstring) — wraps the map-mode switch through the
                    # info-box below; deliberately stops short of Slit
                    # Explorer, which stays its own separate panel (see
                    # the comment above slit-explorer-container).
                    html.Div([
                    # 2D (Aladin coordinate map) / 3D (flux cube) mode
                    # selector — explicit request: "on the top of that
                    # panel we have two options of 2d (coordinate map)
                    # and 3d (datacube) mode where 2d is also default."
                    # Aladin has no 3D mode of its own at all ("as aladin
                    # cannot handle it, I want that if I set that option
                    # of 3D, then it use another plotting panel instead
                    # of the current aladin one") — see flux-cube-graph
                    # below for that separate panel.
                    dcc.RadioItems(
                        id="map-mode",
                        options=[
                            {"label": " 2D (coordinate map)", "value": "2d"},
                            {"label": " 3D (data cube)", "value": "3d"},
                        ],
                        value="2d", style={"fontSize": "12px", "marginBottom": "8px"},
                        inputStyle={"marginRight": "3px", "marginLeft": "10px"},
                    ),
                    # Every one of these is meaningful only in 2D mode
                    # (Aladin-specific: the panel itself, its own legend,
                    # DSS/True-size, and the contributing-exposures
                    # overlay+legend) — one wrapping div so 3D mode can
                    # hide all of them with a single style toggle rather
                    # than one per control.
                    #
                    # Both this and flux-cube-container below (see that
                    # id's own closing bracket for where this Loading
                    # wrapper ends) sit *inside* one shared dcc.Loading —
                    # explicit report: "When I change from 2d to 3d
                    # there is no loading notification and it seems like
                    # eveything is stuck." update_map_mode (aps_explorer.
                    # py) rebuilds the 3D figure AND flips
                    # flux-cube-container's own style from display:none
                    # to display:block in the *same* callback response
                    # (a deliberate fix for a *different*, earlier race —
                    # see that callback's own docstring), which means
                    # flux-cube-container is still display:none for the
                    # callback's entire in-flight duration — a
                    # dcc.Loading nested only *inside* it (as the
                    # existing one around flux-cube-graph alone still is)
                    # has no visible area to render its own spinner into
                    # until the moment the callback resolves, i.e. never
                    # gets a chance to show at all. Wrapping *outside*
                    # both toggled containers keeps the spinner overlay
                    # itself outside that display:none gate, so it's
                    # visible for the callback's whole in-flight window
                    # regardless of which inner panel is still hidden —
                    # scoped to just this map area (not a page-wide
                    # overlay), per the explicit "only in that panel not
                    # the other panels of course."
                    dcc.Loading(**aps_style.LOADING_KWARGS, children=[
                    html.Div(id="aladin-2d-controls", children=[
                        # The one and only spatial view now — sized/positioned
                        # entirely by the aladin-target clientside_callback
                        # (which measures this column's own box each time it
                        # runs), not by this server-rendered style, so it
                        # stays correctly matched even if the sidebar/window
                        # resizes it after this initial render.
                        html.Div(id="aladin-div", style=_ALADIN_DIV_STYLE),
                        html.Div(
                            id="aladin-legend",
                            children=_aladin_legend_children() if loaded else None,
                            style={"marginTop": "6px", "minHeight": "18px"},
                        ),
                        html.Div([
                            dcc.Checklist(
                                id="aladin-show-dss",
                                options=[{"label": " Show DSS background", "value": "show"}],
                                # On by default — explicit follow-up request:
                                # "always the DSS overlay is active[;] right
                                # now it is disable[d] by default and user
                                # sh[ould] tick it on to see the DSS."
                                # Previously off (the catalog overlay was
                                # considered the primary content, DSS opt-in
                                # context) — still freely toggleable off per
                                # dataset/session via the checkbox itself.
                                value=["show"], style={"display": "inline-block", "fontSize": "12px"},
                            ),
                            # Flip removed — explicit request, "nobody
                            # uses it".
                            dcc.Checklist(
                                id="aladin-true-size",
                                options=[{"label": " True fibre/spaxel size", "value": "true"}],
                                # Off by default — a fixed-pixel marker stays
                                # comfortably clickable at any zoom level; true
                                # size (WEAVE fibre core diameter, or the IFU
                                # pixel/spaxel scale) is the more scientifically
                                # meaningful view but can render as tiny,
                                # near-invisible dots when zoomed out.
                                value=[], style={"display": "inline-block", "fontSize": "12px",
                                                  "marginLeft": "16px"},
                            ),
                        ], style={"marginTop": "8px"}),
                        # "Show contributing exposures" — L1 stacked/
                        # superstacked/cube datasets only (see
                        # _contrib_exposures_row_style); a separate row below
                        # DSS/True-size rather than folded into it, since
                        # this one is conditionally hidden entirely while
                        # those two are always present.
                        html.Div([
                            dcc.Checklist(
                                id="contrib-exposures-toggle",
                                options=[{"label": " Show contributing exposures", "value": "show"}],
                                value=[], style={"display": "inline-block", "fontSize": "12px"},
                            ),
                            # dcc.Loading only around the note/legend, not
                            # the checkbox itself — this is the one-time
                            # (per session) real cost left in this feature
                            # (building every contributing file's own
                            # overlay the first time the checkbox is
                            # turned on, a few real seconds on a large
                            # multi-exposure stack); every *subsequent*
                            # per-file checklist tick/untick is a pure
                            # client-side show()/hide() with no server
                            # round-trip at all, so it never triggers this
                            # spinner.
                            dcc.Loading(**aps_style.LOADING_KWARGS, children=[
                                html.Div(id="contrib-exposures-note"),
                            ]),
                        ], id="contrib-exposures-row", style=_contrib_exposures_row_style()),
                    ]),
                    # The 3D alternative — hidden by default (2D is the
                    # starting mode), built lazily (only once map-mode is
                    # actually switched to "3d") since it's real, if
                    # cheap, extra computation nobody should pay for
                    # unless they ask for it, same principle as the
                    # contributing-exposures overlay above. A genuinely
                    # multi-second build for a large (~20,000-spaxel) IFU
                    # dataset even after optimisation (144,000 points,
                    # real Plotly.py array-coercion overhead) — wrapped in
                    # dcc.Loading (same "circle" spinner used around
                    # load-panel/tab-content elsewhere in this app) so the
                    # wait reads as "working" rather than "broken."
                    html.Div(
                        id="flux-cube-container", style={"display": "none"},
                        children=[
                            # Explicit request: "add the width of the z
                            # axis bin in angstrom to the top of the 3d
                            # panel so users can set it[,] but give a
                            # default one" — a static child of this
                            # always-present container (not conditional on
                            # a dataset being loaded), so update_map_mode
                            # can take it as a direct Input safely, the
                            # same reasoning cube-transparency-toggle used
                            # to rely on before it moved elsewhere.
                            # debounce=True: a full cube rebuild is
                            # expensive, so typing a new width shouldn't
                            # refire after every keystroke (same reasoning
                            # as color-vmin/color-vmax).
                            #
                            # The "Apply" button alongside it is a real,
                            # confirmed-necessary fix, not decoration —
                            # explicit report: "in 3d view, when I change
                            # the bin width for slice, it always keep the
                            # spacial distance between slices the same."
                            # Direct testing (real HTTP-response capture,
                            # not just visual inspection — see this app's
                            # own [[feedback_rigorous_verification]]
                            # convention) found the *binning itself* is
                            # already correct (verified directly:
                            # `flux_cube_figure` genuinely does keep the
                            # real wavelength span fixed and re-divides it
                            # for any bin width given to it) — the actual
                            # bug is that `debounce=True`'s implicit
                            # blur/Enter commit for *this specific* input
                            # frequently never reaches the server at all
                            # (confirmed: zero `update_map_mode` HTTP
                            # requests fire for most typed values, on both
                            # this session's build *and* the unmodified
                            # prior production image — a real, pre-existing
                            # Dash reliability gap, not something this
                            # session introduced), so the 3D view silently
                            # never rebuilds and genuinely does look
                            # unchanged regardless of what's typed. Kept
                            # `flux-cube-bin-width` itself as a direct
                            # Input too (harmless, still fires on the
                            # occasions debounce *does* commit) — the
                            # button is an explicit, reliable fallback
                            # trigger, the same "n_clicks button, not an
                            # implicit blur" shape the ◀/▶ wavelength-
                            # scroll buttons already use successfully.
                            # "Slice" (flat, no z-extent — the existing
                            # default, unchanged) vs "Cube" (each element
                            # extruded to fill its bin's real depth) —
                            # explicit request: "I have a radio botton
                            # that be select between slice or cube. IF
                            # slice we show wavelenght slices as thery
                            # are..and if cube. then instead of a slice,
                            # I want you to add a dept to each element
                            # along z...still default mode to slice."
                            # See flux_cube_figure's own `depth_mode`
                            # docstring for the actual geometry.
                            html.Div([
                                html.Label("Depth:", style={"fontSize": "12px", "fontWeight": "600", "marginRight": "6px"}),
                                dcc.RadioItems(
                                    id="flux-cube-depth-mode",
                                    options=[
                                        {"label": " Slice", "value": "slice"},
                                        {"label": " Cube", "value": "cube"},
                                    ],
                                    value="slice", style={"display": "inline-block", "fontSize": "12px"},
                                    inputStyle={"marginRight": "3px", "marginLeft": "10px"},
                                ),
                            ], style={"marginBottom": "6px"}),
                            html.Div([
                                html.Label("Wavelength bin width (Å):",
                                           style={"fontSize": "12px", "fontWeight": "600", "marginRight": "6px"}),
                                dcc.Input(
                                    id="flux-cube-bin-width", type="text",
                                    # int() when it's a whole number — "100" reads
                                    # more like a plain default than "100.0" does.
                                    value=str(int(DEFAULT_WAVE_BIN_WIDTH_ANGSTROM))
                                    if DEFAULT_WAVE_BIN_WIDTH_ANGSTROM == int(DEFAULT_WAVE_BIN_WIDTH_ANGSTROM)
                                    else str(DEFAULT_WAVE_BIN_WIDTH_ANGSTROM),
                                    debounce=True,
                                    style={"width": "80px", "marginRight": "6px"},
                                ),
                                html.Button("Apply", id="flux-cube-bin-width-apply", n_clicks=0,
                                            title="Rebuild the 3D view with this bin width",
                                            style={"fontSize": "11px", "padding": "3px 10px"}),
                            ], style={"marginBottom": "6px", "display": "flex", "alignItems": "center"}),
                            # Wavelength sub-range selector — explicit
                            # request: "scroll through the wavelength
                            # range and select a start and end." Empty
                            # Start/End (the default) leaves the cube at
                            # its full native coverage, matching
                            # flux_cube_figure's own wave_min/wave_max=
                            # None default; typing either restricts which
                            # samples get binned at all, not just what's
                            # displayed (see that function's own
                            # docstring). Same type="text"+debounce
                            # convention as flux-cube-bin-width just above
                            # (and the same reason: a debounced
                            # type="number" dcc.Input can commit None on
                            # blur after a clear-then-retype, confirmed
                            # live for that field — see update_map_mode's
                            # own docstring).
                            #
                            # "Lock range width" ties End to Start + the
                            # bin width above (a clientside callback, pure
                            # arithmetic on already-present values, no
                            # server round trip needed) — explicit
                            # follow-up: the width to lock to "is the one
                            # we already have [i.e. the bin-width field]
                            # ... so no other computation is needed."
                            # With it checked, "scrolling" through the
                            # cube is just repeatedly changing Start (by
                            # hand, or via the ◀/▶ step buttons below,
                            # which shift Start — and, indirectly via the
                            # same lock, End — by one bin width per
                            # click) while End always follows automatically.
                            #
                            # The lock control itself sits *between*
                            # Start and End now (replacing the old
                            # separate "Lock range width..." checklist
                            # row entirely) — explicit follow-up request:
                            # "instead of a radio botton can you add the
                            # lock size that connect end and start
                            # together similar to what usuallay people
                            # use in scaling images...where if you click
                            # on it it lock or unlock...something like a
                            # U shape that each side is connected to
                            # start and end and then behind it there is
                            # a lock sign." flux-cube-lock-width is now a
                            # dcc.Store (was a dcc.Checklist) holding the
                            # exact same []/["locked"] value shape the
                            # existing lock-sync/scroll clientside
                            # callbacks already read — swapping the
                            # component type needed no change to either
                            # of those, only to whatever *writes*
                            # flux-cube-lock-width now (the new toggle
                            # button's own clientside callback, right
                            # after this layout).
                            #
                            # The bracket-glyph-wedged-between-the-fields
                            # design from that first attempt is gone —
                            # explicit follow-up: "The lock sign/symbol
                            # between start and end is not clear what it
                            # does...the sumbol is not clear and it stuck
                            # to both start and end fields and make the
                            # design very ugly. I need something more
                            # profetional design." Replaced with a
                            # labelled pill button (glyph *and* the word
                            # "Lock"/"Unlock", not a bare icon guessing at
                            # meaning) with real breathing room on every
                            # side (`gap` on the row, not the previous
                            # negative-margin "wedge it flush against both
                            # neighbours" approach) — the same "icon plus
                            # an actual word" shape the Collapse/Expand
                            # dataset-info toggle already used successfully
                            # last round, deliberately reused here rather
                            # than inventing a new visual language.
                            html.Div([
                                html.Label("Wavelength range (Å):",
                                           style={"fontSize": "12px", "fontWeight": "600"}),
                                html.Button("◀", id="flux-cube-wave-scroll-back", n_clicks=0,
                                            title="Scroll range back by one bin width",
                                            style={"fontSize": "11px", "padding": "2px 7px"}),
                                dcc.Input(id="flux-cube-wave-min", type="text", placeholder="Start",
                                          debounce=True, style={"width": "70px"}),
                                html.Span("–", style={"color": "var(--pyaps-ink-faint)"}),
                                dcc.Input(id="flux-cube-wave-max", type="text", placeholder="End",
                                          debounce=True, style={"width": "70px"}),
                                html.Button("▶", id="flux-cube-wave-scroll-fwd", n_clicks=0,
                                            title="Scroll range forward by one bin width",
                                            style={"fontSize": "11px", "padding": "2px 7px"}),
                                html.Button(
                                    # Action-oriented label (what clicking
                                    # *will do*, not the current state) —
                                    # starts unlocked, so this starts as
                                    # "Lock"; the clientside toggle above
                                    # flips it to "Unlock" once clicked,
                                    # matching a common mute/unmute-style
                                    # button convention.
                                    "🔒 Lock", id="flux-cube-lock-toggle", n_clicks=0,
                                    title="Lock End to Start + bin width (scroll by Start only)",
                                    style={
                                        "border": "1px solid var(--pyaps-line-strong)",
                                        "borderRadius": "var(--pyaps-radius-pill)",
                                        "backgroundColor": "var(--pyaps-paper-raised)", "color": "var(--pyaps-ink-muted)",
                                        "fontSize": "11px", "fontWeight": "600", "padding": "4px 12px",
                                        "cursor": "pointer", "whiteSpace": "nowrap",
                                    },
                                ),
                                html.Button(
                                    "Reset", id="flux-cube-wave-range-reset", n_clicks=0,
                                    title="Reset Start/End back to this dataset's full native wavelength coverage",
                                    style={"fontSize": "11px", "padding": "4px 10px"},
                                ),
                                dcc.Store(id="flux-cube-lock-width", data=[]),
                            ], style={"marginBottom": "4px", "display": "flex", "alignItems": "center", "gap": "6px", "flexWrap": "wrap"}),
                            # Out-of-range / empty-selection feedback —
                            # explicit report: "When I go before the
                            # wavelenght range and after the max, I need
                            # it put a message or wanrng next to the
                            # range saying out of range...right now if we
                            # go out of range nothing happens and just
                            # the 3d plot get white/empty." Investigated
                            # directly (not assumed) whether this and the
                            # separately-reported "change one range, then
                            # try a different range, screen stays white"
                            # were a genuine caching bug: built a
                            # dedicated live test cycling through several
                            # in-range, out-of-range, and back-to-in-range
                            # selections (both Slice and Cube, with and
                            # without Lock) — every transition correctly
                            # recovered real data whenever the requested
                            # range genuinely overlapped the dataset's own
                            # coverage, and correctly (if silently) showed
                            # nothing when it didn't. There never was a
                            # stuck/poisoned state — the *only* real gap
                            # was the total silence on a genuinely empty
                            # selection, which is exactly what this text
                            # now fixes; a still-empty screen after fixing
                            # a real out-of-range value was never
                            # reproduced. Populated by the same clientside
                            # wave-filter callback that already does the
                            # NaN'ing (flux-cube-wave-filter-dummy's own
                            # callback, right below) — it already knows
                            # exactly how many vertices survived a given
                            # [Start, End], with no extra computation
                            # needed to know when to show this.
                            # display:none only when genuinely empty (see
                            # assets/style.css's `:empty` rule) rather
                            # than toggled by the callback itself — an
                            # empty children="" already collapses to no
                            # visible text, this only needs to also stop
                            # reserving a blank line's worth of vertical
                            # space when there's nothing to show.
                            html.Div(
                                id="flux-cube-wave-range-warning",
                                style={"fontSize": "11px", "color": "var(--pyaps-signal)", "marginBottom": "6px"},
                            ),
                            # A "Apply range to 2D colour" button used to
                            # live here (reusing these Start/End fields to
                            # recolour the 2D Aladin map's points by a
                            # wavelength sub-range) — removed per explicit
                            # follow-up: "remove apply to 2d feature for
                            # wavelenght range as it does not work well
                            # and it is not what I want."
                            # Off by default per explicit request — the
                            # underlying mechanism (see the clientside
                            # click-handling callback below, which is what
                            # this toggle actually gates: it never rebinds
                            # or re-runs, it just reads
                            # window._pyapsFluxCubeClickEnabled at click
                            # time) is still slow enough on a large cube
                            # that click-to-select was reported as "still
                            # very slow," so it now has to be turned on
                            # explicitly rather than always being live.
                            html.Div([
                                dcc.Checklist(
                                    id="flux-cube-click-toggle",
                                    options=[{"label": " Enable click-to-select in 3D view",
                                              "value": "enabled"}],
                                    value=[], style={"display": "inline-block", "fontSize": "12px"},
                                ),
                                html.Div(
                                    "Experimental feature — selecting a fibre/spaxel by clicking "
                                    "directly in the 3D view can be slow on large datasets.",
                                    style={"fontSize": "11px", "fontStyle": "italic", "color": "var(--pyaps-ink-faint)",
                                           "marginTop": "2px"},
                                ),
                            ], style={"marginBottom": "6px"}),
                            html.Div(id="flux-cube-click-toggle-dummy", style={"display": "none"}),
                            dcc.Loading(**aps_style.LOADING_KWARGS, children=[
                                # Same bordered/rounded treatment as
                                # _ALADIN_DIV_STYLE's 2D counterpart, so
                                # switching map-mode doesn't drop the
                                # boundary the request asked for either
                                # way ("a border around the coordinate
                                # (2d or 3d) plots").
                                dcc.Graph(id="flux-cube-graph", config=_FLUX_CUBE_GRAPH_CONFIG,
                                          style={"height": _MAP_HEIGHT,
                                                 "border": "1px solid var(--pyaps-line-strong)",
                                                 "borderRadius": "var(--pyaps-radius-m)"}),
                            ]),
                        ],
                    ),
                    ]),
                    # Click-handling target for the flux cube — see the
                    # clientside callback below for why this can't just
                    # be Input("flux-cube-graph", "clickData") the way
                    # every other click-driven selection in this app is.
                    html.Div(id="flux-cube-click-dummy", style={"display": "none"}),
                    # Dummy Output for the client-side wavelength-range
                    # filter callback (see its own long comment, right
                    # after the ◀/▶ scroll clientside callback) — that
                    # callback's real work is a Plotly.restyle() side
                    # effect, not a Dash prop write.
                    html.Div(id="flux-cube-wave-filter-dummy", style={"display": "none"}),
                    # Dummy Output for the reliable-field-commit
                    # clientside callback (see its own long comment, right
                    # before the ◀/▶ scroll clientside callback) — that
                    # callback's real work is binding native listeners and
                    # dash_clientside.set_props() side effects, not a Dash
                    # prop write.
                    html.Div(id="flux-cube-field-commit-dummy", style={"display": "none"}),
                    # Colour range + Scale + Palette — deliberately a
                    # sibling positioned *after both* aladin-2d-controls
                    # and flux-cube-container (not before either, and not
                    # duplicated per mode) so it always lands directly
                    # below whichever one is actually visible, matching
                    # explicit requests from two different rounds: "put
                    # the colour range option below the colorbar" (2D,
                    # established first) and, later, "make sure the color
                    # range and scale and palette are below the color bar
                    # and centred" for *both* 2D and 3D once the 3D cube
                    # got its own colourbar (see flux_cube_figure's own
                    # colorbar= — reshaped horizontal, matching the 2D
                    # legend's own shape, specifically so this ordering
                    # reads the same way in either mode). Visible/usable
                    # in both modes — explicit follow-up: "in 3d view can
                    # we have other scaling as we have in 2d like log,
                    # sinh, power etc?" (Scale) and "...in all maps either
                    # 2D or 3D..." (Palette) — see update_map_mode's own
                    # Inputs for how a change here reaches the 3D figure
                    # too.
                    html.Div(
                        id="aladin-color-range-container",
                        children=_aladin_color_range_control() if loaded else None,
                    ),
                    html.Div(
                        id="aladin-info-box",
                        children=_aladin_info_text(initial_item) if loaded else "No selection.",
                        style={
                            "marginTop": "8px", "padding": "8px 12px", "backgroundColor": "var(--pyaps-paper-sunken)",
                            "border": "1px solid var(--pyaps-line-strong)", "borderRadius": "4px",
                            # Bold — per explicit user request, to make a
                            # changed selection visually dominant/eye-catching
                            # at a glance rather than blending into the page.
                            "fontSize": "12px", "fontFamily": "monospace", "fontWeight": "700",
                        },
                    ),
                    ], style=_SECTION_CARD_STYLE),
                    # Slit Explorer is deliberately last, below everything
                    # else in this column (map/legend/colour-range/DSS-
                    # true-size checkboxes/info box) — explicit user
                    # request: those all belong to "the Aladin coordinate
                    # map" as one cluster, and Slit Explorer is a genuinely
                    # separate panel, not something to interleave into the
                    # middle of that cluster. The top border + spacing
                    # below is what actually reads as "separate panel," not
                    # just DOM order.
                    html.Div(
                        id="slit-explorer-container",
                        children=_slit_explorer_container(initial_item) if loaded else None,
                        style={"marginTop": "14px", "paddingTop": "10px",
                               "borderTop": "1px solid var(--pyaps-line)"},
                    ),
                # flex: "0 0 44%" (was "width": "44%") -- a plain
                # percentage width plus the row's own 20px gap summed to
                # only ~98% of the row's actual content box, leaving the
                # tabs/spectra column's own right edge short of the
                # container's right edge -- explicit follow-up report:
                # "the Additional map panel [goes] further right than
                # the right end of the spectra panel... aligned with the
                # Bin galaxy table [instead]... the spectra panel must be
                # extended more to the right." A fixed flex-basis here
                # plus flex:1 on the tabs column (below) makes the tabs
                # column consume exactly whatever's left after this
                # column and the gap, landing flush on the container's
                # right edge regardless of gap/rounding -- matching both
                # the additional-map-panels section and the Bin/Spaxel
                # tables below, which are both plain full-width blocks.
                ], style={"flex": "0 0 44%", "minWidth": 0, "display": "inline-block", "verticalAlign": "top"}),
                html.Div([
                    dcc.Tabs(id="tabs", value=(tabs[0].value if tabs else "none"), children=tabs,
                             style=aps_style.TABS_STYLE),
                    # dcc.Loading here (not just around load-panel) per
                    # explicit user request — clicking a new point/bin
                    # gave no feedback that anything had registered at all
                    # short of reading the BIN_ID/APS_ID text and noticing
                    # it changed; a spinner over the plot the instant a
                    # click fires makes "yes, your click registered, this
                    # is loading" immediately obvious even before the
                    # (potentially slow, for a large fit) new plot arrives.
                    dcc.Loading(**aps_style.LOADING_KWARGS, children=[html.Div(id="tab-content")]),
                # flex: "1 1 0%" (was "width": "54%") -- see the map
                # column's own comment just above: this makes the tabs/
                # spectra card grow to fill whatever width is actually
                # left over (container width minus the map column's
                # fixed 44% minus the row's own 20px gap), so its right
                # edge always lands exactly on the container's right
                # edge, flush with the additional-map-panels section and
                # the Bin/Spaxel tables below instead of falling short of
                # them by the old fixed-percentage rounding.
                ], style={**_SECTION_CARD_STYLE, "flex": "1 1 0%", "minWidth": 0, "display": "inline-block", "verticalAlign": "top"}),
            # alignItems: "stretch" (was "flex-start") -- explicit
            # follow-up report: "the bottom of the main map plot and the
            # bottom of the main spectra plot are not aligned... I want
            # them to be well vertically aligned." "flex-start" let each
            # column's own card be exactly as tall as its own content,
            # so whichever side happened to be shorter left its own
            # bottom border floating above the taller one's. "stretch"
            # (flexbox's own default, only ever overridden away from it
            # here) makes both cards match the taller one's height
            # instead, so their bottom borders land at the same line
            # regardless of which side's actual content is shorter.
            ], style={"display": "flex", "gap": "20px", "alignItems": "stretch"}),
            _extra_map_panels_section(),
            html.Div(id="value-tables-panel", children=_value_tables_children()),
        ], style=_main_panel_style(loaded)),
        _log_panel(),
    ], style={"padding": "16px", "fontFamily": "Helvetica, Arial, sans-serif"})


# update_title=None: Dash's default behaviour rewrites the browser tab's
# title to "Updating..." for the duration of every single callback
# round-trip, including the always-on 700ms log-panel poll (log-poll,
# below — needed for the live log panel, not something to remove) — with
# that firing forever in the background, the title flickered constantly.
# This is exactly what the parameter exists for: disables the automatic
# title rewrite without affecting any callback's actual behaviour.
app = Dash(__name__, title="PyAPS Explorer", suppress_callback_exceptions=True,
           update_title=None, url_base_pathname=_sess.URL_PREFIX,
           # Excludes assets/vendor/ (currently just aladin.js — see its
           # own README) from Dash's usual "auto-inject every .js/.css
           # under assets/ into every page" behaviour. Aladin Lite is
           # only ever needed once the 2D map actually mounts, loaded
           # lazily by the aladin-target clientside_callback below — an
           # extra ~1.8MB script tag landing on *every* page load
           # (including ones that never touch 2D mode at all) would be
           # its own new performance regression in the name of fixing a
           # network-reachability one.
           #
           # `assets_path_ignore` (matched against *directory*-path
           # components), not `assets_ignore` (matched only against bare
           # *filenames* — confirmed live this was the wrong one to use
           # here: "vendor/.*" against a bare "aladin.js" never matches
           # anything, so the very first attempt silently auto-injected
           # the full 1.8MB script into every page anyway, caught only
           # because it visibly slowed the whole page down enough to
           # start failing this project's own existing drawer-timing
           # regression tests).
           assets_path_ignore=["vendor"])


def _table_rows_to_csv_download(table_id, virtual_data, raw_data, columns):
    """The actual data-to-CSV logic behind every table's "Export CSV"
    button — a plain function (not a Dash-callback closure) specifically
    so it's directly unit-testable without fighting Dash's own callback-
    dispatch machinery (which wraps registered callbacks in a way that
    isn't callable like a normal function outside a real request).

    `derived_virtual_data` (`virtual_data`) is used in preference to the
    raw `data` prop when present — it reflects whatever sort/filter the
    user currently has active, matching the "export exactly what's
    shown" behaviour DataTable's own native export offered. It's `None`
    until the user has actually triggered a sort/filter at least once in
    the browser, so `raw_data` is the fallback, not a redundant belt-
    and-braces read. Returns `no_update` (not an empty/broken CSV) if
    there are no rows to export at all (e.g. a MOS target with no
    galaxy/stellar data for that particular table).
    """
    rows = virtual_data if virtual_data is not None else raw_data
    if not rows:
        return no_update
    colids = [c["id"] for c in columns] if columns else list(rows[0].keys())
    colids = [c for c in colids if c in rows[0]]
    colnames = {c["id"]: c["name"] for c in columns} if columns else {}
    df = pd.DataFrame(rows)[colids]
    df.columns = [colnames.get(c, c) for c in colids]
    return dcc.send_data_frame(df.to_csv, f"{table_id}.csv", index=False)


def register_csv_export(table_id):
    """Wire up `table_id`'s "Export CSV" button (built by
    `apsPlot.datatable_export.csv_export_row`, called once per real table
    across this file/aps_l1_preview.py/aps_IFUviewer.py/aps_MOSviewer.py —
    see that module's own docstring for why this exists instead of
    DataTable's native `export_format`) to a callback sending the
    table's currently displayed rows as a CSV download — see
    `_table_rows_to_csv_download`'s own docstring for the actual logic.
    """
    @app.callback(
        Output({"type": "csv-download", "table": table_id}, "data"),
        Input({"type": "csv-export-btn", "table": table_id}, "n_clicks"),
        State(table_id, "derived_virtual_data"),
        State(table_id, "data"),
        State(table_id, "columns"),
        prevent_initial_call=True,
    )
    def _export(n_clicks, virtual_data, raw_data, columns):
        return _table_rows_to_csv_download(table_id, virtual_data, raw_data, columns)


# One register_csv_export() call per real table's own id — every id used
# with csv_export_row() anywhere in this file/aps_l1_preview.py/
# aps_IFUviewer.py/aps_MOSviewer.py must be listed here, or that table's
# button is wired to nothing and silently does nothing when clicked.
# l1_mod.HEADER_TABLE_MAX_FILES-many ids for the per-input-file L1
# header tables (see that constant's own docstring for why this is a
# fixed, generous-headroom count rather than a dynamic one).
for _table_id in (
    "l1-info-table",
    "l2-info-table",
    *(f"l1-header-table-{_i}" for _i in range(l1_mod.HEADER_TABLE_MAX_FILES)),
    "spaxel-values-table",
    "bin-values-table",
    "class-values-table",
    "star-values-table",
    "galaxy-values-table",
    "ifu-processing-history-table",
):
    register_csv_export(_table_id)
del _table_id


# Reserves the vertical scrollbar's space permanently, so a page-height
# change (e.g. a taller/shorter tab-content plot after selecting a
# different item) never toggles the scrollbar's presence and shifts every
# element's available width by its ~15-17px — general good practice
# against layout shift, kept from an earlier round where it was one of
# several (now-superseded, since the Plotly map is gone) theories for the
# zoom-reset bug.
# Client-side inactivity auto-kill (MULTI_SESSION only — a standalone,
# single-user process has no reason to evict its own only session). Real
# mouse/keyboard/click/scroll activity resets the timer; SESSION_TTL_
# SECONDS-based staleness (aps_explorer_session.py) does NOT count as
# activity here, deliberately -- the log panel's own 700ms poll
# (log-poll, below) would otherwise keep any merely-open tab "active"
# forever, defeating the whole point (see IDLE_TIMEOUT_MINUTES's own
# comment in aps_explorer_session.py). Once idle, this fires
# idle-kill-trigger (handled server-side by handle_idle_kill, which
# actually frees the session's memory), disables the log-poll interval
# so the now-dead tab stops generating any further requests at all, and
# shows a plain full-page notice rather than leaving a silently-reset,
# confusing-looking page.
_idle_timeout_script = ""
if _sess.MULTI_SESSION:
    _idle_timeout_script = """
        <script>
            (function() {
                var IDLE_MS = %(idle_ms)d;
                var IDLE_MIN = %(idle_min)d;
                var lastActivity = Date.now();
                var killed = false;
                ["mousemove", "keydown", "click", "scroll", "touchstart"].forEach(function(evt) {
                    window.addEventListener(evt, function() { lastActivity = Date.now(); }, {passive: true});
                });
                setInterval(function() {
                    if (killed || (Date.now() - lastActivity) < IDLE_MS) { return; }
                    killed = true;
                    function trySet(attemptsLeft) {
                        if (window.dash_clientside && window.dash_clientside.set_props) {
                            window.dash_clientside.set_props("idle-kill-trigger", {data: Date.now()});
                            window.dash_clientside.set_props("log-poll", {disabled: true});
                        } else if (attemptsLeft > 0) {
                            setTimeout(function() { trySet(attemptsLeft - 1); }, 200);
                        }
                    }
                    trySet(30);
                    var overlay = document.createElement("div");
                    overlay.style.cssText = "position:fixed;top:0;left:0;right:0;bottom:0;" +
                        "background:rgba(255,255,255,0.97);z-index:99999;display:flex;" +
                        "align-items:center;justify-content:center;font-family:sans-serif;" +
                        "font-size:16px;color:#333;text-align:center;padding:20px;";
                    overlay.innerHTML = "<div>Session ended after " + IDLE_MIN +
                        " minutes of inactivity.<br>Refresh the page to start a new one.</div>";
                    document.body.appendChild(overlay);
                }, 15000);
            })();
        </script>
    """ % {"idle_ms": _sess.IDLE_TIMEOUT_MINUTES * 60 * 1000, "idle_min": _sess.IDLE_TIMEOUT_MINUTES}

app.index_string = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            html { overflow-y: scroll; scrollbar-gutter: stable; }
            /* Tells the browser which theme its own *native* chrome
               (checkbox/radio ticks, autofill highlight, a couple of
               scrollbar renderers) should follow — otherwise those bits
               track the OS's own dark/light setting regardless of what
               the in-app light/auto/dark toggle picked, a real mismatch
               a user could actually hit (e.g. explicitly choosing
               "Light" while their OS is in dark mode). Same three-state
               contract as every token in assets/style.css — dark is the
               base/default there now (2026-08-16), light the override,
               so this flips to match: default dark, light only for an
               explicit choice or a genuine OS light preference. */
            :root { color-scheme: dark; }
            @media (prefers-color-scheme: light) {
                :root:not([data-theme="dark"]) { color-scheme: light; }
            }
            :root[data-theme="light"] { color-scheme: light; }
            /* Default browser body margin would otherwise leave a thin gap
               around _banner()'s edge-to-edge strip. */
            body { margin: 0; }
            .path-browser-row:hover { background-color: var(--pyaps-accent-soft); }
            /* Placeholder ("example value") text must read as an example,
               not as a real current value — smaller, greyer, italic, and
               never allowed to overflow its own input box (::placeholder
               text doesn't get clipped/ellipsized by default the way real
               overflowing content would). Applies app-wide: every dcc.Input/
               dcc.Textarea placeholder in every load-form field and the
               colour-range boxes, per explicit user request. */
            ::placeholder {
                color: var(--pyaps-ink-faint) !important; font-style: italic; opacity: 1;
                font-size: 11px !important;
            }
            input, textarea { text-overflow: ellipsis; }
            .modern-tabs .tab--selected { border-bottom: 3px solid var(--pyaps-accent); }
            /* Belt-and-braces alongside showFullscreenControl:false in the
               A.aladin() init options (item 6: "disable aladin maximize...
               it is destroying the page structure and some text boxes are
               overlaid and some behind") — this app loads Aladin Lite's
               "latest" build (unpinned), so a CSS fallback that hides the
               control outright survives even if some future version adds
               it back by default or renames how it's toggled. */
            .aladin-fullscreen, .aladinFullscreenButton { display: none !important; }
            #open-load-panel-btn:hover { background-color: var(--pyaps-accent-soft) !important; }
            /* The fixed-height, `overflow-y: auto` boxes wrapping every
               tab's plot(s) (L1's Spectra/FWHM, MOS's Redrock/RVS/FERRE/
               PPXF/EMI, IFU's Spectrum/Stellar/Emission/LS/RVS/FERRE/
               Processing History) already only show a scrollbar when
               content genuinely overflows the box — `auto`, not `scroll`,
               is correct for that part. What explicit user testing found
               wrong is a *browser/OS* behaviour on top of that: many
               platforms (macOS trackpad/touch-device settings especially)
               render `overflow-y: auto`'s scrollbar as an "overlay" one
               that's invisible except while actively scrolling/hovering,
               which reads as "no scrollbar at all" the rest of the time —
               easy to miss that a plot extends below the visible box.
               `::-webkit-scrollbar` rules replace that OS overlay
               scrollbar with a normal always-rendered one in any
               Chromium/WebKit browser; `scrollbar-width`/`scrollbar-color`
               do the same for Firefox. Applies only to boxes explicitly
               opted in via this class (not every scrollable element on
               the page) — see `pyaps-scrollbox` in aps_l1_preview.py/
               aps_MOSviewer.py/aps_IFUviewer.py. */
            .pyaps-scrollbox {
                scrollbar-width: thin;
                scrollbar-color: var(--pyaps-line-strong) var(--pyaps-paper-sunken);
            }
            .pyaps-scrollbox::-webkit-scrollbar { width: 12px; height: 12px; }
            .pyaps-scrollbox::-webkit-scrollbar-track { background: var(--pyaps-paper-sunken); }
            .pyaps-scrollbox::-webkit-scrollbar-thumb {
                background-color: var(--pyaps-line-strong); border-radius: 6px; border: 2px solid var(--pyaps-paper-sunken);
            }
            .pyaps-scrollbox::-webkit-scrollbar-thumb:hover { background-color: var(--pyaps-ink-faint); }

            /* Full-page loading overlay for "loading a new dataset" --
               explicit request: seeing the *previous* dataset's plots
               still on screen while a new one is loading is misleading,
               so the whole page goes white with a centred spinner
               instead, until the new dataset's own content replaces it.
               Deliberately NOT dcc.Loading's own fullscreen=True prop --
               confirmed live earlier this same project that multiple
               simultaneous fullscreen dcc.Loading instances broke real
               interactive use in a way not fully understood (laggy,
               tabs producing no content) -- position:fixed + a plain
               clientside show/hide is fully self-contained instead, no
               dependency on that prop at all. #global-load-overlay
               itself has no children in the Dash layout; the spinner
               is pure CSS via ::after so nothing here needs its own
               lazily-loaded JS chunk (the same "ChunkLoadError on first
               use" class of bug this app has hit before for dcc.Graph/
               dash_table.DataTable). The Log panel gets a higher
               z-index (see _log_panel's own style) so it stays visible
               and usable through the overlay, per explicit request. */
            #global-load-overlay {
                display: none;
                position: fixed;
                inset: 0;
                background: var(--pyaps-paper);
                z-index: 9500;
                align-items: center;
                justify-content: center;
            }
            #global-load-overlay::after {
                content: "";
                width: 60px; height: 60px;
                border: 6px solid var(--pyaps-line-strong);
                border-top-color: var(--pyaps-accent);
                border-radius: 50%;
                animation: pyaps-spin 0.8s linear infinite;
            }
            @keyframes pyaps-spin { to { transform: rotate(360deg); } }
        </style>
        <script>
            // Every prior "why don't we see an error for this" report this
            // session (ChunkLoadError, the round-6 zoom-reset bug, etc.) has
            // had the same explanation: the Log panel is a tee of *server-
            // side* Python stdout/stderr only (see the _LogBuffer/_StreamTee
            // comment in this file) — it has zero visibility into the
            // browser, so a purely client-side failure can be happening
            // constantly while the log looks perfectly clean. This installs
            // as early as possible (before Dash's own bundle even loads, so
            // nothing gets missed) a console.error/window.onerror/
            // unhandledrejection interceptor that forwards every one into
            // the same Log panel via a Store (js-error-sink) a normal Python
            // callback then appends into LOG — console.warn is deliberately
            // NOT forwarded (WebGL/perf hints fire constantly and would be
            // pure noise; console.error is what React/webpack/Aladin all
            // actually use for real failures).
            (function() {
                var seq = 0;
                function forward(kind, msg) {
                    try {
                        seq += 1;
                        var entry = {seq: seq, kind: kind, msg: String(msg).slice(0, 800)};
                        function trySet(attemptsLeft) {
                            if (window.dash_clientside && window.dash_clientside.set_props) {
                                window.dash_clientside.set_props("js-error-sink", {data: entry});
                            } else if (attemptsLeft > 0) {
                                setTimeout(function() { trySet(attemptsLeft - 1); }, 200);
                            }
                        }
                        trySet(30);
                    } catch (e) { /* never let error-reporting itself throw */ }
                }
                var origError = console.error;
                console.error = function() {
                    forward("console.error", Array.prototype.slice.call(arguments).join(" "));
                    origError.apply(console, arguments);
                };
                window.addEventListener("error", function(e) {
                    forward("uncaught", (e.message || "Error") + " @ " + (e.filename || "") + ":" + (e.lineno || ""));
                });
                window.addEventListener("unhandledrejection", function(e) {
                    var r = e.reason;
                    forward("unhandledrejection", (r && r.message) ? r.message : String(r));
                });
            })();
        </script>
        __IDLE_TIMEOUT_SCRIPT__
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>""".replace("__IDLE_TIMEOUT_SCRIPT__", _idle_timeout_script)


# --------------------------------------------------------------------------- #
# Per-browser session isolation (only active with --multi-session /
# PYAPS_EXPLORER_MULTI_SESSION=1 — the Docker/gunicorn deployment path).
# See aps_explorer_session.py's module docstring for the full design.
# Exempt /healthz and Dash's own static-asset paths: a container
# orchestrator's liveness probe hits /healthz every 10-30s and would
# otherwise mint a throwaway empty session on every single check.
# --------------------------------------------------------------------------- #

_SESSION_EXEMPT_PATHS = {"/healthz"}
# Prefix-relative: Dash's own static-asset/component-suite routes carry
# whatever url_base_pathname the app was constructed with (see `app =
# Dash(...)` above) — e.g. "/weave/assets/..." rather than "/assets/..."
# when PYAPS_EXPLORER_URL_PREFIX=/weave/ is set. /healthz above is a
# plain Flask route outside Dash's own routing entirely (see _healthz()
# below) and stays unprefixed regardless — a container's own internal
# liveness probe hits it directly on the container's port, never through
# whatever external path prefix a reverse proxy adds.
_SESSION_EXEMPT_PREFIXES = tuple(
    _sess.URL_PREFIX.rstrip("/") + p
    for p in ("/assets/", "/_dash-component-suites/", "/_favicon")
)


def _session_exempt(path):
    return path in _SESSION_EXEMPT_PATHS or path.startswith(_SESSION_EXEMPT_PREFIXES)


# Dash's own internal XHR routes (confirmed directly against this app's
# real url_map, not guessed) — none of these are the actual page
# navigation, so none of them carry the URL the *user* actually asked
# for as their own request's query string; see
# `_incoming_url_wants_different_dataset`'s own docstring for exactly
# why that distinction matters.
_DASH_INTERNAL_SUFFIXES = (
    "_dash-layout", "_dash-dependencies", "_dash-update-component",
    "_reload-hash", "_favicon.ico",
)


def _is_dash_internal_request(path):
    stripped = path[len(_sess.URL_PREFIX.rstrip("/")):] if path.startswith(_sess.URL_PREFIX.rstrip("/")) else path
    stripped = stripped.lstrip("/")
    return stripped in _DASH_INTERNAL_SUFFIXES or _session_exempt(path)


@app.server.before_request
def _bind_session():
    if not _sess.MULTI_SESSION or _session_exempt(flask_request.path):
        return
    bundle, sid, token = _sess.bind_request(flask_request.cookies.get(_sess.SESSION_COOKIE_NAME))
    flask_g._pyaps_session_id = sid
    # Stashed here, read back by `_incoming_url_wants_different_dataset`
    # (called from `serve_layout()`) — see that function's own docstring
    # for why capturing it *here*, on the genuine page-navigation
    # request, is the only place it's actually available: `serve_layout`
    # itself doesn't run as part of handling *this* request at all in
    # this Dash version (confirmed directly against this app's real
    # url_map) — it runs later, from within a separate `/_dash-layout`
    # XHR dash-renderer's own JS fires once page bootstrap finishes,
    # which carries no query string of its own to read.
    if not _is_dash_internal_request(flask_request.path):
        bundle.slots["_pending_url_search"] = flask_request.query_string.decode("utf-8", "ignore")
    flask_g._pyaps_session_token = token


def _weaveor_redirect_response():
    """A small, self-contained interstitial page shown instead of the
    normal (empty, useless without a dataset) explorer shell -- see
    `_require_weaveor_entry` below for exactly when this fires. Explicit
    request: a direct, tokenless visit to this deployment should send
    people back to WeaveOR with a clear explanation, not silently render
    a page with nothing on it. A meta-refresh does the actual redirect
    (works with JS disabled too); the button is there for anyone who
    doesn't want to wait the 5 seconds out."""
    # Where to send visitors who arrive without a valid handoff token. No built-in
    # default: unset means the page just explains, without a redirect/button.
    weaveor_url = os.environ.get("PYAPS_EXPLORER_WEAVEOR_URL", "")
    _refresh = f'<meta http-equiv="refresh" content="5;url={weaveor_url}"/>' if weaveor_url else ""
    _button = f'<a class="btn" href="{weaveor_url}">Go to WeaveOR</a>' if weaveor_url else ""
    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8"/>
{_refresh}
<title>WEAVE Data Explorer</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; background: #f8fafb; color: #1a2733; display: flex; align-items: center; justify-content: center; min-height: 100vh; margin: 0; }}
.card {{ background: #fff; border-radius: 12px; box-shadow: 0 4px 12px rgba(26,77,86,0.12); border: 1px solid #d4e4e8; max-width: 480px; padding: 40px; text-align: center; }}
h1 {{ color: #1a4d56; font-size: 20px; margin: 0 0 14px 0; }}
p {{ font-size: 14px; line-height: 1.6; color: #4a5f70; }}
a.btn {{ display: inline-block; margin-top: 20px; padding: 12px 28px; background: linear-gradient(135deg, #1a4d56 0%, #286571 100%); color: #fff; text-decoration: none; border-radius: 8px; font-weight: 700; }}
</style>
</head><body>
<div class="card">
<h1>This page isn't accessible directly</h1>
<p>The WEAVE Data Explorer can only be opened from a WeaveOR query results page.</p>
{_button}
</div>
</body></html>"""
    return html, 200, {"Content-Type": "text/html; charset=utf-8"}


@app.server.before_request
def _require_weaveor_entry():
    """Only gates the app's own base page load -- not assets, not
    Dash's own internal XHR endpoints (those never even get requested
    by a real browser if the base page itself was redirected away, but
    excluding them here too keeps this narrowly scoped to exactly the
    one request type that matters). Registered *after* `_bind_session`
    above so `_sess.current_bundle()` already resolves the right
    per-session bundle in MULTI_SESSION mode before this checks it.

    A no-op entirely unless REQUIRE_WEAVEOR_AUTH is genuinely on --
    same "zero behavior change unless already opted in" shape as every
    other REQUIRE_WEAVEOR_AUTH-gated check in this file."""
    if not _auth.REQUIRE_WEAVEOR_AUTH:
        return
    if flask_request.path != _sess.URL_PREFIX:
        return
    if flask_request.args.get("token"):
        return  # a real handoff in progress -- let handle_url_handoff verify it
    bundle = _sess.current_bundle()
    if _auth.current_auth(bundle) is not None:
        return  # already-authorized session (e.g. reloading after a real handoff)
    return _weaveor_redirect_response()


@app.server.teardown_request
def _unbind_session(exc=None):
    token = getattr(flask_g, "_pyaps_session_token", None)
    if token is not None:
        _sess.unbind_request(token)


@app.server.after_request
def _set_session_cookie(response):
    sid = getattr(flask_g, "_pyaps_session_id", None)
    if sid is not None:
        response.set_cookie(
            _sess.SESSION_COOKIE_NAME, sid, max_age=_sess.SESSION_TTL_SECONDS,
            httponly=True, samesite="Lax",
        )
    return response


@app.server.route("/healthz")
def _healthz():
    """Container liveness/readiness probe target — confirms the process
    is actually serving requests, not just that it started. Deliberately
    outside Dash's own callback machinery (a plain Flask view) so it
    stays cheap and independent of any dataset being loaded."""
    return {"status": "ok"}, 200


@app.server.after_request
def _no_cache(response):
    """Given how much trouble this session has had with genuine server-
    side fixes not appearing to take effect in the browser, remove
    caching as a possible confound entirely rather than keep guessing
    about it: no response from this app — the initial page included —
    should ever be served from a cached copy. A plain SSH port-forward
    has no caching proxy in the middle, so any staleness has to be the
    browser's own HTTP cache, and this header set eliminates that
    unconditionally, on every request/reload."""
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    return response
app.layout = serve_layout


@app.callback(Output("log-panel", "children"), Input("log-poll", "n_intervals"))
def update_log_panel(n_intervals):
    return "\n".join(LOG.snapshot())


@app.callback(
    Output("log-panel", "children", allow_duplicate=True),
    Input("log-clear", "n_clicks"),
    prevent_initial_call=True,
)
def on_log_clear(n_clicks):
    LOG.clear()
    return ""


_last_js_error_msg = None


@app.callback(
    Output("js-error-dummy", "children"),
    Input("js-error-sink", "data"),
    prevent_initial_call=True,
)
def on_js_error(entry):
    """Feeds the console.error/window.onerror interceptor installed in
    index_string's <head> into the same Log panel every server-side
    print()/traceback already goes to — see that script's own comment for
    why this exists at all. Consecutive-identical-message dedup because a
    repeating client-side failure (e.g. the same ChunkLoadError firing on
    every retry) would otherwise flood the log with dozens of copies of the
    exact same line — still logs the first occurrence, just not every
    repeat of it."""
    global _last_js_error_msg
    if not entry or not isinstance(entry, dict):
        return no_update
    msg = entry.get("msg", "")
    if not msg or msg == _last_js_error_msg:
        return no_update
    _last_js_error_msg = msg
    LOG.add(" [browser]", f"[{entry.get('kind', '?')}] {msg}")
    return no_update


@app.callback(
    Output("idle-kill-status", "children"),
    Input("idle-kill-trigger", "data"),
    prevent_initial_call=True,
)
def handle_idle_kill(_data):
    """Fired once by index_string's own client-side inactivity timer
    (MULTI_SESSION-only — see that script's own comment, and
    aps_explorer_session.IDLE_TIMEOUT_MINUTES) once a tab has gone
    genuinely unused. Evicts this session's own SessionBundle right away
    rather than waiting for the much longer SESSION_TTL_SECONDS passive
    sweep — the actual memory reclaim a busy multi-user deployment needs,
    not just a UI nicety."""
    sid = getattr(flask_g, "_pyaps_session_id", None)
    if sid:
        _sess.end_session(sid)
        print(f">>> Session {sid[:8]}... ended after "
              f"{_sess.IDLE_TIMEOUT_MINUTES} min of inactivity.")
    return ""


app.clientside_callback(
    """
    function(target, showDssValue) {
        var div = document.getElementById("aladin-div");
        if (!div) { return ""; }

        // Explicit request: "in all coordinate in 2d and 3d plots, can
        // you disable the click to centre behaviour... right now in 2d
        // aladin or 3d when I double click on a point, it make[s] it
        // cent[]re of the field which is annoying." Confirmed directly
        // by reading Aladin Lite's own (unminified-enough-to-read)
        // bundled JS rather than guessing/trusting the public API
        // reference (which is silent on this): Aladin binds its own
        // `dblclick` listener straight onto its `catalogCanvas` element
        // via a plain, non-capturing `addEventListener` (`l.on = (el,
        // types, fn) => el.addEventListener(type, fn)` — bubble phase,
        // confirmed by reading that helper's own definition), whose
        // handler calls `pointTo()`/`setRotation(0)` — a genuine
        // recentre-and-reset-rotation on every double-click, exactly
        // matching the reported behaviour, and with no documented public
        // option to disable it. Since it's an ordinary bubble-phase
        // listener on a *descendant* of this div, a `dblclick` listener
        // registered here in the *capture* phase (the `true` third
        // argument below) runs first and, by calling `stopPropagation()`,
        // prevents the event from ever reaching `catalogCanvas` at all —
        // Aladin's own handler never fires. Guarded on the DOM node
        // itself (this callback re-runs on every target change) so it's
        // only ever attached once.
        if (!div._pyapsDblclickGuarded) {
            div._pyapsDblclickGuarded = true;
            div.addEventListener("dblclick", function(e) {
                e.stopPropagation();
                e.preventDefault();
            }, true);
        }

        // Always the single source of truth for "should DSS be visible
        // right now" — read from here (not a closure-captured value) so a
        // retry scheduled by an *earlier* invocation (see applyDssVisibility
        // below) still picks up the latest desired state if the checkbox
        // changed again before that retry fired.
        window._aladinDesiredDssVisible = showDssValue && showDssValue.indexOf("show") !== -1;

        function applyDssVisibility(attemptsLeft) {
            if (attemptsLeft === undefined) { attemptsLeft = 20; }
            if (!window._aladinInstance || !window._aladinInstance.getBaseImageLayer) { return; }
            var base = window._aladinInstance.getBaseImageLayer();
            if (base && base.setOpacity) {
                base.setOpacity(window._aladinDesiredDssVisible ? 1 : 0);
            } else if (attemptsLeft > 0) {
                // The base image layer's survey metadata loads
                // asynchronously (an HTTP fetch to the HiPS server's
                // properties file) — immediately after A.aladin() returns,
                // getBaseImageLayer() can still return something not yet
                // usable. Retry rather than silently leaving DSS at
                // Aladin's own default (visible) until the next unrelated
                // target/checkbox change happens to re-trigger this — this
                // is exactly why "hidden by default" wasn't taking effect
                // even though toggling the checkbox later worked fine.
                setTimeout(function() { applyDssVisibility(attemptsLeft - 1); }, 200);
            }
        }

        if (!target || target.ra === undefined || target.ra === null) {
            applyDssVisibility();
            return "";
        }

        // Always the single source of truth for "where should Aladin be
        // pointed right now" — every path below (already ready, still
        // loading, first-ever call) reads from here rather than closing
        // over this particular invocation's `target`, so a target update
        // that arrives while the script is still loading is never lost
        // and never overwritten by a stale one.
        window._aladinPendingTarget = target;

        function goToPending() {
            var t = window._aladinPendingTarget;
            if (!t || !window._aladinInstance) { return; }
            window._aladinInstance.gotoRaDec(t.ra, t.dec);
            // Deliberately NOT calling setFov() here on every recentre —
            // a fresh target arrives on every selection/colour-range
            // change with `fov` sized to the *whole dataset*, and
            // unconditionally re-applying it discarded any zoom the user
            // had set by scrolling/zooming inside Aladin itself.
            // setFov() is only ever called once, at instance creation
            // (see createAndGo below), to establish a sensible initial
            // view.
            applyDssVisibility();
        }

        // Aladin Lite reads the container's *current* pixel box when it
        // is first created and does not reliably notice a size that
        // wasn't actually laid out yet (e.g. a percentage width that
        // hadn't resolved against its parent column) — creating the
        // instance against that would produce a wrongly-sized view. Once
        // created, Aladin Lite's own view *does* auto-adapt to later
        // container resizes (per its docs), so this retry-until-laid-out
        // dance is only needed for the one-time creation step, not for
        // every subsequent target update. Sizing itself is pure CSS now
        // (aladin-div's own width:100%/height:_MAP_HEIGHT, set
        // server-side — see _ALADIN_DIV_STYLE) since there's no longer a
        // separate Plotly map to measure/match.
        function createAndGo(attemptsLeft) {
            // Guard against this function ever running more than once.
            // Three separate async paths can each eventually call it (a
            // requestAnimationFrame retry loop waiting for layout, a
            // script.onload -> A.init.then() chain, and a setInterval
            // poller for an already-in-flight script load) — and this
            // whole clientside_callback can genuinely re-fire more than
            // once before the very first call finishes: its Input,
            // aladin-target, changes on both selected-item and dataset-
            // version, which commonly both change within a single load
            // (see handle_l1_load/_route_and_load_l2), so a second
            // invocation landing here while the first is still mid-async
            // is a real, not theoretical, race — confirmed as the most
            // plausible mechanism after two independent live reports
            // ("Aladin background goes white after several clicks" and
            // "the Slit Explorer sometimes doesn't pick up the new NSPEC
            // after clicking a target in Aladin") that this file's
            // earlier defensive try/catch mitigation didn't resolve.
            // Without this guard, a second call would create a *second*
            // A.aladin() instance/WebGL context fighting the first one
            // over the same <div> and register a *second* objectClicked
            // listener — a real, concrete explanation for both a broken
            // canvas and inconsistent click routing (whichever instance
            // happens to still be attached handles the click, not
            // necessarily the one every other part of the page still
            // thinks is "the" instance).
            if (window._aladinInstance) {
                goToPending();
                return;
            }
            var rect = div.getBoundingClientRect();
            if (rect.width < 5 || rect.height < 5) {
                if (attemptsLeft > 0) {
                    window.requestAnimationFrame(function() { createAndGo(attemptsLeft - 1); });
                }
                return;
            }
            var t = window._aladinPendingTarget;
            // ICRS matches WEAVE's data throughout (CNAME encoding,
            // aps_utils.py's own SkyCoord(frame="icrs") usage) —
            // "equatorial" is not a valid cooFrame value at all. SIN
            // (Aladin's own default) is the standard zenithal/gnomonic
            // projection for a small pointed field like every WEAVE map
            // here, as opposed to an all-sky projection (AIT/Mollweide
            // etc).
            window._aladinInstance = A.aladin('#aladin-div', {
                target: t.ra + ' ' + t.dec,
                fov: t.fov || 0.1,
                survey: 'P/DSS2/color',
                cooFrame: 'ICRS',
                projection: 'SIN',
                showCooGridControl: true,
                // Disabled per explicit user request ("disable aladin
                // maximize... it is destroying the page structure and
                // some text boxes are overlaid and some behind"):
                // Aladin's own fullscreen control apparently escapes this
                // div's position:relative/zIndex containment (that
                // containment only holds position:absolute descendants —
                // Aladin's fullscreen toggle plausibly reflows the page
                // itself rather than staying inside an absolutely
                // positioned box), so disabling the control outright at
                // the source is far more robust than trying to keep
                // containing whatever it does next.
                showFullscreenControl: false,
            });
            window._aladinInstance.on('objectClicked', function(obj) {
                if (obj && obj.data && obj.data.item !== undefined) {
                    window.dash_clientside.set_props(
                        "selected-item", {data: obj.data.item}
                    );
                }
            });
            if (window._aladinPendingCatalog) {
                window._aladinApplyCatalog(window._aladinPendingCatalog);
            }
            if (window._contribPendingData && window._aladinApplyContribOverlay) {
                window._aladinApplyContribOverlay(window._contribPendingData);
            }
            applyDssVisibility();
        }

        if (window._aladinInstance) {
            goToPending();
            return "";
        }
        if (window._aladinReady) {
            createAndGo(30);
            return "";
        }
        if (!window._aladinScriptLoading) {
            window._aladinScriptLoading = true;
            var script = document.createElement('script');
            // Self-hosted (assets/vendor/aladin.js — see its own README
            // for version/license/why), not the external
            // aladin.cds.unistra.fr CDN this used to load straight from.
            // Explicit report: "Could not load the sky-map library from
            // aladin.cds.unistra.fr — check your network connection...
            // it may be blocking this external site." A real, reported
            // failure mode, not hypothetical: some institutional VPNs
            // block or reroute traffic to that external CDN, and every
            // single 2D map on the page depends on reaching it before
            // any of this even runs. A relative URL (not one hardcoded
            // to a particular url_base_pathname) — the current page is
            // always already being served at exactly the right prefix,
            // so "assets/vendor/aladin.js" resolves correctly against it
            // in any deployment (server-mode /weave/ prefix, standalone
            // with none, or a future different one) with no Python-side
            // string-building needed.
            //
            // Note this only fixes the *library* failing to load — the
            // actual DSS/HiPS background imagery and catalog lookups
            // Aladin Lite performs at runtime are still separate live
            // requests to CDS's own servers regardless (see the vendor
            // README's own "not fixed by this" section); a VPN blocking
            // *those* specifically would still show a blank/broken sky
            // background even though the map's own UI now loads fine.
            script.src = 'assets/vendor/aladin.js';
            script.onload = function() {
                A.init.then(function() {
                    window._aladinReady = true;
                    createAndGo(30);
                });
            };
            // Kept even though this is now a same-origin, always-
            // reachable request in practice — a deployment/caching
            // hiccup on our own server is still possible, and silently
            // hanging forever with zero user-facing sign anything's
            // wrong (window._aladinScriptLoading never reset) is worse
            // than a clear, resettable-for-retry error either way.
            script.onerror = function() {
                window._aladinScriptLoading = false;
                var div = document.getElementById('aladin-div');
                if (div) {
                    div.textContent = 'Could not load the sky-map library — ' +
                        'try reloading the page. If this keeps happening, ' +
                        'contact your PyAPS administrator.';
                    div.style.padding = '20px';
                    div.style.color = 'var(--pyaps-error)';
                    div.style.fontSize = '13px';
                }
            };
            document.head.appendChild(script);
        } else {
            // Script load already kicked off by an earlier invocation —
            // wait for it the same way, rather than doing nothing until
            // the *next* target update happens to fire.
            var waitForReady = setInterval(function() {
                if (window._aladinReady) {
                    clearInterval(waitForReady);
                    createAndGo(30);
                }
            }, 100);
        }
        return "";
    }
    """,
    Output("aladin-lite-dummy", "children"),
    Input("aladin-target", "data"),
    Input("aladin-show-dss", "value"),
)


app.clientside_callback(
    """
    function(catalogData) {
        if (!catalogData || !catalogData.buckets) {
            return "";
        }

        // Same "always read the latest pending value, never a stale
        // closure" pattern as aladin-target's goToPending — a catalog
        // update that arrives before Aladin has finished loading must
        // not be lost.
        window._aladinPendingCatalog = catalogData;

        window._aladinApplyCatalog = function(data) {
            if (!window._aladinInstance || !window.A) { return; }
            // Every Aladin call below is wrapped in try/catch and logged
            // via console.error (which index_string's own interceptor
            // already forwards into the Log panel — see that script's own
            // comment) rather than left to throw silently. Added directly
            // in response to a real report ("double or triple click on a
            // fibre in true-size mode sometimes turns the Aladin
            // background white") that couldn't be reproduced or root-
            // caused without browser access — this at minimum turns any
            // future recurrence into a visible, diagnosable Log entry
            // instead of a silent blank canvas.
            try {
                if (window._aladinCatalogLayers) {
                    window._aladinCatalogLayers.forEach(function(cat) {
                        window._aladinInstance.removeLayer(cat);
                    });
                }
            } catch (e) { console.error('PyAPS: removing old catalog layers failed: ' + e); }
            window._aladinCatalogLayers = [];
            // Separate from the catalog layers above — see the
            // radius_deg branch below for why these exist as their own
            // thing rather than just resizing the catalog markers.
            try {
                if (window._aladinOverlayLayers) {
                    window._aladinOverlayLayers.forEach(function(ov) {
                        window._aladinInstance.removeLayer(ov);
                    });
                }
            } catch (e) { console.error('PyAPS: removing old true-size overlay layers failed: ' + e); }
            window._aladinOverlayLayers = [];

            // "True fibre/spaxel size" mode (item: "a circle with the
            // size of the fibre width, so I can see exactly how mapping
            // is fibre on the sky"). Confirmed against Aladin Lite v3's
            // own API reference before writing this, not guessed:
            // A.catalog's own `sourceSize` is explicitly documented as
            // "size in pixels" — a fixed *screen* size with no sky-
            // angular equivalent — so real angular sizing has to go
            // through the separate A.graphicOverlay()/A.circle(ra, dec,
            // radiusDeg) shape-overlay API instead (the docs' own
            // explicitly recommended workaround for per-source real-size
            // shapes, not a workaround invented here). That API's Circle
            // objects, in turn, have no documented way to attach a
            // custom data payload the way A.source(ra, dec, {item:...})
            // does (confirmed by reading the Circle shape's own
            // constructor) — so circles alone can't drive click-to-
            // select, and (confirmed too) Circle/overlay shapes carry
            // their *own* built-in select()/hover() interactivity that
            // can't be turned off via any documented option. Solution:
            // keep the normal A.catalog/A.source markers as the *only*
            // click target, at their normal full size in both modes
            // (deliberately NOT shrunk in true-size mode — a smaller
            // click target only raises the odds of a near-miss landing on
            // the circle's own undocumented-and-unremovable interactivity
            // instead, plausibly related to the white-background report),
            // and layer real-angular-size A.circle() shapes purely for
            // the visual footprint underneath/around them.
            var trueSizeDeg = data.radius_deg;

            // One A.catalog() per colour bucket (see _bucket_by_color's
            // docstring for why) — each catalog's sources all share that
            // bucket's single colour, set catalog-wide via the confirmed
            // `color` option, rather than relying on an unconfirmed
            // per-source colour option.
            data.buckets.forEach(function(bucket) {
                try {
                    var cat = A.catalog({
                        name: 'PyAPS', sourceSize: 8, shape: 'circle', color: bucket.color,
                    });
                    var sources = bucket.points.map(function(p) {
                        return A.source(p.ra, p.dec, {item: p.item});
                    });
                    cat.addSources(sources);
                    window._aladinInstance.addCatalog(cat);
                    window._aladinCatalogLayers.push(cat);
                } catch (e) { console.error('PyAPS: building catalog layer failed: ' + e); }

                if (trueSizeDeg) {
                    try {
                        var overlay = A.graphicOverlay({color: bucket.color, lineWidth: 1});
                        window._aladinInstance.addOverlay(overlay);
                        var circles = bucket.points.map(function(p) {
                            return A.circle(p.ra, p.dec, trueSizeDeg);
                        });
                        overlay.addFootprints(circles);
                        window._aladinOverlayLayers.push(overlay);
                    } catch (e) { console.error('PyAPS: building true-size overlay failed: ' + e); }
                }
            });
        };

        if (window._aladinReady && window._aladinInstance) {
            window._aladinApplyCatalog(catalogData);
        }
        return "";
    }
    """,
    Output("aladin-catalog-dummy", "children"),
    Input("aladin-catalog-data", "data"),
)


app.clientside_callback(
    """
    function(data) {
        window._contribPendingData = data;
        // window._contribFilesData: {file: [overlay, overlay, ...]} — every
        // known file's own real angular-size A.circle/A.graphicOverlay
        // shapes, built once here and never torn down/rebuilt again just
        // because a per-file checkbox was ticked/unticked (see below and
        // _contrib_exposures_payload's own docstring for why a full
        // rebuild-on-every-toggle design was replaced: a live JS-timing
        // profile showed the rebuild alone costing 1-2s and *growing* on
        // each successive toggle — a real, apparently cumulative cost in
        // repeatedly tearing down and recreating thousands of Aladin
        // shapes, not something worth paying more than once per genuine
        // data change).
        window._contribFilesData = window._contribFilesData || {};

        window._contribApplyVisibility = function() {
            // null/undefined (nothing has told us otherwise yet, e.g. the
            // very first draw before the checklist's own clientside
            // callback has had a chance to fire) means "show everything" —
            // matches the checklist's own default `value` (every file
            // checked) on a fresh build.
            var checked = window._contribCheckedFiles;
            Object.keys(window._contribFilesData).forEach(function(file) {
                var show = (checked == null) || checked.indexOf(file) !== -1;
                try { show ? window._contribFilesData[file].show() : window._contribFilesData[file].hide(); }
                catch (e) { /* overlay not yet fully initialised */ }
            });
        };

        window._aladinApplyContribOverlay = function(payload) {
            if (!window._aladinInstance || !window.A) { return; }
            // Same try/catch + console.error-per-step pattern as
            // window._aladinApplyCatalog above, for the same reason —
            // any future failure here becomes a diagnosable Log-panel
            // entry instead of a silent blank/broken overlay. This
            // teardown only ever runs on a genuine data change (fresh
            // load, colour-by change, or the main on/off checkbox) — see
            // the separate, pure-visibility-toggle callback below for
            // per-file show/hide, which never reaches this function at all.
            try {
                Object.keys(window._contribFilesData).forEach(function(file) {
                    window._aladinInstance.removeLayer(window._contribFilesData[file]);
                });
            } catch (e) { console.error('PyAPS: clearing old contributing-exposure layers failed: ' + e); }
            window._contribFilesData = {};
            window._contribCheckedFiles = null;  // back to "everything" until the checklist says otherwise

            if (!payload || !payload.files) { return; }

            // Real angular-size A.circle/A.graphicOverlay shapes (this
            // feature's own established design — see
            // _contrib_exposures_payload's docstring for the two earlier,
            // explicitly-rejected alternatives), one A.graphicOverlay per
            // *file* (confirmed live that A.circle's own per-shape
            // `color` option works, so each file's whole point set —
            // spanning however many colour buckets — fits in a single
            // overlay object, not one per file-and-bucket pair) — colour
            // still matches the main catalog's own flux/S-N colouring
            // exactly (explicit request: "the color code [should] come
            // again from the total flux or S/N or whatever the main plot
            // is"), file identity now determines only *grouping* (so a
            // whole file's shapes can be shown/hidden together via one
            // show()/hide() call), never appearance.
            payload.files.forEach(function(fileEntry) {
                try {
                    var overlay = A.graphicOverlay({lineWidth: 1});
                    window._aladinInstance.addOverlay(overlay);
                    var circles = fileEntry.points.map(function(p) {
                        return A.circle(p.ra, p.dec, payload.radius_deg, {color: p.color});
                    });
                    overlay.addFootprints(circles);
                    window._contribFilesData[fileEntry.file] = overlay;
                } catch (e) { console.error('PyAPS: building contributing-exposure overlay failed: ' + e); }
            });
            window._contribApplyVisibility();
        };

        if (window._aladinReady && window._aladinInstance) {
            window._aladinApplyContribOverlay(data);
        }
        return "";
    }
    """,
    Output("contrib-fibre-dummy", "children"),
    Input("contrib-fibre-data", "data"),
)


app.clientside_callback(
    """
    function(checked_files) {
        // Purely client-side — explicit request: "I want to have the
        // option to select and deselect each contributed single file in
        // the plot... so I can manage it by myself." No server round-trip
        // at all: every file's own overlay shapes already exist (built
        // once by the callback above), so showing/hiding one is just a
        // show()/hide() call on objects Aladin already has — see
        // _contrib_exposures_payload's own docstring for why this is
        // *correct*, not just faster (colour-bucket boundaries never
        // actually depended on which files were selected).
        window._contribCheckedFiles = checked_files || [];
        if (window._contribApplyVisibility) { window._contribApplyVisibility(); }
        return "";
    }
    """,
    Output("contrib-legend-toggle-dummy", "children"),
    Input("contrib-legend-checklist", "value"),
)


app.clientside_callback(
    """
    function(children) {
        var el = document.getElementById("log-panel");
        if (!el) { return ""; }
        // Stick to the bottom on new content only while the user hasn't
        // scrolled up to read earlier lines — an unconditional
        // scrollTop=scrollHeight on every poll made it impossible to
        // scroll up at all, since the next poll (every 700ms) would yank
        // it straight back down.
        if (!window._logPanelScrollBound) {
            window._logPanelStickToBottom = true;
            el.addEventListener("scroll", function() {
                var threshold = 30;
                window._logPanelStickToBottom =
                    (el.scrollTop + el.clientHeight >= el.scrollHeight - threshold);
            });
            window._logPanelScrollBound = true;
        }
        if (window._logPanelStickToBottom) {
            el.scrollTop = el.scrollHeight;
        }
        return "";
    }
    """,
    Output("log-scroll-dummy", "children"),
    Input("log-panel", "children"),
)


app.clientside_callback(
    """
    function(flags) {
        // Visual counterpart to load_from_form_fields' server-side
        // override: previously "raw" forced the other flags off in the
        // actually-applied Namespace but left their checkboxes visibly
        // checked, which is exactly what made it look like raw mode
        // wasn't doing anything (e.g. the flux y-axis still said
        // "erg/s/cm2/A" while a checked "Sensitivity correction" box sat
        // right there implying it was still on).
        if (!flags || flags.indexOf("raw") === -1) {
            return window.dash_clientside.no_update;
        }
        var allowed = flags.filter(function(f) { return f === "raw"; });
        if (allowed.length === flags.length) {
            return window.dash_clientside.no_update;
        }
        return allowed;
    }
    """,
    Output("lf-flags", "value"),
    Input("lf-flags", "value"),
)


# --------------------------------------------------------------------------- #
# Load / routing
# --------------------------------------------------------------------------- #

@app.callback(
    Output("load-panel", "style", allow_duplicate=True),
    Input("close-load-panel", "n_clicks"),
    prevent_initial_call=True,
)
def on_close_load_panel(n_clicks):
    """Close button pinned inside the sidebar itself."""
    return _load_panel_style(False)


@app.callback(
    Output("load-panel", "style", allow_duplicate=True),
    Input("open-load-panel-btn", "n_clicks"),
    prevent_initial_call=True,
)
def on_open_load_panel(n_clicks):
    """The one button that opens the sidebar, always present regardless of
    whether a dataset is loaded — now living in _toolbar(), see its own
    docstring for why it's `position: sticky` rather than `fixed`."""
    return _load_panel_style(True)


app.clientside_callback(
    """
    function(panel_style) {
        // Keeps #load-panel-backdrop in sync with load-panel's real
        // open/closed state by watching the same style Dash already
        // sets on every open/close path — including the several
        // *business-logic*-driven ones (auto-close on a successful
        // load, handle_url_handoff, etc.) that never touch the open/
        // close buttons at all. This is the only reason this can be one
        // small clientside callback instead of editing every one of
        // those existing server callbacks to also drive the backdrop.
        // Mirrors _load_panel_backdrop_style() in aps_explorer.py —
        // keep the two in sync if either ever changes.
        var open = panel_style && panel_style.transform &&
                   panel_style.transform.indexOf("-100%") === -1;
        // `top` is measured live (banner + toolbar's actual rendered
        // bottom edge) rather than a hardcoded pixel value — confirmed
        // live via Playwright that a full-viewport (`inset: 0`) backdrop
        // sits, by z-index, on top of the entire #app-banner (including
        // the theme toggle), making it unclickable any time the drawer
        // is open. Falls back to the same estimate the SSR-only initial
        // render uses if either element isn't found for some reason,
        // rather than raising.
        var toolbar = document.getElementById("app-toolbar");
        var top = toolbar ? Math.round(toolbar.getBoundingClientRect().bottom) : 190;
        return {
            position: "fixed", top: top + "px", left: "0", right: "0", bottom: "0",
            backgroundColor: "rgba(0,0,0,0.35)",
            zIndex: 9690,
            opacity: open ? "1" : "0",
            pointerEvents: open ? "auto" : "none",
            transition: "opacity .28s ease",
        };
    }
    """,
    Output("load-panel-backdrop", "style"),
    Input("load-panel", "style"),
)


# --------------------------------------------------------------------------- #
# Path browser callbacks — see _path_browser_modal's own comment for why
# this exists (server-side paths, browser-side native file dialogs would
# pick from the wrong machine entirely).
# --------------------------------------------------------------------------- #

@app.callback(
    Output("path-browser-modal", "style", allow_duplicate=True),
    Output("path-browser-state", "data", allow_duplicate=True),
    Output("path-browser-hint", "children"),
    Input({"type": "path-browse-btn", "target": ALL, "mode": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def open_path_browser(n_clicks_list):
    triggered = ctx.triggered_id
    if not triggered or not any(n_clicks_list):
        return no_update, no_update, no_update
    target, mode = triggered["target"], triggered["mode"]
    hint = ("Pick a directory" if mode == "dir" else
            "Pick a file, or navigate into a directory" if mode == "file" else
            "Pick a file to append to the list")
    hint += f" — target field: {target}"
    return (_path_browser_modal_style(True),
            {"target": target, "mode": mode, "cwd": _PATH_BROWSER_DEFAULT_ROOT}, hint)


@app.callback(
    Output("path-browser-modal", "style", allow_duplicate=True),
    Input("path-browser-close", "n_clicks"),
    Input("path-browser-cancel", "n_clicks"),
    prevent_initial_call=True,
)
def close_path_browser(close_clicks, cancel_clicks):
    return _path_browser_modal_style(False)


@app.callback(
    Output("path-browser-state", "data", allow_duplicate=True),
    Input("path-browser-up", "n_clicks"),
    Input("path-browser-path-input", "value"),
    Input({"type": "path-browser-entry", "path": ALL, "kind": "dir"}, "n_clicks"),
    State("path-browser-state", "data"),
    prevent_initial_call=True,
)
def navigate_path_browser(up_clicks, typed_path, entry_clicks, state):
    """Any of: clicking "Up", typing+Enter in the path box, or clicking a
    folder row — all just change `cwd` and let the listing-render callback
    below pick it up. File clicks are handled separately (they commit a
    selection, not a navigation)."""
    if state is None:
        return no_update
    trigger = ctx.triggered_id
    cwd = state.get("cwd", _PATH_BROWSER_DEFAULT_ROOT)
    if trigger == "path-browser-up":
        cwd = os.path.dirname(cwd.rstrip("/")) or "/"
    elif trigger == "path-browser-path-input":
        if typed_path and os.path.isdir(typed_path):
            cwd = typed_path
        else:
            return no_update
    elif isinstance(trigger, dict) and trigger.get("type") == "path-browser-entry":
        if not any(entry_clicks):
            return no_update
        cwd = trigger["path"]
    else:
        return no_update
    return {**state, "cwd": cwd}


@app.callback(
    Output("path-browser-listing", "children"),
    Output("path-browser-path-input", "value"),
    Input("path-browser-state", "data"),
)
def render_path_browser_listing(state):
    if not state:
        return no_update, no_update
    cwd = state.get("cwd", _PATH_BROWSER_DEFAULT_ROOT)
    return _path_browser_listing(cwd, state.get("mode", "dir")), cwd


@app.callback(
    Output("path-browser-result", "data"),
    Output("path-browser-modal", "style", allow_duplicate=True),
    Input({"type": "path-browser-entry", "path": ALL, "kind": "file"}, "n_clicks"),
    Input("path-browser-use-dir", "n_clicks"),
    State("path-browser-state", "data"),
    prevent_initial_call=True,
)
def select_path(file_clicks, use_dir_clicks, state):
    """The actual "commit a selection" step — either a file row was
    clicked (mode != "dir") or "Use this directory" was clicked (any
    mode). Writes {target, mode, value} to path-browser-result, which the
    commit_path_browser_result clientside callback below turns into the
    real field write (has to be clientside since the target field id is
    only known at runtime, not something a normal Dash Output can point
    at)."""
    if state is None:
        return no_update, no_update
    trigger = ctx.triggered_id
    if trigger == "path-browser-use-dir":
        if not use_dir_clicks:
            return no_update, no_update
        value = state.get("cwd", _PATH_BROWSER_DEFAULT_ROOT)
    elif isinstance(trigger, dict) and trigger.get("type") == "path-browser-entry":
        if not any(file_clicks):
            return no_update, no_update
        value = trigger["path"]
    else:
        return no_update, no_update
    return {"target": state["target"], "mode": state["mode"], "value": value}, _path_browser_modal_style(False)


app.clientside_callback(
    """
    function(result) {
        if (!result || !result.target || !result.value) { return ""; }
        if (window.dash_clientside && window.dash_clientside.set_props) {
            if (result.mode === "file-append") {
                // lf-infiles is a multi-line Textarea (one file per line) —
                // append rather than replace, so browsing for a second/third
                // arm file doesn't erase the first one already typed in.
                var el = document.getElementById(result.target);
                var current = el ? el.value : "";
                var next = current && current.trim() ? current.replace(/\\n$/, "") + "\\n" + result.value : result.value;
                window.dash_clientside.set_props(result.target, {value: next});
            } else {
                window.dash_clientside.set_props(result.target, {value: result.value});
            }
        }
        return "";
    }
    """,
    Output("path-browser-commit-dummy", "children"),
    Input("path-browser-result", "data"),
    prevent_initial_call=True,
)


@app.callback(
    Output("load-panel", "style"),
    Output("main-panel", "style"),
    Output("selected-item", "data", allow_duplicate=True),
    Output("dataset-version", "data"),
    Output("lf-status", "children"),
    Input("lf-run", "n_clicks"),
    State("lf-l2-fullfile", "value"), State("lf-outpath", "value"), State("lf-headname", "value"),
    State("dataset-version", "data"),
    prevent_initial_call=True,
)
def handle_l2_load(run_clicks, fullfile, outpath, headname, version):
    if not run_clicks:
        # Real bug found live via Playwright while chasing "the drawer
        # reopens by itself partway through a load": handle_url_handoff's
        # own success return replaces load-panel's entire `children` with
        # a freshly-rendered _load_form() (see that callback's own
        # docstring for why -- keeping the identity text/locked state
        # current). That remount recreates lf-run/lf-run-l1/
        # close-load-panel as brand-new DOM nodes, and Dash re-fires
        # *their* own prevent_initial_call=True callbacks anyway the
        # moment a component's Input first exists in the DOM, regardless
        # of the app's own overall initial-page-load guard -- a known
        # Dash quirk, confirmed directly via the browser's own network
        # trace (a phantom POST with changedPropIds=['lf-run.n_clicks']
        # firing mid-load, run_clicks=0, every field empty/default, no
        # click ever happened). Before this feature's own reopen-on-
        # failure fix (see the "missing outpath/headname" branch below),
        # that phantom firing's "missing required field" no_update return
        # was invisible/harmless; now it would visibly reopen the drawer
        # for no real reason. run_clicks=0 (its declared default, only
        # ever nonzero after a genuine click) is exactly what distinguishes
        # a real click from this phantom remount-triggered one.
        return no_update, no_update, no_update, no_update, no_update
    bundle = _sess.current_bundle()
    auth = _auth.current_auth(bundle)
    if auth is not None and auth.get("kind") != "l2":
        # Same session (pyaps_sid cookie), but currently recorded for a
        # *different* kind of dataset -- confirmed live: browsers share
        # one cookie across every tab of the same origin, so opening an
        # L1 link and an L2 link from weaveOR in two different tabs
        # (both target="_blank") lands both in the *same* session; the
        # one opened later overwrites _AUTH_SLOT (apply_token/
        # record_dataset_loaded both "replace", not accumulate — see
        # their own docstrings), so the *other*, now-stale tab's own
        # Load/Reload button would otherwise try to unpack this kind's
        # authorized_dataset tuple as if it were its own shape and crash
        # (confirmed live: "too many values to unpack" -- l2's 3-tuple
        # read as l1's 2-tuple). A real, disclosed rejection instead.
        #
        # _load_panel_style(True) here (and on every other failure return
        # in this function), not no_update: the clientside callback that
        # closes load-panel the instant "Load / Reload" is clicked (see
        # its own comment) has already done so by the time this runs, so
        # no_update would leave the panel closed with this message never
        # actually seen. main-panel is still left as no_update -- same
        # established behaviour as reopening the panel via the "☰
        # Dataset & Settings" button while a dataset is already showing.
        return (_load_panel_style(True), no_update, no_update, no_update,
                "This session is now showing a different dataset (opened in "
                "another tab) — please reopen this one via its own link.")
    if auth is not None:
        # Locked (a dataset is already recorded for this session — via a
        # real weaveOR token or a server-mode deep link): the locked L2
        # tab never populates lf-l2-fullfile/lf-outpath/lf-headname with
        # the real path, so whatever came back from the browser here is
        # meaningless by design — always reload the session's own
        # already-authorized dataset instead of trusting the submission.
        _, outpath, headname = auth["authorized_dataset"]
        fullfile = None
    elif fullfile:
        suffix = "_APS.fits"
        if not fullfile.endswith(suffix):
            return _load_panel_style(True), no_update, no_update, no_update, f'Full L2 filename must end in "{suffix}".'
        outpath, headname = os.path.split(fullfile[: -len(suffix)])
    if not outpath or not headname:
        return (_load_panel_style(True), no_update, no_update, no_update,
                "Either a full L2 filename, or both outpath and headname, are required.")
    # No-op unless REQUIRE_WEAVEOR_AUTH or MULTI_SESSION is on — see
    # aps_explorer_auth.check_load_authorized's own docstring. Given the
    # override above, this now only ever rejects the *first* load
    # attempt of a MULTI_SESSION session that never went through a
    # deep link at all (auth is None) — every subsequent reload
    # necessarily matches, by construction.
    auth_error = _auth.check_load_authorized(bundle, "l2", outpath=outpath, headname=headname)
    if auth_error:
        return _load_panel_style(True), no_update, no_update, no_update, auth_error
    try:
        initial_item = _route_and_load_l2(outpath, headname)
    except Exception as e:
        print(f">>> Load failed: {e}")
        traceback.print_exc()
        return _load_panel_style(True), no_update, no_update, no_update, f"Load failed: {e}"
    return _load_panel_style(False), _main_panel_style(True), initial_item, (version or 0) + 1, ""


@app.callback(
    Output("load-panel", "style", allow_duplicate=True),
    Output("main-panel", "style", allow_duplicate=True),
    Output("selected-item", "data", allow_duplicate=True),
    Output("dataset-version", "data", allow_duplicate=True),
    Output("lf-status-l1", "children"),
    Input("lf-run-l1", "n_clicks"),
    State("lf-infiles", "value"), State("lf-infiles-list", "value"),
    State("lf-l1ref", "value"), State("lf-l2ref", "value"),
    State("lf-apsids", "value"), State("lf-targsrvy", "value"), State("lf-targclass", "value"),
    State("lf-maskids", "value"), State("lf-decimate-stride", "value"),
    State("lf-area", "value"), State("lf-mask-areas", "value"),
    State("lf-wlranges", "value"), State("lf-arms-ratio", "value"),
    State("lf-caldir", "value"), State("lf-catdir", "value"), State("lf-configdir", "value"),
    State("lf-flags", "value"),
    State("lf-flags-advanced", "value"), State("lf-lsftype", "value"),
    State("lf-ivar-norm-mode", "value"), State("lf-edge-pixels", "value"),
    State("lf-gap-offset-pix", "value"), State("lf-funit", "value"),
    State("lf-template-sigma0", "value"),
    State("dataset-version", "data"),
    prevent_initial_call=True,
)
def handle_l1_load(run_clicks, infiles_text, infiles_list, l1ref, l2ref, apsids, targsrvy, targclass, maskids,
                    decimate_stride, area, mask_areas, wlranges, arms_ratio, caldir, catdir, configdir, flags,
                    advanced_flags, lsftype, ivar_norm_mode, edge_pixels, gap_offset_pix, funit,
                    template_sigma0, version):
    if not run_clicks:
        # Same phantom-remount guard as handle_l2_load's own -- see its
        # comment for the full story (a successful weaveOR handoff
        # remounts lf-run-l1 fresh via load-panel's own children
        # replacement, and Dash re-fires this callback with
        # run_clicks=0 and every field empty/default even though nobody
        # clicked anything).
        return no_update, no_update, no_update, no_update, no_update
    bundle = _sess.current_bundle()
    auth = _auth.current_auth(bundle)
    if auth is not None and auth.get("kind") != "l1":
        # Same session (pyaps_sid cookie), but currently recorded for a
        # *different* kind of dataset -- see handle_l2_load's own
        # matching comment for the full story (browsers share one
        # cookie across every same-origin tab, so opening L1 and L2
        # links from weaveOR in two tabs, both target="_blank", lands
        # both in the same session; whichever loaded later overwrites
        # _AUTH_SLOT). A real, disclosed rejection instead of crashing
        # trying to unpack the other kind's authorized_dataset shape.
        #
        # _load_panel_style(True) here (and on every other failure return
        # in this function), not no_update -- see handle_l2_load's own
        # matching comment for why: the clientside callback that closes
        # load-panel the instant "Load / Reload" is clicked has already
        # done so by the time this runs.
        return _load_panel_style(True), no_update, no_update, no_update, (
            "This session is now showing a different dataset (opened in "
            "another tab) — please reopen this one via its own link.")
    if auth is not None:
        # Locked (a dataset is already recorded for this session — via a
        # real weaveOR token or a server-mode deep link): the locked
        # form (aps_l1_preview._load_form's own `locked` branch) never
        # populates lf-infiles/lf-infiles-list with the real path, so
        # whatever came back from the browser here is meaningless by
        # design — always reload the session's own already-authorized
        # files instead of trusting the submission. --infiles_list is
        # never valid once locked either way (a *file listing* infiles
        # can't be matched against a recorded infiles tuple).
        _, infiles_tuple = auth["authorized_dataset"]
        infiles_text = "\n".join(infiles_tuple)
        infiles_list = None
    print(f">>> Loading L1 dataset: {infiles_text or infiles_list}")
    # No-op unless REQUIRE_WEAVEOR_AUTH or MULTI_SESSION is on — see
    # aps_explorer_auth.check_load_authorized's own docstring. Given the
    # override above, this now only ever rejects the *first* load
    # attempt of a MULTI_SESSION session that never went through a
    # deep link at all (auth is None) — every subsequent reload
    # necessarily matches, by construction.
    infiles_for_auth = [f.strip() for f in (infiles_text or "").splitlines() if f.strip()]
    auth_error = _auth.check_load_authorized(bundle, "l1", infiles=infiles_for_auth)
    if auth_error:
        return _load_panel_style(True), no_update, no_update, no_update, auth_error
    # Same deployment-wide fallback as handle_url_handoff's own L1
    # branches -- an empty caldir/catdir here (a genuinely blank field,
    # or the very first load reusing values a token never provided)
    # shouldn't silently lose LSF/FWHM diagnostics when a sensible
    # default is configured for this deployment.
    caldir = caldir or _sess.DEFAULT_CALDIR
    catdir = catdir or _sess.DEFAULT_CATDIR
    try:
        l1_mod.load_from_form_fields(infiles_text, infiles_list, l1ref, l2ref, apsids, targsrvy, targclass,
                                      maskids, area, mask_areas, wlranges, arms_ratio, caldir, catdir,
                                      configdir, flags, advanced_flags=advanced_flags, lsftype=lsftype,
                                      ivar_normalization_mode=ivar_norm_mode, edge_pixels_to_mask=edge_pixels,
                                      offset_gap_pix=gap_offset_pix, funit=funit,
                                      template_sigma0_angstrom=template_sigma0,
                                      decimate_stride=decimate_stride)
        EXPLORER.kind = "l1"
        EXPLORER.file_info = None
        initial_item = _central_item("l1")
    except Exception as e:
        print(f">>> Load failed: {e}")
        traceback.print_exc()
        return _load_panel_style(True), no_update, no_update, no_update, f"Load failed: {e}"
    print(">>> Load complete.")
    return _load_panel_style(False), _main_panel_style(True), initial_item, (version or 0) + 1, ""


# global-load-overlay show/hide. L1 and L2 have their own separate
# "Load / Reload" buttons (lf-run-l1 / lf-run) since dcc.Tabs only ever
# mounts the *active* tab's children — confirmed live that the other
# tab's button genuinely does not exist in the DOM at all while its
# tab isn't selected. Two separate SHOW callbacks, one per button, are
# required for exactly the same reason this app already avoids
# multi-Input callbacks that mix a permanently-present id with a
# conditionally-mounted one elsewhere: a clientside callback whose
# Input list includes a currently-absent component id does not fire at
# all, for *any* of its inputs, not just the missing one.
#
# HIDE only needs one trigger though: dataset-version (a plain
# dcc.Store, always present) is bumped by both handle_l1_load and
# handle_l2_load on a genuine successful load, regardless of which tab
# was used. A failed load does *not* bump it (a real edge case since
# lf-status/lf-status-l1 have exactly the same conditional-mount
# problem as the buttons do), so each SHOW call also arms a plain JS
# timeout as a safety net — generation-numbered (matching this file's
# own window._pyaps*-flag convention elsewhere) so an *earlier* click's
# timeout can never hide the overlay for a *later*, still-genuinely-
# loading click.
#
# Also closes #load-panel itself, instantly, on the same click — real
# user complaint: "when I load a new data, [the settings panel is] open
# ... if the load take[s] a few seconds I see that panel instead of the
# log [panel]." load-panel's own z-index (9700) sits *above* the log
# panel's (9600) and #global-load-overlay's (9500, see that element's
# own CSS comment for why the log panel is deliberately kept visible
# above the overlay already) — so a still-open drawer silently defeated
# that existing "watch progress in the log panel while a slow load
# runs" design the whole time it stayed open, which is the whole
# (potentially many-second) duration of a real load. Handled here,
# clientside, rather than in handle_l1_load/handle_l2_load themselves,
# for the same reason the overlay itself is: those are slow, synchronous
# server callbacks, so nothing they return can appear before the load
# they're doing finishes — only a clientside callback reacting to the
# same click gives *instant* feedback.
#
# The style object below is a literal copy of
# _load_panel_style(False)'s own output (_LOAD_PANEL_BASE_STYLE plus
# that function's "not visible" branch) -- keep the two in sync if
# either ever changes, same caution already noted on the
# load-panel-backdrop-sync clientside callback just below this one.
# Every *failure* path in handle_l1_load/handle_l2_load explicitly
# reopens the panel (via _load_panel_style(True), replacing what used
# to be a bare no_update there) specifically so this instant close can
# never swallow an error message the user needs to see.
_CLOSED_LOAD_PANEL_STYLE_JS = """{
        position: "fixed", top: 0, left: 0, height: "100vh", width: "920px",
        maxWidth: "95vw", overflowY: "auto", backgroundColor: "var(--pyaps-paper-raised)",
        boxShadow: "var(--pyaps-shadow-drawer)",
        transition: "transform .3s cubic-bezier(.22,.9,.32,1), visibility 0s linear .3s",
        zIndex: 9700, transform: "translateX(-100%)", visibility: "hidden", pointerEvents: "none",
    }"""

app.clientside_callback(
    """
    function(n_clicks) {
        window._pyapsLoadOverlayGen = (window._pyapsLoadOverlayGen || 0) + 1;
        var myGen = window._pyapsLoadOverlayGen;
        var overlay = document.getElementById("global-load-overlay");
        if (overlay) { overlay.style.display = "flex"; }
        setTimeout(function() {
            if (window._pyapsLoadOverlayGen !== myGen) { return; }
            var el = document.getElementById("global-load-overlay");
            if (el) { el.style.display = "none"; }
        }, 30000);
        return ["", """ + _CLOSED_LOAD_PANEL_STYLE_JS + """];
    }
    """,
    Output("global-load-overlay-show-l2-dummy", "children"),
    Output("load-panel", "style", allow_duplicate=True),
    Input("lf-run", "n_clicks"),
    prevent_initial_call=True,
)


app.clientside_callback(
    """
    function(n_clicks) {
        window._pyapsLoadOverlayGen = (window._pyapsLoadOverlayGen || 0) + 1;
        var myGen = window._pyapsLoadOverlayGen;
        var overlay = document.getElementById("global-load-overlay");
        if (overlay) { overlay.style.display = "flex"; }
        setTimeout(function() {
            if (window._pyapsLoadOverlayGen !== myGen) { return; }
            var el = document.getElementById("global-load-overlay");
            if (el) { el.style.display = "none"; }
        }, 30000);
        return ["", """ + _CLOSED_LOAD_PANEL_STYLE_JS + """];
    }
    """,
    Output("global-load-overlay-show-l1-dummy", "children"),
    Output("load-panel", "style", allow_duplicate=True),
    Input("lf-run-l1", "n_clicks"),
    prevent_initial_call=True,
)


app.clientside_callback(
    """
    function(version) {
        window._pyapsLoadOverlayGen = (window._pyapsLoadOverlayGen || 0) + 1;
        var overlay = document.getElementById("global-load-overlay");
        if (overlay) { overlay.style.display = "none"; }
        return "";
    }
    """,
    Output("global-load-overlay-hide-dummy", "children"),
    Input("dataset-version", "data"),
    prevent_initial_call=True,
)


@app.callback(
    Output("load-panel", "style", allow_duplicate=True),
    Output("main-panel", "style", allow_duplicate=True),
    Output("selected-item", "data", allow_duplicate=True),
    Output("dataset-version", "data", allow_duplicate=True),
    Output("url", "search"),
    Output("toolbar-identity", "children", allow_duplicate=True),
    Output("load-panel", "children", allow_duplicate=True),
    Input("url", "search"),
    State("dataset-version", "data"),
    # 'initial_duplicate', not plain False: Dash requires this exact
    # value (not prevent_initial_call=False) whenever a callback both
    # fires on initial page load *and* shares an allow_duplicate=True
    # Output with another callback (confirmed live — plain False raises
    # dash.exceptions.DuplicateCallback at import time).
    prevent_initial_call="initial_duplicate",
)
def handle_url_handoff(search, version):
    """Consumes `?token=...` (a weaveOR handoff — see
    aps_explorer_auth.py's own module docstring for the full design) or,
    when --require-weaveor-auth is off, a plain `?kind=l1&infiles=...` /
    `?kind=l2&outpath=...&headname=...` deep link. Fires once, on the
    very first page load (prevent_initial_call=False is deliberate — a
    token in the URL is only ever meaningful the instant the page
    opens, unlike every other callback in this file which reacts to a
    genuine user click). Reuses the exact same load functions the
    sidebar form itself calls (_route_and_load_l2/_load_l1) — one load
    path, not two. Clears the query string once handled (its own
    "url.search" Output) so a page refresh never re-attempts to verify
    an already-consumed, possibly now-expired token — the session's own
    stored authorization (aps_explorer_auth.apply_token, on the current
    SessionBundle) is what persists across a refresh, not the URL.

    Also writes "toolbar-identity" and "load-panel"'s own children
    directly on a successful handoff — both _toolbar() and _load_form()
    render once, server-side, *before* this callback ever runs (a client
    round-trip, not a fresh page load), so without this the identity text
    and the load panel's now-should-be-locked contents would only ever
    show one page-load stale (see _identity_text()'s own docstring, and
    _load_form()'s own docstring for the "blocked"/"locked" states this
    is fixing the staleness of)."""
    if not search:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update

    params = parse_qs(search.lstrip("?"))
    token = (params.get("token") or [None])[0]

    if token:
        try:
            claims = _auth.verify_token(token)
        except _auth.TokenError as e:
            print(f">>> WeaveOR handoff rejected: {e}")
            # _load_panel_style(True), not no_update: a real load was
            # attempted (a token was present) and it failed -- reopen the
            # form rather than leaving the page's now-closed-by-default
            # drawer (see serve_layout's own comment) with no visible
            # sign anything was even tried. Every other genuine-attempt
            # failure below in this function does the same; the two
            # "there was nothing to attempt in the first place" early
            # returns above/below (no search string at all; a query
            # string present but REQUIRE_WEAVEOR_AUTH with no token)
            # deliberately do not, and stay closed.
            return _load_panel_style(True), no_update, no_update, no_update, "", no_update, no_update
        bundle = _sess.current_bundle()
        _auth.apply_token(bundle, claims)
        # At most one live session per identified weaveOR user (see
        # aps_explorer_session.claim_session_for_user's own docstring for
        # why this is needed even though same-tab reloads already share
        # one cookie) -- evicts any *other* session already open for
        # this same user before proceeding with this one.
        this_sid = getattr(flask_g, "_pyaps_session_id", None)
        if this_sid:
            _sess.claim_session_for_user(claims.get("user"), this_sid)
        kind = claims.get("kind")
        try:
            if kind == "l2":
                initial_item = _route_and_load_l2(claims["outpath"], claims["headname"])
            elif kind == "l1":
                l1_args = l1_mod._build_arg_parser().parse_args([])
                l1_args.infiles = claims.get("infiles")
                # weaveOR's own token has no clean source for caldir/catdir
                # today (see weave/lib/aps_explorer_handoff.py's own
                # docstring) -- fall back to this deployment's own archive
                # root (PYAPS_EXPLORER_DEFAULT_CALDIR/_CATDIR) rather than
                # silently loading with no LSF/FWHM diagnostics at all.
                l1_args.caldir = claims.get("caldir") or _sess.DEFAULT_CALDIR
                l1_args.catdir = claims.get("catdir") or _sess.DEFAULT_CATDIR
                _load_l1(l1_args)
                initial_item = _central_item("l1")
                # L1-only defense-in-depth (see aps_explorer_auth.py's own
                # docstring for why there's no L2-IFU equivalent): confirm
                # the file's own survey codes are really a subset of what
                # the token claims, via the same fast catalog-only read
                # l1_fileinfo's own load sequence already does internally.
                found = l1_fileinfo(l1_args.infiles, catdir=l1_args.catdir, caldir=l1_args.caldir).get("srvys")
                err = _auth.verify_l1_surveys_allowed(found, claims.get("allowed_surveys"))
                if err:
                    print(f">>> WeaveOR handoff rejected after load: {err}")
                    return _load_panel_style(True), no_update, no_update, no_update, "", no_update, no_update
            else:
                raise ValueError(f"unknown kind {kind!r} in handoff token")
        except Exception as e:
            print(f">>> Handoff load failed: {e}")
            traceback.print_exc()
            return _load_panel_style(True), no_update, no_update, no_update, "", no_update, no_update
        print(">>> WeaveOR handoff load complete.")
        return (_load_panel_style(False), _main_panel_style(True), initial_item, (version or 0) + 1, "",
                _identity_text(bundle), _load_form())

    if _auth.REQUIRE_WEAVEOR_AUTH:
        # No token, and unauthenticated direct access is disabled —
        # leave the query string alone (nothing to consume) and load
        # nothing; the sidebar form's own access-denied messaging (via
        # check_load_authorized) covers the rest of this session's visit.
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update

    kind = (params.get("kind") or [None])[0]
    if kind == "l2":
        outpath = (params.get("outpath") or [None])[0]
        headname = (params.get("headname") or [None])[0]
        if not outpath or not headname:
            # kind=l2 was explicit but malformed (missing a required
            # param) -- a real attempt, so reopen (see the token
            # branch's own comment on this distinction).
            return _load_panel_style(True), no_update, no_update, no_update, no_update, no_update, no_update
        try:
            initial_item = _route_and_load_l2(outpath, headname)
        except Exception as e:
            print(f">>> Deep-link load failed: {e}")
            traceback.print_exc()
            return _load_panel_style(True), no_update, no_update, no_update, "", no_update, no_update
        # Records this dataset as "what this session is showing", the
        # same as a real weaveOR token would (apply_token above) — see
        # aps_explorer_auth.check_load_authorized's MULTI_SESSION branch:
        # once recorded, the sidebar form may only reload *this* dataset
        # with different params, never switch to a different one. A
        # cheap no-op when MULTI_SESSION is off (nothing reads it then).
        _auth.record_dataset_loaded(_sess.current_bundle(), "l2", outpath=outpath, headname=headname)
        return (_load_panel_style(False), _main_panel_style(True), initial_item, (version or 0) + 1, "",
                no_update, _load_form())

    if kind == "l1":
        infiles = params.get("infiles")
        if not infiles:
            # kind=l1 was explicit but malformed (no infiles) -- a real
            # attempt, so reopen (see the token branch's own comment).
            return _load_panel_style(True), no_update, no_update, no_update, no_update, no_update, no_update
        l1_args = l1_mod._build_arg_parser().parse_args([])
        l1_args.infiles = infiles
        # See the token branch above for why this fallback exists.
        l1_args.caldir = (params.get("caldir") or [None])[0] or _sess.DEFAULT_CALDIR
        l1_args.catdir = (params.get("catdir") or [None])[0] or _sess.DEFAULT_CATDIR
        try:
            _load_l1(l1_args)
        except Exception as e:
            print(f">>> Deep-link load failed: {e}")
            traceback.print_exc()
            return _load_panel_style(True), no_update, no_update, no_update, "", no_update, no_update
        initial_item = _central_item("l1")
        # See the L2 branch above for why this is recorded unconditionally.
        _auth.record_dataset_loaded(_sess.current_bundle(), "l1", infiles=infiles)
        return (_load_panel_style(False), _main_panel_style(True), initial_item, (version or 0) + 1, "",
                no_update, _load_form())

    return no_update, no_update, no_update, no_update, no_update, no_update, no_update


@app.callback(
    Output("file-info-panel", "children"),
    Output("settings-panel", "children"),
    Output("value-tables-panel", "children"),
    Output("aladin-color-range-container", "children"),
    Input("dataset-version", "data"),
)
def refresh_shell(version):
    """Rebuilds every kind-dependent panel wholesale on dataset (re)load —
    switching between an L1, IFU, or MOS dataset via "Load different
    dataset" changes not just which maps/tabs apply but the entire
    settings-row and value-table DOM shape, so nothing here can be a
    partial update."""
    if not EXPLORER.loaded():
        return no_update, no_update, no_update, no_update
    return (_file_info_panel(), _settings_panel_children(), _value_tables_children(),
            _aladin_color_range_control())


@app.callback(
    Output("tabs", "children"),
    Output("tabs", "value"),
    Input("selected-item", "data"),
    Input("dataset-version", "data"),
    State("tabs", "value"),
)
def refresh_tabs(selected_item, version, current_tab):
    """Input on selected-item (not just dataset-version) because MOS-kind
    tabs are per-target (Redrock always; Stellar/Galaxy vary target to
    target) — IFU- and L1-kind tabs only vary per-dataset, so their
    _available_tabs()-equivalents just ignore selected_item; see
    aps_MOSviewer._available_tabs for why the MOS side needs it.

    Keeps whatever tab the user is currently on if it's still valid in the
    rebuilt list — dataset-version also bumps for in-place settings changes
    (maptype/AoN/marker-colour/coord-source) that have nothing to do with
    tab selection, so unconditionally resetting to tabs[0] here would (and
    did) clobber a manual tab click, e.g. jumping straight back to
    "Spectrum" out from under a click on "Processing History"."""
    if not EXPLORER.loaded():
        return no_update, no_update
    tabs = _tabs_for_current(selected_item)
    values = [t.value for t in tabs]
    new_value = current_tab if current_tab in values else tabs[0].value
    return tabs, new_value


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #

@app.callback(
    Output("dataset-version", "data", allow_duplicate=True),
    Input("maptype-dd", "value"),
    State("dataset-version", "data"),
    prevent_initial_call=True,
)
def on_ifu_maptype_change(maptype, version):
    """The no-op guard (skip if unchanged) matters more than it looks:
    dcc.Dropdown's Input fires once when the settings panel is first
    inserted into the DOM after a load (a Dash quirk — prevent_initial_call
    only suppresses *app*-startup firing, not a dynamically-added
    component's first mount), always with maptype == STATE.current_maptype
    already. Without this guard that spurious firing bumps dataset-version
    for no real reason, which — since dataset-version also drives
    refresh_tabs — could clobber a just-made manual tab selection."""
    if EXPLORER.kind != "ifu" or not maptype or maptype == ifu_mod.STATE.current_maptype:
        return no_update
    ifu_mod.STATE.current_maptype = maptype
    return (version or 0) + 1


@app.callback(
    Output("dataset-version", "data", allow_duplicate=True),
    Input("maptype-category-dd", "value"),
    State("dataset-version", "data"),
    prevent_initial_call=True,
)
def on_ifu_maptype_category_change(category, version):
    """Switching category picks that category's first quantity as the new
    current_maptype and bumps dataset-version, which rebuilds the settings
    panel (showing the new category's own, much shorter, Map dropdown) —
    same no-op-guard pattern as on_ifu_maptype_change above."""
    if EXPLORER.kind != "ifu" or not category:
        return no_update
    current_category = ifu_mod._maptype_category_for(ifu_mod.STATE.data, ifu_mod.STATE.current_maptype)
    if category == current_category:
        return no_update
    opts = dict(ifu_mod._list_maptypes(ifu_mod.STATE.data)).get(category)
    if not opts:
        return no_update
    ifu_mod.STATE.current_maptype = opts[0][0]
    return (version or 0) + 1


@app.callback(
    Output("selected-item", "data", allow_duplicate=True),
    Input("bin-id-go", "n_clicks"),
    State("bin-id-input", "value"),
    prevent_initial_call=True,
)
def on_ifu_id_entry(n_clicks, bin_id):
    if EXPLORER.kind != "ifu" or bin_id is None:
        return no_update
    return int(bin_id)


@app.callback(
    Output("selected-item", "data", allow_duplicate=True),
    Input("target-id-go", "n_clicks"),
    State("target-id-input", "value"),
    prevent_initial_call=True,
)
def on_mos_id_entry(n_clicks, aps_id):
    if EXPLORER.kind != "mos" or aps_id is None:
        return no_update
    return int(aps_id)


# --------------------------------------------------------------------------- #
# Tab content + value tables — reuse each library's own callback logic
# directly (undecorated, plain functions).
# --------------------------------------------------------------------------- #

@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value"),
    Input("selected-item", "data"),
    Input("dataset-version", "data"),
)
def update_tab_content(tab, selected_item, version):
    if not EXPLORER.loaded() or selected_item is None:
        return html.Div("No selection.")
    try:
        if EXPLORER.kind == "ifu":
            return ifu_mod.update_tab_content(tab, selected_item, version, locked=_is_locked())
        if EXPLORER.kind == "mos":
            return mos_mod.update_tab_content(tab, selected_item, version)
        return l1_mod.update_tab_content(tab, selected_item, locked=_is_locked())
    except Exception as e:
        # An uncaught exception here leaves this Output at whatever it
        # last successfully rendered (Dash doesn't clear it) — clicking a
        # tab would then look like nothing happened, or like it silently
        # switched back to a previous tab's content, rather than showing
        # what actually went wrong.
        print(f">>> Error rendering tab {tab!r} for item {selected_item!r}: {e}")
        traceback.print_exc()
        return html.Div(f"Error rendering this tab: {e}", style={"color": "crimson"})


@app.callback(
    Output("spaxel-values-table", "data"),
    Output("spaxel-values-table", "columns"),
    Output("bin-values-table", "data"),
    Output("bin-values-title", "children"),
    Input("selected-item", "data"),
    Input("dataset-version", "data"),
)
def update_ifu_value_tables(selected_item, version):
    if EXPLORER.kind != "ifu":
        return no_update, no_update, no_update, no_update
    return ifu_mod.update_value_tables(selected_item, version)


@app.callback(
    Output("class-values-table", "data"),
    Output("class-values-table", "columns"),
    Output("star-values-table", "data"),
    Output("galaxy-values-table", "data"),
    Input("selected-item", "data"),
    Input("dataset-version", "data"),
)
def update_mos_value_tables(selected_item, version):
    if EXPLORER.kind != "mos":
        return no_update, no_update, no_update, no_update
    return mos_mod.update_value_tables(selected_item, version)


# --------------------------------------------------------------------------- #
# Kind-specific settings — reuse each library's own callback logic directly.
# --------------------------------------------------------------------------- #

    # coord-source-toggle / on_mos_coord_source_change removed — MOS
    # loading now uses L1 FIBTABLE coordinates automatically whenever
    # available (see aps_MOSviewer._load_mos_fits), per explicit
    # request to drop the manual toggle in favour of just doing the
    # better thing by default.


@app.callback(
    Output("dataset-version", "data", allow_duplicate=True),
    Input("mos-color-by", "value"),
    State("dataset-version", "data"),
    prevent_initial_call=True,
)
def on_mos_color_by_change(color_by, version):
    """No-op guard for the same reason as on_ifu_maptype_change above.
    Resets any previously-typed colour range on every switch — a Min/Max
    left over from "redshift" would silently miscolour "S/N" (different
    units, different sane range) if carried across. Bumping dataset-version
    (rather than writing aladin-catalog-data directly) is deliberate: this
    also drives refresh_shell, which is what makes the Min/Max/Reset
    control itself appear/disappear as color_by switches to/from
    "availability"."""
    if EXPLORER.kind != "mos" or not color_by or color_by == mos_mod.STATE.color_by:
        return no_update
    mos_mod.STATE.color_by = color_by
    mos_mod.STATE.color_vmin = None
    mos_mod.STATE.color_vmax = None
    return (version or 0) + 1


@app.callback(
    Output("dataset-version", "data", allow_duplicate=True),
    Input("l1-color-by", "value"),
    State("dataset-version", "data"),
    prevent_initial_call=True,
)
def on_l1_color_by_change(color_by, version):
    """No-op guard for the same reason as on_ifu_maptype_change above."""
    if EXPLORER.kind != "l1" or not color_by or color_by == l1_mod.STATE.color_by:
        return no_update
    l1_mod.STATE.color_by = color_by
    l1_mod.STATE.color_vmin = None
    l1_mod.STATE.color_vmax = None
    return (version or 0) + 1


def _color_range_state():
    """The AppState that owns color_vmin/color_vmax for whichever kind is
    currently loaded — shared by IFU, L1 and (when its colour-by is a
    continuous quantity, not the fixed categorical "availability" swatches)
    MOS, which all reuse the same color-vmin/color-vmax/color-range-reset
    component ids (safe: their settings panels are never rendered at the
    same time, see _settings_panel_children). None for MOS-in-categorical-
    mode/nothing-loaded, where these components don't exist in the DOM at
    all."""
    if EXPLORER.kind == "ifu":
        return ifu_mod.STATE
    if EXPLORER.kind == "l1":
        return l1_mod.STATE
    if EXPLORER.kind == "mos" and mos_mod.STATE.color_by != "availability":
        return mos_mod.STATE
    return None


def _apply_color_range(vmin, vmax, reset):
    """Shared vmin/vmax bookkeeping for both the typed-value and Reset
    paths below — returns True if state.color_vmin/vmax actually changed
    (a no-op guard shared across the now-split single-output callbacks
    below, each of which calls this independently rather than relying on
    execution order between them)."""
    state = _color_range_state()
    if state is None:
        return None, False
    if reset:
        if state.color_vmin is None and state.color_vmax is None:
            return state, False
        state.color_vmin = None
        state.color_vmax = None
        return state, True
    new_vmin = float(vmin) if vmin is not None else None
    new_vmax = float(vmax) if vmax is not None else None
    if new_vmin == state.color_vmin and new_vmax == state.color_vmax:
        return state, False
    state.color_vmin = new_vmin
    state.color_vmax = new_vmax
    return state, True


@app.callback(
    Output("aladin-catalog-data", "data", allow_duplicate=True),
    Output("color-range-version", "data", allow_duplicate=True),
    Input("color-vmin", "value"),
    Input("color-vmax", "value"),
    State("color-range-version", "data"),
    prevent_initial_call=True,
)
def on_color_range_typed(vmin, vmax, crversion):
    state, changed = _apply_color_range(vmin, vmax, reset=False)
    if not changed or state is None:
        return no_update, no_update
    # Bumping color-range-version (not just aladin-catalog-data) is what
    # makes this also reach the 3D flux cube — see update_map_mode's own
    # docstring for why it can't take color-vmin/color-vmax as direct
    # Inputs itself.
    return _aladin_catalog_payload(), (crversion or 0) + 1


@app.callback(
    Output("aladin-catalog-data", "data", allow_duplicate=True),
    Output("color-vmin", "value"),
    Output("color-vmax", "value"),
    Output("color-range-version", "data", allow_duplicate=True),
    Input("color-range-reset", "n_clicks"),
    State("color-range-version", "data"),
    prevent_initial_call=True,
)
def on_color_range_reset(n_clicks, crversion):
    state, changed = _apply_color_range(None, None, reset=True)
    if not changed:
        return no_update, no_update, no_update, no_update
    return _aladin_catalog_payload(), None, None, (crversion or 0) + 1


@app.callback(
    Output("aladin-catalog-data", "data", allow_duplicate=True),
    Output("color-range-version", "data", allow_duplicate=True),
    Input("color-scale-dd", "value"),
    State("color-range-version", "data"),
    prevent_initial_call=True,
)
def on_color_scale_change(scale, crversion):
    """Linear/Log/Sqrt/Power/Asinh — the astronomical-image-viewing
    "stretch" options (item 8: "is it possible to have linear, log and
    power and most popular options... for the colour viewing?"; extended
    to the 3D flux cube on a later follow-up: "in 3d view can we have
    other scaling as we have in 2d like log, sinh, power etc?"). Same
    no-op guard as every other dynamically-mounted dropdown here (fires
    once on mount with its already-current value)."""
    state = _color_range_state()
    if state is None or not scale or scale == state.color_scale:
        return no_update, no_update
    state.color_scale = scale
    return _aladin_catalog_payload(), (crversion or 0) + 1


@app.callback(
    Output("aladin-catalog-data", "data", allow_duplicate=True),
    Output("color-range-version", "data", allow_duplicate=True),
    Input("color-palette-dd", "value"),
    State("color-range-version", "data"),
    prevent_initial_call=True,
)
def on_color_palette_change(palette, crversion):
    """Explicit request: "is it possible to instead of using the current
    color map, we use something that is black for low signal and white
    for high signal... Can I have it in all maps either 2D or 3D so we
    can switch between this blue to red to black to white?" One shared
    EXPLORER-level setting (not per-kind, unlike color_scale/color_vmin/
    color_vmax) — see _resolve_palette_override's own docstring for why
    a single setting is correct here (the user wants one switch that
    applies uniformly, not a per-kind preference). Bumps color-range-
    version the same way on_color_scale_change does, so the 3D flux cube
    picks up the change too — see update_map_mode's own docstring."""
    if not palette or palette == EXPLORER.color_palette:
        return no_update, no_update
    EXPLORER.color_palette = palette
    return _aladin_catalog_payload(), (crversion or 0) + 1


@app.callback(
    Output("color-range-version", "data", allow_duplicate=True),
    Input("cube-transparency-toggle", "value"),
    State("color-range-version", "data"),
    prevent_initial_call=True,
)
def on_cube_transparency_toggle_change(value, crversion):
    """"Transparency by signal (low signal fades)" — 3D-only, so unlike
    on_color_scale_change/on_color_palette_change this never touches
    `aladin-catalog-data` (the 2D catalog is unaffected either way), only
    bumps `color-range-version` so `update_map_mode` rebuilds the cube.
    Moved this round from a static child of `flux-cube-container` to
    living inside `_aladin_color_range_control()` (next to Scale/
    Palette) — explicit request: "Add the transparency by signal
    o[p]tion next to the col[o]r range setting below the col[o]r bar."
    Wired through the exact same `color-range-version`-bump indirection
    Scale/Palette already use for the same reason (this control now only
    exists in the DOM once a dataset is loaded, so it can no longer
    safely be a direct Input on the always-registered `update_map_mode` —
    see that callback's own docstring)."""
    new_value = "on" in (value or [])
    if new_value == EXPLORER.cube_transparency_on:
        return no_update
    EXPLORER.cube_transparency_on = new_value
    return (crversion or 0) + 1


# --------------------------------------------------------------------------- #
# Aladin Lite — the one and only spatial view (the Plotly coordinate/map
# figure was dropped entirely, per explicit user request: this panel
# already shows everything it did — a click-to-select catalog overlay
# coloured the same way the map was, plus optional DSS imagery — and then
# some). Always visible now, no Off/Separate-panel toggle.
#
# Deliberately one-way recentring only (no continuous linked zoom/pan): an
# earlier "Background" mode tried to keep a separate Plotly map's zoom
# continuously synced with Aladin's, but Aladin Lite v3 has no documented
# pan/zoom event to hook (only objectHovered/objectClicked), so that had to
# be a client-side poll comparing getRaDec()/getFov() against last-known
# values — proved unreliable in practice and was removed in favour of this
# simpler, robust design: selecting a point (a click in Aladin itself, an
# ID-entry box, or Aladin's own objectClicked handler writing straight into
# selected-item) recenters the view on that point's exact position, full
# stop — no ongoing zoom sync needed now there's nothing else to sync with.
# --------------------------------------------------------------------------- #

@app.callback(
    Output("aladin-target", "data"),
    Input("selected-item", "data"),
    Input("dataset-version", "data"),
)
def update_aladin_target(selected_item, version):
    if not EXPLORER.loaded():
        return no_update
    target = _aladin_target(selected_item)
    return target if target is not None else no_update


@app.callback(
    Output("aladin-catalog-data", "data"),
    Input("dataset-version", "data"),
)
def update_aladin_catalog(version):
    """Rebuilds the whole point set only when the data/colouring could
    have changed (dataset-version already covers a fresh load, a maptype
    switch, or a colour-range tweak) — not on every click, unlike
    aladin-target above."""
    if not EXPLORER.loaded():
        return no_update
    return _aladin_catalog_payload()


@app.callback(
    Output("contrib-exposures-row", "style"),
    Input("dataset-version", "data"),
)
def update_contrib_exposures_row_visibility(version):
    return _contrib_exposures_row_style()


@app.callback(
    Output("contrib-fibre-data", "data"),
    Input("dataset-version", "data"),
)
def update_contrib_fibre_data(version):
    """Mirrors update_aladin_catalog's own pattern exactly: recomputes on
    every dataset-version change (fresh load or "Load different
    dataset"), not just on the checkbox itself — `_contrib_exposures_
    payload()` already internally checks `EXPLORER.contrib_exposures_on`
    and no-ops (returns None) when it's off, so this is what makes a
    previously-checked toggle keep working after a reload without any
    special-cased persistence logic, the same free behaviour
    aladin-true-size already gets from update_aladin_catalog above."""
    if not EXPLORER.loaded():
        return no_update
    return _contrib_exposures_payload()


@app.callback(
    Output("contrib-exposures-note", "children"),
    Input("contrib-fibre-data", "data"),
)
def update_contrib_exposures_note(payload):
    return _contrib_exposures_note_text(payload)


@app.callback(
    Output("contrib-fibre-data", "data", allow_duplicate=True),
    Input("contrib-exposures-toggle", "value"),
    prevent_initial_call=True,
)
def on_contrib_exposures_toggle(value):
    """No-op guard for the usual dynamically-mounted-checkbox-fires-once
    reason (same as on_aladin_true_size_toggle just above)."""
    if EXPLORER.kind != "l1":
        return no_update
    new_value = "show" in (value or [])
    if new_value == EXPLORER.contrib_exposures_on:
        return no_update
    EXPLORER.contrib_exposures_on = new_value
    return _contrib_exposures_payload()

# Explicit request: "I want to have the option to select and deselect
# each contributed single file in the plot... a[checkbox] next to the
# name of each single one to be able to select or deselect that one...
# so I can manage it by myself." No server callback here any more — the
# per-file checklist toggle is now purely client-side (a show()/hide()
# call on that file's own already-built overlay objects, see the
# clientside_callback below and _contrib_exposures_payload's own
# docstring for why this is correct, not just faster: colour-bucket
# boundaries never actually depended on file selection in the first
# place).


@app.callback(
    Output("aladin-2d-controls", "style"),
    Output("flux-cube-container", "style"),
    Output("main-panel", "className"),
    Output("flux-cube-graph", "figure"),
    Output("flux-cube-rebuild-version", "data"),
    Input("map-mode", "value"),
    Input("dataset-version", "data"),
    Input("selected-item", "data"),
    Input("color-range-version", "data"),
    # Debounced, not flux-cube-bin-width's own live "value" — see
    # flux-cube-bin-width-debounced's own comment where it's declared for
    # why (rapid changes collapsing into one rebuild instead of one per
    # change).
    Input("flux-cube-bin-width-debounced", "data"),
    # A second, independent trigger for the exact same rebuild — see the
    # "Apply" button's own comment (right above where it's built) for why
    # this exists alongside flux-cube-bin-width's own Input above rather
    # than replacing it: that Input's debounce=True commit was confirmed
    # unreliable (a real, pre-existing Dash gap, not new). n_clicks
    # itself isn't read below — bin_width already comes from the Input
    # right above regardless of *which* of the two actually fired. Always
    # immediate, deliberately never debounced like the value Input above
    # is — clicking Apply is itself already a deliberate, low-frequency
    # "do it now" action, and the reliable-commit callback below forces
    # flux-cube-bin-width-debounced to skip its own delay the instant
    # Apply is clicked, so this never waits on the debounce either.
    Input("flux-cube-bin-width-apply", "n_clicks"),
    # Debounced, not flux-cube-depth-mode's own live "value" — same
    # reasoning as bin-width above.
    Input("flux-cube-depth-mode-debounced", "data"),
    State("flux-cube-camera-store", "data"),
    State("flux-cube-rebuild-version", "data"),
    prevent_initial_call=True,
)
def update_map_mode(mode, version, selected_item, color_range_version, bin_width,
                     _bin_width_apply_clicks, depth_mode, stored_camera, rebuild_version):
    """Show/hide the two spatial-view panels *and* (re)build the 3D
    figure in one callback, not two.

    This used to be two separate callbacks — one toggling
    aladin-2d-controls/flux-cube-container's `style`, the other building
    flux-cube-graph's `figure` — both triggered by the same `map-mode`
    Input but landing in the browser as two independent HTTP round-trips
    applied in two independent React commits. Found live (real Playwright
    runs, real IFU data, `window.Plotly.react`/`gd.data.length`
    introspection) that this is a genuine race, not a rare edge case:
    whichever commit happened to land first determined the outcome —
    if the figure arrived while flux-cube-container was still its
    initial `display:none`, react-plotly.js measures a 0x0 container,
    silently skips the actual Plotly.newPlot/react call, and never
    retries later (no ResizeObserver was ever attached to catch the
    container becoming visible afterwards, since rendering was skipped
    before that point) — leaving flux-cube-graph permanently stuck on
    its empty initial figure (confirmed via gd.data.length === 0 and
    gd.layout having no 'scene' key at all, i.e. still the dcc.Graph
    default, not our figure) even though the server-side callback that
    built it logged a perfectly good 3-trace, 144,000-point figure.
    Merging both into one callback makes Dash return both outputs in a
    single response, which dash-renderer applies in a single dispatch —
    the container's `style` and the graph's `figure` land in the same
    React commit, so by the time react-plotly.js measures the
    container it is already visible.

    `prevent_initial_call=True` (added after that fix, live-diagnosed a
    second, distinct race): Dash fires every callback once automatically
    at page mount using each Input's own initial value — for this one,
    `map-mode`'s default ("2d"). That mount-time firing and a later
    genuine user click (e.g. switching to "3d") could land at the browser
    *out of order* (confirmed live: inline `style` attributes read
    straight off the DOM showed `aladin-2d-controls` back to `display:
    block`/`flux-cube-container` back to `display: none` — the 2D-mode
    values — seconds *after* a real 3D click, even though the server log
    showed the 3D response had already been built and, moments earlier,
    correctly applied), the late-arriving stale 2D response silently
    overwriting the correct 3D one already on screen. Skipping the
    mount-time firing entirely removes the race rather than trying to
    win it: the static layout already bakes in the correct 2D-mode
    defaults with no callback needed (`flux-cube-container` is hardcoded
    `style={"display": "none"}` at construction, `aladin-2d-controls` has
    no style override at all, i.e. visible by default) — a genuinely new
    dataset load or mode switch still fires this callback completely
    normally afterward, only the redundant, race-prone very-first
    automatic firing is gone.

    Recomputed on a fresh load (dataset-version) and on every new
    selection (so the highlighted column always tracks whatever's
    currently selected, the same "always show the current selection"
    convention as the 2D map's own black-ring marker / Slit Explorer's
    dashed guide line) — not on every click *within* the 3D scene itself
    (orbiting/zooming doesn't touch these Inputs at all, so it stays
    purely a client-side Plotly.js interaction with no server
    round-trip). Never destroying/recreating aladin-2d-controls's own
    children on toggle — same established pattern as _load_panel_style/
    _main_panel_style (and _contrib_exposures_row_style just above) for
    exactly the same reason: replacing a component wholesale risks
    losing whatever state it was holding (here, most importantly,
    Aladin's own already-initialised JS instance living inside
    aladin-div — destroying that div would mean re-running the whole
    createAndGo() dance from scratch every time someone flips back to
    2D).

    Takes `color-range-version` (not `color-scale-dd`/`color-vmin`/
    `color-vmax` directly) so a Colour range/Scale change also rebuilds
    the 3D figure — explicit follow-up request: "in 3d view can we have
    other scaling as we have in 2d like log, sinh, power etc?" Those three
    components only exist in the DOM once a dataset is loaded (they're
    nested inside `_aladin_color_range_control()`'s conditional output),
    so taking them as direct Inputs here would risk this project's own
    documented "Input on a component that doesn't exist yet silently
    kills the *whole* callback" gotcha for every one of this callback's
    other, always-valid Inputs — `color-range-version` is a plain
    `dcc.Store` defined unconditionally in the base layout instead, bumped
    by `on_color_range_typed`/`_reset`/`on_color_scale_change` alongside
    their existing `aladin-catalog-data` output. `color-scale-dd`'s own
    palette override (`EXPLORER.color_palette`, via
    `_resolve_palette_override`) reaches the cube through that same
    `color-range-version` bump, applied here as `colorscale=`.

    `cube-transparency-toggle` used to be taken as a direct Input here
    (it used to live as a static child of `flux-cube-container` itself
    specifically so it could be, avoiding the gotcha above) — moved this
    round to sit next to Scale/Palette inside `_aladin_color_range_
    control()` instead, per explicit request: "Add the transparency by
    signal o[p]tion next to the col[o]r range setting below the col[o]r
    bar." Now that it lives inside that conditionally-mounted control
    too, it's wired exactly like Scale/Palette: its own small
    `on_cube_transparency_toggle_change` callback sets `EXPLORER.
    cube_transparency_on` and bumps `color-range-version`, and this
    callback reads that state back off `EXPLORER` rather than taking the
    checkbox as a direct Input. Explicit report motivating the toggle
    itself: "when I look at the cross section of 3d data cubes, I do not
    see what I usually see in 2d maps... probably due to transparency
    issue" — see flux_cube_figure's own `transparent` docstring.

    `flux-cube-bin-width` (Å) *is* taken as a direct Input, like
    `cube-transparency-toggle` originally was — it lives as a static
    child of `flux-cube-container` itself (always present in the DOM
    from page load, not conditional on a dataset being loaded), so it
    never risks the missing-component gotcha the trio above does.
    Explicit request: "add the width of the z axis bin in angstrom to
    the top of the 3d panel so users can set it[,] but give a default
    one." Deliberately `type="text"` in the layout, not `type="number"`,
    parsed manually here (`bin_width_angstrom`) rather than trusted as
    already-numeric — a real, confirmed Dash bug found by live testing
    (reproduced identically via three different real select-all-then-
    retype interaction methods, not a Playwright artifact): a *debounced*
    `dcc.Input(type="number")` can commit `None` on blur after its field
    is cleared and retyped, even though the actual DOM value at that
    moment is genuinely the newly-typed one — the browser's own native
    number-input semantics and Dash's number-specific prop handling seem
    to interact badly with an intermediate empty state mid-edit.
    `type="text"` sidesteps that code path entirely (a fix confirmed
    live: the identical select-all-then-retype sequence that reliably
    produced `None` under `type="number"` now correctly produces the
    typed value every time).

    `flux-cube-wave-min`/`-max` (Å) are deliberately NOT Inputs here, even
    though they used to be — explicit follow-up report: "every time I
    change the start and end, it reloads the whole 3d view... it should
    be a precomputed thing and behave the same way as we scroll through
    one axis without reloading the 3d object." The first version of this
    feature did wire them as direct Inputs (parsed the same defensively
    -optional way as `bin_width`, passed straight through to
    `flux_cube_figure`'s own `wave_min`/`wave_max`) — genuinely correct,
    but architecturally wrong for what was actually being asked: any
    change to either field triggered a full server round trip (rebuild
    + re-transmit + re-mount the whole cube), which reads exactly like
    "reloading," no matter how fast the rebuild itself is (and it *is*
    fast — profiled directly, well under half a second even for a wide
    native range on a real dataset). This callback now always builds the
    cube across the dataset's *full* native wavelength coverage (no
    `wave_min`/`wave_max` passed to `flux_cube_figure` at all, i.e. its
    own `None`/`None` default) exactly once per genuine rebuild (mode
    switch, new selection, colour/scale/bin-width change) — real range
    *scrolling* is a separate, purely client-side operation on that
    already-downloaded full cube (see the clientside callback right
    after this one), the same "no server round trip, just a local
    WebGL/DOM operation" category orbiting/zooming the camera already
    is. `wave_min`/`wave_max` stay real, useful parameters on
    `flux_cube_figure` itself (a genuine, still-used primitive) — just
    no longer invoked with anything but their own `None` defaults from
    this specific caller.

    `State("flux-cube-camera-store", "data")` (`stored_camera`) is how the
    user's current camera/zoom survives every one of this callback's own
    rebuilds instead of snapping back to the default top-down view on
    every single click — explicit report: "when I click on a map[[a
    point]] in 3d view, it reload[s] the image prob[a]bly because it
    want[s] to put that point in the centre... it still reload[s] the
    whole window every time I click to select a spaxel... very few times
    it tried to save the camera angle."

    Two things were tried before landing on this. (1) Plotly's own
    *documented* mechanism, `layout.uirevision` — confirmed (by reading
    `dcc.Graph`'s own async-graph.js bundle directly) NOT to work for this
    component in this app's installed Dash/Plotly.js combination
    (4.4.1/3.1.1): `dcc.Graph`'s `plotly_relayout` handler unconditionally
    writes the live camera back into its own `figure` prop on every
    interaction, corrupting `uirevision`'s own bookkeeping — the exact
    same gotcha this project already hit and fixed a different way for
    the (since-removed) 2D Plotly map. (2) Reading that same live camera
    back from `flux-cube-graph`'s own `figure` prop via `State` instead
    (bypassing `uirevision` but still relying on `dcc.Graph`'s own
    internal figure-prop-sync) — this *mostly* worked but not reliably
    ("very few times it tried to save... but still reload[ed]"), because
    it depends on `dcc.Graph`'s own async relayout→figure-prop write-back
    having already landed in the browser's local component-prop store
    *before* a separate, later click-triggered dispatch reads it — two
    independent pieces of Dash-internal bookkeeping this app doesn't
    control the relative timing of, a genuine (if usually narrow) race.

    Fixed for real by not depending on `dcc.Graph`'s own bookkeeping at
    all: a dedicated clientside `plotly_relayout` listener (see the
    click-handling clientside_callback below) writes the camera directly
    into `flux-cube-camera-store` the instant Plotly.js itself fires the
    event — synchronous, single code path, no intermediate prop-sync
    layer to race against. This callback just reads that store back via
    `State` and passes it straight into `flux_cube_figure`'s own
    `camera=` so every rebuild explicitly re-renders with it, the same
    "read back the graph's own current state and manually reapply it"
    shape the 2D map's own original fix used, just with a more directly-
    controlled capture point this time.

    (Separately, but related: a real contributor to "it reload[s] on
    every click" turned out to be Plotly's own *native* double-click
    behaviour — `doubleClick` config resets `scene.camera` to its initial
    value — combined with the reporter's own habit of double- rather than
    single-clicking to select. Disabled outright for this graph via
    `_FLUX_CUBE_GRAPH_CONFIG`'s `doubleClick: False`, see that constant's
    own comment; Aladin's equivalent native double-click-recentre
    behaviour for the 2D view was disabled the same way, see the
    aladin-target clientside_callback's own comment.)

    Only reused when `EXPLORER.flux_cube_camera_version == version`
    (see that field's own docstring) — i.e. only when this rebuild is for
    the *same* dataset the stored camera was itself captured against (a
    click/colour-range/scale/palette/transparency/2D<->3D-toggle change).
    The first rebuild for a genuinely *different* dataset forces
    `camera=None` instead, falling through to `flux_cube_figure`'s own
    default top-down view — otherwise a dataset switch made while already
    in 3D mode (or a switch back into 3D after loading a new dataset
    while still in 2D mode) would silently inherit whatever camera angle
    was left over from the *previous* dataset's cube, breaking "default
    orientation is top-down on first switch to 3D" for every dataset
    after the very first one. See `flux_cube_figure`'s own `camera`
    docstring for the read-back mechanism itself."""
    is_3d = mode == "3d"
    style_2d = {"display": "none" if is_3d else "block"}
    style_3d = {"display": "block" if is_3d else "none"}
    # "'COlor aladin point by' shoujlkd be only visible in 2d mode as in
    # 3d it is not releavant." Targets #aladin-color-by-row (see that
    # id's own comment in aps_l1_preview.py/aps_MOSviewer.py/
    # aps_IFUviewer.py) via a pure CSS rule (assets/style.css's
    # `.mode-3d #aladin-color-by-row`) keyed off *this* always-present
    # element's own className, rather than targeting that id directly as
    # a second Output here — confirmed live that a direct Output breaks
    # this callback outright ("A nonexistent object was used in an
    # Output of a Dash callback") the moment it fires before a dataset
    # has ever loaded: #aladin-color-by-row only exists once
    # _settings_panel_children() has rendered a kind's real content,
    # which (unlike aladin-2d-controls/flux-cube-container, both
    # unconditional parts of the base layout, see this callback's own
    # docstring) genuinely doesn't happen until after this app's normal
    # load sequence — the very same class of "Output/Input on a
    # component that doesn't exist yet" gotcha this file has hit and
    # fixed several times before, just discovered here on an Output
    # instead of the Input side those other fixes were for. A CSS class
    # on #main-panel (always present from initial page mount) can never
    # hit that failure mode: an unmatched selector is just inert, not an
    # error, regardless of whether #aladin-color-by-row exists yet.
    main_panel_class = "mode-3d" if is_3d else ""

    if not is_3d or not EXPLORER.loaded():
        return style_2d, style_3d, main_panel_class, no_update, no_update
    result = _flux_cube_data()
    if result is None:
        return style_2d, style_3d, main_panel_class, no_update, no_update
    ra, dec, wave, flux_matrix, items, radius, shape = result
    # See EXPLORER.flux_cube_camera_version's own docstring: only reuse
    # the stored camera when this rebuild is for the *same* dataset it
    # was itself captured against — otherwise (a genuinely new dataset)
    # reset to flux_cube_figure's own default top-down view rather than
    # inheriting a stale angle left over from a previous dataset's cube.
    camera = stored_camera if EXPLORER.flux_cube_camera_version == version else None
    EXPLORER.flux_cube_camera_version = version
    # A real, confirmed Dash quirk found by live testing (not a Playwright
    # artifact — reproduced identically with three different real
    # select-all-then-retype interaction methods): a *debounced*
    # `dcc.Input(type="number")` can commit `None` on blur after its
    # field was cleared and retyped, even though the actual DOM value at
    # that moment is genuinely the newly-typed one. Switched this field
    # to `type="text"` (sidesteps the browser's own native number-input
    # semantics and Dash's number-specific handling entirely — the same
    # class of workaround commonly used for this exact `dcc.Input`
    # flakiness) with manual parsing here instead of trusting a numeric
    # prop type. `bin_width` is therefore a plain string (or `None`
    # before the field has ever rendered) — parsed defensively, falling
    # back to `flux_cube_figure`'s own default (`None`) for anything
    # empty/unparseable rather than erroring the whole callback.
    try:
        bin_width_angstrom = float(bin_width) if bin_width not in (None, "") else None
    except (TypeError, ValueError):
        bin_width_angstrom = None
    # Only the *stretch* (scale) is shared with the 2D view's own Colour
    # range control, not its literal vmin/vmax numbers — those describe a
    # different quantity entirely (whatever the 2D catalog is currently
    # coloured by, e.g. "S/N", a handful to a few hundred) from the 3D
    # cube's own per-bin summed flux (routinely 1e5-1e7 on real data, see
    # flux_cube_figure's own vmin/vmax docstring) — reusing 2D's typed
    # numbers here would silently saturate/blank the whole cube on any
    # dataset where the two quantities' natural ranges don't happen to
    # overlap. vmin/vmax stay `None` (flux_cube_figure's own 1st/99th
    # percentile fallback, computed from the cube's own real data) either
    # way; only the shape of the stretch applied to that range is shared.
    range_state = _color_range_state()
    scale = range_state.color_scale if range_state is not None else "linear"
    try:
        fig = flux_cube_figure(ra, dec, wave, flux_matrix, items,
                                 fibre_radius_deg=radius, shape=shape, selected_item=selected_item,
                                 scale=scale, colorscale=_resolve_palette_override("Jet"),
                                 transparent=EXPLORER.cube_transparency_on,
                                 wave_bin_width_angstrom=bin_width_angstrom,
                                 depth_mode=depth_mode if depth_mode in ("slice", "cube") else "slice",
                                 height=int(_MAP_HEIGHT.rstrip("px")),
                                 camera=camera)
    except Exception as e:
        traceback.print_exc()
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.update_layout(title=f"Flux cube error: {e}", height=int(_MAP_HEIGHT.rstrip("px")))
    next_rebuild_version = (rebuild_version or 0) + 1
    return style_2d, style_3d, main_panel_class, fig, next_rebuild_version


def _flux_cube_wave_bounds():
    """Real full native wavelength coverage (Å, min/max) of whatever's
    currently loaded — sourced from the exact same `_flux_cube_data()`
    the 3D figure itself is built from (never a separately-maintained
    lookup that could silently drift from what the cube actually
    contains). `None` whenever there's nothing to build a cube from at
    all, matching `_flux_cube_data()`'s own convention."""
    result = _flux_cube_data()
    if result is None:
        return None
    wave = result[2]
    if wave is None or len(wave) == 0:
        return None
    return float(np.nanmin(wave)), float(np.nanmax(wave))


def _fmt_wave_angstrom(v):
    """"100" for a whole number, "100.35" otherwise — same "don't show a
    fake extra decimal for a round default" convention `flux-cube-bin-
    width`'s own initial `value=` already uses."""
    return str(int(round(v))) if abs(v - round(v)) < 1e-6 else f"{v:.2f}"


@app.callback(
    Output("flux-cube-wave-min", "value"),
    Output("flux-cube-wave-max", "value"),
    Input("dataset-version", "data"),
    prevent_initial_call=True,
)
def on_dataset_loaded_set_wave_range(version):
    """Pre-fills the wavelength sub-range fields with the dataset's real
    full native coverage on every fresh load — explicit request: "scroll
    through the wavelength range and select a start and end," which
    needs real numbers to start from rather than blank fields (blank
    means "full range" to `update_map_mode`/`flux_cube_figure` too, but
    a user can't usefully *scroll* from an unknown starting point). A
    genuine no-op (`no_update`/`no_update`, not a blank-out) whenever
    nothing's loaded yet or this kind/mode has no cube data at all — the
    fields are simply left as whatever they last showed rather than
    being cleared out from under whoever's mid-edit."""
    bounds = _flux_cube_wave_bounds()
    if bounds is None:
        return no_update, no_update
    lo, hi = bounds
    return _fmt_wave_angstrom(lo), _fmt_wave_angstrom(hi)


@app.callback(
    Output("flux-cube-wave-min", "value", allow_duplicate=True),
    Output("flux-cube-wave-max", "value", allow_duplicate=True),
    Input("flux-cube-wave-range-reset", "n_clicks"),
    prevent_initial_call=True,
)
def on_flux_cube_wave_range_reset(n_clicks):
    """"For 3d changing range I also want the reset option to reset to
    the default wavelenht range without changing the zoom or camera
    angle." Reuses the exact same `_flux_cube_wave_bounds()`/
    `_fmt_wave_angstrom()` on_dataset_loaded_set_wave_range (right
    above) already uses to fill these fields with the dataset's true
    full native coverage on a fresh load — this is that same "full
    range" value, just re-triggerable on demand instead of only once at
    load time.

    Writing new Start/End values here only ever reaches the purely
    client-side wave-filter callback below (flux-cube-wave-min/-max are
    deliberately never Inputs to update_map_mode itself — see that
    callback's own docstring) — so, like every other Start/End change,
    this can't touch the camera at all, satisfied by the existing
    architecture rather than needing anything camera-specific added
    here."""
    bounds = _flux_cube_wave_bounds()
    if bounds is None:
        return no_update, no_update
    lo, hi = bounds
    return _fmt_wave_angstrom(lo), _fmt_wave_angstrom(hi)


# Lock-width convenience + ◀/▶ scroll — both pure client-side arithmetic
# on flux-cube-wave-min/-max/-bin-width's own already-present values, no
# server round trip needed for either (matching this app's own established
# "clientside for pure UI/derived-value computation, server callback only
# once real dataset-backed work is needed" convention, e.g. the camera-
# capture/click-handling callbacks just below). Explicit request: "scroll
# through the wavelength range and select a start and end... [a] ratio
# [checkbox that] locks start and end together... this width is the one
# we already have [flux-cube-bin-width]... so no other computation is
# needed."
#
# Both callbacks write flux-cube-wave-max with allow_duplicate=True
# (same "Dash replaces the whole prop on every Output" pattern already
# established via _load_panel_style's own multiple allow_duplicate=True
# Outputs elsewhere in this file) — when Lock is checked, the two never
# actually disagree: the scroll callback shifts both endpoints by the
# same delta, so if the range was already exactly one bin width wide
# before the click, it still is afterward, and the lock-sync callback
# below (independently re-triggered by wave-min's own change) just
# recomputes the identical value. When Lock is off, the lock-sync
# callback is a no-op (its own JS returns no_update), so only the scroll
# callback's own shift (which preserves whatever custom width the user
# had, locked or not) ever actually lands.
app.clientside_callback(
    """
    function(waveMin, binWidth, lockValue) {
        if (!lockValue || lockValue.indexOf("locked") === -1) {
            return window.dash_clientside.no_update;
        }
        var lo = parseFloat(waveMin);
        var w = parseFloat(binWidth);
        if (!isFinite(lo) || !isFinite(w)) {
            return window.dash_clientside.no_update;
        }
        var hi = lo + w;
        return Number.isInteger(hi) ? String(hi) : hi.toFixed(2);
    }
    """,
    Output("flux-cube-wave-max", "value", allow_duplicate=True),
    Input("flux-cube-wave-min", "value"),
    Input("flux-cube-bin-width", "value"),
    # "data", not "value" — flux-cube-lock-width is a dcc.Store now (see
    # its own definition's comment for why), not the dcc.Checklist this
    # read from before; same []/["locked"] value shape either way, so
    # this JS body needed no change, only the prop name did.
    Input("flux-cube-lock-width", "data"),
    prevent_initial_call=True,
)


# Reliable field-commit for flux-cube-bin-width/-wave-min/-wave-max —
# explicit bug report, reproduced live (real Playwright runs against a
# real dataset, not assumed): "if I set the bin at range of 7000-7100 for
# example then for any range out of it like 5000-5100 or 5100-5200 or so
# it says No data in the wavelength range for the current dataset which
# is clearly wrong... Solve the issue that after a bit of playing with
# the wavelength bin and range it lost the rest of data."
#
# Root cause, confirmed directly: all three fields use `debounce=True`,
# and (see `flux-cube-bin-width`'s own docstring for how this was first
# found) that field's blur/Enter commit-to-Dash is genuinely unreliable
# in this Dash/browser combination — confirmed *again* here, live,
# stacking bin-width Apply clicks: `pristineDistinctZ` (the mesh's real
# number of wavelength layers) sometimes updated to match a newly-typed
# bin width, sometimes silently stayed at the *previous* value with no
# error and no visual sign anything failed. The existing Apply button is
# an *additional trigger* for update_map_mode, but it still reads
# `bin_width` from that same unreliable debounced prop — clicking it
# doesn't force the browser to flush whatever's currently typed, so Apply
# can itself fire using stale text.
#
# The user-visible fallout chases through the Lock/scroll machinery
# purely client-side, no server round trip needed to reproduce: the
# Lock-sync callback computes End = Start + (the *raw*, always-current
# bin-width field text) regardless of whether that width was ever
# actually applied to the mesh's real geometry. When Apply silently
# fails, the mesh stays stuck on its old, coarser layer spacing while
# Start/End march on using the new, unapplied (often much narrower)
# width — a narrow window has a real chance of falling entirely between
# two of the old mesh's sparse layers, correctly reporting "0 survived"
# for a range that verifiably *is* within the dataset's native coverage,
# just not aligned with the stale geometry still on screen.
#
# Fix: stop trusting dcc.Input's own internal debounce bookkeeping at
# all. Bind our own plain native 'change' (fires reliably on blur when
# the value actually changed — standard, well-tested browser behaviour,
# no Dash internals involved) and Enter-keydown listeners directly on
# each field (bind-once guard on the element itself, same pattern as the
# flux-cube-graph click-to-select binding below) that read the DOM's own
# always-current `.value` and force-commit it via `dash_clientside.
# set_props` — a direct, synchronous, already-proven-reliable technique
# in this exact file (the click-to-select and camera-capture listeners
# just below use the identical call). The Apply button additionally
# force-flushes bin-width's current value the instant it's clicked,
# rather than assuming a browser blur/change event already fired first
# (a mouse click straight from a focused text field to a button doesn't
# reliably fire 'change' before the click lands in every browser).
app.clientside_callback(
    """
    function(nClicksApply) {
        function bindReliableCommit(id) {
            var el = document.getElementById(id);
            if (!el || el._pyapsReliableCommitBound) { return; }
            el._pyapsReliableCommitBound = true;
            function commit() {
                window.dash_clientside.set_props(id, {value: el.value});
            }
            el.addEventListener("change", commit);
            el.addEventListener("keydown", function(e) {
                if (e.key === "Enter") { commit(); }
            });
        }
        bindReliableCommit("flux-cube-bin-width");
        bindReliableCommit("flux-cube-wave-min");
        bindReliableCommit("flux-cube-wave-max");
        if (nClicksApply) {
            var bw = document.getElementById("flux-cube-bin-width");
            if (bw) {
                // Setting "value" here is itself an Input change to the
                // separate debounce-scheduling callback just below (its
                // own Input is flux-cube-bin-width's own "value") --
                // confirmed live this was silently creating a *third*,
                // redundant, ~500ms-delayed rebuild for every single
                // Apply click (a genuine race against the two correct,
                // immediate ones below, and — since update_map_mode also
                // mutates shared per-session server state, e.g. EXPLORER.
                // flux_cube_camera_version — landing after them could
                // clobber a just-applied camera/filter state with a
                // stale one). This flag tells that callback's own next
                // firing to skip scheduling anything, since the line
                // right after this one already commits the value
                // immediately, with no debounce, itself.
                window._pyapsBinWidthSuppressNextDebounce = true;
                window.dash_clientside.set_props("flux-cube-bin-width", {value: bw.value});
                // Also skip flux-cube-bin-width-debounced's own delay
                // (see that Store's own comment) -- Apply is a deliberate
                // "do it now" click, not something that should still wait
                // out a debounce window meant for rapid typing/clicking.
                // Cancelling any pending timer here too, so a delayed
                // update scheduled just before Apply was clicked can't
                // land *after* this one and clobber it with a stale value.
                if (window._pyapsBinWidthDebounceTimer) {
                    clearTimeout(window._pyapsBinWidthDebounceTimer);
                    window._pyapsBinWidthDebounceTimer = null;
                }
                window.dash_clientside.set_props("flux-cube-bin-width-debounced", {data: bw.value});
            }
        }
        return "";
    }
    """,
    Output("flux-cube-field-commit-dummy", "children"),
    Input("flux-cube-bin-width-apply", "n_clicks"),
    prevent_initial_call=False,
)


# Debounce-scheduling pair for flux-cube-depth-mode/-bin-width — see
# flux-cube-depth-mode-debounced/flux-cube-bin-width-debounced's own
# comments (where they're declared) for the full "why". Deliberately
# *not* returning a value straight from this callback (which would fire
# immediately, defeating the debounce) — instead schedules a delayed
# `dash_clientside.set_props` call, same technique the reliable-commit
# callback just above already uses successfully, cancelling any
# still-pending timer from an *earlier* change first so only the very
# last change in a rapid burst ever actually lands, at a fixed delay
# after that burst goes quiet — not accumulating one delay per change.
app.clientside_callback(
    """
    function(value) {
        if (window._pyapsDepthModeDebounceTimer) {
            clearTimeout(window._pyapsDepthModeDebounceTimer);
        }
        window._pyapsDepthModeDebounceTimer = setTimeout(function() {
            window.dash_clientside.set_props("flux-cube-depth-mode-debounced", {data: value});
        }, 500);
        return "";
    }
    """,
    Output("flux-cube-depth-mode-debounce-dummy", "children"),
    Input("flux-cube-depth-mode", "value"),
    prevent_initial_call=True,
)


app.clientside_callback(
    """
    function(value) {
        // See the reliable-commit/Apply callback above, right where it
        // sets this same flag, for why this guard exists -- an Apply
        // click force-writes flux-cube-bin-width's own "value" (to keep
        // the field's displayed text in sync even when a click landed
        // before a native 'change' event had a chance to fire first),
        // which is itself this callback's own Input and would otherwise
        // schedule a redundant, stale-by-500ms rebuild on top of the one
        // Apply already triggers immediately.
        if (window._pyapsBinWidthSuppressNextDebounce) {
            window._pyapsBinWidthSuppressNextDebounce = false;
            return "";
        }
        if (window._pyapsBinWidthDebounceTimer) {
            clearTimeout(window._pyapsBinWidthDebounceTimer);
        }
        window._pyapsBinWidthDebounceTimer = setTimeout(function() {
            window.dash_clientside.set_props("flux-cube-bin-width-debounced", {data: value});
        }, 500);
        return "";
    }
    """,
    Output("flux-cube-bin-width-debounce-dummy", "children"),
    Input("flux-cube-bin-width", "value"),
    prevent_initial_call=True,
)


# The lock toggle button itself — flips flux-cube-lock-width's stored
# []/["locked"] value on every click (the exact same shape the old
# dcc.Checklist produced, see that Store's own comment) and re-styles
# the button in place (glyph+word, plus colour) so the *locked* state is
# legible at a glance, not just from the emoji swap alone — a labelled
# pill now (see the layout site's own comment for why the previous
# bracket-glyph design was dropped), not a bare icon.
app.clientside_callback(
    """
    function(nClicks, currentValue) {
        if (!nClicks) { return window.dash_clientside.no_update; }
        var isLocked = !!(currentValue && currentValue.indexOf("locked") !== -1);
        var next = !isLocked;
        // Action-oriented (what clicking *again* would now do, i.e. the
        // opposite of `next`'s own state) — was backwards on the first
        // pass (confirmed live: the button's own style correctly
        // flipped to the locked look, but the text stayed "Lock" while
        // already locked), fixed here.
        var label = next ? "🔓 Unlock" : "🔒 Lock";
        var style = {
            border: "1px solid " + (next ? "var(--pyaps-accent)" : "var(--pyaps-line-strong)"),
            borderRadius: "var(--pyaps-radius-pill)",
            backgroundColor: next ? "var(--pyaps-accent-soft)" : "var(--pyaps-paper-raised)",
            color: next ? "var(--pyaps-accent)" : "var(--pyaps-ink-muted)",
            fontSize: "11px", fontWeight: "600", padding: "4px 12px", cursor: "pointer",
            whiteSpace: "nowrap",
        };
        return [next ? ["locked"] : [], label, style];
    }
    """,
    Output("flux-cube-lock-width", "data"),
    Output("flux-cube-lock-toggle", "children"),
    Output("flux-cube-lock-toggle", "style"),
    Input("flux-cube-lock-toggle", "n_clicks"),
    State("flux-cube-lock-width", "data"),
    prevent_initial_call=True,
)


app.clientside_callback(
    """
    function(nBack, nFwd, waveMin, waveMax, binWidth) {
        var trigId = window.dash_clientside.callback_context.triggered_id;
        if (!trigId) {
            return [window.dash_clientside.no_update, window.dash_clientside.no_update];
        }
        var lo = parseFloat(waveMin), hi = parseFloat(waveMax);
        if (!isFinite(lo) || !isFinite(hi)) {
            // Nothing real to scroll from yet (e.g. no dataset loaded) —
            // see on_dataset_loaded_set_wave_range for how these two
            // fields normally get real starting values on a fresh load.
            return [window.dash_clientside.no_update, window.dash_clientside.no_update];
        }
        var w = parseFloat(binWidth);
        if (!isFinite(w) || w <= 0) { w = hi - lo; }
        if (!isFinite(w) || w <= 0) { w = 100; }
        var delta = (trigId === "flux-cube-wave-scroll-back") ? -w : w;
        function fmt(v) { return Number.isInteger(v) ? String(v) : v.toFixed(2); }
        return [fmt(lo + delta), fmt(hi + delta)];
    }
    """,
    Output("flux-cube-wave-min", "value", allow_duplicate=True),
    Output("flux-cube-wave-max", "value", allow_duplicate=True),
    Input("flux-cube-wave-scroll-back", "n_clicks"),
    Input("flux-cube-wave-scroll-fwd", "n_clicks"),
    State("flux-cube-wave-min", "value"),
    State("flux-cube-wave-max", "value"),
    State("flux-cube-bin-width", "value"),
    prevent_initial_call=True,
)


# Purely client-side wavelength-range *scrolling* — explicit follow-up
# report: "every time I change the start and end, it reloads the whole
# 3d view... it should be a precomputed thing and behave the same way as
# we scroll through one axis without reloading the 3d object." See
# update_map_mode's own docstring for the full "why" (that callback now
# always builds the *full* native-range cube exactly once per genuine
# rebuild, no wave_min/wave_max involved at all) — this callback is the
# other half: it re-runs on every Start/End (and full-cube-rebuild)
# change and does the actual windowing itself, entirely in the browser.
#
# Mechanism: every vertex in the cube already carries its own real
# wavelength as its `z` coordinate (`flux_cube.py`'s own
# `bin_centers_wave`, shared by both the "circle" ring-line trace and
# the "square" flat-plate mesh) — no new data field needed at all, this
# just reads what's already there. `window._pyapsFluxCubePristine`
# caches every trace's own *unfiltered* x/y/z arrays, refreshed only
# when `flux-cube-graph`'s `figure` prop itself changes (i.e. only on a
# genuine server rebuild — mode switch, new selection, colour/scale/bin
# -width change, or a brand new dataset); every filter-only trigger
# (Start/End typed, ◀/▶ scroll, Lock toggle) re-slices *that* cached
# copy, never a get-smaller-over-time already-filtered one, so scrolling
# forward and back losslessly recovers whatever was hidden.
#
# Refresh trigger, two real gotchas deep, both found live (not guessed)
# chasing the same "lost the rest of data after playing with bin/range"
# report:
#
# First: originally gated purely on `triggered_id === "flux-cube-graph"`.
# Confirmed unreliable whenever a bin-width Apply cascades into several
# near-simultaneous round trips (a genuine rebuild *plus* the Lock-sync
# callback recomputing End off the new bin width) — Dash's own
# `callback_context.triggered_id` reports only *one* of the Inputs that
# changed in a given wave, and on the invocation that happened to land
# with the figure genuinely new but `triggered_id` pointing at
# `flux-cube-wave-max` instead, the pristine refresh was silently
# skipped. Tried comparing the `figure` argument's own object reference
# against the one last cached against instead — better, but still wrong:
#
# Second, worse gotcha: `flux-cube-graph`'s own `figure` prop turns out
# to change on far more than genuine server rebuilds — `dcc.Graph`
# mirrors back *any* Plotly.js relayout into that same prop, including a
# plain camera drag/pan/zoom that never touches a single vertex.
# Confirmed live: rotating the cube by hand (a `scene.camera`-only
# relayout, no data involved at all) handed this callback a new `figure`
# reference on its very next invocation, which re-cached `gd._fullData`
# at that moment as the new "pristine" — except `gd._fullData` at any
# given instant reflects whatever's *currently displayed*, i.e. already
# NaN-filtered down to the live Start/End window from this callback's own
# most recent `Plotly.restyle()`. One ordinary rotate-then-scroll
# sequence was enough to permanently discard everything outside whatever
# window happened to be showing at rotate time — this, not the range
# controls in isolation, is what "after a bit of playing... it lost the
# rest of data" actually was: ordinary camera dragging interleaved with
# range changes, silently baking each moment's filtered view in as if it
# were the dataset's full coverage.
#
# Fixed properly this time by not watching `flux-cube-graph`'s `figure`
# prop *at all* — `flux-cube-rebuild-version` (a small dedicated
# `dcc.Store`, see its own comment where it's declared) is bumped by
# `update_map_mode` on, and only on, its own genuine-rebuild return path,
# untouched by anything client-side-only (camera relayout, hover,
# click-to-select, or this callback's own restyle side effect) — the one
# signal with no ambiguity left to exploit.
#
# A real, confirmed gotcha caught building this (live Playwright test
# against a real ~115,000-point trace, not assumed): `gd.data[i].x/y/z`
# is NOT reliably a plain array to read from — this app's installed
# Plotly.js version transparently switches large numeric array props to
# a compact `{dtype, bdata}` base64-binary wire format, which has no
# `.slice()` and silently isn't what it looks like. The pristine cache
# is built from `gd._fullData` instead — Plotly.js's own fully-resolved
# working copy (confirmed directly: real decoded `Float64Array`s there,
# parallel to `gd.data` one trace at a time), falling back to `gd.data`
# only if `_fullData` genuinely doesn't exist yet.
#
# Vertices outside [Start, End] are set to NaN, not removed from the
# array — Plotly's own established "break the line here"/empty-point
# convention (confirmed directly against a real Plotly.js Mesh3d, not
# assumed: NaN-ing one plate's own private vertices cleanly hides just
# that plate with no artifact on any neighbour, since — per flux_cube.py
# own construction — no two plates or rings ever share a vertex).
# Circle mode's own pre-existing ring-closing NaN break-points survive
# this untouched (a NaN z already fails the `>= lo && <= hi` range test
# on its own, landing in the same "hide it" branch it would already be
# in) — no special-casing needed to avoid corrupting them.
#
# `Plotly.restyle(gd, {x:[...], y:[...], z:[...]})` (array-valued, no
# explicit trace-index list) applies one array per trace in `gd.data`
# order — covers the main cube trace *and* the separate "selected item"
# highlight-marker trace (when present) uniformly, in one call, with no
# server round trip and no `customdata`/click-mapping disruption (never
# touched, so click-to-select keeps working on whatever's still visible
# exactly as before).
app.clientside_callback(
    """
    function(rebuildVersion, waveMin, waveMax) {
        var wrapper = document.getElementById("flux-cube-graph");
        var gd = wrapper ? wrapper.querySelector(".js-plotly-plot") : null;
        if (!gd || !gd.data || !gd.data.length) {
            return [window.dash_clientside.no_update, window.dash_clientside.no_update];
        }
        if (rebuildVersion !== window._pyapsFluxCubeLastRebuildVersion) {
            window._pyapsFluxCubeLastRebuildVersion = rebuildVersion;
            // gd.data itself can hold this app's own installed Plotly.js
            // version's compact {dtype, bdata} binary wire format for
            // large numeric arrays (confirmed directly against a real
            // ~115,000-point trace, not assumed) rather than a plain
            // array/typed array -- .slice() on that throws. gd._fullData
            // is Plotly.js's own fully-resolved, already-decoded working
            // copy (real Float64Arrays, confirmed directly) it actually
            // draws from, parallel to gd.data one trace at a time -- read
            // from there instead.
            var fullData = gd._fullData || gd.data;
            window._pyapsFluxCubePristine = fullData.map(function(tr) {
                return {x: (tr.x || []).slice(), y: (tr.y || []).slice(), z: (tr.z || []).slice()};
            });
            // Real median spacing between this mesh's own distinct
            // wavelength-bin centres -- see this callback's own slack/
            // epsilon comment below for why it's captured here, once per
            // genuine rebuild, rather than recomputed on every filter.
            var firstTrace = window._pyapsFluxCubePristine[0];
            var distinctSorted = firstTrace
                ? Array.from(new Set(firstTrace.z.filter(function(v) { return v !== null && isFinite(v); })))
                    .sort(function(a, b) { return a - b; })
                : [];
            if (distinctSorted.length >= 2) {
                var gaps = [];
                for (var g = 1; g < distinctSorted.length; g++) {
                    gaps.push(distinctSorted[g] - distinctSorted[g - 1]);
                }
                gaps.sort(function(a, b) { return a - b; });
                window._pyapsFluxCubeMedianGap = gaps[Math.floor(gaps.length / 2)];
            } else {
                window._pyapsFluxCubeMedianGap = 0;
            }
        }
        var pristine = window._pyapsFluxCubePristine;
        if (!pristine || pristine.length !== gd.data.length) {
            // Nothing cached yet, or it's from a different figure
            // (trace count changed) -- nothing safe to filter.
            return [window.dash_clientside.no_update, window.dash_clientside.no_update];
        }
        var lo = parseFloat(waveMin), hi = parseFloat(waveMax);
        var hasRange = isFinite(lo) && isFinite(hi);
        // Swapped, not left as an always-empty [hi, lo] range -- matches
        // flux_cube_figure's own server-side wave_min/wave_max swap
        // (see its docstring: "Swapped automatically if wave_min is
        // given larger than wave_max"), so a reversed Start/End reads
        // the same way here as it would on a fresh server rebuild,
        // rather than silently landing on the *other* new "no data"
        // message below.
        if (hasRange && lo > hi) { var tmp = lo; lo = hi; hi = tmp; }
        // Bin-alignment slack -- explicit bug report, reproduced live
        // (real Playwright runs against real data, not assumed): "if I
        // set the bin at range of 7000-7100... then for any range out of
        // it like 5000-5100 or 5100-5200... it says No data... which is
        // clearly wrong." Root cause confirmed directly: flux_cube_figure
        // bins the *dataset's own* native wavelength span into
        // `round(wave_span / bin_width)` equal bins anchored at the
        // dataset's real minimum -- never at whatever Start the user
        // later types -- so the mesh's true bin spacing (confirmed live:
        // ~102.1 Å for a typed "100") is always a hair off the typed bin
        // width, and is essentially never exactly equal to it. A
        // [Start, Start+width) query window that's narrower than the
        // real spacing has a genuine dead zone every period where it can
        // straddle two real bins and catch neither -- confirmed live: an
        // ordinary two-click ◀/▶ scroll from a perfectly good starting
        // range reached one within 3 steps. This isn't a data problem
        // (the dataset's real coverage is continuous and this range
        // genuinely is inside it) -- it's this client-side filter being
        // stricter than the mesh's own coarse, evenly-spaced-but-not
        // -width-matched binning can actually resolve. Widening the
        // match window by half of (real spacing − requested width) on
        // each side exactly closes that dead zone when the requested
        // width is narrower than the mesh's real spacing, and is a no-op
        // (zero slack) once it's wide enough to always span at least one
        // real bin on its own -- a query genuinely far outside the
        // dataset's real coverage (e.g. typing 50000) still correctly
        // shows nothing.
        var slack = 0;
        if (hasRange && window._pyapsFluxCubeMedianGap) {
            slack = Math.max(0, (window._pyapsFluxCubeMedianGap - (hi - lo)) / 2);
        }
        var xArrays = [], yArrays = [], zArrays = [];
        var survived = 0;
        for (var t = 0; t < pristine.length; t++) {
            var src = pristine[t];
            if (!hasRange) {
                xArrays.push(src.x); yArrays.push(src.y); zArrays.push(src.z);
                survived += src.z.length;
                continue;
            }
            var n = src.z.length;
            var x2 = new Array(n), y2 = new Array(n), z2 = new Array(n);
            for (var i = 0; i < n; i++) {
                var zv = src.z[i];
                if (zv !== null && isFinite(zv) && zv >= lo - slack && zv <= hi + slack) {
                    x2[i] = src.x[i]; y2[i] = src.y[i]; z2[i] = zv;
                    survived++;
                } else {
                    x2[i] = NaN; y2[i] = NaN; z2[i] = NaN;
                }
            }
            xArrays.push(x2); yArrays.push(y2); zArrays.push(z2);
        }
        Plotly.restyle(gd, {x: xArrays, y: yArrays, z: zArrays});
        // "When I go before the wavelenght range and after the max, I
        // need it put a message or wanrng next to the range saying out
        // of range...right now if we go out of range nothing happens
        // and just the 3d plot get white/empty." A real, dedicated
        // check for whether a genuine [Start, End] selection matched
        // *anything at all* -- not a guess at whether the values look
        // plausible, since (see this callback's own module-level
        // comment) the actual filtering already computes exactly this
        // for free while it NaNs the non-matching vertices out.
        var warning = (hasRange && survived === 0)
            ? "⚠ No data in this wavelength range for the current dataset — try a range within its native coverage, or click Reset."
            : "";
        return ["", warning];
    }
    """,
    Output("flux-cube-wave-filter-dummy", "children"),
    Output("flux-cube-wave-range-warning", "children"),
    Input("flux-cube-rebuild-version", "data"),
    Input("flux-cube-wave-min", "value"),
    Input("flux-cube-wave-max", "value"),
    prevent_initial_call=True,
)


app.clientside_callback(
    """
    function(figure) {
        // Explicit requirement: "for selection, it must be only by
        // selecting from the xy coordinate as it is now" — every single
        // point in the cube (any wavelength bin, any ring position
        // around an item's own circular cross-section) carries that
        // item's real ID as customdata, so a click anywhere along an
        // item's own depth column should resolve to the exact same
        // selection a 2D click on that same item would have.
        //
        // This is NOT Input("flux-cube-graph", "clickData") the way
        // every other click-driven selection in this app is — confirmed
        // live (Playwright, real mouse clicks against the real running
        // app) that Plotly.js's own "plotly_click" event never fires at
        // all for scatter3d traces in this app's installed Plotly.js
        // version (3.1.1), even though "plotly_hover" fires correctly
        // for the exact same points — a known, documented Plotly.js gl3d
        // limitation (plotly/plotly.js community reports), not a bug in
        // this app's own wiring. Confirmed working workaround, also
        // tested live: track the most recently *hovered* point's
        // customdata (plotly_hover does fire) in a variable on the
        // graph's own DOM node, then read that back on a plain native
        // DOM "click" listener (which always fires — it's the browser's
        // own standard click event on the canvas element, nothing
        // Plotly-specific about it) rather than Plotly's own synthetic
        // click event.
        var wrapper = document.getElementById("flux-cube-graph");
        if (!wrapper) { return ""; }
        var gd = wrapper.querySelector(".js-plotly-plot");
        if (!gd) { return ""; }
        // Guarded on the DOM node itself (not a global flag) so a
        // genuinely new graph element (e.g. after being hidden/shown)
        // gets its own fresh listeners rather than silently relying on
        // one attached to a since-removed node.
        if (gd._pyapsFluxCubeBound) { return ""; }
        gd._pyapsFluxCubeBound = true;
        gd._pyapsHoveredItem = null;
        try {
            gd.on("plotly_hover", function(d) {
                if (d && d.points && d.points[0] && d.points[0].customdata !== undefined) {
                    gd._pyapsHoveredItem = d.points[0].customdata;
                }
            });
            gd.on("plotly_unhover", function() { gd._pyapsHoveredItem = null; });
            gd.addEventListener("click", function() {
                // Explicit request: off by default, opt-in only (see the
                // toggle's own comment in the layout) — the toggle-sync
                // clientside callback below is the only thing that ever
                // writes this flag; this listener itself is bound once
                // and never rebuilt, so it has to read the flag fresh on
                // every click rather than capture it at bind time.
                if (!window._pyapsFluxCubeClickEnabled) { return; }
                if (gd._pyapsHoveredItem !== null && gd._pyapsHoveredItem !== undefined) {
                    window.dash_clientside.set_props(
                        "selected-item", {data: gd._pyapsHoveredItem}
                    );
                }
            });
            // Explicit report: "it still reload[s] the whole window every
            // time I click to select a spaxel... very few times it tried
            // to save the camera angle." The previous mechanism
            // (update_map_mode reading flux-cube-graph's own "figure"
            // prop back via State) depended on dcc.Graph's own internal
            // plotly_relayout handler having already finished writing the
            // live camera into that prop *before* a later, separate click
            // dispatch captured its current value — a genuine race
            // between two independent pieces of Dash-internal bookkeeping
            // this app doesn't control the timing of, matching exactly
            // the "sometimes works" symptom reported (not a deterministic
            // failure, which a permanently-broken mechanism would be).
            // Fixed by owning the capture directly: this listener writes
            // the camera straight into flux-cube-camera-store the instant
            // Plotly.js itself fires the event — no dependency on
            // dcc.Graph's own async-graph.js figure-prop-sync logic or
            // its timing at all, the same "read the graph's own current
            // state and reapply it" principle as before, just cutting out
            // the one indirection layer that was actually racy.
            gd.on("plotly_relayout", function(d) {
                if (d && d["scene.camera"]) {
                    window.dash_clientside.set_props(
                        "flux-cube-camera-store", {data: d["scene.camera"]}
                    );
                }
            });
        } catch (e) { console.error("PyAPS: wiring flux-cube click handling failed: " + e); }
        return "";
    }
    """,
    Output("flux-cube-click-dummy", "children"),
    Input("flux-cube-graph", "figure"),
)


app.clientside_callback(
    """
    function(value) {
        // The only writer of this flag — the click listener above is
        // bound once (guarded by gd._pyapsFluxCubeBound) and never
        // rebuilt, so it reads this window global fresh on every click
        // rather than being re-wired by Dash each time the toggle
        // changes. Undefined/never-fired reads as falsy (off), matching
        // the checklist's own default value=[] (off).
        window._pyapsFluxCubeClickEnabled = (value || []).includes("enabled");
        return "";
    }
    """,
    Output("flux-cube-click-toggle-dummy", "children"),
    Input("flux-cube-click-toggle", "value"),
)


@app.callback(
    Output("aladin-catalog-data", "data", allow_duplicate=True),
    Input("aladin-true-size", "value"),
    prevent_initial_call=True,
)
def on_aladin_true_size_toggle(value):
    """"True fibre/spaxel size" overlay mode — explicit user request (see
    _true_size_radius_deg's own docstring). No-op guard for the usual
    dynamically-mounted-checkbox-fires-once reason."""
    if not EXPLORER.loaded():
        return no_update
    new_value = "true" in (value or [])
    if new_value == EXPLORER.aladin_true_size:
        return no_update
    EXPLORER.aladin_true_size = new_value
    return _aladin_catalog_payload()


@app.callback(
    Output("aladin-info-box", "children"),
    Input("selected-item", "data"),
    Input("dataset-version", "data"),
)
def update_aladin_info_box(selected_item, version):
    return _aladin_info_text(selected_item)


@app.callback(
    Output("slit-explorer-container", "children"),
    Input("selected-item", "data"),
    Input("dataset-version", "data"),
    Input("aladin-catalog-data", "data"),
)
def update_slit_explorer(selected_item, version, catalog_data):
    """Rebuilt wholesale on any trigger (not split into a separate
    dataset-version-only "does the panel exist at all" callback plus a
    selected-item-only "move the marker" one) deliberately — with two
    callbacks both potentially firing on the same dataset-version tick,
    nothing would guarantee the container's own Graph existed in the DOM
    yet by the time a figure-only update tried to target it. Rebuilding
    the whole (small, cheap) figure every time sidesteps that ordering
    hazard entirely.

    `Input("aladin-catalog-data", ...)` added for a real, confirmed gap —
    explicit report: "when I change the scale in the coordinate map, it
    should also change the colour scale in the Slit Explorer to match" —
    `_build_slit_explorer_figure` already reads `STATE.color_scale`
    (and color_vmin/color_vmax/color_by) fresh on every call, so the
    figure itself was never wrong once rebuilt; the gap was purely that
    nothing told this callback *to* rebuild when only the Scale dropdown/
    Min/Max boxes/Reset button changed — `on_color_scale_change`/
    `on_color_range_typed`/`on_color_range_reset` all write straight to
    `aladin-catalog-data` (the Aladin overlay's own redraw trigger)
    without touching `dataset-version` at all, so before this fix the
    Slit Explorer only visually caught up the next time a *selection* or
    full dataset change happened to fire it anyway — easy to miss as "not
    updating" when only the scale itself was actually changed. Listening
    to the exact same store the Aladin overlay redraws from guarantees
    the two can never disagree, the same guarantee the docstring on
    `_build_slit_explorer_figure` already claimed but didn't fully honour
    until now.

    Wrapped in try/except (same pattern as update_tab_content above) in
    direct response to a real report — "sometimes the connection between
    the Aladin map and Slit Explorer drops... it doesn't get the updated
    NSPEC when I click a new target" — that's most simply explained by an
    uncaught exception somewhere in this path: Dash leaves an Output at
    whatever it last successfully rendered when a callback raises, which
    looks exactly like "stopped updating" from the outside, with nothing
    surfaced to the user at all. This doesn't claim to have found (or
    fixed) whatever specific input triggers it, if that's really what's
    happening — but it guarantees any future occurrence becomes a visible
    Log-panel entry instead of a silent, undiagnosable stall."""
    try:
        return _slit_explorer_container(selected_item)
    except Exception as e:
        print(f">>> Error rendering Slit Explorer for item {selected_item!r}: {e}")
        traceback.print_exc()
        return html.Div(f"Slit Explorer error: {e}", style={"color": "crimson", "fontSize": "11px"})


@app.callback(
    Output("selected-item", "data", allow_duplicate=True),
    Input("slit-explorer-graph", "clickData"),
    prevent_initial_call=True,
)
def on_slit_explorer_click(click_data):
    """Click-to-select from the Slit Explorer strip itself, same
    customdata-lookup convention as fiber_map_figure's own click handling
    — keeps selection fully bidirectional between Aladin and this panel."""
    if EXPLORER.kind != "l1" or not click_data:
        return no_update
    points = click_data.get("points") or []
    if not points:
        return no_update
    customdata = points[0].get("customdata")
    if not customdata:
        return no_update
    return int(customdata[0])


@app.callback(
    Output("aladin-legend", "children"),
    Input("aladin-catalog-data", "data"),
)
def update_aladin_legend(catalog_data):
    """Triggered off aladin-catalog-data itself (not dataset-version
    directly) so the legend redraws on exactly the same events the catalog
    overlay's own colours do — a fresh load, a maptype switch, or a
    colour-range typed/reset — without listing each of those triggers a
    second time here and risking the two drifting out of sync."""
    if not EXPLORER.loaded():
        return None
    return _aladin_legend_children()


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

def _build_arg_parser():
    parser = argparse.ArgumentParser(description="PyAPS Explorer — unified L1/L2 viewer")
    parser.add_argument("--outpath", type=none_or_str, default=None,
                         help="L2: results directory (used with --headname)")
    parser.add_argument("--headname", type=none_or_str, default=None,
                         help="L2: head name / file prefix (used with --outpath)")
    parser.add_argument("--infiles", nargs="+", type=str, default=None,
                         help="L1: input stack/stackcube filename(s) — alternative to --outpath/--headname")
    parser.add_argument("--infiles_list", type=none_or_str, default=None,
                         help="L1: file containing one infile name per line")
    parser.add_argument("--caldir", type=none_or_str, default=None, help="L1: calibration directory (LSF files)")
    parser.add_argument("--catdir", type=none_or_str, default=None, help="L1: catalogue directory")
    parser.add_argument("--configdir", type=none_or_str, default=None, help="L1: config directory")
    parser.add_argument("--port", type=int, default=8080, required=False, help="Dash server port")
    parser.add_argument("--debug", action="store_true", default=False, help="Run the Dash server in debug mode")
    parser.add_argument("--multi-session", action="store_true", default=False,
                         help="Isolate state per browser (cookie-based) instead of one shared "
                              "dataset for the whole process — for a server reachable by more "
                              "than one person at once. Off by default: standalone/single-user "
                              "usage (this flag, or the Docker/gunicorn entrypoint's own "
                              "PYAPS_EXPLORER_MULTI_SESSION=1, is the only way this turns on). "
                              "Don't combine with --infiles/--outpath+--headname: a CLI-preloaded "
                              "dataset only seeds the transient default bundle, which no real "
                              "browser session in multi-session mode ever sees.")
    parser.add_argument("--require-weaveor-auth", action="store_true", default=False,
                         help="Require a valid signed handoff token (from a trusted upstream "
                              "app, e.g. weaveOR) before loading any dataset — see "
                              "aps_explorer_auth.py's own module docstring for the full design. "
                              "Off by default: standalone/single-user usage (this flag, or "
                              "PYAPS_EXPLORER_REQUIRE_WEAVEOR_AUTH=1, is the only way this turns "
                              "on). Implies --multi-session (a shared-but-unauthenticated "
                              "deployment makes no sense) — turning this on also turns that on, "
                              "even if not passed separately. Needs "
                              "PYAPS_EXPLORER_WEAVEOR_SECRET set to the same value the token-"
                              "issuing app signs with, or the server refuses to start. Don't "
                              "combine with --infiles/--outpath+--headname, for the same reason "
                              "as --multi-session above.")
    return parser


# --------------------------------------------------------------------------- #
# Multi-panel L2 map feature callbacks — see _extra_map_panels_section's
# own module-level comment (near _value_tables_children) for the full
# design/reasoning.
# --------------------------------------------------------------------------- #

@app.callback(
    Output("extra-map-rows-dd", "value", allow_duplicate=True),
    Output("extra-map-cols-dd", "value", allow_duplicate=True),
    Input("extra-map-rows-dd", "value"),
    Input("extra-map-cols-dd", "value"),
    prevent_initial_call=True,
)
def sync_extra_map_grid_dims(rows, cols):
    """Explicit follow-up report: "if I set the column to 1 or 2...
    the row must change to at least one... or if I set the row to 1,
    the column must change to 1 as well... otherwise we change[d] row
    and nothing happen[s] as column is still 0." 0/0 is still the
    genuine "show nothing" starting state (see _extra_map_panels_grid's
    own comment) — this only bumps the *other* dimension up to 1 the
    moment the user deliberately sets one of them away from 0, so a
    single dropdown change always actually shows something instead of
    silently doing nothing until the second one is also touched.

    Reads `ctx.triggered_id` to only ever push the dimension the user
    did *not* just touch — otherwise setting Rows to 2 while Cols is
    already 3 would wrongly reset Cols back to 1. Self-terminates safely
    (no infinite loop): the write below changes the *other* dropdown's
    own value, which re-fires this same callback, but by then both
    dimensions are already >0, so neither `if` matches and it returns
    `no_update, no_update`."""
    rows = int(rows or 0)
    cols = int(cols or 0)
    trig = ctx.triggered_id
    if trig == "extra-map-rows-dd" and rows > 0 and cols == 0:
        return no_update, 1
    if trig == "extra-map-cols-dd" and cols > 0 and rows == 0:
        return 1, no_update
    return no_update, no_update


@app.callback(
    Output("extra-map-panels-grid", "children"),
    Output("extra-map-panels-section", "style"),
    Input("extra-map-rows-dd", "value"),
    Input("extra-map-cols-dd", "value"),
    Input("dataset-version", "data"),
    State("selected-item", "data"),
)
def update_extra_map_panels_grid(rows, cols, _version, selected_item):
    """(Re)builds the whole panel grid for an *ordinary* Rows/Cols
    resize or a new dataset — the one place panel *count* changes,
    deliberately not MATCH-scoped (changing Rows/Cols unavoidably
    changes how many panels exist at all). Every other per-panel change
    (Map/Scale/Palette/Min/Max, and cross-panel selection sync) is
    handled by its own MATCH/clientside callback below instead,
    precisely so those never have to rebuild — or even touch — any
    *other* panel. `prevent_initial_call` deliberately left at its
    default (fires on mount too) — a fresh page load for a session
    that's already looking at a loaded IFU/MOS dataset (e.g. a same-tab
    reload) never fires a *new* `dataset-version` change, so skipping
    the initial call here would leave the grid empty until the user
    touched Rows/Cols themselves.

    A genuine Load (see on_extra_map_load below) is deliberately a
    *separate* callback writing to these same two Outputs (via
    `allow_duplicate=True`) rather than a third Input feeding this one —
    confirmed live this was a real, not hypothetical, race: Loading used
    to write Rows/Cols *and* a shared "pending config" Store together in
    one response, and Dash dispatched this callback once per changed
    Input in that wave rather than once combined — whichever invocation
    (rows-triggered, cols-triggered, or pending-triggered) happened to
    finish last silently won, discarding the loaded per-panel settings
    about half the time. Two independent callbacks, each fully
    responsible for building the grid *itself* for its own trigger, has
    no such race — there's nothing left to reconcile between them."""
    if not _extra_map_visible():
        return [], _EXTRA_MAP_SECTION_STYLE_HIDDEN
    return _extra_map_panels_grid(rows, cols, None, selected_item), _EXTRA_MAP_SECTION_STYLE


@app.callback(
    Output({"type": "extra-map-graph", "index": MATCH}, "figure"),
    Input({"type": "extra-map-type-dd", "index": MATCH}, "value"),
    Input({"type": "extra-map-scale-dd", "index": MATCH}, "value"),
    Input({"type": "extra-map-palette-dd", "index": MATCH}, "value"),
    Input({"type": "extra-map-vmin", "index": MATCH}, "value"),
    Input({"type": "extra-map-vmax", "index": MATCH}, "value"),
    State("selected-item", "data"),
    prevent_initial_call=True,
)
def update_extra_map_panel_figure(maptype, scale, palette, vmin, vmax, selected_item):
    """One panel's own figure, and *only* that panel's — MATCH means
    Dash dispatches one callback invocation per changed panel index,
    never touching the other N-1 panels' own outputs. `prevent_initial
    _call=True`: each panel's figure is already built once, correctly,
    at construction time (`_extra_map_panel_div`'s own `figure=` kwarg)
    — this only needs to react to a genuine *later* control change, not
    redundantly rebuild the same figure again the instant a panel first
    mounts. Reads only STATE.data/_resolve_maptype/_color_by_values
    (already in memory) — never touches a file, matching the explicit
    "avoid to load files every time for each [panel]" requirement."""
    if not _extra_map_visible() or maptype is None:
        return no_update
    return _build_extra_map_figure(maptype, scale, palette, vmin, vmax, selected_item)


@app.callback(
    Output("selected-item", "data", allow_duplicate=True),
    Input({"type": "extra-map-graph", "index": MATCH}, "clickData"),
    prevent_initial_call=True,
)
def on_extra_map_panel_click(click_data):
    """Finishes ifu_mod/mos_mod's own on_map_click — correct logic that
    existed with zero callers before this feature (see that function's
    own docstring). Plain 2D Scatter traces fire plotly_click reliably
    (the scatter3d-specific unreliability found in earlier rounds
    doesn't apply here), so no clientside workaround is needed for *this*
    direction — only the reverse (every panel reflecting a new shared
    selection) needs one, see extra-map-panels-selection-sync below."""
    if not click_data:
        return no_update
    if EXPLORER.kind == "ifu":
        return ifu_mod.on_map_click(click_data)
    if EXPLORER.kind == "mos":
        return mos_mod.on_map_click(click_data)
    return no_update


# Cross-panel selection sync — deliberately clientside, not a server
# -side figure rebuild for every panel on every click (see this
# feature's own module-level comment for the explicit efficiency
# requirement behind that choice). Each panel's own trace index 1 is
# *always* present (see spaxel_map_figure/source_map_figure's own
# comments on why — an empty x/y when nothing's selected, not a
# conditionally-omitted trace), so this can always Plotly.restyle()
# straight onto it, for every currently-rendered panel, with no server
# round trip at all. Each point's own [index, x, y] customdata triple
# (see both figure builders' own docstrings) is what makes this possible
# purely client-side — no separate coordinate lookup service needed.
app.clientside_callback(
    """
    function(selectedItem) {
        // Confirmed live: a .js-plotly-plot element's own "id" is always
        // empty -- Dash sets the real id on its *wrapper* div instead
        // (same established pattern used elsewhere in this file, e.g.
        // document.getElementById("flux-cube-graph").querySelector(
        // ".js-plotly-plot")) -- querying .js-plotly-plot.id directly
        // here meant this callback silently matched nothing at all,
        // ever, caught only by live testing (a real click correctly
        // updated the shared selection and the main map, but every
        // panel's own marker stayed frozen at whatever it was built
        // with), not by reading the diff.
        window._pyapsLastSelectedItem = selectedItem;

        function applySel(gd, item) {
            if (!gd || !gd.data || gd.data.length < 2) { return false; }
            var main = gd.data[0];
            var cd = main.customdata;
            if (!cd) { return false; }
            // Build (and cache on the graph element itself) an id -> [x,
            // y] Map once per distinct customdata array, instead of a
            // fresh O(n) linear scan through every point on *every*
            // single click -- explicit follow-up report: "the click on
            // the main panel become[s] really laggy and nonresponsive"
            // once several panels are open. The cache key is the
            // customdata array's own identity, so a genuine figure
            // rebuild (a new array) transparently invalidates it -- no
            // separate cache-clearing needed.
            if (gd._pyapsSelCd !== cd) {
                var m = new Map();
                for (var i = 0; i < cd.length; i++) {
                    if (cd[i]) { m.set(cd[i][0], [cd[i][1], cd[i][2]]); }
                }
                gd._pyapsSelMap = m;
                gd._pyapsSelCd = cd;
            }
            var xy = (item !== null && item !== undefined) ? gd._pyapsSelMap.get(item) : undefined;
            var x = xy ? xy[0] : null, y = xy ? xy[1] : null;
            Plotly.restyle(gd, {x: [x === null ? [] : [x]], y: [y === null ? [] : [y]]}, [1]);
            gd._pyapsSelApplied = item;
            return true;
        }
        var wrappers = Array.from(document.querySelectorAll('div[id*="extra-map-graph"]'));

        // Measured live: with the grid at its own max size (3 rows x 6
        // columns = 18 panels), a single click's Plotly.restyle() loop
        // over every panel blocked the main thread solid for the whole
        // batch -- Plotly.restyle() isn't free per call regardless of
        // how little data changes, and calling it 18 times synchronously
        // in a row (competing with the same selection change's other,
        // pre-existing callbacks -- a new spectrum plot, value tables,
        // the info box -- all sharing this one JS thread) is what
        // actually produced "the click on the main panel become[s]
        // really laggy and nonresponsive" once several panels are open,
        // and the broader "more memory management is needed... it seems
        // very laggy" ask. Fix: spread *this feature's own* restyle
        // calls across several animation frames, a handful of panels
        // each, instead of one long synchronous loop. This can't make
        // the underlying per-panel Plotly cost any smaller, and a
        // maxed-out 18-panel grid can still take a couple of seconds to
        // finish rippling through every panel while the rest of that
        // selection change's own work is also competing for the same
        // thread -- but the page keeps responding to input throughout
        // instead of freezing solid for the duration, which is what was
        // actually reported. Every panel is still updated eventually,
        // regardless of whether it's currently on-screen (a simpler,
        // safer trade-off than skipping off-screen panels for extra
        // speed — see this callback's git history for why that path,
        // once tried, was reverted: an off-screen panel could still be
        // mid-render from page load, and a one-shot catch-up could miss
        // it permanently).
        var CHUNK = 3, i = 0;
        function step() {
            var end = Math.min(i + CHUNK, wrappers.length);
            for (; i < end; i++) { applySel(wrappers[i].querySelector('.js-plotly-plot'), selectedItem); }
            if (i < wrappers.length) { requestAnimationFrame(step); }
        }
        if (wrappers.length) { step(); }
        return "";
    }
    """,
    Output("extra-map-selection-sync-dummy", "children"),
    Input("selected-item", "data"),
    prevent_initial_call=True,
)


app.clientside_callback(
    """
    function(nClicks) {
        if (!nClicks) { return window.dash_clientside.no_update; }
        // A real prompt at the moment of saving -- explicit follow-up:
        // "I want when we hit save[]... layout we see the option to
        // select a name; it is not good that the default name is [on]
        // the main page." Cancelling (null) means "do nothing" -- the
        // downstream server save callback is triggered by *this*
        // Store's own data changing, so a cancelled prompt (no_update,
        // the Store keeps its old value, no new change fires) correctly
        // never produces a download at all.
        var name = window.prompt("Save layout as:", "pyaps_layout.json");
        if (!name) { return window.dash_clientside.no_update; }
        return name;
    }
    """,
    Output("extra-map-save-filename-store", "data"),
    Input("extra-map-save-btn", "n_clicks"),
    prevent_initial_call=True,
)


@app.callback(
    Output("extra-map-download", "data"),
    Input("extra-map-save-filename-store", "data"),
    State("extra-map-rows-dd", "value"),
    State("extra-map-cols-dd", "value"),
    State({"type": "extra-map-type-dd", "index": ALL}, "value"),
    State({"type": "extra-map-scale-dd", "index": ALL}, "value"),
    State({"type": "extra-map-palette-dd", "index": ALL}, "value"),
    State({"type": "extra-map-vmin", "index": ALL}, "value"),
    State({"type": "extra-map-vmax", "index": ALL}, "value"),
    prevent_initial_call=True,
)
def on_extra_map_save(filename, rows, cols, maptypes, scales, palettes, vmins, vmaxs):
    """Saves the *view*, not the data — explicit request: "just say how
    many extra panels, in what grid, each presenting what table (not the
    data)... what parameter, what number of grids and how many, range
    for each, scale and pallete for each." `State(..., ALL)` here (not
    MATCH) is the one place genuinely reading every panel at once is
    correct — this fires once, on an explicit button click, not per
    panel-change.

    Triggered by `extra-map-save-filename-store`'s own data (the
    filename the user typed into the clientside `window.prompt()` just
    above), not the button's own `n_clicks` directly — the prompt can
    be cancelled, and a cancelled prompt correctly leaves this Store
    unchanged, so this callback simply never fires rather than needing
    its own "was it cancelled" check. Always ensures a real ".json"
    extension rather than trusting whatever the user typed."""
    if not filename:
        return no_update
    filename = filename.strip() or "pyaps_layout.json"
    if not filename.lower().endswith(".json"):
        filename += ".json"
    panels = [{"maptype": mt, "scale": sc, "palette": pl, "vmin": vn, "vmax": vx}
              for mt, sc, pl, vn, vx in zip(maptypes, scales, palettes, vmins, vmaxs)]
    payload = {"kind": EXPLORER.kind, "rows": rows, "cols": cols, "panels": panels}
    return dict(content=json.dumps(payload, indent=2), filename=filename)


def _validate_extra_map_layout(payload):
    """Real structural validation for an uploaded layout file — explicit
    follow-up: "make sure you have a format checker before acc[e]pting
    any json file as [a] layout file." Before this, "validation" was
    just whatever `payload["rows"]`/`int(...)` happened to raise on its
    own (a `KeyError`/`TypeError` for most malformed input, technically
    caught, but with no real check of the actual *shape* — e.g. a
    `"panels"` that's a string or a dict-of-dicts instead of a list
    would have quietly produced a broken grid rather than a clear
    error). Raises `ValueError` with a message meant to be shown
    directly to the user (see on_extra_map_load's own try/except) on any
    genuine problem; returns `(rows, cols, panels)` — never partially
    valid — on success. Deliberately does *not* require `"kind"` to
    match `EXPLORER.kind` — a saved layout describes a view, not one
    specific dataset (see _extra_map_panel_div's own docstring for the
    same "falls back gracefully" principle already applied per-panel)."""
    if not isinstance(payload, dict):
        raise ValueError("not a layout file (expected a JSON object)")
    if "rows" not in payload or "cols" not in payload:
        raise ValueError('missing "rows"/"cols"')
    try:
        rows = int(payload["rows"])
        cols = int(payload["cols"])
    except (TypeError, ValueError):
        raise ValueError('"rows"/"cols" must be whole numbers')
    if not (0 <= rows <= _EXTRA_MAP_MAX_ROWS and 0 <= cols <= _EXTRA_MAP_MAX_COLS):
        raise ValueError(f"rows/cols out of range (0-{_EXTRA_MAP_MAX_ROWS} rows, 0-{_EXTRA_MAP_MAX_COLS} cols)")
    panels = payload.get("panels", [])
    if not isinstance(panels, list):
        raise ValueError('"panels" must be a list')
    for i, p in enumerate(panels):
        if not isinstance(p, dict):
            raise ValueError(f"panel {i} is not a valid entry")
        for key in ("maptype", "scale", "palette"):
            if key in p and p[key] is not None and not isinstance(p[key], str):
                raise ValueError(f'panel {i}\'s "{key}" must be text')
        for key in ("vmin", "vmax"):
            if key in p and p[key] is not None and not isinstance(p[key], (int, float)):
                raise ValueError(f'panel {i}\'s "{key}" must be a number')
    return rows, cols, panels


@app.callback(
    Output("extra-map-panels-grid", "children", allow_duplicate=True),
    Output("extra-map-panels-section", "style", allow_duplicate=True),
    Output("extra-map-load-error", "children"),
    Input("extra-map-upload", "contents"),
    State("selected-item", "data"),
    prevent_initial_call=True,
)
def on_extra_map_load(contents, selected_item):
    """Parses an uploaded layout file and builds the grid directly
    itself — a separate callback from update_extra_map_panels_grid
    above (sharing its two main Outputs via `allow_duplicate=True`, a
    supported, deliberate Dash pattern for "two different triggers each
    independently produce the same output"), not a third Input feeding
    that one. See that callback's own docstring for the first real race
    this avoids: an earlier version drove Rows/Cols and a separate
    "pending config" Store from this one response, and having a
    *different* downstream callback react to all three Inputs together
    was confirmed live to silently drop the loaded settings about half
    the time.

    Deliberately does **not** also write `extra-map-rows-dd`/`-cols-dd`'s
    own "value" — confirmed live that doing so reintroduces the exact
    same race one level up: those two components are themselves Inputs
    to `update_extra_map_panels_grid` above, so setting them here fires
    that *other* callback as a side effect, which then overwrites this
    one's correctly-loaded grid with its own default-valued rebuild
    moments later (same underlying cause — two callbacks able to write
    the same Output, one now triggered as a side effect of the other's
    own Output write — just found on the second layer of this feature
    rather than the first). The Rows/Cols dropdowns showing their own
    previous values right after a Load (rather than the loaded shape)
    is a real, accepted cosmetic gap from this fix, not an oversight —
    the grid itself, which is what actually matters, is always correct.

    Malformed/foreign uploads show a small error and change nothing else
    — never crash the page over a bad file (see `_validate_extra_map_
    layout`'s own docstring — explicit follow-up: "make sure you have a
    format checker before acc[e]pting any json file as [a] layout
    file")."""
    if not contents:
        return no_update, no_update, ""
    try:
        _header, b64data = contents.split(",", 1)
        payload = json.loads(base64.b64decode(b64data).decode("utf-8"))
        rows, cols, panels = _validate_extra_map_layout(payload)
    except Exception as e:
        return no_update, no_update, f"⚠ Could not read layout file: {e}"
    if not _extra_map_visible():
        return [], _EXTRA_MAP_SECTION_STYLE_HIDDEN, ""
    grid = _extra_map_panels_grid(rows, cols, panels, selected_item)
    return grid, _EXTRA_MAP_SECTION_STYLE, ""


def explorer_worker(options=None):
    """CLI entry point: parse args, optionally load+route an initial
    dataset (L1 via --infiles/--infiles_list, L2 via --outpath/--headname
    — L1 takes priority if both are given), launch the Dash server (open
    the printed URL in a browser)."""
    parser = _build_arg_parser()
    args = parser.parse_args(options if options is not None else sys.argv[1:])

    if args.require_weaveor_auth:
        _auth.REQUIRE_WEAVEOR_AUTH = True
        if not _sess.MULTI_SESSION:
            print(">>> --require-weaveor-auth implies --multi-session — enabling it too.")
            _sess.MULTI_SESSION = True
        _auth.require_secret_configured()  # fail fast if the shared secret isn't set

    if args.multi_session:
        _sess.MULTI_SESSION = True

    if args.infiles is not None or args.infiles_list is not None:
        l1_args = l1_mod._build_arg_parser().parse_args([])
        l1_args.infiles = args.infiles
        l1_args.infiles_list = args.infiles_list
        l1_args.caldir = args.caldir
        l1_args.catdir = args.catdir
        l1_args.configdir = args.configdir
        _load_l1(l1_args)
    elif args.outpath and args.headname:
        _route_and_load_l2(args.outpath, args.headname)

    print(f"PyAPS Explorer — starting Dash server on port {args.port}")
    # threaded=True so the log-poll interval can keep updating the browser
    # while a long Load callback (single-threaded Flask dev server
    # otherwise) is still running server-side.
    app.run(debug=args.debug, port=args.port, host="::", threaded=True)


if __name__ == "__main__":
    explorer_worker()
