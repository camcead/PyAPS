"""
Slit Explorer — a compact, wide strip showing every fibre's position on
the physical spectrograph *slit* (its `NSPEC` value), not its position on
the sky/in the field (`APS_ID`). These are two independent numbering
schemes: two fibres next to each other on the sky are not necessarily
anywhere near each other on the slit, and vice versa (confirmed on real
data — `NSPEC` and `APS_ID` are literally uncorrelated permutations of
1..N for the same fibre set). A fibre sitting on the slit right next to an
unusually bright one can pick up scattered-light contamination that's
otherwise invisible from its APS_ID/sky position alone — this panel exists
to make that adjacency visible at a glance.

Only meaningful for single-exposure, fibre-level L1 data (MOS/MOSLIFU/
MOSMIFU) — a stacked/co-added IFU cube (plain LIFU/MIFU) has no single
NSPEC per spatial position (each position is built from potentially many
different fibres across dithered exposures), so the caller is expected to
gate on that before building this figure at all; this module itself has
no opinion about it (framework-agnostic, plain arrays in, `go.Figure` out,
same convention as `fiber_map.py`/`spectra.py`).
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from . import style

# A real file can have 600-1000+ fibres — all of them in one view meant
# every dot was just a few pixels apart, colours blurred together, and
# clicking a specific one was hit-or-miss (direct user report). Rather
# than showing everything at once, the default view is a fixed-width
# window centred on the current selection (originally spelled out as an
# exact width/behaviour: "always show only max 100 points per zoom... so
# when I click nspec=120 it shows from 120-50 to 120+50" — then narrowed
# further, 100 points down to 50, once uniform-size square markers still
# showed visible edge-to-edge overlap at the 100-wide window on a dense
# real file) — the *data* isn't filtered, every fibre is still in the
# trace and reachable by panning, only the *initial camera* is windowed.
WINDOW_HALF_WIDTH = 25

# Uniform marker size for every point (see the comment above the marker
# trace itself for why this is fixed rather than value-scaled).
MARKER_SIZE = 14


def slit_explorer_figure(nspec, aps_ids, values, *, vmin, vmax, colorscale,
                          scale="linear", selected_aps_id=None, status=None,
                          targuse=None, value_label="Flux", height=130):
    """Build the Slit Explorer strip.

    Parameters
    ----------
    nspec : array
        Per-fibre position on the spectrograph slit (`NSPEC`). Fibres with
        a non-finite value are dropped (no slit position on record).
    aps_ids : array[int]
        Per-fibre APS_ID — stored in `customdata` for click lookups, same
        convention as `fiber_map.fiber_map_figure`.
    values : array
        Per-fibre quantity driving marker colour (total flux by default —
        whatever the caller's current Aladin colour-by is, so the two
        views always agree). Colour-only, uniform size: an earlier version
        also scaled marker *size* with this same value, but live testing
        found that made things read worse, not better — a bigger circle
        looks like it's sitting between its same-size neighbours rather
        than in line with them. Squares at one fixed size sit flush
        against each other regardless of colour, which is what the marker
        trace below actually renders.
    vmin, vmax, colorscale, scale : as in `_bucket_by_color`/
        `apply_color_scale` — the exact colour range/stretch currently in
        effect elsewhere in the app, passed straight through so this strip
        never disagrees with the Aladin overlay's own colouring.
    selected_aps_id : int, optional
        Currently-selected fibre — drawn as a dashed vertical guide line
        plus a marker directly on the slit axis, so it's immediately clear
        where the current selection sits relative to its slit neighbours.
        Also recentres the default (pannable) view on this fibre's own
        NSPEC, +/- WINDOW_HALF_WIDTH.
    status : array[str], optional
        Per-fibre `FIB_STATUS` (WEAVE convention: `'A'` = active/usable;
        anything else is some kind of non-active fibre) — non-active
        fibres get a warning-coloured marker border so a problem fibre
        stands out without needing to hover every point.
    targuse : array[str], optional
        Per-fibre `TARGUSE` (`T`/`S`/`C`/blank) — shown in hover text only.
    value_label : str
        Label for `values` in hover text (e.g. "Flux" or "S/N").
    height : int
        Total figure height in px — deliberately small ("very narrow"),
        this is an at-a-glance overview strip, not a full plot.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    nspec = np.asarray(nspec, dtype=np.float64)
    aps_ids = np.asarray(aps_ids)
    values = np.asarray(values, dtype=np.float64)
    n = len(nspec)
    status = np.asarray(status if status is not None else [""] * n)
    targuse = np.asarray(targuse if targuse is not None else [""] * n)

    finite = np.isfinite(nspec)
    fig = go.Figure()

    if finite.any():
        span = (vmax - vmin) or 1.0
        frac = np.clip((values - vmin) / span, 0.0, 1.0)
        frac = style.apply_color_scale(frac, scale)

        is_active = np.array([str(s).strip().upper() == "A" for s in status])
        # Reverted to the original semi-transparent black — see
        # apsPlot.style.DATA_COLOR's own comment for why dark-mode
        # trace/marker-colour visibility now lives client-side
        # (explorer_ui.js's themePlotlyFigures) instead of here, kept
        # consistent across every figure builder even though this one
        # specifically has no static/pipeline consumer of its own.
        border_color = np.where(finite & ~is_active, "#d62728", "rgba(0,0,0,0.35)")
        border_width = np.where(finite & ~is_active, 1.8, 0.5)

        customdata = np.stack([
            aps_ids.astype(object), values, status.astype(object), targuse.astype(object),
        ], axis=1)

        fig.add_trace(go.Scatter(
            x=nspec[finite], y=np.zeros(int(finite.sum())), mode="markers",
            # Square, one uniform size, colour-only — a first version also
            # scaled marker *size* with the same value, which turned out
            # to look worse in practice, not better (direct live-testing
            # report): a bigger circle visually reads as sitting slightly
            # apart from/between its same-size neighbours rather than
            # cleanly in line with them, since a differently-sized round
            # marker's edges land at different points along the shared
            # baseline. Uniform squares sit flush against each other
            # regardless of colour, so only the colour difference reads —
            # explicit user request ("I want them all the same size next
            # to each other squares but with different colors").
            marker=dict(
                symbol="square", size=MARKER_SIZE, color=frac[finite],
                colorscale=colorscale, cmin=0, cmax=1,
                line=dict(width=border_width[finite].tolist(), color=border_color[finite].tolist()),
            ),
            customdata=customdata[finite].tolist(),
            hovertemplate=(
                "APS_ID=%{customdata[0]}<br>NSPEC (slit pos)=%{x:.0f}<br>"
                f"{value_label}=" + "%{customdata[1]:.3g}<br>"
                "status=%{customdata[2]}  use=%{customdata[3]}<extra></extra>"
            ),
            showlegend=False,
        ))

    x_min = float(np.nanmin(nspec[finite])) if finite.any() else 0.0
    x_max = float(np.nanmax(nspec[finite])) if finite.any() else 1.0

    sel_x = None
    if selected_aps_id is not None:
        sel = np.flatnonzero(aps_ids == selected_aps_id)
        if sel.size and np.isfinite(nspec[sel[0]]):
            sel_x = float(nspec[sel[0]])
            fig.add_vline(x=sel_x, line_width=2, line_dash="dash", line_color="#d62728")
            fig.add_trace(go.Scatter(
                x=[sel_x], y=[0], mode="markers",
                marker=dict(symbol="triangle-down", size=13, color="#d62728"),
                hoverinfo="skip", showlegend=False,
            ))

    # Default (initial) view: a fixed WINDOW_HALF_WIDTH-either-side window
    # centred on the current selection, or the full-range midpoint if
    # nothing's selected yet — clamped to the real data span so the window
    # doesn't show mostly empty space at either end of the slit. The axis
    # itself stays pannable (fixedrange=False) so the rest of the slit is
    # still reachable; it's the *zoom level* that's fixed (see the config
    # this figure is rendered with in aps_explorer.py — scrollZoom off,
    # box/lasso-zoom modebar buttons removed, drag mode "pan" — a
    # deliberate, explicit user request: "probably better if I cannot zoom
    # in zoom out... so a fixed zoom is fine").
    center = sel_x if sel_x is not None else (x_min + x_max) / 2.0
    view_min = center - WINDOW_HALF_WIDTH
    view_max = center + WINDOW_HALF_WIDTH
    # Slide the window back on-span if the centre sits near either edge,
    # rather than clamping each side independently (which would silently
    # shrink the window's width near the edges instead of just sliding it).
    span_full = (x_max + 1) - (x_min - 1)
    if span_full > 2 * WINDOW_HALF_WIDTH:
        if view_min < x_min - 1:
            shift = (x_min - 1) - view_min
            view_min += shift
            view_max += shift
        elif view_max > x_max + 1:
            shift = view_max - (x_max + 1)
            view_min -= shift
            view_max -= shift
    else:
        view_min, view_max = x_min - 1, x_max + 1

    fig.update_layout(
        template="plotly_white",
        font=dict(family=style.FONT_FAMILY, size=11),
        height=height,
        margin=dict(l=45, r=15, t=8, b=26),
        dragmode="pan",
        xaxis=dict(title="Slit position (NSPEC) — drag to scroll, scroll-wheel/zoom disabled",
                   range=[view_min, view_max], showgrid=False, zeroline=False, fixedrange=False),
        yaxis=dict(visible=False, range=[-1, 1], fixedrange=True),
        showlegend=False,
    )
    return fig
