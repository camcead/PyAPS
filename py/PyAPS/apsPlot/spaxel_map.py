"""
Clickable IFU spatial-quantity scatter map for mapviewer — port of the
matplotlib `plotMap` scatter (`mapviewer/plotData.py`) in the legacy
`aps_Mapviewer.py`.

Framework-agnostic: returns a `plotly.graph_objects.Figure`. Unlike the
legacy version, this actually wires up per-quantity colourscale
selection (the equivalent `selectColormap` helper existed in the legacy
code but was never called — every map used a hardcoded 'viridis').
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from . import style


def select_colorscale(maptype: str) -> str:
    """Per-quantity colourscale, matching the *intent* of legacy's
    unused `selectColormap` helper (Plotly colourscale names, not
    matplotlib's — `RdBu_r`/`RdYlBu_r`/`Viridis` all exist natively in
    Plotly under the same names; `tab10` has no Plotly equivalent, so
    categorical maps (BIN_ID) fall back to a cyclical-looking `Turbo`).
    """
    if any(x in maptype for x in ("V_", "PPXF_V", "V_SFH", "V_Dif", "VRAD")):
        return "RdBu_r"
    if any(x in maptype for x in ("H3", "H4", "H5", "H6")):
        return "RdBu_r"
    if any(x in maptype for x in ("SIGMA_", "PPXF_SIGMA", "FLUX_", "AMPL_", "AON_", "AGE", "SNR", "NSPAX")):
        return "Viridis"
    if any(x in maptype for x in ("METALS", "ALPHA", "LS_", "FEH")):
        return "RdYlBu_r"
    if any(x in maptype for x in ("DETECTION", "COMPONENT", "TIE_GROUP", "BIN_ID")):
        return "Turbo"
    return "Inferno"


def spaxel_map_figure(x, y, data, indices, *, title=None, colorbar_label="",
                       colorscale=None, selected_xy=None, marker_color="red",
                       vmin=None, vmax=None, scale="linear", width=650, height=600):
    """Build the clickable IFU spatial scatter map.

    Parameters
    ----------
    x, y : array
        Per-spaxel/bin position (arcsec).
    data : array
        The quantity being mapped (already resolved by the caller —
        this function doesn't know about maptype strings).
    indices : array[int]
        Per-point index into the caller's spaxel/bin arrays — stored as
        `customdata` (as a plain list, not a numpy array: Plotly's
        binary "bdata" typed-array encoding for numpy customdata can
        drop it from the `plotly_click` event payload even though hover
        renders it correctly — see apsPlot.fiber_map for the same fix).
    title, colorbar_label : str
    colorscale : str, optional
        Defaults to `select_colorscale` applied to `title` if not given.
    selected_xy : tuple[float, float], optional
        (x, y) position of the currently-selected bin's centroid —
        drawn as a highlighted marker on top (matching legacy's
        `binMarker`, plotted at the bin's `XBIN`/`YBIN`, not an
        arbitrary member spaxel's raw `X`/`Y`).
    marker_color : str
        Colour for the selection marker (legacy's configurable
        `self.markercolor`, default red).
    scale : str
        One of `apsPlot.style.COLOR_SCALE_OPTIONS` ("linear"/"log"/
        "sqrt"/"power"/"asinh") — added for the multi-panel L2 map
        feature's own per-panel "Scale" control (this function had no
        stretch option at all before). Defaults to "linear", which is a
        complete no-op through `style.stretch_for_marker_color` — every
        pre-existing call site (none currently pass `scale` at all)
        keeps its exact previous behaviour unchanged. See that helper's
        own docstring for why the hover template below reads the
        original value from `text=`, not `marker.color`, once a
        non-linear scale is in play.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    data = np.asarray(data, dtype=np.float64)
    indices = np.asarray(indices)

    finite = np.isfinite(data)
    x, y, data, indices = x[finite], y[finite], data[finite], indices[finite]

    if colorscale is None:
        colorscale = select_colorscale(title or "")
    if vmin is None or vmax is None:
        lo, hi = np.percentile(data, [1, 99]) if data.size else (0, 1)
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax
    if colorscale == "RdBu_r" and vmin < 0 < vmax:
        # keep diverging maps centred on zero
        m = max(abs(vmin), abs(vmax))
        vmin, vmax = -m, m

    color_values, cmin, cmax = style.stretch_for_marker_color(data, vmin, vmax, scale)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(
            size=7, color=color_values, colorscale=colorscale, cmin=cmin, cmax=cmax,
            showscale=True, colorbar=dict(title=colorbar_label, thickness=12),
            line=dict(width=0),
        ),
        # [index, x, y] per point, not a plain index list — a plain
        # Python list of lists (not a numpy array — see this parameter's
        # own docstring above for why that distinction matters for
        # click-safety), so a client-side click-to-select handler can
        # still read customdata[0] as the id exactly as before, *and* the
        # multi-panel L2 map feature's own cross-panel-selection
        # clientside restyle (aps_explorer.py's
        # extra-map-panels-selection-sync) can look a selected id's own
        # x/y back up from customdata[1]/[2] without any server round
        # trip — this figure's own x/y arrays aren't otherwise reachable
        # from plain client-side JS at all.
        customdata=[[int(i), float(xi), float(yi)] for i, xi, yi in zip(indices, x, y)],
        # text=, not marker.color, for the real (un-stretched) value —
        # see stretch_for_marker_color's own docstring for why.
        text=data,
        hovertemplate="x=%{x:.2f}″  y=%{y:.2f}″<br>value=%{text:.4g}<extra></extra>",
        name="",
    ))

    # Always added as trace index 1, even with nothing selected yet (an
    # empty x/y) — not just when selected_xy is given. The multi-panel L2
    # map feature's own cross-panel selection sync (see aps_explorer.py's
    # extra-map-panels-selection-sync clientside callback) needs a
    # trace index it can rely on always existing to Plotly.restyle() the
    # highlight onto, cheaply, without a full figure rebuild on every
    # click elsewhere — conditionally adding this trace only when there's
    # already a selection would leave that restyle with nothing to target
    # for a panel's very first selection.
    sel_x = [selected_xy[0]] if selected_xy is not None else []
    sel_y = [selected_xy[1]] if selected_xy is not None else []
    fig.add_trace(go.Scatter(
        x=sel_x, y=sel_y, mode="markers",
        # size=10, line width=2 (was 14/3) -- explicit follow-up report:
        # "the cross (x) sign on the additional maps is really large...
        # no need to be that large" -- this trace's original size was
        # tuned for this function's own 650x600px design, never actually
        # rendered anywhere until the multi-panel L2 map feature's own,
        # much smaller panels started using it.
        marker=dict(symbol="x", size=10, color=marker_color, line=dict(width=2)),
        name="Selected", hoverinfo="skip", showlegend=False,
    ))

    base_layout = {k: v for k, v in style.BASE_LAYOUT.items() if k not in ("hovermode", "margin")}
    fig.update_layout(
        **base_layout,
        hovermode="closest",
        # Left-aligned (x=0.02, xanchor="left"), not Plotly's own default
        # centred title — confirmed live (real pixel bounding-box
        # comparison, not visual guessing) that a centred title sits
        # *inside* the modebar's own horizontal span (which always
        # occupies the top-right, regardless of the plot's own width),
        # not just visually close to it — increasing top margin alone
        # doesn't fix this, since the modebar and a centred title share
        # the same vertical band too. The modebar is always top-right,
        # so a left-aligned title can never collide with it regardless
        # of how narrow the figure is (the multi-panel L2 map feature's
        # own panels are exactly this — much narrower than this
        # function's original 650px design ever needed to account for).
        title=dict(text=title, x=0.02, xanchor="left", y=0.97, yanchor="top"),
        height=height,
        # width=None (the multi-panel L2 map feature's own per-panel
        # call, see that feature's own comment on this) means "fill
        # whatever width the container actually is" via autosize=True,
        # not a fixed pixel width — explicit follow-up report: "the plot
        # is left aligned but the map selection bar fill[s] the whole
        # width of that panel." A real, non-None width (every other,
        # dead-code call site) keeps its exact previous fixed-size
        # behaviour untouched.
        **({"autosize": True} if width is None else {"width": width}),
        xaxis=dict(title="X [arcsec]"),
        yaxis=dict(title="Y [arcsec]", scaleanchor="x"),
        # t=70 (was 50) -- explicit follow-up report: "the title of each
        # additional map[] is overlap[ping] the plotly default bottons
        # and option[s]." Narrower panels put the (centred) title and
        # the modebar's own top-right icons closer together than the
        # original fixed-650px design ever needed to account for; more
        # top margin gives both their own clear space regardless of the
        # figure's actual width.
        margin=dict(l=60, r=90, t=70, b=40),
    )
    return fig
