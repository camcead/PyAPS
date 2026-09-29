"""
Clickable RA/DEC fiber-map figure for aps_l1_preview — port of the
pyqtgraph `ScatterPlotItem` built in `L1_preview_interactive.run()`
(win1) in the legacy `aps_l1_preview.py`.

Framework-agnostic: returns a `plotly.graph_objects.Figure`. The caller
(a Dash callback) wires up `clickData` to drive fiber selection —
nearest-fiber lookup on click uses `closest_node()` below, ported
verbatim from the legacy file (plain numpy, no Qt dependency there
either).
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from . import style


def closest_node(ra, dec, ra_arr, dec_arr):
    """Index of the fiber nearest (ra, dec), correcting for spherical sky
    geometry (small-angle approximation with cos(dec) correction) — same
    logic as the legacy `L1_preview_interactive.closest_node`.
    """
    cos_dec = np.cos(np.radians(dec))
    delta_ra = (ra_arr - ra) * cos_dec
    delta_dec = dec_arr - dec
    dist2 = delta_ra ** 2 + delta_dec ** 2
    return int(np.argmin(dist2))


def fiber_map_figure(targ_ra, targ_dec, targ_flux, targ_use, aps_ids, *,
                      selected_aps_id=None, vmin=12.0, vmax=None,
                      title=None, width=650, height=600):
    """Build the RA/DEC fiber-selection map.

    Parameters
    ----------
    targ_ra, targ_dec : array
        Per-target sky position (deg).
    targ_flux : array
        Per-target integrated mean flux (summed across arms) — colours
        the marker, same "jet" colourscale/range as the legacy
        `custom_brush(val, "jet", 12.0, max_flux)`.
    targ_use : array[str]
        Per-target `TARGUSE` flag; `"T"` (science target) gets the
        hexagon marker, anything else (sky/calib) gets an X — matches
        legacy's `symbol="h"`/`"x"`, `size=6`/`10`.
    aps_ids : array[int]
        Per-target APS_ID, stored in `customdata` for click lookups.
    selected_aps_id : int, optional
        Currently-selected fiber — drawn as a highlighted ring on top.
    vmin, vmax : float, optional
        Colour range for `targ_flux` (default vmin=12.0 matches legacy;
        vmax defaults to `max(targ_flux)`).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    targ_ra = np.asarray(targ_ra, dtype=np.float64)
    targ_dec = np.asarray(targ_dec, dtype=np.float64)
    targ_flux = np.asarray(targ_flux, dtype=np.float64)
    aps_ids = np.asarray(aps_ids)
    is_target = np.array([str(u).strip().upper() == "T" for u in targ_use])
    vmax = float(np.nanmax(targ_flux)) if vmax is None else vmax

    # 1 degree of RA subtends cos(dec) degrees of true angular separation
    # — same spherical-projection correction applied to spatial X in the
    # IFU pipeline (aps_ifu_v0.py/aps_ifu_ExGal.py: `cube['x'] = -dx_deg *
    # 3600 * cos(dec)`). Plotting raw RA against Dec with a naive 1:1
    # aspect lock stretches the field in RA (X) by 1/cos(dec) — using
    # `scaleratio` here instead of pre-multiplying the data keeps RA/Dec
    # in the hover template as real sky coordinates.
    cos_dec = np.cos(np.radians(np.nanmean(targ_dec)))

    fig = go.Figure()

    for mask, symbol, size, name in (
        (is_target, "hexagon", 8, "Target fibres"),
        (~is_target, "x", 11, "Sky / calib fibres"),
    ):
        if not mask.any():
            continue
        # Plain SVG Scatter, not Scattergl/WebGL: this trace is the one
        # persistent, click-driven dcc.Graph in the app (every other
        # figure is rebuilt fresh inside a tab-content callback). WebGL
        # traces can have their canvas/context recreated on the first
        # Plotly.react() call after mount, silently dropping Dash's own
        # click-listener attachment — observed directly (Plotly's raw
        # plotly_click JS event still fired; Dash's clickData prop never
        # updated). Point count here (hundreds to a few thousand fibres)
        # is well within comfortable range for SVG anyway.
        #
        # customdata is passed as a plain Python list, not a numpy array:
        # Plotly's JSON encoder switches numpy arrays to a compact binary
        # "bdata"/"dtype" typed-array encoding, which renders and hovers
        # fine but — observed directly — Plotly.js's plotly_click event
        # payload sometimes omits customdata entirely for points from a
        # bdata-encoded trace (hovertemplate still resolves it correctly,
        # since that path reads the typed array directly rather than
        # round-tripping through the click event object). Plain lists
        # serialize as regular JSON arrays and don't hit that path.
        fig.add_trace(go.Scatter(
            x=targ_ra[mask], y=targ_dec[mask], mode="markers",
            marker=dict(
                symbol=symbol, size=size, color=targ_flux[mask],
                colorscale="Jet", cmin=vmin, cmax=vmax,
                showscale=mask is is_target,
                colorbar=dict(title="Flux", thickness=12) if mask is is_target else None,
                line=dict(width=0.5, color="rgba(0,0,0,0.4)"),
            ),
            customdata=aps_ids[mask].tolist(), name=name,
            hovertemplate="APS_ID=%{customdata}<br>RA=%{x:.5f}°  "
                           "Dec=%{y:.5f}°<br>flux=%{marker.color:.3g}<extra></extra>",
        ))

    if selected_aps_id is not None:
        sel = np.flatnonzero(aps_ids == selected_aps_id)
        if sel.size:
            i = sel[0]
            fig.add_trace(go.Scatter(
                x=[targ_ra[i]], y=[targ_dec[i]], mode="markers",
                marker=dict(symbol="circle-open", size=18, color="red", line=dict(width=2.5)),
                name="Selected", hoverinfo="skip", showlegend=False,
            ))

    # A static, explicit reversed range (max first, then min) gives the same
    # "RA increases to the left" convention as autorange="reversed" without
    # its side effect: autorange="reversed" stays permanently "on", so on
    # every rebuild (e.g. a fibre click elsewhere in the app triggers a
    # fresh Plotly.react() with a new Figure) Plotly recomputes the range
    # from scratch and discards any uirevision-preserved user zoom/pan. A
    # plain numeric range is just the *default* view — uirevision can still
    # override it with whatever the user last zoomed to.
    ra_min, ra_max = float(np.nanmin(targ_ra)), float(np.nanmax(targ_ra))
    ra_pad = (ra_max - ra_min) * 0.05 or 0.001

    base_layout = {k: v for k, v in style.BASE_LAYOUT.items() if k not in ("hovermode", "margin")}
    fig.update_layout(
        **base_layout,
        hovermode="closest",
        title=title,
        width=width, height=height,
        xaxis=dict(title="RA (deg)", range=[ra_max + ra_pad, ra_min - ra_pad]),
        yaxis=dict(title="Dec (deg)", scaleanchor="x", scaleratio=1.0 / cos_dec),
        margin=dict(l=60, r=30, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
    )
    return fig
