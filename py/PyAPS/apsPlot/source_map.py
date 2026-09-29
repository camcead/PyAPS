"""
Clickable MOS sky-position scatter map, coloured by a continuous
quantity — the MOS counterpart of `apsPlot/spaxel_map.py`'s
`spaxel_map_figure`, built for the multi-panel L2 map feature (IFU
already had a reusable continuous-colour Plotly map; MOS only had a
categorical one, `aps_MOSviewer.py`'s `_build_source_map_figure`, with
no vmin/vmax/colorbar/scale support at all).

Framework-agnostic: returns a `plotly.graph_objects.Figure`. Deliberately
mirrors `spaxel_map_figure`'s own parameter names/shapes (`x, y, data,
indices, *, title, colorbar_label, colorscale, selected_xy, vmin, vmax,
scale`) so both viewers' per-panel callback wiring in aps_explorer.py can
share the same call shape, differing only in `x, y` being RA/Dec here
instead of arcsec spaxel offsets.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from . import style


def source_map_figure(ra, dec, data, indices, *, title=None, colorbar_label="",
                       colorscale="Viridis", selected_radec=None, marker_color="red",
                       vmin=None, vmax=None, scale="linear", width=650, height=600):
    """Build the clickable MOS sky-position scatter map.

    Parameters
    ----------
    ra, dec : array
        Per-target sky position (degrees).
    data : array
        The quantity being mapped (already resolved by the caller, e.g.
        via `aps_MOSviewer._color_by_values` — this function doesn't
        know about column names).
    indices : array[int]
        Per-point index into the caller's own target arrays (typically
        APS_ID) — stored as `customdata` (a plain list, not a numpy
        array — see `spaxel_map_figure`'s own docstring for why: Plotly's
        binary "bdata" typed-array encoding can silently drop customdata
        from the `plotly_click` event payload even though hover renders
        it correctly).
    title, colorbar_label : str
    colorscale : str
        No `select_colorscale`-style auto-pick here (unlike
        `spaxel_map_figure`) — MOS's own quantities are already a short,
        fixed list (`COLOR_BY_OPTIONS`, redshift/S-N), each with its own
        established palette pick made by the caller already; still
        overridable per panel, same as IFU's own per-panel palette
        control.
    selected_radec : tuple[float, float], optional
        (ra, dec) of the currently-selected target — drawn as a
        highlighted marker on top, matching `spaxel_map_figure`'s own
        `selected_xy` convention (and `_build_source_map_figure`'s own
        existing "Selected" marker style, reused here).
    marker_color : str
        Colour for the selection marker.
    scale : str
        One of `apsPlot.style.COLOR_SCALE_OPTIONS` — see
        `style.stretch_for_marker_color`'s own docstring. "linear" is a
        complete no-op, matching `spaxel_map_figure`'s own default.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    ra = np.asarray(ra, dtype=np.float64)
    dec = np.asarray(dec, dtype=np.float64)
    data = np.asarray(data, dtype=np.float64)
    indices = np.asarray(indices)

    finite = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(data)
    ra, dec, data, indices = ra[finite], dec[finite], data[finite], indices[finite]

    if vmin is None or vmax is None:
        lo, hi = np.percentile(data, [1, 99]) if data.size else (0, 1)
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax
    if colorscale == "RdBu_r" and vmin < 0 < vmax:
        # keep diverging maps centred on zero — same convention as
        # spaxel_map_figure's own identical check.
        m = max(abs(vmin), abs(vmax))
        vmin, vmax = -m, m

    color_values, cmin, cmax = style.stretch_for_marker_color(data, vmin, vmax, scale)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ra, y=dec, mode="markers",
        marker=dict(
            size=7, color=color_values, colorscale=colorscale, cmin=cmin, cmax=cmax,
            showscale=True, colorbar=dict(title=colorbar_label, thickness=12),
            line=dict(width=0.5, color="rgba(0,0,0,0.3)"),
        ),
        # [index, ra, dec] per point — see spaxel_map_figure's identical
        # convention/docstring for why (client-side selection-sync needs
        # to look a selected id's own position back up with no server
        # round trip; click-to-select still reads customdata[0]).
        customdata=[[int(i), float(r), float(d)] for i, r, d in zip(indices, ra, dec)],
        # text=, not marker.color, for the real (un-stretched) value —
        # see style.stretch_for_marker_color's own docstring for why.
        text=data,
        hovertemplate="RA=%{x:.5f}°  Dec=%{y:.5f}°<br>value=%{text:.4g}<extra></extra>",
        name="",
    ))

    # Always added as trace index 1, even with nothing selected yet — see
    # spaxel_map_figure's own identical comment for why (the multi-panel
    # L2 map feature's cross-panel selection sync needs a stable trace
    # index to Plotly.restyle() onto, cheaply, without a full rebuild).
    sel_x = [selected_radec[0]] if selected_radec is not None else []
    sel_y = [selected_radec[1]] if selected_radec is not None else []
    fig.add_trace(go.Scatter(
        x=sel_x, y=sel_y, mode="markers",
        # size=11, line width=2 (was 16/2.5) -- same scale-down as
        # spaxel_map_figure's own identical marker (explicit follow-up:
        # "the cross... sign on the additional maps is really large"),
        # for the same reason: tuned for this function's own 650x600px
        # design, not the multi-panel feature's much smaller panels.
        marker=dict(symbol="circle-open", size=11, color=marker_color, line=dict(width=2)),
        name="Selected", hoverinfo="skip", showlegend=False,
    ))

    # cos(dec) scaling + reversed RA range — identical convention to
    # aps_MOSviewer.py's own _build_source_map_figure, so this panel's
    # sky orientation always matches the main map's.
    cos_dec = np.cos(np.radians(np.nanmean(dec))) if dec.size else 1.0
    if ra.size:
        ra_min, ra_max = float(np.nanmin(ra)), float(np.nanmax(ra))
    else:
        ra_min, ra_max = 0.0, 1.0
    ra_pad = (ra_max - ra_min) * 0.05 or 0.001

    base_layout = {k: v for k, v in style.BASE_LAYOUT.items() if k not in ("hovermode", "margin")}
    fig.update_layout(
        **base_layout,
        hovermode="closest",
        # Left-aligned — see spaxel_map_figure's own identical comment
        # for why (confirmed live: a centred title sits inside the
        # modebar's own always-top-right span, not just near it).
        title=dict(text=title, x=0.02, xanchor="left", y=0.97, yanchor="top"),
        height=height,
        # width=None means "fill the container" (autosize) — see
        # spaxel_map_figure's own identical comment for the full why.
        **({"autosize": True} if width is None else {"width": width}),
        xaxis=dict(title="RA (deg)", range=[ra_max + ra_pad, ra_min - ra_pad]),
        yaxis=dict(title="Dec (deg)", scaleanchor="x", scaleratio=1.0 / cos_dec if cos_dec else 1.0),
        # t=70 (was 50) -- see spaxel_map_figure's own identical comment
        # (panel title/modebar overlap report).
        margin=dict(l=60, r=90, t=70, b=40),
    )
    return fig
