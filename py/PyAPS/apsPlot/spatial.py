"""
Generalized 2D spatial "parameter map grid" figure builder — the IFU
counterpart to spectra.spectrum_overlay_figure.

Shared shape across IFU's spatial diagnostics: stellar/gas kinematics
(V/Sigma/h3/h4, IFUExGalPlot_kinematics) and line-strength index maps
(IFUExGalPlot_ls) — both rasterize per-bin values onto a spaxel grid from
bin-table X/Y/FLUX/BIN_ID columns and display the result as a grid of
independently-scaled heatmaps. Each module's own `viz.<module>` wrapper
adapts its result table into the primitives below; this function itself
knows nothing about PPXF/SFH/line-strength.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import style


def rasterize(x, y, val, pixelsize, pad=6):
    """Bin scattered (x, y, val) triples onto a regular pixel grid — the
    same nearest-pixel binning IFUExGalPlot_kinematics/_ls use.

    Returns (image, extent): `image` is a 2D array (ny, nx), NaN where no
    bin lands; `extent` is (xmin, xmax, ymin, ymax), padded by half a
    pixel on each side, in the same units as x/y.
    """
    xmin, xmax = np.nanmin(x) - pad, np.nanmax(x) + pad
    ymin, ymax = np.nanmin(y) - pad, np.nanmax(y) + pad
    nx = int(np.round((xmax - xmin) / pixelsize)) + 1
    ny = int(np.round((ymax - ymin) / pixelsize)) + 1
    i = np.round((x - xmin) / pixelsize).astype(np.int64)
    j = np.round((y - ymin) / pixelsize).astype(np.int64)
    image = np.full((nx, ny), np.nan)
    image[i, j] = val
    return image.T, (xmin - pixelsize / 2, xmax + pixelsize / 2,
                      ymin - pixelsize / 2, ymax + pixelsize / 2)


def parameter_map_grid_figure(
    labels,
    values,
    x,
    y,
    pixelsize,
    *,
    nrows=None,
    ncols=None,
    vminmax=None,
    value_format=".2f",
    colorscale="inferno",
    invert_x=True,
    pad=6,
    figure_title=None,
    height_per_row=420,
    width_per_col=420,
):
    """Build a grid of independently-scaled 2D spatial parameter maps.

    Parameters
    ----------
    labels : sequence[str]
        One label per panel, e.g. ["V", "Sigma", "h3", "h4"].
    values : sequence[array]
        One 1D array per panel, aligned with `x`/`y` — per-spaxel,
        already expanded from per-bin via BIN_ID (the same "long" array
        the legacy plotter builds).
    x, y : array
        Per-spaxel spatial coordinates (arcsec offsets), shared by every
        panel.
    pixelsize : float
        Spaxel size, same units as x/y.
    nrows, ncols : int, optional
        Grid layout. Defaults to a single row of len(labels) panels.
    vminmax : sequence[(float, float)], optional
        Per-panel (vmin, vmax) override; defaults to that panel's own
        nanmin/nanmax, matching the legacy pipeline-mode behaviour.
    value_format : str or sequence[str], optional
        Format spec (Python format-mini-language) for the vmin/vmax
        corner label and hover text. A single string applies to every
        panel; pass a per-panel list when panels need different
        precision (legacy uses "{:.0f}" for V/Sigma, "{:.2f}" for h3/h4).
    colorscale : str, optional
        Plotly colorscale name (default matches legacy's matplotlib
        'inferno').
    invert_x : bool, optional
        Flip the x-axis (RA increases to the left — astronomy
        convention), matching the legacy plot.
    pad : float, optional
        Padding (same units as x/y) added around the data extent before
        rasterizing (legacy uses 6 for kinematics maps, 5 for
        line-strength maps).
    figure_title : str, optional
        Overall figure title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    n = len(labels)
    if ncols is None:
        ncols = n if nrows is None else int(np.ceil(n / nrows))
    if nrows is None:
        nrows = int(np.ceil(n / ncols))

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    fig = make_subplots(rows=nrows, cols=ncols,
                         horizontal_spacing=0.02, vertical_spacing=0.05)

    formats = value_format if isinstance(value_format, (list, tuple)) else [value_format] * n

    for k, (label, val) in enumerate(zip(labels, values)):
        row = k // ncols + 1
        col = k % ncols + 1
        val = np.asarray(val, dtype=np.float64)
        vfmt = formats[k]
        image, (xmin, xmax, ymin, ymax) = rasterize(x, y, val, pixelsize, pad=pad)

        if vminmax is not None:
            vmin, vmax = vminmax[k]
        else:
            valid = val[np.isfinite(val)]
            vmin, vmax = (float(np.nanmin(valid)), float(np.nanmax(valid))) if valid.size else (0.0, 1.0)

        axis_num = k + 1
        xref_id = "x" if axis_num == 1 else f"x{axis_num}"

        fig.add_trace(
            go.Heatmap(
                z=image,
                x=np.linspace(xmin, xmax, image.shape[1]),
                y=np.linspace(ymin, ymax, image.shape[0]),
                zmin=vmin, zmax=vmax,
                colorscale=colorscale,
                colorbar=dict(
                    len=0.85 / nrows, thickness=12,
                    x=1.03 + (col - 1) * 0.09, y=1 - (row - 0.5) / nrows,
                ),
                hovertemplate=(f"{label}: %{{z:{vfmt}}}<br>"
                                "Δα=%{x:.1f}\"<br>Δδ=%{y:.1f}\"<extra></extra>"),
            ),
            row=row, col=col,
        )
        fig.add_annotation(
            text=label, xref="x domain", yref="y domain",
            x=0.02, y=0.98, showarrow=False, xanchor="left", yanchor="top",
            font=dict(size=13, color="black"),
            row=row, col=col,
        )
        fig.add_annotation(
            text=f"{vmin:{vfmt}} / {vmax:{vfmt}}",
            xref="x domain", yref="y domain",
            x=0.98, y=0.02, showarrow=False, xanchor="right", yanchor="bottom",
            font=dict(size=11, color="black"),
            row=row, col=col,
        )
        if invert_x:
            fig.update_xaxes(autorange="reversed", row=row, col=col)
        fig.update_yaxes(scaleanchor=xref_id, scaleratio=1, row=row, col=col)
        if col == 1:
            fig.update_yaxes(title_text="Δδ [arcsec]", row=row, col=col)
        if row == nrows:
            fig.update_xaxes(title_text="Δα [arcsec]", row=row, col=col)

    # Extra right margin for the ncols colorbars stacked outside the plot
    # area (see the per-trace colorbar x offsets above) — otherwise later
    # columns' colorbars get clipped off the canvas edge.
    base_margin = dict(style.BASE_LAYOUT["margin"])
    base_layout = {k: v for k, v in style.BASE_LAYOUT.items() if k != "margin"}
    base_margin["r"] = base_margin.get("r", 30) + 70 * ncols

    fig.update_layout(
        **base_layout,
        margin=base_margin,
        title=figure_title,
        height=height_per_row * nrows,
        width=width_per_col * ncols + 70 * ncols,
    )
    return fig
