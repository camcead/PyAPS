"""
WCS-aware 2D image figures (sky images with RA/Dec-labeled axes and
region/ellipse overlays), built directly on Plotly.

Plotly has no equivalent to astropy's WCSAxes (angular gridlines, sky
projections drawn natively). Per design decision: images are rendered in
plain pixel space (fast, fully interactive pan/zoom/hover), with tick
labels and hover text computed from the WCS object instead of true
angular gridlines. Tick labels use one row/column of pixels to establish
the RA/Dec-per-pixel mapping — exact for the common case (no image
rotation, i.e. a diagonal CD/PC matrix, true for WEAVE's white-light
images), an approximation for a rotated WCS.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from . import style


def _tick_positions(n, n_ticks=6):
    """Evenly spaced pixel indices for axis ticks, including both ends."""
    if n <= 1:
        return np.array([0])
    return np.unique(np.round(np.linspace(0, n - 1, min(n_ticks, n))).astype(int))


def add_wcs_image(fig, row, col, data, wcs, *, colorscale="gray", log=False,
                   zmin=None, zmax=None, title=None, colorbar=None, cbar_label=""):
    """Add a WCS-aware image panel: a Heatmap in pixel space, with RA/Dec
    tick labels computed from `wcs`, and per-pixel RA/Dec available on
    hover. `data` is a 2D array indexed [y, x] (FITS/numpy convention).

    `colorbar`, if given, is a fully-formed Plotly colorbar dict (e.g.
    from a caller-side layout helper that knows how many colorbar-bearing
    panels share the figure) — this function doesn't guess its own
    position, since a per-call guess is exactly what produced the
    overlapping-colorbar bug fixed in viz.exgal_prepare.
    """
    data = np.asarray(data, dtype=np.float64)
    ny, nx = data.shape
    z = np.log10(np.clip(data, np.nanpercentile(data[np.isfinite(data)], 0.5)
                          if np.any(np.isfinite(data)) else 1e-10, None)) if log else data

    xt = _tick_positions(nx)
    yt = _tick_positions(ny)
    # world_to_array_index inverse: use pixel_to_world for tick labels —
    # row/column 0 (or the image centre row/col) establishes the RA/Dec
    # per pixel mapping; exact when the WCS has no rotation term.
    xt_world = wcs.pixel_to_world(xt, np.full_like(xt, ny // 2))
    yt_world = wcs.pixel_to_world(np.full_like(yt, nx // 2), yt)
    xt_labels = [f"{c.ra.deg:.4f}" for c in xt_world]
    yt_labels = [f"{c.dec.deg:.4f}" for c in yt_world]

    # Per-pixel RA/Dec for hover (accurate regardless of rotation).
    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))
    world = wcs.pixel_to_world(xx.ravel(), yy.ravel())
    ra_grid = world.ra.deg.reshape(ny, nx)
    dec_grid = world.dec.deg.reshape(ny, nx)
    customdata = np.dstack([ra_grid, dec_grid])

    if colorbar is not None:
        colorbar = dict(colorbar, title=cbar_label)

    fig.add_trace(
        go.Heatmap(
            z=z, colorscale=colorscale, zmin=zmin, zmax=zmax,
            showscale=colorbar is not None, colorbar=colorbar,
            customdata=customdata,
            hovertemplate="x=%{x}  y=%{y}<br>RA=%{customdata[0]:.5f}°  "
                           "Dec=%{customdata[1]:.5f}°<br>value=%{z:.4g}<extra></extra>",
        ),
        row=row, col=col,
    )
    # scaleanchor must reference THIS panel's own x-axis id (e.g. "x2" for
    # the 2nd subplot), not the literal string "x" -- "x" is only correct
    # by coincidence for the very first subplot (row=1, col=1). Plotly does
    # NOT remap "x" to the target subplot's own axis based on row/col; a
    # hardcoded "x" silently scale-locks every OTHER panel's y-axis against
    # the FIRST panel's x-axis instead of its own, which only looks right
    # when every panel happens to share an identical x pixel range/domain
    # width -- otherwise it stretches that panel's image (e.g. the "3D
    # matched-filter candidates" panel visibly stretched along x relative
    # to "2D white-light + SExtractor" in the seg3d comparison figure
    # before this fix). Read back the axis id Plotly actually assigned
    # this trace instead of assuming it.
    xaxis_id = fig.data[-1].xaxis
    fig.update_xaxes(tickmode="array", tickvals=xt, ticktext=xt_labels,
                      title_text="RA (deg)", row=row, col=col)
    fig.update_yaxes(tickmode="array", tickvals=yt, ticktext=yt_labels,
                      title_text="Dec (deg)", scaleanchor=xaxis_id, row=row, col=col)
    if title:
        fig.add_annotation(text=title, xref="x domain", yref="y domain",
                            x=0.5, y=1.08, showarrow=False,
                            font=dict(size=11), row=row, col=col)


def add_ellipse_outline(fig, row, col, cx, cy, width, height, angle_deg=0.0, *,
                         color="red", line_width=1.5, n_points=60, name=None):
    """Add a rotated ellipse outline (pixel coordinates) as a closed line
    trace — Plotly's native `add_shape(type="circle")` has no rotation
    parameter, so extracted-source/mask ellipses (which do have a
    position angle) are drawn parametrically instead.
    """
    t = np.linspace(0, 2 * np.pi, n_points)
    a, b = width / 2.0, height / 2.0
    theta = np.deg2rad(angle_deg)
    ex = a * np.cos(t)
    ey = b * np.sin(t)
    x = cx + ex * np.cos(theta) - ey * np.sin(theta)
    y = cy + ex * np.sin(theta) + ey * np.cos(theta)
    fig.add_trace(
        go.Scatter(x=x, y=y, mode="lines", line=dict(color=color, width=line_width),
                    name=name or "", showlegend=bool(name), hoverinfo="skip"),
        row=row, col=col,
    )


def add_circle_outline(fig, row, col, cx, cy, radius, **kwargs):
    """Axis-aligned circle — thin wrapper around add_ellipse_outline."""
    add_ellipse_outline(fig, row, col, cx, cy, 2 * radius, 2 * radius, 0.0, **kwargs)
