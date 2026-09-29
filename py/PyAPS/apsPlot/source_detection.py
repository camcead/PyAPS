"""
Source-detection diagnostic figure (6-panel: Background / Background RMS
/ GAIA sources / Mask / Sources / Segmentation Map), built on the shared
viz.wcs_image primitives — port of the plotting block inside
aps_ifu_prepare.ifu_seg2d.object_extract.
"""

from __future__ import annotations

import numpy as np
from plotly.subplots import make_subplots

from .wcs_image import add_wcs_image, add_ellipse_outline
from . import style

_WIDTH_PER_COL = 320
_MARGIN_PER_PANEL = 80


def _colorbar_for(n_panels, index, ncols):
    """Same fraction-based colorbar placement as viz.exgal_prepare
    (see that module for why the naive version overlaps the last panel):
    paper x=[0,1] spans the plot area regardless of pixel width, so the
    paper-x needed to reach the margin's far edge is margin_px /
    plot_area_px, not margin_px / total_px."""
    plot_px = _WIDTH_PER_COL * ncols
    margin_px = _MARGIN_PER_PANEL * n_panels
    step = (margin_px / plot_px) / (n_panels + 1)
    return dict(thickness=8, len=0.38, x=1.0 + step * (index + 1))


def build_figure(wcs, bkg_image, bkg_rms, data_sub, mask, objects, segmap, *,
                  gaia_pix=None, mask_dim_pix=None, radii_factor=7.0, headname=None):
    """Build the 6-panel source-detection figure for one white-light image.

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        WCS of the white-light image (shared by every panel — all arrays
        below are on the same pixel grid).
    bkg_image, bkg_rms, data_sub, mask, segmap : 2D array
        SEP background model, background RMS, background-subtracted
        data, bad-pixel mask, and segmentation map (`sep.Background`/
        `sep.extract` outputs).
    objects : structured array
        `sep.extract`'s detected-object table (needs 'x', 'y', 'a', 'b',
        'theta' fields, all in pixel units/radians as SEP returns them).
    gaia_pix : (x_pix, y_pix), optional
        GAIA star pixel positions (already WCS-transformed). When
        omitted, panel 3 still shows the underlying image (legacy skips
        creating that subplot entirely in this case, leaving a gap —
        this shows the data without star overlays instead, since an
        empty Plotly subplot doesn't get properly allocated and throws
        off every annotation positioned after it).
    mask_dim_pix : float, optional
        GAIA mask circle diameter in pixels (required if `gaia_pix` given).
    radii_factor : float
        Multiplier on each detected object's (a, b) to draw its ellipse —
        matches legacy's `self.radii_factor` (roughly twice the isophotal
        footprint).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    has_gaia = gaia_pix is not None and len(gaia_pix[0]) > 0
    ncols = 3
    n_panels = 6  # colorbar count stays fixed at 6 even if panel 3 is empty,
                  # so panel spacing doesn't shift depending on GAIA availability

    fig = make_subplots(rows=2, cols=3, horizontal_spacing=0.05, vertical_spacing=0.12)

    add_wcs_image(fig, 1, 1, bkg_image, wcs, title="Background",
                  colorbar=_colorbar_for(n_panels, 0, ncols), cbar_label="")
    add_wcs_image(fig, 1, 2, bkg_rms, wcs, title="Background RMS",
                  colorbar=_colorbar_for(n_panels, 1, ncols), cbar_label="")

    finite = data_sub[np.isfinite(data_sub)]
    m, s = (float(np.nanmean(finite)), float(np.nanstd(finite))) if finite.size else (0.0, 1.0)

    # Always draw the underlying image here (even with no GAIA cross-
    # match, just without star overlays) rather than leaving the panel
    # trace-less — an empty subplot doesn't get properly allocated by
    # Plotly, which throws off every annotation position after it.
    add_wcs_image(fig, 1, 3, data_sub, wcs, zmin=m - s, zmax=m + s,
                  title="GAIA sources  [to be masked!]" if has_gaia else "GAIA sources  (none found)",
                  colorbar=_colorbar_for(n_panels, 2, ncols), cbar_label="")
    if has_gaia:
        gx, gy = gaia_pix
        for i in range(len(gx)):
            add_ellipse_outline(fig, 1, 3, gx[i], gy[i], mask_dim_pix, mask_dim_pix,
                                 0.0, color="yellow", line_width=1.2)

    add_wcs_image(fig, 2, 1, mask, wcs, title="Mask",
                  colorbar=_colorbar_for(n_panels, 3, ncols), cbar_label="")

    add_wcs_image(fig, 2, 2, data_sub, wcs, zmin=m - s, zmax=m + s, title="Sources",
                  colorbar=_colorbar_for(n_panels, 4, ncols), cbar_label="")
    for i in range(len(objects)):
        add_ellipse_outline(
            fig, 2, 2, float(objects["x"][i]), float(objects["y"][i]),
            radii_factor * float(objects["a"][i]), radii_factor * float(objects["b"][i]),
            float(objects["theta"][i]) * 180.0 / np.pi, color="red", line_width=1.2,
        )

    add_wcs_image(fig, 2, 3, segmap, wcs, title="Segmentation Map",
                  colorbar=_colorbar_for(n_panels, 5, ncols), cbar_label="")

    base_layout = {k: v for k, v in style.BASE_LAYOUT.items() if k != "margin"}
    margin_px = _MARGIN_PER_PANEL * n_panels
    fig.update_layout(
        **base_layout,
        title=dict(text=headname, x=0.5) if headname else None,
        width=_WIDTH_PER_COL * ncols + margin_px,
        height=760,
        margin=dict(l=60, r=40 + margin_px, t=90, b=60),
    )
    return fig
