"""
Single-panel WCS sky-image figure with target/mask region overlays,
built on the shared viz.wcs_image primitives — port of
aps_ifu_prepare.plot_region.
"""

from __future__ import annotations

import os

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from .wcs_image import add_wcs_image, add_circle_outline
from plotly.subplots import make_subplots
from . import style


def _sky_circle_to_pixel(wcs, ra_deg, dec_deg, radius_arcsec):
    """Convert an ICRS (ra, dec, radius_arcsec) region into pixel-space
    (cx, cy, radius_px), matching what legacy's
    `transform=ax.get_transform('icrs')` did implicitly for the Circle
    patch."""
    cx, cy = wcs.world_to_pixel(SkyCoord(ra=ra_deg, dec=dec_deg, unit="deg"))
    pixel_scale_deg = wcs.proj_plane_pixel_scales()[0].value
    radius_px = (radius_arcsec / 3600.0) / pixel_scale_deg
    return float(cx), float(cy), float(radius_px)


def build_figure(infile, headname, ext=1, area=None, mask_areas=None):
    """Build the region-overlay figure for one input FITS image.

    Parameters
    ----------
    infile : str
        Path to the FITS file to display.
    ext : int
        HDU extension index containing the 2D image + WCS.
    area : (ra_deg, dec_deg, radius_arcsec), optional
        Target region, drawn in yellow (legacy convention).
    mask_areas : sequence[(ra_deg, dec_deg, radius_arcsec)], optional
        Masked regions, drawn in red (legacy convention).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    hdu = fits.open(infile)[ext]
    wcs = WCS(hdu.header)
    data = np.asarray(hdu.data, dtype=np.float64)

    fig = make_subplots(rows=1, cols=1)
    add_wcs_image(fig, 1, 1, data, wcs, log=True, title=headname)

    if area is not None:
        cx, cy, r = _sky_circle_to_pixel(wcs, area[0], area[1], area[2])
        add_circle_outline(fig, 1, 1, cx, cy, r, color="yellow", name="target area")

    if mask_areas is not None:
        for i, each_mask in enumerate(mask_areas):
            cx, cy, r = _sky_circle_to_pixel(wcs, each_mask[0], each_mask[1], each_mask[2])
            add_circle_outline(fig, 1, 1, cx, cy, r, color="red",
                                name="mask area" if i == 0 else None)

    base_layout = {k: v for k, v in style.BASE_LAYOUT.items() if k != "margin"}
    fig.update_layout(**base_layout, width=650, height=620,
                       margin=dict(l=70, r=40, t=70, b=60))
    return fig


def plot_region(infile, headname, figdir, ext=1, area=None, mask_areas=None):
    """Drop-in replacement for the legacy matplotlib `plot_region`.

    Same `<figdir>/<headname>_reg.png` output convention, now rendered
    through `build_figure()` + `write_image()`.
    """
    if not os.path.isdir(figdir):
        os.makedirs(figdir, exist_ok=True)
        print("FIGDIR: %s Created!" % figdir)

    figfname = os.path.join(figdir, f"{headname}_reg.png")
    fig = build_figure(infile, headname, ext=ext, area=area, mask_areas=mask_areas)
    fig.write_image(figfname, scale=2)
