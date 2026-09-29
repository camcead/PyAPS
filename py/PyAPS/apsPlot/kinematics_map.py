"""
Stellar/gas kinematics (V/Sigma/h3/h4) spatial map figures, built on the
shared viz.spatial parameter-map-grid builder.
"""

from __future__ import annotations

import os

import numpy as np
from astropy.io import fits

from .spatial import parameter_map_grid_figure

LABELS = ["V", "SIGMA", "H3", "H4"]
# Legacy: V/Sigma corner labels use "{:.0f}", h3/h4 use "{:.2f}".
VALUE_FORMATS = [".0f", ".0f", ".2f", ".2f"]


def _read_table(outpath, headname):
    """Read the bin table and return the per-spaxel (long) coordinate
    arrays plus the bin-id lookup needed to expand per-bin results.
    Mirrors IFUExGalPlot_kinematics.plot_maps exactly.
    """
    table_hdu = fits.open(outpath + headname + '_table.fits')
    idx_inside = np.where(table_hdu[1].data.BIN_ID >= 0)[0]
    x = np.array(table_hdu[1].data.X[idx_inside]) * -1
    y = np.array(table_hdu[1].data.Y[idx_inside])
    flux = np.array(table_hdu[1].data.FLUX[idx_inside])
    bin_num_long = np.array(table_hdu[1].data.BIN_ID[idx_inside])
    ubins = np.unique(np.abs(np.array(table_hdu[1].data.BIN_ID)))
    pixelsize = table_hdu[0].header['PIXSIZE']
    return x, y, flux, bin_num_long, ubins, pixelsize


def _expand_to_long(result, ubins, bin_num_long):
    """Map per-bin result rows back onto the per-spaxel (long) array via
    BIN_ID, exactly as the legacy plotter does."""
    result_long = np.full((len(bin_num_long), result.shape[1]), np.nan)
    for i in range(len(ubins)):
        idx = np.where(ubins[i] == np.abs(bin_num_long))[0]
        result_long[idx, :] = result[i, :]
    return result_long


def build_figure(outpath, headname, flag='PPXF'):
    """Build the stellar-kinematics spatial-map figure for one target.

    Parameters
    ----------
    outpath : str
        Directory containing `<headname>_table.fits` and
        `<headname>_ppxf.fits` / `<headname>_sfh.fits`.
    headname : str
        Target/pointing root name.
    flag : {"PPXF", "SFH"}
        Which kinematics result file to read (unregularized PPXF run vs
        regularized star-formation-history run).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    x, y, flux, bin_num_long, ubins, pixelsize = _read_table(outpath, headname)

    if flag == 'PPXF':
        hdu = fits.open(outpath + headname + '_ppxf.fits')
    elif flag == 'SFH':
        hdu = fits.open(outpath + headname + '_sfh.fits')
    else:
        raise ValueError(f"flag must be 'PPXF' or 'SFH', got {flag!r}")

    result = np.empty((len(ubins), 4))
    result[:, 0] = np.array(hdu[1].data.V)
    result[:, 1] = np.array(hdu[1].data.SIGMA)
    result[:, 2] = np.array(hdu[1].data.H3)
    result[:, 3] = np.array(hdu[1].data.H4)

    result_long = _expand_to_long(result, ubins, bin_num_long)
    result_long[:, 0] = result_long[:, 0] - np.nanmedian(result_long[:, 0])

    return parameter_map_grid_figure(
        ["V (stellar) [km/s]", "σ (stellar) [km/s]", "h3", "h4"],
        [result_long[:, k] for k in range(4)],
        x, y, pixelsize,
        nrows=2, ncols=2,
        value_format=VALUE_FORMATS,
        figure_title=headname,
        # X is already negated above (matching legacy's `X = ... * -1`).
        # Legacy also calls matplotlib's invert_xaxis() on each of the 4
        # panels — but reproducing its exact on-disk output (verified
        # against LWVE_12141807+5936554_01_BR_L1_P0001, regenerated from
        # git history for a rigorous side-by-side) requires *not*
        # reversing the Plotly axis on top of the negation. Root cause:
        # legacy's AxesGrid(..., share_all=True) shares x-axis state
        # across panels, and invert_xaxis() *toggles* that shared state
        # each call — 4 panels here means 4 toggles, an even number, so
        # it cancels back to non-inverted. viz.ls_map's 1x3 grid calls it
        # 3x (odd → net inverted) and needs invert_x=True instead. Same
        # source pattern, opposite result, purely from grid shape — check
        # this per-module rather than assuming either direction.
        invert_x=False,
    )


def plot_maps(flag, outpath, headname, INTERACTIVE=False, vminmax=np.zeros((4, 2)),
              contour_offset_saved=0.20, SAVE_AFTER_INTERACTIVE=False):
    """Drop-in replacement for the legacy matplotlib `plot_maps`.

    Same `<outpath>/maps/<headname>_ppxf.png` (or `_sfh_kin.png`) output
    convention, now rendered through `build_figure()` + `write_image()`.

    The INTERACTIVE terminal-prompt workflow (manually tuning vmin/vmax
    via stdin before saving) is not ported — the Plotly output is
    natively zoomable/hoverable, which covers the same need for
    inspecting the data without a save/reload loop. If you still need
    scripted vmin/vmax control, call `build_figure()` directly with
    `PyAPS.apsPlot.spatial.parameter_map_grid_figure`'s `vminmax` parameter.
    """
    if INTERACTIVE:
        print(
            "NOTE: IFUExGalPlot_kinematics.plot_maps()'s INTERACTIVE stdin "
            "vmin/vmax-tuning workflow is not ported to PyAPS.apsPlot — "
            "rendering with auto-scaled vmin/vmax instead (same as "
            "pipeline mode). The Plotly output is natively zoomable/"
            "hoverable, so tune by inspecting the saved figure rather "
            "than via terminal prompts. For scripted control, call "
            "build_figure() directly and pass vminmax to "
            "PyAPS.apsPlot.spatial.parameter_map_grid_figure()."
        )

    fig = build_figure(outpath, headname, flag=flag)

    figdir = outpath + 'maps/'
    if not os.path.isdir(figdir):
        os.mkdir(figdir)

    if flag == 'PPXF':
        fig_fname = figdir + headname + '_ppxf.png'
    elif flag == 'SFH':
        fig_fname = figdir + headname + '_sfh_kin.png'
    else:
        raise ValueError(f"flag must be 'PPXF' or 'SFH', got {flag!r}")

    fig.write_image(fig_fname, scale=2)
