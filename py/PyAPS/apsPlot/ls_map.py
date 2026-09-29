"""
Line-strength (absorption index) spatial map figures, built on the
shared viz.spatial parameter-map-grid builder.
"""

from __future__ import annotations

import os

import numpy as np
from astropy.io import fits

from .spatial import parameter_map_grid_figure

LABELS = ["Hbeta_o", "Fe5015", "Mgb"]


def build_figure(outpath, headname, resolution='ADAPTED'):
    """Build the line-strength spatial-map figure for one target.

    Parameters
    ----------
    outpath : str
        Directory containing `<headname>_table.fits` and
        `<headname>_ls_ORIGINAL.fits` / `<headname>_ls_ADAPTED.fits`.
    headname : str
        Target/pointing root name.
    resolution : {"ORIGINAL", "ADAPTED"}
        Which line-strength result file to read.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if resolution == 'ORIGINAL':
        ls_hdu = fits.open(outpath + headname + '_ls_ORIGINAL.fits')
    elif resolution == 'ADAPTED':
        ls_hdu = fits.open(outpath + headname + '_ls_ADAPTED.fits')
    else:
        raise ValueError(f"resolution must be 'ORIGINAL' or 'ADAPTED', got {resolution!r}")

    ubins = np.arange(0, len(ls_hdu[1].data.Hbeta_o))
    result = np.empty((len(ubins), 3))
    result[:, 0] = np.array(ls_hdu[1].data.Hbeta_o)
    result[:, 1] = np.array(ls_hdu[1].data.Fe5015)
    result[:, 2] = np.array(ls_hdu[1].data.Mgb)

    table_hdu = fits.open(outpath + headname + '_table.fits')
    idx_inside = np.where(table_hdu[1].data.BIN_ID >= 0)[0]
    x = np.array(table_hdu[1].data.X[idx_inside]) * -1
    y = np.array(table_hdu[1].data.Y[idx_inside])
    bin_num_long = np.array(table_hdu[1].data.BIN_ID[idx_inside])
    pixelsize = table_hdu[0].header['PIXSIZE']

    result_long = np.full((len(bin_num_long), result.shape[1]), np.nan)
    for i in range(len(ubins)):
        idx = np.where(ubins[i] == np.abs(bin_num_long))[0]
        result_long[idx, :] = result[i, :]

    return parameter_map_grid_figure(
        ["Hβ₀", "Fe5015", "Mg b"],
        [result_long[:, k] for k in range(3)],
        x, y, pixelsize,
        nrows=1, ncols=3,
        value_format=".2f",
        pad=5,
        figure_title=headname,
        # X is already negated above, same as kinematics — but unlike
        # kinematics this DOES also need invert_x=True (the builder's
        # default) to match legacy. Verified by regenerating the actual
        # legacy matplotlib output from git history and comparing feature
        # positions directly, not just tick labels. Root cause: legacy's
        # AxesGrid(..., share_all=True) shares x-axis state across
        # panels, and matplotlib's invert_xaxis() *toggles* that shared
        # state each time it's called — kinematics' 2x2 grid calls it 4x
        # (even → cancels out, net non-inverted), LS's 1x3 grid calls it
        # 3x (odd → net inverted). Same source code pattern, opposite
        # final result, purely from grid shape. Not a design choice worth
        # replicating faithfully — noted here so the next port (lambdar)
        # doesn't assume either direction and instead checks its own
        # grid's call count.
        invert_x=True,
    )


def plot_maps(outpath, headname, RESOLUTION, INTERACTIVE=False, vminmax=np.zeros((4, 2)),
              contour_offset_saved=0.20, SAVE_AFTER_INTERACTIVE=False):
    """Drop-in replacement for the legacy matplotlib `plot_maps`.

    Same `<outpath>/maps/<headname>_ls_ORIGINAL.png` / `_ls_ADAPTED.png`
    output convention, now rendered through `build_figure()` +
    `write_image()`.

    The INTERACTIVE terminal-prompt workflow is not ported — see
    `PyAPS.apsPlot.kinematics_map.plot_maps` for the rationale, which
    applies identically here.
    """
    if INTERACTIVE:
        print(
            "NOTE: IFUExGalPlot_ls.plot_maps()'s INTERACTIVE stdin "
            "vmin/vmax-tuning workflow is not ported to PyAPS.apsPlot — "
            "rendering with auto-scaled vmin/vmax instead (same as "
            "pipeline mode). The Plotly output is natively zoomable/"
            "hoverable, so tune by inspecting the saved figure rather "
            "than via terminal prompts. For scripted control, call "
            "build_figure() directly and pass vminmax to "
            "PyAPS.apsPlot.spatial.parameter_map_grid_figure()."
        )

    fig = build_figure(outpath, headname, resolution=RESOLUTION)

    figdir = outpath + 'maps/'
    if not os.path.isdir(figdir):
        os.mkdir(figdir)

    if RESOLUTION == 'ORIGINAL':
        fig_fname = figdir + headname + '_ls_ORIGINAL.png'
    elif RESOLUTION == 'ADAPTED':
        fig_fname = figdir + headname + '_ls_ADAPTED.png'
    else:
        raise ValueError(f"RESOLUTION must be 'ORIGINAL' or 'ADAPTED', got {RESOLUTION!r}")

    fig.write_image(fig_fname, scale=2)
