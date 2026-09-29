"""
Lambda_R (specific stellar angular momentum proxy) spatial map figure,
built on the shared viz.spatial parameter-map-grid builder.
"""

from __future__ import annotations

import os

import numpy as np
from astropy.io import fits

from .spatial import parameter_map_grid_figure


def build_figure(outpath, headname, vminmax=(0.0, 1.0)):
    """Build the single-panel Lambda_R spatial-map figure for one target.

    Parameters
    ----------
    outpath : str
        Directory containing `<headname>_table.fits` and
        `<headname>_ppxf.fits` (LAMBDA_R is a column on the same PPXF
        result file the stellar-kinematics maps read).
    headname : str
        Target/pointing root name.
    vminmax : (float, float), optional
        Colour-scale range. Legacy hardcodes (0.0, 1.0) in pipeline mode
        (LAMBDA_R is bounded in that range by construction) rather than
        auto-scaling from the data like kinematics/LS do — kept as the
        default here too.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    table_hdu = fits.open(outpath + headname + '_table.fits')
    idx_inside = np.where(table_hdu[1].data.BIN_ID >= 0)[0]
    x = np.array(table_hdu[1].data.X[idx_inside]) * -1
    y = np.array(table_hdu[1].data.Y[idx_inside])
    bin_num_long = np.array(table_hdu[1].data.BIN_ID[idx_inside])
    ubins = np.unique(np.abs(np.array(table_hdu[1].data.BIN_ID)))
    pixelsize = table_hdu[0].header['PIXSIZE']

    result = np.array(fits.open(outpath + headname + '_ppxf.fits')[1].data.LAMBDA_R)
    result_long = np.full(len(bin_num_long), np.nan)
    for i in range(len(ubins)):
        idx = np.where(ubins[i] == np.abs(bin_num_long))[0]
        result_long[idx] = result[i]

    return parameter_map_grid_figure(
        ["λ"],
        [result_long],
        x, y, pixelsize,
        nrows=1, ncols=1,
        value_format=".2f",
        vminmax=[vminmax],
        pad=6,
        figure_title=headname,
        # X negated above, same as kinematics/LS. This grid has a single
        # panel -> a single invert_xaxis() call in legacy -> odd count ->
        # net inverted, same parity as LS's 1x3 grid (3 calls). See
        # viz.kinematics_map.build_figure for the full explanation of why
        # this isn't the same for every module. Verified against the
        # existing legacy PNG for LWVE_12141807+5936554_01_BR_L1_P0001
        # before touching IFUExGalPlot_lambdar.py, not just by parity
        # theory.
        invert_x=True,
    )


def plot_maps(outpath, headname, INTERACTIVE=False, vminmax=np.array([0.0, 1.0]),
              contour_offset_saved=0.20, SAVE_AFTER_INTERACTIVE=False):
    """Drop-in replacement for the legacy matplotlib `plot_maps`.

    Same `<outpath>/maps/<headname>_lambda.png` output convention, now
    rendered through `build_figure()` + `write_image()`.

    The INTERACTIVE terminal-prompt workflow is not ported — see
    `PyAPS.apsPlot.kinematics_map.plot_maps` for the rationale, which
    applies identically here.
    """
    if INTERACTIVE:
        print(
            "NOTE: IFUExGalPlot_lambdar.plot_maps()'s INTERACTIVE stdin "
            "vmin/vmax-tuning workflow is not ported to PyAPS.apsPlot — "
            "rendering with the default (0.0, 1.0) scale instead (same "
            "as pipeline mode). The Plotly output is natively zoomable/"
            "hoverable, so tune by inspecting the saved figure rather "
            "than via terminal prompts. For scripted control, call "
            "build_figure() directly with a different vminmax."
        )

    fig = build_figure(outpath, headname, vminmax=tuple(vminmax))

    figdir = outpath + 'maps/'
    if not os.path.isdir(figdir):
        os.mkdir(figdir)

    fig.write_image(figdir + headname + '_lambda.png', scale=2)
