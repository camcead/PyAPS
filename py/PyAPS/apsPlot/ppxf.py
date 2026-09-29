"""
PPXF stellar-kinematics fit figures, built on the shared viz.spectra
overlay builder. Unlike Redrock/FERRE/RVS, PPXF fits a single continuum
model to one already arm-combined (log-rebinned, rest-frame) spectrum per
target — no per-arm split, no halves.
"""

from __future__ import annotations

import numpy as np

from .spectra import spectrum_overlay_figure


def _masked_regions(goodpixels_row, n_wave):
    """Reproduce the legacy masked-region boundary detection from
    PyAPS.MOSExGalPPXF.make_ppxf_plot exactly (same array ops, same
    edge-pixel override), returning a per-pixel boolean mask (True =
    shaded) instead of matplotlib axvspans.
    """
    gpp = np.array(goodpixels_row, dtype=float)
    gpp[np.isnan(gpp)] = -1
    gpp[gpp < 0] = -1
    gpp[0] = 0
    gpp[-1] = 0
    gpp = gpp.astype(int)

    masked = np.flatnonzero(np.abs(np.diff(gpp)) > 1)
    vlines = []
    for mi in masked:
        vlines.append(gpp[mi] + 1)
        vlines.append(gpp[mi + 1] - 1)
    vlines = np.array(vlines)

    bad = np.zeros(n_wave, dtype=bool)
    for mi in range(len(np.where(vlines != 0)[0])):
        if mi % 2 == 0:
            try:
                s, e = vlines[mi], vlines[mi + 1]
                bad[s:e + 1] = True
            except Exception:
                pass
    return bad


def build_figure(ppxf_result, ppxf_logLam, ppxf_spec, ppxf_bestfit,
                  ppxf_goodpixels, i, flux_unit=None):
    """Build the PPXF fit-overlay figure for one target row.

    Parameters mirror PyAPS.MOSExGalPPXF.make_ppxf_plot's per-target 2D
    arrays (all indexed by row `i`): `ppxf_logLam` is the rest-frame,
    log-rebinned wavelength grid, `ppxf_spec` the input spectrum,
    `ppxf_bestfit` the PPXF model, `ppxf_goodpixels` the fitted
    good-pixel index array used to derive masked-region shading.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    wave_lin = np.exp(ppxf_logLam[i, :])
    n = len(wave_lin)

    bad = _masked_regions(ppxf_goodpixels[i, :], n)
    good_only = np.copy(ppxf_spec[i, :])
    good_only[bad] = np.nan

    arm = "combined"
    title = (
        f"V={ppxf_result[i][0]:.3f}  SIGMA={ppxf_result[i][1]:.3f}  "
        f"H3={ppxf_result[i][2]:.3f}  H4={ppxf_result[i][3]:.3f}"
    )

    return spectrum_overlay_figure(
        [arm],
        {arm: wave_lin},
        {arm: ppxf_spec[i, :]},
        {arm: ppxf_bestfit[i, :]},
        mask={arm: bad},
        extra_traces={arm: [{
            "name": "Only good pixels", "y": good_only,
            "color": "#1f77b4", "width": 0.5,
        }]},
        rank_labels=["Best fitted PPXF Model"],
        panel_titles={arm: title},
        percentile_clip=None,
        flux_unit=flux_unit,
        wave_label="λ (AIR) [Å] [REST-FRAME(corrected for input Z)]",
    )


def make_ppxf_plot(figdir, configs, ppxf_metalist, ppxf_result, ppxf_logLam,
                    ppxf_spec, ppxf_error, ppxf_bestfit, ppxf_goodpixels):
    """Drop-in replacement for the legacy matplotlib `make_ppxf_plot`.

    Same signature and `PPXF_<APS_ID>_<TARGID>_<CNAME>.png` output
    convention, now rendered through `build_figure()` + `write_image()`
    (requires the `kaleido` package).
    """
    if configs['sens_corr']:
        flux_unit_str = '%2e erg/(s cm**2 Angstrom)' % (configs['funits'])
    else:
        flux_unit_str = 'count'

    from astropy import units
    flux_unit = str(units.Unit(flux_unit_str))

    for i in range(len(ppxf_metalist)):
        fig_fname = figdir + 'PPXF_%s_%s_%s.png' % (
            ppxf_metalist[i]['APS_ID'], ppxf_metalist[i]['TARGID'],
            ppxf_metalist[i]['CNAME'])
        fig = build_figure(ppxf_result, ppxf_logLam, ppxf_spec,
                            ppxf_bestfit, ppxf_goodpixels, i,
                            flux_unit=flux_unit)
        fig.write_image(fig_fname, scale=2)
    return 1
