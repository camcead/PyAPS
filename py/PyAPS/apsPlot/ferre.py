"""
FERRE stellar-parameter fit figures, built on the shared viz.spectra
overlay builder. Like RVS, FERRE fits a single best-fit synthetic
spectrum per arm (no ranked alternatives like Redrock) — each arm is
split into two halves for display, matching the legacy layout.
"""

from __future__ import annotations

import numpy as np

from .spectra import spectrum_overlay_figure


def build_figure(outdict, i, setups, flux_unit=None):
    """Build the FERRE fit-overlay figure for one target row.

    Parameters
    ----------
    outdict : dict
        FERRE result dict, as produced by aps_ferre (plain dict of lists/
        arrays keyed by e.g. "LAMBDA_FR_<arm>", "FLUX_FR_<arm>",
        "MODEL_FR_<arm>", plus scalar per-target params FEH/TEFF/LOGG/
        ALPHA/MICRO).
    i : int
        Target row index within outdict.
    setups : sequence[str]
        Arm/setup codes, e.g. ["BLUE", "RED"] — only the first character
        of each is used to key into outdict's columns.
    flux_unit : str, optional
        Y-axis label for flux panels.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    arms, wave, flux, models = [], {}, {}, {}

    for s in setups:
        sk = s[0]
        lam = np.asarray(outdict[f"LAMBDA_FR_{sk}"][i])
        fl = np.asarray(outdict[f"FLUX_FR_{sk}"][i])
        mod = np.asarray(outdict[f"MODEL_FR_{sk}"][i])
        half = len(lam) // 2

        for part, sl in ((1, slice(0, half)), (2, slice(half, None))):
            arm = f"{s} (part {part})"
            arms.append(arm)
            wave[arm] = lam[sl]
            flux[arm] = fl[sl]
            models[arm] = mod[sl]

    title = (
        f"FEH={outdict['FEH'][i]:.3f}  TEFF={outdict['TEFF'][i]:.1f}  "
        f"LOGG={outdict['LOGG'][i]:.2f}  ALPHA={outdict['ALPHA'][i]:.2f}  "
        f"MICRO={outdict['MICRO'][i]:.2f}"
    )

    return spectrum_overlay_figure(
        arms, wave, flux, models,
        rank_labels=["Model"],
        panel_titles={arms[0]: title},
        percentile_clip=None,
        flux_unit=flux_unit,
        # Matches the legacy label exactly — like aps_rvs, this function
        # has no vacuum/air flag available, only whatever convention the
        # caller already applied to LAMBDA_FR.
        wave_label="λ (AIR) [Å]",
    )


def make_fr_plot(outdict, figdir, setups, units_str=None):
    """Drop-in replacement for the legacy matplotlib `make_fr_plot`.

    Same signature and `FR_<TARGID>_<CNAME>_<APS_ID>.png` output
    convention, now rendered through `build_figure()` + `write_image()`
    (requires the `kaleido` package).
    """
    flux_unit = None
    if units_str is not None:
        from astropy import units
        flux_unit = str(units.Unit(units_str['flux']))

    for i in range(len(outdict['APS_ID'])):
        fig_fname = figdir + 'FR_%s_%s_%s.png' % (
            outdict['TARGID'][i], outdict['CNAME'][i], outdict['APS_ID'][i])
        fig = build_figure(outdict, i, setups, flux_unit=flux_unit)
        fig.write_image(fig_fname, scale=2)
    return 1
