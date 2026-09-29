"""
RVS (radial velocity, via rvspecfit) fit figures, built on the shared
viz.spectra overlay builder. Like FERRE, RVS fits a single best-fit
template per arm (no ranked alternatives like Redrock) — each arm is
split into two halves for display, matching the legacy layout. Bad
pixels are expected to already be NaN-masked in `specdata`/`yfit` by the
caller (aps_rvs does this before calling make_rvs_plot).
"""

from __future__ import annotations

import numpy as np

from .spectra import spectrum_overlay_figure


def build_figure(specdata, yfit, title, flux_unit=None):
    """Build the RVS fit-overlay figure for one target.

    Parameters
    ----------
    specdata : sequence[rvspecfit.spec_fit.SpecData]
        Per-arm observed spectrum objects (`.lam`, `.spec`), in the same
        order as `yfit`.
    yfit : sequence[array]
        Per-arm best-fit model array, aligned with `specdata` (i.e.
        `res_dict['yfit']` from rvspecfit's `vel_fit.process`).
    title : str
        Precomputed title string (FEH/TEFF/LOGG/ALPHA/Vrad) — RVS has no
        per-arm setup labels available at this point, only arm position.
    flux_unit : str, optional
        Y-axis label for flux panels.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    arms, wave, flux, models = [], {}, {}, {}

    for n, sd in enumerate(specdata):
        lam = np.asarray(sd.lam)
        fl = np.asarray(sd.spec)
        mod = np.asarray(yfit[n])
        half = len(lam) // 2

        for part, sl in ((1, slice(0, half)), (2, slice(half, None))):
            arm = f"Arm {n} (part {part})"
            arms.append(arm)
            wave[arm] = lam[sl]
            flux[arm] = fl[sl]
            models[arm] = mod[sl]

    return spectrum_overlay_figure(
        arms, wave, flux, models,
        rank_labels=["Model"],
        panel_titles={arms[0]: title},
        percentile_clip=None,
        flux_unit=flux_unit,
        # Matches the legacy label exactly — like the original, this
        # function has no vacuum/air flag available, only whatever
        # convention the caller already applied to specdata.
        wave_label="λ (AIR) [Å]",
    )


def make_rvs_plot(specdata, res_dict, title, fig_fname, units_str=None):
    """Drop-in replacement for the legacy matplotlib `make_rvs_plot`.

    Same signature and output file convention, now rendered through
    `build_figure()` + `write_image()` (requires `kaleido`).
    """
    flux_unit = None
    if units_str is not None:
        from astropy import units
        flux_unit = str(units.Unit(units_str['flux']))

    fig = build_figure(specdata, res_dict['yfit'], title, flux_unit=flux_unit)
    fig.write_image(str(fig_fname), scale=2)
