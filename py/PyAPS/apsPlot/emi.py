"""
EMIPPXF (PPXF emission-line / "gas") kinematics figures, built on the
shared viz.spectra overlay builder.

Unlike the other PyAPS.apsPlot modules, this one isn't a single flux panel:
the legacy plot stacks up to three panels (raw spectrum + stellar +
combined model, stellar-subtracted spectrum + emission model, residuals)
plus several floating text boxes (per-line kinematics, emission-line
fluxes, stellar kinematics, redshift, component-tying info). Each of the
first two panels is modelled as its own virtual "arm" through the shared
builder; the residual panel is attached to the second arm so it renders
directly below it, matching the legacy row order.
"""

from __future__ import annotations

import numpy as np

from .spectra import spectrum_overlay_figure
from PyAPS import ExGalPrepare

# Overall canvas width and how much of it (in pixels) is reserved as blank
# margin on each side for the floating text boxes — mirrors the legacy
# matplotlib layout's subplots_adjust(left=0.15, right=0.70), which
# shrinks the plot area to leave room for text outside it. Kept as
# constants (not magic fractions) so the margins and annotation x-positions
# below stay geometrically consistent with each other.
_FIG_WIDTH = 1600
_LEFT_MARGIN = 200
_RIGHT_MARGIN = 340

# Approximate rendered pixel height of the first ("flux") row, used only
# to convert a desired top-margin pixel amount into an equivalent "y
# domain" offset for that row (see build_figure). Derived from
# spectrum_overlay_figure's own defaults: height_per_row(170) * n_rows,
# distributed across rows by their weight (flux=2.2) — this is an
# approximation (varies slightly with 2 vs 3 rows), not an exact figure,
# so it's fine if the text box lands a few pixels off from row 1's edge.
_ROW1_HEIGHT_PX = 220


def _masked_regions(goodpixels_row, n_wave):
    """Reproduce make_emi_kinematics_plots' masked-region derivation:
    goodpixels_row lists good-pixel indices, padded with -1. Returns a
    per-pixel boolean mask (True = masked/bad).
    """
    if goodpixels_row is None:
        return np.zeros(n_wave, dtype=bool)
    valid_raw = np.asarray(goodpixels_row)
    valid_raw = valid_raw[valid_raw >= 0]
    if len(valid_raw) > 0:
        valid = np.asarray(valid_raw, dtype=int)
        valid = valid[valid < n_wave]
    else:
        valid = np.array([], dtype=int)
    bad = np.ones(n_wave, dtype=bool)
    if len(valid) > 0:
        bad[valid] = False
    return bad


def _per_line_kinematics_text(i, emi_result, emission_lines, line_names, mc_results, debug=False,
                               stellar_bestfit_row=None, wave=None, redshift=None):
    """Reproduce the per-line kinematics text-box logic exactly, including
    its fallback to a single global V/Sigma when per-line data isn't
    available. Returns (text, valid_kinematics_count).

    `stellar_bestfit_row`/`wave`/`redshift` (all optional, default None):
    when given, each line's equivalent width is appended to its text line
    -- FLUX / local stellar continuum at that line's observed wavelength,
    via ExGalPrepare.compute_equivalent_width (same helper, same sign
    convention -- positive for emission -- as the EW_<line>/ERR_EW_<line>
    output table columns; see that function's own docstring). The line's
    rest wavelength is parsed from its own name (every line name in this
    codebase's convention is `<label>_<rest_wavelength>`, e.g.
    "Hbeta_4861.32" -- the same convention the output FITS columns
    already rely on), rather than requiring a separate emission_config/
    line_wavelengths parameter to be threaded all the way through here.
    """
    text = ""
    valid_count = 0

    if mc_results is not None and i < len(mc_results) and isinstance(mc_results[i], dict):
        line_velocities = mc_results[i].get('line_velocities')
        line_sigmas = mc_results[i].get('line_sigmas')

        if line_velocities is None or line_sigmas is None:
            if i < len(emi_result) and isinstance(emi_result[i], dict):
                line_velocities = emi_result[i].get('line_velocities', line_velocities)
                line_sigmas = emi_result[i].get('line_sigmas', line_sigmas)

        if (line_velocities is not None and line_sigmas is not None and
                hasattr(line_velocities, '__len__') and hasattr(line_sigmas, '__len__') and
                len(line_velocities) > 0 and len(line_sigmas) > 0):
            text = "Per-Line Kinematics:\n"
            for j, line_name in enumerate(line_names[i]):
                if j < len(line_velocities) and j < len(line_sigmas):
                    v_line = line_velocities[j]
                    sigma_line = line_sigmas[j]
                    flux_line = 0.0
                    if emission_lines[i] is not None:
                        if isinstance(emission_lines[i], np.ndarray) and len(emission_lines[i]) > 0:
                            flux_line = emission_lines[i][j]
                    if np.isfinite(v_line) and np.isfinite(sigma_line) and flux_line > 0.0:
                        ew_str = ""
                        if (stellar_bestfit_row is not None and wave is not None
                                and redshift is not None):
                            try:
                                rest_wave = float(line_name.rsplit('_', 1)[-1])
                            except (ValueError, IndexError):
                                rest_wave = np.nan
                            ew, _cont = ExGalPrepare.compute_equivalent_width(
                                flux_line, rest_wave, redshift, wave, stellar_bestfit_row)
                            if np.isfinite(ew):
                                ew_str = f", EW={ew:.2f}Å"
                        text += f"{line_name}: V={v_line:.1f}, Sigma={sigma_line:.1f} km/s{ew_str}\n"
                        valid_count += 1
                else:
                    text += f"{line_name}: V=N/A, Sigma=N/A km/s\n"

    if not text and isinstance(emi_result[i]["emi"], np.ndarray) and len(emi_result[i]["emi"]) >= 2:
        v_fallback = emi_result[i]["emi"][0]
        sigma_fallback = emi_result[i]["emi"][1]
        text = f"emi Kinematics (Global): V={v_fallback:.1f}, Sigma={sigma_fallback:.1f} km/s\n"
        text += "(Per-line kinematics not available)\n"
        valid_count = 1
    elif not text:
        text = "No valid emi kinematics available\n"
        valid_count = 0

    return text, valid_count


def build_figure(i, emi_metalist, emi_result, emission_lines, line_names,
                  logLam, bin_data, emi_bestfit, stellar_bestfit=None,
                  goodpixels_array=None, mc_results=None, flux_unit=None,
                  debug=False):
    """Build the EMIPPXF fit-overlay figure for one target row.

    Parameters mirror PyAPS.MOSExGalEMIPPXF.make_emi_kinematics_plots'
    per-target 2D arrays (all indexed by row `i`). See that function's
    docstring for the full parameter semantics — this is a faithful port,
    not a redesign.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    wave = np.exp(logLam[i, :])
    n = len(wave)
    bad = _masked_regions(
        goodpixels_array[i, :] if goodpixels_array is not None else None, n)

    kin_text, valid_kin_count = _per_line_kinematics_text(
        i, emi_result, emission_lines, line_names, mc_results, debug=debug,
        stellar_bestfit_row=(stellar_bestfit[i, :] if stellar_bestfit is not None else None),
        wave=wave,
        redshift=(emi_metalist[i]['Z'] if 'Z' in emi_metalist[i] else None))

    if valid_kin_count > 0:
        title = f'APS_ID={emi_metalist[i]["APS_ID"]}: {valid_kin_count} lines with valid kinematics'
    else:
        title = f'APS_ID={emi_metalist[i]["APS_ID"]}: emi fit failed'

    has_stellar = stellar_bestfit is not None
    raw = bin_data[i, :]
    emi_model = emi_bestfit[i, :]

    arms = ["raw"]
    wave_d = {"raw": wave}
    flux_d = {"raw": raw}
    mask_d = {"raw": bad}
    extra_d = {}
    residuals_d = {}
    residual_colors = {}
    residual_labels = {}

    if has_stellar:
        stellar = stellar_bestfit[i, :]
        combined_model = stellar + emi_model
        models_d = {"raw": combined_model}
        extra_d["raw"] = [{"name": "Stellar Continuum", "y": stellar,
                            "color": "#1f77b4", "width": 1.0}]

        arms.append("stellar_subtracted")
        stellar_subtracted = raw - stellar
        wave_d["stellar_subtracted"] = wave
        flux_d["stellar_subtracted"] = stellar_subtracted
        models_d["stellar_subtracted"] = emi_model
        mask_d["stellar_subtracted"] = bad

        total_residuals = raw - stellar - emi_model
        residuals_d["stellar_subtracted"] = total_residuals
        residual_colors["stellar_subtracted"] = "#1a1a1a"
        residual_labels["stellar_subtracted"] = "Final Residuals"
        residual_arm = "stellar_subtracted"
    else:
        models_d = {"raw": emi_model}
        residuals_d["raw"] = raw - emi_model
        residual_colors["raw"] = "#1a1a1a"
        residual_labels["raw"] = "Residuals"
        residual_arm = "raw"

    panel_titles = {arms[0]: title}
    flux_labels = {
        "raw": flux_unit or "",
        "stellar_subtracted": ("Stellar-Subtracted " + flux_unit) if flux_unit else "Stellar-Subtracted",
    }

    last_panel_annotations = []
    valid_goodpix = np.flatnonzero(~bad)

    if has_stellar and len(valid_goodpix) > 0:
        residuals_goodpix = residuals_d[residual_arm][valid_goodpix]
        finite = np.isfinite(residuals_goodpix)
        if finite.any():
            rms = float(np.sqrt(np.mean(residuals_goodpix[finite] ** 2)))
            n_goodpix = int(finite.sum())
            last_panel_annotations.append(dict(
                text=f"RMS Residual: {rms:.2e}<br>({n_goodpix} good pixels)",
                x=0.02, y=0.98, showarrow=False, align="left",
                xanchor="left", yanchor="top", font=dict(size=10),
                bgcolor="lightblue", opacity=0.7,
            ))

    if has_stellar and isinstance(emi_result[i], dict) and "stellar" in emi_result[i]:
        stellar_kin = emi_result[i]["stellar"]
        if isinstance(stellar_kin, np.ndarray) and len(stellar_kin) >= 2:
            v_stellar, sigma_stellar = stellar_kin[0], stellar_kin[1]
            last_panel_annotations.append(dict(
                text=f"Stellar: V={v_stellar:.1f} km/s, Sigma={sigma_stellar:.1f} km/s",
                x=0.02, y=0.02, showarrow=False, align="left",
                xanchor="left", yanchor="bottom", font=dict(size=10),
                bgcolor="lightcyan", opacity=0.7,
            ))

    if 'Z' in emi_metalist[i] and np.isfinite(emi_metalist[i]['Z']):
        z_info = f"z = {emi_metalist[i]['Z']:.4f}"
        if 'ZERR' in emi_metalist[i] and np.isfinite(emi_metalist[i]['ZERR']):
            z_info += f" ± {emi_metalist[i]['ZERR']:.4f}"
        last_panel_annotations.append(dict(
            text=z_info, x=0.98, y=0.02, showarrow=False,
            xanchor="right", yanchor="bottom", font=dict(size=10),
            bgcolor="lightyellow", opacity=0.7,
        ))

    if valid_kin_count > 1 and mc_results is not None and i < len(mc_results) and isinstance(mc_results[i], dict):
        line_velocities = mc_results[i].get('line_velocities', [])
        line_sigmas = mc_results[i].get('line_sigmas', [])
        finite_v = [v for v in line_velocities if np.isfinite(v)]
        finite_s = [s for s in line_sigmas if np.isfinite(s)]
        if len(finite_v) > 1 and len(finite_s) > 1:
            v_range = np.max(finite_v) - np.min(finite_v)
            s_range = np.max(finite_s) - np.min(finite_s)
            kind = "Multi-component" if (v_range > 5.0 or s_range > 5.0) else "Single-component"
            last_panel_annotations.append(dict(
                text=f"{kind}: ΔV={v_range:.1f}, ΔSigma={s_range:.1f} km/s",
                x=0.50, y=0.02, showarrow=False, xanchor="center", yanchor="bottom",
                font=dict(size=10), bgcolor="lightpink", opacity=0.7,
            ))

    # Build the emission-line-fluxes text upfront (needed for the line
    # count below) before laying out either box.
    info_text = ""
    if isinstance(emission_lines[i], np.ndarray) and len(emission_lines[i]) > 0:
        info_text = "Emission Line Fluxes:\n"
        for j, name in enumerate(line_names[i]):
            if j < len(emission_lines[i]):
                flux_val = emission_lines[i][j]
                if np.isfinite(flux_val) and flux_val > 0:
                    info_text += f"{name}: {flux_val:.2e}\n"

    # Scale the top margin (and total figure height, so panels don't
    # shrink to make room) to the longer of the two text boxes — a
    # busy, many-emission-line target must not visually swamp the plot.
    n_text_lines = max(kin_text.count("\n"), info_text.count("\n")) + 1
    top_margin = max(90, 40 + n_text_lines * 14)

    # These two boxes sit in the blank top margin above the first panel.
    # Positioned via "x domain"/"y domain" tied to row 1 rather than
    # "paper" coordinates — Plotly (kaleido export, v6.3.1) renders
    # "paper" y unreliably once a large custom margin is set, effectively
    # clamping it near the plot area instead of the true canvas edge;
    # domain coordinates on an actual row don't have that problem. y > 1
    # here pushes above row 1's own top edge, into the margin; the
    # anchor sits `top_margin` pixels above that edge (approximating row
    # 1's own rendered height as _ROW1_HEIGHT_PX) and the text (yanchor
    # ="top") grows back down to meet it.
    y_top = 1.0 + top_margin / _ROW1_HEIGHT_PX
    figure_annotations = [dict(
        text=kin_text.replace("\n", "<br>"), x=1.0, y=y_top, showarrow=False, align="left",
        xanchor="left", yanchor="top", font=dict(size=9, family="monospace"),
        bgcolor="lightgreen", opacity=0.8, width=_RIGHT_MARGIN - 20,
    )]
    if info_text:
        figure_annotations.append(dict(
            text=info_text.replace("\n", "<br>"), x=0.0, y=y_top, xanchor="right", yanchor="top",
            showarrow=False, align="left",
            font=dict(size=9, family="monospace"),
            bgcolor="wheat", opacity=0.7, width=_LEFT_MARGIN - 20,
        ))

    fig = spectrum_overlay_figure(
        arms, wave_d, flux_d, models_d,
        mask=mask_d,
        extra_traces=extra_d,
        residuals=residuals_d,
        residual_colors=residual_colors,
        residual_labels=residual_labels,
        residual_zero_line=True,
        data_label=({"raw": "Input Spectrum", "stellar_subtracted": "Stellar-Subtracted Spectrum"}
                    if has_stellar else {"raw": "Input Spectrum"}),
        rank_labels=({"raw": ["Stellar + emi Model"], "stellar_subtracted": ["emi Model"]}
                      if has_stellar else ["emi Model"]),
        panel_titles=panel_titles,
        figure_annotations=figure_annotations,
        last_panel_annotations=last_panel_annotations,
        percentile_clip=None,
        flux_unit=flux_labels,
        wave_label="λ (AIR) [Å]",
        width=_FIG_WIDTH,
    )
    # Reserve blank margins for the floating text boxes above (see
    # _LEFT_MARGIN/_RIGHT_MARGIN) — matches legacy's subplots_adjust.
    # Total height grows by the same amount as the top margin so the
    # panels themselves keep their normal size regardless of top_margin.
    fig.update_layout(
        margin=dict(l=_LEFT_MARGIN, r=_RIGHT_MARGIN, t=top_margin, b=60),
        height=fig.layout.height + top_margin,
    )
    return fig


def make_emi_kinematics_plots(rootname, figdir, configs, emi_metalist, emi_result, emission_lines, line_names,
                               logLam, bin_data, emi_bestfit, stellar_bestfit=None, goodpixels_array=None,
                               mc_results=None, debug=False):
    """Drop-in replacement for the legacy matplotlib `make_emi_kinematics_plots`.

    Same signature and `EMIPPXF_<rootname>_APS_ID_<id>.png` output
    convention, now rendered through `build_figure()` + `write_image()`.
    """
    if configs['sens_corr']:
        flux_unit_str = '%2e erg/(s cm**2 Angstrom)' % (configs['funits'])
    else:
        flux_unit_str = 'count'

    from astropy import units
    flux_unit = str(units.Unit(flux_unit_str))

    for i in range(len(emi_metalist)):
        fig_fname = figdir + f'EMIPPXF_{rootname}_APS_ID_{emi_metalist[i]["APS_ID"]}.png'
        fig = build_figure(
            i, emi_metalist, emi_result, emission_lines, line_names,
            logLam, bin_data, emi_bestfit, stellar_bestfit=stellar_bestfit,
            goodpixels_array=goodpixels_array, mc_results=mc_results,
            flux_unit=flux_unit, debug=debug,
        )
        fig.write_image(fig_fname, scale=2)
    return 1
