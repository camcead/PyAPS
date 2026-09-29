"""
Per-Voronoi-bin PPXF + emission-line diagnostic figure, built on the
shared viz.spectra overlay builder — port of
PyAPS.IFUExGalEMIPPXF.save_ppxf_emi_plot.

Unlike viz.emi's MOS-EMI figure (raw + stellar-subtracted panels, each a
genuinely different spectrum), this legacy plot shows the *same* spectrum
built up progressively across 4 panels (original data -> + stellar
continuum -> continuum-subtracted + emi model -> complete model
overview). Following the same "virtual arm per panel" approach as
viz.emi (see its module docstring), each of the 4 panels here is one
arm, stacked vertically rather than in the legacy's literal 2x2 grid.

Every arm is given `models[arm] = None` (spectrum_overlay_figure's
data-only row, no builder-drawn model trace) and every coloured curve is
added via `extra_traces` instead — needed because each panel uses a
*different* semantic colour for its "model-like" curves (blue stellar
continuum, red emi model, magenta total model), whereas the builder's
own model-trace slot is coloured by fit rank, not by arm.
"""

from __future__ import annotations

import os

import numpy as np

from .spectra import spectrum_overlay_figure

_FIG_WIDTH = 1500
_LEFT_MARGIN = 200
_RIGHT_MARGIN = 320
_ROW_HEIGHT_PX = 170  # spectrum_overlay_figure's own default height_per_row

_PANEL_TITLES = [
    "1. Original Data",
    "2. Data + Stellar Continuum",
    "3. Continuum-Subtracted + emi Model",
    "4. Complete Model Overview",
]


def _mask_from_goodpixels(goodpixels, n_wave):
    """True = masked/bad pixel; mirrors legacy's mask construction."""
    mask = np.ones(n_wave, dtype=bool)
    if len(goodpixels) > 0:
        valid = goodpixels[goodpixels < n_wave]
        mask[valid] = False
    else:
        mask = np.zeros(n_wave, dtype=bool)
    return mask


def _per_line_kinematics_text(bin_id, mc_results, line_names, emission_lines, pp, debug=False):
    """Faithful port of save_ppxf_emi_plot's per-line kinematics / component
    grouping text-block logic, including its PPXF-solution fallback.

    Returns (kin_text, component_text, valid_kinematics_count).
    """
    kin_text = ""
    component_text = ""
    valid_count = 0

    if mc_results is not None and isinstance(mc_results, dict):
        line_velocities = mc_results.get('line_velocities')
        line_sigmas = mc_results.get('line_sigmas')
        line_components = mc_results.get('line_components')
        component_names = mc_results.get('component_names')
        tie_groups = mc_results.get('tie_groups')

        if (line_velocities is not None and line_sigmas is not None and
                hasattr(line_velocities, '__len__') and hasattr(line_sigmas, '__len__') and
                len(line_velocities) > 0 and len(line_sigmas) > 0 and line_names is not None):

            kin_text = "Per-Line Kinematics:\n"
            component_groups = {}

            for j, line_name in enumerate(line_names):
                if j < len(line_velocities) and j < len(line_sigmas):
                    v_line = line_velocities[j]
                    sigma_line = line_sigmas[j]
                    flux_line = 0.0
                    if (emission_lines is not None and isinstance(emission_lines, np.ndarray)
                            and len(emission_lines) > 0):
                        flux_line = emission_lines[j]
                    comp_info = ""
                    if tie_groups is not None and j < len(tie_groups):
                        group_name = tie_groups[j]
                        comp_info = f" ({group_name})"
                        component_groups.setdefault(group_name, []).append(line_name)
                    elif line_components is not None and j < len(line_components):
                        comp_id = line_components[j]
                        if component_names is not None and comp_id < len(component_names):
                            comp_name = component_names[comp_id]
                            comp_info = f" (C{comp_id}:{comp_name})"
                        else:
                            comp_info = f" (C{comp_id})"
                        comp_key = f"C{comp_id}" if component_names is None else component_names[comp_id]
                        component_groups.setdefault(comp_key, []).append(line_name)

                    if np.isfinite(v_line) and np.isfinite(sigma_line) and flux_line > 0.0:
                        kin_text += f"{line_name}{comp_info}: V={v_line:.1f}, Sigma={sigma_line:.1f} km/s\n"
                        valid_count += 1
                else:
                    kin_text += f"{line_name}: V=N/A, Sigma=N/A km/s\n"

            if len(component_groups) > 1:
                component_text = "Component Groups:\n"
                for comp_name, lines in component_groups.items():
                    component_text += f"{comp_name}: {', '.join(lines)}\n"

                finite_v = [v for v in line_velocities if np.isfinite(v)]
                finite_s = [s for s in line_sigmas if np.isfinite(s)]
                if len(finite_v) > 1 and len(finite_s) > 1:
                    v_range = np.max(finite_v) - np.min(finite_v)
                    s_range = np.max(finite_s) - np.min(finite_s)
                    kind = "Multi-component" if (v_range > 5.0 or s_range > 5.0) else "Single-component"
                    component_text += f"\n{kind}: ΔV={v_range:.1f}, ΔSigma={s_range:.1f} km/s"

            if debug:
                print(f"DEBUG: Bin {bin_id} has {valid_count} valid per-line kinematics")
                if component_groups:
                    print(f"DEBUG: Found {len(component_groups)} component groups: {list(component_groups.keys())}")

    if not kin_text:
        if hasattr(pp, 'sol') and pp.sol is not None:
            sol = pp.sol
            if isinstance(sol, list):
                kin_text = f"Multi-component ({len(sol)} comp.):\n"
                for comp_idx, comp_sol in enumerate(sol[:3]):
                    if len(comp_sol) >= 2:
                        kin_text += f"C{comp_idx}: V={comp_sol[0]:.1f}, Sigma={comp_sol[1]:.1f} km/s\n"
                        valid_count += 1
            elif len(sol) >= 2:
                kin_text = f"emi Kinematics: V={sol[0]:.1f}, Sigma={sol[1]:.1f} km/s\n"
                kin_text += "(Per-line kinematics not available)\n"
                valid_count = 1
        else:
            kin_text = "No valid emi kinematics available\n"
            valid_count = 0

    if not component_text and valid_count > 1 and mc_results is not None:
        line_velocities = mc_results.get('line_velocities', [])
        line_sigmas = mc_results.get('line_sigmas', [])
        if len(line_velocities) > 1 and len(line_sigmas) > 1:
            finite_v = [v for v in line_velocities if np.isfinite(v)]
            finite_s = [s for s in line_sigmas if np.isfinite(s)]
            if len(finite_v) > 1 and len(finite_s) > 1:
                v_range = np.max(finite_v) - np.min(finite_v)
                s_range = np.max(finite_s) - np.min(finite_s)
                kind = "Multi-component detected" if (v_range > 5.0 or s_range > 5.0) else "Single-component fit"
                component_text = f"{kind}: ΔV={v_range:.1f}, ΔSigma={s_range:.1f} km/s"

    return kin_text, component_text, valid_count


def _flux_text(emission_lines, line_names):
    text = ""
    if emission_lines is not None and line_names is not None:
        if isinstance(emission_lines, np.ndarray) and len(emission_lines) > 0:
            text = "Emission Line Fluxes:\n"
            for j, name in enumerate(line_names):
                if j < len(emission_lines):
                    flux_val = emission_lines[j]
                    if np.isfinite(flux_val) and flux_val != 0:
                        text += f"{name}: {flux_val:.2e}\n"
    return text


def build_figure(rootname, pp, data, error, wave, goodpixels, bin_id, *,
                  line_names=None, stellar_component=None, emi_component=None,
                  mc_results=None, emission_lines=None, debug=False):
    """Build the per-bin PPXF+emission diagnostic figure.

    Parameters mirror PyAPS.IFUExGalEMIPPXF.save_ppxf_emi_plot (`error`
    and `emi_component` are accepted but unused there too — kept here
    only for drop-in signature compatibility). This is a faithful port
    of the 4 progressive-buildup panels + floating text boxes, aside
    from stacking the panels vertically instead of a literal 2x2 grid
    (see module docstring).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    wave = np.asarray(wave, dtype=np.float64)
    data = np.asarray(data, dtype=np.float64)
    n = len(wave)
    goodpixels = np.asarray(goodpixels, dtype=int) if goodpixels is not None else np.array([], dtype=int)

    bad = _mask_from_goodpixels(goodpixels, n)
    has_stellar = stellar_component is not None
    stellar = np.asarray(stellar_component, dtype=np.float64) if has_stellar else None
    continuum_subtracted = data - stellar if has_stellar else None

    bestfit = getattr(pp, "bestfit", None)
    emi_valid = (bestfit is not None and np.isfinite(bestfit).all()
                 and float(np.max(np.abs(bestfit))) < 1e10)
    bestfit = np.asarray(bestfit, dtype=np.float64) if bestfit is not None else None

    arms = ["original", "stellar", "continuum_subtracted", "total"]
    wave_d = {a: wave for a in arms}
    flux_d = {a: data for a in arms}
    models_d = {a: None for a in arms}
    mask_d = {a: bad for a in arms}
    extra_d = {a: [] for a in arms}

    if has_stellar:
        extra_d["stellar"].append({"name": "Stellar Continuum", "y": stellar,
                                    "color": "#1f77b4", "width": 1.2})
        extra_d["stellar"].append({"name": "Continuum-Subtracted", "y": continuum_subtracted,
                                    "color": "#17becf", "width": 1.0, "dash": "dash"})

    if has_stellar:
        extra_d["continuum_subtracted"].append({"name": "Continuum-Subtracted Data",
                                                  "y": continuum_subtracted,
                                                  "color": "#17becf", "width": 1.2})
    if emi_valid:
        extra_d["continuum_subtracted"].append({"name": "emi Model (PPXF)", "y": bestfit,
                                                  "color": "#d62728", "width": 1.5})

    if has_stellar and emi_valid:
        total_model = stellar + bestfit
        extra_d["total"].append({"name": "Stellar Continuum", "y": stellar,
                                  "color": "#1f77b4", "width": 1.0})
        extra_d["total"].append({"name": "emi Model", "y": bestfit,
                                  "color": "#d62728", "width": 1.0})
        extra_d["total"].append({"name": "Total Model (Stellar + emi)", "y": total_model,
                                  "color": "#e377c2", "width": 1.5})
    elif has_stellar:
        extra_d["total"].append({"name": "Stellar Continuum (emi model failed)",
                                  "y": stellar, "color": "#1f77b4", "width": 1.2})
    elif emi_valid:
        extra_d["total"].append({"name": "emi Model Only", "y": bestfit,
                                  "color": "#d62728", "width": 1.5})

    highlight_region = None
    if len(goodpixels) > 0:
        x0, x1 = float(wave[goodpixels[0]]), float(wave[goodpixels[-1]])
        highlight_region = {a: (x0, x1) for a in arms}

    kin_text, component_text, valid_kin_count = _per_line_kinematics_text(
        bin_id, mc_results, line_names, emission_lines, pp, debug=debug)
    flux_text = _flux_text(emission_lines, line_names)

    n_text_lines = max(kin_text.count("\n"), flux_text.count("\n")) + 1
    top_margin = max(90, 40 + n_text_lines * 14)
    # The stats box sits below the wavelength axis title/ticks (not
    # instead of them), so its margin needs that ~55px of clearance on
    # top of its own (at most 3-line) text height.
    bottom_margin = 150

    y_top = 1.0 + top_margin / _ROW_HEIGHT_PX
    figure_annotations = [dict(
        text=kin_text.replace("\n", "<br>"), x=1.0, y=y_top, showarrow=False, align="left",
        xanchor="left", yanchor="top", font=dict(size=9, family="monospace"),
        bgcolor="lightgreen", opacity=0.8, width=_RIGHT_MARGIN - 20,
    )]
    if flux_text:
        figure_annotations.append(dict(
            text=flux_text.replace("\n", "<br>"), x=0.0, y=y_top, xanchor="right", yanchor="top",
            showarrow=False, align="left", font=dict(size=9, family="monospace"),
            bgcolor="wheat", opacity=0.7, width=_LEFT_MARGIN - 20,
        ))

    stats_text = f"Bin ID: {bin_id}"
    if hasattr(pp, "chi2"):
        stats_text += f"\nχ²/DOF = {pp.chi2:.2f}"
    if valid_kin_count > 0:
        stats_text += f"\nValid kinematics: {valid_kin_count} lines"
    # y_bottom sits exactly at the canvas's outer edge (bottom_margin px
    # below row 4's own bottom edge) — yanchor="bottom" so the text grows
    # *upward* from there, same margin-relative logic as figure_annotations'
    # y_top/yanchor="top" above row 1, just mirrored.
    y_bottom = -bottom_margin / _ROW_HEIGHT_PX
    last_panel_annotations = [dict(
        text=stats_text.replace("\n", "<br>"), x=0.5, y=y_bottom, showarrow=False, align="center",
        xanchor="center", yanchor="bottom", font=dict(size=10, family="monospace"),
        bgcolor="lightblue", opacity=0.8,
    )]

    fig = spectrum_overlay_figure(
        arms, wave_d, flux_d, models_d,
        mask=mask_d, mask_color="gray",
        highlight_region=highlight_region, highlight_color="green",
        extra_traces=extra_d,
        data_label="Original Data",
        figure_annotations=figure_annotations,
        last_panel_annotations=last_panel_annotations,
        percentile_clip=(1, 99),
        flux_unit="Flux",
        wave_label="Wavelength [Å]",
        width=_FIG_WIDTH,
    )

    # Per-panel titles ("1. Original Data", ...): spectrum_overlay_figure's
    # own panel_titles mechanism only annotates arms[0] (it was designed
    # for the single-shared-banner case, e.g. Redrock's rank header) —
    # here every arm needs its own, so they're added directly instead.
    # Placed *inside* each panel's own top-right corner (not pushed above
    # it via y>1, like the figure-level boxes) since y>1 on a middle row
    # collides with the row above's x-axis tick labels — there's no
    # dedicated top margin between rows the way there is above row 1.
    # Top-right (not top-left) so it doesn't collide with the builder's
    # own unconditional "Arm: <arm>" label, which always sits top-left.
    for row, title_text in enumerate(_PANEL_TITLES, start=1):
        fig.add_annotation(
            text=f"<b>{title_text}</b>", xref="x domain", yref="y domain",
            x=0.99, y=0.99, showarrow=False, align="right",
            xanchor="right", yanchor="top", font=dict(size=12),
            bgcolor="rgba(255,255,255,0.75)", row=row, col=1,
        )

    # Component-groups text box: positioned "below" the kinematics box,
    # attached to the second panel's own domain rather than the figure
    # margin trick used for the top/bottom boxes (it isn't pinned to the
    # very top or bottom of the figure, so it doesn't need one).
    if component_text:
        fig.add_annotation(
            text=component_text.replace("\n", "<br>"), xref="x domain", yref="y domain",
            x=1.0, y=0.5, xanchor="left", yanchor="middle", showarrow=False, align="left",
            font=dict(size=9, family="monospace"), bgcolor="lightcyan", opacity=0.8,
            width=_RIGHT_MARGIN - 20, row=2, col=1,
        )

    fig.update_layout(
        margin=dict(l=_LEFT_MARGIN, r=_RIGHT_MARGIN, t=top_margin, b=bottom_margin),
        height=fig.layout.height + top_margin + bottom_margin,
    )
    return fig


def save_ppxf_emi_plot(rootname, pp, data, error, wave, goodpixels, bin_id,
                        plot_dir, line_names=None, prefix="ppxf_emi",
                        stellar_component=None, emi_component=None,
                        mc_results=None, emission_lines=None, debug=True):
    """Drop-in replacement for the legacy matplotlib `save_ppxf_emi_plot`.

    Same `<plot_dir>/<prefix>_<rootname>_bin_<bin_id:04d>.png` output
    convention, now rendered through `build_figure()` + `write_image()`.
    """
    os.makedirs(plot_dir, exist_ok=True)
    fig = build_figure(
        rootname, pp, data, error, wave, goodpixels, bin_id,
        line_names=line_names, stellar_component=stellar_component,
        emi_component=emi_component, mc_results=mc_results,
        emission_lines=emission_lines, debug=debug,
    )
    plot_filename = os.path.join(plot_dir, f'{prefix}_{rootname}_bin_{bin_id:04d}.png')
    fig.write_image(plot_filename, scale=2)

    if debug:
        print(f"Enhanced emi kinematics diagnostic plot saved: {plot_filename}")
        valid_kin_count = _per_line_kinematics_text(
            bin_id, mc_results, line_names, emission_lines, pp, debug=False)[2]
        if valid_kin_count > 0:
            print(f"  Per-line kinematics: {valid_kin_count} lines with valid data")

    return plot_filename
