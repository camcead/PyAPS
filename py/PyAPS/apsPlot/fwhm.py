"""
FWHM diagnostic figures for aps_l1_preview — port of
`L1_preview_interactive.plot_fwhm_all_fibers` (win5, overview) and
`plot_fwhm_detailed` (win6, detail) in the legacy `aps_l1_preview.py`.

Framework-agnostic: both return `plotly.graph_objects.Figure`. FWHM
values themselves are not computed here (same as legacy — they come
from `APSOB.get_fwhm()`'s cached interpolators); these functions only
render arrays the caller has already evaluated.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import style


def _hsv_color(i, n):
    """Distinct per-fiber colour, same HSV-spaced scheme as legacy's
    `QColor.fromHsv(hue, 200, 200)` (hue = i*360/n)."""
    import colorsys
    hue = (i * 360.0 / max(n, 1)) % 360.0
    r, g, b = colorsys.hsv_to_rgb(hue / 360.0, 200 / 255.0, 200 / 255.0)
    return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"


_MAX_CLOUD_FIBERS = 500


def fwhm_overview_figure(arm_names, wavelength_grids, fiber_fwhm, global_fwhm,
                          aps_ids, *, highlighted_aps_id=None, mode_label=None,
                          height_per_row=260, width=900, max_cloud_fibers=_MAX_CLOUD_FIBERS):
    """Build the FWHM overview figure: one panel per arm, each showing
    every fiber's FWHM curve as a thin translucent gray line (the
    "cloud"), a bold black global-FWHM curve on top, and — if
    `highlighted_aps_id` is given — that fiber's curve redrawn bold and
    coloured.

    Parameters
    ----------
    arm_names : sequence[str]
    wavelength_grids : sequence[array]
        Per-arm wavelength grid the FWHM curves were evaluated on.
    fiber_fwhm : dict[int, sequence[array or None]]
        Per-fiber (keyed by APS_ID), per-arm FWHM values.
    global_fwhm : sequence[array]
        Per-arm global (all-fibre) FWHM values.
    aps_ids : sequence[int]
        Fibre APS_IDs, in the same order used to colour-index
        `_hsv_color` (must be stable across calls for consistent colours).
    max_cloud_fibers : int
        Caps how many individual fibres get their own translucent "cloud"
        line — see the comment below for why this exists at all.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    n_arms = len(arm_names)
    fig = make_subplots(rows=n_arms, cols=1, vertical_spacing=0.12)
    n_fibers = len(aps_ids)

    # One Scattergl trace per fibre does not scale: a full LIFU cube can
    # have 30,000+ spaxels (a real 164x188 dataset hit this) — at 1000
    # points/trace and 2 arms that's 60,000+ traces and ~120M floats in a
    # single Plotly figure, serializing to a JSON HTTP response close to a
    # gigabyte. Sending that response over a single connection (an SSH
    # port-forward, in the case that surfaced this) is exactly what
    # crashed the whole Dash server mid-transfer (confirmed: the SSH
    # tunnel log showed a huge sustained byte stream immediately followed
    # by "Connection reset by peer" the moment this figure was rendered
    # after loading that dataset) rather than a genuine memory shortage
    # (this machine has hundreds of GB free) — building/serializing a
    # response this large is itself the problem, independent of how much
    # RAM is technically available. Past max_cloud_fibers, evenly
    # subsample which fibres get their own "cloud" line — visually
    # indistinguishable at these fibre counts (individual lines already
    # blur into an overlapping density well before even a few hundred of
    # them), and does not touch fiber_fwhm itself, so the global curve and
    # (via the separate highlight block below) any single fibre's own
    # detail curve are always drawn in full regardless of this cap.
    cloud_aps_ids = aps_ids
    if n_fibers > max_cloud_fibers:
        # Ceiling division: floor division here can leave the subsample
        # slightly *over* max_cloud_fibers (e.g. 30832 fibres, cap 500 ->
        # floor gives stride 61 -> 506 fibres selected, still above cap).
        stride = max(1, -(-n_fibers // max_cloud_fibers))
        cloud_aps_ids = aps_ids[::stride]

    for row, arm in enumerate(arm_names, start=1):
        wave = wavelength_grids[row - 1]
        for i, aps_id in enumerate(cloud_aps_ids):
            vals = fiber_fwhm.get(aps_id)
            if vals is None or vals[row - 1] is None:
                continue
            fig.add_trace(go.Scattergl(
                x=wave, y=vals[row - 1], mode="lines",
                line=dict(color="rgba(100,100,100,0.12)", width=0.5),
                showlegend=False, hoverinfo="skip",
            ), row=row, col=1)

        gv = global_fwhm[row - 1] if row - 1 < len(global_fwhm) else None
        if gv is not None:
            fig.add_trace(go.Scattergl(
                x=wave, y=gv, mode="lines",
                # Reverted to plain "black" — see apsPlot.style.
                # DATA_COLOR's own comment for why: dark-mode
                # trace-colour visibility is now handled client-side
                # (explorer_ui.js's themePlotlyFigures) instead of at
                # the shared Python source, which pipeline diagnostic
                # exports also depend on.
                line=dict(color="black", width=3),
                name="Global FWHM", legendgroup="global",
                showlegend=(row == 1),
            ), row=row, col=1)

        if highlighted_aps_id is not None and highlighted_aps_id in fiber_fwhm:
            vals = fiber_fwhm[highlighted_aps_id]
            if vals[row - 1] is not None:
                idx = list(aps_ids).index(highlighted_aps_id)
                fig.add_trace(go.Scattergl(
                    x=wave, y=vals[row - 1], mode="lines",
                    line=dict(color=_hsv_color(idx, n_fibers), width=2.5),
                    name=f"Fibre {highlighted_aps_id}", legendgroup="highlight",
                    showlegend=(row == 1),
                ), row=row, col=1)

        fig.update_yaxes(title_text="FWHM [Å]", row=row, col=1)
        fig.add_annotation(
            text=f"<b>{arm}</b>", xref="x domain", yref="y domain",
            x=0.01, y=0.98, showarrow=False, xanchor="left", yanchor="top",
            font=dict(size=11), bgcolor="rgba(255,255,255,0.7)", row=row, col=1,
        )
        if row == n_arms:
            fig.update_xaxes(title_text="Wavelength [Å]", row=row, col=1)

    base_layout = {k: v for k, v in style.BASE_LAYOUT.items() if k != "margin"}
    title = "FWHM Overview"
    if mode_label:
        title += f" — {mode_label}"
    if len(cloud_aps_ids) < n_fibers:
        title += f" — showing {len(cloud_aps_ids):,} of {n_fibers:,} fibres"
    if highlighted_aps_id is not None:
        title += f" — Fibre {highlighted_aps_id} highlighted"
    fig.update_layout(
        **base_layout,
        title=title,
        width=width, height=max(300, height_per_row * n_arms),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def fwhm_detail_figure(arm_names, wavelength_grids, fiber_fwhm, global_fwhm,
                        raw_points, aps_id, *, height_per_row=260, width=900):
    """Build the per-fibre detailed FWHM inspection figure: per arm, a
    dashed gray global-FWHM curve, a bold blue interpolated fibre-FWHM
    curve, and up to 3 scatter categories of raw arc-line measurements
    (used / excluded / large-residual).

    Parameters
    ----------
    raw_points : sequence[dict or None]
        Per-arm dict with keys `wavelengths`, `fwhm`, `weights`
        (optional), `residuals` (optional) — same fields as legacy's
        `meta[setup]['fwhm']['fiber_file_fits'][file]`.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    n_arms = len(arm_names)
    fig = make_subplots(rows=n_arms, cols=1, vertical_spacing=0.12)

    for row, arm in enumerate(arm_names, start=1):
        wave = wavelength_grids[row - 1]

        gv = global_fwhm[row - 1] if row - 1 < len(global_fwhm) else None
        if gv is not None:
            fig.add_trace(go.Scattergl(
                x=wave, y=gv, mode="lines",
                line=dict(color="rgb(150,150,150)", width=2, dash="dash"),
                name="Global FWHM", legendgroup="global", showlegend=(row == 1),
            ), row=row, col=1)

        fv = fiber_fwhm[row - 1] if row - 1 < len(fiber_fwhm) else None
        if fv is not None:
            fig.add_trace(go.Scattergl(
                x=wave, y=fv, mode="lines",
                line=dict(color="blue", width=3),
                name=f"Fibre {aps_id} (interpolated)", legendgroup="fiber",
                showlegend=(row == 1),
            ), row=row, col=1)

        rp = raw_points[row - 1] if row - 1 < len(raw_points) else None
        if rp is not None:
            raw_wave = np.asarray(rp["wavelengths"])
            raw_fwhm = np.asarray(rp["fwhm"])
            weights = np.asarray(rp["weights"]) if rp.get("weights") is not None else None
            residuals = np.asarray(rp["residuals"]) if rp.get("residuals") is not None else None

            if weights is not None:
                excluded = weights == 0
            else:
                excluded = np.zeros(len(raw_wave), dtype=bool)
            good = ~excluded

            if good.any():
                fig.add_trace(go.Scattergl(
                    x=raw_wave[good], y=raw_fwhm[good], mode="markers",
                    marker=dict(symbol="circle", size=8, color="rgba(70,130,180,0.8)",
                                line=dict(color="blue", width=1)),
                    name="Used in fit", legendgroup="used", showlegend=(row == 1),
                ), row=row, col=1)
            if excluded.any():
                fig.add_trace(go.Scattergl(
                    x=raw_wave[excluded], y=raw_fwhm[excluded], mode="markers",
                    marker=dict(symbol="x", size=10, color="rgba(128,128,128,0.6)",
                                line=dict(color="gray", width=2)),
                    name="Excluded (weight=0)", legendgroup="excluded", showlegend=(row == 1),
                ), row=row, col=1)
            if residuals is not None:
                large_resid = (np.abs(residuals) > 0.15) & good
                if large_resid.any():
                    fig.add_trace(go.Scattergl(
                        x=raw_wave[large_resid], y=raw_fwhm[large_resid], mode="markers",
                        marker=dict(symbol="circle-open", size=12, color="orange",
                                    line=dict(color="orange", width=2)),
                        name="Large residual (>0.15Å)", legendgroup="large_resid",
                        showlegend=(row == 1),
                    ), row=row, col=1)

        fig.update_yaxes(title_text="FWHM [Å]", row=row, col=1)
        fig.add_annotation(
            text=f"<b>{arm}</b>", xref="x domain", yref="y domain",
            x=0.01, y=0.98, showarrow=False, xanchor="left", yanchor="top",
            font=dict(size=11), bgcolor="rgba(255,255,255,0.7)", row=row, col=1,
        )
        if row == n_arms:
            fig.update_xaxes(title_text="Wavelength [Å]", row=row, col=1)

    base_layout = {k: v for k, v in style.BASE_LAYOUT.items() if k != "margin"}
    fig.update_layout(
        **base_layout,
        title=f"Detailed FWHM Inspection — Fibre {aps_id}",
        width=width, height=max(300, height_per_row * n_arms),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
