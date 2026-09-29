"""
Generalized "observed spectrum + best-fit model(s)" figure builder.

Shared shape across PyAPS's spectral fitters: Redrock (up to a handful of
ranked models, with an IVAR sub-panel), FERRE and RVS (a single model per
arm), and PPXF (a single model plus a residual trace and masked-pixel
shading). Each module's own `viz.<module>` wrapper extracts its result
table into the primitives below and calls `spectrum_overlay_figure()` —
this function itself knows nothing about Redrock/FERRE/RVS/PPXF.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import style


def _f64(x):
    """Cast to a native-byte-order float64 array.

    FITS-derived arrays (astropy.io.fits) are commonly big-endian; Plotly's
    kaleido export chokes on non-native byte order ("numpy array is not
    native-endianness"). `.astype(np.float64)`/`np.asarray(..., dtype=...)`
    without an explicit endianness character both produce native order, so
    a plain float64 cast is enough — applied once here rather than at every
    call site.
    """
    return np.asarray(x, dtype=np.float64)


def spectrum_overlay_figure(
    arms,
    wave,
    flux,
    models,
    *,
    ivar=None,
    residuals=None,
    residual_colors=None,
    residual_labels=None,
    residual_zero_line=False,
    mask=None,
    mask_color=None,
    highlight_region=None,
    highlight_color="green",
    extra_traces=None,
    data_label="Data",
    rank_labels=None,
    panel_titles=None,
    figure_title=None,
    figure_annotations=None,
    last_panel_annotations=None,
    percentile_clip=(2, 98),
    flux_unit=None,
    wave_label="λ [Å]",
    height_per_row=170,
    width=1100,
):
    """Build a stacked observed-vs-model spectrum overlay figure.

    Parameters
    ----------
    arms : sequence[str]
        Arm/setup labels, e.g. ["blue", "red"]. One row-group per arm.
    wave, flux : dict[str, array]
        Per-arm wavelength and observed-flux arrays, keyed by `arms`.
    models : dict[str, array] or dict[str, sequence[array]]
        Per-arm best-fit model(s). Pass a single array per arm for a
        one-model fit (FERRE, RVS, PPXF), or a list of arrays per arm for
        a multi-rank fit (Redrock).
    ivar : dict[str, array], optional
        Per-arm inverse-variance array. Adds an IVAR sub-panel below each
        flux panel and shades ivar<=0 regions as masked.
    residuals : dict[str, array], optional
        Per-arm residual curve (e.g. PPXF/EMI data-model), drawn as a thin
        sub-panel below the flux panel.
    residual_colors, residual_labels : dict[str, str], optional
        Per-arm overrides for the residual trace's colour/legend name
        (default: a fixed style colour, legend name "Residual").
    residual_zero_line : bool, optional
        Draw a dashed red y=0 reference line on every residual panel.
    mask : dict[str, array[bool]], optional
        Per-arm boolean array (True = masked/bad pixel), shaded on both
        the flux panel and (if present) that arm's residual panel,
        independent of `ivar`.
    mask_color : str, optional
        Fill colour for `mask` shading (default: `style.MASK_COLOR`, red).
        Pass e.g. "grey" for modules where red would clash with other
        overlays (EMI's per-bin diagnostic uses grey).
    highlight_region : dict[str, (float, float)], optional
        Per-arm single contiguous (x0, x1) span to highlight (e.g. EMI's
        "fitted wavelength range" marker) — distinct from `mask`, which
        shades an arbitrary boolean pattern rather than one span.
    highlight_color : str, optional
        Fill colour for `highlight_region` (default "green").
    extra_traces : dict[str, list[dict]], optional
        Per-arm list of additional curves drawn on the flux panel (e.g.
        PPXF's "good pixels only" overlay). Each dict: `{"name": str,
        "y": array, "color": str, "width": float, "dash": str}`. `dash`
        follows Plotly's line-dash values (e.g. "dash", "dot"); omit for
        solid. Shown in the legend once, on the first arm.
    data_label : str or dict[str, str], optional
        Legend name for the observed-data trace (default "Data"). Pass a
        dict keyed by arm for per-arm names (EMI: "Input Spectrum" vs
        "Stellar-Subtracted Spectrum").
    rank_labels : sequence[str] or dict[str, sequence[str]], optional
        Label per model rank when `models[arm]` is a list of ranks. Pass a
        dict keyed by arm when different arms' models mean different
        things (EMI: arm 1 is "stellar + emi", arm 2 is "emi" alone) —
        otherwise the same list is shared across all arms.
    panel_titles : dict[str, str], optional
        Per-arm annotation text (e.g. fitted parameters) shown above that
        arm's first flux panel.
    figure_title : str, optional
        Overall figure title.
    figure_annotations : list[dict], optional
        Extra text boxes positioned relative to the top-most panel's own
        axes, with `y` > 1 pushing them up into the top margin (Plotly's
        `xref="paper"` behaves unreliably combined with a large margin —
        this is a workaround, not `transform=fig.transFigure`). Each dict
        is passed to `fig.add_annotation(xref="x domain", yref="y domain",
        row=1, col=1, **d)`.
    last_panel_annotations : list[dict], optional
        Extra text boxes positioned relative to the bottom-most panel's
        own axes (matplotlib's `transform=ax.transAxes` equivalent on the
        last subplot). Each dict is passed to `fig.add_annotation(xref=
        "x domain", yref="y domain", row=<last row>, col=1, **d)`.
    percentile_clip : (low, high) or None
        Percentile pair used to clip each flux panel's y-range to the
        observed data, or None to autoscale.
    flux_unit : str or dict[str, str], optional
        Y-axis label for flux panels — a single string applied to every
        arm, or a dict keyed by arm for per-arm labels (e.g. EMI's
        "Stellar-Subtracted ..." on its second panel).
        Y-axis label for flux panels.
    wave_label : str, optional
        X-axis label on the bottom panel (e.g. "λ (vacuum) [Å]" or
        "λ (air) [Å]") — wavelength convention varies by module/run, so
        callers must state it explicitly rather than relying on a
        one-size-fits-all default.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    ivar = ivar or {}
    residuals = residuals or {}
    residual_colors = residual_colors or {}
    residual_labels = residual_labels or {}
    mask = mask or {}
    extra_traces = extra_traces or {}
    panel_titles = panel_titles or {}
    figure_annotations = figure_annotations or []
    last_panel_annotations = last_panel_annotations or []

    def panel_title(rk, arm):
        # panel_titles may be keyed by arm (shown once, at rank 0 — the
        # single-model FERRE/RVS/PPXF case) or by (rank, arm) tuples for
        # per-rank annotations (the multi-rank Redrock case).
        if (rk, arm) in panel_titles:
            return panel_titles[(rk, arm)]
        if rk == 0:
            return panel_titles.get(arm)
        return None

    def arm_models(arm):
        m = models.get(arm)
        if m is None:
            return []
        return list(m) if isinstance(m, (list, tuple)) else [m]

    n_ranks = max((len(arm_models(a)) for a in arms), default=1) or 1
    _default_rank_labels = ["Rank 0 (best)"] + [f"Rank {r}" for r in range(1, n_ranks)]
    if rank_labels is None:
        rank_labels = _default_rank_labels

    def model_label(arm, rk):
        # rank_labels may be a flat list (shared across arms — the usual
        # case: FERRE/RVS/PPXF's split-into-halves arms all mean the same
        # thing) or a dict[arm, list] when different arms' models mean
        # different things (EMI: arm 1's model is "stellar + emi", arm
        # 2's is "emi" alone).
        labels = rank_labels.get(arm, _default_rank_labels) if isinstance(rank_labels, dict) else rank_labels
        return labels[rk] if rk < len(labels) else f"Rank {rk}"

    def has_model(arm):
        # Distinguish "no model at all for this arm" (still show one flux
        # row of data-only) from "fewer ranks than other arms" (rows past
        # this arm's own rank count are skipped) — models.get(arm) is None
        # means the former (EMI's panel 1, "Original Data").
        return models.get(arm) is not None

    rows = []
    for rk in range(n_ranks):
        for arm in arms:
            if has_model(arm):
                if rk >= len(arm_models(arm)):
                    continue
            elif rk > 0:
                continue
            rows.append((rk, arm, "flux"))
            if arm in ivar:
                rows.append((rk, arm, "ivar"))
            if arm in residuals:
                rows.append((rk, arm, "residual"))
    if not rows:
        raise ValueError("spectrum_overlay_figure: no arms/models to plot.")

    row_height = {"flux": 2.2, "ivar": 1.0, "residual": 1.0}
    fig = make_subplots(
        rows=len(rows), cols=1, shared_xaxes=False,
        row_heights=[row_height[kind] for _, _, kind in rows],
        vertical_spacing=min(0.35 / len(rows), 0.06),
    )

    yrange = {}
    if percentile_clip is not None:
        lo_pct, hi_pct = percentile_clip
        for arm in arms:
            f = _f64(flux.get(arm)) if flux.get(arm) is not None else None
            valid = f[np.isfinite(f)] if f is not None else np.array([])
            if valid.size:
                ylo, yhi = np.percentile(valid, [lo_pct, hi_pct])
                pad = (yhi - ylo) * 0.1
                yrange[arm] = (ylo - pad, yhi + pad)

    # Collected here and applied once via a single fig.update_layout(shapes=...)
    # at the very end, rather than via fig.add_vrect()/add_hline() calls
    # scattered through the loop below — a real, confirmed Plotly.py
    # performance trap found by direct profiling of a real, live-reported
    # slow Redrock render ("I can quickly see the RVS or FERRE output...
    # but Redrock always take[s] tooooo long"): each add_vrect/add_hline
    # call re-validates and re-resolves *every* shape already on the
    # figure (`_process_multiple_axis_spanning_shapes`), not just the one
    # being added, making a loop of N calls cost roughly O(N^2) — 18
    # add_vrect calls (Redrock's own per-row ivar<=0 masked-region
    # shading, 3 ranks x 2 arms) measured at 0.44s of a 0.66s total build
    # time on real data; RVS/FERRE rarely hit this path at all (no IVAR
    # sub-panel), which is the entire reason this was invisible there.
    # `_shade_regions` below appends plain shape dicts to this list
    # instead of touching `fig` directly; the two standalone
    # add_vrect/add_hline call sites in the loop do the same. Dict
    # schema (`xref`/`yref`/`type`/...) confirmed empirically to be
    # byte-identical to what `add_vrect(row=..., col=1)`/`add_hline(
    # row=..., col=1)` themselves produce for this single-column
    # make_subplots layout (`"x"`/`"y domain"` for row 1, `"x{row}"`/
    # `"y{row} domain"` for row>1, and analogously `"x domain"`/`"y"`
    # for add_hline) — not guessed from documentation, checked directly
    # against real `add_vrect`/`add_hline` output before relying on it.
    # Measured ~47x faster for an equivalent 18-shape figure with
    # byte-identical resulting `layout.shapes`.
    shapes = []

    def _vrect(x0, x1, row, fillcolor, opacity, line_width=0):
        xref = "x" if row == 1 else f"x{row}"
        yref = "y domain" if row == 1 else f"y{row} domain"
        shapes.append(dict(type="rect", x0=x0, x1=x1, y0=0, y1=1,
                            xref=xref, yref=yref, fillcolor=fillcolor,
                            opacity=opacity, line=dict(width=line_width)))

    def _hline(y, row, line, opacity):
        xref = "x domain" if row == 1 else f"x{row} domain"
        yref = "y" if row == 1 else f"y{row}"
        shapes.append(dict(type="line", x0=0, x1=1, y0=y, y1=y,
                            xref=xref, yref=yref, line=line, opacity=opacity))

    # Same batching fix as `shapes`/`_vrect` above, for the exact same
    # reason — profiling this function *again* after the shapes fix
    # landed (explicit follow-up question: "are you sure we do not have
    # such [an] issue in any other plots") found `add_annotation` as the
    # new dominant cost (9 calls, 0.12s of a 0.31s total) via the
    # identical Plotly.py "each call re-validates everything already on
    # the figure" trap. Every real call site here already always passes
    # `xref="x domain", yref="y domain"` literally (the row-scoped
    # "domain" form, never a bare axis or figure-level ref), so the row
    # suffix can just always be computed the same deterministic way
    # `_vrect`/`_hline` do, with no need to accept/parse a caller
    # -supplied ref.
    annotations = []

    def _annotation(row, **kwargs):
        suffix = "" if row == 1 else str(row)
        kwargs["xref"] = f"x{suffix} domain"
        kwargs["yref"] = f"y{suffix} domain"
        annotations.append(kwargs)

    seen_legend_names = set()

    def _showlegend(name):
        # Show a given legend name only the first time it appears in the
        # figure. Repeated identical names (e.g. "Data"/"Model" reused
        # across FERRE/RVS's split-into-halves arms) collapse to one
        # legend entry; distinct names (e.g. EMI's two arms having
        # different model meanings) each get their own entry.
        first = name not in seen_legend_names
        seen_legend_names.add(name)
        return first

    for i, (rk, arm, kind) in enumerate(rows):
        row = i + 1
        w = _f64(wave[arm])

        if kind == "flux":
            f = _f64(flux[arm])
            d_label = data_label.get(arm, "Data") if isinstance(data_label, dict) else data_label
            fig.add_trace(
                go.Scattergl(
                    x=w, y=f, mode="lines",
                    line=dict(color=style.DATA_COLOR, width=1),
                    name=d_label, legendgroup=d_label,
                    showlegend=_showlegend(d_label),
                ),
                row=row, col=1,
            )
            if has_model(arm):
                model = arm_models(arm)[rk]
                label = model_label(arm, rk)
                fig.add_trace(
                    go.Scattergl(
                        x=w, y=_f64(model), mode="lines",
                        line=dict(color=style.rank_color(rk), width=1.3),
                        name=label, legendgroup=label,
                        showlegend=_showlegend(label),
                    ),
                    row=row, col=1,
                )
            for extra in extra_traces.get(arm, []):
                fig.add_trace(
                    go.Scattergl(
                        x=w, y=_f64(extra["y"]), mode="lines",
                        line=dict(
                            color=extra.get("color", "#1f77b4"),
                            width=extra.get("width", 0.8),
                            dash=extra.get("dash"),
                        ),
                        name=extra["name"], legendgroup=extra["name"],
                        showlegend=_showlegend(extra["name"]),
                    ),
                    row=row, col=1,
                )
            if arm in yrange:
                fig.update_yaxes(range=yrange[arm], row=row, col=1)
            arm_flux_unit = flux_unit.get(arm) if isinstance(flux_unit, dict) else flux_unit
            fig.update_yaxes(title_text=arm_flux_unit or "", title_font_size=10, row=row, col=1)

            title_text = panel_title(rk, arm) if arm == arms[0] else None
            if title_text:
                _annotation(row, text=title_text, x=0.0, y=1.2, showarrow=False,
                            align="left", font=dict(size=10))
            _annotation(row, text=f"Arm: {arm}", x=0.01, y=0.88, showarrow=False,
                        font=dict(size=9, color="navy"))
            if row == 1:
                for ann in figure_annotations:
                    _annotation(1, **ann)
            if arm in mask:
                _shade_regions(_vrect, w, np.asarray(mask[arm], dtype=bool), row,
                                color=mask_color)
            if highlight_region and arm in highlight_region:
                x0, x1 = highlight_region[arm]
                _vrect(x0, x1, row, highlight_color, 0.12)

        elif kind == "ivar":
            iv = _f64(ivar[arm])
            iv_plot = iv.copy()
            iv_plot[iv_plot <= 0] = np.nan
            fig.add_trace(
                go.Scattergl(
                    x=w, y=iv_plot, mode="lines", fill="tozeroy",
                    line=dict(color=style.IVAR_COLOR, width=0.8),
                    fillcolor="rgba(76,120,168,0.2)",
                    name="IVAR", showlegend=False,
                ),
                row=row, col=1,
            )
            bad = iv <= 0
            if bad.any():
                _shade_regions(_vrect, w, bad, row)
            fig.update_yaxes(title_text="IVAR", title_font_size=9, row=row, col=1)

        elif kind == "residual":
            res = _f64(residuals[arm])
            fig.add_trace(
                go.Scattergl(
                    x=w, y=res, mode="lines",
                    line=dict(
                        color=residual_colors.get(arm, style.RESIDUAL_COLOR),
                        width=0.8,
                    ),
                    name=residual_labels.get(arm, "Residual"),
                    showlegend=_showlegend(residual_labels.get(arm, "Residual")),
                ),
                row=row, col=1,
            )
            if residual_zero_line:
                _hline(0, row, dict(color="red", dash="dash", width=1), 0.5)
            if arm in mask:
                _shade_regions(_vrect, w, np.asarray(mask[arm], dtype=bool), row,
                                color=mask_color)
            if highlight_region and arm in highlight_region:
                x0, x1 = highlight_region[arm]
                _vrect(x0, x1, row, highlight_color, 0.12)
            fig.update_yaxes(title_text="Resid.", title_font_size=9, row=row, col=1)

        if row == len(rows):
            fig.update_xaxes(title_text=wave_label, row=row, col=1)
            for ann in last_panel_annotations:
                _annotation(row, **ann)

    fig.update_layout(
        **style.BASE_LAYOUT,
        title=figure_title,
        height=max(320, height_per_row * len(rows)),
        width=width,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
        shapes=shapes,
        annotations=annotations,
    )
    return fig


def _shade_regions(add_vrect, wave, bad_mask, row, color=None):
    """Add translucent vrects over contiguous True runs of `bad_mask`, via
    the caller's own `add_vrect(x0, x1, row, fillcolor, opacity)` callback
    (a plain-dict-building closure — see spectrum_overlay_figure's own
    `shapes`/`_vrect` for why this isn't `fig.add_vrect()` directly)."""
    if not bad_mask.any():
        return
    edges = np.flatnonzero(np.diff(np.concatenate(([0], bad_mask.astype(int), [0]))))
    for s, e in zip(edges[0::2], edges[1::2]):
        # `e` is the diff-array index of the falling edge, i.e. one past
        # the run's last True index — subtract 1 or every shaded region
        # extends one pixel beyond the actual masked run.
        e = min(e - 1, len(wave) - 1)
        add_vrect(wave[s], wave[e], row, color or style.MASK_COLOR, 0.15)
