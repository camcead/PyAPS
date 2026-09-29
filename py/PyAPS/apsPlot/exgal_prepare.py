"""
IFU data-preparation diagnostic figures (SNR chain, raw/cut/binned/Voronoi
summaries, spatial binning, Voronoi overview), built directly on Plotly —
the port of PyAPS.ExGalPrepare (legacy ExGalPrepare_plots).

Unlike viz.spectra/viz.spatial, this isn't one shared grid builder: the
legacy module itself is a handful of low-level scatter/histogram panel
helpers (`_scatter_map`, `_bin_map`) composed freely into five different
panel layouts (2 to 5 panels, sometimes mixing spatial maps with
histograms). This module mirrors that structure directly — small
per-panel trace builders, and one function per legacy entry point that
composes them into its own `make_subplots` grid.

Scatter markers (not rasterized heatmaps) are used throughout, matching
the legacy `ax.scatter(..., marker="s")` approach — each spaxel/bin is
its own coloured square marker at its true (x, y) position.
"""

from __future__ import annotations

import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import style

# Two-colour map for binary keep/remove plots — same colours as legacy's
# _BINARY_CMAP (ColorBrewer RdBu subset): red=removed, blue=kept.
BINARY_COLORSCALE = [[0.0, "#D6604D"], [0.5, "#D6604D"],
                      [0.5, "#2166AC"], [1.0, "#2166AC"]]


def _f64(x):
    return np.asarray(x, dtype=np.float64)


def _infer_pixelsize(x, fallback=1.0):
    try:
        ux = np.unique(np.round(x, 3))
        if len(ux) > 1:
            return float(np.min(np.diff(np.sort(ux))))
    except Exception:
        pass
    return fallback


# Shared with _finalize_layout, which sizes the canvas using the same
# numbers — _colorbar_for's positions are only correct if the two stay
# in sync (colorbar paper-x > 1 lands in the margin, scaled by how much
# of the canvas *is* margin, not by a fixed guessed offset).
_WIDTH_PER_COL = 330
_MARGIN_PER_PANEL = 90


def _colorbar_for(n_panels, index, ncols, thickness=10):
    """Colorbar geometry for the `index`-th (0-based) of `n_panels`
    colorbar-bearing traces, stacked left to right in the figure's right
    margin — same trick used in viz.spatial for the same reason (Plotly's
    default colorbar placement collides when multiple traces need their
    own).

    Paper x=[0,1] always spans the *plot* area, however many pixels that
    is — margin.r pixels are appended beyond x=1.0 using that exact same
    per-unit scale, so the paper-x needed to reach the far edge of the
    margin is `margin_px / plot_area_px`, not `margin_px / total_px`
    (that second, wrong ratio is what the previous version of this
    function used, and it placed colorbars for panel counts > ~2 back
    inside the last panel instead of in the margin — total_px includes
    the margin itself, so dividing by it always undershoots).
    """
    plot_px = _WIDTH_PER_COL * ncols
    margin_px = _MARGIN_PER_PANEL * n_panels
    max_beyond_one = margin_px / plot_px
    step = max_beyond_one / (n_panels + 1)
    return dict(thickness=thickness, len=0.8, y=0.5,
                x=1.0 + step * (index + 1))


def add_scatter_map(fig, row, col, x, y, values, *, n_cols_total, panel_index, ncols,
                     colorscale="RdYlBu_r", vmin=None, vmax=None,
                     cbar_label="", title=None, marker_size=4):
    """Add a single scatter-map panel (legacy `_scatter_map`)."""
    x = _f64(x)
    y = _f64(y)
    values = _f64(values)
    if x.size == 0:
        fig.add_annotation(text="no data", xref="x domain", yref="y domain",
                            x=0.5, y=0.5, showarrow=False, row=row, col=col)
        return
    vmin = float(np.nanpercentile(values, 2)) if vmin is None else vmin
    vmax = float(np.nanpercentile(values, 98)) if vmax is None else vmax
    if vmin == vmax:
        vmax = vmin + 1.0

    fig.add_trace(
        go.Scattergl(
            x=x, y=y, mode="markers",
            marker=dict(
                color=values, colorscale=colorscale, cmin=vmin, cmax=vmax,
                size=marker_size, symbol="square",
                colorbar=_colorbar_for(n_cols_total, panel_index, ncols),
                colorbar_title_text=cbar_label,
            ),
            showlegend=False,
            hovertemplate=f"{cbar_label}: %{{marker.color:.3g}}<br>x=%{{x:.1f}}\"<br>y=%{{y:.1f}}\"<extra></extra>",
        ),
        row=row, col=col,
    )
    if title:
        fig.add_annotation(text=title, xref="x domain", yref="y domain",
                            x=0.5, y=1.08, showarrow=False,
                            font=dict(size=11), row=row, col=col)
    fig.update_xaxes(title_text="x (arcsec)", row=row, col=col)
    fig.update_yaxes(title_text="y (arcsec)", scaleanchor="x" if panel_index == 0 else None,
                      row=row, col=col)


def add_bin_map(fig, row, col, x, y, bin_num, bin_values, *, n_cols_total, panel_index, ncols,
                 colorscale="RdYlBu_r", vmin=None, vmax=None, cbar_label="",
                 title=None, node_x=None, node_y=None, marker_size=4):
    """Add a per-bin scatter map (legacy `_bin_map`): bin_values are
    mapped back to spaxel level via bin_num, with optional bin-centroid
    star markers overlaid."""
    x = _f64(x)
    y = _f64(y)
    bin_num = np.asarray(bin_num)
    bin_values = _f64(bin_values)
    if x.size == 0 or bin_values.size == 0:
        fig.add_annotation(text="no data", xref="x domain", yref="y domain",
                            x=0.5, y=0.5, showarrow=False, row=row, col=col)
        return

    valid = bin_num >= 0
    x_v, y_v = x[valid], y[valid]
    bnum_v = bin_num[valid].astype(int)
    ubins = np.unique(bnum_v)
    remap = {int(b): i for i, b in enumerate(ubins)}
    bnum_c = np.array([remap[int(b)] for b in bnum_v], dtype=int)
    n_bins = len(bin_values)
    bnum_c = np.clip(bnum_c, 0, n_bins - 1)
    spaxel_vals = bin_values[bnum_c]

    vmin = float(np.nanpercentile(spaxel_vals, 2)) if vmin is None else vmin
    vmax = float(np.nanpercentile(spaxel_vals, 98)) if vmax is None else vmax
    if vmin == vmax:
        vmax = vmin + 1.0

    fig.add_trace(
        go.Scattergl(
            x=x_v, y=y_v, mode="markers",
            marker=dict(
                color=spaxel_vals, colorscale=colorscale, cmin=vmin, cmax=vmax,
                size=marker_size, symbol="square",
                colorbar=_colorbar_for(n_cols_total, panel_index, ncols),
                colorbar_title_text=cbar_label,
            ),
            showlegend=False,
            hovertemplate=f"{cbar_label}: %{{marker.color:.3g}}<extra></extra>",
        ),
        row=row, col=col,
    )
    if node_x is not None and len(node_x) > 0:
        fig.add_trace(
            go.Scattergl(
                x=_f64(node_x), y=_f64(node_y), mode="markers",
                marker=dict(color="red", size=9, symbol="star",
                            line=dict(color="white", width=0.8)),
                showlegend=False, hoverinfo="skip",
            ),
            row=row, col=col,
        )
    if title:
        fig.add_annotation(text=title, xref="x domain", yref="y domain",
                            x=0.5, y=1.08, showarrow=False,
                            font=dict(size=11), row=row, col=col)
    fig.update_xaxes(title_text="x (arcsec)", row=row, col=col)
    fig.update_yaxes(title_text="y (arcsec)", row=row, col=col)


def add_histogram(fig, row, col, datasets, *, xlabel="", ylabel="Density",
                   title=None, vline=None, vline_label=None):
    """Add a histogram panel (density-normalised, semi-transparent
    overlaid series — legacy uses `ax.hist(..., density=True, alpha=...)`).

    `datasets`: list of dict(values, bins, color, label).
    """
    for d in datasets:
        fig.add_trace(
            go.Histogram(
                x=_f64(d["values"]), nbinsx=d.get("nbins", 50),
                histnorm="probability density",
                marker=dict(color=d.get("color")), opacity=d.get("opacity", 0.6),
                name=d.get("label", ""), showlegend=True,
            ),
            row=row, col=col,
        )
    if vline is not None:
        fig.add_vline(x=vline, line=dict(color="black", dash="dash", width=1.2),
                       annotation_text=vline_label, row=row, col=col)
    if title:
        fig.add_annotation(text=title, xref="x domain", yref="y domain",
                            x=0.5, y=1.08, showarrow=False,
                            font=dict(size=11), row=row, col=col)
    fig.update_xaxes(title_text=xlabel, row=row, col=col)
    fig.update_yaxes(title_text=ylabel, row=row, col=col)
    fig.update_layout(barmode="overlay")


def _finalize_layout(fig, n_cols_total, ncols, suptitle=None, height=520):
    """Common figure-level layout: enough right margin for n_cols_total
    stacked colorbars (see `_colorbar_for`), a title, and a sensible
    canvas size. Call once per figure, after all panels are added.

    Uses the same _WIDTH_PER_COL/_MARGIN_PER_PANEL constants as
    `_colorbar_for` so the reserved margin and the colorbar positions
    computed against it stay consistent.
    """
    base_layout = {k: v for k, v in style.BASE_LAYOUT.items() if k != "margin"}
    margin_px = _MARGIN_PER_PANEL * max(n_cols_total, 1)
    fig.update_layout(
        **base_layout,
        title=dict(text=suptitle, x=0.5) if suptitle else None,
        width=_WIDTH_PER_COL * ncols + margin_px,
        height=height,
        margin=dict(l=60, r=40 + margin_px, t=80, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
    )
    return fig


# ---------------------------------------------------------------------------
# 1. plot_snr_stages
# ---------------------------------------------------------------------------

def build_snr_stages_figure(cube, spatial_bins=None, voronoi_data=None,
                             min_snr=None, target_snr=None, headname="target"):
    """Five-panel SNR chain: raw / MIN_SNR cut / spatial-bin SNR /
    SB-filter / Voronoi-bin SNR. Port of ExGalPrepare_plots.plot_snr_stages.
    """
    x = _f64(cube["x"])
    y = _f64(cube["y"])
    snr = _f64(cube["snr"])

    fig = make_subplots(rows=1, cols=5, horizontal_spacing=0.03)

    add_scatter_map(fig, 1, 1, x, y, snr, n_cols_total=5, panel_index=0, ncols=5,
                     cbar_label="SNR", title=f"1. Raw SNR  N={len(x)}")

    if min_snr is not None:
        kept = snr >= float(min_snr)
        add_scatter_map(
            fig, 1, 2, x, y, kept.astype(float), n_cols_total=5, panel_index=1, ncols=5,
            colorscale=BINARY_COLORSCALE, vmin=0.0, vmax=1.0,
            cbar_label="1=keep  0=removed",
            title=f"2. MIN_SNR={float(min_snr):.4g}  kept={int(kept.sum())}  removed={int((~kept).sum())}",
        )
    else:
        fig.add_annotation(text="MIN_SNR not set", xref="x domain", yref="y domain",
                            x=0.5, y=0.5, showarrow=False, row=1, col=2)
        fig.add_annotation(text="2. MIN_SNR cut", xref="x domain", yref="y domain",
                            x=0.5, y=1.08, showarrow=False, font=dict(size=11), row=1, col=2)

    if spatial_bins is not None:
        sx = _f64(spatial_bins["x"])
        sy = _f64(spatial_bins["y"])
        ssnr = _f64(spatial_bins["snr"])
        sflag = np.asarray(spatial_bins.get("flag", np.ones(len(sx), dtype=int)))
        spbin = float(spatial_bins.get("bin_size", 0.5))
        n_vb = int((sflag == 1).sum())
        n_rb = int((sflag == 0).sum())

        add_scatter_map(fig, 1, 3, sx, sy, ssnr, n_cols_total=5, panel_index=2, ncols=5,
                         cbar_label="bin SNR",
                         title=f"3. Spatial-bin SNR  N={len(sx)}  {spbin:.2g} arcsec")
        add_scatter_map(
            fig, 1, 4, sx, sy, sflag.astype(float), n_cols_total=5, panel_index=3, ncols=5,
            colorscale=BINARY_COLORSCALE, vmin=0.0, vmax=1.0,
            cbar_label="1=valid  0=SB-removed",
            title=f"4. SB filter on spatial bins  valid={n_vb}  removed={n_rb}",
        )
    else:
        for c in (3, 4):
            fig.add_annotation(text="No spatial bins", xref="x domain", yref="y domain",
                                x=0.5, y=0.5, showarrow=False, row=1, col=c)

    if voronoi_data is not None:
        vx = _f64(voronoi_data["x"])
        vy = _f64(voronoi_data["y"])
        bnum = np.asarray(voronoi_data["binNum"], dtype=int)
        xNode = _f64(voronoi_data["xNode"])
        yNode = _f64(voronoi_data["yNode"])
        ubins = np.unique(bnum[bnum >= 0])
        rsnr = voronoi_data.get("snr")
        if rsnr is not None:
            rsnr = _f64(rsnr)
            bin_snr = np.array([float(np.median(rsnr[bnum == b])) for b in ubins])
            # No title kwarg here — legacy immediately overwrites panel 5's
            # title with the generic one below regardless, and in Plotly
            # (unlike matplotlib's set_title) adding both would render as
            # two overlapping annotations instead of one replacing the other.
            add_bin_map(fig, 1, 5, vx, vy, bnum, bin_snr, n_cols_total=5, panel_index=4, ncols=5,
                        cbar_label="bin SNR", node_x=xNode, node_y=yNode)
        else:
            fig.add_annotation(text="SNR not in table", xref="x domain", yref="y domain",
                                x=0.5, y=0.5, showarrow=False, row=1, col=5)
    else:
        fig.add_annotation(text="No Voronoi data", xref="x domain", yref="y domain",
                            x=0.5, y=0.5, showarrow=False, row=1, col=5)
    fig.add_annotation(text="5. Voronoi-bin SNR", xref="x domain", yref="y domain",
                        x=0.5, y=1.08, showarrow=False, font=dict(size=11), row=1, col=5)

    return _finalize_layout(fig, n_cols_total=5, ncols=5,
                             suptitle=f"{headname}  -  SNR through preparation chain",
                             height=430)


def plot_snr_stages(cube, spatial_bins=None, voronoi_data=None,
                     min_snr=None, target_snr=None,
                     headname="target", figdir=".", scan_label="", dpi=150):
    """Drop-in replacement for the legacy matplotlib `plot_snr_stages`.

    Same `<figdir>/<headname>_snr_stages<_scan_label>.png` output
    convention, now rendered through `build_snr_stages_figure()` +
    `write_image()`.
    """
    sfx = ("_" + scan_label) if scan_label else ""
    try:
        fig = build_snr_stages_figure(cube, spatial_bins=spatial_bins,
                                       voronoi_data=voronoi_data,
                                       min_snr=min_snr, target_snr=target_snr,
                                       headname=f"{headname}{sfx}")
        path = os.path.join(figdir, f"{headname}_snr_stages{sfx}.png")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.write_image(path, scale=2)
        print("  [PLOT] Saved:", path)
    except Exception as exc:
        print("  Warning: snr_stages failed:", exc)


# ---------------------------------------------------------------------------
# 2. plot_preparation_summary  (four separate figures)
# ---------------------------------------------------------------------------

def _param_str(min_snr, target_snr, spbin_size):
    parts = []
    if min_snr is not None:
        parts.append(f"MIN_SNR={min_snr}")
    if target_snr is not None:
        parts.append(f"TARGET_SNR={target_snr}")
    if spbin_size is not None:
        parts.append(f"SPBIN={spbin_size} arcsec")
    return "  |  ".join(parts)


def build_prep_raw_figure(cube, headname="target", param_str=""):
    """3-panel: Raw SNR / Raw Signal / Raw Noise."""
    x, y = _f64(cube["x"]), _f64(cube["y"])
    snr, sig, nse = _f64(cube["snr"]), _f64(cube["signal"]), _f64(cube["noise"])
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.05)
    add_scatter_map(fig, 1, 1, x, y, snr, n_cols_total=3, panel_index=0, ncols=3,
                     cbar_label="SNR", title=f"Raw SNR  N={len(x)}")
    add_scatter_map(fig, 1, 2, x, y, sig, n_cols_total=3, panel_index=1, ncols=3,
                     colorscale="Inferno", cbar_label="Signal", title="Raw Signal")
    add_scatter_map(fig, 1, 3, x, y, nse, n_cols_total=3, panel_index=2, ncols=3,
                     colorscale="Viridis", cbar_label="Noise", title="Raw Noise")
    return _finalize_layout(fig, n_cols_total=3, ncols=3,
                             suptitle=f"{headname}  -  Raw data  |  {param_str}")


def build_prep_snrcut_figure(cube, min_snr=None, headname="target", param_str=""):
    """3-panel: keep/cut binary map, SNR of kept spaxels, SNR distribution histogram."""
    x, y, snr = _f64(cube["x"]), _f64(cube["y"]), _f64(cube["snr"])
    _min = float(min_snr) if min_snr is not None else 0.0
    kept = snr >= _min

    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.05)
    add_scatter_map(fig, 1, 1, x, y, kept.astype(float), n_cols_total=2, panel_index=0, ncols=3,
                     colorscale=BINARY_COLORSCALE, vmin=0.0, vmax=1.0, cbar_label="1=keep  0=cut",
                     title=f"MIN_SNR={_min:.4g}  kept={int(kept.sum())}  removed={int((~kept).sum())}")
    add_scatter_map(fig, 1, 2, x[kept], y[kept], snr[kept], n_cols_total=2, panel_index=1, ncols=3,
                     cbar_label="SNR", title="SNR of kept spaxels")
    add_histogram(fig, 1, 3, [
        dict(values=snr[~kept], color="#D6604D", label=f"removed ({int((~kept).sum())})"),
        dict(values=snr[kept], color="#2166AC", label=f"kept ({int(kept.sum())})"),
    ], xlabel="SNR", vline=_min, vline_label=f"MIN_SNR={_min:.4g}", title="SNR distribution")
    return _finalize_layout(fig, n_cols_total=2, ncols=3,
                             suptitle=f"{headname}  -  MIN_SNR cut  |  {param_str}")


def build_prep_spatialbin_figure(cube, spatial_bins, target_snr=None, headname="target", param_str=""):
    """4-panel: bin SNR (all), flux filter on raw spaxels, bin signal, SNR distributions."""
    from scipy.spatial import cKDTree

    x, y, snr = _f64(cube["x"]), _f64(cube["y"]), _f64(cube["snr"])
    sx, sy = _f64(spatial_bins["x"]), _f64(spatial_bins["y"])
    ssnr = _f64(spatial_bins["snr"])
    ssig = _f64(spatial_bins["signal"])
    spbin_pxsz = float(spatial_bins.get("bin_size", 0.5))

    half = spbin_pxsz / 2.0
    r = half * np.sqrt(2) * 1.05
    tree_b = cKDTree(np.column_stack([sx, sy]))
    dists, _ = tree_b.query(np.column_stack([x, y]), k=1)
    kept_flux = dists <= r
    n_kf, n_rf = int(kept_flux.sum()), int((~kept_flux).sum())

    fig = make_subplots(rows=1, cols=4, horizontal_spacing=0.04)
    add_scatter_map(fig, 1, 1, sx, sy, ssnr, n_cols_total=2, panel_index=0, ncols=4,
                     cbar_label="bin SNR",
                     title=f"Spatial-bin SNR (all)  N={len(sx)}  {spbin_pxsz:.2g} arcsec  flux-removed={n_rf}")
    add_scatter_map(fig, 1, 2, x, y, kept_flux.astype(float), n_cols_total=2, panel_index=1, ncols=4,
                     colorscale=BINARY_COLORSCALE, vmin=0.0, vmax=1.0, cbar_label="1=kept  0=flux-removed",
                     title=f"Flux filter (raw spaxels)  kept={n_kf}  removed={n_rf}")
    add_scatter_map(fig, 1, 3, sx, sy, ssig, n_cols_total=2, panel_index=2, ncols=4,
                     colorscale="Inferno", cbar_label="bin signal",
                     title="Signal per spatial bin (after flux filter)")
    add_histogram(fig, 1, 4, [
        dict(values=snr[~kept_flux], color="#D6604D", label=f"flux-removed spaxels ({n_rf})", opacity=0.6),
        dict(values=snr[kept_flux], color="#2166AC", label=f"flux-kept spaxels ({n_kf})", opacity=0.6),
        dict(values=ssnr, color="#2166AC", label=f"spatial bins ({len(ssnr)})", opacity=0.7),
    ], xlabel="SNR", title="SNR dists  (filled=spaxels  outline=bins)",
       vline=float(target_snr) if target_snr is not None else None,
       vline_label=f"TARGET_SNR={target_snr}" if target_snr is not None else None)
    return _finalize_layout(fig, n_cols_total=2, ncols=4,
                             suptitle=f"{headname}  -  Spatial binning  |  {param_str}")


def build_prep_voronoi_figure(voronoi_data, target_snr=None, headname="target"):
    """3-panel: Voronoi bin IDs, SNR per bin, N spaxels per bin."""
    vx, vy = _f64(voronoi_data["x"]), _f64(voronoi_data["y"])
    bnum = np.asarray(voronoi_data["binNum"], dtype=int)
    xNode, yNode = _f64(voronoi_data["xNode"]), _f64(voronoi_data["yNode"])
    n_bins = int(voronoi_data["n_bins"])
    ubins = np.unique(bnum[bnum >= 0])
    rsnr_c = voronoi_data.get("snr")
    bin_snr = (np.array([float(np.median(_f64(rsnr_c)[bnum == b])) for b in ubins])
               if rsnr_c is not None else np.ones(len(ubins)))
    ns_c = voronoi_data.get("nspax")
    bin_ns = (np.array([float(np.median(_f64(ns_c)[bnum == b])) for b in ubins])
              if ns_c is not None else np.array([float(np.sum(bnum == b)) for b in ubins]))
    bin_ids = np.arange(len(ubins), dtype=float)

    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.05)
    # Legacy cycles a 20-colour qualitative palette (tab20) for bin IDs;
    # Plotly has no built-in equivalent, so this uses a continuous rainbow
    # scale instead. Visually distinguishes neighbouring bins just as
    # well; doesn't reproduce the exact "every 20th bin looks identical"
    # wraparound, which isn't informative content anyway.
    add_bin_map(fig, 1, 1, vx, vy, bnum, bin_ids, n_cols_total=3, panel_index=0, ncols=3,
                colorscale="Turbo", vmin=0, vmax=float(max(len(ubins) - 1, 1)),
                cbar_label="Bin ID", node_x=xNode, node_y=yNode,
                title=f"Voronoi bin IDs  N={n_bins}  TARGET_SNR={target_snr or '?'}")
    add_bin_map(fig, 1, 2, vx, vy, bnum, bin_snr, n_cols_total=3, panel_index=1, ncols=3,
                cbar_label="bin SNR", node_x=xNode, node_y=yNode, title="SNR per Voronoi bin")
    add_bin_map(fig, 1, 3, vx, vy, bnum, bin_ns, n_cols_total=3, panel_index=2, ncols=3,
                colorscale="Plasma", cbar_label="N spaxels", node_x=xNode, node_y=yNode,
                title="N spaxels per Voronoi bin")
    return _finalize_layout(fig, n_cols_total=3, ncols=3,
                             suptitle=f"{headname}  -  Voronoi binning")


def plot_preparation_summary(cube, spatial_bins=None, voronoi_data=None, configs=None,
                              headname="target", figdir=".", scan_label="", dpi=150):
    """Drop-in replacement for the legacy matplotlib `plot_preparation_summary`.

    Same four `<figdir>/<headname>_prep_{raw,snrcut,spatialbin,voronoi}<_scan_label>.png`
    outputs, each now rendered through the corresponding `build_prep_*_figure()` + `write_image()`.
    """
    configs = configs or {}
    sfx = ("_" + scan_label) if scan_label else ""
    min_snr = configs.get("MIN_SNR")
    target_snr = configs.get("TARGET_SNR")
    spbin_size = configs.get("SPBIN_SIZE_EXGAL")
    param_str = _param_str(min_snr, target_snr, spbin_size)
    hn = f"{headname}{sfx}"

    try:
        fig = build_prep_raw_figure(cube, headname=hn, param_str=param_str)
        path = os.path.join(figdir, f"{headname}_prep_raw{sfx}.png")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.write_image(path, scale=2)
        print("  [PLOT] Saved:", path)
    except Exception as exc:
        print("  Warning: prep_raw failed:", exc)

    try:
        fig = build_prep_snrcut_figure(cube, min_snr=min_snr, headname=hn, param_str=param_str)
        path = os.path.join(figdir, f"{headname}_prep_snrcut{sfx}.png")
        fig.write_image(path, scale=2)
        print("  [PLOT] Saved:", path)
    except Exception as exc:
        print("  Warning: prep_snrcut failed:", exc)

    if spatial_bins is not None:
        try:
            fig = build_prep_spatialbin_figure(cube, spatial_bins, target_snr=target_snr,
                                                headname=hn, param_str=param_str)
            path = os.path.join(figdir, f"{headname}_prep_spatialbin{sfx}.png")
            fig.write_image(path, scale=2)
            print("  [PLOT] Saved:", path)
        except Exception as exc:
            print("  Warning: prep_spatialbin failed:", exc)

    if voronoi_data is not None:
        try:
            fig = build_prep_voronoi_figure(voronoi_data, target_snr=target_snr, headname=hn)
            path = os.path.join(figdir, f"{headname}_prep_voronoi{sfx}.png")
            fig.write_image(path, scale=2)
            print("  [PLOT] Saved:", path)
        except Exception as exc:
            print("  Warning: prep_voronoi failed:", exc)


# ---------------------------------------------------------------------------
# 3. plot_spatial  (drop-in replacement for ExGalPrepare.plot_spatial)
# ---------------------------------------------------------------------------

def build_spatial_figure(cube, spatial_bins=None, color_field="snr", bin_size_label=None):
    """Two-panel: original spaxels | spatially-binned data. Port of
    ExGalPrepare_plots.plot_spatial."""
    x_orig, y_orig = _f64(cube["x"]), _f64(cube["y"])
    c_orig = cube.get(color_field)
    if c_orig is None:
        raise ValueError(f"Field '{color_field}' not found in cube.")
    c_orig = _f64(c_orig)

    ncols = 2 if spatial_bins is not None else 1
    fig = make_subplots(rows=1, cols=ncols, horizontal_spacing=0.06)
    add_scatter_map(fig, 1, 1, x_orig, y_orig, c_orig, n_cols_total=ncols, panel_index=0, ncols=ncols,
                     cbar_label=color_field, title=f"Original spaxels  (N={len(x_orig)})")

    if spatial_bins is not None:
        x_bin, y_bin = _f64(spatial_bins["x"]), _f64(spatial_bins["y"])
        c_bin = spatial_bins.get(color_field)
        if c_bin is None:
            raise ValueError(f"Field '{color_field}' not found in spatial_bins.")
        c_bin = _f64(c_bin)
        pxsz = _infer_pixelsize(x_orig)
        spbin = float(spatial_bins.get("bin_size", pxsz))
        lbl = bin_size_label or spbin
        add_scatter_map(fig, 1, 2, x_bin, y_bin, c_bin, n_cols_total=ncols, panel_index=1, ncols=ncols,
                         cbar_label=color_field,
                         title=f"Spatial bins  {lbl}\"x{lbl}\"  N={len(x_bin)}")

    return _finalize_layout(fig, n_cols_total=ncols, ncols=ncols, height=560)


def plot_spatial(cube, spatial_bins=None, color_field="snr", bin_size_label=None,
                  figdir=".", headname="headname", dpi=150):
    """Drop-in replacement for the legacy matplotlib `plot_spatial`.

    Same `<figdir>/<headname>_sbin[_<bin_size_label>].png` output
    convention, now rendered through `build_spatial_figure()` +
    `write_image()`.
    """
    from PyAPS import ExGalutil
    ExGalutil.prettyOutput_Running(f"Plotting the {color_field} field")
    try:
        fig = build_spatial_figure(cube, spatial_bins=spatial_bins, color_field=color_field,
                                    bin_size_label=bin_size_label)
        fname = (f"{headname}_sbin_{bin_size_label}.png" if bin_size_label
                  else f"{headname}_sbin.png")
        outpath = os.path.join(figdir, fname)
        os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
        fig.write_image(outpath, scale=2)
        ExGalutil.prettyOutput_Done(f"Plotting the {color_field} field in {outpath}", progressbar=True)
    except Exception as exc:
        print("  Warning: plot_spatial failed:", exc)


# ---------------------------------------------------------------------------
# 4. voronoi_visualization  (drop-in replacement)
# ---------------------------------------------------------------------------

# Matplotlib's tab20 palette, reproduced exactly so categorical bin
# colouring looks the same as legacy (cycled every 20 bins, same as
# legacy's `cmap_tab(i % 20)`).
_TAB20 = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
          "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
          "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
          "#17becf", "#9edae5"]


def build_voronoi_overview_figure(spatial_bins, voronoi_data, quantity=None,
                                   title=None, show_filtered=True):
    """Three-panel Voronoi figure: spatial bins by quantity | Voronoi bin
    IDs | spatial-to-Voronoi mapping. Port of
    ExGalPrepare_plots.voronoi_visualization.
    """
    from scipy.spatial import cKDTree

    vor_x, vor_y = _f64(voronoi_data["x"]), _f64(voronoi_data["y"])
    vor_binNum = np.asarray(voronoi_data["binNum"], dtype=int)
    vor_xNode, vor_yNode = _f64(voronoi_data["xNode"]), _f64(voronoi_data["yNode"])

    valid_mask = vor_binNum >= 0
    filtered_mask = ~valid_mask
    n_valid, n_filtered = int(valid_mask.sum()), int(filtered_mask.sum())
    ubins = np.unique(vor_binNum[valid_mask])
    n_bins = len(ubins)

    pxsz = _infer_pixelsize(vor_x[valid_mask] if n_valid else vor_x)
    if spatial_bins is not None and "bin_size" in spatial_bins:
        pxsz = float(spatial_bins["bin_size"])

    bin_colors = [_TAB20[i % 20] for i in range(max(n_bins, 1))]

    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.05)

    # ---- Panel 1: spatial bins by quantity ----
    if spatial_bins is not None and "x" in spatial_bins:
        sx, sy = _f64(spatial_bins["x"]), _f64(spatial_bins["y"])
        q = quantity
        if q is None:
            for cand in ("snr", "signal", "noise", "flux"):
                if cand in spatial_bins:
                    q = cand
                    break
        q_val = _f64(spatial_bins.get(q, np.zeros(len(sx))))
        add_scatter_map(fig, 1, 1, sx, sy, q_val, n_cols_total=1, panel_index=0, ncols=3,
                         colorscale="Viridis", cbar_label=q or "",
                         title=f"Spatial bins  ({q or ''})  N={len(sx)}")
    else:
        fig.add_annotation(text="No spatial_bins", xref="x domain", yref="y domain",
                            x=0.5, y=0.5, showarrow=False, row=1, col=1)

    # ---- Panel 2: Voronoi bin IDs (categorical) ----
    if n_valid > 0:
        remap = {int(b): i for i, b in enumerate(ubins)}
        bnum_c = np.array([remap[int(b)] for b in vor_binNum[valid_mask]], dtype=int)
        point_colors = [bin_colors[i % n_bins] for i in bnum_c]
        fig.add_trace(go.Scattergl(
            x=vor_x[valid_mask], y=vor_y[valid_mask], mode="markers",
            marker=dict(color=point_colors, size=4, symbol="square"),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=2)
    if show_filtered and n_filtered > 0:
        fig.add_trace(go.Scattergl(
            x=vor_x[filtered_mask], y=vor_y[filtered_mask], mode="markers",
            marker=dict(color="lightgray", size=3, symbol="x", opacity=0.4),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=2)
    fig.add_trace(go.Scattergl(
        x=vor_xNode, y=vor_yNode, mode="markers",
        marker=dict(color="red", size=9, symbol="star", line=dict(color="white", width=0.8)),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=2)
    fig.add_annotation(text=f"Voronoi bins  N={n_bins}  valid={n_valid}  filtered={n_filtered}",
                        xref="x domain", yref="y domain", x=0.5, y=1.08, showarrow=False,
                        font=dict(size=11), row=1, col=2)
    fig.update_xaxes(title_text="x (arcsec)", row=1, col=2)
    fig.update_yaxes(title_text="y (arcsec)", row=1, col=2)

    # ---- Panel 3: spatial -> Voronoi mapping (cKDTree) ----
    n_valid_spatial = n_filtered_spatial = 0
    if spatial_bins is not None and "x" in spatial_bins:
        sx, sy = _f64(spatial_bins["x"]), _f64(spatial_bins["y"])
        bin_size = float(spatial_bins.get("bin_size", pxsz))
        half = bin_size / 2.0
        r = half * np.sqrt(2) * 1.01
        centres = np.column_stack([sx, sy])

        has_valid = np.zeros(len(sx), dtype=bool)
        if n_valid > 0:
            tv = cKDTree(np.column_stack([vor_x[valid_mask], vor_y[valid_mask]]))
            dv, _ = tv.query(centres, k=1)
            has_valid = dv <= r

        has_filt = np.zeros(len(sx), dtype=bool)
        if n_filtered > 0:
            tf = cKDTree(np.column_stack([vor_x[filtered_mask], vor_y[filtered_mask]]))
            df, _ = tf.query(centres, k=1)
            has_filt = df <= r

        tn = cKDTree(np.column_stack([vor_xNode, vor_yNode]))
        _, cidx = tn.query(centres, k=1)

        v_idx = np.where(has_valid)[0]
        f_idx = np.where(~has_valid & has_filt)[0]
        e_idx = np.where(~has_valid & ~has_filt)[0]

        if len(v_idx):
            vc = [bin_colors[cidx[i] % n_bins] for i in v_idx]
            fig.add_trace(go.Scattergl(
                x=sx[v_idx], y=sy[v_idx], mode="markers",
                marker=dict(color=vc, size=4, symbol="square", opacity=0.7),
                name=f"in Voronoi bins ({len(v_idx)})", showlegend=True, hoverinfo="skip",
            ), row=1, col=3)
            n_valid_spatial = len(v_idx)
        if len(f_idx):
            fig.add_trace(go.Scattergl(
                x=sx[f_idx], y=sy[f_idx], mode="markers",
                marker=dict(color="lightgray", size=4, symbol="square", opacity=0.5),
                name=f"filtered ({len(f_idx)})", showlegend=True, hoverinfo="skip",
            ), row=1, col=3)
            n_filtered_spatial = len(f_idx)
        if show_filtered and len(e_idx):
            fig.add_trace(go.Scattergl(
                x=sx[e_idx], y=sy[e_idx], mode="markers",
                marker=dict(color="whitesmoke", size=4, symbol="square", opacity=0.2),
                showlegend=False, hoverinfo="skip",
            ), row=1, col=3)

    fig.add_trace(go.Scattergl(
        x=vor_xNode, y=vor_yNode, mode="markers",
        marker=dict(color="red", size=9, symbol="star", line=dict(color="white", width=0.8)),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=3)
    fig.add_annotation(
        text=f"Spatial to Voronoi mapping — coloured={n_valid_spatial}  grey={n_filtered_spatial}",
        xref="x domain", yref="y domain", x=0.5, y=1.08, showarrow=False,
        font=dict(size=11), row=1, col=3)
    fig.update_xaxes(title_text="x (arcsec)", row=1, col=3)
    fig.update_yaxes(title_text="y (arcsec)", row=1, col=3)

    if title is None:
        tsnr = voronoi_data.get("target_snr")
        title = f"Voronoi Binning  N={n_bins}  valid={n_valid}"
        if tsnr:
            title += f"  TARGET_SNR={tsnr}"

    return _finalize_layout(fig, n_cols_total=1, ncols=3, suptitle=title, height=560)


def voronoi_visualization(spatial_bins, voronoi_data, figsize=(21, 7), title=None,
                           save_path=None, dpi=150, cmap="viridis", quantity=None,
                           show_filtered=True, verbose=False):
    """Drop-in replacement for the legacy matplotlib `voronoi_visualization`.

    Renders through `build_voronoi_overview_figure()`. If `save_path` is
    given, writes the PNG there (legacy convention); otherwise returns
    the figure (legacy returns `(fig, axes)` when not saving — this
    returns just the Plotly figure, since there's no axes-list
    equivalent to hand back).
    """
    try:
        fig = build_voronoi_overview_figure(spatial_bins or {}, voronoi_data, quantity=quantity,
                                             title=title, show_filtered=show_filtered)
        if save_path:
            os.makedirs(os.path.dirname(str(save_path)) or ".", exist_ok=True)
            fig.write_image(str(save_path), scale=2)
            print("  [PLOT] Saved:", save_path)
        else:
            return fig
    except Exception as exc:
        print("  Warning: voronoi_visualization failed:", exc)
    return fig


# ---------------------------------------------------------------------------
# 5. plot_all  — convenience wrapper
# ---------------------------------------------------------------------------

def plot_all(cube, spatial_bins=None, voronoi_data=None, configs=None,
             headname="target", figdir=".", scan_label="", dpi=150):
    """Drop-in replacement for the legacy `plot_all`. Runs every plot for
    one patch; each figure is saved and closed independently, same as
    legacy (Plotly figures don't hold the same kind of memory pressure
    matplotlib's `plt.close()`/`gc.collect()` dance was managing, so
    there's no equivalent cleanup step needed here)."""
    configs = configs or {}
    sfx = ("_" + scan_label) if scan_label else ""
    min_snr = configs.get("MIN_SNR")
    target_snr = configs.get("TARGET_SNR")
    spbin_size = configs.get("SPBIN_SIZE_EXGAL")

    plot_snr_stages(cube=cube, spatial_bins=spatial_bins, voronoi_data=voronoi_data,
                     min_snr=min_snr, target_snr=target_snr, headname=headname,
                     figdir=figdir, scan_label=scan_label, dpi=dpi)

    plot_preparation_summary(cube=cube, spatial_bins=spatial_bins, voronoi_data=voronoi_data,
                              configs=configs, headname=headname, figdir=figdir,
                              scan_label=scan_label, dpi=dpi)

    if spatial_bins is not None:
        plot_spatial(cube=cube, spatial_bins=spatial_bins, color_field="snr",
                     bin_size_label=spbin_size, figdir=figdir, headname=headname, dpi=dpi)

    if voronoi_data is not None:
        voronoi_visualization(
            spatial_bins=spatial_bins or {}, voronoi_data=voronoi_data,
            save_path=os.path.join(figdir, f"{headname}_voronoi_overview{sfx}.png"), dpi=dpi,
        )
