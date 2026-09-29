"""
2D-vs-3D source-detection comparison figure: white-light image + 2D sep
catalog side by side with the same white-light image + 3D matched-filter
candidates (aps_ifu_seg3d.run_seg3d), plus the purity-vs-SNR-threshold
self-check and (when group_purity_scan is supplied) the multi-line GROUP
purity-vs-combined-S/N self-check, on the shared WCS-aware image primitives
(PyAPS.apsPlot.wcs_image) used across the PyAPS.apsPlot platform -- same
house style as source_detection.py's 2D-only figure.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .wcs_image import add_wcs_image, add_ellipse_outline, add_circle_outline
from . import style

# Aperture-circle colours in panel 2: candidates actually merged into the
# seg2d patch table (aps_ifu_seg3d.merge_into_patch_table) get a bold,
# distinct outline; every other post-veto candidate that's shown still
# gets its own (fainter) aperture outline too -- the point is to make
# what "added to 2D segmentation" means, and what size each candidate's
# aperture actually is, both visible at a glance, not to hide the
# candidates that didn't make the cut.
MERGED_COLOR = style.RANK_COLORS[3]  # green -- "added"
CANDIDATE_APERTURE_COLOR = "rgba(255,255,255,0.45)"  # faint -- "shown, not added"


def build_figure(wcs, collapse, objects_2d, candidates_3d, purity_scan, *,
                  radii_factor=7.0, headname=None,
                  merged_ids=None, aperture_arcsec=None, pixscale_arcsec=None,
                  group_purity_scan=None):
    """Build the 2D-vs-3D comparison figure.

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        Celestial WCS of `collapse`'s pixel grid (the spatially-binned
        grid the 3D method actually detected on -- see
        aps_ifu_seg3d.run_seg3d's returned 'wcs').
    collapse : 2D array
        White-light image on that same grid (run_seg3d's returned
        'collapse').
    objects_2d : structured array
        sep.extract's output (needs 'x', 'y', 'a', 'b', 'theta'), from
        run_seg3d's returned 'objects' -- this module's own internal sep
        pass on the binned grid, not necessarily the same catalog a
        sibling ifu_seg2d run produced (see aps_ifu_seg3d module docstring
        for why: different white-light source/grid is possible there).
    candidates_3d : list of (Candidate, is_new)
        run_seg3d's returned 'candidates'. The Nth entry (1-indexed) is
        identified by seg3d_table['id'] == N (candidates_to_table assigns
        id in this same pre-sort order) -- that's how merged_ids below
        lines up with this list.
    purity_scan : list of (threshold, n_pos, n_neg, purity)
        run_seg3d's returned 'purity_scan'.
    group_purity_scan : list of (threshold, n_pos, n_neg, purity), optional
        run_seg3d's returned 'group_purity_scan' -- the same shape as
        purity_scan but for MULTI-LINE GROUPS vs a combined-S/N threshold
        (see aps_ifu_seg3d.post_veto_group_purity_scan /
        combined_group_snr), the calibration merge_into_patch_table's
        group_min_lines/group_min_snr path is actually resolved against.
        None (default) -- the 4th panel is omitted, e.g. for a caller not
        using the multi-line grouping path at all.
    radii_factor : float
        Ellipse radii scale factor for the 2D sep catalog overlay (matches
        ifu_seg2d's self.radii_factor convention).
    headname : str, optional
        Figure title.
    merged_ids : array-like of int, optional
        seg3d_table['id'] values that aps_ifu_seg3d.merge_into_patch_table
        actually added to the seg2d patch table (its 4th return value).
        Those candidates' aperture outline is drawn in MERGED_COLOR;
        every other shown candidate still gets an outline, in
        CANDIDATE_APERTURE_COLOR, so "added" vs "seen but not added" is
        visible without hiding anything that passed the main veto chain.
        None/empty -- nothing marked as merged (e.g. seg3d_merge=False).
    aperture_arcsec : float, optional
        Circular aperture radius (arcsec) to draw around each candidate --
        the same radius aps_ifu_seg3d.merge_into_patch_table gives a
        merged target (seg3d_merge_aperture_arcsec). None -- skip the
        aperture overlay entirely (candidates are still shown as markers).
    pixscale_arcsec : float, optional
        Arcsec/pixel on `collapse`'s grid (run_seg3d's returned
        'pixscale_arcsec', already accounts for spatial_bin) -- needed to
        convert aperture_arcsec into the pixel radius add_circle_outline
        expects. Required together with aperture_arcsec.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    has_group_panel = bool(group_purity_scan)
    if has_group_panel:
        fig = make_subplots(
            rows=3, cols=2,
            specs=[[{}, {}], [{"colspan": 2, "secondary_y": True}, None],
                   [{"colspan": 2, "secondary_y": True}, None]],
            subplot_titles=("2D white-light + SExtractor", "3D matched-filter candidates",
                             "Post-veto purity vs SNR threshold",
                             "Multi-line group purity vs combined S/N threshold"),
            vertical_spacing=0.12, horizontal_spacing=0.08,
            row_heights=[0.5, 0.25, 0.25],
        )
    else:
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{}, {}], [{"colspan": 2, "secondary_y": True}, None]],
            subplot_titles=("2D white-light + SExtractor", "3D matched-filter candidates",
                             "Post-veto purity vs SNR threshold"),
            vertical_spacing=0.16, horizontal_spacing=0.08,
            row_heights=[0.62, 0.38],
        )

    finite = collapse[np.isfinite(collapse)]
    m, s = (float(np.nanmean(finite)), float(np.nanstd(finite))) if finite.size else (0.0, 1.0)

    # --- panel 1: 2D continuum catalog -----------------------------------
    add_wcs_image(fig, 1, 1, collapse, wcs, zmin=m - s, zmax=m + 2 * s)
    for i in range(len(objects_2d)):
        add_ellipse_outline(
            fig, 1, 1, float(objects_2d["x"][i]), float(objects_2d["y"][i]),
            radii_factor * float(objects_2d["a"][i]), radii_factor * float(objects_2d["b"][i]),
            float(objects_2d["theta"][i]) * 180.0 / np.pi, color=style.RANK_COLORS[0], line_width=1.5,
        )

    # --- panel 2: 3D matched-filter candidates ---------------------------
    add_wcs_image(fig, 1, 2, collapse, wcs, zmin=m - s, zmax=m + 2 * s)
    if candidates_3d:
        xs = np.array([c.ix for c, _ in candidates_3d], dtype=float)
        ys = np.array([c.iy for c, _ in candidates_3d], dtype=float)
        snrs = np.array([c.snr for c, _ in candidates_3d], dtype=float)
        is_new = np.array([is_new for _, is_new in candidates_3d], dtype=bool)
        waves = np.array([c.wave for c, _ in candidates_3d], dtype=float)
        symbols = np.where(is_new, "circle", "x")
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="markers",
                marker=dict(size=9, color=snrs, colorscale="Plasma", showscale=True,
                            colorbar=dict(title="SNR", thickness=10, len=0.55, x=1.02, y=0.8),
                            symbol=symbols, line=dict(width=1, color="white")),
                customdata=np.stack([waves, is_new], axis=1),
                hovertemplate="x=%{x}  y=%{y}<br>wave=%{customdata[0]:.1f}A<br>"
                               "SNR=%{marker.color:.2f}<br>no continuum: %{customdata[1]}<extra></extra>",
                name="3D candidates", showlegend=False,
            ),
            row=1, col=2,
        )

        if aperture_arcsec and pixscale_arcsec:
            aperture_px = aperture_arcsec / pixscale_arcsec
            merged_set = set(int(i) for i in merged_ids) if merged_ids is not None else set()
            legend_done = {True: False, False: False}
            for cid, (c, _) in enumerate(candidates_3d, start=1):
                is_merged = cid in merged_set
                add_circle_outline(
                    fig, 1, 2, float(c.ix), float(c.iy), aperture_px,
                    color=MERGED_COLOR if is_merged else CANDIDATE_APERTURE_COLOR,
                    line_width=2.5 if is_merged else 1.0,
                    name=(("added to 2D segmentation" if is_merged else "candidate, not added")
                          if not legend_done[is_merged] else None),
                )
                legend_done[is_merged] = True

    # --- panel 3: purity vs threshold -------------------------------------
    if purity_scan:
        t, n_pos, n_neg, purity = zip(*purity_scan)
        fig.add_trace(go.Scatter(x=t, y=purity, mode="lines+markers", name="purity",
                                  line=dict(color=style.DATA_COLOR, width=2)),
                      row=2, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=t, y=n_pos, mode="lines+markers", name="n_pos (candidates)",
                                  line=dict(color=style.RANK_COLORS[3], width=1.5, dash="dot")),
                      row=2, col=1, secondary_y=True)
        fig.add_trace(go.Scatter(x=t, y=n_neg, mode="lines+markers", name="n_neg (sign-flipped)",
                                  line=dict(color=style.MASK_COLOR, width=1.5, dash="dot")),
                      row=2, col=1, secondary_y=True)
        fig.add_hline(y=0, line=dict(color="gray", width=1, dash="dash"), row=2, col=1,
                      secondary_y=False)
        fig.update_yaxes(title_text="purity = 1 - n_neg/n_pos", row=2, col=1, secondary_y=False)
        fig.update_yaxes(title_text="candidate count", row=2, col=1, secondary_y=True)
        fig.update_xaxes(title_text="SNR threshold", row=2, col=1)

    # --- panel 4 (optional): multi-line group purity vs combined S/N -----
    # Same self-check construction as panel 3, applied to GROUPS (see
    # aps_ifu_seg3d.post_veto_group_purity_scan) instead of individual
    # candidates -- this is the calibration merge_into_patch_table's
    # group_min_lines/group_min_snr path actually resolves its effective
    # threshold against (aps_ifu_seg3d.resolve_group_min_snr), not a
    # cosmetic addition.
    if has_group_panel:
        t, n_pos, n_neg, purity = zip(*group_purity_scan)
        fig.add_trace(go.Scatter(x=t, y=purity, mode="lines+markers", name="group purity",
                                  line=dict(color=style.DATA_COLOR, width=2, dash="solid"),
                                  marker=dict(symbol="diamond")),
                      row=3, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=t, y=n_pos, mode="lines+markers", name="n_pos (real groups)",
                                  line=dict(color=style.RANK_COLORS[3], width=1.5, dash="dot")),
                      row=3, col=1, secondary_y=True)
        fig.add_trace(go.Scatter(x=t, y=n_neg, mode="lines+markers", name="n_neg (sign-flipped groups)",
                                  line=dict(color=style.MASK_COLOR, width=1.5, dash="dot")),
                      row=3, col=1, secondary_y=True)
        fig.add_hline(y=0, line=dict(color="gray", width=1, dash="dash"), row=3, col=1,
                      secondary_y=False)
        fig.update_yaxes(title_text="group purity = 1 - n_neg/n_pos", row=3, col=1, secondary_y=False)
        fig.update_yaxes(title_text="group count", row=3, col=1, secondary_y=True)
        fig.update_xaxes(title_text="combined S/N threshold", row=3, col=1)

    fig.update_layout(
        **{k: v for k, v in style.BASE_LAYOUT.items() if k != "margin"},
        title=dict(text=headname, x=0.5) if headname else None,
        width=980, height=(980 if has_group_panel else 760),
        margin=dict(l=60, r=100, t=90, b=60),
        legend=dict(orientation="h", y=-0.1),
    )
    return fig
