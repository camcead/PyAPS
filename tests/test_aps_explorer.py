"""
tests/test_aps_explorer.py - Terminal-runnable smoke tests for aps_explorer.py
================================================================================

No browser/Playwright required — every test here calls straight into the
same plain (undecorated-in-practice) functions the real Dash callbacks in
`aps_explorer.py` use, so a pass here means the underlying data-loading and
routing logic works; it does not by itself prove the browser wiring (clicks,
tab switches, the log panel) is correct — see the manual interface checklist
in `doc/TESTING.md` for that half.

Run with:
    cd <PYAPS_DIR> && pytest tests/test_aps_explorer.py -v

Uses real WEAVE reference data (set PYAPS_TEST_DATA).
Tests are skipped (not failed) if a given reference file isn't found, so the
suite still runs cleanly if the data layout differs.
"""

from __future__ import annotations
import os as _os
# Root of a directory tree holding real WEAVE data (L1/, L2/, CAL/, CAT/ ...). Tests that need
# real data are skipped when it is not available; point PYAPS_TEST_DATA at your copy to run them.
PYAPS_DATA = _os.environ.get("PYAPS_TEST_DATA", "<PYAPS_DATA>")
PYAPS_HOME = _os.environ.get("PYAPS_HOME", _os.path.expanduser("~/PyAPS"))

import os
from pathlib import Path

import pytest

# --------------------------------------------------------------------------- #
# Real reference data (under PYAPS_TEST_DATA) — same files used throughout the
# aps_explorer.py build/verification this session.
# --------------------------------------------------------------------------- #

MOS_OUTPATH = PYAPS_DATA + "/L2/20231027"
MOS_HEADNAME = "single_3029717__single_3029716"

IFU_EXGAL_OUTPATH = PYAPS_DATA + "/L2/20230514"
IFU_EXGAL_HEADNAME = "LWVE_12141807+5936554_01_BR_L1_P0001"

IFU_GAL_OUTPATH = PYAPS_DATA + "/L2/20240715"
IFU_GAL_HEADNAME = "LWVE_19592005+4044163_01_BR_L1_P0002"

# Small MOS-mode L1 file (~960 fibres) — fast enough for the LSF/FWHM test;
# a full LIFU stackcube (~29k spaxels) takes minutes to build the FWHM cache.
L1_SMALL_FILE = PYAPS_DATA + "/L1/20250707/stack_3097460.fit"
CALDIR = PYAPS_DATA + "/CAL"
CATDIR = PYAPS_DATA + "/CAT"

# Real single-exposure, fibre-level LIFU file (mode=MOSLIFU, 600 fibres,
# confirmed real per-fibre NSPEC) — the "L1 at fibre level (MOS or IFU)"
# case the Slit Explorer/true-fibre-size features explicitly target,
# distinct from L1_SMALL_FILE's plain MOS mode.
L1_MOSLIFU_FILE = PYAPS_DATA + "/L1/20250702/single_3096266.fit"

# Real stacked/co-added LIFU cube (mode=LIFU, ~31k spaxels/arm, 2 arms) —
# the "not fibre-level" case both features must gracefully decline for
# (no single real NSPEC/fibre-diameter per spatial position). Same files
# round 14's LSF-cache-staleness investigation used, so already confirmed
# to load in ~25s with a current-format cache.
L1_STACKED_LIFU_FILES = [
    PYAPS_DATA + "/L1/20250912/stackcube_3111051.fit",
    PYAPS_DATA + "/L1/20250912/stackcube_3111052.fit",
]

# A real stacked LIFU cube with a genuine multi-exposure PROV#### chain
# (NCOMB=16, confirmed via direct FITS header inspection) — the
# "contributing exposures" provenance-overlay feature's own real test
# data; L1_STACKED_LIFU_FILES above wasn't reused for this because both
# its files turned out to have NCOMB=1 (a single-exposure "stack"), which
# wouldn't exercise the actual multi-file resolution/dedup logic at all.
L1_STACKED_MULTI_PROV_FILE = PYAPS_DATA + "/L1/20251010/stackcube_3117934.fit"


def _skip_unless_exists(*paths):
    for p in paths:
        if not Path(p).exists():
            pytest.skip(f"reference data not found: {p}")


@pytest.fixture(autouse=True)
def _reset_explorer_state():
    """Each test loads its own dataset before asserting anything, but reset
    ExplorerState.kind first so a failed/skipped test can't leave a
    misleading `EXPLORER.kind` for the next one to accidentally rely on."""
    from PyAPS import aps_explorer as ex
    ex.EXPLORER.kind = None
    ex.EXPLORER.file_info = None
    yield


# --------------------------------------------------------------------------- #
# aps_utils.aps_file_info — schema/obsmode detection (headers only, no data)
# --------------------------------------------------------------------------- #

def test_aps_file_info_mos():
    _skip_unless_exists(f"{MOS_OUTPATH}/{MOS_HEADNAME}_APS.fits")
    from PyAPS.aps_utils import aps_file_info
    info = aps_file_info(f"{MOS_OUTPATH}/{MOS_HEADNAME}_APS.fits")
    assert info["schema"] == "mos"
    assert info["obsmode"] in ("MOS", "MOSLIFU", "MOSMIFU")


def test_aps_file_info_ifu_exgal():
    _skip_unless_exists(f"{IFU_EXGAL_OUTPATH}/{IFU_EXGAL_HEADNAME}_APS.fits")
    from PyAPS.aps_utils import aps_file_info
    info = aps_file_info(f"{IFU_EXGAL_OUTPATH}/{IFU_EXGAL_HEADNAME}_APS.fits")
    assert info["schema"] == "ifu_exgal"
    assert info["obsmode"] in ("LIFU", "MIFU")


def test_aps_file_info_ifu_gal():
    _skip_unless_exists(f"{IFU_GAL_OUTPATH}/{IFU_GAL_HEADNAME}_APS.fits")
    from PyAPS.aps_utils import aps_file_info
    info = aps_file_info(f"{IFU_GAL_OUTPATH}/{IFU_GAL_HEADNAME}_APS.fits")
    assert info["schema"] == "ifu_gal"


# --------------------------------------------------------------------------- #
# L2 routing + loading, via the actual Dash-callback functions (they're
# plain, directly-callable Python functions underneath the @app.callback
# decorator — no Dash server/request context needed to call them).
# --------------------------------------------------------------------------- #

def test_l2_load_mos_by_outpath_headname():
    _skip_unless_exists(f"{MOS_OUTPATH}/{MOS_HEADNAME}_APS.fits")
    from PyAPS import aps_explorer as ex
    load_style, main_style, selected, version, status = ex.handle_l2_load(
        1, None, MOS_OUTPATH, MOS_HEADNAME, 0
    )
    assert status == ""
    assert ex.EXPLORER.kind == "mos"
    assert main_style == ex._main_panel_style(True)
    tabs = ex._tabs_for_current(selected)
    assert any(t.label == "Redrock" for t in tabs)


def test_l2_load_via_full_filename():
    """The "full L2 filename" load-form field, added so users don't have to
    split a known _APS.fits path into outpath+headname by hand."""
    aps_file = f"{IFU_EXGAL_OUTPATH}/{IFU_EXGAL_HEADNAME}_APS.fits"
    _skip_unless_exists(aps_file)
    from PyAPS import aps_explorer as ex
    load_style, main_style, selected, version, status = ex.handle_l2_load(
        1, aps_file, None, None, 0
    )
    assert status == ""
    assert ex.EXPLORER.kind == "ifu"
    tabs = ex._tabs_for_current(selected)
    assert any(t.label == "Processing History" for t in tabs)


def test_l2_load_full_filename_wrong_suffix_errors_cleanly():
    from PyAPS import aps_explorer as ex
    result = ex.handle_l2_load(1, "/some/path/not_an_aps_file.txt", None, None, 0)
    status = result[4]
    assert "_APS.fits" in status


def test_l2_load_bad_path_errors_cleanly():
    """A load failure must come back as a status string, not raise —
    otherwise the Dash callback itself would 500."""
    from PyAPS import aps_explorer as ex
    result = ex.handle_l2_load(1, None, "/nonexistent/path", "doesnotexist", 0)
    status = result[4]
    assert "Load failed" in status


def test_ifu_exgal_map_and_tabs():
    _skip_unless_exists(f"{IFU_EXGAL_OUTPATH}/{IFU_EXGAL_HEADNAME}_APS.fits")
    from PyAPS import aps_explorer as ex
    initial_item = ex._route_and_load_l2(IFU_EXGAL_OUTPATH, IFU_EXGAL_HEADNAME)
    assert ex.EXPLORER.kind == "ifu"
    assert ex._aladin_catalog_points()
    labels = [t.label for t in ex._tabs_for_current(initial_item)]
    assert "Stellar (PPXF)" in labels
    assert "Processing History" in labels


def test_ifu_gal_map_and_tabs():
    _skip_unless_exists(f"{IFU_GAL_OUTPATH}/{IFU_GAL_HEADNAME}_APS.fits")
    from PyAPS import aps_explorer as ex
    initial_item = ex._route_and_load_l2(IFU_GAL_OUTPATH, IFU_GAL_HEADNAME)
    assert ex.EXPLORER.kind == "ifu"
    labels = [t.label for t in ex._tabs_for_current(initial_item)]
    assert "Stellar (RVS)" in labels or "Stellar (FERRE)" in labels


# --------------------------------------------------------------------------- #
# IFU "Processing History" tab — pipeline-only artifacts (target detection,
# segmentation, Voronoi-binning diagnostics) not reproducible from the final
# merged _APS.fits.
# --------------------------------------------------------------------------- #

def test_ifu_processing_history_finds_real_artifacts():
    _skip_unless_exists(f"{IFU_EXGAL_OUTPATH}/{IFU_EXGAL_HEADNAME}_APS.fits")
    from PyAPS import aps_IFUviewer as ifu_mod
    from dash import dash_table, dcc, html

    ifu_mod.STATE.load(IFU_EXGAL_OUTPATH, IFU_EXGAL_HEADNAME)
    content = ifu_mod._processing_history_tab_content()

    n_tables = n_graphs = n_imgs = 0

    def count(node):
        nonlocal n_tables, n_graphs, n_imgs
        if isinstance(node, dash_table.DataTable):
            n_tables += 1
            return
        if isinstance(node, dcc.Graph):
            n_graphs += 1
            return
        if isinstance(node, html.Img):
            n_imgs += 1
            return
        kids = getattr(node, "children", None)
        if kids is None:
            return
        for k in (kids if isinstance(kids, list) else [kids]):
            count(k)

    count(content)
    assert n_tables >= 1, "expected the target-detection table to be found"
    assert n_graphs >= 1, "expected the segmentation-map figure to be found"
    assert n_imgs >= 1, "expected at least one target-selection/binning diagnostic PNG"


def test_ifu_segmentation_figure_serializes_to_json():
    """Regression test for the FITS-big-endian-vs-Plotly bug: fits.getdata()
    arrays are big-endian and previously 500'd at JSON-serialization time
    with `TypeError: numpy array is not native-endianness`."""
    _skip_unless_exists(f"{IFU_EXGAL_OUTPATH}/{IFU_EXGAL_HEADNAME}_APS.fits")
    import plotly.io as pio
    from PyAPS import aps_IFUviewer as ifu_mod
    from dash import dcc

    ifu_mod.STATE.load(IFU_EXGAL_OUTPATH, IFU_EXGAL_HEADNAME)
    content = ifu_mod._processing_history_tab_content()
    graphs = [c for c in content.children if isinstance(c, dcc.Graph)]
    assert graphs, "no segmentation-map Graph found"
    pio.to_json(graphs[0].figure)  # raises on the endianness bug


# --------------------------------------------------------------------------- #
# L1: caldir/catdir → real LSF/FWHM (the bug that started this round of
# work — silently broken with no error shown anywhere in the browser).
# --------------------------------------------------------------------------- #

def test_l1_caldir_catdir_builds_real_fwhm():
    _skip_unless_exists(L1_SMALL_FILE, CALDIR, CATDIR)
    from PyAPS import aps_l1_preview as l1_mod

    l1_mod.load_from_form_fields(
        infiles_text=L1_SMALL_FILE, infiles_list=None, l1ref=None, l2ref=None,
        apsids=None, targsrvy=None, targclass=None, maskids=None,
        area=None, mask_areas_text=None, wlranges_text=None, arms_ratio=None,
        caldir=CALDIR, catdir=CATDIR, configdir=None,
        flags=["sens_corr", "safe_mask_gaps"],
    )
    assert l1_mod.STATE.fwhm_cache is not None, (
        "FWHM cache is None — caldir/catdir LSF interpolator loading is broken again"
    )
    aps_id = int(l1_mod.STATE.targs[0].aps_id)
    overview, detail = l1_mod._build_fwhm_figures(aps_id)
    assert overview is not None
    assert len(overview.data) > 1


def test_l1_raw_toggle_overrides_other_flags():
    """The "load raw/unmodified L1 data" checklist option must force off
    sens_corr/mask_gaps/safe_mask_gaps/tellurics/join_arms/vacuum/fill_gap
    regardless of what the individual checkboxes say."""
    from PyAPS import aps_l1_preview as l1_mod

    captured = {}
    real_load = l1_mod.STATE.load

    def fake_load(args):
        captured["args"] = args

    l1_mod.STATE.load = fake_load
    try:
        l1_mod.load_from_form_fields(
            infiles_text=L1_SMALL_FILE, infiles_list=None, l1ref=None, l2ref=None,
            apsids=None, targsrvy=None, targclass=None, maskids=None,
            area=None, mask_areas_text=None, wlranges_text=None, arms_ratio=None,
            caldir=None, catdir=None, configdir=None,
            flags=["sens_corr", "join_arms", "vacuum", "mask_gaps", "safe_mask_gaps",
                   "tellurics", "fill_gap", "raw"],
        )
    finally:
        l1_mod.STATE.load = real_load

    a = captured["args"]
    for flag in ("sens_corr", "join_arms", "vacuum", "mask_gaps", "safe_mask_gaps",
                 "tellurics", "fill_gap"):
        assert getattr(a, flag) is False, f"{flag} should be forced False by the raw toggle"


def test_l1_comma_separated_filters_parse():
    """APS IDs / target survey / target class / mask IDs / area must accept
    comma-separated multi-value input."""
    from PyAPS import aps_l1_preview as l1_mod

    captured = {}
    real_load = l1_mod.STATE.load
    l1_mod.STATE.load = lambda args: captured.update(args=args)
    try:
        l1_mod.load_from_form_fields(
            infiles_text=L1_SMALL_FILE, infiles_list=None, l1ref=None, l2ref=None,
            apsids="1,2,3", targsrvy="WS2023A1-013,WS2023B2-004",
            targclass="STAR,GALAXY", maskids="4,5",
            area="10.0,20.0,60.0,60.0,0.0", mask_areas_text=None, wlranges_text=None,
            arms_ratio=None, caldir=None, catdir=None, configdir=None, flags=[],
        )
    finally:
        l1_mod.STATE.load = real_load

    a = captured["args"]
    assert a.aps_ids == "1,2,3"
    assert a.targsrvy == "WS2023A1-013,WS2023B2-004"
    assert a.area == "10.0,20.0,60.0,60.0,0.0"


# --------------------------------------------------------------------------- #
# Dash-callback robustness regressions caught this session — a spurious
# dataset-version bump on settings-panel mount was silently reverting a
# user's manual tab click back to the first tab.
# --------------------------------------------------------------------------- #

def test_ifu_maptype_change_is_noop_when_unchanged():
    _skip_unless_exists(f"{IFU_EXGAL_OUTPATH}/{IFU_EXGAL_HEADNAME}_APS.fits")
    from PyAPS import aps_explorer as ex
    from dash import no_update

    ex._route_and_load_l2(IFU_EXGAL_OUTPATH, IFU_EXGAL_HEADNAME)
    current = ex.ifu_mod.STATE.current_maptype
    result = ex.on_ifu_maptype_change(current, 1)
    assert result is no_update


def test_ifu_maptype_category_split():
    """Item 5: a single flat "Map" dropdown became genuinely unusable on a
    large dataset (787 options on one real file) — split into a Category
    dropdown plus a Map dropdown scoped to just that category. Switching
    category must pick that category's first quantity and update
    current_maptype; switching back to the same category must be a no-op
    (same guard pattern as on_ifu_maptype_change)."""
    _skip_unless_exists(f"{IFU_EXGAL_OUTPATH}/{IFU_EXGAL_HEADNAME}_APS.fits")
    from PyAPS import aps_explorer as ex
    from dash import no_update

    ex._route_and_load_l2(IFU_EXGAL_OUTPATH, IFU_EXGAL_HEADNAME)
    groups = ex.ifu_mod._list_maptypes(ex.ifu_mod.STATE.data)
    assert len(groups) >= 1
    current_category = ex.ifu_mod._maptype_category_for(ex.ifu_mod.STATE.data, ex.ifu_mod.STATE.current_maptype)
    assert current_category == groups[0][0] or any(current_category == g[0] for g in groups)

    # No-op guard: re-selecting the current category changes nothing.
    assert ex.on_ifu_maptype_category_change(current_category, 1) is no_update

    # A different category (if this dataset has more than one) actually
    # switches current_maptype to that category's first option.
    other = next((label for label, _ in groups if label != current_category), None)
    if other is not None:
        version = ex.on_ifu_maptype_category_change(other, 1)
        assert version == 2
        assert ex.ifu_mod._maptype_category_for(ex.ifu_mod.STATE.data, ex.ifu_mod.STATE.current_maptype) == other
        expected_first = dict(groups)[other][0][0]
        assert ex.ifu_mod.STATE.current_maptype == expected_first

    # The settings panel itself must render both dropdowns with sane ids.
    panel = ex.ifu_mod._settings_panel()
    ids = set()
    _find_ids(panel, ids)
    assert "maptype-category-dd" in ids
    assert "maptype-dd" in ids
    assert "aon-threshold" not in ids  # item 2: removed entirely


def test_central_item_picks_a_central_point_not_row_zero():
    """Item 11: the default selection on load used to be row 0 of
    whichever table/list (Voronoi-binning/target-list order has no
    relation to spatial position), which routinely landed on a corner/
    edge point. _central_item must pick whichever point is spatially
    closest to the field's own mean position instead — checked directly
    against an independently recomputed nearest-point search, not just
    "it returned something"."""
    _skip_unless_exists(f"{IFU_EXGAL_OUTPATH}/{IFU_EXGAL_HEADNAME}_APS.fits", L1_SMALL_FILE)
    import numpy as np
    from PyAPS import aps_explorer as ex

    ex._route_and_load_l2(IFU_EXGAL_OUTPATH, IFU_EXGAL_HEADNAME)
    table = ex.ifu_mod.STATE.data["table"]
    x, y = np.asarray(table["X"], dtype=np.float64), np.asarray(table["Y"], dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    d2 = np.where(finite, x ** 2 + y ** 2, np.inf)
    expected_bin = int(table["BIN_ID"][int(np.argmin(d2))])
    assert ex._central_item("ifu") == expected_bin

    l1_args = ex.l1_mod._build_arg_parser().parse_args([])
    l1_args.infiles = [L1_SMALL_FILE]
    ex._load_l1(l1_args)
    st = ex.l1_mod.STATE
    ra, dec = st.coord_arr[:, 0], st.coord_arr[:, 1]
    finite = np.isfinite(ra) & np.isfinite(dec)
    # A real gotcha this test itself hit on first run: plain np.argmin over
    # an array containing NaN doesn't skip them — any comparison against a
    # NaN "current minimum" is False, so the very first NaN encountered
    # while scanning freezes the result at that (meaningless) index for
    # good. Every fibre with no real coordinate at all (960 - 418 = 542 on
    # this real file — unused/sky fibres) must be excluded *before*
    # argmin, exactly like _central_item itself does, not just excluded
    # from the mean.
    assert finite.sum() < len(ra), "test fixture no longer exercises the non-finite-coordinate path"
    ra0, dec0 = float(np.nanmean(ra)), float(np.nanmean(dec))
    cos_dec = np.cos(np.radians(dec0))
    d2 = np.where(finite, ((ra - ra0) * cos_dec) ** 2 + (dec - dec0) ** 2, np.inf)
    expected_id = int(st.aps_id_arr[int(np.argmin(d2))])
    assert ex._central_item("l1") == expected_id
    # Sanity: the picked point must really be at/near the minimum distance
    # to the field centre, not an arbitrary point that merely satisfies
    # some other coincidental condition.
    assert d2[int(np.argmin(d2))] == d2.min()
    assert np.isfinite(d2.min())


def test_color_scale_options_present_and_transform_is_monotonic():
    """Item 8: linear/log/sqrt/power/asinh stretch options. Each transform
    must be monotonically non-decreasing over [0,1] (never scrambles which
    end is which) and must map 0->0 and 1->1 exactly (so vmin/vmax always
    land on the two ends of the colour bar regardless of scale)."""
    from PyAPS import aps_explorer as ex
    import numpy as np

    assert set(ex.COLOR_SCALE_OPTIONS) == {"linear", "log", "sqrt", "power", "asinh"}
    frac = np.linspace(0.0, 1.0, 101)
    for scale in ex.COLOR_SCALE_OPTIONS:
        out = ex._apply_color_scale(frac, scale)
        assert out[0] == pytest.approx(0.0, abs=1e-9)
        assert out[-1] == pytest.approx(1.0, abs=1e-9)
        assert np.all(np.diff(out) >= -1e-9)  # monotonic non-decreasing


def test_color_scale_dropdown_wired_for_ifu_l1_mos():
    """The Scale dropdown must exist whenever the colour-range control
    does, default to "linear", and on_color_scale_change must actually
    change STATE.color_scale and refresh the Aladin catalog — mirroring
    on_color_range_typed's own contract."""
    _skip_unless_exists(f"{IFU_EXGAL_OUTPATH}/{IFU_EXGAL_HEADNAME}_APS.fits")
    from PyAPS import aps_explorer as ex
    from dash import no_update

    ex._route_and_load_l2(IFU_EXGAL_OUTPATH, IFU_EXGAL_HEADNAME)
    assert ex.ifu_mod.STATE.color_scale == "linear"
    control = ex._aladin_color_range_control()
    ids = set()
    _find_ids(control, ids)
    assert "color-scale-dd" in ids

    result, crversion = ex.on_color_scale_change("log", 0)
    assert result is not no_update and result["buckets"]
    assert crversion == 1
    assert ex.ifu_mod.STATE.color_scale == "log"
    # no-op guard
    assert ex.on_color_scale_change("log", 1) == (no_update, no_update)
    ex.ifu_mod.STATE.color_scale = "linear"  # restore default for other tests


def test_resolve_palette_override_all_three_states():
    """Explicit request: "is it possible to instead of using the current
    color map, we use something that is black for low signal and white
    for high signal... Can I have it in all maps either 2D or 3D so we
    can switch between this blue to red to black to white?" One shared
    EXPLORER.color_palette setting, not per-kind — checked directly
    against the three real states: "default" leaves whatever colourscale
    the caller already resolved untouched, "jet" always returns "Jet"
    regardless of what was passed in, "greyscale" returns a literal
    black-to-white (not white-to-black — Plotly's own built-in "Greys"
    colourscale runs the *opposite* direction by default, confirmed
    directly rather than assumed) 2-stop colourscale."""
    from PyAPS import aps_explorer as ex

    try:
        ex.EXPLORER.color_palette = "default"
        assert ex._resolve_palette_override("Plasma") == "Plasma"
        assert ex._resolve_palette_override([[0.0, "rgb(1,2,3)"], [1.0, "rgb(4,5,6)"]]) == \
            [[0.0, "rgb(1,2,3)"], [1.0, "rgb(4,5,6)"]]

        ex.EXPLORER.color_palette = "jet"
        assert ex._resolve_palette_override("Plasma") == "Jet"

        ex.EXPLORER.color_palette = "greyscale"
        result = ex._resolve_palette_override("Plasma")
        assert result == ex._GREYSCALE_COLORSCALE
        # Low end is genuinely black, high end genuinely white — not the
        # reverse (Plotly's own "Greys" runs light-to-dark by default).
        assert result[0] == [0.0, "rgb(0,0,0)"]
        assert result[-1] == [1.0, "rgb(255,255,255)"]
    finally:
        ex.EXPLORER.color_palette = "default"  # restore default for other tests


def test_color_palette_dropdown_wired_and_reaches_catalog():
    """The Palette dropdown must exist alongside Scale/Colour range, and
    on_color_palette_change must actually change EXPLORER.color_palette
    and refresh the Aladin catalog with genuinely different colours —
    mirroring on_color_scale_change's own contract exactly, but as one
    EXPLORER-level setting rather than a per-kind STATE field."""
    _skip_unless_exists(f"{IFU_EXGAL_OUTPATH}/{IFU_EXGAL_HEADNAME}_APS.fits")
    from PyAPS import aps_explorer as ex
    from dash import no_update

    ex._route_and_load_l2(IFU_EXGAL_OUTPATH, IFU_EXGAL_HEADNAME)
    assert ex.EXPLORER.color_palette == "default"
    control = ex._aladin_color_range_control()
    ids = set()
    _find_ids(control, ids)
    assert "color-palette-dd" in ids

    payload_default = ex._aladin_catalog_payload()
    colors_default = {b["color"] for b in payload_default["buckets"]}

    try:
        result, crversion = ex.on_color_palette_change("greyscale", 0)
        assert result is not no_update and result["buckets"]
        assert crversion == 1
        assert ex.EXPLORER.color_palette == "greyscale"
        colors_grey = {b["color"] for b in result["buckets"]}
        assert colors_grey != colors_default
        assert all(c.startswith("rgb(") for c in colors_grey)
        # no-op guard
        assert ex.on_color_palette_change("greyscale", 1) == (no_update, no_update)
    finally:
        ex.EXPLORER.color_palette = "default"  # restore default for other tests


def test_refresh_tabs_preserves_manual_selection():
    _skip_unless_exists(f"{IFU_EXGAL_OUTPATH}/{IFU_EXGAL_HEADNAME}_APS.fits")
    from PyAPS import aps_explorer as ex

    initial_item = ex._route_and_load_l2(IFU_EXGAL_OUTPATH, IFU_EXGAL_HEADNAME)
    tabs, value = ex.refresh_tabs(initial_item, 1, "history")
    assert value == "history"


# --------------------------------------------------------------------------- #
# UX feedback round: raw-mode flux units, load_args recall, dataset-info
# panel, and the area/mask_areas unit fix (all root-caused against
# aps_utils.py's actual implementation, not the pre-existing — and wrong —
# CLI help text / docs).
# --------------------------------------------------------------------------- #

def test_raw_load_yields_uncalibrated_flux_label():
    _skip_unless_exists(L1_SMALL_FILE)
    from PyAPS import aps_l1_preview as l1_mod

    l1_mod.load_from_form_fields(
        infiles_text=L1_SMALL_FILE, infiles_list=None, l1ref=None, l2ref=None,
        apsids=None, targsrvy=None, targclass=None, maskids=None,
        area=None, mask_areas_text=None, wlranges_text=None, arms_ratio=None,
        caldir=None, catdir=None, configdir=None,
        flags=["sens_corr", "join_arms", "vacuum", "mask_gaps", "safe_mask_gaps",
               "tellurics", "fill_gap", "raw"],
    )
    aps_id = int(l1_mod.STATE.targs[0].aps_id)
    flux_fig, _ = l1_mod._build_spectra_figure(aps_id)
    label = flux_fig.layout.yaxis.title.text
    assert "erg/s" not in label, f"raw load should not claim calibrated flux units, got: {label!r}"


def test_normal_load_yields_calibrated_flux_label():
    _skip_unless_exists(L1_SMALL_FILE)
    from PyAPS import aps_l1_preview as l1_mod

    l1_mod.load_from_form_fields(
        infiles_text=L1_SMALL_FILE, infiles_list=None, l1ref=None, l2ref=None,
        apsids=None, targsrvy=None, targclass=None, maskids=None,
        area=None, mask_areas_text=None, wlranges_text=None, arms_ratio=None,
        caldir=None, catdir=None, configdir=None,
        flags=["sens_corr", "safe_mask_gaps"],
    )
    aps_id = int(l1_mod.STATE.targs[0].aps_id)
    flux_fig, _ = l1_mod._build_spectra_figure(aps_id)
    assert "erg/s" in flux_fig.layout.yaxis.title.text


def test_ivar_normalization_defaults_to_false():
    """Explicit request: "By default I do not want the ivar normalisation
    for the PyAPS data explorer... make default to False." Checked in
    all three places this default lives: the CLI arg parser itself, the
    "Advanced processing options" checklist's own default (nothing
    loaded yet, a fresh form), and a real load through
    load_from_form_fields with no advanced_flags override at all (the
    same as a user who never touches that checklist)."""
    from PyAPS import aps_l1_preview as l1_mod

    args = l1_mod._build_arg_parser().parse_args([])
    assert args.normalize_ivar is False

    assert "normalize_ivar" not in l1_mod._ADVANCED_FLAG_DEFAULTS

    _skip_unless_exists(L1_SMALL_FILE)
    l1_mod.load_from_form_fields(
        infiles_text=L1_SMALL_FILE, infiles_list=None, l1ref=None, l2ref=None,
        apsids=None, targsrvy=None, targclass=None, maskids=None,
        area=None, mask_areas_text=None, wlranges_text=None, arms_ratio=None,
        caldir=None, catdir=None, configdir=None,
        flags=["sens_corr", "safe_mask_gaps"],
    )
    assert l1_mod.STATE.load_args.normalize_ivar is False


def test_decimate_stride_reduces_count_and_preserves_spatial_extent():
    """Explicit request: "is there a smart way that it only load for
    example one out of each 3 or 4 or whatever... however I do not want
    that change the Spatial." Checked directly against real headers-only
    candidate-list resolution (gen_targlist, no expensive per-target
    processing needed to check this) on a real large LIFU stackcube: a
    stride of N keeps ~1/N of the full candidate list, and the resulting
    RA/Dec span still reaches (within a small fraction of the field, not
    literally identical — a stride can't land exactly on both extremes
    every time) the same min/max as the undecimated full list — the
    field's own on-sky coverage is preserved, only density drops."""
    _skip_unless_exists(*L1_STACKED_LIFU_FILES)
    import numpy as np
    from PyAPS import aps_l1_preview as l1_mod
    from PyAPS.aps_utils import gen_targlist, l1_fileinfo

    infile = L1_STACKED_LIFU_FILES[0]
    info = l1_fileinfo([infile])
    full_ids, idf, aps_info, _, _ = gen_targlist(infile, info["mode"])
    n_full = len(full_ids)
    assert n_full > 1000  # genuinely a large cube, not a trivial case

    ra_full, dec_full = aps_info["TARGRA"], aps_info["TARGDEC"]
    ra_span_full = np.nanmax(ra_full) - np.nanmin(ra_full)
    dec_span_full = np.nanmax(dec_full) - np.nanmin(dec_full)

    for stride in (2, 4):
        decimated = full_ids[::stride]
        assert len(decimated) == pytest.approx(n_full / stride, abs=1)
        idx = [idf[i] for i in decimated]
        ra_span = np.nanmax(ra_full[idx]) - np.nanmin(ra_full[idx])
        dec_span = np.nanmax(dec_full[idx]) - np.nanmin(dec_full[idx])
        # Within 5% of the full field's own span — not shrunk to a crop.
        assert ra_span == pytest.approx(ra_span_full, rel=0.05)
        assert dec_span == pytest.approx(dec_span_full, rel=0.05)

    # Argparse/form defaults: decimation is opt-in, off (None) unless set.
    args = l1_mod._build_arg_parser().parse_args([])
    assert args.decimate_stride is None


def test_decimate_stride_end_to_end_real_load():
    """The full path (not just the headers-only stride math above): a
    real load through _load_dataset with decimate_stride set must
    actually produce exactly that fraction of targets — this is what
    makes the *expensive* per-target APSOB processing loop shorter,
    which is the actual point ("loading all is unnecessary... make the
    cubes much faster")."""
    _skip_unless_exists(*L1_STACKED_LIFU_FILES, CALDIR, CATDIR)
    from PyAPS import aps_l1_preview as l1_mod

    l1_args = l1_mod._build_arg_parser().parse_args([])
    l1_args.infiles = L1_STACKED_LIFU_FILES
    l1_args.caldir, l1_args.catdir = CALDIR, CATDIR
    l1_args.decimate_stride = 4
    l1_mod.STATE.load(l1_args)
    # Confirmed directly against this exact real file (31,132 candidate
    # spaxels for one arm) before writing this assertion, not guessed.
    assert len(l1_mod.STATE.targs) == pytest.approx(31132 / 4, abs=2)


def test_load_args_recorded_and_summarized():
    _skip_unless_exists(L1_SMALL_FILE)
    from PyAPS import aps_l1_preview as l1_mod

    l1_mod.load_from_form_fields(
        infiles_text=L1_SMALL_FILE, infiles_list=None, l1ref=None, l2ref=None,
        apsids=None, targsrvy=None, targclass=None, maskids=None,
        area=None, mask_areas_text=None, wlranges_text=None, arms_ratio=None,
        caldir=CALDIR, catdir=CATDIR, configdir=None,
        flags=["sens_corr", "safe_mask_gaps"],
    )
    assert l1_mod.STATE.load_args is not None
    summary = l1_mod.STATE.load_summary()
    assert isinstance(summary, list) and isinstance(summary[0], dict)
    assert set(summary[0].keys()) == {"Parameter", "Value"}
    joined = " ".join(f"{r['Parameter']}={r['Value']}" for r in summary)
    assert L1_SMALL_FILE in joined
    assert CALDIR in joined
    assert "Sensitivity correction=Yes" in joined


def test_raw_toggle_marked_in_summary():
    from PyAPS import aps_l1_preview as l1_mod

    captured = {}
    real_load = l1_mod.STATE.load
    l1_mod.STATE.load = lambda args: captured.update(args=args) or real_load(args)
    try:
        l1_mod.load_from_form_fields(
            infiles_text=L1_SMALL_FILE, infiles_list=None, l1ref=None, l2ref=None,
            apsids=None, targsrvy=None, targclass=None, maskids=None,
            area=None, mask_areas_text=None, wlranges_text=None, arms_ratio=None,
            caldir=None, catdir=None, configdir=None, flags=["raw"],
        )
    finally:
        l1_mod.STATE.load = real_load
    summary = l1_mod.STATE.load_summary()
    mode_row = next(r for r in summary if r["Parameter"] == "Mode")
    assert "RAW" in mode_row["Value"]


def test_area_format_matches_aps_utils_ground_truth():
    """area/mask_areas must be exactly 5 comma-separated values —
    RA_CENT(deg), DEC_CENT(deg), A(arcsec), B(arcsec), ANGLE(deg CCW) — per
    aps_utils.py's gen_targlist (`EllipseSkyRegion(..., width=Angle(area[2],
    "arcsec"), height=Angle(area[3], "arcsec"), ...)`), not arcmin as the
    (wrong) pre-existing CLI help text and doc/aps_l1_preview.md claimed."""
    from PyAPS import aps_l1_preview as l1_mod

    captured = {}
    real_load = l1_mod.STATE.load
    l1_mod.STATE.load = lambda args: captured.update(args=args)
    try:
        l1_mod.load_from_form_fields(
            infiles_text=L1_SMALL_FILE, infiles_list=None, l1ref=None, l2ref=None,
            apsids=None, targsrvy=None, targclass=None, maskids=None,
            area="246.29,40.90,60.0,60.0,0.0",
            mask_areas_text="246.0,40.0,30.0,30.0,0.0\n247.0,41.0,20.0,20.0,45.0",
            wlranges_text=None, arms_ratio=None, caldir=None, catdir=None, configdir=None,
            flags=[],
        )
    finally:
        l1_mod.STATE.load = real_load
    a = captured["args"]
    assert len([float(x) for x in a.area.split(",")]) == 5
    for line in a.mask_areas:
        assert len([float(x) for x in line.split(",")]) == 5


def test_file_info_panel_shows_l1_files_and_options():
    _skip_unless_exists(L1_SMALL_FILE)
    from PyAPS import aps_explorer as ex

    l1_args = ex.l1_mod._build_arg_parser().parse_args([])
    l1_args.infiles = [L1_SMALL_FILE]
    ex._load_l1(l1_args)
    panel = ex._file_info_panel()
    text = str(panel)
    assert L1_SMALL_FILE in text


def test_file_info_panel_shows_l2_filepath():
    _skip_unless_exists(f"{MOS_OUTPATH}/{MOS_HEADNAME}_APS.fits")
    from PyAPS import aps_explorer as ex

    ex._route_and_load_l2(MOS_OUTPATH, MOS_HEADNAME)
    panel = ex._file_info_panel()
    text = str(panel)
    assert f"{MOS_OUTPATH}/{MOS_HEADNAME}_APS.fits" in text


def test_l1_load_summary_masks_real_paths_when_locked():
    """Direct-call regression guard for the real path exposure reported
    against the always-visible "Current dataset" summary specifically
    (a genuinely separate display from _load_form's own already-masked
    fields, built from a fresh copy of the same load_args) — "the top
    panel where show used param still reveal the full path for L1 and
    CAL and CAT which is dangerous." locked=False (the default) must
    keep showing the real values — this is not a behavior change for
    standalone/unlocked usage."""
    _skip_unless_exists(L1_SMALL_FILE, CALDIR, CATDIR)
    from PyAPS import aps_explorer as ex

    l1_args = ex.l1_mod._build_arg_parser().parse_args([])
    l1_args.infiles = [L1_SMALL_FILE]
    l1_args.caldir, l1_args.catdir = CALDIR, CATDIR
    ex._load_l1(l1_args)

    unlocked = {r["Parameter"]: r["Value"] for r in ex.l1_mod.STATE.load_summary(locked=False)}
    assert unlocked["Files"] == L1_SMALL_FILE
    assert unlocked["caldir"] == CALDIR
    assert unlocked["catdir"] == CATDIR

    locked = {r["Parameter"]: r["Value"] for r in ex.l1_mod.STATE.load_summary(locked=True)}
    assert L1_SMALL_FILE not in locked["Files"]
    assert CALDIR not in locked["caldir"]
    assert CATDIR not in locked["catdir"]
    assert locked["caldir"] == "(using this server's default)"
    assert locked["catdir"] == "(using this server's default)"
    assert "opened via a direct link" in locked["Files"]


def test_file_info_panel_masks_real_paths_when_locked(monkeypatch):
    """_file_info_panel's own wiring of _is_locked() into both the L1
    and L2 branches — same report as the direct load_summary test
    above ("in all modes L1 or L2"), exercised through the actual panel
    function this time rather than load_summary directly."""
    _skip_unless_exists(L1_SMALL_FILE, f"{MOS_OUTPATH}/{MOS_HEADNAME}_APS.fits")
    from PyAPS import aps_explorer as ex

    l1_args = ex.l1_mod._build_arg_parser().parse_args([])
    l1_args.infiles = [L1_SMALL_FILE]
    ex._load_l1(l1_args)
    monkeypatch.setattr(ex, "_is_locked", lambda: True)
    l1_text = str(ex._file_info_panel())
    assert L1_SMALL_FILE not in l1_text
    assert "opened via a direct link" in l1_text

    ex._route_and_load_l2(MOS_OUTPATH, MOS_HEADNAME)
    l2_text = str(ex._file_info_panel())
    assert f"{MOS_OUTPATH}/{MOS_HEADNAME}_APS.fits" not in l2_text
    assert "opened via a direct link" in l2_text


def test_load_panel_style_helper_preserves_sidebar_css():
    """Every Output("load-panel","style") writer must go through
    _load_panel_style — a bare {"display": ...} would silently wipe the
    sidebar's position/width/etc since Dash replaces the whole style dict
    per Output, not merges it."""
    from PyAPS import aps_explorer as ex

    visible = ex._load_panel_style(True)
    hidden = ex._load_panel_style(False)
    assert visible["position"] == "fixed" == hidden["position"]
    assert visible["display"] == "block"
    assert hidden["display"] == "none"


def test_main_panel_style_helper_preserves_padding():
    """Same gotcha, same fix, for main-panel's own padding (added
    alongside the banner — see _banner()'s docstring)."""
    from PyAPS import aps_explorer as ex

    visible = ex._main_panel_style(True)
    hidden = ex._main_panel_style(False)
    assert visible["padding"] == hidden["padding"]
    assert visible["display"] == "block"
    assert hidden["display"] == "none"


def test_banner_present_with_both_logos_and_version():
    """A banner strip must exist with both logos (WEAVE left, camCEAD
    right) and the running PyAPS version — real <img> tags now, not the
    earlier text placeholder."""
    from PyAPS import aps_explorer as ex
    from dash import html

    ids = set()
    _find_ids(ex.serve_layout(), ids)
    assert "app-banner" in ids
    assert "app-toolbar" in ids
    # The earlier text-placeholder slot is gone now real logo images exist.
    assert "banner-logo-slot" not in ids

    banner = ex._banner()

    def _find_imgs(node, out):
        if isinstance(node, html.Img):
            out.append(node)
        kids = getattr(node, "children", None)
        for k in (kids if isinstance(kids, list) else [kids] if kids is not None else []):
            _find_imgs(k, out)

    imgs = []
    _find_imgs(banner, imgs)
    assert len(imgs) == 2
    assert all(img.src.startswith("data:image/png;base64,") for img in imgs)

    banner_text = str(banner)
    assert "WEAVE Data Explorer" in banner_text
    assert f"PyAPS v{ex.PyAPS.__version__}" in banner_text


def test_banner_logos_and_fonts_are_enlarged():
    """Two follow-up rounds of explicit sizing requests, checked directly
    against the real style dicts and the actual bytes on disk, not just "a
    logo renders somehow": (1) both logos 2.5x their pre-this-round pixel
    height, camCEAD using `camCEAD_main_logo.png` specifically (the second
    of two logo-file swaps this session — kept at the same 2.5x size on
    the swap, per explicit "keep the size as the previous one"); (2)
    banner title/version text first went 2x, then was explicitly walked
    back 30% ("now it is too big") to net 1.4x the original — 28px/17px."""
    from PyAPS import aps_explorer as ex
    from dash import html

    assert ex._CAMCEAD_LOGO_URI == ex._logo_data_uri("camCEAD_main_logo.png")
    assert (ex._LOGO_DIR / "camCEAD_main_logo.png").exists()
    assert (ex._LOGO_DIR / "weave_logo.png").exists()

    banner = ex._banner()
    imgs = []

    def _find_imgs(node, out):
        if isinstance(node, html.Img):
            out.append(node)
        kids = getattr(node, "children", None)
        for k in (kids if isinstance(kids, list) else [kids] if kids is not None else []):
            _find_imgs(k, out)

    _find_imgs(banner, imgs)
    heights = sorted(int(img.style["height"].rstrip("px")) for img in imgs)
    assert heights == [95, 110]  # camCEAD, WEAVE — both grown 2.5x from 38px/44px

    banner_text = str(banner)
    assert "'fontSize': '28px'" in banner_text  # title, was 20px -> 40px -> 28px
    assert "'fontSize': '17px'" in banner_text  # version, was 12px -> 24px -> 17px


def test_logo_dir_resolves_inside_installed_package_not_repo_doc():
    """Regression guard for "I do not see the camcead logo when loading
    aps_explorer from the server" — confirmed live to be neither logo,
    not just camCEAD's, and to have nothing to do with screen resolution:
    the old path (three parents up from this file, then /doc) only ever
    resolved correctly for a source checkout run in place. A Docker image
    that only COPYs py/ (this repo's actual Dockerfile) or any plain pip
    install puts this file under .../site-packages/PyAPS/, where three
    parents up lands nowhere near this repo's doc/ directory — silently
    returning None (caught by _logo_data_uri's own except Exception) with
    no error, just a missing image. _LOGO_DIR must resolve to a real
    subdirectory of the installed PyAPS package itself (bundled via
    pyproject.toml's package-data, the same mechanism configs/templates
    already rely on), so it works identically in a checkout, a `pip
    install .`, and this project's own Docker image."""
    import PyAPS
    from PyAPS import aps_explorer as ex
    from pathlib import Path

    package_dir = Path(PyAPS.__file__).resolve().parent
    assert ex._LOGO_DIR == package_dir / "data"
    assert ex._LOGO_DIR.is_relative_to(package_dir)

    import tomllib
    pyproject_path = package_dir.parent.parent / "pyproject.toml"
    package_data = tomllib.loads(pyproject_path.read_text())["tool"]["setuptools"]["package-data"]
    assert "data/*" in package_data.get("PyAPS", []), \
        "pyproject.toml's package-data must bundle PyAPS/data/* or the " \
        "logos silently vanish again from any pip-installed/Docker build"


def test_load_button_lives_in_toolbar_not_floating():
    """The old `position: fixed` floating button (with a hand-tuned pixel
    offset meant to avoid the banner) could still end up overlapping it
    per direct user report. Replaced with a normal, in-flow button inside
    a sticky toolbar row of its own — this test checks that replacement
    actually took: the button is a real descendant of #app-toolbar, not
    styled with `position: fixed` anywhere, and both the banner and
    toolbar use `position: sticky` (which cannot overlap normal-flow
    siblings) rather than `fixed`."""
    from PyAPS import aps_explorer as ex

    banner = ex._banner()
    toolbar = ex._toolbar()
    assert banner.style["position"] == "sticky"
    assert toolbar.style["position"] == "sticky"

    def _find(node, node_id):
        if getattr(node, "id", None) == node_id:
            return node
        kids = getattr(node, "children", None)
        for k in (kids if isinstance(kids, list) else [kids] if kids is not None else []):
            found = _find(k, node_id)
            if found is not None:
                return found
        return None

    button = _find(toolbar, "open-load-panel-btn")
    assert button is not None
    assert button.style.get("position") != "fixed"

    # And it's reachable from the real page layout too, not just when
    # _toolbar() is called in isolation.
    assert _find(ex.serve_layout(), "open-load-panel-btn") is not None


# --------------------------------------------------------------------------- #
# Second UX feedback round: scrollZoom config, interactive colour-range
# control, and the Aladin Lite target dispatch.
# --------------------------------------------------------------------------- #

def test_graph_config_scrollzoom():
    """scrollZoom was enabled (round 3), dropped again (round 5 — mouse-
    wheel zoom "shook"/jittered, reportedly on every plot with it), then
    re-enabled everywhere once the real cause was found and fixed: a
    dcc.Graph zoom-reset bug that no longer applies at all now the Plotly
    coordinate/map figure has been dropped in favour of the Aladin panel
    (round 6). aps_explorer.py itself no longer owns any dcc.Graph (and so
    no _GRAPH_CONFIG) — only the three library modules' own tab-content
    plots (spectra/FWHM/Redrock/etc.) do."""
    from PyAPS import aps_l1_preview as l1_mod
    from PyAPS import aps_IFUviewer as ifu_mod
    from PyAPS import aps_MOSviewer as mos_mod

    for mod in (l1_mod, ifu_mod, mos_mod):
        assert mod._GRAPH_CONFIG.get("scrollZoom") is True


def test_ifu_color_range_round_trips_into_figure():
    _skip_unless_exists(f"{IFU_EXGAL_OUTPATH}/{IFU_EXGAL_HEADNAME}_APS.fits")
    from PyAPS import aps_IFUviewer as ifu_mod

    ifu_mod.STATE.load(IFU_EXGAL_OUTPATH, IFU_EXGAL_HEADNAME)
    assert ifu_mod.STATE.color_vmin is None and ifu_mod.STATE.color_vmax is None
    ifu_mod.STATE.color_vmin, ifu_mod.STATE.color_vmax = 5.0, 50.0
    fig = ifu_mod._build_map_figure(ifu_mod.STATE.current_maptype, None)
    assert fig.data[0].marker.cmin == 5.0
    assert fig.data[0].marker.cmax == 50.0


def test_color_range_reset_clears_override():
    """on_color_range_reset must clear the STATE override, refresh the
    Aladin catalog's bucket colours to match, and blank the two number
    inputs — the map figure itself no longer exists to worry about
    zoom-preservation for (round 6: the Plotly map was dropped). Also
    bumps color-range-version (a later addition) so a Colour range/Scale
    change reaches the 3D flux cube too — see update_map_mode's own
    docstring for why that can't be a direct Input there instead."""
    _skip_unless_exists(f"{IFU_EXGAL_OUTPATH}/{IFU_EXGAL_HEADNAME}_APS.fits")
    from PyAPS import aps_explorer as ex
    from dash import no_update

    ex._route_and_load_l2(IFU_EXGAL_OUTPATH, IFU_EXGAL_HEADNAME)
    ex.ifu_mod.STATE.color_vmin, ex.ifu_mod.STATE.color_vmax = 5.0, 50.0
    catalog, vmin_out, vmax_out, crversion = ex.on_color_range_reset(1, 0)
    assert catalog is not no_update and catalog["buckets"]
    assert vmin_out is None and vmax_out is None
    assert crversion == 1
    assert ex.ifu_mod.STATE.color_vmin is None
    assert ex.ifu_mod.STATE.color_vmax is None
    # a second reset with nothing to clear is a no-op
    assert ex.on_color_range_reset(2, 1) == (no_update, no_update, no_update, no_update)


def test_color_range_typed_updates_aladin_catalog():
    """on_color_range_typed applies the new vmin/vmax and refreshes the
    Aladin catalog's bucket colours to match — the primary (only, since
    round 6) consumer of the colour range now that the Plotly map is
    gone. Also bumps color-range-version (see test_color_range_reset_
    clears_override's own docstring for why)."""
    _skip_unless_exists(f"{IFU_EXGAL_OUTPATH}/{IFU_EXGAL_HEADNAME}_APS.fits")
    from PyAPS import aps_explorer as ex
    from dash import no_update

    ex._route_and_load_l2(IFU_EXGAL_OUTPATH, IFU_EXGAL_HEADNAME)
    catalog, crversion = ex.on_color_range_typed(3.0, 40.0, 0)
    assert catalog is not no_update and catalog["buckets"]
    assert crversion == 1
    assert ex.ifu_mod.STATE.color_vmin == 3.0
    assert ex.ifu_mod.STATE.color_vmax == 40.0
    # re-firing with the same values (e.g. the mount-time spurious fire) is a no-op
    assert ex.on_color_range_typed(3.0, 40.0, 1) == (no_update, no_update)


def test_l1_color_range_defaults_to_12():
    """fiber_map_figure's own default vmin=12.0 must survive when no
    override is set (STATE.color_vmin is None) — passing None straight
    through would silently disable the lower colour clip."""
    _skip_unless_exists(L1_SMALL_FILE)
    from PyAPS import aps_l1_preview as l1_mod

    args = l1_mod._build_arg_parser().parse_args([])
    args.infiles = [L1_SMALL_FILE]
    l1_mod.STATE.load(args)
    assert l1_mod.STATE.color_vmin is None
    fig = l1_mod._build_fiber_map_figure(None)
    assert fig.data[0].marker.cmin == 12.0


def test_aladin_target_dispatch_all_kinds():
    _skip_unless_exists(f"{MOS_OUTPATH}/{MOS_HEADNAME}_APS.fits",
                         f"{IFU_EXGAL_OUTPATH}/{IFU_EXGAL_HEADNAME}_APS.fits",
                         L1_SMALL_FILE)
    from PyAPS import aps_explorer as ex
    import numpy as np

    def _check(target):
        assert target is not None
        assert np.isfinite(target["ra"]) and 0 <= target["ra"] < 360
        assert np.isfinite(target["dec"]) and -90 <= target["dec"] <= 90
        assert np.isfinite(target["fov"]) and target["fov"] > 0

    ex._route_and_load_l2(MOS_OUTPATH, MOS_HEADNAME)
    _check(ex._aladin_target(ex.mos_mod.STATE.selected_aps_id))

    ex._route_and_load_l2(IFU_EXGAL_OUTPATH, IFU_EXGAL_HEADNAME)
    initial_bin = int(ex.ifu_mod.STATE.data["table"]["BIN_ID"][0])
    _check(ex._aladin_target(initial_bin))

    l1_args = ex.l1_mod._build_arg_parser().parse_args([])
    l1_args.infiles = [L1_SMALL_FILE]
    ex._load_l1(l1_args)
    _check(ex._aladin_target(int(ex.l1_mod.STATE.targs[0].aps_id)))


def test_ifu_aladin_target_recenters_on_the_selected_bin():
    """Regression test: _aladin_target's IFU branch used to always return
    the constant field-centre (PATCH_TABLE.X_0/Y_0, the same value on
    every row) regardless of which bin was selected, so clicking a
    different spaxel/bin never actually recentred Aladin on it. Fixed to
    use the selected bin's own XBIN/YBIN centroid — same convention
    aps_IFUviewer._build_map_figure uses for its "Selected" marker."""
    _skip_unless_exists(f"{IFU_EXGAL_OUTPATH}/{IFU_EXGAL_HEADNAME}_APS.fits")
    from PyAPS import aps_explorer as ex

    ex._route_and_load_l2(IFU_EXGAL_OUTPATH, IFU_EXGAL_HEADNAME)
    bin_ids = sorted(set(int(b) for b in ex.ifu_mod.STATE.data["table"]["BIN_ID"]))
    assert len(bin_ids) > 1, "need at least two distinct bins to test recentring"

    t1 = ex._aladin_target(bin_ids[0])
    t2 = ex._aladin_target(bin_ids[-1])
    assert t1 is not None and t2 is not None
    assert (t1["ra"], t1["dec"]) != (t2["ra"], t2["dec"]), (
        "different bins must recentre Aladin on different coordinates, "
        "not the constant field centre"
    )

    # And the field-centre-fallback path (selected_item not found / None)
    # still returns *something* sane, not a crash.
    fallback = ex._aladin_target(None)
    assert fallback is not None
    ra0 = float(ex.np.nanmean(ex.ifu_mod.STATE.data["patch_table_rec"].X_0))
    dec0 = float(ex.np.nanmean(ex.ifu_mod.STATE.data["patch_table_rec"].Y_0))
    assert fallback["ra"] == pytest.approx(ra0) and fallback["dec"] == pytest.approx(dec0)


def test_aladin_target_none_when_not_loaded():
    from PyAPS import aps_explorer as ex

    ex.EXPLORER.kind = None
    assert ex._aladin_target(1) is None


def test_log_clear_empties_buffer():
    from PyAPS import aps_explorer as ex

    ex.LOG.add("", "some log line")
    assert ex.LOG.snapshot()
    result = ex.on_log_clear(1)
    assert result == ""
    assert ex.LOG.snapshot() == []


# --------------------------------------------------------------------------- #
# Round-4 feedback: floating reopen button, always-reachable sidebar, the
# autorange="reversed" zoom-reset bug, and the Aladin catalog-overlay
# redesign.
# --------------------------------------------------------------------------- #

def test_open_load_panel_button_reopens_panel():
    """The toolbar's "☰ Load dataset" button (on_open_load_panel) opens
    the sidebar — the sidebar itself is still `position: fixed` (a real,
    deliberate full-height overlay, unchanged); only the *button* that
    triggers it moved out of a floating position into the toolbar."""
    from PyAPS import aps_explorer as ex

    style = ex.on_open_load_panel(1)
    assert style["display"] == "block"
    assert style["position"] == "fixed"


def test_fiber_map_no_longer_uses_autorange():
    """Regression test for the zoom-reset-on-click bug: autorange="reversed"
    forces Plotly to recompute the range from scratch on every rebuild,
    discarding any uirevision-preserved user zoom — must be a static range
    instead."""
    import numpy as np
    from PyAPS.apsPlot.fiber_map import fiber_map_figure

    ra = np.array([10.0, 10.01, 9.99, 10.02])
    dec = np.array([20.0, 20.01, 19.99, 20.02])
    flux = np.array([100.0, 50.0, 75.0, 20.0])
    use = np.array(["T", "T", "T", "T"])
    aps_ids = np.array([1, 2, 3, 4])
    fig = fiber_map_figure(ra, dec, flux, use, aps_ids)
    assert fig.layout.xaxis.autorange is None
    assert fig.layout.xaxis.range is not None
    assert fig.layout.xaxis.range[0] > fig.layout.xaxis.range[1]  # still RA-reversed


def test_mos_source_map_no_longer_uses_autorange():
    _skip_unless_exists(f"{MOS_OUTPATH}/{MOS_HEADNAME}_APS.fits")
    from PyAPS import aps_MOSviewer as mos_mod

    mos_mod.STATE.load(MOS_OUTPATH, MOS_HEADNAME)
    fig = mos_mod._build_source_map_figure(mos_mod.STATE.selected_aps_id)
    assert fig.layout.xaxis.autorange is None
    assert fig.layout.xaxis.range is not None
    assert fig.layout.xaxis.range[0] > fig.layout.xaxis.range[1]


def test_color_range_debounce_enabled():
    """color-vmin/color-vmax now live under the Aladin legend (see
    aps_explorer._aladin_color_range_control), not in the IFU settings
    panel — moved there so the control sits visually next to the colour
    bar it actually controls, per explicit user request."""
    _skip_unless_exists(f"{IFU_EXGAL_OUTPATH}/{IFU_EXGAL_HEADNAME}_APS.fits")
    from PyAPS import aps_explorer as ex

    ex._route_and_load_l2(IFU_EXGAL_OUTPATH, IFU_EXGAL_HEADNAME)
    panel = ex._aladin_color_range_control()

    def find_debounce_flags(node, out):
        if hasattr(node, "id") and getattr(node, "id", None) in ("color-vmin", "color-vmax"):
            out.append(getattr(node, "debounce", None))
        kids = getattr(node, "children", None)
        if kids is None:
            return
        for k in (kids if isinstance(kids, list) else [kids]):
            find_debounce_flags(k, out)

    flags = []
    find_debounce_flags(panel, flags)
    assert len(flags) == 2
    assert all(flags), "color-vmin/color-vmax must have debounce=True"


def test_aladin_catalog_buckets_all_kinds():
    _skip_unless_exists(f"{MOS_OUTPATH}/{MOS_HEADNAME}_APS.fits",
                         f"{IFU_EXGAL_OUTPATH}/{IFU_EXGAL_HEADNAME}_APS.fits",
                         L1_SMALL_FILE)
    from PyAPS import aps_explorer as ex
    import numpy as np

    def _check(buckets, expect_total):
        assert buckets
        total = 0
        for b in buckets:
            assert b["color"]
            assert b["points"]
            for p in b["points"]:
                assert np.isfinite(p["ra"]) and 0 <= p["ra"] < 360
                assert np.isfinite(p["dec"]) and -90 <= p["dec"] <= 90
                total += 1
        assert total == expect_total

    ex._route_and_load_l2(MOS_OUTPATH, MOS_HEADNAME)
    n_mos = len(ex.mos_mod.STATE.data["class_table"])
    _check(ex._aladin_catalog_points(), n_mos)

    ex._route_and_load_l2(IFU_EXGAL_OUTPATH, IFU_EXGAL_HEADNAME)
    n_ifu = len(ex.ifu_mod.STATE.data["table"]["BIN_ID"])
    _check(ex._aladin_catalog_points(), n_ifu)

    l1_args = ex.l1_mod._build_arg_parser().parse_args([])
    l1_args.infiles = [L1_SMALL_FILE]
    ex._load_l1(l1_args)
    n_l1 = len(ex.l1_mod.STATE.targs)
    buckets_l1 = ex._aladin_catalog_points()
    total_l1 = sum(len(b["points"]) for b in buckets_l1)
    assert total_l1 <= n_l1  # some fibres have no coordinates at all (skipped)
    assert total_l1 > 0


def test_aladin_catalog_ifu_bucket_count_capped():
    _skip_unless_exists(f"{IFU_EXGAL_OUTPATH}/{IFU_EXGAL_HEADNAME}_APS.fits")
    from PyAPS import aps_explorer as ex

    ex._route_and_load_l2(IFU_EXGAL_OUTPATH, IFU_EXGAL_HEADNAME)
    buckets = ex._aladin_catalog_points()
    assert len(buckets) <= ex._ALADIN_N_COLOR_BUCKETS


def test_aladin_target_and_catalog_update_when_loaded():
    """update_aladin_target/update_aladin_catalog always compute now — no
    more Off/Separate-panel mode to gate on, since the Aladin panel is the
    one and only spatial view (round 6: the Plotly map was dropped)."""
    _skip_unless_exists(f"{IFU_EXGAL_OUTPATH}/{IFU_EXGAL_HEADNAME}_APS.fits")
    from PyAPS import aps_explorer as ex
    from dash import no_update

    initial_item = ex._route_and_load_l2(IFU_EXGAL_OUTPATH, IFU_EXGAL_HEADNAME)
    target = ex.update_aladin_target(initial_item, 1)
    assert target is not no_update and target["ra"] is not None
    catalog = ex.update_aladin_catalog(1)
    assert catalog is not no_update and catalog["buckets"]

    ex.EXPLORER.kind = None
    assert ex.update_aladin_target(initial_item, 1) is no_update
    assert ex.update_aladin_catalog(1) is no_update


def test_aladin_div_style():
    """The "Background (linked zoom)" mode and the "Separate panel" toggle
    were both removed — Aladin is always visible now, one single style
    (renamed _ALADIN_DIV_STYLES -> _ALADIN_DIV_STYLE, no longer a dict of
    modes)."""
    from PyAPS import aps_explorer as ex

    assert ex._ALADIN_DIV_STYLE["display"] == "block"
    assert ex._ALADIN_DIV_STYLE["height"] == ex._MAP_HEIGHT
    assert ex._ALADIN_DIV_STYLE.get("overflow") == "hidden"
    # Regression guard: Aladin Lite draws its own UI controls as CSS
    # position:absolute children of this div — without a positioned
    # ancestor here, those controls escape to the *viewport* instead,
    # which is exactly what "Aladin spans the whole screen" looked like
    # in an earlier round. Must never regress back to "static".
    assert ex._ALADIN_DIV_STYLE["position"] == "relative"


def test_aladin_info_text_shows_id_and_coords():
    """The info box under the Aladin panel — added in round 6 to replace
    the removed Plotly map's hover tooltip — must show the right ID label
    (BIN_ID for IFU, APS_ID for L1/MOS) and finite RA/Dec."""
    _skip_unless_exists(f"{MOS_OUTPATH}/{MOS_HEADNAME}_APS.fits",
                         f"{IFU_EXGAL_OUTPATH}/{IFU_EXGAL_HEADNAME}_APS.fits")
    from PyAPS import aps_explorer as ex

    assert ex._aladin_info_text(None) == "No selection."
    ex.EXPLORER.kind = None
    assert ex._aladin_info_text(1) == "No selection."

    ex._route_and_load_l2(MOS_OUTPATH, MOS_HEADNAME)
    text = ex._aladin_info_text(ex.mos_mod.STATE.selected_aps_id)
    assert "APS_ID" in text and "RA" in text and "Dec" in text

    ex._route_and_load_l2(IFU_EXGAL_OUTPATH, IFU_EXGAL_HEADNAME)
    initial_bin = int(ex.ifu_mod.STATE.data["table"]["BIN_ID"][0])
    text = ex._aladin_info_text(initial_bin)
    assert "BIN_ID" in text and str(initial_bin) in text


def test_explorer_state_has_no_leftover_aladin_or_samp_fields():
    """Regression guard for the round-5 SAMP/Background-mode descoping and
    the round-6 "drop the Plotly map entirely" simplification — all this
    stuff must actually be gone, not just unused."""
    from PyAPS import aps_explorer as ex

    # Round 5 (SAMP / Background-mode linked-zoom).
    assert not hasattr(ex.EXPLORER, "aladin_mode")
    assert not hasattr(ex.EXPLORER, "last_relayout_sync")
    assert not hasattr(ex.l1_mod.STATE, "aladin_enabled")
    assert not hasattr(ex.l1_mod.STATE, "samp")
    assert not hasattr(ex, "on_map_relayout_sync_aladin")
    assert not hasattr(ex, "on_aladin_mode_change")
    assert not hasattr(ex, "run_samp_poll")
    assert not hasattr(ex, "run_send_to_aladin")
    assert not hasattr(ex.l1_mod, "SampBridge")
    assert not hasattr(ex.l1_mod, "on_samp_poll")
    assert not hasattr(ex.l1_mod, "send_to_aladin")

    # Round 6 (Plotly coordinate/map figure dropped entirely).
    assert not hasattr(ex.EXPLORER, "uirevision")
    assert not hasattr(ex, "_map_figure")
    assert not hasattr(ex, "_apply_current_view")
    assert not hasattr(ex, "update_map")
    assert not hasattr(ex, "on_map_click")
    assert not hasattr(ex, "on_color_range_typed_map")
    assert not hasattr(ex, "on_color_range_reset_map")
    assert not hasattr(ex, "on_color_range_typed_aladin")
    assert not hasattr(ex, "on_color_range_reset_aladin")
    assert not hasattr(ex, "_GRAPH_CONFIG")
    assert not hasattr(ex, "_ALADIN_DIV_STYLES")


# --------------------------------------------------------------------------- #
# Round-5 feedback: a tab-content error being swallowed instead of shown.
# --------------------------------------------------------------------------- #

def test_tab_content_exception_shown_not_swallowed():
    """A future exception inside a kind's own update_tab_content must
    surface as visible error text, not silently leave the previous tab's
    content on screen (indistinguishable from "nothing happened" or
    "reverted to the previous tab")."""
    _skip_unless_exists(f"{MOS_OUTPATH}/{MOS_HEADNAME}_APS.fits")
    from PyAPS import aps_explorer as ex

    ex._route_and_load_l2(MOS_OUTPATH, MOS_HEADNAME)
    real_fn = ex.mos_mod.update_tab_content
    ex.mos_mod.update_tab_content = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        result = ex.update_tab_content("redrock", ex.mos_mod.STATE.selected_aps_id, 1)
    finally:
        ex.mos_mod.update_tab_content = real_fn
    assert "boom" in str(result)


# --------------------------------------------------------------------------- #
# Round 6: dropped the Plotly coordinate/map figure entirely in favour of
# the Aladin panel as the one and only spatial view, per explicit user
# request ("why not fully drop the coordinate map and use Aladin panel as
# default instead").
# --------------------------------------------------------------------------- #

def _find_ids(node, out):
    if hasattr(node, "id") and getattr(node, "id", None) is not None:
        node_id = node.id
        # Pattern-matching (dict) component ids, e.g.
        # {"type": "path-browse-btn", "target": "lf-caldir"} — not
        # hashable as-is; store a stable string form instead so callers
        # can still assert "was a browse button for lf-caldir registered"
        # via substring checks.
        out.add(repr(sorted(node_id.items())) if isinstance(node_id, dict) else node_id)
    kids = getattr(node, "children", None)
    if kids is None:
        return
    for k in (kids if isinstance(kids, list) else [kids]):
        _find_ids(k, out)


def test_serve_layout_has_aladin_but_no_plotly_map_or_mode_toggle():
    """The rendered layout must contain aladin-div/aladin-show-dss/
    aladin-info-box but no main-map or aladin-mode — a left-behind
    Input on a removed component id would silently break whatever
    callback references it (see aps_explorer.py's own comment on this
    Dash gotcha), so this is worth guarding directly rather than
    trusting the individual component removals above. aladin-flip
    itself is deliberately asserted *absent* — dropped per explicit
    request ("nobody uses it")."""
    _skip_unless_exists(f"{IFU_EXGAL_OUTPATH}/{IFU_EXGAL_HEADNAME}_APS.fits")
    from PyAPS import aps_explorer as ex

    ex._route_and_load_l2(IFU_EXGAL_OUTPATH, IFU_EXGAL_HEADNAME)
    ids = set()
    _find_ids(ex.serve_layout(), ids)

    assert "aladin-div" in ids
    assert "aladin-show-dss" in ids
    assert "aladin-info-box" in ids
    assert "main-map" not in ids
    assert "aladin-mode" not in ids
    assert "aladin-flip" not in ids  # dropped — "nobody uses it"
    assert "open-load-panel" not in ids  # the redundant button, also removed this round


def test_aladin_show_dss_default_on():
    """DSS background must be on by default — explicit follow-up request:
    "always the DSS overlay is active[;] right now it is disable[d] by
    default and user sh[ould] tick it on to see the DSS." (Previously
    off by an earlier, since-reversed decision — the catalog overlay was
    considered the primary content, DSS imagery opt-in context.) Still
    freely toggleable off via the checkbox itself."""
    _skip_unless_exists(f"{IFU_EXGAL_OUTPATH}/{IFU_EXGAL_HEADNAME}_APS.fits")
    from PyAPS import aps_explorer as ex
    from dash import dcc

    ex._route_and_load_l2(IFU_EXGAL_OUTPATH, IFU_EXGAL_HEADNAME)

    def find(node):
        if isinstance(node, dcc.Checklist) and getattr(node, "id", None) == "aladin-show-dss":
            return node
        kids = getattr(node, "children", None)
        if kids is None:
            return None
        for k in (kids if isinstance(kids, list) else [kids]):
            found = find(k)
            if found is not None:
                return found
        return None

    checklist = find(ex.serve_layout())
    assert checklist is not None
    assert checklist.value == ["show"]


# --------------------------------------------------------------------------- #
# Round 7: ChunkLoadError fix, AoN-threshold/Marker-colour settings cleanup,
# the Aladin-panel colour legend.
# --------------------------------------------------------------------------- #

def _find_component(node, matches):
    """Depth-first search through a Dash component tree, returning the
    first node for which matches(node) is True, or None."""
    if matches(node):
        return node
    kids = getattr(node, "children", None)
    if kids is None:
        return None
    for k in (kids if isinstance(kids, list) else [kids]):
        found = _find_component(k, matches)
        if found is not None:
            return found
    return None


def test_graph_preload_dummy_present_before_any_load():
    """Regression guard for the ChunkLoadError bug: with the Plotly
    main-map dcc.Graph gone, every dcc.Graph in the app now first appears
    from inside a tab-content callback's return value rather than the
    initial layout, which a real browser fails to load the first time
    ("ChunkLoadError: Loading chunk ... failed (missing: .../async-graph.js)",
    confirmed live via Playwright — see aps_explorer.py's own comment on
    this Dash gotcha). A permanently present, hidden dcc.Graph in the
    initial layout — even before any dataset is loaded — is the fix; this
    guards it never gets removed again."""
    from PyAPS import aps_explorer as ex
    from dash import dcc

    ex.EXPLORER.kind = None
    ids = set()
    _find_ids(ex.serve_layout(), ids)
    assert "_graph_preload_dummy" in ids

    graph = _find_component(
        ex.serve_layout(),
        lambda n: isinstance(n, dcc.Graph) and getattr(n, "id", None) == "_graph_preload_dummy",
    )
    assert graph is not None


def test_marker_color_and_aon_threshold_controls_both_removed():
    """The "Marker colour" dropdown became entirely non-functional once the
    Plotly map (its only consumer, via _build_map_figure's marker_color
    parameter) was removed from the live rendering path, and was dropped
    outright rather than left in a dead/de-emphasized state (explicit user
    request). The AoN-threshold input was later dropped too (a second,
    separate explicit user request — "not necessary") — the underlying
    behaviour it controlled (_add_emission_line_markers) still runs, just
    fixed at STATE.aon_threshold's default rather than user-adjustable."""
    _skip_unless_exists(f"{IFU_EXGAL_OUTPATH}/{IFU_EXGAL_HEADNAME}_APS.fits")
    from PyAPS import aps_explorer as ex

    ex._route_and_load_l2(IFU_EXGAL_OUTPATH, IFU_EXGAL_HEADNAME)
    ids = set()
    _find_ids(ex.serve_layout(), ids)
    assert "marker-color-dd" not in ids
    assert "aon-threshold" not in ids
    assert ex.ifu_mod.STATE.aon_threshold == 4.0


def test_aladin_legend_none_when_not_loaded():
    from PyAPS import aps_explorer as ex

    ex.EXPLORER.kind = None
    assert ex._aladin_legend_info() is None
    assert ex._aladin_legend_children() is None


def test_aladin_legend_matches_catalog_color_range():
    """Rigorous, not just visual, check that the legend can never disagree
    with what _aladin_catalog_points() actually plots: independently
    recompute the same 1st/99th-percentile vmin/vmax fallback from the raw
    maptype values and compare against what the legend reports, both with
    and without an explicit colour-range override."""
    _skip_unless_exists(f"{IFU_EXGAL_OUTPATH}/{IFU_EXGAL_HEADNAME}_APS.fits")
    from PyAPS import aps_explorer as ex
    import numpy as np

    ex._route_and_load_l2(IFU_EXGAL_OUTPATH, IFU_EXGAL_HEADNAME)
    try:
        info = ex._aladin_legend_info()
        assert info["kind"] == "continuous"
        values, _, _ = ex.ifu_mod._resolve_maptype(ex.ifu_mod.STATE.data, ex.ifu_mod.STATE.current_maptype)
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        expected_vmin, expected_vmax = np.percentile(finite, [1, 99])
        assert info["vmin"] == pytest.approx(expected_vmin)
        assert info["vmax"] == pytest.approx(expected_vmax)
        assert info["colorscale"] == ex.ifu_mod.select_colorscale(ex.ifu_mod.STATE.current_maptype)

        # With an explicit override, the legend must track it exactly, not
        # the percentile fallback.
        ex.ifu_mod.STATE.color_vmin = 10.0
        ex.ifu_mod.STATE.color_vmax = 200.0
        info2 = ex._aladin_legend_info()
        assert info2["vmin"] == 10.0
        assert info2["vmax"] == 200.0
    finally:
        ex.ifu_mod.STATE.color_vmin = None
        ex.ifu_mod.STATE.color_vmax = None


def test_aladin_legend_l1_matches_catalog_defaults():
    _skip_unless_exists(L1_SMALL_FILE, CALDIR, CATDIR)
    from PyAPS import aps_explorer as ex

    l1_args = ex.l1_mod._build_arg_parser().parse_args([])
    l1_args.infiles = [L1_SMALL_FILE]
    ex._load_l1(l1_args)
    info = ex._aladin_legend_info()
    assert info["kind"] == "continuous"
    assert info["vmin"] == 12.0
    assert info["colorscale"] == "Jet"
    assert info["label"] == "Total flux (all arms)"


def test_aladin_legend_categorical_for_mos():
    """MOS is categorical (Gal/ExGal/Both/None) — exact swatches straight
    from _AVAIL_STYLE, no continuous bar, matching what
    _aladin_catalog_points's MOS branch colours points with."""
    _skip_unless_exists(f"{MOS_OUTPATH}/{MOS_HEADNAME}_APS.fits")
    from PyAPS import aps_explorer as ex

    ex._route_and_load_l2(MOS_OUTPATH, MOS_HEADNAME)
    info = ex._aladin_legend_info()
    assert info["kind"] == "categorical"
    expected = [{"color": color, "label": label} for label, color in ex.mos_mod._AVAIL_STYLE.values()]
    assert info["entries"] == expected


def test_mos_color_by_switches_catalog_and_legend_to_continuous():
    """The "Colour Aladin points by" dropdown (item 8: "can we have more
    options?") must actually change what _aladin_catalog_points/
    _aladin_legend_info return — redshift/S/N are continuous, unlike the
    default categorical availability view."""
    _skip_unless_exists(f"{MOS_OUTPATH}/{MOS_HEADNAME}_APS.fits")
    from PyAPS import aps_explorer as ex

    ex._route_and_load_l2(MOS_OUTPATH, MOS_HEADNAME)
    assert ex.mos_mod.STATE.color_by == "availability"
    assert ex._aladin_legend_info()["kind"] == "categorical"
    assert ex._color_range_state() is None
    # The control itself still renders in categorical mode (it's no
    # longer None) — the "Transparency by signal" checkbox is a 3D-cube
    # setting, independent of the 2D catalog's own colour-by mode, so it
    # must stay available even while Min/Max/Scale/Palette (genuinely
    # tied to a continuous 2D quantity) are absent — see
    # _aladin_color_range_control's own docstring.
    _assert_only_transparency_checkbox(ex._aladin_color_range_control())

    for key in ("redshift", "snr"):
        version = ex.on_mos_color_by_change(key, 0)
        assert version == 1
        assert ex.mos_mod.STATE.color_by == key
        info = ex._aladin_legend_info()
        assert info["kind"] == "continuous"
        buckets = ex._aladin_catalog_points()
        assert buckets and all(b["color"] for b in buckets)
        # the shared colour-range control now applies to MOS too
        assert ex._color_range_state() is ex.mos_mod.STATE
        assert ex._aladin_color_range_control() is not None
        # re-firing with the same value is a no-op (mount-time spurious fire guard)
        from dash import no_update
        assert ex.on_mos_color_by_change(key, 1) is no_update

    # switching back to availability hides the Min/Max/Scale/Palette row
    # again (but not the whole control — see above)
    ex.on_mos_color_by_change("availability", 1)
    assert ex._color_range_state() is None
    _assert_only_transparency_checkbox(ex._aladin_color_range_control())


def _assert_only_transparency_checkbox(control):
    """Walks `control`'s children (via the shared `_find_ids` helper)
    confirming none of the Min/Max/Scale/Palette component ids are
    present, but `cube-transparency-toggle` is — the exact contract
    `_aladin_color_range_control` promises for MOS-categorical mode (see
    that function's own docstring)."""
    assert control is not None
    ids = set()
    _find_ids(control, ids)
    assert "cube-transparency-toggle" in ids
    for absent_id in ("color-vmin", "color-vmax", "color-range-reset", "color-scale-dd", "color-palette-dd"):
        assert absent_id not in ids


def test_mos_color_by_values_scalarize_per_rank_columns():
    """Regression test for a real bug caught against live data: TARGSRVY/
    TARGCLASS/Z/SNR are stored one array-per-target (per Redrock rank), not
    a scalar — _color_by_values must pull the best-fit (rank 0) value, not
    stringify/average the whole array."""
    _skip_unless_exists(f"{MOS_OUTPATH}/{MOS_HEADNAME}_APS.fits")
    import numpy as np
    from PyAPS import aps_MOSviewer as mos_mod

    mos_mod.STATE.load(MOS_OUTPATH, MOS_HEADNAME)
    ct = mos_mod.STATE.data["class_table"]
    values = mos_mod._color_by_values(mos_mod.STATE.data, "Z")
    assert len(values) == len(ct["APS_ID"])
    finite = values[np.isfinite(values)]
    assert finite.size > 0
    # every finite value must match that target's own rank-0 Z exactly
    for i, v in enumerate(values):
        if np.isfinite(v):
            rank0 = np.asarray(ct["Z"][i])
            expected = float(rank0[0]) if rank0.ndim >= 1 else float(rank0)
            assert v == pytest.approx(expected)


def test_aladin_extra_fields_mos_scalarizes_targsrvy():
    """Regression test for the exact bug caught this round against live
    data: str(ct["TARGSRVY"][i]) on a per-rank array printed
    "[b'WS2023B2-004' b'WS2023B2-004' b'WS2023B2-004']" instead of a clean
    decoded string — _aladin_extra_fields must scalarize+decode instead."""
    _skip_unless_exists(f"{MOS_OUTPATH}/{MOS_HEADNAME}_APS.fits")
    from PyAPS import aps_explorer as ex

    ex._route_and_load_l2(MOS_OUTPATH, MOS_HEADNAME)
    aps_id = int(ex.mos_mod.STATE.data["class_table"]["APS_ID"][0])
    fields = ex._aladin_extra_fields(aps_id)
    labels = dict(fields)
    for label in ("Survey", "Class"):
        if label in labels:
            val = labels[label]
            assert not val.startswith("["), f"{label} was not scalarized: {val!r}"
            assert "b'" not in val, f"{label} still has an undecoded byte-string repr: {val!r}"


def test_l1_color_by_switches_catalog_and_legend():
    """Same "more colouring options" request, L1 side: flux (default,
    unchanged behaviour) vs S/N."""
    _skip_unless_exists(L1_SMALL_FILE, CALDIR, CATDIR)
    from PyAPS import aps_explorer as ex

    l1_args = ex.l1_mod._build_arg_parser().parse_args([])
    l1_args.infiles = [L1_SMALL_FILE]
    ex._load_l1(l1_args)
    assert ex.l1_mod.STATE.color_by == "flux"
    info = ex._aladin_legend_info()
    assert info["vmin"] == 12.0 and info["colorscale"] == "Jet"

    version = ex.on_l1_color_by_change("snr", 0)
    assert version == 1
    assert ex.l1_mod.STATE.color_by == "snr"
    info = ex._aladin_legend_info()
    assert info["kind"] == "continuous"
    assert info["colorscale"] == "Plasma"
    buckets = ex._aladin_catalog_points()
    assert buckets and all(b["color"] for b in buckets)

    from dash import no_update
    assert ex.on_l1_color_by_change("snr", 1) is no_update


# --------------------------------------------------------------------------- #
# Slit Explorer + "true fibre/spaxel size" — L1 fibre-level-only features.
# --------------------------------------------------------------------------- #

def test_is_fibre_level_true_for_mos_and_moslifu_false_for_stacked_cube():
    """The gate both new features share: NSPEC/fibre-diameter are only
    physically meaningful for single-exposure, fibre-level L1 data
    (MOS/MOSLIFU/MOSMIFU), never for a stacked/co-added cube (plain LIFU/
    MIFU — each spatial position there can be built from several
    different fibres across dithered exposures)."""
    _skip_unless_exists(L1_SMALL_FILE, L1_MOSLIFU_FILE)
    from PyAPS import aps_l1_preview as l1_mod

    l1_args = l1_mod._build_arg_parser().parse_args([])
    l1_args.infiles = [L1_SMALL_FILE]
    l1_mod.STATE.load(l1_args)
    assert l1_mod.STATE.targs[0].meta[0].get("mode") == "MOS"
    assert l1_mod.STATE.is_fibre_level() is True

    l1_args2 = l1_mod._build_arg_parser().parse_args([])
    l1_args2.infiles = [L1_MOSLIFU_FILE]
    l1_mod.STATE.load(l1_args2)
    assert l1_mod.STATE.targs[0].meta[0].get("mode") == "MOSLIFU"
    assert l1_mod.STATE.is_fibre_level() is True


def test_is_fibre_level_false_for_real_stacked_lifu_cube():
    _skip_unless_exists(*L1_STACKED_LIFU_FILES, CALDIR, CATDIR)
    from PyAPS import aps_l1_preview as l1_mod

    l1_args = l1_mod._build_arg_parser().parse_args([])
    l1_args.infiles = L1_STACKED_LIFU_FILES
    l1_args.caldir = CALDIR
    l1_args.catdir = CATDIR
    l1_mod.STATE.load(l1_args)
    assert l1_mod.STATE.targs[0].meta[0].get("mode") == "LIFU"
    assert l1_mod.STATE.is_fibre_level() is False


def test_nspec_differs_from_aps_id_and_is_used_correctly():
    """The whole premise of the feature, checked directly against real
    data: NSPEC (slit position) and APS_ID (sky/fibre-table position) are
    independent numbering schemes for the same fibre, not the same value
    or even correlated — confirmed on a real file rather than assumed
    from the user's own description alone."""
    _skip_unless_exists(L1_SMALL_FILE)
    from PyAPS import aps_l1_preview as l1_mod
    import numpy as np

    l1_args = l1_mod._build_arg_parser().parse_args([])
    l1_args.infiles = [L1_SMALL_FILE]
    l1_mod.STATE.load(l1_args)
    targs = l1_mod.STATE.targs
    nspecs = np.array([t.meta[0].get("NSPEC") for t in targs])
    aps_ids = np.array([t.aps_id for t in targs])
    assert len(set(nspecs.tolist())) == len(targs)  # NSPEC really is unique per fibre
    assert not np.array_equal(np.argsort(nspecs), np.argsort(aps_ids))


def test_slit_explorer_figure_built_for_fibre_level_none_for_stacked():
    _skip_unless_exists(L1_SMALL_FILE, *L1_STACKED_LIFU_FILES, CALDIR, CATDIR)
    from PyAPS import aps_l1_preview as l1_mod
    import plotly.io as pio

    l1_args = l1_mod._build_arg_parser().parse_args([])
    l1_args.infiles = [L1_SMALL_FILE]
    l1_mod.STATE.load(l1_args)
    fig = l1_mod._build_slit_explorer_figure(l1_mod.STATE.targs[0].aps_id)
    assert fig is not None
    pio.to_json(fig)  # must actually serialize, not just construct
    assert len(fig.data) == 2  # main strip + selected-fibre marker

    fig_no_selection = l1_mod._build_slit_explorer_figure(None)
    assert fig_no_selection is not None
    assert len(fig_no_selection.data) == 1  # no selection marker/guide line

    l1_args2 = l1_mod._build_arg_parser().parse_args([])
    l1_args2.infiles = L1_STACKED_LIFU_FILES
    l1_args2.caldir = CALDIR
    l1_args2.catdir = CATDIR
    l1_mod.STATE.load(l1_args2)
    assert l1_mod._build_slit_explorer_figure(l1_mod.STATE.targs[0].aps_id) is None


def test_slit_explorer_container_dispatch_by_kind_and_fibre_level():
    """_slit_explorer_container: None for MOS/IFU (L2) kinds (no clutter
    for the two-thirds of users who never load L1), an explanatory message
    for L1-but-stacked, a real dcc.Graph for L1-fibre-level."""
    _skip_unless_exists(f"{MOS_OUTPATH}/{MOS_HEADNAME}_APS.fits", L1_SMALL_FILE)
    from PyAPS import aps_explorer as ex
    from dash import dcc, html

    ex._route_and_load_l2(MOS_OUTPATH, MOS_HEADNAME)
    assert ex._slit_explorer_container(ex.mos_mod.STATE.selected_aps_id) is None

    l1_args = ex.l1_mod._build_arg_parser().parse_args([])
    l1_args.infiles = [L1_SMALL_FILE]
    ex._load_l1(l1_args)
    container = ex._slit_explorer_container(ex.l1_mod.STATE.targs[0].aps_id)
    assert isinstance(container, html.Div)

    def _find_graph(node):
        if isinstance(node, dcc.Graph):
            return node
        kids = getattr(node, "children", None)
        for k in (kids if isinstance(kids, list) else [kids] if kids is not None else []):
            found = _find_graph(k)
            if found is not None:
                return found
        return None

    graph = _find_graph(container)
    assert graph is not None and graph.id == "slit-explorer-graph"


def test_slit_explorer_click_selects_fibre():
    _skip_unless_exists(L1_SMALL_FILE)
    from PyAPS import aps_explorer as ex
    from dash import no_update

    l1_args = ex.l1_mod._build_arg_parser().parse_args([])
    l1_args.infiles = [L1_SMALL_FILE]
    ex._load_l1(l1_args)
    target_aps_id = int(ex.l1_mod.STATE.targs[5].aps_id)
    click_data = {"points": [{"customdata": [target_aps_id, 42.0, "A", "T"]}]}
    assert ex.on_slit_explorer_click(click_data) == target_aps_id
    assert ex.on_slit_explorer_click(None) is no_update
    assert ex.on_slit_explorer_click({"points": []}) is no_update


def test_slit_explorer_click_recentres_aladin():
    """Direct user request: clicking a fibre in the Slit Explorer must
    recentre Aladin on it, the same way clicking it in Aladin itself does
    — both write to the same shared selected-item Store, so
    update_aladin_target (Input on selected-item) fires regardless of
    which callback changed it. Checked against two real fibres with
    genuinely different sky positions, not assumed from the wiring alone."""
    _skip_unless_exists(L1_SMALL_FILE)
    from PyAPS import aps_explorer as ex
    from dash import no_update
    import numpy as np

    l1_args = ex.l1_mod._build_arg_parser().parse_args([])
    l1_args.infiles = [L1_SMALL_FILE]
    ex._load_l1(l1_args)
    st = ex.l1_mod.STATE
    # Pick two targets with a real, non-trivial sky separation (not just
    # any two indices, which could coincidentally sit at ~the same RA/Dec
    # — or, as this test's own first draft hit, at NaN: parked/unallocated
    # fibres have no real coordinate, and plain np.argmax doesn't skip
    # NaNs the way nanargmax does — a fibre-with-NaN-coordinates sorting
    # itself to "furthest away" via a NaN comparison silently winning is
    # exactly the same class of bug already caught and fixed once this
    # session in _central_item's own test — mask to finite coordinates
    # first, same fix).
    ra, dec = st.coord_arr[:, 0], st.coord_arr[:, 1]
    finite = np.isfinite(ra) & np.isfinite(dec)
    ref_idx = int(np.flatnonzero(finite)[0])  # index 0 itself may be one of the NaN ones
    dist2 = np.where(finite, (ra - ra[ref_idx]) ** 2 + (dec - dec[ref_idx]) ** 2, -np.inf)
    far_idx = int(np.argmax(dist2))
    assert dist2[far_idx] > 1e-6  # a real, measurable separation exists in this file

    click_data = {"points": [{"customdata": [int(st.aps_id_arr[far_idx]), 1.0, "A", "T"]}]}
    new_selected = ex.on_slit_explorer_click(click_data)
    assert new_selected == int(st.aps_id_arr[far_idx])

    target_before = ex._aladin_target(int(st.aps_id_arr[ref_idx]))
    target_after = ex.update_aladin_target(new_selected, 1)
    assert target_after != no_update
    assert (target_after["ra"], target_after["dec"]) != (target_before["ra"], target_before["dec"])
    assert target_after["ra"] == pytest.approx(float(ra[far_idx]))
    assert target_after["dec"] == pytest.approx(float(dec[far_idx]))


def test_slit_explorer_window_fixed_width_and_centred_on_selection():
    """Item: "always show only max 100 points per zoom... so when I click
    nspec=120 it shows from 120-50 to 120+50" — checked directly against
    the figure's own x-axis range, for a selection near the middle of the
    slit and near its very start (where the window must slide, not shrink)."""
    _skip_unless_exists(L1_SMALL_FILE)
    from PyAPS import aps_l1_preview as l1_mod
    from PyAPS.apsPlot.slit_explorer import WINDOW_HALF_WIDTH
    import numpy as np

    l1_args = l1_mod._build_arg_parser().parse_args([])
    l1_args.infiles = [L1_SMALL_FILE]
    l1_mod.STATE.load(l1_args)
    st = l1_mod.STATE
    nspecs = np.array([t.meta[0].get("NSPEC") for t in st.targs])

    mid_idx = int(len(st.targs) / 2)
    mid_id = int(st.targs[mid_idx].aps_id)
    fig = l1_mod._build_slit_explorer_figure(mid_id)
    rng = fig.layout.xaxis.range
    assert rng[1] - rng[0] == pytest.approx(2 * WINDOW_HALF_WIDTH)
    center = (rng[0] + rng[1]) / 2.0
    assert center == pytest.approx(float(nspecs[mid_idx]), abs=1.0)

    # Axis must stay pannable (not fully locked) — "I must be able to
    # scroll to other ranges" — but not zoomable, which is enforced by the
    # graph's own config (checked separately) rather than the axis itself.
    assert fig.layout.xaxis.fixedrange is False
    assert fig.layout.dragmode == "pan"


def test_slit_explorer_graph_config_disables_zoom_allows_pan():
    _skip_unless_exists(L1_SMALL_FILE)
    from PyAPS import aps_explorer as ex
    from dash import dcc

    l1_args = ex.l1_mod._build_arg_parser().parse_args([])
    l1_args.infiles = [L1_SMALL_FILE]
    ex._load_l1(l1_args)
    container = ex._slit_explorer_container(int(ex.l1_mod.STATE.targs[0].aps_id))

    def _find_graph(node):
        if isinstance(node, dcc.Graph):
            return node
        kids = getattr(node, "children", None)
        for k in (kids if isinstance(kids, list) else [kids] if kids is not None else []):
            found = _find_graph(k)
            if found is not None:
                return found
        return None

    graph = _find_graph(container)
    assert graph.config["scrollZoom"] is False
    for btn in ("zoomIn2d", "zoomOut2d", "zoom2d"):
        assert btn in graph.config["modeBarButtonsToRemove"]


def test_slit_explorer_markers_are_uniform_squares():
    """Follow-up, reversing the previous round's item: live testing found
    size-scaled circles read as "sitting between" their same-size
    neighbours rather than in line with them (explicit user report). Now
    every fibre must render as the same-size square, colour the only thing
    that varies — checked directly rather than just trusting the code
    reads correctly."""
    _skip_unless_exists(L1_SMALL_FILE)
    from PyAPS import aps_l1_preview as l1_mod
    from PyAPS.apsPlot import slit_explorer as se_mod

    l1_args = l1_mod._build_arg_parser().parse_args([])
    l1_args.infiles = [L1_SMALL_FILE]
    l1_mod.STATE.load(l1_args)
    fig = l1_mod._build_slit_explorer_figure(int(l1_mod.STATE.targs[0].aps_id))
    marker = fig.data[0].marker
    assert marker.symbol == "square"
    # A scalar (not a per-point array) size is itself proof every point
    # shares one size — Plotly only accepts an array when it varies.
    assert marker.size == se_mod.MARKER_SIZE
    assert len(set(marker.color)) > 1  # colour still varies per fibre


def test_slit_info_text_matches_selected_fibre():
    """The always-visible slit-info readout (not just a hover tooltip) —
    checked directly against the selected fibre's own real metadata."""
    _skip_unless_exists(L1_SMALL_FILE)
    from PyAPS import aps_l1_preview as l1_mod

    l1_args = l1_mod._build_arg_parser().parse_args([])
    l1_args.infiles = [L1_SMALL_FILE]
    l1_mod.STATE.load(l1_args)
    t = l1_mod.STATE.targs[3]
    text = l1_mod.slit_info_text(int(t.aps_id))
    assert text is not None
    assert f"NSPEC (slit pos): {t.meta[0].get('NSPEC')}" in text
    assert str(t.meta[0].get("FIB_STATUS")).strip() in text
    assert l1_mod.slit_info_text(None) is None


def test_slit_explorer_container_comes_after_aladin_info_box_in_layout():
    """Explicit user request: the Slit Explorer must be its own separate
    panel, below the map/legend/colour-range/DSS-flip-true-size checkboxes/
    info box (all of which "belong to the Aladin coordinate map" as one
    cluster) — not interleaved into the middle of them.

    `aladin-color-range-container` has moved twice across rounds. First to
    *before* `aladin-div` (explicit follow-up: "in 3d view can we have
    other scaling as we have in 2d like log, sinh, power etc?"), so the
    same Colour range/Scale control worked in both 2D and 3D mode rather
    than only existing inside the 2D-only panel. Then, this round, back to
    *after* both `aladin-2d-controls` (which holds `aladin-div`/
    `aladin-legend`) and `flux-cube-container` — explicit follow-up: "for
    both 2d and 3d make sure the col[o]r range and scale and pallete are
    below the color bar and cent[]red" — so the shared control sits below
    *whichever* panel (2D Aladin or 3D cube) is actually visible, rather
    than always sitting above both. Still part of the same "Aladin
    cluster" conceptually (comes before slit-explorer-container either
    way), just relocated within it."""
    _skip_unless_exists(L1_SMALL_FILE)
    from PyAPS import aps_explorer as ex

    l1_args = ex.l1_mod._build_arg_parser().parse_args([])
    l1_args.infiles = [L1_SMALL_FILE]
    ex._load_l1(l1_args)

    order = []

    def _walk(node):
        node_id = getattr(node, "id", None)
        if node_id in ("aladin-div", "aladin-legend", "aladin-color-range-container",
                       "aladin-info-box", "slit-explorer-container"):
            order.append(node_id)
        kids = getattr(node, "children", None)
        for k in (kids if isinstance(kids, list) else [kids] if kids is not None else []):
            _walk(k)

    _walk(ex.serve_layout())
    assert order.index("slit-explorer-container") > order.index("aladin-info-box")
    assert order.index("slit-explorer-container") > order.index("aladin-color-range-container")
    assert order.index("slit-explorer-container") > order.index("aladin-legend")
    # aladin-div/aladin-legend (inside aladin-2d-controls) come first, then
    # aladin-color-range-container (now after both aladin-2d-controls and
    # flux-cube-container), then aladin-info-box.
    assert order[0] == "aladin-div"
    assert order.index("aladin-legend") == 1
    assert order.index("aladin-color-range-container") > order.index("aladin-legend")
    assert order.index("aladin-info-box") > order.index("aladin-color-range-container")


def test_true_size_radius_off_by_default_and_computed_correctly_when_on():
    """_true_size_radius_deg: None while the toggle is off (the default);
    once on, MOS uses the real obsmode's WEAVE_FIBRE_DIAMETER_ARCSEC entry,
    L1 fibre-level uses its own target mode's entry, and a stacked L1 cube
    stays None even with the toggle on (no single fibre per position)."""
    _skip_unless_exists(f"{MOS_OUTPATH}/{MOS_HEADNAME}_APS.fits", L1_SMALL_FILE, *L1_STACKED_LIFU_FILES, CALDIR, CATDIR)
    from PyAPS import aps_explorer as ex
    from PyAPS import aps_constants

    ex._route_and_load_l2(MOS_OUTPATH, MOS_HEADNAME)
    assert ex.EXPLORER.aladin_true_size is False
    assert ex._true_size_radius_deg() is None

    ex.EXPLORER.aladin_true_size = True
    obsmode = ex.EXPLORER.file_info["obsmode"]
    expected = aps_constants.WEAVE_FIBRE_DIAMETER_ARCSEC[obsmode] / 2.0 / 3600.0
    assert ex._true_size_radius_deg() == pytest.approx(expected)

    l1_args = ex.l1_mod._build_arg_parser().parse_args([])
    l1_args.infiles = [L1_SMALL_FILE]
    ex._load_l1(l1_args)
    # aladin_true_size is not reset by a load (mirrors aladin-show-dss's
    # own always-sticky-across-loads behaviour) — still True.
    assert ex.EXPLORER.aladin_true_size is True
    assert ex._true_size_radius_deg() == pytest.approx(1.3 / 2.0 / 3600.0)

    l1_args2 = ex.l1_mod._build_arg_parser().parse_args([])
    l1_args2.infiles = L1_STACKED_LIFU_FILES
    l1_args2.caldir = CALDIR
    l1_args2.catdir = CATDIR
    ex._load_l1(l1_args2)
    assert ex._true_size_radius_deg() is None  # stacked cube: still nothing to show

    ex.EXPLORER.aladin_true_size = False  # restore default for other tests


def test_aladin_catalog_payload_always_has_both_keys():
    """Every writer of aladin-catalog-data must go through
    _aladin_catalog_payload (not hand-build {"buckets": ...}) or a
    colour-range/scale/maptype tweak would silently blank out radius_deg
    on its next fire — same "Dash replaces the whole prop" trap
    _load_panel_style/_main_panel_style were fixed for earlier."""
    _skip_unless_exists(f"{IFU_EXGAL_OUTPATH}/{IFU_EXGAL_HEADNAME}_APS.fits")
    from PyAPS import aps_explorer as ex

    ex._route_and_load_l2(IFU_EXGAL_OUTPATH, IFU_EXGAL_HEADNAME)
    payload = ex._aladin_catalog_payload()
    assert set(payload.keys()) == {"buckets", "radius_deg"}
    assert payload["buckets"]


def test_aladin_true_size_toggle_callback():
    _skip_unless_exists(f"{IFU_EXGAL_OUTPATH}/{IFU_EXGAL_HEADNAME}_APS.fits")
    from PyAPS import aps_explorer as ex
    from dash import no_update

    ex._route_and_load_l2(IFU_EXGAL_OUTPATH, IFU_EXGAL_HEADNAME)
    try:
        assert ex.EXPLORER.aladin_true_size is False
        result = ex.on_aladin_true_size_toggle(["true"])
        assert result is not no_update
        assert set(result.keys()) == {"buckets", "radius_deg"}
        assert ex.EXPLORER.aladin_true_size is True
        assert result["radius_deg"] is not None  # IFU: real pixelsize available
        # no-op guard: re-firing with the same effective value changes nothing
        assert ex.on_aladin_true_size_toggle(["true"]) is no_update
        result2 = ex.on_aladin_true_size_toggle([])
        assert result2["radius_deg"] is None
        assert ex.EXPLORER.aladin_true_size is False
    finally:
        ex.EXPLORER.aladin_true_size = False


def test_aladin_legend_children_render_without_error():
    """Smoke test that _aladin_legend_children() produces real Dash
    components (not a crash) for both the continuous and categorical
    branches — the CSS-gradient string-building in particular has no other
    coverage above."""
    _skip_unless_exists(f"{IFU_EXGAL_OUTPATH}/{IFU_EXGAL_HEADNAME}_APS.fits",
                         f"{MOS_OUTPATH}/{MOS_HEADNAME}_APS.fits")
    from PyAPS import aps_explorer as ex
    from dash import html

    ex._route_and_load_l2(IFU_EXGAL_OUTPATH, IFU_EXGAL_HEADNAME)
    children = ex._aladin_legend_children()
    assert isinstance(children, list) and len(children) == 3
    assert all(isinstance(c, html.Div) for c in children)

    ex._route_and_load_l2(MOS_OUTPATH, MOS_HEADNAME)
    children = ex._aladin_legend_children()
    assert isinstance(children, list) and len(children) == 4  # one per _AVAIL_STYLE entry


def test_serve_layout_includes_aladin_legend():
    _skip_unless_exists(f"{IFU_EXGAL_OUTPATH}/{IFU_EXGAL_HEADNAME}_APS.fits")
    from PyAPS import aps_explorer as ex

    ex._route_and_load_l2(IFU_EXGAL_OUTPATH, IFU_EXGAL_HEADNAME)
    ids = set()
    _find_ids(ex.serve_layout(), ids)
    assert "aladin-legend" in ids


# --------------------------------------------------------------------------- #
# Round 9: HR two-arm L1 dataset investigation — a second ChunkLoadError
# (dash_table's own async chunk, not dcc.Graph's) that broke the L1
# "Metadata" tab specifically, plus browser-side error forwarding into the
# Log panel so a purely client-side failure like this one is never silently
# invisible again.
# --------------------------------------------------------------------------- #

L1_HR_2ARM_FILES = [
    PYAPS_DATA + "/L1/20240915/single_3077580.fit",
    PYAPS_DATA + "/L1/20240915/single_3077579.fit",
]


def test_datatable_preload_dummy_present_before_any_load():
    """Regression guard for a second, independent ChunkLoadError: L1 kind
    has no dash_table.DataTable anywhere in its initial value-tables-panel
    (only IFU/MOS do — see _value_tables_children), so a browser session
    that loads an L1 dataset first (never having loaded IFU/MOS before)
    hits the "component's lazily-loaded JS chunk never preloaded" gotcha
    (see test_graph_preload_dummy_present_before_any_load's docstring for
    the general mechanism) the moment it tries to render the "Metadata"
    tab's DataTable for the first time — confirmed live via Playwright on
    the real two-arm HR dataset this was reported against: repeated
    "ChunkLoadError: ... missing: .../dash_table/async-highlight.js" in the
    browser console, and the tab never actually switching away from
    Spectra despite being clicked. Fixed the same way as the dcc.Graph
    case: a permanently present, hidden DataTable in the initial layout."""
    from PyAPS import aps_explorer as ex
    from dash import dash_table

    ex.EXPLORER.kind = None
    ids = set()
    _find_ids(ex.serve_layout(), ids)
    assert "_datatable_preload_dummy" in ids

    table = _find_component(
        ex.serve_layout(),
        lambda n: isinstance(n, dash_table.DataTable) and getattr(n, "id", None) == "_datatable_preload_dummy",
    )
    assert table is not None


def test_js_error_sink_and_dummy_present_in_layout():
    """Regression guard for the browser-error-forwarding fix: js-error-sink
    (written by index_string's console.error/window.onerror interceptor)
    and js-error-dummy (the on_js_error callback's required Output) must
    both exist in every layout, loaded or not, since the interceptor can
    fire before any dataset is ever loaded."""
    from PyAPS import aps_explorer as ex

    ex.EXPLORER.kind = None
    ids = set()
    _find_ids(ex.serve_layout(), ids)
    assert "js-error-sink" in ids
    assert "js-error-dummy" in ids


def test_index_string_installs_console_error_interceptor():
    """The interceptor must actually be wired to console.error/
    window.onerror/unhandledrejection and write into js-error-sink — this
    is the entire mechanism by which a purely client-side failure (a
    ChunkLoadError, an Aladin JS exception, anything the Python-side Log
    panel has zero visibility into on its own) ever becomes visible at
    all; a regression here would silently restore the exact "the log looks
    perfectly clean while things are visibly broken" complaint this fixes."""
    from PyAPS import aps_explorer as ex

    src = ex.app.index_string
    assert "console.error" in src
    assert "window.addEventListener(\"error\"" in src or "window.addEventListener('error'" in src
    assert "unhandledrejection" in src
    assert "js-error-sink" in src
    assert "dash_clientside.set_props" in src


def test_on_js_error_dedups_and_logs():
    """Direct-call test of the forwarding callback itself: logs a genuinely
    new message, skips an exact repeat (the ChunkLoadError case fires the
    same message dozens of times in a row — see the regression guard
    above's docstring), and logs again once the message actually changes."""
    from PyAPS import aps_explorer as ex
    from dash import no_update

    ex.LOG.clear()
    ex._last_js_error_msg = None

    result = ex.on_js_error({"kind": "console.error", "msg": "ChunkLoadError: Loading chunk 254 failed."})
    assert result is no_update  # the callback's Output is always a no-op; LOG is the real side effect
    assert any("ChunkLoadError" in line for line in ex.LOG.snapshot())
    n_after_first = len(ex.LOG.snapshot())

    # Exact repeat - must not add a second identical line.
    ex.on_js_error({"kind": "console.error", "msg": "ChunkLoadError: Loading chunk 254 failed."})
    assert len(ex.LOG.snapshot()) == n_after_first

    # A genuinely different message must still get through.
    ex.on_js_error({"kind": "uncaught", "msg": "TypeError: something else entirely"})
    assert len(ex.LOG.snapshot()) == n_after_first + 1
    assert any("something else entirely" in line for line in ex.LOG.snapshot())

    # None/empty input is a no-op, not a crash.
    assert ex.on_js_error(None) is no_update
    assert ex.on_js_error({}) is no_update

    ex.LOG.clear()
    ex._last_js_error_msg = None


# --------------------------------------------------------------------------- #
# Round: Spectra/FWHM/L2 plots filling their tab's full width, with a proper
# spacer (not an overlap) between stacked plots. Explicit user report: "the
# right side of ... the flux plot is empty within the spectra tab ... the
# same with FWHM and the same with L2 plots" and "the header of the lower
# set of plots overlap the axis of the x in topper panel plots".
# --------------------------------------------------------------------------- #

def test_spectra_and_fwhm_figures_have_no_fixed_width():
    """`fill_container_width` must have been applied: a live Dash figure
    should carry no fixed pixel `layout.width` (so it stretches to its
    container) and `autosize=True` — checked directly on the figure
    objects, not just trusted from reading the source."""
    _skip_unless_exists(L1_SMALL_FILE, CALDIR, CATDIR)
    from PyAPS import aps_l1_preview as l1_mod

    l1_mod.load_from_form_fields(
        infiles_text=L1_SMALL_FILE, infiles_list=None, l1ref=None, l2ref=None,
        apsids=None, targsrvy=None, targclass=None, maskids=None,
        area=None, mask_areas_text=None, wlranges_text=None, arms_ratio=None,
        caldir=CALDIR, catdir=CATDIR, configdir=None,
        flags=["sens_corr", "safe_mask_gaps"],
    )
    aps_id = int(l1_mod.STATE.targs[0].aps_id)

    flux_fig, ivar_fig = l1_mod._build_spectra_figure(aps_id)
    for fig in (flux_fig, ivar_fig):
        assert fig.layout.width is None
        assert fig.layout.autosize is True

    assert l1_mod.STATE.fwhm_cache is not None  # needed for the assertions below to mean anything
    overview, detail = l1_mod._build_fwhm_figures(aps_id)
    assert overview.layout.width is None and overview.layout.autosize is True
    if detail is not None:
        assert detail.layout.width is None and detail.layout.autosize is True


def test_fwhm_figures_cached_per_target_and_invalidated_on_reload():
    """Explicit follow-up report: "when I am going from spectra to fwhm
    tab it takes a few seconds to load and then if for the same selected
    target I get back to spectra tab and then fwhm, it again reload[s]
    the whole plotting system for fwhm... that information is already
    there and I have not changed anything." `fwhm_overview_figure`
    rebuilds one real Plotly trace per "cloud" fibre from scratch on
    every call — genuinely expensive, and, for a fixed `aps_id` against
    the same loaded dataset, entirely deterministic. Same
    `id(load_args)`-keyed cache pattern as `_color_by_values`'s own
    (see that test's own docstring) — checked the same three ways: a
    second call for the same `aps_id` is a genuine cache hit (identical
    object, not a coincidentally-equal rebuild) and dramatically
    faster; a different `aps_id` gets its own slot; a fresh load()
    invalidates the cache."""
    import time
    _skip_unless_exists(L1_SMALL_FILE, CALDIR, CATDIR)
    from PyAPS import aps_l1_preview as l1_mod

    l1_mod.load_from_form_fields(
        infiles_text=L1_SMALL_FILE, infiles_list=None, l1ref=None, l2ref=None,
        apsids=None, targsrvy=None, targclass=None, maskids=None,
        area=None, mask_areas_text=None, wlranges_text=None, arms_ratio=None,
        caldir=CALDIR, catdir=CATDIR, configdir=None,
        flags=["sens_corr", "safe_mask_gaps"],
    )
    assert l1_mod.STATE.fwhm_cache is not None
    aps_id_a = int(l1_mod.STATE.targs[0].aps_id)
    aps_id_b = int(l1_mod.STATE.targs[1].aps_id)

    t0 = time.time()
    first_overview, first_detail = l1_mod._build_fwhm_figures(aps_id_a)
    first_elapsed = time.time() - t0

    t0 = time.time()
    second_overview, second_detail = l1_mod._build_fwhm_figures(aps_id_a)
    second_elapsed = time.time() - t0

    assert second_overview is first_overview  # genuine cache hit
    assert second_detail is first_detail
    assert second_elapsed < max(first_elapsed / 5, 1e-6)
    assert second_elapsed < 0.05

    # A different aps_id gets its own slot, not target A's.
    other_overview, _ = l1_mod._build_fwhm_figures(aps_id_b)
    assert other_overview is not first_overview

    # A fresh load() must invalidate the cache — same id(load_args)
    # identity check _color_by_cache's own precedent uses.
    l1_mod.load_from_form_fields(
        infiles_text=L1_SMALL_FILE, infiles_list=None, l1ref=None, l2ref=None,
        apsids=None, targsrvy=None, targclass=None, maskids=None,
        area=None, mask_areas_text=None, wlranges_text=None, arms_ratio=None,
        caldir=CALDIR, catdir=CATDIR, configdir=None,
        flags=["sens_corr", "safe_mask_gaps"],
    )
    third_overview, _ = l1_mod._build_fwhm_figures(aps_id_a)
    assert third_overview is not first_overview


def test_spectrum_overlay_figure_default_width_unchanged_for_static_export():
    """The fix above must not touch `spectrum_overlay_figure`'s own
    default — pipeline diagnostic PNG export (aps_rr.make_rrplot and
    friends) calls it directly, with no `fill_container_width` in that
    path, and relies on getting the same fixed-size image it always has."""
    from PyAPS.apsPlot.spectra import spectrum_overlay_figure
    import numpy as np

    fig = spectrum_overlay_figure(
        ["R"], {"R": np.array([1.0, 2.0, 3.0])}, {"R": np.array([1.0, 2.0, 1.0])},
        {"R": None}, figure_title="t",
    )
    assert fig.layout.width == 1100  # unchanged default


def test_stacked_graphs_sized_to_own_figure_height_with_spacer():
    """Item: "the header of the lower set of plots overlap the axis of
    the x in topper panel plots ... need proper spacer between each set".
    Each stacked dcc.Graph's own style height must match its figure's own
    computed `layout.height` (not some independent fixed guess that could
    be smaller than the real content and let it overflow into the next
    graph) and every graph but the last must carry a visible marginBottom
    spacer."""
    _skip_unless_exists(L1_SMALL_FILE)
    from PyAPS import aps_l1_preview as l1_mod

    l1_args = l1_mod._build_arg_parser().parse_args([])
    l1_args.infiles = [L1_SMALL_FILE]
    l1_mod.STATE.load(l1_args)
    aps_id = int(l1_mod.STATE.targs[0].aps_id)

    container = l1_mod.update_tab_content("spectra", aps_id)
    graphs = container.children
    assert len(graphs) == 2
    for g in graphs:
        assert g.style["height"] == f"{g.figure.layout.height}px"
    assert "marginBottom" in graphs[0].style   # spacer before the next plot
    assert "marginBottom" not in graphs[1].style  # none needed after the last


def test_mos_and_ifu_l2_figures_have_no_fixed_width():
    """Same "empty space on the right" report, generalised to L2's own
    spectral-fit tabs ("the same with L2 plots") — Redrock/RVS/FERRE/
    PPXF/EMI (MOS) and Spectrum/Stellar/Emission (IFU)."""
    from PyAPS import aps_explorer as ex
    from dash import dcc

    def _graph_in(node):
        if isinstance(node, dcc.Graph):
            return node
        kids = getattr(node, "children", None)
        for k in (kids if isinstance(kids, list) else [kids] if kids is not None else []):
            found = _graph_in(k)
            if found is not None:
                return found
        return None

    _skip_unless_exists(f"{MOS_OUTPATH}/{MOS_HEADNAME}_APS.fits")
    ex._route_and_load_l2(MOS_OUTPATH, MOS_HEADNAME)
    aps_id = ex.mos_mod.STATE.selected_aps_id
    mos_container = ex.mos_mod.update_tab_content("redrock", aps_id, 1)
    mos_graph = _graph_in(mos_container)
    assert mos_graph.figure.layout.width is None
    assert mos_graph.figure.layout.autosize is True

    _skip_unless_exists(f"{IFU_EXGAL_OUTPATH}/{IFU_EXGAL_HEADNAME}_APS.fits")
    ex._route_and_load_l2(IFU_EXGAL_OUTPATH, IFU_EXGAL_HEADNAME)
    bin_id = int(ex.ifu_mod.STATE.data["table"]["BIN_ID"][0])
    ifu_container = ex.ifu_mod.update_tab_content("spectrum", bin_id, 1)
    ifu_graph = _graph_in(ifu_container)
    assert ifu_graph.figure.layout.width is None
    assert ifu_graph.figure.layout.autosize is True


def test_mos_and_ifu_l2_graphs_get_explicit_pixel_height_style():
    """Regression guard for a real, confirmed Dash bug (found reading
    `dcc.Graph`'s own async-graph.js bundle directly, live-diagnosed from
    a real report: "loading takes too long... unusually unstable... only
    for Redrock, not RVS/FERRE/PPXF... all 6 rows fit... but then when I
    click or mouse... only see 1 rank"): `responsive='auto'` (dcc.Graph's
    own default) treats `fill_container_width`'s `width=None` alone as
    reason enough to *also* silently discard the figure's real, deliberately
    -computed `layout.height`, collapsing it to `height:100%` of
    `_scrollable_graph`/`_fit_graph`'s fixed-size wrapper regardless of how
    many panel-rows the figure actually has — invisible for short figures,
    catastrophic for Redrock's up to 12 rows. Fix (same already-proven
    pattern as `aps_l1_preview._stacked_graph`): give the `dcc.Graph`
    itself an explicit pixel `style["height"]` matching `figure.layout.
    height`, checked directly here rather than trusting dcc.Graph's own
    (in this case incorrect) claim that leaving height unset lets it
    "render at its own figure.layout.height."."""
    from PyAPS import aps_explorer as ex
    from dash import dcc

    def _graph_in(node):
        if isinstance(node, dcc.Graph):
            return node
        kids = getattr(node, "children", None)
        for k in (kids if isinstance(kids, list) else [kids] if kids is not None else []):
            found = _graph_in(k)
            if found is not None:
                return found
        return None

    _skip_unless_exists(f"{MOS_OUTPATH}/{MOS_HEADNAME}_APS.fits")
    ex._route_and_load_l2(MOS_OUTPATH, MOS_HEADNAME)
    aps_id = ex.mos_mod.STATE.selected_aps_id
    mos_container = ex.mos_mod.update_tab_content("redrock", aps_id, 1)
    mos_graph = _graph_in(mos_container)
    real_height = mos_graph.figure.layout.height
    assert real_height is not None
    assert mos_graph.style["height"] == f"{real_height}px"

    _skip_unless_exists(f"{IFU_EXGAL_OUTPATH}/{IFU_EXGAL_HEADNAME}_APS.fits")
    ex._route_and_load_l2(IFU_EXGAL_OUTPATH, IFU_EXGAL_HEADNAME)
    bin_id = int(ex.ifu_mod.STATE.data["table"]["BIN_ID"][0])
    ifu_container = ex.ifu_mod.update_tab_content("spectrum", bin_id, 1)
    ifu_graph = _graph_in(ifu_container)
    real_height = ifu_graph.figure.layout.height
    assert real_height is not None
    assert ifu_graph.style["height"] == f"{real_height}px"


def test_tab_plot_scrollboxes_use_persistent_scrollbar_class():
    """Explicit user report: a plot taller than its tab's fixed-height box
    must always show a *visible* scrollbar, not an OS-level auto-hide
    overlay one that's invisible except while actively scrolling. Checked
    two ways: every scrollable tab-content wrapper across all three
    viewers carries the shared `pyaps-scrollbox` class, and
    `index_string` actually defines persistent-scrollbar CSS for it
    (`scrollbar-width`/`scrollbar-color` plus the `::-webkit-scrollbar`
    family) rather than just leaving the class name unstyled."""
    from PyAPS import aps_l1_preview as l1_mod
    from PyAPS import aps_MOSviewer as mos_mod
    from PyAPS import aps_IFUviewer as ifu_mod
    from PyAPS import aps_explorer as ex

    l1_box = l1_mod._scrollable_tab([])
    assert "pyaps-scrollbox" in l1_box.className

    ifu_box = ifu_mod._scrollable_tab([])
    assert "pyaps-scrollbox" in ifu_box.className

    import plotly.graph_objects as go
    mos_box = mos_mod._scrollable_graph(go.Figure())
    assert "pyaps-scrollbox" in mos_box.className

    assert ".pyaps-scrollbox" in ex.app.index_string
    assert "::-webkit-scrollbar" in ex.app.index_string
    assert "scrollbar-width" in ex.app.index_string
    assert "scrollbar-color" in ex.app.index_string


# --------------------------------------------------------------------------- #
# Round: L1 "Header" tab, Slit Explorer colour-scale sync, Slit Explorer bold
# label, and per-contributing-exposure fibre positions on a stacked/
# superstacked/cube L1 load.
# --------------------------------------------------------------------------- #

def test_l1_header_tab_present_and_shows_real_primary_header():
    """Item 1: "another tab to display the primary header... preferably
    after fwhm tab" — checked directly against a real file's actual FITS
    keywords, and that the tab is positioned right after FWHM."""
    _skip_unless_exists(L1_SMALL_FILE)
    from PyAPS import aps_l1_preview as l1_mod
    from PyAPS import aps_explorer as ex

    l1_args = l1_mod._build_arg_parser().parse_args([])
    l1_args.infiles = [L1_SMALL_FILE]
    ex._load_l1(l1_args)

    tabs = ex._tabs_for_current(int(l1_mod.STATE.targs[0].aps_id))
    values = [t.value for t in tabs]
    assert values == ["spectra", "fwhm", "header"]

    headers = l1_mod.STATE.primary_headers()
    assert len(headers) >= 1
    assert "cards" in headers[0]
    keywords = {k for k, v, c in headers[0]["cards"]}
    assert "SIMPLE" in keywords or "NAXIS" in keywords  # every real FITS primary header has these

    content = l1_mod.update_tab_content("header", None)  # no fibre selection needed
    from dash import dash_table
    def _find_table(node):
        if isinstance(node, dash_table.DataTable):
            return node
        kids = getattr(node, "children", None)
        for k in (kids if isinstance(kids, list) else [kids] if kids is not None else []):
            found = _find_table(k)
            if found is not None:
                return found
        return None
    table = _find_table(content)
    assert table is not None
    assert {"Keyword", "Value", "Comment"} == {c["id"] for c in table.columns}


def test_slit_explorer_callback_listens_to_aladin_catalog_data():
    """Item 2: "when I change the scale in coordinate maps, it should
    also change the colour code scale in the Slit Explorer" —
    `_build_slit_explorer_figure` already read `STATE.color_scale` fresh
    every time, so the real gap was that nothing told the Slit Explorer's
    own callback to *rebuild* when only the Scale dropdown/Min/Max/Reset
    changed (those all write straight to `aladin-catalog-data` without
    touching `dataset-version`). Checked directly against the registered
    Dash callback graph, not just by reading the source, so a future
    accidental removal of this Input is caught."""
    from PyAPS import aps_explorer as ex

    entry = ex.app.callback_map["slit-explorer-container.children"]
    input_ids = {i["id"] for i in entry["inputs"]}
    assert "aladin-catalog-data" in input_ids
    assert "selected-item" in input_ids
    assert "dataset-version" in input_ids


def test_slit_explorer_label_bold_description_not():
    """Item 3: "Text for Slit Explorer only must be bold like Log title
    ... no need to make the description... bold." — checked directly
    against the rendered style dicts, not just visually."""
    _skip_unless_exists(L1_SMALL_FILE)
    from PyAPS import aps_explorer as ex

    l1_args = ex.l1_mod._build_arg_parser().parse_args([])
    l1_args.infiles = [L1_SMALL_FILE]
    ex._load_l1(l1_args)
    container = ex._slit_explorer_container(int(ex.l1_mod.STATE.targs[0].aps_id))
    text = str(container)
    # "Slit Explorer" itself must appear inside a fontWeight:700 span...
    assert "Slit Explorer" in text and "'fontWeight': '700'" in text
    # ...and the description sentence must appear inside a fontWeight:400 one.
    assert "'fontWeight': '400'" in text
    assert "fibre position on the spectrograph slit" in text


def test_contributing_fibre_positions_resolves_real_provenance():
    """Item 4: "I need to see the coordinates of each individual single
    file[s] contributing on the coordinate map... you can find it in the
    provenance properties." Checked directly against a real stacked LIFU
    cube with a genuine multi-exposure PROV#### chain (16 combined
    exposures, confirmed via direct FITS header inspection before writing
    any code)."""
    _skip_unless_exists(L1_STACKED_MULTI_PROV_FILE)
    import numpy as np
    from PyAPS import aps_l1_preview as l1_mod

    paths = l1_mod._resolve_single_exposure_files(L1_STACKED_MULTI_PROV_FILE)
    assert len(paths) == 16
    assert all(p.endswith(".fit") and "single_" in p for p in paths)

    ra, dec, status, targuse = l1_mod._read_fibtable_positions(paths[0])
    assert len(ra) > 0
    assert np.isfinite(ra).any() and np.isfinite(dec).any()
    assert set(np.unique(targuse)) <= {"T", "S", "C", ""}


def test_contributing_fibre_positions_state_method_and_dedup_note():
    """Same file, through the real (lazy, memoized) AppState method this
    time, plus the multi-arm dedup: "no need to load the fibre position
    for each arm of the single but not bad that you write the name of
    both single blue and red next to the [checkbox]" — the second arm
    here is faked (not re-run through the full expensive APSOB pipeline a
    second time) purely to check the arm-pairing/camera-labelling logic,
    which only needs `load_args.infiles` plus each arm's own (cheap,
    headers-only) provenance chain."""
    _skip_unless_exists(L1_STACKED_MULTI_PROV_FILE)
    import argparse
    from PyAPS import aps_l1_preview as l1_mod
    from PyAPS import aps_explorer as ex

    l1_args = l1_mod._build_arg_parser().parse_args([])
    l1_args.infiles = [L1_STACKED_MULTI_PROV_FILE]
    l1_mod.STATE.load(l1_args)
    assert not l1_mod.STATE.is_fibre_level()

    result = l1_mod.STATE.contributing_fibre_positions()
    assert result is not None
    assert len(result["files"]) == 16
    assert result["n_arms_loaded"] == 1
    assert result["files"][0]["other_files"] == []  # only one arm loaded, nothing to pair
    # Memoized: a second call without a reload must return the identical
    # cached object, not recompute (would be trivially true either way
    # for correctness, but the whole point of caching is to avoid the
    # real file I/O a second time).
    assert l1_mod.STATE.contributing_fibre_positions() is result

    fake_args = argparse.Namespace(**vars(l1_args))
    fake_args.infiles = [
        L1_STACKED_MULTI_PROV_FILE,
        PYAPS_DATA + "/L1/20251010/stackcube_3117935.fit",  # real paired arm, confirmed BLUE
    ]
    l1_mod.STATE.load_args = fake_args
    l1_mod.STATE._contrib_cache = None
    l1_mod.STATE._contrib_cache_key = None
    result2 = l1_mod.STATE.contributing_fibre_positions()
    assert result2["n_arms_loaded"] == 2
    assert len(result2["files"]) == 16  # still only reads ONE arm's fibre positions
    f0 = result2["files"][0]
    assert f0["camera"] == "RED"
    assert len(f0["other_files"]) == 1
    assert f0["other_files"][0]["camera"] == "BLUE"
    assert f0["other_files"][0]["file"].startswith("single_")

    ex.EXPLORER.kind = "l1"
    ex.EXPLORER.contrib_exposures_on = True
    payload = ex._contrib_exposures_payload()
    assert payload is not None
    assert len(payload["files"]) == 16
    assert "RED" in payload["files"][0]["label"] and "BLUE" in payload["files"][0]["label"]
    assert payload["radius_deg"] == ex._CONTRIB_MARKER_RADIUS_DEG  # real angular size again
    # Flattened to one point-list per file (each point carrying its own
    # colour), not one nested list per (file, colour-bucket) pair — see
    # _contrib_exposures_payload's own docstring for why (confirmed live
    # that A.circle's per-shape `color` option works, cutting overlay
    # object count from up to files*buckets down to just one per file).
    assert all("points" in f and "buckets" not in f for f in payload["files"])
    colors = {p["color"] for f in payload["files"] for p in f["points"]}
    assert len(colors) > 1  # actually multiple distinct colours, not one flat colour
    total_points = sum(len(f["points"]) for f in payload["files"])
    assert total_points == 16 * 600  # every fibre from every file present (all files, always)
    note = ex._contrib_exposures_note_text(payload)
    note_text = str(note)
    assert "1 of 2" in note_text

    ex.EXPLORER.contrib_exposures_on = False
    assert ex._contrib_exposures_payload() is None


def test_contrib_exposures_colour_matches_main_catalog():
    """Explicit reversal request: "I want the color code [to] come again
    from the total flux or S/N or whatever the main plot is... do not
    color code them based on the single files." Checked directly: the
    vmin/vmax/colourscale actually used for the contributing-exposures
    buckets must be byte-for-byte the same values `_l1_active_color_params`
    (the main catalog's own colour source) returns, and switching the
    main "Colour Aladin points by" option must change this overlay's own
    bucket colours too, not just the main catalog's."""
    _skip_unless_exists(L1_STACKED_MULTI_PROV_FILE)
    from PyAPS import aps_l1_preview as l1_mod
    from PyAPS import aps_explorer as ex

    l1_args = l1_mod._build_arg_parser().parse_args([])
    l1_args.infiles = [L1_STACKED_MULTI_PROV_FILE]
    l1_mod.STATE.load(l1_args)
    ex.EXPLORER.kind = "l1"
    ex.EXPLORER.contrib_exposures_on = True

    l1_mod.STATE.color_by = "flux"
    payload_flux = ex._contrib_exposures_payload()
    colors_flux = {p["color"] for f in payload_flux["files"] for p in f["points"]}

    l1_mod.STATE.color_by = "snr"
    payload_snr = ex._contrib_exposures_payload()
    colors_snr = {p["color"] for f in payload_snr["files"] for p in f["points"]}

    # Different colour-by must actually change what gets drawn here too —
    # if this overlay were still colouring by file (or some fixed
    # palette), switching the main plot's colour-by would have no effect
    # on it at all.
    assert colors_flux != colors_snr

    l1_mod.STATE.color_by = "flux"  # restore default for other tests
    ex.EXPLORER.contrib_exposures_on = False


def test_contrib_exposures_legend_checkbox_present_and_payload_always_full():
    """Explicit follow-up request: "I want to have the option to select
    and deselect each contributed single file in the plot... a[checkbox]
    next to the name of each single one... so I can manage it by
    myself." The legend must be a real `dcc.Checklist` (one option per
    contributing exposure, all checked by default).

    Unlike the previous design, `_contrib_exposures_payload()` itself no
    longer filters by selection at all — it always returns *every* known
    file's full point set, unconditionally, on every call (see its own
    docstring for why this is correct: colour-bucket boundaries never
    actually depended on file selection). Showing/hiding a file is a
    pure client-side show()/hide() toggle now — see
    test_contrib_exposures_toggle_is_clientside_not_serverside for that
    half of the contract."""
    _skip_unless_exists(L1_STACKED_MULTI_PROV_FILE)
    from dash import dcc
    from PyAPS import aps_l1_preview as l1_mod
    from PyAPS import aps_explorer as ex

    l1_args = l1_mod._build_arg_parser().parse_args([])
    l1_args.infiles = [L1_STACKED_MULTI_PROV_FILE]
    l1_mod.STATE.load(l1_args)
    ex.EXPLORER.kind = "l1"
    ex.EXPLORER.contrib_exposures_on = True
    payload = ex._contrib_exposures_payload()
    note = ex._contrib_exposures_note_text(payload)

    def _find_checklist(node):
        if isinstance(node, dcc.Checklist):
            return node
        kids = getattr(node, "children", None)
        for k in (kids if isinstance(kids, list) else [kids] if kids is not None else []):
            found = _find_checklist(k)
            if found is not None:
                return found
        return None

    checklist = _find_checklist(note)
    assert checklist is not None
    assert checklist.id == "contrib-legend-checklist"
    all_files = {f["file"] for f in payload["files"]}
    assert len(all_files) == 16
    assert {opt["value"] for opt in checklist.options} == all_files
    assert set(checklist.value) == all_files  # every file shown by default
    # No colour/shape styling in the label any more — colour is data-
    # driven now, a fixed per-file swatch would just be misleading.
    assert all(isinstance(opt["label"], str) for opt in checklist.options)

    total_before = sum(len(f["points"]) for f in payload["files"])
    assert total_before == 16 * 600

    # A second call — nothing about selection can ever reduce this any
    # more, there is no selection state left server-side to consult.
    payload2 = ex._contrib_exposures_payload()
    total_after = sum(len(f["points"]) for f in payload2["files"])
    assert total_after == total_before

    ex.EXPLORER.contrib_exposures_on = False


def test_contrib_exposures_toggle_is_clientside_not_serverside():
    """The per-file legend checkbox toggle must be pure client-side (a
    show()/hide() call on that file's own already-built overlay object)
    with no server round-trip at all — explicit user report, mid-session,
    that the previous (real, server-round-trip) design was "really slow
    and laggy to plot provinces or even remove them from the plot,"
    root-caused (live JS-timing profile, not guessed) to the client-side
    Aladin redraw itself, not the server: tearing down and rebuilding
    thousands of real `A.circle` shapes on every single toggle measured
    1-2 seconds and *grew* on each successive toggle. Checked the same
    way test_flux_cube_click_is_clientside_not_serverside checks its own
    equivalent contract: no server callback may have
    contrib-legend-checklist as an Input, and the registered clientside
    callback's own JS source must genuinely call show()/hide()."""
    from PyAPS import aps_explorer as ex

    # app.callback_map holds both server and clientside registrations
    # together (Dash doesn't distinguish them there) — so the real check
    # isn't "nothing has this as an Input" (something legitimately does:
    # the clientside toggle callback itself), it's that *only* that one
    # callback (identified by its own dummy Output, not a real server
    # Output) does.
    callbacks_with_checklist_input = [
        key for key, cb in ex.app.callback_map.items()
        if "contrib-legend-checklist" in {i["id"] for i in cb.get("inputs", [])}
    ]
    assert callbacks_with_checklist_input == ["contrib-legend-toggle-dummy.children"]

    entry = ex.app.callback_map["contrib-legend-toggle-dummy.children"]
    input_ids = {i["id"] for i in entry["inputs"]}
    assert input_ids == {"contrib-legend-checklist"}

    import inspect
    src = inspect.getsource(ex)
    assert "_contribCheckedFiles = checked_files" in src
    assert "_contribApplyVisibility" in src
    assert ".show()" in src and ".hide()" in src


def test_color_by_values_cached_across_calls_and_invalidated_on_reload():
    """Explicit user report, mid-session: "when I want to overlay or
    deoverlay the fibres from provinces... it is really slow and laggy to
    plot provinces or even remove them from the plot." Profiled directly
    (cProfile against a real 32,490-target stacked L1 cube, not guessed)
    to `_color_by_values` alone: 7.3 of `_contrib_exposures_payload`'s 8.9
    total seconds, almost entirely `np.nanmean` called once per target in
    a plain Python loop — and it was being recomputed from scratch on
    every single "Show contributing exposures" checkbox tick/untick, even
    though its result depends only on `st.targs`/`color_by` and is fixed
    for the life of one loaded dataset. Checked three ways here: a second
    call with the same `color_by` is dramatically faster and returns the
    exact same array (cache hit, not just a coincidentally-equal
    recomputation); a different `color_by` value gets its own cache slot
    (not reusing the wrong one); and a fresh load() invalidates the
    cache (a stale value from a previous dataset must never leak into a
    new one)."""
    import time
    _skip_unless_exists(L1_STACKED_MULTI_PROV_FILE)
    from PyAPS import aps_l1_preview as l1_mod

    l1_args = l1_mod._build_arg_parser().parse_args([])
    l1_args.infiles = [L1_STACKED_MULTI_PROV_FILE]
    l1_mod.STATE.load(l1_args)

    t0 = time.time()
    first = l1_mod._color_by_values(l1_mod.STATE, "flux")
    first_elapsed = time.time() - t0

    t0 = time.time()
    second = l1_mod._color_by_values(l1_mod.STATE, "flux")
    second_elapsed = time.time() - t0

    assert second is first  # genuine cache hit, not a recompute that happens to match
    assert second_elapsed < first_elapsed / 5  # dramatically faster, not just "a bit"
    assert second_elapsed < 0.5

    # A different color_by gets its own slot, not the "flux" one.
    snr = l1_mod._color_by_values(l1_mod.STATE, "snr")
    assert snr is not first

    # A fresh load() must invalidate the cache — a stale array from the
    # previous dataset must never leak into a new one (same load_args
    # identity check contributing_fibre_positions()'s own cache already
    # uses, for the same reason).
    l1_args2 = l1_mod._build_arg_parser().parse_args([])
    l1_args2.infiles = [L1_STACKED_MULTI_PROV_FILE]
    l1_mod.STATE.load(l1_args2)
    third = l1_mod._color_by_values(l1_mod.STATE, "flux")
    assert third is not first


def test_contrib_exposures_drawing_callback_uses_full_payload_not_checklist():
    """The drawing callback must depend only on `contrib-fibre-data`
    (checked directly against the registered Dash callback graph) — the
    legend checklist does *not* feed back into any server callback that
    rebuilds that Store any more (a real design change from an earlier
    round — see test_contrib_exposures_toggle_is_clientside_not_
    serverside for the checklist's own, now purely client-side, contract).
    Only the two genuine data-changing triggers (a fresh load, and the
    main on/off checkbox) may write contrib-fibre-data.data."""
    from PyAPS import aps_explorer as ex

    draw_entry = ex.app.callback_map["contrib-fibre-dummy.children"]
    draw_input_ids = {i["id"] for i in draw_entry["inputs"]}
    assert draw_input_ids == {"contrib-fibre-data"}

    # Two separate callbacks legitimately write contrib-fibre-data.data
    # (dataset-version and the main checkbox) — each allow_duplicate=True
    # registration gets its own "output@<hash>" key in Dash's
    # callback_map, so check across all of them rather than one exact key.
    writers = [cb["inputs"] for key, cb in ex.app.callback_map.items()
               if key == "contrib-fibre-data.data" or key.startswith("contrib-fibre-data.data@")]
    all_input_ids = {i["id"] for inputs in writers for i in inputs}
    assert {"dataset-version", "contrib-exposures-toggle"} <= all_input_ids
    assert "contrib-legend-checklist" not in all_input_ids


def test_contributing_fibre_positions_none_for_fibre_level_l1():
    """The whole feature must be a no-op for genuine single-exposure
    fibre-level L1 data — there is nothing to resolve when the loaded
    file already *is* the single exposure."""
    _skip_unless_exists(L1_SMALL_FILE)
    from PyAPS import aps_l1_preview as l1_mod
    from PyAPS import aps_explorer as ex

    l1_args = l1_mod._build_arg_parser().parse_args([])
    l1_args.infiles = [L1_SMALL_FILE]
    l1_mod.STATE.load(l1_args)
    assert l1_mod.STATE.is_fibre_level()
    assert l1_mod.STATE.contributing_fibre_positions() is None

    ex.EXPLORER.kind = "l1"
    ex.EXPLORER.contrib_exposures_on = True
    assert ex._contrib_exposures_payload() is None
    style = ex._contrib_exposures_row_style()
    assert style["display"] == "none"
    ex.EXPLORER.contrib_exposures_on = False


# --------------------------------------------------------------------------- #
# 3D flux cube — an alternative to the Aladin coordinate map. Explicit user
# request: "instead of the aladin coordinate map, we have a 3d data, so x
# and y are coordinates and z could be the sum flux over a range of
# wavelength... I want something that I can rotate or change FOV or angle
# and go inside the cube and out. However for selection, it must be only by
# selecting from the xy coordinate as it is now... as aladin cannot handle
# it, I want that if I set that option of 3d, then it use another plotting
# panel instead of the current aladin one."
# --------------------------------------------------------------------------- #

def test_flux_cube_figure_circle_is_a_filled_disc_mesh():
    """Explicit follow-up request: "instead of showing circles as
    fibre[s] in 3d view I want filled circles with the same filling as
    they are already in their circular borders... fill up the
    circles... it makes more sense" — superseding an earlier design
    (itself a fix for "For Fibre level data at 3d I see a ring[]e of
    circular points for each fibre... it should be a connected circle
    not a set of points forming a circle") that only ever drew a hollow
    outline. `shape="circle"` (the default) must now be a genuine
    `go.Mesh3d` (same trace *type* as `shape="square"`, just fan
    -triangulated from a ring instead of a quad-pair), with each (item,
    wavelength-bin) disc's own centre vertex and ring vertices forming a
    real filled surface, not a hollow loop or scattered points."""
    import numpy as np
    from PyAPS.apsPlot.flux_cube import flux_cube_figure, _RING_POINTS

    n_items, n_wave = 5, 100
    ra = np.linspace(10.0, 10.01, n_items)
    dec = np.linspace(40.0, 40.01, n_items)
    wave = np.linspace(4000.0, 5000.0, n_wave)
    flux_matrix = np.random.default_rng(0).normal(size=(n_items, n_wave))
    items = np.arange(100, 100 + n_items)

    fig = flux_cube_figure(ra, dec, wave, flux_matrix, items,
                            n_wave_bins=10, fibre_radius_deg=1e-5)
    assert len(fig.data) == 1
    tr = fig.data[0]
    assert tr.type == "mesh3d"

    n_discs = n_items * 10
    n_verts_per_disc = _RING_POINTS + 1  # ring points + centre, no closing/break vertex needed
    expected_points = n_discs * n_verts_per_disc
    assert len(tr.x) == expected_points
    assert len(tr.y) == expected_points
    assert len(tr.z) == expected_points
    # No NaN vertices at all in this design — every vertex is real and
    # used by a real triangle (unlike the old outline's unreachable
    # closing/break points).
    assert not np.isnan(np.asarray(tr.x, dtype=float)).any()

    # A real fan triangulation: RING_POINTS triangles per disc, each
    # referencing that disc's own centre vertex (local index
    # _RING_POINTS) and two adjacent ring vertices — no triangle indices
    # ever cross into a different disc's own private vertex block.
    n_triangles = n_discs * _RING_POINTS
    i_idx, j_idx, k_idx = np.asarray(tr.i), np.asarray(tr.j), np.asarray(tr.k)
    assert len(i_idx) == n_triangles
    disc_of = i_idx // n_verts_per_disc
    assert np.array_equal(disc_of, j_idx // n_verts_per_disc)
    assert np.array_equal(disc_of, k_idx // n_verts_per_disc)
    assert ((i_idx % n_verts_per_disc) == _RING_POINTS).all()  # i is always the centre vertex

    assert len(set(np.round(np.asarray(tr.z, dtype=float), 6))) == 10  # 10 distinct depths
    # Every vertex in a given disc shares the same customdata (its item
    # ID), including the centre vertex.
    customdata = np.asarray(tr.customdata).ravel()
    assert set(customdata) == set(items)
    # `intensity` is the real numeric flux-per-bin value, not a string —
    # see _MIN_ALPHA/_MAX_ALPHA's own docstring for why a literal
    # per-vertex RGBA string array would be a serious performance trap.
    assert np.issubdtype(np.asarray(tr.intensity).dtype, np.number)

    # Must be JSON-serializable end to end (catches the FITS big-endian
    # "numpy array is not native-endianness" class of bug directly, not
    # just trusting the dtype casts read correctly).
    import plotly.io as pio
    pio.to_json(fig)


def test_flux_cube_figure_square_is_connected_flat_plates():
    """Explicit follow-up request, after live-testing the earlier solid-
    cuboid design: "I do not like the making a cube out of x and y and
    z... instead of cubes we just have a plate along x and y (ra and
    dec) and for z, then when for example you stack data from 3000 to
    3100 then we have a plane representing that slice at the middle of
    3000-3100 which is 3050." `shape="square"` must be a genuine
    `go.Mesh3d` trace — 4 vertices and 2 triangles per (item, wavelength-
    bin) *flat quad*, all four vertices at the SAME z (that bin's own
    real centre wavelength, not a top/bottom pair) — with a real x/y
    half-width measured directly from the actual item spacing (not
    passed in — see `_auto_square_radius_deg`'s own docstring for the
    real overlap bug this fixes), checked directly (not assumed) that
    adjacent items' plates touch exactly in x (the literal "connected"
    contract from the original cuboid request, still true for flat
    plates) — and that adjacent bins' plates for the *same* item sit at
    genuinely different, non-touching z (automatic for a flat sheet with
    no z-extent at all, not an explicitly-shrunk gap the way the earlier
    cuboid design needed).

    `fibre_radius_deg` deliberately NOT passed at all — the whole point
    of this test is confirming the *automatic*, data-driven sizing (the
    real, production code path for every caller now) rather than one a
    caller happens to compute correctly by hand.
    """
    import numpy as np
    from PyAPS.apsPlot.flux_cube import flux_cube_figure

    n_items, n_wave = 2, 30
    spacing = 0.005
    ra = np.array([10.0, 10.0 + spacing])
    dec = np.array([0.0, 0.0])  # dec=0 removes the cos(dec) RA-projection confound
    wave = np.linspace(4000.0, 5000.0, n_wave)
    flux_matrix = np.random.default_rng(0).normal(loc=100, size=(n_items, n_wave))
    items = np.arange(100, 100 + n_items)

    fig = flux_cube_figure(ra, dec, wave, flux_matrix, items,
                            n_wave_bins=5, shape="square")
    assert len(fig.data) == 1
    tr = fig.data[0]
    assert tr.type == "mesh3d"

    n_plates = n_items * 5
    assert len(tr.x) == n_plates * 4
    assert len(tr.i) == n_plates * 2
    assert len(tr.j) == n_plates * 2
    assert len(tr.k) == n_plates * 2
    assert len(tr.intensity) == n_plates * 4
    assert np.issubdtype(np.asarray(tr.intensity).dtype, np.number)

    x = np.asarray(tr.x, dtype=float)
    z = np.asarray(tr.z, dtype=float)
    # item0 bin0's own 4 vertices (0-3) must all sit at exactly one z —
    # a genuinely flat plate, not an extruded box with two z-levels.
    assert z[:4].max() == pytest.approx(z[:4].min(), abs=1e-9)
    # item0 bin0's own z and item0 bin1's own z (vertices 4-7) must
    # differ — a real, distinct slice depth per bin, automatically
    # non-touching (both are flat sheets with zero thickness).
    assert z[:4].mean() != pytest.approx(z[4:8].mean())
    # item0's own (all 5 bins, vertices 0-19) max x must exactly match
    # item1's (vertices 20-39) own min x — adjacent items' plates touch
    # exactly, since the x/y half-width was measured directly from their
    # own real spacing (0.005) rather than trusted from any passed-in
    # value.
    assert x[:20].max() == pytest.approx(x[20:40].min(), abs=1e-9)
    # z spans the real wavelength range end to end (bin *centres*, so
    # strictly inside [wave[0], extrapolated last edge], not touching
    # either endpoint exactly).
    assert z.min() > wave[0]
    assert z.max() < wave[-1] + (wave[-1] - wave[-2])

    customdata = np.asarray(tr.customdata).ravel()
    assert set(customdata) == set(items)

    import plotly.io as pio
    pio.to_json(fig)


def test_flux_cube_figure_wave_bin_width_angstrom_derives_bin_count():
    """Explicit follow-up: "I guess it is better if user be able to
    change the width in wavelength range where those bins stacked (sum)
    together to create one cross section... add the width of the z axis
    bin in angstrom to the top of the 3d panel so users can set it[,]
    but give a default one." `wave_bin_width_angstrom` must derive an
    equivalent bin count from the real wavelength span (checked against
    a hand-computed expectation), must be clamped to `MAX_WAVE_BINS` for
    an unreasonably small width, and must leave `n_wave_bins` completely
    unaffected when not given (`None`, the default) — full backward
    compatibility for every other test in this file that still uses the
    raw bin-count parameter directly."""
    import numpy as np
    from PyAPS.apsPlot.flux_cube import flux_cube_figure, MAX_WAVE_BINS, DEFAULT_N_WAVE_BINS

    n_items, n_wave = 3, 200
    ra = np.linspace(10.0, 10.01, n_items)
    dec = np.linspace(40.0, 40.01, n_items)
    wave = np.linspace(4000.0, 5000.0, n_wave)  # 1000 A span
    flux_matrix = np.random.default_rng(0).normal(loc=100, size=(n_items, n_wave))
    items = np.arange(n_items)

    def _n_bins(fig):
        return len(set(np.round(np.asarray(fig.data[0].z, dtype=float), 6)))

    fig_100 = flux_cube_figure(ra, dec, wave, flux_matrix, items,
                                shape="square", wave_bin_width_angstrom=100.0)
    assert _n_bins(fig_100) == round(1000.0 / 100.0)  # 10

    fig_50 = flux_cube_figure(ra, dec, wave, flux_matrix, items,
                               shape="square", wave_bin_width_angstrom=50.0)
    assert _n_bins(fig_50) == round(1000.0 / 50.0)  # 20

    # An unreasonably small width must clamp to MAX_WAVE_BINS (further
    # capped by n_wave itself, the same downstream min() every bin-count
    # path already goes through).
    fig_tiny = flux_cube_figure(ra, dec, wave, flux_matrix, items,
                                 shape="square", wave_bin_width_angstrom=0.001)
    assert _n_bins(fig_tiny) == min(MAX_WAVE_BINS, n_wave)

    # None (the default) leaves the raw bin-count parameter untouched.
    fig_default = flux_cube_figure(ra, dec, wave, flux_matrix, items, shape="square")
    assert _n_bins(fig_default) == DEFAULT_N_WAVE_BINS

    # An explicit n_wave_bins is honoured exactly the same way when no
    # width is given (pure backward compatibility).
    fig_explicit_count = flux_cube_figure(ra, dec, wave, flux_matrix, items,
                                           shape="square", n_wave_bins=7)
    assert _n_bins(fig_explicit_count) == 7


def test_flux_cube_figure_z_axis_aspect_is_stretched_not_cubic():
    """Explicit follow-up: "probably better if we scale it along z to
    show longer along z... give it a try and tell me." The scene must
    use `aspectmode="manual"` with an explicit `aspectratio` whose z is
    genuinely larger than x/y — not the previous `aspectmode="cube"`
    (RA/Dec degrees and wavelength Angstrom share no physical unit, so a
    forced-cubic bounding box was never more "correct" than a stretched
    one, just simpler)."""
    import numpy as np
    from PyAPS.apsPlot.flux_cube import flux_cube_figure, _Z_ASPECT_STRETCH

    ra = np.array([10.0, 10.01])
    dec = np.array([40.0, 40.01])
    wave = np.linspace(4000.0, 5000.0, 50)
    flux_matrix = np.random.default_rng(0).normal(size=(2, 50))
    items = np.array([1, 2])

    fig = flux_cube_figure(ra, dec, wave, flux_matrix, items, shape="square")
    assert fig.layout.scene.aspectmode == "manual"
    ratio = fig.layout.scene.aspectratio
    assert ratio.x == pytest.approx(1.0)
    assert ratio.y == pytest.approx(1.0)
    assert ratio.z == pytest.approx(_Z_ASPECT_STRETCH)
    assert ratio.z > ratio.x  # genuinely "longer along z", not cubic


def test_auto_square_radius_deg_measures_real_spacing_and_ignores_bad_fallback():
    """Direct unit test of `_auto_square_radius_deg` — the fix for a
    real, confirmed bug: a stacked/co-added L1 cube's own square-voxel
    size used to come from the single-*fibre* aperture diameter (a
    completely different, typically much larger, physical quantity than
    the reconstructed cube's actual WCS pixel spacing), producing voxels
    several times too large and heavily overlapping — "I guess if you
    select the right spaxel size these should not have overlapped at all
    as they are blocks." Checked directly against a regular synthetic
    grid with a known spacing, confirming the *median* nearest-neighbour
    distance is used (immune to a handful of outliers), and that a
    deliberately wrong `fallback` value is only ever used when there
    really are fewer than 2 points to measure from."""
    import numpy as np
    from PyAPS.apsPlot.flux_cube import _auto_square_radius_deg

    # A regular 4x4 grid, spacing 0.02 deg — every interior point's own
    # nearest neighbour is exactly 0.02 away.
    n = 4
    spacing = 0.02
    xs, ys = np.meshgrid(np.arange(n) * spacing, np.arange(n) * spacing)
    ra = xs.ravel() + 10.0
    dec = ys.ravel()
    r = _auto_square_radius_deg(ra, dec, cos_dec=1.0, fallback=999.0)
    assert r == pytest.approx(spacing / 2.0, abs=1e-9)  # NOT the wrong fallback

    # A handful of much-closer duplicate-ish points must not collapse the
    # whole result down — median, not minimum.
    ra_outlier = np.concatenate([ra, [ra[0] + 1e-6]])
    dec_outlier = np.concatenate([dec, [dec[0]]])
    r_outlier = _auto_square_radius_deg(ra_outlier, dec_outlier, cos_dec=1.0)
    assert r_outlier == pytest.approx(spacing / 2.0, abs=1e-6)

    # Fewer than 2 points: falls back exactly to `fallback` when given.
    assert _auto_square_radius_deg(np.array([10.0]), np.array([40.0]),
                                    cos_dec=1.0, fallback=0.25) == 0.25
    # ... and to a small field-spread-based value when no fallback is given.
    r_none = _auto_square_radius_deg(np.array([10.0]), np.array([40.0]), cos_dec=1.0)
    assert r_none > 0


def test_flux_cube_figure_selected_item_highlight_and_subsampling():
    import numpy as np
    from PyAPS.apsPlot.flux_cube import flux_cube_figure, _RING_POINTS

    n_items, n_wave = 20, 50
    ra = np.linspace(10.0, 10.02, n_items)
    dec = np.linspace(40.0, 40.02, n_items)
    wave = np.linspace(4000.0, 5000.0, n_wave)
    flux_matrix = np.random.default_rng(1).normal(size=(n_items, n_wave))
    items = np.arange(n_items)

    fig = flux_cube_figure(ra, dec, wave, flux_matrix, items,
                            n_wave_bins=8, fibre_radius_deg=1e-5, selected_item=5)
    assert len(fig.data) == 2  # main cloud + highlighted selection
    assert len(fig.data[1].x) == 8 * _RING_POINTS  # highlight stays a plain marker ring, not a filled disc

    # A max_items cap smaller than n_items must genuinely subsample, not
    # silently ignore the limit, and must say so in the title.
    fig2 = flux_cube_figure(ra, dec, wave, flux_matrix, items,
                             n_wave_bins=8, fibre_radius_deg=1e-5, max_items=10)
    assert len(fig2.data[0].x) == 10 * 8 * (_RING_POINTS + 1)  # ring points + centre, per disc
    assert "10" in fig2.layout.title.text and "20" in fig2.layout.title.text


def test_flux_cube_figure_transparency_scales_with_signal():
    """Explicit request: "some level of transparency in 3D mode so if
    signal (total flux) is lower[,] more transparent[,] so the 3d map is
    not dominated by [low-signal] bins."

    Transparency is encoded in the *colourscale* now (a small number of
    alpha-blended stops that Plotly.js interpolates client-side), not in
    a literal per-point RGBA string the way the first implementation did
    — that first version profiled at 7+ seconds against real ~20,000-item
    IFU data (Plotly.py validates a string-array colour one element at a
    time; a numeric array of the same size validates in bulk — confirmed
    directly, 0.2s vs 3.4s for an identical-sized trace), so it was
    replaced with this design. Checked two ways here: the colourscale's
    own alpha channel is monotonically increasing end-to-end (_MIN_ALPHA
    at fraction 0 to _MAX_ALPHA at fraction 1), *and* the lowest- and
    highest-signal items land at the low/high ends of that same numeric
    range via `intensity`/`cmin`/`cmax` — so the browser-visible
    transparency genuinely tracks signal strength end to end, the same
    guarantee the old per-point test made, just checked through the new
    mechanism.

    `transparent=True` explicitly passed here — a later follow-up
    ("when I look at the cross section of 3d data cubes, I do not see
    what I usually see in 2d maps... probably due to transparency
    issue") flipped the *default* to `False` (full, constant opacity,
    matching a 2D map's own always-opaque points) — see
    test_flux_cube_figure_transparent_false_by_default_is_fully_opaque
    for that new default's own contract."""
    import re
    import numpy as np
    from PyAPS.apsPlot.flux_cube import flux_cube_figure, _MIN_ALPHA, _MAX_ALPHA

    n_items, n_wave = 4, 40
    ra = np.linspace(10.0, 10.01, n_items)
    dec = np.linspace(40.0, 40.01, n_items)
    wave = np.linspace(4000.0, 5000.0, n_wave)
    # Item 0: uniformly tiny flux (lowest signal). Item 3: uniformly huge
    # flux (highest signal) — deliberately not random, so the resulting
    # bin sums have a known, unambiguous ordering to check against.
    flux_matrix = np.zeros((n_items, n_wave))
    flux_matrix[0, :] = 0.001
    flux_matrix[1, :] = 1.0
    flux_matrix[2, :] = 10.0
    flux_matrix[3, :] = 1000.0
    items = np.arange(n_items)

    fig = flux_cube_figure(ra, dec, wave, flux_matrix, items,
                            n_wave_bins=5, fibre_radius_deg=1e-5, transparent=True)

    def _alpha(rgba_str):
        return float(re.match(r"rgba\([\d.]+, [\d.]+, [\d.]+, ([\d.]+)\)", rgba_str).group(1))

    stops = fig.data[0].colorscale
    alpha_lo = _alpha(stops[0][1])
    alpha_hi = _alpha(stops[-1][1])
    assert alpha_lo == pytest.approx(_MIN_ALPHA, abs=0.01)
    assert alpha_hi == pytest.approx(_MAX_ALPHA, abs=0.01)
    # Monotonically increasing across every stop in between, not just the
    # two ends.
    alphas = [_alpha(s[1]) for s in stops]
    assert all(a2 >= a1 for a1, a2 in zip(alphas, alphas[1:]))

    colors = np.asarray(fig.data[0].intensity)
    customdata = np.asarray(fig.data[0].customdata)
    lowest = colors[customdata == 0].mean()
    highest = colors[customdata == 3].mean()
    assert lowest < highest
    assert lowest == pytest.approx(fig.data[0].cmin, abs=1e-6)
    assert highest == pytest.approx(fig.data[0].cmax, rel=0.05)


def test_flux_cube_figure_transparent_false_by_default_is_fully_opaque():
    """Explicit report: "when I look at the cross section of 3d data
    cubes, I do not see what I usually see in 2d maps... probably due to
    transparency issue... Can you make it optional or have transparency?"
    A 2D map's own points are always fully opaque regardless of signal
    strength, so the default was flipped: `transparent=False` (the new
    default, not passed explicitly here) must render every colourscale
    stop at alpha=1.0, uniformly — no fade at all, matching what a 2D
    cross-section actually looks like. Checked for both shapes — both
    are `go.Mesh3d` traces now (see test_flux_cube_figure_circle_is_a_
    filled_disc_mesh's own docstring for "circle" mode's own history),
    so both bake this into the same top-level `colorscale` attribute."""
    import re
    import numpy as np
    from PyAPS.apsPlot.flux_cube import flux_cube_figure

    n_items, n_wave = 4, 40
    ra = np.linspace(10.0, 10.01, n_items)
    dec = np.linspace(40.0, 40.01, n_items)
    wave = np.linspace(4000.0, 5000.0, n_wave)
    flux_matrix = np.random.default_rng(5).uniform(1, 1000, size=(n_items, n_wave))
    items = np.arange(n_items)

    def _alpha(rgba_str):
        return float(re.match(r"rgba\([\d.]+, [\d.]+, [\d.]+, ([\d.]+)\)", rgba_str).group(1))

    fig = flux_cube_figure(ra, dec, wave, flux_matrix, items, n_wave_bins=5, fibre_radius_deg=1e-5)
    alphas = [_alpha(s[1]) for s in fig.data[0].colorscale]
    assert all(a == pytest.approx(1.0) for a in alphas)

    fig_sq = flux_cube_figure(ra, dec, wave, flux_matrix, items,
                               n_wave_bins=5, fibre_radius_deg=1e-5, shape="square")
    alphas_sq = [_alpha(s[1]) for s in fig_sq.data[0].colorscale]
    assert all(a == pytest.approx(1.0) for a in alphas_sq)


def test_flux_cube_figure_scale_stops_monotonic_both_shapes():
    """Follow-up request: "in 3d view can we have other scaling as we
    have in 2d like log, sinh, power etc?" Checked directly: every scale
    option produces a strictly monotonic set of colourscale stop
    *positions* spanning exactly [0, 1] — the mechanism that applies the
    stretch without touching the underlying flux values/cmin/cmax (see
    flux_cube_figure's own `scale` docstring for why) — for both shapes,
    which now both bake the same alpha_colorscale into the same
    top-level Mesh3d `colorscale` attribute (see test_flux_cube_figure_
    circle_is_a_filled_disc_mesh's own docstring for "circle" mode's own
    history — it used to be a `Scatter3d`'s `line.colorscale` instead)."""
    import numpy as np
    from PyAPS.apsPlot.flux_cube import flux_cube_figure
    from PyAPS.apsPlot import style

    n_items, n_wave = 4, 40
    ra = np.linspace(10.0, 10.01, n_items)
    dec = np.linspace(40.0, 40.01, n_items)
    wave = np.linspace(4000.0, 5000.0, n_wave)
    flux_matrix = np.random.default_rng(4).normal(loc=100, size=(n_items, n_wave))
    items = np.arange(n_items)

    def _assert_monotonic_stops(stops):
        positions = [p for p, _ in stops]
        assert positions[0] == pytest.approx(0.0, abs=1e-9)
        assert positions[-1] == pytest.approx(1.0, abs=1e-9)
        assert all(p2 > p1 for p1, p2 in zip(positions, positions[1:]))

    for scale in style.COLOR_SCALE_OPTIONS:
        fig = flux_cube_figure(ra, dec, wave, flux_matrix, items,
                                n_wave_bins=5, fibre_radius_deg=1e-5, scale=scale)
        _assert_monotonic_stops(fig.data[0].colorscale)

        fig_sq = flux_cube_figure(ra, dec, wave, flux_matrix, items,
                                   n_wave_bins=5, fibre_radius_deg=1e-5, scale=scale, shape="square")
        _assert_monotonic_stops(fig_sq.data[0].colorscale)

    # An explicit vmin/vmax override must reach the trace's own cmin/cmax
    # unchanged — the same "None means auto-percentile, otherwise use
    # exactly what was given" contract the 2D view's own colour range
    # control has (aps_explorer._apply_color_range) — checked for both
    # shapes, which now share the same top-level cmin/cmax attribute.
    fig2 = flux_cube_figure(ra, dec, wave, flux_matrix, items,
                             n_wave_bins=5, fibre_radius_deg=1e-5, vmin=-10.0, vmax=999.0)
    assert fig2.data[0].cmin == -10.0
    assert fig2.data[0].cmax == 999.0

    fig2_sq = flux_cube_figure(ra, dec, wave, flux_matrix, items,
                                n_wave_bins=5, fibre_radius_deg=1e-5, vmin=-10.0, vmax=999.0, shape="square")
    assert fig2_sq.data[0].cmin == -10.0
    assert fig2_sq.data[0].cmax == 999.0


def test_flux_cube_figure_colorbar_is_horizontal_matching_2d_legend():
    """Explicit request: "make sure the color bar is the same shape as
    the 2d map['s legend] which sit[s] at the bottom" — the 2D view's own
    colourbar is a thin horizontal bar directly under the map, not
    Plotly's own default vertical/right-hand colorbar. Checked for
    "circle" mode's own top-level `colorbar` attribute (both shapes are
    `go.Mesh3d` traces now, see test_flux_cube_figure_circle_is_a_
    filled_disc_mesh's own docstring for that mode's own history)."""
    import numpy as np
    from PyAPS.apsPlot.flux_cube import flux_cube_figure

    n_items, n_wave = 3, 20
    ra = np.linspace(10.0, 10.01, n_items)
    dec = np.linspace(40.0, 40.01, n_items)
    wave = np.linspace(4000.0, 5000.0, n_wave)
    flux_matrix = np.random.default_rng(6).normal(loc=100, size=(n_items, n_wave))
    items = np.arange(n_items)

    fig = flux_cube_figure(ra, dec, wave, flux_matrix, items, n_wave_bins=4, fibre_radius_deg=1e-5)
    assert fig.data[0].colorbar.orientation == "h"

    fig_sq = flux_cube_figure(ra, dec, wave, flux_matrix, items,
                               n_wave_bins=4, fibre_radius_deg=1e-5, shape="square")
    assert fig_sq.data[0].colorbar.orientation == "h"


def test_flux_cube_figure_non_finite_items_dropped_gracefully():
    """A fibre with no real sky position (NaN RA/Dec — parked/unallocated,
    a real and common case in WEAVE MOS data) must be silently excluded,
    not crash or poison the whole figure."""
    import numpy as np
    from PyAPS.apsPlot.flux_cube import flux_cube_figure

    ra = np.array([10.0, np.nan, 10.01])
    dec = np.array([40.0, 40.0, np.nan])
    wave = np.linspace(4000.0, 5000.0, 20)
    flux_matrix = np.random.default_rng(2).normal(size=(3, 20))
    items = np.array([1, 2, 3])

    fig = flux_cube_figure(ra, dec, wave, flux_matrix, items,
                            n_wave_bins=5, fibre_radius_deg=1e-5)
    customdata = np.asarray(fig.data[0].customdata).ravel()
    assert set(customdata) == {1}  # only the one item with finite ra/dec


def test_l1_flux_cube_data_matches_real_state():
    _skip_unless_exists(L1_SMALL_FILE)
    import numpy as np
    from PyAPS import aps_l1_preview as l1_mod
    from PyAPS import aps_explorer as ex

    l1_args = l1_mod._build_arg_parser().parse_args([])
    l1_args.infiles = [L1_SMALL_FILE]
    ex._load_l1(l1_args)

    result = ex._flux_cube_data()
    assert result is not None
    ra, dec, wave, flux_matrix, items, radius, shape = result
    assert len(ra) == len(l1_mod.STATE.targs)
    assert flux_matrix.shape == (len(ra), len(wave))
    assert radius is not None and radius > 0
    # This particular file is fibre-level (confirmed by other tests in
    # this file) — real fibres get "circle", never "square".
    assert shape == "circle"
    # JSON-serializable end to end against real (FITS-derived) data.
    import plotly.io as pio
    fig = ex.flux_cube_figure(ra, dec, wave, flux_matrix, items, fibre_radius_deg=radius, shape=shape)
    pio.to_json(fig)


def test_mos_flux_cube_data_matches_real_state():
    _skip_unless_exists(f"{MOS_OUTPATH}/{MOS_HEADNAME}_APS.fits")
    from PyAPS import aps_explorer as ex

    ex._route_and_load_l2(MOS_OUTPATH, MOS_HEADNAME)
    result = ex._flux_cube_data()
    assert result is not None
    ra, dec, wave, flux_matrix, items, radius, shape = result
    assert len(ra) > 0
    assert flux_matrix.shape == (len(ra), len(wave))
    assert radius is not None and radius > 0
    assert shape == "circle"  # MOS is always real fibres
    import plotly.io as pio
    fig = ex.flux_cube_figure(ra, dec, wave, flux_matrix, items, fibre_radius_deg=radius,
                               shape=shape, max_items=50)
    pio.to_json(fig)


def test_ifu_flux_cube_data_matches_real_state():
    """The IFU case specifically exercises the big-endian-items bug fix
    (BIN_ID read straight off fits.getdata()) on real data, not just the
    synthetic-array unit test above — this exact combination (a real
    20,000+-spaxel dataset) is what first surfaced it."""
    _skip_unless_exists(f"{IFU_EXGAL_OUTPATH}/{IFU_EXGAL_HEADNAME}_APS.fits")
    from PyAPS import aps_explorer as ex

    ex._route_and_load_l2(IFU_EXGAL_OUTPATH, IFU_EXGAL_HEADNAME)
    result = ex._flux_cube_data()
    assert result is not None
    ra, dec, wave, flux_matrix, items, radius, shape = result
    assert len(ra) > 0
    assert flux_matrix.shape == (len(ra), len(wave))
    assert shape == "square"  # a PATCH_TABLE row is always a WCS spaxel, never a real fibre
    import plotly.io as pio
    fig = ex.flux_cube_figure(ra, dec, wave, flux_matrix, items, fibre_radius_deg=radius,
                               shape=shape, max_items=50)
    pio.to_json(fig)


def test_l1_stacked_cube_flux_cube_shape_is_square():
    """The reverse of the fibre-level check above — a stacked/co-added
    L1 cube (no single real fibre per spatial position, see
    AppState.is_fibre_level's own docstring) must get "square" too."""
    _skip_unless_exists(*L1_STACKED_LIFU_FILES)
    from PyAPS import aps_l1_preview as l1_mod
    from PyAPS import aps_explorer as ex

    l1_args = l1_mod._build_arg_parser().parse_args([])
    l1_args.infiles = L1_STACKED_LIFU_FILES
    ex._load_l1(l1_args)
    assert not l1_mod.STATE.is_fibre_level()

    result = ex._flux_cube_data()
    assert result is not None
    shape = result[6]
    assert shape == "square"


def test_map_mode_defaults_to_2d_and_toggles_visibility():
    """Explicit request: "2d is also default." Checked both in the real
    initial layout and via the mode callback itself.

    Visibility and the 3D figure used to be two separate callbacks, both
    triggered by the same `map-mode` Input but landing in the browser as
    two independent HTTP round-trips / React commits — found live
    (Playwright against real IFU data, `gd.data.length` / `window.
    Plotly.react` introspection) that this was a genuine race: whichever
    commit landed first decided the outcome, and if the figure arrived
    while flux-cube-container was still `display:none`, react-plotly.js
    measured a 0x0 container, silently skipped rendering, and never
    retried (confirmed: server-side logging showed the figure being
    built correctly every time; the browser's own `gd.data.length` was 0
    only when the two commits landed in the "wrong" order). Merged into
    one callback (`update_map_mode`) so Dash returns both outputs in a
    single response — see its own docstring for the full account.

    A second, distinct race surfaced later (also live-diagnosed, inline
    `style` attributes read straight off the DOM): the callback's own
    automatic mount-time firing (using `map-mode`'s default "2d") could
    still land *after* a later, genuine 3D click's response, silently
    reverting the container back to 2D-mode styles. Fixed with
    `prevent_initial_call=True` — safe here specifically because the
    static layout already bakes in the correct 2D-mode defaults with no
    callback needed at all (see update_map_mode's own docstring)."""
    from PyAPS import aps_explorer as ex
    from dash import dcc, no_update

    def _find_radio(node):
        if isinstance(node, dcc.RadioItems) and node.id == "map-mode":
            return node
        kids = getattr(node, "children", None)
        for k in (kids if isinstance(kids, list) else [kids] if kids is not None else []):
            found = _find_radio(k)
            if found is not None:
                return found
        return None

    radio = _find_radio(ex.serve_layout())
    assert radio is not None
    assert radio.value == "2d"

    aladin_style, cube_style, fig = ex.update_map_mode("2d", 0, None, 0, None, None)
    assert aladin_style["display"] == "block"
    assert cube_style["display"] == "none"
    assert fig is no_update  # nothing loaded yet: no-op on the figure output
    aladin_style, cube_style, fig = ex.update_map_mode("3d", 0, None, 0, None, None)
    assert aladin_style["display"] == "none"
    assert cube_style["display"] == "block"
    assert fig is no_update  # still nothing loaded


def test_update_map_mode_has_prevent_initial_call_and_color_range_version_input():
    """Regression guard for the second race described in this test
    module's own docstring above (`update_map_mode`'s own docstring has
    the full live-diagnosed account): the callback must skip Dash's
    automatic mount-time firing (`prevent_initial_call=True`). Checked
    against `app.callback_map` for the Input list — that dict genuinely
    does track `inputs` (confirmed directly: the same technique already
    used elsewhere in this file) — but `prevent_initial_call` itself
    turned out (checked directly, not assumed) not to be one of the keys
    Dash's `callback_map` exposes at all in this Dash version, so that
    half is checked by reading the decorator's own source instead, the
    same technique this file already uses for clientside-JS content that
    `callback_map` likewise can't expose."""
    from PyAPS import aps_explorer as ex
    import inspect
    import re

    entry = ex.app.callback_map["..aladin-2d-controls.style...flux-cube-container.style...flux-cube-graph.figure.."]
    input_ids = {i["id"] for i in entry["inputs"]}
    # cube-transparency-toggle is deliberately NOT here any more — it
    # moved (this round) from a static, always-present child of
    # flux-cube-container to living inside _aladin_color_range_control()
    # (next to Scale/Palette), so it now only exists in the DOM once a
    # dataset is loaded and is wired through the same color-range-version
    # indirection Scale/Palette already use, not as a direct Input — see
    # update_map_mode's and on_cube_transparency_toggle_change's own
    # docstrings. flux-cube-bin-width, by contrast, *is* a direct Input —
    # it's a static, always-present child of flux-cube-container (the
    # position cube-transparency-toggle vacated), so it never risks the
    # same missing-component gotcha.
    # flux-cube-wave-min/-max are deliberately NOT here — see
    # update_map_mode's own docstring: range scrolling is now a purely
    # client-side operation on an always-full-native-range cube, not a
    # server rebuild trigger (explicit report: "every time I change the
    # start and end, it reloads the whole 3d view").
    assert input_ids == {"map-mode", "dataset-version", "selected-item",
                          "color-range-version", "flux-cube-bin-width"}

    state_ids = {i["id"] for i in entry["state"]}
    assert state_ids == {"flux-cube-camera-store"}

    src = inspect.getsource(ex)
    decorator_match = re.search(
        r'@app\.callback\((.*?)\)\s*\ndef update_map_mode\(', src, re.DOTALL)
    assert decorator_match is not None
    assert "prevent_initial_call=True" in decorator_match.group(1)


def test_flux_cube_callback_lazy_and_figure_has_valid_customdata():
    """The figure must only ever be (re)built while 3D mode is actually
    selected (explicit "don't load extra data unless asked" principle
    already established for the contributing-exposures overlay). Every
    point's customdata must be a real, resolvable item ID — the actual
    thing the clientside click handler below reads."""
    _skip_unless_exists(L1_SMALL_FILE)
    from dash import no_update
    from PyAPS import aps_l1_preview as l1_mod
    from PyAPS import aps_explorer as ex

    l1_args = l1_mod._build_arg_parser().parse_args([])
    l1_args.infiles = [L1_SMALL_FILE]
    ex._load_l1(l1_args)

    _, _, fig = ex.update_map_mode("2d", 1, None, 0, None, None)
    assert fig is no_update  # not in 3D mode: no-op

    _, _, fig = ex.update_map_mode("3d", 1, None, 0, None, None)
    assert fig is not no_update
    assert len(fig.data) >= 1
    real_item = int(fig.data[0].customdata[0])
    assert real_item in {int(t.aps_id) for t in l1_mod.STATE.targs}


def test_flux_cube_bin_width_parses_text_input_and_survives_a_real_dash_quirk():
    """`flux-cube-bin-width` is deliberately `type="text"` (not
    `type="number"`) and parsed manually inside `update_map_mode` — a
    real, confirmed Dash bug found by live testing (reproduced
    identically via three different real select-all-then-retype
    interaction methods against the actually-running app, not a
    Playwright artifact): a *debounced* `dcc.Input(type="number")` can
    commit `None` on blur after its field is cleared and retyped, even
    though the DOM value at that exact moment is genuinely the newly-
    typed one. Checked directly here that the callback correctly parses
    a real numeric string, tolerates `None`/empty-string/garbage
    (falling back to `flux_cube_figure`'s own default rather than
    erroring the whole callback), and that a real numeric string
    genuinely changes the resulting bin count end to end."""
    _skip_unless_exists(L1_SMALL_FILE)
    from PyAPS import aps_l1_preview as l1_mod
    from PyAPS import aps_explorer as ex

    l1_args = l1_mod._build_arg_parser().parse_args([])
    l1_args.infiles = [L1_SMALL_FILE]
    ex._load_l1(l1_args)

    def _n_bins(fig):
        import numpy as np
        return len(set(np.round(np.asarray(fig.data[0].z, dtype=float), 4)))

    _, _, fig_default = ex.update_map_mode("3d", 1, None, 0, None, None)
    _, _, fig_50 = ex.update_map_mode("3d", 1, None, 0, "50", None)
    _, _, fig_garbage = ex.update_map_mode("3d", 1, None, 0, "not a number", None)
    _, _, fig_empty = ex.update_map_mode("3d", 1, None, 0, "", None)

    assert _n_bins(fig_50) != _n_bins(fig_default)  # a real string genuinely changes the result
    # Garbage/empty input never crashes the callback — falls back to the
    # same default behaviour as None (flux_cube_figure's own DEFAULT_N_
    # WAVE_BINS path), not an exception bubbling out of update_map_mode.
    assert _n_bins(fig_garbage) == _n_bins(fig_default)
    assert _n_bins(fig_empty) == _n_bins(fig_default)


def test_update_map_mode_always_builds_the_full_native_wavelength_range():
    """`update_map_mode` deliberately does NOT take `flux-cube-wave-min`/
    `-max` as Inputs (and never passes them to `flux_cube_figure`) —
    explicit follow-up report: "every time I change the start and
    end, it reloads the whole 3d view... it should be a precomputed
    thing." Range *scrolling* is now a purely client-side operation on
    an always-full-native-range cube (see the clientside callback right
    after the ◀/▶ scroll one, and this function's own docstring for the
    full "why") — checked here that a real load genuinely produces a
    figure spanning the dataset's own full coverage regardless of
    whatever `flux-cube-wave-min`/`-max` happen to hold, since this
    callback no longer reads them at all."""
    _skip_unless_exists(L1_SMALL_FILE)
    from PyAPS import aps_l1_preview as l1_mod
    from PyAPS import aps_explorer as ex
    import numpy as np

    l1_args = l1_mod._build_arg_parser().parse_args([])
    l1_args.infiles = [L1_SMALL_FILE]
    ex._load_l1(l1_args)
    assert l1_mod.STATE.loaded()

    entry = ex.app.callback_map["..aladin-2d-controls.style...flux-cube-container.style...flux-cube-graph.figure.."]
    input_ids = {i["id"] for i in entry["inputs"]}
    assert "flux-cube-wave-min" not in input_ids
    assert "flux-cube-wave-max" not in input_ids

    _, _, fig = ex.update_map_mode("3d", 1, None, 0, "50", None)
    real_lo, real_hi = ex._flux_cube_wave_bounds()
    z = np.asarray(fig.data[0].z, dtype=float)
    # The built cube's own bin centres must span nearly the dataset's
    # real full native range (bin centres sit inset from the true edges
    # by roughly half a bin width each, so an exact match isn't
    # expected — a generous 200 Å tolerance, well under one bin width,
    # is enough to confirm this is the *full* range, not some
    # accidentally-narrow one).
    assert z.min() < real_lo + 200
    assert z.max() > real_hi - 200


def test_flux_cube_figure_wave_range_still_restricts_binning_directly():
    """`flux_cube_figure`'s own `wave_min`/`wave_max` remain a real,
    supported primitive (just no longer invoked by `update_map_mode` —
    see that test's own docstring) — checked directly against the
    function itself that samples outside the range are genuinely
    dropped *before* binning (a real selection, not a display crop),
    swapped min/max doesn't silently produce an empty cube, and a
    narrower range produces fewer distinct z-slices than the full one."""
    from PyAPS.apsPlot.flux_cube import flux_cube_figure
    import numpy as np

    n_items, n_wave = 5, 200
    rng = np.random.default_rng(0)
    ra = rng.uniform(10.0, 10.1, n_items)
    dec = rng.uniform(20.0, 20.1, n_items)
    wave = np.linspace(4000.0, 6000.0, n_wave)
    flux_matrix = rng.uniform(1.0, 10.0, (n_items, n_wave))
    items = np.arange(n_items)

    def _n_bins(fig):
        return len(set(np.round(np.asarray(fig.data[0].z, dtype=float), 4)))

    fig_full = flux_cube_figure(ra, dec, wave, flux_matrix, items, wave_bin_width_angstrom=50)
    fig_narrow = flux_cube_figure(ra, dec, wave, flux_matrix, items, wave_bin_width_angstrom=50,
                                    wave_min=4900, wave_max=5100)
    n_full, n_narrow = _n_bins(fig_full), _n_bins(fig_narrow)
    assert 0 < n_narrow < n_full

    z_narrow = np.asarray(fig_narrow.data[0].z, dtype=float)
    assert z_narrow.min() >= 4900 - 50 and z_narrow.max() <= 5100 + 50

    # Swapped min/max (user typed them backwards) must not silently
    # produce an empty cube.
    fig_swapped = flux_cube_figure(ra, dec, wave, flux_matrix, items, wave_bin_width_angstrom=50,
                                     wave_min=5100, wave_max=4900)
    assert len(fig_swapped.data) >= 1
    assert _n_bins(fig_swapped) == n_narrow


def test_dataset_load_prefills_flux_cube_wave_range_with_real_coverage():
    """`on_dataset_loaded_set_wave_range` — explicit request: "scroll
    through the wavelength range," which needs real starting numbers
    (not blank fields) to scroll from on every fresh load. Checked
    against the dataset's own real wavelength coverage (not a guessed
    round number) so this can never silently drift from what the cube
    itself actually spans."""
    _skip_unless_exists(L1_SMALL_FILE)
    from PyAPS import aps_l1_preview as l1_mod
    from PyAPS import aps_explorer as ex
    from dash import no_update

    # Nothing loaded yet (the autouse _reset_explorer_state fixture resets
    # EXPLORER.kind before every test): a genuine no-op, not a blank-out.
    lo0, hi0 = ex.on_dataset_loaded_set_wave_range(0)
    assert lo0 is no_update and hi0 is no_update

    l1_args = l1_mod._build_arg_parser().parse_args([])
    l1_args.infiles = [L1_SMALL_FILE]
    ex._load_l1(l1_args)

    lo, hi = ex.on_dataset_loaded_set_wave_range(1)
    assert lo is not no_update and hi is not no_update
    real_lo, real_hi = ex._flux_cube_wave_bounds()
    assert abs(float(lo) - real_lo) < 1.0
    assert abs(float(hi) - real_hi) < 1.0
    assert float(lo) < float(hi)


def test_flux_cube_camera_preserved_across_click_but_reset_on_new_dataset():
    """Explicit report: "when I click on a map[[a point]] in 3d view, it
    reload[s] the image ... it change[s] the 3d orientation every time i
    click on a point ... I want that if i [c]lick somehwere it does not
    change the zoom scale or viewing angle." Direct unit test of
    `update_map_mode`'s own camera-readback logic (see its and
    `EXPLORER.flux_cube_camera_version`'s own docstrings) — a rotated
    camera read back via `State("flux-cube-camera-store", "data")` (a
    plain camera dict, written directly by a dedicated clientside
    `plotly_relayout` listener — not `flux-cube-graph`'s own `figure`
    prop, an earlier version of this fix that turned out racy, "very few
    times it tried to save the camera angle... but still reload[ed]") must
    be reapplied verbatim on a same-dataset rebuild (a click/colour-range
    change), but a genuinely *new* dataset load must reset to
    `flux_cube_figure`'s own default top-down view rather than silently
    inherit the previous dataset's rotated camera — the exact regression
    a naive "always read back the stored camera" implementation would
    have (checked directly here, not just live in a browser, so this can
    never silently regress)."""
    _skip_unless_exists(L1_SMALL_FILE)
    from PyAPS import aps_l1_preview as l1_mod
    from PyAPS import aps_explorer as ex

    l1_args = l1_mod._build_arg_parser().parse_args([])
    l1_args.infiles = [L1_SMALL_FILE]
    ex._load_l1(l1_args)

    default_camera = {"eye": {"x": 0.0, "y": 0.0, "z": 2.5}, "up": {"x": 0.0, "y": 1.0, "z": 0.0}}
    rotated_camera = {"eye": {"x": -0.15, "y": -0.08, "z": -2.49},
                       "up": {"x": -0.8, "y": 0.6, "z": 0.03}, "center": {"x": 0, "y": 0, "z": 0}}

    # First build for dataset-version=1 (stored_camera=None, as if the
    # store had never been written to before): must use the default camera.
    _, _, fig1 = ex.update_map_mode("3d", 1, None, 0, None, None)
    assert fig1.layout.scene.camera.eye.x == 0.0 and fig1.layout.scene.camera.eye.z == 2.5

    # A same-dataset rebuild (still version=1) that reads back a rotated
    # camera from the store must reuse it exactly.
    _, _, fig2 = ex.update_map_mode("3d", 1, None, 0, None, rotated_camera)
    assert fig2.layout.scene.camera.eye.x == pytest.approx(-0.15)
    assert fig2.layout.scene.camera.up.x == pytest.approx(-0.8)

    # A rebuild for a *different* dataset-version (a genuinely new load)
    # must NOT inherit that same rotated camera, even though the store
    # still (realistically) carries it — must reset to default.
    _, _, fig3 = ex.update_map_mode("3d", 2, None, 0, None, rotated_camera)
    assert fig3.layout.scene.camera.eye.x == 0.0 and fig3.layout.scene.camera.eye.z == 2.5

    # And now that version=2 is the "current" one, a further same-version
    # rebuild reusing *that* rebuild's own stored camera (still the
    # default, since nothing rotated it yet) stays at the default too.
    _, _, fig4 = ex.update_map_mode("3d", 2, None, 0, None, default_camera)
    assert fig4.layout.scene.camera.eye.x == 0.0 and fig4.layout.scene.camera.eye.z == 2.5


def test_flux_cube_click_is_clientside_not_serverside():
    """Explicit finding from live Playwright testing against the actual
    running app, not a design choice made up front: Plotly.js's own
    "plotly_click" event never fires at all for scatter3d traces in this
    app's installed Plotly.js version (confirmed live — "plotly_hover"
    fires correctly for the exact same points, "plotly_click" never
    does, a known/documented Plotly.js gl3d limitation, not a bug in
    this app's own wiring) — so there must be **no** server-side
    `Input("flux-cube-graph", "clickData")` callback at all (it would
    simply never fire), and the real click-handling must instead be a
    clientside callback reacting to the figure itself, tracking
    plotly_hover's own customdata and reading it back on a native DOM
    "click" listener (confirmed live to work reliably)."""
    from PyAPS import aps_explorer as ex

    assert "flux-cube-graph.clickData" not in {
        i["id"] + "." + i["property"]
        for cb in ex.app.callback_map.values()
        for i in cb.get("inputs", [])
    }
    entry = ex.app.callback_map["flux-cube-click-dummy.children"]
    input_ids = {i["id"] for i in entry["inputs"]}
    assert input_ids == {"flux-cube-graph"}
    # Dash doesn't expose registered clientside callback source through
    # any public API — read the module's own source directly (same
    # technique already used elsewhere in this file for node
    # --check-style JS validation) to confirm the hover-tracking +
    # native-click workaround is really what's wired up, not just that
    # *some* callback exists on the right Input.
    import inspect
    src = inspect.getsource(ex)
    assert 'gd.on("plotly_hover"' in src
    assert 'addEventListener("click"' in src


# --------------------------------------------------------------------------- #
# CSV export — explicit request: "whatever it is a L2 output table or
# header or metadata, whenever it is a table, we should be able to
# export it... a button for each table to save as csv". DataTable's own
# native export_format prop was tried first and reverted — confirmed via
# live Playwright reproduction against the real running app that it
# fails with a genuine ChunkLoadError on dash_table's own async-
# export.js chunk (100% reproducible, not an intermittent race) — see
# apsPlot/datatable_export.py's module docstring for the full
# investigation. Replaced with a custom dcc.Download-based mechanism:
# apsPlot.datatable_export.csv_export_row() wraps each table with its
# own button + dcc.Download; aps_explorer.py's register_csv_export()
# wires the actual data-to-CSV callback. These tests confirm every real
# table across all three kinds is wrapped and wired correctly, and that
# the callback produces correct CSV content from real data.
# --------------------------------------------------------------------------- #

def _find_by_id_type(node, id_type, found=None):
    """Depth-first walk of a Dash component tree collecting every
    component whose `id` is a pattern-matching dict with `"type": id_type`
    (the shape apsPlot.datatable_export.csv_export_row uses for its
    button/Download pair) — independent of which concrete `dash`/
    `dash_table` classes each module imports its components as."""
    found = found if found is not None else []
    node_id = getattr(node, "id", None)
    if isinstance(node_id, dict) and node_id.get("type") == id_type:
        found.append(node)
    children = getattr(node, "children", None)
    if children is None:
        return found
    if not isinstance(children, list):
        children = [children]
    for c in children:
        _find_by_id_type(c, id_type, found)
    return found


def _export_button_table_ids(node):
    return {b.id["table"] for b in _find_by_id_type(node, "csv-export-btn")}


def test_every_real_table_has_a_registered_export_callback():
    """Every table_id ever passed to csv_export_row() must also be
    passed to register_csv_export() (aps_explorer.py's own module-level
    loop does this) — otherwise that table's button is wired to nothing
    and silently does nothing when clicked, exactly the original bug
    report. Checked via app.callback_map's own key format (a JSON-
    serialized, alphabetically-sorted dict) rather than guessing at
    string containment."""
    from PyAPS import aps_explorer as ex

    expected = {
        "l1-info-table",
        *(f"l1-header-table-{i}" for i in range(ex.l1_mod.HEADER_TABLE_MAX_FILES)),
        "spaxel-values-table", "bin-values-table",
        "class-values-table", "star-values-table", "galaxy-values-table",
        "ifu-processing-history-table",
    }
    registered = {
        k.split(".data")[0]
        for k in ex.app.callback_map
        if '"type":"csv-download"' in k
    }
    import json
    registered_table_ids = {json.loads(k)["table"] for k in registered}
    assert registered_table_ids == expected


def test_l1_tables_have_csv_export(tmp_path):
    _skip_unless_exists(L1_SMALL_FILE, CALDIR, CATDIR)
    from PyAPS import aps_explorer as ex

    l1_args = ex.l1_mod._build_arg_parser().parse_args([])
    l1_args.infiles = [L1_SMALL_FILE]
    l1_args.caldir, l1_args.catdir = CALDIR, CATDIR
    ex._load_l1(l1_args)

    assert _export_button_table_ids(ex._file_info_panel()) == {"l1-info-table"}

    aps_id = ex.l1_mod.STATE.targs[0].aps_id
    assert _export_button_table_ids(ex.l1_mod.update_tab_content("header", aps_id)) == {"l1-header-table-0"}


def test_ifu_value_tables_have_csv_export():
    _skip_unless_exists(f"{IFU_EXGAL_OUTPATH}/{IFU_EXGAL_HEADNAME}_APS.fits")
    from PyAPS import aps_explorer as ex

    ex._route_and_load_l2(IFU_EXGAL_OUTPATH, IFU_EXGAL_HEADNAME)
    assert _export_button_table_ids(ex.ifu_mod._value_tables_panel()) == \
        {"spaxel-values-table", "bin-values-table"}


def test_mos_value_tables_have_csv_export():
    _skip_unless_exists(f"{MOS_OUTPATH}/{MOS_HEADNAME}_APS.fits")
    from PyAPS import aps_explorer as ex

    ex._route_and_load_l2(MOS_OUTPATH, MOS_HEADNAME)
    assert _export_button_table_ids(ex.mos_mod._value_tables_panel()) == \
        {"class-values-table", "star-values-table", "galaxy-values-table"}


def test_hidden_preload_dummy_table_has_no_export_button():
    """The one DataTable that must NOT get an export button: a hidden,
    empty, display:none placeholder that exists only to prevent a real-
    browser-only ChunkLoadError (see the History section of
    doc/TESTING.md), not a real user-facing table."""
    import inspect
    from PyAPS import aps_explorer as ex

    src = inspect.getsource(ex)
    idx = src.index('id="_datatable_preload_dummy"')
    snippet = src[idx - 40: idx + 200]
    assert "csv_export_row" not in snippet
    assert "export" not in snippet.lower()


def test_csv_export_callback_produces_correct_content():
    """Exercises _table_rows_to_csv_download() — the plain (non-Dash-
    callback-wrapped) function register_csv_export()'s actual callback
    delegates to — directly. Bypasses the browser/dcc.Download plumbing
    (standard, always-loaded dash-core-components machinery, not this
    app's own logic); the part worth testing here is which rows/columns
    get exported and in what order/naming."""
    from PyAPS import aps_explorer as ex

    export_fn = ex._table_rows_to_csv_download

    columns = [{"name": "Key", "id": "Key"}, {"name": "Value", "id": "Value"}]
    data = [{"Key": "a", "Value": "1"}, {"Key": "b", "Value": "2"}]
    result = export_fn("l1-info-table", None, data, columns)
    assert result["filename"] == "l1-info-table.csv"
    assert result["content"].splitlines() == ["Key,Value", "a,1", "b,2"]

    # derived_virtual_data (reflecting an active sort/filter) takes
    # priority over the raw data prop when present — matches what
    # DataTable's own native export_columns="visible" used to do.
    filtered = [{"Key": "b", "Value": "2"}]
    result = export_fn("l1-info-table", filtered, data, columns)
    assert result["content"].splitlines() == ["Key,Value", "b,2"]

    # No rows at all (e.g. a MOS target with no galaxy data for this
    # table) must not produce an empty/broken download.
    assert export_fn("l1-info-table", None, [], columns) is ex.no_update


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
