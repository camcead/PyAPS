"""
tests/test_spaxel_weighted_lsf.py - Tests for the opt-in spaxel-weighted
LSF/FWHM feature for IFU cubes (`aps_ifu_spaxel_contrib.py`,
`aps_lsf.LSFInterpolator`/`aps_fwhm.FWHMInterpolator.
build_spaxel_weighted_entries`, `aps_utils.APSOB`'s `spaxel_weighted_lsf`
parameter).

Background: for MOS/MOSLIFU/MOSMIFU data, `_assign_arm_results_to_targets`
looks each target up by its real physical fibre NSPEC. For a true IFU
cube (LIFU/MIFU/IFU mode), every spaxel used to get the flat *global*
FWHM/LSF unconditionally, since a cube spaxel is a regrid of several
different physical fibres from different dithered single exposures and
there was no code working out which ones. This feature closes that gap,
off by default.

Uses real reference stackcubes already present on this machine
(set PYAPS_TEST_DATA); skipped (not failed) if not found.
"""

from __future__ import annotations
import os as _os
# Root of a directory tree holding real WEAVE data (L1/, L2/, CAL/, CAT/ ...). Tests that need
# real data are skipped when it is not available; point PYAPS_TEST_DATA at your copy to run them.
PYAPS_DATA = _os.environ.get("PYAPS_TEST_DATA", "<PYAPS_DATA>")
PYAPS_HOME = _os.environ.get("PYAPS_HOME", _os.path.expanduser("~/PyAPS"))

from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from PyAPS.aps_ifu_spaxel_contrib import (
    _circle_overlap_area,
    contributing_fibre_weights,
    build_spaxel_weighted_entries,
)

# Real LIFU stackcubes (blue + red arms of the same OB), one dataset of
# several the user pointed at directly for this feature's own
# verification. CAL/CAT live under the same `weave` tree.
REAL_INFILES = [
    PYAPS_DATA + "/L1/20240105/stackcube_3039542.fit",  # BLUE
    PYAPS_DATA + "/L1/20240105/stackcube_3039541.fit",  # RED
]
REAL_CALDIR = PYAPS_DATA + "/CAL"
REAL_CATDIR = PYAPS_DATA + "/CAT"


def _skip_unless_exists(*paths):
    for p in paths:
        if not Path(p).exists():
            pytest.skip(f"reference data not found: {p}")


# --------------------------------------------------------------------------- #
# Geometry: circle-circle overlap area
# --------------------------------------------------------------------------- #

def test_circle_overlap_area_known_cases():
    """Concentric (full overlap), tangent (zero), disjoint (zero), and one
    circle fully inside another (small circle's own area) — closed-form
    cases with an exactly-known answer, not just "ran without error"."""
    r1 = np.array([1.0, 1.0, 1.0, 1.0])
    r2 = np.array([1.0, 1.0, 1.0, 0.5])
    d = np.array([0.0, 2.0, 3.0, 0.5])
    areas = _circle_overlap_area(d, r1, r2)
    assert np.isclose(areas[0], np.pi * 1.0**2)
    assert np.isclose(areas[1], 0.0, atol=1e-6)
    assert np.isclose(areas[2], 0.0)
    assert np.isclose(areas[3], np.pi * 0.5**2)


# --------------------------------------------------------------------------- #
# Geometry: contributing_fibre_weights
# --------------------------------------------------------------------------- #

def test_contributing_fibre_weights_closer_fibre_gets_more_weight_and_rows_normalize():
    dec0 = 20.0
    cos_dec = np.cos(np.radians(dec0))
    spaxel_ra = np.array([10.0])
    spaxel_dec = np.array([dec0])
    # fibre 0: dead centre; fibre 1: a bit off; fibre 2: far away (out of range)
    fibre_ra = np.array([10.0, 10.0 + 0.0002 / cos_dec, 10.0 + 10.0 / cos_dec])
    fibre_dec = np.array([dec0, dec0, dec0])
    fibre_radius_deg = 1.3 / 2 / 3600
    spaxel_radius_deg = 0.5 / 3600

    W = contributing_fibre_weights(spaxel_ra, spaxel_dec, fibre_ra, fibre_dec,
                                    fibre_radius_deg, spaxel_radius_deg, weighting="overlap")
    row = W.toarray()[0]
    assert row[2] == 0.0, "fibre far outside the search radius must not contribute"
    assert row[0] > row[1] > 0.0, "the closer fibre must get more weight"
    assert np.isclose(row.sum(), 1.0), "a non-empty row must be normalized to 1"


def test_contributing_fibre_weights_zero_candidates_gives_empty_matrix():
    spaxel_ra = np.array([10.0, 11.0])
    spaxel_dec = np.array([20.0, 21.0])
    W = contributing_fibre_weights(spaxel_ra, spaxel_dec,
                                    np.empty(0), np.empty(0), 0.001, 0.0001)
    assert W.shape == (2, 0)
    assert W.nnz == 0


def test_contributing_fibre_weights_out_of_range_spaxel_gets_all_zero_row():
    """The documented zero-contributing-fibre fallback case: a spaxel with
    nothing in range gets an all-zero row, not an error/NaN."""
    spaxel_ra = np.array([10.0])
    spaxel_dec = np.array([20.0])
    fibre_ra = np.array([50.0])  # far away
    fibre_dec = np.array([20.0])
    W = contributing_fibre_weights(spaxel_ra, spaxel_dec, fibre_ra, fibre_dec,
                                    1.3 / 2 / 3600, 0.5 / 3600)
    assert W.toarray().sum() == 0.0


def test_contributing_fibre_weights_distance_weighting_selectable():
    dec0 = 20.0
    spaxel_ra = np.array([10.0])
    spaxel_dec = np.array([dec0])
    fibre_ra = np.array([10.0, 10.0002])
    fibre_dec = np.array([dec0, dec0])
    W = contributing_fibre_weights(spaxel_ra, spaxel_dec, fibre_ra, fibre_dec,
                                    1.3 / 2 / 3600, 0.5 / 3600, weighting="distance")
    row = W.toarray()[0]
    assert row[0] > row[1] > 0.0
    assert np.isclose(row.sum(), 1.0)

    with pytest.raises(ValueError):
        contributing_fibre_weights(spaxel_ra, spaxel_dec, fibre_ra, fibre_dec,
                                    1.3 / 2 / 3600, 0.5 / 3600, weighting="bogus")


# --------------------------------------------------------------------------- #
# Combine engine: build_spaxel_weighted_entries
# --------------------------------------------------------------------------- #

def _const_func(val):
    def f(w):
        w = np.atleast_1d(w)
        out = np.full_like(w, val, dtype=float)
        return out if len(w) > 1 else float(out[0])
    return f


def test_build_spaxel_weighted_entries_weighted_average_and_dither_collapse():
    """Spaxel 0 gets 100% of fibre NSPEC=1 (fwhm=1.0). Spaxel 1 gets two
    fibre-instances that are both physical NSPEC=1 seen twice (two
    dithers, weight 0.25 each) plus NSPEC=2 (fwhm=3.0, weight 0.5) -- so
    the two NSPEC=1 instances must collapse into one 0.5-weight
    contribution before combining, giving a 50/50 average of 1.0 and 3.0.
    Spaxel 2 has zero weight and must be entirely absent from the result.
    """
    wave_grid = np.linspace(4000, 5000, 11)
    interpolator_dict = {
        "global": {"interpolate_function": _const_func(2.5)},
        1: {"interpolate_function": _const_func(1.0)},
        2: {"interpolate_function": _const_func(3.0)},
    }
    fibre_nspecs = np.array([1, 2, 1])  # instance columns: nspec1, nspec2, nspec1(dither dup)
    W = csr_matrix(np.array([
        [1.0, 0.0, 0.0],
        [0.25, 0.5, 0.25],
        [0.0, 0.0, 0.0],
    ]))
    aps_ids = np.array([100, 200, 300])

    entries = build_spaxel_weighted_entries(
        interpolator_dict, wave_grid, W, fibre_nspecs, aps_ids,
        entry_type="spaxel_lsf_weighted",
    )

    assert set(entries.keys()) == {100, 200}, "spaxel with zero weight must be absent (falls back to global)"
    assert np.isclose(entries[100]["interpolate_function"](4500.0), 1.0)
    assert np.isclose(entries[200]["interpolate_function"](4500.0), 2.0)
    assert entries[100]["n_contrib_fibres"] == 1
    assert entries[200]["n_contrib_fibres"] == 2  # 2 unique NSPECs, dither dup collapsed
    assert entries[100]["type"] == "spaxel_lsf_weighted"


def test_build_spaxel_weighted_entries_missing_fibre_falls_back_to_global_curve():
    """A fibre-instance whose NSPEC has no entry in interpolator_dict at
    all (e.g. it was filled/copied differently, or genuinely absent) must
    use the global curve for that one fibre rather than raising."""
    wave_grid = np.linspace(4000, 5000, 5)
    interpolator_dict = {"global": {"interpolate_function": _const_func(9.0)}}
    W = csr_matrix(np.array([[1.0]]))
    entries = build_spaxel_weighted_entries(
        interpolator_dict, wave_grid, W, np.array([999]), np.array([1]),
        entry_type="spaxel_fwhm_weighted",
    )
    assert np.isclose(entries[1]["interpolate_function"](4500.0), 9.0)


def test_build_spaxel_weighted_entries_empty_inputs():
    assert build_spaxel_weighted_entries({}, None, csr_matrix((0, 0)), np.empty(0), np.empty(0), "t") == {}
    wave_grid = np.linspace(4000, 5000, 5)
    assert build_spaxel_weighted_entries({}, wave_grid, csr_matrix((3, 0)), np.empty(0), [1, 2, 3], "t") == {}


# --------------------------------------------------------------------------- #
# aps_explorer's "Advanced processing options" form wiring
# (aps_l1_preview.py) -- no real data needed, just checks the toggle is
# reachable and stays off by default.
# --------------------------------------------------------------------------- #

def test_explorer_form_exposes_the_flag_off_by_default():
    import PyAPS.aps_l1_preview as l1

    assert "spaxel_weighted_lsf" in l1._ADVANCED_FLAG_NAMES
    assert "spaxel_weighted_lsf" not in l1._ADVANCED_FLAG_DEFAULTS

    parser = l1._build_arg_parser()
    args = parser.parse_args([])
    assert args.spaxel_weighted_lsf is False

    # The form's own default Checklist value (what a user sees the very
    # first time they open the "Load L1 dataset" panel) must not include
    # it either -- _v_flags(current=None, ...) returns `default` verbatim.
    assert "spaxel_weighted_lsf" not in l1._v_flags(None, l1._ADVANCED_FLAG_NAMES, l1._ADVANCED_FLAG_DEFAULTS)


# --------------------------------------------------------------------------- #
# End-to-end against a real stackcube
# --------------------------------------------------------------------------- #

def _load_real(tmp_path, spaxel_weighted_lsf, weighting="overlap",
                join_arms=False, split_arms=False):
    from PyAPS.aps_utils import APSOB
    configdir = tmp_path / "pyaps_config"
    configdir.mkdir(exist_ok=True)
    return APSOB(
        REAL_INFILES,
        caldir=REAL_CALDIR,
        catdir=REAL_CATDIR,
        configdir=str(configdir),
        lsftype="FWHM",
        join_arms=join_arms,
        split_arms=split_arms,
        spaxel_weighted_lsf=spaxel_weighted_lsf,
        spaxel_weighted_lsf_weighting=weighting,
    )


def _summarize_weighted_vs_fallback(ob):
    n_weighted = n_fallback = n_none = 0
    for targ in ob.data():
        for arm_idx, m in enumerate(targ.meta):
            fwhm_f, gfwhm_f = m.get("fwhm"), m.get("gfwhm")
            if fwhm_f is None:
                n_none += 1
            elif fwhm_f is gfwhm_f:
                n_fallback += 1
            else:
                n_weighted += 1
    return n_weighted, n_fallback, n_none


def test_default_off_matches_current_global_only_behaviour(tmp_path):
    """Regression guard: with the flag at its default (False), every IFU
    target's 'fwhm' must be the exact same object as its 'gfwhm' (the
    global interpolator entry) for every arm -- i.e. byte-identical to
    behaviour before this feature existed. Uses a scratch tmp_path
    configdir, never a shared production config/cache path."""
    _skip_unless_exists(*REAL_INFILES, REAL_CALDIR, REAL_CATDIR)
    ob = _load_real(tmp_path, spaxel_weighted_lsf=False)
    assert ob.mode() in ("LIFU", "MIFU", "IFU")
    n_checked = 0
    for targ in ob.data():
        for arm_idx, m in enumerate(targ.meta):
            assert m.get("fwhm") is m.get("gfwhm")
            n_checked += 1
    assert n_checked > 0


def test_enabled_produces_real_weighted_spaxels_that_differ_from_global(tmp_path):
    """With the flag on, at least some spaxels must get a genuinely
    different (weighted) FWHM curve from the flat global one -- otherwise
    the whole feature is silently a no-op on real data."""
    _skip_unless_exists(*REAL_INFILES, REAL_CALDIR, REAL_CATDIR)
    ob = _load_real(tmp_path, spaxel_weighted_lsf=True)

    n_weighted, n_fallback = 0, 0
    for targ in ob.data():
        for arm_idx, m in enumerate(targ.meta):
            fwhm_f, gfwhm_f = m.get("fwhm"), m.get("gfwhm")
            if fwhm_f is None or gfwhm_f is None:
                continue
            if fwhm_f is gfwhm_f:
                n_fallback += 1
            else:
                n_weighted += 1
                assert fwhm_f.get("n_contrib_fibres", 0) > 0
    assert n_weighted > 0, "expected at least some spaxels to get a real weighted LSF/FWHM on real data"


def test_enabled_still_produces_finite_positive_fwhm_values(tmp_path):
    """Whatever spaxel-weighted or fallback value a target ends up with,
    it must still be a sane, usable FWHM curve (finite, positive) at a
    representative wavelength for each arm -- guards against a geometry
    bug silently producing NaN/zero instead of either a real value or a
    clean fallback."""
    _skip_unless_exists(*REAL_INFILES, REAL_CALDIR, REAL_CATDIR)
    ob = _load_real(tmp_path, spaxel_weighted_lsf=True)
    checked = 0
    for targ in ob.data()[:200]:
        for arm_idx, m in enumerate(targ.meta):
            fwhm_f = m.get("fwhm")
            if fwhm_f is None:
                continue
            wave_mid = float(np.mean(targ.spectra[arm_idx].wave))
            val = fwhm_f["interpolate_function"](wave_mid)
            assert np.isfinite(val) and val > 0
            checked += 1
    assert checked > 0


# --------------------------------------------------------------------------- #
# join_arms / split_arms interaction (APSTARG.join_arms's own separate
# fwhm_interp_dict_join lookup, distinct from _process_single_arm_
# vectorized's per-arm one -- see aps_utils.APSOB.__init__'s
# "OPT-IN: spaxel-weighted LSF/FWHM" block in the join_arms section, and
# APSTARG.join_arms's own namespaced cube-mode branch)
# --------------------------------------------------------------------------- #

def test_join_arms_default_off_never_produces_a_weighted_entry(tmp_path):
    """join_arms=True, spaxel_weighted_lsf left at its default (False):
    every joined target must still just get the global FWHM, exactly as
    before this feature's join_arms handling existed."""
    _skip_unless_exists(*REAL_INFILES, REAL_CALDIR, REAL_CATDIR)
    ob = _load_real(tmp_path, spaxel_weighted_lsf=False, join_arms=True, split_arms=False)
    n_weighted, n_fallback, n_none = _summarize_weighted_vs_fallback(ob)
    assert n_weighted == 0
    assert n_fallback + n_none == len(ob.data())  # one meta entry per target after a real join


def test_join_arms_enabled_produces_weighted_entries_on_the_joined_curve(tmp_path):
    """join_arms=True (no split): the per-arm spaxel-weighted result must
    NOT leak through unmodified (it would be wrong -- built from only one
    arm's own wavelength range) -- a fresh weighted result built against
    the joined/stitched interpolator (spanning the full combined
    wavelength range) must be used instead. Confirms real, non-fallback
    entries exist and that exactly one meta entry per target survives
    (no split_arms == everything folded into spectra[0])."""
    _skip_unless_exists(*REAL_INFILES, REAL_CALDIR, REAL_CATDIR)
    ob = _load_real(tmp_path, spaxel_weighted_lsf=True, join_arms=True, split_arms=False)
    n_weighted, n_fallback, n_none = _summarize_weighted_vs_fallback(ob)
    assert n_weighted > 0
    assert (n_weighted + n_fallback + n_none) == len(ob.data())


def test_split_arms_preserves_the_per_arm_weighted_result(tmp_path):
    """join_arms=True *with* split_arms=True: APSTARG.join_arms's own
    split_arms branch deliberately does not touch meta at all (see its
    "we do not modify the meta info" comment), so the per-arm
    spaxel-weighted result from _assign_arm_results_to_targets must
    survive completely unmodified -- and since it does, it must show the
    same kind of real weighted (non-fallback) entries as the plain
    per-arm case does."""
    _skip_unless_exists(*REAL_INFILES, REAL_CALDIR, REAL_CATDIR)
    ob = _load_real(tmp_path, spaxel_weighted_lsf=True, join_arms=True, split_arms=True)
    n_weighted, n_fallback, n_none = _summarize_weighted_vs_fallback(ob)
    assert n_weighted > 0
