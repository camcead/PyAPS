"""
tests/test_l1_fwhm_lsf.py - Regression tests for round-10/11 fixes found
while investigating a real crash + a real performance report against a
large L1 IFU stackcube (164x188 = 30,832 spaxels):

1. `apsPlot/fwhm.py:fwhm_overview_figure` used to draw one Scattergl trace
   per fibre unconditionally — on a real 30k-spaxel dataset that's 60k+
   traces and ~120M floats in a single Plotly JSON response, large enough
   (confirmed via a live SSH tunnel log showing a huge sustained byte
   stream immediately followed by "Connection reset by peer") to crash the
   whole Dash server sending it. Now capped via subsampling.

2. `aps_lsf.py`'s per-fibre LSF interpolator closure (`make_fiber_func`'s
   `interp_func`) evaluated a wavelength grid one point at a time in a
   plain Python `for` loop instead of one vectorized scipy BSpline call —
   ~2.2ms/call. Now vectorized, ~20x faster, confirmed bit-identical
   output against the old implementation.

3. `aps_lsf.py`'s *global* interpolator closure (`global_interp_func`) had
   the same per-point-loop anti-pattern — and turned out to be the real
   dominant bottleneck for IFU/LIFU/MIFU L1 loads specifically:
   `aps_utils._assign_arm_results_to_targets` always assigns
   `fwhm_interp_dict['global']` (never a per-fibre one) for these modes,
   so `aps_l1_preview._build_fwhm_cache` evaluates *this* closure once per
   spaxel per arm, not the per-fibre one fix #2 above covers. Confirmed
   live end-to-end on the real 30,832-spaxel dataset:
   `_build_fwhm_cache` 219.74s -> 3.90s, total L1 load 252.05s -> 36.12s.

4. `aps_lsf.py`'s *merged-arms* interpolator closure (`make_merged_func`'s
   `interp_func`, used only when `join_arms=True` stitches multiple
   calibration files across a wavelength gap/overlap) had the same
   per-point-loop anti-pattern — fixed identically, verified against both
   a synthetic overlap case and a synthetic gap case (20,000+ points plus
   every region's exact boundary) and against real merged GREENH11+REDH11
   calibration data with a genuine wavelength gap.

5. `aps_utils.APSOB._process_arms_serial` processed each arm's file
   read + LSF setup + vectorized processing one after another despite
   each arm being fully independent (no shared mutable state) — now runs
   them concurrently via a `ThreadPoolExecutor` (not a process pool —
   arms share multi-GB flux/ivar arrays that would be expensive to pickle
   across a process boundary; threads share memory, no such cost).
   Genuinely effective because `OMP_NUM_THREADS`/etc. are already pinned
   to 1 in this module (see its own top-of-file `os.environ` lines,
   there to avoid oversubscription with this codebase's *other*
   multiprocessing elsewhere) and both arms' time is dominated by FITS
   I/O and numpy/scipy C-level calls, which release the GIL — confirmed
   live: 36.12s -> 27.42s total L1 load time on the real dataset (on top
   of, not instead of, fix #3's much larger win).

6. Fix #3 above turned out to have an operational gap: `dill.dump(self, ...)`
   freezes an LSFInterpolator's closures' actual code objects at save
   time, so an already-cached `.dill` built *before* fix #3 kept running
   its old, slow closure forever regardless of the source fix — loading
   it just deserializes the old function unchanged. Confirmed live
   against a real dataset whose cache predated the fix by ~6 months:
   per-call cost was still ~7.8ms (matching the *old* pre-fix behaviour,
   not fix #3's ~106µs benchmark), and a full L1 load of a comparably
   large (31,132-spaxel) 2-arm LIFU stackcube took 280.1s end to end.
   Fixed properly this time (not just "should self-heal" as fix #3's own
   notes optimistically claimed, which turned out not to be true — the
   default `overwrite=False` never rebuilds an existing pickle no matter
   how stale): every `LSFInterpolator` now stamps itself with
   `_cache_format_version` (`aps_lsf._LSF_CACHE_FORMAT_VERSION`) at
   construction time, and `run_lsf_analysis` checks that stamp on load,
   silently and automatically rebuilding (once) any cache that predates
   it instead of trusting it forever. Confirmed live end to end on the
   same real dataset: 280.1s (stale cache) -> 32.8s (one-time rebuild,
   both arms) -> 24.5s (now-current cache, matching fix #3's own
   ballpark) on repeat.

Uses real WEAVE reference data (set PYAPS_TEST_DATA);
skipped (not failed) if a given file isn't found.
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

LSF_CAL_FILE = PYAPS_DATA + "/CAL/20240320/lsf_GREENH11_LIFU.fits"
LSF_CAL_FILE_2 = PYAPS_DATA + "/CAL/20240320/lsf_REDH11_LIFU.fits"


def _skip_unless_exists(*paths):
    for p in paths:
        if not Path(p).exists():
            pytest.skip(f"reference data not found: {p}")


# --------------------------------------------------------------------------- #
# aps_lsf.py: vectorized per-fibre interpolator
# --------------------------------------------------------------------------- #

def test_lsf_interp_func_vectorized_matches_reference_and_is_fast():
    """Rigorous check, not just a timing number: independently recompute
    what the (fixed) interp_func should return — _bspl(wavelengths) inside
    [wmin, wmax], first_v/last_v clamped outside it — from the same
    picklable BSpline object each fibre's closure wraps, and compare
    exactly. Also asserts a generous performance ceiling (500us/call, well
    below the pre-fix ~2227us/call but with real margin above the
    post-fix ~106us/call measured on this exact file) so a future
    regression back to a per-point Python loop gets caught."""
    _skip_unless_exists(LSF_CAL_FILE)
    import time
    from PyAPS.aps_lsf import run_lsf_analysis

    interp = run_lsf_analysis(
        LSF_CAL_FILE, figdir=None, figname="test_lsf", debug=False, make_plot=False,
        smooth_length=None, kernel_type="gaussian", overwrite=False, save_pickle=False,
    )
    assert interp is not None
    d = interp.get_interpolator_dict()
    fiber_keys = [k for k in d.keys() if k != "global"]
    assert len(fiber_keys) > 0

    # A grid spanning well outside the valid range on both ends, to also
    # exercise the boundary-clamping branches, not just the in-range path.
    grids = [np.linspace(4740.0, 5420.0, 1000), np.linspace(4000.0, 6000.0, 500)]

    sample_keys = fiber_keys[:20]
    total_calls = 0
    t0 = time.time()
    for k in sample_keys:
        func = d[k]["interpolate_function"]
        bspl, wmin, wmax, first_v, last_v = func.__defaults__
        for grid in grids:
            got = np.atleast_1d(func(grid))
            expected = np.where(
                grid < wmin, first_v, np.where(grid > wmax, last_v, bspl(grid))
            )
            expected = np.maximum(expected, 0.1)
            assert np.array_equal(got, expected), f"mismatch for fibre {k}"
            total_calls += 1
    dt = time.time() - t0

    per_call_us = dt / total_calls * 1e6
    assert per_call_us < 500, (
        f"interp_func averaged {per_call_us:.0f}us/call — expected well under 500us "
        "(pre-fix per-point-loop version measured ~2227us/call on this same file); "
        "this likely means the vectorized fix regressed back to a per-point Python loop."
    )


def test_lsf_interp_func_boundary_clamping():
    """Explicit, minimal check of the clamping behaviour alone (in addition
    to the broader comparison above) — values below wmin must equal
    first_v exactly, above wmax must equal last_v exactly."""
    _skip_unless_exists(LSF_CAL_FILE)
    from PyAPS.aps_lsf import run_lsf_analysis

    interp = run_lsf_analysis(
        LSF_CAL_FILE, figdir=None, figname="test_lsf2", debug=False, make_plot=False,
        smooth_length=None, kernel_type="gaussian", overwrite=False, save_pickle=False,
    )
    d = interp.get_interpolator_dict()
    fiber_keys = [k for k in d.keys() if k != "global"]
    func = d[fiber_keys[0]]["interpolate_function"]
    _, wmin, wmax, first_v, last_v = func.__defaults__

    # A single-element input triggers interp_func's own "return a plain
    # float, not a 1-element array" branch (see its final line) -- match
    # that here rather than indexing into it.
    below = func(np.array([wmin - 500.0]))
    above = func(np.array([wmax + 500.0]))
    assert below == max(first_v, 0.1)
    assert above == max(last_v, 0.1)


# --------------------------------------------------------------------------- #
# apsPlot/fwhm.py: capped "cloud" trace count on the overview figure
# --------------------------------------------------------------------------- #

def _fake_fwhm_cache(n_fibers, n_wave=200):
    """Synthetic (fast, no real data needed) stand-in for a real
    _build_fwhm_cache() result, sized to exercise the subsampling cap."""
    aps_ids = list(range(1, n_fibers + 1))
    wave = np.linspace(4700.0, 5400.0, n_wave)
    fiber_fwhm = {aid: [np.full(n_wave, 1.0 + 0.001 * aid)] for aid in aps_ids}
    global_fwhm = [np.ones(n_wave)]
    return aps_ids, wave, fiber_fwhm, global_fwhm


def _count_cloud_traces(fig):
    """"Cloud" traces are the thin grey unlabelled ones — everything except
    the always-drawn "Global FWHM" trace and any highlighted-fibre trace."""
    return sum(
        1 for tr in fig.data
        if tr.name not in ("Global FWHM",) and not (tr.name or "").startswith("Fibre ")
    )


def test_fwhm_overview_caps_cloud_traces_for_large_fiber_counts():
    """Regression guard for the crash: on a fibre count far above the cap
    (mirroring a real ~30k-spaxel stackcube), the number of individual
    "cloud" traces must stay capped, not scale linearly with fibre count."""
    from PyAPS.apsPlot.fwhm import fwhm_overview_figure, _MAX_CLOUD_FIBERS

    aps_ids, wave, fiber_fwhm, global_fwhm = _fake_fwhm_cache(30832)
    fig = fwhm_overview_figure(["Arm0"], [wave], fiber_fwhm, global_fwhm, aps_ids)

    n_cloud = _count_cloud_traces(fig)
    assert n_cloud <= _MAX_CLOUD_FIBERS
    assert n_cloud > 0  # didn't accidentally drop the cloud entirely
    assert "showing" in fig.layout.title.text and "of 30,832 fibres" in fig.layout.title.text


def test_fwhm_overview_does_not_subsample_small_fiber_counts():
    """The other direction: a normal-sized dataset (well under the cap,
    matching every real dataset tested before this round) must keep
    showing every fibre's own line, not silently start dropping some —
    the cap must only ever kick in when it's actually needed."""
    from PyAPS.apsPlot.fwhm import fwhm_overview_figure, _MAX_CLOUD_FIBERS

    n = 200
    assert n < _MAX_CLOUD_FIBERS
    aps_ids, wave, fiber_fwhm, global_fwhm = _fake_fwhm_cache(n)
    fig = fwhm_overview_figure(["Arm0"], [wave], fiber_fwhm, global_fwhm, aps_ids)

    n_cloud = _count_cloud_traces(fig)
    assert n_cloud == n
    assert "showing" not in fig.layout.title.text


def test_fwhm_overview_highlighted_fiber_always_included_even_when_capped():
    """A highlighted (selected) fibre must always get its own bold curve
    even if the random/strided cloud subsample wouldn't otherwise have
    included that exact fibre — this is a separate code path from the
    cloud loop and must never be affected by the cap."""
    from PyAPS.apsPlot.fwhm import fwhm_overview_figure

    aps_ids, wave, fiber_fwhm, global_fwhm = _fake_fwhm_cache(30832)
    highlighted = 12345  # not a multiple of any small stride -- exercises the general case
    fig = fwhm_overview_figure(
        ["Arm0"], [wave], fiber_fwhm, global_fwhm, aps_ids, highlighted_aps_id=highlighted
    )
    names = [tr.name for tr in fig.data]
    assert f"Fibre {highlighted}" in names


# --------------------------------------------------------------------------- #
# aps_lsf.py: vectorized *global* interpolator (the actual real-world
# bottleneck for IFU/LIFU/MIFU L1 loads — see module docstring, fix #3)
# --------------------------------------------------------------------------- #

def test_lsf_global_interp_func_matches_reference_and_is_fast():
    """Same rigor as the per-fibre test above, applied to the 'global'
    entry specifically — this is the one aps_l1_preview._build_fwhm_cache
    actually calls 61,664 times on a real IFU/LIFU stackcube."""
    _skip_unless_exists(LSF_CAL_FILE)
    import time
    from PyAPS.aps_lsf import run_lsf_analysis

    interp = run_lsf_analysis(
        LSF_CAL_FILE, figdir=None, figname="test_lsf_global", debug=False, make_plot=False,
        smooth_length=None, kernel_type="gaussian", overwrite=False, save_pickle=False,
    )
    global_entry = interp.get_interpolator_dict()["global"]
    func = global_entry["interpolate_function"]
    defaults = func.__defaults__
    names = func.__code__.co_varnames[1:1 + len(defaults)]
    kw = dict(zip(names, defaults))

    grids = [np.linspace(kw["_wave_min"], kw["_wave_max"], 1000),
             np.linspace(kw["_wave_min"] - 500, kw["_wave_max"] + 500, 2000)]

    t0 = time.time()
    n_calls = 0
    for grid in grids:
        got = np.atleast_1d(func(grid))
        spline_vals = kw["_spline"](grid)
        expected = np.where(grid < kw["_wave_min"], kw["_first"],
                             np.where(grid > kw["_wave_max"], kw["_last"], spline_vals))
        expected = np.maximum(expected, 0.1)
        assert np.array_equal(got, expected)
        n_calls += 1
    dt = time.time() - t0

    assert dt / n_calls < 0.01, (
        f"global_interp_func averaged {dt/n_calls*1000:.1f}ms/call — expected well under 10ms "
        "(pre-fix per-point-loop version measured ~3.4-3.6ms/call end-to-end on a real stackcube); "
        "this likely means the vectorized fix regressed back to a per-point Python loop."
    )


def test_lsf_global_interp_func_gap_branch():
    """Explicit synthetic check of the gap-linear-interpolation branch
    (not exercised by every real calibration file — the GREENH11 file used
    above has no gap), including exact gap-boundary edge points."""
    from PyAPS.aps_lsf import LSFInterpolator
    from scipy.interpolate import InterpolatedUnivariateSpline

    np.random.seed(0)
    wave = np.linspace(4000, 5000, 50)
    fwhm_vals = 1.5 + 0.001 * wave + np.random.normal(0, 0.01, len(wave))
    spline = InterpolatedUnivariateSpline(wave, fwhm_vals, k=3)

    interp = LSFInterpolator()
    interp.wave_grid = wave
    interp.global_spline = spline
    interp.has_gap = True
    interp.gap_start, interp.gap_end = 4400.0, 4600.0
    interp.gap_fwhm_start = float(spline(interp.gap_start))
    interp.gap_fwhm_end = float(spline(interp.gap_end))
    interp.fiber_splines = {}
    interp._create_interpolator_dict()

    func = interp.interpolator_dict["global"]["interpolate_function"]
    edge_grid = np.array([4000.0, 5000.0, 4400.0, 4600.0, 4400.0 + 1e-6, 4600.0 - 1e-6, 4500.0])
    out = func(edge_grid)
    assert np.all(np.isfinite(out))
    # At the exact gap midpoint, the value must be the linear blend, not
    # the (very different) spline value that region intentionally ignores.
    mid_expected = 0.5 * interp.gap_fwhm_start + 0.5 * interp.gap_fwhm_end
    assert abs(func(np.array([4500.0])) - mid_expected) < 1e-9


# --------------------------------------------------------------------------- #
# aps_lsf.py: vectorized merged-arms (join_arms) interpolator
# --------------------------------------------------------------------------- #

def test_lsf_merged_interp_func_real_gap_data():
    """End-to-end check against real merged GREENH11+REDH11 calibration
    data (a genuine ~600A wavelength gap between the two arms) — not just
    synthetic: builds the real merged interpolator and checks every fibre
    produces distinct, finite, positive values both in-range and inside
    the gap, plus a generous performance ceiling."""
    _skip_unless_exists(LSF_CAL_FILE, LSF_CAL_FILE_2)
    import time
    from PyAPS.aps_lsf import run_lsf_analysis

    interp = run_lsf_analysis(
        [LSF_CAL_FILE, LSF_CAL_FILE_2], figdir=None, figname="test_merged",
        debug=False, make_plot=False, smooth_length=None, kernel_type="gaussian",
        overwrite=False, save_pickle=False,
    )
    d = interp.get_interpolator_dict()
    fiber_keys = [k for k in d.keys() if k != "global"]
    assert len(fiber_keys) > 0

    wave_grid = np.linspace(4740.0, 6760.0, 500)  # spans the real gap
    t0 = time.time()
    sample_vals = []
    for k in fiber_keys[:50]:
        vals = d[k]["interpolate_function"](wave_grid)
        assert np.all(np.isfinite(vals)) and np.all(vals > 0)
        sample_vals.append(vals)
    dt = time.time() - t0
    assert dt / 50 < 0.01, f"merged interp_func averaged {dt/50*1000:.1f}ms/call — expected well under 10ms"

    # Fibres must be genuinely individualized, not all collapsed to the
    # same curve (a real bug this exact codebase hit before, per aps_lsf.py's
    # own debug-mode self-check at construction time).
    n_unique = len({tuple(np.round(v, 6)) for v in sample_vals})
    assert n_unique > 1


def _reference_merged_interp(wavelengths, _f0_fwhm, _f0_first, _f0_last,
                              _f1_fwhm, _f1_first, _f1_last, _wave_eval,
                              _w1_min, _w1_max, _w2_min, _w2_max,
                              _ovlp_start, _ovlp_end, _has_ovlp):
    """The original (pre-fix) scalar per-point loop, kept here only as an
    independent reference to check the real, live vectorized closure
    against — not itself the code under test."""
    wavelengths = np.atleast_1d(wavelengths)
    fwhm = np.zeros_like(wavelengths, dtype=float)
    for j, w in enumerate(wavelengths):
        if w < _w1_min:
            fwhm[j] = _f0_first
            continue
        if w > _w2_max:
            fwhm[j] = _f1_last
            continue
        idx = np.searchsorted(_wave_eval, w)
        if idx == 0:
            idx = 0
        elif idx >= len(_wave_eval):
            idx = len(_wave_eval) - 1
        elif abs(_wave_eval[idx] - w) > abs(_wave_eval[idx - 1] - w):
            idx = idx - 1
        if w <= _w1_max and (not _has_ovlp or w < _ovlp_start):
            fwhm[j] = _f0_fwhm[idx]
        elif not _has_ovlp and _w1_max < w < _w2_min:
            alpha = (w - _w1_max) / (_w2_min - _w1_max)
            fwhm[j] = (1 - alpha) * _f0_last + alpha * _f1_first
        elif _has_ovlp and _ovlp_start <= w <= _ovlp_end:
            fwhm[j] = (_f0_fwhm[idx] + _f1_fwhm[idx]) / 2.0
        elif w >= _w2_min:
            fwhm[j] = _f1_fwhm[idx]
        else:
            fwhm[j] = np.nan
    return fwhm


def _make_synthetic_fiber_bspline(wmin, wmax, offset, seed):
    """A real, picklable scipy BSpline (extrapolate=False, matching what
    read_lsf_splines_file's real output looks like) — used to drive
    aps_lsf.py's actual _merge_multiple_files with synthetic-but-realistic
    per-fibre calibration data, rather than faking its output."""
    from scipy.interpolate import BSpline, make_interp_spline
    rng = np.random.RandomState(seed)
    x = np.linspace(wmin, wmax, 30)
    y = 1.5 + offset + 0.0003 * x + rng.normal(0, 0.01, len(x))
    spl = make_interp_spline(x, y, k=3)
    return BSpline(spl.t, spl.c, spl.k, extrapolate=False)


def test_lsf_merged_interp_func_matches_reference_overlap_and_gap():
    """Rigorous check covering both branches the real-calibration-data test
    above can't independently isolate (that file pair has a gap, not an
    overlap): drives aps_lsf.py's real, private `_merge_multiple_files`
    directly with synthetic-but-realistic per-fibre BSpline data for both
    a has_overlap=True and a has_overlap=False (gap) wavelength layout,
    extracts the resulting *real* interp_func closure's own bound
    parameters via `__defaults__` (so the reference comparison uses
    exactly what the real closure actually used internally, not a
    reconstruction), and compares 5000+ random points plus every region's
    exact boundary against the original scalar-loop formula."""
    from PyAPS.aps_lsf import LSFInterpolator

    for has_ovlp in (True, False):
        if has_ovlp:
            w1_range, w2_range = (4000.0, 5200.0), (5000.0, 6000.0)
        else:
            w1_range, w2_range = (4000.0, 5200.0), (5400.0, 6000.0)

        file0_splines = {1: {"bspline": _make_synthetic_fiber_bspline(*w1_range, 0.01, seed=10)}}
        file1_splines = {1: {"bspline": _make_synthetic_fiber_bspline(*w2_range, 0.02, seed=11)}}
        file0_splines[1]["wave_range"] = w1_range
        file1_splines[1]["wave_range"] = w2_range

        interp = LSFInterpolator()
        merged = interp._merge_multiple_files(
            [file0_splines, file1_splines], [w1_range, w2_range], wave_grid_resolution=1.0,
        )
        real_func = merged[1]["interpolate_function"]
        defaults = real_func.__defaults__
        names = real_func.__code__.co_varnames[1:1 + len(defaults)]
        kw = dict(zip(names, defaults))
        assert kw["_has_ovlp"] == has_ovlp  # sanity: got the scenario we asked for

        w1_min, w1_max = w1_range
        w2_min, w2_max = w2_range
        dense = np.random.RandomState(42).uniform(3500, 6500, 5000)
        edges = [w1_min, w1_max, w2_min, w2_max, w1_min - 1e-3, w1_min + 1e-3,
                 w2_max - 1e-3, w2_max + 1e-3]
        if has_ovlp:
            edges += [kw["_ovlp_start"], kw["_ovlp_end"],
                      kw["_ovlp_start"] - 1e-3, kw["_ovlp_end"] + 1e-3]
        else:
            gap_mid = (w1_max + w2_min) / 2
            edges += [gap_mid, w1_max + 1e-3, w2_min - 1e-3]
        test_grid = np.concatenate([dense, np.array(edges)])

        expected = _reference_merged_interp(test_grid, **kw)
        actual = real_func(test_grid)

        both_nan = np.isnan(expected) & np.isnan(actual)
        diffs = np.where(both_nan, 0.0, np.abs(expected - actual))
        assert np.nanmax(diffs) < 1e-9, f"has_overlap={has_ovlp}: mismatch found"
        assert np.sum(np.isnan(expected) != np.isnan(actual)) == 0


# --------------------------------------------------------------------------- #
# aps_utils.py: concurrent (threaded) arm processing
# --------------------------------------------------------------------------- #

L1_HR_2ARM_FILES = [
    PYAPS_DATA + "/L1/20240915/single_3077580.fit",
    PYAPS_DATA + "/L1/20240915/single_3077579.fit",
]
CALDIR = PYAPS_DATA + "/CAL"
CATDIR = PYAPS_DATA + "/CAT"


def test_process_arms_serial_threaded_preserves_order_and_correctness():
    """_process_arms_serial now runs each arm's file-read/LSF-setup/
    vectorized-processing concurrently via ThreadPoolExecutor instead of a
    plain for loop (see module docstring, fix #5) — the one thing that
    absolutely must not regress is per-arm result ordering (arm_results[i]
    must correspond to self._infiles[i], since downstream code indexes by
    that same position) and per-target spectrum assignment. Checked
    against a real two-arm HR dataset: each target's spectra[0]/[1] must
    fall within *that specific arm file's own* wavelength range, and results
    must be reproducible across repeated loads (no race condition silently
    scrambling which arm's data lands in which slot)."""
    _skip_unless_exists(*L1_HR_2ARM_FILES, CALDIR, CATDIR)
    from PyAPS import aps_l1_preview as l1_mod

    # A small aps_ids subset keeps this fast — correctness of the
    # concurrency, not full-dataset performance, is what's under test here
    # (performance is covered by the live-timed numbers in the module
    # docstring, not re-measured on every CI run).
    apsids = ",".join(str(i) for i in range(1, 31))

    for _ in range(3):  # repeat: a race condition wouldn't necessarily show up once
        l1_mod.load_from_form_fields(
            infiles_text="\n".join(L1_HR_2ARM_FILES), infiles_list=None, l1ref=None, l2ref=None,
            apsids=apsids, targsrvy=None, targclass=None, maskids=None,
            area=None, mask_areas_text=None, wlranges_text=None, arms_ratio=None,
            caldir=CALDIR, catdir=CATDIR, configdir=None,
            flags=["sens_corr", "safe_mask_gaps"],
        )
        targs = l1_mod.STATE.targs
        # Not every requested id in 1..30 necessarily exists in this real
        # file (e.g. sky/calib slots) — the exact count isn't the point of
        # this test, just that a reasonable number came back at all.
        assert 20 <= len(targs) <= 30
        for t in targs:
            assert len(t.spectra) == 2
            wave0, wave1 = t.spectra[0].wave, t.spectra[1].wave
            # GREENH11 (file 0, single_3077580.fit) and REDH11 (file 1,
            # single_3077579.fit) occupy disjoint wavelength windows — if
            # arm_results ever got scrambled by the thread pool, a target's
            # spectra[0]/[1] would show the wrong (or overlapping) ranges.
            assert wave0.max() < 5600.0, f"aps_id={t.aps_id}: arm 0 wave range looks wrong: {wave0.min()}-{wave0.max()}"
            assert wave1.min() > 5900.0, f"aps_id={t.aps_id}: arm 1 wave range looks wrong: {wave1.min()}-{wave1.max()}"
            assert np.all(np.isfinite(t.spectra[0].flux))
            assert np.all(np.isfinite(t.spectra[1].flux))


# --------------------------------------------------------------------------- #
# aps_lsf.py: automatic stale-cache detection/rebuild (fix #6)
# --------------------------------------------------------------------------- #

def test_stale_lsf_cache_is_detected_and_rebuilt(tmp_path):
    """A `.dill` with no `_cache_format_version` at all (every real cache
    built before this fix existed, confirmed on real data — see module
    docstring) or an outdated one must be rebuilt automatically on next
    load, not trusted forever. Uses a scratch tmp_path, never a real
    configs/ cache directory (see feedback_test_isolation)."""
    _skip_unless_exists(LSF_CAL_FILE)
    from PyAPS import aps_lsf

    pickle_dir = tmp_path / "lsf_cache"
    pickle_dir.mkdir()

    # Build a genuine, current-format interpolator first.
    interp = aps_lsf.run_lsf_analysis(
        LSF_CAL_FILE, figdir=None, make_plot=False,
        pickle_dir=str(pickle_dir), overwrite=True,
    )
    assert interp is not None
    assert interp._cache_format_version == aps_lsf._LSF_CACHE_FORMAT_VERSION
    pickle_path = aps_lsf.get_output_pickle_path(LSF_CAL_FILE, str(pickle_dir))
    assert Path(pickle_path).is_file()
    mtime_fresh = Path(pickle_path).stat().st_mtime

    # Simulate a genuinely old cache the way every real one on this
    # machine actually looked before this fix: no attribute at all
    # (rather than hand-setting an old integer, which a pickle built by
    # any *actual* prior version of this code would never have had).
    stale = aps_lsf.LSFInterpolator.load(pickle_path)
    del stale.__dict__["_cache_format_version"]
    with open(pickle_path, "wb") as f:
        aps_lsf.PICKLE_MODULE.dump(stale, f)

    # Loading it again (overwrite still False, exactly like the real load
    # path) must notice the stale format and rebuild rather than return
    # the stale object as-is.
    rebuilt = aps_lsf.run_lsf_analysis(
        LSF_CAL_FILE, figdir=None, make_plot=False,
        pickle_dir=str(pickle_dir), overwrite=False,
    )
    assert rebuilt is not None
    assert rebuilt._cache_format_version == aps_lsf._LSF_CACHE_FORMAT_VERSION
    mtime_rebuilt = Path(pickle_path).stat().st_mtime
    assert mtime_rebuilt > mtime_fresh, "stale cache was not actually rewritten to disk"


def test_current_format_lsf_cache_is_not_needlessly_rebuilt(tmp_path):
    """The flip side of the test above — a cache that's already current
    must be loaded as-is, not rebuilt on every single load (that would
    defeat the point of caching at all)."""
    _skip_unless_exists(LSF_CAL_FILE)
    from PyAPS import aps_lsf

    pickle_dir = tmp_path / "lsf_cache"
    pickle_dir.mkdir()

    aps_lsf.run_lsf_analysis(LSF_CAL_FILE, figdir=None, make_plot=False,
                              pickle_dir=str(pickle_dir), overwrite=True)
    pickle_path = aps_lsf.get_output_pickle_path(LSF_CAL_FILE, str(pickle_dir))
    mtime_first = Path(pickle_path).stat().st_mtime

    aps_lsf.run_lsf_analysis(LSF_CAL_FILE, figdir=None, make_plot=False,
                              pickle_dir=str(pickle_dir), overwrite=False)
    mtime_second = Path(pickle_path).stat().st_mtime
    assert mtime_second == mtime_first, "an already-current cache was rebuilt unnecessarily"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
