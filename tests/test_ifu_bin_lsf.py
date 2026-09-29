"""
tests/test_ifu_bin_lsf.py - Tests for threading per-spaxel LSF/FWHM
(`aps_utils.APSOB`'s `spaxel_weighted_lsf`, see
`tests/test_spaxel_weighted_lsf.py`) down to Voronoi/PowerBin *bin*-level
resolution buckets in the IFU Gal/ExGal pipeline (`aps_ifu_Gal.py`,
`aps_ifu_ExGal.py`, `IFUExGalPPXF.py`, `IFUExGalEMIPPXF.py`,
`IFUExGalLS.py`, `aps_ifu_rvs.py`, `aps_ifu_ferre.py`), and the PowerBin
(Cappellari 2025) binning-engine swap that replaced vorbin/Voronoi
(Cappellari & Copin 2003) in `ExGalPrepare.define_voronoi_bins`.

Background: `aps_ifu_Gal.py`/`aps_ifu_ExGal.py` used to call
`APSOBJ_inst.get_fwhm(aps_id=None, fwhm_key="gfwhm")` -- explicitly the
single global curve, from target 0, reused for every bin/spaxel in the
whole patch (RVS/FERRE in Gal; template convolution -> every Voronoi
bin's PPXF/EMIPPXF/LineStrength fit in ExGal). `aps_ifu_spaxel_contrib.
aggregate_bin_lsf`/`bucket_lsf_curves`/`bucket_lsf_for_voronoi_bins`
close that gap: flux-weight-average each bin's member spaxels' own
per-spaxel LSF, then quantize into a handful of resolution buckets so
template preparation (the expensive part) runs once per bucket instead
of once per patch.

Uses real reference stackcubes/CAL/templates already present on this
machine (set PYAPS_TEST_DATA); skipped (not failed) if not found.
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

from PyAPS import ExGalPrepare
from PyAPS.aps_ifu_spaxel_contrib import (
    aggregate_bin_lsf,
    bucket_lsf_curves,
    bucket_lsf_for_voronoi_bins,
)

REAL_INFILES = [
    PYAPS_DATA + "/L1/20240105/stackcube_3039542.fit",  # BLUE
    PYAPS_DATA + "/L1/20240105/stackcube_3039541.fit",  # RED
]
REAL_CALDIR = PYAPS_DATA + "/CAL"
REAL_CATDIR = PYAPS_DATA + "/CAT"
REAL_IFU_PARAMS = PYAPS_HOME + "/configs/ExGal_configs/LIFULR11.json"
REAL_IFU_CONFIG_DIR = PYAPS_HOME + "/configs/ExGal_configs"
REAL_TEMPLATES_TARBALL = PYAPS_DATA + "/downloads/PyAPS_templates.tar.gz"


def _skip_unless_exists(*paths):
    for p in paths:
        if not Path(p).exists():
            pytest.skip(f"reference data not found: {p}")


def _const_func(val):
    def f(w):
        w = np.atleast_1d(w)
        out = np.full_like(w, val, dtype=float)
        return out if len(w) > 1 else float(out[0])
    return f


# --------------------------------------------------------------------------- #
# aggregate_bin_lsf: flux-weighted spaxel -> bin aggregation
# --------------------------------------------------------------------------- #

def test_aggregate_bin_lsf_flux_weighted_average():
    wave_grid = np.linspace(4000, 5000, 11)
    # bin 0: spaxels [0,1] fwhm=1.0,2.0 flux=1,3 -> (1*1+2*3)/4 = 1.75
    # bin 1: spaxels [2,3,4] fwhm all 5.0
    funcs = [_const_func(1.0), _const_func(2.0), _const_func(5.0), _const_func(5.0), _const_func(5.0)]
    bin_num = np.array([0, 0, 1, 1, 1])
    flux = np.array([1.0, 3.0, 2.0, 2.0, 2.0])

    entries = aggregate_bin_lsf(funcs, bin_num, flux, wave_grid)
    assert sorted(entries.keys()) == [0, 1]
    assert np.isclose(entries[0]["interpolate_function"](4500.0), 1.75)
    assert np.isclose(entries[1]["interpolate_function"](4500.0), 5.0)
    assert entries[0]["n_contrib_spaxels"] == 2
    assert entries[1]["n_contrib_spaxels"] == 3


def test_aggregate_bin_lsf_none_func_and_zero_weight_handling():
    wave_grid = np.linspace(4000, 5000, 11)
    # spaxel with None func contributes nothing; bin with all-zero weight is absent
    entries = aggregate_bin_lsf(
        [None, _const_func(3.0)], np.array([7, 7]), np.array([1.0, 1.0]), wave_grid)
    assert entries[7]["interpolate_function"](4500.0) == 3.0
    assert entries[7]["n_contrib_spaxels"] == 1

    entries2 = aggregate_bin_lsf([_const_func(9.0)], np.array([3]), np.array([0.0]), wave_grid)
    assert entries2 == {}


def test_aggregate_bin_lsf_toggling_feature_off_is_a_no_op():
    """When every spaxel shares the same (global) curve -- i.e. the
    spaxel_weighted_lsf feature is off -- the flux-weighted average of N
    identical curves must be that exact same curve back out."""
    wave_grid = np.linspace(4000, 5000, 11)
    funcs = [_const_func(2.5)] * 5
    flux = np.array([1.0, 7.0, 0.3, 2.2, 5.0])  # arbitrary, shouldn't matter
    entries = aggregate_bin_lsf(funcs, np.zeros(5, dtype=int), flux, wave_grid)
    assert np.isclose(entries[0]["interpolate_function"](4500.0), 2.5)


# --------------------------------------------------------------------------- #
# bucket_lsf_curves: quantization
# --------------------------------------------------------------------------- #

def test_bucket_lsf_curves_identity_when_few_bins():
    wave_grid = np.linspace(4000, 5000, 11)
    bin_curves = {10: _const_func(1.0), 20: _const_func(2.0)}
    b2b, bcurves = bucket_lsf_curves(bin_curves, wave_grid, max_buckets=6)
    assert len(set(b2b.values())) == 2


def test_bucket_lsf_curves_no_cross_group_mixing():
    wave_grid = np.linspace(4000, 5000, 11)
    rng = np.random.default_rng(0)
    bin_curves, true_val = {}, {}
    for i in range(60):
        v = (1.0 if i < 20 else 3.0 if i < 40 else 6.0) + rng.normal(0, 0.02)
        bin_curves[i] = _const_func(v)
        true_val[i] = v

    b2b, bcurves = bucket_lsf_curves(bin_curves, wave_grid, max_buckets=6)
    assert len(set(b2b.values())) <= 6

    from collections import defaultdict
    by_bucket = defaultdict(list)
    for i, b in b2b.items():
        by_bucket[b].append(true_val[i])
    max_spread = max(max(vals) - min(vals) for vals in by_bucket.values())
    assert max_spread < 0.5, "a bucket must never mix genuinely different-resolution bins"

    for b, vals in by_bucket.items():
        assert abs(bcurves[b][0] - np.mean(vals)) < 0.05


# --------------------------------------------------------------------------- #
# bucket_lsf_for_voronoi_bins: full APS_ID-keyed convenience wrapper
# --------------------------------------------------------------------------- #

def test_bucket_lsf_for_voronoi_bins_composition_and_exclusion():
    wave_grid = np.linspace(4000, 5000, 11)
    aps_id_to_fwhm = {1: _const_func(1.0), 2: _const_func(3.0), 3: _const_func(5.0),
                       4: _const_func(5.0), 5: _const_func(99.0), 6: _const_func(2.0)}
    aps_id_to_flux = {i: 1.0 for i in range(1, 7)}
    aps_id_to_bin = {1: 0, 2: 0, 3: 1, 4: 1, 5: None, 6: 1}  # 5 excluded

    bin_to_bucket, lsf_by_bucket = bucket_lsf_for_voronoi_bins(
        aps_id_to_fwhm, aps_id_to_flux, aps_id_to_bin, n_bins=2,
        wave_grid=wave_grid, max_buckets=6)

    assert bin_to_bucket.shape == (2,)
    v0 = lsf_by_bucket[bin_to_bucket[0]](4500.0)
    v1 = lsf_by_bucket[bin_to_bucket[1]](4500.0)
    assert np.isclose(v0, 2.0)   # avg(1,3)
    assert np.isclose(v1, 4.0)   # avg(5,5,2)


def test_bucket_lsf_for_voronoi_bins_empty_and_missing_bin_fallback():
    wave_grid = np.linspace(4000, 5000, 11)
    b2b, lsf = bucket_lsf_for_voronoi_bins({}, {}, {}, 3, wave_grid)
    assert b2b.shape == (3,) and (b2b == 0).all() and lsf == {}

    # bin id 1 never appears among the spaxels -- must still get a safe,
    # real fallback bucket, not an out-of-range index or crash.
    b2b2, lsf2 = bucket_lsf_for_voronoi_bins(
        {1: _const_func(1.0), 2: _const_func(1.0)}, {1: 1.0, 2: 1.0}, {1: 0, 2: 0}, 3, wave_grid)
    assert b2b2.shape == (3,)
    assert all(b in lsf2 for b in b2b2)


# --------------------------------------------------------------------------- #
# PowerBin engine swap
# --------------------------------------------------------------------------- #

def test_powerbin_define_voronoi_bins_synthetic():
    """define_voronoi_bins (VORONOI=1) runs end to end on synthetic data,
    producing a sane (not degenerate single-bin, not one-bin-per-pixel)
    number of PowerBin bins, matching every downstream consumer's own
    expectations (binNum in [0, n_bins), same array length as input)."""
    rng = np.random.default_rng(42)
    n = 2000
    x = rng.uniform(-20, 20, n)
    y = rng.uniform(-20, 20, n)
    r = np.sqrt(x**2 + y**2)
    signal = 100 * np.exp(-r / 8.0) + 0.5
    noise = np.sqrt(signal) * 2 + 0.3
    snr = signal / noise
    idx_inside = np.arange(n)
    idx_outside = np.array([], dtype=int)

    binNum = ExGalPrepare.define_voronoi_bins(
        1, list(range(n)), [f"t{i}" for i in range(n)], [f"c{i}" for i in range(n)],
        x, y, np.zeros(n), np.zeros(n), np.zeros(n, dtype=np.int64),
        np.zeros(n), np.zeros(n), signal, noise, 1.0, snr,
        target_snr=20.0, covar_vor=0.0,
        idx_inside=idx_inside, idx_outside=idx_outside,
        rootname="pytest_powerbin", outdir="/tmp/",
        configs={"CONFIG_FILE": "test.json", "infiles": ["a.fit"]},
    )
    assert len(binNum) == n
    n_bins = len(np.unique(binNum))
    assert 1 < n_bins < n, f"expected a real, non-degenerate binning, got {n_bins} bins from {n} pixels"
    assert binNum.min() == 0 and binNum.max() == n_bins - 1


def test_powerbin_no_binning_path_unaffected():
    """VORONOI=0 (each spaxel its own bin) never touches the binning
    engine at all -- confirms the swap didn't change this path."""
    n = 50
    x = np.arange(n, dtype=float)
    y = np.zeros(n)
    signal = np.ones(n)
    noise = np.ones(n)
    snr = signal / noise
    idx_inside = np.arange(n)
    idx_outside = np.array([], dtype=int)

    binNum = ExGalPrepare.define_voronoi_bins(
        0, list(range(n)), [f"t{i}" for i in range(n)], [f"c{i}" for i in range(n)],
        x, y, np.zeros(n), np.zeros(n), np.zeros(n, dtype=np.int64),
        np.zeros(n), np.zeros(n), signal, noise, 1.0, snr,
        target_snr=20.0, covar_vor=0.0,
        idx_inside=idx_inside, idx_outside=idx_outside,
        rootname="pytest_nobinning", outdir="/tmp/",
        configs={"CONFIG_FILE": "test.json", "infiles": ["a.fit"]},
    )
    assert len(binNum) == n
    assert len(np.unique(binNum)) == n  # every spaxel is its own bin


# --------------------------------------------------------------------------- #
# Real-data end-to-end: ExGal prep + PPXF bucket threading
# --------------------------------------------------------------------------- #

def _extract_real_templates(tmp_path):
    import tarfile
    dest = tmp_path / "PyAPS_templates"
    dest.mkdir()
    with tarfile.open(REAL_TEMPLATES_TARBALL) as tf:
        member = "PyAPS_templates/templates_GIST/popstar_total_PCA.npy"
        tf.extract(member, path=tmp_path, filter="data")
    return str(tmp_path / "PyAPS_templates" / "templates_GIST")


def _writable_configs(tmp_path):
    import shutil
    dest = tmp_path / "configs_writable"
    shutil.copytree(REAL_IFU_CONFIG_DIR, dest / "ExGal_configs")
    return str(dest / "ExGal_configs" / "LIFULR11.json"), str(dest / "ExGal_configs")


def test_real_exgal_prep_bucket_mechanism(tmp_path):
    _skip_unless_exists(*REAL_INFILES, REAL_CALDIR, REAL_CATDIR,
                        REAL_IFU_PARAMS, REAL_TEMPLATES_TARBALL)
    from astropy.io import fits
    from PyAPS.aps_ifu_ExGal import ifu_ExGal_prepare

    ifu_params, ifu_config_dir = _writable_configs(tmp_path)
    with fits.open(REAL_INFILES[0]) as h:
        ra0, dec0 = h[1].header["CRVAL1"], h[1].header["CRVAL2"]
    area = [ra0, dec0, 6.0, 6.0, 0.0]
    z_input = [0.02, 0.001]

    def prep(spaxel_weighted_lsf, label):
        outpath = tmp_path / f"exgal_{label}"
        outpath.mkdir()
        return ifu_ExGal_prepare(
            REAL_INFILES, f"pytest_{label}", ifu_params, str(outpath) + "/", ifu_config_dir,
            area=area, z_input=z_input, join_arms=True,
            catdir=REAL_CATDIR, caldir=REAL_CALDIR,
            spaxel_weighted_lsf=spaxel_weighted_lsf,
        )

    prep_off = prep(False, "off")
    prep_on = prep(True, "on")
    assert prep_off is not None and prep_on is not None

    assert prep_off["bin_to_bucket"] is None
    assert prep_off["LSF_Data_by_bucket"] is None
    assert prep_on["bin_to_bucket"] is not None
    assert prep_on["LSF_Data_by_bucket"]

    wave_mid = prep_off["cube"]["wave"][len(prep_off["cube"]["wave"]) // 2]
    v_global = prep_off["LSF_Data"](wave_mid)
    bucket_vals = [f(wave_mid) for f in prep_on["LSF_Data_by_bucket"].values()]
    assert all(np.isfinite(v) and v > 0 for v in bucket_vals)
    # Real per-bucket variation must exist (not silently collapsed to the
    # global value for every bucket) on this real dataset.
    assert max(bucket_vals) - min(bucket_vals) > 0


def test_real_exgal_ppxf_bucket_threading(tmp_path):
    """Full real-data run: PPXF fit through the bucketed template path
    must complete and produce physically sane (finite, non-degenerate)
    kinematics, matching in shape (not value -- different resolution
    correction is the whole point) the un-bucketed baseline."""
    _skip_unless_exists(*REAL_INFILES, REAL_CALDIR, REAL_CATDIR,
                        REAL_IFU_PARAMS, REAL_TEMPLATES_TARBALL)
    from astropy.io import fits
    from PyAPS.aps_ifu_ExGal import ifu_ExGal_prepare
    from PyAPS import IFUExGalPPXF

    ifu_params, ifu_config_dir = _writable_configs(tmp_path)
    templates_dir = _extract_real_templates(tmp_path)
    with fits.open(REAL_INFILES[0]) as h:
        ra0, dec0 = h[1].header["CRVAL1"], h[1].header["CRVAL2"]
    area = [ra0, dec0, 6.0, 6.0, 0.0]
    z_input = [0.02, 0.001]

    def run(spaxel_weighted_lsf, label):
        outpath = tmp_path / f"ppxf_{label}"
        outpath.mkdir()
        prep = ifu_ExGal_prepare(
            REAL_INFILES, f"pytest_ppxf_{label}", ifu_params, str(outpath) + "/", ifu_config_dir,
            area=area, z_input=z_input, join_arms=True,
            catdir=REAL_CATDIR, caldir=REAL_CALDIR,
            spaxel_weighted_lsf=spaxel_weighted_lsf,
        )
        assert prep is not None
        cube, configs = prep["cube"], prep["configs"]
        return IFUExGalPPXF.runModule_PPXF(
            1, configs, cube["velscale"], prep["LSF_Data"], prep["LSF_Templates"],
            str(outpath) + "/", ifu_config_dir, templates_dir,
            str(outpath) + "/", f"pytest_ppxf_{label}",
            z_input[0], z_input[1], 1.0e18, debug=False, output=True,
            bin_to_bucket=prep["bin_to_bucket"], LSF_Data_by_bucket=prep["LSF_Data_by_bucket"],
        )

    res_off = run(False, "off")
    res_on = run(True, "on")
    assert res_off is not None and res_on is not None

    ppxf_off, ppxf_on = res_off[0], res_on[0]
    assert ppxf_off.shape == ppxf_on.shape
    assert np.sum(~np.isnan(ppxf_on[:, 0])) > 0, "bucketed PPXF must fit at least some bins"
