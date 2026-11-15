"""FESWI reads FEH / FEH_ERR; aps_ferre writes MH / MH_ERR (FERRE wrapper 2.1).

aps_feswi.feswi_param_file hands FESWI a copy of ferre_<head>.fits with FEH / FEH_ERR added from MH / MH_ERR, under the
same base name (FESWI names its outputs from it); the original file keeps its names.

Run with:
    cd <PYAPS_DIR> && pytest tests/test_feswi_mh_compat.py -v
"""
from __future__ import annotations

import inspect
import os

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from PyAPS import aps_feswi


def _write(path, names=("MH", "MH_ERR"), n=4):
    t = Table()
    t["APS_ID"] = np.arange(1, n + 1)
    t["TEFF"] = np.full(n, 5500.0)
    for k, c in enumerate(names):
        t[c] = np.linspace(-2.0, 0.2, n) + 0.01 * k
        t[c].unit = "mag"
    hdu = fits.table_to_hdu(t)
    hdu.name = "FR_TABLE"
    hdu.header["APS_FR_V"] = "2.1"
    fits.HDUList([fits.PrimaryHDU(), hdu, fits.ImageHDU(np.arange(6.0), name="EXTRA")]).writeto(path, overwrite=True)


def test_mh_file_gets_feh_aliases_in_a_copy_with_the_same_name(tmp_path):
    src = tmp_path / "ferre_pixel.fits"
    _write(src)
    work = tmp_path / "w"
    work.mkdir()
    out = aps_feswi.feswi_param_file(str(src), str(work))
    assert out == str(work / "ferre_pixel.fits") and os.path.basename(out) == src.name
    with fits.open(out) as h, fits.open(src) as o:
        d, d0 = h[1].data, o[1].data
        assert {"MH", "MH_ERR", "FEH", "FEH_ERR"} <= set(h[1].columns.names)
        assert np.array_equal(d["FEH"], d0["MH"]) and np.array_equal(d["FEH_ERR"], d0["MH_ERR"])
        assert h[1].columns["FEH"].unit == "mag" and h[1].header["APS_FR_V"] == "2.1" and h[1].name == "FR_TABLE"
        assert len(h) == 3 and h[2].name == "EXTRA"
        assert "FEH" not in o[1].columns.names                    # the original is not touched


def test_file_with_feh_or_without_the_table_is_returned_as_it_is(tmp_path):
    old = tmp_path / "old.fits"
    _write(old, names=("FEH", "FEH_ERR"))
    assert aps_feswi.feswi_param_file(str(old), str(tmp_path)) == str(old)
    bare = tmp_path / "bare.fits"
    fits.HDUList([fits.PrimaryHDU()]).writeto(bare)
    assert aps_feswi.feswi_param_file(str(bare), str(tmp_path)) == str(bare)


def test_only_the_missing_alias_is_added(tmp_path):
    src = tmp_path / "half.fits"
    _write(src, names=("MH",))
    work = tmp_path / "w"
    work.mkdir()
    out = aps_feswi.feswi_param_file(str(src), str(work))
    with fits.open(out) as h:
        assert "FEH" in h[1].columns.names and "FEH_ERR" not in h[1].columns.names


def test_the_feswi_call_uses_the_compatible_file_and_removes_it():
    src = inspect.getsource(aps_feswi)
    i = src.rindex("running FESWI'")
    block = src[i:i + 900]
    assert "feswi_param_file(proc_args.param_fits" in block and "_call_feswi(feswi, feswi_param" in block and "shutil.rmtree(compat_dir" in block
