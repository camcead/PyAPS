"""FITS arrays are made native once, where they are read (aps_native), so that a serial loop and a pool worker fit
the same bytes. IFU_ExGal `--mp_ExGal 1` against the pool differed in the last bin because of this (2026-11-15)."""
import pickle
import re
from pathlib import Path

import numpy as np

from PyAPS import aps_native

PYAPS = Path(__file__).resolve().parents[1] / "py" / "PyAPS"


def _big_endian_table(npix=1000, nbins=4, seed=3):
    rng = np.random.default_rng(seed)
    native = rng.normal(1.0, 0.1, (nbins, npix))
    return native, native.astype(">f8")          # what astropy returns from a FITS binary table


def test_native_array_converts_value_for_value():
    native, be = _big_endian_table()
    out = aps_native.native_array(be)
    assert out.dtype.isnative and out.dtype == np.float64
    assert np.array_equal(out, native)


def test_native_input_is_unchanged_and_still_copied():
    native, _ = _big_endian_table()
    out = aps_native.native_array(native)
    assert out.dtype == native.dtype and np.array_equal(out, native)
    assert not np.shares_memory(out, native)       # replaces np.array(): never a view of the memory-mapped file


def test_memory_order_is_kept():
    # SPEC.T is a Fortran-ordered view; the code downstream was written for the layout np.array() gave it
    _, be = _big_endian_table()
    for src in (be.T, np.ascontiguousarray(be.T)):
        assert aps_native.native_array(src).flags['F_CONTIGUOUS'] == np.array(src).flags['F_CONTIGUOUS']
        assert aps_native.native_array(src).strides == np.array(src).strides


def test_is_native():
    native, be = _big_endian_table()
    assert aps_native.is_native(native) and not aps_native.is_native(be)
    assert aps_native.is_native(aps_native.native_array(be))
    assert not aps_native.is_native([1.0, 2.0])


def test_serial_column_is_what_a_pool_worker_receives():
    """A multiprocessing queue pickles each job: the worker sees native float64 whatever the parent held. After
    native_array() the serial loop sees the same dtype and the same bits."""
    _, be = _big_endian_table()
    served = aps_native.native_array(be.T)                       # as the stage reads it
    for i in range(be.shape[0]):
        in_serial = served[:, i]
        in_worker = pickle.loads(pickle.dumps(in_serial))
        assert in_serial.dtype == in_worker.dtype == np.float64
        assert in_serial.tobytes() == in_worker.tobytes()


def test_reductions_agree_with_the_pool_after_the_helper():
    """The failure mode itself: sums of more than 8192 big-endian values need not equal the native ones. The helper
    makes the question moot; this checks the equality that the stages rely on, on data long enough to hit the
    chunked path (it does not assert that the big-endian sum differs: that depends on numpy and the data)."""
    _, be = _big_endian_table(npix=10711, nbins=21, seed=11)
    served = aps_native.native_array(be.T)
    for i in range(be.shape[0]):
        col = served[:, i]
        worker = pickle.loads(pickle.dumps(col))
        assert np.sum(col) == np.sum(worker)
        assert np.std(col) == np.std(worker)


def test_ifu_exgal_stages_read_their_fit_inputs_through_the_helper():
    """Guard against a later edit going back to np.array(hdu[1].data.SPEC...) in the stages that fit the bins."""
    for name in ("IFUExGalPPXF.py", "IFUExGalEMIPPXF.py"):
        src = (PYAPS / name).read_text()
        assert "from PyAPS import aps_native" in src, name
        assert not re.search(r"np\.array\(\s*hdu\[1\]\.data\.(SPEC|ESPEC)", src), name
        assert re.search(r"aps_native\.native_array\(\s*hdu\[1\]\.data\.SPEC\.T", src), name
        assert re.search(r"aps_native\.native_array\(\s*hdu\[1\]\.data\.ESPEC\.T", src), name
