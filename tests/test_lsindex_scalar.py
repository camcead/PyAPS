"""lsindex_spec on NumPy 2: a band edge that falls inside a pixel must give a scalar index.

Before: sum_counts() added ``c_clean[pixb] * fracb`` (a 1-element array) to the sum, so every index
became a (1,) array and ``index[k] = index0[0]`` raised "setting an array element with a sequence"
on NumPy 2.5. IFUExGalLS / MOSExGalLS caught it ("lsindex measurement failed") and wrote NaN for
EVERY line-strength index of the run (OB 26133: 0 finite values out of 59,000).
"""
import numpy as np
import pytest

lsindex_spec = pytest.importorskip("PyAPS.lsindex_spec")


def _grid():
    wave = np.arange(4000.0, 5600.0, 0.3356)             # band edges fall inside pixels
    flux = 1.0 + 0.0 * wave
    flux -= 0.3 * np.exp(-0.5 * ((wave - 5175.0) / 4.0) ** 2)   # an absorption line
    return wave, flux


def test_sum_counts_is_a_scalar_when_the_band_edge_is_inside_a_pixel():
    wave, flux = _grid()
    s = lsindex_spec.sum_counts(wave, flux, 5142.6, 5161.4)
    assert np.ndim(s) == 0
    assert np.isfinite(s)
    assert s == pytest.approx(flux[(wave > 5142.6) & (wave < 5161.4)].sum() / 1.0, rel=0.01)


def test_lsindex_gives_finite_values_and_errors(tmp_path):
    wave, flux = _grid()
    lick = tmp_path / "bands.config"
    lick.write_text(
        "# b1 b2 b3 b4 b5 b6 b7 names spp origin\n"
        "5142.625 5161.375 5160.125 5192.625 5191.375 5206.375 1 Mgb 1 test\n"
        "4897.375 4958.250 4958.250 5004.000 5004.000 5065.125 1 Fe5015like 1 test\n")
    names, idx, err = lsindex_spec.lsindex(wave, flux.copy(), 0.01 + 0 * flux, 0.0, str(lick), sims=4, z_err=1e-4)
    idx = np.real(np.asarray(idx))
    assert list(names) == ["Mgb", "Fe5015like"]
    assert np.all(np.isfinite(idx))
    assert idx[0] > 0.5            # the absorption line is detected as a positive atomic index
