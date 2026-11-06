"""Line-strength broadening: the 'gaussian_filter1d() got an unexpected keyword argument truncate' regression.

IFUExGalLS / MOSExGalLS called ``ppxf.ppxf_util.gaussian_filter1d(spec, sigma, truncate=..., mode=...)``.
pPXF's function has the signature ``(spec, sig)``, so the call raised TypeError, the ``except`` printed
WARNING "Convolution failed" and the spectrum stayed unbroadened (FWHM_FLAG = 3). The modules now use
``aps_ls_broaden.broaden_variable_sigma``.
"""
import ast
import inspect
import pathlib

import numpy as np
import pytest

from PyAPS.aps_ls_broaden import broaden_variable_sigma

PKG = pathlib.Path(inspect.getfile(broaden_variable_sigma)).parent
FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))


def _fwhm(x, y):
    y = y - y.min()
    half = y.max() / 2.0
    above = np.where(y >= half)[0]
    lo, hi = above[0], above[-1]
    xl = np.interp(half, [y[lo - 1], y[lo]], [x[lo - 1], x[lo]])
    xr = np.interp(half, [y[hi + 1], y[hi]], [x[hi + 1], x[hi]])
    return xr - xl


def test_constant_sigma_matches_scipy():
    ndimage = pytest.importorskip("scipy.ndimage")
    rng = np.random.default_rng(1)
    spec = rng.normal(10.0, 1.0, 600)
    for sigma in (2.0, 3.0, 5.0):                      # 4*sigma is an integer: same kernel support
        ours = broaden_variable_sigma(spec, np.full(spec.size, sigma), truncate=4.0, mode='nearest')
        ref = ndimage.gaussian_filter1d(spec, sigma, truncate=4.0, mode='nearest')
        assert np.allclose(ours, ref, rtol=0, atol=1e-10)


def test_flux_and_constant_are_conserved():
    flat = np.full(300, 7.0)
    sig = np.linspace(0.0, 30.0, 300)
    assert np.allclose(broaden_variable_sigma(flat, sig), 7.0)         # 'nearest' edges: no droop
    line = np.zeros(2000)
    line[1000] = 1.0
    out = broaden_variable_sigma(line, np.full(2000, 8.0))
    assert out.sum() == pytest.approx(1.0, abs=1e-12)


def test_width_follows_sigma_per_pixel():
    n = 4000
    for sigma in (1.5, 6.0, 25.0, 90.0):               # includes kernels that used the 3 / 2 sigma tiers
        spec = np.zeros(n)
        spec[2000] = 1.0
        sig = np.full(n, sigma)
        out = broaden_variable_sigma(spec, sig, truncate=4.0)
        x = np.arange(n, dtype=float)
        assert _fwhm(x, out) == pytest.approx(FWHM * sigma, rel=0.01)


def test_variable_sigma_two_lines():
    n = 3000
    spec = np.zeros(n)
    spec[600] = spec[2400] = 1.0
    sig = np.where(np.arange(n) < 1500, 3.0, 12.0)
    out = broaden_variable_sigma(spec, sig)
    x = np.arange(n, dtype=float)
    assert _fwhm(x[:1500], out[:1500]) == pytest.approx(FWHM * 3.0, rel=0.02)
    assert _fwhm(x[1500:], out[1500:]) == pytest.approx(FWHM * 12.0, rel=0.02)


def test_zero_sigma_is_identity_and_edge_modes():
    spec = np.arange(50, dtype=float)
    assert np.array_equal(broaden_variable_sigma(spec, np.zeros(50)), spec)
    assert np.array_equal(broaden_variable_sigma(spec, 0.0), spec)
    edge = broaden_variable_sigma(np.ones(200), np.full(200, 5.0), mode='constant')
    assert edge[0] < 0.7 and edge[100] == pytest.approx(1.0, abs=1e-6)  # zero padding droops at the end
    with pytest.raises(ValueError):
        broaden_variable_sigma(spec, np.ones(50), mode='wrap')


def test_nan_stays_local():
    spec = np.ones(500)
    spec[100] = np.nan
    out = broaden_variable_sigma(spec, np.full(500, 2.0), truncate=4.0)
    bad = np.where(~np.isfinite(out))[0]
    assert bad.min() == 100 - 8 and bad.max() == 100 + 8
    assert np.all(np.isfinite(out[:90])) and np.all(np.isfinite(out[110:]))


def test_long_spectrum_with_wide_kernel_is_fast_enough_and_chunked():
    n = 12000
    spec = np.random.default_rng(0).normal(1.0, 0.1, n)
    out = broaden_variable_sigma(spec, np.linspace(0.0, 60.0, n))
    assert out.shape == (n,) and np.all(np.isfinite(out))


@pytest.mark.parametrize("module", ["IFUExGalLS.py", "MOSExGalLS.py"])
def test_modules_call_the_helper_with_arguments_it_accepts(module):
    """The call that failed on the pod: every keyword passed to the helper must exist in its signature,
    and the pPXF function (signature (spec, sig)) must not be used for the LS broadening again."""
    src = (PKG / module).read_text()
    assert "from ppxf.ppxf_util import gaussian_filter1d" not in src
    tree = ast.parse(src)
    accepted = set(inspect.signature(broaden_variable_sigma).parameters)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) == "broaden_variable_sigma"]
    assert len(calls) == 2                              # spectrum and error spectrum
    for c in calls:
        assert {k.arg for k in c.keywords} <= accepted
        assert len(c.args) == 2
    assert not any(isinstance(n, ast.Call) and getattr(n.func, "id", None) == "gaussian_filter1d"
                   for n in ast.walk(tree))


def test_the_ppxf_function_really_has_no_truncate():
    """Documents why the old call failed (skipped if pPXF is not installed or changes this)."""
    util = pytest.importorskip("ppxf.ppxf_util")
    params = inspect.signature(util.gaussian_filter1d).parameters
    if "truncate" in params:
        pytest.skip("this pPXF accepts truncate")
    with pytest.raises(TypeError, match="truncate"):
        util.gaussian_filter1d(np.ones(10), np.ones(10), truncate=4.0, mode='nearest')
