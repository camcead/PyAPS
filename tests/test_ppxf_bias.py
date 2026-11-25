"""BIAS_PPXF: the penalisation of the stellar-kinematics pPXF fit can be chosen in the config files (IFU and MOS ExGal).

Absent / None / "None" gives pPXF's own default (what was always used), a number >= 0 goes to the main fit only; the Monte-Carlo error
simulations keep bias=0. The pPXF call is replaced by a recorder, so no templates or data are needed.
"""
import numpy as np
import pytest

from PyAPS import ExGalutil


@pytest.mark.parametrize("value", [None, "None", "none", "default", "", " null "])
def test_default_values_mean_the_ppxf_default(value):
    assert ExGalutil.ppxf_bias({"BIAS_PPXF": value}) is None


def test_missing_key_is_the_ppxf_default():
    assert ExGalutil.ppxf_bias({}) is None
    assert ExGalutil.ppxf_bias({"MOM": 4}) is None


@pytest.mark.parametrize("value,expected", [(0.2, 0.2), (0, 0.0), (1, 1.0), ("0.3", 0.3), (np.float64(0.25), 0.25)])
def test_numbers_are_accepted(value, expected):
    out = ExGalutil.ppxf_bias({"BIAS_PPXF": value})
    assert out == expected and isinstance(out, float)


@pytest.mark.parametrize("value", [-0.1, "abc", float("nan"), True, [0.2], {"a": 1}])
def test_bad_values_raise(value):
    with pytest.raises(ValueError, match="BIAS_PPXF"):
        ExGalutil.ppxf_bias({"BIAS_PPXF": value})


def _recorder(calls):
    class FakePPXF:
        def __init__(self, templates, galaxy, noise, velscale, start, **kw):
            calls.append(kw)
            self.weights = np.ones(templates.shape[1])
            self.error = np.ones(kw["moments"])
            self.chi2 = 1.0
            self.bestfit = np.array(galaxy, dtype=float)
            self.sol = np.zeros(kw["moments"])
    return FakePPXF


def _inputs(npix=60):
    rng = np.random.default_rng(1)
    return dict(templates=rng.normal(size=(npix, 3)) + 5, data=rng.normal(size=npix) + 5, error=np.ones(npix),
                velscale=50.0, start=np.array([0.0, 100.0]), goodpixels=np.arange(npix))


def test_ifu_stellar_fit_passes_the_bias_to_the_main_fit_only(monkeypatch):
    pytest.importorskip("ppxf")
    from PyAPS import IFUExGalPPXF as mod

    calls = []
    monkeypatch.setattr(mod, "ppxf", _recorder(calls))
    x = _inputs()
    lam = np.linspace(4000, 5000, 60)
    for bias in (None, 0.0, 0.3):
        calls.clear()
        out = mod.run_ppxf(x["templates"], x["data"], x["error"], x["velscale"], x["start"], x["goodpixels"], 1e12, 2, -1, 0, 0.0, 1,
                           2, 1, lam, lam, 0, bias=bias)
        assert not np.isnan(out[0]).any()                      # the fit ran (no exception swallowed into NaN)
        assert len(calls) == 3                                 # main fit + 2 Monte-Carlo
        assert calls[0]["bias"] == bias
        assert [c["bias"] for c in calls[1:]] == [0.0, 0.0]    # the MC simulations keep bias=0


def test_ifu_default_call_is_unchanged(monkeypatch):
    pytest.importorskip("ppxf")
    from PyAPS import IFUExGalPPXF as mod

    calls = []
    monkeypatch.setattr(mod, "ppxf", _recorder(calls))
    x = _inputs()
    lam = np.linspace(4000, 5000, 60)
    mod.run_ppxf(x["templates"], x["data"], x["error"], x["velscale"], x["start"], x["goodpixels"], 1e12, 2, -1, 0, 0.0, 1, 0, 1, lam, lam, 0)
    assert calls[0]["bias"] is None                            # pPXF's own default, as before the key existed


def test_mos_stellar_fit_passes_the_bias_to_the_main_fit_only(monkeypatch):
    pytest.importorskip("ppxf")
    from PyAPS import MOSExGalPPXF as mod

    calls = []
    monkeypatch.setattr(mod, "ppxf", _recorder(calls))
    x = _inputs()
    loglam = np.log(np.linspace(4000, 5000, 60))
    for bias in (None, 0.25):
        calls.clear()
        out = mod.run_ppxf(x["templates"], x["data"], x["error"], x["velscale"], x["start"], x["goodpixels"], 2, -1, 0, 0.0, 1.0, 2, 1,
                           loglam, loglam, False, 0, bias=bias)
        assert len(calls) == 3
        assert calls[0]["bias"] == bias
        assert [c["bias"] for c in calls[1:]] == [0.0, 0.0]
        assert not np.isnan(out[0]).any()
