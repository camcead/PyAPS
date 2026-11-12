"""FERRE metallicity columns are called MH / MH_ERR (FERRE wrapper version 2.1, 2026-11-12).

The FERRE grid dimension METALS is the overall metallicity [M/H], not [Fe/H], so ``aps_ferre`` writes ``MH`` / ``MH_ERR``
(FR_TABLE of ferre_<head>.fits and, through aps_L2merge, the L2 STAR_TABLE) instead of ``FEH`` / ``FEH_ERR``. Readers of
L2 files (aps_MOSviewer, aps_rrlew, aps_l1_multi_plot) still accept ``FEH`` from files written before the change.

Run with:
    cd <PYAPS_DIR> && pytest tests/test_ferre_mh_columns.py -v
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from PyAPS import aps_constants
from PyAPS import aps_ferre
from PyAPS.apsPlot import ferre as fr_plot

PARAMS = ["TEFF", "TEFF_ERR", "LOGG", "LOGG_ERR", "MH", "MH_ERR", "ALPHA", "ALPHA_ERR", "MICRO", "MICRO_ERR"]
NPIX = 8


def _outdict(n=2):
    """A small table shaped like the one ferre_outdict returns (5-D grid, one arm 'B')."""
    t = Table()
    t["APS_ID"] = np.arange(1, n + 1)
    t["TARGID"] = [f"T{i}" for i in range(n)]
    t["CNAME"] = [f"WVE_{i:08d}" for i in range(n)]
    for k, c in enumerate(PARAMS):
        t[c] = np.full(n, 0.1 * (k + 1))
    t["COVAR"] = np.zeros((n, 5, 5))
    t["ELEM"] = np.full((n, 2), np.nan)
    t["ELEM_ERR"] = np.full((n, 2), np.nan)
    t["SNR_FR"] = np.full(n, 30.0)
    t["CHISQ_TOT"] = np.full(n, 1.2)
    t["FLAG_FR"] = np.ones(n, dtype=int)
    for c in ("LAMBDA_FR_B", "FLUX_FR_B", "ERROR_FR_B", "MODEL_FR_B"):
        t[c] = np.tile(np.linspace(4000.0, 4010.0, NPIX), (n, 1))
    for c in ("MH", "MH_ERR", "TEFF", "TEFF_ERR", "LOGG", "LOGG_ERR", "ALPHA", "ALPHA_ERR", "MICRO", "MICRO_ERR"):
        t[c].unit = "K" if c.startswith("TEFF") else "dex"
    return t


def test_ferre_wrapper_version_is_2_1():
    assert float(aps_constants.__aps_ferre_version__) == pytest.approx(2.1)


def test_aps_ferre_has_no_feh_column_names():
    src = Path(aps_ferre.__file__).read_text()
    assert not re.search(r"['\"]FEH(_ERR)?['\"]", src)
    assert "'MH'" in src and "'MH_ERR'" in src


def test_ferre_outdict_declares_mh_columns():
    """The output-table structure of ferre_outdict lists MH / MH_ERR and gives them the unit dex."""
    src = inspect.getsource(aps_ferre.ferre_outdict)
    assert "'MH', 'MH_ERR'" in src
    assert "outdict['MH'].unit = 'dex'" in src and "outdict['MH_ERR'].unit = 'dex'" in src
    assert src.count("outdict['MH'].append") == 3 and src.count("outdict['MH_ERR'].append") == 3


def test_fr_table_header_has_mh_columns(tmp_path):
    param_file = tmp_path / "ferre_test.fits"
    spec_file = tmp_path / "ferre_spec_test.fits"
    t = _outdict()
    aps_ferre.ferre_write_fits(t, ["single_r0000001.fit"], ["BLUE"], False,
                               {"wave": "Angstrom", "flux": "normalised"}, str(param_file), spec_file=str(spec_file))

    with fits.open(param_file) as h:
        hdr = h[1].header
        names = h[1].columns.names
        assert hdr["EXTNAME"] == "FR_TABLE"
        assert "MH" in names and "MH_ERR" in names
        assert "FEH" not in names and "FEH_ERR" not in names
        # same position as FEH / FEH_ERR had: after LOGG_ERR, before ALPHA
        assert names[names.index("LOGG_ERR") + 1:names.index("ALPHA")] == ["MH", "MH_ERR"]
        i = names.index("MH") + 1
        assert hdr[f"TTYPE{i}"] == "MH" and hdr[f"TTYPE{i + 1}"] == "MH_ERR"
        # astropy cannot write the unit 'dex' as a FITS TUNIT (already so for LOGG and the former FEH);
        # MH / MH_ERR must be written exactly like LOGG
        j = names.index("LOGG") + 1
        assert hdr.get(f"TUNIT{i}") == hdr.get(f"TUNIT{j}")
        assert hdr.get(f"TUNIT{i + 1}") == hdr.get(f"TUNIT{j + 1}")
        assert float(hdr["APS_FR_V"]) == pytest.approx(2.1)
        np.testing.assert_allclose(h[1].data["MH"], 0.5)

    with fits.open(spec_file) as h:
        names = h[1].columns.names
        assert not ({"MH", "MH_ERR", "FEH", "FEH_ERR"} & set(names))
        assert "FLUX_FR_B" in names


def _figure_title(fig):
    return fig.layout.title.text if fig.layout.title and fig.layout.title.text else fig.to_json()


def test_plot_title_uses_mh():
    t = _outdict(1)
    fig = fr_plot.build_figure({c: list(t[c]) for c in t.colnames}, 0, ["BLUE"])
    assert "MH=0.500" in _figure_title(fig)


def _star_tables(mh_name):
    """STAR_TABLE / STAR_SPEC FITS_recs as aps_MOSviewer loads them; mh_name 'MH' (new) or 'FEH' (old files)."""
    cols = [fits.Column(name="APS_ID", format="K", array=np.array([7])),
            fits.Column(name="TEFF", format="D", array=np.array([5000.0])),
            fits.Column(name="LOGG", format="D", array=np.array([2.5])),
            fits.Column(name=mh_name, format="D", array=np.array([-0.75])),
            fits.Column(name=mh_name + "_ERR", format="D", array=np.array([0.05])),
            fits.Column(name="ALPHA", format="D", array=np.array([0.1])),
            fits.Column(name="MICRO", format="D", array=np.array([0.2]))]
    star = fits.BinTableHDU.from_columns(cols).data
    lam = np.linspace(4000.0, 4010.0, NPIX)[None, :]
    spec = fits.BinTableHDU.from_columns(
        [fits.Column(name="APS_ID", format="K", array=np.array([7]))]
        + [fits.Column(name=f"{c}_FR_B", format=f"{NPIX}D", array=lam) for c in ("LAMBDA", "FLUX", "MODEL")]).data
    return {"star_table": star, "star_spec": spec, "star_pos": {7: 0}, "star_spec_pos": {7: 0}, "arms": ["B"]}


@pytest.mark.parametrize("mh_name", ["MH", "FEH"])
def test_mosviewer_ferre_figure_reads_new_and_old_files(monkeypatch, mh_name):
    pytest.importorskip("dash")
    from PyAPS import aps_MOSviewer as mos_mod

    monkeypatch.setattr(mos_mod, "STATE", SimpleNamespace(data=_star_tables(mh_name)))
    fig = mos_mod._ferre_figure(7)
    assert "MH=-0.750" in _figure_title(fig)


@pytest.mark.parametrize("module", ["aps_rrlew", "aps_l1_multi_plot"])
def test_l2_readers_fall_back_to_feh(module):
    """aps_rrlew / aps_l1_multi_plot pick MH when the L2 file has it and FEH otherwise (no hard-coded 'FEH' read)."""
    src = (Path(aps_ferre.__file__).parent / f"{module}.py").read_text()
    assert "'MH' if 'MH' in" in src
    assert not re.search(r"\[\s*'FEH(_ERR)?'\s*\]", src)
