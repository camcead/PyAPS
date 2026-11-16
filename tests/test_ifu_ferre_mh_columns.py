"""IFU Galactic mode: the FERRE metallicity columns are called MH_FERRE / MH_ERR_FERRE (aps_ifu_ferre 1.2, 2026-11-16).

Same rule as the MOS rename of 2026-11-12 (aps_ferre 2.1, tests/test_ferre_mh_columns.py): the FERRE grid dimension METALS is the
overall metallicity [M/H], not [Fe/H]. `aps_ifu_ferre` writes MH_FERRE / MH_ERR_FERRE (FERRE_IFU_TABLE and, through
aps_L2merge.ifuGalL2merge, the IFU Gal L2 STAR_TABLE); the viewer still reads FEH_FERRE from files written before.

Run with:
    cd <PYAPS_DIR> && pytest tests/test_ifu_ferre_mh_columns.py -v
"""
import re
from pathlib import Path

import numpy as np
import pytest

from PyAPS import aps_constants
from PyAPS import aps_ifu_ferre as F

SRC = Path(F.__file__).read_text()


def _opf(tmp_path, cells):
    (tmp_path / "bin0000.opf").write_text(" ".join(str(c) for c in cells) + "\n")
    return str(tmp_path), "bin0000"


def test_3d_grid_parse_names_the_metallicity_mh(tmp_path):
    cells = ["id", 0.25, 5000.0, 4.0, 0.03, 50.0, 0.1, 0.0, 30.0, 0.05] + [0.0] * 9      # 19 cells
    wd, px = _opf(tmp_path, cells)
    r = F.parse_ferre_output_ifu(wd, px, 7, "n")
    assert r["MH_FERRE"] == pytest.approx(0.25) and r["MH_ERR_FERRE"] == pytest.approx(0.03)
    assert "FEH_FERRE" not in r and "FEH_ERR_FERRE" not in r
    assert r["TEFF_FERRE"] == 5000.0 and r["LOGG_FERRE"] == 4.0          # positions of the other parameters unchanged


def test_5d_grid_parse_keeps_the_positions(tmp_path):
    cells = ["id", -1.1, 0.3, 0.9, 4800.0, 3.5, 0.04, 0.02, 0.01, 40.0, 0.2, 0.0, 25.0, 0.1] + [0.0] * 25      # 39 cells
    wd, px = _opf(tmp_path, cells)
    r = F.parse_ferre_output_ifu(wd, px, 3, "m")
    assert r["MH_FERRE"] == pytest.approx(-1.1) and r["MH_ERR_FERRE"] == pytest.approx(0.04)
    assert r["ALPHA_FERRE"] == pytest.approx(0.3) and r["ALPHA_ERR_FERRE"] == pytest.approx(0.02)
    assert not [k for k in r if k.startswith("FEH")]


def test_white_dwarf_marker_is_in_mh(tmp_path):
    cells = ["id", 12000.0, 8.0, 100.0, 0.1, 0.0, 20.0, 0.1]  + [0.0] * 4      # 12 cells
    wd, px = _opf(tmp_path, cells)
    r = F.parse_ferre_output_ifu(wd, px, 1, "n")
    assert r["MH_FERRE"] == -10.0 and np.isnan(r["MH_ERR_FERRE"])


def test_writer_source_carries_no_old_name_outside_the_history_note():
    code = [l for l in SRC.splitlines() if "FEH_FERRE" in l or "FEH_ERR_FERRE" in l]
    assert all(l.lstrip().startswith(("- 1.2", "FEH_FERRE /", "through")) or "before" in l or "FEH_FERRE /" in l or "has FEH_FERRE" in l for l in code), code
    for name in ("MH_FERRE", "MH_ERR_FERRE"):
        assert len(re.findall(rf"'{name}'", SRC)) >= 3          # result dict, unit map, parameter column list


def test_the_naming_version_card_is_written_and_is_2_1():
    assert "outtab.meta['APS_FR_V'] = (aps_constants.__aps_ferre_version__" in SRC
    assert float(aps_constants.__aps_ferre_version__) >= 2.1


def test_viewer_reads_both_names():
    viewer = pytest.importorskip("PyAPS.aps_IFUviewer")
    assert viewer._ferre_mh_col(["BIN_ID", "TEFF_FERRE", "MH_FERRE"]) == "MH_FERRE"
    assert viewer._ferre_mh_col(["BIN_ID", "TEFF_FERRE", "FEH_FERRE"]) == "FEH_FERRE"       # an old L2 file
    for names, col in ((["BIN_ID", "TEFF_FERRE", "MH_FERRE"], "MH_FERRE"), (["BIN_ID", "TEFF_FERRE", "FEH_FERRE"], "FEH_FERRE")):
        rec = type("Rec", (), {"columns": type("C", (), {"names": names})(), "__getitem__": lambda s, k: np.array([0.5, -0.2])})()
        st = {"table": {"BIN_ID": np.array([1, 2])}, "ferre_available": True, "star_table_rec": rec,
              "star_table_pos": {1: 0, 2: 1}}
        for mt in ("FERRE_MH", "FERRE_FEH"):                                                    # a saved FERRE_FEH map type still works
            data, title, cbar = viewer._resolve_maptype(st, mt)
            assert list(data) == [0.5, -0.2] and "Metallicity" in title and cbar == "[M/H]"


def test_map_options_offer_mh_not_feh_for_ferre():
    viewer = pytest.importorskip("PyAPS.aps_IFUviewer")
    st = {"table": {}, "ferre_available": True, "ppxf_available": False, "rvs_available": False,
          "gandalf_results": None, "emi_available": False, "ls_available": False}
    groups = viewer._list_maptypes(st)
    opts = [v for g, items in groups if g == "Stellar (FERRE)" for v, _ in items]
    assert "FERRE_MH" in opts and "FERRE_FEH" not in opts
