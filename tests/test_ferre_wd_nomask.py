"""aps_ferre --wd_nomask: white dwarfs are fitted without the Balmer mask and the linemask (2026-11-12).

A target is exempt from both masks if the target catalogue marks it as a white dwarf (TARGSRVY contains the survey
code WD, or TARGCLASS is STAR_WD). The Redrock classification (rank 0: CLASS STAR with a white dwarf SUBCLASS) is used
only as a fallback for targets without catalogue class information (TARGCLASS empty or UNKNOWN); for every other
TARGCLASS it is ignored. With the option off, nothing changes.

Run with:
    cd <PYAPS_DIR> && pytest tests/test_ferre_wd_nomask.py -v
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from PyAPS import aps_ferre

NPIX = 8


# --------------------------------------------------------------------------- catalogue test
@pytest.mark.parametrize("targsrvy, targclass, expected", [
    ("WD", "STAR_WD", True),
    ("WD", "", True),                       # survey code alone is enough
    (" wd ", "unknown", True),              # whitespace / case
    (b"WD", b"STAR_WD", True),              # bytes from FITS
    ("GA-LRHIGHLAT,WD", "STAR", True),      # comma-separated list of surveys
    ("WS2022A2-002", "STAR_WD", True),      # STAR_WD in a non-WD survey
    ("WS2022A2-002", "STAR_FGK", False),
    ("WDX", "STAR", False),                 # exact token, not a substring
    ("SCIP-WD", "STAR", False),
    ("", "", False),
    (None, None, False),
    ("WS2023A1-025", "UNKNOWN", False),
    (np.array(["WD", "WD", "WD"]), np.array(["STAR_WD"] * 3), True),   # rank arrays of a classifier table
])
def test_catalogue_test(targsrvy, targclass, expected):
    assert aps_ferre._is_catalogue_wd(targsrvy, targclass) is expected


# --------------------------------------------------------------------------- Redrock subclass spellings
@pytest.mark.parametrize("subclass, expected", [
    ("WD", True),
    ("WD:DA_1", True),
    ("WD:DB_0", True),
    ("WDmagnetic_37;WD:DA_1;WD:DA_2", True),
    ("CalciumWD_35", True),
    ("CarbonWD", True),
    ("WD:DA_1;B_4;B5(338529)_11", True),
    ("O8e(188001)_31;WD:DA_1;B_4", False),     # only the best archetype counts
    ("CV_38;CV_39;WD:DB_0", False),
    ("A0", False),
    ("F5", False),
    ("", False),
])
def test_wd_subclass(subclass, expected):
    assert aps_ferre._wd_subclass(subclass) is expected


@pytest.mark.parametrize("classes, subclasses, expected", [
    (["STAR", "STAR", "STAR"], ["WD:DA_1", "A0", "F5"], True),
    (["STAR", "STAR", "STAR"], ["A0", "WD:DA_1", "WD"], False),          # white dwarf only at rank 1/2
    (["GALAXY", "STAR", "STAR"], ["", "WD", "WD"], False),               # rank 0 is not STAR
    (["QSO", "QSO", "STAR"], ["LOZ", "HIZ", "WD"], False),
    (np.array([b"STAR", b"GALAXY", b"QSO"]), np.array([b"WDmagnetic_37", b"", b""]), True),
    ("STAR", "WD", True),                                               # single mode (ntop=1): scalars
    (b"STAR", b"F5", False),
    ([], [], False),
])
def test_redrock_rank0(classes, subclasses, expected):
    assert aps_ferre._is_redrock_wd_rank0(classes, subclasses) is expected


# --------------------------------------------------------------------------- combined decision (fallback gate)
RR_WD = (["STAR", "STAR", "STAR"], ["WD:DA_1", "WD", "A0"])
RR_NOWD = (["STAR", "STAR", "STAR"], ["F5", "G2", "K1"])


@pytest.mark.parametrize("targsrvy, targclass, rr, exempt, reason", [
    ("WD", "STAR_WD", RR_NOWD, True, "catalogue"),           # catalogue wins, Redrock not needed
    ("WD", "STAR_WD", RR_WD, True, "catalogue"),
    ("WS2023A1-025", "UNKNOWN", RR_WD, True, "redrock_fallback"),
    ("", "", RR_WD, True, "redrock_fallback"),
    ("WS2023A1-025", "UNKNOWN", RR_NOWD, False, ""),
    # Redrock is ignored for every TARGCLASS with catalogue information
    ("WS2022A2-002", "STAR", RR_WD, False, ""),
    ("WS2022A2-002", "STAR_FGK", RR_WD, False, ""),
    ("WS2022A2-002", "STAR_OB", RR_WD, False, ""),
    ("WS2022A2-002", "SKY", RR_WD, False, ""),
    ("WC", "GALAXY", RR_WD, False, ""),
    ("WQ", "QSO", RR_WD, False, ""),
])
def test_wd_exemption(targsrvy, targclass, rr, exempt, reason):
    d = aps_ferre._wd_exemption(targsrvy, targclass, *rr)
    assert d["exempt"] is exempt
    assert d["reason"] == reason
    assert d["redrock_used"] is (aps_ferre._norm(targclass) in aps_ferre.WD_RR_FALLBACK_TARGCLASSES)


def test_fallback_classes_are_only_empty_and_unknown():
    assert tuple(aps_ferre.WD_RR_FALLBACK_TARGCLASSES) == ("", "UNKNOWN")


# --------------------------------------------------------------------------- wiring
def test_proc_ferre_default_is_off_and_masks_are_guarded():
    sig = inspect.signature(aps_ferre.proc_ferre)
    assert sig.parameters["wd_nomask"].default is False
    src = inspect.getsource(aps_ferre.proc_ferre)
    assert "if maskbalmer and not skip_masks:" in src
    assert "if uselinemasks and not skip_masks:" in src


def test_cli_option_default_false():
    src = inspect.getsource(aps_ferre.ferre_weave)
    assert '"--wd_nomask"' in src and "wd_nomask=args.wd_nomask" in src


def _outdict(n=2):
    t = Table()
    t["APS_ID"] = np.arange(1, n + 1)
    t["TARGID"] = [f"T{i}" for i in range(n)]
    t["CNAME"] = [f"WVE_{i:08d}" for i in range(n)]
    for c in ("TEFF", "TEFF_ERR", "LOGG", "LOGG_ERR", "MH", "MH_ERR", "ALPHA", "ALPHA_ERR", "MICRO", "MICRO_ERR"):
        t[c] = np.full(n, 0.5)
    t["COVAR"] = np.zeros((n, 5, 5))
    t["ELEM"] = np.full((n, 2), np.nan)
    t["ELEM_ERR"] = np.full((n, 2), np.nan)
    t["SNR_FR"] = np.full(n, 30.0)
    t["CHISQ_TOT"] = np.full(n, 1.2)
    t["FLAG_FR"] = np.ones(n, dtype=int)
    for c in ("LAMBDA_FR_B", "FLUX_FR_B", "ERROR_FR_B", "MODEL_FR_B"):
        t[c] = np.tile(np.linspace(4000.0, 4010.0, NPIX), (n, 1))
    return t


def _write(tmp_path, info):
    p, s = tmp_path / "ferre_t.fits", tmp_path / "ferre_spec_t.fits"
    aps_ferre.ferre_write_fits(_outdict(), ["single_r0000001.fit"], ["BLUE"], False,
                               {"wave": "Angstrom", "flux": "normalised"}, str(p), spec_file=str(s),
                               wd_nomask_info=info)
    return fits.getheader(p, 1), fits.getheader(s, 1)


def test_header_cards_only_when_on(tmp_path):
    for hdr in _write(tmp_path, {"n_catalogue": 25, "n_redrock": 1}):
        assert hdr["WDNOMASK"] is True
        assert hdr["NWDCAT"] == 25 and hdr["NWDRR"] == 1
    for hdr in _write(tmp_path, None):
        assert not ({"WDNOMASK", "NWDCAT", "NWDRR"} & set(hdr.keys()))
