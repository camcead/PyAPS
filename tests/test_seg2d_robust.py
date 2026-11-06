"""seg2d on a white-light image with very bright sources (OB 26133, 2026-11-06): robust sky, calibrated threshold,
fibre-footprint floor, patch tally. Synthetic images, no data files needed (needs numpy, scipy, astropy, sep)."""
import io
import json
import pathlib

import numpy as np
import pytest

sep = pytest.importorskip("sep")
pytest.importorskip("astropy")
pytest.importorskip("scipy")

from PyAPS import aps_ifu_seg3d as sg                      # noqa: E402
from PyAPS.aps_ifu_patch_report import PatchTally          # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def fibre_mosaic(seed=0, n_bright=(4500.0, 2500.0), real=(30.0, 12.0), offset=8000.0, sigma=9500.0, side=170, pitch=7):
    """A LIFU-like white-light image: ~600 single-fibre footprints (disks of ~22 px, flat value + a little pixel
    noise) with blank (0) gaps between them, noise sigma `sigma` on a pedestal `offset`, a few very bright fibres.
    Returns (image, list of (x, y) of the bright/real fibres)."""
    rng = np.random.default_rng(seed)
    img = np.zeros((side, side))
    yy, xx = np.mgrid[0:side, 0:side]
    centres = [(x, y) for y in range(4, side - 4, pitch) for x in range(4, side - 4, pitch)]
    strong = {}
    order = rng.permutation(len(centres))
    levels = list(n_bright) + list(real)
    for k, lev in zip(order[:len(levels)], levels):
        strong[centres[k]] = lev
    for (cx, cy) in centres:
        disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= 2.65 ** 2          # ~22 px
        val = offset + rng.normal(0, sigma) + strong.get((cx, cy), 0.0) * sigma
        img[disk] = val + rng.normal(0, 0.3 * sigma, disk.sum())
    img[(img == 0) & (np.zeros_like(img, bool))] = 0                  # blanks stay exactly 0 (they are the gaps)
    return img, list(strong)


def plain_sep_extract(img, thr, minarea):
    """what the code did before 2026-11-06"""
    d = img.copy(); d[d == 0] = np.nan
    m = np.isnan(d); d = np.ascontiguousarray(np.nan_to_num(d), dtype="<f8")
    bkg = sep.Background(d, mask=m, bw=64, bh=64)
    return sep.extract(d - bkg, thr, minarea=minarea, err=bkg.globalrms, mask=m)


def test_robust_background_is_not_biased_by_two_very_bright_sources():
    img, _ = fibre_mosaic()
    d, m = sg.prepare_white(img)
    bkg, rms, info = sg.robust_background(d, m)
    assert abs(info["level"] - 8000.0) < 0.3 * 9500.0, info
    assert abs(rms - 9500.0) < 0.2 * 9500.0, rms
    assert info["constant"] is True                      # a sparse mosaic: constant clipped sky
    assert info["rejected_fraction"] < 0.2


def test_new_method_finds_the_real_sources_and_not_a_flood():
    """Two very bright fibres (4500 and 2500 sigma) and two real ones (30 and 12 sigma) among ~600 noise fibres.
    The old code (plain sep.Background) put the threshold 0.37 sigma above the sky on the real OB 26133 and found
    148 detections; the synthetic image cannot reproduce sep's degenerate background, so only the new method is
    tested here (the real-data comparison is in doc/aps_ifu_seg2d_robust.md)."""
    img, strong = fibre_mosaic()
    d, m = sg.prepare_white(img)
    est = sg.estimate_seg2d_params(img, verbose=False)
    bkg, rms, info = sg.robust_background(d, m)
    new = sep.extract(d - bkg, est["ext_thresh"], minarea=est["minarea"], deblend_nthresh=4, err=rms, mask=m)
    thr_sigma = (est["ext_thresh"] * rms + info["level"] - 8000.0) / 9500.0
    assert thr_sigma >= 2.0, thr_sigma                   # the threshold lies well above the true sky
    assert 2 <= len(new) <= 8, len(new)                  # the real fibres (at least the two very bright ones), no flood
    peaks = sorted(new["peak"])[-2:]
    assert peaks[0] > 1e7 / 4, peaks                     # the two very bright sources are among them


def test_fibre_footprint_floor_for_a_mosaic_only():
    img, _ = fibre_mosaic()
    d, m = sg.prepare_white(img)
    fp = sg.fibre_footprint_px(m)
    assert fp is not None and 15 <= fp <= 30, fp
    est = sg.estimate_seg2d_params(img, verbose=False)
    assert est["minarea"] >= round(sg.SEG2D_FIBRE_MIN_FRAC * fp)
    smooth = np.random.default_rng(1).normal(100.0, 5.0, (120, 120))          # a smooth (PS1-like) image: no floor
    ds, ms = sg.prepare_white(smooth)
    assert sg.fibre_footprint_px(ms) is None


def test_pure_noise_field_fails_calibration_loudly_and_finds_nothing():
    rng = np.random.default_rng(3)
    img, _ = fibre_mosaic(n_bright=(), real=(), seed=3)
    est = sg.estimate_seg2d_params(img, verbose=False)
    d, m = sg.prepare_white(img)
    bkg, rms, _ = sg.robust_background(d, m)
    n = len(sep.extract(d - bkg, est["ext_thresh"], minarea=est["minarea"], err=rms, mask=m))
    assert est["calibration_ok"] is False and est["ext_thresh"] == max(sg.SEG2D_THRESH_GRID)
    assert n <= 1, n


def test_verbose_output_names_the_robust_sky_and_the_warning(capsys):
    img, _ = fibre_mosaic(n_bright=(), real=(), seed=4)
    sg.estimate_seg2d_params(img, verbose=True)
    out = capsys.readouterr().out
    assert "robust sky:" in out and "WARNING: no threshold of the grid reached purity" in out


def test_patch_tally_summary_and_failure_rule():
    out = io.StringIO()
    t = PatchTally("ExGal", "/nonexistent/should/not/break", "H")
    for _ in range(226):
        t.attempt()
    for _ in range(222):
        t.no_valid_spaxels()
    for _ in range(4):
        t.written()
    assert t.finish(out) is False                         # 4 written: the job completes, loudly
    text = out.getvalue()
    assert 'IFU_PATCH_SUMMARY {' in text and '"no_valid_spaxels": 222' in text and "WARNING: IFU_ExGal produced a product for only 4 of 226" in text
    t2 = PatchTally("Gal", "/nonexistent/x", "H")
    for _ in range(12):
        t2.attempt(); t2.no_valid_spaxels()
    assert t2.finish(io.StringIO()) is True               # nothing written from >= 10 patches: fail
    t3 = PatchTally("Gal", "/nonexistent/x", "H")
    for _ in range(3):
        t3.attempt(); t3.no_valid_spaxels()
    assert t3.finish(io.StringIO()) is False              # few patches, none written: not enough to fail


def test_patch_tally_writes_json(tmp_path):
    t = PatchTally("ExGal", tmp_path, "HEAD")
    t.attempt(); t.written()
    t.finish(io.StringIO())
    data = json.loads((tmp_path / "HEAD_ExGal_patches.json").read_text())
    assert data["attempted"] == 1 and data["written"] == 1 and data["yield"] == 1.0


def test_pipeline_wiring_is_in_place():
    prep = (ROOT / "py/PyAPS/aps_ifu_prepare.py").read_text()
    assert "robust_background(extc_clean" in prep and "SEG2D_SUMMARY" in prep and "max_targets" in prep
    assert "sep.Background(extc_data" not in prep, "object_extract must not use the plain sep.Background any more"
    for stage in ("aps_ifu_ExGal", "aps_ifu_Gal"):
        src = (ROOT / "py/PyAPS" / (stage + ".py")).read_text()
        assert "PatchTally" in src and "tally.finish()" in src and "tally.written()" in src
    seg = (ROOT / "py/PyAPS/aps_ifu_seg3d.py").read_text()
    assert seg.count("sep.Background(") <= 3, "sep.Background only inside robust_background"
