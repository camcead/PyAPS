"""seg2d_deblend_mode (fixed / auto): crowd-adaptive splitting of blends of point sources (PyAPS.aps_ifu_deblend).

Synthetic white-light images only (gaussian stars, an exponential-disc galaxy with knots, gaussian noise);
needs numpy, scipy, astropy and sep. The real-data regression (OB 14699, a globular cluster) runs when
PYAPS_TEST_DATA points at a data root that has that OB, and is skipped otherwise."""
import json
import os
import pathlib

import numpy as np
import pytest

sep = pytest.importorskip("sep")
pytest.importorskip("scipy")
fits = pytest.importorskip("astropy.io.fits")
wcs_mod = pytest.importorskip("astropy.wcs")

from PyAPS import aps_ifu_deblend as D            # noqa: E402
from PyAPS import aps_ifu_seg3d as S              # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
SHAPE = (185, 171)
SIGMA = 1.0                      # noise per pixel
PSF_SIGMA = 2.3                  # pixels (FWHM 5.4 px = 2.7 arcsec at 0.5 arcsec/pixel)
PIX = 0.5                        # arcsec / pixel
THR, MINAREA = 1.0, 8
COARSE = 1                       # sub-thresholds of the coarse pass of the chain tests: 1 = no deblending at all, the
                                 # limit of the shipped 4, so the chain is surely one detection (see doc section 7)
SHIPPED = 4                      # the shipped seg2d_deblend_nthresh


# ----------------------------------------------------------------------------------------------------------------
# images
# ----------------------------------------------------------------------------------------------------------------
def star(img, x, y, peak, sig=PSF_SIGMA):
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    img += peak * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sig ** 2))


def galaxy(img, x, y, peak, scale, knots=(), q=0.7, theta=0.5):
    """exponential disc with `scale` (px) and star-forming knots [(dx, dy, peak)] on top"""
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    c, s = np.cos(theta), np.sin(theta)
    u = (xx - x) * c + (yy - y) * s
    v = (-(xx - x) * s + (yy - y) * c) / q
    img += peak * np.exp(-np.hypot(u, v) / scale)
    for dx, dy, kp in knots:
        star(img, x + dx, y + dy, kp)


def sparse_stars(img, n=14, seed=1, avoid=()):
    """n isolated stars (peaks 15-60 sigma) on a loose grid, away from `avoid` [(x, y, r)]"""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(400):
        x, y = rng.uniform(10, img.shape[1] - 10), rng.uniform(10, img.shape[0] - 10)
        if all(np.hypot(x - ax, y - ay) > r for ax, ay, r in avoid) and all(np.hypot(x - ox, y - oy) > 30 for ox, oy in out):
            out.append((x, y))
            star(img, x, y, rng.uniform(15, 60))
        if len(out) == n:
            break
    return out


def finish(img, seed=0):
    return img + np.random.default_rng(seed).normal(0.0, SIGMA, img.shape)


def extract(data, nthresh=COARSE):
    """the normal extraction of object_extract on a background-subtracted image"""
    return sep.extract(data, THR, minarea=MINAREA, deblend_nthresh=nthresh, err=SIGMA, segmentation_map=True)


def run_auto(data, nthresh=COARSE, **kw):
    objects, seg = extract(data, nthresh)
    new, nseg, info = D.auto_deblend(data, objects, seg, ext_thresh=THR, minarea=MINAREA, err=SIGMA,
                                     mask=np.zeros(data.shape, bool), pixscale_arcsec=PIX, **kw)
    return objects, seg, new, nseg, info


def chain_image(spacing=10.0, n=5, peak=45.0, y0=60.0, x0=35.0, seed=0):
    img = np.zeros(SHAPE)
    for k in range(n):
        star(img, x0 + k * spacing, y0 + 4.0 * np.sin(k), peak)
    return img


def crowded_with_chain(seed=0, with_galaxy=False):
    img = np.zeros(SHAPE)
    sparse_stars(img, n=14, seed=1, avoid=[(55, 60, 45), (130, 130, 40)])
    for k in range(5):
        star(img, 35 + k * 10.0, 60 + 4.0 * np.sin(k), 45.0)
    if with_galaxy:
        galaxy(img, 130, 130, 12.0, 6.0, knots=[(-7, 3, 14.0), (6, -5, 12.0), (2, 8, 10.0)])
    return finish(img, seed)


def segcheck(objects, seg):
    """labels are 1..n, one per row, and the object columns agree with the map"""
    assert seg.dtype == np.int32
    assert sorted(set(np.unique(seg)) - {0}) == list(range(1, len(objects) + 1))
    for i in range(len(objects)):
        ys, xs = np.nonzero(seg == i + 1)
        assert len(xs) == objects["npix"][i]
        assert objects["xmin"][i] == xs.min() and objects["xmax"][i] == xs.max()
        assert objects["ymin"][i] == ys.min() and objects["ymax"][i] == ys.max()


# ----------------------------------------------------------------------------------------------------------------
# the measured columns agree with sep
# ----------------------------------------------------------------------------------------------------------------
def test_measure_pixels_reproduces_sep_for_an_undeblended_object():
    data = finish(chain_image(n=3), seed=3)
    objects, seg = extract(data, nthresh=1)
    conv = D.filtered(data)
    for i in range(len(objects)):
        ys, xs = np.nonzero(seg == i + 1)
        row = D.measure_pixels(ys, xs, data, conv, float(objects["thresh"][i]), SIGMA, int(objects["flag"][i]), objects.dtype)
        for name in ("npix", "tnpix", "x", "y", "x2", "y2", "xy", "a", "b", "theta", "cxx", "cyy", "cxy", "flux", "cflux",
                     "peak", "xpeak", "ypeak", "errx2", "erry2", "errxy"):
            assert float(row[name][0]) == pytest.approx(float(objects[name][i]), rel=2e-3, abs=1e-6), name
        assert row.dtype == objects.dtype


# ----------------------------------------------------------------------------------------------------------------
# (b) a crowded field: the chain of close stars is split into the stars
# ----------------------------------------------------------------------------------------------------------------
def test_chain_of_stars_in_a_crowded_field_is_split_into_the_stars():
    data = crowded_with_chain()
    objects, seg, new, nseg, info = run_auto(data)
    assert info["gate_passed"] and info["n_compact"] >= 10, info
    # the coarse pass merged the chain: one large elongated detection at the chain
    near = [i for i in range(len(objects)) if 25 <= objects["x"][i] <= 85 and 50 <= objects["y"][i] <= 72]
    assert len(near) == 1, "test image: the coarse pass must merge the chain"
    elong = objects["a"][near[0]] / objects["b"][near[0]]
    assert elong > 2.0, elong
    assert info["n_parents_split"] == 1 and info["n_children"] == 5, info
    kids = [i for i in range(len(new)) if 25 <= new["x"][i] <= 85 and 50 <= new["y"][i] <= 72]
    assert len(kids) == 5
    got = sorted(new["x"][kids])
    want = [35 + k * 10.0 for k in range(5)]
    assert np.allclose(got, want, atol=1.5), (got, want)
    assert max(new["a"][kids]) < 0.5 * objects["a"][near[0]]            # each child is star-sized
    assert len(new) == len(objects) + 4
    segcheck(new, nseg)
    # nothing else changed: every other detection is the same row with the same pixels
    others_old = [i for i in range(len(objects)) if i not in near]
    others_new = [i for i in range(len(new)) if i not in kids]
    assert len(others_old) == len(others_new)
    for io, in_ in zip(others_old, others_new):
        assert np.array_equal(objects[io], new[in_])
        assert np.array_equal(seg == io + 1, nseg == in_ + 1)
    # pixels outside the chain keep their label, the chain pixels are all assigned to a child
    assert np.array_equal(seg > 0, nseg > 0)


def test_children_get_their_own_ellipse_in_the_patch_table_columns():
    data = crowded_with_chain()
    objects, seg, new, nseg, info = run_auto(data)
    parent = [i for i in range(len(objects)) if 25 <= objects["x"][i] <= 85 and 50 <= objects["y"][i] <= 72][0]
    kids = [i for i in range(len(new)) if 25 <= new["x"][i] <= 85 and 50 <= new["y"][i] <= 72]
    # A_world = a * radii_factor * pixel scale: the five small ellipses together are clearly smaller than the parent's
    bins = lambda o: np.pi / 4 * (0.5 * o["a"] * 8 * PIX) * (0.5 * o["b"] * 8 * PIX) / 4
    assert bins(new[kids]).sum() < 0.7 * bins(objects[parent:parent + 1]).sum()
    assert new.dtype == objects.dtype
    for k in kids:
        assert new["flux"][k] > 0 and new["a"][k] >= new["b"][k] > 0
        assert new["xmin"][k] <= new["x"][k] <= new["xmax"][k] and new["ymin"][k] <= new["y"][k] <= new["ymax"][k]


# ----------------------------------------------------------------------------------------------------------------
# (c) sparse field: identical to fixed
# ----------------------------------------------------------------------------------------------------------------
def test_sparse_field_is_identical_to_fixed():
    img = np.zeros(SHAPE)
    for k in range(5):
        star(img, 35 + k * 10.0, 60 + 4.0 * np.sin(k), 45.0)    # the same blend, but only 3 other stars
    sparse_stars(img, n=3, seed=2, avoid=[(55, 60, 45)])
    data = finish(img)
    objects, seg, new, nseg, info = run_auto(data)
    assert not info["gate_passed"] and info["n_parents_split"] == 0
    assert new is objects and nseg is seg                       # untouched, not merely equal


# ----------------------------------------------------------------------------------------------------------------
# (a) an isolated galaxy with knots in a sparse field: identical to fixed
# ----------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("nthresh", [COARSE, SHIPPED])
def test_galaxy_with_knots_in_a_sparse_field_is_identical_to_fixed(nthresh):
    img = np.zeros(SHAPE)
    galaxy(img, 90, 90, 12.0, 8.0, knots=[(-9, 4, 16.0), (8, -6, 14.0), (3, 10, 12.0), (-4, -9, 10.0)])
    sparse_stars(img, n=4, seed=3, avoid=[(90, 90, 50)])
    data = finish(img)
    objects, seg, new, nseg, info = run_auto(data, nthresh)
    assert new is objects and nseg is seg
    assert info["n_parents_split"] == 0


@pytest.mark.parametrize("nthresh", [COARSE, SHIPPED])
def test_galaxy_with_knots_is_not_split_even_when_the_field_is_crowded(nthresh):
    """the gate passes (many stars) but the galaxy -- smooth light, knots on it -- must stay one detection"""
    data = finish(np.zeros(SHAPE) + _galaxy_and_stars())
    objects, seg, new, nseg, info = run_auto(data, nthresh)
    assert info["gate_passed"], info
    gal = [i for i in range(len(objects)) if np.hypot(objects["x"][i] - 100, objects["y"][i] - 100) < 14]
    assert gal and info["n_parents_split"] == 0, info
    assert new is objects


def _galaxy_and_stars():
    img = np.zeros(SHAPE)
    galaxy(img, 100, 100, 12.0, 8.0, knots=[(-9, 4, 16.0), (8, -6, 14.0), (3, 10, 12.0), (-4, -9, 10.0)])
    sparse_stars(img, n=14, seed=4, avoid=[(100, 100, 45)])
    return img


# ----------------------------------------------------------------------------------------------------------------
# (d) a crowded field with a big galaxy: stars split, galaxy kept
# ----------------------------------------------------------------------------------------------------------------
def test_crowded_field_with_a_big_galaxy_splits_the_stars_and_keeps_the_galaxy():
    data = crowded_with_chain(with_galaxy=True)
    objects, seg, new, nseg, info = run_auto(data)
    assert info["gate_passed"]
    assert info["n_parents_split"] == 1 and info["n_children"] == 5, info
    gal_old = [i for i in range(len(objects)) if np.hypot(objects["x"][i] - 130, objects["y"][i] - 130) < 14]
    gal_new = [i for i in range(len(new)) if np.hypot(new["x"][i] - 130, new["y"][i] - 130) < 14]
    assert len(gal_old) == 1 and len(gal_new) == 1
    assert np.array_equal(objects[gal_old[0]], new[gal_new[0]])
    assert np.array_equal(seg == gal_old[0] + 1, nseg == gal_new[0] + 1)
    segcheck(new, nseg)


# ----------------------------------------------------------------------------------------------------------------
# the cap
# ----------------------------------------------------------------------------------------------------------------
def test_a_split_never_goes_over_the_cap_and_the_brightest_parents_come_first():
    data = crowded_with_chain()
    objects, seg = extract(data)
    n = len(objects)
    new, nseg, info = D.auto_deblend(data, objects, seg, ext_thresh=THR, minarea=MINAREA, err=SIGMA,
                                     mask=np.zeros(data.shape, bool), pixscale_arcsec=PIX, max_total=n + 2)
    assert info["n_skipped_by_cap"] == 1 and info["n_parents_split"] == 0 and new is objects     # needs +4
    new, nseg, info = D.auto_deblend(data, objects, seg, ext_thresh=THR, minarea=MINAREA, err=SIGMA,
                                     mask=np.zeros(data.shape, bool), pixscale_arcsec=PIX, max_total=n + 4)
    assert info["n_skipped_by_cap"] == 0 and len(new) == n + 4


# ----------------------------------------------------------------------------------------------------------------
# parameters
# ----------------------------------------------------------------------------------------------------------------
def test_parse_mode():
    assert D.parse_mode(None) == "auto" and D.parse_mode("None") == "auto" and D.parse_mode("") == "auto"
    assert D.parse_mode("FIXED") == "fixed" and D.parse_mode(" Auto ") == "auto"
    with pytest.raises(ValueError):
        D.parse_mode("fine")


def test_gate_parameter_switches_the_field_on_and_off():
    data = crowded_with_chain()
    objects, seg, new, nseg, info = run_auto(data, params=dict(gate_min_compact=1000))
    assert not info["gate_passed"] and new is objects
    objects, seg, new, nseg, info = run_auto(data, params=dict(gate_min_compact=3))
    assert info["gate_passed"] and info["n_parents_split"] == 1


# ----------------------------------------------------------------------------------------------------------------
# (e)(f) the pipeline step: fixed is the old code path, auto is fixed where nothing is split, the summary says it
# ----------------------------------------------------------------------------------------------------------------
def make_white(tmp_path, sub, level=5000.0, name="OBX_white.fits"):
    """a white-light FITS with an ICRS TAN WCS (0.5 arcsec/pixel), blanks 0 at the corners, sky `level`"""
    img = (sub * 1000.0 + level).astype("float32")
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    img[(xx - 85) ** 2 + (yy - 92) ** 2 > 90 ** 2] = 0.0
    h = fits.Header()
    h["CTYPE1"], h["CTYPE2"] = "RA---TAN", "DEC--TAN"
    h["CRVAL1"], h["CRVAL2"] = 150.0, 20.0
    h["CRPIX1"], h["CRPIX2"] = 86.0, 93.0
    h["CDELT1"], h["CDELT2"] = -PIX / 3600.0, PIX / 3600.0
    h["RADESYS"] = "ICRS"
    p = tmp_path / name
    fits.PrimaryHDU(img, header=h).writeto(p, overwrite=True)
    return str(p)


def new_seg2d(tmp_path, white, mode, max_targets=150):
    from PyAPS import aps_ifu_prepare as P
    o = object.__new__(P.ifu_seg2d)
    o.whiteimage = white
    o.gaia_stars, o.mask_stars, o.extc_mask = None, False, None
    o.max_targets, o.radii_factor, o._seg2d_est = max_targets, 8.0, None
    o.headname, o.outpath = "HEAD", str(tmp_path)
    o.figdir = tmp_path
    o.deblend_mode = mode
    o.deblend_params = {}
    o.seg2d_summary = {}
    return o


def old_pipeline_extract(white, ext_thresh, minarea, nthresh):
    """the extraction of object_extract before seg2d_deblend_mode existed (kept here as the reference)"""
    d = fits.open(white)[0].data.astype(float)
    d[d == 0] = np.nan
    mask = np.isnan(d)
    clean, _ = S.prepare_white(d)
    bkg, rms, _info = S.robust_background(clean, mask, bw=64, bh=64)
    sub = np.where(mask, 0.0, clean - bkg)
    return sep.extract(sub, ext_thresh, minarea=minarea, deblend_nthresh=nthresh, err=rms, mask=mask, segmentation_map=True)


@pytest.mark.parametrize("builder", ["chain", "sparse", "galaxy"])
def test_fixed_mode_is_exactly_the_old_code_path(tmp_path, builder):
    data = {"chain": crowded_with_chain, "sparse": lambda: finish(chain_image(n=2)),
            "galaxy": lambda: finish(_galaxy_and_stars())}[builder]()
    white = make_white(tmp_path, data)
    nt = COARSE if builder == "chain" else SHIPPED
    o = new_seg2d(tmp_path, white, "fixed")
    o.object_extract(ext_thresh=THR, minarea=MINAREA, deblend_nthresh=nt, mask_stars=False, patch_table=False, fig=False)
    ref_objects, ref_seg = old_pipeline_extract(white, THR, MINAREA, nt)
    assert np.array_equal(o.objects, ref_objects) and o.objects.dtype == ref_objects.dtype
    assert np.array_equal(o.segmap, ref_seg)
    s = json.loads((tmp_path / "HEAD_seg2d_summary.json").read_text())
    assert s["deblend_mode"] == "fixed" and s["n_parents_split"] == 0 and s["n_children"] == 0
    assert s["n_parents"] == len(ref_objects) == s["n_detected"] and s["deblend_auto"] is None


@pytest.mark.parametrize("builder", ["sparse", "galaxy"])
def test_auto_mode_equals_fixed_when_nothing_is_split(tmp_path, builder):
    data = {"sparse": lambda: finish(chain_image(n=2)), "galaxy": lambda: finish(_galaxy_and_stars())}[builder]()
    white = make_white(tmp_path, data)
    o = new_seg2d(tmp_path, white, "auto")
    o.object_extract(ext_thresh=THR, minarea=MINAREA, deblend_nthresh=SHIPPED, mask_stars=False, patch_table=False, fig=False)
    ref_objects, ref_seg = old_pipeline_extract(white, THR, MINAREA, SHIPPED)
    assert np.array_equal(o.objects, ref_objects) and np.array_equal(o.segmap, ref_seg)
    s = json.loads((tmp_path / "HEAD_seg2d_summary.json").read_text())
    assert s["deblend_mode"] == "auto" and s["n_parents_split"] == 0


def test_auto_mode_splits_the_chain_and_the_summary_reports_it(tmp_path, capsys):
    white = make_white(tmp_path, crowded_with_chain())
    ref_objects, ref_seg = old_pipeline_extract(white, THR, MINAREA, COARSE)
    o = new_seg2d(tmp_path, white, "auto")
    o.object_extract(ext_thresh=THR, minarea=MINAREA, deblend_nthresh=COARSE, mask_stars=False, patch_table=False, fig=False)
    s = json.loads((tmp_path / "HEAD_seg2d_summary.json").read_text())
    assert s["deblend_mode"] == "auto"
    assert s["n_parents"] == len(ref_objects)
    assert s["n_parents_split"] == 1 and s["n_children"] == 5
    assert s["n_detected"] == len(ref_objects) + 4 == s["n_kept"] == len(o.objects)
    a = s["deblend_auto"]
    assert a["gate_passed"] is True and a["n_compact"] >= a["gate_min_compact"] and a["psf_size_arcsec"] > 0
    assert a["splits"] and a["splits"][0]["n_children"] == 5 and set(a["n_rejected"]) >= {"one_child", "too_few_real", "extended"}
    # the existing keys are still there
    for k in ("ext_thresh", "minarea", "deblend_nthresh", "n_detected", "n_kept", "n_dropped_by_cap", "max_targets", "sky_level", "sky_rms"):
        assert k in s
    out = capsys.readouterr().out
    assert "seg2d deblend auto: crowded field" in out and "1 split into 5 children" in out
    segcheck(o.objects, o.segmap)


def test_cap_still_keeps_the_brightest_after_the_split(tmp_path):
    white = make_white(tmp_path, crowded_with_chain())
    o = new_seg2d(tmp_path, white, "auto", max_targets=10)
    o.object_extract(ext_thresh=THR, minarea=MINAREA, deblend_nthresh=COARSE, mask_stars=False, patch_table=False, fig=False)
    s = json.loads((tmp_path / "HEAD_seg2d_summary.json").read_text())
    assert s["n_kept"] == 10 and s["n_dropped_by_cap"] > 0 and s["n_parents_split"] == 0     # no room: nothing split
    assert s["deblend_auto"]["n_skipped_by_cap"] == 1


def test_auto_failure_keeps_the_normal_extraction(tmp_path, monkeypatch, capsys):
    white = make_white(tmp_path, crowded_with_chain())
    ref_objects, ref_seg = old_pipeline_extract(white, THR, MINAREA, COARSE)

    def boom(*a, **k):
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(D, "auto_deblend", boom)
    o = new_seg2d(tmp_path, white, "auto")
    o.object_extract(ext_thresh=THR, minarea=MINAREA, deblend_nthresh=COARSE, mask_stars=False, patch_table=False, fig=False)
    assert np.array_equal(o.objects, ref_objects) and np.array_equal(o.segmap, ref_seg)
    assert "WARNING: seg2d_deblend_mode=auto failed" in capsys.readouterr().out
    s = json.loads((tmp_path / "HEAD_seg2d_summary.json").read_text())
    assert s["deblend_mode"] == "auto" and "synthetic failure" in s["deblend_auto"]["error"]


def test_the_patch_table_gets_one_row_per_child(tmp_path):
    white = make_white(tmp_path, crowded_with_chain())
    o = new_seg2d(tmp_path, white, "auto")
    o.headname = "HEAD"
    from astropy.wcs import WCS
    o.extc_wcs = None
    # run the whole step through the class' own table builder
    o.object_extract(ext_thresh=THR, minarea=MINAREA, deblend_nthresh=COARSE, mask_stars=False, patch_table=False, fig=False)
    o.icrs_param = dict(RA_CENT=150.0, DEC_CENT=20.0, RA_WIDTH=0.02, DEC_WIDTH=0.02)
    o.mask_stars, o._class_ntop = False, 1
    o.create_aps_targets()
    t = o.patch_table
    rows = t[t["type"] == "T"]
    assert len(rows) == len(o.objects)
    a_world = np.sort(np.asarray(rows["A_world"], float))
    assert a_world[-1] < 40.0 / 3600.0                      # no 60-arcsec patch for the chain any more


# ----------------------------------------------------------------------------------------------------------------
# wiring
# ----------------------------------------------------------------------------------------------------------------
def test_flag_and_runner_and_configs_carry_the_mode():
    prep = (ROOT / "py/PyAPS/aps_ifu_prepare.py").read_text()
    assert '"--seg2d_deblend_mode"' in prep and "seg2d_deblend_mode=args.seg2d_deblend_mode" in prep
    assert "deblend_mode=seg2d_deblend_mode" in prep and "aps_ifu_deblend.auto_deblend" in prep
    runner = (ROOT / "py/PyAPS/aps_runner.py").read_text()
    assert "' --seg2d_deblend_mode '" in runner and "'seg2d_deblend_mode',   'auto'" in runner
    for cfg in sorted((ROOT / "configs/ExGal_configs").glob("*IFU*.json")):
        if cfg.name.startswith("MOS"):
            continue
        assert json.loads(cfg.read_text())["seg2d_deblend_mode"] == "auto", cfg.name
    assert "seg2d_deblend_mode: 'auto'" in (ROOT / "configs/script_params.yaml.example").read_text()


def test_cli_rejects_an_unknown_mode():
    import argparse
    from PyAPS import aps_ifu_prepare as P
    assert P._deblend_mode_arg("fixed") == "fixed" and P._deblend_mode_arg("AUTO") == "auto"
    with pytest.raises(argparse.ArgumentTypeError):
        P._deblend_mode_arg("coarse")


# ----------------------------------------------------------------------------------------------------------------
# real data: OB 14699 (globular cluster M3, night 20260315), only where the data are
# ----------------------------------------------------------------------------------------------------------------
def _ob14699():
    root = os.environ.get("PYAPS_TEST_DATA")
    if not root:
        pytest.skip("real WEAVE data not available (set PYAPS_TEST_DATA)")
    p = pathlib.Path(root) / "L2/20260315/14699/LWVE_13421721+2822398_01_GR_H1_white.fits"
    if not p.exists():
        pytest.skip("OB 14699 white-light image not found under PYAPS_TEST_DATA")
    return p


def test_ob14699_globular_cluster_blends_are_split(tmp_path):
    white = _ob14699()
    ref_objects, ref_seg = old_pipeline_extract(str(white), 1.0, 8, 4)
    assert len(ref_objects) == 44                                  # what the pipeline found
    o = new_seg2d(tmp_path, str(white), "auto")
    o.object_extract(ext_thresh=1.0, minarea=8, deblend_nthresh=4, mask_stars=False, patch_table=False, fig=False)
    s = json.loads((tmp_path / "HEAD_seg2d_summary.json").read_text())
    assert s["n_parents"] == 44 and s["n_parents_split"] >= 2 and s["n_detected"] > 44
    px = 0.5
    assert ref_objects["a"].max() * px > 11.0                       # the chain of eight stars, a = 12 arcsec
    assert o.objects["a"].max() * px < 6.0                          # no detection longer than 6 arcsec now
    bins = lambda ob: float((np.pi / 4 * (0.5 * ob["a"] * 8 * px) * (0.5 * ob["b"] * 8 * px) / 4).sum())
    assert bins(o.objects) < 0.9 * bins(ref_objects)
    segcheck(o.objects, o.segmap)


# ----------------------------------------------------------------------------------------------------------------
# thresholds of the auto mode can be set from the configuration (seg2d_deblend_params)
# ----------------------------------------------------------------------------------------------------------------
def test_parse_params_reads_overrides_and_rejects_typos():
    from PyAPS import aps_ifu_deblend as D
    assert D.parse_params(None) == D.parse_params("") == D.parse_params("none") == {}
    got = D.parse_params("gate_min_compact=8, min_separation=1.2;fine_nthresh=32")
    assert got == {"gate_min_compact": 8, "min_separation": 1.2, "fine_nthresh": 32}
    assert isinstance(got["gate_min_compact"], int) and isinstance(got["min_separation"], float)
    assert D.parse_params({"child_min_flux_frac": 0.1}) == {"child_min_flux_frac": 0.1}
    for bad in ("gate_min_compct=8", "gate_min_compact", "gate_min_compact=x", "gate_min_compact=nan"):
        with pytest.raises(ValueError):
            D.parse_params(bad)


def test_a_bad_override_stops_the_stage_before_any_work():
    from PyAPS import aps_ifu_deblend as D
    objs = np.zeros(0, dtype=[("x", float)])
    # an empty detection list returns at once, but the overrides are validated before anything else
    with pytest.raises(ValueError):
        D.auto_deblend(np.zeros((4, 4)), objs, np.zeros((4, 4), int), ext_thresh=2.5, minarea=1, err=1.0,
                       mask=np.zeros((4, 4), bool), pixscale_arcsec=0.5, params="nonsense=1")


def test_flag_runner_and_config_carry_the_params():
    prep = (ROOT / "py/PyAPS/aps_ifu_prepare.py").read_text()
    assert '"--seg2d_deblend_params"' in prep and "seg2d_deblend_params=args.seg2d_deblend_params" in prep
    assert "deblend_params=seg2d_deblend_params" in prep
    runner = (ROOT / "py/PyAPS/aps_runner.py").read_text()
    assert "'seg2d_deblend_params', 'none'" in runner and "' --seg2d_deblend_params '" in runner
