"""Safe, crowd-adaptive deblending of the IFU seg2d detections (stage IFU_PR).

Why: `sep.extract` with the shipped `deblend_nthresh=4` leaves chains of close stars of a crowded field (a
globular cluster, OB 14699 of night 20260315) as a few large "blends"; each blend becomes one IFU patch and
the Galactic stage then bins it and runs FERRE on every bin. A finer deblending (64 levels) splits them, but
the same setting would also tear a galaxy apart: noise ripples and star-forming knots become "objects", and in
the collapsed image (fibres of 2.6 arcsec) a compact galaxy and a pair of stars look alike.
`seg2d_deblend_mode = auto` therefore splits a detection (a "parent") only when ALL of these hold:

 1. the field is crowded: at least `gate_min_compact` (10) compact, round, measurable detections;
 2. the fine pass finds at least two children that are real sources: flux >= 7 % of the parent's, a
    topographic prominence (peak minus the saddle to a brighter neighbour) >= 10 % of the child's peak and
    >= 6 sigma of the filtered image, and the peaks are >= 1.5 x the PSF size apart (closer ones are one source);
 3. the parent is not extended: no more than 35 % of its flux lies outside the real children, the fine pass
    does not break it into 25 or more pieces, and it does not touch an object that is extended in that sense
    (the knots of a galaxy touch the galaxy);
 4. every real child, with the fragments around it given to it, is point-like (core size <= 2 x PSF) and
    covers at least one PSF disc, and none holds more than 70 % of the flux;
 5. the split pays: it saves at least 10 Galactic-stage bins and 15 % of the parent's bins.

A detection that fails any of them is kept exactly as `sep.extract` returned it, and when no detection is
split the input arrays are returned untouched, so a sparse field or a galaxy field gives bit-for-bit the
same result as `seg2d_deblend_mode = fixed`. See doc/aps_ifu_seg2d_robust.md section 7.

Everything here works on the arrays `ifu_seg2d.object_extract` already has (background-subtracted image,
`objects`, `segmap`); the image must be the one given to sep.extract (blanks 0, `mask` True on blanks).
"""
from __future__ import annotations

import numpy as np
import sep
from scipy import ndimage

FWHM_SIGMA = 2.3548200450309493          # 2*sqrt(2*ln2)

DEBLEND_MODES = ("fixed", "auto")

#: defaults of the `auto` mode (all can be overridden through `params` of `auto_deblend`)
AUTO_PARAMS = dict(
    fine_nthresh=64,            # fine pass: sub-thresholds ...
    fine_cont=1e-4,             # ... and minimum contrast (sep.extract deblend_cont)
    gate_min_compact=10,        # a field is crowded with at least this many compact point-like detections
    gate_compact_size=1.5,      # a detection is compact if its core size <= this x the PSF size ...
    gate_min_roundness=0.5,     # ... its b/a >= this ...
    gate_min_peak_sigma=5.0,    # ... and its filtered peak >= this x the noise of the filtered image
    child_min_flux_frac=0.07,   # a real child holds at least this fraction of the parent's flux ...
    child_min_prominence=0.10,  # ... a prominence of at least this fraction of its own peak ...
    child_min_prom_sigma=6.0,   # ... and of at least this many sigma of the filtered image
    min_separation=1.5,         # real children must be at least this x the PSF size apart (else they are one source)
    max_diffuse=0.35,           # at most this share of the parent's flux may lie outside the real children
    extended_diffuse=0.5,       # an object with more than this share outside its real children is extended ...
    extended_n_fine=25,         # ... as is one that the fine pass breaks into at least this many pieces ...
    touch_px=2,                 # ... and a parent within this many pixels of an extended object is not split
    child_max_size=2.0,         # a child (with its fragments) is point-like if its core size <= this x the PSF size
    min_child_area=1.0,         # a child must cover at least this x the area of a PSF disc (pi (psf/2)^2)
    dominant_flux_frac=0.70,    # a child with more than this fraction of the flux makes the parent one object
    min_bins_saved=10.0,        # a split must save at least this many Galactic-stage bins (2 x 2 arcsec) ...
    min_saving=0.15,            # ... and cut the parent's bins by at least this fraction
    psf_min_peak_sigma=10.0,    # PSF size = median core size of the detections with a peak above this ...
    psf_min_sources=3,          # ... when there are at least this many (else psf_default_arcsec)
    psf_min_arcsec=2.0,         # the PSF size is kept within this ...
    psf_max_arcsec=3.2,         # ... and this range (arcsec)
    psf_default_arcsec=2.6,     # the LIFU fibre diameter
    max_listed=40,              # splits listed in the summary
)

# the default filter kernel of sep.extract (3x3 gaussian-like); sep measures moments and cflux on the image
# convolved with it, and so does `measure_pixels`
_KERNEL = np.array([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]])
_KERNEL = _KERNEL / _KERNEL.sum()
_KERNEL_NOISE = float(np.sqrt((_KERNEL ** 2).sum()))        # noise of the filtered image / noise of the image

_STRUCT8 = np.ones((3, 3), dtype=bool)


def parse_mode(value, default="auto"):
    """'auto' / 'fixed' (any case; None or '' gives `default`). Anything else is an error, not a silent fallback."""
    if value is None or str(value).strip() == "" or str(value).strip().lower() in ("none", "null"):
        return default
    v = str(value).strip().lower()
    if v not in DEBLEND_MODES:
        raise ValueError("seg2d_deblend_mode must be one of %s, got %r" % (", ".join(DEBLEND_MODES), value))
    return v


def filtered(data):
    """The image as sep.extract filters it for detection and moments (default kernel, zero padded)."""
    return ndimage.convolve(np.asarray(data, dtype=float), _KERNEL, mode="constant", cval=0.0)


def measure_pixels(ys, xs, data, conv, thresh, err, flag, dtype):
    """One row with the columns of a `sep.extract` object (same formulas as sep) for the pixels (ys, xs).

    Moments, centroid and cflux use the filtered image `conv`, flux and peak the image `data`; `x2`/`y2`
    get sep's 1/12 pixel-size term for degenerate shapes; errx2/erry2/errxy follow sep for a constant `err`.
    `dtype` is the dtype of the `objects` array to extend (keeps the columns and types of sep)."""
    row = np.zeros(1, dtype=dtype)
    v = data[ys, xs]
    c = conv[ys, xs]
    w = float(c.sum())
    if not w > 0:                                   # cannot happen for pixels above threshold; stay finite
        c = np.ones_like(c)
        w = float(c.sum())
    x = float((c * xs).sum() / w)
    y = float((c * ys).sum() / w)
    x2 = float((c * xs * xs).sum() / w - x * x)
    y2 = float((c * ys * ys).sum() / w - y * y)
    xy = float((c * xs * ys).sum() / w - x * y)
    errx2 = float(err ** 2 * ((xs - x) ** 2).sum() / w ** 2)
    erry2 = float(err ** 2 * ((ys - y) ** 2).sum() / w ** 2)
    errxy = float(err ** 2 * ((xs - x) * (ys - y)).sum() / w ** 2)
    det = x2 * y2 - xy * xy
    if det < 0.00694:                               # sep: add the variance of a pixel
        x2 += 1.0 / 12.0
        y2 += 1.0 / 12.0
        det = x2 * y2 - xy * xy
    half = 0.5 * (x2 + y2)
    rad = np.sqrt(0.25 * (x2 - y2) ** 2 + xy * xy)
    ip = int(np.argmax(v))
    ic = int(np.argmax(c))
    r = row[0]
    r["thresh"] = thresh
    r["npix"] = len(xs)
    r["tnpix"] = int((v > thresh).sum())
    r["xmin"], r["xmax"] = int(xs.min()), int(xs.max())
    r["ymin"], r["ymax"] = int(ys.min()), int(ys.max())
    r["x"], r["y"], r["x2"], r["y2"], r["xy"] = x, y, x2, y2, xy
    r["errx2"], r["erry2"], r["errxy"] = errx2, erry2, errxy
    r["a"] = np.sqrt(half + rad)
    r["b"] = np.sqrt(max(half - rad, 0.0))
    r["theta"] = 0.5 * np.arctan2(2.0 * xy, x2 - y2)
    r["cxx"], r["cyy"], r["cxy"] = y2 / det, x2 / det, -2.0 * xy / det
    r["cflux"] = float(c.sum()) if w > 0 else 0.0
    r["flux"] = float(v.sum())
    r["cpeak"], r["peak"] = float(conv[ys, xs][ic]), float(v[ip])
    r["xcpeak"], r["ycpeak"] = int(xs[ic]), int(ys[ic])
    r["xpeak"], r["ypeak"] = int(xs[ip]), int(ys[ip])
    r["flag"] = flag
    return row


def core_fwhm_px(conv_values):
    """Core size (pixels) of an object from the pixels of the filtered image that belong to it: the FWHM of
    the disc that has the area of the pixels above half of the peak. It does not grow with the brightness of
    the object (the isophotal size at a fixed threshold does: the wings of a bright star reach far), so it
    is the same for a faint and a bright star, and larger for a resolved galaxy core."""
    c = np.asarray(conv_values, float)
    n = int((c >= 0.5 * c.max()).sum())
    return float(2.0 * np.sqrt(n / np.pi))


def _core_sizes(objects, segmap, conv):
    """core FWHM and filtered peak of every object of `segmap` (label = row + 1)."""
    n = len(objects)
    core, peak = np.zeros(n), np.zeros(n)
    for i, sl in enumerate(ndimage.find_objects(segmap, max_label=n)):
        if sl is None:
            continue
        c = conv[sl][segmap[sl] == i + 1]
        if c.size:
            core[i], peak[i] = core_fwhm_px(c), float(c.max())
    return core, peak


def psf_size_px(core, peak, noise_f, pixscale_arcsec, p):
    """PSF size in pixels: the median core size of the detections that have a filtered peak above
    `psf_min_peak_sigma` x noise (their shape is not noise), kept within [psf_min_arcsec, psf_max_arcsec]
    (a LIFU fibre is 2.6 arcsec across: a field of large galaxies must not widen what counts as
    'point-like'). With fewer than `psf_min_sources` such detections the default (the fibre size) is used."""
    ok = peak >= p["psf_min_peak_sigma"] * noise_f
    if ok.sum() >= int(p["psf_min_sources"]):
        size = float(np.median(core[ok]))
    else:
        size = p["psf_default_arcsec"] / pixscale_arcsec
    lo, hi = p["psf_min_arcsec"] / pixscale_arcsec, p["psf_max_arcsec"] / pixscale_arcsec
    return float(min(max(size, lo), hi))


def crowding_gate(objects, core, peak, noise_f, psf_px, p):
    """(passed, n_compact): the field is crowded when at least `gate_min_compact` detections are compact
    (core size <= gate_compact_size x PSF), round (b/a >= gate_min_roundness) and measurable
    (filtered peak >= gate_min_peak_sigma x noise)."""
    if len(objects) == 0:
        return False, 0
    a, b = np.asarray(objects["a"], float), np.asarray(objects["b"], float)
    compact = ((core <= p["gate_compact_size"] * psf_px) & ((b / np.maximum(a, 1e-9)) >= p["gate_min_roundness"])
               & (peak >= p["gate_min_peak_sigma"] * noise_f))
    n = int(compact.sum())
    return n >= int(p["gate_min_compact"]), n


def _saddles(lab, conv, ids):
    """{(i, j): level} the highest pass between two children (8-connectivity): the larger of the smaller
    of the two filtered values over all touching pixel pairs."""
    out = {}
    h, w = lab.shape
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        ys0, ys1 = (0, h - dy)
        xs0, xs1 = (0, w - dx) if dx >= 0 else (-dx, w)
        a = lab[ys0:ys1, xs0:xs1]
        b = lab[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx]
        va = conv[ys0:ys1, xs0:xs1]
        vb = conv[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx]
        sel = (a > 0) & (b > 0) & (a != b)
        for la, lb, lv in zip(a[sel], b[sel], np.minimum(va[sel], vb[sel])):
            k = (int(min(la, lb)), int(max(la, lb)))
            if k not in out or lv > out[k]:
                out[k] = float(lv)
    return out


def _prominence_and_basins(ids, peak, saddles, significant=None):
    """Union-find over the saddles from the highest down.

    significant is None: return {child: prominence}, the peak minus the highest saddle that joins the child
    to a brighter one (the brightest child of a connected group has prominence = its peak).
    significant given: return {child: the real child it belongs to} where every child is merged into the real
    child it is joined to by the highest saddle (two groups that each hold a significant child never merge)."""
    parent = {i: i for i in ids}
    top = {i: i for i in ids}                      # group -> its brightest child
    has_sig = {i: (significant is not None and i in significant) for i in ids}
    owner = {i: i for i in ids if has_sig[i]}          # group -> its real child

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    prom = {i: float(peak[i]) for i in ids}
    for (i, j), lv in sorted(saddles.items(), key=lambda kv: -kv[1]):
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        if significant is not None and has_sig[ri] and has_sig[rj]:
            continue
        lo, hi = (ri, rj) if peak[top[ri]] < peak[top[rj]] else (rj, ri)
        prom[top[lo]] = float(peak[top[lo]]) - lv
        parent[lo] = hi
        if lo in owner and hi not in owner:
            owner[hi] = owner[lo]
        has_sig[hi] = has_sig[hi] or has_sig[lo]
    if significant is None:
        return prom
    return {i: owner.get(find(i)) for i in ids}


def _bins(a_px, b_px, pixscale_arcsec, radii_factor):
    """Number of 2 x 2 arcsec spatial bins the Galactic stage makes in the patch ellipse of an object: the patch
    has semi-axes a x radii_factor / 2 and b x radii_factor / 2 (A_world, B_world are the full axes), one
    bin is 4 arcsec^2, so bins = pi/4 (A/2)(B/2)/4 with A = a radii_factor px."""
    A = float(a_px) * radii_factor * pixscale_arcsec
    B = float(b_px) * radii_factor * pixscale_arcsec
    return float(np.pi / 4.0 * (0.5 * A) * (0.5 * B) / 4.0)


def _analyse_parent(i, sl, segmap, fine, data, conv, noise_f, psf, p):
    """Everything the decision needs about parent i: its fine children (restricted to the parent's pixels, the
    unlabelled pixels given to the nearest child), flux, peak, prominence, the real children and the share
    of the flux that is not in a real child (`diffuse`)."""
    inside = segmap[sl] == (i + 1)
    flab = np.where(inside, fine[sl], 0)
    ids = [int(k) for k in np.unique(flab) if k > 0]
    a = dict(i=i, sl=sl, inside=inside, ids=ids, sig=set(), diffuse=0.0, n_fine=len(ids))
    if len(ids) == 0:
        return a
    if (inside & (flab == 0)).any():
        idx = ndimage.distance_transform_edt(flab == 0, return_distances=False, return_indices=True)
        flab = np.where(inside, flab[tuple(idx)], 0)
    sub, cv = data[sl], conv[sl]
    flux = {k: float(sub[flab == k].sum()) for k in ids}
    peak = {k: float(cv[flab == k].max()) for k in ids}
    total = float(sum(flux.values()))
    a.update(flab=flab, flux=flux, peak=peak, total=total)
    if total <= 0:
        return a
    saddles = _saddles(flab, cv, ids)
    prom = _prominence_and_basins(ids, peak, saddles)
    sig = [k for k in sorted(ids, key=lambda k: -flux[k])
           if flux[k] >= p["child_min_flux_frac"] * total
           and prom[k] >= max(p["child_min_prominence"] * peak[k], p["child_min_prom_sigma"] * noise_f)]
    # two peaks closer than `min_separation` x PSF are one source: keep the brighter (fibre-to-fibre noise makes a
    # bright star look like several peaks, and a pair that close cannot be told from one star in this image)
    pos = {k: np.unravel_index(np.argmax(np.where(flab == k, cv, -np.inf)), flab.shape) for k in ids}
    diffuse = 1.0 - sum(flux[k] for k in sig) / total      # flux outside the real children
    kept = []
    for k in sig:
        if all(np.hypot(pos[k][0] - pos[m][0], pos[k][1] - pos[m][1]) >= p["min_separation"] * psf for m in kept):
            kept.append(k)
    a.update(saddles=saddles, sig=set(kept), diffuse=float(diffuse))
    return a


def auto_deblend(data, objects, segmap, *, ext_thresh, minarea, err, mask, pixscale_arcsec,
                 max_total=None, params=None, diag=None, radii_factor=8.0):
    """Crowd-adaptive split of the blends of `objects` / `segmap` (the result of the normal extraction).

    data: the background-subtracted image given to sep.extract; err: its noise (scalar); mask: True on
    blanks; pixscale_arcsec: pixel size; max_total: upper limit of the number of detections after the
    splits (the `seg2d_max_targets` cap; None or 0 = no limit): splits are made brightest parent first
    and a split that would exceed it is skipped, so a split never costs a faint detection to the cap.

    Returns (objects, segmap, info). When nothing is split the inputs are returned unchanged (same objects).
    `info` is JSON-ready: gate, counts and the list of splits. `diag`: an optional list that receives one
    dict per parent with two or more fine children (the numbers behind the decision, for validation)."""
    p = dict(AUTO_PARAMS)
    p.update(params or {})
    info = dict(mode="auto", gate_passed=False, n_compact=0, gate_min_compact=int(p["gate_min_compact"]),
                psf_size_px=None, psf_size_arcsec=None, n_parents=int(len(objects)), n_parents_candidates=0,
                n_parents_split=0, n_children=0, n_skipped_by_cap=0, n_rejected=dict(), splits=[])
    if len(objects) == 0:
        return objects, segmap, info
    conv = filtered(data)
    noise_f = err * _KERNEL_NOISE                    # noise of the filtered image
    core, peak0 = _core_sizes(objects, segmap, conv)
    psf = psf_size_px(core, peak0, noise_f, pixscale_arcsec, p)
    info["psf_size_px"] = round(psf, 3)
    info["psf_size_arcsec"] = round(psf * pixscale_arcsec, 3)
    passed, n_comp = crowding_gate(objects, core, peak0, noise_f, psf, p)
    info["gate_passed"], info["n_compact"] = bool(passed), int(n_comp)
    if not passed:
        return objects, segmap, info

    # fine pass: same image, threshold and minarea, finer deblending; only its partition into children is used
    _, fine = sep.extract(data, ext_thresh, minarea=minarea, deblend_nthresh=int(p["fine_nthresh"]),
                          deblend_cont=float(p["fine_cont"]), err=err, mask=mask, segmentation_map=True)
    slices = ndimage.find_objects(segmap, max_label=len(objects))
    an = {}
    for i, sl in enumerate(slices):
        if sl is not None:
            an[i] = _analyse_parent(i, sl, segmap, fine, data, conv, noise_f, psf, p)

    # extended objects: most of the flux lies outside the real children (a smooth object breaks into noise ripples)
    extended = set()
    for i, a in an.items():
        if a["n_fine"] >= p["extended_n_fine"] or (a["n_fine"] >= 2 and a["diffuse"] >= p["extended_diffuse"]):
            extended.add(i)

    rejected = dict(one_child=0, too_few_real=0, extended=0, diffuse=0, next_to_extended=0, child_too_small=0,
                    not_pointlike=0, dominant=0, no_saving=0)
    cand = []
    for i, a in an.items():
        dg = None
        if diag is not None and a["n_fine"] >= 2:
            dg = dict(parent=i, x=float(objects["x"][i]), y=float(objects["y"][i]), n_fine=a["n_fine"], decision="split",
                      parent_a_arcsec=round(float(objects["a"][i]) * pixscale_arcsec, 2), core_px=round(float(core[i]), 1),
                      n_real=len(a["sig"]), diffuse=round(float(a["diffuse"]), 3), extended=i in extended,
                      npix=int(objects["npix"][i]))
            diag.append(dg)

        def reject(why):
            rejected[why] += 1
            if dg is not None:
                dg["decision"] = why

        if a["n_fine"] < 2:
            rejected["one_child"] += 1
            continue
        if len(a["sig"]) < 2:
            reject("too_few_real")
            continue
        if i in extended:
            reject("extended")
            continue
        if a["diffuse"] > p["max_diffuse"]:
            reject("diffuse")
            continue
        # neighbours: a parent that touches an extended object is part of it (knots of a galaxy)
        sl = a["sl"]
        pad = int(p["touch_px"])
        sl2 = tuple(slice(max(s.start - pad, 0), s.stop + pad) for s in sl)
        grown = ndimage.binary_dilation(segmap[sl2] == (i + 1), structure=_STRUCT8, iterations=pad)
        touching = {int(k) - 1 for k in np.unique(segmap[sl2][grown]) if k > 0 and k != i + 1}
        if touching & extended:
            reject("next_to_extended")
            continue

        # real children + their fragments (pixels of the others go to the real child they join by the
        # highest saddle), then measure those and check they are point-like and none dominates
        flab, ids, sig = a["flab"], a["ids"], a["sig"]
        group = _prominence_and_basins(ids, a["peak"], a["saddles"], significant=sig)
        glab = np.zeros_like(flab)
        for k in ids:
            glab[flab == k] = group[k] or 0
        keep = np.isin(glab, list(sig))
        if (a["inside"] & ~keep).any():                 # fragments cut off from every real child: nearest one
            idx = ndimage.distance_transform_edt(~keep, return_distances=False, return_indices=True)
            glab = np.where(a["inside"], glab[tuple(idx)], 0)
        gids = sorted(int(k) for k in np.unique(glab) if k > 0)
        cv = conv[sl]
        rows, sizes, fluxes = [], [], []
        for g in gids:
            ys, xs = np.nonzero(glab == g)
            row = measure_pixels(ys + sl[0].start, xs + sl[1].start, data, conv, float(objects["thresh"][i]),
                                 err, int(objects["flag"][i]), objects.dtype)
            rows.append(row)
            sizes.append(core_fwhm_px(cv[glab == g]))
            fluxes.append(float(row["flux"][0]))
        if dg is not None:
            dg["core_children_px"] = [round(v, 1) for v in sizes]
            dg["final_flux_frac"] = [round(v / sum(fluxes), 3) for v in fluxes]
        if min(float(r["npix"][0]) for r in rows) < p["min_child_area"] * np.pi * (psf / 2.0) ** 2:
            reject("child_too_small")
            continue
        if max(sizes) > p["child_max_size"] * psf:
            reject("not_pointlike")
            continue
        if max(fluxes) > p["dominant_flux_frac"] * sum(fluxes):
            reject("dominant")
            continue
        # it must pay: the work of the Galactic stage grows with the ellipse area (a x b), so a split that
        # does not cut it clearly is not worth any risk
        cost_parent = _bins(objects["a"][i], objects["b"][i], pixscale_arcsec, radii_factor)
        cost_children = float(sum(_bins(r["a"][0], r["b"][0], pixscale_arcsec, radii_factor) for r in rows))
        saved = cost_parent - cost_children
        if dg is not None:
            dg["bins_parent"], dg["bins_saved"] = round(cost_parent, 1), round(saved, 1)
        if saved < p["min_bins_saved"] or cost_children > (1.0 - p["min_saving"]) * cost_parent:
            reject("no_saving")
            continue
        cand.append((i, rows, glab, sl, gids))
    info["n_rejected"] = rejected
    info["n_parents_candidates"] = int(len(cand))

    # the cap: split the brightest parents first and never go over max_total
    budget = None
    if max_total:
        budget = int(max_total) - len(objects)
    order = sorted(range(len(cand)), key=lambda k: -float(objects["flux"][cand[k][0]]))
    accepted = {}
    for k in order:
        i, rows, glab, sl, gids = cand[k]
        extra = len(rows) - 1
        if budget is not None and extra > budget:
            info["n_skipped_by_cap"] += 1
            continue
        if budget is not None:
            budget -= extra
        accepted[i] = cand[k]
    if not accepted:
        return objects, segmap, info

    # rebuild objects and segmap with consistent labels (label = row index + 1, rows in the old order)
    new_rows = []
    new_seg = np.zeros_like(segmap)
    remap = {}
    for i in range(len(objects)):
        if i in accepted:
            _, rows, glab, sl, gids = accepted[i]
            for row, g in sorted(zip(rows, gids), key=lambda t: -float(t[0]["flux"][0])):
                new_rows.append(row)
                new_seg[sl][glab == g] = len(new_rows)
        else:
            new_rows.append(objects[i:i + 1])
            remap[i + 1] = len(new_rows)
    lut = np.zeros(int(segmap.max()) + 1, dtype=segmap.dtype)
    for old, new in remap.items():
        lut[old] = new
    new_seg = np.where(np.isin(segmap, list(remap)), lut[segmap], new_seg).astype(segmap.dtype)
    new_objects = np.concatenate(new_rows).astype(objects.dtype)

    info["n_parents_split"] = int(len(accepted))
    info["n_children"] = int(sum(len(v[1]) for v in accepted.values()))
    info["splits"] = [dict(x=round(float(objects["x"][i]), 1), y=round(float(objects["y"][i]), 1),
                           a_arcsec=round(float(objects["a"][i]) * pixscale_arcsec, 2), n_children=len(accepted[i][1]))
                      for i in sorted(accepted, key=lambda t: -float(objects["flux"][t]))[: int(p["max_listed"])]]
    return new_objects, new_seg, info
