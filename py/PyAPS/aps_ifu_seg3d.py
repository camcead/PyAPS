"""
aps_ifu_seg3d.py
=================

Optional 3D matched-filter emission-line source detection for WEAVE IFU
L1 stackcube files -- a companion to the 2D white-light-collapse +
SExtractor detection (``ifu_seg2d`` in aps_ifu_prepare.py), NOT a
replacement. seg2d collapses the cube to a single white-light image and
runs sep on it, which is the right tool for continuum sources but
structurally cannot find a source whose significance is concentrated in a
handful of spectral channels (an emission line, or a feature visible in
only part of the bandpass): summing over thousands of channels dilutes
such a signal by roughly sqrt(n_channels) while adding n_channels worth of
noise, so an 8-sigma detection in 5 channels can be a ~1-sigma bump in a
4000-channel collapse.

This module runs the LSDCat/CubEx/ORIGIN family of methods instead:
continuum removal -> ivar noise recalibration -> a bank of 3D matched
filters (Gaussian spatial PSF x Gaussian spectral line profile) ->
3D connected-component candidate extraction -> a veto chain -> an
empirical purity self-check via the sign-flipped cube.

Everything here was iteratively validated against a real WEAVE LIFU cube
(stackcube_3067087.fit, BLUE arm) before being wired in. Two findings from
that validation are baked into the defaults and are worth knowing before
changing them:

  - ivar noise recalibration is the single most important step. The L1
    ivar cube underestimated the true per-pixel noise by a factor of
    ~3.2x in sigma (~10x in variance) on the test cube -- confirmed via
    the tail-inclusive (16th/84th percentile) spread of the pull
    distribution (residual/sigma_ivar), NOT a sigma-clipped std (iterative
    3-sigma clipping discards the very tail this needs to measure, and
    gave a falsely reassuring ~1.08x). Skipping this step reproduces the
    original failure mode: purity goes *negative* and gets *worse* at
    higher SNR threshold, because real (if unremarkable) noise
    fluctuations cross nominal thresholds far too easily.
  - PCA/eigenspectrum continuum subtraction (ORIGIN-style) is needed, not
    a running-mean/median high-pass filter: a real stellar absorption or
    emission line is a narrow spectral *feature*, not continuum curvature,
    and no amount of smoothing removes it. A low-rank eigenspectrum basis
    built from the field's own spaxels captures line shapes that recur
    across many continuum-bearing spaxels (even ones too faint for the sep
    broadband catalog) while leaving genuinely rare, spaxel-unique
    emission lines in the residual.

Even with both fixes, purity plateaus around 0.6-0.8 rather than reaching
1 (see post_veto_purity_scan / the returned 'purity_scan' entry) -- that
residual contamination needs a different class of tool (injection-recovery
testing, or a trained classifier on the candidate list, as ORIGIN/MUSE-Wide
do) rather than more threshold tuning, and is not attempted here.

This module is opt-in end to end: ``ifu_worker_prepare(seg3d=True, ...)``
runs it *alongside* whatever seg2d/classification already do, and it never
touches the patch_file / patch_table / classification pipeline. The
default (``seg3d=False``) leaves current behaviour completely unchanged.

Scope note: the comparison figure's "2D" panel is this module's own
internal sep pass on its own white-light collapse, not a re-use of
whatever the sibling seg2d run may have computed on a possibly different
(reprojected/PS1) grid -- deliberately, to avoid silently mixing two
different pixel grids. It is still a fair, like-for-like comparison of
"2D continuum detection" vs "3D matched-filter detection" on the same data.

Cube loading (load_wavelength_range) is built directly on aps_utils.APSOB
-- PyAPS's own established, tested L1-reading pipeline (sensitivity
correction, gap masking, safe_mask_gaps, tellurics, and join_arms with its
mask_bad_overlap arm-overlap handling), the same mechanism every other
aps_ifu_*.py script uses, rather than a parallel hand-rolled reader. The
raster (nwave, ny, nx) cube shape a spatial matched filter needs is
reconstructed from APSOB's per-spaxel output via the same APS_ID-raster
pattern aps_make_joint_cube.py already established for exactly this
purpose (build an APS_ID grid matching the cube's pixel shape, then place
each spaxel's processed spectrum at its (y, x) position).
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import sep
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table, hstack, vstack
from scipy import ndimage

from PyAPS.aps_utils import APSOB

FWHM_SIGMA = 2.3548200450309493  # 2*sqrt(2*ln2)


# ----------------------------------------------------------------------
# Cube I/O
# ----------------------------------------------------------------------

@dataclass
class Cube:
    data: np.ndarray      # (nwave, ny, nx)
    ivar: np.ndarray      # (nwave, ny, nx)
    header: fits.Header
    wave: np.ndarray      # (nwave,)
    collapse: np.ndarray  # (ny, nx) white-light image, from the L1 COLLAPSE3 ext
    collapse_ivar: np.ndarray


def _native_wave_range(fname: str) -> tuple[float, float]:
    """A file's own (unclipped) wavelength coverage, from its AWAV WCS
    (CRVAL3/CD3_3/NAXIS3 on the same HDU 1 that load_wavelength_range
    already reads for pixel_shape) -- no APSOB/L1-reader round trip needed
    just to answer "does this arm even reach into [lmin, lmax]"."""
    h = fits.getheader(fname, 1)
    lo = h["CRVAL3"]
    hi = h["CRVAL3"] + h["NAXIS3"] * h["CD3_3"]
    return (lo, hi) if lo <= hi else (hi, lo)


def load_wavelength_range(infiles: list[str], lmin: float, lmax: float, *,
                           sens_corr: bool = True, mask_gaps: bool = True,
                           safe_mask_gaps: bool = True, tellurics: bool = False,
                           vacuum: bool = False, arms_ratio: list[float] | None = None,
                           crr: bool = False) -> Cube:
    """Load exactly the data covering [lmin, lmax] Angstrom, built directly
    on aps_utils.APSOB -- PyAPS's own established, tested L1-reading
    pipeline (the same one every other aps_ifu_*.py script uses for real
    science-grade extraction), not a parallel hand-rolled FITS reader.

    infiles can be one file or several (whatever was already passed to the
    surrounding aps_ifu_prepare.py run -- this does not go hunting for
    companion files on disk). Before touching APSOB at all, each file's own
    native wavelength range (from its header, see _native_wave_range) is
    clipped to [lmin, lmax] and only kept if it contributes a meaningful
    amount of EXCLUSIVE coverage beyond the other file(s) already kept (see
    the note above _native_wave_range's call site for why this can't be
    left to APSOB itself), and join_arms is enabled only when more than one
    file survives that filter -- generic over 1, 2, or more arms, nothing
    here is hardcoded to a blue/red pair. When join_arms does run, that's
    also where APSOB's own mask_bad_overlap handles the physical overlap
    between arms (WEAVE blue/red overlap by ~145A) by comparing median
    flux in the shared region, rather than this module guessing
    which arm should "win" a boundary.

    sens_corr/mask_gaps/safe_mask_gaps/tellurics/vacuum/arms_ratio/crr are
    passed straight through to APSOB -- see its docstring for each; these
    are the SAME flags (--sens_corr, --mask_gaps, --safe_mask_gaps,
    --tellurics, --vacuum) aps_ifu_prepare.py already accepts on its CLI
    for the rest of the pipeline, reused here rather than inventing
    parallel seg3d-specific settings. Skipping safe_mask_gaps/mask_gaps
    risks reporting fake line-only "detections" that are really just
    unflagged CCD-gap residuals -- keep both on unless you have a specific
    reason not to.

    The raster (nwave, ny, nx) shape a spatial matched filter needs is not
    APSOB's native output (a flat list of per-spaxel spectra) -- it's
    reconstructed via the same APS_ID-grid pattern aps_make_joint_cube.py
    already established for exactly this purpose: build an APS_ID array
    matching the reference file's pixel_shape, then place each spaxel's
    processed spectrum at its (y, x) position, using idfx() for O(1)
    aps_id -> spaxel lookup (the make_joint_cube.py reference implementation
    uses a slow `id in targs.id()` list-membership check per spaxel;
    idfx() avoids that here).

    White-light collapse: each file actually used by APSOB (targs.infiles())
    contributes its own COLLAPSE3/IVAR extension (already the full native-
    bandpass image for that file, independent of lmin/lmax), combined with
    the same flux-sum/variance-add convention bin_cube_spatial uses -- more
    bandpass is better for the continuum catalog's own S/N regardless of
    how narrow a line-search window was requested. Read directly from the
    raw files (not through APSOB, which doesn't process this extension).
    """
    # APSOB does NOT reliably drop a file whose native coverage doesn't
    # reach [lmin, lmax] before join_arms runs. Worse, a file can carry a
    # genuine (if thin) *nominal* overlap with [lmin, lmax] and still be
    # entirely redundant once clipped -- e.g. WEAVE's blue/red arms overlap
    # by ~145A around the arm boundary, but if [lmin, lmax] itself already
    # ends inside (or before) that boundary, the blue arm alone already
    # covers the FULL requested window and the red arm's clipped slice
    # contributes zero NEW wavelengths beyond it. join_arms' overlap-region
    # search then looks for red-arm wavelengths above blue's clipped max,
    # finds none, and crashes on an empty array. So: decide which files are
    # actually worth joining ourselves, by how much EXCLUSIVE clipped
    # coverage each contributes beyond what's already kept -- not just
    # whether its native range nominally touches [lmin, lmax] -- before
    # APSOB/join_arms ever sees them.
    MIN_EXCLUSIVE_COVERAGE_A = 20.0  # a real second arm contributes far more than this within any sane seg3d window; a redundant sliver (the failure mode above) contributes ~0

    clipped = []
    for i, f in enumerate(infiles):
        lo, hi = _native_wave_range(f)
        clo, chi = max(lo, lmin), min(hi, lmax)
        if chi > clo:
            clipped.append((i, f, clo, chi))
    if not clipped:
        raise ValueError(
            f"None of {list(infiles)} overlap the requested wavelength range "
            f"[{lmin:.1f}, {lmax:.1f}]A"
        )
    # widest clipped range first, so the "primary" arm is whichever one
    # actually carries the bulk of the requested window -- this ordering is
    # only used to decide what to keep; the original infiles order (and
    # arms_ratio's alignment to it) is preserved below.
    by_width = sorted(clipped, key=lambda c: c[3] - c[2], reverse=True)

    kept_idx = {by_width[0][0]}
    covered_lo, covered_hi = by_width[0][2], by_width[0][3]
    for i, f, clo, chi in by_width[1:]:
        exclusive = max(0.0, chi - covered_hi) + max(0.0, covered_lo - clo)
        if exclusive >= MIN_EXCLUSIVE_COVERAGE_A:
            kept_idx.add(i)
            covered_lo, covered_hi = min(covered_lo, clo), max(covered_hi, chi)
        else:
            print(f"aps_ifu_seg3d: dropping {f} -- its clipped range [{clo:.1f}, {chi:.1f}]A "
                  f"contributes only {exclusive:.1f}A beyond what [{covered_lo:.1f}, "
                  f"{covered_hi:.1f}]A already covers within the requested "
                  f"[{lmin:.1f}, {lmax:.1f}]A window -- too little to safely join_arms")

    overlapping = [f for i, f in enumerate(infiles) if i in kept_idx]
    overlapping_arms_ratio = (
        [r for i, r in enumerate(arms_ratio) if i in kept_idx] if arms_ratio is not None else None
    )

    join_arms = len(overlapping) > 1
    wlranges = [[lmin, lmax]] * len(overlapping)

    targs = APSOB(
        overlapping, wlranges=wlranges, sens_corr=sens_corr, mask_gaps=mask_gaps,
        safe_mask_gaps=safe_mask_gaps, tellurics=tellurics, vacuum=vacuum,
        arms_ratio=overlapping_arms_ratio, join_arms=join_arms, crr=crr, skysub=True, collapse=False,
    )

    used_infiles = list(targs.infiles())
    print(f"aps_ifu_seg3d: APSOB loaded [{lmin:.1f}, {lmax:.1f}]A from {used_infiles} "
          f"(join_arms={join_arms}, sens_corr={sens_corr}, mask_gaps={mask_gaps}, "
          f"safe_mask_gaps={safe_mask_gaps})")

    ref_file = used_infiles[0]
    ref_header = fits.getheader(ref_file, 1)
    nx, ny = WCS(ref_header).pixel_shape[:2]  # FITS axis order: [NAXIS1, NAXIS2, NAXIS3]

    wave = np.asarray(targs.wavelist()[0], dtype=np.float64)
    nwave = len(wave)

    data = np.zeros((nwave, ny, nx))
    ivar = np.zeros((nwave, ny, nx))

    aps_id_grid = np.arange(nx * ny).reshape(ny, nx) + 1
    idfx = targs.idfx()
    targetlist = targs.data()
    n_filled = 0
    for iy in range(ny):
        for ix in range(nx):
            idx = idfx.get(int(aps_id_grid[iy, ix]))
            if idx is None:
                continue
            spec = targetlist[idx].spectra[0]
            data[:, iy, ix] = spec.flux
            ivar[:, iy, ix] = spec.ivar
            n_filled += 1
    print(f"aps_ifu_seg3d: filled {n_filled}/{nx * ny} spaxels from APSOB output")

    ivar = np.where(np.isfinite(ivar) & (ivar > 0), ivar, 0.0)
    data = np.where(np.isfinite(data), data, 0.0)

    collapse_list = [fits.getdata(f, 6).astype(np.float64) for f in used_infiles]
    collapse_ivar_list = [fits.getdata(f, 7).astype(np.float64) for f in used_infiles]
    collapse = sum(collapse_list)
    var_collapse = sum(np.divide(1.0, civ, out=np.zeros_like(civ), where=civ > 0)
                        for civ in collapse_ivar_list)
    collapse_ivar = np.divide(1.0, var_collapse, out=np.zeros_like(var_collapse), where=var_collapse > 0)

    return Cube(data, ivar, ref_header, wave, collapse, collapse_ivar)


def rebin_wcs(wcs: WCS, bin_factor: int) -> WCS:
    """WCS for a block-binned image (bin_factor x bin_factor, starting at
    pixel 0) -- standard CRPIX/CD rescaling, exact for the no-rotation case
    (true for WEAVE's white-light images) and used here only for
    diagnostic-figure tick labels/hover, not astrometry."""
    if bin_factor <= 1:
        return wcs
    out = deepcopy(wcs)
    out.wcs.crpix = (np.array(out.wcs.crpix) - 0.5) / bin_factor + 0.5
    if out.wcs.has_cd():
        out.wcs.cd = np.array(out.wcs.cd) * bin_factor
    else:
        out.wcs.cdelt = np.array(out.wcs.cdelt) * bin_factor
    return out


# ----------------------------------------------------------------------
# Preprocessing: wavelength masking + spatial binning
# ----------------------------------------------------------------------

def wavelength_mask(wave: np.ndarray, sky_dominated_min_A: float | None = None) -> np.ndarray:
    """Static, physics-informed bad-wavelength mask (True = usable):
    optionally excludes everything redward of sky_dominated_min_A (OH
    airglow forest dominates there -- relevant if lmax reaches into the
    RED arm, e.g. pass 8000; None=off). Complementary to coincidence_veto,
    which is data-driven and only sees problems visible in this cube's own
    statistics -- this is the static list of known-bad regions applied up
    front, standard practice in MUSE/WEAVE-style pipelines.

    No longer trims a fixed margin off the loaded data's own edges (an
    earlier edge_trim_A parameter did this) -- load_wavelength_range's
    lmin/lmax ARE that margin now: the caller picks bounds with enough
    clearance from an arm's true edge themselves, rather than the pipeline
    auto-trimming a fixed amount off of whatever was loaded.
    """
    if sky_dominated_min_A is None:
        return np.ones_like(wave, dtype=bool)
    return wave < sky_dominated_min_A


def apply_wavelength_mask(ivar: np.ndarray, good_wave: np.ndarray) -> np.ndarray:
    """Zero ivar at all spaxels for every masked wavelength channel."""
    out = ivar.copy()
    out[~good_wave, :, :] = 0.0
    return out


def bin_cube_spatial(data: np.ndarray, ivar: np.ndarray, bin_factor: int,
                      min_valid_frac: float = 0.8):
    """Flux-conserving spatial rebinning by bin_factor x bin_factor blocks.

    Flux is summed (correct for an additive quantity spread over more
    pixels). ivar is combined as if the input spaxels were independent
    (variance-add then invert) -- the same convention PyAPS already uses in
    ExGalPrepare.spatial_bin_with_provenance (bin_noise = sqrt(sum(noise**2))),
    just for a rectangular-grid cube instead of that function's flattened
    spaxel-list representation (which this module can't reuse directly: the
    3D matched filter needs a proper raster grid for spatial convolution and
    connected-component labeling, not a possibly-gapped flat bin list).
    Known to be only approximately correct on a drizzled/oversampled cube
    (adjacent native spaxels are correlated), but this is the standard
    first-order treatment, and the pipeline's own purity self-check
    (post_veto_purity_scan) is re-run post-binning to verify empirically
    rather than assume it's exact. Bins with fewer than min_valid_frac of
    their sub-pixels unmasked are themselves masked (ivar=0) rather than
    extrapolated from a partial footprint (relevant at the field-of-view
    edge/footprint boundary).

    min_valid_frac default raised from an earlier 0.5 to 0.8: on real WEAVE
    data, a 4x4 bin at the oval/hex footprint boundary that was only ~50%
    covered still got treated as a normal bin, and several of the highest-
    SNR matched-filter "candidates" turned out to sit exactly 1 binned
    pixel from the coverage edge (see edge_distance_veto) -- a stricter
    floor keeps genuinely marginal-coverage bins out of the detection grid
    entirely instead of letting them in with reduced but nonzero weight.
    """
    nwave, ny, nx = data.shape
    ny2, nx2 = ny - (ny % bin_factor), nx - (nx % bin_factor)
    if ny2 != ny or nx2 != nx:
        data = data[:, :ny2, :nx2]
        ivar = ivar[:, :ny2, :nx2]
    mask = ivar > 0
    var = np.where(mask, np.divide(1.0, ivar, out=np.zeros_like(ivar), where=mask), 0.0)

    shape5 = (nwave, ny2 // bin_factor, bin_factor, nx2 // bin_factor, bin_factor)
    d5 = np.where(mask, data, 0.0).reshape(shape5)
    v5 = var.reshape(shape5)
    m5 = mask.reshape(shape5)

    flux_binned = d5.sum(axis=(2, 4))
    var_binned = v5.sum(axis=(2, 4))
    npix_binned = m5.sum(axis=(2, 4))

    good = npix_binned >= min_valid_frac * bin_factor * bin_factor
    ivar_binned = np.where(good & (var_binned > 0),
                            np.divide(1.0, var_binned, out=np.zeros_like(var_binned), where=var_binned > 0),
                            0.0)
    flux_binned = np.where(good, flux_binned, 0.0)
    return flux_binned, ivar_binned


def bin_image_spatial(image: np.ndarray, bin_factor: int) -> np.ndarray:
    """Same block-sum rebinning as bin_cube_spatial, for the 2D white-light
    collapse image used by the sep continuum catalog."""
    if bin_factor <= 1:
        return image
    ny, nx = image.shape
    ny2, nx2 = ny - (ny % bin_factor), nx - (nx % bin_factor)
    img = np.nan_to_num(image[:ny2, :nx2])
    return img.reshape(ny2 // bin_factor, bin_factor, nx2 // bin_factor, bin_factor).sum(axis=(1, 3))


# ----------------------------------------------------------------------
# ivar noise recalibration (the dominant real-data fix, see module docstring)
# ----------------------------------------------------------------------

def calibrate_ivar_scale(residual: np.ndarray, ivar: np.ndarray,
                          low_pct: float = 16.0, high_pct: float = 84.0) -> float:
    """Global multiplicative noise-underestimation factor k, such that
    ivar/k**2 makes the pull distribution (residual/sigma_ivar) have unit
    width. Uses the robust (16th, 84th) percentile spread of the pull
    distribution rather than a sigma-clipped std -- see module docstring
    for why that distinction matters (it made a ~3x difference in the
    measured factor on real WEAVE data). (16, 84) percentiles bracket
    exactly +-1sigma for a true Gaussian, so the ratio needs no further
    scaling beyond the standard-normal quantile difference.
    """
    from scipy.stats import norm
    good = ivar > 0
    sigma_ivar = np.zeros_like(ivar)
    sigma_ivar[good] = 1.0 / np.sqrt(ivar[good])
    pull = residual[good] / sigma_ivar[good]
    p_lo, p_hi = np.percentile(pull, [low_pct, high_pct])
    z_lo, z_hi = norm.ppf(low_pct / 100.0), norm.ppf(high_pct / 100.0)
    return (p_hi - p_lo) / (z_hi - z_lo)


def calibrate_ivar_scale_per_wave(residual: np.ndarray, ivar: np.ndarray,
                                   low_pct: float = 16.0, high_pct: float = 84.0,
                                   smooth_window: int = 51, min_good: int = 50) -> np.ndarray:
    """Same idea as calibrate_ivar_scale, but k(wavelength) instead of one
    global number: real noise miscalibration in a spectrograph is typically
    wavelength-dependent (sky brightness, detector response, sensitivity-
    function errors all vary with lambda). On the validated test cube this
    gave essentially the same purity as the global factor (k ranged only
    1.0-3.5, median 3.2, close to flat) -- kept as the default anyway since
    it can only help on a cube where the miscalibration genuinely varies
    with wavelength, and costs little extra. Vectorized across all spaxels
    at once per wavelength channel via nanpercentile (bad pixels -> NaN)
    rather than a Python loop over thousands of channels; smoothed
    afterward since a single-channel percentile estimate is noisier than
    the whole-cube one. Never returns k < 1 (only correct underestimated
    noise, don't let ivar be shrunk below what it already claims).
    """
    from scipy.stats import norm
    nwave = residual.shape[0]
    good = ivar > 0
    sigma_ivar = np.full(ivar.shape, np.nan)
    sigma_ivar[good] = 1.0 / np.sqrt(ivar[good])
    pull = residual / sigma_ivar  # NaN where masked
    pull_flat = pull.reshape(nwave, -1)

    n_good = np.sum(~np.isnan(pull_flat), axis=1)
    with np.errstate(invalid="ignore"):
        p_lo, p_hi = np.nanpercentile(pull_flat, [low_pct, high_pct], axis=1)
    z_lo, z_hi = norm.ppf(low_pct / 100.0), norm.ppf(high_pct / 100.0)
    k = (p_hi - p_lo) / (z_hi - z_lo)
    k = np.where(n_good >= min_good, k, 1.0)
    k = np.clip(np.nan_to_num(k, nan=1.0), 1.0, None)

    if smooth_window > 1:
        k = ndimage.uniform_filter1d(k, size=smooth_window, mode="nearest")
    return k


def normalize_ivar(ivar: np.ndarray, window: int = 5, cap_factor: float = 3.0) -> np.ndarray:
    """Despike the ivar cube along the spectral axis: caps a pixel's ivar
    at cap_factor times the local (small-window) running-mean ivar around
    it in wavelength, so one miscalibrated/over-confident pixel can't
    dominate the matched-filter weighting for that voxel. On the validated
    test cube this was a measured no-op (0.000% of voxels capped, identical
    purity with/without) -- kept as a cheap defense-in-depth safeguard for
    cubes/fibers where it would matter, not a fix for anything specific.
    """
    local_mean = ndimage.uniform_filter1d(ivar, size=window, axis=0, mode="nearest")
    return np.minimum(ivar, cap_factor * local_mean)


# ----------------------------------------------------------------------
# Continuum removal
# ----------------------------------------------------------------------

def continuum_subtract(data: np.ndarray, window: int = 151) -> np.ndarray:
    """Running-mean high-pass filter along the spectral axis. Fast, but
    cannot remove a real absorption/emission line (a narrow feature, not
    continuum curvature) -- see continuum_subtract_pca, the recommended
    default. Kept for comparison/fallback only."""
    continuum = ndimage.uniform_filter1d(data, size=window, axis=0, mode="nearest")
    return data - continuum


def continuum_subtract_robust(data: np.ndarray, window: int = 151, niter: int = 2,
                               clip_sigma: float = 4.0) -> np.ndarray:
    """Sigma-clipped iterative running-mean high-pass filter: fit a running
    mean, mask pixels deviating from it by more than clip_sigma (a
    symmetric clip -- an earlier asymmetric lo/hi version was found to
    systematically bias the continuum estimate, see aps_ifu_seg3d dev
    history), refit over the unmasked pixels, repeat. Still cannot remove
    a real line SHAPE the way continuum_subtract_pca can (validated: gave
    essentially the same purity as the plain running mean on real WEAVE
    data) -- kept as a middle-ground option, not the default.
    """
    mask = np.ones_like(data, dtype=bool)
    cont = np.zeros_like(data)
    for _ in range(niter):
        num = ndimage.uniform_filter1d(np.where(mask, data, 0.0), size=window, axis=0, mode="nearest")
        den = ndimage.uniform_filter1d(mask.astype(np.float64), size=window, axis=0, mode="nearest")
        cont = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        resid = data - cont
        med = np.median(resid, axis=0, keepdims=True)
        mad = np.median(np.abs(resid - med), axis=0, keepdims=True)
        sigma = 1.4826 * mad + 1e-6
        mask = np.abs(resid - med) < clip_sigma * sigma
    return data - cont


def continuum_subtract_pca(data: np.ndarray, ivar: np.ndarray, good_wave: np.ndarray,
                            n_components: int = 15, niter: int = 2,
                            clip_sigma: float = 4.0) -> np.ndarray:
    """ORIGIN-style eigenspectrum continuum subtraction -- the validated
    default (see module docstring for why continuum_subtract/_robust are
    not enough). Build a low-rank eigenspectrum basis (SVD) from the
    field's own spaxels: a real absorption-line spectral shape recurs
    across many continuum-bearing spaxels (even ones too faint for the sep
    broadband catalog), so it's captured by the leading few principal
    components; a genuine narrow emission line is essentially unique to a
    handful of spaxels, so it is NOT captured by a small number of
    components and survives into the residual.

    Implementation: flatten to a (n_spaxel, n_wave_good) matrix (wavelength
    channels already excluded by wavelength_mask are dropped, not fit),
    subtract the mean spectrum, take the top n_components singular vectors
    (scipy.sparse.linalg.svds, so only the requested rank is computed
    rather than a full SVD), reconstruct the continuum from that low-rank
    basis, and iterate with sigma-clipping so line cores don't bias the
    basis they're about to be measured against.
    """
    from scipy.sparse.linalg import svds

    nwave, ny, nx = data.shape
    nspax = ny * nx
    M_full = data.reshape(nwave, nspax).T  # (nspax, nwave)
    wave_idx = np.where(good_wave)[0]
    M = M_full[:, wave_idx].copy()  # (nspax, nwave_good)

    clip_mask = ivar.reshape(nwave, nspax).T[:, wave_idx] > 0
    recon = np.zeros_like(M)
    for _ in range(niter):
        counts = clip_mask.sum(axis=0)
        mean_spec = np.divide(np.where(clip_mask, M, 0.0).sum(axis=0), counts,
                               out=np.zeros(M.shape[1]), where=counts > 0)
        centered = np.where(clip_mask, M - mean_spec[None, :], 0.0)

        k = max(1, min(n_components, min(centered.shape) - 1))
        U, S, Vt = svds(centered, k=k)
        recon = mean_spec[None, :] + (U * S) @ Vt

        resid = M - recon
        med = np.median(resid, axis=1, keepdims=True)
        mad = np.median(np.abs(resid - med), axis=1, keepdims=True)
        sigma = 1.4826 * mad + 1e-6
        clip_mask = (resid > med - clip_sigma * sigma) & (resid < med + clip_sigma * sigma)

    continuum_full = np.zeros_like(M_full)
    continuum_full[:, wave_idx] = recon
    return (M_full - continuum_full).T.reshape(nwave, ny, nx)


# ----------------------------------------------------------------------
# Separable 3D matched filter
# ----------------------------------------------------------------------

def _gauss_kernel_1d(sigma_pix: float, truncate: float = 4.0) -> np.ndarray:
    radius = max(1, int(truncate * sigma_pix + 0.5))
    x = np.arange(-radius, radius + 1)
    return np.exp(-0.5 * (x / sigma_pix) ** 2)


def matched_filter_snr(residual: np.ndarray, ivar: np.ndarray, spatial_sigma_pix: float,
                        spectral_sigma_pix: float, min_coverage_frac: float = 0.5) -> np.ndarray:
    """Exact matched-filter SNR cube for a separable Gaussian(spatial) x
    Gaussian(spectral) template, ivar-weighted:

        a_hat = sum(w*d*k) / sum(w*k^2)   (amplitude estimate)
        SNR   = sum(w*d*k) / sqrt(sum(w*k^2))

    Both sums are separable 3D convolutions, implemented as three
    sequential 1D convolutions per term instead of one heavy 3D one.

    min_coverage_frac guards against the division-blowup right at the edge
    of the cube's unmasked footprint (WEAVE LIFU's oval/hex footprint
    boundary, or the wavelength-mask edges): where most neighbours within
    the kernel support have ivar==0, D -> 0 while N does not shrink
    proportionally, producing spuriously huge SNR that is a coverage
    artifact, not a detection. SNR is zeroed wherever the local D falls
    below min_coverage_frac * the typical (median) D in well-covered voxels.
    """
    k_spec = _gauss_kernel_1d(spectral_sigma_pix)
    k_spat = _gauss_kernel_1d(spatial_sigma_pix)
    k_spec2, k_spat2 = k_spec ** 2, k_spat ** 2

    wd, w = ivar * residual, ivar

    def conv1(arr, k, axis):
        return ndimage.convolve1d(arr, k, axis=axis, mode="constant", cval=0.0)

    N = conv1(conv1(conv1(wd, k_spec, 0), k_spat, 1), k_spat, 2)
    D = conv1(conv1(conv1(w, k_spec2, 0), k_spat2, 1), k_spat2, 2)
    D = np.clip(D, 0, None)

    good = D > 0
    d_typical = np.median(D[good]) if np.any(good) else 0.0
    well_covered = good & (D >= min_coverage_frac * d_typical)

    snr = np.zeros_like(N)
    snr[well_covered] = N[well_covered] / np.sqrt(D[well_covered])
    return snr


# ----------------------------------------------------------------------
# 3D connected-component candidate extraction
# ----------------------------------------------------------------------

@dataclass
class Candidate:
    iwave: int
    iy: int
    ix: int
    wave: float
    snr: float
    npix: int
    template_fwhm_pix: float = 0.0


def extract_candidates(snr_cube: np.ndarray, wave: np.ndarray, threshold: float,
                        template_fwhm_pix: float = 0.0) -> list[Candidate]:
    mask = snr_cube > threshold
    structure = np.ones((3, 3, 3), dtype=bool)  # full 26-connectivity
    labels, nlab = ndimage.label(mask, structure=structure)
    if nlab == 0:
        return []
    peak_pos = ndimage.maximum_position(snr_cube, labels=labels, index=np.arange(1, nlab + 1))
    sizes = ndimage.sum(mask, labels=labels, index=np.arange(1, nlab + 1))
    cands = []
    for (iw, iy, ix), npix in zip(peak_pos, sizes):
        cands.append(Candidate(iwave=int(iw), iy=int(iy), ix=int(ix), wave=float(wave[int(iw)]),
                                snr=float(snr_cube[iw, iy, ix]), npix=int(npix),
                                template_fwhm_pix=template_fwhm_pix))
    return cands


def dedupe_candidates(cands: list[Candidate], dxy: int = 3, dw: int = 4) -> list[Candidate]:
    """Merge candidates from different bank widths / neighbouring voxels
    that plausibly refer to the same physical blob; keep the highest-SNR
    one."""
    cands = sorted(cands, key=lambda c: -c.snr)
    kept: list[Candidate] = []
    for c in cands:
        if any(abs(c.ix - k.ix) <= dxy and abs(c.iy - k.iy) <= dxy and abs(c.iwave - k.iwave) <= dw
               for k in kept):
            continue
        kept.append(c)
    return kept


def bank_candidates_from_cubes(snr_cubes, fwhms_pix, wave, threshold) -> list[Candidate]:
    """Extract + dedupe candidates from pre-computed per-width SNR cubes."""
    all_cands: list[Candidate] = []
    for snr_cube, fwhm in zip(snr_cubes, fwhms_pix):
        all_cands.extend(extract_candidates(snr_cube, wave, threshold, template_fwhm_pix=fwhm))
    return dedupe_candidates(all_cands)


def run_width_bank(residual, ivar, wave, spatial_sigma_pix, spectral_sigmas_pix, threshold) -> list[Candidate]:
    """LSDCat-style bank of spectral widths; keep the best (max SNR)
    detection per spatial/spectral neighbourhood across the bank."""
    snr_cubes = [matched_filter_snr(residual, ivar, spatial_sigma_pix, s) for s in spectral_sigmas_pix]
    fwhms = [s * FWHM_SIGMA for s in spectral_sigmas_pix]
    return bank_candidates_from_cubes(snr_cubes, fwhms, wave, threshold)


# ----------------------------------------------------------------------
# Vetoes: reject artifacts before they reach the final catalog
# ----------------------------------------------------------------------

def resolvedness_filter(cands: list[Candidate], residual: np.ndarray,
                         min_frac: float = 0.3) -> list[Candidate]:
    """Line-width veto: a real line, convolved with the template width that
    triggered it, cannot be narrower than that template -- the raw
    (unfiltered) residual at the +-1 spectral pixel neighbours of the peak
    must retain at least min_frac of the amplitude a Gaussian of the
    claimed template sigma would predict. A candidate that is essentially a
    single-pixel spike fails this regardless of bank width, and is
    rejected as unresolved (single miscalibrated pixel / unflagged cosmic
    ray)."""
    nwave = residual.shape[0]
    kept = []
    for c in cands:
        sigma = c.template_fwhm_pix / FWHM_SIGMA if c.template_fwhm_pix > 0 else 1.0
        expected_ratio = np.exp(-0.5 * (1.0 / sigma) ** 2)
        peak = residual[c.iwave, c.iy, c.ix]
        if peak == 0:
            continue
        neighbours = []
        if c.iwave - 1 >= 0:
            neighbours.append(residual[c.iwave - 1, c.iy, c.ix])
        if c.iwave + 1 < nwave:
            neighbours.append(residual[c.iwave + 1, c.iy, c.ix])
        if not neighbours:
            continue
        if np.mean(neighbours) / peak >= min_frac * expected_ratio:
            kept.append(c)
    return kept


def spatial_extent_veto(cands: list[Candidate], max_npix: int = 5000,
                         min_npix: int = 3) -> list[Candidate]:
    """Compactness veto, both directions:

    max_npix -- a real point-like emission-line source's thresholded
    matched-filter blob occupies a volume set by the template (spatial PSF
    footprint x spectral line width) -- at most a few thousand voxels for
    the widths used here. npix is heavily right-skewed on real data
    (validated: median 28, p99=2035, p99.9=11363, max=826193) -- the small
    population of enormous blobs is extended sky-residual/bad-CCD-region
    artifacts. Because those are spatially contiguous, 3D connected-
    component labeling already merges each into a *single* blob, which is
    exactly why coincidence_veto (counting independent candidates per
    wavelength bin) cannot see them -- this half of the veto is the direct
    complement to that one.

    min_npix -- the opposite failure mode: a blob of just 1-2 voxels above
    threshold is, by construction, narrower than the matched-filter
    template's own kernel support (which spans several voxels in both the
    spatial and spectral directions) -- a genuine, resolved detection of
    that template cannot produce a footprint that small. A 1-voxel blob
    surviving to this point is far more likely a single anomalously-
    weighted pixel than a real spatially/spectrally resolved feature.
    Default min_npix=3 is a low, conservative floor (well below the
    median-28 typical size) that only excludes the most degenerate single-
    or two-voxel cases, not real compact sources.
    """
    return [c for c in cands if min_npix <= c.npix <= max_npix]


def coincidence_veto(cands: list[Candidate], wave: np.ndarray, wave_tol_pix: int = 2,
                      mad_k: float = 6.0, min_count: int = 8):
    """Data-driven bad-wavelength mask: a real, independent source
    population scattered across the field should not pile up at one
    wavelength. Bin candidates by wavelength (coarse bins of
    +-wave_tol_pix), flag any bin whose count is a robust outlier
    (median + mad_k*MAD, floor min_count) as instrumental (sky-subtraction
    residual, bad CCD column/region, ghost, ...) -- the same logic
    LSDCat/CubEx apply via a static skyline mask, but built empirically
    from the candidate list itself."""
    if not cands:
        return cands, np.array([], dtype=bool)
    iwaves = np.array([c.iwave for c in cands])
    nwave = len(wave)
    binw = 2 * wave_tol_pix + 1
    bins = iwaves // binw
    nbins = nwave // binw + 1
    counts = np.bincount(bins, minlength=nbins)
    nonzero = counts[counts > 0]
    med = np.median(nonzero)
    mad = np.median(np.abs(nonzero - med)) or 1.0
    thresh = max(min_count, med + mad_k * 1.4826 * mad)
    bad_bins = np.where(counts > thresh)[0]
    bad_bin_set = set(bad_bins.tolist())
    kept = [c for c in cands if (c.iwave // binw) not in bad_bin_set]
    bad_waves = wave[np.clip(bad_bins * binw, 0, nwave - 1)]
    return kept, bad_waves


def edge_distance_veto(cands: list[Candidate], good_bins: np.ndarray,
                        min_dist_px: float = 3.0) -> list[Candidate]:
    """Reject candidates within min_dist_px (Euclidean, detection-grid
    pixels) of the nearest edge of the spatial coverage mask.

    The spatial matched-filter kernel has a finite support radius (~2
    binned pixels at this module's defaults); a candidate whose peak sits
    within that radius of the coverage-mask boundary has necessarily lost
    real kernel support on one side. min_coverage_frac in
    matched_filter_snr does not always catch this cleanly in practice --
    confirmed on real WEAVE data: several of the highest-SNR "candidates"
    sat exactly 1 binned pixel from the edge (SNR 37-38, ranked #2 and #3
    overall), and 33% of all candidates were within 2 binned pixels of it.
    This is the direct, easy-to-verify complement to that guard.

    good_bins: 2D (ny, nx) boolean coverage mask of the detection grid,
    e.g. np.any(ivar > 0, axis=0) on the (already spatially binned) ivar
    cube -- reflects exactly which bins survived bin_cube_spatial's
    min_valid_frac gate.
    """
    if not cands:
        return cands
    edt = ndimage.distance_transform_edt(good_bins)
    ny, nx = good_bins.shape
    kept = []
    for c in cands:
        if 0 <= c.iy < ny and 0 <= c.ix < nx and edt[c.iy, c.ix] >= min_dist_px:
            kept.append(c)
    return kept


# ----------------------------------------------------------------------
# Empirical purity via negative-side (sign-flipped) detections
# ----------------------------------------------------------------------

def catalog_at_threshold(snr_cubes, fwhms_pix, residual, wave, threshold,
                          resolvedness_min_frac, max_npix, coincidence_min_count,
                          good_bins=None, min_edge_dist_px=3.0, min_npix=3) -> list[Candidate]:
    """The full veto chain (bank -> dedupe -> resolvedness -> compactness ->
    coincidence -> edge-distance), re-thresholding pre-computed SNR cubes
    instead of recomputing the matched filter -- lets many thresholds be
    scanned for the price of one matched-filter pass per bank width.
    good_bins=None skips the edge-distance veto (e.g. for callers without
    a coverage mask handy)."""
    cands = bank_candidates_from_cubes(snr_cubes, fwhms_pix, wave, threshold)
    cands = resolvedness_filter(cands, residual, min_frac=resolvedness_min_frac)
    cands = spatial_extent_veto(cands, max_npix=max_npix, min_npix=min_npix)
    cands, _ = coincidence_veto(cands, wave, min_count=coincidence_min_count)
    if good_bins is not None:
        cands = edge_distance_veto(cands, good_bins, min_dist_px=min_edge_dist_px)
    return cands


def post_veto_purity_scan(snr_pos, snr_neg, fwhms, residual, wave, thresholds,
                           resolvedness_min_frac, max_npix, coincidence_min_count,
                           good_bins=None, min_edge_dist_px=3.0, min_npix=3):
    """Purity of the FINAL (post-veto) catalog vs SNR threshold -- the
    object actually shipped as the catalog, both signs, using pre-computed
    SNR cubes so the expensive matched-filter convolution runs once per
    bank width regardless of how many thresholds are scanned. Returns a
    list of (threshold, n_pos, n_neg, purity) tuples."""
    results = []
    for t in thresholds:
        pos = catalog_at_threshold(snr_pos, fwhms, residual, wave, t,
                                    resolvedness_min_frac, max_npix, coincidence_min_count,
                                    good_bins=good_bins, min_edge_dist_px=min_edge_dist_px,
                                    min_npix=min_npix)
        neg = catalog_at_threshold(snr_neg, fwhms, -residual, wave, t,
                                    resolvedness_min_frac, max_npix, coincidence_min_count,
                                    good_bins=good_bins, min_edge_dist_px=min_edge_dist_px,
                                    min_npix=min_npix)
        purity = 1 - len(neg) / max(len(pos), 1)
        results.append((float(t), len(pos), len(neg), purity))
    return results


# ----------------------------------------------------------------------
# Data-driven seg2d (2D SExtractor) parameter estimation.
#
# Standalone -- operates purely on a white-light image, independent of the
# rest of this module's 3D machinery. Replaces guessed ext_thresh/minarea
# values with two measurements from the data itself:
#   minarea    <- the image's own PSF size (measured from its own compact,
#                 round, well-detected sources), not an arbitrary pixel count.
#   ext_thresh <- the same empirical purity calibration (sign-flipped image,
#                 count spurious detections) this module already uses for
#                 the 3D matched-filter search, applied here to sep instead.
# Motivated by a very concrete real-data failure: on the WEAVE LIFU cube
# this module was validated against, the pre-existing "recommended"
# default minarea=300 found only 3 objects on a field where minarea=20
# found 51 and minarea=10 found 57 on the identical image -- 300 pixels is
# ~75 sq.arcsec at 0.5"/pix, far larger than any compact source's real
# footprint, and was simply too large a guess, not a principled choice.
# ----------------------------------------------------------------------

def estimate_psf_fwhm_pix(image: np.ndarray, ext_thresh: float = 1.5, minarea: int = 5,
                           bw: int = 64, bh: int = 64, roundness_min: float = 0.6,
                           n_use: int = 20):
    """Measure the image's own PSF FWHM (pixels) from its most compact,
    round, well-detected sources -- instead of trusting a header seeing
    keyword (found unreliable/ambiguous-units on real WEAVE L1 files) or
    guessing.

    Runs a permissive sep pass (low ext_thresh, small minarea) to get a
    raw candidate list including real compact sources, keeps the
    roundest ones (b/a >= roundness_min, excludes elongated/blended
    objects), takes the n_use most compact of those by isophotal area
    (npix) -- extended sources bias a PSF estimate upward -- and reports
    the median of 2.3548*sqrt(a*b) (the standard SExtractor second-moment
    FWHM convention) over that subset.

    Returns (fwhm_pix, n_sources_used, raw_objects). fwhm_pix is None if
    fewer than 3 suitable sources were found (falls back to a caller-
    supplied default in that case).
    """
    data = image.copy()
    data[data == 0] = np.nan
    mask = np.isnan(data)
    d = np.ascontiguousarray(np.nan_to_num(data), dtype="<f8")
    bw_eff, bh_eff = max(4, min(bw, d.shape[1] // 2)), max(4, min(bh, d.shape[0] // 2))
    bkg = sep.Background(d, mask=mask, bw=bw_eff, bh=bh_eff)
    d_sub = d - bkg
    objects = sep.extract(d_sub, ext_thresh, minarea=minarea, err=bkg.globalrms, mask=mask)

    if len(objects) == 0:
        return None, 0, objects

    roundness = np.divide(objects["b"], objects["a"], out=np.zeros(len(objects)),
                           where=objects["a"] > 0)
    round_enough = roundness >= roundness_min
    candidates = objects[round_enough]
    if len(candidates) < 3:
        return None, len(candidates), objects

    order = np.argsort(candidates["npix"])[:n_use]  # most compact first
    compact = candidates[order]
    fwhm_per_source = FWHM_SIGMA * np.sqrt(compact["a"] * compact["b"])
    return float(np.median(fwhm_per_source)), len(compact), objects


def calibrate_sep_purity(image: np.ndarray, minarea: int, ext_thresh_grid=None,
                          bw: int = 64, bh: int = 64, purity_target: float = 0.9):
    """Empirical ext_thresh calibration via the sign-flipped image, at a
    grid of trial thresholds, holding minarea fixed at a given (already
    PSF-derived) value: exactly the same sign-flip purity philosophy
    post_veto_purity_scan uses for the 3D matched filter, applied here to
    plain 2D sep extraction. Returns (recommended_ext_thresh, curve),
    where curve is a list of (ext_thresh, n_pos, n_neg, purity) and
    recommended_ext_thresh is the SMALLEST (most complete/permissive)
    threshold in the grid whose purity still clears purity_target -- i.e.
    the most sources you can keep without giving up on the purity bar,
    rather than an arbitrary fixed sigma cut.
    """
    if ext_thresh_grid is None:
        ext_thresh_grid = [1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0]

    data = image.copy()
    data[data == 0] = np.nan
    mask = np.isnan(data)
    d = np.ascontiguousarray(np.nan_to_num(data), dtype="<f8")
    bw_eff, bh_eff = max(4, min(bw, d.shape[1] // 2)), max(4, min(bh, d.shape[0] // 2))
    bkg = sep.Background(d, mask=mask, bw=bw_eff, bh=bh_eff)
    d_sub = d - bkg

    curve = []
    for t in ext_thresh_grid:
        pos = sep.extract(d_sub, t, minarea=minarea, err=bkg.globalrms, mask=mask)
        neg = sep.extract(-d_sub, t, minarea=minarea, err=bkg.globalrms, mask=mask)
        n_pos, n_neg = len(pos), len(neg)
        purity = 1 - n_neg / max(n_pos, 1)
        curve.append((float(t), n_pos, n_neg, purity))

    passing = [t for t, _, _, p in curve if p >= purity_target]
    recommended = min(passing) if passing else max(ext_thresh_grid)
    return recommended, curve


def estimate_seg2d_params(image: np.ndarray, minarea_factor: float = 1.0,
                           purity_target: float = 0.9, ext_thresh_grid=None,
                           default_fwhm_pix: float = 4.0, verbose: bool = True):
    """Top-level data-driven seg2d parameter estimate: measure the PSF from
    the image, derive minarea from it, then calibrate ext_thresh by
    empirical purity at that minarea. See estimate_psf_fwhm_pix and
    calibrate_sep_purity for the two measurements this combines.

    minarea = minarea_factor * pi * (fwhm_pix/2)^2 -- the pixel area of a
    circle the width of one measured PSF FWHM; minarea_factor=1.0 requires
    a detection to span at least a full PSF disk to count (a standard,
    principled SExtractor convention, not a guess). Raise minarea_factor
    for a more conservative (fewer, more certain) catalog, lower it for a
    more complete/permissive one.

    Falls back to default_fwhm_pix if too few compact sources were found
    to measure the PSF (e.g. a very sparse or very crowded field).

    Returns a dict: fwhm_pix, fwhm_source, minarea, ext_thresh,
    purity_curve, n_psf_sources_used.
    """
    fwhm_pix, n_used, _ = estimate_psf_fwhm_pix(image)
    fwhm_source = "measured"
    if fwhm_pix is None:
        fwhm_pix = default_fwhm_pix
        fwhm_source = "default (too few compact sources to measure)"

    minarea = max(3, int(round(minarea_factor * np.pi * (fwhm_pix / 2.0) ** 2)))
    ext_thresh, curve = calibrate_sep_purity(image, minarea, ext_thresh_grid=ext_thresh_grid,
                                              purity_target=purity_target)

    if verbose:
        print(f"estimate_seg2d_params: PSF FWHM={fwhm_pix:.2f}px ({fwhm_source}, "
              f"from {n_used} compact sources) -> minarea={minarea}")
        print(f"  ext_thresh purity calibration (target purity>={purity_target}):")
        for t, n_pos, n_neg, p in curve:
            flag = " <- recommended" if t == ext_thresh else ""
            print(f"    ext_thresh={t:4.1f}: n_pos={n_pos:4d} n_neg={n_neg:4d} "
                  f"purity={p:.3f}{flag}")

    return dict(fwhm_pix=fwhm_pix, fwhm_source=fwhm_source, minarea=minarea,
                ext_thresh=ext_thresh, purity_curve=curve, n_psf_sources_used=n_used)


# ----------------------------------------------------------------------
# Continuum (sep) catalog + cross-match
# ----------------------------------------------------------------------

def sep_catalog(collapse: np.ndarray, ext_thresh: float = 2.5, minarea: int = 20,
                 bw: int = 32, bh: int = 32):
    """bw/bh/minarea default values are tuned for the native (unbinned)
    white-light image; on a spatially-binned image the caller should scale
    them down (run_seg3d does this automatically) or the background mesh
    barely resolves any structure and minarea can demand more connected
    pixels than a real, now much smaller, point source spans -- silently
    under-detecting continuum sources, which matters because
    bright_source_mask depends on this catalog being complete."""
    data = collapse.copy()
    data[data == 0] = np.nan
    mask = np.isnan(data)
    data = np.ascontiguousarray(np.nan_to_num(data), dtype="<f8")
    bw = max(4, min(bw, data.shape[1] // 2))
    bh = max(4, min(bh, data.shape[0] // 2))
    bkg = sep.Background(data, mask=mask, bw=bw, bh=bh)
    data_sub = data - bkg
    return sep.extract(data_sub, ext_thresh, minarea=minarea, err=bkg.globalrms, mask=mask)


def bright_source_mask(shape_yx: tuple[int, int], objects, radius_factor: float = 3.0,
                        min_radius: float = 5.0) -> np.ndarray:
    """Spatial mask (True = excluded) around every cataloged continuum
    source. A continuum filter -- smoothed, sigma-clipped, or even
    eigenspectrum-based -- can track a star's slowly-varying continuum,
    but can never remove that star's OWN absorption/emission lines from
    the residual (those are real narrow spectral features, not continuum
    curvature); on real WEAVE data these dominated the false-candidate
    list by orders of magnitude in SNR until masked out. This is the
    standard fix used by ORIGIN/LSDCat: exclude continuum-bright spaxels
    from the emission-line search entirely rather than try to model their
    spectra away."""
    ny, nx = shape_yx
    mask = np.zeros((ny, nx), dtype=bool)
    if len(objects) == 0:
        return mask
    yy, xx = np.mgrid[0:ny, 0:nx]
    for obj in objects:
        r = max(min_radius, radius_factor * np.sqrt(max(obj["a"] * obj["b"], 0.0)))
        mask |= (np.hypot(xx - obj["x"], yy - obj["y"]) <= r)
    return mask


def flag_line_only(cands: list[Candidate], objects, radius_pix: float = 4.0):
    """Mark matched-filter candidates with no nearby continuum (sep)
    source. Returns a list of (Candidate, is_new) pairs."""
    if len(objects) == 0:
        return [(c, True) for c in cands]
    obj_x, obj_y = objects["x"], objects["y"]
    out = []
    for c in cands:
        d = np.hypot(obj_x - c.ix, obj_y - c.iy)
        out.append((c, bool(np.min(d) > radius_pix)))
    return out


def group_multiline_candidates(cands: list[Candidate], radius_px: float = 2.0) -> np.ndarray:
    """Group post-veto candidates by spatial proximity alone (single-linkage
    / friends-of-friends within radius_px in the detection-grid pixel
    scale), independent of wavelength.

    Why wavelength is deliberately ignored here: extract_candidates
    connected-component-labels each matched-filter blob as its own
    Candidate, and two real emission lines from the SAME physical source
    (e.g. Hbeta and [OIII]5007, or [OIII] and Halpha) are typically many
    resolution elements apart in wavelength -- they can never be joined by
    the 3D connected-component step that produces individual Candidates in
    the first place. Confirmed on real data: of 127 post-veto candidates in
    the field this module was validated against, 7 spatial positions carry
    2+ independently-detected lines each (one position carries 4), which
    extract_candidates necessarily reports as 4 separate, spatially
    coincident Candidates -- this function is what recognises that as one
    source rather than 4 unrelated ones.

    radius_px default (0.0, i.e. the exact same detection-grid pixel --
    already "one aperture/spatial-bin unit" since ix/iy live on the binned
    grid) is deliberately exact, not a tolerance. Checked empirically
    against the sign-flipped (noise) cube on the validated test field: even
    a 1-pixel tolerance let a materially higher fraction of pure-noise
    candidates fall into spurious "multi-line" groups (55% of negative
    candidates at radius=1, vs 19% at radius=0) without a compensating gain
    in real multi-line groups recovered -- there was no accuracy benefit
    to the wider net, only more false groups. Widening this from the exact-
    pixel default is not recommended without re-running that same check.

    Returns a 0-indexed group_id array, same length/order as cands. A
    candidate with no spatial neighbours within radius_px gets its own
    singleton group (no special-casing needed downstream: "group size" is
    just the group's member count either way).
    """
    n = len(cands)
    if n == 0:
        return np.array([], dtype=int)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    xs = np.array([c.ix for c in cands], dtype=float)
    ys = np.array([c.iy for c in cands], dtype=float)
    r2 = radius_px ** 2
    for i in range(n):
        for j in range(i + 1, n):
            if (xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2 <= r2:
                union(i, j)

    roots = np.array([find(i) for i in range(n)])
    _, group_id = np.unique(roots, return_inverse=True)
    return group_id


def combined_group_snr(snrs) -> float:
    """Combined detection significance of several independent lines at the
    same source (quadrature sum, i.e. sqrt of the sum of squares).

    Under the null hypothesis (pure noise, properly calibrated so each
    line's matched-filter SNR is ~unit-variance), N independent such
    statistics sum-of-squares to ~chi^2 with N degrees of freedom; its
    square root is the natural combined test statistic -- the same spirit
    as the GLR-style combined significance MUSE's ORIGIN uses to merge a
    source's individual line evidence, and mathematically the same
    construction as combining independent chi-by-eye detections in any
    matched-filter search. Deliberately uses EVERY group member, not just
    the strongest one: a group of several weak-but-real lines should be
    able to out-rank a single marginal line, which max()-based scoring
    cannot express (two lines at SNR=7 combine to a combined SNR of ~9.9,
    for instance -- meaningfully more significant than either alone)."""
    arr = np.asarray(snrs, dtype=float)
    return float(np.sqrt(np.sum(arr ** 2)))


def group_candidates(cands: list[Candidate], radius_px: float = 0.0,
                      min_lines: int = 2) -> list[tuple[int, list[Candidate]]]:
    """Group cands via group_multiline_candidates and return only the
    groups with >= min_lines members, as (group_id, members) pairs."""
    if not cands:
        return []
    group_id = group_multiline_candidates(cands, radius_px=radius_px)
    groups: dict[int, list[Candidate]] = {}
    for gid, c in zip(group_id, cands):
        groups.setdefault(int(gid), []).append(c)
    return [(gid, members) for gid, members in groups.items() if len(members) >= min_lines]


def post_veto_group_purity_scan(cands_pos: list[Candidate], cands_neg: list[Candidate],
                                 thresholds: list[float], radius_px: float = 0.0,
                                 min_lines: int = 2):
    """Purity of MULTI-LINE GROUPS vs a combined-SNR (see combined_group_snr)
    threshold, using the same real-cube-vs-sign-flipped-cube comparison
    post_veto_purity_scan already uses for single candidates -- groups need
    their OWN calibration rather than reusing the single-candidate one
    (checked empirically: a criterion validated for single detections is
    not automatically valid for a spatial-coincidence criterion, which has
    a completely different false-positive mechanism -- see
    merge_into_patch_table's docstring).

    cands_pos/cands_neg must already be the FULL post-veto candidate lists
    at the pipeline's one primary detection threshold (what run_seg3d calls
    `cands`, and its sign-flipped-cube equivalent) -- NOT re-extracted per
    threshold the way post_veto_purity_scan re-runs catalog_at_threshold
    for single candidates. That re-extraction matters there because
    raising the SNR-cube threshold changes which voxels connected-component
    label into a blob at all; it does not apply here, since `threshold`
    only ever filters each group's already-fixed combined SNR, computed
    from lines that were already independently extracted and vetted once,
    at one fixed primary threshold.

    Returns a list of (threshold, n_pos_groups, n_neg_groups, purity)
    tuples, the same shape as post_veto_purity_scan's return value, so
    resolve_group_min_snr can reuse resolve_merge_min_snr's exact logic.
    """
    groups_pos = group_candidates(cands_pos, radius_px=radius_px, min_lines=min_lines)
    groups_neg = group_candidates(cands_neg, radius_px=radius_px, min_lines=min_lines)
    combined_pos = np.array([combined_group_snr([c.snr for c in members]) for _, members in groups_pos])
    combined_neg = np.array([combined_group_snr([c.snr for c in members]) for _, members in groups_neg])
    results = []
    for t in thresholds:
        n_pos = int(np.sum(combined_pos >= t))
        n_neg = int(np.sum(combined_neg >= t))
        purity = 1 - n_neg / max(n_pos, 1)
        results.append((float(t), n_pos, n_neg, purity))
    return results


def resolve_group_min_snr(group_purity_scan, min_snr_floor: float = 10.0,
                           purity_target: float = 0.8, min_n_trust: int = 3):
    """resolve_merge_min_snr's exact logic, applied to a group_purity_scan
    instead of a single-candidate purity_scan. min_n_trust defaults much
    lower here (3, vs 15 for single candidates) because multi-line GROUPS
    are intrinsically rarer than individual candidates by construction --
    demanding 15 trustworthy groups on a field that may only ever produce
    a handful would make this path unusable in practice. This is a real,
    explicit trade-off (a purity ratio computed from 3-4 groups carries
    more sampling noise than one from 15 candidates) rather than a
    like-for-like match to the single-candidate default -- documented here
    rather than silently reusing 15 and disabling the path on most fields.
    """
    if not group_purity_scan:
        return min_snr_floor, "no group_purity_scan given, using min_snr_floor as-is"
    trustworthy = [(t, p) for t, n_pos, n_neg, p in group_purity_scan if n_pos >= min_n_trust]
    passing = [t for t, p in trustworthy if p >= purity_target]
    if not passing:
        return None, (f"no combined-SNR threshold in the group purity scan both clears "
                       f"purity>={purity_target} and has n_pos>={min_n_trust} groups to trust "
                       f"that measurement -- disabling the multi-line path for this field")
    calibrated = min(passing)
    effective = calibrated if min_snr_floor is None else max(min_snr_floor, calibrated)
    return effective, (f"group purity-calibrated threshold={calibrated} (purity>={purity_target} "
                        f"with n_pos>={min_n_trust} groups), floor={min_snr_floor} -> using {effective}")


def candidates_to_table(flagged: list[tuple[Candidate, bool]], wcs: WCS,
                         group_radius_px: float = 0.0) -> Table:
    """Build the on-disk candidate catalog (Astropy Table) from a
    flag_line_only() result. Positions are in the (possibly spatially
    binned) pixel grid used for detection; RA/Dec come from that same
    grid's WCS (see rebin_wcs).

    Also runs group_multiline_candidates() over ALL candidates (regardless
    of continuum-counterpart status) and adds two columns from it:
    group_id (0-indexed, shared by spatially coincident candidates) and
    n_group_lines (that group's total member count) -- so a source with
    several detected lines is visible as such directly in the saved
    catalog, not just inferred later. merge_into_patch_table uses these to
    decide multi-line promotion; see its docstring.
    """
    if not flagged:
        return Table(names=["id", "ix", "iy", "RA_icrs", "DEC_icrs", "wave_A", "snr", "npix",
                             "fwhm_template_pix", "no_continuum_counterpart",
                             "group_id", "n_group_lines"],
                     dtype=[int, int, int, float, float, float, float, int, float, bool, int, int])
    xs = np.array([c.ix for c, _ in flagged], dtype=float)
    ys = np.array([c.iy for c, _ in flagged], dtype=float)
    sky = wcs.pixel_to_world(xs, ys)
    group_id = group_multiline_candidates([c for c, _ in flagged], radius_px=group_radius_px)
    n_group_lines = np.bincount(group_id)[group_id]
    tbl = Table()
    tbl["id"] = np.arange(1, len(flagged) + 1)
    tbl["ix"] = [c.ix for c, _ in flagged]
    tbl["iy"] = [c.iy for c, _ in flagged]
    tbl["RA_icrs"] = sky.ra.deg
    tbl["DEC_icrs"] = sky.dec.deg
    tbl["wave_A"] = [c.wave for c, _ in flagged]
    tbl["snr"] = [c.snr for c, _ in flagged]
    tbl["npix"] = [c.npix for c, _ in flagged]
    tbl["fwhm_template_pix"] = [c.template_fwhm_pix for c, _ in flagged]
    tbl["no_continuum_counterpart"] = [is_new for _, is_new in flagged]
    tbl["group_id"] = group_id.astype(int)
    tbl["n_group_lines"] = n_group_lines.astype(int)
    tbl.sort("snr", reverse=True)
    return tbl


# ----------------------------------------------------------------------
# Merging "obvious" seg3d candidates into the real (seg2d) patch table
# ----------------------------------------------------------------------

# patch_table['flag'] is always 0.0 for every row written by
# ifu_seg2d.create_aps_targets() and, as of this writing, is never read
# back from the physical patch_table anywhere in the live pipeline --
# checked every production consumer (aps_ifu_ExGal.py, aps_ifu_Gal.py,
# IFUExGalPrepare.py, MOSExGalPrepare.py, aps_cubepreview.py); the only
# `['flag'] == 0` gate that looks related (update_patch_table(), line
# ~347) reads `targX['flag']` from index_in_class_patch()'s return value,
# a *different* per-classgroup match indicator with the same field name,
# not this column. It is therefore safe to repurpose as a provenance
# marker without affecting any current pipeline logic -- confirmed by
# direct code inspection, not assumption, given the stakes of getting this
# wrong (this table drives real classification/analysis downstream).
SEG3D_PROVENANCE_FLAG = 3.0


def resolve_merge_min_snr(purity_scan, min_snr_floor: float = 10.0,
                           purity_target: float = 0.8, min_n_trust: int = 15):
    """Pick the SNR threshold actually used by merge_into_patch_table from
    the cube's OWN purity_scan (see run_seg3d), instead of trusting one
    fixed number for every field -- the same "most permissive threshold
    that still clears the purity bar" principle
    aps_ifu_seg3d.calibrate_sep_purity already uses for seg2d's ext_thresh,
    applied here to the merge decision.

    Only scan entries with n_pos >= min_n_trust are considered: a purity
    ratio from a handful of candidates (e.g. n_pos=4, n_neg=1 -> "0.75") is
    not a trustworthy measurement, however good it looks -- on the
    validated test cube, purity swung noisily between 0.25 and 0.91 at
    thresholds above SNR~10 purely because n_pos had dropped into the
    single-to-low-double digits there; a naive fixed SNR>=10 rule would
    have been trusting exactly that noise.

    Returns (effective_min_snr, reason_str). effective_min_snr is
    max(min_snr_floor, the smallest sufficiently-well-sampled threshold
    whose purity >= purity_target) -- the floor can only be raised, never
    lowered, by this calibration. If no scanned threshold both clears
    purity_target AND has enough candidates to trust, returns
    (None, reason) -- signalling the caller to skip merging for this field
    entirely rather than fall back to an uncalibrated guess.
    """
    if not purity_scan:
        return min_snr_floor, "no purity_scan given, using min_snr_floor as-is"

    trustworthy = [(t, p) for t, n_pos, n_neg, p in purity_scan if n_pos >= min_n_trust]
    passing = [t for t, p in trustworthy if p >= purity_target]
    if not passing:
        return None, (f"no SNR threshold in the purity scan both clears purity>={purity_target} "
                       f"and has n_pos>={min_n_trust} candidates to trust that measurement -- "
                       f"disabling merge for this field rather than guessing")
    calibrated = min(passing)
    effective = max(min_snr_floor, calibrated)
    return effective, (f"purity-calibrated threshold={calibrated} (purity>={purity_target} with "
                        f"n_pos>={min_n_trust}), floor={min_snr_floor} -> using {effective}")


def loudest_event_snr(snr_cubes) -> float:
    """The single loudest (highest) matched-filter S/N ANY voxel of ANY
    template width in `snr_cubes` ever reached -- no threshold, veto, or
    candidate extraction applied first, just the raw maximum over the
    entire cube stack. Intended to be called on the SIGN-FLIPPED cubes
    (run_seg3d's snr_neg), giving the loudest fluctuation the full noise
    search ever produced across every independent (spaxel, wavelength,
    template-width) trial -- millions of them for a typical field.

    This is the standard fallback once a ratio-based background estimate
    (post_veto_purity_scan / post_veto_group_purity_scan) runs out of
    samples at high threshold: the same "loudest event" statistic used to
    validate extreme-significance gravitational-wave detections once no
    background trial set is large enough to measure a p-value directly at
    the candidate's own significance. See resolve_extreme_snr_min, which
    uses this value to gate a genuinely extreme single-line (ungrouped)
    candidate that the ratio-based single-line path cannot calibrate high
    enough to reach -- not a replacement for that path, a fallback for
    when it structurally cannot answer at all (every scanned threshold has
    n_pos too small to trust, all the way up to the candidate's own S/N).
    """
    return float(max(np.nanmax(c) for c in snr_cubes))


def resolve_extreme_snr_min(loudest_neg_event, safety_margin: float = 1.1):
    """Effective S/N threshold for the "extreme single-line" merge path:
    loudest_neg_event (see loudest_event_snr, called on the sign-flipped
    cubes) times safety_margin.

    Deliberately NOT a ratio/purity calibration like resolve_merge_min_snr
    / resolve_group_min_snr -- there is only one number here (the loudest
    noise fluctuation the ENTIRE sign-flipped search ever produced), not a
    scan, because this path exists precisely for the regime where a
    ratio-based estimate has no statistical power left (n_pos too small at
    every threshold that high). safety_margin>1.0 requires clearing that
    single loudest event by a margin, not just matching it -- 1.1 (10%) is
    a pragmatic, round default consistent with the margin used in earlier,
    ad hoc applications of this same idea (a floor of 65.0 against a
    loudest event of 60.71, ~1.07x), not a calibrated false-alarm rate.
    Tighten it (larger margin) for a more conservative gate.

    Returns (effective_min_snr, reason_str). Returns (None, reason) if
    loudest_neg_event is falsy (None or 0) -- disabling this path rather
    than guessing, the same convention resolve_merge_min_snr /
    resolve_group_min_snr use when their own calibration is unusable.
    """
    if not loudest_neg_event:
        return None, "no loudest_neg_event given, disabling the extreme-S/N path"
    effective = float(loudest_neg_event) * safety_margin
    return effective, (f"loudest sign-flipped event (full search)={loudest_neg_event:.2f}, "
                        f"safety_margin={safety_margin} -> effective threshold={effective:.2f}")


def merge_into_patch_table(patch_table: Table, seg3d_table: Table, min_snr: float = 10.0,
                            purity_scan=None, purity_target: float = 0.8, min_n_trust: int = 15,
                            dedup_radius_arcsec: float = 3.0, aperture_radius_arcsec: float = 2.0,
                            class_ntop: int = 3, group_min_lines: int | None = None,
                            group_min_snr: float | None = 10.0, group_purity_scan=None,
                            group_min_n_trust: int = 3, extreme_snr_loudest_event=None,
                            extreme_snr_safety_margin: float = 1.1):
    """Merge RELIABLE seg3d line-only candidates into an existing seg2d
    patch table as auxiliary type='T' targets, so they are processed by
    the same downstream classification/extraction as any 2D-detected
    source. seg2d remains the default, primary detection method -- this
    is only invoked when the caller explicitly opts in (aps_ifu_prepare's
    seg3d_merge=True), never automatically. Deliberately conservative:
    the goal is a short, trustworthy list of genuine line-only sources
    (weak/no continuum), not maximum recall.

    A candidate is merged if it has no_continuum_counterpart AND is not
    already within dedup_radius_arcsec of an existing non-mask patch_table
    row, AND EITHER of two independent paths qualifies it:

      (a) single-line path: snr >= the EFFECTIVE minimum SNR, which is
          min_snr UNLESS a purity_scan is supplied, in which case
          resolve_merge_min_snr derives a (possibly higher) data-driven
          value from the cube's own purity self-check and min_snr becomes
          a floor -- see that function's docstring. This path is what the
          field-level sign-flip purity self-check can actually support.

      (b) multi-line path (group_min_lines, default 2): the candidate
          belongs to a spatial group (see group_multiline_candidates /
          candidates_to_table's group_id -- exact same detection-grid
          pixel by default, group_radius_px=0) with >= group_min_lines
          no_continuum_counterpart candidates at that position, AND the
          group's COMBINED significance (combined_group_snr -- quadrature
          sum over EVERY member's SNR, not just the strongest one, so
          several weak-but-real lines can outrank one marginal line the
          way a single max()-based score cannot) clears the EFFECTIVE
          group threshold. That threshold is group_min_snr UNLESS
          group_purity_scan is supplied, in which case resolve_group_min_
          snr derives a (possibly higher) data-driven value from THIS
          field's own group-level purity self-check (post_veto_group_
          purity_scan, using group_min_n_trust -- default 3, deliberately
          much lower than min_n_trust's 15, because groups are rarer than
          individual candidates by construction) and group_min_snr becomes
          a floor -- exactly the same purity_scan/floor relationship path
          (a) has, just calibrated on groups instead of single candidates
          (see that function's docstring for why groups need their OWN
          calibration rather than reusing path (a)'s).
          This path is NOT a substitute for verifying the group's lines
          are physically consistent (same redshift) -- that is not checked
          here; every merged target, single- or multi-line, still goes
          through the same Redrock classification pass as any seg2d target
          immediately after this function runs, which is where that
          consistency actually gets tested.
          The radius=0 (exact-pixel) default was checked against the
          sign-flipped (noise) cube on the validated test field before
          being set, not chosen by inspection of the real candidates alone
          -- a 1-pixel tolerance let noise-group contamination rise
          sharply with no compensating gain in real groups recovered (see
          group_multiline_candidates's docstring). Re-run that check
          before loosening it on a new field/config. Set
          group_min_lines=None to disable this path entirely and fall back
          to single-line-only behaviour.

      (c) extreme-S/N path (extreme_snr_loudest_event): for a candidate
          that clears NEITHER path above -- typically a singleton (no
          corroborating line, so (b) cannot help) whose own S/N is so high
          that a ratio-based purity self-check simply never gets measured
          with enough samples (n_pos>=min_n_trust) that far out on the
          curve to certify it via path (a) either. Rather than leave such
          a candidate unmerged purely on a statistical-power technicality,
          this path compares its S/N directly against the single loudest
          fluctuation the FULL sign-flipped search ever produced
          (loudest_event_snr / run_seg3d's returned loudest_neg_event --
          every voxel of every template width, no veto or threshold
          applied first), the same "loudest event" fallback used to
          validate extreme-significance detections once a ratio-based
          background estimate runs out of trials (see resolve_extreme_
          snr_min). effective_extreme_snr = loudest_neg_event *
          extreme_snr_safety_margin (default 1.1, i.e. 10% above the
          loudest observed noise fluctuation) -- a pragmatic margin, not a
          calibrated false-alarm rate; tighten it for a stricter gate.
          extreme_snr_loudest_event=None (default) disables this path
          entirely -- it is opt-in, since loudest_event_snr must be
          computed from the full SNR cubes (run_seg3d does this
          automatically when it runs; a caller building seg3d_table by
          other means must supply it explicitly). Like path (b), this is
          independent of paths (a)/(b): failure of either does not disable
          this one, and vice versa.

    Preserves patch_table's exact column schema (the analysis-mode loader
    asserts on it exactly, so no new column can be added) -- provenance is
    carried instead via patch_table['flag'] = SEG3D_PROVENANCE_FLAG, so a
    merged target can always be identified/filtered later with
    `patch_table['flag'] == SEG3D_PROVENANCE_FLAG`. Merged targets get a
    circular aperture of aperture_radius_arcsec (default 2.0", matching
    this module's default spatial_fwhm_arcsec) centred on the group's
    highest-SNR member when merged via the multi-line path (recovers the
    other line(s) too -- they land inside the same aperture by
    construction, since that's exactly what made them a group); Z/ZERR/
    ZWARN/CLASS are initialized to the same not-yet-classified placeholder
    shape as any other row -- Redrock fills them in during the
    classification step that already follows seg2d.

    Returns (merged_table, n_merged, n_rejected_as_duplicate, merged_ids).
    merged_ids is the seg3d_table['id'] value of each candidate actually
    merged (empty array if none; for a multi-line group, only the
    representative highest-SNR member's id, not every line in the group)
    -- lets a caller (e.g. the comparison figure) mark exactly which of the
    full candidate list made it in, without re-deriving the same filtering
    itself. If purity_scan rules out every single-line threshold as
    untrustworthy AND no multi-line group qualifies either, returns
    (patch_table, 0, 0, empty) unchanged -- the field found no candidates
    reliable enough to add, not an error.
    """
    empty_ids = np.array([], dtype=int)
    if len(seg3d_table) == 0:
        return patch_table, 0, 0, empty_ids

    effective_min_snr = min_snr
    if purity_scan is not None:
        effective_min_snr, reason = resolve_merge_min_snr(
            purity_scan, min_snr_floor=min_snr, purity_target=purity_target, min_n_trust=min_n_trust)
        print(f"merge_into_patch_table: {reason}")
        if effective_min_snr is None:
            # The single-line path is unusable for this field, but the
            # multi-line path below is independent of this calibration and
            # may still qualify candidates -- do not bail out yet.
            effective_min_snr = np.inf

    line_only = seg3d_table[seg3d_table["no_continuum_counterpart"]]
    snr_mask = seg3d_table["snr"] >= effective_min_snr

    group_mask = np.zeros(len(seg3d_table), dtype=bool)
    if group_min_lines is not None and "group_id" in seg3d_table.colnames and len(line_only) > 0:
        counts = Counter(line_only["group_id"].tolist())

        effective_group_min_snr = group_min_snr
        if group_purity_scan is not None:
            effective_group_min_snr, group_reason = resolve_group_min_snr(
                group_purity_scan, min_snr_floor=group_min_snr,
                purity_target=purity_target, min_n_trust=group_min_n_trust)
            print(f"merge_into_patch_table: {group_reason}")
            if effective_group_min_snr is None:
                # The multi-line path is unusable for this field, but the
                # single-line path above is independent of this calibration
                # and may still have qualified candidates -- do not let a
                # failed group calibration undo that.
                effective_group_min_snr = np.inf

        # Combined-significance requirement: the group's COMBINED S/N
        # (combined_group_snr, quadrature sum over EVERY member, not just
        # the strongest one -- see that function's docstring) must clear
        # effective_group_min_snr. Empirically validated against the
        # sign-flipped cube (see aps_ifu_prepare.md): at this module's
        # default grouping (exact same detection-grid pixel, radius_px=0),
        # a fixed combined floor of 10.0 gives 0/8 false multi-line groups
        # in pure noise while retaining real ones on the validated test
        # field; group_purity_scan (when supplied) refines that per-field
        # via the same self-calibration machinery as the single-line path.
        snrs_per_group: dict[int, list[float]] = {}
        for gid, snr in zip(line_only["group_id"], line_only["snr"]):
            snrs_per_group.setdefault(int(gid), []).append(float(snr))
        combined_snr_per_group = {gid: combined_group_snr(snrs) for gid, snrs in snrs_per_group.items()}
        qualifying_groups = {
            gid for gid, n in counts.items()
            if n >= group_min_lines and (
                effective_group_min_snr is None
                or combined_snr_per_group.get(gid, -np.inf) >= effective_group_min_snr)
        }
        group_mask = (np.isin(seg3d_table["group_id"], list(qualifying_groups))
                      & seg3d_table["no_continuum_counterpart"])
        if qualifying_groups:
            print(f"merge_into_patch_table: {len(qualifying_groups)} multi-line group(s) "
                  f"(>={group_min_lines} lines, combined S/N>={effective_group_min_snr}) qualify "
                  f"independently of the single-line SNR/purity path")

    extreme_mask = np.zeros(len(seg3d_table), dtype=bool)
    if extreme_snr_loudest_event is not None:
        effective_extreme_snr, extreme_reason = resolve_extreme_snr_min(
            extreme_snr_loudest_event, safety_margin=extreme_snr_safety_margin)
        print(f"merge_into_patch_table: {extreme_reason}")
        if effective_extreme_snr is not None:
            extreme_mask = seg3d_table["snr"] >= effective_extreme_snr
            n_extreme = int(np.sum(extreme_mask & seg3d_table["no_continuum_counterpart"]))
            if n_extreme:
                print(f"merge_into_patch_table: {n_extreme} candidate(s) qualify via the "
                      f"extreme-S/N path (S/N>={effective_extreme_snr:.2f}) independently of "
                      f"the single-line and multi-line paths")

    promote_mask = (snr_mask | group_mask | extreme_mask) & seg3d_table["no_continuum_counterpart"]
    if "group_id" in seg3d_table.colnames and promote_mask.any():
        # One representative per group among the promoted rows (its
        # highest-SNR member, since seg3d_table is already SNR-sorted)
        # rather than one merged row per line -- a source is one target,
        # however many of its lines individually qualified it. Applies
        # regardless of which path (single-line or multi-line) promoted
        # each row, since group_id reflects physical position, not the
        # promotion reason.
        seen = set()
        drop_ids = []
        for gid, row in zip(seg3d_table["group_id"][promote_mask], seg3d_table[promote_mask]):
            if gid in seen:
                drop_ids.append(int(row["id"]))
            else:
                seen.add(gid)
        if drop_ids:
            promote_mask &= ~np.isin(seg3d_table["id"], drop_ids)

    candidates = seg3d_table[promote_mask]
    if len(candidates) == 0:
        return patch_table, 0, 0, empty_ids

    real_rows = patch_table[patch_table["type"] != "M"]
    n_rejected = 0
    if len(real_rows) > 0:
        existing = SkyCoord(ra=np.asarray(real_rows["RA_icrs"]) * u.degree,
                             dec=np.asarray(real_rows["DEC_icrs"]) * u.degree)
        cand_coord = SkyCoord(ra=np.asarray(candidates["RA_icrs"]) * u.degree,
                               dec=np.asarray(candidates["DEC_icrs"]) * u.degree)
        _, d2d, _ = cand_coord.match_to_catalog_sky(existing)
        keep_mask = d2d.arcsec > dedup_radius_arcsec
        n_rejected = int((~keep_mask).sum())
        candidates = candidates[keep_mask]

    if len(candidates) == 0:
        return patch_table, 0, n_rejected, empty_ids

    n_new = len(candidates)
    merged_ids = np.asarray(candidates["id"], dtype=int)
    new_id0 = int(np.max(patch_table["id"])) + 1
    aperture_deg = aperture_radius_arcsec / 3600.0

    fixed = Table()
    fixed["id"] = np.arange(new_id0, new_id0 + n_new, dtype=int)
    fixed["RA_icrs"] = np.asarray(candidates["RA_icrs"], dtype=float) * u.degree
    fixed["DEC_icrs"] = np.asarray(candidates["DEC_icrs"], dtype=float) * u.degree
    fixed["A_world"] = np.full(n_new, aperture_deg) * u.degree
    fixed["B_world"] = np.full(n_new, aperture_deg) * u.degree
    fixed["angle"] = np.zeros(n_new) * u.degree
    fixed["flag"] = np.full(n_new, SEG3D_PROVENANCE_FLAG)
    fixed["type"] = ["T"] * n_new

    # Same construction pattern as ifu_seg2d.create_aps_targets's own STEP5
    # (method 2): a separate Table with explicit dtype for the
    # not-yet-classified redshift/class array columns, then hstack -- this
    # is what makes the result vstack-compatible with the real patch_table.
    class_block = Table(
        [[[np.nan] * class_ntop for _ in range(n_new)],
         [[np.nan] * class_ntop for _ in range(n_new)],
         [[np.nan] * class_ntop for _ in range(n_new)],
         [[""] * class_ntop for _ in range(n_new)]],
        names=("Z", "ZERR", "ZWARN", "CLASS"),
        dtype=("float", "float", "float", "U18"),
    )
    new_rows = hstack([fixed, class_block])[list(patch_table.colnames)]
    merged = vstack([patch_table, new_rows], join_type="exact")
    return merged, n_new, n_rejected, merged_ids


# ----------------------------------------------------------------------
# Top-level orchestration
# ----------------------------------------------------------------------

def run_seg3d(
    infiles: list[str],
    headname: str,
    outpath: str,
    *,
    lmin: float = 3700.0,
    lmax: float = 9390.0,
    sens_corr: bool = True,
    mask_gaps: bool = True,
    safe_mask_gaps: bool = True,
    tellurics: bool = False,
    vacuum: bool = False,
    arms_ratio: list[float] | None = None,
    crr: bool = False,
    threshold: float = 6.0,
    spatial_bin: int = 4,
    sky_mask_min_A: float | None = None,
    continuum_method: str = "pca",
    pca_n_components: int = 15,
    pca_niter: int = 2,
    ivar_calibration: str = "per-wave",
    ivar_calib_smooth_A: float = 100.0,
    ivar_cap_factor: float = 3.0,
    spatial_fwhm_arcsec: float = 2.0,
    spectral_fwhm_pix: tuple[float, ...] = (3.0, 8.0),
    pixscale_arcsec: float = 0.5,
    resolvedness_min_frac: float = 0.3,
    max_npix: int = 5000,
    min_npix: int = 3,
    coincidence_min_count: int = 8,
    bright_mask_radius_factor: float = 3.0,
    bright_mask_min_radius: float = 5.0,
    no_bright_mask: bool = False,
    min_edge_dist_px: float = 3.0,
    purity_scan_thresholds: list[float] | None = None,
    group_radius_px: float = 0.0,
) -> dict:
    """Run the full 3D matched-filter detection pipeline over [lmin, lmax]
    Angstrom -- one arm, or a concatenation of however many arms overlap
    that range (see load_wavelength_range). Returns a dict (never raises
    for a "no candidates" result; does raise if no arm covers the
    requested range, or if multiple overlapping arms don't share a
    spatial grid):

      candidates   : list of (Candidate, is_new) pairs, post-veto
      table        : the same, as an Astropy Table (see candidates_to_table
                     -- includes group_id/n_group_lines from
                     group_multiline_candidates(radius_px=group_radius_px),
                     i.e. spatially coincident, wavelength-separated
                     detections such as a source's Hbeta+[OIII] pair are
                     already identified as one group here, not left for a
                     caller to work out)
      objects      : sep continuum catalog (this module's own binned pass)
      wcs          : celestial WCS of the (binned) detection grid
      collapse     : binned white-light image used for the sep pass
      bad_waves    : wavelengths flagged instrumental by coincidence_veto
      purity_scan  : list of (threshold, n_pos, n_neg, purity) tuples
      group_purity_scan : the same shape, but for multi-line GROUPS vs a
                     combined-S/N threshold (see post_veto_group_purity_scan
                     / combined_group_snr) -- what merge_into_patch_table's
                     group_purity_scan argument expects
      loudest_neg_event : single loudest raw S/N anywhere in the sign-
                     flipped cubes, no veto/threshold applied (see
                     loudest_event_snr) -- what merge_into_patch_table's
                     extreme_snr_loudest_event argument expects
      k_global     : measured global ivar noise-underestimation factor
      n_masked_wave, n_bright_masked, spatial_bin, pixscale_arcsec : bookkeeping

    Defaults (lmin=3700, lmax=9390) cover the LOWRES (LIFULR/MIFULR)
    blue+red combined native wavelength range, measured directly from real
    L1 file headers (blue arm 3600-5955.5A, red arm 5820-9490.5A, ~135A of
    overlap between them), trimmed by ~100A at each end -- the same
    per-edge clearance the old edge_trim_A parameter used to apply
    automatically, now the caller's responsibility via the choice of
    lmin/lmax itself, not a separate parameter. `load_wavelength_range`
    (via APSOB) draws on whichever file(s) among `infiles` actually cover
    the requested window, so a range spanning both arms is exactly as
    valid as one confined to a single arm -- this module was originally
    validated on a BLUE-only window (3680-5835A) and nothing about that
    validation depended on staying within one arm.
    HIGHRES (LIFUHR/MIFUHR) fields should NOT use this module default --
    their native coverage (blue 4695-5505A, red 5930-6840A) has a real gap
    between the two arms (unlike LOWRES's overlap) and does not overlap
    this LOWRES-derived window at all; pass lmin/lmax explicitly (or via
    IFU_params JSON, which every HIGHRES survey config in
    configs/ExGal_configs/ already does) instead. See doc/aps_ifu_prepare.md.
    """
    cube = load_wavelength_range(infiles, lmin, lmax, sens_corr=sens_corr, mask_gaps=mask_gaps,
                                  safe_mask_gaps=safe_mask_gaps, tellurics=tellurics,
                                  vacuum=vacuum, arms_ratio=arms_ratio, crr=crr)
    print(f"  shape={cube.data.shape}  wave={cube.wave[0]:.1f}-{cube.wave[-1]:.1f}A")

    wcs_native = WCS(cube.header).celestial

    good_wave = wavelength_mask(cube.wave, sky_dominated_min_A=sky_mask_min_A)
    n_masked_wave = int((~good_wave).sum())
    if n_masked_wave:
        print(f"  sky-dominated mask: excluding {n_masked_wave}/{len(cube.wave)} channels "
              f"(sky_mask_min={sky_mask_min_A})")
    cube.ivar = apply_wavelength_mask(cube.ivar, good_wave)

    effective_pixscale = pixscale_arcsec
    wcs_use = wcs_native
    if spatial_bin > 1:
        ny0, nx0 = cube.data.shape[1:]
        cube.data, cube.ivar = bin_cube_spatial(cube.data, cube.ivar, spatial_bin)
        cube.collapse = bin_image_spatial(cube.collapse, spatial_bin)
        effective_pixscale = pixscale_arcsec * spatial_bin
        wcs_use = rebin_wcs(wcs_native, spatial_bin)
        print(f"  spatial binning {spatial_bin}x{spatial_bin}: {ny0}x{nx0} -> "
              f"{cube.data.shape[1]}x{cube.data.shape[2]} spaxels ({effective_pixscale}\"/bin)")

    # Spatial coverage mask for edge_distance_veto -- computed from coverage
    # alone (any wavelength with ivar>0), BEFORE the bright-source mask is
    # applied below, so a bright star's exclusion zone is never mistaken
    # for the field-of-view edge.
    good_bins = np.any(cube.ivar > 0, axis=0)

    print(f"  continuum removal: {continuum_method}")
    if continuum_method == "pca":
        residual = continuum_subtract_pca(cube.data, cube.ivar, good_wave,
                                           n_components=pca_n_components, niter=pca_niter)
    elif continuum_method == "robust":
        residual = continuum_subtract_robust(cube.data)
    else:
        residual = continuum_subtract(cube.data)

    spatial_sigma_pix = (spatial_fwhm_arcsec / effective_pixscale) / FWHM_SIGMA
    spectral_sigmas_pix = [f / FWHM_SIGMA for f in spectral_fwhm_pix]
    fwhms = [s * FWHM_SIGMA for s in spectral_sigmas_pix]

    k_global = calibrate_ivar_scale(residual, cube.ivar)
    print(f"  ivar noise-underestimation factor: k={k_global:.3f} "
          f"(ivar too optimistic by ~{k_global**2:.1f}x in variance)")
    if ivar_calibration == "per-wave":
        smooth_ch = max(1, int(round(ivar_calib_smooth_A / abs(cube.wave[1] - cube.wave[0]))))
        k_wave = calibrate_ivar_scale_per_wave(residual, cube.ivar, smooth_window=smooth_ch)
        print(f"  per-wavelength k range: {k_wave.min():.2f}-{k_wave.max():.2f}, "
              f"median {np.median(k_wave):.2f}")
        ivar_calibrated = cube.ivar / k_wave[:, None, None] ** 2
    else:
        ivar_calibrated = cube.ivar / max(k_global, 1.0) ** 2

    ivar_use = normalize_ivar(ivar_calibrated, cap_factor=ivar_cap_factor)

    sep_bin = max(1, spatial_bin)
    objects = sep_catalog(cube.collapse, bw=max(4, 32 // sep_bin), bh=max(4, 32 // sep_bin),
                           minarea=max(3, 20 // sep_bin ** 2))
    print(f"  {len(objects)} continuum sources from sep (binned grid)")

    n_bright_masked = 0
    if not no_bright_mask and len(objects) > 0:
        bmask = bright_source_mask(cube.data.shape[1:], objects,
                                    radius_factor=bright_mask_radius_factor,
                                    min_radius=bright_mask_min_radius)
        n_bright_masked = int(bmask.sum())
        print(f"  bright-source mask: excluding {n_bright_masked}/{bmask.size} spaxels "
              f"({100 * n_bright_masked / bmask.size:.1f}%) around {len(objects)} continuum sources")
        ivar_use = ivar_use * (~bmask)[None, :, :]

    print("  running matched-filter width bank ...")
    snr_pos = [matched_filter_snr(residual, ivar_use, spatial_sigma_pix, s) for s in spectral_sigmas_pix]
    cands = bank_candidates_from_cubes(snr_pos, fwhms, cube.wave, threshold)
    cands = resolvedness_filter(cands, residual, min_frac=resolvedness_min_frac)
    n_pre_npix = len(cands)
    cands = spatial_extent_veto(cands, max_npix=max_npix, min_npix=min_npix)
    n_rej_npix = n_pre_npix - len(cands)
    cands, bad_waves = coincidence_veto(cands, cube.wave, min_count=coincidence_min_count)
    n_pre_edge = len(cands)
    cands = edge_distance_veto(cands, good_bins, min_dist_px=min_edge_dist_px)
    print(f"  {len(cands)} candidates survive the full veto chain "
          f"({len(bad_waves)} wavelength bins flagged instrumental, "
          f"{n_rej_npix} rejected by the npix floor/ceiling (min={min_npix}, max={max_npix}), "
          f"{n_pre_edge - len(cands)} rejected within {min_edge_dist_px}px of the field edge)")

    flagged = flag_line_only(cands, objects)
    n_new = sum(1 for _, is_new in flagged if is_new)
    print(f"  {n_new}/{len(cands)} have no continuum counterpart")

    scan_thresholds = sorted(set(purity_scan_thresholds or [threshold, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0]))
    snr_neg = [matched_filter_snr(-residual, ivar_use, spatial_sigma_pix, s) for s in spectral_sigmas_pix]
    purity_scan = post_veto_purity_scan(snr_pos, snr_neg, fwhms, residual, cube.wave, scan_thresholds,
                                         resolvedness_min_frac, max_npix, coincidence_min_count,
                                         good_bins=good_bins, min_edge_dist_px=min_edge_dist_px,
                                         min_npix=min_npix)
    print("  purity vs threshold: " +
          ", ".join(f"SNR>{t:.0f}:{p:.2f}" for t, _, _, p in purity_scan))

    # Sign-flipped candidate list AT THE PRIMARY THRESHOLD, full veto chain
    # applied -- the group-level analog of snr_neg's role above. Re-uses
    # catalog_at_threshold (the same re-thresholding path post_veto_purity_
    # scan already runs) rather than recomputing the veto chain by hand, so
    # this stays a genuine full-post-veto sign-flipped population, not an
    # approximation of one. post_veto_group_purity_scan then groups both
    # `cands` and `cands_neg` itself (see its docstring for why no further
    # per-threshold re-extraction is needed here, unlike snr_neg above).
    cands_neg = catalog_at_threshold(snr_neg, fwhms, -residual, cube.wave, threshold,
                                      resolvedness_min_frac, max_npix, coincidence_min_count,
                                      good_bins=good_bins, min_edge_dist_px=min_edge_dist_px,
                                      min_npix=min_npix)
    group_purity_scan = post_veto_group_purity_scan(cands, cands_neg, scan_thresholds,
                                                      radius_px=group_radius_px)
    print("  group purity vs combined-S/N threshold: " +
          ", ".join(f"S/N>{t:.0f}:{p:.2f}({n_pos}v{n_neg})"
                     for t, n_pos, n_neg, p in group_purity_scan))

    # Loudest event the FULL sign-flipped search ever produced (see
    # loudest_event_snr) -- no threshold/veto applied, every voxel of both
    # snr_neg cubes. The fallback merge_into_patch_table's extreme-S/N path
    # (resolve_extreme_snr_min) uses when the ratio-based purity_scan above
    # has no statistical power left to calibrate a genuinely extreme,
    # ungrouped candidate's own S/N.
    loudest_neg_event = loudest_event_snr(snr_neg)
    print(f"  loudest sign-flipped event (full search, no veto): S/N={loudest_neg_event:.2f}")

    table = candidates_to_table(flagged, wcs_use, group_radius_px=group_radius_px)
    if len(table) > 0:
        line_only = table[table["no_continuum_counterpart"]]
        multi = np.unique(line_only["group_id"][line_only["n_group_lines"] > 1])
        if len(multi):
            sizes = [int(line_only["n_group_lines"][line_only["group_id"] == g][0]) for g in multi]
            print(f"  {len(multi)} multi-line group(s) among the no-continuum-counterpart "
                  f"candidates (radius={group_radius_px}px), sizes={sizes} -- see group_id/"
                  f"n_group_lines in the candidate table")

    return dict(
        candidates=flagged, table=table, objects=objects, wcs=wcs_use, collapse=cube.collapse,
        bad_waves=bad_waves, purity_scan=purity_scan, group_purity_scan=group_purity_scan,
        loudest_neg_event=loudest_neg_event, k_global=k_global,
        n_masked_wave=n_masked_wave, n_bright_masked=n_bright_masked,
        spatial_bin=spatial_bin, pixscale_arcsec=effective_pixscale, group_radius_px=group_radius_px,
        lmin=lmin, lmax=lmax, headname=headname,
    )
