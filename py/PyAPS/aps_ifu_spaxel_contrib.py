"""
aps_ifu_spaxel_contrib.py
==========================

Shared, dependency-light helpers for the opt-in "spaxel-weighted LSF/
FWHM" feature (see `aps_utils.APSOB`'s `spaxel_weighted_lsf` parameter
and `aps_lsf.LSFInterpolator`/`aps_fwhm.FWHMInterpolator`'s own
`build_spaxel_weighted_entries` method): for a stacked/co-added IFU cube
(LIFU/MIFU/IFU mode, `is_fibre_level()`-false data), work out *which*
physical fibres from the underlying single-exposure files geometrically
contribute to each output spaxel, and by how much.

Two logically separate pieces, both pure numpy/astropy/scipy — no Dash,
no app-global state — deliberately usable from both the batch pipeline
(`aps_utils.py`) and the interactive explorer (`aps_l1_preview.py`,
which now imports `resolve_single_exposure_files`/
`read_fibtable_positions` from here instead of keeping its own copies —
see that module's own `contributing_fibre_positions()`):

1. **Provenance walk** (`resolve_single_exposure_files`,
   `read_fibtable_positions`, `resolve_single_exposure_fibres`): given a
   stacked L1 file, find its underlying single-exposure files via their
   PROV#### primary-header chain, and read each one's own FIBTABLE fibre
   sky positions (headers/one-small-table-only I/O — never touches image
   data).

2. **Geometry/weighting** (`contributing_fibre_weights`): given a set of
   spaxel sky positions and a set of (candidate) fibre sky positions,
   work out which fibres are close enough to matter to which spaxels and
   by how much, as one normalized sparse (n_spaxel x n_fibre) weight
   matrix — vectorized throughout (one batched `cKDTree` query, weight
   maths done once over every matched pair as plain numpy arrays), never
   a per-spaxel Python loop, matching the vectorization discipline
   `aps_lsf.py` established for its own per-fibre/global interpolators
   (see that module's v1.2 changelog).

3. **Combine engine** (`build_spaxel_weighted_entries`): given one of
   those weight matrices plus an *existing* `LSFInterpolator`/
   `FWHMInterpolator`'s own `interpolator_dict`/`wave_grid`, produces the
   per-spaxel weighted-combination interpolator entries. Implemented
   once here rather than twice (in `aps_lsf.py` and `aps_fwhm.py`
   separately) since the combine maths only touches each interpolator's
   already-generic `interpolator_dict`/`wave_grid` shape — the two
   modules' own `build_spaxel_weighted_entries` methods are thin
   wrappers around this function, keeping the per-fitting-method code
   (BSpline vs UnivariateSpline) the only thing that differs between
   them.

4. **Voronoi/PowerBin-bin-level LSF** (`aggregate_bin_lsf`,
   `bucket_lsf_curves`): the Gal/ExGal IFU pipeline (`aps_ifu_Gal.py`,
   `aps_ifu_ExGal.py`, `aps_ifu_prepare.py`'s classification stage) fits
   one spectrum per spatial *bin* (or, for classification, one collapsed
   spectrum for a whole patch), not one per spaxel — so per-spaxel LSF
   (piece 3 above) needs one more aggregation step down to bin level
   before it's usable there, plus a way to keep template-preparation cost
   from scaling with the (potentially large) number of bins. Both
   deliberately generic over *what* produced the per-spaxel curves being
   aggregated — global-only (feature off) or genuinely per-spaxel
   (`spaxel_weighted_lsf=True`) both work identically here, the same way
   `build_spaxel_weighted_entries`'s own per-spaxel entries always fall
   back to a real value (global) when there's nothing more specific.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

# Same caps as aps_l1_preview.py's own provenance walk -- a stacked file
# realistically has a handful to a couple dozen contributing exposures;
# these just stop a pathological/circular PROV chain from running away.
_MAX_PROV_FILES = 60
_MAX_PROV_DEPTH = 5


# --------------------------------------------------------------------------- #
# 1. Provenance walk
# --------------------------------------------------------------------------- #

def resolve_single_exposure_files(start_path):
    """Walk `start_path`'s own PROV#### provenance chain down to the
    genuine single-exposure files at the bottom of it — headers-only the
    whole way (`fits.open(..., memmap=True, lazy_load_hdus=True)`, never
    touching `.data`).

    A referenced file is treated as a genuine single-exposure contributor
    the moment it turns out to actually *have* a FIBTABLE extension (the
    real per-fibre table only single-exposure files carry — a stacked/
    cube L1 file's own extension list never includes one). If a
    referenced file doesn't have one, it must itself be another stack
    ("superstacked" data) — recurse into *that* file's own PROV list
    instead of trying to read a FIBTABLE that isn't there.

    Returns a deduplicated list of resolved single-exposure file paths
    (capped at `_MAX_PROV_FILES`) — never raises; a file that can't be
    opened, or has no provenance cards at all, is just absent from the
    result rather than a hard failure for the whole dataset."""
    resolved = []
    seen = set()

    def _walk(path, depth):
        if len(resolved) >= _MAX_PROV_FILES or depth > _MAX_PROV_DEPTH:
            return
        try:
            with fits.open(path, memmap=True, lazy_load_hdus=True) as hdul:
                extnames = [h.name for h in hdul]
                if "FIBTABLE" in extnames:
                    if path not in seen:
                        seen.add(path)
                        resolved.append(path)
                    return
                h0 = hdul[0].header
                # Zero-padded to a fixed width in every real card seen
                # (PROV0000..PROV0016, etc.), so a plain string sort
                # already puts them in the right order — PROV0000 (the
                # file's own name) is excluded, not a real contributor.
                prov_keys = sorted(
                    k for k in h0.keys()
                    if k.upper().startswith("PROV") and k.upper() != "PROV0000"
                )
                parent_dir = Path(path).parent
                for k in prov_keys:
                    if len(resolved) >= _MAX_PROV_FILES:
                        return
                    fname = str(h0[k]).strip()
                    if not fname:
                        continue
                    child = parent_dir / fname
                    child_str = str(child)
                    if child_str in seen or not child.exists():
                        continue
                    _walk(child_str, depth + 1)
        except Exception:
            return

    _walk(start_path, 0)
    return resolved


def read_fibtable_positions(path):
    """NSPEC/RA/Dec/status/targuse straight from one file's own FIBTABLE
    extension — the same columns `aps_utils.gen_targlist`'s MOS/MOSLIFU/
    MOSMIFU branch reads (`Nspec`/`TARGRA`/`TARGDEC`/`STATUS`/`TARGUSE`),
    read directly here via a plain headers-plus-one-small-table open
    rather than routed through APSOB's own, far more expensive, full
    per-target processing pipeline."""
    with fits.open(path, memmap=True, lazy_load_hdus=True) as hdul:
        fib = hdul["FIBTABLE"].data
        try:
            nspec = np.asarray(fib["NSPEC"], dtype=np.int64)
        except KeyError:
            nspec = np.asarray(fib["Nspec"], dtype=np.int64)
        ra = np.asarray(fib["TARGRA"], dtype=np.float64)
        dec = np.asarray(fib["TARGDEC"], dtype=np.float64)
        status = np.char.strip(np.char.upper(np.asarray(fib["STATUS"]).astype(str)))
        targuse = np.char.strip(np.char.upper(np.asarray(fib["TARGUSE"]).astype(str)))
    return nspec, ra, dec, status, targuse


def resolve_single_exposure_fibres(infname, active_status=("A",)):
    """Resolve `infname`'s (a stacked IFU cube's) underlying single-
    exposure files and concatenate every one of their own FIBTABLE fibre
    positions into one flat set of "fibre instances" — one row per
    (physical fibre, single exposure it was read in), not one row per
    unique physical fibre. That's deliberate: the same physical NSPEC
    sits at a different sky position in each dithered exposure, so each
    occurrence is a distinct geometric contributor and must be weighted
    separately by `contributing_fibre_weights` — deduplicating by NSPEC
    here would silently drop real dither coverage.

    Rows are filtered to `status in active_status` (default: just 'A',
    i.e. an active fibre actually placed on a target/sky position) —
    parked/dead/retracted fibres carry no meaningful sky position and
    must not be allowed to contribute.

    Returns a dict: `{"nspec": int array, "ra": float array (deg),
    "dec": float array (deg), "source_files": [str, ...]}`, all arrays
    the same length (n_fibre_instances). Never raises — a file that
    fails to resolve/open is just skipped; if nothing resolves at all,
    every array comes back empty (length 0), which
    `contributing_fibre_weights` handles as "no candidate fibres" (every
    spaxel falls back to global), not an error.
    """
    single_paths = resolve_single_exposure_files(infname)

    nspec_parts, ra_parts, dec_parts = [], [], []
    for path in single_paths:
        try:
            nspec, ra, dec, status, _targuse = read_fibtable_positions(path)
        except Exception:
            continue
        good = np.isin(status, list(active_status)) & np.isfinite(ra) & np.isfinite(dec)
        if not good.any():
            continue
        nspec_parts.append(nspec[good])
        ra_parts.append(ra[good])
        dec_parts.append(dec[good])

    if not nspec_parts:
        return {
            "nspec": np.empty(0, dtype=np.int64),
            "ra": np.empty(0, dtype=np.float64),
            "dec": np.empty(0, dtype=np.float64),
            "source_files": single_paths,
        }

    return {
        "nspec": np.concatenate(nspec_parts),
        "ra": np.concatenate(ra_parts),
        "dec": np.concatenate(dec_parts),
        "source_files": single_paths,
    }


# --------------------------------------------------------------------------- #
# 2. Geometry / weighting
# --------------------------------------------------------------------------- #

def _circle_overlap_area(d, r1, r2):
    """Closed-form intersection area of two circles (radius r1, r2,
    centre-to-centre distance d), all same units, all arrays (broadcast
    together). Standard lens-area formula; branch selection is
    vectorized via `np.where`, not a per-pair Python loop, matching
    `aps_lsf.py`'s own established branch-logic-via-np.where convention.
    """
    d = np.asarray(d, dtype=np.float64)
    r1 = np.broadcast_to(np.asarray(r1, dtype=np.float64), d.shape)
    r2 = np.broadcast_to(np.asarray(r2, dtype=np.float64), d.shape)

    no_overlap = d >= (r1 + r2)
    r_min = np.minimum(r1, r2)
    r_max = np.maximum(r1, r2)
    fully_inside = d <= (r_max - r_min)

    # General case — clip the acos arguments to [-1, 1]: floating-point
    # rounding can push a value fractionally outside that range right at
    # the tangency boundary, which would otherwise turn a legitimate
    # near-zero overlap into a NaN.
    with np.errstate(invalid="ignore", divide="ignore"):
        a1 = np.clip((d**2 + r1**2 - r2**2) / (2 * d * r1), -1.0, 1.0)
        a2 = np.clip((d**2 + r2**2 - r1**2) / (2 * d * r2), -1.0, 1.0)
        term = (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)
        general = (
            r1**2 * np.arccos(a1)
            + r2**2 * np.arccos(a2)
            - 0.5 * np.sqrt(np.clip(term, 0.0, None))
        )

    area = np.where(no_overlap, 0.0,
                     np.where(fully_inside, np.pi * r_min**2, general))
    return area


def contributing_fibre_weights(spaxel_ra, spaxel_dec, fibre_ra, fibre_dec,
                                fibre_radius_deg, spaxel_radius_deg,
                                weighting="overlap"):
    """Normalized (n_spaxel x n_fibre) sparse weight matrix of which
    fibre-instances (see `resolve_single_exposure_fibres`) contribute to
    which spaxels, and by how much.

    Vectorized: one batched `cKDTree.query_ball_point` call (over every
    spaxel at once, against a single tree built over every fibre) finds
    the candidate (spaxel, fibre) pairs — this is what keeps the cost at
    O(n_spaxel * avg_fibres_per_spaxel) rather than the O(n_spaxel *
    n_fibre) a naive full pairwise-distance matrix would cost on a real
    ~30k-spaxel cube. Weight maths (overlap area or inverse-distance) is
    then done once over the full flat array of matched pairs, plain
    numpy, no per-pair Python-level work.

    RA is pre-multiplied by cos(dec) (evaluated at the mean spaxel Dec)
    so the tree/distance maths operates in a locally flat, isotropic
    degree frame — same convention `apsPlot/flux_cube.py`'s own RA/Dec
    offset math and `aps_l1_preview.contributing_fibre_positions` already
    use.

    Parameters
    ----------
    spaxel_ra, spaxel_dec : array, shape (n_spaxel,)
        Spaxel sky positions, degrees.
    fibre_ra, fibre_dec : array, shape (n_fibre,)
        Candidate fibre-instance sky positions, degrees (see
        `resolve_single_exposure_fibres`).
    fibre_radius_deg : float
        Real WEAVE fibre core radius, degrees (half of
        `aps_constants.WEAVE_FIBRE_DIAMETER_ARCSEC[...] / 3600`).
    spaxel_radius_deg : float
        Equivalent-area circle radius for one output spaxel, degrees —
        derived from the cube's own real WCS pixel scale
        (`sqrt(pixel_area / pi)`), not a guessed/fixed value.
    weighting : {"overlap", "distance"}
        "overlap" (default): weight = circle-circle overlap area between
        the fibre's footprint and the spaxel's own equivalent-area
        circle. "distance": weight = 1/d^2 from spaxel centre to fibre
        centre (a small epsilon floor on d avoids a division blow-up for
        the on-centre case).

    Returns
    -------
    scipy.sparse.csr_matrix, shape (n_spaxel, n_fibre)
        Row-normalized (each non-empty row sums to 1). A spaxel with no
        fibre within `fibre_radius_deg + spaxel_radius_deg` gets an
        all-zero row — callers treat that as "no entry" (fall back to
        the global FWHM/LSF), not as an error.
    """
    n_spaxel = len(spaxel_ra)
    n_fibre = len(fibre_ra)
    if n_spaxel == 0 or n_fibre == 0:
        return csr_matrix((n_spaxel, n_fibre))

    dec0 = float(np.nanmean(spaxel_dec))
    cos_dec = np.cos(np.radians(dec0)) or 1.0

    fibre_xy = np.column_stack([fibre_ra * cos_dec, fibre_dec])
    spaxel_xy = np.column_stack([spaxel_ra * cos_dec, spaxel_dec])

    tree = cKDTree(fibre_xy)
    search_radius = fibre_radius_deg + spaxel_radius_deg
    # One batched call across every spaxel — scipy loops internally in
    # compiled code, not a Python-level per-spaxel query.
    neighbour_lists = tree.query_ball_point(spaxel_xy, r=search_radius)

    row_idx = np.concatenate([np.full(len(nbrs), i, dtype=np.int64)
                               for i, nbrs in enumerate(neighbour_lists)]) \
        if any(neighbour_lists) else np.empty(0, dtype=np.int64)
    col_idx = np.concatenate([np.asarray(nbrs, dtype=np.int64)
                               for nbrs in neighbour_lists if nbrs]) \
        if any(neighbour_lists) else np.empty(0, dtype=np.int64)

    if row_idx.size == 0:
        return csr_matrix((n_spaxel, n_fibre))

    dx = (spaxel_xy[row_idx, 0] - fibre_xy[col_idx, 0])
    dy = (spaxel_xy[row_idx, 1] - fibre_xy[col_idx, 1])
    dist = np.sqrt(dx**2 + dy**2)

    if weighting == "overlap":
        weights = _circle_overlap_area(dist, fibre_radius_deg, spaxel_radius_deg)
    elif weighting == "distance":
        eps = max(fibre_radius_deg, spaxel_radius_deg) * 1e-3
        weights = 1.0 / np.maximum(dist, eps) ** 2
    else:
        raise ValueError(f"Unknown weighting={weighting!r}, expected 'overlap' or 'distance'")

    # Drop true-zero-weight pairs (e.g. exact tangency under "overlap")
    # before building the sparse matrix, rather than storing explicit
    # zeros.
    nonzero = weights > 0
    mat = csr_matrix((weights[nonzero], (row_idx[nonzero], col_idx[nonzero])),
                      shape=(n_spaxel, n_fibre))

    row_sums = np.asarray(mat.sum(axis=1)).ravel()
    nonzero_rows = row_sums > 0
    if nonzero_rows.any():
        inv = np.zeros_like(row_sums)
        inv[nonzero_rows] = 1.0 / row_sums[nonzero_rows]
        mat = mat.multiply(inv[:, None]).tocsr()

    return mat


# --------------------------------------------------------------------------- #
# 3. Combine engine
# --------------------------------------------------------------------------- #

def build_spaxel_weighted_entries(interpolator_dict, wave_grid, weight_matrix,
                                   fibre_nspecs, aps_ids, entry_type):
    """Combine existing per-fibre `interpolate_function`s into one
    per-spaxel weighted-average `interpolate_function` each, using an
    (n_spaxel x n_fibre_instance) weight matrix from
    `contributing_fibre_weights`.

    Deliberately does **not** touch `interpolator_dict` itself — every
    per-spaxel entry this returns is a fresh, separate dict; callers own
    merging it into their own copy under a namespaced key (e.g.
    `fwhm_interp_dict['spaxel_weighted']`, never a bare NSPEC/APS_ID key
    — see `aps_utils._assign_arm_results_to_targets`'s own comment on
    why a cube's fake per-spaxel NSPEC can collide with a real
    calibration-fibre NSPEC). Mutating `interpolator_dict` in place would
    also corrupt that same `LSFInterpolator`/`FWHMInterpolator` object's
    *own* `get_available_fibers()` (and everything downstream of it —
    smoothing, missing-fibre fill, diagnostic plots all iterate that
    list expecting every entry to be a real fibre).

    No new spline fit per spaxel: each *unique* contributing NSPEC's
    existing `interpolate_function` is evaluated exactly once on the
    shared `wave_grid` (fibre instances that are the same physical NSPEC
    seen in different dithers reuse that one evaluation — the geometry
    weight, not the resolution curve, differs between them), giving an
    (n_unique_nspec x n_wave) dense matrix; the per-spaxel result is then
    one sparse-weight-matrix x dense-fibre-grid multiply, i.e. every
    spaxel's combination happens in a single vectorized matrix product,
    not a Python loop over spaxels doing per-wavelength work. Building
    the actual lightweight `np.interp`-based closures below *is* a
    Python loop over spaxels, but it's cheap object/closure construction
    (no numeric work happens until a closure is actually called) — the
    same closure-per-item pattern `aps_lsf.py`'s own per-fibre
    `make_fiber_func` already uses, just one call per spaxel instead of
    per physical fibre.

    Parameters
    ----------
    interpolator_dict : dict
        An `LSFInterpolator`/`FWHMInterpolator`'s own
        `get_interpolator_dict()` result — read-only here.
    wave_grid : array
        The same shared wavelength grid `interpolator_dict['global']`
        and every per-fibre entry are defined over.
    weight_matrix : scipy.sparse matrix, shape (n_spaxel, n_fibre_instance)
        From `contributing_fibre_weights` — row-normalized, all-zero
        rows mean "no contributing fibre".
    fibre_nspecs : array, shape (n_fibre_instance,)
        Physical NSPEC per fibre-instance column of `weight_matrix` (see
        `resolve_single_exposure_fibres` — duplicates across dithers are
        expected and handled).
    aps_ids : array, shape (n_spaxel,)
        Output spaxel APS_ID per row of `weight_matrix` — becomes the
        key of the returned dict.
    entry_type : str
        Value stored in each returned entry's `'type'` field (e.g.
        `'spaxel_lsf_weighted'` / `'spaxel_fwhm_weighted'`) — cosmetic,
        mirrors the existing `'fiber_lsf'`/`'fiber_fwhm'` convention.

    Returns
    -------
    dict
        `{aps_id: {'interpolate_function': callable, 'type': entry_type,
        'n_contrib_fibres': int}, ...}` — one entry per spaxel with at
        least one contributing fibre; spaxels with none are simply
        absent (caller falls back to `interpolator_dict['global']`).
    """
    n_spaxel, n_fibre = weight_matrix.shape
    if n_spaxel == 0 or n_fibre == 0 or wave_grid is None:
        return {}

    wave_grid = np.asarray(wave_grid, dtype=np.float64)

    unique_nspecs, inverse = np.unique(np.asarray(fibre_nspecs, dtype=np.int64),
                                        return_inverse=True)
    fibre_grid = np.empty((len(unique_nspecs), len(wave_grid)), dtype=np.float64)
    for i, nspec in enumerate(unique_nspecs):
        entry = interpolator_dict.get(int(nspec))
        func = entry['interpolate_function'] if entry is not None \
            else interpolator_dict['global']['interpolate_function']
        fibre_grid[i] = np.asarray(func(wave_grid), dtype=np.float64)

    # Collapse fibre-instance columns that share the same physical NSPEC
    # (several dithers of the same fibre) into one column per unique
    # NSPEC — a second sparse matrix product, still fully vectorized.
    collapse = csr_matrix(
        (np.ones(n_fibre), (np.arange(n_fibre), inverse)),
        shape=(n_fibre, len(unique_nspecs)),
    )
    collapsed_weights = weight_matrix.tocsr() @ collapse  # (n_spaxel, n_unique_nspec)

    combined = np.asarray(collapsed_weights @ fibre_grid)  # dense (n_spaxel, n_wave)
    row_weight_sum = np.asarray(collapsed_weights.sum(axis=1)).ravel()
    n_contrib = np.asarray((collapsed_weights > 0).sum(axis=1)).ravel()

    entries = {}
    aps_ids = np.asarray(aps_ids)
    for i in np.flatnonzero(row_weight_sum > 0):
        fwhm_row = combined[i]

        def interp_func(wavelengths, _wg=wave_grid, _fw=fwhm_row):
            # Same calling convention as every other interpolate_function
            # in this module family (aps_lsf.py/aps_fwhm.py): accepts
            # scalar or array, returns matching shape. np.interp's
            # default behaviour (clip to the endpoint value outside
            # [_wg[0], _wg[-1]]) is exactly this family's own "flat
            # extrapolation" convention, so no extra branching is needed
            # here.
            wavelengths = np.atleast_1d(wavelengths)
            fwhm = np.interp(wavelengths, _wg, _fw)
            return fwhm if len(wavelengths) > 1 else float(fwhm[0])

        entries[int(aps_ids[i])] = {
            'interpolate_function': interp_func,
            'type': entry_type,
            'n_contrib_fibres': int(n_contrib[i]),
        }

    return entries


# --------------------------------------------------------------------------- #
# 4. Voronoi/PowerBin-bin-level LSF
# --------------------------------------------------------------------------- #

def aggregate_bin_lsf(spaxel_fwhm_funcs, bin_num, flux_weight, wave_grid):
    """Combine per-spaxel LSF/FWHM curves into one curve per spatial bin
    (a Voronoi/PowerBin bin, or — for the classification/collapse-for-
    redshift stage — a single bin covering an entire patch), flux-weighted
    to mirror how each bin's own *spectrum* is combined
    (`ExGalPrepare.voronoi_binning`: member spaxel spectra are summed, so a
    brighter spaxel already dominates the bin's spectral shape more — its
    resolution should dominate the assumed bin resolution the same way).

    Fully generic over what produced `spaxel_fwhm_funcs` — works
    identically whether those are real per-spaxel curves
    (`spaxel_weighted_lsf=True`) or every spaxel sharing the same global
    curve (feature off): in the latter case every bin's own aggregate is
    just that same global curve back out, so calling this unconditionally
    is safe and never changes behaviour when the underlying feature is off.

    Parameters
    ----------
    spaxel_fwhm_funcs : sequence, length n_spaxel
        Each element is a per-spaxel `interpolate_function` (e.g.
        `target.meta[i]['fwhm']['interpolate_function']`) or `None` (that
        spaxel contributes no weight at all — e.g. no FWHM info available).
    bin_num : array, length n_spaxel
        Each spaxel's bin id (any hashable/orderable value — typically the
        integer `binNum`/`BIN_ID` `ExGalPrepare.define_voronoi_bins`
        already produces; negative "outside the fitted region" ids are
        treated as their own ordinary groups here, same as every other
        value — this function has no opinion on which bins are meaningful,
        that's entirely the caller's call).
    flux_weight : array, length n_spaxel
        Per-spaxel weight (typically `signal`/broadband flux, already
        computed for the bin definition itself — no new weight concept
        introduced). Spaxels with non-finite or non-positive weight
        contribute zero.
    wave_grid : array
        Shared wavelength grid to evaluate every curve on before combining.

    Returns
    -------
    dict
        `{bin_id: {'interpolate_function': callable, 'n_contrib_spaxels':
        int}}` — same entry shape as `build_spaxel_weighted_entries`. A bin
        with zero total weight (every member spaxel had `None`/non-finite/
        non-positive weight) is simply absent — caller's job to fall back
        to whatever "no bin LSF" means in its own context (e.g. the global
        curve).
    """
    n_spaxel = len(bin_num)
    wave_grid = np.asarray(wave_grid, dtype=np.float64)
    flux_weight = np.asarray(flux_weight, dtype=np.float64)

    if n_spaxel == 0:
        return {}

    unique_bins, bin_inverse = np.unique(np.asarray(bin_num), return_inverse=True)
    n_bins = len(unique_bins)

    curve_grid = np.zeros((n_spaxel, len(wave_grid)), dtype=np.float64)
    valid = np.zeros(n_spaxel, dtype=bool)
    for i, func in enumerate(spaxel_fwhm_funcs):
        if func is None:
            continue
        curve_grid[i] = np.asarray(func(wave_grid), dtype=np.float64)
        valid[i] = True

    weight = np.where(valid & np.isfinite(flux_weight) & (flux_weight > 0), flux_weight, 0.0)

    W = csr_matrix((weight, (np.arange(n_spaxel), bin_inverse)), shape=(n_spaxel, n_bins))
    bin_weight_sum = np.asarray(W.sum(axis=0)).ravel()
    n_contrib = np.asarray((W > 0).sum(axis=0)).ravel()

    entries = {}
    for b in np.flatnonzero(bin_weight_sum > 0):
        member = (bin_inverse == b) & (weight > 0)
        w = weight[member]
        fwhm_row = (w[:, None] * curve_grid[member]).sum(axis=0) / w.sum()

        def interp_func(wavelengths, _wg=wave_grid, _fw=fwhm_row):
            # Same calling convention as every other interpolate_function
            # in this module family — see build_spaxel_weighted_entries.
            wavelengths = np.atleast_1d(wavelengths)
            fwhm = np.interp(wavelengths, _wg, _fw)
            return fwhm if len(wavelengths) > 1 else float(fwhm[0])

        entries[unique_bins[b]] = {
            'interpolate_function': interp_func,
            'n_contrib_spaxels': int(n_contrib[b]),
        }

    return entries


def bucket_lsf_curves(bin_curves, wave_grid, max_buckets=6, n_features=5, seed=0):
    """Quantize a (potentially large) set of per-bin LSF/FWHM curves into a
    small number of representative "buckets", so template preparation
    (the expensive step — see `IFUExGalPrepare.prepareSpectralTemplate
    Library`) can be done once per bucket instead of once per bin.

    Motivation: bin-to-bin LSF variation within one IFU patch mostly comes
    from *which* dithered fibres happen to overlap which part of the
    field — a smoothly, mildly varying effect across a single patch's own
    small footprint (the dominant LSF variation is *across the full ~960-
    fibre focal plane*, not within one patch's handful of dithers) — so a
    handful of buckets should already capture nearly all of the real
    signal a full per-bin convolution would, at a fraction of the cost.

    Parameters
    ----------
    bin_curves : dict
        `{bin_id: interpolate_function}` — typically
        `aggregate_bin_lsf(...)`'s own entries with just the callable
        pulled out, or any other per-bin curve source with the same shape.
    wave_grid : array
        Shared wavelength grid — both for evaluating input curves and for
        the returned representative bucket curves.
    max_buckets : int, optional
        Upper bound on the number of buckets (default 6). If there are
        fewer bins than this, every bin gets its own bucket (no
        approximation at all — the whole point of bucketing is amortizing
        cost over *many* bins, so it never kicks in for a small patch).
    n_features : int, optional
        Number of representative wavelengths (evenly spaced across
        `wave_grid`'s own range) used as the clustering feature vector —
        enough to capture a curve's overall shape (e.g. blue-vs-red slope)
        without over-fitting noise, not the full high-resolution curve.
    seed : int, optional
        RNG seed for the k-means initialization, for reproducibility.

    Returns
    -------
    bin_to_bucket : dict
        `{bin_id: bucket_id}` (`bucket_id` is `0..n_buckets-1`).
    bucket_curves : dict
        `{bucket_id: array}` — each bucket's own representative curve,
        evaluated on `wave_grid`: the *mean* curve of every bin assigned
        to that bucket (not an arbitrary member), so the approximation is
        centred rather than biased toward whichever bin happened to seed
        the cluster.
    """
    wave_grid = np.asarray(wave_grid, dtype=np.float64)
    bin_ids = list(bin_curves.keys())
    n_bins = len(bin_ids)

    if n_bins == 0:
        return {}, {}

    curves = np.array([np.asarray(bin_curves[b](wave_grid), dtype=np.float64) for b in bin_ids])

    if n_bins <= max_buckets:
        # No approximation needed -- every bin is its own bucket.
        bin_to_bucket = {b: i for i, b in enumerate(bin_ids)}
        bucket_curves = {i: curves[i] for i in range(n_bins)}
        return bin_to_bucket, bucket_curves

    feature_waves = np.linspace(wave_grid[0], wave_grid[-1], n_features)
    features = np.stack([np.interp(feature_waves, wave_grid, c) for c in curves])

    try:
        from scipy.cluster.vq import kmeans2
        # whiten manually (divide by per-feature std) rather than
        # scipy.cluster.vq.whiten, which would also need guarding against
        # a zero-std feature (e.g. every curve identical at one
        # wavelength) -- np.errstate/np.where here is the same "safe
        # because discarded" guard used elsewhere in this module.
        std = features.std(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            features_w = np.where(std > 0, features / std, features)
        centroids, labels = kmeans2(features_w, max_buckets, seed=seed, minit='++')
        n_actual_buckets = int(labels.max()) + 1
    except Exception:
        # Defensive fallback: simple 1D quantile bucketing on mean FWHM
        # across the fit range -- never let a clustering-library failure
        # take down the whole patch (same degrade-gracefully philosophy
        # as the rest of this module).
        mean_fwhm = features.mean(axis=1)
        edges = np.quantile(mean_fwhm, np.linspace(0, 1, max_buckets + 1)[1:-1])
        labels = np.searchsorted(edges, mean_fwhm)
        n_actual_buckets = int(labels.max()) + 1

    bin_to_bucket = {b: int(labels[i]) for i, b in enumerate(bin_ids)}
    bucket_curves = {
        bucket: curves[labels == bucket].mean(axis=0)
        for bucket in range(n_actual_buckets)
    }
    return bin_to_bucket, bucket_curves


def bucket_lsf_for_voronoi_bins(aps_id_to_fwhm_func, aps_id_to_flux, aps_id_to_bin,
                                 n_bins, wave_grid, max_buckets=6):
    """High-level convenience wrapping `aggregate_bin_lsf` + `bucket_lsf_
    curves` for the common IFU Gal/ExGal calling pattern: given every
    original spaxel's own per-spaxel LSF/FWHM `interpolate_function` and
    flux, and a mapping from that spaxel's own APS_ID straight through to
    its *final* fitted bin id, build the `(bin_to_bucket, LSF_Data_by_
    bucket)` pair `IFUExGalPPXF.runModule_PPXF`/`IFUExGalEMIPPXF.
    runModule_EMIPPXF`/`IFUExGalLS.runModule_LINESTRENGTH`'s own same-named
    parameters expect directly.

    Deliberately keyed by APS_ID throughout (a spaxel's real, stable
    identity), not by position in any particular array -- this is what
    lets a caller correctly aggregate through an intermediate stage (e.g.
    `ExGalPrepare.spatial_bin_with_provenance`'s spatial pre-binning,
    itself Voronoi/PowerBin-binned again afterward) just by composing
    `aps_id_to_bin` end to end (raw spaxel -> final bin), without this
    function needing to know anything about the intermediate stage at all
    -- the caller does that composition, this function only ever sees the
    final result.

    Parameters
    ----------
    aps_id_to_fwhm_func : dict
        `{aps_id: interpolate_function or None}`.
    aps_id_to_flux : dict
        `{aps_id: float}` -- flux weight (e.g. broadband signal).
    aps_id_to_bin : dict
        `{aps_id: int or None}` -- final positional bin id (`0..n_bins-1`,
        the same "position in `ExGalPrepare.define_voronoi_bins`'s own
        `binNum`" convention `runModule_PPXF`'s own `ubins = np.arange(0,
        nbins)` already uses). A spaxel absent from this dict, or mapped
        to `None`/a negative value, contributes nothing -- same
        excluded/outside convention as everywhere else in this file.
    n_bins : int
        Total number of final bins -- needed so the returned
        `bin_to_bucket` array has one entry per bin even if some bin
        happens to end up with zero valid-LSF spaxels (falls back to
        whichever bucket id is first in `LSF_Data_by_bucket`, a real
        curve either way, not a crash).
    wave_grid : array
        Shared wavelength grid.
    max_buckets : int, optional
        Passed straight through to `bucket_lsf_curves`.

    Returns
    -------
    bin_to_bucket : array, length n_bins
        Positional -- ready to pass directly as `runModule_PPXF`/etc.'s
        own `bin_to_bucket`.
    LSF_Data_by_bucket : dict
        `{bucket_id: interpolate_function}` -- ready to pass directly as
        `LSF_Data_by_bucket`. Empty dict if there was nothing to
        aggregate at all (caller's own `LSF_Data_by_bucket:` truthiness
        check, same as every runModule_* function already does, then
        naturally falls back to its pre-existing single-`LSF_Data` path).
    """
    common_ids = [aid for aid, b in aps_id_to_bin.items() if b is not None and b >= 0]
    funcs = [aps_id_to_fwhm_func.get(aid) for aid in common_ids]
    flux = np.array([aps_id_to_flux.get(aid, 0.0) for aid in common_ids])
    bin_ids = np.array([aps_id_to_bin[aid] for aid in common_ids])

    bin_entries = aggregate_bin_lsf(funcs, bin_ids, flux, wave_grid)
    bin_curves = {b: e['interpolate_function'] for b, e in bin_entries.items()}

    bin_to_bucket_map, bucket_curves = bucket_lsf_curves(bin_curves, wave_grid, max_buckets=max_buckets)

    def _make_func(curve, wg):
        def f(w):
            w = np.atleast_1d(w)
            out = np.interp(w, wg, curve)
            return out if len(w) > 1 else float(out[0])
        return f

    LSF_Data_by_bucket = {b: _make_func(c, wave_grid) for b, c in bucket_curves.items()}

    if not LSF_Data_by_bucket:
        return np.zeros(n_bins, dtype=int), {}

    fallback_bucket = next(iter(LSF_Data_by_bucket))
    bin_to_bucket = np.full(n_bins, fallback_bucket, dtype=int)
    for b, bucket in bin_to_bucket_map.items():
        if 0 <= b < n_bins:
            bin_to_bucket[b] = bucket

    return bin_to_bucket, LSF_Data_by_bucket
