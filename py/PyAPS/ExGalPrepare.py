import functools
import logging
import os
import pickle
import sys
import time

import matplotlib
import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance as dist
from astropy.coordinates import FK5, ICRS, SkyCoord
from astropy.io import fits
from astropy.table import Table, hstack, vstack
from astropy.utils.data import get_pkg_data_filename
from astropy.visualization import simple_norm
from astropy.wcs import WCS, utils
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import Voronoi, voronoi_plot_2d

from PyAPS.apsPlot.exgal_prepare import (
    plot_all,
    plot_preparation_summary,
    plot_snr_stages,
    plot_spatial,
    voronoi_visualization,
)

mpl_version = matplotlib.__version__
if tuple(map(int, mpl_version.split(".")[:2])) >= (3, 6):
    # For matplotlib 3.6+
    def get_cmap(name):
        return plt.colormaps[name]

else:
    # For older matplotlib versions
    def get_cmap(name):
        return cm.get_cmap(name)


import logging

import PyAPS
from PyAPS import aps_constants
from PyAPS import ExGalutil

APSVERS = PyAPS.__version__


# Binning engine: PowerBin (Cappellari 2025, MNRAS 544, 1432) replaces the
# earlier vorbin/Cappellari & Copin (2003) Voronoi-binning implementation --
# explicit user request: "instead of voronoi binning i want to use the new
# version of the code https://pypi.org/project/powerbin... no need to change
# filenames or column names, just replace the engine." Confirmed by tracing
# every call site: `define_voronoi_bins` (this file) is the only place that
# calls the binning library itself -- every downstream consumer
# (`apply_voronoi_bins`/`voronoi_binning`, the saved FITS columns, everything
# in aps_ifu_Gal.py/aps_ifu_ExGal.py) only ever touches the resulting integer
# `binNum` array and has no idea which algorithm produced it, so the function/
# column names (`VORONOI`/`VORONOI_GAL` config keys, `BIN_ID` etc.) are
# unchanged -- only the engine inside `define_voronoi_bins` differs.
#
# License note (disclosed, not silently absorbed): PowerBin's license is
# "non-commercial use" / "redistribution... strictly prohibited without
# prior written permission from the author" -- materially different from
# every other PyAPS dependency (all permissive). Accepted explicitly for
# WEAVE/PyAPS's non-commercial consortium-science use case; see
# pyproject.toml's own comment on this dependency.
#
# Deliberately *not* imported at module level (real production bug, found
# live: this module is also imported transitively by aps_MOSviewer.py, via
# apsPlot/emi.py's own `ExGalPrepare.compute_equivalent_width` call for the
# explorer's MOS emission-line figure -- a genuinely different, unrelated
# function, nowhere near powerbin/Voronoi binning). `powerbin` is scoped to
# pyproject.toml's own `[pipeline]` extras precisely so the explorer's own
# server deployment (`[server]` extras only, see the Dockerfile) never needs
# it -- a module-level `import powerbin` here broke that isolation, crashing
# gunicorn worker boot with `ModuleNotFoundError: No module named 'powerbin'`
# on every single explorer deployment, confirmed live, not hypothetical, and
# not fixed by clearing the Docker build cache. Deferred into
# define_voronoi_bins itself instead -- confirmed by tracing every call site
# (see the comment above) that it's the *only* function in the whole PyAPS
# tree that actually calls into powerbin, and every one of its own callers
# is a pipeline script (aps_ifu_v0.py/aps_ifu_Gal.py/aps_ifu_ExGal.py/
# aps_cubepreview.py/aps_ifu_spaxel_contrib.py), never the explorer -- so
# only a genuine pipeline run, which already installs `[pipeline]`, ever
# actually needs powerbin importable at all.

"""
PURPOSE:
  This file contains a collection of functions necessary to spatially bin
  the data (Voronoi-style adaptive binning). Binning is performed with the
  PowerBin algorithm of Cappellari (2025, MNRAS, 544, 1432) -- a successor
  to the earlier Voronoi/Cappellari & Copin (2003) approach this module
  used until now, built around Centroidal Power Diagrams for faster,
  more numerically robust, always-connected/-convex bins.
"""


def sn_func(index, signal=None, noise=None, covar_vor=0.00):
    """
    This function is passed to the Voronoi binning routine of Cappellari &
    Copin 2003 (ui.adsabs.harvard.edu/?#abs/2003MNRAS.342..345C) and used to
    estimate the noise in the bin from the noise in the spaxels. This
    implementation is identical to the default one, but accounts for spatial
    correlations in the noise by applying an empirical equation (see e.g.
    Garcia-Benito et al. 2015;
    ui.adsabs.harvard.edu/?#abs/2015A&A...576A.135G) together with the
    parameter defined in the Config-file.
    """

    # Add the noise in the spaxels to obtain the noise in the bin
    sn = np.sum(signal[index]) / np.sqrt(np.sum(noise[index] ** 2))

    # Account for spatial correlations in the noise by applying an empirical
    # equation (see e.g. Garcia-Benito et al. 2015;
    # ui.adsabs.harvard.edu/?#abs/2015A&A...576A.135G)
    sn /= 1 + covar_vor * np.log10(index.size)

    return sn


def resolve_ebmv_extinction(configs, extinction_corr=None, extinction_ebv_fixed=None):
    """Resolve the ExGal config's 'EBmV' key (plus any explicit override)
    into the APSOB extinction_corr/extinction_ebv_scale/extinction_ebv_fixed
    keyword arguments -- shared by MOS (aps_mosExGal.py) and IFU
    (aps_ifu_ExGal.py), since both build their spectra through APSOB and
    both feed PPXF/EMIPPXF/LS from that same, single pre-corrected
    spectrum (see doc/aps_rr.md's "Galactic Extinction Correction"
    section for the underlying SFD98+Fitzpatrick99 machinery, added for
    REDROCK; this reuses it for ExGal -- added 2026-09-01).

    Unlike the REDROCK/classification use of this same machinery, ExGal
    only ever processes GALAXY/QSO targets (`working_classlist =
    ['GALAXY','QSO']` in aps_mosExGal.py) -- background sources sit
    beyond the Milky Way's entire dust column, so there is no "over-
    correcting a finite-distance foreground star" ambiguity here. Full-
    strength SFD correction (ebv_scale=1.0) is simply the physically
    correct choice, with no per-target caveat to hedge with a partial
    ebv_scale the way the STAR/REDROCK case needs.

    Precedence (argument > config > default), matching the pattern
    already used for `spaxel_weighted_lsf` in aps_ifu_ExGal.py:
      extinction_corr=False (explicit)  -> correction OFF entirely,
          regardless of the config -- e.g. for a diagnostic run.
      extinction_corr=True (explicit) and extinction_ebv_fixed given
          -> that fixed E(B-V) [mag], applied to every target.
      extinction_corr=True (explicit), no fixed value given
          -> automatic per-target SFD sky-position lookup.
      extinction_corr=None (not given, the normal case) -> read the
          config's 'EBmV' key instead:
            "None" / None / missing / "" -> automatic per-target SFD
                lookup (same as extinction_corr=True with no fixed
                value) -- this is the new default behaviour: every
                bundled ExGal_configs/*.json ships EBmV="None", so
                Galactic extinction correction is now ON by default
                for all ExGal processing.
            a number -> that fixed E(B-V) [mag], applied to every
                target in this run (e.g. a manually-verified value for
                one field, in place of the automatic per-target
                lookup).

    Parameters
    ----------
    configs : dict -- the loaded ExGal JSON config (must contain 'EBmV'
        if extinction_corr is not explicitly given)
    extinction_corr : bool or None -- explicit override, see above
    extinction_ebv_fixed : float or None -- explicit fixed E(B-V) [mag],
        only used when extinction_corr=True is also given explicitly

    Returns
    -------
    dict with keys extinction_corr, extinction_ebv_scale,
    extinction_ebv_fixed -- ready to `**`-splat into an APSOB(...) call.
    """
    off = dict(extinction_corr=False, extinction_ebv_scale=1.0,
               extinction_ebv_fixed=None)

    if extinction_corr is False:
        return off

    if extinction_corr is True:
        return dict(extinction_corr=True, extinction_ebv_scale=1.0,
                    extinction_ebv_fixed=extinction_ebv_fixed)

    # extinction_corr is None -- defer to the config's EBmV key.
    ebmv_cfg = configs.get('EBmV', None)
    ebmv_str = str(ebmv_cfg).strip().lower()
    if ebmv_cfg is None or ebmv_str in ('none', 'null', ''):
        return dict(extinction_corr=True, extinction_ebv_scale=1.0,
                    extinction_ebv_fixed=None)

    try:
        ebv_fixed = float(ebmv_cfg)
    except (TypeError, ValueError):
        print(f"WARNING: EBmV config value {ebmv_cfg!r} is neither "
              f"'None' nor a number -- disabling extinction correction "
              f"for this run")
        return off

    return dict(extinction_corr=True, extinction_ebv_scale=1.0,
                extinction_ebv_fixed=ebv_fixed)


def note_reddening_unused(configs):
    """Print a one-time, explicit notice that the ExGal config's
    'REDDENING' key is intentionally unused -- retired, not a bug.

    'REDDENING' (e.g. [0.1, 0.1] in most bundled configs) corresponds to
    ppxf's own optional `reddening=` free-parameter dust-law fit -- a
    different mechanism from 'EBmV' (which pre-corrects the observed
    spectrum for KNOWN Galactic foreground extinction before any fit
    runs; see resolve_ebmv_extinction above). It was traced (1 Sep 2026)
    and found to never be read from the config dict anywhere in the
    codebase: the actual `ppxf(...)` calls in MOSExGalPPXF.py/
    IFUExGalPPXF.py's `run_ppxf` already use `mdegree=<config MDEG>` (a
    multiplicative polynomial) for exactly this kind of residual
    continuum-shape correction, and ppxf's own documentation is explicit
    that combining `mdegree>0` with `reddening=` in the same fit is a
    degenerate double-parameterisation of the same effect. Formally
    retired rather than wired up, to avoid destabilising the existing
    stellar-kinematics/population fits with an unvalidated methodology
    change -- see doc/aps_rr.md's "Reused for ExGal" subsection.

    Call once per run (not per bin/target) right after loading the
    config, so this shows up once in the log rather than being silently
    forgotten -- exactly the kind of mystery 'EBmV' was before it got
    traced and fixed.
    """
    redd = configs.get('REDDENING', None)
    redd_str = str(redd).strip().lower()
    if redd is not None and redd_str not in ('none', 'null', ''):
        print(f"NOTE: config key 'REDDENING' ({redd!r}) is intentionally "
              f"unused -- ppxf's own mdegree={configs.get('MDEG', '?')} "
              f"multiplicative polynomial already handles residual "
              f"continuum-shape correction, and combining mdegree>0 with "
              f"a reddening= fit would be a degenerate double-"
              f"parameterisation per ppxf's own documentation. This is "
              f"retired, not a bug -- see doc/aps_rr.md.")


def write_ebmv_header(header, configs):
    """Stamp the actually-applied Galactic extinction correction into a
    FITS header -- shared by PPXF/EMIPPXF/LS output HDUs so the same
    provenance is visible everywhere, not just on the EMIPPXF table.

    Reads `configs['EBmV_APPLIED']` (set by `proc_mosExGaL`/
    `ifu_ExGal_prepare` right after resolving/measuring it -- see
    `resolve_ebmv_extinction`): 'OFF', a real float E(B-V) [mag], or the
    legacy 'AUTO (per-target SFD98)' string (only if the real median
    couldn't be recomputed -- see that fallback's own warning). For MOS,
    where each target gets its own per-target SFD lookup,
    `EBmV_APPLIED` is the *median* across the batch (mirrors many targets
    close together on sky) -- `EBmV_APPLIED_MIN`/`_MAX`, if present, give
    the real spread across the batch instead of just its centre.

    Parameters
    ----------
    header : astropy.io.fits.Header -- modified in place
    configs : dict -- must contain 'EBmV_APPLIED' (set upstream)
    """
    applied = configs.get('EBmV_APPLIED', 'OFF')
    header['EBMVAPPL'] = (str(applied) if isinstance(applied, str) else applied,
                           'Galactic E(B-V) actually applied (SFD98+Fitzpatrick99)')
    if 'EBmV_APPLIED_MIN' in configs and 'EBmV_APPLIED_MAX' in configs:
        header['EBMVMIN'] = (configs['EBmV_APPLIED_MIN'], 'Min E(B-V) across this batch (MOS: per-target)')
        header['EBMVMAX'] = (configs['EBmV_APPLIED_MAX'], 'Max E(B-V) across this batch (MOS: per-target)')


def compute_equivalent_width(line_flux, rest_wavelength, redshift, wave, continuum):
    """Equivalent width of one emission line: integrated line flux divided
    by the local stellar continuum flux density at the line's observed
    (redshifted) wavelength, sampled by linear interpolation of the
    fitted stellar continuum model (ppxf's own `stellar_bestfit`/
    `MODEL_PPXF`) -- shared by MOS (`MOSExGalEMIPPXF.py`) and IFU
    (`IFUExGalEMIPPXF.py`), added 1 Sep 2026 per user/team request.

    Sign convention: **positive for genuine emission** (a real emission
    line has `line_flux > 0` over a positive continuum, so `EW > 0`).
    This is the opposite convention from the classical stellar-absorption
    equivalent width used elsewhere in this codebase for the Lick-style
    indices (`MOSExGalLS.py`/`IFUExGalLS.py`'s `Hbeta`/`Mgb`/`Fe5270`/
    etc. columns, positive for absorption, negative for net emission
    filling) -- chosen to match common emission-line-catalogue convention
    (e.g. SDSS) and be the most directly useful sign for someone
    measuring emission-line strength, but do not directly compare an
    `EW_<line>` value against an LS index without accounting for the
    sign flip.

    Error propagation is flux-error-only (`ERR_EW = ERR_FLUX / continuum`)
    -- the continuum itself is treated as known/fixed, a standard
    simplification since its uncertainty is normally much smaller than
    the line-flux uncertainty at any S/N where the line is usefully
    measured at all; this is not a full covariance propagation.

    Parameters
    ----------
    line_flux : float -- integrated line flux, e.g. erg/s/cm^2 (same
        flux unit as `continuum` integrated over wavelength)
    rest_wavelength : float -- line's rest-frame wavelength [Angstrom]
    redshift : float -- target's redshift, to get the observed wavelength
    wave : array -- wavelength grid the continuum model is sampled on
        [Angstrom], e.g. `np.exp(logLam[i])`
    continuum : array -- fitted stellar continuum model, same length as
        `wave`, in flux density units (e.g. erg/s/cm^2/Angstrom)

    Returns
    -------
    ew : float -- equivalent width [Angstrom], NaN if the line falls
        outside `wave`'s range or the local continuum is non-positive/
        non-finite/unavailable
    continuum_at_line : float -- the interpolated continuum value used
        (NaN under the same conditions as `ew`) -- returned so callers
        can also propagate `ERR_EW` without a second interpolation
    """
    if (line_flux is None or not np.isfinite(line_flux)
            or rest_wavelength is None or not np.isfinite(rest_wavelength)
            or continuum is None or wave is None or len(wave) == 0):
        return np.nan, np.nan
    lam_obs = rest_wavelength * (1.0 + redshift)
    if lam_obs < wave[0] or lam_obs > wave[-1]:
        return np.nan, np.nan
    continuum_at_line = float(np.interp(lam_obs, wave, continuum))
    if not np.isfinite(continuum_at_line) or continuum_at_line <= 0:
        return np.nan, np.nan
    return line_flux / continuum_at_line, continuum_at_line


def prepare_table(
    aps_id,
    targid,
    cname,
    x,
    y,
    z,
    zerr,
    healpix,
    x_0,
    y_0,
    signal,
    noise,
    pixelsize,
    snr,
    rootname,
    outdir,
    configs,
):
    """
    FOR MOS mode ONLY
    Create a Table structure file for all targets
    """

    ExGalutil.prettyOutput_Done("Preparing data to generate the Table file.")
    logging.info("Preparing data to generate the Table file.")

    binNum = np.arange(0, len(aps_id))
    xNode = x[:]
    yNode = y[:]
    sn = snr[:]
    nPixels = np.ones(len(aps_id))
    binNum_outside = np.array([], dtype=np.int32)

    ubins = np.unique(binNum)
    nbins = len(ubins)
    binNum_long = np.zeros(len(x))
    binNum_long[:] = np.nan
    binNum_long[:] = binNum

    # Save bintable: data for *ALL* spectra inside and outside of the Voronoi region!
    ref_table_path = save_table(
        rootname,
        outdir,
        aps_id,
        targid,
        cname,
        x,
        y,
        z,
        zerr,
        healpix,
        x_0,
        y_0,
        signal,
        snr,
        binNum_long,
        ubins,
        xNode,
        yNode,
        sn,
        nPixels,
        pixelsize,
        configs,
    )

    return binNum


def adaptive_flux_filter(
    signal,
    noise,
    snr_min,
    filter_mode="safe",
    percentile=10,
    delta_fallback=3.0,
    min_spaxels_transition=10,
    min_keep_fraction=0.1,
    verbose=True,
):
    """
    Adaptive flux filtering with comprehensive error handling.

    This function applies flux-based filtering to remove low surface brightness
    regions before spatial binning. It includes multiple fallback mechanisms to
    ensure robustness across different data types (galaxies, stars, QSOs, etc.).

    Parameters
    ----------
    signal : ndarray
        Signal/flux values for each spaxel
    noise : ndarray
        Noise values for each spaxel
    snr_min : float
        Target SNR threshold for binning (used to find transition zone)
    filter_mode : str, optional
        Filtering strategy:
        - 'safe': Conservative mode with multiple fallbacks (DEFAULT)
        - 'transition': Original BAYES-LOSVD method (SNR transition zone)
        - 'percentile': Percentile-based removal
        - 'none': Disable filtering (returns all spaxels)
    percentile : float, optional
        For percentile mode: remove lowest N% by flux. Default: 10
        Lower values = more aggressive filtering
    delta_fallback : float, optional
        SNR window for transition zone. Default: 3.0
    min_spaxels_transition : int, optional
        Minimum spaxels needed in transition zone. Default: 10
    min_keep_fraction : float, optional
        Safety threshold: always keep at least this fraction of data.
        Default: 0.1 (10%)
    verbose : bool, optional
        Print diagnostic information. Default: True

    Returns
    -------
    idx : ndarray (bool)
        Boolean mask: True = keep spaxel, False = remove spaxel
    flux_thresh : float
        Applied flux threshold value
    method_used : str
        Description of which method was actually used

    Notes
    -----
    FILTER_MODE DESCRIPTIONS:

    'safe' mode (RECOMMENDED):
        1. Tries SNR transition zone method (original BAYES-LOSVD)
        2. If too few spaxels found, widens transition zone automatically
        3. If still fails, falls back to percentile method
        4. Always ensures minimum fraction of data is kept
        5. Never fails - worst case returns all data with warning

    'transition' mode:
        Original BAYES-LOSVD approach:
        - Finds spaxels within delta of target SNR
        - Uses mean flux from those spaxels as threshold
        - Removes all spaxels below that flux
        Good for: Extended objects with clear SNR gradient

    'percentile' mode:
        Removes lowest N% of spaxels by flux:
        - Simple, predictable behavior
        - No dependency on SNR distribution
        - Good for: Compact sources, point sources, varied data types

    'none' mode:
        Disables filtering entirely

    AUTOMATIC FALLBACKS IN 'SAFE' MODE:
    - If transition zone has < min_spaxels_transition: try wider delta
    - If wider delta fails: use percentile method
    - If percentile would remove > (1 - min_keep_fraction): reduce aggressiveness
    - If all else fails: keep all data

    Examples
    --------
    >>> # Safe mode with automatic fallbacks (recommended)
    >>> idx, thresh, method = adaptive_flux_filter(signal, noise, snr_min=3.0)

    >>> # More aggressive percentile-based filtering
    >>> idx, thresh, method = adaptive_flux_filter(
    ...     signal, noise, snr_min=3.0,
    ...     filter_mode='percentile', percentile=5
    ... )

    >>> # Disable filtering
    >>> idx, thresh, method = adaptive_flux_filter(
    ...     signal, noise, snr_min=3.0, filter_mode='none'
    ... )
    """

    n_total = len(signal)
    snr = signal / noise

    # Handle 'none' mode - no filtering
    if filter_mode == "none":
        if verbose:
            logging.info("Flux filter: DISABLED (mode='none')")
        return np.ones(n_total, dtype=bool), 0.0, "no_filtering"

    # Ensure minimum data retention
    min_keep = max(int(n_total * min_keep_fraction), 1)

    # ========================================================================
    # MODE: 'percentile' - Simple percentile-based removal
    # ========================================================================
    if filter_mode == "percentile":
        try:
            # Calculate flux threshold from percentile
            flux_thresh = np.percentile(signal[signal > 0], percentile)
            idx = signal >= flux_thresh

            # Safety check: ensure we keep minimum fraction
            if np.sum(idx) < min_keep:
                if verbose:
                    logging.warning(
                        f"Percentile filter would remove too much data "
                        + f"({np.sum(idx)}/{n_total} kept). "
                        + f"Adjusting to keep {min_keep} spaxels."
                    )
                # Keep top min_keep by signal
                sorted_idx = np.argsort(signal)[::-1]
                idx = np.zeros(n_total, dtype=bool)
                idx[sorted_idx[:min_keep]] = True
                flux_thresh = signal[sorted_idx[min_keep - 1]]

            method_used = f"percentile_{percentile}"

            if verbose:
                logging.info(
                    f"Flux filter (percentile mode): removed {n_total - np.sum(idx)}/{n_total} "
                    + f"({100*(n_total - np.sum(idx))/n_total:.1f}%), "
                    + f"threshold={flux_thresh:.2e}"
                )

            return idx, flux_thresh, method_used

        except Exception as e:
            logging.warning(f"Percentile filter failed: {e}. Keeping all data.")
            return np.ones(n_total, dtype=bool), 0.0, "percentile_failed_keepall"

    # ========================================================================
    # MODE: 'transition' - Original BAYES-LOSVD method
    # ========================================================================
    elif filter_mode == "transition":
        try:
            # Find spaxels in SNR transition zone
            idx_transition = np.abs(snr - snr_min) <= delta_fallback
            n_transition = np.sum(idx_transition)

            if n_transition >= min_spaxels_transition:
                # Sufficient spaxels in transition zone
                flux_thresh = np.median(signal[idx_transition])
                idx = signal >= flux_thresh

                # Safety check
                if np.sum(idx) < min_keep:
                    logging.warning(
                        f"Transition filter would remove too much data. "
                        + f"Keeping minimum {min_keep} spaxels."
                    )
                    sorted_idx = np.argsort(signal)[::-1]
                    idx = np.zeros(n_total, dtype=bool)
                    idx[sorted_idx[:min_keep]] = True
                    flux_thresh = signal[sorted_idx[min_keep - 1]]

                method_used = f"transition_delta{delta_fallback}"

                if verbose:
                    logging.info(
                        f"Flux filter (transition mode): {n_transition} spaxels in transition, "
                        + f"removed {n_total - np.sum(idx)}/{n_total} "
                        + f"({100*(n_total - np.sum(idx))/n_total:.1f}%), "
                        + f"threshold={flux_thresh:.2e}"
                    )

                return idx, flux_thresh, method_used

            else:
                # Too few spaxels in transition zone - fall back
                logging.warning(
                    f"Only {n_transition} spaxels in SNR transition zone "
                    + f"(delta={delta_fallback}). Using percentile fallback."
                )
                return adaptive_flux_filter(
                    signal,
                    noise,
                    snr_min,
                    filter_mode="percentile",
                    percentile=percentile,
                    min_keep_fraction=min_keep_fraction,
                    verbose=verbose,
                )

        except Exception as e:
            logging.warning(f"Transition filter failed: {e}. Keeping all data.")
            return np.ones(n_total, dtype=bool), 0.0, "transition_failed_keepall"

    # ========================================================================
    # MODE: 'safe' - Multi-stage approach with automatic fallbacks
    # ========================================================================
    elif filter_mode == "safe":
        # Strategy: Try transition method with progressively wider deltas,
        # fall back to percentile if needed

        try:
            # Stage 1: Try narrow transition zone (delta=2)
            idx_transition = np.abs(snr - snr_min) <= 2.0
            n_transition = np.sum(idx_transition)

            if n_transition >= min_spaxels_transition:
                flux_thresh = np.median(signal[idx_transition])
                idx = signal >= flux_thresh

                if np.sum(idx) >= min_keep:
                    method_used = "safe_transition_delta2"
                    if verbose:
                        logging.info(
                            f"Flux filter (safe mode, delta=2): {n_transition} in transition, "
                            + f"removed {n_total - np.sum(idx)}/{n_total} "
                            + f"({100*(n_total - np.sum(idx))/n_total:.1f}%), "
                            + f"threshold={flux_thresh:.2e}"
                        )
                    return idx, flux_thresh, method_used

            # Stage 2: Try standard transition zone (delta=3)
            idx_transition = np.abs(snr - snr_min) <= 3.0
            n_transition = np.sum(idx_transition)

            if n_transition >= min_spaxels_transition:
                flux_thresh = np.median(signal[idx_transition])
                idx = signal >= flux_thresh

                if np.sum(idx) >= min_keep:
                    method_used = "safe_transition_delta3"
                    if verbose:
                        logging.info(
                            f"Flux filter (safe mode, delta=3): {n_transition} in transition, "
                            + f"removed {n_total - np.sum(idx)}/{n_total} "
                            + f"({100*(n_total - np.sum(idx))/n_total:.1f}%), "
                            + f"threshold={flux_thresh:.2e}"
                        )
                    return idx, flux_thresh, method_used

            # Stage 3: Try wider transition zone (delta=5)
            idx_transition = np.abs(snr - snr_min) <= 5.0
            n_transition = np.sum(idx_transition)

            if n_transition >= min_spaxels_transition:
                flux_thresh = np.median(signal[idx_transition])
                idx = signal >= flux_thresh

                if np.sum(idx) >= min_keep:
                    method_used = "safe_transition_delta5"
                    if verbose:
                        logging.info(
                            f"Flux filter (safe mode, delta=5): {n_transition} in transition, "
                            + f"removed {n_total - np.sum(idx)}/{n_total} "
                            + f"({100*(n_total - np.sum(idx))/n_total:.1f}%), "
                            + f"threshold={flux_thresh:.2e}"
                        )
                    return idx, flux_thresh, method_used

            # Stage 4: Transition method didn't work - fall back to percentile
            if verbose:
                logging.info(
                    f"Flux filter (safe mode): Transition zones insufficient, "
                    + f"using percentile={percentile} fallback"
                )

            return adaptive_flux_filter(
                signal,
                noise,
                snr_min,
                filter_mode="percentile",
                percentile=percentile,
                min_keep_fraction=min_keep_fraction,
                verbose=verbose,
            )

        except Exception as e:
            logging.warning(f"Safe mode filter failed: {e}. Keeping all data.")
            return np.ones(n_total, dtype=bool), 0.0, "safe_failed_keepall"

    else:
        raise ValueError(
            f"Unknown filter_mode: '{filter_mode}'. "
            + f"Options: 'safe', 'transition', 'percentile', 'none'"
        )


def spatial_bin_with_provenance(
    cube,
    bin_size,
    min_snr=0.0,
    apply_flux_filter=False,
    flux_filter_mode="safe",
    flux_filter_snr_min=3.0,          # NEW - decoupled from min_snr
    flux_filter_delta=2.0,            # NEW
    flux_filter_min_spaxels=10,       # NEW
    flux_filter_percentile=10,
    flux_filter_min_keep_fraction=0.1,
    verbose=True,
):
    """
    Extended spatial binning with optional flux-based pre-filtering.

    Applies flux filtering FIRST (before SNR filtering) to remove low surface
    brightness regions. Then proceeds with standard spatial binning while
    maintaining full provenance tracking.

    This function spatially bins spectroscopic data while maintaining provenance
    tracking of original spaxels and handling various data quality checks.

    Parameters
    ----------
    cube : dict
        Dictionary containing:
            'x', 'y'     : coordinates (1D arrays)
            'signal'     : signal values (1D array)
            'noise'      : noise values (1D array)
            'snr'        : signal-to-noise ratio (1D array)
            'z', 'zerr'  : redshift and error (1D arrays)
            'aps_id'     : unique spaxel identifiers (1D array)
            'targid', 'cname': arrays of strings (1D arrays)
            'healpix'    : HEALPix index (1D array)
            'spec'       : 2D spectral data (n_lambda, n_spaxels)
            'error'      : 2D error data (n_lambda, n_spaxels)
            'wave'       : wavelength array (n_lambda,)
            'x_0', 'y_0' : reference coordinates (1D arrays)
            'pixelsize'  : pixel size (scalar)
            'velscale'   : original velocity scale (scalar)

    bin_size : float
        Size of each spatial bin in the same units as x, y coordinates.
        Must be positive.

    min_snr : float, optional
        Minimum SNR threshold for valid spaxels. Default is 0.0.
        Applied AFTER flux filtering.

    apply_flux_filter : bool, optional
        If True, apply flux-based pre-filtering before SNR filtering.
        Default: False (maintains backward compatibility)

    flux_filter_mode : str, optional
        Flux filtering strategy. Default: 'safe'
        Options:
        - 'safe': Conservative with automatic fallbacks (RECOMMENDED)
                  Works for galaxies, point sources, all target types
        - 'transition': Original BAYES-LOSVD method (SNR transition zone)
                       Best for extended sources with clear gradients
        - 'percentile': Simple percentile-based removal
                       Good for compact sources, point sources
        - 'none': Disable flux filtering

    flux_filter_percentile : float, optional
        For percentile mode: percentage of lowest-flux spaxels to remove.
        Default: 10 (remove lowest 10% by flux)
        Lower = more aggressive filtering

    flux_filter_min_keep_fraction : float, optional
        Safety threshold: always keep at least this fraction of data.
        Default: 0.1 (keep at least 10% of spaxels)
        Prevents over-aggressive filtering on any target type.

    verbose : bool, optional
        Print diagnostic information. Default: True

    Returns
    -------
    spatial_bins : dict
        Dictionary containing binned data with keys:
            'x', 'y' : bin center coordinates
            'signal', 'noise', 'snr' : binned photometric quantities
            'aps_ids' : list of original spaxel IDs per bin (PROVENANCE)
            'bin_id' : unique bin identifiers
            'bin_size' : input bin size
            'x_edges', 'y_edges' : bin edge arrays
            'z', 'zerr', 'targid', 'cname', 'healpix' : metadata from last spaxel in bin
            'x_0', 'y_0' : reference coordinates
            'pixelsize' : original pixel size
            'velscale' : original velocity scale
            'spec' : binned spectra (n_lambda, n_bins)
            'error' : binned errors (n_lambda, n_bins)
            'wave' : wavelength array (n_lambda,)
            'flag' : bin validity flags (1=valid, 0=empty)
            'flux_filter_applied' : bool, whether flux filtering was used
            'flux_filter_method' : str, which method was used
            'n_original_spaxels' : int, number before flux filtering
            'n_filtered_spaxels' : int, number after flux filtering

    aps_id_to_spatial_bin : dict
        Mapping from original spaxel IDs to bin IDs for provenance tracking.
        IMPORTANT: Maps ALL original spaxel IDs, including those removed
        by flux filtering (mapped to -1) and SNR filtering (mapped to -999).

    Raises
    ------
    TypeError
        If cube is not a dictionary
    ValueError
        If bin_size is not positive, arrays have mismatched shapes, or data is invalid
    KeyError
        If required keys are missing from cube dictionary

    Notes
    -----
    FILTERING ORDER:
    1. Flux filtering (if enabled) - removes low surface brightness
    2. SNR filtering (always applied) - removes low SNR spaxels
    3. Spatial binning - groups remaining spaxels into bins

    PROVENANCE TRACKING:
    The aps_id_to_spatial_bin dictionary maps ALL original spaxel IDs:
    - bin_id >= 0: spaxel included in this bin
    - bin_id = -1: removed by flux filter
    - bin_id = -999: removed by SNR filter
    This ensures complete traceability of every input spaxel.

    FLUX FILTER RECOMMENDATIONS BY TARGET TYPE:
    - Extended galaxies: 'safe' mode (default)
    - Compact galaxies: 'safe' or 'percentile' with percentile=5-10
    - Point sources (stars, QSOs): 'percentile' with percentile=20-30 or 'none'
    - Mixed fields: 'safe' mode (automatically adapts)
    - Unknown targets: 'safe' mode (robust to all cases)

    Examples
    --------
    >>> # Standard usage without flux filtering (backward compatible)
    >>> bins, provenance = spatial_bin_with_provenance(cube, bin_size=0.5)

    >>> # With flux filtering (recommended for galaxies)
    >>> bins, provenance = spatial_bin_with_provenance(
    ...     cube, bin_size=0.5, apply_flux_filter=True
    ... )

    >>> # Custom flux filtering for faint galaxies
    >>> bins, provenance = spatial_bin_with_provenance(
    ...     cube, bin_size=0.5,
    ...     apply_flux_filter=True,
    ...     flux_filter_mode='percentile',
    ...     flux_filter_percentile=5  # More aggressive
    ... )

    >>> # For point sources - disable or use gentle filtering
    >>> bins, provenance = spatial_bin_with_provenance(
    ...     cube, bin_size=0.5,
    ...     apply_flux_filter=True,
    ...     flux_filter_mode='percentile',
    ...     flux_filter_percentile=30  # Very gentle
    ... )
    """

    from PyAPS import ExGalutil

    ExGalutil.prettyOutput_Running(
        f"Spatial Binning Mode with spatial bin size {bin_size} arcsec"
    )

    # ==================== INPUT VALIDATION ====================

    # Check basic input types
    if not isinstance(cube, dict):
        raise TypeError("cube must be a dictionary")

    if not isinstance(bin_size, (int, float)) or bin_size <= 0:
        raise ValueError("bin_size must be a positive number")

    if not isinstance(min_snr, (int, float)):
        raise ValueError("min_snr must be a number")

    # Define required keys for the cube dictionary
    required_keys = [
        "x",
        "y",
        "signal",
        "noise",
        "snr",
        "aps_id",
        "z",
        "zerr",
        "targid",
        "cname",
        "healpix",
        "spec",
        "error",
        "wave",
        "x_0",
        "y_0",
        "pixelsize",
        "velscale",
    ]

    # Check for missing keys
    missing_keys = [key for key in required_keys if key not in cube]
    if missing_keys:
        raise KeyError(f"Missing required keys in cube: {missing_keys}")

    # ==================== DATA EXTRACTION AND CONVERSION ====================

    try:
        # Convert all arrays to numpy arrays with EXPLICIT float64 for coordinates
        x = np.asarray(cube["x"], dtype=np.float64)
        y = np.asarray(cube["y"], dtype=np.float64)
        signal = np.asarray(cube["signal"], dtype=float)
        noise = np.asarray(cube["noise"], dtype=float)
        snr = np.asarray(cube["snr"], dtype=float)
        aps_id = np.asarray(cube["aps_id"])
        z = np.asarray(cube["z"], dtype=float)
        zerr = np.asarray(cube["zerr"], dtype=float)
        targid = np.asarray(cube["targid"])
        cname = np.asarray(cube["cname"])
        healpix = np.asarray(cube["healpix"])
        x_0 = np.asarray(cube["x_0"], dtype=float)
        y_0 = np.asarray(cube["y_0"], dtype=float)

        # 2D arrays
        spec = np.asarray(cube["spec"], dtype=float)
        error = np.asarray(cube["error"], dtype=float)
        wave = np.asarray(cube["wave"], dtype=float)

        # Scalars
        pixelsize = float(cube["pixelsize"])
        velscale = float(cube["velscale"])

    except Exception as e:
        raise ValueError(f"Error converting cube arrays to numpy arrays: {e}")

    # Store original counts for provenance
    n_original = len(x)

    # ==================== ARRAY SHAPE VALIDATION ====================

    # Check that all 1D arrays have same length
    arrays_1d = {
        "x": x,
        "y": y,
        "signal": signal,
        "noise": noise,
        "snr": snr,
        "aps_id": aps_id,
        "z": z,
        "zerr": zerr,
        "targid": targid,
        "cname": cname,
        "healpix": healpix,
        "x_0": x_0,
        "y_0": y_0,
    }

    lengths = {name: len(arr) for name, arr in arrays_1d.items()}
    unique_lengths = set(lengths.values())

    if len(unique_lengths) != 1:
        mismatched = {name: length for name, length in lengths.items()}
        raise ValueError(f"Inconsistent array lengths: {mismatched}")

    n_spaxels = len(x)

    # Check 2D arrays
    if spec.shape[1] != n_spaxels:
        raise ValueError(
            f"spec shape {spec.shape} inconsistent with n_spaxels={n_spaxels}"
        )
    if error.shape[1] != n_spaxels:
        raise ValueError(
            f"error shape {error.shape} inconsistent with n_spaxels={n_spaxels}"
        )
    if spec.shape[0] != len(wave):
        raise ValueError(
            f"spec wavelength axis {spec.shape[0]} != wave length {len(wave)}"
        )

    # Check for NaN/Inf in coordinates
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("x or y coordinates contain NaN or Inf values")

    if verbose:
        logging.info(f"Input validation passed: {n_spaxels} spaxels")

    # ==================== FLUX FILTERING (STAGE 1) ====================
    # This is applied FIRST, before SNR filtering

    flux_filter_applied = False
    flux_filter_method = "none"
    flux_filter_idx = np.ones(n_spaxels, dtype=bool)
    flux_thresh = 0.0

    if apply_flux_filter:
        if verbose:
            ExGalutil.prettyOutput_Running("Applying flux-based pre-filter (Stage 1)")

        try:
            # Apply adaptive flux filter
            flux_filter_idx, flux_thresh, flux_filter_method = adaptive_flux_filter(
                signal=signal,
                noise=noise,
                snr_min=flux_filter_snr_min,          # use dedicated SNR, not min_snr
                filter_mode=flux_filter_mode,
                delta_fallback=flux_filter_delta,
                min_spaxels_transition=flux_filter_min_spaxels,
                percentile=flux_filter_percentile,
                min_keep_fraction=flux_filter_min_keep_fraction,
                verbose=verbose,
            )

            flux_filter_applied = True
            n_after_flux = np.sum(flux_filter_idx)
            n_removed_flux = n_spaxels - n_after_flux

            if verbose:
                ExGalutil.prettyOutput_Done(
                    f"Flux filtering: removed {n_removed_flux}/{n_spaxels} spaxels "
                    + f"({100*n_removed_flux/n_spaxels:.1f}%), method={flux_filter_method}"
                )
                logging.info(
                    f"Flux filter: {n_after_flux} spaxels remain, "
                    + f"threshold={flux_thresh:.2e}"
                )

            # Filter all arrays
            x = x[flux_filter_idx]
            y = y[flux_filter_idx]
            signal = signal[flux_filter_idx]
            noise = noise[flux_filter_idx]
            snr = snr[flux_filter_idx]
            aps_id_filtered = aps_id[flux_filter_idx]
            z = z[flux_filter_idx]
            zerr = zerr[flux_filter_idx]
            targid = targid[flux_filter_idx]
            cname = cname[flux_filter_idx]
            healpix = healpix[flux_filter_idx]
            x_0 = x_0[flux_filter_idx]
            y_0 = y_0[flux_filter_idx]
            spec = spec[:, flux_filter_idx]
            error = error[:, flux_filter_idx]

        except Exception as e:
            logging.error(
                f"Flux filtering failed: {e}. Proceeding without flux filter."
            )
            flux_filter_applied = False
            flux_filter_method = "failed_keepall"
            flux_filter_idx = np.ones(n_spaxels, dtype=bool)
            aps_id_filtered = aps_id
    else:
        # No flux filtering - keep all
        if verbose:
            logging.info("Flux filtering: DISABLED")
        aps_id_filtered = aps_id

    n_after_flux = len(x)

    # ==================== SNR FILTERING (STAGE 2) ====================
    # This is the standard SNR filter, applied after flux filtering

    if verbose:
        ExGalutil.prettyOutput_Running(
            f"Applying SNR filter (Stage 2): min_snr={min_snr}"
        )

    # Apply SNR threshold with buffer
    snr_buffer = 0.1 * min_snr
    valid_snr = snr >= (min_snr - snr_buffer)

    # Additional validity checks
    valid_signal = np.isfinite(signal) & (signal > 0)
    valid_noise = np.isfinite(noise) & (noise > 0)
    valid_coords = np.isfinite(x) & np.isfinite(y)

    # Combine all validity checks
    valid = valid_snr & valid_signal & valid_noise & valid_coords

    n_invalid = np.sum(~valid)
    if n_invalid > 0:
        if verbose:
            logging.warning(
                f"Removing {n_invalid} spaxels that fail SNR or validity checks"
            )

    # Filter arrays based on validity
    x_valid = x[valid]
    y_valid = y[valid]
    signal_valid = signal[valid]
    noise_valid = noise[valid]
    snr_valid = snr[valid]
    aps_id_valid = aps_id_filtered[valid]
    z_valid = z[valid]
    zerr_valid = zerr[valid]
    targid_valid = targid[valid]
    cname_valid = cname[valid]
    healpix_valid = healpix[valid]
    x_0_valid = x_0[valid]
    y_0_valid = y_0[valid]
    spec_valid = spec[:, valid]
    error_valid = error[:, valid]

    n_valid = len(x_valid)

    if verbose:
        ExGalutil.prettyOutput_Done(
            f"SNR filtering: {n_valid}/{n_after_flux} spaxels pass "
            + f"(removed {n_after_flux - n_valid})"
        )
        logging.info(f"Valid spaxels after SNR filter: {n_valid}")

    if n_valid == 0:
        raise ValueError("No valid spaxels remaining after filters!")

    # ==================== SPATIAL BINNING SETUP ====================

    if verbose:
        ExGalutil.prettyOutput_Running("Setting up spatial bins")

    # Calculate data extent
    x_min, x_max = np.float64(x_valid.min()), np.float64(x_valid.max())
    y_min, y_max = np.float64(y_valid.min()), np.float64(y_valid.max())

    # Add small padding
    padding = 0.01 * bin_size
    x_min -= padding
    x_max += padding
    y_min -= padding
    y_max += padding

    # Create bin edges with EXPLICIT float64
    x_edges = np.arange(x_min, x_max + bin_size, bin_size, dtype=np.float64)
    y_edges = np.arange(y_min, y_max + bin_size, bin_size, dtype=np.float64)

    n_bins_x = len(x_edges) - 1
    n_bins_y = len(y_edges) - 1
    n_bins_total = n_bins_x * n_bins_y

    if verbose:
        logging.info(f"Created {n_bins_x} x {n_bins_y} = {n_bins_total} spatial bins")

    # ==================== ASSIGN SPAXELS TO BINS ====================

    # Calculate bin indices for each spaxel using EXPLICIT float64
    # CRITICAL: Use consistent ROW-MAJOR ordering
    bin_idx_x = np.floor((x_valid.astype(np.float64) - x_min) / bin_size).astype(
        np.int64
    )
    bin_idx_y = np.floor((y_valid.astype(np.float64) - y_min) / bin_size).astype(
        np.int64
    )

    # Ensure indices are within bounds
    bin_idx_x = np.clip(bin_idx_x, 0, n_bins_x - 1)
    bin_idx_y = np.clip(bin_idx_y, 0, n_bins_y - 1)

    # Calculate 1D bin IDs using ROW-MAJOR order
    bin_ids = (bin_idx_y * n_bins_x + bin_idx_x).astype(np.int64)

    # ==================== AGGREGATE SPAXELS INTO BINS ====================

    if verbose:
        ExGalutil.prettyOutput_Running("Aggregating spaxels into bins")

    # Initialize output arrays
    unique_bin_ids = np.unique(bin_ids)
    n_filled_bins = len(unique_bin_ids)

    # Pre-allocate arrays
    bin_x = np.zeros(n_filled_bins, dtype=np.float64)
    bin_y = np.zeros(n_filled_bins, dtype=np.float64)
    bin_signal = np.zeros(n_filled_bins, dtype=float)
    bin_noise = np.zeros(n_filled_bins, dtype=float)
    bin_snr = np.zeros(n_filled_bins, dtype=float)
    bin_z = np.zeros(n_filled_bins, dtype=float)
    bin_zerr = np.zeros(n_filled_bins, dtype=float)
    bin_targid = np.empty(n_filled_bins, dtype=targid_valid.dtype)
    bin_cname = np.empty(n_filled_bins, dtype=cname_valid.dtype)
    bin_healpix = np.zeros(n_filled_bins, dtype=healpix_valid.dtype)
    bin_x_0 = np.zeros(n_filled_bins, dtype=float)
    bin_y_0 = np.zeros(n_filled_bins, dtype=float)
    bin_flag = np.ones(n_filled_bins, dtype=int)
    bin_aps_ids = []

    n_lambda = spec_valid.shape[0]
    bin_spec = np.zeros((n_lambda, n_filled_bins), dtype=float)
    bin_error = np.zeros((n_lambda, n_filled_bins), dtype=float)

    # Aggregate data for each bin
    for i, bid in enumerate(unique_bin_ids):
        mask = bin_ids == bid
        n_in_bin = np.sum(mask)

        # Store list of aps_ids for provenance
        bin_aps_ids.append(aps_id_valid[mask].tolist())

        # Bin center (mean position)
        bin_x[i] = np.mean(x_valid[mask])
        bin_y[i] = np.mean(y_valid[mask])

        # Sum signal and noise
        bin_signal[i] = np.sum(signal_valid[mask])
        bin_noise[i] = np.sqrt(np.sum(noise_valid[mask] ** 2))
        bin_snr[i] = bin_signal[i] / bin_noise[i] if bin_noise[i] > 0 else 0.0

        # Metadata from last spaxel (arbitrary choice)
        last_idx = np.where(mask)[0][-1]
        bin_z[i] = z_valid[last_idx]
        bin_zerr[i] = zerr_valid[last_idx]
        bin_targid[i] = targid_valid[last_idx]
        bin_cname[i] = cname_valid[last_idx]
        bin_healpix[i] = healpix_valid[last_idx]
        bin_x_0[i] = x_0_valid[last_idx]
        bin_y_0[i] = y_0_valid[last_idx]

        # Co-add spectra
        if n_in_bin == 1:
            bin_spec[:, i] = spec_valid[:, mask].flatten()
            bin_error[:, i] = error_valid[:, mask].flatten()
        else:
            bin_spec[:, i] = np.sum(spec_valid[:, mask], axis=1)
            bin_error[:, i] = np.sqrt(np.sum(error_valid[:, mask] ** 2, axis=1))

    if verbose:
        ExGalutil.prettyOutput_Done(
            f"Created {n_filled_bins} filled bins from {n_valid} spaxels"
        )
        logging.info(f"Filled bins: {n_filled_bins}/{n_bins_total}")

    # ==================== CREATE PROVENANCE MAPPING ====================

    # Map ALL original aps_ids to bin IDs for complete provenance
    # -1 = removed by flux filter
    # -999 = removed by SNR filter
    # >= 0 = bin ID

    aps_id_to_spatial_bin = {}

    # Start with all original aps_ids
    for orig_id in aps_id:
        aps_id_to_spatial_bin[orig_id] = -1  # Default: removed by flux filter

    # Update for those that passed flux filter but failed SNR filter
    for filt_id in aps_id_filtered[~valid]:
        aps_id_to_spatial_bin[filt_id] = -999  # Removed by SNR filter

    # Update for those that made it into bins
    for i, bid in enumerate(unique_bin_ids):
        mask = bin_ids == bid
        for aid in aps_id_valid[mask]:
            aps_id_to_spatial_bin[aid] = int(bid)

    # ==================== PACKAGE OUTPUT ====================

    spatial_bins = {
        "x": bin_x,
        "y": bin_y,
        "signal": bin_signal,
        "noise": bin_noise,
        "snr": bin_snr,
        "z": bin_z,
        "zerr": bin_zerr,
        "targid": bin_targid,
        "cname": bin_cname,
        "healpix": bin_healpix,
        "x_0": bin_x_0,
        "y_0": bin_y_0,
        "aps_ids": bin_aps_ids,
        "bin_id": unique_bin_ids,
        "bin_size": bin_size,
        "x_edges": x_edges,
        "y_edges": y_edges,
        "pixelsize": pixelsize,
        "velscale": velscale,
        "spec": bin_spec,
        "error": bin_error,
        "wave": wave,
        "flag": bin_flag,
        "flux_filter_applied": flux_filter_applied,
        "flux_filter_method": flux_filter_method,
        "n_original_spaxels": n_original,
        "n_filtered_spaxels": n_after_flux,
        "n_valid_spaxels": n_valid,
        "n_bins": n_filled_bins,
    }

    if verbose:
        ExGalutil.prettyOutput_Done(
            f"Spatial binning complete: {n_filled_bins} bins from {n_valid} spaxels"
        )

        # Print summary statistics
        print("\n" + "=" * 70)
        print("SPATIAL BINNING SUMMARY")
        print("=" * 70)
        print(f"Original spaxels:          {n_original}")
        if flux_filter_applied:
            print(
                f"After flux filter:         {n_after_flux} "
                + f"(removed {n_original - n_after_flux}, {100*(n_original - n_after_flux)/n_original:.1f}%)"
            )
            print(f"  Filter method:           {flux_filter_method}")
            print(f"  Flux threshold:          {flux_thresh:.2e}")
        print(
            f"After SNR filter:          {n_valid} "
            + f"(removed {n_after_flux - n_valid}, {100*(n_after_flux - n_valid)/n_after_flux:.1f}%)"
        )
        print(f"Final spatial bins:        {n_filled_bins}")
        print(f"Bin size:                  {bin_size:.3f} arcsec")
        print(f"Mean spaxels per bin:      {n_valid/n_filled_bins:.1f}")
        print(f"Median bin SNR:            {np.median(bin_snr):.1f}")
        print(f"SNR range:                 {bin_snr.min():.1f} - {bin_snr.max():.1f}")
        print("=" * 70 + "\n")

    return spatial_bins, aps_id_to_spatial_bin



def read_voronoi_fits_table(table_file):
    """
    Read Voronoi binning data from FITS table file.

    This function reads the output FITS table from the define_voronoi_bins function
    and extracts the necessary information for plotting Voronoi bins.

    Parameters
    ----------
    table_file : str
        Path to the FITS table file containing Voronoi binning results.
        Expected to contain columns like 'X', 'Y', 'BIN_ID', etc.

    Returns
    -------
    voronoi_data : dict
        Dictionary containing:
            'x' : x coordinates of all points
            'y' : y coordinates of all points
            'binNum' : Voronoi bin assignments
            'xNode' : x coordinates of Voronoi bin centers
            'yNode' : y coordinates of Voronoi bin centers
            'aps_id' : original spaxel IDs
            'signal' : signal values per point
            'snr' : SNR values per point
            'target_snr' : target SNR (if available in header)
            'n_bins' : number of Voronoi bins
            'n_points' : total number of points

    spatial_data : dict
        Dictionary containing spatial information that can be used
        to reconstruct spatial bins if needed:
            'x' : x coordinates
            'y' : y coordinates
            'aps_id' : spaxel IDs
            'signal', 'snr', etc. : other quantities

    Examples
    --------
    >>> voronoi_data, spatial_data = read_voronoi_fits_table('table.fits')
    >>> fig, ax, vor = plot_voronoi_bins_overlay(spatial_bins, voronoi_data)
    """

    print(f"Reading Voronoi data from: {table_file}")

    # try 10 times with delay to handle potential file access issues
    max_attempts = 10
    delay = 1.0  # seconds between attempts
    for attempt in range(max_attempts):
        try:
            hdul = fits.open(table_file)
            break
        except Exception:
            if attempt < max_attempts - 1:
                print(f"Attempt {attempt + 1} failed, waiting {delay}s...")
                time.sleep(delay)
            else:
                raise
    try:
        # Read the FITS table
        table = Table.read(hdul[1])  # Usually data is in extension 1
        header = hdul[1].header

        print(f"FITS table contains {len(table)} rows")
        print(f"Available columns: {table.colnames}")

        # ==================== EXTRACT COORDINATE DATA ====================

        # Try different possible column names for coordinates
        x_col_names = ["X", "x", "X_COORD", "RA", "x_coord"]
        y_col_names = ["Y", "y", "Y_COORD", "DEC", "y_coord"]

        x_col = None
        y_col = None

        for name in x_col_names:
            if name in table.colnames:
                x_col = name
                break

        for name in y_col_names:
            if name in table.colnames:
                y_col = name
                break

        if x_col is None or y_col is None:
            raise ValueError(
                f"Could not find X,Y coordinate columns. Available: {table.colnames}"
            )

        x = np.array(table[x_col])
        y = np.array(table[y_col])

        print(f"Using coordinates: {x_col}, {y_col}")

        # ==================== EXTRACT BIN ASSIGNMENTS ====================

        # Try different possible column names for bin IDs
        bin_col_names = [
            "BIN_ID",
            "BINNUM",
            "bin_id",
            "binNum",
            "BIN_NUM",
            "VORONOI_BIN",
        ]

        bin_col = None
        for name in bin_col_names:
            if name in table.colnames:
                bin_col = name
                break

        if bin_col is None:
            raise ValueError(
                f"Could not find bin ID column. Available: {table.colnames}"
            )

        binNum = np.array(table[bin_col])

        print(f"Using bin assignments: {bin_col}")
        print(f"Bin ID range: {np.min(binNum)} to {np.max(binNum)}")

        # Handle negative bin IDs (points outside Voronoi region)
        positive_mask = binNum >= 0
        n_inside = np.sum(positive_mask)
        n_outside = np.sum(~positive_mask)

        print(f"Points: {n_inside} inside Voronoi region, {n_outside} outside")

        # ==================== CALCULATE BIN CENTERS ====================

        # Get unique positive bin IDs
        unique_bins = np.unique(binNum[positive_mask])
        n_bins = len(unique_bins)

        print(f"Number of Voronoi bins: {n_bins}")

        # Calculate bin centers
        xNode = np.zeros(n_bins)
        yNode = np.zeros(n_bins)

        for i, bin_id in enumerate(unique_bins):
            mask = binNum == bin_id
            xNode[i] = np.mean(x[mask])
            yNode[i] = np.mean(y[mask])

        # ==================== EXTRACT OTHER QUANTITIES ====================

        # Extract other available quantities
        quantity_cols = ["SIGNAL", "SNR", "signal", "snr", "S_N", "APS_ID", "aps_id"]
        extracted_data = {}

        for col in table.colnames:
            if col not in [x_col, y_col, bin_col]:
                try:
                    extracted_data[col.lower()] = np.array(table[col])
                except:
                    continue

        # ==================== EXTRACT HEADER INFO ====================

        # Try to get target SNR from header
        target_snr = None
        target_snr_keys = ["TARGSNR", "TARGET_SNR", "TARG_SNR", "SNR_TARG"]

        for key in target_snr_keys:
            if key in header:
                target_snr = header[key]
                break

        # ==================== ASSEMBLE OUTPUT DICTIONARIES ====================

        voronoi_data = {
            "x": x,
            "y": y,
            "binNum": binNum,
            "xNode": xNode,
            "yNode": yNode,
            "unique_bins": unique_bins,
            "n_bins": n_bins,
            "n_points": len(x),
            "n_inside": n_inside,
            "n_outside": n_outside,
        }

        # Add target SNR if found
        if target_snr is not None:
            voronoi_data["target_snr"] = target_snr
            print(f"Target SNR: {target_snr}")

        # Add other quantities to voronoi_data
        for key, value in extracted_data.items():
            voronoi_data[key] = value

        # ==================== SUMMARY ====================

        print("\nVoronoi Data Summary:")
        print(f"  Total points: {len(x)}")
        print(f"  Inside region: {n_inside}")
        print(f"  Outside region: {n_outside}")
        print(f"  Voronoi bins: {n_bins}")
        print(
            f"  Coordinate range: X[{np.min(x):.2f}, {np.max(x):.2f}], Y[{np.min(y):.2f}, {np.max(y):.2f}]"
        )

        if "signal" in extracted_data:
            print(
                f"  Signal range: [{np.min(extracted_data['signal']):.3f}, {np.max(extracted_data['signal']):.3f}]"
            )

        if "snr" in extracted_data:
            print(
                f"  SNR range: [{np.min(extracted_data['snr']):.3f}, {np.max(extracted_data['snr']):.3f}]"
            )

        hdul.close()

        return voronoi_data

    except Exception as e:
        print(f"Error reading FITS file: {e}")
        raise


def vorbin_colormap(n_colors, seed=42, verbose=False):
    """
    Create a random colormap based on tab20 for any number of colors.

    This randomly assigns tab20 colors to indices, ensuring neighboring
    indices get different colors even for 10,000+ categories.

    Parameters
    ----------
    n_colors : int
        Number of colors needed (can be 10,000+)
    seed : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    colormap : array
        Array of RGBA colors that can be used as a colormap
        Shape: (n_colors, 4)

    Examples
    --------
    >>> # Create colormap for 10,000 bins
    >>> colors = create_random_tab20_colormap(10000)
    >>> fig, ax, vor = plot_voronoi_only(voronoi_data, cmap=colors)
    >>>
    >>> # Or use with debug function
    >>> fig, axes = debug_voronoi_visualization(spatial_bins, voronoi_data, cmap=colors)
    """

    np.random.seed(seed)

    print(f"Creating random tab20-based colormap for {n_colors} colors...")

    # Get the 20 tab20 colors
    tab20_colors = plt.cm.tab20(np.arange(20))

    # Randomly assign tab20 colors to all indices
    random_indices = np.random.randint(0, 20, size=n_colors)

    # Create the colormap array
    colormap = tab20_colors[random_indices]

    # Statistics
    unique_colors_used = len(np.unique(random_indices))
    color_counts = np.bincount(random_indices, minlength=20)

    if verbose:
        print(f"✓ Used {unique_colors_used}/20 tab20 colors")
        print(
            f"✓ Color distribution: min={np.min(color_counts)}, max={np.max(color_counts)}, avg={np.mean(color_counts):.1f}"
        )
        print(f"✓ Created colormap shape: {colormap.shape}")

    return colormap


def define_voronoi_bins(
    VORONOI,
    aps_id,
    targid,
    cname,
    x,
    y,
    z,
    zerr,
    healpix,
    x_0,
    y_0,
    signal,
    noise,
    pixelsize,
    snr,
    target_snr,
    covar_vor,
    idx_inside,
    idx_outside,
    rootname,
    outdir,
    configs,
):
    """
    Applies the Voronoi-binning algorithm of Cappellari & Copin (2003) to the data.

    This updated version supports use after a prior fixed-size spatial binning step.
    It preserves traceability to original spaxel IDs (aps_id) and assigns BIN_IDs to all
    spaxels, even if they are outside the Voronoi-binned region (negative BIN_IDs).

    Parameters
    ----------
    VORONOI : int
        Whether to apply Voronoi binning (1) or not (0).
    aps_id : array_like
        Original unique spaxel identifiers.
    targid, cname, x, y, z, zerr, healpix, x_0, y_0, signal, noise, snr : arrays
        Per-spaxel information to be preserved in output.
    pixelsize : float
        Pixel size for the Voronoi binning algorithm.
    target_snr : float or str
        Target S/N ratio for Voronoi binning (will be converted to float).
    covar_vor : float
        Spatial covariance correction factor.
    idx_inside, idx_outside : arrays
        Indices of valid/invalid pixels to include/exclude from Voronoi binning.
    rootname : str
        Root filename for output.
    outdir : str
        Directory where output files are saved.
    configs : dict
        Dictionary of configuration values used for writing FITS headers.

    Returns
    -------
    binNum : array
        Array of Voronoi bin IDs for valid pixels.
    """
    # Deferred import -- see this module's own top-of-file comment for
    # why (the explorer's server deployment must never need this
    # installed, only a real pipeline run does).
    import powerbin

    # Convert target_snr to float if it's a string
    if isinstance(target_snr, str):
        try:
            target_snr = float(target_snr)
            ExGalutil.prettyOutput_Done(
                f"Converted target_snr from string to float: {target_snr}"
            )
            logging.info(f"Converted target_snr from string to float: {target_snr}")
        except ValueError:
            ExGalutil.prettyOutput_Warning(
                f"Could not convert target_snr '{target_snr}' to float. Using default value 0.0"
            )
            logging.warning(
                f"Could not convert target_snr '{target_snr}' to float. Using default value 0.0"
            )
            target_snr = 0.0

    # Ensure covar_vor is also numeric
    if isinstance(covar_vor, str):
        try:
            covar_vor = float(covar_vor)
        except ValueError:
            covar_vor = 0.00

    sn_func_covariances = functools.partial(sn_func, covar_vor=covar_vor)

    if VORONOI == 1:
        ExGalutil.prettyOutput_Running("Defining the Voronoi bins")
        logging.info("Defining the Voronoi bins")

        attempts = 3
        success = False

        for attempt in range(attempts):
            try:
                # PowerBin works on (S/N)^2 ("capacity"), not plain S/N --
                # sn_func_covariances (unchanged, still the Garcia-Benito+15
                # correlated-noise correction) returns plain S/N, so square
                # it here rather than touching that function's own contract.
                #
                # signal/noise must be bound explicitly here: vorbin's own
                # voronoi_2d_binning used to supply its `sn_func` callback
                # with signal/noise itself (positionally, from the same
                # sliced arrays passed to voronoi_2d_binning) -- PowerBin's
                # capacity_spec(index) contract has no such auto-supply, so
                # without this binding sn_func's signal/noise stay at their
                # own None defaults (confirmed live: "'NoneType' object is
                # not subscriptable" the moment sn_func tried signal[index]).
                # Index space matches: PowerBin's capacity_spec receives
                # indices into the *same* xy array constructed just below
                # (i.e. into the idx_inside-sliced arrays), so signal/noise
                # must be sliced identically, not the full unfiltered arrays.
                sn_func_data = functools.partial(
                    sn_func_covariances,
                    signal=signal[idx_inside], noise=noise[idx_inside],
                )
                # sn_func indexes with `signal[index]` and calls
                # `index.size` -- both need a real ndarray. PowerBin's own
                # internal bin-accretion stage sometimes hands capacity_spec
                # a plain Python list rather than an array (confirmed live:
                # "'list' object has no attribute 'size'"), so normalize
                # here rather than assume sn_func's own input type.
                capacity_spec = lambda idx: sn_func_data(np.asarray(idx)) ** 2
                pb = powerbin.PowerBin(
                    np.column_stack([x[idx_inside], y[idx_inside]]),
                    capacity_spec,
                    target_capacity=float(target_snr) ** 2,
                    pixelsize=pixelsize,
                    verbose=0,
                )
                binNum = pb.bin_num
                # PowerBin's own bin centers are always literal centroids of
                # their member pixels (see `update_bins` in powerbin.py) --
                # there's no separate "propagating generator" vs "mass-
                # weighted centroid" distinction the way classic Voronoi/CVT
                # has, so the same coordinates fill both roles downstream.
                xNode, yNode = pb.xybin[:, 0], pb.xybin[:, 1]
                xBar,  yBar  = xNode, yNode
                sn, nPixels  = np.sqrt(pb.bin_capacity), pb.npix

                ExGalutil.prettyOutput_Done(
                    f"Defining the Voronoi bins with target_snr = {target_snr}"
                )
                print(
                    "             "
                    + str(np.max(binNum) + 1)
                    + " voronoi bins generated!"
                )
                logging.info(
                    str(np.max(binNum) + 1)
                    + f" Voronoi bins generated with target_snr = {target_snr}"
                )

                success = True
                break

            except Exception as e:
                ExGalutil.prettyOutput_Warning(
                    f"Voronoi binning attempt {attempt + 1} failed"
                )
                print(
                    f"             Attempt {attempt + 1}: The Voronoi-binning routine failed."
                )
                print(f"             Error: {str(e)}\n")
                logging.warning(
                    f"Voronoi binning attempt {attempt + 1} failed: {str(e)}"
                )

                if attempt < attempts - 1:
                    # Ensure target_snr is numeric before division
                    target_snr = float(target_snr) / 2.0
                    ExGalutil.prettyOutput_Done(
                        f"Retrying with target_snr = {target_snr}"
                    )
                    logging.warning(f"Retrying with target_snr = {target_snr}")
                    logging.info(f"Retrying with target_snr = {target_snr}")
                else:
                    ExGalutil.prettyOutput_Done("Final attempt failed")
                    logging.warning("Final attempt failed")

        if not success:
            ExGalutil.prettyOutput_Warning("Defining the Voronoi bins")
            print(
                "             The Voronoi-binning routine of Cappellari & Copin (2003) failed to generate voronoi bins"
            )
            print("             Analysis will continue without Voronoi-binning!")
            logging.warning(
                "The Voronoi-binning routine of Cappellari & Copin (2003) failed to generate voronoi bins"
            )
            logging.info(
                f"Analysis will continue without Voronoi-binning! {len(idx_inside)} spaxels will be treated as Voronoi-bins."
            )

            binNum, xNode, yNode, sn, nPixels = noBinning(x, y, snr, idx_inside)

    else:
        ExGalutil.prettyOutput_Done("No Voronoi-bins are requested/generated.")
        logging.info("No Voronoi-bins are requested/generated.")
        binNum, xNode, yNode, sn, nPixels = noBinning(x, y, snr, idx_inside)

    # MODIFIED: Track pixels outside valid region with negative binNum
    binNum_outside = find_nearest_voronoibin(x, y, idx_outside, xNode, yNode)
    ubins = np.unique(binNum)
    nbins = len(ubins)

    binNum_long = np.full(len(x), np.nan)
    binNum_long[idx_inside] = binNum
    binNum_long[idx_outside] = -1 * binNum_outside  # MODIFIED

    # Check if aps_id is a list of lists (nested structure)
    # in that case we first flatten the outputs and then save the table
    if len(aps_id) == 0 or not isinstance(aps_id[0], list):

        ref_table_path = save_table(
            rootname,
            outdir,
            aps_id,
            targid,
            cname,
            x,
            y,
            z,
            zerr,
            healpix,
            x_0,
            y_0,
            signal,
            snr,
            binNum_long,
            ubins,
            xNode,
            yNode,
            sn,
            nPixels,
            pixelsize,
            configs,
        )

    else:
        # Flattened outputs
        flat_aps_id = []
        flat_binNum = []
        flat_targid = []
        flat_cname = []
        flat_x = []
        flat_y = []
        flat_z = []
        flat_zerr = []
        flat_healpix = []
        flat_x0 = []
        flat_y0 = []
        flat_signal = []
        flat_snr = []

        # Iterate over all bins
        for i, aps_list in enumerate(aps_id):
            if len(aps_list) == 0:
                continue  # skip empty bins

            for aid in aps_list:
                flat_aps_id.append(aid)
                flat_binNum.append(binNum_long[i])
                flat_targid.append(targid[i])
                flat_cname.append(cname[i])
                flat_x.append(x[i])
                flat_y.append(y[i])
                flat_z.append(z[i])
                flat_zerr.append(zerr[i])
                flat_healpix.append(healpix[i])
                flat_x0.append(x_0[i])
                flat_y0.append(y_0[i])
                flat_signal.append(signal[i])
                flat_snr.append(snr[i])

        # Convert to NumPy arrays
        flat_aps_id = np.array(flat_aps_id)
        flat_binNum = np.array(flat_binNum)
        flat_targid = np.array(flat_targid)
        flat_cname = np.array(flat_cname)
        flat_x = np.array(flat_x)
        flat_y = np.array(flat_y)
        flat_z = np.array(flat_z)
        flat_zerr = np.array(flat_zerr)
        flat_healpix = np.array(flat_healpix)
        flat_x0 = np.array(flat_x0)
        flat_y0 = np.array(flat_y0)
        flat_signal = np.array(flat_signal)
        flat_snr = np.array(flat_snr)

        # update nPixels based on the original aps_ids instead of spatially binned ones
        nPixels_updated = np.array(
            [np.sum(flat_binNum == bin_val) for bin_val in ubins], dtype=int
        )

        ref_table_path = save_table(
            rootname,
            outdir,
            flat_aps_id,
            flat_targid,
            flat_cname,
            flat_x,
            flat_y,
            flat_z,
            flat_zerr,
            flat_healpix,
            flat_x0,
            flat_y0,
            flat_signal,
            flat_snr,
            flat_binNum,
            ubins,
            xNode,
            yNode,
            sn,
            nPixels_updated,
            pixelsize,
            configs,
        )

    return binNum


def noBinning(x, y, snr, idx_inside):
    """
    In case no Voronoi-binning is required/possible, treat spaxels in the input
    data as Voronoi bins, in order to continue the analysis.
    """
    binNum = np.arange(0, len(idx_inside))
    xNode = x[idx_inside]
    yNode = y[idx_inside]
    sn = snr[idx_inside]
    nPixels = np.ones(len(idx_inside))

    return (binNum, xNode, yNode, sn, nPixels)


def find_nearest_voronoibin(x, y, idx_outside, xNode, yNode):
    """
    This function determines the nearest Voronoi-bin for all spaxels which do
    not satisfy the minimum SNR threshold.
    """
    x = x[idx_outside]
    y = y[idx_outside]
    pix_coords = np.concatenate(
        (x.reshape((len(x), 1)), y.reshape((len(y), 1))), axis=1
    )
    bin_coords = np.concatenate(
        (xNode.reshape((len(xNode), 1)), yNode.reshape((len(yNode), 1))), axis=1
    )

    dists = dist.cdist(pix_coords, bin_coords, "euclidean")
    closest = np.argmin(dists, axis=1)

    return closest


def save_table(
    rootname,
    outdir,
    aps_id,
    targid,
    cname,
    x,
    y,
    z,
    zerr,
    healpix,
    x_0,
    y_0,
    signal,
    snr,
    binNum_new,
    ubins,
    xNode,
    yNode,
    sn,
    nPixels,
    pixelsize,
    configs,
):
    """
    Save all relevant information about the Voronoi binning to disk. In
    particular, this allows to later match spaxels and their corresponding bins.
    """
    outfits_table = outdir + rootname + "_table.fits"
    ExGalutil.prettyOutput_Running("Writing: " + rootname + "_table.fits")

    # Expand data to spaxel level
    xNode_new = np.zeros(len(x))
    yNode_new = np.zeros(len(x))
    sn_new = np.zeros(len(x))
    nPixels_new = np.zeros(len(x))
    for i in range(len(ubins)):
        idx = np.where(ubins[i] == np.abs(binNum_new))[0]
        xNode_new[idx] = xNode[i]
        yNode_new[idx] = yNode[i]
        sn_new[idx] = sn[i]
        nPixels_new[idx] = nPixels[i]

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Table HDU with output data
    cols = []
    cols.append(fits.Column(name="ID", format="J", array=np.arange(len(x))))
    cols.append(fits.Column(name="BIN_ID", format="J", array=binNum_new))
    cols.append(fits.Column(name="APS_ID", format="J", array=aps_id))
    cols.append(fits.Column(name="TARGID", format="40A", array=targid))
    cols.append(fits.Column(name="CNAME", format="40A", array=cname))
    cols.append(fits.Column(name="X", format="D", array=x))
    cols.append(fits.Column(name="Y", format="D", array=y))
    cols.append(fits.Column(name="Z", format="D", array=z))
    cols.append(fits.Column(name="ZERR", format="D", array=zerr))
    cols.append(fits.Column(name="HEALPIX_ID", format="J", array=healpix))
    cols.append(fits.Column(name="X_0", format="D", array=x_0))
    cols.append(fits.Column(name="Y_0", format="D", array=y_0))
    cols.append(fits.Column(name="FLUX", format="D", array=signal))
    cols.append(fits.Column(name="SNR", format="D", array=snr))
    cols.append(fits.Column(name="XBIN", format="D", array=xNode_new))
    cols.append(fits.Column(name="YBIN", format="D", array=yNode_new))
    cols.append(fits.Column(name="SNRBIN", format="D", array=sn_new))
    cols.append(fits.Column(name="NSPAX", format="J", array=nPixels_new))

    tbhdu = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    tbhdu.name = "REF_TABLE"
    tbhdu.header["PIXSIZE"] = pixelsize
    tbhdu.header["CONFIG_F"] = (configs["CONFIG_FILE"], "PyAPS-ExGal Configs. Filename")

    tbhdu.header["APSVERS"] = (APSVERS, "APS version")
    tbhdu.header["APSOAT_V"] = (
        aps_constants.__aps_OAT_version__,
        "PyAPS-ExGal Opt. Adaptive Tessellation version",
    )
    # keep the basename of input file (infiles) to save as provinces later
    for n_province, province in enumerate(configs["infiles"]):
        tbhdu.header["APSREF_%d" % (n_province)] = (
            os.path.basename(province),
            "L1 reference file",
        )

    ## Update header, with information coming from the config file
    ## This is mostly useful for IFU modes, where we apply Voronoi binning in general

    if "VORONOI" in configs.keys():
        tbhdu.header["TESSEL"] = (
            configs["VORONOI"],
            "0: No adaptive tessellations. 1: VORONOI",
        )
        tbhdu.header["MINSNR"] = (
            configs["MIN_SNR"],
            "Min S/N per spaxel to be accepted for Tessellation",
        )
        tbhdu.header["TARGSNR"] = (
            configs["TARGET_SNR"],
            "Target S/N ratio for the Opt. Adaptive Tessellation",
        )
        tbhdu.header["CO_VOR"] = (
            configs["COVAR_VOR"],
            "Correct for spatial correlations of the noise in the Voronoi",
        )

    # Create HDU list and write to file
    HDUList = fits.HDUList([priHDU, tbhdu])
    HDUList.writeto(outfits_table, overwrite=True)
    fits.setval(outfits_table, "PIXSIZE", value=pixelsize)

    ExGalutil.prettyOutput_Done("Writing: " + rootname + "_table.fits")
    logging.info("Wrote Voronoi table: " + outfits_table)
    return outfits_table


def prepare_spec_file(binNum, spec, espec, rootname, outdir, wave, flag, verbose=True):
    """
    FOR MOS mode ONLY
    Prepare and save an SPEC structure file into the disk
    """
    # Apply Voronoi bins
    if verbose:
        ExGalutil.prettyOutput_Running("Preparing Spec file for " + flag + "-data")

    ubins = np.unique(binNum)
    nbins = len(ubins)
    npix = spec.shape[0]
    bin_data = np.zeros([npix, nbins])
    bin_error = np.zeros([npix, nbins])
    bin_flux = np.zeros(nbins)
    for i in range(nbins):
        k = np.where(binNum == ubins[i])[0]

        bin_data[:, i] = np.ravel(spec[:, k])
        bin_error[:, i] = np.ravel(espec[:, k])
        # one scalar per bin: mean over the pixels (and spectra) of the bin. spec[:, k] has
        # shape (npix, len(k)), so a per-column mean with axis=0 is an array, and recent NumPy
        # 2.x refuses to store a size-1 array in a scalar element (deprecated since 1.25).
        bin_flux[i] = np.mean(spec[:, k])

    if verbose:
        ExGalutil.prettyOutput_Done(
            "Preparing Spec file for " + flag + "-data", progressbar=True
        )

    logging.info("Preparing Spec file for " + flag + "-data")

    # Save SEMI Voronoi binned spectra
    save_vorspectra(rootname, outdir, binNum, bin_data, bin_error, wave, flag, "MOS")
    return None


def apply_voronoi_bins(
    binNum, spec, espec, rootname, outdir, wave, flag, mode, verbose=True
):
    """
    The constructed Voronoi-binning is applied to the underlying spectra. The
    resulting Voronoi-binned spectra are saved to disk.
    """
    # Apply Voronoi bins
    if verbose:
        ExGalutil.prettyOutput_Running("Applying the Voronoi bins to " + flag + "-data")

    bin_data, bin_error, bin_flux = voronoi_binning(binNum, spec, espec)

    if verbose:
        ExGalutil.prettyOutput_Done(
            "Applying the Voronoi bins to " + flag + "-data", progressbar=True
        )

    logging.info("Applied Voronoi bins to " + flag + "-data")

    # Save Voronoi binned spectra
    save_vorspectra(rootname, outdir, binNum, bin_data, bin_error, wave, flag, mode)
    return None


def voronoi_binning(binNum, spec, error):
    """Spectra belonging to the same Voronoi-bin are added."""
    ubins = np.unique(binNum)
    nbins = len(ubins)
    npix = spec.shape[0]
    bin_data = np.zeros([npix, nbins])
    bin_error = np.zeros([npix, nbins])
    bin_flux = np.zeros(nbins)

    for i in range(nbins):
        k = np.where(binNum == ubins[i])[0]
        valbin = len(k)
        if valbin == 1:
            av_spec = spec[:, k]
            av_err_spec = error[:, k]
        else:
            av_spec = np.nansum(spec[:, k], axis=1)
            av_err_spec = np.sqrt(np.sum(error[:, k], axis=1))

        bin_data[:, i] = np.ravel(av_spec)
        bin_error[:, i] = np.ravel(av_err_spec)
        bin_flux[i] = np.mean(av_spec)  # scalar (av_spec is (npix, 1) or (npix,))
        ExGalutil.printProgress(i + 1, nbins, barLength=50)

    return (bin_data, bin_error, bin_flux)


def save_vorspectra(rootname, outdir, binNum, log_spec, log_error, logLam, flag, mode):
    """Voronoi-binned spectra and error spectra are saved to disk."""
    if flag == "log":
        outfits_spectra = outdir + rootname + "_BINSpectra.fits"
        ExGalutil.prettyOutput_Running("Writing: " + rootname + "_BINSpectra.fits")
    elif flag == "lin":
        outfits_spectra = outdir + rootname + "_BINSpectra_linear.fits"
        ExGalutil.prettyOutput_Running(
            "Writing: " + rootname + "_BINSpectra_linear.fits"
        )

    npix = log_spec.shape[0]
    nbins = log_spec.shape[1]
    ubins = np.arange(0, nbins)
    # Create primary HDU
    priHDU = fits.PrimaryHDU()

    # Table HDU for spectra
    cols = []

    cols.append(fits.Column(name="BIN_ID", format="J", array=ubins))

    if mode.upper().replace(" ", "") == "MOS":
        cols.append(fits.Column(name="LOGLAM", format=str(npix) + "D", array=logLam.T))

    elif mode.upper().replace(" ", "") == "IFU":
        ## Create a copy of loglam array to be place in each row of the emippxf output
        loglam_tile = np.tile(logLam, (nbins, 1))
        cols.append(
            fits.Column(name="LOGLAM", format=str(npix) + "D", array=loglam_tile)
        )
    else:
        sys.exit(
            "Working mode for VORONOI BINNING is not within the accepted list: [MOS, IFU]"
        )

    cols.append(fits.Column(name="SPEC", format=str(npix) + "D", array=log_spec.T))
    cols.append(fits.Column(name="ESPEC", format=str(npix) + "D", array=log_error.T))

    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = "BIN_SPECTRA"

    HDUList = fits.HDUList([priHDU, dataHDU])
    HDUList.writeto(outfits_spectra, overwrite=True)

    # Set header values
    # fits.setval(outfits_spectra,'VELSCALE',value=velscale)
    # fits.setval(outfits_spectra,'CRPIX1',  value=1.0)
    # fits.setval(outfits_spectra,'CRVAL1',  value=logLam[0])
    # fits.setval(outfits_spectra,'CDELT1',  value=logLam[1]-logLam[0])

    if flag == "log":
        ExGalutil.prettyOutput_Done("Writing: " + rootname + "_BINSpectra.fits")
    elif flag == "lin":
        ExGalutil.prettyOutput_Done("Writing: " + rootname + "_BINSpectra_linear.fits")
    logging.info("Wrote: " + outfits_spectra)


# Add this function to ExGalPrepare.py


def save_binned_spectra_novor(
    spec, error, wave, headname, outpath, flag="log", binNum=None
):
    """
    Simple function to save binned spectra directly without Voronoi mapping.
    Used when spatial binning is applied with VORONOI=0.

    Parameters
    ----------
    spec : array
        Binned spectra, shape (n_wave, n_bins)
    error : array
        Binned errors, shape (n_wave, n_bins)
    wave : array
        Wavelength array, shape (n_wave,) or (n_wave, n_bins)
    headname : str
        Base name for output file
    outpath : str
        Output directory
    flag : str
        'log' for log-rebinned spectra (ExGal), 'lin' for linear spectra (Gal)
    binNum : array, optional
        Bin IDs. If None, uses sequential numbering

    Returns
    -------
    outfile : str
        Path to created FITS file
    """

    import logging
    import os

    import numpy as np
    from astropy.io import fits

    from PyAPS import ExGalutil

    # Determine output filename based on flag
    if flag == "log":
        outfile = os.path.join(outpath, f"{headname}_BINSpectra.fits")
        column_name = "LOGLAM"
        file_desc = f"{headname}_BINSpectra.fits"
    else:
        outfile = os.path.join(outpath, f"{headname}_BINSpectra_linear.fits")
        column_name = "LAM"
        file_desc = f"{headname}_BINSpectra_linear.fits"

    ExGalutil.prettyOutput_Running(f"Writing: {file_desc}")

    # Get dimensions
    n_wave = spec.shape[0]
    n_bins = spec.shape[1]

    # Create bin IDs if not provided
    if binNum is None or len(binNum) != n_bins:
        binNum = np.arange(n_bins)

    # Ensure wave is 1D
    if wave.ndim > 1:
        wave = wave[:, 0]

    # Create primary HDU
    priHDU = fits.PrimaryHDU()
    priHDU.header["COMMENT"] = (
        f'{"Log-rebinned" if flag == "log" else "Linear"} binned spectra'
    )
    priHDU.header["NBINS"] = n_bins
    priHDU.header["NWAVE"] = n_wave
    priHDU.header["WAVEMIN"] = wave[0]
    priHDU.header["WAVEMAX"] = wave[-1]
    priHDU.header["BINTYPE"] = "SPATIAL_DIRECT"  # Indicates direct spatial binning

    # Create table HDU with spectra
    cols = []

    # Bin IDs
    cols.append(fits.Column(name="BIN_ID", format="J", array=binNum))

    # Wavelength array (same for all bins)
    wave_tile = np.tile(wave, (n_bins, 1))
    cols.append(fits.Column(name=column_name, format=f"{n_wave}D", array=wave_tile))

    # Spectra and errors
    cols.append(fits.Column(name="SPEC", format=f"{n_wave}D", array=spec.T))
    cols.append(fits.Column(name="ESPEC", format=f"{n_wave}D", array=error.T))

    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = "BIN_SPECTRA"

    # Create HDU list and write
    HDUList = fits.HDUList([priHDU, dataHDU])
    HDUList.writeto(outfile, overwrite=True)

    ExGalutil.prettyOutput_Done(f"Writing: {file_desc}")
    logging.info(f"Wrote: {outfile}")

    print(f"Saved binned spectra with {n_bins} bins")
    print(f"  File: {outfile}")
    print(f"  Wavelength range: {wave[0]:.1f} - {wave[-1]:.1f}")
    print(f"  Dimensions: {n_wave} wavelength points × {n_bins} bins")

    return outfile
