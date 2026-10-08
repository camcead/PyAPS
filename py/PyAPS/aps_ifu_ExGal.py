"""
aps_ifu_exgal.py
================
ExGal IFU pipeline functions.

Public API
----------
ifu_ExGal_prepare(...)   ->  dict | None
    Data ingestion, cube assembly, binning, log-rebinning, LSF.

ifu_ExGal(...)
    Loops over a patch table and runs ifu_ExGal_prepare +
    PPXF / EMIPPXF / LS + L2merge for every GALAXY / QSO target.

    Patch table resolution order:
      1. patch_array  — single injected row (debug / test mode).
                        ctarg_excluded is ignored.
                        class_patch still runs if Z is NaN.
      2. patch_file   — path to file on disk, loaded and split.
      3. neither      → AssertionError.

Helper
------
make_patch_array(...)    ->  dict
    Convenience constructor for a single patch_array row.

test_patch_table(...)    ->  bool   (re-exported from _patch_utils)
"""

import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
from astropy.coordinates import ICRS, SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy_healpix import HEALPix
from scipy.interpolate import interp1d

from PyAPS import aps_constants

# shared utilities
from PyAPS.aps_ifu_utils import (
    CLASS_PRIORITY,
    adaptive_min_snr,
    decode_class,
    ensure_table,
    first_valid_class,
    first_valid_z,
    load_and_split_patch_file,
    test_patch_table,
)
from PyAPS.aps_L2merge import ifuExGalL2merge
from PyAPS.aps_utils import APSOB, apply_redshift_to_fwhm_corrected, gen_targlist
from PyAPS.aps_ifu_spaxel_contrib import bucket_lsf_for_voronoi_bins
from PyAPS import IFUExGalEMIPPXF, IFUExGalLS, IFUExGalPPXF
from PyAPS import IFUExGalPrepare as IFUExGalPrepare
from PyAPS import ExGalPrepare as ExGalPrepare
from PyAPS import ExGalutil
from PyAPS.apsPlot.exgal_prepare import plot_all

# re-export so callers can do: from PyAPS.aps_ifu_exgal import test_patch_table
__all__ = [
    "ifu_ExGal_prepare", "ifu_ExGal",
    "make_patch_array", "test_patch_table",
    "find_nearest_snr","ppxf_limits",
]

# --------------------------------------------------------------------------- #
Clight      = 299792.458
large_error = aps_constants.large_error
reference_SNR = [20.0, 30.0, 40.0]
# --------------------------------------------------------------------------- #

def ppxf_limits(
    z,
    obs_lmin=3800.,
    obs_lmax=9280.,
    lib_lmin=1000.,
    lib_lmax=10000.,
    lowz_lmin=3800.,
    lowz_lmax=6000.,
    edge_buffer=50.,
    min_window=500.,
    cah_blue_clamp=True,
    cah_blue_edge=3800.,
    cah_clamp_zmax=1.0,
):
    """
    Compute the rest-frame wavelength window [LMIN_PPXF, LMAX_PPXF] for
    pPXF stellar kinematics at a given redshift, given the spectrograph
    coverage and the stellar template library limits.

    The function has two regimes separated by a natural threshold redshift
    z_thresh, which is derived from the instrument and preferred window
    rather than hardcoded:

        z_thresh = obs_lmax / lowz_lmax - 1
                 = 9280 / 6000 - 1
                 = 0.547  (for default WEAVE + 3800-6000 Å preference)

    This is the redshift at which your preferred red limit (lowz_lmax)
    shifts beyond the spectrograph red cutoff (obs_lmax) and can no longer
    be observed.  It is the natural point at which the pipeline must adapt.

    LOW-Z REGIME  (z < z_thresh)
    -----------------------------
    The full preferred optical window [lowz_lmin, lowz_lmax] is always
    observable.  It is used exactly as specified, with no clipping.
    This preserves the well-characterised set of stellar absorption
    features (Ca H&K, G-band, Hβ, Mg b, Fe 5270/5335) that anchor
    stellar kinematics at low redshift.

    Example at z=0.3:
        rest coverage = 2923–7138 Å  →  window forced to 3800–6000 Å

    HIGH-Z REGIME  (z >= z_thresh)
    --------------------------------
    The red end of the preferred window has shifted beyond obs_lmax and
    can no longer be observed.  The window is set to the full intersection
    of spectrograph coverage and template library, clipped by edge_buffer
    on both sides to avoid arm edges and template extrapolation.

    An optional blue clamp (cah_blue_clamp=True) prevents the window
    opening up into the UV below Ca H&K (3800 Å rest) when the redshift
    is still low enough that Ca H&K is observable.  This avoids including
    UV continuum (2000–3800 Å rest) where template reliability is lower
    and kinematic information is sparse, for targets where the optical
    is still partially available.  The clamp is disabled above
    cah_clamp_zmax (default z=1.0) where Ca H&K has itself redshifted
    beyond the blue spectrograph limit.

    Example at z=0.6  (with Ca H&K clamp):
        rest coverage = 2375–5800 Å
        raw window    = 2425–5750 Å
        after clamp   = 3800–5750 Å   ← preferred blue edge, trimmed red end

    Example at z=0.6  (without Ca H&K clamp):
        window        = 2425–5750 Å   ← full UV+optical coverage

    Example at z=1.0:
        rest coverage = 1900–4640 Å
        Ca H&K clamp disabled (z > cah_clamp_zmax=1.0 not yet, so still on)
        Ca H&K (3933 Å) is still in coverage → clamp applies
        window        = 3800–4590 Å   ← narrow but Ca H&K + G-band + Hβ

    Example at z=1.5:
        rest coverage = 1520–3712 Å
        Ca H&K (3933 Å) has shifted to 9832 Å observed → beyond obs_lmax
        clamp no longer possible → window = 1570–3662 Å

    Example at z=2.0:
        rest coverage = 1267–3093 Å
        Lyα forest floor kicks in (z > 2) → lmin raised to 1266 Å
        window        = 1266–3043 Å

    LYMAN ALPHA FOREST
    -------------------
    For z > 2, IGM absorption increasingly neutralises flux blueward of
    Lyα (1216 Å rest).  The blue limit is raised to 1216 + edge_buffer
    to avoid the forest, regardless of library coverage.

    MINIMUM WINDOW CHECK
    ---------------------
    If the resulting window is narrower than min_window (default 500 Å
    rest-frame), the function returns (None, None).  This signals to the
    pipeline that stellar kinematics cannot be reliably constrained at
    this redshift with this instrument+library combination, and the PPXF
    step should be skipped or flagged.

    Approximate redshift ceiling for default parameters:
        z ~ 4.5–5.0  (window narrows below 500 Å beyond this)

    Parameters
    ----------
    z : float
        Spectroscopic redshift of the target.

    obs_lmin : float, optional
        Blue cutoff of the spectrograph in observed Angstroms.
        Default: 3800.0  (WEAVE blue arm)

    obs_lmax : float, optional
        Red cutoff of the spectrograph in observed Angstroms.
        Default: 9280.0  (WEAVE red arm)

    lib_lmin : float, optional
        Blue limit of the stellar template library in rest-frame Angstroms.
        Default: 1000.0

    lib_lmax : float, optional
        Red limit of the stellar template library in rest-frame Angstroms.
        Default: 10000.0

    lowz_lmin : float, optional
        Preferred blue limit of the pPXF fitting window at low redshift,
        rest-frame Angstroms.  Used exactly when z < z_thresh.
        Default: 3800.0  (blue edge of optical stellar features)

    lowz_lmax : float, optional
        Preferred red limit of the pPXF fitting window at low redshift,
        rest-frame Angstroms.  Used exactly when z < z_thresh.
        Also defines z_thresh = obs_lmax / lowz_lmax - 1.
        Default: 6000.0

    edge_buffer : float, optional
        Margin in Angstroms to trim from each end of the spectrograph
        coverage before computing the window.  Avoids arm edge artefacts
        and template extrapolation regions.
        Default: 50.0

    min_window : float, optional
        Minimum acceptable window width in rest-frame Angstroms.
        Returns (None, None) if the window is narrower than this.
        Default: 500.0

    cah_blue_clamp : bool, optional
        If True, clamp lmin to cah_blue_edge when z < cah_clamp_zmax
        and Ca H&K is still within the observable rest-frame range.
        This prevents the window opening into the UV (below Ca H&K) for
        intermediate-z targets where optical features are still available.
        Default: True

    cah_blue_edge : float, optional
        Rest-frame blue clamp value when cah_blue_clamp is active.
        Default: 3800.0 Å  (just blueward of Ca H&K at 3933/3968 Å)

    cah_clamp_zmax : float, optional
        Maximum redshift at which the Ca H&K blue clamp is applied.
        Above this redshift Ca H&K itself shifts beyond the spectrograph
        red limit and the clamp is disabled.
        Default: 1.0

    Returns
    -------
    lmin : float or None
        Rest-frame blue limit of the pPXF window in Angstroms.
        None if the window is too narrow to constrain kinematics.

    lmax : float or None
        Rest-frame red limit of the pPXF window in Angstroms.
        None if the window is too narrow to constrain kinematics.

    Examples
    --------
    >>> # Default WEAVE setup
    >>> ppxf_limits(0.0)
    (3800.0, 6000.0)

    >>> ppxf_limits(0.3)
    (3800.0, 6000.0)

    >>> ppxf_limits(0.6)    # above threshold, Ca H&K clamp active
    (3800.0, 5750.0)

    >>> ppxf_limits(0.6, cah_blue_clamp=False)
    (2425.0, 5750.0)

    >>> ppxf_limits(1.0)    # Ca H&K clamp at its limit
    (3800.0, 4590.0)

    >>> ppxf_limits(1.5)    # clamp disabled, Ca H&K gone
    (1570.0, 3662.0)

    >>> ppxf_limits(2.5)    # Lya floor active
    (1266.0, 2601.0)

    >>> ppxf_limits(5.0)    # window too narrow
    (None, None)

    Notes
    -----
    The function should be called in ifu_ExGal_prepare immediately after
    the redshift is known, replacing the hard HZ_LMIN_PPXF / HZ_LMAX_PPXF
    switch:

        lmin_ppxf, lmax_ppxf = ppxf_limits(
            z_input[0],
            lib_lmin=configs.get("PPXF_LIB_LMIN", 1000.),
            lib_lmax=configs.get("PPXF_LIB_LMAX", 10000.),
            lowz_lmin=configs.get("PPXF_LOWZ_LMIN", 3800.),
            lowz_lmax=configs.get("PPXF_LOWZ_LMAX", 6000.),
        )
        if lmin_ppxf is None:
            print("WARNING: z=%.3f — window too narrow, PPXF skipped")
            configs["SKIP_PPXF"] = True
        else:
            configs["LMIN_PPXF"] = lmin_ppxf
            configs["LMAX_PPXF"] = lmax_ppxf

    Recommended JSON config keys (add to LIFULR11.json etc.):
        "PPXF_LOWZ_LMIN"  : 3800,
        "PPXF_LOWZ_LMAX"  : 6000,
        "PPXF_LIB_LMIN"   : 1000,
        "PPXF_LIB_LMAX"   : 10000
    """

    # ------------------------------------------------------------------
    # Natural threshold: redshift at which lowz_lmax shifts beyond obs_lmax
    # ------------------------------------------------------------------
    z_thresh = (obs_lmax / lowz_lmax) - 1.0

    # Rest-frame coverage available from the spectrograph at this redshift
    rest_blue = obs_lmin / (1.0 + z)
    rest_red  = obs_lmax / (1.0 + z)

    # ------------------------------------------------------------------
    # Low-z regime: use preferred optical window exactly
    # ------------------------------------------------------------------
    if z < z_thresh:
        lmin = float(lowz_lmin)
        lmax = float(lowz_lmax)

    # ------------------------------------------------------------------
    # High-z regime: use full intersection of coverage and library
    # ------------------------------------------------------------------
    else:
        lmin = max(rest_blue + edge_buffer, float(lib_lmin))
        lmax = min(rest_red  - edge_buffer, float(lib_lmax))

        # Ca H&K blue clamp: keep lmin at optical edge when Ca H&K is
        # still observable, avoiding low-information UV continuum
        if cah_blue_clamp and z <= cah_clamp_zmax:
            # Check Ca H&K (3933 Å rest) is still within obs coverage
            cah_obs = 3933.0 * (1.0 + z)
            if cah_obs <= obs_lmax:
                lmin = max(lmin, float(cah_blue_edge))

        # Lyman alpha forest floor for z > 2
        if z > 2.0:
            lmin = max(lmin, 1216.0 + edge_buffer)

    # ------------------------------------------------------------------
    # Minimum window check
    # ------------------------------------------------------------------
    if (lmax - lmin) < min_window:
        return None, None

    return float(lmin), float(lmax)

# ------------------------------------------------------------------

def find_nearest_snr(reference_values, value):
    return min(reference_values, key=lambda x: abs(x - value))

# ------------------------------------------------------------------


# =========================================================================== #
#  Convenience constructor                                                     #
# =========================================================================== #

def make_patch_array(
    ra: float,
    dec: float,
    a_arcsec: float,
    b_arcsec: float,
    z: float,
    zerr: float,
    class_str: str,
    zwarn: int = 0,
    angle: float = 0.0,
    row_id: int = 1,
    row_type: str = "T",
) -> dict:
    """
    Build a single-row patch_array dict ready to pass to ifu_ExGal
    or ifu_Gal.

    Parameters
    ----------
    ra, dec : float
        Target centre in degrees (ICRS).
    a_arcsec, b_arcsec : float
        FULL major / minor axis lengths (diameters) in **arcsec**, the
        same meaning as the ``A_world`` / ``B_world`` columns of a patch
        table and the ``width`` / ``height`` of the extraction ellipse.
        They are not semi-axes: ``a_arcsec=10, b_arcsec=6`` extracts up
        to 5 arcsec from the centre along the major axis and 3 arcsec
        along the minor axis (see ``aps_utils.aperture_sky_region``).
    z, zerr : float
        Redshift and uncertainty.
    class_str : str
        Classification string e.g. ``'GALAXY'``, ``'STAR'``.
    zwarn : int
        Redshift warning flag (default 0).
    angle : float
        Position angle in degrees.
    row_id : int
        Row id (default 1).
    row_type : str
        Row type: ``'T'`` target (default), ``'C'`` central, ``'M'`` mask.

    Returns
    -------
    dict  — pass directly as ``patch_array=`` argument.
    """
    return {
        "id":       row_id,
        "RA_icrs":  ra,
        "DEC_icrs": dec,
        "A_world":  a_arcsec / 3600.0,
        "B_world":  b_arcsec / 3600.0,
        "angle":    angle,
        "flag":     0,
        "type":     row_type,
        "Z":        [float(z)],
        "ZERR":     [float(zerr)],
        "ZWARN":    [int(zwarn)],
        "CLASS":    [str(class_str).strip()],
    }


# =========================================================================== #
#  STAGE 1 — data preparation (unchanged from previous version)               #
# =========================================================================== #

def ifu_ExGal_prepare(
    infiles, headname, IFU_params, outpath, IFU_config_dir,
    wlranges=None, aps_ids=None, targsrvy=None, targclass=None,
    mask_aps_ids=None, area=None, mask_areas=None,
    sens_corr=True, mask_gaps=True, safe_mask_gaps=True,
    vacuum=False, tellurics=False, fill_gap=False,
    arms_ratio=None, join_arms=False, z_input=None,
    catdir=None, caldir=None,
    spaxel_weighted_lsf=None,
    extinction_corr=None, extinction_ebv_fixed=None,
) -> dict:
    """
    Data preparation stage for the ExGal IFU pipeline.

    Stages
    ------
    1. Load IFU_params JSON config.
    2. Build APSOB and assemble the spaxel cube.
    3. Spatial binning + SB flux filter  (if SPBIN_SIZE_EXGAL > 0).
    4. Voronoi tessellation.
    5. Log-rebinning and write BINSpectra.fits.
    6. Diagnostic plots via apsPlot.exgal_prepare.plot_all.
    7. Build LSF_Data and LSF_Templates interpolators.

    Parameters
    ----------
    infiles : list of str
        L1 FITS files (blue + red arms).
    headname : str
        Root name for all output files.
    IFU_params : str
        Path to the IFU parameter JSON file (e.g. LIFULR11.json).
    outpath : str
        Output directory (trailing separator added internally).
    IFU_config_dir : str
        Directory containing LSF-Config files and related configs.
    wlranges : list of [float, float] or None
        Wavelength ranges per arm.
    aps_ids : list of int or None
        Restrict to these APS IDs.
    targsrvy, targclass : list of str or None
        Survey / class filters passed to gen_targlist.
    mask_aps_ids : list of int or None
        APS IDs to mask out.
    area : [RA, Dec, A_arcsec, B_arcsec, angle] or None
        Elliptical aperture for this patch.
    mask_areas : list of areas or None
        Apertures to mask (type=M rows).
    sens_corr, mask_gaps, safe_mask_gaps : bool
        Spectral processing flags.
    vacuum : bool
        Convert wavelengths to vacuum.
    tellurics : bool
        Apply telluric correction.
    fill_gap : bool
        Fill inter-arm gap.
    arms_ratio : [float, float] or None
        Flux-scale ratio between arms.
    join_arms : bool
        Stitch arms into a single spectrum.
    z_input : [z, z_err]
        Redshift and uncertainty for this target.
    catdir, caldir : str or None
        Catalogue and calibration directories.
    spaxel_weighted_lsf : bool or None, optional
        Per-bin resolution. `None` (default): resolved from
        `configs['SPAXEL_WEIGHTED_LSF']` in the IFU_params JSON, itself
        **default on as of v1.9** if absent -- same argument > config >
        default precedence `aps_ifu_Gal.py`'s own `VORONOI_GAL`/
        `SPBIN_SIZE_GAL` already use. Every bundled LIFU*/MIFU*
        IFU_params JSON also sets this key explicitly (`1`); pass
        `spaxel_weighted_lsf=False` (or set `"SPAXEL_WEIGHTED_LSF": 0` in
        a config) to fall back to the pre-v1.9 flat-global-curve
        behaviour. When (however resolved) on: resolves real per-spaxel
        LSF/FWHM (`aps_utils.APSOB`'s own `spaxel_weighted_lsf` parameter
        — see `aps_ifu_spaxel_contrib.py`), flux-weight-averages it up to
        each Voronoi/PowerBin bin (composed correctly through the optional
        spatial-pre-binning stage too, via `ExGalPrepare.spatial_bin_
        with_provenance`'s own provenance), then quantizes into a handful
        of resolution buckets (`configs['LSF_N_BUCKETS']`, default 6) so
        PPXF/EMIPPXF/LineStrength template preparation stays cheap. Off:
        every bin keeps using the single global `LSF_Data` curve, exactly
        as before this parameter existed.

    Returns
    -------
    dict with keys:
        cube               — spaxel data dict
        configs            — merged config dict
        LSF_Data           — per-spaxel LSF array (redshift-corrected)
        LSF_Templates      — interpolator for template LSF
        figdir             — path to figure directory (str)
        bin_to_bucket      — array, length nbins (None if the resolved spaxel_weighted_lsf is off)
        LSF_Data_by_bucket — {bucket_id: interpolate_function} (None if the resolved spaxel_weighted_lsf is off)

    Returns None if no valid APS IDs are found for this patch.
    """
    assert z_input is not None, "z_input=[z, z_err] is required"
    assert len(z_input) == 2,   "z_input must be [z, z_err]"

    if not os.path.exists(IFU_params):
        sys.exit("No parameter file found at: %s" % IFU_params)

    configs = json.load(open(IFU_params))
    configs["CONFIG_FILE"] = os.path.basename(IFU_params)

    # argument > config > default -- same precedence aps_ifu_Gal.py's own
    # VORONOI_GAL/SPBIN_SIZE_GAL resolution already uses, so a patch's
    # IFU_params JSON can turn this on/off without touching any Python
    # call site, while an explicit spaxel_weighted_lsf=True/False
    # argument still wins if given. Default (when neither an explicit
    # argument nor a config key is present) is **on** as of v1.9 -- all
    # IFU pipeline code uses spaxel-weighted LSF unless a config or
    # caller explicitly opts out (every bundled LIFU*/MIFU* IFU_params
    # JSON also sets "SPAXEL_WEIGHTED_LSF": 1 explicitly, for
    # discoverability -- this fallback covers any config that doesn't).
    if spaxel_weighted_lsf is None:
        spaxel_weighted_lsf = bool(configs.get("SPAXEL_WEIGHTED_LSF", 1))

    # ------------------------------------------------------------------
    # Galactic extinction correction (SFD98+Fitzpatrick99, shared with
    # REDROCK -- see doc/aps_rr.md and ExGalPrepare.resolve_ebmv_extinction's
    # own docstring). Same argument > config > default precedence as
    # spaxel_weighted_lsf above.
    #
    # IFU-specific: unlike MOS (which uses APSOB's normal per-target sky-
    # position SFD lookup), an "auto" (EBmV="None") resolution here uses
    # ONE E(B-V) for the whole patch, evaluated at the cube's own WCS
    # reference point (CRVAL1/CRVAL2 -- the same field-centre coordinate
    # APSOB itself later exposes via .origin()). At IFU angular scales
    # the SFD map is effectively constant across one patch anyway, so
    # this is both simpler and, in practice, no different from a
    # per-spaxel lookup -- but it must be resolved from the raw FITS
    # header BEFORE building APSOB, since the correction is applied
    # while APSOB is being built.
    # ------------------------------------------------------------------
    extinction_kwargs = ExGalPrepare.resolve_ebmv_extinction(
        configs, extinction_corr=extinction_corr,
        extinction_ebv_fixed=extinction_ebv_fixed)
    ExGalPrepare.note_reddening_unused(configs)
    if extinction_kwargs['extinction_corr'] and \
            extinction_kwargs['extinction_ebv_fixed'] is None:
        from astropy.wcs import WCS as _WCS
        from PyAPS.aps_utils import get_sfd_ebv as _get_sfd_ebv
        _wcs_h1 = _WCS(fits.getheader(infiles[0], 1))
        _patch_ra, _patch_dec = _wcs_h1.wcs.crval[0], _wcs_h1.wcs.crval[1]
        extinction_kwargs['extinction_ebv_fixed'] = float(
            _get_sfd_ebv(_patch_ra, _patch_dec))
        print(f"  Extinction (auto, whole-patch): field centre "
              f"RA={_patch_ra:.5f} Dec={_patch_dec:.5f} -> "
              f"E(B-V)_SFD={extinction_kwargs['extinction_ebv_fixed']:.4f}")

    if not extinction_kwargs['extinction_corr']:
        configs['EBmV_APPLIED'] = 'OFF'
    else:
        configs['EBmV_APPLIED'] = extinction_kwargs['extinction_ebv_fixed']

    figdir = Path(outpath) / "figs_ExGal"
    figdir.mkdir(parents=True, exist_ok=True)
    figdir = str(figdir) + os.sep

    # ------------------------------------------------------------------
    # Build target list and APSOB
    # ------------------------------------------------------------------
    aps_ids_in_class, _, _, _, _ = gen_targlist(
        infiles[0], "IFU",
        aps_ids=aps_ids, targsrvy=targsrvy, targclass=targclass,
        mask_aps_ids=mask_aps_ids, area=area, mask_areas=mask_areas,
        la_out=False,
    )
    if len(aps_ids_in_class) == 0:
        print("WARNING: No valid APS IDs found for this ExGal patch. Skipping...")
        return None

    APSOBJ_inst = APSOB(
        infiles,
        targsrvy=targsrvy, targclass=targclass,
        aps_ids=aps_ids_in_class, mask_aps_ids=mask_aps_ids,
        area=area, mask_areas=mask_areas, wlranges=wlranges,
        sens_corr=sens_corr, mask_gaps=mask_gaps,
        safe_mask_gaps=safe_mask_gaps, vacuum=vacuum,
        tellurics=tellurics, fill_gap=fill_gap,
        arms_ratio=arms_ratio, join_arms=join_arms,
        catdir=catdir, caldir=caldir, configdir=IFU_config_dir,
        spaxel_weighted_lsf=spaxel_weighted_lsf,
        **extinction_kwargs,
    )

    targs             = APSOBJ_inst.data()
    targs_infiles     = APSOBJ_inst.infiles()
    targs_id          = APSOBJ_inst.id()
    targs_idfx        = APSOBJ_inst.idfx()
    targs_nbands      = APSOBJ_inst.nbands()
    targs_funits      = APSOBJ_inst.funits()
    targs_join_arms   = APSOBJ_inst.join_arms()
    targs_origin      = APSOBJ_inst.origin()
    targs_lsf         = APSOBJ_inst.get_fwhm(aps_id=None, fwhm_key="gfwhm")
    targs_setups      = APSOBJ_inst.setups()
    setups_original   = APSOBJ_inst.setups_original()
    wlranges_original = APSOBJ_inst.wlranges_original()

    if len(targs_lsf) > 1:
        sys.exit("LSF: arm stitching inconsistency")
    targs_lsf = targs_lsf[0]

    orig_setups = (
        ["_".join(setups_original)]
        if targs_join_arms and (targs_nbands == 1) and (len(setups_original) > 1)
        else targs_setups
    )

    # ------------------------------------------------------------------
    # Assemble spaxel cube
    # ------------------------------------------------------------------
    nwave       = len(targs[0].spectra[0].wave)
    pixel_scale = targs[0].spectra[0].wave[1] - targs[0].spectra[0].wave[0]
    targ_len    = len(targs_id)

    cube = {
        k: np.zeros(targ_len, dtype=dt)
        for k, dt in [
            ("aps_id", np.int32), ("x", float), ("y", float),
            ("z", float), ("zerr", float), ("healpix", np.int64),
            ("x_0", float), ("y_0", float), ("wave", float),
            ("snr", float), ("signal", float), ("noise", float),
        ]
    }
    cube.update({
        "targid":      np.empty(targ_len, dtype="U40"),
        "cname":       np.empty(targ_len, dtype="U40"),
        "spec":        np.zeros((nwave, targ_len)),
        "error":       np.zeros((nwave, targ_len)),
        "velscale":    0.0,
        "pixelsize":   0.0,
        "pixel_scale": 0.0,
    })

    configs.update({
        "sens_corr":       sens_corr,      "mask_gaps":    mask_gaps,
        "safe_mask_gaps":  safe_mask_gaps, "vacuum":       vacuum,
        "tellurics":       tellurics,      "fill_gap":     fill_gap,
        "arms_ratio":      arms_ratio,     "funits":       targs_funits,
        "stitched":        targs_join_arms,"infiles":      targs_infiles,
        "orig_setups":     orig_setups,    "targs_setups": targs_setups,
        "orig_wlranges":   wlranges_original,
    })

    hp      = HEALPix(nside=1024, order="nested", frame=ICRS())
    ref_ra  = targs_origin[0]
    ref_dec = targs_origin[1]
    cos_dec = np.cos(np.deg2rad(ref_dec))
    ivar_mask_value = 1.0 / (large_error ** 2)

    for ntgs, tgs in enumerate(targs_id):
        tgs_indx = targs_idfx[tgs]
        cube["aps_id"][ntgs]   = targs[tgs_indx].aps_id
        cube["targid"][ntgs]   = targs[tgs_indx].targid
        cube["cname"][ntgs]    = targs[tgs_indx].cname
        cube["spec"][:, ntgs]  = targs[tgs_indx].spectra[0].flux

        ivar_tgs           = targs[tgs_indx].spectra[0].ivar
        mask_tgs           = ivar_tgs <= 10 * ivar_mask_value
        nomask_tgs         = ~mask_tgs
        ivar_tgs[mask_tgs] = ivar_mask_value
        espec_tgs          = 1.0 / (ivar_tgs ** 0.5)
        cube["error"][:, ntgs] = espec_tgs

        cube["signal"][ntgs] = np.nanmean(
            targs[tgs_indx].spectra[0].flux[nomask_tgs])
        cube["noise"][ntgs]  = np.sqrt(
            np.nanmean(espec_tgs[nomask_tgs] ** 2))
        cube["snr"][ntgs]    = (
            cube["signal"][ntgs] / cube["noise"][ntgs]
            if cube["noise"][ntgs] > 0 else 0.0
        )

        dx_deg = targs[tgs_indx].targra  - ref_ra
        dy_deg = targs[tgs_indx].targdec - ref_dec
        cube["x"][ntgs]    = float(-dx_deg * 3600.0 * cos_dec)
        cube["y"][ntgs]    = float( dy_deg * 3600.0)
        cube["x_0"][ntgs]  = ref_ra
        cube["y_0"][ntgs]  = ref_dec
        cube["z"][ntgs]    = z_input[0]
        cube["zerr"][ntgs] = z_input[1]
        cube["healpix"][ntgs] = hp.skycoord_to_healpix(
            SkyCoord("%fd %fd" % (targs[tgs_indx].targra,
                                   targs[tgs_indx].targdec))
        )

    cube["wave"]       = targs[0].spectra[0].wave / (1 + z_input[0])
    cube["pixelsize"]  = configs["PIXELSIZE"]
    cube["pixel_scale"]= pixel_scale

    # Per-raw-spaxel LSF/flux, keyed by APS_ID -- used below (both Path A
    # and Path B) to build the bin-level resolution buckets when
    # spaxel_weighted_lsf is on. Built unconditionally (cheap: just two
    # dict comprehensions over already-loaded target metadata) rather than
    # gated on the flag, so the flag can be flipped without restructuring
    # this function. Arm index 0: after join_arms every target has exactly
    # one meta entry (see the "LSF: arm stitching inconsistency" check
    # above, which already assumes this) -- 'fwhm' is real per-spaxel data
    # when spaxel_weighted_lsf=True, the same global curve for every
    # spaxel otherwise (see aps_utils._assign_arm_results_to_targets), so
    # this is safe/meaningful either way.
    aps_id_to_fwhm_func = {
        int(cube["aps_id"][ntgs]): (targs[targs_idfx[tgs]].meta[0].get('fwhm') or {}).get('interpolate_function')
        for ntgs, tgs in enumerate(targs_id)
    }
    aps_id_to_flux = {int(cube["aps_id"][ntgs]): float(cube["signal"][ntgs]) for ntgs in range(targ_len)}


# ------------------------------------------------------------------

    # Set PPXF and EMI wavelength windows based on redshift.
    #
    # The config values LMIN_PPXF / LMAX_PPXF are treated as the
    # *preferred low-z window* (used exactly when the full range is
    # observable).  At higher redshift the window adapts automatically.
    #
    # Similarly LMIN_EMI / LMAX_EMI are treated as the preferred
    # emission-line fitting range and are updated in the same way,
    # but with a wider blue tolerance (emission lines are less sensitive
    # to UV template reliability than stellar continuum).
    # ------------------------------------------------------------------
    lmin_ppxf, lmax_ppxf = ppxf_limits(
        z_input[0],
        obs_lmin   = float(np.min(targs[0].spectra[0].wave)),
        obs_lmax   = float(np.max(targs[0].spectra[0].wave)),
        lib_lmin   = float(configs.get("PPXF_LIB_LMIN", 1000.)),
        lib_lmax   = float(configs.get("PPXF_LIB_LMAX", 10000.)),
        lowz_lmin  = float(configs.get("LMIN_PPXF",     3000.)),
        lowz_lmax  = float(configs.get("LMAX_PPXF",     6000.)),
        edge_buffer    = 50.,
        min_window     = 500.,
        cah_blue_clamp = True,
        cah_blue_edge  = float(configs.get("LMIN_PPXF", 3000.)),
        cah_clamp_zmax = 1.0,
    )

    if lmin_ppxf is None:
        # Window too narrow — flag the target and disable PPXF/LS
        print(("WARNING: z=%.4f — rest-frame pPXF window < 500 Å. "
               "PPXF and LS will be skipped for this target.") % z_input[0])
        configs["LMIN_PPXF"] = None
        configs["LMAX_PPXF"] = None
        configs["SKIP_PPXF"] = True
        configs["SKIP_LS"]   = True
    else:
        configs["LMIN_PPXF"] = lmin_ppxf
        configs["LMAX_PPXF"] = lmax_ppxf
        configs["SKIP_PPXF"] = False
        configs["SKIP_LS"]   = False
        print(("pPXF window:  %.0f – %.0f Å rest  "
               "(z=%.4f  window=%.0f Å)") % (
            lmin_ppxf, lmax_ppxf,
            z_input[0], lmax_ppxf - lmin_ppxf))

    # EMI window: same logic but using the config EMI limits as the
    # preferred low-z range.  No Ca H&K clamp — emission lines extend
    # further into the UV and the blue edge matters less for gas kinematics.
    lmin_emi, lmax_emi = ppxf_limits(
        z_input[0],
        obs_lmin   = float(np.min(targs[0].spectra[0].wave)),
        obs_lmax   = float(np.max(targs[0].spectra[0].wave)),
        lib_lmin   = float(configs.get("PPXF_LIB_LMIN", 1000.)),
        lib_lmax   = float(configs.get("PPXF_LIB_LMAX", 10000.)),
        lowz_lmin  = float(configs.get("LMIN_EMI",  2000.)),
        lowz_lmax  = float(configs.get("LMAX_EMI",  8600.)),
        edge_buffer    = 50.,
        min_window     = 200.,    # EMI needs less range than stellar continuum
        cah_blue_clamp = False,   # no blue clamp for emission lines
    )

    if lmin_emi is None:
        configs["LMIN_EMI"] = None
        configs["LMAX_EMI"] = None
        configs["SKIP_EMI"] = True
        print(("WARNING: z=%.4f — EMI window < 200 Å. "
               "EMIPPXF will be skipped.") % z_input[0])
    else:
        configs["LMIN_EMI"] = lmin_emi
        configs["LMAX_EMI"] = lmax_emi
        configs["SKIP_EMI"] = False
        print(("EMI  window:  %.0f – %.0f Å rest  "
               "(z=%.4f  window=%.0f Å)") % (
            lmin_emi, lmax_emi,
            z_input[0], lmax_emi - lmin_emi))



    if str(configs["VELSCALE"]).replace(" ", "").lower() in ["none", "null", ""]:
        s_lam     = len(cube["wave"])
        lam_range = [cube["wave"][0], cube["wave"][-1]]
        dlam      = (lam_range[1] - lam_range[0]) / (s_lam - 1.0)
        lim       = np.array(lam_range) / dlam + np.array([-0.5, 0.5])
        # Pre-existing bug, unrelated to spaxel-weighted LSF, found and
        # fixed here as a disclosed side effect while verifying that
        # feature end to end: np.diff(np.log(lim)) is a real 1-element
        # array (lim has 2 elements), and NumPy >=1.25 raises
        # "TypeError: only 0-dimensional arrays can be converted to
        # Python scalars" on float() of anything but a true scalar --
        # this line has apparently never been exercised against a NumPy
        # this strict before (confirmed live: NumPy 2.4.6, both this
        # session's test environment and the production pyaps-explorer
        # image itself). .item() extracts the single value explicitly.
        cube["velscale"] = float((np.diff(np.log(lim)) / s_lam * Clight).item())
    else:
        cube["velscale"] = configs["VELSCALE"]


    # ------------------------------------------------------------------
    # Adaptive MIN_SNR: removes only technically bad spaxels
    # (dead fibres, negative flux, IFU edge artifacts).
    # The SB flux filter handles science-driven low-SB edge removal.
    # User can override by setting MIN_SNR to a float in the JSON config.
    # ------------------------------------------------------------------
    _min_snr_user = configs.get("MIN_SNR", None)
    _use_adaptive = (
        _min_snr_user is None or
        str(_min_snr_user).strip().lower() in ("none", "null", "")
    )

    if _use_adaptive:
        _min_snr = adaptive_min_snr(
            cube["snr"],
            method           = str(configs.get("MIN_SNR_METHOD",     "percentile")),
            percentile       = float(configs.get("MIN_SNR_PERCENTILE", 2)),
            absolute_floor   = float(configs.get("MIN_SNR_FLOOR",      0.1)),
            absolute_ceiling = float(configs.get("MIN_SNR_CEILING",    1.5)),
        )
        print("MIN_SNR adaptive: %.4g  "
              "(z=%.4f  N_spaxels=%d  SNR p2=%.3f  p50=%.3f  p98=%.3f)" % (
            _min_snr, z_input[0], len(cube["snr"]),
            float(np.percentile(cube["snr"][cube["snr"] > 0], 2))
            if (cube["snr"] > 0).any() else 0.0,
            float(np.median(cube["snr"])),
            float(np.percentile(cube["snr"], 98)),
        ))
    else:
        try:
            _min_snr = float(_min_snr_user)
            print("MIN_SNR from config: %.4g" % _min_snr)
        except (ValueError, TypeError):
            _min_snr = adaptive_min_snr(cube["snr"])
            print("MIN_SNR fallback to adaptive: %.4g "
                  "(could not parse config value: %r)" % (_min_snr, _min_snr_user))

    configs["MIN_SNR"] = _min_snr



    spbin_size   = configs.get("SPBIN_SIZE_EXGAL", -1.0)
    spatial_bins = None     # kept in scope for plot_all

    # ==================================================================
    # PATH A — with spatial pre-binning
    # ==================================================================
    if spbin_size > 0:

        ExGalutil.prettyOutput_Done(
            f"Spatial Binning  size={spbin_size} arcsec", progressbar=True)

        spatial_bins, _ = ExGalPrepare.spatial_bin_with_provenance(
            cube, spbin_size,
            min_snr=configs["MIN_SNR"],
            verbose=True,
            apply_flux_filter=(configs["SB_FILTER"] == 1),
            flux_filter_mode=configs["SB_FILTER_MODE"],
            flux_filter_snr_min=configs.get("SB_FILTER_SNR_MIN", 3.0),
            flux_filter_delta=configs.get("SB_FILTER_DELTA", 2.0),
            flux_filter_min_spaxels=configs.get("SB_FILTER_MIN_SPAXELS", 10),
            flux_filter_percentile=configs.get("SB_FILTER_PERCENT", 10),
            flux_filter_min_keep_fraction=configs["SB_FILTER_MIN_FRAC"],
        )

        idx_inside  = np.where(spatial_bins["flag"] == 1)[0]
        idx_outside = np.where(spatial_bins["flag"] == 0)[0]
        if len(idx_inside) == 0:
            raise ValueError("No valid spaxels remaining after filters!")

        if str(configs["TARGET_SNR"]).replace(" ", "").lower() in ["none", "null", ""]:
            configs["TARGET_SNR"] = find_nearest_snr(
                reference_SNR,
                np.nanquantile(spatial_bins["snr"], 0.9))
            print(f"TARGET_SNR auto-set to {configs['TARGET_SNR']}")

        binNum = ExGalPrepare.define_voronoi_bins(
            configs["VORONOI"],
            spatial_bins["aps_ids"], spatial_bins["targid"],
            spatial_bins["cname"],   spatial_bins["x"],
            spatial_bins["y"],       spatial_bins["z"],
            spatial_bins["zerr"],    spatial_bins["healpix"],
            spatial_bins["x_0"],     spatial_bins["y_0"],
            spatial_bins["signal"],  spatial_bins["noise"],
            cube["pixelsize"],       spatial_bins["snr"],
            configs["TARGET_SNR"],   configs["COVAR_VOR"],
            idx_inside, idx_outside, headname, outpath, configs,
        )

        # aps_id (raw spaxel) -> final Voronoi/PowerBin bin id, composed
        # through the spatial-pre-binning stage: spatial_bins["aps_ids"]
        # is itself "list of original spaxel IDs per [spatial] bin"
        # (ExGalPrepare.spatial_bin_with_provenance's own provenance
        # tracking) -- every raw spaxel inside a given spatial bin
        # inherits that spatial bin's own final Voronoi bin id. Spaxels in
        # spatial bins that never made it into idx_inside (SNR-rejected,
        # flux-filtered, etc.) are simply absent -- same "excluded"
        # convention `bucket_lsf_for_voronoi_bins` already expects.
        aps_id_to_bin = {}
        for k, spatial_pos in enumerate(idx_inside):
            for raw_id in spatial_bins["aps_ids"][spatial_pos]:
                aps_id_to_bin[int(raw_id)] = int(binNum[k])

        log_spec, log_error, logLam = IFUExGalPrepare.log_rebinning(
            spatial_bins, configs, headname, outpath, save=False)

        if configs["VORONOI"] == 1:
            ExGalPrepare.apply_voronoi_bins(
                binNum, log_spec[:, idx_inside], log_error[:, idx_inside],
                headname, outpath, logLam, "log", "IFU")
        else:
            ExGalPrepare.save_binned_spectra_novor(
                log_spec[:, idx_inside], log_error[:, idx_inside],
                logLam, headname, outpath, flag="log", binNum=binNum)

    # ==================================================================
    # PATH B — no spatial pre-binning (direct Voronoi on raw spaxels)
    # ==================================================================
    else:
        ExGalutil.prettyOutput_Done("No Spatial Binning Mode", progressbar=True)

        # Restore the MIN_SNR if PATH A is not active or it has been failed

        idx_inside, idx_outside = \
            IFUExGalPrepare.rejectDefunctSpaxels_applySNRThreshold(cube, configs)
        if len(idx_inside) == 0:
            raise ValueError("No valid spaxels remaining after filters!")

        if str(configs["TARGET_SNR"]).replace(" ", "").lower() in ["none", "null", ""]:
            configs["TARGET_SNR"] = find_nearest_snr(
                reference_SNR,
                np.nanquantile(cube["snr"], 0.9))
            print(f"TARGET_SNR auto-set to {configs['TARGET_SNR']}")

        binNum = ExGalPrepare.define_voronoi_bins(
            configs["VORONOI"],
            cube["aps_id"], cube["targid"], cube["cname"],
            cube["x"],      cube["y"],      cube["z"],
            cube["zerr"],   cube["healpix"],cube["x_0"],
            cube["y_0"],    cube["signal"], cube["noise"],
            cube["pixelsize"], cube["snr"],
            configs["TARGET_SNR"], configs["COVAR_VOR"],
            idx_inside, idx_outside, headname, outpath, configs,
        )

        # No spatial pre-binning stage here -- binNum maps directly onto
        # cube["aps_id"][idx_inside], one raw spaxel per position.
        aps_id_to_bin = {
            int(cube["aps_id"][pos]): int(binNum[k])
            for k, pos in enumerate(idx_inside)
        }

        log_spec, log_error, logLam = \
            IFUExGalPrepare.log_rebinning(cube, configs, headname, outpath)

        if configs["VORONOI"] == 1:
            ExGalPrepare.apply_voronoi_bins(
                binNum, log_spec[:, idx_inside], log_error[:, idx_inside],
                headname, outpath, logLam, "log", "IFU")
        else:
            ExGalPrepare.save_binned_spectra_novor(
                log_spec[:, idx_inside], log_error[:, idx_inside],
                logLam, headname, outpath, flag="log", binNum=binNum)

    # ==================================================================
    # Diagnostic plots — both paths land here
    # ==================================================================
    try:
        voronoi_data = ExGalPrepare.read_voronoi_fits_table(
            os.path.join(outpath, f"{headname}_table.fits"))

        # Legacy three-panel figure (kept for backward compatibility)
        ExGalPrepare.voronoi_visualization(
            spatial_bins if spatial_bins is not None else cube,
            voronoi_data,
            save_path=os.path.join(figdir, f"{headname}_voronoi_cube.png"),
            dpi=72, quantity="snr", show_filtered=False,
        )

        # Professional preparation-chain plots
        plot_all(
            cube=cube,
            spatial_bins=spatial_bins,      # None in path B — handled inside
            voronoi_data=voronoi_data,
            configs=configs,
            headname=headname,
            figdir=figdir,
        )

    except Exception as e:
        print(f"Warning: diagnostic plots failed: {e}")

    # ==================================================================
    # LSF interpolators
    # ==================================================================
    LSF_Data = apply_redshift_to_fwhm_corrected(targs_lsf, z_input[0])
    LSF_raw  = np.genfromtxt(
        os.path.join(IFU_config_dir, "LSF-Config_" + configs["SSP_LIB"]),
        comments="#")
    LSF_Templates = interp1d(
        LSF_raw[:, 0], LSF_raw[:, 1], "linear", fill_value="extrapolate")

    # Opt-in per-bin resolution: bin_to_bucket/LSF_Data_by_bucket are None
    # unless spaxel_weighted_lsf=True, in which case runModule_PPXF/
    # runModule_EMIPPXF/runModule_LINESTRENGTH all take the same-named
    # parameters and fall back to the single LSF_Data above whenever
    # either is None/empty -- see each of their own docstrings.
    bin_to_bucket = None
    LSF_Data_by_bucket = None
    if spaxel_weighted_lsf:
        n_bins = int(np.max(binNum)) + 1 if len(binNum) else 0
        if n_bins > 0:
            bin_to_bucket, LSF_Data_by_bucket = bucket_lsf_for_voronoi_bins(
                aps_id_to_fwhm_func, aps_id_to_flux, aps_id_to_bin, n_bins,
                cube["wave"], max_buckets=configs.get("LSF_N_BUCKETS", 6),
            )
            if LSF_Data_by_bucket:
                print(f"spaxel_weighted_lsf: {len(LSF_Data_by_bucket)} resolution "
                      f"bucket(s) across {n_bins} bin(s) for this patch")

    return {
        "cube":               cube,
        "configs":            configs,
        "LSF_Data":           LSF_Data,
        "LSF_Templates":      LSF_Templates,
        "figdir":             figdir,
        "bin_to_bucket":      bin_to_bucket,
        "LSF_Data_by_bucket": LSF_Data_by_bucket,
    }

# =========================================================================== #
#  STAGE 2 — main ExGal function (owns the loop)                              #
# =========================================================================== #

def ifu_ExGal(
    infiles,
    headname,
    outpath,
    IFU_config_dir,
    ExGal_templates,
    IFU_params=None,
    # --- patch table source (exactly one required) ---
    patch_file=None,
    patch_array=None,
    # --- analysis switches ---
    PPXF=False,
    EMIPPXF=False,
    LS=False,
    nthreads=1,
    # --- spectral flags ---
    wlranges=None,
    aps_ids=None,
    targsrvy=None,
    targclass=None,
    mask_aps_ids=None,
    sens_corr=True,
    mask_gaps=True,
    safe_mask_gaps=True,
    vacuum=False,
    tellurics=False,
    fill_gap=False,
    arms_ratio=None,
    join_arms=False,
    # --- misc ---
    overwrite=True,
    UAPSID=None,
    catdir=None,
    caldir=None,
    no_spec_ext=False,
    spaxel_weighted_lsf=None,
):
    """
    ExGal IFU analysis — loops over a patch table and processes every
    GALAXY / QSO target.

    spaxel_weighted_lsf : bool or None, optional
        Per-bin resolution for PPXF/EMIPPXF/LineStrength template
        preparation. `None` (default): resolved per-patch from each
        patch's own IFU_params JSON (`SPAXEL_WEIGHTED_LSF`, itself
        default **on** as of v1.9) — see `ifu_ExGal_prepare`'s own
        docstring for the full mechanism. Passing `True`/`False` here
        overrides every patch's own config value.

    Patch table resolution
    ----------------------
    1. ``patch_array`` — single-row dict / Row / one-row Table injected
       directly.  ``ctarg_excluded`` is set to False and ignored.
       Classification still runs if Z is NaN and ``class_patch=True``.
    2. ``patch_file``  — path to a FITS or ASCII patch file on disk.
       Loaded, multi-class split applied, ``ctarg_excluded`` derived
       from file content.
    3. Neither provided → AssertionError.

    Parameters
    ----------
    patch_array : dict | astropy.table.Row | one-row Table | None
        Single injected target for debug / test mode.
        Use :func:`make_patch_array` to build one conveniently.
    class_patch : bool
        If True, run Redrock classifier on any row whose Z is NaN
        before entering the analysis loop.  Works in both patch_file
        and patch_array modes.
    """

    assert patch_file is not None or patch_array is not None, \
        "Either patch_file or patch_array must be provided"
    assert Path(ExGal_templates).is_dir(), \
        f"ExGal_templates not found: {ExGal_templates}"
    assert Path(IFU_config_dir).is_dir(), \
        f"IFU_config_dir not found: {IFU_config_dir}"

    # ------------------------------------------------------------------ #
    # 1. Resolve patch table                                              #
    # ------------------------------------------------------------------ #
    if patch_array is not None:
        print("INFO: patch_array mode — single injected row")
        wp_table       = ensure_table(patch_array)
        ctarg_excluded = False
        patch_array_mode = True
    else:
        print(f"INFO: Loading patch file: {patch_file}")
        wp_table, ctarg_excluded = load_and_split_patch_file(patch_file)
        patch_array_mode = False

    print(f"INFO: {len(wp_table)} row(s) loaded  "
          f"ctarg_excluded={ctarg_excluded}")

    # ------------------------------------------------------------------ #
    # 2. Health check                                                     #
    # ------------------------------------------------------------------ #
    # test_patch_table(wp_table, mode="ExGal",
    #                  patch_array_mode=patch_array_mode)

    # ------------------------------------------------------------------ #
    # 3. Resolve IFU_params                                               #
    # ------------------------------------------------------------------ #
    from PyAPS.aps_ifu_prepare import gen_parms
    if IFU_params is None or not Path(str(IFU_params)).exists():
        IFU_params = gen_parms(infiles, IFU_config_dir, wlranges)
        print(f"WARNING: IFU_params auto-derived: {IFU_params}")
    assert Path(IFU_params).exists(), \
        f"IFU_params not found: {IFU_params}"

    # ------------------------------------------------------------------ #
    # 4. Loop over patch table                                            #
    # ------------------------------------------------------------------ #
    proper_targets  = len(wp_table)
    fault_counter   = 0

    for trgs in wp_table:

        # Skip masked sources
        if str(trgs["type"]).replace(" ", "").upper() == "M":
            print("** ExGAL: skip Mask  id=%d" % trgs["id"])
            proper_targets -= 1
            continue

        # Skip non-ExGal / unclassified
        class_value = decode_class(first_valid_class(trgs))
        if class_value not in ("GALAXY", "QSO") or not class_value:
            print("** ExGAL: skip non-ExGal / unclassified  id=%d  "
                  "CLASS=%s" % (trgs["id"], class_value or "—"))
            proper_targets -= 1
            continue

        patch_headname = headname + "_" + ("P%04d" % trgs["id"])

        # Radius enhancement for single / large targets
        path_area = float(trgs["A_world"]) * float(trgs["B_world"])
        if patch_array_mode:
            enhanced_rad_factor = 1.0
        elif (
            len(wp_table[wp_table["type"] == "T"]) == 1 and ctarg_excluded
        ) or (path_area > 0.4 * 0.00054 and ctarg_excluded):
            enhanced_rad_factor = 3.0
            print(f"  Single/large ExGal target — radius x{enhanced_rad_factor}")
        else:
            enhanced_rad_factor = 1.0

        patch_area = [
            float(trgs["RA_icrs"]),  float(trgs["DEC_icrs"]),
            float(trgs["A_world"]) * 3600.0 * enhanced_rad_factor,
            float(trgs["B_world"]) * 3600.0 * enhanced_rad_factor,
            float(trgs["angle"]),
        ]

        # Mask areas from type=M rows (skip in patch_array mode)
        mask_patch_area = None
        if not patch_array_mode:
            mask_rows = wp_table[wp_table["type"] == "M"]
            if len(mask_rows) > 0:
                mask_patch_area = [
                    [float(m["RA_icrs"]), float(m["DEC_icrs"]),
                     float(m["A_world"]) * 3600.0,
                     float(m["B_world"]) * 3600.0,
                     float(m["angle"])]
                    for m in mask_rows
                ]

        z, zerr, _ = first_valid_z(trgs)
        if np.isnan(z):
            print("** ExGAL: skip — NaN redshift  id=%d. "
                  "Use class_patch=True or provide Z in patch_array."
                  % trgs["id"])
            proper_targets -= 1
            continue

        z_input_patch = [z, zerr]

        print("\n" + "=" * 70)
        print("** ExGAL: id=%d  Z=%.5f  CLASS=%s"
              % (trgs["id"], z, class_value))
        print("=" * 70)

        try:
            prep = ifu_ExGal_prepare(
                infiles, patch_headname, IFU_params, outpath,
                IFU_config_dir,
                wlranges=wlranges, aps_ids=aps_ids,
                targsrvy=targsrvy, targclass=targclass,
                mask_aps_ids=mask_aps_ids,
                area=patch_area, mask_areas=mask_patch_area,
                sens_corr=sens_corr, mask_gaps=mask_gaps,
                safe_mask_gaps=safe_mask_gaps, vacuum=vacuum,
                tellurics=tellurics, fill_gap=fill_gap,
                arms_ratio=arms_ratio, join_arms=join_arms,
                z_input=z_input_patch,
                catdir=catdir, caldir=caldir,
                spaxel_weighted_lsf=spaxel_weighted_lsf,
            )

            if prep is None:
                print(f"  Skipped (no valid spaxels in aperture)")
                proper_targets -= 1
                continue

            cube          = prep["cube"]
            configs       = prep["configs"]
            LSF_Data      = prep["LSF_Data"]
            LSF_Templates = prep["LSF_Templates"]
            figdir        = prep["figdir"]
            bin_to_bucket      = prep["bin_to_bucket"]
            LSF_Data_by_bucket = prep["LSF_Data_by_bucket"]

            if PPXF:
                IFUExGalPPXF.runModule_PPXF(
                    nthreads, configs, cube["velscale"],
                    LSF_Data, LSF_Templates,
                    outpath, IFU_config_dir, ExGal_templates,
                    figdir, patch_headname,
                    z, zerr, large_error, debug=False,
                    bin_to_bucket=bin_to_bucket, LSF_Data_by_bucket=LSF_Data_by_bucket)

            if EMIPPXF:
                IFUExGalEMIPPXF.runModule_EMIPPXF(
                    nthreads, configs, cube["velscale"],
                    LSF_Data, LSF_Templates,
                    outpath, IFU_config_dir, ExGal_templates,
                    figdir, patch_headname,
                    z, zerr, large_error,
                    debug=False, diag_plots=True,
                    tie_mode="optimised", save_all_plots=True,
                    bin_to_bucket=bin_to_bucket, LSF_Data_by_bucket=LSF_Data_by_bucket)

            if LS:
                IFUExGalLS.runModule_LINESTRENGTH(
                    configs["LS_MODE"], configs["LS_RES"],
                    nthreads, configs, cube["velscale"],
                    LSF_Data, outpath, IFU_config_dir,
                    ExGal_templates, figdir, patch_headname,
                    debug=False,
                    bin_to_bucket=bin_to_bucket, LSF_Data_by_bucket=LSF_Data_by_bucket)

            ifuExGalL2merge(
                infiles, outpath, patch_headname,
                wlranges=wlranges, outfile_suffix="_APS",
                EMIPPXF_LEVEL="BIN", LS_RES="ADAPTED",
                patch_area=patch_area, patch_id=int(trgs["id"]),
                UAPSID=UAPSID, no_spec_ext=no_spec_ext)

        except ValueError as e:
            if "No valid spaxels remaining after filters!" in str(e):
                print(f"  No valid spaxels — skipping id={trgs['id']}")
            else:
                fault_counter += 1
                import traceback
                print(f"** ExGAL FAILED  id={trgs['id']}: {e}")
                traceback.print_exc()
            sys.stdout.flush()

        except Exception:
            fault_counter += 1
            import traceback
            print(f"** ExGAL FAILED  id={trgs['id']}")
            traceback.print_exc()
            sys.stdout.flush()

    if proper_targets > 0 and proper_targets == fault_counter:
        sys.exit("All ExGal targets failed")



"""
aps_ExGal_worker.py  —  WEAVE IFU ExGal analysis
=================================================
Reads a patch file produced by aps_ifu.py and processes all GALAXY/QSO
targets.  Calls ifu_ExGal() which owns the loop internally.
"""
import argparse
import os
import sys

import numpy as np

import PyAPS
from PyAPS import aps_constants
from PyAPS.aps_utils import l1_fileinfo, none_or_str, print_args, str2bool
from PyAPS.aps_common_args import build_common_parser, resolve_common_args

APSVERS = PyAPS.__version__


def exgal_runner(options=None):
    parser = build_common_parser(
        description="aps_ExGal_worker — IFU ExGal analysis",
        groups=["target_selection", "wavelength", "l1_processing",
                "caldirs", "output"],
        overrides={
            # this script's production default is True, unlike the
            # canonical False shared by most other scripts
            "overwrite": {"default": True},
            # patch-based IFU analysis defaults to True here (vs the
            # canonical False)
            "join_arms": {"default": True},
            "vacuum": {"default": False},
        },
        exclude=["configdir"],  # this script has no --configdir flag
        extra_args=[
            (("--patch_file",), dict(type=none_or_str, required=True,
                                      help="Patch file from aps_ifu.py")),
            (("--IFU_config_dir",), dict(type=none_or_str, required=True)),
            (("--ExGal_templates",), dict(type=none_or_str, required=True)),
            (("--IFU_params",), dict(type=none_or_str, default=None)),
            (("--PPXF",), dict(type=str2bool, default=False)),
            (("--EMIPPXF",), dict(type=str2bool, default=False)),
            (("--LS",), dict(type=str2bool, default=False)),
            (("--mp_ExGal",), dict(type=int, default=1)),
            (("--uapsid",), dict(type=none_or_str, default=None)),
            (("--no_spec_ext",), dict(type=str2bool, default=False)),
            (("--patch_array",), dict(
                type=none_or_str, default=None,
                help=(
                    "Single-target patch row as comma-separated values in fixed order: "
                    "id,RA_deg,Dec_deg,A_arcsec,B_arcsec,Z,ZERR,CLASS "
                    "Example: 1,185.198164,58.092634,101.52,47.25,0.01003,0.0001,GALAXY"
                ),
            )),
        ],
    )

    args = parser.parse_args() if len(sys.argv) > 1 else parser.parse_args(options)

    # resolve_common_args now also enforces
    # assert len(arms_ratio) == len(infiles) and writes the normalized
    # wlranges/arms_ratio back onto args (for print_args()'s own
    # reporting) — both previously absent here; disclosed behavior
    # improvements, not silent changes.
    resolved = resolve_common_args(args)
    wlranges, arms_ratio = resolved.wlranges, resolved.arms_ratio
    aps_ids, targsrvy, targclass, mask_aps_ids = (
        resolved.aps_ids, resolved.targsrvy, resolved.targclass, resolved.mask_aps_ids)

    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)
    outpath = (args.outpath + os.sep).replace(" ", "")


    # --- resolve patch_array ---
    patch_array = None

    if args.patch_array is not None:
        assert args.patch_file is None, \
            "--patch_array and --patch_file are mutually exclusive"
        try:
            parts = [p.strip() for p in args.patch_array.split(",")]
            assert len(parts) == 8, (
                f"--patch_array expects exactly 8 comma-separated values: "
                f"id,RA,Dec,A_arcsec,B_arcsec,Z,ZERR,CLASS  "
                f"(got {len(parts)}: {parts})"
            )
            patch_array = make_patch_array(
                row_id    = int(parts[0]),
                ra        = float(parts[1]),
                dec       = float(parts[2]),
                a_arcsec  = float(parts[3]),
                b_arcsec  = float(parts[4]),
                z         = float(parts[5]),
                zerr      = float(parts[6]),
                class_str = str(parts[7]).upper().strip(),
            )
            print(f"INFO: patch_array parsed: "
                f"id={parts[0]}  RA={parts[1]}  Dec={parts[2]}  "
                f"A={parts[3]}\"  B={parts[4]}\"  "
                f"Z={parts[5]}  CLASS={parts[7]}")
        except (ValueError, AssertionError) as e:
            sys.exit(f"ERROR parsing --patch_array: {e}")




    print_args(args, module="aps_ExGal_worker",
               version=aps_constants.__aps_ifu_version__,
               path=outpath, headname=args.headname)

    ifu_ExGal(
        infiles=args.infiles,
        headname=args.headname,
        outpath=outpath,
        IFU_config_dir=args.IFU_config_dir,
        ExGal_templates=args.ExGal_templates,
        IFU_params=args.IFU_params,
        patch_file=args.patch_file,
        patch_array=patch_array,
        PPXF=args.PPXF, EMIPPXF=args.EMIPPXF, LS=args.LS,
        nthreads=args.mp_ExGal,
        wlranges=wlranges, aps_ids=aps_ids,
        targsrvy=targsrvy, targclass=targclass,
        mask_aps_ids=mask_aps_ids,
        sens_corr=args.sens_corr, mask_gaps=args.mask_gaps,
        safe_mask_gaps=args.safe_mask_gaps, vacuum=args.vacuum,
        tellurics=args.tellurics, fill_gap=args.fill_gap,
        arms_ratio=arms_ratio, join_arms=args.join_arms,
        overwrite=args.overwrite,
        UAPSID=args.uapsid,
        catdir=args.catdir, caldir=args.caldir,
        no_spec_ext=args.no_spec_ext
    )


if __name__ == "__main__":
    # DEMO settings: edit for your setup. Replace the <PYAPS_DATA>, <PYAPS_DIR>, <night>, <runid>, <obid>
    # markers below with your own locations and identifiers (no machine paths belong in this repository).
    debug_LIFU = [
        "--infiles",
        "<PYAPS_DATA>/L1/<night>/stackcube_<runid>.fit",
        "<PYAPS_DATA>/L1/<night>/stackcube_<runid>.fit",
        "--headname",        "LWVE_<target>_01_GR_H1",
        "--outpath",         "<PYAPS_DATA>/L2/<night>/<obid>/",
        "--patch_file",      "<PYAPS_DATA>/L2/<night>/<obid>/LWVE_<target>_01_GR_H1_targets_mod.fits",
        "--IFU_config_dir",  "<PYAPS_DIR>/configs/ExGal_configs/",
        "--ExGal_templates", "<PYAPS_DIR>/PyAPS_templates/templates_ExGal/",
        "--IFU_params",      "<PYAPS_DIR>/configs/ExGal_configs/LIFUHR11.json",
        "--PPXF",    "True", "--EMIPPXF", "True", "--LS", "True",
        "--mp_ExGal","6",
        "--wlranges","None", "--arms_ratio","1.0,1.0",
        "--sens_corr","True","--mask_gaps","True","--safe_mask_gaps","True",
        "--tellurics","False","--vacuum","False","--fill_gap","False",
        "--join_arms","True","--overwrite","True",
        "--caldir","<PYAPS_DATA>/CAL",
        "--catdir","<PYAPS_DATA>/CAT",
    ]
    exgal_runner(options=debug_LIFU)
