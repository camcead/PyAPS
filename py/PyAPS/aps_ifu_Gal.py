"""
aps_ifu_gal.py
==============
Gal IFU pipeline functions.

Public API
----------
ifu_Gal_prepare(...)   ->  dict | None
ifu_Gal(...)
    Loops over a patch table and processes every STAR / WD target.
    Same patch_array / patch_file resolution as ifu_ExGal.

make_patch_array(...)  ->  dict   (same helper as in aps_ifu_exgal)
test_patch_table(...)  ->  bool
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from astropy.coordinates import ICRS, SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy_healpix import HEALPix

from PyAPS import aps_constants
from PyAPS.aps_ifu_utils import (
    adaptive_min_snr,
    decode_class,
    ensure_table,
    first_valid_class,
    first_valid_z,
    load_and_split_patch_file,
    test_patch_table,
)
from PyAPS.aps_L2merge import ifuGalL2merge
from PyAPS.aps_utils import APSOB, fix_non_unicode_string, gen_targlist
from PyAPS import ExGalPrepare as ExGalPrepare
from PyAPS.aps_ifu_spaxel_contrib import bucket_lsf_for_voronoi_bins

__all__ = [
    "ifu_Gal_prepare", "ifu_Gal",
    "make_patch_array", "test_patch_table",
]

Clight      = 299792.458
large_error = aps_constants.large_error


reference_SNR = [20.0, 30.0, 40.0]

def find_nearest_snr(reference_values, value):
    return min(reference_values, key=lambda x: abs(x - value))



# =========================================================================== #
#  Convenience constructor (same interface as aps_ifu_exgal)                  #
# =========================================================================== #

def make_patch_array(
    ra, dec, a_arcsec, b_arcsec, z, zerr, class_str,
    zwarn=0, angle=0.0, row_id=1, row_type="T",
) -> dict:
    """Build a single-row patch_array dict. See aps_ifu_exgal for details."""
    return {
        "id": row_id, "RA_icrs": ra, "DEC_icrs": dec,
        "A_world": a_arcsec / 3600.0, "B_world": b_arcsec / 3600.0,
        "angle": angle, "flag": 0, "type": row_type,
        "Z": [float(z)], "ZERR": [float(zerr)],
        "ZWARN": [int(zwarn)], "CLASS": [str(class_str).strip()],
    }


# =========================================================================== #
#  STAGE 1 — data preparation                                                 #
# =========================================================================== #

def ifu_Gal_prepare(
    infiles, headname, IFU_params, outpath,
    wlranges=None, aps_ids=None, targsrvy=None, targclass=None,
    mask_aps_ids=None, area=None, mask_areas=None,
    sens_corr=True, mask_gaps=True, safe_mask_gaps=True,
    vacuum=False, tellurics=False, fill_gap=False,
    arms_ratio=None, join_arms=True,
    z_input=None,
    spbin_size_gal=None, min_snr_gal=None,
    voronoi_gal=None, target_snr_gal=None,
    catdir=None, caldir=None, IFU_config_dir=None,
    spaxel_weighted_lsf=None,
) -> dict:
    """
    Data preparation stage for the Galactic IFU pipeline.
    Returns dict with keys: gal_targ, configs_gal, lsf_values, figdir,
    bin_to_bucket, lsf_by_bucket.

    spaxel_weighted_lsf : bool or None, optional
        Per-bin resolution for RVS/FERRE. `None` (default): resolved from
        `configs_gal['SPAXEL_WEIGHTED_LSF']` in the IFU_params JSON,
        itself **default on as of v1.9** if the key is absent -- same
        argument > config > default precedence as `VORONOI_GAL`/
        `SPBIN_SIZE_GAL` above. Every bundled LIFU*/MIFU* IFU_params JSON
        also sets this key explicitly. See `aps_ifu_ExGal.ifu_ExGal_
        prepare`'s own docstring for the full mechanism (identical here,
        just producing evaluated arrays for RVS/FERRE instead of
        callables for PPXF/EMIPPXF/LineStrength).
    """
    assert z_input is not None
    join_arms = True  # always forced for Gal

    figdir = Path(outpath) / "figs_Gal"
    figdir.mkdir(parents=True, exist_ok=True)
    figdir = str(figdir) + os.sep

    if not os.path.exists(IFU_params):
        sys.exit("No parameter file: %s" % IFU_params)
    configs_gal = json.load(open(IFU_params))
    configs_gal["CONFIG_FILE"] = os.path.basename(IFU_params)

    # argument > config > default -- see aps_ifu_ExGal.ifu_ExGal_prepare's
    # own identical resolution. Default on as of v1.9 (see that file's
    # own comment for the full rationale).
    if spaxel_weighted_lsf is None:
        spaxel_weighted_lsf = bool(configs_gal.get("SPAXEL_WEIGHTED_LSF", 1))

    # ------------------------------------------------------------------
    # Resolve parameter overrides
    # ------------------------------------------------------------------

    # SPBIN_SIZE_GAL: argument > config > -1.0 (disabled)
    if spbin_size_gal is not None:
        configs_gal["SPBIN_SIZE_GAL"] = spbin_size_gal
        print("  SPBIN_SIZE_GAL overridden by argument: %s" % spbin_size_gal)
    elif configs_gal.get("SPBIN_SIZE_GAL") is not None:
        print("  SPBIN_SIZE_GAL from config: %s" % configs_gal["SPBIN_SIZE_GAL"])
    else:
        configs_gal["SPBIN_SIZE_GAL"] = -1.0
        print("  SPBIN_SIZE_GAL not in config — spatial binning disabled")
    spbin_size_gal = float(configs_gal["SPBIN_SIZE_GAL"]) or -1.0

    # VORONOI_GAL: argument > config > 0 (disabled)
    if voronoi_gal is not None:
        configs_gal["VORONOI_GAL"] = int(voronoi_gal)
        print("  VORONOI_GAL overridden by argument: %s" % voronoi_gal)
    elif configs_gal.get("VORONOI_GAL") is not None:
        print("  VORONOI_GAL from config: %s" % configs_gal["VORONOI_GAL"])
    else:
        configs_gal["VORONOI_GAL"] = 0
        print("  VORONOI_GAL not in config — Voronoi binning disabled")

    configs_gal.setdefault("COVAR_VOR", 0.0)

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
        print("WARNING: No valid APS IDs for this Gal patch. Skipping...")
        return None

    APSOBJ_inst = APSOB(
        infiles, targsrvy=targsrvy, targclass=targclass,
        mask_aps_ids=mask_aps_ids, area=area, mask_areas=mask_areas,
        wlranges=wlranges, aps_ids=aps_ids_in_class,
        sens_corr=sens_corr, mask_gaps=mask_gaps,
        safe_mask_gaps=safe_mask_gaps, vacuum=vacuum,
        tellurics=tellurics, fill_gap=fill_gap,
        arms_ratio=arms_ratio, join_arms=True,
        catdir=catdir, caldir=caldir, configdir=IFU_config_dir,
        spaxel_weighted_lsf=spaxel_weighted_lsf,
    )

    targs             = APSOBJ_inst.data()
    targs_infiles     = APSOBJ_inst.infiles()
    targs_id          = APSOBJ_inst.id()
    targs_idfx        = APSOBJ_inst.idfx()
    targs_funits      = APSOBJ_inst.funits()
    targs_mode        = APSOBJ_inst.mode()
    targs_join_arms   = APSOBJ_inst.join_arms()
    targs_origin      = APSOBJ_inst.origin()
    targs_setups      = APSOBJ_inst.setups()
    setups_original   = APSOBJ_inst.setups_original()
    wlranges_original = APSOBJ_inst.wlranges_original()
    targs_lsf         = APSOBJ_inst.get_fwhm(aps_id=None, fwhm_key="gfwhm")

    if targs_lsf is not None and len(targs_lsf) > 1:
        targs_lsf = targs_lsf[0]

    # ------------------------------------------------------------------
    # Assemble spaxel cube
    # ------------------------------------------------------------------
    nwave    = len(targs[0].spectra[0].wave)
    targ_len = len(targs_id)

    gal_targ = {
        k: np.zeros(targ_len, dtype=dt) for k, dt in [
            ("aps_id", np.int32), ("x", float), ("y", float),
            ("z", float), ("zerr", float), ("healpix", np.int64),
            ("x_0", float), ("y_0", float), ("wave", float),
            ("snr", float), ("signal", float), ("noise", float),
        ]
    }
    gal_targ.update({
        "targid":    np.empty(targ_len, dtype="U40"),
        "cname":     np.empty(targ_len, dtype="U40"),
        "spec":      np.zeros((nwave, targ_len)),
        "error":     np.zeros((nwave, targ_len)),
        "velscale":  0.0,
        "pixelsize": 0.0,
    })

    configs_gal.update({
        "sens_corr":    sens_corr,    "mask_gaps":    mask_gaps,
        "safe_mask_gaps": safe_mask_gaps, "vacuum":   vacuum,
        "tellurics":    tellurics,    "fill_gap":     fill_gap,
        "arms_ratio":   arms_ratio,   "funits":       targs_funits,
        "stitched":     True,         "infiles":      targs_infiles,
        "join_arms":    True,         "targs_mode":   targs_mode,
        "orig_setups":  setups_original,
        "targs_setups": targs_setups,
        "orig_wlranges": wlranges_original,
    })

    hp      = HEALPix(nside=1024, order="nested", frame=ICRS())
    ref_ra  = targs_origin[0]
    ref_dec = targs_origin[1]
    cos_dec = np.cos(np.deg2rad(ref_dec))
    ivar_mask_value = 1.0 / (large_error ** 2)

    for ntgs, tgs in enumerate(targs_id):
        tgs_indx = targs_idfx[tgs]
        gal_targ["targid"][ntgs]   = targs[tgs_indx].targid
        gal_targ["aps_id"][ntgs]   = targs[tgs_indx].aps_id
        gal_targ["cname"][ntgs]    = targs[tgs_indx].cname
        gal_targ["spec"][:, ntgs]  = targs[tgs_indx].spectra[0].flux

        ivar_tgs           = targs[tgs_indx].spectra[0].ivar
        mask_tgs           = ivar_tgs <= 10 * ivar_mask_value
        nomask_tgs         = ~mask_tgs
        ivar_tgs[mask_tgs] = ivar_mask_value
        espec_tgs          = 1.0 / (ivar_tgs ** 0.5)
        gal_targ["error"][:, ntgs] = espec_tgs

        gal_targ["signal"][ntgs] = np.nanmean(
            targs[tgs_indx].spectra[0].flux[nomask_tgs])
        gal_targ["noise"][ntgs]  = np.sqrt(
            np.nanmean(espec_tgs[nomask_tgs] ** 2))
        gal_targ["snr"][ntgs]    = (
            gal_targ["signal"][ntgs] / gal_targ["noise"][ntgs]
            if gal_targ["noise"][ntgs] > 0 else 0.0
        )

        dx_deg = targs[tgs_indx].targra  - ref_ra
        dy_deg = targs[tgs_indx].targdec - ref_dec
        gal_targ["x"][ntgs]    = float(-dx_deg * 3600.0 * cos_dec)
        gal_targ["y"][ntgs]    = float(dy_deg * 3600.0)
        gal_targ["z"][ntgs]    = z_input[0]
        gal_targ["zerr"][ntgs] = z_input[1]
        gal_targ["x_0"][ntgs]  = ref_ra
        gal_targ["y_0"][ntgs]  = ref_dec
        gal_targ["healpix"][ntgs] = hp.skycoord_to_healpix(
            SkyCoord("%fd %fd" % (targs[tgs_indx].targra,
                                   targs[tgs_indx].targdec)))

    gal_targ["wave"]      = targs[0].spectra[0].wave
    gal_targ["velscale"]  = 0
    gal_targ["pixelsize"] = 0.0

    lsf_values = targs_lsf[0](gal_targ["wave"]) if targs_lsf is not None else None

    # Per-raw-spaxel LSF/flux, keyed by APS_ID -- see aps_ifu_ExGal.
    # ifu_ExGal_prepare's own identical comment for the full rationale.
    aps_id_to_fwhm_func = {
        int(gal_targ["aps_id"][ntgs]): (targs[targs_idfx[tgs]].meta[0].get('fwhm') or {}).get('interpolate_function')
        for ntgs, tgs in enumerate(targs_id)
    }
    aps_id_to_flux = {int(gal_targ["aps_id"][ntgs]): float(gal_targ["signal"][ntgs]) for ntgs in range(targ_len)}

    # ------------------------------------------------------------------
    # Adaptive MIN_SNR_GAL: removes only technically bad spaxels
    # (dead fibres, negative flux, IFU edge artifacts).
    # For Gal mode the adaptive ceiling is slightly higher than ExGal
    # (default 3.0 vs 1.5) because stellar continuum is always present
    # and a spaxel with SNR < 1 in Gal mode is genuinely noise-dominated.
    # ------------------------------------------------------------------
    _min_snr_gal_user = configs_gal.get("MIN_SNR_GAL", None)
    _use_adaptive = (
        _min_snr_gal_user is None or
        str(_min_snr_gal_user).strip().lower() in ("none", "null", "")
    )

    if _use_adaptive:
        _min_snr_gal = adaptive_min_snr(
        gal_targ["snr"],
        method           = "percentile",  # hardcoded for Gal
        percentile       = 3,             # hardcoded for Gal
        absolute_floor   = 0.3,           # hardcoded for Gal
        absolute_ceiling = 3.0,           # higher than ExGal — stellar continuum always present
        )
        print("MIN_SNR_GAL adaptive: %.4g  "
              "(z=%.4f  N_spaxels=%d  SNR p2=%.3f  p50=%.3f  p98=%.3f)" % (
            _min_snr_gal, z_input[0], len(gal_targ["snr"]),
            float(np.percentile(gal_targ["snr"][gal_targ["snr"] > 0], 2))
            if (gal_targ["snr"] > 0).any() else 0.0,
            float(np.median(gal_targ["snr"])),
            float(np.percentile(gal_targ["snr"], 98)),
        ))
    else:
        try:
            _min_snr_gal = float(_min_snr_gal_user)
            print("MIN_SNR_GAL from config: %.4g" % _min_snr_gal)
        except (ValueError, TypeError):
            _min_snr_gal = adaptive_min_snr(gal_targ["snr"])
            print("MIN_SNR_GAL fallback to adaptive: %.4g "
                  "(could not parse config value: %r)" % (_min_snr_gal, _min_snr_gal_user))

    configs_gal["MIN_SNR_GAL"] = _min_snr_gal

    # Build a configs dict with the keys plot_all expects
    # (mirrors ExGal convention so apsPlot.exgal_prepare.plot_all works unchanged)
    _plot_configs = {
        "MIN_SNR":          configs_gal.get("MIN_SNR_GAL"),
        "TARGET_SNR":       configs_gal.get("TARGET_SNR_GAL"),
        "SPBIN_SIZE_EXGAL": spbin_size_gal if spbin_size_gal > 0 else None,
        "VORONOI":          configs_gal.get("VORONOI_GAL"),
        "SB_FILTER":        0,   # no flux/SB filter in Gal mode
    }

    spatial_bins_for_plot = None   # kept in scope for plot_all

    # ==================================================================
    # PATH A — with spatial pre-binning
    # ==================================================================
    if spbin_size_gal > 0:
        spatial_bins, _ = ExGalPrepare.spatial_bin_with_provenance(
            gal_targ, spbin_size_gal,
            min_snr=configs_gal["MIN_SNR_GAL"],
            apply_flux_filter=False,
            verbose=True,
        )
        spatial_bins_for_plot = spatial_bins
        idx_inside  = np.where(spatial_bins["flag"] == 1)[0]
        idx_outside = np.where(spatial_bins["flag"] == 0)[0]
        if len(idx_inside) == 0:
            raise ValueError("No valid spaxels remaining after filters!")

        binNum = ExGalPrepare.define_voronoi_bins(
            configs_gal["VORONOI_GAL"],
            spatial_bins["aps_ids"], spatial_bins["targid"],
            spatial_bins["cname"],   spatial_bins["x"],
            spatial_bins["y"],       spatial_bins["z"],
            spatial_bins["zerr"],    spatial_bins["healpix"],
            spatial_bins["x_0"],     spatial_bins["y_0"],
            spatial_bins["signal"],  spatial_bins["noise"],
            gal_targ["pixelsize"],   spatial_bins["snr"],
            configs_gal["TARGET_SNR_GAL"], configs_gal["COVAR_VOR"],
            idx_inside, idx_outside, headname, outpath, configs_gal,
        )

        # aps_id (raw spaxel) -> final bin id, composed through spatial
        # pre-binning -- see aps_ifu_ExGal.ifu_ExGal_prepare's own
        # identical composition for the full rationale.
        aps_id_to_bin = {}
        for k, spatial_pos in enumerate(idx_inside):
            for raw_id in spatial_bins["aps_ids"][spatial_pos]:
                aps_id_to_bin[int(raw_id)] = int(binNum[k])

        if configs_gal["VORONOI_GAL"] == 1:
            ExGalPrepare.apply_voronoi_bins(
                binNum,
                spatial_bins["spec"][:, idx_inside],
                spatial_bins["error"][:, idx_inside],
                headname, outpath, spatial_bins["wave"], "lin", "IFU",
            )
        else:
            ExGalPrepare.save_binned_spectra_novor(
                spatial_bins["spec"][:, idx_inside],
                spatial_bins["error"][:, idx_inside],
                spatial_bins["wave"], headname, outpath,
                flag="lin", binNum=binNum,
            )

    # ==================================================================
    # PATH B — no spatial pre-binning
    # ==================================================================
    else:
        idx_inside  = np.where(gal_targ["snr"] >= configs_gal["MIN_SNR_GAL"])[0]
        idx_outside = np.where(gal_targ["snr"] <  configs_gal["MIN_SNR_GAL"])[0]
        if len(idx_inside) == 0:
            raise ValueError("No valid spaxels remaining after filters!")

        binNum = ExGalPrepare.define_voronoi_bins(
            configs_gal["VORONOI_GAL"],
            gal_targ["aps_id"],  gal_targ["targid"],
            gal_targ["cname"],   gal_targ["x"],
            gal_targ["y"],       gal_targ["z"],
            gal_targ["zerr"],    gal_targ["healpix"],
            gal_targ["x_0"],     gal_targ["y_0"],
            gal_targ["signal"],  gal_targ["noise"],
            gal_targ["pixelsize"], gal_targ["snr"],
            configs_gal["TARGET_SNR_GAL"], configs_gal["COVAR_VOR"],
            idx_inside, idx_outside, headname, outpath, configs_gal,
        )

        # No spatial pre-binning stage here -- binNum maps directly onto
        # gal_targ["aps_id"][idx_inside], one raw spaxel per position.
        aps_id_to_bin = {
            int(gal_targ["aps_id"][pos]): int(binNum[k])
            for k, pos in enumerate(idx_inside)
        }

        if configs_gal["VORONOI_GAL"] == 1:
            ExGalPrepare.apply_voronoi_bins(
                binNum,
                gal_targ["spec"][:, idx_inside],
                gal_targ["error"][:, idx_inside],
                headname, outpath, gal_targ["wave"], "lin", "IFU",
            )
        else:
            ExGalPrepare.save_binned_spectra_novor(
                gal_targ["spec"][:, idx_inside],
                gal_targ["error"][:, idx_inside],
                gal_targ["wave"], headname, outpath,
                flag="lin", binNum=binNum,
            )

    # ==================================================================
    # Diagnostic plots — both paths land here
    # ==================================================================
    try:
        from PyAPS.apsPlot.exgal_prepare import plot_all as _plot_all

        voronoi_data = ExGalPrepare.read_voronoi_fits_table(
            os.path.join(outpath, "%s_table.fits" % headname))

        # Legacy spatial plot (kept for backward compat)
        try:
            ExGalPrepare.plot_spatial(
                gal_targ,
                spatial_bins=spatial_bins_for_plot,
                color_field="snr",
                bin_size_label=spbin_size_gal if spbin_size_gal > 0 else None,
                figdir=figdir,
                headname=headname,
            )
        except Exception as _e:
            print("Warning: plot_spatial failed: %s" % _e)

        # Professional preparation-chain plots
        _plot_all(
            cube=gal_targ,
            spatial_bins=spatial_bins_for_plot,   # None in path B
            voronoi_data=voronoi_data,
            configs=_plot_configs,
            headname=headname,
            figdir=figdir,
        )

    except Exception as exc:
        print("Warning: Gal diagnostic plots failed: %s" % exc)

    # Opt-in per-bin resolution: bin_to_bucket/lsf_by_bucket are None
    # unless spaxel_weighted_lsf resolved True, in which case
    # run_rvs_for_ifu_gal/run_ferre_for_ifu_gal's own same-named
    # parameters fall back to the single lsf_values above whenever either
    # is None/empty.
    bin_to_bucket = None
    lsf_by_bucket = None
    if spaxel_weighted_lsf:
        n_bins = int(np.max(binNum)) + 1 if len(binNum) else 0
        if n_bins > 0:
            bin_to_bucket, LSF_Data_by_bucket = bucket_lsf_for_voronoi_bins(
                aps_id_to_fwhm_func, aps_id_to_flux, aps_id_to_bin, n_bins,
                gal_targ["wave"], max_buckets=configs_gal.get("LSF_N_BUCKETS", 6),
            )
            if LSF_Data_by_bucket:
                # run_rvs_for_ifu_gal/run_ferre_for_ifu_gal consume plain
                # evaluated arrays (matching lsf_values's own convention),
                # not callables -- evaluate each bucket's curve once here.
                lsf_by_bucket = {b: f(gal_targ["wave"]) for b, f in LSF_Data_by_bucket.items()}
                print(f"spaxel_weighted_lsf: {len(lsf_by_bucket)} resolution "
                      f"bucket(s) across {n_bins} bin(s) for this patch")

    return {
        "gal_targ":      gal_targ,
        "configs_gal":   configs_gal,
        "lsf_values":    lsf_values,
        "figdir":        figdir,
        "bin_to_bucket": bin_to_bucket,
        "lsf_by_bucket": lsf_by_bucket,
    }


# =========================================================================== #
#  STAGE 2 — main Gal function (owns the loop)                                #
# =========================================================================== #

def ifu_Gal(
    infiles,
    headname,
    outpath,
    IFU_config_dir,
    IFU_params=None,
    # --- patch table source ---
    patch_file=None,
    patch_array=None,
    # --- FERRE / RVS ---
    rvs_config=None,
    ferre_exe=None,
    ferre_templates=None,
    ferre_grid_ids=None,
    ferre_grid_prefix=None,
    # --- binning overrides ---
    spbin_size_gal=None,
    min_snr_gal=None,
    voronoi_gal=None,
    target_snr_gal=None,
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
    join_arms=True,
    # --- classification (if Z is missing) ---
    class_patch=False,
    class_templates=None,
    class_templates_ARC=None,
    class_z_rad=1.5,
    class_ntop=1,
    mp_prep=1,
    # --- threading ---
    nthreads=1,
    # --- misc ---
    overwrite=True,
    UAPSID=None,
    catdir=None,
    caldir=None,
    no_spec_ext=False,
    spaxel_weighted_lsf=None,
):
    """
    Gal IFU analysis — loops over a patch table and processes every
    STAR / WD target.

    Patch table resolution (same as ifu_ExGal):
      1. patch_array — single injected row, ctarg_excluded ignored.
      2. patch_file  — loaded from disk, split applied.
      3. Neither     → AssertionError.

    spaxel_weighted_lsf : bool or None, optional
        Per-bin resolution for RVS/FERRE. `None` (default): resolved
        per-patch from each patch's own IFU_params JSON
        (`SPAXEL_WEIGHTED_LSF`, itself default **on** as of v1.9) — see
        `ifu_Gal_prepare`'s own docstring. Passing `True`/`False` here
        overrides every patch's own config value.
    """

    assert patch_file is not None or patch_array is not None, \
        "Either patch_file or patch_array must be provided"
    assert Path(IFU_config_dir).is_dir(), \
        f"IFU_config_dir not found: {IFU_config_dir}"

    # ------------------------------------------------------------------ #
    # 1. Resolve patch table                                              #
    # ------------------------------------------------------------------ #
    if patch_array is not None:
        print("INFO: patch_array mode — single injected row")
        wp_table         = ensure_table(patch_array)
        ctarg_excluded   = False
        patch_array_mode = True
    else:
        print(f"INFO: Loading patch file: {patch_file}")
        wp_table, ctarg_excluded = load_and_split_patch_file(patch_file)
        patch_array_mode = False

    print(f"INFO: {len(wp_table)} row(s)  ctarg_excluded={ctarg_excluded}")

    # ------------------------------------------------------------------ #
    # 2. Optional classification for missing Z                           #
    # ------------------------------------------------------------------ #
    if class_patch:
        assert class_templates is not None and Path(class_templates).is_dir(), \
            "class_templates required when class_patch=True"
        import tempfile

        from PyAPS.aps_ifu_prepare import ifu_class
        figdir_cls = str(Path(outpath) / "figs")
        Path(figdir_cls).mkdir(parents=True, exist_ok=True)
        tmp_patch = tempfile.NamedTemporaryFile(
            suffix=".fits", delete=False, dir=outpath)
        tmp_patch.close()
        fits.table_to_hdu(wp_table).writeto(tmp_patch.name, overwrite=True)
        wp_table = ifu_class(
            infiles, headname, tmp_patch.name, class_templates,
            class_templates_ARC=class_templates_ARC,
            aps_ids=aps_ids, mask_aps_ids=mask_aps_ids,
            targsrvy=targsrvy, targclass=targclass,
            z_rad=class_z_rad, wlranges=wlranges,
            figdir=figdir_cls, ncpus=mp_prep,
            arms_ratio=arms_ratio, class_ntop=class_ntop,
            catdir=catdir, caldir=caldir,
            IFU_config_dir=IFU_config_dir,
        )
        Path(tmp_patch.name).unlink(missing_ok=True)

    # ------------------------------------------------------------------ #
    # 3. Health check                                                     #
    # ------------------------------------------------------------------ #
    # test_patch_table(wp_table, mode="Gal",
    #                  patch_array_mode=patch_array_mode)

    # ------------------------------------------------------------------ #
    # 4. Resolve IFU_params                                               #
    # ------------------------------------------------------------------ #
    from PyAPS.aps_ifu_prepare import gen_parms
    if IFU_params is None or not Path(str(IFU_params)).exists():
        IFU_params = gen_parms(infiles, IFU_config_dir, wlranges)
        print(f"WARNING: IFU_params auto-derived: {IFU_params}")
    assert Path(IFU_params).exists(), f"IFU_params not found: {IFU_params}"

    # ------------------------------------------------------------------ #
    # 5. Loop over patch table                                            #
    # ------------------------------------------------------------------ #
    enhanced_rad_factor = 0.5   # always shrink aperture for Gal sources
    proper_targets      = len(wp_table)
    fault_counter       = 0

    for trgs in wp_table:

        # Skip masked
        if str(trgs["type"]).replace(" ", "").upper() == "M":
            print("** GAL: skip Mask  id=%d" % trgs["id"])
            proper_targets -= 1
            continue

        # Skip central target when other extracted targets exist
        if (
            not patch_array_mode
            and str(trgs["type"]).replace(" ", "").upper() == "C"
            and "T" in list(wp_table["type"])
        ):
            print("** GAL: skip Central target  id=%d" % trgs["id"])
            proper_targets -= 1
            continue

        # Skip ExGal or unclassified
        class_value = fix_non_unicode_string(
            decode_class(first_valid_class(trgs)).upper()
        )
        if class_value in ("GALAXY", "QSO") or not class_value:
            print("** GAL: skip non-Gal / unclassified  id=%d  CLASS=%s"
                  % (trgs["id"], class_value or "—"))
            proper_targets -= 1
            continue

        patch_headname = headname + "_" + ("P%04d" % trgs["id"])

        rad_factor = 1.0 if patch_array_mode else enhanced_rad_factor
        patch_area = [
            float(trgs["RA_icrs"]),  float(trgs["DEC_icrs"]),
            float(trgs["A_world"]) * 3600.0 * rad_factor,
            float(trgs["B_world"]) * 3600.0 * rad_factor,
            float(trgs["angle"]),
        ]

        z, zerr, _ = first_valid_z(trgs)
        if np.isnan(z):
            print("** GAL: skip — NaN redshift  id=%d" % trgs["id"])
            proper_targets -= 1
            continue

        z_input_patch = [z, zerr]

        print("\n" + "=" * 70)
        print("** GAL: id=%d  Z=%.5f  CLASS=%s"
              % (trgs["id"], z, class_value))
        print("=" * 70)

        try:
            prep = ifu_Gal_prepare(
                infiles, patch_headname, IFU_params, outpath,
                wlranges=wlranges, aps_ids=aps_ids,
                targsrvy=targsrvy, targclass=targclass,
                mask_aps_ids=mask_aps_ids,
                area=patch_area, mask_areas=None,
                sens_corr=sens_corr, mask_gaps=mask_gaps,
                safe_mask_gaps=safe_mask_gaps, vacuum=vacuum,
                tellurics=tellurics, fill_gap=fill_gap,
                arms_ratio=arms_ratio, join_arms=True,
                z_input=z_input_patch,
                spbin_size_gal=spbin_size_gal,
                min_snr_gal=min_snr_gal,
                voronoi_gal=voronoi_gal,
                target_snr_gal=target_snr_gal,
                catdir=catdir, caldir=caldir,
                IFU_config_dir=IFU_config_dir,
                spaxel_weighted_lsf=spaxel_weighted_lsf,
            )

            if prep is None:
                print(f"  Skipped (no valid spaxels in aperture)")
                proper_targets -= 1
                continue

            configs_gal = prep["configs_gal"]
            lsf_values  = prep["lsf_values"]
            figdir      = prep["figdir"]
            bin_to_bucket = prep["bin_to_bucket"]
            lsf_by_bucket = prep["lsf_by_bucket"]

            # Resolve RVS / FERRE config

            # Priority: explicit argument > config file > hardcoded default (last resort only)
            for attr, key, hardcoded_default in [
                (rvs_config,        "RVS_CONFIG",        None),
                (ferre_exe,         "FERRE_EXE",         None),
                (ferre_templates,   "FERRE_TEMPLATES",   None),
                (ferre_grid_prefix, "FERRE_GRID_PREFIX", "n"),
            ]:
                if attr is not None:
                    # explicit command-line argument wins
                    configs_gal[key] = attr
                elif configs_gal.get(key):
                    # already in config file — use it, print for visibility
                    print(f"  {key} from config: {configs_gal[key]}")
                elif hardcoded_default is not None:
                    # genuine last resort
                    configs_gal[key] = hardcoded_default
                    print(f"  WARNING: {key} not in config — using hardcoded default: {hardcoded_default}")
                else:
                    print(f"  WARNING: {key} not set (not in config, not passed as argument)")

            # ferre_grid_ids — same logic but needs list handling
            if ferre_grid_ids is not None:
                configs_gal["FERRE_GRID_IDS"] = (
                    ",".join(str(x) for x in ferre_grid_ids)
                    if isinstance(ferre_grid_ids, list)
                    else ferre_grid_ids
                )
            elif configs_gal.get("FERRE_GRID_IDS"):
                print(f"  FERRE_GRID_IDS from config: {configs_gal['FERRE_GRID_IDS']}")
            else:
                print("  WARNING: FERRE_GRID_IDS not set")


            _ferre_ids = [
                str(x).strip()
                for x in configs_gal["FERRE_GRID_IDS"].split(",")
            ] if isinstance(configs_gal["FERRE_GRID_IDS"], str) \
              else list(configs_gal["FERRE_GRID_IDS"])

            # RVS
            from PyAPS.aps_ifu_rvs import run_rvs_for_ifu_gal
            rvs_ok = run_rvs_for_ifu_gal(
                headname=patch_headname, outpath=outpath,
                rvs_config=configs_gal["RVS_CONFIG"],
                figdir=figdir, nthreads=nthreads,
                configs_gal=configs_gal, overwrite=overwrite,
                lsf=lsf_values,
                bin_to_bucket=bin_to_bucket, lsf_by_bucket=lsf_by_bucket)
            if not rvs_ok:
                print("WARNING: RVS failed for id=%d" % trgs["id"])

            # FERRE
            _fe = configs_gal.get("FERRE_EXE")
            _ft = configs_gal.get("FERRE_TEMPLATES")
            if _fe and _ft:
                from PyAPS.aps_ifu_ferre import run_ferre_for_ifu_gal
                fe_ok = run_ferre_for_ifu_gal(
                    headname=patch_headname, outpath=outpath,
                    ferre_exe=_fe, ferre_templates=_ft,
                    ferre_grid_ids=_ferre_ids,
                    ferre_grid_prefix=configs_gal["FERRE_GRID_PREFIX"],
                    configs_gal=configs_gal, nthreads=nthreads,
                    overwrite=overwrite, outspec=True, lsf=lsf_values,
                    bin_to_bucket=bin_to_bucket, lsf_by_bucket=lsf_by_bucket)
                if not fe_ok:
                    print("WARNING: FERRE failed for id=%d" % trgs["id"])
            else:
                print("INFO: FERRE skipped (ferre_exe or ferre_templates not set)")

            ifuGalL2merge(
                infiles, outpath, patch_headname,
                wlranges=wlranges, outfile_suffix="_APS",
                patch_area=patch_area, patch_id=int(trgs["id"]),
                UAPSID=UAPSID, no_spec_ext=no_spec_ext)

        except Exception:
            fault_counter += 1
            import traceback
            print("** GAL FAILED  id=%d" % trgs["id"])
            traceback.print_exc()
            sys.stdout.flush()

    if proper_targets > 0 and proper_targets == fault_counter:
        sys.exit("All Gal targets failed")


"""
aps_Gal_worker.py  —  WEAVE IFU Gal analysis
=============================================
Reads a patch file produced by aps_ifu.py and processes all STAR/WD
targets.  Calls ifu_Gal() which owns the loop internally.
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


def gal_runner(options=None):
    parser = build_common_parser(
        description="aps_Gal_worker — IFU Gal analysis",
        groups=["target_selection", "wavelength", "l1_processing",
                "caldirs", "output"],
        overrides={
            # this script's production default is True, unlike the
            # canonical False shared by most other scripts
            "overwrite": {"default": True},
            "join_arms": {"default": True},
            "vacuum": {"default": False},
        },
        exclude=["configdir"],  # this script has no --configdir flag
        extra_args=[
            (("--patch_file",), dict(type=none_or_str, required=True)),
            (("--IFU_config_dir",), dict(type=none_or_str, required=False, default=None)),
            (("--IFU_params",), dict(type=none_or_str, default=None)),
            (("--ferre_exe",), dict(type=none_or_str, default=None)),
            (("--ferre_templates",), dict(type=none_or_str, default=None)),
            (("--ferre_grid_ids",), dict(type=none_or_str, default=None)),
            (("--ferre_grid_prefix",), dict(type=none_or_str, default=None)),
            (("--rvs_config",), dict(type=none_or_str, default=None)),
            (("--spbin_size_gal",), dict(type=float, default=None)),
            (("--min_snr_gal",), dict(type=float, default=None)),
            (("--voronoi_gal",), dict(type=int, default=None)),
            (("--target_snr_gal",), dict(type=float, default=None)),
            (("--class_patch",), dict(type=str2bool, default=False)),
            (("--class_templates",), dict(type=none_or_str, default=None)),
            (("--class_templates_ARC",), dict(type=none_or_str, default=None)),
            (("--class_z_rad",), dict(type=float, default=1.5)),
            (("--class_ntop",), dict(type=int, default=1)),
            (("--mp_prep",), dict(type=int, default=1)),
            (("--mp_Gal",), dict(type=int, default=1)),
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
    # assert len(arms_ratio) == len(infiles), applies the join_arms=False
    # correction for < 2 infiles (previously entirely absent here — a
    # real, disclosed behavior fix), and writes the normalized
    # wlranges/arms_ratio back onto args (for print_args()'s own
    # reporting), also previously absent.
    resolved = resolve_common_args(args)
    wlranges, arms_ratio = resolved.wlranges, resolved.arms_ratio
    aps_ids, targsrvy, targclass, mask_aps_ids = (
        resolved.aps_ids, resolved.targsrvy, resolved.targclass, resolved.mask_aps_ids)

    ferre_grid_ids = (
        [x.strip() for x in args.ferre_grid_ids.split(",")]
        if args.ferre_grid_ids else None
    )

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




    print_args(args, module="aps_Gal_worker",
               version=aps_constants.__aps_ifu_version__,
               path=outpath, headname=args.headname)

    ifu_Gal(
        infiles=args.infiles,
        headname=args.headname,
        outpath=outpath,
        IFU_config_dir=args.IFU_config_dir,
        IFU_params=args.IFU_params,
        patch_file=args.patch_file,
        patch_array=patch_array,
        rvs_config=args.rvs_config,
        ferre_exe=args.ferre_exe,
        ferre_templates=args.ferre_templates,
        ferre_grid_ids=ferre_grid_ids,
        ferre_grid_prefix=args.ferre_grid_prefix,
        spbin_size_gal=args.spbin_size_gal,
        min_snr_gal=args.min_snr_gal,
        voronoi_gal=args.voronoi_gal,
        target_snr_gal=args.target_snr_gal,
        wlranges=wlranges, aps_ids=aps_ids,
        targsrvy=targsrvy, targclass=targclass,
        mask_aps_ids=mask_aps_ids,
        sens_corr=args.sens_corr, mask_gaps=args.mask_gaps,
        safe_mask_gaps=args.safe_mask_gaps, vacuum=args.vacuum,
        tellurics=args.tellurics, fill_gap=args.fill_gap,
        arms_ratio=arms_ratio, join_arms=args.join_arms,
        class_patch=args.class_patch,
        class_templates=args.class_templates,
        class_templates_ARC=args.class_templates_ARC,
        class_z_rad=args.class_z_rad,
        class_ntop=args.class_ntop,
        mp_prep=args.mp_prep,
        nthreads=args.mp_Gal,
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
        "--headname",       "LWVE_<target>_01_GR_H1",
        "--outpath",        "<PYAPS_DATA>/L2/<night>/<obid>/",
        "--patch_file",     "<PYAPS_DATA>/L2/<night>/<obid>/LWVE_<target>_01_GR_H1_targets_mod.fits",
        "--IFU_config_dir", "<PYAPS_DIR>/configs/ExGal_configs/",
        "--IFU_params",     "<PYAPS_DIR>/configs/ExGal_configs/LIFUHR11.json",
        "--mp_Gal",         "4",
        "--wlranges",       "None", "--arms_ratio","1.0,1.0",
        "--sens_corr","True","--mask_gaps","True","--safe_mask_gaps","True",
        "--tellurics","False","--vacuum","False","--fill_gap","False",
        "--rvs_config",     "<PYAPS_DIR>/configs/rvs_config.yaml",
        "--overwrite","True",
        "--caldir","<PYAPS_DATA>/CAL",
        "--catdir","<PYAPS_DATA>/CAT",
        "--RVS_CONFIG","<PYAPS_DIR>/configs/rvs_config.yaml",
        "--FERRE_EXE", "<PYAPS_DIR>/externals/ferre/bin/ferre.x",
        "--FERRE_TEMPLATES", "<PYAPS_DIR>/PyAPS_templates/templates_FR/"
    ]
    gal_runner(options=debug_LIFU)
