import os
import sys
import time
import numpy as np
from collections import OrderedDict
from astropy.io import fits
from astropy.table import Table
import multiprocessing as mp
from rvspecfit import fitter_ccf, vel_fit, spec_fit, utils
import matplotlib.pyplot as plt


def read_config_APS_RVS(fname=None):
    """
    APS customised version of the read_config from rvspecfit.utils
    """
    import yaml
    from rvspecfit import frozendict

    if fname is None:
        fname = "config.yaml"
    with open(fname) as fp:
        config = yaml.safe_load(fp)

        # Expand environment variables in paths
        def expand_paths(d):
            if isinstance(d, dict):
                d1 = {}
                for k, v in d.items():
                    if isinstance(v, str):
                        v = os.path.expanduser(os.path.expandvars(v))
                    d1[k] = expand_paths(v)
                return frozendict.frozendict(d1)
            else:
                return d

        return expand_paths(config)


def make_rvs_ifu_plot(wave, flux, error, model, badmask, title, fig_fname):
    """
    Create diagnostic plot for IFU RVS fitting
    """
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        flux_plot = flux.copy()
        model_plot = model.copy()
        flux_plot[badmask] = np.nan
        model_plot[badmask] = np.nan

        # Top panel
        ax1.plot(wave, flux_plot, "k-", linewidth=0.8, label="Data")
        ax1.plot(wave, model_plot, "r-", alpha=0.7, linewidth=0.8, label="Model")
        ax1.set_xlabel("Wavelength [Å]")
        ax1.set_ylabel("Flux [1e-18 erg/s/cm²/Å]")
        ax1.set_title(title)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Residuals
        residuals = flux_plot - model_plot
        ax2.plot(wave, residuals, "b-", linewidth=0.5, alpha=0.7)
        ax2.axhline(0, color="k", linestyle="--", alpha=0.5)
        ax2.set_xlabel("Wavelength [Å]")
        ax2.set_ylabel("Residuals")
        ax2.grid(True, alpha=0.3)

        rms = np.nanstd(residuals)
        ax2.text(0.02, 0.95, f"RMS = {rms:.3f}", transform=ax2.transAxes)

        plt.tight_layout()
        plt.savefig(fig_fname, dpi=100, bbox_inches="tight")
        plt.close()

    except Exception as e:
        print(f"Plot generation failed: {e}")
        if plt.get_fignums():
            plt.close("all")


def process_single_bin(
    bin_table, config, bin_idx, figdir=None, headname="", configs_gal=None, lsf=None,
    bin_to_bucket=None, lsf_by_bucket=None,
):
    """
    Process a single bin through RVSPECFIT

    `bin_to_bucket`/`lsf_by_bucket` (both optional, default `None`): opt-in
    per-bin resolution (see `aps_ifu_spaxel_contrib.bucket_lsf_curves` and
    `IFUExGalPPXF.runModule_PPXF`'s own matching parameters, same
    contract) -- `lsf_by_bucket` holds already-evaluated FWHM arrays (one
    per bucket, matching `lsf`'s own convention here -- this module
    consumes a plain array, not a callable, unlike the ExGal fit modules)
    keyed by bucket id. When both are `None`, `bin_idx`'s own bucket is
    looked up and used instead of the single shared `lsf` -- identical
    behaviour to before these parameters existed.
    """
    if bin_to_bucket is not None and lsf_by_bucket is not None:
        lsf = lsf_by_bucket[bin_to_bucket[bin_idx]]
    from rvspecfit import spec_fit, fitter_ccf, vel_fit
    import PyAPS.aps_constants as aps_constants

    options = {"npoly": 19}
    large_error = aps_constants.large_error

    try:
        row = bin_table[bin_idx]  # row = FITS_record
        colnames = bin_table.columns.names

        bin_id = row["BIN_ID"] if "BIN_ID" in colnames else bin_idx

        # Extract spectra depending on available columns
        if "SPEC_LIN_WAVE" in colnames:
            wave = row["SPEC_LIN_WAVE"]
            flux = row["SPEC_LIN_FLUX"]
            error = row["SPEC_LIN_ERR"]
        elif "SPEC_LIN_LAMBDA" in colnames:
            wave = row["SPEC_LIN_LAMBDA"]
            flux = row["SPEC_LIN"]
            error = row["ERR_LIN"]
        else:
            wave = row["LAM"]
            flux = row["SPEC"]
            error = row["ESPEC"]

        # Loop through original setups and wavelength ranges
        specdata = []
        wave_all, flux_all, error_all, badmask_all = [], [], [], []

        for ns, s in enumerate(configs_gal["orig_setups"]):
            wlrange = configs_gal["orig_wlranges"][ns]
            wmin, wmax = wlrange

            mask = (wave >= wmin) & (wave <= wmax)
            wave_sel = wave[mask]
            flux_sel = flux[mask]
            error_sel = error[mask]

            if len(wave_sel) == 0:
                print(f"    Bin {bin_id}, setup {s}: No data in {wlrange}")
                continue

            badmask = (
                (error_sel <= 0)
                | np.isnan(flux_sel)
                | np.isnan(error_sel)
                | np.isinf(flux_sel)
                | np.isinf(error_sel)
            )
            if np.sum(~badmask) < 100:
                print(f"    Bin {bin_id}, setup {s}: Too many bad pixels, skipping")
                continue

            error_clean = error_sel.copy()
            error_clean[badmask] = large_error
            flux_clean = flux_sel.copy()
            if np.any(badmask) and np.any(~badmask):
                good_median = np.nanmedian(flux_sel[~badmask])
                flux_clean[badmask] = good_median

            if np.any(~badmask):
                snr = np.nanmedian(flux_clean[~badmask] / error_clean[~badmask])
            else:
                snr = 0.0

            if snr < 1.0:
                print(f"    Bin {bin_id}, setup {s}: SNR={snr:.2f} too low, skipping")
                continue

            print(f"    Processing BIN_ID {bin_id}, setup {s}: SNR={snr:.2f}")

            specdata.append(
                spec_fit.SpecData(
                    f"{configs_gal['targs_mode']}_{s}",
                    wave_sel,
                    flux_clean,
                    error_clean,
                    badmask=badmask,
                )
            )

            wave_all.append(wave_sel)
            flux_all.append(flux_sel)
            error_all.append(error_sel)
            badmask_all.append(badmask)

        if not specdata:
            print(f"    Bin {bin_id}: no valid arms, skipping")
            return None

        # Run fitting
        res = fitter_ccf.fit(specdata, config)
        paramDict0 = res["best_par"]
        fixParam = []

        if res.get("best_vsini") is not None:
            paramDict0["vsini"] = min(
                max(res["best_vsini"], config.get("min_vsini", 0.0)),
                config.get("max_vsini", 500.0),
            )

        res1 = vel_fit.process(
            specdata, paramDict0, fixParam=fixParam, config=config, options=options
        )

        chisq_cont = spec_fit.get_chisq_continuum(specdata, options=options)[
            "chisq_array"
        ][0]

        # Concatenate arms for storage/plotting
        wave_all = np.concatenate(wave_all)
        flux_all = np.concatenate(flux_all)
        error_all = np.concatenate(error_all)
        badmask_all = np.concatenate(badmask_all)
        model_all = np.concatenate(res1["yfit"])

        result = {
            "BIN_ID": bin_id,
            "VRAD": res1["vel"],
            "VRAD_ERR": res1["vel_err"],
            "SKEWNESS_RVS": res1["vel_skewness"],
            "KURTOSIS_RVS": res1["vel_kurtosis"],
            "LOGG_RVS": res1["param"]["logg"],
            "TEFF_RVS": res1["param"]["teff"],
            "ALPHA_RVS": res1["param"]["alpha"],
            "FEH_RVS": res1["param"]["feh"],
            "LOGG_ERR_RVS": res1["param_err"]["logg"],
            "TEFF_ERR_RVS": res1["param_err"]["teff"],
            "ALPHA_ERR_RVS": res1["param_err"]["alpha"],
            "FEH_ERR_RVS": res1["param_err"]["feh"],
            "VSINI_RVS": res1.get("vsini", 0.0),
            "SNR_RVS": snr,
            "CHISQ_TOT_RVS": sum(res1["chisq_array"]),
            "CHISQ_C": chisq_cont,
            "LAMBDA_RVS": wave_all,
            "FLUX_RVS": flux_all,
            "ERROR_RVS": error_all,
            "MODEL_RVS": model_all,
        }

        for key in ["APS_ID", "TARGID", "CNAME", "X", "Y"]:
            if key in colnames:
                result[key] = row[key]

        if figdir is not None:
            fig_fname = os.path.join(figdir, f"RVS_{headname}_BIN{bin_id:04d}.png")
            title = (
                f"BIN {bin_id}: logg={res1['param']['logg']:.1f} "
                f"teff={res1['param']['teff']:.0f} "
                f"[Fe/H]={res1['param']['feh']:.2f} "
                f"[α/Fe]={res1['param']['alpha']:.2f} "
                f"Vrad={res1['vel']:.1f}±{res1['vel_err']:.1f} km/s"
            )
            make_rvs_ifu_plot(
                wave_all, flux_all, error_all, model_all, badmask_all, title, fig_fname
            )

        return result

    except Exception as e:
        print(f"    Bin {bin_idx}: Processing failed - {e}")
        return None


def proc_rvs_ifu(
    headname,
    outpath,
    rvs_config,
    figdir=None,
    nthreads=1,
    configs_gal=None,
    overwrite=False,
    lsf=None,
    bin_to_bucket=None,
    lsf_by_bucket=None,
):
    """
    Main function to process IFU binned spectra through RVS.

    `bin_to_bucket`/`lsf_by_bucket`: opt-in per-bin resolution, see
    `process_single_bin`'s own docstring for the full contract.
    """
    binspec_file = os.path.join(outpath, f"{headname}_BINSpectra_linear.fits")

    if not os.path.exists(binspec_file):
        print(f"ERROR: Input file not found: {binspec_file}")
        return False

    rvs_param_out = os.path.join(outpath, f"rvs_{headname}.fits")
    rvs_spec_out = os.path.join(outpath, f"rvsspec_{headname}.fits")

    if not overwrite and os.path.exists(rvs_param_out):
        print(f"Output exists and overwrite=False: {rvs_param_out}")
        return True

    print(f"Processing binned spectra: {binspec_file}")

    if isinstance(rvs_config, str):
        if not os.path.exists(rvs_config):
            print(f"ERROR: Config file not found: {rvs_config}")
            return False
        config = read_config_APS_RVS(rvs_config)
    else:
        config = rvs_config

    try:
        with fits.open(binspec_file) as hdul:
            bin_table = hdul["BIN_SPECTRA"].data
            bin_header = hdul["BIN_SPECTRA"].header
            metadata = dict(bin_header)
    except Exception as e:
        print(f"ERROR reading input file: {e}")
        return False

    n_bins = len(bin_table)
    print(f"Found {n_bins} bins to process")

    results = []

    if nthreads > 1:
        print(f"Using {nthreads} threads for parallel processing")
        pool = mp.Pool(nthreads)
        async_results = [
            pool.apply_async(
                process_single_bin,
                (bin_table, config, i_bin, figdir, headname, configs_gal, lsf,
                 bin_to_bucket, lsf_by_bucket),
            )
            for i_bin in range(n_bins)
        ]
        pool.close()
        pool.join()

        for async_res in async_results:
            res = async_res.get()
            if res is not None:
                results.append(res)
    else:
        print("Using serial processing")
        for i_bin in range(n_bins):
            res = process_single_bin(
                bin_table, config, i_bin, figdir, headname, configs_gal, lsf,
                bin_to_bucket=bin_to_bucket, lsf_by_bucket=lsf_by_bucket,
            )
            if res is not None:
                results.append(res)

    if len(results) == 0:
        print("ERROR: No valid results obtained from RVS processing")
        return False

    print(f"Successfully processed {len(results)}/{n_bins} bins")

    # Collect keys
    all_keys = set()
    for res in results:
        all_keys.update(res.keys())

    array_cols = ["LAMBDA_RVS", "FLUX_RVS", "ERROR_RVS", "MODEL_RVS"]
    scalar_cols = [k for k in all_keys if k not in array_cols]

    outdict = OrderedDict()
    for col in scalar_cols:
        outdict[col] = []
    for col in array_cols:
        outdict[col] = []

    for res in results:
        for col in scalar_cols:
            outdict[col].append(res.get(col, np.nan))
        for col in array_cols:
            outdict[col].append(res.get(col, None))

    outtab = Table(outdict)

    unit_map = {
        "VRAD": "km/s",
        "VRAD_ERR": "km/s",
        "SKEWNESS_RVS": "km/s",
        "KURTOSIS_RVS": "km/s",
        "VSINI_RVS": "km/s",
        "TEFF_RVS": "K",
        "TEFF_ERR_RVS": "K",
        "LOGG_RVS": "dex",
        "FEH_RVS": "dex",
        "ALPHA_RVS": "dex",
        "LOGG_ERR_RVS": "dex",
        "FEH_ERR_RVS": "dex",
        "ALPHA_ERR_RVS": "dex",
        "LAMBDA_RVS": "Angstrom",
        "FLUX_RVS": "1e-18 erg/(s cm**2 Angstrom)",
        "ERROR_RVS": "1e-18 erg/(s cm**2 Angstrom)",
        "MODEL_RVS": "1e-18 erg/(s cm**2 Angstrom)",
    }

    for col, unit in unit_map.items():
        if col in outtab.colnames:
            outtab[col].unit = unit

    import PyAPS

    outtab.meta["EXTNAME"] = "RVS_IFU_TABLE"
    outtab.meta["APSVERS"] = (PyAPS.__version__, "APS version")
    outtab.meta["BINSPEC"] = (
        os.path.basename(binspec_file),
        "Input binned spectra file",
    )
    outtab.meta["NBINS"] = (n_bins, "Total number of bins in input")
    outtab.meta["NPROC"] = (len(results), "Number of successfully processed bins")
    outtab.meta["CONFIG"] = (
        rvs_config if isinstance(rvs_config, str) else "dict",
        "RVS config source",
    )

    for key, value in metadata.items():
        if key not in ["EXTNAME", "XTENSION", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2"]:
            outtab.meta[key] = value

    if "BIN_ID" in outtab.colnames:
        outtab.sort("BIN_ID")

    try:
        # Parameter file
        param_table = outtab.copy()
        for col in array_cols:
            if col in param_table.colnames:
                param_table.remove_column(col)

        hx = fits.HDUList()
        hx.append(fits.PrimaryHDU())
        hx.append(fits.convenience.table_to_hdu(param_table))
        hx.writeto(rvs_param_out, overwrite=True)
        print(f"✓ RVS parameters written to: {rvs_param_out}")

        # Spectra file
        hx_spec = fits.HDUList()
        hx_spec.append(fits.PrimaryHDU())
        hx_spec.append(fits.convenience.table_to_hdu(outtab))
        hx_spec.writeto(rvs_spec_out, overwrite=True)
        print(f"✓ RVS spectra written to: {rvs_spec_out}")

    except Exception as e:
        print(f"ERROR writing output files: {e}")
        return False

    return True


def run_rvs_for_ifu_gal(
    headname,
    outpath,
    rvs_config,
    figdir=None,
    nthreads=1,
    configs_gal=None,
    overwrite=False,
    lsf=None,
    bin_to_bucket=None,
    lsf_by_bucket=None,
):
    success = proc_rvs_ifu(
        headname,
        outpath,
        rvs_config,
        figdir=figdir,
        nthreads=nthreads,
        configs_gal=configs_gal,
        overwrite=overwrite,
        lsf=lsf,
        bin_to_bucket=bin_to_bucket,
        lsf_by_bucket=lsf_by_bucket,
    )
    if success:
        print("RVS processing completed successfully")
    else:
        print("RVS processing failed")
    return success
