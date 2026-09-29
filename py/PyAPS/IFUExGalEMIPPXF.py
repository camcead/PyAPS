import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from astropy.io import fits
from multiprocessing import Queue, Process
from scipy import interpolate

import time
import logging

import sys

import PyAPS
from PyAPS import ExGalutil
from PyAPS import IFUExGalPrepare as IFUExGalPrepare
from PyAPS import ExGalPrepare
from PyAPS.apsPlot.emi_bin import save_ppxf_emi_plot

from PyAPS import aps_constants
from astropy import units

from ppxf.ppxf import ppxf
from ppxf import ppxf_util
from ppxf import __version__ as PPXF_VERSION

import warnings
import traceback
APSVERS = PyAPS.__version__

# PHYSICAL CONSTANTS
C = 299792.458  # km/s
# PHYSICAL CONSTANTS
Clight = 299792.458  # km/s
"""
PURPOSE:
  This module executes the analysis of emi kinematics in the pipeline.
  It processes emission lines using pPXF to extract emi velocities and velocity dispersions,
  as well as emission line fluxes. It builds on the stellar kinematics results.
"""


def worker_stellar_fit_for_emi(inQueue, outQueue):
    """
    Worker process for preparing stellar fits on emi wavelength range.
    """
    for template_emi_range, bin_spectrum, noise_spectrum, lam_emi, lam_emi_template, velscale, velscale_ratio, \
        start_fixed, configs, error_limit, bin_id \
        in iter(inQueue.get, 'STOP'):

        try:
            # Create goodpixels mask
            goodpixels = np.arange(len(lam_emi))

            # Filter out high-error regions
            if noise_spectrum is not None:
                error_threshold = 0.95 * np.sqrt(error_limit)
                good_error_mask = noise_spectrum < error_threshold
                goodpixels = goodpixels[good_error_mask]

            if len(goodpixels) < 50:  # Minimum pixels needed
                print(f"    Warning: Only {len(goodpixels)} good pixels in bin {bin_id}")
                outQueue.put((bin_id, np.full(len(lam_emi), np.nan), start_fixed))
                continue

            # Run PPXF with stellar template only, fixed kinematics
            regularization = max(0.1, np.std(bin_spectrum[goodpixels]) * 1e-3)

            # Fix kinematics by using very tight bounds
            bounds_list = [
                [start_fixed[0]-1, start_fixed[0]+1],  # V ± 1 km/s
                [max(30, start_fixed[1]-5), start_fixed[1]+5]  # sigma with minimum 30 km/s
            ] + [[-0.01, 0.01]] * (len(start_fixed)-2) if len(start_fixed) > 2 else []

            # Validate bounds
            if len(bounds_list) >= 2:
                if bounds_list[0][0] >= bounds_list[0][1]:
                    bounds_list[0] = [start_fixed[0]-10, start_fixed[0]+10]
                if bounds_list[1][0] >= bounds_list[1][1]:
                    bounds_list[1] = [30, 300]

            pp = ppxf(template_emi_range, bin_spectrum, noise_spectrum, velscale, velscale_ratio = velscale_ratio,
                        start=start_fixed, goodpixels=goodpixels, plot=False, quiet=True,
                        moments=len(start_fixed), degree=configs.get('ADEG', 12),
                        mdegree=configs.get('MDEG', 0), lam=lam_emi, lam_temp=lam_emi_template,
                        bias=regularization,
                        bounds=bounds_list)


            # Store results with bin_id for proper ordering
            stellar_fit = pp.bestfit
            stellar_kinematics_fitted = pp.sol

            outQueue.put((bin_id, stellar_fit, stellar_kinematics_fitted))

        except Exception as e:
            print(f"    Error in bin {bin_id}: {str(e)}")
            outQueue.put((bin_id, np.full(len(lam_emi), np.nan), start_fixed))


def prepare_stellar_fit_for_emi_range_parallel(outdir, rootname, logLam_emi, logLam_emi_template, bin_data_emi,
                                             noise_emi, template_emi_range, velscale, velscale_ratio,
                                             configs, nbins, error_limit,
                                             stellar_kinematics=None, nthreads=1, debug=False,
                                             bin_to_bucket=None, template_by_bucket=None):
    """
    Re-run PPXF on the emi wavelength range with fixed stellar kinematics (parallel version).

    Parameters:
    -----------
    nthreads : int
        Number of threads for parallel processing
    bin_to_bucket, template_by_bucket : optional
        Opt-in per-bin resolution (see `aps_ifu_spaxel_contrib.
        bucket_lsf_curves`) -- `bin_to_bucket[i]` gives bin i's bucket id,
        `template_by_bucket[bucket]` that bucket's own resolution-matched
        stellar template. Both `None` (default): every bin uses the single
        `template_emi_range` exactly as before this parameter existed.

    All other parameters same as serial version

    Returns:
    --------
    stellar_fit_emi_range : array
        Stellar fit models for emi range [nbins, npix_emi]
    stellar_kinematics_used : array
        Stellar kinematics used for the fits [nbins, nmoments]
    """
    def _template_for_bin(i):
        if template_by_bucket is not None and bin_to_bucket is not None:
            return template_by_bucket[bin_to_bucket[i]]
        return template_emi_range

    # Load stellar kinematics if not provided
    if stellar_kinematics is None:
        try:
            hdu_ppxf = fits.open(outdir + rootname + '_ppxf.fits')
            ppxf_stellar = hdu_ppxf[1].data

            # Extract all moments used in stellar fit
            nmoments = configs.get('MOM', 4)
            stellar_kinematics = np.zeros((nbins, nmoments))

            stellar_kinematics[:, 0] = ppxf_stellar['V'][:nbins]
            stellar_kinematics[:, 1] = ppxf_stellar['SIGMA'][:nbins]

            if nmoments > 2 and 'H3' in ppxf_stellar.names:
                stellar_kinematics[:, 2] = ppxf_stellar['H3'][:nbins]
            if nmoments > 3 and 'H4' in ppxf_stellar.names:
                stellar_kinematics[:, 3] = ppxf_stellar['H4'][:nbins]
            if nmoments > 4 and 'H5' in ppxf_stellar.names:
                stellar_kinematics[:, 4] = ppxf_stellar['H5'][:nbins]
            if nmoments > 5 and 'H6' in ppxf_stellar.names:
                stellar_kinematics[:, 5] = ppxf_stellar['H6'][:nbins]

            hdu_ppxf.close()
            if debug:
                print(f"Loaded stellar kinematics for {nbins} bins")

        except Exception as e:
            ExGalutil.prettyOutput_Error(f"Error loading stellar kinematics: {e}")
            ExGalutil.prettyOutput_Warning("Using default stellar kinematics (V=0, sigma=150)")
            nmoments = configs.get('MOM', 4)
            stellar_kinematics = np.zeros((nbins, nmoments))
            # stellar_kinematics[:, 1] = 150.0  # Default sigma

    # Initialize output arrays
    npix_emi = len(logLam_emi)
    stellar_fit_emi_range = np.zeros((nbins, npix_emi))
    stellar_kinematics_output = stellar_kinematics.copy()

    lam_emi = np.exp(logLam_emi)  # Convert logLam_emi to linear scale
    lam_emi_template = np.exp(logLam_emi_template)  # Convert logLam_emi_template to linear scale

    if nthreads > 1:
        print(f"Running stellar fit preparation in parallel mode with {nthreads} threads")

        from multiprocessing import Queue, Process

        # Create Queues
        inQueue = Queue()
        outQueue = Queue()

        # Create worker processes
        ps = [Process(target=worker_stellar_fit_for_emi, args=(inQueue, outQueue))
              for _ in range(nthreads)]

        # Start worker processes
        for p in ps:
            p.start()

        # Fill the queue with jobs
        for i in range(nbins):
            inQueue.put((
                _template_for_bin(i),
                bin_data_emi[:, i],
                noise_emi[:, i],
                lam_emi,
                lam_emi_template,
                velscale,
                velscale_ratio,
                stellar_kinematics[i, :].copy(),
                configs,
                error_limit,
                i  # bin_id for proper ordering
            ))

        # Get results - IMPORTANT: results may come back in any order!
        results = [outQueue.get() for _ in range(nbins)]

        # Send stop signal to stop iteration
        for _ in range(nthreads):
            inQueue.put('STOP')

        # Stop processes
        for p in ps:
            p.join()

        # Sort results by bin_id to maintain proper order
        results.sort(key=lambda x: x[0])  # Sort by bin_id (first element)

        # Extract sorted results
        for i, (bin_id, stellar_fit, stellar_kin_fitted) in enumerate(results):
            if bin_id != i:
                ExGalutil.prettyOutput_Warning(f"Expected bin {i} but got bin {bin_id}")

            stellar_fit_emi_range[i, :] = stellar_fit
            stellar_kinematics_output[i, :len(stellar_kin_fitted)] = stellar_kin_fitted

    else:

        if debug:
            print("Running stellar fit preparation in serial mode")

        # Serial processing (original code)
        for i in range(nbins):
            try:
                if i % 50 == 0 or i < 5:  # Progress updates
                    print(f"  Processing bin {i+1}/{nbins}")

                # Get stellar kinematics for this bin
                start_fixed = stellar_kinematics[i, :].copy()

                # Create simple goodpixels mask
                goodpixels = np.arange(len(logLam_emi))

                # Filter out high-error regions
                if noise_emi is not None:
                    error_threshold = 0.95 * np.sqrt(error_limit)
                    good_error_mask = noise_emi[:, i] < error_threshold
                    goodpixels = goodpixels[good_error_mask]

                if len(goodpixels) < 50:  # Minimum pixels needed
                    ExGalutil.prettyOutput_Warning(f"    Warning: Only {len(goodpixels)} good pixels in bin {i}")
                    continue

                # Run PPXF with stellar template only, fixed kinematics
                regularization = max(0.1, np.std(bin_data_emi[goodpixels, i]) * 1e-3)

                pp = ppxf(_template_for_bin(i), bin_data_emi[:, i], noise_emi[:, i], velscale, velscale_ratio=velscale_ratio,
                         start=start_fixed, goodpixels=goodpixels, plot=False, quiet=True,
                         moments=len(start_fixed), degree=configs.get('ADEG', 12),
                         mdegree=configs.get('MDEG', 0), lam=lam_emi, lam_temp= lam_emi_template,
                         bias=regularization,
                         # Fix kinematics by using very tight bounds
                         bounds=[
                             [start_fixed[0]-1, start_fixed[0]+1],  # V ± 1 km/s
                             [max(30, start_fixed[1]-5), start_fixed[1]+5]  # sigma with minimum 30 km/s
                         ] + [[-0.01, 0.01]] * (len(start_fixed)-2)  # Higher moments ± 0.01
                         if len(start_fixed) > 2 else [
                             [start_fixed[0]-1, start_fixed[0]+1],
                             [max(30, start_fixed[1]-5), start_fixed[1]+5]
                         ])

                # Store the stellar fit model
                stellar_fit_emi_range[i, :] = pp.bestfit

                # Update stellar kinematics with the actual fitted values
                stellar_kinematics_output[i, :len(pp.sol)] = pp.sol

            except Exception as e:
                ExGalutil.prettyOutput_Error(f"    Error in bin {i}: {str(e)}")
                # Fill with NaN for failed fits
                stellar_fit_emi_range[i, :] = np.nan
                continue

    if debug:
        print(f"Successfully fitted {np.sum(~np.isnan(stellar_fit_emi_range[:, 0]))} out of {nbins} bins")

    return stellar_fit_emi_range, stellar_kinematics_output




def workerPPXF_emi(inQueue, outQueue):
    """
    Worker process for emi kinematics analysis with proper error handling.
    """
    for item in iter(inQueue.get, 'STOP'):
        try:
            # Unpack parameters
            (rootname,template, emi_template, emi_template_unbroadened, bin_data, noise, velscale, start, goodpixels_ppxf,
             nmoments, adeg, mdeg, velscale_ratio, error_limit, nsims, nbins, i,
             emission_setup, stellar_bestfit, line_names, tie_settings,
             line_wavelengths, lam_emi, lam_emi_template, figdir, diag_plots, group_order, scaling_factor) = item

            # Run the actual emi fitting
            sol, bestfit, emi_bestfit, emission_lines, mc_results, formal_error, goodpixels_final = \
                run_ppxf_emi(rootname, template, emi_template,emi_template_unbroadened, bin_data, noise, velscale, start,
                           goodpixels_ppxf, error_limit, nmoments, adeg, mdeg,
                           velscale_ratio, nsims, nbins, i, emission_setup,
                           stellar_bestfit, line_names, tie_settings, line_wavelengths,
                           lam_emi, lam_emi_template, figdir, diag_plots, group_order, scaling_factor)

            # Put successful result
            outQueue.put((i, sol, bestfit, emi_bestfit, emission_lines, mc_results, formal_error, goodpixels_final))

        except Exception as e:
            # Log the error
            print(f"Error in worker for bin {i if 'i' in locals() else 'unknown'}: {str(e)}")
            import traceback
            traceback.print_exc()

            # Create dummy/failed results with proper structure
            try:
                # Get dimensions from the input if possible
                npix = len(bin_data) if 'bin_data' in locals() else 1000  # fallback
                nlines = len(line_names) if 'line_names' in locals() else 10  # fallback
                nmoments_emi = emission_setup.get('emi_moments', 2) if 'emission_setup' in locals() else 2
                nmoments_stellar = nmoments if 'nmoments' in locals() else 4
                bin_id = i if 'i' in locals() else -1

                # Create failed result structure
                failed_sol = {
                    "stellar": np.full(nmoments_stellar, np.nan),
                    "emi": np.full(nmoments_emi, np.nan)
                }
                failed_bestfit = np.full(npix, np.nan)
                failed_emi_bestfit = np.full(npix, np.nan)
                failed_emission_lines = np.full(nlines, np.nan)
                failed_mc_results = {
                    "stellar": np.full(nmoments_stellar, np.nan),
                    "emi": np.full(nmoments_emi, np.nan),
                    "fluxes": np.full(nlines, np.nan),
                    "amplitudes": np.full(nlines, np.nan),
                    "amplitude_errors": np.full(nlines, np.nan),
                    "aon": np.full(nlines, np.nan)
                }
                failed_formal_error = {
                    "stellar": np.full(nmoments_stellar, np.nan),
                    "emi": np.full(nmoments_emi, np.nan)
                }
                failed_goodpixels = np.array([])

                # Put failed result - THIS IS CRUCIAL
                outQueue.put((bin_id, failed_sol, failed_bestfit, failed_emi_bestfit,
                            failed_emission_lines, failed_mc_results, failed_formal_error, failed_goodpixels))

            except Exception as e2:
                # If even creating dummy results fails, put minimal result
                print(f"Critical error creating dummy results: {str(e2)}")
                outQueue.put((-1, {}, np.array([]), np.array([]), np.array([]), {}, {}, np.array([])))


def _emi_fit_nan_result(nmoments, emission_setup, line_names, log_bin_data, goodpixels):
    """The NaN-filled failure result for one bin's emi fit -- same shape
    run_ppxf_emi returns for any other failure. Mirrors
    MOSExGalEMIPPXF._emi_fit_nan_result (same convention, same
    fields); this version additionally carries goodpixels through
    unchanged since run_ppxf_emi's own signature returns it as its 7th
    element (MOS's run_ppxf_emi_mos does not). Factored out so the
    early short-circuit below and this function's own except-clause
    can't drift out of sync with each other."""
    n_emi_moments = emission_setup.get('emi_moments', 2)
    n_lines = len(line_names)
    return ({"stellar": np.full(nmoments, np.nan), "emi": np.full(n_emi_moments, np.nan)},
            np.full_like(log_bin_data, np.nan),
            np.full_like(log_bin_data, np.nan),
            np.full(n_lines, np.nan),
            {"stellar": np.full(nmoments, np.nan), "emi": np.full(n_emi_moments, np.nan),
             "fluxes": np.full(n_lines, np.nan), "amplitudes": np.full(n_lines, np.nan),
             "amplitude_errors": np.full(n_lines, np.nan), "aon": np.full(n_lines, np.nan),
             "line_velocities": np.full(n_lines, np.nan), "line_sigmas": np.full(n_lines, np.nan),
             "line_velocity_errors": np.full(n_lines, np.nan), "line_sigma_errors": np.full(n_lines, np.nan)},
            {"stellar": np.full(nmoments, np.nan), "emi": np.full(n_emi_moments, np.nan),
             "line_velocities": np.full(n_lines, np.nan), "line_sigmas": np.full(n_lines, np.nan),
             "fluxes": np.full(n_lines, np.nan), "amplitudes": np.full(n_lines, np.nan)},
            goodpixels)


def run_ppxf_emi(rootname, template, emi_template, emi_template_unbroadened, log_bin_data, log_bin_error, velscale, start, goodpixels,
                error_limit, nmoments, adeg, mdeg, velscale_ratio, nsims, nbins, i,
                emission_setup, stellar_bestfit, line_names, tie_settings, line_wavelengths, lam_emi, lam_emi_template, figdir, diag_plots, group_order, scaling_factor):
    """
    FIXED VERSION: Properly handles multi-component emi kinematics.
    Now extracts V and Sigma for each emi component/line, not just the first component.

    Calls the penalised Pixel-Fitting routine to determine the emi kinematics.
    This version includes both stellar template and emission line template with
    sophisticated tying based on tie_settings configuration.

    Parameters:
    -----------
    tie_settings : dict, optional
        Dictionary containing line tying configuration
    line_wavelengths : dict, optional
        Dictionary mapping line names to rest wavelengths for AON calculation
    """
    ExGalutil.printProgress(i, nbins, barLength=50)

    # Defensive consistency with MOSExGalEMIPPXF.run_ppxf_emi_mos: neither
    # current caller of this function currently produces a non-array
    # template/emi_template (both the serial and parallel bin loops in
    # runModule_EMIPPXF already skip a bin entirely -- with a clean
    # warning, no crash -- when it has zero good pixels or an unusable
    # template, before this function is ever called), so this guard is not
    # known to fire today. It's here so that if a future caller ever
    # adopts the "return a NaN sentinel on prep failure" convention
    # MOSExGalEMIPPXF.preparePPXF_emi_mos uses, this function skips
    # cleanly (one INFO line) instead of crashing on template.shape /
    # emi_template.shape with a confusing AttributeError, same as the fix
    # applied there.
    if not isinstance(template, np.ndarray) or not isinstance(emi_template, np.ndarray):
        print(f"INFO: bin {i}: skipping emi fit -- template/emi_template is not a valid array")
        return _emi_fit_nan_result(nmoments, emission_setup, line_names, log_bin_data, goodpixels)

    # Use the same tie settings and component building logic as MOS
    emi_component = build_emi_component_tying(line_names, tie_settings, group_order=group_order, debug=True if i==0 else False)

    try:
        # Create start array from stellar kinematics or defaults
        if start is not None:
            emi_start = start.copy()
        else:
            start = np.zeros(max(nmoments, 2))
            start[0] = 0.0  # Default velocity
            start[1] = 150.0  # Default sigma
            emi_start = start.copy()

        # Filter out regions with high errors (e.g., CCD gaps)
        goodpix_idx = np.ravel(np.where(log_bin_error[goodpixels] < 0.95 * np.sqrt(error_limit)))
        if len(goodpix_idx) > 0:
            goodpixels = goodpixels[goodpix_idx]

        # PPXF emi fit - includes both stellar and emission template
        component = np.zeros(template.shape[1] + emi_template.shape[1], dtype=int)
        component[template.shape[1]:] = 1  # Mark emi template with component=1

        # Setup the emi fitting parameters
        moments = [nmoments, emission_setup.get('emi_moments', 2)]  # emi usually has fewer moments

        # Convert to linear wavelength
        lam_emi = np.exp(lam_emi) if lam_emi[0] < 10 else lam_emi  # Handle if already linear
        lam_emi_template = np.exp(lam_emi_template) if lam_emi_template[0] < 10 else lam_emi_template

        # Include stellar continuum from previous fit if available
        if stellar_bestfit is not None and emission_setup.get('use_stellar_continuum', True):
            if emission_setup.get('fix_stellar_continuum', False):
                # Option 1: Use stellar bestfit as a fixed component
                continuum_subtracted = log_bin_data - stellar_bestfit


                # CRITICAL FIX: Clean the continuum-subtracted spectrum
                # Check for and handle non-finite values
                bad_pixels = ~np.isfinite(continuum_subtracted)
                if np.any(bad_pixels):
                    n_bad = np.sum(bad_pixels)
                    print(f"    Warning: {n_bad} non-finite pixels in continuum-subtracted spectrum for bin {i}")

                    # Option 1: Set bad pixels to zero
                    continuum_subtracted[bad_pixels] = 0.0

                    # Remove these bad pixels from goodpixels array
                    # Only keep goodpixels that are not in the bad_pixels mask
                    goodpixels_mask = np.ones(len(continuum_subtracted), dtype=bool)
                    goodpixels_mask[:] = False
                    goodpixels_mask[goodpixels] = True
                    goodpixels_mask[bad_pixels] = False
                    goodpixels = np.where(goodpixels_mask)[0]

                    if len(goodpixels) < 50:
                        print(f"    Warning: Only {len(goodpixels)} good pixels remaining after cleaning")

                # Additional safety check
                continuum_subtracted = np.nan_to_num(continuum_subtracted, nan=0.0, posinf=0.0, neginf=0.0)

                emi_start_simple = emi_start.copy() if len(emi_start) >= 2 else [0.0, 150.0]
                if len(emi_start_simple) > emission_setup.get('emi_moments', 2):
                    emi_start_simple = emi_start_simple[:emission_setup.get('emi_moments', 2)]

                # Get bounds and bias
                emi_bounds = get_emi_bounds(emi_start_simple, stellar_present=True)
                # Add validation
                if emi_bounds and len(emi_bounds) >= 2:
                    # Validate velocity bounds
                    if emi_bounds[0][0] >= emi_bounds[0][1]:
                        print(f"    Warning: Invalid velocity bounds: {emi_bounds[0]}")
                        emi_bounds[0] = [-500, 500]

                    # Validate sigma bounds
                    if emi_bounds[1][0] >= emi_bounds[1][1]:
                        print(f"    Warning: Invalid sigma bounds: {emi_bounds[1]}")
                        emi_bounds[1] = [25, 400]

                bias_val = get_bias_value(continuum_subtracted, goodpixels, stellar_present=True)


                # For emi-only fitting with tying
                if emi_component is not None:
                    n_emi_components = len(np.unique(emi_component))
                    emi_start_multi = [emi_start_simple.copy() for _ in range(n_emi_components)]
                    if n_emi_components == 1:
                        emi_start_multi = emi_start_simple.copy()
                    emi_moments_multi = [emission_setup.get('emi_moments', 2)] * n_emi_components

                    # For multi-component, replicate bounds
                    if n_emi_components > 1:
                        multi_bounds = [emi_bounds.copy() for _ in range(n_emi_components)]
                        # Validate each component's bounds
                        for comp_idx, comp_bounds in enumerate(multi_bounds):
                            if len(comp_bounds) >= 2:
                                if comp_bounds[0][0] >= comp_bounds[0][1]:
                                    comp_bounds[0] = [-500, 500]
                                if comp_bounds[1][0] >= comp_bounds[1][1]:
                                    comp_bounds[1] = [25, 400]
                    else:
                        multi_bounds = emi_bounds

                    pp = ppxf(emi_template_unbroadened, continuum_subtracted, log_bin_error, velscale,
                                start=emi_start_multi, goodpixels=goodpixels, plot=False,
                                quiet=True, moments=emi_moments_multi,
                                bounds=multi_bounds,
                                degree=-1, mdegree=-1,
                                velscale_ratio=velscale_ratio, lam=lam_emi, lam_temp=lam_emi_template,
                                component=emi_component,
                                bias=bias_val)
                else:
                    pp = ppxf(emi_template_unbroadened, continuum_subtracted, log_bin_error, velscale,
                                start=emi_start_simple, goodpixels=goodpixels, plot=False,
                                quiet=True, moments=emission_setup.get('emi_moments', 2),
                                bounds=emi_bounds,
                                degree=-1, mdegree=-1,
                                velscale_ratio=velscale_ratio, lam=lam_emi, lam_temp=lam_emi_template,
                                bias=bias_val)

                # FIXED: Extract ALL emi component solutions
                stellar_sol = emi_start.copy() if len(emi_start) >= nmoments else np.zeros(nmoments)

                # NEW: Handle multi-component emi solutions properly
                if emi_component is not None and hasattr(pp, 'sol') and isinstance(pp.sol, list):
                    # Multi-component case: pp.sol is a list of [emi_comp1_sol, emi_comp2_sol, ...]
                    emi_solutions = pp.sol  # All emi component solutions
                    n_emi_components = len(emi_solutions)
                else:
                    # Single component case
                    emi_solutions = [pp.sol] if hasattr(pp, 'sol') else [np.zeros(emission_setup.get('emi_moments', 2))]
                    n_emi_components = 1

                # set the bestfit to zero for all bad pixels
                pp.bestfit[np.setdiff1d(np.arange(len(pp.bestfit)), goodpixels)] = 0.0
                # stellar_bestfit[np.setdiff1d(np.arange(len(pp.bestfit)), goodpixels)] = 0.0
                combined_bestfit = stellar_bestfit + pp.bestfit
                emi_template_weighted = pp.bestfit

            else:
                # Option 2: Re-fit both stellar and emi simultaneously
                all_template = np.column_stack([template, emi_template])

                stellar_start = emi_start.copy() if len(emi_start) >= nmoments else np.zeros(nmoments)
                emi_start_component = emi_start.copy() if len(emi_start) >= 2 else [0.0, 150.0]
                if len(emi_start_component) > emission_setup.get('emi_moments', 2):
                    emi_start_component = emi_start_component[:emission_setup.get('emi_moments', 2)]

                # Build start parameters and moments for multi-component fitting
                if emi_component is not None:
                    n_emi_components = len(np.unique(emi_component))
                    start_multicomponent = [stellar_start] + [emi_start_component.copy() for _ in range(n_emi_components)]
                    if n_emi_components == 1:
                        start_multicomponent = [stellar_start, emi_start_component]
                    moments_multicomponent = [nmoments] + [emission_setup.get('emi_moments', 2)] * n_emi_components

                    full_component = np.concatenate([
                        np.zeros(template.shape[1], dtype=int),
                        emi_component + 1
                    ])

                    # Build bounds for stellar + emi components
                    stellar_bounds = [
                        [stellar_start[0] - 200, stellar_start[0] + 200],  # V bounds
                        [max(30, stellar_start[1] * 0.7), stellar_start[1] * 1.5]  # Sigma bounds
                    ] + [[-0.2, 0.2]] * (len(stellar_start) - 2)

                    emi_bounds = get_emi_bounds(emi_start_component, stellar_present=True)

                    # Validate emi bounds
                    if emi_bounds and len(emi_bounds) >= 2:
                        if emi_bounds[0][0] >= emi_bounds[0][1]:
                            print(f"    Warning: Invalid velocity bounds in simultaneous fit: {emi_bounds[0]}")
                            emi_bounds[0] = [-500, 500]
                        if emi_bounds[1][0] >= emi_bounds[1][1]:
                            print(f"    Warning: Invalid sigma bounds in simultaneous fit: {emi_bounds[1]}")
                            emi_bounds[1] = [25, 400]

                    if n_emi_components == 1:
                        multi_bounds = [stellar_bounds, emi_bounds]
                    else:
                        multi_bounds = [stellar_bounds] + [emi_bounds.copy() for _ in range(n_emi_components)]

                else:
                    start_multicomponent = [stellar_start, emi_start_component]
                    moments_multicomponent = moments
                    full_component = component
                    n_emi_components = 1

                    # Bounds for stellar + single emi component
                    stellar_bounds = [
                        [stellar_start[0] - 200, stellar_start[0] + 200],
                        [max(30, stellar_start[1] * 0.7), stellar_start[1] * 1.5]
                    ] + [[-0.2, 0.2]] * (len(stellar_start) - 2)

                    emi_bounds = get_emi_bounds(emi_start_component, stellar_present=True)
                    multi_bounds = [stellar_bounds, emi_bounds]

                bias_val = get_bias_value(log_bin_data, goodpixels, stellar_present=True)

                pp = ppxf(all_template, log_bin_data, log_bin_error, velscale,
                            start=start_multicomponent, goodpixels=goodpixels, plot=False,
                            quiet=True, moments=moments_multicomponent,
                            bounds=multi_bounds,
                            degree=adeg, mdegree=mdeg,
                            velscale_ratio=velscale_ratio, lam=lam_emi, lam_temp=lam_emi_template,
                            component=full_component,
                            bias=bias_val)

                # FIXED: Extract results properly for multi-component case
                stellar_sol = pp.sol[0] if isinstance(pp.sol, list) and len(pp.sol) > 0 else np.zeros(nmoments)

                # NEW: Extract ALL emi component solutions
                if isinstance(pp.sol, list) and len(pp.sol) > 1:
                    emi_solutions = pp.sol[1:]  # All emi components (excluding stellar)
                    n_emi_components = len(emi_solutions)
                else:
                    emi_solutions = [np.zeros(emission_setup.get('emi_moments', 2))]
                    n_emi_components = 1

                # set the bestfit to zero for all bad pixels
                pp.bestfit[np.setdiff1d(np.arange(len(pp.bestfit)), goodpixels)] = 0.0
                combined_bestfit = pp.bestfit

                # Extract emi bestfit (emission lines only)
                if hasattr(pp, 'weights'):
                    weights = pp.weights
                    emi_weights = weights[template.shape[1]:]
                    emi_template_weighted = np.zeros_like(log_bin_data)

                    for j in range(emi_template.shape[1]):
                        if j < len(emi_weights):
                            emi_template_weighted += emi_template[:, j] * emi_weights[j]
                else:
                    emi_template_weighted = np.zeros_like(log_bin_data)
        else:
            # No stellar continuum available - fit emi only
            emi_start_simple = emi_start.copy() if len(emi_start) >= 2 else [0.0, 150.0]
            if len(emi_start_simple) > emission_setup.get('emi_moments', 2):
                emi_start_simple = emi_start_simple[:emission_setup.get('emi_moments', 2)]

            # Get bounds and bias for emi-only fit
            emi_bounds = get_emi_bounds(emi_start_simple, stellar_present=False)

            # Validate emi-only bounds
            if emi_bounds and len(emi_bounds) >= 2:
                if emi_bounds[0][0] >= emi_bounds[0][1]:
                    print(f"    Warning: Invalid velocity bounds in emi-only fit: {emi_bounds[0]}")
                    emi_bounds[0] = [-500, 500]
                if emi_bounds[1][0] >= emi_bounds[1][1]:
                    print(f"    Warning: Invalid sigma bounds in emi-only fit: {emi_bounds[1]}")
                    emi_bounds[1] = [25, 400]
            bias_val = get_bias_value(log_bin_data, goodpixels, stellar_present=False)

            # Apply tying for emi-only fit
            if emi_component is not None:
                n_emi_components = len(np.unique(emi_component))
                emi_start_multi = [emi_start_simple.copy() for _ in range(n_emi_components)]
                if n_emi_components == 1:
                    emi_start_multi = emi_start_simple.copy()
                emi_moments_multi = [emission_setup.get('emi_moments', 2)] * n_emi_components

                # Replicate bounds for multiple components
                if n_emi_components > 1:
                    multi_bounds = [emi_bounds.copy() for _ in range(n_emi_components)]
                else:
                    multi_bounds = emi_bounds

                pp = ppxf(emi_template_unbroadened, log_bin_data, log_bin_error, velscale,
                            start=emi_start_multi, goodpixels=goodpixels, plot=False,
                            quiet=True, moments=emi_moments_multi,
                            bounds=multi_bounds,
                            degree=-1, mdegree=-1,
                            velscale_ratio=velscale_ratio, lam=lam_emi, lam_temp=lam_emi_template,
                            component=emi_component,
                            bias=bias_val)
            else:
                n_emi_components = 1
                pp = ppxf(emi_template_unbroadened, log_bin_data, log_bin_error, velscale,
                            start=emi_start_simple, goodpixels=goodpixels, plot=False,
                            quiet=True, moments=emission_setup.get('emi_moments', 2),
                            bounds=emi_bounds,
                            degree=-1, mdegree=-1,
                            velscale_ratio=velscale_ratio, lam=lam_emi, lam_temp=lam_emi_template,
                            bias=bias_val)

            # Results for emi-only fit
            stellar_sol = np.full(nmoments, np.nan)

            # NEW: Handle multi-component emi-only solutions
            if emi_component is not None and hasattr(pp, 'sol') and isinstance(pp.sol, list):
                emi_solutions = pp.sol  # All emi component solutions
                n_emi_components = len(emi_solutions)
            else:
                emi_solutions = [pp.sol] if hasattr(pp, 'sol') else [np.zeros(emission_setup.get('emi_moments', 2))]
                n_emi_components = 1

            # set the bestfit to zero for all bad pixels
            pp.bestfit[np.setdiff1d(np.arange(len(pp.bestfit)), goodpixels)] = 0.0
            combined_bestfit = pp.bestfit
            emi_template_weighted = pp.bestfit

        # NEW: Create per-line kinematics arrays instead of single emi_sol
        # This is the KEY FIX: we now store V and Sigma for each line/component

        # Map emi components to individual lines
        line_velocities = np.zeros(len(line_names))
        line_sigmas = np.zeros(len(line_names))

        if emi_component is not None:
            # Multi-component case: assign each line's kinematics based on its component
            for line_idx, line_name in enumerate(line_names):
                component_id = emi_component[line_idx]
                if component_id < len(emi_solutions):
                    emi_sol_for_line = emi_solutions[component_id]
                    if len(emi_sol_for_line) >= 2:
                        line_velocities[line_idx] = emi_sol_for_line[0]
                        line_sigmas[line_idx] = emi_sol_for_line[1]
                else:
                    # Fallback to first component if mapping fails
                    if len(emi_solutions) > 0 and len(emi_solutions[0]) >= 2:
                        line_velocities[line_idx] = emi_solutions[0][0]
                        line_sigmas[line_idx] = emi_solutions[0][1]
        else:
            # Single component case: all lines share the same kinematics
            if len(emi_solutions) > 0 and len(emi_solutions[0]) >= 2:
                line_velocities.fill(emi_solutions[0][0])
                line_sigmas.fill(emi_solutions[0][1])

        # For backward compatibility, also keep the old emi_sol format (first component)
        emi_sol = emi_solutions[0] if len(emi_solutions) > 0 else np.zeros(emission_setup.get('emi_moments', 2))

        # Extract emission line fluxes and calculate amplitudes/AON
        emission_fluxes = np.zeros(len(line_names))
        emission_amplitudes = np.zeros(len(line_names))
        emission_aon = np.zeros(len(line_names))

        if hasattr(pp, 'weights'):
            if emission_setup.get('fix_stellar_continuum', False) or not emission_setup.get('use_stellar_continuum', True):
                emi_weights = pp.weights
            else:
                emi_weights = pp.weights[template.shape[1]:]

            for j, name in enumerate(line_names):
                if j < len(emi_weights):
                    # CORRECTED FLUX CALCULATION: Divide by scaling factor to recover true flux
                    # The templates were scaled by scaling_factor, so weights are scaled too
                    # True flux = weight / scaling_factor
                    true_weight = emi_weights[j] / scaling_factor
                    emission_fluxes[j] = true_weight  # This is now the correct flux

                    if j < emi_template.shape[1]:
                        weighted_template = emi_template[:, j] * emi_weights[j]
                        peak_amplitude_scaled = np.max(weighted_template)
                        emission_amplitudes[j] = peak_amplitude_scaled / scaling_factor

                        if line_wavelengths is not None:
                            line_wave = line_wavelengths.get(name, 0)
                        else:
                            line_wave = 0
                        if line_wave > 0:
                            line_pixel = np.argmin(np.abs(lam_emi - line_wave))

                            if line_pixel < len(log_bin_error):
                                noise_at_line = log_bin_error[line_pixel]
                                if noise_at_line > 0:
                                    emission_aon[j] = emission_amplitudes[j] / noise_at_line
                                else:
                                    emission_aon[j] = 0.0
                            else:
                                emission_aon[j] = 0.0
                        else:
                            emission_aon[j] = 0.0

        # Diagnostic plots if requested
        if diag_plots:
            try:
                plot_file = save_ppxf_emi_plot(
                    rootname= rootname,
                    pp=pp,
                    data=log_bin_data,
                    error=log_bin_error,
                    wave=lam_emi,
                    goodpixels=goodpixels,
                    bin_id=i,
                    plot_dir=figdir,
                    line_names=line_names,
                    stellar_component=stellar_bestfit,
                    emi_component=pp.bestfit,
                    debug=False
                )
            except Exception as plot_error:
                ExGalutil.prettyOutput_Warning(f"Diagnostic plot failed for bin {i}: {plot_error}")

        # Error analysis via Monte Carlo simulations (reduced for efficiency)
        nsims_actual = min(nsims, 20)  # Limit MC sims to prevent excessive runtime
        sol_MC = np.zeros((nsims_actual, nmoments))
        emi_sol_MC = np.zeros((nsims_actual, emission_setup.get('emi_moments', 2)))

        # NEW: MC arrays for per-line kinematics
        line_velocities_MC = np.zeros((nsims_actual, len(line_names)))
        line_sigmas_MC = np.zeros((nsims_actual, len(line_names)))

        emission_fluxes_MC = np.zeros((nsims_actual, len(line_names)))
        emission_amplitudes_MC = np.zeros((nsims_actual, len(line_names)))
        emission_aon_MC = np.zeros((nsims_actual, len(line_names)))

        for o in range(nsims_actual):
            try:
                # Add noise to bestfit: same as PPXF stellar
                noise_std = np.std(log_bin_data[goodpixels] - combined_bestfit[goodpixels])
                noisy_bestfit = combined_bestfit + np.random.normal(0, 1, len(log_bin_data)) * noise_std * 0.5  # Reduced noise

                # FIXED: Prepare MC start parameters to match the main fitting approach
                if emi_component is not None:
                    n_emi_components = len(np.unique(emi_component))
                    if n_emi_components > 1:
                        # Multi-component case: need list of start vectors
                        mc_emi_start_multi = [emi_start_simple.copy() for _ in range(n_emi_components)]
                        mc_emi_moments_multi = [emission_setup.get('emi_moments', 2)] * n_emi_components
                        mc_multi_bounds = [get_emi_bounds(emi_start_simple, stellar_present=(stellar_bestfit is not None)).copy() for _ in range(n_emi_components)]
                    else:
                        # Single component case: can use simple arrays
                        mc_emi_start_multi = emi_start_simple.copy()
                        mc_emi_moments_multi = emission_setup.get('emi_moments', 2)
                        mc_multi_bounds = get_emi_bounds(emi_start_simple, stellar_present=(stellar_bestfit is not None))

                    mc = ppxf(emi_template_unbroadened, noisy_bestfit, log_bin_error, velscale,
                              start=mc_emi_start_multi, goodpixels=goodpixels, plot=False,
                              quiet=True, moments=mc_emi_moments_multi,
                              bounds=mc_multi_bounds,
                              degree=-1, mdegree=-1, velscale_ratio=velscale_ratio, bias=0.0,
                              lam=lam_emi, lam_temp=lam_emi_template,
                              component=emi_component)
                else:
                    # No emi component tying - single component MC
                    mc = ppxf(emi_template_unbroadened, noisy_bestfit, log_bin_error, velscale,
                              start=emi_start_simple, goodpixels=goodpixels, plot=False,
                              quiet=True, moments=emission_setup.get('emi_moments', 2),
                              bounds=get_emi_bounds(emi_start_simple, stellar_present=(stellar_bestfit is not None)),
                              degree=-1, mdegree=-1, velscale_ratio=velscale_ratio, bias=0.0,
                              lam=lam_emi, lam_temp=lam_emi_template)

                # Extract kinematics from MC
                if hasattr(mc, 'sol'):
                    if isinstance(mc.sol, list) and len(mc.sol) > 0:
                        # Multi-component MC result
                        if emi_component is not None and len(np.unique(emi_component)) > 1:
                            # Store first component for backward compatibility
                            emi_sol_MC[o, :] = mc.sol[0][:emission_setup.get('emi_moments', 2)]

                            # Extract per-line kinematics from MC (multi-component)
                            mc_emi_solutions = mc.sol
                            for line_idx, line_name in enumerate(line_names):
                                component_id = emi_component[line_idx]
                                if component_id < len(mc_emi_solutions):
                                    mc_emi_sol_for_line = mc_emi_solutions[component_id]
                                    if len(mc_emi_sol_for_line) >= 2:
                                        line_velocities_MC[o, line_idx] = mc_emi_sol_for_line[0]
                                        line_sigmas_MC[o, line_idx] = mc_emi_sol_for_line[1]
                        else:
                            # Single component case (list format)
                            emi_sol_MC[o, :] = mc.sol[0][:emission_setup.get('emi_moments', 2)]
                            if len(mc.sol[0]) >= 2:
                                line_velocities_MC[o, :] = mc.sol[0][0]
                                line_sigmas_MC[o, :] = mc.sol[0][1]
                    else:
                        # Single component case (array format)
                        emi_sol_MC[o, :] = mc.sol[:emission_setup.get('emi_moments', 2)]
                        if len(mc.sol) >= 2:
                            line_velocities_MC[o, :] = mc.sol[0]
                            line_sigmas_MC[o, :] = mc.sol[1]

                # NEW: Extract flux/amplitude variations from MC
                if hasattr(mc, 'weights'):
                    if emission_setup.get('fix_stellar_continuum', False) or not emission_setup.get('use_stellar_continuum', True):
                        mc_emi_weights = mc.weights
                    else:
                        mc_emi_weights = mc.weights[template.shape[1]:]

                    # Store MC flux and amplitude values
                    for j, name in enumerate(line_names):
                        if j < len(mc_emi_weights):
                            # Store MC flux value
                            emission_fluxes_MC[o, j] = mc_emi_weights[j] / scaling_factor

                            # Calculate MC amplitude from weighted template
                            if j < emi_template.shape[1]:
                                mc_weighted_template = emi_template[:, j] * mc_emi_weights[j]
                                peak_amplitude_scaled = np.max(mc_weighted_template)
                                emission_amplitudes_MC[o, j] = peak_amplitude_scaled / scaling_factor

                                # Calculate MC AON (amplitude over noise)
                                if line_wavelengths is not None:
                                    line_wave = line_wavelengths.get(name, 0)
                                    if line_wave > 0:
                                        line_pixel = np.argmin(np.abs(lam_emi - line_wave))
                                        if line_pixel < len(log_bin_error):
                                            noise_at_line = log_bin_error[line_pixel]
                                            if noise_at_line > 0:
                                                emission_aon_MC[o, j] = emission_amplitudes_MC[o, j] / noise_at_line
                                            else:
                                                emission_aon_MC[o, j] = 0.0
                                        else:
                                            emission_aon_MC[o, j] = 0.0
                                    else:
                                        emission_aon_MC[o, j] = 0.0
                                else:
                                    emission_aon_MC[o, j] = 0.0

            except Exception as mc_error:
                if o < 3:
                    print(f"    MC simulation {o} failed: {str(mc_error)[:50]}...")
                continue

        # Calculate errors from MC simulations
        mc_results_stellar = np.nanstd(sol_MC, axis=0) if nsims_actual > 0 else np.zeros(nmoments)
        mc_results_emi = np.nanstd(emi_sol_MC, axis=0) if nsims_actual > 0 else np.zeros(emission_setup.get('emi_moments', 2))

        # NEW: Calculate per-line kinematic errors
        mc_results_line_velocities = np.nanstd(line_velocities_MC, axis=0) if nsims_actual > 0 else np.zeros(len(line_names))
        mc_results_line_sigmas = np.nanstd(line_sigmas_MC, axis=0) if nsims_actual > 0 else np.zeros(len(line_names))

        mc_results_fluxes = np.nanstd(emission_fluxes_MC, axis=0) if nsims_actual > 0 else np.zeros(len(line_names))
        mc_results_amplitudes = np.nanstd(emission_amplitudes_MC, axis=0) if nsims_actual > 0 else np.zeros(len(line_names))
        mc_results_aon = np.nanstd(emission_aon_MC, axis=0) if nsims_actual > 0 else np.zeros(len(line_names))

        # Extract formal errors for emission line fluxes and amplitudes
        formal_error_fluxes = np.zeros(len(line_names))
        formal_error_amplitudes = np.zeros(len(line_names))

        if hasattr(pp, 'weights_error') and hasattr(pp, 'chi2'):
            # pPXF provides weights_error which are formal errors on the template weights
            weights_formal_errors = pp.weights_error * np.sqrt(pp.chi2)

            # Extract emi template weights errors (skip stellar templates)
            if emission_setup.get('fix_stellar_continuum', False) or not emission_setup.get('use_stellar_continuum', True):
                emi_weights_errors = weights_formal_errors
            else:
                emi_weights_errors = weights_formal_errors[template.shape[1]:]

            # Assign to individual lines
            for j, name in enumerate(line_names):
                if j < len(emi_weights_errors):
                    # Flux formal error = weight formal error
                    formal_error_fluxes[j] = emi_weights_errors[j]

                    # Amplitude formal error: convert from weight error using template scaling
                    if j < emi_template.shape[1]:
                        template_max = np.max(np.abs(emi_template[:, j]))
                        if template_max > 0:
                            formal_error_amplitudes[j] = emi_weights_errors[j] * template_max
                        else:
                            formal_error_amplitudes[j] = 0.0

        elif hasattr(pp, 'dof') and hasattr(pp, 'chi2'):
            # Fallback: estimate from chi2 and degrees of freedom
            chi2_per_dof = pp.chi2 / pp.dof if pp.dof > 0 else 1.0
            estimated_error = np.sqrt(chi2_per_dof)

            # Apply to all lines
            formal_error_fluxes.fill(estimated_error)
            formal_error_amplitudes.fill(estimated_error)

        else:
            # No formal errors available
            formal_error_fluxes.fill(np.nan)
            formal_error_amplitudes.fill(np.nan)

        # NEW: Enhanced mc_results structure with per-line kinematics
        mc_results = {
            "stellar": mc_results_stellar,
            "emi": mc_results_emi,
            "fluxes": mc_results_fluxes,
            "amplitudes": emission_amplitudes,
            "amplitude_errors": mc_results_amplitudes,
            "aon": emission_aon,
            # NEW: Per-line kinematics and errors
            "line_velocities": line_velocities,
            "line_sigmas": line_sigmas,
            "line_velocity_errors": mc_results_line_velocities,
            "line_sigma_errors": mc_results_line_sigmas
        }

        # Formal errors from PPXF - same as PPXF stellar
        formal_error_stellar = np.zeros(nmoments)
        formal_error_emi = np.zeros(emission_setup.get('emi_moments', 2))

        # NEW: Formal errors for per-line kinematics
        formal_error_line_velocities = np.zeros(len(line_names))
        formal_error_line_sigmas = np.zeros(len(line_names))

        if hasattr(pp, 'error') and hasattr(pp, 'chi2'):
            if isinstance(pp.error, list):
                if len(pp.error) > 0:
                    formal_error_stellar = pp.error[0] * np.sqrt(pp.chi2) if not emission_setup.get('fix_stellar_continuum', False) else np.full(nmoments, np.nan)

                # NEW: Handle formal errors for each emi component
                for comp_idx in range(len(pp.error) - 1):  # Skip stellar component
                    emi_comp_error = pp.error[comp_idx + 1] * np.sqrt(pp.chi2)

                    # Assign to lines belonging to this component
                    if emi_component is not None:
                        for line_idx, line_name in enumerate(line_names):
                            if emi_component[line_idx] == comp_idx:
                                if len(emi_comp_error) >= 2:
                                    formal_error_line_velocities[line_idx] = emi_comp_error[0]
                                    formal_error_line_sigmas[line_idx] = emi_comp_error[1]
                    else:
                        # Single component: all lines get same formal errors
                        if len(emi_comp_error) >= 2:
                            formal_error_line_velocities.fill(emi_comp_error[0])
                            formal_error_line_sigmas.fill(emi_comp_error[1])

                    # For backward compatibility, keep first component as emi formal error
                    if comp_idx == 0:
                        formal_error_emi = emi_comp_error

            else:
                if emission_setup.get('fix_stellar_continuum', False) or not emission_setup.get('use_stellar_continuum', True):
                    formal_error_stellar = np.full(nmoments, np.nan)
                    formal_error_emi = pp.error * np.sqrt(pp.chi2)
                    # Single component case
                    if len(pp.error) >= 2:
                        formal_error_line_velocities.fill(pp.error[0] * np.sqrt(pp.chi2))
                        formal_error_line_sigmas.fill(pp.error[1] * np.sqrt(pp.chi2))
                else:
                    formal_error_emi = pp.error * np.sqrt(pp.chi2) if len(pp.error) >= emission_setup.get('emi_moments', 2) else np.zeros(emission_setup.get('emi_moments', 2))
                    if len(pp.error) >= 2:
                        formal_error_line_velocities.fill(pp.error[0] * np.sqrt(pp.chi2))
                        formal_error_line_sigmas.fill(pp.error[1] * np.sqrt(pp.chi2))

        # NEW: Enhanced formal_error structure
        formal_error = {
            "stellar": formal_error_stellar,
            "emi": formal_error_emi,
            "line_velocities": formal_error_line_velocities,
            "line_sigmas": formal_error_line_sigmas,
            "fluxes": formal_error_fluxes,           # NEW
            "amplitudes": formal_error_amplitudes    # NEW
        }

        return ({"stellar": stellar_sol, "emi": emi_sol},
                combined_bestfit,
                emi_template_weighted,
                emission_fluxes,
                mc_results,
                formal_error,
                goodpixels)

    except Exception as e:
        print(f"Error in emi fitting for bin {i}: {str(e)}")
        traceback.print_exc()
        return _emi_fit_nan_result(nmoments, emission_setup, line_names, log_bin_data, goodpixels)


def get_emi_bounds(emi_start_vals, stellar_present=False):
    """Get proper bounds for emi kinematics to prevent extreme solutions"""
    bounds = []

    if len(emi_start_vals) >= 1:
        # Velocity bounds: ±300 km/s from start
        v_start = emi_start_vals[0]
        bounds.append([v_start - 300, v_start + 300])

    if len(emi_start_vals) >= 2:
        # CRITICAL FIX: Ensure sigma bounds are always valid
        sigma_start = emi_start_vals[1]

        # Ensure sigma_start is reasonable
        if sigma_start <= 0 or np.isnan(sigma_start) or np.isinf(sigma_start):
            sigma_start = 150.0  # Default fallback

        if stellar_present:
            sigma_min = max(25.0, sigma_start * 0.5)
            sigma_max = max(sigma_start * 2.5, 400.0)  # Changed min to max
        else:
            sigma_min = max(25.0, sigma_start * 0.7)
            sigma_max = max(sigma_start * 2.0, 400.0)  # Changed min to max

        # CRITICAL: Ensure min < max
        if sigma_min >= sigma_max:
            # Force valid bounds
            sigma_min = 25.0
            sigma_max = 400.0
            print(f"    Warning: Invalid sigma bounds detected. Using defaults: [{sigma_min}, {sigma_max}]")

        bounds.append([sigma_min, sigma_max])

    # Higher moments bounds
    for j in range(2, len(emi_start_vals)):
        bounds.append([-0.2, 0.2])

    return bounds


def get_bias_value(spectrum, goodpixels, stellar_present=False):
    """Calculate appropriate bias for regularization"""
    data_std = np.std(spectrum[goodpixels])

    # Stronger regularization for emi-only fits
    if stellar_present:
        bias_factor = 0.0001  # Light regularization when stellar is present
    else:
        bias_factor = 0.001  # Stronger regularization for emi-only

    return max(1e-8, data_std * bias_factor)





def extract_group_order_from_tie_settings(tie_settings):
    """
    Dynamically extract group names from tie_settings dictionary.
    This completely eliminates the need for hardcoded group lists.

    Parameters:
    -----------
    tie_settings : dict
        The tie settings dictionary loaded from config

    Returns:
    --------
    group_order : list
        List of group names found in tie_settings (excluding special flags)
    """

    # Define special keys that are not group names
    special_keys = {
        'tie_all', '[OIII]_5006.77', '[NII]_6583.34'  # Legacy group references
    }

    # Extract all keys that are lists (these are the groups)
    group_order = []
    for key, value in tie_settings.items():
        if (key not in special_keys and
            isinstance(value, list) and
            len(value) > 0):
            group_order.append(key)

    # Sort for consistent ordering
    group_order.sort()

    ExGalutil.prettyOutput_Info(f"Dynamically extracted {len(group_order)} groups from tie_settings:")
    for i, group in enumerate(group_order, 1):
        group_lines = tie_settings[group]
        print(f"  {i}. {group}: {len(group_lines)} lines")

    return group_order





def build_emi_component_tying(line_names, tie_settings, group_order=None, debug=True):
    """
    Build sophisticated emi component tying based on tie_settings configuration.

    This function determines which emission lines should share the same kinematic
    components (velocity and velocity dispersion) during pPXF fitting. Lines that
    are "tied" together will have identical kinematics, reducing the number of
    free parameters and improving computational efficiency.

    HOW IT WORKS:
    =============

    1. **Single Component Check**:
       - If tie_all=True, all lines share one component → return array of zeros

    2. **Component Assignment Process**:
       - Initialize component counter = 0
       - Process groups in priority order:
         b) New-style groups (high_ion_forbidden, low_ion_forbidden, etc.)
         c) Legacy groups ([OIII]_5006.77, [NII]_6583.34)
         e) Independent components for remaining lines

    3. **Group Processing Logic**:
       - For each group, check which lines from that group are present in line_names
       - If any lines are found, create a new component
       - Assign all group lines to the same component ID
       - Mark those lines as "assigned" to prevent double-assignment

    4. **Output**:
       - Returns integer array where each element = component ID for that line
       - Lines with same ID share kinematics
       - Returns None if only one component (all lines tied)

    EXAMPLE:
    ========
    line_names = ['Hbeta', '[OIII]_4958.83', '[OIII]_5006.77', '[NII]_6583.34', 'Halpha']
    tie_settings = {
        'high_ion_forbidden': ['[OIII]_4958.83', '[OIII]_5006.77'],
        'low_ion_forbidden': ['[NII]_6583.34']
    }

    Result: [0, 1, 1, 2, 0]
    Meaning:
    - Component 0: Hbeta, Halpha (Balmer lines)
    - Component 1: [OIII]_4958.83, [OIII]_5006.77 (high ionization)
    - Component 2: [NII]_6583.34 (low ionization)

    COMPUTATIONAL IMPACT:
    ====================
    Without tying: 5 lines x 2 moments = 10 free parameters
    With tying:    3 components x 2 moments = 6 free parameters
    Reduction:     40% fewer parameters → faster, more stable fits

    CONFIGURATION GUIDE:
    ===================

    To add a new group to group_order:
    1. Add the group name to the group_order list (around line 45)
    2. Ensure the group name matches exactly what's in your emission config
    3. The group should contain a list of line names in tie_settings

    Current group_order processes:
    - 'high_ion_forbidden'  # High-excitation forbidden lines
    - 'low_ion_forbidden'   # Low-excitation forbidden lines
    - 'sulfur_group'               # Sulfur lines
    - 'hei_recombination'          # Neutral helium recombination
    - 'heii_recombination'         # Ionized helium recombination
    - 'uv_resonance'               # UV resonance lines (often broad)
    - 'uv_semiforbidden'           # UV semi-forbidden transitions
    - 'iron_lines'                 # Iron transitions (complex atoms)
    - 'miscellaneous'              # Other weak lines

    Legacy support (for backward compatibility):
    - "[OIII]_5006.77": [list of lines]  # Lines tied to OIII 5006.77 kinematics
    - "[NII]_6583.34": [list of lines]   # Lines tied to NII 6583.34 kinematics

    TROUBLESHOOTING:
    ===============

    1. **Group not processed**: Check spelling in group_order vs tie_settings
    2. **Lines not tied**: Verify line names match exactly between
       tie_settings groups and line_names input
    3. **Too many components**: Check if lines are being double-assigned
    4. **No tying**: Check if tie_settings contains the expected groups

    PERFORMANCE NOTES:
    =================
    - More components = slower fitting but more physical detail
    - Fewer components = faster fitting but may miss kinematic differences
    - Typical reduction: 89 lines → 5-9 components (90-95% parameter reduction)

    Parameters:
    -----------
    line_names : list
        List of emission line names that will be fitted
        Example: ['Hbeta', "[OIII]_4958.83", "[OIII]_5006.77", 'Halpha']

    tie_settings : dict
        Dictionary containing tying configuration. Can contain:

        Universal flags:
        - 'tie_all': bool - Tie all lines to single component
        - group_order (lists of line names):

    Returns:
    --------
    emi_component : numpy.ndarray or None
        - If multiple components: integer array of length len(line_names)
          where each element is the component ID (0, 1, 2, ...) for that line
        - If single component: None (pPXF optimization)

        Example return: np.array([0, 1, 1, 2, 0]) means:
        - Lines 0,4 belong to component 0
        - Lines 1,2 belong to component 1
        - Line 3 belongs to component 2

    Notes:
    ------
    - Lines are assigned to groups in the order they appear in group_order
    - First match wins - no line can belong to multiple groups
    - Unassigned lines become independent components
    - Component numbering starts from 0
    - Function prints component summary for debugging
    """


    if tie_settings.get('tie_all', False):
        # All emission lines share kinematics - simplest case
        return np.zeros(len(line_names), dtype=int)

    # Initialize component array
    emi_component = np.zeros(len(line_names), dtype=int)
    component_counter = 0

    # Track which lines have been assigned
    assigned_lines = set()
    component_map = {}


    # Handle all other tie groups from the new configuration
    # Process groups in a specific order (sorted) to maintain consistency
    default_group_order = sorted([
        'balmer',
        'high_ion_forbidden',
        'low_ion_forbidden',
        'sulfur_group',
        'hei_recombination',
        'heii_recombination',
        'uv_resonance',
        'uv_semiforbidden',
        'iron_lines',
        'miscellaneous'
    ])

    if group_order is None:
        # If no group_order provided, use default order
        group_order = default_group_order
        print(f"Using default group order: {', '.join(group_order)}")

    # if debug:
    #     print(f"Processing {len(group_order)} groups from tie_settings:")


    # Also handle legacy group names for backward compatibility
    legacy_groups = ["balmer", "forbidden", "[OIII]_5006.77", "[NII]_6583.34"]

    # Process new-style groups
    for group_name in group_order:
        if group_name in tie_settings and isinstance(tie_settings[group_name], list):
            tied_lines = tie_settings[group_name]

            # Check if any of the tied lines are in our line_names
            group_lines_present = [line for line in tied_lines if line in line_names and line not in assigned_lines]

            if group_lines_present:
                group_component = component_counter
                component_map[group_name] = group_component
                component_counter += 1

                # Assign all lines in this group to the same component
                for j, line_name in enumerate(line_names):
                    if line_name in tied_lines and line_name not in assigned_lines:
                        emi_component[j] = group_component
                        assigned_lines.add(line_name)

    # Handle legacy group names for backward compatibility
    for group_name in legacy_groups:
        if group_name in tie_settings and isinstance(tie_settings[group_name], list):
            tied_lines = tie_settings[group_name]

            # For legacy groups, also include the reference line
            if group_name == "[OIII]_5006.77":
                reference_line = "[OIII]_5006.77"
            elif group_name == "[NII]_6583.34":
                reference_line = "[NII]_6583.34"
            else:
                reference_line = None

            # Check if reference line or any tied lines are present
            all_group_lines = tied_lines.copy()
            if reference_line and reference_line not in all_group_lines:
                all_group_lines.append(reference_line)

            group_lines_present = [line for line in all_group_lines if line in line_names and line not in assigned_lines]

            if group_lines_present:
                group_component = component_counter
                component_map[group_name] = group_component
                component_counter += 1

                # Assign all lines in this group to the same component
                for j, line_name in enumerate(line_names):
                    if line_name in all_group_lines and line_name not in assigned_lines:
                        emi_component[j] = group_component
                        assigned_lines.add(line_name)


    # Handle any remaining unassigned lines as independent components
    for j, line_name in enumerate(line_names):
        if line_name not in assigned_lines:
            emi_component[j] = component_counter
            component_map[f'Independent_{line_name}'] = component_counter
            component_counter += 1
            assigned_lines.add(line_name)

    # Check if we have multiple components
    unique_components = np.unique(emi_component)

    if len(unique_components) > 1:
        # Print tying information for debugging (only first call to avoid spam)
        if len(line_names) > 0:
            if debug:
                ExGalutil.prettyOutput_Info(f"  emi component tying: {len(unique_components)} kinematic components")
                for comp in unique_components:
                    tied_lines = [line_names[j] for j in range(len(line_names)) if emi_component[j] == comp]
                    component_name = next((name for name, comp_id in component_map.items() if comp_id == comp), f"Component_{comp}")
                    print(f"    Component {comp} ({component_name}): {', '.join(tied_lines)}")
        return emi_component
    else:
        # All lines in single component - return None for simpler PPXF call
        return None




def get_tie_settings_from_config(config_file=None, config_dict=None):
    """
    Load tie settings from configuration file or dictionary.
    Updated to handle the new emission line configuration format.

    Parameters:
    -----------
    config_file : str, optional
        Path to configuration file containing tie_settings
    config_dict : dict, optional
        Pre-loaded configuration dictionary

    Returns:
    --------
    tie_settings : dict
        Dictionary containing line tying configuration
    """
    # Default tie settings (fallback)
    default_tie_settings = {
        'balmer': [
            "H12_3749.93","H11_3770.93","H10_3797.92","H9_3835.91",
            "H5_3889.05","H8_3888.90","He_3970.07","Hd_4101.73",
            "Hg_4340.46","Hbeta_4861.32","Halpha_6562.80"
        ],
        'high_ion_forbidden': [
            "[OIII]_4958.83",
            "[OIII]_5006.77",
            "[OIII]_4363.15",
            "[OIII]_4363.21",
            "[OIII]_4931.23",
            "[NeIII]_3868.69",
            "[NeIII]_3967.40",
            "[NeV]_3345.81",
            "[NeV]_3425.81",
            "[NeVI]_3425.87",
            "[NeIV]_2438.76",
            "[ArIV]_4711.30",
            "[ArIV]_4740.10",
            "[FeXIV]_5302.86"
        ],
        'low_ion_forbidden': [
            "[NII]_6547.96",
            "[NII]_6583.34",
            "[NII]_5754.40",
            "[SII]_6716.31",
            "[SII]_6730.68",
            "[SII]_4071.15",
            "[OI]_6300.20",
            "[OI]_6363.67",
            "[OII]_3726.03",
            "[OII]_3728.73",
            "[OII]_7319.46",
            "[OII]_7329.98",
            "[ArIII]_7135.67",
            "[NI]_5197.90",
            "[NI]_5200.39",
            "[NI]_6527.23",
            "[ClIII]_5537.89"
        ],
        'hei_recombination': [
            "HeI_3887.90",
            "HeI_4471.74",
            "HeI_5875.60",
            "HeI_5875.37",
            "HeI_6678.16"
        ],
        'tie_all': False
    }

    # If config_dict is provided, use it directly
    if config_dict is not None:
        tie_settings = config_dict.get('tie_settings', default_tie_settings)
        print(f"Loaded tie settings from provided configuration dictionary")
        return tie_settings

    # If config_file is provided, try to load it
    if config_file is not None:
        try:
            # Handle different file formats
            if config_file.endswith('.py'):
                # Python configuration file
                config_globals = {}
                with open(config_file, 'r') as f:
                    exec(f.read(), config_globals)
                tie_settings = config_globals.get('tie_settings', default_tie_settings)
                print(f"Loaded tie settings from Python config file: {config_file}")

            elif config_file.endswith('.json'):
                # JSON configuration file
                import json
                with open(config_file, 'r') as f:
                    config = json.load(f)
                tie_settings = config.get('tie_settings', default_tie_settings)
                print(f"Loaded tie settings from JSON config file: {config_file}")

            else:
                print(f"Warning: Unsupported config file format: {config_file}")
                print("Using default tie settings")
                tie_settings = default_tie_settings

        except Exception as e:
            print(f"Warning: Could not load tie settings from {config_file}: {e}")
            print("Using default tie settings")
            tie_settings = default_tie_settings
    else:
        # No config provided, use defaults
        print("No configuration provided, using default tie settings")
        tie_settings = default_tie_settings

    return tie_settings


def estimate_component_count(line_names, tie_settings, group_order=None):
    """
    Estimate the number of kinematic components that will be created.

    Parameters:
    -----------
    line_names : list
        List of emission line names
    tie_settings : dict
        Dictionary containing tying configuration

    Returns:
    --------
    component_count : int
        Estimated number of kinematic components
    component_breakdown : dict
        Breakdown of which lines go into which components
    """
    if tie_settings.get('tie_all', False):
        return 1, {'all_lines': line_names}

    assigned_lines = set()
    component_breakdown = {}
    component_count = 0


    # Count tie groups
    default_tie_groups = [
        'balmer', 'forbidden', 'high_ion_forbidden', 'low_ion_forbidden', 'sulfur_group',
        'hei_recombination', 'heii_recombination', 'uv_resonance',
        'uv_semiforbidden', 'iron_lines', 'miscellaneous',
        '[OIII]_5006.77', '[NII]_6583.34'  # Include legacy groups with updated names
    ]


    tie_groups = default_tie_groups if group_order is None else group_order + ['[OIII]_5006.77', '[NII]_6583.34']

    for group_name in tie_groups:
        if group_name in tie_settings and isinstance(tie_settings[group_name], list):
            group_lines = tie_settings[group_name]

            # For legacy groups, add reference line
            if group_name == 'oiii_5006.77' and "[OIII]_5006.77" not in group_lines:
                group_lines = group_lines + ["[OIII]_5006.77"]
            elif group_name == 'nii_6583.34' and "[NII]_6583.34" not in group_lines:
                group_lines = group_lines + ["[NII]_6583.34"]

            lines_in_data = [line for line in group_lines if line in line_names and line not in assigned_lines]
            if lines_in_data:
                component_count += 1
                component_breakdown[group_name] = lines_in_data
                assigned_lines.update(lines_in_data)


    # Count independent lines
    independent_lines = [line for line in line_names if line not in assigned_lines]
    component_count += len(independent_lines)

    if independent_lines:
        component_breakdown['independent'] = independent_lines

    return component_count, component_breakdown



def apply_ppxf_emission_masking(logLam, velscale, emission_config,
                                error_spectrum=None, error_limit=1e+18,
                                configs=None, verbose=False, LSF_Data=None,
                                redshift=0.0):
    """
    Apply masking based on pPXF emission line configuration with LSF-dependent widths.

    Parameters:
    -----------
    logLam : ndarray
        Log wavelength array (natural log) - already de-redshifted to rest frame
    velscale : float
        Velocity scale in km/s per pixel
    emission_config : dict
        pPXF emission line configuration dictionary
    error_spectrum : ndarray, optional
        Error spectrum for CCD gap masking
    error_limit : float, optional
        Error limit for CCD gap masking
    configs : dict, optional
        Additional configuration parameters
    verbose : bool, optional
        Print masking information
    LSF_Data : callable, optional
        Interpolation function giving data FWHM(λ) in Å
    redshift : float, optional
        Target redshift. Sky line mask wavelengths are in observed frame and
        must be divided by (1+z) to convert to rest frame before masking
        the de-redshifted spectrum. Astrophysical emission line masks are
        already in rest frame and need no correction. Default: 0.0

    Returns:
    --------
    goodpixels : ndarray
        Array of good pixel indices

    Notes:
    ------
    logLam is in the rest frame (spectrum has been de-redshifted).
    Astrophysical emission line masks are already in rest frame — no correction needed.
    Sky/telluric masks (name starts with 'sky_') are in OBSERVED frame and must be
    divided by (1+z) to place them correctly in the rest-frame spectrum.
    """
    if verbose:
        print("Applying LSF-dependent pPXF emission line masking")
        if redshift != 0.0:
            print(f"  Redshift z={redshift:.4f}: sky line masks will be converted from "
                  f"observed frame to rest frame")

    # Speed of light
    c = 299792.458  # km/s

    # Handle finite part of logLam (for padded arrays)
    finite_logLam = logLam[np.isfinite(logLam)]

    if len(finite_logLam) < 2:
        raise ValueError("logLam does not contain enough finite values for masking.")

    l0_gal = finite_logLam[0]
    lstep_gal = finite_logLam[1] - finite_logLam[0]
    npix = len(logLam)

    # Start with all pixels
    goodpixels = np.arange(0, npix)

    # Apply wavelength range limits if specified
    if configs is not None:
        l_rf_range = None
        if 'LMIN_EMI' in configs and 'LMAX_EMI' in configs:
            l_rf_range = [configs['LMIN_EMI'], configs['LMAX_EMI']]
        elif 'LMIN_PPXF' in configs and 'LMAX_PPXF' in configs:
            l_rf_range = [configs['LMIN_PPXF'], configs['LMAX_PPXF']]

        if l_rf_range is not None:
            pix0 = int(np.ceil((np.log(l_rf_range[0]) - l0_gal) / lstep_gal))
            pix1 = int(np.ceil((np.log(l_rf_range[1]) - l0_gal) / lstep_gal))

            if pix0 >= npix or pix1 <= 0:
                if verbose:
                    print(f"Warning: Wavelength range [{l_rf_range[0]:.1f}-{l_rf_range[1]:.1f}Å] "
                          f"is outside spectrum coverage")
                    print(f"Spectrum covers approximately "
                          f"[{np.exp(l0_gal):.1f}-{np.exp(l0_gal + (npix-1)*lstep_gal):.1f}Å]")
                goodpixels = np.array([], dtype=int)
            else:
                goodpixels = np.arange(np.max([pix0, 0]), np.min([pix1, npix]))
                if verbose:
                    print(f"Applied wavelength range {l_rf_range[0]:.1f}-{l_rf_range[1]:.1f}Å: "
                          f"{len(goodpixels)} pixels")

    tmppixels = goodpixels.copy()

    # Apply masking for regions in mask_regions
    mask_regions = emission_config.get('mask_regions', []) if emission_config else []
    n_masked_total = 0
    lsf_enhanced_lines = 0
    lsf_failed_lines = 0
    n_sky_corrected = 0
    n_restframe = 0

    if verbose and LSF_Data is not None:
        print(f"Using LSF-dependent masking widths")
    elif verbose:
        print(f"Using fixed masking widths (no LSF provided)")

    for mask_region in mask_regions:
        name = mask_region['name']
        wavelength_obs = mask_region['wavelength']
        base_width = mask_region.get('width', 3.0)
        safety_factor = mask_region.get('safety_factor', 2.5)

        # Determine whether this mask is a sky/telluric line (observed frame)
        # or an astrophysical line (rest frame).
        # All entries in mask_regions with name starting 'sky_' or containing
        # 'sky' or 'telluric' are atmospheric features at fixed observed-frame
        # wavelengths and must be divided by (1+z) to convert to rest frame.
        is_sky = 'sky' in name.lower() or 'telluric' in name.lower()

        if is_sky and redshift != 0.0:
            wavelength = wavelength_obs / (1.0 + redshift)
            n_sky_corrected += 1
            if verbose and n_sky_corrected <= 3:
                print(f"  Sky mask '{name}': {wavelength_obs:.2f} Å (obs) → "
                      f"{wavelength:.2f} Å (rest, z={redshift:.4f})")
        else:
            wavelength = wavelength_obs
            n_restframe += 1

        # Calculate effective width using LSF if available
        if LSF_Data is not None:
            try:
                lsf_fwhm = LSF_Data(wavelength)
                total_width = np.sqrt(base_width**2 + lsf_fwhm**2)
                effective_width = total_width * safety_factor
                lsf_enhanced_lines += 1

                if verbose and np.random.random() < 0.1:
                    print(f"LSF-enhanced masking {name} at {wavelength:.1f}Å: "
                          f"base={base_width:.2f}Å + LSF={lsf_fwhm:.2f}Å → "
                          f"total={total_width:.2f}Å × {safety_factor:.1f} = {effective_width:.2f}Å")

            except (ValueError, TypeError, AttributeError) as e:
                effective_width = base_width * safety_factor
                lsf_failed_lines += 1

                if verbose and lsf_failed_lines <= 3:
                    print(f"Warning: LSF evaluation failed for {name} at {wavelength:.1f}Å: {e}")
                    print(f"  Using fallback width: {effective_width:.2f}Å")
        else:
            effective_width = base_width * safety_factor

        # Convert effective width to sigma in velocity units, then to pixels
        sigma_angstrom = effective_width / 2.355
        sigma_velocity = sigma_angstrom / wavelength * c
        msigma = 3 * sigma_velocity / velscale

        meml_cpix = np.ceil((np.log(wavelength) - l0_gal) / lstep_gal)

        if verbose and np.random.random() < 0.05:
            print(f"Masking {name} at {wavelength:.1f}Å: "
                  f"width={effective_width:.2f}Å → {msigma:.1f} pixels")

        meml_bpix = meml_cpix - msigma
        meml_rpix = meml_cpix + msigma

        w = np.where((goodpixels >= meml_bpix) & (goodpixels <= meml_rpix))[0]
        if len(w) > 0:
            tmppixels[w] = -1
            n_masked_total += len(w)

    if verbose:
        total_regions = len(mask_regions)
        print(f"Mask region summary:")
        print(f"  Total mask regions: {total_regions}")
        print(f"  Sky/telluric masks (observed→rest frame corrected): {n_sky_corrected}")
        print(f"  Astrophysical masks (already rest frame): {n_restframe}")
        if LSF_Data is not None:
            print(f"  LSF-enhanced: {lsf_enhanced_lines}")
            print(f"  LSF failed (fallback): {lsf_failed_lines}")
            print(f"  LSF success rate: "
                  f"{lsf_enhanced_lines/total_regions*100:.1f}%" if total_regions > 0 else "")
        print(f"  Total pixels masked: {n_masked_total}")

    # Sanity check for z=0.0653 case: sky_5577 should land at ~5236 Å
    if verbose and redshift != 0.0:
        for mask_region in mask_regions[:1]:  # Just check first sky line
            if 'sky' in mask_region['name'].lower():
                wave_obs = mask_region['wavelength']
                wave_rest = wave_obs / (1.0 + redshift)
                print(f"  Sanity check: {mask_region['name']} "
                      f"{wave_obs:.1f} Å (obs) → {wave_rest:.1f} Å (rest)")
                break

    # Remove masked pixels
    w = np.where(tmppixels != -1)[0]
    goodpixels = goodpixels[w]

    if verbose:
        print(f"After spectral masking: {len(goodpixels)} pixels")

    # Apply error-based masking (CCD gaps)
    if error_spectrum is not None:
        error_threshold = 0.95 * np.sqrt(error_limit)
        errors_at_goodpix = error_spectrum[goodpixels]
        goodpix_idx = np.where(errors_at_goodpix < error_threshold)[0]

        if len(goodpix_idx) > 0:
            goodpixels = goodpixels[goodpix_idx]
            if verbose:
                print(f"After error masking: {len(goodpixels)} pixels")
        else:
            goodpixels = np.array([], dtype=int)
            if verbose:
                print("Warning: No pixels survive error masking")

    # Final check for finite values
    if len(goodpixels) > 0 and error_spectrum is not None:
        finite_mask = np.isfinite(error_spectrum[goodpixels])
        goodpixels = goodpixels[finite_mask]
        if verbose:
            print(f"Final good pixels: {len(goodpixels)}")

    return goodpixels



def prepare_emission_template(logLam, line_names, line_wavelengths,
                                    LSF_Template, LSF_Data, wl_offset=300.0, line_ties=None, debug=False):
    """
    FIXED VERSION: Prepare emission line template for pPXF fitting using wavelength-dependent LSF.
    This version correctly uses ppxf_util.gaussian function and returns both broadened and unbroadened templates.

    Parameters:
    ----------
    logLam : array
        Log-wavelength array (natural log) for the data
    line_names : list
        List of emission line names
    line_wavelengths : dict
        Dict mapping line name to rest-frame wavelength (Å)
    LSF_Template : callable
        Interpolation function giving template FWHM(λ) in Å
    LSF_Data : callable
        Interpolation function giving data FWHM(λ) in Å
    wl_offset : float, optional
        Wavelength buffer in Å (default: 300.0, matching stellar template)
    line_ties : dict, optional
        Placeholder for future line tying functionality
    debug : bool, optional
        Enable detailed diagnostics (default: False)

    Returns:
    -------
    emission_template_broadened : ndarray, shape (n_template_pixels, n_lines)
        Broadened emission line template ready for pPXF fitting
    emission_template_unbroadened : ndarray, shape (n_template_pixels, n_lines)
        Unbroadened emission line template with minimal FWHM (1e-3 Å)
    template_logLam : ndarray
        Log-wavelength array for the template (extended if wl_offset > 0)
    """
    lam = np.exp(logLam)
    n_lines = len(line_names)
    lmin_data = lam.min()
    lmax_data = lam.max()

    if debug:
        print("=== FIXED EMISSION TEMPLATE CREATION WITH DIAGNOSTICS ===")
        print(f"Data wavelength range: {lmin_data:.1f} - {lmax_data:.1f} Å")
        print(f"Data pixels: {len(logLam)}")
        print(f"wl_offset: {wl_offset:.1f} Å")
        print(f"Debug mode: {debug}")
        print("Creating both broadened and unbroadened templates")





    if wl_offset > 0:
        # CREATE EXTENDED WAVELENGTH GRID
        lmin_eff = lmin_data - wl_offset
        lmax_eff = lmax_data + wl_offset

        # Create extended log-wavelength grid with same sampling as data
        dlogLam = logLam[1] - logLam[0]  # Log wavelength step

        # Calculate extended grid bounds
        logLam_min_eff = np.log(lmin_eff)
        logLam_max_eff = np.log(lmax_eff)

        # Create extended grid
        template_logLam = np.arange(
            logLam_min_eff,
            logLam_max_eff + dlogLam/2,  # Small buffer to include endpoint
            dlogLam
        )
        template_lam = np.exp(template_logLam)

        if debug:
            print(f"Extended wavelength range: {lmin_eff:.1f} - {lmax_eff:.1f} Å")
            print(f"Extended template pixels: {len(template_logLam)}")

        # Effective range for line inclusion
        lmin_inclusion = lmin_eff
        lmax_inclusion = lmax_eff

    else:
        # Use data grid directly (no extension)
        template_logLam = logLam.copy()
        template_lam = lam
        lmin_inclusion = lmin_data
        lmax_inclusion = lmax_data


    n_template_pixels = len(template_logLam)
    if debug:
        print(f"Template grid info:")
        print(f"  Wavelength range: {template_lam.min():.1f} - {template_lam.max():.1f} Å")
        print(f"  Pixels: {n_template_pixels}")
        print(f"  Velocity sampling: {(template_logLam[1] - template_logLam[0]) * C:.2f} km/s per pixel")
        print(f"Number of lines to process: {n_lines}")

    # DEBUG: Initial wavelength range analysis
    if debug:
        print(f"\n=== DEBUG: WAVELENGTH RANGE ANALYSIS ===")
        line_waves = [line_wavelengths.get(line, 0) for line in line_names if line_wavelengths.get(line, 0) > 0]
        if line_waves:
            line_min = min(line_waves)
            line_max = max(line_waves)
            template_min = template_lam.min()
            template_max = template_lam.max()

            print(f"Line wavelength range: {line_min:.1f} - {line_max:.1f} Å")
            print(f"Template range:        {template_min:.1f} - {template_max:.1f} Å")

            lines_in_range = sum(1 for w in line_waves if template_min <= w <= template_max)
            lines_below = sum(1 for w in line_waves if w < template_min)
            lines_above = sum(1 for w in line_waves if w > template_max)

            print(f"Lines in template range: {lines_in_range}/{len(line_waves)}")
            print(f"Lines below range: {lines_below}/{len(line_waves)}")
            print(f"Lines above range: {lines_above}/{len(line_waves)}")

    successful_lines = 0
    failed_lines = []
    outside_range_lines = []
    offset_range_lines = []

    # Collect lines to process
    lines_to_process = []

    for j, line in enumerate(line_names):
        line_wave = line_wavelengths.get(line)

        if line_wave is None:
            if debug or successful_lines < 3:
                print(f"[WARN] Line {j}: {line} - Missing wavelength")
            failed_lines.append(f"{line} (no wavelength)")
            continue

        # Check if line is within inclusion range
        if line_wave < lmin_inclusion or line_wave > lmax_inclusion:
            outside_range_lines.append(f"{line} ({line_wave:.1f} Å)")
            if debug:
                print(f"[SKIP] Line {j}: {line} at {line_wave:.1f} Å - Outside range")
            continue

        # Check if line is within original data range or offset range
        in_data_range = (lmin_data <= line_wave <= lmax_data)
        in_offset_range = ((lmin_inclusion <= line_wave < lmin_data) or
                          (lmax_data < line_wave <= lmax_inclusion))

        if in_offset_range:
            offset_range_lines.append(f"{line} ({line_wave:.1f} Å)")

        lines_to_process.append({
            'index': j,
            'name': line,
            'wavelength': line_wave,
            'in_data_range': in_data_range
        })

    if debug:
        print(f"\nLines to process: {len(lines_to_process)}")

    # Prepare arrays for batch processing
    if len(lines_to_process) > 0:
        # Extract wavelengths for all lines to process
        line_wavelengths_array = np.array([line_info['wavelength'] for line_info in lines_to_process])

        if debug:
            print(f"\n=== DEBUG: PROCESSING LINES IN BATCH ===")
            print(f"Processing {len(line_wavelengths_array)} lines at once")
            print(f"Line wavelengths: {line_wavelengths_array[:10]}...")  # Show first 10

        try:
            # Calculate FWHM for all lines at once
            data_lsf_array = np.array([LSF_Data(lw) for lw in line_wavelengths_array])
            temp_lsf_array = np.array([LSF_Template(lw) for lw in line_wavelengths_array])

            # Calculate FWHM difference (convolution needed) for broadened template
            fwhm_diff_sq = data_lsf_array**2 - temp_lsf_array**2

            # Handle cases where data LSF < template LSF
            problematic_mask = (fwhm_diff_sq <= 0) | ~np.isfinite(fwhm_diff_sq)
            fwhm_diff_broadened = np.sqrt(np.maximum(fwhm_diff_sq, (1e-2)**2))  # Minimum fallback

            # Create unbroadened FWHM array (minimal broadening)
            fwhm_diff_unbroadened = np.full_like(line_wavelengths_array, 1e-3)

            if debug and np.any(problematic_mask):
                n_problematic = np.sum(problematic_mask)
                print(f"   {n_problematic} lines have data LSF <= template LSF (using fallback)")

            # Create broadened emission template
            if debug:
                print(f"Creating broadened emission line template using ppxf_util.gaussian...")

            emission_template_broadened = ppxf_util.gaussian(
                ln_lam_temp=template_logLam,           # Log-wavelength array
                line_wave=line_wavelengths_array,      # Line wavelengths in Å
                FWHM_gal=fwhm_diff_broadened,         # FWHM in Å (broadened)
                pixel=True                            # Analytic integration over pixels
            )
            if debug:
                # Create unbroadened emission template
                print(f"Creating unbroadened emission line template using ppxf_util.gaussian...")

            emission_template_unbroadened = ppxf_util.gaussian(
                ln_lam_temp=template_logLam,           # Log-wavelength array
                line_wave=line_wavelengths_array,      # Line wavelengths in Å
                FWHM_gal=fwhm_diff_unbroadened,       # FWHM in Å (minimal: 1e-3)
                pixel=True                            # Analytic integration over pixels
            )

            if debug:
                print(f"Broadened template shape: {emission_template_broadened.shape}")
                print(f"Unbroadened template shape: {emission_template_unbroadened.shape}")
                print(f"Expected shape: ({n_template_pixels}, {len(line_wavelengths_array)})")

            # Check template quality for both templates
            for template_name, template in [("broadened", emission_template_broadened),
                                          ("unbroadened", emission_template_unbroadened)]:
                if template.shape[0] != n_template_pixels:
                    raise ValueError(f"{template_name} template pixel count mismatch: {template.shape[0]} vs {n_template_pixels}")

                if template.shape[1] != len(line_wavelengths_array):
                    raise ValueError(f"{template_name} template line count mismatch: {template.shape[1]} vs {len(line_wavelengths_array)}")

                # Verify template is finite and non-zero
                if not np.all(np.isfinite(template)):
                    print(f"WARNING: {template_name} template contains non-finite values")
                    template = np.nan_to_num(template, nan=0.0)

                template_max = np.max(np.abs(template))
                if template_max == 0:
                    raise ValueError(f"All {template_name} templates are zero")

                if debug:
                    print(f"{template_name.capitalize()} template quality:")
                    print(f"  Template max amplitude: {template_max:.6e}")
                    print(f"  Template RMS: {np.sqrt(np.mean(template**2)):.6e}")

            successful_lines = len(line_wavelengths_array)

            if debug:
                print(f"Template creation successful!")

                # Check where peaks are located for broadened template
                peak_pixels = []
                peak_wavelengths = []
                for j in range(emission_template_broadened.shape[1]):
                    peak_idx = np.argmax(np.abs(emission_template_broadened[:, j]))
                    peak_wave = template_lam[peak_idx]
                    peak_pixels.append(peak_idx)
                    peak_wavelengths.append(peak_wave)

                    line_info = lines_to_process[j]
                    expected_wave = line_info['wavelength']
                    offset = peak_wave - expected_wave

                    if j < 10:  # Show first 10
                        print(f"  {line_info['name']}: Expected {expected_wave:.1f} Å, Peak at {peak_wave:.1f} Å (offset {offset:+.1f} Å)")

                unique_peaks = len(set(peak_pixels))
                print(f"  Unique peak locations: {unique_peaks}/{len(peak_pixels)}")

                if unique_peaks < len(peak_pixels) / 2:
                    print(f"  ⚠️ WARNING: Too few unique peak locations")
                else:
                    print(f"  ✅ Good spread of peak locations")

        except Exception as e:
            print(f"ERROR in batch template creation: {e}")
            failed_lines.extend([line_info['name'] for line_info in lines_to_process])
            emission_template_broadened = np.zeros((n_template_pixels, n_lines))
            emission_template_unbroadened = np.zeros((n_template_pixels, n_lines))
            successful_lines = 0
    else:
        print("No lines to process")
        emission_template_broadened = np.zeros((n_template_pixels, n_lines))
        emission_template_unbroadened = np.zeros((n_template_pixels, n_lines))
        successful_lines = 0

    # Create full template arrays (including skipped lines as zeros)
    full_emission_template_broadened = np.zeros((n_template_pixels, n_lines))
    full_emission_template_unbroadened = np.zeros((n_template_pixels, n_lines))

    if successful_lines > 0:
        # Fill in the successfully created templates
        for template_idx, line_info in enumerate(lines_to_process):
            original_idx = line_info['index']
            full_emission_template_broadened[:, original_idx] = emission_template_broadened[:, template_idx]
            full_emission_template_unbroadened[:, original_idx] = emission_template_unbroadened[:, template_idx]

    # Final cleanup
    full_emission_template_broadened = np.nan_to_num(full_emission_template_broadened, nan=0.0)
    full_emission_template_unbroadened = np.nan_to_num(full_emission_template_unbroadened, nan=0.0)

    # Check which columns are non-zero
    non_zero_template_broadened = np.sum(np.max(np.abs(full_emission_template_broadened), axis=0) > 0)
    non_zero_template_unbroadened = np.sum(np.max(np.abs(full_emission_template_unbroadened), axis=0) > 0)
    if debug:
        # Summary
        print(f"\n=== FIXED EMISSION TEMPLATE CREATION SUMMARY ===")
        print(f"Template wavelength grid:")
        print(f"  Range: {template_lam.min():.1f} - {template_lam.max():.1f} Å")
        print(f"  Pixels: {n_template_pixels}")
        print(f"  Log range: {template_logLam[0]:.6f} to {template_logLam[-1]:.6f}")

    if wl_offset > 0:
        print(f"  Extended by: ±{wl_offset:.1f} Å from data range")
        data_pixels_in_template = np.sum((template_lam >= lmin_data) & (template_lam <= lmax_data))
        print(f"  Data pixels covered: {data_pixels_in_template}/{len(logLam)}")
    if debug:
        print(f"Line processing results:")
        print(f"  Total lines in config: {n_lines}")
        print(f"  Successfully created: {successful_lines}")
        print(f"  Failed during creation: {len(failed_lines)}")
        print(f"  Outside extended range: {len(outside_range_lines)}")
        print(f"  In offset range only: {len(offset_range_lines)}")
        print(f"  Broadened template with non-zero values: {non_zero_template_broadened}")
        print(f"  Unbroadened template with non-zero values: {non_zero_template_unbroadened}")

    if len(offset_range_lines) > 0:
        print(f"Lines in offset range: {offset_range_lines[:5]}{'...' if len(offset_range_lines) > 5 else ''}")

    if len(failed_lines) > 0:
        print(f"Failed lines: {failed_lines[:5]}{'...' if len(failed_lines) > 5 else ''}")

    # Validation
    if successful_lines == 0:
        print("WARNING: No emission line templates were successfully created!")
    elif non_zero_template_broadened < successful_lines or non_zero_template_unbroadened < successful_lines:
        print(f"WARNING: Templates with issues detected!")

    if debug:
        # Template quality check
        template_max_broadened = np.max(np.abs(full_emission_template_broadened))
        template_rms_broadened = np.sqrt(np.mean(full_emission_template_broadened**2))
        template_max_unbroadened = np.max(np.abs(full_emission_template_unbroadened))
        template_rms_unbroadened = np.sqrt(np.mean(full_emission_template_unbroadened**2))

        print(f"Template quality:")
        print(f"  Broadened - Maximum amplitude: {template_max_broadened:.6e}")
        print(f"  Broadened - RMS amplitude: {template_rms_broadened:.6e}")
        print(f"  Unbroadened - Maximum amplitude: {template_max_unbroadened:.6e}")
        print(f"  Unbroadened - RMS amplitude: {template_rms_unbroadened:.6e}")

        if template_max_broadened > 0 and template_max_unbroadened > 0:
            print(f"✅ Both templates created successfully using correct ppxf_util.gaussian!")
        else:
            print(f"❌ One or both templates are zero - check wavelength ranges and LSF")

        print("=" * 65)

    return full_emission_template_broadened, full_emission_template_unbroadened, template_logLam



def categorize_emission_lines(emission_config, line_names_fitted, wavelength_range, wavelength_range_config):
    """
    Categorize ALL emission lines from config into fitted, masked, and out-of-range.

    Parameters:
    -----------
    emission_config : dict
        The full emission line configuration
    line_names_fitted : list
        Lines that were actually fitted by pPXF
    wavelength_range : tuple
        (lmin, lmax) wavelength range used for fitting (rest frame)

    Returns:
    --------
    categories : dict
        {
            'fitted': [list of fitted line names],
            'masked': [list of masked line names],
            'out_of_range': [list of lines outside wavelength range],
            'missing_wavelength': [list of lines without wavelengths]
        }
    """
    if emission_config is None:
        return {
            'fitted': line_names_fitted,
            'masked': [],
            'out_of_range': [],
            'missing_wavelength': []
        }

    all_line_names = list(emission_config.get('emission_lines', {}).keys())
    all_line_wavelengths = emission_config.get('emission_lines', {})
    mask_regions = emission_config.get('mask_regions', [])

    # Create sets for efficient lookup
    fitted_set = set(line_names_fitted)

    # Build list of masked wavelengths (all in rest frame)
    masked_wavelengths = []
    for mask_region in mask_regions:
        wavelength = mask_region['wavelength']  # Already in rest frame
        width = mask_region.get('width', 3.0)

        # Store as (center, half_width) for easy checking
        masked_wavelengths.append((wavelength, width/2.0))

    categories = {
        'fitted': [],
        'masked': [],
        'out_of_range': [],
        'missing_wavelength': []
    }

    lmin, lmax = wavelength_range

    for line_name in all_line_names:
        line_wave = all_line_wavelengths.get(line_name)

        # Check if line is within wavelength range
        # also for the fitted lines we check if the line is already in the line_names_fit (coming from ppxf worker)
        min_allowed = max(wavelength_range[0] - 50.0, wavelength_range_config[0])
        max_allowed = min(wavelength_range[1] + 50.0, wavelength_range_config[1])


        if line_wave is None:
            categories['missing_wavelength'].append(line_name)
            continue

        # All wavelengths are already in rest frame - no redshift correction needed
        line_wave_rest = line_wave

        if line_name in fitted_set and line_wave_rest > min_allowed and line_wave_rest < max_allowed:
            categories['fitted'].append(line_name)
        else:
            # Check if line is masked
            is_masked = False
            for mask_center, mask_half_width in masked_wavelengths:
                if abs(line_wave_rest - mask_center) <= mask_half_width:
                    is_masked = True
                    break

            if is_masked:
                categories['masked'].append(line_name)
            elif line_wave_rest < min_allowed or line_wave_rest > max_allowed:
                categories['out_of_range'].append(line_name)
            else:
                # Line is in range and not masked, but wasn't fitted -
                # This could be due to other reasons (low S/N, etc.)
                categories['out_of_range'].append(line_name)  # Or create 'other' category

    return categories

def save_emi_kinematics(rootname, configs, outdir, results, mc_errors, formal_errors,
                       lambdaR, bestfit, logLam, spectra, error, goodpixels,
                       emi_bestfit, emission_lines, line_names, npix, ubins,
                       tie_settings, emi_components=None, z_in=None, z_err=None,
                       emission_config=None, line_wavelengths=None, group_order=None, debug=False):
    """
    UPDATED VERSION: Save emi kinematics results with per-line kinematics and proper handling of masked lines.
    Creates columns for fitted lines (with data), masked lines, and out-of-range lines (with np.nan).
    Updated to handle new comprehensive tie settings format and proper line categorization.

    FLUX FILTERING: Kinematics are set to NaN for lines with flux ≤ 0 or NaN.

    New parameters:
    ---------------
    emission_config : dict, optional
        The loaded pPXF emission configuration containing emission_lines, tie_settings, etc.
    line_wavelengths : dict, optional
        Dictionary mapping line names to rest wavelengths
    """

    # Determine wavelength range used for fitting
    if 'LMIN_EMI' in configs and 'LMAX_EMI' in configs:
        wavelength_range_config = (configs['LMIN_EMI'], configs['LMAX_EMI'])
    elif 'LMIN_PPXF' in configs and 'LMAX_PPXF' in configs:
        wavelength_range_config = (configs['LMIN_PPXF'], configs['LMAX_PPXF'])
    else:
        # Fallback to data range
        wavelength_range_config = (np.nan, np.nan)


    # Fallback to data range
    lam = np.exp(logLam)
    wavelength_range = (lam.min(), lam.max())

    # Get ALL lines from emission config
    assert emission_config is not None and 'emission_lines' in emission_config, 'no config file found to build the all_line_names'
    all_line_names = list(emission_config['emission_lines'].keys())
    all_line_wavelengths = emission_config['emission_lines']

    # Categorize ALL lines properly
    line_categories = categorize_emission_lines(
        emission_config,
        all_line_names,
        wavelength_range,
        wavelength_range_config
    )

    if debug:
        print(f"Processing ALL {len(all_line_names)} lines from emission config:")
        print(f"  - Fitted: {len(line_categories['fitted'])}")
        print(f"  - Masked: {len(line_categories['masked'])}")
        print(f"  - Out of range: {len(line_categories['out_of_range'])}")
        print(f"  - Missing wavelength: {len(line_categories['missing_wavelength'])}")

    # Create mapping from fitted lines to all lines
    fitted_line_indices = []
    line_status = []  # Track status: 'FITTED', 'MASKED', 'OUT_OF_RANGE', 'NO_WAVELENGTH'

    for name in all_line_names:
        if name in line_categories['fitted']:
            fitted_line_indices.append(all_line_names.index(name))
            line_status.append('FITTED')
        elif name in line_categories['masked']:
            fitted_line_indices.append(-1)  # Not fitted
            line_status.append('MASKED')
        elif name in line_categories['out_of_range']:
            fitted_line_indices.append(-1)  # Not fitted
            line_status.append('OUT_OF_RANGE')
        elif name in line_categories['missing_wavelength']:
            fitted_line_indices.append(-1)  # Not fitted
            line_status.append('NO_WAVELENGTH')
        else:
            fitted_line_indices.append(-1)  # Not fitted - other reason
            line_status.append('OTHER')

    # Extract emission line properties from mc_errors results - for ALL lines
    emission_amplitudes_all = np.full((len(ubins), len(all_line_names)), np.nan)
    emission_amplitude_errors_all = np.full((len(ubins), len(all_line_names)), np.nan)
    emission_aon_all = np.full((len(ubins), len(all_line_names)), np.nan)
    emission_fluxes_all = np.full((len(ubins), len(all_line_names)), np.nan)
    emission_flux_errors_all = np.full((len(ubins), len(all_line_names)), np.nan)

    # NEW: Extract per-line kinematics - for ALL lines
    line_velocities_all = np.full((len(ubins), len(all_line_names)), np.nan)
    line_sigmas_all = np.full((len(ubins), len(all_line_names)), np.nan)
    line_velocity_errors_all = np.full((len(ubins), len(all_line_names)), np.nan)
    line_sigma_errors_all = np.full((len(ubins), len(all_line_names)), np.nan)
    line_formal_velocity_errors_all = np.full((len(ubins), len(all_line_names)), np.nan)
    line_formal_sigma_errors_all = np.full((len(ubins), len(all_line_names)), np.nan)

    # NEW: Extract per-line formal flux/amplitude errors
    line_formal_flux_errors_all = np.full((len(ubins), len(all_line_names)), np.nan)
    line_formal_amplitude_errors_all = np.full((len(ubins), len(all_line_names)), np.nan)

    for i, bin_id in enumerate(ubins):
        if isinstance(mc_errors[i], dict):
            # Fill data for fitted lines only
            for j, line_name in enumerate(all_line_names):
                try:
                    fitted_line_position = line_names.index(line_name)

                    if fitted_line_position >= 0:  # Line was fitted

                        # FIRST: Extract flux to check if it's valid
                        current_flux = np.nan
                        if isinstance(emission_lines[i], np.ndarray) and len(emission_lines[i]) > fitted_line_position:
                            current_flux = emission_lines[i][fitted_line_position]
                            emission_fluxes_all[i, j] = current_flux

                        # FLUX CONDITION: Only extract kinematics if flux > 0 and finite
                        flux_is_valid = (not np.isnan(current_flux) and current_flux is not None and current_flux > 1e-10 and line_name in line_categories['fitted'])

                        if not flux_is_valid:
                           emission_fluxes_all[i, j] =  np.nan

                        if flux_is_valid:
                            # Extract amplitudes
                            if 'amplitudes' in mc_errors[i] and isinstance(mc_errors[i]['amplitudes'], np.ndarray):
                                if len(mc_errors[i]['amplitudes']) > fitted_line_position:
                                    emission_amplitudes_all[i, j] = mc_errors[i]['amplitudes'][fitted_line_position]

                            # Extract amplitude errors
                            if 'amplitude_errors' in mc_errors[i] and isinstance(mc_errors[i]['amplitude_errors'], np.ndarray):
                                if len(mc_errors[i]['amplitude_errors']) > fitted_line_position:
                                    emission_amplitude_errors_all[i, j] = mc_errors[i]['amplitude_errors'][fitted_line_position]

                            # Extract AON
                            if 'aon' in mc_errors[i] and isinstance(mc_errors[i]['aon'], np.ndarray):
                                if len(mc_errors[i]['aon']) > fitted_line_position:
                                    emission_aon_all[i, j] = mc_errors[i]['aon'][fitted_line_position]

                            # Extract flux errors
                            if 'fluxes' in mc_errors[i] and isinstance(mc_errors[i]['fluxes'], np.ndarray):
                                if len(mc_errors[i]['fluxes']) > fitted_line_position:
                                    emission_flux_errors_all[i, j] = mc_errors[i]['fluxes'][fitted_line_position]

                        # NEW: Extract per-line kinematics ONLY if flux is valid
                        if flux_is_valid:
                            if 'line_velocities' in mc_errors[i] and isinstance(mc_errors[i]['line_velocities'], np.ndarray):
                                if len(mc_errors[i]['line_velocities']) > fitted_line_position:
                                    line_velocities_all[i, j] = mc_errors[i]['line_velocities'][fitted_line_position]

                            if 'line_sigmas' in mc_errors[i] and isinstance(mc_errors[i]['line_sigmas'], np.ndarray):
                                if len(mc_errors[i]['line_sigmas']) > fitted_line_position:
                                    line_sigmas_all[i, j] = mc_errors[i]['line_sigmas'][fitted_line_position]

                            if 'line_velocity_errors' in mc_errors[i] and isinstance(mc_errors[i]['line_velocity_errors'], np.ndarray):
                                if len(mc_errors[i]['line_velocity_errors']) > fitted_line_position:
                                    line_velocity_errors_all[i, j] = mc_errors[i]['line_velocity_errors'][fitted_line_position]

                            if 'line_sigma_errors' in mc_errors[i] and isinstance(mc_errors[i]['line_sigma_errors'], np.ndarray):
                                if len(mc_errors[i]['line_sigma_errors']) > fitted_line_position:
                                    line_sigma_errors_all[i, j] = mc_errors[i]['line_sigma_errors'][fitted_line_position]

                            # NEW: Extract per-line formal errors ONLY if flux is valid
                            if isinstance(formal_errors[i], dict):
                                if 'line_velocities' in formal_errors[i] and isinstance(formal_errors[i]['line_velocities'], np.ndarray):
                                    if len(formal_errors[i]['line_velocities']) > fitted_line_position:
                                        line_formal_velocity_errors_all[i, j] = formal_errors[i]['line_velocities'][fitted_line_position]

                                if 'line_sigmas' in formal_errors[i] and isinstance(formal_errors[i]['line_sigmas'], np.ndarray):
                                    if len(formal_errors[i]['line_sigmas']) > fitted_line_position:
                                        line_formal_sigma_errors_all[i, j] = formal_errors[i]['line_sigmas'][fitted_line_position]

                        if flux_is_valid:
                            # Extract formal flux/amplitude errors regardless of flux validity
                            if isinstance(formal_errors[i], dict):
                                if 'fluxes' in formal_errors[i] and isinstance(formal_errors[i]['fluxes'], np.ndarray):
                                    if len(formal_errors[i]['fluxes']) > fitted_line_position:
                                        line_formal_flux_errors_all[i, j] = formal_errors[i]['fluxes'][fitted_line_position]

                                if 'amplitudes' in formal_errors[i] and isinstance(formal_errors[i]['amplitudes'], np.ndarray):
                                    if len(formal_errors[i]['amplitudes']) > fitted_line_position:
                                        line_formal_amplitude_errors_all[i, j] = formal_errors[i]['amplitudes'][fitted_line_position]
                except:
                    # Non-fitted lines (masked, out-of-range, etc.) remain np.nan (already initialized)
                    pass

    # ========================
    # POST-PROCESSING: Apply flux filtering to all kinematics arrays
    # This ensures consistency even if some kinematics were extracted before flux check
    if debug:
        n_flux_filtered = 0
        n_total_fitted = 0

    for i in range(len(ubins)):
        for j in range(len(all_line_names)):
            current_flux = emission_fluxes_all[i, j]
            is_fitted = fitted_line_indices[j] >= 0

            if is_fitted:
                if debug:
                    n_total_fitted += 1

                # If flux is invalid (NaN or ≤ 0), set all kinematics to NaN
                if not (np.isfinite(current_flux) and current_flux > 0):
                    line_velocities_all[i, j] = np.nan
                    line_sigmas_all[i, j] = np.nan
                    line_velocity_errors_all[i, j] = np.nan
                    line_sigma_errors_all[i, j] = np.nan
                    line_formal_velocity_errors_all[i, j] = np.nan
                    line_formal_sigma_errors_all[i, j] = np.nan

                    if debug:
                        n_flux_filtered += 1

    if debug and n_total_fitted > 0:
        print(f"  Flux filtering: {n_flux_filtered}/{n_total_fitted} fitted line measurements had invalid flux (set kinematics to NaN)")

    # Determine the component structure using tie settings format
    if emi_components is not None and len(emi_components) > 0:
        # Find maximum number of components across all bins
        max_components = 1
        for i in range(len(ubins)):
            if emi_components[i] is not None:
                n_comp = len(np.unique(emi_components[i]))
                max_components = max(max_components, n_comp)

        if debug:
            print(f"Detected maximum {max_components} kinematic components")

        # Create component mapping for interpretation using tie settings
        component_names = []
        if tie_settings.get('tie_all', False):
            component_names = ['ALL_LINES']
        else:
            # Build component names based on tie settings format
            comp_idx = 0

            if group_order is not None and isinstance(group_order, list):
                # Handle new-style groups
                new_groups = [(group_name, group_name.upper()) for group_name in group_order]

                for group_key, group_name in new_groups:
                    if tie_settings.get(group_key) and any(line in line_names for line in tie_settings[group_key]):
                        component_names.append(group_name)
                        comp_idx += 1

            # Handle legacy groups for backward compatibility
            if tie_settings.get("[OIII]_5006.77") and any(line in line_names for line in tie_settings["[OIII]_5006.77"]):
                component_names.append('OIII_GROUP')
                comp_idx += 1
            if tie_settings.get("[NII]_6583.34") and any(line in line_names for line in tie_settings["[NII]_6583.34"]):
                component_names.append('NII_GROUP')
                comp_idx += 1

            # Fill remaining component names for independent lines
            while len(component_names) < max_components:
                component_names.append(f'COMPONENT_{len(component_names)}')
    else:
        max_components = 1
        component_names = ['SINGLE_COMPONENT']

    # Extract results for each component
    emi_kinematics_multicomp = np.full((len(ubins), max_components, 2), np.nan)  # [bin, component, moment]
    emi_mc_errors_multicomp = np.full((len(ubins), max_components, 2), np.nan)
    emi_formal_errors_multicomp = np.full((len(ubins), max_components, 2), np.nan)

    # Also keep the representative (primary) component for backward compatibility
    emi_kinematics_primary = np.zeros((len(ubins), 2))  # Primary component (usually component 0)
    emi_mc_errors_primary = np.zeros((len(ubins), 2))
    emi_formal_errors_primary = np.zeros((len(ubins), 2))

    # Extract stellar kinematics
    stellar_kinematics = np.zeros((len(ubins), 6))  # Full stellar kinematics

    for i, bin_id in enumerate(ubins):
        # Handle multi-component emi results
        if isinstance(results[i]["emi"], list) and len(results[i]["emi"]) > 0:
            # Multi-component case
            for comp_idx in range(min(len(results[i]["emi"]), max_components)):
                if isinstance(results[i]["emi"][comp_idx], np.ndarray) and len(results[i]["emi"][comp_idx]) >= 2:
                    emi_kinematics_multicomp[i, comp_idx, 0] = results[i]["emi"][comp_idx][0]  # V
                    emi_kinematics_multicomp[i, comp_idx, 1] = results[i]["emi"][comp_idx][1]  # sigma

            # Use first component as primary
            if len(results[i]["emi"]) > 0:
                emi_kinematics_primary[i, :] = emi_kinematics_multicomp[i, 0, :]

        elif isinstance(results[i]["emi"], np.ndarray) and len(results[i]["emi"]) >= 2:
            # Single component case
            emi_kinematics_primary[i, 0] = results[i]["emi"][0]  # V
            emi_kinematics_primary[i, 1] = results[i]["emi"][1]  # sigma
            emi_kinematics_multicomp[i, 0, 0] = results[i]["emi"][0]
            emi_kinematics_multicomp[i, 0, 1] = results[i]["emi"][1]

        # Handle stellar results
        if isinstance(results[i]["stellar"], np.ndarray) and len(results[i]["stellar"]) >= 6:
            stellar_kinematics[i, :] = results[i]["stellar"]

        # Handle multi-component errors
        if isinstance(mc_errors[i]["emi"], list) and len(mc_errors[i]["emi"]) > 0:
            for comp_idx in range(min(len(mc_errors[i]["emi"]), max_components)):
                if isinstance(mc_errors[i]["emi"][comp_idx], np.ndarray) and len(mc_errors[i]["emi"][comp_idx]) >= 2:
                    emi_mc_errors_multicomp[i, comp_idx, 0] = mc_errors[i]["emi"][comp_idx][0]
                    emi_mc_errors_multicomp[i, comp_idx, 1] = mc_errors[i]["emi"][comp_idx][1]

            # Primary component errors
            if len(mc_errors[i]["emi"]) > 0:
                emi_mc_errors_primary[i, :] = emi_mc_errors_multicomp[i, 0, :]

        elif isinstance(mc_errors[i]["emi"], np.ndarray) and len(mc_errors[i]["emi"]) >= 2:
            emi_mc_errors_primary[i, 0] = mc_errors[i]["emi"][0]
            emi_mc_errors_primary[i, 1] = mc_errors[i]["emi"][1]
            emi_mc_errors_multicomp[i, 0, 0] = mc_errors[i]["emi"][0]
            emi_mc_errors_multicomp[i, 0, 1] = mc_errors[i]["emi"][1]

        # Handle formal errors similarly
        if isinstance(formal_errors[i]["emi"], list) and len(formal_errors[i]["emi"]) > 0:
            for comp_idx in range(min(len(formal_errors[i]["emi"]), max_components)):
                if isinstance(formal_errors[i]["emi"][comp_idx], np.ndarray) and len(formal_errors[i]["emi"][comp_idx]) >= 2:
                    emi_formal_errors_multicomp[i, comp_idx, 0] = formal_errors[i]["emi"][comp_idx][0]
                    emi_formal_errors_multicomp[i, comp_idx, 1] = formal_errors[i]["emi"][comp_idx][1]

            if len(formal_errors[i]["emi"]) > 0:
                emi_formal_errors_primary[i, :] = emi_formal_errors_multicomp[i, 0, :]

        elif isinstance(formal_errors[i]["emi"], np.ndarray) and len(formal_errors[i]["emi"]) >= 2:
            emi_formal_errors_primary[i, 0] = formal_errors[i]["emi"][0]
            emi_formal_errors_primary[i, 1] = formal_errors[i]["emi"][1]
            emi_formal_errors_multicomp[i, 0, 0] = formal_errors[i]["emi"][0]
            emi_formal_errors_multicomp[i, 0, 1] = formal_errors[i]["emi"][1]

    # Set flux units for saving
    if configs['sens_corr']:
        flux_unit_str = '%2e erg/(s cm**2 Angstrom)' % (configs['funits'])
    else:
        flux_unit_str = 'count'
    wave_unit_str = 'Angstrom'

    # ========================
    # SAVE PRIMARY EMI ANALYSIS (backward compatibility) - WITH ALL LINES AND PROPER STATUS
    outfits_primary = outdir + rootname + '_emippxf.fits'
    ExGalutil.prettyOutput_Running("Writing: " + rootname + '_emippxf.fits')

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Table HDU with PRIMARY component emi kinematics and emission line data - ALL LINES
    cols = []
    cols.append(fits.Column(name='BIN_ID', format='J', array=ubins))

    # Add redshift information
    if z_in is not None:
        z_array = np.full(len(ubins), z_in)
        cols.append(fits.Column(name='Z_INPUT', format='D', array=z_array))
        if z_err is not None:
            z_err_array = np.full(len(ubins), z_err)
            cols.append(fits.Column(name='ERR_Z_INPUT', format='D', array=z_err_array))

    # Primary emi kinematics columns
    # cols.append(fits.Column(name='V_EMI', format='D', array=emi_kinematics_primary[:, 0]))
    # cols.append(fits.Column(name='SIGMA_EMI', format='D', array=emi_kinematics_primary[:, 1]))
    # cols.append(fits.Column(name='ERR_V_EMI', format='D', array=emi_mc_errors_primary[:, 0]))
    # cols.append(fits.Column(name='ERR_SIGMA_EMI', format='D', array=emi_mc_errors_primary[:, 1]))
    # cols.append(fits.Column(name='FORM_ERR_V_EMI', format='D', array=emi_formal_errors_primary[:, 0]))
    # cols.append(fits.Column(name='FORM_ERR_SIGMA_EMI', format='D', array=emi_formal_errors_primary[:, 1]))

    # Emission line flux columns - FOR ALL LINES (fitted, masked, out-of-range)
    for i, line in enumerate(all_line_names):
        cols.append(fits.Column(name=f'FLUX_{line}', unit=flux_unit_str, format='D', array=emission_fluxes_all[:, i]))
        cols.append(fits.Column(name=f'ERR_FLUX_{line}', unit=flux_unit_str, format='D', array=emission_flux_errors_all[:, i]))
        cols.append(fits.Column(name=f'AMPL_{line}', unit=flux_unit_str, format='D', array=emission_amplitudes_all[:, i]))
        cols.append(fits.Column(name=f'ERR_AMPL_{line}', unit=flux_unit_str, format='D', array=emission_amplitude_errors_all[:, i]))
        cols.append(fits.Column(name=f'AON_{line}', format='D', array=emission_aon_all[:, i]))

        # NEW: Per-line kinematics columns (flux-filtered)
        cols.append(fits.Column(name=f'V_{line}', unit='km/s', format='D', array=line_velocities_all[:, i]))
        cols.append(fits.Column(name=f'SIGMA_{line}', unit='km/s', format='D', array=line_sigmas_all[:, i]))
        cols.append(fits.Column(name=f'ERR_V_{line}', unit='km/s', format='D', array=line_velocity_errors_all[:, i]))
        cols.append(fits.Column(name=f'ERR_SIGMA_{line}', unit='km/s', format='D', array=line_sigma_errors_all[:, i]))
        cols.append(fits.Column(name=f'FORM_ERR_V_{line}', unit='km/s', format='D', array=line_formal_velocity_errors_all[:, i]))
        cols.append(fits.Column(name=f'FORM_ERR_SIGMA_{line}', unit='km/s', format='D', array=line_formal_sigma_errors_all[:, i]))

    # Create the primary table
    primaryHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    primaryHDU.name = 'EMI_ANALYSIS'

    # Add headers including tie settings information and DETAILED line statistics
    primaryHDU.header['CNF_EMI'] = (configs['CONFIG_FILE'], 'Configs. Filename used by emi analysis')
    primaryHDU.header['TEMPL'] = (configs['SSP_LIB'], 'The library of spectral template')
    primaryHDU.header['PPXF_V'] = (PPXF_VERSION, 'PPXF version')
    primaryHDU.header['APSVERS'] = (APSVERS, 'APS version')
    primaryHDU.header['APSPPXFV'] = (aps_constants.__aps_ifuppxf_version__, 'PyAPS (IFU) PPXF wrapper version')
    primaryHDU.header['CSB_PPXF'] = (configs['stitched'], 'Combines Spectral Bands Status for PPXF')

    # Add basic tie settings to header
    primaryHDU.header['TIE_ALL'] = (tie_settings.get('tie_all', False), 'All lines tied together')
    primaryHDU.header['N_COMP'] = (max_components, 'Maximum number of kinematic components')

    # Add information about new tie groups
    new_groups_info = []
    if group_order is not None and isinstance(group_order, list):
        new_groups = [(group_name, group_name.upper()) for group_name in group_order]
        for group_name in new_groups:
            if tie_settings.get(group_name):
                group_lines = tie_settings[group_name]
                present_lines = [line for line in group_lines if line in line_names]
                if present_lines:
                    new_groups_info.append(f"{group_name}({len(present_lines)})")

    if new_groups_info:
        # Store as comma-separated string, truncate if too long
        groups_str = ', '.join(new_groups_info)
        if len(groups_str) > 60:
            groups_str = f"{len(new_groups_info)} active groups"
        primaryHDU.header['NEW_GROUPS'] = (groups_str, 'Active new-style tie groups')

    # Add DETAILED line statistics with proper categorization
    primaryHDU.header['N_LINES_ALL'] = (len(all_line_names), 'Total lines in emission config')
    primaryHDU.header['N_FITTED'] = (len(line_categories['fitted']), 'Lines successfully fitted')
    primaryHDU.header['N_MASKED'] = (len(line_categories['masked']), 'Lines masked (sky lines, etc.)')
    primaryHDU.header['N_OUT_RANGE'] = (len(line_categories['out_of_range']), 'Lines outside wavelength range')
    primaryHDU.header['N_NO_WAVE'] = (len(line_categories['missing_wavelength']), 'Lines without wavelengths')

    # Add wavelength range used for fitting
    primaryHDU.header['LMIN_USED'] = (wavelength_range[0], 'Minimum wavelength used for fitting')
    primaryHDU.header['LMAX_USED'] = (wavelength_range[1], 'Maximum wavelength used for fitting')

    # Add component names
    for i, comp_name in enumerate(component_names):
        primaryHDU.header[f'COMP_{i:02d}'] = (comp_name, f'Component {i} description')

    # Add ALL line names and their STATUS (fitted, masked, out-of-range, etc.)
    for i, (line, status) in enumerate(zip(all_line_names, line_status)):
        if i < 50:  # Limit to prevent header overflow
            primaryHDU.header[f'LINE_{i:02d}'] = (f'{line}_{status}', f'Line {i+1}: {status}')

    # Add masking information if available
    if emission_config and 'mask_regions' in emission_config:
        n_mask_regions = len(emission_config['mask_regions'])
        primaryHDU.header['N_MASK_REG'] = (n_mask_regions, 'Number of mask regions applied')

        # Add some mask region examples
        for i, mask_region in enumerate(emission_config['mask_regions'][:10]):  # First 10
            name = mask_region['name']
            wave = mask_region['wavelength']
            width = mask_region.get('width', 3.0)
            primaryHDU.header[f'MASK_{i:02d}'] = (f'{name}', f'Mask region {i+1}')

    # Keep the basename of input file (infiles)
    for n_province, province in enumerate(configs['infiles']):
        primaryHDU.header['APSREF_%d' % (n_province)] = (os.path.basename(province), 'L1 reference file')

    # NEW: Add information about per-line kinematics and flux filtering
    primaryHDU.header['PER_LINE_KIN'] = (True, 'Per-line kinematics available (V_<line>, SIGMA_<line>)')
    # primaryHDU.header['FLUX_FILTER'] = (True, 'Kinematics set to NaN for lines with flux ≤ 0 or NaN')
    primaryHDU.header['COMMENT'] = 'V_<line>/SIGMA_<line>: Individual line kinematics from component tying'
    primaryHDU.header['COMMENT'] = 'ERR_FLUX/AMPL_<line>: Monte Carlo errors from simulations'
    # primaryHDU.header['COMMENT'] = 'Kinematics are NaN for lines with flux ≤ 0 or NaN'
    primaryHDU.header['COMMENT'] = 'For single-component fits: all valid lines have identical kinematics'
    primaryHDU.header['COMMENT'] = 'For multi-component fits: V_<line> varies by component assignment'

    # Create HDU list and write primary file
    HDUList = fits.HDUList([priHDU, primaryHDU])
    HDUList.writeto(outfits_primary, overwrite=True)

    ExGalutil.prettyOutput_Done("Writing: " + rootname + '_emippxf.fits')
    logging.info("Wrote: " + outfits_primary)

    # ========================
    # SAVE EMI MODEL SPECTRA
    outfits_spec = outdir + rootname + '_emippxfSpec.fits'
    ExGalutil.prettyOutput_Running("Writing: " + rootname + '_emippxfSpec.fits')

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Table HDU with emi spectra
    cols = []
    loglam_tile = np.tile(logLam, (len(ubins), 1))

    cols.append(fits.Column(name='BIN_ID', format='J', array=ubins))
    cols.append(fits.Column(name='LOGLAM', unit=wave_unit_str, format=str(npix) + 'D', array=loglam_tile))
    cols.append(fits.Column(name='FLUX', unit=flux_unit_str, format=str(npix) + 'D', array=np.transpose(spectra)))
    cols.append(fits.Column(name='ERROR', unit=flux_unit_str, format=str(npix) + 'D', array=np.transpose(error)))
    cols.append(fits.Column(name='BESTFIT', unit=flux_unit_str, format=str(npix) + 'D', array=bestfit))
    cols.append(fits.Column(name='EMI_MODEL', unit=flux_unit_str, format=str(npix) + 'D', array=emi_bestfit))
    cols.append(fits.Column(name='GOODPIX', format=str(npix) + 'J', array=goodpixels))

    specHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    specHDU.name = 'BESTFIT'

    specHDU.header['CNF_EMI'] = (configs['CONFIG_FILE'], 'Configs. Filename used by emi analysis')
    specHDU.header['PPXF_V'] = (PPXF_VERSION, 'PPXF version')
    specHDU.header['APSVERS'] = (APSVERS, 'APS version')
    specHDU.header['SAMPLING'] = (1, 'Sampling mode (0: linear, 1: logarithmic)')

    # Add line categorization statistics to spectra file too
    specHDU.header['N_LINES_ALL'] = (len(all_line_names), 'Total lines in emission config')
    specHDU.header['N_FITTED'] = (len(line_categories['fitted']), 'Lines successfully fitted')
    specHDU.header['N_MASKED'] = (len(line_categories['masked']), 'Lines masked (sky lines, etc.)')
    specHDU.header['N_OUT_RANGE'] = (len(line_categories['out_of_range']), 'Lines outside wavelength range')
    specHDU.header['N_NO_WAVE'] = (len(line_categories['missing_wavelength']), 'Lines without wavelengths')

    # Keep the basename of input file (infiles)
    for n_province, province in enumerate(configs['infiles']):
        specHDU.header['APSREF_%d' % (n_province)] = (os.path.basename(province), 'L1 reference file')

    # Create HDU list and write to file
    HDUList = fits.HDUList([priHDU, specHDU])
    HDUList.writeto(outfits_spec, overwrite=True)

    ExGalutil.prettyOutput_Done("Writing: " + rootname + '_emippxfSpec.fits')
    logging.info("Wrote: " + outfits_spec)

    # ========================
    # SAVE LINE-TO-COMPONENT MAPPING WITH DETAILED STATUS - WITH ALL LINES
    outfits_mapping = outdir + rootname + '_emippxf_mapping.fits'
    ExGalutil.prettyOutput_Running("Writing: " + rootname + '_emippxf_mapping.fits')

    # Create mapping table showing which lines belong to which component - ALL LINES WITH STATUS
    mapping_cols = []
    mapping_cols.append(fits.Column(name='LINE_NAME', format='A20', array=all_line_names))

    # Add detailed status for each line
    mapping_cols.append(fits.Column(name='LINE_STATUS', format='A15', array=line_status))

    # Add wavelengths for all lines
    wavelength_array = [all_line_wavelengths.get(name, np.nan) for name in all_line_names]
    mapping_cols.append(fits.Column(name='WAVELENGTH', format='D', array=wavelength_array))

    # Add observed wavelengths (with redshift)
    if z_in is not None:
        wavelength_obs_array = [w * (1 + z_in) if not np.isnan(w) else np.nan for w in wavelength_array]
        mapping_cols.append(fits.Column(name='WAVELENGTH_OBS', format='D', array=wavelength_obs_array))

    # For each bin, save the component assignment (only for fitted lines)
    if emi_components is not None and len(emi_components) > 0:
        # Use the first bin's component assignment as representative
        if emi_components[0] is not None:
            component_ids = []
            component_names_for_lines = []

            for line_idx, line_name in enumerate(all_line_names):
                fitted_idx = fitted_line_indices[line_idx]
                status = line_status[line_idx]

                if fitted_idx >= 0 and fitted_idx < len(emi_components[0]):
                    comp_id = emi_components[0][fitted_idx]
                    component_ids.append(comp_id)
                    if comp_id < len(component_names):
                        component_names_for_lines.append(component_names[comp_id])
                    else:
                        component_names_for_lines.append(f'COMPONENT_{comp_id}')
                else:
                    component_ids.append(-1)  # Not fitted
                    component_names_for_lines.append(status)  # Use status as component name

            mapping_cols.append(fits.Column(name='COMPONENT_ID', format='J', array=component_ids))
            mapping_cols.append(fits.Column(name='COMPONENT_NAME', format='A20', array=component_names_for_lines))
    else:
        # Single component case - fitted lines get component 0, non-fitted get -1
        component_ids = []
        component_names_for_lines = []

        for line_idx, line_name in enumerate(all_line_names):
            fitted_idx = fitted_line_indices[line_idx]
            status = line_status[line_idx]

            if fitted_idx >= 0:
                component_ids.append(0)
                component_names_for_lines.append('SINGLE_COMPONENT')
            else:
                component_ids.append(-1)
                component_names_for_lines.append(status)

        mapping_cols.append(fits.Column(name='COMPONENT_ID', format='J', array=component_ids))
        mapping_cols.append(fits.Column(name='COMPONENT_NAME', format='A20', array=component_names_for_lines))

    # Add information about why lines were not fitted
    reason_array = []
    for line_idx, line_name in enumerate(all_line_names):
        status = line_status[line_idx]
        if status == 'FITTED':
            reason_array.append('Successfully fitted')
        elif status == 'MASKED':
            reason_array.append('Masked due to sky contamination or bad pixels')
        elif status == 'OUT_OF_RANGE':
            reason_array.append('Outside fitting wavelength range')
        elif status == 'NO_WAVELENGTH':
            reason_array.append('No wavelength defined in config')
        else:
            reason_array.append('Other reason')

    mapping_cols.append(fits.Column(name='REASON', format='A50', array=reason_array))

    # Add per-line statistics across all bins (with flux filtering applied)
    mean_velocity_per_line = []
    std_velocity_per_line = []
    mean_sigma_per_line = []
    std_sigma_per_line = []
    n_valid_kinematics_per_line = []
    n_valid_flux_per_line = []  # NEW: Track lines with valid flux

    # NEW: Add per-line kinematics statistics (flux-filtered)
    for line_idx in range(len(all_line_names)):
        line_velocities = line_velocities_all[:, line_idx]
        line_sigmas = line_sigmas_all[:, line_idx]
        line_fluxes = emission_fluxes_all[:, line_idx]

        valid_vel_mask = np.isfinite(line_velocities)
        valid_sig_mask = np.isfinite(line_sigmas)
        valid_flux_mask = np.isfinite(line_fluxes) & (line_fluxes > 0)

        if np.any(valid_vel_mask):
            mean_velocity_per_line.append(np.mean(line_velocities[valid_vel_mask]))
            std_velocity_per_line.append(np.std(line_velocities[valid_vel_mask]))
        else:
            mean_velocity_per_line.append(np.nan)
            std_velocity_per_line.append(np.nan)

        if np.any(valid_sig_mask):
            mean_sigma_per_line.append(np.mean(line_sigmas[valid_sig_mask]))
            std_sigma_per_line.append(np.std(line_sigmas[valid_sig_mask]))
        else:
            mean_sigma_per_line.append(np.nan)
            std_sigma_per_line.append(np.nan)

        n_valid_kinematics_per_line.append(np.sum(valid_vel_mask & valid_sig_mask))
        n_valid_flux_per_line.append(np.sum(valid_flux_mask))

    # NEW: Add per-line kinematics statistics columns
    mapping_cols.append(fits.Column(name='MEAN_VELOCITY', unit='km/s', format='D', array=mean_velocity_per_line))
    mapping_cols.append(fits.Column(name='STD_VELOCITY', unit='km/s', format='D', array=std_velocity_per_line))
    mapping_cols.append(fits.Column(name='MEAN_SIGMA', unit='km/s', format='D', array=mean_sigma_per_line))
    mapping_cols.append(fits.Column(name='STD_SIGMA', unit='km/s', format='D', array=std_sigma_per_line))
    mapping_cols.append(fits.Column(name='N_VALID_KINEMATICS', format='J', array=n_valid_kinematics_per_line))
    mapping_cols.append(fits.Column(name='N_VALID_FLUX', format='J', array=n_valid_flux_per_line))

    mappingHDU = fits.BinTableHDU.from_columns(fits.ColDefs(mapping_cols))
    mappingHDU.name = 'LINE_MAPPING'

    # Add comprehensive statistics to mapping header
    mappingHDU.header['N_LINES_ALL'] = (len(all_line_names), 'Total lines in emission config')
    mappingHDU.header['N_FITTED'] = (len(line_categories['fitted']), 'Lines successfully fitted')
    mappingHDU.header['N_MASKED'] = (len(line_categories['masked']), 'Lines masked (sky lines, etc.)')
    mappingHDU.header['N_OUT_RANGE'] = (len(line_categories['out_of_range']), 'Lines outside wavelength range')
    mappingHDU.header['N_NO_WAVE'] = (len(line_categories['missing_wavelength']), 'Lines without wavelengths')

    # Add wavelength range info
    mappingHDU.header['LMIN_USED'] = (wavelength_range[0], 'Minimum wavelength used for fitting')
    mappingHDU.header['LMAX_USED'] = (wavelength_range[1], 'Maximum wavelength used for fitting')

    if z_in is not None:
        mappingHDU.header['Z_USED'] = (z_in, 'Redshift used for line and mask shifting')

    # Add masking information
    if emission_config and 'mask_regions' in emission_config:
        n_mask_regions = len(emission_config['mask_regions'])
        mappingHDU.header['N_MASK_REG'] = (n_mask_regions, 'Number of mask regions applied')

        # Count different types of mask regions
        sky_masks = sum(1 for mr in emission_config['mask_regions'] if 'sky' in mr['name'].lower())
        emission_masks = n_mask_regions - sky_masks
        mappingHDU.header['N_SKY_MASK'] = (sky_masks, 'Number of sky line masks')
        mappingHDU.header['N_EMI_MASK'] = (emission_masks, 'Number of emission line masks')

    # Add tie settings to mapping header (truncated for header space)
    for key, value in list(tie_settings.items())[:10]:  # Limit to prevent header overflow
        header_key = f'HIERARCH T_{key.upper()}'
        if isinstance(value, bool):
            mappingHDU.header[header_key] = (value, f'Tie setting: {key}')
        elif isinstance(value, list):
            # Build value string safely
            if len(value) <= 3:
                value_str = ', '.join([str(v) for v in value])
                if len(value_str) > 40:
                    value_str = f'{len(value)} items'
            else:
                value_str = f'{len(value)} items'
            mappingHDU.header[header_key] = (value_str, f'Tie: {key}')

    # Add comments about data interpretation - UPDATED with flux filtering info
    mappingHDU.header['COMMENT'] = 'LINE_STATUS describes why each line was/was not fitted'
    mappingHDU.header['COMMENT'] = 'FITTED: Line successfully fitted and has real measurements'
    mappingHDU.header['COMMENT'] = 'MASKED: Line excluded due to sky contamination, bad pixels, etc.'
    mappingHDU.header['COMMENT'] = 'OUT_OF_RANGE: Line outside fitting wavelength range'
    mappingHDU.header['COMMENT'] = 'NO_WAVELENGTH: Line has no wavelength defined in config'
    mappingHDU.header['COMMENT'] = 'Per-line kinematics: Each fitted line has individual V and sigma'
    # mappingHDU.header['COMMENT'] = 'Kinematics set to NaN for lines with flux ≤ 0 or NaN'
    mappingHDU.header['COMMENT'] = 'MEAN_VELOCITY/SIGMA: Average kinematics across all bins for each line'
    mappingHDU.header['COMMENT'] = 'STD_VELOCITY/SIGMA: Standard deviation of kinematics for each line'
    mappingHDU.header['COMMENT'] = 'N_VALID_KINEMATICS: Number of bins with valid kinematics per line'
    mappingHDU.header['COMMENT'] = 'N_VALID_FLUX: Number of bins with valid flux (>0 and finite) per line'
    mappingHDU.header['COMMENT'] = 'For single-component fits: all valid lines have identical kinematics'
    mappingHDU.header['COMMENT'] = 'For multi-component fits: lines tied by component have same kinematics'

    # Create HDU list and write mapping file
    HDUList = fits.HDUList([fits.PrimaryHDU(), mappingHDU])
    HDUList.writeto(outfits_mapping, overwrite=True)

    ExGalutil.prettyOutput_Done("Writing: " + rootname + '_emippxf_mapping.fits')
    logging.info("Wrote: " + outfits_mapping)

    if debug:
        print(f"Enhanced IFU emi kinematics output with per-line kinematics and flux filtering:")
        print(f"  Total lines in config: {len(all_line_names)}")
        print(f"  Fitted lines: {len(line_categories['fitted'])}")
        print(f"  Masked lines: {len(line_categories['masked'])} (sky lines, bad regions)")
        print(f"  Out-of-range lines: {len(line_categories['out_of_range'])}")
        print(f"  Missing wavelength: {len(line_categories['missing_wavelength'])}")

        # Calculate and print per-line kinematics statistics
        valid_velocities = [v for v in mean_velocity_per_line if not np.isnan(v)]
        valid_sigmas = [s for s in mean_sigma_per_line if not np.isnan(s)]

        if len(valid_velocities) > 0:
            print(f"  Velocity range across lines: {np.min(valid_velocities):.1f} to {np.max(valid_velocities):.1f} km/s")
        if len(valid_sigmas) > 0:
            print(f"  Sigma range across lines: {np.min(valid_sigmas):.1f} to {np.max(valid_sigmas):.1f} km/s")
        print(f"  Lines with valid kinematics: {len(valid_velocities)}/{len(all_line_names)}")

        # Print flux filtering statistics
        total_valid_flux = sum(n_valid_flux_per_line)
        total_measurements = len(ubins) * len([i for i in fitted_line_indices if i >= 0])
        if total_measurements > 0:
            print(f"  Valid flux measurements: {total_valid_flux}/{total_measurements} ({total_valid_flux/total_measurements*100:.1f}%)")

        if emission_config and 'mask_regions' in emission_config:
            n_mask_regions = len(emission_config['mask_regions'])
            sky_masks = sum(1 for mr in emission_config['mask_regions'] if 'sky' in mr['name'].lower())
            print(f"  Applied {n_mask_regions} mask regions ({sky_masks} sky, {n_mask_regions-sky_masks} emission)")

        print(f"  Files created: _emippxf.fits (with flux-filtered per-line V/Sigma), _emippxfSpec.fits, _emippxf_mapping.fits")

    return True




# def save_emi_kinematics_emippxf_format(rootname, configs, outdir, results, mc_errors, formal_errors,
#                                        lambdaR, bestfit, logLam, spectra, error, goodpixels,
#                                        emi_bestfit, emission_lines, line_names, line_wavelengths,
#                                        npix, ubins, tie_settings, emission_config, z_in=None, z_err=None, group_order=None, debug=False):
#     """
#     UPDATED VERSION: Save emi kinematics results in EMIPPXF-compatible format with proper masking and per-line kinematics.
#     FLUX FILTERING: Kinematics are set to NaN for lines with flux ≤ 0 or NaN.
#     Creates additional files that match the EMIPPXF output structure for database compatibility.
#     Updated to handle new comprehensive tie settings format, proper line categorization, and per-line kinematics.

#     Parameters:
#     -----------
#     emission_config : dict
#         The loaded pPXF emission configuration containing emission_lines, tie_settings, etc.
#     line_wavelengths : dict
#         Dictionary mapping line names to rest wavelengths
#     All other parameters same as the original save_emi_kinematics function
#     """

#     if debug:
#         print("Creating EMIPPXF-compatible output with proper masking, per-line kinematics, and flux filtering...")

#     # Determine wavelength range used for fitting
#     if 'LMIN_EMI' in configs and 'LMAX_EMI' in configs:
#         wavelength_range_config = (configs['LMIN_EMI'], configs['LMAX_EMI'])
#     elif 'LMIN_PPXF' in configs and 'LMAX_PPXF' in configs:
#         wavelength_range_config = (configs['LMIN_PPXF'], configs['LMAX_PPXF'])
#     else:
#         # Fallback to data range
#         wavelength_range_config = (np.nan, np.nan)

#     # also get the wavelenght range from the spectra itself
#     lam = np.exp(logLam)
#     wavelength_range = (lam.min(), lam.max())

#     # Get ALL lines from emission config
#     assert emission_config is not None and 'emission_lines' in emission_config, 'no config file found to build the all_line_names'
#     all_line_names = list(emission_config['emission_lines'].keys())
#     all_line_wavelengths = emission_config['emission_lines']
#     if debug:
#         print(f"Processing ALL {len(all_line_names)} lines from emission config:")
#         print(f"  - Fitted: {len(line_categories['fitted'])}")
#         print(f"  - Masked: {len(line_categories['masked'])}")
#         print(f"  - Out of range: {len(line_categories['out_of_range'])}")
#         print(f"  - Missing wavelength: {len(line_categories['missing_wavelength'])}")

#     # Categorize ALL lines properly using the same function as main save
#     line_categories = categorize_emission_lines(
#         emission_config,
#         all_line_names,
#         wavelength_range,
#         wavelength_range_config
#     )

#     # Create mapping from fitted lines to all lines
#     fitted_line_indices = []
#     line_status = []  # Track status: 'FITTED', 'MASKED', 'OUT_OF_RANGE', 'NO_WAVELENGTH'

#     for name in all_line_names:
#         if name in line_categories['fitted']:
#             fitted_line_indices.append(all_line_names.index(name))
#             line_status.append('FITTED')
#         elif name in line_categories['masked']:
#             fitted_line_indices.append(-1)  # Not fitted
#             line_status.append('MASKED')
#         elif name in line_categories['out_of_range']:
#             fitted_line_indices.append(-1)  # Not fitted
#             line_status.append('OUT_OF_RANGE')
#         elif name in line_categories['missing_wavelength']:
#             fitted_line_indices.append(-1)  # Not fitted
#             line_status.append('NO_WAVELENGTH')
#         else:
#             fitted_line_indices.append(-1)  # Not fitted - other reason
#             line_status.append('OTHER')

#     # Set flux units
#     if configs['sens_corr']:
#         flux_unit_str = '%2e erg/(s cm**2 Angstrom)' % (configs['funits'])
#         ivar_unit_str = '%2e cm**4 Angstrom**2 /(s**2 erg**2)' % (configs['funits']**-2)
#     else:
#         flux_unit_str = 'count'
#         ivar_unit_str = '1/count**2'
#     wave_unit_str = 'Angstrom'

#     # Extract emission line properties from results - for ALL lines
#     emission_amplitudes_all = np.full((len(ubins), len(all_line_names)), np.nan)
#     emission_amplitude_errors_all = np.full((len(ubins), len(all_line_names)), np.nan)
#     emission_aon_all = np.full((len(ubins), len(all_line_names)), np.nan)
#     emission_fluxes_all = np.full((len(ubins), len(all_line_names)), np.nan)
#     emission_flux_errors_all = np.full((len(ubins), len(all_line_names)), np.nan)

#     # NEW: Extract per-line kinematics - for ALL lines
#     line_velocities_all = np.full((len(ubins), len(all_line_names)), np.nan)
#     line_sigmas_all = np.full((len(ubins), len(all_line_names)), np.nan)
#     line_velocity_errors_all = np.full((len(ubins), len(all_line_names)), np.nan)
#     line_sigma_errors_all = np.full((len(ubins), len(all_line_names)), np.nan)

#     # Extract traditional emi kinematics (primary component for EMIPPXF compatibility)
#     emi_velocities = np.zeros(len(ubins))
#     emi_sigmas = np.zeros(len(ubins))
#     emi_v_errors = np.zeros(len(ubins))
#     emi_s_errors = np.zeros(len(ubins))

#     for i, bin_id in enumerate(ubins):
#         # Extract primary emi kinematics (for EMIPPXF compatibility)
#         if isinstance(results[i]["emi"], np.ndarray) and len(results[i]["emi"]) >= 2:
#             emi_velocities[i] = results[i]["emi"][0]
#             emi_sigmas[i] = results[i]["emi"][1]
#         elif isinstance(results[i]["emi"], list) and len(results[i]["emi"]) > 0:
#             if isinstance(results[i]["emi"][0], np.ndarray) and len(results[i]["emi"][0]) >= 2:
#                 emi_velocities[i] = results[i]["emi"][0][0]
#                 emi_sigmas[i] = results[i]["emi"][0][1]

#         # Extract primary emi errors
#         if isinstance(mc_errors[i]["emi"], np.ndarray) and len(mc_errors[i]["emi"]) >= 2:
#             emi_v_errors[i] = mc_errors[i]["emi"][0]
#             emi_s_errors[i] = mc_errors[i]["emi"][1]
#         elif isinstance(mc_errors[i]["emi"], list) and len(mc_errors[i]["emi"]) > 0:
#             if isinstance(mc_errors[i]["emi"][0], np.ndarray) and len(mc_errors[i]["emi"][0]) >= 2:
#                 emi_v_errors[i] = mc_errors[i]["emi"][0][0]
#                 emi_s_errors[i] = mc_errors[i]["emi"][0][1]

#         # Extract emission line properties for ALL lines
#         if isinstance(mc_errors[i], dict):
#             for j, line_name in enumerate(all_line_names):
#                 try:
#                     fitted_line_position = line_names.index(line_name)
#                     if fitted_line_position >= 0:  # Line was fitted

#                         # FIRST: Extract flux to check validity
#                         current_flux = np.nan
#                         if isinstance(emission_lines[i], np.ndarray) and len(emission_lines[i]) > fitted_line_position:
#                             current_flux = emission_lines[i][fitted_line_position]
#                             emission_fluxes_all[i, j] = current_flux

#                         # FLUX CONDITION: Only extract kinematics if flux > 0 and finite
#                         flux_is_valid = (not np.isnan(current_flux) and current_flux is not None and current_flux > 1e-10 and line_name in line_categories['fitted'])


#                         if not flux_is_valid:
#                             emission_fluxes_all[i, j] = np.nan

#                         if flux_is_valid:

#                             # Extract flux errors
#                             if 'fluxes' in mc_errors[i] and isinstance(mc_errors[i]['fluxes'], np.ndarray):
#                                 if len(mc_errors[i]['fluxes']) > fitted_line_position:
#                                     emission_flux_errors_all[i, j] = mc_errors[i]['fluxes'][fitted_line_position]

#                             # Extract amplitudes
#                             if 'amplitudes' in mc_errors[i] and isinstance(mc_errors[i]['amplitudes'], np.ndarray):
#                                 if len(mc_errors[i]['amplitudes']) > fitted_line_position:
#                                     emission_amplitudes_all[i, j] = mc_errors[i]['amplitudes'][fitted_line_position]

#                             # Extract amplitude errors
#                             if 'amplitude_errors' in mc_errors[i] and isinstance(mc_errors[i]['amplitude_errors'], np.ndarray):
#                                 if len(mc_errors[i]['amplitude_errors']) > fitted_line_position:
#                                     emission_amplitude_errors_all[i, j] = mc_errors[i]['amplitude_errors'][fitted_line_position]

#                             # Extract AON
#                             if 'aon' in mc_errors[i] and isinstance(mc_errors[i]['aon'], np.ndarray):
#                                 if len(mc_errors[i]['aon']) > fitted_line_position:
#                                     emission_aon_all[i, j] = mc_errors[i]['aon'][fitted_line_position]

#                         # NEW: Extract per-line kinematics ONLY if flux is valid
#                         if flux_is_valid:
#                             if 'line_velocities' in mc_errors[i] and isinstance(mc_errors[i]['line_velocities'], np.ndarray):
#                                 if len(mc_errors[i]['line_velocities']) > fitted_line_position:
#                                     line_velocities_all[i, j] = mc_errors[i]['line_velocities'][fitted_line_position]

#                             if 'line_sigmas' in mc_errors[i] and isinstance(mc_errors[i]['line_sigmas'], np.ndarray):
#                                 if len(mc_errors[i]['line_sigmas']) > fitted_line_position:
#                                     line_sigmas_all[i, j] = mc_errors[i]['line_sigmas'][fitted_line_position]

#                             if 'line_velocity_errors' in mc_errors[i] and isinstance(mc_errors[i]['line_velocity_errors'], np.ndarray):
#                                 if len(mc_errors[i]['line_velocity_errors']) > fitted_line_position:
#                                     line_velocity_errors_all[i, j] = mc_errors[i]['line_velocity_errors'][fitted_line_position]

#                             if 'line_sigma_errors' in mc_errors[i] and isinstance(mc_errors[i]['line_sigma_errors'], np.ndarray):
#                                 if len(mc_errors[i]['line_sigma_errors']) > fitted_line_position:
#                                     line_sigma_errors_all[i, j] = mc_errors[i]['line_sigma_errors'][fitted_line_position]
#                 except:
#                     # Non-fitted lines (masked, out-of-range, etc.) remain np.nan (already initialized)
#                     pass

#     # ========================
#     # POST-PROCESSING: Apply flux filtering to kinematics
#     if debug:
#         n_flux_filtered = 0
#         n_total_fitted = 0

#     for i in range(len(ubins)):
#         for j in range(len(all_line_names)):
#             current_flux = emission_fluxes_all[i, j]
#             is_fitted = fitted_line_indices[j] >= 0

#             if is_fitted:
#                 if debug:
#                     n_total_fitted += 1

#                 # If flux is invalid (NaN or ≤ 0), set all kinematics to NaN
#                 if not (np.isfinite(current_flux) and current_flux > 0):
#                     line_velocities_all[i, j] = np.nan
#                     line_sigmas_all[i, j] = np.nan
#                     line_velocity_errors_all[i, j] = np.nan
#                     line_sigma_errors_all[i, j] = np.nan

#                     if debug:
#                         n_flux_filtered += 1

#     if debug and n_total_fitted > 0:
#         print(f"  Flux filtering: {n_flux_filtered}/{n_total_fitted} fitted line measurements had invalid flux (set kinematics to NaN)")

#     # Create EMIPPXF-style emission setup data from pPXF configuration
#     # Use ALL lines from emission_config, not just those fitted
#     nlines_all = len(all_line_names)

#     # Build emission setup arrays that match EMIPPXF structure (for ALL lines)
#     iis = list(range(nlines_all))  # Line indices
#     names = all_line_names  # ALL line names from config
#     lambdas = [all_line_wavelengths.get(name, 0.0) for name in all_line_names]  # ALL rest wavelengths

#     # CORRECT Actions based on actual categorization - not just fitted vs not fitted
#     actions = []
#     for name in all_line_names:
#         if name in line_categories['fitted']:
#             actions.append('f')  # Fitted
#         elif name in line_categories['masked']:
#             actions.append('m')  # Masked (EMIPPXF convention for sky lines, bad regions)
#         elif name in line_categories['out_of_range']:
#             actions.append('o')  # Outside range (custom - could use 'm' if preferred)
#         elif name in line_categories['missing_wavelength']:
#             actions.append('n')  # No wavelength (custom)
#         else:
#             actions.append('u')  # Unknown/other reason

#     # Kinds - determine from line groups and tie settings (for ALL lines)
#     kinds = []
#     # handle None mode for group_order
#     if group_order is None:
#         group_order = []

#     # Assuming this is inside a loop where 'name' is the current line name
#     for name in all_line_names:
#         if tie_settings.get('tie_all', False):
#             kinds.append('all_tied')
#         # Check new-style tie groups dynamically
#         elif any(name in tie_settings.get(group, []) for group in group_order):
#             # Find which group contains this line
#             for group_name in group_order:
#                 if group_name in tie_settings and name in tie_settings.get(group_name, []):
#                     kinds.append(group_name)
#                     break
#         # Legacy groups for backward compatibility
#         elif name in tie_settings.get("[OIII]_5006.77", []) or name == "[OIII]_5006.77":
#             kinds.append('[OIII]_group')
#         elif name in tie_settings.get("[NII]_6583.34", []) or name == "[NII]_6583.34":
#             kinds.append('[NII]_group')
#         else:
#             kinds.append('independent')

#     # A values: amplitude ratios (from line_ratios if available, otherwise 1.0) - for ALL lines
#     aas = []
#     if emission_config and 'line_ratios' in emission_config:
#         line_ratios = emission_config['line_ratios']
#         for name in all_line_names:
#             if name in line_ratios:
#                 aas.append(line_ratios[name]['ratio'])
#             else:
#                 aas.append(1.0)
#     else:
#         aas = [1.0] * nlines_all

#     # V and S values: use kinematic settings or defaults
#     vs = []
#     ss = []
#     if emission_config and 'kinematic_settings' in emission_config:
#         kin_settings = emission_config['kinematic_settings']
#         default_v = kin_settings.get('initial_velocity', 0.0)
#         default_s = kin_settings.get('initial_dispersion', 10.0)
#     else:
#         default_v = 0.0
#         default_s = 10.0

#     vs = [default_v] * nlines_all
#     ss = [default_s] * nlines_all

#     # Fit flags: properly reflect actual fitting status
#     ffits = []
#     for name in all_line_names:
#         if name in line_categories['fitted']:
#             ffits.append('True')   # Actually fitted
#         else:
#             ffits.append('False')  # Not fitted (any reason)

#     # AoN values: use measurement settings or default (for ALL lines)
#     if emission_config and 'measurement_settings' in emission_config:
#         default_aon = emission_config['measurement_settings'].get('aon_threshold', 4.0)
#     else:
#         default_aon = 4.0
#     aons = [default_aon] * nlines_all

#     # ========================
#     # SAVE EMIPPXF-style results with proper masking information
#     outfits = outdir + rootname + '_emippxf_BIN.fits'
#     ExGalutil.prettyOutput_Running("Writing: " + rootname + '_emippxf_BIN.fits')

#     # Primary HDU
#     priHDU = fits.PrimaryHDU()

#     # Extension 1: Table HDU with emission_setup data (EMIPPXF equivalent) - ALL LINES WITH PROPER ACTIONS
#     cols = []
#     cols.append(fits.Column(name='LINE_ID', format='D', array=np.arange(nlines_all)))
#     cols.append(fits.Column(name='i', format='D', array=np.array(iis)))
#     cols.append(fits.Column(name='name', format='15A', array=np.array(names)))
#     cols.append(fits.Column(name='_lambda', format='D', array=np.array(lambdas)))
#     cols.append(fits.Column(name='action', format='15A', array=np.array(actions)))
#     cols.append(fits.Column(name='kind', format='15A', array=np.array(kinds)))
#     cols.append(fits.Column(name='a', format='D', array=np.array(aas)))
#     cols.append(fits.Column(name='v', format='D', array=np.array(vs)))
#     cols.append(fits.Column(name='s', format='D', array=np.array(ss)))
#     cols.append(fits.Column(name='fit', format='15A', array=np.array(ffits)))
#     cols.append(fits.Column(name='aon', format='D', array=np.array(aons)))
#     emission_setup_HDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
#     emission_setup_HDU.name = 'EMISSION_SETUP'

#     # Add headers to emission setup with proper masking statistics
#     emission_setup_HDU.header['N_LINES_ALL'] = (nlines_all, 'Total lines in emission config')
#     emission_setup_HDU.header['N_FITTED'] = (len(line_categories['fitted']), 'Lines successfully fitted (action=f)')
#     emission_setup_HDU.header['N_MASKED'] = (len(line_categories['masked']), 'Lines masked (action=m)')
#     emission_setup_HDU.header['N_OUT_RANGE'] = (len(line_categories['out_of_range']), 'Lines outside range (action=o)')
#     emission_setup_HDU.header['N_NO_WAVE'] = (len(line_categories['missing_wavelength']), 'Lines no wavelength (action=n)')

#     # Add action code meanings
#     emission_setup_HDU.header['COMMENT'] = 'Action codes: f=fitted, m=masked, o=out_of_range, n=no_wavelength'
#     emission_setup_HDU.header['COMMENT'] = 'Masked lines were excluded due to sky contamination, bad pixels, etc.'
#     emission_setup_HDU.header['COMMENT'] = 'Out-of-range lines were outside fitting wavelength limits'
#     emission_setup_HDU.header['COMMENT'] = 'Only lines with action=f have measured values in main table'
#     emission_setup_HDU.header['COMMENT'] = 'Kinematics are NaN for lines with flux less than 0 or NaN'

#     # Extension 2: Table HDU with EMIPPXF-style output data - FITTED LINES ONLY WITH PER-LINE KINEMATICS
#     cols = []
#     cols.append(fits.Column(name='BIN_ID', format='J', array=ubins))

#     # Add redshift information
#     if z_in is not None:
#         z_array = np.full(len(ubins), z_in)
#         cols.append(fits.Column(name='Z_INPUT', format='D', array=z_array))
#         if z_err is not None:
#             z_err_array = np.full(len(ubins), z_err)
#             cols.append(fits.Column(name='ERR_Z_INPUT', format='D', array=z_err_array))

#     # For each FITTED emission line, add EMIPPXF-style columns with per-line kinematics
#     # Note: Only include lines that were actually fitted, not all lines
#     for iin, line_name in enumerate(line_names):
#         # Find index in all_line_names for this fitted line
#         all_line_idx = all_line_names.index(line_name) if line_name in all_line_names else -1

#         # EMIPPXF column naming: FLUX_<name>_<wavelength>
#         cols.append(fits.Column(name='FLUX_' + line_name ,
#                                format='D', array=emission_fluxes_all[:, all_line_idx] if all_line_idx >= 0 else np.full(len(ubins), np.nan)))
#         cols.append(fits.Column(name='AMPL_' + line_name ,
#                                format='D', array=emission_amplitudes_all[:, all_line_idx] if all_line_idx >= 0 else np.full(len(ubins), np.nan)))

#         # NEW: Use per-line kinematics instead of shared emi kinematics (flux-filtered)
#         cols.append(fits.Column(name='V_' + line_name ,
#                                format='D', array=line_velocities_all[:, all_line_idx] if all_line_idx >= 0 else np.full(len(ubins), np.nan)))
#         cols.append(fits.Column(name='SIGMA_' + line_name ,
#                                format='D', array=line_sigmas_all[:, all_line_idx] if all_line_idx >= 0 else np.full(len(ubins), np.nan)))

#         cols.append(fits.Column(name='AON_' + line_name ,
#                                format='D', array=emission_aon_all[:, all_line_idx] if all_line_idx >= 0 else np.full(len(ubins), np.nan)))

#         # Error columns
#         cols.append(fits.Column(name='ERR_FLUX_' + line_name ,
#                                format='D', array=emission_flux_errors_all[:, all_line_idx] if all_line_idx >= 0 else np.full(len(ubins), np.nan)))
#         cols.append(fits.Column(name='ERR_AMPL_' + line_name ,
#                                format='D', array=emission_amplitude_errors_all[:, all_line_idx] if all_line_idx >= 0 else np.full(len(ubins), np.nan)))

#         # NEW: Per-line kinematic errors (flux-filtered)
#         cols.append(fits.Column(name='ERR_V_' + line_name ,
#                                format='D', array=line_velocity_errors_all[:, all_line_idx] if all_line_idx >= 0 else np.full(len(ubins), np.nan)))
#         cols.append(fits.Column(name='ERR_SIGMA_' + line_name ,
#                                format='D', array=line_sigma_errors_all[:, all_line_idx] if all_line_idx >= 0 else np.full(len(ubins), np.nan)))



#     # No EBMV columns in emi analysis (that's for stellar reddening)
#     # But add placeholder for compatibility
#     len_EBMV = 1
#     nan_array_EBMV = np.full((len_EBMV, len(ubins)), np.nan)
#     cols.append(fits.Column(name='EBMV', format=str(len_EBMV) + 'D', array=nan_array_EBMV.T))
#     cols.append(fits.Column(name='ERR_EBMV', format=str(len_EBMV) + 'D', array=nan_array_EBMV.T))

#     # Create the main data table
#     dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
#     dataHDU.name = 'EMIPPXF_TABLE'
#     dataHDU.header['SAMPLING'] = (1, 'Sampling mode (0: linear, 1: logarithmic)')

#     # Add headers similar to EMIPPXF with tie settings and masking info
#     dataHDU.header['CNF_EMI'] = (configs['CONFIG_FILE'], 'Configs. Filename used by emi analysis')
#     dataHDU.header['EMI_LEV'] = ('BIN', 'emi analysis working level: BIN')
#     dataHDU.header['EMI_SIG'] = (configs.get('SIGMA', 150.0), 'Initial guess for SIGMA in emi analysis')
#     dataHDU.header['EMI_FILE'] = (configs.get('EMI_FILE', 'emission_lines.py'), 'Emission Lines config. filename')
#     dataHDU.header['FOR_ERR'] = (configs.get('MC_EMI', 30) > 0, 'Derive errors on the emission-line analysis')

#     # Add basic tie settings to header
#     dataHDU.header['TIE_ALL'] = (tie_settings.get('tie_all', False), 'All lines tied together')

#     # Add information about new tie groups
#     if group_order is not None and isinstance(group_order, list):
#         new_groups = [(group_name, group_name.upper()) for group_name in group_order]

#         active_new_groups = []
#         for group_name, _ in new_groups:
#             if tie_settings.get(group_name):
#                 group_lines = tie_settings[group_name]
#                 present_lines = [line for line in group_lines if line in line_names]
#                 if present_lines:
#                     active_new_groups.append(f"{group_name}({len(present_lines)})")

#         if active_new_groups:
#             # Store as comma-separated string, truncate if too long
#             groups_str = ', '.join(active_new_groups)
#             if len(groups_str) > 60:
#                 groups_str = f"{len(active_new_groups)} active groups"
#             dataHDU.header['NEW_GROUPS'] = (groups_str, 'Active new-style tie groups')

#     # Estimate and report computational reduction
#     estimated_components, _ = estimate_component_count(line_names, tie_settings, group_order=group_order)
#     computational_reduction = (len(line_names) - estimated_components) / len(line_names) * 100
#     dataHDU.header['N_COMP_EST'] = (estimated_components, 'Estimated kinematic components')
#     dataHDU.header['COMP_REDUC'] = (computational_reduction, 'Computational reduction (%)')

#     # Add wavelength range and masking info
#     dataHDU.header['LW_EMI'] = (wavelength_range_config[0], 'Min wavelength (OBS.) used by emi analysis')
#     dataHDU.header['HW_EMI'] = (wavelength_range_config[1], 'Max wavelength (OBS.) used by emi analysis')

#     if z_in is not None:
#         dataHDU.header['Z_EMI'] = (z_in, 'Redshift used for line and mask shifting')

#     # Add masking statistics
#     dataHDU.header['N_LINES_ALL'] = (nlines_all, 'Total lines in emission config')
#     dataHDU.header['N_FITTED'] = (len(line_categories['fitted']), 'Lines fitted (have data columns)')
#     dataHDU.header['N_MASKED'] = (len(line_categories['masked']), 'Lines masked (sky, bad pixels)')
#     dataHDU.header['N_OUT_RANGE'] = (len(line_categories['out_of_range']), 'Lines outside wavelength range')
#     dataHDU.header['N_NO_WAVE'] = (len(line_categories['missing_wavelength']), 'Lines without wavelengths')

#     # Add masking details
#     if emission_config and 'mask_regions' in emission_config:
#         n_mask_regions = len(emission_config['mask_regions'])
#         sky_masks = sum(1 for mr in emission_config['mask_regions'] if 'sky' in mr['name'].lower())
#         emission_masks = n_mask_regions - sky_masks
#         dataHDU.header['N_MASK_REG'] = (n_mask_regions, 'Number of mask regions applied')
#         dataHDU.header['N_SKY_MASK'] = (sky_masks, 'Sky line mask regions')
#         dataHDU.header['N_EMI_MASK'] = (emission_masks, 'Emission line mask regions')

#     dataHDU.header['TEMPL'] = (configs['SSP_LIB'], 'The library of spectral template')
#     dataHDU.header['PPXF_V'] = (PPXF_VERSION, 'PPXF version used for emi analysis')
#     dataHDU.header['APSVERS'] = (APSVERS, 'APS version')
#     dataHDU.header['CSB_EMI'] = (configs['stitched'], 'Combines Spectral Bands Status for emi analysis')

#     # Keep the basename of input file (infiles)
#     for n_province, province in enumerate(configs['infiles']):
#         dataHDU.header['APSREF_%d' % (n_province)] = (os.path.basename(province), 'L1 reference file')

#     # Add emission line setup in header (similar to EMIPPXF) - ALL LINES WITH STATUS
#     dataHDU.header['NGLLINES'] = (len(all_line_names), 'Number of Lines in configuration')
#     dataHDU.header['NFITLINES'] = (len(line_names), 'Number of Lines actually fitted')
#     dataHDU.header['NMASKED'] = (len(line_categories['masked']), 'Number of Lines masked')
#     dataHDU.header['NOUTRANGE'] = (len(line_categories['out_of_range']), 'Number of Lines outside range')

#     # Add summary of tie group types used
#     unique_kinds = list(set(kinds))
#     tie_types_str = ', '.join(unique_kinds[:10])  # Limit length
#     if len(tie_types_str) > 60:
#         tie_types_str = f"{len(unique_kinds)} tie types"
#     dataHDU.header['TIE_TYPES'] = (tie_types_str, 'Types of tie groups used')

#     # NEW: Add per-line kinematics and flux filtering information
#     dataHDU.header['PER_LINE_KIN'] = (True, 'Per-line kinematics available (V_<line>, SIGMA_<line>)')
#     dataHDU.header['FLUX_FILTER'] = (True, 'Kinematics set to NaN for lines with flux less than 0 or NaN')
#     dataHDU.header['COMMENT'] = 'V_<line>/SIGMA_<line>: Individual line kinematics from component tying'
#     dataHDU.header['COMMENT'] = 'Kinematics are NaN for lines with flux less than 0 or NaN'
#     dataHDU.header['COMMENT'] = 'For single-component fits: all valid lines have identical kinematics'
#     dataHDU.header['COMMENT'] = 'For multi-component fits: V_<line> varies by component assignment'

#     # Add individual line information (limit to prevent header overflow) - WITH STATUS
#     for nii, line_name in enumerate(all_line_names):
#         if nii < 200:  # Reduced limit to accommodate status info
#             status_code = actions[nii]  # f, m, o, n
#             dataHDU.header['GLID_%s' % (nii)] = '%30s' % (nii)
#             dataHDU.header['GLI_%s' % (nii)] = '%30s' % (iis[nii])
#             dataHDU.header['GLNAM_%s' % (nii)] = '%30s' % (names[nii])
#             dataHDU.header['GLLAM_%s' % (nii)] = '%30s' % (lambdas[nii])
#             dataHDU.header['GLACT_%s' % (nii)] = '%30s' % (actions[nii])
#             dataHDU.header['GLKIN_%s' % (nii)] = '%30s' % (kinds[nii])
#             dataHDU.header['GLA_%s' % (nii)] = '%30s' % (aas[nii])
#             dataHDU.header['GLV_%s' % (nii)] = '%30s' % (vs[nii])
#             dataHDU.header['GLS_%s' % (nii)] = '%30s' % (ss[nii])
#             dataHDU.header['GLFIT_%s' % (nii)] = '%30s' % (ffits[nii])
#             dataHDU.header['GLAON_%s' % (nii)] = '%30s' % (aons[nii])
#             dataHDU.header['GLSTAT_%s' % (nii)] = '%30s' % (status_code)  # NEW: Status code
#         elif nii == 200:
#             # Add comment about truncation
#             dataHDU.header['COMMENT'] = f'Line info truncated at 200/{len(all_line_names)} lines'
#             dataHDU.header['COMMENT'] = f'See EMISSION_SETUP extension for complete line list'

#     # Create HDU list and write to file
#     HDUList = fits.HDUList([priHDU, emission_setup_HDU, dataHDU])
#     HDUList.writeto(outfits, overwrite=True)

#     ExGalutil.prettyOutput_Done("Writing: " + rootname + '_emippxf_BIN.fits')
#     logging.info("Wrote: " + outfits)

#     # ========================
#     # SAVE EMIPPXF-style BESTFIT spectra with masking info
#     outfits_spec = outdir + rootname + '_emippxf_spec_BIN.fits'
#     ExGalutil.prettyOutput_Running("Writing: " + rootname + '_emippxf_spec_BIN.fits')

#     # Primary HDU
#     priHDU = fits.PrimaryHDU()

#     # Table HDU with spectra (EMIPPXF equivalent)
#     cols = []
#     loglam_tile = np.tile(logLam, (len(ubins), 1))

#     cols.append(fits.Column(name='BIN_ID', format='J', array=ubins))
#     cols.append(fits.Column(name='LOGLAM_EMI', unit=wave_unit_str, format=str(npix) + 'D', array=loglam_tile))
#     cols.append(fits.Column(name='FLUX_EMI', unit=flux_unit_str, format=str(npix) + 'D', array=np.transpose(spectra)))
#     cols.append(fits.Column(name='ERROR_EMI', unit=flux_unit_str, format=str(npix) + 'D', array=np.transpose(error)))
#     cols.append(fits.Column(name='MODEL_EMI', unit=flux_unit_str, format=str(npix) + 'D', array=bestfit))
#     cols.append(fits.Column(name='EMISSION_EMI', unit=flux_unit_str, format=str(npix) + 'D', array=emi_bestfit))

#     # Calculate emission-subtracted spectrum for EMIPPXF compatibility
#     flux_clean = np.transpose(spectra) - emi_bestfit
#     model_clean = bestfit - emi_bestfit
#     cols.append(fits.Column(name='FLUX_CLEAN_EMI', unit=flux_unit_str, format=str(npix) + 'D', array=flux_clean))
#     cols.append(fits.Column(name='MODEL_CLEAN_EMI', unit=flux_unit_str, format=str(npix) + 'D', array=model_clean))
#     cols.append(fits.Column(name='GOODPIX_EMI', format=str(npix) + 'J', array=goodpixels))

#     specHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
#     specHDU.name = 'EMIPPXF_SPEC'

#     # Add headers with tie settings and masking information
#     specHDU.header['CNF_EMI'] = (configs['CONFIG_FILE'], 'Configs. Filename used by emi analysis')
#     specHDU.header['EMI_LEV'] = ('BIN', 'emi analysis working level: BIN')
#     specHDU.header['SAMPLING'] = (1, 'Sampling mode (0: linear, 1: logarithmic)')
#     specHDU.header['TEMPL'] = (configs['SSP_LIB'], 'The library of spectral template')
#     specHDU.header['PPXF_V'] = (PPXF_VERSION, 'PPXF version used for emi analysis')
#     specHDU.header['APSVERS'] = (APSVERS, 'APS version')
#     specHDU.header['CSB_EMI'] = (configs['stitched'], 'Combines Spectral Bands Status for emi analysis')

#     # Add tie mode information based on number of groups
#     if group_order is not None and isinstance(group_order, list):
#         new_groups = [(group_name, group_name.upper()) for group_name in group_order]
#         active_new_groups = []
#         for group_name, _ in new_groups:
#             if tie_settings.get(group_name):
#                 group_lines = tie_settings[group_name]
#                 present_lines = [line for line in group_lines if line in line_names]
#                 if present_lines:
#                     active_new_groups.append(f"{group_name}({len(present_lines)})")
#     else:
#         active_new_groups = []

#     if len(active_new_groups) == 0:
#         tie_mode_used = 'legacy'
#     elif len(active_new_groups) > 3:
#         tie_mode_used = 'optimized'
#     else:
#         tie_mode_used = 'aggressive'

#     specHDU.header['TIE_MODE'] = (tie_mode_used, 'Tie settings mode used')
#     specHDU.header['N_GROUPS'] = (len(active_new_groups), 'Number of active tie groups')

#     # Add masking statistics to spectra file
#     specHDU.header['N_LINES_ALL'] = (nlines_all, 'Total lines in emission config')
#     specHDU.header['N_FITTED'] = (len(line_categories['fitted']), 'Lines fitted (in EMIPPXF_TABLE)')
#     specHDU.header['N_MASKED'] = (len(line_categories['masked']), 'Lines masked (sky, bad pixels)')
#     specHDU.header['N_OUT_RANGE'] = (len(line_categories['out_of_range']), 'Lines outside wavelength range')
#     specHDU.header['N_NO_WAVE'] = (len(line_categories['missing_wavelength']), 'Lines without wavelengths')

#     # Add wavelength range and redshift info
#     specHDU.header['LMIN_USED'] = (wavelength_range_config[0], 'Min wavelength used for fitting')
#     specHDU.header['LMAX_USED'] = (wavelength_range_config[1], 'Max wavelength used for fitting')

#     if z_in is not None:
#         specHDU.header['Z_USED'] = (z_in, 'Redshift used for line and mask shifting')

#     # Add masking details
#     if emission_config and 'mask_regions' in emission_config:
#         n_mask_regions = len(emission_config['mask_regions'])
#         sky_masks = sum(1 for mr in emission_config['mask_regions'] if 'sky' in mr['name'].lower())
#         emission_masks = n_mask_regions - sky_masks
#         specHDU.header['N_MASK_REG'] = (n_mask_regions, 'Number of mask regions applied')
#         specHDU.header['N_SKY_MASK'] = (sky_masks, 'Sky line mask regions')
#         specHDU.header['N_EMI_MASK'] = (emission_masks, 'Emission line mask regions')

#     # NEW: Add per-line kinematics and flux filtering information to spectra file
#     specHDU.header['PER_LINE_KIN'] = (True, 'Per-line kinematics available in EMIPPXF_TABLE')
#     specHDU.header['FLUX_FILTER'] = (True, 'Kinematics NaN for lines with flux less than 0 or NaN')

#     # Add comments about data interpretation
#     specHDU.header['COMMENT'] = 'EMISSION_EMI contains fitted emission line model'
#     specHDU.header['COMMENT'] = 'FLUX_CLEAN_EMI = FLUX_EMI - EMISSION_EMI'
#     specHDU.header['COMMENT'] = 'MODEL_CLEAN_EMI = MODEL_EMI - EMISSION_EMI'
#     specHDU.header['COMMENT'] = 'Only fitted lines contribute to EMISSION_EMI model'
#     specHDU.header['COMMENT'] = 'Masked lines were excluded from fitting'
#     specHDU.header['COMMENT'] = 'Per-line kinematics: V_<line>, SIGMA_<line> in EMIPPXF_TABLE'
#     specHDU.header['COMMENT'] = 'Kinematics are NaN for lines with flux less than 0 or NaN'

#     for n_province, province in enumerate(configs['infiles']):
#         specHDU.header['APSREF_%d' % (n_province)] = (os.path.basename(province), 'L1 reference file')

#     HDUList = fits.HDUList([priHDU, specHDU])
#     HDUList.writeto(outfits_spec, overwrite=True)

#     ExGalutil.prettyOutput_Done("Writing: " + rootname + '_emippxf_spec_BIN.fits')
#     logging.info("Wrote: " + outfits_spec)

#     # ========================
#     # SAVE ADDITIONAL EMIPPXF-STYLE LINE MAPPING FILE WITH PER-LINE KINEMATICS
#     outfits_lines = outdir + rootname + '_emippxf_lines_BIN.fits'
#     ExGalutil.prettyOutput_Running("Writing: " + rootname + '_emippxf_lines_BIN.fits')

#     # Primary HDU
#     priHDU = fits.PrimaryHDU()

#     # Create detailed line mapping table showing ALL lines and their status
#     mapping_cols = []
#     mapping_cols.append(fits.Column(name='LINE_ID', format='J', array=np.arange(nlines_all)))
#     mapping_cols.append(fits.Column(name='LINE_NAME', format='A20', array=all_line_names))
#     mapping_cols.append(fits.Column(name='WAVELENGTH', format='D', array=lambdas))

#     # Add observed wavelengths (with redshift)
#     if z_in is not None:
#         wavelength_obs_array = [w * (1 + z_in) if w > 0 else np.nan for w in lambdas]
#         mapping_cols.append(fits.Column(name='WAVELENGTH_OBS', format='D', array=wavelength_obs_array))

#     mapping_cols.append(fits.Column(name='ACTION', format='A5', array=actions))
#     mapping_cols.append(fits.Column(name='KIND', format='A20', array=kinds))
#     mapping_cols.append(fits.Column(name='FIT_FLAG', format='A10', array=ffits))
#     mapping_cols.append(fits.Column(name='LINE_STATUS', format='A15', array=line_status))

#     # Add detailed status descriptions
#     status_descriptions = []
#     for action in actions:
#         if action == 'f':
#             status_descriptions.append('Fitted successfully')
#         elif action == 'm':
#             status_descriptions.append('Masked (sky/bad region)')
#         elif action == 'o':
#             status_descriptions.append('Outside wavelength range')
#         elif action == 'n':
#             status_descriptions.append('No wavelength defined')
#         else:
#             status_descriptions.append('Unknown status')

#     mapping_cols.append(fits.Column(name='STATUS_DESC', format='A30', array=status_descriptions))

#     # Add tie group information
#     tie_group_names = []
#     for i, kind in enumerate(kinds):
#         line_name = all_line_names[i]
#         if kind == 'balmer':
#             tie_group_names.append('BALMER_GROUP')
#         elif kind in ['high_ion_forbidden', 'low_ion_forbidden', 'sulfur_group',
#                       'hei_recombination', 'heii_recombination', 'uv_resonance',
#                       'uv_semiforbidden', 'iron_lines', 'miscellaneous']:
#             tie_group_names.append(kind.upper())
#         elif kind in ['oiii_group', 'nii_group']:
#             tie_group_names.append(kind.upper())
#         elif kind == 'forbidden':
#             tie_group_names.append('FORBIDDEN_LEGACY')
#         elif kind == 'independent':
#             tie_group_names.append('INDEPENDENT')
#         elif kind == 'all_tied':
#             tie_group_names.append('ALL_TIED')
#         else:
#             tie_group_names.append('UNKNOWN')

#     mapping_cols.append(fits.Column(name='TIE_GROUP', format='A20', array=tie_group_names))

#     # Add amplitude ratios
#     mapping_cols.append(fits.Column(name='AMPLITUDE_RATIO', format='D', array=aas))

#     # NEW: Add per-line kinematics statistics across all bins (flux-filtered)
#     mean_velocity_per_line = []
#     std_velocity_per_line = []
#     mean_sigma_per_line = []
#     std_sigma_per_line = []
#     n_valid_kinematics_per_line = []
#     n_valid_flux_per_line = []  # NEW: Track lines with valid flux

#     for line_idx in range(len(all_line_names)):
#         line_velocities = line_velocities_all[:, line_idx]
#         line_sigmas = line_sigmas_all[:, line_idx]
#         line_fluxes = emission_fluxes_all[:, line_idx]

#         valid_vel_mask = np.isfinite(line_velocities)
#         valid_sig_mask = np.isfinite(line_sigmas)
#         valid_flux_mask = np.isfinite(line_fluxes) & (line_fluxes > 0)

#         if np.any(valid_vel_mask):
#             mean_velocity_per_line.append(np.mean(line_velocities[valid_vel_mask]))
#             std_velocity_per_line.append(np.std(line_velocities[valid_vel_mask]))
#         else:
#             mean_velocity_per_line.append(np.nan)
#             std_velocity_per_line.append(np.nan)

#         if np.any(valid_sig_mask):
#             mean_sigma_per_line.append(np.mean(line_sigmas[valid_sig_mask]))
#             std_sigma_per_line.append(np.std(line_sigmas[valid_sig_mask]))
#         else:
#             mean_sigma_per_line.append(np.nan)
#             std_sigma_per_line.append(np.nan)

#         n_valid_kinematics_per_line.append(np.sum(valid_vel_mask & valid_sig_mask))
#         n_valid_flux_per_line.append(np.sum(valid_flux_mask))

#     # NEW: Add per-line kinematics statistics columns
#     mapping_cols.append(fits.Column(name='MEAN_VELOCITY', unit='km/s', format='D', array=mean_velocity_per_line))
#     mapping_cols.append(fits.Column(name='STD_VELOCITY', unit='km/s', format='D', array=std_velocity_per_line))
#     mapping_cols.append(fits.Column(name='MEAN_SIGMA', unit='km/s', format='D', array=mean_sigma_per_line))
#     mapping_cols.append(fits.Column(name='STD_SIGMA', unit='km/s', format='D', array=std_sigma_per_line))
#     mapping_cols.append(fits.Column(name='N_VALID_KINEMATICS', format='J', array=n_valid_kinematics_per_line))
#     mapping_cols.append(fits.Column(name='N_VALID_FLUX', format='J', array=n_valid_flux_per_line))

#     # Create the line mapping table
#     lineHDU = fits.BinTableHDU.from_columns(fits.ColDefs(mapping_cols))
#     lineHDU.name = 'LINE_MAPPING'

#     # Add comprehensive headers
#     lineHDU.header['N_LINES_ALL'] = (nlines_all, 'Total lines in emission config')
#     lineHDU.header['N_FITTED'] = (len(line_categories['fitted']), 'Lines fitted (action=f)')
#     lineHDU.header['N_MASKED'] = (len(line_categories['masked']), 'Lines masked (action=m)')
#     lineHDU.header['N_OUT_RANGE'] = (len(line_categories['out_of_range']), 'Lines outside range (action=o)')
#     lineHDU.header['N_NO_WAVE'] = (len(line_categories['missing_wavelength']), 'Lines no wavelength (action=n)')

#     # Add wavelength range and redshift info
#     lineHDU.header['LMIN_USED'] = (wavelength_range_config[0], 'Min wavelength used for fitting')
#     lineHDU.header['LMAX_USED'] = (wavelength_range_config[1], 'Max wavelength used for fitting')

#     if z_in is not None:
#         lineHDU.header['Z_USED'] = (z_in, 'Redshift used for line and mask shifting')

#     # Add tie settings summary
#     lineHDU.header['TIE_MODE'] = (tie_mode_used, 'Tie settings mode used')
#     lineHDU.header['TIE_ALL'] = (tie_settings.get('tie_all', False), 'All lines tied together')

#     # Add masking information
#     if emission_config and 'mask_regions' in emission_config:
#         n_mask_regions = len(emission_config['mask_regions'])
#         sky_masks = sum(1 for mr in emission_config['mask_regions'] if 'sky' in mr['name'].lower())
#         emission_masks = n_mask_regions - sky_masks
#         lineHDU.header['N_MASK_REG'] = (n_mask_regions, 'Number of mask regions applied')
#         lineHDU.header['N_SKY_MASK'] = (sky_masks, 'Sky line mask regions')
#         lineHDU.header['N_EMI_MASK'] = (emission_masks, 'Emission line mask regions')

#         # Add examples of mask regions
#         for i, mask_region in enumerate(emission_config['mask_regions'][:5]):  # First 5
#             name = mask_region['name']
#             wave = mask_region['wavelength']
#             width = mask_region.get('width', 3.0)
#             lineHDU.header[f'MASKEX_{i:02d}'] = (f'{name}', f'Mask example {i+1}')

#     # Add computational statistics
#     if 'estimated_components' in locals():
#         lineHDU.header['N_COMP_EST'] = (estimated_components, 'Estimated kinematic components')
#         lineHDU.header['COMP_REDUC'] = (computational_reduction, 'Computational reduction (%)')

#     # NEW: Add per-line kinematics summary statistics
#     valid_velocities = [v for v in mean_velocity_per_line if not np.isnan(v)]
#     valid_sigmas = [s for s in mean_sigma_per_line if not np.isnan(s)]

#     if len(valid_velocities) > 0:
#         lineHDU.header['V_MIN'] = (np.min(valid_velocities), 'Minimum velocity across lines (km/s)')
#         lineHDU.header['V_MAX'] = (np.max(valid_velocities), 'Maximum velocity across lines (km/s)')
#         lineHDU.header['V_RANGE'] = (np.max(valid_velocities) - np.min(valid_velocities), 'Velocity range across lines (km/s)')

#     if len(valid_sigmas) > 0:
#         lineHDU.header['S_MIN'] = (np.min(valid_sigmas), 'Minimum sigma across lines (km/s)')
#         lineHDU.header['S_MAX'] = (np.max(valid_sigmas), 'Maximum sigma across lines (km/s)')
#         lineHDU.header['S_RANGE'] = (np.max(valid_sigmas) - np.min(valid_sigmas), 'Sigma range across lines (km/s)')

#     lineHDU.header['N_LINES_KINEMATIC'] = (len(valid_velocities), 'Lines with valid kinematics')

#     # Add detailed comments about per-line kinematics and flux filtering
#     lineHDU.header['COMMENT'] = 'Complete mapping of all emission lines and their status'
#     lineHDU.header['COMMENT'] = 'ACTION codes: f=fitted, m=masked, o=out_of_range, n=no_wavelength'
#     lineHDU.header['COMMENT'] = 'Only lines with ACTION=f appear in EMIPPXF_TABLE'
#     lineHDU.header['COMMENT'] = 'MASKED lines were excluded due to sky contamination, bad pixels'
#     lineHDU.header['COMMENT'] = 'OUT_OF_RANGE lines were outside fitting wavelength limits'
#     lineHDU.header['COMMENT'] = 'TIE_GROUP shows which lines share kinematics'
#     lineHDU.header['COMMENT'] = 'WAVELENGTH_OBS includes redshift correction if applicable'
#     lineHDU.header['COMMENT'] = 'Per-line kinematics: Each fitted line has individual V and sigma'
#     lineHDU.header['COMMENT'] = 'Kinematics set to NaN for lines with flux less than 0 or NaN'
#     lineHDU.header['COMMENT'] = 'MEAN_VELOCITY/SIGMA: Average kinematics across all bins for each line'
#     lineHDU.header['COMMENT'] = 'STD_VELOCITY/SIGMA: Standard deviation of kinematics for each line'
#     lineHDU.header['COMMENT'] = 'N_VALID_KINEMATICS: Number of bins with valid kinematics per line'
#     lineHDU.header['COMMENT'] = 'N_VALID_FLUX: Number of bins with valid flux (>0 and finite) per line'
#     lineHDU.header['COMMENT'] = 'For single-component fits: all valid lines have identical kinematics'
#     lineHDU.header['COMMENT'] = 'For multi-component fits: lines tied by component have same kinematics'

#     # Create HDU list and write line mapping file
#     HDUList = fits.HDUList([priHDU, lineHDU])
#     HDUList.writeto(outfits_lines, overwrite=True)

#     ExGalutil.prettyOutput_Done("Writing: " + rootname + '_emippxf_lines_BIN.fits')
#     logging.info("Wrote: " + outfits_lines)

#     if debug:
#         print("EMIPPXF-compatible output with proper masking, per-line kinematics, and flux filtering created successfully!")
#         print(f"  Main results: {rootname}_emippxf_BIN.fits")
#         print(f"    - EMISSION_SETUP: All {nlines_all} lines with proper action codes")
#         print(f"    - EMIPPXF_TABLE: {len(line_names)} fitted lines with flux-filtered per-line kinematics")
#         print(f"  Spectra: {rootname}_emippxf_spec_BIN.fits")
#         print(f"    - Complete spectral models and emission-subtracted data")
#         print(f"  Line mapping: {rootname}_emippxf_lines_BIN.fits")
#         print(f"    - Detailed status and categorization of all lines with flux-filtered per-line kinematics")
#         print(f"  Line categorization:")
#         print(f"    - Fitted: {len(line_categories['fitted'])} (action=f)")
#         print(f"    - Masked: {len(line_categories['masked'])} (action=m)")
#         print(f"    - Out-of-range: {len(line_categories['out_of_range'])} (action=o)")
#         print(f"    - Missing wavelength: {len(line_categories['missing_wavelength'])} (action=n)")

#         # NEW: Add per-line kinematics and flux filtering statistics
#         if len(valid_velocities) > 0:
#             print(f"  Per-line kinematics statistics (flux-filtered):")
#             print(f"    - Velocity range: {np.min(valid_velocities):.1f} to {np.max(valid_velocities):.1f} km/s")
#             print(f"    - Sigma range: {np.min(valid_sigmas):.1f} to {np.max(valid_sigmas):.1f} km/s")
#             print(f"    - Lines with valid kinematics: {len(valid_velocities)}/{len(all_line_names)}")

#         # Print flux filtering statistics
#         total_valid_flux = sum(n_valid_flux_per_line)
#         total_measurements = len(ubins) * len([i for i in fitted_line_indices if i >= 0])
#         if total_measurements > 0:
#             print(f"    - Valid flux measurements: {total_valid_flux}/{total_measurements} ({total_valid_flux/total_measurements*100:.1f}%)")

#         # Add masking statistics
#         if emission_config and 'mask_regions' in emission_config:
#             n_mask_regions = len(emission_config['mask_regions'])
#             sky_masks = sum(1 for mr in emission_config['mask_regions'] if 'sky' in mr['name'].lower())
#             print(f"    - Applied {n_mask_regions} mask regions ({sky_masks} sky, {n_mask_regions-sky_masks} emission)")

#     return True


def save_emi_kinematics_emippxf_format(rootname, configs, outdir, results, mc_errors, formal_errors,
                                       lambdaR, bestfit, logLam, spectra, error, goodpixels,
                                       emi_bestfit, emission_lines, line_names, line_wavelengths,
                                       npix, ubins, tie_settings, emission_config, z_in=None, z_err=None, group_order=None, debug=False):
    """
    CORRECTED VERSION: Save emi kinematics results in EMIPPXF-compatible format.
    Creates columns for ALL lines in emission_config to ensure consistent data structure
    across different observations. Only populates data for actually fitted lines.

    CRITICAL: Maintains exact column order and structure regardless of which lines were fitted.
    """

    if debug:
        print("Creating EMIPPXF-compatible output with ALL lines for consistent structure...")

    # Determine wavelength range used for fitting
    if 'LMIN_EMI' in configs and 'LMAX_EMI' in configs:
        wavelength_range_config = (configs['LMIN_EMI'], configs['LMAX_EMI'])
    elif 'LMIN_PPXF' in configs and 'LMAX_PPXF' in configs:
        wavelength_range_config = (configs['LMIN_PPXF'], configs['LMAX_PPXF'])
    else:
        wavelength_range_config = (np.nan, np.nan)

    # Get wavelength range from the spectra itself
    lam = np.exp(logLam)
    wavelength_range = (lam.min(), lam.max())

    # Get ALL lines from emission config - this defines the column structure
    assert emission_config is not None and 'emission_lines' in emission_config, 'no config file found to build the all_line_names'
    all_line_names = list(emission_config['emission_lines'].keys())
    all_line_wavelengths = emission_config['emission_lines']

    # Categorize ALL lines
    line_categories = categorize_emission_lines(
        emission_config,
        line_names,
        wavelength_range,
        wavelength_range_config
    )

    if debug:
        print(f"Total lines in config: {len(all_line_names)}")
        print(f"Actually fitted lines: {len(line_names)}")
        print(f"Column structure will include ALL {len(all_line_names)} lines")

    # Create mapping from all lines to fitted lines
    # CRITICAL: This preserves the order from all_line_names
    fitted_line_mapping = {}
    for i, name in enumerate(all_line_names):
        if name in line_names:
            fitted_line_mapping[i] = line_names.index(name)
        else:
            fitted_line_mapping[i] = -1  # Not fitted

    # Track line status for each line in all_line_names order
    line_status = []
    for name in all_line_names:
        if name in line_categories['fitted']:
            line_status.append('FITTED')
        elif name in line_categories['masked']:
            line_status.append('MASKED')
        elif name in line_categories['out_of_range']:
            line_status.append('OUT_OF_RANGE')
        elif name in line_categories['missing_wavelength']:
            line_status.append('NO_WAVELENGTH')
        else:
            line_status.append('OTHER')

    # Set flux units
    if configs['sens_corr']:
        flux_unit_str = '%2e erg/(s cm**2 Angstrom)' % (configs['funits'])
        ivar_unit_str = '%2e cm**4 Angstrom**2 /(s**2 erg**2)' % (configs['funits']**-2)
    else:
        flux_unit_str = 'count'
        ivar_unit_str = '1/count**2'
    wave_unit_str = 'Angstrom'

    # Initialize arrays for ALL lines (not just fitted)
    n_all_lines = len(all_line_names)
    emission_amplitudes_all = np.full((len(ubins), n_all_lines), np.nan)
    emission_amplitude_errors_all = np.full((len(ubins), n_all_lines), np.nan)
    emission_aon_all = np.full((len(ubins), n_all_lines), np.nan)
    emission_fluxes_all = np.full((len(ubins), n_all_lines), np.nan)
    emission_flux_errors_all = np.full((len(ubins), n_all_lines), np.nan)
    # Equivalent widths -- see ExGalPrepare.compute_equivalent_width's own
    # docstring for the sign convention and error-propagation caveat.
    equivalent_widths_all = np.full((len(ubins), n_all_lines), np.nan)
    equivalent_width_errors_all = np.full((len(ubins), n_all_lines), np.nan)
    # bestfit here is the COMBINED stellar+emission model (see the two
    # call sites: `bestfit=combined_bestfit`); subtracting emi_bestfit
    # recovers the stellar-only continuum EW needs, on the shared `lam`
    # wavelength grid computed above -- mirrors MOSExGalEMIPPXF.py's own
    # `emissionSubtractedBestfit = bestfit - emi_bestfit` derivation.
    stellar_continuum_for_ew = (
        bestfit - emi_bestfit if bestfit is not None and emi_bestfit is not None else None)

    # Per-line kinematics for ALL lines
    line_velocities_all = np.full((len(ubins), n_all_lines), np.nan)
    line_sigmas_all = np.full((len(ubins), n_all_lines), np.nan)
    line_velocity_errors_all = np.full((len(ubins), n_all_lines), np.nan)
    line_sigma_errors_all = np.full((len(ubins), n_all_lines), np.nan)

    # Extract emi kinematics (for legacy compatibility)
    emi_velocities = np.zeros(len(ubins))
    emi_sigmas = np.zeros(len(ubins))
    emi_v_errors = np.zeros(len(ubins))
    emi_s_errors = np.zeros(len(ubins))

    # Populate data ONLY for fitted lines, maintaining all_line_names order
    for i, bin_id in enumerate(ubins):
        # Extract primary emi kinematics
        if isinstance(results[i]["emi"], np.ndarray) and len(results[i]["emi"]) >= 2:
            emi_velocities[i] = results[i]["emi"][0]
            emi_sigmas[i] = results[i]["emi"][1]
        elif isinstance(results[i]["emi"], list) and len(results[i]["emi"]) > 0:
            if isinstance(results[i]["emi"][0], np.ndarray) and len(results[i]["emi"][0]) >= 2:
                emi_velocities[i] = results[i]["emi"][0][0]
                emi_sigmas[i] = results[i]["emi"][0][1]

        # Extract primary emi errors
        if isinstance(mc_errors[i]["emi"], np.ndarray) and len(mc_errors[i]["emi"]) >= 2:
            emi_v_errors[i] = mc_errors[i]["emi"][0]
            emi_s_errors[i] = mc_errors[i]["emi"][1]
        elif isinstance(mc_errors[i]["emi"], list) and len(mc_errors[i]["emi"]) > 0:
            if isinstance(mc_errors[i]["emi"][0], np.ndarray) and len(mc_errors[i]["emi"][0]) >= 2:
                emi_v_errors[i] = mc_errors[i]["emi"][0][0]
                emi_s_errors[i] = mc_errors[i]["emi"][0][1]

        # Extract emission line properties - map from fitted to all lines
        if isinstance(mc_errors[i], dict):
            for all_idx, line_name in enumerate(all_line_names):
                fitted_idx = fitted_line_mapping[all_idx]

                if fitted_idx >= 0:  # This line was fitted
                    # Extract flux first to check validity
                    current_flux = np.nan
                    if isinstance(emission_lines[i], np.ndarray) and len(emission_lines[i]) > fitted_idx:
                        current_flux = emission_lines[i][fitted_idx]
                        emission_fluxes_all[i, all_idx] = current_flux

                    # Check if flux is valid for kinematics
                    flux_is_valid = (not np.isnan(current_flux) and current_flux is not None
                                    and current_flux > 1e-10 and line_name in line_categories['fitted'])

                    if not flux_is_valid:
                        emission_fluxes_all[i, all_idx] = np.nan

                    if flux_is_valid:
                        # Extract flux errors
                        if 'fluxes' in mc_errors[i] and isinstance(mc_errors[i]['fluxes'], np.ndarray):
                            if len(mc_errors[i]['fluxes']) > fitted_idx:
                                emission_flux_errors_all[i, all_idx] = mc_errors[i]['fluxes'][fitted_idx]

                        # Equivalent width -- FLUX / local stellar
                        # continuum at this line's observed wavelength
                        # (see ExGalPrepare.compute_equivalent_width's
                        # docstring). IFU EMIPPXF fits one systemic
                        # redshift per patch (z_in, scalar), not a
                        # per-bin redshift like MOS.
                        if stellar_continuum_for_ew is not None and z_in is not None:
                            rest_wave = all_line_wavelengths.get(line_name, np.nan)
                            ew, cont_at_line = ExGalPrepare.compute_equivalent_width(
                                current_flux, rest_wave, z_in, lam, stellar_continuum_for_ew[i])
                            equivalent_widths_all[i, all_idx] = ew
                            flux_err = emission_flux_errors_all[i, all_idx]
                            if np.isfinite(cont_at_line) and np.isfinite(flux_err):
                                equivalent_width_errors_all[i, all_idx] = flux_err / cont_at_line

                        # Extract amplitudes
                        if 'amplitudes' in mc_errors[i] and isinstance(mc_errors[i]['amplitudes'], np.ndarray):
                            if len(mc_errors[i]['amplitudes']) > fitted_idx:
                                emission_amplitudes_all[i, all_idx] = mc_errors[i]['amplitudes'][fitted_idx]

                        # Extract amplitude errors
                        if 'amplitude_errors' in mc_errors[i] and isinstance(mc_errors[i]['amplitude_errors'], np.ndarray):
                            if len(mc_errors[i]['amplitude_errors']) > fitted_idx:
                                emission_amplitude_errors_all[i, all_idx] = mc_errors[i]['amplitude_errors'][fitted_idx]

                        # Extract AON
                        if 'aon' in mc_errors[i] and isinstance(mc_errors[i]['aon'], np.ndarray):
                            if len(mc_errors[i]['aon']) > fitted_idx:
                                emission_aon_all[i, all_idx] = mc_errors[i]['aon'][fitted_idx]

                        # Extract per-line kinematics only if flux is valid
                        if 'line_velocities' in mc_errors[i] and isinstance(mc_errors[i]['line_velocities'], np.ndarray):
                            if len(mc_errors[i]['line_velocities']) > fitted_idx:
                                line_velocities_all[i, all_idx] = mc_errors[i]['line_velocities'][fitted_idx]

                        if 'line_sigmas' in mc_errors[i] and isinstance(mc_errors[i]['line_sigmas'], np.ndarray):
                            if len(mc_errors[i]['line_sigmas']) > fitted_idx:
                                line_sigmas_all[i, all_idx] = mc_errors[i]['line_sigmas'][fitted_idx]

                        if 'line_velocity_errors' in mc_errors[i] and isinstance(mc_errors[i]['line_velocity_errors'], np.ndarray):
                            if len(mc_errors[i]['line_velocity_errors']) > fitted_idx:
                                line_velocity_errors_all[i, all_idx] = mc_errors[i]['line_velocity_errors'][fitted_idx]

                        if 'line_sigma_errors' in mc_errors[i] and isinstance(mc_errors[i]['line_sigma_errors'], np.ndarray):
                            if len(mc_errors[i]['line_sigma_errors']) > fitted_idx:
                                line_sigma_errors_all[i, all_idx] = mc_errors[i]['line_sigma_errors'][fitted_idx]
                # else: line not fitted, remains NaN

    # Build emission setup arrays for ALL lines
    nlines_all = len(all_line_names)
    iis = list(range(nlines_all))
    names = all_line_names
    lambdas = [all_line_wavelengths.get(name, 0.0) for name in all_line_names]

    # Actions based on categorization
    actions = []
    for name in all_line_names:
        if name in line_categories['fitted']:
            actions.append('f')  # Fitted
        elif name in line_categories['masked']:
            actions.append('m')  # Masked
        elif name in line_categories['out_of_range']:
            actions.append('o')  # Out of range
        elif name in line_categories['missing_wavelength']:
            actions.append('n')  # No wavelength
        else:
            actions.append('u')  # Unknown

    # Kinds - determine from tie settings
    kinds = []
    if group_order is None:
        group_order = []

    for name in all_line_names:
        if tie_settings.get('tie_all', False):
            kinds.append('all_tied')
        elif any(name in tie_settings.get(group, []) for group in group_order):
            for group_name in group_order:
                if group_name in tie_settings and name in tie_settings.get(group_name, []):
                    kinds.append(group_name)
                    break
        elif name in tie_settings.get("[OIII]_5006.77", []) or name == "[OIII]_5006.77":
            kinds.append('[OIII]_group')
        elif name in tie_settings.get("[NII]_6583.34", []) or name == "[NII]_6583.34":
            kinds.append('[NII]_group')
        else:
            kinds.append('independent')

    # Amplitude ratios
    aas = []
    if emission_config and 'line_ratios' in emission_config:
        line_ratios = emission_config['line_ratios']
        for name in all_line_names:
            if name in line_ratios:
                aas.append(line_ratios[name]['ratio'])
            else:
                aas.append(1.0)
    else:
        aas = [1.0] * nlines_all

    # V and S values
    if emission_config and 'kinematic_settings' in emission_config:
        kin_settings = emission_config['kinematic_settings']
        default_v = kin_settings.get('initial_velocity', 0.0)
        default_s = kin_settings.get('initial_dispersion', 10.0)
    else:
        default_v = 0.0
        default_s = 10.0

    vs = [default_v] * nlines_all
    ss = [default_s] * nlines_all

    # Fit flags
    ffits = []
    for name in all_line_names:
        if name in line_categories['fitted']:
            ffits.append('True')
        else:
            ffits.append('False')

    # AoN values
    if emission_config and 'measurement_settings' in emission_config:
        default_aon = emission_config['measurement_settings'].get('aon_threshold', 4.0)
    else:
        default_aon = 4.0
    aons = [default_aon] * nlines_all

    # ========================
    # SAVE EMIPPXF-style results
    outfits = outdir + rootname + '_emippxf_BIN.fits'
    ExGalutil.prettyOutput_Running("Writing: " + rootname + '_emippxf_BIN.fits')

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Extension 1: Table HDU with emission_setup data
    cols = []
    cols.append(fits.Column(name='LINE_ID', format='D', array=np.arange(nlines_all)))
    cols.append(fits.Column(name='i', format='D', array=np.array(iis)))
    cols.append(fits.Column(name='name', format='15A', array=np.array(names)))
    cols.append(fits.Column(name='_lambda', format='D', array=np.array(lambdas)))
    cols.append(fits.Column(name='action', format='15A', array=np.array(actions)))
    cols.append(fits.Column(name='kind', format='15A', array=np.array(kinds)))
    cols.append(fits.Column(name='a', format='D', array=np.array(aas)))
    cols.append(fits.Column(name='v', format='D', array=np.array(vs)))
    cols.append(fits.Column(name='s', format='D', array=np.array(ss)))
    cols.append(fits.Column(name='fit', format='15A', array=np.array(ffits)))
    cols.append(fits.Column(name='aon', format='D', array=np.array(aons)))
    emission_setup_HDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    emission_setup_HDU.name = 'EMISSION_SETUP'

    # Add headers
    emission_setup_HDU.header['N_LINES_ALL'] = (nlines_all, 'Total lines in emission config')
    emission_setup_HDU.header['N_FITTED'] = (len(line_categories['fitted']), 'Lines successfully fitted')
    emission_setup_HDU.header['N_MASKED'] = (len(line_categories['masked']), 'Lines masked')
    emission_setup_HDU.header['N_OUT_RANGE'] = (len(line_categories['out_of_range']), 'Lines outside range')
    emission_setup_HDU.header['N_NO_WAVE'] = (len(line_categories['missing_wavelength']), 'Lines no wavelength')

    # Extension 2: Table HDU with EMIPPXF-style output data
    cols = []
    cols.append(fits.Column(name='BIN_ID', format='J', array=ubins))

    # Add redshift information
    if z_in is not None:
        z_array = np.full(len(ubins), z_in)
        cols.append(fits.Column(name='Z_INPUT', format='D', array=z_array))
        if z_err is not None:
            z_err_array = np.full(len(ubins), z_err)
            cols.append(fits.Column(name='ERR_Z_INPUT', format='D', array=z_err_array))

    # CRITICAL: Add columns for ALL lines in all_line_names order
    # This ensures consistent column structure across all observations
    for all_idx, line_name in enumerate(all_line_names):
        # EMIPPXF column naming: FLUX_<name>_<wavelength>
        wavelength = lambdas[all_idx]

        # Main measurements
        cols.append(fits.Column(name='FLUX_' + line_name,
                               format='D', array=emission_fluxes_all[:, all_idx]))
        cols.append(fits.Column(name='AMPL_' + line_name,
                               format='D', array=emission_amplitudes_all[:, all_idx]))
        cols.append(fits.Column(name='V_' + line_name,
                               format='D', array=line_velocities_all[:, all_idx]))
        cols.append(fits.Column(name='SIGMA_' + line_name,
                               format='D', array=line_sigmas_all[:, all_idx]))
        cols.append(fits.Column(name='AON_' + line_name,
                               format='D', array=emission_aon_all[:, all_idx]))

        # Error columns
        cols.append(fits.Column(name='ERR_FLUX_' + line_name,
                               format='D', array=emission_flux_errors_all[:, all_idx]))
        cols.append(fits.Column(name='ERR_AMPL_' + line_name,
                               format='D', array=emission_amplitude_errors_all[:, all_idx]))

        # Equivalent width -- positive for emission, see
        # ExGalPrepare.compute_equivalent_width's docstring for the sign
        # convention and error-propagation caveat.
        cols.append(fits.Column(name='EW_' + line_name, unit='Angstrom',
                               format='D', array=equivalent_widths_all[:, all_idx]))
        cols.append(fits.Column(name='ERR_EW_' + line_name, unit='Angstrom',
                               format='D', array=equivalent_width_errors_all[:, all_idx]))
        cols.append(fits.Column(name='ERR_V_' + line_name,
                               format='D', array=line_velocity_errors_all[:, all_idx]))
        cols.append(fits.Column(name='ERR_SIGMA_' + line_name,
                               format='D', array=line_sigma_errors_all[:, all_idx]))

    # EBMV placeholder columns
    len_EBMV = 1
    nan_array_EBMV = np.full((len_EBMV, len(ubins)), np.nan)
    cols.append(fits.Column(name='EBMV', format=str(len_EBMV) + 'D', array=nan_array_EBMV.T))
    cols.append(fits.Column(name='ERR_EBMV', format=str(len_EBMV) + 'D', array=nan_array_EBMV.T))

    # Create the main data table
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = 'EMIPPXF_TABLE'
    dataHDU.header['SAMPLING'] = (1, 'Sampling mode (0: linear, 1: logarithmic)')

    # Add headers
    dataHDU.header['CNF_EMI'] = (configs['CONFIG_FILE'], 'Configs. Filename used by emi analysis')
    dataHDU.header['EMI_LEV'] = ('BIN', 'emi analysis working level: BIN')
    dataHDU.header['EMI_SIG'] = (configs.get('SIGMA', 150.0), 'Initial guess for SIGMA')
    dataHDU.header['EMI_FILE'] = (configs.get('EMI_FILE', 'emission_lines.py'), 'Emission config')
    dataHDU.header['FOR_ERR'] = (configs.get('MC_EMI', 30) > 0, 'Errors derived')

    # Tie settings
    dataHDU.header['TIE_ALL'] = (tie_settings.get('tie_all', False), 'All lines tied')

    # Estimate computational reduction
    try:
        estimated_components, _ = estimate_component_count(line_names, tie_settings, group_order=group_order)
    except NameError:
        # Fallback calculation if function not available
        unique_groups = len(set(kinds))
        independent_lines = kinds.count('independent')
        estimated_components = unique_groups - 1 + independent_lines if 'independent' in kinds else unique_groups

    computational_reduction = (len(line_names) - estimated_components) / len(line_names) * 100 if len(line_names) > 0 else 0
    dataHDU.header['N_COMP_EST'] = (estimated_components, 'Estimated kinematic components')
    dataHDU.header['COMP_REDUC'] = (computational_reduction, 'Computational reduction (%)')

    # Wavelength range
    dataHDU.header['LW_EMI'] = (wavelength_range_config[0], 'Min wavelength used')
    dataHDU.header['HW_EMI'] = (wavelength_range_config[1], 'Max wavelength used')

    if z_in is not None:
        dataHDU.header['Z_EMI'] = (z_in, 'Redshift used')

    # Statistics
    dataHDU.header['N_LINES_ALL'] = (nlines_all, 'Total lines in config')
    dataHDU.header['N_FITTED'] = (len(line_categories['fitted']), 'Lines fitted')
    dataHDU.header['N_MASKED'] = (len(line_categories['masked']), 'Lines masked')
    dataHDU.header['N_OUT_RANGE'] = (len(line_categories['out_of_range']), 'Lines out of range')
    dataHDU.header['N_NO_WAVE'] = (len(line_categories['missing_wavelength']), 'Lines no wavelength')

    # Column structure info
    dataHDU.header['N_COLUMNS'] = (nlines_all * 9 + 6, 'Total data columns')
    dataHDU.header['COMMENT'] = f'Column structure includes ALL {nlines_all} lines from config'
    dataHDU.header['COMMENT'] = 'NaN values for non-fitted lines ensure consistent structure'
    dataHDU.header['COMMENT'] = 'This ensures identical data model across all observations'
    dataHDU.header['COMMENT'] = 'EW_<line> [Angstrom]: FLUX / local stellar continuum, POSITIVE for emission'
    dataHDU.header['COMMENT'] = '(opposite sign convention from the LS module Lick-style absorption indices)'
    dataHDU.header['COMMENT'] = 'ERR_EW_<line>: flux-error-only propagation (continuum treated as fixed)'
    ExGalPrepare.write_ebmv_header(dataHDU.header, configs)

    # Masking details
    if emission_config and 'mask_regions' in emission_config:
        n_mask_regions = len(emission_config['mask_regions'])
        sky_masks = sum(1 for mr in emission_config['mask_regions'] if 'sky' in mr['name'].lower())
        emission_masks = n_mask_regions - sky_masks
        dataHDU.header['N_MASK_REG'] = (n_mask_regions, 'Number of mask regions')
        dataHDU.header['N_SKY_MASK'] = (sky_masks, 'Sky line mask regions')
        dataHDU.header['N_EMI_MASK'] = (emission_masks, 'Emission line mask regions')

    dataHDU.header['TEMPL'] = (configs['SSP_LIB'], 'Spectral template library')
    dataHDU.header['PPXF_V'] = (PPXF_VERSION, 'PPXF version')
    dataHDU.header['APSVERS'] = (APSVERS, 'APS version')
    dataHDU.header['CSB_EMI'] = (configs['stitched'], 'Spectral Bands Status')

    # Input files
    for n_province, province in enumerate(configs['infiles']):
        dataHDU.header['APSREF_%d' % (n_province)] = (os.path.basename(province), 'L1 reference')

    # Line information in header (limited)
    dataHDU.header['NGLLINES'] = (nlines_all, 'Number of Lines in configuration')
    dataHDU.header['NFITLINES'] = (len(line_names), 'Number of Lines actually fitted')

    # Per-line kinematics info
    dataHDU.header['PER_LINE_KIN'] = (True, 'Per-line kinematics available')
    # dataHDU.header['FLUX_FILTER'] = (True, 'Kinematics NaN for flux ≤ 0')

    # Add line setup details
    for nii in range(nlines_all):
        status_code = actions[nii]
        dataHDU.header['GLID_%s' % (nii)] = '%30s' % (nii)
        dataHDU.header['GLNAM_%s' % (nii)] = '%30s' % (names[nii])
        dataHDU.header['GLLAM_%s' % (nii)] = '%30s' % (lambdas[nii])
        dataHDU.header['GLACT_%s' % (nii)] = '%30s' % (actions[nii])
        dataHDU.header['GLFIT_%s' % (nii)] = '%30s' % (ffits[nii])
        dataHDU.header['GLSTAT_%s' % (nii)] = '%30s' % (line_status[nii])

    # if nlines_all > 40:
    #     dataHDU.header['COMMENT'] = f'Line info truncated at 40/{nlines_all} lines'

    # Create HDU list and write
    HDUList = fits.HDUList([priHDU, emission_setup_HDU, dataHDU])
    HDUList.writeto(outfits, overwrite=True)

    ExGalutil.prettyOutput_Done("Writing: " + rootname + '_emippxf_BIN.fits')
    logging.info("Wrote: " + outfits)

    # ========================
    # SAVE EMIPPXF-style BESTFIT spectra
    outfits_spec = outdir + rootname + '_emippxf_spec_BIN.fits'
    ExGalutil.prettyOutput_Running("Writing: " + rootname + '_emippxf_spec_BIN.fits')

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Table HDU with spectra
    cols = []
    loglam_tile = np.tile(logLam, (len(ubins), 1))

    cols.append(fits.Column(name='BIN_ID', format='J', array=ubins))
    cols.append(fits.Column(name='LOGLAM_EMI', unit=wave_unit_str, format=str(npix) + 'D', array=loglam_tile))
    cols.append(fits.Column(name='FLUX_EMI', unit=flux_unit_str, format=str(npix) + 'D', array=np.transpose(spectra)))
    cols.append(fits.Column(name='ERROR_EMI', unit=flux_unit_str, format=str(npix) + 'D', array=np.transpose(error)))
    cols.append(fits.Column(name='MODEL_EMI', unit=flux_unit_str, format=str(npix) + 'D', array=bestfit))
    cols.append(fits.Column(name='EMISSION_EMI', unit=flux_unit_str, format=str(npix) + 'D', array=emi_bestfit))

    # Calculate emission-subtracted spectrum
    flux_clean = np.transpose(spectra) - emi_bestfit
    model_clean = bestfit - emi_bestfit
    cols.append(fits.Column(name='FLUX_CLEAN_EMI', unit=flux_unit_str, format=str(npix) + 'D', array=flux_clean))
    cols.append(fits.Column(name='MODEL_CLEAN_EMI', unit=flux_unit_str, format=str(npix) + 'D', array=model_clean))
    cols.append(fits.Column(name='GOODPIX_EMI', format=str(npix) + 'J', array=goodpixels))

    specHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    specHDU.name = 'EMIPPXF_SPEC'

    # Add headers
    specHDU.header['CNF_EMI'] = (configs['CONFIG_FILE'], 'Configs. Filename')
    specHDU.header['EMI_LEV'] = ('BIN', 'emi analysis working level')
    specHDU.header['SAMPLING'] = (1, 'Sampling mode (1: logarithmic)')
    specHDU.header['TEMPL'] = (configs['SSP_LIB'], 'Spectral template library')
    specHDU.header['PPXF_V'] = (PPXF_VERSION, 'PPXF version')
    specHDU.header['APSVERS'] = (APSVERS, 'APS version')
    specHDU.header['CSB_EMI'] = (configs['stitched'], 'Spectral Bands Status')

    # Statistics
    specHDU.header['N_LINES_ALL'] = (nlines_all, 'Total lines in config')
    specHDU.header['N_FITTED'] = (len(line_categories['fitted']), 'Lines fitted')
    specHDU.header['N_MASKED'] = (len(line_categories['masked']), 'Lines masked')

    # Wavelength info
    specHDU.header['LMIN_USED'] = (wavelength_range_config[0], 'Min wavelength')
    specHDU.header['LMAX_USED'] = (wavelength_range_config[1], 'Max wavelength')

    if z_in is not None:
        specHDU.header['Z_USED'] = (z_in, 'Redshift used')

    # Comments about structure
    specHDU.header['COMMENT'] = 'EMISSION_EMI contains fitted emission line model'
    specHDU.header['COMMENT'] = 'FLUX_CLEAN_EMI = FLUX_EMI - EMISSION_EMI'
    specHDU.header['COMMENT'] = f'Main table has columns for ALL {nlines_all} lines'

    for n_province, province in enumerate(configs['infiles']):
        specHDU.header['APSREF_%d' % (n_province)] = (os.path.basename(province), 'L1 reference')

    HDUList = fits.HDUList([priHDU, specHDU])
    HDUList.writeto(outfits_spec, overwrite=True)

    ExGalutil.prettyOutput_Done("Writing: " + rootname + '_emippxf_spec_BIN.fits')
    logging.info("Wrote: " + outfits_spec)

    # ========================
    # SAVE LINE MAPPING FILE
    outfits_lines = outdir + rootname + '_emippxf_lines_BIN.fits'
    ExGalutil.prettyOutput_Running("Writing: " + rootname + '_emippxf_lines_BIN.fits')

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Create line mapping table
    mapping_cols = []
    mapping_cols.append(fits.Column(name='LINE_ID', format='J', array=np.arange(nlines_all)))
    mapping_cols.append(fits.Column(name='LINE_NAME', format='A20', array=all_line_names))
    mapping_cols.append(fits.Column(name='WAVELENGTH', format='D', array=lambdas))

    if z_in is not None:
        wavelength_obs_array = [w * (1 + z_in) if w > 0 else np.nan for w in lambdas]
        mapping_cols.append(fits.Column(name='WAVELENGTH_OBS', format='D', array=wavelength_obs_array))

    mapping_cols.append(fits.Column(name='ACTION', format='A5', array=actions))
    mapping_cols.append(fits.Column(name='KIND', format='A20', array=kinds))
    mapping_cols.append(fits.Column(name='FIT_FLAG', format='A10', array=ffits))
    mapping_cols.append(fits.Column(name='LINE_STATUS', format='A15', array=line_status))

    # Status descriptions
    status_descriptions = []
    for action in actions:
        if action == 'f':
            status_descriptions.append('Fitted successfully')
        elif action == 'm':
            status_descriptions.append('Masked (sky/bad region)')
        elif action == 'o':
            status_descriptions.append('Outside wavelength range')
        elif action == 'n':
            status_descriptions.append('No wavelength defined')
        else:
            status_descriptions.append('Unknown status')

    mapping_cols.append(fits.Column(name='STATUS_DESC', format='A30', array=status_descriptions))
    mapping_cols.append(fits.Column(name='AMPLITUDE_RATIO', format='D', array=aas))

    # Per-line statistics
    mean_velocity_per_line = []
    std_velocity_per_line = []
    mean_sigma_per_line = []
    std_sigma_per_line = []
    n_valid_kinematics_per_line = []
    n_valid_flux_per_line = []

    for line_idx in range(nlines_all):
        line_velocities = line_velocities_all[:, line_idx]
        line_sigmas = line_sigmas_all[:, line_idx]
        line_fluxes = emission_fluxes_all[:, line_idx]

        valid_vel_mask = np.isfinite(line_velocities)
        valid_sig_mask = np.isfinite(line_sigmas)
        valid_flux_mask = np.isfinite(line_fluxes) & (line_fluxes > 0)

        if np.any(valid_vel_mask):
            mean_velocity_per_line.append(np.mean(line_velocities[valid_vel_mask]))
            std_velocity_per_line.append(np.std(line_velocities[valid_vel_mask]))
        else:
            mean_velocity_per_line.append(np.nan)
            std_velocity_per_line.append(np.nan)

        if np.any(valid_sig_mask):
            mean_sigma_per_line.append(np.mean(line_sigmas[valid_sig_mask]))
            std_sigma_per_line.append(np.std(line_sigmas[valid_sig_mask]))
        else:
            mean_sigma_per_line.append(np.nan)
            std_sigma_per_line.append(np.nan)

        n_valid_kinematics_per_line.append(np.sum(valid_vel_mask & valid_sig_mask))
        n_valid_flux_per_line.append(np.sum(valid_flux_mask))

    mapping_cols.append(fits.Column(name='MEAN_VELOCITY', unit='km/s', format='D', array=mean_velocity_per_line))
    mapping_cols.append(fits.Column(name='STD_VELOCITY', unit='km/s', format='D', array=std_velocity_per_line))
    mapping_cols.append(fits.Column(name='MEAN_SIGMA', unit='km/s', format='D', array=mean_sigma_per_line))
    mapping_cols.append(fits.Column(name='STD_SIGMA', unit='km/s', format='D', array=std_sigma_per_line))
    mapping_cols.append(fits.Column(name='N_VALID_KINEMATICS', format='J', array=n_valid_kinematics_per_line))
    mapping_cols.append(fits.Column(name='N_VALID_FLUX', format='J', array=n_valid_flux_per_line))

    # Create the line mapping table
    lineHDU = fits.BinTableHDU.from_columns(fits.ColDefs(mapping_cols))
    lineHDU.name = 'LINE_MAPPING'

    # Headers
    lineHDU.header['N_LINES_ALL'] = (nlines_all, 'Total lines in config')
    lineHDU.header['N_FITTED'] = (len(line_categories['fitted']), 'Lines fitted')
    lineHDU.header['N_MASKED'] = (len(line_categories['masked']), 'Lines masked')
    lineHDU.header['N_OUT_RANGE'] = (len(line_categories['out_of_range']), 'Lines out of range')
    lineHDU.header['LMIN_USED'] = (wavelength_range_config[0], 'Min wavelength')
    lineHDU.header['LMAX_USED'] = (wavelength_range_config[1], 'Max wavelength')

    if z_in is not None:
        lineHDU.header['Z_USED'] = (z_in, 'Redshift used')

    # Comments
    lineHDU.header['COMMENT'] = 'Complete mapping of ALL emission lines'
    lineHDU.header['COMMENT'] = 'Ensures consistent data model across observations'
    lineHDU.header['COMMENT'] = 'NaN values for non-fitted lines maintain structure'

    # Create HDU list and write
    HDUList = fits.HDUList([priHDU, lineHDU])
    HDUList.writeto(outfits_lines, overwrite=True)

    ExGalutil.prettyOutput_Done("Writing: " + rootname + '_emippxf_lines_BIN.fits')
    logging.info("Wrote: " + outfits_lines)

    if debug:
        print("CORRECTED IFU EMIPPXF output with consistent column structure:")
        print(f"  ALL {nlines_all} lines included in column structure")
        print(f"  Actually fitted: {len(line_names)} lines")
        print(f"  Column count: {nlines_all * 9 + 6} data columns")
        print(f"  Files created with identical structure to MOS mode")
        print(f"  NaN values for non-fitted lines ensure consistency")

    return True





def load_ppxf_emission_config(config_file_path,debug=False):
    """
    Load the pPXF emission line configuration file.

    Parameters:
    -----------
    config_file_path : str
        Path to the pPXF emission line configuration file

    Returns:
    --------
    config_dict : dict
        Dictionary containing all configuration variables
    """
    config_globals = {}

    try:
        with open(config_file_path, 'r') as f:
            exec(f.read(), config_globals)

        # Start with ALL variables from the config file (excluding built-ins)
        config_dict = {}
        for key, value in config_globals.items():
            # Skip built-in Python variables and imports
            if not key.startswith('__') and not callable(value):
                config_dict[key] = value

        # Ensure default keys exist (with empty defaults if not defined)
        default_keys = {
            'emission_lines': {},
            'line_groups': {},
            'line_ratios': {},
            'tie_settings': {},
            'kinematic_settings': {},
            'measurement_settings': {},
            'mask_regions': [],
            'line_limits': {},
            'emippxf_settings': {}
        }

        # Add defaults for any missing keys
        for key, default_value in default_keys.items():
            if key not in config_dict:
                config_dict[key] = default_value

        if debug:
            print(f"Loaded pPXF emission config with:")
            print(f"  {len(config_dict['emission_lines'])} emission lines")
            print(f"  {len(config_dict['mask_regions'])} mask regions")
            print(f"  {len(config_dict['line_groups'])} line groups")

            # Print ALL loaded keys for debugging
            print(f"  All loaded keys: {sorted(config_dict.keys())}")

        # Check for tie settings variants
        tie_variants = [k for k in config_dict.keys() if k.startswith('tie_settings')]
        if tie_variants:
            if debug:
                print(f"  Available tie settings: {tie_variants}")

        return config_dict

    except Exception as e:
        print(f"Error loading pPXF emission config: {e}")
        return None









def runModule_EMIPPXF(nthreads, configs, velscale, LSF_Data, LSF_Template, outdir, config_dir, template_dir, figdir, rootname, Z_IN, Z_IN_ERR=None, error_limit=1e18, debug=False, diag_plots=False, tie_mode='optimised', save_all_plots=False, bin_to_bucket=None, LSF_Data_by_bucket=None):
    """
    Starts the analysis of the emi kinematics. This function should be called after
    the stellar kinematics analysis has been completed.

    Parameters:
    -----------
    Z_IN : float
        Single redshift value for the entire field
    Z_IN_ERR : float, optional
        Error on the redshift (default: 0.01)
    bin_to_bucket, LSF_Data_by_bucket : optional
        Opt-in per-bin resolution (see `aps_ifu_spaxel_contrib.
        bucket_lsf_curves` and `IFUExGalPPXF.runModule_PPXF`'s own matching
        parameters). Both `None` (default): every bin uses the single
        `LSF_Data` for everything -- stellar template, emission template,
        the stellar-refit-for-emi-range step, and spectral masking -- byte-
        identical to this function's behaviour before these parameters
        existed. When given, each of those steps is done once per bucket
        (not once per patch) and the existing per-bin loops below (serial
        and parallel, structurally unchanged) pick the right bucket's own
        result per bin.
    """
    print("\033[0;37m" + " - - - - - Starting EMI KINEMATICS - - - - -" + "\033[0;39m")
    logging.info(" - - - Starting EMI KINEMATICS - - - ")




    # Read data from file
    hdu = fits.open(outdir + rootname + '_BINSpectra.fits')
    bin_data = np.array(hdu[1].data.SPEC.T)
    noise = np.array(hdu[1].data.ESPEC.T)

    if hdu[1].data.LOGLAM.ndim > 1:
        logLam = np.array(hdu[1].data.LOGLAM[0, :])
    else:
        logLam = np.array(hdu[1].data.LOGLAM)

    # Load stellar kinematics results to use as input
    stellar_available = True
    stellar_velocities = None
    stellar_sigmas = None
    stellar_v_errors = None

    try:
        hdu_ppxf = fits.open(outdir + rootname + '_ppxf.fits')
        ppxf_stellar = hdu_ppxf[1].data

        hdu_ppxf_spec = fits.open(outdir + rootname + '_ppxf_spec.fits')
        stellar_bestfit = np.array(hdu_ppxf_spec[1].data.MODEL_PPXF)

        # Extract stellar velocities and sigmas
        stellar_velocities = ppxf_stellar['V']
        stellar_sigmas = ppxf_stellar['SIGMA']
        stellar_v_errors = ppxf_stellar['ERR_V']

        ExGalutil.prettyOutput_Info(f"Using stellar kinematics as initial guess")
        ExGalutil.prettyOutput_Info(f"Stellar V range: {np.min(stellar_velocities):.1f} - {np.max(stellar_velocities):.1f} km/s")
        ExGalutil.prettyOutput_Info(f"Stellar sigma range: {np.min(stellar_sigmas):.1f} - {np.max(stellar_sigmas):.1f} km/s")

    except Exception as e:
        logging.error(f"Failed to load stellar kinematics results: {str(e)}")
        ExGalutil.prettyOutput_Error(f"Error loading stellar kinematics: {str(e)}")
        ExGalutil.prettyOutput_Warning("Proceeding without stellar kinematics input")
        stellar_bestfit = None
        ppxf_stellar = None
        stellar_available = False



    emission_config = None

    emission_config_file = os.path.join(config_dir, configs.get('EMI_FILE'))
    ExGalutil.prettyOutput_Info(f"Loading emission line configuration from: {emission_config_file}")
    emission_config = load_ppxf_emission_config(emission_config_file, debug=debug)

    # Use your emission lines instead of defaults
    line_wavelengths = emission_config['emission_lines']
    line_names = list(line_wavelengths.keys())
    line_groups = emission_config['line_groups']

    # SELECT TIE SETTINGS BASED ON MODE
    if tie_mode == 'optimised':
        tie_settings = emission_config['tie_settings_optimised']  # Default optimised settings
        ExGalutil.prettyOutput_Info(f"Using OPTIMISED tie settings: ~9 components, 90% computational reduction")

    elif tie_mode == 'aggressive':
        tie_settings = emission_config['tie_settings_aggressive']  # Aggressive settings
        ExGalutil.prettyOutput_Info(f"Using AGGRESSIVE tie settings: ~5 components, 94% computational reduction")

    elif tie_mode == 'legacy':
        # Legacy tie settings for backward compatibility
        tie_settings = emission_config['tie_settings_legacy']  # Aggressive settings
        ExGalutil.prettyOutput_Info(f"Using LEGACY tie settings: ~70 components, 30% computational reduction")

    elif tie_mode == 'default':
        # Legacy tie settings for backward compatibility
        tie_settings = {
            "balmer": [
                "H12_3749.93","H11_3770.93","H10_3797.92","H9_3835.91",
                "H5_3889.05","H8_3888.90","He_3970.07","Hd_4101.73",
                "Hg_4340.46","Hbeta_4861.32","Halpha_6562.80"
            ],
            "forbidden": [
                    "[OIII]_4363.15",
                    "[OIII]_4363.21",
                    "[OIII]_4931.23",
                    "[NII]_5754.40",
                    "[SII]_4071.15",
                    "[OII]_3726.03",
                    "[OII]_3728.73",
                    "[OII]_7319.46",
                    "[OII]_7329.98"
            ],
            '[OIII]_5006.77': [
                '[OIII]_4958.83',
                '[NI]_5197.90',
                '[NI]_5200.39'
            ],
            '[NII]_6583.34': [
                'HeI_5875.60',
                '[OI]_6300.20',
                '[OI]_6363.67',
                '[NII]_6547.96',
                '[SII]_6716.31',
                '[SII]_6730.68',
                '[ArIII]_7135.67'
            ],
            'tie_all': False
        }
        ExGalutil.prettyOutput_Info(f"Using DEFAULT tie settings for backward compatibility")

    elif tie_mode == 'custom':
        tie_settings = emission_config.get('tie_settings', emission_config['tie_settings'])
        ExGalutil.prettyOutput_Info(f"Using CUSTOM tie settings from emission config")

    else:
        raise ValueError(f"Unknown tie_mode: {tie_mode}. Use 'optimised', 'aggressive', 'legacy', or 'custom'")

    if debug:
        print(f"Selected tie mode '{tie_mode}':")

    group_order = extract_group_order_from_tie_settings(tie_settings)
    if debug:
        print(f" Using group order: {group_order}")

    # COLLECT EMI COMPONENTS FOR ALL BINS
    emi_component = build_emi_component_tying(line_names, tie_settings, group_order = group_order, debug=debug)
    # Later we will use this to create the emi_components for all bins
    # emi_components = [emi_component] * nbins


    # Emission line setup
    emission_setup = {
        'emi_moments': configs.get('EMI_MOM', 2),  # Default: only V and sigma for emi
        'use_stellar_continuum': configs.get('USE_STELLAR_CONTINUUM', True),  # Use stellar results
        'fix_stellar_continuum': configs.get('FIX_STELLAR_CONTINUUM', False)  # Fix stellar, fit emi only
    }


    # Define wavelength range for emi kinematics analysis FIRST
    if 'LMIN_EMI' in configs and 'LMAX_EMI' in configs:
        lmin_emi = configs['LMIN_EMI']
        lmax_emi = configs['LMAX_EMI']
    elif 'LMIN_PPXF' in configs and 'LMAX_PPXF' in configs:
        lmin_emi = configs['LMIN_PPXF']
        lmax_emi = configs['LMAX_PPXF']
    else:
        # Fallback to full data range
        lmin_emi = np.exp(logLam[0])
        lmax_emi = np.exp(logLam[-1])

    if debug:
        print(f"emi analysis wavelength range: {lmin_emi:.1f} - {lmax_emi:.1f} Å")

    # Slice data to the emi wavelength range
    idx_lam = np.where(np.logical_and(np.exp(logLam) >= lmin_emi, np.exp(logLam) <= lmax_emi))[0]




    if len(idx_lam) == 0:
        # ExGalutil.prettyOutput_Warning("Wavelength range for emi kinematics not valid.")
        # return

        # update 5 Nov 2024 for PPXF and 24 July 2025 for PPXF_EMI
        # in case overlap between the deredshifted spectrum and the LMIN/MAX_EMI is zero. we consider all to create a Null filled PPXF_EMI output
        idx_lam_nan_file  = np.where(  np.exp(logLam) > -1 )[0]

        bin_data = bin_data[idx_lam_nan_file,:]
        noise    = noise[idx_lam_nan_file,:]
        logLam   = logLam[idx_lam_nan_file]
        npix     = bin_data.shape[0]
        nbins    = bin_data.shape[1]
        ubins    = np.arange(0, nbins)


        # Create a emi_result structure with NaN values
        emi_result = [
            {
            'stellar': np.zeros(configs['MOM']),
            'emi': np.zeros(configs['EMI_MOM'])
            }
            for _ in range(nbins)
        ]

        # Create a mc_results structure with zero values
        mc_results = [
            {
                'stellar': np.zeros(configs['MOM']),
                'emi': np.zeros(configs['EMI_MOM']),
                'fluxes': np.zeros(len(line_names)),
                'amplitudes': np.zeros(len(line_names)),
                'amplitude_errors': np.zeros(len(line_names)),
                'aon': np.zeros(len(line_names))
            }
            for _ in range(nbins)
        ]

        # Create a formal_error structure with NaN values
        formal_error = [
            {
                'stellar': np.full(configs['MOM'], np.nan),
                'emi': np.full(configs['EMI_MOM'], np.nan)
            }
            for _ in range(nbins)
        ]

        emission_lines = [np.full(len(line_names), np.nan) for _ in range(nbins)]
        emi_bestfit = np.zeros((nbins, npix))
        goodpixels_emi = np.full((nbins, npix), -1)
        combined_bestfit = np.zeros((nbins, npix))


        # Save emi kinematics to file
        ExGalutil.prettyOutput_Running("Saving emi kinematics (all set to default/empty values) results to disk")
        save_emi_kinematics(
            rootname, configs, outdir, emi_result, mc_results, formal_error,
            None, emi_bestfit, logLam, bin_data, noise, goodpixels_emi,
            None, emission_lines, line_names, npix, ubins, tie_settings,
            emi_components=[emi_component] * nbins, z_in=Z_IN, z_err=Z_IN_ERR if Z_IN_ERR is not None else -1.0,
            emission_config=emission_config, line_wavelengths=line_wavelengths,
            group_order=group_order, debug=debug
        )


        # Save EMIPPXF-compatible output
        save_emi_kinematics_emippxf_format(
            rootname, configs, outdir, emi_result, mc_results, formal_error,
            None, combined_bestfit, logLam, bin_data, noise, goodpixels_emi,
            emi_bestfit, emission_lines, line_names, line_wavelengths,
            npix, ubins, tie_settings, emission_config, Z_IN, Z_IN_ERR if Z_IN_ERR is not None else -1.0,
            group_order=group_order, debug=debug
        )


        ExGalutil.prettyOutput_Warning("LMIN/LMAX_EMI [coming from config file] and Spectra wavelength are not consistent")
        logging.warning("LMIN/LMAX_EMI [coming from config file] and Spectra wavelength are not consistent")
        ExGalutil.prettyOutput_Warning("A null PPXF_EMI output just generated. Skipping PPXF_EMI")
        logging.warning("A null PPXF_EMI output just generated. Skipping PPXF_EMI")

        return


    # If we have valid indices, proceed with slicing
    # Slice data
    bin_data = bin_data[idx_lam, :]
    noise = noise[idx_lam, :]
    logLam = logLam[idx_lam]

    # Get dimensions
    npix = bin_data.shape[0]
    nbins = bin_data.shape[1]
    ubins = np.arange(0, nbins)

    ExGalutil.prettyOutput_Info(f"Preparing template for wavelength range: {lmin_emi:.1f} - {lmax_emi:.1f} Å")
    if debug:
        print(f"Data has {npix} pixels")


    # Raw SSP library loaded once regardless of bucket count -- see
    # IFUExGalPPXF.runModule_PPXF's own identical comment for the full
    # rationale (load_raw_template_library's docstring has the details).
    raw_templates_emi = IFUExGalPrepare.load_raw_template_library(
        "PPXF", template_dir, configs, lmin_emi, lmax_emi)
    velscale_ratio, template_velscale, data_velscale = IFUExGalPrepare.calculate_velscale_ratio(
        configs, logLam, velscale, LSF_Data, LSF_Template, template_dir, debug=debug, preloaded=raw_templates_emi)

    if LSF_Data_by_bucket:
        logging.info(f"Using {len(LSF_Data_by_bucket)} resolution bucket(s) for EMIPPXF template preparation")
        template_by_bucket = {}
        emi_raw_by_bucket = {}
        emi_unbroad_raw_by_bucket = {}
        non_zero_mask = None  # union across buckets -- see below

        for bucket, lsf_func in LSF_Data_by_bucket.items():
            tpl, lamRange_spmod, logLam_template, ntemplate = IFUExGalPrepare.prepareSpectralTemplateLibrary(
                "PPXF", template_dir, configs, lmin_emi, lmax_emi, velscale, velscale_ratio,
                lsf_func, LSF_Template, preloaded=raw_templates_emi)[:4]
            template_by_bucket[bucket] = tpl.reshape((tpl.shape[0], ntemplate))

            # note: as we are using the logLam_template from the stellar template, no need to apply any extra offset as the offset is already applied in the stellar template preparation
            emi_tpl, emi_tpl_unbroad, logLam_emi_template = prepare_emission_template(
                logLam_template, line_names, line_wavelengths,
                LSF_Template, lsf_func,
                wl_offset=0.0, line_ties=configs.get('line_ties', None), debug=debug)
            emi_raw_by_bucket[bucket] = emi_tpl
            emi_unbroad_raw_by_bucket[bucket] = emi_tpl_unbroad

            # Union, not per-bucket: a line judged "non-zero" in even one
            # bucket must stay a real output column for every bucket --
            # save_emi_kinematics writes one shared line_names/column set
            # for the whole bin table, so every bucket's emi_template must
            # keep the exact same column count/order.
            mask = np.max(np.abs(emi_tpl), axis=0) > 1e-10
            non_zero_mask = mask if non_zero_mask is None else (non_zero_mask | mask)

        if debug:
            print(f"non_zero_mask (union across {len(LSF_Data_by_bucket)} buckets) sum: {np.sum(non_zero_mask)}")

        line_names = [line_names[i] for i in range(len(line_names)) if non_zero_mask[i]]
        emi_template_by_bucket = {b: t[:, non_zero_mask] for b, t in emi_raw_by_bucket.items()}
        emi_template_unbroadened_by_bucket = {b: t[:, non_zero_mask] for b, t in emi_unbroad_raw_by_bucket.items()}

        # Reference copies (shape/consistency checks below, and the
        # single-value fallback anywhere a helper hasn't been threaded
        # through yet) -- any bucket's own arrays are equally valid here.
        _default_bucket = next(iter(template_by_bucket))
        template = template_by_bucket[_default_bucket]
        emi_template = emi_template_by_bucket[_default_bucket]
        emi_template_unbroadened = emi_template_unbroadened_by_bucket[_default_bucket]

    else:
        template_by_bucket = None
        emi_template_by_bucket = None
        emi_template_unbroadened_by_bucket = None

        template, lamRange_spmod, logLam_template, ntemplate = IFUExGalPrepare.prepareSpectralTemplateLibrary(
            "PPXF", template_dir, configs, lmin_emi, lmax_emi, velscale, velscale_ratio,
            LSF_Data, LSF_Template, preloaded=raw_templates_emi)[:4]
        template = template.reshape((template.shape[0], ntemplate))

        # note: as we are using the logLam_template from the stellar template, no need to apply any extra offset as the offset is already applied in the stellar template preparation
        emi_template, emi_template_unbroadened,  logLam_emi_template = prepare_emission_template(
            logLam_template, line_names, line_wavelengths,
            LSF_Template, LSF_Data,  # ADD LSF_Data here
            wl_offset=0.0,  # ADD wl_offset matching stellar template
            line_ties=configs.get('line_ties', None), debug=debug)

        # Find which template are actually non-zero
        non_zero_mask = np.max(np.abs(emi_template), axis=0) > 1e-10
        if debug:
            print(f"non_zero_mask shape: {non_zero_mask.shape}")
            print(f"non_zero_mask sum: {np.sum(non_zero_mask)}")

        # The filtering operation
        emi_template_filtered = emi_template[:, non_zero_mask]  # Use new variable name
        emi_template_unbroadened_filtered = emi_template_unbroadened[:, non_zero_mask]
        line_names_filtered = [line_names[i] for i in range(len(line_names)) if non_zero_mask[i]]

        if debug:
            # Check if original array changed
            print(f"Original emi_template max after filtering: {np.max(np.abs(emi_template)):.6e}")

        # Assign the filtered versions
        emi_template = emi_template_filtered
        emi_template_unbroadened = emi_template_unbroadened_filtered
        line_names = line_names_filtered

    if debug:
        print(f"Template shape: {template.shape}")
        print(f"Template wavelength range: {lamRange_spmod}")

    def _template_for_bin(i):
        if template_by_bucket is not None:
            return template_by_bucket[bin_to_bucket[i]]
        return template

    def _emi_template_for_bin(i):
        if emi_template_by_bucket is not None:
            return emi_template_by_bucket[bin_to_bucket[i]]
        return emi_template

    def _emi_template_unbroadened_for_bin(i):
        if emi_template_unbroadened_by_bucket is not None:
            return emi_template_unbroadened_by_bucket[bin_to_bucket[i]]
        return emi_template_unbroadened

    def _lsf_for_bin(i):
        if LSF_Data_by_bucket is not None:
            return LSF_Data_by_bucket[bin_to_bucket[i]]
        return LSF_Data

    if debug:
        print("==============================")

        # CRITICAL: Template size consistency for PPXF (all arrays must be npix length)
        print("=== TEMPLATE SIZE CONSISTENCY CHECK ===")
        print(f"Data pixels (npix): {npix}")
        print(f"Stellar template: {template.shape}")
        print(f"emi template: {emi_template.shape}")
        print(f"logLam_template length: {len(logLam_template)}")
        print(f"logLam (data) length: {len(logLam)}")



        # Final validation
        print("=== FINAL SIZE VALIDATION ===")
        print(f"Data: {npix}")
        print(f"Stellar template: {template.shape}")
        print(f"emi template: {emi_template.shape}")


    # Check if emi wavelength range differs from stellar
    emi_range_differs = False
    if ('LMIN_EMI' in configs and 'LMAX_EMI' in configs and
        'LMIN_PPXF' in configs and 'LMAX_PPXF' in configs):
        if (configs['LMIN_EMI'] != configs['LMIN_PPXF'] or
            configs['LMAX_EMI'] != configs['LMAX_PPXF']):
            emi_range_differs = True
            ExGalutil.prettyOutput_Info(f"emi wavelength range ({configs['LMIN_EMI']}-{configs['LMAX_EMI']} Å) "
                  f"differs from stellar range ({configs['LMIN_PPXF']}-{configs['LMAX_PPXF']} Å)")

    # Prepare stellar fit for emi range if needed
    stellar_fit_for_emi = None

    # Calculate offset after template consistency is ensured
    # No offset needed anymore as we provide logLam and logLam_template directly
    # offset = (logLam_template[0] - logLam[0]) * C

    if emi_range_differs and stellar_available:
        ExGalutil.prettyOutput_Info("Preparing fixed stellar fit for emi wavelength range...")

        ExGalutil.prettyOutput_Running(f"stellar RE-fit for emi range")

        # Prepare stellar fit for emi wavelength range
        # This will use the same template and wavelength range as emi analysis
        stellar_fit_for_emi, stellar_kinematics_emi_range = prepare_stellar_fit_for_emi_range_parallel(
            outdir, rootname, logLam, logLam_emi_template, bin_data, noise, template, velscale, velscale_ratio,
            configs, nbins, error_limit, stellar_kinematics=None,
            nthreads=nthreads,  # Same threads as main emi analysis
            debug = debug,
            bin_to_bucket=bin_to_bucket, template_by_bucket=template_by_bucket,
        )
        ExGalutil.prettyOutput_Done(f"stellar RE-fit for emi range")


    elif not emi_range_differs and stellar_available:
        ExGalutil.prettyOutput_Warning("emi and stellar wavelength ranges are the same - using existing stellar results")
        # Can use stellar_bestfit directly from the original stellar analysis
        stellar_fit_for_emi = stellar_bestfit

    else:
        ExGalutil.prettyOutput_Warning("No stellar fit will be used for emi analysis")

    # Use single redshift for all bins
    z_field = Z_IN
    z_error_field = Z_IN_ERR if Z_IN_ERR is not None else 0.01

    ExGalutil.prettyOutput_Info(f"Using field redshift z = {z_field:.4f} ± {z_error_field:.4f}")

    # Last preparatory steps
    nsims = configs.get('MC_EMI', configs.get('MC_PPXF', 30))  # Use emi MC sims or stellar default
    nmoments = configs.get('MOM', 4)  # Get actual number of moments
    start = np.zeros((nbins, max(nmoments, 2)))  # Ensure enough space for both stellar and emi

    if stellar_available and stellar_velocities is not None and stellar_sigmas is not None:
        start[:, 0] = stellar_velocities[:nbins] if len(stellar_velocities) >= nbins else stellar_velocities[0]
        start[:, 1] = stellar_sigmas[:nbins] if len(stellar_sigmas) >= nbins else stellar_sigmas[0]

        if nmoments > 2 and 'H3' in ppxf_stellar.names:
            start[:, 2] = ppxf_stellar['H3'][:nbins]
        if nmoments > 3 and 'H4' in ppxf_stellar.names:
            start[:, 3] = ppxf_stellar['H4'][:nbins]
    else:
        start[:, 0] = 0.0
        start[:, 1] = configs.get('SIGMA', 150.0)
        # Higher moments default to 0
        if nmoments > 2:
            start[:, 2] = 0.0
        if nmoments > 3:
            start[:, 3] = 0.0
        print(f"Using default initial guess: V=0 km/s, Sigma={configs.get('SIGMA', 150.0)} km/s")

    # Array to store results of emi kinematics fitting
    emi_metalist = []
    emi_result = []
    emi_bestfit = np.zeros((nbins, npix))
    emission_lines = []
    mc_results = []
    formal_error = []
    goodpixels_emi = np.empty((nbins, npix))
    goodpixels_emi.fill(-1)



    if debug:
        print("=== SIMPLIFIED EMI TEMPLATE SCALING ===")
        print(f"Before scaling:")
        print(f"  Data scale: {np.max(np.abs(bin_data)):.3e}")
        print(f"  Stellar template scale: {np.max(np.abs(template)):.3e}")
        print(f"  emi template scale: {np.max(np.abs(emi_template)):.3e}")
        print(f"  emi template shape: {emi_template.shape}")

    stellar_scale = np.max(np.abs(template))
    emi_scale = np.max(np.abs(emi_template))  # Simple - no filtering needed!

    # Calculate scaling
    target_scale = stellar_scale * 0.1  # 10% of stellar
    scaling_factor = target_scale / emi_scale

    if debug:
        print(f"Scaling emi template by factor: {scaling_factor:.3e}")

    # Apply scaling
    emi_template *= scaling_factor
    emi_template_unbroadened *= scaling_factor

    # STORE the scaling factor for flux correction
    configs['EMI_TEMPLATE_SCALING'] = scaling_factor  # Add this line

    if debug:
        print(f"After scaling:")
        print(f"  emi template scale: {np.max(np.abs(emi_template)):.3e}")
        print(f"  Ratio to stellar: {np.max(np.abs(emi_template)) / stellar_scale:.3f}")
        print("============================================")

    # define the linear wavelength for emi and emi template to be directly used in pPXF
    lam_emi = np.exp(logLam)  # Convert logLam to linear wavelength for emi template
    lam_emi_template = np.exp(logLam_emi_template)  # Convert logLam_emi_template to linear wavelength

    # Start running emi kinematics in serial or parallel mode
    if nthreads > 1:
            logging.info("Running emi kinematics in parallel mode")
            ExGalutil.prettyOutput_Running("Running emi kinematics in parallel mode")

            # Create Queues
            inQueue = Queue()
            outQueue = Queue()

            # Create worker processes
            ps = [Process(target=workerPPXF_emi, args=(inQueue, outQueue))
                for _ in range(nthreads)]

            # Start worker processes
            for p in ps:
                p.start()

            # Track which bins we actually submit to the queue
            submitted_bins = []
            skipped_bins = []

            # Fill the queue and track submitted bins
            for i in range(nbins):
                # Use field redshift for all bins
                z_bin = z_field

                # Notes:
                # logLam is in the rest frame (spectrum has been de-redshifted).
                # Astrophysical emission line masks are already in rest frame — no correction needed.
                # Sky/telluric masks (name starts with 'sky_') are in OBSERVED frame and must be
                # divided by (1+z) to place them correctly in the rest-frame spectrum.

                if emission_config is not None:
                    goodpixels_ppxf = apply_ppxf_emission_masking(
                        logLam=logLam,
                        velscale=velscale,
                        emission_config=emission_config,
                        error_spectrum=noise[:, i],
                        error_limit=error_limit,
                        configs=configs,
                        verbose=debug,  # Reduce verbosity in parallel mode
                        LSF_Data=_lsf_for_bin(i),
                        redshift=z_bin
                    )
                else:
                    # Fallback to original masking if emission config not available
                    # note : as the spectrum is already deredshifted, we do not apply any redshift offset
                    goodpixels_ppxf = IFUExGalPrepare.spectralMasking(
                        config_dir, configs, logLam, 'EMI', z_bin, LSF_Data=_lsf_for_bin(i), verbose=debug
                    )
                    goodpixels_ppxf = IFUExGalPrepare.MaskGaps(noise[:, i], goodpixels_ppxf, error_limit)

                if len(goodpixels_ppxf) == 0:
                    print(f"Warning: No valid pixels for bin {i}, skipping")
                    skipped_bins.append(i)
                    continue

                # Get stellar continuum for this bin
                stellar_bestfit_bin = (stellar_fit_for_emi[i, :]
                                    if stellar_fit_for_emi is not None
                                    else None)

                # Submit job to queue
                inQueue.put((
                    rootname, _template_for_bin(i), _emi_template_for_bin(i), _emi_template_unbroadened_for_bin(i), bin_data[:, i], noise[:, i], velscale, start[i, :],
                    goodpixels_ppxf, configs['MOM'], configs['ADEG'], configs['MDEG'],
                    velscale_ratio, error_limit, nsims, nbins, i, emission_setup,
                    stellar_bestfit_bin, line_names, tie_settings, line_wavelengths,
                    lam_emi, lam_emi_template, figdir, diag_plots, group_order, scaling_factor
                ))
                submitted_bins.append(i)

            jobs_submitted = len(submitted_bins)
            if debug:
                print(f"Submitted {jobs_submitted} jobs to queue (skipped {len(skipped_bins)} bins: {skipped_bins})")

            # Initialize all result arrays with proper size (for ALL bins)
            emi_result = [{"stellar": np.full(configs['MOM'], np.nan),
                        "emi": np.full(emission_setup['emi_moments'], np.nan)} for _ in range(nbins)]
            emission_lines = [np.full(len(line_names), np.nan) for _ in range(nbins)]
            mc_results = [{"stellar": np.full(configs['MOM'], np.nan),
                        "emi": np.full(emission_setup['emi_moments'], np.nan),
                        "fluxes": np.full(len(line_names), np.nan),
                        "amplitudes": np.full(len(line_names), np.nan),
                        "amplitude_errors": np.full(len(line_names), np.nan),
                        "aon": np.full(len(line_names), np.nan)} for _ in range(nbins)]
            formal_error = [{"stellar": np.full(configs['MOM'], np.nan),
                            "emi": np.full(emission_setup['emi_moments'], np.nan)} for _ in range(nbins)]
            emi_bestfit = np.full((nbins, npix), np.nan)

            # Track which bins we've received results for
            received_bins = set()
            timeout_seconds = 600  # 10 minutes per bin

            # Collect results
            for job_idx in range(jobs_submitted):
                try:
                    result = outQueue.get(timeout=timeout_seconds)
                    bin_id, sol, bestfit, emi_bf, emission_flux, mc_err, formal_err, goodpix = result

                    # Validate bin_id and store result
                    if 0 <= bin_id < nbins:
                        received_bins.add(bin_id)
                        emi_result[bin_id] = sol
                        emi_bestfit[bin_id, :] = emi_bf
                        emission_lines[bin_id] = emission_flux
                        mc_results[bin_id] = mc_err
                        formal_error[bin_id] = formal_err

                        # Handle goodpixels
                        goodpix_len = len(goodpix)
                        if goodpix_len > 0:
                            goodpixels_emi[bin_id, 0:goodpix_len] = goodpix

                        if len(received_bins) % 10 == 0 or len(received_bins) <= 5:
                            if debug:
                                print(f"Completed {len(received_bins)}/{jobs_submitted} jobs")
                    else:
                        print(f"Warning: Received invalid bin_id {bin_id}, ignoring result")

                except Exception as e:
                    print(f"Warning: Failed to get result for job {job_idx + 1}/{jobs_submitted}: {e}")
                    # We can't create a meaningful dummy result here because we don't know
                    # which specific bin failed. The missing bins will remain as NaN.

            # Send stop signal to stop iteration
            for _ in range(nthreads):
                inQueue.put('STOP')

            # Stop processes with timeout
            for p in ps:
                p.join(timeout=30)  # 30 second timeout
                if p.is_alive():
                    ExGalutil.prettyOutput_Warning(f"Warning: Process {p.pid} did not terminate, killing it")
                    p.terminate()
                    p.join()

            # Check which submitted bins we didn't receive results for
            submitted_bins_set = set(submitted_bins)
            missing_bins = submitted_bins_set - received_bins

            if missing_bins:
                ExGalutil.prettyOutput_Warning(f"Warning: No results received for {len(missing_bins)} submitted bins: {sorted(missing_bins)}")
                ExGalutil.prettyOutput_Warning(f"These bins will have NaN results")

            if debug:
                print(f"Parallel processing completed:")
                print(f"  - Total bins: {nbins}")
                print(f"  - Skipped bins (no valid pixels): {len(skipped_bins)}")
                print(f"  - Submitted bins: {jobs_submitted}")
                print(f"  - Successful bins: {len(received_bins)}")
                print(f"  - Failed/missing bins: {len(missing_bins)}")

            ExGalutil.prettyOutput_Done("Running emi kinematics in parallel mode", progressbar=True)

    else:
        ExGalutil.prettyOutput_Running("Running emi kinematics in serial mode")
        logging.info("Running emi kinematics in serial mode")

        for i in range(nbins):
            # Use field redshift for this bin
            z_bin = z_field

            if debug:
                print(f"Processing bin {i}/{nbins}: z = {z_bin:.4f}")

            if emission_config is not None:
                # Apply comprehensive masking using field redshift
                goodpixels_ppxf = apply_ppxf_emission_masking(
                    logLam=logLam,
                    velscale=velscale,
                    emission_config=emission_config,
                    error_spectrum=noise[:, i],
                    error_limit=error_limit,
                    configs=configs,
                    verbose=(i < 3),  # Print details for first 3 bins only
                    LSF_Data=_lsf_for_bin(i),
                    redshift=z_bin
                )
            else:
                # Fallback to original masking if emission config not available
                # note : as the spectrum is already deredshifted, we do not apply any redshift offset
                goodpixels_ppxf = IFUExGalPrepare.spectralMasking(
                    config_dir, configs, logLam, 'EMI', z_bin, LSF_Data=_lsf_for_bin(i), verbose=(i < 3)
                )
                goodpixels_ppxf = IFUExGalPrepare.MaskGaps(noise[:, i], goodpixels_ppxf, error_limit)


            if len(goodpixels_ppxf) == 0:
                ExGalutil.prettyOutput_Warning(f"No valid pixels for bin {i}, skipping")
                # Add placeholder results
                emi_result.append({"stellar": np.full(configs['MOM'], np.nan),
                                "emi": np.full(emission_setup['emi_moments'], np.nan)})
                emission_lines.append(np.full(len(line_names), np.nan))
                mc_results.append({"stellar": np.full(configs['MOM'], np.nan),
                                "emi": np.full(emission_setup['emi_moments'], np.nan),
                                "fluxes": np.full(len(line_names), np.nan)})
                formal_error.append({"stellar": np.full(configs['MOM'], np.nan),
                                "emi": np.full(emission_setup['emi_moments'], np.nan)})
                continue

            # Get stellar bestfit for this bin
            stellar_bestfit_bin = (stellar_fit_for_emi[i, :]
                                   if stellar_fit_for_emi is not None
                                   else None)


            # Run emi kinematics for this bin - FIXED: Added logLam parameter
            sol, bestfit, emi_bf, emission_flux, mc_err, formal_err, goodpix = run_ppxf_emi(
                rootname, _template_for_bin(i), _emi_template_for_bin(i), _emi_template_unbroadened_for_bin(i), bin_data[:, i], noise[:, i], velscale, start[i, :],
                goodpixels_ppxf, error_limit, configs['MOM'], configs['ADEG'], configs['MDEG'],
                velscale_ratio, nsims, nbins, i, emission_setup, stellar_bestfit_bin,
                line_names, tie_settings, line_wavelengths, lam_emi, lam_emi_template, figdir, diag_plots, group_order,scaling_factor
            )






            # Store results
            emi_result.append(sol)
            emi_bestfit[i, :] = emi_bf
            emission_lines.append(emission_flux)
            mc_results.append(mc_err)
            formal_error.append(formal_err)
            goodpix_len = len(goodpix)
            if goodpix_len > 0:
                goodpixels_emi[i, 0:goodpix_len] = goodpix
        ExGalutil.prettyOutput_Done("Running emi kinematics in serial mode", progressbar=True)

    # Check for exceptions which occurred during the analysis
    n_failed = 0
    for i in range(nbins):
        if isinstance(emi_result[i]["emi"], np.ndarray):
            if np.isnan(emi_result[i]["emi"]).any():
                n_failed += 1
        else:
            n_failed += 1

    if n_failed > 0:
        ExGalutil.prettyOutput_Warning(f"emi kinematics analysis failed for {n_failed} bins")
        logging.warning(f"emi kinematics analysis failed for {n_failed} bins")
    else:
        ExGalutil.prettyOutput_Info("             No problems in the emi kinematics analysis.")
        logging.info("No problems in the emi kinematics analysis.")
    print("")



    # CREATE COMBINED BESTFIT HERE - AFTER EMI FITTING IS COMPLETE
    # Create combined bestfit (stellar + emi) for EMIPPXF output
    combined_bestfit = np.zeros_like(emi_bestfit)
    for i in range(nbins):
        if stellar_fit_for_emi is not None:
            combined_bestfit[i, :] = stellar_fit_for_emi[i, :] + emi_bestfit[i, :]
        else:
            combined_bestfit[i, :] = emi_bestfit[i, :]  # emi-only if no stellar


    # Save emi kinematics to file
    ExGalutil.prettyOutput_Running("Saving emi kinematics results to disk")
    save_emi_kinematics(
        rootname, configs, outdir, emi_result, mc_results, formal_error,
        None, emi_bestfit, logLam, bin_data, noise, goodpixels_emi,
        None, emission_lines, line_names, npix, ubins, tie_settings,
        emi_components=[emi_component] * nbins, z_in=z_field, z_err=z_error_field,
        emission_config=emission_config, line_wavelengths=line_wavelengths,
        group_order=group_order, debug=debug
    )
    ExGalutil.prettyOutput_Done("Saving emi kinematics results to disk")



    # Save EMIPPXF-compatible output
    try:
        ExGalutil.prettyOutput_Running("Saving emi kinematics in EMIPPXF format to disk")
        save_emi_kinematics_emippxf_format(
            rootname, configs, outdir, emi_result, mc_results, formal_error,
            None, combined_bestfit, logLam, bin_data, noise, goodpixels_emi,
            emi_bestfit, emission_lines, line_names, line_wavelengths,
            npix, ubins, tie_settings, emission_config, z_field, z_error_field,
            group_order=group_order, debug=debug
        )
        ExGalutil.prettyOutput_Done("Saving emi kinematics in EMIPPXF format to disk")

    except Exception as e:
        ExGalutil.prettyOutput_Warning(f"Failed to create EMIPPXF-compatible output: {e}")
        logging.warning(f"Failed to create EMIPPXF-compatible output: {e}")




    ExGalutil.prettyOutput_Done("Saving emi kinematics results to disk")


    if debug:
        return emi_result, mc_results, formal_error, emi_bestfit, emission_lines, line_names, z_field
    else:
        return




# Per-bin PPXF+emission diagnostic plot ("ppxf_emi_<rootname>_bin_<id>.png"):
# ported to PyAPS.apsPlot.emi_bin (Plotly), which now supplies
# save_ppxf_emi_plot (imported above) as a drop-in replacement for the
# matplotlib version that used to live here.
