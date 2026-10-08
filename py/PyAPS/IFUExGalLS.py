from astropy.io import fits, ascii
import numpy as np
from multiprocessing import Queue, Process
from scipy.interpolate import interp1d

import time
import logging
import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys
import PyAPS
from PyAPS import ExGalutil
from PyAPS import ExGalPrepare
from PyAPS.apsPlot import ls_map  as util_plot_ls

from PyAPS import lsindex_spec   as lsindex
from PyAPS import ssppop_fitting as ssppop
from PyAPS import aps_constants
from astropy import units
APSVERS = PyAPS.__version__

import warnings
import traceback

from ppxf.ppxf_util import gaussian_filter1d

Clight  = 299792.458

"""
PURPOSE:
  This module executes the measurement of line strength indices in the IFU pipeline.
  IMPROVED VERSION with:
  - Proper velocity dispersion handling
  - Robust NaN value processing
  - Complex value error handling
  - Better resolution management
  - Enhanced error reporting
"""


def _pixel_index_range(first, last):
    """All pixel indices from `first` to `last`, both included. `first`/`last` come from np.where(...)[0] (one-element
    arrays); NumPy 2 no longer turns such an array into a Python scalar, so np.arange(first, last+1) raised
    "only 0-dimensional arrays can be converted to Python scalars" and every IFU ExGal job of a night failed."""
    return np.arange(int(np.ravel(first)[0]), int(np.ravel(last)[0]) + 1)


def workerLS(inQueue, outQueue):
    """
    Defines the worker process of the parallelisation with multiprocessing.Queue
    and multiprocessing.Process.
    IMPROVED: Better error handling for complex values and NaN data
    """
    for wave, spec, espec, redshift, configs, lickfile, names, index_names,\
        model_indices, params, tri, labels, outdir, nbins, i, MCMC\
        in iter(inQueue.get, 'STOP'):

        try:
            if MCMC == True:
                indices, errors, vals, percentile = run_ls( wave, spec, espec, redshift, configs, lickfile, names, index_names,\
                model_indices, params, tri, labels, outdir, nbins, i, MCMC )

                outQueue.put(( i, indices, errors, vals, percentile ))

            elif MCMC == False:
                indices, errors = run_ls( wave, spec, espec, redshift, configs, lickfile, names, index_names,\
                model_indices, params, tri, labels, outdir, nbins, i, MCMC )

                outQueue.put(( i, indices, errors ))

        except Exception as e:
            ExGalutil.prettyOutput_Warning(f"Worker process failed for bin {i}: {str(e)}")
            # Return NaN results for failed analysis
            n_indices = len(names) if 'names' in locals() else 89
            n_labels = len(labels) if 'labels' in locals() and labels != "dummy" else 5

            if MCMC:
                outQueue.put((i, np.full(n_indices, np.nan), np.full(n_indices, np.nan),
                            np.full(n_labels*3+2, np.nan), np.full((101, n_labels), np.nan)))
            else:
                outQueue.put((i, np.full(n_indices, np.nan), np.full(n_indices, np.nan)))

def run_ls(wave, spec, espec, redshift, configs, lickfile, names, index_names,\
           model_indices, params, tri, labels, outdir, nbins, i, MCMC):
    """
    IMPROVED: Calls line strength measurement with robust error handling
    - Handles NaN values intelligently
    - Manages complex number results
    - Provides detailed diagnostics
    """
    ExGalutil.printProgress(i, nbins, barLength = 50)
    nindex = len(index_names)

    try:
        # IMPROVED: Input validation with intelligent NaN handling
        total_pixels = len(spec)
        nan_pixels = np.sum(np.isnan(spec))
        nan_wave_pixels = np.sum(np.isnan(wave))

        # Check for completely invalid data
        if nan_pixels == total_pixels or nan_wave_pixels == len(wave):
            ExGalutil.prettyOutput_Warning(f"Bin {i}: Completely invalid spectrum (all NaN)")
            if MCMC:
                return( np.full(len(names), np.nan), np.full(len(names), np.nan),
                       np.full(len(labels)*3+2, np.nan), np.full((101, len(labels)), np.nan) )
            else:
                return( np.full(len(names), np.nan), np.full(len(names), np.nan) )

        # Handle partial NaN data (common for different redshift bins)
        if nan_pixels > 0:
            frac_nan = nan_pixels / total_pixels * 100

            # Only skip if >80% of spectrum is NaN (very conservative threshold)
            if frac_nan > 80.0:
                ExGalutil.prettyOutput_Warning(f"Bin {i}: Too much missing data ({frac_nan:.1f}% NaN), skipping")
                if MCMC:
                    return( np.full(len(names), np.nan), np.full(len(names), np.nan),
                           np.full(len(labels)*3+2, np.nan), np.full((101, len(labels)), np.nan) )
                else:
                    return( np.full(len(names), np.nan), np.full(len(names), np.nan) )

            # For small gaps, try interpolation
            if frac_nan < 5.0:  # Small gaps - interpolate
                valid_idx = ~np.isnan(spec)
                if np.sum(valid_idx) > 10:  # Need enough points for interpolation
                    try:
                        interp_func = interp1d(wave[valid_idx], spec[valid_idx],
                                             kind='linear', fill_value='extrapolate',
                                             bounds_error=False)
                        spec_clean = interp_func(wave)

                        # Replace NaN regions with interpolated values
                        spec[np.isnan(spec)] = spec_clean[np.isnan(spec)]
                    except:
                        pass  # If interpolation fails, proceed with NaN pixels

        # Handle error spectrum NaN values
        nan_error_pixels = np.sum(np.isnan(espec))
        if nan_error_pixels > 0:
            if nan_error_pixels == total_pixels:
                # Use simple Poisson errors as fallback
                espec = np.sqrt(np.abs(spec))
            else:
                # For partial NaN in errors, use median error estimate
                valid_err_idx = ~np.isnan(espec)
                if np.sum(valid_err_idx) > 10:
                    median_err = np.nanmedian(espec[valid_err_idx])
                    espec[np.isnan(espec)] = median_err

        # Check for non-positive flux (informational only)
        negative_flux = np.where(spec <= 0)[0]
        if len(negative_flux) > 0:
            frac_negative = len(negative_flux) / len(spec) * 100

            # Only skip for severe cases (>50% negative)
            if frac_negative > 50.0:
                ExGalutil.prettyOutput_Warning(f"Bin {i}: Severe data quality issue - {frac_negative:.1f}% negative flux")
                if MCMC:
                    return( np.full(len(names), np.nan), np.full(len(names), np.nan),
                           np.full(len(labels)*3+2, np.nan), np.full((101, len(labels)), np.nan) )
                else:
                    return( np.full(len(names), np.nan), np.full(len(names), np.nan) )

        # IMPROVED: Robust line strength measurement with complex value handling
        try:
            # Measure the LS indices
            names_out, indices_raw, errors_raw = lsindex.lsindex\
                        (wave, spec, espec, redshift[0], lickfile, sims=configs['MC_LS'], z_err=redshift[1], plot=0)

            # FIXED: Handle complex values that might arise from mathematical edge cases
            indices = np.real(indices_raw) if np.iscomplexobj(indices_raw) else indices_raw
            errors = np.real(errors_raw) if np.iscomplexobj(errors_raw) else errors_raw

            # Additional cleanup: handle any remaining problematic values
            indices = np.where(np.isfinite(indices), indices, np.nan)
            errors = np.where(np.isfinite(errors) & (errors >= 0), errors, np.nan)

            if np.iscomplexobj(indices_raw) or np.iscomplexobj(errors_raw):
                ExGalutil.prettyOutput_Warning(f"Bin {i}: Complex values detected in line strength results, converted to real")

        except Exception as e:
            ExGalutil.prettyOutput_Warning(f"Bin {i}: lsindex measurement failed: {str(e)}")
            if MCMC:
                return( np.full(len(names), np.nan), np.full(len(names), np.nan),
                       np.full(len(labels)*3+2, np.nan), np.full((101, len(labels)), np.nan) )
            else:
                return( np.full(len(names), np.nan), np.full(len(names), np.nan) )

        # IMPROVED: Get the indices with better error handling
        data  = np.zeros(nindex)
        error = np.zeros(nindex)

        for o in range( nindex ):
            idx = np.where( names_out == index_names[o] )[0]
            if len(idx) > 0:
                # FIXED: Handle complex values from line strength measurements
                index_val = indices[idx[0]]
                error_val = errors[idx[0]]

                # Check if values are complex and extract real part
                if np.iscomplexobj(index_val):
                    index_val = np.real(index_val)

                if np.iscomplexobj(error_val):
                    error_val = np.real(error_val)

                # Check if the measurement is valid (not NaN)
                if not np.isnan(index_val):
                    data[o]  = float(index_val)  # Ensure real float
                    error[o] = float(error_val)  # Ensure real float
                else:
                    # Index was attempted but returned NaN (e.g., outside wavelength range)
                    data[o] = np.nan
                    error[o] = np.nan
            else:
                # Index name not found in results
                data[o] = np.nan
                error[o] = np.nan

        # Count successful measurements
        valid_indices = np.sum(~np.isnan(data))
        total_indices = len(data)

        if valid_indices == 0:
            ExGalutil.prettyOutput_Warning(f"Bin {i}: No valid index measurements obtained")
        elif valid_indices < total_indices * 0.5:  # Less than 50% of indices measured
            ExGalutil.prettyOutput_Warning(f"Bin {i}: Only {valid_indices}/{total_indices} indices measured successfully")

        if MCMC == True:
            # IMPROVED: Check for sufficient valid data before MCMC
            if valid_indices < 5:  # Need at least 5 valid indices for meaningful MCMC
                ExGalutil.prettyOutput_Warning(f"Bin {i}: Insufficient data for MCMC ({valid_indices} valid indices), skipping MCMC")
                return(indices, errors, np.full(len(labels)*3+2, np.nan), np.full((101, len(labels)), np.nan))

            # Check for valid errors (needed for MCMC)
            valid_errors = np.sum(~np.isnan(error) & (error > 0))
            if valid_errors < valid_indices:
                # Use 10% errors as fallback for indices with invalid error estimates
                invalid_err_idx = np.isnan(error) | (error <= 0)
                error[invalid_err_idx] = 0.1 * np.abs(data[invalid_err_idx])

            # Run the conversion of LS indices to SSP properties
            vals   = np.zeros(len(labels)*3+2)
            chains = np.zeros((int(configs['NWALKER']*configs['NCHAIN']/2), len(labels)))

            try:
                vals[:], chains[:,:] = ssppop.ssppop_fitting\
                    (data, error, model_indices, params, tri, labels, configs['NWALKER'], configs['NCHAIN'], False, 0, i, nbins, outdir)
            except Exception as e:
                ExGalutil.prettyOutput_Warning(f"Bin {i}: MCMC fitting failed: {str(e)}")
                vals[:-1] = np.nan  # Mark as failed but preserve flag
                chains[:,:] = np.nan

            percentiles = np.percentile( chains, np.arange(101), axis=0 )

            return(indices, errors, vals, percentiles)

        elif MCMC == False:
            return(indices, errors)

    except Exception as e:
        ExGalutil.prettyOutput_Warning(f"Bin {i}: Complete analysis failure: {str(e)}")
        if MCMC == True:
            return( np.full(len(names), np.nan), np.full(len(names), np.nan),
                   np.full(len(labels)*3+2, np.nan), np.full((101, len(labels)), np.nan) )
        elif MCMC == False:
            return( np.full(len(names), np.nan), np.full(len(names), np.nan) )

# def save_ls(names, configs, ls_indices, ls_errors, index_names, labels, RESOLUTION, MCMC, totalFWHM_flag, outdir, rootname, ubins, vals=None, percentile=None ):
#     """
#     IMPROVED: Saves all results to disk with enhanced documentation and error handling
#     """
#     outfits = (outdir+rootname+'_ls_'+RESOLUTION+'.fits').replace(" ", "")
#     ExGalutil.prettyOutput_Running("Writing: "+str(outfits))

#     ## add Units to flux, model and ivar in the ppxf's spec table
#     if configs['sens_corr']:
#         flux_unit_str = '%2e erg/(s cm**2 Angstrom)' %(configs['funits'])
#         ivar_unit_str = '%2e cm**4 Angstrom**2 /(s**2 erg**2)' %(configs['funits']**-2)
#     else:
#         flux_unit_str = 'count'
#         ivar_unit_str = '1/count**2'
#     wave_unit_str = 'Angstrom'

#     # Primary HDU
#     priHDU = fits.PrimaryHDU()

#     # Extension 1: Table HDU with LS output data
#     cols = []
#     ndim  = len(names)

#     ## Assuming data is coming from the _emippxf-cleaned_BIN.fits data, ubins is supose to indicate the BIN ID
#     cols.append( fits.Column(name='BIN_ID',                                format='J', array=ubins             ))

#     # IMPROVED: Ensure all data is real before saving
#     for i in range(ndim):
#         # Handle any remaining complex values
#         index_data = np.real(ls_indices[:,i]) if np.iscomplexobj(ls_indices[:,i]) else ls_indices[:,i]
#         error_data = np.real(ls_errors[:,i]) if np.iscomplexobj(ls_errors[:,i]) else ls_errors[:,i]

#         cols.append( fits.Column(name=names[i],        unit=flux_unit_str, format='D', array=index_data   ))
#         cols.append( fits.Column(name="ERR_"+names[i], unit=flux_unit_str, format='D', array=error_data   ))

#     cols.append( fits.Column(name="FWHM_FLAG",                             format='I', array=totalFWHM_flag[:] ))
#     lsHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
#     lsHDU.name = "LS_TABLE"

#     # IMPROVED: Enhanced header documentation
#     lsHDU.header['CNF_LS'] = (configs['CONFIG_FILE'],'Configs. Filename used by LS')
#     lsHDU.header['LS_RES'] = (RESOLUTION,'LS Resolution mode')
#     lsHDU.header['CONV_COR'] = (configs['CONV_COR'],'Resolution of the index measurement [Ang.]')
#     lsHDU.header['MC_LS'] = (configs['MC_LS'],'N. of MC simulations to extract err on the LS indices')
#     lsHDU.header['LS_FILE'] = (configs['LS_FILE'],'Line indices filename')

#     # ADDED: Document FWHM flag meanings
#     lsHDU.header['FWHM_FL0'] = ('Normal processing - no resolution issues')
#     lsHDU.header['FWHM_FL1'] = ('Target resolution < total resolution at some wavelengths')
#     lsHDU.header['FWHM_FL2'] = ('Extreme broadening required (>100 pixel sigma)')
#     lsHDU.header['FWHM_FL3'] = ('Convolution failed - returned unconvolved spectrum')
#     lsHDU.header['FWHM_FL4'] = ('Complete analysis failure')

#     lsHDU.header['TEMPL'] = (configs['SSP_LIB'],'The library of spectral templates')
#     lsHDU.header['LS_V'] = (aps_constants.__ls_version__, 'LS version')
#     lsHDU.header['APSVERS'] = (APSVERS,'APS version')
#     lsHDU.header['APSLSV'] = (aps_constants.__aps_ifuls_version__,'PyAPS (IFU) LS wrapper version')
#     lsHDU.header['CSB_LS'] = (configs['stitched'], 'Combines Spectral Bands Status for LS')

#     # ADDED: Document velocity dispersion and resolution handling
#     lsHDU.header['VEL_DISP'] = ('TRUE', 'Velocity dispersion included in resolution calculation')
#     lsHDU.header['LS_MODE'] = (configs.get('LS_RESOLUTION_MODE', 'SKIP_NEGATIVE'), 'Method for handling negative FWHM')

#     #keep the basename of input file (infiles) to save as provinces later
#     for n_province, province in enumerate(configs['infiles']):
#         lsHDU.header['APSREF_%d' %(n_province)] = (os.path.basename(province), 'L1 reference file')

#     lsHDU.header['NLS'] = (len(names), 'Number of indices, applied to LS')
#     for nii, ii in enumerate(names):
#         lsHDU.header['LS_%s' %(nii)]    = '%30s' %(names[nii])

#     if MCMC == True:
#         # Extension 2: Table HDU with SSP-equivalent output data
#         nparam  = len(labels)
#         cols = []

#         cols.append( fits.Column(name='BIN_ID',         format='J', array=ubins             ))
#         for i in range(nparam):
#             # IMPROVED: Handle complex values in MCMC results
#             percentile_data = np.real(percentile[:,:,i]) if np.iscomplexobj(percentile[:,:,i]) else percentile[:,:,i]
#             cols.append( fits.Column(name=labels[i],           format='101D', array=percentile_data ))

#         vals_lnP = np.real(vals[:,-2]) if np.iscomplexobj(vals[:,-2]) else vals[:,-2]
#         vals_flag = np.real(vals[:,-1]) if np.iscomplexobj(vals[:,-1]) else vals[:,-1]

#         cols.append( fits.Column(    name='lnP',               format='D',    array=vals_lnP        ))
#         cols.append( fits.Column(    name='Flag',              format='D',    array=vals_flag       ))
#         sspHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
#         sspHDU.name = "SSP_TABLE"

#         sspHDU.header['CNF_LS'] = (configs['CONFIG_FILE'],'Configs. Filename used by LS')
#         sspHDU.header['LS_RES'] = (RESOLUTION,'LS Resolution mode')
#         sspHDU.header['NWALKER'] = (configs['NWALKER'],'N. of walkers for the MCMC algorithm for SSP)')
#         sspHDU.header['NCHAIN'] = (configs['NCHAIN'],'N. of iteration for the MCMC algorithm for SSP)')
#         sspHDU.header['LS_FILE'] = (configs['LS_FILE'],'Line indices filename')

#         sspHDU.header['TEMPL'] = (configs['SSP_LIB'],'The library of spectral templates')
#         sspHDU.header['LS_V'] = (aps_constants.__ls_version__, 'LS version')
#         sspHDU.header['APSVERS'] = (APSVERS,'APS version')
#         sspHDU.header['APSLSV'] = (aps_constants.__aps_ls_version__,'PyAPS (IFU) LS wrapper version')
#         sspHDU.header['CSB_LS'] = (configs['stitched'], 'Combines Spectral Bands Status for LS')

#         #keep the basename of input file (infiles) to save as provinces later
#         for n_province, province in enumerate(configs['infiles']):
#             sspHDU.header['APSREF_%d' %(n_province)] = (os.path.basename(province), 'L1 reference file')

#         sspHDU.header['NLS'] = (len(names), 'Number of indices, applied to LS')
#         for nii, ii in enumerate(names):
#             sspHDU.header['LS_%s' %(nii)]    = '%30s' %(names[nii])

#         # Create HDU list
#         HDUList = fits.HDUList([priHDU, lsHDU, sspHDU])

#     if MCMC == False:
#         # Create HDU list
#         HDUList = fits.HDUList([priHDU, lsHDU])

#     # Write HDU list to file
#     HDUList.writeto(outfits, overwrite=True)

#     ExGalutil.prettyOutput_Done("Writing: "+str(outfits))
#     logging.info("Wrote: "+str(outfits))

def save_ls(names, configs, ls_indices, ls_errors, index_names, labels, RESOLUTION, MCMC, totalFWHM_flag, outdir, rootname, ubins, config_dir, vals=None, percentile=None):
    """
    IMPROVED: Saves all results to disk with CORRECT units and enhanced documentation

    FIXED: Units now properly assigned based on index type:
           - D4000: dimensionless (flux ratio)
           - b7=1: Angstrom (equivalent width)
           - b7=2: mag (magnitude indices)

    Args:
        names: List of all index names
        configs: Configuration dictionary
        ls_indices: Array of measured index values
        ls_errors: Array of index errors
        index_names: List of index names used for SSP fitting
        labels: SSP parameter labels
        RESOLUTION: Resolution mode string
        MCMC: Boolean indicating if MCMC was performed
        totalFWHM_flag: Array of FWHM flags for each spectrum
        outdir: Output directory path
        rootname: Base name for output files
        ubins: Array of unique bin IDs
        config_dir: Configuration directory path (ADDED for unit handling)
        vals: SSP parameter values (optional, for MCMC)
        percentile: SSP parameter percentiles (optional, for MCMC)
    """

    # Load Lick indices file to get index types
    lickfile = config_dir + configs['LS_FILE']
    lick_table = ascii.read(lickfile, comment='\\s*#')

    def get_index_unit(index_name):
        """
        Return appropriate unit based on Lick file definition.

        Args:
            index_name: Name of the spectral index

        Returns:
            String with appropriate unit:
            - '' (empty string) for dimensionless (D4000)
            - 'Angstrom' for equivalent width indices (b7=1)
            - 'mag' for magnitude indices (b7=2)
        """
        # Special case: D4000 is always dimensionless (flux ratio)
        # D4000 = F_red / F_blue, so it has no units
        if index_name == 'D4000':
            return ''

        # Find index in Lick table
        idx_row = np.where(lick_table['names'] == index_name)[0]

        if len(idx_row) == 0:
            # Index not found - default to Angstrom (most common)
            ExGalutil.prettyOutput_Warning(f"Index {index_name} not found in Lick file, defaulting to Angstrom")
            return 'Angstrom'

        # Check b7 column: 1=EW (Angstrom), 2=magnitude
        index_type = lick_table['b7'][idx_row[0]]

        if index_type == 1:
            # Equivalent width index
            return 'Angstrom'
        elif index_type == 2:
            # Magnitude index (dimensionless but expressed in mag)
            return 'mag'
        else:
            # Unknown type - default to Angstrom
            ExGalutil.prettyOutput_Warning(f"Unknown index type {index_type} for {index_name}, defaulting to Angstrom")
            return 'Angstrom'

    outfits = (outdir + rootname + '_ls_' + RESOLUTION + '.fits').replace(" ", "")
    ExGalutil.prettyOutput_Running("Writing: " + str(outfits))

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # ========================
    # Extension 1: Table HDU with LS output data
    # ========================
    cols = []
    ndim = len(names)

    # Assuming data is coming from the _emippxf-cleaned_BIN.fits data, ubins is supposed to indicate the BIN ID
    cols.append(fits.Column(name='BIN_ID', format='J', array=ubins))

    # IMPROVED: Add each index with proper unit based on Lick file
    for i in range(ndim):
        # Get correct unit for each index based on Lick file
        index_unit = get_index_unit(names[i])

        # Handle any remaining complex values
        index_data = np.real(ls_indices[:,i]) if np.iscomplexobj(ls_indices[:,i]) else ls_indices[:,i]
        error_data = np.real(ls_errors[:,i]) if np.iscomplexobj(ls_errors[:,i]) else ls_errors[:,i]

        # Add index value column with proper unit
        cols.append(fits.Column(name=names[i],
                                unit=index_unit,
                                format='D',
                                array=index_data))

        # Add index error column with same unit
        cols.append(fits.Column(name="ERR_"+names[i],
                                unit=index_unit,
                                format='D',
                                array=error_data))

    # Add FWHM flag column
    cols.append(fits.Column(name="FWHM_FLAG", format='I', array=totalFWHM_flag[:]))

    lsHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    lsHDU.name = "LS_TABLE"

    # ========================
    # Header keywords for LS_TABLE
    # ========================
    lsHDU.header['CNF_LS'] = (configs['CONFIG_FILE'], 'Configs. Filename used by LS')
    ExGalPrepare.write_ebmv_header(lsHDU.header, configs)
    lsHDU.header['LS_RES'] = (RESOLUTION, 'LS Resolution mode')
    lsHDU.header['CONV_COR'] = (configs['CONV_COR'], 'Resolution of the index measurement [Ang.]')
    lsHDU.header['MC_LS'] = (configs['MC_LS'], 'N. of MC simulations to extract err on the LS indices')
    lsHDU.header['LS_FILE'] = (configs['LS_FILE'], 'Line indices filename')

    # IMPROVED: Document unit system
    lsHDU.header['COMMENT'] = '=========================================='
    lsHDU.header['COMMENT'] = 'INDEX UNITS (determined from Lick file):'
    lsHDU.header['COMMENT'] = '=========================================='
    lsHDU.header['COMMENT'] = 'Equivalent Width indices (b7=1): Angstrom'
    lsHDU.header['COMMENT'] = 'Magnitude indices (b7=2): mag'
    lsHDU.header['COMMENT'] = 'D4000 (flux ratio): dimensionless'
    lsHDU.header['COMMENT'] = '=========================================='

    # IMPROVED: Document FWHM flag meanings
    lsHDU.header['COMMENT'] = '=========================================='
    lsHDU.header['COMMENT'] = 'FWHM_FLAG VALUES:'
    lsHDU.header['COMMENT'] = '=========================================='
    lsHDU.header['FWHM_FL0'] = 'Normal processing - no resolution issues'
    lsHDU.header['FWHM_FL1'] = 'Target resolution < total resolution at some wavelengths'
    lsHDU.header['FWHM_FL2'] = 'Extreme broadening required (>100 pixel sigma)'
    lsHDU.header['FWHM_FL3'] = 'Convolution failed - returned unconvolved spectrum'
    lsHDU.header['FWHM_FL4'] = 'Complete analysis failure'
    lsHDU.header['COMMENT'] = '=========================================='

    lsHDU.header['TEMPL'] = (configs['SSP_LIB'], 'The library of spectral templates')
    lsHDU.header['LS_V'] = (aps_constants.__ls_version__, 'LS version')
    lsHDU.header['APSVERS'] = (APSVERS, 'APS version')
    lsHDU.header['APSLSV'] = (aps_constants.__aps_ifuls_version__, 'PyAPS (IFU) LS wrapper version')
    lsHDU.header['CSB_LS'] = (configs['stitched'], 'Combines Spectral Bands Status for LS')

    # Document velocity dispersion and resolution handling
    lsHDU.header['VEL_DISP'] = ('TRUE', 'Velocity dispersion included in resolution calculation')
    lsHDU.header['LS_MODE'] = (configs.get('LS_RESOLUTION_MODE', 'SKIP_NEGATIVE'),
                               'Method for handling negative FWHM')

    # Keep the basename of input file (infiles) to save as provinces later
    for n_province, province in enumerate(configs['infiles']):
        lsHDU.header['APSREF_%d' % (n_province)] = (os.path.basename(province), 'L1 reference file')

    # Document which indices were measured
    lsHDU.header['NLS'] = (len(names), 'Number of indices, applied to LS')
    for nii, ii in enumerate(names):
        lsHDU.header['LS_%s' % (nii)] = '%30s' % (names[nii])

    # ========================
    # Extension 2: SSP_TABLE (if MCMC was performed)
    # ========================
    if MCMC == True:
        nparam = len(labels)
        cols = []

        cols.append(fits.Column(name='BIN_ID', format='J', array=ubins))

        for i in range(nparam):
            # IMPROVED: Handle complex values in MCMC results
            percentile_data = np.real(percentile[:,:,i]) if np.iscomplexobj(percentile[:,:,i]) else percentile[:,:,i]
            cols.append(fits.Column(name=labels[i], format='101D', array=percentile_data))

        # Handle complex values in vals array
        vals_lnP = np.real(vals[:,-2]) if np.iscomplexobj(vals[:,-2]) else vals[:,-2]
        vals_flag = np.real(vals[:,-1]) if np.iscomplexobj(vals[:,-1]) else vals[:,-1]

        cols.append(fits.Column(name='lnP', format='D', array=vals_lnP))
        cols.append(fits.Column(name='Flag', format='D', array=vals_flag))

        sspHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
        sspHDU.name = "SSP_TABLE"

        # Header keywords for SSP_TABLE
        sspHDU.header['CNF_LS'] = (configs['CONFIG_FILE'], 'Configs. Filename used by LS')
        ExGalPrepare.write_ebmv_header(sspHDU.header, configs)
        sspHDU.header['LS_RES'] = (RESOLUTION, 'LS Resolution mode')
        sspHDU.header['NWALKER'] = (configs['NWALKER'], 'N. of walkers for the MCMC algorithm for SSP)')
        sspHDU.header['NCHAIN'] = (configs['NCHAIN'], 'N. of iteration for the MCMC algorithm for SSP)')
        sspHDU.header['LS_FILE'] = (configs['LS_FILE'], 'Line indices filename')

        sspHDU.header['TEMPL'] = (configs['SSP_LIB'], 'The library of spectral templates')
        sspHDU.header['LS_V'] = (aps_constants.__ls_version__, 'LS version')
        sspHDU.header['APSVERS'] = (APSVERS, 'APS version')
        sspHDU.header['APSLSV'] = (aps_constants.__aps_ls_version__, 'PyAPS (IFU) LS wrapper version')
        sspHDU.header['CSB_LS'] = (configs['stitched'], 'Combines Spectral Bands Status for LS')

        # Keep the basename of input file (infiles) to save as provinces later
        for n_province, province in enumerate(configs['infiles']):
            sspHDU.header['APSREF_%d' % (n_province)] = (os.path.basename(province), 'L1 reference file')

        sspHDU.header['NLS'] = (len(names), 'Number of indices, applied to LS')
        for nii, ii in enumerate(names):
            sspHDU.header['LS_%s' % (nii)] = '%30s' % (names[nii])

        # Create HDU list with SSP_TABLE
        HDUList = fits.HDUList([priHDU, lsHDU, sspHDU])

    else:
        # Create HDU list without SSP_TABLE
        HDUList = fits.HDUList([priHDU, lsHDU])

    # Write HDU list to file
    HDUList.writeto(outfits, overwrite=True)

    ExGalutil.prettyOutput_Done("Writing: " + str(outfits))
    logging.info("Wrote: " + str(outfits))


def saveCleanedLinearSpectra(spec, espec, wave, npix, outdir, rootname):
    """ Save emission-subtracted, linearly binned spectra to disk. """
    outfits = outdir+rootname+'_ls-cleaned_linear.fits'
    ExGalutil.prettyOutput_Running("Writing: "+rootname+'_ls-cleaned_linear.fits')

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Extension 1: Table HDU with cleaned, linear spectra
    cols = []
    cols.append( fits.Column(name='SPEC',  format=str(npix)+'D', array=spec ) )
    cols.append( fits.Column(name='ESPEC', format=str(npix)+'D', array=espec ) )
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = 'CLEANED_SPECTRA'

    # Extension 2: Table HDU with wave
    cols = []
    cols.append( fits.Column(name='LAM', format='D', array=wave) )
    logLamHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    logLamHDU.name = 'LAM'

    # Create HDU list and write to file
    HDUList = fits.HDUList([priHDU, dataHDU, logLamHDU])
    HDUList.writeto(outfits, overwrite=True)

    ExGalutil.prettyOutput_Done("Writing: "+rootname+'_ls-cleaned_linear.fits')
    logging.info("Wrote: "+outfits)

def log_unbinning(lamRange, spec, oversample=1, flux=True):
    """
    This function transforms logarithmically binned spectra back to linear
    binning. It is a Python translation of Michele Cappellari's
    "log_rebin_invert" function. Thanks to Michele Cappellari for his permission
    to include this function in the pipeline.
    """
    # Length of arrays
    n = len(spec)
    m = n * oversample

    # Log space
    dLam = (lamRange[1]-lamRange[0]) / (n - 1)             # Step in log-space
    lim = lamRange + np.array([-0.5, 0.5])*dLam            # Min and max wavelength in log-space
    borders = np.linspace( lim[0], lim[1], n+1 )           # OLD logLam in log-space

    # Wavelength domain
    logLim     = np.exp(lim)                               # Min and max wavelength in Angst.
    lamNew     = np.linspace( logLim[0], logLim[1], m+1 )  # new logLam in Angstroem
    newBorders = np.log(lamNew)                            # new logLam in log-space

    # Translate indices of arrays so that newBorders[j] corresponds to borders[k[j]]
    k = np.floor( (newBorders-lim[0]) / dLam ).astype('int')

    # Construct new spectrum
    specNew = np.zeros(m)
    for j in range(0, m-1):
        a = (newBorders[j]   - borders[k[j]])   / dLam
        b = (borders[k[j+1]] - newBorders[j+1]) / dLam

        specNew[j] = np.sum( spec[k[j]:k[j+1]] ) - a*spec[k[j]] - b*spec[k[j+1]]

    # Rescale flux
    if flux == True:
        specNew = specNew / ( newBorders[1:] - newBorders[:-1] ) * np.mean( newBorders[1:] - newBorders[:-1] ) * oversample

    # Shift back the wavelength arrays
    lamNew = lamNew[:-1] + 0.5 * (lamNew[1]-lamNew[0])

    return( specNew, lamNew )

def runModule_LINESTRENGTH(LINE_STRENGTH, RESOLUTION, nthreads, configs, velscale, LSF_Data, outdir, config_dir, templates_dir, figdir, rootname, debug=False, bin_to_bucket=None, LSF_Data_by_bucket=None):
    """
    IMPROVED: Starts the line strength analysis with comprehensive error handling
    - Proper velocity dispersion inclusion
    - Robust NaN handling for different redshift bins
    - Complex value management
    - Enhanced resolution diagnostics

    `bin_to_bucket`/`LSF_Data_by_bucket` (both optional, default `None`):
    opt-in per-bin resolution, same contract as `IFUExGalPPXF.
    runModule_PPXF`'s own matching parameters. This module already
    re-evaluates `LSF_Data(wave)` fresh inside its own per-bin loop, so
    wiring this in is a one-line change there -- no template preparation
    to bucket here at all (line-strength broadening is applied directly
    to the data, not via a convolved template library).
    """
    # Run MCMC only on the indices measured from convoluted spectra
    if LINE_STRENGTH == 2  and  RESOLUTION == "ADAPTED":
        MCMC = True
    else:
        MCMC = False

    # Double check what to do
    if RESOLUTION == "ADAPTED"  and  LINE_STRENGTH != 0:
        SKIP = False
    elif RESOLUTION == "ORIGINAL" and  LINE_STRENGTH != 0:
        SKIP = False
    else:
        SKIP = True

    if SKIP == False:
        print("\033[0;37m"+" - - - - - Running LINE STRENGTHS (IFU) - - - - -"+"\033[0;39m")
        logging.info(" - - - Running LINE STRENGTHS (IFU) - - - ")

        # ADDED: Validate target resolution before processing
        target_resolution = configs.get('CONV_COR', 8.4)
        logging.info(f"Target resolution (CONV_COR): {target_resolution:.1f} Å FWHM")

        # ADDED: Set default resolution handling mode if not specified
        if 'LS_RESOLUTION_MODE' not in configs:
            configs['LS_RESOLUTION_MODE'] = 'SKIP_NEGATIVE'
            logging.info("Using default resolution mode: SKIP_NEGATIVE")

        # Read cleaned spectra
        logging.info("Reading "+outdir+rootname+"_emippxf_spec_BIN.fits")
        hdu_emi  = fits.open(outdir+rootname+'_emippxf_spec_BIN.fits')
        hdu_Vor = fits.open(outdir+rootname+'_BINSpectra.fits')

        if hdu_Vor[1].data.LOGLAM.ndim > 1:
            Vor_LOGLAM= hdu_Vor[1].data.LOGLAM[0,:]
        else:
            Vor_LOGLAM= hdu_Vor[1].data.LOGLAM

        ## Just in case, if we just had one spaxel or Voronoi Bin:
        if hdu_emi[1].data.LOGLAM_EMI.ndim > 1:
            idx_lamMin = np.where( hdu_emi[1].data.LOGLAM_EMI[0,:][0]  == Vor_LOGLAM )[0]
            idx_lamMax = np.where( hdu_emi[1].data.LOGLAM_EMI[0,:][-1] == Vor_LOGLAM )[0]
            wave     = np.array( hdu_emi[1].data.LOGLAM_EMI[0,:] )
        else:
            idx_lamMin = np.where( hdu_emi[1].data.LOGLAM_EMI[0]  == Vor_LOGLAM )[0]
            idx_lamMax = np.where( hdu_emi[1].data.LOGLAM_EMI[-1] == Vor_LOGLAM )[0]
            wave     = np.array( hdu_emi[1].data.LOGLAM_EMI )

        idx_lam    = _pixel_index_range(idx_lamMin, idx_lamMax)
        oldspec  = np.array( hdu_emi[1].data.FLUX_CLEAN_EMI   )
        oldespec = np.sqrt( np.array( hdu_Vor[1].data.ESPEC )[:,idx_lam] )

        nbins    = oldspec.shape[0]
        ubins    = np.arange(0, nbins)
        npix     = oldspec.shape[1]
        lamRange = np.array([ wave[0], wave[-1] ])
        spec     = np.zeros( oldspec.shape  )
        espec    = np.zeros( oldespec.shape )

        # Rebin the cleaned spectra from log to lin
        ExGalutil.prettyOutput_Running("Rebinning the cleaned spectra from log to lin")
        for i in range( nbins ):
            ExGalutil.printProgress(i, nbins, barLength = 50)
            spec[i,:], wave = log_unbinning( lamRange, oldspec[i,:] )
        ExGalutil.prettyOutput_Done("Rebinning the cleaned spectra from log to lin", progressbar=True)

        # Rebin the error spectra from log to lin
        ExGalutil.prettyOutput_Running("Rebinning the error spectra from log to lin")
        for i in range( nbins ):
            ExGalutil.printProgress(i, nbins, barLength = 50)
            espec[i,:], _ = log_unbinning( lamRange, oldespec[i,:] )
        ExGalutil.prettyOutput_Done("Rebinning the error spectra from log to lin", progressbar=True)

        # Save cleaned, linear spectra
        saveCleanedLinearSpectra(spec, espec, wave, npix, outdir, rootname)

        # Read PPXF results
        ppxf_data     = fits.open(outdir+rootname+'_ppxf.fits')[1].data
        redshift      = np.zeros((nbins, 2))                       # Dimensionless z

        # FIXED: Improved redshift calculation for high-z targets
        redshift[:,0] = np.exp(np.array( ppxf_data.V[:] )/Clight) - 1 ## Corrected redshift for high redshift targets
        redshift[:,1] = np.exp(np.array(ppxf_data.FORM_ERR_V[:])/Clight) - 1 ## Corrected ERROR in redshift for high redshift targets

        veldisp_kin   = np.array( ppxf_data.SIGMA[:] )

        # ADDED: Validate velocity dispersion values
        invalid_veldisp = np.where((veldisp_kin <= 0) | (veldisp_kin > 1000))[0]
        if len(invalid_veldisp) > 0:
            ExGalutil.prettyOutput_Warning(f"Found {len(invalid_veldisp)} bins with invalid velocity dispersions")
            ExGalutil.prettyOutput_Warning(f"  Range: {veldisp_kin[invalid_veldisp].min():.1f} - {veldisp_kin[invalid_veldisp].max():.1f} km/s")
            # Set minimum reasonable velocity dispersion
            veldisp_kin[invalid_veldisp] = np.clip(veldisp_kin[invalid_veldisp], 30.0, 800.0)

        logging.info(f"Velocity dispersion range: {veldisp_kin.min():.1f} - {veldisp_kin.max():.1f} km/s")

        # Read file defining the LS bands
        lickfile = config_dir+configs['LS_FILE']
        tab   = ascii.read(lickfile, comment='\\s*#')
        names = tab['names']

        # Flag spectra for which the total intrinsic dispersion is larger than the LIS measurement resolution
        totalFWHM_flag = np.zeros(spec.shape[0], dtype=int)

        # FIXED: Broaden spectra to LIS resolution INCLUDING velocity dispersion
        # This was incorrectly commented out - velocity dispersion MUST be included
        if RESOLUTION == "ADAPTED":
            ExGalutil.prettyOutput_Running("Broadening the spectra to LIS resolution")
            # Iterate over all bins
            for i in range(0, spec.shape[0]):
                ExGalutil.printProgress(i, nbins, barLength = 50)

                # FIXED: Convert velocity dispersion of galaxy (from PPXF) to Angstroms
                # This was incorrectly commented out - velocity dispersion MUST be included
                veldisp_kin_Angst = veldisp_kin[i] * wave / Clight * 2.355

                # FIXED: Total dispersion for this bin includes BOTH instrumental and velocity dispersion
                # This is the physically correct approach for line strength measurements
                lsf_data_bin = LSF_Data_by_bucket[bin_to_bucket[i]] if LSF_Data_by_bucket is not None else LSF_Data
                total_dispersion = np.sqrt( lsf_data_bin(wave)**2 + veldisp_kin_Angst**2 )

                # Target resolution from config
                target_resolution = configs['CONV_COR']

                # ADDED: Check for problematic cases before computing square root
                FWHM_dif_squared = target_resolution**2 - total_dispersion**2

                # ADDED: Detailed handling of negative FWHM differences
                negative_idx = np.where(FWHM_dif_squared < 0)[0]

                if len(negative_idx) > 0:
                    # Log detailed information about the problem
                    min_wave = wave[negative_idx].min()
                    max_wave = wave[negative_idx].max()
                    max_total_disp = total_dispersion[negative_idx].max()

                    ExGalutil.prettyOutput_Warning(
                        f"BIN_ID {i}: Total resolution ({max_total_disp:.3f} Å) > target ({target_resolution:.1f} Å) "
                        f"at {len(negative_idx)} wavelengths ({min_wave:.0f}-{max_wave:.0f} Å)"
                    )
                    ExGalutil.prettyOutput_Warning(
                        f"  Velocity dispersion: {veldisp_kin[i]:.1f} km/s, "
                        f"Max veldisp contribution: {veldisp_kin_Angst[negative_idx].max():.3f} Å"
                    )

                    # Set flag to indicate this spectrum has resolution issues
                    totalFWHM_flag[i] = 1

                    # ADDED: Handle negative cases based on configuration
                    if configs.get('LS_RESOLUTION_MODE', 'SKIP_NEGATIVE') == 'SKIP_NEGATIVE':
                        # Set problematic wavelengths to zero broadening
                        FWHM_dif_squared[negative_idx] = 0.0
                        ExGalutil.prettyOutput_Warning("  -> Setting zero additional broadening for problematic wavelengths")

                    elif configs.get('LS_RESOLUTION_MODE', 'SKIP_NEGATIVE') == 'ADAPTIVE_TARGET':
                        # Adaptively increase target resolution
                        new_target = np.ceil(total_dispersion.max() * 1.1)  # 10% margin
                        FWHM_dif_squared = new_target**2 - total_dispersion**2
                        ExGalutil.prettyOutput_Warning(f"  -> Adaptively increasing target to {new_target:.1f} Å")

                    else:  # Default: SKIP_NEGATIVE
                        FWHM_dif_squared[negative_idx] = 0.0

                # Compute additional broadening needed
                FWHM_dif = np.sqrt(np.maximum(FWHM_dif_squared, 0.0))  # Ensure non-negative

                # Convert resolution difference from Angstrom to pixel
                sigma = (FWHM_dif / wave) * Clight / 2.355 / velscale

                # ADDED: Additional validation for extreme broadening
                max_sigma = sigma.max()
                if max_sigma > 50:  # More than 50 pixels FWHM
                    ExGalutil.prettyOutput_Warning(
                        f"BIN_ID {i}: Extreme broadening required (max σ = {max_sigma:.1f} pixels)"
                    )
                    if max_sigma > 100:  # Very extreme case
                        totalFWHM_flag[i] = 2  # Flag for extreme broadening

                # ADDED: Handle any remaining NaN values
                nan_idx = np.where(np.isnan(sigma))[0]
                if len(nan_idx) > 0:
                    sigma[nan_idx] = 0.0
                    totalFWHM_flag[i] = max(totalFWHM_flag[i], 1)

                # IMPROVED: Convolve spectra pixel-wise with better error handling
                try:
                    max_sigma_value = sigma.max()

                    if max_sigma_value > 0.1:  # Only convolve if significant broadening needed
                        # Adaptive truncate parameter
                        if max_sigma_value < 10:
                            truncate = 4.0  # Standard truncation
                        elif max_sigma_value < 50:
                            truncate = 3.0  # Reduce for large kernels
                        else:
                            truncate = 2.0  # Minimal for very large kernels

                        spec_orig = spec[i,:].copy()  # Keep original for fallback
                        espec_orig = espec[i,:].copy()

                        spec[i,:]  = gaussian_filter1d(spec[i,:],  sigma, truncate=truncate, mode='nearest')
                        espec[i,:] = gaussian_filter1d(espec[i,:], sigma, truncate=truncate, mode='nearest')

                        # Validate convolution results
                        if np.any(np.isnan(spec[i,:])) or np.any(np.isinf(spec[i,:])):
                            ExGalutil.prettyOutput_Warning(f"BIN_ID {i}: Convolution produced invalid values, using original")
                            spec[i,:] = spec_orig
                            espec[i,:] = espec_orig
                            totalFWHM_flag[i] = 3

                except Exception as e:
                    ExGalutil.prettyOutput_Warning(f"BIN_ID {i}: Convolution failed: {str(e)}")
                    totalFWHM_flag[i] = 3  # Flag for convolution failure

            ExGalutil.prettyOutput_Done("Broadening the spectra to LIS resolution", progressbar=True)

        # Get indices that are considered in SSP-conversion
        idx         = np.where( tab['spp'] == 1 )[0]
        index_names = tab['names'][idx].tolist()

        # Loading model predictions
        if MCMC == True:
            modelfile = config_dir + configs['SSP_LIB']+"_KB_LIS"+str(configs['CONV_COR'])+".fits"
            try:
                model_indices, params, tri, labels = ssppop.load_models(modelfile, index_names)
                logging.info("Loading LS model file at "+modelfile)
            except Exception as e:
                ExGalutil.prettyOutput_Warning(f"Failed to load MCMC models: {str(e)}")
                MCMC = False  # Disable MCMC if models can't be loaded
                model_indices, params, tri, labels = "dummy", "dummy", "dummy", "dummy"
        elif MCMC == False:
            model_indices, params, tri, labels = "dummy", "dummy", "dummy", "dummy"

        # Arrays to store results
        ls_indices = np.zeros((nbins, len(names)))
        ls_errors  = np.zeros((nbins, len(names)))
        if MCMC == True:
            vals       = np.zeros((nbins, len(labels)*3+2))
            percentile = np.zeros((nbins, 101, len(labels)))

        # ADDED: Pre-analysis diagnostics
        logging.info("=== IFU PRE-ANALYSIS DIAGNOSTICS ===")
        logging.info(f"Number of spectra: {nbins}")
        logging.info(f"Spectral pixels: {npix}")
        logging.info(f"Wavelength range: {wave.min():.1f} - {wave.max():.1f} Å")
        logging.info(f"Target resolution: {configs['CONV_COR']:.1f} Å FWHM")
        logging.info(f"Resolution mode: {configs.get('LS_RESOLUTION_MODE', 'SKIP_NEGATIVE')}")
        logging.info(f"Number of line indices: {len(names)}")
        logging.info(f"MCMC enabled: {MCMC}")

        # ADDED: Wavelength coverage diagnostics
        logging.info("=== WAVELENGTH COVERAGE ANALYSIS ===")
        wave_min, wave_max = wave.min(), wave.max()
        logging.info(f"Spectral coverage: {wave_min:.1f} - {wave_max:.1f} Å")

        # Check which indices are outside wavelength coverage
        for i, index_name in enumerate(names):
            idx_row = np.where(tab['names'] == index_name)[0]
            if len(idx_row) > 0:
                try:
                    index_wave_min = tab['b1'][idx_row[0]]  # Blue continuum start
                    index_wave_max = tab['b6'][idx_row[0]]  # Red continuum end

                    if index_wave_min < wave_min or index_wave_max > wave_max:
                        coverage_issue = []
                        if index_wave_min < wave_min:
                            coverage_issue.append(f"needs {index_wave_min:.0f} Å (have {wave_min:.0f} Å)")
                        if index_wave_max > wave_max:
                            coverage_issue.append(f"needs {index_wave_max:.0f} Å (have {wave_max:.0f} Å)")
                        if debug:
                            logging.warning(f"Index {index_name}: Outside coverage - {', '.join(coverage_issue)}")

                except (KeyError, IndexError):
                    logging.warning(f"Could not determine wavelength range for index {index_name}")

        logging.info("=== END WAVELENGTH COVERAGE ANALYSIS ===")

        # Run LS Measurements
        start_time = time.time()

        ###### IF RUNNING IN Parallel
        if nthreads > 1:
            ExGalutil.prettyOutput_Running("Running LINE_STRENGTH in parallel mode")
            logging.info("Running LINE_STRENGTH in parallel mode")

            # Create Queues
            inQueue  = Queue()
            outQueue = Queue()

            # Create worker processes
            ps = [Process(target=workerLS, args=(inQueue, outQueue))
                  for _ in range(nthreads)]

            # Start worker processes
            for p in ps: p.start()

            # Fill the queue
            for i in range(nbins):
                inQueue.put( ( wave, spec[i,:], espec[i,:], redshift[i,:], configs, lickfile, names, index_names,\
                               model_indices, params, tri, labels, outdir, nbins, i, MCMC ) )

            # now get the results with indices
            ls_tmp = [outQueue.get() for _ in range(nbins)]

            # send stop signal to stop iteration
            for _ in range(nthreads): inQueue.put('STOP')

            # stop processes
            for p in ps: p.join()

            # IMPROVED: Get output with complex value handling
            index = np.zeros(nbins)
            for i in range(0, nbins):
                index[i] = ls_tmp[i][0]

                # Handle complex values in parallel results
                ls_indices_tmp = ls_tmp[i][1]
                ls_errors_tmp = ls_tmp[i][2]

                # Ensure all values are real
                if np.iscomplexobj(ls_indices_tmp):
                    ls_indices_tmp = np.real(ls_indices_tmp)
                if np.iscomplexobj(ls_errors_tmp):
                    ls_errors_tmp = np.real(ls_errors_tmp)

                ls_indices[i,:] = ls_indices_tmp
                ls_errors[i,:]  = ls_errors_tmp

                if MCMC == True:
                    vals[i,:]         = ls_tmp[i][3]
                    percentile[i,:,:] = ls_tmp[i][4]

            # Sort output
            argidx = np.argsort( index )
            ls_indices = ls_indices[argidx,:]
            ls_errors  = ls_errors[argidx,:]
            if MCMC == True:
                vals       = vals[argidx,:]
                percentile = percentile[argidx,:,:]

            ExGalutil.prettyOutput_Done("Running LINE_STRENGTH in parallel mode", progressbar=True)

        ###### IF RUNNING IN SERIAL
        if nthreads < 2:
            ExGalutil.prettyOutput_Running("Running LINE_STRENGTH in serial mode")
            logging.info("Running LINE_STRENGTH in serial mode")

            if MCMC == True:
                for i in range(nbins):
                    ls_indices[i,:], ls_errors[i,:], vals[i,:], percentile[i,:,:] = run_ls\
                            (wave, spec[i,:], espec[i,:], redshift[i,:], configs, lickfile, names, index_names,\
                            model_indices, params, tri, labels, outdir, nbins, i, MCMC)
            elif MCMC == False:
                for i in range(nbins):
                    ls_indices[i,:], ls_errors[i,:] = run_ls\
                            (wave, spec[i,:], espec[i,:], redshift[i,:], configs, lickfile, names, index_names,\
                            model_indices, params, tri, labels, outdir, nbins, i, MCMC)

            ExGalutil.prettyOutput_Done("Running LINE_STRENGTH in serial mode", progressbar=True)

        print("             Running LINE_STRENGTH on %s spectra took %.2fs using %i cores" % (nbins, time.time() - start_time, nthreads))
        logging.info("Running LINE_STRENGTH on %s spectra took %.2fs using %i cores" % (nbins, time.time() - start_time, nthreads))

        # IMPROVED: Enhanced error reporting and diagnostics
        print("=== IFU ANALYSIS RESULTS SUMMARY ===")

        # Check for exceptions which occurred during the analysis
        idx_error = np.where( np.all( np.isnan(ls_indices[:,:]), axis=1 ) == True )[0]
        if len(idx_error) != 0:
            ExGalutil.prettyOutput_Warning("There was a problem in the analysis of the spectra with the following BINID's: ")
            print("             "+str(idx_error))
            logging.warning("There was a problem in the analysis of the spectra with the following BINID's: "+str(idx_error))
        else:
            print("             "+"There were no problems in the analysis.")
            ExGalutil.prettyOutput_Info("There were no problems in the analysis.")

        if debug:
            # ADDED: Resolution flag statistics
            flag_counts = np.bincount(totalFWHM_flag, minlength=5)
            print(f"Resolution flags: Normal={flag_counts[0]}, Resolution issues={flag_counts[1]}, "
                f"Extreme broadening={flag_counts[2]}, Convolution failed={flag_counts[3]}, Complete failure={flag_counts[4]}")
            logging.info(f"FWHM flags distribution: {flag_counts}")

            # ADDED: Index measurement statistics
            valid_measurements = np.sum(~np.isnan(ls_indices), axis=0)
            print("Valid measurements per index:")
            for i, name in enumerate(names):
                print(f"  {name}: {valid_measurements[i]}/{nbins} ({100*valid_measurements[i]/nbins:.1f}%)")

            # ADDED: Velocity dispersion statistics for problematic cases
            high_sigma_bins = np.where(veldisp_kin > 300)[0]
            if len(high_sigma_bins) > 0:
                print(f"High velocity dispersion bins (>300 km/s): {len(high_sigma_bins)}")
                high_sigma_flags = totalFWHM_flag[high_sigma_bins]
                print(f"  Flag distribution: {np.bincount(high_sigma_flags, minlength=5)}")

        print("")

        # Save Results
        try:
            if MCMC == True:
                save_ls(names,configs, ls_indices, ls_errors, index_names, labels, RESOLUTION, MCMC, totalFWHM_flag, outdir, rootname, ubins, config_dir, vals=vals, percentile=percentile)
            elif MCMC == False:
                save_ls(names,configs, ls_indices, ls_errors, index_names, labels, RESOLUTION, MCMC, totalFWHM_flag, outdir, rootname, ubins, config_dir)
        except Exception as e:
            ExGalutil.prettyOutput_Warning(f"Failed to save results: {str(e)}")
            logging.error(f"Save failed: {str(e)}")

        # Do Plots
        try:
            ExGalutil.prettyOutput_Running("Producing line strength maps")
            logging.info("Producing line strength maps")
            util_plot_ls.plot_maps(outdir,rootname, RESOLUTION)
            ExGalutil.prettyOutput_Done("Producing line strength maps")
        except Exception as plot_error:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            print("".join(lines))
            sys.stdout.flush()
            ExGalutil.prettyOutput_Failed("Producing line strength maps")
            logging.warning(f"Failed to produce line strength maps: {str(plot_error)}. Analysis continues!")

        # ADDED: Final summary statistics
        logging.info("=== IFU FINAL SUMMARY ===")
        logging.info(f"Successfully processed: {nbins - len(idx_error)}/{nbins} spectra")
        logging.info(f"Average measurements per spectrum: {np.nanmean(np.sum(~np.isnan(ls_indices), axis=1)):.1f}/{len(names)}")
        if MCMC and 'vals' in locals():
            mcmc_success = np.sum(~np.isnan(vals[:,-2]))  # Count non-NaN likelihood values
            logging.info(f"MCMC successful: {mcmc_success}/{nbins} spectra")

        if debug:
            print("\033[0;37m"+" - - - - - LINE STRENGTHS (IFU) done - - - - -"+"\033[0;39m")
            print("")
            logging.info(" - - - LINE STRENGTHS (IFU) Done - - - \n")

        # return Results (for debugging purpose)
        if MCMC == True:
            if debug:
                return ls_indices, ls_errors, index_names, vals, percentile, totalFWHM_flag
            else:
                return
        elif MCMC == False:
            if debug:
                return ls_indices, ls_errors, np.nan, np.nan, np.nan, np.nan
            else:
                return

    elif SKIP == True:
        print("")
        print(ExGalutil.prettyOutput_WarningPrefix()+"Skipping LINE STRENGTHS!")
        print("")
        logging.warning("Skipping LINE STRENGTHS\n")
