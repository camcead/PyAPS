from   astropy.io          import fits, ascii
import numpy               as np

import os
import traceback
import warnings
import sys
os.environ['OMP_NUM_THREADS'] = '1'
import logging
import glob
from   scipy.interpolate import interp1d
import astropy.io.fits as pyfits
from multiprocessing import Queue, Process
import PyAPS
from PyAPS import ExGalutil
from PyAPS import ExGalPrepare
from PyAPS.aps_utils import APSOB, makeR, print_args, none_or_str, str2bool, aps_ids_class, apply_redshift_to_fwhm_corrected
from PyAPS import MOSExGalPrepare as MOSExGalPrepare

from PyAPS import lsindex_spec   as lsindex
from PyAPS import ssppop_fitting as ssppop
from PyAPS import aps_constants
from astropy import units
APSVERS = PyAPS.__version__

from ppxf.ppxf_util import gaussian_filter1d

# PHYSICAL CONSTANTS
Clight = 299792.458  # km/s

############################################################################################################################

def workerLS(inQueue, outQueue):
    """
    Defines the worker process of the parallelisation with multiprocessing.Queue
    and multiprocessing.Process.
    """
    for outdir,config_dir, templates_dir, configs,info_tab, veldisp_kin, redshift, wave, spec, espec, velscale, MCMC,\
    RESOLUTION, lickfile, names, index_names, model_indices, params,\
    tri, labels, nbins , i in iter(inQueue.get, 'STOP'):

        ## make sure outputs of the workerLS_serial are in the following order
        ## i, ls_indices, ls_errors, vals, percentile, ls_meta, totalFWHM_flag

        outQueue.put(workerLS_serial(outdir, config_dir, templates_dir,configs,info_tab, veldisp_kin, redshift, wave, spec,\
        espec, velscale, MCMC, RESOLUTION, lickfile, names, index_names, model_indices, params,\
        tri, labels, nbins , i))

##################################################################################################################
def prepareLS(outdir, config_dir, templates_dir,  configs, info_tab, veldisp_kin, wave, spec, espec, velscale, MCMC, RESOLUTION , index):
    """
    Prepare spectra for line strength analysis by broadening to target resolution.

    FIXED: Now properly includes velocity dispersion in total resolution calculation
    ADDED: Robust error handling for negative FWHM differences
    ADDED: Detailed logging for resolution degradation issues
    """
    ## Here we assumed bin_ID in the _table.fits table and the index in this loop are consistent!
    ## Must be checked!!!
    ls_meta = {}
    binid = np.ravel(np.where(info_tab['BIN_ID'] == index))[0]

    ls_meta['BIN_ID'] = info_tab['BIN_ID'][binid]
    ls_meta['APS_ID'] = info_tab['APS_ID'][binid]
    ls_meta['TARGID'] = info_tab['TARGID'][binid]
    ls_meta['CNAME'] = info_tab['CNAME'][binid]
    ls_meta['Z'] = info_tab['Z'][binid]
    ls_meta['ZERR'] = info_tab['ZERR'][binid]

    # predefine totalFWHM_flag variable
    totalFWHM_flag = 0

    # Broaden spectra to LIS resolution taking into account BOTH instrumental LSF AND velocity dispersion
    # FIXED: Uncommented and corrected velocity dispersion contribution
    if RESOLUTION == "ADAPTED":
        ExGalutil.prettyOutput_Running("Broadening the spectra to LIS resolution")

        # # Read LSF of observation and construct an interpolation function
        # LSF           = np.genfromtxt(config_dir+'LSF-Config_'+configs['SETMODE'], comments='#')
        # LSF[:,0]      = LSF[:,0] / (1 + ls_meta['Z'])  # Correct for redshift
        # LSF[:,1]      = LSF[:,1] / (1 + ls_meta['Z'])  # Correct for redshift
        # LSF_Data      = interp1d(LSF[:,0], LSF[:,1], 'linear', fill_value = 'extrapolate')


        # new method with reading LSF from the L1 calibrations
        lsf_data = MOSExGalPrepare.load_lsf_data(configs)
        lsf_indx = np.where(lsf_data['aps_id'] == ls_meta['APS_ID'])[0]
        lsf_func = lsf_data['lsf'][lsf_indx][0]
        LSF_Data = apply_redshift_to_fwhm_corrected(lsf_func, ls_meta['Z'])

        # FIXED: Convert velocity dispersion of galaxy (from PPXF) to Angstrom
        # This was incorrectly commented out - velocity dispersion MUST be included
        veldisp_kin_Angst = veldisp_kin * wave / Clight * 2.355

        # FIXED: Total dispersion for this bin includes BOTH instrumental and velocity dispersion
        # This is the physically correct approach for line strength measurements
        total_dispersion = np.sqrt( LSF_Data(wave)**2 + veldisp_kin_Angst**2 )

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
                f"BIN_ID {ls_meta['BIN_ID']}: Total resolution ({max_total_disp:.3f} Å) > target ({target_resolution:.1f} Å) "
                f"at {len(negative_idx)} wavelengths ({min_wave:.0f}-{max_wave:.0f} Å)"
            )
            ExGalutil.prettyOutput_Warning(
                f"  Velocity dispersion: {veldisp_kin:.1f} km/s, "
                f"Max veldisp contribution: {veldisp_kin_Angst[negative_idx].max():.3f} Å"
            )

            # Set flag to indicate this spectrum has resolution issues
            totalFWHM_flag = 1

            # ADDED: Options for handling negative cases
            if configs.get('LS_RESOLUTION_MODE', 'SKIP_NEGATIVE') == 'SKIP_NEGATIVE':
                # Set problematic wavelengths to zero broadening (no additional convolution)
                FWHM_dif_squared[negative_idx] = 0.0
                ExGalutil.prettyOutput_Warning("  -> Setting zero additional broadening for problematic wavelengths")

            elif configs.get('LS_RESOLUTION_MODE', 'SKIP_NEGATIVE') == 'ADAPTIVE_TARGET':
                # Adaptively increase target resolution to accommodate this spectrum
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
                f"BIN_ID {ls_meta['BIN_ID']}: Extreme broadening required (max σ = {max_sigma:.1f} pixels)"
            )
            if max_sigma > 100:  # Very extreme case
                totalFWHM_flag = 2  # Flag for extreme broadening

        # ADDED: Handle any remaining NaN values (should be rare after above fixes)
        nan_idx = np.where(np.isnan(sigma))[0]
        if len(nan_idx) > 0:
            sigma[nan_idx] = 0.0
            totalFWHM_flag = max(totalFWHM_flag, 1)
            ExGalutil.prettyOutput_Warning(f"BIN_ID {ls_meta['BIN_ID']}: Found {len(nan_idx)} NaN sigma values, set to zero")

        # Convolve spectra pixel-wise with improved error handling
        try:
            # IMPROVED: Better convolution with multiple fallback strategies
            max_sigma_value = sigma.max()

            if max_sigma_value > 0.1:  # Only convolve if significant broadening needed
                # ADDED: Adaptive truncate parameter
                if max_sigma_value < 10:
                    truncate = 4.0  # Standard truncation
                elif max_sigma_value < 50:
                    truncate = 3.0  # Reduce for large kernels
                else:
                    truncate = 2.0  # Minimal for very large kernels

                spec_orig = spec.copy()  # Keep original for fallback
                espec_orig = espec.copy()

                spec = gaussian_filter1d(spec,  sigma, truncate=truncate, mode='nearest')
                espec = gaussian_filter1d(espec, sigma, truncate=truncate, mode='nearest')

                # ADDED: Validate convolution results
                if np.any(np.isnan(spec)) or np.any(np.isinf(spec)):
                    ExGalutil.prettyOutput_Warning(f"BIN_ID {ls_meta['BIN_ID']}: Convolution produced invalid values, using original")
                    spec = spec_orig
                    espec = espec_orig
                    totalFWHM_flag = 3

            else:
                ExGalutil.prettyOutput_Warning(f"BIN_ID {ls_meta['BIN_ID']}: Minimal broadening needed (max σ = {max_sigma_value:.3f}), skipping convolution")

        except Exception as e:
            ExGalutil.prettyOutput_Warning(f"BIN_ID {ls_meta['BIN_ID']}: Convolution failed: {str(e)}")
            totalFWHM_flag = 3  # Flag for convolution failure
            # Return original spectra as fallback - don't modify spec/espec

        ExGalutil.prettyOutput_Done("Broadening the spectra to LIS resolution", progressbar=True)

    return ls_meta, wave, spec, espec, totalFWHM_flag

##################################################################################################################

def workerLS_serial(outdir, config_dir,templates_dir, configs, info_tab,veldisp_kin, redshift, wave, spec, espec, velscale, MCMC, RESOLUTION,\
 lickfile, names, index_names, model_indices, params, tri, labels, nbins , i):
    """Serial worker with improved error handling"""
    try:
        ls_meta, wave, spec, espec, totalFWHM_flag  = prepareLS(outdir, config_dir, templates_dir, configs, info_tab,veldisp_kin, wave, spec, \
            espec, velscale, MCMC, RESOLUTION , i)

        ls_indices, ls_errors, vals, percentile = run_ls(wave, spec, espec, redshift, configs, lickfile,\
            names, index_names, model_indices, params, tri, labels, outdir, nbins, i, MCMC)

        return i, ls_indices, ls_errors, vals, percentile, ls_meta, totalFWHM_flag

    except Exception as e:
        ExGalutil.prettyOutput_Warning(f"Worker failed for bin {i}: {str(e)}")
        # Return NaN results for failed analysis
        n_indices = len(names) if 'names' in locals() else 10  # Fallback
        n_labels = len(labels) if 'labels' in locals() else 5   # Fallback

        # FIXED: Get actual metadata from info_tab instead of hardcoding FAILED
        try:
            binid = np.ravel(np.where(info_tab['BIN_ID'] == i))[0]
            failed_meta = {
                'BIN_ID': info_tab['BIN_ID'][binid],
                'APS_ID': info_tab['APS_ID'][binid],
                'TARGID': info_tab['TARGID'][binid],
                'CNAME': info_tab['CNAME'][binid],
                'Z': info_tab['Z'][binid] if 'Z' in info_tab.columns else 0.0,
                'ZERR': info_tab['ZERR'][binid] if 'ZERR' in info_tab.columns else 0.0
            }
        except Exception as meta_error:
            # If we can't get metadata, use minimal placeholder
            ExGalutil.prettyOutput_Warning(f"Could not retrieve metadata for bin {i}: {str(meta_error)}")
            failed_meta = {
                'BIN_ID': i,
                'APS_ID': -1,
                'TARGID': 'UNKNOWN',
                'CNAME': 'UNKNOWN',
                'Z': 0.0,
                'ZERR': 0.0
            }

        return (i, np.full(n_indices, np.nan), np.full(n_indices, np.nan),
                np.full(n_labels*3+2, np.nan), np.full((101, n_labels), np.nan),
                failed_meta,
                4)  # Flag 4 for complete failure

##################################################################################################################
# def save_ls(metalist,configs, names, ls_indices, ls_errors, index_names, labels, RESOLUTION, MCMC, totalFWHM_flag, outdir, rootname, ubins, vals=None, percentile=None ):
#     """
#     Saves all results to disk with improved FWHM flag documentation.

#     ADDED: Better documentation of FWHM flag meanings
#     """
#     # Save results
#     outfits = outdir+rootname+'_ls_'+RESOLUTION+'.fits'
#     ExGalutil.prettyOutput_Running("Writing: "+str(outfits))

#     # ========================

#     #metalist is the basket to transfer basic info and parameters from the original files
#     APS_ID = [item['APS_ID'] for item in metalist]
#     TARGID = [item['TARGID'] for item in metalist]
#     CNAME = [item['CNAME'] for item in metalist]
#     Z = [item['Z'] for item in metalist]
#     ZERR = [item['ZERR'] for item in metalist]
#     BINID_meta = [item['BIN_ID'] for item in metalist]

#     # Just to make sure data in our basket (metalist) are consistent with the PPXF output
#     assert BINID_meta == list(ubins), 'input and output BIN_ID are not identical'

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
#     cols.append( fits.Column(name='BIN_ID',                               format='J', array=ubins             ))
#     cols.append( fits.Column(name='APS_ID',                               format='J', array=APS_ID            ))
#     cols.append( fits.Column(name='TARGID',                               format='40A', array=TARGID          ))
#     cols.append( fits.Column(name='CNAME',                                format='40A', array=CNAME           ))
#     ndim  = len(names)
#     for i in range(ndim):
#         cols.append( fits.Column(name=names[i],        unit=flux_unit_str, format='D', array=ls_indices[:,i]  ))
#         cols.append( fits.Column(name="ERR_"+names[i], unit=flux_unit_str, format='D', array=ls_errors[:,i]   ))
#     cols.append( fits.Column(name="FWHM_FLAG",                             format='I', array=totalFWHM_flag[:]))
#     lsHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
#     lsHDU.name = "LS_TABLE"

#     # IMPROVED: Better header documentation
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
#     lsHDU.header['APSLSV'] = (aps_constants.__aps_ls_version__,'PyAPS (MOS) LS wrapper version')
#     lsHDU.header['CSB_LS'] = (configs['stitched'], 'Combines Spectral Bands Status for LS')

#     # ADDED: Document velocity dispersion handling
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
#         cols.append( fits.Column(name='BIN_ID',        format='J',   array=ubins          ))
#         cols.append( fits.Column(name='APS_ID',        format='J',   array=APS_ID         ))
#         cols.append( fits.Column(name='TARGID',        format='40A', array=TARGID         ))
#         cols.append( fits.Column(name='CNAME',         format='40A', array=CNAME          ))

#         for i in range(nparam):
#             cols.append( fits.Column(name=labels[i],           format='101D', array=percentile[:,:,i] ))
#         cols.append( fits.Column(    name='lnP',               format='D',    array=vals[:,-2]        ))
#         cols.append( fits.Column(    name='Flag',       format='D',    array=vals[:,-1]        ))
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
#         sspHDU.header['APSLSV'] = (aps_constants.__aps_ls_version__,'PyAPS (MOS) LS wrapper version')
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

def save_ls(metalist, configs, names, ls_indices, ls_errors, index_names, labels,
            RESOLUTION, MCMC, totalFWHM_flag, outdir, rootname, ubins,
            config_dir, vals=None, percentile=None):
    """
    Saves all results to disk with improved FWHM flag documentation and CORRECT units.

    FIXED: Units now properly assigned based on index type:
           - D4000: dimensionless (flux ratio)
           - b7=1: Angstrom (equivalent width)
           - b7=2: mag (magnitude indices)

    Args:
        metalist: List of metadata dictionaries for each bin
        configs: Configuration dictionary
        names: List of all index names
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
        config_dir: Configuration directory path
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

    # Save results
    outfits = outdir + rootname + '_ls_' + RESOLUTION + '.fits'
    ExGalutil.prettyOutput_Running("Writing: " + str(outfits))

    # ========================
    # Extract metadata from metalist
    # metalist is the basket to transfer basic info and parameters from the original files
    APS_ID = [item['APS_ID'] for item in metalist]
    TARGID = [item['TARGID'] for item in metalist]
    CNAME = [item['CNAME'] for item in metalist]
    Z = [item['Z'] for item in metalist]
    ZERR = [item['ZERR'] for item in metalist]
    BINID_meta = [item['BIN_ID'] for item in metalist]

    # Just to make sure data in our basket (metalist) are consistent with the PPXF output
    assert BINID_meta == list(ubins), 'input and output BIN_ID are not identical'

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # ========================
    # Extension 1: Table HDU with LS output data
    # ========================
    cols = []
    cols.append(fits.Column(name='BIN_ID', format='J', array=ubins))
    cols.append(fits.Column(name='APS_ID', format='J', array=APS_ID))
    cols.append(fits.Column(name='TARGID', format='40A', array=TARGID))
    cols.append(fits.Column(name='CNAME', format='40A', array=CNAME))

    ndim = len(names)
    for i in range(ndim):
        # FIXED: Get correct unit for each index based on Lick file
        index_unit = get_index_unit(names[i])

        # Add index value column with proper unit
        cols.append(fits.Column(name=names[i],
                                unit=index_unit,
                                format='D',
                                array=ls_indices[:,i]))

        # Add index error column with same unit
        cols.append(fits.Column(name="ERR_"+names[i],
                                unit=index_unit,
                                format='D',
                                array=ls_errors[:,i]))

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
    lsHDU.header['APSLSV'] = (aps_constants.__aps_ls_version__, 'PyAPS (MOS) LS wrapper version')
    lsHDU.header['CSB_LS'] = (configs['stitched'], 'Combines Spectral Bands Status for LS')

    # Document velocity dispersion handling
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
        cols.append(fits.Column(name='APS_ID', format='J', array=APS_ID))
        cols.append(fits.Column(name='TARGID', format='40A', array=TARGID))
        cols.append(fits.Column(name='CNAME', format='40A', array=CNAME))

        for i in range(nparam):
            cols.append(fits.Column(name=labels[i], format='101D', array=percentile[:,:,i]))

        cols.append(fits.Column(name='lnP', format='D', array=vals[:,-2]))
        cols.append(fits.Column(name='Flag', format='D', array=vals[:,-1]))

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
        sspHDU.header['APSLSV'] = (aps_constants.__aps_ls_version__, 'PyAPS (MOS) LS wrapper version')
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



##################################################################################################################

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
    cols.append( fits.Column(name='LAM', format=str(npix)+'D', array=wave) )
    logLamHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    logLamHDU.name = 'LAM'

    # Create HDU list and write to file
    HDUList = fits.HDUList([priHDU, dataHDU, logLamHDU])
    HDUList.writeto(outfits, overwrite=True)

    ExGalutil.prettyOutput_Done("Writing: "+rootname+'_ls-cleaned_linear.fits')
    logging.info("Wrote: "+outfits)

##################################################################################################################

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

##################################################################################################################
def run_ls(wave, spec, espec, redshift, configs, lickfile, names, index_names,\
           model_indices, params, tri, labels, outdir, nbins, i, MCMC):
    """
    Calls a Python version of the line strength measurement routine of
    Kuntschner et al. 2006 (ui.adsabs.harvard.edu/?#abs/2006MNRAS.369..497K),
    and if required, the MCMC algorithm from Martin-Navaroo et al. 2018
    (ui.adsabs.harvard.edu/#abs/2018MNRAS.475.3700M) to determine SSP
    properties.

    IMPROVED: Better error handling and validation
    """
    ExGalutil.printProgress(i, nbins, barLength = 50)
    nindex = len(index_names)

    try:
        # ADDED: Input validation
        if np.any(np.isnan(spec)) or np.any(np.isnan(wave)):
            ExGalutil.prettyOutput_Warning(f"Bin {i}: NaN values in input spectrum")
            return( np.full(len(names), np.nan), np.full(len(names), np.nan), np.nan, np.nan )

        # IMPROVED: Smart validation - handle partial NaN data intelligently
        total_pixels = len(spec)
        nan_pixels = np.sum(np.isnan(spec))
        nan_wave_pixels = np.sum(np.isnan(wave))

        # Check for completely invalid data
        if nan_pixels == total_pixels or nan_wave_pixels == len(wave):
            ExGalutil.prettyOutput_Warning(f"Bin {i}: Completely invalid spectrum (all NaN)")
            return( np.full(len(names), np.nan), np.full(len(names), np.nan), np.nan, np.nan )

        # Handle partial NaN data (common for different redshift bins)
        if nan_pixels > 0:
            frac_nan = nan_pixels / total_pixels * 100
            ExGalutil.prettyOutput_Warning(f"Bin {i}: {nan_pixels} NaN pixels ({frac_nan:.1f}% of spectrum) - proceeding with valid data")

            # Only skip if >80% of spectrum is NaN (very conservative threshold)
            if frac_nan > 80.0:
                ExGalutil.prettyOutput_Warning(f"Bin {i}: Too much missing data ({frac_nan:.1f}% NaN), skipping")
                return( np.full(len(names), np.nan), np.full(len(names), np.nan), np.nan, np.nan )

            # For partial NaN: Clean the data but continue processing
            # Option 1: Interpolate small gaps (for small NaN regions)
            if frac_nan < 5.0:  # Small gaps - interpolate
                valid_idx = ~np.isnan(spec)
                if np.sum(valid_idx) > 10:  # Need enough points for interpolation
                    from scipy.interpolate import interp1d
                    try:
                        interp_func = interp1d(wave[valid_idx], spec[valid_idx],
                                             kind='linear', fill_value='extrapolate',
                                             bounds_error=False)
                        spec_clean = interp_func(wave)

                        # Replace NaN regions with interpolated values
                        spec[np.isnan(spec)] = spec_clean[np.isnan(spec)]
                        ExGalutil.prettyOutput_Warning(f"Bin {i}: Interpolated {nan_pixels} NaN pixels")
                    except:
                        ExGalutil.prettyOutput_Warning(f"Bin {i}: Interpolation failed, proceeding with NaN pixels")

            # Option 2: For larger gaps, let the Lick algorithm handle them naturally
            # (it will return NaN for indices that can't be measured)

        # IMPROVED: Handle error spectrum NaN values similarly
        nan_error_pixels = np.sum(np.isnan(espec))
        if nan_error_pixels > 0:
            if nan_error_pixels == total_pixels:
                ExGalutil.prettyOutput_Warning(f"Bin {i}: All error values are NaN - using flux-based errors")
                # Use simple Poisson errors as fallback
                espec = np.sqrt(np.abs(spec))
            else:
                # For partial NaN in errors, interpolate or use local estimates
                valid_err_idx = ~np.isnan(espec)
                if np.sum(valid_err_idx) > 10:
                    try:
                        median_err = np.nanmedian(espec)
                        espec[np.isnan(espec)] = median_err
                    except:
                        espec[np.isnan(espec)] = np.sqrt(np.abs(spec[np.isnan(espec)]))

        # IMPROVED: Better handling of non-positive flux values
        negative_flux = np.where(spec <= 0)[0]
        if len(negative_flux) > 0:
            frac_negative = len(negative_flux) / len(spec) * 100

            # Only intervene for severe cases (>50% negative)
            if frac_negative > 50.0:
                ExGalutil.prettyOutput_Warning(f"Bin {i}: Severe data quality issue - {frac_negative:.1f}% negative flux")
                return( np.full(len(names), np.nan), np.full(len(names), np.nan), np.nan, np.nan )

            # Just log for moderate cases, don't modify data
            elif frac_negative > 10.0:
                ExGalutil.prettyOutput_Warning(f"Bin {i}: {frac_negative:.1f}% negative flux values (proceeding)")

            # Do NOT clip or modify the spectrum - let Lick algorithm handle it

        # IMPROVED: Robust line strength measurement with complex value handling
        try:
            # Measure the LS indices with improved error handling
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
            # Return NaN for all indices if core measurement fails
            return( np.full(len(names), np.nan), np.full(len(names), np.nan), np.nan, np.nan )

        # IMPROVED: Get the indices with better error handling for missing indices
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
                    ExGalutil.prettyOutput_Warning(f"Bin {i}: Complex value for index {index_names[o]}, taking real part")
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
                # Index name not found in results (shouldn't happen with proper config)
                ExGalutil.prettyOutput_Warning(f"Bin {i}: Index {index_names[o]} not found in measurement results")
                data[o] = np.nan
                error[o] = np.nan

        # IMPROVED: Count how many indices were successfully measured
        valid_indices = np.sum(~np.isnan(data))
        total_indices = len(data)

        if valid_indices == 0:
            ExGalutil.prettyOutput_Warning(f"Bin {i}: No valid index measurements obtained")
            if MCMC:
                return(np.full(len(names_out), np.nan), np.full(len(names_out), np.nan),
                       np.full(len(labels)*3+2, np.nan), np.full((101, len(labels)), np.nan))
            else:
                return(np.full(len(names_out), np.nan), np.full(len(names_out), np.nan), np.nan, np.nan)

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
                ExGalutil.prettyOutput_Warning(f"Bin {i}: Invalid error estimates for MCMC, using default errors")
                # Use 10% errors as fallback for indices with invalid error estimates
                error[np.isnan(error) | (error <= 0)] = 0.1 * np.abs(data[np.isnan(error) | (error <= 0)])

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
            return(indices, errors, np.nan, np.nan)

    except Exception as e:
        ExGalutil.prettyOutput_Warning(f"Bin {i}: Line strength measurement failed: {str(e)}")
        return( np.full(len(names), np.nan), np.full(len(names), np.nan), np.nan, np.nan )

##################################################################################################################

def runModule_LINESTRENGTH(LINE_STRENGTH, RESOLUTION, nthreads, configs, velscale, outdir, config_dir, templates_dir, rootname, debug=False):
    """
    Main line strength analysis module with improved error handling and validation.

    FIXED: Velocity dispersion now properly included in resolution calculation
    ADDED: Comprehensive validation and error reporting
    ADDED: Support for different resolution handling modes
    """

    # Run MCMC only on the indices measured from convoluted spectra
    if LINE_STRENGTH == 2  and  RESOLUTION == "ADAPTED":
        MCMC = True
    else:
        MCMC = False

    ## READ INFO TABLE
    hdu_t = fits.open(outdir+rootname+'_table.fits')
    info_tab = hdu_t[1].data

    # ADDED: Validate target resolution before processing
    target_resolution = configs.get('CONV_COR', 8.4)
    logging.info(f"Target resolution (CONV_COR): {target_resolution:.1f} Å FWHM")

    # ADDED: Set default resolution handling mode if not specified
    if 'LS_RESOLUTION_MODE' not in configs:
        configs['LS_RESOLUTION_MODE'] = 'SKIP_NEGATIVE'
        logging.info("Using default resolution mode: SKIP_NEGATIVE")

    # Read the log-rebinned, cleaned spectra from EMIPPXF and log-unbin them
    # if os.path.isfile(outdir+rootname+'_ls-cleaned_linear.fits') == False:

    # Read cleaned spectra
    logging.info("Reading "+outdir+rootname+"_emippxf_spec_BIN.fits")
    hdu_spec  = fits.open(outdir+rootname+'_emippxf_spec_BIN.fits')

    oldspec  = np.array( hdu_spec[1].data.FLUX_CLEAN_EMI)
    oldespec = np.sqrt( hdu_spec[1].data.ERROR_EMI)
    logLAM     = np.array( hdu_spec[1].data.LOGLAM_EMI)

    nbins    = oldspec.shape[0]
    ubins = np.arange(0, nbins)
    npix     = oldspec.shape[1]
    spec     = np.zeros( oldspec.shape  )
    espec    = np.zeros( oldespec.shape )
    wave     = np.zeros( logLAM.shape )

    # Rebin the cleaned spectra from log to lin
    ExGalutil.prettyOutput_Running("Rebinning the cleaned spectra from log to lin")
    for i in range( nbins ):
        ExGalutil.printProgress(i, nbins, barLength = 50)
        lamRange = np.array([ logLAM[i,:][0], logLAM[i,:][-1]])
        spec[i,:], wave[i,:] = log_unbinning( lamRange, oldspec[i,:] )
    ExGalutil.prettyOutput_Done("Rebinning the cleaned spectra from log to lin", progressbar=True)

    # Rebin the error spectra from log to lin
    ExGalutil.prettyOutput_Running("Rebinning the error spectra from log to lin")
    for i in range( nbins ):
        ExGalutil.printProgress(i, nbins, barLength = 50)
        lamRange = np.array([ logLAM[i,:][0], logLAM[i,:][-1]])
        espec[i,:], _ = log_unbinning( lamRange, oldespec[i,:] )
    ExGalutil.prettyOutput_Done("Rebinning the error spectra from log to lin", progressbar=True)

    # Save cleaned, linear spectra
    saveCleanedLinearSpectra(spec, espec, wave, npix, outdir, rootname)

    # Read the linearly-binned, cleaned spectra provided by previous LS-run
    # else:
    logging.info("Reading "+outdir+rootname+'_ls-cleaned_linear.fits')
    hdu   = fits.open(outdir+rootname+'_ls-cleaned_linear.fits')
    spec  = np.array( hdu[1].data.SPEC  )
    espec = np.array( hdu[1].data.ESPEC )
    wave  = np.array( hdu[2].data.LAM   )
    nbins = spec.shape[0]
    ubins = np.arange(0, nbins)

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

    # ADDED: Wavelength coverage diagnostics
    logging.info("=== WAVELENGTH COVERAGE ANALYSIS ===")
    wave_min, wave_max = wave.min(), wave.max()
    logging.info(f"Spectral coverage: {wave_min:.1f} - {wave_max:.1f} Å")

    lickfile = config_dir+configs['LS_FILE']
    tab   = ascii.read(lickfile, comment='\\s*#')
    names = tab['names']


    # Check which indices are outside wavelength coverage
    lick_tab = ascii.read(lickfile, comment='\\s*#')
    for i, index_name in enumerate(names):
        idx_row = np.where(lick_tab['names'] == index_name)[0]
        if len(idx_row) > 0:
            # Get index wavelength range (assuming columns b1-b6 define the range)
            try:
                index_wave_min = lick_tab['b1'][idx_row[0]]  # Blue continuum start
                index_wave_max = lick_tab['b6'][idx_row[0]]  # Red continuum end

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
    logging.info("")

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
    ls_metalist= [None for _ in range(nbins)]
    ls_indices = np.zeros((nbins, len(names)))
    ls_errors  = np.zeros((nbins, len(names)))

    vals       = np.zeros((nbins, len(labels)*3+2)) if MCMC else np.full((nbins, 5), np.nan)
    percentile = np.zeros((nbins, 101, len(labels))) if MCMC else np.full((nbins, 101, 5), np.nan)

    # Flag spectra for which the total intrinsic dispersion is larger than the LIS measurement resolution
    totalFWHM_flag = np.zeros(spec.shape[0], dtype=int)

    # ADDED: Pre-analysis diagnostics
    logging.info("=== PRE-ANALYSIS DIAGNOSTICS ===")
    logging.info(f"Number of spectra: {nbins}")
    logging.info(f"Spectral pixels: {npix}")
    logging.info(f"Wavelength range: {wave.min():.1f} - {wave.max():.1f} Å")
    logging.info(f"Target resolution: {configs['CONV_COR']:.1f} Å FWHM")
    logging.info(f"Resolution mode: {configs.get('LS_RESOLUTION_MODE', 'SKIP_NEGATIVE')}")
    logging.info(f"Number of line indices: {len(names)}")
    logging.info(f"MCMC enabled: {MCMC}")

    ###### IF RUNNING IN Parallel
    if nthreads > 1:
        ExGalutil.prettyOutput_Running("Running LINE_STRENGTH in parallel mode")
        logging.info("Running LINE_STRENGTH in parallel mode")

        # Create Queues
        inQueue  = Queue()
        outQueue = Queue()

        # Create worker processes
        ps = [Process(target=workerLS, args=(inQueue, outQueue)) for _ in range(nthreads)]

        # Start worker processes
        for p in ps: p.start()

        # Fill the queue
        for i in range(nbins):
            inQueue.put( ( outdir,config_dir,templates_dir, configs, info_tab, veldisp_kin[i], redshift[i,:], wave[i,:], spec[i,:],\
            espec[i,:], velscale[i], MCMC, RESOLUTION, lickfile, names, index_names, model_indices, params,\
            tri, labels, nbins , i ) )

        # now get the results with indices
        ls_tmp = [outQueue.get() for _ in range(nbins)]

        # send stop signal to stop iteration
        for _ in range(nthreads): inQueue.put('STOP')

        # stop processes
        for p in ps: p.join()

        # Get output
        index = np.zeros(nbins)
        for i in range(0, nbins):
            index[i]          = ls_tmp[i][0]
            ls_indices[i,:]   = ls_tmp[i][1]
            ls_errors[i,:]    = ls_tmp[i][2]
            vals[i,:]         = ls_tmp[i][3] if MCMC else np.nan
            percentile[i,:,:] = ls_tmp[i][4] if MCMC else np.nan
            ls_metalist[i]    = ls_tmp[i][5]
            totalFWHM_flag[i] = ls_tmp[i][6]

        # Sort output
        argidx = np.argsort( index )
        print("Processed bins:", index[argidx])
        ls_indices = ls_indices[argidx,:]
        ls_errors  = ls_errors[argidx,:]
        vals       = vals[argidx,:]
        percentile = percentile[argidx,:,:]
        totalFWHM_flag = totalFWHM_flag[argidx]
        ls_metalist    = [ls_metalist[_lms] for _lms in argidx]

        ExGalutil.prettyOutput_Done("Running LINE_STRENGTH in parallel mode", progressbar=True)

    ###### IF RUNNING IN SERIAL
    if nthreads < 2:
        ExGalutil.prettyOutput_Running("Running LINE_STRENGTH in serial mode")
        logging.info("Running LINE_STRENGTH in serial mode")

        for i in range(nbins):
            _, ls_indices[i,:], ls_errors[i,:], vals[i,:], percentile[i,:,:], ls_metalist[i], totalFWHM_flag[i] = \
            workerLS_serial(outdir, config_dir, templates_dir, configs,info_tab,veldisp_kin[i], redshift[i,:],\
            wave[i,:], spec[i,:], espec[i,:], velscale[i], MCMC, RESOLUTION,\
            lickfile, names, index_names, model_indices, params, tri, labels, nbins , i)

        ExGalutil.prettyOutput_Done("Running LINE_STRENGTH in serial mode", progressbar=True)

    #### IMPROVED: Enhanced error reporting and diagnostics
    # print("=== ANALYSIS RESULTS SUMMARY ===")

    # Check for exceptions which occurred during the analysis
    idx_error = np.where( np.all( np.isnan(ls_indices[:,:]), axis=1 ) == True )[0]
    if len(idx_error) != 0:
        aps_id_list_error=[]
        for idxerr in idx_error:
            aps_id_list_error.append(ls_metalist[idxerr]['APS_ID'])

        ExGalutil.prettyOutput_Warning("There was a problem in the analysis of the spectra with the following APS_ID's: ")
        print("             "+str(aps_id_list_error))
        logging.warning("There was a problem in the analysis of the spectra with the following APS_ID's: "+str(aps_id_list_error))
    else:
        ExGalutil.prettyOutput_Info("             "+"There were no problems in the analysis.")
        logging.info("There were no problems in the analysis.")

    # ADDED: Resolution flag statistics
    flag_counts = np.bincount(totalFWHM_flag, minlength=5)
    if debug:
        ExGalutil.prettyOutput_Info(f"Resolution flags: Normal={flag_counts[0]}, Resolution issues={flag_counts[1]}, "
            f"Extreme broadening={flag_counts[2]}, Convolution failed={flag_counts[3]}, Complete failure={flag_counts[4]}")
        logging.info(f"FWHM flags distribution: {flag_counts}")

    # ADDED: Index measurement statistics
    valid_measurements = np.sum(~np.isnan(ls_indices), axis=0)
    if debug:
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

    #### Save Results
    ExGalutil.prettyOutput_Running("Saving LS result into the disk")
    try:
        save_ls(ls_metalist,configs, names, ls_indices, ls_errors, index_names, labels, RESOLUTION, MCMC, totalFWHM_flag, outdir, rootname, ubins, config_dir, vals=vals, percentile=percentile)
        ExGalutil.prettyOutput_Done("Saving LS result into the disk")
    except Exception as e:
        ExGalutil.prettyOutput_Warning(f"Failed to save results: {str(e)}")
        logging.error(f"Save failed: {str(e)}")

    # ADDED: Final summary statistics
    logging.info("=== FINAL SUMMARY ===")
    logging.info(f"Successfully processed: {nbins - len(idx_error)}/{nbins} spectra")
    logging.info(f"Average measurements per spectrum: {np.nanmean(np.sum(~np.isnan(ls_indices), axis=1)):.1f}/{len(names)}")
    if MCMC:
        mcmc_success = np.sum(~np.isnan(vals[:,-2]))  # Count non-NaN likelihood values
        logging.info(f"MCMC successful: {mcmc_success}/{nbins} spectra")

    if debug:
        return ls_indices, ls_errors, index_names,  vals, percentile, ls_metalist, totalFWHM_flag
    else:
        return
