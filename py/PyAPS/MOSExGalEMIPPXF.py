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
import pickle
import PyAPS
from PyAPS import ExGalutil
from PyAPS import MOSExGalPrepare as MOSExGalPrepare
from PyAPS import ExGalPrepare
from PyAPS.aps_utils import APSOB, makeR, print_args, none_or_str, str2bool, aps_ids_class, apply_redshift_to_fwhm_corrected
from PyAPS import aps_constants
from PyAPS.apsPlot.emi import make_emi_kinematics_plots

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
  This module executes the analysis of emi kinematics in the MOS pipeline.
  It processes emission lines using pPXF to extract emi velocities and velocity dispersions,
  as well as emission line fluxes. It builds on the stellar kinematics results.
"""


# make_emi_kinematics_plots now lives in PyAPS.apsPlot.emi, built on the
# shared spectrum_overlay_figure() (Plotly) used across the PyAPS.apsPlot
# platform. Imported at module top as
# `from PyAPS.apsPlot.emi import make_emi_kinematics_plots`.


def save_emi_kinematics_mos(rootname, configs, outdir, metalist, bin_data, bin_error, emi_result, mc_results, formal_error,
                           emi_bestfit, logLam, goodpixels, emission_lines, line_names, npix, ubins,
                           tie_settings, emission_config=None, line_wavelengths=None, group_order=None, debug=False):
    """
    Save emi kinematics results to disk - MOS version.
    UPDATED: Now saves per-line kinematics (V and Sigma for each line) in addition to overall results.
    Creates results for ALL emission lines in config (fitted, masked, out-of-range).
    UPDATED: Only stores kinematics for lines with meaningful flux (flux > 0 and not NaN).
    """

    # Extract metadata from metalist
    APS_ID = [item['APS_ID'] for item in metalist]
    TARGID = [item['TARGID'] for item in metalist]
    CNAME = [item['CNAME'] for item in metalist]
    Z = np.array([item['Z'] for item in metalist])
    ZERR = np.array([item['ZERR'] for item in metalist])
    BINID_meta = [item['BIN_ID'] for item in metalist]

    # Ensure consistency
    assert BINID_meta == list(ubins), 'input and output BIN_ID are not identical'

    # Determine wavelength range used for fitting defined in the config
    if 'LMIN_EMI' in configs and 'LMAX_EMI' in configs:
        wavelength_range_config = (configs['LMIN_EMI'], configs['LMAX_EMI'])
    elif 'LMIN_PPXF' in configs and 'LMAX_PPXF' in configs:
        wavelength_range_config = (configs['LMIN_PPXF'], configs['LMAX_PPXF'])
    else:
        # Fallback to data range
        wavelength_range_config = (np.nan, np.nan)

    # Categorize lines individually for each target (MOS has wide redshift range!)
    if debug:
        print(f"Processing {len(ubins)} targets with individual redshift-dependent line categorization")
        print(f"Redshift range: {np.min(Z):.3f} - {np.max(Z):.3f}")

    # Get ALL lines from emission config
    assert emission_config is not None and 'emission_lines' in emission_config, 'no emission config found to build all line_names'
    all_line_names = list(emission_config['emission_lines'].keys())
    all_line_wavelengths = emission_config['emission_lines']

    # Create per-target line categorization
    target_line_categories = []
    target_fitted_indices = []
    target_line_status = []

    for i, (target_z, bin_id) in enumerate(zip(Z, ubins)):
        # Categorize lines for this specific target's redshift

        lam_i = np.exp(logLam[i])
        wavelength_range_i = (lam_i.min(), lam_i.max())

        # categorize each line from the all_list (full list) based on the wavelenght range for target i and also double check if this line exists in the line_names_filtered for this target
        target_categories = categorize_emission_lines(
            emission_config,
            all_line_names,
            wavelength_range_i,
            wavelength_range_config,
            line_names[i]
        )
        target_line_categories.append(target_categories)
        # Create mapping from fitted lines to all lines for this target
        fitted_indices = []
        line_status = []

        for name in all_line_names:
            if name in target_categories['fitted']:
                fitted_indices.append(line_names[i].index(name))
                line_status.append('FITTED')
            elif name in target_categories['masked']:
                fitted_indices.append(-1)
                line_status.append('MASKED')
            elif name in target_categories['out_of_range']:
                fitted_indices.append(-1)
                line_status.append('OUT_OF_RANGE')
            elif name in target_categories['missing_wavelength']:
                fitted_indices.append(-1)
                line_status.append('NO_WAVELENGTH')
            else:
                fitted_indices.append(-1)
                line_status.append('OTHER')

        target_fitted_indices.append(fitted_indices)
        target_line_status.append(line_status)

    # Print summary statistics
    n_fitted_per_target = [len(cat['fitted']) for cat in target_line_categories]
    n_masked_per_target = [len(cat['masked']) for cat in target_line_categories]
    n_out_range_per_target = [len(cat['out_of_range']) for cat in target_line_categories]

    if debug:
        print(f"  Lines fitted per target: {np.min(n_fitted_per_target)} - {np.max(n_fitted_per_target)} (mean: {np.mean(n_fitted_per_target):.1f})")
        print(f"  Lines out-of-range per target: {np.min(n_out_range_per_target)} - {np.max(n_out_range_per_target)} (mean: {np.mean(n_out_range_per_target):.1f})")
        print(f"  Lines masked per target: {np.min(n_masked_per_target)} - {np.max(n_masked_per_target)} (mean: {np.mean(n_masked_per_target):.1f})")

    # Set flux units for saving
    if configs['sens_corr']:
        flux_unit_str = '%2e erg/(s cm**2 Angstrom)' % (configs['funits'])
    else:
        flux_unit_str = 'count'
    wave_unit_str = 'Angstrom'

    # NEW: Extract per-line kinematics - for ALL lines, per target
    line_velocities_all = np.full((len(ubins), len(all_line_names)), np.nan)
    line_sigmas_all = np.full((len(ubins), len(all_line_names)), np.nan)
    line_velocity_errors_all = np.full((len(ubins), len(all_line_names)), np.nan)
    line_sigma_errors_all = np.full((len(ubins), len(all_line_names)), np.nan)
    line_formal_velocity_errors_all = np.full((len(ubins), len(all_line_names)), np.nan)
    line_formal_sigma_errors_all = np.full((len(ubins), len(all_line_names)), np.nan)

    # NEW: Extract per-line formal flux/amplitude errors
    line_formal_flux_errors_all = np.full((len(ubins), len(all_line_names)), np.nan)
    line_formal_amplitude_errors_all = np.full((len(ubins), len(all_line_names)), np.nan)

    # Extract emission line properties - for ALL lines, per target with individual redshift categorization
    emission_fluxes = np.full((len(ubins), len(all_line_names)), np.nan)
    emission_flux_errors = np.full((len(ubins), len(all_line_names)), np.nan)
    emission_amplitudes = np.full((len(ubins), len(all_line_names)), np.nan)
    emission_amplitude_errors = np.full((len(ubins), len(all_line_names)), np.nan)
    emission_aon = np.full((len(ubins), len(all_line_names)), np.nan)


    for i, bin_id in enumerate(ubins):
            if isinstance(mc_results[i], dict):
                # Use the actual fitted lines (first N lines from line_names that have flux data)
                if isinstance(emission_lines[i], np.ndarray) and len(emission_lines[i]) > 0:

                    # Fill data for fitted lines only (based on this target's redshift)
                    for j, line_name in enumerate(all_line_names):
                        try:
                            fitted_line_position = line_names[i].index(line_name)
                            # Extract fluxes first to check if meaningful
                            line_flux = np.nan
                            if isinstance(emission_lines[i], np.ndarray) and fitted_line_position < len(emission_lines[i]):
                                line_flux = emission_lines[i][fitted_line_position]
                                emission_fluxes[i, j] = line_flux

                            # UPDATED: Only store kinematics if flux is meaningful (> 0 and not NaN)
                            flux_is_meaningful = (not np.isnan(line_flux) and line_flux is not None and line_flux > 1e-10 and line_name in target_line_categories[i]['fitted'])

                            if not flux_is_meaningful:
                                emission_fluxes[i, j] = np.nan

                            # UPDATED: Only extract per-line kinematics if flux is meaningful
                            if flux_is_meaningful:
                                if 'line_velocities' in mc_results[i] and isinstance(mc_results[i]['line_velocities'], np.ndarray):
                                    if fitted_line_position < len(mc_results[i]['line_velocities']):
                                        line_velocities_all[i, j] = mc_results[i]['line_velocities'][fitted_line_position]

                                if 'line_sigmas' in mc_results[i] and isinstance(mc_results[i]['line_sigmas'], np.ndarray):
                                    if fitted_line_position < len(mc_results[i]['line_sigmas']):
                                        line_sigmas_all[i, j] = mc_results[i]['line_sigmas'][fitted_line_position]

                                if 'line_velocity_errors' in mc_results[i] and isinstance(mc_results[i]['line_velocity_errors'], np.ndarray):
                                    if fitted_line_position < len(mc_results[i]['line_velocity_errors']):
                                        line_velocity_errors_all[i, j] = mc_results[i]['line_velocity_errors'][fitted_line_position]

                                if 'line_sigma_errors' in mc_results[i] and isinstance(mc_results[i]['line_sigma_errors'], np.ndarray):
                                    if fitted_line_position < len(mc_results[i]['line_sigma_errors']):
                                        line_sigma_errors_all[i, j] = mc_results[i]['line_sigma_errors'][fitted_line_position]

                                # UPDATED: Only extract per-line formal errors if flux is meaningful
                                if isinstance(formal_error[i], dict):
                                    if 'line_velocities' in formal_error[i] and isinstance(formal_error[i]['line_velocities'], np.ndarray):
                                        if fitted_line_position < len(formal_error[i]['line_velocities']):
                                            line_formal_velocity_errors_all[i, j] = formal_error[i]['line_velocities'][fitted_line_position]

                                    if 'line_sigmas' in formal_error[i] and isinstance(formal_error[i]['line_sigmas'], np.ndarray):
                                        if fitted_line_position < len(formal_error[i]['line_sigmas']):
                                            line_formal_sigma_errors_all[i, j] = formal_error[i]['line_sigmas'][fitted_line_position]

                                # UPDATED: Extract per-line formal flux/amplitude errors regardless of flux value
                                # (these are measurement uncertainties, not kinematics)
                                if isinstance(formal_error[i], dict):
                                    if 'fluxes' in formal_error[i] and isinstance(formal_error[i]['fluxes'], np.ndarray):
                                        if fitted_line_position < len(formal_error[i]['fluxes']):
                                            line_formal_flux_errors_all[i, j] = formal_error[i]['fluxes'][fitted_line_position]

                                    if 'amplitudes' in formal_error[i] and isinstance(formal_error[i]['amplitudes'], np.ndarray):
                                        if fitted_line_position < len(formal_error[i]['amplitudes']):
                                            line_formal_amplitude_errors_all[i, j] = formal_error[i]['amplitudes'][fitted_line_position]

                                # Extract flux errors
                                if 'fluxes' in mc_results[i] and isinstance(mc_results[i]['fluxes'], np.ndarray):
                                    if fitted_line_position < len(mc_results[i]['fluxes']):
                                        emission_flux_errors[i, j] = mc_results[i]['fluxes'][fitted_line_position]

                                # Extract amplitudes
                                if 'amplitudes' in mc_results[i] and isinstance(mc_results[i]['amplitudes'], np.ndarray):
                                    if fitted_line_position < len(mc_results[i]['amplitudes']):
                                        emission_amplitudes[i, j] = mc_results[i]['amplitudes'][fitted_line_position]

                                # Extract amplitude errors
                                if 'amplitude_errors' in mc_results[i] and isinstance(mc_results[i]['amplitude_errors'], np.ndarray):
                                    if fitted_line_position < len(mc_results[i]['amplitude_errors']):
                                        emission_amplitude_errors[i, j] = mc_results[i]['amplitude_errors'][fitted_line_position]

                                # Extract AON
                                if 'aon' in mc_results[i] and isinstance(mc_results[i]['aon'], np.ndarray):
                                    if fitted_line_position < len(mc_results[i]['aon']):
                                        emission_aon[i, j] = mc_results[i]['aon'][fitted_line_position]

                        except ValueError:
                            # Leave as NaN (already initialized)
                            pass

    # ========================
    # SAVE MAIN EMI ANALYSIS RESULTS
    outfits_emi = outdir + rootname + '_emippxf.fits'
    ExGalutil.prettyOutput_Running("Writing: " + rootname + '_emippxf.fits')

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Table HDU with emi kinematics and emission line data - ALL LINES + PER-LINE KINEMATICS
    cols = []
    cols.append(fits.Column(name='BIN_ID', format='J', array=ubins))
    cols.append(fits.Column(name='APS_ID', format='J', array=APS_ID))
    cols.append(fits.Column(name='TARGID', format='40A', array=TARGID))
    cols.append(fits.Column(name='CNAME', format='40A', array=CNAME))

    # Add redshift information
    cols.append(fits.Column(name='Z_INPUT', format='D', array=Z))
    cols.append(fits.Column(name='ERR_Z_INPUT', format='D', array=ZERR))

    # Emission line flux columns - FOR ALL LINES (fitted and non-fitted)
    for i, line in enumerate(all_line_names):
        cols.append(fits.Column(name=f'FLUX_{line}', unit=flux_unit_str, format='D', array=emission_fluxes[:, i]))
        cols.append(fits.Column(name=f'ERR_FLUX_{line}', unit=flux_unit_str, format='D', array=emission_flux_errors[:, i]))
        cols.append(fits.Column(name=f'AMPL_{line}', unit=flux_unit_str, format='D', array=emission_amplitudes[:, i]))
        cols.append(fits.Column(name=f'ERR_AMPL_{line}', unit=flux_unit_str, format='D', array=emission_amplitude_errors[:, i]))
        cols.append(fits.Column(name=f'AON_{line}', format='D', array=emission_aon[:, i]))

        # UPDATED: Per-line kinematics columns (will be NaN if flux not meaningful)
        cols.append(fits.Column(name=f'V_{line}', unit='km/s', format='D', array=line_velocities_all[:, i]))
        cols.append(fits.Column(name=f'SIGMA_{line}', unit='km/s', format='D', array=line_sigmas_all[:, i]))
        cols.append(fits.Column(name=f'ERR_V_{line}', unit='km/s', format='D', array=line_velocity_errors_all[:, i]))
        cols.append(fits.Column(name=f'ERR_SIGMA_{line}', unit='km/s', format='D', array=line_sigma_errors_all[:, i]))
        cols.append(fits.Column(name=f'FORM_ERR_V_{line}', unit='km/s', format='D', array=line_formal_velocity_errors_all[:, i]))
        cols.append(fits.Column(name=f'FORM_ERR_SIGMA_{line}', unit='km/s', format='D', array=line_formal_sigma_errors_all[:, i]))

    # Create the emi table
    emiHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    emiHDU.name = 'EMI_ANALYSIS'

    # Add headers with line categorization information
    emiHDU.header['CNF_EMI'] = (configs['CONFIG_FILE'], 'Configs. Filename used by emi analysis')
    emiHDU.header['TEMPL'] = (configs['SSP_LIB'], 'The library of spectral template')
    emiHDU.header['PPXF_V'] = (PPXF_VERSION, 'PPXF version')
    emiHDU.header['APSVERS'] = (APSVERS, 'APS version')
    emiHDU.header['APSPPXFV'] = (aps_constants.__aps_ppxf_version__, 'PyAPS (MOS) PPXF wrapper version')
    emiHDU.header['CSB_PPXF'] = (configs['stitched'], 'Combines Spectral Bands Status for PPXF')

    # Add comprehensive line categorization statistics (per-target aware)
    total_fitted = sum(n_fitted_per_target)
    total_masked = sum(n_masked_per_target)
    total_out_range = sum(n_out_range_per_target)
    total_no_wave = sum([len(cat['missing_wavelength']) for cat in target_line_categories])

    emiHDU.header['N_LINES_ALL'] = (len(all_line_names), 'Total lines in emission config')
    emiHDU.header['TOT_FITTED'] = (total_fitted, 'Total line fittings across all targets')
    emiHDU.header['TOT_MASKED'] = (total_masked, 'Total line maskings across all targets')
    emiHDU.header['TOT_OUT_RNG'] = (total_out_range, 'Total lines out-of-range across all targets')
    emiHDU.header['TOT_NO_WAVE'] = (total_no_wave, 'Total lines without wavelengths')

    # Add per-target statistics
    emiHDU.header['FIT_MIN'] = (np.min(n_fitted_per_target), 'Min fitted lines per target')
    emiHDU.header['FIT_MAX'] = (np.max(n_fitted_per_target), 'Max fitted lines per target')
    emiHDU.header['FIT_MEAN'] = (np.mean(n_fitted_per_target), 'Mean fitted lines per target')
    emiHDU.header['OUT_MIN'] = (np.min(n_out_range_per_target), 'Min out-of-range lines per target')
    emiHDU.header['OUT_MAX'] = (np.max(n_out_range_per_target), 'Max out-of-range lines per target')
    emiHDU.header['OUT_MEAN'] = (np.mean(n_out_range_per_target), 'Mean out-of-range lines per target')

    # Add individual target redshift statistics to header
    emiHDU.header['Z_MIN'] = (np.min(Z), 'Minimum redshift among targets')
    emiHDU.header['Z_MAX'] = (np.max(Z), 'Maximum redshift among targets')
    emiHDU.header['Z_MEDIAN'] = (np.median(Z), 'Median redshift among targets')
    emiHDU.header['Z_MEAN'] = (np.mean(Z), 'Mean redshift among targets')
    emiHDU.header['Z_STD'] = (np.std(Z), 'Redshift standard deviation')

    # Add tie settings information
    if tie_settings:
        emiHDU.header['TIE_ALL'] = (tie_settings.get('tie_all', False), 'All lines tied together')

        # Add information about fitted lines
        emiHDU.header['N_LINES'] = (len(line_names), 'Number of emission lines fitted')

        # Add component information if available
        if group_order:
            active_groups = []
            for group_name in group_order:
                if tie_settings.get(group_name):
                    group_lines = tie_settings[group_name]
                    present_lines = [line for line in group_lines if line in line_names]
                    if present_lines:
                        active_groups.append(f"{group_name}({len(present_lines)})")

            if active_groups:
                groups_str = ', '.join(active_groups)
                if len(groups_str) > 60:
                    groups_str = f"{len(active_groups)} active groups"
                emiHDU.header['GROUPS'] = (groups_str, 'Active tie groups')

    # Add masking information if available
    if emission_config and 'mask_regions' in emission_config:
        n_mask_regions = len(emission_config['mask_regions'])
        emiHDU.header['N_MASK_REG'] = (n_mask_regions, 'Number of mask regions applied')

        # Count different types of mask regions
        sky_masks = sum(1 for mr in emission_config['mask_regions'] if 'sky' in mr['name'].lower())
        emission_masks = n_mask_regions - sky_masks
        emiHDU.header['N_SKY_MASK'] = (sky_masks, 'Number of sky line masks')
        emiHDU.header['N_EMI_MASK'] = (emission_masks, 'Number of emission line masks')

        # Add some mask region examples
        for i, mask_region in enumerate(emission_config['mask_regions'][:10]):  # First 10
            name = mask_region['name']
            wave = mask_region['wavelength']
            width = mask_region.get('width', 3.0)
            emiHDU.header[f'MASK_{i:02d}'] = (f'{name}', f'Mask region {i+1}')

    # Keep the basename of input file (infiles)
    for n_province, province in enumerate(configs['infiles']):
        emiHDU.header['APSREF_%d' % (n_province)] = (os.path.basename(province), 'L1 reference file')

    # UPDATED: Add information about per-line kinematics with flux filtering
    emiHDU.header['PER_LINE_KIN'] = (True, 'Per-line kinematics available (V_<line>, SIGMA_<line>)')
    emiHDU.header['FLUX_FILTER'] = (True, 'Kinematics only stored for lines with flux > 0')
    emiHDU.header['COMMENT'] = 'V_<line>/SIGMA_<line>: Individual line kinematics from component tying'
    emiHDU.header['COMMENT'] = 'ERR_FLUX/AMPL_<line>: Monte Carlo errors from simulations'
    emiHDU.header['COMMENT'] = 'Kinematics (V/SIGMA) only saved for lines with meaningful flux (>0, not NaN)'
    emiHDU.header['COMMENT'] = 'Lines with poor flux have NaN kinematics even if part of tied group'
    emiHDU.header['COMMENT'] = 'For single-component fits: all detected lines have identical kinematics'
    emiHDU.header['COMMENT'] = 'For multi-component fits: V_<line> varies by component assignment'

    # Create HDU list and write
    HDUList = fits.HDUList([priHDU, emiHDU])
    HDUList.writeto(outfits_emi, overwrite=True)

    ExGalutil.prettyOutput_Done("Writing: " + rootname + '_emippxf.fits')
    logging.info("Wrote: " + outfits_emi)

    # ========================
    # SAVE EMI MODEL SPECTRA (unchanged)
    outfits_spec = outdir + rootname + '_emippxf_spec.fits'
    ExGalutil.prettyOutput_Running("Writing: " + rootname + '_emippxf_spec.fits')

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Table HDU with emi spectra
    cols = []
    cols.append(fits.Column(name='BIN_ID', format='J', array=ubins))
    cols.append(fits.Column(name='APS_ID', format='J', array=APS_ID))
    cols.append(fits.Column(name='TARGID', format='40A', array=TARGID))
    cols.append(fits.Column(name='CNAME', format='40A', array=CNAME))
    cols.append(fits.Column(name='LOGLAM_EMI', unit=wave_unit_str, format=str(npix) + 'D', array=logLam))
    cols.append(fits.Column(name='FLUX_EMI', unit=flux_unit_str, format=str(npix) + 'D', array=bin_data))
    cols.append(fits.Column(name='ERROR_EMI', unit=flux_unit_str, format=str(npix) + 'D', array=bin_error))
    cols.append(fits.Column(name='MODEL_EMI', unit=flux_unit_str, format=str(npix) + 'D', array=emi_bestfit))
    cols.append(fits.Column(name='GOODPIX_EMI', format=str(npix) + 'J', array=goodpixels))

    specHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    specHDU.name = 'EMI_SPEC'

    specHDU.header['CNF_EMI'] = (configs['CONFIG_FILE'], 'Configs. Filename used by emi analysis')
    specHDU.header['PPXF_V'] = (PPXF_VERSION, 'PPXF version')
    specHDU.header['APSVERS'] = (APSVERS, 'APS version')
    specHDU.header['SAMPLING'] = (1, 'Sampling mode (0: linear, 1: logarithmic)')

    # Add line categorization statistics to spectra file too - per-target aware
    specHDU.header['N_LINES_ALL'] = (len(all_line_names), 'Total lines in emission config')
    specHDU.header['TOT_FITTED'] = (total_fitted, 'Total line fittings across all targets')
    specHDU.header['TOT_MASKED'] = (total_masked, 'Total line maskings across all targets')
    specHDU.header['TOT_OUT_RNG'] = (total_out_range, 'Total lines out-of-range across all targets')
    specHDU.header['TOT_NO_WAVE'] = (total_no_wave, 'Total lines without wavelengths')
    specHDU.header['FIT_MIN'] = (np.min(n_fitted_per_target), 'Min fitted lines per target')
    specHDU.header['FIT_MAX'] = (np.max(n_fitted_per_target), 'Max fitted lines per target')
    specHDU.header['FIT_MEAN'] = (np.mean(n_fitted_per_target), 'Mean fitted lines per target')

    # UPDATED: Add flux filtering info to spectra header
    specHDU.header['FLUX_FILTER'] = (True, 'Kinematics only stored for lines with flux > 0')

    # Keep the basename of input file (infiles)
    for n_province, province in enumerate(configs['infiles']):
        specHDU.header['APSREF_%d' % (n_province)] = (os.path.basename(province), 'L1 reference file')

    HDUList = fits.HDUList([priHDU, specHDU])
    HDUList.writeto(outfits_spec, overwrite=True)

    ExGalutil.prettyOutput_Done("Writing: " + rootname + '_emippxf_spec.fits')
    logging.info("Wrote: " + outfits_spec)

    # ========================
    # SAVE ADDITIONAL LINE MAPPING FILE (updated with per-line kinematics info)
    outfits_mapping = outdir + rootname + '_emippxf_mapping.fits'
    ExGalutil.prettyOutput_Running("Writing: " + rootname + '_emippxf_mapping.fits')

    # Create mapping table showing ALL lines and their per-target status
    mapping_cols = []
    mapping_cols.append(fits.Column(name='LINE_NAME', format='A20', array=all_line_names))

    # Add wavelengths for all lines
    wavelength_array = [all_line_wavelengths.get(name, np.nan) for name in all_line_names]
    mapping_cols.append(fits.Column(name='WAVELENGTH', format='D', array=wavelength_array))

    # Add observed wavelengths (with individual redshifts for each target)
    if len(Z) > 0:
        # For mapping file, show range of observed wavelengths across all targets
        wavelength_obs_min = []
        wavelength_obs_max = []
        wavelength_obs_mean = []

        for w_rest in wavelength_array:
            if not np.isnan(w_rest):
                obs_waves = [w_rest * (1 + z_target) for z_target in Z]
                wavelength_obs_min.append(np.min(obs_waves))
                wavelength_obs_max.append(np.max(obs_waves))
                wavelength_obs_mean.append(np.mean(obs_waves))
            else:
                wavelength_obs_min.append(np.nan)
                wavelength_obs_max.append(np.nan)
                wavelength_obs_mean.append(np.nan)

        mapping_cols.append(fits.Column(name='WAVELENGTH_OBS_MIN', format='D', array=wavelength_obs_min))
        mapping_cols.append(fits.Column(name='WAVELENGTH_OBS_MAX', format='D', array=wavelength_obs_max))
        mapping_cols.append(fits.Column(name='WAVELENGTH_OBS_MEAN', format='D', array=wavelength_obs_mean))

    # Add per-line statistics across all targets
    n_fitted_per_line = []
    n_masked_per_line = []
    n_out_range_per_line = []
    most_common_status_per_line = []

    # UPDATED: Add per-line kinematics statistics with flux filtering
    mean_velocity_per_line = []
    std_velocity_per_line = []
    mean_sigma_per_line = []
    std_sigma_per_line = []
    n_valid_kinematics_per_line = []

    # UPDATED: Add flux statistics per line
    mean_flux_per_line = []
    std_flux_per_line = []
    n_positive_flux_per_line = []

    # NEW: Add formal error statistics
    mean_formal_flux_errors_per_line = []
    mean_formal_ampl_errors_per_line = []

    for line_idx in range(len(all_line_names)):
        line_statuses = [target_line_status[t][line_idx] for t in range(len(ubins))]

        n_fitted_per_line.append(line_statuses.count('FITTED'))
        n_masked_per_line.append(line_statuses.count('MASKED'))
        n_out_range_per_line.append(line_statuses.count('OUT_OF_RANGE'))

        # Find most common status
        status_counts = {}
        for status in line_statuses:
            status_counts[status] = status_counts.get(status, 0) + 1
        most_common_status = max(status_counts, key=status_counts.get)
        most_common_status_per_line.append(most_common_status)

        # UPDATED: Calculate per-line kinematics statistics (only for meaningful flux)
        line_velocities = line_velocities_all[:, line_idx]
        line_sigmas = line_sigmas_all[:, line_idx]
        line_fluxes = emission_fluxes[:, line_idx]

        valid_vel_mask = np.isfinite(line_velocities)
        valid_sig_mask = np.isfinite(line_sigmas)
        valid_flux_mask = np.isfinite(line_fluxes)
        positive_flux_mask = valid_flux_mask & (line_fluxes > 0)

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

        # UPDATED: Calculate flux statistics
        if np.any(valid_flux_mask):
            mean_flux_per_line.append(np.mean(line_fluxes[valid_flux_mask]))
            std_flux_per_line.append(np.std(line_fluxes[valid_flux_mask]))
        else:
            mean_flux_per_line.append(np.nan)
            std_flux_per_line.append(np.nan)

        n_positive_flux_per_line.append(np.sum(positive_flux_mask))

        # NEW: Calculate formal error statistics
        formal_flux_errs = line_formal_flux_errors_all[:, line_idx]
        formal_ampl_errs = line_formal_amplitude_errors_all[:, line_idx]

        valid_flux_err_mask = np.isfinite(formal_flux_errs)
        valid_ampl_err_mask = np.isfinite(formal_ampl_errs)

        if np.any(valid_flux_err_mask):
            mean_formal_flux_errors_per_line.append(np.mean(formal_flux_errs[valid_flux_err_mask]))
        else:
            mean_formal_flux_errors_per_line.append(np.nan)

        if np.any(valid_ampl_err_mask):
            mean_formal_ampl_errors_per_line.append(np.mean(formal_ampl_errs[valid_ampl_err_mask]))
        else:
            mean_formal_ampl_errors_per_line.append(np.nan)

    mapping_cols.append(fits.Column(name='N_FITTED', format='J', array=n_fitted_per_line))
    mapping_cols.append(fits.Column(name='N_MASKED', format='J', array=n_masked_per_line))
    mapping_cols.append(fits.Column(name='N_OUT_RANGE', format='J', array=n_out_range_per_line))
    mapping_cols.append(fits.Column(name='MOST_COMMON_STATUS', format='A15', array=most_common_status_per_line))

    # UPDATED: Add per-line kinematics statistics columns
    mapping_cols.append(fits.Column(name='MEAN_VELOCITY', unit='km/s', format='D', array=mean_velocity_per_line))
    mapping_cols.append(fits.Column(name='STD_VELOCITY', unit='km/s', format='D', array=std_velocity_per_line))
    mapping_cols.append(fits.Column(name='MEAN_SIGMA', unit='km/s', format='D', array=mean_sigma_per_line))
    mapping_cols.append(fits.Column(name='STD_SIGMA', unit='km/s', format='D', array=std_sigma_per_line))
    mapping_cols.append(fits.Column(name='N_VALID_KINEMATICS', format='J', array=n_valid_kinematics_per_line))

    # UPDATED: Add flux statistics columns
    mapping_cols.append(fits.Column(name='MEAN_FLUX', unit=flux_unit_str, format='D', array=mean_flux_per_line))
    mapping_cols.append(fits.Column(name='STD_FLUX', unit=flux_unit_str, format='D', array=std_flux_per_line))
    mapping_cols.append(fits.Column(name='N_POSITIVE_FLUX', format='J', array=n_positive_flux_per_line))

    # NEW: Add formal error statistics columns
    mapping_cols.append(fits.Column(name='MEAN_FORMAL_FLUX_ERR', unit=flux_unit_str, format='D', array=mean_formal_flux_errors_per_line))
    mapping_cols.append(fits.Column(name='MEAN_FORMAL_AMPL_ERR', unit=flux_unit_str, format='D', array=mean_formal_ampl_errors_per_line))

    # UPDATED: Add information about why lines were not fitted (generalized with flux info)
    reason_array = []
    for line_idx, line_name in enumerate(all_line_names):
        status = most_common_status_per_line[line_idx]
        n_fitted = n_fitted_per_line[line_idx]
        n_out = n_out_range_per_line[line_idx]
        n_masked = n_masked_per_line[line_idx]
        n_valid_kin = n_valid_kinematics_per_line[line_idx]
        n_pos_flux = n_positive_flux_per_line[line_idx]

        if status == 'FITTED':
            if n_fitted == len(ubins):
                reason_array.append(f'Fitted for all targets ({n_pos_flux} pos. flux, {n_valid_kin} kinematics)')
            else:
                reason_array.append(f'Fitted for {n_fitted}/{len(ubins)} targets ({n_pos_flux} pos. flux, {n_valid_kin} kinematics)')
        elif status == 'OUT_OF_RANGE':
            reason_array.append(f'Out-of-range for {n_out}/{len(ubins)} targets due to redshift')
        elif status == 'MASKED':
            reason_array.append(f'Masked for {n_masked}/{len(ubins)} targets (sky/bad pixels)')
        elif status == 'NO_WAVELENGTH':
            reason_array.append('No wavelength defined in config')
        else:
            reason_array.append('Mixed/other reasons across targets')

    mapping_cols.append(fits.Column(name='REASON', format='A120', array=reason_array))

    mappingHDU = fits.BinTableHDU.from_columns(fits.ColDefs(mapping_cols))
    mappingHDU.name = 'LINE_MAPPING'

    # Add comprehensive statistics to mapping header - per-target aware + per-line kinematics + flux filtering
    mappingHDU.header['N_LINES_ALL'] = (len(all_line_names), 'Total lines in emission config')
    mappingHDU.header['N_TARGETS'] = (len(ubins), 'Total number of targets')
    mappingHDU.header['TOT_FITTED'] = (total_fitted, 'Total line fittings across all targets')
    mappingHDU.header['TOT_MASKED'] = (total_masked, 'Total line maskings across all targets')
    mappingHDU.header['TOT_OUT_RNG'] = (total_out_range, 'Total lines out-of-range across all targets')
    mappingHDU.header['TOT_NO_WAVE'] = (total_no_wave, 'Total lines without wavelengths')

    # Add per-target statistics ranges
    mappingHDU.header['FIT_MIN'] = (np.min(n_fitted_per_target), 'Min fitted lines per target')
    mappingHDU.header['FIT_MAX'] = (np.max(n_fitted_per_target), 'Max fitted lines per target')
    mappingHDU.header['FIT_MEAN'] = (np.mean(n_fitted_per_target), 'Mean fitted lines per target')
    mappingHDU.header['OUT_MIN'] = (np.min(n_out_range_per_target), 'Min out-of-range lines per target')
    mappingHDU.header['OUT_MAX'] = (np.max(n_out_range_per_target), 'Max out-of-range lines per target')
    mappingHDU.header['OUT_MEAN'] = (np.mean(n_out_range_per_target), 'Mean out-of-range lines per target')

    # Add per-line statistics ranges
    mappingHDU.header['LINE_FIT_MIN'] = (np.min(n_fitted_per_line), 'Min targets fitting each line')
    mappingHDU.header['LINE_FIT_MAX'] = (np.max(n_fitted_per_line), 'Max targets fitting each line')

    # UPDATED: Add per-line kinematics statistics to header
    valid_velocities = [v for v in mean_velocity_per_line if not np.isnan(v)]
    valid_sigmas = [s for s in mean_sigma_per_line if not np.isnan(s)]

    if len(valid_velocities) > 0:
        mappingHDU.header['V_LINE_MIN'] = (np.min(valid_velocities), 'Min mean velocity across lines')
        mappingHDU.header['V_LINE_MAX'] = (np.max(valid_velocities), 'Max mean velocity across lines')
        mappingHDU.header['V_LINE_RANGE'] = (np.max(valid_velocities) - np.min(valid_velocities), 'Velocity range across lines')

    if len(valid_sigmas) > 0:
        mappingHDU.header['S_LINE_MIN'] = (np.min(valid_sigmas), 'Min mean sigma across lines')
        mappingHDU.header['S_LINE_MAX'] = (np.max(valid_sigmas), 'Max mean sigma across lines')
        mappingHDU.header['S_LINE_RANGE'] = (np.max(valid_sigmas) - np.min(valid_sigmas), 'Sigma range across lines')

    mappingHDU.header['N_LINES_KIN'] = (len(valid_velocities), 'Number of lines with valid kinematics')

    # UPDATED: Add flux filtering statistics to header
    total_positive_flux = sum(n_positive_flux_per_line)
    mappingHDU.header['TOT_POS_FLUX'] = (total_positive_flux, 'Total positive flux measurements')
    mappingHDU.header['FLUX_FILTER'] = (True, 'Kinematics only stored for lines with flux > 0')

    # Add wavelength range info
    mappingHDU.header['LMIN_USED'] = (wavelength_range_config[0], 'Minimum wavelength used for fitting')
    mappingHDU.header['LMAX_USED'] = (wavelength_range_config[1], 'Maximum wavelength used for fitting')

    if len(Z) > 0:
        mappingHDU.header['Z_MIN'] = (np.min(Z), 'Minimum redshift among targets')
        mappingHDU.header['Z_MAX'] = (np.max(Z), 'Maximum redshift among targets')
        mappingHDU.header['Z_RANGE'] = (np.max(Z) - np.min(Z), 'Redshift range among targets')
        mappingHDU.header['Z_MEDIAN'] = (np.median(Z), 'Median redshift among targets')

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

    # UPDATED: Add comments about data interpretation - with flux filtering info
    mappingHDU.header['COMMENT'] = 'LINE_STATUS describes why each line was/was not fitted'
    mappingHDU.header['COMMENT'] = 'FITTED: Line successfully fitted and has real measurements'
    mappingHDU.header['COMMENT'] = 'MASKED: Line excluded due to sky contamination, bad pixels, etc.'
    mappingHDU.header['COMMENT'] = 'OUT_OF_RANGE: Line outside fitting wavelength range'
    mappingHDU.header['COMMENT'] = 'NO_WAVELENGTH: Line has no wavelength defined in config'
    mappingHDU.header['COMMENT'] = 'Per-line kinematics: Each fitted line has individual V and sigma'
    mappingHDU.header['COMMENT'] = 'FLUX FILTERING: Kinematics only saved for lines with flux > 0'
    mappingHDU.header['COMMENT'] = 'N_POSITIVE_FLUX: Number of targets with positive flux for each line'
    mappingHDU.header['COMMENT'] = 'N_VALID_KINEMATICS: Number of targets with kinematics for each line'
    mappingHDU.header['COMMENT'] = 'MEAN_VELOCITY/SIGMA: Average kinematics across valid targets per line'
    mappingHDU.header['COMMENT'] = 'STD_VELOCITY/SIGMA: Standard deviation of kinematics for each line'
    mappingHDU.header['COMMENT'] = 'For single-component fits: all detected lines have identical kinematics'
    mappingHDU.header['COMMENT'] = 'For multi-component fits: lines tied by component have same kinematics'
    mappingHDU.header['COMMENT'] = 'Lines with weak/negative flux get NaN kinematics despite group membership'

    # Create HDU list and write mapping file
    HDUList = fits.HDUList([fits.PrimaryHDU(), mappingHDU])
    HDUList.writeto(outfits_mapping, overwrite=True)

    ExGalutil.prettyOutput_Done("Writing: " + rootname + '_emippxf_mapping.fits')
    logging.info("Wrote: " + outfits_mapping)

    if debug:
        print(f"Enhanced MOS emi kinematics output with flux-filtered per-line kinematics:")
        print(f"  Total lines in config: {len(all_line_names)}")
        print(f"  Redshift range: {np.min(Z):.3f} - {np.max(Z):.3f}")
        print(f"  Lines fitted per target: {np.min(n_fitted_per_target)} - {np.max(n_fitted_per_target)} (mean: {np.mean(n_fitted_per_target):.1f})")
        print(f"  Lines out-of-range per target: {np.min(n_out_range_per_target)} - {np.max(n_out_range_per_target)} (mean: {np.mean(n_out_range_per_target):.1f})")
        print(f"  Total line fittings across all targets: {total_fitted}")
        print(f"  Total lines out-of-range across all targets: {total_out_range}")
        print(f"  Total positive flux measurements: {total_positive_flux}")
        print(f"  Lines with valid kinematics: {len(valid_velocities)}/{len(all_line_names)}")

        if len(valid_velocities) > 0:
            print(f"  Velocity range across lines: {np.min(valid_velocities):.1f} to {np.max(valid_velocities):.1f} km/s")
        if len(valid_sigmas) > 0:
            print(f"  Sigma range across lines: {np.min(valid_sigmas):.1f} to {np.max(valid_sigmas):.1f} km/s")

        print(f"  Using flux filtering: kinematics only for lines with flux > 0")

    if emission_config and 'mask_regions' in emission_config:
        n_mask_regions = len(emission_config['mask_regions'])
        sky_masks = sum(1 for mr in emission_config['mask_regions'] if 'sky' in mr['name'].lower())
        if debug:
            print(f"  Applied {n_mask_regions} mask regions ({sky_masks} sky, {n_mask_regions-sky_masks} emission)")

    if debug:
        print(f"  Files created: _emi.fits (with flux-filtered per-line V/Sigma), _emi_spec.fits, _emi_mapping.fits")

    return True



def categorize_emission_lines(emission_config, fitted_line_names, wavelength_range, wavelength_range_config, line_names_fit):
    """
    Categorize emission lines into fitted, masked, out-of-range, and missing wavelength.

    Parameters:
    -----------
    emission_config : dict
        Loaded emission configuration with emission_lines, mask_regions, etc.
    fitted_line_names : list
        Names of lines that were actually fitted
    wavelength_range : tuple
        (min_wavelength, max_wavelength) used for fitting


    Returns:
    --------
    dict : Dictionary with keys 'fitted', 'masked', 'out_of_range', 'missing_wavelength'
    """

    if emission_config is None or 'emission_lines' not in emission_config:
        # Fallback: all fitted lines are considered fitted
        return {
            'fitted': fitted_line_names,
            'masked': [],
            'out_of_range': [],
            'missing_wavelength': []
        }

    all_lines = emission_config['emission_lines']
    mask_regions = emission_config.get('mask_regions', [])

    # Get masked line names from mask regions
    masked_line_names = set()
    for mask_region in mask_regions:
        name = mask_region['name']
        # Check if this is an emission line mask (not sky line)
        if name in all_lines:
            masked_line_names.add(name)

    categorized = {
        'fitted': [],
        'masked': [],
        'out_of_range': [],
        'missing_wavelength': []
    }

    for line_name in all_lines:
        rest_wavelength = all_lines[line_name]

        if rest_wavelength <= 0 or not np.isfinite(rest_wavelength):
            categorized['missing_wavelength'].append(line_name)
        elif line_name in masked_line_names:
            categorized['masked'].append(line_name)
        else:
            # Check if line is within wavelength range
            # also for the fitted lines we check if the line is already in the line_names_fit (coming from ppxf worker)
            min_allowed = max(wavelength_range[0] - 50.0 , wavelength_range_config[0])
            max_allowed = min(wavelength_range[1] + 50.0 , wavelength_range_config[1])

            if rest_wavelength < min_allowed or rest_wavelength > max_allowed:
                categorized['out_of_range'].append(line_name)
            elif line_name in fitted_line_names and line_name in line_names_fit:
                categorized['fitted'].append(line_name)
            else:
                categorized['masked'].append(line_name)  # In range but not fitted

    return categorized




def save_ppxf_as_emippxf(LEVEL, rootname, configs, outdir, metalist, spectra, error,
                         emi_result, mc_results, formal_error, emission_lines, line_names,
                         line_wavelengths, logLam, bestfit, goodpixels, emi_bestfit,
                         stellar_bestfit, npix, ubins, emission_config=None,
                         tie_settings=None, group_order=None, z_in=None, z_err=None,
                         reddening=None, debug=False):
    """
    Save pPXF emi kinematics results in the exact same format as EMIPPXF output.
    UPDATED: Now uses per-line kinematics (no backward compatibility columns).
    Creates results for ALL emission lines in config (fitted, masked, out-of-range).
    UPDATED: Only stores kinematics for lines with meaningful flux (flux > 0 and not NaN).

    This function converts pPXF results to match the EMIPPXF output format exactly,
    ensuring compatibility with downstream analysis tools and consistent output structure.

    Note: line_names is a list for each target comming from the PPXF worker
    """

    # Extract metadata from metalist
    APS_ID = [item['APS_ID'] for item in metalist]
    CNAME = [item['CNAME'] for item in metalist]
    TARGID = [item['TARGID'] for item in metalist]
    BINID_meta = [item['BIN_ID'] for item in metalist]
    Z = np.array([item['Z'] for item in metalist])
    ZERR = np.array([item['ZERR'] for item in metalist])

    # Consistency check
    assert BINID_meta == list(ubins), 'input and output BIN_ID are not identical'

    # Determine wavelength range used for fitting
    if 'LMIN_EMI' in configs and 'LMAX_EMI' in configs:
        wavelength_range_config = (configs['LMIN_EMI'], configs['LMAX_EMI'])
    elif 'LMIN_PPXF' in configs and 'LMAX_PPXF' in configs:
        wavelength_range_config = (configs['LMIN_PPXF'], configs['LMAX_PPXF'])
    else:
        # Fallback to data range
        wavelength_range_config = (np.nan, np.nan)

    if debug:
        # Categorize lines individually for each target (MOS has wide redshift range!)
        print(f"Processing {len(ubins)} targets with individual redshift-dependent line categorization")
        print(f"Redshift range: {np.min(Z):.3f} - {np.max(Z):.3f}")

    # Initialize arrays to track which lines are fitted/masked/out-of-range for each target
    assert emission_config and 'emission_lines' in emission_config, 'no config file found to build all_line_names'
    all_line_names = list(emission_config['emission_lines'].keys())
    all_line_wavelengths = emission_config['emission_lines']

    # Create per-target line categorization
    target_line_categories = []
    target_fitted_indices = []
    target_line_status = []

    for i, (target_z, bin_id) in enumerate(zip(Z, ubins)):
        # Categorize lines for this specific target's redshift
        lam_i = np.exp(logLam[i])
        wavelength_range_i = (lam_i.min(), lam_i.max())

        target_categories = categorize_emission_lines(
            emission_config,
            all_line_names,
            wavelength_range_i,
            wavelength_range_config,
            line_names[i]
        )
        target_line_categories.append(target_categories)

        # Create mapping from fitted lines to all lines for this target
        fitted_indices = []
        line_status = []

        for name in all_line_names:
            if name in target_categories['fitted']:
                try:
                    fitted_idx = line_names[i].index(name)  # Find in actually fitted lines
                    fitted_indices.append(fitted_idx)
                    line_status.append('FITTED')
                except ValueError:
                    # Line supposedly fitted but not in line_names - this is the bug!
                    print(f"WARNING: Line {name} marked as fitted but not in fitted line_names")
                    fitted_indices.append(-1)
                    line_status.append('ERROR')
            elif name in target_categories['masked']:
                fitted_indices.append(-1)
                line_status.append('MASKED')
            elif name in target_categories['out_of_range']:
                fitted_indices.append(-1)
                line_status.append('OUT_OF_RANGE')
            else:
                fitted_indices.append(-1)
                line_status.append('OTHER')

        target_fitted_indices.append(fitted_indices)
        target_line_status.append(line_status)

    # Print summary statistics
    n_fitted_per_target = [len(cat['fitted']) for cat in target_line_categories]
    n_masked_per_target = [len(cat['masked']) for cat in target_line_categories]
    n_out_range_per_target = [len(cat['out_of_range']) for cat in target_line_categories]

    if debug:
        print(f"  Lines fitted per target: {np.min(n_fitted_per_target)} - {np.max(n_fitted_per_target)} (mean: {np.mean(n_fitted_per_target):.1f})")
        print(f"  Lines out-of-range per target: {np.min(n_out_range_per_target)} - {np.max(n_out_range_per_target)} (mean: {np.mean(n_out_range_per_target):.1f})")
        print(f"  Lines masked per target: {np.min(n_masked_per_target)} - {np.max(n_masked_per_target)} (mean: {np.mean(n_masked_per_target):.1f})")

    # Set flux units
    if configs['sens_corr']:
        flux_unit_str = '%2e erg/(s cm**2 Angstrom)' % (configs['funits'])
        ivar_unit_str = '%2e cm**4 Angstrom**2 /(s**2 erg**2)' % (configs['funits']**-2)
    else:
        flux_unit_str = 'count'
        ivar_unit_str = '1/count**2'
    wave_unit_str = 'Angstrom'

    # Extract emission line properties from mc_results - for ALL lines, per target
    emission_amplitudes_all = np.full((len(ubins), len(all_line_names)), np.nan)
    emission_amplitude_errors_all = np.full((len(ubins), len(all_line_names)), np.nan)
    emission_aon_all = np.full((len(ubins), len(all_line_names)), np.nan)
    emission_fluxes_all = np.full((len(ubins), len(all_line_names)), np.nan)
    emission_flux_errors_all = np.full((len(ubins), len(all_line_names)), np.nan)
    # Equivalent widths -- see ExGalPrepare.compute_equivalent_width's own
    # docstring for the sign convention and error-propagation caveat.
    equivalent_widths_all = np.full((len(ubins), len(all_line_names)), np.nan)
    equivalent_width_errors_all = np.full((len(ubins), len(all_line_names)), np.nan)

    # NEW: Extract per-line kinematics - for ALL lines, per target
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
        if isinstance(mc_results[i], dict):
            # Use this target's specific line categorization

            fitted_indices = target_fitted_indices[i]

            # Fill data for fitted lines only (based on this target's redshift)
            for j, line_name in enumerate(all_line_names):
                # we put try as only some lines from the full list are available in the results list
                try:
                    fitted_line_position = line_names[i].index(line_name)
                    if fitted_line_position >= 0:  # Line was fitted for this target
                        # Extract fluxes first to check if meaningful
                        line_flux = np.nan
                        if isinstance(emission_lines[i], np.ndarray) and len(emission_lines[i]) > fitted_line_position:
                            line_flux = emission_lines[i][fitted_line_position]
                            emission_fluxes_all[i, j] = line_flux

                        # UPDATED: Only store kinematics if flux is meaningful (> 0 and not NaN)
                        flux_is_meaningful = (not np.isnan(line_flux) and line_flux is not None and line_flux > 1e-10 and line_name in target_line_categories[i]['fitted'])


                        if not flux_is_meaningful:
                            emission_fluxes_all[i, j] = np.nan

                        if flux_is_meaningful:
                            # Extract amplitudes
                            if 'amplitudes' in mc_results[i] and isinstance(mc_results[i]['amplitudes'], np.ndarray):
                                if len(mc_results[i]['amplitudes']) > fitted_line_position:
                                    emission_amplitudes_all[i, j] = mc_results[i]['amplitudes'][fitted_line_position]

                            # Extract amplitude errors
                            if 'amplitude_errors' in mc_results[i] and isinstance(mc_results[i]['amplitude_errors'], np.ndarray):
                                if len(mc_results[i]['amplitude_errors']) > fitted_line_position:
                                    emission_amplitude_errors_all[i, j] = mc_results[i]['amplitude_errors'][fitted_line_position]

                            # Extract AON
                            if 'aon' in mc_results[i] and isinstance(mc_results[i]['aon'], np.ndarray):
                                if len(mc_results[i]['aon']) > fitted_line_position:
                                    emission_aon_all[i, j] = mc_results[i]['aon'][fitted_line_position]

                            # Extract flux errors
                            if 'fluxes' in mc_results[i] and isinstance(mc_results[i]['fluxes'], np.ndarray):
                                if len(mc_results[i]['fluxes']) > fitted_line_position:
                                    emission_flux_errors_all[i, j] = mc_results[i]['fluxes'][fitted_line_position]

                            # Equivalent width -- FLUX / local stellar
                            # continuum at this line's observed
                            # wavelength (see ExGalPrepare.
                            # compute_equivalent_width's docstring).
                            if stellar_bestfit is not None:
                                rest_wave = all_line_wavelengths.get(line_name, np.nan)
                                wave_i = np.exp(logLam[i])
                                ew, cont_at_line = ExGalPrepare.compute_equivalent_width(
                                    line_flux, rest_wave, Z[i], wave_i, stellar_bestfit[i])
                                equivalent_widths_all[i, j] = ew
                                flux_err = emission_flux_errors_all[i, j]
                                if np.isfinite(cont_at_line) and np.isfinite(flux_err):
                                    equivalent_width_errors_all[i, j] = flux_err / cont_at_line

                        # UPDATED: Only extract per-line kinematics if flux is meaningful
                        if flux_is_meaningful:
                            if 'line_velocities' in mc_results[i] and isinstance(mc_results[i]['line_velocities'], np.ndarray):
                                if len(mc_results[i]['line_velocities']) > fitted_line_position:
                                    line_velocities_all[i, j] = mc_results[i]['line_velocities'][fitted_line_position]

                            if 'line_sigmas' in mc_results[i] and isinstance(mc_results[i]['line_sigmas'], np.ndarray):
                                if len(mc_results[i]['line_sigmas']) > fitted_line_position:
                                    line_sigmas_all[i, j] = mc_results[i]['line_sigmas'][fitted_line_position]

                            if 'line_velocity_errors' in mc_results[i] and isinstance(mc_results[i]['line_velocity_errors'], np.ndarray):
                                if len(mc_results[i]['line_velocity_errors']) > fitted_line_position:
                                    line_velocity_errors_all[i, j] = mc_results[i]['line_velocity_errors'][fitted_line_position]

                            if 'line_sigma_errors' in mc_results[i] and isinstance(mc_results[i]['line_sigma_errors'], np.ndarray):
                                if len(mc_results[i]['line_sigma_errors']) > fitted_line_position:
                                    line_sigma_errors_all[i, j] = mc_results[i]['line_sigma_errors'][fitted_line_position]

                            # UPDATED: Only extract per-line formal errors if flux is meaningful
                            if isinstance(formal_error[i], dict):
                                if 'line_velocities' in formal_error[i] and isinstance(formal_error[i]['line_velocities'], np.ndarray):
                                    if len(formal_error[i]['line_velocities']) > fitted_line_position:
                                        line_formal_velocity_errors_all[i, j] = formal_error[i]['line_velocities'][fitted_line_position]

                                if 'line_sigmas' in formal_error[i] and isinstance(formal_error[i]['line_sigmas'], np.ndarray):
                                    if len(formal_error[i]['line_sigmas']) > fitted_line_position:
                                        line_formal_sigma_errors_all[i, j] = formal_error[i]['line_sigmas'][fitted_line_position]

                        # UPDATED: Extract per-line formal flux/amplitude errors regardless of flux value
                        # (these are measurement uncertainties, not kinematics)
                        if isinstance(formal_error[i], dict):
                            if 'fluxes' in formal_error[i] and isinstance(formal_error[i]['fluxes'], np.ndarray):
                                if len(formal_error[i]['fluxes']) > fitted_line_position:
                                    line_formal_flux_errors_all[i, j] = formal_error[i]['fluxes'][fitted_line_position]

                            if 'amplitudes' in formal_error[i] and isinstance(formal_error[i]['amplitudes'], np.ndarray):
                                if len(formal_error[i]['amplitudes']) > fitted_line_position:
                                    line_formal_amplitude_errors_all[i, j] = formal_error[i]['amplitudes'][fitted_line_position]
                except:
                    # Non-fitted lines for this target remain np.nan
                    pass
    # Reconstruct emission_setup structure from pPXF results - for ALL lines
    nlines = len(all_line_names)  # Use ALL lines, not just fitted
    idx_l = list(range(nlines))  # Sequential indices for ALL lines

    # Create EMIPPXF-style emission setup arrays - for ALL lines
    iis = []
    names = []
    lambdas = []
    actions = []
    kinds = []
    aas = []
    vs = []
    ss = []
    ffits = []
    aons = []

    # Build emission setup data from ALL lines in emission config
    # Use most common action for each line across targets
    for i, line_name in enumerate(all_line_names):
        iis.append(i)
        names.append(str(line_name))
        lambdas.append(all_line_wavelengths.get(line_name, 0.0))

        # Determine most common action across all targets for this line
        line_actions = [target_line_status[t][i] for t in range(len(ubins))]
        action_counts = {}
        for action in line_actions:
            action_counts[action] = action_counts.get(action, 0) + 1

        most_common_action = max(action_counts, key=action_counts.get)

        # Set action based on most common categorization
        if most_common_action == 'FITTED':
            actions.append('f')  # Fitted
            ffits.append('y')    # Actually fitted
        elif most_common_action == 'MASKED':
            actions.append('m')  # Masked
            ffits.append('n')    # Not fitted
        elif most_common_action == 'OUT_OF_RANGE':
            actions.append('o')  # Out of range
            ffits.append('n')    # Not fitted
        elif most_common_action == 'NO_WAVELENGTH':
            actions.append('n')  # No wavelength
            ffits.append('n')    # Not fitted
        else:
            actions.append('u')  # Unknown
            ffits.append('n')    # Not fitted

        kinds.append('emission')  # All are emission lines
        aas.append(1.0)  # Default amplitude
        vs.append(0.0)   # Default velocity (will be filled from results)
        ss.append(configs.get('SIGMA', 150.0))  # Default sigma
        aons.append(0.0)  # Will be filled from results

    # Convert pPXF results to EMIPPXF format
    ntargets = len(ubins)

    # Initialize solution arrays in EMIPPXF format (4 parameters per line: F, A, V, S)
    sol = np.zeros((ntargets, nlines * 4))
    esol = np.zeros((ntargets, nlines * 4))
    sol_emi_AoN = np.zeros((ntargets, nlines))

    # Fill solution arrays from pPXF results using per-line kinematics
    for i in range(ntargets):
        # Fill EMIPPXF-style solution array (F, A, V, S for ALL lines, per target)
        fitted_indices = target_fitted_indices[i]  # This target's specific line categorization

        for j in range(nlines):
            fitted_idx = fitted_indices[j]

            if fitted_idx >= 0:  # Line was fitted for this target
                # Flux
                sol[i, j*4 + 0] = emission_fluxes_all[i, j]
                esol[i, j*4 + 0] = emission_flux_errors_all[i, j]

                # Amplitude
                sol[i, j*4 + 1] = emission_amplitudes_all[i, j]
                esol[i, j*4 + 1] = emission_amplitude_errors_all[i, j]

                # UPDATED: Use per-line velocity and sigma (only if flux is meaningful)
                sol[i, j*4 + 2] = line_velocities_all[i, j]  # Will be NaN if flux not meaningful
                esol[i, j*4 + 2] = line_velocity_errors_all[i, j]

                sol[i, j*4 + 3] = line_sigmas_all[i, j]  # Will be NaN if flux not meaningful
                esol[i, j*4 + 3] = line_sigma_errors_all[i, j]

                # Amplitude over Noise
                sol_emi_AoN[i, j] = emission_aon_all[i, j]
            else:
                # Line was not fitted for this target - set to NaN
                sol[i, j*4 + 0] = np.nan
                esol[i, j*4 + 0] = np.nan
                sol[i, j*4 + 1] = np.nan
                esol[i, j*4 + 1] = np.nan
                sol[i, j*4 + 2] = np.nan
                esol[i, j*4 + 2] = np.nan
                sol[i, j*4 + 3] = np.nan
                esol[i, j*4 + 3] = np.nan
                sol_emi_AoN[i, j] = np.nan

    # Handle reddening parameters
    if reddening is not None and len(reddening) > 0:
        # Add reddening parameters to solution array
        n_reddening = len(reddening)
        sol_extended = np.zeros((ntargets, nlines * 4 + n_reddening))
        esol_extended = np.zeros((ntargets, nlines * 4 + n_reddening))

        sol_extended[:, :nlines*4] = sol
        esol_extended[:, :nlines*4] = esol

        # Fill reddening values (would need to be extracted from pPXF if available)
        # For now, set to zero as pPXF doesn't typically fit reddening
        sol_extended[:, nlines*4:] = 0.0
        esol_extended[:, nlines*4:] = 0.0

        sol = sol_extended
        esol = esol_extended

    # Extract solutions in EMIPPXF format
    sol_emi_F = np.array(sol[:, np.arange(len(idx_l))*4+0])
    sol_emi_A = np.array(sol[:, np.arange(len(idx_l))*4+1])
    sol_emi_V = np.array(sol[:, np.arange(len(idx_l))*4+2])  # Now per-line velocities (NaN if no meaningful flux)
    sol_emi_S = np.array(sol[:, np.arange(len(idx_l))*4+3])  # Now per-line sigmas (NaN if no meaningful flux)
    sol_EBmV_MDEG = np.array(sol[:, len(idx_l)*4:None])

    # Error arrays
    for_errors = configs.get('FOR_ERRORS', True)
    if for_errors:
        esol_emi_F = np.array(esol[:, np.arange(len(idx_l))*4+0])
        esol_emi_A = np.array(esol[:, np.arange(len(idx_l))*4+1])
        esol_emi_V = np.array(esol[:, np.arange(len(idx_l))*4+2])  # Now per-line velocity errors
        esol_emi_S = np.array(esol[:, np.arange(len(idx_l))*4+3])  # Now per-line sigma errors
        esol_EBmV_MDEG = np.array(esol[:, len(idx_l)*4:None])

    # ========================
    # SAVE RESULTS (Main table)
    outfits = os.path.join(outdir, f"{rootname}_emippxf_{LEVEL}.fits")
    ExGalutil.prettyOutput_Running(f"Writing: {rootname}_emippxf_{LEVEL}.fits")

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Extension 1: Table HDU with emission_setup data
    cols = []
    cols.append(fits.Column(name='LINE_ID', format='D', array=np.arange(nlines)))
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
    emission_setup_HDU.name = 'EMIPPXF_EMISSION_SETUP'

    # Extension 2: Table HDU with EMIPPXF output data
    cols = []
    cols.append(fits.Column(name='BIN_ID', format='J', array=ubins))
    cols.append(fits.Column(name='APS_ID', format='J', array=APS_ID))
    cols.append(fits.Column(name='TARGID', format='40A', array=TARGID))
    cols.append(fits.Column(name='CNAME', format='40A', array=CNAME))

    # Add line-by-line results - FOR ALL LINES (fitted and non-fitted) with per-line kinematics
    for iin, ii in enumerate(idx_l):
        nan_array = np.full_like(sol_emi_F[:, iin], np.nan)

        # Main parameters - includes NaN for non-fitted lines, per-line kinematics for fitted lines
        cols.append(fits.Column(name='FLUX'+'_'+names[iin] , format='D',
                               array=np.array(sol_emi_F[:, iin])))
        cols.append(fits.Column(name='AMPL'+'_'+names[iin], format='D',
                               array=np.array(sol_emi_A[:, iin])))

        # Convert per-line velocity to redshift (EMIPPXF format) - handle NaN properly
        # In MOS: each target has its own redshift, so use individual Z values
        with np.errstate(invalid='ignore'):
            Z_LINE = (Z + 1.0) * (1.0 + (np.array(sol_emi_V[:, iin]) / C)) - 1
        cols.append(fits.Column(name='Z'+'_'+names[iin], format='D',
                               array=Z_LINE))

        # Per-line sigma (not global) - will be NaN if flux not meaningful
        cols.append(fits.Column(name='SIGMA'+'_'+names[iin], format='D',
                               array=np.array(sol_emi_S[:, iin])))
        cols.append(fits.Column(name='AON'+'_'+names[iin], format='D',
                               array=np.array(sol_emi_AoN[:, iin])))

        # Equivalent width -- positive for emission, see
        # ExGalPrepare.compute_equivalent_width's docstring for the sign
        # convention and error-propagation caveat.
        cols.append(fits.Column(name='EW'+'_'+names[iin], unit='Angstrom', format='D',
                               array=equivalent_widths_all[:, iin]))
        cols.append(fits.Column(name='ERR_EW'+'_'+names[iin], unit='Angstrom', format='D',
                               array=equivalent_width_errors_all[:, iin]))

        # Error parameters
        if for_errors:
            cols.append(fits.Column(name='ERR_FLUX'+'_'+names[iin], format='D',
                                   array=np.array(esol_emi_F[:, iin])))
            cols.append(fits.Column(name='ERR_AMPL'+'_'+names[iin], format='D',
                                   array=np.array(esol_emi_A[:, iin])))

            # Per-line redshift error calculation (same as EMIPPXF) - handle NaN properly
            # In MOS: each target has its own redshift error
            with np.errstate(invalid='ignore', divide='ignore'):
                ZERR_LINE_p2 = ((ZERR)**2.0) + (1.0/C**2.0) * ((((ZERR)**2.0)/((Z)**2.0)) +
                               (((np.array(esol_emi_V[:, iin]))**2.0)/((np.array(sol_emi_V[:, iin]))**2.0)) +
                               ((np.array(esol_emi_V[:, iin]))**2.0))
                Z_error = np.sqrt(ZERR_LINE_p2)
            cols.append(fits.Column(name='ERR_Z'+'_'+names[iin], format='D',
                                   array=Z_error))

            # Per-line sigma error (not global) - will be NaN if flux not meaningful
            cols.append(fits.Column(name='ERR_SIGMA'+'_'+names[iin], format='D',
                                   array=np.array(esol_emi_S[:, iin])))

            # NEW: Add formal error columns for kinematics in EMIPPXF format
            cols.append(fits.Column(name='FORM_ERR_Z'+'_'+names[iin], format='D',
                                   array=np.array(line_formal_velocity_errors_all[:, iin])))
            cols.append(fits.Column(name='FORM_ERR_SIGMA'+'_'+names[iin], format='D',
                                   array=np.array(line_formal_sigma_errors_all[:, iin])))
        else:
            cols.append(fits.Column(name='ERR_FLUX'+'_'+names[iin], format='D',
                                   array=nan_array))
            cols.append(fits.Column(name='ERR_AMPL'+'_'+names[iin], format='D',
                                   array=nan_array))
            cols.append(fits.Column(name='ERR_Z'+'_'+names[iin], format='D',
                                   array=nan_array))
            cols.append(fits.Column(name='ERR_SIGMA'+'_'+names[iin], format='D',
                                   array=nan_array))
            cols.append(fits.Column(name='FORM_ERR_Z'+'_'+names[iin], format='D',
                                   array=nan_array))
            cols.append(fits.Column(name='FORM_ERR_SIGMA'+'_'+names[iin], format='D',
                                   array=nan_array))

    # Handle reddening columns
    len_EBMV = len(reddening) if reddening is not None and len(reddening) > 0 else 1
    n_targets = len(sol_emi_F[:, 0])  # Number of targets

    if reddening is not None and len(reddening) > 0:
        # Use actual reddening data if available
        cols.append(fits.Column(name='EBMV', format=str(len_EBMV)+'D', array=np.array(sol_EBmV_MDEG)))
        if for_errors:
            cols.append(fits.Column(name='ERR_EBMV', format=str(len_EBMV)+'D', array=np.array(esol_EBmV_MDEG)))
        else:
            nan_array_EBMV = np.full((n_targets, len_EBMV), np.nan)
            cols.append(fits.Column(name='ERR_EBMV', format=str(len_EBMV)+'D', array=nan_array_EBMV))
    else:
        # No reddening data - create NaN arrays with correct shape
        nan_array_EBMV = np.full((n_targets, len_EBMV), np.nan)
        cols.append(fits.Column(name='EBMV', format=str(len_EBMV)+'D', array=nan_array_EBMV))
        cols.append(fits.Column(name='ERR_EBMV', format=str(len_EBMV)+'D', array=nan_array_EBMV))

    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = 'EMIPPXF_TABLE'

    # Add headers (matching EMIPPXF format exactly)
    dataHDU.header['CNF_EMI'] = (configs['CONFIG_FILE'], 'Configs. Filename used by EMIPPXF')
    dataHDU.header['EMI_LEV'] = (LEVEL, 'EMIPPXF working level:  BIN, SPAXEL')
    dataHDU.header['EMI_SIG'] = (configs.get('SIGMA', 150.0), 'Initial guess for SIGMA in EMIPPXF')
    dataHDU.header['EMI_FILE'] = (configs.get('EMI_FILE', ''), 'Emission Lines config. filename')
    dataHDU.header['FOR_ERR'] = (for_errors, 'Derive errors on the emission-line analysis')

    # Add comprehensive line categorization statistics (per-target aware)
    total_fitted = sum(n_fitted_per_target)
    total_masked = sum(n_masked_per_target)
    total_out_range = sum(n_out_range_per_target)
    total_no_wave = sum([len(cat['missing_wavelength']) for cat in target_line_categories])

    dataHDU.header['N_LINES_ALL'] = (len(all_line_names), 'Total lines in emission config')
    dataHDU.header['TOT_FITTED'] = (total_fitted, 'Total line fittings across all targets')
    dataHDU.header['TOT_MASKED'] = (total_masked, 'Total line maskings across all targets')
    dataHDU.header['TOT_OUT_RNG'] = (total_out_range, 'Total lines out-of-range across all targets')
    dataHDU.header['TOT_NO_WAVE'] = (total_no_wave, 'Total lines without wavelengths')

    # Add per-target statistics
    dataHDU.header['FIT_MIN'] = (np.min(n_fitted_per_target), 'Min fitted lines per target')
    dataHDU.header['FIT_MAX'] = (np.max(n_fitted_per_target), 'Max fitted lines per target')
    dataHDU.header['FIT_MEAN'] = (np.mean(n_fitted_per_target), 'Mean fitted lines per target')
    dataHDU.header['OUT_MIN'] = (np.min(n_out_range_per_target), 'Min out-of-range lines per target')
    dataHDU.header['OUT_MAX'] = (np.max(n_out_range_per_target), 'Max out-of-range lines per target')
    dataHDU.header['OUT_MEAN'] = (np.mean(n_out_range_per_target), 'Mean out-of-range lines per target')

    # Add wavelength range information
    dataHDU.header['LMIN_USED'] = (wavelength_range_config[0], 'Minimum wavelength used for fitting')
    dataHDU.header['LMAX_USED'] = (wavelength_range_config[1], 'Maximum wavelength used for fitting')

    # Add individual target redshift information to header
    dataHDU.header['Z_MIN'] = (np.min(Z), 'Minimum redshift among targets')
    dataHDU.header['Z_MAX'] = (np.max(Z), 'Maximum redshift among targets')
    dataHDU.header['Z_MEDIAN'] = (np.median(Z), 'Median redshift among targets')
    dataHDU.header['Z_MEAN'] = (np.mean(Z), 'Mean redshift among targets')
    dataHDU.header['Z_STD'] = (np.std(Z), 'Redshift standard deviation')

    # Add reddening information
    if reddening is not None:
        for redx_i, redx in enumerate(reddening):
            dataHDU.header['REDDEN_%s' % (redx_i+1)] = (redx, 'Reddening by dust in EMIPPXF comp=%s' % (redx_i+1))

    # EBmV_APPLIED (set by aps_mosExGal.py's proc_mosExGaL, see
    # ExGalPrepare.resolve_ebmv_extinction/write_ebmv_header) records the
    # REAL measured E(B-V) actually applied -- 'OFF', a fixed float, or
    # (for auto/per-target mode) the real median SFD98 value across this
    # batch -- as opposed to the raw, unresolved 'EBmV' config string
    # (kept for backward compatibility with older output; historically
    # always "None" since this correction was never wired up before).
    dataHDU.header['EBmV'] = (str(configs.get('EBmV', 'None')), 'Raw EBmV config value (see EBMVAPPL)')
    ExGalPrepare.write_ebmv_header(dataHDU.header, configs)

    if 'LMIN_EMI' in configs:
        dataHDU.header['LW_EMI'] = (configs['LMIN_EMI'], 'Min wavelength (OBS.) used by EMIPPXF')
    if 'LMAX_EMI' in configs:
        dataHDU.header['HW_EMI'] = (configs['LMAX_EMI'], 'Max wavelength (OBS.) used by EMIPPXF')

    dataHDU.header['TEMPL'] = (configs['SSP_LIB'], 'The library of spectral templates')
    dataHDU.header['EMI_V'] = ('pPXF_conversion', 'EMIPPXF version (converted from pPXF)')
    dataHDU.header['APSVERS'] = (APSVERS, 'APS version')
    dataHDU.header['APSEMIV'] = ('pPXF_v1.0', 'PyAPS (MOS) EMIPPXF wrapper version')
    dataHDU.header['CSB_EMI'] = (configs['stitched'], 'Combines Spectral Bands Status for EMIPPXF')

    # Keep basename of input files
    for n_province, province in enumerate(configs['infiles']):
        dataHDU.header['APSREF_%d' % (n_province)] = (os.path.basename(province), 'L1 reference file')

    # Add emission setup info to header - FOR ALL LINES WITH STATUS
    dataHDU.header['NGLLINES'] = (len(idx_l), 'Number of Lines in emission config')
    for nii, ii in enumerate(idx_l):
        # if nii < 40:  # Limit to prevent header overflow
        status = line_status[nii] if nii < len(line_status) else 'UNKNOWN'
        dataHDU.header['GLID_%s' % (nii)] = '%30s' % (nii)
        dataHDU.header['GLI_%s' % (nii)] = '%30s' % (iis[nii])
        dataHDU.header['GLNAM_%s' % (nii)] = '%30s' % (names[nii])
        dataHDU.header['GLLAM_%s' % (nii)] = '%30s' % (lambdas[nii])
        dataHDU.header['GLACT_%s' % (nii)] = '%30s' % (actions[nii])
        dataHDU.header['GLKIN_%s' % (nii)] = '%30s' % (kinds[nii])
        dataHDU.header['GLA_%s' % (nii)] = '%30s' % (aas[nii])
        dataHDU.header['GLV_%s' % (nii)] = '%30s' % (vs[nii])
        dataHDU.header['GLS_%s' % (nii)] = '%30s' % (ss[nii])
        dataHDU.header['GLFIT_%s' % (nii)] = '%30s' % (ffits[nii])
        dataHDU.header['GLAON_%s' % (nii)] = '%30s' % (aons[nii])
        dataHDU.header['GLSTAT_%s' % (nii)] = '%30s' % (status)  # Line status
        # elif nii == 40:
        #     dataHDU.header['COMMENT'] = f'Line info truncated at 40/{len(all_line_names)} lines'

    # UPDATED: Add information about per-line kinematics with flux filtering
    dataHDU.header['PER_LINE_KIN'] = (True, 'Per-line kinematics used (no global V/Sigma)')
    dataHDU.header['FLUX_FILTER'] = (True, 'Kinematics only stored for lines with flux > 0')
    dataHDU.header['COMMENT'] = 'MOS mode: Each target has individual redshift in Z_INPUT column'
    dataHDU.header['COMMENT'] = 'Z_<line> columns computed using individual target redshifts'
    dataHDU.header['COMMENT'] = 'EW_<line> [Angstrom]: FLUX / local stellar continuum, POSITIVE for emission'
    dataHDU.header['COMMENT'] = '(opposite sign convention from the LS module Lick-style absorption indices)'
    dataHDU.header['COMMENT'] = 'ERR_EW_<line>: flux-error-only propagation (continuum treated as fixed)'
    dataHDU.header['COMMENT'] = 'V/SIGMA values are per-line from component tying (no global values)'
    dataHDU.header['COMMENT'] = 'Kinematics (V/SIGMA) only saved for lines with meaningful flux (>0, not NaN)'
    dataHDU.header['COMMENT'] = 'Lines with poor flux have NaN kinematics even if part of tied group'
    dataHDU.header['COMMENT'] = 'For single-component fits: all detected lines have identical kinematics'
    dataHDU.header['COMMENT'] = 'For multi-component fits: V/SIGMA varies by component assignment'

    # Create HDU list and write to file
    HDUList = fits.HDUList([priHDU, emission_setup_HDU, dataHDU])
    HDUList.writeto(outfits, overwrite=True)

    ExGalutil.prettyOutput_Done(f"Writing: {rootname}_emippxf_{LEVEL}.fits")

    logging.info("Wrote: " + outfits)

    # ========================
    # SAVE BESTFIT SPECTRA

    outfits_spec = os.path.join(outdir, f"{rootname}_emippxf_spec_{LEVEL}.fits")
    ExGalutil.prettyOutput_Running(f"Writing: {rootname}_emippxf_spec_{LEVEL}.fits")

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Prepare spectra arrays
    cleaned_spectrum = spectra - emi_bestfit  # Emission-subtracted spectrum
    emissionSubtractedBestfit = bestfit - emi_bestfit  # Stellar continuum only

    cols = []
    cols.append(fits.Column(name='BIN_ID', format='J', array=ubins))
    cols.append(fits.Column(name='APS_ID', format='J', array=APS_ID))
    cols.append(fits.Column(name='TARGID', format='40A', array=TARGID))
    cols.append(fits.Column(name='CNAME', format='40A', array=CNAME))
    cols.append(fits.Column(name='LOGLAM_EMI', unit=wave_unit_str, format=str(npix)+'D', array=logLam))
    cols.append(fits.Column(name='FLUX_EMI', unit=flux_unit_str, format=str(npix)+'D', array=spectra))
    cols.append(fits.Column(name='ERROR_EMI', unit=flux_unit_str, format=str(npix)+'D', array=error))
    cols.append(fits.Column(name='MODEL_EMI', unit=flux_unit_str, format=str(npix)+'D', array=bestfit))
    cols.append(fits.Column(name='EMISSION_EMI', unit=flux_unit_str, format=str(npix)+'D', array=emi_bestfit))
    cols.append(fits.Column(name='FLUX_CLEAN_EMI', unit=flux_unit_str, format=str(npix)+'D', array=cleaned_spectrum))
    cols.append(fits.Column(name='MODEL_CLEAN_EMI', unit=flux_unit_str, format=str(npix)+'D', array=emissionSubtractedBestfit))
    cols.append(fits.Column(name='GOODPIX_EMI', format=str(npix)+'J', array=goodpixels))

    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = 'EMIPPXF_SPEC'
    dataHDU.header['EMI_LEV'] = (LEVEL, 'EMIPPXF working level: 1: BIN, 2: SPAXEL')
    dataHDU.header['CNF_EMI'] = (configs['CONFIG_FILE'], 'Configs. Filename used by EMIPPXF')
    dataHDU.header['SAMPLING'] = (1, 'Sampling mode (0: linear, 1: logarithmic')
    dataHDU.header['TEMPL'] = (configs['SSP_LIB'], 'The library of spectral templates')
    dataHDU.header['EMI_V'] = ('pPXF_conversion', 'EMIPPXF version (converted from pPXF)')
    dataHDU.header['APSVERS'] = (APSVERS, 'APS version')
    dataHDU.header['APSEMIV'] = ('pPXF_v1.0', 'PyAPS (MOS) EMIPPXF wrapper version')
    dataHDU.header['CSB_EMI'] = (configs['stitched'], 'Combines Spectral Bands Status for EMIPPXF')

    # UPDATED: Add per-line kinematics info to spectra header too
    dataHDU.header['PER_LINE_KIN'] = (True, 'Per-line kinematics used (no global V/Sigma)')
    dataHDU.header['FLUX_FILTER'] = (True, 'Kinematics only stored for lines with flux > 0')
    dataHDU.header['COMMENT'] = 'V/SIGMA values in main table are per-line from component tying'
    dataHDU.header['COMMENT'] = 'Kinematics saved only for lines with meaningful flux measurements'

    # Keep basename of input files
    for n_province, province in enumerate(configs['infiles']):
        dataHDU.header['APSREF_%d' % (n_province)] = (os.path.basename(province), 'L1 reference file')

    # Create HDU list and write
    HDUList = fits.HDUList([priHDU, dataHDU])
    HDUList.writeto(outfits_spec, overwrite=True)

    ExGalutil.prettyOutput_Done(f"Writing: {rootname}_emippxf-spectra_{LEVEL}.fits")

    logging.info("Wrote: " + outfits_spec)

    if debug:
        print(f"Enhanced EMIPPXF-compatible output with flux-filtered per-line kinematics:")
        print(f"  Total lines in config: {len(all_line_names)}")
        print(f"  Redshift range: {np.min(Z):.3f} - {np.max(Z):.3f}")
        print(f"  Lines fitted per target: {np.min(n_fitted_per_target)} - {np.max(n_fitted_per_target)} (mean: {np.mean(n_fitted_per_target):.1f})")
        print(f"  Lines out-of-range per target: {np.min(n_out_range_per_target)} - {np.max(n_out_range_per_target)} (mean: {np.mean(n_out_range_per_target):.1f})")
        print(f"  Total line fittings across all targets: {total_fitted}")
        print(f"  Total lines out-of-range across all targets: {total_out_range}")
        print(f"  Using per-line kinematics with flux filtering (kinematics only for flux > 0)")
        print(f"  Files created: _emippxf_1.fits, _emippxf_spec_1.fits")

    return True






def workerPPXF_emi_mos(inQueue, outQueue):
    """
    Worker process for emi kinematics analysis - MOS version.
    Following PPXF stellar pattern exactly.

    Defines the worker process of the parallelisation with multiprocessing.Queue
    and multiprocessing.Process for emi kinematics analysis.
    """
    for outdir, config_dir, templates_dir, logLam, bin_data, bin_error, configs, info_tab, velscale, \
        LSF_Templates, error_limit, nsims, ntargets, i, emission_setup, \
        stellar_bestfit, start, line_names, tie_settings, line_wavelengths, figdir, diag_plots, group_order \
        in iter(inQueue.get, 'STOP'):


        try:
            # Call the serial worker function - same pattern as PPXF stellar
            result = workerPPXF_emi_serial(outdir, config_dir, templates_dir, logLam, bin_data, bin_error,
                                          configs, info_tab, velscale, LSF_Templates,
                                          error_limit, nsims, ntargets, i, emission_setup, stellar_bestfit, start,
                                          line_names, tie_settings, line_wavelengths, figdir, diag_plots, group_order)

            # Put successful result
            outQueue.put(result)

        except Exception as e:
            print(f"Error in worker for target {i}: {str(e)}")
            import traceback
            traceback.print_exc()

            # Create failed result structure - same as PPXF stellar pattern
            # Determine dimensions for failed results
            try:
                npix = len(bin_data) if 'bin_data' in locals() else 1000  # fallback
                nlines = len(line_names) if 'line_names' in locals() else 10  # fallback
                nmoments_emi = emission_setup.get('emi_moments', 2) if 'emission_setup' in locals() else 2
                nmoments_stellar = configs.get('MOM', 4) if 'configs' in locals() else 4

                # Get target metadata for the failed case - SAME AS PPXF STELLAR
                binid = np.ravel(np.where(info_tab['BIN_ID'] == i))[0]
                emi_meta_failed = {}
                emi_meta_failed['APS_ID'] = info_tab['APS_ID'][binid]
                emi_meta_failed['BIN_ID'] = info_tab['BIN_ID'][binid]
                emi_meta_failed['TARGID'] = info_tab['TARGID'][binid]
                emi_meta_failed['CNAME'] = info_tab['CNAME'][binid]
                emi_meta_failed['Z'] = info_tab['Z'][binid]
                emi_meta_failed['ZERR'] = info_tab['ZERR'][binid]


                # Create failed result structure matching expected output format
                failed_result = (
                    i,  # target index
                    {"stellar": np.full(nmoments_stellar, np.nan), "emi": np.full(nmoments_emi, np.nan)},  # emi_result
                    np.full(npix, np.nan),  # emi_bestfit
                    np.full(npix, np.nan),  # emi_bestfit (emission component)
                    np.full(nlines, np.nan),  # emission_lines
                    {"stellar": np.full(nmoments_stellar, np.nan), "emi": np.full(nmoments_emi, np.nan),
                     "fluxes": np.full(nlines, np.nan), "amplitudes": np.full(nlines, np.nan),
                     "amplitude_errors": np.full(nlines, np.nan), "aon": np.full(nlines, np.nan)},  # mc_results
                    {"stellar": np.full(nmoments_stellar, np.nan), "emi": np.full(nmoments_emi, np.nan)},  # formal_error
                    np.array([]),  # goodpixels
                    emi_meta_failed  # emi_meta
                )

                outQueue.put(failed_result)

            except Exception as e2:
                # If even creating dummy results fails, put minimal result
                print(f"Critical error creating dummy results: {str(e2)}")
                minimal_result = (i, {}, np.array([]), np.array([]), np.array([]), {}, {}, np.array([]),
                                {'APS_ID': -1, 'BIN_ID': -1, 'TARGID': 'FAILED', 'CNAME': 'FAILED', 'Z': 0.0, 'ZERR': 0.0})
                outQueue.put(minimal_result)



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



def PPXF_spectralMasking_MOS(config_dir, configs, logLam, module, redshift, LSF_Data=None, verbose=False):
    """
    Construct a spectral mask with LSF-dependent widths, according to the information
    provided in the file spectralMasking_[module].config.

    Now handles logLam with NaN padding properly and includes LMIN/LMAX handling.
    """

    # Read file
    mask_file = config_dir + "spectralMasking_" + module + ".config"

    try:
        mask = np.genfromtxt(mask_file, usecols=(0,1))
        maskComment = np.genfromtxt(mask_file, usecols=(2), dtype=str)
    except Exception as e:
        if verbose:
            print(f"Warning: Could not read masking file {mask_file}: {e}")
        return np.where(np.isfinite(logLam))[0]  # Only return finite pixels as good

    # Handle finite part of logLam
    finite_indices = np.where(np.isfinite(logLam))[0]
    if len(finite_indices) < 2:
        raise ValueError("logLam must contain at least two finite values for masking.")
    l0_gal = logLam[finite_indices[0]]
    lstep_gal = logLam[finite_indices[1]] - logLam[finite_indices[0]]

    npix = len(logLam)
    goodMask = np.ones(npix, dtype=bool)
    goodMask[np.isnan(logLam)] = False  # NaNs are not good pixels

    # Apply LMIN/LMAX wavelength range limits - SAME AS PPXF STELLAR
    if 'LMIN_EMI' in configs and 'LMAX_EMI' in configs:
        lmin_emi = configs['LMIN_EMI']
        lmax_emi = configs['LMAX_EMI']

        pix0 = int(np.ceil((np.log(lmin_emi) - l0_gal) / lstep_gal))
        pix1 = int(np.ceil((np.log(lmax_emi) - l0_gal) / lstep_gal))

        # Clamp to valid range
        pix0 = max(0, pix0)
        pix1 = min(npix, pix1)

        # Apply wavelength range mask
        wavelength_mask = np.zeros(npix, dtype=bool)
        if pix0 < pix1:
            wavelength_mask[pix0:pix1] = True
            goodMask = goodMask & wavelength_mask

        if verbose:
            print(f"Applied emi wavelength range {lmin_emi:.1f}-{lmax_emi:.1f}Å: pixels {pix0}-{pix1}")

    # In case only one mask
    if len(mask.shape) == 1 and mask.shape[0] != 0:
        mask = mask.reshape(1, 2)
        if isinstance(maskComment, str):
            maskComment = np.array([maskComment])
        else:
            maskComment = maskComment.reshape(1)

    if verbose:
        print(f"Processing {len(mask)} mask regions from {mask_file}")
        if redshift != 0.0:
            print(f"Warning: Using redshift {redshift:.4f} - expected 0.0 for rest-frame data")
        if LSF_Data is not None:
            print("Using LSF-dependent masking widths")

    lsf_enhanced_regions = 0
    lsf_failed_regions = 0

    for i in range(mask.shape[0]):
        central_wavelength = mask[i, 0]
        base_width = mask[i, 1]
        comment = str(maskComment[i]).upper() if i < len(maskComment) else ""

        if 'SKY' in comment and redshift != 0.0:
            corrected_wavelength = central_wavelength / (1 + redshift)
        else:
            corrected_wavelength = central_wavelength

        # Compute effective width
        if LSF_Data is not None:
            try:
                lsf_fwhm = LSF_Data(corrected_wavelength)
                total_width = np.sqrt(base_width**2 + lsf_fwhm**2)
                effective_width = total_width * 2.5
                lsf_enhanced_regions += 1
            except Exception as e:
                effective_width = base_width * 2.5
                lsf_failed_regions += 1
                if verbose and lsf_failed_regions <= 3:
                    print(f"Warning: LSF failed for region {i+1} ({corrected_wavelength:.1f}Å): {e}")
        else:
            effective_width = base_width * 2.0

        # Convert wavelength limits to pixel range
        try:
            log_wave_min = np.log(corrected_wavelength - effective_width/2.)
            log_wave_max = np.log(corrected_wavelength + effective_width/2.)
        except ValueError:
            continue  # Skip invalid region

        pix_min = int(np.round((log_wave_min - l0_gal) / lstep_gal))
        pix_max = int(np.round((log_wave_max - l0_gal) / lstep_gal))

        # Clip to within bounds
        pix_min = max(0, pix_min)
        pix_max = min(npix - 1, pix_max)

        if pix_min <= pix_max:
            goodMask[pix_min:pix_max+1] = False
            if verbose and i < 5:
                print(f"Masked region {i+1}: {pix_min}–{pix_max} ({pix_max - pix_min + 1} px) for {comment} at {corrected_wavelength:.1f}Å ±{effective_width/2:.1f}Å")

    if verbose and LSF_Data is not None:
        total_regions = len(mask)
        print("\nLSF masking summary:")
        print(f"  Total mask regions: {total_regions}")
        print(f"  LSF-enhanced: {lsf_enhanced_regions}")
        print(f"  LSF failed: {lsf_failed_regions}")
        if total_regions > 0:
            print(f"  Success rate: {lsf_enhanced_regions/total_regions*100:.1f}%")

    goodPixels = np.where(goodMask)[0]

    if verbose:
        n_masked_total = npix - len(goodPixels)
        print(f"Final masking result: {n_masked_total} pixels masked, {len(goodPixels)} pixels remaining")

    return goodPixels



def workerPPXF_emi_serial(outdir, config_dir, templates_dir, logLam, bin_data, bin_error, configs, info_tab,
                         velscale, LSF_Templates, error_limit, nsims, ntargets, i,
                         emission_setup, stellar_bestfit, start, line_names, tie_settings, line_wavelengths,
                         figdir, diag_plots, group_order):
    """
    Worker function for emi analysis - following PPXF stellar pattern.
    """
    # Get emission config from configs (assume it's loaded in main function)
    emission_config_file = os.path.join(config_dir, configs.get('EMI_FILE'))

    # Import the config loading function
    from . import IFUExGalEMIPPXF
    emission_config = IFUExGalEMIPPXF.load_ppxf_emission_config(emission_config_file)
    # Prepare emi analysis for this target - SAME PATTERN AS PPXF STELLAR
    emi_meta, LSF_Data, templates, emi_template_filtered, emi_template_unbroadened_filtered, line_names_filtered, \
    offset, goodpixels_ppxf, optimal_template, logLam_template, velscale_ratio, scaling_factor = \
    preparePPXF_emi_mos(i, info_tab, configs, outdir, config_dir, templates_dir,
                       bin_data, bin_error, logLam, velscale,
                       LSF_Templates, error_limit, emission_config, line_names, line_wavelengths)


    # Run emi fitting
    sol, bestfit, emi_bestfit, emission_lines, mc_results, formal_error = \
    run_ppxf_emi_mos(templates, emi_template_filtered, emi_template_unbroadened_filtered, bin_data, bin_error, velscale,
                     goodpixels_ppxf, configs['MOM'], configs['ADEG'], configs['MDEG'],
                     offset, start, velscale_ratio, nsims, ntargets, logLam, logLam_template, i,
                     emission_setup, stellar_bestfit, line_names_filtered, tie_settings,
                     line_wavelengths, group_order, scaling_factor=scaling_factor)
    return i, sol, bestfit, emi_bestfit, emission_lines, mc_results, formal_error, goodpixels_ppxf, emi_meta, line_names_filtered





def preparePPXF_emi_mos(index, info_tab, configs, outdir, config_dir, templates_dir, bin_data, bin_error,
                       logLam, velscale, LSF_Templates, error_limit, emission_config,
                       line_names, line_wavelengths):
    """
    Prepare PPXF emi analysis - FOLLOWING PPXF STELLAR PATTERN EXACTLY.
    """

    # Get target metadata - SAME AS PPXF STELLAR
    binid = np.ravel(np.where(info_tab['BIN_ID'] == index))[0]
    emi_meta = {}
    emi_meta['APS_ID'] = info_tab['APS_ID'][binid]
    emi_meta['BIN_ID'] = info_tab['BIN_ID'][binid]
    emi_meta['TARGID'] = info_tab['TARGID'][binid]
    emi_meta['CNAME'] = info_tab['CNAME'][binid]
    emi_meta['Z'] = info_tab['Z'][binid]
    emi_meta['ZERR'] = info_tab['ZERR'][binid]

    try:
        # # Read LSF - SAME AS PPXF STELLAR
        # LSF = np.genfromtxt(config_dir+'LSF-Config_'+configs['SETMODE'], comments='#')
        # LSF[:,0] = LSF[:,0] / (1 + emi_meta['Z'])
        # LSF[:,1] = LSF[:,1] / (1 + emi_meta['Z'])
        # LSF_Data = interpolate.interp1d(LSF[:,0], LSF[:,1], 'linear', fill_value='extrapolate')

        # new method with reading LSF from the L1 calibrations
        lsf_data = MOSExGalPrepare.load_lsf_data(configs)
        lsf_indx = np.where(lsf_data['aps_id'] == emi_meta['APS_ID'])[0]
        lsf_func = lsf_data['lsf'][lsf_indx][0]
        LSF_Data = apply_redshift_to_fwhm_corrected(lsf_func, emi_meta['Z'])


        velscale_ratio, template_velscale, data_velscale = MOSExGalPrepare.calculate_velscale_ratio(configs, logLam, velscale, LSF_Data, LSF_Templates, templates_dir, debug=False)

        # APPLY MASKING - KEY DIFFERENCE: Handle emi wavelength range in masking
        # This replaces all the padding logic!
        if emission_config is not None:
            goodpixels_ppxf = apply_ppxf_emission_masking(
                logLam=logLam,
                velscale=velscale,
                emission_config=emission_config,
                error_spectrum=bin_error,
                error_limit=error_limit,
                configs=configs,     # This will handle LMIN_EMI/LMAX_EMI inside masking
                verbose=False,
                LSF_Data=LSF_Data,
                redshift=emi_meta['Z']
            )
        else:
            # Fallback masking that handles LMIN_EMI/LMAX_EMI
            goodpixels_ppxf = PPXF_spectralMasking_MOS(config_dir, configs, logLam, 'PPXF', emi_meta['Z'], LSF_Data=LSF_Data, verbose=False)
            goodpixels_ppxf = PPXF_MaskGaps_MOS(bin_error, goodpixels_ppxf, error_limit)

        assert len(goodpixels_ppxf) > 0, 'ERROR: null spectrum after masking for emi analysis'

        # Prepare templates - SAME PATTERN AS PPXF STELLAR
        lmin = np.min(np.exp(logLam))
        lmax = np.max(np.exp(logLam))

        templates, lamRange_spmod, logLam_template, ntemplates = MOSExGalPrepare.prepareSpectralTemplateLibrary(
            "PPXF", templates_dir, configs, lmin, lmax, velscale, velscale_ratio, LSF_Data, LSF_Templates)[:4]

        # Prepare emi templates
        from . import IFUExGalEMIPPXF
        emi_template, emi_template_unbroadened,  logLam_emi_template = IFUExGalEMIPPXF.prepare_emission_template(
            logLam_template, line_names, line_wavelengths,
            LSF_Templates, LSF_Data, wl_offset=0.0, debug=False)

        # Filter and scale templates
        non_zero_mask = np.max(np.abs(emi_template), axis=0) > 1e-10
        emi_template_filtered = emi_template[:, non_zero_mask]
        emi_template_unbroadened_filtered = emi_template_unbroadened[:, non_zero_mask]
        line_names_filtered = [line_names[i] for i in range(len(line_names)) if non_zero_mask[i]]

        stellar_scale = np.max(np.abs(templates))
        emi_scale = np.max(np.abs(emi_template_filtered))
        target_scale = stellar_scale * 0.1
        scaling_factor = target_scale / emi_scale
        emi_template_filtered *= scaling_factor
        emi_template_unbroadened_filtered *= scaling_factor

        offset = (logLam_template[0] - logLam[0]) * C
        optimal_template = np.zeros(templates.shape[0])

    except Exception as e:
        print(f"Error in preparePPXF_emi_mos for target {emi_meta['APS_ID']}: {str(e)}")
        # Return NaN values like PPXF stellar does
        LSF_Data = np.nan
        templates = np.nan
        emi_template_filtered = np.nan
        emi_template_unbroadened_filtered = np.nan
        line_names_filtered = []
        lamRange_spmod = np.nan
        logLam_template = np.nan
        ntemplates = 0
        offset = np.nan
        optimal_template = np.nan
        velscale_ratio = np.nan
        scaling_factor = 1.0

    return (emi_meta, LSF_Data, templates, emi_template_filtered, emi_template_unbroadened_filtered, line_names_filtered,
            offset, goodpixels_ppxf, optimal_template, logLam_template, velscale_ratio, scaling_factor)


def get_emi_bounds(emi_start_vals, stellar_present=False):
    """Get proper bounds for emi kinematics to prevent extreme solutions"""
    bounds = []

    if len(emi_start_vals) >= 1:
        # Velocity bounds: ±300 km/s from start
        v_start = emi_start_vals[0]
        # Validate v_start
        if np.isnan(v_start) or np.isinf(v_start):
            v_start = 0.0
        bounds.append([v_start - 300, v_start + 300])

    if len(emi_start_vals) >= 2:
        # CRITICAL: Dispersion bounds to prevent doubling
        sigma_start = emi_start_vals[1]

        # Validate sigma_start
        if sigma_start <= 0 or np.isnan(sigma_start) or np.isinf(sigma_start):
            sigma_start = 150.0

        if stellar_present:
            # Allow 0.5x to 2.5x of stellar dispersion, but at least 25 km/s
            sigma_min = max(25.0, sigma_start * 0.5)
            sigma_max = max(sigma_start * 2.5, 400.0)  # Changed from min() to max()
        else:
            # For emi-only fits, reasonable range
            sigma_min = max(25.0, sigma_start * 0.7)
            sigma_max = max(sigma_start * 2.0, 400.0)  # Changed from min() to max()

        # Ensure valid bounds
        if sigma_min >= sigma_max:
            sigma_min = 25.0
            sigma_max = 400.0

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
        bias_factor = 0.01  # Light regularization when stellar is present
    else:
        bias_factor = 0.02  # Stronger regularization for emi-only

    return max(0.001, data_std * bias_factor)


def _emi_fit_nan_result(nmoments, emission_setup, line_names, log_bin_data):
    """The NaN-filled failure result for one target's emi fit -- same
    shape run_ppxf_emi_mos returns for any other failure, so a caller
    (the n_failed count in runModule_EMIPPXF, save_emi_kinematics_mos,
    etc.) can't tell "prep already failed upstream" apart from "the fit
    itself raised" and doesn't need to. Factored out so the two places
    that need it (the early short-circuit below, and this function's own
    except-clause) can't drift out of sync with each other."""
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
             "fluxes": np.full(n_lines, np.nan), "amplitudes": np.full(n_lines, np.nan)})


def run_ppxf_emi_mos(templates, emi_template, emi_template_unbroadened, log_bin_data, log_bin_error, velscale, goodpixels,
                    nmoments, adeg, mdeg, offset, start, velscale_ratio, nsims, ntargets, logLam, logLam_template, i,
                    emission_setup, stellar_bestfit, line_names, tie_settings, line_wavelengths, group_order, scaling_factor=1.0):
    """
    FIXED VERSION: Properly handles multi-component emi kinematics.
    Now extracts V and Sigma for each emi component/line, not just the first component.
    """
    ExGalutil.printProgress(i, ntargets, barLength=50)

    # preparePPXF_emi_mos's own except-clause returns np.nan sentinels for
    # templates/emi_template (etc.) when template prep itself failed for
    # this target -- matching the PPXF-stellar convention, per that
    # function's docstring. That failure is ALREADY reported (one line,
    # from preparePPXF_emi_mos) by the time we get here, so short-circuit
    # straight to the same NaN-filled result this function's own
    # except-clause below would produce, rather than proceeding into real
    # array/ppxf code and hitting a confusing low-level AttributeError
    # (e.g. templates.shape / emi_template.shape on a float) for an
    # already-known, already-explained failure.
    if not isinstance(templates, np.ndarray) or not isinstance(emi_template, np.ndarray):
        print(f"INFO: target {i}: skipping emi fit -- template preparation already failed "
              f"for this target (see preparePPXF_emi_mos's message above)")
        return _emi_fit_nan_result(nmoments, emission_setup, line_names, log_bin_data)

    # Import the necessary functions from IFU version (reuse the logic)
    from . import IFUExGalEMIPPXF  # Assuming this contains the emi fitting functions

    # Use the same tie settings and component building logic as IFU
    emi_component = IFUExGalEMIPPXF.build_emi_component_tying(line_names, tie_settings, group_order=group_order, debug=False)

    try:
        # Create start array from stellar kinematics or defaults
        if start is not None:
            emi_start = start.copy()
        else:
            start = np.zeros(max(nmoments, 2))
            start[0] = 0.0  # Default velocity
            start[1] = 150.0  # Default sigma
            emi_start = start.copy()

        # PPXF emi fit - includes both stellar and emission template
        component = np.zeros(templates.shape[1] + emi_template.shape[1], dtype=int)
        component[templates.shape[1]:] = 1  # Mark emi template with component=1

        # Setup the emi fitting parameters
        moments = [nmoments, emission_setup.get('emi_moments', 2)]  # emi usually has fewer moments

        # Convert to linear wavelength
        lam_emi = np.exp(logLam)
        lam_emi_template = np.exp(logLam_template)

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
                    print(f"Warning: {n_bad} non-finite pixels in continuum-subtracted spectrum for target {i}")

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
                        print(f"Warning: Only {len(goodpixels)} good pixels remaining after cleaning")

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
                        print(f"Warning: Invalid velocity bounds: {emi_bounds[0]}")
                        emi_bounds[0] = [-500, 500]
                    # Validate sigma bounds
                    if emi_bounds[1][0] >= emi_bounds[1][1]:
                        print(f"Warning: Invalid sigma bounds: {emi_bounds[1]}")
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

                    if len(goodpixels) == 0:
                        raise ValueError(f"No good pixels remaining after continuum cleaning for target {i}")

                    pp = ppxf(emi_template_unbroadened, continuum_subtracted, log_bin_error, velscale,
                                start=emi_start_multi, goodpixels=goodpixels, plot=False,
                                quiet=True, moments=emi_moments_multi,
                                bounds=multi_bounds,
                                degree=-1, mdegree=-1,
                                velscale_ratio=velscale_ratio, lam=lam_emi, lam_temp=lam_emi_template,
                                component=emi_component,
                                bias=bias_val)
                else:
                    if len(goodpixels) == 0:
                        raise ValueError(f"No good pixels remaining after continuum cleaning for target {i}")

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
                    # Multi-component case: pp.sol is a list of [stellar_sol, emi_comp1_sol, emi_comp2_sol, ...]
                    emi_solutions = pp.sol  # All emi component solutions
                    n_emi_components = len(emi_solutions)
                else:
                    # Single component case
                    emi_solutions = [pp.sol] if hasattr(pp, 'sol') else [np.zeros(emission_setup.get('emi_moments', 2))]
                    n_emi_components = 1

                # set the bestfit to zero for all bad pixels
                pp.bestfit[np.setdiff1d(np.arange(len(pp.bestfit)), goodpixels)] = 0.0
                combined_bestfit = stellar_bestfit + pp.bestfit
                emi_template_weighted = pp.bestfit

            else:
                # Option 2: Re-fit both stellar and emi simultaneously
                all_template = np.column_stack([templates, emi_template])

                # Validate input data before fitting
                if not np.all(np.isfinite(log_bin_data)):
                    log_bin_data = np.nan_to_num(log_bin_data, nan=0.0, posinf=0.0, neginf=0.0)
                if not np.all(np.isfinite(log_bin_error)):
                    log_bin_error = np.nan_to_num(log_bin_error, nan=1e10, posinf=1e10, neginf=1e10)

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
                        np.zeros(templates.shape[1], dtype=int),
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
                            print(f"Warning: Invalid velocity bounds in simultaneous fit: {emi_bounds[0]}")
                            emi_bounds[0] = [-500, 500]
                        if emi_bounds[1][0] >= emi_bounds[1][1]:
                            print(f"Warning: Invalid sigma bounds in simultaneous fit: {emi_bounds[1]}")
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
                    emi_weights = weights[templates.shape[1]:]
                    emi_template_weighted = np.zeros_like(log_bin_data)

                    for j in range(emi_template.shape[1]):
                        if j < len(emi_weights):
                            emi_template_weighted += emi_template[:, j] * emi_weights[j]
                else:
                    emi_template_weighted = np.zeros_like(log_bin_data)
        else:
            # No stellar continuum available - fit emi only

            # Validate input data before emi-only fit
            if not np.all(np.isfinite(log_bin_data)):
                log_bin_data = np.nan_to_num(log_bin_data, nan=0.0, posinf=0.0, neginf=0.0)
            if not np.all(np.isfinite(log_bin_error)):
                log_bin_error = np.nan_to_num(log_bin_error, nan=1e10, posinf=1e10, neginf=1e10)

            emi_start_simple = emi_start.copy() if len(emi_start) >= 2 else [0.0, 150.0]
            if len(emi_start_simple) > emission_setup.get('emi_moments', 2):
                emi_start_simple = emi_start_simple[:emission_setup.get('emi_moments', 2)]

            # Get bounds and bias for emi-only fit
            emi_bounds = get_emi_bounds(emi_start_simple, stellar_present=False)

            # Validate emi-only bounds
            if emi_bounds and len(emi_bounds) >= 2:
                if emi_bounds[0][0] >= emi_bounds[0][1]:
                    print(f"Warning: Invalid velocity bounds in emi-only fit: {emi_bounds[0]}")
                    emi_bounds[0] = [-500, 500]
                if emi_bounds[1][0] >= emi_bounds[1][1]:
                    print(f"Warning: Invalid sigma bounds in emi-only fit: {emi_bounds[1]}")
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
                    # Validate each component's bounds
                    for comp_idx, comp_bounds in enumerate(multi_bounds):
                        if len(comp_bounds) >= 2:
                            if comp_bounds[0][0] >= comp_bounds[0][1]:
                                comp_bounds[0] = [-500, 500]
                            if comp_bounds[1][0] >= comp_bounds[1][1]:
                                comp_bounds[1] = [25, 400]


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
                emi_weights = pp.weights[templates.shape[1]:]

            for j, name in enumerate(line_names):
                if j < len(emi_weights):
                    emission_fluxes[j] = emi_weights[j] / scaling_factor

                    if j < emi_template.shape[1]:
                        weighted_template = emi_template[:, j] * emi_weights[j]
                        emission_amplitudes[j] = np.max(weighted_template) / scaling_factor

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
                        mc_emi_weights = mc.weights[templates.shape[1]:]

                    # Store MC flux and amplitude values
                    for j, name in enumerate(line_names):
                        if j < len(mc_emi_weights):
                            # Store MC flux value
                            emission_fluxes_MC[o, j] = mc_emi_weights[j] / scaling_factor

                            # Calculate MC amplitude from weighted template
                            if j < emi_template.shape[1]:
                                mc_weighted_template = emi_template[:, j] * mc_emi_weights[j]
                                emission_amplitudes_MC[o, j] = np.max(mc_weighted_template) / scaling_factor

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
                emi_weights_errors = weights_formal_errors[templates.shape[1]:]

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
                formal_error)

    except Exception as e:
        # "No good pixels remaining after continuum cleaning" is a known,
        # self-descriptive data-quality condition (this target's spectrum
        # has nothing usable left after masking/continuum subtraction) --
        # already raised with a clear message at the point it's detected,
        # so one line here is enough; a full traceback through ppxf/capfit
        # internals adds nothing and, at the failure rates seen on some
        # fields (dozens of targets per job), drowns out logs that
        # genuinely need the traceback. Anything else is unexpected and
        # keeps the full traceback for debugging.
        if isinstance(e, ValueError) and "No good pixels remaining" in str(e):
            print(f"INFO: target {i}: {e}")
        else:
            print(f"Error in emi fitting for target {i}: {str(e)}")
            traceback.print_exc()
        return _emi_fit_nan_result(nmoments, emission_setup, line_names, log_bin_data)



def runModule_EMIPPXF(nthreads, configs, velscale, LSF_Templates, outdir, config_dir,
                               templates_dir, figdir, rootname, error_limit,
                               tie_mode='optimised', debug=False, diag_plots=False, LEVEL="BIN"):
    """
    Starts the analysis of the emi kinematics for MOS data - following PPXF stellar approach
    with proper stellar continuum handling for different wavelength ranges.

    Parameters:
    -----------
    tie_mode : str
        Tie settings mode: 'optimised', 'aggressive', 'legacy', 'default', or 'custom'
    """
    print("\033[0;37m" + " - - - - - Starting EMI KINEMATICS (MOS) - - - - -" + "\033[0;39m")
    logging.info(" - - - Starting EMI KINEMATICS (MOS) - - - ")







    # Read data from file - SAME AS PPXF STELLAR
    hdu_t = fits.open(outdir + rootname + '_table.fits')
    info_tab = hdu_t[1].data
    hdu_s = fits.open(outdir + rootname + '_BINSpectra.fits')
    bin_data = np.array(hdu_s[1].data.SPEC)      # Use original arrays directly
    bin_error = np.array(hdu_s[1].data.ESPEC)    # Use original arrays directly
    logLam = np.array(hdu_s[1].data.LOGLAM)      # Use original arrays directly

    ntargets = bin_data.shape[0]
    ubins = np.arange(0, ntargets)
    npix = bin_data.shape[1]                     # Use original npix

    nsims = configs.get('MC_EMI', configs.get('MC_PPXF', 30))

    # Load stellar kinematics results to use as input
    stellar_available = True
    ppxf_stellar = None
    stellar_bestfit_original = None
    stellar_velocities = None
    stellar_sigmas = None
    try:
        hdu_ppxf = fits.open(outdir + rootname + '_ppxf.fits')
        ppxf_stellar = hdu_ppxf[1].data
        hdu_ppxf.close()

        hdu_ppxf_spec = fits.open(outdir + rootname + '_ppxf_spec.fits')
        stellar_bestfit_original = np.array(hdu_ppxf_spec[1].data.MODEL_PPXF)  # Use as-is
        hdu_ppxf_spec.close()

        # Extract stellar kinematics for initial conditions
        stellar_velocities = ppxf_stellar['V']
        stellar_sigmas = ppxf_stellar['SIGMA']
        if debug:
            print(f"Loaded stellar kinematics for {len(ppxf_stellar)} targets")
            print(f"Stellar V range: {np.min(stellar_velocities):.1f} - {np.max(stellar_velocities):.1f} km/s")
            print(f"Stellar SIGMA range: {np.min(stellar_sigmas):.1f} - {np.max(stellar_sigmas):.1f} km/s")

    except Exception as e:
        logging.error(f"Failed to load stellar kinematics results: {str(e)}")
        print(f"Error loading stellar kinematics: {str(e)}")
        print("Proceeding without stellar kinematics input")
        stellar_available = False
        stellar_bestfit_original = None

    # Load emission line configuration
    emission_config = None
    emission_config_file = os.path.join(config_dir, configs.get('EMI_FILE'))
    ExGalutil.prettyOutput_Info(f"Loading emission line configuration from: {emission_config_file}")

    # Import the config loading function from IFU version
    from . import IFUExGalEMIPPXF
    emission_config = IFUExGalEMIPPXF.load_ppxf_emission_config(emission_config_file)

    # Use emission lines from config
    line_wavelengths = emission_config['emission_lines']
    line_names = list(line_wavelengths.keys())
    line_groups = emission_config['line_groups']

    # SELECT TIE SETTINGS BASED ON MODE
    if tie_mode == 'optimised':
        tie_settings = emission_config['tie_settings_optimised']
        ExGalutil.prettyOutput_Info(f"Using OPTIMISED tie settings: ~9 components, 90% computational reduction")

    elif tie_mode == 'aggressive':
        tie_settings = emission_config['tie_settings_aggressive']
        ExGalutil.prettyOutput_Info(f"Using AGGRESSIVE tie settings: ~5 components, 94% computational reduction")

    elif tie_mode == 'legacy':
        tie_settings = emission_config['tie_settings_legacy']
        ExGalutil.prettyOutput_Info(f"Using LEGACY tie settings: ~70 components, 30% computational reduction")

    elif tie_mode == 'default':
        tie_settings = {
            "balmer": [
                "H12_3749.93","H11_3770.93","H10_3797.92","H9_3835.91",
                "H5_3889.05","H8_3888.90","He_3970.07","Hd_4101.73",
                "Hg_4340.46","Hbeta_4861.32","Halpha_6562.80"
            ],
            "forbidden": [
                "[OIII]_4363.15", "[OIII]_4363.21", "[OIII]_4931.23",
                "[NII]_5754.40", "[SII]_4071.15", "[OII]_3726.03",
                "[OII]_3728.73", "[OII]_7319.46", "[OII]_7329.98"
            ],
            '[OIII]_5006.77': [
                '[OIII]_4958.83', '[NI]_5197.90', '[NI]_5200.39'
            ],
            '[NII]_6583.34': [
                'HeI_5875.60', '[OI]_6300.20', '[OI]_6363.67',
                '[NII]_6547.96', '[SII]_6716.31', '[SII]_6730.68',
                '[ArIII]_7135.67'
            ],
            'tie_all': False
        }
        print(f"Using DEFAULT tie settings for backward compatibility")

    elif tie_mode == 'custom':
        tie_settings = emission_config.get('tie_settings', emission_config['tie_settings'])
        print(f"Using CUSTOM tie settings from emission config")

    else:
        raise ValueError(f"Unknown tie_mode: {tie_mode}. Use 'optimised', 'aggressive', 'legacy', 'default', or 'custom'")

    ExGalutil.prettyOutput_Info(f"Selected tie mode '{tie_mode}':")

    group_order = IFUExGalEMIPPXF.extract_group_order_from_tie_settings(tie_settings)
    ExGalutil.prettyOutput_Info(f" Using group order: {group_order}")

    # Build emi component tying
    emi_component = IFUExGalEMIPPXF.build_emi_component_tying(line_names, tie_settings, group_order=group_order, debug=False)

    # Emission line setup
    emission_setup = {
        'emi_moments': configs.get('EMI_MOM', 2),
        'use_stellar_continuum': configs.get('USE_STELLAR_CONTINUUM', True),
        'fix_stellar_continuum': configs.get('FIX_STELLAR_CONTINUUM', False)
    }

    # Check if emi wavelength range differs from stellar - SAME AS IFU
    emi_range_differs = False
    if ('LMIN_EMI' in configs and 'LMAX_EMI' in configs and
        'LMIN_PPXF' in configs and 'LMAX_PPXF' in configs):
        if (configs['LMIN_EMI'] != configs['LMIN_PPXF'] or
            configs['LMAX_EMI'] != configs['LMAX_PPXF']):
            emi_range_differs = True
            ExGalutil.prettyOutput_Warning(f"emi wavelength range ({configs['LMIN_EMI']}-{configs['LMAX_EMI']} Å) "
                  f"differs from stellar range ({configs['LMIN_PPXF']}-{configs['LMAX_PPXF']} Å)")

    # PREPARE STELLAR FIT FOR EMI RANGE - SAME AS IFU BUT ADAPTED FOR MOS
    stellar_fit_for_emi = None

    if emi_range_differs and stellar_available:
        ExGalutil.prettyOutput_Running("Preparing fixed stellar fit for emi wavelength range...")
        # First need to prepare templates for emi range
        # Use same approach as individual target preparation but for all targets
        if debug:
            print("Loading templates for emi wavelength range...")

        # Use first target as reference for wavelength range
        lmin = np.nanmin(np.exp(logLam))
        lmax = np.nanmax(np.exp(logLam))

        # Read LSF for first target as representative
        LSF = np.genfromtxt(config_dir+'LSF-Config_'+configs['SETMODE'], comments='#')
        z_ref = info_tab['Z'][0]  # Use first target's redshift as reference
        LSF[:,0] = LSF[:,0] / (1 + z_ref)
        LSF[:,1] = LSF[:,1] / (1 + z_ref)
        LSF_Data = interpolate.interp1d(LSF[:,0], LSF[:,1], 'linear', fill_value='extrapolate')

        velscale_ratio, template_velscale, data_velscale = MOSExGalPrepare.calculate_velscale_ratio(configs, logLam, velscale, LSF_Data, LSF_Templates, templates_dir, debug=False)


        # Prepare stellar templates for emi range
        template, lamRange_spmod, logLam_template, ntemplates = MOSExGalPrepare.prepareSpectralTemplateLibrary(
            "PPXF", templates_dir, configs, lmin, lmax, velscale[0], velscale_ratio, LSF_Data, LSF_Templates)[:4]

        # Prepare emi templates to get logLam_emi_template
        emi_template, emi_template_unbroadened,  logLam_emi_template = IFUExGalEMIPPXF.prepare_emission_template(
            logLam_template, line_names, line_wavelengths,
            LSF_Templates, LSF_Data, wl_offset=0.0, debug=False)

        # Prepare stellar fit for emi wavelength range - ADAPTED FROM IFU
        stellar_fit_for_emi, stellar_kinematics_emi_range = prepare_stellar_fit_for_emi_range_mos(
            outdir, rootname, logLam, logLam_emi_template, bin_data, bin_error, template, velscale, velscale_ratio,
            configs, info_tab, error_limit, stellar_kinematics=None, nthreads=nthreads, debug=debug
        )

        ExGalutil.prettyOutput_Done("Preparing fixed stellar fit for emi wavelength range...", progressbar=True)

    elif not emi_range_differs and stellar_available:
        print("emi and stellar wavelength ranges are the same - using existing stellar results")
        # Can use stellar_bestfit directly from the original stellar analysis
        stellar_fit_for_emi = stellar_bestfit_original

    else:
        print("No stellar fit will be used for emi analysis")

    # PREPARE INITIAL CONDITIONS - SAME AS IFU
    nmoments = configs.get('MOM', 4)  # Get actual number of moments
    start = np.zeros((ntargets, max(nmoments, 2)))  # Ensure enough space for both stellar and emi

    if stellar_available and stellar_velocities is not None and stellar_sigmas is not None:
        start[:, 0] = stellar_velocities[:ntargets] if len(stellar_velocities) >= ntargets else stellar_velocities[0]
        start[:, 1] = stellar_sigmas[:ntargets] if len(stellar_sigmas) >= ntargets else stellar_sigmas[0]

        if nmoments > 2 and 'H3' in ppxf_stellar.names:
            start[:, 2] = ppxf_stellar['H3'][:ntargets]
        if nmoments > 3 and 'H4' in ppxf_stellar.names:
            start[:, 3] = ppxf_stellar['H4'][:ntargets]
        if nmoments > 4 and 'H5' in ppxf_stellar.names:
            start[:, 4] = ppxf_stellar['H5'][:ntargets]
        if nmoments > 5 and 'H6' in ppxf_stellar.names:
            start[:, 5] = ppxf_stellar['H6'][:ntargets]

        if debug:
            print(f"Using stellar kinematics as initial guess:")
            print(f"  V range: {np.min(start[:, 0]):.1f} - {np.max(start[:, 0]):.1f} km/s")
            print(f"  SIGMA range: {np.min(start[:, 1]):.1f} - {np.max(start[:, 1]):.1f} km/s")
    else:
        start[:, 0] = 0.0
        start[:, 1] = configs.get('SIGMA', 150.0)
        # Higher moments default to 0
        if nmoments > 2:
            start[:, 2] = 0.0
        if nmoments > 3:
            start[:, 3] = 0.0
        print(f"Using default initial guess: V=0 km/s, SIGMA={configs.get('SIGMA', 150.0)} km/s")

    # Arrays to store results - USE ORIGINAL npix
    emi_metalist = [None for _ in range(ntargets)]
    emi_result = []
    emi_bestfit = np.zeros((ntargets, npix))      # Original npix
    emission_lines = []
    mc_results = []
    formal_error = []
    emi_goodpixels = np.empty((ntargets, npix))   # Original npix
    emi_goodpixels.fill(-1)
    line_names_filterred = []
    # START RUNNING EMI KINEMATICS in serial or parallel mode - FOLLOWING PPXF STELLAR PATTERN
    if nthreads > 1:
        logging.info("Running emi kinematics in parallel mode (MOS)")
        ExGalutil.prettyOutput_Running("Running emi kinematics in parallel mode (MOS)")

        # Create Queues
        inQueue = Queue()
        outQueue = Queue()

        # Create worker processes
        ps = [Process(target=workerPPXF_emi_mos, args=(inQueue, outQueue))
              for _ in range(nthreads)]

        # Start worker processes
        for p in ps:
            p.start()

        # Fill the queue - each target processed individually like PPXF stellar
        for i in range(ntargets):
            stellar_bestfit_target = (stellar_fit_for_emi[i, :]
                                    if stellar_fit_for_emi is not None
                                    else None)

            inQueue.put((outdir, config_dir, templates_dir, logLam[i,:], bin_data[i,:], bin_error[i,:],
                        configs, info_tab, velscale[i], LSF_Templates,
                        error_limit, nsims, ntargets, i, emission_setup,
                        stellar_bestfit_target, start[i, :],  # Pass start array for this target
                        line_names, tie_settings, line_wavelengths, figdir, diag_plots, group_order))

        # Get results - same as PPXF stellar
        emi_tmp = [outQueue.get() for _ in range(ntargets)]

        # Send stop signal to stop iteration
        for _ in range(nthreads):
            inQueue.put('STOP')

        # Stop processes
        for p in ps:
            p.join()

        # Get output - same pattern as PPXF stellar
        index = np.zeros(ntargets)
        for i in range(ntargets):
            index[i] = emi_tmp[i][0]                    # i (target index)
            emi_result.append(emi_tmp[i][1])           # sol
            bestfit_combined = emi_tmp[i][2]           # bestfit (combined)
            emi_bestfit[i,:] = emi_tmp[i][3]           # emi_bestfit
            emission_lines.append(emi_tmp[i][4])       # emission_lines
            mc_results.append(emi_tmp[i][5])           # mc_results
            formal_error.append(emi_tmp[i][6])         # formal_error
            goodpixels_array = emi_tmp[i][7]           # goodpixels_ppxf
            emi_metalist[i] = emi_tmp[i][8]            # emi_meta
            line_names_filterred.append(emi_tmp[i][9]) # list of filtered line_names for this specific target
            # Handle goodpixels properly
            if isinstance(goodpixels_array, np.ndarray) and len(goodpixels_array) > 0:
                goodpix_len = len(goodpixels_array)
                emi_goodpixels[i, 0:goodpix_len] = goodpixels_array

        # Sort output - same as PPXF stellar
        argidx = np.argsort(index)
        emi_bestfit = emi_bestfit[argidx,:]
        emi_goodpixels = emi_goodpixels[argidx,:]
        emi_result = [emi_result[_lms] for _lms in argidx]
        emission_lines = [emission_lines[_lms] for _lms in argidx]
        mc_results = [mc_results[_lms] for _lms in argidx]
        formal_error = [formal_error[_lms] for _lms in argidx]
        emi_metalist = [emi_metalist[_lms] for _lms in argidx]

        ExGalutil.prettyOutput_Done("Running emi kinematics in parallel mode (MOS)", progressbar=True)

    else:
        # Serial processing - same pattern as PPXF stellar
        ExGalutil.prettyOutput_Running("Running emi kinematics in serial mode (MOS)")
        logging.info("Running emi kinematics in serial mode (MOS)")

        for i in range(ntargets):
            stellar_bestfit_target = (stellar_fit_for_emi[i, :]
                                    if stellar_fit_for_emi is not None
                                    else None)

            _, emi_result_i, bestfit_i, emi_bestfit_i, emission_lines_i, mc_results_i, formal_error_i, goodpixels_i, emi_meta_i, line_names_filterred_i = \
                workerPPXF_emi_serial(outdir, config_dir, templates_dir, logLam[i,:], bin_data[i,:], bin_error[i,:],
                                     configs, info_tab, velscale[i], LSF_Templates,
                                     error_limit, nsims, ntargets, i, emission_setup,
                                     stellar_bestfit_target, start[i, :],  # Pass start array for this target
                                     line_names, tie_settings, line_wavelengths, figdir, diag_plots, group_order)




            emi_result.append(emi_result_i)
            emi_bestfit[i,:] = emi_bestfit_i
            emission_lines.append(emission_lines_i)
            mc_results.append(mc_results_i)
            formal_error.append(formal_error_i)
            goodpix_len = len(goodpixels_i)
            if goodpix_len > 0:
                emi_goodpixels[i, 0:goodpix_len] = goodpixels_i
            emi_metalist[i] = emi_meta_i
            line_names_filterred.append(line_names_filterred_i)

        ExGalutil.prettyOutput_Done("Running emi kinematics in serial mode (MOS)", progressbar=True)
    # Check for exceptions which occurred during the analysis
    n_failed = 0
    for i in range(ntargets):
        if isinstance(emi_result[i]["emi"], np.ndarray):
            if np.isnan(emi_result[i]["emi"]).any():
                n_failed += 1
        else:
            n_failed += 1

    if n_failed > 0:
        ExGalutil.prettyOutput_Warning(f"emi kinematics analysis failed for {n_failed} targets")
        logging.warning(f"emi kinematics analysis failed for {n_failed} targets")
    else:
        print("             No problems in the emi kinematics analysis.")
        logging.info("No problems in the emi kinematics analysis.")
    print("")



    # Save emi kinematics to file - UPDATED VERSION
    ExGalutil.prettyOutput_Running("Saving emi kinematics results to disk")
    # NEW: Save standard emi kinematics with ALL lines
    save_emi_kinematics_mos(
        rootname, configs, outdir, emi_metalist, bin_data, bin_error, emi_result,
        mc_results, formal_error, emi_bestfit, logLam, emi_goodpixels,
        emission_lines, line_names_filterred, npix, ubins, tie_settings,
        emission_config=emission_config, line_wavelengths=line_wavelengths,
        group_order=group_order
    )

    # NEW: Also save EMIPPXF-compatible format
    save_ppxf_as_emippxf(
        LEVEL= LEVEL ,  #Default level for MOS
        rootname=rootname,
        configs=configs,
        outdir=outdir,
        metalist=emi_metalist,
        spectra=bin_data,
        error=bin_error,
        emi_result=emi_result,
        mc_results=mc_results,
        formal_error=formal_error,
        emission_lines=emission_lines,
        line_names=line_names_filterred,
        line_wavelengths=line_wavelengths,
        logLam=logLam,
        bestfit=stellar_fit_for_emi + emi_bestfit if 'stellar_fit_for_emi' in locals() and stellar_fit_for_emi is not None else emi_bestfit,
        goodpixels=emi_goodpixels,
        emi_bestfit=emi_bestfit,
        stellar_bestfit=stellar_fit_for_emi if 'stellar_fit_for_emi' in locals() else None,
        npix=npix,
        ubins=ubins,
        emission_config=emission_config,
        tie_settings=tie_settings,
        group_order=group_order,
        z_in=None,  # Don't pass single redshift - functions now handle individual redshifts
        z_err=None, # Don't pass single redshift error
        reddening=None  # MOS doesn't typically use reddening
    )

    ExGalutil.prettyOutput_Done("Saving emi kinematics results to disk")

    # Create diagnostic plots if figdir is specified
    if figdir:
        try:
            ExGalutil.prettyOutput_Running("Creating emi kinematics diagnostic plots")

            # Pass the stellar fit data if available
            stellar_for_plots = stellar_fit_for_emi if stellar_fit_for_emi is not None else stellar_bestfit_original


            make_emi_kinematics_plots(
                rootname,figdir, configs, emi_metalist, emi_result, emission_lines, line_names_filterred,
                logLam, bin_data, emi_bestfit, stellar_bestfit=stellar_for_plots,
                goodpixels_array=emi_goodpixels,
                mc_results=mc_results, debug=debug
            )

            ExGalutil.prettyOutput_Done("Creating emi kinematics diagnostic plots")
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            print("".join(lines))
            sys.stdout.flush()
            ExGalutil.prettyOutput_Failed("Producing emi kinematics diagnostic plots")
            logging.warning("Failed to produce emi kinematics diagnostic plots. Analysis continues!")
            pass
    if debug:
        return emi_result, mc_results, formal_error, emi_bestfit, emission_lines, line_names, emi_metalist
    else:
        return





def PPXF_MaskGaps_MOS(error, goodPixels, error_limit):
    """
    Masking CCD GAPS - same as PPXF stellar approach.

    New approach, here for APS [to mask CCD GAPS]
    we also masked those pixels with errors higher than a certain value to be sure they do not
    contribute in the final fittings
    ## only select those goodpixels for which error is lower than error_limit (CCD gaps)
    ## Please note, as error is affected by Voronoi Binning
    ## and the error after voronio bining is av_err_spec = np.sqrt(np.sum(error[:,k],axis=1))
    ## we use 0.95 * np.sqrt(error_limit) as our error limit
    ## where 0.95 is 2 sigma around this value
    ## In MOS mode, in opposite to IFU, we usually do not run VORONOI
    ## So, we do not expect the error to be affected.
    ## Just to make sure we are on the safe side, here we also use sqrt(error_limit) instead
    ## of error_limit, itself.

    Parameters:
    -----------
    error : ndarray
        Error spectrum array
    goodPixels : ndarray
        Array of good pixel indices
    error_limit : float
        Error limit threshold

    Returns:
    --------
    goodPixels : ndarray
        Filtered array of good pixel indices
    """
    try:
        # Calculate error threshold
        error_threshold = 0.95 * np.sqrt(error_limit)

        # Find good pixels below error threshold
        goodpix_idx = np.ravel(np.where(error[goodPixels] < error_threshold))

        if len(goodpix_idx) > 0:
            goodPixels = goodPixels[goodpix_idx]
        else:
            goodPixels = np.array([], dtype=int)

    except Exception as e:
        print(f"Warning: CCD gap masking failed: {e}")
        goodPixels = np.array([], dtype=int)

    return goodPixels






def worker_stellar_fit_for_emi_mos(inQueue, outQueue):
    """
    Fixed worker process for preparing stellar fits on emi wavelength range - MOS version.
    """
    for template_emi_range, bin_spectrum, noise_spectrum, lam_emi, lam_emi_template, velscale, velscale_ratio, \
        start_fixed, configs, error_limit, target_info \
        in iter(inQueue.get, 'STOP'):

        try:
            # Create goodpixels mask with proper masking
            goodpixels = np.arange(len(lam_emi))

            # Apply spectral masking first (sky lines, etc.)
            try:
                # This should use the same masking as the emi analysis
                config_dir = configs.get('CONFIG_DIR', './configs/')
                if 'LMIN_EMI' in configs and 'LMAX_EMI' in configs:
                    # Apply emi wavelength range
                    logLam_target = np.log(lam_emi)
                    l0_gal = logLam_target[0]
                    lstep_gal = logLam_target[1] - logLam_target[0]
                    npix = len(logLam_target)

                    lmin_emi = configs['LMIN_EMI']
                    lmax_emi = configs['LMAX_EMI']

                    pix0 = int(np.ceil((np.log(lmin_emi) - l0_gal) / lstep_gal))
                    pix1 = int(np.ceil((np.log(lmax_emi) - l0_gal) / lstep_gal))

                    pix0 = max(0, pix0)
                    pix1 = min(npix, pix1)

                    if pix0 < pix1:
                        goodpixels = np.arange(pix0, pix1)
                    else:
                        goodpixels = np.array([], dtype=int)

            except Exception as mask_error:
                print(f"    Warning: Masking failed for target {target_info['APS_ID']}: {mask_error}")
                goodpixels = np.arange(len(lam_emi))

            # Filter out high-error regions
            if noise_spectrum is not None and len(goodpixels) > 0:
                error_threshold = 0.95 * np.sqrt(error_limit)
                good_error_mask = noise_spectrum[goodpixels] < error_threshold
                goodpixels = goodpixels[good_error_mask]

            # Check for finite values
            if len(goodpixels) > 0:
                finite_mask = (np.isfinite(bin_spectrum[goodpixels]) &
                              np.isfinite(noise_spectrum[goodpixels]) &
                              (noise_spectrum[goodpixels] > 0))
                goodpixels = goodpixels[finite_mask]

            if len(goodpixels) < 50:  # Minimum pixels needed
                print(f"    Warning: Only {len(goodpixels)} good pixels in target {target_info['APS_ID']}")
                outQueue.put((target_info, np.full(len(lam_emi), np.nan), start_fixed))
                continue

            # CRITICAL FIX: Use much tighter bounds to truly "fix" stellar kinematics
            # The original bounds were too loose
            bounds = [
                [start_fixed[0]-2, start_fixed[0]+2],  # V ± 2 km/s (tighter)
                [max(20, start_fixed[1]-10), start_fixed[1]+10]  # SIGMA ± 10 km/s (tighter)
            ]

            # Validate bounds
            if bounds[0][0] >= bounds[0][1]:
                bounds[0] = [start_fixed[0]-10, start_fixed[0]+10]
            if bounds[1][0] >= bounds[1][1]:
                bounds[1] = [30, 300]

            # Add bounds for higher moments with very tight constraints
            for j in range(2, len(start_fixed)):
                bounds.append([start_fixed[j]-0.02, start_fixed[j]+0.02])  # Very tight

            # CRITICAL FIX: Use stronger regularization to prevent overfitting
            data_std = np.std(bin_spectrum[goodpixels])
            regularization = max(0.01, data_std * 0.05)  # Stronger regularization

            # CRITICAL FIX: Use lower polynomial degree for emi range
            # emi ranges often have fewer continuum features
            adeg = min(configs.get('ADEG', 12), 8)  # Cap at degree 8
            mdeg = configs.get('MDEG', 0)

            # Run PPXF with stellar template only, quasi-fixed kinematics
            pp = ppxf(template_emi_range, bin_spectrum, noise_spectrum, velscale, velscale_ratio=velscale_ratio,
                     start=start_fixed, goodpixels=goodpixels, plot=False, quiet=True,
                     moments=len(start_fixed), degree=adeg, mdegree=mdeg,
                     lam=lam_emi, lam_temp=lam_emi_template,
                     bounds=bounds,
                     bias=regularization)

            # Check for reasonable results
            if hasattr(pp, 'bestfit') and hasattr(pp, 'sol'):
                bestfit_scale = np.std(pp.bestfit[goodpixels])
                data_scale = np.std(bin_spectrum[goodpixels])


                # # Check if fit is reasonable
                # if bestfit_scale > 10 * data_scale or bestfit_scale < 0.1 * data_scale:
                #     print(f"    Warning: Suspicious stellar fit scale for target {target_info['APS_ID']}")
                #     print(f"      Data scale: {data_scale:.2e}, Fit scale: {bestfit_scale:.2e}")

                # Check kinematics stayed close to input
                if len(pp.sol) >= 2:
                    v_diff = abs(pp.sol[0] - start_fixed[0])
                    sigma_diff = abs(pp.sol[1] - start_fixed[1])

                    if v_diff > 20 or sigma_diff > 20:
                        print(f"    Warning: Stellar kinematics drifted for target {target_info['APS_ID']}")
                        print(f"      V: {start_fixed[0]:.1f} → {pp.sol[0]:.1f} (Δ={v_diff:.1f})")
                        print(f"      SIGMA: {start_fixed[1]:.1f} → {pp.sol[1]:.1f} (Δ={sigma_diff:.1f})")

                # Store results with target_info for proper ordering
                stellar_fit = pp.bestfit
                stellar_kinematics_fitted = pp.sol

            else:
                print(f"    Error: PPXF failed for target {target_info['APS_ID']}")
                stellar_fit = np.full(len(lam_emi), np.nan)
                stellar_kinematics_fitted = start_fixed

            outQueue.put((target_info, stellar_fit, stellar_kinematics_fitted))

        except Exception as e:
            print(f"    Error in target {target_info['APS_ID']}: {str(e)}")
            outQueue.put((target_info, np.full(len(lam_emi), np.nan), start_fixed))


def prepare_stellar_fit_for_emi_range_mos(outdir, rootname, logLam_emi, logLam_emi_template, bin_data_emi,
                                         noise_emi, template_emi_range, velscale, velscale_ratio,
                                         configs, info_tab, error_limit,
                                         stellar_kinematics=None, nthreads=1, debug=False):
    """
    Re-run PPXF on the emi wavelength range with fixed stellar kinematics - MOS version.
    FIXED VERSION with proper masking, bounds, and regularization.
    """
    ExGalutil.prettyOutput_Info("Re-running PPXF on emi wavelength range with fixed stellar kinematics (MOS)...")

    ntargets = bin_data_emi.shape[0]

    # Load stellar kinematics if not provided
    if stellar_kinematics is None:
        try:
            hdu_ppxf = fits.open(outdir + rootname + '_ppxf.fits')
            ppxf_stellar = hdu_ppxf[1].data

            # Extract all moments used in stellar fit
            nmoments = configs.get('MOM', 4)
            stellar_kinematics = np.zeros((ntargets, nmoments))

            stellar_kinematics[:, 0] = ppxf_stellar['V'][:ntargets]
            stellar_kinematics[:, 1] = ppxf_stellar['SIGMA'][:ntargets]

            if nmoments > 2 and 'H3' in ppxf_stellar.names:
                stellar_kinematics[:, 2] = ppxf_stellar['H3'][:ntargets]
            if nmoments > 3 and 'H4' in ppxf_stellar.names:
                stellar_kinematics[:, 3] = ppxf_stellar['H4'][:ntargets]
            if nmoments > 4 and 'H5' in ppxf_stellar.names:
                stellar_kinematics[:, 4] = ppxf_stellar['H5'][:ntargets]
            if nmoments > 5 and 'H6' in ppxf_stellar.names:
                stellar_kinematics[:, 5] = ppxf_stellar['H6'][:ntargets]

            hdu_ppxf.close()
            if debug:
                print(f"Loaded stellar kinematics for {ntargets} targets")

                # Print summary statistics
                print(f"  V range: {np.nanmin(stellar_kinematics[:, 0]):.1f} to {np.nanmax(stellar_kinematics[:, 0]):.1f} km/s")
                print(f"  SIGMA range: {np.nanmin(stellar_kinematics[:, 1]):.1f} to {np.nanmax(stellar_kinematics[:, 1]):.1f} km/s")

        except Exception as e:
            print(f"Error loading stellar kinematics: {e}")
            print("Using default stellar kinematics (V=0, sigma=150)")
            nmoments = configs.get('MOM', 4)
            stellar_kinematics = np.zeros((ntargets, nmoments))
            # stellar_kinematics[:, 1] = 150.0  # Default sigma

    # Initialize output arrays
    npix_emi = len(logLam_emi[0, :])  # Assuming first target's wavelength grid
    stellar_fit_emi_range = np.zeros((ntargets, npix_emi))
    stellar_kinematics_output = stellar_kinematics.copy()

    lam_emi = np.exp(logLam_emi)  # Convert logLam_emi to linear scale
    lam_emi_template = np.exp(logLam_emi_template)  # Convert logLam_emi_template to linear scale

    if debug:
        # CRITICAL: Validate inputs
        print(f"Input validation:")
        print(f"  emi wavelength range: {np.nanmin(lam_emi):.1f} - {np.nanmax(lam_emi):.1f} Å")
        print(f"  Template wavelength range: {np.nanmin(lam_emi_template):.1f} - {np.nanmax(lam_emi_template):.1f} Å")
        print(f"  Template shape: {template_emi_range.shape}")
        print(f"  Data shape: {bin_data_emi.shape}")

    if nthreads > 1:
        ExGalutil.prettyOutput_Running(f"Running stellar fit preparation in parallel mode with {nthreads} threads")

        from multiprocessing import Queue, Process

        # Create Queues
        inQueue = Queue()
        outQueue = Queue()

        # Create worker processes - use the FIXED worker
        ps = [Process(target=worker_stellar_fit_for_emi_mos, args=(inQueue, outQueue))
              for _ in range(nthreads)]

        # Start worker processes
        for p in ps:
            p.start()

        # Fill the queue with jobs
        for i in range(ntargets):
            target_info = {
                'index': i,
                'APS_ID': info_tab['APS_ID'][i],
                'TARGID': info_tab['TARGID'][i],
                'CNAME': info_tab['CNAME'][i]
            }

            inQueue.put((
                template_emi_range,
                bin_data_emi[i, :],
                noise_emi[i, :],
                lam_emi[i, :],
                lam_emi_template,
                velscale[i] if hasattr(velscale, '__len__') else velscale,
                velscale_ratio,
                stellar_kinematics[i, :].copy(),
                configs,
                error_limit,
                target_info
            ))

        # Get results - IMPORTANT: results may come back in any order!
        results = [outQueue.get() for _ in range(ntargets)]

        # Send stop signal to stop iteration
        for _ in range(nthreads):
            inQueue.put('STOP')

        # Stop processes
        for p in ps:
            p.join()

        # Sort results by target index to maintain proper order
        results.sort(key=lambda x: x[0]['index'])  # Sort by target index

        # Extract sorted results
        for i, (target_info, stellar_fit, stellar_kin_fitted) in enumerate(results):
            if target_info['index'] != i:
                print(f"Warning: Expected target {i} but got target {target_info['index']}")

            stellar_fit_emi_range[i, :] = stellar_fit
            stellar_kinematics_output[i, :len(stellar_kin_fitted)] = stellar_kin_fitted

        ExGalutil.prettyOutput_Done("Running stellar fit preparation in parallel mode (MOS)", progressbar=True)
    else:
        ExGalutil.prettyOutput_Running("Running stellar fit preparation in serial mode (MOS)")
        # Serial processing with the same fixes as parallel version
        for i in range(ntargets):
            try:
                if i % 50 == 0 or i < 5:  # Progress updates
                    print(f"  Processing target {i+1}/{ntargets}")

                # Get stellar kinematics for this target
                start_fixed = stellar_kinematics[i, :].copy()

                # Create goodpixels mask with proper masking
                goodpixels = np.arange(len(logLam_emi[i, :]))

                # Apply emi wavelength range masking
                try:
                    if 'LMIN_EMI' in configs and 'LMAX_EMI' in configs:
                        logLam_target = logLam_emi[i, :]
                        l0_gal = logLam_target[0]
                        lstep_gal = logLam_target[1] - logLam_target[0]
                        npix = len(logLam_target)

                        lmin_emi = configs['LMIN_EMI']
                        lmax_emi = configs['LMAX_EMI']

                        pix0 = int(np.ceil((np.log(lmin_emi) - l0_gal) / lstep_gal))
                        pix1 = int(np.ceil((np.log(lmax_emi) - l0_gal) / lstep_gal))

                        pix0 = max(0, pix0)
                        pix1 = min(npix, pix1)

                        if pix0 < pix1:
                            goodpixels = np.arange(pix0, pix1)
                        else:
                            goodpixels = np.array([], dtype=int)
                except Exception:
                    goodpixels = np.arange(len(logLam_emi[i, :]))

                # Filter out high-error regions
                if noise_emi is not None and len(goodpixels) > 0:
                    error_threshold = 0.95 * np.sqrt(error_limit)
                    good_error_mask = noise_emi[i, goodpixels] < error_threshold
                    goodpixels = goodpixels[good_error_mask]

                # Check for finite values
                if len(goodpixels) > 0:
                    finite_mask = (np.isfinite(bin_data_emi[i, goodpixels]) &
                                  np.isfinite(noise_emi[i, goodpixels]) &
                                  (noise_emi[i, goodpixels] > 0))
                    goodpixels = goodpixels[finite_mask]

                if len(goodpixels) < 50:  # Minimum pixels needed
                    print(f"    Warning: Only {len(goodpixels)} good pixels in target {i}")
                    stellar_fit_emi_range[i, :] = np.nan
                    continue

                # FIXED: Use much tighter bounds and stronger regularization
                bounds = [
                    [start_fixed[0]-2, start_fixed[0]+2],  # V ± 2 km/s
                    [max(20, start_fixed[1]-10), start_fixed[1]+10]  # SIGMA ± 10 km/s
                ]

                for j in range(2, len(start_fixed)):
                    bounds.append([start_fixed[j]-0.02, start_fixed[j]+0.02])

                data_std = np.std(bin_data_emi[i, goodpixels])
                regularization = max(0.01, data_std * 0.05)

                adeg = min(configs.get('ADEG', 12), 8)  # Cap polynomial degree
                mdeg = configs.get('MDEG', 0)

                # Run PPXF with stellar template only
                pp = ppxf(template_emi_range, bin_data_emi[i, :], noise_emi[i, :],
                         velscale[i] if hasattr(velscale, '__len__') else velscale, velscale_ratio=velscale_ratio,
                         start=start_fixed, goodpixels=goodpixels, plot=False, quiet=True,
                         moments=len(start_fixed), degree=adeg, mdegree=mdeg,
                         lam=lam_emi[i, :], lam_temp=lam_emi_template,
                         bounds=bounds, bias=regularization)

                # Store the stellar fit model and check quality
                if hasattr(pp, 'bestfit') and hasattr(pp, 'sol'):
                    stellar_fit_emi_range[i, :] = pp.bestfit
                    stellar_kinematics_output[i, :len(pp.sol)] = pp.sol

                    # Quality check
                    if i < 3:  # Print for first few targets
                        bestfit_scale = np.std(pp.bestfit[goodpixels])
                        data_scale = np.std(bin_data_emi[i, goodpixels])
                        print(f"    Target {i}: Data scale {data_scale:.2e}, Fit scale {bestfit_scale:.2e}")

                        if len(pp.sol) >= 2:
                            v_diff = abs(pp.sol[0] - start_fixed[0])
                            sigma_diff = abs(pp.sol[1] - start_fixed[1])
                            print(f"    Kinematics drift: ΔV={v_diff:.1f}, ΔSIGMA={sigma_diff:.1f}")
                else:
                    stellar_fit_emi_range[i, :] = np.nan
                ExGalutil.prettyOutput_Done("Running stellar fit preparation in serial mode (MOS)", progressbar=True)

            except Exception as e:
                print(f"    Error in target {i}: {str(e)}")
                stellar_fit_emi_range[i, :] = np.nan
                continue

    # Final validation
    n_successful = np.sum(~np.isnan(stellar_fit_emi_range[:, 0]))
    n_failed = ntargets - n_successful

    ExGalutil.prettyOutput_Info(f"Successfully fitted: {n_successful}/{ntargets} targets ({n_successful/ntargets*100:.1f}%)")


    if debug:
        if n_failed > 0:
            print(f"  Failed fits: {n_failed} targets")

        # Check for suspicious results

        if n_successful > 0:
            successful_fits = stellar_fit_emi_range[~np.isnan(stellar_fit_emi_range[:, 0]), :]
            fit_scales = np.std(successful_fits, axis=1)

            print(f"  Fit scale range: {np.min(fit_scales):.2e} - {np.max(fit_scales):.2e}")

            if np.max(fit_scales) > 1e6:
                print(f"  ⚠️  WARNING: Some fits have extremely large values!")
            if np.min(fit_scales) < 1e-6:
                print(f"  ⚠️  WARNING: Some fits have extremely small values!")

    return stellar_fit_emi_range, stellar_kinematics_output
