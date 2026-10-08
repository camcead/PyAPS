#!/usr/bin/env python
"""
aps_fwhm.py - FWHM Interpolator from CAL/Wave FITS Files

This module reads CAL/wave FITS files and fits splines to create FWHM
interpolation functions for each fiber, similar to aps_lsf.py but working
from raw measurement data rather than pre-computed B-splines.

Key features:
- Individual spline for EACH fiber (~960 splines)
- Global master spline from median stacking
- Linear interpolation in gap between arms
- Flat extrapolation outside data range
- Pickle save/load for caching
- Compatible API with aps_lsf.py

Author: APS Team
Date: December 2024
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.interpolate import UnivariateSpline
from scipy.stats import median_abs_deviation
from sklearn.mixture import GaussianMixture
import warnings
import glob
import time


# Set threading environment variables
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Import dill for pickle serialization (fallback to pickle if not available)
try:
    import dill
    PICKLE_MODULE = dill
    PICKLE_EXT = '.dill'
except ImportError:
    import pickle
    PICKLE_MODULE = pickle
    PICKLE_EXT = '.pkl'
    warnings.warn("dill not available, falling back to pickle. Some functions may not serialize correctly.")


# =============================================================================
# HELPER FUNCTIONS (matching aps_lsf.py)
# =============================================================================

###########################################################################
def check_and_replace_binned_files(file_paths):
    """
    Check if files exist and replace binned versions (21, 41) with unbinned (11) if not found.

    Parameters:
    -----------
    file_paths : str or list of str
        Single file path or list of file paths to check

    Returns:
    --------
    str or list of str
        Single file path or list of file paths (same type as input)
        Modified to use unbinned versions where necessary
    """
    # Check if input is a single string
    is_single_file = isinstance(file_paths, str)

    # Convert single string to list for uniform processing
    if is_single_file:
        file_paths = [file_paths]

    modified_paths = []
    replaced_files = []

    for file_path in file_paths:
        if os.path.exists(file_path):
            # File exists, use it as-is
            modified_paths.append(file_path)
        else:
            # File doesn't exist, try to find unbinned version
            directory = os.path.dirname(file_path)
            filename = os.path.basename(file_path)

            # Try replacing binning patterns (21 or 41) with 11
            replacement_found = False
            for pattern in ['21', '41']:
                if pattern in filename:
                    new_filename = filename.replace(pattern, '11')
                    new_path = os.path.join(directory, new_filename)

                    if os.path.exists(new_path):
                        modified_paths.append(new_path)
                        replaced_files.append((filename, new_filename))
                        replacement_found = True
                        break

            if not replacement_found:
                # No replacement found, keep original path (will likely fail later)
                modified_paths.append(file_path)

    # Issue warning if any files were replaced
    if replaced_files:
        warning_msg = "Files have been replaced with unbinned ones as the binned files were not found:\n"
        for original, replacement in replaced_files:
            warning_msg += f"  {original} -> {replacement}\n"
        print(" ")
        print(f"  ⚠️   WARNING: {warning_msg.strip()}", file=sys.stderr)
        print(" ")
    # Return same type as input
    return modified_paths[0] if is_single_file else modified_paths




def generate_file_prefix(file_paths):
    """Generate prefix for output filenames based on input FITS files."""
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    if not file_paths:
        return ''
    basenames = []
    for filepath in file_paths:
        basename = os.path.basename(filepath)
        name_without_ext = os.path.splitext(basename)[0]
        basenames.append(name_without_ext)
    prefix = '_'.join(basenames) + '_'
    return prefix


def generate_output_headname(file_paths):
    """
    Generate output headname from input file paths.

    For single file: wave_3100962_all → wave_3100962_all
    For multiple files: wave_3100962_all + wave_3100961_all → wave_3100962_all__wave_3100961_all

    Parameters
    ----------
    file_paths : str or list
        Input file path(s)

    Returns
    -------
    str : Output headname
    """
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    if not file_paths:
        return 'fwhm_output'

    # Extract base names without extension
    basenames = []
    for fpath in file_paths:
        basename = os.path.basename(fpath)
        name_without_ext = os.path.splitext(basename)[0]
        basenames.append(name_without_ext)

    # Join with double underscore for multiple files
    if len(basenames) == 1:
        return basenames[0]
    else:
        return '__'.join(basenames)


def get_output_pickle_path(file_paths, output_dir=None):
    """
    Get the full path for output pickle file.

    Parameters
    ----------
    file_paths : str or list
        Input file path(s)
    output_dir : str, optional
        Output directory. If None, uses directory of first input file.

    Returns
    -------
    str : Full path for pickle file
    """
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    headname = generate_output_headname(file_paths)

    if output_dir is None:
        output_dir = os.path.dirname(file_paths[0])

    return os.path.join(output_dir, f"{headname}{PICKLE_EXT}")


def parse_wave_filename(filename):
    """
    Parse wave filename to extract setup information.

    Example: "wave_3100962_all.fit" -> {'obsid': '3100962', 'type': 'all'}

    Parameters
    ----------
    filename : str
        Wave filename

    Returns
    -------
    dict : Parsed setup information
    """
    basename = os.path.basename(filename)
    name_without_ext = os.path.splitext(basename)[0]

    parts = name_without_ext.split('_')

    parsed = {'filename': basename}

    if len(parts) >= 2:
        if parts[0] == 'wave':
            parsed['type'] = 'wave'
            parsed['obsid'] = parts[1] if len(parts) > 1 else ''
            parsed['suffix'] = parts[2] if len(parts) > 2 else ''

    return parsed


# =============================================================================
# FILE READING AND FILTERING
# =============================================================================

def read_fits_and_filter(file_path, wave_bin_width=50, debug=False):
    """Read a FITS file and apply wavelength-binned outlier detection."""

    if debug:
        print(f"\n📖 Reading wave file: {os.path.basename(file_path)}")
        print(f"   Path: {file_path}")

    start_time = time.time()

    with fits.open(file_path) as hdul:
        extension_results = {}
        valid_extensions = []

        for ext_num, hdu in enumerate(hdul):
            try:
                if hdu.data is not None and len(hdu.data) > 0:
                    ext_results = {}

                    for i, row in enumerate(hdu.data):
                        if row['fiblive'] == 1 and row['ngood'] > 0:
                            fwhm_array = row['fwhm']
                            wave_true_array = row['wave_true']
                            wave_calc_array = row['wave_calc']
                            fit_flag_array = row['fit_flag']
                            specnum = row['specnum']
                            ngood = row['ngood']
                            medresid = row['medresid']
                            fit_rms = row['fit_rms']

                            valid_indices = np.where(fit_flag_array == 0)[0]

                            if len(valid_indices) > 0:
                                filtered_wave_true = wave_true_array[valid_indices]
                                filtered_wave_calc = wave_calc_array[valid_indices]
                                filtered_fwhm = fwhm_array[valid_indices]

                                ext_results[specnum] = {
                                    'wave_true': filtered_wave_true,
                                    'wave_calc': filtered_wave_calc,
                                    'fwhm': filtered_fwhm,
                                    'ngood': ngood,
                                    'medresid': medresid,
                                    'fit_rms': fit_rms,
                                    'row_index': i,
                                    'extension': ext_num
                                }

                    if ext_results:
                        extension_results[ext_num] = ext_results
                        valid_extensions.append(ext_num)

            except Exception as e:
                if debug:
                    print(f"  Extension {ext_num}: Error - {e}")
                continue

    if not extension_results:
        raise ValueError(f"No valid data found in any extension of {os.path.basename(file_path)}")

    # Merge extensions
    merged_results = _merge_extensions(extension_results, valid_extensions, file_path)

    # Apply wavelength-binned filtering
    filtered_results = _apply_wavelength_binned_filtering(merged_results, wave_bin_width, debug=debug)

    if debug:
        total_time = time.time() - start_time
        print(f"  ✓ Processed {len(filtered_results)} fibers in {total_time:.2f}s")

    return filtered_results


def _merge_extensions(extension_results, valid_extensions, file_path):
    """Merge extensions by specnum."""
    merged_results = {}

    if len(extension_results) == 1:
        ext_data = list(extension_results.values())[0]
        for specnum, data in ext_data.items():
            data['source_file'] = os.path.basename(file_path)
            merged_results[specnum] = data
    else:
        reference_ext = valid_extensions[0]

        for specnum in extension_results[reference_ext].keys():
            all_wave_true = []
            all_wave_calc = []
            all_fwhm = []

            ref_data = extension_results[reference_ext][specnum]

            for ext_num in valid_extensions:
                if specnum in extension_results[ext_num]:
                    ext_data = extension_results[ext_num][specnum]
                    all_wave_true.extend(ext_data['wave_true'])
                    all_wave_calc.extend(ext_data['wave_calc'])
                    all_fwhm.extend(ext_data['fwhm'])

            combined_wave = np.array(all_wave_true)
            combined_calc = np.array(all_wave_calc)
            combined_fwhm = np.array(all_fwhm)

            sort_idx = np.argsort(combined_wave)

            merged_results[specnum] = {
                'wave_true': combined_wave[sort_idx],
                'wave_calc': combined_calc[sort_idx],
                'fwhm': combined_fwhm[sort_idx],
                'ngood': ref_data['ngood'],
                'medresid': ref_data['medresid'],
                'fit_rms': ref_data['fit_rms'],
                'source_file': os.path.basename(file_path),
                'extensions_merged': valid_extensions
            }

    return merged_results


def _apply_wavelength_binned_filtering(file_results, wave_bin_width=50, debug=False):
    """Optimized outlier detection in wavelength bins."""

    total_points = sum(len(data['wave_true']) for data in file_results.values())

    if total_points == 0:
        return file_results

    # Collect all data
    all_wavelengths = np.empty(total_points)
    all_fwhm = np.empty(total_points)
    fiber_indices = np.empty(total_points, dtype=int)
    point_indices = np.empty(total_points, dtype=int)

    idx = 0
    specnum_to_int = {}
    int_to_specnum = {}

    for fiber_int, (specnum, data) in enumerate(file_results.items()):
        specnum_to_int[specnum] = fiber_int
        int_to_specnum[fiber_int] = specnum

        wave = data['wave_true']
        fwhm = data['fwhm']
        n_points = len(wave)

        all_wavelengths[idx:idx+n_points] = wave
        all_fwhm[idx:idx+n_points] = fwhm
        fiber_indices[idx:idx+n_points] = fiber_int
        point_indices[idx:idx+n_points] = np.arange(n_points)
        idx += n_points

    # Bin the data
    wave_min = np.min(all_wavelengths)
    wave_max = np.max(all_wavelengths)
    n_bins = max(1, int((wave_max - wave_min) / wave_bin_width))

    bin_edges = np.linspace(wave_min, wave_max, n_bins + 1)
    bin_assignments = np.digitize(all_wavelengths, bin_edges) - 1
    bin_assignments = np.clip(bin_assignments, 0, n_bins - 1)

    outlier_mask = np.zeros(total_points, dtype=bool)

    for bin_idx in range(n_bins):
        bin_mask = (bin_assignments == bin_idx)
        bin_points = np.sum(bin_mask)

        if bin_points < 5:
            continue

        bin_fwhm = all_fwhm[bin_mask]
        median_fwhm = np.median(bin_fwhm)
        std_fwhm = np.std(bin_fwhm)

        if std_fwhm > 0:
            threshold = 3 * std_fwhm
            bin_outliers = np.abs(bin_fwhm - median_fwhm) > threshold
            bin_indices = np.where(bin_mask)[0]
            outlier_mask[bin_indices[bin_outliers]] = True

    # Build filtered results
    filtered_results = {}
    total_removed = 0

    for fiber_int, specnum in int_to_specnum.items():
        data = file_results[specnum]

        fiber_mask = (fiber_indices == fiber_int)
        fiber_outlier_mask = outlier_mask[fiber_mask]
        fiber_good_mask = ~fiber_outlier_mask

        fiber_point_indices = point_indices[fiber_mask]
        good_indices = fiber_point_indices[fiber_good_mask]

        if len(good_indices) > 0:
            filtered_results[specnum] = {
                'wave_true': data['wave_true'][good_indices],
                'wave_calc': data['wave_calc'][good_indices],
                'fwhm': data['fwhm'][good_indices],
                'ngood': data['ngood'],
                'medresid': data['medresid'],
                'fit_rms': data['fit_rms'],
                'source_file': data.get('source_file', ''),
                'n_original': len(data['wave_true']),
                'n_filtered': len(good_indices),
                'outliers_removed': len(data['wave_true']) - len(good_indices)
            }
            total_removed += len(data['wave_true']) - len(good_indices)
        else:
            filtered_results[specnum] = data.copy()
            filtered_results[specnum]['n_original'] = len(data['wave_true'])
            filtered_results[specnum]['n_filtered'] = len(data['wave_true'])
            filtered_results[specnum]['outliers_removed'] = 0

    if debug:
        print(f"  ✓ Outlier filtering: removed {total_removed} points")

    return filtered_results


def _apply_bimodal_filtering(file_results, min_second_group_fraction=0.20, debug=False):
    """Apply bimodal filtering to remove secondary population."""

    total_points = sum(len(data['fwhm']) for data in file_results.values())
    all_fwhm = np.empty(total_points)

    idx = 0
    for data in file_results.values():
        n_points = len(data['fwhm'])
        all_fwhm[idx:idx+n_points] = data['fwhm']
        idx += n_points

    if len(all_fwhm) < 20:
        return file_results

    # Test for bimodality
    mean_val = np.mean(all_fwhm)
    std_val = np.std(all_fwhm)

    if std_val == 0:
        return file_results

    skewness = np.mean(((all_fwhm - mean_val) / std_val) ** 3)
    if abs(skewness) < 0.5:
        return file_results

    try:
        gmm = GaussianMixture(n_components=2, random_state=42, max_iter=50)
        labels = gmm.fit_predict(all_fwhm.reshape(-1, 1))

        group_counts = np.bincount(labels)
        group_fractions = group_counts / len(all_fwhm)
        min_fraction = min(group_fractions)

        if min_fraction < min_second_group_fraction:
            return file_results

        dominant_component = np.argmax(group_counts)
        global_mask = (labels == dominant_component)

        # Apply filtering
        filtered_results = {}
        idx = 0

        for specnum, data in file_results.items():
            n_points = len(data['fwhm'])
            fiber_mask = global_mask[idx:idx+n_points]
            idx += n_points

            if np.any(fiber_mask):
                filtered_results[specnum] = {
                    'wave_true': data['wave_true'][fiber_mask],
                    'wave_calc': data['wave_calc'][fiber_mask],
                    'fwhm': data['fwhm'][fiber_mask],
                    'ngood': data['ngood'],
                    'medresid': data['medresid'],
                    'fit_rms': data['fit_rms'],
                    'source_file': data.get('source_file', ''),
                    'bimodal_filtered': True
                }
            else:
                filtered_results[specnum] = data.copy()
                filtered_results[specnum]['bimodal_filtered'] = False

        if debug:
            total_removed = sum(len(file_results[s]['fwhm']) - len(filtered_results[s]['fwhm'])
                              for s in file_results.keys())
            print(f"  ✓ Bimodal filtering: removed {total_removed} points")

        return filtered_results

    except:
        return file_results


def read_multiple_fits_files(file_paths, wave_bin_width=50, apply_bimodal_filtering=False, debug=False):
    """Read multiple FITS files and return combined results."""

    if isinstance(file_paths, str):
        if '*' in file_paths or '?' in file_paths:
            file_list = sorted(glob.glob(file_paths))
        else:
            file_list = [file_paths]
    else:
        file_list = file_paths

    if debug:
        print(f"\n{'='*70}")
        print(f"FWHM INTERPOLATOR - Reading Wave FITS Files")
        print(f"  Files: {len(file_list)}")
        print(f"{'='*70}")

    all_file_results = {}
    file_coverage = {}

    for file_path in file_list:
        try:
            file_results = read_fits_and_filter(file_path, wave_bin_width, debug=debug)

            if apply_bimodal_filtering:
                file_results = _apply_bimodal_filtering(file_results, debug=debug)

            all_file_results[file_path] = file_results

            if file_results:
                all_waves = np.concatenate([data['wave_true'] for data in file_results.values()])
                wave_range = (np.min(all_waves), np.max(all_waves))
                file_coverage[file_path] = wave_range

                if debug:
                    print(f"  ✓ Wavelength range: {wave_range[0]:.1f} - {wave_range[1]:.1f} Å")

        except Exception as e:
            if debug:
                print(f"  ✗ Error reading {os.path.basename(file_path)}: {e}")
            continue

    if not all_file_results:
        raise ValueError("No valid CAL/wave files found")

    return all_file_results, file_coverage


# =============================================================================
# SPLINE FITTING
# =============================================================================

def fit_individual_fiber_splines(all_file_results, spline_order=3, spline_smoothing=None,
                                  apply_residual_filtering=True, debug=False):
    """
    Fit individual spline for EACH fiber.

    Parameters
    ----------
    all_file_results : dict
        Dict of {file_path: {specnum: fiber_data}}
    spline_order : int
        Order of spline (k parameter, 1-5). Default: 3 (cubic)
    spline_smoothing : float or None
        Smoothing parameter. None = auto
    apply_residual_filtering : bool
        Remove outliers in residuals after initial fit
    debug : bool
        Print debug info

    Returns
    -------
    fiber_splines : dict
        Dict of {specnum: {'spline': spline, 'wave_range': (min, max), ...}}
    """

    if debug:
        print(f"\n🔬 Fitting individual splines for each fiber...")

    # Collect all fibers across all files
    all_fibers = {}
    for file_path, file_results in all_file_results.items():
        for specnum, data in file_results.items():
            if specnum not in all_fibers:
                all_fibers[specnum] = {
                    'wavelengths': [],
                    'fwhm': [],
                    'files': []
                }
            all_fibers[specnum]['wavelengths'].extend(data['wave_true'])
            all_fibers[specnum]['fwhm'].extend(data['fwhm'])
            all_fibers[specnum]['files'].append(file_path)

    # Fit spline for each fiber
    fiber_splines = {}
    n_success = 0
    n_failed = 0

    for specnum, fiber_data in all_fibers.items():
        try:
            wavelengths = np.array(fiber_data['wavelengths'])
            fwhm = np.array(fiber_data['fwhm'])

            # Sort by wavelength
            sort_idx = np.argsort(wavelengths)
            wavelengths = wavelengths[sort_idx]
            fwhm = fwhm[sort_idx]

            # Remove outliers (5-MAD threshold)
            median_fwhm = np.median(fwhm)
            mad = np.median(np.abs(fwhm - median_fwhm))

            if mad > 0:
                outlier_threshold = 5 * 1.4826 * mad
                good_mask = np.abs(fwhm - median_fwhm) <= outlier_threshold

                if np.sum(good_mask) < 5:
                    n_failed += 1
                    continue

                wavelengths = wavelengths[good_mask]
                fwhm = fwhm[good_mask]

            # Handle duplicate wavelengths
            unique_waves, inverse = np.unique(wavelengths, return_inverse=True)
            if len(unique_waves) < len(wavelengths):
                averaged_fwhm = np.zeros(len(unique_waves))
                for i in range(len(unique_waves)):
                    mask = inverse == i
                    averaged_fwhm[i] = np.median(fwhm[mask])
                wavelengths = unique_waves
                fwhm = averaged_fwhm

            # Need enough points
            if len(wavelengths) < max(spline_order + 1, 5):
                n_failed += 1
                continue

            # Calculate smoothing
            if spline_smoothing is None:
                clean_median = np.median(fwhm)
                residuals = fwhm - clean_median
                noise_var = np.var(residuals)
                auto_smooth = len(wavelengths) * noise_var * 0.5
                auto_smooth = max(0.01, min(10.0, auto_smooth))
                s = auto_smooth
            else:
                s = spline_smoothing

            # Fit spline
            fiber_spline = UnivariateSpline(wavelengths, fwhm, s=s, k=spline_order)

            # Get fitted values
            fitted_fwhm = fiber_spline(wavelengths)
            residuals = fwhm - fitted_fwhm
            rms = np.sqrt(np.mean(residuals**2))

            # Optional residual filtering
            if apply_residual_filtering and len(residuals) >= 10:
                median_resid = np.median(residuals)
                mad_resid = np.median(np.abs(residuals - median_resid))

                if mad_resid > 0:
                    threshold = 3 * 1.4826 * mad_resid
                    good_mask = np.abs(residuals - median_resid) <= threshold

                    if np.sum(good_mask) >= max(spline_order + 1, 5):
                        wavelengths_clean = wavelengths[good_mask]
                        fwhm_clean = fwhm[good_mask]

                        fiber_spline = UnivariateSpline(wavelengths_clean, fwhm_clean, s=s, k=spline_order)

                        wavelengths = wavelengths_clean
                        fwhm = fwhm_clean
                        fitted_fwhm = fiber_spline(wavelengths)
                        residuals = fwhm - fitted_fwhm
                        rms = np.sqrt(np.mean(residuals**2))

            # Get boundary values for extrapolation
            wave_min = np.min(wavelengths)
            wave_max = np.max(wavelengths)
            first_val = float(fiber_spline(wave_min))
            last_val = float(fiber_spline(wave_max))

            # Store results
            fiber_splines[specnum] = {
                'spline': fiber_spline,
                'wavelengths': wavelengths,
                'fwhm': fwhm,
                'fitted_fwhm': fitted_fwhm,
                'residuals': residuals,
                'rms': rms,
                'n_points': len(wavelengths),
                'wave_range': (wave_min, wave_max),
                'first_val': first_val,
                'last_val': last_val,
                'smoothing': s
            }

            n_success += 1

            if debug and specnum in list(all_fibers.keys())[:3]:
                print(f"  ✓ Fiber {specnum}: {len(wavelengths)} points, RMS={rms:.4f}")

        except Exception as e:
            if debug and specnum in list(all_fibers.keys())[:3]:
                print(f"  ✗ Fiber {specnum}: Failed - {e}")
            n_failed += 1
            continue

    if debug:
        print(f"  ✓ Successfully fit {n_success} fiber splines ({n_failed} failed)")

    return fiber_splines


def create_global_master_spline(fiber_splines, spline_order=3, spline_smoothing=None,
                                 wave_grid_resolution=1.0, debug=False):
    """
    Create global master spline from stacking all fiber splines.

    For single-arm: Stack all fibers and fit single spline.
    For multi-arm: This is called AFTER merging, so fibers already have linear transitions.

    Parameters
    ----------
    fiber_splines : dict
        Individual fiber splines (may be merged for multi-arm)
    spline_order : int
        Order for global spline
    spline_smoothing : float or None
        Smoothing parameter
    wave_grid_resolution : float
        Wavelength grid resolution (Å)
    debug : bool
        Print debug info

    Returns
    -------
    global_spline : callable
        Master spline/interpolation function
    wave_grid : array
        Wavelength grid
    stacked_fwhm : array
        Median FWHM at each wavelength
    """

    if debug:
        print(f"\n🌍 Creating global master from {len(fiber_splines)} fibers...")

    # Find wavelength range from fiber data
    all_wave_mins = []
    all_wave_maxs = []

    for info in fiber_splines.values():
        if 'wave_range' in info:
            all_wave_mins.append(info['wave_range'][0])
            all_wave_maxs.append(info['wave_range'][1])

    wave_min = np.min(all_wave_mins)
    wave_max = np.max(all_wave_maxs)

    # Create evaluation grid
    n_points = int((wave_max - wave_min) / wave_grid_resolution) + 1
    wave_grid = np.linspace(wave_min, wave_max, n_points)

    if debug:
        print(f"  Wavelength grid: {n_points} points ({wave_min:.1f} - {wave_max:.1f} Å)")

    # Evaluate all fiber interpolators on grid to get stacked median
    fiber_fwhm_matrix = np.zeros((len(fiber_splines), len(wave_grid)))

    for i, (specnum, fiber_info) in enumerate(fiber_splines.items()):
        try:
            # Use interpolate_function if available (merged fibers)
            if 'interpolate_function' in fiber_info:
                fiber_fwhm_matrix[i, :] = fiber_info['interpolate_function'](wave_grid)
            else:
                # Single-arm: use spline with flat extrapolation
                fiber_wave_min, fiber_wave_max = fiber_info['wave_range']
                first_val = fiber_info['first_val']
                last_val = fiber_info['last_val']
                spline = fiber_info['spline']

                for j, w in enumerate(wave_grid):
                    if w < fiber_wave_min:
                        fiber_fwhm_matrix[i, j] = first_val
                    elif w > fiber_wave_max:
                        fiber_fwhm_matrix[i, j] = last_val
                    else:
                        fiber_fwhm_matrix[i, j] = float(spline(w))
        except:
            fiber_fwhm_matrix[i, :] = np.nan

    # Stack using median
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        stacked_fwhm = np.nanmedian(fiber_fwhm_matrix, axis=0)

    # Remove NaN values
    valid_mask = np.isfinite(stacked_fwhm)
    wave_grid_clean = wave_grid[valid_mask]
    stacked_fwhm_clean = stacked_fwhm[valid_mask]

    if debug:
        print(f"  Stacked FWHM range: {np.min(stacked_fwhm_clean):.4f} - {np.max(stacked_fwhm_clean):.4f}")

    # Fit a spline to the stacked data for smooth evaluation
    if spline_smoothing is None:
        noise_var = np.var(stacked_fwhm_clean - np.median(stacked_fwhm_clean))
        auto_smooth = len(wave_grid_clean) * noise_var * 0.1
        auto_smooth = max(1.0, min(1000.0, auto_smooth))
        s = auto_smooth
    else:
        s = spline_smoothing

    # Create the global spline
    global_spline = UnivariateSpline(wave_grid_clean, stacked_fwhm_clean, s=s, k=spline_order)

    if debug:
        global_fitted = global_spline(wave_grid_clean)
        global_rms = np.sqrt(np.mean((stacked_fwhm_clean - global_fitted)**2))
        print(f"  Global spline RMS: {global_rms:.4f}")

    return global_spline, wave_grid_clean, stacked_fwhm_clean


def create_global_per_arm(all_fiber_splines, all_wave_ranges, spline_order=3,
                          spline_smoothing=None, wave_grid_resolution=1.0, debug=False):
    """
    Create global master spline PER ARM, then combine with linear transition.

    CRITICAL: Each fiber has its OWN wavelength range. When creating the global,
    we only evaluate each fiber within ITS range to avoid extrapolation artifacts.

    This ensures the global has the same shape as individual fibers:
    - Arm1 global spline for arm1 range
    - Arm2 global spline for arm2 range
    - LINEAR transition between arms
    - FLAT extrapolation outside

    Parameters
    ----------
    all_fiber_splines : list
        List of fiber spline dicts, one per arm
    all_wave_ranges : list
        List of (min, max) wavelength ranges per arm (GLOBAL ranges)
    spline_order : int
        Spline order
    spline_smoothing : float or None
        Smoothing parameter
    wave_grid_resolution : float
        Grid resolution (Å)
    debug : bool
        Print debug info

    Returns
    -------
    global_func : callable
        Global interpolation function with linear transition
    wave_grid : array
        Combined wavelength grid
    stacked_fwhm : array
        Combined stacked FWHM values
    arm_globals : list
        Per-arm global splines (for plotting)
    """

    if debug:
        print(f"\n🌍 Creating per-arm global masters...")

    arm_globals = []
    arm_grids = []
    arm_stacked = []
    arm_first_vals = []  # First value of each arm's global
    arm_last_vals = []   # Last value of each arm's global

    # Create global spline for each arm
    for arm_idx, (fiber_splines, wave_range) in enumerate(zip(all_fiber_splines, all_wave_ranges)):

        wave_min_global, wave_max_global = wave_range
        n_points = int((wave_max_global - wave_min_global) / wave_grid_resolution) + 1
        wave_grid = np.linspace(wave_min_global, wave_max_global, n_points)

        # Evaluate all fibers on this arm's grid
        # CRITICAL: Only evaluate each fiber within ITS range, set others to NaN
        fiber_fwhm_matrix = np.full((len(fiber_splines), len(wave_grid)), np.nan)

        for i, (specnum, fiber_info) in enumerate(fiber_splines.items()):
            try:
                spline = fiber_info['spline']
                fiber_wave_min, fiber_wave_max = fiber_info['wave_range']

                # Only evaluate within THIS fiber's range
                for j, w in enumerate(wave_grid):
                    if fiber_wave_min <= w <= fiber_wave_max:
                        fiber_fwhm_matrix[i, j] = float(spline(w))
                    # Outside fiber's range -> leave as NaN (don't extrapolate)
            except Exception as e:
                if debug:
                    print(f"    Warning: Failed to evaluate fiber {specnum}: {e}")
                pass  # Leave as NaN

        # Median stack (ignoring NaN)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            stacked_fwhm = np.nanmedian(fiber_fwhm_matrix, axis=0)

        # Remove wavelengths where ALL fibers were NaN
        valid_mask = np.isfinite(stacked_fwhm)
        wave_grid_clean = wave_grid[valid_mask]
        stacked_fwhm_clean = stacked_fwhm[valid_mask]

        if len(wave_grid_clean) < 10:
            raise ValueError(f"Arm {arm_idx + 1}: Too few valid wavelengths ({len(wave_grid_clean)})")

        # Fit spline to the stacked median
        if spline_smoothing is None:
            noise_var = np.var(stacked_fwhm_clean - np.median(stacked_fwhm_clean))
            auto_smooth = len(wave_grid_clean) * noise_var * 0.1
            s = max(0.1, min(100.0, auto_smooth))
        else:
            s = spline_smoothing

        arm_spline = UnivariateSpline(wave_grid_clean, stacked_fwhm_clean, s=s, k=spline_order)

        # Get boundary values at the ACTUAL data range (not global range)
        actual_wave_min = np.min(wave_grid_clean)
        actual_wave_max = np.max(wave_grid_clean)
        arm_first = float(arm_spline(actual_wave_min))
        arm_last = float(arm_spline(actual_wave_max))

        arm_globals.append(arm_spline)
        arm_grids.append(wave_grid_clean)
        arm_stacked.append(stacked_fwhm_clean)
        arm_first_vals.append((actual_wave_min, arm_first))
        arm_last_vals.append((actual_wave_max, arm_last))

        if debug:
            print(f"  Arm {arm_idx + 1}: {actual_wave_min:.1f} - {actual_wave_max:.1f} Å")
            print(f"    (Global range was {wave_min_global:.1f} - {wave_max_global:.1f} Å)")
            print(f"    {len(fiber_splines)} fibers, {len(wave_grid_clean)} valid points")
            print(f"    First val: {arm_first:.4f} at {actual_wave_min:.1f} Å")
            print(f"    Last val: {arm_last:.4f} at {actual_wave_max:.1f} Å")

    # Get actual arm ranges from the data (not global input ranges)
    arm1_actual_min, arm1_first = arm_first_vals[0]
    arm1_actual_max, arm1_last = arm_last_vals[0]
    arm2_actual_min, arm2_first = arm_first_vals[1]
    arm2_actual_max, arm2_last = arm_last_vals[1]

    # Transition region: from end of arm1 to start of arm2
    trans_start = arm1_actual_max
    trans_end = arm2_actual_min
    trans_val_start = arm1_last   # Value at end of arm1
    trans_val_end = arm2_first    # Value at start of arm2

    # Extrapolation values
    first_val = arm1_first  # For wavelengths before arm1
    last_val = arm2_last    # For wavelengths after arm2

    if debug:
        print(f"\n  Transition: {trans_start:.1f} - {trans_end:.1f} Å")
        print(f"  Transition values: {trans_val_start:.4f} → {trans_val_end:.4f}")
        if trans_end < trans_start:
            print(f"  (Note: overlap region - arms overlap by {trans_start - trans_end:.1f} Å)")

    # Create combined global function
    def global_interp_func(wavelengths):
        wavelengths = np.atleast_1d(wavelengths)
        fwhm = np.zeros_like(wavelengths, dtype=float)

        arm1_spline = arm_globals[0]
        arm2_spline = arm_globals[1]

        for i, w in enumerate(wavelengths):
            # FLAT before arm1
            if w < arm1_actual_min:
                fwhm[i] = first_val

            # FLAT after arm2
            elif w > arm2_actual_max:
                fwhm[i] = last_val

            # Arm1 SPLINE (within arm1's actual range)
            elif w <= arm1_actual_max:
                fwhm[i] = float(arm1_spline(w))

            # Arm2 SPLINE (within arm2's actual range)
            elif w >= arm2_actual_min:
                fwhm[i] = float(arm2_spline(w))

            # Transition (LINEAR) - between arm1_max and arm2_min
            else:
                if trans_end > trans_start:
                    # Gap: normal linear interpolation
                    alpha = (w - trans_start) / (trans_end - trans_start)
                else:
                    # Overlap: this shouldn't happen if w is between arm1_max and arm2_min
                    # but handle gracefully
                    alpha = 0.5
                fwhm[i] = (1 - alpha) * trans_val_start + alpha * trans_val_end

        fwhm = np.maximum(fwhm, 0.1)
        return fwhm if len(wavelengths) > 1 else float(fwhm[0])

    # Create combined grid and stacked values for plotting
    wave_grid_combined = np.concatenate(arm_grids)
    stacked_combined = np.concatenate(arm_stacked)
    sort_idx = np.argsort(wave_grid_combined)
    wave_grid_combined = wave_grid_combined[sort_idx]
    stacked_combined = stacked_combined[sort_idx]

    return global_interp_func, wave_grid_combined, stacked_combined, arm_globals


# =============================================================================
# FWHM INTERPOLATOR CLASS (matching aps_lsf.py interface)
# =============================================================================

class FWHMInterpolator:
    """
    FWHM Interpolator from CAL/Wave FITS files.

    Compatible API with LSFInterpolator from aps_lsf.py.

    Features:
    - Individual spline for each fiber PER ARM
    - Global master spline from median stacking (per arm)
    - LINEAR interpolation in transition between arms (gap OR overlap)
    - FLAT extrapolation outside data range
    - Pickle save/load support

    Interpolation Strategy (matching aps_lsf.py):
    - Region 0: λ < Arm1_min → FLAT extrapolation (first value of arm 1)
    - Region 1: Arm1_min ≤ λ ≤ Arm1_max → SPLINE from arm 1
    - Region 2: Transition (gap or overlap) → LINEAR from arm1_max value to arm2_min value
    - Region 3: Arm2_min ≤ λ ≤ Arm2_max → SPLINE from arm 2
    - Region 4: λ > Arm2_max → FLAT extrapolation (last value of arm 2)

    IMPORTANT: Both gap and overlap are treated the SAME way with LINEAR transition.
    This ensures the spline shape within each arm is IDENTICAL whether processing
    single-arm or double-arm data.
    """

    def __init__(self):
        # Python minor version the pickled closures' bytecode belongs to (see aps_lsf)
        self._cache_python = (sys.version_info.major, sys.version_info.minor)
        # Per-file splines (for multi-arm handling)
        self.all_fiber_splines = []  # List of dicts, one per file
        self.all_wave_ranges = []    # List of (min, max) per file

        # Merged splines (after combining arms)
        self.fiber_splines = None
        self.global_spline = None
        self.wave_grid = None
        self.stacked_fwhm = None
        self.interpolator_dict = None

        # File info
        self.all_file_results = None
        self.file_coverage = None
        self.file_paths = None
        self.wave_ranges = []
        self.debug = False
        self.file_prefix = ''
        self.metadata = {}

        # Gap/Overlap handling
        self.has_gap = False
        self.has_overlap = False
        self.gap_start = None
        self.gap_end = None
        self.overlap_start = None
        self.overlap_end = None

    def set_debug(self, debug):
        """Set debug mode."""
        self.debug = debug

    def create_from_files(self, file_paths, wave_bin_width=50, apply_bimodal_filtering=False,
                          apply_residual_filtering=True, spline_order=3, spline_smoothing=None,
                          wave_grid_resolution=1.0):
        """
        Create FWHM interpolator from wave FITS files.

        Parameters
        ----------
        file_paths : str or list
            Path(s) to wave FITS file(s)
        wave_bin_width : float
            Wavelength bin width for outlier detection (Å)
        apply_bimodal_filtering : bool
            Apply bimodal filtering during file reading
        apply_residual_filtering : bool
            Apply residual filtering after spline fit
        spline_order : int
            Spline order (k parameter, 1-5)
        spline_smoothing : float or None
            Smoothing parameter (None = auto)
        wave_grid_resolution : float
            Wavelength grid resolution for evaluation (Å)

        Returns
        -------
        bool : True if successful
        """

        if isinstance(file_paths, str):
            file_paths = [file_paths]

        self.file_paths = file_paths
        self.file_prefix = generate_file_prefix(file_paths)

        # Read files
        self.all_file_results, self.file_coverage = read_multiple_fits_files(
            file_paths, wave_bin_width, apply_bimodal_filtering, debug=self.debug
        )

        # Store wavelength ranges per file (sorted by wavelength)
        self.wave_ranges = [self.file_coverage[fp] for fp in file_paths if fp in self.file_coverage]

        # Fit individual fiber splines PER FILE (not merged yet)
        self.all_fiber_splines = []
        self.all_wave_ranges = []

        for file_path in file_paths:
            if file_path not in self.all_file_results:
                continue

            file_results = {file_path: self.all_file_results[file_path]}

            if self.debug:
                print(f"\n🔧 Fitting splines for {os.path.basename(file_path)}...")

            fiber_splines = fit_individual_fiber_splines(
                file_results,
                spline_order=spline_order,
                spline_smoothing=spline_smoothing,
                apply_residual_filtering=apply_residual_filtering,
                debug=self.debug
            )

            if fiber_splines and len(fiber_splines) > 0:
                self.all_fiber_splines.append(fiber_splines)
                self.all_wave_ranges.append(self.file_coverage[file_path])

        if len(self.all_fiber_splines) == 0:
            raise ValueError("Failed to fit any fiber splines")

        # Detect gap/overlap between files
        self._detect_gap_or_overlap()

        # Handle single vs multi-arm differently
        if len(self.all_fiber_splines) == 1:
            # SINGLE ARM: use fiber splines directly, create single global
            self.fiber_splines = self.all_fiber_splines[0]
            self.arm_globals = None

            # Create global master spline
            self.global_spline, self.wave_grid, self.stacked_fwhm = create_global_master_spline(
                self.fiber_splines,
                spline_order=spline_order,
                spline_smoothing=spline_smoothing,
                wave_grid_resolution=wave_grid_resolution,
                debug=self.debug
            )
        else:
            # MULTI-ARM: merge fibers with linear transition, create per-arm globals
            self._merge_fiber_splines(wave_grid_resolution)

            # Create per-arm global splines, then combine with linear transition
            # This ensures global has same shape as individual fibers
            self.global_spline, self.wave_grid, self.stacked_fwhm, self.arm_globals = \
                create_global_per_arm(
                    self.all_fiber_splines,
                    self.all_wave_ranges,
                    spline_order=spline_order,
                    spline_smoothing=spline_smoothing,
                    wave_grid_resolution=wave_grid_resolution,
                    debug=self.debug
                )

        # Create interpolator dictionary
        self._create_interpolator_dict()

        # Store metadata
        self.metadata = {
            'files': [os.path.basename(f) for f in file_paths],
            'n_fibers': len(self.fiber_splines),
            'spline_order': spline_order,
            'wave_range': (np.min(self.wave_grid), np.max(self.wave_grid)),
            'has_gap': self.has_gap,
            'has_overlap': self.has_overlap
        }

        if self.debug:
            print(f"\n✅ FWHM Interpolator created successfully!")
            self.print_summary()

        return True

    def _detect_gap_or_overlap(self):
        """
        Detect gap or overlap between multiple files (arms).

        IMPORTANT: Both gap and overlap are treated the SAME way:
        - LINEAR interpolation connecting end of arm1 to start of arm2
        - This ensures the spline shape within each arm is preserved
        """

        if len(self.all_wave_ranges) < 2:
            self.has_gap = False
            self.has_overlap = False
            self.transition_start = None
            self.transition_end = None
            return

        # Sort by wavelength (blue arm first)
        sorted_indices = sorted(range(len(self.all_wave_ranges)),
                               key=lambda i: self.all_wave_ranges[i][0])

        # Reorder to ensure blue arm is first
        self.all_fiber_splines = [self.all_fiber_splines[i] for i in sorted_indices]
        self.all_wave_ranges = [self.all_wave_ranges[i] for i in sorted_indices]

        wave1_min, wave1_max = self.all_wave_ranges[0]  # Blue arm
        wave2_min, wave2_max = self.all_wave_ranges[1]  # Red arm

        # Check for overlap
        if wave1_max > wave2_min:
            # OVERLAP: arms overlap - transition from arm1_max to arm2_min
            self.has_overlap = True
            self.has_gap = False
            self.overlap_start = wave2_min   # Where red arm starts
            self.overlap_end = wave1_max     # Where blue arm ends
            # Transition region: from end of blue arm to start of red arm
            self.transition_start = wave1_max  # End of blue arm
            self.transition_end = wave2_min    # Start of red arm

            if self.debug:
                print(f"\n🔗 OVERLAP detected:")
                print(f"   Blue arm: {wave1_min:.1f} - {wave1_max:.1f} Å")
                print(f"   Red arm: {wave2_min:.1f} - {wave2_max:.1f} Å")
                print(f"   Overlap region: {wave2_min:.1f} - {wave1_max:.1f} Å")
                print(f"   → Will use LINEAR connection from {wave1_max:.1f} to {wave2_min:.1f} Å")
        else:
            # GAP: arms don't overlap - transition from arm1_max to arm2_min
            self.has_gap = True
            self.has_overlap = False
            self.gap_start = wave1_max
            self.gap_end = wave2_min
            # Transition region: from end of blue arm to start of red arm
            self.transition_start = wave1_max
            self.transition_end = wave2_min

            if self.debug:
                print(f"\n🔗 GAP detected:")
                print(f"   Blue arm: {wave1_min:.1f} - {wave1_max:.1f} Å")
                print(f"   Red arm: {wave2_min:.1f} - {wave2_max:.1f} Å")
                print(f"   Gap: {wave1_max:.1f} - {wave2_min:.1f} Å ({wave2_min - wave1_max:.1f} Å wide)")
                print(f"   → Will use LINEAR connection from {wave1_max:.1f} to {wave2_min:.1f} Å")

    def _merge_fiber_splines(self, wave_grid_resolution=1.0):
        """
        Merge fiber splines from multiple files with LINEAR transition.

        CRITICAL: Each fiber has its OWN wavelength range, so the transition
        region is DIFFERENT for each fiber. We must use per-fiber ranges,
        not global ranges.

        Strategy (SAME for both gap and overlap):
        - λ < fiber_arm1_min: FLAT at fiber's first value
        - fiber_arm1_min ≤ λ ≤ fiber_arm1_max: SPLINE from arm1
        - fiber_arm1_max < λ < fiber_arm2_min: LINEAR transition
        - fiber_arm2_min ≤ λ ≤ fiber_arm2_max: SPLINE from arm2
        - λ > fiber_arm2_max: FLAT at fiber's last value

        This ensures the spline shape within each arm is IDENTICAL whether
        processing single-arm or double-arm data.
        """

        if self.debug:
            print(f"\n🔗 Merging {len(self.all_fiber_splines)} files...")

        # Global ranges (for reference only)
        wave1_min_global, wave1_max_global = self.all_wave_ranges[0]  # Blue arm global
        wave2_min_global, wave2_max_global = self.all_wave_ranges[1]  # Red arm global

        if self.debug:
            print(f"  Global Arm1 range: {wave1_min_global:.1f} - {wave1_max_global:.1f} Å")
            print(f"  Global Arm2 range: {wave2_min_global:.1f} - {wave2_max_global:.1f} Å")
            print(f"  ⚠️ Using PER-FIBER ranges for transition (not global!)")

        # Find common fibers
        fiber_ids_0 = set(self.all_fiber_splines[0].keys())
        fiber_ids_1 = set(self.all_fiber_splines[1].keys())
        common_fiber_ids = sorted(fiber_ids_0.intersection(fiber_ids_1))

        if self.debug:
            print(f"  Common fibers: {len(common_fiber_ids)}")

        # Create merged fiber splines
        self.fiber_splines = {}

        # Track range variations for debug
        trans_variations = []

        for fib_idx, fib_id in enumerate(common_fiber_ids):

            fiber0 = self.all_fiber_splines[0][fib_id]  # Blue arm
            fiber1 = self.all_fiber_splines[1][fib_id]  # Red arm

            # Get PER-FIBER wavelength ranges (CRITICAL!)
            f0_wave_min, f0_wave_max = fiber0['wave_range']  # This fiber's blue arm range
            f1_wave_min, f1_wave_max = fiber1['wave_range']  # This fiber's red arm range

            # Get per-fiber splines
            f0_spline = fiber0['spline']
            f1_spline = fiber1['spline']

            # Get per-fiber boundary values
            f0_first = fiber0['first_val']  # Value at start of this fiber's blue arm
            f0_last = fiber0['last_val']    # Value at END of this fiber's blue arm
            f1_first = fiber1['first_val']  # Value at START of this fiber's red arm
            f1_last = fiber1['last_val']    # Value at end of this fiber's red arm

            # Per-fiber transition region
            # Transition is from END of arm1 to START of arm2
            fiber_trans_start = f0_wave_max  # End of this fiber's blue arm
            fiber_trans_end = f1_wave_min    # Start of this fiber's red arm

            # Values at transition boundaries (evaluated at actual fiber boundaries)
            f0_at_trans = f0_last   # Already evaluated at f0_wave_max
            f1_at_trans = f1_first  # Already evaluated at f1_wave_min

            # Track variations
            trans_variations.append((fiber_trans_start, fiber_trans_end))

            # Debug first few fibers
            if self.debug and fib_idx < 3:
                global_diff_start = abs(fiber_trans_start - wave1_max_global)
                global_diff_end = abs(fiber_trans_end - wave2_min_global)
                print(f"  Fiber {fib_id}:")
                print(f"    Arm1: {f0_wave_min:.1f} - {f0_wave_max:.1f} Å (global ends at {wave1_max_global:.1f}, diff={global_diff_start:.1f})")
                print(f"    Arm2: {f1_wave_min:.1f} - {f1_wave_max:.1f} Å (global starts at {wave2_min_global:.1f}, diff={global_diff_end:.1f})")
                print(f"    Transition: {fiber_trans_start:.1f} - {fiber_trans_end:.1f} Å")
                print(f"    Transition values: {f0_at_trans:.4f} → {f1_at_trans:.4f}")

            # Create merged interpolation function with PER-FIBER ranges
            def make_merged_func(spline0, spline1,
                                w0_min, w0_max, w1_min, w1_max,
                                first0, last0, first1, last1,
                                trans_start, trans_end,
                                val0_at_trans, val1_at_trans):
                """
                Create merged interpolation function using PER-FIBER ranges.

                Regions (using THIS FIBER's ranges):
                - λ < w0_min: FLAT at first0 (before this fiber's blue arm)
                - w0_min ≤ λ ≤ w0_max: SPLINE from blue arm
                - w0_max < λ < w1_min: LINEAR from val0_at_trans to val1_at_trans
                - w1_min ≤ λ ≤ w1_max: SPLINE from red arm
                - λ > w1_max: FLAT at last1 (after this fiber's red arm)
                """

                def merged_interp_func(wavelengths):
                    wavelengths = np.atleast_1d(wavelengths)
                    fwhm = np.zeros_like(wavelengths, dtype=float)

                    for i, w in enumerate(wavelengths):

                        # Region 0: FLAT before this fiber's blue arm
                        if w < w0_min:
                            fwhm[i] = first0

                        # Region 4: FLAT after this fiber's red arm
                        elif w > w1_max:
                            fwhm[i] = last1

                        # Region 1: Blue arm SPLINE (within this fiber's blue arm range)
                        elif w <= w0_max:
                            try:
                                fwhm[i] = float(spline0(w))
                            except:
                                fwhm[i] = last0  # Fallback

                        # Region 3: Red arm SPLINE (within this fiber's red arm range)
                        elif w >= w1_min:
                            try:
                                fwhm[i] = float(spline1(w))
                            except:
                                fwhm[i] = first1  # Fallback

                        # Region 2: Transition (between this fiber's arms) - LINEAR
                        else:
                            # Linear interpolation from blue end to red start
                            if trans_end > trans_start:
                                alpha = (w - trans_start) / (trans_end - trans_start)
                            else:
                                # Edge case: arms overlap for this fiber
                                alpha = 0.5
                            fwhm[i] = (1 - alpha) * val0_at_trans + alpha * val1_at_trans

                    fwhm = np.maximum(fwhm, 0.1)  # Ensure positive
                    return fwhm if len(wavelengths) > 1 else float(fwhm[0])

                return merged_interp_func

            merged_func = make_merged_func(
                f0_spline, f1_spline,
                f0_wave_min, f0_wave_max, f1_wave_min, f1_wave_max,
                f0_first, f0_last, f1_first, f1_last,
                fiber_trans_start, fiber_trans_end,
                f0_at_trans, f1_at_trans
            )

            # Store merged fiber data
            # Global range covers all fibers
            wave_min_global = min(f0_wave_min, wave1_min_global)
            wave_max_global = max(f1_wave_max, wave2_max_global)

            self.fiber_splines[fib_id] = {
                'spline': merged_func,  # This is now a merged function
                'interpolate_function': merged_func,
                'wavelengths': None,
                'fwhm': None,
                'fitted_fwhm': None,
                'residuals': np.array([]),
                'rms': (fiber0['rms'] + fiber1['rms']) / 2,
                'n_points': fiber0['n_points'] + fiber1['n_points'],
                'wave_range': (f0_wave_min, f1_wave_max),  # This fiber's full range
                'first_val': f0_first,
                'last_val': f1_last,
                'type': 'merged',
                # Store per-arm info
                'arm1_range': (f0_wave_min, f0_wave_max),
                'arm2_range': (f1_wave_min, f1_wave_max),
                'arm1_spline': f0_spline,
                'arm2_spline': f1_spline,
                'transition_range': (fiber_trans_start, fiber_trans_end),
                'transition_values': (f0_at_trans, f1_at_trans)
            }

        # Store transition info (global approximation for plotting)
        self.transition_start = wave1_max_global
        self.transition_end = wave2_min_global

        if self.debug:
            # Show range of transition variations
            trans_starts = [t[0] for t in trans_variations]
            trans_ends = [t[1] for t in trans_variations]
            print(f"\n  ✓ Merged {len(self.fiber_splines)} fibers")
            print(f"  Transition start range: {min(trans_starts):.1f} - {max(trans_starts):.1f} Å")
            print(f"  Transition end range: {min(trans_ends):.1f} - {max(trans_ends):.1f} Å")

    def _create_interpolator_dict(self):
        """Create the interpolator dictionary."""

        self.interpolator_dict = {}

        # Get global wavelength range
        wave_min_g = np.min(self.wave_grid)
        wave_max_g = np.max(self.wave_grid)

        # For multi-arm: global_spline is already a function with all regions handled
        # For single-arm: global_spline is a UnivariateSpline, need to add flat extrapolation

        if len(self.all_fiber_splines) == 1:
            # SINGLE ARM: need to wrap spline with flat extrapolation
            first_val = float(self.global_spline(wave_min_g))
            last_val = float(self.global_spline(wave_max_g))

            def make_single_arm_global(spline, wmin, wmax, first_v, last_v):
                def global_func(wavelengths):
                    wavelengths = np.atleast_1d(wavelengths)
                    fwhm = np.zeros_like(wavelengths, dtype=float)

                    for i, w in enumerate(wavelengths):
                        if w < wmin:
                            fwhm[i] = first_v  # FLAT before
                        elif w > wmax:
                            fwhm[i] = last_v   # FLAT after
                        else:
                            fwhm[i] = float(spline(w))  # SPLINE

                    fwhm = np.maximum(fwhm, 0.1)
                    return fwhm if len(wavelengths) > 1 else float(fwhm[0])
                return global_func

            global_interp_func = make_single_arm_global(
                self.global_spline, wave_min_g, wave_max_g, first_val, last_val
            )
        else:
            # MULTI-ARM: global_spline is already a function with all regions
            global_interp_func = self.global_spline

        self.interpolator_dict['global'] = {
            'interpolate_function': global_interp_func,
            'type': 'global_fwhm',
            'description': 'Global master from median of all fibers',
            'wavelength_range': (wave_min_g, wave_max_g),
            'n_fibers': len(self.fiber_splines)
        }

        # Create fiber-specific interpolation functions
        for specnum, fiber_info in self.fiber_splines.items():

            # For merged fibers, use the merged interpolation function
            if 'interpolate_function' in fiber_info:
                fiber_interp_func = fiber_info['interpolate_function']
            else:
                # Single-file case - create function with flat extrapolation
                fiber_wave_min, fiber_wave_max = fiber_info['wave_range']
                fiber_first_val = fiber_info['first_val']
                fiber_last_val = fiber_info['last_val']
                fiber_spline = fiber_info['spline']

                def make_fiber_func(spline, wmin, wmax, first_v, last_v):
                    """Create fiber interpolation function with proper closure."""

                    def fiber_interp_func(wavelengths):
                        wavelengths = np.atleast_1d(wavelengths)
                        fwhm = np.zeros_like(wavelengths, dtype=float)

                        for i, w in enumerate(wavelengths):
                            if w < wmin:
                                fwhm[i] = first_v  # FLAT before
                            elif w > wmax:
                                fwhm[i] = last_v   # FLAT after
                            else:
                                try:
                                    fwhm[i] = float(spline(w))  # SPLINE
                                except:
                                    if w < (wmin + wmax) / 2:
                                        fwhm[i] = first_v
                                    else:
                                        fwhm[i] = last_v

                        fwhm = np.maximum(fwhm, 0.1)
                        return fwhm if len(wavelengths) > 1 else float(fwhm[0])

                    return fiber_interp_func

                fiber_interp_func = make_fiber_func(
                    fiber_spline, fiber_wave_min, fiber_wave_max,
                    fiber_first_val, fiber_last_val
                )

            self.interpolator_dict[specnum] = {
                'interpolate_function': fiber_interp_func,
                'type': 'fiber_fwhm',
                'specnum': specnum,
                'wavelength_range': fiber_info['wave_range'],
                'n_points': fiber_info['n_points'],
                'rms': fiber_info['rms']
            }

    def get_fwhm(self, wavelength, specnum=None):
        """
        Get FWHM value(s) at given wavelength(s).

        Parameters
        ----------
        wavelength : float or array
            Wavelength(s) to query (Angstroms)
        specnum : int, optional
            Fiber specnum. If None, returns global FWHM.

        Returns
        -------
        float or array : FWHM value(s)
        """

        if self.interpolator_dict is None:
            raise ValueError("Interpolator not initialized. Call create_from_files() first.")

        if specnum is None:
            return self.interpolator_dict['global']['interpolate_function'](wavelength)
        else:
            if specnum not in self.interpolator_dict:
                raise ValueError(f"Fiber {specnum} not found. Available fibers: {self.get_available_fibers()[:10]}...")
            return self.interpolator_dict[specnum]['interpolate_function'](wavelength)

    def get_interpolator_dict(self):
        """Get the comprehensive interpolator dictionary."""
        return self.interpolator_dict

    def get_available_fibers(self):
        """Get list of available fiber specnums."""
        if self.interpolator_dict is None:
            return []
        return [k for k in self.interpolator_dict.keys() if k != 'global']

    def build_spaxel_weighted_entries(self, weight_matrix, fibre_nspecs, aps_ids):
        """Opt-in IFU-cube feature: combine this interpolator's existing
        per-fibre FWHM `interpolate_function`s into one per-spaxel
        weighted-average `interpolate_function` each, using a
        (n_spaxel x n_fibre_instance) geometric contribution-weight
        matrix (see `aps_ifu_spaxel_contrib.contributing_fibre_weights`).

        Thin wrapper around `aps_ifu_spaxel_contrib.
        build_spaxel_weighted_entries` — identical contract to
        `aps_lsf.LSFInterpolator`'s own method of the same name (see
        that one's docstring for the full behaviour): no new fitting, no
        mutation of this object's own `interpolator_dict`, and callers
        merge the result into their own copy of `get_interpolator_dict()`
        under a namespaced key rather than in place.

        Returns
        -------
        dict
            `{aps_id: {'interpolate_function': callable,
            'type': 'spaxel_fwhm_weighted', 'n_contrib_fibres': int}, ...}`
            — see `aps_ifu_spaxel_contrib.build_spaxel_weighted_entries`.
        """
        if self.interpolator_dict is None or self.wave_grid is None:
            return {}
        from PyAPS.aps_ifu_spaxel_contrib import build_spaxel_weighted_entries
        return build_spaxel_weighted_entries(
            self.interpolator_dict, self.wave_grid, weight_matrix,
            fibre_nspecs, aps_ids, entry_type='spaxel_fwhm_weighted',
        )

    def get_file_prefix(self):
        """Get the file prefix for output naming."""
        return self.file_prefix

    def print_summary(self):
        """Print summary of interpolator."""

        print(f"\nFWHM INTERPOLATOR SUMMARY:")
        print(f"  Source: CAL/Wave FITS files")
        print(f"  Individual fiber splines: {len(self.fiber_splines)}")
        print(f"  Global master: Per-arm medians with linear transition")
        print(f"  Wavelength range: {np.min(self.wave_grid):.1f} - {np.max(self.wave_grid):.1f} Å")
        print(f"  Files processed: {len(self.file_paths)}")

        if len(self.all_wave_ranges) >= 2:
            print(f"  Arm 1 range: {self.all_wave_ranges[0][0]:.1f} - {self.all_wave_ranges[0][1]:.1f} Å → SPLINE")
            print(f"  Arm 2 range: {self.all_wave_ranges[1][0]:.1f} - {self.all_wave_ranges[1][1]:.1f} Å → SPLINE")

            if self.has_gap:
                print(f"  Gap: {self.gap_start:.1f} - {self.gap_end:.1f} Å → LINEAR")
            elif self.has_overlap:
                print(f"  Overlap: {self.overlap_start:.1f} - {self.overlap_end:.1f} Å → LINEAR")

        print(f"  Extrapolation: FLAT (constant at boundary values)")

    def save(self, filepath=None):
        """
        Save interpolator to pickle file.

        Parameters
        ----------
        filepath : str, optional
            Output filepath. If None, auto-generates from input files.

        Returns
        -------
        str : Path to saved file
        """

        if filepath is None:
            filepath = get_output_pickle_path(self.file_paths)

        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

        with open(filepath, 'wb') as f:
            PICKLE_MODULE.dump(self, f)

        print(f"✓ Saved FWHM interpolator to: {filepath}")
        return filepath

    @classmethod
    def load(cls, filepath):
        """
        Load interpolator from pickle file.

        Parameters
        ----------
        filepath : str
            Path to pickle file

        Returns
        -------
        FWHMInterpolator : Loaded interpolator
        """

        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"FWHM interpolator file not found: {filepath}\n"
                f"Please run run_fwhm_analysis() first to create the interpolator,\n"
                f"or check that the path is correct."
            )

        with open(filepath, 'rb') as f:
            interpolator = PICKLE_MODULE.load(f)

        print(f"✓ Loaded FWHM interpolator from: {filepath}")
        return interpolator


def load_fwhm_interpolator(pickle_path):
    """
    Convenience function to load a FWHM interpolator from pickle.

    Parameters
    ----------
    pickle_path : str
        Path to the pickle file

    Returns
    -------
    FWHMInterpolator : Loaded interpolator
    """
    return FWHMInterpolator.load(pickle_path)


# =============================================================================
# DIAGNOSTIC PLOTS (matching aps_lsf.py style)
# =============================================================================

def create_diagnostic_plots(interpolator, output_dir, figname):
    """
    Create comprehensive diagnostic plots showing all interpolation regions.

    Regions shown (matching aps_lsf.py):
    - BLUE: Flat extrapolation before blue arm
    - CYAN: Blue arm spline region
    - YELLOW/GREEN: Gap (linear) or Overlap (average) region
    - ORANGE: Red arm spline region
    - RED: Flat extrapolation after red arm

    Parameters
    ----------
    interpolator : FWHMInterpolator
        The interpolator to visualize
    output_dir : str
        Output directory for plots
    figname : str
        Base name for plot file
    """

    print(f"\n📊 Creating diagnostic plots...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Get wavelength info
    wave_min_data = np.min(interpolator.wave_grid)
    wave_max_data = np.max(interpolator.wave_grid)

    # Extend range for extrapolation visualization
    padding = (wave_max_data - wave_min_data) * 0.1
    wave_extended_min = wave_min_data - padding
    wave_extended_max = wave_max_data + padding

    # Get arm ranges
    if len(interpolator.all_wave_ranges) >= 2:
        arm1_min, arm1_max = interpolator.all_wave_ranges[0]
        arm2_min, arm2_max = interpolator.all_wave_ranges[1]
    else:
        arm1_min, arm1_max = wave_min_data, wave_max_data
        arm2_min, arm2_max = None, None

    # =========================================================================
    # Panel 1: Full interpolation with ALL REGIONS shown
    # =========================================================================
    ax = axes[0, 0]

    # Create wavelength arrays for each region
    n_pts = 200

    # Region 0: Flat extrapolation BEFORE blue arm
    wave_before = np.linspace(wave_extended_min, arm1_min, n_pts)
    fwhm_before = interpolator.get_fwhm(wave_before)

    # Region 1: Blue arm spline
    wave_arm1 = np.linspace(arm1_min, arm1_max, n_pts)
    fwhm_arm1 = interpolator.get_fwhm(wave_arm1)

    if arm2_min is not None:
        # Region 2/3: Transition (LINEAR for both gap and overlap)
        # Get transition region bounds
        if interpolator.has_gap:
            trans_start = interpolator.gap_start
            trans_end = interpolator.gap_end
            trans_label = f'Gap LINEAR ({trans_start:.0f}-{trans_end:.0f} Å)'
        elif interpolator.has_overlap:
            trans_start = interpolator.overlap_start  # arm2_min
            trans_end = interpolator.overlap_end      # arm1_max
            trans_label = f'Overlap LINEAR ({trans_start:.0f}-{trans_end:.0f} Å)'
        else:
            trans_start = arm1_max
            trans_end = arm2_min
            trans_label = f'Transition LINEAR'

        # Sample transition region
        wave_trans = np.linspace(trans_start, trans_end, n_pts)
        fwhm_trans = interpolator.get_fwhm(wave_trans)

        # Region 4: Red arm spline
        wave_arm2 = np.linspace(arm2_min, arm2_max, n_pts)
        fwhm_arm2 = interpolator.get_fwhm(wave_arm2)

        # Region 5: Flat extrapolation AFTER red arm
        wave_after = np.linspace(arm2_max, wave_extended_max, n_pts)
        fwhm_after = interpolator.get_fwhm(wave_after)
    else:
        # Single arm - just extrapolation after
        wave_after = np.linspace(arm1_max, wave_extended_max, n_pts)
        fwhm_after = interpolator.get_fwhm(wave_after)

    # Plot each region with distinct colors and styles
    # Region 0: Before data - FLAT (dashed blue)
    ax.plot(wave_before, fwhm_before, 'b--', linewidth=2.5,
            label='Flat extrap (before)', zorder=10)

    # Region 1: Arm 1 - SPLINE (solid cyan)
    ax.plot(wave_arm1, fwhm_arm1, 'c-', linewidth=2.5,
            label=f'Arm 1 spline ({arm1_min:.0f}-{arm1_max:.0f} Å)', zorder=20)

    if arm2_min is not None:
        # Region 2/3: Transition - LINEAR (dashed gold) - SAME for gap and overlap
        ax.plot(wave_trans, fwhm_trans, color='gold', linestyle='--', linewidth=2.5,
                label=trans_label, zorder=15)
        # Mark transition region
        ax.axvspan(trans_start, trans_end, alpha=0.15, color='yellow', zorder=1)

        # Region 4: Arm 2 - SPLINE (solid orange)
        ax.plot(wave_arm2, fwhm_arm2, color='orange', linestyle='-', linewidth=2.5,
                label=f'Arm 2 spline ({arm2_min:.0f}-{arm2_max:.0f} Å)', zorder=20)

        # Region 5: After data - FLAT (dashed red)
        ax.plot(wave_after, fwhm_after, 'r--', linewidth=2.5,
                label='Flat extrap (after)', zorder=10)
    else:
        # Single arm - just extrapolation after
        ax.plot(wave_after, fwhm_after, 'r--', linewidth=2.5,
                label='Flat extrap (after)', zorder=10)

    # Mark arm boundaries with vertical lines
    ax.axvline(arm1_min, color='blue', linestyle=':', alpha=0.7, linewidth=1.5)
    ax.axvline(arm1_max, color='cyan', linestyle=':', alpha=0.7, linewidth=1.5)
    if arm2_min is not None:
        ax.axvline(arm2_min, color='orange', linestyle=':', alpha=0.7, linewidth=1.5)
        ax.axvline(arm2_max, color='red', linestyle=':', alpha=0.7, linewidth=1.5)

    # Shade extrapolation regions
    ax.axvspan(wave_extended_min, arm1_min, alpha=0.1, color='blue', zorder=0)
    if arm2_min is not None:
        ax.axvspan(arm2_max, wave_extended_max, alpha=0.1, color='red', zorder=0)
    else:
        ax.axvspan(arm1_max, wave_extended_max, alpha=0.1, color='red', zorder=0)

    ax.set_xlabel('Wavelength (Å)', fontsize=12)
    ax.set_ylabel('FWHM (Å)', fontsize=12)
    ax.set_title('Global FWHM Interpolation - All Regions\n(matching aps_lsf.py behavior)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 2: Fiber splines + global master (data region only)
    # =========================================================================
    ax = axes[0, 1]

    # Plot sample of fiber splines (faded)
    n_sample = min(100, len(interpolator.fiber_splines))
    sample_fibers = list(interpolator.fiber_splines.keys())[:n_sample]

    for specnum in sample_fibers:
        fiber_fwhm = interpolator.get_fwhm(interpolator.wave_grid, specnum=specnum)
        ax.plot(interpolator.wave_grid, fiber_fwhm, 'gray', linewidth=0.5, alpha=0.2)

    # Plot global master
    global_fwhm = interpolator.get_fwhm(interpolator.wave_grid)
    ax.plot(interpolator.wave_grid, global_fwhm, 'r-', linewidth=3,
            label='Global Master', zorder=100)

    # Plot stacked median
    ax.plot(interpolator.wave_grid, interpolator.stacked_fwhm, 'b--', linewidth=2,
            label='Median Stack', alpha=0.7, zorder=99)

    # Mark regions
    ax.axvline(arm1_min, color='cyan', linestyle=':', alpha=0.7, label='Arm 1 bounds')
    ax.axvline(arm1_max, color='cyan', linestyle=':', alpha=0.7)
    if arm2_min is not None:
        ax.axvline(arm2_min, color='orange', linestyle=':', alpha=0.7, label='Arm 2 bounds')
        ax.axvline(arm2_max, color='orange', linestyle=':', alpha=0.7)

        if interpolator.has_gap:
            ax.axvspan(interpolator.gap_start, interpolator.gap_end,
                       alpha=0.2, color='yellow', label='Gap')
        elif interpolator.has_overlap:
            ax.axvspan(interpolator.overlap_start, interpolator.overlap_end,
                       alpha=0.2, color='lightgreen', label='Overlap')

    ax.set_xlabel('Wavelength (Å)', fontsize=12)
    ax.set_ylabel('FWHM (Å)', fontsize=12)
    ax.set_title(f'All Fiber Splines + Global Master\n({len(interpolator.fiber_splines)} fibers)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 3: Fiber-to-fiber variation (boxplot)
    # =========================================================================
    ax = axes[1, 0]

    test_waves = np.linspace(wave_min_data, wave_max_data, 20)
    fiber_matrix = np.zeros((len(interpolator.fiber_splines), len(test_waves)))

    for i, specnum in enumerate(interpolator.fiber_splines.keys()):
        try:
            fiber_matrix[i, :] = interpolator.get_fwhm(test_waves, specnum=specnum)
        except:
            fiber_matrix[i, :] = np.nan

    bp = ax.boxplot([fiber_matrix[:, i] for i in range(len(test_waves))],
                     positions=test_waves,
                     widths=(test_waves[-1] - test_waves[0]) / 25,
                     patch_artist=True,
                     showfliers=False)

    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)

    global_at_test = interpolator.get_fwhm(test_waves)
    ax.plot(test_waves, global_at_test, 'ro-', linewidth=2, markersize=8,
            label='Global Master', zorder=100)

    # Mark regions
    if arm2_min is not None:
        if interpolator.has_gap:
            ax.axvspan(interpolator.gap_start, interpolator.gap_end,
                       alpha=0.2, color='yellow', label='Gap')
        elif interpolator.has_overlap:
            ax.axvspan(interpolator.overlap_start, interpolator.overlap_end,
                       alpha=0.2, color='lightgreen', label='Overlap')

    ax.set_xlabel('Wavelength (Å)', fontsize=12)
    ax.set_ylabel('FWHM (Å)', fontsize=12)
    ax.set_title('Fiber-to-Fiber Variation', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 4: Interpolation strategy summary
    # =========================================================================
    ax = axes[1, 1]

    # Create a visual summary of the interpolation regions
    # Extended wavelength range
    wave_full = np.linspace(wave_extended_min, wave_extended_max, 500)
    fwhm_full = interpolator.get_fwhm(wave_full)

    # Color-code each region
    colors = []
    for w in wave_full:
        if w < arm1_min:
            colors.append('blue')  # Flat before
        elif w <= arm1_max:
            colors.append('cyan')  # Arm 1 spline
        elif arm2_min is not None:
            if interpolator.has_gap and interpolator.gap_start < w < interpolator.gap_end:
                colors.append('gold')  # Gap linear
            elif interpolator.has_overlap and interpolator.overlap_start <= w <= interpolator.overlap_end:
                colors.append('limegreen')  # Overlap average
            elif w < arm2_min:
                colors.append('gold')  # In gap region
            elif w <= arm2_max:
                colors.append('orange')  # Arm 2 spline
            else:
                colors.append('red')  # Flat after
        else:
            colors.append('red')  # Flat after (single arm)

    # Plot with color segments
    for i in range(len(wave_full) - 1):
        ax.plot(wave_full[i:i+2], fwhm_full[i:i+2], color=colors[i], linewidth=2)

    # Add legend entries manually
    ax.plot([], [], 'b-', linewidth=3, label='Flat extrap (before arm 1)')
    ax.plot([], [], 'c-', linewidth=3, label='Arm 1: Spline')
    if arm2_min is not None:
        if interpolator.has_gap:
            ax.plot([], [], color='gold', linestyle='-', linewidth=3, label='Gap: Linear interp')
        elif interpolator.has_overlap:
            ax.plot([], [], color='limegreen', linestyle='-', linewidth=3, label='Overlap: Average')
        ax.plot([], [], color='orange', linestyle='-', linewidth=3, label='Arm 2: Spline')
    ax.plot([], [], 'r-', linewidth=3, label='Flat extrap (after arm 2)')

    # Add vertical boundary markers
    ax.axvline(arm1_min, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(arm1_max, color='black', linestyle='--', alpha=0.5, linewidth=1)
    if arm2_min is not None:
        ax.axvline(arm2_min, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(arm2_max, color='black', linestyle='--', alpha=0.5, linewidth=1)

    # Add region labels
    y_label = np.max(fwhm_full) + 0.05 * (np.max(fwhm_full) - np.min(fwhm_full))
    ax.text((wave_extended_min + arm1_min) / 2, y_label, 'FLAT', ha='center', fontsize=9, color='blue')
    ax.text((arm1_min + arm1_max) / 2, y_label, 'ARM 1\nSPLINE', ha='center', fontsize=9, color='cyan')
    if arm2_min is not None:
        if interpolator.has_gap:
            ax.text((interpolator.gap_start + interpolator.gap_end) / 2, y_label,
                    'GAP\nLINEAR', ha='center', fontsize=9, color='goldenrod')
        elif interpolator.has_overlap:
            ax.text((interpolator.overlap_start + interpolator.overlap_end) / 2, y_label,
                    'OVERLAP\nAVERAGE', ha='center', fontsize=9, color='green')
        ax.text((arm2_min + arm2_max) / 2, y_label, 'ARM 2\nSPLINE', ha='center', fontsize=9, color='orange')
        ax.text((arm2_max + wave_extended_max) / 2, y_label, 'FLAT', ha='center', fontsize=9, color='red')
    else:
        ax.text((arm1_max + wave_extended_max) / 2, y_label, 'FLAT', ha='center', fontsize=9, color='red')

    ax.set_xlabel('Wavelength (Å)', fontsize=12)
    ax.set_ylabel('FWHM (Å)', fontsize=12)
    ax.set_title('Interpolation Strategy Summary\n(Color-coded by region type)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"{figname}.png")
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Plot saved: {plot_path}")

    plt.close(fig)


# =============================================================================
# MAIN ANALYSIS FUNCTION (matching aps_lsf.py interface)
# =============================================================================

def run_fwhm_analysis(file_paths, figdir=None, figname="analysis", debug=False, make_plot=True,
                      wave_bin_width=50, apply_bimodal_filtering=False, apply_residual_filtering=True,
                      spline_order=3, spline_smoothing=None, wave_grid_resolution=1.0,
                      overwrite=False, save_pickle=True, pickle_dir=None,replace_binned=False):
    """
    Run FWHM analysis and create interpolator from wave FITS files.

    This is the main entry point, matching the interface of run_lsf_analysis().

    Parameters
    ----------
    file_paths : str or list
        Path(s) to wave FITS file(s)

    figdir : str, optional
        Directory for output figures. If None, no figures created.

    figname : str
        Base name for output figures (will be prefixed with file names)

    debug : bool
        Print detailed debug information

    make_plot : bool
        Create diagnostic plots

    wave_bin_width : float
        Wavelength bin width for outlier detection (Å)

    apply_bimodal_filtering : bool
        Apply bimodal filtering during file reading

    apply_residual_filtering : bool
        Apply residual filtering after spline fit

    spline_order : int
        Spline order (k parameter, 1-5). Default: 3 (cubic)

    spline_smoothing : float or None
        Smoothing parameter. None = auto (recommended)

    wave_grid_resolution : float
        Wavelength grid resolution for evaluation (Å)

    overwrite : bool
        If True, recreate even if pickle exists. If False, load from pickle if available.

    save_pickle : bool
        If True, save interpolator to pickle file

    pickle_dir : str, optional
        Directory for pickle file. If None, uses figdir or input file directory.

    Returns
    -------
    FWHMInterpolator : The created/loaded interpolator

    Examples
    --------
    # Single file (one arm)
    interp = run_fwhm_analysis(
        file_paths='<PYAPS_DATA>/CAL/wave_<runid>_all.fit',
        figdir='<OUTPUT_DIR>/plots',
        figname='blue_arm',
        debug=True
    )

    # Multiple files (two arms)
    interp = run_fwhm_analysis(
        file_paths=['<PYAPS_DATA>/CAL/wave_blue.fit', '<PYAPS_DATA>/CAL/wave_red.fit'],
        figdir='<OUTPUT_DIR>/plots',
        figname='combined',
        debug=True,
        overwrite=False  # Load from cache if available
    )

    # Get FWHM values
    global_fwhm = interp.get_fwhm(4500.0)
    fiber_fwhm = interp.get_fwhm(4500.0, specnum=123)
    """

    if isinstance(file_paths, str):
        file_paths = [file_paths]

    # Optionally replace binned files with unbinned versions
    if replace_binned:
        file_paths = check_and_replace_binned_files(file_paths)


    # Generate output headname
    headname = generate_output_headname(file_paths)

    # Determine pickle path
    if pickle_dir is not None:
        pickle_path = os.path.join(pickle_dir, f"{headname}{PICKLE_EXT}")
    elif figdir is not None:
        pickle_path = os.path.join(figdir, f"{headname}{PICKLE_EXT}")
    else:
        pickle_path = get_output_pickle_path(file_paths)

    # Check if we can load from pickle
    if not overwrite and os.path.exists(pickle_path):
        print(f"\n📂 Found existing pickle: {pickle_path}")
        print(f"   Loading from cache (use overwrite=True to recreate)")

        try:
            interpolator = FWHMInterpolator.load(pickle_path)
            # dill stores closures by bytecode: a cache written under another Python minor
            # version loads fine but fails when called (aps_lsf._lsf_cache_is_usable).
            from PyAPS.aps_lsf import _lsf_cache_is_usable
            cache_ok, cache_problem = _lsf_cache_is_usable(interpolator)
            if not cache_ok:
                raise RuntimeError(cache_problem)
            return interpolator
        except Exception as e:
            # Same class of failure as aps_lsf.py's identical guard (see its
            # own comment for the full story) - a load that raises outright
            # means the pickle is unreadable under the current Python/scipy/
            # dill, most often after an interpreter/library upgrade, NOT
            # that this target has insufficient calibration data. Those are
            # different problems and must not look the same in the logs -
            # flagged loudly here, then safely regenerated from source.
            print("!"*70)
            print("FWHM ANALYSIS - STALE/UNREADABLE CACHE (not a data problem)")
            print("!"*70)
            print(f"\n📂 {pickle_path}")
            print(f"   failed to load: {type(e).__name__}: {e}")
            print(f"   This is almost always a pickle written under a Python/scipy/dill")
            print(f"   version that has since changed its object format underneath it -")
            print(f"   NOT missing or insufficient calibration data. Regenerating from")
            print(f"   source now; delete this file to skip straight to a clean rebuild")
            print(f"   next time.")

    # Create new interpolator
    print(f"\n{'='*70}")
    print(f"FWHM ANALYSIS - Creating Interpolator from Wave Files")
    print(f"{'='*70}")

    interpolator = FWHMInterpolator()
    interpolator.set_debug(debug)

    try:
        success = interpolator.create_from_files(
            file_paths=file_paths,
            wave_bin_width=wave_bin_width,
            apply_bimodal_filtering=apply_bimodal_filtering,
            apply_residual_filtering=apply_residual_filtering,
            spline_order=spline_order,
            spline_smoothing=spline_smoothing,
            wave_grid_resolution=wave_grid_resolution
        )

        if not success:
            print(f"\n❌ Failed to create FWHM interpolator")
            return None

        # Save pickle
        if save_pickle:
            os.makedirs(os.path.dirname(pickle_path) if os.path.dirname(pickle_path) else '.', exist_ok=True)
            interpolator.save(pickle_path)

        # Create diagnostic plots
        if make_plot and figdir:
            prefixed_figname = f"{headname}_{figname}"
            create_diagnostic_plots(interpolator, figdir, prefixed_figname)

        return interpolator

    except Exception as e:
        print(f"\n❌ ERROR in run_fwhm_analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# RUNNABLE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    # DEMO settings: edit for your setup. Replace the <PYAPS_DATA>, <PYAPS_DIR>, <night>, <runid>, <obid>
    # markers below with your own locations and identifiers (no machine paths belong in this repository).

    print("\n" + "="*80)
    print("🎯 FWHM INTERPOLATOR - RUNNABLE EXAMPLES")
    print("="*80)

    # =========================================================================
    # INPUT FILES - UPDATE THESE FOR YOUR DATA
    # =========================================================================
    single_file = "<PYAPS_DATA>/CAL/<night>/wave_<runid>_all.fit"
    blue_file = "<PYAPS_DATA>/CAL/<night>/wave_<runid>_all.fit"
    red_file = "<PYAPS_DATA>/CAL/<night>/wave_<runid>_all.fit"
    output_dir = "<PYAPS_DATA>/L2<env_suffix>/fwhm_test"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # =========================================================================
    # EXAMPLE 1: SINGLE ARM (One File)
    # =========================================================================
    print("\n" + "="*80)
    print("EXAMPLE 1: SINGLE ARM (One File)")
    print("="*80)

    interp_single = run_fwhm_analysis(
        # =====================================================================
        # INPUT FILES
        # =====================================================================
        file_paths=single_file,           # Single wave FITS file (str or list)

        # =====================================================================
        # OUTPUT OPTIONS
        # =====================================================================
        figdir=output_dir,                # Directory for output plots (None = no plots)
        figname="single_arm",             # Base name for plot files (prefixed with input names)
        debug=True,                       # Print detailed debug information
        make_plot=True,                   # Create diagnostic plots

        # =====================================================================
        # FILTERING OPTIONS
        # =====================================================================
        wave_bin_width=50,                # Wavelength bin width for outlier detection (Å)
                                          # Smaller = more local detection, larger = more global
        apply_bimodal_filtering=False,    # Apply GMM bimodal filtering during file reading
                                          # Usually False - wavelength binning handles this
        apply_residual_filtering=True,    # Filter outliers in spline residuals after fit
                                          # Usually True - cleans up bad spline fits

        # =====================================================================
        # SPLINE OPTIONS
        # =====================================================================
        spline_order=3,                   # Spline order (k parameter):
                                          #   k=1: Linear (piecewise, has corners)
                                          #   k=2: Quadratic (smooth 1st derivative)
                                          #   k=3: Cubic (RECOMMENDED - C² continuous)
                                          #   k=4: Quartic (very smooth, can oscillate)
                                          #   k=5: Quintic (prone to Runge's phenomenon)
        spline_smoothing=None,            # Smoothing parameter (s):
                                          #   None: Auto-smoothing (RECOMMENDED)
                                          #   0: Interpolating (passes through all points)
                                          #   >0: Smoothing (larger = smoother)
        wave_grid_resolution=1.0,         # Wavelength grid resolution for evaluation (Å)
                                          # Smaller = finer grid, larger = coarser

        # =====================================================================
        # CACHING OPTIONS
        # =====================================================================
        overwrite=True,                   # If True: always recreate interpolator
                                          # If False: load from pickle if exists
        save_pickle=True,                 # Save interpolator to pickle file
        pickle_dir=None                   # Directory for pickle file
                                          # None = use figdir, or input file directory
    )

    if interp_single is not None:
        print("\n📊 Testing Single Arm Interpolator:")

        # Test wavelengths
        test_waves = np.array([4000.0, 4500.0, 5000.0, 5500.0])

        # Get global FWHM (median of all fibers)
        global_fwhm = interp_single.get_fwhm(test_waves)
        print(f"  Global FWHM at {test_waves}: {global_fwhm}")

        # Test FLAT extrapolation (before and after data range)
        print(f"\n  Testing FLAT extrapolation:")
        wave_range = interp_single.all_wave_ranges[0]
        print(f"    Data range: {wave_range[0]:.1f} - {wave_range[1]:.1f} Å")

        before_val = interp_single.get_fwhm(wave_range[0] - 500)  # 500 Å before
        at_start = interp_single.get_fwhm(wave_range[0])
        after_val = interp_single.get_fwhm(wave_range[1] + 500)   # 500 Å after
        at_end = interp_single.get_fwhm(wave_range[1])

        print(f"    FWHM at start ({wave_range[0]:.0f} Å): {at_start:.4f}")
        print(f"    FWHM 500 Å BEFORE: {before_val:.4f} (should equal start → FLAT)")
        print(f"    FWHM at end ({wave_range[1]:.0f} Å): {at_end:.4f}")
        print(f"    FWHM 500 Å AFTER: {after_val:.4f} (should equal end → FLAT)")

        # Get fiber-specific FWHM
        available_fibers = interp_single.get_available_fibers()
        if len(available_fibers) > 0:
            test_fiber = available_fibers[0]
            fiber_fwhm = interp_single.get_fwhm(test_waves, specnum=test_fiber)
            print(f"  Fiber {test_fiber} FWHM at {test_waves}: {fiber_fwhm}")

        # Get interpolator dict (for pipeline integration)
        interp_dict = interp_single.get_interpolator_dict()
        print(f"  Available fibers: {len(available_fibers)}")
        print(f"  Interpolator dict keys: 'global' + {len(available_fibers)} fibers")

    # =========================================================================
    # EXAMPLE 2: TWO ARMS (Two Files with Gap)
    # =========================================================================
    print("\n" + "="*80)
    print("EXAMPLE 2: TWO ARMS (Two Files with Gap)")
    print("="*80)

    interp_double = run_fwhm_analysis(
        # =====================================================================
        # INPUT FILES - List of two files (blue + red arms)
        # =====================================================================
        file_paths=[blue_file, red_file],  # List of wave FITS files

        # =====================================================================
        # OUTPUT OPTIONS
        # =====================================================================
        figdir=output_dir,                # Directory for output plots
        figname="two_arms",               # Base name for plot files
        debug=True,                       # Print detailed debug information
        make_plot=True,                   # Create diagnostic plots

        # =====================================================================
        # FILTERING OPTIONS
        # =====================================================================
        wave_bin_width=50,                # Wavelength bin width for outlier detection (Å)
        apply_bimodal_filtering=False,    # Apply GMM bimodal filtering
        apply_residual_filtering=True,    # Filter outliers in spline residuals

        # =====================================================================
        # SPLINE OPTIONS
        # =====================================================================
        spline_order=3,                   # Cubic splines (recommended)
        spline_smoothing=None,            # Auto-smoothing (recommended)
        wave_grid_resolution=1.0,         # 1Å grid resolution

        # =====================================================================
        # CACHING OPTIONS
        # =====================================================================
        overwrite=True,                   # Force recreate (set False to use cache)
        save_pickle=True,                 # Save to pickle
        pickle_dir=None,                   # Use figdir for pickle
        replace_binned=False               # Replace binned files with unbinned versions
    )

    if interp_double is not None:
        print("\n📊 Testing Two-Arm Interpolator:")

        # Show arm ranges
        print(f"\n  Arm ranges:")
        print(f"    Arm 1: {interp_double.all_wave_ranges[0][0]:.1f} - {interp_double.all_wave_ranges[0][1]:.1f} Å")
        print(f"    Arm 2: {interp_double.all_wave_ranges[1][0]:.1f} - {interp_double.all_wave_ranges[1][1]:.1f} Å")

        # Test wavelengths in arm regions
        arm1_min, arm1_max = interp_double.all_wave_ranges[0]
        arm2_min, arm2_max = interp_double.all_wave_ranges[1]

        test_arm1 = np.linspace(arm1_min + 100, arm1_max - 100, 4)
        test_arm2 = np.linspace(arm2_min + 100, arm2_max - 100, 4)

        print(f"\n  Arm 1 region (SPLINE):")
        global_fwhm_arm1 = interp_double.get_fwhm(test_arm1)
        print(f"    Wavelengths: {test_arm1}")
        print(f"    Global FWHM: {global_fwhm_arm1}")

        # Check for GAP or OVERLAP
        if interp_double.has_gap:
            print(f"\n  GAP region ({interp_double.gap_start:.1f} - {interp_double.gap_end:.1f} Å):")
            print(f"    → Uses LINEAR interpolation between arms")
            test_gap = np.linspace(interp_double.gap_start + 10, interp_double.gap_end - 10, 5)
            global_fwhm_gap = interp_double.get_fwhm(test_gap)
            print(f"    Wavelengths: {test_gap}")
            print(f"    Global FWHM: {global_fwhm_gap}")

            # Verify linear interpolation
            gap_fwhm_start = interp_double.get_fwhm(interp_double.gap_start)
            gap_fwhm_end = interp_double.get_fwhm(interp_double.gap_end)
            gap_mid = (interp_double.gap_start + interp_double.gap_end) / 2
            gap_fwhm_mid = interp_double.get_fwhm(gap_mid)
            expected_mid = (gap_fwhm_start + gap_fwhm_end) / 2
            print(f"    Linear check at mid-gap ({gap_mid:.1f} Å): {gap_fwhm_mid:.4f} (expected: {expected_mid:.4f})")

        elif interp_double.has_overlap:
            print(f"\n  OVERLAP region ({interp_double.overlap_start:.1f} - {interp_double.overlap_end:.1f} Å):")
            print(f"    → Uses LINEAR interpolation (same as gap)")
            test_overlap = np.linspace(interp_double.overlap_start + 10, interp_double.overlap_end - 10, 5)
            global_fwhm_overlap = interp_double.get_fwhm(test_overlap)
            print(f"    Wavelengths: {test_overlap}")
            print(f"    Global FWHM: {global_fwhm_overlap}")

            # Verify linear interpolation
            ovl_fwhm_start = interp_double.get_fwhm(interp_double.overlap_start)
            ovl_fwhm_end = interp_double.get_fwhm(interp_double.overlap_end)
            ovl_mid = (interp_double.overlap_start + interp_double.overlap_end) / 2
            ovl_fwhm_mid = interp_double.get_fwhm(ovl_mid)
            expected_mid = (ovl_fwhm_start + ovl_fwhm_end) / 2
            print(f"    Linear check at mid-overlap ({ovl_mid:.1f} Å): {ovl_fwhm_mid:.4f} (expected: {expected_mid:.4f})")

        print(f"\n  Arm 2 region (SPLINE):")
        global_fwhm_arm2 = interp_double.get_fwhm(test_arm2)
        print(f"    Wavelengths: {test_arm2}")
        print(f"    Global FWHM: {global_fwhm_arm2}")

        # Test FLAT extrapolation beyond data range
        print(f"\n  FLAT extrapolation (outside data range):")

        # Before arm 1
        before_val = interp_double.get_fwhm(arm1_min - 500)
        at_arm1_start = interp_double.get_fwhm(arm1_min)
        print(f"    At arm1 start ({arm1_min:.0f} Å): {at_arm1_start:.4f}")
        print(f"    500 Å BEFORE arm1: {before_val:.4f} (should equal arm1 start → FLAT)")

        # After arm 2
        after_val = interp_double.get_fwhm(arm2_max + 500)
        at_arm2_end = interp_double.get_fwhm(arm2_max)
        print(f"    At arm2 end ({arm2_max:.0f} Å): {at_arm2_end:.4f}")
        print(f"    500 Å AFTER arm2: {after_val:.4f} (should equal arm2 end → FLAT)")

        # Test fiber-specific
        available_fibers = interp_double.get_available_fibers()
        if len(available_fibers) >= 3:
            mid_wave = (arm1_min + arm2_max) / 2
            print(f"\n  Fiber-specific FWHM at {mid_wave:.0f} Å:")
            for fib in available_fibers[:3]:
                fib_fwhm = interp_double.get_fwhm(mid_wave, specnum=fib)
                global_fwhm = interp_double.get_fwhm(mid_wave)
                diff = fib_fwhm - global_fwhm
                print(f"    Fiber {fib}: {fib_fwhm:.4f} (diff from global: {diff:+.4f})")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("✅ EXAMPLES COMPLETE")
    print("="*80)

    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")

    # List generated files
    import glob as glob_module
    for f in sorted(glob_module.glob(os.path.join(output_dir, "*"))):
        print(f"  - {os.path.basename(f)}")

    print("\n" + "="*80)
    print("USAGE SUMMARY")
    print("="*80)
    print("""
    # Single arm:
    interp = run_fwhm_analysis(
        file_paths='/path/to/wave_file.fit',
        figdir='<OUTPUT_DIR>',
        debug=True
    )

    # Two arms (with gap handling):
    interp = run_fwhm_analysis(
        file_paths=['/path/to/blue.fit', '/path/to/red.fit'],
        figdir='<OUTPUT_DIR>',
        debug=True
    )

    # Load from cache:
    interp = run_fwhm_analysis(
        file_paths=[...],
        figdir='<OUTPUT_DIR>',
        overwrite=False  # Load from pickle if exists
    )

    # Direct load from pickle:
    interp = load_fwhm_interpolator('/path/to/file.dill')

    # Get FWHM values:
    global_fwhm = interp.get_fwhm(wavelength)
    fiber_fwhm = interp.get_fwhm(wavelength, specnum=123)

    # Get interpolator dict (for pipeline):
    interp_dict = interp.get_interpolator_dict()
    """)
