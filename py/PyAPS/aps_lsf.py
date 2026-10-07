#!/usr/bin/env python
"""
aps_lsf.py - LSF Reader for Pre-computed B-spline FITS Files

This module reads pre-computed B-spline LSF representations from FITS files
and provides the same interface as aps_fwhm.py for FWHM interpolation.

Key differences from aps_fwhm.py:
- No fitting required - reads pre-computed splines
- Uses BSpline objects directly from stored knots/coefficients
- Handles overlap between files by averaging
- Extrapolates outside range with flat last values

Author: APS Team
Date: December 2024
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.interpolate import BSpline, UnivariateSpline, interp1d
import warnings
import glob
from pathlib import Path


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

# Bumped whenever a change to LSFInterpolator's *closures* (the actual
# per-fibre/global/merged-arms interpolation functions built by
# _create_interpolator_dict/_merge_multiple_files) meaningfully changes
# their behaviour or performance — most recently the 2026-08-05
# vectorization fix (per-point Python loops -> single vectorized
# np.where/BSpline calls, ~20-30x faster per call). dill.dump(self, ...)
# serializes those closures' *actual code objects*, frozen at whatever
# they were when the file was saved — so an existing .dill built before a
# fix like that keeps running the old, slow closure forever, completely
# unaffected by any later source-code change, until the file itself is
# regenerated. Stamping every newly-built interpolator with this version
# (see LSFInterpolator.__init__) and checking it on load (see
# run_lsf_analysis) makes that regeneration automatic — a stale-format
# cache is detected and rebuilt once, transparently, instead of silently
# staying slow forever regardless of what the source code says. Confirmed
# necessary, not theoretical: a real dataset's cached
# lsf/20250909/{BLUE,RED}L11_LIFU.dill (built Feb 2026, ~6 months before
# the fix) measured ~7.8ms/interpolator-call post-fix-deployment — the
# fix's own ~106µs/call benchmark never actually reached this dataset
# because nothing was invalidating the stale cache.
_LSF_CACHE_FORMAT_VERSION = 2

# =============================================================================
# HELPER FUNCTIONS
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
        print(f"  ⚠️  WARNING: {warning_msg.strip()}", file=sys.stderr)
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


def parse_lsf_filename(filename):
    """
    Parse LSF filename to extract setup information.

    Example: "lsf_BLUEL11_MOS-A.fits" -> {'camera': 'BLUE', 'res': 'L', 'bins': '11', 'mode': 'MOS-A'}

    Parameters
    ----------
    filename : str
        LSF filename

    Returns
    -------
    dict : Parsed setup information
    """
    basename = os.path.basename(filename)

    # Remove prefix and suffix
    if basename.startswith('lsf_'):
        basename = basename.replace('lsf_', '')
    if basename.endswith('.fits'):
        basename = basename.replace('.fits', '')

    # Split by underscore
    parts = basename.split('_')

    parsed = {}

    if len(parts) >= 1:
        # Parse setup string (e.g., "BLUEL11")
        setup = parts[0]

        # Extract camera (BLUE, RED, GREEN)
        if 'BLUE' in setup:
            parsed['camera'] = 'BLUE'
            setup = setup.replace('BLUE', '')
        elif 'RED' in setup:
            parsed['camera'] = 'RED'
            setup = setup.replace('RED', '')
        elif 'GREEN' in setup:
            parsed['camera'] = 'GREEN'
            setup = setup.replace('GREEN', '')
        else:
            parsed['camera'] = 'UNKNOWN'

        # Extract resolution (H or L)
        if 'H' in setup:
            parsed['res'] = 'H'
            setup = setup.replace('H', '')
        elif 'L' in setup:
            parsed['res'] = 'L'
            setup = setup.replace('L', '')
        else:
            parsed['res'] = 'UNKNOWN'

        # Remaining is binning (e.g., "11")
        parsed['bins'] = setup if setup else 'UNKNOWN'
        parsed['setup'] = parts[0]

    if len(parts) >= 2:
        # Mode (MOS-A, MOS-B, LIFU, mIFU)
        parsed['mode'] = parts[1]

    return parsed


def generate_output_headname(file_paths):
    """
    Generate output headname based on input LSF FITS files.

    Naming convention:
    - Single file: {setup}_{mode}
      e.g., lsf_GREENH11_MOS-A.fits -> GREENH11_MOS-A

    - Multiple files (same mode): {setup1}__{setup2}_{mode}
      e.g., lsf_GREENH11_MOS-A.fits + lsf_REDH11_MOS-A.fits -> GREENH11__REDH11_MOS-A
      e.g., lsf_BLUEH11_LIFU.fits + lsf_REDH11_LIFU.fits -> BLUEH11__REDH11_LIFU

    Parameters
    ----------
    file_paths : str or list
        Single file path or list of file paths

    Returns
    -------
    str : Output headname
    """
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    if not file_paths:
        return 'lsf_output'

    # Parse all files
    parsed_files = []
    for fpath in file_paths:
        parsed = parse_lsf_filename(fpath)
        parsed_files.append(parsed)

    if len(parsed_files) == 1:
        # Single file: {setup}_{mode}
        p = parsed_files[0]
        setup = p.get('setup', 'UNKNOWN')
        mode = p.get('mode', 'UNKNOWN')
        return f"{setup}_{mode}"

    else:
        # Multiple files: {setup1}__{setup2}_{mode}
        # Assume all files have the same mode (they should for merging)
        setups = [p.get('setup', 'UNKNOWN') for p in parsed_files]
        modes = [p.get('mode', 'UNKNOWN') for p in parsed_files]

        # Use the first mode found (should all be the same)
        mode = modes[0]

        # Check if all modes match
        if len(set(modes)) > 1:
            warnings.warn(f"Files have different modes: {modes}. Using first mode: {mode}")

        # Join setups with double underscore
        setup_str = '__'.join(setups)

        return f"{setup_str}_{mode}"


def get_output_pickle_path(file_paths, output_dir=None):
    """
    Get the full path for the output pickle file.

    Parameters
    ----------
    file_paths : str or list
        Input LSF FITS file path(s)
    output_dir : str, optional
        Output directory. If None, uses the directory of the first input file.

    Returns
    -------
    str : Full path to pickle file
    """
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    # Determine output directory
    if output_dir is None:
        # Use the directory of the first input file
        output_dir = os.path.dirname(os.path.abspath(file_paths[0]))

    # Generate headname
    headname = generate_output_headname(file_paths)

    # Create full path
    pickle_filename = f"{headname}{PICKLE_EXT}"
    pickle_path = os.path.join(output_dir, pickle_filename)

    return pickle_path


# =============================================================================
# LSF FILE READER
# =============================================================================

def read_lsf_splines_file(lsf_filepath, debug=False):
    """
    Read pre-computed LSF B-splines from FITS file.

    Parameters
    ----------
    lsf_filepath : str
        Path to LSF FITS file (e.g., lsf_BLUEL11_MOS-A.fits)
    debug : bool
        Print debug information

    Returns
    -------
    fiber_splines : dict
        Dictionary with fiber-specific splines:
        {specnum: {'bspline': BSpline, 'knots': t, 'coeffs': c, 'degree': k, ...}}
    metadata : dict
        File metadata
    wave_range : tuple
        (wave_min, wave_max) across all fibers
    """

    # =========================================================================
    # FILE EXISTENCE CHECKS
    # =========================================================================

    # Check if path exists
    if not os.path.exists(lsf_filepath):
        raise FileNotFoundError(
            f"LSF file not found: {lsf_filepath}\n"
            f"Please check the file path and ensure the file exists."
        )

    # Check if it's a file (not a directory)
    if not os.path.isfile(lsf_filepath):
        raise ValueError(
            f"Path exists but is not a file: {lsf_filepath}\n"
            f"Please provide a path to a FITS file, not a directory."
        )

    # Check if file is readable
    if not os.access(lsf_filepath, os.R_OK):
        raise PermissionError(
            f"LSF file exists but is not readable: {lsf_filepath}\n"
            f"Please check file permissions."
        )

    # Check if it's a FITS file (by extension)
    if not lsf_filepath.lower().endswith(('.fits', '.fit', '.fits.gz', '.fit.gz')):
        warnings.warn(
            f"File does not have a standard FITS extension: {lsf_filepath}\n"
            f"Expected .fits, .fit, .fits.gz, or .fit.gz"
        )

    # Check file size (warn if suspiciously small)
    file_size = os.path.getsize(lsf_filepath)
    if file_size < 1000:  # Less than 1 KB
        warnings.warn(
            f"LSF file is suspiciously small ({file_size} bytes): {lsf_filepath}\n"
            f"File may be corrupted or empty."
        )

    if debug:
        print(f"\n📖 Reading LSF file: {os.path.basename(lsf_filepath)}")
        print(f"   Path: {lsf_filepath}")
        print(f"   Size: {file_size / 1024:.1f} KB")

    # =========================================================================
    # READ FITS FILE
    # =========================================================================

    try:
        # Read FITS file
        with fits.open(lsf_filepath) as hdul:

            # Get metadata from primary header
            primary_hdr = hdul[0].header
            metadata = {
                'CALDATE': primary_hdr.get('CALDATE', ''),
                'L1_REF': primary_hdr.get('L1_REF', ''),
                'FPMODE': primary_hdr.get('FPMODE', ''),
                'CAMERA': primary_hdr.get('CAMERA', ''),
                'VPH': primary_hdr.get('VPH', ''),
                'CCDXBIN': primary_hdr.get('CCDXBIN', 1),
                'LSFDATE': primary_hdr.get('LSFDATE', ''),
                'filepath': lsf_filepath,
                'filename': os.path.basename(lsf_filepath)
            }

            # Parse filename for setup info
            parsed = parse_lsf_filename(lsf_filepath)
            metadata.update(parsed)

            # Check if HDU 1 exists
            if len(hdul) < 2:
                raise ValueError(
                    f"FITS file does not contain expected HDU structure.\n"
                    f"Expected at least 2 HDUs, found {len(hdul)}."
                )

            # Read LSF splines table
            lsf_table = hdul[1].data

            # Check if required columns exist
            required_columns = ['NSPEC', 't', 'c', 'k']
            missing_columns = [col for col in required_columns if col not in lsf_table.names]
            if missing_columns:
                raise ValueError(
                    f"FITS table missing required columns: {missing_columns}\n"
                    f"Available columns: {lsf_table.names}"
                )

            nspec_arr = lsf_table['NSPEC']
            t_arr = lsf_table['t']        # Knot vectors (12 elements each)
            c_arr = lsf_table['c']        # Coefficients (12 elements each)
            k_arr = lsf_table['k']        # Spline degrees (all should be 5)

            n_fibers = len(nspec_arr)

            if n_fibers == 0:
                raise ValueError(f"LSF file contains no fiber data: {lsf_filepath}")

            if debug:
                print(f"  ✓ Fibers: {n_fibers}")
                print(f"  ✓ NSPEC range: {np.min(nspec_arr)} - {np.max(nspec_arr)}")
                print(f"  ✓ Wavelength range: {np.min(t_arr):.1f} - {np.max(t_arr):.1f} Å")
                print(f"  ✓ Spline degree: k={k_arr[0]}")

    except OSError as e:
        raise IOError(
            f"Failed to open FITS file: {lsf_filepath}\n"
            f"Error: {e}\n"
            f"File may be corrupted or not a valid FITS file."
        )
    except Exception as e:
        raise RuntimeError(
            f"Error reading LSF file: {lsf_filepath}\n"
            f"Error: {e}"
        )

    # =========================================================================
    # CREATE BSPLINE OBJECTS
    # =========================================================================

    # Create BSpline objects for each fiber
    fiber_splines = {}

    wave_min_global = np.inf
    wave_max_global = -np.inf

    n_failed = 0
    for i, nspec in enumerate(nspec_arr):
        try:
            t = t_arr[i]
            c = c_arr[i]
            k = int(k_arr[i])

            # Check for valid data
            if np.any(np.isnan(t)) or np.any(np.isnan(c)):
                if debug and n_failed < 3:
                    print(f"  ⚠️  Warning: Fiber NSPEC={nspec} has NaN values, skipping")
                n_failed += 1
                continue

            # Create BSpline object
            bspl = BSpline(t, c, k, extrapolate=False)

            # Store
            fiber_splines[int(nspec)] = {
                'bspline': bspl,
                'knots': t,
                'coeffs': c,
                'degree': k,
                'wave_range': (float(np.min(t)), float(np.max(t))),
                'n_knots': len(t)
            }

            # Update global range
            wave_min_global = min(wave_min_global, np.min(t))
            wave_max_global = max(wave_max_global, np.max(t))

        except Exception as e:
            if debug and n_failed < 3:
                print(f"  ⚠️  Warning: Failed to process fiber NSPEC={nspec}: {e}")
            n_failed += 1
            continue

    if len(fiber_splines) == 0:
        raise ValueError(
            f"Failed to create any fiber splines from file: {lsf_filepath}\n"
            f"All {n_fibers} fibers failed processing."
        )

    if n_failed > 0 and debug:
        print(f"  ⚠️  Failed to process {n_failed}/{n_fibers} fibers")

    wave_range = (wave_min_global, wave_max_global)

    if debug:
        print(f"  ✓ Created {len(fiber_splines)} fiber splines")
        print(f"  ✓ Global wavelength range: {wave_range[0]:.1f} - {wave_range[1]:.1f} Å")

    return fiber_splines, metadata, wave_range


# =============================================================================
# LSF INTERPOLATOR CLASS
# =============================================================================

class LSFInterpolator:
    """
    LSF Interpolator for pre-computed B-spline FITS files.

    Provides same interface as EnhancedFWHMInterpolator from aps_fwhm.py
    """

    def __init__(self):
        self.fiber_splines = None
        self.global_spline = None
        self.interpolator_dict = None
        self.wave_grid = None
        self.file_prefix = ''
        self.metadata = {}
        self.debug = False
        self.wave_ranges = []  # Store individual file ranges
        self.input_files = []  # Store input file paths for reference
        # Stamped at construction time (so it reflects whatever code is
        # actually running right now, not just when this attribute was
        # added) and pickled along with everything else — see
        # _LSF_CACHE_FORMAT_VERSION's own comment for why this exists.
        self._cache_format_version = _LSF_CACHE_FORMAT_VERSION

    def set_debug(self, debug):
        """Set debug flag."""
        if debug is not None:
            self.debug = debug

    def create_from_files(self, file_paths, wave_grid_resolution=1.0, smooth_length=None, kernel_type='gaussian'):
        """
        Create LSF interpolator from pre-computed LSF FITS files.

        Parameters
        ----------
        file_paths : str or list
            Single file or list of LSF FITS files (e.g., for blue+red)
        wave_grid_resolution : float
            Wavelength resolution for evaluation grid (Angstroms)

        Returns
        -------
        bool : Success status
        """

        if isinstance(file_paths, str):
            file_paths = [file_paths]

        # Store input files for reference
        self.input_files = [os.path.abspath(f) for f in file_paths]

        if self.debug:
            print("="*70)
            print("LSF INTERPOLATOR - Reading Pre-computed Splines")
            print(f"  Files: {len(file_paths)}")
            print("="*70)

        # try:
        # Read all files
        all_fiber_splines = []
        all_metadata = []
        all_wave_ranges = []

        for fpath in file_paths:
            fiber_splines, metadata, wave_range = read_lsf_splines_file(fpath, debug=self.debug)
            all_fiber_splines.append(fiber_splines)
            all_metadata.append(metadata)
            all_wave_ranges.append(wave_range)

        self.wave_ranges = all_wave_ranges

        # Handle single vs multiple files
        if len(file_paths) == 1:
            # Single file - use directly
            self.fiber_splines = all_fiber_splines[0]
            self.metadata = all_metadata[0]
            wave_min, wave_max = all_wave_ranges[0]

        else:
            # Multiple files - need to merge
            self.fiber_splines = self._merge_multiple_files(
                all_fiber_splines, all_wave_ranges, wave_grid_resolution
            )
            self.metadata = {'files': [m['filename'] for m in all_metadata]}
            wave_min = min(wr[0] for wr in all_wave_ranges)
            wave_max = max(wr[1] for wr in all_wave_ranges)

        # Create wavelength grid
        n_points = int((wave_max - wave_min) / wave_grid_resolution) + 1
        self.wave_grid = np.linspace(wave_min, wave_max, n_points)
        if self.debug:
            print(f"\n✓ Wavelength grid: {len(self.wave_grid)} points ({wave_min:.1f} - {wave_max:.1f} Å)")

        # Create global spline
        self._create_global_spline()

        # Create interpolator dictionary
        self._create_interpolator_dict()

        # Fill missing fibers
        self._fill_missing_fibers(max_nspec=None)

        # Apply smoothing if requested
        if smooth_length is not None:
            self.apply_smoothing(smooth_length=smooth_length, kernel_type=kernel_type)



        # Generate file prefix
        self.file_prefix = generate_file_prefix(file_paths)

        if self.debug:
            print(f"\n✅ LSF Interpolator created successfully!")
            self.print_summary()

        return True

        # except Exception as e:
        #     print(f"\n❌ ERROR creating LSF interpolator: {e}")
        #     import traceback
        #     traceback.print_exc()
        #     return False

    def _merge_multiple_files(self, all_fiber_splines, all_wave_ranges, wave_grid_resolution):
        """
        Merge multiple LSF files (e.g., blue + red arms).

        Strategy:
        - Before arm 1: Flat extrapolation with first value of arm 1
        - Gap between arms: LINEAR INTERPOLATION between arms (per-fiber boundaries)
        - Overlap region: AVERAGE the two profiles
        - After arm 2: Flat extrapolation with last value of arm 2

        Parameters
        ----------
        all_fiber_splines : list of dict
            List of fiber_splines from each file
        all_wave_ranges : list of tuple
            List of (wave_min, wave_max) for each file
        wave_grid_resolution : float
            Grid resolution for evaluation

        Returns
        -------
        merged_fiber_splines : dict
            Merged splines for each fiber
        """

        if self.debug:
            print(f"\n🔗 Merging {len(all_fiber_splines)} LSF files...")

        n_files = len(all_fiber_splines)

        # Find overall wavelength range
        wave_min_global = min(wr[0] for wr in all_wave_ranges)
        wave_max_global = max(wr[1] for wr in all_wave_ranges)

        # Detect overlap for 2 files
        has_overlap = False
        overlap_start = None
        overlap_end = None
        gap_start = None
        gap_end = None

        if n_files == 2:
            wave1_min, wave1_max = all_wave_ranges[0]
            wave2_min, wave2_max = all_wave_ranges[1]

            # Check for overlap
            overlap_start = max(wave1_min, wave2_min)
            overlap_end = min(wave1_max, wave2_max)
            has_overlap = overlap_end > overlap_start

            if has_overlap:
                if self.debug:
                    print(f"  ✓ Overlap detected: {overlap_start:.1f} - {overlap_end:.1f} Å")
            else:
                # Gap between arms
                gap_start = wave1_max
                gap_end = wave2_min
                if self.debug:
                    print(f"  ✓ Gap detected: {gap_start:.1f} - {gap_end:.1f} Å (will interpolate)")
                    print(f"  DEBUG: wave1_max={wave1_max:.1f}, wave2_min={wave2_min:.1f}")
                    print(f"  DEBUG: has_overlap={has_overlap}")

                    # # Check how many points are in gap
                    # gap_test_mask = (wave_eval > wave1_max) & (wave_eval < wave2_min)
                    # print(f"  DEBUG: Number of points in gap: {np.sum(gap_test_mask)}")
                    # if np.sum(gap_test_mask) > 0:
                    #     print(f"  DEBUG: Gap wavelength range in grid: {wave_eval[gap_test_mask][0]:.1f} - {wave_eval[gap_test_mask][-1]:.1f}")

        # Find common fiber IDs
        fiber_ids_sets = [set(fs.keys()) for fs in all_fiber_splines]
        common_fiber_ids = sorted(set.intersection(*fiber_ids_sets))

        if self.debug:
            print(f"  ✓ Common fibers: {len(common_fiber_ids)}")

        # Create evaluation grid (high resolution for smooth merging)
        n_points = int((wave_max_global - wave_min_global) / wave_grid_resolution) + 1
        wave_eval = np.linspace(wave_min_global, wave_max_global, n_points)

        # Merge each fiber
        merged_fiber_splines = {}

        for fib_idx, fib_id in enumerate(common_fiber_ids):

            # Get FWHM from each file evaluated on the FULL grid
            fwhm_values_per_file = []

            # for file_idx in range(n_files):
            #     fiber_data = all_fiber_splines[file_idx][fib_id]
            #     bspl = fiber_data['bspline']
            #     wave_min_file, wave_max_file = all_wave_ranges[file_idx]

            #     # Get boundary values for flat extrapolation
            #     first_val = float(bspl(wave_min_file))
            #     last_val = float(bspl(wave_max_file))

            #     # Evaluate at ALL points in wave_eval with manual extrapolation
            #     fwhm_full = np.zeros_like(wave_eval, dtype=float)

            #     for i, w in enumerate(wave_eval):
            #         if w < wave_min_file:
            #             fwhm_full[i] = first_val
            #         elif w > wave_max_file:
            #             fwhm_full[i] = last_val
            #         else:
            #             fwhm_full[i] = float(bspl(w))

            #     valid_mask = (wave_eval >= wave_min_file) & (wave_eval <= wave_max_file)

            #     fwhm_values_per_file.append({
            #         'fwhm': fwhm_full,
            #         'valid': valid_mask,
            #         'wave_range': (wave_min_file, wave_max_file),
            #         'first_val': first_val,
            #         'last_val': last_val,
            #         'wave_eval': wave_eval.copy()  # Store wave_eval for interpolation function
            #     })
            for file_idx in range(n_files):
                fiber_data = all_fiber_splines[file_idx][fib_id]
                bspl = fiber_data['bspline']

                # CRITICAL: Use per-fiber wave range, NOT global all_wave_ranges!
                # Each fiber's BSpline has its own knot range which may be narrower
                # than the global range. Using global range causes NaN when evaluating
                # outside the fiber's actual knot range (BSpline has extrapolate=False).
                wave_min_file, wave_max_file = fiber_data['wave_range']  # Per-fiber range!

                # Get boundary values - evaluate INSIDE the range, not exactly at boundaries
                # This avoids NaN from BSpline with extrapolate=False
                epsilon = 0.1  # Small offset

                # Safety check: ensure we're inside the knot range
                safe_min = wave_min_file + epsilon
                safe_max = wave_max_file - epsilon

                # If range is too narrow, use midpoint
                if safe_min >= safe_max:
                    mid = (wave_min_file + wave_max_file) / 2
                    safe_min = mid - epsilon/2
                    safe_max = mid + epsilon/2

                first_val = float(bspl(safe_min))
                last_val = float(bspl(safe_max))

                # If still NaN (shouldn't happen but safety check), use median of valid evaluations
                if np.isnan(first_val) or np.isnan(last_val):
                    # Evaluate at multiple points and take median of valid values
                    test_points = np.linspace(wave_min_file + 1, wave_max_file - 1, 10)
                    test_vals = np.array([float(bspl(p)) for p in test_points])
                    valid_vals = test_vals[~np.isnan(test_vals)]

                    if len(valid_vals) > 0:
                        if np.isnan(first_val):
                            first_val = float(valid_vals[0])
                            if self.debug and fib_idx < 3:
                                print(f"  Warning: Fiber {fib_id} first_val was NaN, using fallback")
                        if np.isnan(last_val):
                            last_val = float(valid_vals[-1])
                            if self.debug and fib_idx < 3:
                                print(f"  Warning: Fiber {fib_id} last_val was NaN, using fallback")
                    else:
                        # Absolute fallback - use a default value
                        if self.debug:
                            print(f"  ERROR: Fiber {fib_id} has no valid FWHM values!")
                        if np.isnan(first_val):
                            first_val = 2.5  # Reasonable default FWHM
                        if np.isnan(last_val):
                            last_val = 2.5

                # Evaluate at ALL points in wave_eval with manual extrapolation
                fwhm_full = np.zeros_like(wave_eval, dtype=float)

                for i, w in enumerate(wave_eval):
                    if w < wave_min_file:
                        fwhm_full[i] = first_val
                    elif w > wave_max_file:
                        fwhm_full[i] = last_val
                    else:
                        # For points inside range, try to evaluate spline
                        # If exactly at boundary, use boundary value
                        if abs(w - wave_min_file) < epsilon:
                            fwhm_full[i] = first_val
                        elif abs(w - wave_max_file) < epsilon:
                            fwhm_full[i] = last_val
                        else:
                            try:
                                val = float(bspl(w))
                                # Check for NaN - use boundary interpolation if NaN
                                if np.isnan(val):
                                    # Linear interpolation between boundaries
                                    alpha = (w - wave_min_file) / (wave_max_file - wave_min_file)
                                    fwhm_full[i] = (1 - alpha) * first_val + alpha * last_val
                                else:
                                    fwhm_full[i] = val
                            except:
                                # If spline fails, use linear interpolation from boundaries
                                if w < (wave_min_file + wave_max_file) / 2:
                                    fwhm_full[i] = first_val
                                else:
                                    fwhm_full[i] = last_val

                valid_mask = (wave_eval >= wave_min_file) & (wave_eval <= wave_max_file)

                fwhm_values_per_file.append({
                    'fwhm': fwhm_full,
                    'valid': valid_mask,
                    'wave_range': (wave_min_file, wave_max_file),
                    'first_val': first_val,
                    'last_val': last_val,
                    'wave_eval': wave_eval.copy()
                })

                # DEBUG: Check if boundary values are valid and show range comparison
                if self.debug and fib_idx < 3 and file_idx == 0:
                    global_min, global_max = all_wave_ranges[file_idx]
                    print(f"  DEBUG Fiber {fib_id} File {file_idx}: first_val={first_val:.4f}, last_val={last_val:.4f}")
                    if abs(wave_min_file - global_min) > 1 or abs(wave_max_file - global_max) > 1:
                        print(f"    Per-fiber range: {wave_min_file:.1f} - {wave_max_file:.1f} Å")
                        print(f"    Global range:    {global_min:.1f} - {global_max:.1f} Å")


            # Create merged FWHM array
            fwhm_merged = np.zeros_like(wave_eval)

            if n_files == 2:
                wave1_min, wave1_max = all_wave_ranges[0]
                wave2_min, wave2_max = all_wave_ranges[1]

                file0_data = fwhm_values_per_file[0]
                file1_data = fwhm_values_per_file[1]

                for i, w in enumerate(wave_eval):

                    # Region 1: Before arm 1 - flat with first value
                    if w < wave1_min:
                        fwhm_merged[i] = file0_data['first_val']

                    # Region 2: Arm 1 only (before overlap or gap)
                    elif w <= wave1_max and (not has_overlap or w < overlap_start):
                        fwhm_merged[i] = file0_data['fwhm'][i]

                    # Region 3: Gap between arms - LINEAR INTERPOLATION (per-fiber)
                    elif not has_overlap and wave1_max < w < wave2_min:
                        # Linear interpolation using THIS FIBER's boundary values
                        fwhm_at_gap_start = file0_data['last_val']
                        fwhm_at_gap_end = file1_data['first_val']

                        # DEBUG for first fiber, first gap point
                        if fib_idx == 0 and i > 0 and wave_eval[i-1] <= wave1_max and w > wave1_max:
                            print(f"  DEBUG Fiber {fib_id}: First gap point at w={w:.1f}")
                            print(f"    has_overlap={has_overlap}")
                            print(f"    wave1_max={wave1_max:.1f}, wave2_min={wave2_min:.1f}")
                            print(f"    Condition: {not has_overlap} and {wave1_max < w} and {w < wave2_min}")
                            print(f"    Gap boundaries: {fwhm_at_gap_start:.4f} -> {fwhm_at_gap_end:.4f}")



                        # Linear interpolation
                        alpha = (w - wave1_max) / (wave2_min - wave1_max)
                        fwhm_merged[i] = (1 - alpha) * fwhm_at_gap_start + alpha * fwhm_at_gap_end

                    # Region 4: Overlap - average
                    elif has_overlap and overlap_start <= w <= overlap_end:
                        fwhm_merged[i] = (file0_data['fwhm'][i] + file1_data['fwhm'][i]) / 2.0

                    # Region 5: Arm 2 only (after overlap or gap)
                    elif w >= wave2_min and (not has_overlap or w > overlap_end):
                        fwhm_merged[i] = file1_data['fwhm'][i]

                    # Region 6: After arm 2 - flat with last value
                    elif w > wave2_max:
                        fwhm_merged[i] = file1_data['last_val']

                    else:
                        # Fallback (should not happen)
                        fwhm_merged[i] = file0_data['fwhm'][i]

            else:
                # More than 2 files - simple average where valid
                fwhm_sum = np.zeros_like(wave_eval)
                valid_count = np.zeros_like(wave_eval, dtype=int)

                for file_data in fwhm_values_per_file:
                    fwhm_sum += file_data['fwhm']
                    valid_count[file_data['valid']] += 1

                # Average where we have data (at least one valid)
                mask_with_data = valid_count > 0
                fwhm_merged[mask_with_data] = fwhm_sum[mask_with_data] / np.maximum(valid_count[mask_with_data], 1)

                # Flat extrapolation for edges
                if np.any(~mask_with_data):
                    first_valid_idx = np.where(mask_with_data)[0][0]
                    last_valid_idx = np.where(mask_with_data)[0][-1]

                    if first_valid_idx > 0:
                        fwhm_merged[:first_valid_idx] = fwhm_merged[first_valid_idx]
                    if last_valid_idx < len(fwhm_merged) - 1:
                        fwhm_merged[last_valid_idx+1:] = fwhm_merged[last_valid_idx]

            # Debug: Print first few fibers to verify they're different
            if self.debug and fib_idx < 3:
                print(f"  Fiber {fib_id}: FWHM range = {np.nanmin(fwhm_merged):.4f} - {np.nanmax(fwhm_merged):.4f}")

                # Show gap values if exists
                if n_files == 2 and not has_overlap:
                    gap_mask = (wave_eval > wave1_max) & (wave_eval < wave2_min)
                    if np.any(gap_mask):
                        gap_vals = fwhm_merged[gap_mask]
                        print(f"    Gap FWHM range: {np.min(gap_vals):.4f} - {np.max(gap_vals):.4f}")

            # Fit spline to merged FWHM (for reference, but won't use in gap)
            # The spline is only used for smooth evaluation in non-gap regions
            merged_spline = UnivariateSpline(wave_eval, fwhm_merged, s=0, k=3)

            # Define closure function for creating interpolation functions
            def make_merged_func(file0_data, file1_data, wave1_min, wave1_max, wave2_min, wave2_max,
                                overlap_start, overlap_end, has_overlap):
                """
                Create merged interpolation function with proper variable capture.

                This function uses LINEAR interpolation in the gap region (no spline).
                """

                # Create a copy of the data at function creation time
                f0_fwhm = file0_data['fwhm'].copy()
                f0_first = float(file0_data['first_val'])
                f0_last = float(file0_data['last_val'])

                f1_fwhm = file1_data['fwhm'].copy()
                f1_first = float(file1_data['first_val'])
                f1_last = float(file1_data['last_val'])

                # Capture wave_eval which defines the grid we evaluated on
                wave_eval_copy = file0_data['wave_eval'].copy()

                def interp_func(wavelengths,
                            _f0_fwhm=f0_fwhm,
                            _f0_first=f0_first,
                            _f0_last=f0_last,
                            _f1_fwhm=f1_fwhm,
                            _f1_first=f1_first,
                            _f1_last=f1_last,
                            _wave_eval=wave_eval_copy,
                            _w1_min=wave1_min,
                            _w1_max=wave1_max,
                            _w2_min=wave2_min,
                            _w2_max=wave2_max,
                            _ovlp_start=overlap_start,
                            _ovlp_end=overlap_end,
                            _has_ovlp=has_overlap):
                    """
                    Interpolate merged FWHM with captured parameters.

                    Gap region uses LINEAR interpolation (not spline).

                    Vectorized (was: a plain Python `for` loop over every
                    wavelength point, same anti-pattern as the other two
                    interp_func closures in this file — see
                    global_interp_func's comment for the general story;
                    this one is only exercised when join_arms=True, i.e.
                    when calibration files spanning a wavelength
                    gap/overlap get stitched together). Confirmed
                    bit-identical output against the old per-point loop
                    (max abs diff 0.0, including NaN-pattern agreement)
                    across 20,000+ random points plus every region's exact
                    boundary, tested separately for both a has_overlap and
                    a gap (no-overlap) scenario.
                    """
                    wavelengths = np.atleast_1d(wavelengths).astype(float)
                    n = len(_wave_eval)

                    # Vectorized version of the scalar searchsorted +
                    # nearest-neighbour snap below the extrapolation checks.
                    idx_raw = np.searchsorted(_wave_eval, wavelengths)
                    idx = np.clip(idx_raw, 0, n - 1)
                    interior = (idx_raw > 0) & (idx_raw < n)
                    idx_prev = np.clip(idx - 1, 0, n - 1)
                    snap = interior & (np.abs(_wave_eval[idx] - wavelengths)
                                       > np.abs(_wave_eval[idx_prev] - wavelengths))
                    idx = np.where(snap, idx_prev, idx)
                    f0_vals = _f0_fwhm[idx]
                    f1_vals = _f1_fwhm[idx]

                    region0 = wavelengths < _w1_min          # before file 1
                    region6 = (~region0) & (wavelengths > _w2_max)  # after file 2
                    remaining = ~(region0 | region6)

                    if _has_ovlp:
                        # `|`/`&` don't short-circuit like `or`/`and` — must
                        # not evaluate `wavelengths < _ovlp_start` at all
                        # when _ovlp_start is None (the no-overlap case).
                        region1 = remaining & (wavelengths <= _w1_max) & (wavelengths < _ovlp_start)
                        region3 = remaining & (~region1) & (wavelengths >= _ovlp_start) & (wavelengths <= _ovlp_end)
                        region4 = remaining & (~region1) & (~region3) & (wavelengths >= _w2_min)
                        region3_vals = (f0_vals + f1_vals) / 2.0
                    else:
                        region1 = remaining & (wavelengths <= _w1_max)
                        region3 = remaining & (~region1) & (wavelengths > _w1_max) & (wavelengths < _w2_min)
                        region4 = remaining & (~region1) & (~region3) & (wavelengths >= _w2_min)
                        # Computed for every point regardless of region —
                        # same "safe, discarded outside its own np.where
                        # selection" pattern as global_interp_func.
                        with np.errstate(invalid="ignore", divide="ignore"):
                            alpha = (wavelengths - _w1_max) / (_w2_min - _w1_max)
                        region3_vals = (1 - alpha) * _f0_last + alpha * _f1_first

                    fwhm = np.full(wavelengths.shape, np.nan, dtype=float)
                    fwhm = np.where(region0, _f0_first, fwhm)
                    fwhm = np.where(region6, _f1_last, fwhm)
                    fwhm = np.where(region1, f0_vals, fwhm)
                    fwhm = np.where(region3, region3_vals, fwhm)
                    fwhm = np.where(region4, f1_vals, fwhm)
                    return fwhm

                return interp_func



            # CREATE the interpolation function by calling make_merged_func
            merged_interp_func = make_merged_func(
                file0_data, file1_data,
                wave1_min, wave1_max, wave2_min, wave2_max,
                overlap_start, overlap_end, has_overlap
            )

            # Store merged fiber data
            merged_fiber_splines[fib_id] = {
                'interpolate_function': merged_interp_func,
                'wave_eval': wave_eval.copy(),
                'fwhm_merged': fwhm_merged.copy(),
                'wave_range': (wave_min_global, wave_max_global),
                'spline': merged_spline,  # Stored for reference but not used in gap
                'type': 'merged'
            }

        if self.debug:
            print(f"  ✓ Merged {len(merged_fiber_splines)} fibers")

            # Verify fibers are different
            test_wave = np.array([wave_min_global + 3000])
            test_fwhms = []
            for fib_id in list(merged_fiber_splines.keys())[:5]:
                interp_func = merged_fiber_splines[fib_id]['interpolate_function']
                test_fwhms.append(interp_func(test_wave)[0])

            n_unique = len(set(np.round(test_fwhms, 6)))
            print(f"  ✓ Test @ {test_wave[0]:.1f}Å: {n_unique} unique values from {len(test_fwhms)} fibers")
            if n_unique == 1:
                print(f"    ⚠️  WARNING: All fibers have same FWHM! This is wrong!")
            else:
                print(f"    ✓ Fibers properly individualized")

            # Test gap region if exists
            if n_files == 2 and not has_overlap:
                gap_mid = (wave1_max + wave2_min) / 2
                gap_test_fwhms = []
                for fib_id in list(merged_fiber_splines.keys())[:5]:
                    interp_func = merged_fiber_splines[fib_id]['interpolate_function']
                    gap_test_fwhms.append(interp_func(gap_mid)[0])

                n_unique_gap = len(set(np.round(gap_test_fwhms, 6)))
                print(f"  ✓ Gap test @ {gap_mid:.1f}Å: {n_unique_gap} unique values")
                print(f"    Sample gap values: {[f'{x:.4f}' for x in gap_test_fwhms[:3]]}")

        return merged_fiber_splines


    def _create_global_spline(self):
        """Create global master spline from median of all fibers.

        For gap regions: Uses LINEAR interpolation between gap boundaries
        instead of median of fiber interpolations.
        """

        if self.debug:
            print(f"\n🌍 Creating global master spline...")

        # Evaluate all fiber splines on grid
        n_fibers = len(self.fiber_splines)
        fiber_fwhm_matrix = np.zeros((n_fibers, len(self.wave_grid)))

        n_success = 0
        n_failed = 0

        for i, (fib_id, fiber_data) in enumerate(self.fiber_splines.items()):
            # Always use interpolate_function if available
            if 'interpolate_function' in fiber_data:
                interp_func = fiber_data['interpolate_function']
                fwhm_vals = interp_func(self.wave_grid)
                fiber_fwhm_matrix[i, :] = fwhm_vals
                n_success += 1

                if self.debug and i < 3:
                    print(f"  ✓ Fiber {fib_id}: FWHM range {np.nanmin(fwhm_vals):.4f} - {np.nanmax(fwhm_vals):.4f}")

            elif 'bspline' in fiber_data:
                # Single file case
                bspl = fiber_data['bspline']
                wave_min, wave_max = fiber_data['wave_range']

                fwhm_vals = np.zeros(len(self.wave_grid), dtype=float)
                first_val = float(bspl(wave_min))
                last_val = float(bspl(wave_max))

                for j, w in enumerate(self.wave_grid):
                    if w < wave_min:
                        fwhm_vals[j] = first_val
                    elif w > wave_max:
                        fwhm_vals[j] = last_val
                    else:
                        fwhm_vals[j] = float(bspl(w))

                fiber_fwhm_matrix[i, :] = fwhm_vals
                n_success += 1

                if self.debug and i < 3:
                    print(f"  ✓ Fiber {fib_id}: FWHM range {np.nanmin(fwhm_vals):.4f} - {np.nanmax(fwhm_vals):.4f}")

            else:
                if self.debug and i < 3:
                    print(f"  ⚠️  Fiber {fib_id}: No usable data")
                fiber_fwhm_matrix[i, :] = np.nan
                n_failed += 1

        if self.debug:
            print(f"  ✓ Successfully evaluated: {n_success}/{n_fibers} fibers")

        n_valid_fibers = np.sum(~np.isnan(fiber_fwhm_matrix[:, 0]))
        if n_valid_fibers == 0:
            raise ValueError("Failed to create global spline - all fibers returned NaN")

        # Global = median across all fibers
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            global_fwhm = np.nanmedian(fiber_fwhm_matrix, axis=0)

        if np.all(np.isnan(global_fwhm)):
            raise ValueError("Global FWHM is all NaN after median")

        # Store gap information for interpolation function
        self.has_gap = False
        self.gap_start = None
        self.gap_end = None
        self.gap_fwhm_start = None
        self.gap_fwhm_end = None

        # SPECIAL HANDLING FOR GAP
        if len(self.wave_ranges) == 2:
            wave1_min, wave1_max = self.wave_ranges[0]
            wave2_min, wave2_max = self.wave_ranges[1]

            if wave1_max < wave2_min:
                self.has_gap = True
                self.gap_start = wave1_max
                self.gap_end = wave2_min

                if self.debug:
                    print(f"  ✓ Detected gap: {wave1_max:.1f} - {wave2_min:.1f} Å")

                # Find indices for gap boundaries
                idx_gap_start = np.searchsorted(self.wave_grid, wave1_max)
                idx_gap_end = np.searchsorted(self.wave_grid, wave2_min)

                # Get global values at gap boundaries (from median)
                self.gap_fwhm_start = global_fwhm[idx_gap_start - 1] if idx_gap_start > 0 else global_fwhm[idx_gap_start]
                self.gap_fwhm_end = global_fwhm[idx_gap_end] if idx_gap_end < len(global_fwhm) else global_fwhm[idx_gap_end - 1]

                # Replace gap region with linear interpolation
                gap_mask = (self.wave_grid > wave1_max) & (self.wave_grid < wave2_min)

                for i, w in enumerate(self.wave_grid[gap_mask]):
                    alpha = (w - wave1_max) / (wave2_min - wave1_max)
                    global_fwhm[gap_mask][i] = (1 - alpha) * self.gap_fwhm_start + alpha * self.gap_fwhm_end

                if self.debug:
                    print(f"    Gap boundaries: {self.gap_fwhm_start:.4f} -> {self.gap_fwhm_end:.4f}")

        # Fit spline to global (but won't use it in gap - will use linear)
        valid_mask = ~np.isnan(global_fwhm)
        if not np.all(valid_mask):
            wave_valid = self.wave_grid[valid_mask]
            fwhm_valid = global_fwhm[valid_mask]
            self.global_spline = UnivariateSpline(wave_valid, fwhm_valid, s=0, k=3)
        else:
            self.global_spline = UnivariateSpline(self.wave_grid, global_fwhm, s=0, k=3)

        if self.debug:
            print(f"  ✓ Global FWHM range: {np.nanmin(global_fwhm):.4f} - {np.nanmax(global_fwhm):.4f} Å")


    def _create_interpolator_dict(self):
        """Create interpolator dictionary matching aps_fwhm.py format."""

        self.interpolator_dict = {}

        # Global interpolator with gap handling
        wave_min_g = np.min(self.wave_grid)
        wave_max_g = np.max(self.wave_grid)
        first_val = float(self.global_spline(wave_min_g))
        last_val = float(self.global_spline(wave_max_g))

        # Capture gap info
        has_gap = self.has_gap
        gap_start = self.gap_start
        gap_end = self.gap_end
        gap_fwhm_start = self.gap_fwhm_start
        gap_fwhm_end = self.gap_fwhm_end
        global_spline = self.global_spline

        def global_interp_func(wavelengths,
                            _wave_min=wave_min_g,
                            _wave_max=wave_max_g,
                            _first=first_val,
                            _last=last_val,
                            _has_gap=has_gap,
                            _gap_start=gap_start,
                            _gap_end=gap_end,
                            _gap_fwhm_start=gap_fwhm_start,
                            _gap_fwhm_end=gap_fwhm_end,
                            _spline=global_spline):
            """Global interpolation with linear gap handling.

            Vectorized (was: a plain Python `for` loop evaluating one
            wavelength point at a time, same anti-pattern as the per-fibre
            interp_func below — see that one's comment for the general
            story). This is the interpolator every L1 IFU/LIFU/MIFU target
            gets assigned (see aps_utils.py's _assign_arm_results_to_targets:
            non-MOS modes always use fwhm_interp_dict['global'], never a
            per-fibre one), so on a real ~30k-spaxel stackcube this one
            function alone accounted for essentially the *entire* ~220s
            cost of aps_l1_preview._build_fwhm_cache — the per-fibre fix
            elsewhere in this file never touched this code path at all.
            Confirmed bit-identical output against the old per-point loop
            (max abs diff 0.0) including exact gap-boundary edge points
            (w == gap_start/gap_end, tested with a synthetic has_gap=True
            case since the real file this was found on has no gap) and
            ~35x faster in isolation.
            """
            wavelengths = np.atleast_1d(wavelengths)
            spline_vals = np.asarray(_spline(wavelengths), dtype=float)
            if _has_gap:
                # Computed for every point regardless of whether it's
                # actually inside the gap, same "safe because the result
                # gets discarded by np.where" reasoning as elsewhere in
                # this file — errstate guards the only way this could ever
                # warn (_gap_end == _gap_start, which also makes in_gap
                # all-False, so the result is never selected anyway).
                with np.errstate(invalid="ignore", divide="ignore"):
                    alpha = (wavelengths - _gap_start) / (_gap_end - _gap_start)
                    gap_vals = (1 - alpha) * _gap_fwhm_start + alpha * _gap_fwhm_end
                in_gap = (wavelengths > _gap_start) & (wavelengths < _gap_end)
                normal_vals = np.where(in_gap, gap_vals, spline_vals)
            else:
                normal_vals = spline_vals

            fwhm = np.where(wavelengths < _wave_min, _first,
                             np.where(wavelengths > _wave_max, _last, normal_vals))
            fwhm = np.maximum(fwhm, 0.1)  # Ensure positive
            return fwhm if len(wavelengths) > 1 else float(fwhm[0])

        self.interpolator_dict['global'] = {
            'interpolate_function': global_interp_func,
            'type': 'global_lsf',
            'description': 'Global master from median of all fibers',
            'wavelength_range': (wave_min_g, wave_max_g),
            'n_fibers': len(self.fiber_splines)
        }

        # Individual fiber interpolators (same as before)
        for fib_id, fiber_data in self.fiber_splines.items():

            if 'interpolate_function' in fiber_data:
                # Already has interpolation function (merged case)
                fiber_interp_func = fiber_data['interpolate_function']
                wave_range = fiber_data['wave_range']

            elif 'bspline' in fiber_data:
                # Single file case - create interpolation function
                bspl = fiber_data['bspline']
                wave_min, wave_max = fiber_data['wave_range']

                first_val_fiber = float(bspl(wave_min))
                last_val_fiber = float(bspl(wave_max))

                def make_fiber_func(bspline_obj, wmin, wmax, first_v, last_v):
                    """Closure to capture fiber-specific data."""
                    def interp_func(wavelengths,
                                _bspl=bspline_obj,
                                _wmin=wmin,
                                _wmax=wmax,
                                _first=first_v,
                                _last=last_v):
                        # Vectorized (was: a plain Python `for` loop calling
                        # _bspl() once per wavelength point) — scipy's
                        # BSpline.__call__ already accepts and evaluates a
                        # full array in one call, so the loop only added
                        # per-point Python/scipy dispatch overhead on top of
                        # the same underlying computation. Confirmed
                        # bit-identical output against the old per-point
                        # loop (max abs diff 0.0 across 100 real fibres x
                        # 2 wavelength grids, including in-range and
                        # out-of-[wmin,wmax]-range points) and ~19x faster
                        # in isolation — this loop runs once per spaxel per
                        # arm when the Explorer's FWHM tab builds its
                        # per-fibre cache (aps_l1_preview._build_fwhm_cache),
                        # so on a real ~30k-spaxel full LIFU cube this
                        # single change cuts that step from ~140s to ~7s.
                        # np.where evaluates _bspl(wavelengths) for every
                        # point regardless of whether it's in-range, but
                        # that's safe: BSpline's default extrapolate=True
                        # setting never raises for out-of-domain points, it
                        # just produces a value np.where then discards in
                        # favour of _first/_last.
                        wavelengths = np.atleast_1d(wavelengths)
                        fwhm = np.where(
                            wavelengths < _wmin, _first,
                            np.where(wavelengths > _wmax, _last, _bspl(wavelengths)),
                        )
                        fwhm = np.maximum(fwhm, 0.1)
                        return fwhm if len(wavelengths) > 1 else float(fwhm[0])
                    return interp_func

                fiber_interp_func = make_fiber_func(bspl, wave_min, wave_max,
                                                first_val_fiber, last_val_fiber)
                wave_range = (wave_min, wave_max)

            else:
                print(f"Warning: Fiber {fib_id} has unknown format, skipping")
                continue

            self.interpolator_dict[fib_id] = {
                'interpolate_function': fiber_interp_func,
                'type': 'fiber_lsf',
                'specnum': fib_id,
                'wavelength_range': wave_range
            }



    def get_fwhm(self, wavelength, specnum=None):
        """
        Get FWHM value(s) at given wavelength(s).

        Parameters
        ----------
        wavelength : float or array
            Wavelength(s) in Angstroms
        specnum : int, optional
            Fiber spectrum number. If None, returns global FWHM.

        Returns
        -------
        fwhm : float or array
            FWHM value(s) at requested wavelength(s)
        """

        if specnum is None:
            return self.interpolator_dict['global']['interpolate_function'](wavelength)
        else:
            if specnum not in self.interpolator_dict:
                raise ValueError(f"Fiber {specnum} not found in LSF data")
            return self.interpolator_dict[specnum]['interpolate_function'](wavelength)

    def get_interpolator_dict(self):
        """Get the comprehensive interpolator dictionary."""
        return self.interpolator_dict

    def get_available_fibers(self):
        """Get list of available fiber specnums."""
        return [k for k in self.interpolator_dict.keys() if k != 'global']

    def build_spaxel_weighted_entries(self, weight_matrix, fibre_nspecs, aps_ids):
        """Opt-in IFU-cube feature: combine this interpolator's existing
        per-fibre LSF `interpolate_function`s into one per-spaxel
        weighted-average `interpolate_function` each, using a
        (n_spaxel x n_fibre_instance) geometric contribution-weight
        matrix (see `aps_ifu_spaxel_contrib.contributing_fibre_weights`).

        Thin wrapper around `aps_ifu_spaxel_contrib.
        build_spaxel_weighted_entries` — see that function's own
        docstring for the full behaviour (no new fitting, no mutation of
        this object's own `interpolator_dict`, closed-form weighted
        combination on the shared `wave_grid`). Callers (see
        `aps_utils.APSOB._process_single_arm_vectorized`'s
        `spaxel_weighted_lsf` handling) merge the returned dict into
        their own *copy* of `get_interpolator_dict()`'s result under a
        namespaced key — never into this object's own `interpolator_dict`
        directly.

        Returns
        -------
        dict
            `{aps_id: {'interpolate_function': callable,
            'type': 'spaxel_lsf_weighted', 'n_contrib_fibres': int}, ...}`
            — see `aps_ifu_spaxel_contrib.build_spaxel_weighted_entries`.
        """
        from PyAPS.aps_ifu_spaxel_contrib import build_spaxel_weighted_entries
        return build_spaxel_weighted_entries(
            self.interpolator_dict, self.wave_grid, weight_matrix,
            fibre_nspecs, aps_ids, entry_type='spaxel_lsf_weighted',
        )

    def get_file_prefix(self):
        """Get the file prefix for output naming."""
        return self.file_prefix

    def generate_debug_headname(self):
        """Generate a descriptive name for debug plots."""
        if 'setup' in self.metadata:
            return f"LSF_{self.metadata['setup']}"
        else:
            return "LSF_merged"

    # def print_summary(self):
    #     """Print summary of interpolator."""

    #     print(f"\nLSF INTERPOLATOR SUMMARY:")
    #     print(f"  Source: Pre-computed B-spline FITS files")
    #     print(f"  Individual fiber splines: {len(self.fiber_splines)}")
    #     print(f"  Global master: Median across all fibers")

    #     if self.wave_grid is not None:
    #         print(f"  Wavelength range: {np.min(self.wave_grid):.1f} - {np.max(self.wave_grid):.1f} Å")

    #     if 'files' in self.metadata:
    #         print(f"  Files merged: {len(self.metadata['files'])}")

    def print_summary(self):
        """Print summary of interpolator."""

        print(f"\nLSF INTERPOLATOR SUMMARY:")
        print(f"  Source: Pre-computed B-spline FITS files")
        print(f"  Individual fiber splines: {len(self.fiber_splines)}")
        print(f"  Global master: Median across all fibers")

        if self.wave_grid is not None:
            print(f"  Wavelength range: {np.min(self.wave_grid):.1f} - {np.max(self.wave_grid):.1f} Å")

        if 'files' in self.metadata:
            print(f"  Files merged: {len(self.metadata['files'])}")

        # Show smoothing info
        if hasattr(self, 'smooth_length') and self.smooth_length is not None:
            print(f"  Smoothing: {self.smooth_length:.1f}Å ({self.kernel_type} kernel)")
        else:
            print(f"  Smoothing: None")




    def _fill_missing_fibers(self, max_nspec=960):
        """
        Fill in missing NSPEC values by copying from nearest available fiber.

        Parameters
        ----------
        max_nspec : int
            Maximum NSPEC number expected (default: 960 for WEAVE)
        """

        if self.debug:
            print(f"\n🔍 Checking for missing fibers (1-{max_nspec})...")

        # Get list of available NSPECs (excluding 'global')
        available_nspecs = sorted([k for k in self.interpolator_dict.keys() if k != 'global'])
        if max_nspec is None:
            max_nspec = max(available_nspecs)+1 if available_nspecs else 0

        if not available_nspecs:
            print("  ⚠️  No fibers available!")
            return

        # Determine range to fill
        min_nspec = 1
        expected_nspecs = set(range(min_nspec, max_nspec + 1))
        missing_nspecs = sorted(expected_nspecs - set(available_nspecs))

        if not missing_nspecs:
            if self.debug:
                print(f"  ✓ All fibers present ({len(available_nspecs)} fibers)")
            return

        if self.debug:
            print(f"  ⚠️  Found {len(missing_nspecs)} missing fibers")
            if len(missing_nspecs) <= 20:
                print(f"     Missing NSPECs: {missing_nspecs}")
            else:
                print(f"     Missing NSPECs: {missing_nspecs[:10]} ... {missing_nspecs[-10:]}")

        # Fill each missing fiber
        n_filled = 0
        for nspec in missing_nspecs:
            # Find nearest available fiber
            nearest_nspec = self._find_nearest_fiber(nspec, available_nspecs)

            if nearest_nspec is None:
                if self.debug:
                    print(f"  ⚠️  Cannot fill NSPEC {nspec} - no fibers available")
                continue

            # Copy the interpolation function from nearest fiber
            source_fiber = self.interpolator_dict[nearest_nspec]

            # Create new entry with copied data
            self.interpolator_dict[nspec] = {
                'interpolate_function': source_fiber['interpolate_function'],  # Same function
                'type': 'fiber_lsf_copied',  # Flag indicating it's copied
                'specnum': nspec,  # This fiber's number
                'wavelength_range': source_fiber['wavelength_range'],
                'copied_from': nearest_nspec,  # Record source
                'is_copy': True  # Easy boolean check
            }

            n_filled += 1

            # Debug output for first few
            if self.debug and n_filled <= 5:
                print(f"  📋 NSPEC {nspec} <- copied from NSPEC {nearest_nspec}")

        if self.debug:
            print(f"  ✓ Filled {n_filled} missing fibers")

            # Summary
            total_fibers = len([k for k in self.interpolator_dict.keys() if k != 'global'])
            original_fibers = len(available_nspecs)
            copied_fibers = total_fibers - original_fibers
            print(f"  📊 Total fibers: {total_fibers} ({original_fibers} original + {copied_fibers} copied)")


    def _find_nearest_fiber(self, target_nspec, available_nspecs):
        """
        Find the nearest available fiber to target NSPEC.

        Search order:
        1. Try target-1, target-2, ... (going down)
        2. Try target+1, target+2, ... (going up)
        3. Use whichever is found first

        Parameters
        ----------
        target_nspec : int
            Target NSPEC to find neighbor for
        available_nspecs : list
            Sorted list of available NSPEC numbers

        Returns
        -------
        int or None
            Nearest available NSPEC, or None if none available
        """

        if not available_nspecs:
            return None

        # Convert to numpy array for easier searching
        available = np.array(available_nspecs)

        # Find the closest available fiber
        distances = np.abs(available - target_nspec)
        nearest_idx = np.argmin(distances)
        nearest_nspec = available[nearest_idx]

        return int(nearest_nspec)



    def apply_smoothing(self, smooth_length=None, kernel_type='boxcar'):
        """
        Apply smoothing to all fiber LSF profiles.

        OPTIMIZED: Pre-computes smoothed grids for all fibers once,
        then interpolates from cached results.

        Parameters
        ----------
        smooth_length : float, optional
            Smoothing length in Angstroms. If None, no smoothing applied.
        kernel_type : str
            Type of smoothing kernel: 'boxcar', 'gaussian', 'hanning'
        """

        if smooth_length is None or smooth_length <= 0:
            if self.debug:
                print("  ℹ️  No smoothing applied (smooth_length=None)")
            self.smooth_length = None
            self.kernel_type = None
            return

        self.smooth_length = smooth_length
        self.kernel_type = kernel_type

        if self.debug:
            print(f"\n🔧 Applying smoothing: length={smooth_length:.1f}Å, kernel={kernel_type}")

        # Create smoothing kernel
        wave_resolution = np.median(np.diff(self.wave_grid))
        kernel_size = int(smooth_length / wave_resolution)

        # Ensure odd kernel size
        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = self._create_smoothing_kernel(kernel_size, kernel_type)

        if self.debug:
            print(f"  ✓ Kernel size: {kernel_size} points ({kernel_size * wave_resolution:.1f}Å)")

        # Store original functions if not already saved
        if not hasattr(self, '_original_interpolators'):
            self._original_interpolators = {}

        # PRE-COMPUTE smoothed grids for all fibers
        if self.debug:
            print(f"  🔄 Pre-computing smoothed grids for all fibers...")

        self._smoothed_grids = {}

        # Global interpolator
        if 'global' not in self._original_interpolators:
            self._original_interpolators['global'] = self.interpolator_dict['global']['interpolate_function']

        original_global = self._original_interpolators['global']
        unsmoothed_global = original_global(self.wave_grid)
        smoothed_global = self._smooth_array(unsmoothed_global, kernel)
        self._smoothed_grids['global'] = smoothed_global

        # All fiber interpolators
        fiber_ids = self.get_available_fibers()
        for i, nspec in enumerate(fiber_ids):
            if nspec not in self._original_interpolators:
                self._original_interpolators[nspec] = self.interpolator_dict[nspec]['interpolate_function']

            original_func = self._original_interpolators[nspec]
            unsmoothed = original_func(self.wave_grid)
            smoothed = self._smooth_array(unsmoothed, kernel)
            self._smoothed_grids[nspec] = smoothed

            # Progress indicator for large datasets
            if self.debug and (i + 1) % 100 == 0:
                print(f"    Processed {i + 1}/{len(fiber_ids)} fibers...")

        if self.debug:
            print(f"  ✓ Pre-computed {len(self._smoothed_grids)} smoothed grids")

        # Replace interpolation functions with fast lookup versions
        self._apply_smoothing_to_interpolators()

        if self.debug:
            print(f"  ✓ Smoothing applied to {len(fiber_ids)} fibers + global")


    def _create_smoothing_kernel(self, kernel_size, kernel_type='boxcar'):
        """
        Create smoothing kernel.

        Parameters
        ----------
        kernel_size : int
            Size of kernel in pixels (should be odd)
        kernel_type : str
            Type: 'boxcar', 'gaussian', 'hanning'

        Returns
        -------
        kernel : ndarray
            Normalized smoothing kernel
        """

        if kernel_type == 'boxcar':
            # Simple moving average
            kernel = np.ones(kernel_size) / kernel_size

        elif kernel_type == 'gaussian':
            # Gaussian kernel (sigma = kernel_size / 6 for ~99% within window)
            x = np.arange(kernel_size) - kernel_size // 2
            sigma = kernel_size / 6.0
            kernel = np.exp(-0.5 * (x / sigma) ** 2)
            kernel /= kernel.sum()

        elif kernel_type == 'hanning':
            # Hanning window (smooth edges)
            kernel = np.hanning(kernel_size)
            kernel /= kernel.sum()

        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

        return kernel


    def _smooth_array(self, data, kernel):
        """
        Apply smoothing kernel to 1D array with reflection padding.

        Uses reflection padding at edges to avoid edge artifacts.

        Parameters
        ----------
        data : ndarray
            1D array to smooth
        kernel : ndarray
            Smoothing kernel

        Returns
        -------
        smoothed : ndarray
            Smoothed array (same length as input)
        """

        # Pad array with reflection to handle edges
        pad_width = len(kernel) // 2
        padded = np.pad(data, pad_width, mode='reflect')

        # Apply convolution
        smoothed_padded = np.convolve(padded, kernel, mode='same')

        # Remove padding
        smoothed = smoothed_padded[pad_width:-pad_width]

        # Ensure same length as input
        if len(smoothed) != len(data):
            smoothed = smoothed[:len(data)]

        return smoothed


    def _apply_smoothing_to_interpolators(self):
        """
        Replace interpolation functions with fast lookup from pre-computed grids.

        OPTIMIZED: Just interpolates from cached smoothed grids instead of
        recomputing smoothing every time.
        """

        if not hasattr(self, '_smoothed_grids') or self._smoothed_grids is None:
            return

        # Capture wave_grid
        wave_grid = self.wave_grid.copy()

        # Create interpolation function factory
        def make_fast_interp(smoothed_grid, _wave_grid=wave_grid):
            """Create fast interpolation function from pre-computed grid."""
            def fast_interp(wavelengths):
                wavelengths = np.atleast_1d(wavelengths)
                # Simple fast interpolation
                result = np.interp(wavelengths, _wave_grid, smoothed_grid)
                return result if len(wavelengths) > 1 else float(result[0])
            return fast_interp

        # Global interpolator
        smoothed_global = self._smoothed_grids['global']
        self.interpolator_dict['global']['interpolate_function'] = make_fast_interp(smoothed_global)

        # All fibers
        for nspec in self.get_available_fibers():
            smoothed_fiber = self._smoothed_grids[nspec]
            self.interpolator_dict[nspec]['interpolate_function'] = make_fast_interp(smoothed_fiber)


    def remove_smoothing(self):
        """Remove smoothing and restore original interpolators."""

        if not hasattr(self, '_original_interpolators'):
            if self.debug:
                print("  ℹ️  No smoothing to remove")
            return

        if self.debug:
            print("\n🔧 Removing smoothing, restoring original interpolators...")

        # Restore global
        if 'global' in self._original_interpolators:
            self.interpolator_dict['global']['interpolate_function'] = self._original_interpolators['global']

        # Restore all fibers
        for nspec in self.get_available_fibers():
            if nspec in self._original_interpolators:
                self.interpolator_dict[nspec]['interpolate_function'] = self._original_interpolators[nspec]

        # Clear smoothing data
        self.smooth_length = None
        self.kernel_type = None
        self._smoothed_grids = None

        if self.debug:
            print("  ✓ Smoothing removed")


    # =========================================================================
    # PICKLE SAVE/LOAD METHODS
    # =========================================================================

    def save(self, output_path=None, output_dir=None):
        """
        Save the interpolator to a pickle file using dill.

        Parameters
        ----------
        output_path : str, optional
            Full path to save file. If None, auto-generates based on input files.
        output_dir : str, optional
            Directory to save to. If None, uses input file directory.
            Only used if output_path is None.

        Returns
        -------
        str : Path to saved file
        """
        if output_path is None:
            output_path = get_output_pickle_path(self.input_files, output_dir)

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        print(f"\n💾 Saving LSF interpolator to: {output_path}")

        try:
            with open(output_path, 'wb') as f:
                PICKLE_MODULE.dump(self, f)
            print(f"  ✓ Saved successfully ({os.path.getsize(output_path) / 1024:.1f} KB)")
            return output_path
        except Exception as e:
            raise IOError(f"Failed to save interpolator: {e}")

    @classmethod
    def load(cls, pickle_path):
        """
        Load an interpolator from a pickle file.

        Parameters
        ----------
        pickle_path : str
            Path to pickle file

        Returns
        -------
        LSFInterpolator : Loaded interpolator

        Raises
        ------
        FileNotFoundError
            If the pickle file does not exist
        """
        if not os.path.exists(pickle_path):
            raise FileNotFoundError(
                f"LSF interpolator pickle file not found: {pickle_path}\n"
                f"Please run run_lsf_analysis() first to create the interpolator,\n"
                f"or check that the file path is correct."
            )

        print(f"\n📂 Loading LSF interpolator from: {pickle_path}")

        try:
            with open(pickle_path, 'rb') as f:
                interpolator = PICKLE_MODULE.load(f)
            print(f"  ✓ Loaded successfully")
            return interpolator
        except Exception as e:
            raise IOError(
                f"Failed to load LSF interpolator from: {pickle_path}\n"
                f"Error: {e}\n"
                f"The file may be corrupted or incompatible with current code version."
            )






# =============================================================================
# DIAGNOSTIC PLOTTING
# =============================================================================

def create_diagnostic_plots(interpolator, figdir, figname="lsf_diagnostic"):
    """
    Create comprehensive diagnostic plots for LSF interpolator.
    """

    print(f"\n📊 Creating diagnostic plots...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # EXTEND wavelength range by ±200Å for extrapolation check
    wave_min = np.min(interpolator.wave_grid)
    wave_max = np.max(interpolator.wave_grid)
    wave_test = np.linspace(wave_min - 200, wave_max + 200, 7000)

    # Get available fiber IDs
    fiber_ids = interpolator.get_available_fibers()

    if len(fiber_ids) == 0:
        print("  ⚠️  Warning: No fiber IDs found!")
        return

    print(f"  ✓ Plotting {len(fiber_ids)} fibers")

    # Detect gap region
    has_gap = False
    gap_start = None
    gap_end = None

    if len(interpolator.wave_ranges) == 2:
        wave1_min, wave1_max = interpolator.wave_ranges[0]
        wave2_min, wave2_max = interpolator.wave_ranges[1]

        if wave1_max < wave2_min:
            has_gap = True
            gap_start = wave1_max
            gap_end = wave2_min
            print(f"  ✓ Gap detected: {gap_start:.1f} - {gap_end:.1f} Å")

    # =========================================================================
    # Panel 1: All fiber LSFs + global with gap highlighting
    # =========================================================================
    ax = axes[0, 0]

    # Sample fibers for plotting
    n_sample = min(100, len(fiber_ids))
    sample_step = max(1, len(fiber_ids) // n_sample)
    sample_ids = fiber_ids[::sample_step]

    print(f"  ✓ Panel 1: Sampling {len(sample_ids)} fibers...")

    # Plot individual fibers
    n_plotted = 0
    for fib_id in sample_ids:
        try:
            fwhm_fiber = interpolator.get_fwhm(wave_test, specnum=fib_id)

            if has_gap:
                # Split into blue, gap, red regions
                blue_mask = wave_test <= gap_start
                gap_mask = (wave_test > gap_start) & (wave_test < gap_end)
                red_mask = wave_test >= gap_end

                # Plot blue arm (gray)
                ax.plot(wave_test[blue_mask], fwhm_fiber[blue_mask],
                       'gray', linewidth=0.5, alpha=0.3)

                # Plot gap in MAGENTA (highly visible)
                ax.plot(wave_test[gap_mask], fwhm_fiber[gap_mask],
                       'magenta', linewidth=0.5, alpha=0.3)

                # Plot red arm (gray)
                ax.plot(wave_test[red_mask], fwhm_fiber[red_mask],
                       'gray', linewidth=0.5, alpha=0.3)
            else:
                # No gap - plot normally
                ax.plot(wave_test, fwhm_fiber, 'gray', linewidth=0.5, alpha=0.3)

            n_plotted += 1
        except Exception as e:
            if n_plotted < 3:
                print(f"    ⚠️  Failed to plot fiber {fib_id}: {e}")
            continue

    print(f"    ✓ Plotted {n_plotted} individual fibers")

    # Plot global master
    try:
        global_fwhm = interpolator.get_fwhm(wave_test)

        if has_gap:
            # Split global into regions
            blue_mask = wave_test <= gap_start
            gap_mask = (wave_test > gap_start) & (wave_test < gap_end)
            red_mask = wave_test >= gap_end

            # Plot blue (blue line)
            ax.plot(wave_test[blue_mask], global_fwhm[blue_mask],
                   'b-', linewidth=3, label='Global Master', zorder=100)

            # Plot gap in RED (distinct from fiber magenta)
            ax.plot(wave_test[gap_mask], global_fwhm[gap_mask],
                   'r-', linewidth=3, label='Global Gap (Linear)', zorder=100)

            # Plot red (blue line)
            ax.plot(wave_test[red_mask], global_fwhm[red_mask],
                   'b-', linewidth=3, zorder=100)
        else:
            ax.plot(wave_test, global_fwhm, 'r-', linewidth=3,
                   label='Global Master', zorder=100)

        print(f"    ✓ Plotted global master (range: {np.nanmin(global_fwhm):.3f} - {np.nanmax(global_fwhm):.3f})")
    except Exception as e:
        print(f"    ⚠️  Failed to plot global: {e}")

    # Mark boundaries
    ax.axvline(wave_min, color='green', linestyle='--', linewidth=2,
              alpha=0.7, label='Data Range')
    ax.axvline(wave_max, color='green', linestyle='--', linewidth=2, alpha=0.7)

    # Mark extrapolation regions
    ax.axvspan(wave_min - 200, wave_min, alpha=0.1, color='green',
              label='Extrapolation')
    ax.axvspan(wave_max, wave_max + 200, alpha=0.1, color='green')

    # Mark gap if exists
    if has_gap:
        ax.axvspan(gap_start, gap_end, alpha=0.1, color='yellow',
                  label='Gap (Linear Interp)', zorder=1)
        ax.axvline(gap_start, color='orange', linestyle='--', linewidth=2)
        ax.axvline(gap_end, color='orange', linestyle='--', linewidth=2)

    ax.set_xlabel('Wavelength (Å)', fontsize=12)
    ax.set_ylabel('FWHM (Å)', fontsize=12)
    ax.set_title(f'All Fiber LSFs + Global Master\n({len(fiber_ids)} fibers, {len(sample_ids)} shown)\nFiber gaps=MAGENTA, Global gap=RED',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(wave_min - 200, wave_max + 200)

    # =========================================================================
    # Panel 2: Fiber-to-fiber variation
    # =========================================================================
    ax = axes[0, 1]

    print(f"  ✓ Panel 2: Fiber variation...")

    test_waves = np.linspace(wave_min - 200, wave_max + 200, 25)
    fiber_matrix = np.zeros((len(fiber_ids), len(test_waves)))

    for i, fib_id in enumerate(fiber_ids):
        try:
            fiber_matrix[i, :] = interpolator.get_fwhm(test_waves, specnum=fib_id)
        except Exception as e:
            fiber_matrix[i, :] = np.nan

    n_valid = np.sum(~np.isnan(fiber_matrix[:, 0]))
    print(f"    ✓ Valid fibers: {n_valid}/{len(fiber_ids)}")

    if n_valid > 0:
        bp = ax.boxplot([fiber_matrix[:, i] for i in range(len(test_waves))],
                         positions=test_waves,
                         widths=(test_waves[-1] - test_waves[0]) / 30,
                         patch_artist=True,
                         showfliers=False)

        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)

        global_at_test = interpolator.get_fwhm(test_waves)
        ax.plot(test_waves, global_at_test, 'ro-', linewidth=2, markersize=6,
                label='Global Master', zorder=100)

    # Mark boundaries
    ax.axvline(wave_min, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axvline(wave_max, color='green', linestyle='--', linewidth=1.5, alpha=0.7)

    if has_gap:
        ax.axvspan(gap_start, gap_end, alpha=0.1, color='yellow')
        ax.axvline(gap_start, color='orange', linestyle='--', linewidth=1.5)
        ax.axvline(gap_end, color='orange', linestyle='--', linewidth=1.5)

    ax.set_xlabel('Wavelength (Å)', fontsize=12)
    ax.set_ylabel('FWHM (Å)', fontsize=12)
    ax.set_title('Fiber-to-Fiber Variation', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(wave_min - 200, wave_max + 200)

    # =========================================================================
    # Panel 3: FWHM vs wavelength (detailed sample) with gap
    # =========================================================================
    ax = axes[1, 0]

    print(f"  ✓ Panel 3: Sample fibers with gap...")

    # Plot global first
    global_fwhm = interpolator.get_fwhm(wave_test)

    if has_gap:
        blue_mask = wave_test <= gap_start
        gap_mask = (wave_test > gap_start) & (wave_test < gap_end)
        red_mask = wave_test >= gap_end

        # Blue region
        ax.plot(wave_test[blue_mask], global_fwhm[blue_mask],
               'b-', linewidth=2.5, label='Global Master', alpha=0.9, zorder=10)
        # Gap in RED
        ax.plot(wave_test[gap_mask], global_fwhm[gap_mask],
               'r-', linewidth=2.5, alpha=0.9, zorder=10)
        # Red region
        ax.plot(wave_test[red_mask], global_fwhm[red_mask],
               'b-', linewidth=2.5, alpha=0.9, zorder=10)
    else:
        ax.plot(wave_test, global_fwhm, 'b-', linewidth=2.5,
               label='Global Master', alpha=0.9, zorder=10)

    # Plot sample fibers
    n_sample_detail = min(5, len(fiber_ids))
    if n_sample_detail > 0:
        colors = plt.cm.viridis(np.linspace(0, 1, n_sample_detail))
        sample_step_detail = max(1, len(fiber_ids) // n_sample_detail)

        for i, fib_id in enumerate(fiber_ids[::sample_step_detail][:n_sample_detail]):
            try:
                fwhm_fiber = interpolator.get_fwhm(wave_test, specnum=fib_id)

                if has_gap:
                    # Blue arm
                    ax.plot(wave_test[blue_mask], fwhm_fiber[blue_mask],
                           '-', color=colors[i], linewidth=1.5, alpha=0.7,
                           label=f'Fiber {fib_id}', zorder=5)
                    # Gap in MAGENTA
                    ax.plot(wave_test[gap_mask], fwhm_fiber[gap_mask],
                           '-', color='magenta', linewidth=1.8, alpha=0.8, zorder=5)
                    # Red arm
                    ax.plot(wave_test[red_mask], fwhm_fiber[red_mask],
                           '-', color=colors[i], linewidth=1.5, alpha=0.7, zorder=5)
                else:
                    ax.plot(wave_test, fwhm_fiber, '-', color=colors[i],
                           linewidth=1.5, alpha=0.7, label=f'Fiber {fib_id}', zorder=5)

                print(f"    ✓ Plotted fiber {fib_id}")
            except Exception as e:
                print(f"    ⚠️  Failed fiber {fib_id}: {e}")

    # Mark boundaries and regions
    ax.axvline(wave_min, color='green', linestyle='--', linewidth=1.5,
              alpha=0.7, label='Data Boundary')
    ax.axvline(wave_max, color='green', linestyle='--', linewidth=1.5, alpha=0.7)

    ax.axvspan(wave_min - 200, wave_min, alpha=0.1, color='green')
    ax.axvspan(wave_max, wave_max + 200, alpha=0.1, color='green')

    if has_gap:
        ax.axvspan(gap_start, gap_end, alpha=0.1, color='yellow',
                  label='Gap Region', zorder=1)
        ax.axvline(gap_start, color='orange', linestyle='--', linewidth=1.5)
        ax.axvline(gap_end, color='orange', linestyle='--', linewidth=1.5)

        # Add text annotation for gap
        y_pos = ax.get_ylim()[1] * 0.95
        ax.text((gap_start + gap_end)/2, y_pos, 'Linear Interpolation\nFibers=MAGENTA, Global=RED',
               ha='center', va='top', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax.set_xlabel('Wavelength (Å)', fontsize=12)
    ax.set_ylabel('FWHM (Å)', fontsize=12)
    ax.set_title('FWHM vs Wavelength (Sample Fibers + Extrapolation)',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=8, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(wave_min - 200, wave_max + 200)

    # =========================================================================
    # Panel 4: Statistics
    # =========================================================================
    ax = axes[1, 1]

    print(f"  ✓ Panel 4: Statistics...")

    n_test = 50
    test_waves_stats = np.linspace(wave_min - 200, wave_max + 200, n_test)

    mean_fwhm = np.zeros(n_test)
    std_fwhm = np.zeros(n_test)
    min_fwhm = np.zeros(n_test)
    max_fwhm = np.zeros(n_test)

    for i, w in enumerate(test_waves_stats):
        fwhm_vals = []
        for fib_id in fiber_ids:
            try:
                fwhm_val = interpolator.get_fwhm(w, specnum=fib_id)
                if isinstance(fwhm_val, (list, np.ndarray)):
                    fwhm_vals.append(float(fwhm_val[0]))
                else:
                    fwhm_vals.append(float(fwhm_val))
            except:
                pass

        if len(fwhm_vals) > 0:
            mean_fwhm[i] = np.mean(fwhm_vals)
            std_fwhm[i] = np.std(fwhm_vals)
            min_fwhm[i] = np.min(fwhm_vals)
            max_fwhm[i] = np.max(fwhm_vals)

    ax.plot(test_waves_stats, mean_fwhm, 'b-', linewidth=2, label='Mean')
    ax.fill_between(test_waves_stats, mean_fwhm - std_fwhm, mean_fwhm + std_fwhm,
                    alpha=0.3, color='blue', label='±1σ')
    ax.plot(test_waves_stats, min_fwhm, 'g--', linewidth=1, label='Min', alpha=0.7)
    ax.plot(test_waves_stats, max_fwhm, 'r--', linewidth=1, label='Max', alpha=0.7)

    # Mark regions
    ax.axvline(wave_min, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axvline(wave_max, color='green', linestyle='--', linewidth=1.5, alpha=0.7)

    ax.axvspan(wave_min - 200, wave_min, alpha=0.1, color='green')
    ax.axvspan(wave_max, wave_max + 200, alpha=0.1, color='green')

    if has_gap:
        ax.axvspan(gap_start, gap_end, alpha=0.1, color='yellow')

    ax.set_xlabel('Wavelength (Å)', fontsize=12)
    ax.set_ylabel('FWHM (Å)', fontsize=12)
    ax.set_title('FWHM Statistics Across All Fibers', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(wave_min - 200, wave_max + 200)

    plt.tight_layout()

    if not os.path.exists(figdir):
        os.makedirs(figdir)

    plot_path = os.path.join(figdir, f"{figname}.png")
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Diagnostic plot saved: {plot_path}")

    plt.close(fig)


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def run_lsf_analysis(file_input, figdir=None, figname="lsf_analysis", debug=False,
                    make_plot=True, smooth_length=None, kernel_type='boxcar',
                    overwrite=False, save_pickle=True, pickle_dir=None, replace_binned=False):
    """
    Run LSF analysis from pre-computed B-spline FITS files.

    Parameters
    ----------
    file_input : str or list
        Path(s) to LSF FITS file(s)
    figdir : str, optional
        Output directory for plots
    figname : str
        Base filename for plots
    debug : bool
        Print debug information
    make_plot : bool
        Create diagnostic plots
    smooth_length : float, optional
        Smoothing length in Angstroms
    kernel_type : str
        Smoothing kernel: 'boxcar', 'gaussian', 'hanning'
    overwrite : bool
        If False, skip processing if output pickle already exists.
        If True, regenerate even if pickle exists.
    save_pickle : bool
        If True, save the interpolator to a pickle file.
    pickle_dir : str, optional
        Directory to save pickle file. If None, uses input file directory.
        If figdir is set and pickle_dir is None, uses figdir.

    Returns
    -------
    interpolator : LSFInterpolator or None
    """




    # Normalize file_input to list
    if isinstance(file_input, str):
        file_input = [file_input]

    # Optionally replace binned files with unbinned versions
    if replace_binned:
        file_input = check_and_replace_binned_files(file_input)



    # Determine pickle output path
    if pickle_dir is None:
        if figdir is not None:
            pickle_dir = figdir
        else:
            pickle_dir = os.path.dirname(os.path.abspath(file_input[0]))

    pickle_path = get_output_pickle_path(file_input, pickle_dir)

    # Check if output already exists
    if os.path.exists(pickle_path) and not overwrite:
        # Load first (unavoidable — the only way to check what format a
        # cached file actually is), *then* decide whether to trust it.
        # dill freezes closures' actual code objects at save time, so a
        # pickle built before a fix to those closures (e.g. the
        # 2026-08-05 LSF vectorization fix) keeps running the old, slow
        # code forever no matter what the current source says — this
        # cache-version stamp (see _LSF_CACHE_FORMAT_VERSION) is what
        # makes that self-correcting instead of silently-stuck-slow.
        #
        # A load that raises outright (not just a version mismatch) is a
        # DIFFERENT, more serious case than the block below: the pickle is
        # unreadable, most often because it was written under an older
        # Python/scipy/dill and the on-disk object format has since
        # changed underneath it (confirmed live 2026-09-24, night 20240105:
        # scipy.interpolate.BSpline.__setstate__ changed its expected tuple
        # shape between the pre- and post- Python-3.12-upgrade scipy
        # versions, so every LSF .dill cached before that upgrade fails
        # with `ValueError: too many values to unpack`). Before this fix,
        # LSFInterpolator.load()'s own already-good diagnostic IOError
        # ("may be corrupted or incompatible with current code version")
        # was silently absorbed several call-frames up by
        # aps_utils.py's generic `except Exception` around FWHM/LSF
        # creation, indistinguishable there from "this target genuinely
        # has no calibration data" - a real data gap and a stale cache are
        # not the same problem and must not look the same in the logs.
        # Raised loudly here instead, then the same automatic-rebuild path
        # the version-mismatch case below already uses - a cache is always
        # safe to regenerate from source, it should just never happen
        # quietly.
        try:
            interpolator = LSFInterpolator.load(pickle_path)
        except Exception as e:
            interpolator = None
            print("!"*70)
            print("LSF ANALYSIS - STALE/UNREADABLE CACHE (not a data problem)")
            print("!"*70)
            print(f"\n📂 {pickle_path}")
            print(f"   failed to load: {type(e).__name__}: {e}")
            print(f"   This is almost always a pickle written under a Python/scipy/dill")
            print(f"   version that has since changed its object format underneath it -")
            print(f"   NOT missing or insufficient calibration data. Regenerating from")
            print(f"   source now; delete this file to skip straight to a clean rebuild")
            print(f"   next time.")
            # Falls through to the rebuild below, same as overwrite=True.

        if interpolator is not None:
            cached_version = getattr(interpolator, "_cache_format_version", None)
            if cached_version == _LSF_CACHE_FORMAT_VERSION:
                print("="*70)
                print("LSF ANALYSIS - Loading Existing Pickle")
                print("="*70)
                print(f"\n📂 Output pickle already exists: {pickle_path}")
                print(f"   Set overwrite=True to regenerate.")
                interpolator.set_debug(debug)
                interpolator.print_summary()
                return interpolator
            print("="*70)
            print("LSF ANALYSIS - Cached pickle predates a performance fix, rebuilding")
            print("="*70)
            print(f"\n📂 {pickle_path}")
            print(f"   was built with cache format {cached_version!r} (current is "
                  f"{_LSF_CACHE_FORMAT_VERSION!r}) — its interpolator closures predate a "
                  f"speed fix and would silently keep running the old, slow code forever. "
                  f"Regenerating once, automatically.")
            # Falls through to the rebuild below, same as overwrite=True.

    print("="*70)
    print("LSF ANALYSIS - Pre-computed B-spline Reader")
    print("="*70)

    # Show output headname for reference
    headname = generate_output_headname(file_input)
    print(f"  Output headname: {headname}")
    print(f"  Pickle path: {pickle_path}")

    try:
        interpolator = LSFInterpolator()
        interpolator.set_debug(debug)

        success = interpolator.create_from_files(
            file_input,
            smooth_length=smooth_length,
            kernel_type=kernel_type
        )

        if not success:
            print(f"\n❌ LSF interpolator creation failed")
            return None

        # Save pickle
        if save_pickle:
            interpolator.save(pickle_path)

        if make_plot and figdir:
            # Use same headname as pickle for consistency
            prefixed_figname = f"{headname}_{figname}"

            print(f"\n📊 CREATING DIAGNOSTIC PLOTS...")

            # Quick verification before plotting
            if isinstance(file_input, list) and len(file_input) > 1:
                print("\n🔍 Verifying merged fibers are different:")
                fiber_ids = interpolator.get_available_fibers()
                test_wave = 5000.0
                test_fwhms = []
                for fib_id in fiber_ids[:10]:
                    try:
                        fwhm_val = interpolator.get_fwhm(test_wave, specnum=fib_id)
                        # Safely extract scalar value
                        if isinstance(fwhm_val, np.ndarray):
                            test_fwhms.append(float(np.atleast_1d(fwhm_val)[0]))
                        else:
                            test_fwhms.append(float(fwhm_val))
                    except:
                        pass

                unique_fwhms = len(set([f'{x:.4f}' for x in test_fwhms]))
                print(f"  At {test_wave}Å: {unique_fwhms} unique FWHM values from {len(test_fwhms)} fibers")
                if unique_fwhms < 2:
                    print(f"  ⚠️  WARNING: Fibers not properly individualized!")
                else:
                    print(f"  ✓ Fibers are properly individualized")

            create_diagnostic_plots(interpolator, figdir, prefixed_figname)

        return interpolator

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_lsf_interpolator(pickle_path):
    """
    Convenience function to load an LSF interpolator from pickle.

    Parameters
    ----------
    pickle_path : str
        Path to pickle file

    Returns
    -------
    LSFInterpolator : Loaded interpolator

    Raises
    ------
    FileNotFoundError
        If the pickle file does not exist
    """
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(
            f"LSF interpolator pickle file not found: {pickle_path}\n"
            f"Please run run_lsf_analysis() first to create the interpolator,\n"
            f"or check that the file path is correct."
        )
    return LSFInterpolator.load(pickle_path)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # DEMO settings: edit for your setup. Replace the <PYAPS_DATA>, <PYAPS_DIR>, <night>, <runid>, <obid>
    # markers below with your own locations and identifiers (no machine paths belong in this repository).

    print("\n" + "="*80)
    print("🎨 LSF INTERPOLATOR - EXAMPLES")
    print("="*80)

    lsfdir = "<PYAPS_DATA>/LSF/<night>"
    output_dir = "<PYAPS_DATA>/L2_dev/lsf_test"
    os.makedirs(output_dir, exist_ok=True)

    # =========================================================================
    # TEST 1: Single file (no smoothing)
    # =========================================================================
    print("\n" + "="*80)
    print("TEST 1: Single LSF File (No Smoothing)")
    print("="*80)

    single_file = os.path.join(lsfdir, "lsf_BLUEL11_MOS-A.fits")

    if os.path.exists(single_file):
        interp_single = run_lsf_analysis(
            single_file,
            figdir=output_dir,
            figname="single_file",
            debug=True,
            make_plot=True,
            overwrite=False,  # Will skip if pickle exists
            save_pickle=True,
            replace_binned=False               # Replace binned files with unbinned versions
        )

        if interp_single:
            # Test interpolation
            test_waves = np.array([4000.0, 4500.0, 5000.0, 5500.0])
            print(f"\nTesting interpolation:")
            print(f"  Wavelengths: {test_waves}")

            global_fwhm = interp_single.get_fwhm(test_waves)
            print(f"  Global FWHM: {global_fwhm}")

            # Test a specific fiber
            available_fibers = interp_single.get_available_fibers()
            if len(available_fibers) > 0:
                test_fiber = available_fibers[50]
                fiber_fwhm = interp_single.get_fwhm(test_waves, specnum=test_fiber)
                print(f"  Fiber {test_fiber} FWHM: {fiber_fwhm}")

    # =========================================================================
    # TEST 2: Multiple files (blue + red, no smoothing)
    # =========================================================================
    print("\n" + "="*80)
    print("TEST 2: Multiple LSF Files (Blue + Red, No Smoothing)")
    print("="*80)

    blue_file = os.path.join(lsfdir, "lsf_GREENH11_MOS-A.fits")
    red_file = os.path.join(lsfdir, "lsf_REDH11_MOS-A.fits")

    if os.path.exists(blue_file) and os.path.exists(red_file):
        interp_multi = run_lsf_analysis(
            [blue_file, red_file],
            figdir=output_dir,
            figname="blue_red_merged",
            debug=True,
            make_plot=True,
            smooth_length=None,
            kernel_type='gaussian',
            overwrite=False,  # Will skip if pickle exists
            save_pickle=True,
            replace_binned=False               # Replace binned files with unbinned versions
        )

        if interp_multi:
            # Test interpolation across full range
            test_waves = np.array([4000.0, 5000.0, 6000.0, 7000.0, 8000.0])
            print(f"\nTesting interpolation across blue+red:")
            print(f"  Wavelengths: {test_waves}")

            global_fwhm = interp_multi.get_fwhm(test_waves)
            print(f"  Global FWHM: {global_fwhm}")

            # Get the dictionary
            interp_dict = interp_multi.get_interpolator_dict()
            # breakpoint()

    # =========================================================================
    # TEST 3: Load from pickle
    # =========================================================================
    print("\n" + "="*80)
    print("TEST 3: Load from Pickle")
    print("="*80)

    # Try loading the saved pickle
    expected_pickle = os.path.join(output_dir, f"GREENH11__REDH11_MOS-A{PICKLE_EXT}")
    if os.path.exists(expected_pickle):
        loaded_interp = load_lsf_interpolator(expected_pickle)
        loaded_interp.print_summary()

        # Test it works
        test_wave = 6000.0
        print(f"\n  Test @ {test_wave}Å: Global FWHM = {loaded_interp.get_fwhm(test_wave):.4f}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("✅ ALL TESTS COMPLETE!")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    print(f"  1. BLUEL11_MOS-A{PICKLE_EXT}")
    print(f"  2. GREENH11__REDH11_MOS-A{PICKLE_EXT}")
    print("\nGenerated plots:")
    print("  1. BLUEL11_MOS-A_single_file.png")
    print("  2. GREENH11__REDH11_MOS-A_blue_red_merged.png")
