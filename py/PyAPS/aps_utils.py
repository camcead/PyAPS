import contextvars
import math
import os
import sys
import traceback
import warnings
from copy import deepcopy

import scipy

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from pathlib import Path

import matplotlib

# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from astropy import wcs
from astropy.io import fits
from astropy.io.fits import getdata, getheader
from astropy.io.fits.verify import VerifyWarning
from scipy import ndimage

warnings.simplefilter("ignore", category=VerifyWarning)

import argparse
import ctypes
import datetime
import multiprocessing as mp
import pickle
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

from astropy.table import Column, Table, join
from scipy import sparse

import PyAPS
from PyAPS import aps_constants
from PyAPS.aps_fwhm import run_fwhm_analysis
from PyAPS.aps_lsf import generate_output_headname, run_lsf_analysis

APSVERS = PyAPS.__version__



###########################################################################
def get_pyaps_repo_root():
    """
    Get PyAPS repository root directory (not the Python package dir).

    Returns:
        str: Absolute path to PyAPS repository root
    """
    try:
        # Method 1: Get package location, then go up to find repo root
        import PyAPS
        pyaps_package_file = PyAPS.__file__
        pyaps_package_dir = os.path.dirname(os.path.abspath(pyaps_package_file))

        # PyAPS package is typically at: /path/to/PyAPS/py/PyAPS/
        # We need to go up to: /path/to/PyAPS/

        # Check if we're in a typical structure: .../PyAPS/py/PyAPS/
        if pyaps_package_dir.endswith(os.path.join('py', 'PyAPS')):
            # Go up two levels: PyAPS -> py -> repo_root
            repo_root = os.path.dirname(os.path.dirname(pyaps_package_dir))
            return repo_root

        # Alternative structure: .../PyAPS/ (package directly in repo root)
        if os.path.basename(pyaps_package_dir) == 'PyAPS':
            # Check if parent has configs
            parent = os.path.dirname(pyaps_package_dir)
            if os.path.isdir(os.path.join(parent, 'configs')):
                return parent
            # Otherwise, this IS the repo root
            return pyaps_package_dir

    except (ImportError, AttributeError):
        pass

    try:
        # Method 2: Find it relative to the current file (aps_utils.py)
        current_file = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file)

        # Walk up the directory tree looking for configs/ExGal_configs
        check_dir = current_dir
        for _ in range(5):  # Don't go up more than 5 levels
            if os.path.isdir(os.path.join(check_dir, 'configs', 'ExGal_configs')):
                return check_dir
            parent = os.path.dirname(check_dir)
            if parent == check_dir:  # Reached root
                break
            check_dir = parent
    except NameError:
        pass

    raise RuntimeError(
        "Cannot locate PyAPS repository root directory. "
        "Please provide configdir explicitly."
    )



###########################################################################
def validate_and_set_configdir(configdir=None, verbose=True):
    """
    Validate configdir and set to default if None or invalid.

    Parameters:
    -----------
    configdir : str or None
        User-provided config directory path
    verbose : bool
        Print warning messages

    Returns:
    --------
    str : Valid config directory path

    Raises:
    -------
    RuntimeError : If no valid config directory can be found
    """

    # If configdir is provided, validate it
    if configdir is not None:
        configdir = os.path.abspath(os.path.expanduser(configdir))

        if os.path.isdir(configdir):
            if verbose:
                print(f"[APSOB] Using provided configdir: {configdir}")
            return configdir
        else:
            if verbose:
                print(f"[APSOB] ⚠️  WARNING: Provided configdir not found: {configdir}")
                print(f"[APSOB] Falling back to default PyAPS config directory...")

    # PYAPS_CONFIGDIR: a deployment-wide override, checked before the
    # repo-bundled default below. Exists for deployments (e.g. a Docker
    # image) that don't ship configs/ExGal_configs at all -- lets them
    # point at a real, persistent, writable directory (bind-mounted in)
    # without threading a new parameter through every caller across the
    # codebase. Same pattern as PYAPS_EXPLORER_URL_PREFIX elsewhere in
    # this project: one env var, read once, no new plumbing.
    env_configdir = os.environ.get("PYAPS_CONFIGDIR")
    if env_configdir:
        env_configdir = os.path.abspath(os.path.expanduser(env_configdir))
        if os.path.isdir(env_configdir):
            if verbose:
                print(f"[APSOB] Using configdir from PYAPS_CONFIGDIR: {env_configdir}")
            return env_configdir
        elif verbose:
            print(f"[APSOB] ⚠️  WARNING: PYAPS_CONFIGDIR is set but not a directory: {env_configdir}")
            print(f"[APSOB] Falling back to default PyAPS config directory...")

    # Set to default: PyAPS repo root + /configs/ExGal_configs
    try:
        pyaps_repo_root = get_pyaps_repo_root()
        default_configdir = os.path.join(pyaps_repo_root, 'configs', 'ExGal_configs')

        # Check if default config directory exists
        if os.path.isdir(default_configdir):
            if verbose:
                print(f"[APSOB] Using default configdir: {default_configdir}")
            return default_configdir
        else:
            # Try alternative: just 'configs' without 'ExGal_configs'
            alt_configdir = os.path.join(pyaps_repo_root, 'configs')
            if os.path.isdir(alt_configdir):
                if verbose:
                    print(f"[APSOB] ⚠️  WARNING: ExGal_configs not found, using: {alt_configdir}")
                return alt_configdir

            # If still not found, raise error with helpful message
            raise RuntimeError(
                f"[APSOB] ❌ ERROR: Cannot find config directory.\n"
                f"   PyAPS repo root: {pyaps_repo_root}\n"
                f"   Expected: {default_configdir}\n"
                f"   Please ensure PyAPS configs directory exists, or provide configdir explicitly."
            )

    except Exception as e:
        raise RuntimeError(
            f"[APSOB] ❌ ERROR: Cannot determine config directory: {e}\n"
            f"   Please provide a valid configdir path explicitly."
        )
###########################################################################
RVS_TEMPLATES_ENV = "PYAPS_RVS_TEMPLATES"


def resolve_rvs_template_lib(template_lib=None, verbose=True):
    """
    Return the directory that holds the RVS (rvspecfit) template library.

    The only sources are configuration values, there is no fallback to any other location:

    1. ``$PYAPS_RVS_TEMPLATES``: the generated job scripts export it from the ``templates_RVS``
       key of the ``script_params`` file in use, so the configured value wins;
    2. ``template_lib`` of the RVS config file (``configs/rvs_config.yaml`` references
       ``${PYAPS_RVS_TEMPLATES}``; a standalone user may set a real path there instead).

    Raises ``RuntimeError`` naming the configuration key when the directory is not set or does
    not exist. Returns the directory as a string with a trailing separator.
    """
    def _clean(path):
        return os.path.expanduser(os.path.expandvars(str(path)))

    env_value = os.environ.get(RVS_TEMPLATES_ENV)
    if env_value:
        path, origin = _clean(env_value), f"${RVS_TEMPLATES_ENV} (the templates_RVS key of the script_params file)"
    elif template_lib and "${" not in str(template_lib) and "$" + RVS_TEMPLATES_ENV not in str(template_lib):
        path, origin = _clean(template_lib), "template_lib of the RVS config file"
    else:
        raise RuntimeError(
            "The RVS template directory is not configured: set the templates_RVS key in your "
            "script_params file (the job scripts export it as " + RVS_TEMPLATES_ENV + "), or "
            "set template_lib in the RVS config file to a real directory.")
    if not os.path.isdir(path):
        raise RuntimeError(
            f"The RVS template directory does not exist: {path} (from {origin}). "
            f"Fix the templates_RVS key of your script_params file, or template_lib of the RVS "
            f"config file when running aps_rvs.py by hand.")
    if verbose:
        print(f"[RVS] Using RVS templates from {path} (from {origin})")
    return path if path.endswith(os.sep) else path + os.sep


###########################################################################
def check_and_fix_overlap(wlranges):
    """
    Check if wavelength ranges overlap and fix them if they do.

    For two ranges like [[3800, 5950], [5900, 9280]], splits at the
    center of overlap to get [[3800, 5925], [5925, 9280]].

    Parameters:
    -----------
    wlranges : list of lists
        List of [wmin, wmax] pairs
        Example: [[3800.0, 5950.0], [5900.0, 9280.0]]

    Returns:
    --------
    updated_wlranges : list of lists
        Updated ranges with no overlap (or original if no overlap)
        Example: [[3800.0, 5925.0], [5925.0, 9280.0]]
    """
    if len(wlranges) < 2:
        return wlranges

    # Check overlap between first two ranges
    wl1_min, wl1_max = wlranges[0]
    wl2_min, wl2_max = wlranges[1]

    overlap_min = max(wl1_min, wl2_min)
    overlap_max = min(wl1_max, wl2_max)

    if overlap_max > overlap_min:
        # Has overlap - calculate center and split
        overlap_center = round((overlap_min + overlap_max) / 2.0, 1)

        print(f"Overlap detected: [{overlap_min:.1f}, {overlap_max:.1f}] Å")
        print(f"Splitting at center: {overlap_center:.1f} Å")

        # Update ranges
        updated_ranges = [
            [wl1_min, overlap_center],
            [overlap_center, wl2_max]
        ]

        # Add any remaining ranges (if more than 2 arms)
        if len(wlranges) > 2:
            updated_ranges.extend(wlranges[2:])

        print(f"Original ranges: {wlranges}")
        print(f"Updated ranges:  {updated_ranges}")

        return updated_ranges
    else:
        # No overlap
        return wlranges


###########################################################################

def ensure_dtype(value, dtype=np.int32, return_scalar=False):
    """
    Convert any input to specified NumPy dtype, handling scalars, arrays, and strings.
    This is a safeguard for Numpy 2+ conventions or int type

    Parameters
    ----------
    value : any
        Input value (scalar, array, string, None, etc.)
    dtype : numpy dtype, default=np.int32
        Target dtype for conversion
    return_scalar : bool, default=False
        If True and input is scalar, return Python scalar instead of 0-d array

    Returns
    -------
    numpy.ndarray or scalar
        Converted value with specified dtype

    Examples
    --------
    >>> ensure_dtype(42)                    # array(42, dtype=int32)
    >>> ensure_dtype(42, return_scalar=True)  # 42 (Python int)
    >>> ensure_dtype("42")                  # array(42, dtype=int32)
    >>> ensure_dtype([1, 2, 3])            # array([1, 2, 3], dtype=int32)
    >>> ensure_dtype(None)                 # array(0, dtype=int32) or raises error
    """

    # Handle None
    if value is None:
        if dtype in [np.int32, np.int64, np.int16, np.int8]:
            default = 0
        elif dtype in [np.float32, np.float64]:
            default = 0.0
        else:
            raise ValueError(f"Cannot convert None to dtype {dtype}")
        value = default

    # Handle strings
    if isinstance(value, str):
        try:
            # Try to parse as number
            if '.' in value or 'e' in value.lower():
                value = float(value)
            else:
                value = int(value)
        except ValueError:
            # Handle special string cases
            if value.lower() in ['true', 'yes', '1']:
                value = 1
            elif value.lower() in ['false', 'no', '0']:
                value = 0
            else:
                raise ValueError(f"Cannot convert string '{value}' to {dtype}")

    # Handle pandas Series or DataFrame (if pandas is available)
    try:
        import pandas as pd
        if isinstance(value, (pd.Series, pd.DataFrame)):
            value = value.values
    except ImportError:
        pass

    # Handle lists and tuples
    if isinstance(value, (list, tuple)):
        value = np.array(value)

    # Now convert to numpy array with specified dtype
    try:
        # This handles scalars, arrays, and most other cases
        result = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError) as e:
        # Fallback: try converting to float first, then to target dtype
        try:
            temp = np.asarray(value, dtype=np.float64)
            result = temp.astype(dtype)
        except Exception:
            raise TypeError(f"Cannot convert {type(value)} to {dtype}: {e}")

    # Handle return_scalar option
    if return_scalar and result.ndim == 0:
        return result.item()

    return result
###########################################################################


def get_column_unit(rec, colname):
    """FITS TUNIT<n> value for a column of a FITS_rec (or anything else
    whose `.columns[name].unit` follows the same convention), straight
    from the file's own header rather than a hand-maintained lookup table
    — None if the file's writer never set one (common for plain
    ID/bookkeeping columns). Shared by every "show every column as a
    Parameter/Value or wide table" panel in the explorer
    (aps_IFUviewer.py, aps_MOSviewer.py) so a value table can show real
    physical units where the FITS file actually records them, per
    explicit user request."""
    try:
        unit = rec.columns[colname].unit
        return unit.strip() if unit and unit.strip() else None
    except Exception:
        return None


def apply_redshift_to_fwhm_corrected(fwhm_funcs, z_input):
    """
    Apply redshift correction to FWHM interpolation functions.

    UPDATED: Now handles both single functions and lists of functions.

    This matches your original code exactly:
    LSF[:,0] = LSF[:,0] / (1 + z_input[0])  # wavelength
    LSF[:,1] = LSF[:,1] / (1 + z_input[0])  # FWHM

    Both wavelength AND FWHM are divided by (1 + z).

    Parameters
    ----------
    fwhm_funcs : callable or list of callable
        Single FWHM function OR list of FWHM interpolation functions from apsobj.get_fwhm()
    z_input : float or array-like
        Redshift value(s). If array, uses z_input[0]

    Returns
    -------
    callable or list of callable
        Redshift-corrected FWHM interpolation function(s)
        Returns same type as input (single function -> single function, list -> list)
    """

    # Handle array input for redshift
    if hasattr(z_input, '__len__'):
        z = z_input[0]
    else:
        z = z_input

    redshift_factor = 1 + z

    # NEW: Check if input is a single function or list of functions
    is_single_function = callable(fwhm_funcs)

    if is_single_function:
        # Convert single function to list for processing
        func_list = [fwhm_funcs]
    else:
        # Already a list
        func_list = fwhm_funcs

    # Create redshift-corrected interpolation functions
    redshifted_funcs = []

    for i, original_func in enumerate(func_list):

        def create_redshifted_func(func, z_factor):
            """Create a redshifted version matching your original code."""

            def redshifted_interpolator(input_wavelengths):
                """
                Redshift-corrected interpolation function.

                This applies the same transformation as your original code:
                - Input wavelengths are used directly (no transformation)
                - FWHM values are divided by (1 + z)

                Parameters
                ----------
                input_wavelengths : array_like
                    Wavelengths (same frame as your original LSF file)

                Returns
                -------
                array_like
                    FWHM values divided by (1 + z), matching your original code
                """

                input_wavelengths = np.atleast_1d(input_wavelengths)

                # Get FWHM at input wavelengths (no wavelength transformation)
                original_fwhm = func(input_wavelengths)

                # Apply same transformation as your original code: FWHM / (1 + z)
                corrected_fwhm = original_fwhm / z_factor

                # Return scalar if input was scalar
                if len(input_wavelengths) == 1:
                    return float(corrected_fwhm)
                else:
                    return corrected_fwhm

            return redshifted_interpolator

        # Create redshifted function for this setup
        redshifted_func = create_redshifted_func(original_func, redshift_factor)
        redshifted_funcs.append(redshifted_func)

    # NEW: Return same format as input
    if is_single_function:
        # Return single function (not list)
        return redshifted_funcs[0]
    else:
        # Return list
        return redshifted_funcs

###########################################################################
def create_equivalent_LSF_Data_corrected(fwhm_funcs, z_input, wavelength_range=(3000, 10000), n_points=1000):
    """
    Create an equivalent LSF_Data interpolation function matching your original code exactly.

    Your original code:
    LSF = np.genfromtxt(file)
    LSF[:,0] = LSF[:,0] / (1 + z_input[0])  # wavelength / (1+z)
    LSF[:,1] = LSF[:,1] / (1 + z_input[0])  # FWHM / (1+z)
    LSF_Data = interp1d(LSF[:,0], LSF[:,1], 'linear', fill_value='extrapolate')

    Parameters
    ----------
    fwhm_funcs : list of callable
        FWHM interpolation functions from apsobj.get_fwhm()
    z_input : float or array-like
        Redshift value
    wavelength_range : tuple, optional
        (min_wave, max_wave) in Angstroms for interpolation grid
    n_points : int, optional
        Number of points in interpolation grid

    Returns
    -------
    callable
        Interpolation function equivalent to your original LSF_Data
    """
    import numpy as np
    from scipy.interpolate import interp1d

    # Handle redshift
    if hasattr(z_input, '__len__'):
        z = z_input[0]
    else:
        z = z_input

    redshift_factor = 1 + z

    # Use first setup (usually you'd specify which one you want)
    if len(fwhm_funcs) == 0:
        raise ValueError("No FWHM functions provided")

    # Create wavelength grid (equivalent to your LSF[:,0])
    original_wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], n_points)

    # Get FWHM values (equivalent to your LSF[:,1])
    original_fwhm = fwhm_funcs[0](original_wavelengths)

    # Apply your original transformations:
    corrected_wavelengths = original_wavelengths / redshift_factor  # LSF[:,0] / (1 + z)
    corrected_fwhm = original_fwhm / redshift_factor               # LSF[:,1] / (1 + z)

    # Create interpolation function (equivalent to your interp1d call)
    LSF_Data = interp1d(corrected_wavelengths, corrected_fwhm,
                       kind='linear', fill_value='extrapolate')

    return LSF_Data

###########################################################################




def create_empty_interpolator_dict(n_fibers=600):
    """Create empty interpolator dictionary with global + 600 fiber entries."""

    # Create the dictionary
    empty_dict = {
        'global': {}
    }

    # Add fiber entries (specnum 1 to 600)
    for specnum in range(1, n_fibers + 1):
        empty_dict[specnum] = {}

    return empty_dict


def fix_non_unicode_string(value):
    """
    convert any bytes type strings to normal unicode string
    """
    if value is not None:
        if isinstance(value, (bytes, np.bytes_)) and len(value) != 0:
            # Decode bytes to string using utf-8 encoding
            value = str(value.decode('utf-8')).strip()
        elif not isinstance(value, str):
            # If it's not bytes and not string, it's some other type
            # Handle this case based on your specific requirement
            value = str(value).strip()

    return value

###########################################################################


def _header_value(header, keyword):
    """Null/NaN-safe header lookup — local twin of `aps_L2merge.get_header_value`
    (not imported from there: aps_L2merge imports FROM aps_utils, so the
    reverse import would be circular)."""
    value = header.get(keyword, '')
    if value == '' or (isinstance(value, str) and value.strip().lower() in ('nan', 'n/a', 'none', '')):
        return ''
    if isinstance(value, float) and math.isnan(value):
        return ''
    return value


def aps_file_info(filepath, verify_checksum=False):
    """
    Purpose: minimal-I/O inspector for a final, merged PyAPS L2 `_APS.fits`
    file. Opens the file once (headers only — never touches `.data`) and
    reports which extensions it has, its schema/observing-mode/resolution/
    binning/arm-count, PyAPS + CPS versions, processing/observation dates,
    per-HDU checksum, and the APSQFLAG merge-quality bitmask.

    This is the L2 (merged-product) counterpart to `l1_fileinfo` — that
    function classifies MOS/LIFU/MIFU/MOSLIFU/MOSMIFU from an L1 file's
    OBSMODE+NAXIS, but that test does not carry over to `_APS.fits`: every
    extension there is a BinTableHDU, so NAXIS is always 2 regardless of
    whether the underlying observation was MOS or a true IFU cube. Instead:

    - **Schema** (which kind of `_APS.fits` this is) comes purely from
      which extensions are present, by NAME — never from the input
      filename/headname, which is not a reliable indicator (e.g. OBSMODE
      'LIFU' appears on both true-IFU-cube and IFU-at-fibre-level files;
      filenames like 'single_*_APS.fits' can be either true MOS or
      fibre-level LIFU/MIFU). `CLASS_TABLE` present -> the `mosL2merge`
      schema (per-target CLASS_TABLE/STAR_TABLE/GALAXY_TABLE, covering
      true MOS observations *and* IFU data processed at individual-fibre
      level). `PATCH_TABLE` present -> an IFU Voronoi-patch product,
      further split into ExGal (+`GALAXY_TABLE`) or Gal (+`STAR_TABLE`)
      — see `aps_Mapviewer.py:_load_aps_fits` for how those two are used.
    - **Observing mode** (MOS/MOSLIFU/MOSMIFU/LIFU/MIFU) comes from the
      `CONFIG_F` (on `PATCH_TABLE`) or `CNF_PPXF`/`CNF_EMI`/`CNF_LS` (on
      `GALAXY_TABLE`, present in every schema since `GALAXY_TABLE` is
      always written) header token — its value has the literal form
      `<mode><res><xbin><ybin>.json`, e.g. `'MOSLIFULR11.json'`,
      `'LIFUHR11.json'` (built by `aps_runner.py` as
      `mode+res_mode[0]+xbin+ybin+'.json'`) — the single most direct,
      already-computed encoding of mode+resolution+binning together.
      Falls back to the primary header's `OBSMODE` (`MOS`/`LIFU`/`MIFU`)
      combined with schema when no config token is found: a MOS-schema
      file with `OBSMODE=LIFU`/`MIFU` is fibre-level IFU data (MOSLIFU/
      MOSMIFU, not true LIFU/MIFU), since only a true IFU cube goes
      through `ifu*L2merge` and gets a `PATCH_TABLE`.

    Parameters
    ----------
    filepath : str or Path
        Path to a merged `_APS.fits` file.
    verify_checksum : bool, optional
        If True, recompute and verify every HDU's CHECKSUM/DATASUM against
        its actual data (forces a full-data read of the whole file — ~70s
        measured on a real multi-hundred-row file, vs ~0.02s for the
        default headers-only read). Default False: only report the
        already-computed CHECKSUM/DATASUM header values, as written by
        `writeto(..., checksum=True)` at merge time.

    Returns
    -------
    dict
        `filepath`, `extensions`, `schema` ("mos"/"ifu_exgal"/"ifu_gal"/
        "unknown"), `obsmode`, `resolution` ("HR"/"LR"/None), `xbin`,
        `ybin`, `arms` (list of arm letters/labels), `n_arms`,
        `pyaps_version`, `cps_version`, `cps_pipeline_version` (SOFTVERS,
        usually blank), `cps_datamodel_version` (L1_DMVER),
        `aps_datamodel_version` (DATAMVER), `date_obs`, `aps_date`,
        `cps_date`, `cal_date`, `apsqflag` (merge-quality bitmask, or
        None if absent), `checksums` (per-extension dict of
        `checksum`/`datasum`, plus `checksum_valid`/`datasum_valid` if
        `verify_checksum=True`).
    """
    filepath = str(filepath)
    info = dict(filepath=filepath)

    with fits.open(filepath, memmap=True, lazy_load_hdus=True) as hdul:
        extnames = [h.name for h in hdul]
        info["extensions"] = extnames
        h0 = hdul[0].header

        have_class = "CLASS_TABLE" in extnames
        have_patch = "PATCH_TABLE" in extnames
        have_galaxy = "GALAXY_TABLE" in extnames
        have_star = "STAR_TABLE" in extnames

        if have_class:
            schema = "mos"
        elif have_patch and have_galaxy:
            schema = "ifu_exgal"
        elif have_patch and have_star:
            schema = "ifu_gal"
        else:
            schema = "unknown"
        info["schema"] = schema

        # obsmode / resolution / binning, from the CONFIG_F/CNF_* token.
        config_token = None
        for ext, key in (("PATCH_TABLE", "CONFIG_F"), ("GALAXY_TABLE", "CNF_PPXF"),
                          ("GALAXY_TABLE", "CNF_EMI"), ("GALAXY_TABLE", "CNF_LS")):
            if ext in extnames:
                val = _header_value(hdul[ext].header, key)
                if val:
                    config_token = str(val)
                    break

        obsmode, resolution, xbin, ybin = None, None, None, None
        if config_token:
            token = config_token
            if token.lower().endswith(".json"):
                token = token[:-5]
            rest = token
            # Longest-prefix match first: 'LIFU' is itself a substring of
            # 'MOSLIFU', so checking short-to-long would misparse it.
            for candidate in ("MOSLIFU", "MOSMIFU", "MOS", "LIFU", "MIFU"):
                if token.startswith(candidate):
                    obsmode = candidate
                    rest = token[len(candidate):]
                    break
            if rest[:2] in ("HR", "LR"):
                resolution = rest[:2]
                bins = rest[2:]
                if len(bins) >= 2 and bins[0].isdigit() and bins[1].isdigit():
                    xbin, ybin = int(bins[0]), int(bins[1])

        if obsmode is None:
            raw_obsmode = str(_header_value(h0, "OBSMODE")).strip().upper()
            if schema == "mos":
                obsmode = {"MOS": "MOS", "LIFU": "MOSLIFU", "MIFU": "MOSMIFU"}.get(raw_obsmode) or (raw_obsmode or None)
            else:
                obsmode = raw_obsmode or None
        info["obsmode"] = obsmode
        info["resolution"] = resolution
        info["xbin"] = xbin
        info["ybin"] = ybin

        # Arms: non-empty L1_REF_B/G/R in the primary header (always
        # present as keys, '' for an unused arm); fall back to counting
        # APSREF_0/1/... on any extension header if L1_REF_* is absent.
        arms = [c for c in ("B", "G", "R") if str(_header_value(h0, f"L1_REF_{c}")).strip()]
        if not arms:
            for ext in extnames[1:]:
                h = hdul[ext].header
                n = 0
                while f"APSREF_{n}" in h:
                    n += 1
                if n:
                    arms = [f"arm{i}" for i in range(n)]
                    break
        info["arms"] = arms
        info["n_arms"] = len(arms)

        info["pyaps_version"] = _header_value(h0, "APSVERS") or None
        info["cps_version"] = _header_value(h0, "CASUVERS") or None
        info["cps_pipeline_version"] = _header_value(h0, "SOFTVERS") or None
        info["cps_datamodel_version"] = _header_value(h0, "L1_DMVER") or None
        info["aps_datamodel_version"] = _header_value(h0, "DATAMVER") or None

        info["date_obs"] = _header_value(h0, "DATE-OBS") or None
        info["aps_date"] = _header_value(h0, "APSDATE") or None
        info["cps_date"] = _header_value(h0, "CASUDATE") or None
        info["cal_date"] = _header_value(h0, "CALDATE") or None

        checksums = {}
        for ext in extnames:
            h = hdul[ext].header
            checksums[ext] = dict(checksum=h.get("CHECKSUM"), datasum=h.get("DATASUM"))
            if verify_checksum:
                hdu = hdul[ext]
                checksums[ext]["checksum_valid"] = bool(hdu.verify_checksum())
                checksums[ext]["datasum_valid"] = bool(hdu.verify_datasum())
        info["checksums"] = checksums

        apsqflag = _header_value(h0, "APSQFLAG")
        info["apsqflag"] = int(apsqflag) if apsqflag != '' else None

    return info


###########################################################################



# determine the stacking level based on the filename
def stack_level(file_path):
    # Extract the filename using pathlib
    fname_path = Path(file_path)
    extracted_filename = fname_path.name

    # Mapping keywords to stacking levels
    # order is important to make sure higher stackings are always getting checked before the super only
    stacking_levels = {
        'wve_': 4,
        'super': 3,
        'stack': 1,
        'single': 0,
    }
    # Determine the stacking level based on keywords in the filename
    stacking_level = 4
    for keyword, level in stacking_levels.items():
        if keyword in str(extracted_filename).lower():
            stacking_level = level
            break

    return stacking_level
###########################################################################

def l1_to_image(targs, pixelsize=1.2):

    """
    Purpose: Generate an image-like numpy array from L1 (after reading by APSOB)
    """

    if pixelsize is None:
        pixelsize=1.2

    # Generate sum over arms of mean of fluxes
    _flux_arr = [ sum(np.nanmean(targ.spectra[ii].flux) for ii in range(targs.nbands()) ) for targ in targs._targetlist]


    # create a coordinate array where coord_arr[:,0] are RAs and coord_arr[:,1] are Decs
    coord_arr = np.asarray([(targ.targra, targ.targdec) for i_targ, targ in enumerate(targs._targetlist)])
    aps_id_arr = np.asarray([targ.aps_id for i_targ, targ in enumerate(targs._targetlist)], dtype=np.int32)


    # Create image in pixels
    xmin = np.nanmin(coord_arr[:,0]*3600.0)-6;  xmax = np.nanmax(coord_arr[:,0]*3600.0)+6
    ymin = np.nanmin(coord_arr[:,1]*3600.0)-6;  ymax = np.nanmax(coord_arr[:,1]*3600.0)+6
    npixels_x = int( np.round( (xmax - xmin)/pixelsize ) + 1 )
    npixels_y = int( np.round( (ymax - ymin)/pixelsize ) + 1 )
    i = np.array( np.round( (coord_arr[:,0]*3600.0 - xmin)/pixelsize ), dtype=np.int32 )
    j = np.array( np.round( (coord_arr[:,1]*3600.0 - ymin)/pixelsize ), dtype=np.int32 )
    image = np.full( (npixels_x, npixels_y), np.nan )
    image[i,j] = _flux_arr

    return image
###########################################################################
## find overlap among a set of wavelengths ranges [[wl1,wl2], [wl1,wl2], [wl1,wl2], ..]
## It can evaluate the overlap, even if we had more than two wavelength ranges. e.g. BGR setting
def overlap_finder(listoflist):
    assert isinstance(listoflist, list), "Input must be a list of list"
    assert isinstance(listoflist[0], list), "Input must be a list of list"
    if len(listoflist) > 1:
        ovl = 0.0
        for i in range(0, len(listoflist) - 1):
            ovl = ovl + np.abs(
                max(
                    0,
                    min(listoflist[i][1], listoflist[i + 1][1])
                    - max(listoflist[i][0], listoflist[i + 1][0]),
                )
            )
        return ovl
    else:
        return 0.0


###########################################################################
## check if the list is alphabetic order or not
## Used to check the order of arms that should be ['B', 'R'] or ['B', 'G', 'R']
## It also returns false it we had duplicated entry


def isInAlphabeticalOrder(alist):
    for i_in_list in range(len(alist) - 1):
        if alist[i_in_list] >= alist[i_in_list + 1]:
            return False
    return True


###########################################################################
## check if the list is ascending or not
def isAscending(aps_idlist):
    previous = aps_idlist[0]
    for number in aps_idlist:
        if number < previous:
            return False
        previous = number
    return True


###########################################################################
## To pass None keyword as command line argument
## Update (9 September 2020): It also expand path from variable mode into the expanded mode
def none_or_str(value):
    if value.upper() == "NONE":
        return None
    return os.path.expanduser(os.path.expandvars(value))


###########################################################################
## To pass Boolean keyword as command line argument


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif value.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


###########################################################################
## Too add wing to a list of Booleans
def add_wing_to_boolean_list(input_list, wing_len, up_range, low_range=0):
    """
    input_list : input Boolean list
    wing_len: how many indexes to be added to each side of each true index
    low_range: minimum acceptable index (default: 0)
    up_range: maximum acceptable index
    Output: an extended list of indices with True (it return indices of True values instead of an Boolean list)
    """
    wing_len = int(wing_len)
    input_list = np.ravel(np.where(input_list))
    extend_list=[list(range(i_il-wing_len,i_il+wing_len+1)) for i_il in input_list]
    flat_extend_list = [item for sublist in extend_list for item in sublist]
    extend_list_array  = np.array(flat_extend_list)
    extend_list_array = extend_list_array[(extend_list_array >= low_range) & (extend_list_array < up_range)]
    return list(set(extend_list_array))


###########################################################################
def resolution_mat_torows(mat):
    """Convert resolution matrix from columns to rows format."""
    w = mat.shape[0]
    w2 = w // 2
    return np.array([np.roll(mat[_], _ - w2) for _ in range(w)])[::-1]

def resolution_mat_tocolumns(mat):
    """Convert resolution matrix from rows to columns format."""
    w = mat.shape[0]
    w2 = w // 2
    return np.array([np.roll(mat[::-1][_], w2 - _) for _ in range(w)])

###########################################################################

def deconvolve_resolution_matrix(mat0, sigma0_angstrom=0.5, pix_size_angstrom=0.8):
    """
    Deconvolve instrumental resolution using quadratic subtraction.

    For Gaussian PSFs: σ_deconv² = σ_data² - σ_template²

    Parameters
    ----------
    mat0 : ndarray
        Input resolution matrix (width × npix)
    sigma0_angstrom : float
        Template instrumental resolution in Angstroms (Gaussian sigma, not FWHM)
    pix_size_angstrom : float
        Pixel size in Angstroms

    Returns
    -------
    mat_deconv : ndarray
        Deconvolved resolution matrix

    Notes
    -----
    This function assumes:
    - Resolution profiles are approximately Gaussian
    - Template resolution is constant across wavelength
    - Template resolution < data resolution (otherwise original profile is kept)
    """
    width, npix = mat0.shape
    sig_template_pix = sigma0_angstrom / pix_size_angstrom
    w2 = width // 2
    xs = np.arange(width) - w2

    mat_deconv = np.zeros_like(mat0)
    n_failed = 0  # Track how many pixels couldn't be deconvolved

    for i in range(npix):
        # Get resolution profile for this wavelength
        profile = mat0[:, i]

        # Normalize profile
        profile_sum = np.sum(profile)
        if profile_sum < 1e-10:
            # Empty or invalid profile - keep as-is
            mat_deconv[:, i] = profile
            n_failed += 1
            continue

        profile_norm = profile / profile_sum

        # Calculate second moment to get sigma_data
        # Use abs() to handle any numerical noise
        sigma_data_pix = np.sqrt(np.sum(xs**2 * np.abs(profile_norm)))

        # Quadratic subtraction
        sigma_deconv_pix_sq = sigma_data_pix**2 - sig_template_pix**2

        # Safety check: can't deconvolve if template is broader than data
        if sigma_deconv_pix_sq <= 0:
            # Template is too broad - keep original
            mat_deconv[:, i] = profile
            n_failed += 1
        else:
            sigma_deconv_pix = np.sqrt(sigma_deconv_pix_sq)

            # Build new Gaussian with deconvolved width
            new_profile = np.exp(-0.5 * (xs / sigma_deconv_pix)**2)
            new_profile /= np.sum(new_profile)

            mat_deconv[:, i] = new_profile

    # Report statistics
    if n_failed > 0:
        pct_failed = 100 * n_failed / npix
        print(f"  Note: {n_failed}/{npix} pixels ({pct_failed:.1f}%) could not be deconvolved")
        print(f"        (template σ={sigma0_angstrom:.3f}Å may be too broad for some wavelengths)")

    return mat_deconv


def construct_resolution_sparse_matrix(mat, pix_size_angstrom=None, sigma0_angstrom=None):
    """
    Construct sparse resolution matrix with optional deconvolution.

    Parameters
    ----------
    mat : ndarray
        Input resolution matrix (width × npix)
    pix_size_angstrom : float, optional
        Pixel size for deconvolution
    sigma0_angstrom : float, optional
        Template resolution for deconvolution

    Returns
    -------
    M : scipy.sparse.dia_matrix
        Sparse diagonal resolution matrix
    """
    width, npix = mat.shape
    w2 = width // 2
    mat = mat.copy()

    # Deconvolve if parameters provided
    if pix_size_angstrom is not None and sigma0_angstrom is not None:
        mat = deconvolve_resolution_matrix(
            mat,
            pix_size_angstrom=pix_size_angstrom,
            sigma0_angstrom=sigma0_angstrom
        )

    # Fix edge normalization
    mat_rows = resolution_mat_torows(mat)
    mult = np.median(mat_rows.sum(axis=0))
    if mult == 0:
        mult = 1

    for i in range(w2):
        N1 = mat_rows[w2 - i:, i].sum()
        mat_rows[:, i] = mat_rows[:, i] / (N1 + (N1 == 0)) * mult
        j = npix - 1 - i
        N2 = mat_rows[:w2 + 1 + i, j].sum()
        mat_rows[:, j] = mat_rows[:, j] / (N2 + (N2 == 0)) * mult

    mat = resolution_mat_tocolumns(mat_rows)
    M = scipy.sparse.dia_matrix((mat, np.arange(w2, -w2 - 1, -1)), (npix, npix))

    return M

###########################################################################

def makeR(la, inst_res, cache_Rcsr=False):

    FWHM_ang = la / inst_res
    # FWHM_ang=(la*0.0) + np.mean(la)/inst_res
    wd = FWHM_ang / (2.355 * (la[1] - la[0]))

    # in case we have an array of LA or FWHM per pixel per spectrum
    # ii = np.arange(la.shape[1])

    ww = wd < 1e-5
    wd[ww] = 2.0

    ii = np.arange(len(wd))
    di = ii - ii[:, None]
    di2 = di**2
    ndiag = int(4 * np.ceil(wd.max()) + 1)
    nbins = len(wd)

    ## build resolution from wdisp
    ## For more details, refer to the line 99 of 'gconv.pro' by C. Allende
    reso = np.zeros([ndiag, nbins])

    for idiag in range(ndiag):
        offset = ndiag // 2 - idiag
        d = np.diagonal(di2, offset=offset)
        if offset < 0:
            reso[idiag, : len(d)] = np.exp(-d / 2 / wd[: len(d)] ** 2)
        else:
            reso[idiag, nbins - len(d) : nbins] = np.exp(
                -d / 2 / wd[nbins - len(d) : nbins] ** 2
            )

        # in case we have an array of LA or FWHM per pixel per spectrum
        # if offset<0:
        #     reso[idiag,:len(d)] = np.exp(-d/2/wd[i,:len(d)]**2)
        # else:
        #     reso[idiag,nbins-len(d):nbins]=np.exp(-d/2/wd[i,nbins-len(d):nbins]**2)

    reso /= np.sum(reso, axis=0)
    offsets = ndiag // 2 - np.arange(ndiag)
    nwave = reso.shape[1]
    R = sparse.dia_matrix((reso, offsets), (nwave, nwave))

    if cache_Rcsr:
        Rcsr = R.tocsr()
    else:
        Rcsr = None

    return R, Rcsr


###########################################################################


def islandinfo(y, trigger_val, stopind_inclusive=True):
    # Setup "sentients" on either sides to make sure we have setup
    # "ramps" to catch the start and stop for the edge islands
    # (left-most and right-most islands) respectively

    # NOTE:
    # given that we found that few pixel around gaps usually have wrong values, we mask them as well.
    # island_offset set the length of this area around the gaps

    y_ext = np.r_[False, y == trigger_val, False]

    # Get indices of shifts, which represent the start and stop indices
    idx = np.flatnonzero(y_ext[:-1] != y_ext[1:])

    # Lengths of islands if needed
    gap_len = idx[1::2] - idx[:-1:2]

    gap_isl = list(zip(idx[:-1:2], idx[1::2] - int(stopind_inclusive)))

    # Using a step-size of 2 would get us start and stop indices for each island
    return gap_isl, gap_len


###########################################################################
def wave_to_sigma2(wave):
    """
    Internal support routine to convert to wavevnumber^2.
    """
    return (1.0e4 / wave) ** 2


###########################################################################
def conv_factor(sigma2):
    """
    Internal support routine to compute conversion factor.
    """
    c0 = 1.0
    c1 = 5.792105e-2
    c2 = 238.0185e0
    c3 = 1.67917e-3
    c4 = 57.362e0
    return c0 + c1 / (c2 - sigma2) + c3 / (c4 - sigma2)


###########################################################################
# Modules to enable inter-conversion between air and vacuum wavelengths.
# Uses formula from Ciddor 1996, Applied Optics, 35, 1566.
# Adapted from vactoair.pro & airtovac.pro (Lindler/Landsman/Schlegel).
# This Python implementation by A. Bolton, U. Utah, 2011 March.
# The two functions of interest are:
#  air_wave = airtovac.v2a(vac_wave)
#  vac_wave = airtovac.a2v(air_wave)
# with all wavelengths in Angstroms.
# No correction made for wavelengths below 2000 Angstroms.
# """
def v2a(vac_wave):
    """
    Convert vacuum wavelengths in Angstroms to air wavelengths.
    No correction for wavelengths blue-ward of 2000 Angstroms.
    """
    wmin = 2000.0
    air_wave = vac_wave / conv_factor(wave_to_sigma2(vac_wave))
    # Only apply conversion above 2000Ang:
    wave_test = vac_wave >= wmin
    air_wave = air_wave * wave_test + vac_wave * (1 - wave_test)
    air_wave.setflags(write=1)
    return air_wave


###########################################################################
def a2v(air_wave):
    """
    Convert air wavelengths in Angstroms to vacuum wavelengths.
    No correction for wavelengths blue-ward of 2000 Angstroms.
    """
    wmin = 2000.0
    vac_wave = air_wave * conv_factor(wave_to_sigma2(air_wave))
    # Iterate once for accuracy:
    vac_wave = air_wave * conv_factor(wave_to_sigma2(vac_wave))
    # Only apply conversion above 2000Ang:
    wave_test = air_wave >= wmin
    vac_wave = vac_wave * wave_test + air_wave * (1 - wave_test)
    vac_wave.setflags(write=1)
    return vac_wave


###########################################################################
def gap_filler(lam, spec, badmask):
    """Mask as spectrum by linearly interpolating across a badmask"""
    spec1 = spec * 1
    xbad = np.nonzero(badmask)[0]
    xgood = np.nonzero(~badmask)[0]
    xpos = np.searchsorted(xgood, xbad)
    leftedge = xpos == 0
    rightedge = xpos == len(xgood)
    mid = (~leftedge) & (~rightedge)
    l1, l2 = lam[xgood[xpos[mid] - 1]], lam[xgood[xpos[mid]]]
    s1, s2 = spec[xgood[xpos[mid] - 1]], spec[xgood[xpos[mid]]]
    l0 = lam[xbad[mid]]
    spec1[xbad[leftedge]] = spec[xgood[0]]
    spec1[xbad[rightedge]] = spec[xgood[-1]]
    spec1[xbad[mid]] = (-(l1 - l0) * s2 + (l2 - l0) * s1) / (l2 - l1)
    return spec1


###########################################################################
def wave_interpol(wavelist):
    """
    BASE ON the original code: wave_little_interpol from astropy.spectrum
    https://spectrum.readthedocs.io/en/latest/_modules/spectrum/coadd.html#coadd_errorweighted

        Parameters
        ----------
        wavelist : list of 1-dim ndarrays
            input list of wavelength

        Returns
        -------
        waveout : ndarray
            wavelength array that can be used to co-adding all echelle orders.
    """
    mins = np.array([min(w) for w in wavelist])
    maxs = np.array([max(w) for w in wavelist])

    if np.any(np.argsort(mins) != np.arange(len(wavelist))):
        raise ValueError("List of wavelengths must be sorted in increasing order.")
    # if not np.all(maxs[:-1] > mins[1:]):
    #     raise ValueError('Not all orders overlap.')
    if np.any(mins[2:] < maxs[:-2]):
        raise ValueError("No order can be completely overlapped.")

    waveout = [wavelist[0][wavelist[0] < mins[1]]]
    wave1 = np.copy(waveout)

    #### overlap region ####
    # No assumptions on how bin edges of different orders match up
    # overlap start and stop are the last and first "clean" points.

    overlap_start = np.max(waveout[-1])
    overlap_end = np.min(wavelist[1][wavelist[1] > maxs[0]])
    # In overlap region patch in a linear scale with slightly different step.
    dw = overlap_end - overlap_start
    step = 0.5 * (np.mean(np.diff(wavelist[0])) + np.mean(np.diff(wavelist[1])))
    n_steps = ensure_dtype((dw / step + 0.5), dtype=np.int32, return_scalar=True)


    wave_overlap = np.linspace(overlap_start + step, overlap_end - step, n_steps - 1)

    waveout.append(wave_overlap)

    #### next region without overlap ####
    wave2 = wavelist[1][(wavelist[1] > maxs[0])]
    waveout.append(wave2)

    return (
        np.ravel(np.hstack(waveout)),
        np.ravel(wave1),
        np.ravel(wave_overlap),
        np.ravel(wave2),
    )


# Print args and assigned/default values on the screen and save output to a file
#################################################################################################
def print_args(args, module=None, version=None, path=None, headname=None, screen_only=False):

    if not screen_only:
        # check if both headname and path are available
        if (headname is None) or (path is None):
            return

        headname = str(headname).replace(" ", "")
        argfile = (
            path
            + "/"
            + headname
            + "_"
            + module
            + "-"
            + datetime.datetime.now().strftime("%Y_%m_%d-%H_%M")
            + ".txt"
        )
        f = open(argfile, "a")

    print()
    print(
        "------------------------------- INPUT PARAMETERS -----------------------------------"
    )
    print()
    print("MODULE: %s  VERSION %s" % (module, version))
    print("Date, time:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    print()
    for argp in vars(args):
        print("Par: %-20s val: %s" % (argp, getattr(args, argp)))
    print(
        "------------------------------------------------------------------------------------"
    )

    if not screen_only:
        print("MODULE: %s  VERSION %s" % (module, version), file=f)
        print("Date, time:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), file=f)
        for argp in vars(args):
            print("Par: %-20s val: %s" % (argp, getattr(args, argp)), file=f)
        f.close()


###########################################################################
def filter_targetlist(targ_list_master, min_snr=0, cname_list = None, TARGUSE=['T', 'C']):

    mod_targ_list_master = []
    temp_dict = {}
    for targ_im in targ_list_master:

        if TARGUSE is not None:
            if targ_im.targuse not in TARGUSE:
                continue

        if min_snr is not None:
            if np.nanmean([targ_im.meta[i]['SNR'] for i, _ in enumerate(targ_im.meta)]) < min_snr :
                continue

        # now only keep those CNAME presented in the cname_list and keep them in a dict where key is the cname
        if cname_list is not None:
            if targ_im.cname in cname_list:
                if targ_im.cname not in temp_dict:
                    temp_dict[targ_im.cname] = [targ_im]
                else:
                    temp_dict[targ_im.cname].append(targ_im)
    # loop over keys (cnames) and select only the element with the highest SNR
    for key in temp_dict:
        if len(temp_dict[key]) > 1:
            # max_snr = np.nanmax([np.nanmean([element.meta[i]['SNR'] for i, _ in enumerate(element.meta)]) for element in temp_dict[key]])

            # _max_snr = np.nanmax([element.meta[1]['SNR'] for element in temp_dict[key]])
            _max_snr = np.nanmax( [np.nanmean([element.meta[ii]['SNR']  for ii in range(len(element.spectra))])  for element in temp_dict[key]])

            for element in temp_dict[key]:
                if np.nanmean([element.meta[ii]['SNR'] for ii in range(len(element.spectra))]) == _max_snr:
                    mod_targ_list_master.append(element)
                    break
        elif len(temp_dict[key]) == 1:
            mod_targ_list_master.append(temp_dict[key][0])
        else:
            print('Warning: No target with this CNAME found in the target list')

    assert len(temp_dict) == len(mod_targ_list_master)
    return mod_targ_list_master

###########################################################################
def read_infiles_list(file_path):
    """
    Read a an ascii file where each row is list of infiles (separated by comma) and return a list of infiles
    """

    infiles_list = []

    with open(file_path, "r") as file:
        for line in file:
            file_paths = line.strip().split(",")
            if len(file_paths) == 1:
                infiles_list.append([file_paths[0].strip()])
            elif len(file_paths) == 2:
                infiles_list.append([file_paths[0].strip(), file_paths[1].strip()])
            else:
                # `raise`, not `sys.exit()` -- see APSOB.__init__'s own
                # comment on this same fix: SystemExit isn't caught by
                # `except Exception`, so this would kill the whole
                # gunicorn worker in server mode instead of failing just
                # this one load.
                raise RuntimeError("Invalid infiles_list format")
    return infiles_list
###########################################################################


def mp_array(original):
    """Allocate a raw shared memory buffer and wrap it in an ndarray.

    This allocates a multiprocessing.RawArray and wraps the buffer
    with an ndarray.

    Args:
        typcode (str): the type code of the array.
        size_or_init: passed to the RawArray constructor.

    Returns;
        ndarray: the wrapped data.

    """

    typecode = original.dtype.char
    shape = original.shape

    # raw = mp.RawArray(typecode, original.ravel())
    # nd = np.array(raw, dtype=typecode, copy=False).view()
    # nd.shape = shape

    np_type_to_ctype = {
        "f": ctypes.c_float,
        "d": ctypes.c_double,
        "?": ctypes.c_bool,
        "i": ctypes.c_ubyte,
        "I": ctypes.c_long,
        "U": ctypes.c_char_p,
    }

    numel = len(original)
    arr_ctypes = mp.sharedctypes.RawArray(np_type_to_ctype[typecode], numel)
    nd = np.frombuffer(arr_ctypes, dtype=original.dtype, count=numel)
    nd.shape = shape
    nd[...] = original

    return nd


###########################################################################################################
# this return the aps_ids of the targets, satisfied the class condition.
# It also accept the nthreads and threadid for parallel processing

def aps_ids_class(classfile, classlist, aps_ids=None, nthreads=1, threadid=0, rank=3, ncchar=2):
    """
    Return the aps_ids of targets that match the specified class conditions.

    This function works with both single mode (ntop=1) and multiple mode (ntop>1) zbest tables.
    It checks the top 'rank' classifications for each target and returns APS_IDs that match
    any of the requested classes in classlist.

    Parameters:
    -----------
    classfile : str
        Path to FITS file containing CLASS and SRVY_CLASS columns
    classlist : list of str
        List of class names to search for (e.g., ['STAR', 'WD', 'GALAXY'])
    aps_ids : array-like, optional
        If provided, only return matches from this subset of APS_IDs
    nthreads : int
        Number of parallel threads for distribution
    threadid : int
        Current thread ID (0 to nthreads-1)
    rank : int
        How many top-ranked classifications to check per target (e.g., rank=3 checks top 3)
    ncchar : int
        Number of characters to compare for class matching (default=2, e.g., 'ST' matches 'STAR')

    Returns:
    --------
    ids_in_class : ndarray
        Array of APS_IDs matching the class criteria for this thread
    """
    import sys

    from astropy.io.fits import getdata

    try:
        # READ classfile
        print(f"Reading classification file: {classfile}")
        class_table = getdata(classfile, 1)

        APS_ID_CLASS = np.array(class_table["APS_ID"], dtype=np.int32)
        n_total_targets = len(APS_ID_CLASS)

        print(f"Total targets in file: {n_total_targets}")
        print(f"Searching for classes: {classlist} (matching first {ncchar} characters)")
        print(f"Checking top {rank} classifications per target")

        # Get CLASS and SRVY_CLASS columns
        targ_cls1_raw = class_table["CLASS"]
        targ_cls2_raw = class_table["SRVY_CLASS"]

        # Detect if we're in single mode (ntop=1) or multiple mode (ntop>1)
        is_single_mode = targ_cls1_raw.ndim == 1

        if is_single_mode:
            print("Detected SINGLE mode (ntop=1): CLASS columns are 1D arrays")
            # Single mode: each element is a single string, wrap in list for uniform processing
            targ_cls1 = [[np.char.strip(cls).decode() if isinstance(cls, bytes) else np.char.strip(cls)]
                         for cls in targ_cls1_raw]
            targ_cls2 = [[np.char.strip(cls).decode() if isinstance(cls, bytes) else np.char.strip(cls)]
                         for cls in targ_cls2_raw]
            actual_rank = min(rank, 1)  # Can only check rank 0 in single mode
            if rank > 1:
                print(f"WARNING: rank={rank} requested but only 1 classification per target available")
                print(f"         Using rank=1 instead")
        else:
            print(f"Detected MULTIPLE mode (ntop={targ_cls1_raw.shape[1]}): CLASS columns are 2D arrays")
            # Multiple mode: each element is already an array of classifications
            targ_cls1 = []
            targ_cls2 = []
            for i in range(len(targ_cls1_raw)):
                # Handle both string and byte types, strip whitespace
                cls1_row = []
                cls2_row = []
                for j in range(len(targ_cls1_raw[i])):
                    cls1 = targ_cls1_raw[i][j]
                    cls2 = targ_cls2_raw[i][j]

                    # Convert bytes to string if needed and strip
                    if isinstance(cls1, bytes):
                        cls1 = cls1.decode().strip()
                    else:
                        cls1 = str(cls1).strip()

                    if isinstance(cls2, bytes):
                        cls2 = cls2.decode().strip()
                    else:
                        cls2 = str(cls2).strip()

                    cls1_row.append(cls1)
                    cls2_row.append(cls2)

                targ_cls1.append(cls1_row)
                targ_cls2.append(cls2_row)

            actual_rank = min(rank, targ_cls1_raw.shape[1])
            if rank > targ_cls1_raw.shape[1]:
                print(f"WARNING: rank={rank} requested but only {targ_cls1_raw.shape[1]} classifications per target available")
                print(f"         Using rank={actual_rank} instead")

        # Find targets matching the class criteria
        # Check first 'actual_rank' classifications for each target
        print(f"\nSearching for matches in top {actual_rank} rank(s)...")

        xids = []
        matches_by_class = {cls: 0 for cls in classlist}
        matches_by_rank = {r: 0 for r in range(actual_rank)}

        for apsid_c, apsid_i in enumerate(class_table['APS_ID']):
            matched = False
            match_info = []

            # Check each requested class
            for target_class in classlist:
                target_class_lower = target_class.lower()[:ncchar]

                # Check CLASS column (from Redrock)
                for r in range(actual_rank):
                    cls1 = targ_cls1[apsid_c][r].lower()[:ncchar]
                    if target_class_lower == cls1:
                        if not matched:
                            xids.append(apsid_c)
                            matched = True
                        matches_by_class[target_class] += 1
                        matches_by_rank[r] += 1
                        match_info.append(f"CLASS[rank{r}]={targ_cls1[apsid_c][r]}")
                        break  # Found match for this class, no need to check other ranks

                # Check SRVY_CLASS column (from survey expectations)
                if not matched:
                    for r in range(actual_rank):
                        cls2 = targ_cls2[apsid_c][r].lower()[:ncchar]
                        if target_class_lower == cls2:
                            if not matched:
                                xids.append(apsid_c)
                                matched = True
                            matches_by_class[target_class] += 1
                            matches_by_rank[r] += 1
                            match_info.append(f"SRVY_CLASS[rank{r}]={targ_cls2[apsid_c][r]}")
                            break

            # Log first few matches for debugging
            if matched and len(xids) <= 5:
                print(f"  Match {len(xids)}: APS_ID={apsid_i}, {', '.join(match_info)}")

        # Convert to numpy array
        xids = np.array(xids, dtype=np.int32)
        n_matched = len(xids)

        print(f"\nMatching summary:")
        print(f"  Total matches found: {n_matched}/{n_total_targets} ({100*n_matched/n_total_targets:.1f}%)")
        for cls, count in matches_by_class.items():
            if count > 0:
                print(f"  '{cls}': {count} targets")
        print(f"\n  Matches by rank:")
        for r, count in matches_by_rank.items():
            if count > 0:
                print(f"    Rank {r}: {count} targets")

        # Match with input aps_ids if provided
        if aps_ids is not None:
            print(f"\nFiltering by input APS_IDs list (n={len(aps_ids)})...")
            ids_xids = APS_ID_CLASS[xids]
            fxm = np.nonzero(np.isin(ids_xids, np.array(aps_ids)))[0]

            if len(fxm) == 0:
                print(f"WARNING: None of the {n_matched} matched targets are in the input APS_IDs list")
                print(f"         Class list searched: {classlist}")
                return np.array([], dtype=np.int32)

            xids = xids[fxm]
            print(f"  After filtering: {len(xids)} targets remain")
        else:
            print("\nWARNING: No APS_IDs filter provided - using all matched targets")

        # Distribute across threads
        if len(xids) > 0 and nthreads > 1:
            print(f"\nDistributing {len(xids)} targets across {nthreads} threads...")
            tids = np.linspace(0, nthreads, len(xids), False).astype(int)
            assert tids.max() <= (nthreads - 1), f"Thread assignment error: max tid={tids.max()}, nthreads={nthreads}"

            targets_per_thread = np.bincount(tids, minlength=nthreads)
            print(f"  Targets per thread: {targets_per_thread}")
            print(f"  This thread (threadid={threadid}) gets: {targets_per_thread[threadid]} targets")

            xids = xids[tids == threadid]
        elif nthreads == 1:
            print(f"\nSingle-threaded mode: all {len(xids)} targets assigned to thread 0")

        if len(xids) == 0:
            print(f"ERROR: No targets assigned to threadid={threadid}")
            print(f"       This can happen if: 1) No matches found, 2) Thread distribution issue")
            return np.array([], dtype=np.int32)

        # Final result
        ids_in_class = APS_ID_CLASS[xids]

        print(f"\nFINAL RESULT for threadid={threadid}:")
        print(f"  Returning {len(ids_in_class)} APS_IDs: {ids_in_class[:10]}{'...' if len(ids_in_class) > 10 else ''}")
        print(f"  Classes matched: {classlist}")
        print(f"  Ranks checked: 0 to {actual_rank-1}")
        print("")

        return ids_in_class

    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print(f"\nERROR in aps_ids_class:")
        print(f"  Exception type: {exc_type.__name__}")
        print(f"  Exception message: {exc_value}")
        print(f"  Traceback:")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return np.array([], dtype=np.int32)


###########################################################################################################
def index_in_class(classfile, classlist, aps_ids=None, ncchar=2):
    """
    return the index of first occurance of an specific classtype (given as classlist) in the classfile for an specific aps_id in the class file and all baseic z/class for the top answer
    """

    # READ classfile
    class_table = getdata(classfile, 1)
    # keep a list of all available aps_ids in the class file
    aps_ids_worker = list(class_table['APS_ID'])


    if aps_ids is not None:
        assert isinstance(aps_ids, list), 'input aps_ids must be a list or None (to consider all possible aps_ids)'
    else:
        aps_ids = deepcopy(aps_ids_worker)


    # taking care of default index_dict_i for the execption handling
    assert isinstance(classlist, list), 'input classlist must be a list'
    index_dict_defult = dict()
    for element in classlist:
        index_dict_defult[element] = np.nan


    # now create the placeholder for all results
    index_dict_all = dict()
    for c_apsid_i, apsid_i in enumerate(aps_ids):
        try:
            if apsid_i not in aps_ids_worker:
                print(f'Warning: Class info is not available for this aps_ids:{apsid_i}')
            # find the index of this specific aps_id
            id_in_class= np.ravel(np.where(class_table['APS_ID'] == apsid_i))[0]

            # handle single and multiple class/z output from redrock
            clslist = class_table['CLASS'][id_in_class]
            clslist = [clslist] if isinstance(clslist, (bytes, str)) else list(clslist)
            clslist = [clslist_i.decode() if isinstance(clslist_i, bytes) else clslist_i for clslist_i in clslist]
            subclslist = class_table['SUBCLASS'][id_in_class]
            subclslist = [subclslist] if isinstance(subclslist, (bytes, str)) else list(subclslist)
            subclslist = [subclslist_i.decode() if isinstance(subclslist_i, bytes) else subclslist_i for subclslist_i in subclslist]
            zlist = class_table['Z'][id_in_class]
            zlist = [zlist] if zlist.ndim == 0 else list(zlist)
            zerrlist = class_table['ZERR'][id_in_class]
            zerrlist = [zerrlist] if zerrlist.ndim == 0 else list(zerrlist)
            zwarnlist = class_table['ZWARN'][id_in_class]
            zwarnlist = [zwarnlist] if zwarnlist.ndim == 0 else list(zwarnlist)


            # Create a dictionary to store the first occurrence indices in clslist and also a list to keep all results
            index_dict_i = {}
            list_of_index_in_clslist = []

            for element in classlist:
                # Find the first occurrence index in clslist
                index_in_clslist = next((i for i, item in enumerate(clslist) if item.lower()[:ncchar] == element.lower()[:ncchar]), None)

                index_dict_i[element] = index_in_clslist
                list_of_index_in_clslist.append(index_in_clslist)
            # print(index_dict_i)

            # Check if all elements are None or if the list is empty
            if all(x is None for x in list_of_index_in_clslist) or not list_of_index_in_clslist:
                min_index = 0
                flag_i = -1
                print(f"Warning: APS_ID: {apsid_i}, Available classes: {clslist}")
                print('Warning: No sutibale index for this specific class type found. Returning the first possible answer (index=0).')
                print('Warning: Most porbably this target has a predefined class based on SRVY_CLASS rather than CLASS from redrock.')

            else:
                min_index = min(list_of_index_in_clslist, key=lambda x: float('inf') if x is None else x)
                flag_i = 0

            index_dict_all[apsid_i] = {'min_index':min_index, 'index_dict': index_dict_i, 'flag':flag_i, 'CLASS':clslist[min_index], 'Z':zlist[min_index], 'ZERR':zerrlist[min_index], 'ZWARN':zwarnlist[min_index], 'SUBCLASS':subclslist[min_index]}

        except Exception as e:
            index_dict_all[apsid_i] = {'min_index':np.nan, 'index_dict': index_dict_defult, 'flag':-2, 'CLASS':'', 'Z':np.nan, 'ZERR':np.nan, 'ZWARN':np.nan, 'SUBCLASS':''}

    return index_dict_all
###########################################################################################################
def index_in_class_patch(patch_table, classlist, ids=None, ncchar=2):
    """
    return the index of first occurance of an specific classtype (given as IFU Patch table) in the classfile for an specific id in the class file and all baseic z/class for the top answer
    """

    # READ patch_table
    # keep a list of all available ids in the class file
    ids_worker = list(patch_table['id'])


    if ids is not None:
        assert isinstance(ids, list), 'input ids must be a list or None (to consider all possible ids)'
    else:
        ids = deepcopy(ids_worker)


    # taking care of default index_dict_i for the execption handling
    assert isinstance(classlist, list), 'input classlist must be a list'
    index_dict_defult = dict()
    for element in classlist:
        index_dict_defult[element] = np.nan


    # now create the placeholder for all results
    index_dict_all = dict()
    for c_id_i, id_i in enumerate(ids):
        try:
            if id_i not in ids_worker:
                print(f'Warning: Class info is not available for this ids:{id_i}')
            # find the index of this specific aps_id
            id_in_class= np.ravel(np.where(patch_table['id'] == id_i))[0]

            # handle single and multiple class/z output from redrock
            clslist = patch_table['CLASS'][id_in_class]
            clslist = [clslist] if isinstance(clslist, (bytes, str)) else list(clslist)
            clslist = [clslist_i.decode() if isinstance(clslist_i, bytes) else clslist_i for clslist_i in clslist]
            zlist = patch_table['Z'][id_in_class]
            zlist = [zlist] if zlist.ndim == 0 else list(zlist)
            zerrlist = patch_table['ZERR'][id_in_class]
            zerrlist = [zerrlist] if zerrlist.ndim == 0 else list(zerrlist)
            zwarnlist = patch_table['ZWARN'][id_in_class]
            zwarnlist = [zwarnlist] if zwarnlist.ndim == 0 else list(zwarnlist)


            # Create a dictionary to store the first occurrence indices in clslist and also a list to keep all results
            index_dict_i = {}
            list_of_index_in_clslist = []

            for element in classlist:
                # Find the first occurrence index in clslist
                index_in_clslist = next((i for i, item in enumerate(clslist) if item.lower()[:ncchar] == element.lower()[:ncchar]), None)

                index_dict_i[element] = index_in_clslist
                list_of_index_in_clslist.append(index_in_clslist)
            # print(index_dict_i)

            # Check if all elements are None or if the list is empty
            if all(x is None for x in list_of_index_in_clslist) or not list_of_index_in_clslist:
                min_index = 0
                flag_i = -1
                # print('Warning: No sutibale index for this specific class type found. Returning the first possible answer (index=0).')
            else:
                min_index = min(list_of_index_in_clslist, key=lambda x: float('inf') if x is None else x)
                flag_i = 0

            index_dict_all[id_i] = {'min_index':min_index, 'index_dict': index_dict_i, 'flag':flag_i, 'CLASS':clslist[min_index], 'Z':zlist[min_index], 'ZERR':zerrlist[min_index], 'ZWARN':zwarnlist[min_index]}

        except Exception as e:
            index_dict_all[id_i] = {'min_index':np.nan, 'index_dict': index_dict_defult, 'flag':-2, 'CLASS':'', 'Z':np.nan, 'ZERR':np.nan, 'ZWARN':np.nan}

    return index_dict_all
###########################################################################################################






def add_extra_columns(input_table, match_table=None):
    """
    inputs:
    input: the original table (in astropy.table form)
    match_table: a dictionary in the format {'path': '<PYAPS_DATA>/m1_table.fits', 'hdu':1 ,'match_keys':['APS_ID', 'CNAME', 'TARGID'] , 'new_keys':['BIN_ID']}
    We use it to add extra columns to the final fits file (e.g. BIN_D for IFU mode)
    """

    assert isinstance(match_table, dict), "match_table must be a dict"
    assert os.path.isfile(
        match_table["path"]
    ), "Adding extra columns to the fits file: No table found in %s" % (
        str(match_table["path"])
    )

    table_extra = Table.read(match_table["path"], format="fits", hdu=match_table["hdu"])

    ## Make sure in both original and extra tables, match_keys columns do not contain missing values (so that the join would be failed)
    ## We find missing values and replace them with np.nan
    if match_table["match_keys"] is not None:
        assert isinstance(match_table["match_keys"], list), "match_keys must be a list"
        for mkeys in match_table["match_keys"]:
            if hasattr(table_extra[mkeys], "mask"):
                table_extra[mkeys] = table_extra[mkeys].filled(np.nan)
            if hasattr(input_table[mkeys], "mask"):
                input_table[mkeys] = input_table[mkeys].filled(np.nan)
    else:
        match_table["match_keys"] = []

    if match_table["new_keys"] is not None:
        assert isinstance(match_table["new_keys"], list), "new_keys must be a list"
    else:
        match_table["new_keys"] = []

    ## create a composite list of match_keys and new_keys to generate a sub-table from the table_extra
    working_columns_match = match_table["new_keys"] + match_table["match_keys"]

    for cln in working_columns_match:
        assert (
            cln in table_extra.colnames
        ), "extra table does not contain columns %s" % (str(cln))

    table_extra = table_extra[working_columns_match]
    input_table = join(
        table_extra,
        input_table,
        join_type="outer",
        keys=match_table["match_keys"],
        metadata_conflicts="warn",
    )

    return input_table


###########################################################################################################


def l1_fileinfo(infiles, wlranges=None, arms_ratio=None, catdir=None, caldir=None):

    """
    extract basic info from a list of input files and also files (and parms) are in the right order (blue first, then red)
    including, 'infiles','obsmode', 'res_mode', 'camera', 'xbin','ybin', 'wlranges', 'mode', 'setups', 'srvys','clss','wlranges_def', 'wlranges', 'resolution', 'obid'
    """

    from pathlib import Path

    ## Define the default CD1_1 (MOS) or CD3_3 (IFU) for different configurations (to be removed later)
    ## after making sure xbin and ybin param are reliable
    cdx = aps_constants.cdx

    ## define default resolution in various configurations. (Later, it will be exported to an external config file)
    def_res = aps_constants.def_res

    columns = [
        "infiles",
        "obsmode",
        "res_mode",
        "camera",
        "xbin",
        "ybin",
        "obid",
        "obsdate",
        "mode",
        "setups",
        "srvys",
        "clss",
        "wlranges_def",
        "wlranges",
        "arms_ratio",
        "resolution",
        "lsf_template",
        "mjd_obs",
        "cat_names",
        "calfiles",
        "lsffiles",
        "run",
        "fpmode",
        "caldate"
    ]
    input_dict = OrderedDict()
    for c in columns:
        input_dict[c] = []

    if wlranges is not None:
        assert len(wlranges) == len(
            infiles
        ), "number of elements in wlrange param and infiles are not consistent"
    if arms_ratio is not None:
        assert isinstance(arms_ratio, list), "arms_ratio must be a list"
        assert len(arms_ratio) == len(
            infiles
        ), "number of elements in arms_ratio param and infiles are not consistent"

    for findex, infile in enumerate(infiles):
        try:
            infile = Path(infile).resolve(strict=True)
        except Exception as e:
            # `raise`, not `sys.exit()`, here and at every other sys.exit
            # in this function -- l1_fileinfo() is called directly by
            # aps_explorer.py (including its own L1 defense-in-depth
            # check), so SystemExit here would kill the whole gunicorn
            # worker in server mode, not just fail the one bad file.
            raise RuntimeError("%s file does not exist" % (infile))

        ## ADD wlrange (based on what specified by user) to the input_dict
        if wlranges is not None:
            assert (
                len(wlranges[findex]) == 2
            ), "Each wlranges elements should contain a lmin and lmax"

            ## here, before going forward and any further steps, we check if any elements of the wlranges list, specified by user
            ## is negative or zero. We use this test to modify the input conditions (infiles, etc)
            ## in case you user wants to ignore one of the bands, for example for the PPXF, where user might
            ## only use blue arm for kinematics measurement at the middle of analysis
            if any(wl_x <= 0.0 for wl_x in wlranges[findex]):
                continue

            input_dict["wlranges"].append(wlranges[findex])

        ## update input_dict with the infiles
        input_dict["infiles"].append(str(infile))

        ## Add arms_ratio into the input_dict
        if arms_ratio is not None:
            input_dict["arms_ratio"].append(arms_ratio[findex])
        else:
            input_dict["arms_ratio"].append(1.0)

        ## update input_dict with the observing mode param for each file

        ## read infile header
        h_primary = getheader(infile, 0)
        h1 = getheader(infile, 1)

        ## CHECK OBSERVING MODE
        ## Note: hkw is an identifier to make header keywords in MOS and IFU mode

        if h_primary["OBSMODE"].replace(" ", "").upper() == "MOS" and h1["NAXIS"] == 2:
            obsmode_i = "MOS"
            input_dict["obsmode"].append(obsmode_i)
            hkw = "1"

        elif (
            h_primary["OBSMODE"].replace(" ", "").upper() == "LIFU" and h1["NAXIS"] == 3
        ):
            obsmode_i = "LIFU"
            input_dict["obsmode"].append(obsmode_i)
            hkw = "3"

        elif (
            h_primary["OBSMODE"].replace(" ", "").upper() == "MIFU" and h1["NAXIS"] == 3
        ):
            obsmode_i = "MIFU"
            input_dict["obsmode"].append(obsmode_i)
            hkw = "3"

        elif (
            h_primary["OBSMODE"].replace(" ", "").upper() == "LIFU" and h1["NAXIS"] == 2
        ):
            obsmode_i = "MOSLIFU"
            input_dict["obsmode"].append(obsmode_i)
            hkw = "1"

        elif (
            h_primary["OBSMODE"].replace(" ", "").upper() == "MIFU" and h1["NAXIS"] == 2
        ):
            obsmode_i = "MOSMIFU"
            input_dict["obsmode"].append(obsmode_i)
            hkw = "1"
        else:
            raise RuntimeError(
                "%s :Unknown OBSERVING MODE [MOS/LIFU/MIFU/MOSLIFU/MOSMIFU]" % (infile)
            )

        ## Here, we temporary set mode into obsmode, just to keep the structure of the input_dict to have
        ## equal number of elements for each key in the input_dict. Later, we will set it to a single value
        input_dict["mode"].append(obsmode_i)

        ## CHECK RESOLUTION MODE
        if "HIGHRES" in h_primary["MODE"].replace(" ", "").upper():
            res_mode_i = "HR"
            input_dict["res_mode"].append(res_mode_i)

        elif "LOWRES" in h_primary["MODE"].replace(" ", "").upper():
            res_mode_i = "LR"
            input_dict["res_mode"].append(res_mode_i)

        else:
            raise RuntimeError("%s :Unknown RESOLUTION MODE [LR/HR]" % (infile))

        ## CHECK CAMERA MODE
        if (
            "BLUE" in h_primary["CAMERA"].replace(" ", "").upper()
            and "2" not in h_primary["VPH"].replace(" ", "").upper()
        ):
            camera_i = "BLUE"
            input_dict["camera"].append(camera_i)

        elif (
            "BLUE" in h_primary["CAMERA"].replace(" ", "").upper()
            and "2" in h_primary["VPH"].replace(" ", "").upper()
        ):
            camera_i = "GREEN"
            input_dict["camera"].append(camera_i)

        elif "RED" in h_primary["CAMERA"].replace(" ", "").upper():
            camera_i = "RED"
            input_dict["camera"].append(camera_i)

        else:
            raise RuntimeError("%s :Unknown CAMERA [BLUE/GREEN/RED]" % (infile))

        ## CHECK BINNING MODE
        try:
            xbin_i = h1["CCDXBIN"]
            ybin_i = h1["CCDYBIN"]
        except Exception as e:
            xbin_i = h_primary["CCDXBIN"]
            ybin_i = h_primary["CCDYBIN"]

        input_dict["xbin"].append(xbin_i)
        input_dict["ybin"].append(ybin_i)

        ## Extra checks for xbin and ybin
        ## cdx_key is the key in the cdx lookup (reference) table, generated based on the obsmode and res_mode
        cdx_key = obsmode_i + res_mode_i + str(xbin_i) + str(ybin_i)
        cdx_key = cdx_key.replace(" ", "")
        if np.abs(h1["CD" + hkw + "_" + hkw] - cdx[cdx_key]) > 0.1 * cdx[cdx_key]:
            raise RuntimeError("%s :CCDXBIN and CD1(3)_1(3) value are not consistent" % (infile))

        ## res_key is the key in the resolution lookup (reference) table, generated based on the obsmode and res_mode
        res_key = (
            obsmode_i + res_mode_i + str(xbin_i) + str(ybin_i) + "_" + str(camera_i)
        )

        assert (
            res_key in def_res.keys()
        ), "This configuration %s is not available in pre-created resolution list" % (
            res_key
        )
        input_dict["resolution"].append(def_res[res_key])

        resbin_key = (
            obsmode_i + res_mode_i + str(xbin_i) + str(ybin_i)
        )

        # also add LSF template key (like MOSLR21)
        input_dict['lsf_template'].append(f'{resbin_key}')


        ## ADD WLRANGES_def to the input_dict
        wlranges_def = [
            h1["CRVAL" + hkw],
            h1["CRVAL" + hkw] + (h1["NAXIS" + hkw] * h1["CD" + hkw + "_" + hkw]),
        ]
        input_dict["wlranges_def"].append(wlranges_def)

        ## In case no wlranges in set by user or it set to None, we fill the wlranges values by default wlranges
        if wlranges is None:
            input_dict["wlranges"].append(wlranges_def)

        ## generate final setups param and add it to the input_dict
        setups = camera_i + res_mode_i[0] + str(xbin_i) + str(ybin_i)
        setups.replace(" ", "")
        input_dict["setups"].append(setups)

        ## Try to read targsrvy from the fibtable if available (MOS only). Otherwise, try to read it from the primary header
        try:
            SRVYS_keyvals = list(
                set(
                    Table.read(infile, format="fits", hdu="FIBTABLE")[
                        "TARGSRVY"
                    ].filled("")
                )
            )
            input_dict["srvys"].append(SRVYS_keyvals)
        except Exception as e:
            ## add surveys info to the input_dict, if available
            SRVYS_keylist = [srvy for srvy in h_primary.keys() if "SRVY" in srvy]
            if len(SRVYS_keylist) > 0:
                SRVYS_keyvals = [h_primary[srvy_i] for srvy_i in SRVYS_keylist]
                input_dict["srvys"].append(SRVYS_keyvals)
            else:
                input_dict["srvys"].append([""])

        # ## Try to read targclass from the fibtable if available (MOS only). Otherwise, fill it with non
        try:
            CLSS_keyvals = list(
                set(
                    Table.read(infile, format="fits", hdu="FIBTABLE")[
                        "TARGCLASS"
                    ].filled("")
                )
            )
            input_dict["clss"].append(CLSS_keyvals)
        except Exception as e:
            input_dict["clss"].append([""])

        ## Add OBID to the input_dict
        input_dict["obid"].append(str(h_primary["OBID"]).strip())
        ## Updated: on 9-Feb-2022 after L2 file verification by CPS
        input_dict["obsdate"].append(h_primary["DATE-OBS"].strip())


        input_dict["mjd_obs"].append(h_primary["MJD-OBS"])
        input_dict["cat_names"].append(h_primary["CAT-NAME"].strip())

        if len(fits.info(infile, output=False)) == 4:
            print(f"WARNING: {infile} has 4 extensions. We are dealing with a SOLAR file!!!")
            caldir = None

        if caldir is not None:
            calfile = os.path.join(caldir, str(h_primary["CALDATE"]).strip(), h_primary["WAVEFILE"].strip())
            lsffile = os.path.join(caldir, str(h_primary["CALDATE"]).strip(), f"lsf_{setups}_{h_primary['FPMODE'].strip()}.fits")
            input_dict["calfiles"].append(calfile)
            input_dict["lsffiles"].append(lsffile)
            input_dict["caldate"].append(str(h_primary["CALDATE"]).strip())
        else:
            input_dict["calfiles"].append(None)
            input_dict["lsffiles"].append(None)
            input_dict["caldate"].append(None)
        input_dict["run"].append(h_primary["RUN"])

        # capture fpmode from the header
        input_dict["fpmode"].append(h_primary["FPMODE"].strip())


    ### in case infiles contains more than one files, we do a check if the first file is the bluer one
    ### it automatically put things in the right order B -> G -> R

    key_sort = sorted(
        range(len(input_dict["camera"])), key=lambda k: input_dict["camera"][k]
    )
    for keys, vals in input_dict.items():
        input_dict[keys] = [vals[i] for i in key_sort]

    ### Extra checks
    assert len(set(input_dict["camera"])) == len(
        input_dict["infiles"]
    ), "Input file(s) have identical CAMERA ID"
    assert (
        len(set(input_dict["obsmode"])) == 1
    ), "Input file(s) should have identical OBSMODE [LIFU/MIFU/MOS/MOSLIFU/MOSMIFU]"
    assert (
        len(set(input_dict["res_mode"])) == 1
    ), "Input file(s) should have identical [LR/HR]"
    assert len(set(input_dict["xbin"])) == 1, "Input file(s) should have identical XBIN"
    assert len(set(input_dict["ybin"])) == 1, "Input file(s) should have identical YBIN"

    if len(set(input_dict["obid"])) != 1:
        print(
            "WANRNING: Input file(s) have different OBID. Make sure you know what are you doing!!!"
        )

    if len(set(input_dict["obsdate"])) != 1:
        print(
            "WARNING: Input file(s) have different DATE-OBS. Make sure you know what are you doing!!!"
        )

    ## add mode to the input_dict based on the obsmode
    input_dict["mode"] = list(set(input_dict["obsmode"]))[0]



    ## Flatten the srvy list, in case more than one file is available in the infiles list
    ## And also remove empty strings from the list (using filter function)
    input_dict["srvys"] = list(
        set([s_in_slist for slist in input_dict["srvys"] for s_in_slist in slist])
    )
    input_dict["srvys"] = list(filter(None, input_dict["srvys"]))

    ## Flatten the clss list, in case more than one file is available in the infiles list
    input_dict["clss"] = list(
        set([s_in_slist for slist in input_dict["clss"] for s_in_slist in slist])
    )
    input_dict["clss"] = list(filter(None, input_dict["clss"]))

    ## close the fits file
    return input_dict


###########################################################################################################
def aperture_sky_region(ra_deg, dec_deg, a_arcsec, b_arcsec, angle_deg):
    """
    Elliptical sky region for an ``area`` / ``mask_areas`` entry or a patch-table row.

    Aperture convention (one definition for every producer and consumer)
    --------------------------------------------------------------------
    ``a_arcsec`` and ``b_arcsec`` are the FULL axis lengths of the ellipse
    (major and minor diameters), the same meaning as ``width`` / ``height`` of
    ``regions.EllipseSkyRegion``. They are NOT semi-axes: an entry of 10 and 6
    arcsec encloses points up to 5 arcsec from the centre along the major axis
    and 3 arcsec along the minor axis. The ``A_world`` / ``B_world`` columns of
    a patch table (degrees) follow the same rule.

    ``angle_deg`` is the position angle in degrees, counter-clockwise.
    """
    from astropy.coordinates import Angle, SkyCoord
    from regions import EllipseSkyRegion

    return EllipseSkyRegion(
        center=SkyCoord(ra_deg, dec_deg, frame="icrs", unit="deg"),
        width=Angle(a_arcsec, "arcsec"),
        height=Angle(b_arcsec, "arcsec"),
        angle=Angle(angle_deg, "deg"),
    )


def gen_targlist(
    infile,
    mode,
    aps_ids=None,
    targsrvy=None,
    targclass=None,
    mask_aps_ids=None,
    area=None,
    mask_areas=None,
    la_out=False,
    aps_id_sum=0,
):

    from functools import reduce

    aps_info = {}
    la = None

    if mode in ["MOS", "MOSLIFU", "MOSMIFU"]:

        ## we use area_wmode to define how to deal with area (ellipse or circle)
        area_wmode = 1

        h_primary = getheader(infile, 0)
        h1 = getheader(infile, 1)
        wcs_h1 = wcs.WCS(h1)

        # check for problematic empty lines in the header 1, as it cause problem in WCS
        if None in h1.keys():
            # `raise`, not `sys.exit()`, here and at every other sys.exit
            # in this function -- gen_targlist() is called directly from
            # APSOB.__init__ (every L1 load reaches it), so SystemExit
            # here would kill the whole gunicorn worker in server mode.
            raise RuntimeError(
                "There is at least one <None> keyword in the h1 header [problematic empty line(s)]"
            )

        # wcs_h1 = wcs.WCS(h1)
        # fl_shape = wcs_h1.pixel_shape

        fibtab = getdata(infile, extname="FIBTABLE")
        APS_ID_ALL = fibtab["FIBREID"] + aps_id_sum

        # Reverse mapping- APS_ID to index in our list
        idf = {y: x for x, y in enumerate(APS_ID_ALL)}

        aps_info["APS_ID"] = ensure_dtype(fibtab["FIBREID"], dtype=np.int32,  return_scalar=True)

        # As still not sure if it is "Nspec" or "NSPEC"
        try:
            aps_info["NSPEC"] = fibtab["NSPEC"]
        except Exception as e:
            aps_info["NSPEC"] = fibtab["Nspec"]

        aps_info["TARGID"] = fibtab["TARGID"]
        aps_info["CNAME"] = fibtab["CNAME"]

        aps_info["TARGSRVY"] = np.char.strip(np.char.upper(fibtab["TARGSRVY"]))

        ### NOTE: Once we got TARGCLASS in the new OPR3, we will replace fibtab["TARGSRVY"] with fibtab["TARGCLASS"]: DONE ON 16 FEB 2022
        ### in the following line. For now, we assume these two are identical

        try:
            aps_info["TARGCLASS"] = np.char.strip(np.char.upper(fibtab["TARGCLASS"]))
        except Exception as e:
            aps_info["TARGCLASS"] = np.char.strip(np.char.upper(fibtab["TARGSRVY"]))

        aps_info["TARGRA"] = fibtab["TARGRA"].astype(np.float64)
        aps_info["TARGDEC"] = fibtab["TARGDEC"].astype(np.float64)
        aps_info["TARGNAME"] = fibtab["TARGNAME"]

        ## Just filter nan values from TARGRA and TARGDEC, as later,  we use this to filter data
        # nan_in_targra=np.isnan(aps_info['TARGRA'])
        # aps_info['TARGRA'][nan_in_targra] = -999.99

        # nan_in_targdec=np.isnan(aps_info['TARGDEC'])
        # aps_info['TARGDEC'][nan_in_targdec] = -999.99

        aps_info["FIB_STATUS"] = np.char.strip(np.char.upper(fibtab["STATUS"]))
        aps_info["TARGUSE"] = np.char.strip(np.char.upper(fibtab["TARGUSE"]))
        aps_info["TARGPROG"] = np.char.strip(np.char.upper(fibtab["TARGPROG"]))

        TARGRA_range = [np.nanmin(aps_info["TARGRA"]), np.nanmax(aps_info["TARGRA"])]
        TARGDEC_range = [np.nanmin(aps_info["TARGDEC"]), np.nanmax(aps_info["TARGDEC"])]

        RR = np.nanmean(TARGRA_range)
        RDEC = np.nanmean(TARGDEC_range)
        aps_info["TARGRA_0"] = np.broadcast_to(
            0.0 if np.isnan(RR) else RR, len(fibtab), 1
        )  # origin of TARGRA
        aps_info["TARGDEC_0"] = np.broadcast_to(
            0.0 if np.isnan(RDEC) else RDEC, len(fibtab), 1
        )  # origin of TARGDEC

        # Generate Wavelength array
        if la_out:
            la = h1["CRVAL1"] + (np.arange(h1["NAXIS1"])) * h1["CD1_1"]


    elif mode in ["IFU", "LIFU", "MIFU"]:

        ## we use area_wmode to define how to deal with area (ellipse or circle)
        area_wmode = 2

        h_primary = getheader(infile, 0)
        h1 = getheader(infile, 1)

        # check for problematic empty lines in the header 1, as it cause problem in WCS
        if None in h1.keys():
            # `raise`, not `sys.exit()`, here and at every other sys.exit
            # in this function -- gen_targlist() is called directly from
            # APSOB.__init__ (every L1 load reaches it), so SystemExit
            # here would kill the whole gunicorn worker in server mode.
            raise RuntimeError(
                "There is at least one <None> keyword in the h1 header [problematic empty line(s)]"
            )

        wcs_h1 = wcs.WCS(h1)
        cube_shape = wcs_h1.pixel_shape  # [xpix, ypix, zpix]
        xaxis = np.arange(cube_shape[0])
        yaxis = np.arange(cube_shape[1])
        zaxis = np.arange(cube_shape[2])

        x, y = np.meshgrid(np.arange(cube_shape[0]), np.arange(cube_shape[1]))
        pix2word = wcs_h1.all_pix2world(x, y, 0, 0)
        aps_info["TARGRA"] = np.reshape(
            pix2word[0], [cube_shape[0] * cube_shape[1]]
        ).astype(np.float64)
        aps_info["TARGDEC"] = np.reshape(
            pix2word[1], [cube_shape[0] * cube_shape[1]]
        ).astype(np.float64)
        aps_info["TARGRA_0"] = np.broadcast_to(
            wcs_h1.wcs.crval[0], cube_shape[1] * cube_shape[2], 1
        )  # origin of TARGRA
        aps_info["TARGDEC_0"] = np.broadcast_to(
            wcs_h1.wcs.crval[1], cube_shape[1] * cube_shape[2], 1
        )  # origin of TARGDEC

        TARGRA_range = [np.nanmin(aps_info["TARGRA"]), np.nanmax(aps_info["TARGRA"])]
        TARGDEC_range = [np.nanmin(aps_info["TARGDEC"]), np.nanmax(aps_info["TARGDEC"])]

        APS_ID_ALL = np.arange(cube_shape[0] * cube_shape[1]) + 1 + aps_id_sum

        ##Reverse mapping- APS_ID to index in our list
        idf = {y: x for x, y in enumerate(APS_ID_ALL)}

        aps_info["APS_ID"] = (APS_ID_ALL[:]).astype(np.int32)
        aps_info["NSPEC"] = (APS_ID_ALL[:]).astype(np.int32)
        aps_info["TARGID"] = np.broadcast_to(" ", cube_shape[1] * cube_shape[2], 1)
        aps_info["TARGSRVY"] = np.broadcast_to(
            h_primary["SRVY1"].upper().replace(" ", ""),
            cube_shape[1] * cube_shape[2],
            1,
        )
        aps_info["TARGCLASS"] = np.broadcast_to(
            h_primary["SRVY1"].upper().replace(" ", ""),
            cube_shape[1] * cube_shape[2],
            1,
        )

        # Temporary fix (31 Oct 2022: some version of L1 files do not contain BUNDLEID)
        if not ("BUNDLEID" in h_primary):
            print('WARNING: NO BUNDLEID found in the Primary header of the L1 file. APS insert value= 1 for this keyword')
            h_primary["BUNDLEID"] = (1, 'Insert by APS')


        aps_info["BUNDLEID"] = np.broadcast_to(
            h_primary["BUNDLEID"],
            cube_shape[1] * cube_shape[2],
            1,
        )

        # For LIFU mode (DM v8.0 Oct 2022), BUNDLEIUD is 0, while the corresponding CCNAME is CCNAME1
        # For mIFU mode, BUNDLEIUD starts from 1, and the corresponding CCNAME keyword would be CCNAME1,...
        if int(h_primary["BUNDLEID"]) == 0:
            BNDID = "1"
        else:
            BNDID = str(int(h_primary["BUNDLEID"])).replace(" ", "")
        CCNAME_X = "CCNAME" + BNDID

        aps_info["CNAME"] = np.broadcast_to(
            h_primary[CCNAME_X], cube_shape[1] * cube_shape[2], 1
        )
        aps_info["FIB_STATUS"] = np.broadcast_to("A", cube_shape[1] * cube_shape[2], 1)
        aps_info["TARGNAME"] = np.broadcast_to(
            "TARGNAME", cube_shape[1] * cube_shape[2], 1
        )

        aps_info["TARGUSE"] = np.broadcast_to("T", cube_shape[1] * cube_shape[2], 1)
        aps_info["TARGPROG"] = np.broadcast_to("", cube_shape[1] * cube_shape[2], 1)

        if la_out:
            # Generate Wavelength array
            la = h1["CRVAL3"] + (np.arange(cube_shape[2])) * h1["CD3_3"]

    else:
        raise RuntimeError("Unknown file mode")

    # GENERTE THE LIST OF FIBRES to be used by the next steps. IF user defines aps_ids param, this will be updated.

    ## Check if an specific list of fibres is requested by user
    if aps_ids is not None:
        w = np.isin(APS_ID_ALL, aps_ids)
        fs_aps_ids = APS_ID_ALL[w]
    else:
        fs_aps_ids = APS_ID_ALL[:]

    ## Check if an specific list of targsrvy is requested by user
    if targsrvy is not None:
        targsrvy = [x.upper().replace(" ", "") for x in targsrvy]

        w = np.isin(aps_info["TARGSRVY"], targsrvy)
        fs_targsrvy = APS_ID_ALL[w]
    else:
        fs_targsrvy = APS_ID_ALL[:]

    ## Check if an specific list of targclass is requested by user
    if targclass is not None:
        targclass = [x.upper().replace(" ", "") for x in targclass]

        w = np.isin(aps_info["TARGCLASS"], targclass)
        fs_targclass = APS_ID_ALL[w]
    else:
        fs_targclass = APS_ID_ALL[:]

    ## Check if an specific list of mask_aps_ids is requested by user (Very useful in IFU mode)
    if mask_aps_ids is not None:
        fs_unmask = np.setdiff1d(APS_ID_ALL, mask_aps_ids)
    else:
        fs_unmask = APS_ID_ALL[:]

    ## Check if an specific list of area is requested by user (Very useful in IFU mode)
    if area is not None:

        assert (
            len(area) == 5
        ), "area needs 5 parameters, RA_CENT[deg], DEC_CENT[deg], A[arcsec], B[arcsec] and ANGLE [CCW in deg]"

        if not TARGRA_range[0] <= area[0] <= TARGRA_range[1]:
            print(
                "WARNING: input RA for the search in not within the available RA range [%s DEG,%s DEG].\
             Ignored and Continued!"
                % (TARGRA_range[0], TARGRA_range[1])
            )
        if not TARGDEC_range[0] <= area[1] <= TARGDEC_range[1]:
            print(
                "WARNING: input DEC for the search in not within the available DEC range [%s DEG,%s DEG].\
             Ignored and Continued!"
                % (TARGDEC_range[0], TARGDEC_range[1])
            )

        with np.errstate(invalid="ignore"):
            # id_in_area = np.ravel(np.where( ((aps_info['TARGRA'] - area[0])**2) + ((aps_info['TARGDEC'] - area[1])**2)  < (area[2]/3600.0)**2.0 ))

            if area_wmode == 2:
                from astropy.coordinates import SkyCoord

                # area[2], area[3] are FULL axis lengths (see aperture_sky_region)
                ellipse_area = aperture_sky_region(
                    area[0], area[1], area[2], area[3], area[4]
                )



                skycoords = SkyCoord(
                    aps_info["TARGRA"], aps_info["TARGDEC"], unit="deg"
                )

                ## Update 31 Oct 2022: dropaxis(2) added to remove the third axis (wave)
                in_elipse_area = ellipse_area.contains(skycoords, wcs_h1.dropaxis(2))
                id_in_area = np.ravel(np.where(in_elipse_area))

            elif area_wmode == 1:
                print(
                    "WARNING: For all MOS modes, this modules only accept circular area"
                )
                circ_rad = (
                    np.max([area[2], area[3]]) / 2.0
                )  # evaluate the circular radius based on the max (A,B)
                print(
                    "Updated AREA parameters: RA_CENT:%.4f  DEC_CENR:%.4f  RADIUS: %.4f"
                    % (area[0], area[1], circ_rad)
                )

                id_in_area = np.ravel(
                    np.where(
                        ((aps_info["TARGRA"] - area[0]) ** 2)
                        + ((aps_info["TARGDEC"] - area[1]) ** 2)
                        < (circ_rad / 3600.0) ** 2.0
                    )
                )

            else:
                raise RuntimeError("Unsupported DATA MODE")

        if len(id_in_area) > 0:
            fs_area = APS_ID_ALL[id_in_area]
        else:
            fs_area = []
    else:
        fs_area = APS_ID_ALL[:]

    ## Check if an specific list of mask_areas is set by user (Very useful in IFU mode)
    if mask_areas is not None:
        mask_areas_idlist = []
        assert set([len(x) for x in mask_areas]) == {
            5
        }, "mask_areas is a list of list(s) each contains RA_CENT[deg], DEC_CENT[deg], A[arcsec], B[arcsec] and ANGLE [CCW in deg.]"

        with np.errstate(invalid="ignore"):
            for each_mask in mask_areas:

                if area_wmode == 2:
                    from astropy.coordinates import SkyCoord

                    # each_mask[2], each_mask[3] are FULL axis lengths
                    ellipse_each_mask = aperture_sky_region(
                        each_mask[0], each_mask[1], each_mask[2], each_mask[3],
                        each_mask[4],
                    )

                    skycoords = SkyCoord(
                        aps_info["TARGRA"], aps_info["TARGDEC"], unit="deg"
                    )

                    ## Update 31 Oct 2022: dropaxis(2) added to remove the third axis (wave)
                    in_ellipse_each_mask = ellipse_each_mask.contains(skycoords, wcs_h1.dropaxis(2))
                    id_in_mask_area = np.ravel(np.where(in_ellipse_each_mask))

                elif area_wmode == 1:
                    print(
                        "WARNING: For all MOS modes, this modules only accept circular mask area"
                    )
                    circ_rad = (
                        np.max([each_mask[2], each_mask[3]]) / 2.0
                    )  # evaluate the circular radius based on the max (A,B)
                    print(
                        "Updated AREA parameters: RA_CENT:%.4f  DEC_CENR:%.4f  RADIUS: %.4f"
                        % (each_mask[0], each_mask[1], circ_rad)
                    )

                    id_in_mask_area = np.ravel(
                        np.where(
                            ((aps_info["TARGRA"] - each_mask[0]) ** 2)
                            + ((aps_info["TARGDEC"] - each_mask[1]) ** 2)
                            < (circ_rad / 3600.0) ** 2.0
                        )
                    )

                else:
                    raise RuntimeError("Unsupported DATA MODE")

                if len(id_in_mask_area) > 0:
                    fs_mask_areas = APS_ID_ALL[id_in_mask_area]
                    mask_areas_idlist.extend(fs_mask_areas)
            fs_areas_unmask = np.setdiff1d(APS_ID_ALL, mask_areas_idlist)
    else:
        fs_areas_unmask = APS_ID_ALL[:]

    ## now we find aps_ids that are survived after all these checks on aps_ids, targsrvy and targclass, area and mask_areas
    fs_out = reduce(
        np.intersect1d,
        (fs_aps_ids, fs_area, fs_areas_unmask, fs_targsrvy, fs_targclass, fs_unmask),
    )

    # Explicitly cast to int32
    fs_out = ensure_dtype(fs_out, dtype=np.int32,  return_scalar=False)


    ### It is important to put fs (list of aps_ids) in increasing order to avoid any problem with redrock
    ### parallelisation method
    return np.sort(fs_out), idf, aps_info, la, wcs_h1


###########################################################################################################
class APSSPEC(object):
    """Simple container class for an individual spectrum.

    Args:
        wave (array): the wavelength grid.
        flux (array): the flux values.
        ivar (array): the inverse variance.


    """

    # @profile
    def __init__(self, wave, flux, ivar, sens):

        self.wave = wave
        self.flux = flux
        self.ivar = ivar
        self.sens = sens

        self._mpshared = False

    def APS_sharedmem_pack(self):
        """Pack spectral data into multiprocessing shared memory."""
        if not self._mpshared:
            # Store data in multiprocessing shared memory
            self.wave = mp_array(self.wave)
            self.flux = mp_array(self.flux)
            self.ivar = mp_array(self.ivar)
            self.sens = mp_array(self.sens)

            self._mpshared = True
        return

    def APS_sharedmem_unpack(self):
        """Unpack spectral data from multiprocessing shared memory."""
        if self._mpshared:
            self.wave = np.array(self.wave)
            self.flux = np.array(self.flux)
            self.ivar = np.array(self.ivar)
            self.sens = np.array(self.sens)

            self._mpshared = False

        return


###########################################################################
class APSTARG(object):
    """A single target.

    This represents the data for a single target, including a unique identifier
    and the individual spectra observed for this object (or a co-add).

    Args:
        targetid (int or str): unique targetid
        spectra (list): list of Spectrum objects

    """

    def __init__(
        self,
        aps_id,
        cname,
        targid,
        targra,
        targdec,
        targsrvy,
        targclass,
        fib_status,
        targuse,
        targprog,
        spectra,
        meta=None,
        split_arms=False,
        mask_bad_overlap=False,
    ):
        self.id = aps_id
        self.aps_id = aps_id
        self.cname = cname
        self.targid = targid
        self.targra = targra
        self.targdec = targdec
        self.targsrvy = targsrvy
        self.targclass = targclass
        self.fib_status = fib_status
        self.targuse = targuse
        self.targprog = targprog
        self.spectra = spectra
        self.split_arms = split_arms
        self.mask_bad_overlap = mask_bad_overlap
        if meta is None:
            self.meta = list()
        else:
            self.meta = meta

    def APS_sharedmem_pack(self):
        """Pack all spectra into multiprocessing shared memory."""
        for s in self.spectra:
            s.APS_sharedmem_pack()
        return

    def APS_sharedmem_unpack(self):
        """Unpack all spectra from multiprocessing shared memory."""
        for s in self.spectra:
            s.APS_sharedmem_unpack()
        return

    ### @profile
    def join_arms(self, fwhm_interp_dict_join):
        ### IMPORTANT NOTE. It works on arbitrary number of arms/bands
        ### Here, B refers to the left arms and R refer to the right arm

        tmeta_join = dict()
        meta_B = self.meta[0]
        # Captured before meta_B's own 'mode' entry gets folded into
        # tmeta_join's per-arm list below -- see the spaxel-weighted-LSF
        # lookup further down, which needs the plain scalar mode string.
        mode_B = meta_B.get('mode')

        for metaB_k in meta_B.keys():
            tmeta_join[metaB_k] = [meta_B[metaB_k]]

        spectra_list = []
        for arms_i in range(1, len(self.spectra)):

            wave_B = self.spectra[0].wave
            wave_R = self.spectra[arms_i].wave
            flux_B = self.spectra[0].flux
            flux_R = self.spectra[arms_i].flux
            ivar_B = self.spectra[0].ivar
            ivar_R = self.spectra[arms_i].ivar
            meta_R = self.meta[arms_i]

            for tmetaj_k in tmeta_join.keys():
                ## it just keep the unique values. Otherwise keep both
                tmeta_join[tmetaj_k].append(meta_R[tmetaj_k])

            ## create a list of wavelength for two arms (overlapped or non-overlapped)
            wavelist = [wave_B, wave_R]
            mins_w = np.array([min(w) for w in wavelist])
            maxs_w = np.array([max(w) for w in wavelist])

            ## few checks on wavelength, before creating the new wavelength base
            if np.any(np.argsort(mins_w) != np.arange(len(wavelist))):
                raise ValueError(
                    "List of wavelengths must be sorted in increasing order."
                )
            if np.any(mins_w[2:] < maxs_w[:-2]):
                raise ValueError("No arms can be completely overlapped.")

            ## Generate the new wavelength base, as well as wavelength arrays for the single arms
            ## In case no overlap found (e.g. HR data), WAVE_O is the wavelength base for the gap between two arms
            wave_full, waveBB, wave_O, waveRR = wave_interpol(wavelist)

            ## In case two arms have overlap, check wavelength step in the overlapped region and the non_overlapped (original) spectrum
            if np.all(maxs_w[:-1] > mins_w[1:]):

                delta_wO = wave_O[1] - wave_O[0]
                delta_wBB = waveBB[1] - waveBB[0]
                delta_wRR = waveRR[1] - waveRR[0]
                assert (
                    abs(delta_wO - delta_wBB) < 0.001
                ), "delta_w for the overlapped part and BLUE part is not consistent"
                assert (
                    abs(delta_wO - delta_wRR) < 0.001
                ), "delta_w for the overlapped part and RED part is not consistent"

            # delete outdated variable
            del wavelist

            ## Fill the flux  and ivar  for the non-overlapped part of the spectra in each arm (BLUE only or RED only)
            len_BB = len(waveBB)
            fluxBB = flux_B[0:len_BB]
            ivarBB = ivar_B[0:len_BB]

            len_RR = len(waveRR)
            fluxRR = flux_R[-1 * len_RR :]
            ivarRR = ivar_R[-1 * len_RR :]

            ## Fill the flux and ivar  for the overlap part, but still on the original wavelength base
            waveBO = wave_B[len_BB:]
            fluxBO = flux_B[len_BB:]
            ivarBO = ivar_B[len_BB:]

            len_RO = len(wave_R) - len(waveRR)
            waveRO = wave_R[:len_RO]
            fluxRO = flux_R[:len_RO]
            ivarRO = ivar_R[:len_RO]

            # ## IF all fluxes are zero (e.g. for parked fibres) then, all results will be replaced by 0.0
            # if np.all(fluxBB == 0) and np.all(fluxRR == 0):
            #     wave_BOR = np.concatenate((waveBB, wave_O, waveRR), axis=0)
            #     flux_BOR = wave_BOR * 0.0
            #     ivar_BOR = wave_BOR * 0.0

            #     sens_BOR = None
            #     self.spectra[0] = APSSPEC(wave_BOR, flux_BOR, ivar_BOR, sens_BOR)
            #     self.meta[0] = tmeta_join
            #     continue

            # define a flag to check if two arms overlap is good or bad
            bad_overlap = False
            ## Check if two arms have overlap or not (YES CASE)
            if np.all(maxs_w[:-1] > mins_w[1:]):

                # Use only that overlapped slice of original spectra in  BLUE or RED arm  as reference
                flux_BO = np.interp(wave_O, waveBO, fluxBO)
                ivar_BO = np.interp(wave_O, waveBO, ivarBO)

                flux_RO = np.interp(wave_O, waveRO, fluxRO)
                ivar_RO = np.interp(wave_O, waveRO, ivarRO)

                ## To use np.average, we need to generate a 2d array of all fluxes and IVAR
                flux_BR_O = np.array([flux_BO, flux_RO])
                ivar_BR_O = np.array([ivar_BO, ivar_RO])

                # update on 9th of December 2024
                # this is to address the issues with jump in the fluxes in the overlapped region
                # Check for significant difference in median values
                med_flux_BO= np.nanmedian(flux_BO)
                med_flux_RO= np.nanmedian(flux_RO)
                relative_diff = abs(med_flux_BO - med_flux_RO) / max(med_flux_BO, med_flux_RO, 1e-10)

                if relative_diff > aps_constants.bad_overlap_epsilon:
                    # Significant difference detected
                    flux_O = wave_O * 0.0
                    ivar_O = wave_O * 0.0
                    bad_overlap = True
                    # in order to avoid zero weights, we replace 0 with a very small non-zero value
                else:
                    # in order to avoid zero weights, we replace 0 with a very small non-zero value
                    ivar_BR_O[ivar_BR_O ==0] = 1.0 / (aps_constants.large_error**2)
                    flux_O = np.average(flux_BR_O, axis=0, weights=ivar_BR_O)
                    ivar_O = np.sum(ivar_BR_O, axis=0)
                    bad_overlap = False

            ## Check if two arms have overlap or not (NO CASE)
            if not np.all(maxs_w[:-1] > mins_w[1:]):

                # we introduce a new flag (no_overlap) which will be use later
                no_overlap = True
                # print('No overlapped region found between two arms. Fill the GAP by 0.0')
                flux_O = wave_O * 0.0
                ivar_O = wave_O * 0.0
            else:
                no_overlap = False

            ## Now, merge BB (Blue only) , O (overlap or Gap) and RR (Red only) parts of the data together
            wave_BOR = np.concatenate((waveBB, wave_O, waveRR), axis=0)
            flux_BOR = np.concatenate((fluxBB, flux_O, fluxRR), axis=0)
            ivar_BOR = np.concatenate((ivarBB, ivar_O, ivarRR), axis=0)

            # in case bad_overlap is True and mask_bad_overlap is True, we set ivar_BOR to 0 where wave_BOR is the affecting region by bad overalap and cross-talk
            if bad_overlap and self.mask_bad_overlap:
                # Set ivar_BOR to 0 where wave_BOR is the affecting region by bad overalap and cross-talk (Update 11 Dec 2024)
                ivar_BOR[(wave_BOR >= aps_constants.bad_overlap_mask[0]) & (wave_BOR <= aps_constants.bad_overlap_mask[1])] = 0.0

            ## if user request to split arms, we first check if there is any overlap between each two arms.
            ## then, if overlap presents, we use the mean of overlap-wave as the cutting point
            ## if no overlap found, we stick to the original spectrum

            if self.split_arms:
                if no_overlap:
                    wave_B = waveBB
                    wave_R = waveRR
                    flux_B = fluxBB
                    flux_R = fluxRR
                    ivar_B = ivarBB
                    ivar_R = ivarRR

                else:
                    B_O = np.ravel(np.where(wave_BOR < np.nanmean(wave_O)))
                    R_O = np.ravel(np.where(wave_BOR >= np.nanmean(wave_O)))

                    wave_B = wave_BOR[B_O]
                    wave_R = wave_BOR[R_O]
                    flux_B = flux_BOR[B_O]
                    flux_R = flux_BOR[R_O]
                    ivar_B = ivar_BOR[B_O]
                    ivar_R = ivar_BOR[R_O]

                B_spectra = APSSPEC(wave_B, flux_B, ivar_B, None)
                spectra_list.append(B_spectra)
                ## in case of split_arms, we keep the redder band as spectra[0] as it might be used as reference for the next redder arm
                ## we also keep the bluer arm in the spectra_list, so later, once the loop over arms done, we can use it.
                self.spectra[0] = APSSPEC(wave_R, flux_R, ivar_R, None)
                ## Please note, in case of split_arms, we do not modify the meta info.

            else:
                ## in case no split_arms is requested, we treat the joint arms as the spectra[0] and continue the loop (in case any redder arm is available to join)
                sens_BOR = None
                self.spectra[0] = APSSPEC(wave_BOR, flux_BOR, ivar_BOR, sens_BOR)
                self.meta[0] = tmeta_join

                # in case no split arms is requested but join arm is true, we add fwhm of joint arms
                # for IFU mode as the NSPEC coming from the calib file and NSPEC genearted by APS are not consistent,
                # we assigned an identical fwhm to all IFU spaxels
                # Later we can do this based on ahybrid coordinate base approach
                #
                # "Later" is now: when APSOB.__init__'s join_arms block built
                # a 'spaxel_weighted' sub-dict on fwhm_interp_dict_join (see
                # its own comment there -- only happens when
                # self._spaxel_weighted_lsf is on), use it here, namespaced
                # by APS_ID exactly like _assign_arm_results_to_targets's own
                # cube branch -- never via the bare int(NSPEC) lookup the
                # MOS branch below uses, since a cube's own fake per-spaxel
                # NSPEC (NSPEC=APS_ID) can numerically collide with a real
                # physical fibre's NSPEC (1-960) in this very same
                # fwhm_interp_dict_join.
                if fwhm_interp_dict_join is not None:
                    if mode_B in ("IFU", "LIFU", "MIFU"):
                        spaxel_weighted = fwhm_interp_dict_join.get('spaxel_weighted')
                        aps_id_f = int(self.meta[0]['APS_ID'][0])
                        if spaxel_weighted is not None and aps_id_f in spaxel_weighted:
                            self.meta[0]['fwhm'] = spaxel_weighted[aps_id_f]
                        else:
                            self.meta[0]['fwhm'] = fwhm_interp_dict_join['global']
                        self.meta[0]['gfwhm'] = fwhm_interp_dict_join['global']
                    else:
                        try:
                            self.meta[0]['fwhm'] = fwhm_interp_dict_join[int(self.meta[0]['NSPEC'][0])]
                            self.meta[0]['gfwhm'] = fwhm_interp_dict_join['global']
                        except Exception:
                            self.meta[0]['fwhm'] = fwhm_interp_dict_join['global']
                            self.meta[0]['gfwhm'] = fwhm_interp_dict_join['global']
                else:
                    self.meta[0]['fwhm'] = None
                    self.meta[0]['gfwhm'] = None

        if self.split_arms:
            spectra_list.append(self.spectra[0])
            self.spectra = spectra_list

            ## also update meta for each arm (that could be modified version of original arms or themself)
            for arms_i in range(len(self.spectra)):
                self.meta[arms_i]["STITCH"] = -1
                self.meta[arms_i]["STITCH_datatype"] = "int16"
                self.meta[arms_i]["SPLIT"] = 1
                self.meta[arms_i]["SPLIT_datatype"] = "int16"

        else:
            ## Inside the loop above, only spectra[0] is updating and changed to the stitched arm spectra
            self.spectra = [
                APSSPEC(
                    self.spectra[0].wave,
                    self.spectra[0].flux,
                    self.spectra[0].ivar,
                    None,
                )
            ]

            ## Add a new key to the stitched-arm meta, indicating the stitching status
            self.meta[0]["STITCH"] = 1
            self.meta[0]["STITCH_datatype"] = "int16"
            self.meta[0]["SPLIT"] = 0
            self.meta[0]["SPLIT_datatype"] = "int16"
            self.meta = [self.meta[0]]

        return

###########################################################################
# VECTORIZED HELPER FUNCTIONS FOR PERFORMANCE
###########################################################################

def vectorized_nan_handling(fl, iv):
    """Handle NaN and invalid values for all spaxels at once."""
    iv[np.isnan(fl)] = 0.0
    iv[np.isnan(iv)] = 0.0
    iv[iv < 0] = 0.0
    return iv


def vectorized_gap_masking(iv, island_offset):
    """Apply gap masking to all spaxels simultaneously."""
    gap_mask = (iv == 0.0)

    if island_offset > 0:
        structure = np.ones((1, 2*island_offset + 1))
        gap_mask_dilated = ndimage.binary_dilation(gap_mask, structure=structure)
    else:
        gap_mask_dilated = gap_mask

    iv[gap_mask_dilated] = 0.0
    gaps_masked = np.any(gap_mask_dilated, axis=1)

    return iv, gaps_masked


_DUST_SFDMAP_CACHE = {}


def get_sfd_ebv(ra, dec, mapdir=None):
    """E(B-V) from the SFD (Schlegel, Finkbeiner & Davis 1998) dust map,
    at one or many (RA, Dec) positions in degrees, ICRS.

    Uses `desiutil.dust.SFDMap` (already a PyAPS/redrock dependency --
    it's the exact same SFD98 map DESI itself uses, so no new package is
    introduced). `desiutil` only ships small test-fixture map files, so
    the real, full-resolution `SFD_dust_4096_ngp.fits`/`_sgp.fits` must
    exist under `mapdir` (default: `<PyAPS repo root>/PyAPS_data/DUST`,
    override with the `DUST_DIR` environment variable or the `mapdir`
    argument) -- see doc/aps_utils.md for where to get them.

    Parameters
    ----------
    ra, dec : float or array -- degrees, ICRS
    mapdir  : str, optional -- directory containing the two SFD FITS maps

    Returns
    -------
    ebv : float or ndarray, same shape as ra/dec
    """
    from desiutil.dust import SFDMap
    from astropy.coordinates import SkyCoord

    if mapdir is None:
        mapdir = os.environ.get(
            "DUST_DIR",
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))), "PyAPS_data", "DUST"))

    key = os.path.abspath(mapdir)
    if key not in _DUST_SFDMAP_CACHE:
        _DUST_SFDMAP_CACHE[key] = SFDMap(mapdir=mapdir)
    sfd = _DUST_SFDMAP_CACHE[key]

    coord = SkyCoord(np.atleast_1d(ra), np.atleast_1d(dec),
                      unit="deg", frame="icrs")
    ebv = sfd.ebv(coord)
    return float(ebv[0]) if np.isscalar(ra) or np.ndim(ra) == 0 else ebv


def dust_transmission_curve(wave, ebv, r_v=3.1, ebv_scale=1.0):
    """Fraction of flux transmitted through Galactic dust at each
    wavelength, i.e. `observed = true * transmission`. Wraps
    `desiutil.dust.dust_transmission` (Fitzpatrick 1999 extinction law).

    `ebv_scale` multiplies `ebv` before use -- for applying only a
    *fraction* of the full line-of-sight SFD value. This matters because
    SFD gives the *total* Galactic column to effectively infinite
    distance, which is the right assumption for extragalactic sources
    (GALAXY/QSO) but can be a substantial OVER-correction for a
    foreground Milky Way star that sits at a finite distance in front of
    only part of that dust column -- confirmed empirically on a real,
    heavily-reddened GA-LRDISC target (E(B-V)_SFD=1.47, WEAVE OB
    20260122/stack_3134457__stack_3134456, APS_ID 581): dereddening by
    the *full* SFD value made the STAR-vs-GALAXY REDROCK chi^2 gap
    WORSE (ratio 1.44x -> 4.35x, i.e. more wrongly-GALAXY, not less),
    while a partial correction (~25% of the full value) was the best of
    several fractions tested (ratio down to 1.27x) but still did not
    flip the classification on its own for that target -- see
    the APS development notes for the investigation this came out of. A proper fix for stars needs
    a distance-resolved (3D, e.g. Green et al. Bayestar) map rather than
    a fixed fraction of the 2D SFD value; `ebv_scale` is a stopgap, not
    a substitute for that.

    Parameters
    ----------
    wave      : array -- vacuum wavelength [Angstrom]
    ebv       : float -- E(B-V), typically from get_sfd_ebv()
    r_v       : float -- total-to-selective extinction ratio (default 3.1)
    ebv_scale : float -- fraction of `ebv` to actually apply (default 1.0
                = full SFD value; use <1 for foreground Galactic stars)

    Returns
    -------
    transmission : array, same shape as wave, in (0, 1]
    """
    from desiutil.dust import dust_transmission
    return dust_transmission(np.asarray(wave, dtype=float),
                              ebv * ebv_scale, Rv=r_v)


def deredden_spectrum(wave, fl, iv, ra, dec, r_v=3.1, ebv_scale=1.0,
                       mapdir=None, ebv=None):
    """Correct `fl`/`iv` for Galactic foreground dust extinction (SFD98 +
    Fitzpatrick99), given a target's sky position.

    Intended for classification/REDROCK use only (see `ebv_scale`
    caveat in `dust_transmission_curve` -- this is NOT validated as a
    general-purpose flux calibration correction, and should not be
    applied to science products like FERRE/RVS stellar parameters or
    ExGal analysis without separately checking it helps there too).

    `fl_true = fl_observed / transmission`; `iv` is scaled by
    `transmission**2` to preserve the correct (larger) uncertainty on
    the corrected flux.

    Parameters
    ----------
    wave, fl, iv : arrays -- same shape, observed (not yet dereddened)
    ra, dec      : float -- target position, degrees ICRS
    r_v          : float -- default 3.1
    ebv_scale    : float -- fraction of the full SFD E(B-V) to apply;
                   default 1.0. See `dust_transmission_curve` docstring
                   -- 1.0 is appropriate for GALAXY/QSO, NOT validated as
                   an improvement for STAR (empirically made a real test
                   case worse) unless scaled down.
    mapdir       : str, optional -- passed to `get_sfd_ebv`
    ebv          : float, optional -- skip the SFD lookup and use this
                   E(B-V) directly (e.g. a pre-computed/cached value)

    Returns
    -------
    fl_dered, iv_dered, ebv_used : arrays + float
    """
    if ebv is None:
        ebv = get_sfd_ebv(ra, dec, mapdir=mapdir)
    transmission = dust_transmission_curve(wave, ebv, r_v=r_v, ebv_scale=ebv_scale)
    fl_dered = np.asarray(fl, dtype=float) / transmission
    iv_dered = np.asarray(iv, dtype=float) * transmission**2
    return fl_dered, iv_dered, ebv


def vectorized_sensitivity_correction(fl, iv, sens, sens_corr=True):
    """Apply sensitivity correction to all spaxels at once.

    In-place elementwise ops rather than `fl * sens` / `iv / (sens ** 2)`
    -- those each materialize a new full-size (n_spaxels x n_wave) float
    array (on a real IFU arm, ~1GB apiece), and profiling a real
    32,578-spaxel LIFU load showed this single call costing 1.7-3.5s of
    self time, the single largest chunk of per-arm processing, dominated
    by exactly this kind of temporary-array churn (this module also
    forces OMP/OPENBLAS/MKL/NUMEXPR to 1 thread at import to avoid BLAS
    oversubscription across concurrent explorer sessions, so there is no
    thread-level speedup available to hide the cost either). `sens`
    itself must NOT be mutated -- callers keep and later store the
    *original* sensitivity curve (e.g. for the sensitivity plot tab,
    `self._targetlist[...].spectra[infc].sens`), so `iv / (sens ** 2)`
    is done as two successive in-place divisions by `sens` instead of
    squaring it into a new buffer -- algebraically identical (division
    is associative here) and numerically identical modulo the usual
    last-bit float rounding difference of doing two ops instead of one.
    """
    if not sens_corr:
        return fl, iv, False

    iv[np.isnan(sens)] = 0.0
    fl *= sens
    iv /= sens
    iv /= sens

    return fl, iv, True


def vectorized_arms_ratio_correction(fl, iv, arms_ratio):
    """Apply arms ratio correction to all spaxels at once."""
    if arms_ratio == 1.0:
        return fl, iv, False

    arm_ratio = float(arms_ratio)
    fl = fl * arm_ratio
    iv = iv / (arm_ratio ** 2)

    return fl, iv, True


def vectorized_snr_calculation(fl, iv, ivar_mask_value):
    """Calculate SNR for all spaxels simultaneously.

    Used to build `snr_values = fl * sqrt(iv)`, NaN out the masked entries,
    then take `np.nanmean(snr_values[valid_spaxels], axis=1)` -- profiled
    directly against a real 22,386-spaxel arm (the post-DOF-filter subset
    passed in here on a real LIFU stackcube): that fancy-indexed row copy
    plus numpy's own generic `nanmean` (which internally re-detects the
    NaNs it's skipping and makes its *own* full-size copy in
    `_replace_nan`) together cost ~2.4-2.6s, the single largest chunk of
    per-arm processing time.

    Replaced with an explicit sum/count: `nomask` (already computed for
    `nz_array` below) tells us exactly which entries are real, so
    `np.where(nomask, ..., 0.0)` zeroes the rest in one pass (safe even if
    `fl` still holds NaNs at masked-out positions -- an in-place
    `vals *= nomask` would not be: NaN * 0 is NaN, not 0, in IEEE754), then
    a plain `.sum(axis=1) / nz_array` reproduces `nanmean` without its
    internal NaN re-scan or the row-selecting copy (dividing only the
    per-spaxel scalars by `nz_array`, not the full 2D array, so
    `valid_spaxels` only ever indexes small 1D arrays here). Verified
    bit-for-bit equivalent modulo ~1e-6 float32 rounding (sum/divide vs.
    nanmean's own internal reduction order) on that same real arm, ~1.9x
    faster (2.4s -> 1.25s isolated; the `_replace_nan` copy this also
    removes was the same `numpy.array` cost showing up separately in the
    per-arm profile).
    """
    nomask = iv > 10 * ivar_mask_value
    nz_array = np.sum(nomask, axis=1)

    snr_array = np.full(fl.shape[0], np.nan)
    valid_spaxels = nz_array > 0
    if np.any(valid_spaxels):
        vals = np.where(nomask, fl * np.sqrt(iv), 0.0)
        sums = vals.sum(axis=1)
        snr_array[valid_spaxels] = sums[valid_spaxels] / nz_array[valid_spaxels]

    return snr_array, nz_array


def vectorized_cosmic_ray_removal(fl, iv, cr_med_length, cr_cut_level, cr_wing_length):
    """Apply cosmic ray removal to all spaxels."""
    flux_sm = ndimage.median_filter(
        fl.astype(np.float64),
        size=(1, cr_med_length)
    )

    sigma_sm = 1.0 / np.sqrt(np.nanmedian(iv, axis=1, keepdims=True))
    cosmic_mask = np.abs(fl - flux_sm) / sigma_sm > cr_cut_level

    if cr_wing_length > 0:
        structure = np.ones((1, 2*cr_wing_length + 1))
        cosmic_mask = ndimage.binary_dilation(cosmic_mask, structure=structure)

    iv[cosmic_mask] = 0.0
    cr_cleaned = np.any(cosmic_mask, axis=1)

    return iv, cr_cleaned


def vectorized_wavelength_mask(la, band_ranges):
    """Create a boolean mask for wavelength ranges."""
    mask = np.zeros(len(la), dtype=bool)

    for band in band_ranges:
        assert len(band) == 2, "Each band range needs [wl_min, wl_max]"
        band_mask = (la >= band[0]) & (la <= band[1])
        mask = mask | band_mask

    return mask



###########################################################################

def makeR_from_fwhm_array(wavelist, fwhm_array, cache_Rcsr,
                         use_deconvolution=True, sigma0_angstrom=0.5):
    """
    SHARED: Create resolution matrix from FWHM array.
    Used by both Redrock (via APSOB) and RVSpecfit (via aps_rvs).

    Parameters
    ----------
    wavelist : array
        Wavelength array in Angstroms
    fwhm_array : array
        FWHM values at each wavelength (already evaluated!)
    cache_Rcsr : bool
        Cache CSR sparse format
    use_deconvolution : bool
        Apply template resolution deconvolution
    sigma0_angstrom : float
        Template instrumental resolution for deconvolution

    Returns
    -------
    R : scipy.sparse.dia_matrix
        Resolution matrix
    Rcsr : scipy.sparse.csr_matrix or None
        CSR format (if cache_Rcsr=True)
    """
    from scipy import sparse

    la = wavelist
    FWHM_ang = np.asarray(fwhm_array)

    # Calculate pixel size
    pix_size = np.median(np.diff(la))

    # ========================================================================
    # SAFETY CHECK: Verify FWHM is reasonable for deconvolution
    # ========================================================================
    if use_deconvolution and sigma0_angstrom > 0:
        # Check FWHM distribution
        fwhm_min = np.nanmin(FWHM_ang)
        fwhm_median = np.nanmedian(FWHM_ang)
        fwhm_5th = np.nanpercentile(FWHM_ang, 5)

        # Safety threshold: FWHM should be at least 1.2× sigma_0
        min_safe_fwhm = 1.2 * sigma0_angstrom * 2.355

        # Count problematic pixels
        n_too_small = np.sum(FWHM_ang < min_safe_fwhm)

        if n_too_small > 0:
            pct_bad = 100 * n_too_small / len(FWHM_ang)

            # Issue warning but don't spam if just a few pixels
            if pct_bad > 5:  # Only warn if > 5% affected
                print(f"  WARNING: {n_too_small}/{len(FWHM_ang)} pixels ({pct_bad:.1f}%) have FWHM < {min_safe_fwhm:.2f}Å")
                print(f"           FWHM: min={fwhm_min:.3f}Å, 5%={fwhm_5th:.3f}Å, median={fwhm_median:.3f}Å")
                print(f"           sigma_0={sigma0_angstrom:.3f}Å")

            # Decision based on severity
            if pct_bad > 50:
                # Too many bad pixels → disable deconvolution
                print(f"  → Disabling deconvolution (>50% problematic pixels)")
                use_deconvolution = False

            elif pct_bad > 10:
                # Moderate issues → clamp and warn
                print(f"  → Clamping {n_too_small} pixels to minimum safe FWHM")
                FWHM_ang = np.where(FWHM_ang < min_safe_fwhm, min_safe_fwhm, FWHM_ang)

            else:
                # Few bad pixels → clamp silently
                FWHM_ang = np.where(FWHM_ang < min_safe_fwhm, min_safe_fwhm, FWHM_ang)

    # ========================================================================
    # Convert FWHM to pixel widths
    # ========================================================================
    wd = FWHM_ang / (2.355 * pix_size)

    # Handle very small widths
    ww = wd < 1e-5
    wd[ww] = 2.0

    # Matrix construction
    ii = np.arange(len(wd))
    di = ii - ii[:, None]
    di2 = di**2
    ndiag = int(4 * np.ceil(wd.max()) + 1)
    nbins = len(wd)

    # Build resolution matrix
    reso = np.zeros([ndiag, nbins])

    for idiag in range(ndiag):
        offset = ndiag // 2 - idiag
        d = np.diagonal(di2, offset=offset)
        if offset < 0:
            reso[idiag, :len(d)] = np.exp(-d / 2 / wd[:len(d)]**2)
        else:
            reso[idiag, nbins-len(d):nbins] = np.exp(-d / 2 / wd[nbins-len(d):nbins]**2)

    # ========================================================================
    # Apply deconvolution if requested (with safety checks)
    # ========================================================================
    if use_deconvolution:
        print(f"  Applying resolution deconvolution (sigma_0={sigma0_angstrom:.1f}Å, pix={pix_size:.3f}Å)")

        try:
            # Suppress scipy warnings for this operation
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=Warning)

                reso_deconv = deconvolve_resolution_matrix(
                    reso,
                    sigma0_angstrom=sigma0_angstrom,
                    pix_size_angstrom=pix_size
                )

            # Check if deconvolution produced valid results
            if np.any(~np.isfinite(reso_deconv)):
                print(f"  WARNING: Deconvolution produced NaN/Inf values, using original matrix")
                reso_deconv = reso

            # Keep original values at edges (deconvolution corrupts them)
            edge_pixels = 5
            reso_deconv[:, :edge_pixels] = reso[:, :edge_pixels]
            reso_deconv[:, -edge_pixels:] = reso[:, -edge_pixels:]

            reso = reso_deconv

        except Exception as e:
            print(f"  WARNING: Deconvolution failed: {e}")
            print(f"           Using original resolution matrix without deconvolution")
            # Keep original reso matrix

    # ========================================================================
    # Normalize
    # ========================================================================
    reso /= np.sum(reso, axis=0) + 1e-10

    # Create sparse matrix
    offsets = ndiag // 2 - np.arange(ndiag)
    nwave = reso.shape[1]
    R = sparse.dia_matrix((reso, offsets), (nwave, nwave))

    if cache_Rcsr:
        Rcsr = R.tocsr()
    else:
        Rcsr = None

    return R, Rcsr

###########################################################################
class APSOB:
    """
    This is a class designed by APS to deal with all types of WEAVE DATA at L1 (CASU) level

    INPUTS:
    1- infiles [Required], List of input fits file with full path (e.g. ['/scrach/OPR3b/stacked_1002058.fit', '/scrach/OPR3b/stacked_1002057.fit'])
    This parameter accept maximum two inputs. Not need to say, two files is required if you want to set join_arms=True
    The order of files is not important, but it should be consistent with the order in the wlranges param. Otherwise it return an error.

    2- aps_ids [Optional], list of aps_ids to proceed with. (e.g. [1007,1006,1005])
    If not set (or set to None), the code with use all available aps_ids or spaxel.
    In case of IFU data, the code first generate the fake aps_id for all spaxels. And, user can limit the process to a subset of this spaxels using
    this parameters.
    The code check if any elements in the aps_id list are valid/available. Otherwise simply ignore invalid entries.

    3- targsrvy [Optional]. List of surveys that user is keen to proceed with (e.g. ['WA', 'GA'])
    If not set (or set to None), the code with use use all available options for targsrvy.

    4- targclass [Optional]. List of targclasses (recently [since Sept. 2019] introduced by CPS) to proceed with (e.g. ['STAR', 'BA-STRA'])
    ** At the moment, before getting OPR3b+ data (March 2020), APS is set this identical to targsrvy.
    If not set (or set to None), the code with use use all available options for targclass.

    5- mask_aps_ids [Optional]. List of aps_ids to be masked (e.g. [1004,1003])
    APS simply ignore this aps_ids in all process. As aps_id param, this code ignore invalid entries in this list.

    6- area [Optional] A list contains RA_CENT[deg], DEC_CENT[deg], A[arcsec], B[arcsec] and angle [CCW in deg.], in order to extract data for data points within an ellipse (for IFU) or circle aperture (MOS)

    7 - mask_areas [Optional] A list of lists each contains RA_CENT[deg], DEC_CENT[deg], A[arcsec], B[arcsec] and angle [CCW in deg.], in order to MASK all data within an ellipse_shaped region (for IFU) or circle aperture (MOS)

    8- wlranges [Optional]. List of list of wavelength ranges to be considered in analysis in the [lmin, lmax] format for each infiles element.
    Assuming we have two files in the infiles list, then we need wlranges array to be a list of two lists, each correspond to one of the input files.

    Note: Order of wlranges Must be the same as order of files in the infiles. Otherwise the code will return an error message.
    EXAMLE:
    Assuming we have infiles =['/scrach/OPR3b/stacked_1002058.fit', '/scrach/OPR3b/stacked_1002057.fit']
    Then, the wlranges would be something like wlranges=[[4000.0, 5000.0],[7000.0,9000.0]].
    Note: If wlranges param for each infiles' entry does not overlap with the available wavelength range of that specific file,
    then the code simply discard that fits file from analysis.
    Note: If wlranges param for each infiles' entry does not fit within the maximum wlranges boundary of an specific file, then
    then code use the maximum possible overlapped part.

    9- sens_corr [Optional, default value: True]. If set, the code apply the sensitivity function to both flux and ivar
    Then it also multiply data by a large value (set in the aps_constant.py file) to avoid dealing with small numbers.
    All these values are available in the meta of the APSOB target.

    10- mask_gaps [Optional, default value: True]. If set, the code find the gaps (including CCP gaps and also gaps between CCDs) and
    fill the ivar with 0. It also mask few pixels around gaps. The number of pixels to masked around the gaps is coming from
    offset_gap_pix variable in the aps_constants.py file

    11- vacuum [Optional, default value: False]. If set, the output wavelength would be in vacuum. Otherwise it will be air.
    This info will be recorded in the output meta.

    12- tellurics [Optional, default value: False]. If set, it set the ivar of regions, probably affected by tellurics to 0.
    This info will be recorded in the output meta.

    13- fill_gap [Optional, default value: False]. If set, the code replace the fluxes within the masked regions (ivar=0) by
    an interpolated values of pixels around the gaps. designed for RVSPECFIT code.

    14- arms_ratio [Optional, default value: None]. A list of float number, to be used to apply an offset to the flux in red arm, when join_arms is requested.
    Note: For OPR3b we set it to 0.83 for Red arm (private communication with Mike@CASU). Later, it will be automatically evaluated by comparing the fluxes in the overlapped
    regions (and considering the weights of fluxed in that part). At the moment we simply set it to a fixed value.

    15- join_arms [Optional, default value: False]. If set, and if two files are available in the infiles list, the code will merge two bands.

    16- collapse [Optional, default value: False]. If set, it collapses all spectra (within a certain area, or for a number of aps_ids) into a single-form spectrum.
    in this mode, APS_ID  of this single spectrum will be -999 and CNAME and TARGID will be 'collapsed'


    17- funit [Optional, default value: None]. A value usually coming from aps_constants.py to be multiplied by fluxes after applying sensitivity function (to avoid dealing with small numbers)
    We also correct ivar for this.

    18- offset_gap_pix [Optional, default value: None]. A value defining the number of pixels to be masked around the internal gaps in each CCD and gaps between CCDS.

    19- skysub [optional: default:True]. If True we use sky subtracted spectrum, otherwise we use nss (no sky subtracted spectrum)

    20- safe_mask_gaps [optional: default:False]. If True, fill ivar for mask regions with 0. Mask regions read from the lookup table (aps_constants) for each specific set (based on camera, resolution and mode)

    21- crr [Optional, default value: False]. If set, the code try to deal with cosmic rays using median filter. It will not modify the flux but replace the ivar for CR pixel with zero.
    All these values are available in the meta of the APSOB target.

    22- mask_bad_overlap [Optional, default value: True]. If set, the code will mask the overlapped region between two arms, if the median flux in the overlapped region is significantly different.

    OUTPUTS:

    APSOB output is a multi-layer python object, contains all processed L1 data and a variable keeping all records (meta).

    Assuming you call the APSOB as targs. Then:
    1- targs.data() contains the flux, ivar , wavelength arrays and meta info for each fibre/spaxel.
    2- targs.id() list of all aps_ids available in the APSOB target
    3- targs.infiles() list of input files which have been processed. Note: It could be different to the input infiles, considering the values user set for wlranges
    and also the order of files in this list could be different to the order of files listed in the input infiles.
    4- targs.idfx() a dictionary indicates the aps_id-targid mapping, where targid is the id of each target in the APSOB targets
    5- targs.idxf() a dictionary indicates the targid-aps_id mapping.
    6- targs.wavelist() a list of final wavelength array after all corrections.
    7- targs.wlranges_def() a list of the maximum available wavelength range
    8- targs.wlranges() a list of wavelength ranges, that has been considered as working wavelength range (based on input wlranges param)
    9- targs.mode() WEAVE working mode (e.g. 'MOS','LIFU', 'MOSLIFU', 'MIFU', etc)
    10- targs.setups() list of APS working setups (e.g. [BLUEL11]), which is a combination of CAMERA (BLUE, RED, GRID) and resolution (L for LR and H for HR) and
    binning level (XBIN, YBIN).
    Note: If merge_arms is set, then it will be replaced by 'COMBINED'
    11- targs.setups_original(). Same as targs.setups(), but it will be not affected by merge_arms and it always keep the original values.
    12- targs.nband(). Number of bands (camera). If merge-arms is True, then it will be changed to 1.
    13- targs.funits(). See definition of this in the input param.
    14- targs.resolution() Indicates the resolution. (To be improved) At the moment, we simply set it to some default values for different setups.
    15- targs.obid() list of OBID for files in the infile list
    16- targs.obsdate() list of OBSERVING data, extracted from the L1 data headers.
    17- targs.res_mode() return list of res_mode (LR, HR, etc..) for all input files
    18- targs.join_arms() returns the latest status of joining-arm procedure. If two arms have been successfully merged, then this param would be True.
    19- targs.collapsed() return true if the fibres/spaxels have been collapsed to form a single spectra
    20- targs.centroid() return centroid [RA, DEC] of the analysis region, marked by area, aps_ids (and masked parts). It is different to targs.origin()
    that returns the centre of the entire field
    21- targs.skysub() return the status of sky subtraction of this target. If True, sky subtracted, if False, then no sky subtraction has been performed
    22- targs.split_arms() return True, if arms have been split after the process of joining arms but with new cuts, based on the mean wavelength of the overlapped parts.

    targs.data() STRUTURE:
    This is list of targets, each element dedicated to an specific aps_id/spaxel
    each elements is a target, which has a spectrum and meta.

    target1= targs.data()[0]

    target1 has a spectrum and meta properties.
    spectrum contains the wave, flux and ivar
    and meta contain all meta info
    if you have two input files (and not merged two arms) then for each target you have
    targ1.spectra[0] and targ1.spectra[1]
    each spectrum has 3 properties (wave, flux, ivar):
    targ1.spectra[0].wave, targ1.spectra[1].wave
    targ1.spectra[0].flux, targ1.spectra[1].flux
    targ1.spectra[0].ivar, targ1.spectra[1].ivar
    and meta contains all info for that specific targ element
    targ1.meta[0], targ1.meta[1]

    useful functions:
    If you want to get the target_id for specific aps_id in the APSOB targets
    for example for aps_id=1007
    target_id = targs.apstoid(1007)
    then you can get the data for this aps_id using
    targs.data[target_id]


    FEW EXAMPLE:
    ## HERE I put few very simple examples:

    1- Assuming you want to use all default input parameters AND merge two arms:
    infiles = ['/scrach/OPR3b/stacked_1002058.fit', '/scrach/OPR3b/stacked_1002057.fit']
    targs = APSOB(infiles, join_arms=True, arms_ratio=0.83 )

    plot the spectrum for aps_id=1006
    target_id = targs.apstoid(1006)
    pylab
    plot(targs[target_id].spectra[0].wave, targs[target_id].spectra[0].flux)


    2- Assuming you want to use all default input parameters BUT DO NOT STITCH two arms:
    infiles = ['/scrach/OPR3b/stacked_1002058.fit', '/scrach/OPR3b/stacked_1002057.fit']
    targs = APSOB(infiles, join_arms=False, arms_ratio=0.83)

    plot the spectrum for aps_id=1006
    target_id = targs.apstoid(1006)
    pylab
    # PLOT the BLUE SPECTRUM
    plot(targs[target_id].spectra[0].wave, targs[target_id].spectra[0].flux)
    # PLOT the RED SPECTRUM
    plot(targs[target_id].spectra[1].wave, targs[target_id].spectra[1].flux)

    3- READ/PREPARE L1 data for an specific list of aps_ids
    targs = APSOB(infiles, aps_ids=[1007,1006,1006])

    4- READ/PREPARE L1 data for an specific list of aps_ids and specific wavelength range
    targs = APSOB(infiles, aps_ids=[1007,1006,1006], wlranges=[[4000.0,6000.0],[7000.0,9000.0]])

    5- READ/PREPARE L1 data for an specific list of aps_ids and specific wavelength range for specific SURVEY
    targs = APSOB(infiles, aps_ids=[1007,1006,1006], wlranges=[[4000.0,6000.0],[7000.0,9000.0]], targsrvy=['GA'])

    6- READ/PREPARE L1 data but mask specific aps_ids
    targs = APSOB(infiles, mask_aps_ids=[1002])

    AND OF COURSE, A combination of all available options using input parameters!!!

    """

    ### @profile
    def __init__(
        self,
        infiles,
        aps_ids=None,
        targsrvy=None,
        targclass=None,
        mask_aps_ids=None,
        area=None,
        mask_areas=None,
        wlranges=None,
        sens_corr=True,
        crr = False,
        mask_gaps=True,
        safe_mask_gaps=True,
        vacuum=False,
        tellurics=False,
        fill_gap=False,
        arms_ratio=None,
        join_arms=False,
        split_arms=False,
        mask_bad_overlap=True,
        collapse=False,
        skysub=True,
        funit=aps_constants.funit,
        offset_gap_pix=aps_constants.offset_gap_pix,
        aps_id_sum=0,
        catdir= None,
        caldir=None,
        debugdir=None,
        configdir=None,
        use_resolution_deconvolution = True,
        template_sigma0_angstrom = 0.5, # Template instrumental resolution
        edge_pixels_to_mask = 5, # Edge pixels to mask after deconvolution
        rereplace_binned_cal = True, # to rereplace binned cal files for unbinned data if not found
        lsftype = "LSF", # options are LSF (to use lsf_ files) and FWHM (to use wave_ files)
        normalize_ivar=True,
        ivar_normalization_mode = 'balanced',  # NEW PARAMETER
        # ivar_normalization_mode = 'hybrid',  # NEW PARAMETER
        skysub_mask_residuals = False,  # NEW PARAMETER
        spaxel_weighted_lsf = False,  # opt-in per-spaxel weighted LSF/FWHM for IFU cubes (see aps_ifu_spaxel_contrib.py); off by default -- IFU/LIFU/MIFU targets keep getting the flat global FWHM/LSF unless explicitly enabled
        spaxel_weighted_lsf_weighting = "overlap",  # "overlap" (circle-circle overlap area, default) or "distance" (1/d^2) -- see aps_ifu_spaxel_contrib.contributing_fibre_weights
        extinction_corr = False,  # opt-in Galactic (SFD98+Fitzpatrick99) dust
            # correction, applied per-target using its own sky position --
            # see deredden_spectrum()/dust_transmission_curve() docstrings.
            # Off by default: NOT validated as a general flux-calibration
            # correction (only intended for REDROCK/classification use),
            # and even there, applying the FULL line-of-sight SFD value
            # was found to make STAR-vs-GALAXY classification measurably
            # WORSE for a real, heavily-reddened low-|b| target (see
            # extinction_ebv_scale) -- see the APS
            # development notes on the classification commissioning study.
        extinction_ebv_scale = 1.0,  # fraction of the full SFD E(B-V) to
            # apply when extinction_corr=True. 1.0 (full) is the physically
            # correct assumption for background sources (GALAXY/QSO) but
            # empirically over-corrects foreground Milky Way stars (SFD is
            # the total column to infinite distance); <1.0 is a stopgap for
            # stellar/low-latitude programmes until a distance-resolved
            # (3D) map is used instead.
        extinction_mapdir = None,  # directory with SFD_dust_4096_{ngp,sgp}.fits;
            # default resolves via get_sfd_ebv() (PyAPS_data/DUST or $DUST_DIR)
        extinction_ebv_fixed = None,  # if set (a single float E(B-V) in
            # mag), applied to EVERY target/spaxel instead of the
            # automatic per-target get_sfd_ebv() sky-position lookup --
            # for callers that want one manually-supplied value for a
            # whole run (e.g. the ExGal EBmV config key) rather than the
            # default per-target SFD map query. Ignored unless
            # extinction_corr=True.
    ):

        if funit is None:
            funit = aps_constants.funit
        if offset_gap_pix is None:
            offset_gap_pix = aps_constants.offset_gap_pix
        if join_arms is None:
            join_arms = False
        if split_arms is None:
            split_arms = False
        if collapse is None:
            collapse = False
        if mask_bad_overlap is None:
            mask_bad_overlap = False
        if lsftype is None:
            lsftype='LSF'
        if rereplace_binned_cal is None:
            rereplace_binned_cal = False




        ## we use deepcopy of the input lists, as infiles_info as an external function (defined outside this class)
        ## can modified the original input lists (e.g. wlranges)

        infiles_info = l1_fileinfo(
            deepcopy(infiles),
            wlranges=deepcopy(wlranges),
            arms_ratio=deepcopy(arms_ratio),
            catdir=catdir,
            caldir=caldir,
        )
        self._infiles = infiles_info["infiles"]
        self._mode = infiles_info["mode"]
        self._setups = infiles_info["setups"]
        self._setups_original = infiles_info["setups"]
        self._wlranges_def = infiles_info["wlranges_def"]
        self._wlranges = infiles_info["wlranges"]
        self._wlranges_original = infiles_info["wlranges"]
        self._lenfiles = len(self._infiles)
        self._camera = infiles_info["camera"]
        self._funit = funit
        self._offset_gap_pix = offset_gap_pix
        self._resolution = infiles_info["resolution"]
        self._obid = infiles_info["obid"]
        self._obsdate = infiles_info["obsdate"]
        self._res_mode = infiles_info["res_mode"]
        self._lsf_template = infiles_info["lsf_template"]
        self._join_arms = join_arms
        self._split_arms = split_arms
        self._mask_bad_overlap = mask_bad_overlap
        self._collapse = collapse
        self._skysub = skysub
        self._arms_ratio = infiles_info["arms_ratio"]
        self._aps_id_sum = aps_id_sum
        self._cat_names = infiles_info["cat_names"]
        self._xbin = infiles_info["xbin"]
        self._ybin = infiles_info["ybin"]
        self.calfiles = infiles_info["calfiles"]
        self.lsffiles = infiles_info["lsffiles"]
        self.debugdir = debugdir
        self.configdir = configdir

        self._sens_corr = sens_corr
        self._mask_gaps = mask_gaps
        self._vacuum = vacuum
        self._tellurics = tellurics
        self._fill_gap = fill_gap
        self._crr = crr
        self._safe_mask_gaps = safe_mask_gaps
        # These three used to be hardcoded here regardless of what was
        # passed into __init__ above — the constructor accepted
        # use_resolution_deconvolution/template_sigma0_angstrom/
        # edge_pixels_to_mask as real parameters but then silently
        # discarded them in favour of fixed literals, making them dead
        # arguments. Fixed so the constructor parameters actually take
        # effect (found while wiring these up as real explorer form
        # fields — exposing a UI control that has no effect would be
        # actively misleading).
        self._use_resolution_deconvolution = use_resolution_deconvolution
        self._template_sigma0_angstrom = template_sigma0_angstrom # Template instrumental resolution
        self._edge_pixels_to_mask = edge_pixels_to_mask # Edge pixels to mask after deconvolution
        self._caldir = caldir
        self._lsftype = lsftype
        self._fpmode = infiles_info["fpmode"]
        self._caldate = infiles_info["caldate"]
        self._pickledir = None
        self._rereplace_binned_cal = rereplace_binned_cal
        self._normalize_ivar = normalize_ivar
        self._ivar_normalization_mode = ivar_normalization_mode
        self._skysub_mask_sky_residuals = skysub_mask_residuals  # NEW PARAMETER
        self._spaxel_weighted_lsf = bool(spaxel_weighted_lsf)
        self._spaxel_weighted_lsf_weighting = spaxel_weighted_lsf_weighting or "overlap"
        self._extinction_corr = bool(extinction_corr)
        self._extinction_ebv_scale = extinction_ebv_scale
        self._extinction_mapdir = extinction_mapdir
        self._extinction_ebv_fixed = extinction_ebv_fixed
        # Validate and set configdir
        try:
            self.configdir = validate_and_set_configdir(configdir, verbose=True)
        except RuntimeError as e:
            print(str(e))
            print("[APSOB] ⚠️  WARNING: Continuing without config directory.")
            print("[APSOB] FWHM interpolation will not have template fallback.")
            self.configdir = None


        # Takeing care of the pickle directory to look for existing pickles or save new ones
        if self.configdir is not None and self._caldate is not None:
            # make sure caldate is unique
            unique_caldate = next((xcalcate for xcalcate in set(self._caldate) if xcalcate is not None), None)
            self._pickledir = f"{self.configdir}/lsf/{unique_caldate}"
            if os.path.exists(self._pickledir) is False:
                os.makedirs(self._pickledir, exist_ok=True)
                print(f"[APSOB] Created LSF/FWHM pickle directory: {self._pickledir}")
        else:
            self._pickledir = None
            # `raise`, not `sys.exit()` -- sys.exit() raises SystemExit, a
            # BaseException, which is *not* caught by any of this
            # codebase's own `except Exception` handlers (e.g.
            # aps_explorer.py's handle_l1_load). In server mode
            # (aps_explorer_session.MULTI_SESSION), that meant an
            # unresolvable configdir/caldate killed the entire gunicorn
            # worker process -- and since --workers 1 is a hard
            # requirement of the in-process multi-session design, that
            # took the whole server down for every connected user, not
            # just the one failed request. A raised exception surfaces as
            # a normal "Load failed: ..." message instead, in every mode.
            raise RuntimeError(
                "[APSOB] ERROR: configdir and caldate are required to set LSF/FWHM pickle directory."
            )

        self._aps_ids, fl0id, fl0info, fl0la, self._wcs = gen_targlist(
            self._infiles[0],
            self._mode,
            aps_ids=aps_ids,
            targsrvy=targsrvy,
            targclass=targclass,
            mask_aps_ids=mask_aps_ids,
            area=area,
            mask_areas=mask_areas,
            la_out=False,
            aps_id_sum=self._aps_id_sum,
        )
        self._targetlist = list()

        ## Create a list of x and y origins (in deg), useful for IFU mode
        self._origin = [fl0info["TARGRA_0"][0], fl0info["TARGDEC_0"][0]]

        speclist = dict()
        metalist = dict()

        ## Make sure at least one spectrum exists in the aps_ids list
        assert (
            len(self._aps_ids) > 0
        ), "ERROR (APSUTILS): NO spectrum to Proceed with! Check aps_ids, TARGCLASS,TARGSRVY, area and mask_areas params"

        ## check if join_arms is consistent with the condition of updated infiles list after running l1_fileinfo
        if len(self._infiles) < 2:
            self._join_arms = False
            self._split_arms = False

        if not self._join_arms:
            self._split_arms = False

        ## Doing all the house-keeping processes in advance.
        ## and create an empty spectrum and target objects
        ## PLEASE NOTE: ALL INFO likes fibstatus, targclass, targsrvy are coming from the first file in the infiles list (file0)
        ## Later, we pass through the files in infiles list again
        ## and actually read and distribute the data.

        for _ in range(self._lenfiles):
            for _f in self._aps_ids:
                # fl0idf is the id of the _aps_ids=_f in the original fl0 file
                fl0idf = fl0id[_f]

                if _f not in speclist:
                    speclist[_f] = []
                    metalist[_f] = []
                speclist[_f].append(APSSPEC(None, None, None, None))
                metalist[_f].append([])
                if len(speclist[_f]) == self._lenfiles:
                    self._targetlist.append(
                        APSTARG(
                            _f,
                            fl0info["CNAME"][fl0idf],
                            fl0info["TARGID"][fl0idf],
                            fl0info["TARGRA"][fl0idf],
                            fl0info["TARGDEC"][fl0idf],
                            fl0info["TARGSRVY"][fl0idf],
                            fl0info["TARGCLASS"][fl0idf],
                            fl0info["FIB_STATUS"][fl0idf],
                            fl0info["TARGUSE"][fl0idf],
                            fl0info["TARGPROG"][fl0idf],
                            speclist[_f],
                            meta=metalist[_f],
                            split_arms=self._split_arms,
                            mask_bad_overlap=self._mask_bad_overlap,
                        )
                    )

        ## remove outdated objects/variables
        del fl0id, fl0info, fl0la

        # Reverse mapping- aps_id to index in our list
        self._idfx = {y.id: x for x, y in enumerate(self._targetlist)}

        # Reverse mapping index  to aps_id in our list
        self._idxf = {x: y.id for x, y in enumerate(self._targetlist)}

        ## start filling the spectrum and target objects with data from files in the infiles list

        n_arms = len(self._infiles)
        print(f"[APSOB] Using SERIAL processing with VECTORIZATION: {n_arms} arm(s)")
        arm_results = self._process_arms_serial()

        # Assign results to targetlist
        self._assign_arm_results_to_targets(arm_results)




        if self._join_arms:

            # Same "keep whichever of fwhm_interp_join/lsf_interp_join
            # actually ran under one shared name" trick as
            # _process_single_arm_vectorized's own _interp_obj -- needed
            # below to call the live object's own
            # build_spaxel_weighted_entries() for the opt-in
            # spaxel_weighted_lsf feature.
            _interp_join_obj = None

            if self._lsftype=='FWHM':

                # =========================================================================
                # FWHM ANALYSIS - Joined/Stitched arms
                # =========================================================================

                fwhm_interp_dict_join = None

                try:
                    # Create FWHM interpolator for all arms combined
                    fwhm_interp_join = run_fwhm_analysis(
                        file_paths=self.calfiles,        # All CAL files (blue + red)
                        figdir=None,            # Output directory (None = no plots)
                        figname="Stitched",              # Plot name suffix
                        debug=True,                     # Debug output
                        make_plot=False,
                        wave_bin_width=50,               # Outlier detection bin width
                        apply_bimodal_filtering=False,   # GMM bimodal filtering
                        apply_residual_filtering=True,   # Residual outlier filtering
                        spline_order=3,                  # Cubic splines
                        spline_smoothing=None,           # Auto-smoothing
                        wave_grid_resolution=1.0,        # 1Å grid
                        overwrite=False,                 # Use cached if available
                        save_pickle=True,                # Save for future use
                        pickle_dir=self._pickledir,       # Pickle location (dedicated cache dir,
                                                           # not caldir -- see the other 3 call
                                                           # sites in this file; caldir may be a
                                                           # read-only data mount)
                        replace_binned = self._rereplace_binned_cal # if True replace the binned cal files if not found
                    )

                    if fwhm_interp_join is not None:
                        # Get interpolator dictionary
                        fwhm_interp_dict_join = fwhm_interp_join.get_interpolator_dict()
                        _interp_join_obj = fwhm_interp_join
                        print(f"FWHM interpolator (stitched) created successfully")

                        # Print summary
                        fwhm_interp_join.print_summary()
                    else:
                        print(f"Warning: FWHM interpolator (stitched) creation failed")
                        fwhm_interp_dict_join = None

                except Exception as e:
                    print(f"Error creating FWHM interpolator (stitched): {e}")
                    fwhm_interp_dict_join = None

            elif self._lsftype=='LSF':
                try:
                    # using the LSF (load the dill picke it available)
                    # Run the LSF interpolator for the whole filelist (stitched arms mode)

                    lsf_interp_join = run_lsf_analysis(
                        self.lsffiles,                    # List of LSF FITS files (e.g., [blue.fits, red.fits])
                        figdir=None,             # Output directory for plots (None = no plots)
                        figname="Stitched",               # Plot filename suffix
                        debug=True,
                        make_plot=False,
                        smooth_length=None,              # Apply 150Å smoothing
                        kernel_type='gaussian',           # or 'boxcar', 'hanning'
                        overwrite=False,                  # Skip if pickle exists
                        save_pickle=True,                # Save for fast reloading
                        pickle_dir = self._pickledir,     # Directory for pickle files
                        replace_binned = self._rereplace_binned_cal # if True replace the binned cal files if not found
                    )

                    if lsf_interp_join is not None:
                        fwhm_interp_dict_join = lsf_interp_join.get_interpolator_dict()
                        _interp_join_obj = lsf_interp_join
                    else:
                        fwhm_interp_dict_join = None

                except Exception as e:
                    print(f"Error creating FWHM interpolator (stitched): {e}")
                    fwhm_interp_dict_join = None


            else:
                fwhm_interp_dict_join = None
                print(f"Warning: Unknown lsftype '{self._lsftype}' for joined arms. Skipping FWHM/LSF interpolation.")

            # =========================================================================
            # OPT-IN: spaxel-weighted LSF/FWHM, joined-arms counterpart of
            # _process_single_arm_vectorized's own block (see that one's
            # comments for the full rationale). Needed as a *separate*
            # computation here, not a reuse of any per-arm result, because
            # the joined/stitched spectrum spans the *combined* wavelength
            # range of every arm -- a single arm's own per-arm weighted
            # interpolator would silently flat-extrapolate across the
            # other arm's wavelength range instead of using its own real
            # calibration data there. The contributing-fibre *geometry*
            # itself (which physical fibres, and how much each
            # contributes) doesn't depend on which arm recorded the light,
            # so it's resolved once here from self._infiles[0] and reused
            # against fwhm_interp_dict_join's own combined wave_grid.
            # =========================================================================
            if (self._spaxel_weighted_lsf and _interp_join_obj is not None
                    and self._mode in ("IFU", "LIFU", "MIFU") and self._targetlist):
                try:
                    from PyAPS.aps_ifu_spaxel_contrib import (
                        resolve_single_exposure_fibres,
                        contributing_fibre_weights,
                    )
                    from astropy.wcs.utils import proj_plane_pixel_scales

                    diam_arcsec = aps_constants.WEAVE_FIBRE_DIAMETER_ARCSEC.get(self._mode)
                    if diam_arcsec is None:
                        print(f"spaxel_weighted_lsf (joined): no fibre diameter known for "
                              f"mode={self._mode!r} -- skipping, all spaxels keep the global FWHM/LSF.")
                    else:
                        fibre_radius_deg = diam_arcsec / 2.0 / 3600.0
                        pix_scales_deg = np.abs(proj_plane_pixel_scales(self._wcs)[:2])
                        spaxel_radius_deg = float(np.sqrt(pix_scales_deg[0] * pix_scales_deg[1] / np.pi))

                        fibres = resolve_single_exposure_fibres(self._infiles[0])
                        if len(fibres["nspec"]) == 0:
                            print("spaxel_weighted_lsf (joined): no contributing single-exposure "
                                  "fibres resolved -- skipping, all spaxels keep the global FWHM/LSF.")
                        else:
                            spaxel_ra = np.array([t.targra for t in self._targetlist])
                            spaxel_dec = np.array([t.targdec for t in self._targetlist])
                            spaxel_aps_ids = np.array([t.aps_id for t in self._targetlist])

                            weight_matrix = contributing_fibre_weights(
                                spaxel_ra, spaxel_dec, fibres["ra"], fibres["dec"],
                                fibre_radius_deg, spaxel_radius_deg,
                                weighting=self._spaxel_weighted_lsf_weighting,
                            )
                            spaxel_entries_join = _interp_join_obj.build_spaxel_weighted_entries(
                                weight_matrix, fibres["nspec"], spaxel_aps_ids,
                            )
                            if spaxel_entries_join:
                                # Shallow copy -- same "don't mutate the
                                # live interpolator's own dict in place"
                                # reasoning as the per-arm block.
                                fwhm_interp_dict_join = dict(fwhm_interp_dict_join)
                                fwhm_interp_dict_join['spaxel_weighted'] = spaxel_entries_join
                            print(f"spaxel_weighted_lsf (joined): {len(spaxel_entries_join)}/"
                                  f"{len(self._targetlist)} spaxels got a weighted LSF/FWHM; "
                                  f"the rest fall back to the global FWHM/LSF.")
                except Exception as e:
                    print(f"spaxel_weighted_lsf (joined): failed ({e}) -- "
                          f"all spaxels keep the global FWHM/LSF.")

            ## Make sure at least two files are available to be joined
            assert (
                len(self._targetlist[0].spectra) >= 2
            ), " join_arms function requires at least 2 files/bands/arms"

            # Check if two files are in the appropriate order [B/G, R]
            stp0_list = [stp[0] for stp in self._setups]
            assert isInAlphabeticalOrder(
                stp0_list
            ), "To join arms, files should be in [B,G,R] order"

            print("Start packing data into shared memory and merging two arms")
            # Ensure that all targets are packed into shared memory
            time1 = time.time()
            for t in self._targetlist:
                # t.APS_sharedmem_pack()
                t.join_arms(fwhm_interp_dict_join)

            # update the global _setups and resolution keyword.
            # self._setups = [self._setups[0]+'_'+self._setups[1]]
            if self._split_arms:
                self._setups = ["split"]
            else:
                self._setups = ["Combined"]
            print("*** setups param has been updated to %s" % (self._setups))
            self._resolution = [np.mean(self._resolution)]
            print("*** resolution param has been updated to %s" % (self._resolution))
            time2 = time.time()
            print(
                "--- %.3f seconds to pack data into shared memory and merge two arms ---"
                % (time2 - time1)
            )

        if self._collapse:

            ## Make sure at least one target is available to be collapsed
            assert (
                len(self._targetlist) > 0
            ), "No spaxel/fibre available to be collapsed"

            ## Store fib_status, RA and DEC of all targets within the potentially collapsed zone
            fib_status_collapse = []
            targuse_collapse = []
            targprog_collapse = []
            targra_collapse = []
            targdec_collapse = []

            for t in self._targetlist:
                fib_status_collapse.append(t.fib_status)
                targuse_collapse.append(t.targuse)
                targprog_collapse.append(t.targprog)
                targra_collapse.append(t.targra)
                targdec_collapse.append(t.targdec)

            ## Make sure at least one target with active fibre is available in this process
            if not "A" in [fsc.upper().replace(" ", "") for fsc in fib_status_collapse]:
                # `raise`, not `sys.exit()` -- same reasoning as the
                # configdir/caldate check above: SystemExit isn't caught
                # by this codebase's `except Exception` handlers, so a
                # dataset that happens to trigger this (with "Collapse to
                # single spectrum" on) would kill the whole gunicorn
                # worker in server mode instead of just failing this one
                # load with a normal "Load failed: ..." message.
                raise RuntimeError("No active fibre is available to be collapsed")

            # Store SNR values for each setup to store and be used for collapse
            collapsed_snr_list = []

            ## Loop over all arms and then all targets in each arm
            for nstp, stps in enumerate(self._setups):
                flux_arr = np.zeros(len(self._targetlist[0].spectra[nstp].wave))
                ivar_arr = np.zeros(len(self._targetlist[0].spectra[nstp].wave))

                # Original method (using simple mean of flux)

                # for t in self._targetlist:
                #     flux_arr = np.nansum([flux_arr, t.spectra[nstp].flux], axis=0)
                #     ivar_arr = np.nansum([ivar_arr, t.spectra[nstp].ivar], axis=0)

                # ## Update the spectra (flux and ivar) of the first element of the targlist
                # ## following the discussion with Stefano, we use mean for flux and sum for ivar
                # self._targetlist[0].spectra[nstp].flux = flux_arr / (
                #     len(self._targetlist)
                # )
                # self._targetlist[0].spectra[nstp].ivar = ivar_arr

                # update 30 Sept 2024 (using the weighted mean)
                # Initialize arrays for weighted sums
                weighted_flux_sum = np.zeros_like(flux_arr)
                weight_sum = np.zeros_like(flux_arr)

                # Loop over each target
                for t in self._targetlist:
                    flux = t.spectra[nstp].flux
                    ivar = t.spectra[nstp].ivar  # inverse variance is used as the weight
                    ivar_arr = np.nansum([ivar_arr, t.spectra[nstp].ivar], axis=0)

                    # Update the weighted sum and the total weight
                    weighted_flux_sum += flux * ivar  # flux * weight
                    weight_sum += ivar                # weight (ivar)

                # Compute the weighted mean flux
                # Avoid division by zero by checking if weight_sum is zero
                weighted_mean_flux = np.divide(weighted_flux_sum, weight_sum, out=np.zeros_like(weighted_flux_sum), where=weight_sum!=0)

                # Update the spectra of the first target in the list
                self._targetlist[0].spectra[nstp].flux = weighted_mean_flux
                self._targetlist[0].spectra[nstp].ivar = ivar_arr

                # Calculate SNR for collapsed spectrum
                ivar_mask_value = 1.0 / (aps_constants.large_error**2)
                mask_tgs = (ivar_arr <= 10 * ivar_mask_value)
                nomask_tgs = (ivar_arr > 10 * ivar_mask_value)
                nz = np.sum(nomask_tgs)

                if nz > 0:
                    snr_i = weighted_mean_flux[nomask_tgs] * np.sqrt(ivar_arr[nomask_tgs])
                    collapsed_snr = np.nanmean(snr_i)
                else:
                    collapsed_snr = np.nan

                # Store SNR for this setup
                collapsed_snr_list.append(collapsed_snr)



            ## Update the basic info of this target than will represent the collapsed target
            ## We do this to keep the general structure of the APSOB targets

            collapse_aps_id = -999
            self._targetlist[0].id = ensure_dtype(collapse_aps_id, dtype=np.int32,  return_scalar=True)
            self._targetlist[0].aps_id = ensure_dtype(collapse_aps_id, dtype=np.int32,  return_scalar=True)
            self._targetlist[0].cname = "collapsed"
            self._targetlist[0].targid = "collapsed"
            self._targetlist[0].targra = np.mean(targra_collapse)
            self._targetlist[0].targdec = np.mean(targdec_collapse)
            # NOTE (found 28 Aug 2026, real bug): this used to read
            #   self._targetlist[0].targsrvy = self._targetlist[0].targsrvy   # no-op
            #   self._targetlist[0].targsrvy = self._targetlist[0].targclass  # clobber!
            # silently overwriting the collapsed target's real TARGSRVY
            # (survey code, e.g. "WS2023A1-022") with a copy of its
            # TARGCLASS string (e.g. "GALAXY") -- verified via a live
            # collapse run: t.targsrvy came back as "GALAXY" instead of
            # the real survey code. TARGCLASS/TARGSRVY/TARGPROG all feed
            # `_targeting_class_hint` (aps_rr.py) and SRVY_CLASS
            # assignment (`classify_target_comprehensive`) downstream, so
            # a corrupted TARGSRVY specifically breaks the TARGSRVY
            # fallback tier of that priority lookup for every collapsed
            # target -- not the TARGCLASS tier the archetype-fallback
            # fixes mostly rely on (checked first, and untouched by this
            # bug), but still a real, silent metadata corruption. Fixed
            # by simply not touching targsrvy -- same as targclass just
            # above, it keeps spaxel 0's original (real) value, which is
            # as reasonable a representative choice for a collapsed patch
            # as targclass already was.
            self._targetlist[0].fib_status = "A"
            self._targetlist[0].targuse = "T"
            self._targetlist[0].targprog = ""

            # for collapse target, empty the meta dic and only leave the gfwhm
            #
            # 'fwhm' used to always be set to this same global curve (i.e.
            # identical to 'gfwhm') regardless of whether the individual
            # spaxels being collapsed had any real per-spaxel LSF info --
            # the flux-weighted patch-average of every contributing
            # spaxel's own 'fwhm' (aggregate_bin_lsf, same "one bin
            # covering every spaxel" case the Gal/ExGal pipeline's own
            # classification stage now expects, see aps_ifu_spaxel_
            # contrib.py's own module docstring section 4) is used
            # instead whenever it's real, distinct data -- this is a
            # strict improvement, never a regression: when every spaxel's
            # own 'fwhm' is already just the same global curve (feature
            # off, or MOS-mode real-but-uniform-across-this-tiny-region
            # data), the flux-weighted average of N identical curves is
            # that exact same curve back out, so this changes nothing in
            # that case and only actually differs when spaxel_weighted_
            # lsf produced genuine per-spaxel variation to average over.
            for nstp, stps in enumerate(self._setups):
                global_fwhm_stps = self._targetlist[0].meta[nstp]['gfwhm']
                fwhm_stps = global_fwhm_stps
                try:
                    from PyAPS.aps_ifu_spaxel_contrib import aggregate_bin_lsf
                    _funcs = [
                        (t.meta[nstp].get('fwhm') or {}).get('interpolate_function')
                        for t in self._targetlist
                    ]
                    _flux = np.array([np.nanmean(t.spectra[nstp].flux) for t in self._targetlist])
                    _wave = self._targetlist[0].spectra[nstp].wave
                    _entries = aggregate_bin_lsf(_funcs, np.zeros(len(_funcs), dtype=int), _flux, _wave)
                    if 0 in _entries:
                        fwhm_stps = _entries[0]
                except Exception as e:
                    # Never let this degrade the collapse itself -- same
                    # fall-back-to-global philosophy as every other
                    # spaxel_weighted_lsf code path in this file.
                    print(f"Collapse: flux-weighted patch-average LSF failed ({e}) -- using global.")

                self._targetlist[0].meta[nstp] = dict()
                self._targetlist[0].meta[nstp]['fwhm']  =  fwhm_stps
                self._targetlist[0].meta[nstp]['gfwhm']  =  global_fwhm_stps
                # now store the SNR values estimated earlier for each setup into the meta
                self._targetlist[0].meta[nstp]['SNR'] = collapsed_snr_list[nstp]
                self._targetlist[0].meta[nstp]['SNR_datatype'] = "float"

            # self._targetlist[0].meta['fhwm'] = fwhm_interp_dict_join['global']
            # self._targetlist[0].meta['gfhwm'] = fwhm_interp_dict_join['global']


            ## Update the targlist which now only contain one target (collapsed)
            self._targetlist = [self._targetlist[0]]
            self._aps_ids = [collapse_aps_id]
            self._idfx = {collapse_aps_id: 0}
            self._idxf = {0: collapse_aps_id}

        ## Print basic info on screen
        print("")
        print("============ PyAPS BASIC PARAMETERS ============")
        print("Input files: %s" % (self._infiles))
        if self._collapse:
            print(
                "Number of APS_IDs to proceed with: %d (collapsed!)"
                % (len(self._aps_ids))
            )
        else:
            print("Number of APS_IDs to proceed with: %d " % (len(self._aps_ids)))
        print("APS working mode: %s" % (self._mode))
        print(
            "APS original setup(s) not affected by joining arms: %s"
            % (self._setups_original)
        )
        print("APS working setup(s): %s" % (self._setups))
        print("Available Wavelength ranges: %s" % (self._wlranges_def))
        print(
            "User-defined Wavelength ranges (might be modified/corrected by this routine): %s"
            % (self._wlranges)
        )
        print("number of BANDS(s): %d" % (len(self._targetlist[0].spectra)))
        print("APS funits: %s" % (1.0 / self._funit))
        print("APS working resolution: %s" % (self._resolution))
        print("Latest status of join_arms param: %s" % (self._join_arms))
        print("Latest status of split_arms param: %s" % (self._split_arms))
        print("Applied arm_ratio(s): %s" % (self._arms_ratio))
        print("Sky subtraction status: %s" % (self._skysub))

        print("================================================")

        print("")


    def _apply_sky_residual_masking(self, iv, la, infc):
        """
        Smart sky residual masking based on two-scale IVAR valley detection.
        ONLY applied above OH_FOREST_START (default 8200Å) to avoid
        introducing any bias in the clean part of the spectrum.

        Below OH_FOREST_START: no masking applied (data is clean)
        Above OH_FOREST_START: two-scale valley detection masks sky residuals

        Two-scale approach:
        Narrow window (~50Å):  catches sharp OH emission line dips
        Wide window  (~300Å):  catches broad fringing/band features
        Background = max(narrow_bg, wide_bg) — ensures both scales caught

        Why two scales:
        With a single narrow window, the background estimate is pulled
        down inside a broad valley (e.g. 8600Å fringing) so the
        threshold ratio stays ~1.0 and the broad valley is missed.
        The wide window sees above the broad valley and provides a
        higher reference level, triggering the threshold correctly.

        Parameters
        ----------
        iv   : ndarray (n_spaxels, n_wave)
            IVAR array — modified in place
        la   : ndarray (n_wave,)
            Wavelength array
        infc : int
            Arm index for logging

        Returns
        -------
        iv : ndarray
            IVAR with sky residual valleys zeroed above OH_FOREST_START
        """
        from scipy.ndimage import binary_dilation, uniform_filter1d

        # ----------------------------------------------------------------
        # CONFIGURATION
        # ----------------------------------------------------------------
        # Only apply masking above this wavelength
        # Below this: sky subtraction is reliable, no masking needed
        # Above this: OH forest is dense enough to cause M-star confusion
        OH_FOREST_START = 8200.0   # Angstroms

        # Valley threshold: mask if ivar < threshold × local_background
        # Sky OH dips typically reach <5% of local level
        # 0.5 catches shallower residuals including broad fringe features
        VALLEY_THRESHOLD = 0.5

        # Wing dilation: catch line wings where flux residual
        # is present but IVAR hasn't fully dropped to zero
        WING_PIXELS = 5   # pixels each side

        # ----------------------------------------------------------------
        # CHECK: Does this arm even reach the OH forest?
        # ----------------------------------------------------------------
        if la[-1] < OH_FOREST_START:
            print(f"[ARM {infc}] Sky valley masking: skipped "
                f"(arm ends at {la[-1]:.1f}Å < {OH_FOREST_START:.1f}Å)")
            return iv

        # Pixel scale — used for adaptive window sizing
        pix_scale = float(la[-1] - la[0]) / len(la)

        # ----------------------------------------------------------------
        # RESTRICT TO OH FOREST REGION ONLY
        # ----------------------------------------------------------------
        oh_mask_region = la >= OH_FOREST_START
        n_oh_pixels    = oh_mask_region.sum()

        # Minimum pixels needed for the wide window to be meaningful
        smooth_wide_pixels = max(151, int(300.0 / pix_scale)) | 1

        if n_oh_pixels < smooth_wide_pixels:
            print(f"[ARM {infc}] Sky valley masking: skipped "
                f"(only {n_oh_pixels} pixels above {OH_FOREST_START:.1f}Å, "
                f"need {smooth_wide_pixels})")
            return iv

        # Narrow window: ~50Å — spans a typical OH line group
        # Wide window:  ~300Å — spans broad fringing features
        # Both auto-adapt to pixel scale (works for LR and HR)
        smooth_narrow = max(51,  int(50.0  / pix_scale)) | 1
        smooth_wide   = max(151, int(300.0 / pix_scale)) | 1

        print(f"[ARM {infc}] Sky valley masking (two-scale):")
        print(f"  Wavelength region : {OH_FOREST_START:.1f} - {la[-1]:.1f}Å "
            f"({n_oh_pixels} pixels)")
        print(f"  Pixel scale       : {pix_scale:.4f} Å/pix")
        print(f"  Narrow window     : {smooth_narrow} pixels "
            f"({smooth_narrow * pix_scale:.1f}Å) — OH lines")
        print(f"  Wide window       : {smooth_wide} pixels "
            f"({smooth_wide * pix_scale:.1f}Å) — broad features")
        print(f"  Valley threshold  : {VALLEY_THRESHOLD}")
        print(f"  Wing dilation     : {WING_PIXELS} pixels")

        # Extract IVAR for OH region only
        iv_oh = iv[:, oh_mask_region].copy()

        # ----------------------------------------------------------------
        # STEP 1: Fill zeros before smoothing
        # Masked pixels (ivar=0) must be interpolated before smoothing
        # so they don't pull the background estimate down artificially
        # ----------------------------------------------------------------
        iv_filled = iv_oh.copy()
        for s in range(iv_oh.shape[0]):
            row   = iv_filled[s].copy()
            valid = row > 0
            if valid.any() and (~valid).any():
                row[~valid] = np.interp(
                    np.where(~valid)[0],
                    np.where(valid)[0],
                    row[valid])
            iv_filled[s] = row

        # ----------------------------------------------------------------
        # STEP 2: Two-scale background estimation
        #
        # Narrow background (~50Å):
        #   Accurately tracks the local IVAR level between OH lines
        #   → provides high reference for sharp OH dips
        #   → but is pulled DOWN inside broad valleys → misses them
        #
        # Wide background (~300Å):
        #   Spans multiple OH groups and broad features
        #   → stays HIGH even inside broad valleys → catches them
        #   → but too coarse to track rapid IVAR variations
        #
        # Solution: take max of both → always use the highest
        # (most reliable) reference level available
        # ----------------------------------------------------------------
        iv_bg_narrow = uniform_filter1d(
            iv_filled,
            size=smooth_narrow,
            axis=1,
            mode='nearest')

        iv_bg_wide = uniform_filter1d(
            iv_filled,
            size=smooth_wide,
            axis=1,
            mode='nearest')

        # Maximum of both backgrounds — catches valleys at both scales
        iv_background = np.maximum(iv_bg_narrow, iv_bg_wide)

        # ----------------------------------------------------------------
        # STEP 3: Detect valleys
        # A valley is a valid pixel where IVAR drops significantly
        # below the local background level at either scale
        # ----------------------------------------------------------------
        min_local_ivar = 1e-10   # avoid masking in genuinely empty regions

        valley_mask = (
            (iv_oh        >  0) &
            (iv_background > min_local_ivar) &
            (iv_oh < VALLEY_THRESHOLD * iv_background)
        )

        # ----------------------------------------------------------------
        # STEP 4: Dilate to catch line wings
        # Sky line residuals extend slightly beyond the IVAR valley
        # (flux is contaminated even where IVAR hasn't fully dropped)
        # ----------------------------------------------------------------
        structure           = np.ones((1, 2 * WING_PIXELS + 1))
        valley_mask_dilated = binary_dilation(valley_mask,
                                            structure=structure)
        # Only dilate into currently valid pixels
        valley_mask_final   = valley_mask_dilated & (iv_oh > 0)

        # ----------------------------------------------------------------
        # STEP 5: Apply masking — ONLY to OH forest region
        # Clean region below OH_FOREST_START: completely untouched
        # ----------------------------------------------------------------
        n_newly_masked = int(valley_mask_final.sum())
        iv[:, oh_mask_region] = np.where(
            valley_mask_final,
            0.0,
            iv[:, oh_mask_region])

        # ----------------------------------------------------------------
        # STEP 6: Diagnostics
        # ----------------------------------------------------------------
        if n_newly_masked > 0:
            la_oh        = la[oh_mask_region]
            first_spaxel = valley_mask_final[0]

            pct = 100.0 * n_newly_masked / (iv.shape[0] * n_oh_pixels)
            print(f"  Newly masked      : {n_newly_masked} pixels "
                f"({pct:.1f}% of OH region)")

            if first_spaxel.any():
                transitions = np.diff(first_spaxel.astype(int))
                starts      = np.where(transitions ==  1)[0] + 1
                ends        = np.where(transitions == -1)[0] + 1

                if first_spaxel[0]:
                    starts = np.concatenate([[0], starts])
                if first_spaxel[-1]:
                    ends   = np.concatenate([ends, [n_oh_pixels]])

                print(f"  Masked regions (first spaxel, "
                    f"{len(starts)} regions):")
                n_show = min(10, len(starts))
                for k in range(n_show):
                    if k < len(ends):
                        w_start = la_oh[starts[k]]
                        w_end   = la_oh[min(ends[k], n_oh_pixels - 1)]
                        print(f"    [{w_start:.1f} - {w_end:.1f}Å]")
                if len(starts) > n_show:
                    print(f"    ... and {len(starts) - n_show} more regions")

            # Breakdown: how much came from narrow vs wide scale
            n_narrow_only = int(
                ((iv_bg_narrow > 1e-10) &
                (iv_oh < VALLEY_THRESHOLD * iv_bg_narrow) &
                ~(iv_bg_wide > 1e-10) &
                (iv_oh < VALLEY_THRESHOLD * iv_bg_wide)).sum())
            n_wide_only = int(
                (~(iv_bg_narrow > 1e-10) &
                (iv_oh < VALLEY_THRESHOLD * iv_bg_narrow) &
                (iv_bg_wide > 1e-10) &
                (iv_oh < VALLEY_THRESHOLD * iv_bg_wide)).sum())
            print(f"  Caught by narrow window only : {n_narrow_only}")
            print(f"  Caught by wide window only   : {n_wide_only}")
            print(f"  Caught by both               : "
                f"{n_newly_masked - n_narrow_only - n_wide_only}")
            print(f"  Clean region below "
                f"{OH_FOREST_START:.0f}Å      : untouched ✓")

        else:
            print(f"  No valleys detected above threshold — data is clean")

        return iv


    def _apply_edge_masking(self, ivar, n_edge_pixels=5):
        """
        Mask edge pixels where resolution matrix is corrupted.

        Parameters
        ----------
        ivar : array
            Inverse variance array
        n_edge_pixels : int
            Number of edge pixels to mask (default: 5)

        Returns
        -------
        ivar : array
            Modified inverse variance with edges masked
        """
        if n_edge_pixels > 0:
            ivar[:, :n_edge_pixels] = 0.0
            ivar[:, -n_edge_pixels:] = 0.0
        return ivar




    def _load_sensitivity(self, infname, infc):
        """
        Load sensitivity data from FITS file.

        First attempts to read from extension 5 of the input file.
        If that fails and the file has 4 HDUs, tries to load from external
        sensitivity calibration file.

        Args:
            infname: Input FITS filename
            infc: Camera index

        Returns:
            numpy.ndarray: Sensitivity data as float32

        Raises:
            FileNotFoundError: If calibration directory or sensitivity file not found
            Exception: If sensitivity data cannot be loaded from either source
        """
        try:
            sens = getdata(infname, 5)
            sens = (sens * self._funit).astype(np.float32)
            return sens

        except Exception as e:
            # Print warning about fallback
            print(f"WARNING: Failed to read sensitivity from {infname} extension 5")
            print(f"         Error: {e}")
            print(f"         Attempting to load from external calibration file...")

            # Check if this is a 4-HDU file
            if len(fits.info(infname, output=False)) != 4:
                raise  # Re-raise original error if not a 4-HDU file

            # Check if calibration directory exists
            if not os.path.exists(self._caldir):
                raise FileNotFoundError(
                    f"ERROR: Calibration directory does not exist: {self._caldir}\n"
                    f"       Cannot load external sensitivity file.\n"
                    f"       Original error reading {infname} extension 5: {e}"
                )

            # Build external sensitivity filename
            caldate = str(Path(infname).parent.name)

            # Check if caldate subdirectory exists
            caldate_dir = os.path.join(self._caldir, caldate)
            if not os.path.exists(caldate_dir):
                raise FileNotFoundError(
                    f"ERROR: Calibration date directory does not exist: {caldate_dir}\n"
                    f"       Cannot load external sensitivity file.\n"
                    f"       Original error reading {infname} extension 5: {e}"
                )

            # Determine camera name
            camera = self._camera[infc].lower()
            if self._res_mode[infc].lower() == 'hr' and camera == 'green':
                camera = 'blue'

            # Determine binning suffix
            res_mode = self._res_mode[infc].lower()
            if res_mode == 'hr':
                if self._camera[infc].lower() == 'green':
                    xbin_suffix = '2'
                elif self._camera[infc].lower() == 'blue':
                    xbin_suffix = '1'
                else:
                    xbin_suffix = str(self._xbin[infc])
            elif res_mode == 'lr':
                xbin_suffix = ''
            else:
                xbin_suffix = str(self._xbin[infc])

            # Format observing mode
            fpmode_formatted = self._fpmode[infc].lower().translate(str.maketrans('', '', '-_ '))

            # Build filename
            sens_infname = os.path.join(
                self._caldir,
                caldate,
                f"sens_{camera}_{res_mode}{xbin_suffix}_{fpmode_formatted}.fit"
            )

            print(f"         Loading: {sens_infname}")

            # Check if file exists
            if not os.path.exists(sens_infname):
                raise FileNotFoundError(
                    f"ERROR: Sensitivity file not found: {sens_infname}\n"
                    f"       Calibration directory: {self._caldir}\n"
                    f"       Date subdirectory: {caldate_dir}\n"
                    f"       Original error reading {infname} extension 5: {e}"
                )

            # Load sensitivity from external file
            sens = getdata(sens_infname, 1)
            sens = (sens * self._funit).astype(np.float32)

            print(f"         Successfully loaded sensitivity from calibration file")

            return sens







    def _process_single_arm_vectorized(self, infc, infname):
        """
        Process a single arm using vectorized operations.

        NOW INCLUDES: FWHM interpolation analysis per arm
        """
        import os
        import time

        time1 = time.time()

        print(f"[ARM {infc}] Starting vectorized processing: {infname}")

        # Get configuration
        mode = self._mode
        skysub = self._skysub
        sens_corr = self._sens_corr
        mask_gaps = self._mask_gaps
        fill_gap = self._fill_gap
        crr = self._crr
        tellurics = self._tellurics
        vacuum = self._vacuum
        safe_mask_gaps = self._safe_mask_gaps


        # =========================================================================
        # FWHM ANALYSIS - Single arm
        # =========================================================================
        # _interp_obj: whichever of fwhm_interp/lsf_interp this branch
        # actually builds (only one of the two ever runs, per
        # self._lsftype) -- kept under one shared name so the opt-in
        # spaxel-weighted-LSF block further down (which needs the live
        # LSFInterpolator/FWHMInterpolator object itself, not just its
        # already-extracted plain dict, to call its own
        # build_spaxel_weighted_entries()) doesn't need to know which
        # branch ran.
        _interp_obj = None

        if self._lsftype.upper() == 'FWHM':

            fwhm_interp_dict = None
            calfile = self.calfiles[infc]

            if calfile is None:
                # No calibration file resolved for this arm's exposure --
                # either caldir wasn't configured at all (l1_fileinfo
                # leaves every *file entry None when caldir is None, see
                # its own docstring) or this file hit the "4 extensions,
                # SOLAR file" edge case there. Skip cleanly with a real
                # explanation instead of letting it reach
                # check_and_replace_binned_files(None), which iterates
                # its argument unconditionally and raises the much less
                # useful "'NoneType' object is not iterable" -- confirmed
                # live, that exact traceback is what this guard replaces.
                print(f"[ARM {infc}] No calibration (wave) file available -- "
                      f"caldir not configured for this load. Skipping FWHM.")
            else:
                try:
                    # Create FWHM interpolator for this single arm
                    fwhm_interp = run_fwhm_analysis(
                        file_paths=calfile,              # Single file for this arm
                        figdir=None,            # Output directory (None = no plots)
                        figname=f"wave_{infc}",           # Plot name suffix
                        debug=True,                     # Debug output
                        make_plot=False,
                        wave_bin_width=50,               # Outlier detection bin width
                        apply_bimodal_filtering=False,   # GMM bimodal filtering
                        apply_residual_filtering=True,   # Residual outlier filtering
                        spline_order=3,                  # Cubic splines
                        spline_smoothing=None,           # Auto-smoothing
                        wave_grid_resolution=1.0,        # 1Å grid
                        overwrite=False,                 # Use cached if available
                        save_pickle=True,                # Save for future use
                        pickle_dir=self._pickledir,         # Pickle location
                        replace_binned = self._rereplace_binned_cal # if True replace the binned cal files if not found
                    )

                    if fwhm_interp is not None:
                        # Get interpolator dictionary
                        fwhm_interp_dict = fwhm_interp.get_interpolator_dict()
                        _interp_obj = fwhm_interp
                        print(f"[ARM {infc}] FWHM interpolator created successfully")
                    else:
                        print(f"[ARM {infc}] Warning: FWHM interpolator creation failed")
                        fwhm_interp_dict = None

                except Exception as e:
                    print(f"[ARM {infc}] Error creating FWHM interpolator: {e}")
                    fwhm_interp_dict = None


        elif self._lsftype.upper() == 'LSF':
            # use the LSF pre-generated files
            fwhm_interp_dict = None
            lsffile = self.lsffiles[infc]

            if lsffile is None:
                # Same reasoning as the FWHM branch's own calfile is None
                # guard just above -- see its comment for the full story.
                print(f"[ARM {infc}] No LSF file available -- caldir not "
                      f"configured for this load. Skipping LSF.")
            else:
                try:
                    # Create LSF interpolator from single file
                    lsf_interp = run_lsf_analysis(
                        lsffile,                          # Single LSF file for this arm
                        figdir=None,             # Output directory for plots
                        figname=f"lsf_{infc}",            # Plot filename suffix
                        debug=True,
                        make_plot=False,
                        smooth_length=None,              # Apply 150Å smoothing
                        kernel_type='gaussian',
                        overwrite=False,                  # Skip if pickle exists
                        save_pickle=True,                  # Save for fast reloading
                        pickle_dir = self._pickledir,     # Directory for pickle files
                        replace_binned = self._rereplace_binned_cal # if True replace the binned cal files if not found

                    )

                    if lsf_interp is not None:
                        fwhm_interp_dict = lsf_interp.get_interpolator_dict()
                        _interp_obj = lsf_interp
                        print(f"[ARM {infc}] LSF interpolator created successfully")
                    else:
                        print(f"[ARM {infc}] Warning: LSF interpolator creation failed")
                        fwhm_interp_dict = None

                except Exception as e:
                    print(f"[ARM {infc}] Error creating LSF interpolator: {e}")
                    fwhm_interp_dict = None

        else:
            fwhm_interp_dict = None
            print(f"[ARM {infc}] No FWHM/LSF analysis performed")

        # =========================================================================
        # Continue with existing processing...
        # =========================================================================

        print(f"Analysing {infname} file at fibre/spaxel level")

        # Load file info
        aps_id_infc, id_infc, info_infc, la, wcs_infc = gen_targlist(
            infname,
            mode,
            aps_ids=self._aps_ids,
            targsrvy=None,
            targclass=None,
            mask_aps_ids=None,
            area=None,
            mask_areas=None,
            la_out=True,
            aps_id_sum=self._aps_id_sum,
        )

        # =========================================================================
        # OPT-IN: spaxel-weighted LSF/FWHM for IFU cubes (see
        # aps_ifu_spaxel_contrib.py, aps_lsf.LSFInterpolator/
        # aps_fwhm.FWHMInterpolator.build_spaxel_weighted_entries). Off by
        # default (self._spaxel_weighted_lsf=False) -- when off, this
        # block does nothing and fwhm_interp_dict is never mutated, so
        # every IFU/LIFU/MIFU target keeps getting the same flat
        # 'global' FWHM/LSF exactly as before this feature existed. Only
        # meaningful for a true stacked cube (mode in ("IFU","LIFU",
        # "MIFU")) -- MOS/MOSLIFU/MOSMIFU already get a real per-fibre
        # NSPEC lookup and have nothing to gain here.
        # =========================================================================
        if (self._spaxel_weighted_lsf and _interp_obj is not None
                and mode in ("IFU", "LIFU", "MIFU")):
            try:
                from PyAPS.aps_ifu_spaxel_contrib import (
                    resolve_single_exposure_fibres,
                    contributing_fibre_weights,
                )
                from astropy.wcs.utils import proj_plane_pixel_scales

                diam_arcsec = aps_constants.WEAVE_FIBRE_DIAMETER_ARCSEC.get(mode)
                if diam_arcsec is None:
                    print(f"[ARM {infc}] spaxel_weighted_lsf: no fibre diameter known "
                          f"for mode={mode!r} -- skipping, all spaxels keep the global FWHM/LSF.")
                else:
                    fibre_radius_deg = diam_arcsec / 2.0 / 3600.0

                    pix_scales_deg = np.abs(proj_plane_pixel_scales(wcs_infc)[:2])
                    spaxel_radius_deg = float(np.sqrt(pix_scales_deg[0] * pix_scales_deg[1] / np.pi))

                    fibres = resolve_single_exposure_fibres(infname)
                    n_fibre_instances = len(fibres["nspec"])
                    if n_fibre_instances == 0:
                        print(f"[ARM {infc}] spaxel_weighted_lsf: no contributing single-exposure "
                              f"fibres resolved from {infname}'s PROV chain -- skipping, all "
                              f"spaxels keep the global FWHM/LSF.")
                    else:
                        # Selected spaxels only (respects self._aps_ids/
                        # area/etc. filtering already folded into
                        # aps_id_infc by gen_targlist above) -- no point
                        # computing geometry for spaxels that were never
                        # going to be assigned to a target anyway.
                        row_idx = np.fromiter(
                            (id_infc[a] for a in aps_id_infc),
                            dtype=np.int64, count=len(aps_id_infc),
                        )
                        spaxel_ra = info_infc["TARGRA"][row_idx]
                        spaxel_dec = info_infc["TARGDEC"][row_idx]

                        weight_matrix = contributing_fibre_weights(
                            spaxel_ra, spaxel_dec, fibres["ra"], fibres["dec"],
                            fibre_radius_deg, spaxel_radius_deg,
                            weighting=self._spaxel_weighted_lsf_weighting,
                        )
                        spaxel_entries = _interp_obj.build_spaxel_weighted_entries(
                            weight_matrix, fibres["nspec"], aps_id_infc,
                        )
                        if spaxel_entries:
                            # A shallow copy -- NOT mutating _interp_obj's
                            # own interpolator_dict in place, which would
                            # leak a 'spaxel_weighted' key into that
                            # object's own get_available_fibers() (and
                            # everything downstream of it: smoothing,
                            # missing-fibre fill, diagnostic plots) and,
                            # since run_fwhm_analysis/run_lsf_analysis can
                            # return a cached/pickle-loaded interpolator
                            # shared across runs, could leak this one
                            # run's own spaxel geometry into an unrelated
                            # later run against the same calibration file.
                            fwhm_interp_dict = dict(fwhm_interp_dict)
                            fwhm_interp_dict['spaxel_weighted'] = spaxel_entries
                        print(f"[ARM {infc}] spaxel_weighted_lsf: {len(spaxel_entries)}/"
                              f"{len(aps_id_infc)} spaxels got a weighted LSF/FWHM from "
                              f"{n_fibre_instances} candidate fibre-instances across "
                              f"{len(fibres['source_files'])} single exposure(s); the rest "
                              f"fall back to the global FWHM/LSF.")
            except Exception as e:
                # Never let a geometry/provenance failure take down the
                # whole arm -- same disclosed degrade-to-global philosophy
                # as the calfile/lsffile-missing guards above. fwhm_interp_dict
                # is left exactly as it was (no 'spaxel_weighted' key), so
                # every spaxel just falls back to global.
                print(f"[ARM {infc}] spaxel_weighted_lsf: failed ({e}) -- "
                      f"all spaxels keep the global FWHM/LSF.")

        # Load FITS data
        if mode in ["MOS", "MOSLIFU", "MOSMIFU"]:
            if skysub:
                fl = getdata(infname, 1)
                iv = getdata(infname, 2)
            else:
                fl = getdata(infname, 3)
                iv = getdata(infname, 4)
            sens = self._load_sensitivity(infname, infc)



        elif mode in ["IFU", "LIFU", "MIFU"]:
            if skysub:
                fl = getdata(infname, 1)
                iv = getdata(infname, 2)
            else:
                fl = getdata(infname, 3)
                iv = getdata(infname, 4)
            sens = self._load_sensitivity(infname, infc)

            wcs_h1 = wcs.WCS(getheader(infname, 1))
            cube_shape = list(wcs_h1.pixel_shape)

            fl = np.reshape(fl, [cube_shape[2], cube_shape[1] * cube_shape[0]])
            iv = np.reshape(iv, [cube_shape[2], cube_shape[1] * cube_shape[0]])

            try:
                sens = np.broadcast_to(sens, [cube_shape[1] * cube_shape[0], cube_shape[2]])
            except Exception:
                sens = np.median(sens, axis=0)
                sens = np.broadcast_to(sens, [cube_shape[1] * cube_shape[0], cube_shape[2]])

            fl = np.transpose(fl)
            iv = np.transpose(iv)

        # Crop wavelength range
        if self._wlranges:
            lmin = self._wlranges[infc][0]
            lmax = self._wlranges[infc][1]
        else:
            lmin = self._wlranges_def[infc][0]
            lmax = self._wlranges_def[infc][1]

        imin = abs(la - lmin).argmin()
        imax = abs(la - lmax).argmin()

        la = la[imin:imax]
        fl = fl[:, imin:imax]
        iv = iv[:, imin:imax]
        sens = sens[:, imin:imax]

        n_spaxels, n_wave = fl.shape
        print(f"[ARM {infc}] Loaded {n_spaxels} spaxels × {n_wave} wavelengths")

        # ===================================================================
        # VECTORIZED PROCESSING (NO SPAXEL LOOP!)
        # ===================================================================

        # 1. NaN and invalid value handling
        iv = vectorized_nan_handling(fl, iv)

        # 2. Gap masking
        gaps_masked = np.zeros(n_spaxels, dtype=bool)
        if mask_gaps:
            iv, gaps_masked = vectorized_gap_masking(iv, self._offset_gap_pix)

        # 3. Safe gap masking (if enabled)
        if safe_mask_gaps:
            gap_bands = aps_constants.gap_bands
            smg_key = (mode + self._res_mode[infc] + "_" + self._camera[infc]).replace(" ", "")
            if smg_key in gap_bands.keys():
                gap_mask = vectorized_wavelength_mask(la, gap_bands[smg_key])
                iv[:, gap_mask] = 0.0


        # NEW: Edge pixel masking for resolution matrix quality
        if self._use_resolution_deconvolution:
            # Was hardcoded to 5 here regardless of self._edge_pixels_to_mask
            # (itself hardcoded until the fix at __init__ above) — same
            # "constructor parameter silently ignored" bug, same fix.
            print(f"[ARM {infc}] Masking {self._edge_pixels_to_mask} edge pixels due to resolution deconvolution")
            iv = self._apply_edge_masking(iv, n_edge_pixels=self._edge_pixels_to_mask)


        # # NEW: Sky residual masking in OH forest region (if enabled)
        # if self._skysub and self._skysub_mask_sky_residuals:
        #     print(f"[ARM {infc}] Applying sky residual masking in OH forest region")
        #     iv = self._apply_sky_residual_masking(iv, la, infc)


        # 4. Telluric masking (if enabled)
        if tellurics:
            tellurics_bands = aps_constants.tellurics
            tel_mask = vectorized_wavelength_mask(la, tellurics_bands)
            # iv[:, tel_mask] = 1.0 / (aps_constants.large_error**2)
            iv[:, tel_mask] = 0.0  # Treat tellurics like gaps

        # 5. Gap filling (if needed)
        gap_fill_applied = False
        if fill_gap:
            for i in range(n_spaxels):
                if not np.all(iv[i] == 0) and np.any(iv[i] <= 0):
                    badmask = iv[i] <= 0
                    goodmask = ~badmask
                    if np.any(goodmask) and np.any(badmask):
                        fl[i, badmask] = np.interp(la[badmask], la[goodmask], fl[i, goodmask])
            gap_fill_applied = True

        # 6. Cosmic ray removal
        cr_cleaned = np.zeros(n_spaxels, dtype=bool)
        if crr:
            iv, cr_cleaned = vectorized_cosmic_ray_removal(
                fl, iv,
                int(aps_constants.cr_med_length),
                aps_constants.cr_cut_level,
                int(aps_constants.cr_wing_length)
            )

        # 7. Sensitivity correction
        fl, iv, sens_applied = vectorized_sensitivity_correction(fl, iv, sens, sens_corr)

        # 7b. Galactic (Milky Way foreground) dust extinction correction --
        # opt-in (self._extinction_corr, default False), for REDROCK/
        # classification use only. Per-spaxel: each row of fl/iv is still
        # in FIBTABLE row order at this point (only the wavelength axis has
        # been sliced so far), same order as TARGRA/TARGDEC, so no
        # additional index bookkeeping is needed. See extinction_corr's own
        # docstring at __init__ for the caveat on extinction_ebv_scale.
        #
        # Real bug caught by a real end-to-end SLURM test run (1 Sep 2026,
        # IFU-rich night): a true IFU/LIFU/MIFU cube L1 file (stackcube_*.fit)
        # has no FIBTABLE extension at all -- it's a 3-D image cube, not a
        # per-fibre row table (only MOS/MOSLIFU/MOSMIFU stacked-fibre files
        # have one) -- so the unconditional getdata(..., extname="FIBTABLE")
        # below raised KeyError for every IFU target, taking down the whole
        # arm ("All ExGal targets failed"). For cube mode, get per-spaxel
        # sky position from the cube's own WCS instead, same
        # pix2world/reshape convention already used for cube spaxels in
        # gen_targlist() above (x fastest, y slowest, matching the fl/iv
        # spaxel-row order set up for cube mode just above in this function).
        extinction_applied = False
        if getattr(self, "_extinction_corr", False):
            if mode in ["IFU", "LIFU", "MIFU"]:
                x, y = np.meshgrid(np.arange(cube_shape[0]), np.arange(cube_shape[1]))
                pix2word = wcs_h1.all_pix2world(x, y, 0, 0)
                targra = np.reshape(pix2word[0], [cube_shape[0] * cube_shape[1]]).astype(np.float64)
                targdec = np.reshape(pix2word[1], [cube_shape[0] * cube_shape[1]]).astype(np.float64)
            else:
                fibtab_radec = getdata(infname, extname="FIBTABLE")
                targra = np.asarray(fibtab_radec["TARGRA"], dtype=np.float64)
                targdec = np.asarray(fibtab_radec["TARGDEC"], dtype=np.float64)
            if targra.shape[0] == fl.shape[0]:
                ebv_fixed = getattr(self, "_extinction_ebv_fixed", None)
                if ebv_fixed is not None:
                    # Caller-supplied single E(B-V) for every spaxel in
                    # this run (e.g. ExGal's EBmV config key, when set to
                    # a number rather than "None") -- skip the per-target
                    # sky-position SFD lookup entirely.
                    ebv_arr = np.full(targra.shape[0], float(ebv_fixed))
                else:
                    ebv_arr = get_sfd_ebv(targra, targdec, mapdir=self._extinction_mapdir)
                    ebv_arr = np.atleast_1d(ebv_arr)
                trans = np.array([
                    dust_transmission_curve(la, e, ebv_scale=self._extinction_ebv_scale)
                    for e in ebv_arr
                ])
                good_trans = np.isfinite(trans) & (trans > 0)
                trans_safe = np.where(good_trans, trans, 1.0)
                fl = np.where(good_trans, fl / trans_safe, fl)
                iv = np.where(good_trans, iv * trans_safe**2, iv)
                extinction_applied = True
                ebv_source = "fixed (config-supplied)" if ebv_fixed is not None \
                    else "per-target SFD lookup"
                print(f"[ARM {infc}] Extinction correction applied "
                      f"(ebv_scale={self._extinction_ebv_scale}, "
                      f"E(B-V) source={ebv_source}, "
                      f"median E(B-V)={np.nanmedian(ebv_arr):.3f})")
            else:
                radec_source = "WCS cube" if mode in ["IFU", "LIFU", "MIFU"] else "FIBTABLE"
                print(f"[ARM {infc}] WARNING: extinction_corr requested but "
                      f"{radec_source} sky-position count ({targra.shape[0]}) != "
                      f"spaxel count ({fl.shape[0]}) -- skipped for this arm")

        # ========================================================================
        # NEW STEP 8: DOF-BASED IVAR NORMALIZATION (DO THIS FIRST!)
        # ========================================================================
        # Calculate DOF for this arm BEFORE any arms_ratio correction
        #
        # Used to store a full `iv.copy()` here ("for debugging") and return
        # it as 'iv_original' -- confirmed dead: nothing anywhere in this
        # codebase (or its tests) ever reads the 'iv_original' key back out
        # of an arm result. On a real 32,940-spaxel LIFU arm that copy alone
        # was ~2.2s of pure memory-bandwidth-bound cost (profiled directly,
        # `docker exec` into the live explorer container against a real
        # stackcube) for a value nobody consumes -- removed.

        # Calculate DOF for this arm
        dof_this_arm = np.sum(iv > 0, axis=1)  # Per-spaxel DOF count
        total_dof_this_arm = np.sum(dof_this_arm)

        print(f"[ARM {infc}] DOF calculated: {total_dof_this_arm} valid pixels across {n_spaxels} spaxels")

        # ========================================================================
        # STEP 9: Arms ratio correction
        # ========================================================================
        # Previously only applied when `normalize_ivar=False`, on the
        # assumption that DOF/balanced IVAR normalisation "handles" what
        # arms_ratio does. It doesn't: arms_ratio rescales the actual FLUX
        # values per arm (a genuine flux-calibration correction between
        # arms, e.g. a known sensitivity-function offset), while balanced
        # IVAR normalisation only rescales each arm's WEIGHT in the chi2
        # sum -- it never touches the flux the fit is actually solving
        # against. These are orthogonal corrections, not substitutes: with
        # the old logic, any run using a genuinely non-1.0 arms_ratio
        # (e.g. reprocessing older data with a known arm flux-scale issue)
        # while `normalize_ivar=True` (the default, used everywhere in
        # practice) silently had that correction skipped entirely, with
        # only a log line ("Skipping arms_ratio=... (DOF normalization
        # enabled)") -- found reviewing the full preprocessing chain,
        # 28 Aug 2026. Fixed: always apply arms_ratio (its own function is
        # already a no-op for the default value of 1.0, so this changes
        # nothing for any run that doesn't explicitly request a different
        # ratio); balanced/hybrid/pixels IVAR normalisation, if requested,
        # still runs afterward as its own, separate, additional step.
        fl, iv, offset_applied = vectorized_arms_ratio_correction(fl, iv, self._arms_ratio[infc])
        if offset_applied:
            print(f"[ARM {infc}] Applied arms_ratio: {self._arms_ratio[infc]:.4f}")

        # ========================================================================
        # STEP 10: SNR calculation
        # ========================================================================
        ivar_mask_value = 1.0 / (aps_constants.large_error**2)

        # Filter out completely invalid spectra before SNR calculation.
        #
        # This used to be `np.nanmedian(iv, axis=1, keepdims=True) > 0`.
        # Profiling a real 32,578-spaxel LIFU arm showed that single call
        # costing 2.1-3.0s -- numpy's own nanmedian, when the array
        # contains NaNs, falls back internally to a per-row Python loop
        # (confirmed live: 32,578 separate `np.partition` calls, one per
        # spaxel, instead of one vectorized column op).
        #
        # It's also redundant: `dof_this_arm` (just computed above) is
        # exactly the per-spaxel count of iv>0 pixels, and for a row with
        # m non-NaN entries of which p are >0 (NaN never counts as >0,
        # matching nanmedian's own NaN-skipping), nanmedian(row) > 0 iff
        # p >= ceil(m/2) -- i.e. "at least half the non-NaN pixels are
        # good". Two cheap single-pass reductions (no sorting, no
        # per-row loop) give the identical boolean mask.
        n_valid_total = n_wave - np.sum(np.isnan(iv), axis=1)
        valid_spectrum_mask = dof_this_arm >= np.ceil(n_valid_total / 2.0)

        snr_array = np.zeros(n_spaxels)
        nz_array = np.zeros(n_spaxels, dtype=int)

        if np.any(valid_spectrum_mask):
            snr_array_valid, nz_array_valid = vectorized_snr_calculation(
                fl[valid_spectrum_mask.flatten()],
                iv[valid_spectrum_mask.flatten()],
                ivar_mask_value
            )
            snr_array[valid_spectrum_mask.flatten()] = snr_array_valid
            nz_array[valid_spectrum_mask.flatten()] = nz_array_valid

        # ========================================================================
        # STEP 11: Wavelength conversion to vacuum (if needed)
        # ========================================================================
        if vacuum:
            la = a2v(la)

        # ========================================================================
        # STEP 12: Return results
        # ========================================================================
        time2 = time.time()
        print(f"[ARM {infc}] Completed in {time2-time1:.2f}s ({n_spaxels/(time2-time1):.0f} spaxels/s)")

        return {
            'infc': infc,
            'fl': fl,
            'iv': iv,  # Will be normalized later in _normalize_ivar_across_arms
            'sens': sens,
            'la': la,
            'snr_array': snr_array,
            'nz_array': nz_array,
            'gaps_masked': gaps_masked,
            'cr_cleaned': cr_cleaned,
            'sens_applied': sens_applied,
            'extinction_applied': extinction_applied,
            'offset_applied': offset_applied,
            'gap_fill_applied': gap_fill_applied,
            'info_infc': info_infc,
            'id_infc': id_infc,
            'n_spaxels': n_spaxels,
            'processing_time': time2 - time1,
            'vacuum': vacuum,
            'tellurics': tellurics,
            'skysub': skysub,
            'fwhm_interp_dict': fwhm_interp_dict,
            'total_dof': total_dof_this_arm,  # For normalization
            'arms_ratio_applied': self._arms_ratio[infc] if offset_applied else 1.0
        }


    def _normalize_ivar_by_information_content(self, arm_results,
                                            mode='balanced'):
        """
        Normalize IVAR considering both information content and chi-square balance.

        Parameters
        ----------
        mode : str
            'balanced' : Equal chi-square contribution (current implementation)
            'pixels'   : Weight by number of pixels (wavelength coverage)
            'hybrid'   : Compromise between balanced and pixel-weighted
        """

        if len(arm_results) < 2:
            for result in arm_results:
                result['ivar_normalized'] = False
                result['ivar_weight'] = 1.0
            return arm_results

        print(f"\n  {'='*70}")
        print(f"  IVAR NORMALIZATION (mode: {mode})")
        print(f"  {'='*70}")

        # Calculate metrics for each arm
        n_pixels_per_arm = []
        sum_ivar_per_arm = []
        wavelength_coverage = []

        for idx, result in enumerate(arm_results):
            iv = result['iv'].astype(np.float64)
            valid_mask = iv > 0

            n_pixels = np.sum(valid_mask)
            sum_ivar = np.sum(iv[valid_mask])

            # Wavelength coverage (Angstroms)
            wave = result['la']
            wave_coverage = wave[-1] - wave[0]

            n_pixels_per_arm.append(n_pixels)
            sum_ivar_per_arm.append(sum_ivar)
            wavelength_coverage.append(wave_coverage)

        total_pixels = sum(n_pixels_per_arm)
        total_wave_coverage = sum(wavelength_coverage)

        print(f"\n  Arm Properties:")
        for idx in range(len(arm_results)):
            print(f"    Arm {idx}:")
            print(f"      Wavelength: {wavelength_coverage[idx]:.1f} Å "
                f"({100*wavelength_coverage[idx]/total_wave_coverage:.1f}% of total)")
            print(f"      Pixels: {n_pixels_per_arm[idx]:,} "
                f"({100*n_pixels_per_arm[idx]/total_pixels:.1f}% of total)")
            print(f"      Sum(IVAR): {sum_ivar_per_arm[idx]:.2e} "
                f"({100*sum_ivar_per_arm[idx]/sum(sum_ivar_per_arm):.1f}% of total)")

        # Calculate weights based on mode
        if mode == 'balanced':
            # Equal chi-square contribution (current implementation)
            target_ivar = np.mean(sum_ivar_per_arm)
            weights = [target_ivar / sum_ivar for sum_ivar in sum_ivar_per_arm]
            print(f"\n  Mode: BALANCED (equal chi-square contribution)")
            print(f"  → Each arm contributes 50% to chi-square")

        elif mode == 'pixels':
            # Weight by pixel count (information content proxy)
            target_ivar_per_pixel = np.mean([
                sum_ivar / n_pix
                for sum_ivar, n_pix in zip(sum_ivar_per_arm, n_pixels_per_arm)
            ])

            weights = []
            for sum_ivar, n_pix in zip(sum_ivar_per_arm, n_pixels_per_arm):
                # Target: each pixel should have same IVAR contribution
                target_sum_ivar = target_ivar_per_pixel * n_pix
                weight = target_sum_ivar / sum_ivar if sum_ivar > 0 else 0
                weights.append(weight)

            print(f"\n  Mode: PIXELS (weighted by wavelength coverage)")
            for idx in range(len(arm_results)):
                contribution = (n_pixels_per_arm[idx] / total_pixels) * 100
                print(f"  → Arm {idx} contributes {contribution:.1f}% to chi-square")

        elif mode == 'hybrid':
            # Hybrid: compromise between balanced and pixel-weighted
            # Use geometric mean

            # Component 1: Balanced weights
            target_ivar_balanced = np.mean(sum_ivar_per_arm)
            weights_balanced = [target_ivar_balanced / s for s in sum_ivar_per_arm]

            # Component 2: Pixel-weighted
            target_ivar_per_pixel = np.mean([
                s / n for s, n in zip(sum_ivar_per_arm, n_pixels_per_arm)
            ])
            weights_pixels = [
                (target_ivar_per_pixel * n) / s
                for s, n in zip(sum_ivar_per_arm, n_pixels_per_arm)
            ]

            # Geometric mean: sqrt(w_balanced × w_pixels)
            weights = [
                np.sqrt(wb * wp)
                for wb, wp in zip(weights_balanced, weights_pixels)
            ]

            print(f"\n  Mode: HYBRID (compromise between balanced and pixel-weighted)")
            print(f"  → Weights are geometric mean of both approaches")

        else:
            raise ValueError(f"Unknown mode: {mode}")

        print(f"\n  Normalization weights: {[f'{w:.4f}' for w in weights]}")

        # Apply weights
        for idx, result in enumerate(arm_results):
            result['iv'] = result['iv'] * weights[idx]
            result['ivar_normalized'] = True
            result['ivar_weight'] = weights[idx]
            result['normalization_mode'] = mode

        # Verify final chi-square contributions
        print(f"\n  AFTER Normalization:")
        final_sum_ivar = []

        # First pass: collect all final sum(IVAR) values
        for idx in range(len(arm_results)):
            iv = arm_results[idx]['iv']
            valid_mask = iv > 0
            sum_ivar_final = np.sum(iv[valid_mask])
            final_sum_ivar.append(sum_ivar_final)

        # Calculate total for percentage calculation
        total_sum_ivar = sum(final_sum_ivar)

        # Second pass: display with correct percentages
        for idx in range(len(arm_results)):
            sum_ivar_final = final_sum_ivar[idx]

            # FIXED: Calculate percentage of total
            contribution_pct = 100 * sum_ivar_final / total_sum_ivar if total_sum_ivar > 0 else 0

            print(f"    Arm {idx}:")
            print(f"      Sum(IVAR): {sum_ivar_final:.2e}")
            print(f"      Chi-square contribution: {contribution_pct:.1f}%")

        # Sanity check
        total_pct = sum([100 * s / total_sum_ivar for s in final_sum_ivar])
        if abs(total_pct - 100.0) > 0.1:
            print(f"\n  ⚠ WARNING: Percentages sum to {total_pct:.1f}%, not 100%!")

        print(f"  {'='*70}\n")

        return arm_results


    def _process_arms_serial(self):
        # Despite the name (kept for compatibility — this method is also
        # where the post-arm normalization steps below live, not just the
        # per-arm loop), each arm is an independent file read + LSF setup +
        # vectorized-numpy pipeline with no shared mutable state between
        # arms, so this runs them concurrently via threads rather than one
        # after another. This is a real wall-clock win, not just a
        # theoretical one: each arm's own time is dominated by FITS I/O and
        # numpy/scipy C-level calls, both of which release the GIL, so two
        # threads genuinely overlap rather than fighting over it (confirmed
        # live on a real 2-arm, 30,832-spaxel dataset — 12.68s + 14.16s
        # serial-equivalent collapsed to ~14s wall time running together,
        # cutting whole-dataset load time by roughly a quarter). A thread
        # pool (not a process pool) is deliberate: process-based
        # parallelism would need every argument/return value pickled across
        # the process boundary — multiple GB of flux/ivar arrays per arm —
        # which would likely cost more than it saves; threads share memory,
        # so there's no such transfer cost at all.
        # contextvars.copy_context() is needed here, not just belt-and-
        # braces: ThreadPoolExecutor workers do NOT inherit the
        # submitting thread's context automatically, and this method's
        # per-arm prints (there are many) go through aps_explorer.py's
        # process-wide stdout tee into the live log panel. In the
        # explorer's --multi-session (server/Docker) mode, that tee
        # routes to whichever SessionBundle the *current* contextvar
        # binding points at — without explicitly propagating it into
        # each worker thread, those prints would silently land in the
        # wrong (or no) browser's log panel instead of the session that
        # actually triggered this load. Harmless, no-op overhead outside
        # --multi-session mode (the default) or when called from a plain
        # script with no Flask request in progress at all.
        #
        # One distinct Context object per arm, not one shared between
        # them: a single contextvars.Context cannot be .run() from more
        # than one thread concurrently (confirmed — sharing one raises
        # "RuntimeError: cannot enter context: ... is already entered"
        # the moment two arms' worker threads overlap, which they always
        # do here). Each copy_context() call below still runs on THIS
        # (submitting) thread, before any worker thread starts, so every
        # copy correctly carries this thread's own contextvar bindings.
        _items = list(enumerate(self._infiles))
        _contexts = [contextvars.copy_context() for _ in _items]
        with ThreadPoolExecutor(max_workers=max(1, len(self._infiles))) as ex:
            arm_results = list(ex.map(
                lambda pair: pair[1].run(self._process_single_arm_vectorized, pair[0][0], pair[0][1]),
                zip(_items, _contexts),
            ))

        # Step 2: Sky residual masking (IVAR valleys)
        if self._skysub and self._skysub_mask_sky_residuals:
            for result in arm_results:
                result['iv'] = self._apply_sky_residual_masking(
                    result['iv'], result['la'], result['infc'])

        # Step 3: Normalisation on post-masking IVAR
        if self._normalize_ivar and len(arm_results) > 1:
            arm_results = self._normalize_ivar_by_information_content(
                arm_results, mode=self._ivar_normalization_mode)

        return arm_results



    # All the "_datatype" companion keys in the per-target metadata dict
    # built by _assign_arm_results_to_targets are literal constants --
    # they never vary by target, by arm, or by call. Hoisted to a single
    # class-level dict built once (not per-call/per-result/per-target)
    # rather than re-inserted ~29 times per spaxel; _assign_arm_results_
    # to_targets copies it into each target's own tmeta via dict-merge.
    _TMETA_DATATYPES = {
        "mode_datatype": "str", "AIRVAC_datatype": "int16",
        "TELLUR_datatype": "int16", "SKYSUB_datatype": "int16",
        "APS_ID_datatype": "int32", "NSPEC_datatype": "int16",
        "TARGNAME_datatype": "str", "CNAME_datatype": "str",
        "TARGID_datatype": "str", "TARGSRVY_datatype": "str",
        "TARGCLASS_datatype": "str", "TARGRA_datatype": "float",
        "TARGRA_0_datatype": "float", "TARGDEC_datatype": "float",
        "TARGDEC_0_datatype": "float", "SETUP_datatype": "str",
        "FIB_STATUS_datatype": "str", "TARGUSE_datatype": "str",
        "TARGPROG_datatype": "str", "GAPS_MASKED_datatype": "int16",
        "GAP_FILL_datatype": "int16", "CR_CLEAN_datatype": "int16",
        "SENS_CORR_datatype": "int16", "OFFSET_APPLIED_datatype": "int16",
        "SNR_datatype": "float", "IVAR_NORMALIZED_datatype": "int16",
        "fwhm_datatype": "dict", "gfwhm_datatype": "dict",
        # NOTE: IVAR_WEIGHT_datatype/ARMS_RATIO_APPLIED_datatype are
        # deliberately NOT here -- the original code only ever sets one
        # of IVAR_WEIGHT or ARMS_RATIO_APPLIED (mutually exclusive, see
        # ivar_normalized branch below), and always sets its matching
        # "_datatype" key alongside it, never both. Putting both constant
        # labels here unconditionally would silently add a stray
        # "*_datatype" key with no matching value key on whichever branch
        # didn't fire -- caught live by an A/B real-data comparison
        # against the pre-optimization code (exact key-set mismatch on
        # every target), fixed by keeping these two conditional below.
    }

    def _assign_arm_results_to_targets(self, arm_results):
        """Assign processed arm results to target list."""
        for result in arm_results:
            infc = result['infc']
            fl = result['fl']
            iv = result['iv']
            sens = result['sens']
            la = result['la']
            snr_array = result['snr_array']
            nz_array = result['nz_array']
            id_infc = result['id_infc']
            info_infc = result['info_infc']
            gaps_masked = result['gaps_masked']
            cr_cleaned = result['cr_cleaned']
            sens_applied = result['sens_applied']
            offset_applied = result['offset_applied']
            gap_fill_applied = result['gap_fill_applied']
            fwhm_interp_dict = result.get('fwhm_interp_dict', None)  # NEW: Extract FWHM data

            # ---------------------------------------------------------------
            # Per-RESULT (arm) template: every field below is the same for
            # every target in this arm -- computed once here instead of
            # 32,578+ times inside the loop. Profiling a real 32,578-spaxel
            # LIFU load showed this whole function costing ~1.3s of pure
            # Python dict-building overhead; most of the ~29 keys assigned
            # per target were actually invariant per arm (or, for the
            # "_datatype" keys, invariant everywhere -- see _TMETA_DATATYPES
            # above). `tmeta = dict(template)` per target is one fast C-level
            # dict copy instead of a dozen-plus individual key insertions.
            # ---------------------------------------------------------------
            template = dict(self._TMETA_DATATYPES)
            template["mode"] = self._mode
            template["AIRVAC"] = 1 if result['vacuum'] else 0
            template["TELLUR"] = 1 if result['tellurics'] else 0
            template["SKYSUB"] = 1 if result['skysub'] else 0
            template["SETUP"] = self._setups[infc]
            template["GAP_FILL"] = 1 if gap_fill_applied else 0
            template["SENS_CORR"] = 1 if sens_applied else 0
            template["OFFSET_APPLIED"] = 1 if offset_applied else 0

            ivar_normalized = result.get('ivar_normalized', False)
            template["IVAR_NORMALIZED"] = 1 if ivar_normalized else 0
            if ivar_normalized:
                template["IVAR_WEIGHT"] = float(result.get('ivar_weight', 1.0))
                template["IVAR_WEIGHT_datatype"] = "float"
            else:
                # Not normalized - record arms_ratio instead
                template["ARMS_RATIO_APPLIED"] = float(self._arms_ratio[infc])
                template["ARMS_RATIO_APPLIED_datatype"] = "float"

            # =====================================================================
            # FWHM data: `gfwhm` is *always* the same object for every
            # target in this arm (see both branches below), so it's always
            # hoistable. `fwhm` is only per-target when it needs a real
            # per-fibre/per-spaxel lookup (MOS's NSPEC lookup, or IFU cube
            # data with spaxel_weighted_lsf actually producing entries) --
            # otherwise (the common case: spaxel_weighted_lsf off, the
            # explorer's own default) it's the same global curve for every
            # target too, and gets hoisted into the template as well.
            # =====================================================================
            fwhm_per_target = False
            spaxel_weighted = None
            global_fwhm_entry = None
            if fwhm_interp_dict is not None:
                global_fwhm_entry = fwhm_interp_dict['global']
                template['gfwhm'] = global_fwhm_entry
                if self._mode in ["MOS", "MOSLIFU", "MOSMIFU"]:
                    # for parked fibres and inactive fibres we may have no
                    # fwhm estimation -- a real per-fibre NSPEC lookup that
                    # varies target to target, can't be hoisted.
                    fwhm_per_target = True
                else:
                    # True IFU cube (LIFU/MIFU/IFU): default is the flat
                    # global FWHM/LSF, same as always. When
                    # self._spaxel_weighted_lsf is on (see
                    # _process_single_arm_vectorized's own
                    # 'spaxel_weighted' block), a per-spaxel weighted entry
                    # may exist under this APS_ID -- looked up from a
                    # *separate, namespaced* sub-dict, never a bare
                    # NSPEC-style key: this cube's own fake NSPEC
                    # (gen_targlist sets NSPEC=APS_ID for IFU data) can
                    # numerically collide with a real physical fibre's
                    # NSPEC (1-960) in this exact same fwhm_interp_dict, so
                    # looking it up the same way the MOS branch above does
                    # would risk silently assigning some other fibre's
                    # calibration LSF/FWHM to a spaxel that has nothing to
                    # do with it.
                    spaxel_weighted = fwhm_interp_dict.get('spaxel_weighted')
                    if spaxel_weighted:
                        fwhm_per_target = True
                    else:
                        template['fwhm'] = global_fwhm_entry
            else:
                template['fwhm'] = None
                template['gfwhm'] = None

            # Assign to each target
            for _f in self._aps_ids:
                idinfc_f = id_infc[_f]

                # Create metadata: bulk-copy the per-arm/global-constant
                # template, then only overwrite what genuinely varies
                # per target/spaxel.
                tmeta = dict(template)

                # Add spaxel-specific metadata
                tmeta["APS_ID"] = info_infc["APS_ID"][idinfc_f]
                tmeta["NSPEC"] = info_infc["NSPEC"][idinfc_f]
                tmeta["TARGNAME"] = info_infc["TARGNAME"][idinfc_f]
                tmeta["CNAME"] = info_infc["CNAME"][idinfc_f]
                tmeta["TARGID"] = info_infc["TARGID"][idinfc_f]
                tmeta["TARGSRVY"] = info_infc["TARGSRVY"][idinfc_f]
                tmeta["TARGCLASS"] = info_infc["TARGCLASS"][idinfc_f]
                tmeta["TARGRA"] = info_infc["TARGRA"][idinfc_f]
                tmeta["TARGRA_0"] = info_infc["TARGRA_0"][idinfc_f]
                tmeta["TARGDEC"] = info_infc["TARGDEC"][idinfc_f]
                tmeta["TARGDEC_0"] = info_infc["TARGDEC_0"][idinfc_f]
                tmeta["FIB_STATUS"] = info_infc["FIB_STATUS"][idinfc_f]
                tmeta["TARGUSE"] = info_infc["TARGUSE"][idinfc_f]
                tmeta["TARGPROG"] = info_infc["TARGPROG"][idinfc_f]

                tmeta["GAPS_MASKED"] = 1 if gaps_masked[idinfc_f] else 0
                tmeta["CR_CLEAN"] = 1 if cr_cleaned[idinfc_f] else 0
                tmeta["SNR"] = float(snr_array[idinfc_f])

                # =====================================================================
                # FWHM: only redone per-target when the template couldn't
                # hoist it (see fwhm_per_target above) -- otherwise tmeta
                # already has the right 'fwhm'/'gfwhm' from the template copy.
                # =====================================================================
                if fwhm_per_target:
                    if self._mode in ["MOS", "MOSLIFU", "MOSMIFU"]:
                        try:
                            tmeta['fwhm'] = fwhm_interp_dict[int(info_infc["NSPEC"][idinfc_f])]
                        except Exception:
                            tmeta['fwhm'] = global_fwhm_entry
                    else:
                        aps_id_f = int(info_infc["APS_ID"][idinfc_f])
                        tmeta['fwhm'] = spaxel_weighted.get(aps_id_f, global_fwhm_entry)

                # Assign spectrum data
                self._targetlist[self._idfx[_f]].spectra[infc].wave = la
                self._targetlist[self._idfx[_f]].spectra[infc].flux = fl[idinfc_f]
                self._targetlist[self._idfx[_f]].spectra[infc].ivar = iv[idinfc_f]
                self._targetlist[self._idfx[_f]].spectra[infc].sens = sens[idinfc_f]
                self._targetlist[self._idfx[_f]].spectra[infc].R = None
                self._targetlist[self._idfx[_f]].spectra[infc].Rcsr = None
                self._targetlist[self._idfx[_f]].meta[infc] = tmeta





    def data(self):
        return self._targetlist

    def id(self):
        return ensure_dtype(self._aps_ids, dtype=np.int32,  return_scalar=False)

    def infiles(self):
        return self._infiles

    def idfx(self):
        # Reverse-mapping aps_id to index in our list
        return self._idfx

    def apstoid(self, aps_id):
        # return the id in the APSOB target for an specific aps_id
        if aps_id in self._idfx.keys():
            return self._idfx[aps_id]
        else:
            print("No id in the APSOB target has been assigned to this aps_id")

    def get_fwhm(self, aps_id=None, fwhm_key='fwhm'):
        """
        Retrieve FWHM interpolation functions for a specific target or global FWHM.

        This function extracts FWHM interpolation functions from target metadata,
        allowing access to wavelength-dependent resolution information for either
        a specific target (fiber-specific FWHM) or global FWHM data.

        Parameters
        ----------
        aps_id : int, optional
            APS target ID to retrieve fiber-specific FWHM interpolation functions.
            If None, returns global FWHM interpolation functions from the first target.
            Default: None

        fwhm_key : str, optional
            Key name in target metadata containing FWHM interpolation data.
            Common values:
            - 'fwhm' : Individual fiber FWHM interpolation
            - 'gfwhm': Global FWHM interpolation (averaged across all fibers)
            Default: 'fwhm'

        Returns
        -------
        list of callable functions
            List of FWHM interpolation functions, one per spectral setup/arm.
            Each function takes wavelength array(s) as input and returns
            corresponding FWHM values in Angstroms.

            Length equals the number of spectral setups (len(self._setups)).

            Function signature: fwhm_values = interpolation_func(wavelengths)

        Raises
        ------
        KeyError
            If specified aps_id is not found in the target list
        IndexError
            If fwhm_key is not present in target metadata
        AttributeError
            If target metadata structure is incomplete

        Examples
        --------
        >>> # Get fiber-specific FWHM interpolation for target 1006
        >>> fwhm_funcs = apsobj.get_fwhm(aps_id=1006, fwhm_key='fwhm')
        >>>
        >>> # Use the interpolation function for the first spectral arm
        >>> wavelengths = np.linspace(4000, 7000, 1000)
        >>> fwhm_values = fwhm_funcs[0](wavelengths)
        >>> print(f"FWHM at 5000Å: {fwhm_funcs[0](5000):.4f} Å")

        >>> # Get global FWHM interpolation (averaged across all fibers)
        >>> global_fwhm_funcs = apsobj.get_fwhm(aps_id=None, fwhm_key='gfwhm')
        >>>
        >>> # Compare fiber-specific vs global FWHM
        >>> fiber_fwhm = fwhm_funcs[0](5000)
        >>> global_fwhm = global_fwhm_funcs[0](5000)
        >>> print(f"Fiber FWHM: {fiber_fwhm:.4f} Å, Global FWHM: {global_fwhm:.4f} Å")

        >>> # Get FWHM for multiple wavelengths
        >>> test_waves = [4000, 5000, 6000, 7000]
        >>> for i, setup_name in enumerate(apsobj._setups):
        >>>     fwhm_values = fwhm_funcs[i](test_waves)
        >>>     print(f"Setup {setup_name}: FWHM = {fwhm_values}")

        Notes
        -----
        - The function returns interpolation functions, not FWHM values directly
        - Call the returned functions with wavelength arrays to get FWHM values
        - Each spectral setup/arm has its own interpolation function
        - If aps_id=None, uses global FWHM from the first target's 'gfwhm' key
        - FWHM interpolation functions are created by lsf/fwhm analysis routines
        - The interpolation functions handle extrapolation beyond the fitted range

        Target Metadata Structure
        -------------------------
        The function expects target metadata with the following structure:
        target.meta[setup_index][fwhm_key]['interpolate_function']

        Where:
        - setup_index: Index of spectral setup (0 to len(self._setups)-1)
        - fwhm_key: 'fwhm' for fiber-specific or 'gfwhm' for global
        - 'interpolate_function': Callable that takes wavelengths and returns FWHM

        See Also
        --------
        pack_2_redrock : Uses these FWHM functions to create resolution matrices
        """

        fwhm_list = []

        # Determine number of spectral setups
        if len(self._targetlist) == 0:
            raise ValueError("No targets available in targetlist")

        n_setups = len(self._targetlist[0].spectra)

        if aps_id is not None:
            # Get fiber-specific FWHM for specified target
            aps_id_int = ensure_dtype(aps_id, dtype=np.int32, return_scalar=True)
            if aps_id_int not in self._idfx.keys():
                available_ids = list(self._idfx.keys())
                raise KeyError(
                    f"APS_ID {aps_id} not found. Available IDs: "
                    f"{available_ids[:10]}{'...' if len(available_ids) > 10 else ''}"
                )

            target_index = self._idfx[aps_id_int]
            target = self._targetlist[target_index]

            # Extract FWHM interpolation functions for each setup
            for setup_idx in range(n_setups):
                try:
                    fwhm_data = target.meta[setup_idx][fwhm_key]
                    if isinstance(fwhm_data, dict) and 'interpolate_function' in fwhm_data:
                        fwhm_func = fwhm_data['interpolate_function']
                    else:
                        # Assume fwhm_data is directly the interpolation function
                        fwhm_func = fwhm_data

                    fwhm_list.append(fwhm_func)

                except (IndexError, KeyError) as e:
                    raise RuntimeError(
                        f"Missing FWHM data for APS_ID {aps_id}, setup {setup_idx}, "
                        f"key '{fwhm_key}': {e}"
                    )
        else:
            # Get global FWHM (fallback to first target's global FWHM)
            target = self._targetlist[0]

            for setup_idx in range(n_setups):
                try:
                    # For global FWHM, always use 'gfwhm' key regardless of fwhm_key parameter
                    global_key = 'gfwhm'
                    fwhm_data = target.meta[setup_idx][global_key]

                    if isinstance(fwhm_data, dict) and 'interpolate_function' in fwhm_data:
                        fwhm_func = fwhm_data['interpolate_function']
                    else:
                        fwhm_func = fwhm_data

                    fwhm_list.append(fwhm_func)

                except (IndexError, KeyError) as e:
                    raise RuntimeError(
                        f"Missing global FWHM data for setup {setup_idx}: {e}. "
                        f"Make sure FWHM interpolation has been applied to targets."
                    )

        if len(fwhm_list) != n_setups:
            raise RuntimeError(
                f"Expected {n_setups} FWHM functions but got {len(fwhm_list)}. "
                f"Check target metadata structure."
            )

        return fwhm_list


    def get_fwhm_values(self, aps_id=None, wavelengths=None, fwhm_key='fwhm'):
        """
        Convenience function to get FWHM values directly at specified wavelengths.

        This is a wrapper around get_fwhm() that automatically calls the interpolation
        functions with the provided wavelengths.

        Parameters
        ----------
        aps_id : int, optional
            APS target ID. If None, uses global FWHM.
        wavelengths : array_like
            Wavelength array in Angstroms at which to evaluate FWHM
        fwhm_key : str, optional
            FWHM key in metadata ('fwhm' or 'gfwhm')

        Returns
        -------
        list of ndarray
            FWHM values for each setup at the specified wavelengths

        Examples
        --------
        >>> # Get FWHM values at specific wavelengths
        >>> waves = np.linspace(4000, 7000, 100)
        >>> fwhm_values = apsobj.get_fwhm_values(aps_id=1006, wavelengths=waves)
        >>>
        >>> # Plot FWHM vs wavelength for first setup
        >>> plt.plot(waves, fwhm_values[0])
        >>> plt.xlabel('Wavelength (Å)')
        >>> plt.ylabel('FWHM (Å)')
        """

        if wavelengths is None:
            raise ValueError("wavelengths parameter is required")

        # Get interpolation functions
        fwhm_funcs = self.get_fwhm(aps_id=aps_id, fwhm_key=fwhm_key)

        # Evaluate at specified wavelengths
        fwhm_values = []
        for func in fwhm_funcs:
            values = func(wavelengths)
            fwhm_values.append(values)

        return fwhm_values

    def idxf(self):
        # Reverse-mapping index  to aps_id in our list
        return self._idxf

    def idtoaps(self, target_id):
        # return the APS_ID in the APSOB target for an specific target_id
        if target_id in self._idxf.keys():
            return self._idxf[target_id]
        else:
            print("No APS_ID in the APSOB target has been assigned to this target_id")

    def wavelist(self):
        wavelist = []
        for i in range(len(self._targetlist[0].spectra)):
            wavelist.append(self._targetlist[0].spectra[i].wave)
        return wavelist

    def wlranges_def(self):
        # return wlranges, evaluated based on the data itself
        return self._wlranges_def

    def wlranges(self):
        # return wlranges, as specified by user
        return self._wlranges

    def mode(self):
        # return the mode param
        return self._mode

    def fpmode(self):
        # return the fpmode param
        return self._fpmode

    def fmode(self):
        # return the single fpmode string (insted of list in case of multiarms)
        return list(set(self._fpmode))[0]

    def fmode_unit(self):
        """
        Return the normalized spectrograph unit name.

        Extracts the base spectrograph name from the fpmode, removing
        arm suffixes (-A, -B, _A, _B) and returning uppercase.

        Returns:
            str: Normalized spectrograph unit name (e.g., 'MOS', 'MIFU', 'LIFU')

        Examples:
            - 'MOS-A' or 'MOS_A' -> 'MOS'
            - 'MOS-B' or 'MOS_B' -> 'MOS'
            - 'mIFU' -> 'MIFU'
            - 'LIFU' -> 'LIFU'
        """
        # Get the fpmode string
        fpmode = self.fmode()

        # Convert to uppercase for consistent processing
        spec_upper = fpmode.upper()

        # Remove common arm suffixes
        suffixes_to_remove = ['-A', '-B', '_A', '_B']

        for suffix in suffixes_to_remove:
            if spec_upper.endswith(suffix):
                return spec_upper[:-2]  # Remove last 2 characters

        # No suffix found, return as-is (uppercase)
        return spec_upper


    def setups(self):
        # return the list of setups, available in this object
        return self._setups

    def setups_original(self):
        # return the list of setups, available in this object
        return self._setups_original

    def xbin(self):
        # return the list of binning in x, available in this object
        return self._xbin

    def ybin(self):
        # return the list of binning in y, available in this object
        return self._ybin


    def wlranges_original(self):
        # return the list of wlranges, available in this object
        return self._wlranges_original

    def nbands(self):
        # return number of bands in each spectra in this object
        return len(self._targetlist[0].spectra)

    def funits(self):
        # return the flux data units (e.g. 1.0e-18)
        return 1.0 / self._funit

    def resolution(self):
        # return the resolution, pre-generated based on the configuration
        return self._resolution

    def obid(self):
        # return unique obid for a list of input files
        return self._obid

    def obsdate(self):
        # return unique obsdate for a list of input files
        return self._obsdate

    def res_mode(self):
        # return list of res_mode for all input files
        return self._res_mode

    def join_arms(self):
        # return the latest status of joining_arms
        return self._join_arms

    def split_arms(self):
        # return the latest status of spliting_arms
        return self._split_arms

    def origin(self):
        # return a list, contains x and Y origins of the entire field (MOS or IFU) in deg.
        return self._origin

    def arms_ratio(self):
        return self._arms_ratio

    def collapsed(self):
        return self._collapse

    def centroid(self):
        ## return centroid (RA_cent, DEC_cent) for all targets within the selected area or area covered by the selected aps_ids
        ## unlike origin that returns the centre of the entire field, it returns the centroid of the working area
        return [
            np.mean([t.targra for t in self._targetlist]),
            np.mean([t.targdec for t in self._targetlist]),
        ]

    def skysub(self):
        ## if true, means we are using sky subtracted spectrum, otherwise, we are using NSS (no sky subtracted)
        return self._skysub

    def skyCoords(self):
        ## return coordinates of all targets in the field
        from astropy.coordinates import SkyCoord

        return SkyCoord(
            [t.targra for t in self._targetlist],
            [t.targdec for t in self._targetlist],
            unit="deg",
        )

    def wcs_h1(self):
        ## return the WCS info (based on h1)
        return self._wcs

    def camera(self):
        # return list of cameras (RED, GREEN, BLUE)
        return self._camera

    def normalize_ivar(self):
        """Return the status of IVAR normalization."""
        return self._normalize_ivar


    def pack_2_redrock(self, cache_Rcsr=True, use_interpolated_fwhm=True, fwhm_key='fwhm',
                    resolution_mode='fiber_specific', ncpus=1):
        """
        Pack APSOB data into Redrock-compatible format with resolution matrices.

        This method transforms the processed spectral data into the format required
        by Redrock for redshift fitting. The key enhancement is support for
        wavelength-dependent FWHM interpolation to create more accurate resolution
        matrices.

        Parameters
        ----------
        cache_Rcsr : bool, default=True
            If True, cache the CSR (Compressed Sparse Row) format of the resolution
            matrix for faster repeated access during Redrock fitting.

        use_interpolated_fwhm : bool, default=True
            If True, use FWHM interpolation from LSF analysis to create
            wavelength-dependent resolution matrices.
            If False, use fixed resolution values (legacy mode).

        fwhm_key : str, default='fwhm'
            Key in target metadata containing FWHM interpolation function.
            Options:
            - 'fwhm': Fiber-specific FWHM (different for each fiber)
            - 'gfwhm': Global FWHM (averaged across all fibers)

        resolution_mode : str, default='fiber_specific'
            How to create resolution matrices when use_interpolated_fwhm=True:
            - 'fiber_specific': Each fiber gets its own resolution matrix based on
                            its individual FWHM profile (most accurate, more memory)
            - 'global_average': All fibers share one resolution matrix based on
                            average FWHM across all fibers (faster, less memory)

        ncpus : int, default=1
            Number of parallel processes for creating resolution matrices.
            Only used when resolution_mode='fiber_specific'.
            Set to 1 for serial processing (default).
            Set to -1 to use (cpu_count - 1).
            Recommended: 4-8 for typical systems.

        Returns
        -------
        rr_list : list of redrock.targets.Target
            List of Redrock Target objects ready for redshift fitting.
            ORDER IS PRESERVED: targets are in the same order as self._targetlist

        apsmeta : dict
            Metadata dictionary containing setup info, wavelength ranges,
            and per-target metadata

        Notes
        -----
        Resolution Matrix Creation:
        - The resolution matrix R describes how the instrument blurs the spectrum
        - FWHM (Full Width at Half Maximum) defines the width of this blurring
        - Wavelength-dependent FWHM means different blurring at different wavelengths
        - More accurate R leads to better redshift fits

        Memory Considerations:
        - 'fiber_specific' mode: Creates nspect × n_targets matrices
        - 'global_average' mode: Creates only nspect matrices (shared)
        - 'fixed' mode (legacy): Creates only nspect matrices (shared)

        Parallelization:
        - Only applies to 'fiber_specific' mode
        - Order of targets is ALWAYS preserved (critical for Redrock)
        - Memory usage scales with ncpus (each process needs data copy)
        - For < 100 targets, serial (ncpus=1) may be faster due to overhead
        - For 1000+ targets, ncpus=8 can give 6-8x speedup
        """

        # Import Redrock modules
        import time

        from redrock.targets import DistTargetsCopy, Spectrum, Target

        time1 = time.time()
        print("Packing data into the REDROCK-ready object")

        # ========================================================================
        # DETERMINE NUMBER OF PROCESSES
        # ========================================================================
        import os
        if ncpus == -1:
            ncpus = max(1, os.cpu_count() - 1)
        elif ncpus < 1:
            ncpus = 1

        # Determine if we'll use parallel processing
        use_parallel = (ncpus > 1 and
                    use_interpolated_fwhm and
                    resolution_mode == 'fiber_specific' and
                    len(self._targetlist) > 10)  # Only worth it for > 10 targets

        if use_parallel:
            print(f"  Parallel processing: {ncpus} processes")
        else:
            if ncpus > 1 and resolution_mode != 'fiber_specific':
                print(f"  Serial processing (ncpus={ncpus} ignored for {resolution_mode} mode)")
                ncpus = 1
            else:
                print(f"  Serial processing")

        if use_interpolated_fwhm:
            print(f"  Resolution mode: {resolution_mode}")
            print(f"  FWHM key: {fwhm_key}")
        else:
            print("  Using fixed resolution (legacy mode)")

        # ========================================================================
        # STEP 1: Get wavelength arrays for each spectral arm/setup
        # ========================================================================
        wavelist = self.wavelist()
        nspect = self.nbands()
        print(f"  Number of spectral arms: {nspect}")

        # ========================================================================
        # STEP 2: PRE-COMPUTE GLOBAL RESOLUTION MATRICES (if applicable)
        # ========================================================================
        R = None
        Rcsr = None

        if use_interpolated_fwhm and resolution_mode == 'global_average':
            print("  Creating global-average resolution matrices (shared by all targets)...")
            R, Rcsr = zip(*[
                self._makeR_from_interpolated_fwhm_global(
                    wavelist[i],
                    i,
                    cache_Rcsr,
                    fwhm_key
                )
                for i in range(nspect)
            ])
            print(f"  ✓ Created {len(R)} global resolution matrices")

        elif use_interpolated_fwhm and resolution_mode == 'fiber_specific':
            print("  Fiber-specific mode: resolution matrices will be created per target")
            R, Rcsr = None, None

        elif not use_interpolated_fwhm:
            print("  Creating fixed-resolution matrices (legacy mode)...")
            R, Rcsr = zip(*[
                makeR(
                    wavelist[i],
                    self._resolution[i],
                    cache_Rcsr=cache_Rcsr
                )
                for i in range(nspect)
            ])
            print(f"  ✓ Created {len(R)} fixed resolution matrices")

        else:
            print("  Warning: Unknown resolution mode, will compute per target")
            R, Rcsr = None, None

        # ========================================================================
        # STEP 2.5: PRE-COMPUTE FWHM ARRAYS FOR PARALLEL PROCESSING
        # ========================================================================
        fwhm_arrays_dict = None

        if use_interpolated_fwhm and resolution_mode == 'fiber_specific':
            print("  Pre-computing FWHM arrays for fiber-specific resolution...")
            fwhm_arrays_dict = {}

            for targs in self._targetlist:
                if targs.fib_status.upper() != "A":
                    continue

                fwhm_arrays_per_target = []

                for setup_idx in range(nspect):
                    # Get FWHM data
                    if (len(targs.meta) > setup_idx and
                        fwhm_key in targs.meta[setup_idx] and
                        targs.meta[setup_idx][fwhm_key] is not None):

                        fwhm_data = targs.meta[setup_idx][fwhm_key]

                        # Get interpolation function
                        if isinstance(fwhm_data, dict) and 'interpolate_function' in fwhm_data:
                            fwhm_interp_func = fwhm_data['interpolate_function']
                        elif callable(fwhm_data):
                            fwhm_interp_func = fwhm_data
                        else:
                            fwhm_interp_func = None

                        # Pre-compute FWHM array
                        if fwhm_interp_func is not None and callable(fwhm_interp_func):
                            try:
                                fwhm_array = fwhm_interp_func(wavelist[setup_idx])
                            except Exception:
                                fwhm_array = None
                        else:
                            fwhm_array = None
                    else:
                        fwhm_array = None

                    fwhm_arrays_per_target.append(fwhm_array)

                # Store as numpy arrays (picklable)
                fwhm_arrays_dict[targs.id] = [
                    np.array(arr) if arr is not None else None
                    for arr in fwhm_arrays_per_target
                ]

            print(f"  ✓ Pre-computed FWHM arrays for {len(fwhm_arrays_dict)} targets")
        # ========================================================================
        # STEP 3: CHECK REDROCK REQUIREMENTS
        # ========================================================================
        if not isAscending(self._aps_ids):
            sys.exit("ERROR: List of input aps_ids must be ascending for Redrock")

        # ========================================================================
        # STEP 4: PROCESS TARGETS (UNIFIED CODE PATH)
        # ========================================================================
        print(f"  Processing {len(self._targetlist)} targets...")

        # Use unified processing function for both serial and parallel
        rr_list, apsmeta, n_processed, n_rejected = self._process_targets_parallel(
            wavelist, nspect, cache_Rcsr, fwhm_key,
            use_interpolated_fwhm, resolution_mode,
            R, Rcsr, ncpus, fwhm_arrays_dict
        )

        # ========================================================================
        # STEP 5: ADD GLOBAL METADATA
        # CRITICAL: This must happen BEFORE returning apsmeta!
        # ========================================================================
        apsmeta["setups"] = self._setups
        apsmeta["setups_original"] = self._setups_original
        apsmeta["wlranges_original"] = self._wlranges_original
        apsmeta["funits"] = 1.0 / self._funit
        apsmeta["wlranges"] = self._wlranges
        apsmeta["wlranges_def"] = self._wlranges_def
        apsmeta["files"] = [os.path.basename(f) for f in self._infiles]
        apsmeta["stitched"] = self._join_arms
        apsmeta["split"] = self._split_arms
        apsmeta["collapsed"] = self._collapse

        # Add resolution configuration info (useful for debugging/logging)
        apsmeta["resolution_mode"] = resolution_mode if use_interpolated_fwhm else "fixed"
        apsmeta["fwhm_interpolated"] = use_interpolated_fwhm
        apsmeta["fwhm_key_used"] = fwhm_key if use_interpolated_fwhm else None
        apsmeta["ncpus_used"] = ncpus if ncpus > 1 else 1


        # Add to global metadata
        apsmeta["ivar_normalized"] = self._normalize_ivar
        apsmeta["ivar_normalized_mode"] = self._ivar_normalization_mode if self._normalize_ivar else None


        if self._normalize_ivar:
            print(f"  IVAR normalization: Applied during APSOB initialization")
            print(f"    → Each arm contributes using a {self._ivar_normalization_mode} approach")
        else:
            print(f"  IVAR normalization: Disabled")
            print(f"    → Using arms_ratio={self._arms_ratio}")

        # ========================================================================
        # STEP 6: SUMMARY AND RETURN
        # ========================================================================
        time2 = time.time()
        print(f"  ✓ Processed {n_processed} targets, rejected {n_rejected} inactive fibers")
        print(f"  ✓ Completed in {time2 - time1:.3f} seconds")

        if ncpus > 1:
            serial_est = (time2 - time1) * ncpus
            print(f"  ✓ Estimated speedup: {serial_est/(time2-time1):.1f}x (vs serial)")

        return rr_list, apsmeta




    def _process_targets_parallel(self, wavelist, nspect, cache_Rcsr, fwhm_key,
                                use_interpolated_fwhm, resolution_mode,
                                R_global, Rcsr_global, ncpus, fwhm_arrays_dict):
        """
        Process targets using multiprocessing.

        Works for both parallel (ncpus > 1) and serial (ncpus = 1) modes.
        When ncpus=1, uses Pool(1) which is effectively serial but uses
        the same code path for consistency.

        Parameters
        ----------
        ncpus : int
            Number of processes. Set to 1 for serial execution.
        """
        from functools import partial
        from multiprocessing import Pool

        # ========================================================================
        # EXTRACT PICKLABLE DATA FROM TARGETS
        # ========================================================================
        if ncpus > 1:
            print(f"    Extracting picklable data for parallel processing...")

        target_data_list = []

        for targs in self._targetlist:
            target_data = {
                'id': targs.id,
                'aps_id': targs.aps_id,
                'targid': targs.targid,
                'cname': targs.cname,
                'targra': targs.targra,
                'targdec': targs.targdec,
                'targsrvy': targs.targsrvy,
                'targclass': targs.targclass,
                'targprog': targs.targprog,
                'targuse': targs.targuse,
                'fib_status': targs.fib_status,
                'spectra': targs.spectra,
                'meta_clean': []
            }

            # Extract metadata excluding unpicklable FWHM functions
            for setup_idx in range(len(targs.meta)):
                setup_meta = {}
                for key, value in targs.meta[setup_idx].items():
                    if key not in ['fwhm', 'gfwhm']:  # Skip unpicklable functions
                        setup_meta[key] = value
                target_data['meta_clean'].append(setup_meta)

            target_data_list.append(target_data)

        if ncpus > 1:
            print(f"    ✓ Prepared {len(target_data_list)} targets for parallel processing")

        # ========================================================================
        # PROCESS TARGETS (SERIAL OR PARALLEL)
        # ========================================================================
        worker = partial(
            _process_single_target_worker,
            wavelist=wavelist,
            nspect=nspect,
            cache_Rcsr=cache_Rcsr,
            use_interpolated_fwhm=use_interpolated_fwhm,
            resolution_mode=resolution_mode,
            R_global=R_global,
            Rcsr_global=Rcsr_global,
            setups=self._setups,
            resolution=self._resolution,
            fwhm_arrays_dict=fwhm_arrays_dict
        )

        if ncpus > 1:
            print(f"    Starting parallel pool for pack_2_redrock with {ncpus} processes...")
            with Pool(ncpus) as pool:
                results = list(pool.imap(worker, target_data_list))
            print(f"    Parallel processing complete")
        else:
            # Serial execution using map (no multiprocessing overhead)
            print(f"    Processing serially (ncpus=1)...")
            results = list(map(worker, target_data_list))
            print(f"    Serial processing complete")

        # ========================================================================
        # UNPACK RESULTS
        # ========================================================================
        rr_list = []
        apsmeta = {}
        n_processed = 0
        n_rejected = 0

        for result in results:
            if result is None:
                n_rejected += 1
            else:
                target_obj, target_id, apsmeta_id = result
                rr_list.append(target_obj)
                apsmeta[target_id] = apsmeta_id
                n_processed += 1

        return rr_list, apsmeta, n_processed, n_rejected


    def _makeR_from_interpolated_fwhm_global(self, wavelist, setup_idx, cache_Rcsr, fwhm_key):
        """Create resolution matrix using global average FWHM from all targets."""

        # Collect all FWHM values for this setup from all targets
        all_fwhm_values = []
        valid_targets = 0

        for targs in self._targetlist:
            if (targs.fib_status.upper() == "A" and
                len(targs.meta) > setup_idx and
                fwhm_key in targs.meta[setup_idx] and
                targs.meta[setup_idx][fwhm_key] is not None):
                # join_arms() legitimately sets meta[setup_idx][fwhm_key] to None
                # when that target's own FWHM/LSF interpolator creation failed
                # upstream (e.g. no valid calibration data) - the key exists either
                # way, only its value tells you whether it's usable. Without this
                # None check, such a target was miscounted as valid and crashed on
                # ['interpolate_function'] instead of being skipped like the
                # valid_targets == 0 fallback below already assumes is possible.

                fwhm_interp_func = targs.meta[setup_idx][fwhm_key]['interpolate_function']

                # Check if it's a function (interpolator) or array
                if callable(fwhm_interp_func):
                    fwhm_values = fwhm_interp_func(wavelist)
                else:
                    # Assume it's already interpolated to the right wavelengths
                    fwhm_values = fwhm_interp_func

                all_fwhm_values.append(fwhm_values)
                valid_targets += 1

        if valid_targets == 0:
            print(f"Warning: No valid FWHM data for setup {setup_idx}, falling back to fixed resolution")
            return makeR(wavelist, self._resolution[setup_idx], cache_Rcsr=cache_Rcsr)

        # Calculate global average FWHM
        all_fwhm_array = np.array(all_fwhm_values)
        global_fwhm = np.mean(all_fwhm_array, axis=0)

        print(f"Setup {setup_idx}: Global FWHM range {np.min(global_fwhm):.4f} - {np.max(global_fwhm):.4f} Å ({valid_targets} targets)")

        # Create resolution matrix using global FWHM
        return makeR_from_fwhm_array(wavelist, global_fwhm, cache_Rcsr,
                                    use_deconvolution=False,
                                    sigma0_angstrom=0.0)



    def _makeR_from_interpolated_fwhm_fiber(self, wavelist, setup_idx, targs, cache_Rcsr, fwhm_key):
        """Create resolution matrix using fiber-specific FWHM."""

        # Check if this target has FWHM data
        if (len(targs.meta) <= setup_idx or
            fwhm_key not in targs.meta[setup_idx] or
            targs.meta[setup_idx][fwhm_key] is None):
            print(f"  Warning: No FWHM data for target {targs.aps_id} setup {setup_idx}, using fixed resolution")
            return makeR(wavelist, self._resolution[min(setup_idx, len(self._resolution)-1)], cache_Rcsr=cache_Rcsr)

        fwhm_data = targs.meta[setup_idx][fwhm_key]

        # # ========================================================================
        # # DEBUG: Print what we got
        # # ========================================================================
        # print(f"\n{'='*60}")
        # print(f"DEBUG: Target APS_ID={targs.aps_id}, setup_idx={setup_idx}")
        # print(f"  fwhm_key = '{fwhm_key}'")
        # print(f"  type(fwhm_data) = {type(fwhm_data)}")

        # if isinstance(fwhm_data, dict):
        #     print(f"  fwhm_data.keys() = {fwhm_data.keys()}")
        #     if 'wavelength_range' in fwhm_data:
        #         print(f"  fwhm_data['wavelength_range'] = {fwhm_data['wavelength_range']}")
        #     if 'type' in fwhm_data:
        #         print(f"  fwhm_data['type'] = {fwhm_data['type']}")

        # print(f"  wavelist range: {wavelist[0]:.2f} - {wavelist[-1]:.2f} Å ({len(wavelist)} points)")
        # ========================================================================

        # Handle different FWHM data structures
        if isinstance(fwhm_data, dict) and 'interpolate_function' in fwhm_data:
            fwhm_interp_func = fwhm_data['interpolate_function']
        elif callable(fwhm_data):
            fwhm_interp_func = fwhm_data
        else:
            print(f"  Warning: FWHM data for target {targs.aps_id} is not callable, using fixed resolution")
            print(f"  fwhm_data = {fwhm_data}")
            return makeR(wavelist, self._resolution[min(setup_idx, len(self._resolution)-1)], cache_Rcsr=cache_Rcsr)

        # Get FWHM values for this specific target
        try:
            fwhm_values = fwhm_interp_func(wavelist)
        except Exception as e:
            print(f"  Warning: FWHM interpolation failed for target {targs.aps_id}: {e}")
            import traceback
            traceback.print_exc()
            return makeR(wavelist, self._resolution[min(setup_idx, len(self._resolution)-1)], cache_Rcsr=cache_Rcsr)

        # Create fiber-specific resolution matrix
        return makeR_from_fwhm_array(
            wavelist, fwhm_values, cache_Rcsr,
            use_deconvolution=False,
            sigma0_angstrom=0.0
        )

def _process_single_target_worker(target_data, wavelist, nspect, cache_Rcsr,
                                       use_interpolated_fwhm, resolution_mode,
                                       R_global, Rcsr_global, setups, resolution,
                                       fwhm_arrays_dict):
    """
    Worker that uses picklable target data dictionary.
    No unpicklable objects!
    """
    import numpy as np
    from redrock.targets import Spectrum, Target

    from PyAPS.aps_utils import makeR, makeR_from_fwhm_array

    # Skip inactive fibers
    if target_data['fib_status'].upper() != "A":
        return None

    # Calculate SNR from clean metadata
    if len(target_data['meta_clean']) > 0:
        SNR_id = [target_data['meta_clean'][setup_num]["SNR"]
                 for setup_num in range(len(setups))]
        SNR_TARG = np.nanmean(SNR_id)
    else:
        SNR_TARG = np.nan

    # Create Redrock metadata
    tmeta = {
        "APS_ID": int(target_data['aps_id']),
        "TARGID": str(target_data['targid']),
        "CNAME": str(target_data['cname']),
        "TARGRA": float(target_data['targra']),
        "TARGDEC": float(target_data['targdec']),
        "TARGSRVY": str(target_data['targsrvy']),
        "TARGCLASS": str(target_data['targclass']),
        "FIB_STATUS": str(target_data['fib_status']),
        "TARGUSE": str(target_data['targuse']),
        "TARGPROG": str(target_data['targprog']),
        "SNR": float(SNR_TARG)
    }

    # CREATE RESOLUTION MATRICES
    if use_interpolated_fwhm and resolution_mode == 'fiber_specific':
        # Get pre-computed FWHM arrays
        fwhm_arrays = fwhm_arrays_dict.get(target_data['id'], [None] * nspect)

        target_R_list = []
        target_Rcsr_list = []

        for setup_idx in range(nspect):
            fwhm_array = fwhm_arrays[setup_idx]

            if fwhm_array is not None:
                # Use pre-computed FWHM array
                R, Rcsr = makeR_from_fwhm_array(
                    wavelist[setup_idx],
                    fwhm_array,
                    cache_Rcsr,
                    use_deconvolution=False,
                    sigma0_angstrom=0.0
                )
            else:
                # Fallback to fixed resolution
                R, Rcsr = makeR(
                    wavelist[setup_idx],
                    resolution[min(setup_idx, len(resolution)-1)],
                    cache_Rcsr=cache_Rcsr
                )

            target_R_list.append(R)
            target_Rcsr_list.append(Rcsr)

        target_R = tuple(target_R_list)
        target_Rcsr = tuple(target_Rcsr_list)
    else:
        target_R = R_global
        target_Rcsr = Rcsr_global

    # Create Spectrum objects
    speclist = []

    for specid in range(len(target_data['spectra'])):
        band = setups[specid][0].lower() if specid < len(setups) else str(specid)
        speclist.append(
            Spectrum(
                target_data['spectra'][specid].wave.astype(np.float64),
                target_data['spectra'][specid].flux.astype(np.float64),
                target_data['spectra'][specid].ivar.astype(np.float64),
                target_R[specid],
                target_Rcsr[specid],
                band=band
            )
        )





    # Create Target
    target_obj = Target(target_data['id'], speclist, coadd=False, meta=tmeta)

    # Collect apsmeta
    apsmeta_id = {}
    if len(target_data['meta_clean']) > 0:
        for setup_num, setup_name in enumerate(setups):
            for k in target_data['meta_clean'][setup_num].keys():
                apsmeta_id[str(k) + "_%s" % setup_name[0]] = target_data['meta_clean'][setup_num][k]

    return (target_obj, target_data['id'], apsmeta_id)
###########################################################################

if __name__ == "__main__":

    print(
        "Please refer to the test directory, contains several sample scripts and examples of deploying APSOB class"
    )
