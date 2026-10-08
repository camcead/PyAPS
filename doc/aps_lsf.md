# aps_lsf.py - LSF Reader for Pre-computed B-spline FITS Files

## 📋 Overview

`aps_lsf.py` reads pre-computed Line Spread Function (LSF) B-spline representations from FITS files and provides the same interface as `aps_fwhm.py` for FWHM (Full Width at Half Maximum) interpolation.

**Key Features:**
- ✅ No fitting required - reads pre-computed B-splines directly
- ✅ Fast interpolation using stored knots and coefficients
- ✅ Handles single or multiple LSF files (e.g., blue + red arms)
- ✅ Automatic gap handling with linear interpolation
- ✅ Overlap region averaging for multi-arm data
- ✅ Flat extrapolation beyond data range
- ✅ Optional smoothing (boxcar, gaussian, hanning kernels)
- ✅ Automatic filling of missing fibers
- ✅ Comprehensive diagnostic plotting
- ✅ **Pickle save/load with dill** for fast reloading
- ✅ **Overwrite control** to skip recomputation if output exists
- ✅ **Consistent naming** between pickle and figure files

---

## 🎯 General Strategy

### Data Flow

```
LSF FITS File(s) → Read B-splines → Create Interpolators → Get FWHM
                                          ↓
                                    [Optional Smoothing]
                                          ↓
                                    [Fill Missing Fibers]
                                          ↓
                                    [Save to Pickle]
                                          ↓
                                    [Diagnostic Plots]
```

### Architecture

1. **Read Phase**: Load B-spline parameters (knots, coefficients, degree) from FITS file(s)
2. **Merge Phase** (if multiple files): Handle gaps/overlaps between arms
3. **Global Creation**: Build median master profile across all fibers
4. **Interpolator Creation**: Package functions into dictionary structure
5. **Enhancement Phase**: Apply smoothing and fill missing fibers
6. **Save Phase**: Serialize to pickle file for fast future loading
7. **Output Phase**: Return interpolator object with get_fwhm() interface

### Merging Strategy (Multi-file)

When merging multiple LSF files (e.g., blue + red arms), the code handles 6 regions:

- **Region 1**: Before arm 1 → Flat extrapolation (first value of arm 1)
- **Region 2**: Arm 1 only → Use arm 1 data
- **Region 3**: Gap between arms → LINEAR interpolation (per-fiber boundaries)
- **Region 4**: Overlap region → Average both arms
- **Region 5**: Arm 2 only → Use arm 2 data
- **Region 6**: After arm 2 → Flat extrapolation (last value of arm 2)

**Gap Handling:**
- Individual fibers: Each fiber uses its own boundary values for linear interpolation
- Global master: Uses median boundary values for linear interpolation

---

## 📦 Requirements

```python
numpy
matplotlib
astropy
scipy
dill          # For pickle serialization (falls back to pickle if unavailable)
```

---

## 🚀 Quick Start

### Example 1: Single File

```python
from PyAPS.aps_lsf import run_lsf_analysis

# Read single LSF file
interpolator = run_lsf_analysis(
    "<PYAPS_DATA>/LSF/20250630/lsf_BLUEL11_MOS-A.fits",
    figdir="/output/plots",
    figname="analysis",
    debug=True,
    make_plot=True
)
# Creates: BLUEL11_MOS-A.dill and BLUEL11_MOS-A_analysis.png

# Get FWHM at specific wavelength(s)
wavelengths = [4000.0, 4500.0, 5000.0]
global_fwhm = interpolator.get_fwhm(wavelengths)
print(f"Global FWHM: {global_fwhm}")

# Get FWHM for specific fiber
fiber_fwhm = interpolator.get_fwhm(wavelengths, specnum=100)
print(f"Fiber 100 FWHM: {fiber_fwhm}")
```

### Example 2: Multiple Files (Blue + Red)

```python
# Read and merge blue + red arms
interpolator = run_lsf_analysis(
    ["<PYAPS_DATA>/LSF/lsf_GREENH11_MOS-A.fits",
     "<PYAPS_DATA>/LSF/lsf_REDH11_MOS-A.fits"],
    figdir="/output/plots",
    figname="merged",
    debug=True,
    make_plot=True
)
# Creates: GREENH11__REDH11_MOS-A.dill and GREENH11__REDH11_MOS-A_merged.png

# Get FWHM across full wavelength range
wavelengths = [4000.0, 5000.0, 6000.0, 7000.0, 8000.0]
fwhm = interpolator.get_fwhm(wavelengths)
```

### Example 3: With Smoothing

```python
# Apply Gaussian smoothing during creation
interpolator = run_lsf_analysis(
    ["<PYAPS_DATA>/LSF/lsf_BLUEL11_MOS-A.fits",
     "<PYAPS_DATA>/LSF/lsf_REDL11_MOS-A.fits"],
    smooth_length=10.0,      # 10 Angstrom smoothing
    kernel_type='gaussian',   # boxcar, gaussian, or hanning
    figdir="/output/plots",
    make_plot=True
)

# Or apply smoothing after creation
interpolator.apply_smoothing(smooth_length=15.0, kernel_type='hanning')

# Remove smoothing
interpolator.remove_smoothing()
```

### Example 4: Using Overwrite Control (NEW)

```python
# First run - creates pickle and plots
interpolator = run_lsf_analysis(
    ["<PYAPS_DATA>/LSF/lsf_GREENH11_MOS-A.fits",
     "<PYAPS_DATA>/LSF/lsf_REDH11_MOS-A.fits"],
    figdir="/output",
    overwrite=False,      # Default: skip if output exists
    save_pickle=True      # Default: save to pickle
)

# Second run - loads from existing pickle (fast!)
interpolator = run_lsf_analysis(
    ["<PYAPS_DATA>/LSF/lsf_GREENH11_MOS-A.fits",
     "<PYAPS_DATA>/LSF/lsf_REDH11_MOS-A.fits"],
    figdir="/output",
    overwrite=False       # Will skip computation and load pickle
)

# Force regeneration
interpolator = run_lsf_analysis(
    ["<PYAPS_DATA>/LSF/lsf_GREENH11_MOS-A.fits",
     "<PYAPS_DATA>/LSF/lsf_REDH11_MOS-A.fits"],
    figdir="/output",
    overwrite=True        # Force recomputation even if pickle exists
)
```

### Example 5: Direct Load from Pickle (NEW)

```python
from PyAPS.aps_lsf import load_lsf_interpolator, LSFInterpolator

# Convenience function
interpolator = load_lsf_interpolator("/output/GREENH11__REDH11_MOS-A.dill")

# Or use class method
interpolator = LSFInterpolator.load("/output/GREENH11__REDH11_MOS-A.dill")

# Use immediately
fwhm = interpolator.get_fwhm(5000.0)
```

---

## 📛 Output Naming Convention (NEW)

The code generates consistent output filenames based on input files:

### Single File
| Input | Output Headname |
|-------|-----------------|
| `lsf_GREENH11_MOS-A.fits` | `GREENH11_MOS-A` |
| `lsf_BLUEL11_MOS-B.fits` | `BLUEL11_MOS-B` |
| `lsf_BLUEH11_LIFU.fits` | `BLUEH11_LIFU` |
| `lsf_REDH11_mIFU.fits` | `REDH11_mIFU` |

### Multiple Files (Double Underscore Separator)
| Input Files | Output Headname |
|-------------|-----------------|
| `lsf_GREENH11_MOS-A.fits` + `lsf_REDH11_MOS-A.fits` | `GREENH11__REDH11_MOS-A` |
| `lsf_BLUEH11_LIFU.fits` + `lsf_REDH11_LIFU.fits` | `BLUEH11__REDH11_LIFU` |
| `lsf_BLUEL11_mIFU.fits` + `lsf_REDL11_mIFU.fits` | `BLUEL11__REDL11_mIFU` |

### Generated Files
For headname `GREENH11__REDH11_MOS-A` with `figname="analysis"`:
- **Pickle**: `GREENH11__REDH11_MOS-A.dill`
- **Figure**: `GREENH11__REDH11_MOS-A_analysis.png`

---

## 🔧 Core Functions

### Main Analysis Function

#### `run_lsf_analysis(file_input, **kwargs)`

Main entry point for LSF analysis.

**Parameters:**
- `file_input` (str or list): Path(s) to LSF FITS file(s)
- `figdir` (str, optional): Output directory for diagnostic plots
- `figname` (str): Base filename for plots (default: "lsf_analysis")
- `debug` (bool): Print detailed debug information (default: False)
- `make_plot` (bool): Create diagnostic plots (default: True)
- `smooth_length` (float, optional): Smoothing length in Angstroms
- `kernel_type` (str): Smoothing kernel type: 'boxcar', 'gaussian', 'hanning' (default: 'boxcar')
- `overwrite` (bool): **NEW** - If False, skip if pickle exists. If True, regenerate. (default: False)
- `save_pickle` (bool): **NEW** - Save interpolator to pickle file (default: True)
- `pickle_dir` (str, optional): **NEW** - Directory for pickle. If None, uses figdir or input file directory.

**Returns:**
- `LSFInterpolator` object or None if failed

**Example:**
```python
interp = run_lsf_analysis(
    "lsf_BLUEL11_MOS-A.fits",
    figdir="/output",
    figname="test",
    debug=True,
    smooth_length=10.0,
    kernel_type='gaussian',
    overwrite=False,
    save_pickle=True
)
```

---

### Pickle Functions (NEW)

#### `load_lsf_interpolator(pickle_path)`

Convenience function to load an LSF interpolator from pickle.

**Parameters:**
- `pickle_path` (str): Path to pickle file

**Returns:**
- `LSFInterpolator`: Loaded interpolator

**Raises:**
- `FileNotFoundError`: If pickle file does not exist (with helpful error message)

**Example:**
```python
from PyAPS.aps_lsf import load_lsf_interpolator

interpolator = load_lsf_interpolator("<PYAPS_DATA>/output/GREENH11__REDH11_MOS-A.dill")
fwhm = interpolator.get_fwhm(5000.0)
```

#### `generate_output_headname(file_paths)`

Generate output headname based on input LSF FITS files.

**Parameters:**
- `file_paths` (str or list): Single file path or list of file paths

**Returns:**
- `str`: Output headname

**Naming Convention:**
- Single file: `{setup}_{mode}` (e.g., `GREENH11_MOS-A`)
- Multiple files: `{setup1}__{setup2}_{mode}` (e.g., `GREENH11__REDH11_MOS-A`)

**Example:**
```python
from PyAPS.aps_lsf import generate_output_headname

# Single file
headname = generate_output_headname("lsf_GREENH11_MOS-A.fits")
# Returns: "GREENH11_MOS-A"

# Multiple files
headname = generate_output_headname([
    "lsf_BLUEH11_LIFU.fits",
    "lsf_REDH11_LIFU.fits"
])
# Returns: "BLUEH11__REDH11_LIFU"
```

#### `get_output_pickle_path(file_paths, output_dir=None)`

Get the full path for the output pickle file.

**Parameters:**
- `file_paths` (str or list): Input LSF FITS file path(s)
- `output_dir` (str, optional): Output directory. If None, uses directory of first input file.

**Returns:**
- `str`: Full path to pickle file

**Example:**
```python
from PyAPS.aps_lsf import get_output_pickle_path

path = get_output_pickle_path(
    ["lsf_GREENH11_MOS-A.fits", "lsf_REDH11_MOS-A.fits"],
    output_dir="<PYAPS_DATA>/output"
)
# Returns: "<PYAPS_DATA>/output/GREENH11__REDH11_MOS-A.dill"
```

---

### Helper Functions

#### `read_lsf_splines_file(lsf_filepath, debug=False)`

Read pre-computed LSF B-splines from a single FITS file.

**Parameters:**
- `lsf_filepath` (str): Path to LSF FITS file
- `debug` (bool): Print debug information

**Returns:**
- `fiber_splines` (dict): Dictionary of fiber splines
- `metadata` (dict): File metadata
- `wave_range` (tuple): (wave_min, wave_max) across all fibers

**Checks performed:**
- File existence
- File is readable
- Valid FITS structure
- Required columns present
- Non-empty data

**Example:**
```python
from PyAPS.aps_lsf import read_lsf_splines_file

fiber_splines, metadata, wave_range = read_lsf_splines_file(
    "<PYAPS_DATA>/LSF/lsf_BLUEL11_MOS-A.fits",
    debug=True
)

print(f"Wavelength range: {wave_range}")
print(f"Number of fibers: {len(fiber_splines)}")
```

#### `parse_lsf_filename(filename)`

Parse LSF filename to extract setup information.

**Parameters:**
- `filename` (str): LSF filename

**Returns:**
- `dict`: Parsed setup information with keys:
  - `camera`: 'BLUE', 'RED', or 'GREEN'
  - `res`: 'H' (high) or 'L' (low) resolution
  - `bins`: Binning value (e.g., '11')
  - `mode`: Observing mode (e.g., 'MOS-A', 'LIFU')
  - `setup`: Full setup string (e.g., 'BLUEL11')

**Example:**
```python
from PyAPS.aps_lsf import parse_lsf_filename

info = parse_lsf_filename("lsf_BLUEL11_MOS-A.fits")
# Returns: {'camera': 'BLUE', 'res': 'L', 'bins': '11',
#           'mode': 'MOS-A', 'setup': 'BLUEL11'}
```

#### `generate_file_prefix(file_paths)`

Generate prefix for output filenames based on input FITS files.

**Parameters:**
- `file_paths` (str or list): Path(s) to input files

**Returns:**
- `str`: Generated prefix

**Example:**
```python
from PyAPS.aps_lsf import generate_file_prefix

prefix = generate_file_prefix(
    ["lsf_BLUEL11_MOS-A.fits", "lsf_REDL11_MOS-A.fits"]
)
# Returns: "lsf_BLUEL11_MOS-A_lsf_REDL11_MOS-A_"
```

#### `create_diagnostic_plots(interpolator, figdir, figname)`

Create comprehensive 4-panel diagnostic plots.

**Parameters:**
- `interpolator` (LSFInterpolator): Interpolator object
- `figdir` (str): Output directory
- `figname` (str): Base filename

**Plots created:**
1. **All fibers + global**: Shows all fiber profiles and global master
2. **Fiber-to-fiber variation**: Boxplots showing variation at test wavelengths
3. **Sample fibers**: Detailed view of 5 sample fibers with gap visualization
4. **Statistics**: Mean, ±1σ, min/max across all fibers

**Features:**
- ±200Å extrapolation range visualization
- Gap regions highlighted in yellow
- Linear interpolation in gaps shown (magenta for fibers, red for global)
- Data boundaries marked in green

**Example:**
```python
from PyAPS.aps_lsf import create_diagnostic_plots

create_diagnostic_plots(
    interpolator,
    figdir="/output/plots",
    figname="my_diagnostic"
)
# Saves: /output/plots/my_diagnostic.png
```

---

## 🎨 LSFInterpolator Class

The main class for LSF interpolation.

### Methods

#### `get_fwhm(wavelength, specnum=None)`

Get FWHM value(s) at given wavelength(s).

**Parameters:**
- `wavelength` (float or array): Wavelength(s) in Angstroms
- `specnum` (int, optional): Fiber spectrum number. If None, returns global FWHM.

**Returns:**
- `float` or `array`: FWHM value(s)

**Example:**
```python
# Single wavelength, global FWHM
fwhm = interpolator.get_fwhm(5000.0)

# Multiple wavelengths, global FWHM
fwhm_array = interpolator.get_fwhm([4000, 5000, 6000])

# Specific fiber
fwhm_fiber = interpolator.get_fwhm(5000.0, specnum=100)
```

#### `save(output_path=None, output_dir=None)` (NEW)

Save the interpolator to a pickle file using dill.

**Parameters:**
- `output_path` (str, optional): Full path to save file. If None, auto-generates based on input files.
- `output_dir` (str, optional): Directory to save to. If None, uses input file directory. Only used if output_path is None.

**Returns:**
- `str`: Path to saved file

**Example:**
```python
# Auto-generate path based on input files
saved_path = interpolator.save()
# Saves to input directory as GREENH11__REDH11_MOS-A.dill

# Specify output directory
saved_path = interpolator.save(output_dir="<PYAPS_DATA>/output")

# Specify exact path
saved_path = interpolator.save(output_path="<PYAPS_DATA>/output/my_interpolator.dill")
```

#### `load(pickle_path)` (NEW - Class Method)

Load an interpolator from a pickle file.

**Parameters:**
- `pickle_path` (str): Path to pickle file

**Returns:**
- `LSFInterpolator`: Loaded interpolator

**Raises:**
- `FileNotFoundError`: If file does not exist (with helpful message)
- `IOError`: If file is corrupted or incompatible

**Example:**
```python
# Load from pickle
interpolator = LSFInterpolator.load("<PYAPS_DATA>/output/GREENH11__REDH11_MOS-A.dill")

# Use immediately
fwhm = interpolator.get_fwhm(5000.0)
interpolator.print_summary()
```

#### `get_interpolator_dict()`

Get the complete interpolator dictionary.

**Returns:**
- `dict`: Dictionary with structure:
  ```python
  {
      'global': {
          'interpolate_function': <function>,
          'type': 'global_lsf',
          'description': 'Global master from median of all fibers',
          'wavelength_range': (wave_min, wave_max),
          'n_fibers': 960
      },
      1: {  # Fiber NSPEC
          'interpolate_function': <function>,
          'type': 'fiber_lsf' or 'fiber_lsf_merged' or 'fiber_lsf_copied',
          'specnum': 1,
          'wavelength_range': (wave_min, wave_max),
          # If copied:
          'copied_from': <source_nspec>,
          'is_copy': True
      },
      # ... more fibers
  }
  ```

**Example:**
```python
interp_dict = interpolator.get_interpolator_dict()

# Access global function
global_func = interp_dict['global']['interpolate_function']
fwhm = global_func([4000, 5000, 6000])

# Check if fiber is copied
if interp_dict[933].get('is_copy', False):
    source = interp_dict[933]['copied_from']
    print(f"Fiber 933 copied from fiber {source}")
```

#### `get_available_fibers()`

Get list of available fiber spectrum numbers.

**Returns:**
- `list`: Sorted list of fiber NSPECs (excluding 'global')

**Example:**
```python
fibers = interpolator.get_available_fibers()
print(f"Available fibers: {len(fibers)}")
print(f"First fiber: {fibers[0]}, Last fiber: {fibers[-1]}")
```

#### `apply_smoothing(smooth_length, kernel_type='boxcar')`

Apply smoothing to all fiber LSF profiles.

**Parameters:**
- `smooth_length` (float): Smoothing length in Angstroms
- `kernel_type` (str): Kernel type - 'boxcar', 'gaussian', or 'hanning'

**Kernel Types:**
- `boxcar`: Simple moving average (rectangular window)
- `gaussian`: Gaussian kernel (sigma = kernel_size / 6 for ~99% coverage)
- `hanning`: Hanning window (smooth edges, good for spectral data)

**Performance:**
- Pre-computes smoothed grids for all fibers once
- Subsequent calls to `get_fwhm()` are very fast (simple interpolation)

**Example:**
```python
# Apply 10Å boxcar smoothing
interpolator.apply_smoothing(10.0, kernel_type='boxcar')

# Change to 15Å Gaussian
interpolator.apply_smoothing(15.0, kernel_type='gaussian')

# Test difference
wave = 5000.0
fwhm_smoothed = interpolator.get_fwhm(wave)
```

#### `remove_smoothing()`

Remove smoothing and restore original interpolators.

**Example:**
```python
# Apply smoothing
interpolator.apply_smoothing(10.0, 'gaussian')
fwhm_smooth = interpolator.get_fwhm(5000.0)

# Remove smoothing
interpolator.remove_smoothing()
fwhm_original = interpolator.get_fwhm(5000.0)

print(f"Difference: {abs(fwhm_smooth - fwhm_original):.6f}")
```

#### `print_summary()`

Print summary of interpolator configuration.

**Example:**
```python
interpolator.print_summary()
```

**Output:**
```
LSF INTERPOLATOR SUMMARY:
  Source: Pre-computed B-spline FITS files
  Individual fiber splines: 847
  Global master: Median across all fibers
  Wavelength range: 3737.0 - 5875.0 Å
  Smoothing: 10.0Å (gaussian kernel)
```

---

## 📊 Output Dictionary Structure

The interpolator returns a dictionary with the following structure:

```python
{
    'global': {
        'interpolate_function': callable,  # Takes wavelength(s), returns FWHM
        'type': 'global_lsf',
        'description': 'Global master from median of all fibers',
        'wavelength_range': (wave_min, wave_max),
        'n_fibers': 960
    },

    # Original fiber
    1: {
        'interpolate_function': callable,
        'type': 'fiber_lsf',  # Single file
                             # or 'fiber_lsf_merged'  # Multi-file
        'specnum': 1,
        'wavelength_range': (wave_min, wave_max)
    },

    # Copied fiber (missing in original data)
    933: {
        'interpolate_function': callable,  # SAME as source fiber
        'type': 'fiber_lsf_copied',
        'specnum': 933,
        'wavelength_range': (wave_min, wave_max),
        'copied_from': 932,  # Source fiber NSPEC
        'is_copy': True      # Boolean flag
    }
}
```

### Accessing the Dictionary

```python
# Get the dictionary
interp_dict = interpolator.get_interpolator_dict()

# Get global FWHM
global_func = interp_dict['global']['interpolate_function']
fwhm = global_func(5000.0)

# Get fiber FWHM
fiber_func = interp_dict[100]['interpolate_function']
fwhm = fiber_func([4000, 5000, 6000])

# Check if fiber is a copy
is_copy = interp_dict[933].get('is_copy', False)
if is_copy:
    source = interp_dict[933]['copied_from']
    print(f"Fiber 933 is copied from fiber {source}")

# Get wavelength range
wave_range = interp_dict[100]['wavelength_range']
print(f"Fiber 100 valid range: {wave_range[0]:.1f} - {wave_range[1]:.1f} Å")
```

---

## 📁 LSF FITS File Format

Expected FITS file structure:

### HDU 0: Primary Header
Metadata keys (optional but recommended):
- `CALDATE`: Calibration date
- `L1_REF`: L1 reference file
- `FPMODE`: Focal plane mode
- `CAMERA`: Camera name (BLUE/RED/GREEN)
- `VPH`: VPH grating
- `CCDXBIN`: CCD binning
- `LSFDATE`: LSF creation date

### HDU 1: Binary Table "LSF_splines"
Required columns:
- `NSPEC` (int): Fiber spectrum number (1-960 for WEAVE)
- `t` (array): Knot vector (typically 12 elements for degree-5 spline)
- `c` (array): B-spline coefficients (typically 12 elements)
- `k` (int): Spline degree (typically 5)

**Example structure:**
```
NSPEC    t[12]                              c[12]                              k
-----    ---------------------------------  ---------------------------------  -
1        3737.0, 3800.0, ..., 5875.0       0.234, 0.245, ..., 0.289          5
2        3737.0, 3800.0, ..., 5875.0       0.236, 0.247, ..., 0.291          5
...
```

---

## 🔍 Use Cases

### Case 1: Quick FWHM Lookup

```python
from PyAPS.aps_lsf import run_lsf_analysis

# Quick setup - no plots
interpolator = run_lsf_analysis(
    "<PYAPS_DATA>/LSF/lsf_BLUEL11_MOS-A.fits",
    make_plot=False,
    debug=False
)

# Get FWHM for many wavelengths
wavelengths = np.linspace(4000, 5500, 1000)
fwhm_values = interpolator.get_fwhm(wavelengths)
```

### Case 2: Compare Fibers

```python
import matplotlib.pyplot as plt

# Get FWHM for multiple fibers
wavelengths = np.linspace(4000, 6000, 500)
fibers = [1, 100, 200, 300, 400, 500]

plt.figure(figsize=(10, 6))
for fib in fibers:
    fwhm = interpolator.get_fwhm(wavelengths, specnum=fib)
    plt.plot(wavelengths, fwhm, label=f'Fiber {fib}')

# Add global
fwhm_global = interpolator.get_fwhm(wavelengths)
plt.plot(wavelengths, fwhm_global, 'k--', linewidth=2, label='Global')

plt.xlabel('Wavelength (Å)')
plt.ylabel('FWHM (Å)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.title('FWHM Comparison Across Fibers')
plt.show()
```

### Case 3: Smoothing Comparison

```python
wavelengths = np.linspace(4000, 6000, 500)

# Original (no smoothing)
fwhm_original = interpolator.get_fwhm(wavelengths)

# Different smoothing levels
smoothing_lengths = [5.0, 10.0, 20.0]
kernels = ['boxcar', 'gaussian', 'hanning']

plt.figure(figsize=(12, 8))

# Plot original
plt.plot(wavelengths, fwhm_original, 'k-', linewidth=2,
         label='Original', alpha=0.8)

# Plot smoothed versions
for smooth_len in smoothing_lengths:
    for kernel in kernels:
        interpolator.apply_smoothing(smooth_len, kernel_type=kernel)
        fwhm_smooth = interpolator.get_fwhm(wavelengths)
        plt.plot(wavelengths, fwhm_smooth,
                label=f'{kernel} {smooth_len}Å', alpha=0.7)

plt.xlabel('Wavelength (Å)')
plt.ylabel('FWHM (Å)')
plt.legend()
plt.title('Effect of Different Smoothing Parameters')
plt.grid(True, alpha=0.3)
plt.show()

# Restore original
interpolator.remove_smoothing()
```

### Case 4: Save and Load Workflow (NEW)

```python
from PyAPS.aps_lsf import run_lsf_analysis, load_lsf_interpolator

# First time: Create and save
interpolator = run_lsf_analysis(
    ["<PYAPS_DATA>/LSF/lsf_GREENH11_MOS-A.fits",
     "<PYAPS_DATA>/LSF/lsf_REDH11_MOS-A.fits"],
    figdir="/output",
    overwrite=False,
    save_pickle=True
)
# Creates: /output/GREENH11__REDH11_MOS-A.dill

# Later: Quick load (no computation needed)
interpolator = load_lsf_interpolator("/output/GREENH11__REDH11_MOS-A.dill")
fwhm = interpolator.get_fwhm(5000.0)

# Or use run_lsf_analysis with overwrite=False (auto-loads if exists)
interpolator = run_lsf_analysis(
    ["<PYAPS_DATA>/LSF/lsf_GREENH11_MOS-A.fits",
     "<PYAPS_DATA>/LSF/lsf_REDH11_MOS-A.fits"],
    figdir="/output",
    overwrite=False  # Will load existing pickle
)
```

### Case 5: Export to JSON

```python
import json

# Get interpolator dictionary
interp_dict = interpolator.get_interpolator_dict()

# Create JSON-serializable version (exclude function objects)
export_dict = {}
for key, value in interp_dict.items():
    export_dict[str(key)] = {
        k: v for k, v in value.items()
        if k != 'interpolate_function'
    }

# Save to file
with open('lsf_metadata.json', 'w') as f:
    json.dump(export_dict, f, indent=2)

print("Exported LSF metadata to lsf_metadata.json")
```

### Case 6: Check Missing Fibers

```python
# Get all available fibers
all_fibers = interpolator.get_available_fibers()
interp_dict = interpolator.get_interpolator_dict()

# Find copied fibers
original_fibers = []
copied_fibers = []

for nspec in all_fibers:
    if interp_dict[nspec].get('is_copy', False):
        copied_fibers.append(nspec)
        source = interp_dict[nspec]['copied_from']
        print(f"NSPEC {nspec} <- copied from {source}")
    else:
        original_fibers.append(nspec)

print(f"\nSummary:")
print(f"  Original fibers: {len(original_fibers)}")
print(f"  Copied fibers: {len(copied_fibers)}")
print(f"  Total fibers: {len(all_fibers)}")
```

### Case 7: Batch Processing with Caching (NEW)

```python
import glob
import os
from PyAPS.aps_lsf import run_lsf_analysis, get_output_pickle_path

# Process all LSF files in directory
lsf_dir = "<PYAPS_DATA>/LSF/20250630"
output_dir = "<PYAPS_DATA>/output/lsf_cache"

# Define file combinations to process
combinations = [
    ["lsf_GREENH11_MOS-A.fits", "lsf_REDH11_MOS-A.fits"],
    ["lsf_BLUEH11_MOS-A.fits", "lsf_REDH11_MOS-A.fits"],
    ["lsf_GREENH11_MOS-B.fits", "lsf_REDH11_MOS-B.fits"],
    ["lsf_BLUEH11_LIFU.fits", "lsf_REDH11_LIFU.fits"],
]

for combo in combinations:
    full_paths = [os.path.join(lsf_dir, f) for f in combo]

    # Check what output will be generated
    pickle_path = get_output_pickle_path(full_paths, output_dir)

    if os.path.exists(pickle_path):
        print(f"⏭️  Skipping (cached): {os.path.basename(pickle_path)}")
    else:
        print(f"🔄 Processing: {combo}")

    # run_lsf_analysis handles caching automatically
    interpolator = run_lsf_analysis(
        full_paths,
        figdir=output_dir,
        make_plot=True,
        overwrite=False,  # Use cache if available
        debug=False
    )

    # Quick test
    fwhm = interpolator.get_fwhm(6000.0)
    print(f"   FWHM @ 6000Å: {fwhm:.4f}")
```

---

## ⚠️ Important Notes

### Python Closure Bug
The code carefully handles Python's closure late-binding issue using default arguments to capture variables by value, not reference. This ensures each fiber has its own unique interpolation function.

**Correct pattern used:**
```python
def make_func(data):
    captured_data = data.copy()
    def func(x, _data=captured_data):  # Capture by value
        return _data[x]
    return func
```

### BSpline Boundary Handling
BSpline evaluation exactly at boundaries with `extrapolate=False` returns NaN. The code evaluates slightly inside boundaries (epsilon = 0.1Å) to avoid this issue.

### Gap vs Overlap Detection
The code automatically detects whether multiple files have:
- **Gap**: `wave1_max < wave2_min` → Linear interpolation
- **Overlap**: `wave1_max ≥ wave2_min` → Averaging

### Smoothing Performance
Smoothing pre-computes all grids once, making subsequent `get_fwhm()` calls very fast. For 960 fibers:
- Without pre-computation: ~30-60 seconds for full diagnostic plots
- With pre-computation: ~1-2 seconds

### Missing Fibers
Missing NSPEC values are automatically filled by copying from the nearest available fiber. Copied fibers are flagged with `'is_copy': True` and `'copied_from': <source_nspec>`.

### Pickle Serialization (NEW)
- Uses `dill` for robust serialization (handles closures and lambda functions)
- Falls back to `pickle` if `dill` is not installed
- File extension is `.dill` when using dill, `.pkl` when using pickle
- Pickle files are portable across sessions but may not be portable across major code versions

---

## 🐛 Error Handling

The code performs extensive validation:

### File Validation
- File existence check
- File vs directory check
- Read permission check
- FITS extension verification
- File size sanity check

### Data Validation
- HDU structure verification
- Required columns check
- Non-empty data check
- NaN value detection
- Valid wavelength ranges

### Pickle Validation (NEW)
- File existence check with helpful error message
- Corruption detection
- Version compatibility warnings

### Example Errors

```python
# File not found
FileNotFoundError: LSF file not found: <PYAPS_DATA>/missing.fits
Please check the file path and ensure the file exists.

# Not a file
ValueError: Path exists but is not a file: <PYAPS_DATA>/LSF/
Please provide a path to a FITS file, not a directory.

# Missing columns
ValueError: FITS table missing required columns: ['t', 'c']
Available columns: ['NSPEC', 'k']

# No data
ValueError: LSF file contains no fiber data: <PYAPS_DATA>/empty.fits

# Pickle not found (NEW)
FileNotFoundError: LSF interpolator pickle file not found: <PYAPS_DATA>/output/missing.dill
Please run run_lsf_analysis() first to create the interpolator,
or check that the file path is correct.

# Corrupted pickle (NEW)
IOError: Failed to load LSF interpolator from: <PYAPS_DATA>/output/corrupted.dill
Error: unpickling error...
The file may be corrupted or incompatible with current code version.
```

---

## 📞 Support

For issues or questions:
1. Check this README
2. Enable `debug=True` for detailed output
3. Check diagnostic plots for data quality issues
4. Review error messages for specific problems

---

## 📝 Version History

- **v1.2** (2026-08-05): Vectorized interpolation — the "Fast interpolation
  using stored knots and coefficients" claim above wasn't actually true
  until this release. All three `interpolate_function` closures this
  module builds (the per-fibre one, the *global* master one, and the
  merged/join-arms one) evaluated their query wavelength grid one point at
  a time in a plain Python `for` loop, each point going through its own
  scipy/Python call overhead — harmless on the small fibre counts every
  dataset happened to be tested against up to this point, but on a real
  full LIFU stackcube (164×188 = 30,832 spaxels) the *global* interpolator
  specifically (the one every IFU/LIFU/MIFU L1 target actually gets
  assigned — see `aps_utils.py`'s `_assign_arm_results_to_targets`) turned
  a routine FWHM-diagnostics cache build into a genuine ~220-second stall,
  found while investigating a real server crash report. All three closures
  now call the underlying spline/lookup once on the full array (`np.where`
  for the branch logic scipy's own vectorized call can't express directly)
  — confirmed bit-identical output against the old per-point loops
  (including every branch's exact boundary condition) before landing,
  ~20-35x faster in isolation, and confirmed live end-to-end on the real
  stackcube: the FWHM cache build that used to take ~220s now takes ~4s.
  See `tests/test_l1_fwhm_lsf.py` for the full verification suite.

- **v1.1** (December 2024): Pickle support
  - Added `dill` pickle save/load functionality
  - Added `overwrite` parameter to skip recomputation
  - Added `save_pickle` and `pickle_dir` parameters
  - Added `generate_output_headname()` for consistent naming
  - Added `get_output_pickle_path()` helper function
  - Added `save()` and `load()` methods to LSFInterpolator
  - Added `load_lsf_interpolator()` convenience function
  - Consistent filenames between pickle and figure outputs
  - Improved error messages for file not found

- **v1.0** (December 2024): Initial release
  - Pre-computed B-spline reader
  - Multi-file merging with gap/overlap handling
  - Smoothing support
  - Missing fiber filling
  - Comprehensive diagnostic plots

---

## 🎓 Citation

If you use this code in your research, please cite the WEAVE project and acknowledge the PyAPS pipeline.

---

## 📄 License

Part of the PyAPS (Python Automated Pipeline System) for WEAVE spectroscopic data processing.
