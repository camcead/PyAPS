# FWHM Interpolator (`aps_fwhm.py`)

## Overview

`aps_fwhm.py` provides FWHM (Full Width at Half Maximum) interpolation from CAL/Wave FITS files for the WEAVE spectroscopic pipeline. It creates smooth, continuous FWHM profiles across wavelength for both single-arm and dual-arm observations.

The module is designed to have an **identical API** to `aps_lsf.py`, allowing interchangeable use in the pipeline.

## Key Features

- **Per-fiber spline fitting**: Individual UnivariateSpline for each fiber
- **Global master**: Median-stacked profile across all fibers
- **Multi-arm support**: Seamless handling of blue+red arm combinations
- **Linear transition**: Gap OR overlap between arms handled with linear interpolation
- **Flat extrapolation**: Constant values outside wavelength coverage
- **Per-fiber wavelength ranges**: Each fiber uses its OWN range (not global)
- **Pickle caching**: Save/load for fast re-use
- **Diagnostic plots**: 4-panel visualization of all interpolation regions

## Interpolation Strategy

The interpolator uses a **region-based** approach that treats each fiber individually:

```
                FLAT          SPLINE         LINEAR        SPLINE          FLAT
               (arm1         (arm1)       (transition)    (arm2)         (arm2
               first)                                                     last)
                 │              │              │             │              │
    ─────────────┼──────────────┼──────────────┼─────────────┼──────────────┼─────────────
                 │              │              │             │              │
              arm1_min      arm1_max       arm2_min      arm2_max
                 │              │              │             │
                 └──── ARM 1 ───┘              └──── ARM 2 ──┘
                                └── GAP/OVERLAP ─┘
```

### Regions (Per-Fiber)

| Region | Condition | Method | Description |
|--------|-----------|--------|-------------|
| 0 | λ < fiber_arm1_min | **FLAT** | Constant at fiber's first FWHM value |
| 1 | fiber_arm1_min ≤ λ ≤ fiber_arm1_max | **SPLINE** | Cubic spline from arm 1 data |
| 2 | fiber_arm1_max < λ < fiber_arm2_min | **LINEAR** | Linear connection between arms |
| 3 | fiber_arm2_min ≤ λ ≤ fiber_arm2_max | **SPLINE** | Cubic spline from arm 2 data |
| 4 | λ > fiber_arm2_max | **FLAT** | Constant at fiber's last FWHM value |

**CRITICAL**: Both **gap** (arms don't touch) and **overlap** (arms overlap) are handled the **same way** - with LINEAR interpolation from end of arm1 to start of arm2. This preserves the spline shape within each arm.

### Per-Fiber vs Global Ranges

Each fiber has its **own wavelength coverage** that may differ from other fibers:

```
Fiber 123: Blue 3755.2 - 5848.3 Å | Red 6053.1 - 9897.5 Å
Fiber 456: Blue 3752.8 - 5851.7 Å | Red 6049.8 - 9901.2 Å
Fiber 789: Blue 3758.1 - 5845.9 Å | Red 6055.7 - 9894.3 Å
```

The interpolator uses **per-fiber ranges** for transitions, ensuring no zero-value artifacts in regions where some fibers have data and others don't.

## Installation

The module requires:
```python
numpy
scipy (UnivariateSpline)
astropy (FITS I/O)
matplotlib (plotting)
dill (pickle serialization, falls back to pickle)
```

## Quick Start

### Single Arm
```python
from PyAPS.aps_fwhm import run_fwhm_analysis

interp = run_fwhm_analysis(
    file_paths="/path/to/wave_3100962_all.fit",
    figdir="/output/directory",
    figname="fwhm_blue",
    debug=True
)

# Get FWHM at wavelengths
fwhm = interp.get_fwhm([4500.0, 5000.0, 5500.0])
```

### Two Arms (Blue + Red)
```python
from PyAPS.aps_fwhm import run_fwhm_analysis

interp = run_fwhm_analysis(
    file_paths=[
        "/path/to/wave_3100962_all.fit",  # Blue arm
        "/path/to/wave_3100961_all.fit"   # Red arm
    ],
    figdir="/output/directory",
    figname="fwhm_combined",
    debug=True
)

# Get FWHM across full range
fwhm = interp.get_fwhm([4500.0, 6000.0, 7500.0, 9000.0])
```

## API Reference

### Main Function

```python
def run_fwhm_analysis(
    file_paths,                      # str or list: wave FITS file(s)
    figdir=None,                     # Output directory for plots
    figname="analysis",              # Base name for plots
    debug=False,                     # Print detailed debug info
    make_plot=True,                  # Create diagnostic plots
    wave_bin_width=50,               # Outlier detection bin width (Å)
    apply_bimodal_filtering=False,   # GMM bimodal filtering
    apply_residual_filtering=True,   # Residual outlier filtering
    spline_order=3,                  # Spline order (1-5, 3=cubic)
    spline_smoothing=None,           # None=auto, 0=interpolating
    wave_grid_resolution=1.0,        # Grid resolution (Å)
    overwrite=False,                 # Force recreate vs load cache
    save_pickle=True,                # Save to pickle file
    pickle_dir=None                  # Pickle directory (None=figdir)
) -> FWHMInterpolator
```

### FWHMInterpolator Class

#### Methods

| Method | Description |
|--------|-------------|
| `get_fwhm(wavelength, specnum=None)` | Get FWHM at wavelength(s). Use `specnum` for fiber-specific. |
| `get_interpolator_dict()` | Get dict of all interpolators for pipeline integration |
| `get_available_fibers()` | List available fiber specnums |
| `print_summary()` | Print interpolator summary |
| `save(filepath=None)` | Save to pickle (auto-generates path if None) |
| `load(filepath)` | Class method to load from pickle |

#### Example Usage

```python
# Get global FWHM (median of all fibers)
fwhm_global = interp.get_fwhm(5000.0)

# Get fiber-specific FWHM
fwhm_fiber = interp.get_fwhm(5000.0, specnum=123)

# Get FWHM at multiple wavelengths
fwhm_array = interp.get_fwhm([4000, 5000, 6000, 7000, 8000])

# Get interpolator dict for pipeline
fwhm_dict = interp.get_interpolator_dict()
# fwhm_dict['global'] -> global interpolator
# fwhm_dict[123] -> fiber 123 interpolator
```

### Convenience Functions

```python
from PyAPS.aps_fwhm import load_fwhm_interpolator, get_output_pickle_path

# Load existing interpolator
interp = load_fwhm_interpolator("/path/to/interpolator.dill")

# Get expected pickle path for files
pickle_path = get_output_pickle_path(
    ["/path/to/blue.fit", "/path/to/red.fit"],
    output_dir="/output/dir"
)
# Returns: "/output/dir/wave_3100962_all__wave_3100961_all.dill"
```

## Pipeline Integration

### In `aps_utils.py`

```python
from PyAPS.aps_fwhm import run_fwhm_analysis

# Create or load FWHM interpolator
fwhm_interp = run_fwhm_analysis(
    self.calfiles,
    figdir=self.debugdir,
    figname="FWHM",
    debug=False,
    make_plot=(self.debugdir is not None),
    overwrite=False,      # Use cached if available
    save_pickle=True
)

if fwhm_interp is not None:
    fwhm_dict = fwhm_interp.get_interpolator_dict()

    for targ in self._targetlist:
        specnum = targ.specnum

        # Use fiber-specific if available, otherwise global
        if specnum in fwhm_dict:
            targ.meta[setup_idx]['fwhm'] = fwhm_dict[specnum]
        else:
            targ.meta[setup_idx]['fwhm'] = fwhm_dict['global']
```

### Using the Interpolator Dict

```python
fwhm_dict = interp.get_interpolator_dict()

# Structure:
# {
#     'global': {
#         'interpolate_function': <callable>,
#         'type': 'global_fwhm',
#         'wavelength_range': (3750.0, 9900.0),
#         'n_fibers': 960
#     },
#     123: {
#         'interpolate_function': <callable>,
#         'type': 'fiber_fwhm',
#         'specnum': 123,
#         'wavelength_range': (3755.2, 9897.5),
#         'n_points': 8192,
#         'rms': 0.0023
#     },
#     ...
# }

# Get FWHM using interpolate_function
wavelengths = np.array([5000.0, 6000.0, 7000.0])
fwhm_values = fwhm_dict['global']['interpolate_function'](wavelengths)
```

## Parameters Guide

### Filtering Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `wave_bin_width` | 50 | Wavelength bin width (Å) for outlier detection. Smaller = more local. |
| `apply_bimodal_filtering` | False | Use GMM to detect bimodal distributions. Usually not needed. |
| `apply_residual_filtering` | True | Remove outliers based on spline fit residuals. Recommended. |

### Spline Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `spline_order` | 3 | Spline degree (k). 3=cubic is recommended for smooth C² continuous fits. |
| `spline_smoothing` | None | Smoothing factor. None=auto (recommended), 0=interpolating, >0=smoothing. |
| `wave_grid_resolution` | 1.0 | Grid spacing (Å) for global master evaluation. |

### Spline Order Guide

| Order | Continuity | Use Case |
|-------|------------|----------|
| k=1 | C⁰ | Piecewise linear, has corners |
| k=2 | C¹ | Quadratic, smooth first derivative |
| k=3 | C² | **Cubic (recommended)**, smooth second derivative |
| k=4 | C³ | Quartic, very smooth but can oscillate |
| k=5 | C⁴ | Quintic, prone to Runge's phenomenon |

### Caching Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `overwrite` | False | If True, always recreate. If False, load from pickle if exists. |
| `save_pickle` | True | Save interpolator to pickle file for future use. |
| `pickle_dir` | None | Directory for pickle. None = use figdir or input file directory. |

## Diagnostic Plots

The module generates a 4-panel diagnostic plot:

### Panel 1: Full Interpolation - All Regions
Shows the complete interpolation with color-coded regions:
- **Blue dashed**: Flat extrapolation before arm 1
- **Cyan solid**: Arm 1 spline region
- **Gold dashed**: Transition (gap/overlap) - LINEAR
- **Orange solid**: Arm 2 spline region
- **Red dashed**: Flat extrapolation after arm 2
- Shaded regions indicate extrapolation zones
- Vertical lines mark arm boundaries

### Panel 2: Fiber Splines + Global Master
- Gray lines: Individual fiber splines (sample)
- Red solid: Global master (median)
- Blue dashed: Median stack
- Shaded region: Gap/overlap zone

### Panel 3: Fiber-to-Fiber Variation
- Boxplot showing FWHM distribution across fibers at each wavelength
- Red points: Global master values

### Panel 4: Interpolation Strategy Summary
- Color-coded line showing which method is used at each wavelength
- Region labels (FLAT, ARM 1 SPLINE, LINEAR, ARM 2 SPLINE, FLAT)

## File Naming Convention

### Output Files

| Input | Output Name |
|-------|-------------|
| Single file: `wave_3100962_all.fit` | `wave_3100962_all` |
| Two files: `wave_3100962_all.fit` + `wave_3100961_all.fit` | `wave_3100962_all__wave_3100961_all` |

### Generated Files

```
output_directory/
├── wave_3100962_all__wave_3100961_all_fwhm_combined.png   # Diagnostic plot
├── wave_3100962_all__wave_3100961_all.dill                # Pickle cache
```

## Technical Details

### Data Source

FWHM values are extracted from CAL/Wave FITS files:
- Extension: `FWHMCOEFFS` (primary) or `MEANFWHM` (fallback)
- Contains wavelength vs FWHM measurements per fiber

### Outlier Filtering

1. **Wavelength binning**: Data divided into bins of `wave_bin_width`
2. **3-sigma clipping**: Per-bin outlier removal
3. **Bimodal filtering** (optional): GMM-based detection
4. **Residual filtering**: Post-spline-fit outlier removal

### Global Master Creation

For multi-arm data:
1. Create global spline for each arm separately (median of all fibers)
2. Evaluate each fiber only within ITS wavelength range (no extrapolation)
3. Connect arm globals with LINEAR transition
4. Add FLAT extrapolation at boundaries

### Comparison with aps_lsf.py

| Feature | aps_fwhm | aps_lsf |
|---------|----------|---------|
| Input files | wave_*.fit (CAL) | lsf_*.fits |
| Fitting method | UnivariateSpline | Pre-computed BSpline |
| Gap handling | LINEAR | LINEAR |
| Overlap handling | LINEAR | LINEAR |
| Edge handling | FLAT | FLAT |
| Per-fiber ranges | ✓ | ✓ |
| Pickle support | ✓ (.dill) | ✓ (.dill) |
| API compatible | ✓ | ✓ |

## Troubleshooting

### Zero Values in Transition Region

**Cause**: Using global ranges instead of per-fiber ranges.

**Solution**: The current version uses per-fiber ranges. If you see zeros, ensure you're using the latest version.

### NaN Values

**Cause**: Spline evaluation outside valid range.

**Solution**: The interpolator uses flat extrapolation outside each fiber's range. Check that input data is valid.

### Pickle Load Fails

**Cause**: Pickle created with different Python/dill version.

**Solution**: Delete pickle and recreate with `overwrite=True`.

### Too Few Points Error

**Cause**: Insufficient valid data points after filtering.

**Solution**:
- Reduce `wave_bin_width` for less aggressive filtering
- Set `apply_residual_filtering=False`
- Check input data quality

## Example Output

```
FWHM INTERPOLATOR SUMMARY:
  Source: CAL/Wave FITS files
  Individual fiber splines: 948
  Global master: Per-arm medians with linear transition
  Wavelength range: 3750.2 - 9901.5 Å
  Files processed: 2
  Arm 1 range: 3750.2 - 5852.3 Å → SPLINE
  Arm 2 range: 6048.7 - 9901.5 Å → SPLINE
  Gap: 5852.3 - 6048.7 Å → LINEAR
  Extrapolation: FLAT (constant at boundary values)
```

## Version History

- **v2.0**: Complete refactor to match aps_lsf.py API
  - Per-fiber wavelength ranges (fixes zero-value bug)
  - LINEAR transition for both gap and overlap
  - Per-arm global creation
  - Enhanced diagnostic plots

- **v1.0**: Initial implementation
  - Basic spline fitting
  - Global ranges only

## Author

WEAVE/PyAPS Development Team
