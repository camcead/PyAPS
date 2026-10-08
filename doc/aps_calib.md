# APS calibration (LSF and FWHM)  Modules — Master Reference Guide

## Line Spread Function Interpolation for the WEAVE PyAPS Pipeline

**Modules:** `PyAPS.aps_lsf`, `PyAPS.aps_fwhm`
**Classes:** `LSFInterpolator`, `FWHMInterpolator`
**Used by:** `APSOB` (via `aps_utils`), all IFU and MOS analysis modules
**Default mode:** `aps_lsf` (pre-computed B-spline LSF from solar twilight flats)
**Maintained by:** CASU, Institute of Astronomy, University of Cambridge

---

## Table of Contents

1. [Background — What Is the LSF and Why Does It Matter](#1-background--what-is-the-lsf-and-why-does-it-matter)
2. [WEAVE Calibration Files — Data Sources](#2-weave-calibration-files--data-sources)
3. [Two Routes to an LSF Interpolator](#3-two-routes-to-an-lsf-interpolator)
4. [When to Use Which Module](#4-when-to-use-which-module)
5. [WEAVE LSF FITS File Format (aps_lsf input)](#5-weave-lsf-fits-file-format-aps_lsf-input)
6. [WEAVE Wave FITS File Format (aps_fwhm input)](#6-weave-wave-fits-file-format-aps_fwhm-input)
7. [File Naming Conventions](#7-file-naming-conventions)
8. [Interpolation Strategy — Common Design](#8-interpolation-strategy--common-design)
9. [aps_lsf — LSFInterpolator in Detail](#9-aps_lsf--lsfinterpolator-in-detail)
10. [aps_fwhm — FWHMInterpolator in Detail](#10-aps_fwhm--fwhminterpolator-in-detail)
11. [Key Differences Between the Two Modules](#11-key-differences-between-the-two-modules)
12. [Integration with APSOB and aps_utils](#12-integration-with-apsob-and-aps_utils)
13. [The Interpolator Dictionary — Pipeline Interface](#13-the-interpolator-dictionary--pipeline-interface)
14. [Running from Python — Complete Examples](#14-running-from-python--complete-examples)
15. [Output File Naming and Caching](#15-output-file-naming-and-caching)
16. [Smoothing (aps_lsf only)](#16-smoothing-aps_lsf-only)
17. [Missing Fiber Filling (aps_lsf only)](#17-missing-fiber-filling-aps_lsf-only)
18. [Diagnostic Plots](#18-diagnostic-plots)
19. [Complete API Reference](#19-complete-api-reference)
20. [Troubleshooting](#20-troubleshooting)

---

## 1. Background — What Is the LSF and Why Does It Matter

The Line Spread Function (LSF) of a spectrograph describes how a monochromatic point source in the input spectrum is spread across detector pixels in the output spectrum. In WEAVE, the LSF:

- **Varies with wavelength** — the optical system does not have uniform resolution across the full 3700–9500 Å range
- **Varies with fiber position** (NSPEC) — fibers at different positions on the slit see slightly different optical paths
- **Differs between instrument setups** — each combination of arm (BLUE/GREEN/RED), resolution (L/H), binning, and mode (MOS-A/B, LIFU, mIFU) has its own LSF

For the analysis modules (`pPXF`, `FERRE`, `RVS`), knowing the LSF is essential because:

- **pPXF** convolves the SSP template library to match the data resolution before fitting stellar kinematics. If the convolution is wrong, the velocity dispersion measurements are biased.
- **FERRE** fits synthetic stellar spectra that must be degraded from their native resolution to the observation resolution at each wavelength.
- **RVS** cross-correlates against templates at the correct resolution for accurate radial velocities.

In practice, the LSF is represented as a FWHM (Full Width at Half Maximum) value in Angstroms at each wavelength for each fiber. The two modules in this document provide that FWHM as a callable function over wavelength.

---

## 2. WEAVE Calibration Files — Data Sources

WEAVE provides two types of calibration files that can be used to derive the LSF. Understanding the difference is important for choosing the right input to aps_lsf vs aps_fwhm.

### 2.1 Solar Twilight Flats — the preferred LSF source

Solar twilight flats are taken by pointing the telescope at the twilight sky, which reflects sunlight and therefore carries the solar spectrum. They are not taken pointing directly at the Sun. The solar absorption lines are unresolved by WEAVE (their intrinsic widths are well below the WEAVE spectral resolution), making them ideal tracers of the instrumental LSF. By comparing the WEAVE solar spectrum to a very high-resolution solar reference spectrum (e.g. from the FTS at Kitt Peak or the HARPS solar atlas), the exact LSF at each wavelength and fiber position can be derived.

**Why these are preferred:**

> *"While an LSF can indeed be determined from warc files, and associated FWHM values are listed in the wave files, caution should be taken as light from the calibration lamps has a different focal ratio (f/7) compared to on-sky observations (f/3). Preference should be given to the LSF determined from the solar twilight spectra."*

The key issue is focal ratio degradation. Arc lamps illuminate the fiber at f/7 while the telescope feeds the fiber at f/3. This changes the far-field illumination pattern of the fiber output, which in turn changes the LSF shape at the spectrograph. The solar twilight flats are on-sky observations and therefore correctly represent the as-observed LSF.

**File names:** `lsf_<ARM><RES><BIN>_<MODE>.fits`

Example: `lsf_BLUEL11_LIFU.fits`, `lsf_REDH11_MOS-A.fits`

**Used by:** `aps_lsf.py` — this is the **default APS mode**

### 2.2 Arc Lamp Wave Files — the fallback source

Arc lamp exposures (`warc` files) are taken as standard calibrations each night. The `wave` files derived from them contain per-fiber FWHM measurements at each arc lamp line wavelength, covering the full wavelength range in discrete steps. These can be used to derive a smooth FWHM interpolation as a fallback when solar twilight LSFs are not available.

In practice, APS uses **master wave files** derived from a stack of arc exposures over a calibration period, rather than individual night-by-night files. This improves S/N on the FWHM measurements and produces more stable spline fits.

**File names:** `wave_<RUN>.fits` or `wave_<RUN>_all.fits` (for MOS)

Example: `wave_3100962_all.fit`

**Used by:** `aps_fwhm.py` — the fallback/arc-based route

### 2.3 MOS Calibration Note

For MOS observations, calibration data is taken in `SALSA` mode where only 1 in 3 fibres is illuminated per exposure. Three observations are combined to cover all fibres. The provided calibration files (`warc`, `wave`, `solar`, `lsf`, `pcross`) already include this merge, indicated by the `_all` suffix on MOS files.

For MOS there are **separate calibrations for plate A and plate B** (`MOS-A` and `MOS-B`).

---

## 3. Two Routes to an LSF Interpolator

```
WEAVE observation
      │
      ├─── Solar twilight flat ──► lsf_*.fits ──► aps_lsf.py
      │    (preferred: on-sky f/3)                    │
      │                                               │
      └─── Arc lamp exposure ────► wave_*.fits ──► aps_fwhm.py
           (fallback: f/7, focal                      │
            ratio mismatch)                           │
                                                      │
                                          Both produce identical API:
                                          interpolator.get_fwhm(λ, specnum)
                                          interpolator.get_interpolator_dict()
```

Both modules produce an object with identical public methods so they can be used interchangeably in the pipeline. The choice of which to use is made once at APSOB construction time (see Section 12).

---

## 4. When to Use Which Module

| Situation | Module | Reason |
|---|---|---|
| Normal APS production run | `aps_lsf` | Default — uses on-sky solar twilight LSF |
| Solar twilight LSF file available | `aps_lsf` | Always preferred |
| No solar twilight file for this setup/date | `aps_fwhm` | Fallback to arc lamp FWHM |
| Testing pipeline with arc calibrations only | `aps_fwhm` | Fine for development |

In the batch job scripts, `aps_lsf` is used by default. If the `lsf_*.fits` file for a given setup is missing from the calibration directory, APSOB falls back to `aps_fwhm` with the corresponding `wave_*.fits` file.

---

## 5. WEAVE LSF FITS File Format (aps_lsf input)

`aps_lsf` reads pre-computed B-spline representations of the LSF derived from solar twilight flats. The CASU calibration team produces one file per instrumental setup per calibration cycle.

### 5.1 File naming

```
lsf_<ARM><RES><BIN>_<MODE>.fits
```

| Field | Values | Example |
|---|---|---|
| `ARM` | `BLUE`, `GREEN`, `RED` | `BLUE`, `RED` |
| `RES` | `L` (low), `H` (high) | `L`, `H` |
| `BIN` | `11` (unbinned, fixed) | `11` |
| `MODE` | `MOS-A`, `MOS-B`, `LIFU`, `mIFU` | `LIFU` |

Full examples: `lsf_BLUEL11_LIFU.fits`, `lsf_REDH11_MOS-A.fits`, `lsf_GREENH11_mIFU.fits`

### 5.2 HDU structure

**HDU 0 — Primary header** (metadata only, no data)

| Key | Description |
|---|---|
| `CALDATE` | Calibration date |
| `L1_REF` | L1 reference file used |
| `FPMODE` | Focal plane mode |
| `CAMERA` | Camera name (BLUE/RED/GREEN) |
| `VPH` | VPH grating identifier |
| `CCDXBIN` | CCD binning factor |
| `LSFDATE` | LSF creation date |

The header also names the parent solar twilight files from which this LSF was derived.

**HDU 1 — Binary table `LSF_splines`**

Required columns:

| Column | Type | Description |
|---|---|---|
| `NSPEC` | int | Fiber spectrum number (1–960 for WEAVE) |
| `t` | float array (12 elements) | B-spline knot vector |
| `c` | float array (12 elements) | B-spline coefficients |
| `k` | int | B-spline degree (typically 5) |

The knot vector `t` and coefficient array `c` together with degree `k` fully define a `scipy.interpolate.BSpline` object: `BSpline(t, c, k, extrapolate=False)`. These represent the FWHM in Angstroms as a function of observed wavelength for that fiber.

### 5.3 Binning fallback

If a binned LSF file is requested (e.g. `lsf_BLUEL21_LIFU.fits`, binning `21`) but does not exist, `check_and_replace_binned_files()` automatically substitutes the unbinned version (`lsf_BLUEL11_LIFU.fits`). A warning is printed to stderr. This is enabled via the `replace_binned=True` flag in `run_lsf_analysis()`.

---

## 6. WEAVE Wave FITS File Format (aps_fwhm input)

`aps_fwhm` reads `wave_*.fits` files produced by the CPS L1 pipeline from arc lamp exposures.

### 6.1 File naming

```
wave_<RUN>.fits          (for IFU modes)
wave_<RUN>_all.fits      (for MOS, merged from three SALSA settings)
```

`RUN` is the run number of the `warc` file from which this `wave` file was derived. For MOS, the run number is that of the first SALSA setting.

### 6.2 Binary table columns

| Column | Type | Description |
|---|---|---|
| `NSPEC` | int | Fiber spectrum number |
| `fiblive` | int | Fiber live flag (1=active, 0=dead) |
| `ngood` | int | Number of good arc lines measured |
| `fwhm` | float array | FWHM in pixels at each arc line |
| `wave_true` | float array | True wavelength of each arc line (Å) |
| `wave_calc` | float array | Calculated wavelength of each arc line (Å) |
| `fit_flag` | int array | Quality flag per arc line (0=good) |
| `medresid` | float | Median residual of wavelength fit |
| `fit_rms` | float | RMS of wavelength fit |

Only rows with `fiblive=1` and `ngood>0` are used. Only measurements with `fit_flag=0` are included in spline fitting.

---

## 7. File Naming Conventions

Both modules use the same naming convention for their output pickle files and diagnostic figures.

### 7.1 Output headname generation

The headname is derived from the input file names:

**Single file:**

| Input | Headname |
|---|---|
| `lsf_GREENH11_MOS-A.fits` | `GREENH11_MOS-A` |
| `lsf_BLUEH11_LIFU.fits` | `BLUEH11_LIFU` |
| `wave_3100962_all.fit` | `wave_3100962_all` |

**Multiple files (double underscore separator):**

| Input files | Headname |
|---|---|
| `lsf_GREENH11_MOS-A.fits` + `lsf_REDH11_MOS-A.fits` | `GREENH11__REDH11_MOS-A` |
| `lsf_BLUEH11_LIFU.fits` + `lsf_REDH11_LIFU.fits` | `BLUEH11__REDH11_LIFU` |
| `wave_3100962_all.fit` + `wave_3100961_all.fit` | `wave_3100962_all__wave_3100961_all` |

### 7.2 Generated files

For headname `GREENH11__REDH11_MOS-A` with `figname="analysis"`:

| File | Description |
|---|---|
| `GREENH11__REDH11_MOS-A.dill` | Pickle cache of the interpolator |
| `GREENH11__REDH11_MOS-A_analysis.png` | 4-panel diagnostic figure |

The pickle extension is `.dill` when the `dill` package is available (preferred), `.pkl` otherwise.

---

## 8. Interpolation Strategy — Common Design

Both modules implement the same region-based interpolation strategy, which is critical to understand before using the interpolator output in pipeline code.

### 8.1 The five regions

For a two-arm observation (e.g. WEAVE blue + red):

```
        ◄ FLAT ►◄─────── ARM 1 SPLINE ──────────►◄ LINEAR ►◄─────── ARM 2 SPLINE ──────────►◄ FLAT ►
        │        │                                │          │                                │        │
  < arm1_min   arm1_min                      arm1_max   arm2_min                       arm2_max  > arm2_max
```

| Region | Wavelength | Method | Description |
|---|---|---|---|
| 0 | λ < arm1_min | **FLAT** | Constant at arm 1 first value |
| 1 | arm1_min ≤ λ ≤ arm1_max | **SPLINE** | B-spline (aps_lsf) or UnivariateSpline (aps_fwhm) |
| 2 | arm1_max < λ < arm2_min | **LINEAR** | Linear interpolation between arm boundary values |
| 3 | arm2_min ≤ λ ≤ arm2_max | **SPLINE** | As above |
| 4 | λ > arm2_max | **FLAT** | Constant at arm 2 last value |

### 8.2 Gap vs overlap — same treatment

Whether the two arms have a gap (arm1_max < arm2_min, common for WEAVE blue+red with no overlap) or an overlap (arm1_max > arm2_min), the transition region is always treated with **linear interpolation** connecting the boundary value at the end of arm 1 to the boundary value at the start of arm 2.

This design choice preserves the spline shape within each arm — processing single-arm and double-arm data produces identical results within each arm's coverage.

**aps_lsf note:** In the overlap case the two arms are averaged in the overlap region; the linear interpolation applies specifically to the gap case.

**aps_fwhm note:** Both gap and overlap use linear interpolation from arm1_max to arm2_min regardless.

### 8.3 Per-fiber wavelength ranges

Each fiber has its own wavelength coverage that differs slightly from other fibers (due to the fiber-to-fiber variation in the detector projection). Both modules use **per-fiber wavelength ranges** for the boundary values and transition region, not the global file-level range. This prevents zero-value artifacts in regions where some fibers have data and others do not.

### 8.4 Boundary value evaluation

For `aps_lsf`, the B-spline has `extrapolate=False` which returns NaN exactly at the boundary knots. The boundary values are evaluated at `knot_min + ε` and `knot_max - ε` (ε = 0.1 Å) to avoid this. If that still produces NaN, the code evaluates at 10 test points and uses the first/last valid values.

---

## 9. aps_lsf — LSFInterpolator in Detail

### 9.1 Processing pipeline

```
read_lsf_splines_file()
    │  Read FITS HDU 1 → BSpline objects per fiber
    │
    ↓
LSFInterpolator.create_from_files()
    │
    ├── Single file:
    │       fiber_splines = {nspec: BSpline, ...}
    │
    └── Multiple files:
            _merge_multiple_files()
                │  For each fiber: evaluate both arms on shared grid
                │  Apply 6-region merging per fiber
                │  Create per-fiber closure interpolation function
                │
                ↓
            _create_global_spline()
                │  Evaluate all fibers on wave_grid
                │  Median across all fibers → global_fwhm
                │  Handle gap with linear interpolation
                │  Fit UnivariateSpline to global_fwhm
                │
                ↓
            _create_interpolator_dict()
                │  Wrap global spline with flat extrapolation + gap linear
                │  Wrap each fiber spline similarly
                │
                ↓
            _fill_missing_fibers()
                │  Find NSPECs in [1, max_nspec] not present
                │  Copy nearest available fiber's function
                │
                ↓
            apply_smoothing() [optional]
                │  Pre-compute smoothed grid for every fiber
                │  Replace interpolation functions with fast numpy.interp
                │
                ↓
            save() → .dill pickle file
```

### 9.2 The `LSFInterpolator` class

```python
from PyAPS.aps_lsf import LSFInterpolator, run_lsf_analysis

# --- Creation ---
interpolator = run_lsf_analysis(
    file_input    = ["<PYAPS_DATA>/CAL/lsf_BLUEH11_LIFU.fits",
                     "<PYAPS_DATA>/CAL/lsf_REDH11_LIFU.fits"],
    figdir        = "<PYAPS_DATA>/L2/figs",
    figname       = "lsf_diagnostic",
    debug         = False,
    make_plot     = True,
    smooth_length = None,        # Å — None means no smoothing
    kernel_type   = "gaussian",  # 'boxcar', 'gaussian', 'hanning'
    overwrite     = False,       # Load from pickle if exists
    save_pickle   = True,
    pickle_dir    = None,        # None → use figdir
    replace_binned= True,        # Auto-substitute unbinned if binned missing
)
```

### 9.3 Smoothing

Optional post-creation smoothing modifies all fiber interpolation functions to smooth out noise in the B-spline coefficients:

```python
# Apply 10 Å Gaussian smoothing
interpolator.apply_smoothing(smooth_length=10.0, kernel_type="gaussian")

# Restore originals
interpolator.remove_smoothing()
```

Kernel types: `"boxcar"` (moving average), `"gaussian"` (sigma = kernel_size/6), `"hanning"` (smooth edges).

Performance: the first call pre-computes all smoothed grids (once) and replaces all interpolation functions with fast `numpy.interp` lookups. Subsequent `get_fwhm()` calls are very fast.

### 9.4 Missing fiber filling

After creating the interpolator dict, `_fill_missing_fibers()` finds NSPEC values not present in the LSF file (dead fibres, edge fibres, SALSA gaps) and copies the nearest available fiber's interpolation function. The copy is flagged:

```python
interp_dict[933] = {
    "interpolate_function": <same function as nearest fiber>,
    "type": "fiber_lsf_copied",
    "specnum": 933,
    "copied_from": 932,
    "is_copy": True,
    ...
}
```

---

## 10. aps_fwhm — FWHMInterpolator in Detail

### 10.1 Processing pipeline

```
read_fits_and_filter()  [per file]
    │  Read FITS table, select fiblive=1, fit_flag=0
    │  Wavelength-binned 3-sigma outlier rejection
    │  Optional: bimodal GMM filtering
    │
    ↓
fit_individual_fiber_splines()  [per file]
    │  For each fiber: collect arc line FWHM measurements
    │  5-MAD global outlier rejection
    │  Handle duplicate wavelengths (median)
    │  Fit UnivariateSpline with auto-smoothing
    │  Optional: residual-based outlier rejection + refit
    │
    ↓ (multi-arm only)
_merge_fiber_splines()
    │  Per-fiber: create closure using per-fiber arm ranges
    │  5 regions: flat / spline1 / linear / spline2 / flat
    │
    ↓
create_global_per_arm()  [multi-arm]
    │  Per arm: evaluate each fiber ONLY within its own range
    │  Median stack → arm_global spline
    │  Combine with linear transition between arms
    │
    ↓
_create_interpolator_dict()
    │  Package all interpolation functions
    │
    ↓
save() → .dill pickle file
```

### 10.2 Key differences from aps_lsf

`aps_fwhm` fits splines to discrete arc line measurements rather than reading pre-computed coefficients. This means it performs actual fitting at module load time, which takes longer (seconds per arm vs milliseconds for aps_lsf). Once cached in a pickle file, subsequent loads are equally fast.

The auto-smoothing algorithm for individual fibers:

```python
clean_median = np.median(fwhm)
residuals    = fwhm - clean_median
noise_var    = np.var(residuals)
auto_smooth  = len(wavelengths) * noise_var * 0.5
auto_smooth  = max(0.01, min(10.0, auto_smooth))
```

For the global master, a looser smoothing is applied (`noise_var * 0.1`, clamped to `[1.0, 1000.0]`).

---

## 11. Key Differences Between the Two Modules

| Aspect | aps_lsf | aps_fwhm |
|---|---|---|
| **Input files** | `lsf_*.fits` (solar twilight) | `wave_*.fits` (arc lamp) |
| **Focal ratio** | f/3 (on-sky, correct) | f/7 (lamp, biased) |
| **Fitting** | None — reads pre-computed B-spline knots/coeffs | Fits `UnivariateSpline` to arc line FWHM measurements |
| **Fiber spline type** | `scipy.interpolate.BSpline` (degree 5) | `scipy.interpolate.UnivariateSpline` (degree 3, default) |
| **Global master** | Median of all fibers, `UnivariateSpline` fit | Per-arm median stack, `UnivariateSpline` fit |
| **Gap handling** | Linear interpolation (per-fiber boundaries) | Linear interpolation (per-fiber boundaries) |
| **Overlap handling** | Average of both arms | Linear from arm1_max to arm2_min (same as gap) |
| **Smoothing** | Optional post-creation (boxcar/gaussian/hanning) | Built into `UnivariateSpline` smoothing parameter |
| **Missing fibres** | Filled by nearest-neighbor copy | Not filled (only fibers with data are present) |
| **Load time** | Fast (reads pre-computed) | Slower first time (fitting), then fast from pickle |
| **Preferred in APS** | **Yes — default** | Fallback only |
| **Recommended for** | Production runs | Development, missing LSF files |

---

## 12. Integration with APSOB and aps_utils

This is the most important section for pipeline developers. The LSF/FWHM interpolator is loaded inside `APSOB` during construction and attached per-fiber to each target's spectral metadata.

### 12.1 How APSOB loads the LSF

Inside `aps_utils.py`, the `APSOB.__init__` method resolves the calibration files and calls the appropriate module:

```python
# Pseudocode of the LSF loading logic in APSOB

from PyAPS.aps_lsf  import run_lsf_analysis, load_lsf_interpolator
from PyAPS.aps_fwhm import run_fwhm_analysis, load_fwhm_interpolator

# Try to find LSF file(s) for this setup
lsf_files = self._find_lsf_files(configdir, setups)   # looks for lsf_*.fits

if lsf_files:
    # Default APS mode: use solar twilight LSF
    lsf_interp = run_lsf_analysis(
        lsf_files,
        figdir      = self.debugdir,
        figname     = "LSF",
        debug       = False,
        make_plot   = (self.debugdir is not None),
        overwrite   = False,      # Use cached pickle if available
        save_pickle = True,
        pickle_dir  = self.debugdir,
        replace_binned = True,    # Auto-substitute unbinned if needed
    )
else:
    # Fallback: derive FWHM from arc lamp wave files
    wave_files = self._find_wave_files(configdir, setups)
    lsf_interp = run_fwhm_analysis(
        wave_files,
        figdir    = self.debugdir,
        figname   = "FWHM",
        debug     = False,
        make_plot = (self.debugdir is not None),
        overwrite = False,
        save_pickle = True,
    )

# Get the interpolator dictionary
lsf_dict = lsf_interp.get_interpolator_dict() if lsf_interp else None
```

### 12.2 How the LSF is attached to each fiber

After loading, the interpolator dict is used to assign a per-fiber FWHM function to each target in the target list:

```python
# In APSOB, for each arm/setup combination:
for targ in self._targetlist:
    specnum = targ.specnum   # WEAVE fiber spectrum number (1–960)
    setup_idx = ...

    if lsf_dict is not None:
        if specnum in lsf_dict:
            # Use fiber-specific interpolator
            targ.meta[setup_idx]['fwhm'] = lsf_dict[specnum]['interpolate_function']
        else:
            # Fall back to global master
            targ.meta[setup_idx]['fwhm'] = lsf_dict['global']['interpolate_function']
    else:
        targ.meta[setup_idx]['fwhm'] = None
```

### 12.3 How the LSF is exposed from APSOB

After construction, `APSOB.get_fwhm()` returns the per-fiber FWHM interpolators:

```python
APSOBJ_inst = APSOB(infiles, ...)

# Returns list of interpolation functions, one per setup/arm combination
targs_lsf = APSOBJ_inst.get_fwhm(aps_id=None, fwhm_key="gfwhm")

# For joined-arm processing, this is a list with one entry
targs_lsf = targs_lsf[0]   # callable: fwhm_Å = targs_lsf(wavelength_Å)
```

The returned function is called with an observed wavelength array and returns FWHM values in Angstroms at each wavelength:

```python
# Evaluate at the observed wavelength grid
fwhm_array = targs_lsf(observed_wave_array)   # shape: (n_wave,)
```

### 12.4 Use in ifu_ExGal_prepare

In `ifu_ExGal_prepare`, the LSF is used to build the `LSF_Data` array that pPXF requires:

```python
# From ifu_ExGal_prepare:
targs_lsf = APSOBJ_inst.get_fwhm(aps_id=None, fwhm_key="gfwhm")
targs_lsf = targs_lsf[0]   # unpack single-arm result

# Apply redshift correction: convert from observed to rest-frame
LSF_Data = apply_redshift_to_fwhm_corrected(targs_lsf, z_input[0])
```

`apply_redshift_to_fwhm_corrected()` divides both the wavelength axis and the FWHM values by `(1 + z)`, converting the observed-frame LSF to a rest-frame LSF for use by pPXF.

### 12.5 Use in ifu_Gal_prepare

For Gal mode (stellar targets), the LSF is evaluated directly at the observed wavelength grid (no redshift correction needed since z ≈ 0):

```python
# From ifu_Gal_prepare:
lsf_values = targs_lsf[0](gal_targ["wave"])   # shape: (n_wave,)
```

This array is passed directly to RVS and FERRE.

### 12.6 LSF_Templates — the template library LSF

In addition to `LSF_Data` (the data resolution), pPXF also needs `LSF_Templates` (the resolution of the SSP template library). This is loaded separately from a text file in `IFU_config_dir`:

```python
LSF_raw       = np.genfromtxt(
    os.path.join(IFU_config_dir, "LSF-Config_" + configs["SSP_LIB"]),
    comments="#")
LSF_Templates = interp1d(
    LSF_raw[:, 0], LSF_raw[:, 1], "linear", fill_value="extrapolate")
```

The `LSF-Config_MILES` file (for example) contains two columns: wavelength (Å) and FWHM (Å) for the MILES stellar library. The interpolator is passed to pPXF alongside `LSF_Data`.

---

## 13. The Interpolator Dictionary — Pipeline Interface

Both modules produce the same dictionary structure from `get_interpolator_dict()`. This is the primary interface used by downstream pipeline code.

```python
{
    "global": {
        "interpolate_function": <callable>,        # Takes wavelengths → returns FWHM array
        "type": "global_lsf",                      # or "global_fwhm"
        "description": "Global master from median of all fibers",
        "wavelength_range": (wave_min, wave_max),  # float tuple, Å
        "n_fibers": 960                            # number of fibers used
    },

    # One entry per fiber NSPEC (1–960 for WEAVE)
    1: {
        "interpolate_function": <callable>,
        "type": "fiber_lsf",                       # single file
               # "fiber_lsf_merged",               # multi-file (aps_lsf)
               # "fiber_lsf_copied",               # missing fiber filled by copy
               # "fiber_fwhm",                     # aps_fwhm
        "specnum": 1,
        "wavelength_range": (wave_min, wave_max)
    },

    # Copied entry (aps_lsf only)
    933: {
        "interpolate_function": <callable>,         # same function as source fiber
        "type": "fiber_lsf_copied",
        "specnum": 933,
        "wavelength_range": (wave_min, wave_max),
        "copied_from": 932,
        "is_copy": True
    },
    ...
}
```

### 13.1 Using the interpolate_function

```python
import numpy as np

interp_dict = interpolator.get_interpolator_dict()

# Global FWHM at a set of wavelengths
wavelengths = np.array([4000., 5000., 6000., 7000., 8000.])
fwhm_global = interp_dict["global"]["interpolate_function"](wavelengths)
# → array([2.43, 2.61, 2.58, 2.72, 2.89])  (values in Å)

# Fiber-specific FWHM
fwhm_fiber_100 = interp_dict[100]["interpolate_function"](wavelengths)

# Check wavelength range
wmin, wmax = interp_dict[100]["wavelength_range"]
print(f"Fiber 100 valid range: {wmin:.1f} – {wmax:.1f} Å")

# Check if fiber is a copy (aps_lsf only)
if interp_dict.get(933, {}).get("is_copy", False):
    print(f"Fiber 933 is copied from {interp_dict[933]['copied_from']}")
```

---

## 14. Running from Python — Complete Examples

### 14.1 aps_lsf — single arm

```python
from PyAPS.aps_lsf import run_lsf_analysis

interpolator = run_lsf_analysis(
    file_input    = "<PYAPS_DATA>/CAL/lsf_BLUEL11_LIFU.fits",
    figdir        = "<PYAPS_DATA>/L2/20240808/figs",
    figname       = "lsf_blue",
    debug         = True,
    make_plot     = True,
    smooth_length = None,
    overwrite     = False,    # Load from .dill if exists
    save_pickle   = True,
    replace_binned= True,
)

# Global FWHM
print(interpolator.get_fwhm(5000.0))           # scalar
print(interpolator.get_fwhm([4000., 5000.]))   # array

# Fiber-specific
print(interpolator.get_fwhm(5000.0, specnum=123))
```

### 14.2 aps_lsf — two arms (LIFU blue + red)

```python
from PyAPS.aps_lsf import run_lsf_analysis

interpolator = run_lsf_analysis(
    file_input    = ["<PYAPS_DATA>/CAL/lsf_BLUEL11_LIFU.fits",
                     "<PYAPS_DATA>/CAL/lsf_REDL11_LIFU.fits"],
    figdir        = "<PYAPS_DATA>/L2/figs",
    figname       = "lsf_lifu_lr",
    debug         = False,
    make_plot     = True,
    smooth_length = 10.0,       # 10 Å Gaussian smoothing
    kernel_type   = "gaussian",
    overwrite     = False,
    save_pickle   = True,
    replace_binned= True,
)

# Test across the full range
import numpy as np
waves = np.linspace(3800., 9280., 200)
fwhm  = interpolator.get_fwhm(waves)

# Get full interpolator dict for pipeline use
lsf_dict = interpolator.get_interpolator_dict()
print("Available fibers:", len(interpolator.get_available_fibers()))
interpolator.print_summary()
```

### 14.3 aps_lsf — loading from pickle

```python
from PyAPS.aps_lsf import load_lsf_interpolator, LSFInterpolator

# Convenience function
interpolator = load_lsf_interpolator("<PYAPS_DATA>/L2/figs/BLUEL11__REDL11_LIFU.dill")

# Class method
interpolator = LSFInterpolator.load("<PYAPS_DATA>/L2/figs/BLUEL11__REDL11_LIFU.dill")

# Use immediately
fwhm = interpolator.get_fwhm(5000.0, specnum=456)
```

### 14.4 aps_fwhm — single arm

```python
from PyAPS.aps_fwhm import run_fwhm_analysis

interpolator = run_fwhm_analysis(
    file_paths             = "<PYAPS_DATA>/CAL/wave_3100962_all.fit",
    figdir                 = "<PYAPS_DATA>/L2/figs",
    figname                = "fwhm_blue",
    debug                  = True,
    make_plot              = True,
    wave_bin_width         = 50,
    apply_bimodal_filtering= False,
    apply_residual_filtering= True,
    spline_order           = 3,
    spline_smoothing       = None,
    wave_grid_resolution   = 1.0,
    overwrite              = False,
    save_pickle            = True,
    replace_binned         = True,
)

print(interpolator.get_fwhm(5000.0))
```

### 14.5 aps_fwhm — two arms

```python
from PyAPS.aps_fwhm import run_fwhm_analysis

interpolator = run_fwhm_analysis(
    file_paths = ["<PYAPS_DATA>/CAL/wave_3100962_all.fit",
                  "<PYAPS_DATA>/CAL/wave_3100961_all.fit"],
    figdir     = "<PYAPS_DATA>/L2/figs",
    figname    = "fwhm_combined",
    debug      = True,
    overwrite  = False,
    save_pickle= True,
)

interpolator.print_summary()
```

### 14.6 Loading from pickle (aps_fwhm)

```python
from PyAPS.aps_fwhm import load_fwhm_interpolator, FWHMInterpolator

interpolator = load_fwhm_interpolator(
    "<PYAPS_DATA>/L2/figs/wave_3100962_all__wave_3100961_all.dill")

# Or class method
interpolator = FWHMInterpolator.load(
    "<PYAPS_DATA>/L2/figs/wave_3100962_all__wave_3100961_all.dill")
```

### 14.7 Checking the output pickle path before running

```python
from PyAPS.aps_lsf  import get_output_pickle_path as lsf_pickle_path
from PyAPS.aps_fwhm import get_output_pickle_path as fwhm_pickle_path

# aps_lsf
path = lsf_pickle_path(
    ["lsf_GREENH11_MOS-A.fits", "lsf_REDH11_MOS-A.fits"],
    output_dir="<PYAPS_DATA>/output")
# → "<PYAPS_DATA>/output/GREENH11__REDH11_MOS-A.dill"

# aps_fwhm
path = fwhm_pickle_path(
    ["wave_3100962_all.fit", "wave_3100961_all.fit"],
    output_dir="<PYAPS_DATA>/output")
# → "<PYAPS_DATA>/output/wave_3100962_all__wave_3100961_all.dill"
```

### 14.8 Comparing fiber profiles

```python
import numpy as np
import matplotlib.pyplot as plt

waves = np.linspace(3800., 9280., 1000)

fig, ax = plt.subplots(figsize=(12, 5))

# Global master
ax.plot(waves, interpolator.get_fwhm(waves), "k-", lw=2, label="Global")

# Sample of 10 individual fibers
for nspec in interpolator.get_available_fibers()[::96]:
    ax.plot(waves, interpolator.get_fwhm(waves, specnum=nspec),
            alpha=0.5, lw=0.8, label=f"Fiber {nspec}")

ax.set_xlabel("Wavelength (Å)")
ax.set_ylabel("FWHM (Å)")
ax.legend(ncol=3, fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("fiber_comparison.png", dpi=150)
```

---

## 15. Output File Naming and Caching

### 15.1 Overwrite control

Both modules support `overwrite=False` (default) which loads from an existing pickle if found, skipping the (potentially slow) computation:

```python
# First run — creates pickle
interpolator = run_lsf_analysis(files, figdir=..., overwrite=False, save_pickle=True)

# Subsequent runs — loads from pickle (milliseconds)
interpolator = run_lsf_analysis(files, figdir=..., overwrite=False)

# Force regeneration
interpolator = run_lsf_analysis(files, figdir=..., overwrite=True)
```

### 15.2 Pickle directory priority

```
pickle_dir argument  >  figdir  >  directory of first input file
```

### 15.3 Pickle serialization

Both modules use `dill` if available (strongly recommended), falling back to `pickle`. `dill` handles closures and lambda functions correctly, which is essential because the interpolation functions are closures capturing per-fiber data. Install with:

```bash
pip install dill
```

If `dill` is not available a warning is printed and `.pkl` files are used instead. Note that `.dill` and `.pkl` files are not interchangeable — if you install `dill` after creating `.pkl` files, set `overwrite=True` to regenerate.

---

## 16. Smoothing (aps_lsf only)

The pre-computed B-splines from solar twilight files are already smooth by construction. However, some setups or wavelength ranges can show residual noise in the B-spline coefficients. The smoothing capability in `aps_lsf` provides an optional post-processing step.

> **Production note:** Smoothing is not currently applied by default in APS production runs. The pre-computed B-splines from solar twilight calibrations are already smooth by construction and do not require further post-processing in normal operation. The smoothing API is available for diagnostic work or for setups where residual coefficient noise is observed.

### 16.1 Kernel types

| Kernel | Formula | Use case |
|---|---|---|
| `"boxcar"` | Uniform moving average | Fast, no assumptions about noise shape |
| `"gaussian"` | Gaussian with σ = kernel_size/6 | Smooth, ~99% of weight within window |
| `"hanning"` | Hanning window | Smooth edges, good for spectral data |

### 16.2 Performance

Smoothing pre-computes a smoothed grid for every fiber (including global) once, then replaces all interpolation functions with fast `numpy.interp` lookups. For 960 fibers at 1 Å resolution the pre-computation takes a few seconds; subsequent `get_fwhm()` calls are near-instantaneous.

```python
# Apply smoothing
interpolator.apply_smoothing(smooth_length=10.0, kernel_type="gaussian")

# Store smoothed version to pickle
interpolator.save("<PYAPS_DATA>/output/LIFU_LR_smoothed10A.dill")

# Remove smoothing (restores original B-spline functions)
interpolator.remove_smoothing()
```

### 16.3 Smoothing is not available in aps_fwhm

In `aps_fwhm`, smoothing is controlled at fit time via the `spline_smoothing` parameter. Setting `spline_smoothing=None` triggers auto-smoothing based on the noise variance. Setting `spline_smoothing=0` produces an interpolating spline (no smoothing). There is no post-creation smoothing method.

---

## 17. Missing Fiber Filling (aps_lsf only)

The solar twilight LSF file (or master wave file) may not contain entries for all NSPEC values. The main reason is **dead fibres** — fibres that were not delivering light at the time the master calibration was taken. Some of these fibres may have been repaired subsequently and are therefore present in science observations but absent from the calibration. The pipeline handles this transparently via nearest-neighbour filling (see below). Note that for master calibrations, "missing" does not necessarily mean permanently dead — the calibration simply predates the repair.

`_fill_missing_fibers()` fills in the gaps by copying the nearest available fiber's interpolation function. The operation:

1. Lists expected NSPECs from 1 to `max(available_nspecs) + 1`
2. Finds the set difference (missing NSPECs)
3. For each missing NSPEC, finds the nearest available NSPEC by `|nspec - target|`
4. Copies the `interpolate_function` reference (not a deep copy — both point to the same function object)

Filled fibers are flagged with `"is_copy": True` and `"copied_from": <source_nspec>` in the interpolator dict.

`aps_fwhm` does not implement this because only fibres with actual arc line measurements are useful; a dead fibre in the arc frame has no meaningful FWHM.

---

## 18. Diagnostic Plots

Both modules produce a 4-panel diagnostic figure saved as `<headname>_<figname>.png`.

### 18.1 aps_lsf diagnostic panels

**Panel 1: All fiber LSFs + global master**
- Gray lines: all sampled fiber profiles (100 shown)
- Blue line: global master in arm regions
- Red line: global master in gap (linear interpolation)
- Magenta: individual fiber profiles in gap
- Green dashed: arm data boundaries
- Yellow shaded: gap region
- Green shaded: ±200 Å extrapolation zones

**Panel 2: Fiber-to-fiber variation (boxplot)**
- Boxplots at 25 test wavelengths across the full range
- Red circles: global master values

**Panel 3: Sample fibers with gap visualization**
- 5 individual fibers (color coded via viridis)
- Blue: global master
- Red dashed: global master in gap
- Magenta: fiber profiles in gap

**Panel 4: Statistics across all fibers**
- Blue line: mean FWHM
- Shaded: ±1σ
- Green/red dashed: min/max

### 18.2 aps_fwhm diagnostic panels

**Panel 1: Full interpolation all regions (color-coded)**
- Blue dashed: flat extrapolation before arm 1
- Cyan solid: arm 1 spline
- Gold dashed: transition (gap or overlap linear)
- Orange solid: arm 2 spline
- Red dashed: flat extrapolation after arm 2
- Region labels annotated

**Panel 2: Fiber splines + global master**
- Gray: individual fiber splines (100 shown)
- Red: global master
- Blue dashed: median stack

**Panel 3: Fiber-to-fiber variation (boxplot)**
- Boxplots at 20 test wavelengths
- Red: global master

**Panel 4: Color-coded interpolation strategy summary**
- Single continuous line colored by region type
- Region labels (FLAT, ARM 1 SPLINE, LINEAR, ARM 2 SPLINE, FLAT)

---

## 19. Complete API Reference

### 19.1 Module-level functions (both modules)

```python
run_lsf_analysis(
    file_input,           # str or list of str: LSF FITS file(s)
    figdir=None,          # str: output directory for plots and pickle
    figname="lsf_analysis",  # str: base name for output figure
    debug=False,          # bool: verbose output
    make_plot=True,       # bool: create diagnostic plots
    smooth_length=None,   # float: smoothing in Å (aps_lsf only)
    kernel_type="boxcar", # str: smoothing kernel (aps_lsf only)
    overwrite=False,      # bool: force regeneration if pickle exists
    save_pickle=True,     # bool: save to .dill file
    pickle_dir=None,      # str: directory for .dill (None=figdir)
    replace_binned=False, # bool: auto-substitute unbinned if binned missing
) → LSFInterpolator | None

run_fwhm_analysis(
    file_paths,                        # str or list of str: wave FITS file(s)
    figdir=None,                       # str
    figname="analysis",                # str
    debug=False,                       # bool
    make_plot=True,                    # bool
    wave_bin_width=50,                 # float: Å, binwidth for outlier detection
    apply_bimodal_filtering=False,     # bool: GMM-based bimodal filtering
    apply_residual_filtering=True,     # bool: post-fit residual filtering
    spline_order=3,                    # int: 1–5, 3=cubic recommended
    spline_smoothing=None,             # float: None=auto, 0=interpolating
    wave_grid_resolution=1.0,          # float: Å, evaluation grid spacing
    overwrite=False,                   # bool
    save_pickle=True,                  # bool
    pickle_dir=None,                   # str
    replace_binned=False,              # bool
) → FWHMInterpolator | None

load_lsf_interpolator(pickle_path)    → LSFInterpolator  # raises FileNotFoundError
load_fwhm_interpolator(pickle_path)   → FWHMInterpolator

generate_output_headname(file_paths)  → str
get_output_pickle_path(file_paths, output_dir=None) → str
parse_lsf_filename(filename)          → dict  # aps_lsf only
```

### 19.2 LSFInterpolator methods

```python
# Get FWHM
interpolator.get_fwhm(wavelength, specnum=None)
# wavelength: float or array, Å
# specnum: int or None (None = global)
# returns: float or array, Å

# Get full dictionary
interpolator.get_interpolator_dict()   → dict

# List available fibers
interpolator.get_available_fibers()    → list of int

# Smoothing
interpolator.apply_smoothing(smooth_length, kernel_type="boxcar")
interpolator.remove_smoothing()

# Persistence
interpolator.save(output_path=None, output_dir=None)  → str (saved path)
LSFInterpolator.load(pickle_path)                      → LSFInterpolator

# Info
interpolator.print_summary()
interpolator.set_debug(debug)          # bool
interpolator.get_file_prefix()         → str
interpolator.generate_debug_headname() → str
```

### 19.3 FWHMInterpolator methods

```python
# Get FWHM
interpolator.get_fwhm(wavelength, specnum=None) → float or array

# Get full dictionary
interpolator.get_interpolator_dict() → dict

# List available fibers
interpolator.get_available_fibers()  → list of int

# Persistence
interpolator.save(filepath=None)     → str
FWHMInterpolator.load(filepath)      → FWHMInterpolator

# Info
interpolator.print_summary()
interpolator.set_debug(debug)
interpolator.get_file_prefix()       → str
```

### 19.4 Spline order guide (aps_fwhm)

| k | Continuity | Notes |
|---|---|---|
| 1 | C⁰ | Piecewise linear, visible corners |
| 2 | C¹ | Smooth first derivative |
| 3 | C² | **Cubic — recommended** |
| 4 | C³ | Very smooth but can oscillate |
| 5 | C⁴ | Prone to Runge's phenomenon |

---

## 20. Troubleshooting

### "LSF file not found"

```
FileNotFoundError: LSF file not found: <PYAPS_DATA>/CAL/lsf_BLUEL21_LIFU.fits
```

The file does not exist. Either:
1. The wrong path is specified — check `IFU_config_dir` and the setup name
2. The binned version does not exist — set `replace_binned=True` to automatically use the unbinned (`11`) version
3. No LSF file exists for this setup and calibration date — use `aps_fwhm` with `wave_*.fits` files as fallback

### "LSF file is suspiciously small"

The file is corrupted or incomplete (under 1 KB). Re-download from CAMCEAD.

### "FITS table missing required columns"

The LSF FITS file has an unexpected structure. This can happen if an older format file is used. Check that the columns `NSPEC`, `t`, `c`, `k` all exist in HDU 1.

### All fibers return the same FWHM

```
WARNING: All fibers have same FWHM! This is wrong!
```

This occurs if per-fiber interpolation functions are accidentally sharing the same closed-over data (a Python closure bug). This was fixed in the current version by using default-argument capture in closure factories. If this occurs, check that you are running the current version of `aps_lsf.py`.

### Zero values in the gap region

Both modules use per-fiber wavelength ranges for the transition. Zeros appear if global ranges are used instead (a bug that was fixed). If you see zeros between arms, check that `fiber_data["wave_range"]` is being used (not the global file-level range).

### NaN values from BSpline at boundaries

The `BSpline` with `extrapolate=False` returns NaN outside the knot range. The boundary evaluation uses `wave_min + 0.1` and `wave_max - 0.1` to avoid this. If NaN still appears for a specific fiber, that fiber's B-spline has a degenerate knot configuration and is excluded from the file with a warning.

### Pickle load fails after code changes

```
IOError: Failed to load LSF interpolator from: .../BLUEL11_LIFU.dill
Error: unpickling error...
```

The pickle was created with a different version of the code or Python. Delete the `.dill` file and regenerate with `overwrite=True`.

### aps_fwhm "Too few points" error

```
ValueError: Too few valid wavelengths (3)
```

After outlier filtering too few arc line measurements survive for that fiber or arm. Try:
1. Increase `wave_bin_width` (less aggressive binning → fewer points removed)
2. Set `apply_residual_filtering=False`
3. Check the input `wave_*.fits` file for that arm — it may have bad data

### Smoothing makes all fibers identical

This is expected for `kernel_type="boxcar"` with very large `smooth_length` — if the kernel is wider than the wavelength range, every fiber converges to the same mean. Use a smaller `smooth_length` or `kernel_type="gaussian"`.

### MOS binned files missing

For MOS in SALSA mode, only `bin=11` (unbinned) LSF files are typically available. If your L1 data uses binning 21 or 41, set `replace_binned=True` and the code will transparently substitute the `11` version. A warning is printed to stderr.

---

*For further information, bug reports, or contributions please open an issue on the PyAPS repository.*
