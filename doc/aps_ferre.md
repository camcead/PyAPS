# aps_ferre — Stellar Parameter Estimation with FERRE

## WEAVE PyAPS Pipeline — Module Reference Guide

**Module:** `PyAPS.aps_ferre`, `PyAPS.aps_ifu_ferre`
**Used by:** MOS Galactic mode (`aps_mosGAL`), IFU Galactic mode (`aps_ifu_Gal`)
**Depends on:** `aps_rvs` (radial velocities as initial conditions), `aps_rr` (classification)
**External code:** [FERRE](https://github.com/callendeprieto/ferre) — C. Allende Prieto et al.
**Maintained by:** CASU, Institute of Astronomy, University of Cambridge

---

## Table of Contents

1. [Overview](#1-overview)
2. [What FERRE Does](#2-what-ferre-does)
3. [Module Architecture](#3-module-architecture)
4. [FERRE Grids](#4-ferre-grids)
5. [MOS Mode — aps_ferre](#5-mos-mode--aps_ferre)
6. [IFU Mode — aps_ifu_ferre](#6-ifu-mode--aps_ifu_ferre)
7. [LSF Handling in FERRE](#7-lsf-handling-in-ferre)
8. [Spectral Arm Handling](#8-spectral-arm-handling)
9. [Multi-Grid Strategy and opfmerge](#9-multi-grid-strategy-and-opfmerge)
10. [Error Handling and IEEE Warnings](#10-error-handling-and-ieee-warnings)
11. [Input and Output File Formats](#11-input-and-output-file-formats)
12. [Output FITS Products](#12-output-fits-products)
13. [Configuration Parameters](#13-configuration-parameters)
14. [Running from the Command Line](#14-running-from-the-command-line)
15. [Running from Python](#15-running-from-python)
16. [Complete API Reference](#16-complete-api-reference)
17. [Troubleshooting](#17-troubleshooting)

---

## 1. Overview

`aps_ferre` is the PyAPS wrapper around the FERRE stellar parameter fitting code. It
determines fundamental stellar atmospheric parameters — effective temperature (T_eff),
surface gravity (log g), and iron abundance ([Fe/H]), and optionally alpha-element
abundance ([α/Fe]) — by fitting observed WEAVE spectra against a library of synthetic
stellar spectra.

The module operates in two modes:

- **MOS mode** (`aps_ferre.py`): Processes individual fibre spectra from MOS observations.
  Runs after RVS to benefit from accurate initial radial velocities. Designed for large
  samples (hundreds to thousands of targets per observation block).

- **IFU mode** (`aps_ifu_ferre.py`): Processes spatially binned spectra from IFU Galactic
  mode observations. Handles the additional complexity of spatially binned spectra and
  stricter arm-overlap splitting requirements.

Both modes share the same underlying FERRE execution engine, grid merging logic, and
output format.

---

## 2. What FERRE Does

FERRE (Full Spectrum Fitting with a Grid of Stellar Models) fits observed spectra by
interpolating within a pre-computed grid of synthetic spectra. For each target it:

1. Takes the observed spectrum and its error array
2. Applies a radial velocity correction (from RVS results)
3. Applies a running mean normalisation to both the observed and the synthetic spectra
4. Searches the grid for the synthetic spectrum that minimises χ²
5. Returns the best-fit parameters with formal uncertainties from the covariance matrix

The fitting is done in log-wavelength space (as output by the WEAVE pipeline) and can
handle multi-arm spectra by concatenating the arms into a single array with a matching
synthetic grid.

FERRE is a Fortran code and is called as an external subprocess by PyAPS. It reads
its parameters from a Fortran namelist file (`input.nml`) and writes its results to
text output files (`.opf`, `.mdl`, `.nrd`).

---

## 3. Module Architecture

```
aps_runner / aps_dms
      │
      ▼
proc_ferre()                          ← MOS entry point (aps_ferre.py)
proc_ferre_ifu()                      ← IFU entry point (aps_ifu_ferre.py)
      │
      ├── Read L1 data via APSOB
      ├── Read RVS results (initial velocity guess)
      ├── Apply velocity correction and normalisation
      ├── write_ferre_input()         ← write .frd, .err, .wav, .vrd
      │
      ├── For each grid k (run in parallel via ThreadPool):
      │       mknml()                 ← build Fortran namelist dict
      │       writenml()              ← write input.nml_k
      │       ferre_exe_worker()      ← launch FERRE subprocess, capture output
      │
      ├── opfmerge()                  ← wait for output, merge across grids
      │       ├── Wait for .opf, .mdl, .nrd files per grid
      │       ├── For each target: select grid with lowest χ²
      │       └── Write merged .opf, .mdl, .nrd
      │
      └── parse_ferre_output()        ← read merged output → FITS table
```

---

## 4. FERRE Grids

FERRE grids are pre-computed libraries of synthetic spectra that span the stellar
parameter space. PyAPS uses multiple grids simultaneously, each optimised for a
different stellar population or parameter range. The pipeline runs FERRE once per grid
and then selects the best-fitting grid for each target.

### 4.1 Grid types used by PyAPS

| Grid index | Stellar type | Temperature range | Dimensionality |
|---|---|---|---|
| 1–5 | Main sequence and giant stars | T_eff 3500–8000 K | 3D or 5D |
| 6–9 | White dwarfs | T_eff 5000–80000 K | 3D |

The exact number and configuration of grids depends on the observation mode and
resolution. For LR mode, 9 grids are used; for HR mode, a subset is used.

### 4.2 Grid dimensionality

**3D grids** fit T_eff, log g, [Fe/H]. Used for white dwarfs and low-resolution data.

**5D grids** fit T_eff, log g, [Fe/H], [α/Fe], and microturbulence. Used for
giant and main-sequence stars at low resolution.

**PCA-compressed grids** store synthetic spectra in principal component form to
reduce disk space and memory footprint. FERRE handles the decompression internally.

### 4.3 Grid file naming

Grid files follow the convention:

```
<grid_name>-<MODE>_<ARM>.hdr       ← header file (FERRE reads this)
<grid_name>-<MODE>_<ARM>.unf       ← binary flux data
```

Example: `GKgiant_LR-MOS_BLUEL11.hdr` for blue-arm LR MOS data.

For single-arm runs the arm suffix is omitted; for multi-arm runs each arm gets its
own matching grid file. FERRE reads all arm grids simultaneously and fits the
concatenated spectrum.

---

## 5. MOS Mode — aps_ferre

### 5.1 Processing flow

```
proc_ferre(nthreads, infiles, outdir, configdir, rootname, configs,
           wlranges, aps_ids, targsrvy, targclass, mask_aps_ids, area,
           mask_areas, cat_list, hpc, debug)
```

**Step 1 — Target selection**

Targets classified by RR as stars (`CLASS_STAR=1` or `SPECTYPE` matching stellar
templates) are selected for FERRE analysis. Targets already processed in a previous
run are skipped if `overwrite_FR=False` in the config.

**Step 2 — RVS data read**

The RVS output file (`rvs_<rootname>.fits`) is read to extract:
- Radial velocity for each target (used as initial condition and for de-redshifting)
- `RVS_WARN` bitmask (targets with fatal RVS failures are excluded)

If no RVS file exists, FERRE runs with zero initial velocity.

**Step 3 — Spectrum preparation**

For each target:
- Spectrum and error arrays are read from the L1 FITS file via APSOB
- The spectrum is shifted to rest frame using the RVS radial velocity
- A running mean normalisation is applied to both the observed and synthetic spectra (order set by `CONT_ORDER_FR`)
- Bad pixels (IVAR = 0) are flagged with a large error value

**Step 4 — FERRE execution**

All targets are batched into a single FERRE run (not one per target). The input files
for all targets are written together and FERRE processes them in one pass per grid.
Grids run in parallel using Python's `ThreadPool`.

**Step 5 — Grid merging**

`opfmerge()` selects the best-fitting grid for each target based on χ² per degree of
freedom, subject to temperature thresholds (some grids are only valid within their
designed temperature range).

**Step 6 — Output**

Results are written to `ferre_<rootname>.fits`.

### 5.2 Parallelisation

FERRE itself is multi-threaded (controlled by `nthreads` in the namelist). PyAPS
additionally runs multiple grids simultaneously via a Python `ThreadPool`. The total
CPU load is therefore `nthreads × n_grids`. For a 9-grid run with 8 threads per
grid, this uses 72 CPU cores. In practice `aps_runner` sets `mp_MOS` to control
`nthreads` and caps the thread pool automatically.

---

## 6. IFU Mode — aps_ifu_ferre

IFU Galactic mode processes spatially binned spectra produced by `aps_ifu_Gal`
(Voronoi or adaptive binning). Each spatial bin is processed independently.

### 6.1 Key differences from MOS mode

| Aspect | MOS | IFU |
|---|---|---|
| Input | Individual fibre spectra | Spatially binned spectra |
| Entry point | `proc_ferre()` | `proc_ferre_ifu()` |
| Targets per run | All targets batched together | One bin at a time |
| Arm handling | Arms concatenated in APSOB | Explicit split at overlap centre |
| Parallelisation | ThreadPool over grids | ThreadPool over grids per bin |

### 6.2 IFU processing flow

```
proc_ferre_ifu()
      │
      ├── read_binned_spectra_for_ferre()
      │       Reads BINSpectra.fits from aps_ifu_Gal output
      │       Returns list of (bin_id, wave, flux, error) tuples
      │
      ├── For each bin:
      │       split_spectrum_by_setup()   ← split arms at overlap centre
      │       write_ferre_input_files()   ← write per-bin .frd, .err, .wav, .vrd
      │       run_ferre_on_bin()          ← execute FERRE on this bin
      │               ├── mknml_ifu()
      │               ├── writenml()
      │               ├── ferre_exe_worker() × n_grids (parallel)
      │               └── opfmerge()
      │       parse_ferre_output_ifu()    ← extract parameters for this bin
      │
      └── Assemble all bins → write FITS output
```

### 6.3 Overlap handling

When the blue and red arms overlap in wavelength (common for WEAVE LR mode where the
overlap region is ~5900–5950 Å), `split_spectrum_by_setup()` finds the centre of the
overlap and splits each arm there:

```
Blue arm: 3800 → 5924 Å   (cut at overlap centre)
Red arm:  5924 → 9280 Å   (starts at overlap centre)
```

This avoids the same wavelength pixels being included in both arms, which would
violate FERRE's assumption of independent pixels. The cut point is logged for each bin.

---

## 7. LSF Handling in FERRE

FERRE does not convolve the synthetic spectra internally at runtime. Instead, PyAPS pre-convolves each synthetic grid to the WEAVE instrumental resolution before storing it. The LSF (characterised as FWHM in Å as a function of wavelength, from either `aps_lsf` or `aps_fwhm`) is applied during grid construction by the grid-generation scripts in `py/grids_generators/FR/`. The resulting grids already encode the instrumental broadening for a specific setup (arm, resolution, mode), which is why separate grids exist per setup (e.g. `GKgiant_LR-MOS_BLUEL11.hdr`). At run time, no further LSF convolution is performed — FERRE compares the observed normalised spectrum directly against the pre-broadened synthetic grid.

---

## 8. Spectral Arm Handling

### 7.1 Single-arm runs

When only one arm is present (e.g. HR blue only), FERRE is given a single grid file
and a single flux array. The namelist parameter `NSYNTH` is set to 1.

### 7.2 Two-arm runs (LR mode)

For LR observations with both blue and red arms, FERRE receives two synthetic grid
files (one per arm) and a concatenated flux array with the two arms joined end-to-end.
The namelist parameter `NSYNTH` is set to 2. FERRE fits both arms simultaneously,
which is important because it breaks the temperature–metallicity degeneracy that
affects single-arm fits.

The arm concatenation order is always blue first, red second, matching the order of
the grid files in the namelist.

### 7.3 Wavelength files

FERRE reads the wavelength array from a `.wav` file (one wavelength per pixel, in the
same order as the flux array). For two-arm runs the wavelength file contains the
concatenated blue+red wavelength array. These files are written by
`write_ferre_input()` / `write_ferre_input_files()`.

---

## 9. Multi-Grid Strategy and opfmerge

Running FERRE on multiple grids is the core of the PyAPS stellar parameter strategy.
Each grid covers a different region of parameter space, and different stellar types
(giants, dwarfs, white dwarfs) have their own optimal grid. Rather than pre-classify
targets and assign them to grids (which would require knowing the answer before the
fit), PyAPS fits all grids and selects the best result for each target.

### 8.1 opfmerge — how it works

```python
opfmerge(pixel, grid_ids, grid_prefix, path,
         min_grids_required=5,
         wait_time=10, max_wait=300, cooldown=5)
```

**Waiting for output files**

FERRE writes its output asynchronously. `opfmerge` polls the output directory for
`.opf`, `.mdl`, and `.nrd` files from each grid, waiting up to `max_wait` seconds.
If a grid's files do not appear within the timeout, that grid is marked as failed and
excluded from the merge.

**Minimum grid requirement**

If fewer than `min_grids_required` grids produce valid output, `opfmerge` raises an
error and the target/bin is marked as failed. This prevents a degenerate merge from
a small number of grids biasing the results.

**Grid selection per target**

For each target in the batch, `opfmerge` reads the χ² values from all successful
grid output files and selects the grid with the lowest χ² per degree of freedom.
Temperature thresholds are applied: if a target's best-fit temperature from grid k
falls outside the valid range for that grid, the next-best grid is used instead.

**Output files**

The merged results are written to:
- `<pixel>.opf` — parameter values (T_eff, log g, [Fe/H], etc.) and χ²
- `<pixel>.mdl` — best-fit synthetic spectrum at each pixel
- `<pixel>.nrd` — normalised observed spectrum

### 8.2 Grid tracking metadata

`opfmerge` writes a metadata file recording which grid was selected for each target.
This is stored in the output FITS file and allows post-processing analysis of which
stellar population grid was most appropriate for each observation.

---

## 10. Error Handling and IEEE Warnings

### 9.1 ferre_exe_worker

FERRE is called as a subprocess with its stdout and stderr captured in real time.
The worker function `ferre_exe_worker()` returns a tuple `(success, return_code, error_msg)`.

```python
success, return_code, error_msg = ferre_exe_worker(
    [nml_path, ferre_executable, True])
```

### 9.2 IEEE floating-point warnings

FERRE (a Fortran code) frequently emits IEEE floating-point warnings on completion:

```
Note: The following floating-point exceptions are signalling:
IEEE_OVERFLOW_FLAG IEEE_UNDERFLOW_FLAG IEEE_DENORMAL
STOP 1
```

These warnings appear in `STDERR` and cause FERRE to exit with a non-zero return code
(`STOP 1`). However, they do not corrupt the output — FERRE has already written all
output files before encountering the floating-point condition. PyAPS therefore treats
this situation as a success if:

1. The output `.opf` file exists and has non-zero size
2. The word `quit: Run ended` appears in stdout (indicating normal completion)
3. No actual error (`ERROR:` in stderr) was reported

```python
# In ferre_exe_worker():
if return_code != 0 and output_file_exists and run_ended_normally:
    # IEEE warning only — treat as success
    return (True, return_code, None)
```

This prevents large fractions of targets being lost to a benign Fortran quirk.

### 9.3 True failures

A true FERRE failure is indicated by one or more of:
- `ERROR:` in stderr
- Missing output file
- `quit: Run ended` absent from stdout
- Return code non-zero AND output file absent

In these cases `ferre_exe_worker()` returns `(False, return_code, error_msg)` and
the grid is excluded from the merge.

---

## 11. Input and Output File Formats

### 10.1 FERRE input files

All input files are written per pixel (MOS: one pixel = all targets in one batch;
IFU: one pixel = one spatial bin).

| File | Content |
|---|---|
| `<pixel>.frd` | Flux array, one target per row, space-separated |
| `<pixel>.err` | Error array, matching `.frd` layout |
| `<pixel>.wav` | Wavelength array (one row, shared across all targets) |
| `<pixel>.vrd` | Initial parameter guesses (T_eff, log g, [Fe/H], RV, ...) |
| `input.nml_k` | Fortran namelist for grid k |

### 10.2 FERRE output files (per grid)

| File | Content |
|---|---|
| `<pixel>.opf<k>` | Best-fit parameters for each target, from grid k |
| `<pixel>.mdl<k>` | Best-fit synthetic spectrum for each target |
| `<pixel>.nrd<k>` | Normalised observed spectrum |

After merging:

| File | Content |
|---|---|
| `<pixel>.opf` | Merged best-fit parameters (one row per target) |
| `<pixel>.mdl` | Merged best-fit synthetic spectra |
| `<pixel>.nrd` | Merged normalised observed spectra |

### 10.3 .opf column layout

The `.opf` file has a fixed column structure that depends on the grid dimensionality:

**3D grid (19 columns)**

```
ID  Teff  logg  FeH  chi2  snr  flag  cov11 cov12 cov13 cov22 cov23 cov33  [...]
```

**5D grid (39 columns)**

```
ID  Teff  logg  FeH  alphaFe  micro  chi2  snr  flag  [covariance matrix ...]
```

**White dwarf grid (12 columns)**

```
ID  Teff  logg  chi2  snr  flag  cov11 cov12 cov22  [...]
```

`parse_ferre_output()` / `parse_ferre_output_ifu()` handles all three layouts by
detecting the number of columns.

---

## 12. Output FITS Products

Results are written to `ferre_<rootname>.fits` with the following HDU structure.

### HDU 1 — `FERRE_RESULTS` (binary table)

One row per target. Key columns:

| Column | Unit | Description |
|---|---|---|
| `APS_ID` | — | WEAVE fibre spectrum identifier |
| `TARGID` | — | WEAVE target identifier string |
| `CNAME` | — | WEAVE coordinate name |
| `TEFF` | K | Effective temperature |
| `ERR_TEFF` | K | Formal uncertainty on T_eff |
| `LOGG` | dex | Surface gravity |
| `ERR_LOGG` | dex | Formal uncertainty on log g |
| `FEH` | dex | Iron abundance [Fe/H] |
| `ERR_FEH` | dex | Formal uncertainty on [Fe/H] |
| `ALPHAFE` | dex | Alpha-element abundance (5D grids only) |
| `ERR_ALPHAFE` | dex | Formal uncertainty on [α/Fe] (5D grids only) |
| `MICRO` | km/s | Microturbulence (5D grids only) |
| `CHI2` | — | χ² per degree of freedom of best fit |
| `SNR` | — | Signal-to-noise ratio of the fitted spectrum |
| `GRID_ID` | — | Index of the grid selected by opfmerge |
| `GRID_NAME` | — | Name of the grid selected by opfmerge |
| `FERRE_WARN` | — | Warning bitmask (see below) |
| `RV_INPUT` | km/s | Radial velocity used for de-redshifting (from RVS) |

### FERRE_WARN bitmask

> **Note:** The flags below reflect the current implementation. Whether all bits are actively set and their thresholds are appropriate should be reviewed as the pipeline evolves.

| Bit | Value | Meaning |
|---|---|---|
| 0 | 1 | No RVS result available; RV=0 used |
| 1 | 2 | IEEE warnings in FERRE stderr |
| 2 | 4 | Fewer than `min_grids_required` grids succeeded |
| 3 | 8 | Best-fit parameters at grid boundary |
| 4 | 16 | χ² > threshold; fit quality poor |
| 5 | 32 | SNR below minimum threshold |
| 6 | 64 | White dwarf grid selected |

### HDU 2 — `FERRE_SPEC` (binary table)

Best-fit spectra for diagnostic purposes.

| Column | Description |
|---|---|
| `APS_ID` | Fibre identifier |
| `WAVE` | Wavelength array (Å) |
| `FLUX_OBS` | Observed normalised spectrum |
| `FLUX_MODEL` | Best-fit synthetic spectrum |
| `FLUX_ERR` | Error spectrum |

---

## 13. Configuration Parameters

FERRE parameters are set in the module config JSON file (for IFU) or in
`script_params.yaml` (for MOS). The key parameters:

| Parameter | Default | Description |
|---|---|---|
| `CONT_ORDER_FR` | 6 | Order of the running mean normalisation applied to both observed and synthetic spectra |
| `NDIM_FR` | 3 | Number of fitted parameters (3=T_eff, log g, [Fe/H]; 5=adds α/Fe and micro) |
| `GRID_FR` | see configs | List of grid names to use |
| `MIN_GRIDS_FR` | 5 | Minimum grids required for opfmerge to succeed |
| `MIN_SNR_FR` | 3.0 | Minimum SNR to attempt FERRE fit |
| `NTHREADS_FR` | 4 | FERRE internal thread count |
| `WAIT_FR` | 10 | Seconds between opfmerge file polls |
| `MAX_WAIT_FR` | 300 | Maximum seconds to wait for FERRE output |
| `OVERWRITE_FR` | True | Overwrite existing FERRE output |
| `FIG_FR` | True | Generate diagnostic spectral plots |

For single-exposure (stacking level 0) data, the pipeline automatically switches to
the lighter 3D grid set and halves `NTHREADS_FR` to avoid memory pressure.

---

## 14. Running from the Command Line

`aps_ferre` is normally invoked by `aps_runner` rather than directly. The generated
script looks like:

```bash
python aps_ferre.py \
  --infiles /data/L1/stack_3095664.fit /data/L1/stack_3095663.fit \
  --outdir /data/L2/20250630/OB12345/ \
  --configdir <PYAPS_DIR>/configs/MOS_LR/ \
  --rootname stack_3095664__stack_3095663 \
  --wlranges 3800.0,5950.0 5900.0,9280.0 \
  --nthreads 8 \
  --debug False
```

For IFU mode:

```bash
python aps_ifu_ferre.py \
  --infiles /data/L1/stackcube_3123971.fit /data/L1/stackcube_3123970.fit \
  --outdir /data/L2/20250630/OB12345/ \
  --configdir <PYAPS_DIR>/configs/LIFU_LR/ \
  --rootname stackcube_3123971__stackcube_3123970 \
  --wlranges 3800.0,5950.0 5900.0,9280.0 \
  --nthreads 4
```

---

## 15. Running from Python

### 14.1 MOS mode

```python
from PyAPS.aps_ferre import proc_ferre

proc_ferre(
    nthreads   = 8,
    infiles    = ["/data/L1/stack_3095664.fit",
                  "/data/L1/stack_3095663.fit"],
    outdir     = "/data/L2/20250630/OB12345/",
    configdir  = "<PYAPS_DIR>/configs/MOS_LR/",
    rootname   = "stack_3095664__stack_3095663",
    configs    = {...},          # loaded from script_params.yaml
    wlranges   = [[3800., 5950.], [5900., 9280.]],
    aps_ids    = None,           # None = process all stellar targets
    targsrvy   = None,
    targclass  = None,
    mask_aps_ids = None,
    area       = None,
    mask_areas = None,
    cat_list   = None,
    hpc        = 0,
    debug      = False,
)
```

### 14.2 IFU mode

```python
from PyAPS.aps_ifu_ferre import proc_ferre_ifu

proc_ferre_ifu(
    nthreads   = 4,
    infiles    = ["/data/L1/stackcube_3123971.fit",
                  "/data/L1/stackcube_3123970.fit"],
    outdir     = "/data/L2/20250630/OB12345/",
    configdir  = "<PYAPS_DIR>/configs/LIFU_LR/",
    rootname   = "stackcube_3123971__stackcube_3123970",
    configs    = {...},
    wlranges   = [[3800., 5950.], [5900., 9280.]],
    debug      = False,
)
```

### 14.3 Calling opfmerge directly (diagnostic use)

```python
from PyAPS.aps_ferre import opfmerge

result = opfmerge(
    pixel            = "stack_3095664__stack_3095663",
    grid_ids         = ["1", "2", "3", "4", "5", "6", "7", "8", "9"],
    grid_prefix      = "n",          # 'n' for normal, 'm' for modified, 'p' for PCA
    path             = "/data/L2/workdir/",
    min_grids_required = 5,
    wait_time        = 10,           # seconds between polls
    max_wait         = 300,          # maximum wait time in seconds
    cooldown         = 5,            # seconds after all files found before reading
)

# result is a dict:
# {
#   "success": True,
#   "n_grids_used": 9,
#   "grid_selection": {0: "3", 1: "1", 2: "5", ...},  # target_idx → grid_id
#   "failed_grids": [],
# }
```

### 14.4 Calling ferre_exe_worker directly (diagnostic use)

```python
from PyAPS.aps_ferre import ferre_exe_worker

success, return_code, error_msg = ferre_exe_worker(
    ["/data/L2/workdir/input.nml_1",   # namelist file path
     "<FERRE_DIR>/src/ferre.x",    # FERRE executable path
     True]                              # capture_output flag
)

if success:
    print("FERRE completed successfully")
else:
    print(f"FERRE failed: {error_msg}")
```

---

## 16. Complete API Reference

### 15.1 aps_ferre.py

```python
proc_ferre(nthreads, infiles, outdir, configdir, rootname, configs,
           wlranges, aps_ids=None, targsrvy=None, targclass=None,
           mask_aps_ids=None, area=None, mask_areas=None, cat_list=None,
           hpc=0, debug=False)
# Main entry point for MOS FERRE analysis.
# Returns: None (writes ferre_<rootname>.fits to outdir)

opfmerge(pixel, grid_ids, grid_prefix, path,
         min_grids_required=5, wait_time=10, max_wait=300, cooldown=5)
# Merge FERRE outputs from multiple grids; select best grid per target.
# Returns: dict with keys success, n_grids_used, grid_selection, failed_grids

ferre_exe_worker(cmdstr)
# Execute FERRE as a subprocess.
# cmdstr: [nml_path, ferre_exe, capture_output]
# Returns: (success: bool, return_code: int, error_msg: str | None)

mknml(synthfiles, path, pixel, grid_id, maxorder, nthreads=4)
# Build the Fortran namelist dictionary for a given grid.
# Returns: dict

writenml(nml, nmlfile='input.nml', path='.')
# Write namelist dict to disk as a Fortran namelist file.
# Returns: None

write_ferre_input(pixel, ids, par, wave, flux, error, path='.')
# Write FERRE input files (.frd, .err, .wav, .vrd) for MOS mode.
# Returns: None

parse_ferre_output(path, pixel, n_targets, grid_dim)
# Parse merged FERRE output into arrays.
# grid_dim: 3, 5, or 12 (for WD grids)
# Returns: dict with keys teff, logg, feh, chi2, snr, warn, covariance, ...
```

### 15.2 aps_ifu_ferre.py

```python
proc_ferre_ifu(nthreads, infiles, outdir, configdir, rootname, configs,
               wlranges, debug=False)
# Main entry point for IFU FERRE analysis.
# Returns: None (writes ferre_<rootname>.fits to outdir)

read_binned_spectra_for_ferre(outdir, rootname)
# Read BINSpectra.fits from aps_ifu_Gal output.
# Returns: list of (bin_id, wave, flux, error) tuples

split_spectrum_by_setup(wave, flux, error, wlranges)
# Split a joined multi-arm spectrum at the overlap centre.
# Returns: list of (setup_name, wave_arm, flux_arm, error_arm) tuples

write_ferre_input_files(pixel, bin_id, arm_spectra, par, path='.')
# Write FERRE input files for a single IFU spatial bin.
# Returns: None

run_ferre_on_bin(pixel, bin_id, arm_spectra, grids, grid_ids, grid_prefix,
                 targs_mode, work_dir, ferre, nthreads, maxorder,
                 min_grids_required=3)
# Execute FERRE on one IFU spatial bin across all grids.
# Returns: dict with merged parameter results

mknml_ifu(synthfiles, path, pixel, grid_id, maxorder, nthreads=4)
# Build Fortran namelist for IFU mode.
# Returns: dict

parse_ferre_output_ifu(work_dir, pixel, bin_id, grid_prefix)
# Parse merged FERRE output for one spatial bin.
# Returns: dict with teff, logg, feh, chi2, snr, warn, ...
```

---

## 17. Troubleshooting

### "End of file" Fortran error

```
At line 35 of file load_control.f90 (unit = 1, file = 'input.nml')
Fortran runtime error: End of file
```

The namelist file is incomplete or malformed. Causes:
- A required namelist key is missing — check that `mknml()` / `mknml_ifu()` is
  producing all required entries
- The grid file path in the namelist does not exist — verify `synthfiles` paths
- Encoding issue in the namelist file — ensure no Unicode characters appear

### "opf file missing" after FERRE reports success

FERRE completed but `opfmerge` cannot find the output files. This is usually a
working directory issue: FERRE writes output to the current working directory, not
the directory containing the namelist. The fix is to ensure FERRE is launched with
the working directory set to `path`:

```python
subprocess.Popen([ferre_exe], stdin=open(nml_path),
                 cwd=path, ...)   # ← cwd must be set
```

### FERRE fails on all grids

Check:
1. The FERRE executable path is correct and executable (`chmod +x ferre.x`)
2. The grid `.hdr` and `.unf` files exist and are readable
3. The `.frd` and `.err` files were written correctly (check file sizes)
4. Available memory — 5D grids can require several GB per thread

### Too few grids succeed (opfmerge error)

If fewer than `min_grids_required` grids produce output, reduce `MIN_GRIDS_FR` in
the config, or investigate why grids are failing (check per-grid error logs in the
working directory, named `<pixel>_grid<k>.err`).

### All targets assigned to same grid

This can happen if the χ² values from most grids are identical (e.g. all NaN or
all zero). Check that the input `.frd` flux values are not all zero — this indicates
a normalisation failure upstream.

### IFU: "missing 1 required positional argument: targs_mode"

`run_ferre_on_bin()` requires the `targs_mode` argument (e.g. `'LIFU LR'`). Ensure
it is passed explicitly from `proc_ferre_ifu()`:

```python
run_ferre_on_bin(..., targs_mode=configs['SETMODE'], ...)
```

### SNR too low — targets skipped

Targets below `MIN_SNR_FR` are skipped with `FERRE_WARN` bit 5 set. Lower the
threshold in the config if legitimate low-SNR targets are being excluded, but note
that FERRE fits below SNR~3 are typically unreliable.

---

*For further information, bug reports, or contributions please open an issue on the
PyAPS repository.*
