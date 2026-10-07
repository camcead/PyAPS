# aps_ifu_exgal — ExGal IFU Analysis Module

## A Reference and Tutorial Guide for the PyAPS Pipeline

**Module:** `PyAPS.aps_ifu_exgal`
**Entry script:** `aps_ExGal_worker.py`
**PyAPS version:** 1.7+
**Applies to:** WEAVE LIFU and MIFU observations, galaxy and QSO targets
**Maintained by:** CASU, Institute of Astronomy, University of Cambridge

---

## Table of Contents

1. [What This Module Does](#1-what-this-module-does)
2. [How It Fits in the Pipeline](#2-how-it-fits-in-the-pipeline)
3. [Public API — Functions and Their Roles](#3-public-api--functions-and-their-roles)
4. [The Two-Stage Architecture](#4-the-two-stage-architecture)
5. [Stage 1 — ifu_ExGal_prepare in Detail](#5-stage-1--ifu_exgal_prepare-in-detail)
6. [Stage 2 — ifu_ExGal Loop in Detail](#6-stage-2--ifu_exgal-loop-in-detail)
7. [Running from the Command Line](#7-running-from-the-command-line)
8. [Running from Python](#8-running-from-python)
9. [Input Parameters — Complete Reference](#9-input-parameters--complete-reference)
10. [Configuration File Reference](#10-configuration-file-reference)
11. [Spaxel-Weighted LSF and Resolution Buckets](#11-spaxel-weighted-lsf-and-resolution-buckets)
12. [The Patch Array — Single Target Mode](#12-the-patch-array--single-target-mode)
13. [Wavelength Window Selection — ppxf_limits](#13-wavelength-window-selection--ppxf_limits)
14. [Adaptive MIN_SNR](#14-adaptive-min_snr)
15. [Output Files](#15-output-files)
16. [Controlling output file size — `--no_spec_ext`](#16-controlling-output-file-size---no_spec_ext)
17. [Troubleshooting](#17-troubleshooting)

---

## 1. What This Module Does

`aps_ifu_exgal` is the ExGal (extragalactic) IFU analysis module for WEAVE data. It processes all `GALAXY` and `QSO` targets found in an IFU observation, running the full chain from raw L1 data to final L2 science products.

For each galaxy or QSO target in the patch table it:

1. Builds a spaxel cube from the L1 data via APSOB
2. Converts from observed to rest-frame wavelengths using the target redshift
3. Derives an adaptive minimum SNR threshold to remove bad spaxels
4. Optionally applies spatial pre-binning and surface brightness filtering
5. Runs Voronoi adaptive tessellation to reach target SNR per bin
6. Log-rebins spectra to a constant velocity scale for pPXF
7. Runs stellar kinematics (pPXF), emission line fitting (EMIPPXF), and line strengths (LS)
8. Merges all results into a single L2 FITS product

This module is self-contained: it owns the loop over all targets, handles all exceptions per target, and calls the L2 merge at the end of each target's processing.

---

## 2. How It Fits in the Pipeline

The data flow through the pipeline is:

```
L1 FITS files
      │
      ▼
aps_ifu_prepare.py          ←  source detection, Redrock classification,
      │                         patch file creation
      │  patch_file (targets_mod.fits)
      │
      ▼
aps_ExGal_worker.py         ←  YOU ARE HERE
      │
      │  calls ifu_ExGal()
      │      │
      │      ├── for each GALAXY/QSO row in patch table:
      │      │       ifu_ExGal_prepare()    ←  cube + binning + Voronoi
      │      │       IFUExGalPPXF            ←  stellar kinematics
      │      │       IFUExGalEMIPPXF         ←  emission lines
      │      │       IFUExGalLS              ←  line strengths
      │      │       ifuExGalL2merge        ←  merge to L2 FITS
      │      │
      └──────┘
             │
             ▼
      L2 FITS products  (<headname>_P####_APS.fits)
```

**Input from aps_ifu_prepare:** The patch file `<headname>_targets_mod.fits` (see `IFU_PIPELINE_README.md`) tells this module which targets to process and provides their RA/Dec apertures, redshifts, and classifications. You must run `aps_ifu_prepare.py` first.

**Parallel with aps_Gal_worker:** In the automated pipeline, `aps_ExGal_worker` and `aps_Gal_worker` run in parallel after the prepare stage completes. They read the same patch file but `aps_ExGal_worker` processes only `GALAXY` and `QSO` rows while `aps_Gal_worker` processes `STAR` rows.

---

## 3. Public API — Functions and Their Roles

The module exports the following public functions:

| Function | Role |
|---|---|
| `ifu_ExGal_prepare(...)` | **Stage 1**: Data ingestion, cube assembly, adaptive MIN_SNR, spatial binning, Voronoi tessellation, log-rebinning, wavelength window determination, LSF preparation. Returns a `dict` or `None`. |
| `ifu_ExGal(...)` | **Stage 2**: Owns the loop over all targets. Calls `ifu_ExGal_prepare` + fitting modules + L2merge for every GALAXY/QSO row. |
| `make_patch_array(...)` | Convenience constructor: builds a single-row patch dict for debug / single-target mode. |
| `ppxf_limits(...)` | Redshift-dependent rest-frame wavelength window for pPXF and EMIPPXF. Handles low-z fixed window and high-z adaptive window transparently. |
| `find_nearest_snr(...)` | Rounds a target SNR value to the nearest reference value in `[20, 30, 40]`. |
| `test_patch_table(...)` | Health-check function (re-exported from `aps_ifu_utils`). |

The command-line entry point is `exgal_runner()`, called by `aps_ExGal_worker.py`.

---

## 4. The Two-Stage Architecture

Understanding why the module is split into `ifu_ExGal_prepare` and `ifu_ExGal` is important before reading the details.

**`ifu_ExGal_prepare`** does everything up to (and including) writing the binned spectra to disk. It knows nothing about the patch table or what other targets exist. It takes raw L1 files and a single redshift and returns a `dict` containing the assembled cube, configuration, and LSF interpolators. If no valid spaxels exist in the aperture it returns `None`.

**`ifu_ExGal`** owns the outer loop. It reads the patch file, validates it, iterates over every GALAXY/QSO row, calls `ifu_ExGal_prepare` for each one, and then hands the result to the fitting modules. If `ifu_ExGal_prepare` returns `None`, the target is skipped cleanly. If a fitting module raises an exception, the error is caught, logged, and the loop continues with the next target.

This separation means:

- You can call `ifu_ExGal_prepare` directly for a single target without building a patch file (use `make_patch_array` and `patch_array=` mode)
- You can test preparation quality without running the slow fitting modules (`PPXF=False, EMIPPXF=False, LS=False`)
- Fitting module failures do not abort the entire observation — other targets in the same field still get processed

---

## 5. Stage 1 — ifu_ExGal_prepare in Detail

### 5.1 Loading the configuration

The first thing `ifu_ExGal_prepare` does is load the JSON config file specified by `IFU_params` (e.g. `LIFULR11.json`). All scientific parameters — spatial bin size, Voronoi target SNR, pPXF wavelength window, SSP library name, etc. — come from this file. The function then merges in the spectral processing flags from the function arguments (these override or supplement the JSON values).

### 5.2 Building the APSOB and spaxel cube

`gen_targlist()` is called first to determine which spaxels fall within the target aperture. This applies the `area` (the elliptical patch aperture), `mask_areas` (type=M exclusion regions from the patch table), `aps_ids`, `targsrvy`, and `targclass` filters simultaneously.

If zero spaxels survive, the function returns `None` immediately and the calling loop skips this target.

Otherwise, `APSOB` is constructed with all the spectral processing parameters. APSOB applies (in order): NaN masking, gap masking, safe gap masking, telluric masking, sensitivity correction, IVAR normalisation across arms, arm joining, and LSF loading. See `APSOB_README.md` for full details of every processing step.

The spaxel cube is then assembled from the APSOB output into a Python dict with arrays of shape `(n_spaxels,)` for scalar quantities and `(n_wave, n_spaxels)` for spectra:

```python
cube = {
    "aps_id":     int32 array of spaxel IDs
    "x", "y":     float offset coordinates in arcsec from field centre
    "z", "zerr":  target redshift and uncertainty (same for all spaxels)
    "healpix":    HEALPix cell index for each spaxel
    "x_0", "y_0": field centre coordinates (RA, Dec) in degrees
    "signal":     broadband mean flux per spaxel
    "noise":      RMS noise per spaxel
    "snr":        broadband SNR per spaxel
    "spec":       (n_wave, n_spaxels) — flux array (from IVAR-masked data)
    "error":      (n_wave, n_spaxels) — error array (1/sqrt(IVAR))
    "wave":       rest-frame wavelength array = observed / (1 + z)
    "velscale":   velocity scale in km/s (for log-rebinning)
    "pixelsize":  IFU pixel size in arcsec
    "pixel_scale": wavelength pixel size in Å
    "targid":     WEAVE TARGID strings
    "cname":      WEAVE CNAME strings
}
```

**Important**: the wavelength array stored in the cube is already in the **rest frame**: `cube["wave"] = observed_wave / (1 + z)`. This is what pPXF operates on. The IVAR conversion to error (`espec = 1/sqrt(IVAR)`) uses a floor of `ivar_mask_value = 1 / large_error²` to prevent division by zero in masked pixels — this sets the error to a very large value (effectively infinity) rather than zero for bad pixels.

### 5.3 pPXF and EMI wavelength window

Immediately after cube assembly, `ppxf_limits()` is called twice — once for the stellar kinematics window (pPXF/LS) and once for the emission line window (EMIPPXF). The observed wavelength range used is taken from the actual data (`targs[0].spectra[0].wave`) rather than hardcoded instrument limits, so it correctly reflects any wavelength clipping applied by `wlranges`.

The resolved limits are written into `configs`:

```python
configs["LMIN_PPXF"] = lmin_ppxf    # or None if window too narrow
configs["LMAX_PPXF"] = lmax_ppxf
configs["SKIP_PPXF"] = True/False
configs["LMIN_EMI"]  = lmin_emi
configs["LMAX_EMI"]  = lmax_emi
configs["SKIP_EMI"]  = True/False
```

See [Section 12](#12-wavelength-window-selection--ppxf_limits) for a full description of the window logic.

### 5.4 Velocity scale

If `VELSCALE` in the config is `"None"` or empty, it is auto-computed from the rest-frame wavelength array:

```
velscale = c × Δlog(λ)
```

This is the standard pPXF convention: a constant velocity step in log-wavelength space. The computation uses the exact endpoints of the wavelength array to ensure the derived velocity scale matches the log-rebinned spectrum precisely.

### 5.5 Adaptive MIN_SNR

`adaptive_min_snr()` (from `aps_ifu_utils`) is called to derive a data-driven threshold for removing bad spaxels. The function looks at the distribution of positive SNR values and returns the Nth percentile (default N=2), clamped between `MIN_SNR_FLOOR=0.1` and `MIN_SNR_CEILING=1.5`. The purpose is to remove dead fibres, negative-flux artefacts, and IFU edge effects — not to remove scientifically faint spaxels (that is the job of the SB flux filter).

If `MIN_SNR` in the config is set to a float, that value is used directly and the adaptive derivation is skipped. Setting `MIN_SNR: null` in the JSON config enables adaptive mode.

The resolved value is printed:
```
MIN_SNR adaptive: 0.3142  (z=0.0923  N_spaxels=5494  SNR p2=0.314  p50=4.821  p98=31.4)
```

### 5.6 Path A — with spatial pre-binning

If `SPBIN_SIZE_EXGAL > 0` in the config, spaxels are first grouped into square spatial bins of side `SPBIN_SIZE_EXGAL` arcsec before Voronoi tessellation. This reduces the number of input points for Voronoi and co-adds nearby spaxels to boost S/N.

The binning is done by `ExGalPrepare.spatial_bin_with_provenance()`, which optionally applies a surface brightness flux filter before binning. The flux filter removes low-SB outer spaxels using one of four modes (`safe`, `transition`, `percentile`, `none`) controlled by `SB_FILTER_MODE` in the config.

After binning, `define_voronoi_bins()` runs Voronoi tessellation on the spatial bin centres, targeting `TARGET_SNR` per bin. If `TARGET_SNR` is `"None"` in the config, it is auto-set to the nearest reference value in `[20, 30, 40]` based on the 90th percentile of the spatial-bin SNR distribution.

The `log_rebinning()` step converts the spatial-bin spectra from linear to logarithmic wavelength scale. This is required by pPXF, which fits templates in velocity space (constant log-λ step = constant Δv).

### 5.7 Path B — no spatial pre-binning

If `SPBIN_SIZE_EXGAL <= 0`, Voronoi tessellation runs directly on the raw spaxels. `rejectDefunctSpaxels_applySNRThreshold()` applies the `MIN_SNR` cut and identifies which spaxels are inside (`idx_inside`) and outside (`idx_outside`) the valid region. Voronoi then bins the inside spaxels.

### 5.8 Diagnostic plots

After both paths converge, `plot_all()` from `ExGalPrepare_plots` generates the full preparation diagnostic suite: raw SNR map, MIN_SNR cut figure, spatial binning figure (4 panels: bin SNR, flux filter on raw spaxels, SB filter on bins, dual histogram), Voronoi overview, and the 5-panel SNR chain figure. See `IFU_PIPELINE_README.md` Section 13 for descriptions of each figure.

### 5.9 LSF interpolators

Two LSF objects are returned:

**`LSF_Data`**: the redshift-corrected FWHM interpolation function for the data. Created by `apply_redshift_to_fwhm_corrected()` which divides both the wavelength grid and the FWHM values by `(1 + z)`, converting from observed to rest frame. pPXF uses this to convolve templates to the observed data resolution at each wavelength.

**`LSF_Templates`**: a simple `interp1d` function loaded from the `LSF-Config_<SSP_LIB>` text file in `IFU_config_dir`. This describes the resolution of the SSP template library, which pPXF needs to apply the correct convolution kernel.

---

## 6. Stage 2 — ifu_ExGal Loop in Detail

### 6.1 Patch table resolution

`ifu_ExGal` accepts two mutually exclusive ways to specify the targets:

**`patch_file` mode**: loads a FITS patch table from disk using `load_and_split_patch_file()`. The table is multi-class split (each class gets the right module) and `ctarg_excluded` is derived from the file content. This is the normal production mode.

**`patch_array` mode**: a single-row dict is injected directly, bypassing the file. This is the debug / test mode. `ctarg_excluded` is forced to `False` and ignored. Use `make_patch_array()` to build the dict.

### 6.2 Target selection within the loop

For each row in the patch table, the loop:

1. Skips rows with `type = 'M'` (mask regions — these are spatial exclusion zones, not targets)
2. Skips rows where `CLASS` is not `GALAXY` or `QSO` — these go to `aps_Gal_worker` instead
3. Skips rows where `Z` is NaN (no redshift available)

For valid targets, the patch aperture is constructed from the row's `RA_icrs`, `DEC_icrs`, `A_world`, `B_world`, and `angle` columns. For single large targets with `ctarg_excluded=True`, the aperture radius is enhanced by a factor of 3 to ensure the full galaxy extent is captured.

Mask regions (type=M rows) from the same patch table are passed as `mask_areas` to `ifu_ExGal_prepare`, which forwards them to APSOB for spatial exclusion.

### 6.3 Error handling

Each target's processing is wrapped in a try/except block. If `ifu_ExGal_prepare` returns `None` (no valid spaxels), the target is skipped cleanly. If any fitting module raises an exception, the full traceback is printed and the loop continues with the next target. A `fault_counter` tracks failures, and if all targets fail, the pipeline exits with an error.

---

## 7. Running from the Command Line

The command-line entry point is `aps_ExGal_worker.py`. It parses arguments and calls `exgal_runner()` → `ifu_ExGal()`.

### Minimal run — preparation only, no fitting

Check the spatial binning and Voronoi quality before committing to full fitting:

```bash
python aps_ExGal_worker.py \
    --infiles $PYAPS_DATA/L1/20240808/stackcube_3071431.fit \
              $PYAPS_DATA/L1/20240808/stackcube_3071430.fit \
    --headname stackcube_3071431__stackcube_3071430 \
    --outpath  $PYAPS_DATA/L2/20240808/11182/ \
    --patch_file $PYAPS_DATA/L2/20240808/11182/stackcube_..._targets_mod.fits \
    --IFU_config_dir  <PYAPS_DIR>/configs/ExGal_configs/ \
    --ExGal_templates <PYAPS_DIR>/PyAPS_templates/templates_ExGal/ \
    --IFU_params      <PYAPS_DIR>/configs/ExGal_configs/LIFULR11.json \
    --PPXF False --EMIPPXF False --LS False \
    --wlranges 3800.0,5950.0 5900.0,9280.0 \
    --arms_ratio 1.0,1.0 \
    --sens_corr True --mask_gaps True --safe_mask_gaps True \
    --tellurics True --join_arms True \
    --caldir $PYAPS_DATA/CAL --catdir $PYAPS_DATA/CAT
```

Inspect the figures in `figs_ExGal/` before proceeding.

### Full run with all fitting modules

```bash
python aps_ExGal_worker.py \
    --infiles $PYAPS_DATA/L1/20240808/stackcube_3071431.fit \
              $PYAPS_DATA/L1/20240808/stackcube_3071430.fit \
    --headname stackcube_3071431__stackcube_3071430 \
    --outpath  $PYAPS_DATA/L2/20240808/11182/ \
    --patch_file $PYAPS_DATA/L2/20240808/11182/stackcube_..._targets_mod.fits \
    --IFU_config_dir  <PYAPS_DIR>/configs/ExGal_configs/ \
    --ExGal_templates <PYAPS_DIR>/PyAPS_templates/templates_ExGal/ \
    --IFU_params      <PYAPS_DIR>/configs/ExGal_configs/LIFULR11.json \
    --PPXF True --EMIPPXF True --LS True \
    --mp_ExGal 6 \
    --wlranges 3800.0,5950.0 5900.0,9280.0 \
    --arms_ratio 1.0,1.0 \
    --sens_corr True --mask_gaps True --safe_mask_gaps True \
    --tellurics True --join_arms True \
    --caldir $PYAPS_DATA/CAL --catdir $PYAPS_DATA/CAT
```

### Single-target debug mode using `--patch_array`

Process one specific galaxy without a patch file:

```bash
python aps_ExGal_worker.py \
    --infiles $PYAPS_DATA/L1/20240808/stackcube_3071431.fit \
              $PYAPS_DATA/L1/20240808/stackcube_3071430.fit \
    --headname stackcube_3071431__stackcube_3071430 \
    --outpath  $PYAPS_DATA/L2/20240808/11182/ \
    --IFU_config_dir  <PYAPS_DIR>/configs/ExGal_configs/ \
    --ExGal_templates <PYAPS_DIR>/PyAPS_templates/templates_ExGal/ \
    --IFU_params      <PYAPS_DIR>/configs/ExGal_configs/LIFULR11.json \
    --patch_array "1,185.198164,58.092634,101.52,47.25,0.0923,0.0001,GALAXY" \
    --PPXF True --EMIPPXF True --LS False \
    --wlranges 3800.0,5950.0 5900.0,9280.0 \
    --arms_ratio 1.0,1.0 \
    --sens_corr True --mask_gaps True --safe_mask_gaps True \
    --tellurics True --join_arms True \
    --caldir $PYAPS_DATA/CAL --catdir $PYAPS_DATA/CAT
```

`--patch_array` and `--patch_file` are mutually exclusive. The format is fixed-order, comma-separated:

```
"id,RA_deg,Dec_deg,A_arcsec,B_arcsec,Z,ZERR,CLASS"
```

---

## 8. Running from Python

### Full pipeline from Python

```python
from PyAPS.aps_ifu_exgal import ifu_ExGal

ifu_ExGal(
    infiles         = ["<PYAPS_DATA>/L1/20240808/stackcube_3071431.fit",
                       "<PYAPS_DATA>/L1/20240808/stackcube_3071430.fit"],
    headname        = "stackcube_3071431__stackcube_3071430",
    outpath         = "<PYAPS_DATA>/L2/20240808/11182/",
    IFU_config_dir  = "<PYAPS_DIR>/configs/ExGal_configs/",
    ExGal_templates = "<PYAPS_DIR>/PyAPS_templates/templates_ExGal/",
    IFU_params      = "<PYAPS_DIR>/configs/ExGal_configs/LIFULR11.json",
    patch_file      = "<PYAPS_DATA>/L2/20240808/11182/stackcube_..._targets_mod.fits",
    PPXF=True, EMIPPXF=True, LS=True,
    nthreads        = 6,
    wlranges        = [[3800.0, 5950.0], [5900.0, 9280.0]],
    arms_ratio      = [1.0, 1.0],
    sens_corr=True, mask_gaps=True, safe_mask_gaps=True,
    tellurics=True, join_arms=True,
    caldir="<PYAPS_DATA>/CAL", catdir="<PYAPS_DATA>/CAT",
)
```

### Single target via `make_patch_array`

```python
from PyAPS.aps_ifu_exgal import ifu_ExGal, make_patch_array

patch = make_patch_array(
    row_id    = 1,
    ra        = 185.198164,
    dec       = 58.092634,
    a_arcsec  = 101.52,
    b_arcsec  = 47.25,
    z         = 0.0923,
    zerr      = 0.0001,
    class_str = "GALAXY",
)

ifu_ExGal(
    infiles         = ["<PYAPS_DATA>/L1/20240808/stackcube_3071431.fit",
                       "<PYAPS_DATA>/L1/20240808/stackcube_3071430.fit"],
    headname        = "stackcube_3071431__stackcube_3071430",
    outpath         = "<PYAPS_DATA>/L2/20240808/11182/",
    IFU_config_dir  = "<PYAPS_DIR>/configs/ExGal_configs/",
    ExGal_templates = "<PYAPS_DIR>/PyAPS_templates/templates_ExGal/",
    IFU_params      = "<PYAPS_DIR>/configs/ExGal_configs/LIFULR11.json",
    patch_array     = patch,
    PPXF=True, EMIPPXF=True, LS=False,
    nthreads=6,
    wlranges=[[3800.0, 5950.0], [5900.0, 9280.0]],
    arms_ratio=[1.0, 1.0],
    sens_corr=True, mask_gaps=True, safe_mask_gaps=True,
    tellurics=True, join_arms=True,
    caldir="<PYAPS_DATA>/CAL", catdir="<PYAPS_DATA>/CAT",
)
```

### Calling `ifu_ExGal_prepare` directly

If you only want the preparation stage output (cube, binning, Voronoi) without running pPXF:

```python
from PyAPS.aps_ifu_exgal import ifu_ExGal_prepare

prep = ifu_ExGal_prepare(
    infiles        = ["<PYAPS_DATA>/L1/20240808/stackcube_3071431.fit",
                      "<PYAPS_DATA>/L1/20240808/stackcube_3071430.fit"],
    headname       = "stackcube_3071431__stackcube_3071430_P0001",
    IFU_params     = "<PYAPS_DIR>/configs/ExGal_configs/LIFULR11.json",
    outpath        = "<PYAPS_DATA>/L2/20240808/11182/",
    IFU_config_dir = "<PYAPS_DIR>/configs/ExGal_configs/",
    z_input        = [0.0923, 0.0001],
    area           = [185.198164, 58.092634, 101.52, 47.25, 0.0],
    wlranges       = [[3800.0, 5950.0], [5900.0, 9280.0]],
    arms_ratio     = [1.0, 1.0],
    join_arms      = True,
    sens_corr=True, mask_gaps=True, safe_mask_gaps=True,
    tellurics=True,
    caldir="<PYAPS_DATA>/CAL",
)

if prep is not None:
    cube          = prep["cube"]
    configs       = prep["configs"]
    LSF_Data      = prep["LSF_Data"]
    LSF_Templates = prep["LSF_Templates"]
    figdir        = prep["figdir"]

    print("N spaxels in cube:   ", cube["spec"].shape[1])
    print("Wavelength range:    ", cube["wave"][[0, -1]])
    print("Velocity scale:      ", cube["velscale"], "km/s")
    print("pPXF window:         ", configs["LMIN_PPXF"], "–", configs["LMAX_PPXF"], "Å rest")
    print("MIN_SNR applied:     ", configs["MIN_SNR"])
    print("TARGET_SNR resolved: ", configs["TARGET_SNR"])
```

### Checking ppxf_limits for any redshift

```python
from PyAPS.aps_ifu_exgal import ppxf_limits

for z in [0.0, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 5.0]:
    lmin, lmax = ppxf_limits(
        z,
        lowz_lmin=3000., lowz_lmax=6000.,
        lib_lmin=1200., lib_lmax=10000.,
    )
    if lmin is None:
        print("z=%.1f  →  window too narrow, PPXF skipped" % z)
    else:
        print("z=%.1f  →  pPXF window: %.0f – %.0f Å rest  (%.0f Å)" % (
            z, lmin, lmax, lmax - lmin))
```

---

## 9. Input Parameters — Complete Reference

### `ifu_ExGal` / `ifu_ExGal_prepare` shared parameters

These are the same parameters used by APSOB (see `APSOB_README.md`). They are forwarded unchanged.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `infiles` | list of str | required | L1 FITS files. Typically two (blue + red arm). |
| `headname` | str | required | Root name for all output files. In `ifu_ExGal`, the patch ID is appended: `<headname>_P<NNNN>`. |
| `IFU_params` | str | required | Path to the JSON config file (e.g. `LIFULR11.json`). |
| `outpath` | str | required | Output directory. |
| `IFU_config_dir` | str | required | Directory containing LSF-Config files and config templates. |
| `wlranges` | list or None | `None` | Wavelength ranges per arm. |
| `aps_ids` | list or None | `None` | Restrict to these APS IDs. |
| `targsrvy` | list or None | `None` | Survey filter (e.g. `['WA']`). |
| `targclass` | list or None | `None` | Class filter (e.g. `['GALAXY']`). |
| `mask_aps_ids` | list or None | `None` | Spaxels to exclude. |
| `area` | list or None | `None` | Elliptical aperture `[RA, Dec, A_arcsec, B_arcsec, angle]`. |
| `mask_areas` | list or None | `None` | List of elliptical apertures to exclude. |
| `sens_corr` | bool | `True` | Apply flux sensitivity correction. |
| `mask_gaps` | bool | `True` | Mask inter-arm gaps. |
| `safe_mask_gaps` | bool | `True` | Apply safe gap masking from lookup table. |
| `vacuum` | bool | `False` | Convert wavelengths to vacuum. |
| `tellurics` | bool | `True` | Mask telluric absorption bands. |
| `fill_gap` | bool | `False` | Interpolate across masked gaps. |
| `arms_ratio` | list or None | `None` | Flux scale per arm (legacy, used if `normalize_ivar=False`). |
| `join_arms` | bool | `False` | Stitch blue and red arms. |
| `z_input` | [z, zerr] | required | Target redshift and uncertainty. |
| `catdir` | str or None | `None` | Catalogue directory. |
| `caldir` | str or None | `None` | Calibration directory. |
| `no_spec_ext` | bool | `False` | if True, it discards the spec relaetd extensions from the final APS file |
| `spaxel_weighted_lsf` | bool or None | `None` | Per-bin resolution (see [Section 11](#11-spaxel-weighted-lsf-and-resolution-buckets)). `None` resolves from each patch's own `SPAXEL_WEIGHTED_LSF` config key (default **on** as of v1.9); pass `True`/`False` to override every patch's own config. |

### `ifu_ExGal`-only parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `patch_file` | str or None | `None` | Path to the patch FITS table from `aps_ifu_prepare`. |
| `patch_array` | dict or None | `None` | Single injected target row (debug mode). Mutually exclusive with `patch_file`. |
| `PPXF` | bool | `False` | Run pPXF stellar kinematics module. |
| `EMIPPXF` | bool | `False` | Run EMIPPXF emission line fitting module. |
| `LS` | bool | `False` | Run line strength index module. |
| `nthreads` | int | `1` | Number of parallel threads for fitting modules. |
| `overwrite` | bool | `True` | Overwrite existing output files. |
| `UAPSID` | str or None | `None` | Unique APS identifier written to L2 FITS header. |

### Command-line-only parameters

| Argument | Type | Default | Description |
|---|---|---|---|
| `--mp_ExGal` | int | `1` | Same as `nthreads` — number of threads for fitting. |
| `--patch_array` | str or None | `None` | Single target as comma-separated string: `"id,RA,Dec,A,B,Z,ZERR,CLASS"`. |

---

## 10. Configuration File Reference

The JSON config file (`LIFULR11.json`, `LIFUHR11.json`, etc.) controls all scientific parameters. Below are the keys relevant to `aps_ifu_exgal`.

### Spatial binning

```json
"PIXELSIZE"              : 0.0,
"SPBIN_SIZE_EXGAL"       : 1.0,
"SB_FILTER"              : 1,
"SB_FILTER_MODE"         : "percentile",
"SB_FILTER_PERCENT"      : 10,
"SB_FILTER_MIN_FRAC"     : 0.01,
"SB_FILTER_SNR_MIN"      : 3.0,
"SB_FILTER_DELTA"        : 2.0,
"SB_FILTER_MIN_SPAXELS"  : 10
```

`SPBIN_SIZE_EXGAL` controls the side length of square spatial bins in arcsec. Set to `-1` to skip spatial pre-binning and run Voronoi directly on raw spaxels. `SB_FILTER=1` enables the surface brightness filter that removes low-flux outer spaxels before binning. `SB_FILTER_MODE` options: `"percentile"` (recommended), `"transition"`, `"safe"`, `"none"`.

### Voronoi tessellation

```json
"VORONOI"       : 1,
"TARGET_SNR"    : "None",
"COVAR_VOR"     : 0.0
```

`VORONOI=1` enables adaptive Voronoi tessellation, using the **PowerBin** engine (Cappellari 2025) as of v1.9 — see [Section 11](#11-spaxel-weighted-lsf-and-resolution-buckets) for the binning-engine note and the resolution-bucketing mechanism below. `VORONOI=0` treats each spatial bin as its own Voronoi bin. `TARGET_SNR: "None"` triggers automatic derivation from the data. `COVAR_VOR` corrects for spatial noise correlations (Garcia-Benito et al. 2015); set to `0.0` to disable.

### Spaxel-weighted LSF

```json
"SPAXEL_WEIGHTED_LSF"  : 1,
"LSF_N_BUCKETS"         : 6
```

**Default on as of v1.9** (every bundled `LIFU*`/`MIFU*` config sets `SPAXEL_WEIGHTED_LSF: 1` explicitly). Set to `0` to fall back to the pre-v1.9 behaviour: a single flat global LSF/FWHM curve for the whole patch. `LSF_N_BUCKETS` caps how many distinct resolution "buckets" template preparation ever computes, regardless of how many Voronoi/PowerBin bins the patch has — see [Section 11](#11-spaxel-weighted-lsf-and-resolution-buckets) for what this actually does and why it exists.

### Adaptive MIN_SNR

```json
"MIN_SNR"            : null,
"MIN_SNR_METHOD"     : "percentile",
"MIN_SNR_PERCENTILE" : 2,
"MIN_SNR_FLOOR"      : 0.1,
"MIN_SNR_CEILING"    : 1.5
```

`MIN_SNR: null` enables adaptive derivation. Set to a float to fix the threshold. The adaptive approach uses the 2nd percentile of the positive-SNR distribution, clamped to `[0.1, 1.5]`. The low ceiling (1.5) protects emission-line-only targets at high redshift where the broadband continuum SNR may be below 1 even for real signal.

### pPXF wavelength window

```json
"LMIN_PPXF"       : 3000.0,
"LMAX_PPXF"       : 6000.0,
"LMIN_EMI"        : 2000.0,
"LMAX_EMI"        : 8600.0,
"PPXF_LIB_LMIN"   : 1200.0,
"PPXF_LIB_LMAX"   : 10000.0,
"VELSCALE"        : "None"
```

`LMIN_PPXF` and `LMAX_PPXF` define the preferred low-z optical window. These values are also used as the Ca H&K blue clamp at intermediate redshift. At high redshift the window adapts automatically. `PPXF_LIB_LMIN/LMAX` are the hard limits of your template library.

### Fitting parameters

```json
"SSP_LIB"     : "MILES",
"LS_MODE"     : "default",
"LS_RES"      : "ADAPTED"
```

`SSP_LIB` determines which `LSF-Config_<SSP_LIB>` file is loaded from `IFU_config_dir`. Must match the name of your template library.

---

## 11. Spaxel-Weighted LSF and Resolution Buckets

### The problem this solves

Every spaxel in a WEAVE IFU cube is a *regrid* of several different physical fibres, from different dithered single exposures, each landing on that patch of sky at a slightly different offset. Each of those contributing fibres has its own, real, individually-measured instrumental resolution (LSF) — physically, fibres at different positions along the spectrograph slit smear light out by slightly different amounts. Before v1.9, none of that mattered: every spaxel, every Voronoi bin, the whole patch — all of it used one single flat *global* LSF/FWHM curve (the median across every fibre in the instrument), regardless of which specific fibres actually contributed to a given spot on the sky.

`spaxel_weighted_lsf` (the underlying `aps_utils.APSOB` feature; see `aps_ifu_spaxel_contrib.py`) fixes the root of this: for each spaxel, it works out which real, physical fibres (from the underlying `single_*.fit` exposures) geometrically overlap that spaxel, and how much — via a real circle-overlap-area calculation between each fibre's true footprint (1.3″ for MOS/mIFU fibres, 2.6″ for LIFU fibres) and the spaxel's own footprint (from the cube's real WCS pixel scale) — then combines those fibres' own already-measured LSF curves, weighted by that overlap.

### From spaxels to bins: `aggregate_bin_lsf`

`aps_ifu_Gal.py`/`aps_ifu_ExGal.py` don't fit one spectrum per spaxel — they fit one spectrum per **Voronoi/PowerBin bin** (several spaxels combined together for signal-to-noise). So the per-spaxel LSF above needs one more step: for each bin, average its member spaxels' own per-spaxel LSF curves, weighted by flux — the same way the bin's own *spectrum* is combined (`ExGalPrepare.voronoi_binning` sums member flux, so a brighter spaxel already dominates the bin's spectral shape more; its resolution dominates the assumed bin resolution the same way). This composes correctly even when spatial pre-binning (`SPBIN_SIZE_EXGAL`) is active: the flux-weighted average is computed straight from the original raw spaxels through to the final bin, not through an intermediate "spatial bin's own LSF" step.

If `SPAXEL_WEIGHTED_LSF=0`, every spaxel already shares the same global curve, so this averaging step is a mathematical no-op — you get back exactly the old flat global value, at every bin, with no special-cased code path needed to guarantee that.

### Why "buckets"? — the actual cost being managed

Having a real, distinct LSF value per bin is one thing; *using* it is another. Fitting a bin's spectrum with pPXF requires the stellar template library to be convolved down to match that bin's own resolution — a real, non-trivial computation (resampling every template spectrum in the library). Doing that separately for every single bin would mean, for a patch with (say) 200 Voronoi bins, 200 separate template-preparation passes instead of 1 — turning a cheap step into one of the most expensive parts of the whole fit.

A **bucket** is a small group of bins whose LSF curves are close enough to be treated as identical for this purpose. Instead of convolving the template library once per bin, `bucket_lsf_curves` (k-means clustering on each bin's own LSF-curve shape) groups bins into at most `LSF_N_BUCKETS` buckets (default 6), and template preparation runs **once per bucket**, not once per bin — every bin in a bucket reuses that one already-prepared template set.

Two things worth knowing about how this behaves:

- **If a patch has fewer bins than `LSF_N_BUCKETS`, nothing gets grouped at all** — every bin gets its own bucket, i.e. genuinely exact per-bin resolution with zero approximation. Bucketing only ever kicks in when there are more bins than the cap.
- **Raising `LSF_N_BUCKETS` trades cost for fidelity**: more buckets means closer to true per-bin resolution, at the cost of more template-preparation passes. The default of 6 was chosen because bin-to-bin LSF variation *within one patch's own small footprint* is a mild, smoothly-varying effect (the real variation across the ~960-fibre focal plane is much larger than what one patch's own handful of dithers ever spans), so a handful of buckets already captures nearly all of the real signal a full per-bin convolution would.

### What actually changes when you turn it on

Real, measured example (a 111-spaxel aperture, 8 Voronoi/PowerBin bins, default 6 buckets): the 6 buckets' own LSF values spread from about 2.84 to 2.90 Å around a global value of 2.84 Å — a small (~1.5%) but real difference. Feeding that through `runModule_PPXF` produced small, physically sensible shifts in fitted velocity and velocity dispersion (a few to ~20 km/s) in the bins with real signal, versus the flag-off baseline. Overhead measured on that same patch: negligible at the `ifu_ExGal_prepare` stage (the geometry/aggregation/bucketing computation is cheap relative to loading the L1 cube itself), and about +8% at the `runModule_PPXF` stage for this small test case specifically — expect that relative overhead to shrink on larger, more typical patches, since the *fixed* cost of `LSF_N_BUCKETS` template preparations gets amortized over more per-bin fits.

`IFUExGalPPXF.py`/`IFUExGalEMIPPXF.py`/`IFUExGalLS.py` (and `aps_ifu_rvs.py`/`aps_ifu_ferre.py` for `aps_ifu_Gal.py`) all consume this the same way: each already loops over bins internally for its own fit, so per-bucket resolution just means picking the right already-prepared template/LSF for a given bin's own bucket inside that existing loop — not a new fitting stage.

---

## 12. The Patch Array — Single Target Mode

The `make_patch_array()` function builds a minimal patch row dict that can be passed directly to `ifu_ExGal` or `ifu_ExGal_prepare` without a patch file on disk. This is the fastest way to test the pipeline on a specific target.

```python
from PyAPS.aps_ifu_exgal import make_patch_array

patch = make_patch_array(
    row_id    = 1,           # integer ID (used in output file names: _P0001)
    ra        = 185.198164,  # degrees ICRS
    dec       = 58.092634,   # degrees ICRS
    a_arcsec  = 101.52,      # semi-major axis (arcsec)
    b_arcsec  = 47.25,       # semi-minor axis (arcsec)
    z         = 0.0923,      # spectroscopic redshift
    zerr      = 0.0001,      # redshift uncertainty
    class_str = "GALAXY",    # 'GALAXY', 'QSO', 'STAR', etc.
    zwarn     = 0,           # Redrock warning flag (default 0 = clean)
    angle     = 0.0,         # position angle degrees east of north
    row_type  = "T",         # 'T'=target, 'C'=central, 'M'=mask
)
```

The resulting dict can be passed as `patch_array=patch` to `ifu_ExGal`, or as the `--patch_array` command-line argument using the string format `"id,RA,Dec,A,B,Z,ZERR,CLASS"`.

---

## 13. Wavelength Window Selection — ppxf_limits

`ppxf_limits()` is one of the most important functions in this module. It computes the rest-frame wavelength window that pPXF and EMIPPXF will use, adapting continuously with redshift so that no hard `HZ_*` switch is needed.

### The two regimes

**Low-z regime (z < z_thresh)**: the preferred optical window `[LMIN_PPXF, LMAX_PPXF]` from the config is always fully observable. It is used exactly as specified. For typical WEAVE + LIFULR11.json this means 3000–6000 Å rest-frame, covering Ca H&K, G-band, Hβ, Mg b, and Fe lines.

The threshold is computed from the data, not hardcoded:
```
z_thresh = obs_lmax / lowz_lmax - 1  =  9280 / 6000 - 1  =  0.547
```
This is the redshift at which the red end of your preferred window shifts beyond the WEAVE red cutoff.

**High-z regime (z ≥ z_thresh)**: the window adapts to the intersection of WEAVE's observed coverage and the template library limits. A Ca H&K blue clamp prevents the window opening into the UV below 3000 Å when Ca H&K is still observable (z ≤ 1.0). Above z = 2 a Lyα forest floor (`lmin ≥ 1266 Å rest`) is applied.

### Window examples for default LIFULR11.json settings

| z | pPXF window | Dominant features |
|---|---|---|
| 0.0 | 3000–6000 Å | Ca H&K, Mg b, Fe lines |
| 0.3 | 3000–6000 Å | same |
| 0.6 | 3000–5750 Å | Ca H&K, Hβ, Mg b (red trimmed) |
| 1.0 | 3000–4590 Å | Ca H&K, G-band, Hβ |
| 1.5 | 1570–3662 Å | Balmer continuum, Fe/Mg UV blend |
| 2.0 | 1317–3043 Å | UV continuum, MgII 2800 approaching |
| 2.5 | 1266–2601 Å | Lyα floor applies |
| 4.5 | None | window < 500 Å, PPXF skipped |

### Behaviour on window failure

If `ppxf_limits()` returns `(None, None)`, `ifu_ExGal_prepare` sets `configs["SKIP_PPXF"] = True` and prints a warning. The preparation stage still completes successfully — the cube, Voronoi bins, and BINSpectra.fits are all written. The fitting modules check `SKIP_PPXF` / `SKIP_EMI` and skip themselves gracefully.

---

## 14. Adaptive MIN_SNR

The adaptive minimum SNR algorithm in `ifu_ExGal_prepare` serves a specific and limited purpose: **remove technically bad spaxels only**. It is not meant to perform science-driven edge filtering (that is the job of the SB flux filter `SB_FILTER`).

Bad spaxels that should be cut:
- Dead or damaged fibres (SNR ≈ 0 or negative)
- Cosmic ray residuals with negative flux
- IFU edge spaxels with near-zero throughput

Real signal that must be preserved:
- Low-SB outer galaxy regions (SNR 0.3–1.0 in broadband)
- High-z targets with emission lines but no detected continuum (broadband SNR < 1)

The algorithm operates on the **positive-SNR distribution only** — non-positive SNR is definitionally bad data and is always removed regardless of the threshold. The 2nd percentile of the positive-SNR distribution is taken as the threshold, then clamped to `[MIN_SNR_FLOOR, MIN_SNR_CEILING]`.

The low ceiling of 1.5 for ExGal (compared to 3.0 for Gal) reflects the fact that emission-line-dominated targets at high redshift can have real broadband SNR below 1.0. Setting the ceiling too high would incorrectly remove these targets.

To disable adaptive MIN_SNR and use a fixed value, set `"MIN_SNR": 1.0` (or any float) in the JSON config.

---

## 15. Output Files

For each successfully processed patch, the following files are written to `outpath`:

| File | Description |
|---|---|
| `<headname>_P####_table.fits` | Voronoi bin table: coordinates, bin IDs, per-bin SNR, N spaxels per bin |
| `<headname>_P####_BINSpectra.fits` | Log-rebinned co-added spectra, one per Voronoi bin |
| `<headname>_P####_ppxf.fits` | pPXF stellar kinematics: V, sigma, h3, h4 per bin (if `PPXF=True`) |
| `<headname>_P####_emcee.fits` | EMIPPXF emission line fluxes, widths, EWs (if `EMIPPXF=True`) |
| `<headname>_P####_ls.fits` | Line strength index measurements (if `LS=True`) |
| `<headname>_P####_APS.fits` | Merged L2 product — all results in one FITS file |

Diagnostic figures are written to `outpath/figs_ExGal/`:

| Figure | Content |
|---|---|
| `*_snr_stages.png` | 5-panel SNR chain: raw → MIN_SNR cut → spatial-bin SNR → SB filter → Voronoi |
| `*_prep_raw.png` | Raw SNR, signal, and noise maps |
| `*_prep_snrcut.png` | MIN_SNR keep/remove map + SNR histogram |
| `*_prep_spatialbin.png` | 4-panel: bin SNR, flux filter on raw spaxels, bin signal map, dual histogram |
| `*_prep_voronoi.png` | Voronoi bin IDs, SNR per bin, N spaxels per bin |
| `*_voronoi_cube.png` | 3-panel Voronoi structure overview |
| `*_sbin_<size>.png` | Side-by-side: raw spaxels vs spatial bins |

---

## 16. Controlling output file size — `--no_spec_ext`

By default the final `_APS.fits` product includes spectral extensions (`GALAXY_SPEC`
for ExGal, `STAR_SPEC` for Gal) containing the observed, model, and residual spectra
for every Voronoi bin. For large IFU fields these extensions can make the file very large.

Pass `--no_spec_ext` to omit them:

```bash
python aps_ExGal_worker.py \
    ... \
    --no_spec_ext
```

When `--no_spec_ext` is set:

- The spectral extensions are not written to the `_APS.fits` file, reducing its size significantly
- All parameter and kinematics extensions (`PATCH_TABLE`, `PATCH_BINSPEC`, `GALAXY_TABLE`) are unaffected
- The primary header records `NO_SPEC_EXT = True` so downstream tools can distinguish a deliberate omission from a processing failure
- A warning is printed to stdout confirming the omission

This option is controlled in `ifuExGalL2merge` / `ifuGalL2merge` via the `no_spec_ext`
parameter (default: `False`). Existing pipelines and database-driven runs are unaffected
unless `--no_spec_ext` is explicitly passed.

> **When to use it:** large LIFU fields at high Voronoi bin counts where disk space or
> transfer bandwidth is a concern, or when only the kinematic and line-strength results
> are needed for a quick science check.



## 17. Troubleshooting

### "No valid APS IDs found for this ExGal patch"

No spaxels fall within the patch aperture. Check that the `A_world` and `B_world` values in the patch table are sensible (they are in degrees — a value of 0.028 degrees = 101 arcsec is typical for a LIFU galaxy). Check that the RA/Dec are within the IFU field.

### "No valid spaxels remaining after filters"

After MIN_SNR and/or flux filtering, no spaxels survive.

1. Check `*_prep_raw.png` — is there any real SNR signal?
2. Try setting `"MIN_SNR": 0.0` in the config to bypass the SNR cut.
3. Try `"SB_FILTER": 0` to disable the flux filter.
4. If the SNR map is empty, check that the aperture overlaps with the actual galaxy position.

### pPXF window warning

```
WARNING: z=4.600 — rest-frame pPXF window < 500 Å. PPXF and LS will be skipped.
```

Expected for z > 4.5. EMIPPXF still runs if the EMI window exceeds 200 Å. The preparation stage completes normally and BINSpectra.fits is written.

### All targets fail: "All ExGal targets failed"

Every target in the patch table raised an exception. Check the individual tracebacks printed before the final exit. Common causes:
- Template directory `ExGal_templates` not found or empty
- `IFU_config_dir` missing LSF-Config files
- Config JSON key missing (e.g. `COVAR_VOR` not in LIFULR11.json)
- Memory error during pPXF on very large fields — reduce `mp_ExGal`

### Voronoi produces only one bin

`TARGET_SNR` is too high for the data depth. Set `"TARGET_SNR": "None"` in the config for automatic derivation, or reduce to `10.0` as a test.

### Output figures not generated

Diagnostic plots fail silently (the exception is caught and printed as a warning). Check the `Warning: diagnostic plots failed:` message. Common causes: missing `voronoi_data` (Voronoi table not written yet — check that `define_voronoi_bins` completed), matplotlib backend issue on headless systems (set `matplotlib.use('Agg')` at the top of the module).

---

*For further information, bug reports, or contributions please open an issue on the PyAPS repository.*
