# aps_ifu_gal — Galactic IFU Analysis Module

## A Reference and Tutorial Guide for the PyAPS Pipeline

**Module:** `PyAPS.aps_ifu_gal`
**Entry script:** `aps_Gal_worker.py`
**PyAPS version:** 1.7+
**Applies to:** WEAVE LIFU and MIFU observations, stellar targets
**Science modules:** RVS (radial velocities) + FERRE (stellar parameters)
**Maintained by:** CASU, Institute of Astronomy, University of Cambridge

---

## Table of Contents

1. [What This Module Does](#1-what-this-module-does)
2. [How It Fits in the Pipeline](#2-how-it-fits-in-the-pipeline)
3. [Key Differences from the ExGal Module](#3-key-differences-from-the-exgal-module)
4. [Public API — Functions and Their Roles](#4-public-api--functions-and-their-roles)
5. [The Two-Stage Architecture](#5-the-two-stage-architecture)
6. [Stage 1 — ifu_Gal_prepare in Detail](#6-stage-1--ifu_gal_prepare-in-detail)
7. [Stage 2 — ifu_Gal Loop in Detail](#7-stage-2--ifu_gal-loop-in-detail)
8. [Running from the Command Line](#8-running-from-the-command-line)
9. [Running from Python](#9-running-from-python)
10. [Input Parameters — Complete Reference](#10-input-parameters--complete-reference)
11. [Configuration File Reference](#11-configuration-file-reference)
12. [Aperture Handling — Stars vs Galaxies](#12-aperture-handling--stars-vs-galaxies)
13. [Adaptive MIN_SNR for Gal Mode](#13-adaptive-min_snr-for-gal-mode)
14. [Binning Strategy — Fewer High-SNR Bins](#14-binning-strategy--fewer-high-snr-bins)
15. [FERRE and RVS Integration](#15-ferre-and-rvs-integration)
16. [Output Files](#16-output-files)
17. [Troubleshooting](#17-troubleshooting)

---

## 1. What This Module Does

`aps_ifu_gal` is the Galactic (Gal) IFU analysis module for WEAVE data. It processes all `STAR` and `WD` (white dwarf) targets found in an IFU observation, running the complete stellar analysis chain from raw L1 data to final L2 science products.

For each stellar target in the patch table it:

1. Builds a spaxel cube from the L1 data via APSOB — arms are **always joined** in Gal mode
2. Works in the **observed frame** (wavelengths are not rest-frame corrected)
3. Derives an adaptive minimum SNR threshold to remove bad spaxels, with a higher ceiling than ExGal because stellar continuum is always present
4. Optionally applies **larger** spatial pre-binning than ExGal (typically 3–5 arcsec vs 1–2 arcsec) to produce fewer, higher-SNR bins
5. Runs Voronoi tessellation at a higher target SNR (typically 50–80 vs 20–40 for ExGal)
6. Saves linear (not log-rebinned) spectra for FERRE and RVS
7. Runs RVS for radial velocity measurement
8. Runs FERRE for stellar atmospheric parameter fitting (Teff, log g, [Fe/H], [α/Fe])
9. Merges all results into a single L2 FITS product

---

## 2. How It Fits in the Pipeline

```
L1 FITS files
      │
      ▼
aps_ifu_prepare.py          ←  source detection, Redrock classification,
      │                         patch file creation (see IFU_PIPELINE_README.md)
      │  patch_file (targets_mod.fits)
      │
      ├──────────────────────────────────────────────┐
      │                                              │
      ▼                                              ▼
aps_ExGal_worker.py                        aps_Gal_worker.py    ←  YOU ARE HERE
(GALAXY / QSO rows)                        (STAR / WD rows)
      │                                              │
      │  calls ifu_Gal()                             │
      │      │                                       │
      │      ├── for each STAR/WD row:               │
      │      │       ifu_Gal_prepare()               │
      │      │       run_rvs_for_ifu_gal()           │
      │      │       run_ferre_for_ifu_gal()         │
      │      │       ifuGalL2merge()                 │
      │      │                                       │
      └──────┘                                       │
             │                                       │
             ▼                                       ▼
      L2 ExGal products                     L2 Gal products
      (<headname>_P####_APS.fits)           (<headname>_P####_APS.fits)
```

`aps_ExGal_worker` and `aps_Gal_worker` run in **parallel** in the automated pipeline — both depend on the prepare stage but are independent of each other. They read the same patch file but process different rows based on target class.

---

## 3. Key Differences from the ExGal Module

This is the most important section for anyone migrating from working with the ExGal module. The two modules share the same architecture and many parameters, but differ in fundamental scientific choices.

| Aspect | ExGal | Gal |
|---|---|---|
| Target classes | `GALAXY`, `QSO` | `STAR`, `WD` |
| Observed wavelength frame | Rest-frame: `wave / (1+z)` | Observed frame: `wave` as-is |
| Wavelength scale for fitting | Log-rebinned (pPXF needs constant Δv) | Linear (FERRE and RVS need linear Å) |
| Arms always joined | No — configurable | **Yes — hardcoded `join_arms=True`** |
| Surface brightness filter | Yes (`SB_FILTER=1` typical) | **No — always `apply_flux_filter=False`** |
| MIN_SNR adaptive ceiling | 1.5 (protects emission-line spaxels) | **3.0** (stellar continuum always present) |
| MIN_SNR adaptive percentile | 2 (very gentle) | **3** (slightly more aggressive) |
| MIN_SNR adaptive floor | 0.1 | **0.3** |
| Spatial bin size | Typically 1.0–2.0 arcsec | **Typically 3.0–5.0 arcsec** |
| Target SNR per bin | Typically 20–30 | **Typically 50–80** |
| Fitting modules | pPXF + EMIPPXF + LS | **RVS + FERRE** |
| Aperture radius enhancement | ×3 for single large targets | **×0.5 (shrunk)** — stellar PSF, not extended |
| Wavelength window | Adaptive `ppxf_limits()` | Not applicable — FERRE handles its own range |
| Central target skipping | No | **Yes** — C-type rows skipped when T-type rows exist |

The reduced aperture (×0.5) for Gal targets deserves explanation: stellar targets are point sources. Their IFU apertures from `aps_ifu_prepare` may be set for the extended emission region of a galaxy (if in a mixed field) or to a default minimum size. For stellar analysis you want to concentrate on the central bright spaxels, not include sky-dominated outer regions.

---

## 4. Public API — Functions and Their Roles

| Function | Role |
|---|---|
| `ifu_Gal_prepare(...)` | **Stage 1**: Data ingestion, cube assembly in observed frame, adaptive MIN_SNR (Gal-specific parameters), spatial binning without flux filter, Voronoi tessellation, linear spectra saving, diagnostic plots. Returns a `dict` or `None`. |
| `ifu_Gal(...)` | **Stage 2**: Owns the loop over all STAR/WD targets. Calls `ifu_Gal_prepare` + RVS + FERRE + L2merge. Also handles optional Redrock re-classification for rows with missing Z. |
| `make_patch_array(...)` | Identical interface to `aps_ifu_exgal.make_patch_array`. Builds a single-row patch dict for debug / single-target mode. |
| `find_nearest_snr(...)` | Rounds target SNR to nearest value in `[20, 30, 40]`. |
| `test_patch_table(...)` | Health-check function (re-exported from `aps_ifu_utils`). |

The command-line entry point is `gal_runner()`, called by `aps_Gal_worker.py`.

---

## 5. The Two-Stage Architecture

The design mirrors `aps_ifu_exgal` exactly (see `aps_ifu_exgal_README.md` Section 4):

**`ifu_Gal_prepare`** knows nothing about the patch table. It takes a single star's position, redshift, and L1 files, and returns the assembled cube with binned linear spectra on disk plus the LSF profile. Returns `None` if no valid spaxels are found.

**`ifu_Gal`** owns the outer loop. It reads the patch file, iterates over every STAR/WD row, optionally re-classifies rows with missing redshifts via Redrock, calls `ifu_Gal_prepare`, and then runs RVS and FERRE. Per-target exceptions are caught so one failure does not abort the entire observation.

This means you can call `ifu_Gal_prepare` directly for a single star (with `make_patch_array` in `patch_array=` mode) to test the preparation quality without running the computationally expensive fitting modules.

---

## 6. Stage 1 — ifu_Gal_prepare in Detail

### 6.1 Configuration loading

The JSON config file (`IFU_params`) is loaded first. Binning parameters (`SPBIN_SIZE_GAL`, `VORONOI_GAL`, `TARGET_SNR_GAL`, `MIN_SNR_GAL`) can be overridden by function arguments, which in turn can be set from the command line. The resolution priority is always:

```
CLI argument  >  function argument  >  JSON config value  >  hardcoded default
```

`join_arms` is hardcoded to `True` regardless of what is passed — stars in WEAVE are always analysed with the full joined blue+red spectrum. There is no scientific reason to analyse arms separately for FERRE stellar parameters.

### 6.2 Building the APSOB and spaxel cube

Identical to `ifu_ExGal_prepare` except:

- `join_arms=True` is forced
- No `IFU_config_dir` remapping through `configdir` parameter in APSOB (the configdir is passed explicitly)

The cube is assembled in the same `dict` format as ExGal, with the same keys (`aps_id`, `x`, `y`, `signal`, `noise`, `snr`, `spec`, `error`, etc.). The critical difference is the wavelength array:

```python
# ExGal: rest-frame
cube["wave"] = targs[0].spectra[0].wave / (1 + z_input[0])

# Gal: observed frame (as-is)
gal_targ["wave"] = targs[0].spectra[0].wave
```

This is correct because FERRE and RVS fit stellar templates in observed wavelengths at z≈0. For most Milky Way stars z_input ≈ 0.0 and the difference is negligible, but the principle is important: do not apply a redshift correction to galactic stellar targets.

### 6.3 LSF values

Rather than building two interpolator objects (`LSF_Data` and `LSF_Templates`) like ExGal, the Gal prepare stage evaluates the LSF directly at the observed wavelength grid:

```python
lsf_values = targs_lsf[0](gal_targ["wave"])
```

This returns an array of FWHM values in Å at each observed wavelength pixel. RVS and FERRE receive this array directly and use it to match template resolution to data resolution. There is no need for an `LSF_Config_*` text file lookup as in ExGal.

`lsf_values` above is the single global curve — as of v1.9 this is only what RVS/FERRE fall back to when `spaxel_weighted_lsf` is off (or for any bin `bucket_lsf_for_voronoi_bins` couldn't assign a real bucket to). When it's on (the default — see [Section 11](#11-configuration-file-reference)'s `SPAXEL_WEIGHTED_LSF` entry, and `aps_ifu_ExGal.md`'s own [Section 11](../aps_ifu_ExGal.md#11-spaxel-weighted-lsf-and-resolution-buckets) for the full mechanism, identical here), `ifu_Gal_prepare` also builds `bin_to_bucket`/`lsf_by_bucket` the same way `ifu_ExGal_prepare` builds `bin_to_bucket`/`LSF_Data_by_bucket` — the one difference is that RVS/FERRE consume plain evaluated arrays (matching `lsf_values`'s own convention), not callables, so each bucket's curve is evaluated on `gal_targ["wave"]` once up front rather than passed as a function.

### 6.4 Adaptive MIN_SNR for Gal

`adaptive_min_snr()` is called with **hardcoded** Gal-specific parameters rather than reading them from the config:

```python
_min_snr_gal = adaptive_min_snr(
    gal_targ["snr"],
    method           = "percentile",
    percentile       = 3,     # slightly more aggressive than ExGal's 2
    absolute_floor   = 0.3,   # higher floor — dead star fibres are obvious
    absolute_ceiling = 3.0,   # much higher than ExGal's 1.5
)
```

The higher ceiling (3.0 vs 1.5) reflects the fundamental difference between Gal and ExGal targets:

- **ExGal**: A spaxel with broadband SNR < 1 might still contain real signal from emission lines on a faint continuum (high-z galaxies, QSOs). Cutting at SNR=1.5 would remove scientifically valid data.
- **Gal**: Every stellar spaxel that is real signal will have detectable continuum. A spaxel at SNR < 0.5 in a stellar field is a dead fibre or edge artefact, not a faint star.

The Gal parameters are hardcoded (not read from the JSON config) because adding `MIN_SNR_METHOD`, `MIN_SNR_PERCENTILE`, etc. to the Gal config file would be confusing — those keys exist in the ExGal config with different intended values. If the user sets `MIN_SNR_GAL` to a float in the config, that value is used directly and adaptive mode is bypassed.

### 6.5 No surface brightness filter

`apply_flux_filter=False` is hardcoded when calling `spatial_bin_with_provenance()`. The SB flux filter was designed for extended galaxies where the outer low-SB regions contain sky-dominated spaxels that should not be co-added into the stellar kinematics bins. For point sources (stars) this concept does not apply — the spatial profile of the target falls off as a PSF, not as a gradual SB gradient. All spaxels with valid signal should be included in the spatial bins.

### 6.6 Path A — with spatial pre-binning

Identical logic to ExGal but without the flux filter. `spatial_bin_with_provenance()` groups spaxels into square bins of `SPBIN_SIZE_GAL` arcsec and passes them to `define_voronoi_bins()`.

Spectra are saved as **linear** (not log-rebinned) using `flag="lin"`:
- `apply_voronoi_bins(..., flag="lin")` — if `VORONOI_GAL=1`
- `save_binned_spectra_novor(..., flag="lin")` — if `VORONOI_GAL=0`

The output files are named `<headname>_BINSpectra_linear.fits` (not `_BINSpectra.fits`). FERRE and RVS require linear wavelength spacing.

### 6.7 Path B — no spatial pre-binning

The `MIN_SNR_GAL` threshold is applied directly to the raw spaxels:

```python
idx_inside  = np.where(gal_targ["snr"] >= configs_gal["MIN_SNR_GAL"])[0]
idx_outside = np.where(gal_targ["snr"] <  configs_gal["MIN_SNR_GAL"])[0]
```

Voronoi then runs on the raw passing spaxels.

### 6.8 Plot configuration adapter

`plot_all()` from `ExGalPrepare_plots` uses config keys named after the ExGal convention (`TARGET_SNR`, `SPBIN_SIZE_EXGAL`, `VORONOI`, `MIN_SNR`, `SB_FILTER`). The Gal module builds a translation dict before calling `plot_all`:

```python
_plot_configs = {
    "MIN_SNR":          configs_gal.get("MIN_SNR_GAL"),
    "TARGET_SNR":       configs_gal.get("TARGET_SNR_GAL"),
    "SPBIN_SIZE_EXGAL": spbin_size_gal if spbin_size_gal > 0 else None,
    "VORONOI":          configs_gal.get("VORONOI_GAL"),
    "SB_FILTER":        0,   # always 0 in Gal mode
}
```

This means the same diagnostic plot functions work for both modes without modification. The `SB_FILTER=0` entry ensures the flux filter panel in `_prep_spatialbin.png` correctly shows 0 removed spaxels (as expected for Gal mode).

---

## 7. Stage 2 — ifu_Gal Loop in Detail

### 7.1 Patch table resolution

Identical to ExGal (`patch_file` vs `patch_array` modes). See `aps_ifu_exgal_README.md` Section 6.1.

### 7.2 Optional re-classification for rows with missing Z

Gal mode has an additional feature not present in ExGal: `class_patch=True` triggers Redrock re-classification for any row whose redshift is `NaN`. This is useful for stellar fields where the catalogue redshift is missing (stars were not pre-assigned a radial velocity).

When `class_patch=True`, the current patch table is saved to a temporary FITS file, `ifu_class()` from `aps_ifu_prepare` runs Redrock on each row's aperture, and the result overwrites the temporary patch table which is then used for the rest of the loop. The temporary file is deleted after use.

### 7.3 Target selection within the loop

The Gal loop applies three selection rules that differ from ExGal:

**Skip mask rows** (`type='M'`): same as ExGal.

**Skip central target when T-type rows exist**: if the patch table contains `type='C'` rows (the central WEAVE catalogue object) alongside `type='T'` rows (extracted field sources), the central target is skipped. This prevents double-processing when the central object was already captured by another patch. In `patch_array` mode this check is bypassed.

**Skip ExGal classes**: rows classified as `GALAXY` or `QSO` are skipped — these go to `aps_ExGal_worker`. Unclassified rows are also skipped.

### 7.4 Aperture shrinkage for stellar targets

```python
enhanced_rad_factor = 0.5   # always, for all Gal targets
rad_factor = 1.0 if patch_array_mode else enhanced_rad_factor
```

In `patch_array` mode the aperture is used as given. In `patch_file` mode the aperture from the patch table is halved. This is the opposite of ExGal which sometimes enlarges apertures (×3) for single large targets. The reasoning:

- Galaxy apertures from `aps_ifu_prepare` are sized to encompass the galaxy's extended emission
- A stellar target is a point source — the IFU aperture around it will include sky if you use the full galaxy-sized aperture
- Halving the aperture concentrates the selection on the stellar PSF core where the SNR is highest

### 7.5 FERRE and RVS configuration resolution

FERRE and RVS require paths to executables, template directories, and configuration files. These can come from three places in priority order:

1. **Explicit function argument** (from the CLI `--ferre_exe`, `--rvs_config`, etc.)
2. **Config file** (JSON key `FERRE_EXE`, `RVS_CONFIG`, etc.)
3. **Hardcoded default** (only for `FERRE_GRID_PREFIX`, defaults to `"n"`)

This is handled by a loop over attribute/key/default triples:

```python
for attr, key, hardcoded_default in [
    (rvs_config,        "RVS_CONFIG",        None),
    (ferre_exe,         "FERRE_EXE",         None),
    (ferre_templates,   "FERRE_TEMPLATES",   None),
    (ferre_grid_prefix, "FERRE_GRID_PREFIX", "n"),
]:
    if attr is not None:          # CLI argument wins
        configs_gal[key] = attr
    elif configs_gal.get(key):    # config file
        ...
    elif hardcoded_default:       # last resort
        ...
```

If neither FERRE_EXE nor FERRE_TEMPLATES is set, FERRE is silently skipped with an informational message. RVS is always attempted if `RVS_CONFIG` is available.

### 7.6 `FERRE_GRID_IDS` handling

FERRE needs to know which model grid(s) to use for each spectral setup. `FERRE_GRID_IDS` is a comma-separated string of grid identifiers. It can be set as:

- `--ferre_grid_ids 1,2,3` from the CLI
- `"FERRE_GRID_IDS": "1,2,3"` in the JSON config

The string is split and passed to `run_ferre_for_ifu_gal()` as a list.

---

## 8. Running from the Command Line

### Minimal run — preparation only

Check spatial binning and Voronoi before running FERRE (which can take a long time):

```bash
python aps_Gal_worker.py \
    --infiles /data/L1/20240808/stackcube_3071431.fit \
              /data/L1/20240808/stackcube_3071430.fit \
    --headname stackcube_3071431__stackcube_3071430 \
    --outpath  /data/L2/20240808/11182/ \
    --patch_file /data/L2/20240808/11182/stackcube_..._targets_mod.fits \
    --IFU_config_dir <PYAPS_DIR>/configs/ExGal_configs/ \
    --IFU_params     <PYAPS_DIR>/configs/Gal_configs/LIFULR11_GAL.json \
    --wlranges 3800.0,5950.0 5900.0,9280.0 \
    --arms_ratio 1.0,1.0 \
    --sens_corr True --mask_gaps True --safe_mask_gaps True \
    --tellurics True --join_arms True \
    --caldir /data/CAL --catdir /data/CAT
```

> **Note:** Not passing `--rvs_config`, `--ferre_exe`, or `--ferre_templates` skips the fitting modules and runs preparation only. Inspect the figures in `figs_Gal/` before proceeding.

### Full Gal run with RVS and FERRE

```bash
python aps_Gal_worker.py \
    --infiles /data/L1/20240808/stackcube_3071431.fit \
              /data/L1/20240808/stackcube_3071430.fit \
    --headname stackcube_3071431__stackcube_3071430 \
    --outpath  /data/L2/20240808/11182/ \
    --patch_file /data/L2/20240808/11182/stackcube_..._targets_mod.fits \
    --IFU_config_dir <PYAPS_DIR>/configs/ExGal_configs/ \
    --IFU_params     <PYAPS_DIR>/configs/Gal_configs/LIFULR11_GAL.json \
    --rvs_config     <PYAPS_DIR>/configs/rvs_config.yaml \
    --ferre_exe      <FERRE_DIR>/src/ferre.x \
    --ferre_templates <PYAPS_DIR>/PyAPS_templates/templates_FERRE/ \
    --ferre_grid_ids 1,2,3 \
    --mp_Gal 4 \
    --wlranges 3800.0,5950.0 5900.0,9280.0 \
    --arms_ratio 1.0,1.0 \
    --sens_corr True --mask_gaps True --safe_mask_gaps True \
    --tellurics True --join_arms True \
    --caldir /data/CAL --catdir /data/CAT
```

### Override binning parameters at runtime

Without editing the JSON config:

```bash
python aps_Gal_worker.py \
    ... \
    --spbin_size_gal 3.0 \
    --target_snr_gal 50.0 \
    --voronoi_gal 1
```

### Single-target debug mode

```bash
python aps_Gal_worker.py \
    --infiles /data/L1/20240808/stackcube_3071431.fit \
              /data/L1/20240808/stackcube_3071430.fit \
    --headname stackcube_3071431__stackcube_3071430 \
    --outpath  /data/L2/20240808/11182/ \
    --IFU_config_dir <PYAPS_DIR>/configs/ExGal_configs/ \
    --IFU_params     <PYAPS_DIR>/configs/Gal_configs/LIFULR11_GAL.json \
    --patch_array "3,185.201,58.093,15.0,15.0,0.0001,0.0001,STAR" \
    --rvs_config <PYAPS_DIR>/configs/rvs_config.yaml \
    --ferre_exe  <FERRE_DIR>/src/ferre.x \
    --ferre_templates <PYAPS_DIR>/PyAPS_templates/templates_FERRE/ \
    --wlranges 3800.0,5950.0 5900.0,9280.0 \
    --arms_ratio 1.0,1.0 \
    --sens_corr True --mask_gaps True --safe_mask_gaps True \
    --tellurics True --join_arms True \
    --caldir /data/CAL --catdir /data/CAT
```

Note the smaller aperture (15×15 arcsec) appropriate for a point source.

### Running with in-pipeline re-classification

If some STAR rows in the patch table have `Z=NaN` (no redshift from Redrock):

```bash
python aps_Gal_worker.py \
    ... \
    --class_patch True \
    --class_templates     <PYAPS_DIR>/PyAPS_templates/templates_RR_PCA/ \
    --class_templates_ARC <PYAPS_DIR>/PyAPS_templates/templates_RR_ARC/ \
    --class_z_rad 1.5 \
    --mp_prep 4
```

---

## 9. Running from Python

### Full pipeline from Python

```python
from PyAPS.aps_ifu_gal import ifu_Gal

ifu_Gal(
    infiles         = ["/data/L1/20240808/stackcube_3071431.fit",
                       "/data/L1/20240808/stackcube_3071430.fit"],
    headname        = "stackcube_3071431__stackcube_3071430",
    outpath         = "/data/L2/20240808/11182/",
    IFU_config_dir  = "<PYAPS_DIR>/configs/ExGal_configs/",
    IFU_params      = "<PYAPS_DIR>/configs/Gal_configs/LIFULR11_GAL.json",
    patch_file      = "/data/L2/20240808/11182/stackcube_..._targets_mod.fits",
    rvs_config      = "<PYAPS_DIR>/configs/rvs_config.yaml",
    ferre_exe       = "<FERRE_DIR>/src/ferre.x",
    ferre_templates = "<PYAPS_DIR>/PyAPS_templates/templates_FERRE/",
    ferre_grid_ids  = ["1", "2", "3"],
    spbin_size_gal  = 3.0,
    voronoi_gal     = 1,
    target_snr_gal  = 50.0,
    nthreads        = 4,
    wlranges        = [[3800.0, 5950.0], [5900.0, 9280.0]],
    arms_ratio      = [1.0, 1.0],
    sens_corr=True, mask_gaps=True, safe_mask_gaps=True,
    tellurics=True, join_arms=True,
    caldir="/data/CAL", catdir="/data/CAT",
)
```

### Single target via `make_patch_array`

```python
from PyAPS.aps_ifu_gal import ifu_Gal, make_patch_array

# A star at z=0 (radial velocity will be measured by RVS)
patch = make_patch_array(
    row_id    = 3,
    ra        = 185.201,
    dec       = 58.093,
    a_arcsec  = 15.0,    # small aperture for point source
    b_arcsec  = 15.0,
    z         = 0.0,     # z=0 for galactic star (RVS measures vrad)
    zerr      = 0.001,
    class_str = "STAR",
)

ifu_Gal(
    infiles=[...], headname=..., outpath=...,
    IFU_config_dir=..., IFU_params=...,
    patch_array     = patch,
    rvs_config      = "<PYAPS_DIR>/configs/rvs_config.yaml",
    ferre_exe       = "<FERRE_DIR>/src/ferre.x",
    ferre_templates = "<PYAPS_DIR>/PyAPS_templates/templates_FERRE/",
    spbin_size_gal  = 3.0,
    nthreads        = 4,
    wlranges=[[3800.0, 5950.0], [5900.0, 9280.0]],
    arms_ratio=[1.0, 1.0],
    caldir="/data/CAL",
)
```

### Calling `ifu_Gal_prepare` directly

For inspection of the preparation stage only:

```python
from PyAPS.aps_ifu_gal import ifu_Gal_prepare

prep = ifu_Gal_prepare(
    infiles        = ["/data/L1/20240808/stackcube_3071431.fit",
                      "/data/L1/20240808/stackcube_3071430.fit"],
    headname       = "stackcube_3071431__stackcube_3071430_P0003",
    IFU_params     = "<PYAPS_DIR>/configs/Gal_configs/LIFULR11_GAL.json",
    outpath        = "/data/L2/20240808/11182/",
    z_input        = [0.0, 0.001],
    area           = [185.201, 58.093, 15.0, 15.0, 0.0],
    spbin_size_gal = 3.0,
    voronoi_gal    = 1,
    target_snr_gal = 50.0,
    wlranges       = [[3800.0, 5950.0], [5900.0, 9280.0]],
    arms_ratio     = [1.0, 1.0],
    join_arms      = True,
    sens_corr=True, mask_gaps=True, safe_mask_gaps=True,
    tellurics=True,
    IFU_config_dir = "<PYAPS_DIR>/configs/ExGal_configs/",
    caldir         = "/data/CAL",
)

if prep is not None:
    gal_targ    = prep["gal_targ"]
    configs_gal = prep["configs_gal"]
    lsf_values  = prep["lsf_values"]   # FWHM array evaluated at wave grid
    figdir      = prep["figdir"]

    print("N spaxels in cube:    ", gal_targ["spec"].shape[1])
    print("Wavelength range:     ", gal_targ["wave"][[0, -1]], "Å (observed)")
    print("MIN_SNR_GAL applied:  ", configs_gal["MIN_SNR_GAL"])
    print("TARGET_SNR_GAL:       ", configs_gal["TARGET_SNR_GAL"])
    print("LSF FWHM at 5000 Å:  ", lsf_values[abs(gal_targ["wave"] - 5000).argmin()])
```

---

## 10. Input Parameters — Complete Reference

### Shared APSOB parameters

These are identical to the ExGal module. See `aps_ifu_exgal_README.md` Section 9 and `APSOB_README.md` for full descriptions.

| Parameter | Type | Default | Note for Gal mode |
|---|---|---|---|
| `infiles` | list of str | required | Same as ExGal |
| `headname` | str | required | Patch ID appended: `<headname>_P<NNNN>` |
| `IFU_params` | str | required | Use the **Gal** config (e.g. `LIFULR11_GAL.json`) |
| `outpath` | str | required | |
| `IFU_config_dir` | str or None | `None` | Same ExGal configs dir — contains LSF files |
| `wlranges` | list or None | `None` | Same as ExGal |
| `aps_ids` | list or None | `None` | Same as ExGal |
| `targsrvy` | list or None | `None` | Same as ExGal |
| `targclass` | list or None | `None` | Same as ExGal |
| `mask_aps_ids` | list or None | `None` | Same as ExGal |
| `area` | list or None | `None` | Elliptical aperture — typically smaller than ExGal for point sources |
| `mask_areas` | list or None | `None` | Always `None` in the main loop (stellar fields have no sub-masks) |
| `sens_corr` | bool | `True` | |
| `mask_gaps` | bool | `True` | |
| `safe_mask_gaps` | bool | `True` | |
| `vacuum` | bool | `False` | |
| `tellurics` | bool | `True` | |
| `fill_gap` | bool | `False` | |
| `arms_ratio` | list or None | `None` | |
| `join_arms` | bool | `True` | **Hardcoded `True` in Gal — argument ignored** |
| `z_input` | [z, zerr] | required | For stars z≈0; RVS measures the actual radial velocity |
| `catdir` | str or None | `None` | |
| `caldir` | str or None | `None` | |
| `no_spec_ext` | bool | `False` | if True, it discards the spec relaetd extensions from the final APS file |
| `spaxel_weighted_lsf` | bool or None | `None` | Per-bin resolution for RVS/FERRE — resolves from `SPAXEL_WEIGHTED_LSF` in the config (default **on** as of v1.9) unless overridden. See [Section 11](#spaxel-weighted-lsf-gal). |


### Gal-specific binning parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `spbin_size_gal` | float or None | `None` | Spatial bin size in arcsec. `None` = use config value. Typical: 3.0–5.0 arcsec. Set to `-1` to skip spatial pre-binning. |
| `min_snr_gal` | float or None | `None` | Minimum SNR per spaxel. `None` = adaptive (recommended). |
| `voronoi_gal` | int or None | `None` | Voronoi tessellation: `1`=enabled, `0`=disabled. `None` = use config. |
| `target_snr_gal` | float or None | `None` | Target SNR per Voronoi bin. `None` = use config. Typical: 50–80. |

### Fitting module parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `rvs_config` | str or None | `None` | Path to RVS YAML configuration file. If not set and not in JSON config, RVS is skipped. |
| `ferre_exe` | str or None | `None` | Path to the compiled FERRE binary (`a.out` or `ferre.x`). |
| `ferre_templates` | str or None | `None` | Path to FERRE model grid directory. |
| `ferre_grid_ids` | list or None | `None` | List of FERRE grid IDs to use (e.g. `["1", "2", "3"]`). |
| `ferre_grid_prefix` | str or None | `None` | FERRE grid filename prefix. Default `"n"` if not set. |

### Re-classification parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `class_patch` | bool | `False` | Run Redrock re-classification on rows with missing Z before the main loop. |
| `class_templates` | str or None | `None` | Redrock PCA template directory. Required if `class_patch=True`. |
| `class_templates_ARC` | str or None | `None` | Redrock archetype template directory. |
| `class_z_rad` | float | `1.5` | Search radius in arcsec for catalogue redshift prior. |
| `class_ntop` | int | `1` | Number of top Redrock solutions to store. |
| `mp_prep` | int | `1` | Threads for the re-classification step. |

### Loop control

| Parameter | Type | Default | Description |
|---|---|---|---|
| `patch_file` | str or None | `None` | Patch FITS table from `aps_ifu_prepare`. Mutually exclusive with `patch_array`. |
| `patch_array` | dict or None | `None` | Single injected row (debug mode). |
| `nthreads` | int | `1` | Threads for RVS and FERRE fitting. CLI: `--mp_Gal`. |
| `overwrite` | bool | `True` | Overwrite existing output files. |
| `UAPSID` | str or None | `None` | Written to L2 FITS header. |

---

## 11. Configuration File Reference

The Gal module reads a separate JSON config from the ExGal module, typically `LIFULR11_GAL.json`. The keys below are specific to Gal mode.

### Spatial binning (Gal)

```json
"PIXELSIZE"        : 0.0,
"SPBIN_SIZE_GAL"   : 3.0,
"VORONOI_GAL"      : 1,
"TARGET_SNR_GAL"   : 50.0,
"COVAR_VOR"        : 0.0
```

`SPBIN_SIZE_GAL` is the spatial pre-bin side length in arcsec. Larger values (3–5 arcsec) give fewer, higher-SNR bins — appropriate for stellar parameter fitting where you want clean spectra, not fine spatial resolution. Set to `-1` to skip spatial pre-binning.

`VORONOI_GAL=1` enables adaptive Voronoi tessellation. `VORONOI_GAL=0` treats each spatial bin as its own final bin — useful when you want a fixed regular grid (spatial bins only, no Voronoi).

`TARGET_SNR_GAL` should be higher than ExGal's `TARGET_SNR` because FERRE needs good continuum SNR to constrain Teff, log g, and [Fe/H] simultaneously. Typical values: 50 for basic parameter estimates, 80 for precise [α/Fe].

### Spaxel-weighted LSF (Gal)

```json
"SPAXEL_WEIGHTED_LSF"  : 1,
"LSF_N_BUCKETS"         : 6
```

Same keys, same mechanism as ExGal's own — see `aps_ifu_ExGal.md`'s [Section 11, "Spaxel-Weighted LSF and Resolution Buckets"](aps_ifu_ExGal.md#11-spaxel-weighted-lsf-and-resolution-buckets) for the full explanation. **Default on as of v1.9.** The only difference in Gal mode: RVS and FERRE consume plain evaluated FWHM arrays (`lsf_by_bucket`), not callables, since that's `lsf_values`'s own pre-existing convention here (see [Section 6.3](#63-lsf-values)) — the bucketing/aggregation logic itself is identical.

### MIN_SNR (Gal)

```json
"MIN_SNR_GAL" : null
```

Set to `null` for adaptive mode (recommended). Set to a float to use a fixed threshold. The adaptive algorithm parameters are **hardcoded** in the module (not read from the JSON config) to avoid confusion with the ExGal `MIN_SNR_*` keys.

### FERRE / RVS paths

```json
"RVS_CONFIG"        : "<PYAPS_DIR>/configs/rvs_config.yaml",
"FERRE_EXE"         : "<FERRE_DIR>/src/ferre.x",
"FERRE_TEMPLATES"   : "<PYAPS_DIR>/PyAPS_templates/templates_FERRE/",
"FERRE_GRID_IDS"    : "1,2,3",
"FERRE_GRID_PREFIX" : "n"
```

These keys in the JSON config are the config-file equivalent of the CLI arguments. The CLI argument always takes priority over the JSON value.

### Full example LIFULR11_GAL.json (key parameters)

```json
{
    "PIXELSIZE"         : 0.0,
    "SPBIN_SIZE_GAL"    : 3.0,
    "VORONOI_GAL"       : 1,
    "TARGET_SNR_GAL"    : 50.0,
    "COVAR_VOR"         : 0.0,
    "MIN_SNR_GAL"       : null,
    "RVS_CONFIG"        : "<PYAPS_DIR>/configs/rvs_config.yaml",
    "FERRE_EXE"         : "<FERRE_DIR>/src/ferre.x",
    "FERRE_TEMPLATES"   : "<PYAPS_DIR>/PyAPS_templates/templates_FERRE/",
    "FERRE_GRID_IDS"    : "1,2,3",
    "FERRE_GRID_PREFIX" : "n",
    "SSP_LIB"           : "MILES"
}
```

---

## 12. Aperture Handling — Stars vs Galaxies

This section explains the `enhanced_rad_factor = 0.5` design choice in detail, because it is one of the most visible differences between the Gal and ExGal loops.

In `aps_ifu_prepare`, the segmentation step fits an elliptical aperture to each detected source using a Kron-radius-based approach. For extended sources (galaxies) this naturally captures the full extent of the light distribution. For point sources (stars) in a mixed IFU field, the segmentation may assign an aperture that is either:

- Too large (the star is in the same field as a large galaxy, and the default minimum aperture is set for that galaxy's size)
- Correctly small (the star was detected and fitted independently)

In practice, stellar targets in LIFU fields where the primary science target is a galaxy will have patch-table apertures sized for the galaxy scale (~30–100 arcsec semi-major axis). Extracting a stellar spectrum with a 100-arcsec aperture would include hundreds of IFU spaxels that are dominated by sky or galaxy light, not the star.

The `×0.5` shrinkage is a conservative default that works in most cases. If you have a specific stellar target where the full aperture is correct (e.g. a star in a pure MIFU stellar field), use `patch_array` mode and specify the correct aperture size directly.

---

## 13. Adaptive MIN_SNR for Gal Mode

The Gal adaptive MIN_SNR uses different parameters from ExGal, hardcoded in the module:

```python
adaptive_min_snr(
    gal_targ["snr"],
    method           = "percentile",
    percentile       = 3,      # ExGal uses 2
    absolute_floor   = 0.3,    # ExGal uses 0.1
    absolute_ceiling = 3.0,    # ExGal uses 1.5
)
```

**Why percentile=3 instead of 2**: Stellar continuum is always detectable in WEAVE Gal targets. The 3rd percentile will typically catch dead fibres and edge spaxels while still being gentle on low-SNR outer spaxels of the PSF wing.

**Why floor=0.3 instead of 0.1**: Any WEAVE stellar spaxel with broadband SNR < 0.3 in the joined blue+red spectrum is essentially measuring noise. Unlike ExGal where emission-line signal can be real even at SNR=0.1, there is no physical mechanism to produce stellar continuum signal at SNR < 0.3.

**Why ceiling=3.0 instead of 1.5**: A stellarcontinuum spaxel with broadband SNR between 1.5 and 3.0 may genuinely be a low-SNR but real outer-PSF spaxel, or it may be a marginal sky-dominated spaxel. In ExGal mode you must be conservative (ceiling=1.5) because removing real low-continuum spaxels from a galaxy removes scientific information. In Gal mode, removing a marginal spaxel reduces the number of bins and actually improves the per-bin SNR of the remaining bins — a feature, not a bug.

---

## 14. Binning Strategy — Fewer High-SNR Bins

The key scientific trade-off in Gal mode binning is spatial resolution vs per-bin SNR. For FERRE stellar parameter fitting, you need:

- **Minimum SNR ~30** for basic Teff, log g, [Fe/H] estimates
- **SNR ~50–80** for precise [α/Fe] and individual abundance measurements
- **Linear spectra** (not log-rebinned) at the original wavelength sampling

Fewer bins means higher per-bin SNR, better parameter recovery, and faster FERRE computation. The recommended approach is:

```json
"SPBIN_SIZE_GAL"   : 3.0,    →  ~500 input bins from 5494 raw spaxels
"TARGET_SNR_GAL"   : 50.0,   →  ~50-100 Voronoi output bins
```

For even fewer bins with higher SNR:

```json
"SPBIN_SIZE_GAL"   : 5.0,    →  ~200 input bins
"TARGET_SNR_GAL"   : 80.0,   →  ~10-30 Voronoi output bins
```

At the extreme, for a single integrated stellar spectrum (e.g. to check a bright field star):

```json
"SPBIN_SIZE_GAL"   : 10.0,
"TARGET_SNR_GAL"   : 200.0,
"VORONOI_GAL"      : 0      →  treat each spatial bin as final (no Voronoi)
```

---

## 15. FERRE and RVS Integration

FERRE and RVS are called after `ifu_Gal_prepare` returns successfully. Both receive:

- `headname`: to find the `_BINSpectra_linear.fits` file on disk
- `outpath`: output directory
- `configs_gal`: the full configuration dict (with resolved FERRE/RVS paths)
- `lsf_values`: the FWHM array evaluated at the observed wavelength grid
- `nthreads`: parallelism level

**RVS** (`run_rvs_for_ifu_gal`): measures radial velocities for each Voronoi bin using template cross-correlation. Returns `True` on success, `False` on failure. Failure is logged as a warning and does not abort the target — FERRE still runs.

**FERRE** (`run_ferre_for_ifu_gal`): measures stellar atmospheric parameters (Teff, log g, [Fe/H], [α/Fe]) for each Voronoi bin by fitting synthetic stellar spectra. FERRE is skipped if `FERRE_EXE` or `FERRE_TEMPLATES` is not set in configs or as function arguments.

Both modules write their own output FITS files which are then combined by `ifuGalL2merge` into the final `_APS.fits` product.

---

## 16. Output Files

For each successfully processed patch:

| File | Description |
|---|---|
| `<headname>_P####_table.fits` | Voronoi bin table: coordinates, bin IDs, per-bin SNR, N spaxels per bin |
| `<headname>_P####_BINSpectra_linear.fits` | Linear-wavelength co-added spectra, one per Voronoi bin |
| `<headname>_P####_RVS.fits` | Radial velocities from RVS (if RVS ran) |
| `<headname>_P####_FERRE.fits` | Stellar parameters: Teff, log g, [Fe/H], [α/Fe] per bin (if FERRE ran) |
| `<headname>_P####_APS.fits` | Merged L2 product — all results in one FITS file |

Diagnostic figures in `outpath/figs_Gal/`:

| Figure | Content |
|---|---|
| `*_snr_stages.png` | 5-panel SNR chain (identical layout to ExGal) |
| `*_prep_raw.png` | Raw SNR, signal, noise maps |
| `*_prep_snrcut.png` | MIN_SNR_GAL keep/remove map + histogram |
| `*_prep_spatialbin.png` | 4-panel spatial binning (note: flux filter panel will show 0 removed — correct for Gal) |
| `*_prep_voronoi.png` | Voronoi bin IDs, SNR per bin, N spaxels per bin |
| `*_sbin_<size>.png` | Raw vs binned side-by-side |

---


## 17. Controlling output file size — `--no_spec_ext`

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



## 18. Troubleshooting

### "No valid APS IDs for this Gal patch"

The aperture contains no spaxels after APSOB filtering. For stellar targets this is often because the aperture was shrunk to 0.5× and is now too small to contain any spaxels. Try:

1. Set `patch_array` mode with a manually specified aperture of 15–30 arcsec
2. Check that the star RA/Dec in the patch table falls within the IFU footprint
3. Check `targsrvy` and `targclass` are not filtering out all spaxels

### "No valid spaxels remaining after filters"

After MIN_SNR_GAL cut, no spaxels survive. For a real stellar target this should not happen. Check:

1. Is `MIN_SNR_GAL` too high? Try setting `"MIN_SNR_GAL": 0.0` in config
2. Is `SPBIN_SIZE_GAL` too large, creating bins with no matching spaxels? Try `-1` (no binning)
3. Check `*_prep_raw.png` — is there any signal in the raw SNR map?

### FERRE is silently skipped

```
INFO: FERRE skipped (ferre_exe or ferre_templates not set)
```

Neither `--ferre_exe` (CLI) nor `"FERRE_EXE"` (JSON config) is set. Add one of these, or set both in the JSON config file.

### RVS fails for every target

```
WARNING: RVS failed for id=3
```

Check:
- Does `rvs_config` point to an existing YAML file?
- Is the RVS config correctly configured for the WEAVE wavelength range?
- Are the `_BINSpectra_linear.fits` files written by `ifu_Gal_prepare`? Check `outpath` for them.

### Spatial binning produces too many bins

The binning is not aggressive enough for FERRE. Increase `SPBIN_SIZE_GAL` (e.g. 5.0) and/or `TARGET_SNR_GAL` (e.g. 80). These can be passed at the CLI without editing the JSON:

```bash
--spbin_size_gal 5.0 --target_snr_gal 80.0
```

### `_prep_spatialbin.png` shows "flux filter: kept=N  removed=0"

This is correct and expected for Gal mode. The flux/SB filter is always disabled (`apply_flux_filter=False`). Panel 2 of the spatial binning figure will always show 0 removed spaxels. This is not a bug.

### Central target (type=C) processed unexpectedly

In `patch_array` mode the type=C check is bypassed. If you are injecting a single-row patch and it has `type='C'`, it will be processed. To avoid this set `row_type='T'` in `make_patch_array`.

---

*For further information, bug reports, or contributions please open an issue on the PyAPS repository.*
