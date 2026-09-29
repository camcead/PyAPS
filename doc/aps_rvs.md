# aps_rvs — Radial Velocity and Stellar Parameter Module

## A Reference and Tutorial Guide for the PyAPS Pipeline

**Module:** `PyAPS.aps_rvs`
**Entry point:** `rvsweave()` / `aps_rvs.py`
**Underlying library:** RVSPECFIT (by Sergey Koposov)
**Targets:** `STAR`, `WD` (white dwarfs)
**PyAPS version:** 3.3+
**Maintained by:** CASU, Institute of Astronomy, University of Cambridge

---

## Table of Contents

1. [What This Module Does](#1-what-this-module-does)
2. [How It Fits in the Pipeline](#2-how-it-fits-in-the-pipeline)
3. [Architecture Overview](#3-architecture-overview)
4. [The rvs_config.yaml File](#4-the-rvs_configyaml-file)
5. [The Class File (classfile)](#5-the-class-file-classfile)
6. [Resolution Matrix — FWHM-Based Fitting](#6-resolution-matrix--fwhm-based-fitting)
7. [The sigma_0 Deconvolution Parameter](#7-the-sigma_0-deconvolution-parameter)
8. [Quality Flags — RVS_WARN Bitmask](#8-quality-flags--rvs_warn-bitmask)
9. [Parallelism](#9-parallelism)
10. [Running from the Command Line](#10-running-from-the-command-line)
11. [Running from Python](#11-running-from-python)
12. [Input Parameters — Complete Reference](#12-input-parameters--complete-reference)
13. [Output Files](#13-output-files)
14. [Output Columns — Complete Reference](#14-output-columns--complete-reference)
15. [Differences Between APS Defaults and rvsweave Defaults](#15-differences-between-aps-defaults-and-rvsweave-defaults)
16. [Troubleshooting](#16-troubleshooting)

---

## 1. What This Module Does

`aps_rvs` wraps the RVSPECFIT code (by Sergey Koposov) and applies it to WEAVE stellar spectra. For each `STAR` or `WD` target it:

1. Reads and pre-processes the L1 FITS spectra via APSOB (sensitivity correction, gap masking, telluric masking, arm joining)
2. Builds a per-fiber, per-wavelength resolution matrix from empirical FWHM interpolation (or falls back to a fixed resolution)
3. Runs a CCF-based initial radial velocity estimate
4. Fits stellar templates to determine radial velocity (VRAD), atmospheric parameters (Teff, log g, [Fe/H], [α/Fe]), and vsini
5. Evaluates fit quality and sets bitmask warning flags
6. Writes a parameter table (and optionally a spectra table) to FITS

The module supports MOS, LIFU, and MIFU modes and handles single-arm or multi-arm (joined or separate) observations.

---

## 2. How It Fits in the Pipeline

```
L1 FITS files
      │
      ├──────── Redrock classification ─────► classfile (zbest_*.fits)
      │                                            │
      │                                            │
      ▼                                            ▼
aps_rvs.py  ◄──────────────────────────────────────┘
      │
      │  rvsweave()
      │      │
      │      ├── proc_rvs()           ← orchestration, parallel/serial dispatch
      │      │       │
      │      │       └── proc_weave() ← per-thread worker (per-target loop)
      │      │               │
      │      │               ├── APSOB construction (L1 data preprocessing)
      │      │               ├── makeR_from_fwhm_array() → resolution matrix
      │      │               ├── fitter_ccf.fit()        → CCF initial velocity
      │      │               ├── vel_fit.process()       → full template fit
      │      │               └── get_rvs_warn()          → quality bitmask
      │      │
      │      └── rvs_write_fits()    ← write rvs_*.fits and rvsspec_*.fits
      │
      ▼
rvs_<headname>.fits          (parameter table)
rvsspec_<headname>.fits      (spectra + model, optional)
figs/RVS_*.png               (plots, optional)
```

In the automated pipeline, `aps_rvs` is typically called after `aps_class` (Redrock classification) and before `aps_ferre` (stellar parameters from full spectral fitting). RVS provides a fast, robust radial velocity used to initialise FERRE.

---

## 3. Architecture Overview

The code is structured in three layers:

**`rvsweave()`** — the command-line entry point. Parses arguments, validates file ordering via `l1_fileinfo()`, constructs output paths, and calls `proc_rvs()`.

**`proc_rvs()`** — orchestration layer. Counts qualifying targets, adjusts thread count, reads the RVSPECFIT config, and dispatches either a serial or parallel (`multiprocessing.Pool`) execution of `proc_weave()`.

**`proc_weave()`** — the per-thread worker. Constructs an `APSOB` object to preprocess the spectra for its assigned subset of targets, then loops over each target to build the resolution matrix, run the CCF fit, run the full template fit, evaluate quality flags, and accumulate results in an `OrderedDict` that is converted to an Astropy `Table` at the end.

The results from all threads are `vstack`-ed in `proc_rvs()` and passed to `rvs_write_fits()` to produce the final FITS outputs.

---

## 4. The rvs_config.yaml File

The RVSPECFIT configuration file controls all fitting parameters. It is passed via `--config` at the command line or `config=` in Python. The APS-customised reader `read_config_APS_RVS()` automatically expands environment variables (e.g. `$HOME`) in path values, making the config file portable across machines.

Key parameters in the config file:

```yaml
# Template library location
template_lib: <PYAPS_DIR>/PyAPS_templates/templates_RVS/

# Velocity search range (km/s)
min_vel: -1000
max_vel: 1000

# Stellar parameter bounds
min_logg: -0.5
max_logg: 6.5
min_teff: 2300
max_teff: 15000
min_feh: -4.0
max_feh: 1.0
min_vsini: 0
max_vsini: 500

# CCF initial fit (recommended True)
ccf_init: True

# Polynomial continuum order
npoly: 19
```

`read_config_APS_RVS()` and `freezeDict_APS_RVS()` are APS-specific extensions of the RVSPECFIT utility functions. The key APS addition is environment variable expansion: any string value containing `$HOME`, `$APSDATA`, or similar is expanded before the config is frozen into a `frozendict.frozendict` for thread-safe sharing.

---

## 5. The Class File (classfile)

The classfile is a FITS table produced by the Redrock classification step (`aps_class`). It contains one row per target with at minimum:

- `APS_ID`: fiber identifier
- `TARGID`: WEAVE target ID string
- `CLASS`: best classification (e.g. `"STAR"`, `"GALAXY"`)
- `Z`: best-fit redshift
- `CNAME`: WEAVE CNAME string

The function `aps_ids_class()` from `aps_utils` reads this file and extracts the `APS_ID` values for targets with `CLASS` in `['STAR', 'WD']`. It also handles thread-based subsetting for parallel processing (distributes targets round-robin across threads).

A special value `classfile='IFU'` was previously supported for IFU mode but has been retracted since IFU RVS is now a separate module (`aps_ifu_rvs`).

---

## 6. Resolution Matrix — FWHM-Based Fitting

One of the most important recent changes (December 2025 / January 2026) is the move from fixed-resolution spectral fitting to a **per-fiber, per-wavelength resolution matrix**. This is now the default mode in production.

### 6.1 What the resolution matrix is

RVSPECFIT's `spec_fit.ResolMatrix` wraps a sparse matrix `R` (shape `n_wave × n_wave`) that describes how each wavelength pixel in the template is spread into the observed pixels by the LSF. For a Gaussian LSF this is a banded diagonal matrix where each row is a Gaussian kernel centred on that wavelength pixel with the local FWHM.

### 6.2 How it is built

For each target and each arm/setup, the code:

1. Calls `APSOBJ.get_fwhm(aps_id=tgsAPS_ID, fwhm_key='fwhm')` to get the per-fiber FWHM interpolation function (from either `aps_lsf` or `aps_fwhm` — see `aps_lsf_fwhm_README.md` for details)
2. Evaluates the FWHM function at every wavelength pixel in the observed spectrum: `fwhm_values = fwhm_func(wave_s)`
3. Calls `makeR_from_fwhm_array(wave_s, fwhm_values, use_deconvolution=True, sigma0_angstrom=targs_sigma0_angstrom)` to construct the resolution matrix `R`

The `makeR_from_fwhm_array()` function is in `aps_utils`. It builds the Gaussian kernel row by row from the FWHM array. When `use_deconvolution=True`, it subtracts the template library's intrinsic resolution (`sigma0_angstrom`) in quadrature before computing the convolution kernel:

```
σ_kernel = sqrt(σ_data² − σ_template²)
```

This ensures the template (which was already broadened to `sigma0_angstrom` during generation) is only broadened by the *additional* instrumental broadening above the template resolution. If `σ_data < σ_template` (which should not happen in practice), the kernel collapses to a delta function.

### 6.3 Fallback behaviour

If FWHM interpolation fails (e.g. the LSF file is missing for this setup), the code falls back to `makeR()` with a fixed resolution from `APSOBJ.resolution()[ns]`. A warning message is printed:

```
APS_ID 1006 setup BLUEL: FWHM failed (...), using fixed resolution
```

### 6.4 Template names and setup matching

RVSPECFIT looks up templates by the `specdata` object's name. In APS, this name is constructed as:

```python
'%s_%s' % (targs_fmode_unit, s)
```

where `targs_fmode_unit` is the normalised focal plane unit name (`MOS`, `MIFU`, or `LIFU`) from `APSOBJ.fmode_unit()`, and `s` is the setup string from `orig_setups` (e.g. `BLUEL11`, `BLUEL11_REDL11` for joined arms). This ensures the right template library (resolution, wavelength range) is selected for each configuration.

---

## 7. The sigma_0 Deconvolution Parameter

`estimate_sigma0_for_deconvolution(fpmode, resmode)` returns the intrinsic Gaussian sigma (in Angstroms) of the RVS template library for a given focal plane mode and resolution mode.

### 7.1 Background

The RVS template spectra are generated from Phoenix/Kurucz model atmospheres at high resolution, then degraded to a fixed resolution using a constant-sigma Gaussian kernel parameterised by `resol_func` (e.g. `'x/0.5'` meaning FWHM = 0.5 Å at all wavelengths, where x = wavelength in Å; i.e. σ = 0.5/2.355 = 0.212 Å). Note that "resolution" is ambiguous: here it refers to the FWHM of the LSF in Angstroms, not the dimensionless resolving power R = λ/FWHM. The FWHM and resolving power both describe the instrumental broadening but are inversely related and wavelength-dependent — FWHM is what is directly measured and used by the pipeline. When fitting the observed data (which has a higher FWHM than the template), pPXF/RVSPECFIT needs to know how much *additional* broadening to apply. If the observed FWHM is 2.5 Å and the template was made at σ_template = 0.212 Å, the kernel applied to the template should have:

```
σ_kernel = sqrt(σ_observed² − σ_template²)
         = sqrt((2.5/2.355)² − 0.212²)
```

The `sigma0_angstrom` parameter carries `σ_template`.

### 7.2 Lookup table

| fpmode | resmode | resol_func used in template generation | σ₀ (Å) |
|---|---|---|---|
| MOS-A, MOS-B, MIFU | HR | x/0.5 | 0.212 |
| MOS-A, MOS-B, MIFU | LR | x/2.4 | 1.019 |
| LIFU | HR | x/0.7 | 0.297 |
| LIFU | LR | x/3.5 | 1.486 |

These values are derived from the January 2026 template generation scripts. The values correspond to `C / 2.355` where C is the FWHM constant in `resol_func = 'x/C'`.

If an unknown combination is passed, the function returns `1.0` Å with a warning. Fallback entries (`MOS` and `IFU`) are included for robustness.

### 7.3 How it is obtained in proc_weave

```python
targs_fmode     = APSOBJ.fmode()       # e.g. 'LIFU', 'MOS-A'
targs_resmode   = list(set(APSOBJ.res_mode()))[0]   # e.g. 'LR', 'HR'
targs_sigma0_angstrom = estimate_sigma0_for_deconvolution(targs_fmode, targs_resmode)
```

This is computed once per APSOB call (i.e. once per thread) and reused for all targets in that thread.

---

## 8. Quality Flags — RVS_WARN Bitmask

`get_rvs_warn()` evaluates seven quality conditions and returns an integer bitmask `RVS_WARN` written to the output table. The bitmask uses powers of 2 so individual flags can be tested with bitwise AND.

| Bit | Value | Name | Condition | Interpretation |
|---|---|---|---|---|
| 0 | 1 | `CHISQ_WARN` | `(χ²_continuum − χ²_fit) < 50` | Fit no better than a featureless continuum |
| 1 | 2 | `RV_WARN` | VRAD within 5 km/s of min_vel or max_vel | Velocity at edge of search range |
| 2 | 4 | `RVERR_WARN` | VRAD_ERR > 100 km/s | Velocity uncertainty too large |
| 3 | 8 | `PARAM_WARN` | Any parameter within threshold of its grid edge | Parameters pegged at boundary |
| 4 | 16 | `VSINI_WARN` | vsini > 20 km/s | Rotation very high (may affect parameters) |
| 5 | 32 | `BAD_SPECTRUM` | (reserved for upstream use) | Issue with spectrum |
| 6 | 64 | `BAD_HESSIAN` | `res1['bad_hessian'] == True` | Hessian evaluation failed at optimum |

`DELTA_CHISQ` = χ²_continuum − χ²_fit is also stored in the output table as a continuous quality metric (higher = better fit relative to continuum).

**Testing flags in Python:**

```python
from astropy.io import fits
import numpy as np

tab = fits.open("rvs_headname.fits")[1].data

# Targets with no warnings
good = tab["RVS_WARN"] == 0

# Targets with RV at edge of search range
rv_edge = (tab["RVS_WARN"] & 2) != 0

# Targets with bad Hessian
bad_hess = (tab["RVS_WARN"] & 64) != 0

# Targets with any warning
any_warn = tab["RVS_WARN"] != 0
```

**PARAM_WARN thresholds:**

| Parameter | Grid range | Edge threshold |
|---|---|---|
| Teff | 2300–15000 K | ±10 K |
| [Fe/H] | −4 to +1 | ±0.01 dex |
| log g | −0.5 to 6.5 | ±0.01 dex |

---

## 9. Parallelism

`proc_rvs()` uses `multiprocessing.Pool` when `nthreads > 1`. The target list is distributed across threads by `aps_ids_class()` using a round-robin scheme based on `threadid` and `nthreads`. Each thread constructs its own independent `APSOB` instance and accumulates its own `OrderedDict`, which is then collected and stacked.

If the number of qualifying targets is less than the requested thread count, `nthreads` is silently reduced to `len(aps_ids_satis_class)` before pool creation to avoid empty threads.

Environment variables for threading libraries are set at module import time to prevent over-subscription:

```python
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
```

This ensures each Python process uses exactly one thread from numpy/scipy/BLAS, allowing `multiprocessing` to provide clean parallelism at the process level without interference from library-level threading.

---

## 10. Running from the Command Line

### 10.1 Minimal MOS run

```bash
python aps_rvs.py \
    --infiles  /data/L1/stacked_1002046.fit \
               /data/L1/stacked_1002045.fit \
    --classfile /data/L2/3294/zbest_stacked_1002046__stacked_1002045.fits \
    --config    <PYAPS_DIR>/configs/rvs_config.yaml \
    --outpath   /data/L2/3294/ \
    --headname  stacked_1002046__stacked_1002045 \
    --wlranges  4200.0,6000.0  6000.0,8000.0 \
    --sens_corr True \
    --mask_gaps True --safe_mask_gaps True \
    --tellurics True --vacuum False --fill_gap True \
    --arms_ratio 1.0,1.0 \
    --join_arms False \
    --mp 1 \
    --overwrite True \
    --catdir /data/CAT --caldir /data/CAL
```

### 10.2 Full run with plots and spectra output

```bash
python aps_rvs.py \
    --infiles  /data/L1/stacked_1002046.fit \
               /data/L1/stacked_1002045.fit \
    --classfile /data/L2/3294/zbest_stacked_1002046__stacked_1002045.fits \
    --config    <PYAPS_DIR>/configs/rvs_config.yaml \
    --outpath   /data/L2/3294/ \
    --headname  stacked_1002046__stacked_1002045 \
    --wlranges  4200.0,6000.0  6000.0,8000.0 \
    --sens_corr True \
    --mask_gaps True --safe_mask_gaps True \
    --tellurics True --vacuum False --fill_gap True \
    --arms_ratio 1.0,0.83 \
    --join_arms False \
    --mp 4 \
    --overwrite True \
    --outspec True \
    --fig True \
    --catdir /data/CAT --caldir /data/CAL
```

### 10.3 LIFU run with joined arms

```bash
python aps_rvs.py \
    --infiles  /data/L1/20240808/stackcube_3071431.fit \
               /data/L1/20240808/stackcube_3071430.fit \
    --classfile /data/L2/20240808/11182/zbest_stackcube_3071431__stackcube_3071430.fits \
    --config    <PYAPS_DIR>/configs/rvs_config.yaml \
    --outpath   /data/L2/20240808/11182/ \
    --headname  stackcube_3071431__stackcube_3071430 \
    --wlranges  3800.0,5950.0  5900.0,9280.0 \
    --sens_corr True \
    --mask_gaps True --safe_mask_gaps True \
    --tellurics True --vacuum False --fill_gap True \
    --arms_ratio 1.0,1.0 \
    --join_arms True \
    --mp 4 \
    --overwrite True \
    --outspec True \
    --fig True \
    --caldir /data/CAL --catdir /data/CAT \
    --configdir <PYAPS_DIR>/configs/ExGal_configs/
```

### 10.4 Restricting to specific targets

```bash
python aps_rvs.py \
    ... \
    --aps_ids 1006,1007,1005,1004 \
    --mask_aps_ids 1003 \
    --targsrvy WA \
    --targclass STAR
```

---

## 11. Running from Python

### 11.1 Direct call to proc_rvs

```python
from PyAPS.aps_rvs import proc_rvs

proc_rvs(
    infiles     = ["/data/L1/stacked_1002046.fit",
                   "/data/L1/stacked_1002045.fit"],
    classfile   = "/data/L2/3294/zbest_stacked_1002046__stacked_1002045.fits",
    param_fits  = "/data/L2/3294/rvs_stacked_1002046__stacked_1002045.fits",
    spec_fits   = "/data/L2/3294/rvsspec_stacked_1002046__stacked_1002045.fits",
    figdir      = "/data/L2/3294/figs/",
    config      = "<PYAPS_DIR>/configs/rvs_config.yaml",
    nthreads    = 4,
    overwrite   = True,
    wlranges    = [[4200.0, 6000.0], [6000.0, 8000.0]],
    arms_ratio  = [1.0, 0.83],
    join_arms   = False,
    sens_corr   = True,
    mask_gaps   = True,
    safe_mask_gaps = True,
    tellurics   = True,
    vacuum      = False,
    fill_gap    = True,
    catdir      = "/data/CAT",
    caldir      = "/data/CAL",
    configdir   = "<PYAPS_DIR>/configs/ExGal_configs/",
)
```

### 11.2 Using rvsweave directly with argument list

```python
from PyAPS.aps_rvs import rvsweave

rvsweave(options=[
    "--infiles",   "/data/L1/stacked_1002046.fit",
                   "/data/L1/stacked_1002045.fit",
    "--classfile", "/data/L2/3294/zbest_stacked_1002046__stacked_1002045.fits",
    "--config",    "<PYAPS_DIR>/configs/rvs_config.yaml",
    "--outpath",   "/data/L2/3294/",
    "--headname",  "stacked_1002046__stacked_1002045",
    "--wlranges",  "4200.0,6000.0",  "6000.0,8000.0",
    "--mp",        "4",
    "--overwrite", "True",
    "--outspec",   "True",
    "--fig",       "True",
    "--sens_corr", "True",
    "--mask_gaps", "True", "--safe_mask_gaps", "True",
    "--tellurics", "True", "--vacuum", "False", "--fill_gap", "True",
    "--arms_ratio", "1.0,0.83",
    "--join_arms", "False",
    "--caldir",    "/data/CAL",
    "--catdir",    "/data/CAT",
])
```

---

## 12. Input Parameters — Complete Reference

### APSOB / spectral processing parameters

These are forwarded unchanged to APSOB. See `APSOB_README.md` for full descriptions.

| Parameter | CLI | Default | Description |
|---|---|---|---|
| `infiles` | `--infiles` | required | L1 FITS file(s), one per arm |
| `wlranges` | `--wlranges` | `None` | Wavelength range per arm (e.g. `4200.0,6000.0`) |
| `aps_ids` | `--aps_ids` | `None` | Comma-separated APS IDs to restrict to |
| `targsrvy` | `--targsrvy` | `None` | Survey filter (e.g. `WA`) |
| `targclass` | `--targclass` | `None` | Class filter (e.g. `STAR`) |
| `mask_aps_ids` | `--mask_aps_ids` | `None` | APS IDs to exclude |
| `area` | `--area` | `None` | Elliptical aperture `RA,Dec,A,B,angle` |
| `mask_areas` | `--mask_areas` | `None` | Apertures to exclude |
| `sens_corr` | `--sens_corr` | `True` | Apply flux sensitivity correction |
| `mask_gaps` | `--mask_gaps` | `True` | Auto-detect and mask CCD gaps |
| `safe_mask_gaps` | `--safe_mask_gaps` | `True` | Mask safe gaps from lookup table |
| `tellurics` | `--tellurics` | `True`† | Mask telluric regions |
| `vacuum` | `--vacuum` | `False` | Convert to vacuum wavelengths |
| `fill_gap` | `--fill_gap` | `True` | Interpolate across masked gaps |
| `arms_ratio` | `--arms_ratio` | `None` | Flux scale per arm (e.g. `1.0,0.83` for OPR3B R-band) |
| `join_arms` | `--join_arms` | `False` | Stitch blue and red arms |
| `catdir` | `--catdir` | `None` | Catalogue directory |
| `caldir` | `--caldir` | `None` | Calibration directory (LSF files) |
| `configdir` | `--configdir` | `None` | Config directory (LSF-Config files) |

† `tellurics` default is `True` in APS production mode (changed from `False` in the rvsweave standalone default — see Section 15).

### RVS-specific parameters

| Parameter | CLI | Default | Description |
|---|---|---|---|
| `classfile` | `--classfile` | required | Redrock output FITS table with classifications |
| `config` | `--config` | required | Path to `rvs_config.yaml` |
| `outpath` | `--outpath` | required | Output directory |
| `headname` | `--headname` | `"headname"` | Root name for output files |
| `nthreads` | `--mp` | `1` | Number of parallel processes |
| `overwrite` | `--overwrite` | `False` | Overwrite existing outputs |
| `outspec` | `--outspec` | `False` | Write spectra + model to `rvsspec_*.fits` |
| `fig` | `--fig` | `False` | Write diagnostic plots to `figs/RVS_*.png` |

---

## 13. Output Files

### 13.1 Parameter table — `rvs_<headname>.fits`

Always produced. Contains one row per processed stellar target. Extension name `RVS_TABLE`. See Section 14 for all columns.

Header keys include:

| Key | Value |
|---|---|
| `APSVERS` | PyAPS version string |
| `APS_RV_V` | aps_rvs module version |
| `APSREF_0`, `APSREF_1`, ... | Basenames of L1 input files |
| `CSB_RVS` | join_arms status (`True`/`False`) |

### 13.2 Spectra table — `rvsspec_<headname>.fits`

Written only when `--outspec True`. Extension name `RVS_SPEC`. Contains one row per target with wavelength, flux, error, and best-fit model arrays for each arm.

Spectra header adds:

| Key | Value |
|---|---|
| `VACUUM` | `False` (wavelengths are in air) |
| `SAMPLING` | `0` (linear wavelength sampling) |

### 13.3 Diagnostic plots — `figs/RVS_<TARGID>_<CNAME>_<APS_ID>.png`

Written only when `--fig True`. One PNG per target showing the observed spectrum (black) and best-fit model (red) for each arm, split into two panels per arm (blue half and red half of each arm's wavelength range).

---

## 14. Output Columns — Complete Reference

### Parameter table (`RVS_TABLE` extension)

| Column | Unit | Description |
|---|---|---|
| `APS_ID` | — | WEAVE fiber APS ID |
| `TARGID` | — | WEAVE TARGID string |
| `CNAME` | — | WEAVE CNAME string |
| `VRAD` | km/s | Best-fit radial velocity |
| `VRAD_ERR` | km/s | Radial velocity uncertainty |
| `VRAD_CCF` | km/s | CCF initial velocity estimate (NaN if `ccf_init=False`) |
| `SKEWNESS_RVS` | km/s | Skewness of velocity PDF |
| `KURTOSIS_RVS` | km/s | Kurtosis of velocity PDF |
| `TEFF_RVS` | K | Effective temperature |
| `TEFF_ERR_RVS` | K | Effective temperature uncertainty |
| `LOGG_RVS` | dex | Surface gravity |
| `LOGG_ERR_RVS` | dex | Surface gravity uncertainty |
| `FEH_RVS` | dex | Iron abundance [Fe/H] |
| `FEH_ERR_RVS` | dex | Iron abundance uncertainty |
| `ALPHA_RVS` | dex | Alpha enhancement [α/Fe] |
| `ALPHA_ERR_RVS` | dex | Alpha enhancement uncertainty |
| `VSINI_RVS` | km/s | Projected rotational velocity |
| `SNR_RVS` | — | Mean S/N across all arms |
| `CHISQ_TOT_RVS` | — | Total fit chi-squared |
| `RVS_WARN` | — | Quality bitmask (see Section 8) |
| `DELTA_CHISQ` | — | χ²_continuum − χ²_fit (higher = better) |

### Spectra table (`RVS_SPEC` extension)

| Column | Unit | Description |
|---|---|---|
| `APS_ID` | — | |
| `TARGID` | — | |
| `CNAME` | — | |
| `LAMBDA_RVS_<ARM>` | Å | Observed wavelength array for each arm |
| `FLUX_RVS_<ARM>` | 1e-18 erg/(s cm² Å) | Flux array (bad pixels set to NaN) |
| `ERROR_RVS_<ARM>` | — | Error array (bad pixels set to NaN) |
| `MODEL_RVS_<ARM>` | 1e-18 erg/(s cm² Å) | Best-fit model (bad pixels set to NaN) |

`<ARM>` is the first character of the setup string, e.g. `B` for blue arm, `R` for red, `G` for green.

---


## 15. Troubleshooting

### No targets processed — "Empty tab"

```
Empty tab: Cannot generate the RVS output table
```

All threads returned `None`. Causes:

1. No targets in the classfile with `CLASS` in `['STAR', 'WD']` — check the classfile content
2. All qualifying targets were excluded by `aps_ids`, `targsrvy`, `targclass`, `area`, or `mask_aps_ids` filters
3. APSOB found no valid spaxels in the L1 files matching the criteria — check `aps_ids` overlap with the L1 file

### "skipping, products already exist"

`rvs_*.fits` or `rvsspec_*.fits` already exists and `--overwrite False`. Either delete the existing files or add `--overwrite True`.

### CCF fit fails / NaN velocities

The CCF failed to converge. This can happen for:
- Very low SNR spectra (SNR < 3)
- Spectra dominated by emission lines (the target may be misclassified as STAR)
- Template library does not cover the observed wavelength range

Check `DELTA_CHISQ` — a value near zero indicates the fit is no better than a featureless continuum.

### FWHM interpolation fails for a setup

```
APS_ID 1006 setup BLUEL: FWHM failed (...), using fixed resolution
```

The LSF file for this setup is missing from `caldir`/`configdir`, or the LSF pickle has not been generated yet. Run `aps_lsf` or `aps_fwhm` first, or pass `--caldir` pointing to a directory containing the `lsf_*.fits` files. The code falls back to fixed resolution and continues.

### "Update nthreads param to N"

The number of STAR/WD targets is smaller than `--mp`. This is normal — the code reduces `nthreads` automatically.

### Parameter pegged at grid edge (`PARAM_WARN` set)

The Teff, log g, or [Fe/H] fit landed at the edge of the template grid. This can indicate:
- A very hot or cool star outside the grid (check TEFF_RVS value against 2300–15000 K)
- A degenerate fit due to low SNR
- Wrong template library for this target type (e.g. subdwarfs, white dwarfs)

For white dwarfs (`WD`) the stellar parameter grid is not appropriate — only VRAD is meaningful.

### Arms ratio — R-band offset

If MOS blue and red arm fluxes look inconsistent at the join, try `--arms_ratio 1.0,0.83` for OPR3B data. For other runs consult the WEAVE calibration reports.

### Plots fail with X11 error

```
Failed to generate plots. 1- Check your X11 configs. 2- Check spectra in output fits file.
```

This is expected when running headlessly. The code catches the exception and continues. Ensure `matplotlib.use('Agg')` is active (it is set at module import time). The spectra data is still available in `rvsspec_*.fits`.

---

*For further information, bug reports, or contributions please open an issue on the PyAPS repository.*
