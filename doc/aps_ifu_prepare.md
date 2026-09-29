# PyAPS IFU Pipeline — Data Preparation and Analysis Guide

**PyAPS version:** 1.7+
**WEAVE Data Model:** 8.0 / 9.0
**Applies to:** WEAVE LIFU and MIFU observations
**Modes:** ExGal (galaxies, QSOs) and Gal (stars)
**Maintained by:** CASU, Institute of Astronomy, University of Cambridge

---

## Table of Contents

1. [What This Pipeline Does](#1-what-this-pipeline-does)
2. [Prerequisites](#2-prerequisites)
3. [Input Files](#3-input-files)
4. [Directory Layout](#4-directory-layout)
5. [Configuration Files](#5-configuration-files)
6. [Step 1 — Target Preparation and Segmentation](#6-step-1--target-preparation-and-segmentation)
7. [Step 2 — Source Classification and Redshift Measurement](#7-step-2--source-classification-and-redshift-measurement)
8. [Step 3 — The Patch File](#8-step-3--the-patch-file)
9. [Step 4 — Running ExGal Analysis](#9-step-4--running-exgal-analysis)
10. [Step 5 — Running Gal Analysis](#10-step-5--running-gal-analysis)
11. [Running as Part of the Full Pipeline](#11-running-as-part-of-the-full-pipeline)
12. [Configuration Parameter Reference](#12-configuration-parameter-reference)
13. [Diagnostic Plots](#13-diagnostic-plots)
14. [Output Files](#14-output-files)
15. [Troubleshooting](#15-troubleshooting)

---

## 1. What This Pipeline Does

The PyAPS IFU pipeline processes WEAVE Integral Field Unit (LIFU and MIFU)
observations from raw L1 stacked cubes to science-ready L2 data products.

The workflow has three stages:

```
┌─────────────────────────────────────────────────────┐
│  STAGE 1 — aps_ifu_prepare.py                       │
│                                                     │
│  • 2D source detection on the white-light image     │
│  • Elliptical aperture fitting per source           │
│  • Star / contaminant identification (optional)     │
│  • Redrock redshift fitting per source aperture     │
│  • Outputs a patch file — one row per source        │
└───────────────────┬─────────────────────────────────┘
                    │
          ┌─────────┴──────────┐
          ▼                    ▼
┌──────────────────┐  ┌──────────────────┐
│  STAGE 2a        │  │  STAGE 2b        │
│  aps_ExGal_      │  │  aps_Gal_        │
│  worker.py       │  │  worker.py       │
│                  │  │                  │
│  GALAXY / QSO    │  │  STAR targets    │
│                  │  │                  │
│  • Cube assembly │  │  • Cube assembly │
│  • Spatial bin   │  │  • Spatial bin   │
│  • Voronoi       │  │  • Voronoi       │
│  • Log-rebin     │  │  • Lin spectra   │
│  • pPXF          │  │  • FERRE         │
│  • EMIPPXF       │  │  • RVS           │
│  • Line strength │  │                  │
│  • L2 merge      │  │  • L2 merge      │
└──────────────────┘  └──────────────────┘
```

Stages 2a and 2b are independent and can run in parallel. Both depend on
Stage 1 completing successfully.

---

## 2. Prerequisites

### Software

- Python 3.9 or later
- PyAPS installed and importable (`import PyAPS`)
- Redrock installed (`import redrock`)
- ppxf installed (`import ppxf`)
- vorbin installed (`from vorbin.voronoi_2d_binning import voronoi_2d_binning`)
- astropy >= 5.0, numpy, scipy, matplotlib

For Gal mode additionally:
- FERRE compiled binary (`a.out`)
- RVS configuration file

### Python environment check

```python
import PyAPS
import redrock
import ppxf
from vorbin.voronoi_2d_binning import voronoi_2d_binning
print("PyAPS version:", PyAPS.__version__)
```

---

## 3. Input Files

### L1 stacked cubes

You need the two L1 stacked cube files produced by the WEAVE CPS pipeline,
one per spectrograph arm:

```
stackcube_<OBSID_blue>.fit    ←  blue arm  (3800–5950 Å observed)
stackcube_<OBSID_red>.fit     ←  red arm   (5900–9280 Å observed)
```

Both files must be present for standard processing. Single-arm processing is
not supported in IFU mode.

The file names follow the convention `stackcube_<OBSID>.fit` where `OBSID`
is the WEAVE observation block ID. The blue arm always has the lower OBSID
of the pair (for LIFULR11 mode).

### Calibration and catalogue files

```
CAL/    ←  photometric calibration, LSF files, sensitivity curves
CAT/    ←  WEAVE input catalogue (for coordinate and redshift priors)
```

Paths to these directories are set in the master pipeline config YAML file
(`pipeline_params.yaml`) and passed via `--caldir` and `--catdir`.

---

## 4. Directory Layout

The pipeline creates all outputs under the `L2/` tree. A typical run
for observation `YYYYMMDD`, output number `OBSNUM`, produces:

```
L2/YYYYMMDD/OBSNUM/
│
├── <headname>_targets.fits           ←  raw patch table (Stage 1 output)
├── <headname>_targets_mod.fits       ←  edited patch table (input to Stage 2)
│
├── <headname>_P0001_table.fits       ←  Voronoi table, patch 1
├── <headname>_P0001_BINSpectra.fits  ←  log-rebinned spectra, patch 1
├── <headname>_P0001_ppxf.fits        ←  pPXF results (if PPXF=True)
├── <headname>_P0001_emcee.fits       ←  EMIPPXF results (if EMIPPXF=True)
├── <headname>_P0001_ls.fits          ←  line strengths (if LS=True)
├── <headname>_P0001_APS.fits         ←  merged L2 product
│
├── <headname>_P0002_...              ←  same for patch 2 (second source)
│
├── figs_ExGal/                       ←  all ExGal diagnostic figures
│   ├── <headname>_P0001_snr_stages.png
│   ├── <headname>_P0001_prep_raw.png
│   ├── <headname>_P0001_prep_spatialbin.png
│   ├── <headname>_P0001_prep_voronoi.png
│   └── ...
│
├── figs_Gal/                         ←  all Gal diagnostic figures
│
├── scripts/                          ←  generated bash and slurm scripts
└── logs/                             ←  stdout / stderr from each stage
```

The `headname` is built from the two input file names:
`stackcube_<ID_blue>__stackcube_<ID_red>`.

---

## 5. Configuration Files

### IFU parameter JSON files

Each survey mode has a JSON config file that controls all pipeline parameters.
These live under `configs/ExGal_configs/` or `configs/Gal_configs/`:

```
configs/
├── ExGal_configs/
│   ├── LIFULR11.json       ←  LIFU low-resolution 1.1" fibre (main ExGal config)
│   ├── LIFUHR11.json       ←  LIFU high-resolution
│   ├── MIFULR12.json       ←  MIFU low-resolution 1.2" fibre
│   └── LSF-Config_*        ←  LSF template files (one per SSP library)
└── Gal_configs/
    ├── LIFULR11_GAL.json
    └── ...
```

The config file to use is determined automatically from the L1 file header
(instrument mode and fibre size) if `--IFU_params` is not provided.
It can always be overridden with `--IFU_params /path/to/config.json`.

### Master pipeline config

When running as part of the automated pipeline, a YAML file
(`pipeline_params.yaml`) holds paths and high-level switches. When running
standalone, all equivalent parameters are passed as command-line arguments.

---

## 6. Step 1 — Target Preparation and Segmentation

`aps_ifu_prepare.py` is the entry point for the full pipeline. It performs
source detection, aperture fitting, star masking, and Redrock classification
in a single run.

### What it does

1. Collapses the IFU cube to a white-light image (flux summed over all wavelengths)
2. Runs 2D segmentation on the white-light image to detect all extended
   sources in the field
3. Fits an elliptical aperture to each detected source
4. Cross-matches with Gaia (optional) to identify foreground stars
5. Runs Redrock on the co-added spectrum of each aperture to measure
   redshift and classify the source
6. Writes the patch file — a FITS table with one row per source

### Running Stage 1

```bash
python aps_ifu_prepare.py \
    --infiles /data/L1/20240808/stackcube_3071431.fit \
              /data/L1/20240808/stackcube_3071430.fit \
    --headname stackcube_3071431__stackcube_3071430 \
    --outpath  /data/L2/20240808/11182/ \
    --IFU_config_dir <PYAPS_DIR>/configs/ExGal_configs/ \
    --IFU_params     <PYAPS_DIR>/configs/ExGal_configs/LIFULR11.json \
    --wlranges 3800.0,5950.0  5900.0,9280.0 \
    --arms_ratio 1.0,1.0 \
    --sens_corr True --mask_gaps True --safe_mask_gaps True \
    --tellurics True --join_arms True \
    --seg2d_white_src     weave \
    --seg2d_search_gaia   False \
    --seg2d_extract       True \
    --seg2d_exclude_ctarg True \
    --seg2d_mask_stars    False \
    --seg2d_ext_thresh    2.5 \
    --seg2d_minarea       300 \
    --seg2d_deblend_nthresh 4 \
    --seg2d_radii_factor  8.0 \
    --class_patch         True \
    --class_templates     <PYAPS_DIR>/PyAPS_templates/templates_RR_PCA/ \
    --class_templates_ARC <PYAPS_DIR>/PyAPS_templates/templates_RR_ARC/ \
    --class_z_rad         2.5 \
    --caldir /data/CAL \
    --catdir /data/CAT
```

### Segmentation parameters explained

These control how sources are detected in the white-light image.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `--seg2d_white_src` | str | `weave` | Source of white-light image. `weave` collapses the IFU cube. `hst` uses an external HST image if available. |
| `--seg2d_search_gaia` | bool | `False` | Cross-match with Gaia DR3 to identify stars. Requires internet access or a local Gaia catalogue. |
| `--seg2d_extract` | bool | `True` | Run source extraction. Set `False` for MIFU mode (single target, no surrounding field). |
| `--seg2d_exclude_ctarg` | bool | `True` | Exclude the central WEAVE catalogue target from extraction to avoid double-counting. |
| `--seg2d_mask_stars` | bool | `False` | Mask Gaia-identified stars. Only useful if `seg2d_search_gaia=True`. |
| `--seg2d_ext_thresh` | float or `None` | `None` (all shipped `IFU_params` JSON configs) | Detection threshold in units of local RMS noise. Lower values detect fainter sources but increase false positives. **`None` auto-estimates it from the white-light image** -- see below. |
| `--seg2d_minarea` | int or `None` | `None` (all shipped `IFU_params` JSON configs) | Minimum source area in pixels. Sources smaller than this are rejected as noise peaks or cosmic rays. **`None` auto-estimates it from the white-light image** -- see below. |
| `--seg2d_deblend_nthresh` | int or `None` | `4` | Number of deblending sub-thresholds. Higher values separate close pairs more aggressively. `None` is accepted for consistency but simply resolves to the fixed default `4` -- **not** data-driven, deliberately (this parameter is left alone rather than auto-tuned). |
| `--seg2d_radii_factor` | float | `8.0` | Kron radius scale factor. Multiplied by the fitted Kron radius to define the aperture boundary. Larger values capture more outer flux at the cost of more contamination. |

> **MIFU mode:** For MIFU observations `seg2d_extract` and
> `seg2d_exclude_ctarg` are automatically forced to `False` since MIFU
> points at a single pre-selected object with no surrounding field to segment.

#### `None` -> auto-estimated ext_thresh/minarea (data-driven, not guessed)

**This is the shipped default** in every `configs/ExGal_configs/*.json` file
(`"seg2d_ext_thresh": "None"`, `"seg2d_minarea": "None"`) as of this writing
-- replacing a previous fixed default of `minarea=300`, found on real WEAVE
data to be badly miscalibrated: it detected only 3 objects on a field where
the auto-estimated value detected 51, on the identical image. `minarea=300`
requires ~75 sq.arcsec of connected area at 0.5"/pix, far larger than any
compact source's real footprint -- it was simply too large a guess, not a
principled choice.

When either parameter resolves to `None` (from the CLI, or from
`IFU_params` JSON -- see [Per-survey configuration via
JSON](#per-survey-configuration-via-json)), `ifu_seg2d` calls
`aps_ifu_seg3d.estimate_seg2d_params()` on the white-light image before
running sep:

1. **minarea** is derived from the image's OWN PSF size: a permissive
   first sep pass finds the most compact, round, well-detected sources in
   the field, measures their FWHM (standard `2.3548*sqrt(a*b)` second-
   moment convention), and sets `minarea` to the pixel area of a disk that
   wide -- a source narrower than one full PSF-width is not a resolved
   real detection.
2. **ext_thresh** is calibrated empirically at that minarea: sep is run on
   both the background-subtracted image and its sign-flipped negative at a
   grid of trial thresholds, and the smallest (most complete/permissive)
   threshold whose purity (`1 - n_neg/n_pos`) still clears 0.9 is chosen --
   the same sign-flip philosophy `post_veto_purity_scan` uses for seg3d,
   applied here to plain 2D extraction.

On the validated test field this recommended `ext_thresh=2.5` right back
(purity=0.911) -- i.e. the auto-calibration independently confirmed the old
`ext_thresh` guess was actually fine; only `minarea` was broken. Both
measurements, and the full ext_thresh purity curve, are printed to stdout
when auto-estimation runs, so every value used is traceable, not silent.

To go back to a fixed value for a specific survey, set that key to an
explicit number (e.g. `"seg2d_minarea": "50"`) in its `IFU_params` JSON
instead of `"None"`.

### Redshift classification parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `--class_patch` | bool | `True` | Run Redrock on each aperture. Set `False` only if you have reliable external redshifts and want to skip Redrock entirely. |
| `--class_templates` | str | — | Path to Redrock PCA template directory. |
| `--class_templates_ARC` | str | — | Path to Redrock archetype template directory. Used for the fine-fitting fallback after PCA classification. |
| `--class_z_rad` | float | `2.5` | Search radius in arcsec for a catalogue redshift prior. A prior within this radius seeds the Redrock fit. |
| `--class_ntop` | int | `3` | Number of top Redrock solutions to store per source. |

### Spectral processing flags

These flags apply identically to Stage 1 and both Stage 2 runs. They must
be consistent across all three stages.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `--wlranges` | list | `None` | Wavelength ranges per arm as `lmin,lmax` pairs. Example: `3800.0,5950.0 5900.0,9280.0`. `None` uses the full arm coverage. |
| `--arms_ratio` | str | `1.0,1.0` | Flux scale factors for blue and red arms. Adjust if the arms are not on the same flux scale after sky subtraction. |
| `--sens_corr` | bool | `True` | Apply sensitivity correction from calibration files. |
| `--mask_gaps` | bool | `True` | Mask the inter-arm gap (5900–5950 Å). Strongly recommended — always use `True`. |
| `--safe_mask_gaps` | bool | `True` | Apply a conservative buffer around the gap mask edges. |
| `--tellurics` | bool | `True` | Apply telluric absorption correction. |
| `--vacuum` | bool | `False` | Convert wavelengths from air to vacuum. Default is air wavelengths. |
| `--fill_gap` | bool | `False` | Interpolate across the inter-arm gap. Not recommended unless specifically required. |
| `--join_arms` | bool | `True` | Stitch blue and red arms into a single spectrum. Required for IFU mode — always `True`. |

### Per-survey configuration via JSON

All segmentation and classification parameters can alternatively be specified
in the IFU_params JSON config file so that each survey has its own tuned
values without needing to change the command line:

```json
{
    "seg2d_white_src"        : "weave",
    "seg2d_search_gaia"      : "False",
    "seg2d_extract"          : "True",
    "seg2d_exclude_ctarg"    : "True",
    "seg2d_mask_stars"       : "False",
    "seg2d_ext_thresh"       : "2.5",
    "seg2d_minarea"          : "300",
    "seg2d_deblend_nthresh"  : "4",
    "seg2d_radii_factor"     : "8.0",
    "seg3d"                  : "False",
    "seg3d_merge"            : "False",
    "seg3d_lmin"             : "3700.0",
    "seg3d_lmax"             : "9390.0",
    "seg3d_threshold"        : "6.0",
    "seg3d_spatial_bin"      : "4",
    "seg3d_continuum_method" : "pca",
    "seg3d_ivar_calibration" : "per-wave",
    "seg3d_min_npix"         : "3",
    "seg3d_merge_min_snr"    : "10.0",
    "seg3d_merge_min_purity" : "0.9",
    "seg3d_merge_aperture_arcsec" : "2.0",
    "seg3d_merge_group_min_lines" : "2",
    "seg3d_merge_group_min_snr"   : "10.0",
    "seg3d_merge_group_min_n_trust" : "3",
    "seg3d_group_radius_px"       : "0.0",
    "seg3d_merge_extreme_safety_margin" : "1.1",
    "class_patch"            : "True",
    "class_z_rad"            : "2.5"
}
```

Unlike `seg2d` (a YAML-only switch), **`seg3d` and `seg3d_merge` themselves
are also JSON-configurable**, per-survey, exactly like every other
`seg3d_*` parameter above — both still default to `'False'` in the YAML
`script_params` file, so nothing changes unless a survey's JSON explicitly
turns them on.

Priority order: **CLI argument > IFU_params JSON > hardcoded default**.

### Output of Stage 1

```
<headname>_targets.fits
```

A FITS binary table with one row per detected source. Key columns:

| Column | Format | Description |
|---|---|---|
| `id` | int | Source ID within this field |
| `RA_icrs` | float | Right ascension (degrees, ICRS) |
| `DEC_icrs` | float | Declination (degrees, ICRS) |
| `A_world` | float | Semi-major axis (degrees) |
| `B_world` | float | Semi-minor axis (degrees) |
| `angle` | float | Position angle (degrees east of north) |
| `type` | str | `T` = target, `C` = central WEAVE object, `M` = mask region |
| `Z` | float array | Redshift (array of top N solutions) |
| `ZERR` | float array | Redshift uncertainty |
| `ZWARN` | int array | Redrock warning bitmask |
| `CLASS` | str array | Classification: `GALAXY`, `QSO`, `STAR`, `WD` |
| `SUBCLASS` | str array | Redrock sub-classification |

### Optional: 3D Matched-Filter Emission-Line Detection (seg3d)

`seg2d` collapses the cube to a single white-light image before running
SExtractor, which is the right tool for continuum sources but structurally
cannot find a source whose signal is concentrated in only a few spectral
channels (e.g. an isolated emission line): summing over the full bandpass
dilutes such a feature by roughly `sqrt(n_channels)` while adding
`n_channels` worth of noise, so an 8-sigma detection in 5 channels can be
a ~1-sigma bump in a 4000-channel collapse.

`seg3d` is an **optional, additive** companion detection pass using a
3D matched filter (continuum removal → ivar noise recalibration → a bank
of Gaussian spatial-PSF × spectral-line-profile matched filters →
3D connected-component candidate extraction → an artifact-veto chain →
an empirical purity self-check via the sign-flipped cube). It is **off
by default** (`--seg3d False`) and never modifies `seg2d`'s output, the
patch file, or classification — it writes its own candidate table and a
comparison figure alongside whatever `seg2d` already produces, and a
seg3d failure cannot break a preparation run (caught and logged as a
warning).

Cube loading (`aps_ifu_seg3d.load_wavelength_range`) is built directly on
`aps_utils.APSOB` — the same established, tested L1-reading pipeline every
other `aps_ifu_*.py` script uses — rather than a parallel hand-rolled FITS
reader. This is why `--sens_corr`/`--mask_gaps`/`--safe_mask_gaps`/
`--tellurics`/`--vacuum` (already-existing flags shared with the rest of
the pipeline, see [Spectral processing
flags](#spectral-processing-flags)) now also govern seg3d's input data:
sensitivity-corrected flux, CCD-gap masking (both the basic and the
per-camera-lookup-table "safe" version), and telluric masking are all
applied before detection, exactly as they would be for a normal spectral
extraction. `join_arms` is enabled automatically whenever seg3d's
requested wavelength range spans more than one arm's file among
`--infiles`, and APSOB's own arm-overlap handling (comparing/normalizing
each arm's statistical weight where they physically overlap) applies —
not a seg3d-specific rule.

> **Status: experimental / diagnostic.** Always check the purity-vs-
> threshold panel in the comparison figure and treat the candidate list as
> a starting point for visual inspection, not a final catalogue. See
> `aps_ifu_seg3d.py`'s module docstring for the method and the real-data
> findings baked into its defaults (ivar noise recalibration, PCA/
> eigenspectrum continuum subtraction). The specific purity numbers quoted
> during this module's development were measured before it was switched
> to APSOB-based loading (i.e. before sensitivity correction and real gap
> masking were applied) and should not be treated as current — re-run the
> purity self-check on your own field rather than trusting a historical
> number from the docstring/commit history.

#### Running seg3d

```bash
python aps_ifu_prepare.py \
    --infiles /data/L1/20240808/stackcube_3071431.fit \
              /data/L1/20240808/stackcube_3071430.fit \
    --headname stackcube_3071431__stackcube_3071430 \
    --outpath  /data/L2/20240808/11182/ \
    --IFU_config_dir <PYAPS_DIR>/configs/ExGal_configs/ \
    --IFU_params     <PYAPS_DIR>/configs/ExGal_configs/LIFULR11.json \
    --wlranges 3800.0,5950.0  5900.0,9280.0 \
    --arms_ratio 1.0,1.0 \
    --sens_corr True --mask_gaps True --safe_mask_gaps True \
    --seg2d      True  \
    ... (all the usual Stage 1 arguments) \
    --seg3d                  True \
    --seg3d_lmin              3700.0 \
    --seg3d_lmax              9390.0 \
    --seg3d_threshold         6.0 \
    --seg3d_spatial_bin       4 \
    --seg3d_continuum_method  pca \
    --seg3d_ivar_calibration  per-wave
```

#### seg3d parameters explained

| Parameter | Type | Default | Description |
|---|---|---|---|
| `--seg3d` | bool | `False` | Enable the 3D matched-filter detection pass. |
| `--seg3d_lmin`, `--seg3d_lmax` | float | `3700.0`, `9390.0` | The wavelength range to search, in Angstrom — **replaces an earlier `--seg3d_arm` + `--seg3d_edge_trim_A` interface**. Not tied to a single arm: pass a range within one arm, or one that spans into/across several — `load_wavelength_range` (via APSOB) draws on whichever file(s) among `--infiles` actually cover it, generic over however many arms are present. There is no separate edge-trim setting any more; picking `lmin`/`lmax` with enough clearance from an arm's true edge (worse flat-fielding/wavelength-solution reliability there) is now entirely the caller's choice, the same way `--wlranges` already works elsewhere in this pipeline. **Defaults cover the LOWRES (LIFULR/MIFULR) blue+red combined native range** — measured directly from real L1 headers (blue 3600–5955.5Å, red 5820–9490.5Å, ~135Å of arm overlap), trimmed by the same ~100Å margin the old `edge_trim_A` used to apply automatically. **HIGHRES (LIFUHR/MIFUHR) surveys must NOT use this default** — HR's native coverage (blue 4695–5505Å, red 5930–6840Å) has a real *gap* between the two arms and does not overlap the LOWRES-derived default window at all; every HIGHRES entry in `configs/ExGal_configs/` already overrides `seg3d_lmin`/`seg3d_lmax` (`4800.0`/`6740.0`) via its own `IFU_params` JSON, so this only matters for a caller invoking `aps_ifu_prepare.py`/`run_seg3d` directly without going through the shipped HR config. |
| `--seg3d_threshold` | float | `6.0` | Matched-filter SNR threshold for the primary candidate list. |
| `--seg3d_spatial_bin` | int | `4` | Spatial rebinning factor before detection (`4` with native 0.5" pixels ≈ 2" bins). Mainly a speed lever (~16x fewer spaxels at the default, ~10x faster end to end) but also reduces the effect of correlated noise on an oversampled/drizzled cube. |
| `--seg3d_sky_mask_min_A` | float | `None` | Mask everything redward of this wavelength (OH airglow forest) — relevant whenever `seg3d_lmax` reaches into the red arm, which the LOWRES **default now does** (unlike the previous BLUE-only default). **Not yet wired into any `IFU_params` JSON or `script_params_*.yaml`** (unlike every other `seg3d_*` parameter — see [Per-survey configuration via JSON](#per-survey-configuration-via-json)), and no empirical OH-forest cutoff has been measured for this pipeline the way `k_global`/`k_wave` are for the ivar calibration; `None` (no masking) is applied by default even though the default window now spans well into the red arm's airglow-affected region. Treat this as an open follow-up, not a validated default — pass an explicit value (e.g. `7500`) for any LOWRES field where red-arm OH residuals are a concern until it is measured and wired through like the other parameters. |
| `--seg3d_continuum_method` | str | `pca` | `mean` / `robust` / `pca`. Keep `pca` — it is the only one of the three that removes real absorption/emission line *shapes* rather than just continuum trend (validated on real data). |
| `--seg3d_ivar_calibration` | str | `per-wave` | `global` / `per-wave`. Corrects the ivar cube's measured noise-underestimation — this is the single most important seg3d step; do not disable. |
| `--seg3d_min_npix` | int | `3` | Minimum voxel count of a candidate's thresholded 3D blob (paired with the existing `max_npix=5000` internal ceiling). A 1-2 voxel blob is narrower than the matched-filter kernel's own support and cannot be a genuine resolved detection of that template — almost always a single anomalously-weighted pixel. See [Avoiding single-spaxel detections](#avoiding-single-spaxel-detections-min_npix) below. |
| `--seg3d_fig` | bool | `True` | Write the 2D-vs-3D comparison figure. |

Every parameter above, **including the `seg3d`/`seg3d_merge` on/off
switches themselves**, follows the same JSON > YAML > hardcoded-default
priority as the `seg2d_*` family (see [Per-survey configuration via
JSON](#per-survey-configuration-via-json)) — set them per-survey in
`IFU_params` JSON rather than only in the shared `script_params` YAML.
`--sens_corr`/`--mask_gaps`/`--safe_mask_gaps`/`--tellurics`/`--vacuum`
are NOT seg3d-specific — they're the same shared flags used by the rest
of the pipeline (see [Spectral processing
flags](#spectral-processing-flags)), now also consumed by seg3d's cube
loading rather than silently ignored by it.

#### Avoiding single-spaxel detections (min_npix)

Real WEAVE data showed the opposite failure mode from the edge artifacts
above: some candidates surviving every other veto were literally 1 voxel
in size — smaller than the matched-filter template's own kernel support,
which cannot happen for a genuinely resolved detection of that template.
`spatial_extent_veto` now enforces both a floor (`min_npix`, default `3`)
and the existing ceiling (`max_npix`, default `5000`) — the floor is
conservative (well below the ~28-voxel median blob size on the validated
test cube) so it only removes the most degenerate 1-2 voxel cases, not
real compact sources.

#### Edge-of-field artifacts (min_edge_dist_px)

Also confirmed on real data: 42/128 candidates (33%) in one test field sat
within 2 binned pixels of the spatial coverage-mask boundary, and two of
the three highest-SNR "candidates" were exactly 1 pixel from the edge —
the spatial matched-filter kernel there has necessarily lost real support
on one side, which `min_coverage_frac` inside `matched_filter_snr` did not
always catch cleanly by itself. Two changes address this:

- `edge_distance_veto` rejects any candidate within `min_edge_dist_px`
  (default `3.0`, sized to the kernel's own ~2-pixel radius at this
  module's default spatial binning) of the nearest edge of the coverage
  mask. Not exposed as a separate CLI flag (its 3px default tracks the
  kernel's own footprint rather than being a free-standing survey-level
  choice) — pass `min_edge_dist_px` to `aps_ifu_seg3d.run_seg3d()`
  directly for a different bin factor/PSF.
- `bin_cube_spatial`'s partial-coverage acceptance threshold
  (`min_valid_frac`) was raised from 0.5 to 0.8, so a boundary bin that
  was only marginally covered gets excluded from the detection grid
  entirely, rather than kept in with reduced but nonzero weight.

#### Known limitations

- Does not attempt the multi-EXPOSURE reprojection/mosaicking
  `ifu_seg2d.create_white_image` does for its white-light image (multiple
  exposures of the SAME arm) — APSOB keeps only the first file it finds
  per arm. Multi-ARM combination (e.g. blue+red for the same pointing) is
  fully supported, generic over however many arms are present, not a
  separate limitation.
- Only the sub-flags listed above are exposed on the CLI; the remaining
  tunables (PCA rank, resolvedness/coincidence thresholds, bright-source
  mask radius, merge purity_target/min_n_trust, ...) use their validated
  defaults in `aps_ifu_seg3d.run_seg3d()` / `merge_into_patch_table()` —
  call those functions directly from Python for finer control.
- No dedicated **spectral-edge veto** exists yet (unlike the spatial
  `edge_distance_veto`) — a candidate whose line sits within ~20-30Å of
  `seg3d_lmin`/`seg3d_lmax` can be a matched-filter-kernel edge-support
  systematic rather than real flux, and nothing in the veto chain catches
  this specifically (checked empirically on a real field: the last 30Å of
  the window had an 11.0% candidate rate vs 5.9% in the sign-flipped cube
  at the same band — an asymmetry the purity self-check does not resolve,
  since it tests coincidence/grouping rates, not this spectral-selection
  effect). Treat any candidate that close to either edge with extra
  scrutiny; a proper fix (or re-running with `lmin`/`lmax` pulled back
  further) is still open follow-up work.
- `seg3d_sky_mask_min_A` (OH-airglow-forest masking) is not wired into any
  `IFU_params` JSON or `script_params_*.yaml` and has no empirically
  measured default — see its row in [seg3d parameters
  explained](#seg3d-parameters-explained). This matters more than it used
  to now that the LOWRES default `seg3d_lmax` reaches well into the red
  arm's airglow-affected region (see below).

#### Output

```
<headname>_seg3d_candidates.fits
figs/<headname>_seg3d_compare.pdf
```

`<headname>_seg3d_candidates.fits` — one row per surviving candidate,
sorted by SNR descending:

| Column | Format | Description |
|---|---|---|
| `id` | int | Candidate ID |
| `ix`, `iy` | int | Pixel position on the (spatially binned) detection grid |
| `RA_icrs`, `DEC_icrs` | float | Sky position (degrees, ICRS) |
| `wave_A` | float | Wavelength of the detection peak (Angstrom) |
| `snr` | float | Matched-filter signal-to-noise |
| `npix` | int | Voxel count of the thresholded 3D blob |
| `fwhm_template_pix` | float | Spectral-width template (from the bank) that produced this detection |
| `no_continuum_counterpart` | bool | `True` if no seg3d-internal sep continuum source lies within 4 pixels — i.e. a genuine "line-only" candidate |
| `group_id` | int | Spatial group index from `group_multiline_candidates` (0-indexed; shared by candidates at the same detection-grid pixel, regardless of wavelength — see [Multi-line grouping](#multi-line-grouping-seg3d_merge_group_) below) |
| `n_group_lines` | int | Total candidate count sharing this `group_id` (`1` for an isolated single-line detection) |

#### Merging RELIABLE candidates into the real patch table (seg3d_merge)

`seg3d` alone only ever writes its own side files (above) — it never
touches `_targets.fits`. Setting **`--seg3d_merge True`** (also opt-in,
also default `False`, requires `--seg2d True` as well since there must be
an existing patch table to merge into) additionally promotes seg3d
candidates that are both (a) genuinely reliable and (b) have weak/no
continuum — i.e. exactly the sources `seg2d` structurally cannot see —
into real auxiliary targets in the patch table, so they get processed by
Redrock classification and the same ExGal/Gal extraction as any
2D-detected source. The design goal is a **short, trustworthy** list, not
maximum recall — see below for why a fixed SNR cut alone was not enough.

**Why the threshold is data-driven, not a fixed number.** An earlier
version simply required `snr >= 10`. Digging into the purity self-check
(`post_veto_purity_scan`) on real data showed this was not safe to trust
blindly: at a dense threshold grid, purity swung noisily between ~0.25 and
~0.91 above SNR~10, purely because the candidate count (`n_pos`) had
dropped into the single-to-low-double digits there — a ratio computed from
4-5 candidates is not a measurement, however good it looks. `merge_into_patch_table`
now calls `resolve_merge_min_snr`, which:

1. Only considers scanned thresholds with **`n_pos >= min_n_trust`**
   (default `15`) candidates — too few to trust are excluded from
   consideration entirely, regardless of their purity ratio.
2. Among the remaining, sufficiently-sampled thresholds, picks the
   smallest one whose purity `1 - n_neg/n_pos` still clears
   **`purity_target`** (default `0.8`) — the same "most permissive
   threshold that still clears the bar" principle
   `aps_ifu_seg3d.calibrate_sep_purity` already uses for seg2d's
   ext_thresh, applied here to the merge decision.
3. The final effective threshold is `max(seg3d_merge_min_snr, that
   calibrated value)` — `seg3d_merge_min_snr` is a **floor**, not the
   value actually used; the calibration can only raise it.
4. If no threshold in the scan is both well-sampled and clears the purity
   bar, **merging is disabled for that field entirely** (0 targets added)
   rather than falling back to an uncalibrated guess. This is a legitimate,
   expected outcome for a field with few real line-only emitters — not a
   bug.

A candidate is merged only if it additionally has no continuum counterpart
(the entire point of `seg3d_merge`) and is not already within 3″ of an
existing target (avoids duplicating a source `seg2d` already found).

```bash
--seg3d        True \
--seg3d_merge  True \
--seg3d_merge_min_snr 10.0   # floor only -- the effective threshold used
                              # may be higher, per-field, see above
```

**Provenance and safety:**
- Merged rows use `type='T'` (indistinguishable from a normal target to
  every downstream consumer) but are marked with **`flag=3.0`** — the
  `flag` column is otherwise always `0.0` and unused anywhere in the
  current pipeline (confirmed by direct code inspection of every live
  consumer before repurposing it), so a merged target can always be
  identified later with `patch_table['flag'] == 3.0`, e.g. to bulk-remove
  them if a field turns out to be too contaminated.
- Merged targets get a fixed 2″-radius circular aperture centred on the
  candidate position.
- Even a purity-calibrated threshold is a *field-level* statistic, not a
  per-candidate guarantee — always check the seg3d comparison figure's
  purity panel, and treat `flag == 3.0` targets as a flag for extra
  scrutiny in any downstream inspection, not as equivalent-confidence to a
  seg2d detection.
- `resolve_merge_min_snr`'s `min_n_trust` (single-line path, default `15`)
  is not yet exposed as its own CLI flag (only reachable by calling
  `aps_ifu_seg3d.merge_into_patch_table()` directly) — `purity_target`
  (`--seg3d_merge_min_purity`) is shared between the single-line path here
  and the multi-line path below, and the multi-line path's own
  `group_min_n_trust` *is* exposed (`--seg3d_merge_group_min_n_trust`, see
  below).
- A seg3d/merge failure is caught and logged as a warning — it can never
  break the seg2d/classification pipeline that already ran.

#### Multi-line grouping (`seg3d_merge_group_*`)

The SNR/purity path above treats every candidate as an independent
detection — but a real emitter with several detected lines (e.g.
[OIII]4959/5007 + Hβ, or a pair of lines either side of a masked gap) shows
up as *multiple* rows at the *same sky position*, each individually
possibly too weak to clear the single-line purity-calibrated threshold on
its own. `seg3d_merge_group_*` is an **additional, independent** path into
`seg3d_merge` that catches exactly this case, evaluated in parallel with
(not instead of) the SNR/purity path — a candidate qualifies for merging if
either path accepts it.

A **group** is a set of `no_continuum_counterpart` candidates at the exact
same detection-grid pixel (`group_multiline_candidates`, `ix`/`iy` match,
**not** a wavelength window — real multi-line sources are typically well
separated in wavelength and can never be joined by the connected-component
extraction itself, so no wavelength check is needed to avoid conflating
unrelated candidates). A group of `>= seg3d_merge_group_min_lines`
(default `2`) members qualifies if its **combined significance**
(`aps_ifu_seg3d.combined_group_snr` — the quadrature sum, `sqrt(sum(snr_i**2))`,
over *every* member's SNR, not just the strongest one, so several
weak-but-real lines can out-rank a single marginal line) clears the
effective group threshold.

**The effective group threshold is field-calibrated, exactly like the
single-line path's threshold**, just calibrated on *groups* rather than
individual candidates:

1. `run_seg3d` builds a sign-flipped-cube candidate list (`cands_neg`, same
   veto chain, same primary threshold as the real `cands`) and calls
   `post_veto_group_purity_scan(cands, cands_neg, ...)`, returning
   `group_purity_scan` — the same `(threshold, n_pos, n_neg, purity)` shape
   as `purity_scan`, but counting *groups* whose combined SNR clears each
   threshold, real vs sign-flipped.
2. `merge_into_patch_table` calls `resolve_group_min_snr(group_purity_scan,
   min_snr_floor=seg3d_merge_group_min_snr, purity_target=seg3d_merge_min_purity,
   min_n_trust=seg3d_merge_group_min_n_trust)` — same "smallest sufficiently-
   sampled threshold that still clears the purity bar" logic as
   `resolve_merge_min_snr`, with `seg3d_merge_group_min_snr` as the floor
   (default `10.0`, can only be raised by calibration, never lowered).
3. `seg3d_merge_group_min_n_trust` (default `3`, **not** `15`) is
   deliberately much lower than the single-line path's `min_n_trust` —
   multi-line groups are intrinsically rarer than individual candidates by
   construction, so demanding 15 trustworthy groups would make this path
   unusable on most fields. This is an explicit trade-off (a purity ratio
   from 3-4 groups carries more sampling noise than one from 15
   candidates), not a like-for-like match to the single-line default.
4. If no threshold in the group scan is both well-sampled and clears the
   purity bar, the multi-line path is disabled for that field — but the
   single-line path above is evaluated completely independently and may
   still merge candidates (and vice versa: a single-line calibration
   failure does not disable the group path).

Neither path checks that a group's lines sit at a physically consistent
redshift — that is deliberately left to the Redrock classification pass
every merged target (single- or multi-line) already goes through
immediately after this step, rather than duplicating that machinery here.

```bash
--seg3d                       True \
--seg3d_merge                 True \
--seg3d_merge_group_min_lines 2 \
--seg3d_merge_group_min_snr   10.0 \
--seg3d_merge_group_min_n_trust 3 \
--seg3d_group_radius_px       0.0    # exact-pixel grouping only -- validated
                                      # against the sign-flipped cube; a 1px
                                      # tolerance let noise-group
                                      # contamination rise sharply with no
                                      # compensating gain in real groups
                                      # recovered, see
                                      # group_multiline_candidates's
                                      # docstring
```

Set `--seg3d_merge_group_min_lines None` to disable the multi-line path
entirely and fall back to single-line-only merge behaviour.

#### Extreme-S/N path (`seg3d_merge_extreme_safety_margin`)

A THIRD, independent path into `seg3d_merge`, for a candidate that clears
**neither** path above — typically a genuinely extreme-S/N **singleton**
(no corroborating line, so the multi-line path cannot help) whose S/N is
so high that the single-line purity self-check simply never gets measured
with enough samples (`n_pos >= min_n_trust`) that far out on the curve to
certify it via path (a) either. Rather than leave such a candidate
unmerged purely on a statistical-power technicality, this path compares
its S/N directly against the single loudest fluctuation the **entire**
sign-flipped search ever produced (`aps_ifu_seg3d.loudest_event_snr` —
every voxel of every template width in the sign-flipped cubes, no veto or
threshold applied first; `run_seg3d`'s returned `loudest_neg_event`), the
same "loudest event" fallback used to validate extreme-significance
detections once a ratio-based background estimate runs out of trials
(e.g. gravitational-wave searches):

```
effective_threshold = loudest_neg_event * seg3d_merge_extreme_safety_margin
```

`seg3d_merge_extreme_safety_margin` defaults to `1.1` (10% above the
loudest observed noise fluctuation) — a pragmatic, round margin, **not** a
calibrated false-alarm rate; tighten it for a stricter gate. Like the
multi-line path, this is independent of the other two paths in both
directions: a calibration failure in one never disables another.

```bash
--seg3d_merge_extreme_safety_margin 1.1
```

Set `--seg3d_merge_extreme_safety_margin None` to disable this path
entirely.

`figs/<headname>_seg3d_compare.pdf` — a comparison figure: the same
white-light image with (1) the 2D SExtractor catalog, (2) the 3D
matched-filter candidates coloured by SNR, (3) the post-veto purity vs
SNR-threshold self-check, and — when `run_seg3d`'s returned
`group_purity_scan` is passed through to the figure builder — (4) the
multi-line group purity vs combined-S/N self-check (see
`apsPlot.source_detection_3d.build_figure`'s `group_purity_scan`
argument). The extreme-S/N path has no dedicated panel of its own (a
single loudest-event number doesn't need a curve); a merged extreme-S/N
candidate is still visible in panel (2) via its aperture outline, same as
any other merged candidate.

---

## 7. Step 2 — Source Classification and Redshift Measurement

Classification is performed inside Stage 1 by Redrock. Understanding the
output and knowing when to intervene is important before proceeding to Stage 2.

As of v1.9, Redrock fits the collapsed patch spectrum's real, flux-weighted
per-spaxel LSF by default (`ifu_class`'s own `spaxel_weighted_lsf=True`
default) rather than an arbitrary single spaxel's own curve — see
[Section 12](#12-configuration-parameter-reference)'s "Spaxel-weighted LSF"
entry for the mechanism this shares with the ExGal/Gal fitting stages.

### Redshift warning flags

Redrock returns a bitmask `ZWARN` for each solution:

| ZWARN | Meaning | Recommended action |
|---|---|---|
| `0` | Clean fit — no issues | Use as-is |
| `4` | Small ΔChi² — top two solutions are close | Inspect the RR plot |
| `8` | Low SNR — fit may be unreliable | Treat with caution |
| `64` | Negative template coefficients | May indicate contamination |
| `256` | Low ΔChi² between galaxy and star templates | Review classification |

Sources with `ZWARN > 0` are still processed by default. The archetype
fine-fitting fallback inside the pipeline will attempt to improve them, but
manual inspection is advisable for important targets.

### Classification priority

When a field contains multiple sources, processing priority follows:

```
GALAXY  >  QSO  >  STAR  >  WD
```

### Inspecting Redrock fits

For any target with a suspicious redshift or `ZWARN > 0`, inspect the
Redrock diagnostic plot saved by Stage 1 in `figs_ExGal/`:

```
figs_ExGal/RR_<TARGID>_<CNAME>_<APS_ID>.png
```

The plot shows the observed spectrum (black), the best-fit Redrock model
(coloured by rank — red for rank 0, orange for rank 1, purple for rank 2),
and the IVAR spectrum below each panel. A good fit will track the absorption
or emission features in the data. Check that IVAR is non-zero across the key
spectral features.

### Inspecting and editing the patch file

Always inspect the patch file before running Stage 2. A quick Python check:

```python
from astropy.table import Table

t = Table.read("/data/L2/20240808/11182/"
               "stackcube_3071431__stackcube_3071430_targets.fits")

for row in t:
    z_val  = row["Z"][0]  if hasattr(row["Z"],  "__len__") else row["Z"]
    zw_val = row["ZWARN"][0] if hasattr(row["ZWARN"], "__len__") else row["ZWARN"]
    cl_val = row["CLASS"][0].strip() if hasattr(row["CLASS"], "__len__") else row["CLASS"]
    print("id=%d  type=%s  CLASS=%s  Z=%.5f  ZWARN=%d" % (
        row["id"], row["type"].strip(), cl_val, z_val, zw_val))
```

To correct misclassifications or wrong redshifts use `aps_ifu_tools`:

```python
from PyAPS.aps_ifu_tools import (
    explore_patch_table, inspect_row,
    set_redshift, set_class, set_type,
    add_mask_region, remove_row, save_patch_table,
)

headname = "stackcube_3071431__stackcube_3071430"
outpath  = "/data/L2/20240808/11182/"

# Load and summarise all rows
data  = explore_patch_table(headname, outpath)
table = data["table"]

# Inspect one row in detail
inspect_row(table, row_id=1)

# Fix a wrong redshift
set_redshift(table, row_id=1, z=0.0923, zerr=0.0001)

# Fix a wrong classification
set_class(table, row_id=2, class_str="GALAXY")

# Mark a contaminating star as a mask region so it is excluded
# from the apertures of neighbouring sources
set_type(table, row_id=3, type_str="M")

# Add a new mask region not found by segmentation
add_mask_region(table, ra=185.201, dec=58.094,
                a_arcsec=5.0, b_arcsec=5.0, angle=0.0)

# Remove a spurious detection entirely
remove_row(table, row_id=4)

# Always save to the _mod file — keeps the original _targets.fits intact
save_patch_table(table,
    outpath + headname + "_targets_mod.fits")
```

Available editing functions summary:

| Function | Description |
|---|---|
| `explore_patch_table(headname, outpath)` | Load and print a summary of all rows |
| `inspect_row(table, row_id)` | Print detailed information for one row |
| `set_redshift(table, row_id, z, zerr)` | Override the redshift for one source |
| `set_class(table, row_id, class_str)` | Override the classification |
| `set_type(table, row_id, type_str)` | Change type: `T`, `C`, or `M` |
| `add_mask_region(table, ra, dec, a, b, angle)` | Add a new mask aperture |
| `remove_row(table, row_id)` | Remove a source entirely |
| `save_patch_table(table, path)` | Write the modified table to disk |

> **Important:** Stage 2 always reads `<headname>_targets_mod.fits`.
> If you make no edits, copy the raw file before running Stage 2:
>
> ```bash
> cp stackcube_..._targets.fits stackcube_..._targets_mod.fits
> ```

---

## 8. Step 3 — The Patch File

The patch file is the central data structure connecting Stage 1 to Stages
2a and 2b. Each row defines one **patch** — a spatial aperture that the
analysis module will process as a self-contained target.

### Row types

| `type` | Processed by | Description |
|---|---|---|
| `T` | ExGal or Gal (depending on CLASS) | Main target |
| `C` | ExGal or Gal | Central WEAVE catalogue object, treated as a target |
| `M` | Neither — used to mask neighbours | Excluded from all surrounding patches |

### Single-source field example

A single isolated galaxy in a LIFU pointing:

```
id  type  CLASS    Z       ZERR    A_world   B_world
 1   T     GALAXY  0.0923  0.0001  0.02822   0.01312
```

This produces one ExGal patch: `<headname>_P0001`.

### Multi-source cluster field example

```
id  type  CLASS    Z       ZERR    A_world   B_world
 1   T     GALAXY  0.0923  0.0001  0.02822   0.01312   →  ExGal P0001
 2   T     GALAXY  0.0941  0.0001  0.01100   0.00880   →  ExGal P0002
 3   T     STAR    0.0000  0.0000  0.00250   0.00250   →  Gal   P0003
 4   M     —       —       —       0.00500   0.00500   →  mask only
```

The ExGal worker processes patches P0001 and P0002. The Gal worker processes
P0003. Row 4 is a mask and is not processed by either.

### Patch naming convention

```
<headname>_P<NNNN>
```

where `NNNN` is the zero-padded `id` from the patch table. All output files
for that source use this as their root name.

### Health check before running Stage 2

```python
from PyAPS.aps_ifu_utils import test_patch_table, load_and_split_patch_file

wp_table, ctarg_excluded = load_and_split_patch_file(
    "/data/L2/20240808/11182/stackcube_..._targets_mod.fits")

# Validation — prints warnings for any detected issues
test_patch_table(wp_table, mode="ExGal")

print("Target rows:", len(wp_table[wp_table["type"] == "T"]))
print("Mask rows:",   len(wp_table[wp_table["type"] == "M"]))
```

---

## 9. Step 4 — Running ExGal Analysis

`aps_ExGal_worker.py` processes all `GALAXY` and `QSO` rows in the patch
file. For each patch it:

1. Extracts all fibres within the patch aperture from the L1 cube
2. Applies an adaptive MIN_SNR cut to remove bad spaxels (dead fibres, artifacts)
3. Optionally applies a surface brightness filter to remove low-flux outer spaxels
4. Runs optional spatial pre-binning (square bins before Voronoi)
5. Runs Voronoi tessellation to reach the target SNR per bin
6. Log-rebins spectra for pPXF
7. Optionally runs pPXF, EMIPPXF, and line strength modules
8. Merges all outputs into a single L2 FITS file

### Minimal run — preparation only

Run this first to check the spatial binning and Voronoi quality before
committing to the full fitting run (which can take hours per field):

```bash
python aps_ExGal_worker.py \
    --infiles /data/L1/20240808/stackcube_3071431.fit \
              /data/L1/20240808/stackcube_3071430.fit \
    --headname stackcube_3071431__stackcube_3071430 \
    --outpath  /data/L2/20240808/11182/ \
    --patch_file /data/L2/20240808/11182/stackcube_3071431__stackcube_3071430_targets_mod.fits \
    --IFU_config_dir  <PYAPS_DIR>/configs/ExGal_configs/ \
    --ExGal_templates <PYAPS_DIR>/PyAPS_templates/templates_ExGal/ \
    --IFU_params      <PYAPS_DIR>/configs/ExGal_configs/LIFULR11.json \
    --PPXF False --EMIPPXF False --LS False \
    --wlranges 3800.0,5950.0 5900.0,9280.0 \
    --arms_ratio 1.0,1.0 \
    --sens_corr True --mask_gaps True --safe_mask_gaps True \
    --tellurics True --join_arms True \
    --caldir /data/CAL --catdir /data/CAT
```

Inspect the diagnostic plots in `figs_ExGal/` before proceeding to the full run.

### Full ExGal run

```bash
python aps_ExGal_worker.py \
    --infiles /data/L1/20240808/stackcube_3071431.fit \
              /data/L1/20240808/stackcube_3071430.fit \
    --headname stackcube_3071431__stackcube_3071430 \
    --outpath  /data/L2/20240808/11182/ \
    --patch_file /data/L2/20240808/11182/stackcube_3071431__stackcube_3071430_targets_mod.fits \
    --IFU_config_dir  <PYAPS_DIR>/configs/ExGal_configs/ \
    --ExGal_templates <PYAPS_DIR>/PyAPS_templates/templates_ExGal/ \
    --IFU_params      <PYAPS_DIR>/configs/ExGal_configs/LIFULR11.json \
    --PPXF True --EMIPPXF True --LS True \
    --mp_ExGal 6 \
    --wlranges 3800.0,5950.0 5900.0,9280.0 \
    --arms_ratio 1.0,1.0 \
    --sens_corr True --mask_gaps True --safe_mask_gaps True \
    --tellurics True --join_arms True \
    --caldir /data/CAL --catdir /data/CAT
```

### Single-target debug mode

Process one specific source by injecting the aperture directly without a
patch file. This is the fastest way to test a particular galaxy:

```bash
python aps_ExGal_worker.py \
    --infiles /data/L1/20240808/stackcube_3071431.fit \
              /data/L1/20240808/stackcube_3071430.fit \
    --headname stackcube_3071431__stackcube_3071430 \
    --outpath  /data/L2/20240808/11182/ \
    --IFU_config_dir  <PYAPS_DIR>/configs/ExGal_configs/ \
    --ExGal_templates <PYAPS_DIR>/PyAPS_templates/templates_ExGal/ \
    --IFU_params      <PYAPS_DIR>/configs/ExGal_configs/LIFULR11.json \
    --patch_array "1,185.198164,58.092634,101.52,47.25,0.0923,0.0001,GALAXY" \
    --PPXF True --EMIPPXF True --LS False \
    --wlranges 3800.0,5950.0 5900.0,9280.0 \
    --arms_ratio 1.0,1.0 \
    --sens_corr True --mask_gaps True --safe_mask_gaps True \
    --tellurics True --join_arms True \
    --caldir /data/CAL --catdir /data/CAT
```

The `--patch_array` format is fixed-order, comma-separated:

```
"id,RA_deg,Dec_deg,A_arcsec,B_arcsec,Z,ZERR,CLASS"
```

> `--patch_array` and `--patch_file` are mutually exclusive.

### Equivalent Python API call

For scripting or notebook use you can call the module directly:

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
    infiles         = ["/data/L1/20240808/stackcube_3071431.fit",
                       "/data/L1/20240808/stackcube_3071430.fit"],
    headname        = "stackcube_3071431__stackcube_3071430",
    outpath         = "/data/L2/20240808/11182/",
    IFU_config_dir  = "<PYAPS_DIR>/configs/ExGal_configs/",
    ExGal_templates = "<PYAPS_DIR>/PyAPS_templates/templates_ExGal/",
    IFU_params      = "<PYAPS_DIR>/configs/ExGal_configs/LIFULR11.json",
    patch_array     = patch,
    PPXF=True, EMIPPXF=True, LS=False,
    nthreads        = 6,
    wlranges        = [[3800.0, 5950.0], [5900.0, 9280.0]],
    arms_ratio      = [1.0, 1.0],
    sens_corr=True, mask_gaps=True, safe_mask_gaps=True,
    tellurics=True, join_arms=True,
    caldir="/data/CAL", catdir="/data/CAT",
)
```

### pPXF wavelength window — how it adapts with redshift

The pipeline automatically determines the rest-frame fitting window. The
config values `LMIN_PPXF` and `LMAX_PPXF` define the preferred low-redshift
optical window. Above a threshold redshift (z ~ 0.55 for the default config)
the window adapts to whatever is available given the observed wavelength range
and template library coverage:

| Redshift | pPXF window behaviour |
|---|---|
| `z < 0.55` | Fixed: `LMIN_PPXF` – `LMAX_PPXF` from config (e.g. 3000–6000 Å rest) |
| `0.55 ≤ z < 1.0` | Adaptive, blue edge clamped near Ca H&K while it is observable |
| `1.0 ≤ z < 2.0` | Adaptive, full intersection of WEAVE coverage and template library |
| `z > 2.0` | Lyα forest floor applied (lmin ≥ 1266 Å rest) |
| `z > 4.5` | Window < 500 Å — pPXF and LS skipped, warning printed |

The window used for each target is always printed:
```
pPXF window:  3000 – 5750 Å rest  (z=0.6000  window=2750 Å)
EMI  window:  2425 – 5750 Å rest  (z=0.6000  window=3325 Å)
```

---

## 10. Step 5 — Running Gal Analysis

`aps_Gal_worker.py` processes all `STAR` rows in the patch file using
FERRE stellar atmosphere models and optionally the RVS radial velocity module.

### Key differences from ExGal

| Aspect | ExGal | Gal |
|---|---|---|
| Processes | `GALAXY`, `QSO` | `STAR` |
| Wavelength scale | Log-rebinned for pPXF | Linear for FERRE / RVS |
| Arms | Configurable join | Always joined |
| Flux / SB filter | Yes — removes low-SB outskirts | No |
| MIN_SNR ceiling | 1.5 (gentle — protects emission-line spaxels) | 3.0 (stellar continuum always present) |
| Spatial bin size | Typically 1.0–2.0 arcsec | Typically 3.0–5.0 arcsec |
| Target SNR | Typically 20–30 | Typically 50–80 |

### Running Stage 2b

```bash
python aps_Gal_worker.py \
    --infiles /data/L1/20240808/stackcube_3071431.fit \
              /data/L1/20240808/stackcube_3071430.fit \
    --headname stackcube_3071431__stackcube_3071430 \
    --outpath  /data/L2/20240808/11182/ \
    --patch_file /data/L2/20240808/11182/stackcube_3071431__stackcube_3071430_targets_mod.fits \
    --IFU_config_dir <PYAPS_DIR>/configs/Gal_configs/ \
    --IFU_params     <PYAPS_DIR>/configs/Gal_configs/LIFULR11_GAL.json \
    --ferre_exe       <FERRE_DIR>/src/ferre.x \
    --ferre_templates <PYAPS_DIR>/PyAPS_templates/templates_FERRE/ \
    --rvs_config      <PYAPS_DIR>/configs/Gal_configs/rvs_config.yaml \
    --wlranges 3800.0,5950.0 5900.0,9280.0 \
    --arms_ratio 1.0,1.0 \
    --sens_corr True --mask_gaps True --safe_mask_gaps True \
    --tellurics True --join_arms True \
    --caldir /data/CAL --catdir /data/CAT
```

### Choosing Gal binning parameters

For stellar kinematics where spatial resolution is less important than
S/N quality per bin, use larger spatial pre-bins and higher Voronoi target:

```json
"SPBIN_SIZE_GAL"   : 3.0,
"TARGET_SNR_GAL"   : 50.0,
"MIN_SNR_GAL"      : null
```

This typically gives 50–100 Voronoi bins from a standard LIFU field.
For very few, high-quality annular-like bins use `SPBIN_SIZE_GAL=5.0`
and `TARGET_SNR_GAL=80.0` to get 10–30 bins.

---

## 11. Running as Part of the Full Pipeline

When running through the automated WEAVE pipeline (KIKO / SLURM), the three
stages are generated as separate bash scripts and submitted as a SLURM job
chain. Stage 1 must complete before Stages 2a and 2b start; 2a and 2b run
in parallel with each other:

```bash
jid_PR    = sbatch IFU_Prepare.sh
jid_ExGal = sbatch --dependency=afterany:$jid_PR  IFU_ExGal.sh
jid_Gal   = sbatch --dependency=afterany:$jid_PR  IFU_Gal.sh
```

The script generator in `aps_runner.py::ifu_scriptGEN` produces all three
scripts automatically. You do not need to write them by hand. The generated
scripts are saved under `scripts/` and logs under `logs/` within the output
directory.

To generate and optionally submit scripts for a specific observation:

```python
from PyAPS.aps_runner import ifu_scriptGEN

ifu_scriptGEN(
    infiles  = ["/data/L1/20240808/stackcube_3071431.fit",
                "/data/L1/20240808/stackcube_3071430.fit"],
    headname = "stackcube_3071431__stackcube_3071430",
    outpath  = "/data/L2/20240808/11182/",
    conf     = pipeline_conf,    # loaded from pipeline_params.yaml
    mp       = [1, 6, 4],        # [prepare_threads, ExGal_threads, Gal_threads]
    submit   = True,             # submit to SLURM immediately
)
```

---

## 12. Configuration Parameter Reference

### Spatial binning (ExGal)

| Key | Type | Default | Description |
|---|---|---|---|
| `SPBIN_SIZE_EXGAL` | float | `1.0` | Spatial bin side length in arcsec. Set to `-1` to skip spatial pre-binning. |
| `SB_FILTER` | int | `1` | Surface brightness filter: `1` = enabled, `0` = disabled. |
| `SB_FILTER_MODE` | str | `percentile` | Filter method: `percentile`, `transition`, `safe`, or `none`. |
| `SB_FILTER_PERCENT` | float | `10` | For `percentile` mode: remove the lowest N% of spaxels by flux. |
| `SB_FILTER_MIN_FRAC` | float | `0.01` | Safety floor: always keep at least this fraction of spaxels. |
| `SB_FILTER_SNR_MIN` | float | `3.0` | For `transition` / `safe` mode: target SNR for the transition zone. |
| `SB_FILTER_DELTA` | float | `2.0` | Width of the SNR transition zone. |
| `SB_FILTER_MIN_SPAXELS` | int | `10` | Minimum spaxels required in transition zone before falling back. |

### Spatial binning (Gal)

| Key | Type | Default | Description |
|---|---|---|---|
| `SPBIN_SIZE_GAL` | float | `3.0` | Spatial bin side length in arcsec for Gal mode. |
| `VORONOI_GAL` | int | `1` | Voronoi tessellation: `1` = enabled, `0` = treat each spatial bin as its own bin. |
| `TARGET_SNR_GAL` | float | `50.0` | Target SNR per Voronoi bin for Gal mode. |
| `MIN_SNR_GAL` | float/null | `null` | Minimum SNR per spaxel for Gal mode. `null` = adaptive. |

### Voronoi tessellation

| Key | Type | Default | Description |
|---|---|---|---|
| `VORONOI` | int | `1` | `1` = Voronoi tessellation, `0` = treat each spatial bin as its own Voronoi bin. |
| `TARGET_SNR` | float/str | `"None"` | Target SNR per bin. `"None"` = auto-derive from the 90th percentile of the spatial-bin SNR distribution, rounded to the nearest value in [20, 30, 40]. |
| `COVAR_VOR` | float | `0.0` | Spatial noise covariance correction (Garcia-Benito et al. 2015). `0.0` disables. |

### Adaptive MIN_SNR

| Key | Type | Default | Description |
|---|---|---|---|
| `MIN_SNR` | float/null | `null` | Minimum SNR threshold. `null` = adaptive from data. Float = fixed override. |
| `MIN_SNR_METHOD` | str | `percentile` | Method: `percentile`, `sigma_clip`, or `snr_gap`. |
| `MIN_SNR_PERCENTILE` | float | `2` | For `percentile` method: bottom N% to cut. Keep low (1–5) to protect faint targets. |
| `MIN_SNR_FLOOR` | float | `0.1` | Hard minimum: threshold never set below this. Protects emission-line-only targets. |
| `MIN_SNR_CEILING` | float | `1.5` | Hard maximum: threshold never set above this (ExGal). Prevents aggressive cutting of faint galaxies. |

### Spaxel-weighted LSF

| Key | Type | Default | Description |
|---|---|---|---|
| `SPAXEL_WEIGHTED_LSF` | int | `1` | Per-bin resolution instead of one flat global LSF/FWHM curve for the whole patch — **default on as of v1.9**. `0` falls back to pre-v1.9 behaviour. Read by both `ifu_ExGal_prepare` and `ifu_Gal_prepare`. See `aps_ifu_ExGal.md`'s [Section 11](aps_ifu_ExGal.md#11-spaxel-weighted-lsf-and-resolution-buckets) for the full mechanism ("buckets" and all). |
| `LSF_N_BUCKETS` | int | `6` | Cap on how many distinct resolution "buckets" template preparation computes per patch, regardless of bin count. |

Classification (`ifu_class`, Section 7 above) is separate: it isn't driven by this JSON config, but its own `spaxel_weighted_lsf` Python parameter also **defaults to `True` as of v1.9** — the collapsed redshift-fit spectrum gets a real flux-weighted patch-average LSF instead of silently inheriting whichever spaxel happened to be first in the collapsed list.

### pPXF wavelength window

| Key | Type | Default | Description |
|---|---|---|---|
| `LMIN_PPXF` | float | `3000.0` | Preferred rest-frame blue limit for pPXF at low-z (Å). |
| `LMAX_PPXF` | float | `6000.0` | Preferred rest-frame red limit for pPXF at low-z (Å). |
| `LMIN_EMI` | float | `2000.0` | Preferred rest-frame blue limit for EMIPPXF at low-z (Å). |
| `LMAX_EMI` | float | `8600.0` | Preferred rest-frame red limit for EMIPPXF at low-z (Å). |
| `PPXF_LIB_LMIN` | float | `1200.0` | Hard blue limit of your stellar template library (Å rest). |
| `PPXF_LIB_LMAX` | float | `10000.0` | Hard red limit of your stellar template library (Å rest). |

### Full example config (LIFULR11.json, key parameters)

```json
{
    "PIXELSIZE"              : 0.0,
    "VORONOI"                : 1,
    "TARGET_SNR"             : "None",
    "COVAR_VOR"              : 0.0,
    "SPAXEL_WEIGHTED_LSF"    : 1,
    "LSF_N_BUCKETS"          : 6,
    "MIN_SNR"                : null,
    "MIN_SNR_METHOD"         : "percentile",
    "MIN_SNR_PERCENTILE"     : 2,
    "MIN_SNR_FLOOR"          : 0.1,
    "MIN_SNR_CEILING"        : 1.5,
    "SPBIN_SIZE_EXGAL"       : 1.0,
    "SB_FILTER"              : 1,
    "SB_FILTER_MODE"         : "percentile",
    "SB_FILTER_PERCENT"      : 10,
    "SB_FILTER_MIN_FRAC"     : 0.01,
    "SB_FILTER_SNR_MIN"      : 3.0,
    "SB_FILTER_DELTA"        : 2.0,
    "SB_FILTER_MIN_SPAXELS"  : 10,
    "LMIN_PPXF"              : 3000.0,
    "LMAX_PPXF"              : 6000.0,
    "LMIN_EMI"               : 2000.0,
    "LMAX_EMI"               : 8600.0,
    "PPXF_LIB_LMIN"          : 1200.0,
    "PPXF_LIB_LMAX"          : 10000.0,
    "SSP_LIB"                : "MILES",
    "VELSCALE"               : "None",
    "LS_MODE"                : "default",
    "LS_RES"                 : "ADAPTED"
}
```

---

## 13. Diagnostic Plots

After each preparation run the pipeline writes a set of diagnostic figures
to `figs_ExGal/` (or `figs_Gal/`). Always inspect these before running the
fitting modules.

### Figure guide

| File | Panels | What to check |
|---|---|---|
| `*_prep_raw.png` | SNR map · Signal map · Noise map | Is there real signal? Any obvious artefacts or dead regions? |
| `*_prep_snrcut.png` | MIN_SNR keep/remove map · SNR histogram | Are removed spaxels genuinely bad (clustered at SNR ≈ 0)? Is the threshold sensible? |
| `*_prep_spatialbin.png` | Spatial-bin SNR (all bins) · Flux filter on raw spaxels · Signal map of surviving bins · Dual SNR histogram | Did the flux filter remove the right spaxels? Are the bins well-distributed? |
| `*_snr_stages.png` | 5 panels: Raw SNR → MIN_SNR cut → Spatial-bin SNR → SB filter on bins → Voronoi SNR | Follow the SNR evolution through every preparation step in one figure |
| `*_prep_voronoi.png` | Voronoi bin IDs · SNR per bin · N spaxels per bin | Are the bins reasonable in size and SNR? Is the target SNR being met? |
| `*_sbin_<size>.png` | Raw spaxels vs spatially binned side-by-side | Does the spatial binning look geometrically sensible for this source? |
| `*_voronoi_cube.png` | 3-panel Voronoi structure overview | Overall view: spatial bins coloured by data, Voronoi boundaries, and combined overlay |

### Colour conventions

- Binary keep/remove maps: **steel blue** = kept, **terracotta** = removed
- SNR maps: `RdYlBu_r` — red = low SNR, blue = high SNR
- Voronoi bin maps: random tab20 colour assignment so neighbouring bins contrast

### Parameter exploration without full fitting

To quickly compare different spatial bin sizes or Voronoi target SNR values:

```bash
python aps_ifu_exgal_diag.py \
    --infiles /data/L1/20240808/stackcube_3071431.fit \
              /data/L1/20240808/stackcube_3071430.fit \
    --headname stackcube_3071431__stackcube_3071430 \
    --outpath  /data/L2/20240808/11182/ \
    --IFU_config_dir <PYAPS_DIR>/configs/ExGal_configs/ \
    --IFU_params     <PYAPS_DIR>/configs/ExGal_configs/LIFULR11.json \
    --patch_file /data/L2/20240808/11182/stackcube_..._targets_mod.fits \
    --scan_spbin 0.5,1.0,1.5,2.0 \
    --scan_target_snr 20,30,40 \
    --figdir_suffix diag
```

Figures are saved to `figs_ExGal_diag/` with the scanned parameter value in
the filename for side-by-side comparison.

---

## 14. Output Files

### Per-patch outputs (ExGal)

| File | Description |
|---|---|
| `<headname>_P####_table.fits` | Voronoi bin table: coordinates, bin IDs, per-bin SNR |
| `<headname>_P####_BINSpectra.fits` | Log-rebinned co-added spectra, one per Voronoi bin |
| `<headname>_P####_ppxf.fits` | pPXF stellar kinematics: V, sigma, h3, h4 per bin |
| `<headname>_P####_emcee.fits` | EMIPPXF emission line fluxes, widths, EWs per bin |
| `<headname>_P####_ls.fits` | Line strength index measurements per bin |
| `<headname>_P####_APS.fits` | Merged L2 product — all results in one FITS file |

### Per-patch outputs (Gal)

| File | Description |
|---|---|
| `<headname>_P####_table.fits` | Voronoi bin table |
| `<headname>_P####_BINSpectra_linear.fits` | Linear spectra per Voronoi bin |
| `<headname>_P####_FERRE.fits` | FERRE stellar parameters: Teff, log g, [Fe/H], [alpha/Fe] |
| `<headname>_P####_RVS.fits` | Radial velocities from RVS |
| `<headname>_P####_APS.fits` | Merged L2 product |

---

## 15. Troubleshooting

### "No valid APS IDs found for this ExGal patch"

No fibres fall within the patch aperture after filtering.

- Check that `A_world` and `B_world` in the patch file are sensible (near
  zero means a failed aperture fit during segmentation).
- Increase `seg2d_radii_factor` to enlarge apertures (try `10.0` or `12.0`).
- Check that `--targsrvy` and `--targclass` are not accidentally filtering
  out all fibres in the field.
- Confirm the RA/Dec in the patch file actually falls within the IFU footprint.

### "No valid spaxels remaining after filters"

All spaxels were removed by the flux filter or MIN_SNR cut.

1. Check `*_prep_raw.png` — is there any real signal in the raw SNR map?
2. Check `*_prep_snrcut.png` — is the MIN_SNR adaptive threshold too high?
   Try setting `"MIN_SNR": 0.0` to bypass the cut and rerun.
3. Try `"SB_FILTER": 0` to disable the flux filter and rerun.
4. If real signal exists but is filtered, try increasing `SB_FILTER_MIN_FRAC`
   to `0.05` to prevent over-aggressive filtering.

### Voronoi fails or produces only one bin

- `TARGET_SNR` is higher than the data can support. Set to `"None"` for
  automatic derivation, or reduce manually to `10.0` as a test.
- Too few spaxels survive the SB filter. Check how many bins appear in
  `*_prep_spatialbin.png` panel 1.
- Try reducing `SPBIN_SIZE_EXGAL` to produce more input points for Voronoi.

### pPXF window warning

```
WARNING: z=4.600 — rest-frame pPXF window < 500 Å. PPXF and LS will be skipped.
```

Expected at z > 4.5. EMIPPXF still runs if the EMI window (minimum 200 Å)
is sufficient. At intermediate redshift check the printed window size — below
~1000 Å rest-frame at z > 1 the kinematic measurements will have high
uncertainties.

### Patch file has wrong redshifts or classifications

Use `aps_ifu_tools` (see [Step 3](#8-step-3--the-patch-file)) to correct
`_targets_mod.fits` and rerun Stage 2 only. Stage 1 does not need to be rerun.

### Checking which MIN_SNR was applied

The adaptive MIN_SNR value is always printed during Stage 2:

```
MIN_SNR adaptive: 0.3142  (z=0.0923  N_spaxels=5494  SNR p2=0.314  p50=4.821  p98=31.4)
```

If this looks too high or too low for your target type, override it in the
config with an explicit float value (`"MIN_SNR": 0.5`) and rerun.

### RR plot inspection for individual targets

To regenerate a Redrock diagnostic plot for any target:

```python
from PyAPS.aps_ifu_prepare import make_rrplot
from astropy.table import Table

zbest = Table.read(".../stackcube_..._P0001_rr_zbest.fits")
zspec = Table.read(".../stackcube_..._P0001_rr_zspec.fits")

make_rrplot(
    zbest          = zbest,
    zspec          = zspec,
    setups         = [["BLU"], ["RED"]],
    figdir         = "/data/L2/20240808/11182/figs_ExGal/",
    collapse_fname = "P0001_debug",
)
```

The plot shows data (black), best-fit model (coloured by rank), and IVAR
below each panel. Verify that the model tracks the spectral features, that
IVAR is non-zero at key absorption or emission lines, and that the inter-arm
gap is properly masked.

---

*For further information, bug reports, or feature requests open an
issue on the PyAPS repository.*
