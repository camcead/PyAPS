# APSOB — The WEAVE Spectral Data Reader

## A Reference and Tutorial Guide for the PyAPS Pipeline

**Module:** `PyAPS.aps_utils.APSOB`
**PyAPS version:** 1.7+
**Applies to:** All WEAVE observing modes — MOS, LIFU, MIFU, MOSLIFU, MOSMIFU
**Maintained by:** CASU, Institute of Astronomy, University of Cambridge

---

## Table of Contents

1. [What APSOB Does](#1-what-apsob-does)
2. [Why APSOB is the Heart of the Pipeline](#2-why-apsob-is-the-heart-of-the-pipeline)
3. [Supported WEAVE Observing Modes](#3-supported-weave-observing-modes)
4. [Processing Chain — Step by Step](#4-processing-chain--step-by-step)
5. [Input Parameters — Complete Reference](#5-input-parameters--complete-reference)
6. [Data Model — What APSOB Produces](#6-data-model--what-apsob-produces)
7. [Methods and Properties](#7-methods-and-properties)
8. [Standalone Usage — Reading WEAVE Files](#8-standalone-usage--reading-weave-files)
9. [Advanced Usage](#9-advanced-usage)
10. [IVAR Normalisation — Theory and Practice](#10-ivar-normalisation--theory-and-practice)
11. [Resolution Matrices and LSF](#11-resolution-matrices-and-lsf)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. What APSOB Does

`APSOB` (**APS Object**) is the universal WEAVE spectral data reader and preprocessor. It takes raw L1 FITS files directly from the WEAVE CPS pipeline and transforms them into clean, analysis-ready spectral data objects — handling every correction, masking, and calibration step along the way.

Every downstream PyAPS module — Redrock classification, pPXF stellar kinematics, EMIPPXF emission line fitting, FERRE stellar parameters, RVS radial velocities — receives its data from APSOB. You will find the same set of parameters throughout the pipeline because they are all passed through to APSOB, which applies them consistently.

In a single call, APSOB:

- Reads one or two L1 FITS files (blue and red spectrograph arms)
- Detects the observing mode, resolution, camera, and binning automatically
- Selects the fibres or spaxels of interest
- Applies wavelength range clipping
- Corrects for sensitivity / flux calibration
- Masks gaps, bad pixels, telluric absorption regions
- Removes cosmic rays (optional)
- Applies arms-ratio flux scaling between blue and red arms
- Stitches blue and red arms into a single joined spectrum
- Normalises the inverse variance (IVAR) across arms by information content
- Computes per-fibre/spaxel SNR
- Loads and builds wavelength-dependent LSF / FWHM interpolation functions
- Computes resolution matrices ready for Redrock
- Optionally collapses all spaxels into a single co-added spectrum

---

## 2. Why APSOB is the Heart of the Pipeline

The PyAPS pipeline is built around a strict separation between data reading and scientific analysis. APSOB is the sole interface between the raw CPS L1 data and every analysis module. This means:

- **Consistency**: the same corrections are applied identically regardless of whether the data goes to Redrock, pPXF, FERRE, or RVS
- **Single point of control**: change a masking or correction setting in one place and it propagates everywhere
- **No code duplication**: analysis modules never touch raw FITS files directly

The parameters you pass to `aps_ifu_prepare`, `aps_ExGal_worker`, `aps_Gal_worker`, `aps_ifu_ExGal`, `aps_ifu_Gal`, and `aps_runner` are all the same parameters — because they are all forwarded to APSOB. The docstring in each module says "see APSOB documentation" for a reason.

---

## 3. Supported WEAVE Observing Modes

APSOB auto-detects the observing mode from the L1 file header (`OBSMODE` keyword) and sets up all internal processing accordingly.

| Mode string | `OBSMODE` header | NAXIS | Description |
|---|---|---|---|
| `MOS` | `MOS` | 2 | Multi-Object Spectrograph — individual fibre spectra |
| `LIFU` | `LIFU` | 3 | Large IFU — 3D spectral cube |
| `MIFU` | `MIFU` | 3 | Mini IFU — 3D spectral cube |
| `MOSLIFU` | `LIFU` | 2 | LIFU data formatted as MOS (legacy) |
| `MOSMIFU` | `MIFU` | 2 | MIFU data formatted as MOS (legacy) |

The mode determines how spaxel/fibre coordinates are extracted, how wavelength arrays are built, how the data cube is reshaped, and which header keywords carry the relevant metadata.

**MOS mode** reads a 2D FITS array `[n_fibres, n_wavelengths]` with fibre metadata from the `FIBTABLE` extension.

**IFU modes (LIFU/MIFU)** read a 3D data cube `[n_x, n_y, n_wavelengths]` which is reshaped internally into `[n_x × n_y, n_wavelengths]` — the IFU spatial pixels (spaxels) are treated as if they were individual fibres for all downstream processing.

---

## 4. Processing Chain — Step by Step

This section describes exactly what APSOB does to the data, in order. Understanding this chain is essential for interpreting the output and diagnosing unexpected results.

### Step 0 — File validation and mode detection

`l1_fileinfo()` reads the primary header of each input file and extracts:

- `OBSMODE` → working mode (MOS / LIFU / MIFU / ...)
- `MODE` → resolution mode (`LOWRES` → `LR`, `HIGHRES` → `HR`)
- `CAMERA` → arm (`BLUE`, `GREEN`, `RED`)
- `CCDXBIN`, `CCDYBIN` → spatial binning
- `CD1_1` or `CD3_3` → wavelength step (cross-checked against known values)
- `OBID`, `DATE-OBS`, `MJD-OBS`, `CAT-NAME`, `CALDATE`, `FPMODE`

Files are automatically sorted into blue-first order `[B, G, R]`. An assertion checks that no two input files have the same camera ID. If the wavelength step from the header differs by more than 10% from the expected value, the pipeline exits with an error.

### Step 1 — Target list generation

`gen_targlist()` determines which fibres or spaxels to process, applying all filters simultaneously:

- `aps_ids` — restrict to a specific list of IDs
- `targsrvy` — restrict to specific survey codes (e.g. `['WA', 'GA']`)
- `targclass` — restrict to specific target classes (e.g. `['GALAXY']`)
- `mask_aps_ids` — exclude specific IDs
- `area` — restrict to spaxels/fibres within an elliptical aperture on sky
- `mask_areas` — exclude spaxels within one or more elliptical apertures

The aperture matching for IFU modes uses `astropy.regions.EllipseSkyRegion` with the WCS from the data cube header to correctly convert RA/Dec apertures to spaxel indices. For MOS modes a circular approximation is used.

The output is a sorted list of APS_IDs and two reverse-mapping dictionaries (`idfx`: aps_id → index, `idxf`: index → aps_id) that are used throughout the pipeline to look up targets efficiently.

### Step 2 — Raw data loading

For each input file, APSOB reads:

```
Extension 1: flux array (sky-subtracted, default)
Extension 2: inverse variance array
Extension 3: flux (no sky subtraction) — used if skysub=False
Extension 4: inverse variance (no sky subtraction)
Extension 5: sensitivity function
```

For IFU modes the 3D cube `[n_x, n_y, n_wave]` is reshaped to `[n_x × n_y, n_wave]` so that all subsequent processing is identical to MOS mode.

The sensitivity function is loaded from extension 5. If that fails (some older or solar files have only 4 extensions), APSOB falls back to loading an external sensitivity file from the calibration directory:

```
CAL/<CALDATE>/sens_<camera>_<res><bin>_<fpmode>.fit
```

### Step 3 — Wavelength array construction and clipping

The wavelength array is constructed from the header keywords:

```
MOS:  la = CRVAL1 + arange(NAXIS1) × CD1_1
IFU:  la = CRVAL3 + arange(NAXIS3) × CD3_3
```

If `wlranges` is specified, the data is clipped to `[lmin, lmax]` for each arm. This is how the inter-arm gap is avoided and how users can restrict analysis to specific spectral regions. The clipping uses `argmin(|la - lmin|)` and `argmin(|la - lmax|)` so it always lands on the nearest actual pixel.

### Step 4 — NaN and invalid value handling

All `NaN` values in the flux array are identified and their corresponding IVAR entries are set to zero. Negative IVAR values are also set to zero. This ensures that downstream code never divides by zero or propagates NaN through mathematical operations.

### Step 5 — Gap masking

The inter-arm gap (and any CCD gaps) are masked by setting IVAR to zero wherever the pipeline has already zeroed the flux. The `offset_gap_pix` parameter (from `aps_constants.py`, default typically 10 pixels) adds a safety buffer of additional masked pixels on either side of each gap. This prevents contamination from the noisy edges of the gap region from leaking into the analysis.

### Step 6 — Safe gap masking

`safe_mask_gaps=True` applies a second, more conservative masking pass using a lookup table of known problematic wavelength bands for each instrument configuration. These bands are stored in `aps_constants.gap_bands` keyed by `mode + res_mode + "_" + camera` (e.g. `LIFUHR_BLUE`). This catches instrument-specific artefacts that are not always flagged by the standard gap detection.

### Step 7 — Sky residual masking in the OH forest (optional)

Above 8200 Å the OH sky emission forest is dense enough that sky residuals can mimic stellar absorption features, causing M-star misclassification at high redshift. When `skysub_mask_residuals=True`, APSOB applies a two-scale valley detection algorithm to find and mask IVAR valleys caused by sky line residuals:

- **Narrow window** (~50 Å): catches sharp OH emission line dips where IVAR drops to near zero
- **Wide window** (~300 Å): catches broader fringing and banding features where individual OH line dips merge
- **Background** = max(narrow, wide): ensures the highest available reference level is used at both scales
- **Threshold**: mask wherever IVAR < 0.5 × local background
- **Wing dilation**: 5-pixel expansion to catch line wings where flux is contaminated but IVAR has not fully dropped

**Critical design choice**: this masking is applied **only above 8200 Å**. Below this wavelength sky subtraction is reliable and no masking is applied, preserving the full clean blue spectrum for stellar kinematics.

### Step 8 — Telluric masking

If `tellurics=True`, wavelength bands known to be strongly affected by atmospheric absorption are set to IVAR = 0. The band definitions are stored in `aps_constants.tellurics`. These are the O₂ A-band (~7600 Å), O₂ B-band (~6900 Å), and water vapour bands in the red.

### Step 9 — Gap filling (optional)

If `fill_gap=True`, masked pixels (IVAR = 0) in the flux array are replaced by linear interpolation from the nearest valid pixels on either side. This is designed specifically for RVSpecfit, which requires a contiguous wavelength array without gaps. For all other analysis modules gap filling should be left disabled — masked IVAR is the correct way to communicate bad data.

### Step 10 — Cosmic ray removal (optional)

If `crr=True`, a median filter of width `aps_constants.cr_med_length` is applied to the flux. Pixels where the residual between the original and median-filtered flux exceeds `cr_cut_level × local_sigma` are flagged as cosmic rays by setting their IVAR to zero. A wing of `cr_wing_length` pixels is also masked on either side of each detected cosmic ray to catch the trailing edge of the impact.

### Step 11 — Sensitivity correction

If `sens_corr=True` (the default), the flux is multiplied by the sensitivity function and the IVAR is divided by its square:

```python
flux = flux × sens
ivar = ivar / sens²
```

This converts the data from counts/pixel/second to physical flux units (scaled by `funit`, typically 10⁻¹⁸ erg/s/cm²/Å). The sensitivity function encodes the combined effect of fibre throughput, spectrograph efficiency, and atmospheric transmission.

### Step 12 — IVAR normalisation across arms

When two arms are being joined, they typically have very different IVAR scales because the sensitivity, throughput, and number of pixels differ between the blue and red arms. Without normalisation, the arm with higher IVAR dominates the chi-square in Redrock fitting, effectively down-weighting the other arm.

APSOB normalises IVAR across arms using one of three modes, controlled by `ivar_normalization_mode`:

**`balanced`** (default): equalises the total chi-square contribution of each arm. Computes `sum(IVAR)` for each arm and scales so that both contribute equally. For a two-arm observation this gives each arm exactly 50% of the total chi-square weight regardless of how many pixels it has.

**`pixels`**: weights by the number of valid pixels. An arm with twice as many pixels contributes twice as much chi-square. This preserves the per-pixel information content ratio.

**`hybrid`**: geometric mean of `balanced` and `pixels` weights. A compromise between the two approaches — each arm's contribution scales with its wavelength coverage but the dynamic range is reduced compared to pure pixel weighting.

The normalisation factors are stored in the target metadata (`IVAR_WEIGHT`, `IVAR_NORMALIZED`) so they can be checked and reproduced.

Set `normalize_ivar=False` to disable normalisation entirely and use the legacy `arms_ratio` correction instead.

### Step 13 — Arms ratio correction (legacy)

If `normalize_ivar=False`, the old `arms_ratio` correction is applied: the flux and IVAR of each arm are scaled by the corresponding element of `arms_ratio`. For example, `arms_ratio=[1.0, 0.83]` would scale the red arm flux by 0.83. This was the original approach before IVAR normalisation was introduced and is kept for backward compatibility.

### Step 14 — SNR calculation

Broadband mean SNR is computed per fibre/spaxel:

```python
snr = mean(flux × sqrt(ivar))   over all unmasked pixels
```

Pixels where `ivar <= 10 × ivar_mask_value` (the masking floor) are excluded. The SNR is stored in the target metadata (`tmeta["SNR"]`) and is used by downstream modules for binning and quality control.

### Step 15 — Wavelength conversion to vacuum (optional)

If `vacuum=True`, the air wavelength array is converted to vacuum wavelengths using the Ciddor (1996) formula (implemented in `a2v()`). This conversion is applied to the wavelength array only; flux and IVAR are unchanged. The conversion is valid above 2000 Å and disabled below that threshold.

### Step 16 — LSF / FWHM interpolation

APSOB loads Line Spread Function (LSF) or FWHM profiles from calibration files and builds wavelength-dependent interpolation functions. There are two modes, controlled by `lsftype`:

**`lsftype='LSF'`** (default): reads pre-computed LSF FITS files (`lsf_<setup>_<fpmode>.fits`) from the calibration directory. These files contain per-fibre FWHM profiles measured from calibration lamp spectra. The `run_lsf_analysis()` function fits splines through the calibration data, applies optional smoothing, and returns a dictionary of interpolation functions — one per fibre plus a global average.

**`lsftype='FWHM'`**: reads wavelength calibration files (`wave_<setup>.fits`) and derives FWHM profiles from the arc line measurements directly. This mode uses `run_fwhm_analysis()` which applies GMM bimodal filtering, residual outlier rejection, and cubic spline fitting.

Both modes save the resulting interpolators as pickle files in `configs/lsf/<CALDATE>/` so subsequent runs can load them instantly without re-fitting.

The interpolation functions are stored in the target metadata under the keys `fwhm` (fibre-specific) and `gfwhm` (global average). They are accessed via the `get_fwhm()` method.

### Step 17 — Arm joining

If `join_arms=True` and two input files are provided, the blue and red spectra are stitched together into a single continuous spectrum. The joining logic:

1. Constructs a combined wavelength grid using `wave_interpol()`, which finds the overlap region and creates a merged grid with a step equal to the mean of both arm steps
2. In the overlap region, computes an inverse-variance-weighted mean of the blue and red fluxes
3. Checks for a bad overlap (where the median flux in the overlap region differs by more than `aps_constants.bad_overlap_epsilon` between arms — indicating a flux calibration jump). If detected, the overlap region is zeroed in IVAR
4. Concatenates the non-overlapping blue section + overlap + non-overlapping red section
5. Updates `fwhm` metadata using the joined-arms interpolator

After joining, `self._setups` is updated from `['BLUELR11', 'REDLR11']` to `['Combined']`.

### Step 18 — Spectrum collapse (optional)

If `collapse=True`, all selected spaxels/fibres are co-added into a single spectrum using an inverse-variance-weighted mean:

```python
flux_collapsed = sum(flux × ivar) / sum(ivar)   (per wavelength pixel)
ivar_collapsed = sum(ivar)
```

This is used when a single high-SNR integrated spectrum of the entire IFU field is needed (e.g. for initial Redrock classification before per-spaxel analysis). The collapsed target is assigned `APS_ID = -999` and `CNAME = TARGID = "collapsed"`.

---

## 5. Input Parameters — Complete Reference

These parameters are used by every pipeline module that calls APSOB. Understanding them here is the key to understanding the pipeline as a whole.

### Required

| Parameter | Type | Description |
|---|---|---|
| `infiles` | list of str | L1 FITS files. One for single-arm, two for blue+red joined analysis. Order does not matter — APSOB sorts by camera automatically. |

### Target selection

| Parameter | Type | Default | Description |
|---|---|---|---|
| `aps_ids` | list of int or None | `None` | Restrict to these APS IDs (fibre IDs in MOS, spaxel IDs in IFU). `None` = use all available. |
| `targsrvy` | list of str or None | `None` | Restrict to targets belonging to these survey codes (e.g. `['WA', 'GA']`). `None` = all surveys. |
| `targclass` | list of str or None | `None` | Restrict to targets with these class strings (e.g. `['GALAXY', 'QSO']`). `None` = all classes. |
| `mask_aps_ids` | list of int or None | `None` | Exclude these APS IDs from all processing. Useful for masking bright stars or artefacts. |
| `area` | list or None | `None` | Elliptical aperture `[RA_deg, Dec_deg, A_arcsec, B_arcsec, angle_deg]`. Only fibres/spaxels within this ellipse are selected. For MOS mode a circular approximation is used. |
| `mask_areas` | list of lists or None | `None` | List of elliptical apertures to exclude. Same format as `area`. |

### Spectral processing

| Parameter | Type | Default | Description |
|---|---|---|---|
| `wlranges` | list of [float, float] or None | `None` | Wavelength ranges per arm as `[lmin, lmax]` pairs. Must have the same number of elements as `infiles`. `None` = use the full available range from the file header. |
| `sens_corr` | bool | `True` | Apply flux sensitivity correction. Almost always `True`. Set `False` only for diagnostic purposes. |
| `mask_gaps` | bool | `True` | Mask inter-arm gaps and CCD gaps by zeroing IVAR. **Always use `True`.** |
| `safe_mask_gaps` | bool | `True` | Apply additional safe-masking for known instrument-specific bad wavelength bands (from `aps_constants.gap_bands`). |
| `vacuum` | bool | `False` | Convert output wavelengths from air to vacuum. |
| `tellurics` | bool | `True` | Mask telluric absorption bands (O₂, H₂O) by zeroing IVAR. |
| `fill_gap` | bool | `False` | Interpolate flux across masked gap regions. Only for RVSpecfit. Leave `False` for all other modules. |
| `crr` | bool | `False` | Cosmic ray removal using median filter. |
| `skysub` | bool | `True` | Use sky-subtracted spectrum (extensions 1, 2). `False` = use non-sky-subtracted (extensions 3, 4). |
| `skysub_mask_residuals` | bool | `False` | Apply two-scale OH forest sky residual masking above 8200 Å. |

### Arm handling

| Parameter | Type | Default | Description |
|---|---|---|---|
| `arms_ratio` | list of float or None | `None` | Flux scale factors per arm for the legacy normalisation approach. Example: `[1.0, 0.83]` scales the red arm by 0.83. Ignored when `normalize_ivar=True`. |
| `join_arms` | bool | `False` | Stitch blue and red arms into a single spectrum. Requires two input files. IFU mode always forces `True`. |
| `split_arms` | bool | `False` | After joining, re-split at the mean overlap wavelength. Used by some specific analysis modes. |
| `mask_bad_overlap` | bool | `True` | If the overlap flux levels differ significantly between arms (flux calibration jump), zero the IVAR in the overlap region. |
| `normalize_ivar` | bool | `True` | Normalise IVAR across arms by information content. Replaces the legacy `arms_ratio` correction. |
| `ivar_normalization_mode` | str | `'balanced'` | Normalisation strategy: `'balanced'` (equal chi-square per arm), `'pixels'` (proportional to wavelength coverage), `'hybrid'` (geometric mean of both). |

### LSF / FWHM

| Parameter | Type | Default | Description |
|---|---|---|---|
| `lsftype` | str | `'LSF'` | Source for LSF profiles: `'LSF'` uses pre-computed LSF FITS files, `'FWHM'` derives profiles from wavelength calibration files. |
| `use_resolution_deconvolution` | bool | `True` | Apply template resolution deconvolution to the resolution matrix. Corrects for the finite resolution of the template spectra. |
| `template_sigma0_angstrom` | float | `0.5` | Gaussian sigma (Å) of the template library instrumental resolution, used for deconvolution. |
| `edge_pixels_to_mask` | int | `5` | Number of edge pixels to zero in IVAR after resolution deconvolution, where the resolution matrix is corrupted. |

### Collapse and co-addition

| Parameter | Type | Default | Description |
|---|---|---|---|
| `collapse` | bool | `False` | Co-add all selected spaxels/fibres into a single integrated spectrum. Output target has `APS_ID = -999`. |

### Directories

| Parameter | Type | Default | Description |
|---|---|---|---|
| `catdir` | str or None | `None` | Path to the WEAVE catalogue directory. Used for coordinate and survey metadata. |
| `caldir` | str or None | `None` | Path to the calibration directory containing sensitivity, wavelength, and LSF files. Required for sensitivity correction and LSF loading. |
| `configdir` | str or None | `None` | Path to the pipeline config directory. Used to locate LSF pickle files. Auto-detected from the PyAPS installation if not provided. |
| `debugdir` | str or None | `None` | Output directory for diagnostic files. Unused in production. |

### Technical

| Parameter | Type | Default | Description |
|---|---|---|---|
| `funit` | float | from `aps_constants` | Flux unit scale factor. Multiplied into the flux after sensitivity correction. Typically `1e18` so that fluxes are in units of 10⁻¹⁸ erg/s/cm²/Å. |
| `offset_gap_pix` | int | from `aps_constants` | Number of additional pixels to mask on each side of a detected gap. |
| `aps_id_sum` | int | `0` | Offset added to all APS IDs. Used for multi-OB stacking where fibre IDs must not collide. |
| `rereplace_binned_cal` | bool | `True` | If a binned calibration file is not found, attempt to replace it with the unbinned version. |

---

## 6. Data Model — What APSOB Produces

After construction, APSOB holds a list of `APSTARG` objects. Each target represents one fibre (MOS) or one spaxel (IFU). Each target contains one or more `APSSPEC` objects (one per arm, or one if arms have been joined).

### The target list

```python
targs = APSOB(infiles, ...)

# Number of targets
print(len(targs.data()))             # e.g. 5494 for a full LIFU field

# First target
target = targs.data()[0]
print(target.aps_id)                 # integer APS ID
print(target.targra, target.targdec) # RA/Dec in degrees
print(target.cname)                  # WEAVE CNAME identifier
print(target.targid)                 # WEAVE TARGID
print(target.targsrvy)               # survey code
print(target.targclass)              # target class
print(target.fib_status)             # 'A' (active) or 'P' (parked)
```

### Spectra

Each target has a `spectra` list. Before joining arms it has one element per arm; after joining it has one element.

```python
# Access the first (or only) spectrum
spec = target.spectra[0]
print(spec.wave)   # wavelength array (Å)
print(spec.flux)   # flux array (in funit units)
print(spec.ivar)   # inverse variance array
print(spec.sens)   # sensitivity function
```

### Metadata

Each arm has a corresponding metadata dictionary `target.meta[arm_index]` containing all processing flags and calibration information:

```python
meta = target.meta[0]

# Identifiers
meta["APS_ID"]       # integer
meta["TARGID"]       # string
meta["CNAME"]        # string
meta["TARGRA"]       # float, degrees
meta["TARGDEC"]      # float, degrees

# Observing mode flags
meta["AIRVAC"]       # 0=air, 1=vacuum
meta["TELLUR"]       # 0=not masked, 1=tellurics masked
meta["SKYSUB"]       # 0=nss, 1=sky subtracted

# Processing flags
meta["GAPS_MASKED"]         # 1 if gap masking applied
meta["GAP_FILL"]            # 1 if gap filling applied
meta["CR_CLEAN"]            # 1 if cosmic ray removal applied
meta["SENS_CORR"]           # 1 if sensitivity correction applied
meta["IVAR_NORMALIZED"]     # 1 if IVAR normalised across arms
meta["IVAR_WEIGHT"]         # normalisation weight applied to this arm

# Quality
meta["SNR"]          # broadband mean SNR

# LSF/FWHM (after loading calibration)
meta["fwhm"]         # dict with 'interpolate_function' for this fibre
meta["gfwhm"]        # dict with 'interpolate_function' for global average
```

---

## 7. Methods and Properties

### Data access

| Method / Property | Return type | Description |
|---|---|---|
| `targs.data()` | list of `APSTARG` | Full list of target objects. Main output. |
| `targs.id()` | ndarray int32 | Array of APS IDs for all targets. |
| `targs.infiles()` | list of str | Input files actually used (may differ from input if wavelength clipping excluded an arm). |
| `targs.idfx()` | dict | Mapping: APS_ID → index in `data()`. |
| `targs.idxf()` | dict | Mapping: index in `data()` → APS_ID. |
| `targs.apstoid(aps_id)` | int | Return the index in `data()` for a given APS_ID. |
| `targs.idtoaps(target_id)` | int | Return the APS_ID for a given index in `data()`. |

### Spectral information

| Method / Property | Return type | Description |
|---|---|---|
| `targs.wavelist()` | list of ndarray | Wavelength arrays, one per arm (or one if joined). |
| `targs.wlranges_def()` | list of [float, float] | Maximum available wavelength ranges from file headers. |
| `targs.wlranges()` | list of [float, float] | Wavelength ranges actually used (after user clipping). |
| `targs.wlranges_original()` | list of [float, float] | Original wavelength ranges before any modification. |
| `targs.nbands()` | int | Number of spectral arms (1 if joined, 2 if not). |
| `targs.funits()` | float | Flux unit scale factor (e.g. `1e-18` means fluxes are in 10⁻¹⁸ erg/s/cm²/Å). |

### Instrument configuration

| Method / Property | Return type | Description |
|---|---|---|
| `targs.mode()` | str | Observing mode: `'MOS'`, `'LIFU'`, `'MIFU'`, etc. |
| `targs.setups()` | list of str | Current setup strings (e.g. `['BLUELR11', 'REDLR11']` or `['Combined']`). |
| `targs.setups_original()` | list of str | Original setup strings before arm joining. |
| `targs.resolution()` | list of float | Default spectral resolution for each setup. |
| `targs.res_mode()` | list of str | Resolution mode per arm: `'LR'` or `'HR'`. |
| `targs.camera()` | list of str | Camera per arm: `'BLUE'`, `'GREEN'`, `'RED'`. |
| `targs.xbin()` | list of int | Spatial binning in X per arm. |
| `targs.ybin()` | list of int | Spatial binning in Y per arm. |
| `targs.fpmode()` | list of str | Focal plane mode per arm (e.g. `'MOS-A'`). |
| `targs.fmode()` | str | Unique focal plane mode string (after deduplication). |
| `targs.fmode_unit()` | str | Normalised spectrograph unit name (removes arm suffix: `'MOS-A'` → `'MOS'`). |

### Observing metadata

| Method / Property | Return type | Description |
|---|---|---|
| `targs.obid()` | list of str | Observation block IDs from file headers. |
| `targs.obsdate()` | list of str | Observation dates (`DATE-OBS`). |
| `targs.arms_ratio()` | list of float | Arms ratio values applied. |

### Processing status

| Method / Property | Return type | Description |
|---|---|---|
| `targs.join_arms()` | bool | Whether arm joining was applied. |
| `targs.split_arms()` | bool | Whether arm splitting was applied after joining. |
| `targs.collapsed()` | bool | Whether spaxel/fibre collapsing was applied. |
| `targs.skysub()` | bool | Whether sky-subtracted data was used. |
| `targs.normalize_ivar()` | bool | Whether IVAR normalisation was applied. |

### Spatial information

| Method / Property | Return type | Description |
|---|---|---|
| `targs.origin()` | [float, float] | Field centre coordinates [RA, Dec] in degrees. |
| `targs.centroid()` | [float, float] | Centroid of selected targets [RA, Dec] in degrees. May differ from `origin()` if an aperture was applied. |
| `targs.skyCoords()` | `SkyCoord` | Astropy SkyCoord object for all target positions. |
| `targs.wcs_h1()` | `astropy.wcs.WCS` | WCS from the first extension header. |

### LSF / FWHM

| Method | Return | Description |
|---|---|---|
| `targs.get_fwhm(aps_id=None, fwhm_key='fwhm')` | list of callables | FWHM interpolation functions, one per arm. Call with a wavelength array to get FWHM values in Å. `aps_id=None` returns the global average. |
| `targs.get_fwhm_values(aps_id=None, wavelengths=None, fwhm_key='fwhm')` | list of ndarray | FWHM values evaluated at `wavelengths`, one array per arm. |

### Packing for Redrock

| Method | Return | Description |
|---|---|---|
| `targs.pack_2_redrock(cache_Rcsr=True, use_interpolated_fwhm=True, fwhm_key='fwhm', resolution_mode='fiber_specific', ncpus=1)` | `(rr_list, apsmeta)` | Pack into Redrock Target objects with resolution matrices. `rr_list` is passed directly to Redrock. `apsmeta` contains per-target metadata. |

---

## 8. Standalone Usage — Reading WEAVE Files

APSOB can be used entirely independently of the rest of the pipeline to read and inspect WEAVE L1 data. Here is a complete tutorial.

### Minimal example — read a LIFU cube

```python
from PyAPS.aps_utils import APSOB

targs = APSOB(
    infiles=[
        "<PYAPS_DATA>/L1/20240808/stackcube_3071431.fit",   # blue arm
        "<PYAPS_DATA>/L1/20240808/stackcube_3071430.fit",   # red arm
    ],
    join_arms=True,
    caldir="<PYAPS_DATA>/CAL",
)

print("Mode:", targs.mode())          # LIFU
print("N spaxels:", len(targs.id()))  # e.g. 5494
print("Setups:", targs.setups())      # ['Combined']
print("Wavelength range:", targs.wavelist()[0][[0, -1]])  # [3800., 9280.]
```

### Access the spectrum for a specific spaxel

```python
# Get spaxel with APS_ID = 1234
target_idx = targs.apstoid(1234)
spec = targs.data()[target_idx].spectra[0]

print("Wavelength range:", spec.wave[0], "to", spec.wave[-1], "Å")
print("N pixels:", len(spec.wave))
print("Median flux:", np.nanmedian(spec.flux))
print("SNR:", targs.data()[target_idx].meta[0]["SNR"])
```

### Plot the spectrum

```python
import matplotlib.pyplot as plt
import numpy as np

target = targs.data()[target_idx]
spec   = target.spectra[0]
meta   = target.meta[0]

fig, axes = plt.subplots(2, 1, figsize=(14, 6),
                          gridspec_kw={"height_ratios": [2, 0.8]},
                          sharex=True)

# Flux + IVAR
ivar_plot = spec.ivar.copy()
ivar_plot[ivar_plot <= 0] = np.nan

axes[0].plot(spec.wave, spec.flux, "k-", lw=0.5, alpha=0.8, label="Flux")
axes[0].set_ylabel("Flux (10⁻¹⁸ erg/s/cm²/Å)")
axes[0].set_title("APS_ID=%d  CNAME=%s  SNR=%.1f" % (
    target.aps_id, target.cname, meta["SNR"]))
axes[0].legend(fontsize=8)

axes[1].plot(spec.wave, ivar_plot, color="steelblue", lw=0.5)
axes[1].fill_between(spec.wave, 0, ivar_plot,
                      color="steelblue", alpha=0.3)
axes[1].set_ylabel("IVAR")
axes[1].set_xlabel("Wavelength (Å)")

plt.tight_layout()
plt.savefig("spectrum.png", dpi=150)
```

### Loop over all spaxels

```python
for target in targs.data():
    spec = target.spectra[0]
    snr  = target.meta[0]["SNR"]
    if snr > 5:
        print("APS_ID=%d  RA=%.4f  Dec=%.4f  SNR=%.1f" % (
            target.aps_id, target.targra, target.targdec, snr))
```

### Read MOS data (fibres)

```python
targs_mos = APSOB(
    infiles=[
        "<PYAPS_DATA>/L1/20240515/stacked_3059328.fit",   # blue
        "<PYAPS_DATA>/L1/20240515/stacked_3059327.fit",   # red
    ],
    targsrvy=["WA"],                  # only WEAVE-Apertif targets
    targclass=["GALAXY"],             # only galaxies
    join_arms=True,
    sens_corr=True,
    mask_gaps=True,
    tellurics=True,
    caldir="<PYAPS_DATA>/CAL",
)

print("N galaxy fibres selected:", len(targs_mos.id()))
```

### Select targets within an aperture

```python
# Only spaxels within an ellipse centred on the galaxy
targs_aperture = APSOB(
    infiles=[...],
    area=[185.198164, 58.092634,   # RA, Dec (degrees)
          60.0, 40.0,               # full major, full minor axis lengths (arcsec)
          45.0],                    # position angle (degrees)
    join_arms=True,
    caldir="<PYAPS_DATA>/CAL",
)
```

### Collapse all spaxels to a single spectrum

```python
targs_collapsed = APSOB(
    infiles=[...],
    collapse=True,
    join_arms=True,
    caldir="<PYAPS_DATA>/CAL",
)

# There is now only one target with APS_ID = -999
collapsed_spec = targs_collapsed.data()[0].spectra[0]
print("Collapsed spectrum SNR:", targs_collapsed.data()[0].meta[0]["SNR"])
```

### Get FWHM / LSF profiles

```python
# Get global FWHM interpolation function (for all arms)
fwhm_funcs = targs.get_fwhm(aps_id=None, fwhm_key='gfwhm')

# Evaluate at a wavelength grid
wave_grid = np.linspace(3800, 9280, 1000)
fwhm_values = fwhm_funcs[0](wave_grid)   # Å

print("FWHM at 5000 Å: %.3f Å" % fwhm_funcs[0](5000.0))
print("FWHM at 8000 Å: %.3f Å" % fwhm_funcs[0](8000.0))

# Get fibre-specific FWHM for spaxel APS_ID=1234
fwhm_fiber = targs.get_fwhm(aps_id=1234, fwhm_key='fwhm')
fwhm_at_5000 = fwhm_fiber[0](5000.0)
```

### Pack for Redrock

```python
# Pack the data into Redrock format
rr_list, apsmeta = targs.pack_2_redrock(
    cache_Rcsr=True,
    use_interpolated_fwhm=True,
    fwhm_key='gfwhm',
    resolution_mode='global_average',   # faster, shared resolution matrix
)

# rr_list is now ready to pass to Redrock
import redrock.photometry
from redrock.fitz import ZFit
# ... standard Redrock call
```

---

## 9. Advanced Usage

### Using only one arm

Pass only one file and set `join_arms=False`:

```python
targs_blue = APSOB(
    infiles=["<PYAPS_DATA>/L1/20240808/stackcube_3071431.fit"],  # blue only
    join_arms=False,
    caldir="<PYAPS_DATA>/CAL",
)
print("Setups:", targs_blue.setups())   # ['BLUELR11']
```

### Disabling sky subtraction

Use the non-sky-subtracted extensions (e.g. for sky spectrum analysis):

```python
targs_nss = APSOB(infiles=[...], skysub=False, caldir="<PYAPS_DATA>/CAL")
```

### Restricting to a specific wavelength range

```python
targs = APSOB(
    infiles=["blue.fit", "red.fit"],
    wlranges=[[3800.0, 5950.0], [5900.0, 9280.0]],
    join_arms=True,
    caldir="<PYAPS_DATA>/CAL",
)
```

### Masking specific spaxels

```python
# Exclude spaxels with APS_IDs 100, 200, 300
targs = APSOB(
    infiles=[...],
    mask_aps_ids=[100, 200, 300],
    caldir="<PYAPS_DATA>/CAL",
)
```

### Parallel Redrock packing

For large IFU fields with per-fibre resolution matrices, parallel packing can significantly reduce processing time:

```python
rr_list, apsmeta = targs.pack_2_redrock(
    use_interpolated_fwhm=True,
    fwhm_key='fwhm',
    resolution_mode='fiber_specific',   # per-fibre resolution matrices
    ncpus=8,                            # 8 parallel processes
)
```

### Checking processing flags

```python
# Did any spaxel have cosmic rays removed?
for target in targs.data():
    if target.meta[0]["CR_CLEAN"] == 1:
        print("CR removed: APS_ID=%d" % target.aps_id)

# Check IVAR normalisation weights
for target in targs.data()[:5]:
    meta = target.meta[0]
    if meta.get("IVAR_NORMALIZED"):
        print("APS_ID=%d  IVAR_WEIGHT=%.4f" % (
            target.aps_id, meta["IVAR_WEIGHT"]))
```

---

## 10. IVAR Normalisation — Theory and Practice

This section explains why IVAR normalisation matters and how to choose the right mode.

### The problem

When fitting a model simultaneously to blue and red arm data, the chi-square is:

```
χ² = Σ_blue (flux_i - model_i)² × ivar_i  +  Σ_red (flux_j - model_j)² × ivar_j
```

If `sum(ivar_blue) >> sum(ivar_red)`, the blue arm dominates the fit and the red arm effectively becomes a passenger. This happens because:

- The blue and red arms have different throughputs (blue is typically lower)
- The blue arm has more wavelength pixels in LIFU LR mode
- The sensitivity correction amplifies the difference (dividing by small numbers gives large ivar)

### The balanced solution

The `balanced` mode ensures each arm contributes equally:

```
weight_blue = mean_sum_ivar / sum(ivar_blue)
weight_red  = mean_sum_ivar / sum(ivar_red)
```

After multiplication both arms have the same `sum(ivar)`, so each contributes 50% of the total chi-square. This is the safest default for most survey targets.

### When to use each mode

**`balanced`**: Use for mixed galaxy/star fields where you want an unbiased classification from the combined blue+red spectrum. Neither arm dominates.

**`pixels`**: Use when the wavelength coverage itself is the key discriminant — e.g. when a strong feature exists only in one arm and you want that arm to have more weight proportional to how many pixels it covers.

**`hybrid`**: A compromise. Useful when you know one arm is particularly noisy but you still want its wavelength coverage to matter somewhat.

**`normalize_ivar=False`**: Use when backward compatibility with the old `arms_ratio` approach is required, or when the absolute IVAR values need to be preserved for external analysis.

### Checking the result

```python
# After construction, check what weights were applied
for target in targs.data()[:3]:
    for arm_idx, meta in enumerate(target.meta):
        print("APS_ID=%d  arm=%d  IVAR_NORMALIZED=%d  IVAR_WEIGHT=%.4f" % (
            target.aps_id, arm_idx,
            meta.get("IVAR_NORMALIZED", 0),
            meta.get("IVAR_WEIGHT", 1.0)))
```

---

## 11. Resolution Matrices and LSF

The resolution matrix `R` describes how the spectrograph blurs an intrinsically sharp spectral feature. A delta function at wavelength `λ₀` is observed as a Gaussian with FWHM = `fwhm(λ₀)`. The resolution matrix encodes this convolution so Redrock can correctly model the observed spectrum.

### Why wavelength-dependent FWHM matters

For WEAVE data the LSF FWHM varies by 20–40% across the wavelength range of a single arm. Using a fixed mean FWHM introduces systematic errors in the resolution matrix that bias the chi-square and can cause misclassification near the edges of the wavelength range.

### The two LSF sources

**`lsftype='LSF'`**: Uses pre-computed files `lsf_<setup>_<fpmode>.fits` in the calibration directory. These are measured from arc lamp calibration exposures and contain per-fibre FWHM profiles at discrete wavelength points. APSOB fits splines through these profiles using `run_lsf_analysis()`.

**`lsftype='FWHM'`**: Uses wavelength calibration files `wave_<setup>_<fpmode>.fits`. More sensitive to individual calibration arcs but potentially higher spatial resolution in the FWHM profile.

### Resolution matrix construction

From the FWHM profile `fwhm(λ)`, the resolution matrix is built as a banded Gaussian:

```
R[i, j] = exp(-0.5 × ((i-j) / sigma_pix(j))²)
```

where `sigma_pix = fwhm / (2.355 × δλ)` converts FWHM in Angstroms to the pixel scale.

### Fiber-specific vs global resolution

**`resolution_mode='fiber_specific'`**: Each fibre/spaxel gets its own `R` matrix computed from its individual FWHM profile. This is the most accurate approach, especially for MOS data where the LSF can vary significantly across the focal plane. Requires more memory and computation time.

**`resolution_mode='global_average'`**: All fibres share one `R` matrix computed from the mean FWHM across all active fibres. Faster and uses less memory. Appropriate when fibre-to-fibre FWHM variation is small (typically IFU data).

### Deconvolution

When `use_resolution_deconvolution=True`, the template library resolution (characterised by `template_sigma0_angstrom`) is deconvolved from the data resolution matrix using quadratic subtraction:

```
σ²_deconv = σ²_data - σ²_template
```

This allows templates at infinite resolution to be used directly. If `σ_data < σ_template` at any wavelength (the template is broader than the data), the original profile is kept for those pixels.

---

## 12. Troubleshooting

### "NO spectrum to Proceed with!"

All fibres/spaxels were filtered out by the combination of `aps_ids`, `targsrvy`, `targclass`, `area`, `mask_areas`, and `mask_aps_ids`. Check:

1. Does `targsrvy` match the actual survey codes in the `FIBTABLE`? Try printing them: `print(set(Table.read(infile, hdu='FIBTABLE')['TARGSRVY'].filled('')))`
2. Is the `area` aperture centred correctly on the target?
3. Are all desired `aps_ids` actually present in the file?

### "Cannot find config directory"

APSOB cannot locate the `configs/ExGal_configs/` directory relative to the PyAPS installation. Either:

- Provide `configdir` explicitly: `APSOB(..., configdir="<PYAPS_DIR>/configs/ExGal_configs/")`
- Check that `PyAPS.__file__` points to the correct installation

### "Input file(s) have identical CAMERA ID"

You have passed two files from the same arm (e.g. two blue arm files). The `infiles` list must contain at most one file per camera colour.

### "CCDXBIN and CD1(3)_1(3) value are not consistent"

The wavelength step in the header (`CD1_1` or `CD3_3`) differs by more than 10% from the expected value for this configuration. This usually indicates a corrupted header or a non-standard binning mode. Check the file header with `astropy.io.fits.getheader(filename, 1)`.

### Bad overlap detected — flux jump at arm junction

```
Significant difference detected — overlap IVAR zeroed
```

The median flux in the overlap region differs by more than `aps_constants.bad_overlap_epsilon` between the blue and red arms. This indicates a flux calibration discontinuity. The affected region is masked in IVAR. Check the `arms_ratio` parameter — it may need adjusting for this observation.

### LSF pickle loading fails

If the LSF analysis fails for any reason, APSOB sets `fwhm = None` in the metadata and continues. Downstream code that calls `get_fwhm()` will raise an error if it encounters `None`. To force recomputation of the LSF profiles, delete the relevant pickle files from `configs/lsf/<CALDATE>/` and rerun.

### Loading a large IFU/LIFU stackcube is slow

For a full spatial cube (tens of thousands of spaxels, as opposed to a
fibre-table MOS/MOSLIFU file), most of the wall time is genuinely I/O and
numeric work proportional to the data volume — but two things are worth
knowing:

- **`_process_arms_serial` runs each arm concurrently (threads), not one
  after another** — each arm's file read + LSF setup + vectorized
  processing is fully independent of the others, and both are dominated by
  FITS I/O and numpy/scipy C-level calls that release the GIL, so this is
  a real wall-clock win (confirmed live: ~24% faster total load time on a
  real 2-arm, 30,832-spaxel dataset), not just a theoretical one. If you
  add a third/fourth arm, the same pool scales to however many `infiles`
  there are.
- **The FWHM/LSF interpolator's per-fibre, global, and merged-arms
  evaluation functions are vectorized** (as of `aps_lsf.py` v1.2 — see
  that file's own version history) — before this fix, evaluating the LSF
  at a wavelength grid was a Python `for` loop over every point, which on
  a large stackcube (30k+ spaxels, each needing its own FWHM lookup for
  `aps_l1_preview.py`'s diagnostics cache) could dominate the *entire*
  load time (confirmed: ~220s of a ~250s total load, on the exact dataset
  that first surfaced this).
- **Stale pre-v1.2 `.dill` caches now self-heal automatically** — every
  `LSFInterpolator` stamps itself with a `_cache_format_version`
  (`aps_lsf._LSF_CACHE_FORMAT_VERSION`) at construction time, and
  `run_lsf_analysis()` checks that stamp before trusting a cached pickle,
  regenerating it once (transparently, logged as "Cached pickle predates
  a performance fix, rebuilding") if it predates the current format —
  `overwrite=False` (the default) no longer means "never re-check," only
  "don't rebuild something that's already current." This replaced an
  earlier, incorrect assumption that stale caches would "self-heal" on
  their own with no actual mechanism behind it — confirmed for real: a
  cache built ~6 months before the vectorization fix (`configs/
  ExGal_configs/lsf/20250909/{BLUE,RED}L11_LIFU.dill`) was still running
  the old, slow closure right up until this stamp/check existed, at
  ~7.8ms/call (vs. the fix's own ~106µs/call benchmark) — a 280s L1 load
  dropped to 32.8s the first time (one-time rebuild of both arms' caches)
  and 24.5s on every load after. If a load still feels unexpectedly slow
  after this, it's no longer a stale-cache question — profile it properly
  (plain wall-clock instrumentation around each stage, not `cProfile` —
  see the methodology note in `tests/test_l1_fwhm_lsf.py`'s history/git
  log for why) rather than assuming the same root cause again.
- **Not yet implemented, worth considering if repeat loads of the same
  dataset are common** (e.g. iterative interactive use of
  `aps_explorer.py`): a "prepared target list" cache — serializing the
  *fully processed* per-arm flux/ivar/sens arrays and per-target metadata
  to disk (HDF5 is the natural format here: chunked, compressible, and
  already a project dependency via `h5py`), keyed by a hash of the input
  file paths/mtimes and every processing-option flag — mirroring the
  proven `lsf/<CALDATE>/*.dill` caching pattern above, but for the whole
  load, not just the LSF step. This would let a *repeat* load of the same
  dataset+options skip FITS re-reading and re-masking/re-correcting
  entirely. Not built yet because getting the cache-invalidation key
  right (it must hash literally every option that can change the output,
  or a stale cache could silently serve wrong data — a correctness risk
  worse than a slow load) needs a deliberate design pass, not a quick add.

### Checking what was actually applied

```python
# Print a summary of all processing flags for the first target
target = targs.data()[0]
meta   = target.meta[0]

for key in sorted(meta.keys()):
    if not key.endswith("_datatype"):
        print("%-30s: %s" % (key, meta[key]))
```

---

*For further information, bug reports, or contributions please open an issue on the PyAPS repository.*
