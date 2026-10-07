# WEAVE Data Products User Guide (iDR1.1)


## Document Revision History

| Date | Version | Author | Description |
|------|---------|--------|-------------|
| March 2026 | 1.1 | CASU Team | Updated for CPS pipeline v0.93 and v0.94, APS pipeline v1.3 and v1.7 |
| 28 October 2025 | 1.0 | CASU Team | Initial publication — Data Model 8.0, CPS pipeline v0.91 and v0.92, APS pipeline v1.3 |

---


## Table of Contents
1. [Introduction](#introduction)
2. [Data Processing Levels Overview](#data-processing-levels-overview)
3. [Raw Data Products](#raw-data-products)
4. [L1 Data Products (CPS)](#l1-data-products-cps)
5. [L2 Data Products (APS)](#l2-data-products-aps)
6. [File Naming Conventions and Data Flow](#file-naming-conventions-and-data-flow)
7. [Quick Reference Tables](#quick-reference-tables)

---

## Introduction

This comprehensive guide provides detailed technical documentation for understanding WEAVE data products, their structure, naming conventions, and the complete data flow from raw observations through to science-ready products. This documentation covers all aspects of Raw, L1 (Level 1), and L2 (Level 2) data files produced by the WEAVE instrument and its processing pipelines.

### Processing Systems

- **CPS**: WEAVE Core Processing System — produces L1 products
- **APS**: WEAVE Advanced Processing System — produces L2 products

### Document Scope
- **Raw Data**: Direct output from the WEAVE instrument via UltraDAS data acquisition system
- **L1 Products**: Processed by the Core Processing System (CPS) at Cambridge
- **L2 Products**: Analysed by the Advanced Processing System (APS)

> **Note**: Contributed Software (CS) products are not yet implemented in Data Model v8.0 and are therefore not covered in this guide.

---

## ⚠️ Important Caveats for iDR1.1

> **CAUTION**: The L1 and L2 data in this internal data release (iDR1.1) were processed using L1 pipeline version 0.91/0.92/0.94 and L2 pipeline version 1.3/1.7 conforming to Data Model 8.0. These pipeline versions contain known issues with wavelength calibration, flux calibration, and inverse variance computation. Users should consult the release notes for CPS v0.94 for detailed information on these issues.

### Known Calibration Issues

#### 1. Fibre Throughput Calibration

Expected spatial variations in fibre throughput are not yet incorporated into the pipeline. Currently, relative fibre throughput is determined using SALSA flats with all fibres positioned in a circle at 0.5 degrees radius from the field centre.

**Impact on Data Quality:**

- **Vignetting Effects**: Throughput variations occur as a function of radial distance from the field centre and can be approximated with a radial correction
- **Fibre Tilt Angle Effects**: When fibres are positioned at angles away from their park position, the flux enters the fibre at different angles, affecting throughput and introducing systematic variations across the field

#### 2. Sensitivity Function Determination

Deriving accurate sensitivity functions requires both precise fibre throughput calibration (see above) and observations of spectrophotometric standard stars across the field of view.

**Current Implementation:**

- Some MOS sensitivity functions required approximation to enable pipeline processing
- Current MOS sensitivity functions are scaled versions of the well-calibrated LIFU sensitivity functions
- Scaling factors differ slightly between the MOS slits corresponding to plates A and B
- While this approach provides a reasonable first-order approximation for flux calibration, it does not represent the final calibration solution

#### 3. Sky Subtraction Accuracy

First-order sky subtraction is performed using the median of all valid sky fibres in the observation.

**Limitation**: As described above, spatial variations in throughput (from vignetting and tilt angle effects) can affect the accuracy of sky subtraction. This can result in residual sky over-subtraction or under-subtraction of approximately 10-20% in localized regions of the field, particularly at larger radial distances from the field centre.

#### 4. Inverse Variance Computation

Issues with the inverse variance computation in L1 pipeline versions 0.91/0.92 propagated into subsequent processing stages, leading to additional uncertainties in the L2 data products produced by APS v1.3.


## Changes Introduced in CPS Pipeline Version 0.94

The following updates were introduced in **CPS pipeline version 0.94** and affect **MOS and LIFU observations** unless otherwise noted.

### 1. Updated Dichroic Feature Correction
**Date:** 2026-01-06
**Applies to:** MOS & LIFU

The method used to correct **dichroic spectral features** has been updated. The correction is now applied during the **fibre-to-fibre processing stage of spectral extraction**, meaning that spectra in **counts (ADU)** are directly corrected.

Previously, the dichroic correction was applied only as a component of the **sensitivity curves**. As a result of this change:

- The **sensitivity curves have been updated** to remove the dichroic correction component.
- The correction is now applied **earlier in the processing chain**, improving consistency in the extracted spectra.

The original dichroic correction was derived for **LIFU and MOS** using **QTH flats obtained prior to the major spectrograph intervention beginning 2024-09-16**. Following this intervention, the position of the dichroic features shifted slightly – particularly in the **blue arm**, where offsets of approximately **2–3 Å** were observed.

Although the effect is subtle, it can be noticeable in **bright standard stars**. This updated correction is included in **all L1 reprocessing performed with CPS v0.94 and later**.

---

### 2. Scattered Light Correction for LIFU
**Date:** 2026-01-06
**Applies to:** LIFU

Support for **scattered light correction** has been introduced for LIFU observations.

This correction uses **master masks derived from reference master fibre traces**. The masks conservatively identify regions:

- Between slitlets
- Near detector edges

These regions are used to **estimate scattered light robustly** and generate a **smooth detector-level X–Y correction function**. The correction is applied to **2D detector images prior to spectral extraction**.

Preliminary testing indicates that one of the main improvements is seen in the **quality of extracted sky-fibre spectra**.

**Additional fixes (2026-02-11):**

- Fixed a bug in the use of the mask file for **specbin = 2** data
- Corrected a **flux conservation issue at detector spectral quadrant boundaries for specbin = 2**

All affected data were subsequently **reprocessed**.

---

### 3. Updates to FIBTABLE Columns
**Date:** 2026-01-06
**Applies to:** MOS & LIFU

Several changes were made to the **FIBTABLE columns produced during L1 processing**:

- **Removed redundant QA columns** that had never been populated.
- **Removed the redundant `EXPTIME` column**, since this information is already present in the **Primary Header Unit (PHU)**.
- **Added two new columns** that record the **number of saturated pixels in each extracted spectrum**, one per detector.

These updates simplify the FIBTABLE structure while providing additional quality diagnostics.

---

### 4. Improved HR Fibre-to-Fibre Normalisation
**Date:** 2026-01-21
**Applies to:** MOS & LIFU

The **high-resolution (HR) fibre-to-fibre normalisation** has been improved by using a more controlled approach based on the **common wavelength coverage of fibres as a function of slit position**.

This update also reduces **induced spectral features caused by shadowing near the edge of the Red HR1 detector (detector #1)**.

The change has **minimal impact on low-resolution (LR) observations**, since LR fibres generally share similar wavelength coverage across slit positions.

---

### 5. Improved Arc Line Detection for MOS
**Date:** 2026-02-09
**Applies to:** MOS

Following the **Primary Mirror Petal usage change introduced in CPS v0.93**, an additional update was required in the **arc line detection algorithm**.

This update addresses pathological cases in **high-resolution (HR) observations** where:

- **Arc exposures are longer than the science exposures**, and
- Arc lines become **superimposed on very bright target spectra**.

The updated detection method improves robustness in these situations and prevents incorrect arc identification.

### Reprocessing Plans

**Full reprocessing of WEAVE data will commence once the improved calibration dataset is complete and validated.** Users are advised to consider these limitations when conducting scientific analysis with iDR1.1 data products. For the most current information regarding data quality, calibration status, and reprocessing schedules, please consult the WEAVE Archive System (WAS) and the latest CPS/APS release notes.

---
## Changes Introduced in APS Pipeline Version 1.7


The **APS (L2) pipeline** has undergone significant architectural improvements in recent development cycles.

---

### Variable LSF Support

A major development is the introduction of **variable Line Spread Function (LSF) support** throughout the APS pipeline. This required regeneration of the spectral classification templates and integration of wavelength-dependent LSF information across several analysis modules, including **Redrock**, **FERRE**, and **RVSPECFIT**.

Starting from **APS version 1.7**, all APS modules make use of the new **LSF calibration files** derived from solar twilight observations. These LSF products allow the spectral resolution to be modelled as a function of **wavelength and slit position (`NSPEC`)**, improving consistency between calibration and science spectra.

In particular, the **extragalactic analysis module using pPXF explicitly incorporates the variable LSF**, allowing both stellar continuum and emission-line modelling to account for the wavelength-dependent instrumental resolution.

---

### Improved Classification and Redshift Determination

The **classification and redshift analysis module** has also been significantly improved. The classification templates now incorporate wavelength-dependent LSF information, improving the reliability of spectral matching and therefore enhancing both **classification accuracy and redshift determination**.

Additionally, an issue related to the **weighting approach used by the Redrock code when analysing multi-arm spectra** has been addressed. In earlier versions, the weighting between arms could lead to suboptimal solutions when combining spectral information from different wavelength ranges.

The updated APS pipeline now applies an **internal weighting strategy**, in which redshift and classification solutions are first determined independently for each arm. The pipeline then selects the optimal global solution based on these arm-level results. This approach improves robustness when combining spectra from multiple arms.

---

### Updated Emission-Line Analysis and Output Naming

Since **June 2025**, **GANDALF** emission-line fitting system has been replaced by the **pPXF framework** for emission-line analysis in the extragalactic module.

However, to maintain compatibility with **Data Model v8.0**, the **`GAND` prefix** in the output column names was retained in APS versions up to **v1.6**, even though the analysis itself was performed using pPXF.

Starting from **APS version 1.7**, this naming convention has been updated and the **`GAND` prefix has been replaced with `EMI`** to reflect the current analysis framework. For example:

---

## Calibration Files

WEAVE calibration products include several types of **master calibration (CAL) files** provided by CASU and in principle generated once per calibration cycle, under standard operations, depending on instrument stability and the needs of the processing pipelines. These files support both the **CPS (L1)** and the **APS / PyAPS (L2)** workflows. For **MOS**, calibration data is taken in **SALSA mode**, where only one in every three fibres is placed; three observations are then combined to cover all fibres. Separate calibration products are provided for **MOS-A** and **MOS-B**.

The main calibration file types are **`warc`**, **`wave`**, **`solar`**, **`pcross`**, and **`lsf`** files. The **`warc_<RUN>.fits`** files are rebinned, wavelength-calibrated arc products, while the associated **`wave_<RUN>.fits`** files contain wavelength-solution information derived from them. For MOS, these files may include the `_all` suffix to indicate that the three SALSA observations have been merged into one calibration product. Although `warc` and `wave` files can be used for LSF-related diagnostics, they should be treated with caution for science-resolution work because the calibration lamps illuminate the system at a different focal ratio (**f/7**) from on-sky data (**f/3**).

The **solar twilight flat** products, named **`solar_<RUN>.fits`**, are the main calibration files used for **fibre-to-fibre normalisation** in the CPS pipeline and also provide the basis for the derived **cross-talk** and **LSF** products. Cross-talk estimates are distributed as **`pcross_<RUN>.fits`** files and quantify the contaminating light contributed by neighbouring fibres due to overlapping spatial profiles on the detector. These values are derived from the same covariance matrices used internally by CPS and describe the underlying cross-talk before the CPS correction is applied.

The **LSF calibration files**, named **`lsf_<ARM><RES><BIN>_<MODE>.fits`**, provide the line-spread function as a function of wavelength and slit position (`NSPEC`). They are derived from the solar twilight spectra by comparison with a high-resolution solar reference spectrum and are intended for use in the advanced analysis pipeline. These LSF products have been used operationally since **PyAPS version 1.7**. In practice, they are the preferred calibration products for resolution-aware spectral modelling, since they better represent the on-sky optical behaviour than arc-lamp-based estimates.


> **Note:** Starting from **APS version 1.7**, all APS modules make use of the new **LSF calibration files** derived from the solar twilight spectra. These LSF products are now used consistently across the pipeline to account for the wavelength- and fibre-dependent spectral resolution during spectral modelling and parameter estimation.

---

## Data Processing Levels Overview

### Complete Processing Pipeline Flow

```
Raw Data (UltraDAS) -> L1 Products (CPS) -> L2 Products (APS)
```

### Data Level Characteristics

| Level | Description | Processing System | Location | Primary Output |
|-------|-------------|-------------------|----------|----------------|
| **Raw** | Unprocessed detector readouts with metadata | UltraDAS | Telescope | 2D detector images + fibre configuration |
| **L1** | Wavelength calibrated, sky-subtracted, flux-calibrated spectra | CPS | CASU Cambridge | Extracted 1D spectra or 3D datacubes |
| **L2** | Science-ready analysed products with derived parameters | APS | Cambridge | Redshifts, stellar parameters, galaxy kinematics |

### Observation Modes

WEAVE operates in three distinct modes that affect data product structure:

| Mode | Description |
|------|-------------|
| **MOS** | Multi-Object Spectroscopy |
| **mIFU** | mini-IFU (Integral Field Unit) mode |
| **LIFU** | Large IFU mode |

---


## Raw Data Products

### Complete File Structure

Raw FITS files from UltraDAS contain exactly 6 Header Data Units (HDUs):

| Extension # | Extension Name | Type | Size | Description |
|-------------|----------------|------|------|-------------|
| 0 | Primary | Header only | - | General observation parameters, no image data |
| 1 | `<detector1>`_DATA | Image | 6276x6192 | First detector raw data |
| 2 | `<detector2>`_DATA | Image | 6276x6192 | Second detector raw data |
| 5 | FIBTABLE | Binary Table | 37 columns x 968/740/609 rows | Complete fibre configuration MOS/mIFU/LIFU |
| 6 | GUIDINFO | Binary Table | Variable | Guide star coordinates and fluxes |
| 7 | METINFO | Binary Table | 1 row/min | Meteorological information |

**Detector Naming Convention:**
- Blue arm: BLUE1, BLUE2
- Red arm: RED1, RED2

### Primary Header Unit – Selected Keyword List

The primary header contains all observation-level metadata. Below are ALL required keywords for CPS processing:

| Keyword | Type | Unit | Description | Example Value |
|---------|------|------|-------------|---------------|
| **RUN** | int | - | Unique run number for this exposure | 1234567 |
| **CCDXBIN** | int | - | Binning factor in spectral direction | 1 |
| **CCDYBIN** | int | - | Binning factor in spatial direction | 1 |
| **DATE-OBS** | char | - | Date of observation [yyyy-mm-dd] | '2025-10-15' |
| **UT** | char | - | UTC start time [hh:mm:ss.sss] | '23:45:12.345' |
| **EXPTIME** | real | s | Exposure time | 1800.0 |
| **OBJECT** | char | - | Object name or calibration type | 'GA-Field-123' or 'ARC' |
| **OBSTYPE** | char | - | Type of observation | 'TARGET', 'ARC', 'BIAS' |
| **CAMERA** | char | - | Spectrograph arm identifier | 'WEAVERED', 'WEAVEBLUE' |
| **LATITUDE** | real | deg | Telescope latitude (polar corrected) | 28.7624 |
| **LONGITUD** | real | deg | Telescope longitude (polar corrected) | -17.8792 |
| **HEIGHT** | real | m | Height above sea level | 2396.0 |
| **MJD-OBS** | real | days | Modified Julian Date at start | 60234.98765 |
| **CAT-NAME** | char | - | Source catalogue name | 'GA-LRHIGHLAT_v2' |
| **CAT-RA** | char | - | Target RA [hh:mm:ss.sss] | '12:34:56.789' |
| **CAT-DEC** | char | - | Target Dec [sdd:mm:ss.ss] | '+12:34:56.78' |
| **ST** | char | - | Local sidereal time [hh:mm:ss.sss] | '13:45:23.456' |
| **AMSTART** | real | - | Airmass at observation start | 1.234 |
| **AMEND** | real | - | Airmass at observation end | 1.345 |
| **AIRMASS** | real | - | Effective mean airmass | 1.290 |
| **VPH** | char | - | Grating name | 'LowRes', 'HighRes1', 'HighRes2' |
| **CENWAVE** | int | nm | Central wavelength estimate | 600 |
| **DISPERSI** | real | nm/mm | Dispersion estimate | 0.45 |
| **ARCSRC** | char | - | Arc lamp(s) used | 'ThAr', 'Ne', 'NONE' |
| **FLATSRC** | char | - | Flat field source identifier | 'W', 'TWILIGHT', '375,490,590' |
| **AGXSEE** | real | arcsec | Mean FWHM of guide stars (x-axis) | 0.95 |
| **AGYSEE** | real | arcsec | Mean FWHM of guide stars (y-axis) | 0.92 |
| **OBTITLE** | char | - | Observation block title | 'GA High-z Survey' |
| **OBID** | int | - | Observation block identifier | 987654 |
| **OBSTART** | real | days | MJD of OB start | 60234.95 |
| **OBCLASS** | char | - | OB classification | 'science', 'calibration' |
| **NSALSA** | int | - | Number of SALSA observations | 0 or 3 |
| **SALSA_I** | int | - | SALSA observation index | 0, 1, 2, or 3 |
| **PLATE** | char | - | Current tumbler position | 'PLATE_A', 'PLATE_B', 'LIFU' |
| **OBSMODE** | char | - | Observation mode | 'MOS', 'mIFU', 'LIFU' |
| **FPMODE** | char | - | Focal plane mode | 'MOS-A', 'MOS-B', 'mIFU', 'LIFU' |
| **FLDRA** | real | deg | Field centre RA | 185.7234 |
| **FLDDEC** | real | deg | Field centre Dec | 12.3456 |
| **CCNAME*n*** | char | - | Central CNAME for IFU bundle n (1-20) | 'WVE_12345678+1234567' |

### Detector Image Extensions — Main Information

Each detector extension (BLUE1_DATA, BLUE2_DATA, RED1_DATA, RED2_DATA) contains:

| Keyword | Type | Description | Example |
|---------|------|-------------|---------|
| **INHERIT** | bool | Extension inherits primary header | T |
| **EXTNAME** | char | Extension name | 'BLUE1_DATA' |
| **CHIPNAME** | char | Name of Detector chip| 'CCDWEAVEBLUE1' |
| **CCDTYPE** | char | Type of Detector chip| 'E2VCCD231-C6' |
| **GAIN** | real | Gain [electrons/ADU] | 1.00 |
| **READNOIS** | real | Read noise [electrons] | 2.5 |
| **BIASSEC1** | char | Overscan region quadrant 1 | '[6229:6267,100:3079]' |
| **BIASSEC2** | char | Overscan region quadrant 2 | '[10:48,100:3079]' |
| **BIASSEC3** | char | Overscan region quadrant 3 | '[10:48,3114:6093]' |
| **BIASSEC4** | char | Overscan region quadrant 4 | '[1:100,2049:4096]' |
| **TRIMSEC1** | char | Science region quadrant 1 | '[3155:6226,1:3080]' |
| **TRIMSEC2** | char | Science region quadrant 2 | '[51:3122,1:3080]' |
| **TRIMSEC3** | char | Science region quadrant 3 | '[51:3122,3113:6192]' |
| **TRIMSEC4** | char | Science region quadrant 4 | '[3155:6226,3113:6192]' |

### FIBTABLE Extension — Complete Column Structure

The fibre table contains comprehensive information for all fibres:

| # | Column Name | Format | Unit | Null Value | Description |
|---|-------------|--------|------|------------|-------------|
| 1 | FIBREID | 1I | - | - | Unique fibre ID (1-1007) |
| 2 | CNAME | 20A | - | - | Coordinate name (WVE_HHMMSSSS+/-DDMMSSS) |
| 3 | FIBRERA | 1D | deg | - | Fibre position RA |
| 4 | FIBREDEC | 1D | deg | - | Fibre position Dec |
| 5 | XPOSITION | 1D | mm | - | Fibre x position on plate |
| 6 | YPOSITION | 1D | mm | - | Fibre y position on plate |
| 7 | STATUS | 1A | - | - | Status: 'P'arked, 'A'llocated, 'D'isabled, 'B'roken |
| 8 | TARGID | 30A | - | - | Survey target identifier |
| 9 | TARGX | 1D | mm | - | Target x position on plate |
| 10 | TARGY | 1D | mm | - | Target y position on plate |
| 11 | TARGSRVY | 15A | - | - | Survey name(s), pipe-separated |
| 12 | ORIENTAT | 1D | deg | - | Azimuthal fibre orientation |
| 13 | RETRIES | 1I | - | - | Number of positioning retries |
| 14 | TARGNAME | 30A | - | - | Target name |
| 15 | TARGRA | 1D | deg | - | Target RA (Gaia frame) |
| 16 | TARGDEC | 1D | deg | - | Target Dec (Gaia frame) |
| 17 | TARGEPOCH | 1D | yr | - | Target epoch |
| 18 | TARGCAT | 30A | - | - | Catalogue name and version |
| 19 | TARGPMRA | 1D | mas/yr | - | Proper motion in RA |
| 20 | TARGPMDEC | 1D | mas/yr | - | Proper motion in Dec |
| 21 | TARGPARAL | 1D | mas | - | Target parallax |
| 22 | TARGUSE | 1A | - | - | 'T'arget, 'S'ky, 'G'uide, 'C'alibration, 'R'andom |
| 23 | TARGPROG | 40A | - | - | Sub-programme name |
| 24 | TARGCLASS | 12A | - | - | Input target classification |
| 25 | TARGPRIO | 1D | - | - | Target priority (1-10) |
| 26 | MAG_G | 1D | mag | - | SDSS g magnitude (AB) |
| 27 | EMAG_G | 1D | mag | - | Error in g magnitude |
| 28 | MAG_R | 1D | mag | - | SDSS r magnitude (AB) |
| 29 | EMAG_R | 1D | mag | - | Error in r magnitude |
| 30 | MAG_I | 1D | mag | - | SDSS i magnitude (AB) |
| 31 | EMAG_I | 1D | mag | - | Error in i magnitude |
| 32 | MAG_GG | 1D | mag | - | Gaia G magnitude (Vega) |
| 33 | EMAG_GG | 1D | mag | - | Error in Gaia G |
| 34 | MAG_BP | 1D | mag | - | Gaia BP magnitude (Vega) |
| 35 | EMAG_BP | 1D | mag | - | Error in Gaia BP |
| 36 | MAG_RP | 1D | mag | - | Gaia RP magnitude (Vega) |
| 37 | EMAG_RP | 1D | mag | - | Error in Gaia RP |

---

## L1 Data Products (CPS)

### L1 Product Types and Input Dependencies

The CPS creates different L1 products based on observation mode and OB structure:

| Product Type | Input Files | Mode | Creation Criteria | Typical Input Count |
|--------------|-------------|------|-------------------|-------------------|
| **single** | Raw file (r*.fit) | All | Every raw exposure | 1 raw file |
| **stack** | Multiple singles from same OB | MOS | >1 exposure in OB | 2-10 singles |
| **stackcube** | Multiple singles from same OB | LIFU | All LIFU exposures in OB | 3-9 singles |
| **stackcubelet** | Multiple singles from same OB | mIFU | All mIFU exposures in OB | 3-9 singles |
| **superstack** | Multiple stacks from duplicate OBs | MOS | Duplicate OB configurations | 2-5 stacks |
| **supercube** | Multiple stackcubes from duplicate OBs | LIFU | Duplicate LIFU OBs | 2-5 stackcubes |
| **supercubelet** | Multiple stackcubelets from duplicate OBs | mIFU | Duplicate mIFU OBs | 2-5 stackcubelets |
| **1D supertarget** | All observations of same target | MOS | Multiple OBs, same config | 5-50 observations |
| **LIFU supertarget** | All LIFU observations of field | LIFU | Multiple OBs, small offsets | 5-20 stackcubes |
| **mIFU supertarget** | All mIFU observations of bundle | mIFU | Multiple OBs, same bundle | 5-20 stackcubelets |

### File Naming to Input Mapping

#### Single Files
```
Input:  r1234567.fit (raw)
Output: single_1234567.fit
```

#### Stack Files
```
Input:  single_1234567.fit, single_1234568.fit, single_1234569.fit
Output: stack_1234567.fit (uses lowest run number)
```

#### Superstack Files
```
Input:  stack_1234567.fit (OB1), stack_1235000.fit (OB2 duplicate)
Output: superstack_1234567.fit
```

### L1 Single/Stack/Superstack File Structure

Complete extension structure for MOS-type L1 products:

| Extension # | Extension Name | Type | Dimensions | Description | Units |
|-------------|---------------|------|------------|-------------|-------|
| 0 | Primary | Header | - | Combined observation metadata | - |
| 1 | `<arm>`_DATA | Image | [NWAVE, NFIBRE] | Calibrated, sky-subtracted spectra | ADU |
| 2 | `<arm>`_IVAR | Image | [NWAVE, NFIBRE] | Inverse variance | ADU^(-2) |
| 3 | `<arm>`_DATA_NOSS | Image | [NWAVE, NFIBRE] | Non-sky-subtracted spectra | ADU |
| 4 | `<arm>`_IVAR_NOSS | Image | [NWAVE, NFIBRE] | Inverse variance (no sky subtraction) | ADU^(-2) |
| 5 | `<arm>`_SENSFUNC | Image | [NWAVE, NFIBRE] | Sensitivity function | erg/(erg/(s\*cm^2\*Angstrom)) per ADU/s) |
| 6 | FIBTABLE | Table | NFIBRE rows | Extended fibre information | Various |

Where:
- `<arm>` = BLUE or RED
- NWAVE = Number of wavelength pixels (typically 10000)
- NFIBRE = Number of fibres (up to 960)

### L1 IFU Cube Product Structure

Complete extension structure for stackcube/supercube/stackcubelet/supercubelet:

| Extension # | Extension Name | Type | Dimensions | Description | Units |
|-------------|---------------|------|------------|-------------|-------|
| 0 | Primary | Header | - | Combined observation metadata | - |
| 1 | `<arm>`_DATA | Image | [NX, NY, NWAVE] | Calibrated datacube | ADU |
| 2 | `<arm>`_IVAR | Image | [NX, NY, NWAVE] | Inverse variance cube | ADU^(-2) |
| 3 | `<arm>`_DATA_NOSS | Image | [NX, NY, NWAVE] | Non-sky-subtracted cube | ADU |
| 4 | `<arm>`_IVAR_NOSS | Image | [NX, NY, NWAVE] | Inverse variance (no sky) | ADU^(-2) |
| 5 | `<arm>`_SENSFUNC | Array | [NWAVE] | Single sensitivity vector | erg/(erg/(s\*cm^2\*Angstrom)*ADU) |
| 6 | `<arm>`_DATA_COLLAPSE3 | Image | [NX, NY] | Wavelength-collapsed image | ADU |
| 7 | `<arm>`_IVAR_COLLAPSE3 | Image | [NX, NY] | Wavelength-collapsed inverse variance | ADU^(-2) |

Where:
- NX,NY = Spatial dimensions (varies with IFU mode)
- NWAVE = Number of wavelength pixels

### L1 Supertarget Product Structure

Structure for 1D MOS supertargets:

| Extension # | Extension Name | Type | Dimensions | Description | Units |
|-------------|---------------|------|------------|-------------|-------|
| 0 | Primary | Header | - | Combined metadata | - |
| 1 | `<arm>`_DATA | Array | [NWAVE] | Final 1D spectrum | ADU |
| 2 | `<arm>`_IVAR | Array | [NWAVE] | Inverse variance | ADU^(-2) |
| 3 | `<arm>`_DATA_NOSS | Array | [NWAVE] | Non-sky-subtracted | ADU |
| 4 | `<arm>`_IVAR_NOSS | Array | [NWAVE] | Inverse variance (no sky) | ADU^(-2) |
| 5 | `<arm>`_SENSFUNC | Array | [NWAVE] | Sensitivity function | erg/(erg/(s\*cm^2\*Angstrom)) per ADU/S) |
| 6 | FIBTABLE | Table | 27 columns 1 row | Reduced target information | Various |
| 7 | PROVENANCE | Table | NINPUT columns 1 row | Input file tracking | Various |

### L1 Primary Header Additions

Keywords added by CPS to raw headers:

| Keyword | Type | Description | Example |
|---------|------|-------------|---------|
| **CASUDATE** | char | CPS processing timestamp | '2025-10-15T12:34:56' |
| **CASUVERS** | char | CPS pipeline version | '3.0' |
| **SOFTVERS** | char | WEAVEDR software version | '1.2.3' |
| **SOFTDATE** | char | Software release date | '2025-10-01' |
| **SOFTAUTH** | char | Software author/contact | 'CASU <casuhelp@ast.cam.ac.uk>' |
| **SOFTINST** | char | Institute URL | 'CASU <http://casu.ast.cam.ac.uk>' |
| **SRVY*n*** | char | Survey names in file | 'GA-LRHIGHLAT' |
| **CALDATE** | int | Master calibration date | 20251015 |
| **PROFILES** | char | PSF profile file used | 'profiles_1234567.fits' |
| **OBTRFILE** | char | OB fibre trace file | 'trace_1234567.fits' |
| **WAVEFILE** | char | Wavelength solution file | 'wave_1234567.fits' |
| **OBWVFILE** | char | OB wavelength file | 'obwave_1234567.fits' |
| **WARCFILE** | char | Associated warc file | 'warc_1234567.fits' |
| **FFLATCOR** | char | Fibre flat correction | 'ffnorm_1234567.fits' |
| **OBFBFLAT** | char | OB fibre flat | 'obflat_12345.fits' |
| **TELLCORR** | char | Telluric correction file | 'telluric_1234567.fits' |
| **SENSFILE** | char | Master sensitivity file | 'sens_1234567.fits' |
| **NSAT** | int | Total saturated pixels | 234 |
| **NCR** | int | Total number of cosmic ray pixels | 25809 |
| **SKYSUB** | bool | Sky subtraction applied | T |
| **NSKY** | int | Number of sky spectra used | 123 |
| **PCASUB** | bool | PCA correction applied | T |
| **PCANUM** | int | Number PCA eigenvectors used | 5 |
| **PCATOTAL** | real | Percentage skyline residuals accounted for [%] | 95.3 |
| **NCALIBRT** | int | Number of calibrators used | 12 |
| **DELMAGZP** | real | Magnitude ZP offset [mag] | 0.023 |
| **DELZPGRA** | real | ZP gradient correction | 0.001 |
| **EXTCORR** | bool | Extinction correction applied | T |
| **MINSNR** | real | Minimum S/N ratio | 0.5 |
| **SPCMINSN** | int | Spectrum with min S/N | 234 |
| **FIBMINSN** | int | Fibre with min S/N | 567 |
| **MAXSNR** | real | Maximum S/N ratio | 145.6 |
| **SPCMAXSN** | int | Spectrum with max S/N | 123 |
| **FIBMAXSN** | int | Fibre with max S/N | 456 |
| **MEDSNR** | real | Median S/N ratio | 23.4 |
| **AVMEDFLX** | real | Average median flux | 23.4 |
| **BUNDLEID** | int | mIFU bundle ID (0 if not mIFU) | 0 or 1-20 |
| **NCOMB** | int | Number combined exposures | 3 |
| **PROV0000** | char | Output filename | 'stack_1234567.fit' |
| **PROV0001** | char | First input file | 'single_1234567.fit[BLUE_DATA]' |
| **PROV0002** | char | Second input file | 'single_1234568.fit[BLUE_DATA]' |
| **PROV0003** | char | Second input file | 'single_1234569.fit[BLUE_DATA]' |

### Extended FIBTABLE Columns (L1 Additions)

Complete list of columns added by CPS to raw FIBTABLE:

| Column Name | Format | Unit | Null | Description |
|-------------|--------|------|------|-------------|
| **Nspec** | 1I | - | -1 | Spectrum number in this file |
| **RMS_arc1** | 1D | Angstrom | - | Wavelength solution RMS detector 1 |
| **RMS_arc2** | 1D | Angstrom | - | Wavelength solution RMS detector 2 |
| **Resol** | 1E | Angstrom | - | Mean FWHM of arc lines |
| **Helio_cor** | 1E | km/s | - | Heliocentric velocity correction |
| **Wave_cor1** | 1D | Angstrom | - | Wavelength offset detector 1 (Note: This column was removed starting from CPS version 0.94) |
| **Wave_corrms1** | 1D | Angstrom | - | RMS of wavelength offset detector 1 (Note: This column was removed starting from CPS version 0.94) |
| **Wave_cor2** | 1D | Angstrom | - | Wavelength offset detector 2 (Note: This column was removed starting from CPS version 0.94) |
| **Wave_corrms2** | 1D | Angstrom | - | RMS of wavelength offset detector 2 (Note: This column was removed starting from CPS version 0.94) |
| **Skyline_off1** | 1D | Angstrom | - | Sky line offset detector 1 (Note: This column was removed starting from CPS version 0.94) |
| **Skyline_rms1** | 1D | Angstrom | - | Sky line RMS detector 1 (Note: This column was removed starting from CPS version 0.94) |
| **Skyline_off2** | 1D | Angstrom | - | Sky line offset detector 2 (Note: This column was removed starting from CPS version 0.94) |
| **Skyline_rms2** | 1D | Angstrom | - | Sky line RMS detector 2 (Note: This column was removed starting from CPS version 0.94) |
| **Sky_shift** | 1E | pixel | - | Sky spectrum alignment shift |
| **Sky_scale** | 1E | - | - | Sky spectrum scaling factor |
| **Exptime** | 1E | s | - | Total exposure time for target (Note: This column was removed starting from CPS version 0.94)|
| **SNR** | 1E | - | - | Mean signal-to-noise ratio |
| **Meanflux** | 1E | ADU | - | Mean flux in spectrum  |
| **Npix_sat1** | 1L | - | - | Number of Satutraed pixel detected in detector 1  (Note: This column has been added starting from CPS version 0.94) |
| **Npix_sat2** | 1L | - | - | Number of Satutraed pixel detected in detector 2  (Note: This column has been added starting from CPS version 0.94) |

### Supertarget FIBTABLE (Reduced)

For 1D supertargets, the FIBTABLE contains only essential columns:

| # | Column Name | Format | Unit | Description |
|---|-------------|--------|------|-------------|
| 1 | CNAME | 20A | - | Coordinate name |
| 2 | TARGID | 30A | - | Target identifier |
| 3 | TARGNAME | 30A | - | Target name |
| 4 | TARGRA | 1D | deg | Target RA |
| 5 | TARGDEC | 1D | deg | Target Dec |
| 6 | TARGEPOCH | 1E | yr | Catalogue epoch |
| 7 | TARGCAT | 30A | - | Catalogue name |
| 8 | TARGPMRA | 1E | mas/yr | Proper motion RA |
| 9 | TARGPMDEC | 1E | mas/yr | Proper motion Dec |
| 10 | TARGPARAL | 1E | mas | Parallax |
| 11 | TARGPROG | 1E | mas | Programme |
| 12 | TARGCLASS | 1E | mas | Object Class |
| 13-24 | MAG_*/EMAG_* | 1E | mag | Magnitudes and errors (g,r,i,GG,BP,RP) |
| 25 | SNR | 1E | - | Combined S/N ratio |
| 26 | Meanflux | 1E | - | Combined mean flux |

### PROVENANCE Table Structure

Complete structure for tracking input files in supertargets:

| Column | Format | Unit | Null | Description | Applies To |
|--------|--------|------|------|-------------|------------|
| **ninput** | 1I | - | -1 | Input spectrum counter | All |
| **filename** | 36A | - | - | Source filename | All |
| **nspec** | 1I | - | -1 | Spectrum index in source | 1D MOS |
| **nfib** | 1I | - | -1 | Fibre ID used | 1D MOS |
| **weight** | 1E | - | - | Stacking weight used | All |
| **ccname** | 20A | - | - | Central CNAME | LIFU/mIFU |
| **mifuid** | 1I | - | -1 | mIFU bundle ID | mIFU only |
| **exptime** | 1E | s | - | Exposure time | All |
| **date_obs** | 10A | - | - | Date [yyyy-mm-dd] | All |
| **mjd_start** | 1D | days | - | MJD at exposure start | All |
| **mjd_end** | 1D | days | - | MJD at exposure end | All |
| **amstart** | 1E | - | - | Starting airmass | All |
| **amend** | 1E | - | - | Ending airmass | All |
| **obid** | 1J | - | -99 | Observation block ID | All |
| **utstart** | 1D | hours | - | UT time at exposure start | All |
| **utend** | 1D | hours | - | UT time at exposure end | All |
| **lststart** | 1D | hours | - | LST time at exposure start | All |
| **lstend** | 1D | hours | - | LST time at exposure end | All |
| **seeing** | 1E | arcsec | - | Seeing | All |
| **snr** | 1E | - | - | Mean S/N ratio | All |
| **meanflux** | 1E | - | - | Mean S/N ratio | All |

### L1 Calibration Products: WARC Files

Wavelength-calibrated arc files for Line Spread Function analysis:

#### Naming Convention
```
warc_<MONTH>_<FPMODE>-<CFG>.fit

Examples:
warc_202510_A-BL1.fit    # MOS-A Blue LowRes Bin1
warc_202510_L-GH2.fit    # LIFU Green HighRes Bin2
```

#### WARC File Structure

| Extension # | Extension Name | Type | Description | Units |
|-------------|---------------|------|-------------|-------|
| 0 | Primary | Header | Inherited from raw arc | - |
| 1 | `<detector1>`_DATA | Image | Wavelength-calibrated arc spectra | ADU |
| 2 | `<detector1>`_DATA_var | Image | Variance of spectra | ADU^2 |
| 3 | `<detector2>`_DATA | Image | Wavelength-calibrated arc spectra | ADU |
| 4 | `<detector2>`_DATA_var | Image | Variance of spectra | ADU^2 |
| 5 | FIBTABLE | Table | Fibre table from raw | Various |

---

## L2 Data Products (APS)

### APS Processing Strategy and Input Selection

The WEAVE Advanced Processing System (APS) follows these principles:

1. **Deepest Data First**: APS preferentially analyses the deepest available L1 product
   - Supertargets > Superstacks > Stacks > Singles
2. **Maximum Wavelength Coverage**: Arms are joined for analysis
   - LR mode: Blue + Red arms joined
   - HR mode: Blue + Green + Red arms joined where available
3. **Automatic Input Selection**: APS automatically identifies and processes the optimal L1 inputs

### L2 Filename Construction from L1 Inputs

#### MOS OB-Level Products

The L2 filename directly reflects the L1 inputs used:

```
Format: <type1>_<run1>__<type2>_<run2>__[<type3>_<run3>]_APS.fits

Examples:
# Two-arm LR observation
single_1234568__single_1234567_APS.fits
   -> Blue L1 file    -> Red L1 file

# Three-arm HR observation
stack_2345680__stack_2345679__stack_2345678_APS.fits
   -> Blue stack   -> Green stack  -> Red stack

# Mixed types (HR with different OB coverage)
superstack_3456791__stack_3456790__stack_3456789_APS.fits
   -> Blue from multiple OBs    -> Green OB1    -> Red OB1
```

L2 filenames always follow arm order: Blue -> Green (if present) -> Red, using run numbers from the corresponding L1 files.

#### Supertarget and IFU Products

CNAME-based naming with unique identifiers:

```
Format: [PREFIX]WVE_<CNAME>_<ID>_<ARMS>_<RESBIN>[_<PATCH>]_APS.fits

Components:
- PREFIX: '' (MOS), 'L' (LIFU), 'm' (mIFU)
- CNAME: WVE_HHMMSSSS+/-DDMMSSS (from CCNAME for IFU)
- ID: Night-specific index
  - '00': Reserved for L1 supertargets
  - '01'-'99': For stacktargets (prevents filename collisions)
- ARMS: 'BR' (LR), 'GR' (Green+Red), 'BGR' (all three)
- RESBIN: Resolution+Binning (L1, L2, L4, H1, H2, H4)
- PATCH: P0000 (no patches) or P0001-P9999 (patch number)

Examples:
WVE_12345678+1234567_00_BR_L1_APS.fits         # MOS supertarget
LWVE_23456789-2345678_01_BGR_H2_P0001_APS.fits # LIFU patch
mWVE_34567890+3456789_02_BR_L1_P0000_APS.fits  # mIFU no patches
```

### Input L1 to Output L2 Mapping Examples

| L1 Input Type | L1 Filename | L2 Output Filename | Notes |
|---------------|-------------|-------------------|-------|
| Single exposures | single_1234568.fit (B), single_1234567.fit (R) | single_1234568__single_1234567_APS.fits | Blue__Red arm joining |
| OB stacks | stack_2345679.fit (B), stack_2345678.fit (R) | stack_2345679__stack_2345678_APS.fits | Blue__Red standard processing |
| Mixed types | superstack_3456790.fit (B), stack_3456789.fit (R) | superstack_3456790__stack_3456789_APS.fits | Blue__Red different depths |
| MOS supertarget | WVE_12345678+1234567_BL1.fit, WVE_12345678+1234567_RL1.fit | WVE_12345678+1234567_00_BR_L1_APS.fits | ID=00 for supertargets |
| LIFU stackcube | stackcube_4567891.fit (B), stackcube_4567890.fit (R) | LWVE_<CCNAME>_01_BR_L1_P0000_APS.fits | ID=01 for first OB |

### L2 MOS Mode Complete File Structure

| HDU # | Extension Name | Type | Contents | Typical Size |
|-------|---------------|------|----------|--------------|
| 0 | PRIMARY | Header | Processing metadata + L1 references | - |
| 1 | CLASS_TABLE | Binary Table | Classification & redshifts | NFIBRE rows |
| 2 | STAR_TABLE | Binary Table | Stellar parameters (RVSPECFIT & FERRE) | NSTAR rows |
| 3 | GALAXY_TABLE | Binary Table | Galaxy kinematics (pPXF/GANDALF) | NGALAXY rows |
| 4 | CLASS_SPEC | Binary Table | Classification spectra & models | NFIBRE x NWAVE |
| 5 | STAR_SPEC | Binary Table | Stellar spectra & fits | NSTAR x NWAVE |
| 6 | GALAXY_SPEC | Binary Table | Galaxy spectra & fits | NGALAXY x NWAVE |

### L2 Primary Header Keywords

Complete list of L2 primary header keywords:

| Keyword | Type | Description | Example |
|---------|------|-------------|---------|
| **SIMPLE** | bool | FITS standard compliance | T |
| **BITPIX** | int | Bits per data value | 8 |
| **NAXIS** | int | Number of data axes | 0 |
| **L1_REF_B** | char | L1 Blue arm reference file | 'stack_1234567.fit' |
| **L1_REF_G** | char | L1 Green arm reference file | 'stack_1234568.fit' |
| **L1_REF_R** | char | L1 Red arm reference file | 'stack_1234569.fit' |
| **DATE-OBS** | char | Observation date from L1 | '2025-10-15' |
| **OBSMODE** | char | Observation mode from L1 | 'MOS' |
| **CASUDATE** | char | CPS processing date from L1 | '2025-10-15T12:34:56' |
| **CASUVERS** | char | CPS version from L1 | '3.0' |
| **SOFTAUTH** | char | L1 software contact | 'CASU <casuhelp@ast.cam.ac.uk>' |
| **SOFTINST** | char | L1 institute URL | 'http://casu.ast.cam.ac.uk' |
| **SOFTVERS** | char | CPS pipeline version | '1.2.3' |
| **CALDATE** | int | Master calibration date | 20251015 |
| **L1_DMVER** | char | L1 data model version | '8.0' |
| **CAT-NAME** | char | Input catalogue name | 'GA-LRHIGHLAT_v2' |
| **CASUDT_B** | char | Blue arm CPS date | '2025-10-15T12:34:56' |
| **CASUDT_G** | char | Green arm CPS date | '2025-10-15T12:34:57' |
| **CASUDT_R** | char | Red arm CPS date | '2025-10-15T12:34:58' |
| **CASCHK_B** | char | Blue arm L1 checksum | 'a1b2c3d4' |
| **CASCHK_G** | char | Green arm L1 checksum | 'e5f6g7h8' |
| **CASCHK_R** | char | Red arm L1 checksum | 'i9j0k1l2' |
| **CASID_B** | int | Blue arm CPS ID | 123456 |
| **CASID_G** | int | Green arm CPS ID | 123457 |
| **CASID_R** | int | Red arm CPS ID | 123458 |
| **OBID_B** | int | Blue arm OB ID | 987654 |
| **OBID_G** | int | Green arm OB ID | 987655 |
| **OBID_R** | int | Red arm OB ID | 987656 |
| **WAVEF_B** | char | Blue wavelength file | 'wave_1234567.fits' |
| **WAVEF_G** | char | Green wavelength file | 'wave_1234568.fits' |
| **WAVEF_R** | char | Red wavelength file | 'wave_1234569.fits' |
| **WARCF_B** | char | Blue arc file | 'warc_202510_A-BL1.fits' |
| **WARCF_G** | char | Green arc file | 'warc_202510_A-GH1.fits' |
| **WARCF_R** | char | Red arc file | 'warc_202510_A-RL1.fits' |
| **APSVERS** | char | APS pipeline version | '2.1.0' |
| **APSDATE** | char | APS processing date | '2025-10-16T08:23:45' |
| **APSUID** | char | APS repository identifier | 'APS-2025-10-16-0001' |
| **APSQFLAG** | int | L2 quality flag | 0 |
| **DATAMVER** | char | Data model version | '8.0' |
| **CHECKSUM** | char | HDU checksum | 'm3n4o5p6' |
| **DATASUM** | char | Data checksum | 'q7r8s9t0' |

### CLASS_TABLE (HDU#1) — Complete Structure

Object classification and redshift measurements from Redrock:

| Column | Type | Unit | Description | Range/Values |
|--------|------|------|-------------|--------------|
| **APS_ID** | int | - | Unique APS identifier (=FIBREID for MOS) | 1-1000 |
| **TARGID** | char[30] | - | Survey target identifier | Various |
| **CNAME** | char[20] | - | Coordinate name | WVE_* format |
| **Z** | float | - | Best-fit barycentric redshift | -0.01 to 10.0 |
| **ZERR** | float | - | Redshift uncertainty | 0.0 to 1.0 |
| **ZWARN** | int | - | Warning bitmask (0=good) | See ZWARN table |
| **CLASS** | char[12] | - | Primary classification | STAR/GALAXY/QSO_HIZ/QSO_LOZ |
| **SUBCLASS** | char[20] | - | Detailed subclassification | See subclass list |
| **TARGSRVY** | char[40] | - | Survey name(s) | Pipe-separated |
| **TARGCLASS** | char[40] | - | Input target classification (This column has been added since APS version 1.7) | Pipe-separated |
| **SNR** | float | - | REDROCK signal-to-noise | 0.0 to 1000.0 |
| **CHI2** | float | - | Best-fit chi-squared | 0.0 to 1e6 |
| **DELTACHI2** | float | - | Delta-chi2 to next best | 0.0 to 1000.0 |
| **NCOEFF** | int | - | Number of template coefficients | 1-20 |
| **COEFF** | float[NCOEFF] | - | Template coefficients | Various |
| **NPIXELS** | int | - | Pixels used in fit | 100-10000 |
| **SRVY_CLASS** | char[20] | - | Survey-based classification | Various |
| **CZZ_GALAXY** | float[NZ] | - | Redshift grid for GALAXY | z values |
| **CZZ_CHI2_GALAXY** | float[NZ] | - | Chi2(z) for GALAXY template | 0.0 to 1e6 |
| **CZZ_QSO_HIZ** | float[NZ] | - | Redshift grid for QSO_HIZ | z values |
| **CZZ_CHI2_QSO_HIZ** | float[NZ] | - | Chi2(z) for QSO_HIZ | 0.0 to 1e6 |
| **CZZ_QSO_LOZ** | float[NZ] | - | Redshift grid for QSO_LOZ | z values |
| **CZZ_CHI2_QSO_LOZ** | float[NZ] | - | Chi2(z) for QSO_LOZ | 0.0 to 1e6 |
| **CZZ_STAR_**** | float[NZ] | - | Redshift grids for stellar types | z values |
| **CZZ_CHI2_STAR_**** | float[NZ] | - | Chi2(z) for stellar templates | 0.0 to 1e6 |

*Note: Stellar templates include STAR_A, STAR_B, STAR_CV, STAR_F, STAR_G, STAR_K, STAR_M, STAR_WD

### STAR_TABLE (HDU#2) — Complete Structure

Stellar atmospheric parameters from RVSPECFIT and FERRE:

*Note: This table contains results from two independent analysis methods – RVSPECFIT (columns ending in _RVS) and FERRE (columns without suffix). Both methods provide complementary stellar parameter measurements.*

| Column | Type | Unit | Description | Typical Range |
|--------|------|------|-------------|---------------|
| **APS_ID** | int | - | Unique identifier | 1-1000 |
| **TARGID** | char[30] | - | Target identifier | Various |
| **CNAME** | char[20] | - | Coordinate name | WVE_* format |
| **VRAD** | float | km/s | Barycentric radial velocity | -500 to 500 |
| **VRAD_ERR** | float | km/s | Radial velocity error | 0.1 to 50 |
| **SKEWNESS_RVS** | float | - | Velocity distribution skewness | -5 to 5 |
| **KURTOSIS_RVS** | float | - | Velocity distribution kurtosis | -5 to 20 |
| **LOGG_RVS** | float | dex | Surface gravity (RVSPECFIT) | 0.0 to 5.5 |
| **TEFF_RVS** | float | K | Effective temperature (RVSPECFIT) | 3000 to 50000 |
| **VSINI_RVS** | float | km/s | Rotational velocity | 0 to 500 |
| **FEH_RVS** | float | dex | Metallicity [Fe/H] (RVSPECFIT) | -5.0 to 1.0 |
| **ALPHA_RVS** | float | dex | [alpha/Fe] ratio (RVSPECFIT) | -0.5 to 0.5 |
| **LOGG_ERR_RVS** | float | dex | Error in log g | 0.01 to 1.0 |
| **TEFF_ERR_RVS** | float | K | Error in Teff | 10 to 500 |
| **FEH_ERR_RVS** | float | dex | Error in [Fe/H] | 0.01 to 0.5 |
| **ALPHA_ERR_RVS** | float | dex | Error in [alpha/Fe] | 0.01 to 0.3 |
| **SNR_RVS** | float | - | RVSPECFIT S/N | 1 to 1000 |
| **CHISQ_TOT_RVS** | float | - | Total chi^2 (RVSPECFIT) | 0 to 1e6 |
| **TEFF** | float | K | Effective temperature (FERRE) | 3000 to 50000 |
| **TEFF_ERR** | float | K | Error in Teff (FERRE) | 10 to 500 |
| **LOGG** | float | dex | Surface gravity (FERRE) | 0.0 to 5.5 |
| **LOGG_ERR** | float | dex | Error in log g (FERRE) | 0.01 to 1.0 |
| **FEH** | float | dex | [Fe/H] (FERRE) | -5.0 to 1.0 |
| **FEH_ERR** | float | dex | Error in [Fe/H] (FERRE) | 0.01 to 0.5 |
| **ALPHA** | float | dex | [alpha/Fe] (FERRE) | -0.5 to 0.5 |
| **ALPHA_ERR** | float | dex | Error in [alpha/Fe] (FERRE) | 0.01 to 0.3 |
| **MICRO** | float | km/s | Microturbulence | 0 to 10 |
| **MICRO_ERR** | float | km/s | Error in microturbulence | 0.1 to 2 |
| **COVAR** | float[5,5] | - | Covariance matrix | Various |
| **ELEM** | float[NELEM] | dex | Elemental abundances [X/Fe] | -2 to 2 |
| **ELEM_ERR** | float[NELEM] | dex | Errors in abundances | 0.01 to 0.5 |
| **SNR_FR** | float | - | FERRE S/N | 1 to 1000 |
| **CHISQ_TOT_FR** | float | - | Total chi^2 (FERRE) | 0 to 1e6 |
| **FLAG_FR** | int | - | Quality flag (1=good, 0=bad) | 0 or 1 |

### GALAXY_TABLE (HDU#3) — Complete Structure

> **Important Note on Emission Line Analysis: Since June 2025, GANDALF has been replaced by pPXF for emission-line analysis. However, to maintain consistency with Data Model v8.0, the column naming scheme retained the GAND prefix in APS versions up to v1.6, even though the analysis itself was performed using pPXF. Starting from APS version 1.7, the GAND prefix has been replaced with EMI (e.g., LOGLAM_GAND → LOGLAM_EMI).

Galaxy kinematics and emission line measurements:

| Column | Type | Unit | Description | Typical Range |
|--------|------|------|-------------|---------------|
| **APS_ID** | int | - | Unique identifier | 1-1000 |
| **TARGID** | char[30] | - | Target identifier | Various |
| **CNAME** | char[20] | - | Coordinate name | WVE_* format |
| **ZCORR** | float | - | Corrected systemic redshift | 0.0 to 5.0 |
| **V** | float | km/s | Stellar velocity | -1000 to 1000 |
| **SIGMA** | float | km/s | Velocity dispersion | 10 to 500 |
| **H3** | float | - | Gauss-Hermite moment h3 | -0.3 to 0.3 |
| **H4** | float | - | Gauss-Hermite moment h4 | -0.3 to 0.3 |
| **H5** | float | - | Gauss-Hermite moment h5 | -0.3 to 0.3 |
| **H6** | float | - | Gauss-Hermite moment h6 | -0.3 to 0.3 |
| **ERR_ZCORR** | float | - | Error in corrected redshift | 1e-5 to 0.01 |
| **ERR_V** | float | km/s | Error in velocity (MC) | 1 to 100 |
| **ERR_SIGMA** | float | km/s | Error in dispersion (MC) | 1 to 50 |
| **ERR_H3** | float | - | Error in h3 (MC) | 0.001 to 0.1 |
| **ERR_H4** | float | - | Error in h4 (MC) | 0.001 to 0.1 |
| **ERR_H5** | float | - | Error in h5 (MC) | 0.001 to 0.1 |
| **ERR_H6** | float | - | Error in h6 (MC) | 0.001 to 0.1 |
| **FORM_ERR_ZCORR** | float | - | Formal error in redshift | 1e-5 to 0.01 |
| **FORM_ERR_V** | float | km/s | Formal error in velocity | 0.5 to 50 |
| **FORM_ERR_SIGMA** | float | km/s | Formal error in dispersion | 0.5 to 30 |
| **FORM_ERR_H3** | float | - | Formal error in h3 | 0.001 to 0.05 |
| **FORM_ERR_H4** | float | - | Formal error in h4 | 0.001 to 0.05 |
| **FORM_ERR_H5** | float | - | Formal error in h5 | 0.001 to 0.05 |
| **FORM_ERR_H6** | float | - | Formal error in h6 | 0.001 to 0.05 |

#### Emission Line Columns (per line)

For each emission line (EL) at wavelength W:

| Column Pattern | Unit | Description |
|----------------|------|-------------|
| **FLUX_*EL*_*W*** | erg/s/cm^2 | Emission line flux |
| **AMPL_*EL*_*W*** | ADU | Line amplitude |
| **Z_*EL*_*W*** | - | Line redshift |
| **SIGMA_*EL*_*W*** | km/s | Line width |
| **AON_*EL*_*W*** | - | Amplitude/noise ratio |
| **EW_*EL*_*W*** | Angstrom | Equivalent width (added APS v1.9, 1 Sep 2026) |
| **ERR_FLUX_*EL*_*W*** | erg/s/cm^2 | Error in flux |
| **ERR_AMPL_*EL*_*W*** | ADU | Error in amplitude |
| **ERR_Z_*EL*_*W*** | - | Error in redshift |
| **ERR_SIGMA_*EL*_*W*** | km/s | Error in width |
| **ERR_EW_*EL*_*W*** | Angstrom | Error in equivalent width (flux-error-only propagation) |

**EW_*EL*_*W*** sign convention: **positive for genuine emission**
(integrated line flux divided by the local fitted stellar continuum at
the line's observed wavelength) — the opposite sign convention from the
Spectral Indices' Lick-style stellar absorption indices below (positive
for absorption, negative for net emission filling). See
`ExGalPrepare.compute_equivalent_width`'s docstring (`py/PyAPS/
ExGalPrepare.py`) for the full definition and error-propagation caveat,
and `doc/aps_rr.md`'s "Reused for ExGal" section for the related
Galactic-extinction-correction context this shares its continuum-fit
dependency with.

**EBMV_*EL*_*W*/ERR_EBMV_*EL*_*W* — not currently implemented.** This
table previously documented a per-line reddening column here, presumably
intended for ppxf's own optional `reddening=` free-parameter dust-law
fit (a *different* mechanism from the Galactic-foreground `EBmV`
correction covered by the header keywords below) — traced 1 Sep 2026 and
confirmed real output tables carry no such column; the corresponding
config key (`REDDENING`) is deliberately retired, not wired up (see
`doc/aps_rr.md`'s "Reused for ExGal" section for the full reasoning: the
actual `ppxf()` calls already use `mdegree` for continuum-shape
correction, and combining it with a `reddening=` fit is a degenerate
double-parameterisation per ppxf's own documentation). Removed from this
column list as aspirational rather than real, to avoid a future reader
expecting a column that was never populated.

**PPXF/EMIPPXF/LS output headers** (added 1 Sep 2026) also carry the
*Galactic* foreground extinction actually applied upstream (SFD98+
Fitzpatrick99, before any fit ran — see `doc/aps_rr.md`'s "Reused for
ExGal" section): `EBMVAPPL` (the applied E(B-V) in mag, or `'OFF'`), and
for MOS's per-target automatic mode, `EBMVMIN`/`EBMVMAX` giving the real
spread of per-target values actually looked up across the batch (MOS
gets one value per target; `EBMVAPPL` there is the batch median). This
is the value already subtracted from the observed spectrum, distinct
from any per-line reddening fit.

Available emission lines include:
- Balmer series: Halpha_6563, Hbeta_4861, Hgamma_4341, Hdelta_4102
- Forbidden lines: [OII]_3726, [OII]_3729, [OIII]_4959, [OIII]_5007
- [NII]_6548, [NII]_6584, [SII]_6717, [SII]_6731
- Additional diagnostic lines

#### Spectral Indices

Standard Lick indices and WEAVE-specific measurements:
- **HD_A**, **HD_F**: Hdelta indices
- **CN_1**, **CN_2**: CN molecular bands
- **Ca4227**: Calcium line
- **G4300**: G-band
- **HG_A**, **HG_F**: Hgamma indices
- **Fe4383**, **Ca4455**, **Fe4531**: Metal lines
- **C4668**: C2 Swan band
- **HB**: Hbeta index
- **Fe5015**: Iron line
- **Mg_1**, **Mg_2**, **Mg_b**: Magnesium features
- **Fe5270**, **Fe5335**: Iron lines
- **Fe5406**, **Fe5709**, **Fe5782**: Additional Fe
- **NaD**: Sodium D lines
- **TiO_1**, **TiO_2**: Titanium oxide bands

Each index includes an **ERR_*INDX*** error column.

**FWHM_FLAG**: Binary flag (1=intrinsic dispersion > instrumental, 0=not resolved)

### Spectral Data Tables (HDUs #4-6)

#### CLASS_SPEC (HDU#4) Structure

| Column | Type | Dimensions | Unit | Description |
|--------|------|------------|------|-------------|
| **APS_ID** | int | - | - | Unique identifier |
| **TARGID** | char[30] | - | - | Target ID |
| **CNAME** | char[20] | - | - | Coordinate name |
| **LAMBDA_RR_B** | float[NW] | - | Angstrom | Blue wavelength array (air) |
| **LAMBDA_RR_G** | float[NW] | - | Angstrom | Green wavelength array |
| **LAMBDA_RR_R** | float[NW] | - | Angstrom | Red wavelength array |
| **LAMBDA_RR_C** | float[NW] | - | Angstrom | Combined wavelength array |
| **FLUX_RR_*ARM*** | float[NW] | - | erg/s/cm^2/Angstrom | Calibrated spectrum |
| **IVAR_RR_*ARM*** | float[NW] | - | (erg/s/cm^2/Angstrom)^(-2) | Inverse variance |
| **MODEL_RR_*ARM*** | float[NRANK,NW] | - | erg/s/cm^2/Angstrom | Best-fit model, one per rank (NRANK up to 3, ranked by DELTACHI2) |

#### STAR_SPEC (HDU#5) Structure

| Column | Type | Dimensions | Unit | Description |
|--------|------|------------|------|-------------|
| **APS_ID** | int | - | - | Unique identifier |
| **TARGID** | char[30] | - | - | Target ID |
| **CNAME** | char[20] | - | - | Coordinate name |
| **LAMBDA_RVS_*ARM*** | float[NW] | - | Angstrom | Wavelength (RVSPECFIT) |
| **FLUX_RVS_*ARM*** | float[NW] | - | erg/s/cm^2/Angstrom | Spectrum (RVSPECFIT) |
| **ERROR_RVS_*ARM*** | float[NW] | - | erg/s/cm^2/Angstrom | Error (RVSPECFIT) |
| **MODEL_RVS_*ARM*** | float[NW] | - | erg/s/cm^2/Angstrom | Model (RVSPECFIT) |
| **LAMBDA_FR_*ARM*** | float[NW] | - | Angstrom | Wavelength (FERRE) |
| **FLUX_FR_*ARM*** | float[NW] | - | normalised | Normalised spectrum (FERRE) |
| **ERROR_FR_*ARM*** | float[NW] | - | normalised | Error (FERRE) |
| **MODEL_FR_*ARM*** | float[NW] | - | normalised | Model (FERRE) |

#### GALAXY_SPEC (HDU#6) Structure

| Column | Type | Dimensions | Unit | Description |
|--------|------|------------|------|-------------|
| **APS_ID** | int | - | - | Unique identifier |
| **TARGID** | char[30] | - | - | Target ID |
| **CNAME** | char[20] | - | - | Coordinate name |
| **LOGLAM_PPXF** | float[NW] | - | log(Angstrom) | Log wavelength (rest) |
| **FLUX_PPXF** | float[NW] | - | ADU | Log-binned spectrum |
| **ERROR_PPXF** | float[NW] | - | ADU | Error array |
| **MODEL_PPXF** | float[NW] | - | ADU | pPXF best fit |
| **GOODPIX_PPXF** | int[NW] | - | - | Good pixel mask |
| **LOGLAM_GAND** (**LOGLAM_EMI**) | float[NW] | - | log(Angstrom) | Log wavelength (GANDALF/pPXF*) |
| **FLUX_GAND** (**FLUX_EMI**) | float[NW] | - | ADU | Log-binned spectrum |
| **ERROR_GAND** (**ERROREMI**) | float[NW] | - | ADU | Error array |
| **MODEL_GAND** (**MODEL_EMI**) | float[NW] | - | ADU | Full model (continuum+lines) |
| **EMISSION_GAND** (**EMISSION_EMI**) | float[NW] | - | ADU | Emission line spectrum |
| **FLUX_CLEAN_GAND** (**FLUX_CLEAN_EMI**) | float[NW] | - | ADU | Emission-cleaned spectrum |
| **MODEL_CLEAN_GAND** (**MODEL_CLEAN_EMI**) | float[NW] | - | ADU | Continuum-only model |
| **GOODPIX_GAND** (**GOODPIX_EMI**) | int[NW] | - | - | Good pixel mask |

*Note: For APS versions up to 1.6, the GAND suffix was retained for historical consistency, even though emission-line analysis has been performed using pPXF since June 2025. Beginning with APS version 1.7, the GAND suffix has been replaced by EMI. For example, the column LOGLAM_GAND has been renamed to LOGLAM_EMI.

### L2 IFU Mode Products

#### IFU Processing Strategy

IFU data undergoes patch-based analysis:
1. Source detection in datacube
2. Patch extraction around sources
3. Voronoi binning for S/N optimisation
4. Individual analysis per patch

#### IFU File Structure – Extragalactic Sources

| HDU # | Extension Name | Type | Description |
|-------|---------------|------|-------------|
| 0 | PRIMARY | Header | Patch metadata + L1 references |
| 1 | PATCH_TABLE | Binary Table | Voronoi binning map |
| 2 | PATCH_BINSPEC | Binary Table | Binned spectra |
| 3 | GALAXY_TABLE | Binary Table | Galaxy parameters per bin |
| 4 | GALAXY_SPEC | Binary Table | Analysis spectra per bin |

#### IFU-Specific Primary Header Keywords

Additional keywords for IFU products:

| Keyword | Type | Unit | Description |
|---------|------|------|-------------|
| **BUNDLEID** | int | - | mIFU bundle ID (1-20) or 0 for LIFU |
| **CCNAME** | char | - | Central coordinate name |
| **P_ID** | int | - | Patch identifier |
| **P_RA** | real | deg | Patch centre RA |
| **P_DEC** | real | deg | Patch centre Dec |
| **P_A** | real | arcsec | Patch major-axis length (full diameter) |
| **P_B** | real | arcsec | Patch minor-axis length (full diameter) |
| **P_THETA** | real | deg | Patch rotation angle |
| **CATOVER** | int | - | Catalogue override flag (1=True, 0=False) |


#### PATCH_TABLE (HDU#1) – Extragalactic

Complete Voronoi binning information:

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| **ID** | int | - | Sequential spaxel index in patch |
| **BIN_ID** | int | - | Voronoi bin assignment |
| **APS_ID** | int | - | Original field spaxel ID |
| **TARGID** | char[30] | - | Central target identifier |
| **CNAME** | char[20] | - | Field centre CCNAME |
| **X** | float | arcsec | RA offset from field centre |
| **Y** | float | arcsec | Dec offset from field centre |
| **Z** | float | - | Initial redshift guess |
| **ZERR** | float | - | Redshift error |
| **HEALPIX_ID** | int | - | HEALPix index (NSide=1024) |
| **X_0** | float | deg | Field centre RA |
| **Y_0** | float | deg | Field centre Dec |
| **FLUX** | float | ADU | Mean spaxel flux |
| **SNR** | float | - | Spaxel S/N ratio |
| **XBIN** | float | arcsec | Bin generator X coordinate |
| **YBIN** | float | arcsec | Bin generator Y coordinate |
| **SNRBIN** | float | - | Final bin S/N after tessellation |
| **NSPAX** | int | - | Number of spaxels in bin |

#### PATCH_BINSPEC (HDU#2) - Extragalactic

| Column | Type | Dimensions | Unit | Description |
|--------|------|------------|------|-------------|
| **BIN_ID** | int | - | - | Voronoi bin identifier |
| **LOGLAM** | float[NW] | - | log(Angstrom) | Rest-frame log wavelength |
| **SPEC** | float[NBIN,NW] | - | ADU | Binned spectra |
| **ESPEC** | float[NBIN,NW] | - | ADU | Error arrays |

#### IFU GALAXY_TABLE (HDU#3) – Per Bin Analysis

Contains same columns as MOS GALAXY_TABLE but with analysis per Voronoi bin:

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| **BIN_ID** | int | - | Links to PATCH_TABLE |
| **V** | float | km/s | Velocity relative to systemic |
| **SIGMA** | float | km/s | Velocity dispersion |
| **H3-H6** | float | - | Gauss-Hermite moments |
| **(Emission line columns)** | Various | Various | Same as MOS mode |
| **(Spectral indices)** | Various | Various | Same as MOS mode |

#### IFU GALAXY_SPEC (HDU#4) – Per Bin Spectra

Same structure as MOS GALAXY_SPEC but per bin:

| Column | Type | Description |
|--------|------|-------------|
| **BIN_ID** | int | Links to PATCH_TABLE |
| **(All spectral columns)** | Various | Same as MOS GALAXY_SPEC |

### IFU Galactic Source Structure

| HDU # | Extension Name | Type | Description |
|-------|---------------|------|-------------|
| 0 | PRIMARY | Header | Patch metadata |
| 1 | PATCH_TABLE | Binary Table | Spatial information per spaxel |
| 2 | STAR_TABLE | Binary Table | Stellar parameters per spaxel |
| 3 | STAR_SPEC | Binary Table | Stellar spectra per spaxel |

#### Galactic PATCH_TABLE (HDU#1)

Same structure as extragalactic PATCH_TABLE but without Voronoi binning:

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| **ID** | int | - | Sequential spaxel index |
| **BIN_ID** | int | - | Spaxel ID (no binning) |
| **APS_ID** | int | - | Original field spaxel ID |
| **X, Y** | float | arcsec | Spatial coordinates |
| **Z, ZERR** | float | - | Redshift and error |
| **FLUX, SNR** | float | Various | Flux and S/N |
| **(Other columns)** | Various | Various | Same as extragalactic |

#### Galactic STAR_TABLE (HDU#2)

Per-spaxel stellar analysis (same columns as MOS STAR_TABLE):

| Column | Type | Description |
|--------|------|-------------|
| **BIN_ID** | int | Links to PATCH_TABLE |
| **(All stellar columns)** | Various | Same as MOS STAR_TABLE |

#### Galactic STAR_SPEC (HDU#3)

Per-spaxel stellar spectra (same structure as MOS STAR_SPEC):

| Column | Type | Description |
|--------|------|-------------|
| **BIN_ID** | int | Links to PATCH_TABLE |
| **(All spectral columns)** | Various | Same as MOS STAR_SPEC |

---



## File Naming Conventions and Data Flow

### Complete Data Flow Example: MOS Observation

```
Step 1: Raw Data Acquisition
r1234567.fit (Red, 1800s exposure)
r1234568.fit (Blue, 1800s exposure)
r1234569.fit (Red, 1800s exposure)
r1234570.fit (Blue, 1800s exposure)

Step 2: L1 Single Processing (CPS)
single_1234567.fit (Red exposure 1)
single_1234568.fit (Blue exposure 1)
single_1234569.fit (Red exposure 2)
single_1234570.fit (Blue exposure 2)

Step 3: L1 Stack Creation (CPS)
stack_1234567.fit (Red stack, uses exposures 1234567 & 1234569)
stack_1234568.fit (Blue stack, uses exposures 1234568 & 1234570)

Step 4: L2 Analysis (APS)
stack_1234568__stack_1234567_APS.fits (Blue__Red arm-joined analysis)
```

### Complex Example: Multi-OB Supertarget

```
OB1 (Night 1):
Raw: r2345678.fit (R), r2345679.fit (B)
L1: single_2345678.fit (Red), single_2345679.fit (Blue)

OB2 (Night 5):
Raw: r2346789.fit (R), r2346790.fit (B)
L1: single_2346789.fit (Red), single_2346790.fit (Blue)

OB3 (Night 10):
Raw: r2347890.fit (R), r2347891.fit (B)
L1: single_2347890.fit (Red), single_2347891.fit (Blue)

L1 Supertarget Creation:
Input: All 6 single files for target WVE_12345678+1234567
Output: WVE_12345678+1234567_BL1.fit (Blue)
        WVE_12345678+1234567_RL1.fit (Red)

L2 Analysis:
WVE_12345678+1234567_00_BR_L1_APS.fits (Blue and Red arms joined)
```

### IFU Data Flow Example

```
LIFU OB with 3 dithered exposures:

Raw Data:
r3456789.fit (R, dither 1), r3456790.fit (B, dither 1)
r3456791.fit (R, dither 2), r3456792.fit (B, dither 2)
r3456793.fit (R, dither 3), r3456794.fit (B, dither 3)

L1 Singles:
single_3456789.fit through single_3456794.fit

L1 Stackcube (per arm):
stackcube_3456789.fit (Red, combines 3 dithers)
stackcube_3456790.fit (Blue, combines 3 dithers)

L2 Analysis:
LWVE_12345678+1234567_01_BR_L1_P0001_APS.fits (Patch 1)
LWVE_12345678+1234567_01_BR_L1_P0002_APS.fits (Patch 2)
```

### Filename Component Reference

#### Raw Files
```
r<RUN>.fit
  └─ 7-digit run number
```

#### L1 Files
```
Single: single_<RUN>.fit
Stack:  stack_<RUN0>.fit (RUN0 = lowest run in stack)
Supertarget: <MODE><CNAME>_<CFG>.fit
             MODE: '' (MOS), 'L' (LIFU), 'm' (mIFU)
             CFG: <ARM><RES><BIN>
                  ARM: B/G/R
                  RES: L/H
                  BIN: 1/2/4
```

#### L2 Files
```
OB Products: <type1>_<run1>__<type2>_<run2>[__<type3>_<run3>]_APS.fits

Supertargets: <PREFIX>WVE_<CNAME>_<ID>_<ARMS>_<RESBIN>[_<PATCH>]_APS.fits
              PREFIX: '' (MOS), 'L' (LIFU), 'm' (mIFU)
              ID: 00 (supertarget), 01-99 (stacks)
              ARMS: BR/GR/BGR
              PATCH: P0000-P9999
```

---

## Quick Reference Tables

### ZWARN Bitmask Definitions

| Bit | Name | Description | Action Required |
|-----|------|-------------|-----------------|
| 0 | SKY | Sky fibre | Ignore for science |
| 1 | LITTLE_COVERAGE | Insufficient wavelength coverage | Review spectrum |
| 2 | SMALL_DELTA_CHI2 | chi^2 too close to second best | Verify classification |
| 5 | Z_FITLIMIT | chi^2 minimum at z range edge | Refit with extended range |
| 7 | UNPLUGGED | No spectrum obtained | Mark as bad |
| 8 | NULL_FIT | Constant chi^2 surface | Check data quality |

### Observation Types (OBSTYPE)

| Type | Description | CPS Processing | APS Processing |
|------|-------------|----------------|---------------|
| TARGET | Science observation | Full pipeline | Full analysis |
| FLUX_STD | Flux standard star | Single only | Limited analysis |
| RV_STD | Radial velocity standard | Single only | RV analysis only |
| ARC | Wavelength calibration | Calibration only | None |
| BIAS | Bias frame | Calibration only | None |
| DARK | Dark current | Calibration only | None |
| FIBRE_FLAT | Fibre flat field | Calibration only | None |
| SKY | Sky observation | Reference only | None |

---

*Last Updated: March 2026*
