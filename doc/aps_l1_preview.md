# APS L1 Preview v3.1 - Complete Documentation

> **⚠️ Outdated (2026-08-03, superseded further 2026-08-14):** `aps_l1_preview.py`
> was first rewritten as its own standalone Dash web app (the PyQt5 desktop GUI
> described below — menus, windows, keyboard shortcuts — no longer exists), then
> unified into `aps_explorer.py` as the single entry point for L1 *and* L2 data —
> `aps_l1_preview.py` is now imported purely as a library (state/loaders/figure
> builders), with no `Dash()` app or CLI of its own left. See
> **[doc/aps_explorer.md](aps_explorer.md)** for current usage and the complete
> feature set. The interactive features themselves carried over (fibre
> click-to-select, FWHM overview/detail — now cached per target — Slit Explorer),
> just via `aps_explorer.py` instead of a standalone app. This document is kept
> for historical reference
> and has not been updated to match.

**Interactive visualization tool for WEAVE spectroscopic L1 data with advanced FWHM diagnostics**

[![Version](https://img.shields.io/badge/version-3.1-blue.svg)](https://github.com/yourusername/aps_l1_preview)
[![Python](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-orange.svg)](LICENSE)

**Authors:** Alireza Molaeinezhad, David Murphy (CASU/IOA)  
**Project:** WEAVE Spectroscopic Survey  
**Institution:** Cambridge Astronomical Survey Unit / Institute of Astronomy

---

## Table of Contents
1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Installation](#installation)
4. [Command-Line Arguments](#command-line-arguments)
5. [Usage Examples](#usage-examples)
6. [Windows Overview](#windows-overview)
7. [FWHM Diagnostics](#fwhm-diagnostics)
8. [Quick Reference Card](#quick-reference-card)
9. [Tips & Tricks](#tips--tricks)
10. [Troubleshooting](#troubleshooting)
11. [Advanced Usage](#advanced-usage)
12. [What's New](#whats-new)
13. [Technical Details](#technical-details)
14. [Contributing](#contributing)

---

## Overview

### Features at a Glance

- 🗺️ **Interactive Fiber Map** - Click-to-select on RA/DEC plot with spherical correction
- 📊 **Dual-Panel Spectra** - View blue and red arms simultaneously  
- 🔍 **Advanced FWHM Diagnostics** - Inspect arc line quality and interpolation
- 🌌 **Aladin Integration** - Sky context for target identification
- 📋 **Smart Metadata Display** - Filtered table view of target properties
- ⚡ **Fast & Responsive** - PyQtGraph for smooth interactive plotting

### What is APS L1 Preview?

**aps_l1_preview** is an interactive GUI tool for exploring WEAVE spectroscopic L1 (reduced) data. It provides real-time visualization of spectra, inverse variance, and comprehensive FWHM diagnostics including raw arc line measurements used in the wavelength calibration and PSF characterization.

The tool features dual FWHM diagnostic windows: an overview showing all fibers simultaneously, and a detailed inspection view that displays individual arc line measurements, their weights, and fit quality.

---

## Quick Start

### Installation

```bash
# Install required packages
pip install numpy astropy pyqtgraph PyQt5 matplotlib
```

### Minimal Example
```bash
# View a single L1 file
python aps_l1_preview.py \
    --infiles /path/to/stackcube_3097086.fit \
    --aladin False
```

### Two-Arm Example
```bash
# View blue and red arms together
python aps_l1_preview.py \
    --infiles blue_arm.fit red_arm.fit \
    --wlranges 3800,5950 5900,9270 \
    --aladin False
```

### With Aladin Integration
```bash
# Integrate with Aladin for sky visualization
python aps_l1_preview.py \
    --infiles stackcube.fit \
    --l1_reference reference.fits \
    --aladin True
```

### Get Help
```bash
python aps_l1_preview.py --help
```

---

## Installation

### Prerequisites

#### Required Packages
```bash
pip install numpy astropy pyqtgraph PyQt5 matplotlib
```

**Package Versions:**
- Python 3.8+
- numpy (any recent version)
- astropy (any recent version)
- pyqtgraph (any recent version)
- PyQt5 (any recent version)
- matplotlib (any recent version)

#### Optional (for Aladin integration)
- Aladin Desktop running on your system
- SAMP protocol enabled in Aladin (Interop menu)

### Verify Installation
```bash
# Check if all imports work
python -c "import numpy, astropy, pyqtgraph, PyQt5, matplotlib; print('All packages OK')"

# Test the tool
python aps_l1_preview.py --help
```

### Platform Notes

**Linux/Unix:**
- Works out of the box
- Tested on Ubuntu 20.04+

**macOS:**
- May need to install Qt separately
- Use `brew install qt5`

**Windows:**
- Full functionality available
- Aladin integration may require additional configuration

---

## Command-Line Arguments

### Input Files (Required - one of these)

```bash
# Single or multiple files
--infiles FILE [FILE ...]
  
  # Examples:
  --infiles stackcube.fit
  --infiles blue_arm.fit red_arm.fit
  --infiles setup1.fit setup2.fit setup3.fit

# File list (one filename per line)
--infiles_list FILE
  
  # Example:
  --infiles_list observations.txt
  
  # observations.txt format:
  # blue1.fit red1.fit
  # blue2.fit red2.fit
```

### Filtering Options

```bash
# Filter by APS ID (comma-separated)
--aps_ids ID,ID,...
  --aps_ids 100,101,102

# Filter by survey type
--targsrvy NAME,NAME,...
  --targsrvy WS2022A2-002,WS2022A2-003

# Filter by target class
--targclass CLASS,CLASS,...
  --targclass GALAXY,QSO,STAR

# Exclude specific fiber IDs
--mask_aps_ids ID,ID,...
  --mask_aps_ids 200,201,202

# Select rectangular area: RA(deg), DEC(deg), Width(arcmin), Height(arcmin), PA(deg)
--area RA,DEC,W,H,PA
  --area 246.29,40.90,10.0,10.0,0.0

# Exclude multiple areas (same format as --area)
--mask_areas AREA [AREA ...]
  --mask_areas 246.0,40.0,5.0,5.0,0.0 247.0,41.0,3.0,3.0,45.0
```

### Wavelength & Processing

```bash
# Wavelength ranges (Angstroms) - one per input file
--wlranges MIN,MAX [MIN,MAX ...]
  --wlranges 3800,5950 5900,9270

# Apply sensitivity correction (default: True)
--sens_corr True|False
  --sens_corr True

# Safe mask gaps (ignore pixels around gaps)
--safe_mask_gaps True|False
  --safe_mask_gaps False

# Mask all gap pixels
--mask_gaps True|False
  --mask_gaps False

# Mask strong telluric bands
--tellurics True|False
  --tellurics False

# Use vacuum wavelengths (default: False = air)
--vacuum True|False
  --vacuum False

# Fill gaps with NaNs
--fill_gap True|False
  --fill_gap False

# Arm flux scaling ratios (one per file)
--arms_ratio RATIO,RATIO,...
  --arms_ratio 1.0,1.0

# Join spectral arms into single spectrum
--join_arms True|False
  --join_arms False

# Cosmic ray removal
--crr True|False
  --crr False

# Collapse spectra within selected area
--collapse True|False
  --collapse False
```

### Reference Catalogs

```bash
# L1 reference catalog for Aladin visualization
--l1_reference FILE
  --l1_reference /path/to/reference_20240515.fits

# L2 reference catalog for parameter display
--l2_reference FILE
  --l2_reference /path/to/l2_catalog.fits
```

### Directory Paths

```bash
# Directory for input catalogs
--catdir PATH
  --catdir $PYAPS_DATA/catalogs

# Directory for calibration files
--caldir PATH
  --caldir $PYAPS_DATA/calibrations

# Directory for configuration files
--configdir PATH
  --configdir $PYAPS_DATA/configs
```

### Visualization

```bash
# Enable/disable Aladin integration
--aladin True|False
  --aladin True    # Enable Aladin
  --aladin False   # Disable (faster startup)
```

---

## Usage Examples

### Example 1: Basic Single Observation
```bash
python aps_l1_preview.py \
    --infiles <PYAPS_DATA>/L1/20240515/stackcube_3059328.fit \
    --aladin False
```
**Use case:** Quick look at a single observation without Aladin

---

### Example 2: Dual-Arm with Wavelength Ranges
```bash
python aps_l1_preview.py \
    --infiles $PYAPS_DATA/L1/blue_3097086.fit $PYAPS_DATA/L1/red_3097085.fit \
    --wlranges 3800,5950 5900,9270 \
    --aladin False
```
**Use case:** Standard two-arm observation (blue + red)

---

### Example 3: Galaxy Survey with Aladin
```bash
python aps_l1_preview.py \
    --infiles $PYAPS_DATA/L1/stackcube.fit \
    --l1_reference $PYAPS_DATA/L1/reference_20240515.fits \
    --targclass GALAXY \
    --aladin True
```
**Use case:** Survey work requiring sky context

---

### Example 4: IFU Observation with Area Selection
```bash
python aps_l1_preview.py \
    --infiles $PYAPS_DATA/L1/LIFU/stackcube_blue.fit $PYAPS_DATA/L1/LIFU/stackcube_red.fit \
    --area 234.666,59.355,157.99,91.50,0.0 \
    --collapse True \
    --aladin False
```
**Use case:** Large IFU field, collapse to single spectrum

---

### Example 5: Multiple Observations from List
```bash
# Create filelist.txt
cat > filelist.txt << EOF
stackcube_3097086.fit stackcube_3097085.fit
stackcube_3097088.fit stackcube_3097087.fit
EOF

python aps_l1_preview.py \
    --infiles_list filelist.txt \
    --aladin False
```
**Use case:** Batch viewing of multiple observations

---

### Example 6: Joined Arms with Cosmic Ray Removal
```bash
python aps_l1_preview.py \
    --infiles blue.fit red.fit \
    --join_arms True \
    --crr True \
    --arms_ratio 1.0,1.0 \
    --aladin False
```
**Use case:** Single continuous spectrum from both arms

---

### Example 7: Filtered by Survey and Target Class
```bash
python aps_l1_preview.py \
    --infiles stackcube.fit \
    --targsrvy WS2022A2-002 \
    --targclass GALAXY,QSO \
    --sens_corr True \
    --aladin False
```
**Use case:** Focus on specific science targets

---

### Example 8: Quality Check Specific Fibers
```bash
python aps_l1_preview.py \
    --infiles observation.fit \
    --aps_ids 100,101,102,103,104 \
    --aladin False
```
**Use case:** Inspect specific fibers for quality assessment

---

### Example 9: With L2 Parameters
```bash
python aps_l1_preview.py \
    --infiles stackcube.fit \
    --l1_reference reference.fits \
    --l2_reference l2_parameters.fits \
    --aladin True
```
**Use case:** View L1 spectra with L2 derived parameters (Teff, logg, etc.)

---

### Example 10: Telluric Masking
```bash
python aps_l1_preview.py \
    --infiles red_arm.fit \
    --tellurics True \
    --wlranges 5900,9270 \
    --aladin False
```
**Use case:** Mask strong telluric absorption bands

---

## Windows Overview

### 1. Target Selection Window
**Purpose:** Interactive RA/DEC map of all fibers

**Features:**
- Each point represents one fiber/spaxel
- Color indicates integrated flux (brighter = more flux)
- Click any fiber to view its spectrum
- Spherical coordinate correction automatically applied (cos DEC)
- Coordinates displayed in degrees

**Symbols:**
- `×` = Fiber not used (`TARGUSE != 'T'`) - larger size
- `⬡` = Active fiber (`TARGUSE == 'T'`) - smaller hexagon

**Controls:**
- **Left Click**: Select fiber and update all windows
- **R key**: Reset Aladin view (if Aladin enabled)
- **Mouse wheel**: Zoom
- **Right drag**: Pan

**Title:** Shows current coordinates in HMS DMS format

---

### 2. Information Window (Spectra + Metadata)
**Purpose:** Display selected fiber's spectrum and properties

**Layout:**
- **Left Panel (70%):** Spectral plots
  - **Top plot**: First arm/setup (typically blue arm)
  - **Bottom plot**: Second arm/setup (typically red arm)
  - **X-axis**: Wavelength (Angstroms)
  - **Y-axis**: Flux (10⁻¹⁸ erg/s/cm²/Å)
  - **Grid**: Enabled for easy reading

- **Right Panel (30%):** Metadata table
  - Target name (TARGNAME)
  - Coordinates (TARGRA, TARGDEC)
  - Survey info (TARGSRVY)
  - Target class (TARGCLASS)
  - S/N ratio (SNR)
  - Fiber status (FIB_STATUS)
  - And more...
  - **Note**: Complex objects filtered for stability

**Features:**
- Auto-ranging on fiber selection
- Grid lines for easy value reading
- Synchronized updates across all windows

---

### 3. Inverse Variance Window
**Purpose:** Show measurement uncertainty/quality

**Layout:**
- Same structure as spectral window
- **Top**: IVAR for first arm
- **Bottom**: IVAR for second arm
- Higher IVAR = better signal-to-noise

**Useful for identifying:**
- Bad pixels (IVAR = 0 or very low)
- Sky line regions (dips in IVAR)
- Telluric absorption (reduced IVAR)
- Gaps between detectors (IVAR = 0)
- Cosmic rays (if not removed)

**Interpretation:**
- **High IVAR** (>100): Excellent quality
- **Medium IVAR** (10-100): Good quality
- **Low IVAR** (<10): Poor quality
- **Zero IVAR**: Masked or bad pixel

---

### 4. FWHM Overview Window
**Purpose:** Compare FWHM across all fibers simultaneously

**What You See:**

| Element | Appearance | Meaning |
|---------|------------|---------|
| Gray lines | Semi-transparent | Individual fiber FWHM profiles (all fibers) |
| Black line | Bold, solid | Global FWHM (average across all fibers) |
| Colored line | Bold, colored | Currently selected fiber (highlighted) |

**Info Panel Shows:**
- **Mode**: FAST (global fit) vs INDIVIDUAL (per-fiber interpolation)
- **Fibers**: Total number of fibers
- **Setups**: Number of arms/setups
- **Selected Fiber**: Currently highlighted fiber ID

**Plot Features:**
- Two panels (one per arm/setup)
- X-axis: Wavelength (Å)
- Y-axis: FWHM (Å)
- Legend showing Global and selected fiber
- Grid for easy reading

**Use Cases:**
- Identify fibers with unusual FWHM
- Verify global FWHM is reasonable
- Check if FWHM interpolation worked correctly
- Spot systematic variations across field
- Compare selected fiber to global

**What to Look For:**
- ✅ Selected fiber close to global = Good
- ⚠️ Selected fiber far from global = Investigate
- ⚠️ Large scatter in gray lines = Possible issues
- ⚠️ Gaps or jumps = Interpolation problems

---

### 5. FWHM Detailed Inspection Window
**Purpose:** Deep dive into single fiber's FWHM with raw arc line data

**What You See:**

| Symbol | Color | Size | Meaning |
|--------|-------|------|---------|
| ● (filled circle) | Blue | 8px | Arc line used in fit (weight > 0) |
| ✕ (X) | Gray | 10px | Arc line excluded by FWHM code (weight = 0) |
| ○ (hollow circle) | Orange outline | 12px | Used but large residual (>0.15Å deviation) |
| ― (solid line) | Blue bold | 3px | Interpolated FWHM function |
| - - (dashed line) | Gray | 2px | Global FWHM for reference |

**Info Panel Shows:**
Per-arm statistics table:
- **Arm name** (e.g., BLUEL11, REDL11)
- **Fiber FWHM**: Mean FWHM for this fiber
- **Global FWHM**: Mean global FWHM
- **Δ (Delta)**: Difference (Fiber - Global)

**Data Source:**
Raw arc line measurements from `meta['fwhm']['fiber_file_fits']`:
- `wavelengths`: Arc line positions
- `fwhm`: Measured FWHM values
- `weights`: Fit weights (0 = excluded)
- `residuals`: Fit quality (measured - fitted)

**Interpretation:**

**Good Fiber:**
```
● ● ● ● ● ● ●     Most points blue (used)
    ○             Few orange (large residuals)
                  No gray X's (all included)
Δ = -0.02 Å       Small difference from global
```

**Problematic Fiber:**
```
● ● × × ● ○ ×     Many gray X's (excluded)
  ○   ○   ○       Many orange (poor fit)
Δ = +0.25 Å       Large difference from global
```

**Use Cases:**
- Check which arc lines were actually used
- Identify problematic wavelength regions
- Verify FWHM fit quality
- Find outlier arc lines
- Diagnose extraction issues
- Compare to global FWHM behavior

**What to Look For:**
- **Clustered exclusions**: Wavelength-dependent issues
- **Systematic residuals**: Fit model problems
- **Edge effects**: Issues at wavelength extremes
- **Large Δ values**: Fiber significantly different from global

---

## FWHM Diagnostics

### Understanding FWHM

**What is FWHM?**
- Full Width at Half Maximum of the Point Spread Function (PSF)
- Measures the spectral resolution
- Determines how well emission/absorption lines are resolved
- Typically 1.5-2.0 Å for WEAVE observations
- Varies with wavelength and fiber position

**Why is FWHM Important?**
- Critical for accurate spectral extraction
- Affects line width measurements
- Impacts velocity dispersion measurements
- Essential for deconvolution and line fitting
- Quality indicator for wavelength calibration

**How is FWHM Measured?**
1. Arc lines are identified in calibration frames
2. FWHM measured for each arc line
3. Linear fit applied: FWHM(λ) = a + b×λ
4. Individual fiber fits or global fit used
5. Interpolation function created for extraction

---

### FWHM Diagnostic Workflow

#### Step 1: Initial Assessment (Overview Window)
1. Select fiber in RA/DEC map
2. Check FWHM Overview window
3. **Ask**: Is the colored line close to the black global line?
   - **Yes**: FWHM is normal, proceed with analysis
   - **No**: Investigate further in detailed view

#### Step 2: Detailed Investigation (Detailed Window)
1. Open FWHM Detailed Inspection window
2. Examine raw arc line points:
   - **Blue circles (●)**: Good measurements, used in fit
   - **Gray X's (✕)**: Excluded by FWHM code (weight=0)
   - **Orange circles (○)**: Used but poor fit (>0.15Å residual)

#### Step 3: Pattern Recognition
Look for:
- **Clustered exclusions**: Problems at specific wavelengths
- **Systematic deviations**: Linear fit may be inadequate
- **Edge effects**: Issues at wavelength boundaries
- **Random scatter**: Normal measurement noise

#### Step 4: Comparison
1. Select neighboring fibers
2. Compare patterns
3. **Ask**: Is this fiber unique or systematic?
   - **Unique**: Fiber-specific issue (damaged fiber?)
   - **Systematic**: Field-wide issue (calibration problem?)

#### Step 5: Decision
Based on findings:
- **Good FWHM**: Proceed with normal analysis
- **Minor issues**: Use with caution, note in analysis
- **Major issues**: Consider excluding fiber or re-reducing

---

### FWHM Quality Indicators

#### Excellent Quality
```
✅ Most points are blue (●)
✅ Very few excluded (✕)
✅ No large residuals (○)
✅ |Δ| < 0.05 Å from global
✅ Smooth interpolated line
✅ Even distribution of arc lines
```

#### Good Quality
```
✅ Majority of points blue (●)
⚠️ Few excluded points (✕)
⚠️ 1-2 large residuals (○)
✅ |Δ| < 0.10 Å from global
✅ Smooth interpolated line
```

#### Acceptable Quality
```
⚠️ Many blue points but some gaps
⚠️ Several excluded points (✕)
⚠️ Multiple large residuals (○)
⚠️ 0.10 < |Δ| < 0.20 Å from global
⚠️ Minor waviness in interpolation
```

#### Poor Quality
```
❌ Many excluded points (✕)
❌ Numerous large residuals (○)
❌ |Δ| > 0.20 Å from global
❌ Irregular interpolated line
❌ Clustered exclusions
```

---

### Common FWHM Issues and Solutions

#### Issue 1: Many Excluded Arc Lines (Gray ✕)

**Symptoms:**
- Numerous gray X's in detailed view
- Sparse blue circles

**Possible Causes:**
- Poor arc frame quality
- Cosmic rays in arc frame
- Saturation of arc lines
- Wrong wavelength solution

**Solutions:**
1. Check arc frame quality
2. Verify wavelength calibration
3. Consider using global FWHM instead
4. Re-reduce with different arc frame

---

#### Issue 2: Large Residuals (Orange ○)

**Symptoms:**
- Many orange hollow circles
- Large differences from fit

**Possible Causes:**
- Non-linear FWHM variation
- Blended arc lines
- Poor S/N arc lines
- Inadequate linear model

**Solutions:**
1. Check if residuals are systematic
2. Consider higher-order FWHM fit
3. Exclude problematic wavelength regions
4. Use more arc lines if available

---

#### Issue 3: Large Δ from Global

**Symptoms:**
- Fiber FWHM significantly different from global
- Δ > 0.20 Å in info panel

**Possible Causes:**
- Damaged or misaligned fiber
- Edge of focal plane effects
- IFU vs MOS differences
- Interpolation failure

**Solutions:**
1. Compare to neighboring fibers
2. Check fiber status in metadata
3. Consider excluding from analysis
4. Use global FWHM if appropriate

---

#### Issue 4: Wavelength-Dependent Exclusions

**Symptoms:**
- Excluded points cluster at specific wavelengths
- Systematic pattern in ✕ positions

**Possible Causes:**
- Detector gaps or defects
- Strong sky lines contamination
- Wavelength-dependent systematics
- Limited arc line coverage

**Solutions:**
1. Check detector map
2. Verify arc line catalog
3. Add more arc lines if possible
4. Accept gaps in coverage

---

### FWHM Mode: FAST vs INDIVIDUAL

#### FAST Mode
**Characteristics:**
- All fibers use the same global FWHM function
- Single fit to all fibers combined
- Faster computation
- More robust for low S/N data

**When Used:**
- High-quality uniform illumination
- Similar fiber performance
- Low S/N individual fibers
- Quick-look reductions

**Advantages:**
- Very fast
- Robust to outliers
- Consistent across field

**Disadvantages:**
- Ignores fiber-to-fiber variations
- May not capture edge effects
- Less accurate for non-uniform fields

---

#### INDIVIDUAL Mode
**Characteristics:**
- Each fiber has its own FWHM function
- Separate fit per fiber
- Slower computation
- More accurate per-fiber

**When Used:**
- High S/N arc frames
- Significant fiber variations expected
- IFU observations
- Final science reductions

**Advantages:**
- Accounts for fiber differences
- More accurate extraction
- Better for wide fields

**Disadvantages:**
- Slower computation
- Can fail for low S/N
- More sensitive to outliers

---

### Interpreting FWHM Info Panel

Example from tool:
```
Detailed FWHM Inspection - Fiber 389

BLUEL11:  Fiber FWHM: 1.7234 Å  Global FWHM: 1.7383 Å  Δ: -0.0149 Å
```

**Interpretation:**
- **Fiber 389**: Currently selected fiber
- **BLUEL11**: Blue arm, low resolution, fiber 11
- **Fiber FWHM: 1.7234 Å**: Mean FWHM for this specific fiber
- **Global FWHM: 1.7383 Å**: Mean FWHM across all fibers
- **Δ: -0.0149 Å**: This fiber's FWHM is 0.015 Å narrower than global

**Assessment:**
- Δ = -0.015 Å is excellent (< 0.05 Å)
- Fiber is performing normally
- No concerns for this fiber

---

## Quick Reference Card

### 🚀 Quick Start Commands
```bash
# Simplest usage
python aps_l1_preview.py --infiles data.fit --aladin False

# Two arms
python aps_l1_preview.py --infiles blue.fit red.fit --wlranges 3800,5950 5900,9270

# With Aladin
python aps_l1_preview.py --infiles data.fit --l1_reference ref.fits --aladin True
```

### ⌨️ Keyboard & Mouse Controls
| Action | Control |
|--------|---------|
| Select fiber | Left click on RA/DEC map |
| Reset Aladin view | R key |
| Exit application | Close any window |
| Zoom | Mouse wheel |
| Pan | Right mouse drag |

### 📊 FWHM Symbol Legend

#### Overview Window
| Symbol | Meaning |
|--------|---------|
| Gray transparent lines | All fiber FWHM profiles |
| Black bold line | Global FWHM (average) |
| Colored bold line | Selected fiber (highlighted) |

#### Detailed Window
| Symbol | Color | Meaning |
|--------|-------|---------|
| ● | Blue | Arc line used in fit (weight > 0) |
| ✕ | Gray | Excluded by FWHM code (weight = 0) |
| ○ | Orange | Large residual but used (>0.15Å) |
| ― | Blue bold | Interpolated FWHM function |
| - - | Gray dashed | Global FWHM reference |

### 🎯 Most Common Arguments
```bash
# Input
--infiles file1.fit [file2.fit ...]
--infiles_list filelist.txt

# Filtering
--targclass GALAXY,QSO,STAR
--aps_ids 100,101,102
--area RA,DEC,W,H,PA

# Wavelength
--wlranges MIN,MAX [MIN,MAX ...]

# Processing
--join_arms True|False
--collapse True|False
--crr True|False
--sens_corr True|False

# Visualization
--aladin True|False
--l1_reference ref.fits
--l2_reference l2cat.fits
```

### 🔧 Quick Troubleshooting
| Problem | Quick Fix |
|---------|-----------|
| Won't start | `pip install pyqtgraph PyQt5` |
| Won't exit | Fixed in v3.1 |
| Vertical stripes | Fixed in v3.1 |
| Table error | Fixed in v3.1 |
| Slow startup | Use `--aladin False` |
| Large dataset | Use `--aps_ids` to filter |

### 💡 Pro Tips
1. **Check FWHM overview first** before detailed analysis
2. **Use --aps_ids** to preview specific fibers quickly
3. **Disable Aladin** for faster startup
4. **Compare neighbors** to verify systematic issues
5. **Filter by --targclass** to focus on science targets
6. **Look for patterns** in excluded arc lines

### 📖 Getting Help
```bash
# Command-line help
python aps_l1_preview.py --help

# Version info
python aps_l1_preview.py --version  # (if implemented)

# Check dependencies
pip list | grep -E 'pyqtgraph|PyQt5|numpy|astropy'
```

---

## Tips & Tricks

### Performance Optimization

#### For Large Datasets (>500 fibers)
```bash
# Preview a subset first
python aps_l1_preview.py --infiles large.fit --aps_ids 1,2,3,4,5

# Filter by target class
python aps_l1_preview.py --infiles large.fit --targclass GALAXY

# Disable Aladin for faster startup
python aps_l1_preview.py --infiles large.fit --aladin False
```

#### For IFU Fields
```bash
# Collapse to single spectrum to reduce memory
python aps_l1_preview.py --infiles ifu.fit --collapse True --area RA,DEC,W,H,PA
```

---

### Visualization Tips

#### Finding Problematic Fibers
1. Load data without filtering
2. Scan FWHM overview for outliers (far from black global line)
3. Click suspicious fibers
4. Examine detailed FWHM view
5. Note fiber IDs for exclusion

#### Comparing Similar Targets
```bash
# Select specific target class
python aps_l1_preview.py --infiles data.fit --targclass GALAXY

# Click through targets of same type
# Compare spectral features, FWHM, S/N
```

#### Edge Fiber Investigation
- Click fibers at field edges
- Check if FWHM degrades
- Verify spherical correction working (no stripes)
- Compare to field center

---

### Workflow Recommendations

#### Workflow 1: Quick Quality Check
```
1. Load data: --infiles obs.fit --aladin False
2. Scan FWHM overview window
3. Look for obvious outliers
4. Click 5-10 random fibers
5. Check spectra look reasonable
6. Done in <5 minutes
```

#### Workflow 2: Detailed Target Inspection
```
1. Load with filtering: --targclass GALAXY
2. Load L2 reference: --l2_reference l2cat.fits
3. Click target of interest
4. Examine spectrum quality
5. Check IVAR for problematic regions
6. Check FWHM detailed view
7. Note metadata from table
8. Export fiber ID for further analysis
```

#### Workflow 3: FWHM Quality Assessment
```
1. Load data with Aladin: --aladin True
2. Open FWHM overview window
3. Identify fibers far from global
4. Click each suspicious fiber
5. Examine detailed FWHM view:
   - How many excluded arc lines?
   - Wavelength-dependent patterns?
   - Large residuals?
6. Check neighboring fibers
7. Decide: systematic or fiber-specific?
8. Document findings for re-reduction
```

#### Workflow 4: Batch Processing
```bash
# Create script: process_all.sh
#!/bin/bash
for file in $PYAPS_DATA/L1/*.fit; do
    echo "Processing $file"
    python aps_l1_preview.py --infiles $file --aladin False
    # Add screenshot or logging here
done
```

---

### Advanced Usage Tips

#### Custom Fiber Selection
Modify code to log selected fibers:
```python
# In onClick method, add:
with open('selected_fibers.txt', 'a') as f:
    f.write(f"{self.aps_id_arr[xy]}\n")
```

#### Automated Screening
Create a script to check FWHM quality:
```python
# Load data programmatically
# Check Δ from global for all fibers
# Flag fibers with |Δ| > 0.15 Å
# Generate report
```

#### Export for Further Analysis
After identifying interesting fibers:
1. Note fiber IDs from table
2. Create fiber list file
3. Re-reduce with only those fibers
4. Or: extract spectra for external tools

---

### Aladin Integration Tips

#### Starting Aladin
1. Launch Aladin Desktop first
2. Enable SAMP: Interop → Connect SAMP Hub
3. Load survey image (PanSTARRS, SDSS, etc.)
4. Run aps_l1_preview with --aladin True

#### Using Aladin Effectively
- Press **R** to reset view if lost
- Click fiber in tool → Aladin centers on target
- Use Aladin to identify field objects
- Load catalog overlays for context
- Compare morphology to spectral type

#### Troubleshooting Aladin
- **Not connecting**: Check SAMP hub running
- **Slow response**: Disable for faster work
- **Wrong position**: Verify L1 reference catalog

---

## Troubleshooting

### Installation Issues

#### Error: "No module named 'pyqtgraph'"
```bash
# Solution
pip install pyqtgraph PyQt5

# If still fails, try
pip install --upgrade pyqtgraph PyQt5
```

#### Error: "No module named 'astropy'"
```bash
# Solution
pip install numpy astropy matplotlib

# Check versions
python -c "import astropy; print(astropy.__version__)"
```

#### Platform-Specific Issues

**macOS Qt Issues:**
```bash
brew install qt5
export PATH="/usr/local/opt/qt5/bin:$PATH"
pip install PyQt5
```

**Linux Display Issues:**
```bash
# If running remotely, enable X11 forwarding
ssh -X user@host
# Or use VNC for full desktop
```

---

### Application Issues

#### Application Won't Start

**Problem:** Crashes immediately on startup

**Checks:**
```bash
# Test imports
python -c "import PyQt5.QtWidgets; print('Qt OK')"
python -c "import pyqtgraph; print('pyqtgraph OK')"

# Check Python version
python --version  # Should be 3.8+
```

**Solutions:**
1. Reinstall packages: `pip install --force-reinstall pyqtgraph PyQt5`
2. Try different Python version
3. Check for conflicting Qt installations

---

#### Application Won't Exit

**Problem:** Terminal still busy after closing windows

**Status:** **FIXED in v3.1**

**If issue persists:**
```bash
# Last resort: force quit
pkill -9 python

# Or find process ID
ps aux | grep aps_l1_preview
kill -9 <PID>
```

**Long-term solution:**
- Verify you're using v3.1 or later
- Check closeEvent implementation

---

#### Error: "Aladin is not running or SAMP is disconnected"

**Problem:** Can't connect to Aladin

**Solutions:**
1. **Start Aladin first**, then run tool
2. **Enable SAMP** in Aladin: Interop menu → Connect
3. **Or disable Aladin**: `--aladin False`
4. **Check SAMP hub**: Look for SAMP icon in system tray

---

### Display Issues

#### Vertical Stripes in RA/DEC Map

**Problem:** Fibers appear in vertical bands

**Status:** **FIXED in v3.1** with spherical correction

**If still occurs:**
- Verify using latest version
- Check coordinates are in degrees (not radians)
- Most noticeable at high declination
- Report with sample data

---

#### Wrong Fiber Selected When Clicking

**Problem:** Click on fiber, different fiber selected

**Status:** **FIXED in v3.1** with cos(DEC) correction

**If still occurs:**
- Click very close to fiber center
- Zoom in for better precision
- Most noticeable at field edges
- Check spherical correction applied

---

### Data Issues

#### Table Display Error

**Error:** `AttributeError: 'int' object has no attribute '__iter__'`

**Status:** **FIXED in v3.1** with filtered metadata

**If still occurs:**
- Check metadata structure in your files
- May have unusual data types
- Report issue with sample file

**Workaround:**
```python
# Metadata will show "ERROR: Could not display metadata"
# Core functionality still works
```

---

#### FWHM Window Shows No Data

**Problem:** FWHM windows are empty

**Possible Causes:**
1. FWHM interpolation failed during reduction
2. Data missing `meta['fwhm']` structure
3. Using older L1 format
4. FWHM mode set to skip interpolation

**Diagnosis:**
```python
from astropy.io import fits
hdu = fits.open('stackcube.fit')
# Check extensions
print(hdu.info())
# Look for FWHM-related extensions or metadata
```

**Solutions:**
1. Re-reduce with FWHM interpolation enabled
2. Check reduction logs for FWHM errors
3. Use newer PyAPS version
4. Contact data support if problem persists

---

#### Spectra Look Very Noisy

**Problem:** Spectra are extremely noisy

**Normal or Problem?**
- **Normal** for low S/N targets (SNR < 5)
- **Normal** for faint targets
- **Problem** if bright targets are noisy

**Checks:**
1. Look at IVAR window (should be non-zero)
2. Check SNR in metadata table
3. Try `--sens_corr False` if correction wrong
4. Verify not looking at sky fiber

**Solutions:**
- Expected for faint targets
- Check reduction quality if unexpected
- Verify sky subtraction worked
- Check for cosmic rays (--crr True)

---

#### Missing or Bad Wavelength Ranges

**Problem:** Spectra show wrong wavelength range

**Checks:**
```bash
# Verify wavelength ranges
python aps_l1_preview.py --infiles blue.fit red.fit \
    --wlranges 3800,5950 5900,9270  # Explicitly set
```

**Common mistakes:**
- Wrong order (red before blue)
- Swapped min/max
- Missing ranges for some files
- Using vacuum when data is air (or vice versa)

---

### Performance Issues

#### Very Slow Startup

**Possible causes:**
1. Large number of fibers
2. Aladin connection timeout
3. Large file size
4. Slow disk I/O

**Solutions:**
```bash
# Disable Aladin
--aladin False

# Filter fibers
--aps_ids 1,2,3,4,5

# Filter by target class
--targclass GALAXY

# Use SSD if available
```

---

#### Laggy Response

**Problem:** Clicking fibers is slow

**Solutions:**
1. Close unused windows
2. Reduce number of visible fibers
3. Use faster hardware
4. Close other applications

---

### Workflow Issues

#### Can't Find Specific Target

**Problem:** Know target name, can't locate in map

**Solutions:**
1. Use L2 reference with target info
2. Filter by target class: `--targclass GALAXY`
3. Use area selection if know coordinates
4. Load in Aladin to visualize field
5. Check metadata table after clicking nearby fibers

---

#### Need to Analyze Multiple Observations

**Problem:** Many files to check

**Solution:** Create batch script
```bash
#!/bin/bash
for file in $PYAPS_DATA/L1/*.fit; do
    python aps_l1_preview.py --infiles $file --aladin False &
    # Wait for user to close
    wait
done
```

---

### Getting Additional Help

#### Reporting Bugs

**Include in report:**
1. **Command used** (exact command line)
2. **Error message** (full traceback)
3. **Python version**: `python --version`
4. **Package versions**:
   ```bash
   pip list | grep -E 'pyqtgraph|PyQt5|numpy|astropy'
   ```
5. **Operating system**
6. **Sample data** (if possible - small test case)
7. **Expected behavior** vs actual behavior

#### Where to Get Help

- **Command-line help**: `python aps_l1_preview.py --help`
- **Code documentation**: First 150 lines of .py file
- **This guide**: Complete documentation here
- **PyAPS documentation**: Check PyAPS pipeline docs
- **Contact**: development team email

---

## Advanced Usage

### Programmatic Access

#### Loading Data Programmatically
```python
import sys
sys.path.insert(0, '/path/to/PyAPS')

from PyAPS.aps_utils import APSOB

# Load data
apsob = APSOB(
    ['stackcube.fit'],
    sens_corr=True,
    vacuum=False
)

# Access targets
for targ in apsob._targetlist:
    print(f"Fiber {targ.aps_id}: SNR={targ.meta[0]['SNR']}")
    wave = targ.spectra[0].wave
    flux = targ.spectra[0].flux
    # Analyze...
```

---

### Custom Modifications

#### Adding Custom Filters
```python
# In l1_preview function, after targ_list_master creation
# Add custom filtering:
custom_targ_list = [
    targ for targ in targ_list_master 
    if targ.meta[0]['SNR'] > 10.0  # Only high SNR
]
```

#### Logging Selected Fibers
```python
# In onClick method, add:
import csv
with open('selected_log.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow([
        self.aps_id_arr[xy],
        self.targs[target_id].targra,
        self.targs[target_id].targdec,
        self.targs[target_id].meta[0]['SNR']
    ])
```

#### Automated Quality Checks
```python
# After loading data:
for targ in self.targs:
    # Check FWHM quality
    if hasattr(targ, 'meta') and 'fwhm' in targ.meta[0]:
        delta = abs(targ.meta[0]['fwhm']['average_offset_from_global'])
        if delta > 0.15:
            print(f"WARNING: Fiber {targ.aps_id} has large FWHM offset: {delta:.3f}")
```

---

### Integration with Other Tools

#### Exporting for DS9
```python
# Create region file from selected fibers
with open('fibers.reg', 'w') as f:
    f.write('global color=green\n')
    for fiber_id in selected_fibers:
        targ = self.targs[self.targs_apstoid[fiber_id]]
        f.write(f'circle({targ.targra},{targ.targdec},2")\n')
```

#### Exporting for Topcat
```python
from astropy.table import Table

# Create table of selected fibers
data = {
    'fiber_id': [],
    'ra': [],
    'dec': [],
    'snr': []
}

for fiber_id in selected_fibers:
    targ = self.targs[self.targs_apstoid[fiber_id]]
    data['fiber_id'].append(targ.aps_id)
    data['ra'].append(targ.targra)
    data['dec'].append(targ.targdec)
    data['snr'].append(targ.meta[0]['SNR'])

table = Table(data)
table.write('selected_fibers.fits', overwrite=True)
```

---

## What's New

### Version 3.1 (January 2025) - Current

#### Major Features
✅ **Raw Arc Line Visualization**
- Displays actual arc line measurements in FWHM detailed view
- Shows which points were used vs excluded
- Color-coded by fit quality (blue=good, gray=excluded, orange=poor fit)

✅ **Weight-Based Outlier Detection**
- Uses actual weights from FWHM fitting code
- No more arbitrary thresholds
- Shows exactly what the reduction pipeline did

✅ **Spherical Coordinate Correction**
- Applies cos(DEC) correction for accurate fiber selection
- Eliminates vertical striping in RA/DEC map
- Essential for high-declination fields and wide FOVs

✅ **Proper Application Exit**
- Closing any window now exits cleanly
- Returns to terminal properly
- No more zombie processes

#### Bug Fixes
✅ **Table Display Error** - Fixed iteration error with complex metadata
✅ **Terminal Won't Release** - Added sys.exit(0) for proper termination
✅ **Vertical Stripes** - Applied spherical geometry correction
✅ **"Joined Arms" Misidentification** - Fixed logic to check both join_arms and n_setups

#### Improvements
✅ **Comprehensive Documentation** - 150+ line header, complete user guide
✅ **Filtered Metadata** - Complex objects excluded from table to prevent crashes
✅ **Smaller Table Font** - Reduced to 12pt (40% smaller than original)
✅ **No Terminal Spam** - All FWHM debug output suppressed
✅ **Actual Arm Names** - Shows real arm names (BLUEL11, REDL11) not "Setup 1/2"
✅ **Inline Documentation** - Extensive code comments and function docs

---

### Version 3.0 (January 2025)

#### Features
- FWHM diagnostic windows added
- Overview window showing all fibers
- Detailed inspection window
- Interactive fiber highlighting

---

### Version 2.1 (September 2023)

#### Improvements
- L2 reference import improvements
- Better handling of catalog matching

---

### Version 2.0 (October 2023)

#### Features
- Aladin Desktop integration via SAMP
- SAMP listener for position broadcasts
- Plots in separate windows
- Improved window management

---

### Version 1.0 (September 2022)

#### Initial Release
- Basic L1 data visualization
- Interactive fiber selection
- Spectral plotting
- Metadata display

---

## Technical Details

### Architecture

#### Core Components
1. **L1_preview_interactive** - Main GUI class
2. **APSOB** - Data loading and management (from PyAPS)
3. **PyQtGraph** - Fast plotting backend
4. **Qt5** - GUI framework
5. **Aladin/SAMP** - Sky visualization (optional)

#### Data Flow
```
L1 FITS files
    ↓
APSOB loading
    ↓
Target list creation
    ↓
GUI initialization
    ↓
User interaction
    ↓
Dynamic plot updates
```

---

### Key Algorithms

#### Spherical Coordinate Correction
```python
def closest_node(self, node, nodes):
    """Apply cos(DEC) correction for accurate angular distances"""
    node_ra, node_dec = node
    cos_dec = np.cos(np.radians(node_dec))
    
    # Project RA differences onto sky
    delta_ra = (nodes[:, 0] - node_ra) * cos_dec
    delta_dec = nodes[:, 1] - node_dec
    
    # Angular separation squared
    dist_2 = delta_ra**2 + delta_dec**2
    return np.argmin(dist_2)
```

**Why needed:**
- At DEC = 40°, cos(DEC) ≈ 0.766
- Without correction: 1° RA = 1° angular distance (WRONG)
- With correction: 1° RA = 0.766° angular distance (CORRECT)
- 23% error causes vertical stripes without correction

---

#### FWHM Outlier Detection
```python
# Based on actual weights from FWHM code
weights = file_data.get('weights', None)
if weights is not None:
    excluded_mask = (weights == 0)  # FWHM code set weight=0
    good_mask = ~excluded_mask
    
    # Additional check for large residuals
    if residuals is not None:
        large_residual_mask = (np.abs(residuals) > 0.15) & good_mask
```

**Why better:**
- Uses actual FWHM fitting decisions
- Not arbitrary threshold
- Shows what pipeline did
- More informative for diagnostics

---

#### Metadata Filtering
```python
# Filter complex objects for table display
filtered_meta = {}
for key, value in meta_dict.items():
    if isinstance(value, (dict, type(lambda: None))):
        continue  # Skip complex objects
    elif isinstance(value, (list, tuple, np.ndarray)):
        if len(value) > 5:
            filtered_meta[key] = f"Array[{len(value)}]"
        else:
            filtered_meta[key] = str(value)
    else:
        filtered_meta[key] = value  # Keep simple values
```

**Why needed:**
- PyQtGraph TableWidget can't iterate over ints
- Complex objects cause crashes
- Nested dicts and functions must be excluded
- Long arrays shown as "Array[N]" for readability

---

### File Structure

#### Input Files
- **L1 stackcube/stack**: Reduced spectral data (FITS)
- **L1 reference catalog**: Fiber positions and metadata (FITS table)
- **L2 reference catalog**: Derived parameters (FITS table)
- **File list**: Text file with paths (one set per line)

#### Metadata Structure
```python
meta[0] = {
    'APS_ID': int,           # Fiber ID
    'NSPEC': int,            # Spectrum number
    'TARGRA': float,         # RA (degrees)
    'TARGDEC': float,        # DEC (degrees)
    'SNR': float,            # Signal-to-noise ratio
    'SETUP': str,            # Arm name (e.g., 'BLUEL11')
    'fwhm': {                # FWHM data structure
        'interpolate_function': function,
        'fiber_file_fits': {
            '/path/to/arc.fit': {
                'wavelengths': array,   # Arc line positions
                'fwhm': array,          # Measured FWHM
                'weights': array,       # Fit weights (0=excluded)
                'residuals': array,     # Fit residuals
                'fitted_fwhm': array    # Fitted values
            }
        }
    },
    # ... many more fields
}
```

---

### Performance Considerations

#### Memory Usage
- **Small datasets** (<100 fibers): ~100 MB
- **Medium datasets** (100-500 fibers): ~500 MB
- **Large datasets** (>500 fibers): >1 GB

**Optimization:**
- Use `--aps_ids` to load subset
- Filter by `--targclass`
- Disable Aladin for faster startup

#### Rendering Speed
- PyQtGraph is very fast for line plots
- Scatter plots with >1000 points may slow down
- FWHM overview with >500 fibers can be laggy
- Solution: Increase transparency, reduce line width

---

### Platform-Specific Notes

#### Linux
- Best performance
- Full Aladin integration works seamlessly
- Tested on Ubuntu 20.04+

#### macOS
- Generally works well
- May need Qt5 from Homebrew
- Aladin integration sometimes requires manual SAMP setup

#### Windows
- Full functionality available
- May need to run as administrator for some operations
- Aladin SAMP can be tricky to configure

---

## Contributing

### How to Contribute

We welcome contributions! Here's how:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Make your changes**
4. **Test thoroughly**
5. **Submit a pull request**

### Development Setup

```bash
# Clone repository
git clone https://github.com/yourorg/aps_l1_preview.git
cd aps_l1_preview

# Install in development mode
pip install -e .

# Run tests (if available)
python -m pytest tests/
```

### Code Style

- Follow PEP 8
- Use descriptive variable names
- Add docstrings to functions
- Comment complex logic
- Keep functions focused and short

### Testing

Before submitting:
- [ ] Test with single-arm data
- [ ] Test with dual-arm data
- [ ] Test with large datasets
- [ ] Test FWHM diagnostics
- [ ] Test Aladin integration
- [ ] Verify no new errors in console
- [ ] Check application exits cleanly

---

### Areas for Contribution

**High Priority:**
- Support for >2 arms/setups
- Performance optimization for large datasets
- Automated quality metrics
- Export functionality (spectra, fiber lists)

**Medium Priority:**
- Additional visualization options
- Custom colormap support
- More sophisticated FWHM analysis
- Batch processing tools

**Low Priority:**
- GUI improvements
- Additional keyboard shortcuts
- Themes/styling options

---

### Contact Development Team

**Lead Developer:** Alireza Molaeinezhad  
**Email:** [Contact via CASU]  
**Institution:** Cambridge Astronomical Survey Unit

---

## License

MIT License

Copyright (c) 2025 Cambridge Astronomical Survey Unit

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{aps_l1_preview,
  author = {Molaeinezhad, Alireza and Murphy, David},
  title = {APS L1 Preview: Interactive WEAVE L1 Data Visualization},
  year = {2025},
  version = {3.1},
  publisher = {Cambridge Astronomical Survey Unit},
  url = {https://github.com/yourorg/aps_l1_preview}
}
```

Also cite the WEAVE survey papers and PyAPS pipeline documentation.

---

## Acknowledgments

- **WEAVE Team** at Isaac Newton Group
- **CASU/IOA** for development support
- **PyAPS Contributors** for pipeline infrastructure
- **PyQtGraph** and **Qt** developers for excellent frameworks
- **Aladin Team** for sky visualization tools

---

## Related Projects

- **[PyAPS](https://github.com/yourorg/PyAPS)** - WEAVE data reduction pipeline
- **[WEAVE](https://www.ing.iac.es/weave/)** - Multi-object spectrograph at WHT
- **[Aladin](https://aladin.u-strasbg.fr/)** - Interactive sky atlas
- **[CASU](https://www.ast.cam.ac.uk/ioa/research/casu)** - Cambridge Astronomical Survey Unit

---

## Changelog

See [What's New](#whats-new) section above for version history.

---

**Last Updated:** January 2025  
**Version:** 3.1  
**Status:** Production Ready ✅

---

**⭐ If you find this tool useful, please star the repository!**

**📧 Questions? Contact the development team.**

**🐛 Found a bug? Please report it on GitHub Issues.**

**📖 Want to contribute? See the Contributing section above.**

---

*This documentation is comprehensive but if you have questions not answered here, please contact the development team or open an issue on GitHub.*
