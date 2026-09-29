# APS MapViewer v3.0 - Complete Documentation

> **⚠️ Outdated (2026-08-03, superseded further 2026-08-14):** `aps_Mapviewer.py`
> was first rewritten as its own standalone Dash web app (the PyQt5/matplotlib
> desktop GUI described below — menus, `--mode` argument, themes, dead
> SFH/GANDALF/DIAGNOSTIC paths — no longer exists), then unified into
> `aps_explorer.py` as `aps_IFUviewer.py`, the single entry point for L1 *and* L2
> data — imported purely as a library now, with no `Dash()` app or CLI of its own
> left. See **[doc/aps_explorer.md](aps_explorer.md)** for current usage and the
> complete feature set. The Gal-mode (RVS/FERRE stellar) support and "Bin & Spaxel
> Data" panel mentioned below both carried over. This document is kept for
> historical reference and
> has not been updated to match.

**Enhanced Interactive IFU Map Visualization Tool for WEAVE Spectroscopic Data**

[![Version](https://img.shields.io/badge/version-3.0--Enhanced-blue.svg)]()
[![Python](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-orange.svg)](LICENSE)

**Author:** Alireza Molaeinezhad (APS Team, IOA, Cambridge, UK)
**Email:** amolaei_at_st.cam.ac.uk
**GitHub:** https://github.com/amolaeinezhad/PyAPS
**Institution:** Institute of Astronomy, University of Cambridge

---

## Table of Contents
1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Installation](#installation)
4. [Command-Line Arguments](#command-line-arguments)
5. [Usage Examples](#usage-examples)
6. [Display Modes](#display-modes)
7. [Interface Guide](#interface-guide)
8. [Data Structure](#data-structure)
9. [Analysis Capabilities](#analysis-capabilities)
10. [Keyboard Shortcuts](#keyboard-shortcuts)
11. [Tips & Tricks](#tips--tricks)
12. [Troubleshooting](#troubleshooting)
13. [Advanced Usage](#advanced-usage)
14. [Technical Details](#technical-details)
15. [Contributing](#contributing)

---

## Overview

### What is APS MapViewer?

**APS MapViewer** is an advanced interactive visualization tool for exploring 2D maps from WEAVE IFU (Integral Field Unit) spectroscopic data. It provides comprehensive visualization and analysis capabilities for:

- **Stellar kinematics** (velocity, dispersion) from pPXF analysis
- **emi kinematics** (emission line velocities, dispersions) from EMIPPXF
- **emi flux distributions** (emission line fluxes, equivalent widths)
- **Line strength indices** (absorption features)
- **Star formation history** parameters
- **Custom derived maps** from L2 analysis

### Key Features at a Glance

🗺️ **Interactive Maps**
- Real-time 2D visualization of IFU data
- Voronoi binned or spaxel-by-spaxel display
- Flexible colormap options
- Interactive point selection

📊 **Advanced Analysis**
- Multiple display modes (ALL, PPXF, EMIPPXF, EMI, LS, SFH)
- Per-emission-line emi kinematics
- Component separation for multi-component fits
- Tie group visualization
- Error/uncertainty maps

🎨 **Customization**
- Multiple themes (default, dark, high_contrast)
- Adjustable color scales and limits
- Contour overlays
- Export to multiple formats (PNG, PDF, SVG)

⚡ **Performance**
- Fast rendering with optimized backends
- Auto, fast, or quality performance modes
- Progress indicators for long operations
- Efficient memory management

🔍 **Diagnostics**
- Debug mode with detailed logging
- Data validation checks
- Quality assessment tools
- Comprehensive error reporting

---

## Quick Start

### Minimal Example
```bash
# View all available maps for a dataset
python aps_Mapviewer.py \
    --outpath /path/to/PyAPS_results/20221025/232/ \
    --headname stackcube_2963103__stackcube_2963102_RUN2_P0000 \
    --mode ALL
```

### emi Kinematics
```bash
# Focus on emi emission lines
python aps_Mapviewer.py \
    --outpath /path/to/results/ \
    --headname observation_P0001 \
    --mode EMI_DETAILED \
    --theme dark
```

### Stellar Kinematics
```bash
# View stellar velocity and dispersion
python aps_Mapviewer.py \
    --outpath /path/to/results/ \
    --headname observation_P0001 \
    --mode PPXF
```

### Interactive Mode
```bash
# Launch without specifying files (will prompt for selection)
python aps_Mapviewer.py
```

---

## Installation

### Prerequisites

#### Required Packages
```bash
# Core dependencies
pip install numpy astropy matplotlib PyQt5

# Additional recommended packages
pip install scipy scikit-image colorama
```

**Minimum Versions:**
- Python 3.8+
- numpy 1.18+
- astropy 4.0+
- matplotlib 3.0+
- PyQt5 5.12+

#### Optional Packages
```bash
# For enhanced functionality
pip install pillow  # Advanced image export
pip install tqdm    # Progress bars
```

### Installation Steps

1. **Clone/Download PyAPS:**
```bash
git clone https://github.com/amolaeinezhad/PyAPS.git
cd PyAPS
```

2. **Verify Dependencies:**
```bash
python -c "import numpy, astropy, matplotlib, PyQt5; print('All dependencies OK')"
```

3. **Test MapViewer:**
```bash
python aps_Mapviewer.py --help
```

### Directory Structure

The tool expects specific files in your data directory:
```
PyAPS_results/20221025/232/
├── stackcube_*_RUN2_P0000_table.fits       # Main data table
├── stackcube_*_RUN2_P0000_VorBinInfo.fits  # Voronoi binning info
├── stackcube_*_RUN2_P0000_ppxf_*.fits      # pPXF results (optional)
├── stackcube_*_RUN2_P0000_emippxf_*.fits   # EMIPPXF results (optional)
├── stackcube_*_RUN2_P0000_ls_*.fits        # Line strength results (optional)
└── stackcube_*_RUN2_P0000_sfh_*.fits       # SFH results (optional)
```

---

## Command-Line Arguments

### Basic Arguments (Required/Common)

```bash
# Output directory containing results
--outpath PATH
  --outpath /data/PyAPS_results/20221025/232/

# Filename prefix (headname)
--headname PREFIX
  --headname stackcube_2963103__stackcube_2963102_RUN2_P0000

# Display mode
--mode MODE
  --mode ALL          # Show all available maps
  --mode PPXF         # Stellar kinematics only
  --mode EMIPPXF      # emi kinematics only
  --mode EMI_DETAILED # Per-line emi analysis
  --mode LS           # Line strength indices
  --mode SFH          # Star formation history

  Default: ALL
```

### Display Options

```bash
# Visual theme
--theme THEME
  --theme default        # Standard light theme
  --theme dark          # Dark mode
  --theme high_contrast # High contrast for presentations

  Default: default

# Contour offset level
--contour_offset FLOAT
  --contour_offset 0.2   # 20% of peak for contours
  --contour_offset 0.1   # 10% of peak (more contours)
  --contour_offset 0.5   # 50% of peak (fewer contours)

  Default: 0.2

# Window size (percentage of screen)
--max_size PERCENT
  --max_size 80         # 80% of screen
  --max_size 100        # Full screen
  --max_size 50         # Half screen

  Default: 90
  Range: 50-100
```

### Performance & Quality

```bash
# Performance mode
--performance MODE
  --performance auto    # Automatic optimization
  --performance fast    # Prioritize speed
  --performance quality # Prioritize visual quality

  Default: auto

# Data validation
--validation BOOL
  --validation true     # Enable validation checks
  --validation false    # Skip validation (faster)

  Default: true
```

### Analysis Parameters

```bash
# Amplitude-over-Noise threshold
--aon_threshold FLOAT
  --aon_threshold 3.0   # Show data with AoN > 3
  --aon_threshold 5.0   # Stricter threshold
  --aon_threshold 2.0   # More permissive

  Default: 4.0

# Export format for saved figures
--export_format FORMAT
  --export_format png   # PNG images
  --export_format pdf   # Vector PDF
  --export_format svg   # SVG vector graphics

  Default: png
```

### Debugging & Development

```bash
# Debug mode
--debug
  Enable detailed diagnostic output
  Shows performance metrics
  Enables additional validation
  Provides verbose error messages

# No argument needed, just add the flag
```

### Complete Example
```bash
python aps_Mapviewer.py \
    --outpath /data/PyAPS_results/20221025/232/ \
    --headname stackcube_2963103__stackcube_2963102_RUN2_P0000 \
    --mode EMI_DETAILED \
    --theme dark \
    --contour_offset 0.1 \
    --max_size 90 \
    --performance quality \
    --aon_threshold 3.0 \
    --export_format pdf \
    --debug
```

---

## Usage Examples

### Example 1: Quick Exploration (All Maps)
```bash
python aps_Mapviewer.py \
    --outpath /data/PyAPS_results/20221025/232/ \
    --headname stackcube_2963103_RUN2_P0000 \
    --mode ALL
```
**Use case:** First look at all available analysis products

---

### Example 2: Stellar Kinematics Analysis
```bash
python aps_Mapviewer.py \
    --outpath /data/PyAPS_results/galaxy_sample/NGC5194/ \
    --headname LWVE_NGC5194_01_BR_L1_P0001 \
    --mode PPXF \
    --contour_offset 0.2 \
    --theme default
```
**Use case:** Study stellar rotation curves and velocity dispersion

---

### Example 3: Emission Line Kinematics
```bash
python aps_Mapviewer.py \
    --outpath /data/PyAPS_results/emission_line_galaxies/NGC1068/ \
    --headname LWVE_NGC1068_IFU_P0000 \
    --mode EMI_DETAILED \
    --aon_threshold 3.0 \
    --theme dark \
    --contour_offset 0.1
```
**Use case:** Analyze ionized emi kinematics line-by-line

**What you'll see:**
- Separate maps for Hα, Hβ, [OIII], [NII], [SII], etc.
- Velocity and dispersion for each line
- Component separation if multi-component fits
- Flux and equivalent width maps

---

### Example 4: AGN Host Galaxy Study
```bash
python aps_Mapviewer.py \
    --outpath /data/PyAPS_results/AGN/Mrk509/ \
    --headname LWVE_Mrk509_center_P0001 \
    --mode EMIPPXF \
    --contour_offset 0.15 \
    --aon_threshold 5.0 \
    --performance quality \
    --export_format pdf
```
**Use case:** High S/N AGN observation, publication-quality figures

---

### Example 5: Line Strength Index Maps
```bash
python aps_Mapviewer.py \
    --outpath /data/PyAPS_results/early_type/NGC4472/ \
    --headname LWVE_NGC4472_L1_P0000 \
    --mode LS \
    --theme default \
    --contour_offset 0.2
```
**Use case:** Study stellar population gradients via absorption features

**Available indices:**
- Lick/IDS indices (Hβ, Mgb, Fe5270, Fe5335)
- Custom defined indices
- Equivalent widths
- Index errors

---

### Example 6: Star Formation History
```bash
python aps_Mapviewer.py \
    --outpath /data/PyAPS_results/star_forming/NGC628/ \
    --headname LWVE_NGC628_spiral_arm_P0001 \
    --mode SFH \
    --theme default
```
**Use case:** Spatially resolved star formation history

**Parameters available:**
- Age distributions
- Metallicity maps
- Mass-weighted age
- Light-weighted age

---

### Example 7: Interactive Mode (No Command Line)
```bash
python aps_Mapviewer.py
```
**What happens:**
1. Welcome dialog appears
2. File browser opens
3. Select directory and headname
4. Choose display mode
5. MapViewer launches

**Use case:** GUI-driven workflow, exploring multiple datasets

---

### Example 8: High-Resolution IFU with Debug
```bash
python aps_Mapviewer.py \
    --outpath /data/PyAPS_results/LIFU/M31_nucleus/ \
    --headname LWVE_M31_nucleus_LIFU_P0000 \
    --mode ALL \
    --debug \
    --performance quality \
    --validation true \
    --max_size 100
```
**Use case:** Large dataset, need diagnostics and quality checks

**Debug output shows:**
- Load times for each component
- Memory usage
- Data validation results
- Performance metrics

---

### Example 9: Fast Preview Mode
```bash
python aps_Mapviewer.py \
    --outpath /data/PyAPS_results/batch_processing/ \
    --headname observation_quick_P0001 \
    --mode PPXF \
    --performance fast \
    --validation false \
    --max_size 70
```
**Use case:** Quick quality check, many files to review

---

### Example 10: Publication Figure Generation
```bash
python aps_Mapviewer.py \
    --outpath /data/PyAPS_results/paper_figures/target123/ \
    --headname LWVE_target123_final_P0001 \
    --mode EMI_DETAILED \
    --theme high_contrast \
    --contour_offset 0.1 \
    --performance quality \
    --export_format pdf \
    --aon_threshold 5.0
```
**Use case:** Generate publication-ready figures

**Workflow:**
1. Load data in quality mode
2. Select map type (e.g., Hα velocity)
3. Adjust color scale in GUI
4. File → Export → Save as PDF
5. Repeat for other maps

---

### Example 11: Batch Script for Multiple Targets
```bash
#!/bin/bash
# Process multiple targets

targets=(
    "LWVE_target001_P0001"
    "LWVE_target002_P0001"
    "LWVE_target003_P0001"
)

for target in "${targets[@]}"; do
    echo "Processing $target"
    python aps_Mapviewer.py \
        --outpath /data/PyAPS_results/batch/ \
        --headname $target \
        --mode ALL \
        --performance auto \
        --theme default &

    # Wait for user to close window
    wait
done
```
**Use case:** Sequential examination of multiple observations

---

## Display Modes

### ALL Mode (Default)
**Command:** `--mode ALL`

**Shows:**
- Everything available in the data
- Table data (flux, continuum, etc.)
- pPXF results if available
- EMIPPXF results if available
- Line strength indices if available
- SFH parameters if available

**Menu structure:**
```
Display Mode
├── Table Data
│   ├── FLUX
│   ├── CONTINUUM
│   ├── S/N
│   └── ...
├── pPXF Results
│   ├── Velocity
│   ├── Velocity Dispersion
│   ├── h3, h4 moments
│   └── ...
├── EMIPPXF Results
│   ├── emi Velocity
│   ├── emi Dispersion
│   ├── Line Fluxes
│   └── ...
├── Line Strength
│   └── ...
└── SFH
    └── ...
```

**Use case:** First exploration, don't know what's available

---

### PPXF Mode
**Command:** `--mode PPXF`

**Shows only:**
- Stellar velocity (V)
- Stellar velocity dispersion (σ)
- Higher-order moments (h3, h4)
- Age, metallicity (if derived)
- χ² maps
- S/N maps

**Menu items:**
```
pPXF Stellar Kinematics
├── Velocity (km/s)
├── Velocity Dispersion (km/s)
├── h3 (asymmetry)
├── h4 (kurtosis)
├── χ² / DOF
└── S/N ratio
```

**Use case:** Stellar kinematics analysis, rotation curves, dynamical modeling

**Typical workflow:**
1. Plot velocity map → identify rotation pattern
2. Plot σ map → identify dynamically hot regions
3. Plot h3 → check for asymmetries
4. Plot h4 → check for non-Gaussian velocity distributions

---

### EMIPPXF Mode
**Command:** `--mode EMIPPXF`

**Shows only:**
- emi emission line kinematics
- Combined maps (all lines together)
- Line fluxes (total)
- Equivalent widths

**Menu items:**
```
EMIPPXF emi Kinematics
├── emi Velocity (km/s)
├── emi Dispersion (km/s)
├── Total Line Flux
├── Equivalent Width
└── AoN ratio
```

**Use case:** Overall emi kinematics without separating individual lines

**Good for:**
- Quick overview of emi motion
- When lines have similar kinematics
- Comparative analysis with stellar velocity

---

### EMI_DETAILED Mode
**Command:** `--mode EMI_DETAILED`

**Shows:**
- **Per-line kinematics** - separate for each emission line
- Component separation (if multi-component fits)
- Tie groups (lines fitted together)
- Individual line fluxes and EWs

**Menu structure:**
```
emi Kinematics (Detailed)
├── Hα_6564
│   ├── Velocity
│   ├── Velocity Dispersion
│   ├── Flux
│   ├── Equivalent Width
│   ├── AoN
│   └── Components (if multi-component)
├── Hβ_4862
│   └── ...
├── [OIII]_5008
│   └── ...
├── [NII]_6585
│   └── ...
├── [SII]_6718
│   └── ...
└── [SII]_6732
    └── ...
```

**Use case:** Detailed emission line analysis, BPT diagrams, line ratio maps

**Typical workflow:**
1. Compare Hα vs [NII] kinematics
2. Check for velocity differences between lines (outflows?)
3. Plot flux ratios ([NII]/Hα, [OIII]/Hβ)
4. Identify multi-component regions

**Advanced features:**
- Component 1, 2, 3... for each line
- Tie groups (e.g., Hα and [NII] fitted with same kinematics)
- Error maps for each parameter

---

### LS Mode (Line Strength)
**Command:** `--mode LS`

**Shows:**
- Lick/IDS absorption indices
- Custom index definitions
- Index errors
- Adapted binning results

**Available indices:**
```
Line Strength Indices
├── Hβ (4861)
├── Mgb (5175)
├── Fe5270
├── Fe5335
├── NaD (5895)
├── TiO indices
├── [Custom indices]
└── Index errors
```

**Binning options:**
- VORONOI: Original Voronoi bins
- ADAPTED: Adaptive binning for line strength
- SPAXEL: Individual spaxels

**Use case:** Stellar population studies, age/metallicity gradients

**Typical maps:**
- Hβ: Young stellar populations
- Mgb: Metallicity indicator
- Fe indices: Alpha abundance
- Age-sensitive indices

---

### SFH Mode (Star Formation History)
**Command:** `--mode SFH`

**Shows:**
- Age distribution maps
- Metallicity maps
- Mass-weighted parameters
- Light-weighted parameters
- Formation timescales

**Menu items:**
```
Star Formation History
├── Mean Age (Gyr)
├── Mean Metallicity [M/H]
├── Age gradient
├── Recent SF fraction
├── sSFR (if available)
└── Lookback time maps
```

**Use case:** Understanding galaxy assembly history spatially

**Applications:**
- Inside-out growth patterns
- Radial age gradients
- Merger signatures
- Recent star formation

---

## Interface Guide

### Main Window Layout

```
┌─────────────────────────────────────────────────────────┐
│ File  View  Display Mode  Tools  Analysis  Help        │
├──────────────┬──────────────────────────────────────────┤
│              │                                          │
│   Menu       │                                          │
│   Tree       │          Main Plot Area                 │
│              │        (2D Map Display)                  │
│   [Maps]     │                                          │
│   • Table    │                                          │
│   • pPXF     │                                          │
│   • EMIPPXF  │                                          │
│   • Lines    │                                          │
│     - Hα     │                                          │
│     - [OIII] │                                          │
│   • LS       │                                          │
│              │                                          │
│              │                                          │
│   Color      │                                          │
│   Scale      │                                          │
│   Controls   │                                          │
│              │                                          │
├──────────────┴──────────────────────────────────────────┤
│ Status: Ready | Bin: VORONOI | Map: Hα Velocity        │
└─────────────────────────────────────────────────────────┘
```

---

### Menu Bar

#### File Menu
```
File
├── Load Data...          Ctrl+O
├── Reload Current        F5
├── ────────────
├── Export Figure         Ctrl+E
│   ├── Export as PNG
│   ├── Export as PDF
│   └── Export as SVG
├── Export All Visible... Ctrl+Shift+E
├── ────────────
├── Save Session...       Ctrl+S
├── Load Session...       Ctrl+Shift+O
├── ────────────
└── Exit                  Ctrl+Q
```

#### View Menu
```
View
├── Zoom In               Ctrl++
├── Zoom Out              Ctrl+-
├── Reset Zoom            Ctrl+0
├── ────────────
├── Show Colorbar         ☑
├── Show Contours         ☑
├── Show Grid             ☐
├── Show Bin Numbers      ☐
├── ────────────
├── Theme
│   ├── Default
│   ├── Dark
│   └── High Contrast
└── ────────────
    Fullscreen            F11
```

#### Display Mode Menu
```
Display Mode
├── All Available         Ctrl+1
├── pPXF Only             Ctrl+2
├── EMIPPXF Only          Ctrl+3
├── emi Detailed          Ctrl+4
├── Line Strength         Ctrl+5
├── SFH                   Ctrl+6
└── Table Only            Ctrl+7
```

#### Tools Menu
```
Tools
├── Adjust Color Scale    Ctrl+C
├── Set AoN Threshold...
├── Contour Settings...
├── ────────────
├── Bin Selection Tool
├── Extract Spectrum...
├── Statistics Window     Ctrl+T
├── ────────────
└── Preferences...        Ctrl+,
```

#### Analysis Menu
```
Analysis
├── Calculate BPT Diagram
├── Line Ratio Maps
│   ├── [NII]/Hα
│   ├── [OIII]/Hβ
│   └── Custom Ratio...
├── ────────────
├── Kinematic Analysis
│   ├── PA Diagram
│   ├── PVD Extraction
│   └── Rotation Curve
├── ────────────
└── Batch Processing...
```

#### Help Menu
```
Help
├── Documentation         F1
├── Keyboard Shortcuts    F2
├── About...
├── ────────────
└── Debug Info            Ctrl+D
```

---

### Left Sidebar (Map Selection Tree)

The tree structure dynamically updates based on available data:

```
📂 Available Maps
├── 📁 Table Data
│   ├── 📊 FLUX
│   ├── 📊 CONTINUUM
│   ├── 📊 SNR
│   └── 📊 ERROR
├── 📁 pPXF Stellar Kinematics
│   ├── 📊 Velocity
│   ├── 📊 Velocity_Dispersion
│   ├── 📊 h3
│   └── 📊 h4
├── 📁 EMIPPXF emi Kinematics
│   ├── 📊 emi_Velocity
│   ├── 📊 emi_Dispersion
│   └── 📊 Total_Flux
└── 📁 Emission Lines (Detailed)
    ├── 📁 Hα_6564
    │   ├── 📊 Velocity
    │   ├── 📊 Dispersion
    │   ├── 📊 Flux
    │   ├── 📊 EW
    │   └── 📊 AoN
    ├── 📁 [OIII]_5008
    │   └── ...
    └── 📁 [NII]_6585
        └── ...
```

**Usage:**
- Click on any map name to display it
- Double-click to expand/collapse folders
- Right-click for context menu (export, stats, etc.)

---

### Color Scale Controls (Bottom Left)

```
┌─────────────────────────┐
│ Color Scale Settings    │
├─────────────────────────┤
│ Min: [___-50___] Auto ☑ │
│ Max: [____50___] Auto ☑ │
│                         │
│ Colormap: [viridis ▼]   │
│                         │
│ ☑ Symmetric around zero │
│ ☐ Log scale             │
│                         │
│ [Apply] [Reset]         │
└─────────────────────────┘
```

**Available colormaps:**
- viridis (default)
- plasma
- inferno
- coolwarm
- RdBu (red-blue diverging)
- seismic
- jet (legacy, not recommended)

---

### Main Plot Area

The central plot shows:

1. **2D Map** - Color-coded data values
2. **Colorbar** - Value scale (right side)
3. **Contours** - Optional overlay (typically continuum)
4. **Voronoi edges** - Bin boundaries (if enabled)
5. **Axis labels** - RA, DEC offsets (arcsec)

**Interactive features:**
- **Click** on any spaxel/bin → Shows value in status bar
- **Right-click** → Context menu (extract spectrum, statistics)
- **Scroll wheel** → Zoom in/out
- **Middle-click drag** → Pan
- **Shift+click** → Select multiple bins

---

### Status Bar

Shows real-time information:

```
Ready | Bin: VORONOI | Map: Hα_6564 Velocity | Value: 125.3 km/s at (2.4, -1.8) arcsec | Mode: EMI_DETAILED
```

**Information displayed:**
- Status (Ready, Loading, Processing, etc.)
- Current binning scheme
- Currently displayed map
- Value at cursor position
- Coordinates at cursor
- Current display mode

---

## Data Structure

### Required Files

MapViewer expects files with specific naming conventions:

#### 1. Main Table File (REQUIRED)
```
{headname}_table.fits
```
**Contains:**
- X, Y coordinates (RA, DEC offsets)
- BIN_ID (Voronoi bin assignment)
- Flux, continuum, S/N
- Any derived table columns

**Structure:**
```
Columns:
- X_IMAGE: X coordinate (pixels or arcsec)
- Y_IMAGE: Y coordinate (pixels or arcsec)
- BIN_ID: Voronoi bin number
- FLUX: Integrated flux
- CONTINUUM: Continuum level
- SNR: Signal-to-noise ratio
- [Additional derived columns]
```

---

#### 2. Voronoi Binning Info (REQUIRED)
```
{headname}_VorBinInfo.fits
```
**Contains:**
- Mapping between spaxels and bins
- Bin centers
- S/N per bin

**Structure:**
```
Columns:
- BIN_NUMBER: Bin ID
- X_CENTER: Bin center X
- Y_CENTER: Bin center Y
- SNR: S/N of bin
- N_SPAXELS: Number of spaxels in bin
```

---

#### 3. pPXF Results (OPTIONAL)
```
{headname}_ppxf_results.fits
{headname}_ppxf_results_VORONOI.fits
```
**Contains:**
- Stellar velocity (V)
- Velocity dispersion (σ)
- h3, h4 moments
- Errors
- χ² values

**Structure:**
```
Extension 0: Primary HDU
Extension 1: VORONOI binning results
Extension 2: SPAXEL-by-spaxel results (if available)

Columns in each extension:
- V: Velocity (km/s)
- SIGMA: Velocity dispersion (km/s)
- H3: Third Hermite moment
- H4: Fourth Hermite moment
- V_ERROR: Velocity uncertainty
- SIGMA_ERROR: Dispersion uncertainty
- CHI2: χ² of fit
```

---

#### 4. EMIPPXF Results (OPTIONAL)
```
{headname}_emippxf_results.fits
{headname}_emippxf_BIN.fits
{headname}_emippxf_ADAPTED.fits
```
**Contains:**
- emi emission line kinematics
- Per-line velocities, dispersions
- Fluxes, equivalent widths
- Multi-component fits (if present)

**Structure:**
```
Extensions:
1. BIN level results (Voronoi binned)
2. ADAPTED level (adaptive binning)
3. SPAXEL level (if available)

Columns pattern:
- {LINE}_V: Velocity for line
- {LINE}_SIGMA: Dispersion for line
- {LINE}_FLUX: Integrated flux
- {LINE}_EW: Equivalent width
- {LINE}_AON: Amplitude over noise
- {LINE}_V_ERROR: Velocity error
- {LINE}_SIGMA_ERROR: Dispersion error

Where {LINE} examples:
- Halpha_6564
- Hbeta_4862
- OIII_5008
- NII_6585
- SII_6718
- SII_6732
```

**Multi-component format:**
```
# If multiple components fitted:
- Halpha_6564_COMP1_V
- Halpha_6564_COMP1_SIGMA
- Halpha_6564_COMP2_V
- Halpha_6564_COMP2_SIGMA
- ...
```

---

#### 5. Line Strength Results (OPTIONAL)
```
{headname}_ls_results.fits
{headname}_ls_VORONOI.fits
{headname}_ls_ADAPTED.fits
```
**Contains:**
- Lick/IDS indices
- Custom absorption indices
- Index errors

**Structure:**
```
Columns:
- Hbeta: Hβ index (Å)
- Hbeta_ERROR: Error (Å)
- Mgb: Mg b index
- Fe5270: Fe5270 index
- Fe5335: Fe5335 index
- [Additional indices]
```

---

#### 6. SFH Results (OPTIONAL)
```
{headname}_sfh_results.fits
```
**Contains:**
- Age distributions
- Metallicity distributions
- Mass/light-weighted parameters

**Structure:**
```
Columns:
- AGE_MW: Mass-weighted age (Gyr)
- AGE_LW: Light-weighted age (Gyr)
- METALLICITY_MW: Mass-weighted [M/H]
- METALLICITY_LW: Light-weighted [M/H]
- [Additional SFH parameters]
```

---

### Coordinate Systems

MapViewer handles multiple coordinate conventions:

1. **Pixel coordinates** (X_IMAGE, Y_IMAGE)
   - Origin at CCD corner
   - Units: pixels

2. **Sky offsets** (RA_OFFSET, DEC_OFFSET)
   - Origin at field center
   - Units: arcseconds
   - Calculated as:
     ```
     RA_offset = (RA - RA_center) * cos(DEC_center) * 3600
     DEC_offset = (DEC - DEC_center) * 3600
     ```

3. **Absolute sky** (RA, DEC)
   - Origin at J2000 equinox
   - Units: degrees

**MapViewer displays:** Sky offsets (arcsec) by default

---

### Binning Schemes

Three binning levels are typically available:

#### VORONOI
- Original Voronoi tessellation
- Target S/N per bin
- Irregular bin shapes
- **Use for:** General kinematics

#### ADAPTED
- Adaptive binning for line strength
- Higher S/N requirement
- Accounts for continuum S/N
- **Use for:** Absorption line indices

#### SPAXEL
- Individual spaxels (no binning)
- Highest spatial resolution
- Lower S/N per point
- **Use for:** High S/N data only

---

## Analysis Capabilities

### emi Kinematics Analysis

#### Velocity Maps
**Purpose:** Trace ionized emi motion

**Typical uses:**
1. **Rotation curves** - Extract along major axis
2. **Outflows** - High-velocity emi away from disk
3. **Inflows** - Radial motion toward center
4. **Comparison between lines** - Different ionization states

**What to look for:**
- Spider diagram pattern → Rotation
- Asymmetries → Non-circular motion, outflows
- Velocity gradients → Ordered motion
- Velocity jumps → Shocks, interaction

#### Dispersion Maps
**Purpose:** Measure emi velocity width

**Typical values:**
- Quiescent emi: σ < 50 km/s
- Active star formation: 50 < σ < 100 km/s
- AGN/outflows: σ > 100 km/s

**What to look for:**
- Central peaks → Nuclear activity
- Extended high-σ → Large-scale turbulence
- σ vs velocity correlation → Outflows

#### Flux Maps
**Purpose:** Trace emission line intensity

**Applications:**
- Star formation rate (via Hα)
- Ionization structure (line ratios)
- Extinction mapping (Hα/Hβ)

---

### Stellar Kinematics Analysis

#### Velocity Field
**Purpose:** Trace stellar rotation

**Analysis:**
1. Extract along major/minor axes
2. Fit rotation curve model
3. Derive dynamical mass
4. Compare to emi velocity

**Features:**
- Regular rotation → Relaxed disk
- Kinematic twists → Warps, bars
- Decoupled components → Mergers

#### Velocity Dispersion
**Purpose:** Measure stellar velocity width

**Typical values:**
- Disk-dominated: σ < 100 km/s
- Bulge: 100 < σ < 200 km/s
- Elliptical: σ > 150 km/s

**Applications:**
- Dynamical mass (via virial theorem)
- Schwarzschild modeling
- Jeans modeling

#### Higher Moments (h3, h4)
**Purpose:** Measure velocity distribution shape

**Interpretation:**
- h3 ≠ 0: Asymmetric velocity distribution
  - h3 × V > 0: Trailing component
  - h3 × V < 0: Leading component

- h4 > 0: Broader wings (than Gaussian)
- h4 < 0: Narrower wings

**Use cases:**
- Detect counter-rotating components
- Identify multiple stellar populations
- Dynamical modeling constraints

---

### Line Strength Analysis

#### Absorption Indices
**Available indices:**
- Hβ: Age-sensitive (young stars)
- Mgb: Metallicity indicator
- Fe5270, Fe5335: Alpha abundance
- NaD: IMF-sensitive
- TiO: M-dwarf sensitive

#### Radial Gradients
**Workflow:**
1. Extract azimuthally averaged profiles
2. Fit linear/polynomial gradients
3. Compare to stellar population models

**Typical gradients:**
- Metallicity: Negative (higher in center)
- Age: Positive (older in center) or flat

---

### Diagnostic Diagrams

#### BPT Diagram
**Purpose:** Classify ionization source

**Lines used:**
- [NII]λ6585 / Hα vs [OIII]λ5008 / Hβ

**Regions:**
- Star formation: Below Kauffmann line
- AGN: Above Kewley line
- Composite: Between lines
- LINER: Lower-right region

**MapViewer workflow:**
1. Analysis → Calculate BPT Diagram
2. Shows spatially-resolved classification
3. Color-coded by ionization source

---

### Custom Line Ratios

**Create custom ratio maps:**

1. Analysis → Line Ratio Maps → Custom Ratio
2. Enter numerator line (e.g., "NII_6585_FLUX")
3. Enter denominator line (e.g., "Halpha_6564_FLUX")
4. Map displays with log or linear scale
5. Export for further analysis

**Common ratios:**
- [NII]/Hα: Metallicity proxy
- [OIII]/Hβ: Ionization parameter
- [SII]/Hα: Density indicator
- Hα/Hβ: Extinction (Balmer decrement)

---

## Keyboard Shortcuts

### Essential Shortcuts

| Shortcut | Action |
|----------|--------|
| **Ctrl+O** | Open/Load new data |
| **Ctrl+E** | Export current figure |
| **Ctrl+Q** | Quit application |
| **F5** | Refresh/Reload data |
| **F1** | Help documentation |
| **Esc** | Cancel current operation |

### Display Mode Shortcuts

| Shortcut | Mode |
|----------|------|
| **Ctrl+1** | ALL mode |
| **Ctrl+2** | PPXF mode |
| **Ctrl+3** | EMIPPXF mode |
| **Ctrl+4** | EMI_DETAILED mode |
| **Ctrl+5** | Line Strength mode |
| **Ctrl+6** | SFH mode |
| **Ctrl+7** | Table only mode |

### View Shortcuts

| Shortcut | Action |
|----------|--------|
| **Ctrl++** | Zoom in |
| **Ctrl+-** | Zoom out |
| **Ctrl+0** | Reset zoom |
| **F11** | Toggle fullscreen |
| **Ctrl+C** | Adjust color scale |

### Navigation

| Shortcut | Action |
|----------|--------|
| **Arrow keys** | Pan view |
| **Ctrl+Home** | Reset view to center |
| **Page Up/Down** | Next/previous map in list |
| **Space** | Play through maps (slideshow) |

### Advanced

| Shortcut | Action |
|----------|--------|
| **Ctrl+D** | Toggle debug mode |
| **Ctrl+T** | Statistics window |
| **Ctrl+R** | Refresh current map |
| **Ctrl+S** | Save session |
| **Ctrl+,** | Preferences |

### Mouse Controls

| Action | Effect |
|--------|--------|
| **Left click** | Select spaxel, show value |
| **Right click** | Context menu |
| **Scroll wheel** | Zoom in/out |
| **Middle click + drag** | Pan view |
| **Shift + click** | Multi-select spaxels |
| **Ctrl + click** | Add to selection |

---

## Tips & Tricks

### Performance Optimization

#### For Large Datasets
```bash
# Use fast performance mode
python aps_Mapviewer.py \
    --outpath /data/large_ifu/ \
    --headname observation \
    --performance fast \
    --mode PPXF
```

**Effect:**
- Reduces plot quality slightly
- Faster rendering
- Lower memory usage
- Good for quick exploration

#### Skip Validation
```bash
# Skip validation checks
--validation false
```
**When to use:** Known good data, repeated viewing

---

### Visual Customization

#### Dark Theme for Presentations
```bash
python aps_Mapviewer.py \
    --outpath /data/ \
    --headname obs \
    --theme dark \
    --mode EMI_DETAILED
```

**Benefits:**
- Better for projection
- Less eye strain in dark rooms
- Higher contrast

#### High Contrast for Publication
```bash
--theme high_contrast
```

**Features:**
- Optimized for print
- Clear axis labels
- Publication-ready defaults

---

### Analysis Workflows

#### Workflow 1: Quick Quality Check
**Goal:** Verify data reduction worked

**Steps:**
1. Launch in ALL mode
2. Check Table → FLUX map (data loaded?)
3. Check pPXF → Velocity (stellar fit OK?)
4. Check EMIPPXF → emi maps (emission line fits?)
5. Look for obvious failures (NaNs, zeros, outliers)

**Time:** 2-3 minutes per target

---

#### Workflow 2: emi Kinematics Study
**Goal:** Characterize ionized emi motion

**Steps:**
1. Launch in EMI_DETAILED mode
2. Load Hα velocity map
   - Identify rotation pattern
   - Note velocity range
3. Load Hα dispersion map
   - Find high-σ regions
   - Compare to velocity field
4. Load [OIII] velocity map
   - Compare to Hα
   - Look for differences (outflows?)
5. Calculate line ratios
   - [NII]/Hα for BPT
   - [OIII]/Hβ for ionization
6. Export maps for publication

**Time:** 15-30 minutes

---

#### Workflow 3: Multi-Component Analysis
**Goal:** Separate kinematic components

**Steps:**
1. Launch with `--mode EMI_DETAILED`
2. Navigate to Hα → Components
3. Plot Component 1 velocity
4. Plot Component 2 velocity (if exists)
5. Compare flux ratio: COMP2/COMP1
6. Identify regions with significant second component
7. Investigate: Outflow? Rotation? Interaction?

**Time:** 20-40 minutes

---

### Data Exploration Strategies

#### Strategy 1: Top-Down
1. Start with ALL mode
2. Browse all available maps
3. Identify interesting features
4. Switch to specific mode for detail
5. Export interesting maps

**Best for:** First-time data exploration

---

#### Strategy 2: Hypothesis-Driven
1. Know what you're looking for (e.g., outflow)
2. Start in EMI_DETAILED mode
3. Go directly to relevant maps
4. Perform targeted analysis
5. Export results

**Best for:** Follow-up analysis

---

#### Strategy 3: Comparative
1. Load multiple observations sequentially
2. Use same colormap/scale for each
3. Compare features across targets
4. Document differences
5. Statistical analysis

**Best for:** Survey work, samples

---

### Export Best Practices

#### For Publications
```python
Settings:
- Performance: quality
- Theme: high_contrast
- Export format: PDF (vector)
- DPI: 300+ for raster elements
- Color scale: Symmetric if velocity
- Contours: ON for context
```

#### For Presentations
```python
Settings:
- Theme: dark (if room is dark)
- Theme: high_contrast (if projected)
- Export format: PNG
- DPI: 150 (sufficient for slides)
- Font size: Increase via preferences
```

#### For Quick Sharing
```python
Settings:
- Performance: auto
- Export format: PNG
- DPI: 100
- Annotate with text if needed
```

---

### Troubleshooting Tips

#### Map Looks Wrong
**Symptom:** Strange values, discontinuities

**Checks:**
1. Verify color scale not saturated (check min/max)
2. Try symmetric scale (for velocity)
3. Check for NaN values (gaps in data)
4. Verify correct binning level selected
5. Check data units (km/s vs m/s?)

#### Missing Maps
**Symptom:** Expected maps don't appear in menu

**Solutions:**
1. Check file exists: `{headname}_emippxf_results.fits`
2. Verify file format (FITS table with correct columns)
3. Try --mode ALL to see what's available
4. Check --debug output for load errors
5. Validate file with `fitsinfo` or similar

#### Slow Performance
**Symptom:** Lagging response, slow rendering

**Solutions:**
1. Use `--performance fast`
2. Reduce window size: `--max_size 70`
3. Disable contours temporarily
4. Close other applications
5. Use VORONOI instead of SPAXEL binning

---

## Troubleshooting

### Installation Issues

#### ImportError: No module named 'PyQt5'
```bash
# Solution
pip install PyQt5

# If fails, try
pip install --upgrade pip
pip install PyQt5==5.15.9
```

#### ImportError: Local modules not found
```bash
# Error: cannot import createWindow, createFigure, etc.

# Solution 1: Check directory structure
ls mapviewer/
# Should contain: createWindow.py, createFigure.py, loadData.py, plotData.py, helperFunctions.py

# Solution 2: Set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/path/to/PyAPS/
```

#### Matplotlib backend errors
```bash
# Warning: Could not set Qt5Agg backend

# Solution: Install matplotlib with Qt support
pip install matplotlib[qt]

# Or explicitly install
pip install matplotlib PyQt5
```

---

### Runtime Issues

#### Error: "Directory does not exist"
```bash
# Check path
ls -la /path/to/PyAPS_results/

# Common mistakes:
# - Typo in path
# - Missing trailing slash (not usually needed)
# - Relative vs absolute path
# - Spaces in path (use quotes)

# Solution:
python aps_Mapviewer.py \
    --outpath "/path/with spaces/results/" \
    --headname observation
```

#### Warning: No files found with prefix
```bash
# Check files exist
ls /path/to/results/ | grep headname

# Verify headname exactly matches file prefix
# Example: If files are "obs_P0001_table.fits"
#          Then headname should be "obs_P0001"

# Check for typos, underscores, case sensitivity
```

#### Error: "Cannot load table file"
```bash
# Verify file integrity
python -c "from astropy.io import fits; fits.info('path/to/file_table.fits')"

# Check file permissions
ls -l path/to/file_table.fits

# Solution: Re-reduce data or fix file permissions
chmod 644 path/to/file_table.fits
```

---

### Display Issues

#### Blank/Empty Main Plot
**Symptoms:** Window opens but plot area is blank

**Causes & Solutions:**
1. **No data loaded**
   - Check console for error messages
   - Verify files exist and are readable

2. **Matplotlib backend issue**
   - Try different backend: `export MPLBACKEND=Qt5Agg`
   - Reinstall matplotlib: `pip install --force-reinstall matplotlib`

3. **Graphics driver issue**
   - Update graphics drivers
   - Try software rendering: `export LIBGL_ALWAYS_SOFTWARE=1`

---

#### Menu Tree Empty
**Symptoms:** Left sidebar shows no maps

**Causes:**
1. **No analysis products found**
   - Check that analysis has been run (pPXF, EMIPPXF, etc.)
   - Verify output files exist

2. **Wrong MODE selected**
   - Try `--mode ALL` to see all available data

3. **File format issues**
   - Validate FITS files: `fitsheader file.fits`

**Solution:**
```bash
# Debug mode to see what's being loaded
python aps_Mapviewer.py \
    --outpath /path/to/data/ \
    --headname obs \
    --mode ALL \
    --debug
```

---

#### Colors Look Wrong
**Symptoms:** Map displayed but colors are strange

**Common issues:**

1. **Saturated scale**
   - Min/max set incorrectly
   - Solution: Click "Auto" for min and max

2. **Wrong colormap**
   - Diverging map for non-diverging data
   - Solution: Change colormap (viridis for scalar, coolwarm for velocity)

3. **Log scale on negative data**
   - Can't take log of negative numbers
   - Solution: Uncheck "Log scale"

---

#### Contours Not Showing
**Symptoms:** Contours checkbox is ticked but nothing shows

**Causes:**
1. **Contour data not available**
   - Usually uses FLUX or CONTINUUM
   - Check these columns exist in table

2. **Contour level too high**
   - Default 0.2 may be above peak
   - Solution: `--contour_offset 0.05` (lower threshold)

3. **Rendering issue**
   - Try turning off and on again
   - Check View → Show Contours

---

### Performance Issues

#### Very Slow Startup
**Symptoms:** Takes >30 seconds to open

**Solutions:**
1. **Disable validation**
   ```bash
   --validation false
   ```

2. **Use fast performance**
   ```bash
   --performance fast
   ```

3. **Check disk I/O**
   - Slow network drive?
   - Copy files to local disk

4. **Reduce data size**
   - Use VORONOI instead of SPAXEL
   - Filter to specific region if possible

---

#### Laggy Response
**Symptoms:** Slow to switch between maps, zooming laggy

**Solutions:**
1. **Reduce window size**
   ```bash
   --max_size 70
   ```

2. **Disable auto-refresh**
   - Turn off in preferences

3. **Close other applications**
   - Free up RAM
   - Close browser tabs

4. **Use faster computer**
   - MapViewer is memory-intensive for large IFU

---

### Data Issues

#### Maps Show Only NaNs/Zeros
**Symptoms:** Map displays but all values are NaN or zero

**Causes:**
1. **Analysis failed**
   - pPXF/EMIPPXF fit didn't converge
   - Check fit quality in original analysis

2. **Wrong binning level**
   - ADAPTED may be empty if analysis not run at that level
   - Solution: Switch to VORONOI or SPAXEL

3. **Threshold too high**
   - AoN threshold excludes all data
   - Solution: Lower `--aon_threshold 2.0`

---

#### Velocity Maps Look Random
**Symptoms:** No coherent structure in velocity field

**Possible causes:**
1. **Low S/N data**
   - Fits are unreliable
   - Check S/N maps

2. **Wrong systemic velocity**
   - Need to subtract systemic before displaying
   - May need to reprocess

3. **Multiple components confused**
   - If multi-component, look at individual components
   - Mode: EMI_DETAILED → Components

---

#### Missing Emission Lines
**Symptoms:** Some lines don't appear in EMI_DETAILED mode

**Reasons:**
1. **Line not fitted**
   - Check EMIPPXF configuration
   - Line may be outside wavelength range

2. **Low S/N**
   - Line detected but below AoN threshold
   - Solution: Lower `--aon_threshold`

3. **File not found**
   - Check for `{headname}_emippxf_results.fits`

---

### Advanced Troubleshooting

#### Debug Mode Diagnostics
```bash
# Run with debug to see detailed info
python aps_Mapviewer.py \
    --outpath /path/to/data/ \
    --headname obs \
    --mode ALL \
    --debug

# Output will show:
# - Files found and loaded
# - Columns in each file
# - Errors during loading
# - Performance metrics
# - Memory usage
```

**What to look for in debug output:**
- "Error loading X": Specific file has issues
- "No columns matching": File format problem
- "Memory warning": Dataset too large
- "Performance: X seconds": Identify bottlenecks

---

#### Common Error Messages

| Error | Meaning | Solution |
|-------|---------|----------|
| FileNotFoundError | Files missing | Check path and headname |
| KeyError: 'COLUMN' | Expected column not in file | Verify file format, check column names |
| ValueError: NaN | Data contains NaNs | Expected for masked regions, lower threshold if too many |
| MemoryError | Insufficient RAM | Use binned data, reduce window size, close apps |
| RuntimeError: Qt | Qt/GUI issue | Reinstall PyQt5, check display settings |

---

#### Validation Mode
```bash
# Enable validation for comprehensive checks
python aps_Mapviewer.py \
    --outpath /path/to/data/ \
    --headname obs \
    --validation true \
    --debug
```

**Validation checks:**
- File existence
- File format correctness
- Column completeness
- Data range sanity
- Coordinate consistency
- Binning information validity

**Output:**
- ✅ Green: All checks passed
- ⚠️ Yellow: Warnings (data may be usable)
- ❌ Red: Errors (data unusable)

---

#### Getting Help

If issues persist:

1. **Check console output** for error messages
2. **Run with --debug** for detailed diagnostics
3. **Verify file formats** using fitsinfo or similar
4. **Test with demo data** (if available)
5. **Contact support** with:
   - Full command used
   - Error message (full traceback)
   - Debug output
   - File structure (ls -R output)
   - Python/package versions

---

## Advanced Usage

### Custom Themes

Create custom theme files:

```python
# In mapviewer/themes/my_theme.py
theme_config = {
    'background': '#1e1e1e',
    'foreground': '#ffffff',
    'grid_color': '#444444',
    'colormap_default': 'viridis',
    'font_size': 12,
    'dpi': 100
}
```

Use custom theme:
```bash
python aps_Mapviewer.py --theme my_theme --outpath ...
```

---

### Batch Processing

Create script to process multiple targets:

```bash
#!/bin/bash
# process_batch.sh

targets=(
    "obs001_P0001"
    "obs002_P0001"
    "obs003_P0001"
)

for target in "${targets[@]}"; do
    echo "Processing $target"

    python aps_Mapviewer.py \
        --outpath /data/PyAPS_results/batch/ \
        --headname $target \
        --mode EMI_DETAILED \
        --performance fast \
        --export_format pdf &

    # Get PID
    pid=$!

    # Wait for window to open and stabilize
    sleep 5

    # Auto-export all visible maps (if implemented)
    # Or wait for user to close
    wait $pid

    echo "$target completed"
done

echo "All targets processed"
```

---

### Programmatic Access

Access MapViewer functionality from Python:

```python
#!/usr/bin/env python
"""
Programmatic MapViewer usage example
"""

from PyQt5 import QtWidgets
import sys

# Import MapViewer
from PyAPS.aps_Mapviewer import EnhancedMapviewer

# Create Qt application
app = QtWidgets.QApplication(sys.argv)

# Create MapViewer instance
viewer = EnhancedMapviewer(
    MODE='EMI_DETAILED',
    outpath='/data/PyAPS_results/target/',
    headname='observation_P0001',
    contour_offset=0.1,
    debug=True,
    theme='dark'
)

# Programmatically plot specific map
viewer.plotMap("Halpha_6564_V")

# Access data
velocity_data = viewer.current_map_data
coordinates = viewer.coordinates

# Export figure
viewer.exportFigure('/tmp/halpha_velocity.pdf', format='pdf')

# Show GUI
viewer.show()

# Run
sys.exit(app.exec_())
```

---

### Extending MapViewer

Add custom analysis functions:

```python
# In mapviewer/custom_analysis.py

def calculate_rotation_curve(viewer):
    """
    Extract rotation curve along major axis
    """
    # Get velocity data
    velocity = viewer.get_current_data()

    # Get coordinates
    x = viewer.coordinates['x']
    y = viewer.coordinates['y']

    # Define major axis (PA from user or fits)
    PA = 45  # degrees

    # Extract along major axis
    # ... implementation ...

    return radius, velocity_profile


def calculate_bpt_classification(viewer):
    """
    Create spatially-resolved BPT classification
    """
    # Get line fluxes
    nii_flux = viewer.get_line_data('NII_6585_FLUX')
    ha_flux = viewer.get_line_data('Halpha_6564_FLUX')
    oiii_flux = viewer.get_line_data('OIII_5008_FLUX')
    hb_flux = viewer.get_line_data('Hbeta_4862_FLUX')

    # Calculate ratios
    nii_ha = np.log10(nii_flux / ha_flux)
    oiii_hb = np.log10(oiii_flux / hb_flux)

    # Apply classification
    # ... Kauffmann & Kewley lines ...

    return classification_map
```

Register custom functions:
```python
# In main code
viewer.register_custom_analysis('Rotation Curve', calculate_rotation_curve)
viewer.register_custom_analysis('BPT Classification', calculate_bpt_classification)
```

---

## Technical Details

### Architecture

```
aps_Mapviewer.py (Main)
├── EnhancedMapviewer (Class)
│   ├── __init__: Setup
│   ├── createFigure: Matplotlib setup
│   ├── createWindow: Qt GUI
│   ├── loadData: Load FITS files
│   ├── plotData: Render maps
│   └── helperFunctions: Utilities
│
└── Modules (mapviewer/)
    ├── createWindow.py: GUI components
    ├── createFigure.py: Matplotlib figures
    ├── loadData.py: FITS I/O
    ├── plotData.py: Plotting functions
    └── helperFunctions.py: Utilities
```

---

### Performance Characteristics

**Memory Usage:**
- Small IFU (100 spaxels): ~50 MB
- Medium IFU (500 spaxels): ~200 MB
- Large IFU (2000 spaxels): ~800 MB
- LIFU (10000+ spaxels): >2 GB

**Rendering Speed:**
- VORONOI binning: Fast (<1 sec)
- SPAXEL level: Moderate (1-3 sec)
- Large SPAXEL: Slow (>5 sec)

**Optimization strategies:**
- Use VORONOI for exploration
- SPAXEL only when needed
- Fast performance mode for large data

---

### File Format Specifications

**Table FITS format:**
```
Primary HDU: Empty
Extension 1 (BINARY TABLE):
  - Columns: X_IMAGE, Y_IMAGE, BIN_ID, FLUX, ...
  - Units: Specified in column headers
  - Coordinates: Arcsec or pixels
```

**Results FITS format:**
```
Primary HDU: Metadata in header
Extension 1: VORONOI results
Extension 2: ADAPTED results (optional)
Extension 3: SPAXEL results (optional)

Each extension:
  - Binary table with results
  - One row per bin/spaxel
  - Column names: PARAMETER_SUFFIX
```

---

## Contributing

### How to Contribute

Contributions welcome! Areas needing development:

**High Priority:**
- Additional analysis tools
- Export automation
- Better error handling
- Performance optimization

**Medium Priority:**
- More themes
- Additional colormaps
- Batch processing tools
- Tutorial notebooks

**Documentation:**
- More examples
- Video tutorials
- Best practices guide

### Development Setup

```bash
# Clone repository
git clone https://github.com/amolaeinezhad/PyAPS.git
cd PyAPS

# Create development environment
python -m venv venv_mapviewer
source venv_mapviewer/bin/activate

# Install in development mode
pip install -e .

# Install development dependencies
pip install pytest sphinx

# Run tests (if available)
pytest tests/
```

---

## Version History

### v3.0-Enhanced (Current)
**Release Date:** 2025-01

**Major Features:**
- Per-line emi kinematics visualization
- Enhanced performance modes
- Multiple themes
- Improved error handling
- Debug mode
- Comprehensive logging

**Bug Fixes:**
- Memory leaks in large datasets
- Contour rendering issues
- Qt compatibility improvements

---

### v2.0
**Release Date:** 2023-10

**Features:**
- Multi-component emi fitting support
- Line strength analysis
- SFH visualization

---

### v1.0
**Release Date:** 2018-09

**Initial Release:**
- Basic IFU map visualization
- pPXF and EMIPPXF support
- Interactive plotting

---

## Citation

If you use APS MapViewer in your research, please cite:

```bibtex
@software{aps_mapviewer,
  author = {Molaeinezhad, Alireza},
  title = {APS MapViewer: Interactive IFU Map Visualization for WEAVE},
  year = {2025},
  version = {3.0-Enhanced},
  publisher = {IOA, University of Cambridge},
  url = {https://github.com/amolaeinezhad/PyAPS}
}
```

Also cite the PyAPS pipeline and WEAVE instrument papers.

---

## License

Copyright (C) 2018-2025, Alireza Molaeinezhad (IOA, Cambridge)

MIT License

---

## Contact & Support

**Author:** Alireza Molaeinezhad
**Email:** amolaei_at_st.cam.ac.uk
**Institution:** Institute of Astronomy, University of Cambridge
**GitHub:** https://github.com/amolaeinezhad/PyAPS

**For questions:**
- Check this documentation first
- Run with --debug for diagnostics
- Contact via email with full error details

---

## Acknowledgments

- **WEAVE Team** at Isaac Newton Group
- **IOA/Cambridge** for development support
- **PyAPS Contributors**
- **pPXF** (Cappellari & Emsellem)
- **EMIPPXF** (Sarzi et al.)

---

**Last Updated:** January 2025
**Version:** 3.0-Enhanced
**Status:** Production Ready ✅

---

*End of Documentation*
