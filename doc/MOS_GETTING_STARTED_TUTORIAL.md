# PyAPS MOS Getting Started Tutorial

> ⚠️ This tutorial is under active development and will be expanded in future releases.

**Complete Guide for First-Time Users**

This tutorial will guide you through running PyAPS for MOS (Multi-Object Spectroscopy) data analysis from scratch. By the end of this tutorial, you'll be able to process WEAVE MOS observations and produce science-ready L2 data products.

---

## ⚠️ Important: When to Use This Tutorial

**This guide is for WEAVE MOS and fiber-level IFU data.**

✅ **Use this MOS tutorial if:**
- Your L1 data is at **fiber level** (e.g., `stack_*.fit`, `single_*.fit`)
- Data contains individual fiber spectra (2D: wavelength, fiber)
- You have MOS observations (hundreds of targets across the field)
- You have IFU data that is NOT yet in cube format
- Applies to **all** MOS configurations:
  - ✓ MOS mode (any resolution: LR, HR)
  - ✓ Any binning (1×1, 2×1, 4×1, etc.)
  - ✓ Single exposures, stacks, or superstacks

❌ **Do NOT use this tutorial if:**
- Your L1 data is in **stackcube** format (3D IFU cubes)
- File names contain "stackcube" (e.g., `stackcube_*.fit`)

> **For IFU cube data:** See [IFU_GETTING_STARTED_TUTORIAL.md](IFU_GETTING_STARTED_TUTORIAL.md) instead.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [PyAPS Installation](#2-pyaps-installation)
3. [Data Structure Setup](#3-data-structure-setup)
4. [Downloading Required Templates](#4-downloading-required-templates)
5. [Configuring the Master Config File](#5-configuring-the-master-config-file)
6. [Understanding Your Input Data](#6-understanding-your-input-data)
7. [Running aps_runner.py for MOS](#7-running-apsrunnerpy-for-mos)
8. [Understanding the Generated Scripts](#8-understanding-the-generated-scripts)
9. [Running the MOS Pipeline Scripts](#9-running-the-mos-pipeline-scripts)
10. [Understanding the Outputs](#10-understanding-the-outputs)
11. [Diagnostic Plots and Quality Assessment](#11-diagnostic-plots-and-quality-assessment)
12. [Advanced Options](#12-advanced-options)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Introduction

### About This Tutorial

This tutorial focuses on **`aps_runner.py`** for MOS mode - the most straightforward method for running PyAPS on fiber-level spectroscopic data. `aps_runner.py` automates the entire workflow from L1 input data to L2 science products.

### User Experience Levels

PyAPS for WEAVE is designed to accommodate different levels of expertise:

**🟢 Beginner Users (This Tutorial)**

- Use `aps_runner.py` to automatically generate and run all pipeline scripts
- The pipeline orchestrates the entire process with minimal user intervention
- Automatically handles L1 data model considerations and applies necessary fixes to raw data

**🟡 Intermediate Users**

- Modify parameters in the master config file (`script_params_master.yaml`)
- Edit the generated scripts before running them for fine-tuned control
- Explore per-module configuration options

**🔴 Advanced Users**

- Run individual PyAPS modules directly (e.g., `aps_rr.py`, `aps_rvs.py`, `aps_ferre.py`)
- Set debug parameters at the bottom of module files for detailed diagnostics
- Customize processing workflows and integrate with custom analysis pipelines

### The PyAPS Architecture

PyAPS for WEAVE is a critical component in the **end-to-end automatic data processing chain** for WEAVE, spanning from observation preparation through to data archiving. This means PyAPS must handle significant complexity:

**Why is PyAPS complex?**

- **Multiple data types**: Single exposures, stacks, superstacks across MOS and IFU modes
- **Evolving instrument**: Different data versions after instrument modifications and upgrades over time
- **Diverse science cases**: Processing everything from nearby stars and nebulae to high-redshift galaxies and QSOs
- **Unexpected scenarios**: Handling edge cases, partial failures, and non-standard configurations
- **Survey diversity**: Supporting dozens of WEAVE surveys with different scientific requirements

PyAPS for WEAVE uses an **automatic orchestration approach** that:

- **Handles L1 data complexities**: Accounts for WEAVE data model specifications, instrument configurations, and observing modes
- **Applies necessary corrections**: Automatically applies fixes and calibrations to raw L1 data before L2 processing
- **Manages dependencies**: Ensures correct execution order (e.g., RR before RVS, RVS before FERRE)
- **Centralizes data handling**: Uses the `APSOB` (APS Object) superclass for unified spectral data access
- **Adapts to context**: Automatically detects data characteristics and adjusts processing parameters

> **For deeper understanding**:
> - See [aps_utils.md](aps_utils.md) for detailed documentation on how `APSOB` is constructed and how it handles WEAVE data internally

### Scope of This Tutorial

**What this tutorial covers:**

- ✅ Basic installation and setup
- ✅ Running the pipeline with default/recommended parameters for MOS
- ✅ Understanding MOS-specific outputs and diagnostic plots
- ✅ Basic troubleshooting

**What this tutorial does NOT cover:**

- ❌ All available configuration parameters (hundreds of options exist!)
- ❌ Advanced customization and parameter tuning
- ❌ Module-specific deep-dives
- ❌ Integration with contributed software (CS) codes

> **This is a getting-started guide only.** For comprehensive documentation on individual modules, configuration options, and advanced usage, consult:
> - [aps_rr.md](aps_rr.md) - Classification and redshift estimation
> - [aps_rvs.md](aps_rvs.md) - Radial velocities and stellar parameters
> - [aps_ferre.md](aps_ferre.md) - Stellar atmospheric parameters
> - [aps_utils.md](aps_utils.md) - Core data handling (APSOB)
> - [aps_calib.md](aps_calib.md) - LSF and FWHM calibration
> - [README.md](../README.md) - Complete PyAPS overview

---

## 2. PyAPS Installation

### Prerequisites

- Python 3.8 - 3.12
- pip package manager
- Git
- Sufficient disk space (~50 GB for minimal templates, ~500 GB for full templates)

### Installation Steps

1. **Clone the PyAPS repository**

```bash
cd $HOME
git clone <repository-url> PyAPS
cd PyAPS
```

2. **Install PyAPS in development mode**

```bash
# Recommended: Create a virtual environment first
python3 -m venv $HOME/venv_WEAVE
source $HOME/venv_WEAVE/bin/activate

# Install PyAPS with all dependencies
python3 -m pip install -e .
```

3. **Verify installation**

```bash
python3 -c "import PyAPS; print(PyAPS.__version__)"
```

If you see a version number, the installation was successful!

### External Dependencies

For stellar analysis (FERRE), you'll need the FERRE binary:

```bash
cd <PYAPS_DIR>/externals
# Follow instructions in externals/README.md
```

---

## 3. Data Structure Setup

### Understanding PyAPS Directory Structure

PyAPS uses a specific directory structure to organize data, templates, and outputs. For testing purposes, you can create your own structure using the `PyAPS_local` directory, which is not tracked by git.

### Creating Your Test Data Structure

```bash
cd <PYAPS_DIR>
mkdir -p PyAPS_local
cd PyAPS_local
```

Create the following directory structure:

```
PyAPS_local/
├── PyAPS_data/
│   ├── CAL/           # Calibration files (LSF, FWHM from arc lamps)
│   │   ├── 20230512/  # Calibration date directories
│   │   ├── 20240808/
│   │   └── 20250909/
│   ├── CAT/           # External target catalogues (optional)
│   ├── CS/            # Contributed Software outputs (optional)
│   ├── L1/            # Input L1 fiber-level spectra
│   │   ├── 20240105/  # Multiple observation dates
│   │   │   ├── single_r3050123.fit    # Single exposure, red arm
│   │   │   └── single_r3050124.fit    # Single exposure, blue arm
│   │   ├── 20240808/  # Another night
│   │   │   ├── stack_3071430.fit      # Red arm (lower number)
│   │   │   ├── stack_3071431.fit      # Blue arm (higher number)
│   │   │   ├── stack_3071500.fit      # Different observation
│   │   │   └── stack_3071501.fit
│   │   └── 20250315/  # Yet another night
│   │       ├── superstack_r3120100.fit
│   │       └── superstack_r3120101.fit
│   ├── L2/            # Output L2 products (created automatically)
│   └── XML/           # WEAVE XML files (optional)
└── PyAPS_templates/   # Templates and grids
    ├── templates_RR/          # Redrock PCA templates
    ├── templates_ARC_RR/      # Redrock archetype templates
    ├── templates_RVS/         # RVSpecfit templates
    ├── templates_FR/          # FERRE grids (for stellar analysis)
    ├── templates_ExGal/        # pPXF templates (for extragalactic)
    └── [other template dirs]
```

Create these directories:

```bash
cd <PYAPS_DIR>/PyAPS_local

# Create data directories
mkdir -p PyAPS_data/{CAL,CAT,CS,L1,L2,XML}

# Create template directories
mkdir -p PyAPS_templates/{templates_RR,templates_ARC_RR,templates_RVS,templates_FR,templates_ExGal}
```

### About Each Directory

#### `L1/` — Input Data

Contains your WEAVE L1 fiber-level spectra, organized by observation date (YYYYMMDD format). Each nightobs subdirectory can contain one or more pairs of files.

**Important File Naming Convention:**

- **Higher OBSID = Blue arm** (e.g., stack_3071431.fit covers ~3800-5950 Å)
- **Lower OBSID = Red arm** (e.g., stack_3071430.fit covers ~5900-9280 Å)

**Examples:**

```
L1/
├── 20240105/  # First observation night
│   ├── single_r3050123.fit      # Red arm, single exposure
│   └── single_r3050124.fit      # Blue arm, single exposure
├── 20240808/  # Second observation night (multiple datasets)
│   ├── stack_3071430.fit        # Red arm (lower number)
│   ├── stack_3071431.fit        # Blue arm (higher number)
│   ├── stack_3071500.fit        # Red arm, different field
│   └── stack_3071501.fit        # Blue arm, different field
└── 20250315/  # Third observation night
    ├── superstack_r3120100.fit  # Red arm, superstack
    └── superstack_r3120101.fit  # Blue arm, superstack
```

You can have multiple observation nights, and each night can contain multiple pairs of files corresponding to different fields or stacking levels.

**Stacking levels:**

- `single_*.fit` - Single exposure (raw, no stacking)
- `stack_*.fit` - Stacked exposures
- `superstack_*.fit` - Super-stacked (multiple nights combined)

#### `L2/` — Output Data

PyAPS automatically creates this structure when you run the pipeline. You can leave it empty for now.

#### `CAL/` — Calibration Data

Contains wavelength calibration files and LSF (Line Spread Function) data. This is **highly recommended** but not mandatory for your first test run.

**Where to get calibration data:**

- Download from: https://camcead.ast.cam.ac.uk/weave/calibquery/search
- Organize by calibration date

**Important:** If you don't provide calibration data, PyAPS will use default LSF values, which are less accurate but sufficient for testing.

---

## 4. Downloading Required Templates

Templates are essential for running the pipeline. You need:

1. **Minimal templates** (for testing): ~10 GB
2. **Full templates** (for production): ~500 GB

### Downloading Minimal Templates

For your first run, download the minimal template set:

```bash
cd <PYAPS_DIR>/PyAPS_local/PyAPS_templates

# Download the minimal template archive
wget https://camcead.ast.cam.ac.uk/weave/downloads/PyAPS_templates.tar.gz

# Extract
tar -xzf PyAPS_templates.tar.gz
```

### Template Directory Contents

After extraction, you should have:

```
PyAPS_templates/
├── templates_RR/          # Redrock PCA templates for classification
├── templates_ARC_RR/      # Redrock archetype templates
├── templates_RVS/         # RVSpecfit templates
├── templates_FR/          # FERRE stellar atmosphere grids
└── templates_ExGal/        # pPXF templates (for extragalactic targets)
```

---

## 5. Configuring the Master Config File

The master configuration file controls all pipeline parameters.

### Copy the Master Template

```bash
cd <PYAPS_DIR>/PyAPS_local
cp <PYAPS_DIR>/configs/script_params_master.yaml ./script_params_local.yaml
```

### Edit the Configuration File

```bash
nano script_params_local.yaml
```

### Essential Parameters to Update

Replace `<PyAPS_DIR>` with your actual PyAPS installation path throughout the file (see IFU tutorial for complete list). Key parameters for MOS:

```yaml
# General PARAMS
PyAPS_DIR: '<PYAPS_DIR>/py/PyAPS'
PyAPS_RES: '<PYAPS_DIR>/PyAPS_local/PyAPS_data/L2'
PyAPS_CAL: '<PYAPS_DIR>/PyAPS_local/PyAPS_data/CAL'

# Template paths
templates_RR: '<PYAPS_DIR>/PyAPS_local/PyAPS_templates/templates_RR'
templates_ARC_RR: '<PYAPS_DIR>/PyAPS_local/PyAPS_templates/templates_ARC_RR'
templates_RVS: '<PYAPS_DIR>/PyAPS_local/PyAPS_templates/templates_RVS'
templates_FR: '<PYAPS_DIR>/PyAPS_local/PyAPS_templates/templates_FR'
templates_ExGal: '<PYAPS_DIR>/PyAPS_local/PyAPS_templates/templates_ExGal'

# Virtual Environment
use_venv: 'True'
venv_path: '$HOME/venv_WEAVE/bin/activate'

# FERRE
FERRE_EXE: '<PYAPS_DIR>/externals/ferre/bin/ferre.x'
```

**Pro tip:** Use find-and-replace: `<PyAPS_DIR>` → `<PYAPS_DIR>`

---

## 6. Understanding Your Input Data

### What are L1 Fiber-Level Spectra?

L1 fiber-level files are the output from the WEAVE CPS pipeline. They contain:

- Flux data for all fibers in the observation
- Wavelength calibration
- Inverse variance (error) information
- Target metadata (RA, Dec, target class, etc.)

### File Naming Convention

```
[prefix]_<OBSID>.fit
```

Where:

- `prefix` = `single`, `stack`, or `superstack`
- `OBSID` = WEAVE observation block ID

For WEAVE observations, you'll typically have **two files** per pointing:

- **Red arm**: **Lower OBSID** (e.g., `stack_3071430.fit`) - covers ~5900-9280 Å
- **Blue arm**: **Higher OBSID** (e.g., `stack_3071431.fit`) - covers ~3800-5950 Å

> **Important:** The OBSID numbering is critical for the pipeline to correctly identify which arm is which!

### Standard Two-Arm Processing

For this tutorial, let's assume you have two stack files from observation date 20240808:

```bash
cd <PYAPS_DIR>/PyAPS_local/PyAPS_data/L1

# Create date directory
mkdir -p 20240808

# Copy or move your data files here
# stack_3071430.fit (red arm - lower number)
# stack_3071431.fit (blue arm - higher number)
```

Your L1 directory should now look like:

```
L1/
└── 20240808/
    ├── stack_3071430.fit  # Red arm (lower OBSID)
    └── stack_3071431.fit  # Blue arm (higher OBSID)
```

---

## 7. Running aps_runner.py for MOS

`aps_runner.py` is the main entry point for the PyAPS pipeline. It performs several important functions:

1. **Health checks** - Validates your installation and file paths
2. **Script generation** - Creates bash scripts tailored to your data
3. **Dependency management** - Ensures correct execution order
4. **Optional execution** - Can run scripts immediately or let you run them manually

### Basic Command Structure

```bash
python3 aps_runner.py \
    --infiles <path_to_red_arm> <path_to_blue_arm> \
    --config_file <path_to_config> \
    [additional options]
```

### Your First MOS Run

Here's a complete example command:

```bash
# Activate your virtual environment first
source $HOME/venv_WEAVE/bin/activate

# Navigate to your PyAPS directory
cd <PYAPS_DIR>

# Run aps_runner
python3 py/PyAPS/aps_runner.py \
    --infiles <PYAPS_DIR>/PyAPS_local/PyAPS_data/L1/20240808/stack_3071431.fit \
              <PYAPS_DIR>/PyAPS_local/PyAPS_data/L1/20240808/stack_3071430.fit \
    --config_file <PYAPS_DIR>/PyAPS_local/script_params_local.yaml \
    --cat_list None \
    --aps_ids None \
    --wlranges None \
    --mod_wlranges True \
    --targsrvy None \
    --targclass None \
    --mask_aps_ids None \
    --area None \
    --mask_areas None \
    --headname stack_3071431__stack_3071430 \
    --hname_suffix None \
    --uapsid 01 \
    --log False \
    --aps_ovr True \
    --CS_ovr True \
    --run_L2 True \
    --run_CS False \
    --mp_MOS 1,1,1,1,1,1,1 \
    --CPS_path True \
    --hpc 0
```

### Understanding MOS-Specific Parameters

#### Target Selection (All Optional)

- `--aps_ids None`: Process specific fiber IDs (None = all fibers)
- `--targsrvy None`: Filter by survey code (e.g., 'WA,WS')
- `--targclass None`: Filter by target class (e.g., 'GALAXY,QSO')
- `--mask_aps_ids None`: Exclude specific fibers
- `--area None`: Process only a specific sky region (RA, Dec, A, B, angle)

#### Multiprocessing for MOS

The `--mp_MOS` parameter controls parallel processing for each module:

```bash
--mp_MOS 16,8,16,8,8,8,1
```

Order: `RR, RVS, FERRE, PPXF, EMI, LS, L2merge`

**Example breakdown:**

- RR (Redrock): 16 cores
- RVS: 8 cores
- FERRE: 16 cores
- PPXF: 8 cores
- EMI: 8 cores
- LS: 8 cores
- L2merge: 1 core (always serial)

For first run, use `--mp_MOS 1` (sets all to 1 core).

---

## 8. Understanding the Generated Scripts

PyAPS generates several scripts for you. For MOS mode:

### The Main MOS Scripts

#### 1. `01_RR.sh` - Classification and Redshift

**What it does:**

- Runs Redrock on all fibers
- Classifies targets (GALAXY, QSO, STAR, WD)
- Measures redshifts
- Generates `zbest_*.fits` and `zspec_*.fits`

**Runtime:** 10-30 minutes for ~500 fibers

#### 2. `01_RVS.sh` - Radial Velocities (Stellar)

**What it does:**

- Measures radial velocities for stellar targets
- Derives stellar parameters (Teff, log g, [Fe/H])
- Uses RR redshifts as initial guess
- Generates `rvs_*.fits`

**Runtime:** 20-60 minutes

#### 3. `01_FERRE.sh` - Stellar Parameters

**What it does:**

- Runs FERRE for detailed stellar atmospheric parameters
- Uses RVS parameters as initial conditions
- Generates `ferre_*.fits`

**Runtime:** 30-90 minutes

#### 4. `01_PPXF.sh` - Extragalactic Stellar Kinematics

**What it does:**

- Runs pPXF on GALAXY and QSO targets
- Measures stellar velocities and velocity dispersion
- Generates stellar population properties

**Runtime:** Variable (depends on number of galaxies)

#### 5. `01_EMI.sh` - Emission Lines

**What it does:**

- Fits emission lines for galaxies
- Measures line fluxes, widths, EWs
- Requires PPXF stellar continuum

**Runtime:** Variable

#### 6. `01_LS.sh` - Line Strengths

**What it does:**

- Measures Lick/IDS indices
- Spectral feature strengths

**Runtime:** Fast (5-15 minutes)

#### 7. `01_L2merge.sh` - Merge All Results

**What it does:**

- Combines all module outputs
- Creates final `*_APS.fits` file
- Adds all extensions and metadata

**Runtime:** Fast (<5 minutes)

### Execution Order

```
RR (runs first, independent)
  ↓
  ├──→ RVS (requires RR redshifts)
  │     ↓
  │     FERRE (requires RVS parameters)
  │
  └──→ PPXF (requires RR redshifts)
        ↓
        ├──→ EMI (requires PPXF continuum)
        └──→ LS (requires PPXF kinematics)

All modules → L2merge (waits for all)
```

---

## 9. Running the MOS Pipeline Scripts

### Step 1: Run Classification (RR)

```bash
cd <PYAPS_DIR>/PyAPS_local/PyAPS_data/L2/20240808/<OBID>/scripts

chmod +x 01_RR.sh
./01_RR.sh
```

**What to watch for:**

- All fibers processed
- Redshift solutions found
- Output files created: `zbest_*.fits`, `zspec_*.fits`

### Step 2: Check Classification Results

```bash
cd ..  # Go to L2 directory

# Quick check with Python
python3 << EOF
from astropy.table import Table
zb = Table.read('zbest_stack_3071431__stack_3071430.fits')
print(f"Total targets: {len(zb)}")
print(f"Galaxies: {sum(zb['CLASS'] == 'GALAXY')}")
print(f"QSOs: {sum(zb['CLASS'] == 'QSO')}")
print(f"Stars: {sum(zb['CLASS'] == 'STAR')}")
EOF
```

### Step 3: Run Stellar Analysis (RVS + FERRE)

For stellar targets:

```bash
cd scripts
./01_RVS.sh
./01_FERRE.sh
```

### Step 4: Run Extragalactic Analysis (PPXF + EMI + LS)

For galaxy/QSO targets:

```bash
./01_PPXF.sh
./01_EMI.sh
./01_LS.sh
```

### Step 5: Merge Results

```bash
./01_L2merge.sh
```

### Step 6: Verify Outputs

```bash
cd ..
ls -lh *_APS.fits
```

You should see the final merged product!

---

## 10. Understanding the Outputs

### Main Output Files

#### `zbest_*.fits` - Classification and Redshifts

**HDU 0: ZBEST table**

Key columns:

- `TARGETID`: Unique fiber identifier
- `Z`: Best-fit redshift
- `ZERR`: Redshift uncertainty
- `ZWARN`: Warning bitmask (0 = clean)
- `CLASS`: Classification (GALAXY, QSO, STAR, WD)
- `SUBCLASS`: Sub-classification
- `DELTACHI2`: χ² difference (quality metric)

#### `zspec_*.fits` - Spectra with Best-Fit Models

Contains observed spectra and Redrock best-fit templates.

#### `rvs_*.fits` - Radial Velocities

Stellar radial velocities and basic parameters.

#### `ferre_*.fits` - Stellar Parameters

Detailed stellar atmospheric parameters:

- `TEFF`: Effective temperature (K)
- `LOGG`: Surface gravity
- `FEH`: Metallicity [Fe/H]
- `ALPHAFE`: Alpha enhancement [α/Fe]

#### `*_APS.fits` - Final Merged L2 Product

Contains all results in one file with multiple extensions:

```
HDU 0: PRIMARY (header only)
HDU 1: ZALL (all redshift solutions)
HDU 2: RR (Redrock best results)
HDU 3: RVS (radial velocities)
HDU 4: FERRE (stellar parameters)
HDU 5: PPXF (galaxy kinematics)
HDU 6: EMI (emission lines)
HDU 7: LS (line strengths)
```

---

## 11. Diagnostic Plots and Quality Assessment

### Redrock Diagnostic Plots

Located in `figs/`:

```
RR_<TARGID>_<CNAME>_<APS_ID>.png
```

Shows:

- Observed spectrum (black)
- Best-fit template (colored by rank)
- Inverse variance
- Redshift and classification info

**What to check:**

- Template matches spectral features
- IVAR non-zero at key features
- ZWARN = 0 (or understand warnings)

### Quality Flags

**ZWARN values:**

- `0` = Clean fit
- `4` = Small ΔChi² (ambiguous)
- `8` = Low SNR
- `64` = Negative template coefficients
- `256` = Galaxy/star ambiguity

### Quick Quality Check

```python
from astropy.table import Table
zb = Table.read('zbest_*.fits')

# Check for warnings
print(f"Clean fits (ZWARN=0): {sum(zb['ZWARN'] == 0)}")
print(f"Problematic fits: {sum(zb['ZWARN'] > 0)}")

# Check redshift distribution
import matplotlib.pyplot as plt
plt.hist(zb['Z'][zb['CLASS'] == 'GALAXY'], bins=50)
plt.xlabel('Redshift')
plt.title('Galaxy Redshift Distribution')
plt.show()
```

---

## 12. Advanced Options

### Processing Specific Target Types

```bash
python3 py/PyAPS/aps_runner.py \
    --infiles ... \
    --targclass GALAXY,QSO \  # Only galaxies and QSOs
    ...
```

### Processing Specific Fibers

```bash
--aps_ids 1,50,100,200  # Only these fiber IDs
```

### Single-Arm Processing

Similar to IFU tutorial - set `join_arms: 'False'` in config.

### Multiprocessing for Production

```bash
--mp_MOS 16,8,16,8,8,8,1  # Speed up processing
--hpc 1                    # Run immediately
```

---

## 13. Troubleshooting

### Common MOS Issues

#### Problem: "No valid targets found"

**Possible causes:**

- All fibers filtered by targclass/targsrvy
- Bad data quality (all IVAR = 0)

**Solution:**

- Check filters: `--targclass None --targsrvy None`
- Inspect L1 file with ds9 or Python

#### Problem: Redrock fails

**Error:** `No templates found`

**Solution:**

- Verify template paths in config
- Check `templates_RR` and `templates_ARC_RR` exist

#### Problem: FERRE fails

**Error:** `FERRE executable not found`

**Solution:**

```bash
# Check FERRE installation
ls <PYAPS_DIR>/externals/ferre/bin/ferre.x

# If missing, compile FERRE (see externals/README.md)
```

#### Problem: Different results for same target

**Cause:** Using different wavelength ranges or calibrations

**Solution:**

- Always use `--mod_wlranges True` for consistency
- Ensure same calibration files

---

## Summary: Quick Reference Checklist

### First-Time Setup

- [ ] Install PyAPS (`pip install -e .`)
- [ ] Create PyAPS_local directory structure
- [ ] Download minimal templates
- [ ] Copy and edit script_params_local.yaml
- [ ] Update all `<PyAPS_DIR>` paths
- [ ] Set virtual environment path

### For Each Observation

- [ ] Copy L1 files to `L1/YYYYMMDD/`
- [ ] Run `aps_runner.py` with `--hpc 0`
- [ ] Verify scripts generated successfully
- [ ] Run scripts in order: RR → RVS/PPXF → FERRE/EMI/LS → L2merge
- [ ] Check diagnostic plots
- [ ] Verify `*_APS.fits` created
- [ ] Inspect results

---

## Appendix: MOS vs IFU Comparison

| Aspect | MOS Mode | IFU Mode |
|--------|----------|----------|
| Input data | Fiber-level (2D) | Cube (3D) |
| File format | `stack_*.fit` | `stackcube_*.fit` |
| Target count | Hundreds per field | 1-10 per field |
| Segmentation | Not needed | Required |
| Main modules | RR, RVS, FERRE | IFU_Prepare, IFU_ExGal, IFU_Gal |
| Spatial binning | None | Voronoi tessellation |
| Output products | Per-fiber | Per-patch |

---

**Congratulations!** You should now be able to process WEAVE MOS data from L1 to L2. For more advanced usage, consult the detailed module documentation. 🎉
