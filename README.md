# PyAPS - Python-based Advance Processing System for WEAVE

[![CI](https://github.com/camcead/PyAPS/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/camcead/PyAPS/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](doc/)
[![DOI](https://zenodo.org/badge/1394828963.svg)](https://doi.org/10.5281/zenodo.23042510)

## Overview

PyAPS is a Python-based platform for processing and analyzing WEAVE survey data, developed as part of the Science Processing and Analysis (SPA) system. It provides state-of-the-art modules in a distributed architecture for analyzing and visualizing WEAVE spectroscopic data in both Multi-Object Spectroscopy (MOS) and Integral Field Unit (IFU) modes.

**Developer**: WEAVE APS team  
**Version**: 2.0

## Features

- **Comprehensive spectral analysis** for stellar and extragalactic targets
- **Multi-mode support**: MOS, mIFU, and LIFU observing modes
- **Modular architecture** with core modules and contributed software integration
- **Distributed processing** with HPC support via SLURM
- **Interactive visualization tools** for L1 and L2 data products
- **Automated script generation** with configurable parameters (`aps_runner`)

## Installation

### Prerequisites

- Python 3.12 or newer (3.12 is the production target; 3.13 and 3.14 are tested too)
- `pip`; a Fortran compiler (`gfortran`) if a compiled dependency has no wheel for your platform
- Git
- [FERRE](externals/README.md) (only for the stellar-parameter module `aps_ferre`)
- SLURM (only if you want to submit jobs with `aps_runner --hpc 3`; everything also runs locally)

### Quick Start

```bash
git clone https://github.com/camcead/PyAPS.git
cd PyAPS

python3 -m venv ~/pyaps-venv && source ~/pyaps-venv/bin/activate   # or use a conda environment

python3 -m pip install -e .              # recommended: links to the source tree, finds configs/ automatically
# python3 -m pip install .               # regular install: also set PYAPS_CONFIGDIR (see below)
```

### Dependency versions

The package follows the newest released versions of its dependencies (numpy 2.5, scipy 1.18, pandas 3, astropy 8, matplotlib 3.11,
dash 4, plotly 7, ...). `requirements.txt` is the loose list (latest of everything) and `requirements-lock-20261006.txt` is the frozen
set that the test suite passed on (Python 3.12); to reproduce that environment exactly:
`pip install -r requirements-lock-20261006.txt && pip install --no-deps -e .` (add
`--extra-index-url https://download.pytorch.org/whl/cpu` on a Linux host without a GPU).

### Choosing what to install

PyAPS is the WEAVE processing pipeline: `aps_runner` and the scientific modules are its main purpose. The pipeline calls a number of
scientific packages (some of them installed from GitHub), so **to process data install it with the `pipeline` option**:
`pip install -e ".[pipeline]"`. A plain `pip install .` installs only the lighter set of packages that the interactive explorer and
the viewers need, which is enough to open and inspect products but not to run the processing. The other options are additions for
special uses (run from the repository root):

| Command | Adds | Needed for |
|---|---|---|
| `pip install .` (or `pip install -e .`), no options | numpy, scipy, matplotlib, pandas, astropy, regions, scikit-learn, dill, plotly, dash, kaleido, flask, werkzeug, itsdangerous | `aps-explorer`, `aps_l1_preview`, `aps_IFUviewer`, `aps_MOSviewer` |
| `pip install ".[pipeline]"` | astropy-healpix, specutils, photutils, reproject, astroquery, numba, emcee, sep, ppxf, powerbin, pyqtgraph, PyQt5, PyYAML, torch, desiutil, redrock, rvspecfit | the processing pipeline (`aps_runner`, `aps_rr`, `aps_rvs`, `aps_ferre`, `aps_ifu_*`, `aps_mosExGal`, ...) |
| `pip install ".[cs]"` | ptemcee, PyAstronomy | the contributed `aps_amy` module |
| `pip install ".[server]"` | gunicorn | running the explorer as a shared service ([Docker](#server--multi-user-deployment-docker)) |
| `pip install ".[performance]"` | pyspark, numba | optional speed-ups (`aps_rrlew`) |
| `pip install ".[dev]"` / `".[docs]"` | pytest, black, flake8, ... / sphinx, ... | contributing / building docs |
| `pip install ".[all]"` | everything above | |

Nothing outside this table is imported by PyAPS - a package that is not listed for the
extra you installed is simply not needed (e.g. no database driver is required).

> PowerBin (`powerbin`, used for IFU Voronoi binning) is licensed for non-commercial use
> and may not be redistributed without its author's permission; it is fetched from PyPI
> only when you install the `pipeline` extra.

### Environment variables

| Variable | Meaning | Default |
|---|---|---|
| `PYAPS_HOME` | Root of your PyAPS working tree (holds `configs/`, `externals/`, `CS/`, `PyAPS_local/`). Used to expand `${PYAPS_HOME}` in `configs/script_params.yaml` | the source checkout, else `~/PyAPS` |
| `PYAPS_PKG_DIR` | Directory holding the `aps_*.py` modules (`${PYAPS_PKG_DIR}` in `script_params.yaml`) | set automatically |
| `PYAPS_CONFIGDIR` | Directory holding the instrument configuration (`ExGal_configs`) and interpolator caches. **A regular (non-editable) `pip install` does not bundle it: copy the repository's `configs/ExGal_configs` somewhere and point this variable at it.** | the checkout's `configs/ExGal_configs` (source / `pip install -e .` only) |
| `PYAPS_RVS_TEMPLATES` | Directory of the RVS (rvspecfit) template library. The generated job scripts export it from the `templates_RVS` key of the `script_params` file in use; it wins over `template_lib` in `configs/rvs_config.yaml`, which in turn is followed by `$PYAPS_HOME/PyAPS_templates/templates_RVS` and `$PYAPS_HOME/PyAPS_local/PyAPS_templates/templates_RVS` | set by the job scripts |
| `PYAPS_DATA_DIR` | Start folder of the explorer's file browser | `$PYAPS_HOME`, then `~` |
| `PYAPS_CS_MAIL` | Contact e-mail written into the `CS_MAIL` FITS header keyword by the RR Lyrae contributed modules (`aps_rrlew`, `aps_rrlgv`) | empty |
| `PYAPS_TEST_DATA` | Root of real WEAVE data for the data-dependent tests (they skip when unset) | unset |

### Placeholders used in this documentation

Paths in the docs and in the demo blocks of the source files are written as placeholders -
substitute your own:

| Placeholder | Meaning |
|---|---|
| `<PYAPS_DIR>` | your PyAPS checkout (the directory containing `py/`, `configs/`, `bin/`) |
| `<PYAPS_DATA>` | directory holding your `L1/`, `L2/`, `CAL/`, `CAT/` trees |
| `<CAL_DIR>` | directory of calibration (`wave_*.fit`, LSF) files |
| `<FERRE_DIR>` | your FERRE installation |
| `<PYAPS_CONFIGDIR>` | value of `PYAPS_CONFIGDIR` |

### External Dependencies

FERRE and the other optional external tools: see [externals/README.md](externals/README.md).

### Data Resources

PyAPS grids and templates are available in two versions:

- **Standard Version**: Full set of grids and templates for complete functionality.
  For access to the standard version please contact the PyAPS admin at amolaei@ast.cam.ac.uk
- **Minimal Version**: Essential grids for basic operations and visualization.
  Available for download at [PyAPS_templates_minimal](https://camcead.ast.cam.ac.uk/weave/downloads/PyAPS_templates.tar.gz)

*Note: Grid downloads are not required for basic L1/L2 visualization features.*

---

## Core Modules

| Module | Description |
|--------|-------------|
| `aps_utils` | Core utilities and APSOB superclass for WEAVE data handling— [📖 Documentation](doc/aps_utils.md) |
| `aps_calib` | LSF interpolators from solar twilight flats (preferred) or arc lamp wave files (fallback) — both characterise the instrumental LSF as FWHM in Å — [📖 Documentation](doc/aps_calib.md) |
| `aps_rr` | Target classification and redshift estimation — [📖 Documentation](doc/aps_rr.md) |
| `aps_rvs` | Radial velocity and stellar atmospheric parameters — [📖 Documentation](doc/aps_rvs.md) |
| `aps_ferre` | Stellar atmospheric parameters using FERRE — [📖 Documentation](doc/aps_ferre.md) |
| `aps_mosGAL` | Extragalactic analysis for MOS mode |
| `aps_ifu_prepare` | Data preparation module for IFU modes (mIFU, LIFU) — [📖 Documentation](doc/aps_ifu_prepare.md) |
| `aps_ifu_ExGal` | Extragalactic analysis for IFU modes (mIFU, LIFU) — [📖 Documentation](doc/aps_ifu_ExGal.md) |
| `aps_ifu_Gal` | Galactic analysis for IFU modes (mIFU, LIFU) — [📖 Documentation](doc/aps_ifu_Gal.md) |
| `aps_runner` | Automated script generation and pipeline execution — [📖 Documentation](doc/aps_runner.md) |
| `aps_L2merge` | L2 output handling and merging tools — [📖 Data Model](doc/weave_datamodel_v8.md) |
| `aps_flags` | APS_FLAGS bitmask management |
| `aps_explorer` | Interactive L1/L2 data explorer (Dash web app) — see [Visualization Tools](#visualization-tools) below, or [📖 Complete Documentation](doc/aps_explorer.md) |
| `aps_make_joint_cube` | Stitched-arm cube creation |

### Controlling Modules

| Module | Description |
|--------|-------------|
| `aps_runner` | Automated script generation and pipeline execution — [📖 Documentation](doc/aps_runner.md) |

---

## Calibration — `aps_calib` (`aps_lsf` and `aps_fwhm`)

📖 **[Complete Documentation](doc/aps_calib.md)**

The `aps_calib` subsystem provides per-fiber, wavelength-dependent Line Spread Function
(LSF) interpolators used throughout the pipeline. Every analysis module that calls pPXF,
FERRE, RVS, or Redrock requires an accurate description of the instrumental resolution at
each wavelength and for each fiber. `aps_calib` provides that description through two
complementary modules. Both `aps_lsf` and `aps_fwhm` characterise the same quantity —
the instrumental LSF represented as FWHM in Angstroms — derived from different
calibration sources.

### Two calibration routes

**`aps_lsf` — preferred: LSF from solar twilight flats (on-sky, f/3)**

Reads pre-computed B-spline representations of the LSF derived from solar twilight
observations. Because solar twilight flats are on-sky observations they correctly
reproduce the f/3 focal ratio of the telescope, which differs from the f/7 focal ratio
of arc lamp calibrations and directly affects the LSF shape at the spectrograph.
Input files are named `lsf_<ARM><RES><BIN>_<MODE>.fits`
(e.g. `lsf_BLUEL11_LIFU.fits`).

**`aps_fwhm` — fallback: LSF from arc lamp wave files (f/7, focal-ratio mismatch)**

Fits smooth splines to the discrete arc-line FWHM measurements stored in the `wave_*.fits`
files produced by the CPS L1 pipeline. Used automatically when no solar twilight LSF
file is available for a given setup or calibration date.

### What the calibration provides

Both modules produce an interpolator object with an identical public API. Given a
wavelength (or wavelength array) and optionally a fiber number, the interpolator
returns the instrumental FWHM in Angstroms at that wavelength for that fiber:

```python
fwhm_angstrom = interpolator.get_fwhm(wavelength, specnum=fiber_id)
```

The returned function is what gets attached to each target in `APSOB` and passed into
pPXF as `LSF_Data`. For extragalactic targets, `apply_redshift_to_fwhm_corrected()`
converts the observed-frame LSF to rest-frame before fitting.

### Interpolation across arms

When data from two spectrograph arms are combined (e.g. WEAVE blue + red), the
interpolator automatically handles the gap or overlap between them. Within each arm
the LSF is represented by a smooth spline; across the inter-arm gap a linear
interpolation connects the two boundary values; outside the full wavelength range the
FWHM is held constant at the nearest measured value.

### Quick usage

```python
from PyAPS.aps_lsf import run_lsf_analysis

# Build interpolator from two-arm solar twilight LSF files
interpolator = run_lsf_analysis(
    file_input    = ["lsf_BLUEL11_LIFU.fits", "lsf_REDL11_LIFU.fits"],
    figdir        = "<PYAPS_DATA>/L2/figs",
    figname       = "lsf_lifu",
    overwrite     = False,     # load from cached pickle if available
    save_pickle   = True,
    replace_binned= True,      # substitute unbinned file if binned missing
)

# Evaluate global FWHM at a wavelength array
import numpy as np
waves = np.linspace(3800., 9280., 500)
fwhm  = interpolator.get_fwhm(waves)

# Evaluate for a specific fiber
fwhm_fiber = interpolator.get_fwhm(waves, specnum=234)

# Get full dictionary for pipeline use
lsf_dict = interpolator.get_interpolator_dict()
```

For the fallback arc-lamp route:

```python
from PyAPS.aps_fwhm import run_fwhm_analysis

interpolator = run_fwhm_analysis(
    file_paths  = ["wave_3100962_all.fit", "wave_3100961_all.fit"],
    figdir      = "<PYAPS_DATA>/L2/figs",
    figname     = "fwhm_combined",
    overwrite   = False,
    save_pickle = True,
)
```

Both modules cache their result to a `.dill` pickle file so that subsequent pipeline
runs load the interpolator in milliseconds rather than recomputing it. The cache path
is derived automatically from the input filenames.

For full details on file formats, interpolation strategy, smoothing, missing-fiber
filling, the interpolator dictionary structure, and diagnostic plots see
**[doc/aps_calib.md](doc/aps_calib.md)**.

---

## Visualization Tools

### 🔭 aps_explorer — Unified L1/L2 Explorer

📖 **[Complete Documentation](doc/aps_explorer.md)** — every feature in detail: the
spatial (Aladin) view, the 3D flux cube (wavelength-range scrolling, filled fibre
discs, camera persistence), all tabs per dataset kind, the Slit Explorer, CSV
export, the live log panel, and the full environment-variable reference for
server deployments. The rest of this section is a quick-start summary.

`aps_explorer.py` is the single Dash web app for interactively viewing both WEAVE L1
(raw/reduced per-fibre spectra) and L2 (`_APS.fits` analysis products) data — no need
to pick the right script by hand. Launch it from the command line, it starts a local
web server, then open the printed `http://localhost:<port>` URL in a browser. It can
also be pointed at a different dataset from inside the browser, without restarting the
process — a single "☰ Load dataset" button (always present, top-left) opens an
in-browser form covering both L1 and L2 loading, remembers whatever values were last
used, and lets you tweak a parameter and reload in place.

All three kinds share one spatial view: an embedded **Aladin Lite** sky panel showing
a click-to-select catalog overlay (one point per spaxel/fibre/target, coloured by
whichever quantity is currently shown — discrete colour buckets approximating a
continuous colourmap for IFU/L1, exact per-category colours for MOS), a colour legend
underneath it (a gradient bar with min/max for continuous quantities, swatches for
MOS's categories), DSS background imagery (on by default, its own
checkbox to turn it off), and an info box with the selected point's ID and RA/Dec. Clicking a point
selects it (updating the tabs and value tables) and recentres the view on its exact
coordinates; a "Go to bin/target ID" box does the same.

Every table anywhere in the app — L1 headers/metadata, L2 value tables (MOS
Class/Stellar/Galaxy, IFU Spaxel/Bin), the info panel — has its own "Export" button
(top-right of the table) to save it as CSV, exactly as currently shown (respects
whatever sorting/filtering/hidden columns are active).

Given a dataset, it detects and routes to the right view automatically:

- **L1** (`--infiles`/`--infiles_list`, or the L1 section of the load form — every
  `aps_l1_preview` CLI option is exposed, including `--caldir`/`--catdir` for
  LSF/FWHM diagnostics, and a one-click "load raw/unmodified L1 data" toggle):
  flux/inverse-variance spectral panels per arm, FWHM overview + per-fibre detail
  diagnostics (raw arc-line measurements), and a target metadata table, alongside
  the Aladin panel above.
- **L2 MOS/fibre-level** (`--outpath`/`--headname`, or a full `_APS.fits` path —
  covers true MOS observations *and* LIFU/MIFU data processed at individual-fibre
  level, auto-detected from the file's extensions/header, never its name): Redrock
  (all ranks) always, and Stellar (RVS/FERRE) / Galaxy (PPXF/EMI) tabs shown
  per-target as available — a single target commonly has both at once.
- **L2 IFU Voronoi-patch**, ExGal (PPXF/EMIPPXF/line-strength) or Gal (RVS/FERRE),
  same loading options as MOS above: spectral-fit tabs matching whichever module
  produced the data, an AoN-threshold-filtered emission-line overlay, a "Bin &
  Spaxel Data" panel (every column for every spaxel in the selected bin, plus every
  fitted parameter for that bin), and a **Processing History** tab — the
  pipeline-only target-detection table, segmentation map, and
  target-selection/Voronoi-binning diagnostic images that can't be regenerated live
  from the final merged `_APS.fits` (unlike the per-bin fits, which are).

For L2 IFU/MOS datasets specifically, an optional **grid of additional 2D map
panels** sits below the master map + spectrum row — pick a Rows × Columns
layout (up to 3×6) and each panel gets its own independent Map/Scale/Palette/
Min-Max controls, its own quantity fully decoupled from the master map's own.
Clicking a point in any panel (or the master map) updates every other open
panel's own highlight in sync. A whole layout — grid size and every panel's own
settings, never any data — can be saved to and reloaded from a JSON file. Works
identically standalone or under the server deployment below, no extra flag
needed either way; see the [Complete Documentation](doc/aps_explorer.md#additional-map-panels-l2-ifumos)
for the full feature reference.

**Quick Start (standalone, single-user):**
```bash
# L2 (outpath + headname)
python aps_explorer.py --outpath /path/to/results/ --headname observation_P0001 --port 8080
# L1
python aps_explorer.py --infiles stackcube.fit --port 8080
# or with no arguments — opens straight into the "Load dataset" form
python aps_explorer.py --port 8080
# then open http://localhost:8080 in a browser
```

If PyAPS is installed via `pip` (not run from a source checkout), a console
script does the same thing without needing the full path:
```bash
aps-explorer --infiles stackcube.fit --port 8080
```

#### Server / multi-user deployment (Docker)

The same codebase also runs as a shared service reachable by several people
at once — each browser gets its own isolated session (own dataset, own
selection, own log panel; see `py/PyAPS/aps_explorer_session.py`'s module
docstring for the design) rather than one shared global. This is **off by
default** — nothing above changes unless you opt in.

```bash
docker build -t pyaps-explorer .
docker run -d --name pyaps-explorer -p 8080:8080 \
    -v <PYAPS_DATA>:/data:ro \
    pyaps-explorer
curl http://localhost:8080/healthz   # {"status": "ok"}
```

- Bind-mount your real `caldir`/`catdir`/`configdir` and the data root read-only
  (`-v host_path:container_path:ro`) rather than baking data into the
  image — swap `<PYAPS_DATA>` above for wherever your data lives. Inside the container it is then the container-side mount point shown after the colon (`-v <PYAPS_DATA>:/data:ro`).
- Runs a single gunicorn worker (multiple threads for concurrency) — this
  is a deliberate design choice, not a temporary limitation: per-session
  state lives in server memory for performance (see the module docstring
  linked above), so more than one worker *process* would each hold a
  different copy of it. Tune the thread count with `-e PYAPS_GUNICORN_THREADS=<n>` (default 8);
  don't add `--workers`.
  To run under a specific user/group (e.g. to read a group-restricted data mount) build with
  `--build-arg PYAPS_UID=<uid> --build-arg PYAPS_GID=<gid>`.
- **No authentication is required by default** — standalone usage is only
  ever protected by however you reach the machine (e.g. an SSH tunnel); a
  server deployment reachable on a real network needs either the infra
  layer handling it (a reverse proxy doing TLS + basic auth/OAuth2-proxy,
  or a network policy) or the handoff mode described next.
- **Which files a session shows can only ever be set by a URL** (a
  weaveOR token, or the plain `?kind=l1&infiles=...` deep link when
  `REQUIRE_WEAVEOR_AUTH` is off) — never by typing a path into the
  sidebar form. Once a session has a dataset loaded, its "☰ Dataset &
  Settings" panel hides the actual file path entirely (not even as a
  disabled field — page source would still reveal it) and locks the
  file-selection fields, but every *processing* parameter (caldir/
  catdir/wlranges/sens_corr/vacuum/sky subtraction/...) stays fully
  editable and reloadable in place. Standalone usage is completely
  unaffected — the full, unrestricted form, exactly as always.
- **Idle sessions are killed automatically**, protecting a busy shared
  deployment from unbounded memory growth: a client-side timer (default
  10 minutes of genuine inactivity — no mouse/keyboard/click/scroll;
  `PYAPS_EXPLORER_IDLE_TIMEOUT_MINUTES`) frees a session's memory the
  moment a tab goes truly unused, rather than merely "open" — the log
  panel's own 700ms poll would otherwise keep a forgotten background tab
  looking active forever. `PYAPS_EXPLORER_SESSION_TTL_SECONDS` (default
  14400 = 4h) remains a passive backstop for the rarer case that JS never
  gets to run at all (disabled JS, a tab killed outright).
- `PYAPS_CONFIGDIR`: points `aps_utils.validate_and_set_configdir` at a
  real, writable, persistent config/cache directory — needed whenever the
  repo-bundled `configs/ExGal_configs` isn't shipped with a deployment (a
  Docker image built from a checkout without it, for example). This is
  also where LSF/FWHM interpolator cache pickles end up
  (`_pickledir`, derived from whatever `configdir` resolves to) — set it
  to writable, persistent storage, separate from your (likely read-only)
  data mount, e.g. `-v <PYAPS_CONFIGDIR>:<PYAPS_CONFIGDIR>
  -e PYAPS_CONFIGDIR=<PYAPS_CONFIGDIR>`.
- To run the multi-session code path outside Docker (e.g. to test it
  locally before deploying), pass `--multi-session` to `aps_explorer.py`/
  `aps-explorer` directly — don't combine it with `--infiles`/
  `--outpath`+`--headname`, since a CLI-preloaded dataset only seeds the
  transient default session, which no real browser session in
  multi-session mode ever sees.

#### Sharing one host across several projects

If this deployment is one of several tools living behind the same
front door (e.g. `hub.example.org/pyaps/` for this project,
`hub.example.org/otherproject/` for something unrelated on the same
host), mount the app under its own path with `PYAPS_EXPLORER_URL_PREFIX`:

```bash
docker run -d --name pyaps-explorer -p 8080:8080 \
    -v <PYAPS_DATA>:/data:ro \
    -e PYAPS_EXPLORER_URL_PREFIX=/pyaps/ \
    pyaps-explorer
```

Point your reverse proxy at the same path, unchanged — Dash itself
generates every internal asset/callback URL under the configured prefix,
so there's no rewriting to do:
```nginx
location /pyaps/ { proxy_pass http://<this-container>:8080/pyaps/; }
```
`/healthz` is a plain Flask route outside Dash's own routing and always
stays unprefixed regardless — add a second proxy rule for it only if
something other than the container's own `HEALTHCHECK` needs to reach it
through the proxy. Unset (the default, `"/"`): the app is mounted at the
host's own root, exactly as in the plain example above — nothing changes
unless you opt in. This has to be an environment variable, not a
`--url-prefix` CLI flag: it configures Dash's own app object, which is
built once at import time, before `aps_explorer.py`'s argument parser
ever runs — set it before the process starts, the same way
`PYAPS_EXPLORER_MULTI_SESSION` already works for `wsgi.py`.

#### Authenticated handoff from a trusted upstream app

If another app (e.g. weaveOR) already authenticates its users and knows
which survey/programme each one may access, it can hand a user off
directly into a specific dataset view — no second login, and the session
is then restricted to *only* that one dataset for the rest of the visit
(it can't be used to browse other data via the sidebar form) — see
`py/PyAPS/aps_explorer_auth.py`'s own module docstring for the full
design and why it's scoped this way. **Off by default.**

```bash
docker run -d --name pyaps-explorer -p 8080:8080 \
    -v <PYAPS_DATA>:/data:ro \
    -e PYAPS_EXPLORER_REQUIRE_WEAVEOR_AUTH=1 \
    -e PYAPS_EXPLORER_WEAVEOR_SECRET=<shared with the token-issuing app> \
    pyaps-explorer
```

The upstream app mints a short-lived signed token (`itsdangerous.
URLSafeTimedSerializer`, HMAC'd with the shared secret above) naming the
user, their allowed survey/programme codes, and the one dataset being
handed off, then redirects the browser to
`https://<explorer>/?token=<token>`. This flag implies `--multi-session`
(turning it on also turns that on, even if not passed separately) and
the server refuses to start without `PYAPS_EXPLORER_WEAVEOR_SECRET` set.
Equivalent local-testing flag: `--require-weaveor-auth`.

---

## Contributed Software (CS)

PyAPS supports integration with various contributed software packages from WEAVE science teams. For detailed information about available CS codes and their usage, please refer to [CS/README.md](CS/README.md).

Available CS modules include:
- `aps_amy` — AMY stellar analysis
- `aps_alfa_neat` — ALFA/NEAT processing
- `aps_space` — SPAce spectral analysis
- `aps_squeze` — SQUEzE quasar analysis
- `aps_rrlew` — RR Lyrae EW analysis
- `aps_rrlgv` — RR Lyrae radial velocity
- `aps_FESWI` — Feature equivalent width analysis

---

## Usage

### Interactive L1/L2 Explorer

```bash
# aps-explorer is available anywhere once PyAPS is pip-installed (no
# extras needed — see Installation above); from a source checkout,
# `python3 aps_explorer.py` (run from py/PyAPS/) works identically.
aps-explorer --infiles /path/to/l1/data.fit \
             --wlranges 3900.0,5950.0 5900.0,9200.0

aps-explorer --outpath /path/to/L2/results/ \
             --headname observation_P0001
```

See [Visualization Tools](#visualization-tools) above for details.

### Python API

```python
from PyAPS.aps_utils import APSOB

apsob = APSOB(l1_file_path)
spectra  = apsob.get_spectra()
metadata = apsob.get_metadata()
```

---

## Configuration

### Configuration: first-time setup

The repository ships **templates**, not installation-specific configuration. Before the first run you
**must copy each `*.example` file to its local name and fill in every value for your own installation**:

```bash
cp configs/script_params.yaml.example configs/script_params.yaml      # pipeline parameters (aps_runner --config_file)
cp configs/explorer.env.example       configs/explorer.env            # only for the explorer server / Docker
chmod 600 configs/explorer.env                                        # it holds a shared secret
python tools/check_config.py                                          # reports every <...> still unfilled
```

| Local file (git-ignored) | What to enter |
|---|---|
| `configs/script_params.yaml` | output / catalogue / calibration / XML folders (`PyAPS_RES`, `CS_RES`, `PyAPS_CAT`, `PyAPS_CAL`, `PyAPS_XML`), template and external-program locations, optional virtual environment (`use_venv`, `venv_path`), per-module processing defaults |
| `configs/explorer.env` | `PYAPS_EXPLORER_WEAVEOR_SECRET` (a long random secret), upstream app URL, default calibration / catalogue folders, session limits |

Placeholders: `<...>` is a value you must supply (`tools/check_config.py` lists every one left),
`${PYAPS_HOME}` is the root of your PyAPS working tree, `<env_suffix>` is empty for a production
environment and `_dev` for a development environment (data trees `L1_dev`, `L2_dev`, ... next to the
production ones). **The local files are ignored by git: never commit them or paste them into issues.**
Every key is explained in [doc/CONFIGURATION.md](doc/CONFIGURATION.md), which also has the safe steps for
migrating a host that used to edit a tracked `configs/script_params.yaml` in place. The public-repository
rules (no secrets, hosts or absolute paths anywhere in tracked files) are in
[doc/PUBLIC_HYGIENE.md](doc/PUBLIC_HYGIENE.md).

### Configuration files

Configuration files are located in the `configs/` directory:

- `script_params.yaml.example` — template of the main pipeline parameters (copy it to the local, git-ignored
  `script_params.yaml`, see above). Paths inside it use `${PYAPS_HOME}` /
  `${PYAPS_PKG_DIR}` (see [Environment variables](#environment-variables)), so it works unchanged
  from any checkout. Make further copies (e.g. `configs/script_params_mysite.yaml`, also ignored) and pass them with
  `--config_file` when you want different output directories, module settings or resources.
- `ExGal_configs/` — per-instrument-mode IFU/MOS extragalactic configuration and LSF settings.
- `rvs_config.yaml`, `weave_cls.json`, `APS_FLAGS.json`, ... — module-specific configuration.

---

## Documentation

- **[WEAVE Data Model v8.0](doc/weave_datamodel_v8.md)**: Complete reference for L1 and L2 data structure, file naming conventions, HDU layouts, and the full data flow from raw observations to science-ready products — **start here if you are new to WEAVE data**
- **[Calibration (LSF)](doc/aps_calib.md)**: LSF interpolation from solar twilight flats (preferred) or arc lamp wave files (fallback) — both modules characterise the instrumental LSF as FWHM in Å
- **[aps_explorer](doc/aps_explorer.md)**: the unified interactive L1/L2 Dash viewer — every feature (spatial view, 3D flux cube, tabs, CSV export, deployment) in full; see also the [quick-start summary](#-aps_explorer--unified-l12-explorer) below
- **[aps_runner](doc/aps_runner.md)**: script generation and SLURM submission internals — per-OB dependency chains for MOS/IFU/CS, output path resolution, ad-hoc test runs (`--test_mode`), full command reference
- **[Getting-started tutorials](doc/MOS_GETTING_STARTED_TUTORIAL.md)**: MOS, and [IFU](doc/IFU_GETTING_STARTED_TUTORIAL.md) walkthroughs

---

## Project Structure

```
PyAPS/
├── configs/              # Configuration files (script_params.yaml, ExGal_configs/, ...)
├── bin/                  # Command-line wrappers (aps_runner, aps_rr, aps_rvs, ...)
├── doc/                  # Documentation (one reference guide per module, data model, tutorials)
├── externals/            # External tools (FERRE, ...) - installed by you, see externals/README.md
├── tests/                # Test suite
├── py/PyAPS/             # Source code
├── CS/                   # Contributed software (installed by you, see CS/README.md)
└── PyAPS_local/          # Your local templates and data (never committed, see below)
```

---

## Running the pipeline (`aps_runner`)

📖 **[Complete Documentation](doc/aps_runner.md)** — the full generation/submission
internals (exact per-OB dependency structure for MOS/IFU/CS, output path resolution,
`sbatch_guard` reliability handling). The rest of this section is a quick-start summary.

`aps_runner.py` is the top-level entry point for the PyAPS pipeline. It reads a single
YAML config file, parses your command-line options, generates bash/SLURM scripts for
every analysis module, and optionally submits them. You never need to call the individual
module scripts (`aps_rr.py`, `aps_rvs.py`, `aps_ferre.py`, etc.) by hand — `aps_runner`
builds and runs them for you.

### Quick start

```bash
python aps_runner.py \
  --infiles $PYAPS_DATA/L1/20250630/stack_3095664.fit \
            $PYAPS_DATA/L1/20250630/stack_3095663.fit \
  --config_file configs/script_params.yaml \
  --cat_list None \
  --headname stack_3095664__stack_3095663 \
  --mod_wlranges True \
  --run_L2 True \
  --run_CS False \
  --hpc 1 \
  --mp_MOS 8 \
  --aps_ovr True
```

### Input parameters

**Required**

| Parameter | Type | Description |
|-----------|------|-------------|
| `--infiles` | list of paths | One or two L1 FITS files. For two-arm data supply both blue and red files in any order — the code reorders them automatically (blue first, then red). |
| `--config_file` | path | YAML config file. Controls all module-level defaults (template paths, pipeline flags, etc.). |
| `--cat_list` | path or `None` | Comma-separated list of external catalogue files. Usually `None` for standard runs. |

**Wavelength ranges**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--wlranges` | list | `None` | Explicit wavelength limits in `wmin,wmax` format, one entry per input file. Example: `--wlranges 3800.0,5925.0 5925.0,9380.0`. Takes precedence over `--mod_wlranges`. |
| `--mod_wlranges` | bool | `False` | If `True` and `--wlranges` is `None`, read wavelength limits from the config file using the key `MODE_RES_CAMERA` (e.g. `MOS_LR_BLUE`). Recommended for production runs. |

**Target selection (all optional)**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--aps_ids` | csv | `None` | Comma-separated list of specific fibre IDs to process. If `None`, all active fibres are processed. |
| `--targsrvy` | csv | `None` | Filter by survey code (e.g. `WS2023A1-022`). |
| `--targclass` | csv | `None` | Filter by target class (e.g. `GALAXY,QSO`). |
| `--mask_aps_ids` | csv | `None` | Fibre IDs to exclude from processing. |
| `--area` | 5 floats | `None` | Elliptical sky region to include: `RA_deg,DEC_deg,A_arcsec,B_arcsec,angle_deg`. MOS mode accepts circular only. |
| `--mask_areas` | list | `None` | One or more elliptical regions to exclude, same format as `--area`. Multiple regions separated by spaces. |

**Output and naming**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--headname` | str | auto | Base name for all output files. If `None`, generated automatically from the input filenames joined by `__`. |
| `--hname_suffix` | str | `None` | Appended to `headname` to label a specific run variant (e.g. `test`, `v2`). |
| `--uapsid` | str | `None` | Unique ID used as the SLURM job name. If `None`, `headname` is used. For a manual/ad-hoc run, see `--test_mode` below instead of setting this by hand. |

**Execution control**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--hpc` | int | `0` | Execution mode: `0` = scripts only, `1` = bash, `2` = kiko, `3` = SLURM. |
| `--run_L2` | bool | `True` | Generate and run the main L2 pipeline (RR, RVS, FERRE, PPXF, EMI, LS, merge). |
| `--run_CS` | bool | `True` | Generate and run the CS scripts (SQUEZE, FESWI, SPACE, AMY, ALFA-NEAT, RRLGV, RRLEW). |
| `--aps_ovr` | bool | `False` | Overwrite an existing final APS product. If `False` and the product exists, the run is aborted. |
| `--CS_ovr` | bool | `False` | Same as `aps_ovr` but for CS products. |
| `--CPS_path` | bool | `True` | Derive the output directory following the CPS recipe (recommended). If `False`, use the observing date from the FITS header. |
| `--log` | bool | `False` | Redirect all module stdout/stderr to per-module log files. |
| `--test_mode` | bool | `False` | Ad-hoc test run: forces a `TEST_`-prefixed job tag (auto-generated if `--uapsid` isn't given) and redirects output into `<PyAPS_RES\|CS_RES>/TEST/...` instead of the real per-OB directory - never collides with a real product. Track it with `squeue --name <tag>`. See [doc/aps_runner.md](doc/aps_runner.md#traceable-ad-hoc-test-runs). |

**Multiprocessing**

Each parameter accepts either a single integer (applied to all modules) or a
comma-separated list of integers (one per module in the order listed).

| Parameter | Modules (in order) | Default |
|-----------|-------------------|---------|
| `--mp_MOS` | RR, RVS, FR, PPXF, EMI, LS, L2merge | `1` |
| `--mp_LIFU` | IFU_Prepare, IFU_ExGal, IFU_Gal | `1` |
| `--mp_MIFU` | IFU_Prepare, IFU_ExGal, IFU_Gal | `1` |
| `--mp_CS` | SQ, FESWI, SPACE, AMY, AN, RRLGV, RRLEW | `1` |

For HR data the code automatically doubles all mp values. For stacking level 0
(single exposures) FERRE mp is halved. Example — give RR 16 CPUs, all others 8:

```bash
--mp_MOS 16,8,8,8,8,8,1
```

**Arms ratio**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--arms_ratio` | csv floats | `None` | Manual flux-correction factor per arm. Example: `1.0,0.83` applies 0.83 to the red arm. Overridden by `GR_ratio_LR`/`BR_ratio_LR` in the config when multiple files are present. |

**IFU-only parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--seg2d_white_images` | csv paths | `None` | Input images for 2D source extraction. |
| `--patch_file` | path | `None` | Pre-existing patch file containing extraction ellipses and class assignments. If `None`, IFU_Prepare generates it. |

### Config file (`script_params.yaml`)

The YAML config file controls everything that does not change between runs.

**Directory paths**

```yaml
PyAPS_DIR:        path to py/PyAPS/          # module scripts
PyAPS_RES:        path to L2 output root
PyAPS_CAT:        path to input catalogues    # optional
PyAPS_CAL:        path to calibration files   # optional
PyAPS_XML:        path to XML files           # optional
PyAPS_EXGALCONFIG: path to ExGal JSON configs
templates_RR:     path to Redrock PCA templates
templates_ARC_RR: path to Redrock archetype templates
templates_RVS:    path to RVSpecfit templates
templates_FR:     path to FERRE grids
templates_ExGal:  path to pPXF/ExGal templates
```

**Wavelength ranges** (used when `--mod_wlranges True`)

```yaml
MOS_LR_BLUE:  '3800.0,5950.0'
MOS_LR_RED:   '5900.0,9280.0'
MOS_HR_BLUE:  '4040.0,4650.0'
MOS_HR_GREEN: '4760.0,5480.0'
MOS_HR_RED:   '6050.0,6780.0'
LIFU_LR_BLUE: '3800.0,5950.0'
LIFU_LR_RED:  '5900.0,9280.0'
# etc.
```

Keys follow the pattern `MODE_RES_CAMERA` where MODE = `MOS`/`LIFU`/`MIFU`/`MOSLIFU`/`MOSMIFU`,
RES = `LR`/`HR`, CAMERA = `BLUE`/`GREEN`/`RED`.

**Arms ratio** (applied automatically, overrides `--arms_ratio`)

```yaml
GR_ratio_LR: '1.0'
BR_ratio_LR: '1.0'
GR_ratio_HR: '1.0'
BR_ratio_HR: '1.0'
```

**Per-module flags** (example for Redrock)

```yaml
nminima_RR:        '3'      # number of chi2 minima to explore
ntop_RR:           '3'      # top N redshifts to return
tellurics_RR:      'False'  # mask telluric bands (set False for extragalactic)
safe_mask_gaps_RR: 'True'   # mask CCD gaps from lookup table
vacuum_RR:         'True'   # output wavelengths in vacuum
fig_RR:            'True'   # generate spectral plots
overwrite_RR:      'True'   # overwrite existing RR output
```

Every module has its own block. Any flag can be changed in the config without
touching the command line.

### How the L2 pipeline runs (MOS mode)

The modules execute in a fixed dependency order. Each module waits for its
dependencies to finish before starting, and L2merge waits for all modules.

```
Stage 1 — independent (run immediately in parallel):
  RR   ─────────────────────────────────────────────────────────────┐
                                                                     │
Stage 2 — depends on RR:                                             │
  RVS  (requires RR redshifts for initial velocity guess) ──────┐   │
  PPXF (requires RR redshifts for de-redshifting)          ─────┼───┤
                                                                 │   │
Stage 3 — depends on RVS or PPXF:                                │   │
  FR   (requires RVS stellar parameters as initial conditions) ──┤   │
  EMI  (requires PPXF stellar continuum for subtraction)    ─────┤   │
                                                                 │   │
Stage 4 — depends on PPXF and EMI:                               │   │
  LS   (requires PPXF kinematics and EMI line fluxes)      ──────┤   │
                                                                 │   │
Stage 5 — depends on all previous stages:                        │   │
  L2merge ───────────────────────────────────────────────────────┴───┘
```

In SLURM mode each stage is submitted with `--dependency=afterany:<job_id>` (not
`afterok`) - a downstream stage starts once its prerequisite finishes, **regardless of
that prerequisite's exit code**. Failure handling/retry is left to the caller - nothing is
"held" at the SLURM level. Each `sbatch` call is
also followed by a check that a job ID actually came back before it's used as the next
stage's dependency (`sbatch_guard()` - see `doc/aps_runner.md`), so a failed submission
partway through the chain fails loudly instead of silently propagating a broken
dependency - and that failure is captured and turned into `aps_runner.py`'s own process exit code
(non-zero), so any script or scheduler that calls it can detect the failure.

### Common usage patterns

**Standard MOS production run**

```bash
python aps_runner.py \
  --infiles $PYAPS_DATA/L1/20250630/stack_3095664.fit \
            $PYAPS_DATA/L1/20250630/stack_3095663.fit \
  --config_file configs/script_params.yaml \
  --cat_list None \
  --mod_wlranges True \
  --run_L2 True --run_CS True \
  --aps_ovr False --CS_ovr False \
  --hpc 3 \
  --mp_MOS 16,8,16,8,8,8,1 \
  --mp_CS 4
```

**Test run on a subset of fibres (scripts only)**

```bash
python aps_runner.py \
  --infiles stack_3095664.fit stack_3095663.fit \
  --config_file configs/script_params.yaml \
  --cat_list None \
  --aps_ids 100,200,300 \
  --mod_wlranges True \
  --headname test_run \
  --hname_suffix v1 \
  --aps_ovr True \
  --hpc 0 \
  --mp_MOS 4 \
  --run_L2 True --run_CS False
```

**IFU (LIFU) run**

```bash
python aps_runner.py \
  --infiles stackcube_3123971.fit stackcube_3123970.fit \
  --config_file configs/script_params.yaml \
  --cat_list None \
  --mod_wlranges True \
  --headname LWVE_07172721_BR_L2 \
  --run_L2 True --run_CS False \
  --hpc 1 \
  --mp_LIFU 4
```

### Output structure

All outputs are written under `PyAPS_RES/<obsdate_or_L1_parent>/<obid>/`:

```
zbest_<headname>.fits       ← Redrock classification and redshifts
zspec_<headname>.fits       ← spectra with best-fit models
rvs_<headname>.fits         ← RVSpecfit stellar velocities
ferre_<headname>.fits       ← FERRE stellar parameters
<headname>_APS.fits         ← final merged L2 product
scripts/                    ← all generated bash and SLURM scripts
logs/                       ← per-module log files (if --log True)
figs/                       ← spectral plots (if fig=True in config)
```

### Tips

- Always set `--mod_wlranges True` for production runs — it reads the correct
  wavelength ranges for your setup from the config automatically.
- Set `--hpc 0` first to inspect the generated scripts before submitting.
- For a real test submission (`--hpc 3`, not just script generation), use
  `--test_mode True` rather than `--hname_suffix` - `--hname_suffix` alone still writes
  into the real per-OB output directory and shares the same overwrite risk as a real
  rerun; `--test_mode` redirects to a separate `TEST/` subtree and gives it a
  trackable tag automatically. See [doc/aps_runner.md](doc/aps_runner.md#traceable-ad-hoc-test-runs).
- For single-exposure data (stacking level 0) FERRE automatically uses the lighter
  3D grid and halves the CPU allocation — no action needed.
- HR mode automatically doubles all mp values — set `--mp_MOS` for the LR equivalent
  and the code scales up for HR.

---

### Directory Structure Requirements

The following directory structure is expected. Items marked *(optional)* are only
required for specific modules or features.

```
<PYAPS_DIR>/
├── py/PyAPS/aps_runner.py
└── configs/script_params.yaml

<data_dir>/
├── L1/
│   └── YYYYMMDD/
│       ├── filename_blue.fit
│       └── filename_red.fit
├── L2/                         ← created automatically by the pipeline
│   └── YYYYMMDD/
├── CAT/                        *(optional)* external target catalogues
│   └── *.fits
├── CAL/                        *(optional)* wavelength calibration files
│   └── YYYYMMDD/
│       └── *.fit
```

`CAT` and `CAL` are only needed when:
- `CAL` — you want fibre-specific LSF/FWHM resolution matrices (set `caldir` in config)
- `CAT` — you have external flag catalogues to pass via `cat_list`

---

## Contributing

Contributions are welcome - see [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup and
pull-request checklist, and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Release history:
[CHANGELOG.md](CHANGELOG.md).

## Support

- Bugs and questions: [GitHub issues](https://github.com/camcead/PyAPS/issues)
- Security problems: see [SECURITY.md](SECURITY.md) (private reporting)
- Contact: amolaei@ast.cam.ac.uk

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgments

PyAPS is developed as part of the WEAVE Science Processing and Analysis system. We acknowledge
the contributions from all WEAVE science teams and the broader astronomical community.

## Citation

If you use PyAPS in your research, please cite it — see [CITATION.cff](CITATION.cff)
for the current metadata. Every release is archived on Zenodo:

- **Cite this exact version (2.0):** [10.5281/zenodo.23042511](https://doi.org/10.5281/zenodo.23042511)
- **Cite PyAPS in general (all versions, always the latest):** [10.5281/zenodo.23042510](https://doi.org/10.5281/zenodo.23042510)

```bibtex
@software{pyaps_2_0,
  author    = {Molaeinezhad, Alireza and {WEAVE APS Team}},
  title     = {PyAPS: Python-based Advance Processing System for WEAVE},
  version   = {2.0},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23042511},
  url       = {https://doi.org/10.5281/zenodo.23042511}
}
```
