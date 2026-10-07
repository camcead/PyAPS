# Changelog

All notable changes to PyAPS are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/) (`MAJOR.MINOR`, patch releases as `MAJOR.MINOR.PATCH`).

## Unreleased

### Security / public hygiene
- New guard `tools/check_public_hygiene.py` (+ `tests/test_public_hygiene.py`, pre-commit hook, CI step) that fails on secrets, internal hosts/IPs, personal account names, e-mail addresses and absolute machine paths in tracked files, and applies a stricter check to module headers and `__main__` demo blocks. Rules, placeholders and allow-list: `doc/PUBLIC_HYGIENE.md`.
- Docs, README, examples, tests and the demo blocks of the `aps_*.py` modules use placeholders (`<PYAPS_DATA>`, `$PYAPS_DATA`, `<night>`, `<runid>`, `<obid>`) instead of absolute paths and real run identifiers.
- `aps_squeze` finds its default SQUEzE data files through `PYAPS_CS_DIR` / `PYAPS_HOME` / the source checkout instead of a hard-coded path. The contributor e-mail written to the `CS_MAIL` FITS keyword by `aps_amy`, `aps_squeze` and `aps_alfa-neat` now comes from `PYAPS_CS_MAIL_AMY` / `PYAPS_CS_MAIL_SQUEZE` / `PYAPS_CS_MAIL_ALFA` (or `PYAPS_CS_MAIL`) and is empty when unset.

### Changed (dependencies, Python)
- Rebuilt on the newest released versions of every dependency (numpy 2.5, scipy 1.18, pandas 3.0, astropy 8.0, matplotlib 3.11, scikit-learn 1.9, plotly 7.1, dash 4.4, kaleido 1.5, flask 3.1, numba 0.68, gunicorn 26, ...). Python 3.12 is now the minimum (3.12 and 3.13 in CI; 3.14 also passes); the Docker image uses `python:3.12-slim`. Dependency floors in `pyproject.toml` follow the new versions; `requirements.txt` (loose) and `requirements-lock-20261006.txt` (frozen, tested) are new. `setuptools>=77` and an SPDX `license = "MIT"` in the packaging metadata.
- `ssppop_fitting.py` imports `scipy.spatial.Delaunay` instead of the deprecated `scipy.spatial.qhull` module; `aps_amy.py` uses timezone-aware `datetime.now(timezone.utc)` (`utcnow()` is deprecated in 3.12).

### Fixed (dependency upgrade)
- Regular expressions and LaTeX labels written as plain string literals (`'\s*#'`, `'$\pm$'`, ...) now escape the backslash: Python 3.12 warns about the invalid escape sequences and a later version will make them syntax errors (`aps_rrlew.py`, `lsindex_spec.py`, `MOSExGalLS.py`, `IFUExGalLS.py`, `aps_ifu_prepare.py`, `aps_ifu_v0.py`, `aps_space.py`). The strings themselves are unchanged.
- `ExGalPrepare.read_voronoi_fits_table`: the retry loop never left the loop after a successful open and referred to an undefined `delay`; it now breaks on success and waits 1 s between attempts. `aps_cubepreview.py` imports `read_ascii_patchfile`; `ssppop_fitting.py` imports its optional `corner`, `joblib` and `h5py` where they are used (they were undefined names).
- Tests: ten explorer tests (five of them only run with real WEAVE data) asserted the old `display:block/none` behaviour of the sidebar, the old six-argument signature and three-value return of `update_map_mode`, the old callback id, and passed a dict to `aps_MOSviewer._color_by_values`; they now follow the code (sliding drawer via `transform`/`visibility`, the nine-argument callback, the `l2-info-table` CSV export, the `AppState` argument). An autouse fixture resets the explorer state between test files (the layout test failed in a full run after an earlier file loaded a dataset), and the token-tamper test changes a character inside the signature (altering the last base64 character sometimes decoded to the same signature, so it failed at random).

### Fixed
- L2 jobs on a host running Python 3.12 with NumPy 2.5: `ExGalPrepare.prepare_spec_file` (MOS PPXF) and `voronoi_binning` stored a size-1 array (`np.mean(..., axis=0)` of an `(npix, 1)` slice) in a scalar element, which NumPy 2 rejects with "setting an array element with a sequence"; both now store the scalar mean of the bin.
- RVS jobs found the template library only through `template_lib` of `configs/rvs_config.yaml` and ignored the `templates_RVS` key of the `script_params` file ("Filename .../ccf_*.h5 does not exist"). The generated job scripts now export `PYAPS_RVS_TEMPLATES` from `templates_RVS`; `configs/rvs_config.yaml` has `template_lib: '${PYAPS_RVS_TEMPLATES}'`; `aps_rvs` and `aps_ifu_rvs` use `$PYAPS_RVS_TEMPLATES`, else a real `template_lib`, and stop with an error naming the configuration key when the directory is not set or does not exist (no fallback to any other location). **Action for existing hosts:** make sure `templates_RVS` of the local `script_params*.yaml` is correct; jobs whose scripts were generated before this change have no export and stop with that error until they are regenerated.
- `aps_ferre` exits at once with a clear message when the RVS output it needs does not exist (the FR job is submitted `afterany` the RVS job, so it also starts when RVS failed) instead of a `FileNotFoundError` traceback from astropy after loading the L1 data.
- IFU galaxy jobs (`ifu_Gal_prepare`): a cached LSF/FWHM interpolator (`*.dill`) written under another Python minor version (e.g. 3.11) unpickled without an error and then failed on its first call with `SystemError: error return without exception set` (dill stores the closures as bytecode), so every galaxy target failed. `aps_lsf.run_lsf_analysis` and `aps_fwhm.run_fwhm_analysis` now call the cached global interpolation function once after loading and rebuild the cache when that fails or when its Python stamp (`_cache_python`, new) differs.
- The official dev/prod code and the tracked configuration never read `PyAPS_local` (a personal, untracked scratch area): `configs/script_params.yaml.example` now defaults to `${PYAPS_HOME}/PyAPS_data` and `${PYAPS_HOME}/PyAPS_templates` (both git-ignored directories or symlinks), and `tests/test_no_pyaps_local_in_official_code.py` fails when tracked code or config mentions it (docs prose, README, CHANGELOG, ignore files allowed).
- Failed SLURM jobs are visible in the runner log: with the new optional `runner_log` key of the `script_params` file (a file, 'None' = off), a stage that ends with a non-zero exit code appends `[time] ERROR: SLURM job <id> <name> failed with exit code <n>; stderr: <path of its .err>` to it, and the MOS L2merge stage (which runs after all other stages, whatever their outcome) also reports the jobs of its OB that were TIMEOUT / CANCELLED / OUT_OF_MEMORY / NODE_FAIL (`aps_job_report.py`). The `--dependency=afterany` chains are unchanged.
- `tests/test_l2_job_failures.py` covers these failures with small synthetic inputs.
- Redrock per-arm chi2 modification (`aps_rr.py`): per-camera archetype columns are solved on, and averaged over, the arms in which they are non-zero (the stored Legendre coefficients were diluted by 1/n_arm; reported chi2 and rankings were unaffected); an arm to which the template does not contribute now adds its weighted flux squared to the chi2, as in the upstream joint fit, instead of discarding the trial redshift.
- `tests/test_aps_common_args.py` reads only the option lines of `--help`, skips scripts whose optional dependencies are missing, and lists the flags added since the original snapshot.

### Added
- `--rr_solver perarm|joint` (default `perarm`, unchanged behaviour) to select the per-arm or the upstream joint multi-arm solve; `tests/test_rr_perarm_patch.py` (synthetic tests, including the scope of the modification).

### Fixed (aperture convention)
- Aperture definition made consistent and documented. `A_world`/`B_world` (patch tables), the `area`/`mask_areas`
  entries and `make_patch_array(a_arcsec, b_arcsec)` are FULL axis lengths (diameters), the `width`/`height` of the
  extraction ellipse, as the Gaia mask rows, the SEP rows and the MOS circular selection already assumed. Docs and
  labels that called them semi-axes (`doc/aps_ifu_prepare.md`, the IFU tutorial, the cube preview
  printout, the `make_patch_array` docstrings) now say full axis. A single helper, `aps_utils.aperture_sky_region`,
  builds the region and is covered by aperture-boundary tests (`tests/test_aperture_convention.py`).
  **No numerical behaviour changed**: patch-table products are unaffected; `make_patch_array(10, 6)` has always
  extracted within 5" and 3" of the centre, and the seg3d merged-target aperture setting
  (`seg3d_merge_aperture_arcsec`, default 2.0) has always been a diameter (1.0" extraction radius). Past L2 products
  therefore need no reprocessing; read those numbers as diameters.

### Added (Galactic aperture factor)
- `gal_aperture_factor` (CLI `--gal_aperture_factor`) for the Galactic IFU route: an explicit factor applied to both
  full aperture axis lengths. The default (`None`) keeps the old behaviour, 0.5 for patch-file rows and 1.0 for a
  `patch_array`, so existing results are unchanged; set a number to make both routes select the same spaxels.

### Documentation
- `doc/aps_rr.md` states where the per-arm solve is active (coarse scan and per-camera archetype solve; Redrock's fine scan keeps the joint solver), what the returned coefficients mean, and removes the unsupported claims of independence from IVAR scaling and of removing a joint-fit bias.

### Documentation (README)
- The README now says plainly that PyAPS is the processing pipeline and that processing needs the `pipeline` installation option; a plain install is the lighter explorer and viewer set. (Whether the pipeline should become the default installation is open for the next release.)

### Changed
- Archived on Zenodo: version DOI 10.5281/zenodo.23042511 (2.0), concept DOI 10.5281/zenodo.23042510 (all versions).
  Added to `CITATION.cff` and the README (badge, citation block, BibTeX).

### Known limitations
- A regular (non-editable) `pip install` does not bundle `configs/ExGal_configs`; set `PYAPS_CONFIGDIR`
  or use `pip install -e .` from a clone.

## [2.0] - 2026-09-29

First public release of PyAPS as a standalone, installable package.

### Added
- Public repository with installation instructions per use case (`pip install .` for the explorer,
  extras `[pipeline]`, `[cs]`, `[server]`, `[performance]`, `[dev]`, `[docs]`) and a table of exactly
  which packages each extra brings in.
- Environment variables for site-independent configuration: `PYAPS_HOME`, `PYAPS_PKG_DIR`,
  `PYAPS_CONFIGDIR`, `PYAPS_DATA_DIR`, `PYAPS_TEST_DATA`, `PYAPS_CS_MAIL`.
- `configs/script_params.yaml` written with `${PYAPS_HOME}` / `${PYAPS_PKG_DIR}` so it works unchanged
  from any checkout.
- Generic multi-user explorer `Dockerfile` with configurable UID/GID and thread count
  (`PYAPS_GUNICORN_THREADS`).
- Optional extension hook: separately installed add-on packages can contribute modules to the
  `PyAPS` namespace through the `pyaps.extensions` entry point. Nothing changes when none is installed.
- Test guard `tests/test_internal_imports.py` (every intra-package import resolves inside the tree)
  and a shared `tests/conftest.py` that skips data-dependent tests when no WEAVE data is available.
- Project files: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, issue and pull-request
  templates, continuous integration, release workflow, dependency-update configuration.

### Changed
- Version stamped into all APS products (`aps_constants`) is now **2.0**.
- Minimum supported Python is **3.9**.
- Documentation and demo code use placeholders (`<PYAPS_DIR>`, `<PYAPS_DATA>`, `<FERRE_DIR>`, ...)
  instead of machine-specific absolute paths.
- The explorer's upstream-app handoff no longer has a built-in redirect URL; set
  `PYAPS_EXPLORER_WEAVEOR_URL` to enable it.
- The RR Lyrae contributed modules read the `CS_MAIL` header value from `PYAPS_CS_MAIL`
  instead of a hard-coded address.
- `TESTING.md` trimmed to the parts useful to any user.
- Dependency lists now reflect what the code imports; unused packages were removed from the extras.

### Removed
- Obsolete conda installation guide (`doc/PyAPS_conda_install.txt`); use the installation section of the README.
- Site-specific operational tooling, scratch/debug code and internal documents moved out of this
  repository (they were never part of the scientific pipeline).
- Tracked symlinks to local data volumes.

## [1.9] - 2026-08-24
Last version developed in the previous single repository.
