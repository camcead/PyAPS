# Changelog

All notable changes to PyAPS are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/) (`MAJOR.MINOR`, patch releases as `MAJOR.MINOR.PATCH`).

## Unreleased

### Fixed (line strengths: every index was NaN and the broadening to the LIS resolution never ran, OB 26133, 2026-11-06; doc/aps_ifu_ExGal.md section 17)
- **`lsindex_spec.sum_counts` returned a 1-element array** when a band edge falls inside a pixel (always, for real bands); on NumPy 2.5 `index[k] = index0[0]` then raised `setting an array element with a sequence`, `run_ls` caught it (`lsindex measurement failed`) and wrote NaN. Result on OB 26133 (IFU_ExGal, 4 patches, and the MOS LS product): **0 finite values out of ~59,000 index and error cells**. Fix: the fractional-pixel terms are summed to a scalar. Test: `tests/test_lsindex_scalar.py`.

### Fixed (IFU seg2d: 226 noise patches on OB 26133, 2026-11-06; doc/aps_ifu_seg2d_robust.md)
- **Robust sky and noise for all seg2d steps** (`aps_ifu_seg3d.robust_background`, `prepare_white`): the white-light image of a WEAVE LIFU cube is a mosaic of single-fibre footprints; plain `sep.Background` was biased by two very bright sources (back 3111, rms 7428 against a clipped sky of 8527 / 9530), the auto threshold sat 0.37 sigma above the true sky, and the sign-flip purity check, which used the same background, recommended the lowest threshold: 148 detections, 226 patch products, an 8.4 h IFU run, almost all noise. PSF estimate, purity calibration, `object_extract` and `sep_catalog` now share the robust helper.
- **Calibration guards**: threshold grid extended to 8, a threshold needs purity >= 0.9 and an excess of 3 detections over the sign-flipped image, a failed calibration is reported (`calibration_ok`, WARNING) instead of silent. `calibrate_sep_purity` keeps its return value (wrapper of `calibrate_sep_purity_ex`).
- **`minarea` floor of half a fibre footprint** on fibre-mosaic white-light images (11 px for LIFU); other images unchanged.
- **New option `--seg2d_max_targets`** (default 150, 0 = off) keeps the brightest detections and warns; IFU_PR prints `SEG2D_SUMMARY` and writes `<headname>_seg2d_summary.json`.
- **Patch reporting in IFU_ExGal / IFU_Gal** (`aps_ifu_patch_report.py`): `IFU_PATCH_SUMMARY` line and `<headname>_<stage>_patches.json` (attempted / written / no_valid_spaxels / failed), warning below 50% yield, job fails when >= 10 patches were attempted and none produced a product (before: COMPLETED with 222 of 226 patches lost).
- Not changed on purpose: the GALAXY/STAR class split of `update_patch_table` (by design, paper Table 5); the legacy `aps_ifu_v0.py`.
- Real data, same code: OB 26133 148 -> 8 detections; 24311 123 -> 114, 25931 47 -> 39, 24316 18 -> 23, 24282 3 -> 1, 24266 0 -> 0. Tests: `tests/test_seg2d_robust.py`, `tests/test_aps_common_args.py` (flag). Docs: `doc/aps_ifu_seg2d_robust.md`, `doc/aps_ifu_prepare.md`, the tutorial defaults (the tutorial still listed 2.5 / 300 while the shipped default is None).

### Fixed (Explorer: the DSS background did not show, 2026-11-06)
- `aps_explorer.py`: Aladin Lite was given the bare CDS id `P/DSS2/color`, which it resolves and loads from CDS (`alasky.cds.unistra.fr`) only. CDS resets HTTPS connections from the IoA servers and from other networks, so the map stayed grey on "DSS colored loading..." for ever (reproduced with the vendored aladin.js 3.8.2 in headless Chromium). The browser now probes **ESA Sky's mirror of the same HiPS** (`https://skies.esac.esa.int/DSSColor`) first, then CDS (`https://alasky.cds.unistra.fr/DSS/DSSColor`), and gives Aladin the first source whose `properties` file answers within 4 s; if none answers it uses the old id (the previous behaviour). The chosen source is logged once (`PyAPS: DSS background source: ...`). Tested in headless Chromium (ESA reachable: DSS tiles load; nothing reachable: previous behaviour, no new errors). `doc/aps_explorer.md`, `assets/vendor/README.md`, test `test_dss_background_source_is_chosen_in_the_browser_with_a_fallback`.

## [2.0] publication snapshot - 2026-10-09 (tag `v2.0-rasti`)

The version label stays 2.0 (the version cited in the paper). The earlier public release, tag `v2.0` (commit 1684edf, 2026-09-29, Zenodo 10.5281/zenodo.23042511), is kept unchanged; this snapshot supersedes it and additionally contains everything listed below, in particular the corrected Redrock per-arm implementation with its tests, the aperture-convention clarification and `gal_aperture_factor`.

### Added
- Version control of the template sets (`py/PyAPS/aps_templates.py`, `tools/pyaps_templates.py`, `pyaps templates ...`, `doc/TEMPLATES.md`, `tests/test_aps_templates.py`): every set (`full`, `mini`) carries a `TEMPLATES_MANIFEST.json` with set, calendar version, layout, tree digest, per-folder digests and (small sets) per-file SHA-256; `check`/`status` report match / older / newer / modified / unstamped / incompatible / missing, `verify`, `diff`, `manifest` and `stamp` build and compare manifests (stamp is a dry run unless `--yes`), `install --set mini` downloads, verifies and installs a released mini set. The script generator and `tools/check_env.py` read the manifest at start-up and warn (stop only on a layout mismatch; `PYAPS_TEMPLATES_CHECK=off|warn|major|strict`). The explorer shows the template set under its version and serves `/templatesz`. Standard library only; nothing is written by the start-up check.

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

### Fixed (interactive mask tool)
- `aps_ifu_utils.add_mask_region` (the `mask <ra> <dec> <radius_arcsec>` command of the patch-table editor) stored the
  typed radius as the full axis length, so the masked radius was half of what was typed. It now stores the diameter, like
  the Gaia mask rows, so the typed radius is the real masked radius. Only mask rows added by hand with that command are
  affected: they masked half the intended radius.

### Added (Galactic aperture factor)
- `gal_aperture_factor` (CLI `--gal_aperture_factor`) for the Galactic IFU route: an explicit factor applied to both
  full aperture axis lengths. The default (`None`) keeps the old behaviour, 0.5 for patch-file rows and 1.0 for a
  `patch_array`, so existing results are unchanged; set a number to make both routes select the same spaxels.

### Documentation
- `doc/aps_rr.md` states where the per-arm solve is active (coarse scan and per-camera archetype solve; Redrock's fine scan keeps the joint solver), what the returned coefficients mean, and removes the unsupported claims of independence from IVAR scaling and of removing a joint-fit bias.

### Documentation (README)
- The README now says plainly that PyAPS is the processing pipeline and that processing needs the `pipeline` installation option; a plain install is the lighter explorer and viewer set. (Whether the pipeline should become the default installation is open for the next release.)

### Changed
- Archived on Zenodo as a new version of the same record: version DOI 10.5281/zenodo.23250780 (2.0, publication snapshot, tag `v2.0-rasti`), concept DOI 10.5281/zenodo.23042510 (all versions); the earlier deposit 10.5281/zenodo.23042511 (tag `v2.0`) is preserved.
  `CITATION.cff`, the README (badge, citation block, BibTeX) and `.zenodo.json` point to the new version; "Advanced Processing System" corrected in all of them.

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
