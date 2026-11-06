# Changelog

All notable changes to PyAPS are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/) (`MAJOR.MINOR`, patch releases as `MAJOR.MINOR.PATCH`).

## Unreleased

### Fixed (line strengths: every index was NaN and the broadening to the LIS resolution never ran, OB 26133, 2026-11-06; doc/aps_ifu_ExGal.md section 17)
- **`lsindex_spec.sum_counts` returned a 1-element array** when a band edge falls inside a pixel (always, for real bands); on NumPy 2.5 `index[k] = index0[0]` then raised `setting an array element with a sequence`, `run_ls` caught it (`lsindex measurement failed`) and wrote NaN. Result on OB 26133 (IFU_ExGal, 4 patches, and the MOS LS product): **0 finite values out of ~59,000 index and error cells**. Fix: the fractional-pixel terms are summed to a scalar. Test: `tests/test_lsindex_scalar.py`.
- **`gaussian_filter1d() got an unexpected keyword argument 'truncate'`** (hundreds of WARNINGs per IFU_ExGal / MOS LS run; 14 in OB 26133 IFU_ExGal, about half of the 553 MOS bins): `IFUExGalLS.py` and `MOSExGalLS.py` called `ppxf.ppxf_util.gaussian_filter1d(spec, sigma, truncate=, mode=)`; that pPXF function (same signature in pPXF 9.5.0 on both pods, and in every pPXF version) is `(spec, sig)`, a deprecated per-pixel-sigma convolution, and takes neither argument. The `TypeError` was caught, a WARNING printed and the spectrum was kept **unbroadened** (FWHM_FLAG = 3), so the line-strength indices of every bin that needed broadening were measured at the original resolution, not at the LIS 8.4 A system. New `py/PyAPS/aps_ls_broaden.py` (`broaden_variable_sigma`, NumPy only, +-4 sigma kernel of the local width, edge pixel continued; independent of the pPXF and SciPy versions) replaces the call in both modules. The former 3 / 2 sigma "adaptive truncate" tiers for wide kernels (they were never executed) are not carried over: they would have narrowed the Gaussian by up to 12 %. Tests: `tests/test_ls_broaden.py` (matches scipy for a constant sigma, FWHM follows sigma per pixel, NaN stays local, both modules call the helper with arguments it accepts). Doc: `doc/aps_ifu_ExGal.md` (FWHM_FLAG table).
- **Width of the extra Gaussian was computed for the wrong pixel grid** (found while testing the fix above; separate commit): `sigma = FWHM_dif / wave * c / 2.355 / velscale` is in pixels of the LOG grid, but the spectra are rebinned to a LINEAR grid (`log_unbinning`) before the broadening. On the OB 26133 grid (0.3356 A/pixel, 24.4 km/s log step) the width was right only at 4122 A, 18 % too small at 5000 A, 31 % too small at 6000 A and 37 % too large at 3000 A, i.e. a 3 A line broadened 'to 8.4 A' came out at 6.9 A at 5200 A and 6.2 A at 6000 A (10.2 A at 3300 A), so the Lick region was left narrower than the LIS system. New `fwhm_to_sigma_pixels(FWHM, wave)` (local pixel step, valid for linear and log grids) in both modules. Effect on 9 bins of OB 26133 patch P0010 (LSF taken as 2.6 A, sigma_v 143 km/s): Mgb 0.47 A, NaD 0.61 A, Fe5335 0.37 A, Fe4383 0.12 A, Fe5270 0.13 A different from the log-grid formula; against the unbroadened spectra (what flag 3 products contain) Mgb -1.7, Fe5270 -1.0, Fe5335 -1.1, NaD +2.2, Hbeta +0.4, HdeltaA -1.2, HgammaA +1.4 A. Tests in `tests/test_ls_broaden.py`.

### Changed (SLURM resources re-measured after the seg2d fix, 2026-11-06; doc/SLURM_RESOURCES.md section 12)
- **Statistics per stage** from the accounting of both pods (n, elapsed p50 / p95 / max, busy cores, MaxRSS) and the eight large-IFU observations of night 20261104 (OB 26133: IFU_PR 13 min, IFU_ExGal 9 min, IFU_Gal 39 min after the fix); how MaxRSS overstates the footprint, why the history is an upper bound (images on, old allocation), how to repeat it.
- **`v2`**: IFU_Gal 4 -> **2 CPUs**, 48 / 96 -> **32 / 64 GB** (serial `ferre.x`: 1.0-1.3 busy cores, MaxRSS at most 10.9 GB); IFU_ExGal 80 / 128 -> **48 / 80 GB** (PSS 14-20 GB). **`v2_openmp`** keeps IFU_Gal at 4 CPUs (OpenMP threads, unmeasured) with the new memory. `default`, the golden files and every MOS row are unchanged; no time limit was changed (reasons in the doc).
- **Machine check**: the line at the top of every stage log is now `PyAPS machine check: policy=.. site=.. target=.. profile=<profile> asked ..`; `aps_runner.slurm_time_arg` (compatibility wrapper) no longer pins the profile `default` but uses the shared resolution (environment, the pod's DMS config, script_params, table default).
- Tests: `tests/test_slurm_resources.py` (new IFU values and their margins, `v2_openmp`), `tests/test_dms_settings.py` (profile in the machine-check line, `slurm_time_arg` follows the DMS config).

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
### Added (one place decides the profile, limits policy, site and plot-export mode: the pod's DMS config counts)
- **`py/PyAPS/aps_dms_settings.py`** (standard library only): reads exactly five keys of the DMS config (`slurm_profile`, `plot_export`, `slurm_limits`, `slurm_site`, `max_inflight_obs`; nothing else of the file, which holds the database password) from `PYAPS_DMS_CONFIG` or `~/.private_env_configs/PyAPS_dms_config.json`, and holds the precedence chain: option > environment > **DMS config** > script_params > site / profile / table default. A caller that passes its own `env` never sees a pod's file.
- `aps_slurm_resources.resolve()`, `choose_profile_ex()`, `detect_environment()` (module 1.2.0) and `apsPlot.export.resolve_mode()` use it, so the operator commands `show | explain | check | site`, a runner started by hand and `aps-offline` take the pod's configured profile (`v2` or `v2_openmp`) instead of `default`. The order for a run the DMS starts is unchanged (the DMS passes the same values as options).
- `show | explain | check | site` and the runner print a `settings` block: the value of each setting, the source that set it and what it overrides (`--json` of `site` / `check`: `settings`). New options `--dms-config FILE|none` and `--plot-export`.
- Tests: `tests/test_dms_settings.py` (12 tests: only the five keys are read and the password never printed, file location, broken file, every precedence step, the machine-check header, the runner, the CLI); `tests/conftest.py` switches the DMS lookup off for every test.
- Documentation: `doc/SLURM_RESOURCES.md` section 11 (how the profile is resolved, the DMS config, why a `slurm_profile` key in `script_params.yaml` of a pod is overwritten by the next render: PyAPS_admin `pyaps-admin local` from `pyaps_site_setup.sh --ensure`).

### Added (the resource checks know the machine: probe, sites, limits policy; depth-first order of the jobs)
- **Machine probe** (`py/PyAPS/aps_resource_probe.py`, standard library only): a SLURM cluster (version, plugins, partitions, every node's CPUs / `RealMemory` / allocation / state; `scontrol` with `sinfo` as the fallback, 5 s per command, a hung controller is tried once) or this machine (CPUs with `sched_getaffinity` and the cgroup v1/v2 quota, RAM with `/proc/meminfo` and the cgroup limit, swap, load, Linux / WSL / macOS, container) and whether `ferre.x` links OpenMP. Every failure is "unknown" with a message, never an exception; the answer is cached for a minute.
- **Limits policy** `limits: off | warn | clamp | error` (new: `error`), chosen by `--slurm_limits`, `PYAPS_SLURM_LIMITS`, script_params `slurm_limits`, the site, the profile. `warn` (the default of every production profile) prints the request that does not fit with the value and the `--slurm_set` that would fit and **keeps** it; `clamp` lowers only what does not fit; `error` refuses before anything is submitted, only when the request can never run (the 380 GB FERRE request on a 351 GB node), listing every stage and the alternative. A request that fits is never changed. The check compares CPUs and memory against the same node (no node with 40 CPUs and 300 GB together is also a finding).
- **Sites**: the table's `sites:` now describes a place (`match`, `profile`, `profile_openmp`, `ferre_openmp`, `limits`, `mem_fraction`, `cpu_fraction`, `expect`; the old `max_*` pins still work for a named site). Generic sites `local` and `slurm` are in the table; an installation lists its own in `slurm_sites.yaml` (`PYAPS_SLURM_SITES_FILE`, `slurm_sites_file`, or `PyAPS_CONFIG`). The site is detected from the machine or named (`--slurm_site`, `PYAPS_SLURM_SITE`, `slurm_site`); the word `site` as profile (`--slurm_profile site`) takes the site's profile (nothing chooses it by itself, so no profile changes). New profile `laptop`.
- **Machines without SLURM** (a bash run, `--hpc 1`): the CPUs are a share (`cpu_fraction`) of what the process may use and the RAM a share (`mem_fraction`) of what is available, both divided by `--parallel_jobs`; a stage whose workers do not fit in the RAM runs with fewer workers (table section `footprint`, measured PSS per stage). Default policy of the `local` site: `clamp`.
- **Operator commands**: `python -m PyAPS.aps_slurm_resources probe|site|check` (read only; `check` exits 0 / 1 / 3; `--machine-file` previews another machine).
- **Recorded**: `slurm_resources_effective.json` gets `environment`, per stage `requested`, `adjustments`, `fits`, `footprint_gb`, `limits_policy`, `nice`, `comment`, and the run's `priority`; every stage log starts with a `PyAPS machine check:` line.
- **Depth-first order of the jobs** (table section `priority`; policy `depth_first`, **off in every profile**: in the private experiments of 2026-11-05 the nice offsets did not change the order, see `doc/SLURM_RESOURCES.md` section 10): every job gets a non-negative `--nice` (class base + 800 x stages still to come + 100 x pair index) and `--comment pyaps:<uapsid>:p<pair>:<class>:<STAGE>`, so the first pair of a night completes first (the queue is FIFO by eligible time and the late stages of the first pair used to queue behind the first stages of every later pair). Classes `scheduled` (4500) and `offline` (no nice at all: it rises above the queue's floor of 1 with its age and passes the scheduled jobs that sit at the floor): a run someone waits for is ahead of the daemon's jobs. `--slurm_priority`, `--priority_class`, `--pair_index` (env `PYAPS_SLURM_PRIORITY`, `PYAPS_PRIORITY_CLASS`, `PYAPS_PAIR_INDEX`). An escalation retry keeps the pair index of its chain. Bounded below 10000, nothing starves. Doc: `doc/SLURM_RESOURCES.md`, sections 9 and 10.
- Tests: `tests/test_resource_probe.py`, `tests/test_slurm_machine_checks.py` (fake clusters, a laptop, a container with cgroup limits, a hung `scontrol`, garbage output; no ordinary run on a production-like fake cluster is blocked or changed); the golden tests of the `default` profile are unchanged.
- Changed: with no look possible (no SLURM, no probe) nothing is checked; the old `limits` argument of `resolve()` and the `sites` pins behave as before; `resolve()` returns a dict that also carries `.environment` and `.priority`.


### Changed (static plot images are OFF by default)
- **`PYAPS_PLOT_EXPORT` unset now means `off`** (`PyAPS.apsPlot.export.DEFAULT_MODE`; it was `worker`). The diagnostic PNG/PDF pictures are rendered by Plotly through Kaleido and a headless Chrome, which was most of the wall time and CPU of an L2 job in the SLURM audit of 2026-11-04 (one RR job: 81 min and 520 CPU-minutes with pictures, 18 min and 117 CPU-minutes without; Chrome used 17-18 threads on an 8-CPU job and 13-22 GB in RVS and IFU_ExGal). No data product depends on a picture: `write_image` never raises and no caller reads its result; FITS products of the same OB with the pictures on and off were compared bit for bit (`tools/compare_l2_products.py` of PyAPS_admin), see `doc/SLURM_RESOURCES.md` section 8.
- Turn them on again with `aps_runner.py --plot_export on`, the script_params key `plot_export: 'on'`, or `PYAPS_PLOT_EXPORT=worker` in the environment of the runner (order: option, environment, file, `off`). `PYAPS_PLOT_EXPORT` accepts `off`, `worker` (also `on`, `true`, `1`) and `inprocess`; an unknown word is ignored. The runner writes `export PYAPS_PLOT_EXPORT=<mode>` into every job script, prints `PyAPS plot export: mode=... source=...` at the top of every job log and with the resource table, and records it as `plot_export` in `slurm_resources_effective.json`.
- With the default a skipped picture logs one INFO line per process, not a warning per picture.
- Tests: `tests/test_plot_export.py` (default off, aliases, precedence option > environment > file > off, the runner's script lines; the worker tests set `PYAPS_PLOT_EXPORT=worker`).
- Documentation: README "Static plot images", `doc/aps_runner.md`, `doc/SLURM_RESOURCES.md` section 8, `configs/script_params.yaml.example`.

### Added (SLURM resources: one table, memory requests, thread consistency)
- **One table for the CPUs, memory and time limit of every job** (`configs/slurm_resources.yaml`, `py/PyAPS/aps_slurm_resources.py`, `doc/SLURM_RESOURCES.md`). The CPU lists the caller used to hand to the runner (`--mp_MOS 8,12,18,10,20,6,1` ...), the HR doubling, the FERRE halving for single exposures, the `--mem` of FERRE / IFU_ExGal / FESWI and the 48 h limits were separate pieces of code; they are now the rows and rules of the profile `default`, which reproduces them value for value (golden test `tests/test_slurm_resources_golden.py`: 29 observation classes, the `sbatch` lines and the `#SBATCH` pragmas of the old code against the new). A profile `v2` carries the allocation from the measurements; rows without a measurement are marked `unmeasured` and hold the old value. A profile can extend another, a row can carry rules (`when` resolution / stack level / FERRE grid), and the table is validated with named errors (`python -m PyAPS.aps_slurm_resources validate|show|explain`).
- Choose a profile with `--slurm_profile`, `PYAPS_SLURM_PROFILE` or the script_params key `slurm_profile`; change one stage with `--slurm_set STAGE.cpus|mem_gb|time=...` or the script_params keys `slurm_cpus_<STAGE>`, `slurm_mem_gb_<STAGE>`, `slurm_time_<STAGE>` (the last one existed before). The legacy `--mp_*` lists still work and replace the base CPU counts.
- Every job states what it asked for and what SLURM gave it at the top of its own log (profile, CPUs, memory, time, `SLURM_NTASKS`, a warning when they differ); the runner prints the table and writes `slurm_resources_effective.json` beside the job scripts. A request larger than the biggest node (read from `sinfo`, or the `sites` entry named by `slurm_site`) is reported, or lowered with the profile's `limits: clamp`.
- Escalation record (`slurm_resources_escalation.json` beside the scripts, written by the operations layer): a stage that ended OUT_OF_MEMORY or TIMEOUT is generated again once with 1.5 x its memory / time limit (never beyond the largest node or the partition limit).
- **FERRE threads match the CPUs of the stage** (`aps_ferre.ferre_threads_per_process`): the stage runs one `ferre.x` per grid at the same time and wrote `NTHREADS = --mp` into each `input.nml`, i.e. 9 grids x 18 threads on 18 CPUs; each process now gets `max(1, mp // min(mp, grids))` threads (`PYAPS_FERRE_THREADS=mp` restores the old value, a number sets it). A `ferre.x` built without OpenMP ignores `NTHREADS`: no change for it.
- The effective values and the escalation record are kept per run (the uapsid, `<uapsid>:CS` for the CS chain) in the scripts folder of the OB, which holds the scripts of every pair of the OB. The runner looks for `slurm_resources.yaml` in the `PyAPS_CONFIG` folder of its script_params first, then in its own checkout (`slurm_resources_file` / `PYAPS_SLURM_RESOURCES_FILE` name another).
- `aps_job_report.py` also reports a job that SLURM killed for exceeding its memory request but listed as FAILED (the job's own script is killed with it and writes no line); it reads the `.err` file.

### Changed (SLURM resources)
- A run without any `--mp_MOS/--mp_LIFU/--mp_MIFU/--mp_CS` list now takes the CPU counts of the table (it used to get one CPU per stage). `slurm_time_arg()` and `DEFAULT_SLURM_TIME` remain as thin wrappers over the table.
- No change of behaviour with the `default` profile.

### Changed
- **Every L2 SLURM stage may run 48 hours** (`DEFAULT_SLURM_TIME` of `py/PyAPS/aps_runner.py`, the partition limit; owner decision 2026-10-26). The earlier limits (RR/FR/PPXF 8 h, RVS/LS/L2merge 2 h, EMI 6 h, IFU 18 h) killed the FR (FERRE) stage of all 12 configurations of three OBs at 8 h. Per-stage values: `slurm_time_<STAGE>` in `script_params`.

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

### Fixed (static plot images can no longer hang a job)
- Eleven L2 jobs (PPXF, EMI, RR, FR, RVS) of two APS hosts stopped at the plotting step in October 2026 and ran into the 2-day SLURM limit: `fig.write_image()` renders through Kaleido 1.x, which drives a headless Chrome and waits for it without a deadline while it starts, navigates and closes (only the render itself has the 90 s limit). In the hung jobs Chrome was alive and idle, Python was waiting for it, no CPU was used, and the night stayed PROCESSING because the runner saw failed jobs for two days. A diagnostic picture must never cost a night.
- New `PyAPS.apsPlot.export.write_image(fig, path, scale=...)` replaces every `fig.write_image(...)` of the pipeline (a test enforces it). It renders in a separate worker process (`apsPlot/_export_worker.py`, its own session, one browser for many images, so also about ten times faster than a browser per image); each image has a deadline (`PYAPS_PLOT_TIMEOUT`, default 120 s; first image `PYAPS_PLOT_START_TIMEOUT`, 180 s); on timeout the worker's stack and the last lines of its log are written to the job log, worker and browser are killed, the picture is skipped and the next one starts a new worker; after `PYAPS_PLOT_MAX_FAILURES` (3) timeouts in a row the pictures of the process are switched off; the worker is replaced every `PYAPS_PLOT_RECYCLE` (200) images. It returns False instead of raising, so a figure never stops a data product. `PYAPS_PLOT_EXPORT=inprocess` restores the plain call (no deadline), `off` writes no static image. Files are written under a temporary name and renamed, so a killed render leaves no truncated picture. `tests/test_plot_export.py` (fake worker: hang, death, errors, fork safety; an opt-in test with the real Kaleido: `PYAPS_TEST_KALEIDO=1`).
- Every L2 SLURM job now has a `--time` limit sized from the jobs that completed (`aps_runner.DEFAULT_SLURM_TIME`: RR/FR/PPXF 8 h, EMI 6 h, RVS/LS/L2merge 2 h, IFU stages 18 h; the longest completed job of each kind is at most half of it), so a job that waits for ever dies in hours instead of 2 days. Override one stage with the `script_params` key `slurm_time_<STAGE>` (quoted, e.g. `'12:00:00'`) or switch it off with `'None'`. The CS stages have no limit. `tests/test_slurm_time_limits.py`.

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
