# aps_runner.py - script generation and SLURM submission

**Last verified against the code:** 2026-08-27

This is the layer that turns one pair of L1 files into a set of processing
scripts and (optionally) submits them to SLURM. It can be invoked by a user
running it directly against a pair of L1 files, or by any external
orchestration layer that shells out to it (see the exit-code contract below).
Traceable ad-hoc runs are covered in
[Traceable ad-hoc test runs](#traceable-ad-hoc-test-runs).

Either way, the same code path runs: read the L1 headers, generate a
per-module bash script for each processing step, generate a SLURM wrapper
script that chains them with `sbatch --dependency`, then run that wrapper
if `--hpc` says to.

---

## What it does, step by step

1. **Read L1 headers** (`l1_fileinfo()`) from the two input files - this is
   where `obsmode`, `res_mode`, `camera`, `obid`, `obsdate`, `mjd_obs`, etc.
   come from. `obsmode` decides which of the three pipelines below runs.
2. **Resolve `headname`** - either given directly (`--headname`), or built
   from the input filenames (`stack_X__stack_Y`), plus an optional
   `--hname_suffix`. This is what output *filenames* are based on.
3. **Resolve the job tag** (`JOB_ID_HEADNAME`) - `--uapsid` if given (this
   is what an orchestration layer would pass), otherwise falls back to
   `headname`. This is what every SLURM job's *name* is based on, and what
   you search on with `squeue --name`.
4. **Resolve the output directory** (`outpath`) - normally
   `<PyAPS_RES|CS_RES>/<nightobs-or-obsdate>/<obid>/` (`CPS_path=True`
   extracts the night from the L1 file's own path; `False` uses the L1
   header's `obsdate` instead) - see [Test mode](#traceable-ad-hoc-test-runs)
   for how `--test_mode` changes this.
5. **Generate one bash script per processing module** (`write_bash()`),
   plus one combined "kiko-ready" script, in `<outpath>/scripts/`.
6. **Generate the SLURM wrapper script** that submits each module with
   `sbatch --dependency=...` in the right order - see next section - in
   `<outpath>/scripts/<tag>_PyAPS_slurm.sh` (or `..._CS_slurm.sh`).
7. **Run it, per `--hpc`**: `0` = generate only, don't run; `1` = run the
   combined script directly with `bash`; `2` = submit via `kiko`; `3` = run
   the SLURM wrapper (i.e. actually `sbatch` everything).

`--run_L2`/`--run_CS` control whether steps 2-7 happen for the L2 pipeline,
the CS pipeline, or both - they're independent (an orchestration layer would call this once per
pipeline, i.e. once for L2 and once for CS, and track each separately).

---

## The three pipelines

`obsmode` from the L1 headers decides which script-generation function
runs (`mos_scriptGEN`, `ifu_scriptGEN`, or `cs_scriptGEN` for the CS path
regardless of L2 mode). Each produces its own SLURM dependency structure -
this is the actual mechanism behind how jobs are bundled per observation block:

### MOS L2 (`mos_scriptGEN`) - straight-line chain

```
RR -> RVS -> FR -\
              PPXF -> EMI -> LS -> L2merge
```
- `RR` has no dependency (first job).
- `RVS`, `PPXF` both depend on `RR` (`--dependency=afterany:$jid_RR`).
- `FR` depends on `RVS`.
- `EMI` depends on `PPXF`.
- `LS` depends on both `PPXF` and `EMI`.
- `L2merge` (the final product) depends on all six of the above.

### IFU L2 (`ifu_scriptGEN`) - fan-out, no merge job

```
IFU_Prepare -> IFU_ExGal
            -> IFU_Gal      (parallel with ExGal)
```
Only `IFU_Prepare` has no dependency; `ExGal` and `Gal` both depend on it
and run in parallel with each other - no final merge step (each is itself
a terminal product).

### CS (`cs_scriptGEN`) - fully independent, no chain

Up to 7 modules (`SQ`, `FESWI`, `SPACE`, `AMY`, `AN`, `RRLGV`, `RRLEW`),
**each only submitted if its script exists** for the current config
(module enable/disable is config-driven, checked via
`scripts_path.joinpath(f"{tag}_CS_<MODULE>.sh").is_file()`). No
dependencies between them, no merge job - they just all run in parallel.
CS also needs the *L2* output path as a read-only input dependency
(`L2_outpath` - CS reads L2 products) - this always points at the real L2
directory, even under `--test_mode`, since a test CS run still needs real
L2 inputs to exist.

All dependencies use `--dependency=afterany:...` (proceeds regardless of
the predecessor's exit code - retry/failure handling is left to whatever
calls this script, not a SLURM dependency concern here).

### Failed jobs in the runner log

Because every dependency is `afterany`, a failed stage does not stop the chain. Set the optional
`runner_log` key of the `script_params` file to the log file of your WDAP runner (default `'None'` = off):
the job scripts export it as `PYAPS_RUNNER_LOG`, and a stage that ends with a non-zero exit code appends
`[time] ERROR: SLURM job <id> <name> failed with exit code <n>; stderr: <logs>/<name>.<id>.err`.
The MOS `L2merge` job, which runs after all other stages, also runs `aps_job_report.py`, which adds one such line
for each job of the OB that was killed (TIMEOUT, CANCELLED, OUT_OF_MEMORY, NODE_FAIL). IFU and CS chains
log their own failing stages the same way. The L2 state in the database is set by the WDAP monitoring, not here.

### Guarding against a broken chain (`sbatch_guard`)

Every `sbatch --parsable` call in the generated wrapper is followed by a
check (`sbatch_guard()`) that a job ID actually came back, before it's
used as a `--dependency` for the next stage - added 2026-08-20 after
noticing the wrapper scripts (plain `#!/bin/bash`, no `set -e`) had no
such check. For chained stages (MOS, IFU's `Prepare`), a missing job ID
aborts the wrapper immediately with a clear error rather than letting a
broken `--dependency=afterany:` propagate silently through the rest of the
chain. For independent stages (CS's modules, IFU's `ExGal`/`Gal`
themselves), it logs and continues instead, so one module failing to
submit doesn't stop its unrelated siblings.

That abort only matters if something outside the wrapper script actually sees
it, so the wrapper's return code is captured and re-raised as `aps_runner.py`'s
own exit code (`sys.exit()` on non-zero). **Contract:** exit code `0` means every
requested stage was submitted; non-zero means a submission failed (for example
`slurmctld` was transiently unreachable). Any caller that shells out to this
script can rely on that to record success or failure.

---

## Traceable ad-hoc test runs

Running this by hand against a pair of L1 files (to test a code change,
try different parameters, etc.) with `--hpc 3` submits real SLURM jobs
as any other run - but by default (`--uapsid None`), those jobs
get named from `headname` instead of an explicit tag, and output lands directly in the real per-OB directory alongside official
products.

**`--test_mode True`** (`apply_test_mode()`) fixes both:

- Forces a `TEST_`-prefixed job tag - auto-generated
  (`TEST_<timestamp>_<random>`) if you don't pass `--uapsid`, or your own
  tag prefixed with `TEST_` if you do. Every job in the chain gets this
  tag in its name (`RR_L2_TEST_...`), so it's unmistakable in `squeue` and
  immediately trackable. If both `--run_L2` and `--run_CS` are requested
  in the same call, they share one tag (resolved once in
  `scriptGen_runner`, not independently by each pipeline function).
- Redirects output into `<PyAPS_RES|CS_RES>/TEST/<night>_<obid>_<tag>/`
  instead of the real per-OB directory - can never land in or collide with
  an official product's files. Safe to bulk
  `rm -rf <l2_dir|cs_dir>/TEST/` any time.

```bash
python3.11 py/PyAPS/aps_runner.py --infiles a.fit b.fit \
    --config_file configs/script_params.yaml --test_mode True \
    --headname test_run --run_L2 True --hpc 3 ...
# [TEST MODE] auto-generated tag: TEST_20260820153531_35e20b

squeue --name TEST_20260820153531_35e20b   # live status
```

Test runs are not registered anywhere. No `--uapsid` means
each run auto-generates its own tag, so repeating the same command never
collides (verified live: two back-to-back identical runs got two different
tags/directories); giving the same `--uapsid` explicitly both times
*does* overwrite, same as reprocessing does for real products, just
confined to the test area. Nothing cleans test output up automatically.

---

## Command reference

Entry point: `scriptGen_runner()`, invoked via `bin/aps_runner` or
`python3.11 py/PyAPS/aps_runner.py ...`.

| Parameter | Default | Description |
|---|---|---|
| `--infiles` | required | The two L1 file paths |
| `--config_file` | required | YAML script-params config (`PyAPS_RES`, `CS_RES`, module settings, etc.) |
| `--headname` | derived from infiles | Output filename base |
| `--hname_suffix` | `None` | Appended to headname |
| `--uapsid` | `None` | Job-name tag; `None` falls back to headname (or gets auto-generated under `--test_mode`) |
| `--test_mode` | `False` | See [above](#traceable-ad-hoc-test-runs) |
| `--run_L2` / `--run_CS` | `True` / `True` | Which pipeline(s) to generate/run |
| `--CPS_path` | `True` | Derive the output night from the L1 file's own path vs. its header `obsdate` |
| `--hpc` | `0` | `0`=generate only, `1`=bash, `2`=kiko, `3`=SLURM |
| `--aps_ovr` / `--CS_ovr` | `False` / `False` | Overwrite existing final L2/CS products |
| `--log` | `False` | Record logs to an auto-generated log file |
| `--cat_list`, `--aps_ids`, `--wlranges`, `--targsrvy`, `--targclass`, `--mask_aps_ids`, `--area`, `--mask_areas`, `--arms_ratio` | `None` | Science/target-selection parameters, passed through to the per-module scripts |
| `--mod_wlranges` | `False` | Auto-adjust wavelength ranges from the L1 header |
| `--mp_MOS`, `--mp_LIFU`, `--mp_MIFU`, `--mp_CS` | `None` | Thread/CPU counts per module (single value or per-module list) |
| `--seg2d_white_images`, `--patch_file` | `None` | IFU-specific inputs |

---
