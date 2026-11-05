# SLURM resources of the PyAPS jobs: one table

Every job that `aps_runner.py` submits (L2 stages, IFU stages, CS stages) asks SLURM for CPUs, memory and a time limit.
All three come from **one table**, `configs/slurm_resources.yaml`, read by `py/PyAPS/aps_slurm_resources.py`. Nothing else in the
code decides a CPU count, a `--mem` or a `--time`.

* Which job gets what: section 2 and `python -m PyAPS.aps_slurm_resources explain ...`
* Change the allocation of the whole pipeline: edit a profile of the table (section 3)
* Change one value for one run: `--slurm_set`, script_params keys or environment variables (section 4)
* See what a job got: section 5
* A job killed for memory or time is retried once with more: section 6
* The static plot images (headless Chrome) are OFF by default and are a large part of what a job costs: section 8

## 1. The model

A **profile** is a named set of rows, one row per stage of each family of jobs:

| family | meaning | stages |
|---|---|---|
| `MOS` | L2 of a fibre (MOS) observation | RR, RVS, FR, PPXF, EMI, LS, L2merge |
| `LIFU` | L2 of a large IFU observation | IFU_PR, IFU_ExGal, IFU_Gal |
| `MIFU` | L2 of a mini IFU observation (smaller field) | IFU_PR, IFU_ExGal, IFU_Gal |
| `CS` | the CS (cross-survey) modules | SQ, FESWI, SPACE, AMY, AN, RRLGV, RRLEW |

A row has `cpus`, `mem_gb` (GB, 1024 MB; `null` = no request, SLURM gives `DefMemPerCPU` per CPU), `time` (`null` = no limit),
a `status` and optional `cases`. The `cases` are rules that change the row for part of the observations: they are applied in the order
written, first the stage's own, then the `common` rules of the family. A rule has a `when` (`res: LR|HR`, `stack: single|stack|super|other`,
`grid: m|n`, numeric ranges `{ge:, lt:}` of the same facts) and the change: a value (set) or one operation `{scale: 2}`, `{add: 4}`,
`{half_plus_one: true}` (cpus), `{per_cpu: 4}` (memory per final CPU), `{set: v}`.

Profiles:

* `default` - **exactly what the runner did before the table existed**, value for value (golden-tested, `tests/test_slurm_resources_golden.py`).
  The CPU lists the DMS used to hand over (`--mp_MOS 8,12,18,10,20,6,1`, `--mp_LIFU 6,30,20`, `--mp_MIFU 3,10,10`, `--mp_CS 6,12,6,12,12,1,1`) are its base values.
* `v2` - the allocation set from the measurements (section 7). A profile may `extends` another and then lists only the rows that differ.

`status` of a row: `legacy` (the value that was in the code), `measured` (set from a measurement), `derived` (from reading the code, e.g.
a stage that starts no worker processes), `unmeasured` (copied from the old value; nobody has measured it yet).

Every stage pins BLAS/OpenMP to one thread and starts one process per CPU, so `cpus` is at the same time the sbatch `--ntasks` and the
stage's own `--mp`. There is one number and one place; the job prints both it and what SLURM granted at the top of its log (section 5).

## 2. What the old code did (now rules of the `default` profile)

| stage | old place (aps_runner.py before the table) | CPUs | memory | time | special cases |
|---|---|---|---|---|---|
| MOS RR / RVS / FR / PPXF / EMI / LS / L2merge | `mos_scriptGEN` | 8 / 12 / 18 / 10 / 20 / 6 / 1 (from the DMS) | FR `--mem=190GB` (grid m, LR) or `380GB` (grid m, HR); others none | 48 h each (`DEFAULT_SLURM_TIME`) | HR: every CPU value x2; single exposure (stack level 0): FR = `int(cpus/2)+1` before the HR doubling, grid forced to `n` |
| IFU_PR / IFU_ExGal / IFU_Gal (LIFU) | `ifu_scriptGEN` | 6 / 30 / 20 | IFU_ExGal `--mem=240GB` | 48 h | HR x2 |
| IFU_PR / IFU_ExGal / IFU_Gal (mini IFU) | `ifu_scriptGEN` | 3 / 10 / 10 | none | 48 h | HR x2; no memory request |
| CS SQ / FESWI / SPACE / AMY / AN / RRLGV / RRLEW | `cs_scriptGEN` | 6 / 12 / 6 / 12 / 12 / 1 / 1 | FESWI `--mem=400GB` (grid m) | none | HR x2 |

`python -m PyAPS.aps_slurm_resources show --family MOS --res HR --stack single --grid m` prints the rows for one observation class;
`explain` adds the rule trace of every value.

## 3. Change the table

Edit `configs/slurm_resources.yaml` (or copy it and point `slurm_resources_file` in script_params, or `PYAPS_SLURM_RESOURCES_FILE`,
at the copy). Check it:

```
python -m PyAPS.aps_slurm_resources validate
```

The validator names the bad place (`profiles.v2.MOS.stages.FR.mem_gb: ...`): unknown keys, a stage that is not in the family, a number
that is not positive, a time that is not a SLURM time, a profile that extends itself, a family that has no CPU count for a stage.
Keep `default` as it is: it is the reference of the golden test. Put a new allocation in a new profile (or in `v2`) and choose it
(section 4).

Memory per job is `mem_gb` as written. A request larger than the biggest node can never run; the runner reads the largest node and the
partition time limit from SLURM (`sinfo`) and the profile's `limits` says what to do: `off`, `warn` (print, never change, the `default` profile)
or `clamp` (lower to the node limit). To pin the limits of a site, list it under `sites:` and name it with the script_params key `slurm_site`.

## 4. Change one run

Order of precedence for the **profile**: `--slurm_profile NAME` of `aps_runner.py` > the environment variable `PYAPS_SLURM_PROFILE` >
the script_params key `slurm_profile` > the table's `default_profile`. (The DMS passes `--slurm_profile auto`, which means "the usual
choice"; its config key `slurm_profile` can fix one.)

For **one stage**, in this order of precedence:

| how | example |
|---|---|
| `aps_runner.py --slurm_set STAGE.key=value` (repeatable) | `--slurm_set FR.mem_gb=120 --slurm_set RR.cpus=4 --slurm_set LS.time=00:30:00` |
| script_params keys | `slurm_cpus_RR: '4'`, `slurm_mem_gb_FR: '120'`, `slurm_time_EMI: '12:00:00'` (quote times; `'None'` = no limit) |
| the legacy `--mp_MOS/--mp_LIFU/--mp_MIFU/--mp_CS` lists | replace the **base** CPU counts; the rules of the profile (HR doubling...) still apply |

A value set with `--slurm_set` or a script_params key is **final** for that stage (no HR doubling on top). Without any `--mp_*` list the CPU
counts come from the table; before the table such a run got one CPU per stage.

## 5. See what a job got

* The runner prints the table of the observation (profile, table file and checksum, per stage cpus / memory / time / status) and writes
  `slurm_resources_effective.json` into the scripts folder of the OB (profile, context, every stage's values and the rule trace).
* The top of each stage's own log (`<jobname>.<id>.out`) says
  `PyAPS SLURM resources: stage=FR profile=default status=legacy cpus=18 mem=190 GB time=2-00:00:00 mp=18 threads=1` and
  `PyAPS SLURM granted: job=... ntasks=18 cpus_on_node=... mem_per_node_MB=...`; it warns when SLURM gave another task count than the stage's `--mp`.
* Every stage script carries `#SBATCH --ntasks` (and `--mem` when the row has one), so a hand resubmission of one stage gets its allocation.
* What a finished job really used (`sacct`: MaxRSS, TotalCPU, Elapsed, ReqMem) is kept per job by the pipeline's resource-usage table when the
  operations layer is installed.

## 6. A job killed for memory or time is retried once with more

When SLURM enforces memory, a job that exceeds its `--mem` is killed: with the cgroup plugin its state is `OUT_OF_MEMORY`; with the
accounting-plugin enforcement (`JobAcctGatherParams=OverMemoryKill`) it is `FAILED` and the job's `.err` says "exceeded memory limit". The
operations layer (the DMS) recognises both, and a stage whose latest job ended `TIMEOUT` too, and - only when the row ended failed and the
stage of that run was not escalated before - writes `slurm_resources_escalation.json` beside the scripts (per run, per stage: memory x1.5 for an
out-of-memory stage, time x1.5 for a timeout) and sends the row back to be scripted. The runner reads the file when it generates the scripts
again and applies the factors (never beyond the largest node or the partition time limit). A second failure of the same stage is not
escalated again: the row stays failed. Delete the file to reset the counters. Without memory enforcement on the cluster nothing is ever killed for memory, so nothing is
retried for memory (a TIMEOUT is retried whatever the cluster enforces).

## 7. Measured allocation (`v2`)

See the table file: every row of `v2` carries `status` and, where measured, the numbers behind it in `note`. Rows still marked
`unmeasured` hold the old value.

## 8. Static plot images are off by default (since 2026-11-04)

The diagnostic pictures (PNG/PDF) of the stages are rendered by Plotly through Kaleido, which drives a headless Chrome. In the SLURM audit of
2026-11-04 this was most of the wall time and CPU of the jobs: the same RR job took 81 min and 520 CPU-minutes with pictures and 18 min and
117 CPU-minutes without; Chrome used 17-18 threads on an 8-CPU job and 13-22 GB in RVS and IFU_ExGal. No data product depends on a picture
(`PyAPS.apsPlot.export.write_image` never raises and no caller reads its result), so **the default is now `off`**.

| how | effect |
|---|---|
| nothing | `off`: the figures are built, no image is rendered, no Chrome is started |
| `aps_runner.py --plot_export on` | the pictures of the jobs this run generates |
| `plot_export: 'on'` in script_params | the pictures for a site or a run directory |
| `PYAPS_PLOT_EXPORT=worker` in the environment of the runner | the same (`on`, `worker`; `inprocess` is the plain call for debugging; `off`) |

Order: option, environment, script_params, `off`. The runner writes `export PYAPS_PLOT_EXPORT=<mode>` into every job script (so the job
does not depend on the environment of `sbatch`), prints `PyAPS plot export: mode=off source=default ...` at the top of every job log, prints it
with the resource table and records it as `plot_export` in `slurm_resources_effective.json`. The DMS passes the option only when its own config
has `plot_export`; `aps-offline --plots on|off` does the same for a standalone run.

With the pictures on, the Chrome load is part of the memory and CPU of the stage: the resource profiles were measured with the pictures **off**
(section 7), and with them on the memory of RVS and IFU_ExGal is 13-22 GB larger and RR needs about twice the time. Switch them on only
for a stage you want to look at.
