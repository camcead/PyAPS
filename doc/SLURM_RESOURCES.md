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
* The request is checked against the machine or cluster it will run on (CPUs, RAM, node sizes), with a policy `warn | clamp | error | off`: section 9
* Depth-first order of the jobs (the first pair of a night completes first; a run someone waits for passes the daemon's jobs): section 10
* Which profile, limits policy, site and plot-export mode a tool or a job really uses on a pod, and which source set each (the DMS config of the pod): section 11
* The re-measurement of 2026-11-06 (statistics per stage from the accounting of both pods, what was tightened and what was left, how to repeat it): section 12

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

Memory per job is `mem_gb` as written. A request larger than the biggest node can never run; the runner looks at the cluster (or at the machine
for a bash run) and the policy `limits` says what to do: `off`, `warn` (print, never change: the default), `clamp` (lower only what does not fit) or
`error` (refuse what can never run). The whole mechanism, the sites and the machine probe are described in section 9. To pin the limits of a site
(not asking SLURM), list it under `sites:` with `max_mem_gb` / `max_cpus` / `max_time` and name it with the script_params key `slurm_site`.

## 4. Change one run

Order of precedence for the **profile**: `--slurm_profile NAME` of `aps_runner.py` > the environment variable `PYAPS_SLURM_PROFILE` >
the `slurm_profile` key of the pod's **DMS config** > the script_params key `slurm_profile` > the table's `default_profile`. (The DMS passes
`--slurm_profile auto` when its config has no value, which means "the usual choice"; with a value it passes that value. On a pod put the profile in
the DMS config, not in `script_params.yaml`: section 11.)

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

`v2` is the allocation set from two measurements: the accounting of two sites (5 546 L2 jobs since 2026-09-23: busy cores, run times) and a
rehearsal on a development site (2026-11-05: the same pairs run with `default` and with `v2`, plot images off, true memory sampled as PSS every
10 s). Every row of `configs/slurm_resources.yaml` carries its `status` and the numbers behind it in `note`.

| stage | CPUs LR / HR | memory LR / HR | time limit | what it rests on |
|---|---|---|---|---|
| RR | 8 / 16 | 24 / 36 GB | 6 h | LR: 6.5-6.8 busy cores of 8, 14-23 min without plot images; HR: 16 CPUs gave 19 min and 11.8 busy cores, 8 CPUs 38 min at the same core-hours, so HR keeps 16 (a HR OB with few fibres does not use them); PSS peak 7-10 GB LR, 17 GB HR |
| RVS | 12 / 24 | 72 / 144 GB | 3 h | pool of `--mp` workers (4-12 min LR, 2-4 min HR); PSS peak 21-45 GB LR (57 GB with 16 workers), 54-97 GB HR |
| FR | 9 / 9 (`m` grids: same) | 8 GB; `m` grids 80 / 160 GB | 30 h; `m` grids 48 h | one `ferre.x` per grid and 9 grids: at most 9 cores on a single-thread build (6.3-6.8 busy measured); PSS 0.6-3 GB; with the fine grids (stack products) 51.5 GB LR, 82 GB HR so far (file-backed grid pages) |
| PPXF | 10 / 10 | 16 GB | 6 h | 7-8.4 busy cores of 10 (HR with 20 CPUs: 13.7 busy, the same 3 min as with 10); PSS peak 4-7 GB |
| EMI | 6 / 6 | 28 / 40 GB | 4 h | 2.4-4.1 busy cores; 1-3 min with 6 CPUs as with 20 or 40; PSS peak 11 GB with 20 workers, 3-5 GB with 6 |
| LS | 2 / 2 | 6 GB | 30 min | 0.5-1.0 core, 1-3 min; PSS 1-2.4 GB (3.5 GB at most in the audit) |
| L2merge | 1 / 1 | 8 / 12 GB | 15 min | 0.3-0.5 core, under 1 min; **a 41 s peak that the 10 s samples missed**: killed at 4 GB and 6 GB (HR, cgroup enforcement on a private SLURM), passed at 8 GB with 5.6 GB MaxRSS |
| IFU_PR | 3 / 3 | 24 / 36 GB | 6 h | 1.1-1.2 busy cores in 3 OBs (6 or 12 CPUs gave the same times); PSS peak 13-17 GB, 27 GB for a heavy HR cube |
| IFU_ExGal | 40 / 40 (24 for a day, 2026-11-13) | 96 / 160 GB (48 / 80 GB until 2026-11-13) | 10 h | `mp` is shared by patches that run side by side and the pool inside each patch, the width of each patch set from its real bin count (section 13.7); one patch lane peaks at 12-13 GB PSS, so 96 GB = 7 lanes; before 2026-11-13 up to 40 busy cores for a patch with very many bins, 2-4 on most OBs |
| IFU_Gal | 10 / 10 (`v2_openmp`: the same) | 48 / 48 GB (32 / 64 GB from 2026-11-06 to 2026-11-13) | 16 h / 24 h | **one `ferre.x` process per grid: 9 grids + 1 = 10 CPUs**; `mp` is shared by patches that run side by side and the bins inside each patch (section 13). Measured on OB 14699 (HR, -O2 `ferre.x`): 65 s per bin and worker, 30-40 s per patch; PSS 13 GB with 4 patches side by side, 20 GB with 10. The reduction to 2 CPUs of 2026-11-06 was a misreading of the serial code |
| mini IFU, CS | as `default` | LIFU memory; FESWI 128 GB | | no mini IFU job exists on either site and no CS job has run on SLURM: unmeasured |

Rules that were dropped: "HR doubles every CPU count" (only RR and RVS have an HR row, the others use the same cores in HR), "single exposure
halves FERRE" (FERRE is 9 CPUs in every class), and the 190 / 380 / 240 / 400 GB requests (they were scheduling tokens larger than any node
could hold: a 380 GB request never runs on a 351 GB node).

**Memory is the true footprint, not `sacct`'s MaxRSS.** `jobacct_gather/linux` adds the RSS of every forked worker, so shared pages are counted
once per process (RR: 28-65 GB for 7-17 GB held; IFU_ExGal: 212 GB for 18 GB). The memory column is the PSS peak (shared pages divided between
the processes that map them) plus a margin of 25-100 %. Until the cluster enforces memory (`task/cgroup`) it is only a scheduling token:
a request that is too large costs queue time, one that is too small costs nothing; once it is enforced it is the kill limit, and a stage above it
is killed and retried once with 1.5 x (section 6).

Profiles: `default` (the old values, the reference of the golden test), `v2` (this table), `v2_openmp` (`v2` with FR at 18 CPUs for a site whose
`ferre.x` is built with OpenMP: two threads for each of the 9 grids; IFU_Gal is the same as in `v2`, section 13). The rehearsal values and what was not measured are in the operator
document of the private repository (`doc/ops/SLURM_RESOURCES_REHEARSAL_20261105.md`).

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

## 9. Machine and site awareness (since 1.1.0 of the module)

A request is only a wish until it is compared with the place that has to run it. Before anything is submitted the runner looks at that place and
checks every stage against it. **Nothing changes unless the policy asks for it**: with the default policy (`warn`) a request is reported, never
altered, and a request that fits is never touched by any policy.

### 9.1 What is looked at (`py/PyAPS/aps_resource_probe.py`, standard library only)

| place | what is read | how |
|---|---|---|
| a **SLURM cluster** (`--hpc 3`, or `--hpc 0` on a machine that has SLURM) | version, `ClusterName`, plugins (is memory enforced? `TaskPlugin`, `JobAcctGatherParams`), the partitions (default one, `MaxTime`, `MaxMemPerNode`) and every node (CPUs, `RealMemory` minus `MemSpecLimit`, what is allocated now, state, features) | `scontrol show config\|nodes\|partition`, `sinfo` as the fallback; each command has a 5 s timeout, a controller that does not answer is given up after one try |
| **this machine** (`--hpc 1`, a laptop, a workstation, a pod or a container without SLURM) | CPUs honouring `os.sched_getaffinity` and the cgroup CPU quota (v1 and v2), RAM from `/proc/meminfo` and the cgroup memory limit (minus what the cgroup already uses), swap, load, the operating system (Linux, WSL, macOS through `sysctl`/`vm_stat`), container or not | files and two commands, no `psutil` |
| `ferre.x` | whether it links OpenMP (`libgomp`): decides between the profiles `v2` (single-thread build) and `v2_openmp` | the file is read, nothing is run |

Every failure (a missing command, a hung `scontrol`, garbage output, no `/proc`) is recorded as a message and the value stays **unknown**: the
requests are then not checked and the run goes on. The whole probe is bounded (12 s) and cached in the process for a minute.

### 9.2 The policy `limits`

| policy | a request that does not fit | a request that fits |
|---|---|---|
| `off` | not looked at (the machine is not even probed) | untouched |
| `warn` (**the default**) | printed with the value that would fit and the command to use it; the request is **kept** | untouched |
| `clamp` | **only the value that does not fit** is lowered, loudly, and recorded as `requested` and `granted` | untouched |
| `error` | refused **before anything is submitted**, and only when it can truly never run (more memory or CPUs than the largest node, more time than the partition allows; on a machine without SLURM: a stage that needs more RAM than the machine has); the message lists every such stage and what would fit | untouched |

Chosen by (first that is set): `aps_runner.py --slurm_limits`, `PYAPS_SLURM_LIMITS`, the `slurm_limits` key of the pod's DMS config (section 11), the script_params key `slurm_limits`, the `limits` of the
**site** (only for a machine without SLURM: `clamp`), the `limits` of the **profile** (`warn` for `default`; `v2` keeps its `clamp`), `warn`.
The line that switches the check off for good: `slurm_limits: 'off'` in the script_params (or `PYAPS_SLURM_LIMITS=off` for one run). A message
always names the way to submit as asked (`--slurm_limits warn` or `off`) and the way to make it fit (`--slurm_set STAGE.key=value`).

What "fits" means: on a cluster, the request is compared with the nodes of the partition the jobs go to (the default one): CPUs and memory must
exist **together** on one node (`--ntasks` and `--mem` are for one node), the time with the partition limit. On a machine without SLURM, with the
shares of 9.4.

### 9.3 Sites

The place is given a **site** name for the report and for a few defaults. The public table has two generic sites, `local` (no scheduler) and `slurm`
(any cluster); a named installation lists its own sites in `slurm_sites.yaml` (next to the script_params, `PYAPS_SLURM_SITES_FILE`, or the key
`slurm_sites_file`) with the same keys. A site is **detected** when every key of its `match` holds (`scheduler`, `kind`, `hostname`, `cluster`,
`any_node: {cpus, real_memory_mb}`, `node_names`; the entry with the most keys wins), or **named** with `--slurm_site`, `PYAPS_SLURM_SITE` or
`slurm_site`. Two clusters that share a hostname and a cluster name are told apart by the size of their nodes. A site may carry: `profile` /
`profile_openmp`, `ferre_openmp`, `limits`, `mem_fraction`, `cpu_fraction`, `expect` (what it should look like: a difference is reported, nothing
else) and, for an explicitly named site only, `max_mem_gb` / `max_cpus` / `max_time` that pin the limits.

The profile is chosen as before: `--slurm_profile` > `PYAPS_SLURM_PROFILE` > script_params `slurm_profile` > the table's `default_profile`. The
word `site` in any of them means "the profile of the detected site": `v2` or `v2_openmp` on a cluster (by the `ferre.x` build), `laptop` on a machine
without SLURM. Nothing chooses `site` by itself: **the profile of a production site does not change until someone writes `site`**.

### 9.4 Machines without SLURM (a laptop, a workstation, a container)

A bash run (`--hpc 1`, `aps-offline --no-slurm`) executes the stages one after the other, each with `--mp` processes. The limits are then the
machine's: `cpu_fraction` (site `local`: 75 %) of the CPUs it may use, divided by `--parallel_jobs` (several runs on one machine must not each
start 16 CPUs), and `mem_fraction` (80 %) of the RAM that is available now, divided the same way. A stage asking for more CPUs gets the share (with
`clamp`); a stage whose workers do not fit in the RAM gets **fewer workers** until they do, from the measured footprints of the table section
`footprint` (`base_gb + per_cpu_gb x CPUs`, worst measured PSS: RVS 1 + 4 GB per worker, IFU_Gal 4 + 2 GB per worker, IFU_ExGal 14 + 0.4 GB per worker (a patch lane peaks at 12-13 GB and more lanes only start while the memory allows), RR 2 + 1 GB per worker; the
fine FERRE grids 55 / 85 GB). The `mem_gb` of a row is a SLURM scheduling token that nothing enforces in a bash run, so it is not compared; a stage
without a footprint estimate is compared with its `mem_gb`. A stage that cannot fit even with one worker is reported (`error`: refused only if the
machine does not have that RAM at all). Not measured on a laptop: the footprints are those of the cluster runs, the `laptop` profile
(4 CPUs for RR/RVS/FR/PPXF, 2 for EMI, 1 for LS and L2merge, IFU_Gal 4, IFU_ExGal 6) is a starting point.

### 9.5 Operator commands (read only)

```
python -m PyAPS.aps_slurm_resources probe                    # what this machine and its SLURM look like        (--json: for --machine-file)
python -m PyAPS.aps_slurm_resources site                     # the detected site, the limits, the policy, the profile that applies
python -m PyAPS.aps_slurm_resources check --family MOS --res HR --stack stack      # does every stage fit? exit 0 / 1 needs a change / 3 can never run
python -m PyAPS.aps_slurm_resources check ... --machine-file other.json --target slurm   # what a run on ANOTHER machine would do (probe --json there)
```
All take `--profile`, `--limits`, `--site`, `--parallel-jobs`, `--ferre-exe`, `--script-params FILE`.

### 9.6 What is recorded

* the runner prints the place, the policy and every request that does not fit, with the value that would fit;
* `slurm_resources_effective.json`: `environment` (target, site and how it was found, the machine, the limits of one job, the shares, the policy and
  where it came from, messages), and for every stage `requested` (before the check), `adjustments` (what, requested, limit, fit, action
  warn / clamp / error, reason, the `--slurm_set` that would fit), `fits`, `footprint_gb`, `limits_policy`;
* the top of each stage's log: `PyAPS machine check: policy=... site=... target=... asked cpus=.. mem=.. -> granted cpus=.. mem=.. (...)`.

### 9.7 What it does not do

It does not change SLURM, does not look at the queue (it reports whether a node has the request free now in `check`, nothing more), does not
measure anything, and does not decide for a machine whose probe failed: it says "unknown" and submits as asked.

## 10. Depth-first order of the jobs (the first pair completes first)

**The problem.** The queue of the production clusters is `priority/multifactor` with only the age weight: priority = 10000 x (hours eligible / 24, at most 1) - nice,
i.e. FIFO by the time a job became **eligible** (a job waiting for its predecessor has priority 1 and no age). When the first pair of a night
reaches PPXF, EMI or L2merge, the RR of every later pair has been aging since it was submitted, so the early stages of all pairs run first and the
night ends with many half-finished chains; in the accounting of three weeks, 83-84 % of the late-stage jobs that waited more than 10 minutes
had been passed by an early-stage job of a later pair, and on a busy night the first complete result came 11-25 hours after the first submission.

**The scheme** (profile policy `priority: depth_first`, table section `priority`; **OFF in every profile**: switch it on with `slurm_priority: 'depth_first'`): every job of a chain gets a
non-negative `--nice` (a normal user cannot give a negative one) and a `--comment`:

```
nice = class base  +  stage_step x (stages still to come after this one)  +  pair_step x min(pair index, pair_cap)
       scheduled 4500   800           RR 4, RVS/PPXF 3, FR/EMI 2, LS 1, L2merge 0   100        8 pairs
comment = pyaps:<uapsid>:p<pair index>:<class>:<STAGE>          (squeue -o "%i %j %k %Q" shows the chain of every job)
```

The first pair and the last stages get the smallest values. Why this orders the jobs depth-first: the queue floors a priority at 1 (priority = max(1, 10000 x hours
waited / 24 - nice)) and breaks ties by job id, which is submission order, i.e. chain order. The early stages of later pairs carry the largest nice, so they stay at 1
for hours (up to 18 h of waiting at nice 7700) instead of passing the late stages of the first pair with their age; the late stages tie at 1 and go in chain order. The largest sum is 8500, below the 10000 of the age cap, so **nothing starves**: nice only
delays a job against jobs that became eligible less than nice/417 hours earlier, the age of every job keeps growing, and SLURM stays work-conserving (a
job with a large nice still runs when cores are idle). Nice orders; it does not reserve. The job names are unchanged (the operations layer finds jobs
by `<STAGE>_L2_<uapsid>`).

* **Pair index**: the order of the pair among the pairs submitted together (`--pair_index` / `PYAPS_PAIR_INDEX`; the DMS passes the position in its
  submit pass plus the chains already in flight when it has `max_inflight_obs`). **A retry keeps the index of the first generation of the chain**: the
  escalation (section 6) regenerates the chain, the runner reads the index from `slurm_resources_effective.json` and does not give the retry the worst
  priority.
* **Classes**: `scheduled` (the daemon, default; the nice above) and `offline` (a run somebody waits for: `aps-offline`, `--priority_class offline`,
  `PYAPS_PRIORITY_CLASS`). **An offline job gets no nice at all** (`classes.offline` has no base and no steps). The queue floors a priority at 1, so a job
  whose nice is above its age sits at 1 and ties are broken by job id; scheduled jobs (nice 4500 and more, i.e. at least 10.8 hours of age) therefore
  stay at the floor, while an offline job rises above 1 with its age at the first priority calculation (5 minutes on the production clusters) and passes
  all of them. A stage or pair step on an offline job would put it at the floor too, so the offline class has none (several offline pairs run in age
  order). Nice cannot pass jobs that have already waited longer than about 10 hours; for a stronger guarantee `aps-offline --boost scontrol|sudo` sets an
  explicit `Priority` on the queued jobs of the run (needs SLURM operator rights, or passwordless sudo), off by default; it never cancels, holds or
  suspends another job. A dependent stage of an offline chain becomes eligible with age 0, so it waits one priority calculation behind the floor
  before it passes the scheduled jobs; `--boost` removes that wait.
* **Cap**: `max_inflight_obs` in the DMS config (existing) holds new pairs until chains in flight have finished; off by default.
* **Switches**: `--slurm_priority off|depth_first`, `PYAPS_SLURM_PRIORITY`, script_params `slurm_priority`, then the profile's `priority.policy` (off everywhere). In a
  bash run (`--hpc 1`) nothing is queued and no nice is given.
* **Stronger modes that are NOT built in** (they would stop other jobs): holding or suspending the pending/running jobs of the daemon while an offline
  run executes, and preemption (`PreemptType`, a QOS). They need the cluster configuration and the owner's decision.

**What the experiments showed (2026-11-05, a private single-node SLURM with the production priority settings, age compressed 10 x, 24 CPUs, 12 chains of seven jobs submitted one after the other).**
With the old allocation (`default`: CPU-bound) the problem reproduces: all 12 chains are open at once, the first one completes after 80 % of the whole span, and 20 of 24 late-stage jobs that waited more than ten minutes (real) were passed by an early-stage job of a later pair. With `v2` (memory requests: the node's memory binds first) it does not occur (3 chains open, the first complete after 15 % of the span). **The nice offsets did not change the order in either case** (same completion times to the second as without them), and a nice-only offline class did not move an offline chain ahead of the queue (it finished last of nine, like an unprioritised chain); an explicit `Priority` on the queued jobs (`--boost`) did (second of nine, 259 s instead of 1044 s). The cause that was seen: the jobs that overtake are small (8 CPUs) and fill the slots that free up one at a time while the big dependent jobs (10-20 CPUs) wait for room; with every priority at the queue's floor the nice does not reorder them. So **nice offsets are shipped but OFF**, and are not a promised fix; what does reduce open chains is holding new pairs back (`max_inflight_obs`, section above) and, for an offline run, `--boost`. Details and numbers: the private operations document `doc/ops/SLURM_RESOURCES_OPS.md`, section 11.

## 11. How the profile is resolved on a pod: the DMS config and where each value comes from

**One function decides, and every consumer calls it.** `py/PyAPS/aps_dms_settings.py` (standard library only) holds the precedence chain and the reader of the
DMS config; `aps_slurm_resources.resolve()` (the runner, `aps-offline`, the operator commands `show | explain | check | site`) and `apsPlot.export.resolve_mode()`
(the static images) both use it. Before 1.2.0 of the module the operator commands and a runner started by hand saw only the environment and
`script_params.yaml`, so on a pod that runs profile `v2` they printed `default`.

### 11.1 The order

For each of the four settings that have all of the sources, the first one that is set wins:

| | 1. option | 2. environment | 3. **DMS config key** | 4. script_params key | 5. otherwise |
|---|---|---|---|---|---|
| profile | `--slurm_profile` | `PYAPS_SLURM_PROFILE` | `slurm_profile` | `slurm_profile` | the profile of the detected site if the word is `site`, else the table's `default_profile` |
| limits policy | `--slurm_limits` | `PYAPS_SLURM_LIMITS` | `slurm_limits` | `slurm_limits` | the site's (machines without SLURM), the profile's, `warn` |
| site | `--slurm_site` | `PYAPS_SLURM_SITE` | `slurm_site` | `slurm_site` | detected from the machine |
| plot export | `--plot_export` | `PYAPS_PLOT_EXPORT` | `plot_export` | `plot_export` | `off` |

The runner's order is **unchanged for a run the DMS starts**: the DMS passes its own config values as options (`--slurm_profile v2`), and an option is
rank 1. The new rank 3 only matters where nothing passed the option: a runner started by hand, `aps-offline` (its profile is the pod's unless
`--slurm-profile` is given), and the operator commands. A fifth key, `max_inflight_obs`, is the DMS's own cap on the observations in flight; the tools only
report it.

### 11.2 What is read from the DMS config

Exactly five keys: `slurm_profile`, `plot_export`, `slurm_limits`, `slurm_site`, `max_inflight_obs`. Nothing else of the file is read or kept (it holds the
database password), and the tools never print the file. Where it is: the file named by `PYAPS_DMS_CONFIG` (`none` switches the lookup off), else
`$HOME/.private_env_configs/PyAPS_dms_config.json`. A missing, unreadable or malformed file means "no values" and a one-line note, never an error. A caller
that passes its own `env` mapping (the tests do, with `env={}`) sees only the file that mapping names, so a test never reads a pod's file; the tests of the
repository set `PYAPS_DMS_CONFIG=none` for every test (`tests/conftest.py`).

A value in the DMS config is treated as **the pod's configured value**: it ranks above script_params because it is what the DMS puts on the runner's command
line anyway. The words `auto` (profile) and `none` mean "not set" and the next source decides.

### 11.3 See which source set each value

`show`, `explain`, `check` and `site` print a block `settings` (and the runner prints the same block at the top of every run):

```
settings  : first set wins: option > environment > DMS config > script_params > site / profile / table default
            DMS config: ~/.private_env_configs/PyAPS_dms_config.json (read for slurm_profile, plot_export, slurm_limits, slurm_site, max_inflight_obs only)
            slurm_profile     v2           set by DMS config slurm_profile   (also set, overridden: script_params slurm_profile=default)
            slurm_limits      clamp        set by profile v2
            slurm_site        imperial     set by detected (...)
            plot_export       off          set by default
            max_inflight_obs  6            set by DMS config max_inflight_obs   [the DMS caps the observations in flight with it; the runner does not read it]
```

`--json` of `site` and `check` carries the same list (`settings`). To see what another pod's file would do, name a copy with `--dms-config FILE` (only the five
keys are read); `--dms-config none` shows what the tool says without it. `--plot-export` and `--script-params FILE` are the other inputs.

### 11.4 Why `script_params.yaml` is not the place on a pod

On the pods `configs/script_params.yaml` is **rendered**, not edited: `pyaps_site_setup.sh --ensure` (cron, every 15 minutes) runs `pyaps-admin local --yes`
(PyAPS_admin `sync.apply_configs`), which renders `configs/script_params.yaml.example` with `sites/script_params.rules.yaml` for the pod's environment and
replaces the file whenever the text differs (the old one is kept as `script_params.yaml.bak`). The step is *deferred* while SLURM jobs of the pod are queued or
running or the runner holds its lock, so a hand edit survives until the first idle pass and then disappears; the public template carries `slurm_profile`,
`slurm_limits`, `slurm_site` and `plot_export` only as commented lines, so none of them survives a render. Put the pod's values in the DMS config (host-local,
git-ignored, never rendered, not mirrored between the sites by `pyaps_sync.sh`) or in the environment of the service.

## 12. Re-measurement of 2026-11-06 (after the seg2d fix): statistics, what changed, how to repeat it

**Source.** `sacct -u aps -S 2026-10-01` on the production pod (3 nodes of 50 / 90 / 90 CPUs; 3 800 L2 jobs, 3 430 COMPLETED) and, read only, on the development pod (2 nodes; 2 059 jobs since 2026-09-01, 1 890 COMPLETED).
`TotalCPU / Elapsed` is the number of busy cores, MaxRSS the largest of the steps of the job. Two cautions decide how to read the numbers:

* **Almost all the history was run with the old allocation and with the static plot images on** (the images became off by default on 2026-11-04, `v2` went live on the production pod on 2026-11-05). Elapsed times, busy cores and MaxRSS before that date are upper bounds for today's code. Only 13 MOS pairs and 8 IFU observations (night 20261104) ran with `v2` and the images off.
* **MaxRSS is not the footprint.** `jobacct_gather/linux` sums the RSS of every worker, so shared pages are counted once per process (IFU_ExGal 107-513 GB for a job whose PSS is 14-20 GB). It is an upper bound of the true memory, never a number to size from; the memory column stays the PSS peak of the rehearsal with a margin (section 7). Neither pod enforces memory (`TaskPlugin` empty, no `OverMemoryKill`) and neither confines CPUs, so today a request is a scheduling token: a request too large costs queue time, one too small costs nothing until the cluster is changed.

All completed jobs; the flood run of OB 26133 (uapsid aadb576526dd, 226 noise patches, 8.4 h) is excluded here and listed below.

| stage | n | elapsed min p50 / p95 / max | busy cores p50 / p95 / max | MaxRSS GB p95 / max | dev pod: n, elapsed p95 / max | requested in `v2` (cpus / mem / time) | p95 x 2 |
|---|---|---|---|---|---|---|---|
| RR | 492 | 96 / 201 / 226 | 6.7 / 8.5 / 12 | 79 / 151 | 260, 129 / 163 | 8 / 24 GB / 6 h (HR 16 / 36) | 6.7 h |
| RVS | 449 | 5.3 / 19 / 46 | 19 / 30 / 43 | 84 / 136 | 246, 16 / 89 | 12 / 72 GB / 3 h | 0.6 h |
| FR | 420 | 96 / 986 / 1988 | 6.3 / 6.8 / 7.1 | 4.8 / 49 | 246, 186 / 254 | 9 / 8 GB / 30 h | 32.9 h |
| PPXF | 446 | 27 / 85 / 205 | 7.1 / 9.9 / 11 | 43 / 46 | 245, 72 / 131 | 10 / 16 GB / 6 h | 2.8 h |
| EMI | 439 | 29 / 83 / 111 | 3.6 / 5.3 / 5.9 | 91 / 98 | 232, 76 / 94 | 6 / 28 GB / 4 h | 2.8 h |
| LS | 442 | 1.4 / 2.5 / 2.9 | 0.9 / 1.0 / 1.3 | 3.1 / 10 | 232, 3.2 / 4.4 | 2 / 6 GB / 30 min | 5 min |
| L2merge | 441 | 0.8 / 1.1 / 1.3 | 0.3 / 0.5 / 0.6 | 3.4 / 3.4 | 221, 1.3 / 4.3 | 1 / 8 GB / 15 min | 2 min |
| IFU_PR | 106 | 22 / 97 / 131 | 1.3 / 2.0 / 2.1 | 19 / 24 | 78, 131 / 415 | 3 / 24 GB / 6 h | 3.2 h |
| IFU_ExGal | 94 | 55 / 215 / 263 | 4.0 / 35 / 42 | 926 / 1154 | 58, 218 / 374 | 40 / 80 GB / 10 h | 7.2 h |
| IFU_Gal | 98 | 83 / 517 / 696 | 1.0 / 1.3 / 4.3 | 73 / 134 | 72, 384 / 453 | 4 / 48 GB / 16 h | 17.2 h |

(The last column is the rule "time = p95 x 2" applied to the production numbers. The "requested" column is the allocation in force before this change; the new IFU values are below.)

The non-completed jobs are not in the table: 285 FAILED (46 chains on 2026-10-07/08, an incident, not a size problem), 53 CANCELLED and 31 TIMEOUT on the production pod (20 FR at a limit of 8 h that was in force then, 1 FR / 3 RR / 3 PPXF / 3 EMI at the 48 h of the old allocation, 1 IFU_Gal at 18 h); on the dev pod 2 PPXF at 48 h. No job ended OUT_OF_MEMORY.

**With `v2` and the images off (2026-11-05/06, production pod).** RR 13 jobs: 10.0 min, 6.5 busy cores; RVS 13: 3.2 min, 8.7-9.5 busy; FR 12: 51 min (p95 67), 6.9 busy; PPXF 13: 2.9 min, 7.1-7.5 busy; EMI 13: 1.9 min, 3.1-3.5 busy; LS 13: 1.5 min, 0.8 busy; L2merge 12: 0.8 min, 0.3 busy. The MOS allocations of section 7 are confirmed and **nothing was changed in MOS**: every stage uses 65-90 % of its CPUs except EMI (55 %), LS (40 %) and L2merge, where a smaller request would save one or two cores for one to three minutes. Queue waiting time on both pods is zero (the cluster is nearly idle), so a smaller request has no effect to measure.

**The IFU stages, the eight observations of night 20261104** (products = patch products of ExGal / of Gal; all rows but the one in bold ran with the code before the seg2d fix):

| OB | patches (products) | IFU_PR | IFU_ExGal (40 CPUs asked) | IFU_Gal (4 CPUs asked) | note |
|---|---|---|---|---|---|
| 24266 | 1 / 0 | 5.7 min | 4.3 min, 1.0 busy | 0.6 min | |
| 24282 | 13 / 1 | 26 | 29 min, 2.1 busy | 6.5 min | |
| 24316 | 18 / 15 | 27 | 49 min, 3.9 busy | 347 min, 1.03 busy | MaxRSS 10.9 GB, the largest Gal figure |
| 25931 | 43 / 25 | 44 | 70 min, 2.6 busy | 46 min | |
| 24311 | 128 / 10 | 123 | 222 min, 2.5 busy | 38 min | the real field with the most patches |
| 26133, before the fix (flood, aadb576526dd) | 180 / 46 | 187 | 319 min, 2.4 busy | 121 min | pathological, excluded above |
| **26133, after the fix (22e3363bdccf)** | **4 / 7** | **13.4 min** | **8.6 min, 2.8 busy** | **39 min, 1.05 busy** | 11 patches instead of 226 |
| 25921 (night 20261030) | 4 / 5 | 11.9 | 11.1 min, 1.6 busy | 10.7 min | |

**Superseded for IFU_Gal and IFU_ExGal by section 13 (2026-11-13).** Findings of 2026-11-06: IFU_Gal is serial (1.00-1.23 busy cores in all 8 jobs, 1.0-1.3 in 98 production jobs; 1.8-23 min per patch product, so the longest production run, 696 min, and a field with many real patches justify the 16 h limit), its MaxRSS is 4.9-10.9 GB with 4 CPUs; IFU_ExGal processes the patches one after the other and parallelises the bins inside a patch, so it used 1-4 busy cores on these observations although 40 were asked (before 2026-11-04 it used up to 42 on the production pod, with the plot images on and on other fields), MaxRSS-sum 107-513 GB (PSS 14-20 GB); IFU_PR 1.2-1.3 busy cores, MaxRSS 11-15 GB. Only OB 26133 has run since the seg2d fix (n = 1), so the effect of the fix on run times rests on that observation and on the per-patch costs above.

**What changed in `v2`** (the `default` profile is untouched, the golden tests are unchanged):

| stage | before (cpus / mem LR / mem HR) | now | why |
|---|---|---|---|
| IFU_Gal | 4 / 48 / 96 GB | **2 / 32 / 64 GB**, time unchanged (16 h, HR 24 h) | serial on the single-thread `ferre.x`: 1.0-1.3 busy cores (p95 1.3); MaxRSS at most 10.9 GB at 4 CPUs (an upper bound of the PSS), 32 GB is 2.9 x that and 1.6 x the footprint estimate at 2 CPUs |
| IFU_ExGal | 40 / 80 / 128 GB | 40 / **48 / 80 GB**, CPUs and time unchanged | PSS peak 14-20 GB with up to 40 workers (rehearsal, images off): 48 GB is 2.4 x |
| `v2_openmp` IFU_Gal | (inherited 4 / 48 / 96) | **4 / 32 / 64 GB**, status `unmeasured` | an OpenMP `ferre.x` uses the CPUs as threads (one rehearsal: 7 cores for 4 min against 1.8 cores for 12 min with 2 CPUs); nothing has run with `v2_openmp` yet, so the CPUs are kept, memory as `v2` |

Expected effect: per large-IFU observation the IFU_Gal job holds 2 fewer cores and 16 GB less, the IFU_ExGal job 32 GB less. For the eight observations of 2026-11-05/06 that is 20 core-hours (16 without the flood run), 162 + 381 GB-hours of reserved memory; for OB 26133 after the fix 1.3 core-hours. These are scheduling reservations, not measured savings of work: nothing queued behind them on either pod, and neither pod enforces memory or CPU confinement.

**What was left unchanged, and why.**

* All MOS rows (confirmed by the 13 pairs; the only candidates were LS 2 -> 1 CPU and EMI 6 -> 4, one or two cores for one to three minutes).
* All **time limits**. The rule "p95 x 2" agrees with the present IFU limits (IFU_ExGal 7.2 h against 10 h, IFU_Gal 17 h against 16 h), and for RVS / LS / L2merge it would give 0.6 h / 5 min / 2 min, but those p95 contain the images-on runs and a time limit only has to catch a hung job (nothing waits behind it, a job killed by it is retried once with 1.5 x and costs a whole stage). IFU_PR keeps 6 h although the production p95 x 2 is 3.2 h: **the dev pod once ran an IFU_PR for 415 min** (2026-10-09, before the seg2d fix; the cause was not looked at), longer than 6 h; if the fix is as complete as OB 26133 suggests the limit can be tightened after a few weeks of data.
* IFU_ExGal CPUs (40): the parallelism is per bin inside a patch and depends on the field; the production pod showed up to 42 busy cores before the images were switched off and the dev rehearsal 40 on one field. Only eight observations of small or medium fields ran since, so there is no evidence to lower it.
* IFU_PR (3 CPUs, 24 / 36 GB): 1.2-2.1 busy cores, MaxRSS 11-24 GB, PSS up to 27 GB for a heavy HR cube.
* The `footprint` table (IFU_Gal 4 + 8 GB per CPU) is used for machines without SLURM and was not changed: the 8 jobs at 4 CPUs had MaxRSS far below it, but those fields were small.
* Everything for `v2_openmp` other than IFU_Gal memory: there are no jobs with that profile in the accounting of the dev pod (only 2 RR and 1 IFU_PR since 2026-11-05); its FR history comes from the default allocation (10-20 CPUs asked, 18.8 busy cores p50, 39 p95), which does not tell how 18 CPUs behave.

**How to repeat the measurement.**

```
sacct -u aps -S 2026-11-05 -n -P --units=G -o JobID,JobName%40,State,ElapsedRaw,TotalCPU,MaxRSS,ReqCPUS,ReqMem,AllocCPUS,Submit,Start,End,Timelimit,NodeList > acct.psv
```

Job rows give Elapsed and TotalCPU; the `.batch` step carries MaxRSS (take the largest over the steps of a job). Job name `<STAGE>_L2_<uapsid>`; busy cores = TotalCPU / Elapsed; skip a run you know is pathological (the flood run), jobs that did not end COMPLETED and runs older than the profile you want to judge. Per stage report n, p50, p95, max; propose time = max(p95 x 2, longest run x 1.25), memory from the PSS rehearsal (not from MaxRSS) with a margin of at least 1.5, CPUs from the p95 of the busy cores plus one. The PSS of a stage is measured by sampling `/proc/<pid>/smaps_rollup` of the job's processes every 10 s (the rehearsal of section 7). After an IFU change count the patch products (`<OB>/*_P####_ppxf.fits`, `ferre_<headname>_P####.fits`) next to the run times to separate cost per patch from cost per observation. The `IFU_PATCH_SUMMARY` line of the stage log (`doc/aps_ifu_seg2d_robust.md`) states the number of patches directly for runs made after 2026-11-06.

**The machine check names its profile.** The first lines of every stage log now read `PyAPS machine check: policy=clamp site=imperial target=slurm profile=v2 asked ...`, so the profile the check was made under is visible next to the policy. `aps_runner.slurm_time_arg` (compatibility wrapper for old callers) took the profile `default` whatever the pod's DMS config said; it now uses the same resolution as every other consumer (section 11).


## 13. IFU_Gal and IFU_ExGal: patches side by side (2026-11-13)

### 13.1 What changed in the code

Both stages used to work on one patch after another (IFU_Gal: and inside a patch one bin after another, 9 grids one after another: 1.0-1.3 busy cores; OB 14699 of
2026-03-15, 44 patches, 582 bins: 15 patches in 24 h on the -O0 `ferre.x`, TIMEOUT). The patches of a table are independent (every patch writes its own
`<headname>_PNNNN` files and work directory), so both now run several at the same time, with one shared driver (`py/PyAPS/aps_ifu_patch_pool.py`):

* The CPU budget `mp` of the job (`--mp_Gal` / `--mp_ExGal` = the CPUs of the sbatch line) is split into **lanes**. A lane is one process that takes patches from a queue and gives each patch `capacity` inner workers (IFU_Gal: RVS pool and FERRE bins at a time, one single-thread `ferre.x` per bin and grid; IFU_ExGal: the PPXF / EMIPPXF / LS pool). The capacities add up to at most `mp`: **never more worker processes than the job has CPUs**, and the pool of a patch is not counted a second time.
* `--patch_workers` (script_params `IFU_patch_workers`): `0` / not given = automatic, `1` = one patch after the other in table order with `mp` workers inside (the behaviour before this change), `N` = N lanes (at most `mp`). With `mp = 1` every setting is the old serial loop, run in the same process.
* Automatic: the stage simulates one lane of `mp` and 2 .. min(patches, `mp`, memory room) lanes from the estimated bins of every patch (area of the aperture: 0.19 bins per arcsec^2, at least 2) and takes the fewest lanes within 5 % of the shortest estimated wall time; many small patches use one lane per CPU, a few big patches use few lanes with several workers each. IFU_ExGal lanes get at least 2 workers each (the PPXF / EMIPPXF / LS modules have a pool path, the one that always ran in production, and a serial path for one worker that is separate code).
* **IFU_ExGal does not plan the widths from the area** (2026-11-14, section 13.7): the number of Voronoi bins follows the S/N, not the aperture, so every patch takes its inner workers from the budget after its binning (`WidthBroker`). IFU_Gal plans from the area (bins do follow it there) and, since 2026-11-14, also tries uneven lane sets (a wide lane for one big patch) when they are more than 5 % shorter than equal lanes; the first patches go to the lanes by width, the widest lane takes the largest.
* Patches start **largest first** (longest processing time first): the sizes are very uneven (OB 14699: 91, 58, 54, 52, 32, 37, 32, 27, 21, 22, 19, 16, 14, 11, 9 bins, then 29 single stars of 2-4 bins), so the small ones fill the gaps.
* **Memory caps the lanes**: a lane holds the aperture of its patch (IFU_Gal about 2.5 GB, IFU_ExGal 12 GB at its peak while it reads and bins), so the number of lanes is at most 90 % of the memory of the job (`SLURM_MEM_PER_NODE`, the cgroup limit or the free RAM, the smallest) divided by that; the CPUs no lane can use go to the workers inside the lanes. An explicit `--patch_workers N` above the room is kept and warned about.
* The plan is printed (`INFO: IFU_Gal patch plan: 44 patches on 7 lane(s) with 2+2+2+1+1+1+1 inner worker(s) ...`); lines a lane prints start with `[P0014] `; every patch ends with `INFO: GAL patch id=14: written in 412 s`.
* A patch that fails (an exception, `sys.exit`, a worker killed for memory) is reported failed and the others go on. The parent adds the answers up in table order: `<headname>_<stage>_patches.json`, the `IFU_PATCH_SUMMARY` line and the exit rule (fail when at least 10 patches were attempted and none was written, or when every patch failed) are as before and do not depend on the order of completion.
* Nothing is shared between patches: every file name carries the patch id (checked on 10 patches in one directory: the only files without it are the stage summary JSON, written by the parent, and the `figs_Gal` directory); the cube is read per patch as before but only the spaxels of the aperture are kept (8 GB of file pages are shared by all lanes); the `fr_wd_*` FERRE work directories are per patch.
* The Monte-Carlo errors of PPXF and the line-strength MCMC are random; every lane starts from a fresh random state (workers forked from one parent used to share one). `PYAPS_PATCH_SEED=<n>` seeds numpy with n + patch id at the start of each IFU_ExGal patch, for comparing runs (reproducible only with `mp` 1: the workers of a pool all inherit the same state and take bins in the order of their timing).
* Two latent faults of the serial path of the EMIPPXF stellar re-fit were found by the equivalence run and fixed (they could not show in production because a pool of 40 never takes that path): `ExGalutil.prettyOutput_Error` did not exist (the error handler itself raised `AttributeError` and failed the patch) and the pPXF bounds were not validated as in the pool path.

### 13.2 Sizing rule

**CPUs = the number of worker processes the stage can keep busy; every CPU of the request is one worker.** Not the busy cores of the previous code: the 2026-11-06 re-measurement read 1.0-1.3 busy cores in IFU_Gal and cut the request from 4 to 2 CPUs, but that was the serial code (one patch, one bin, one grid at a time) and not the size of the job. The design of the stage is **one `ferre.x` process per grid**: 9 grids per bin + the python process = **10 CPUs** (the MOS FR stage asks 9 for the same reason: `aps_ferre.py` runs a thread pool over the 9 grids). The 20 CPUs of the legacy values (LIFU; 10 for the mini IFU) and the 30-40 of IFU_ExGal are now spent as lanes and workers, no more than that.

| | IFU_Gal | IFU_ExGal |
|---|---|---|
| `v2` CPUs | **10** (4 on 2026-11-13 morning, 2 from 2026-11-06) | **40** (24 on 2026-11-13, back to 40 on 2026-11-14: OB 26037 with one giant patch ran 28 % slower on 24; 40 before the lanes) |
| memory | 48 GB (HR the same): footprint 4 + 2 GB per worker = 24 GB at 10 CPUs, measured peak 20 GB | 96 GB (HR 160): 7 lanes of 12 GB, footprint at least 14 + 0.4 GB per worker |
| time | 16 h (HR 24 h), unchanged: the longest known field needs about 3 h at 10 CPUs and the limit only has to catch a hung job | 10 h, unchanged (longest completed run 374 min with 1-4 busy cores) |
| `v2_openmp` | same as `v2`: `mknml_ifu` starts every `ferre.x` with NTHREADS=1 and wall times with NTHREADS 1, 2 and 4 were identical on single-object bins, so OpenMP threads do not help | same as `v2` |
| `laptop` | 4 CPUs, 48 GB token (the limits policy `clamp` lowers it to the machine and its RAM) | 6 CPUs |
| `default` (legacy) | 20 CPUs LIFU / 10 MIFU, HR doubled, no memory request: untouched (golden); the lanes follow the memory SLURM gives (1 GB per CPU) | 30 / 10 CPUs, 240 GB LIFU: untouched |
| MIFU `v2` | 10 CPUs (20 in HR by the old rule), 72 GB, unmeasured | 10 CPUs, 80 GB, unmeasured |

### 13.3 Measurements (Imperial pod, 2026-11-13, nice 10, the pod busy with other jobs)

IFU_Gal, OB 14699 night 20260315 (LIFU HR, patches of the first 15 and single stars), `ferre.x` -O2 build:

* cost: 62-65 s per bin and worker serial (RVS 1-3 s, 9 grids 6-8 s each), 30-40 s per patch to read and bin the aperture (maximum RSS 8.2 GB, mostly the file pages of the two 4 GB cubes);
* patches 14, 15, 43, 44 (11 + 9 + 1 + 1 bins): 26 min with `mp` 1; patches 13, 14, 15, 43, 44 on 8 CPUs (4 lanes of 2): 8.2 min (the 14-bin patch 492 s); 10 patches of 1-4 bins on 10 CPUs (10 lanes): 5.1 min;
* PSS: 13.2 GB with 4 lanes (8 CPUs), 19.6 GB with 10 lanes (10 CPUs): about 2 GB per worker process; the 36 GB of the old audit were 20 RVS workers;
* the whole OB (582 bins estimated, 44 patches) by the plan: 68 bin-times on 10 CPUs, about 75 min (ideal 58 bin-times); one lane of 10 workers would take 131 bin-times; serial on the -O0 build it was 29 h.

IFU_ExGal, OB 26140 (LIFU LR, patches 3, 11, 14, 15, galaxies of z 0.09-1.2, PPXF + EMIPPXF + LS):

* 4 lanes of 2 workers on 8 CPUs: 8.1 min wall (patches 141, 475, 186, 240 s); peak PSS 48.8 GB in the first minute, while the four lanes read and bin their apertures (14.6 GB RSS in the largest process), 8 GB after 90 s, 1.6 GB at the end; with 4 lanes of 1 worker 51.5 GB. That is 12-13 GB per lane: the number of lanes is capped by the memory (`lane_cap`).

### 13.4 Equivalence of the results

The products of the patches are the same as before. Compared column by column (every table of every FITS product of the patch, the merged `*_APS.fits` included; maximum absolute and relative difference of every numeric column, NaN pattern, strings):

* IFU_Gal: the serial run of the old code (-O0 `ferre.x`, the 24 h run of 2026-11-12/13) against the new code with `mp` 8 (lanes) for patches 13 and 14: **0 differing columns** in `ferre_`, `ferre_spec_`, `rvs_`, `rvsspec_`, `BINSpectra_linear`, `_table` and the three tables of `_APS.fits` (max abs difference 0, rel 0; the files differ only in header keywords such as the work path). New serial (`mp` 1, `--patch_workers 1`) against new lanes (`mp` 8 and `mp` 10) for patches 14, 15, 43 and 44: **0 differing columns**; patch 15 of the old run has RVS only (its FERRE was cut by the timeout) and its RVS table is identical.
* IFU_ExGal: the production run of 2026-11-07 (old code, pool of 40) against the new code with 4 lanes of 2 workers for patches 11 and 14: every column that is not an error column of a Monte-Carlo fit is identical (bit for bit); the `ERR_*` columns of PPXF, EMIPPXF and LS (Monte-Carlo and MCMC: random) differ, as they do between two runs of the old code. The same run with the final code (2 lanes of 2 workers, patches 3 and 14) and the IFU_Gal run with 4 lanes of 1 worker give the same result. The serial path of IFU_ExGal (`--mp_ExGal 1`, not used by lanes) after the PPXF fix: PPXF tables identical to the pool path; EMIPPXF and LS columns that are not Monte-Carlo errors are identical for patch 14 and differ by at most 7.5e-4 (relative) for patch 3 (4 bins); the cause of that remainder was not found.

### 13.5 Standalone: a workstation

```
python aps_ifu_Gal.py ... --mp_Gal 8                       # up to 8 worker processes: patches side by side x bins inside a patch; the plan is printed
python aps_ifu_Gal.py ... --mp_Gal 8 --patch_workers 1     # the old behaviour with 8 bins at a time
python aps_ifu_Gal.py ... --mp_Gal 1                       # the plain serial loop (the default of the stage scripts)
aps_runner.py ... --slurm_profile laptop                   # CPUs 4 (IFU_Gal) / 6 (IFU_ExGal), clamped to the machine (75 % of the usable CPUs, 80 % of the RAM)
```

Export `OMP_NUM_THREADS=1` (the generated scripts do). `ferre.x` must be the upstream build f50a006 or later ([externals/README.md](../externals/README.md)): `python tools/check_env.py` warns when it does not link libgomp, the signature of the -O0 build (3 times slower). `mp` is never above the machine: the `local` site clamps CPUs to a share of the machine and the workers to what the RAM holds (footprint 4 + 2 GB per IFU_Gal worker), and the lanes follow the free RAM.

### 13.6 Risks and what to watch

* The memory of IFU_ExGal lanes (12-13 GB at the peak of the first minute, measured on one LR observation; HR 160 GB is a scale-up, not measured). The lane cap protects a run whose job asked for less memory, and the first nights give the numbers: `sacct` MaxRSS of IFU_ExGal and the `INFO: IFU_ExGal patch plan` line of the log.
* The IFU_Gal run time estimates use a bins-per-area ratio taken from one observation; a wrong ratio only costs wall time (the order of the patches), never a result. For IFU_ExGal the area is NOT used for the widths (13.7).
* SLURM here has `TaskPlugin=task/none`: `--mem` is only a reservation of the scheduler, not a limit, and `sacct` MaxRSS adds up the forked workers (the old 40-CPU ExGal runs show 240-265 GB under a 48 GB request), so it is not a peak. The lane cap of 7 comes from the request (96 GB x 0.9 / 12 GB). A true peak needs sampling of the node during a run (not done).
* IFU_Gal at 10 CPUs holds 10 of the pod's 90 cores per observation for about 1-2 hours (24 h before for the big fields); 2 or 3 observations at a time fill a node.

### 13.7 Measurements of 2026-11-13/14 (Imperial pod) and the width of an IFU_ExGal patch

**Re-queued night 20261112** (new code, resources v2). IFU_ExGal: OB 16940 (41 patches, 6 lanes of 4) **19:17** on 24 CPUs against 1:33:30 for the old code on 40 CPUs; OB 26037 (11 patches: one of 606 Voronoi bins, ten single stars) **53:14** on 24 CPUs against 41:41 for the old code on 40, patches one after the other because the plan from the area said "one lane is as fast". IFU_Gal: OB 16940 25:52 (the old run was cancelled at 2:22 without a result), OB 26037 34:43.

**Why the area cannot plan ExGal lanes.** 384 patches of 29 OBs (Imperial): aperture area 3974 arcsec^2 gave 2 Voronoi bins, 4035 gave 505, 6908 gave 606, 12154 gave 3326 (the S/N sets the bins). With a wrong cost the equal lanes slow the giant patch by the same factor as they are narrower, so the planner chose one lane. IFU_Gal bins follow the area (420 predicted, 495 measured), so IFU_Gal keeps the planner (now also with uneven lanes).

**How IFU_ExGal does it now** (`aps_ifu_patch_pool.WidthBroker`, `fill_widths`; default, `PYAPS_PATCH_WIDTH=static` returns to the area plan, `--patch_workers 1` is the serial code): lanes = what the memory allows (7 at 96 GB; `--patch_workers N` = N); after its binning a patch (`ifu_ExGal_prepare` returns `n_bins`) asks the broker for inner workers. The first patch of every lane waits for the others and the budget is shared: everyone gets 2, then one worker at a time to the patch that would finish last (equal patches get equal widths; a giant patch gets what the small ones leave). A patch that starts later takes `min(bins, free workers / lanes without workers)`, at least 2, and waits while fewer than 2 are free. The held widths never add up to more than `mp`; a skipped or failed patch tells the barrier it has no work; a lane that dies gives its workers back. Printed: `INFO: IFU_ExGal patch plan: 11 patches on 7 lane(s); the inner workers of each patch are set when its bins are known ...` and per patch `INFO: ExGAL id=1: 606 bin(s) -> 28 inner worker(s)`.

**Test of the widths** (scratch re-run of OB 26037 ExGal, 40 CPUs, job 5636, outside the production chain): **28:03** (patch 1: 606 bins on 28 workers 1677 s; ten small patches 2 workers each, 87-280 s side by side) against 53:14 (24 CPUs, one lane) and 41:41 (40 CPUs, old code): 33 % less than the old code. Only the `ERR_*` columns (unseeded Monte Carlo) differ from the production products.

**Gain of the patch lanes alone** (OB 14699 IFU_Gal, 10 CPUs, scratch run job 5634 against the production run): `--patch_workers 1` 2:02:08, lanes 1:15:15: **1.62 x** (-38 % wall time; the planner had predicted 2.4 x). The 110 table columns compared on five patches are identical. The rest of the gain over the 24 h time-out (15 of 44 patches) comes from the -O2 FERRE build (2.9 x), bins at a time and the crowd-adaptive deblending (49 patches, 17 % fewer bins).
