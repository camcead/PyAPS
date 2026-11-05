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
| IFU_ExGal | 40 / 40 | 80 / 128 GB | 10 h | up to 40 busy cores, 13 on a small cube (run 10 min); PSS 18 GB (the sum of the workers' RSS is 212 GB: shared pages counted per process) |
| IFU_Gal | 4 / 4 | 48 / 96 GB | 16 h / 24 h | serial on the single-thread `ferre.x` of the production pod (1.0-1.3 busy cores, 4.3 at most); on an OpenMP `ferre.x` a small cube used 7 cores for 4 min against 1.8 cores for 12 min with 2 CPUs |
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
`ferre.x` is built with OpenMP: two threads for each of the 9 grids). The rehearsal values and what was not measured are in the operator
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

Chosen by (first that is set): `aps_runner.py --slurm_limits`, `PYAPS_SLURM_LIMITS`, the script_params key `slurm_limits`, the `limits` of the
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
`footprint` (`base_gb + per_cpu_gb x CPUs`, worst measured PSS: RVS 1 + 4 GB per worker, IFU_Gal 4 + 8 GB per worker, RR 2 + 1 GB per worker; the
fine FERRE grids 55 / 85 GB). The `mem_gb` of a row is a SLURM scheduling token that nothing enforces in a bash run, so it is not compared; a stage
without a footprint estimate is compared with its `mem_gb`. A stage that cannot fit even with one worker is reported (`error`: refused only if the
machine does not have that RAM at all). Not measured on a laptop: the footprints are those of the cluster runs, the `laptop` profile
(4 CPUs for RR/RVS/FR/PPXF, 2 for EMI, 1 for LS and L2merge, IFU_ExGal 6) is a starting point.

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
