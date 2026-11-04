#!/usr/bin/env python3
"""
aps_slurm_resources.py - the ONE table of the SLURM resources (CPUs, memory, time limit) of every PyAPS job.

Before this module the resources of the jobs were spread over the code: the CPU counts were handed to ``aps_runner.py`` as
``--mp_MOS 8,12,18,10,20,6,1`` style lists by the caller, the runner doubled them for HR and halved FERRE for single
exposures in place, the memory of three stages was an ``if`` on the grid prefix, the time limit came from a dictionary in
the runner, and no other job had a memory request at all.  Now every decision comes from a YAML table
(``configs/slurm_resources.yaml``) read by this module:

* ``profiles``      named sets of rows: stage x observation class -> cpus, memory (GB), time limit.  ``default`` reproduces,
                    value for value, what the runner did before the table existed; other profiles (``v2``) hold new
                    allocations.  A profile may ``extends`` another and then lists only the stages that differ.
* ``sites``         optional per-site limits (largest memory and CPUs of one node, longest time limit) used to stop a request
                    that no node can ever run; without a ``site`` the limits are read from SLURM (``sinfo``).
* overrides         one run can change single values: the ``--slurm_profile`` / ``--slurm_set STAGE.key=value`` options of
                    ``aps_runner.py``, the script_params keys ``slurm_profile``, ``slurm_site``, ``slurm_cpus_<STAGE>``,
                    ``slurm_mem_gb_<STAGE>``, ``slurm_time_<STAGE>``, or the environment variables ``PYAPS_SLURM_PROFILE``,
                    ``PYAPS_SLURM_SITE``, ``PYAPS_SLURM_RESOURCES_FILE``.
* escalation        a job that ended OUT_OF_MEMORY or TIMEOUT is retried once with 1.5 x its memory / time limit; the DMS records
                    that in ``slurm_resources_escalation.json`` next to the job scripts and this module applies it when the
                    scripts are generated again.

The effective values of every job are printed by the runner, written to ``slurm_resources_effective.json`` beside the scripts and
printed again in the header of the job's own log.  ``python -m PyAPS.aps_slurm_resources show|validate|explain`` is the operator tool.

Only the standard library and PyYAML are used.

Rules of one table row (``cases``): every case whose ``when`` matches is applied in the order written; first the stage's own
cases, then the ``common`` cases of the family.  A value is a number/string (set) or one operation: ``{set: v}``, ``{scale: f}``,
``{add: n}``, ``{half_plus_one: true}`` (cpus: ``int(cpus / 2) + 1``), ``{per_cpu: g}`` (mem_gb: g per final CPU).  Conditions
(``when``): ``res`` LR|HR, ``stack`` single|stack|super|other (or the level 0|1|3|4), ``grid`` (FERRE grid prefix m|n), ``family``;
a numeric fact of the observation may be tested with ``{ge: x, lt: y}``.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

__version__ = "1.0.0"

TABLE_NAME = "slurm_resources.yaml"
EFFECTIVE_NAME = "slurm_resources_effective.json"
ESCALATION_NAME = "slurm_resources_escalation.json"

#: families of jobs and their stages in the order of the runner's ``mp`` lists
FAMILIES = {
    "MOS": ("RR", "RVS", "FR", "PPXF", "EMI", "LS", "L2merge"),
    "LIFU": ("IFU_PR", "IFU_ExGal", "IFU_Gal"),
    "MIFU": ("IFU_PR", "IFU_ExGal", "IFU_Gal"),
    "CS": ("SQ", "FESWI", "SPACE", "AMY", "AN", "RRLGV", "RRLEW"),
}
#: level of a stage: the job name of SLURM is ``<stage>_<LEVEL>_<uapsid>``
LEVEL = {"MOS": "L2", "LIFU": "L2", "MIFU": "L2", "CS": "CS"}
STACK_NAMES = {0: "single", 1: "stack", 3: "super", 4: "other"}
STATUSES = ("legacy", "measured", "derived", "unmeasured")
LIMIT_POLICIES = ("off", "warn", "clamp")
CTX_KEYS = ("family", "res", "stack", "stack_level", "grid")
KEYS_OF_ROW = ("cpus", "mem_gb", "time", "status", "note", "cases")
KEYS_OF_CASE = ("when", "cpus", "mem_gb", "time", "note")
OPS = ("set", "scale", "add", "half_plus_one", "per_cpu")
#: escalation: factor of one retry after OUT_OF_MEMORY (memory) or TIMEOUT (time)
ESCALATION_FACTOR = 1.5
ESCALATION_MAX_ATTEMPTS = 1

_TIME_RE = re.compile(r"^(?:(\d+)-)?(\d+)(?::(\d{2}))?(?::(\d{2}))?$")
_STRICT_TIME_RE = re.compile(r"^(\d+|\d+:\d{2}|\d+:\d{2}:\d{2}|\d+-\d{1,2}|\d+-\d{1,2}:\d{2}|\d+-\d{1,2}:\d{2}:\d{2})$")
_SET_RE = re.compile(r"^([A-Za-z0-9_]+)\.(cpus|mem_gb|time)=(.*)$")


class ResourceTableError(ValueError):
    """The table (or an override) is not valid; the message names the place."""


# ---------------------------------------------------------------------------------------------------------- time helpers
def parse_time_seconds(value) -> int:
    """SLURM time (``minutes``, ``HH:MM:SS``, ``D-HH``, ``D-HH:MM``, ``D-HH:MM:SS``, ``M:SS``) in seconds."""
    text = str(value).strip()
    if not _STRICT_TIME_RE.match(text):
        raise ResourceTableError("%r is not a SLURM time (minutes, MM:SS, HH:MM:SS, D-HH, D-HH:MM or D-HH:MM:SS)" % (value,))
    if "-" in text:
        days, rest = text.split("-", 1)
        parts = [int(p) for p in rest.split(":")]
        parts += [0] * (3 - len(parts))
        return int(days) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]
    parts = [int(p) for p in text.split(":")]
    if len(parts) == 1:
        return parts[0] * 60                                    # plain number = minutes
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]                         # MM:SS
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def format_time(seconds: int) -> str:
    """Seconds as ``D-HH:MM:SS`` (the form SLURM prints and the table uses)."""
    seconds = int(round(seconds))
    days, rest = divmod(seconds, 86400)
    hours, rest = divmod(rest, 3600)
    minutes, secs = divmod(rest, 60)
    return "%d-%02d:%02d:%02d" % (days, hours, minutes, secs)


# --------------------------------------------------------------------------------------------------------------- loading
def default_table_path() -> Path:
    """``configs/slurm_resources.yaml`` of this checkout (``PYAPS_SLURM_RESOURCES_FILE`` wins)."""
    env = os.environ.get("PYAPS_SLURM_RESOURCES_FILE")
    if env:
        return Path(os.path.expandvars(os.path.expanduser(env)))
    here = Path(__file__).resolve()
    for base in (here.parents[2], here.parents[1], Path.cwd()):
        cand = base / "configs" / TABLE_NAME
        if cand.is_file():
            return cand
    return here.parents[2] / "configs" / TABLE_NAME


def _num(value, where, integer=False, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResourceTableError("%s: %r is not a number" % (where, value))
    if value <= minimum:
        raise ResourceTableError("%s: %r must be greater than %s" % (where, value, minimum))
    if integer and int(value) != value:
        raise ResourceTableError("%s: %r must be a whole number" % (where, value))
    return value


def _check_value(kind, value, where):
    """Validate the value (or operation) of ``cpus`` / ``mem_gb`` / ``time`` in a row or a case."""
    if isinstance(value, dict):
        if len(value) != 1 or next(iter(value)) not in OPS:
            raise ResourceTableError("%s: an operation is one of %s, got %r" % (where, ", ".join(OPS), value))
        op, arg = next(iter(value.items()))
        if kind == "time" and op not in ("set", "scale"):
            raise ResourceTableError("%s: time accepts set and scale only" % where)
        if kind == "cpus" and op == "per_cpu":
            raise ResourceTableError("%s: per_cpu is a memory operation" % where)
        if kind == "mem_gb" and op == "half_plus_one":
            raise ResourceTableError("%s: half_plus_one is a cpus operation" % where)
        if op == "half_plus_one":
            if arg is not True:
                raise ResourceTableError("%s: half_plus_one: true" % where)
        elif op == "set":
            _check_value(kind, arg, where)
        elif op == "add":
            if isinstance(arg, bool) or not isinstance(arg, (int, float)):
                raise ResourceTableError("%s: add needs a number" % where)
        else:                                                     # scale, per_cpu
            _num(arg, where + "." + op)
        return
    if kind == "time":
        if value is not None:
            parse_time_seconds(value)
    elif kind == "cpus":
        _num(value, where, integer=True)
    else:                                                         # mem_gb
        if value is not None:
            _num(value, where)


def _check_when(when, where):
    if not isinstance(when, dict) or not when:
        raise ResourceTableError("%s: when must be a non-empty mapping" % where)
    for key, val in when.items():
        if key not in CTX_KEYS:
            raise ResourceTableError("%s: unknown condition %r (known: %s)" % (where, key, ", ".join(CTX_KEYS)))
        if isinstance(val, dict):
            if not val or any(k not in ("ge", "gt", "le", "lt") for k in val):
                raise ResourceTableError("%s: %s: a range uses ge/gt/le/lt, got %r" % (where, key, val))
        elif isinstance(val, list):
            if not val:
                raise ResourceTableError("%s: %s: empty list" % (where, key))


def _check_cases(cases, where):
    if cases is None:
        return
    if not isinstance(cases, list):
        raise ResourceTableError("%s: cases must be a list" % where)
    for i, case in enumerate(cases):
        w = "%s.cases[%d]" % (where, i)
        if not isinstance(case, dict):
            raise ResourceTableError("%s: a case is a mapping" % w)
        for key in case:
            if key not in KEYS_OF_CASE:
                raise ResourceTableError("%s: unknown key %r (known: %s)" % (w, key, ", ".join(KEYS_OF_CASE)))
        if "when" not in case:
            raise ResourceTableError("%s: a case needs a 'when'" % w)
        _check_when(case["when"], w)
        if not any(k in case for k in ("cpus", "mem_gb", "time")):
            raise ResourceTableError("%s: a case changes cpus, mem_gb or time" % w)
        for kind in ("cpus", "mem_gb", "time"):
            if kind in case:
                _check_value(kind, case[kind], "%s.%s" % (w, kind))


def validate_table(table: dict) -> dict:
    """Check the whole table; returns it. Raises :class:`ResourceTableError` naming the first bad place."""
    if not isinstance(table, dict):
        raise ResourceTableError("the table is not a mapping")
    if table.get("schema") != 1:
        raise ResourceTableError("schema: 1 expected, got %r" % (table.get("schema"),))
    profiles = table.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ResourceTableError("profiles: at least one profile is needed")
    default = table.get("default_profile", "default")
    if default not in profiles:
        raise ResourceTableError("default_profile %r is not a profile" % (default,))
    for pname, prof in profiles.items():
        where = "profiles.%s" % pname
        if not isinstance(prof, dict):
            raise ResourceTableError("%s: not a mapping" % where)
        for key in prof:
            if key not in ("description", "extends", "limits", "status") and key not in FAMILIES:
                raise ResourceTableError("%s: unknown key %r" % (where, key))
        if prof.get("limits", "warn") not in LIMIT_POLICIES:
            raise ResourceTableError("%s.limits: one of %s" % (where, ", ".join(LIMIT_POLICIES)))
        parent = prof.get("extends")
        if parent is not None and parent not in profiles:
            raise ResourceTableError("%s.extends: %r is not a profile" % (where, parent))
        for fam, famdef in prof.items():
            if fam not in FAMILIES:
                continue
            fw = "%s.%s" % (where, fam)
            if not isinstance(famdef, dict):
                raise ResourceTableError("%s: not a mapping" % fw)
            for key in famdef:
                if key not in ("stages", "common"):
                    raise ResourceTableError("%s: unknown key %r (stages, common)" % (fw, key))
            _check_cases(famdef.get("common"), fw + ".common")
            for stage, row in (famdef.get("stages") or {}).items():
                sw = "%s.stages.%s" % (fw, stage)
                if stage not in FAMILIES[fam]:
                    raise ResourceTableError("%s: %r is not a stage of %s (%s)" % (sw, stage, fam, ", ".join(FAMILIES[fam])))
                if not isinstance(row, dict):
                    raise ResourceTableError("%s: not a mapping" % sw)
                for key in row:
                    if key not in KEYS_OF_ROW:
                        raise ResourceTableError("%s: unknown key %r (known: %s)" % (sw, key, ", ".join(KEYS_OF_ROW)))
                if "cpus" in row:
                    _check_value("cpus", row["cpus"], sw + ".cpus")
                if "mem_gb" in row:
                    _check_value("mem_gb", row["mem_gb"], sw + ".mem_gb")
                if "time" in row:
                    _check_value("time", row["time"], sw + ".time")
                if row.get("status", "legacy") not in STATUSES:
                    raise ResourceTableError("%s.status: one of %s" % (sw, ", ".join(STATUSES)))
                _check_cases(row.get("cases"), sw)
    # every profile must resolve to a complete set: each family has all its stages with a cpus value
    for pname in profiles:
        flat = flatten_profile(table, pname)
        for fam, stages in FAMILIES.items():
            for stage in stages:
                row = flat.get(fam, {}).get("stages", {}).get(stage)
                if row is None or "cpus" not in row:
                    raise ResourceTableError("profiles.%s: %s stage %s has no cpus (list it, or extend a profile that does)" % (pname, fam, stage))
    for sname, site in (table.get("sites") or {}).items():
        if not isinstance(site, dict):
            raise ResourceTableError("sites.%s: not a mapping" % sname)
        for key, val in site.items():
            if key == "max_mem_gb" or key == "max_cpus":
                _num(val, "sites.%s.%s" % (sname, key))
            elif key == "max_time":
                parse_time_seconds(val)
            elif key != "note":
                raise ResourceTableError("sites.%s: unknown key %r (max_mem_gb, max_cpus, max_time, note)" % (sname, key))
    return table


def flatten_profile(table: dict, name: str) -> dict:
    """The profile ``name`` with its ``extends`` chain merged: per family, per stage the whole row of the nearest profile that lists the
    stage; the ``common`` cases of the nearest profile that has them. Returns ``{family: {stages: {...}, common: [...]}}``."""
    chain, seen, cur = [], set(), name
    profiles = table["profiles"]
    while cur is not None:
        if cur in seen:
            raise ResourceTableError("profiles.%s: extends loops back to %s" % (name, cur))
        seen.add(cur)
        chain.append(cur)
        cur = profiles[cur].get("extends")
    flat: dict = {}
    for pname in reversed(chain):                        # base first, the extending profile last
        for fam in FAMILIES:
            fd = profiles[pname].get(fam)
            if not fd:
                continue
            slot = flat.setdefault(fam, {"stages": {}, "common": None})
            for stage, row in (fd.get("stages") or {}).items():
                slot["stages"][stage] = {"_profile": pname, **copy.deepcopy(row)}
            if "common" in fd:
                slot["common"] = copy.deepcopy(fd["common"])
    return flat


def load_table(path=None, validate=True) -> dict:
    """Read and validate the resource table (default: ``configs/slurm_resources.yaml`` of this checkout)."""
    import yaml
    p = Path(path) if path else default_table_path()
    if not p.is_file():
        raise ResourceTableError("resource table not found: %s" % p)
    with open(p) as fh:
        table = yaml.safe_load(fh)
    if validate:
        validate_table(table)
    table["_path"] = str(p)
    data = p.read_bytes()
    table["_sha256"] = hashlib.sha256(data).hexdigest()[:12]
    return table


# ------------------------------------------------------------------------------------------------------ cluster limits
def parse_sinfo_limits(text: str) -> dict:
    """Limits from ``sinfo -h -N -o '%c|%m|%l'`` lines (CPUs, memory in MB, partition time limit); the largest node counts."""
    cpus, mem_mb, tmax = 0, 0, None
    for line in (text or "").splitlines():
        parts = line.strip().split("|")
        if len(parts) < 2:
            continue
        try:
            cpus = max(cpus, int(parts[0].rstrip("+")))
            mem_mb = max(mem_mb, int(parts[1].rstrip("+")))
        except ValueError:
            continue
        if len(parts) > 2 and parts[2] and parts[2] != "infinite" and _STRICT_TIME_RE.match(parts[2]):
            sec = parse_time_seconds(parts[2])
            tmax = sec if tmax is None else max(tmax, sec)
    out = {}
    if cpus:
        out["max_cpus"] = cpus
    if mem_mb:
        out["max_mem_gb"] = math.floor(mem_mb / 1024.0)
    if tmax:
        out["max_time"] = format_time(tmax)
    return out


_LIMITS_CACHE: dict = {}


def detect_cluster_limits(timeout=10) -> dict:
    """Largest node and longest time limit of this SLURM cluster (empty when ``sinfo`` is not available)."""
    if "limits" in _LIMITS_CACHE:
        return dict(_LIMITS_CACHE["limits"])
    try:
        out = subprocess.run(["sinfo", "-h", "-N", "-o", "%c|%m|%l"], capture_output=True, text=True, timeout=timeout)
        limits = parse_sinfo_limits(out.stdout) if out.returncode == 0 else {}
    except (OSError, subprocess.SubprocessError):
        limits = {}
    _LIMITS_CACHE["limits"] = limits
    return dict(limits)


# ----------------------------------------------------------------------------------------------------------- escalation
def read_escalation(scripts_path) -> dict:
    """The escalation record of an OB's scripts folder: ``{stage: {attempts, mem_factor, time_factor, ...}}`` (empty when none)."""
    if not scripts_path:
        return {}
    p = Path(scripts_path) / ESCALATION_NAME
    try:
        data = json.loads(p.read_text())
    except (OSError, ValueError):
        return {}
    return data.get("stages", {}) if isinstance(data, dict) else {}


def can_escalate(scripts_path, stage) -> bool:
    return int(read_escalation(scripts_path).get(stage, {}).get("attempts", 0)) < ESCALATION_MAX_ATTEMPTS


def record_escalation(scripts_path, stage, reason, job_id=None, uapsid=None, previous=None, now=None) -> dict:
    """Record one escalation (OUT_OF_MEMORY -> memory x1.5, TIMEOUT -> time x1.5) of ``stage`` and return the stage's record.
    The file is written whole (temporary file + rename) so a reader never sees a half file."""
    reason = str(reason).upper()
    if reason not in ("OUT_OF_MEMORY", "TIMEOUT"):
        raise ResourceTableError("escalation reason must be OUT_OF_MEMORY or TIMEOUT, got %r" % reason)
    p = Path(scripts_path) / ESCALATION_NAME
    try:
        data = json.loads(p.read_text())
    except (OSError, ValueError):
        data = {}
    data.setdefault("schema", 1)
    if uapsid:
        data["uapsid"] = uapsid
    stages = data.setdefault("stages", {})
    rec = stages.get(stage) or {"attempts": 0, "mem_factor": 1.0, "time_factor": 1.0, "history": []}
    rec["attempts"] = int(rec.get("attempts", 0)) + 1
    if reason == "OUT_OF_MEMORY":
        rec["mem_factor"] = round(float(rec.get("mem_factor", 1.0)) * ESCALATION_FACTOR, 4)
    else:
        rec["time_factor"] = round(float(rec.get("time_factor", 1.0)) * ESCALATION_FACTOR, 4)
    rec["history"].append({"reason": reason, "job_id": job_id, "previous": previous or {},
                           "at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now or time.time()))})
    stages[stage] = rec
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, p)
    return rec


_JOBNAME_RE = re.compile(r"^(?P<stage>.+)_(?P<level>L2|CS)_(?P<uapsid>[0-9A-Za-z]+)$")
_OOM_TEXT = ("exceeded memory limit", "exceeded job memory limit", "oom-kill", "out of memory", "oom_kill", "cannot allocate memory",
             "memoryerror")


def _err_says_oom(logs_dir, job_name, job_id):
    """True when the job's own .err file says the memory limit killed it (SLURM's wording or Python's MemoryError)."""
    if not logs_dir:
        return False
    try:
        text = (Path(logs_dir) / ("%s.%s.err" % (job_name, job_id))).read_text(errors="replace")[-20000:].lower()
    except OSError:
        return False
    return any(sig in text for sig in _OOM_TEXT)


def resource_failures(job_info: dict, logs_dir=None) -> dict:
    """Stages whose latest job was killed for memory or time: ``{stage: {reason, job_id, job_name, state}}``.

    ``job_info`` is ``aps_slurm.get_slurm_status``'s ``{job_id: {Status, JobName, ...}}`` (the latest job of each name).
    OUT_OF_MEMORY and TIMEOUT are SLURM's own states. A job SLURM reports as FAILED or CANCELLED is counted as out of memory only when its
    .err file in ``logs_dir`` says the memory limit killed it (a cluster without cgroup enforcement kills through the accounting plugin
    and may report it that way)."""
    out = {}
    for job_id, info in (job_info or {}).items():
        m = _JOBNAME_RE.match(str(info.get("JobName", "")))
        if not m:
            continue
        state = str(info.get("Status", "")).split()[0].upper() if info.get("Status") else ""
        reason = None
        if state.startswith("OUT_OF_MEMORY"):
            reason = "OUT_OF_MEMORY"
        elif state.startswith("TIMEOUT"):
            reason = "TIMEOUT"
        elif state in ("FAILED", "CANCELLED") and _err_says_oom(logs_dir, info["JobName"], job_id):
            reason = "OUT_OF_MEMORY"
        if reason:
            out[m.group("stage")] = {"reason": reason, "job_id": str(job_id), "job_name": info["JobName"], "state": state}
    return out


def read_effective(scripts_path, level="L2") -> dict:
    """The effective-resources record the runner wrote beside the scripts (``{}`` when there is none)."""
    try:
        data = json.loads((Path(scripts_path) / EFFECTIVE_NAME).read_text())
    except (OSError, ValueError):
        return {}
    return data.get("CS" if level == "CS" else "L2", {})


# ----------------------------------------------------------------------------------------------------------- resolution
@dataclass
class Resources:
    """What one job asks of SLURM, and where each value came from."""
    stage: str
    family: str
    cpus: int
    mem_gb: int | None
    time: str | None
    status: str = "legacy"
    profile: str = "default"
    trace: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def mp(self) -> int:
        """Processes the stage starts (``--mp``): one per requested CPU (every stage pins its BLAS/OpenMP threads to 1)."""
        return self.cpus

    def mem_arg(self) -> str:
        return "--mem=%dGB" % self.mem_gb if self.mem_gb else ""

    def time_arg(self) -> str:
        return " --time=%s" % self.time if self.time else ""

    def as_dict(self) -> dict:
        return {"stage": self.stage, "family": self.family, "cpus": self.cpus, "mem_gb": self.mem_gb, "time": self.time,
                "mp": self.mp, "status": self.status, "profile": self.profile, "trace": list(self.trace),
                "warnings": list(self.warnings)}


def _match_one(cond, actual):
    if isinstance(cond, dict):
        if actual is None or isinstance(actual, str):
            return False
        for op, lim in cond.items():
            if (op == "ge" and not actual >= lim) or (op == "gt" and not actual > lim) \
                    or (op == "le" and not actual <= lim) or (op == "lt" and not actual < lim):
                return False
        return True
    if isinstance(cond, list):
        return any(_match_one(c, actual) for c in cond)
    if actual is None:
        return False
    return str(cond).strip().upper() == str(actual).strip().upper()


def _matches(when, ctx):
    for key, cond in when.items():
        actual = ctx.get(key)
        if key == "stack" and not isinstance(cond, (dict, list)) and str(cond).strip().isdigit():
            actual = ctx.get("stack_level")                      # stack: 0 / 1 / 3 / 4 by level
        if key == "stack" and isinstance(cond, list) and all(str(c).strip().isdigit() for c in cond):
            actual = ctx.get("stack_level")
        if not _match_one(cond, actual):
            return False
    return True


def _apply_op(kind, current, spec, cpus_now=None):
    """Value after applying ``spec`` (scalar = set, or one operation) to ``current``."""
    if isinstance(spec, dict):
        op, arg = next(iter(spec.items()))
    else:
        op, arg = "set", spec
    if op == "set":
        return arg
    if op == "scale":
        if kind == "time":
            return format_time(math.ceil(parse_time_seconds(current) * float(arg))) if current else current
        return None if current is None else (math.ceil(current * float(arg)) if kind == "cpus" else current * float(arg))
    if op == "add":
        return None if current is None else current + arg
    if op == "half_plus_one":
        return int(current / 2) + 1
    if op == "per_cpu":
        return float(arg) * cpus_now
    raise ResourceTableError("unknown operation %r" % op)


def _describe(case, kind):
    spec = case[kind]
    return "%s %s%s" % (kind, json.dumps(spec, sort_keys=True) if isinstance(spec, dict) else "= %s" % (spec,),
                        " (%s)" % case["note"] if case.get("note") else "")


def parse_set_option(text: str):
    """``STAGE.key=value`` of ``--slurm_set`` -> ``(stage, key, value)`` (cpus int, mem_gb float, time string; ``none`` clears mem/time)."""
    m = _SET_RE.match(str(text).strip())
    if not m:
        raise ResourceTableError("--slurm_set %r: use STAGE.cpus=N, STAGE.mem_gb=G or STAGE.time=D-HH:MM:SS" % (text,))
    stage, key, raw = m.groups()
    raw = raw.strip()
    if key == "cpus":
        if not raw.isdigit() or int(raw) < 1:
            raise ResourceTableError("--slurm_set %s.cpus: a whole number >= 1, got %r" % (stage, raw))
        return stage, key, int(raw)
    if raw.lower() in ("none", "off", ""):
        return stage, key, None
    if key == "mem_gb":
        try:
            val = float(raw)
        except ValueError:
            raise ResourceTableError("--slurm_set %s.mem_gb: a number of GB or none, got %r" % (stage, raw))
        if val <= 0:
            raise ResourceTableError("--slurm_set %s.mem_gb must be > 0" % stage)
        return stage, key, val
    parse_time_seconds(raw)
    return stage, key, raw


def collect_overrides(conf=None, cli_set=None, family=None):
    """All per-stage overrides: script_params keys (lowest) then ``--slurm_set`` (highest). ``{stage: {key: value}}``."""
    out: dict = {}
    stages = set(FAMILIES[family]) if family else {s for st in FAMILIES.values() for s in st}
    for key, raw in (conf or {}).items():
        m = re.match(r"^slurm_(cpus|mem_gb|time)_([A-Za-z0-9_]+)$", str(key))
        if not m or m.group(2) not in stages:
            continue
        kind, stage = m.groups()
        text = str(raw).strip()
        if kind == "time":
            if text.upper() in ("", "NONE"):
                out.setdefault(stage, {})["time"] = None
            elif _STRICT_TIME_RE.match(text):
                out.setdefault(stage, {})["time"] = text
            else:
                out.setdefault(stage, {}).setdefault("_bad_time", text)
        else:
            try:
                st, k, v = parse_set_option("%s.%s=%s" % (stage, kind, text))
            except ResourceTableError as exc:
                print("Warning: script_params %s: %s (ignored)" % (key, exc))
                continue
            out.setdefault(st, {})[k] = v
    for opt in cli_set or []:
        st, k, v = parse_set_option(opt)
        if st in stages:
            out.setdefault(st, {})[k] = v
    return out


def choose_profile(table, conf=None, cli_profile=None, env=None) -> str:
    env = os.environ if env is None else env
    if cli_profile is not None and str(cli_profile).strip().lower() in ("", "auto", "none"):
        cli_profile = None                                     # 'auto' (what the DMS passes) = the usual choice below
    name = cli_profile or env.get("PYAPS_SLURM_PROFILE") or (conf or {}).get("slurm_profile") or table.get("default_profile", "default")
    name = str(name).strip()
    if name not in table["profiles"]:
        raise ResourceTableError("resource profile %r is not in %s (profiles: %s)"
                                 % (name, table.get("_path", "the table"), ", ".join(table["profiles"])))
    return name


def resolve(family, ctx, table=None, conf=None, profile=None, mp_base=None, cli_set=None, escalation=None, site=None,
            limits=None, env=None, quiet=False):
    """Resolve the resources of every stage of ``family`` for the observation described by ``ctx``.

    ``ctx``       ``{res: 'LR'|'HR', stack_level: 0|1|3|4, grid: 'm'|'n'|..., ...}`` (family is added).
    ``mp_base``   the legacy ``--mp_*`` list: replaces the base CPU counts of the family (the cases still apply to it).
    ``cli_set``   ``['FR.mem_gb=120', ...]`` from ``--slurm_set``.
    ``escalation``  ``{stage: {mem_factor, time_factor}}`` from :func:`read_escalation`.
    ``limits``    dict (max_mem_gb, max_cpus, max_time) to use instead of the site/SLURM limits; ``{}`` = none.
    Returns ``{stage: Resources}`` in stage order.
    """
    if family not in FAMILIES:
        raise ResourceTableError("unknown family %r" % family)
    table = table or load_table()
    env = os.environ if env is None else env
    pname = choose_profile(table, conf, profile, env)
    prof = table["profiles"][pname]
    policy = prof.get("limits", "warn")
    flat = flatten_profile(table, pname)[family]
    stages = FAMILIES[family]
    ctx = dict(ctx)
    ctx["family"] = family
    if "stack_level" in ctx and "stack" not in ctx:
        ctx["stack"] = STACK_NAMES.get(int(ctx["stack_level"]), "other")
    if mp_base is not None and len(mp_base) != len(stages):
        raise ResourceTableError("--mp list for %s needs %d values (%s), got %d" % (family, len(stages), ",".join(stages), len(mp_base)))
    overrides = collect_overrides(conf, cli_set, family)

    # limits: explicit argument > the named site > what SLURM reports
    if limits is None:
        site_name = site or env.get("PYAPS_SLURM_SITE") or (conf or {}).get("slurm_site")
        if site_name:
            if site_name not in (table.get("sites") or {}):
                raise ResourceTableError("site %r is not in the table (sites: %s)" % (site_name, ", ".join(table.get("sites") or {})))
            limits = dict(table["sites"][site_name])
        elif policy != "off":
            limits = detect_cluster_limits()
        else:
            limits = {}
    escalation = escalation or {}
    out = {}
    for i, stage in enumerate(stages):
        row = flat["stages"][stage]
        trace = ["profile %s row (%s)" % (pname, row["_profile"])]
        cpus = int(mp_base[i]) if mp_base is not None else int(row["cpus"])
        if mp_base is not None:
            trace.append("cpus %d from the --mp list of the caller" % cpus)
        mem = row.get("mem_gb")
        tlim = row.get("time")
        for case in list(row.get("cases") or []) + list(flat["common"] or []):
            if not _matches(case["when"], ctx):
                continue
            if "cpus" in case:
                cpus = int(_apply_op("cpus", cpus, case["cpus"]))
                trace.append("case %s: %s" % (json.dumps(case["when"], sort_keys=True), _describe(case, "cpus")))
            if "time" in case:
                tlim = _apply_op("time", tlim, case["time"])
                trace.append("case %s: %s" % (json.dumps(case["when"], sort_keys=True), _describe(case, "time")))
        for case in list(row.get("cases") or []) + list(flat["common"] or []):      # memory after the final CPU count
            if "mem_gb" in case and _matches(case["when"], ctx):
                mem = _apply_op("mem_gb", mem, case["mem_gb"], cpus_now=cpus)
                trace.append("case %s: %s" % (json.dumps(case["when"], sort_keys=True), _describe(case, "mem_gb")))
        # operator overrides: the final value of this job (no HR doubling on top)
        ov = overrides.get(stage, {})
        if "cpus" in ov:
            cpus = int(ov["cpus"]); trace.append("override cpus = %d" % cpus)
        if "mem_gb" in ov:
            mem = ov["mem_gb"]; trace.append("override mem_gb = %s" % (mem,))
        warnings = []
        if "time" in ov:
            tlim = ov["time"]; trace.append("override time = %s" % (tlim,))
        elif "_bad_time" in ov:
            warnings.append("slurm_time_%s = %r is not a SLURM time (minutes, HH:MM:SS or D-HH:MM:SS); using %s"
                            % (stage, ov["_bad_time"], tlim or "no limit"))
        # escalation after an OUT_OF_MEMORY / TIMEOUT
        esc = escalation.get(stage)
        if esc:
            if mem and float(esc.get("mem_factor", 1.0)) > 1.0:
                before = mem
                mem = mem * float(esc["mem_factor"])
                trace.append("escalation: memory x%s after OUT_OF_MEMORY (%s -> %s GB)" % (esc["mem_factor"], before, math.ceil(mem)))
            elif float(esc.get("mem_factor", 1.0)) > 1.0 and not mem:
                base = cpus * float(table.get("assumed_default_mem_gb_per_cpu", 1.0))   # what SLURM gave it (DefMemPerCPU x cpus)
                mem = base * float(esc["mem_factor"])
                trace.append("escalation: memory x%s after OUT_OF_MEMORY (SLURM default ~%s GB -> %s GB)"
                             % (esc["mem_factor"], math.ceil(base), math.ceil(mem)))
            if tlim and float(esc.get("time_factor", 1.0)) > 1.0:
                before = tlim
                tlim = format_time(math.ceil(parse_time_seconds(tlim) * float(esc["time_factor"])))
                trace.append("escalation: time x%s after TIMEOUT (%s -> %s)" % (esc["time_factor"], before, tlim))
        # an escalated value must stay runnable whatever the policy: a request above the node / partition is rejected or waits for ever
        mem = None if mem is None else int(math.ceil(float(mem)))
        if esc and limits:
            if mem and limits.get("max_mem_gb") and mem > limits["max_mem_gb"]:
                trace.append("escalation: memory limited to the largest node (%d GB)" % limits["max_mem_gb"])
                mem = int(limits["max_mem_gb"])
            if tlim and limits.get("max_time") and parse_time_seconds(tlim) > parse_time_seconds(limits["max_time"]):
                trace.append("escalation: time limited to the partition maximum (%s)" % limits["max_time"])
                tlim = limits["max_time"]
        # cluster limits
        if limits and policy != "off":
            def _limit(what, value, cap, fmt="%s"):
                msg = "%s %s exceeds the largest node/partition (%s): " % (what, fmt % value, fmt % cap)
                if policy == "clamp":
                    warnings.append(msg + "clamped to %s" % (fmt % cap))
                    return cap
                warnings.append(msg + "no node can run it as asked (limits policy 'warn' keeps the request; use 'clamp' or lower it)")
                return value
            if mem and limits.get("max_mem_gb") and mem > limits["max_mem_gb"]:
                mem = _limit("memory", mem, int(limits["max_mem_gb"]), "%d GB")
            if limits.get("max_cpus") and cpus > limits["max_cpus"]:
                cpus = _limit("cpus", cpus, int(limits["max_cpus"]), "%d")
            if tlim and limits.get("max_time") and parse_time_seconds(tlim) > parse_time_seconds(limits["max_time"]):
                tlim = _limit("time", tlim, limits["max_time"], "%s")
        if tlim is not None:
            tlim = str(tlim)
        out[stage] = Resources(stage=stage, family=family, cpus=int(cpus), mem_gb=mem, time=tlim,
                               status=row.get("status", "legacy"), profile=pname, trace=trace, warnings=warnings)
        if not quiet:
            for w in warnings:
                print("Warning: SLURM resources %s: %s" % (stage, w))
    return out


# --------------------------------------------------------------------------------------------------------- reporting
def effective_summary(resolved: dict, ctx: dict, table: dict, family: str, uapsid=None) -> dict:
    stages = list(resolved.values())
    return {
        "schema": 1, "module_version": __version__, "uapsid": uapsid, "family": family,
        "profile": stages[0].profile if stages else None, "table": table.get("_path"), "table_sha256": table.get("_sha256"),
        "context": {k: ctx[k] for k in sorted(ctx)}, "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stages": {r.stage: r.as_dict() for r in stages},
    }


def write_effective(path, summary: dict) -> None:
    """Write the effective values beside the job scripts (never raises: a read-only folder must not stop the run)."""
    try:
        p = Path(path) / EFFECTIVE_NAME
        # one file per OB and level: the CS stages keep their own entry under the same folder name
        data = {}
        if p.is_file():
            try:
                data = json.loads(p.read_text())
            except ValueError:
                data = {}
        data[summary["family"] if summary["family"] == "CS" else "L2"] = summary
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
        os.replace(tmp, p)
    except OSError as exc:
        print("Warning: could not write %s: %s" % (EFFECTIVE_NAME, exc))


def format_table(resolved: dict) -> str:
    rows = ["  %-10s %5s %8s  %-11s %-10s %s" % ("stage", "cpus", "mem", "time", "status", "profile")]
    for r in resolved.values():
        rows.append("  %-10s %5d %8s  %-11s %-10s %s" % (r.stage, r.cpus, ("%d GB" % r.mem_gb) if r.mem_gb else "default",
                                                        r.time or "-", r.status, r.profile))
    return "\n".join(rows)


def job_header_lines(res: Resources) -> list:
    """Shell lines for the top of a stage script: print what the job asked for and what SLURM gave it."""
    mem = ("%d GB" % res.mem_gb) if res.mem_gb else "SLURM default (DefMemPerCPU)"
    q = lambda s: str(s).replace("'", "'\\''")
    return [
        "echo 'PyAPS SLURM resources: stage=%s profile=%s status=%s cpus=%d mem=%s time=%s mp=%d threads=1'"
        % (q(res.stage), q(res.profile), q(res.status), res.cpus, q(mem), q(res.time or "no limit"), res.mp),
        'echo "PyAPS SLURM granted:   job=${SLURM_JOB_ID:-n/a} ntasks=${SLURM_NTASKS:-n/a} cpus_on_node=${SLURM_CPUS_ON_NODE:-n/a} '
        'mem_per_node_MB=${SLURM_MEM_PER_NODE:-n/a} node=${SLURMD_NODENAME:-n/a}"',
        'if [ -n "${SLURM_NTASKS:-}" ] && [ "${SLURM_NTASKS}" != "%d" ]; then echo "WARNING: requested %d CPUs but SLURM_NTASKS=${SLURM_NTASKS}"; fi'
        % (res.cpus, res.cpus),
    ]


# ------------------------------------------------------------------------------------------------------------------ CLI
def _ctx_from_args(args) -> dict:
    ctx = {"res": args.res.upper(), "grid": args.grid}
    level = {"single": 0, "stack": 1, "super": 3, "other": 4}[args.stack]
    ctx["stack_level"] = level
    return ctx


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m PyAPS.aps_slurm_resources", description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, hlp in (("show", "print the resources of every stage for one observation class"),
                      ("explain", "like show, with the rule trace of every value"),
                      ("validate", "check the table")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("--file", help="resource table (default: configs/slurm_resources.yaml)")
        if name != "validate":
            p.add_argument("--profile", help="profile name (default: PYAPS_SLURM_PROFILE or the table's default)")
            p.add_argument("--family", choices=sorted(FAMILIES), default="MOS")
            p.add_argument("--res", choices=["LR", "HR", "lr", "hr"], default="LR")
            p.add_argument("--stack", choices=["single", "stack", "super", "other"], default="stack")
            p.add_argument("--grid", default="m", help="FERRE grid prefix of the run (m fine, n coarse)")
            p.add_argument("--site", help="site of the table whose limits apply (default: read from SLURM)")
            p.add_argument("--set", action="append", default=[], metavar="STAGE.key=value")
            p.add_argument("--no-limits", action="store_true", help="ignore node limits")
    args = ap.parse_args(argv)
    try:
        table = load_table(args.file)
        if args.cmd == "validate":
            n = len(table["profiles"])
            print("OK: %s (%d profiles: %s)" % (table["_path"], n, ", ".join(table["profiles"])))
            return 0
        res = resolve(args.family, _ctx_from_args(args), table=table, profile=args.profile, cli_set=args.set, site=args.site,
                      limits={} if args.no_limits else None)
    except ResourceTableError as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 2
    print("%s  profile=%s  %s %s grid=%s  (table %s %s)" % (args.family, next(iter(res.values())).profile, args.res.upper(),
                                                         args.stack, args.grid, table["_path"], table["_sha256"]))
    print(format_table(res))
    if args.cmd == "explain":
        for r in res.values():
            print("\n%s:" % r.stage)
            for t in r.trace + ["WARNING: " + w for w in r.warnings]:
                print("   - " + t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
