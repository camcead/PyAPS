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

MACHINE AND SITE AWARENESS (1.1.0): before a request is checked, ``aps_resource_probe`` looks at the place the jobs will run (a SLURM cluster:
nodes, RealMemory, partition limits; or this machine: CPUs honouring affinity and cgroup limits, RAM, swap), the place is matched to a ``site``
of the table (or of ``slurm_sites.yaml``), and each stage request is compared with what the place can give.  The policy ``limits`` is off |
warn (default: report, never change) | clamp (lower only what does not fit) | error (refuse what can never run, with the working alternative).
A request that fits is never changed.  Depth-first priority (``priority:``) orders the jobs of a night so that the first pair completes first.
``python -m PyAPS.aps_slurm_resources site|probe|check`` are the operator views.

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
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

try:                                                     # the package, or the file run as a script
    from PyAPS import aps_resource_probe as rprobe
except ImportError:                                      # pragma: no cover
    import aps_resource_probe as rprobe

__version__ = "1.1.0"

TABLE_NAME = "slurm_resources.yaml"
SITES_NAME = "slurm_sites.yaml"
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
LIMIT_POLICIES = ("off", "warn", "clamp", "error")
PRIORITY_POLICIES = ("off", "depth_first")
PRIORITY_CLASSES = ("offline", "scheduled")
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


def norm_word(value):
    """A policy word as written: YAML reads an unquoted ``off`` as False; text is lower-cased and stripped (None stays None)."""
    if value is False:
        return "off"
    return None if value is None else str(value).strip().lower()


class ResourceTableError(ValueError):
    """The table (or an override) is not valid; the message names the place."""


class ResourceLimitError(ResourceTableError):
    """Policy ``error``: a request can never run on this machine or cluster (the message says what would fit)."""


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


SITE_KEYS = ("max_mem_gb", "max_cpus", "max_time", "note", "description", "match", "profile", "profile_openmp", "ferre_openmp", "limits",
             "mem_fraction", "cpu_fraction", "expect", "scheduler")
MATCH_KEYS = ("scheduler", "kind", "hostname", "cluster", "any_node", "node_names")
EXPECT_KEYS = ("nodes", "node_cpus", "node_mem_gb", "cpus", "mem_gb")


def _check_site(sname, site, profiles):
    where = "sites.%s" % sname
    if not isinstance(site, dict):
        raise ResourceTableError("%s: not a mapping" % where)
    for key, val in site.items():
        if key not in SITE_KEYS:
            raise ResourceTableError("%s: unknown key %r (known: %s)" % (where, key, ", ".join(SITE_KEYS)))
        if key in ("max_mem_gb", "max_cpus"):
            _num(val, "%s.%s" % (where, key))
        elif key == "max_time":
            parse_time_seconds(val)
        elif key in ("mem_fraction", "cpu_fraction"):
            _num(val, "%s.%s" % (where, key))
            if val > 1:
                raise ResourceTableError("%s.%s: a share between 0 and 1, got %r" % (where, key, val))
        elif key in ("profile", "profile_openmp"):
            if val not in profiles:
                raise ResourceTableError("%s.%s: %r is not a profile (%s)" % (where, key, val, ", ".join(profiles)))
        elif key == "limits":
            if norm_word(val) not in LIMIT_POLICIES:
                raise ResourceTableError("%s.limits: one of %s" % (where, ", ".join(LIMIT_POLICIES)))
        elif key == "ferre_openmp":
            if not isinstance(val, bool):
                raise ResourceTableError("%s.ferre_openmp: true or false" % where)
        elif key == "scheduler":
            if val not in ("slurm", "none"):
                raise ResourceTableError("%s.scheduler: slurm or none" % where)
        elif key == "match":
            if not isinstance(val, dict) or not val:
                raise ResourceTableError("%s.match: a non-empty mapping" % where)
            for mk, mv in val.items():
                if mk not in MATCH_KEYS:
                    raise ResourceTableError("%s.match: unknown key %r (known: %s)" % (where, mk, ", ".join(MATCH_KEYS)))
                if mk in ("hostname", "cluster", "node_names"):
                    try:
                        re.compile(str(mv))
                    except re.error as exc:
                        raise ResourceTableError("%s.match.%s: bad regular expression (%s)" % (where, mk, exc))
                if mk == "any_node" and (not isinstance(mv, dict) or not set(mv) <= {"cpus", "real_memory_mb"}):
                    raise ResourceTableError("%s.match.any_node: {cpus:, real_memory_mb:}" % where)
        elif key == "expect":
            if not isinstance(val, dict) or any(k not in EXPECT_KEYS for k in val):
                raise ResourceTableError("%s.expect: keys among %s" % (where, ", ".join(EXPECT_KEYS)))


def _check_priority(block, where, top=False):
    if not isinstance(block, dict):
        raise ResourceTableError("%s: not a mapping" % where)
    known = ("policy", "default_class", "classes", "stage_step", "pair_step", "pair_cap", "stage_remaining", "note")
    for key, val in block.items():
        if key not in known:
            raise ResourceTableError("%s: unknown key %r (known: %s)" % (where, key, ", ".join(known)))
        if key == "policy" and norm_word(val) not in PRIORITY_POLICIES:
            raise ResourceTableError("%s.policy: one of %s" % (where, ", ".join(PRIORITY_POLICIES)))
        if key == "default_class" and val not in PRIORITY_CLASSES:
            raise ResourceTableError("%s.default_class: one of %s" % (where, ", ".join(PRIORITY_CLASSES)))
        if key == "classes":
            if not isinstance(val, dict) or any(c not in PRIORITY_CLASSES for c in val):
                raise ResourceTableError("%s.classes: keys among %s" % (where, ", ".join(PRIORITY_CLASSES)))
            for c, n in val.items():
                _num(n, "%s.classes.%s" % (where, c), minimum=-1)
        if key in ("stage_step", "pair_step"):
            _num(val, "%s.%s" % (where, key), minimum=-1)
        if key == "pair_cap":
            _num(val, "%s.pair_cap" % where, integer=True, minimum=-1)
        if key == "stage_remaining":
            if not isinstance(val, dict):
                raise ResourceTableError("%s.stage_remaining: a mapping stage -> number of stages still to come" % where)
            known_stages = {st for fam in FAMILIES.values() for st in fam}
            for st, n in val.items():
                if st not in known_stages:
                    raise ResourceTableError("%s.stage_remaining: %r is not a stage" % (where, st))
                _num(n, "%s.stage_remaining.%s" % (where, st), integer=True, minimum=-1)


def _check_footprint(fp):
    if fp is None:
        return
    if not isinstance(fp, dict):
        raise ResourceTableError("footprint: not a mapping")
    known_stages = {st for fam in FAMILIES.values() for st in fam}
    for st, row in fp.items():
        where = "footprint.%s" % st
        if st not in known_stages:
            raise ResourceTableError("%s: not a stage" % where)
        if not isinstance(row, dict):
            raise ResourceTableError("%s: not a mapping" % where)
        for key, val in row.items():
            if key in ("base_gb", "per_cpu_gb"):
                _num(val, "%s.%s" % (where, key), minimum=-1)
            elif key == "cases":
                for i, case in enumerate(val or []):
                    w = "%s.cases[%d]" % (where, i)
                    if "when" not in case:
                        raise ResourceTableError("%s: a case needs a 'when'" % w)
                    _check_when(case["when"], w)
                    for k2, v2 in case.items():
                        if k2 in ("base_gb", "per_cpu_gb"):
                            _num(v2, "%s.%s" % (w, k2), minimum=-1)
            elif key != "note":
                raise ResourceTableError("%s: unknown key %r (base_gb, per_cpu_gb, cases, note)" % (where, key))


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
            if key not in ("description", "extends", "limits", "status", "priority") and key not in FAMILIES:
                raise ResourceTableError("%s: unknown key %r" % (where, key))
        if norm_word(prof.get("limits", "warn")) not in LIMIT_POLICIES:
            raise ResourceTableError("%s.limits: one of %s" % (where, ", ".join(LIMIT_POLICIES)))
        if "priority" in prof:
            _check_priority(prof["priority"], where + ".priority")
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
        _check_site(sname, site, profiles)
    if "priority" in table:
        _check_priority(table["priority"], "priority", top=True)
    _check_footprint(table.get("footprint"))
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
def _read_json(path):
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def read_escalation(scripts_path, tag) -> dict:
    """The escalation record of one run (``tag``: the uapsid, or the head name of a manual run) in an OB's scripts folder:
    ``{stage: {attempts, mem_factor, time_factor, history}}`` (empty when none). An OB folder holds the scripts of every pair of
    the OB, so the record is kept per tag."""
    if not scripts_path or not tag:
        return {}
    return _read_json(Path(scripts_path) / ESCALATION_NAME).get("runs", {}).get(str(tag), {})


def can_escalate(scripts_path, tag, stage) -> bool:
    return int(read_escalation(scripts_path, tag).get(stage, {}).get("attempts", 0)) < ESCALATION_MAX_ATTEMPTS


def record_escalation(scripts_path, tag, stage, reason, job_id=None, previous=None, now=None) -> dict:
    """Record one escalation (OUT_OF_MEMORY -> memory x1.5, TIMEOUT -> time x1.5) of ``stage`` of the run ``tag`` and return the stage's
    record. The file is written whole (temporary file + rename) so a reader never sees a half file."""
    reason = str(reason).upper()
    if reason not in ("OUT_OF_MEMORY", "TIMEOUT"):
        raise ResourceTableError("escalation reason must be OUT_OF_MEMORY or TIMEOUT, got %r" % reason)
    p = Path(scripts_path) / ESCALATION_NAME
    data = _read_json(p)
    data.setdefault("schema", 2)
    stages = data.setdefault("runs", {}).setdefault(str(tag), {})
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


def effective_tag(uapsid, family) -> str:
    """Key of a run in ``slurm_resources_effective.json``: the uapsid, plus ``:CS`` for the CS chain of the same pair."""
    return "%s:CS" % uapsid if family == "CS" else str(uapsid)


def read_effective(scripts_path, tag=None) -> dict:
    """The effective-resources record the runner wrote beside the scripts, for the run ``tag`` (see :func:`effective_tag`; ``{}`` when
    there is none). The file holds the runs of every pair of the OB: ``{"runs": {tag: summary}}``."""
    return _read_json(Path(scripts_path) / EFFECTIVE_NAME).get("runs", {}).get(str(tag), {}) if tag else {}


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
    requested: dict = field(default_factory=dict)       # cpus / mem_gb / time before the check against the machine
    adjustments: list = field(default_factory=list)     # what did not fit: {what, requested, limit, fit, reason, action: warn|clamp|error, use}
    fits: bool | None = None                            # False when something does not fit and was not (or could not be) lowered
    footprint_gb: float | None = None                   # estimated true memory of the stage at the final CPU count (table `footprint`)
    policy: str | None = None                           # limits policy used for the check
    nice: int | None = None                             # sbatch --nice of the depth-first scheme (None = not set)
    comment: str | None = None                          # sbatch --comment: pyaps:<uapsid>:p<pair>:<class>:<STAGE>
    environment: object = field(default=None, repr=False, compare=False)   # the Environment these values were checked against

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
                "warnings": list(self.warnings), "requested": dict(self.requested), "adjustments": [dict(a) for a in self.adjustments],
                "fits": self.fits, "footprint_gb": self.footprint_gb, "limits_policy": self.policy, "nice": self.nice, "comment": self.comment}


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


def choose_profile_ex(table, conf=None, cli_profile=None, env=None, site_profile=None):
    """``(profile name, where it came from)``: ``--slurm_profile`` > ``PYAPS_SLURM_PROFILE`` > script_params ``slurm_profile`` > the table's
    ``default_profile``. The word ``site`` (in any of the three) means the profile of the detected site (``site_profile``)."""
    env = os.environ if env is None else env
    if cli_profile is not None and str(cli_profile).strip().lower() in ("", "auto", "none"):
        cli_profile = None                                     # 'auto' (what the DMS passes) = the usual choice below
    for val, src in ((cli_profile, "--slurm_profile"), (env.get("PYAPS_SLURM_PROFILE"), "PYAPS_SLURM_PROFILE"),
                     ((conf or {}).get("slurm_profile"), "script_params slurm_profile")):
        if val:
            name = str(val).strip()
            break
    else:
        name, src = table.get("default_profile", "default"), "table default_profile"
    if name.lower() == "site":
        if site_profile:
            name, src = site_profile, src + " = site"
        else:
            name, src = table.get("default_profile", "default"), src + " = site, but no site profile is known: table default_profile"
    if name not in table["profiles"]:
        raise ResourceTableError("resource profile %r is not in %s (profiles: %s)"
                                 % (name, table.get("_path", "the table"), ", ".join(table["profiles"])))
    return name, src


def choose_profile(table, conf=None, cli_profile=None, env=None, site_profile=None) -> str:
    return choose_profile_ex(table, conf, cli_profile, env, site_profile)[0]


# ------------------------------------------------------------------------------------------------- sites and the environment
def sites_file_path(conf=None, env=None):
    """The installation's own site list: ``PYAPS_SLURM_SITES_FILE`` > script_params ``slurm_sites_file`` > ``slurm_sites.yaml`` in PyAPS_CONFIG."""
    env = os.environ if env is None else env
    conf = conf or {}
    for val in (env.get("PYAPS_SLURM_SITES_FILE"), conf.get("slurm_sites_file")):
        if val and str(val).strip().upper() not in ("", "NONE"):
            return Path(os.path.expandvars(os.path.expanduser(str(val))))
    cfg_dir = conf.get("PyAPS_CONFIG")
    if cfg_dir:
        cand = Path(os.path.expandvars(os.path.expanduser(str(cfg_dir)))) / SITES_NAME
        if cand.is_file():
            return cand
    return None


def all_sites(table, conf=None, env=None):
    """``(sites, messages)``: the sites of the table, then those of the installation's ``slurm_sites.yaml`` (same name = the file wins)."""
    sites = {k: dict(v) for k, v in (table.get("sites") or {}).items()}
    messages = []
    path = sites_file_path(conf, env)
    if path is not None:
        try:
            import yaml
            data = yaml.safe_load(path.read_text()) or {}
            extra = data.get("sites") if isinstance(data, dict) else None
            if not isinstance(extra, dict):
                raise ResourceTableError("no 'sites:' mapping")
            for name, entry in extra.items():
                _check_site(name, entry, table["profiles"])
                sites[name] = dict(entry, _origin=str(path))
        except (OSError, ResourceTableError, ValueError, ImportError) as exc:
            messages.append("sites file %s ignored: %s" % (path, exc))
    return sites, messages


def _operator(arg, env_name, conf_key, env, conf):
    """First of: the argument, the environment variable, the script_params key -> ``(value, source)`` or ``(None, None)``."""
    for val, src in ((arg, "option"), (env.get(env_name) if env_name else None, env_name), ((conf or {}).get(conf_key) if conf_key else None, "script_params " + str(conf_key))):
        if val is not None and str(val).strip() != "" and str(val).strip().upper() != "NONE":
            return val, src
    return None, None


def _match_site(entry, facts):
    """Number of keys of ``entry['match']`` that hold, or -1 when one does not (a site without ``match`` is never detected)."""
    match = entry.get("match")
    if not match:
        return -1
    for key, want in match.items():
        if key == "scheduler" and facts["scheduler"] != want:
            return -1
        if key == "kind" and facts["kind"] != want:
            return -1
        if key == "hostname" and not (re.fullmatch(str(want), facts["hostname"].split(".")[0]) or re.fullmatch(str(want), facts["hostname"])):
            return -1
        if key == "cluster" and not (facts["cluster"] and re.fullmatch(str(want), facts["cluster"])):
            return -1
        if key == "node_names" and not any(re.fullmatch(str(want), n) for n in facts["node_names"]):
            return -1
        if key == "any_node" and not any((("cpus" not in want or n[0] == want["cpus"]) and ("real_memory_mb" not in want or n[1] == want["real_memory_mb"]))
                                         for n in facts["nodes"]):
            return -1
    return len(match)


@dataclass
class Environment:
    """The place the jobs will run and the rules that were chosen for checking a request against it."""
    target: str = "slurm"                  # slurm | local
    machine: dict | None = None            # the probe (aps_resource_probe.probe), None when it was not made
    site: dict | None = None               # {name, source, description}
    limits: dict = field(default_factory=dict)      # what one job can ask for (max_cpus, max_mem_gb, max_time, nodes_fit, source)
    budget: dict = field(default_factory=dict)      # local target: {cpus, mem_gb, usable_cpus, available_gb, cpu_fraction, mem_fraction}
    parallel_jobs: int = 1
    ferre_openmp: bool | None = None
    site_entry: dict = field(default_factory=dict)
    messages: list = field(default_factory=list)
    policy: str = "warn"
    policy_source: str = "default"

    def summary(self) -> dict:
        m = self.machine or {}
        loc, sl = m.get("local", {}), m.get("slurm", {})
        out = {"target": self.target, "site": self.site, "policy": self.policy, "policy_source": self.policy_source,
               "parallel_jobs": self.parallel_jobs, "limits": {k: v for k, v in self.limits.items() if k != "nodes_fit"} if self.limits else {},
               "budget": dict(self.budget), "ferre_openmp": self.ferre_openmp, "messages": list(self.messages)}
        if self.limits.get("nodes_fit"):
            out["limits"]["nodes_fit"] = [list(x) for x in self.limits["nodes_fit"]]
        if m:
            out["machine"] = {"kind": m.get("kind"), "os": m.get("os", {}).get("system"), "hostname": m.get("os", {}).get("hostname"),
                              "container": m.get("os", {}).get("container"), "cpus_usable": loc.get("cpus", {}).get("usable"),
                              "ram_total_gb": loc.get("memory", {}).get("total_gb"), "ram_available_gb": loc.get("memory", {}).get("usable_available_gb"),
                              "swap_total_gb": loc.get("memory", {}).get("swap_total_gb"), "probe_seconds": m.get("seconds"), "problems": list(m.get("problems", []))}
            if sl.get("present"):
                out["machine"]["slurm"] = {"version": sl.get("version"), "cluster": sl.get("cluster"), "source": sl.get("source"),
                                           "nodes": len(sl.get("nodes", [])), "nodes_up": sl.get("nodes_up"), "memory_enforced": sl.get("memory_enforced"),
                                           "totals": sl.get("totals"), "plugins": sl.get("plugins"), "usable": sl.get("usable")}
        return out


def detect_environment(table, conf=None, env=None, site=None, machine=None, target=None, hpc=None, ferre_exe=None, parallel_jobs=None,
                       probe=True, quiet=True) -> Environment:
    """Look at the place the jobs will run and name it. Never raises on a probe failure (the answer is then 'unknown': no limits).

    ``target``  'slurm' | 'local' | None: None follows ``hpc`` (3 = SLURM, 1 = bash on this machine) and otherwise the machine (SLURM when present).
    ``machine`` a probe result to use instead of probing (tests, ``--machine-file``). ``probe=False``: do not look at all.
    """
    env = os.environ if env is None else env
    conf = conf or {}
    e = Environment()
    sites, msgs = all_sites(table, conf, env)
    e.messages.extend(msgs)
    # --- the machine
    if machine is None and probe:
        if ferre_exe is None:
            fe = conf.get("FERRE_EXE")
            ferre_exe = os.path.expandvars(os.path.expanduser(str(fe))) if fe and str(fe).upper() != "NONE" else None
        try:
            machine = rprobe.probe(ferre_exe=ferre_exe)
        except Exception as exc:                                # a probe must never stop a run
            e.messages.append("the machine probe failed (%s): limits unknown" % exc)
            machine = None
    e.machine = machine
    sl = (machine or {}).get("slurm", {}) or {}
    if target is None:
        h = None if hpc is None else str(hpc).strip()
        target = "local" if h == "1" else ("slurm" if h == "3" else ("slurm" if sl.get("present") else "local"))
    e.target = target
    # --- parallel jobs
    pj, _src = _operator(parallel_jobs, "PYAPS_PARALLEL_JOBS", "parallel_jobs", env, conf)
    try:
        e.parallel_jobs = max(1, int(pj)) if pj is not None else 1
    except (TypeError, ValueError):
        raise ResourceTableError("parallel_jobs %r is not a whole number" % (pj,))
    # --- the site: named, or detected
    name, how = _operator(site, "PYAPS_SLURM_SITE", "slurm_site", env, conf)
    if name is not None and str(name).strip().lower() in ("auto", "detect"):
        name = None
    if name is not None:
        name = str(name).strip()
        if name not in sites:
            raise ResourceTableError("site %r is not in the table or the sites file (sites: %s)" % (name, ", ".join(sites)))
        e.site = {"name": name, "source": "named by " + how}
    elif machine is not None:
        loc = machine.get("local", {})
        facts = {"scheduler": "slurm" if target == "slurm" else "none",
                 "kind": machine.get("kind") if target == "slurm" else rprobe.classify(machine.get("os", {}), loc, None),
                 "hostname": machine.get("os", {}).get("hostname") or "", "cluster": sl.get("cluster") if target == "slurm" else None,
                 "node_names": [n["name"] for n in sl.get("nodes", [])] if target == "slurm" else [],
                 "nodes": [(n["cpus"], n["real_mem_mb"]) for n in sl.get("nodes", [])] if target == "slurm" else []}
        best, best_score = None, -1
        for sname, entry in sites.items():
            sc = _match_site(entry, facts)
            if sc > best_score:
                best, best_score = sname, sc
        if best is not None:
            e.site = {"name": best, "source": "detected (%d match keys: %s)" % (best_score, json.dumps(sites[best]["match"], sort_keys=True))}
    if e.site:
        e.site_entry = dict(sites[e.site["name"]])
        e.site["description"] = e.site_entry.get("description") or e.site_entry.get("note")
        if e.site_entry.get("_origin"):
            e.site["file"] = e.site_entry["_origin"]
    # --- ferre build: the file when it can be read, else what the site says
    feat = (machine or {}).get("features", {})
    e.ferre_openmp = feat.get("ferre_openmp")
    if e.ferre_openmp is None and "ferre_openmp" in e.site_entry:
        e.ferre_openmp = bool(e.site_entry["ferre_openmp"])
    # --- the limits of one job
    pins = {k: e.site_entry[k] for k in ("max_mem_gb", "max_cpus", "max_time") if k in e.site_entry}
    explicit = bool(e.site and e.site["source"].startswith("named"))
    if pins and explicit:
        e.limits = dict(pins, source="pinned by site %s" % e.site["name"])
    elif target == "slurm":
        if sl.get("present") and sl.get("usable") and sl.get("limits"):
            e.limits = dict(sl["limits"], source="SLURM (%s)" % sl.get("source"))
        elif machine is not None:
            e.messages.append("SLURM limits are unknown (%s): requests are not checked against the cluster" %
                              ("; ".join(sl.get("messages", [])) or sl.get("reason") or "no answer"))
    else:
        loc = (machine or {}).get("local", {})
        cpus_u, mem_a = loc.get("cpus", {}).get("usable"), loc.get("memory", {}).get("usable_available_gb")
        cf, _ = _operator(None, "PYAPS_CPU_FRACTION", "slurm_cpu_fraction", env, conf)
        mf, _ = _operator(None, "PYAPS_MEM_FRACTION", "slurm_mem_fraction", env, conf)
        try:
            cf = float(cf) if cf is not None else float(e.site_entry.get("cpu_fraction", 1.0))
            mf = float(mf) if mf is not None else float(e.site_entry.get("mem_fraction", 0.8))
        except ValueError:
            raise ResourceTableError("cpu/mem fraction must be a number between 0 and 1")
        if not (0 < cf <= 1 and 0 < mf <= 1):
            raise ResourceTableError("cpu/mem fraction must be between 0 and 1 (got %s, %s)" % (cf, mf))
        b = {"cpu_fraction": cf, "mem_fraction": mf, "usable_cpus": cpus_u, "available_gb": mem_a,
             "total_gb": loc.get("memory", {}).get("usable_total_gb")}
        if cpus_u:
            b["cpus"] = max(1, int(math.floor(cpus_u * cf / e.parallel_jobs + 1e-9)))
        if mem_a:
            b["mem_gb"] = round(mem_a * mf / e.parallel_jobs, 2)
        e.budget = b
        if b:
            e.limits = {k: b[k] for k in ("cpus", "mem_gb") if k in b}
            if "cpus" in b:
                e.limits["max_cpus"] = b["cpus"]
            if "mem_gb" in b:
                e.limits["max_mem_gb"] = b["mem_gb"]
            e.limits["source"] = "this machine"
        if machine is not None and not b.get("cpus") and not b.get("mem_gb"):
            e.messages.append("this machine's CPUs and RAM are unknown: requests are not checked")
    # --- what the site expected
    ex = e.site_entry.get("expect") or {}
    if ex and machine is not None:
        if target == "slurm" and sl.get("usable"):
            nodes = sl.get("nodes", [])
            got = {"nodes": len(nodes), "node_cpus": max((n["cpus"] for n in nodes), default=None),
                   "node_mem_gb": int(max((n["real_mem_mb"] for n in nodes), default=0) // 1024)}
            for k in ("nodes", "node_cpus", "node_mem_gb"):
                if k in ex and got[k] != ex[k]:
                    e.messages.append("site %s expects %s=%s but the cluster reports %s: a node is down or the site changed" % (e.site["name"], k, ex[k], got[k]))
        elif target == "local":
            for k, key in (("cpus", "cpus"), ("mem_gb", "available_gb")):
                have = e.budget.get("usable_cpus") if k == "cpus" else e.budget.get("total_gb")
                if k in ex and have is not None and have + 0.5 < ex[k]:
                    e.messages.append("site %s expects %s=%s but this machine has %s" % (e.site["name"], k, ex[k], have))
    if machine is not None:
        e.messages.extend(machine.get("problems", []))
    return e


# ----------------------------------------------------------------------------------------------------------- footprint, fit
def footprint_params(table, stage, ctx):
    """``(base_gb, per_cpu_gb)`` of the stage for this observation class, or None when the table has no estimate for it."""
    row = (table.get("footprint") or {}).get(stage)
    if not row:
        return None
    base, per = float(row.get("base_gb", 0)), float(row.get("per_cpu_gb", 0))
    for case in row.get("cases") or []:
        if _matches(case["when"], ctx):
            base, per = float(case.get("base_gb", base)), float(case.get("per_cpu_gb", per))
    return base, per


def footprint_gb(table, stage, ctx, cpus):
    p = footprint_params(table, stage, ctx)
    return None if p is None else round(p[0] + p[1] * cpus, 2)


def _alt(stage, what, fit):
    return {"cpus": "--slurm_set %s.cpus=%s", "mem": "--slurm_set %s.mem_gb=%s", "time": "--slurm_set %s.time=%s"}[what] % (stage, fit)


def _issue(stage, what, requested, limit, fit, reason, hard):
    return {"what": what, "requested": requested, "limit": limit, "fit": fit, "reason": reason, "hard": hard,
            "use": _alt(stage, what, fit) if fit is not None else None}


def _fit_slurm(stage, cpus, mem, tlim, limits):
    """Issues of a request against the largest node / partition of a cluster, and the values that fit. ``(issues, cpus, mem, time)``."""
    issues, c, m, t = [], cpus, mem, tlim
    src = limits.get("source", "SLURM")
    if t and limits.get("max_time") and parse_time_seconds(t) > parse_time_seconds(limits["max_time"]):
        issues.append(_issue(stage, "time", t, limits["max_time"], limits["max_time"],
                             "time %s exceeds the partition limit %s (%s): it would stay pending for ever" % (t, limits["max_time"], src), True))
        t = limits["max_time"]
    if m and limits.get("max_mem_gb") and m > limits["max_mem_gb"]:
        issues.append(_issue(stage, "mem", m, limits["max_mem_gb"], int(limits["max_mem_gb"]),
                             "memory %d GB exceeds the largest node (%d GB, %s): no node can run it" % (m, limits["max_mem_gb"], src), True))
        m = int(limits["max_mem_gb"])
    if limits.get("max_cpus") and c > limits["max_cpus"]:
        issues.append(_issue(stage, "cpus", c, limits["max_cpus"], int(limits["max_cpus"]),
                             "%d CPUs exceed the largest node (%d CPUs, %s): no node can run it" % (c, limits["max_cpus"], src), True))
        c = int(limits["max_cpus"])
    nodes = limits.get("nodes_fit")
    if nodes and not any(nc >= c and nm >= (m or 0) for nc, nm in nodes):
        cand = [(nc, nm) for nc, nm in nodes if nm >= (m or 0)]
        best = max(cand)[0] if cand else None
        if best:
            issues.append(_issue(stage, "cpus", c, best, best, "no single node has %d CPUs and %d GB together (nodes: %s); the nodes with %d GB have at most %d CPUs"
                                 % (c, m or 0, ", ".join("%dc/%dGB" % n for n in nodes), m or 0, best), True))
            c = best
    return issues, c, m, t


def _fit_local(stage, cpus, mem, ctx, table, budget):
    """Issues of a request against this machine's CPUs and RAM (after the shares), and the values that fit."""
    issues, c, m = [], cpus, mem
    bc, bm = budget.get("cpus"), budget.get("mem_gb")
    share = "%d%% of the %s CPUs usable here%s" % (round(100 * budget.get("cpu_fraction", 1)), budget.get("usable_cpus"),
                                                    (", shared by %s parallel jobs" % budget["parallel_jobs"]) if budget.get("parallel_jobs", 1) > 1 else "")
    if bc and c > bc:
        issues.append(_issue(stage, "cpus", c, bc, bc, "%d CPUs asked, this machine can give %d (%s)" % (c, bc, share), False))
        c = bc
    fpp = footprint_params(table, stage, ctx)
    if fpp is not None and bm:
        base, per = fpp
        need = base + per * c
        if need > bm:
            n = c
            while n > 1 and base + per * n > bm:
                n -= 1
            what = "%s needs about %.0f GB with %d workers (%.1f GB each + %.0f GB) and %.0f GB of the %.0f GB available is allowed (%d%%)" % (
                stage, need, c, per, base, bm, budget.get("available_gb") or 0, round(100 * budget.get("mem_fraction", 1)))
            if base + per * n <= bm:
                issues.append(_issue(stage, "cpus", c, n, n, what + ": %d workers fit" % n, False))
                c = n
            else:
                total = budget.get("total_gb")
                hard = bool(total and base + per > total)
                issues.append(_issue(stage, "mem", round(base + per, 1), round(bm, 1), None,
                                     what + (": even one worker (%.0f GB) does not fit%s" % (base + per, " in the RAM of this machine, it can never run" if hard else " the allowed share"))
                                     , hard))
                if n < c:
                    c = max(1, n)
    elif fpp is None and m and bm and m > bm:
        issues.append(_issue(stage, "mem", m, int(bm), int(bm), "memory %d GB asked, this machine can give %.0f GB (%d%% of what is available)"
                             % (m, bm, round(100 * budget.get("mem_fraction", 1))), False))
        m = max(1, int(bm))
    return issues, c, m


def _priority_settings(table, pname, conf, env, policy_arg=None, class_arg=None, pair_arg=None, previous=None, escalated=False):
    """The resolved depth-first settings of this run, or None when the policy is off."""
    chain, cur = [], pname
    while cur is not None:
        chain.append(cur)
        cur = table["profiles"][cur].get("extends")
    merged = dict(table.get("priority") or {})                # the table's block, then each profile of the chain, nearest last (key by key)
    for n in reversed(chain):
        merged.update(table["profiles"][n].get("priority") or {})
    flat = merged
    pol, pol_src = _operator(policy_arg, "PYAPS_SLURM_PRIORITY", "slurm_priority", env, conf)
    pol = norm_word(pol) if pol is not None else norm_word(flat.get("policy", "off"))
    pol_src = pol_src or "table"
    if pol not in PRIORITY_POLICIES:
        raise ResourceTableError("priority policy %r: one of %s" % (pol, ", ".join(PRIORITY_POLICIES)))
    cls, cls_src = _operator(class_arg, "PYAPS_PRIORITY_CLASS", "slurm_priority_class", env, conf)
    cls = norm_word(cls) if cls is not None else flat.get("default_class", "scheduled")
    cls_src = cls_src or "table default_class"
    if cls not in PRIORITY_CLASSES:
        raise ResourceTableError("priority class %r: one of %s" % (cls, ", ".join(PRIORITY_CLASSES)))
    pair, pair_src = _operator(pair_arg, "PYAPS_PAIR_INDEX", None, env, conf)
    try:
        pair = int(pair) if pair is not None else 0
    except (TypeError, ValueError):
        raise ResourceTableError("pair index %r is not a whole number" % (pair,))
    pair_src = pair_src or "default 0"
    if escalated and previous and previous.get("pair_index") is not None:
        pair, pair_src = int(previous["pair_index"]), "kept from the first generation of this chain (escalation retry)"
    base = (flat.get("classes") or {}).get(cls, 0)
    out = {"policy": pol, "policy_source": pol_src, "class": cls, "class_source": cls_src, "pair_index": max(0, pair), "pair_source": pair_src,
           "class_base": base, "stage_step": flat.get("stage_step", 0), "pair_step": flat.get("pair_step", 0), "pair_cap": int(flat.get("pair_cap", 8)),
           "stage_remaining": dict(flat.get("stage_remaining") or {})}
    out["pair_nice"] = out["pair_step"] * min(out["pair_index"], out["pair_cap"])
    return out if pol == "depth_first" else dict(out, off=True)


def nice_of(prio, stage):
    """``--nice`` of a stage under the settings ``prio`` (class base + stage_step x stages still to come + pair_step x pair index)."""
    return int(prio["class_base"] + prio["stage_step"] * prio["stage_remaining"].get(stage, 0) + prio["pair_nice"])


def resolve(family, ctx, table=None, conf=None, profile=None, mp_base=None, cli_set=None, escalation=None, site=None,
            limits=None, env=None, quiet=False, environment=None, limits_policy=None, hpc=None, target=None, machine=None, ferre_exe=None,
            parallel_jobs=None, priority_policy=None, priority_class=None, pair_index=None, previous_priority=None, uapsid=None):
    """Resolve the resources of every stage of ``family`` for the observation described by ``ctx``.

    ``ctx``       ``{res: 'LR'|'HR', stack_level: 0|1|3|4, grid: 'm'|'n'|..., ...}`` (family is added).
    ``mp_base``   the legacy ``--mp_*`` list: replaces the base CPU counts of the family (the cases still apply to it).
    ``cli_set``   ``['FR.mem_gb=120', ...]`` from ``--slurm_set``.
    ``escalation``  ``{stage: {mem_factor, time_factor}}`` from :func:`read_escalation`.
    ``limits``    dict (max_mem_gb, max_cpus, max_time) to use instead of looking at the place; ``{}`` = no limits.
    ``environment``  an :class:`Environment` made once by :func:`detect_environment` (the runner shares one between the families); otherwise
                  the place is probed here (unless ``limits`` is given).
    ``limits_policy`` off | warn | clamp | error (else ``PYAPS_SLURM_LIMITS``, script_params ``slurm_limits``, the site's, the profile's, warn).
    ``hpc``/``target``/``machine``/``ferre_exe``/``parallel_jobs``  see :func:`detect_environment`.
    ``priority_policy``/``priority_class``/``pair_index``/``previous_priority``/``uapsid``  depth-first order of the jobs (``--nice``, ``--comment``).
    Returns ``{stage: Resources}`` in stage order (a dict that also carries ``.environment`` and ``.priority``).
    Raises :class:`ResourceLimitError` under the policy ``error`` when a request can never run (all stages are listed, with what would fit).
    """
    if family not in FAMILIES:
        raise ResourceTableError("unknown family %r" % family)
    table = table or load_table()
    env = os.environ if env is None else env
    conf = conf or {}
    # ---- the place and the policy
    op_policy, op_src = _operator(limits_policy, "PYAPS_SLURM_LIMITS", "slurm_limits", env, conf)
    op_policy = norm_word(op_policy) if op_policy is not None else None
    if op_policy is not None and op_policy not in LIMIT_POLICIES:
        raise ResourceTableError("limits policy %r (from %s): one of %s" % (op_policy, op_src, ", ".join(LIMIT_POLICIES)))
    if environment is None:
        if limits is not None:                                 # the caller names the limits: nothing is looked at
            environment = Environment(target="slurm", limits=dict(limits, source="given") if limits else {})
        else:
            guess = choose_profile_ex(table, conf, profile, env, site_profile=None)[0]      # the word `site` needs the place: it falls back here
            wants_site = any(str(v or "").strip().lower() == "site" for v in (profile, env.get("PYAPS_SLURM_PROFILE"), conf.get("slurm_profile")))
            named = bool(site or env.get("PYAPS_SLURM_SITE") or conf.get("slurm_site"))
            skip = op_policy == "off" or (op_policy is None and norm_word(table["profiles"][guess].get("limits", "warn")) == "off"
                                          and not wants_site and not named)
            environment = detect_environment(table, conf, env, site=site, machine=machine, target=target, hpc=hpc, ferre_exe=ferre_exe,
                                             parallel_jobs=parallel_jobs, probe=not skip)
    environment.budget.setdefault("parallel_jobs", environment.parallel_jobs)
    site_profile = None
    if environment.site_entry:
        site_profile = environment.site_entry.get("profile")
        if environment.ferre_openmp and environment.site_entry.get("profile_openmp"):
            site_profile = environment.site_entry["profile_openmp"]
    pname, pname_src = choose_profile_ex(table, conf, profile, env, site_profile=site_profile)
    prof = table["profiles"][pname]
    if op_policy is not None:
        policy, policy_src = op_policy, op_src
    elif environment.site_entry.get("limits") and environment.target == "local":
        policy, policy_src = norm_word(environment.site_entry["limits"]), "site %s" % environment.site["name"]
    else:
        policy, policy_src = norm_word(prof.get("limits", "warn")), "profile %s" % pname
    environment.policy, environment.policy_source = policy, policy_src
    flat = flatten_profile(table, pname)[family]
    stages = FAMILIES[family]
    ctx = dict(ctx)
    ctx["family"] = family
    if "stack_level" in ctx and "stack" not in ctx:
        ctx["stack"] = STACK_NAMES.get(int(ctx["stack_level"]), "other")
    if mp_base is not None and len(mp_base) != len(stages):
        raise ResourceTableError("--mp list for %s needs %d values (%s), got %d" % (family, len(stages), ",".join(stages), len(mp_base)))
    overrides = collect_overrides(conf, cli_set, family)
    eff_limits = {} if policy == "off" else environment.limits
    escalation = escalation or {}
    prio = _priority_settings(table, pname, conf, env, priority_policy, priority_class, pair_index, previous_priority, escalated=bool(escalation))
    if prio and environment.target == "local":              # nothing is queued in a bash run: no --nice to give
        prio = dict(prio, off=True)
    out, errors = ResolvedSet(), []
    out.environment, out.priority, out.profile_source = environment, prio, pname_src
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
        if esc and environment.target == "slurm" and environment.limits:
            lim = environment.limits
            if mem and lim.get("max_mem_gb") and mem > lim["max_mem_gb"]:
                trace.append("escalation: memory limited to the largest node (%d GB)" % lim["max_mem_gb"])
                mem = int(lim["max_mem_gb"])
            if tlim and lim.get("max_time") and parse_time_seconds(tlim) > parse_time_seconds(lim["max_time"]):
                trace.append("escalation: time limited to the partition maximum (%s)" % lim["max_time"])
                tlim = lim["max_time"]
        requested = {"cpus": int(cpus), "mem_gb": mem, "time": None if tlim is None else str(tlim)}
        # ---- compare with the place; the policy decides what happens to a request that does not fit (one that fits is never touched)
        adjustments, unresolved = [], False
        if eff_limits or (environment.target == "local" and environment.budget):
            if environment.target == "slurm":
                issues, fc, fm, ft = _fit_slurm(stage, cpus, mem, tlim, eff_limits) if eff_limits else ([], cpus, mem, tlim)
            else:
                b = dict(environment.budget)
                issues, fc, fm = _fit_local(stage, cpus, mem, ctx, table, b) if policy != "off" else ([], cpus, mem)
                ft = tlim
            for iss in issues:
                if policy == "clamp":
                    action = "clamp" if iss["fit"] is not None else "warn"
                elif policy == "error":
                    action = "error" if iss["hard"] else "warn"
                else:
                    action = "warn"
                iss["action"] = action
                adjustments.append(iss)
                if action == "error":
                    errors.append("%s: %s. %s To submit anyway: --slurm_limits warn (or off)." % (
                        stage, iss["reason"], ("It would fit with %s." % iss["use"]) if iss.get("use") else "No smaller request fits: this place needs more memory."))
                elif action == "warn" or iss["fit"] is None:
                    unresolved = True
                    tail = ("; it would fit with %s" % iss["use"]) if iss.get("use") else ""
                    warnings.append("%s (limits policy %s: kept as asked%s; --slurm_limits clamp lowers what can be lowered, --slurm_limits off silences the check)"
                                    % (iss["reason"], policy, tail))
                else:                                              # clamp
                    warnings.append("%s: clamped to %s (limits policy clamp; asked %s)" % (iss["reason"], iss["fit"], iss["requested"]))
            if policy == "clamp":
                cpus, mem, tlim = fc, fm, ft
                for iss in issues:
                    if iss["fit"] is not None:
                        trace.append("limits clamp: %s %s -> %s (%s)" % (iss["what"], iss["requested"], iss["fit"], iss["reason"]))
        if tlim is not None:
            tlim = str(tlim)
        fpg = footprint_gb(table, stage, ctx, int(cpus))
        nice = comment = None
        if prio and not prio.get("off"):
            nice = nice_of(prio, stage)
            comment = "pyaps:%s:p%d:%s:%s" % (re.sub(r"[^0-9A-Za-z_.-]", "", str(uapsid)) if uapsid else "manual", prio["pair_index"], prio["class"], stage)
        out[stage] = Resources(stage=stage, family=family, cpus=int(cpus), mem_gb=mem, time=tlim,
                               status=row.get("status", "legacy"), profile=pname, trace=trace, warnings=warnings,
                               requested=requested, adjustments=adjustments, fits=not unresolved, footprint_gb=fpg, policy=policy,
                               nice=nice, comment=comment, environment=environment)
        if not quiet:
            for w in warnings:
                print("Warning: SLURM resources %s: %s" % (stage, w))
    if errors:
        raise ResourceLimitError("a request can never run (limits policy error, %s):\n  " % environment.policy_source + "\n  ".join(errors))
    return out


class ResolvedSet(dict):
    """``{stage: Resources}`` plus what was looked at: ``.environment`` (:class:`Environment`), ``.priority`` (settings or None), ``.profile_source``."""
    environment = None
    priority = None
    profile_source = None


# --------------------------------------------------------------------------------------------------------- reporting
def effective_summary(resolved: dict, ctx: dict, table: dict, family: str, uapsid=None) -> dict:
    stages = list(resolved.values())
    out = {
        "schema": 1, "module_version": __version__, "uapsid": uapsid, "family": family,
        "profile": stages[0].profile if stages else None, "table": table.get("_path"), "table_sha256": table.get("_sha256"),
        "context": {k: ctx[k] for k in sorted(ctx)}, "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stages": {r.stage: r.as_dict() for r in stages},
    }
    envm = getattr(resolved, "environment", None)
    if envm is not None:
        out["profile_source"] = getattr(resolved, "profile_source", None)
        out["environment"] = envm.summary()
        out["limits_policy"] = envm.policy
    prio = getattr(resolved, "priority", None)
    if prio:
        out["priority"] = {k: v for k, v in prio.items() if k != "stage_remaining"}
    return out


def write_effective(path, summary: dict) -> None:
    """Write the effective values beside the job scripts, under the run's tag (the uapsid): the folder of an OB holds the scripts of every
    pair of the OB, and the L2 and CS chains of a pair. Never raises: a read-only folder must not stop the run."""
    try:
        p = Path(path) / EFFECTIVE_NAME
        data = _read_json(p)
        data["schema"] = 2
        data.setdefault("runs", {})[effective_tag(summary.get("uapsid"), summary.get("family"))] = summary
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


def job_header_lines(res: Resources, environment=None) -> list:
    """Shell lines for the top of a stage script: print what the job asked for, what the machine check made of it, and what SLURM gave it."""
    mem = ("%d GB" % res.mem_gb) if res.mem_gb else "SLURM default (DefMemPerCPU)"
    q = lambda s: str(s).replace("'", "'\\''")
    lines = [
        "echo 'PyAPS SLURM resources: stage=%s profile=%s status=%s cpus=%d mem=%s time=%s mp=%d threads=1'"
        % (q(res.stage), q(res.profile), q(res.status), res.cpus, q(mem), q(res.time or "no limit"), res.mp),
    ]
    if res.requested:
        rq = res.requested
        asked = "cpus=%s mem=%s time=%s" % (rq.get("cpus"), ("%s GB" % rq["mem_gb"]) if rq.get("mem_gb") else "default", rq.get("time") or "no limit")
        changed = [a for a in res.adjustments if a.get("action") == "clamp"]
        env_txt = ""
        if environment is not None:
            site = (environment.site or {}).get("name") or "unnamed"
            env_txt = " site=%s target=%s" % (site, environment.target)
        lines.append("echo 'PyAPS machine check: policy=%s%s asked %s -> granted cpus=%d mem=%s%s%s'" % (
            q(res.policy or "n/a"), q(env_txt), q(asked), res.cpus, q(mem),
            " (%d value(s) lowered)" % len(changed) if changed else (" (does not fit, kept as asked)" if res.fits is False else " (fits or unchecked)"),
            ("; footprint ~%s GB" % res.footprint_gb) if res.footprint_gb is not None else ""))
    if res.nice is not None or res.comment:
        lines.append("echo 'PyAPS priority: nice=%s comment=%s'" % (q(res.nice if res.nice is not None else 0), q(res.comment or "-")))
    lines += [
        'echo "PyAPS SLURM granted:   job=${SLURM_JOB_ID:-n/a} ntasks=${SLURM_NTASKS:-n/a} cpus_on_node=${SLURM_CPUS_ON_NODE:-n/a} '
        'mem_per_node_MB=${SLURM_MEM_PER_NODE:-n/a} node=${SLURMD_NODENAME:-n/a}"',
        'if [ -n "${SLURM_NTASKS:-}" ] && [ "${SLURM_NTASKS}" != "%d" ]; then echo "WARNING: requested %d CPUs but SLURM_NTASKS=${SLURM_NTASKS}"; fi'
        % (res.cpus, res.cpus),
    ]
    return lines


def format_environment(envm: Environment, profile_source=None) -> str:
    """Operator text: the place, the site, the limits and the policy."""
    L = []
    site = envm.site or {}
    L.append("place     : target=%s  site=%s%s" % (envm.target, site.get("name") or "not named", (" [%s]" % site["source"]) if site.get("source") else ""))
    m = envm.machine
    if m:
        loc, sl = m.get("local", {}), m.get("slurm", {})
        if envm.target == "slurm" and sl.get("present"):
            L.append("machine   : SLURM %s cluster=%s, %d nodes (%s), memory enforced=%s, ferre.x=%s" % (
                sl.get("version"), sl.get("cluster"), len(sl.get("nodes", [])), "up" if sl.get("usable") else "UNREADABLE",
                sl.get("memory_enforced"), {True: "OpenMP", False: "single-thread", None: "unknown"}[envm.ferre_openmp]))
        else:
            c, mm = loc.get("cpus", {}), loc.get("memory", {})
            L.append("machine   : %s (%s) %s usable CPUs, %s GB RAM available (%s GB total, cgroup limit %s), ferre.x=%s" % (
                m.get("os", {}).get("hostname"), m.get("kind"), c.get("usable"), mm.get("usable_available_gb"), mm.get("total_gb"), mm.get("cgroup_limit_gb"),
                {True: "OpenMP", False: "single-thread", None: "unknown"}[envm.ferre_openmp]))
    else:
        L.append("machine   : not probed")
    if envm.limits:
        lim = {k: v for k, v in envm.limits.items() if k not in ("nodes_fit", "cpus", "mem_gb")}
        L.append("one job   : %s" % ", ".join("%s=%s" % kv for kv in lim.items()))
    if envm.target == "local" and envm.budget:
        b = envm.budget
        L.append("shares    : %s%% of the usable CPUs and %s%% of the available RAM, %s parallel job(s) -> %s CPUs, %s GB per job" % (
            round(100 * b.get("cpu_fraction", 1)), round(100 * b.get("mem_fraction", 1)), envm.parallel_jobs, b.get("cpus"), b.get("mem_gb")))
    L.append("policy    : limits=%s (%s)%s" % (envm.policy, envm.policy_source, ("   profile from: %s" % profile_source) if profile_source else ""))
    for msg in envm.messages:
        L.append("note      : " + msg)
    return "\n".join(L)


def format_adjustments(resolved: dict) -> str:
    rows = []
    for r in resolved.values():
        for a in r.adjustments:
            rows.append("  %-10s %-5s %s -> %s  [%s]  %s%s" % (r.stage, a["what"], a["requested"], a["fit"] if a["action"] == "clamp" else "(kept)" if a["action"] == "warn" else "(refused)",
                                                             a["action"], a["reason"], ("  | use: " + a["use"]) if a.get("use") and a["action"] != "clamp" else ""))
    return "\n".join(rows)


# ------------------------------------------------------------------------------------------------------------------ CLI
def _ctx_from_args(args) -> dict:
    ctx = {"res": args.res.upper(), "grid": args.grid}
    level = {"single": 0, "stack": 1, "super": 3, "other": 4}[args.stack]
    ctx["stack_level"] = level
    return ctx


def _add_place_options(p):
    p.add_argument("--limits", choices=list(LIMIT_POLICIES), help="policy for a request that does not fit the machine (default: PYAPS_SLURM_LIMITS, the site's, the profile's)")
    p.add_argument("--site", help="site of the table or of slurm_sites.yaml (default: detected)")
    p.add_argument("--target", choices=["slurm", "local"], help="where the jobs run: the SLURM cluster, or this machine in bash (default: SLURM when present)")
    p.add_argument("--parallel-jobs", type=int, help="jobs running at the same time on this machine (the CPUs and RAM are shared between them)")
    p.add_argument("--ferre-exe", help="ferre.x to look at (OpenMP or not); default: FERRE_EXE of the script_params")
    p.add_argument("--machine-file", help="a probe made elsewhere (python -m PyAPS.aps_slurm_resources probe --json > file): see what a run THERE would do")
    p.add_argument("--script-params", help="script_params.yaml whose keys (slurm_profile, slurm_site, slurm_limits, FERRE_EXE, PyAPS_CONFIG ...) apply")


def _conf_from(args) -> dict:
    path = getattr(args, "script_params", None)
    if not path:
        return {}
    import yaml
    data = yaml.safe_load(Path(path).read_text()) or {}
    conf = {k: v for k, v in data.items() if isinstance(k, str)}
    for k, v in list(conf.items()):                          # ${PYAPS_HOME} and friends, as the runner expands them
        if isinstance(v, str):
            conf[k] = os.path.expandvars(os.path.expanduser(v))
    return conf


def _machine_from(args):
    path = getattr(args, "machine_file", None)
    if not path:
        return None
    data = json.loads(Path(path).read_text())
    return data.get("machine", data) if isinstance(data, dict) else None


def _resolve_from_args(args, table, family=None, **extra):
    ctx = _ctx_from_args(args)
    kw = dict(table=table, profile=getattr(args, "profile", None), cli_set=getattr(args, "set", []), site=getattr(args, "site", None),
              limits={} if getattr(args, "no_limits", False) else None, limits_policy=getattr(args, "limits", None), target=getattr(args, "target", None),
              parallel_jobs=getattr(args, "parallel_jobs", None), ferre_exe=getattr(args, "ferre_exe", None), machine=_machine_from(args),
              conf=_conf_from(args), quiet=True, priority_policy=getattr(args, "priority", None), priority_class=getattr(args, "priority_class", None),
              pair_index=getattr(args, "pair_index", None))
    kw.update(extra)
    return resolve(family or args.family, ctx, **kw)


def _cmd_probe(args) -> int:
    res = rprobe.probe(ferre_exe=args.ferre_exe, use_cache=False)
    print(json.dumps(res, indent=1, default=str) if args.json else rprobe.format_probe(res))
    return 0


def _cmd_site(args, table) -> int:
    res = _resolve_from_args(args, table)
    envm = res.environment
    if args.json:
        print(json.dumps({"environment": envm.summary(), "profile": next(iter(res.values())).profile, "profile_source": res.profile_source,
                          "priority": {k: v for k, v in (res.priority or {}).items() if k != "stage_remaining"}}, indent=1, default=str))
        return 0
    print(format_environment(envm, res.profile_source))
    first = next(iter(res.values()))
    print("profile   : %s%s" % (first.profile, "   (the word `site` in --slurm_profile / slurm_profile would take it from the site)" if envm.site_entry.get("profile") else ""))
    if res.priority and not res.priority.get("off"):
        p = res.priority
        print("priority  : depth_first, class %s (%s), pair %s -> nice %s (RR) ... %s (last stage)" % (
            p["class"], p["class_source"], p["pair_index"], max(nice_of(p, s) for s in res.keys()), min(nice_of(p, s) for s in res.keys())))
    else:
        print("priority  : off (no --nice, no --comment)")
    print("")
    print("%s  %s %s grid=%s  (table %s)" % (args.family, args.res.upper(), args.stack, args.grid, table["_sha256"]))
    print(format_table(res))
    adj = format_adjustments(res)
    if adj:
        print("\nrequests that do not fit this place:\n" + adj)
    return 0


def _cmd_check(args, table) -> int:
    """Every stage of one observation class against this place: fits / needs a change / can never run."""
    asked = _resolve_from_args(args, table, limits_policy="warn")           # what the table asks, nothing changed
    envm = asked.environment
    would = _resolve_from_args(args, table, limits_policy="clamp", machine=envm.machine) if envm.limits else asked
    rows, worst = [], 0
    nodes = (envm.machine or {}).get("slurm", {}).get("nodes", []) if envm.target == "slurm" else []
    for st, r in asked.items():
        w = would[st]
        if not r.adjustments:
            verdict = "fits"
            if nodes:
                free = [n for n in nodes if n["up"] and n["cpus_alloc"] is not None and n["alloc_mem_mb"] is not None
                        and n["cpus"] - n["cpus_alloc"] >= r.cpus and (n["real_mem_mb"] - n["alloc_mem_mb"]) / 1024.0 >= (r.mem_gb or 0)]
                verdict += "  (a node has it free now)" if free else "  (busy now: it would queue)"
            elif r.footprint_gb is not None and envm.target == "local":
                verdict += "  (~%.0f GB expected)" % r.footprint_gb
        else:
            hard = any(a["hard"] for a in r.adjustments)
            merged = {}                                   # one line per value: asked first, what fits last (a stage can be cut by the CPU share and then by the RAM)
            for a in r.adjustments:
                m = merged.setdefault(a["what"], dict(a))
                m["fit"], m["use"] = a["fit"], a.get("use")
            parts = ["%s %s -> %s" % (a["what"], a["requested"], a["fit"] if a["fit"] is not None else "nothing fits") for a in merged.values()]
            verdict = ("NEVER RUNS as asked: " if hard else "needs a change: ") + "; ".join(parts) + "  | " + "; ".join(a["use"] for a in merged.values() if a.get("use"))
            worst = max(worst, 3 if hard else 1)
        rows.append("  %-10s asked cpus=%-3d mem=%-8s time=%-10s %s" % (st, r.cpus, ("%d GB" % r.mem_gb) if r.mem_gb else "default", r.time or "-", verdict))
    if args.json:
        print(json.dumps({"environment": envm.summary(), "stages": {st: asked[st].as_dict() for st in asked},
                          "if_clamped": {st: {"cpus": would[st].cpus, "mem_gb": would[st].mem_gb, "time": would[st].time} for st in would}}, indent=1, default=str))
        return worst
    real = _resolve_from_args(args, table, machine=envm.machine)            # the policy that would really apply (the check itself looks with warn)
    print(format_environment(real.environment, asked.profile_source))
    print("\n%s %s %s grid=%s, profile %s:" % (args.family, args.res.upper(), args.stack, args.grid, next(iter(asked.values())).profile))
    print("\n".join(rows))
    print("\n" + ["Everything fits.", "Some stages need a change (policy warn keeps them as asked; clamp would lower them).",
                   "", "At least one stage can never run on this place as asked (policy error would refuse it)."][worst])
    return worst


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m PyAPS.aps_slurm_resources", description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, hlp in (("show", "print the resources of every stage for one observation class"),
                      ("explain", "like show, with the rule trace of every value"),
                      ("site", "the place the jobs would run (site, machine or cluster, limits, policy) and the profile that applies"),
                      ("check", "does each stage fit this machine / cluster? (exit 0 fits, 1 needs a change, 3 can never run)"),
                      ("probe", "what this machine and its SLURM look like (read only)"),
                      ("validate", "check the table")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("--file", help="resource table (default: configs/slurm_resources.yaml)")
        if name == "probe":
            p.add_argument("--json", action="store_true")
            p.add_argument("--ferre-exe", help="ferre.x to look at")
        elif name != "validate":
            p.add_argument("--profile", help="profile name, or `site` (default: PYAPS_SLURM_PROFILE or the table's default)")
            p.add_argument("--family", choices=sorted(FAMILIES), default="MOS")
            p.add_argument("--res", choices=["LR", "HR", "lr", "hr"], default="LR")
            p.add_argument("--stack", choices=["single", "stack", "super", "other"], default="stack")
            p.add_argument("--grid", default="m", help="FERRE grid prefix of the run (m fine, n coarse)")
            p.add_argument("--set", action="append", default=[], metavar="STAGE.key=value")
            p.add_argument("--no-limits", action="store_true", help="ignore node limits (nothing is looked at)")
            p.add_argument("--priority", choices=list(PRIORITY_POLICIES), help="depth-first priority policy")
            p.add_argument("--priority-class", choices=list(PRIORITY_CLASSES))
            p.add_argument("--pair-index", type=int, help="order of the pair in the night (0 = first)")
            p.add_argument("--json", action="store_true")
            _add_place_options(p)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "probe":
            return _cmd_probe(args)
        table = load_table(args.file)
        if args.cmd == "validate":
            n = len(table["profiles"])
            print("OK: %s (%d profiles: %s; sites: %s)" % (table["_path"], n, ", ".join(table["profiles"]), ", ".join(table.get("sites") or {}) or "none"))
            return 0
        if args.cmd == "site":
            return _cmd_site(args, table)
        if args.cmd == "check":
            return _cmd_check(args, table)
        res = _resolve_from_args(args, table)
    except ResourceTableError as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 2
    print("%s  profile=%s  %s %s grid=%s  (table %s %s)" % (args.family, next(iter(res.values())).profile, args.res.upper(),
                                                         args.stack, args.grid, table["_path"], table["_sha256"]))
    print(format_table(res))
    adj = format_adjustments(res)
    if adj:
        print("\nrequests that do not fit this place:\n" + adj)
    if args.cmd == "explain":
        print("\n" + format_environment(res.environment, res.profile_source))
        for r in res.values():
            print("\n%s:" % r.stage)
            for t in r.trace + ["WARNING: " + w for w in r.warnings]:
                print("   - " + t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
