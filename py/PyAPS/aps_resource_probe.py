#!/usr/bin/env python3
"""
aps_resource_probe.py - what machine (or cluster) will run the jobs, and what it can really give.

The SLURM resource table (``aps_slurm_resources``) says what a stage would like; this module says what the place where the stage runs
can give, so that a request is checked before it is made.  Two kinds of place are probed:

* a **SLURM cluster**: version, plugins (is memory enforced?), partitions (``MaxTime``, ``MaxMemPerNode``, the default one) and nodes
  (CPUs, ``RealMemory``, what is allocated now, state, features), from ``scontrol show config|nodes|partition`` with ``sinfo`` as the
  fallback; every command has a timeout, a hung ``scontrol`` is given up after one try;
* **this machine** (a laptop, a workstation, a shared host, a pod or a container without SLURM, or the node itself): the CPUs it may use
  (``os.sched_getaffinity`` and the cgroup CPU quota, v1 and v2), the RAM it may use (``/proc/meminfo`` and the cgroup memory limit),
  swap, load, the operating system (Linux, macOS, WSL) and whether it looks like a container.

Plus one feature that decides how to ask for CPUs: whether ``ferre.x`` is built with OpenMP (it is, when it links ``libgomp``).

Rules: standard library only (``psutil`` is not needed); nothing is ever written; a failure of any probe is recorded as a message in
``problems`` and the value stays ``None`` ("unknown"), never an exception; the whole probe is bounded by ``deadline`` seconds
(default 12); the answer is cached in the process for ``PYAPS_PROBE_TTL`` seconds (default 60).

    python -m PyAPS.aps_slurm_resources probe      # prints what this module sees
"""
from __future__ import annotations

import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time

__version__ = "1.0.0"

GIB = 1024 ** 3
#: seconds one external command may take, and the whole probe
CMD_TIMEOUT = 5.0
DEADLINE = 12.0
CACHE_TTL = 60.0
#: node states in which nothing can start now (the node still counts for "can it ever run")
_DOWN_WORDS = ("DOWN", "DRAIN", "FAIL", "NOT_RESPONDING", "UNKNOWN", "FUTURE", "POWERED_DOWN", "MAINT")


# ------------------------------------------------------------------------------------------------------------ small helpers
class CmdResult:
    """Outcome of one external command: ``status`` is ok | missing | timeout | error."""
    __slots__ = ("status", "rc", "out", "err")

    def __init__(self, status, rc=None, out="", err=""):
        self.status, self.rc, self.out, self.err = status, rc, out or "", err or ""

    @property
    def ok(self):
        return self.status == "ok"


def run_command(argv, timeout=CMD_TIMEOUT) -> CmdResult:
    """Run ``argv`` with a timeout; never raises."""
    try:
        cp = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return CmdResult("missing")
    except subprocess.TimeoutExpired:
        return CmdResult("timeout")
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return CmdResult("error", err=str(exc))
    if cp.returncode != 0:
        return CmdResult("error", rc=cp.returncode, out=cp.stdout, err=(cp.stderr or "").strip()[:300])
    return CmdResult("ok", rc=0, out=cp.stdout)


class Host:
    """The window onto a machine: real by default, replaced in tests (a fake root with ``proc/`` and ``sys/``, a fake command runner...)."""

    def __init__(self, root="/", run=None, environ=None, cpu_count=None, affinity=None, platform_name=None, hostname=None,
                 loadavg=None, which=None, clock=time.monotonic):
        self.root = str(root)
        self.run = run or run_command
        self.environ = os.environ if environ is None else environ
        self._cpu_count, self._affinity, self._loadavg = cpu_count, affinity, loadavg
        self.platform_name = platform_name or sys.platform
        self._hostname = hostname
        self.which = which or shutil.which
        self.clock = clock

    def path(self, p):
        return os.path.join(self.root, str(p).lstrip("/"))

    def read(self, p, limit=2_000_000):
        try:
            with open(self.path(p), "r", errors="replace") as fh:
                return fh.read(limit)
        except (OSError, ValueError):
            return None

    def exists(self, p):
        return os.path.exists(self.path(p))

    def cpu_count(self):
        if self._cpu_count is not None:
            return self._cpu_count
        return os.cpu_count()

    def affinity(self):
        """Number of CPUs this process may run on (``None`` where the call does not exist, e.g. macOS)."""
        if self._affinity is not None:
            return self._affinity
        try:
            return len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            return None

    def loadavg(self):
        if self._loadavg is not None:
            return self._loadavg
        try:
            return os.getloadavg()
        except (AttributeError, OSError):
            return None

    def hostname(self):
        return self._hostname or platform.node() or ""


def _int(text, default=None):
    try:
        return int(str(text).strip().rstrip("+*"))
    except (TypeError, ValueError):
        return default


def _gb(nbytes):
    return None if nbytes is None else round(nbytes / GIB, 2)


# -------------------------------------------------------------------------------------------------------------- hostlists
def expand_hostlist(text) -> list:
    """``weave-master,weave-node[1-2]`` -> names; ``n[01-03,07]`` -> n01 n02 n03 n07. Unparseable pieces are kept as they are."""
    text = (text or "").strip()
    if not text or text in ("(null)", "N/A"):
        return []
    parts, depth, cur = [], 0, ""
    for ch in text:                                           # split on commas outside brackets
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    names = []
    for part in parts:
        m = re.match(r"^(.*?)\[([^\]]+)\](.*)$", part)
        if not m:
            if part:
                names.append(part)
            continue
        pre, body, post = m.groups()
        for rng in body.split(","):
            a, _, b = rng.partition("-")
            if a.isdigit() and (not b or b.isdigit()):
                width = len(a) if a.startswith("0") else 0
                for i in range(int(a), int(b or a) + 1):
                    names.append("%s%0*d%s" % (pre, width, i, post))
            else:
                names.append(pre + rng + post)
            if len(names) > 20000:                           # a runaway range must not eat the memory
                return names
    return names


# ---------------------------------------------------------------------------------------------------------------- OS / kind
def probe_os(host: Host) -> dict:
    plat = host.platform_name
    info = {"system": "linux" if plat.startswith("linux") else ("macos" if plat == "darwin" else ("windows" if plat.startswith("win") else plat)),
            "hostname": host.hostname()}
    try:
        info["release"] = platform.release()
    except Exception:                                         # pragma: no cover - platform never really fails
        info["release"] = None
    version = (host.read("/proc/version", 4000) or "").lower()
    info["wsl"] = bool("microsoft" in version or "wsl" in version) if info["system"] == "linux" else False
    cg = host.read("/proc/1/cgroup", 4000) or ""
    info["container"] = bool(host.exists("/.dockerenv") or host.exists("/run/.containerenv") or host.environ.get("KUBERNETES_SERVICE_HOST")
                             or re.search(r"(docker|kubepods|containerd|libpod|lxc)", cg))
    info["battery"] = bool(info["system"] == "macos" or any(host.exists("/sys/class/power_supply/BAT%d" % i) for i in (0, 1)))
    return info


# --------------------------------------------------------------------------------------------------------------------- cgroups
def _process_cgroups(host: Host) -> dict:
    """``{'v2': '/path' or None, 'v1': {controller: '/path'}}`` from /proc/self/cgroup."""
    out = {"v2": None, "v1": {}}
    for line in (host.read("/proc/self/cgroup", 20000) or "").splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        hid, ctrls, path = parts
        if hid == "0" and ctrls == "":
            out["v2"] = path or "/"
        else:
            for c in ctrls.split(","):
                out["v1"][c] = path or "/"
    return out


def _ancestors(path):
    path = "/" + str(path).strip("/")
    out = []
    while True:
        out.append(path)
        if path == "/":
            break
        path = path.rsplit("/", 1)[0] or "/"
    return out


def _huge(value):
    """cgroup 'no limit' values: 'max' or a number around 2**63."""
    return value is None or value >= 2 ** 60


def _kv_file(text):
    out = {}
    for line in (text or "").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("-").isdigit():
            out[parts[0]] = int(parts[1])
    return out


def probe_cgroup(host: Host) -> dict:
    """CPU quota (CPUs, a float) and memory limit / usage (bytes) of this process's cgroup, the tightest of the cgroup and its ancestors.
    ``{'version': 1|2|None, 'cpu_quota': float|None, 'mem_limit': int|None, 'mem_used': int|None}``."""
    res = {"version": None, "cpu_quota": None, "mem_limit": None, "mem_used": None}
    cgs = _process_cgroups(host)
    if host.exists("/sys/fs/cgroup/cgroup.controllers") and cgs["v2"] is not None:
        res["version"] = 2
        quotas, limits, used = [], [], None
        for d in _ancestors(cgs["v2"]):
            base = "/sys/fs/cgroup" + ("" if d == "/" else d)
            cpu = (host.read(base + "/cpu.max", 200) or "").split()
            if len(cpu) >= 2 and cpu[0] != "max" and _int(cpu[0]) and _int(cpu[1]):
                quotas.append(_int(cpu[0]) / _int(cpu[1]))
            raw = (host.read(base + "/memory.max", 100) or "max").strip()
            mem = _int(raw) if raw != "max" else None
            if mem is not None and not _huge(mem):
                limits.append((mem, base))
        if limits:
            lim, where = min(limits)
            res["mem_limit"] = lim
            cur = _int(host.read(where + "/memory.current", 100))
            stat = _kv_file(host.read(where + "/memory.stat", 20000))
            if cur is not None:
                res["mem_used"] = max(0, cur - stat.get("inactive_file", 0))
        if quotas:
            res["cpu_quota"] = round(min(quotas), 3)
        return res
    v1 = cgs["v1"]
    if v1:
        quotas, limits = [], []
        res["version"] = 1
        for name in ("cpu", "cpu,cpuacct", "cpuacct,cpu"):
            for d in _ancestors(v1.get("cpu", v1.get("cpu,cpuacct", v1.get("cpuacct", "/")))):
                base = "/sys/fs/cgroup/%s%s" % (name, "" if d == "/" else d)
                q, p = _int(host.read(base + "/cpu.cfs_quota_us", 50)), _int(host.read(base + "/cpu.cfs_period_us", 50))
                if q and q > 0 and p and p > 0:
                    quotas.append(q / p)
            if quotas:
                break
        for d in _ancestors(v1.get("memory", "/")):
            base = "/sys/fs/cgroup/memory" + ("" if d == "/" else d)
            lim = _int(host.read(base + "/memory.limit_in_bytes", 50))
            if lim and not _huge(lim):
                limits.append((lim, base))
        if limits:
            lim, where = min(limits)
            res["mem_limit"] = lim
            cur = _int(host.read(where + "/memory.usage_in_bytes", 50))
            stat = _kv_file(host.read(where + "/memory.stat", 20000))
            if cur is not None:
                res["mem_used"] = max(0, cur - stat.get("total_inactive_file", stat.get("inactive_file", 0)))
        if quotas:
            res["cpu_quota"] = round(min(quotas), 3)
    return res


# ------------------------------------------------------------------------------------------------------- local CPU and memory
def _meminfo(host: Host) -> dict:
    out = {}
    for line in (host.read("/proc/meminfo", 100000) or "").splitlines():
        m = re.match(r"^(\w+):\s+(\d+)\s*(kB)?", line)
        if m:
            out[m.group(1)] = int(m.group(2)) * (1024 if m.group(3) else 1)
    return out


def _macos_memory(host: Host, problems):
    mem = {}
    r = host.run(["sysctl", "-n", "hw.memsize"], 3)
    total = _int(r.out) if r.ok else None
    if total is None:
        problems.append("macOS: sysctl hw.memsize gave nothing (%s)" % r.status)
        return mem
    mem["MemTotal"] = total
    r = host.run(["vm_stat"], 3)
    if r.ok:
        m = re.search(r"page size of (\d+) bytes", r.out)
        page = int(m.group(1)) if m else 4096
        pages = {}
        for line in r.out.splitlines():
            mm = re.match(r"^Pages ([a-z ]+):\s+(\d+)\.", line)
            if mm:
                pages[mm.group(1).strip()] = int(mm.group(2))
        mem["MemAvailable"] = page * (pages.get("free", 0) + pages.get("inactive", 0) + pages.get("speculative", 0))
    return mem


def probe_local(host: Host, problems=None) -> dict:
    """CPUs and memory this process may use on this machine (never raises)."""
    problems = problems if problems is not None else []
    out = {"cpus": {}, "memory": {}, "load": None}
    system = "macos" if host.platform_name == "darwin" else "linux"
    cg = probe_cgroup(host) if system == "linux" else {"version": None, "cpu_quota": None, "mem_limit": None, "mem_used": None}
    # ---- CPUs
    logical = host.cpu_count()
    if system == "macos":
        r = host.run(["sysctl", "-n", "hw.ncpu"], 3)
        logical = _int(r.out, logical) if r.ok else logical
    aff = host.affinity()
    usable, why = None, []
    for label, val in (("logical", logical), ("affinity", aff)):
        if val:
            usable = val if usable is None else min(usable, val)
    if usable is None:
        problems.append("the number of CPUs is unknown (os.cpu_count gave nothing)")
    if cg["cpu_quota"] is not None:
        q = max(1, int(math.floor(cg["cpu_quota"] + 1e-9)) or 1)
        if usable is None or q < usable:
            why.append("cgroup CPU quota %.2f" % cg["cpu_quota"])
            usable = q
    if aff is not None and logical and aff < logical:
        why.append("CPU affinity %d of %d" % (aff, logical))
    out["cpus"] = {"logical": logical, "affinity": aff, "cgroup_quota": cg["cpu_quota"], "usable": usable, "limited_by": why}
    # ---- memory
    mi = _meminfo(host) if system == "linux" else _macos_memory(host, problems)
    total = mi.get("MemTotal")
    if system == "linux":
        avail = mi.get("MemAvailable")
        if avail is None and total is not None:                       # kernels before 3.14
            avail = mi.get("MemFree", 0) + mi.get("Buffers", 0) + mi.get("Cached", 0)
    else:
        avail = mi.get("MemAvailable")
    if total is None:
        problems.append("total RAM is unknown (no /proc/meminfo)" if system == "linux" else "total RAM is unknown")
    limit_total, limit_avail = total, avail
    mwhy = []
    if cg["mem_limit"] is not None and (total is None or cg["mem_limit"] < total):
        limit_total = cg["mem_limit"]
        left = None if cg["mem_used"] is None else max(0, cg["mem_limit"] - cg["mem_used"])
        limit_avail = left if avail is None or left is None else min(avail, left)
        if left is None and avail is None:
            limit_avail = cg["mem_limit"]
        mwhy.append("cgroup memory limit %.1f GB" % (cg["mem_limit"] / GIB))
    out["memory"] = {"total_gb": _gb(total), "available_gb": _gb(avail), "usable_total_gb": _gb(limit_total),
                     "usable_available_gb": _gb(limit_avail), "cgroup_limit_gb": _gb(cg["mem_limit"]),
                     "swap_total_gb": _gb(mi.get("SwapTotal")), "swap_free_gb": _gb(mi.get("SwapFree")), "limited_by": mwhy}
    out["cgroup_version"] = cg["version"]
    la = host.loadavg()
    out["load"] = None if la is None else [round(x, 2) for x in la]
    return out


# ----------------------------------------------------------------------------------------------------------------------- SLURM
_KEY_RE = re.compile(r"(?:^|\s)([A-Z][A-Za-z0-9]*(?:\[\d+\])?)=")


def parse_kv_line(line) -> dict:
    """``Key=value Key2=value with spaces Key3=...`` (one ``scontrol -o`` line) -> dict. A key starts with a capital letter."""
    out, hits = {}, list(_KEY_RE.finditer(line))
    for i, m in enumerate(hits):
        end = hits[i + 1].start() if i + 1 < len(hits) else len(line)
        out[m.group(1)] = line[m.end():end].strip()
    return out


def _node_from_kv(kv) -> dict | None:
    name = kv.get("NodeName")
    if not name:
        return None
    cpus = _int(kv.get("CPUEfctv"), None) or _int(kv.get("CPUTot"), None)
    real = _int(kv.get("RealMemory"))
    spec = _int(kv.get("MemSpecLimit"), 0) or 0
    state = (kv.get("State") or "").upper()
    return {"name": name, "cpus": cpus, "cpus_alloc": _int(kv.get("CPUAlloc")), "cpu_load": kv.get("CPULoad"),
            "real_mem_mb": real, "mem_spec_mb": spec, "alloc_mem_mb": _int(kv.get("AllocMem")), "free_mem_mb": _int(kv.get("FreeMem")),
            "state": state, "up": not any(w in state for w in _DOWN_WORDS),
            "partitions": [p for p in (kv.get("Partitions") or "").split(",") if p],
            "features": [f for f in (kv.get("AvailableFeatures") or "").split(",") if f and f != "(null)"],
            "gres": None if kv.get("Gres") in (None, "(null)") else kv.get("Gres")}


def parse_scontrol_nodes(text) -> list:
    nodes = []
    for line in (text or "").splitlines():
        if "NodeName=" not in line:
            continue
        n = _node_from_kv(parse_kv_line(line))
        if n and n["cpus"] and n["real_mem_mb"]:
            nodes.append(n)
    return nodes


def _time_to_seconds(text):
    text = (text or "").strip()
    if not text or text.upper() in ("UNLIMITED", "INFINITE", "NONE", "(NULL)", "N/A"):
        return None
    m = re.match(r"^(?:(\d+)-)?(\d+)(?::(\d+))?(?::(\d+))?$", text)
    if not m:
        return None
    d, a, b, c = m.groups()
    d = int(d or 0)
    if "-" in text:                                           # D-HH[:MM[:SS]]
        return d * 86400 + int(a) * 3600 + int(b or 0) * 60 + int(c or 0)
    if c is not None:
        return int(a) * 3600 + int(b) * 60 + int(c)
    if b is not None:
        return int(a) * 60 + int(b)
    return int(a) * 60


def parse_scontrol_partitions(text) -> list:
    parts = []
    for line in (text or "").splitlines():
        if "PartitionName=" not in line:
            continue
        kv = parse_kv_line(line)
        mm = kv.get("MaxMemPerNode", "")
        parts.append({"name": kv["PartitionName"], "default": kv.get("Default", "").upper() == "YES", "state": kv.get("State"),
                      "max_time_s": _time_to_seconds(kv.get("MaxTime")), "max_mem_per_node_mb": _int(mm) if mm.isdigit() else None,
                      "max_cpus_per_node": _int(kv.get("MaxCPUsPerNode")) if (kv.get("MaxCPUsPerNode") or "").isdigit() else None,
                      "nodes": expand_hostlist(kv.get("Nodes", "")), "total_nodes": _int(kv.get("TotalNodes"))})
    return parts


def parse_scontrol_config(text) -> dict:
    out = {}
    for line in (text or "").splitlines():
        k, sep, v = line.partition("=")
        if sep and k.strip():
            out[k.strip()] = v.strip()
    return out


def parse_sinfo_nodes(text):
    """Fallback: ``sinfo -h -N -o '%N|%c|%m|%t|%P|%l|%f|%e|%C'`` -> (nodes, partitions)."""
    nodes, parts = {}, {}
    for line in (text or "").splitlines():
        f = line.strip().split("|")
        if len(f) < 6:
            continue
        name, cpus, mem, state, part, tl = f[:6]
        cpus_n, mem_n = _int(cpus), _int(mem)
        if not cpus_n or not mem_n:
            continue
        default = part.endswith("*")
        part = part.rstrip("*")
        state_u = state.upper()
        feats = [x for x in (f[6].split(",") if len(f) > 6 else []) if x and x != "(null)"]
        n = nodes.setdefault(name, {"name": name, "cpus": cpus_n, "cpus_alloc": None, "cpu_load": None, "real_mem_mb": mem_n, "mem_spec_mb": 0,
                                    "alloc_mem_mb": None, "free_mem_mb": _int(f[7]) if len(f) > 7 else None, "state": state_u,
                                    "up": not any(w in state_u for w in _DOWN_WORDS), "partitions": [], "features": feats, "gres": None})
        n["partitions"].append(part)
        parts.setdefault(part, {"name": part, "default": default, "state": None, "max_time_s": _time_to_seconds(tl),
                                "max_mem_per_node_mb": None, "max_cpus_per_node": None, "nodes": [], "total_nodes": None})["nodes"].append(name)
    for p in parts.values():
        p["total_nodes"] = len(p["nodes"])
    return list(nodes.values()), list(parts.values())


def slurm_limits_of(nodes, partitions, config=None, partition=None) -> dict:
    """What one job can ask for on this cluster: the largest node of the partition the jobs go to (the default one, or all when there is no
    default), ``MaxMemPerNode`` and the partition time limit. ``nodes_fit`` lists the (cpus, mem_gb) of every node so that "a node with 40
    CPUs and 300 GB" can be asked for. A node that is down still counts (it is a question of "can it ever run")."""
    pick = None
    if partition:
        pick = next((p for p in partitions if p["name"] == partition), None)
    if pick is None:
        pick = next((p for p in partitions if p["default"]), None)
    names = set(pick["nodes"]) if pick and pick["nodes"] else None
    sel = [n for n in nodes if names is None or n["name"] in names] or list(nodes)
    cap_mb = None
    cands = [pick.get("max_mem_per_node_mb")] if pick else []
    cfg_max = (config or {}).get("MaxMemPerNode", "")
    if cfg_max.isdigit():
        cands.append(int(cfg_max))
    cands = [c for c in cands if c]
    if cands:
        cap_mb = min(cands)
    fit = []
    for n in sel:
        mem = n["real_mem_mb"] - (n.get("mem_spec_mb") or 0)
        if cap_mb:
            mem = min(mem, cap_mb)
        fit.append((n["cpus"], int(mem // 1024)))
    out = {"partition": pick["name"] if pick else None, "nodes_in_scope": len(sel)}
    if fit:
        out["max_cpus"] = max(c for c, _ in fit)
        out["max_mem_gb"] = max(m for _, m in fit)
        out["nodes_fit"] = sorted(set(fit), reverse=True)
    if pick and pick.get("max_time_s"):
        s = pick["max_time_s"]
        out["max_time"] = "%d-%02d:%02d:%02d" % (s // 86400, s % 86400 // 3600, s % 3600 // 60, s % 60)
        out["max_time_s"] = s
    return out


def probe_slurm(host: Host, problems=None, deadline=None) -> dict:
    """The SLURM cluster this machine submits to, or ``{'present': False, 'reason': ...}``. Never raises, never waits more than the deadline."""
    problems = problems if problems is not None else []
    t0 = host.clock()
    left = lambda: max(0.5, (deadline or DEADLINE) - (host.clock() - t0))     # noqa: E731
    has_sinfo, has_scontrol = bool(host.which("sinfo")), bool(host.which("scontrol"))
    if not (has_sinfo or has_scontrol):
        return {"present": False, "reason": "no scontrol or sinfo on this machine"}
    out = {"present": True, "version": None, "cluster": None, "source": None, "nodes": [], "partitions": [], "plugins": {},
           "memory_enforced": None, "limits": {}, "messages": []}
    hung = False
    if has_scontrol:
        r = host.run(["scontrol", "show", "config"], min(CMD_TIMEOUT, left()))
        if r.status == "timeout":
            hung = True
            out["messages"].append("scontrol did not answer in %.0f s: SLURM looks unresponsive" % CMD_TIMEOUT)
        elif r.ok:
            cfg = parse_scontrol_config(r.out)
            out["version"] = cfg.get("SLURM_VERSION")
            out["cluster"] = cfg.get("ClusterName")
            out["plugins"] = {k: cfg.get(k) for k in ("SelectType", "SelectTypeParameters", "TaskPlugin", "ProctrackType", "JobAcctGatherType",
                                                       "JobAcctGatherParams", "DefMemPerCPU", "MaxMemPerNode") if cfg.get(k) is not None}
            task = (cfg.get("TaskPlugin") or "") + " " + (cfg.get("JobAcctGatherParams") or "")
            out["memory_enforced"] = bool("cgroup" in task.lower() or "overmemorykill" in task.lower())
            out["_config"] = cfg
        else:
            out["messages"].append("scontrol show config: %s%s" % (r.status, (" " + r.err) if r.err else ""))
        if not hung:
            rn = host.run(["scontrol", "show", "nodes", "-o"], min(CMD_TIMEOUT, left()))
            if rn.status == "timeout":
                hung = True
                out["messages"].append("scontrol show nodes timed out")
            elif rn.ok:
                out["nodes"] = parse_scontrol_nodes(rn.out)
                rp = host.run(["scontrol", "show", "partition", "-o"], min(CMD_TIMEOUT, left()))
                if rp.ok:
                    out["partitions"] = parse_scontrol_partitions(rp.out)
                out["source"] = "scontrol"
            else:
                out["messages"].append("scontrol show nodes: %s%s" % (rn.status, (" " + rn.err) if rn.err else ""))
    if not out["nodes"] and has_sinfo and not hung:
        r = host.run(["sinfo", "-h", "-N", "-o", "%N|%c|%m|%t|%P|%l|%f|%e"], min(CMD_TIMEOUT, left()))
        if r.ok:
            out["nodes"], out["partitions"] = parse_sinfo_nodes(r.out)
            out["source"] = "sinfo"
        else:
            out["messages"].append("sinfo: %s%s" % (r.status, (" " + r.err) if r.err else ""))
    if not out["nodes"]:
        out["messages"].append("no node could be read: limits of the cluster are unknown")
        problems.append("SLURM is installed but no node could be read (%s)" % "; ".join(out["messages"]))
        out["usable"] = False
    else:
        out["usable"] = True
        out["limits"] = slurm_limits_of(out["nodes"], out["partitions"], out.get("_config"))
        n_up = sum(1 for n in out["nodes"] if n["up"])
        out["nodes_up"] = n_up
        out["totals"] = {"nodes": len(out["nodes"]), "cpus": sum(n["cpus"] for n in out["nodes"]),
                         "cpus_alloc": sum(n["cpus_alloc"] or 0 for n in out["nodes"]) if all(n["cpus_alloc"] is not None for n in out["nodes"]) else None,
                         "real_mem_gb": round(sum(n["real_mem_mb"] for n in out["nodes"]) / 1024.0, 1),
                         "alloc_mem_gb": round(sum(n["alloc_mem_mb"] or 0 for n in out["nodes"]) / 1024.0, 1)
                         if all(n["alloc_mem_mb"] is not None for n in out["nodes"]) else None}
        if n_up < len(out["nodes"]):
            out["messages"].append("%d of %d nodes are not up (down, drained or unreachable)" % (len(out["nodes"]) - n_up, len(out["nodes"])))
    out.pop("_config", None)
    return out


# ------------------------------------------------------------------------------------------------------------------ features
def ferre_is_openmp(path) -> bool | None:
    """True when ``ferre.x`` links an OpenMP runtime (``libgomp`` / ``libomp`` / ``libiomp`` in its dynamic section or its symbols),
    False when it does not, None when the file cannot be read."""
    try:
        p = os.path.expanduser(os.path.expandvars(str(path)))
        if not os.path.isfile(p):
            return None
        with open(p, "rb") as fh:
            blob = fh.read(64 * 1024 * 1024)
    except (OSError, ValueError, TypeError):
        return None
    return any(tag in blob for tag in (b"libgomp", b"libiomp", b"libomp.so", b"GOMP_parallel", b"__kmpc_fork_call"))


# -------------------------------------------------------------------------------------------------------------------- probe()
_CACHE: dict = {}


def clear_cache():
    _CACHE.clear()


def classify(os_info, local, slurm) -> str:
    """A label, heuristic: slurm | container | laptop | workstation."""
    if slurm and slurm.get("present") and slurm.get("usable"):
        return "slurm"
    if os_info.get("container") or local.get("cgroup_version") and local["cpus"].get("cgroup_quota"):
        return "container"
    if os_info.get("battery") or os_info.get("wsl"):
        return "laptop"
    return "workstation"


def probe(host: Host | None = None, want_slurm=True, ferre_exe=None, ttl=None, deadline=None, use_cache=True) -> dict:
    """The whole picture: ``{os, kind, local, slurm, features, problems, probed_at, seconds}``. Cached for ``ttl`` seconds in this process."""
    real = host is None
    host = host or Host()
    ttl = CACHE_TTL if ttl is None else ttl
    try:
        ttl = float(host.environ.get("PYAPS_PROBE_TTL", ttl))
    except (TypeError, ValueError):
        pass
    key = (want_slurm, str(ferre_exe))
    now = time.time()
    if real and use_cache and key in _CACHE and now - _CACHE[key][0] < ttl:
        return _CACHE[key][1]
    t0 = host.clock()
    problems: list = []
    try:
        os_info = probe_os(host)
    except Exception as exc:                                  # a probe must never stop a run
        os_info, _ = {"system": host.platform_name, "hostname": host.hostname()}, problems.append("os probe failed: %s" % exc)
    try:
        local = probe_local(host, problems)
    except Exception as exc:
        local, _ = {"cpus": {}, "memory": {}, "load": None}, problems.append("local probe failed: %s" % exc)
    slurm = {"present": False, "reason": "not asked"}
    if want_slurm:
        try:
            slurm = probe_slurm(host, problems, deadline)
        except Exception as exc:
            slurm = {"present": False, "reason": "probe failed: %s" % exc}
            problems.append("SLURM probe failed: %s" % exc)
    features = {"ferre_openmp": None, "ferre_exe": None}
    if ferre_exe:
        features["ferre_exe"] = str(ferre_exe)
        try:
            features["ferre_openmp"] = ferre_is_openmp(ferre_exe)
        except Exception:
            pass
        if features["ferre_openmp"] is None:
            problems.append("ferre.x not readable (%s): OpenMP build unknown" % ferre_exe)
    result = {"schema": 1, "module_version": __version__, "probed_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "os": os_info, "local": local,
              "slurm": slurm, "features": features, "problems": problems, "kind": classify(os_info, local, slurm)}
    result["seconds"] = round(host.clock() - t0, 2)
    if real and use_cache:
        _CACHE[key] = (now, result)
    return result


def format_probe(p: dict) -> str:
    """Operator text of a probe result."""
    L = []
    o, loc, sl = p.get("os", {}), p.get("local", {}), p.get("slurm", {})
    L.append("host       : %s  (%s%s%s)  kind=%s" % (o.get("hostname") or "?", o.get("system"), ", WSL" if o.get("wsl") else "",
                                                  ", container" if o.get("container") else "", p.get("kind")))
    c, m = loc.get("cpus", {}), loc.get("memory", {})
    L.append("this machine: %s usable CPUs (logical %s, affinity %s, cgroup quota %s)%s" % (
        c.get("usable", "?"), c.get("logical"), c.get("affinity"), c.get("cgroup_quota"), ("  limited by " + ", ".join(c["limited_by"])) if c.get("limited_by") else ""))
    L.append("             RAM total %s GB, available %s GB, usable %s GB (cgroup limit %s), swap %s/%s GB, load %s%s" % (
        m.get("total_gb"), m.get("available_gb"), m.get("usable_available_gb"), m.get("cgroup_limit_gb"), m.get("swap_free_gb"), m.get("swap_total_gb"),
        loc.get("load"), ("  limited by " + ", ".join(m["limited_by"])) if m.get("limited_by") else ""))
    if sl.get("present"):
        L.append("SLURM       : %s  cluster=%s  nodes=%s  source=%s%s" % (sl.get("version"), sl.get("cluster"), len(sl.get("nodes", [])), sl.get("source"),
                                                                      "" if sl.get("usable") else "  (UNUSABLE: " + "; ".join(sl.get("messages", [])) + ")"))
        pl = sl.get("plugins", {})
        L.append("             select=%s %s  task=%s  acct_gather=%s %s  memory enforced=%s" % (
            pl.get("SelectType"), pl.get("SelectTypeParameters"), pl.get("TaskPlugin"), pl.get("JobAcctGatherType"), pl.get("JobAcctGatherParams"), sl.get("memory_enforced")))
        for pt in sl.get("partitions", []):
            ms = pt.get("max_time_s")
            L.append("  partition %-12s%s state=%s MaxTime=%s MaxMemPerNode=%s nodes=%d" % (
                pt["name"], "*" if pt["default"] else " ", pt.get("state"),
                "%d-%02d:%02d:%02d" % (ms // 86400, ms % 86400 // 3600, ms % 3600 // 60, ms % 60) if ms else "unlimited",
                pt.get("max_mem_per_node_mb") or "unlimited", len(pt.get("nodes", []))))
        for n in sl.get("nodes", []):
            L.append("  node %-14s cpus %3s (alloc %s)  RealMemory %s MB (alloc %s, free %s)  %s %s" % (
                n["name"], n["cpus"], n["cpus_alloc"], n["real_mem_mb"], n["alloc_mem_mb"], n["free_mem_mb"], n["state"], ",".join(n["features"])))
        lim = sl.get("limits", {})
        if lim:
            L.append("  one job can ask at most: %s CPUs, %s GB, %s  (partition %s)" % (lim.get("max_cpus"), lim.get("max_mem_gb"), lim.get("max_time"), lim.get("partition")))
        for msg in sl.get("messages", []):
            L.append("  note: " + msg)
    else:
        L.append("SLURM       : not present (%s)" % sl.get("reason"))
    f = p.get("features", {})
    L.append("ferre.x     : %s" % ("OpenMP build" if f.get("ferre_openmp") else ("single-thread build" if f.get("ferre_openmp") is False else "unknown (no ferre.x given or readable)")))
    for pr in p.get("problems", []):
        L.append("problem     : " + pr)
    L.append("probe took %s s" % p.get("seconds"))
    return "\n".join(L)


if __name__ == "__main__":                                   # python -m PyAPS.aps_resource_probe
    import json
    exe = sys.argv[sys.argv.index("--ferre-exe") + 1] if "--ferre-exe" in sys.argv[:-1] else None
    res = probe(use_cache=False, ferre_exe=exe)
    print(json.dumps(res, indent=1, default=str) if "--json" in sys.argv else format_probe(res))
