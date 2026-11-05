"""Fake machines for the tests of the machine probe and of the machine checks (no SLURM, no /proc of the test host is read).

Everything is built here: a fake root with ``proc/`` and ``sys/fs/cgroup`` (``make_root``), the text of ``scontrol``/``sinfo`` in the real format
(``scontrol_nodes`` ...), a command runner that answers from a table (``Runner``) and a ``Host`` that uses them. The numbers of ``CLUSTER_BIG`` and
``CLUSTER_SMALL`` are the node sizes read from the two production-style clusters (a big one whose FERRE is single-thread, a smaller dev-sized one whose FERRE is OpenMP).
"""
from __future__ import annotations

import os
from pathlib import Path

from PyAPS import aps_resource_probe as rp

# (name, cpus, RealMemory MB, CPUAlloc, AllocMem MB, state)
CLUSTER_BIG = [("ctl", 50, 429496, 0, 0, "IDLE"), ("nodeA", 90, 720000, 0, 0, "IDLE"), ("nodeB", 90, 720000, 0, 0, "IDLE")]
CLUSTER_SMALL = [("ctl", 32, 68719, 30, 30720, "MIXED"), ("nodeA", 90, 360000, 90, 337920, "ALLOCATED")]
SMALL_NODE = [("nodeA", 16, 120000, 0, 0, "IDLE")]                     # 117 GB, as the node IFU_ExGal's 240 GB can never fit

CONFIG = """ClusterName             = cluster
DefMemPerCPU            = 1024
JobAcctGatherType       = jobacct_gather/linux
JobAcctGatherParams     = (null)
MaxMemPerNode           = UNLIMITED
ProctrackType           = proctrack/linuxproc
SelectType              = select/cons_tres
SelectTypeParameters    = CR_CORE_MEMORY
SLURM_VERSION           = 23.11.4
TaskPlugin              = %s
"""


def scontrol_config(task_plugin="(null)"):
    return CONFIG % task_plugin


def scontrol_nodes(nodes, partition="main"):
    lines = []
    for name, cpus, mem, calloc, amem, state in nodes:
        lines.append("NodeName=%s Arch=x86_64 CoresPerSocket=%d  CPUAlloc=%d CPUEfctv=%d CPUTot=%d CPULoad=1.00 AvailableFeatures=(null) "
                     "ActiveFeatures=(null) Gres=(null) NodeAddr=%s NodeHostName=%s Version=23.11.4 OS=Linux 6.8.0-142-generic #142-Ubuntu SMP "
                     "PREEMPT_DYNAMIC Wed Sep  2 14:24:27 UTC 2026  RealMemory=%d AllocMem=%d FreeMem=%d Sockets=1 Boards=1 State=%s ThreadsPerCore=1 "
                     "TmpDisk=0 Weight=1 Owner=N/A MCS_label=N/A Partitions=%s  BootTime=2026-09-10T14:08:39 CfgTRES=cpu=%d,mem=%dM,billing=%d "
                     "AllocTRES=cpu=%d" % (name, cpus, calloc, cpus, cpus, name, name, mem, amem, mem - amem, state, partition, cpus, mem, cpus, calloc))
    return "\n".join(lines) + "\n"


def scontrol_partitions(names, partition="main", maxtime="2-00:00:00", default=True):
    return ("PartitionName=%s AllowGroups=ALL AllowAccounts=ALL AllowQos=ALL AllocNodes=ALL Default=%s QoS=N/A DefaultTime=2-00:00:00 "
            "MaxNodes=UNLIMITED MaxTime=%s MinNodes=0 LLN=NO MaxCPUsPerNode=UNLIMITED Nodes=%s PriorityJobFactor=1 PriorityTier=1 State=UP "
            "TotalCPUs=1 TotalNodes=%d DefMemPerNode=UNLIMITED MaxMemPerNode=UNLIMITED\n" % (partition, "YES" if default else "NO", maxtime, nodelist(names), len(names)))


def nodelist(names):
    """['nodeA', 'nodeB'] -> 'node[A-B]' is not needed; 'a,b' is valid SLURM."""
    return ",".join(names)


def sinfo_nodes(nodes, partition="main*", maxtime="2-00:00:00"):
    return "".join("%s|%d|%d|%s|%s|%s|(null)|%d\n" % (n[0], n[1], n[2], n[5].lower(), partition, maxtime, n[2] - n[4]) for n in nodes)


class Runner:
    """Answers commands from a table ``{argv-prefix-tuple: CmdResult | callable}``; unknown commands are 'missing'. Records the calls."""

    def __init__(self, table=None):
        self.table, self.calls = table or {}, []

    def __call__(self, argv, timeout=5.0):
        self.calls.append(tuple(argv))
        for n in range(len(argv), 0, -1):
            hit = self.table.get(tuple(argv[:n]))
            if hit is not None:
                return hit(argv) if callable(hit) else hit
        return rp.CmdResult("missing")


def ok(text):
    return rp.CmdResult("ok", 0, text)


def slurm_runner(nodes, task_plugin="(null)", partition="main", maxtime="2-00:00:00", scontrol=True, sinfo=True):
    t = {}
    if scontrol:
        t[("scontrol", "show", "config")] = ok(scontrol_config(task_plugin))
        t[("scontrol", "show", "nodes")] = ok(scontrol_nodes(nodes, partition))
        t[("scontrol", "show", "partition")] = ok(scontrol_partitions([n[0] for n in nodes], partition, maxtime))
    if sinfo:
        t[("sinfo",)] = ok(sinfo_nodes(nodes, partition + "*", maxtime))
    return Runner(t)


def make_root(tmp_path, cpus=8, mem_total_gb=16, mem_avail_gb=10, swap_gb=2, cgroup=None, v1=False, container=False, battery=False, wsl=False):
    """A fake ``/`` with proc/meminfo, proc/self/cgroup and the cgroup files of ``cgroup`` = {cpu_max, memory_max, memory_current, ...}."""
    root = Path(tmp_path) / "root"
    (root / "proc" / "self").mkdir(parents=True)
    kb = lambda gb: int(gb * 1024 * 1024)          # noqa: E731
    (root / "proc" / "meminfo").write_text("MemTotal: %d kB\nMemFree: %d kB\nMemAvailable: %d kB\nBuffers: 1 kB\nCached: 1 kB\nSwapTotal: %d kB\nSwapFree: %d kB\n"
                                           % (kb(mem_total_gb), kb(mem_avail_gb / 2), kb(mem_avail_gb), kb(swap_gb), kb(swap_gb)))
    (root / "proc" / "version").write_text("Linux version 6.8.0 (Microsoft WSL2)\n" if wsl else "Linux version 6.8.0\n")
    if container:
        (root / ".dockerenv").write_text("")
    if battery:
        (root / "sys" / "class" / "power_supply" / "BAT0").mkdir(parents=True)
    cg = root / "sys" / "fs" / "cgroup"
    if cgroup and not v1:
        cg.mkdir(parents=True)
        (cg / "cgroup.controllers").write_text("cpu memory\n")
        (root / "proc" / "self" / "cgroup").write_text("0::/\n")
        (cg / "cpu.max").write_text(cgroup.get("cpu_max", "max 100000") + "\n")
        (cg / "memory.max").write_text(str(cgroup.get("memory_max", "max")) + "\n")
        (cg / "memory.current").write_text(str(cgroup.get("memory_current", 0)) + "\n")
        (cg / "memory.stat").write_text("inactive_file %d\n" % cgroup.get("inactive_file", 0))
    elif cgroup and v1:
        (cg / "cpu,cpuacct").mkdir(parents=True)
        (cg / "memory").mkdir(parents=True)
        (root / "proc" / "self" / "cgroup").write_text("4:cpu,cpuacct:/\n3:memory:/\n")
        q, p = cgroup.get("cfs_quota", -1), cgroup.get("cfs_period", 100000)
        (cg / "cpu,cpuacct" / "cpu.cfs_quota_us").write_text("%d\n" % q)
        (cg / "cpu,cpuacct" / "cpu.cfs_period_us").write_text("%d\n" % p)
        (cg / "memory" / "memory.limit_in_bytes").write_text("%d\n" % cgroup.get("memory_max", 2 ** 63 - 4096))
        (cg / "memory" / "memory.usage_in_bytes").write_text("%d\n" % cgroup.get("memory_current", 0))
        (cg / "memory" / "memory.stat").write_text("total_inactive_file 0\n")
    else:
        (root / "proc" / "self" / "cgroup").write_text("0::/user.slice\n")
    return root


def host(tmp_path, runner=None, cpus=8, affinity=None, which_slurm=None, hostname="testhost", environ=None, platform_name="linux", **root_kw):
    """A Host over a fake root. ``which_slurm`` None = SLURM present when the runner knows scontrol/sinfo."""
    root = make_root(tmp_path, **root_kw)
    runner = runner or Runner()
    present = (lambda name: "/usr/bin/" + name if any(k[0] == name for k in runner.table) else None) if which_slurm is None else \
        (lambda name: "/usr/bin/" + name if which_slurm else None)
    return rp.Host(root=str(root), run=runner, environ=environ if environ is not None else {}, cpu_count=cpus, affinity=affinity if affinity is not None else cpus,
                   platform_name=platform_name, hostname=hostname, loadavg=(0.5, 0.4, 0.3), which=present)


def write_ferre(path, openmp):
    Path(path).write_bytes(b"\x7fELF....libgfortran.so.5\0libm.so.6\0" + (b"libgomp.so.1\0" if openmp else b"") + b"libc.so.6\0")
    return str(path)
