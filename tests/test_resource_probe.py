"""The machine probe: SLURM clusters (scontrol / sinfo), this machine (affinity, cgroups, procfs meminfo), failure modes.

Standard library only; every machine is a fake (tests/resource_fakes.py). Run: ``cd <PYAPS_DIR> && pytest tests/test_resource_probe.py -v``
"""
from __future__ import annotations

import time

import pytest

import resource_fakes as F

rp = pytest.importorskip("PyAPS.aps_resource_probe")


# ------------------------------------------------------------------------------------------------------------------ parsers
def test_hostlist_expansion():
    assert rp.expand_hostlist("nodeA,nodeB") == ["nodeA", "nodeB"]
    assert rp.expand_hostlist("weave-master,weave-node[1-2]") == ["weave-master", "weave-node1", "weave-node2"]
    assert rp.expand_hostlist("n[01-03,07]") == ["n01", "n02", "n03", "n07"]
    assert rp.expand_hostlist("(null)") == [] and rp.expand_hostlist("") == []


def test_node_line_with_spaces_in_a_value_is_parsed():
    nodes = rp.parse_scontrol_nodes(F.scontrol_nodes(F.CLUSTER_SMALL))
    n = {x["name"]: x for x in nodes}
    assert n["nodeA"]["cpus"] == 90 and n["nodeA"]["real_mem_mb"] == 360000 and n["nodeA"]["alloc_mem_mb"] == 337920
    assert n["nodeA"]["state"] == "ALLOCATED" and n["nodeA"]["up"] and n["ctl"]["cpus_alloc"] == 30


def test_garbage_output_gives_no_nodes_and_no_exception():
    assert rp.parse_scontrol_nodes("this is not scontrol\nNodeName=\nfoo=bar") == []
    assert rp.parse_scontrol_partitions("garbage") == []
    assert rp.parse_scontrol_config("no equals here") == {}
    assert rp.parse_sinfo_nodes("a|b|c\n||\n")[0] == []


# ----------------------------------------------------------------------------------------------------------- SLURM clusters
def test_big_cluster_cluster(tmp_path):
    h = F.host(tmp_path, F.slurm_runner(F.CLUSTER_BIG), cpus=100)
    p = rp.probe(h, use_cache=False)
    sl = p["slurm"]
    assert sl["present"] and sl["usable"] and sl["version"] == "23.11.4" and sl["cluster"] == "cluster" and sl["source"] == "scontrol"
    assert sl["limits"]["max_cpus"] == 90 and sl["limits"]["max_mem_gb"] == 703 and sl["limits"]["max_time"] == "2-00:00:00"
    assert sl["limits"]["partition"] == "main" and (90, 703) in sl["limits"]["nodes_fit"] and (50, 419) in sl["limits"]["nodes_fit"]
    assert sl["memory_enforced"] is False                     # TaskPlugin (null): the memory request is a scheduling token
    assert p["kind"] == "slurm" and p["problems"] == []


def test_small_cluster_cluster_largest_node_is_351_gb(tmp_path):
    p = rp.probe(F.host(tmp_path, F.slurm_runner(F.CLUSTER_SMALL)), use_cache=False)
    assert p["slurm"]["limits"]["max_mem_gb"] == 351 and p["slurm"]["limits"]["max_cpus"] == 90
    assert p["slurm"]["totals"]["nodes"] == 2 and p["slurm"]["totals"]["cpus_alloc"] == 120


def test_cgroup_task_plugin_means_memory_is_enforced(tmp_path):
    p = rp.probe(F.host(tmp_path, F.slurm_runner(F.CLUSTER_SMALL, task_plugin="task/cgroup")), use_cache=False)
    assert p["slurm"]["memory_enforced"] is True


def test_sinfo_is_the_fallback_when_scontrol_is_missing(tmp_path):
    p = rp.probe(F.host(tmp_path, F.slurm_runner(F.CLUSTER_SMALL, scontrol=False)), use_cache=False)
    sl = p["slurm"]
    assert sl["usable"] and sl["source"] == "sinfo" and sl["limits"]["max_mem_gb"] == 351 and sl["limits"]["max_time"] == "2-00:00:00"


def test_a_down_node_still_counts_for_can_it_ever_run_and_is_reported(tmp_path):
    nodes = F.CLUSTER_SMALL[:1] + [("nodeA", 90, 360000, 0, 0, "DOWN*")]
    sl = rp.probe(F.host(tmp_path, F.slurm_runner(nodes)), use_cache=False)["slurm"]
    assert sl["limits"]["max_mem_gb"] == 351 and sl["nodes_up"] == 1
    assert any("not up" in m for m in sl["messages"])


def test_scontrol_that_hangs_is_given_up_after_one_try(tmp_path):
    runner = F.Runner({("scontrol",): rp.CmdResult("timeout"), ("sinfo",): F.ok(F.sinfo_nodes(F.CLUSTER_SMALL))})
    t0 = time.time()
    p = rp.probe(F.host(tmp_path, runner), use_cache=False)
    assert time.time() - t0 < 2
    sl = p["slurm"]
    assert sl["present"] and not sl["usable"] and any("unresponsive" in m for m in sl["messages"])
    assert len([c for c in runner.calls if c[0] == "scontrol"]) == 1                # no second scontrol, no sinfo behind a hung controller
    assert p["problems"] and p["kind"] != "slurm"


def test_garbage_from_every_command_degrades_to_unknown(tmp_path):
    runner = F.Runner({("scontrol",): F.ok("<html>502</html>"), ("sinfo",): F.ok("not|a|table\n")})
    p = rp.probe(F.host(tmp_path, runner), use_cache=False)
    assert p["slurm"]["usable"] is False and p["slurm"]["limits"] == {}
    assert p["local"]["cpus"]["usable"] == 8                                       # the machine itself is still known


def test_scontrol_error_exit_is_reported_not_raised(tmp_path):
    runner = F.Runner({("scontrol",): rp.CmdResult("error", 1, "", "slurm_load_ctl_conf error: Unable to contact slurm controller"),
                       ("sinfo",): rp.CmdResult("error", 1, "", "unable to contact")})
    sl = rp.probe(F.host(tmp_path, runner), use_cache=False)["slurm"]
    assert not sl["usable"] and any("Unable to contact" in m for m in sl["messages"])


def test_no_scheduler_at_all(tmp_path):
    p = rp.probe(F.host(tmp_path, F.Runner(), which_slurm=False), use_cache=False)
    assert p["slurm"]["present"] is False and "no scontrol" in p["slurm"]["reason"]


# ------------------------------------------------------------------------------------------------------------ this machine
def test_laptop_without_slurm(tmp_path):
    p = rp.probe(F.host(tmp_path, cpus=8, mem_total_gb=16, mem_avail_gb=10, swap_gb=2, which_slurm=False, battery=True), use_cache=False)
    loc = p["local"]
    assert loc["cpus"]["usable"] == 8 and loc["memory"]["total_gb"] == 16.0 and loc["memory"]["available_gb"] == 10.0
    assert loc["memory"]["swap_total_gb"] == 2.0 and loc["load"] == [0.5, 0.4, 0.3]
    assert p["kind"] == "laptop" and p["os"]["system"] == "linux" and not p["os"]["wsl"]


def test_workstation_and_wsl_labels(tmp_path):
    assert rp.probe(F.host(tmp_path, which_slurm=False), use_cache=False)["kind"] == "workstation"
    assert rp.probe(F.host(tmp_path / "w", which_slurm=False, wsl=True), use_cache=False)["os"]["wsl"] is True


def test_cpu_affinity_below_the_core_count_wins(tmp_path):
    loc = rp.probe(F.host(tmp_path, cpus=64, affinity=4, which_slurm=False), use_cache=False)["local"]
    assert loc["cpus"]["usable"] == 4 and any("affinity" in w for w in loc["cpus"]["limited_by"])


def test_container_with_cgroup_v2_limits(tmp_path):
    cg = {"cpu_max": "200000 100000", "memory_max": 4 * 2 ** 30, "memory_current": 1 * 2 ** 30}
    p = rp.probe(F.host(tmp_path, cpus=64, mem_total_gb=256, mem_avail_gb=200, which_slurm=False, cgroup=cg, container=True), use_cache=False)
    loc = p["local"]
    assert loc["cpus"]["usable"] == 2 and loc["cpus"]["cgroup_quota"] == 2.0
    assert loc["memory"]["cgroup_limit_gb"] == 4.0 and loc["memory"]["usable_total_gb"] == 4.0 and loc["memory"]["usable_available_gb"] == 3.0
    assert loc["memory"]["total_gb"] == 256.0                                       # the host's, kept for the report
    assert p["kind"] == "container" and p["os"]["container"]


def test_container_cgroup_unlimited_values_are_ignored(tmp_path):
    p = rp.probe(F.host(tmp_path, cpus=8, which_slurm=False, cgroup={"cpu_max": "max 100000", "memory_max": "max"}), use_cache=False)
    assert p["local"]["cpus"]["usable"] == 8 and p["local"]["memory"]["cgroup_limit_gb"] is None


def test_cgroup_v1_quota_and_limit(tmp_path):
    cg = {"cfs_quota": 300000, "cfs_period": 100000, "memory_max": 8 * 2 ** 30, "memory_current": 2 * 2 ** 30}
    loc = rp.probe(F.host(tmp_path, cpus=32, mem_total_gb=128, mem_avail_gb=100, which_slurm=False, cgroup=cg, v1=True), use_cache=False)["local"]
    assert loc["cpus"]["usable"] == 3 and loc["memory"]["cgroup_limit_gb"] == 8.0 and loc["memory"]["usable_available_gb"] == 6.0


def test_cgroup_v1_without_a_limit_changes_nothing(tmp_path):
    loc = rp.probe(F.host(tmp_path, cpus=4, which_slurm=False, cgroup={"cfs_quota": -1}, v1=True), use_cache=False)["local"]
    assert loc["cpus"]["usable"] == 4 and loc["memory"]["cgroup_limit_gb"] is None


def test_missing_proc_is_unknown_not_a_crash(tmp_path):
    h = F.host(tmp_path, which_slurm=False)
    (tmp_path / "root" / "proc" / "meminfo").unlink()
    p = rp.probe(h, use_cache=False)
    assert p["local"]["memory"]["total_gb"] is None and any("RAM" in x for x in p["problems"])
    assert p["local"]["cpus"]["usable"] == 8


def test_macos_uses_sysctl_and_vm_stat(tmp_path):
    vm = ("Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free:      100000.\nPages active: 5.\nPages inactive:   50000.\n"
          "Pages speculative:  10000.\n")
    runner = F.Runner({("sysctl", "-n", "hw.memsize"): F.ok("17179869184\n"), ("sysctl", "-n", "hw.ncpu"): F.ok("10\n"), ("vm_stat",): F.ok(vm)})
    p = rp.probe(F.host(tmp_path, runner, cpus=10, affinity=None, platform_name="darwin", which_slurm=False), use_cache=False)
    # affinity is None on macOS: the CPU count of sysctl is used
    assert p["os"]["system"] == "macos" and p["local"]["cpus"]["usable"] == 10
    assert p["local"]["memory"]["total_gb"] == 16.0 and p["local"]["memory"]["available_gb"] == round(160000 * 16384 / 2 ** 30, 2)
    assert p["kind"] == "laptop"


# ----------------------------------------------------------------------------------------------------------------- ferre.x
def test_ferre_openmp_detection(tmp_path):
    assert rp.ferre_is_openmp(F.write_ferre(tmp_path / "omp.x", True)) is True
    assert rp.ferre_is_openmp(F.write_ferre(tmp_path / "single.x", False)) is False
    assert rp.ferre_is_openmp(str(tmp_path / "nothing.x")) is None


def test_probe_reports_the_ferre_build_and_a_missing_file(tmp_path):
    h = F.host(tmp_path, which_slurm=False)
    assert rp.probe(h, ferre_exe=F.write_ferre(tmp_path / "f.x", True), use_cache=False)["features"]["ferre_openmp"] is True
    p = rp.probe(h, ferre_exe=str(tmp_path / "absent"), use_cache=False)
    assert p["features"]["ferre_openmp"] is None and any("ferre.x not readable" in x for x in p["problems"])


# ----------------------------------------------------------------------------------------------------- never crashes, cache
def test_an_exception_inside_a_probe_becomes_a_problem(tmp_path):
    h = F.host(tmp_path, which_slurm=False)
    h.read = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    p = rp.probe(h, use_cache=False)
    assert p["problems"] and p["local"] is not None


def test_real_probe_of_this_machine_works():
    p = rp.probe(use_cache=False)
    assert p["local"]["cpus"]["usable"] is None or p["local"]["cpus"]["usable"] >= 1
    assert rp.format_probe(p)


def test_cache_returns_the_same_answer_within_the_ttl(monkeypatch):
    rp.clear_cache()
    a = rp.probe(ttl=60)
    b = rp.probe(ttl=60)
    assert a is b
    rp.clear_cache()
