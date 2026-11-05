"""Machine and site awareness of the SLURM resource table: sites, limits policies warn / clamp / error / off, laptops, containers, depth-first priority.

The machines are fakes (tests/resource_fakes.py); no SLURM and no network. Run: ``cd <PYAPS_DIR> && pytest tests/test_slurm_machine_checks.py -v``
"""
from __future__ import annotations

import json

import pytest

import resource_fakes as F

slres = pytest.importorskip("PyAPS.aps_slurm_resources")
rp = pytest.importorskip("PyAPS.aps_resource_probe")

CTX_LR_STACK_M = {"res": "LR", "stack_level": 1, "grid": "m"}
CTX_HR_STACK_M = {"res": "HR", "stack_level": 1, "grid": "m"}
CTX_LR_SINGLE_N = {"res": "LR", "stack_level": 0, "grid": "n"}


@pytest.fixture(scope="module")
def table():
    return slres.load_table()


def probe_of(tmp_path, nodes=None, ferre_openmp=None, **kw):
    runner = F.slurm_runner(nodes) if nodes else F.Runner()
    h = F.host(tmp_path, runner, which_slurm=bool(nodes), **kw)
    ferre = F.write_ferre(tmp_path / "ferre.x", ferre_openmp) if ferre_openmp is not None else None
    return rp.probe(h, ferre_exe=ferre, use_cache=False)


def R(table, family, ctx, machine, **kw):
    kw.setdefault("env", {})
    kw.setdefault("quiet", True)
    return slres.resolve(family, ctx, table=table, machine=machine, **kw)


def values(res):
    return {s: (r.cpus, r.mem_gb, r.time) for s, r in res.items()}


# ------------------------------------------------------------------------- production-style clusters are never blocked or changed
@pytest.mark.parametrize("nodes", [F.CLUSTER_BIG, F.CLUSTER_SMALL], ids=["big_cluster", "small_cluster"])
@pytest.mark.parametrize("profile", ["default", "v2", "v2_openmp"])
@pytest.mark.parametrize("family,ctx", [("MOS", CTX_LR_STACK_M), ("MOS", CTX_HR_STACK_M), ("MOS", CTX_LR_SINGLE_N), ("LIFU", {"res": "LR", "stack_level": 1, "grid": "n"}),
                                        ("LIFU", {"res": "HR", "stack_level": 1, "grid": "n"}), ("CS", CTX_HR_STACK_M)])
def test_an_ordinary_run_on_a_production_like_cluster_is_unchanged(tmp_path, table, nodes, profile, family, ctx):
    """With the default policy (warn) a request is never changed; and the only requests reported are those no node can ever run."""
    machine = probe_of(tmp_path, nodes)
    res = R(table, family, ctx, machine, profile=profile, hpc=3)
    legacy = slres.resolve(family, ctx, table=table, profile=profile, limits={}, env={}, quiet=True)      # no look at the cluster at all
    assert values(res) == values(legacy)
    for r in res.values():
        assert all(a["action"] in ("warn", "clamp") for a in r.adjustments)                         # nothing refused


def test_big_cluster_has_nothing_to_report_for_any_profile_and_class(tmp_path, table):
    machine = probe_of(tmp_path, F.CLUSTER_BIG)
    for profile in ("default", "v2", "v2_openmp"):
        for family, ctx in (("MOS", CTX_HR_STACK_M), ("LIFU", {"res": "HR", "stack_level": 1, "grid": "n"}), ("CS", CTX_HR_STACK_M)):
            res = R(table, family, ctx, machine, profile=profile, hpc=3)
            assert [a for r in res.values() for a in r.adjustments] == [], (profile, family)
            assert res.environment.policy in ("warn", "clamp")


# ---------------------------------------------------------------------------------- the requests that can never run (the old 380 / 240 GB)
def test_380_gb_ferre_on_a_351_gb_node_policy_warn_keeps_it_and_says_what_fits(tmp_path, table):
    machine = probe_of(tmp_path, F.CLUSTER_SMALL)
    res = R(table, "MOS", CTX_HR_STACK_M, machine, profile="default", hpc=3)
    fr = res["FR"]
    assert res.environment.policy == "warn" and fr.mem_gb == 380                                      # never silently changed
    assert fr.fits is False and fr.adjustments[0]["action"] == "warn" and fr.adjustments[0]["fit"] == 351
    assert "--slurm_set FR.mem_gb=351" in fr.warnings[0] and "--slurm_limits clamp" in fr.warnings[0]
    assert fr.requested["mem_gb"] == 380


def test_policy_clamp_lowers_only_what_does_not_fit_and_records_both(tmp_path, table):
    machine = probe_of(tmp_path, F.CLUSTER_SMALL)
    res = R(table, "MOS", CTX_HR_STACK_M, machine, profile="default", hpc=3, limits_policy="clamp")
    assert res["FR"].mem_gb == 351 and res["FR"].requested["mem_gb"] == 380 and res["FR"].adjustments[0]["action"] == "clamp"
    legacy = slres.resolve("MOS", CTX_HR_STACK_M, table=table, profile="default", limits={}, env={}, quiet=True)
    for st in res:
        if st != "FR":
            assert (res[st].cpus, res[st].mem_gb, res[st].time) == (legacy[st].cpus, legacy[st].mem_gb, legacy[st].time)   # what fits is untouched
    assert (res["FR"].cpus, res["FR"].time) == (legacy["FR"].cpus, legacy["FR"].time)


def test_policy_error_refuses_before_submitting_and_names_the_alternative(tmp_path, table):
    machine = probe_of(tmp_path, F.SMALL_NODE)                                                          # 117 GB node
    with pytest.raises(slres.ResourceLimitError) as exc:
        R(table, "LIFU", {"res": "LR", "stack_level": 1, "grid": "n"}, machine, profile="default", hpc=3, limits_policy="error")
    msg = str(exc.value)
    assert "IFU_ExGal" in msg and "240 GB" in msg and "--slurm_set IFU_ExGal.mem_gb=117" in msg and "--slurm_limits warn" in msg
    assert isinstance(exc.value, slres.ResourceTableError)                                               # the runner already exits on this class


def test_policy_error_does_not_refuse_a_request_that_fits(tmp_path, table):
    machine = probe_of(tmp_path, F.CLUSTER_BIG)
    res = R(table, "MOS", CTX_HR_STACK_M, machine, profile="v2", hpc=3, limits_policy="error")
    assert res["FR"].fits and values(res) == values(slres.resolve("MOS", CTX_HR_STACK_M, table=table, profile="v2", limits={}, env={}, quiet=True))


def test_policy_off_does_not_look_at_the_machine_at_all(tmp_path, table):
    runner = F.slurm_runner(F.CLUSTER_SMALL)
    h = F.host(tmp_path, runner)
    called = []
    orig = rp.probe
    rp.probe = lambda *a, **k: called.append(1) or orig(h, *a[1:], **k)
    try:
        res = slres.resolve("MOS", CTX_HR_STACK_M, table=table, profile="default", env={"PYAPS_SLURM_LIMITS": "off"}, quiet=True, hpc=3)
    finally:
        rp.probe = orig
    assert not called and res["FR"].mem_gb == 380 and res["FR"].adjustments == []


def test_operator_policy_order_option_env_script_params_profile(tmp_path, table):
    machine = probe_of(tmp_path, F.CLUSTER_SMALL)
    pol = lambda **kw: R(table, "MOS", CTX_HR_STACK_M, machine, profile="default", hpc=3, **kw).environment    # noqa: E731
    assert pol().policy == "warn" and pol().policy_source == "profile default"
    assert pol(conf={"slurm_limits": "clamp"}).policy == "clamp"
    assert pol(conf={"slurm_limits": "clamp"}, env={"PYAPS_SLURM_LIMITS": "warn"}).policy == "warn"
    assert pol(conf={"slurm_limits": "clamp"}, env={"PYAPS_SLURM_LIMITS": "warn"}, limits_policy="off").policy == "off"
    with pytest.raises(slres.ResourceTableError):
        pol(limits_policy="strict")


def test_v2_keeps_its_clamp_policy_on_a_cluster_and_lowers_nothing_that_fits(tmp_path, table):
    machine = probe_of(tmp_path, F.CLUSTER_SMALL)
    res = R(table, "MOS", CTX_HR_STACK_M, machine, profile="v2", hpc=3)
    assert res.environment.policy == "clamp" and all(r.adjustments == [] for r in res.values())


def test_unknown_cluster_means_no_check_and_a_message(tmp_path, table):
    runner = F.Runner({("scontrol",): rp.CmdResult("timeout")})
    machine = rp.probe(F.host(tmp_path, runner), use_cache=False)
    res = R(table, "MOS", CTX_HR_STACK_M, machine, profile="default", hpc=3)
    assert res["FR"].mem_gb == 380 and res["FR"].adjustments == [] and any("unknown" in m for m in res.environment.messages)


# ------------------------------------------------------------------------------------------------------------- a laptop, no SLURM
def laptop(tmp_path, cpus=8, total=16, avail=10, **kw):
    return probe_of(tmp_path, None, cpus=cpus, mem_total_gb=total, mem_avail_gb=avail, battery=True, **kw)


def test_laptop_is_detected_as_the_local_site_and_clamps_by_default(tmp_path, table):
    machine = laptop(tmp_path)
    res = R(table, "MOS", CTX_LR_SINGLE_N, machine, profile="default", hpc=1)
    env = res.environment
    assert env.target == "local" and env.site["name"] == "local" and env.policy == "clamp" and env.policy_source == "site local"
    assert env.budget["cpus"] == 6 and env.budget["mem_gb"] == 8.0                                      # 75 % of 8 CPUs, 80 % of 10 GB
    assert all(r.cpus <= 6 for r in res.values())
    assert res["RR"].cpus == 6 and res["RR"].requested["cpus"] == 8                                     # asked 8, granted 6, both recorded
    assert res["L2merge"].cpus == 1 and res["L2merge"].adjustments == []                                # fits: untouched


def test_laptop_ram_hungry_stage_falls_back_to_fewer_workers(tmp_path, table):
    machine = laptop(tmp_path, avail=10)                                                                # budget 8 GB
    res = R(table, "MOS", CTX_LR_STACK_M, machine, profile="default", hpc=1)
    rvs = res["RVS"]
    assert rvs.cpus == 1 and rvs.requested["cpus"] == 12 and rvs.footprint_gb == 5.0                    # 1 GB + 4 GB per worker: only one fits in 8 GB
    assert "workers fit" in rvs.adjustments[-1]["reason"]
    big = R(table, "MOS", CTX_LR_STACK_M, laptop(tmp_path / "b", total=64, avail=48), profile="default", hpc=1)
    assert big["RVS"].cpus == 6 and big["RVS"].footprint_gb == 25.0                                     # plenty of RAM: the CPU share is the limit


def test_laptop_fine_grid_ferre_that_cannot_fit_is_reported_not_hidden(tmp_path, table):
    res = R(table, "MOS", CTX_LR_STACK_M, laptop(tmp_path), profile="default", hpc=1)
    fr = res["FR"]
    assert fr.fits is False and any(a["what"] == "mem" and a["fit"] is None for a in fr.adjustments)
    assert any("does not fit" in w or "even one worker" in w for w in fr.warnings)


def test_parallel_jobs_share_the_cpus_and_the_ram(tmp_path, table):
    machine = laptop(tmp_path, cpus=8, total=64, avail=40)
    res = R(table, "MOS", CTX_LR_SINGLE_N, machine, profile="default", hpc=1, parallel_jobs=3)
    assert res.environment.budget["cpus"] == 2 and all(r.cpus <= 2 for r in res.values())              # 8 x 0.75 / 3
    assert res.environment.budget["mem_gb"] == round(40 * 0.8 / 3, 2)
    assert R(table, "MOS", CTX_LR_SINGLE_N, machine, profile="default", hpc=1, env={"PYAPS_PARALLEL_JOBS": "2"}).environment.budget["cpus"] == 3


def test_laptop_warn_changes_nothing_but_says_so(tmp_path, table):
    res = R(table, "MOS", CTX_LR_SINGLE_N, laptop(tmp_path), profile="default", hpc=1, limits_policy="warn")
    assert [r.cpus for r in res.values()] == [8, 12, 10, 10, 20, 6, 1] and res["RR"].warnings and res["RR"].fits is False


def test_laptop_error_only_refuses_what_can_never_run(tmp_path, table):
    # 8 CPUs asked on 6: it can run (slowly): not refused. A stage that needs more RAM than the machine HAS: refused.
    R(table, "MOS", CTX_LR_SINGLE_N, laptop(tmp_path), profile="default", hpc=1, limits_policy="error")
    with pytest.raises(slres.ResourceLimitError) as exc:
        R(table, "MOS", CTX_LR_STACK_M, laptop(tmp_path / "x"), profile="default", hpc=1, limits_policy="error")      # fine-grid FERRE 55 GB on 16 GB
    assert "FR" in str(exc.value) and "--slurm_limits warn" in str(exc.value)


def test_laptop_profile_asks_little_to_begin_with(tmp_path, table):
    res = R(table, "MOS", CTX_LR_SINGLE_N, laptop(tmp_path, total=32, avail=24), profile="laptop", hpc=1)
    assert [r.cpus for r in res.values()] == [4, 4, 4, 4, 2, 1, 1] and all(r.adjustments == [] for r in res.values())


def test_profile_word_site_takes_the_profile_of_the_site(tmp_path, table):
    res = R(table, "MOS", CTX_LR_SINGLE_N, laptop(tmp_path), profile="site", hpc=1)
    assert next(iter(res.values())).profile == "laptop" and "= site" in res.profile_source


def test_bash_run_on_a_machine_that_also_has_slurm_uses_the_machine(tmp_path, table):
    machine = probe_of(tmp_path, F.CLUSTER_SMALL, cpus=32)
    res = R(table, "MOS", CTX_LR_SINGLE_N, machine, profile="default", hpc=1)
    assert res.environment.target == "local" and res.environment.site["name"] == "local"


# ----------------------------------------------------------------------------------------------------------------- containers
def test_container_with_cgroup_limits_gets_the_shares_of_its_quota(tmp_path, table):
    cg = {"cpu_max": "400000 100000", "memory_max": 8 * 2 ** 30, "memory_current": 2 * 2 ** 30}
    machine = probe_of(tmp_path, None, cpus=64, mem_total_gb=256, mem_avail_gb=200, cgroup=cg, container=True)
    res = R(table, "MOS", CTX_LR_SINGLE_N, machine, profile="default", hpc=1)
    assert res.environment.budget["usable_cpus"] == 4 and res.environment.budget["cpus"] == 3
    assert res.environment.budget["available_gb"] == 6.0 and res.environment.budget["mem_gb"] == 4.8
    assert max(r.cpus for r in res.values()) == 3 and res.environment.machine["kind"] == "container"


# ---------------------------------------------------------------------------------------------------- sites file and detection
SITES_FILE = """
sites:
  site_big:
    description: the big cluster
    match: {scheduler: slurm, any_node: {cpus: 90, real_memory_mb: 720000}}
    profile: v2
    ferre_openmp: false
    expect: {nodes: 3, node_cpus: 90, node_mem_gb: 703}
  site_small:
    description: the dev cluster
    match: {scheduler: slurm, any_node: {cpus: 90, real_memory_mb: 360000}}
    profile: v2
    profile_openmp: v2_openmp
    ferre_openmp: true
    expect: {nodes: 3, node_cpus: 90, node_mem_gb: 351}
  shared_host:
    match: {scheduler: none, hostname: 'shared[0-9]+'}
    limits: warn
    mem_fraction: 0.5
"""


def conf_with_sites(tmp_path):
    p = tmp_path / "slurm_sites.yaml"
    p.write_text(SITES_FILE)
    return {"slurm_sites_file": str(p)}


def test_two_clusters_with_the_same_hostname_are_told_apart_by_their_nodes(tmp_path, table):
    conf = conf_with_sites(tmp_path)
    big = R(table, "MOS", CTX_LR_SINGLE_N, probe_of(tmp_path / "a", F.CLUSTER_BIG), profile="site", hpc=3, conf=conf)
    small = R(table, "MOS", CTX_LR_SINGLE_N, probe_of(tmp_path / "b", F.CLUSTER_SMALL), profile="site", hpc=3, conf=conf)
    assert big.environment.site["name"] == "site_big" and small.environment.site["name"] == "site_small"
    assert next(iter(big.values())).profile == "v2"
    assert any("expects nodes=3" in m for m in small.environment.messages)                              # the fake dev cluster has 2 nodes, the site says 3


def test_ferre_build_picks_the_profile_of_an_unnamed_cluster(tmp_path, table):
    omp = R(table, "MOS", CTX_LR_SINGLE_N, probe_of(tmp_path / "a", F.CLUSTER_SMALL, ferre_openmp=True), profile="site", hpc=3)
    single = R(table, "MOS", CTX_LR_SINGLE_N, probe_of(tmp_path / "b", F.CLUSTER_SMALL, ferre_openmp=False), profile="site", hpc=3)
    assert omp.environment.site["name"] == "slurm" and next(iter(omp.values())).profile == "v2_openmp" and omp["FR"].cpus == 18
    assert next(iter(single.values())).profile == "v2" and single["FR"].cpus == 9


def test_sites_file_site_for_a_shared_host_applies_its_own_shares(tmp_path, table):
    machine = probe_of(tmp_path, None, hostname="shared07", cpus=32, mem_total_gb=128, mem_avail_gb=100)
    res = R(table, "MOS", CTX_LR_SINGLE_N, machine, profile="default", hpc=1, conf=conf_with_sites(tmp_path))
    assert res.environment.site["name"] == "shared_host" and res.environment.policy == "warn" and res.environment.budget["mem_gb"] == 50.0


def test_explicit_site_overrides_detection_env_and_option(tmp_path, table):
    conf = conf_with_sites(tmp_path)
    machine = probe_of(tmp_path, F.CLUSTER_BIG)
    assert R(table, "MOS", CTX_LR_SINGLE_N, machine, hpc=3, conf=conf, site="site_small").environment.site["source"].startswith("named by option")
    assert R(table, "MOS", CTX_LR_SINGLE_N, machine, hpc=3, conf=conf, env={"PYAPS_SLURM_SITE": "site_small"}).environment.site["name"] == "site_small"
    with pytest.raises(slres.ResourceTableError):
        R(table, "MOS", CTX_LR_SINGLE_N, machine, hpc=3, site="no_such_site")


def test_a_pinned_site_keeps_the_old_behaviour(tmp_path, table):
    t = json.loads(json.dumps({k: v for k, v in table.items() if not k.startswith("_")}))
    t["sites"]["pin"] = {"max_mem_gb": 340}
    res = R(t, "MOS", CTX_HR_STACK_M, probe_of(tmp_path, F.CLUSTER_BIG), profile="default", hpc=3, site="pin")
    assert res.environment.limits["max_mem_gb"] == 340 and res["FR"].mem_gb == 380 and res["FR"].adjustments[0]["fit"] == 340


def test_a_broken_sites_file_is_ignored_with_a_message(tmp_path, table):
    p = tmp_path / "bad.yaml"
    p.write_text("sites:\n  x:\n    match: {nonsense: 1}\n")
    res = R(table, "MOS", CTX_LR_SINGLE_N, laptop(tmp_path), hpc=1, conf={"slurm_sites_file": str(p)})
    assert any("ignored" in m for m in res.environment.messages) and res.environment.site["name"] == "local"


def test_the_table_validates_new_keys(table):
    bad = json.loads(json.dumps({k: v for k, v in table.items() if not k.startswith("_")}))
    bad["sites"]["local"]["mem_fraction"] = 1.5
    with pytest.raises(slres.ResourceTableError, match="share"):
        slres.validate_table(bad)
    bad = json.loads(json.dumps({k: v for k, v in table.items() if not k.startswith("_")}))
    bad["profiles"]["v2"]["priority"] = {"policy": "sideways"}
    with pytest.raises(slres.ResourceTableError, match="policy"):
        slres.validate_table(bad)
    bad = json.loads(json.dumps({k: v for k, v in table.items() if not k.startswith("_")}))
    bad["footprint"]["NOPE"] = {"base_gb": 1}
    with pytest.raises(slres.ResourceTableError, match="not a stage"):
        slres.validate_table(bad)


# --------------------------------------------------------------------------------------------------- effective JSON, header lines
def test_effective_summary_carries_the_machine_the_policy_and_both_values(tmp_path, table):
    machine = probe_of(tmp_path, F.CLUSTER_SMALL)
    res = R(table, "MOS", CTX_HR_STACK_M, machine, profile="default", hpc=3, limits_policy="clamp")
    s = slres.effective_summary(res, CTX_HR_STACK_M, table, "MOS", uapsid="u1")
    json.dumps(s)                                                                                       # serialisable
    assert s["environment"]["target"] == "slurm" and s["environment"]["policy"] == "clamp" and s["environment"]["machine"]["slurm"]["cluster"] == "cluster"
    assert s["environment"]["limits"]["max_mem_gb"] == 351
    fr = s["stages"]["FR"]
    assert fr["requested"]["mem_gb"] == 380 and fr["mem_gb"] == 351 and fr["adjustments"][0]["action"] == "clamp" and fr["limits_policy"] == "clamp"


def test_stage_log_header_says_asked_granted_and_the_place(tmp_path, table):
    res = R(table, "MOS", CTX_HR_STACK_M, probe_of(tmp_path, F.CLUSTER_SMALL), profile="default", hpc=3, limits_policy="clamp")
    text = "\n".join(slres.job_header_lines(res["FR"], res.environment))
    assert "PyAPS machine check: policy=clamp" in text and "asked cpus=36 mem=380 GB" in text and "granted cpus=36 mem=351 GB (1 value(s) lowered)" in text
    assert "target=slurm" in text


# ------------------------------------------------------------------------------------------------------------ depth-first priority
def test_priority_is_off_in_every_profile_unless_asked(tmp_path, table):
    machine = probe_of(tmp_path, F.CLUSTER_BIG)
    d = R(table, "MOS", CTX_LR_SINGLE_N, machine, profile="default", hpc=3, uapsid="abc")
    assert all(r.nice is None and r.comment is None for r in d.values())
    for prof in ("v2", "v2_openmp", "laptop"):
        assert all(r.nice is None for r in R(table, "MOS", CTX_LR_SINGLE_N, machine, profile=prof, hpc=3, uapsid="abc").values())
    v = R(table, "MOS", CTX_LR_SINGLE_N, machine, profile="v2", hpc=3, uapsid="abc", priority_policy="depth_first")
    assert [v[s].nice for s in slres.FAMILIES["MOS"]] == [7700, 6900, 6100, 6900, 6100, 5300, 4500]    # scheduled: 4500 + 800 x stages still to come
    assert v["RR"].comment == "pyaps:abc:p0:scheduled:RR"


def test_the_first_pair_and_the_last_stages_get_the_smallest_nice(tmp_path, table):
    machine = probe_of(tmp_path, F.CLUSTER_BIG)
    nice = lambda **kw: R(table, "MOS", CTX_LR_SINGLE_N, machine, profile="v2", hpc=3, priority_policy="depth_first", **kw)             # noqa: E731
    first, third, far = nice(pair_index=0), nice(pair_index=2), nice(pair_index=50)
    assert third["RR"].nice == first["RR"].nice + 200 and far["RR"].nice == first["RR"].nice + 800        # pair_step 100, capped at 8 pairs
    assert first["L2merge"].nice < first["LS"].nice < first["FR"].nice < first["RVS"].nice < first["RR"].nice
    assert max(r.nice for r in far.values()) == 8500 and min(r.nice for r in first.values()) >= 0       # bounded, never negative


def test_offline_class_is_always_ahead_of_scheduled(tmp_path, table):
    machine = probe_of(tmp_path, F.CLUSTER_BIG)
    off = R(table, "MOS", CTX_LR_SINGLE_N, machine, profile="v2", hpc=3, priority_policy="depth_first", priority_class="offline", pair_index=8)
    sch = R(table, "MOS", CTX_LR_SINGLE_N, machine, profile="v2", hpc=3, priority_policy="depth_first", priority_class="scheduled", pair_index=0)
    assert max(r.nice for r in off.values()) < min(r.nice for r in sch.values())
    assert all(r.nice == 0 for r in off.values())                       # no nice at all: a nice above the age of a job keeps it at the queue's floor of 1
    assert off["RR"].comment.endswith(":offline:RR") and off.priority["class_base"] == 0 and off.priority["pair_nice"] == 0


def test_priority_operator_order_and_off(tmp_path, table):
    machine = probe_of(tmp_path, F.CLUSTER_BIG)
    a = R(table, "MOS", CTX_LR_SINGLE_N, machine, profile="v2", hpc=3, priority_policy="off")
    b = R(table, "MOS", CTX_LR_SINGLE_N, machine, profile="v2", hpc=3, env={"PYAPS_SLURM_PRIORITY": "off", "PYAPS_SLURM_PRIORITY_X": "1"})
    c = R(table, "MOS", CTX_LR_SINGLE_N, machine, profile="default", hpc=3, priority_policy="depth_first", env={"PYAPS_PRIORITY_CLASS": "scheduled", "PYAPS_PAIR_INDEX": "3"})
    assert all(r.nice is None for r in a.values()) and all(r.nice is None for r in b.values())
    assert c["RR"].nice == 4500 + 3200 + 300 and c.priority["class_source"] == "PYAPS_PRIORITY_CLASS"


def test_escalation_retry_keeps_the_pair_index_of_the_first_generation(tmp_path, table):
    machine = probe_of(tmp_path, F.CLUSTER_BIG)
    esc = {"FR": {"mem_factor": 1.5, "time_factor": 1.0}}
    again = R(table, "MOS", CTX_LR_SINGLE_N, machine, profile="v2", hpc=3, priority_policy="depth_first", escalation=esc, pair_index=40, previous_priority={"pair_index": 1})
    fresh = R(table, "MOS", CTX_LR_SINGLE_N, machine, profile="v2", hpc=3, priority_policy="depth_first", pair_index=40, previous_priority={"pair_index": 1})
    assert again.priority["pair_index"] == 1 and "escalation" in again.priority["pair_source"] and fresh.priority["pair_index"] == 40


def test_no_nice_in_a_bash_run(tmp_path, table):
    res = R(table, "MOS", CTX_LR_SINGLE_N, laptop(tmp_path), profile="v2", hpc=1, priority_policy="depth_first")
    assert all(r.nice is None for r in res.values())


# ----------------------------------------------------------------------------------------------------------------------- the CLI
def test_cli_check_exit_codes_and_machine_file(tmp_path, capsys):
    machine = probe_of(tmp_path, F.CLUSTER_SMALL)
    mf = tmp_path / "m.json"
    mf.write_text(json.dumps(machine))
    assert slres.main(["check", "--machine-file", str(mf), "--target", "slurm", "--profile", "v2", "--res", "LR", "--stack", "stack"]) == 0
    out = capsys.readouterr().out
    assert "Everything fits." in out and "target=slurm" in out
    assert slres.main(["check", "--machine-file", str(mf), "--target", "slurm", "--profile", "default", "--res", "HR", "--stack", "stack"]) == 3
    out = capsys.readouterr().out
    assert "NEVER RUNS as asked" in out and "--slurm_set FR.mem_gb=351" in out


def test_cli_site_and_probe(tmp_path, capsys):
    assert slres.main(["site", "--profile", "laptop"]) == 0
    assert "place     : target=" in capsys.readouterr().out
    assert slres.main(["probe"]) == 0
    assert "this machine" in capsys.readouterr().out
    assert slres.main(["site", "--json", "--no-limits"]) == 0
    json.loads(capsys.readouterr().out)
    assert slres.main(["validate"]) == 0
