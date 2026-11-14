"""IFU_Gal / IFU_ExGal: the patches of a patch table run side by side inside the CPU budget of the job (aps_ifu_patch_pool).

Before 2026-11-13 both stages worked on one patch after another (OB 14699: 15 of 44 patches in 24 h). Now the budget `mp` is shared out over LANES (a lane
= one process taking patches from a queue, `capacity` inner workers per patch). These tests use fake patch work: the order, the budget arithmetic, the
isolation of a failing patch, the aggregation and the identity of the one-lane path with the old serial loop.
"""
import io
import json
import os
import time

import pytest

from PyAPS import aps_ifu_patch_pool as pool

# bins of the patches of OB 14699 (LIFU HR, 44 patches): 15 big ones, then single stars
OB14699 = [91, 58, 54, 52, 32, 37, 32, 27, 21, 22, 19, 16, 14, 11, 9] + [3] * 29


# ----------------------------------------------------------------------------------------------------------------------- estimates and order
def test_estimate_bins_floor_and_junk():
    assert pool.estimate_bins(22.0, 22.0) == pytest.approx(22 * 22 * pool.BINS_PER_ARCSEC2)
    assert pool.estimate_bins(1.0, 1.0) == pool.MIN_BINS                 # a single star is never free
    assert pool.estimate_bins(float('nan'), 5) == pool.MIN_BINS
    assert pool.estimate_bins(None, 5) == pool.MIN_BINS


def test_lpt_order_largest_first_ties_keep_table_order():
    assert pool.lpt_order([3, 9, 3, 5, 9]) == [1, 4, 3, 0, 2]
    assert pool.lpt_order([]) == []


def test_split_even():
    assert pool.split_even(10, 3) == [4, 3, 3]
    assert pool.split_even(8, 4) == [2, 2, 2, 2]
    assert sum(pool.split_even(10, 7)) == 10 and min(pool.split_even(10, 7)) == 1


# ----------------------------------------------------------------------------------------------------------------------- budget arithmetic
@pytest.mark.parametrize("costs", [[5], [5, 5], [91, 3, 3], OB14699, [2] * 200, [100] * 3])
@pytest.mark.parametrize("budget", [1, 2, 3, 4, 7, 8, 10, 20, 40])
@pytest.mark.parametrize("pw", [None, 0, 1, 2, 5, 64])
def test_plan_never_exceeds_the_budget(costs, budget, pw):
    plan = pool.plan_workers(costs, budget, pw)
    assert sum(plan.lanes) <= budget
    assert all(c >= 1 for c in plan.lanes)
    assert len(plan.lanes) <= len(costs)
    assert sorted(plan.order) == list(range(len(costs)))
    if len(plan.lanes) > 1:
        assert plan.order == pool.lpt_order(costs)
    if budget == 1 or pw == 1 or len(costs) == 1:
        assert plan.serial and plan.lanes == [budget] and plan.order == list(range(len(costs)))     # the old loop: table order, `mp` inner
    if pw and pw > 1 and budget > 1 and len(costs) > 1:
        assert len(plan.lanes) == min(pw, budget, len(costs))
        assert sum(plan.lanes) == budget


def test_plan_empty_and_junk_budget():
    p = pool.plan_workers([], 8)
    assert p.lanes == [] and p.order == [] and "no patch" in p.describe("X")
    assert pool.plan_workers([4, 4], 0).lanes == [1]                    # a budget of 0 is 1
    assert pool.plan_workers([4, 4], None).lanes == [1]
    assert pool.plan_workers([4, 4], 8, -3).lanes == pool.plan_workers([4, 4], 8, 0).lanes      # a negative count is auto


def test_plan_many_small_patches_use_one_lane_per_cpu():
    plan = pool.plan_workers([3] * 40, 10)
    assert plan.lanes == [1] * 10 and not plan.serial


def test_plan_few_big_patches_share_the_budget():
    plan = pool.plan_workers([6, 6, 6, 6], 10)                          # rounds of bins: 6 bins on 10 workers would leave 4 idle
    assert len(plan.lanes) == 4 and sum(plan.lanes) == 10
    plan = pool.plan_workers([80, 40], 10)                              # big patches scale inside: one lane, the budget in the patch
    assert plan.serial and plan.inner == 10
    plan = pool.plan_workers([30], 10)
    assert plan.serial and plan.inner == 10                             # one patch: all the budget goes inside it


# OB 26037 (night 20261112, ExGal): ONE patch with 606 Voronoi bins (1941 s on 24 workers) and ten single stars (84-153 s each, mostly the fixed work of reading and
# binning the aperture). Equal lanes slow the big patch by the same factor as the lanes are narrower, so the plan was one lane of 24 and 53 min.
OB26037 = [1313] + [2] * 10


def test_plan_one_giant_patch_and_small_ones_gets_a_wide_lane():
    for budget in (24, 40):
        plan = pool.plan_workers(OB26037, budget, None, max_lanes=7, min_inner=2)
        assert not plan.serial and plan.lanes[0] >= budget - 6 and sum(plan.lanes) == budget and min(plan.lanes) >= 2
        assert "wide lane" in plan.reason
        assert plan.est_wall < 0.85 * plan.est_serial                     # the estimated gain over one lane (20 % at 24 workers, 33 % at 40)
        assert pool.simulate(OB26037, plan.lanes, plan.order) == plan.est_wall


def test_plan_similar_patches_keep_equal_lanes():
    plan = pool.plan_workers([20] * 40, 24, None, max_lanes=7, min_inner=2)
    assert max(plan.lanes) - min(plan.lanes) <= 1 and "wide lane" not in plan.reason


def test_plan_wide_lane_respects_min_inner_and_the_lane_cap():
    plan = pool.plan_workers(OB26037, 24, None, max_lanes=2, min_inner=2)
    assert len(plan.lanes) <= 2
    plan = pool.plan_workers(OB26037, 5, None, min_inner=2)
    assert sum(plan.lanes) <= 5 and min(plan.lanes) >= 2 and len(plan.lanes) <= 2


def test_the_widest_lane_gets_the_largest_patch():
    """With lanes of different width a race for the queue could give the giant patch to the narrow lane."""
    def work(task, inner):
        time.sleep(0.05)
        return _ok(task, inner)
    costs = [2, 2, 1313] + [2] * 8
    tasks = [{"id": 10 + i, "cost": c} for i, c in enumerate(costs)]
    plan = pool.plan_workers(costs, 24, None, max_lanes=7, min_inner=2)
    assert plan.lanes[0] > plan.lanes[-1]
    for _ in range(5):                                                      # a race would show up in some of the runs
        res = pool.run_patches(tasks, work, plan)
        assert res[2]["inner"] == plan.lanes[0]


# ----------------------------------------------------------------------------------------------------------------------- dynamic widths (IFU_ExGal)
def test_fill_widths_giant_takes_what_the_small_ones_leave():
    w = pool.fill_widths({0: 3900, 1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2}, 40, 2)
    assert w[0] == 28 and all(w[i] == 2 for i in range(1, 7)) and sum(w.values()) == 40


def test_fill_widths_equal_demands_get_equal_widths_and_never_more_than_asked():
    w = pool.fill_widths({i: 51 for i in range(7)}, 40, 2)
    assert sum(w.values()) == 40 and max(w.values()) - min(w.values()) <= 1
    assert pool.fill_widths({0: 3, 1: 3}, 40, 2) == {0: 3, 1: 3}              # a patch of 3 bins never gets more than 3
    assert pool.fill_widths({0: 0, 1: 9}, 8, 2) == {1: 8}                     # a patch without work takes nothing


def test_dynamic_plan_has_lanes_from_memory_and_budget_not_from_costs():
    plan = pool.plan_workers(OB26037, 40, None, max_lanes=7, min_inner=2, dynamic=True)
    assert plan.dynamic and len(plan.lanes) == 7 and sum(plan.lanes) == 40 and "dynamic" in plan.reason and "bins are known" in plan.describe("X")
    assert pool.plan_workers([5], 40, None, min_inner=2, dynamic=True).serial                   # one patch: the whole budget inside it
    assert pool.plan_workers(OB26037, 40, 1, min_inner=2, dynamic=True).serial                  # patch_workers=1 stays the old serial code
    assert len(pool.plan_workers(OB26037, 40, 3, max_lanes=7, min_inner=2, dynamic=True).lanes) == 3
    assert len(pool.plan_workers(OB26037, 6, None, min_inner=2, dynamic=True).lanes) == 3       # 6 workers, 2 each


def _dyn_work(bins, hold=0.15, log=None):
    def work(task, inner):
        b = pool.current_broker()
        w = None
        try:
            if bins.get(task["id"], 0) > 0:
                w = b.acquire(task["pos"], bins[task["id"]])
            t0 = time.time()
            time.sleep(hold if (w or 0) > 2 else 0.02)
            return _ok(task, w, t0=t0, t1=time.time())
        finally:
            b.release(task["pos"])
    return work


def test_dynamic_widths_giant_gets_the_budget_and_the_total_never_exceeds_it():
    costs = [3900] + [2] * 8
    bins = {100 + i: c for i, c in enumerate(costs)}
    tasks = [{"id": 100 + i, "cost": c} for i, c in enumerate(costs)]
    plan = pool.plan_workers(costs, 12, 3, min_inner=2, dynamic=True)
    res = pool.run_patches(tasks, _dyn_work(bins), plan)
    assert res[0]["inner"] == 8 and all(r["inner"] == 2 for r in res[1:])                      # first wave 8 + 2 + 2, later patches 2
    events = sorted([(r["t0"], r["inner"]) for r in res] + [(r["t1"], -r["inner"]) for r in res], key=lambda e: (e[0], e[1]))
    busy = peak = 0
    for _, d in events:
        busy += d
        peak = max(peak, busy)
    assert peak <= 12


def test_dynamic_widths_a_skipped_patch_does_not_block_the_first_wave():
    costs = [50, 40, 30, 2, 2]
    bins = {1: 50, 2: 0, 3: 30, 4: 2, 5: 2}                                                   # patch 2 has no valid spaxels: it never acquires
    tasks = [{"id": i + 1, "cost": c} for i, c in enumerate(costs)]
    plan = pool.plan_workers(costs, 12, 3, min_inner=2, dynamic=True)
    t0 = time.time()
    res = pool.run_patches(tasks, _dyn_work(bins, hold=0.05), plan)
    assert time.time() - t0 < 20 and len(res) == 5 and res[1]["inner"] is None
    assert res[0]["inner"] + res[2]["inner"] <= 12 and res[0]["inner"] >= 2 and res[2]["inner"] >= 2


def test_dynamic_widths_a_lane_that_dies_gives_its_workers_back():
    costs = [20, 20, 20, 2, 2, 2, 2, 2]
    bins = {i + 1: c for i, c in enumerate(costs)}
    tasks = [{"id": i + 1, "cost": c} for i, c in enumerate(costs)]
    plan = pool.plan_workers(costs, 6, 3, min_inner=2, dynamic=True)

    def work(task, inner):
        b = pool.current_broker()
        w = b.acquire(task["pos"], bins[task["id"]])
        if task["id"] == 2:
            os._exit(3)                                                                         # the lane dies holding its workers
        time.sleep(0.05)
        b.release(task["pos"])
        return _ok(task, w)
    t0 = time.time()
    res = pool.run_patches(tasks, work, plan)
    assert time.time() - t0 < 60 and len(res) == 8
    assert res[1]["outcome"] != "written" and sum(1 for r in res if r["outcome"] == "written") == 7


def test_plan_real_field_beats_the_old_serial_order():
    plan = pool.plan_workers(OB14699, 10)
    assert not plan.serial and sum(plan.lanes) <= 10
    old = pool.simulate(OB14699, [10], list(range(len(OB14699))))      # one lane of 10, table order = the old code with mp 10
    assert plan.est_wall < old
    assert plan.est_serial == pytest.approx(old)
    one_cpu = pool.simulate(OB14699, [1])
    assert one_cpu / plan.est_wall > 7                                  # about 10 CPUs of work in a bit more than a tenth of the time
    assert "lane" in plan.describe("IFU_Gal") and "largest patch first" in plan.describe("IFU_Gal")


def test_lpt_beats_table_order_on_uneven_patches():
    costs = [3, 3, 3, 3, 3, 3, 3, 3, 40]
    assert pool.simulate(costs, [1, 1], pool.lpt_order(costs)) < pool.simulate(costs, [1, 1], list(range(len(costs))))


# ----------------------------------------------------------------------------------------------------------------------- memory
def test_available_memory_takes_the_smallest_limit(tmp_path):
    mi = tmp_path / "meminfo"
    mi.write_text("MemTotal: 999999999 kB\nMemAvailable:   33554432 kB\n")                 # 32 GB free
    assert pool.available_memory_gb({}, str(mi)) <= 32.0 + 1e-9
    assert pool.available_memory_gb({"SLURM_MEM_PER_NODE": "24576"}, str(mi)) <= 24.0 + 1e-9          # --mem=24GB is smaller
    assert pool.available_memory_gb({"SLURM_MEM_PER_CPU": "2048", "SLURM_NTASKS": "4"}, str(mi)) <= 8.0 + 1e-9
    pool.available_memory_gb({"SLURM_MEM_PER_NODE": "junk"}, str(tmp_path / "none"))                      # junk is ignored, never raises


def test_lane_cap():
    assert pool.lane_cap(12.0, 96.0) == 7                   # 90 % of 96 GB / 12 GB per ExGal lane
    assert pool.lane_cap(2.5, 48.0) == 17
    assert pool.lane_cap(12.0, 5.0) == 1                    # never below one lane
    assert pool.lane_cap(12.0, None) is None and pool.lane_cap(0, 50.0) is None


@pytest.mark.parametrize("cap", [1, 2, 3, 7])
def test_plan_respects_the_memory_cap_in_auto_mode(cap):
    plan = pool.plan_workers([20] * 40, 24, None, max_lanes=cap)
    assert len(plan.lanes) <= cap and sum(plan.lanes) <= 24
    assert plan.cap == cap and "memory" in plan.describe("X") or cap >= 24
    if cap == 1:
        assert plan.serial and plan.inner == 24


def test_plan_explicit_patch_workers_above_the_memory_cap_is_kept_with_a_warning():
    plan = pool.plan_workers([20] * 40, 24, 12, max_lanes=3)
    assert len(plan.lanes) == 12 and "memory" in plan.warning and "WARNING" in plan.describe("X")
    assert pool.plan_workers([20] * 40, 24, 3, max_lanes=3).warning is None


def test_cpus_that_no_lane_can_use_go_inside_the_lanes():
    plan = pool.plan_workers([30] * 40, 24, None, max_lanes=4)
    assert sum(plan.lanes) <= 24 and len(plan.lanes) <= 4 and min(plan.lanes) >= 1


# ----------------------------------------------------------------------------------------------------------------------- the driver
def _tasks(ids):
    return [{"id": i, "cost": 1.0} for i in ids]


def _ok(task, inner, **extra):
    r = {"id": task["id"], "outcome": "written", "fault": False, "proper_drop": False, "rvs_ok": True, "ferre_ok": True, "inner": inner,
         "pid": os.getpid()}
    r.update(extra)
    return r


def test_serial_plan_runs_in_this_process_in_table_order_with_all_the_budget():
    seen = []

    def work(task, inner):
        seen.append((task["id"], inner, os.getpid()))
        return _ok(task, inner)
    plan = pool.plan_workers([1, 9, 5], 6, 1)
    res = pool.run_patches(_tasks([10, 20, 30]), work, plan)
    assert [s[0] for s in seen] == [10, 20, 30] and {s[1] for s in seen} == {6} and {s[2] for s in seen} == {os.getpid()}
    assert [r["id"] for r in res] == [10, 20, 30]


def test_serial_plan_lets_an_exception_of_the_work_through_like_the_old_loop():
    def work(task, inner):
        raise SystemExit("No parameter file")           # the old loop did not catch this either
    with pytest.raises(SystemExit):
        pool.run_patches(_tasks([1, 2]), work, pool.plan_workers([1, 1], 4, 1))


def test_parallel_answers_in_table_order_and_biggest_patch_starts_first():
    costs = [1, 1, 1, 30, 1, 1, 20, 1]
    ids = [101, 102, 103, 104, 105, 106, 107, 108]

    def work(task, inner):
        t0 = time.time()
        time.sleep(0.15 if task["id"] in (104, 107) else 0.02)
        return _ok(task, inner, t0=t0)
    plan = pool.plan_workers(costs, 3, 3)
    tasks = [{"id": i, "cost": c} for i, c in zip(ids, costs)]
    res = pool.run_patches(tasks, work, plan)
    assert [r["id"] for r in res] == ids                                    # the order of the table, not of completion
    by_id = {r["id"]: r for r in res}
    for big in (104, 107):                                                  # the two biggest are the FIRST patch of their lane (the queue is largest first);
        assert min(r["t0"] for r in res if r["pid"] == by_id[big]["pid"]) == by_id[big]["t0"]    # not "the first three to start": a lane may be slow to fork
    assert by_id[104]["pid"] != by_id[107]["pid"]                           # two different lanes
    assert {r["inner"] for r in res} == {1}                                 # 3 lanes of 1 inner worker: budget 3


def test_parallel_never_uses_more_workers_than_the_budget():
    def work(task, inner):
        t0 = time.time()
        time.sleep(0.05)
        return _ok(task, inner, t0=t0, t1=time.time())
    plan = pool.plan_workers([4] * 12, 6, 3)                                # 3 lanes of 2
    assert plan.lanes == [2, 2, 2]
    res = pool.run_patches(_tasks(range(1, 13)), work, plan)
    events = sorted([(r["t0"], r["inner"]) for r in res] + [(r["t1"], -r["inner"]) for r in res])
    busy = peak = 0
    for _, d in events:
        busy += d
        peak = max(peak, busy)
    assert peak <= 6
    assert len({r["pid"] for r in res}) <= 3 and os.getpid() not in {r["pid"] for r in res}


def test_a_failing_patch_does_not_stop_the_others_and_is_reported():
    def work(task, inner):
        if task["id"] == 3:
            raise RuntimeError("boom")
        if task["id"] == 5:
            raise SystemExit("sys.exit inside a patch")
        if task["id"] == 7:
            os._exit(9)                                                     # the worker dies (OOM kill)
        return _ok(task, inner)
    plan = pool.plan_workers([1] * 9, 3, 3)
    res = pool.run_patches(_tasks(range(1, 10)), work, plan)
    out = {r["id"]: r["outcome"] for r in res}
    assert [out[i] for i in (1, 2, 4, 6, 8, 9)] == ["written"] * 6
    assert out[3] == out[5] == out[7] == "failed"
    assert all(r["fault"] for r in res if r["outcome"] == "failed")
    assert "RuntimeError" in next(r for r in res if r["id"] == 3)["error"]
    assert "died" in next(r for r in res if r["id"] == 7)["error"]


def test_every_worker_dying_ends_the_run_with_failed_patches_not_a_hang():
    def work(task, inner):
        os._exit(3)
    res = pool.run_patches(_tasks(range(1, 5)), work, pool.plan_workers([1] * 4, 2, 2))
    assert len(res) == 4 and all(r["outcome"] == "failed" for r in res)


def test_lines_of_a_patch_carry_its_id(capfd):
    def work(task, inner):
        print("hello from the patch")
        return _ok(task, inner)
    pool.run_patches(_tasks([12, 34]), work, pool.plan_workers([1, 1], 2, 2))
    out = capfd.readouterr().out
    assert "[P0012] hello from the patch" in out and "[P0034] hello from the patch" in out


def test_line_prefixer_whole_lines_only():
    raw = io.StringIO()
    p = pool.LinePrefixer(raw, "[P0001] ")
    p.write("a b")
    p.write("c\nsecond\n\nthird")
    assert raw.getvalue() == "[P0001] a bc\n[P0001] second\n[P0001] \n"
    p.flush()
    assert raw.getvalue().endswith("[P0001] third\n")


# ----------------------------------------------------------------------------------------------------------------------- aggregation
class FakeTally(object):
    def __init__(self):
        self.n = {"attempted": 0, "written": 0, "no_valid_spaxels": 0, "failed": 0}

    def attempt(self): self.n["attempted"] += 1
    def written(self): self.n["written"] += 1
    def no_valid_spaxels(self): self.n["no_valid_spaxels"] += 1
    def failed(self): self.n["failed"] += 1


def test_apply_results_adds_up_like_the_serial_loops():
    res = [
        {"outcome": "written", "fault": False, "proper_drop": False},
        {"outcome": "no_valid_spaxels", "fault": False, "proper_drop": True},      # prepare returned None
        {"outcome": "no_valid_spaxels", "fault": True, "proper_drop": False},      # Gal: counted as a fault, as before
        {"outcome": "failed", "fault": True, "proper_drop": False},
        pool.failed_result({"id": 9}, "died"),
    ]
    t = FakeTally()
    proper, fault = pool.apply_results(res, t, proper_targets=10)
    assert t.n == {"attempted": 0, "written": 1, "no_valid_spaxels": 2, "failed": 2}
    assert (proper, fault) == (9, 3)


# ----------------------------------------------------------------------------------------------------------------------- the stage functions
def _patch_table(n):
    from astropy.table import Table
    import numpy as np
    i = np.arange(1, n + 1)
    return Table({
        "id": i, "RA_icrs": 150.0 + i * 0.001, "DEC_icrs": np.full(n, 2.0), "A_world": (40.0 - i) / 3600.0, "B_world": (20.0 - i) / 3600.0,
        "angle": np.zeros(n), "flag": np.zeros(n, int), "type": np.array(["T"] * n), "Z": np.full((n, 1), 0.0002), "ZERR": np.full((n, 1), 1e-5),
        "ZWARN": np.zeros((n, 1), int), "CLASS": np.array([["STAR"]] * n),
    })


@pytest.fixture
def gal_env(tmp_path, monkeypatch):
    """ifu_Gal with fake prepare / RVS / FERRE / merge, behaviour by patch id: 3 no spaxels (None), 4 raises 'No valid spaxels', 5 raises, rest ok."""
    pytest.importorskip("scipy")
    from PyAPS import aps_ifu_Gal as G
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    params = tmp_path / "p.json"
    params.write_text("{}")
    out = tmp_path / "out"
    out.mkdir()
    calls = []

    def fake_prepare(infiles, patch_headname, IFU_params, outpath, **kw):
        pid = int(patch_headname.rsplit("_P", 1)[1])
        calls.append(("prepare", pid))
        if pid == 3:
            return None
        if pid == 4:
            raise ValueError("No valid spaxels remaining after filters!")
        if pid == 5:
            raise RuntimeError("prepare broke")
        return {"configs_gal": {"RVS_CONFIG": "r", "FERRE_EXE": "f", "FERRE_TEMPLATES": "t", "FERRE_GRID_PREFIX": "n", "FERRE_GRID_IDS": "1,2"},
                "lsf_values": None, "figdir": str(out), "bin_to_bucket": None, "lsf_by_bucket": None, "gal_targ": {}}

    def fake_rvs(headname, outpath, rvs_config, figdir, nthreads, **kw):
        (out / ("rvs_%s.txt" % headname)).write_text("nthreads=%d" % nthreads)
        return True

    def fake_ferre(headname, outpath, ferre_exe, ferre_templates, ferre_grid_ids, ferre_grid_prefix, configs_gal, nthreads, **kw):
        (out / ("ferre_%s.txt" % headname)).write_text("nthreads=%d" % nthreads)
        return True

    def fake_merge(infiles, outpath, headname, **kw):
        (out / ("%s_APS.txt" % headname)).write_text("patch_id=%s" % kw["patch_id"])
        return 0

    import PyAPS.aps_ifu_rvs as R
    import PyAPS.aps_ifu_ferre as F
    monkeypatch.setattr(G, "ifu_Gal_prepare", fake_prepare)
    monkeypatch.setattr(R, "run_rvs_for_ifu_gal", fake_rvs)
    monkeypatch.setattr(F, "run_ferre_for_ifu_gal", fake_ferre)
    monkeypatch.setattr(G, "ifuGalL2merge", fake_merge)

    def run(table, nthreads, patch_workers):
        for f in out.iterdir():
            f.unlink()
        monkeypatch.setattr(G, "load_and_split_patch_file", lambda pf: (table, True))
        err = None
        try:
            G.ifu_Gal(["a.fit", "b.fit"], "HEAD", str(out) + os.sep, str(cfg), IFU_params=str(params), patch_file="x.fits",
                      rvs_config="r", ferre_exe="f", ferre_templates="t", ferre_grid_ids=["1", "2"], ferre_grid_prefix="n",
                      nthreads=nthreads, patch_workers=patch_workers)
        except SystemExit as e:
            err = str(e)
        summary = json.load(open(out / "HEAD_Gal_patches.json"))
        files = {f.name: f.read_text() for f in out.iterdir() if f.suffix == ".txt"}
        return summary, files, err
    return run


def test_gal_parallel_gives_the_same_products_summary_and_status_as_serial(gal_env):
    t = _patch_table(9)
    s1, f1, e1 = gal_env(t, 1, 1)
    s4, f4, e4 = gal_env(t, 4, 0)
    sp, fp, ep = gal_env(t, 4, 1)               # old behaviour with a budget: one patch at a time, 4 inner
    for k in ("attempted", "written", "no_valid_spaxels", "failed", "yield"):
        assert s1[k] == s4[k] == sp[k]
    assert (s1["attempted"], s1["written"], s1["no_valid_spaxels"], s1["failed"]) == (9, 6, 2, 1)
    assert e1 == e4 == ep is None
    assert sorted(f1) == sorted(f4) == sorted(fp)
    assert all(f1[k] == f4[k] for k in f1 if k.endswith("_APS.txt"))
    assert all(v == "nthreads=1" for k, v in f1.items() if k.startswith(("rvs_", "ferre_")))          # budget 1: every bin loop is serial
    assert all(v == "nthreads=4" for k, v in fp.items() if k.startswith(("rvs_", "ferre_")))          # patch_workers=1: all the budget inside a patch
    assert all(v in ("nthreads=1", "nthreads=2", "nthreads=3", "nthreads=4") for k, v in f4.items() if k.startswith(("rvs_", "ferre_")))


def test_gal_exit_rules_are_the_same_in_parallel(gal_env):
    from astropy.table import Table
    # 12 patches that all fail: >= 10 attempted and none written -> the job fails; "all targets failed" comes first
    t = _patch_table(12)
    t["id"] = [5] * 12                                       # patch id 5 raises in the fake
    for w in (1, 0):
        s, f, e = gal_env(t, 4, w)
        assert e == "All Gal targets failed" and s["written"] == 0 and s["failed"] == 12
    # 12 patches of 'no valid spaxels' by prepare -> None: nothing written from >= 10 attempts
    t["id"] = [3] * 12
    for w in (1, 0):
        s, f, e = gal_env(t, 4, w)
        assert e and e.startswith("No Gal product was written") and s["no_valid_spaxels"] == 12
    # skipped rows (mask, central, galaxy, NaN redshift) are not attempted, in either mode
    t = _patch_table(6)
    t["type"] = ["T", "M", "C", "T", "T", "T"]
    t["CLASS"] = [["STAR"], ["STAR"], ["STAR"], ["GALAXY"], ["STAR"], ["STAR"]]
    t["Z"] = [[0.0], [0.0], [0.0], [0.0], [float("nan")], [0.0]]
    for w in (1, 0):
        s, f, e = gal_env(t, 4, w)
        assert s["attempted"] == 2 and s["written"] == 2 and e is None        # rows 1 and 6 only


@pytest.fixture
def exgal_env(tmp_path, monkeypatch):
    """ifu_ExGal with fake prepare / PPXF / merge, behaviour by patch id: 3 no spaxels (None), 4 raises 'No valid spaxels', 5 raises RuntimeError, 6 another
    ValueError, the rest fine."""
    pytest.importorskip("scipy")
    from PyAPS import aps_ifu_ExGal as E
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    tpl = tmp_path / "tpl"
    tpl.mkdir()
    params = tmp_path / "p.json"
    params.write_text("{}")
    out = tmp_path / "out"
    out.mkdir()

    def fake_prepare(infiles, patch_headname, IFU_params, outpath, IFU_config_dir, **kw):
        pid = int(patch_headname.rsplit("_P", 1)[1])
        if pid == 3:
            return None
        if pid == 4:
            raise ValueError("No valid spaxels remaining after filters!")
        if pid == 5:
            raise RuntimeError("prepare broke")
        if pid == 6:
            raise ValueError("some other value error")
        return {"cube": {"velscale": 1.0}, "configs": {"LS_MODE": 1, "LS_RES": 1}, "LSF_Data": None, "LSF_Templates": None, "figdir": str(out),
                "bin_to_bucket": None, "LSF_Data_by_bucket": None}

    def fake_ppxf(nthreads, configs, velscale, LSF_Data, LSF_Templates, outdir, config_dir, templates_dir, figdir, rootname, *a, **k):
        (out / ("ppxf_%s.txt" % rootname)).write_text("inner=%d" % nthreads)

    def fake_merge(infiles, outpath, headname, **kw):
        (out / ("%s_APS.txt" % headname)).write_text("patch_id=%s" % kw["patch_id"])
        return 0
    monkeypatch.setattr(E, "ifu_ExGal_prepare", fake_prepare)
    monkeypatch.setattr(E.IFUExGalPPXF, "runModule_PPXF", fake_ppxf)
    monkeypatch.setattr(E, "ifuExGalL2merge", fake_merge)

    def run(table, nthreads, patch_workers):
        for f in out.iterdir():
            f.unlink()
        monkeypatch.setattr(E, "load_and_split_patch_file", lambda pf: (table, True))
        err = None
        try:
            E.ifu_ExGal(["a.fit", "b.fit"], "HEAD", str(out) + os.sep, str(cfg), str(tpl), IFU_params=str(params), patch_file="x.fits",
                        PPXF=True, nthreads=nthreads, patch_workers=patch_workers)
        except SystemExit as e:
            err = str(e)
        summary = json.load(open(out / "HEAD_ExGal_patches.json"))
        return summary, {f.name: f.read_text() for f in out.iterdir() if f.suffix == ".txt"}, err
    return run


def test_exgal_parallel_gives_the_same_products_summary_and_status_as_serial(exgal_env):
    t = _patch_table(10)
    t["CLASS"] = [["GALAXY"]] * 10
    s1, f1, e1 = exgal_env(t, 1, 1)
    s8, f8, e8 = exgal_env(t, 8, 0)
    sp, fp, ep = exgal_env(t, 8, 1)
    for k in ("attempted", "written", "no_valid_spaxels", "failed", "yield"):
        assert s1[k] == s8[k] == sp[k]
    assert (s1["attempted"], s1["written"], s1["no_valid_spaxels"], s1["failed"]) == (10, 6, 2, 2)
    assert e1 == e8 == ep is None
    assert sorted(f1) == sorted(f8) == sorted(fp)
    assert all(f1[k] == f8[k] for k in f1 if k.endswith("_APS.txt"))
    assert {v for k, v in f1.items() if k.startswith("ppxf_")} == {"inner=1"}                      # budget 1: the plain serial code
    assert {v for k, v in fp.items() if k.startswith("ppxf_")} == {"inner=8"}                      # patch_workers=1: all the budget in the pool of one patch
    inner = {int(v.split("=")[1]) for k, v in f8.items() if k.startswith("ppxf_")}
    assert min(inner) >= 2 and max(inner) <= 8                                                     # never the serial branch of the modules when there are lanes


def test_exgal_exit_rules_count_faults_as_before(exgal_env):
    t = _patch_table(12)
    t["CLASS"] = [["GALAXY"]] * 12
    t["id"] = [4] * 12                                        # 'No valid spaxels' by exception: not a fault in ExGal
    for w in (1, 0):
        s, f, e = exgal_env(t, 4, w)
        assert e and e.startswith("No ExGal product was written") and s["no_valid_spaxels"] == 12 and s["failed"] == 0
    t["id"] = [5] * 12
    for w in (1, 0):
        s, f, e = exgal_env(t, 4, w)
        assert e == "All ExGal targets failed" and s["failed"] == 12


def test_plan_min_inner_keeps_the_pool_path_of_the_modules():
    plan = pool.plan_workers([30] * 40, 40, None, min_inner=2)
    assert not plan.serial and min(plan.lanes) >= 2 and sum(plan.lanes) <= 40 and len(plan.lanes) <= 20
    assert pool.plan_workers([30] * 40, 3, None, min_inner=2).lanes == [3]          # a budget of 3 cannot make two lanes of 2
    assert pool.plan_workers([30] * 40, 1, None, min_inner=2).lanes == [1]          # a budget of 1 is the old serial loop
    p = pool.plan_workers([30] * 40, 8, 8, min_inner=2)                              # asked for 8 lanes: 4 lanes of 2
    assert len(p.lanes) == 4 and min(p.lanes) == 2


# ----------------------------------------------------------------------------------------------------------------------- the runner
@pytest.mark.parametrize("workers,expect", [(None, None), ("1", "--patch_workers 1"), ("4", "--patch_workers 4"), ("auto", None), ("0", None)])
def test_runner_gives_the_stage_scripts_the_cpu_budget_and_the_patch_workers_option(tmp_path, monkeypatch, workers, expect):
    """--mp_Gal / --mp_ExGal are the CPUs of the resource table (v2: 10 and 40); --patch_workers appears only when script_params sets IFU_patch_workers."""
    pytest.importorskip("yaml")
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    import slurm_script_harness as H
    monkeypatch.setenv("PYAPS_DMS_CONFIG", str(tmp_path / "none.json"))          # the pod's DMS config would choose the profile
    monkeypatch.setenv("PYAPS_SLURM_PROFILE", "v2")
    monkeypatch.setenv("PYAPS_SLURM_LIMITS", "off")
    mod = H.load_runner(H.HERE.parent / "py" / "PyAPS" / "aps_runner.py")
    extra = {"IFU_Gal_grid_prefix_FR": "n"}
    if workers is not None:
        extra["IFU_patch_workers"] = workers
    conf = H.make_conf(tmp_path, extra)
    infiles = [str(tmp_path / "L1" / "20300101" / ("stack_%d.fit" % i)) for i in (1, 2)]
    info = H.fake_l1_info(infiles, "LIFU", "LR", tmp_path)
    monkeypatch.setattr(mod, "l1_fileinfo", lambda files, wlranges=None, arms_ratio=None: dict(info))
    H._relax_paths(mod, monkeypatch)
    scripts = mod.ifu_scriptGEN(infiles=infiles, config_file=str(conf), headname="hn", UAPSID="abcdef012345", hpc=0, aps_ovr=True, mp_LIFU=None, mp_MIFU=None)
    gal = (scripts / "abcdef012345_IFU_Gal.sh").read_text()
    exg = (scripts / "abcdef012345_IFU_ExGal.sh").read_text()
    assert " --mp_Gal 10" in gal and " --mp_ExGal 40" in exg and "#SBATCH --ntasks=10" in gal and "#SBATCH --ntasks=40" in exg
    for text in (gal, exg):
        assert (expect in text) if expect else ("--patch_workers" not in text)
