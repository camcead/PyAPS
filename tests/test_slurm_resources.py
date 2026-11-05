"""The SLURM resource table: loading, validation, rules, overrides, limits, escalation.

Standard library + PyYAML only (no WEAVE data, no SLURM). Run with:
    cd <PYAPS_DIR> && pytest tests/test_slurm_resources.py -v
"""
from __future__ import annotations

import copy
import json

import pytest

slres = pytest.importorskip("PyAPS.aps_slurm_resources")

NO_LIMITS = {}


@pytest.fixture(scope="module")
def table():
    return slres.load_table()


def R(family, table, ctx=None, **kw):
    ctx = ctx or {"res": "LR", "stack_level": 1, "grid": "n"}
    kw.setdefault("limits", NO_LIMITS)
    kw.setdefault("env", {})
    kw.setdefault("quiet", True)
    return slres.resolve(family, ctx, table=table, **kw)


# ------------------------------------------------------------------------------------------------ the shipped table
def test_the_shipped_table_is_valid_and_complete(table):
    assert table["default_profile"] == "default"
    for name in table["profiles"]:
        flat = slres.flatten_profile(table, name)
        for fam, stages in slres.FAMILIES.items():
            assert set(flat[fam]["stages"]) >= set(stages), (name, fam)


def test_default_profile_holds_the_values_of_the_old_code(table):
    mos = R("MOS", table)
    assert [mos[s].cpus for s in slres.FAMILIES["MOS"]] == [8, 12, 18, 10, 20, 6, 1]
    assert all(mos[s].time == "2-00:00:00" for s in slres.FAMILIES["MOS"])
    assert all(mos[s].mem_gb is None for s in mos)                      # grid n: no memory request
    lifu = R("LIFU", table)
    assert [lifu[s].cpus for s in slres.FAMILIES["LIFU"]] == [6, 30, 20]
    assert lifu["IFU_ExGal"].mem_gb == 240
    mifu = R("MIFU", table)
    assert [mifu[s].cpus for s in slres.FAMILIES["MIFU"]] == [3, 10, 10]
    assert all(mifu[s].mem_gb is None for s in mifu)
    cs = R("CS", table)
    assert [cs[s].cpus for s in slres.FAMILIES["CS"]] == [6, 12, 6, 12, 12, 1, 1]
    assert all(cs[s].time is None for s in cs)                          # the CS stages have never had a limit


def test_special_cases_of_the_old_code(table):
    hr = R("MOS", table, {"res": "HR", "stack_level": 1, "grid": "m"})
    assert [hr[s].cpus for s in slres.FAMILIES["MOS"]] == [16, 24, 36, 20, 40, 12, 2]       # HR doubles every CPU value
    assert hr["FR"].mem_gb == 380
    lr = R("MOS", table, {"res": "LR", "stack_level": 1, "grid": "m"})
    assert lr["FR"].mem_gb == 190 and lr["FR"].cpus == 18
    single_lr = R("MOS", table, {"res": "LR", "stack_level": 0, "grid": "n"})
    assert single_lr["FR"].cpus == 10 and single_lr["RR"].cpus == 8     # int(18/2) + 1
    single_hr = R("MOS", table, {"res": "HR", "stack_level": 0, "grid": "n"})
    assert single_hr["FR"].cpus == 20                                   # halved BEFORE the doubling: (18//2+1)*2
    assert R("CS", table, {"res": "HR", "stack_level": 1, "grid": "m"})["FESWI"].mem_gb == 400
    assert R("CS", table, {"res": "HR", "stack_level": 1, "grid": "n"})["FESWI"].mem_gb is None


def test_stack_level_names_and_numbers_both_match(table):
    t = copy.deepcopy(table)
    t["profiles"]["default"]["MOS"]["stages"]["RR"]["cases"] = [{"when": {"stack": [0, 1]}, "cpus": {"set": 3}}]
    slres.validate_table(t)
    assert R("MOS", t, {"res": "LR", "stack_level": 0, "grid": "n"})["RR"].cpus == 3
    assert R("MOS", t, {"res": "LR", "stack_level": 1, "grid": "n"})["RR"].cpus == 3
    assert R("MOS", t, {"res": "LR", "stack_level": 3, "grid": "n"})["RR"].cpus == 8
    t["profiles"]["default"]["MOS"]["stages"]["RR"]["cases"] = [{"when": {"stack": "single"}, "cpus": {"set": 3}}]
    assert R("MOS", t, {"res": "LR", "stack_level": 0, "grid": "n"})["RR"].cpus == 3
    assert R("MOS", t, {"res": "LR", "stack_level": 1, "grid": "n"})["RR"].cpus == 8


# ----------------------------------------------------------------------------------------------------- validation
def _bad(table, mutate, match):
    t = copy.deepcopy(table)
    mutate(t)
    with pytest.raises(slres.ResourceTableError, match=match):
        slres.validate_table(t)


def test_validation_names_the_bad_place(table):
    _bad(table, lambda t: t["profiles"]["default"]["MOS"]["stages"]["RR"].update(cpus=0), r"profiles\.default\.MOS\.stages\.RR\.cpus")
    _bad(table, lambda t: t["profiles"]["default"]["MOS"]["stages"]["RR"].update(cpus=2.5), "whole number")
    _bad(table, lambda t: t["profiles"]["default"]["MOS"]["stages"]["RR"].update(mem_gb=-4), "mem_gb")
    _bad(table, lambda t: t["profiles"]["default"]["MOS"]["stages"]["RR"].update(time="soon"), "not a SLURM time")
    _bad(table, lambda t: t["profiles"]["default"]["MOS"]["stages"]["RR"].update(status="great"), "status")
    _bad(table, lambda t: t["profiles"]["default"]["MOS"]["stages"]["RR"].update(speed=1), "unknown key")
    _bad(table, lambda t: t["profiles"]["default"]["MOS"]["stages"].update(NOPE={"cpus": 1}), "not a stage of MOS")
    _bad(table, lambda t: t["profiles"]["default"]["MOS"]["stages"]["RR"].update(cases=[{"when": {"colour": "red"}, "cpus": 1}]), "unknown condition")
    _bad(table, lambda t: t["profiles"]["default"]["MOS"]["stages"]["RR"].update(cases=[{"when": {"res": "HR"}}]), "changes cpus")
    _bad(table, lambda t: t["profiles"]["default"]["MOS"]["stages"]["RR"].update(cases=[{"when": {"res": "HR"}, "cpus": {"per_cpu": 2}}]), "memory operation")
    _bad(table, lambda t: t["profiles"]["default"]["MOS"]["stages"]["RR"].update(cases=[{"when": {"res": "HR"}, "time": {"add": 1}}]), "set and scale")
    _bad(table, lambda t: t["profiles"].update(v3={"extends": "nope"}), "not a profile")
    _bad(table, lambda t: t["profiles"].update(a={"extends": "b"}, b={"extends": "a"}), "loops back")
    _bad(table, lambda t: t["profiles"]["default"].update(limits="maybe"), "limits")
    _bad(table, lambda t: t["profiles"]["default"]["MOS"]["stages"].pop("EMI"), "stage EMI has no cpus")
    _bad(table, lambda t: t.update(schema=2), "schema")
    _bad(table, lambda t: t.update(sites={"s": {"max_time": "never"}}), "not a SLURM time")
    _bad(table, lambda t: t.update(sites={"s": {"max_gpus": 1}}), "unknown key")


def test_a_profile_that_extends_lists_only_what_changes(table):
    t = copy.deepcopy(table)
    t["profiles"]["tiny"] = {"extends": "default", "MOS": {"stages": {"RR": {"cpus": 2, "mem_gb": 16, "time": "04:00:00", "status": "measured"}}}}
    slres.validate_table(t)
    res = R("MOS", t, profile="tiny")
    assert (res["RR"].cpus, res["RR"].mem_gb, res["RR"].time, res["RR"].status) == (2, 16, "04:00:00", "measured")
    assert res["RVS"].cpus == 12 and res["RVS"].profile == "tiny"       # untouched rows come from the parent
    assert R("MOS", t, {"res": "HR", "stack_level": 1, "grid": "n"}, profile="tiny")["RR"].cpus == 4      # the parent's HR rule still applies


# -------------------------------------------------------------------------------------------------------- rules
def test_operations_and_memory_per_cpu(table):
    t = copy.deepcopy(table)
    t["profiles"]["x"] = {"extends": "default", "MOS": {"stages": {"FR": {"cpus": 10, "time": "01:00:00", "cases": [
        {"when": {"res": "HR"}, "cpus": {"add": 2}},
        {"when": {"res": "HR"}, "time": {"scale": 1.5}},
        {"when": {"res": "HR"}, "mem_gb": {"per_cpu": 4}},
    ]}}}}
    slres.validate_table(t)
    hr = R("MOS", t, {"res": "HR", "stack_level": 1, "grid": "n"}, profile="x")["FR"]
    assert hr.cpus == (10 + 2) * 2                     # add, then the family's HR doubling
    assert hr.time == "0-01:30:00"
    assert hr.mem_gb == 4 * hr.cpus                    # per CPU of the FINAL count


def test_numeric_ranges_in_conditions(table):
    t = copy.deepcopy(table)
    t["profiles"]["x"] = {"extends": "default", "MOS": {"stages": {"RR": {"cpus": 8, "cases": [
        {"when": {"stack_level": {"ge": 3}}, "cpus": {"set": 2}}]}}}}
    slres.validate_table(t)
    assert R("MOS", t, {"res": "LR", "stack_level": 3, "grid": "n"}, profile="x")["RR"].cpus == 2
    assert R("MOS", t, {"res": "LR", "stack_level": 1, "grid": "n"}, profile="x")["RR"].cpus == 8


def test_the_trace_says_why(table):
    fr = R("MOS", table, {"res": "HR", "stack_level": 0, "grid": "n"})["FR"]
    text = "\n".join(fr.trace)
    assert "half_plus_one" in text and "scale" in text


# ----------------------------------------------------------------------------------------------------- overrides
def test_legacy_mp_list_replaces_the_base_cpus_and_the_rules_still_apply(table):
    res = R("MOS", table, {"res": "HR", "stack_level": 0, "grid": "n"}, mp_base=[3, 5, 7, 9, 11, 13, 2])
    assert [res[s].cpus for s in slres.FAMILIES["MOS"]] == [6, 10, 8, 18, 22, 26, 4]
    with pytest.raises(slres.ResourceTableError, match="needs 7 values"):
        R("MOS", table, mp_base=[1, 2])


def test_override_precedence_cli_over_script_params_over_profile(table):
    conf = {"slurm_cpus_RR": "5", "slurm_mem_gb_RR": "30", "slurm_time_RR": "06:00:00"}
    res = R("MOS", table, conf=conf)
    assert (res["RR"].cpus, res["RR"].mem_gb, res["RR"].time) == (5, 30, "06:00:00")
    res = R("MOS", table, conf=conf, cli_set=["RR.cpus=2", "RR.time=none", "FR.mem_gb=77"])
    assert (res["RR"].cpus, res["RR"].mem_gb, res["RR"].time) == (2, 30, None)
    assert res["FR"].mem_gb == 77
    assert res["RVS"].cpus == 12


def test_an_override_is_final_no_hr_doubling_on_top(table):
    res = R("MOS", table, {"res": "HR", "stack_level": 1, "grid": "n"}, cli_set=["RR.cpus=4"])
    assert res["RR"].cpus == 4 and res["RVS"].cpus == 24


def test_a_bad_time_override_falls_back_with_a_warning(table):
    res = R("MOS", table, conf={"slurm_time_EMI": "soon"})
    assert res["EMI"].time == "2-00:00:00"
    assert "not a SLURM time" in res["EMI"].warnings[0]


def test_set_option_parser():
    assert slres.parse_set_option("FR.cpus=12") == ("FR", "cpus", 12)
    assert slres.parse_set_option("FR.mem_gb=120.5") == ("FR", "mem_gb", 120.5)
    assert slres.parse_set_option("FR.mem_gb=none") == ("FR", "mem_gb", None)
    assert slres.parse_set_option("LS.time=1-12:00:00") == ("LS", "time", "1-12:00:00")
    for bad in ("FR", "FR.speed=1", "FR.cpus=0", "FR.cpus=x", "FR.mem_gb=-1", "FR.time=soon"):
        with pytest.raises(slres.ResourceTableError):
            slres.parse_set_option(bad)


def test_profile_choice_order(table, monkeypatch):
    assert R("MOS", table)["RR"].profile == "default"
    assert R("MOS", table, conf={"slurm_profile": "v2"})["RR"].profile == "v2"
    assert R("MOS", table, conf={"slurm_profile": "v2"}, env={"PYAPS_SLURM_PROFILE": "default"})["RR"].profile == "default"
    assert R("MOS", table, conf={"slurm_profile": "v2"}, env={"PYAPS_SLURM_PROFILE": "default"}, profile="v2")["RR"].profile == "v2"
    assert R("MOS", table, conf={"slurm_profile": "v2"}, profile="auto")["RR"].profile == "v2"     # what the DMS passes
    with pytest.raises(slres.ResourceTableError, match="not in"):
        R("MOS", table, profile="nope")


# -------------------------------------------------------------------------------------------------- node limits
def test_limits_policy_warn_keeps_and_clamp_lowers(table):
    ctx = {"res": "HR", "stack_level": 1, "grid": "m"}
    warn = R("MOS", table, ctx, limits={"max_mem_gb": 340, "max_cpus": 90}, profile="default")
    assert warn["FR"].mem_gb == 380 and "no node can run it" in warn["FR"].warnings[0]
    t = copy.deepcopy(table)
    t["profiles"]["default"]["limits"] = "clamp"
    clamp = R("MOS", t, ctx, limits={"max_mem_gb": 340})
    assert clamp["FR"].mem_gb == 340 and "clamped" in clamp["FR"].warnings[0]
    t["profiles"]["default"]["limits"] = "off"
    assert R("MOS", t, ctx, limits={"max_mem_gb": 340})["FR"].mem_gb == 380


def test_named_site_supplies_the_limits(table):
    t = copy.deepcopy(table)
    t["sites"] = {"dev": {"max_mem_gb": 340, "max_cpus": 90, "max_time": "2-00:00:00"}}
    slres.validate_table(t)
    t["profiles"]["default"]["limits"] = "clamp"
    res = slres.resolve("MOS", {"res": "HR", "stack_level": 1, "grid": "m"}, table=t, site="dev", env={}, quiet=True)
    assert res["FR"].mem_gb == 340
    with pytest.raises(slres.ResourceTableError, match="site"):
        slres.resolve("MOS", {"res": "LR", "stack_level": 1, "grid": "n"}, table=t, site="mars", env={}, quiet=True)


def test_sinfo_parser():
    out = "90|360000|2-00:00:00\n32|68719|2-00:00:00\n"
    assert slres.parse_sinfo_limits(out) == {"max_cpus": 90, "max_mem_gb": 351, "max_time": "2-00:00:00"}
    assert slres.parse_sinfo_limits("") == {}
    assert slres.parse_sinfo_limits("90+|1024+|infinite\n") == {"max_cpus": 90, "max_mem_gb": 1}


# ----------------------------------------------------------------------------------------------------- time
def test_time_helpers():
    assert slres.parse_time_seconds("2-00:00:00") == 172800
    assert slres.parse_time_seconds("90") == 5400
    assert slres.parse_time_seconds("10:30") == 630
    assert slres.parse_time_seconds("1-12") == 129600
    assert slres.format_time(5400) == "0-01:30:00"
    with pytest.raises(slres.ResourceTableError):
        slres.parse_time_seconds("two days")


# ---------------------------------------------------------------------------------------------- escalation
def test_escalation_record_is_the_retry_counter(tmp_path):
    assert slres.can_escalate(tmp_path, "abc", "FR")
    rec = slres.record_escalation(tmp_path, "abc", "FR", "OUT_OF_MEMORY", job_id="12", previous={"mem_gb": 190})
    assert rec["attempts"] == 1 and rec["mem_factor"] == 1.5 and rec["time_factor"] == 1.0
    assert not slres.can_escalate(tmp_path, "abc", "FR")             # ONCE
    assert slres.can_escalate(tmp_path, "abc", "RR")                 # per stage
    assert slres.can_escalate(tmp_path, "other", "FR")               # per run: the folder of an OB holds the scripts of every pair of the OB
    data = json.loads((tmp_path / slres.ESCALATION_NAME).read_text())
    assert data["runs"]["abc"]["FR"]["history"][0]["previous"] == {"mem_gb": 190}
    assert slres.read_escalation(tmp_path, "abc")["FR"]["mem_factor"] == 1.5 and slres.read_escalation(tmp_path, "other") == {}
    with pytest.raises(slres.ResourceTableError):
        slres.record_escalation(tmp_path, "abc", "RR", "FAILED")


def test_escalation_scales_memory_and_time_when_the_scripts_are_generated_again(table):
    esc = {"FR": {"mem_factor": 1.5, "time_factor": 1.0}, "RR": {"mem_factor": 1.0, "time_factor": 1.5}}
    t = copy.deepcopy(table)
    t["profiles"]["default"]["MOS"]["stages"]["RR"]["time"] = "10:00:00"
    res = R("MOS", t, {"res": "LR", "stack_level": 1, "grid": "m"}, escalation=esc)
    assert res["FR"].mem_gb == 285                                    # 190 x 1.5
    assert res["FR"].time == "2-00:00:00"
    assert res["RR"].time == "0-15:00:00"                             # 10 h x 1.5
    assert res["RR"].mem_gb is None


def test_escalation_of_a_stage_without_a_memory_request_scales_the_slurm_default(table):
    res = R("MOS", table, escalation={"RR": {"mem_factor": 1.5, "time_factor": 1.0}})
    assert res["RR"].mem_gb == 12                                     # 8 cpus x 1 GB (DefMemPerCPU) x 1.5


def test_escalated_values_never_pass_the_node_or_partition_limits(table):
    esc = {"FR": {"mem_factor": 1.5, "time_factor": 1.5}}
    res = R("MOS", table, {"res": "HR", "stack_level": 1, "grid": "m"}, escalation=esc, limits={"max_mem_gb": 500, "max_time": "2-00:00:00"},
            profile="default")
    assert res["FR"].mem_gb == 500 and res["FR"].time == "2-00:00:00"


def test_resource_failures_from_job_info(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "EMI_L2_abc.9.err").write_text("slurmstepd: error: Detected 1 oom-kill event(s) in StepId=9.batch\n")
    info = {
        "1": {"Status": "OUT_OF_MEMORY", "JobName": "FR_L2_abc"},
        "2": {"Status": "TIMEOUT", "JobName": "RR_L2_abc"},
        "3": {"Status": "COMPLETED", "JobName": "RVS_L2_abc"},
        "4": {"Status": "FAILED", "JobName": "LS_L2_abc"},
        "9": {"Status": "FAILED", "JobName": "EMI_L2_abc"},
        "5": {"Status": "CANCELLED by 1000", "JobName": "PPXF_L2_abc"},
        "6": {"Status": "OUT_OF_MEMORY", "JobName": "SQ_CS_abc"},
        "7": {"Status": "TIMEOUT", "JobName": "weird-name"},
    }
    got = slres.resource_failures(info, logs_dir=logs)
    assert {k: v["reason"] for k, v in got.items()} == {"FR": "OUT_OF_MEMORY", "RR": "TIMEOUT", "EMI": "OUT_OF_MEMORY", "SQ": "OUT_OF_MEMORY"}
    assert got["FR"]["job_id"] == "1"


# ------------------------------------------------------------------------------------------- reporting / header
def test_effective_file_and_job_header(tmp_path, table):
    res = R("MOS", table, {"res": "HR", "stack_level": 1, "grid": "m"})
    summary = slres.effective_summary(res, {"res": "HR", "grid": "m"}, table, "MOS", uapsid="abc")
    slres.write_effective(tmp_path, summary)
    slres.write_effective(tmp_path, slres.effective_summary(R("CS", table), {"res": "LR"}, table, "CS", uapsid="abc"))
    slres.write_effective(tmp_path, slres.effective_summary(R("MOS", table), {"res": "LR"}, table, "MOS", uapsid="def"))    # another pair of the OB
    data = json.loads((tmp_path / slres.EFFECTIVE_NAME).read_text())
    assert data["runs"]["abc"]["stages"]["FR"]["mem_gb"] == 380 and data["runs"]["abc:CS"]["stages"]["FESWI"]["cpus"] == 12
    assert slres.read_effective(tmp_path, "abc")["stages"]["FR"]["cpus"] == 36
    assert slres.read_effective(tmp_path, "def")["stages"]["FR"]["cpus"] == 18
    assert slres.read_effective(tmp_path, slres.effective_tag("abc", "CS"))["family"] == "CS"
    assert slres.read_effective(tmp_path, "none") == {}
    head = "\n".join(slres.job_header_lines(res["FR"]))
    assert "cpus=36" in head and "mem=380 GB" in head and "SLURM_NTASKS" in head and "profile=default" in head


def test_write_effective_never_raises_on_a_missing_folder(tmp_path, table, capsys):
    slres.write_effective(tmp_path / "no" / "such", slres.effective_summary(R("MOS", table), {}, table, "MOS"))
    assert "could not write" in capsys.readouterr().out


def test_cli_validate_show_explain(capsys):
    assert slres.main(["validate"]) == 0
    assert slres.main(["show", "--family", "MOS", "--res", "HR", "--stack", "single", "--grid", "m", "--no-limits"]) == 0
    out = capsys.readouterr().out
    assert "FR" in out and "380 GB" in out
    assert slres.main(["explain", "--family", "LIFU", "--no-limits"]) == 0
    assert slres.main(["show", "--profile", "nope"]) == 2


# ------------------------------------------------------------------------------- FERRE thread consistency
def test_ferre_threads_per_process_fill_the_cpus_of_the_job_and_no_more():
    aps_ferre = pytest.importorskip("PyAPS.aps_ferre")
    f = aps_ferre.ferre_threads_per_process
    assert f(18, 9, env={}) == 2            # 9 grids x 2 threads = the 18 CPUs asked for (it used to be 9 x 18 = 162 threads)
    assert f(36, 9, env={}) == 4
    assert f(9, 9, env={}) == 1
    assert f(10, 9, env={}) == 1
    assert f(4, 9, env={}) == 1             # fewer CPUs than grids: 4 processes at a time, one thread each
    assert f(20, 1, env={}) == 20           # one grid: all CPUs in its threads
    assert f(18, 9, env={"PYAPS_FERRE_THREADS": "mp"}) == 18      # the old behaviour on request
    assert f(18, 9, env={"PYAPS_FERRE_THREADS": "3"}) == 3
    assert f(18, 9, env={"PYAPS_FERRE_THREADS": "zero"}) == 2


# --------------------------------------------------------------------------------------------------- profile v2
def test_v2_asks_for_memory_in_every_l2_stage_so_enforcement_can_be_switched_on(table):
    """With memory enforced a job without --mem gets DefMemPerCPU x cpus (1 GB per CPU) and RR (28 GB) would be killed: every L2 stage of v2
    carries an explicit request, in every class."""
    for fam in ("MOS", "LIFU", "MIFU"):
        for res in ("LR", "HR"):
            for stack in (0, 1):
                for grid in ("m", "n"):
                    rows = R(fam, table, {"res": res, "stack_level": stack, "grid": grid}, profile="v2")
                    for stage, r in rows.items():
                        assert r.mem_gb and r.mem_gb >= 1 and r.time, (fam, res, stack, grid, stage)


def test_v2_has_no_hr_doubling_except_where_it_was_measured(table):
    """RR keeps 16 in HR (19 min with 16 CPUs against 38 min with 8 at the same core-hours, dev rehearsal 2026-11-05) and RVS 24 (its pool is --mp);
    every other stage uses the same cores in HR."""
    lr = R("MOS", table, {"res": "LR", "stack_level": 1, "grid": "n"}, profile="v2")
    hr = R("MOS", table, {"res": "HR", "stack_level": 1, "grid": "n"}, profile="v2")
    assert {s: lr[s].cpus for s in lr} == {"RR": 8, "RVS": 12, "FR": 9, "PPXF": 10, "EMI": 6, "LS": 2, "L2merge": 1}
    assert {s: hr[s].cpus for s in hr} == {**{s: lr[s].cpus for s in lr}, "RR": 16, "RVS": 24}
    lifu = R("LIFU", table, {"res": "HR", "stack_level": 1, "grid": "n"}, profile="v2")
    assert [lifu[s].cpus for s in ("IFU_PR", "IFU_ExGal", "IFU_Gal")] == [3, 40, 4]


def test_v2_fr_asks_less_than_the_old_190_and_380_gb_but_enough_for_the_fine_grids(table):
    fr = R("MOS", table, {"res": "LR", "stack_level": 1, "grid": "m"}, profile="v2")["FR"]
    assert fr.mem_gb == 64 and fr.time == "2-00:00:00" and fr.cpus == 9
    single = R("MOS", table, {"res": "LR", "stack_level": 0, "grid": "n"}, profile="v2")["FR"]
    assert single.mem_gb == 8 and single.time == "1-06:00:00"


def test_v2_never_asks_more_than_the_smallest_big_node_of_the_two_sites(table):
    """The smallest big node of the two sites has 360000 MB (351 GB), the other 720000 MB: nothing may ask for more than 340 GB."""
    for fam in ("MOS", "LIFU", "MIFU", "CS"):
        for res in ("LR", "HR"):
            for grid in ("m", "n"):
                for r in R(fam, table, {"res": res, "stack_level": 1, "grid": grid}, profile="v2").values():
                    assert (r.mem_gb or 0) <= 340, (fam, res, grid, r.stage, r.mem_gb)


def test_v2_openmp_profile_gives_fr_two_threads_per_grid(table):
    r = R("MOS", table, {"res": "LR", "stack_level": 1, "grid": "n"}, profile="v2_openmp")
    assert r["FR"].cpus == 18 and r["RR"].cpus == 8 and r["FR"].profile == "v2_openmp"


def test_v2_memory_is_the_true_footprint_with_a_margin_not_the_rss_sum(table):
    """The memory column was set from the PSS peaks of the dev rehearsal (RR 7-10 GB LR, 17 GB HR; RVS 21-57 GB LR, 97 GB HR; ...), not from the RSS sum
    of sacct (RR 28-65 GB, IFU_ExGal 212 GB). It must stay above the largest peak seen: with cgroup enforcement it is the kill limit."""
    lr = R("MOS", table, {"res": "LR", "stack_level": 0, "grid": "n"}, profile="v2")
    hr = R("MOS", table, {"res": "HR", "stack_level": 0, "grid": "n"}, profile="v2")
    peaks_lr = {"RR": 10.4, "RVS": 57.2, "FR": 3.0, "PPXF": 7.0, "EMI": 11.0, "LS": 3.5, "L2merge": 1.5}
    peaks_hr = {"RR": 17.5, "RVS": 97.3, "FR": 3.0, "PPXF": 7.0, "EMI": 6.0, "LS": 3.5, "L2merge": 1.5}
    for s, peak in peaks_lr.items():
        assert lr[s].mem_gb >= 1.25 * peak, (s, lr[s].mem_gb, peak)
    for s, peak in peaks_hr.items():
        assert hr[s].mem_gb >= 1.25 * peak, (s, hr[s].mem_gb, peak)
    assert lr["RR"].mem_gb == 24 and hr["RR"].mem_gb == 36 and lr["RVS"].mem_gb == 72 and hr["RVS"].mem_gb == 144


def test_the_old_v2_pss_profile_is_gone(table):
    assert "v2_pss" not in table["profiles"]
