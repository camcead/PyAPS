"""The pod's own configured values (DMS config keys slurm_profile, plot_export, slurm_limits, slurm_site, max_inflight_obs) and the one precedence chain.

No pod, no SLURM, no network: the DMS config is a file made here, with a fake password that must never come out. Run:
``cd <PYAPS_DIR> && pytest tests/test_dms_settings.py -v``
"""
from __future__ import annotations

import json

import pytest

dmsset = pytest.importorskip("PyAPS.aps_dms_settings")
slres = pytest.importorskip("PyAPS.aps_slurm_resources")
export = pytest.importorskip("PyAPS.apsPlot.export")

FAKE_VALUE = "value-that-must-never-be-printed"
CTX = {"res": "LR", "stack_level": 1, "grid": "m"}


@pytest.fixture(scope="module")
def table():
    return slres.load_table()


def dms_file(tmp_path, **keys):
    path = tmp_path / "PyAPS_dms_config.json"
    path.write_text(json.dumps(dict({"connection": {"host": "db", "password": FAKE_VALUE}, "l1_dir": "/somewhere", "hpc_mode_l2": "3"}, **keys)))
    return path


def R(table, dms, **kw):
    kw.setdefault("env", {})
    kw.setdefault("limits", {})
    kw.setdefault("quiet", True)
    return slres.resolve("MOS", CTX, table=table, dms=dms, **kw)


# ------------------------------------------------------------------------------------------------------------------- reading the file
def test_reads_only_the_five_keys_and_nothing_else(tmp_path):
    path = dms_file(tmp_path, slurm_profile="v2", max_inflight_obs=6, plot_export="")
    got = dmsset.read_dms_settings({}, path=path)
    assert dict(got) == {"slurm_profile": "v2", "max_inflight_obs": 6}              # empty values are not set; connection, l1_dir ... are not kept
    assert FAKE_VALUE not in repr(got) and got.path == str(path) and got.error is None
    assert set(dmsset.DMS_KEYS) == {"slurm_profile", "plot_export", "slurm_limits", "slurm_site", "max_inflight_obs"}


def test_where_the_file_is(tmp_path, monkeypatch):
    path = dms_file(tmp_path, slurm_profile="v2")
    assert dmsset.dms_config_path({"PYAPS_DMS_CONFIG": str(path)}) == path
    assert dmsset.dms_config_path({"HOME": "/h/u"}).as_posix() == "/h/u/.private_env_configs/PyAPS_dms_config.json"
    assert dmsset.dms_config_path({}) is None                                        # a caller that passes env={} never sees a pod's file
    assert dmsset.dms_config_path({"PYAPS_DMS_CONFIG": "none", "HOME": "/h/u"}) is None
    home = tmp_path / "h"
    (home / ".private_env_configs").mkdir(parents=True)
    dms_file(home / ".private_env_configs", slurm_profile="v2_openmp")
    assert dmsset.read_dms_settings({"HOME": str(home)})["slurm_profile"] == "v2_openmp"
    monkeypatch.setenv("PYAPS_DMS_CONFIG", "none")                                   # what conftest does for every test
    assert dict(dmsset.read_dms_settings()) == {}


def test_missing_or_broken_file_is_an_empty_answer_never_an_error(tmp_path):
    assert dict(dmsset.read_dms_settings({}, path=tmp_path / "nope.json")) == {}
    bad = tmp_path / "bad.json"
    bad.write_text('{"connection": {"password": "%s"' % FAKE_VALUE)                      # truncated JSON
    got = dmsset.read_dms_settings({}, path=bad)
    assert dict(got) == {} and got.error and FAKE_VALUE not in got.error
    lst = tmp_path / "list.json"
    lst.write_text("[1, 2]")
    assert dmsset.read_dms_settings({}, path=lst).error


# ------------------------------------------------------------------------------------------------------------------------- precedence
def test_profile_precedence_option_env_dms_script_params_table(table):
    dms, conf = {"slurm_profile": "v2"}, {"slurm_profile": "v2_openmp"}
    pick = lambda **kw: slres.choose_profile_ex(table, kw.get("conf"), kw.get("cli"), kw.get("env", {}), dms=kw.get("dms", {}))     # noqa: E731
    assert pick() == ("default", "table default_profile")
    assert pick(conf=conf) == ("v2_openmp", "script_params slurm_profile")
    assert pick(conf=conf, dms=dms) == ("v2", "DMS config slurm_profile")                      # the pod's value beats script_params
    assert pick(conf=conf, dms=dms, env={"PYAPS_SLURM_PROFILE": "laptop"}) == ("laptop", "PYAPS_SLURM_PROFILE")
    assert pick(conf=conf, dms=dms, env={"PYAPS_SLURM_PROFILE": "laptop"}, cli="default") == ("default", "--slurm_profile")
    assert pick(conf=conf, dms=dms, cli="auto") == ("v2", "DMS config slurm_profile")          # `auto` is what the DMS passes when it has no value
    assert pick(conf=conf, dms={"slurm_profile": "auto"}) == ("v2_openmp", "script_params slurm_profile")


def test_the_word_site_in_the_dms_config_takes_the_profile_of_the_site(table):
    assert slres.choose_profile_ex(table, None, None, {}, site_profile="v2_openmp", dms={"slurm_profile": "site"}) == ("v2_openmp", "DMS config slurm_profile = site")


def test_resolve_uses_the_dms_profile_everywhere_the_runner_does(table):
    res = R(table, {"slurm_profile": "v2"})
    assert all(r.profile == "v2" for r in res.values()) and res.profile_source == "DMS config slurm_profile"
    legacy = R(table, {}, profile="v2")
    assert {s: (r.cpus, r.mem_gb, r.time) for s, r in res.items()} == {s: (r.cpus, r.mem_gb, r.time) for s, r in legacy.items()}
    assert "profile=v2" in "\n".join(slres.job_header_lines(res["FR"], res.environment))      # the line at the top of every job log
    assert R(table, {}).profile_source == "table default_profile" and next(iter(R(table, {}).values())).profile == "default"
    # a runner that is told the profile on the command line (what the DMS does) is not changed by the file
    assert next(iter(R(table, {"slurm_profile": "v2"}, profile="v2_openmp").values())).profile == "v2_openmp"


def test_limits_and_site_of_the_dms_config(table):
    env = R(table, {"slurm_limits": "clamp"}, limits=None, machine=None, target="local").environment
    assert env.policy == "clamp" and env.policy_source == "DMS config slurm_limits"
    env = R(table, {"slurm_limits": "clamp"}, limits=None, target="local", env={"PYAPS_SLURM_LIMITS": "warn"}).environment
    assert env.policy == "warn" and env.policy_source == "PYAPS_SLURM_LIMITS"                  # the environment variable beats the file
    env = R(table, {"slurm_limits": "clamp"}, limits=None, target="local", conf={"slurm_limits": "off"}).environment
    assert env.policy == "clamp"                                                                # and the file beats script_params
    env = R(table, {"slurm_limits": "off", "slurm_site": "slurm"}, limits=None).environment
    assert env.site["name"] == "slurm" and env.site["source"] == "named by DMS config slurm_site"


def test_a_wrong_value_in_the_dms_config_is_named_as_coming_from_it(table):
    with pytest.raises(slres.ResourceTableError, match="DMS config slurm_limits"):
        R(table, {"slurm_limits": "strict"}, limits=None)
    with pytest.raises(slres.ResourceTableError, match="not in"):
        R(table, {"slurm_profile": "nope"})


def test_plot_export_order_includes_the_dms_config():
    assert export.resolve_mode(None, {"plot_export": "off"}, {}, {"plot_export": "on"}) == ("worker", "DMS config plot_export")
    assert export.resolve_mode(None, {"plot_export": "off"}, {"PYAPS_PLOT_EXPORT": "off"}, {"plot_export": "on"}) == ("off", "environment PYAPS_PLOT_EXPORT")
    assert export.resolve_mode("on", {}, {"PYAPS_PLOT_EXPORT": "off"}, {"plot_export": "off"}) == ("worker", "option --plot_export")
    mode, src = export.resolve_mode(None, {"plot_export": "on"}, {}, {"plot_export": "bogus"})
    assert mode == "worker" and src.startswith("script_params plot_export") and "ignored DMS config plot_export" in src
    assert export.resolve_mode(None, {}, {}, {}) == ("off", "default")


# --------------------------------------------------------------------------------------------------------------------------- reporting
def test_settings_report_names_the_source_of_each_value_and_what_it_overrides(table):
    conf = {"slurm_profile": "default", "plot_export": "on"}
    dms = dmsset.DmsSettings(slurm_profile="v2", max_inflight_obs=6)
    res = R(table, dms, conf=conf, env={"PYAPS_PLOT_EXPORT": "off"})
    items = {it["name"]: it for it in slres.effective_settings(res, conf, {"PYAPS_PLOT_EXPORT": "off"}, {})}
    assert items["slurm_profile"]["value"] == "v2" and items["slurm_profile"]["source"] == "DMS config slurm_profile"
    assert items["slurm_profile"]["overridden"] == [("script_params slurm_profile", "default")]
    assert items["plot_export"]["value"] == "off" and items["plot_export"]["source"] == "environment PYAPS_PLOT_EXPORT"
    assert items["plot_export"]["overridden"] == [("script_params plot_export", "on")]
    assert items["max_inflight_obs"]["value"] == 6 and "DMS config" in items["max_inflight_obs"]["source"]
    text = slres.format_settings(res, conf, {"PYAPS_PLOT_EXPORT": "off"}, {})
    assert "set by DMS config slurm_profile" in text and "overridden: script_params slurm_profile=default" in text and "max_inflight_obs" in text
    cli = {item["name"]: item for item in slres.effective_settings(res, conf, {}, {"profile": "v2_openmp"})}
    assert cli["slurm_profile"]["overridden"][0] == ("DMS config slurm_profile", "v2")          # a CLI value would also say what it hides


def test_cli_show_check_explain_site_use_the_dms_config_and_never_print_the_file(tmp_path, capsys):
    path = dms_file(tmp_path, slurm_profile="v2", max_inflight_obs=6)
    sp = tmp_path / "sp.yaml"
    sp.write_text("slurm_profile: 'default'\n")
    common = ["--no-limits", "--dms-config", str(path), "--script-params", str(sp)]
    for cmd in ("show", "explain", "check", "site"):
        assert slres.main([cmd] + common) in (0, 1)
        out = capsys.readouterr().out
        assert FAKE_VALUE not in out and "/somewhere" not in out, cmd
        assert "set by DMS config slurm_profile" in out and "overridden: script_params slurm_profile=default" in out, cmd
        assert "max_inflight_obs" in out and " 6 " in out, cmd
        assert "profile=v2" in out or "profile v2" in out or "v2" in out, cmd
    slres.main(["show"] + common)
    assert "profile=v2" in capsys.readouterr().out
    slres.main(["show", "--no-limits", "--dms-config", "none", "--script-params", str(sp)])           # the file switched off: script_params decides again
    out = capsys.readouterr().out
    assert "profile=default" in out and "DMS config: none found" in out
    assert slres.main(["site", "--json"] + common) == 0
    data = json.loads(capsys.readouterr().out)
    assert {s["name"]: s["value"] for s in data["settings"]}["slurm_profile"] == "v2" and FAKE_VALUE not in json.dumps(data)


def test_the_runner_prints_and_records_the_pod_profile(tmp_path, monkeypatch, capsys):
    """aps_runner.resolve_slurm_resources: no --slurm_profile (a run started by hand, aps-offline): the pod's DMS config still decides."""
    runner = pytest.importorskip("PyAPS.aps_runner")
    path = dms_file(tmp_path, slurm_profile="v2", max_inflight_obs=6)
    monkeypatch.setenv("PYAPS_DMS_CONFIG", str(path))
    monkeypatch.delenv("PYAPS_SLURM_PROFILE", raising=False)
    monkeypatch.setattr(runner, "_ENV_CACHE", {})
    monkeypatch.setattr(runner, "_RES_CLI", {})
    monkeypatch.setattr(runner, "_PLOT_EXPORT_CLI", None)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    res = runner.resolve_slurm_resources("MOS", {"slurm_limits": "off"}, CTX, None, scripts, uapsid="abc")
    out = capsys.readouterr().out
    assert all(r.profile == "v2" for r in res.values())
    assert "(profile v2," in out and "set by DMS config slurm_profile" in out and FAKE_VALUE not in out
    rec = json.loads((scripts / slres.EFFECTIVE_NAME).read_text())["runs"]["abc"]
    assert rec["profile"] == "v2" and rec["profile_source"] == "DMS config slurm_profile"
    assert "profile=v2" in "\n".join(slres.job_header_lines(res["FR"], res.environment))
    # with the command line (what the DMS passes) nothing changes
    monkeypatch.setattr(runner, "_ENV_CACHE", {})
    res = runner.resolve_slurm_resources("MOS", {"slurm_limits": "off"}, CTX, None, scripts, uapsid="abc", slurm_profile="v2_openmp")
    assert all(r.profile == "v2_openmp" for r in res.values()) and res.profile_source == "--slurm_profile"
