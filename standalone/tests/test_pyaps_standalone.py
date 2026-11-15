# -*- coding: utf-8 -*-
"""Tests of pyaps-standalone: policy by access level, the two-step confirmation, the plan, jobs. A local bare git repository stands in for GitHub; nothing leaves the machine."""
import json
import os
import subprocess
import sys
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from pyaps_standalone import core, tools  # noqa: E402


def git(*a, cwd=None):
    subprocess.check_call(["git"] + list(a), cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    work = tmp_path / "work"
    work.mkdir()
    git("init", "-b", "casper", cwd=work)
    git("config", "user.email", "tester", cwd=work)
    git("config", "user.name", "t", cwd=work)
    (work / "tools").mkdir()
    (work / "tools" / "check_env.py").write_text("print('env ok')\n")
    (work / "pyproject.toml").write_text("[project]\nname='fake'\nversion='1'\n")
    git("add", "-A", cwd=work)
    git("commit", "-m", "init", cwd=work)
    git("branch", "main", cwd=work)
    git("branch", "dev", cwd=work)
    git("branch", "prod", cwd=work)
    git("tag", "v1.0", cwd=work)
    bare = tmp_path / "remote.git"
    git("clone", "--bare", str(work), str(bare))

    def write_config(access="member"):
        path = core.config_path(tools.NAME)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"access": access, "home": str(tmp_path / "px"), "remote": str(bare)}, f)
    return write_config


def test_default_is_member_and_casper_only(env):
    s = tools.status()
    assert s["access"] == "member" and "casper" in s["may_install"]
    assert tools.refs()["installable"] == ["casper"]


def test_member_cannot_install_other_branches(env):
    env("member")
    for ref in ("main", "dev", "prod", "v1.0", "casper2"):
        with pytest.raises(core.StandaloneError, match="member"):
            tools.install(ref)


def test_admin_sees_branches_and_may_install_prod_and_main(env):
    env("admin")
    assert tools.install("main")["status"] == "plan"
    r = tools.refs()
    assert {"casper", "main", "dev", "prod"} <= set(r["branches"]) and "v1.0" in r["tags"] and r["operational"] == ["prod", "dev"]
    plan = tools.install("prod")
    assert plan["status"] == "plan" and plan["plan"]["ref"] == "prod" and any("operational branch" in w for w in plan["plan"]["warnings"])


def test_admin_unknown_ref_is_refused(env):
    env("admin")
    with pytest.raises(core.StandaloneError, match="no branch or tag"):
        tools.install("nonesuch")


@pytest.mark.parametrize("bad", ["../x", "a b", "x;rm", "-rf", "a..b", ""])
def test_ref_syntax_is_checked_before_anything(env, bad):
    env("admin")
    with pytest.raises(core.StandaloneError):
        tools.install(bad)


def test_two_step_confirmation(env):
    env("member")
    plan = tools.install("casper", "explorer")
    assert plan["status"] == "plan" and plan["confirm_with"]
    assert not os.path.exists(os.path.join(tools.config()["home"], "jobs")), "the first call must not start anything"
    wrong = tools.install("casper", "explorer", "0000000000")
    assert wrong["status"] == "refused"
    other_profile = tools.install("casper", "pipeline", plan["confirm_with"])
    assert other_profile["status"] == "refused", "a token belongs to exactly one plan"


def test_plan_steps_for_a_fresh_install(env):
    env("member")
    steps = tools.install("casper", "explorer")["plan"]["steps"]
    assert steps[0].startswith("clone") and "create the virtual environment" in steps and steps[-1] == "record what was installed"


def test_bad_profile(env):
    env("member")
    with pytest.raises(core.StandaloneError, match="profile"):
        tools.install("casper", "everything")


def test_access_cannot_be_raised_by_environment_or_arguments(env, monkeypatch):
    env("member")
    monkeypatch.setenv("PYAPS_STANDALONE_ACCESS", "admin")
    monkeypatch.setenv("ACCESS", "admin")
    assert tools.config()["access"] == "member"
    with pytest.raises(TypeError):
        tools.install("dev", access="admin")


def test_bad_access_value_in_config_is_an_error(env):
    path = core.config_path(tools.NAME)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write(json.dumps({"access": "root"}))
    with pytest.raises(core.StandaloneError, match="member"):
        tools.config()


def test_job_runner_records_exit_and_log(tmp_path):
    jid = core.start_job(str(tmp_path), "echo", ["/bin/sh", "-c", "echo hello; exit 3"])
    for _ in range(50):
        st = core.job_status(str(tmp_path), jid)
        if st["state"] != "running":
            break
        time.sleep(0.1)
    assert st["state"] == "failed" and st["exit_code"] == 3
    assert "hello" in core.job_log(str(tmp_path), jid)


def test_job_stop(tmp_path):
    jid = core.start_job(str(tmp_path), "sleep", ["sleep", "30"])
    assert core.job_status(str(tmp_path), jid)["state"] == "running"
    assert core.stop_job(str(tmp_path), jid)["stopped"]
    for _ in range(50):
        if core.job_status(str(tmp_path), jid)["state"] != "running":
            break
        time.sleep(0.1)
    assert core.job_status(str(tmp_path), jid)["state"] != "running"


def test_run_script_refuses_paths_outside_bin(env):
    env("member")
    cfg = tools.config()
    os.makedirs(os.path.join(tools.tree(cfg), ".git"))
    os.makedirs(os.path.dirname(tools.venv_bin(cfg, "python")))
    open(tools.venv_bin(cfg, "python"), "w").write("")
    for bad in ("../../etc/passwd", "/bin/sh", "nonexistent.py"):
        with pytest.raises(core.StandaloneError):
            tools.run_script(bad, [])


def test_explorer_paths_must_be_absolute_and_exist(env, tmp_path):
    env("member")
    cfg = tools.config()
    os.makedirs(os.path.join(tools.tree(cfg), ".git"))
    os.makedirs(os.path.dirname(tools.venv_bin(cfg, "python")))
    open(tools.venv_bin(cfg, "python"), "w").write("")
    with pytest.raises(core.StandaloneError):
        tools.explorer_start(["relative.fits"])
    with pytest.raises(core.StandaloneError):
        tools.explorer_start(["/nonexistent/x.fits"])


def test_mcp_server_lists_the_tools(env):
    pytest.importorskip("mcp")
    import asyncio
    from pyaps_standalone import server
    srv = server.build()
    names = {t.name for t in asyncio.run(srv.list_tools())}
    assert {"pyaps_status", "pyaps_refs", "pyaps_install", "pyaps_update", "pyaps_check_env", "pyaps_run", "pyaps_job_log"} <= names
