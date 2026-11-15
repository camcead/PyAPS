# -*- coding: utf-8 -*-
"""tools: what the PyAPS standalone MCP server can do, as plain functions that return dictionaries (the server only wraps them).

PyAPS is public (github.com/camcead/PyAPS, MIT licence). Who may install which branch:

* member (the default): the branch `casper` only: the stable public version (always the latest prod commit that is running and has passed its tests; see CASPER_BRANCH.md).
* admin (set in the person's own config, see docs/ACCESS.md): any branch or tag that exists on the remote, for instance `main` (the frozen branch of the paper), `prod` and `dev`.

The access level is a rule of THIS tool, read from the person's own config file; it is not a lock. The branches `dev` and `prod` of the PyAPS GitHub repository are
readable by anyone, so nothing here can hide them; the rule keeps members on the released code and keeps the dev branch for the people who run it.
The private operator layer PyAPS_admin is NOT handled here at all.
"""
import glob
import os
import shutil
import sys

from . import core
from .core import StandaloneError

NAME = "pyaps-standalone"
MODULE = "pyaps_standalone.cli"
DEFAULTS = {
    "access": "member",
    "home": "~/pyaps-standalone",
    "remote": "https://github.com/camcead/PyAPS.git",
    "python": None,
}
PUBLIC_REF = "casper"
PROFILES = {"explorer": ("the interactive explorer and the viewers: open and inspect products", "."),
            "pipeline": ("the processing pipeline as well (aps_runner and the science modules; a large install: torch, redrock, ...)", ".[pipeline]")}
# What `pyaps_run` may start: the console commands of the installed package, and the scripts below the checkout's bin/ folder.
ENTRY_POINTS = ("aps-explorer", "pyaps")


def config():
    return core.load_config(NAME, DEFAULTS)


def tree(cfg):
    return os.path.join(cfg["home"], "PyAPS")


def venv_bin(cfg, name):
    return os.path.join(cfg["home"], "venv", "bin", name)


def allowed_refs(cfg):
    return [PUBLIC_REF] if cfg["access"] == "member" else None      # None = any ref the remote has


def check_ref(cfg, ref):
    core.check_ref_syntax(ref)
    if cfg["access"] == "member" and ref != PUBLIC_REF:
        raise StandaloneError("Your access level here is 'member': you can install the stable public version, the branch 'casper', only. The operational branches (prod, dev) are for the people who "
                              "run the WEAVE pipeline: ask a WEAVE project administrator if you need one.")
    return ref


# ---------------------------------------------------------------------------------------------------------------------------------- read-only tools
def status():
    cfg = config()
    st = core.read_state(cfg["home"])
    tr = tree(cfg)
    gs = core.git_state(tr)
    py = core.find_python((3, 12), cfg.get("python"))
    out = {
        "access": cfg["access"], "config": cfg["_source"], "home": cfg["home"], "remote": cfg["remote"],
        "may_install": "the branch 'casper' only (the stable public version)" if cfg["access"] == "member" else "any branch or tag of the remote (for example casper, prod, dev, main)",
        "python_3_12_or_newer": py or "NOT FOUND: PyAPS needs Python 3.12 or newer (set \"python\" in the config file, or install one)",
        "git": shutil.which("git") or "NOT FOUND",
        "installed": bool(gs and os.path.isfile(venv_bin(cfg, "python"))),
        "checkout": gs, "recorded": {k: st.get(k) for k in ("ref", "profile", "installed_at") if k in st},
        "recent_jobs": core.list_jobs(cfg["home"], 5),
    }
    if gs and gs["tracked_changes"]:
        out["warning"] = "the checkout has %d locally changed tracked file(s); an update will not overwrite them" % gs["tracked_changes"]
    return out


def refs():
    cfg = config()
    if cfg["access"] == "member":
        return {"access": "member", "installable": [PUBLIC_REF], "note": "Members install the stable public version (the branch casper). A WEAVE administrator can enable other branches for you."}
    r = core.remote_refs(cfg["remote"])
    return {"access": "admin", "branches": r["heads"], "tags": r["tags"][-20:],
            "operational": [b for b in ("prod", "dev") if b in r["heads"]],
            "note": "prod is what runs the WEAVE pipeline now, dev is the next release on trial; casper is the latest prod that has proved itself (what members get); main is the frozen branch of the PyAPS paper."}


# ---------------------------------------------------------------------------------------------------------------------------------- install and update
def _install_plan(cfg, ref, profile, update):
    if profile not in PROFILES:
        raise StandaloneError("profile must be one of: %s." % ", ".join(PROFILES))
    py = core.find_python((3, 12), cfg.get("python"))
    if not py:
        raise StandaloneError("No Python 3.12 or newer was found. Install one, or set \"python\" in %s." % core.config_path(NAME))
    if not shutil.which("git"):
        raise StandaloneError("git was not found on this machine.")
    tr, venv = tree(cfg), os.path.join(cfg["home"], "venv")
    pip = venv_bin(cfg, "pip")
    steps = []
    if os.path.isdir(os.path.join(tr, ".git")):
        gs = core.git_state(tr)
        if gs and gs["tracked_changes"]:
            raise StandaloneError("The checkout %s has locally changed tracked files; commit or discard them first (this tool never overwrites your changes)." % tr)
        steps += [{"name": "fetch %s" % ref, "argv": ["git", "-C", tr, "fetch", "origin", "+refs/heads/%s:refs/remotes/origin/%s" % (ref, ref), "--tags"]},
                  {"name": "check out %s" % ref, "argv": ["git", "-C", tr, "checkout", "-B", ref, "origin/%s" % ref]}]
    else:
        if os.path.exists(tr):
            raise StandaloneError("%s exists and is not a git checkout; move it away first." % tr)
        steps.append({"name": "clone %s (branch %s)" % (cfg["remote"], ref), "argv": ["git", "clone", "--branch", ref, cfg["remote"], tr]})
    if not os.path.isfile(venv_bin(cfg, "python")):
        steps.append({"name": "create the virtual environment", "argv": [py, "-m", "venv", venv]})
    lock = None
    if profile == "pipeline":
        lock = "requirements-lock-*.txt"
    extra = ["--extra-index-url", "https://download.pytorch.org/whl/cpu"] if profile == "pipeline" and not shutil.which("nvidia-smi") else []
    steps.append({"name": "upgrade pip", "argv": [pip, "install", "--upgrade", "pip"]})
    steps.append({"name": "install the packages of PyAPS (%s)" % profile, "argv": [sys.executable, "-m", "pyaps_standalone.cli", "_pip_install", tr, pip, profile] + extra,
                  "env": {"PYTHONPATH": os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}})
    steps.append({"name": "check the environment", "argv": [venv_bin(cfg, "python"), os.path.join("tools", "check_env.py"), "--profile", profile], "cwd": tr})
    steps.append({"name": "record what was installed", "argv": [sys.executable, "-m", "pyaps_standalone.cli", "_record", ref, profile],
                  "env": {"PYTHONPATH": os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}})
    warn = []
    if ref in ("dev", "prod") or ref.startswith("dev/") or ref.startswith("prod/"):
        warn.append("'%s' is an operational branch: it is what the WEAVE team runs or tries next, not a release. Use it to reproduce or test, not to publish." % ref)
    return {"action": "update" if update else "install", "ref": ref, "profile": profile, "profile_means": PROFILES[profile][0], "remote": cfg["remote"], "folder": cfg["home"],
            "steps": [s["name"] for s in steps], "warnings": warn, "runs_in_background": "yes: a job; follow it with pyaps_job_status / pyaps_job_log",
            "_steps": steps}


def install(ref=PUBLIC_REF, profile="explorer", confirm=None, update=False):
    cfg = config()
    check_ref(cfg, ref)
    if cfg["access"] == "admin":
        avail = core.remote_refs(cfg["remote"])
        if ref not in avail["heads"] + avail["tags"]:
            raise StandaloneError("The remote has no branch or tag '%s'. Use pyaps_refs to see what exists." % ref)
    plan = _install_plan(cfg, ref, profile, update)
    steps = plan.pop("_steps")
    public_plan = dict(plan)

    def go():
        pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        job = core.start_plan_job(cfg["home"], "%s PyAPS %s (%s)" % (plan["action"], ref, profile), {"steps": steps}, pkg_root, MODULE)
        return {"status": "started", "job": job, "next": "Follow it with pyaps_job_status(%d) and pyaps_job_log(%d). A first install takes minutes (the pipeline profile much longer)." % (job, job)}
    return core.two_step(public_plan, confirm, go)


def update(confirm=None, ref=None, profile=None):
    cfg = config()
    st = core.read_state(cfg["home"])
    if not st.get("ref"):
        raise StandaloneError("PyAPS is not installed here yet: use pyaps_install.")
    return install(ref or st["ref"], profile or st.get("profile", "explorer"), confirm, update=True)


def pip_install(tr, pip, profile, extra_args):
    """The pip steps of an install, run inside the job (`cli _pip_install`): the frozen set first when the profile needs the pipeline, then the package itself."""
    import subprocess
    lock = sorted(glob.glob(os.path.join(tr, "requirements-lock-*.txt")))
    spec = PROFILES[profile][1]
    if profile == "pipeline" and lock:
        rc = subprocess.call([pip, "install", "-r", lock[-1]] + extra_args, cwd=tr)
        if rc:
            return rc
        return subprocess.call([pip, "install", "--no-deps", "-e", spec] + extra_args, cwd=tr)
    return subprocess.call([pip, "install", "-e", spec] + extra_args, cwd=tr)


def record(ref, profile):
    import time
    cfg = config()
    gs = core.git_state(tree(cfg)) or {}
    core.write_state(cfg["home"], {"ref": ref, "profile": profile, "commit": gs.get("commit"), "installed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    return 0


# ---------------------------------------------------------------------------------------------------------------------------------- use
def _need_installed(cfg):
    if not os.path.isfile(venv_bin(cfg, "python")) or not os.path.isdir(os.path.join(tree(cfg), ".git")):
        raise StandaloneError("PyAPS is not installed here yet: use pyaps_install first.")


def check_env(profile=None):
    cfg = config()
    _need_installed(cfg)
    st = core.read_state(cfg["home"])
    prof = profile or st.get("profile", "explorer")
    if prof not in ("explorer", "pipeline", "all"):
        raise StandaloneError("profile must be explorer, pipeline or all.")
    rc, out = core.run([venv_bin(cfg, "python"), os.path.join("tools", "check_env.py"), "--profile", prof], cwd=tree(cfg), timeout=300)
    return {"profile": prof, "ok": rc == 0, "exit_code": rc, "report": out}


def templates(command="status"):
    cfg = config()
    _need_installed(cfg)
    if command not in ("status", "check"):
        raise StandaloneError("Only the read-only commands 'status' and 'check' are offered here.")
    rc, out = core.run([venv_bin(cfg, "pyaps"), "templates", command], cwd=tree(cfg), timeout=300)
    return {"command": command, "ok": rc == 0, "exit_code": rc, "output": out}


def _folder_ok(path):
    return os.path.isabs(path) and ".." not in path.split(os.sep)


def explorer_start(infiles, port=8080, confirm=None):
    cfg = config()
    _need_installed(cfg)
    if not infiles or not all(isinstance(p, str) and _folder_ok(p) for p in infiles):
        raise StandaloneError("infiles must be a list of absolute paths of products (no '..').")
    missing = [p for p in infiles if not os.path.exists(p)]
    if missing:
        raise StandaloneError("These files do not exist: %s" % ", ".join(missing[:5]))
    if not 1024 <= int(port) <= 65535:
        raise StandaloneError("port must be between 1024 and 65535.")
    argv = [venv_bin(cfg, "aps-explorer"), "--infiles"] + list(infiles) + ["--port", str(int(port))]
    plan = {"action": "start the explorer", "argv": argv, "address": "http://127.0.0.1:%d (this machine only, if the explorer binds to localhost; use an SSH tunnel from your own computer)" % int(port)}

    def go():
        job = core.start_job(cfg["home"], "explorer on %d" % int(port), argv, cwd=tree(cfg))
        return {"status": "started", "job": job, "stop_with": "pyaps_job_stop(%d)" % job}
    return core.two_step(plan, confirm, go)


def run_script(script, args=None, confirm=None):
    """Starts one PyAPS command in the background: a console command of the package (aps-explorer, pyaps) or a script below the checkout's bin/ folder."""
    cfg = config()
    _need_installed(cfg)
    args = [str(a) for a in (args or [])]
    if script in ENTRY_POINTS:
        argv = [venv_bin(cfg, script)] + args
    else:
        bin_root = os.path.realpath(os.path.join(tree(cfg), "bin"))
        path = os.path.realpath(os.path.join(bin_root, script))
        if not path.startswith(bin_root + os.sep) or not os.path.isfile(path):
            raise StandaloneError("'%s' is neither a console command of PyAPS (%s) nor a script below bin/ of the checkout." % (script, ", ".join(ENTRY_POINTS)))
        argv = [venv_bin(cfg, "python"), path] + args
    plan = {"action": "run a PyAPS command", "argv": argv, "folder": tree(cfg), "note": "Runs as you, on this machine, in the background. It can write wherever the command's own options say."}

    def go():
        job = core.start_job(cfg["home"], "run " + script, argv, cwd=tree(cfg))
        return {"status": "started", "job": job}
    return core.two_step(plan, confirm, go)


def job_status(job):
    return core.job_status(config()["home"], job)


def job_log(job, tail=60):
    return {"job": job, "log": core.job_log(config()["home"], job, tail)}


def job_list():
    return {"jobs": core.list_jobs(config()["home"], 15)}


def job_stop(job, confirm=None):
    cfg = config()
    plan = {"action": "stop job %d" % int(job)}
    return core.two_step(plan, confirm, lambda: core.stop_job(cfg["home"], job))
