# -*- coding: utf-8 -*-
"""core: what the two standalone packages (pyaps-standalone, weavedr-standalone) share.

THE SOURCE OF THIS FILE IS _shared/core.py. Each package carries a byte-identical copy as <package>/core.py (tools/sync_core.py copies it; a test checks it),
because a package folder must work on its own.

* the local configuration (one JSON file, read only from the person's own config folder: no environment variable and no tool argument can change the access level)
* the two-step confirmation (a plan, and a token that is only valid for exactly that plan)
* a small job runner (a detached process that survives the end of a Claude session; status and log are files)
* git helpers that never use a shell
No secret is read, written or printed. Nothing here talks to CASPER.
"""
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time

SAFE_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,100}$")
ACCESS_LEVELS = ("member", "admin")


class StandaloneError(Exception):
    """A refusal or a failure with a plain message for the person."""


# ---------------------------------------------------------------------------------------------------------------------------------- configuration
def config_path(name):
    return os.path.join(os.path.expanduser("~"), ".config", "casper-" + name, "config.json")


def load_config(name, defaults):
    """The defaults, overlaid by the person's config file. `access` is member unless the file says admin."""
    cfg = dict(defaults)
    path = config_path(name)
    cfg["_source"] = "defaults (no config file)"
    if os.path.isfile(path):
        with open(path) as f:
            try:
                data = json.load(f)
            except ValueError:
                raise StandaloneError("The config file %s is not valid JSON; fix it or delete it." % path)
        for k, v in data.items():
            if k in defaults:
                cfg[k] = v
        cfg["_source"] = path
    if cfg.get("access") not in ACCESS_LEVELS:
        raise StandaloneError("access in %s must be 'member' or 'admin'." % cfg["_source"])
    cfg["home"] = os.path.abspath(os.path.expanduser(cfg["home"]))
    return cfg


def save_config(name, cfg, keys):
    path = config_path(name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {}
    if os.path.isfile(path):
        with open(path) as f:
            data = json.load(f)
    data.update({k: cfg[k] for k in keys})
    with open(path, "w") as f:
        json.dump(data, f, indent=1, sort_keys=True)
    return path


# ---------------------------------------------------------------------------------------------------------------------------------- two-step confirmation
def plan_token(plan):
    """A short token that belongs to exactly this plan: the second call must present it, so a changed plan is never run on an old yes."""
    blob = json.dumps(plan, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:10]


def two_step(plan, confirm, runner):
    """First call (no confirm): return the plan and its token. Second call (confirm == token): run it."""
    token = plan_token(plan)
    if not confirm:
        return {"status": "plan", "plan": plan, "confirm_with": token,
                "next": "Nothing has been done. To run this plan call the same tool again with confirm=\"%s\"." % token}
    if confirm != token:
        return {"status": "refused", "message": "That confirm value does not belong to this plan (the plan changed or the value is wrong). Call without confirm to see the plan again."}
    return runner()


# ---------------------------------------------------------------------------------------------------------------------------------- processes
def run(argv, cwd=None, timeout=120, env=None):
    """Runs a command (never through a shell) and returns (exit code, combined output text, capped)."""
    try:
        p = subprocess.run(argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, universal_newlines=True)
    except FileNotFoundError:
        return 127, "%s: command not found" % argv[0]
    except subprocess.TimeoutExpired:
        return 124, "%s timed out after %d s" % (argv[0], timeout)
    out = p.stdout or ""
    return p.returncode, out[-6000:]


def find_python(minimum, preferred=None):
    """A Python interpreter at least `minimum` (a tuple), searched among the usual names; returns its path or None."""
    names = [preferred] if preferred else []
    names += ["python%d.%d" % (minimum[0], m) for m in range(14, minimum[1] - 1, -1)] + ["python3", sys.executable]
    for n in names:
        if not n:
            continue
        path = n if os.path.isabs(n) else shutil.which(n)
        if not path or not os.path.isfile(path):
            continue
        rc, out = run([path, "-c", "import sys;print('%d.%d' % sys.version_info[:2])"], timeout=20)
        if rc == 0:
            try:
                got = tuple(int(x) for x in out.strip().split("."))
            except ValueError:
                continue
            if got >= minimum:
                return path
    return None


# ---------------------------------------------------------------------------------------------------------------------------------- git
def check_ref_syntax(ref):
    if not isinstance(ref, str) or not SAFE_REF.match(ref) or ".." in ref or ref.endswith("/") or ref.endswith(".lock"):
        raise StandaloneError("'%s' is not a branch or tag name this tool accepts (letters, digits, . _ - / only)." % ref)
    return ref


def remote_refs(remote, timeout=60):
    """{'heads': [...], 'tags': [...]} of a remote through `git ls-remote`, with the person's own credentials. Raises StandaloneError if unreachable."""
    rc, out = run(["git", "ls-remote", "--heads", "--tags", remote], timeout=timeout, env=dict(os.environ, GIT_TERMINAL_PROMPT="0"))
    if rc != 0:
        raise StandaloneError("Could not read %s with your own git credentials (git said: %s). If this is a private repository you may not have access to it."
                              % (remote, " ".join(out.split())[:300]))
    heads, tags = [], []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        ref = parts[1]
        if ref.startswith("refs/heads/"):
            heads.append(ref[len("refs/heads/"):])
        elif ref.startswith("refs/tags/") and not ref.endswith("^{}"):
            tags.append(ref[len("refs/tags/"):])
    return {"heads": sorted(heads), "tags": sorted(tags)}


def git_state(tree):
    """The checkout at `tree`: branch, commit, dirty or not, or None if it is not a git tree."""
    if not os.path.isdir(os.path.join(tree, ".git")):
        return None
    rc, branch = run(["git", "-C", tree, "rev-parse", "--abbrev-ref", "HEAD"], timeout=20)
    rc2, commit = run(["git", "-C", tree, "rev-parse", "--short=10", "HEAD"], timeout=20)
    rc3, dirty = run(["git", "-C", tree, "status", "--porcelain", "--untracked-files=no"], timeout=30)
    if rc or rc2:
        return None
    return {"branch": branch.strip(), "commit": commit.strip(), "tracked_changes": len([l for l in dirty.splitlines() if l.strip()]) if rc3 == 0 else None}


# ---------------------------------------------------------------------------------------------------------------------------------- state file
def read_state(home):
    path = os.path.join(home, "state.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except ValueError:
        return {}


def write_state(home, state):
    os.makedirs(home, exist_ok=True)
    with open(os.path.join(home, "state.json"), "w") as f:
        json.dump(state, f, indent=1, sort_keys=True)


# ---------------------------------------------------------------------------------------------------------------------------------- jobs (detached processes)
def _jobs_dir(home):
    return os.path.join(home, "jobs")


def start_job(home, label, argv, cwd=None, env=None):
    """Starts argv detached (new session, so it survives the end of the Claude session). The exit code lands in <job>/exit. Returns the job id."""
    jd = _jobs_dir(home)
    os.makedirs(jd, exist_ok=True)
    n = 1 + max([int(d) for d in os.listdir(jd) if d.isdigit()] or [0])
    folder = os.path.join(jd, str(n))
    os.makedirs(folder)
    with open(os.path.join(folder, "job.json"), "w") as f:
        json.dump({"id": n, "label": label, "argv": argv, "cwd": cwd, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, f, indent=1)
    # exit code of the command written to <folder>/exit by a tiny shell wrapper; the command itself is passed as arguments, never parsed by the shell
    wrapper = 'folder="$1"; shift; "$@" > "$folder/log.txt" 2>&1; echo $? > "$folder/exit"'
    p = subprocess.Popen(["/bin/sh", "-c", wrapper, "sh", folder] + list(argv), cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
    with open(os.path.join(folder, "pid"), "w") as f:
        f.write(str(p.pid))
    return n


def _alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        with open("/proc/%d/stat" % pid) as f:
            return f.read().split(") ")[-1].split()[0] != "Z"
    except OSError:
        return True


def job_status(home, job_id):
    folder = os.path.join(_jobs_dir(home), str(int(job_id)))
    if not os.path.isdir(folder):
        raise StandaloneError("There is no job %s." % job_id)
    with open(os.path.join(folder, "job.json")) as f:
        info = json.load(f)
    exit_file = os.path.join(folder, "exit")
    if os.path.isfile(exit_file):
        try:
            code = int(open(exit_file).read().strip())
        except ValueError:
            code = None
        info.update(state="finished" if code == 0 else "failed", exit_code=code)
    else:
        try:
            pid = int(open(os.path.join(folder, "pid")).read())
        except (OSError, ValueError):
            pid = None
        info.update(state="running" if pid and _alive(pid) else "lost", exit_code=None)
    return info


def job_log(home, job_id, tail=60):
    folder = os.path.join(_jobs_dir(home), str(int(job_id)))
    path = os.path.join(folder, "log.txt")
    if not os.path.isdir(folder):
        raise StandaloneError("There is no job %s." % job_id)
    if not os.path.isfile(path):
        return ""
    with open(path, errors="replace") as f:
        lines = f.read().splitlines()
    return "\n".join(lines[-max(1, min(int(tail), 400)):])[-8000:]


def list_jobs(home, limit=10):
    jd = _jobs_dir(home)
    if not os.path.isdir(jd):
        return []
    ids = sorted((int(d) for d in os.listdir(jd) if d.isdigit()), reverse=True)[:limit]
    return [{k: job_status(home, i).get(k) for k in ("id", "label", "state", "exit_code", "started")} for i in ids]


def stop_job(home, job_id):
    info = job_status(home, job_id)
    if info["state"] != "running":
        return {"stopped": False, "state": info["state"]}
    pid = int(open(os.path.join(_jobs_dir(home), str(int(job_id)), "pid")).read())
    try:
        os.killpg(pid, signal.SIGTERM)
    except OSError as e:
        return {"stopped": False, "message": str(e)}
    return {"stopped": True}


# ---------------------------------------------------------------------------------------------------------------------------------- plan runner (several steps in one job)
def run_plan_file(plan_file):
    """Runs the steps of a plan one after the other (used by `cli _run_plan`); stops at the first failing step; prints a line per step."""
    with open(plan_file) as f:
        plan = json.load(f)
    for i, step in enumerate(plan["steps"], 1):
        print("== step %d/%d: %s" % (i, len(plan["steps"]), step["name"]), flush=True)
        print("   $ " + " ".join(step["argv"]), flush=True)
        rc = subprocess.call(step["argv"], cwd=step.get("cwd"), env=dict(os.environ, **step.get("env", {})), stdin=subprocess.DEVNULL)
        if rc != 0:
            print("== step %d FAILED with exit code %d" % (i, rc), flush=True)
            return rc
    print("== done", flush=True)
    return 0


def start_plan_job(home, label, plan, python_module_cwd, module):
    """Writes the plan to a file and runs it as one detached job through `python -m <module> _run_plan <file>`."""
    pdir = os.path.join(home, "plans")
    os.makedirs(pdir, exist_ok=True)
    pf = os.path.join(pdir, "%s-%s.json" % (time.strftime("%Y%m%dT%H%M%S"), plan_token(plan)))
    with open(pf, "w") as f:
        json.dump(plan, f, indent=1)
    return start_job(home, label, [sys.executable, "-m", module, "_run_plan", pf], cwd=python_module_cwd,
                     env=dict(os.environ, PYTHONPATH=python_module_cwd))
