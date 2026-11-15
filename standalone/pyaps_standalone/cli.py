# -*- coding: utf-8 -*-
"""cli: `python -m pyaps_standalone.cli doctor | status | refs | install | update | set-access | serve | mcp-config`.

The same actions as the MCP tools, for a person at a terminal, and the two hidden helpers the install job calls (_run_plan, _pip_install, _record).
"""
import argparse
import json
import os
import shutil
import sys

from . import __version__, tools
from . import core
from .core import StandaloneError


def _print(obj):
    print(json.dumps(obj, indent=1, default=str))


def doctor():
    bad = 0

    def line(ok, what):
        nonlocal bad
        bad += (not ok)
        print(("ok   " if ok else "FAIL ") + what)
    line(sys.version_info >= (3, 10), "python %s runs this tool (3.10 or newer)" % sys.version.split()[0])
    try:
        import mcp  # noqa: F401
        line(True, "the MCP library is installed")
    except ImportError:
        line(False, "the MCP library is missing: pip install mcp   (in the environment that runs `serve`)")
    try:
        cfg = tools.config()
        line(True, "config: %s; access level: %s; folder: %s" % (cfg["_source"], cfg["access"], cfg["home"]))
        py = core.find_python((3, 12), cfg.get("python"))
        line(bool(py), "Python 3.12 or newer for PyAPS itself: %s" % (py or "not found"))
        line(bool(shutil.which("git")), "git: %s" % (shutil.which("git") or "not found"))
        st = tools.status()
        print("info installed: %s" % (st["checkout"] or "no"))
    except StandaloneError as e:
        line(False, str(e))
    print("\n%s" % ("all fine" if not bad else "%d problem(s) above" % bad))
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pyaps-standalone", description="PyAPS on this machine, installable and usable through Claude (local MCP server).")
    ap.add_argument("--version", action="version", version=__version__)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for n in ("doctor", "status", "refs", "serve", "mcp-config"):
        sp.add_parser(n)
    i = sp.add_parser("install")
    i.add_argument("--ref", default=tools.PUBLIC_REF)
    i.add_argument("--profile", default="explorer", choices=sorted(tools.PROFILES))
    i.add_argument("--yes", action="store_true", help="skip the plan question (the plan is still printed)")
    u = sp.add_parser("update")
    u.add_argument("--yes", action="store_true")
    s = sp.add_parser("set-access", help="member (default) or admin: which branches this tool will install; see docs/ACCESS.md")
    s.add_argument("level", choices=core.ACCESS_LEVELS)
    h = sp.add_parser("_run_plan")
    h.add_argument("plan_file")
    p = sp.add_parser("_pip_install")
    p.add_argument("tree")
    p.add_argument("pip")
    p.add_argument("profile")
    p.add_argument("extra", nargs="*")
    r = sp.add_parser("_record")
    r.add_argument("ref")
    r.add_argument("profile")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "doctor":
            return doctor()
        if a.cmd == "_run_plan":
            return core.run_plan_file(a.plan_file)
        if a.cmd == "_pip_install":
            return tools.pip_install(a.tree, a.pip, a.profile, a.extra)
        if a.cmd == "_record":
            return tools.record(a.ref, a.profile)
        if a.cmd == "status":
            _print(tools.status())
        elif a.cmd == "refs":
            _print(tools.refs())
        elif a.cmd in ("install", "update"):
            res = tools.install(a.ref, a.profile) if a.cmd == "install" else tools.update()
            _print(res)
            if res.get("status") == "plan":
                if not a.yes and input("Run this plan? [y/N] ").strip().lower() != "y":
                    return 1
                _print(tools.install(a.ref, a.profile, res["confirm_with"]) if a.cmd == "install" else tools.update(res["confirm_with"]))
        elif a.cmd == "set-access":
            cfg = tools.config()
            cfg["access"] = a.level
            print("written: " + core.save_config(tools.NAME, cfg, ["access"]))
        elif a.cmd == "serve":
            from . import server
            server.run_stdio()
        elif a.cmd == "mcp-config":
            here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            print("claude mcp add casper-pyaps-standalone -- env PYTHONPATH=%s %s -m pyaps_standalone.cli serve" % (here, sys.executable))
        return 0
    except StandaloneError as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
