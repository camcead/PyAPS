# -*- coding: utf-8 -*-
"""server: the PyAPS standalone tools as a LOCAL MCP server (stdio). No CASPER token or sign-in is read or needed; the access level comes only from the local config file."""
from . import tools
from .core import StandaloneError

INSTRUCTIONS = ("PyAPS (the WEAVE Advanced Processing System, public code) on THIS machine. Start with pyaps_status. Anything that installs or runs something first returns a plan and a "
                "confirm token; show the plan to the person and call again with confirm only after they agree. Members install the stable public version (branch casper); administrators may choose "
                "other branches (see pyaps_refs).")


def _guard(fn, *a, **k):
    try:
        return fn(*a, **k)
    except StandaloneError as e:
        return {"error": True, "message": str(e)}
    except Exception as e:  # noqa: BLE001 - never leak a path or secret in an exception text
        return {"error": True, "message": "The tool failed unexpectedly (%s). Run `python -m pyaps_standalone.cli doctor` on the machine." % type(e).__name__}


def build():
    try:
        from mcp.server.mcpserver import MCPServer
    except ImportError:
        from mcp.server.fastmcp import FastMCP as MCPServer
    srv = MCPServer("casper-pyaps-standalone", instructions=INSTRUCTIONS)

    @srv.tool(name="pyaps_status", description="What this machine has: your access level here, whether PyAPS is installed (branch, commit, profile), whether Python 3.12+ and git exist, the latest jobs. Read-only. Call this first.")
    def pyaps_status():
        return _guard(tools.status)

    @srv.tool(name="pyaps_refs", description="Which branches can be installed. Members: casper (the stable public version). Administrators: every branch and the latest tags of the remote (reads the remote with git ls-remote).")
    def pyaps_refs():
        return _guard(tools.refs)

    @srv.tool(name="pyaps_install", description="Install PyAPS into your own folder (clone, virtual environment, packages, environment check). profile 'explorer' = viewers and the explorer; 'pipeline' = also the processing pipeline (large). ref 'casper' = stable public version; other refs need administrator access. First call returns the plan and a confirm token; nothing is done until you call again with confirm.")
    def pyaps_install(ref: str = "casper", profile: str = "explorer", confirm: str = ""):
        return _guard(tools.install, ref, profile, confirm or None)

    @srv.tool(name="pyaps_update", description="Update the installed PyAPS to the newest commit of the same branch (or switch ref/profile). Same two-step confirmation. Refuses if the checkout has local changes.")
    def pyaps_update(confirm: str = "", ref: str = "", profile: str = ""):
        return _guard(tools.update, confirm or None, ref or None, profile or None)

    @srv.tool(name="pyaps_check_env", description="Run PyAPS's own environment check (tools/check_env.py) in the installed environment: missing or out-of-range packages, modules that do not import. Read-only.")
    def pyaps_check_env(profile: str = ""):
        return _guard(tools.check_env, profile or None)

    @srv.tool(name="pyaps_templates", description="Read-only status or check of the PyAPS template sets (pyaps templates status|check).")
    def pyaps_templates(command: str = "status"):
        return _guard(tools.templates, command)

    @srv.tool(name="pyaps_explorer_start", description="Start the interactive PyAPS explorer on your own L1/L2 product files (absolute paths) in the background, on a local port. Two-step confirmation. Stop it with pyaps_job_stop.")
    def pyaps_explorer_start(infiles: list[str], port: int = 8080, confirm: str = ""):
        return _guard(tools.explorer_start, infiles, port, confirm or None)

    @srv.tool(name="pyaps_run", description="Run one PyAPS command in the background: 'aps-explorer', 'pyaps', or a script below bin/ of the checkout (for example aps_runner/aps_runner.py), with your arguments. Runs as you; two-step confirmation.")
    def pyaps_run(script: str, args: list[str] = [], confirm: str = ""):
        return _guard(tools.run_script, script, args, confirm or None)

    @srv.tool(name="pyaps_job_list", description="The latest background jobs of this tool (installs, explorer, runs) with their state.")
    def pyaps_job_list():
        return _guard(tools.job_list)

    @srv.tool(name="pyaps_job_status", description="State of one background job: running, finished, failed (with exit code) or lost.")
    def pyaps_job_status(job: int):
        return _guard(tools.job_status, job)

    @srv.tool(name="pyaps_job_log", description="The last lines of a background job's log.")
    def pyaps_job_log(job: int, tail: int = 60):
        return _guard(tools.job_log, job, tail)

    @srv.tool(name="pyaps_job_stop", description="Stop a running background job (two-step confirmation).")
    def pyaps_job_stop(job: int, confirm: str = ""):
        return _guard(tools.job_stop, job, confirm or None)

    return srv


def run_stdio():
    build().run(transport="stdio")
