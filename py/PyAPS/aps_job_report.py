#!/usr/bin/env python3
"""
aps_job_report.py - write one ERROR line per failed SLURM job of an OB into the runner log.

The runner submits an OB's jobs with ``--dependency=afterany`` and exits, so a failed stage is
otherwise only visible in the job's own ``.err`` file. The last stage of the MOS chain (L2merge, which
runs after every other stage has ended) calls this tool for the stages that were killed
(TIMEOUT, CANCELLED, OUT_OF_MEMORY, NODE_FAIL, ...); stages that fail with an exit code log their own
line from their job script (see ``aps_runner.write_bash``).

Line format (same time stamp as the runner shell script)::

    [2026-01-01 12:00:00] ERROR: SLURM job 1234 RVS_L2_<uapsid> TIMEOUT exit code 0:0; stderr: <logs>/RVS_L2_<uapsid>.1234.err

The runner log is ``$PYAPS_RUNNER_LOG`` (set by the job scripts from the ``runner_log`` key of the
script_params file) or ``--runner-log``. Without one the lines go to standard output only.

Usage:
    python aps_job_report.py --tag <uapsid> --logs-path <logs dir> [--runner-log <file>]
"""
import argparse
import datetime
import os
import subprocess
import sys

# States that mean "this job did not finish normally". FAILED is not listed: a stage that exits
# with an error code writes its own line from its job script.
KILLED_STATES = ("TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL", "DEADLINE", "PREEMPTED")
ALL_BAD_STATES = KILLED_STATES + ("FAILED",)


def parse_sacct(text, tag, states=KILLED_STATES):
    """Rows of ``sacct -X -n -P -o JobID,JobName,State,ExitCode`` text for jobs of ``tag`` in a bad state."""
    rows = []
    for line in text.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 4:
            continue
        job_id, name, state, exit_code = (p.strip() for p in parts[:4])
        base_state = state.split()[0].upper()          # e.g. "CANCELLED by 1000"
        if name.endswith("_" + str(tag)) and base_state in states:
            rows.append({"job_id": job_id, "name": name, "state": base_state, "exit_code": exit_code})
    return rows


def format_line(row, logs_path, now=None):
    now = now or datetime.datetime.now()
    err = os.path.join(str(logs_path), f"{row['name']}.{row['job_id']}.err")
    return (f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] ERROR: SLURM job {row['job_id']} {row['name']} "
            f"{row['state']} exit code {row['exit_code']}; stderr: {err}")


def query_sacct(days=30):
    start = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")
    out = subprocess.run(["sacct", "-X", "-n", "-P", "-S", start, "-o", "JobID,JobName%60,State,ExitCode"],
                         capture_output=True, text=True, timeout=120, check=True)
    return out.stdout


def report(tag, logs_path, runner_log=None, sacct_text=None):
    """Print (and append to the runner log) one line per killed job of ``tag``; returns the lines."""
    if sacct_text is None:
        try:
            sacct_text = query_sacct()
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"aps_job_report: could not query sacct: {exc}", file=sys.stderr)
            return []
    lines = [format_line(r, logs_path) for r in parse_sacct(sacct_text, tag)]
    for line in lines:
        print(line)
    runner_log = runner_log or os.environ.get("PYAPS_RUNNER_LOG")
    if lines and runner_log:
        try:
            with open(runner_log, "a") as fh:
                fh.write("\n".join(lines) + "\n")
        except OSError as exc:
            print(f"aps_job_report: cannot write {runner_log}: {exc}", file=sys.stderr)
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tag", required=True, help="uapsid: the job names end with _<tag>")
    ap.add_argument("--logs-path", required=True, help="directory of the job .err/.out files")
    ap.add_argument("--runner-log", default=None, help="file to append to (default: $PYAPS_RUNNER_LOG)")
    args = ap.parse_args(argv)
    report(args.tag, args.logs_path, args.runner_log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
