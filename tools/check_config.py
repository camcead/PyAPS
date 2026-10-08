#!/usr/bin/env python3
"""Check the host-local configuration files that were created from the ``*.example`` templates.

    python tools/check_config.py                       # configs/script_params.yaml, configs/explorer.env
    python tools/check_config.py path/to/file ...      # any other local config file

Reports, per file: the file is missing (and which template to copy), a ``<...>`` placeholder that was
never replaced, a ``${VAR}`` reference whose environment variable is not set, and a local file that is
tracked by git by mistake. Exit status 0 only when every checked file is complete.
Standard library only; never prints values other than the placeholder text itself.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILES = ["configs/script_params.yaml", "configs/explorer.env"]
PLACEHOLDER = re.compile(r"<[A-Za-z][^<>\n]*>")
ENVREF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
AUTO_SET = {"PYAPS_PKG_DIR", "PYAPS_HOME"}   # defaulted by aps_runner itself (PYAPS_HOME: the checkout)


def is_tracked(path: Path) -> bool:
    try:
        r = subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], cwd=ROOT, capture_output=True)
        return r.returncode == 0
    except OSError:
        return False


def check_file(path: Path) -> list[str]:
    rel = path.relative_to(ROOT) if path.is_absolute() and ROOT in path.parents else path
    problems: list[str] = []
    if not path.exists():
        tmpl = Path(str(path) + ".example")
        hint = f" -- copy {tmpl.name} to {path.name} and fill it in" if tmpl.exists() else ""
        return [f"{rel}: missing{hint}"]
    if not path.name.endswith(".example") and is_tracked(path):
        problems.append(f"{rel}: is TRACKED by git; local config must be ignored (see doc/CONFIGURATION.md)")
    for n, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        code = line.split("#", 1)[0] if not line.lstrip().startswith("#") else ""
        for m in PLACEHOLDER.finditer(code):
            problems.append(f"{rel}:{n}: unfilled placeholder {m.group(0)}")
        for m in ENVREF.finditer(code):
            if m.group(1) not in os.environ and m.group(1) not in AUTO_SET:
                problems.append(f"{rel}:{n}: environment variable {m.group(1)} is not set")
    return problems


def main(argv: list[str]) -> int:
    files = [Path(a) for a in argv] or [ROOT / f for f in DEFAULT_FILES]
    all_problems: list[str] = []
    checked = 0
    for f in files:
        if not argv and not f.exists():
            continue                      # optional default files: only check what the host actually has
        checked += 1
        all_problems += check_file(f)
    if not checked:
        print("no local config files found; create them from the *.example templates (doc/CONFIGURATION.md)")
        return 1
    for p in all_problems:
        print(p)
    print("config check FAILED" if all_problems else "config check passed")
    return 1 if all_problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
