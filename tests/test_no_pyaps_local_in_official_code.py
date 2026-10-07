"""Guard: official code and tracked configuration never read the personal, untracked work area.

The directory named ``PyAPS_local`` (checkout root) is a personal scratch area: it is ignored by git and
the official dev/prod runs take every location from the configuration files in ``configs/`` only.
This test fails when a tracked file other than the allow-listed ones mentions it, i.e. when code, a
default value, a tracked config or a template would read from there.

Allowed: ``.gitignore`` / ``.dockerignore`` entries, the area's own README, and prose in the top-level
README, CHANGELOG and ``doc/*.md`` that merely describes the directory.

Run with:
    cd <PYAPS_DIR> && pytest tests/test_no_pyaps_local_in_official_code.py -v
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NAME = "PyAPS" + "_local"          # split so that this file does not match itself
ALLOWED = {".gitignore", ".dockerignore", "README.md", "CHANGELOG.md", f"{NAME}/README.md",
           "tests/test_no_pyaps_local_in_official_code.py"}


def _tracked_files():
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return [f for f in out.splitlines() if f]


def _allowed(rel: str) -> bool:
    return rel in ALLOWED or (rel.startswith("doc/") and rel.endswith(".md"))


def test_tracked_code_and_configs_do_not_reference_the_personal_area():
    offenders = []
    for rel in _tracked_files():
        if _allowed(rel):
            continue
        path = ROOT / rel
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if NAME in line:
                offenders.append(f"{rel}:{n}: {line.strip()[:100]}")
    assert not offenders, ("official code/config must not use the personal untracked area; use the keys "
                           "of configs/script_params*.yaml instead:\n  " + "\n  ".join(offenders))


def test_the_personal_area_is_git_ignored():
    text = (ROOT / ".gitignore").read_text()
    assert f"{NAME}/*" in text
