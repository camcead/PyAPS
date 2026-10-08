#!/usr/bin/env python3
"""Check that the Python environment matches what this PyAPS checkout needs.

    python tools/check_env.py                       # the processing pipeline (default profile)
    python tools/check_env.py --profile explorer    # only the interactive explorer and viewers
    python tools/check_env.py --profile all         # explorer + pipeline + cs + server
    python tools/check_env.py --lock none           # skip the comparison with the frozen lock file

Why it exists: a long-lived virtual environment does not follow the code. After the checkout is switched to a newer
branch the environment can silently lack packages (torch, desiutil, ...) and the jobs of a night then fail a few seconds after
they start. This tool finds that before any job is submitted.

What it checks, for the packages the chosen profile needs (the lists are read from ``pyproject.toml``, so they cannot drift from
what ``pip install ".[pipeline]"`` installs):
  * every package is installed, and its version satisfies the declared range;
  * every package those depend on is installed too (what ``pip check`` reports), including the transitive ones;
  * the key modules really import (a wheel can be installed and still be broken);
  * versions that differ from the frozen set ``requirements-lock-*.txt`` are listed as warnings (never an error);
  * a Chrome for kaleido (plot images) is present; if not, a warning says to run ``plotly_get_chrome -y``.

Exit status 0 only when nothing is missing or broken. Standard library only (``packaging`` is used when present). Prints package
names and versions only.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PROFILES = {
    "explorer": ["dependencies"],
    "pipeline": ["dependencies", "pipeline"],
    "cs": ["cs"],
    "server": ["server"],
    "all": ["dependencies", "pipeline", "cs", "server"],
}

# distribution name -> module that must import (only modules that import without a display or a GPU)
IMPORT_CHECKS = {
    "numpy": "numpy", "scipy": "scipy", "matplotlib": "matplotlib", "pandas": "pandas", "astropy": "astropy",
    "regions": "regions", "scikit-learn": "sklearn", "dill": "dill", "plotly": "plotly", "dash": "dash",
    "kaleido": "kaleido", "flask": "flask", "werkzeug": "werkzeug", "itsdangerous": "itsdangerous",
    "gunicorn": "gunicorn", "astropy-healpix": "astropy_healpix", "specutils": "specutils", "photutils": "photutils",
    "reproject": "reproject", "astroquery": "astroquery", "numba": "numba", "emcee": "emcee", "sep": "sep",
    "ppxf": "ppxf.ppxf", "powerbin": "powerbin", "pyyaml": "yaml", "torch": "torch", "desiutil": "desiutil.dust",
    "redrock": "redrock", "rvspecfit": "rvspecfit", "ptemcee": "ptemcee", "pyastronomy": "PyAstronomy.pyasl",
}

def chrome_for_kaleido():
    """Path of a Chrome that kaleido can use to write plot images (png/pdf), or None. kaleido 1.x does not bundle one:
    `plotly_get_chrome -y` installs it under ~/.local/share/choreographer (not a pip package, so pip check cannot see it)."""
    import os
    import shutil
    env = os.environ.get("BROWSER_PATH")
    if env and Path(env).exists():
        return env
    cand = Path.home() / ".local" / "share" / "choreographer" / "deps" / "chrome-linux64" / "chrome"
    if cand.exists():
        return str(cand)
    for name in ("google-chrome", "chromium", "chromium-browser", "chrome"):
        found = shutil.which(name)
        if found:
            return found
    return None


try:                                         # optional: exact version and marker handling
    from packaging.requirements import Requirement
    from packaging.version import Version
except ImportError:                          # pragma: no cover - packaging is present in practice
    Requirement = None
    Version = None


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(spec: str) -> str:
    m = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", spec)
    return norm(m.group(1)) if m else ""


def read_pyproject(path: Path) -> dict:
    """Return {"dependencies": [...], "<extra>": [...]} from pyproject.toml (tomllib when available, else a small parser)."""
    text = path.read_text(encoding="utf-8")
    try:
        import tomllib
        data = tomllib.loads(text)
        out = {"dependencies": list(data["project"].get("dependencies", []))}
        for k, v in data["project"].get("optional-dependencies", {}).items():
            out[k] = list(v)
        return out
    except ImportError:
        pass
    lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
    body = "\n".join(lines)

    def array(key: str, start: int = 0):
        m = re.compile(r"^%s\s*=\s*\[(.*?)\n\]" % re.escape(key), re.S | re.M).search(body, start)
        return re.findall(r'"([^"]+)"', m.group(1)) if m else []

    out = {"dependencies": array("dependencies")}
    opt = body.find("[project.optional-dependencies]")
    for key in ("pipeline", "cs", "server"):
        out[key] = array(key, opt) if opt >= 0 else []
    return out


def profile_requirements(profile: str, pyproject: Path) -> list[str]:
    data = read_pyproject(pyproject)
    specs: list[str] = []
    for section in PROFILES[profile]:
        specs += data.get(section, [])
    return specs


def installed_version(name: str):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def version_ok(spec: str, version: str) -> bool:
    if Requirement is None:
        return True
    try:
        req = Requirement(spec.split("@")[0].strip() if " @ " in spec else spec)
        return not req.specifier or req.specifier.contains(Version(version), prereleases=True)
    except Exception:
        return True


def marker_applies(req_str: str) -> bool:
    if Requirement is None:
        return "extra" not in req_str
    try:
        r = Requirement(req_str)
        return r.marker is None or r.marker.evaluate({"extra": ""})
    except Exception:
        return True


def dependency_closure(names: list[str]):
    """Walk the declared dependencies of every installed package. Returns (closure, unmet), unmet = [(missing, required_by)]."""
    seen: set[str] = set()
    unmet: list[tuple[str, str]] = []
    queue = list(names)
    while queue:
        n = queue.pop()
        if n in seen:
            continue
        seen.add(n)
        if installed_version(n) is None:
            continue
        try:
            requires = md.requires(n) or []
        except md.PackageNotFoundError:
            continue
        for r in requires:
            if not marker_applies(r):
                continue
            rn = requirement_name(r)
            if not rn:
                continue
            if installed_version(rn) is None:
                if (rn, n) not in unmet:
                    unmet.append((rn, n))
            else:
                queue.append(rn)
    return seen, unmet


def read_lock(arg: str):
    if arg == "none":
        return None
    path = Path(arg) if arg != "auto" else None
    if path is None:
        found = sorted(ROOT.glob("requirements-lock-*.txt"))
        if not found:
            return None
        path = found[-1]
    lock = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"([A-Za-z0-9][A-Za-z0-9._-]*)==([^;\s]+)", line.strip())
        if m:
            lock[norm(m.group(1))] = m.group(2)
    return path.name, lock


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--profile", choices=sorted(PROFILES), default="pipeline")
    ap.add_argument("--pyproject", default=str(ROOT / "pyproject.toml"))
    ap.add_argument("--lock", default="auto", help="frozen requirements file, 'auto' (newest in the checkout) or 'none'")
    ap.add_argument("--no-import", action="store_true", help="do not import the key modules (faster, weaker)")
    ap.add_argument("--quiet", action="store_true", help="print only problems and the final line")
    args = ap.parse_args(argv)

    problems: list[str] = []
    warnings: list[str] = []
    specs = profile_requirements(args.profile, Path(args.pyproject))
    if not specs:
        print(f"environment check FAILED: no requirements found for profile '{args.profile}' in {args.pyproject}")
        return 1
    names = []
    for spec in specs:
        n = requirement_name(spec)
        names.append(n)
        v = installed_version(n)
        if v is None:
            problems.append(f"MISSING   {n}   (declared: {spec.split(' @ ')[0].strip()})")
        elif not version_ok(spec, v):
            problems.append(f"VERSION   {n} {v} does not satisfy {spec.strip()}")
        elif not args.quiet:
            print(f"ok        {n} {v}")

    closure, unmet = dependency_closure(names)
    for rn, by in unmet:
        problems.append(f"UNMET     {rn}   (required by {by})")

    if not args.no_import:
        for n in names:
            mod = IMPORT_CHECKS.get(n)
            if mod and installed_version(n) is not None:
                try:
                    importlib.import_module(mod)
                except Exception as exc:          # a broken wheel can raise anything
                    problems.append(f"IMPORT    {mod}: {type(exc).__name__}: {str(exc).splitlines()[0][:120] if str(exc) else ''}")

    if "kaleido" in names and installed_version("kaleido") is not None and chrome_for_kaleido() is None:
        warnings.append("CHROME    kaleido has no Chrome: plot images (png) cannot be written and the jobs log a RuntimeError per figure; "
                        "run `plotly_get_chrome -y` in this environment")

    lock = read_lock(args.lock)
    if lock:
        lock_name, pins = lock
        for n in sorted(closure):
            have, want = installed_version(n), pins.get(n)
            if have and want and have.split("+")[0] != want.split("+")[0]:
                warnings.append(f"DRIFT     {n} {have} (lock {want})")
            elif have and not want and n not in names:
                warnings.append(f"NOT-IN-LOCK {n} {have}")
        if not args.quiet:
            print(f"(compared with {lock_name}: {len(pins)} pins)")

    for w in warnings:
        print(w)
    for p in problems:
        print(p)
    if problems:
        print(f"environment check FAILED for profile '{args.profile}': {len(problems)} problem(s). Repair: in the same environment run "
              f"`pip install -r requirements-lock-*.txt` (or `pip install -e \".[{args.profile}]\"`) and then `pip check`; see README, "
              f"'Updating an existing installation'.")
        return 1
    print(f"environment check passed for profile '{args.profile}'" + (f" ({len(warnings)} warning(s))" if warnings else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
