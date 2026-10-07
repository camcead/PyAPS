"""tests/test_check_env.py - the environment checker (tools/check_env.py) and the dependency declarations it reads.

The pipeline jobs of a night fail a few seconds after they start when the virtual environment lacks a package, so the packages
the pipeline needs must stay declared in ``pyproject.toml`` and pinned in the frozen requirements file, and the checker must
report a missing package as a failure.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_env", ROOT / "tools" / "check_env.py")
check_env = importlib.util.module_from_spec(spec)
sys.modules["check_env"] = check_env
spec.loader.exec_module(check_env)


def declared(profile):
    return {check_env.requirement_name(s) for s in check_env.profile_requirements(profile, ROOT / "pyproject.toml")}


def test_pipeline_profile_declares_the_packages_the_jobs_import():
    names = declared("pipeline")
    # aps_rvs imports torch; aps_utils imports desiutil.dust (PPXF and LS jobs); both failed on a pod that lacked them.
    for needed in ("torch", "desiutil", "ppxf", "numba", "redrock", "rvspecfit", "numpy", "astropy"):
        assert needed in names, f"{needed} must be declared for the pipeline profile"


def test_cs_profile_declares_the_contributed_module_packages():
    assert {"ptemcee", "pyastronomy"} <= declared("cs")


def test_explorer_profile_is_the_light_set():
    names = declared("explorer")
    assert "dash" in names and "torch" not in names


def test_all_profile_is_the_union():
    assert declared("all") >= declared("pipeline") | declared("cs") | declared("server")


def test_lock_file_pins_the_packages_that_torch_and_pyastronomy_pull_in():
    locks = sorted(ROOT.glob("requirements-lock-*.txt"))
    if not locks:
        pytest.skip("no frozen requirements file on this branch")
    pins = check_env.read_lock(str(locks[-1]))[1]
    for pkg in ("torch", "desiutil", "ptemcee", "pyastronomy", "filelock", "mpmath", "sympy", "networkx", "quantities", "bidict"):
        assert pkg in pins, f"{pkg} missing from {locks[-1].name}"


def test_missing_package_fails_the_check(monkeypatch, capsys):
    real = check_env.installed_version
    monkeypatch.setattr(check_env, "installed_version", lambda n: None if n == "desiutil" else real(n))
    rc = check_env.main(["--profile", "pipeline", "--no-import", "--lock", "none", "--quiet"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "MISSING   desiutil" in out and "FAILED" in out


def test_requirement_name_handles_extras_urls_and_markers():
    f = check_env.requirement_name
    assert f("numpy>=2.5") == "numpy"
    assert f("PyYAML>=6.0") == "pyyaml"
    assert f("redrock @ git+https://github.com/desihub/redrock.git") == "redrock"
    assert f("astropy_healpix>=2.0 ; python_version >= '3.12'") == "astropy-healpix"
