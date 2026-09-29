"""
tests/test_aps_utils_configdir.py - Tests for aps_utils.validate_and_set_configdir's
PYAPS_CONFIGDIR env-var fallback (added alongside the aps_explorer.py
server-mode work: a deployment that doesn't ship the repo-bundled
configs/ExGal_configs directory at all needs a real, persistent,
writable configdir to point at, without threading a new parameter
through every caller across the codebase — see aps_utils.py's own
comment on this fallback for the full reasoning).

No APSOB/L1-data fixtures needed — validate_and_set_configdir is a
plain, standalone function.

Run with:
    cd <PYAPS_DIR> && pytest tests/test_aps_utils_configdir.py -v
"""

from __future__ import annotations

import pytest

from PyAPS.aps_utils import validate_and_set_configdir


def test_pyaps_configdir_env_used_when_no_explicit_arg(tmp_path, monkeypatch):
    real_dir = tmp_path / "configs"
    real_dir.mkdir()
    monkeypatch.setenv("PYAPS_CONFIGDIR", str(real_dir))
    assert validate_and_set_configdir(None, verbose=False) == str(real_dir)


def test_explicit_arg_takes_priority_over_pyaps_configdir_env(tmp_path, monkeypatch):
    explicit_dir = tmp_path / "explicit"
    explicit_dir.mkdir()
    env_dir = tmp_path / "from_env"
    env_dir.mkdir()
    monkeypatch.setenv("PYAPS_CONFIGDIR", str(env_dir))
    assert validate_and_set_configdir(str(explicit_dir), verbose=False) == str(explicit_dir)


def test_invalid_pyaps_configdir_env_falls_through(tmp_path, monkeypatch):
    """A set-but-nonexistent PYAPS_CONFIGDIR must not itself be returned
    (silently pointing every caller at a directory that doesn't exist) —
    it falls through to whatever the next candidate in the chain is,
    exactly like an invalid explicit `configdir` argument already does."""
    monkeypatch.setenv("PYAPS_CONFIGDIR", str(tmp_path / "does_not_exist"))
    # No assertion on the final resolved value here (that depends on
    # whether this checkout happens to ship configs/ExGal_configs, which
    # it does not as of this session — see the aps_explorer.py plan's own
    # note) -- only that it doesn't blow up returning a nonexistent path.
    try:
        result = validate_and_set_configdir(None, verbose=False)
    except RuntimeError:
        return  # also acceptable: no valid configdir anywhere at all
    import os
    assert os.path.isdir(result)


def test_unset_pyaps_configdir_does_not_affect_explicit_arg(tmp_path, monkeypatch):
    monkeypatch.delenv("PYAPS_CONFIGDIR", raising=False)
    explicit_dir = tmp_path / "explicit"
    explicit_dir.mkdir()
    assert validate_and_set_configdir(str(explicit_dir), verbose=False) == str(explicit_dir)


def test_apsob_lsf_fwhm_pickledir_source_regression_guard():
    """One real bug found and fixed by direct code reading (see the
    aps_explorer.py deployment-hardening plan for the full story) inside
    APSOB.__init__ — not practically unit-testable in isolation, since
    constructing a real APSOB needs actual L1 FITS/catalog data this
    test environment doesn't have (matching this repo's existing lack of
    any full-APSOB fixture). A source-level guard against an accidental
    revert, not a substitute for the live deployment verification this
    fix was actually confirmed with: of the four run_lsf_analysis/
    run_fwhm_analysis call sites that set pickle_dir, one (the join_arms
    + FWHM branch) used self._caldir instead of self._pickledir --
    writing LSF/FWHM cache straight into the (possibly read-only)
    calibration directory instead of the dedicated cache dir the other
    three correctly use."""
    import inspect
    from PyAPS import aps_utils

    class_source = inspect.getsource(aps_utils.APSOB)
    # Both __init__'s two join_arms-branch call sites and
    # _process_single_arm_vectorized's two call sites, across the class.
    pickle_dir_lines = [line for line in class_source.splitlines()
                         if line.strip().startswith("pickle_dir") and "=" in line]
    assert len(pickle_dir_lines) == 4, (
        f"expected 4 pickle_dir= call sites across APSOB, found {len(pickle_dir_lines)} "
        "-- this test needs updating if that count genuinely changed"
    )
    assert not any("self._caldir" in line for line in pickle_dir_lines), (
        "a pickle_dir= call site still resolves to self._caldir -- LSF/FWHM cache "
        "would be written into the calibration directory again"
    )


def test_no_sys_exit_reachable_from_the_explorers_own_l1_load_path():
    """A second, much larger reliability finding than the single
    configdir/caldate sys.exit() the aps_explorer.py plan set out to fix:
    aps_utils.py had 12 sys.exit() calls total. sys.exit() raises
    SystemExit, a BaseException -- not caught by any `except Exception`
    handler anywhere in this codebase -- so any one of them being hit
    kills the entire gunicorn worker process in server mode (MULTI_SESSION),
    not just the one bad request; with --workers 1 a hard requirement of
    that design, that takes the whole server down for every connected
    user over something as ordinary as one L1 file with an unexpected
    CAMERA header value.

    Traced each occurrence to its enclosing function via source
    inspection (not assumed) before fixing: 11 of the 12 live in
    l1_fileinfo() (called directly by aps_explorer.py itself, including
    its own L1 defense-in-depth check), gen_targlist() (called from
    inside APSOB.__init__ -- every single L1 load reaches it), and
    read_infiles_list() -- all genuinely reachable from the explorer's
    own load path, all fixed to `raise RuntimeError(...)` instead. The
    12th, inside APSOB.pack_2_redrock(), is confirmed pipeline-only
    (never called by aps_explorer.py's own code path -- see the
    dependency-split audit from earlier this same project) and
    deliberately left alone; this test excludes exactly that one
    function, not the whole file, so a *new* sys.exit() introduced
    elsewhere in this file still fails this test."""
    import inspect
    from PyAPS import aps_utils

    full_source = inspect.getsource(aps_utils)
    pack_2_redrock_source = inspect.getsource(aps_utils.APSOB.pack_2_redrock)
    non_pipeline_source = full_source.replace(pack_2_redrock_source, "")

    offending_lines = [line for line in non_pipeline_source.splitlines()
                        if not line.strip().startswith("#") and "sys.exit(" in line]
    assert offending_lines == [], (
        f"found sys.exit() outside APSOB.pack_2_redrock() (pipeline-only, "
        f"deliberately excluded): {offending_lines!r} -- SystemExit isn't caught "
        "by this codebase's `except Exception` handlers and would kill the whole "
        "server process in multi-session mode; convert to `raise RuntimeError(...)`"
    )
