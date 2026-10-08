"""Regression tests for the four L2 job failure families seen on a dev host running
Python 3.12 with NumPy 2.5 (no WEAVE data needed, all inputs are tiny synthetic arrays/files).

1. RVS jobs: the RVS template directory came only from ``template_lib`` in
   ``configs/rvs_config.yaml`` and ignored ``templates_RVS`` of the script_params file.
2. FERRE jobs: ran (SLURM ``afterany``) after a failed RVS job and died with a raw
   ``FileNotFoundError`` deep inside astropy.
3. PPXF jobs: ``ExGalPrepare.prepare_spec_file`` stored a size-1 array in a scalar element,
   which NumPy 2 rejects ("setting an array element with a sequence").
4. IFU galaxy jobs: a cached LSF/FWHM ``.dill`` written under another Python minor version
   loaded fine and then failed when called ("SystemError: error return without exception set").

Run with:
    cd <PYAPS_DIR> && pytest tests/test_l2_job_failures.py -v
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path

import numpy as np
import pytest

from PyAPS.aps_utils import RVS_TEMPLATES_ENV, resolve_rvs_template_lib


# --------------------------------------------------------------------------------------
# 3. NumPy 2: scalar element assignment in the Voronoi / MOS spectrum preparation
# --------------------------------------------------------------------------------------
@pytest.fixture
def scalar_conversion_is_error():
    """NumPy 2.x raises for size-1 arrays stored in scalar elements; NumPy 1.25-2.3 only
    warn. Turn the warning into an error so the tests fail on either."""
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*ndim > 0 to a scalar.*")
        yield


def test_prepare_spec_file_one_spectrum_per_bin(tmp_path, scalar_conversion_is_error):
    ExGalPrepare = pytest.importorskip("PyAPS.ExGalPrepare")
    from astropy.io import fits

    npix, nbins = 8, 3
    spec = np.arange(npix * nbins, dtype=float).reshape(npix, nbins) + 1.0
    espec = np.ones_like(spec)
    wave = np.tile(np.linspace(4000.0, 4100.0, npix)[:, None], (1, nbins))
    bin_num = np.arange(nbins)  # MOS: every spectrum is its own bin

    ExGalPrepare.prepare_spec_file(
        bin_num, spec, espec, "demo", str(tmp_path) + os.sep, wave, "lin", verbose=False
    )

    out = tmp_path / "demo_BINSpectra_linear.fits"
    assert out.exists()
    with fits.open(out) as hdul:
        data = hdul["BIN_SPECTRA"].data
        np.testing.assert_allclose(np.asarray(data["SPEC"]).T, spec)
        np.testing.assert_allclose(np.asarray(data["ESPEC"]).T, espec)


def test_voronoi_binning_single_and_multi_spaxel_bins(scalar_conversion_is_error):
    ExGalPrepare = pytest.importorskip("PyAPS.ExGalPrepare")

    npix = 6
    spec = np.arange(npix * 4, dtype=float).reshape(npix, 4) + 1.0
    err = np.ones_like(spec)
    bin_num = np.array([0, 1, 1, 2])  # bin 0 and 2: one spaxel, bin 1: two spaxels

    bin_data, bin_error, bin_flux = ExGalPrepare.voronoi_binning(bin_num, spec, err)

    assert bin_data.shape == (npix, 3) and bin_flux.shape == (3,)
    np.testing.assert_allclose(bin_data[:, 0], spec[:, 0])
    np.testing.assert_allclose(bin_data[:, 1], spec[:, 1] + spec[:, 2])
    np.testing.assert_allclose(bin_flux, [spec[:, 0].mean(), (spec[:, 1] + spec[:, 2]).mean(),
                                          spec[:, 3].mean()])


# --------------------------------------------------------------------------------------
# 1. RVS template directory resolution
# --------------------------------------------------------------------------------------
@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv(RVS_TEMPLATES_ENV, raising=False)
    monkeypatch.delenv("PYAPS_HOME", raising=False)
    return monkeypatch


def _mk(tmp_path, *parts):
    p = tmp_path.joinpath(*parts)
    p.mkdir(parents=True)
    return p


def test_env_templates_dir_wins_over_config_template_lib(tmp_path, clean_env):
    configured = _mk(tmp_path, "shared", "templates_RVS")
    in_cfg = _mk(tmp_path, "other", "templates_RVS")
    clean_env.setenv(RVS_TEMPLATES_ENV, str(configured))
    got = resolve_rvs_template_lib(str(in_cfg) + "/", verbose=False)
    assert Path(got) == configured
    assert got.endswith(os.sep)


def test_config_template_lib_used_when_no_env(tmp_path, clean_env):
    in_cfg = _mk(tmp_path, "lib")
    assert Path(resolve_rvs_template_lib(str(in_cfg), verbose=False)) == in_cfg


def test_env_value_is_expanded(tmp_path, clean_env):
    tdir = _mk(tmp_path, "t")
    clean_env.setenv("MY_TEST_TEMPLATE_ROOT", str(tmp_path))
    clean_env.setenv(RVS_TEMPLATES_ENV, "$MY_TEST_TEMPLATE_ROOT/t")
    assert Path(resolve_rvs_template_lib(None, verbose=False)) == tdir


def test_missing_configured_directory_is_an_error_naming_the_key(tmp_path, clean_env):
    clean_env.setenv(RVS_TEMPLATES_ENV, str(tmp_path / "nope"))
    with pytest.raises(RuntimeError, match="templates_RVS"):
        resolve_rvs_template_lib("/also/missing/", verbose=False)
    clean_env.delenv(RVS_TEMPLATES_ENV)
    with pytest.raises(RuntimeError, match="template_lib"):
        resolve_rvs_template_lib(str(tmp_path / "nope"), verbose=False)


def test_unset_placeholder_is_an_error_not_a_fallback(tmp_path, clean_env):
    """No environment value and the tracked placeholder unresolved: stop, never search
    other locations (even if a templates_RVS directory sits under PYAPS_HOME)."""
    _mk(tmp_path, "PyAPS_templates", "templates_RVS")
    clean_env.setenv("PYAPS_HOME", str(tmp_path))
    with pytest.raises(RuntimeError, match="templates_RVS"):
        resolve_rvs_template_lib("${PYAPS_RVS_TEMPLATES}", verbose=False)
    with pytest.raises(RuntimeError):
        resolve_rvs_template_lib(None, verbose=False)


def test_tracked_rvs_config_refers_to_the_configured_variable():
    cfg = Path(__file__).resolve().parent.parent / "configs" / "rvs_config.yaml"
    first = [ln for ln in cfg.read_text().splitlines() if ln.startswith("template_lib")]
    assert first and "${" + RVS_TEMPLATES_ENV + "}" in first[0]


def test_read_config_resolves_template_lib(tmp_path, clean_env):
    """The reader used by aps_rvs and aps_ifu_rvs must hand the configured directory to
    rvspecfit (the 'Filename .../ccf_*.h5 does not exist' failure)."""
    pytest.importorskip("rvspecfit")
    import PyAPS.aps_ifu_rvs as aps_ifu_rvs
    import PyAPS.aps_rvs as aps_rvs

    configured = _mk(tmp_path, "shared", "templates_RVS")
    cfg = tmp_path / "rvs_config.yaml"
    cfg.write_text("template_lib: '${PYAPS_RVS_TEMPLATES}'\nmin_vel: -1000\n")
    clean_env.setenv(RVS_TEMPLATES_ENV, str(configured))
    for reader in (aps_rvs.read_config_APS_RVS, aps_ifu_rvs.read_config_APS_RVS):
        conf = reader(str(cfg))
        assert Path(conf["template_lib"]) == configured
        assert conf["min_vel"] == -1000
    clean_env.delenv(RVS_TEMPLATES_ENV)
    for reader in (aps_rvs.read_config_APS_RVS, aps_ifu_rvs.read_config_APS_RVS):
        with pytest.raises(RuntimeError, match="templates_RVS"):
            reader(str(cfg))


def test_job_scripts_export_configured_rvs_templates(tmp_path):
    """write_bash exports templates_RVS of the script_params file in use, quoted."""
    aps_runner = pytest.importorskip("PyAPS.aps_runner")

    tdir = tmp_path / "templates with space" / "templates_RVS"
    tdir.mkdir(parents=True)
    conf = {"use_venv": "False", "templates_RVS": str(tdir)}
    script = tmp_path / "job.sh"
    aps_runner.write_bash("head", script, "echo hi", "JOB", conf, log=False,
                          logs_path=tmp_path / "logs")
    lines = script.read_text().splitlines()
    exports = [ln for ln in lines if ln.startswith("export " + RVS_TEMPLATES_ENV + "=")]
    assert len(exports) == 1
    value = shlex.split(exports[0].split("=", 1)[1])[0]
    assert value == str(tdir)

    # not set / 'NONE': no export, no crash
    for off in (None, "None", ""):
        conf = {"use_venv": "False", "templates_RVS": off}
        aps_runner.write_bash("head", script, "echo hi", "JOB", conf, log=False,
                              logs_path=tmp_path / "logs")
        assert RVS_TEMPLATES_ENV not in script.read_text()


# --------------------------------------------------------------------------------------
# 2. FERRE without its RVS input
# --------------------------------------------------------------------------------------
def test_ferre_fails_cleanly_when_rvs_output_is_missing(tmp_path):
    aps_ferre = pytest.importorskip("PyAPS.aps_ferre")
    missing = tmp_path / "rvs_single_x.fits"
    with pytest.raises(SystemExit) as exc:
        aps_ferre.check_rvs_input(str(missing))
    msg = str(exc.value)
    assert "RVS" in msg and str(missing) in msg and "FileNotFoundError" not in msg
    with pytest.raises(SystemExit):
        aps_ferre.check_rvs_input(None)

    present = tmp_path / "rvs_single_y.fits"
    present.write_bytes(b"")
    aps_ferre.check_rvs_input(str(present))  # no exception


# --------------------------------------------------------------------------------------
# 4. IFU galaxy jobs: LSF/FWHM cache written under another Python minor version
# --------------------------------------------------------------------------------------
class _FakeInterp:
    """Stand-in for a loaded LSF/FWHM interpolator (only the attributes the check reads)."""

    def __init__(self, func, stamp="current"):
        from PyAPS import aps_lsf

        self.interpolator_dict = {"global": {"interpolate_function": func,
                                             "wavelength_range": (3800.0, 9300.0)}}
        if stamp == "current":
            self._cache_python = aps_lsf._python_minor()
        elif stamp is not None:
            self._cache_python = stamp


def test_lsf_cache_with_working_closure_is_usable():
    from PyAPS import aps_lsf

    ok, why = aps_lsf._lsf_cache_is_usable(_FakeInterp(lambda w: np.full_like(w, 2.0)))
    assert ok and why == ""
    # a cache from before the stamp existed but with a working closure is kept as is
    ok, _ = aps_lsf._lsf_cache_is_usable(_FakeInterp(lambda w: w, stamp=None))
    assert ok


def test_lsf_cache_from_another_python_minor_is_rejected():
    from PyAPS import aps_lsf

    major, minor = aps_lsf._python_minor()
    ok, why = aps_lsf._lsf_cache_is_usable(_FakeInterp(lambda w: w, stamp=(major, minor - 1)))
    assert not ok and "Python" in why


def test_lsf_cache_whose_closure_fails_when_called_is_rejected():
    """What a Python 3.11 closure does under 3.12: unpickles fine, then
    'SystemError: error return without exception set' on the first call."""
    from PyAPS import aps_lsf

    def broken(_w):
        raise SystemError("error return without exception set")

    ok, why = aps_lsf._lsf_cache_is_usable(_FakeInterp(broken, stamp=None))
    assert not ok and "SystemError" in why


def test_run_lsf_analysis_rebuilds_an_unusable_cache(tmp_path, monkeypatch):
    from PyAPS import aps_lsf

    cal = tmp_path / "BLUEL11_cal.fits"
    cal.write_bytes(b"")
    pickle_path = aps_lsf.get_output_pickle_path([str(cal)], str(tmp_path))
    Path(pickle_path).write_bytes(b"stub")

    def broken(_w):
        raise SystemError("error return without exception set")

    bad = _FakeInterp(broken, stamp=None)
    bad._cache_format_version = aps_lsf._LSF_CACHE_FORMAT_VERSION
    bad.set_debug = lambda *_a, **_k: None
    bad.print_summary = lambda: None
    monkeypatch.setattr(aps_lsf.LSFInterpolator, "load", classmethod(lambda cls, p: bad))

    built = []

    def fake_create(self, files, **kw):
        built.append(files)
        return False  # stop right after the rebuild decision

    monkeypatch.setattr(aps_lsf.LSFInterpolator, "create_from_files", fake_create)
    result = aps_lsf.run_lsf_analysis([str(cal)], figdir=None, make_plot=False,
                                      pickle_dir=str(tmp_path), overwrite=False)
    assert built, "an unusable cache must trigger a rebuild"
    assert result is None  # the fake rebuild reports failure


# --------------------------------------------------------------------------------------
# 5. Failure visibility in the runner log
# --------------------------------------------------------------------------------------
def _stage_script(tmp_path, runner_log, post_command=None, command="bash -c 'exit 3'"):
    import subprocess

    aps_runner = pytest.importorskip("PyAPS.aps_runner")
    conf = {"use_venv": "False", "templates_RVS": None, "runner_log": runner_log}
    logs = tmp_path / "logs"
    script = tmp_path / "stage.sh"
    aps_runner.write_bash("head", script, command, "abc123", conf, log=False, logs_path=logs,
                          log_prefix="MOS_RVS", post_command=post_command)
    text = script.read_text().replace("sleep 30\n", "true\n")   # do not wait in the test
    script.write_text(text)
    return script, subprocess


def test_failed_stage_appends_error_line_to_runner_log(tmp_path):
    log = tmp_path / "runner.log"
    script, subprocess = _stage_script(tmp_path, str(log))
    env = dict(os.environ, SLURM_JOB_ID="4711", SLURM_JOB_NAME="RVS_L2_abc123")
    subprocess.run(["bash", str(script)], env=env, capture_output=True)
    lines = log.read_text().splitlines()
    assert len(lines) == 1
    line = lines[0]
    assert "ERROR: SLURM job 4711 RVS_L2_abc123 failed with exit code 3" in line
    assert str(tmp_path / "logs" / "RVS_L2_abc123.4711.err") in line
    assert line.startswith("[") and line[5] == "-" and line[20] == "]"


def test_successful_stage_writes_nothing_and_off_switch_works(tmp_path):
    log = tmp_path / "runner.log"
    script, subprocess = _stage_script(tmp_path, str(log), command="true")
    subprocess.run(["bash", str(script)], capture_output=True)
    assert not log.exists()
    off = tmp_path / "off"
    off.mkdir()
    script2, subprocess = _stage_script(off, "None")
    assert "export PYAPS_RUNNER_LOG" not in script2.read_text()


def test_post_command_runs_and_never_changes_the_exit_status(tmp_path):
    script, subprocess = _stage_script(tmp_path, "None", post_command="false", command="true")
    res = subprocess.run(["bash", str(script)], capture_output=True)
    assert res.returncode == 0
    marker = tmp_path / "ran"
    script, subprocess = _stage_script(tmp_path, "None", post_command=f"touch {marker}", command="bash -c 'exit 4'")
    res = subprocess.run(["bash", str(script)], capture_output=True)
    assert marker.exists() and res.returncode == 4


SACCT_SAMPLE = """\
101|RR_L2_abc123|COMPLETED|0:0
102|RVS_L2_abc123|TIMEOUT|0:0
103|FR_L2_abc123|CANCELLED by 1000|0:15
104|PPXF_L2_abc123|FAILED|1:0
105|PPXF_L2_other999|TIMEOUT|0:0
106|EMI_L2_abc123|OUT_OF_MEMORY|0:125
"""


def test_job_report_lists_killed_jobs_of_the_tag_only(tmp_path):
    from PyAPS import aps_job_report

    rows = aps_job_report.parse_sacct(SACCT_SAMPLE, "abc123")
    assert [(r["job_id"], r["state"]) for r in rows] == [("102", "TIMEOUT"), ("103", "CANCELLED"),
                                                          ("106", "OUT_OF_MEMORY")]
    log = tmp_path / "runner.log"
    lines = aps_job_report.report("abc123", tmp_path / "logs", runner_log=str(log), sacct_text=SACCT_SAMPLE)
    assert len(lines) == 3 and log.read_text().splitlines() == lines
    assert "ERROR: SLURM job 102 RVS_L2_abc123 TIMEOUT exit code 0:0; stderr: " in lines[0]
    assert lines[0].endswith(str(tmp_path / "logs" / "RVS_L2_abc123.102.err"))
    assert aps_job_report.report("nomatch", tmp_path, sacct_text=SACCT_SAMPLE) == []


def test_mos_l2merge_script_calls_the_job_report():
    text = (Path(__file__).resolve().parent.parent / "py" / "PyAPS" / "aps_runner.py").read_text()
    assert "aps_job_report.py" in text and "post_command=post_cmd" in text
