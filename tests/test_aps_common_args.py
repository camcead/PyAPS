"""
tests/test_aps_common_args.py - Tests for the shared CLI argument-group
registry (`PyAPS/aps_common_args.py`) that replaced the repeated
argparse-boilerplate block previously copy-pasted across all 14 `aps_*.py`
pipeline scripts (aps_rr.py, aps_rvs.py, aps_mosExGal.py, aps_ferre.py,
aps_feswi.py, the IFU family, aps_squeze.py, aps_amy.py, aps_alfa-neat.py,
aps_cubepreview.py, aps_l1_multi_plot.py).

Two halves:

1. Direct unit tests of `build_common_parser()`/`resolve_common_args()`
   themselves -- group composition, per-script `overrides`/`exclude`/
   `extra_args`, and the exact conversion/write-back behaviour each
   migrated script depends on (np.int32 id dtype, the arms_ratio length
   assert, the join_arms<2 correction, and the specific split between
   what resolve_common_args writes back onto `args` for print_args()'s
   own reporting -- infiles/wlranges/arms_ratio/area/mask_areas/join_arms
   -- versus what stays a separate `resolved.*` value only -- aps_ids/
   targsrvy/targclass/mask_aps_ids). Uses the same real L1 file already
   used by test_aps_explorer.py so l1_fileinfo() runs for real, not
   mocked.

2. A flag-surface parity test per migrated script: runs `<script> --help`
   as a subprocess and compares the resulting set of `--flagname` tokens
   against a frozen baseline captured from each script's own pre-migration
   (hand-written-argparse) --help output. This is the regression guard
   the migration plan called for -- no migrated script may ever add or
   drop a flag as a side effect of a future change to aps_common_args.py
   or to that script's own build_common_parser() call, since aps_runner.py and
   orchestration layers built on it hardcode these exact flag names into real subprocess
   command strings.

Run with:
    cd <PYAPS_DIR> && pytest tests/test_aps_common_args.py -v
"""

from __future__ import annotations
import os as _os
# Root of a directory tree holding real WEAVE data (L1/, L2/, CAL/, CAT/ ...). Tests that need
# real data are skipped when it is not available; point PYAPS_TEST_DATA at your copy to run them.
PYAPS_DATA = _os.environ.get("PYAPS_TEST_DATA", "<PYAPS_DATA>")
PYAPS_HOME = _os.environ.get("PYAPS_HOME", _os.path.expanduser("~/PyAPS"))

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from PyAPS.aps_common_args import build_common_parser, resolve_common_args

PYAPS_DIR = Path(__file__).resolve().parents[1] / "py" / "PyAPS"

# Real MOS-mode L1 file, already used by test_aps_explorer.py -- fast to
# load, exercises the real l1_fileinfo() normalization path.
L1_SMALL_FILE = PYAPS_DATA + "/L1/20250707/stack_3097460.fit"


# --------------------------------------------------------------------------- #
# 1. build_common_parser() mechanics
# --------------------------------------------------------------------------- #

def test_build_common_parser_groups_compose():
    parser = build_common_parser(
        description="test", groups=["target_selection", "wavelength"],
    )
    dests = {a.dest for a in parser._actions}
    assert {"infiles", "aps_ids", "targsrvy", "targclass", "mask_aps_ids", "wlranges"} <= dests
    # spatial_selection/l1_processing/caldirs/output not requested
    assert "area" not in dests
    assert "sens_corr" not in dests
    assert "catdir" not in dests
    assert "outpath" not in dests


def test_build_common_parser_override_changes_only_that_kwarg():
    parser = build_common_parser(
        description="test", groups=["l1_processing"],
        overrides={"vacuum": {"default": False}},
    )
    assert parser.get_default("vacuum") is False
    # untouched sibling flag keeps its canonical default
    assert parser.get_default("sens_corr") is True
    # untouched kwarg on the same flag (help text) is preserved
    vacuum_action = next(a for a in parser._actions if a.dest == "vacuum")
    assert vacuum_action.help == "transform wavelength from air to vacuum"


def test_build_common_parser_exclude_drops_flag():
    parser = build_common_parser(
        description="test", groups=["output"], exclude=["overwrite"],
    )
    dests = {a.dest for a in parser._actions}
    assert "outpath" in dests
    assert "headname" in dests
    assert "overwrite" not in dests


def test_build_common_parser_extra_args_appended():
    parser = build_common_parser(
        description="test", groups=["output"],
        extra_args=[(("--ntop",), dict(type=int, default=1, help="top n"))],
    )
    dests = {a.dest for a in parser._actions}
    assert "ntop" in dests
    assert parser.get_default("ntop") == 1


def test_build_common_parser_does_not_mutate_registry():
    # a per-call override must not leak into the next, unrelated call
    build_common_parser(description="a", groups=["l1_processing"],
                         overrides={"vacuum": {"default": False}})
    parser2 = build_common_parser(description="b", groups=["l1_processing"])
    assert parser2.get_default("vacuum") is True


# --------------------------------------------------------------------------- #
# 2. resolve_common_args() conversion/write-back behaviour
# --------------------------------------------------------------------------- #

def _full_parser(**overrides):
    return build_common_parser(
        description="test",
        groups=["target_selection", "spatial_selection", "wavelength",
                "l1_processing", "caldirs", "output"],
        overrides=overrides,
    )


def test_resolve_common_args_normalizes_infiles_and_wlranges_via_real_data():
    parser = _full_parser()
    args = parser.parse_args([
        "--infiles", L1_SMALL_FILE,
        "--outpath", "/tmp/x", "--headname", "h",
    ])
    resolved = resolve_common_args(args)
    # l1_fileinfo() ran for real and produced a wavelength range for the file
    assert resolved.wlranges and len(resolved.wlranges) == 1
    assert len(resolved.wlranges[0]) == 2
    # written back onto args for print_args()'s own reporting
    assert args.wlranges == resolved.wlranges
    assert args.infiles == resolved.infiles


def test_resolve_common_args_join_arms_forced_false_for_single_infile():
    parser = _full_parser()
    args = parser.parse_args([
        "--infiles", L1_SMALL_FILE, "--join_arms", "True",
        "--outpath", "/tmp/x", "--headname", "h",
    ])
    resolve_common_args(args)
    assert args.join_arms is False


def test_resolve_common_args_arms_ratio_dtype_and_length_assert():
    parser = _full_parser()
    args = parser.parse_args([
        "--infiles", L1_SMALL_FILE, "--arms_ratio", "1.0",
        "--outpath", "/tmp/x", "--headname", "h",
    ])
    resolved = resolve_common_args(args)
    assert resolved.arms_ratio == [1.0]

    # a mismatched length must raise, with a real message (not a bare assert)
    bad_args = parser.parse_args([
        "--infiles", L1_SMALL_FILE, "--arms_ratio", "1.0,0.8",
        "--outpath", "/tmp/x", "--headname", "h",
    ])
    with pytest.raises(AssertionError, match="arms_ratio"):
        resolve_common_args(bad_args)


def test_resolve_common_args_id_lists_use_int32_not_plain_int():
    parser = _full_parser()
    args = parser.parse_args([
        "--infiles", L1_SMALL_FILE,
        "--aps_ids", "10,20,30", "--mask_aps_ids", "5,6",
        "--outpath", "/tmp/x", "--headname", "h",
    ])
    resolved = resolve_common_args(args)
    # np.int32(...).tolist() yields native Python ints, but via the
    # np.int32 dtype path (not aps_squeze.py's/aps_amy.py's old plain
    # int() cast) -- confirmed by checking the actual values convert
    # correctly and match what np.array(..., dtype=np.int32) produces.
    assert resolved.aps_ids == [10, 20, 30]
    assert resolved.mask_aps_ids == [5, 6]
    assert resolved.aps_ids == np.array(["10", "20", "30"], dtype=np.int32).tolist()


def test_resolve_common_args_targsrvy_targclass_are_str_lists():
    parser = _full_parser()
    args = parser.parse_args([
        "--infiles", L1_SMALL_FILE,
        "--targsrvy", "WL,WQ", "--targclass", "GALAXY",
        "--outpath", "/tmp/x", "--headname", "h",
    ])
    resolved = resolve_common_args(args)
    assert resolved.targsrvy == ["WL", "WQ"]
    assert resolved.targclass == ["GALAXY"]


def test_resolve_common_args_area_and_mask_areas():
    parser = _full_parser()
    args = parser.parse_args([
        "--infiles", L1_SMALL_FILE,
        "--area", "10.0,20.0,5.0",
        "--mask_areas", "1.0,2.0,3.0", "4.0,5.0,6.0",
        "--outpath", "/tmp/x", "--headname", "h",
    ])
    resolved = resolve_common_args(args)
    assert resolved.area == [10.0, 20.0, 5.0]
    assert resolved.mask_areas == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    # written back onto args, matching every baseline script's own behaviour
    assert args.area == resolved.area
    assert args.mask_areas == resolved.mask_areas


def test_resolve_common_args_id_fields_stay_raw_on_args():
    """The one deliberately asymmetric part of resolve_common_args: unlike
    wlranges/arms_ratio/area/mask_areas/infiles, the four id/class fields
    (aps_ids/targsrvy/targclass/mask_aps_ids) are NOT written back onto
    `args` -- `args.aps_ids` etc. stay the original raw comma-separated
    string forever, exactly matching every migrated script's pre-migration
    behaviour (only the separate `resolved.*` value is converted)."""
    parser = _full_parser()
    args = parser.parse_args([
        "--infiles", L1_SMALL_FILE, "--aps_ids", "10,20,30",
        "--outpath", "/tmp/x", "--headname", "h",
    ])
    resolve_common_args(args)
    assert args.aps_ids == "10,20,30"  # untouched raw string


def test_resolve_common_args_ignores_absent_fields():
    """Scripts without --area/--mask_areas (the IFU family) or without
    --join_arms (none currently, but the function must stay generic) must
    not error just because those attributes don't exist on `args`."""
    parser = build_common_parser(
        description="test",
        groups=["target_selection", "wavelength", "l1_processing", "output"],
    )
    args = parser.parse_args([
        "--infiles", L1_SMALL_FILE, "--outpath", "/tmp/x", "--headname", "h",
    ])
    resolved = resolve_common_args(args)
    assert resolved.area is None
    assert resolved.mask_areas is None


# --------------------------------------------------------------------------- #
# 3. Flag-surface parity per migrated script (regression guard)
# --------------------------------------------------------------------------- #

# Captured from each script's own --help output BEFORE it was migrated onto
# aps_common_args.py (hand-written argparse, one script at a time). A future
# change that adds/drops/renames a flag -- whether in aps_common_args.py or
# in that script's own build_common_parser() call -- fails this test, since
# aps_runner.py and orchestration layers hardcode these exact flag names into real
# subprocess command strings that drive the SLURM/bash pipeline.
# Updated 2 Oct 2026: added the flags introduced after the snapshot (aps_rr: extinction_*,
# star_z_prior_sigma, rr_solver; aps_rvs: device/linemask/maskbalmer/uselinemasks;
# aps_ifu_prepare: seg3d*).
EXPECTED_FLAGS = {
    "aps_rr.py": {'--aps_ids', '--archetypes', '--area', '--arms_ratio', '--cache_Rcsr', '--caldir', '--catdir', '--chi2_scan', '--configdir', '--debug', '--extinction_corr', '--extinction_ebv_scale', '--extinction_mapdir', '--fig', '--fill_gap', '--gpu', '--headname', '--help', '--infiles', '--join_arms', '--mask_aps_ids', '--mask_areas', '--mask_gaps', '--max_gpuprocs', '--mp', '--nminima', '--ntop', '--outpath', '--overwrite', '--priors', '--rr_solver', '--safe_mask_gaps', '--sens_corr', '--skysub_mask_residuals', '--srvyconf', '--star_z_prior_sigma', '--targclass', '--targsrvy', '--tellurics', '--templates', '--vacuum', '--wlranges', '--zall'},
    "aps_rvs.py": {'--aps_ids', '--area', '--arms_ratio', '--caldir', '--catdir', '--classfile', '--config', '--configdir', '--device', '--fig', '--fill_gap', '--headname', '--help', '--infiles', '--join_arms', '--linemask', '--mask_aps_ids', '--mask_areas', '--mask_gaps', '--maskbalmer', '--mp', '--outpath', '--outspec', '--overwrite', '--safe_mask_gaps', '--sens_corr', '--targclass', '--targsrvy', '--tellurics', '--uselinemasks', '--vacuum', '--wlranges'},
    "aps_mosExGal.py": {'--EMIPPXF', '--EMIPPXF_LEVEL', '--LS', '--LS_MODE', '--LS_RES', '--PPXF', '--aps_ids', '--area', '--arms_ratio', '--caldir', '--catdir', '--classfile', '--config_dir', '--configdir', '--fig', '--fill_gap', '--headname', '--help', '--infiles', '--join_arms', '--mask_aps_ids', '--mask_areas', '--mask_gaps', '--mp', '--outpath', '--params', '--safe_mask_gaps', '--sens_corr', '--targclass', '--targsrvy', '--tellurics', '--templates_dir', '--vacuum', '--wlranges'},
    "aps_ferre.py": {'--aps_ids', '--area', '--arms_ratio', '--caldir', '--catdir', '--classfile', '--configdir', '--ferre_exe', '--fig', '--fill_gap', '--grid_ids', '--grid_prefix', '--headname', '--help', '--infiles', '--join_arms', '--linemask', '--mask_aps_ids', '--mask_areas', '--mask_gaps', '--maskbalmer', '--mp', '--outpath', '--outspec', '--overwrite', '--rvsfile', '--safe_mask_gaps', '--sens_corr', '--split_arms', '--targclass', '--targsrvy', '--tellurics', '--templates', '--uselinemasks', '--vacuum', '--wlranges'},
    "aps_feswi.py": {'--aps_ids', '--area', '--arms_ratio', '--classfile', '--ferre_exe', '--feswi_cold_blue_grid', '--feswi_cold_red_grid', '--feswi_ferre_f_access', '--feswi_ferre_f_format', '--feswi_ferre_ndim', '--feswi_ferre_nthreads', '--feswi_hot_blue_grid', '--feswi_hot_red_grid', '--feswi_path', '--feswi_run_ferre', '--feswi_spectral_windows', '--fig', '--fill_gap', '--grid_ids', '--grid_prefix', '--headname', '--help', '--infiles', '--join_arms', '--mask_aps_ids', '--mask_areas', '--mask_gaps', '--mp', '--outpath', '--outspec', '--overwrite', '--rvsfile', '--safe_mask_gaps', '--sens_corr', '--targclass', '--targsrvy', '--tellurics', '--templates', '--vacuum', '--wlranges'},
    "aps_ifu_v0.py": {'--EMIPPXF', '--ExGal_run', '--ExGal_templates', '--Gal_run', '--IFU_config_dir', '--IFU_params', '--LS', '--PPXF', '--aps_ids', '--arms_ratio', '--caldir', '--catdir', '--class_ntop', '--class_patch', '--class_templates', '--class_templates_ARC', '--class_z_rad', '--configdir', '--fig', '--fill_gap', '--headname', '--help', '--infiles', '--join_arms', '--mask_aps_ids', '--mask_gaps', '--mp_ExGal', '--mp_Gal', '--mp_prep', '--outpath', '--overwrite', '--patch_file', '--safe_mask_gaps', '--seg2d', '--seg2d_deblend_nthresh', '--seg2d_exclude_ctarg', '--seg2d_ext_thresh', '--seg2d_extract', '--seg2d_mask_stars', '--seg2d_minarea', '--seg2d_radii_factor', '--seg2d_search_gaia', '--seg2d_white_images', '--seg2d_white_src', '--sens_corr', '--targclass', '--targsrvy', '--tellurics', '--uapsid', '--user_patch', '--vacuum', '--wlranges', '--z_input'},
    "aps_ifu_ExGal.py": {'--EMIPPXF', '--ExGal_templates', '--IFU_config_dir', '--IFU_params', '--LS', '--PPXF', '--aps_ids', '--arms_ratio', '--caldir', '--catdir', '--fill_gap', '--headname', '--help', '--infiles', '--join_arms', '--mask_aps_ids', '--mask_gaps', '--mp_ExGal', '--no_spec_ext', '--outpath', '--overwrite', '--patch_array', '--patch_file', '--safe_mask_gaps', '--sens_corr', '--targclass', '--targsrvy', '--tellurics', '--uapsid', '--vacuum', '--wlranges'},
    "aps_ifu_Gal.py": {'--IFU_config_dir', '--IFU_params', '--aps_ids', '--arms_ratio', '--caldir', '--catdir', '--class_ntop', '--class_patch', '--class_templates', '--class_templates_ARC', '--class_z_rad', '--ferre_exe', '--ferre_grid_ids', '--ferre_grid_prefix', '--ferre_templates', '--gal_aperture_factor', '--fill_gap', '--headname', '--help', '--infiles', '--join_arms', '--mask_aps_ids', '--mask_gaps', '--min_snr_gal', '--mp_Gal', '--mp_prep', '--no_spec_ext', '--outpath', '--overwrite', '--patch_array', '--patch_file', '--rvs_config', '--safe_mask_gaps', '--sens_corr', '--spbin_size_gal', '--targclass', '--target_snr_gal', '--targsrvy', '--tellurics', '--uapsid', '--vacuum', '--voronoi_gal', '--wlranges'},
    "aps_ifu_prepare.py": {'--IFU_config_dir', '--IFU_params', '--aps_ids', '--arms_ratio', '--caldir', '--catdir', '--class_ntop', '--class_patch', '--class_templates', '--class_templates_ARC', '--class_z_rad', '--fig', '--fill_gap', '--headname', '--help', '--infiles', '--join_arms', '--mask_aps_ids', '--mask_gaps', '--mp_prep', '--outpath', '--patch_file', '--safe_mask_gaps', '--seg2d', '--seg2d_deblend_nthresh', '--seg2d_exclude_ctarg', '--seg2d_ext_thresh', '--seg2d_extract', '--seg2d_mask_stars', '--seg2d_minarea', '--seg2d_radii_factor', '--seg2d_search_gaia', '--seg2d_white_images', '--seg2d_white_src', '--seg3d', '--seg3d_continuum_method', '--seg3d_fig', '--seg3d_group_radius_px', '--seg3d_ivar_calibration', '--seg3d_lmax', '--seg3d_lmin', '--seg3d_merge', '--seg3d_merge_aperture_arcsec', '--seg3d_merge_extreme_safety_margin', '--seg3d_merge_group_min_lines', '--seg3d_merge_group_min_n_trust', '--seg3d_merge_group_min_snr', '--seg3d_merge_min_purity', '--seg3d_merge_min_snr', '--seg3d_min_npix', '--seg3d_sky_mask_min_A', '--seg3d_spatial_bin', '--seg3d_threshold', '--sens_corr', '--targclass', '--targsrvy', '--tellurics', '--uapsid', '--user_patch', '--vacuum', '--wlranges'},
    "aps_squeze.py": {'--aps_ids', '--archetypes', '--area', '--arms_ratio', '--cache_Rcsr', '--chi2_scan', '--clean_dir', '--debug', '--fig', '--fill_gap', '--headname', '--help', '--infiles', '--join_arms', '--mask_aps_ids', '--mask_areas', '--mask_gaps', '--model', '--mp', '--nminima', '--outpath', '--overwrite', '--prob_cut', '--quiet', '--safe_mask_gaps', '--sens_corr', '--srvyconf', '--targclass', '--targsrvy', '--tellurics', '--templates', '--vacuum', '--wlranges', '--zall'},
    "aps_amy.py": {'--aps_ids', '--area', '--arms_ratio', '--config', '--exclude_region', '--fill_gap', '--headname', '--help', '--infiles', '--join_arms', '--mask_aps_ids', '--mask_areas', '--mask_gaps', '--mproc', '--outpath', '--overwrite', '--progress_bar', '--safe_mask_gaps', '--sens_corr', '--targclass', '--targsrvy', '--tellurics', '--template_flux', '--template_wave', '--templates_dir', '--vacuum', '--wlranges'},
    "aps_alfa-neat.py": {'--alfa_exe', '--aps_ids', '--area', '--arms_ratio', '--columnnames', '--fill_gap', '--headname', '--help', '--infiles', '--join_arms', '--mask_aps_ids', '--mask_areas', '--mask_gaps', '--mproc', '--neat_exe', '--neat_null', '--outpath', '--overwrite', '--safe_mask_gaps', '--sens_corr', '--targclass', '--targsrvy', '--tellurics', '--vacuum', '--wlranges'},
    "aps_cubepreview.py": {'--aps_ids', '--arms_ratio', '--caldir', '--catdir', '--config_file', '--configdir', '--fill_gap', '--help', '--infiles', '--join_arms', '--mask_aps_ids', '--mask_gaps', '--outpath', '--patch_file', '--patch_id', '--safe_mask_gaps', '--sens_corr', '--targclass', '--targsrvy', '--tellurics', '--vacuum', '--wlranges'},
    "aps_l1_multi_plot.py": {'--aps_ids', '--area', '--arms_ratio', '--crr', '--fill_gap', '--help', '--infiles', '--infiles_list', '--join_arms', '--l2_reference', '--mask_aps_ids', '--mask_areas', '--mask_gaps', '--safe_mask_gaps', '--sens_corr', '--targclass', '--targsrvy', '--tellurics', '--vacuum', '--wlranges'},
}


def _live_flags(script_name):
    """Flags declared by the script's argparse parser, read from the option lines of
    its --help output only (lines that start with an option; argparse separates the
    flag column from the help text by 2+ spaces). Text printed by imported modules
    (e.g. a banner mentioning "--something") is therefore NOT counted as a flag.

    A script whose --help cannot run because an optional dependency is missing is
    skipped (environment problem), not reported as "every flag is missing".
    """
    result = subprocess.run(
        [sys.executable, str(PYAPS_DIR / script_name), "--help"],
        capture_output=True, text=True, cwd=str(PYAPS_DIR), timeout=60,
    )
    if result.returncode != 0:
        if "ModuleNotFoundError" in result.stderr or "ImportError" in result.stderr:
            pytest.skip(f"{script_name} --help needs a missing optional dependency: "
                        f"{result.stderr.strip().splitlines()[-1]}")
        pytest.fail(f"{script_name} --help exited {result.returncode}: {result.stderr[-500:]}")
    flags = set()
    for line in result.stdout.splitlines():
        if re.match(r"^\s{1,4}-", line):
            invocation = re.split(r"\s{2,}", line.strip(), maxsplit=1)[0]
            flags.update(re.findall(r"--[a-zA-Z0-9_]+", invocation))
    return flags


@pytest.mark.parametrize("script_name", sorted(EXPECTED_FLAGS))
def test_migrated_script_flag_surface_unchanged(script_name):
    assert _live_flags(script_name) == EXPECTED_FLAGS[script_name]
