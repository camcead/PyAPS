"""Every L2 SLURM job has a wall-time limit sized from the jobs that completed (hung jobs of 2026-10-08..12 held 2 days).

No WEAVE data needed. Run with:
    cd <PYAPS_DIR> && pytest tests/test_slurm_time_limits.py -v
"""
from __future__ import annotations

import ast
import pathlib

import pytest

aps_runner = pytest.importorskip("PyAPS.aps_runner")

STAGES = ("RR", "RVS", "FR", "PPXF", "EMI", "LS", "L2merge", "IFU_PR", "IFU_ExGal", "IFU_Gal")


def minutes(text):
    h, m, s = (int(x) for x in text.split(":"))
    return h * 60 + m + s / 60


def test_every_l2_stage_has_a_default_limit_well_above_the_longest_completed_job():
    longest = {"RR": 162, "RVS": 23, "FR": 204, "PPXF": 205, "EMI": 111, "LS": 3, "L2merge": 2,
               "IFU_PR": 415, "IFU_ExGal": 374, "IFU_Gal": 518}      # minutes, both pods, 2026-10-07..12
    assert set(aps_runner.DEFAULT_SLURM_TIME) == set(STAGES)
    for stage in STAGES:
        arg = aps_runner.slurm_time_arg({}, stage)
        assert arg.startswith(" --time=")
        limit = minutes(arg.split("=")[1])
        assert limit >= 2 * longest[stage], stage                      # never kills a healthy job
        assert limit <= 18 * 60, stage                                 # a hung job dies in hours, not in 2 days


def test_a_stage_can_be_overridden_or_switched_off_in_script_params():
    assert aps_runner.slurm_time_arg({"slurm_time_RR": "12:00:00"}, "RR") == " --time=12:00:00"
    assert aps_runner.slurm_time_arg({"slurm_time_RR": "1-12:00:00"}, "RR") == " --time=1-12:00:00"
    assert aps_runner.slurm_time_arg({"slurm_time_RR": "90"}, "RR") == " --time=90"
    assert aps_runner.slurm_time_arg({"slurm_time_RR": "None"}, "RR") == ""
    assert aps_runner.slurm_time_arg({"slurm_time_RR": "none"}, "RR") == ""
    assert aps_runner.slurm_time_arg({"slurm_time_RR": "12:00:00"}, "PPXF") == " --time=08:00:00"   # other stages keep theirs


def test_a_value_that_is_not_a_slurm_time_falls_back_to_the_default(capsys):
    assert aps_runner.slurm_time_arg({"slurm_time_EMI": "soon"}, "EMI") == " --time=06:00:00"
    assert "not a SLURM time" in capsys.readouterr().out


def test_the_cs_stages_and_unknown_stages_have_no_limit():
    assert aps_runner.slurm_time_arg({}, "SQ") == ""
    assert aps_runner.slurm_time_arg(None, "RR") == " --time=08:00:00"


def test_every_mos_and_ifu_sbatch_line_of_the_runner_passes_the_limit():
    """Source check: each sbatch line of an L2 stage must add slurm_time_arg for its own stage."""
    path = pathlib.Path(aps_runner.__file__)
    source = path.read_text()
    tree = ast.parse(source)
    used = {node.args[1].value for node in ast.walk(tree)
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "slurm_time_arg"
            and len(node.args) == 2 and isinstance(node.args[1], ast.Constant)}
    assert used == set(STAGES)
