"""Every L2 SLURM job has a wall-time limit: 48 hours for every stage (owner decision 2026-10-26; the 8 h limits of 2026-10-12 killed the FR stage of 12 configurations).

No WEAVE data needed. Run with:
    cd <PYAPS_DIR> && pytest tests/test_slurm_time_limits.py -v
"""
from __future__ import annotations

import ast
import pathlib

import pytest

aps_runner = pytest.importorskip("PyAPS.aps_runner")

STAGES = ("RR", "RVS", "FR", "PPXF", "EMI", "LS", "L2merge", "IFU_PR", "IFU_ExGal", "IFU_Gal")


def test_every_l2_stage_has_a_default_limit_of_48_hours():
    """The limit is the partition limit (2 days) for every stage until the module resources are set from CPU-efficiency measurements; it must
    never be below twice the longest job that ever completed (which would kill healthy jobs, as the 8 h limits did)."""
    longest = {"RR": 162, "RVS": 23, "FR": 204, "PPXF": 205, "EMI": 111, "LS": 3, "L2merge": 2,
               "IFU_PR": 415, "IFU_ExGal": 374, "IFU_Gal": 518}      # minutes, both pods, 2026-10-07..12
    assert set(aps_runner.DEFAULT_SLURM_TIME) == set(STAGES)
    for stage in STAGES:
        arg = aps_runner.slurm_time_arg({}, stage)
        assert arg == " --time=2-00:00:00", stage
        assert 2 * 24 * 60 >= 2 * longest[stage], stage


def test_a_stage_can_be_overridden_or_switched_off_in_script_params():
    assert aps_runner.slurm_time_arg({"slurm_time_RR": "12:00:00"}, "RR") == " --time=12:00:00"
    assert aps_runner.slurm_time_arg({"slurm_time_RR": "1-12:00:00"}, "RR") == " --time=1-12:00:00"
    assert aps_runner.slurm_time_arg({"slurm_time_RR": "90"}, "RR") == " --time=90"
    assert aps_runner.slurm_time_arg({"slurm_time_RR": "None"}, "RR") == ""
    assert aps_runner.slurm_time_arg({"slurm_time_RR": "none"}, "RR") == ""
    assert aps_runner.slurm_time_arg({"slurm_time_RR": "12:00:00"}, "PPXF") == " --time=2-00:00:00"   # other stages keep theirs


def test_a_value_that_is_not_a_slurm_time_falls_back_to_the_default(capsys):
    assert aps_runner.slurm_time_arg({"slurm_time_EMI": "soon"}, "EMI") == " --time=2-00:00:00"
    assert "not a SLURM time" in capsys.readouterr().out


def test_the_cs_stages_and_unknown_stages_have_no_limit():
    assert aps_runner.slurm_time_arg({}, "SQ") == ""
    assert aps_runner.slurm_time_arg(None, "RR") == " --time=2-00:00:00"


def test_every_mos_and_ifu_sbatch_line_of_the_runner_passes_the_limit():
    """Source check: each sbatch line of an L2 stage must add slurm_time_arg for its own stage."""
    path = pathlib.Path(aps_runner.__file__)
    source = path.read_text()
    tree = ast.parse(source)
    used = {node.args[1].value for node in ast.walk(tree)
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "slurm_time_arg"
            and len(node.args) == 2 and isinstance(node.args[1], ast.Constant)}
    assert used == set(STAGES)
