"""The default resource profile reproduces the SLURM options of the code before the table existed (PyAPS 2.0, commit e01c3ea).

``tests/golden/slurm_sbatch_pyaps_2_0.json`` was made by running the script generators of that OLD ``aps_runner.py`` over a matrix of
observation classes (MOS LR/HR x stack level 0/1/3/4 x FERRE grid m/n, explicit ``--mp`` lists, time overrides, LIFU/MIFU LR/HR, CS
LR/HR x grid) with ``tests/slurm_script_harness.py``: the ``sbatch`` option lines of the driver script and the ``#SBATCH`` pragmas of
every stage script.  The same matrix run through the current runner (default profile) must give the same lines.  A change of the
default profile that is not a deliberate change of behaviour fails here.

No WEAVE data is needed.  Regenerate the golden file ONLY with the old runner:
    git show e01c3ea:py/PyAPS/aps_runner.py > /tmp/old/py/PyAPS/aps_runner.py ... python tests/slurm_script_harness.py <old runner> <out.json>
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("PyAPS.aps_runner")
import slurm_script_harness as H            # noqa: E402  (tests/ is on sys.path through conftest)

HERE = Path(__file__).resolve().parent
GOLDEN = json.loads((HERE / "golden" / "slurm_sbatch_pyaps_2_0.json").read_text())
RUNNER = HERE.parent / "py" / "PyAPS" / "aps_runner.py"


@pytest.fixture(scope="module")
def from_table(tmp_path_factory):
    """The same matrix, but the caller gives no CPU list: base CPUs, HR/stack rules and memory all come from the default profile."""
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    try:
        yield H.generate_all(RUNNER, tmp_path_factory.mktemp("golden_table"), mp, table_mode=True)
    finally:
        mp.undo()


@pytest.fixture(scope="module")
def current(tmp_path_factory):
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    try:
        # limits {} would hide nothing here: the default profile only warns about limits and never changes a value
        yield H.generate_all(RUNNER, tmp_path_factory.mktemp("golden"), mp)
    finally:
        mp.undo()


def test_the_matrix_is_the_golden_matrix(current):
    assert sorted(current) == sorted(GOLDEN)
    assert len(GOLDEN) >= 29


@pytest.mark.parametrize("key", sorted(GOLDEN))
def test_driver_sbatch_lines_equal_the_old_code(current, key):
    assert current[key]["driver"] == GOLDEN[key]["driver"]


@pytest.mark.parametrize("key", sorted(GOLDEN))
def test_stage_script_pragmas_equal_the_old_code(current, key):
    assert current[key]["pragmas"] == GOLDEN[key]["pragmas"]


TABLE_KEYS = sorted(k for k, kw in H.case_matrix() if "mp" not in kw)


@pytest.mark.parametrize("key", TABLE_KEYS)
def test_default_profile_alone_gives_the_lines_the_dms_values_gave(from_table, key):
    """The DMS used to hand --mp_MOS 8,12,18,10,20,6,1 (and the LIFU/MIFU/CS lists) to the runner; the default profile holds those
    numbers, so a run WITHOUT any --mp list gives the same sbatch lines and pragmas."""
    assert from_table[key]["driver"] == GOLDEN[key]["driver"]
    assert from_table[key]["pragmas"] == GOLDEN[key]["pragmas"]
