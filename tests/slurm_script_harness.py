"""Helper of the SLURM-resource tests: runs the script generators of ``aps_runner`` without WEAVE data.

``l1_fileinfo`` is replaced by a fake that returns the facts of one observation class (family, resolution, stack level, ...),
the paths checks are relaxed, and the generated driver script (``*_PyAPS_slurm.sh`` / ``*_CS_slurm.sh``) and the per-stage
scripts are read back and reduced to the SLURM facts: the ``sbatch`` option lines of the driver and the ``#SBATCH`` pragmas
of the stage scripts.  The same function runs against an OLD copy of ``aps_runner`` (the golden files were made with the code
of PyAPS 2.0 before the resource table) and against the current one.

Not a test file (no ``test_`` prefix); imported by ``test_slurm_resources_golden.py``.
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXAMPLE_CONF = HERE.parent / "configs" / "script_params.yaml.example"

STACK_FILE = {0: "single_{n}.fit", 1: "stack_{n}.fit", 3: "super_{n}.fit", 4: "wve_{n}.fit"}


def load_runner(path: str | Path, name: str = "aps_runner_under_test"):
    """Import an ``aps_runner.py`` from an explicit path (the old copy or the current one) as a fresh module."""
    path = Path(path)
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def make_conf(tmp: Path, extra: dict | None = None) -> Path:
    """A script_params file made from the example with every path inside ``tmp``."""
    text = EXAMPLE_CONF.read_text()
    home = tmp / "home"
    for sub in ("PyAPS_data/L2", "PyAPS_data/CS", "configs/ExGal_configs", "bin"):
        (home / sub).mkdir(parents=True, exist_ok=True)
    text = text.replace("${PYAPS_HOME}", str(home)).replace("${PYAPS_PKG_DIR}", str(HERE.parent / "py" / "PyAPS"))
    text = text.replace("<env_suffix>", "")
    lines = [ln for ln in text.splitlines()]
    for key, val in (extra or {}).items():
        lines = [ln for ln in lines if not ln.startswith(key + ":")]
        lines.append(f"{key}: '{val}'")
    conf = tmp / "script_params.yaml"
    conf.write_text("\n".join(lines) + "\n")
    return conf


def fake_l1_info(infiles, family, res, tmp):
    mode = {"MOS": "MOS", "LIFU": "LIFU", "MIFU": "MIFU"}[family]
    return {
        "infiles": list(infiles), "obsmode": [family] * len(infiles), "mode": mode, "res_mode": [res] * len(infiles),
        "camera": ["BLUE", "RED"][: len(infiles)], "xbin": [1] * len(infiles), "ybin": [1] * len(infiles),
        "setups": ["x"] * len(infiles), "srvys": ["WA"] * len(infiles),
        "clss": ["QSO", "STAR-CEP", "STAR_FGK", "STAR_BA", "SKY", "STAR_RRL"], "wlranges_def": [[3900.0, 4900.0], [7000.0, 8100.0]],
        "wlranges": [[3900.0, 4900.0], [7000.0, 8100.0]], "resolution": [5000] * len(infiles),
        "obid": ["1234"] * len(infiles), "obsdate": ["2030-01-01"] * len(infiles), "arms_ratio": [1.0] * len(infiles),
    }


def _relax_paths(mod, monkeypatch):
    def check_path(aps_path, create=False, verbos=True):
        if create and not aps_path.exists():
            aps_path.mkdir(parents=True)
    monkeypatch.setattr(mod, "check_path", check_path)
    monkeypatch.setattr(mod, "warning_path", lambda *a, **k: True)


SBATCH_RE = re.compile(r"sbatch --parsable (.*)\)")


def sbatch_lines(slurm_file: Path):
    """The option lists of the ``sbatch`` calls of a driver script, whitespace-normalised, the output paths and the script path
    replaced by placeholders (they hold the temporary folder)."""
    out = []
    for ln in Path(slurm_file).read_text().splitlines():
        m = SBATCH_RE.search(ln)
        if not m:
            continue
        toks = m.group(1).split()
        norm = []
        for t in toks:
            if t.startswith("--output=") or t.startswith("--error="):
                t = t.split("=")[0] + "=<LOG>"
            elif t.endswith(".sh") and "/" in t:
                t = "<SCRIPT:" + os.path.basename(t).split("_", 1)[-1] + ">"
                t = re.sub(r"<SCRIPT:[0-9a-f]+_", "<SCRIPT:", t)
            norm.append(t)
        out.append(" ".join(norm))
    return out


def pragmas(script: Path):
    return [ln.strip() for ln in Path(script).read_text().splitlines() if ln.startswith("#SBATCH")]


TABLE = "table"      # mp value: give the generator no CPU list (the new runner then takes the resource table)


def generate(runner_path, family, res, stack, grid, tmp, monkeypatch, mp=None, conf_extra=None, wmode="l2",
             **gen_kwargs):
    """Run the generator of ``family`` and return the SLURM facts: ``{'driver': [...], 'pragmas': {script: [...]}, 'mp_args': {...}}``."""
    mod = load_runner(runner_path)
    tmp = Path(tmp)
    extra = dict(conf_extra or {})
    extra["MOS_Gal_grid_prefix_FR"] = grid
    extra["IFU_Gal_grid_prefix_FR"] = grid
    extra["FESWI_grid_prefix"] = grid
    extra.setdefault("FESWI_RES", "LR,HR")      # every CS stage is generated for both resolutions
    extra.setdefault("AMY_RES", "LR,HR")
    conf = make_conf(tmp, extra)
    n = {0: 1, 1: 2, 3: 3, 4: 4}[stack]
    infiles = [str(tmp / "L1" / "20300101" / STACK_FILE[stack].format(n=n + i)) for i in range(2)]
    info = fake_l1_info(infiles, family, res, tmp)
    monkeypatch.setattr(mod, "l1_fileinfo", lambda files, wlranges=None, arms_ratio=None: dict(info))
    _relax_paths(mod, monkeypatch)
    common = dict(infiles=infiles, config_file=str(conf), headname="hn", UAPSID="abcdef012345", hpc=0, aps_ovr=True)
    table = mp == TABLE
    if table:
        mp = None
    if family == "MOS" and wmode == "l2":
        scripts = mod.mos_scriptGEN(mp=None if table else (mp if mp is not None else [8, 12, 18, 10, 20, 6, 1]), **common, **gen_kwargs)
        driver = "abcdef012345_PyAPS_slurm.sh"
    elif family == "MOS":
        scripts = mod.cs_scriptGEN(mp=None if table else (mp if mp is not None else [6, 12, 6, 12, 12, 1, 1]), **{k: v for k, v in common.items() if k != "aps_ovr"}, **gen_kwargs)
        driver = "abcdef012345_CS_slurm.sh"
    else:
        scripts = mod.ifu_scriptGEN(mp_LIFU=None if table else ([6, 30, 20] if mp is None else mp),
                                    mp_MIFU=None if table else ([3, 10, 10] if mp is None else mp), **common, **gen_kwargs)
        driver = "abcdef012345_PyAPS_slurm.sh"
    scripts = Path(scripts)
    res_out = {"driver": sbatch_lines(scripts / driver), "pragmas": {}}
    for sh in sorted(scripts.glob("*.sh")):
        if sh.name == driver or sh.name.endswith("_PyAPS_bash.sh") or sh.name.endswith("_CS_bash.sh"):
            continue
        res_out["pragmas"][re.sub(r"^abcdef012345_", "", sh.name)] = pragmas(sh)
    return res_out


def case_matrix():
    """The observation classes the golden file pins: ``(key, kwargs of generate())``."""
    cases = []
    for res in ("LR", "HR"):
        for stack in (0, 1, 3, 4):
            for grid in ("m", "n"):
                cases.append((f"MOS-L2-{res}-stack{stack}-grid{grid}", dict(family="MOS", res=res, stack=stack, grid=grid)))
    cases.append(("MOS-L2-LR-stack1-gridm-mp4", dict(family="MOS", res="LR", stack=1, grid="m", mp=[4] * 7)))
    cases.append(("MOS-L2-HR-stack0-gridm-mpcustom", dict(family="MOS", res="HR", stack=0, grid="m", mp=[3, 5, 7, 9, 11, 13, 2])))
    cases.append(("MOS-L2-LR-stack1-gridm-time", dict(family="MOS", res="LR", stack=1, grid="m",
                                                      conf_extra={"slurm_time_RR": "12:00:00", "slurm_time_FR": "None", "slurm_time_LS": "soon"})))
    for family in ("LIFU", "MIFU"):
        for res in ("LR", "HR"):
            cases.append((f"{family}-L2-{res}", dict(family=family, res=res, stack=1, grid="m")))
    cases.append(("LIFU-L2-LR-time", dict(family="LIFU", res="LR", stack=1, grid="m", conf_extra={"slurm_time_IFU_ExGal": "1-12:00:00"})))
    cases.append(("MIFU-L2-LR-mp5", dict(family="MIFU", res="LR", stack=1, grid="m", mp=[5, 5, 5])))
    for res in ("LR", "HR"):
        for grid in ("m", "n"):
            cases.append((f"MOS-CS-{res}-grid{grid}", dict(family="MOS", res=res, stack=1, grid=grid, wmode="cs")))
    return cases


def generate_all(runner_path, tmp_root, monkeypatch, table_mode=False, **extra):
    """All cases of the matrix. ``table_mode``: the cases that did not name their own CPU list are generated WITHOUT a list (current
    runner only: the CPU counts then come from the resource table)."""
    import contextlib
    import io
    out = {}
    for key, kw in case_matrix():
        if table_mode and "mp" not in kw:
            kw = dict(kw, mp=TABLE)
        tmp = Path(tmp_root) / (key + ("-t" if table_mode else ""))
        tmp.mkdir(parents=True, exist_ok=True)
        with contextlib.redirect_stdout(io.StringIO()):
            out[key] = generate(runner_path, tmp=tmp, monkeypatch=monkeypatch, **kw, **extra)
    return out


if __name__ == "__main__":      # python slurm_script_harness.py <aps_runner.py> <out.json>   (makes a golden file)
    import json
    import tempfile
    from _pytest.monkeypatch import MonkeyPatch
    mp_ = MonkeyPatch()
    data = generate_all(sys.argv[1], tempfile.mkdtemp(), mp_)
    Path(sys.argv[2]).write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    print("wrote", sys.argv[2], len(data), "cases")
