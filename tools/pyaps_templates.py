#!/usr/bin/env python3
"""Stand-alone entry point of the template version control: works from a plain checkout (or a single copied
file next to py/PyAPS/aps_templates.py), without installing PyAPS and without PyAPS_admin.

    python tools/pyaps_templates.py check
    python tools/pyaps_templates.py install --set mini
    python tools/pyaps_templates.py --help

It is the same program as ``pyaps templates ...`` and ``python -m PyAPS.aps_templates``; see doc/TEMPLATES.md.
"""
import importlib.util
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
_candidates = [_here.parent / "py" / "PyAPS" / "aps_templates.py", _here / "aps_templates.py"]
for _c in _candidates:
    if _c.is_file():
        _spec = importlib.util.spec_from_file_location("aps_templates", _c)
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules["aps_templates"] = _mod
        _spec.loader.exec_module(_mod)
        break
else:
    sys.exit("aps_templates.py not found next to tools/ (expected py/PyAPS/aps_templates.py)")

if __name__ == "__main__":
    raise SystemExit(_mod.main())
