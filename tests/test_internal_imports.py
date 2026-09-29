"""Self-containment guard: every ``PyAPS`` module that another module imports must exist in
this source tree.

A plain ``pip install PyAPS`` must never hit an ImportError for one of PyAPS's own modules.
This scans every ``from PyAPS import x`` / ``from PyAPS.x import y`` / ``import PyAPS.x``
(including lazy imports inside functions) with the AST and checks the target file is present.
Imports guarded by ``try/except ImportError`` are exempt: those are deliberate optional paths.
"""
import ast
import pathlib

PKG = pathlib.Path(__file__).resolve().parent.parent / "py" / "PyAPS"


def _module_exists(name: str) -> bool:
    return (PKG / f"{name}.py").exists() or (PKG / name / "__init__.py").exists()


def _guarded_lines(tree: ast.AST) -> set:
    lines = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            catches_import_error = any(
                h.type is None
                or (isinstance(h.type, ast.Name) and h.type.id in ("ImportError", "ModuleNotFoundError", "Exception"))
                or (isinstance(h.type, ast.Tuple) and any(
                    isinstance(e, ast.Name) and e.id in ("ImportError", "ModuleNotFoundError") for e in h.type.elts))
                for h in node.handlers
            )
            if catches_import_error:
                for sub in ast.walk(ast.Module(body=node.body, type_ignores=[])):
                    if hasattr(sub, "lineno"):
                        lines.add(sub.lineno)
    return lines


def _internal_targets(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            parts = node.module.split(".")
            if parts[0] == "PyAPS":
                if len(parts) > 1:
                    yield node.lineno, parts[1]
                else:
                    for alias in node.names:
                        yield node.lineno, alias.name
        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == "PyAPS" and len(parts) > 1:
                    yield node.lineno, parts[1]


def test_all_internal_imports_resolve():
    missing = []
    for path in sorted(PKG.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        guarded = _guarded_lines(tree)
        for lineno, target in _internal_targets(tree):
            if lineno in guarded or _module_exists(target):
                continue
            missing.append(f"{path.relative_to(PKG.parent.parent)}:{lineno} -> PyAPS.{target}")
    assert not missing, "imports of PyAPS modules that are not in the tree:\n" + "\n".join(missing)
