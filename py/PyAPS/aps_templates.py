#!/usr/bin/env python3
"""Version control of the PyAPS template sets (standard library only).

PyAPS needs a large directory of templates and grids (redshift templates, FERRE grids, RVS
libraries, ...). The directory is far too big for git, and different machines hold different
parts of it, so a copy of it was until now identified by nothing but a path. This module gives
every template SET an identity that travels with the files:

* a **set** name: ``full`` (the complete library of an operations site) or ``mini`` (the small
  subset for test users and for viewer-only installs);
* a **version** string, calendar style ``YYYY.MM.N`` (``2026.10.0``); ordering is numeric;
* a **layout** number, the compatibility major: it changes only when PyAPS needs a different
  directory structure; a PyAPS release works with exactly one layout;
* a **manifest**, ``TEMPLATES_MANIFEST.json``, stored next to the files. It carries the version,
  the layout, the number of files and bytes, a *tree digest* and a per-component summary
  (Merkle-like: the tree digest is a hash of the component digests, a component is a top-level
  folder, so one folder can be checked without reading the others), and the list of PyAPS
  versions it was built for. For small sets the manifest also lists every file with its SHA-256;
  for big sets the list is a sidecar, ``TEMPLATES_FILES.json.gz``.

Only the manifest is versioned, never the data. Hash policy: a file up to ``full_below`` bytes
is hashed completely (``sha256:``), a larger one is fingerprinted from its size and three
blocks, the head, the middle and the tail (``s3:``); ``--full`` hashes everything.

What "status" means (the first line that applies; the others are listed as notes)
---------------------------------------------------------------------------------
``missing``       no template directory at the expected place
``unstamped``     the directory exists but has no manifest, so its version is unknown
``incompatible``  the layout is not the one this PyAPS needs, or the version is below the minimum
                  this PyAPS needs
``modified``      the files no longer match the manifest (added, removed or resized files)
``older``         the version is lower than the reference (the latest released one)
``newer``         the version is higher than the reference
``match``         manifest present, files match, version equals the reference (or no reference)

Command line (``python -m PyAPS.aps_templates ...``, ``pyaps templates ...`` once installed, or
``python tools/pyaps_templates.py ...`` from a checkout)::

    check     human readable status of the templates this PyAPS uses (exit 0 ok, 1 warning, 2 problem)
    status    the same as JSON (--json) or as one line (--brief) for monitoring
    manifest  build a manifest of a directory and print it (writes nothing unless --out)
    verify    compare a directory with its manifest (or another manifest file)
    diff      compare two manifests (files or directories)
    stamp     give a directory a new version and write its manifest (dry run unless --yes)
    install   download a released set, verify it and put it in place
    info      what this PyAPS requires

The startup check used by the PyAPS scripts (``startup_check``) only reads the small manifest;
it never walks the tree. Policy: ``PYAPS_TEMPLATES_CHECK`` = ``off`` | ``warn`` | ``major``
(default: warn about everything, refuse only a layout mismatch) | ``strict`` (also refuse
versions below the minimum and a modified tree).
"""
from __future__ import annotations

import argparse
import fnmatch
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

SCHEMA = 1
MANIFEST_NAME = "TEMPLATES_MANIFEST.json"
FILES_NAME = "TEMPLATES_FILES.json.gz"
SET_NAMES = ("full", "mini")
LAYOUT = 1                       # directory layout this PyAPS release understands
DEFAULT_BASE_URL = "https://weave.ast.cam.ac.uk/WOR/downloads/templates"
FULL_BELOW = 256 * 1024 * 1024   # files up to this size are hashed completely
SAMPLE_BYTES = 1024 * 1024       # block size of the head / middle / tail fingerprint
INLINE_LIMIT = 5000              # up to this many files the manifest lists them itself
ENV_DIR = "PYAPS_TEMPLATES_DIR"
ENV_POLICY = "PYAPS_TEMPLATES_CHECK"

# Never part of a set: git metadata, editor and OS litter, our own bookkeeping files, backups.
DEFAULT_EXCLUDES = (".git", ".DS_Store", "._*", "*~", "*.bak", "*.bak.*", "__pycache__", "*.pyc",
                    "TEMPLATES_MANIFEST*", "TEMPLATES_FILES*", ".templates_stage*")

# Compatibility matrix: which template sets a PyAPS release can use. `pyaps` is a version range,
# `layout` the compatibility major, `min_version` the lowest set version that has everything
# that PyAPS release reads. Edit this table (and doc/TEMPLATES.md) when a release needs more.
REQUIREMENTS = {
    "schema": 1,
    "releases": [
        {"pyaps": ">=2.0,<3", "layout": 1, "min_version": {"full": "2026.10.0", "mini": "2026.6.0"}},
    ],
}


class TemplatesError(RuntimeError):
    """A problem the user must act on (bad input, refused operation)."""


class TemplatesIncompatible(TemplatesError):
    """The installed template set cannot be used by this PyAPS (layout mismatch)."""


# ------------------------------------------------------------------------------------ versions
_VER_RX = re.compile(r"^\d+(\.\d+)*$")


def parse_version(text: str) -> Tuple[int, ...]:
    """'2026.10.0' -> (2026, 10, 0). Leading digits of a PyAPS version like '2.0.1.dev3' are used."""
    text = str(text).strip().lstrip("v")
    m = re.match(r"^\d+(?:\.\d+)*", text)
    if not m:
        raise TemplatesError(f"not a version: {text!r}")
    return tuple(int(p) for p in m.group(0).split("."))


def check_set_version(text: str) -> str:
    if not _VER_RX.match(str(text)) or len(str(text).split(".")) < 2:
        raise TemplatesError(f"template version must look like 2026.10.0 (digits and dots), got {text!r}")
    return str(text)


def compare_versions(a: str, b: str) -> int:
    pa, pb = parse_version(a), parse_version(b)
    n = max(len(pa), len(pb))
    pa += (0,) * (n - len(pa))
    pb += (0,) * (n - len(pb))
    return (pa > pb) - (pa < pb)


def version_in_range(version: str, spec: str) -> bool:
    """spec like '>=2.0,<3' (comparators >=, >, <=, <, ==, !=; comma means and)."""
    for part in [p.strip() for p in spec.split(",") if p.strip()]:
        m = re.match(r"^(>=|<=|==|!=|>|<)\s*(.+)$", part)
        if not m:
            raise TemplatesError(f"bad version specifier {part!r}")
        op, ref = m.groups()
        c = compare_versions(version, ref)
        ok = {">=": c >= 0, "<=": c <= 0, "==": c == 0, "!=": c != 0, ">": c > 0, "<": c < 0}[op]
        if not ok:
            return False
    return True


def pyaps_version() -> Optional[str]:
    try:
        import importlib.metadata as md
        return md.version("PyAPS")
    except Exception:
        pass
    f0 = globals().get("__file__")                  # undefined when the source is piped to `python3 -`
    if not f0:
        return None
    here = Path(f0).resolve()
    for up in (here.parent, *here.parents[:3]):
        f = up / "version.txt"
        if f.is_file():
            try:
                return f.read_text().strip()
            except OSError:
                return None
    return None


def requirements_for(pyaps_ver: Optional[str], table: Optional[dict] = None) -> Optional[dict]:
    """The matrix row that applies to *pyaps_ver* (None if the version is unknown or no row matches)."""
    table = table or load_requirements()
    if not pyaps_ver:
        return None
    for row in table.get("releases", []):
        if version_in_range(pyaps_ver, row["pyaps"]):
            return row
    return None


def load_requirements() -> dict:
    """The built-in matrix, or the JSON file named by $PYAPS_TEMPLATES_REQUIREMENTS."""
    p = os.environ.get("PYAPS_TEMPLATES_REQUIREMENTS")
    if p:
        try:
            return json.loads(Path(p).read_text())
        except (OSError, ValueError) as e:
            raise TemplatesError(f"cannot read requirements file {p}: {e}")
    return REQUIREMENTS


# --------------------------------------------------------------------------------- scanning
def _excluded(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def scan_tree(root: str, excludes: Iterable[str] = DEFAULT_EXCLUDES) -> Dict[str, dict]:
    """Every file and symlink under *root* (excluded names pruned at any depth), without reading contents.

    Returns {relative posix path: {"size": int, "link": target or None}}. Symlinks are recorded, never followed.
    """
    root = os.fspath(root)
    excludes = tuple(excludes)
    out: Dict[str, dict] = {}
    stack = [""]
    while stack:
        rel = stack.pop()
        base = os.path.join(root, rel) if rel else root
        try:
            it = list(os.scandir(base))
        except OSError as e:
            raise TemplatesError(f"cannot read {base}: {e}")
        for e in it:
            if _excluded(e.name, excludes):
                continue
            r = f"{rel}/{e.name}" if rel else e.name
            try:
                if e.is_symlink():
                    out[r] = {"size": 0, "link": os.readlink(e.path)}
                elif e.is_dir(follow_symlinks=False):
                    stack.append(r)
                elif e.is_file(follow_symlinks=False):
                    out[r] = {"size": e.stat(follow_symlinks=False).st_size, "link": None}
            except OSError as ex:
                raise TemplatesError(f"cannot stat {e.path}: {ex}")
    return out


def _component(path: str) -> str:
    return path.split("/", 1)[0] if "/" in path else "."


def _lines_digest(lines: Iterable[str]) -> str:
    h = hashlib.sha256()
    for ln in lines:
        h.update(ln.encode("utf-8", "surrogateescape"))
    return "sha256:" + h.hexdigest()


def hash_file(path: str, size: int, full_below: int = FULL_BELOW, sample: int = SAMPLE_BYTES) -> str:
    """sha256 of the whole file (size <= full_below) or an s3 fingerprint (size + head + middle + tail blocks)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        if size <= full_below:
            for blk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
                h.update(blk)
            return "sha256:" + h.hexdigest()
        h.update(b"s3\0" + str(size).encode() + b"\0")
        for off in (0, max(0, size // 2 - sample // 2), max(0, size - sample)):
            fh.seek(off)
            h.update(fh.read(sample))
        return "s3:" + h.hexdigest()


def hash_entries(root: str, entries: Dict[str, dict], full_below: int = FULL_BELOW, sample: int = SAMPLE_BYTES,
                 workers: int = 8, progress: Optional[Callable[[int, int], None]] = None) -> Dict[str, list]:
    """{path: [size, hash]} for the scanned *entries*."""
    out: Dict[str, list] = {}
    todo = []
    for p, e in entries.items():
        if e["link"] is not None:
            out[p] = [0, "link:" + e["link"]]
        else:
            todo.append((p, e["size"]))
    done = 0

    def one(item):
        p, size = item
        try:
            return p, size, hash_file(os.path.join(root, p), size, full_below, sample)
        except OSError as ex:
            raise TemplatesError(f"cannot read {os.path.join(root, p)}: {ex}")

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for p, size, h in ex.map(one, todo):
            out[p] = [size, h]
            done += 1
            if progress and done % 200 == 0:
                progress(done, len(todo))
    return out


def summarize(files: Dict[str, list]) -> Tuple[Dict[str, dict], str]:
    """Per-component summary {name: {files, bytes, shape, digest}} and the tree digest."""
    groups: Dict[str, List[str]] = {}
    for p in files:
        groups.setdefault(_component(p), []).append(p)
    comps = {}
    for c in sorted(groups):
        paths = sorted(groups[c], key=lambda s: s.encode("utf-8", "surrogateescape"))
        comps[c] = {
            "files": len(paths),
            "bytes": sum(files[p][0] for p in paths),
            "shape": _lines_digest(f"{p}\0{files[p][0]}\n" for p in paths),
            "digest": _lines_digest(f"{p}\0{files[p][0]}\0{files[p][1]}\n" for p in paths),
        }
    return comps, tree_digest(comps)


def tree_digest(comps: Dict[str, dict]) -> str:
    return _lines_digest(f"{c}\0{v['files']}\0{v['bytes']}\0{v['digest']}\n" for c, v in sorted(comps.items()))


def shape_summary(entries: Dict[str, dict]) -> Dict[str, dict]:
    """Component summary from stat() data only (names and sizes): what the cheap status compares."""
    groups: Dict[str, List[str]] = {}
    for p in entries:
        groups.setdefault(_component(p), []).append(p)
    out = {}
    for c in sorted(groups):
        paths = sorted(groups[c], key=lambda s: s.encode("utf-8", "surrogateescape"))
        out[c] = {"files": len(paths), "bytes": sum(entries[p]["size"] for p in paths),
                  "shape": _lines_digest(f"{p}\0{entries[p]['size']}\n" for p in paths)}
    return out


# ---------------------------------------------------------------------------------- manifest
def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_manifest(root: str, set_name: str, version: str, *, layout: int = LAYOUT,
                   excludes: Iterable[str] = DEFAULT_EXCLUDES, full_below: int = FULL_BELOW,
                   sample: int = SAMPLE_BYTES, parent: Optional[dict] = None, pyaps: str = ">=2.0,<3",
                   changelog: Optional[list] = None, created: Optional[str] = None, notes: str = "",
                   inline_limit: int = INLINE_LIMIT, workers: int = 8,
                   progress: Optional[Callable[[int, int], None]] = None) -> Tuple[dict, Dict[str, list]]:
    """Build the manifest of *root*. Returns (manifest, files); *files* is {path: [size, hash]}."""
    if set_name not in SET_NAMES:
        raise TemplatesError(f"set must be one of {SET_NAMES}, got {set_name!r}")
    check_set_version(version)
    if not os.path.isdir(root):
        raise TemplatesError(f"not a directory: {root}")
    excludes = tuple(excludes)
    entries = scan_tree(root, excludes)
    if not entries:
        raise TemplatesError(f"no files under {root} (after exclusions)")
    files = hash_entries(root, entries, full_below, sample, workers, progress)
    comps, tdig = summarize(files)
    man = {
        "schema": SCHEMA,
        "kind": "pyaps-templates-manifest",
        "set": set_name,
        "version": version,
        "layout": layout,
        "created": created or _utcnow(),
        "parent": parent or {"version": None, "tree_digest": None},
        "pyaps": pyaps,
        "hash": {"algo": "sha256", "full_below": full_below, "sample_bytes": sample},
        "excludes": list(excludes),
        "tree_digest": tdig,
        "file_count": len(files),
        "total_bytes": sum(v[0] for v in files.values()),
        "components": comps,
        "changelog": list(changelog or []),
        "notes": notes,
    }
    if len(files) <= inline_limit:
        man["files"] = {p: files[p] for p in sorted(files)}
    else:
        man["files"] = None                     # the list is in the sidecar (see write_manifest)
    return man, files


def sidecar_bytes(files: Dict[str, list]) -> bytes:
    raw = json.dumps({"schema": SCHEMA, "files": {p: files[p] for p in sorted(files)}},
                     separators=(",", ":"), ensure_ascii=False).encode("utf-8", "surrogateescape")
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0, compresslevel=6) as g:
        g.write(raw)
    return buf.getvalue()


def manifest_text(man: dict) -> str:
    return json.dumps(man, indent=1, sort_keys=False, ensure_ascii=False) + "\n"


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    except OSError as e:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise TemplatesError(f"cannot write {path}: {e}")


def write_manifest(directory: str, man: dict, files: Optional[Dict[str, list]] = None) -> List[str]:
    """Write TEMPLATES_MANIFEST.json (and the sidecar when the manifest does not list the files itself)."""
    d = Path(directory)
    written = []
    man = dict(man)
    if man.get("files") is None and files is not None:
        data = sidecar_bytes(files)
        _atomic_write(d / FILES_NAME, data)
        man["files_sidecar"] = {"name": FILES_NAME, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        written.append(str(d / FILES_NAME))
    if man.get("files") is None:
        man.pop("files", None)
    _atomic_write(d / MANIFEST_NAME, manifest_text(man).encode("utf-8"))
    written.append(str(d / MANIFEST_NAME))
    return written


def read_manifest(path: str) -> dict:
    """Read a manifest given its file or the directory that holds it."""
    p = Path(path)
    if p.is_dir():
        p = p / MANIFEST_NAME
    try:
        man = json.loads(p.read_text())
    except OSError as e:
        raise TemplatesError(f"cannot read manifest {p}: {e}")
    except ValueError as e:
        raise TemplatesError(f"manifest {p} is not valid JSON: {e}")
    if man.get("kind") != "pyaps-templates-manifest" or "version" not in man or "tree_digest" not in man:
        raise TemplatesError(f"{p} is not a PyAPS templates manifest")
    if man.get("schema", 1) > SCHEMA:
        raise TemplatesError(f"{p} has schema {man.get('schema')}, newer than this tool understands ({SCHEMA}); update PyAPS")
    return man


def manifest_files(man: dict, directory: Optional[str] = None) -> Optional[Dict[str, list]]:
    """The per-file list of a manifest: inline, or from the sidecar next to *directory*; None if unavailable."""
    if man.get("files"):
        return man["files"]
    sc = man.get("files_sidecar")
    if sc and directory:
        f = Path(directory) / sc["name"]
        try:
            data = f.read_bytes()
        except OSError:
            return None
        if hashlib.sha256(data).hexdigest() != sc["sha256"]:
            raise TemplatesError(f"{f} does not match the checksum recorded in the manifest")
        return json.loads(gzip.decompress(data).decode("utf-8", "surrogateescape"))["files"]
    return None


# ----------------------------------------------------------------------------------- verify
def verify(directory: str, man: dict, level: str = "size", files: Optional[Dict[str, list]] = None,
           workers: int = 8, progress: Optional[Callable[[int, int], None]] = None) -> dict:
    """Compare *directory* with *man*.

    level 'size'    names and sizes (stat only: seconds for 100 000 files)
    level 'content' also every hash, with the policy recorded in the manifest (reads the data)
    Returns {"ok", "level", "components": {name: "match"|"modified"|"missing"|"extra"}, "missing", "extra",
    "changed" (lists capped at 200, 'detail' says whether file level lists were available)}.
    """
    if level not in ("size", "content"):
        raise TemplatesError("level must be 'size' or 'content'")
    excludes = man.get("excludes", DEFAULT_EXCLUDES)
    entries = scan_tree(directory, excludes)
    if files is None:
        files = manifest_files(man, directory)
    res = {"ok": False, "level": level, "components": {}, "missing": [], "extra": [], "changed": [],
           "detail": "files" if files else "components"}
    if files:
        want = set(files)
        have = set(entries)
        res["missing"] = sorted(want - have)
        res["extra"] = sorted(have - want)
        res["changed"] = sorted(p for p in want & have
                                if files[p][0] != entries[p]["size"]
                                or (files[p][1].startswith("link:")) != (entries[p]["link"] is not None)
                                or (entries[p]["link"] is not None and files[p][1] != "link:" + entries[p]["link"]))
        if level == "content":
            hp = man.get("hash", {})
            common = {p: entries[p] for p in want & have if p not in set(res["changed"])}
            got = hash_entries(directory, common, hp.get("full_below", FULL_BELOW), hp.get("sample_bytes", SAMPLE_BYTES), workers, progress)
            res["changed"] = sorted(set(res["changed"]) | {p for p, v in got.items() if v[1] != files[p][1]})
        bad = set(res["missing"]) | set(res["extra"]) | set(res["changed"])
        for c in man["components"]:
            res["components"][c] = "modified" if any(_component(p) == c for p in bad) else "match"
        for c in {_component(p) for p in res["extra"]} - set(man["components"]):
            res["components"][c] = "extra"
        res["ok"] = not bad
    else:
        shape = shape_summary(entries)
        for c, v in man["components"].items():
            s = shape.get(c)
            res["components"][c] = "missing" if s is None else ("match" if (s["files"], s["bytes"], s["shape"]) == (v["files"], v["bytes"], v["shape"]) else "modified")
        for c in set(shape) - set(man["components"]):
            res["components"][c] = "extra"
        ok = all(v == "match" for v in res["components"].values())
        if ok and level == "content":
            got = hash_entries(directory, entries, man.get("hash", {}).get("full_below", FULL_BELOW),
                               man.get("hash", {}).get("sample_bytes", SAMPLE_BYTES), workers, progress)
            comps, tdig = summarize(got)
            for c, v in man["components"].items():
                if comps.get(c, {}).get("digest") != v["digest"]:
                    res["components"][c] = "modified"
            ok = tdig == man["tree_digest"]
        res["ok"] = ok
    for k in ("missing", "extra", "changed"):
        res["n_" + k] = len(res[k])
        res[k] = res[k][:200]
    return res


# -------------------------------------------------------------------------------------- diff
def diff_manifests(a: dict, b: dict, dir_a: Optional[str] = None, dir_b: Optional[str] = None) -> dict:
    """What differs between two manifests (a = old, b = new): header, components, and files when both lists exist."""
    d = {"a": {"set": a["set"], "version": a["version"], "tree_digest": a["tree_digest"]},
         "b": {"set": b["set"], "version": b["version"], "tree_digest": b["tree_digest"]},
         "same_tree": a["tree_digest"] == b["tree_digest"], "layout": [a.get("layout"), b.get("layout")],
         "components": {}, "files": None}
    ca, cb = a["components"], b["components"]
    for c in sorted(set(ca) | set(cb)):
        if c not in cb:
            d["components"][c] = "removed"
        elif c not in ca:
            d["components"][c] = "added"
        else:
            d["components"][c] = "same" if ca[c]["digest"] == cb[c]["digest"] else "changed"
    fa, fb = manifest_files(a, dir_a), manifest_files(b, dir_b)
    if fa is not None and fb is not None:
        d["files"] = {"added": sorted(set(fb) - set(fa)), "removed": sorted(set(fa) - set(fb)),
                      "changed": sorted(p for p in set(fa) & set(fb) if fa[p] != fb[p])}
    return d


# --------------------------------------------------------------------------------- locating
_KEY_RX = re.compile(r"""^\s*(templates_[A-Za-z_]+|SQ_templates|SPACE_templates_dir|AMY_templates_dir|IFU_templates_CLASS_\w+)\s*:\s*['"]?([^'"#\n]+?)['"]?\s*(?:#.*)?$""")


def _expand(p: str) -> str:
    return os.path.expanduser(os.path.expandvars(p))


def _root_of(path: str) -> Optional[str]:
    """The directory named PyAPS_templates that contains *path* (or None)."""
    parts = Path(_expand(path)).parts
    for i, part in enumerate(parts):
        if part == "PyAPS_templates":
            return str(Path(*parts[:i + 1]))
    return None


def candidate_dirs(conf: Optional[dict] = None) -> List[str]:
    """Where this machine keeps its templates, in the order PyAPS looks: $PYAPS_TEMPLATES_DIR, the paths in the
    script_params config (given as a dict, or found through $PYAPS_CONFIGDIR / the PyAPS checkout), then the
    default places under $PYAPS_HOME and ~/PyAPS."""
    out: List[str] = []

    def add(p):
        if p and p not in out:
            out.append(p)

    add(os.environ.get(ENV_DIR) and _expand(os.environ[ENV_DIR]))
    if conf:
        for k, v in conf.items():
            if isinstance(v, str) and "templates" in k.lower():
                add(_root_of(v))
    else:
        for cfg in _script_params_files():
            try:
                for line in Path(cfg).read_text().splitlines():
                    m = _KEY_RX.match(line)
                    if m and "<" not in m.group(2):
                        add(_root_of(m.group(2)))
            except OSError:
                pass
    home = os.environ.get("PYAPS_HOME")
    for base in ([home] if home else []) + [str(Path.home() / "PyAPS")]:
        add(os.path.join(base, "PyAPS_templates"))
        add(os.path.join(base, "PyAPS_local", "PyAPS_templates"))
    return out


def _script_params_files() -> List[str]:
    out = []
    cfgdir = os.environ.get("PYAPS_CONFIGDIR")
    roots = [cfgdir] if cfgdir else []
    home = os.environ.get("PYAPS_HOME")
    roots += [os.path.join(home, "configs")] if home else []
    roots += [str(Path.home() / "PyAPS" / "configs")]
    for r in roots:
        f = os.path.join(_expand(r), "script_params.yaml")
        if os.path.isfile(f) and f not in out:
            out.append(f)
    return out


def locate(directory: Optional[str] = None, conf: Optional[dict] = None) -> Optional[str]:
    """The template directory to use: the one given, else the first candidate that exists."""
    if directory:
        return _expand(directory) if os.path.isdir(_expand(directory)) else None
    for c in candidate_dirs(conf):
        if os.path.isdir(c):
            return c
    return None


# ------------------------------------------------------------------------------------ status
def status(directory: Optional[str] = None, *, level: str = "size", reference: Optional[dict] = None,
           pyaps_ver: Optional[str] = None, conf: Optional[dict] = None, expected_set: Optional[str] = None,
           candidate: Optional[dict] = None, manifest_only: bool = False) -> dict:
    """The full status record of one template directory (see the module docstring for the states).

    *reference* is the manifest (dict) of the latest released set of the same name, if known: it decides older / newer.
    *candidate* is a manifest to compare an UNSTAMPED directory with (component level): the record then says whether
    the files are those of that version. *manifest_only* does not look at the files at all (a viewer-only host that
    keeps just a copy of the manifest, such as the hub).
    """
    pv = pyaps_ver or pyaps_version()
    rec: dict = {"schema": SCHEMA, "kind": "pyaps-templates-status", "time": _utcnow(), "pyaps_version": pv,
                 "dir": None, "set": expected_set, "version": None, "tree_digest": None, "state": None,
                 "level": level, "components": {}, "requires": None, "reference": None, "notes": []}
    d = locate(directory, conf)
    if d is None:
        rec["state"] = "missing"
        rec["notes"].append("no template directory found (looked at: %s)" % ", ".join(
            [directory] if directory else candidate_dirs(conf)[:4]))
        rec["summary"] = "missing"
        return rec
    rec["dir"] = d
    mpath = Path(d) / MANIFEST_NAME
    if not mpath.is_file():
        rec["state"] = "unstamped"
        rec["notes"].append(f"{MANIFEST_NAME} not found: the version of this set is unknown")
        rec["summary"] = "unstamped"
        if candidate and not manifest_only:
            rec["candidate"] = _candidate_report(d, candidate, rec)
        return rec
    try:
        man = read_manifest(d)
    except TemplatesError as e:
        rec["state"] = "unstamped"
        rec["notes"].append(str(e))
        rec["summary"] = "unstamped"
        return rec
    rec.update(set=man["set"], version=man["version"], layout=man.get("layout"), tree_digest=man["tree_digest"],
               files=man["file_count"], bytes=man["total_bytes"], stamped=man.get("created"))
    problems: List[str] = []
    row = requirements_for(pv)
    if row:
        minv = row.get("min_version", {}).get(man["set"])
        rec["requires"] = {"layout": row["layout"], "min_version": minv, "pyaps": row["pyaps"]}
        if man.get("layout") != row["layout"]:
            problems.append(f"layout {man.get('layout')} but PyAPS {pv} needs layout {row['layout']}")
        if minv and compare_versions(man["version"], minv) < 0:
            problems.append(f"version {man['version']} is below the minimum {minv} for PyAPS {pv}")
    elif pv:
        rec["notes"].append(f"no compatibility row for PyAPS {pv}")
    if expected_set and man["set"] != expected_set:
        problems.append(f"this is the {man['set']} set, {expected_set} expected")
    v = None
    if manifest_only:
        rec["notes"].append("manifest only: the files were not examined")
        rec["level"] = "manifest"
    else:
        try:
            v = verify(d, man, level=level)
        except TemplatesError as e:
            rec["notes"].append(str(e))
    modified = False
    if v is not None:
        rec["components"] = v["components"]
        modified = not v["ok"]
        if modified:
            rec["notes"].append("differs from the manifest: %d missing, %d extra, %d changed%s" % (
                v["n_missing"], v["n_extra"], v["n_changed"],
                "" if v["detail"] == "files" else " (component level; no file list)"))
            rec["changes"] = {k: v[k][:20] for k in ("missing", "extra", "changed")}
    order = None
    if reference:
        rec["reference"] = {"set": reference.get("set"), "version": reference["version"], "tree_digest": reference.get("tree_digest")}
        order = compare_versions(man["version"], reference["version"])
        if order == 0 and reference.get("tree_digest") != man["tree_digest"]:
            rec["notes"].append("same version number as the reference but a different tree digest")
            problems.append("tree digest differs from the released manifest of the same version")
    if problems:
        rec["state"] = "incompatible" if any(p.startswith(("layout", "version", "this is")) for p in problems) else "modified"
        rec["notes"] = problems + rec["notes"]
    elif modified:
        rec["state"] = "modified"
    elif order is not None and order < 0:
        rec["state"] = "older"
    elif order is not None and order > 0:
        rec["state"] = "newer"
    else:
        rec["state"] = "match"
    rec["summary"] = f"{man['set']} {man['version']} {rec['state']}"
    return rec


def _candidate_report(d: str, cand: dict, rec: dict) -> dict:
    """Component-level comparison of an unstamped directory with the manifest of a known version."""
    try:
        v = verify(d, cand, level="size", files={})          # {} forces the component level (no file list)
    except TemplatesError as e:
        rec["notes"].append(str(e))
        return {"set": cand.get("set"), "version": cand.get("version"), "error": str(e)}
    differ = sorted(c for c, st in v["components"].items() if st != "match")
    rep = {"set": cand["set"], "version": cand["version"], "same_files": v["ok"], "components_differing": differ}
    rec["notes"].append(("the files are exactly those of %s %s (names and sizes)" % (cand["set"], cand["version"])) if v["ok"]
                        else "the files differ from %s %s in: %s" % (cand["set"], cand["version"], ", ".join(differ) or "?"))
    return rep


# ---------------------------------------------------------------------------------- stamping
def stamp(directory: str, set_name: str, version: str, message: str, *, yes: bool = False, force: bool = False,
          layout: int = LAYOUT, excludes: Optional[Iterable[str]] = None, pyaps: str = ">=2.0,<3",
          full_below: int = FULL_BELOW, workers: int = 8, by: str = "",
          progress: Optional[Callable[[int, int], None]] = None) -> dict:
    """Give *directory* a new version. Never silent: the version and a message are required, nothing is
    written without *yes*, the previous manifest is kept as TEMPLATES_MANIFEST.<version>.json, and the
    version must be higher than the parent's. Returns the plan / result."""
    if not message or not message.strip():
        raise TemplatesError("a stamp needs a message (what changed in this version)")
    check_set_version(version)
    d = Path(directory)
    parent = None
    prev = None
    old_changelog: list = []
    if (d / MANIFEST_NAME).is_file():
        prev = read_manifest(d)
        parent = {"version": prev["version"], "tree_digest": prev["tree_digest"]}
        old_changelog = prev.get("changelog", [])
        if prev["set"] != set_name:
            raise TemplatesError(f"{directory} is stamped as the {prev['set']} set, not {set_name}")
        if excludes is None:
            excludes = prev.get("excludes")
        c = compare_versions(version, prev["version"])
        if c < 0 or (c == 0):
            raise TemplatesError(f"version {version} is not higher than the current {prev['version']}")
    ex = tuple(excludes) if excludes is not None else DEFAULT_EXCLUDES
    entry = {"version": version, "date": _utcnow()[:10], "note": message.strip()}
    if by:
        entry["by"] = by
    man, files = build_manifest(str(d), set_name, version, layout=layout, excludes=ex, full_below=full_below, parent=parent,
                                pyaps=pyaps, changelog=old_changelog + [entry], workers=workers, progress=progress)
    plan = {"dir": str(d), "set": set_name, "version": version, "parent": parent, "tree_digest": man["tree_digest"],
            "file_count": man["file_count"], "total_bytes": man["total_bytes"], "written": [], "applied": False}
    if prev and prev["tree_digest"] == man["tree_digest"] and not force:
        raise TemplatesError(f"the files are identical to version {prev['version']} (tree digest unchanged); nothing to stamp (use --force to stamp anyway)")
    if prev:
        plan["delta"] = diff_manifests(prev, man, str(d), None)["components"]
    if not yes:
        return plan
    if prev:
        _atomic_write(d / f"TEMPLATES_MANIFEST.{prev['version']}.json", (d / MANIFEST_NAME).read_bytes())
    plan["written"] = write_manifest(str(d), man, files)
    plan["applied"] = True
    return plan


# ---------------------------------------------------------------------------------- zip, install
def build_zip(root: str, man: dict, out_zip: str, top: str = "PyAPS_templates") -> str:
    """Deterministic zip of the files of *root* that the manifest lists, plus the manifest, below one top folder."""
    files = man.get("files")
    if not files:
        raise TemplatesError("only a manifest that lists its files can be zipped")
    date = tuple(int(x) for x in re.split(r"[-T:Z]", man["created"])[:6])
    tmp = out_zip + ".part"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(files):
            if files[p][1].startswith("link:"):
                zi = zipfile.ZipInfo(f"{top}/{p}", date)
                zi.create_system = 3
                zi.external_attr = (0o120777 << 16)
                z.writestr(zi, files[p][1][5:])
            else:
                zi = zipfile.ZipInfo(f"{top}/{p}", date)
                zi.create_system = 3
                zi.external_attr = (0o100644 << 16)
                zi.compress_type = zipfile.ZIP_DEFLATED
                with open(os.path.join(root, p), "rb") as src, z.open(zi, "w") as dst:
                    shutil.copyfileobj(src, dst, 4 * 1024 * 1024)
        zi = zipfile.ZipInfo(f"{top}/{MANIFEST_NAME}", date)
        zi.create_system = 3
        zi.external_attr = (0o100644 << 16)
        z.writestr(zi, manifest_text(man), compress_type=zipfile.ZIP_DEFLATED)
    os.replace(tmp, out_zip)
    return out_zip


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(blk)
    return h.hexdigest()


def _fetch(base: str, rel: str, dest: str) -> None:
    """Copy base/rel (a URL or a directory) to dest."""
    if re.match(r"^https?://", base):
        import urllib.request
        url = base.rstrip("/") + "/" + rel
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(dest, "wb") as out:
                shutil.copyfileobj(r, out, 1024 * 1024)
        except Exception as e:
            raise TemplatesError(f"download of {url} failed: {e}")
    else:
        b = base[7:] if base.startswith("file://") else base
        try:
            shutil.copyfile(os.path.join(b, rel), dest)
        except OSError as e:
            raise TemplatesError(f"cannot read {os.path.join(b, rel)}: {e}")


def install(set_name: str, dest: str, *, version: str = "latest", base_url: str = DEFAULT_BASE_URL,
            upgrade: bool = False, dry_run: bool = False, pyaps_ver: Optional[str] = None) -> dict:
    """Download a released set, verify the zip (sha256) and the tree (manifest, content level), refuse an
    incompatible one, and move it to *dest*. An existing non-empty *dest* is only replaced with *upgrade*,
    and then it is renamed to <dest>.<version>.<date>, never deleted."""
    if set_name != "mini":
        raise TemplatesError("only the mini set is published for download; the full set is installed by the operator")
    work = Path(tempfile.mkdtemp(prefix=".templates_stage", dir=str(Path(dest).resolve().parent) if Path(dest).resolve().parent.is_dir() else None))
    rec = {"set": set_name, "dest": dest, "applied": False}
    try:
        latest = work / "latest.json"
        _fetch(base_url, f"{set_name}/latest.json", str(latest))
        li = json.loads(latest.read_text())
        ver = li["version"] if version == "latest" else check_set_version(version)
        if version == "latest":
            rel_dir = li.get("path", f"{set_name}/{ver}")
        else:
            rel_dir = f"{set_name}/{ver}"
        zname = f"PyAPS_templates_{set_name}-{ver}.zip"
        _fetch(base_url, f"{rel_dir}/{zname}.sha256", str(work / "z.sha256"))
        want = (work / "z.sha256").read_text().split()[0].lower()
        rec.update(version=ver, sha256=want)
        pv = pyaps_ver if pyaps_ver is not None else pyaps_version()
        row = requirements_for(pv)
        if row:
            minv = row["min_version"].get(set_name)
            if minv and compare_versions(ver, minv) < 0:
                raise TemplatesError(f"{set_name} {ver} is below the minimum {minv} that PyAPS {pv} needs")
        if dry_run:
            rec["note"] = "dry run: would download %s and install it at %s" % (zname, dest)
            return rec
        _fetch(base_url, f"{rel_dir}/{zname}", str(work / zname))
        got = sha256_file(str(work / zname))
        if got != want:
            raise TemplatesError(f"checksum mismatch for {zname}: expected {want}, got {got}")
        ex = work / "x"
        with zipfile.ZipFile(work / zname) as z:
            for info in z.infolist():
                target = (ex / info.filename).resolve()
                if not str(target).startswith(str(ex.resolve()) + os.sep):
                    raise TemplatesError(f"unsafe path in zip: {info.filename}")
            z.extractall(ex)
            # symlinks were stored as small files with the link bit: restore them
            for info in z.infolist():
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    f = ex / info.filename
                    t = f.read_text()
                    f.unlink()
                    os.symlink(t, f)
        inner = ex / "PyAPS_templates"
        man = read_manifest(str(inner))
        if man["version"] != ver or man["set"] != set_name:
            raise TemplatesError("the manifest inside the zip does not match the requested set and version")
        if row and man.get("layout") != row["layout"]:
            raise TemplatesIncompatible(f"layout {man.get('layout')} but PyAPS {pv} needs layout {row['layout']}")
        v = verify(str(inner), man, level="content")
        if not v["ok"]:
            raise TemplatesError("the extracted files do not match their manifest: %d missing, %d extra, %d changed" % (v["n_missing"], v["n_extra"], v["n_changed"]))
        dpath = Path(dest)
        if dpath.exists() and (not dpath.is_dir() or any(dpath.iterdir())):
            if not upgrade:
                old = ""
                try:
                    old = " (" + read_manifest(str(dpath))["version"] + ")"
                except Exception:
                    pass
                raise TemplatesError(f"{dest} already holds templates{old}; use --upgrade to replace it (it is renamed, not deleted)")
            old_ver = "unstamped"
            try:
                old_ver = read_manifest(str(dpath))["version"]
            except Exception:
                pass
            keep = f"{dest}.{old_ver}.{time.strftime('%Y%m%d%H%M%S')}"
            os.rename(dest, keep)
            rec["previous"] = keep
        elif dpath.is_dir():
            dpath.rmdir()
        dpath.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(inner), dest)
        rec["applied"] = True
        rec["files"] = man["file_count"]
        return rec
    finally:
        shutil.rmtree(work, ignore_errors=True)


# -------------------------------------------------------------------------- display / feed
def display_info(extra_dirs: Iterable[str] = ()) -> Optional[dict]:
    """The identity of the template set a viewer should show: the manifest in $PYAPS_TEMPLATES_DIR, in
    $PYAPS_CONFIGDIR (a viewer-only host such as the hub keeps a copy of the mini manifest there) or in the
    usual template directory. Reads one small file; returns None when there is none."""
    dirs = [os.environ.get(ENV_DIR), *extra_dirs, os.environ.get("PYAPS_CONFIGDIR")]
    for d in [x for x in dirs if x] + candidate_dirs():
        try:
            man = read_manifest(_expand(d))
        except TemplatesError:
            continue
        return {"set": man["set"], "version": man["version"], "layout": man.get("layout"),
                "tree_digest": man["tree_digest"], "created": man.get("created"), "files": man.get("file_count")}
    return None


def display_line() -> Optional[str]:
    """'Templates: mini 2026.10.0' for a banner, or None."""
    info = display_info()
    return f"Templates: {info['set']} {info['version']}" if info else None


# -------------------------------------------------------------------------- startup check
_WARNED: set = set()


def startup_check(conf: Optional[dict] = None, *, policy: Optional[str] = None, out=None, pyaps_ver: Optional[str] = None) -> Optional[dict]:
    """Cheap check for the PyAPS scripts: read the manifest of the templates in use and compare version and
    layout with what this PyAPS needs. Reads one small file; never walks the tree; never writes.

    Warns on stderr (once per process and message). Raises TemplatesIncompatible only on a layout mismatch
    (policy 'major', the default) or also on a version below the minimum / unknown set (policy 'strict');
    policy 'warn' never raises and 'off' does nothing. Returns the small record, or None when off.
    """
    policy = (policy or os.environ.get(ENV_POLICY) or "major").lower()
    if policy == "off":
        return None
    if policy not in ("warn", "major", "strict"):
        policy = "major"
    out = out or sys.stderr
    pv = pyaps_ver if pyaps_ver is not None else pyaps_version()
    d = locate(None, conf)
    msgs: List[str] = []
    fatal: List[str] = []
    rec: dict = {"dir": d, "set": None, "version": None}
    if d is None:
        msgs.append("PyAPS templates: no template directory found (see doc/TEMPLATES.md: pyaps templates install --set mini)")
    elif not (Path(d) / MANIFEST_NAME).is_file():
        msgs.append(f"PyAPS templates: {d} has no {MANIFEST_NAME}; the template version is unknown (run: pyaps templates check)")
    else:
        try:
            man = read_manifest(d)
            rec.update(set=man["set"], version=man["version"], layout=man.get("layout"))
            row = requirements_for(pv)
            if row:
                if man.get("layout") != row["layout"]:
                    fatal.append(f"PyAPS templates: {d} is layout {man.get('layout')} ({man['set']} {man['version']}) but PyAPS {pv} needs layout {row['layout']}")
                minv = row["min_version"].get(man["set"])
                if minv and compare_versions(man["version"], minv) < 0:
                    (fatal if policy == "strict" else msgs).append(
                        f"PyAPS templates: {man['set']} {man['version']} in {d} is older than the {minv} that PyAPS {pv} needs")
        except TemplatesError as e:
            msgs.append(f"PyAPS templates: {e}")
    for m in fatal if policy != "warn" else []:
        raise TemplatesIncompatible(m + f". Install a matching set, or set {ENV_POLICY}=warn to continue at your own risk.")
    for m in msgs + (fatal if policy == "warn" else []):
        if m not in _WARNED:
            _WARNED.add(m)
            print("WARNING: " + m, file=out)
    return rec


# ------------------------------------------------------------------------------------- CLI
def _progress(a: int, b: int) -> None:
    print(f"  hashed {a}/{b} files", file=sys.stderr)


def _print_status(rec: dict, out=None) -> None:
    out = out or sys.stdout
    print(f"templates : {rec.get('dir') or '(none found)'}", file=out)
    print(f"set       : {rec.get('set') or '?'}   version: {rec.get('version') or '?'}   layout: {rec.get('layout', '?')}", file=out)
    print(f"PyAPS     : {rec.get('pyaps_version') or '?'}   requires: {json.dumps(rec.get('requires'))}", file=out)
    if rec.get("reference"):
        print(f"reference : {rec['reference']['set']} {rec['reference']['version']}", file=out)
    if rec.get("files") is not None:
        print(f"content   : {rec['files']} files, {rec['bytes'] / 1e9:.3f} GB, tree {str(rec.get('tree_digest'))[:23]}...", file=out)
    print(f"STATE     : {rec['state']}", file=out)
    for n in rec.get("notes", []):
        print(f"  - {n}", file=out)
    bad = {c: s for c, s in rec.get("components", {}).items() if s != "match"}
    for c, s in bad.items():
        print(f"  - component {c}: {s}", file=out)


def _exit_code(rec: dict, strict: bool = False) -> int:
    s = rec["state"]
    if s in ("missing", "incompatible"):
        return 2
    if s == "match" or (s == "newer" and not strict):
        return 0
    return 1


def _load_ref(path: Optional[str]) -> Optional[dict]:
    return read_manifest(path) if path else None


def _man_arg(path: str) -> Tuple[dict, Optional[str]]:
    p = Path(path)
    return read_manifest(path), str(p if p.is_dir() else p.parent)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pyaps templates", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--dir", help=f"template directory (default: ${ENV_DIR}, then the paths of script_params.yaml, then ~/PyAPS/PyAPS_templates)")
        sp.add_argument("--level", choices=["size", "content"], default="size", help="size: names and sizes, seconds (default); content: also every hash, reads the data")
        sp.add_argument("--reference", help="manifest (file or directory) of the latest released set of the same name: decides older / newer")
        sp.add_argument("--reference-json", help="the same as JSON text (set, version, tree_digest), for programs")
        sp.add_argument("--candidate-json", help="manifest as JSON text to compare an unstamped folder with (component level)")
        sp.add_argument("--manifest-only", action="store_true", help="only a copy of the manifest is here: do not look at the files")
        sp.add_argument("--pyaps-version", help="PyAPS version to check against (default: the installed one)")
        sp.add_argument("--set", dest="expect", choices=SET_NAMES, help="the set this machine should hold")

    s = sub.add_parser("check", help="human readable status; exit 0 ok, 1 warning, 2 problem")
    common(s)
    s.add_argument("--strict", action="store_true", help="a newer set also counts as a warning")
    s = sub.add_parser("status", help="status as JSON (--json) or one line (--brief)")
    common(s)
    s.add_argument("--json", action="store_true")
    s.add_argument("--brief", action="store_true", help="'<set> <version> <state>' on one line")
    s = sub.add_parser("manifest", help="build the manifest of a directory (prints it; writes only with --out)")
    s.add_argument("dir")
    s.add_argument("--set", required=True, choices=SET_NAMES)
    s.add_argument("--version", required=True)
    s.add_argument("--exclude", action="append", help="extra name pattern to leave out (repeatable)")
    s.add_argument("--full", action="store_true", help="hash every file completely (default: files above %d MiB are fingerprinted)" % (FULL_BELOW >> 20))
    s.add_argument("--out", help="write the manifest to this file instead of printing")
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--progress", action="store_true")
    s = sub.add_parser("verify", help="compare a directory with its manifest; exit 1 if they differ")
    s.add_argument("dir")
    s.add_argument("--manifest", help="manifest file (default: the one in the directory)")
    s.add_argument("--level", choices=["size", "content"], default="size")
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--json", action="store_true")
    s = sub.add_parser("diff", help="compare two manifests (files or the directories holding them)")
    s.add_argument("a")
    s.add_argument("b")
    s.add_argument("--json", action="store_true")
    s = sub.add_parser("stamp", help="give a directory a new version and write its manifest")
    s.add_argument("dir")
    s.add_argument("--set", required=True, choices=SET_NAMES)
    s.add_argument("--version", required=True)
    s.add_argument("--message", "-m", required=True, help="what changed in this version")
    s.add_argument("--by", default="", help="who stamps (free text, kept in the changelog)")
    s.add_argument("--exclude", action="append")
    s.add_argument("--full", action="store_true")
    s.add_argument("--force", action="store_true", help="stamp although the files are identical to the parent")
    s.add_argument("--yes", action="store_true", help="really write (default: show the plan only)")
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--progress", action="store_true")
    s = sub.add_parser("install", help="download a released set, verify and put it in place")
    s.add_argument("--set", default="mini", choices=SET_NAMES)
    s.add_argument("--version", default="latest")
    s.add_argument("--dest", help="where to put it (default: ~/PyAPS/PyAPS_templates or $%s)" % ENV_DIR)
    s.add_argument("--base-url", default=os.environ.get("PYAPS_TEMPLATES_URL", DEFAULT_BASE_URL), help="download area (URL or directory)")
    s.add_argument("--upgrade", action="store_true", help="replace an existing set (it is renamed, not deleted)")
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--pyaps-version", help="PyAPS version to check the download against (default: the installed one)")
    s = sub.add_parser("info", help="what this PyAPS requires")
    s.add_argument("--pyaps-version")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _run(args)
    except TemplatesError as e:
        print(f"pyaps templates: {e}", file=sys.stderr)
        return 2


def _run(args) -> int:
    if args.cmd in ("check", "status"):
        ref = json.loads(args.reference_json) if args.reference_json else _load_ref(args.reference)
        cand = json.loads(args.candidate_json) if args.candidate_json else None
        rec = status(args.dir, level=args.level, reference=ref, pyaps_ver=args.pyaps_version, expected_set=args.expect,
                     candidate=cand, manifest_only=args.manifest_only)
        if args.cmd == "status":
            print(rec["summary"] if args.brief else json.dumps(rec, indent=1))
            return 0
        _print_status(rec)
        return _exit_code(rec, args.strict)
    if args.cmd == "manifest":
        ex = DEFAULT_EXCLUDES + tuple(args.exclude or ())
        man, files = build_manifest(args.dir, args.set, args.version, excludes=ex, full_below=(1 << 62) if args.full else FULL_BELOW,
                                    workers=args.workers, progress=_progress if args.progress else None)
        if args.out:
            if man.get("files") is None:
                write_manifest(os.path.dirname(os.path.abspath(args.out)) or ".", man, files)
                print("sidecar written next to the manifest", file=sys.stderr)
            else:
                _atomic_write(Path(args.out), manifest_text(man).encode())
            print(f"wrote {args.out}: {man['file_count']} files, tree {man['tree_digest']}", file=sys.stderr)
        else:
            sys.stdout.write(manifest_text(man))
        return 0
    if args.cmd == "verify":
        man = read_manifest(args.manifest or args.dir)
        res = verify(args.dir, man, level=args.level, workers=args.workers, files=manifest_files(man, os.path.dirname(args.manifest) if args.manifest else args.dir))
        if args.json:
            print(json.dumps(res, indent=1))
        else:
            print(f"{man['set']} {man['version']}: " + ("MATCH" if res["ok"] else "DIFFERS") + f" ({res['level']} level, {res['detail']} detail)")
            for k in ("missing", "extra", "changed"):
                for pth in res[k][:20]:
                    print(f"  {k}: {pth}")
                if res["n_" + k] > 20:
                    print(f"  ... {res['n_' + k]} {k} in all")
            for c, s in res["components"].items():
                if s != "match":
                    print(f"  component {c}: {s}")
        return 0 if res["ok"] else 1
    if args.cmd == "diff":
        (ma, da), (mb, db) = _man_arg(args.a), _man_arg(args.b)
        d = diff_manifests(ma, mb, da, db)
        if args.json:
            print(json.dumps(d, indent=1))
        else:
            print(f"{d['a']['set']} {d['a']['version']}  ->  {d['b']['set']} {d['b']['version']}: " + ("same tree" if d["same_tree"] else "tree differs"))
            for c, s in d["components"].items():
                if s != "same":
                    print(f"  component {c}: {s}")
            if d["files"]:
                for k, v in d["files"].items():
                    for pth in v[:30]:
                        print(f"  {k}: {pth}")
                    if len(v) > 30:
                        print(f"  ... {len(v)} {k} in all")
        return 0 if d["same_tree"] else 1
    if args.cmd == "stamp":
        plan = stamp(args.dir, args.set, args.version, args.message, yes=args.yes, force=args.force, by=args.by,
                     excludes=(None if not args.exclude else DEFAULT_EXCLUDES + tuple(args.exclude)),
                     full_below=(1 << 62) if args.full else FULL_BELOW, workers=args.workers, progress=_progress if args.progress else None)
        print(json.dumps(plan, indent=1))
        if not plan["applied"]:
            print("dry run: nothing written (add --yes)", file=sys.stderr)
        return 0
    if args.cmd == "install":
        dest = args.dest or os.environ.get(ENV_DIR) or str(Path.home() / "PyAPS" / "PyAPS_templates")
        rec = install(args.set, _expand(dest), version=args.version, base_url=args.base_url, upgrade=args.upgrade, dry_run=args.dry_run,
                      pyaps_ver=args.pyaps_version or None)
        print(json.dumps(rec, indent=1))
        return 0
    if args.cmd == "info":
        pv = args.pyaps_version or pyaps_version()
        row = requirements_for(pv)
        print(json.dumps({"pyaps_version": pv, "row": row, "layout_of_this_tool": LAYOUT, "default_download_area": DEFAULT_BASE_URL}, indent=1))
        return 0
    return 2


def pyaps_main(argv: Optional[List[str]] = None) -> int:
    """Entry point of the ``pyaps`` command: ``pyaps templates <command> ...``."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] != "templates":
        print("usage: pyaps templates {check,status,manifest,verify,diff,stamp,install,info} ...", file=sys.stderr)
        return 2
    return main(argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
