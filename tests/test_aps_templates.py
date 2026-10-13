"""Tests of the template-set version control (py/PyAPS/aps_templates.py), on tiny fake trees.

The module is loaded from its file, so these tests need neither an installed PyAPS nor any real
template data.   Run with:   pytest tests/test_aps_templates.py -v
"""
from __future__ import annotations

import importlib.util
import json
import os
import zipfile
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "py" / "PyAPS" / "aps_templates.py"
_spec = importlib.util.spec_from_file_location("aps_templates_under_test", _SRC)
T = importlib.util.module_from_spec(_spec)
import sys
sys.modules[_spec.name] = T
_spec.loader.exec_module(T)

PYAPS = "2.0"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in (T.ENV_DIR, T.ENV_POLICY, "PYAPS_HOME", "PYAPS_CONFIGDIR", "PYAPS_TEMPLATES_REQUIREMENTS"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HOME", "/nonexistent-home-for-tests")
    T._WARNED.clear()


def make_tree(root: Path, extra: dict | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    files = {"templates_RR/a.fits": b"A" * 100, "templates_RR/README.md": b"hello\n",
             "templates_RR/.git/objects/x": b"gitjunk", "templates_RR/.DS_Store": b"ds",
             "templates_ARC_RR/b.fits": b"B" * 2000, "top.txt": b"top"}
    files.update(extra or {})
    for rel, data in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    os.symlink("a.fits", root / "templates_RR" / "link.fits")
    return root


def stamped(tmp_path, set_name="mini", version="2026.10.0") -> Path:
    root = make_tree(tmp_path / "PyAPS_templates")
    T.stamp(str(root), set_name, version, "first release", yes=True)
    return root


# ------------------------------------------------------------------------------ versions
def test_version_order_is_numeric():
    assert T.compare_versions("2026.10.0", "2026.9.0") > 0
    assert T.compare_versions("2026.10", "2026.10.0") == 0
    assert T.compare_versions("2026.6.0", "2026.10.0") < 0


def test_version_ranges():
    assert T.version_in_range("2.0", ">=2.0,<3")
    assert T.version_in_range("2.4.1.dev3", ">=2.0,<3")
    assert not T.version_in_range("3.0", ">=2.0,<3")
    with pytest.raises(T.TemplatesError):
        T.version_in_range("2.0", "~=2")


@pytest.mark.parametrize("bad", ["latest", "2026", "v1.x", "2026.10.0-rc1"])
def test_bad_set_versions_refused(bad):
    with pytest.raises(T.TemplatesError):
        T.check_set_version(bad)


# ------------------------------------------------------------------------------- manifest
def test_manifest_ignores_litter_and_records_links(tmp_path):
    root = make_tree(tmp_path / "t")
    man, files = T.build_manifest(str(root), "mini", "2026.10.0")
    assert ".DS_Store" not in " ".join(files)
    assert not any(".git" in p for p in files)
    assert files["templates_RR/link.fits"] == [0, "link:a.fits"]
    assert man["file_count"] == 5 and man["total_bytes"] == 100 + 6 + 2000 + 3
    assert set(man["components"]) == {"templates_RR", "templates_ARC_RR", "."}
    assert man["files"] == files                       # small set: listed inline


def test_manifest_is_deterministic_and_content_sensitive(tmp_path):
    root = make_tree(tmp_path / "t")
    a, _ = T.build_manifest(str(root), "mini", "2026.10.0", created="2026-10-13T00:00:00Z")
    b, _ = T.build_manifest(str(root), "mini", "2026.10.0", created="2026-10-14T00:00:00Z")
    assert a["tree_digest"] == b["tree_digest"]
    (root / "templates_ARC_RR" / "b.fits").write_bytes(b"C" * 2000)       # same size, other content
    c, _ = T.build_manifest(str(root), "mini", "2026.10.0")
    assert c["tree_digest"] != a["tree_digest"]
    assert c["components"]["templates_RR"] == a["components"]["templates_RR"]       # Merkle: other folder untouched
    assert c["components"]["templates_ARC_RR"]["shape"] == a["components"]["templates_ARC_RR"]["shape"]
    assert c["components"]["templates_ARC_RR"]["digest"] != a["components"]["templates_ARC_RR"]["digest"]


def test_large_files_are_fingerprinted_not_read_whole(tmp_path):
    root = make_tree(tmp_path / "t", {"templates_RR/big.bin": os.urandom(3000)})
    man, files = T.build_manifest(str(root), "full", "2026.10.0", full_below=1000, sample=100)
    assert files["templates_RR/big.bin"][1].startswith("s3:")
    assert files["templates_ARC_RR/b.fits"][1].startswith("s3:")        # 2000 > 1000
    assert files["top.txt"][1].startswith("sha256:")
    man_full, files_full = T.build_manifest(str(root), "full", "2026.10.0", full_below=1 << 40)
    assert files_full["templates_RR/big.bin"][1].startswith("sha256:")


def test_big_set_lists_files_in_a_sidecar(tmp_path):
    root = make_tree(tmp_path / "t")
    man, files = T.build_manifest(str(root), "full", "2026.10.0", inline_limit=2)
    assert man["files"] is None
    T.write_manifest(str(root), man, files)
    stored = T.read_manifest(str(root))
    assert "files" not in stored and stored["files_sidecar"]["name"] == T.FILES_NAME
    assert T.manifest_files(stored, str(root)) == files
    # the sidecar is bound to the manifest by its checksum
    (root / T.FILES_NAME).write_bytes(T.sidecar_bytes({"x": [1, "sha256:0"]}))
    with pytest.raises(T.TemplatesError):
        T.manifest_files(stored, str(root))


def test_read_manifest_refuses_foreign_json_and_newer_schema(tmp_path):
    (tmp_path / T.MANIFEST_NAME).write_text('{"hello": 1}')
    with pytest.raises(T.TemplatesError):
        T.read_manifest(str(tmp_path))
    (tmp_path / T.MANIFEST_NAME).write_text(json.dumps({"kind": "pyaps-templates-manifest", "version": "1.0", "tree_digest": "x", "schema": 99}))
    with pytest.raises(T.TemplatesError, match="newer"):
        T.read_manifest(str(tmp_path))


# ---------------------------------------------------------------------------------- verify
def test_verify_match_then_each_kind_of_change(tmp_path):
    root = stamped(tmp_path)
    man = T.read_manifest(str(root))
    assert T.verify(str(root), man)["ok"]
    assert T.verify(str(root), man, level="content")["ok"]
    # same size, other bytes: size level cannot see it, content level must
    (root / "templates_ARC_RR" / "b.fits").write_bytes(b"Z" * 2000)
    assert T.verify(str(root), man)["ok"]
    v = T.verify(str(root), man, level="content")
    assert not v["ok"] and v["changed"] == ["templates_ARC_RR/b.fits"] and v["components"]["templates_ARC_RR"] == "modified"
    # resized, added, removed
    (root / "templates_RR" / "a.fits").write_bytes(b"A" * 101)
    (root / "new.txt").write_text("n")
    (root / "top.txt").unlink()
    v = T.verify(str(root), man)
    assert (v["n_changed"], v["n_extra"], v["n_missing"]) == (1, 1, 1)
    assert v["components"]["templates_RR"] == "modified" and v["components"]["."] == "modified"


def test_verify_component_level_without_file_list(tmp_path):
    root = make_tree(tmp_path / "t")
    man, files = T.build_manifest(str(root), "full", "2026.10.0", inline_limit=1)
    T.write_manifest(str(root), man, files)
    (root / T.FILES_NAME).unlink()                         # sidecar lost: still verifiable per component
    stored = T.read_manifest(str(root))
    v = T.verify(str(root), stored)
    assert v["ok"] and v["detail"] == "components"
    (root / "templates_ARC_RR" / "b.fits").write_bytes(b"B" * 5)
    v = T.verify(str(root), stored)
    assert not v["ok"] and v["components"] == {"templates_RR": "match", "templates_ARC_RR": "modified", ".": "match"}
    assert T.verify(str(root), stored, level="content")["ok"] is False


# ------------------------------------------------------------------------------------- diff
def test_diff_names_added_removed_changed(tmp_path):
    root = stamped(tmp_path)
    old = T.read_manifest(str(root))
    (root / "templates_RR" / "c.fits").write_bytes(b"c")
    (root / "top.txt").unlink()
    (root / "templates_ARC_RR" / "b.fits").write_bytes(b"B" * 7)
    T.stamp(str(root), "mini", "2026.11.0", "tweak", yes=True)
    new = T.read_manifest(str(root))
    d = T.diff_manifests(old, new)
    assert not d["same_tree"]
    assert d["components"] == {".": "removed", "templates_ARC_RR": "changed", "templates_RR": "changed"}
    assert d["files"] == {"added": ["templates_RR/c.fits"], "removed": ["top.txt"], "changed": ["templates_ARC_RR/b.fits"]}
    assert T.diff_manifests(new, new)["same_tree"]


# ------------------------------------------------------------------------------------ stamp
def test_stamp_is_never_silent(tmp_path):
    root = make_tree(tmp_path / "t")
    plan = T.stamp(str(root), "mini", "2026.10.0", "first", yes=False)
    assert plan["applied"] is False and not (root / T.MANIFEST_NAME).exists()          # dry run writes nothing
    with pytest.raises(T.TemplatesError, match="message"):
        T.stamp(str(root), "mini", "2026.10.0", "  ", yes=True)
    T.stamp(str(root), "mini", "2026.10.0", "first", yes=True, by="tester")
    man = T.read_manifest(str(root))
    assert man["changelog"] == [{"version": "2026.10.0", "date": man["changelog"][0]["date"], "note": "first", "by": "tester"}]
    # same files again: refused; a lower or equal version: refused; other set name: refused
    with pytest.raises(T.TemplatesError, match="identical"):
        T.stamp(str(root), "mini", "2026.11.0", "again", yes=True)
    (root / "x.txt").write_text("x")
    with pytest.raises(T.TemplatesError, match="not higher"):
        T.stamp(str(root), "mini", "2026.10.0", "again", yes=True)
    with pytest.raises(T.TemplatesError, match="not full"):
        T.stamp(str(root), "full", "2026.11.0", "again", yes=True)


def test_stamp_keeps_the_old_manifest_and_links_the_parent(tmp_path):
    root = stamped(tmp_path)
    first = T.read_manifest(str(root))
    (root / "templates_RR" / "new.fits").write_bytes(b"1234")
    plan = T.stamp(str(root), "mini", "2026.11.0", "added new.fits", yes=True)
    assert plan["applied"] and plan["delta"]["templates_RR"] == "changed"
    second = T.read_manifest(str(root))
    assert second["parent"] == {"version": "2026.10.0", "tree_digest": first["tree_digest"]}
    assert [c["version"] for c in second["changelog"]] == ["2026.10.0", "2026.11.0"]
    assert T.read_manifest(str(root / "TEMPLATES_MANIFEST.2026.10.0.json"))["tree_digest"] == first["tree_digest"]
    assert T.verify(str(root), second)["ok"]                      # bookkeeping files are not part of the set


# ------------------------------------------------------------------------------------ status
def test_status_states(tmp_path, monkeypatch):
    assert T.status(str(tmp_path / "nope"), pyaps_ver=PYAPS)["state"] == "missing"
    bare = make_tree(tmp_path / "bare")
    assert T.status(str(bare), pyaps_ver=PYAPS)["state"] == "unstamped"
    root = stamped(tmp_path / "ok", version="2026.10.0")
    rec = T.status(str(root), pyaps_ver=PYAPS)
    assert rec["state"] == "match" and rec["summary"] == "mini 2026.10.0 match" and rec["files"] == 5
    ref = T.read_manifest(str(root))
    newer_ref = dict(ref, version="2026.12.0")
    assert T.status(str(root), pyaps_ver=PYAPS, reference=newer_ref)["state"] == "older"
    older_ref = dict(ref, version="2026.7.0")
    assert T.status(str(root), pyaps_ver=PYAPS, reference=older_ref)["state"] == "newer"
    (root / "templates_RR" / "a.fits").write_bytes(b"short")
    rec = T.status(str(root), pyaps_ver=PYAPS)
    assert rec["state"] == "modified" and rec["changes"]["changed"] == ["templates_RR/a.fits"]


def test_status_incompatible_layout_and_minimum(tmp_path):
    root = make_tree(tmp_path / "t")
    T.stamp(str(root), "full", "2026.9.0", "old", yes=True)                  # below the minimum 2026.10.0 of the matrix
    rec = T.status(str(root), pyaps_ver=PYAPS)
    assert rec["state"] == "incompatible" and "below the minimum" in rec["notes"][0]
    root2 = make_tree(tmp_path / "t2")
    T.stamp(str(root2), "mini", "2026.10.0", "x", yes=True, layout=7)
    rec = T.status(str(root2), pyaps_ver=PYAPS)
    assert rec["state"] == "incompatible" and "layout 7" in rec["notes"][0]
    # a future PyAPS with no row in the matrix: no verdict, only a note
    rec = T.status(str(root2), pyaps_ver="9.0")
    assert rec["state"] == "match" and "no compatibility row" in rec["notes"][0]
    # wrong set for the machine
    root3 = stamped(tmp_path / "t3")
    assert T.status(str(root3), pyaps_ver=PYAPS, expected_set="full")["state"] == "incompatible"


def test_same_version_other_digest_is_flagged(tmp_path):
    root = stamped(tmp_path)
    ref = dict(T.read_manifest(str(root)), tree_digest="sha256:" + "0" * 64)
    rec = T.status(str(root), pyaps_ver=PYAPS, reference=ref)
    assert rec["state"] == "modified" and "tree digest" in " ".join(rec["notes"])


def test_exit_codes():
    assert T._exit_code({"state": "match"}) == 0
    assert T._exit_code({"state": "older"}) == 1
    assert T._exit_code({"state": "newer"}) == 0 and T._exit_code({"state": "newer"}, strict=True) == 1
    assert T._exit_code({"state": "missing"}) == 2 and T._exit_code({"state": "incompatible"}) == 2


# ------------------------------------------------------------------------------- locate / env
def test_locate_order(tmp_path, monkeypatch):
    a = tmp_path / "a" / "PyAPS_templates"
    b = tmp_path / "b" / "PyAPS_templates"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    monkeypatch.setenv("PYAPS_HOME", str(tmp_path / "b"))
    assert T.locate() == str(b)
    monkeypatch.setenv(T.ENV_DIR, str(a))
    assert T.locate() == str(a)
    assert T.locate(str(b)) == str(b)
    assert T.locate(str(tmp_path / "zzz")) is None


def test_locate_from_script_params(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    tdir = tmp_path / "data" / "PyAPS_templates"
    (tdir / "templates_RR").mkdir(parents=True)
    (cfg / "script_params.yaml").write_text(f"templates_RR: '{tdir}/templates_RR'\nother: 1\n")
    monkeypatch.setenv("PYAPS_CONFIGDIR", str(cfg))
    assert T.locate() == str(tdir)
    assert T.locate(None, {"templates_RR": f"{tdir}/templates_RR", "x": 1}) == str(tdir)


# ------------------------------------------------------------------------------- startup check
def test_startup_check_warns_once_and_never_walks(tmp_path, capsys, monkeypatch):
    bare = make_tree(tmp_path / "bare")
    monkeypatch.setenv(T.ENV_DIR, str(bare))
    T.startup_check(pyaps_ver=PYAPS)
    T.startup_check(pyaps_ver=PYAPS)
    err = capsys.readouterr().err
    assert err.count("WARNING") == 1 and "no TEMPLATES_MANIFEST.json" in err


def test_startup_check_refuses_only_a_layout_mismatch(tmp_path, monkeypatch, capsys):
    root = make_tree(tmp_path / "t")
    T.stamp(str(root), "full", "2026.9.0", "old", yes=True)                 # too old: warning only
    monkeypatch.setenv(T.ENV_DIR, str(root))
    T.startup_check(pyaps_ver=PYAPS)
    assert "older than the 2026.10.0" in capsys.readouterr().err
    with pytest.raises(T.TemplatesIncompatible, match="older"):
        T.startup_check(pyaps_ver=PYAPS, policy="strict")
    root2 = make_tree(tmp_path / "t2")
    T.stamp(str(root2), "mini", "2026.10.0", "x", yes=True, layout=2)
    monkeypatch.setenv(T.ENV_DIR, str(root2))
    with pytest.raises(T.TemplatesIncompatible, match="layout 2"):
        T.startup_check(pyaps_ver=PYAPS)
    T._WARNED.clear()
    T.startup_check(pyaps_ver=PYAPS, policy="warn")                             # configurable: continue at own risk
    assert "layout 2" in capsys.readouterr().err
    assert T.startup_check(pyaps_ver=PYAPS, policy="off") is None


def test_startup_check_without_directory_is_a_warning(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv(T.ENV_DIR, str(tmp_path / "gone"))
    rec = T.startup_check(pyaps_ver=PYAPS)
    assert rec["dir"] is None and "no template directory" in capsys.readouterr().err


# --------------------------------------------------------------------------- zip and install
def make_release(tmp_path, version="2026.10.0") -> Path:
    """A fake download area: <area>/mini/latest.json, <area>/mini/<version>/{zip,sha256}."""
    src = stamped(tmp_path / "src", "mini", version)
    man = T.read_manifest(str(src))
    area = tmp_path / "area"
    rel = area / "mini" / version
    rel.mkdir(parents=True)
    z = rel / f"PyAPS_templates_mini-{version}.zip"
    T.build_zip(str(src), man, str(z))
    (rel / (z.name + ".sha256")).write_text(f"{T.sha256_file(str(z))}  {z.name}\n")
    (area / "mini" / "latest.json").write_text(json.dumps({"set": "mini", "version": version, "path": f"mini/{version}"}))
    return area


def test_zip_is_deterministic_and_keeps_links(tmp_path):
    src = stamped(tmp_path / "s")
    man = T.read_manifest(str(src))
    a, b = tmp_path / "a.zip", tmp_path / "b.zip"
    T.build_zip(str(src), man, str(a))
    T.build_zip(str(src), man, str(b))
    assert T.sha256_file(str(a)) == T.sha256_file(str(b))
    with zipfile.ZipFile(a) as z:
        names = z.namelist()
        assert "PyAPS_templates/TEMPLATES_MANIFEST.json" in names and not any(".git" in n for n in names)


def test_install_roundtrip_verifies_and_restores_links(tmp_path):
    area = make_release(tmp_path)
    dest = tmp_path / "home" / "PyAPS_templates"
    rec = T.install("mini", str(dest), base_url=str(area), pyaps_ver=PYAPS)
    assert rec["applied"] and rec["version"] == "2026.10.0"
    assert os.path.islink(dest / "templates_RR" / "link.fits")
    assert T.status(str(dest), pyaps_ver=PYAPS, level="content")["state"] == "match"
    assert not [p for p in (tmp_path / "home").iterdir() if p.name.startswith(".templates_stage")]       # staging cleaned up


def test_install_refuses_to_overwrite_and_renames_on_upgrade(tmp_path):
    area = make_release(tmp_path)
    dest = tmp_path / "home" / "PyAPS_templates"
    T.install("mini", str(dest), base_url=str(area), pyaps_ver=PYAPS)
    with pytest.raises(T.TemplatesError, match="--upgrade"):
        T.install("mini", str(dest), base_url=str(area), pyaps_ver=PYAPS)
    rec = T.install("mini", str(dest), base_url=str(area), pyaps_ver=PYAPS, upgrade=True)
    assert rec["applied"] and Path(rec["previous"]).is_dir()                    # kept, not deleted


def test_install_detects_a_corrupt_download(tmp_path):
    area = make_release(tmp_path)
    z = next((area / "mini").rglob("*.zip"))
    z.write_bytes(z.read_bytes()[:-20] + b"x" * 20)
    with pytest.raises(T.TemplatesError, match="checksum mismatch"):
        T.install("mini", str(tmp_path / "d" / "PyAPS_templates"), base_url=str(area), pyaps_ver=PYAPS)
    assert not (tmp_path / "d" / "PyAPS_templates").exists()


def test_install_refuses_a_set_below_the_minimum(tmp_path):
    area = make_release(tmp_path, "2026.1.0")
    with pytest.raises(T.TemplatesError, match="below the minimum"):
        T.install("mini", str(tmp_path / "d" / "PyAPS_templates"), base_url=str(area), pyaps_ver=PYAPS)


def test_install_full_set_is_not_downloadable(tmp_path):
    with pytest.raises(T.TemplatesError, match="only the mini"):
        T.install("full", str(tmp_path / "x"), base_url=str(tmp_path))


# --------------------------------------------------------------------------------------- CLI
def test_cli_check_status_and_stamp(tmp_path, capsys):
    root = make_tree(tmp_path / "t")
    assert T.main(["stamp", str(root), "--set", "mini", "--version", "2026.10.0", "-m", "first"]) == 0
    assert not (root / T.MANIFEST_NAME).exists()                                  # no --yes: dry run
    assert T.main(["stamp", str(root), "--set", "mini", "--version", "2026.10.0", "-m", "first", "--yes"]) == 0
    capsys.readouterr()
    assert T.main(["status", "--dir", str(root), "--brief", "--pyaps-version", PYAPS]) == 0
    assert capsys.readouterr().out.strip() == "mini 2026.10.0 match"
    assert T.main(["check", "--dir", str(root), "--pyaps-version", PYAPS]) == 0
    (root / "top.txt").unlink()
    assert T.main(["check", "--dir", str(root), "--pyaps-version", PYAPS]) == 1
    assert T.main(["verify", str(root)]) == 1
    assert T.main(["check", "--dir", str(tmp_path / "none"), "--pyaps-version", PYAPS]) == 2
    out = capsys.readouterr().out
    assert "MISSING" not in out and "STATE" in out


def test_cli_manifest_prints_and_diff_compares(tmp_path, capsys):
    root = make_tree(tmp_path / "t")
    assert T.main(["manifest", str(root), "--set", "mini", "--version", "2026.10.0"]) == 0
    man = json.loads(capsys.readouterr().out)
    assert man["file_count"] == 5 and not (root / T.MANIFEST_NAME).exists()          # printing writes nothing
    out = tmp_path / "m1.json"
    assert T.main(["manifest", str(root), "--set", "mini", "--version", "2026.10.0", "--out", str(out)]) == 0
    (root / "top.txt").write_text("changed!")
    out2 = tmp_path / "m2.json"
    T.main(["manifest", str(root), "--set", "mini", "--version", "2026.11.0", "--out", str(out2)])
    capsys.readouterr()
    assert T.main(["diff", str(out), str(out2)]) == 1
    assert "changed: top.txt" in capsys.readouterr().out
    assert T.main(["diff", str(out), str(out)]) == 0


def test_cli_bad_input_is_a_clean_error(tmp_path, capsys):
    assert T.main(["manifest", str(tmp_path / "nope"), "--set", "mini", "--version", "2026.10.0"]) == 2
    assert "not a directory" in capsys.readouterr().err
    assert T.pyaps_main(["other"]) == 2


def test_requirements_override_file(tmp_path, monkeypatch):
    f = tmp_path / "req.json"
    f.write_text(json.dumps({"releases": [{"pyaps": ">=2.0,<3", "layout": 2, "min_version": {"mini": "2030.1.0"}}]}))
    monkeypatch.setenv("PYAPS_TEMPLATES_REQUIREMENTS", str(f))
    assert T.requirements_for("2.1")["layout"] == 2


def test_symlinked_directory_is_recorded_not_followed(tmp_path):
    root = make_tree(tmp_path / "t")
    os.symlink("templates_RR", root / "alias_dir")
    _, files = T.build_manifest(str(root), "mini", "2026.10.0")
    assert files["alias_dir"] == [0, "link:templates_RR"]
    assert not any(p.startswith("alias_dir/") for p in files)
