"""Public-hygiene guard (standing owner rule, 2026-10-07).

Tracked files must contain no secrets, internal hosts/IPs, personal account names or absolute
machine paths (including in module headers and ``__main__`` demo blocks). The scanner lives in
``tools/check_public_hygiene.py``; intentional exceptions are listed, with reasons, in
``tools/public_hygiene_allowlist.json``. See ``doc/PUBLIC_HYGIENE.md``.

Test strings below are assembled at run time so this file does not itself trip the scanner.
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("check_public_hygiene", ROOT / "tools" / "check_public_hygiene.py")
hyg = importlib.util.module_from_spec(_spec)
sys.modules["check_public_hygiene"] = hyg
_spec.loader.exec_module(hyg)


def _in_git_checkout():
    try:
        subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=ROOT, check=True, capture_output=True)
        return True
    except Exception:
        return False


def scan(text, rel="module.py", allow=()):
    return {rule for _, _, rule, _ in hyg.scan_text(rel, text, list(allow))}


# ----------------------------------------------------------------------------------------
# The repository itself
# ----------------------------------------------------------------------------------------
@pytest.mark.skipif(not _in_git_checkout(), reason="needs a git checkout")
def test_tracked_files_are_public_clean():
    findings = hyg.scan_files(hyg.tracked_files(), hyg.load_allowlist())
    assert not findings, "public hygiene violations (see doc/PUBLIC_HYGIENE.md):\n" + "\n".join(
        f"{p}:{i}: {rule}: {m}" for p, i, rule, m in findings[:50])


def test_allowlist_is_well_formed():
    data = json.loads(hyg.ALLOWLIST_FILE.read_text())
    for e in data["allow"]:
        assert e.get("reason", "").strip(), f"allow-list entry without a reason: {e}"
        assert e["rule"] not in hyg.NEVER_ALLOW, f"secrets can never be allow-listed: {e}"
        assert e.get("path"), e


def test_secret_rules_cannot_be_allowlisted(tmp_path, monkeypatch):
    bad = tmp_path / "allow.json"
    bad.write_text(json.dumps({"allow": [{"rule": "SECRET-TOKEN", "path": "*", "reason": "no"}]}))
    monkeypatch.setattr(hyg, "ALLOWLIST_FILE", bad)
    with pytest.raises(SystemExit):
        hyg.load_allowlist()


# ----------------------------------------------------------------------------------------
# Rule self-tests (positives)
# ----------------------------------------------------------------------------------------
def test_flags_site_absolute_paths():
    for root in ("home", "data", "scratch", "Users", "mnt", "srv", "local"):
        assert "ABS-PATH" in scan(f'x = "/{root}/someone/file.fit"\n'), root
    assert "ABS-PATH-WIN" in scan("p = 'C:" + "\\" + "Users" + "\\" + "me'\n")


def test_flags_internal_hosts_and_ips():
    assert "INFRA-HOST" in scan("ssh weave@" + "apm" + "63.example.org\n")
    assert "INFRA-HOST" in scan("host: aps." + "weave" + "master." + "arc" + "us\n")
    assert "INFRA-HOST" in scan("https://" + "gitlab" + ".ast.cam.ac.uk/x\n")
    assert "INFRA-IP" in scan("HOST = '" + ".".join(["131", "111", "5", "9"]) + "'\n")


def test_flags_secrets_and_masks_them():
    key = "-----BEGIN " + "OPENSSH PRIVATE KEY-----"
    assert "SECRET-KEY" in scan(key + "\n")
    assert "SECRET-TOKEN" in scan("t = '" + "ghp_" + "a1" * 20 + "'\n")
    assert "SECRET-URLCRED" in scan("wget " + "ftp://" + "someone:" + "hunter22@host.invalid/f\n")
    assert "SECRET-ASSIGN" in scan('password = "' + "Tr0ub4dor" + '&3x"\n')
    found = hyg.scan_text("a.py", 'password = "' + "Tr0ub4dor" + '&3x"\n', [])
    assert all("Tr0ub4dor" not in m for _, _, _, m in found), "secret values must be masked in reports"


def test_flags_personal_account_names_and_emails():
    assert "INTERNAL-NAME" in scan("user=" + "am" + "2931\n")
    assert "EMAIL" in scan("mail someone" + "@" + "example.org\n")


def test_flags_forbidden_tracked_files():
    for name in ("PyAPS_dms_config_x.json", "prod.env", "id_" + "rsa", "script_params_site.yaml"):
        assert hyg.check_name(name, []), name
    assert not hyg.check_name("PyAPS_dms_config.json.example", [])
    assert not hyg.check_name("redeploy.env.example", [])


# ----------------------------------------------------------------------------------------
# Rule self-tests (negatives: the placeholder conventions must pass)
# ----------------------------------------------------------------------------------------
def test_placeholders_pass():
    ok = (
        'infiles = ["<PYAPS_DATA>/L1/<night>/stack_<runid>.fit"]\n'
        "aps_runner --outpath $PYAPS_DATA/L2/<night>/<obid>/ --caldir ~/cal\n"
        'password = "<DB_PASSWORD>"\n'
        'token = os.environ["TOKEN"]\n'
        "docker run -v <PYAPS_DATA>:/data:ro image\n"
        "git clone git@github.com:org/repo.git\n"
        "host = '<DB_HOST>'   # 127.0.0.1 and 0.0.0.0 are fine\n"
        "x = '127.0.0.1'\n"
        "#!/usr/bin/env python3\n"
        "DATA = os.path.join(os.environ.get('PYAPS_DATA_DIR', '<PYAPS_DATA>'), 'L1')\n"
    )
    assert scan(ok) == set()


# ----------------------------------------------------------------------------------------
# Header and __main__ demo blocks are held to the stricter standard
# ----------------------------------------------------------------------------------------
def test_demo_blocks_reject_real_data_ids_and_any_absolute_path():
    src = (
        '"""Module header.\n\nExample:\n    run --infiles <PYAPS_DATA>/L1/<night>/stack_<runid>.fit\n"""\n'
        "import os\n\n"
        "def f():\n    return 1\n\n"
        "if __name__ == '__main__':\n"
        "    infiles = ['<PYAPS_DATA>/L1/20" + "240808/stack_" + "3071431.fit']\n"
        "    out = '/some/where/" + "else/results'\n"
    )
    rules = scan(src)
    assert "DEMO-DATA-ID" in rules and "DEMO-ABS-PATH" in rules
    # the very same strings outside header / __main__ are not demo-checked (code, not a demo)
    body = "import os\n\ndef f():\n    return '/some/where/" + "else/results', 'stack_" + "3071431'\n"
    assert not scan(body) & {"DEMO-DATA-ID", "DEMO-ABS-PATH"}


def test_header_with_real_identifiers_is_rejected():
    src = '"""Usage: tool --infiles stack_' + '3071431.fit --outpath <PYAPS_DIR>/' + 'results/20' + '240808/"""\nimport os\n'
    assert "DEMO-DATA-ID" in scan(src)


def test_demo_ranges_cover_header_and_main():
    src = '"""doc"""\n# banner\nimport os\n\nx = 1\n\nif __name__ == "__main__":\n    run()\n'
    ranges = hyg.python_demo_ranges("m.py", src)
    assert ranges[0][0] == 1 and ranges[0][1] >= 2
    assert ranges[-1][0] == 7


def test_every_aps_module_demo_block_is_scanned():
    """Guard the guard: every aps_*.py with a __main__ block must yield a demo range."""
    for p in sorted((ROOT / "py" / "PyAPS").glob("aps_*.py")):
        text = p.read_text(encoding="utf-8")
        if "__name__" in text and "__main__" in text:
            ranges = hyg.python_demo_ranges(str(p), text)
            assert ranges, p.name


# ----------------------------------------------------------------------------------------
# *.example templates may contain placeholders only; local configs are ignored by git
# ----------------------------------------------------------------------------------------
def test_example_templates_allow_only_placeholders_for_sensitive_keys():
    bad = "PYAPS_EXPLORER_WEAVEOR_" + "SECRET=Abc123longvalue\nDB_HOST=db" + ".example.org\n"
    rules = {r for _, _, r, _ in hyg.scan_text("configs/x.env.example", bad, [])}
    assert "EXAMPLE-VALUE" in rules
    good = (
        "PYAPS_EXPLORER_WEAVEOR_SECRET=<long random secret>\n"
        "PYAPS_EXPLORER_WEAVEOR_URL=<https URL of the upstream app>\n"
        "PYAPS_EXPLORER_MULTI_SESSION=1\n"
        "venv_path: '$HOME/venv/bin/activate'\n"
        '"password": "****",\n'
        "# DB_HOST=commented lines are ignored\n"
    )
    assert not {r for _, _, r, _ in hyg.scan_text("configs/x.env.example", good, [])} & {"EXAMPLE-VALUE"}
    # the same line in a non-template file is not judged by this rule
    assert "EXAMPLE-VALUE" not in {r for _, _, r, _ in hyg.scan_text("configs/x.cfg", bad, [])}


def test_example_value_rule_cannot_be_allowlisted():
    assert "EXAMPLE-VALUE" in hyg.NEVER_ALLOW


def test_shipped_templates_exist_and_local_configs_are_ignored():
    assert (ROOT / "configs" / "script_params.yaml.example").is_file()
    assert (ROOT / "configs" / "explorer.env.example").is_file()
    if not _in_git_checkout():
        return
    tracked = set(hyg.tracked_files())
    assert "configs/script_params.yaml" not in tracked, "the filled-in local file must not be tracked"
    for name in ("configs/script_params.yaml", "configs/explorer.env", "configs/PyAPS_dms_config.json",
                 "configs/script_params_mysite.yaml", "configs/pyaps_site.env", "configs/ACTIVE_SITE",
                 "configs/script_params.yaml.bak"):
        r = subprocess.run(["git", "check-ignore", "-q", name], cwd=ROOT)
        assert r.returncode == 0, f"{name} must be git-ignored"
    for name in ("configs/script_params.yaml.example", "configs/explorer.env.example"):
        r = subprocess.run(["git", "check-ignore", "-q", name], cwd=ROOT)
        assert r.returncode != 0, f"{name} (a template) must not be ignored"


def test_check_config_reports_unfilled_placeholders(tmp_path):
    spec = importlib.util.spec_from_file_location("check_config", ROOT / "tools" / "check_config.py")
    cc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cc)
    f = tmp_path / "local.env"
    f.write_text("A=<fill me in>\nB=ok\n# C=<commented>\nD=${SURELY_UNSET_VARIABLE_XYZ}\n")
    problems = cc.check_file(f)
    assert any("unfilled placeholder <fill me in>" in p for p in problems)
    assert any("SURELY_UNSET_VARIABLE_XYZ" in p for p in problems)
    assert not any("<commented>" in p for p in problems)
    assert cc.check_file(tmp_path / "missing.yaml")[0].endswith("missing")
