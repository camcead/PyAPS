#!/usr/bin/env python3
"""Public-hygiene guard: fail if tracked files expose secrets, internal hosts/IPs or machine paths.

Standing rule (owner, 2026-10-07): this repository must contain no security information,
passwords, tokens, keys, internal host names or IPs, and no absolute machine paths in code
comments, docs, README, examples, config templates, test data or scripts.

Usage
-----
    python tools/check_public_hygiene.py              # scan every tracked file (CI / tests)
    python tools/check_public_hygiene.py --staged     # scan only staged files (pre-commit)
    python tools/check_public_hygiene.py --history    # report-only: scan every past revision
    python tools/check_public_hygiene.py --list-rules

Findings are printed as ``path:line: RULE: message`` with the matched text MASKED for secrets.
Intentional exceptions live in ``tools/public_hygiene_allowlist.json`` (see
``doc/PUBLIC_HYGIENE.md``); every entry needs a ``reason``. Standard library only.
"""
from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST_FILE = ROOT / "tools" / "public_hygiene_allowlist.json"

# --------------------------------------------------------------------------------------
# Rules.  Each rule: name -> (compiled regex, message).  Regexes are written so that the
# placeholders the project uses (<PYAPS_DATA>, $PYAPS_DATA, <DB_HOST>, ~/..., ${VAR}) never match.
# --------------------------------------------------------------------------------------
_NOT_PATH_BEFORE = r"(?<![\w.:/<>$~{}\-@=%])"   # a path must start at a "word boundary"

_SITE_ROOTS = (
    "home|data|local|Users|scratch|mnt|net|nfs|srv|root|opt|var|media|Volumes|work|workspace|"
    "PyAPS|raid|lustre|gpfs|ceph|archive|export|cephfs|projects|group"
)

# Findings that may be waived through the allow-list.
RULES = {
    "SECRET-KEY": (
        re.compile(r"-----BEGIN (?:[A-Z ]*)PRIVATE KEY-----|\bssh-(?:rsa|ed25519) AAAA[0-9A-Za-z+/]{20,}"),
        "private/public key material",
    ),
    "SECRET-TOKEN": (
        re.compile(
            r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}|\bgithub_pat_[A-Za-z0-9_]{30,}|\bglpat-[A-Za-z0-9_-]{15,}|"
            r"\bxox[abprs]-[A-Za-z0-9-]{10,}|\bAKIA[0-9A-Z]{16}\b|\bAIza[0-9A-Za-z_-]{35}\b|\bsk-[A-Za-z0-9]{32,}\b"
        ),
        "API token / access key pattern",
    ),
    "SECRET-URLCRED": (
        re.compile(r"\b[a-z][a-z0-9+.-]*://[^/\s:@'\"<>$]+:[^/\s@'\"<>$*]+@[^/\s'\"]+", re.I),
        "URL with embedded user:password",
    ),
    "SECRET-ASSIGN": (
        re.compile(
            r"""(?ix)\b(?:pass(?:word|wd)?|pwd|secret|api[_-]?key|access[_-]?key|auth[_-]?token|token)\w*["']?\s*[:=]\s*
                (["'])(?!\s*\1)(?P<v>[^"'\n]{6,})\1"""
        ),
        "hard-coded credential assignment",
    ),
    "INFRA-IP": (
        re.compile(r"(?<![\w.=<>~!])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\w.])"),
        "IPv4 address",
    ),
    "INFRA-HOST": (
        re.compile(
            r"(?i)(?<![\w])(?<!_at_)(?:apm\d{2}[\w.-]*|[\w.-]*weavemaster[\w.-]*|[\w.-]*moonsmaster[\w.-]*|[\w-]*\barcus\b[\w.-]*|"
            r"[\w-]+\.(?:ast\.cam\.ac\.uk|ic\.ac\.uk|cam\.ac\.uk|ucam\.org|hpc\.cam\.ac\.uk)|[\w.-]*\.(?:internal|corp|lan)|"
            r"[\w.-]*imperial[\w.-]*\.(?:ac\.uk|edu))\b"
        ),
        "internal / institutional host name",
    ),
    "EMAIL": (
        re.compile(r"(?<![\w.%+-])(?!git@)[A-Za-z0-9._%+-]+(?:@|_at_)[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"),
        "e-mail address",
    ),
    "INTERNAL-NAME": (
        re.compile(r"\b(?:am2931|dmurphy|am\d{4}|dm\d{3,4})\b", re.I),
        "personal / service account name",
    ),
    "ABS-PATH": (
        re.compile(_NOT_PATH_BEFORE + r"/(?:" + _SITE_ROOTS + r")(?=/|\b)(?![\w-])"),
        "absolute machine path",
    ),
    "ABS-PATH-WIN": (
        re.compile(r"(?<![\w])[A-Za-z]:(?:\\\\?|/)(?:Users|Documents|Program|Data|Work)\b"),
        "absolute Windows path",
    ),
}

# Stricter rules applied only to module headers and ``__main__`` demo blocks of Python files.
DEMO_RULES = {
    "DEMO-ABS-PATH": (
        re.compile(_NOT_PATH_BEFORE + r"/(?!(?:usr|bin|dev|tmp|sbin|path|dir|output)/)[A-Za-z_][\w.-]*/[\w.*-]+"),
        "absolute path in header/demo (use <PYAPS_DATA>/<PYAPS_DIR> or an env-driven DEMO_* variable)",
    ),
    "DEMO-DATA-ID": (
        re.compile(
            r"/20[12]\d(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])(?!\d)|"          # night directory
            r"(?<=[A-Za-z])_\d{6,7}(?!\d)|"                                    # run number in a file name
            r"\bLWVE_\d{8}[+-]\d{7}|"                                          # real target name
            r"/\d{3,6}/"                                                       # OB / plate directory
        ),
        "real night / run / observation identifier in header/demo (use <night>, <runid>, <obid>, <target>)",
    ),
}

# Tracked files that must never exist (real configs / credentials); name patterns, repo-relative.
FORBIDDEN_FILES = [
    "*.env", ".env", ".env.*", "*.pem", "*.key", "id_rsa*", "id_ed25519*", "*.p12", "*.pfx", "*.kdbx",
    "PyAPS_dms_config*.json", "*credentials*", ".netrc", "*.sqlite", "*.dump",
    "script_params_*.yaml", "script_params_*.yml",
]
# ... unless they are templates.
TEMPLATE_SUFFIXES = (".example", ".template", ".sample", ".dist")

SKIP_EXT = {
    ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".fits", ".fit", ".fz", ".dill", ".pkl", ".npy", ".npz", ".gz",
    ".tar", ".zip", ".ico", ".woff", ".woff2", ".ttf", ".h5", ".so", ".pyc", ".svg", ".bz2", ".xz", ".dat",
}
SKIP_PATHS = ("py/PyAPS/assets/vendor/",)
# The guard's own files (and its documentation) necessarily spell out the patterns they detect:
# they are exempt from the pattern rules (never from the secret rules).
SELF_FILES = ("tools/check_public_hygiene.py", "tools/public_hygiene_allowlist.json", "tests/test_public_hygiene.py",
              "doc/PUBLIC_HYGIENE.md")       # vendored third-party bundles (documented in the allow-list doc)
MAX_BYTES = 3_000_000

# Rules that may never be waived by the allow-list (a secret is never "intentional").
NEVER_ALLOW = {"SECRET-KEY", "SECRET-TOKEN", "SECRET-URLCRED", "SECRET-ASSIGN", "FORBIDDEN-FILE", "EXAMPLE-VALUE"}

# In *.example templates every value whose key looks sensitive must be a placeholder.
EXAMPLE_KEY = re.compile(r"(?i)secret|pass(?:word|wd)?\b|passw|pwd|token|api[_-]?key|_key\b|^key$|user|host|url|e?mail|addr|credential|dsn|connection")
EXAMPLE_LINE = re.compile(r"""^\s*["']?([A-Za-z_][\w.-]*)["']?\s*[:=]\s*(.*?),?\s*$""")
EXAMPLE_OK = re.compile(r"""^["']?\s*$|<[^<>]+>|\$\{?[A-Za-z_(]|\*{3,}|^["']?(?:None|none|null|True|False|true|false|\d+)["']?$""")

PLACEHOLDER_HINT = re.compile(r"(?i)<[^>]+>|\$\{?[A-Za-z_]|\*{3,}|\.{3}|xxx|changeme|your[_-]|example|placeholder|dummy|not-for-production|fake|test")


def load_allowlist() -> list[dict]:
    if not ALLOWLIST_FILE.exists():
        return []
    data = json.loads(ALLOWLIST_FILE.read_text())
    entries = data.get("allow", [])
    for e in entries:
        if not e.get("reason"):
            raise SystemExit(f"allow-list entry without a reason: {e}")
        if e.get("rule") in NEVER_ALLOW:
            raise SystemExit(f"rule {e['rule']} cannot be allow-listed: {e}")
    return entries


def is_allowed(allow: list[dict], rule: str, path: str, text: str) -> bool:
    if rule in NEVER_ALLOW:
        return False
    for e in allow:
        if e["rule"] not in (rule, "*"):
            continue
        if not fnmatch.fnmatch(path, e["path"]):
            continue
        pat = e.get("match")
        if pat is None or re.search(pat, text):
            return True
    return False


def tracked_files(staged: bool = False) -> list[str]:
    cmd = ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"] if staged else ["git", "ls-files"]
    out = subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True).stdout
    return [p for p in out.splitlines() if p]


def read_text(path: Path) -> str | None:
    try:
        if path.stat().st_size > MAX_BYTES:
            return None
        raw = path.read_bytes()
    except OSError:
        return None
    if b"\0" in raw[:4096]:
        return None
    return raw.decode("utf-8", errors="replace")


def python_demo_ranges(rel: str, text: str) -> list[tuple[int, int]]:
    """1-based inclusive line ranges of the module header and the ``__main__`` demo block."""
    lines = text.splitlines()
    n = len(lines)
    header_end = None
    main_start = None
    try:
        tree = ast.parse(text)
        body = list(tree.body)
        first = body[0] if body else None
        if first is not None and isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) \
                and isinstance(first.value.value, str):
            body = body[1:]
        header_end = (body[0].lineno - 1) if body else n
        for node in tree.body:
            if isinstance(node, ast.If) and "__name__" in ast.dump(node.test) and "__main__" in ast.dump(node.test):
                main_start = node.lineno
                break
    except SyntaxError:
        for i, ln in enumerate(lines, 1):
            if header_end is None and re.match(r"(import|from|def|class)\s", ln):
                header_end = i - 1
            if main_start is None and re.match(r"if\s+__name__\s*==", ln):
                main_start = i
        header_end = header_end if header_end is not None else min(n, 120)
    ranges = []
    if header_end:
        ranges.append((1, header_end))
    if main_start:
        ranges.append((main_start, n))         # to end of file: also catches trailing commented demos
    return ranges


def mask(rule: str, s: str) -> str:
    if rule.startswith("SECRET"):
        return s[:3] + "***MASKED***"
    return s if len(s) <= 60 else s[:57] + "..."


def scan_text(rel: str, text: str, allow: list[dict]) -> list[tuple[str, int, str, str]]:
    findings: list[tuple[str, int, str, str]] = []
    self_file = rel in SELF_FILES
    lines = text.splitlines()
    demo_lines: set[int] = set()
    if rel.endswith(".py") or Path(rel).parent.name == "bin" and lines and lines[0].startswith("#!") and "python" in lines[0]:
        for a, b in python_demo_ranges(rel, text):
            demo_lines.update(range(a, b + 1))

    def emit(rule, i, line, m):
        if is_allowed(allow, rule, rel, line):
            return
        findings.append((rel, i, rule, mask(rule, m.group(0))))

    for i, line in enumerate(lines, 1):
        if len(line) > 5000:                      # minified blobs
            continue
        for rule, (rx, _msg) in RULES.items():
            if self_file and not rule.startswith("SECRET"):
                continue
            for m in rx.finditer(line):
                if rule == "SECRET-ASSIGN" and PLACEHOLDER_HINT.search(m.group("v")):
                    continue
                if rule == "INFRA-IP" and m.group(0) in {"127.0.0.1", "0.0.0.0", "255.255.255.255"}:
                    continue
                if rule == "INFRA-IP" and re.match(r"(?:192\.0\.2|198\.51\.100|203\.0\.113)\.", m.group(0)):
                    continue
                emit(rule, i, line, m)
        if rel.endswith(TEMPLATE_SUFFIXES) and not line.lstrip().startswith("#"):
            m = EXAMPLE_LINE.match(line)
            if m and EXAMPLE_KEY.search(m.group(1)) and not EXAMPLE_OK.search(m.group(2)):
                findings.append((rel, i, "EXAMPLE-VALUE", f"{m.group(1)}: <non-placeholder value>"))
        if i in demo_lines and not self_file:
            for rule, (rx, _msg) in DEMO_RULES.items():
                for m in rx.finditer(line):
                    emit(rule, i, line, m)
    return findings


def check_name(rel: str, allow: list[dict]):
    base = Path(rel).name
    if base.endswith(TEMPLATE_SUFFIXES):
        return []
    for pat in FORBIDDEN_FILES:
        if fnmatch.fnmatch(base, pat):
            return [(rel, 0, "FORBIDDEN-FILE", f"tracked file matches {pat}")]
    return []


def scan_files(files: list[str], allow: list[dict]):
    out = []
    for rel in files:
        out += check_name(rel, allow)
        p = Path(rel)
        if any(rel.startswith(s) for s in SKIP_PATHS) or p.suffix.lower() in SKIP_EXT:
            continue
        full = ROOT / rel
        if not full.is_file() or full.is_symlink():
            continue
        text = read_text(full)
        if text is None:
            continue
        out += scan_text(rel, text, allow)
    return out


def scan_history(allow: list[dict]):
    revs = subprocess.run(["git", "rev-list", "--all"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    seen: dict[tuple, set] = {}
    for rev in revs:
        files = subprocess.run(["git", "ls-tree", "-r", "--name-only", rev], cwd=ROOT, capture_output=True, text=True).stdout.splitlines()
        for rel in files:
            p = Path(rel)
            if any(rel.startswith(s) for s in SKIP_PATHS) or p.suffix.lower() in SKIP_EXT:
                continue
            blob = subprocess.run(["git", "show", f"{rev}:{rel}"], cwd=ROOT, capture_output=True).stdout
            if len(blob) > MAX_BYTES or b"\0" in blob[:4096]:
                continue
            for f in scan_text(rel, blob.decode("utf-8", "replace"), allow):
                seen.setdefault((f[0], f[2], f[3]), set()).add(rev[:7])
    return seen


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--staged", action="store_true", help="scan only staged files")
    ap.add_argument("--history", action="store_true", help="report-only scan of every revision")
    ap.add_argument("--list-rules", action="store_true")
    ap.add_argument("files", nargs="*", help="explicit files (default: all tracked)")
    a = ap.parse_args(argv)
    if a.list_rules:
        for k, (_, msg) in {**RULES, **DEMO_RULES}.items():
            print(f"{k:16s} {msg}")
        print(f"{'FORBIDDEN-FILE':16s} tracked file that should be host-local (templates: *.example)")
        print(f"{'EXAMPLE-VALUE':16s} *.example template with a real-looking value for a sensitive key (only <...> placeholders allowed)")
        return 0
    allow = load_allowlist()
    if a.history:
        seen = scan_history(allow)
        for (rel, rule, m), revs in sorted(seen.items()):
            print(f"{rel}: {rule}: {m}  [{len(revs)} revisions]")
        print(f"{len(seen)} distinct historical findings (report only)")
        return 0
    files = a.files or tracked_files(a.staged)
    findings = scan_files(files, allow)
    for rel, i, rule, m in findings:
        print(f"{rel}:{i}: {rule}: {m}")
    if findings:
        print(f"\npublic hygiene check FAILED: {len(findings)} finding(s). See doc/PUBLIC_HYGIENE.md "
              f"(placeholders, allow-list).", file=sys.stderr)
        return 1
    print("public hygiene check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
