# Public hygiene

PyAPS is a **public** repository. Standing rule (project owner, 2026-10-07):

> No security information, passwords or tokens, internal host names or IP addresses, and not even
> absolute machine paths may appear in code comments, docs, README, examples, config templates,
> test data or scripts.

The rule applies to every branch and every commit you push. It is enforced automatically (below).

## What is forbidden

| Class | Examples | Where it is checked |
|---|---|---|
| Secrets | private keys, `ghp_...`/`glpat-...` tokens, cloud access keys, URLs with a user and password embedded before the `@`, `password = "..."`, tracked `*.env`, `PyAPS_dms_config*.json`, `script_params_<site>.yaml`, `*.pem` | everywhere, **never** waivable |
| Internal hosts / IPs | machine names of the processing or archive hosts, cluster login nodes, Imperial/Cambridge institutional hosts, any IPv4 address (except `127.0.0.1`, `0.0.0.0`) | everywhere |
| Personal accounts / e-mails | login names of staff or service accounts, e-mail addresses of individuals (contributors too) | everywhere |
| Absolute machine paths | `/home/...`, `/data/...`, `/scratch/...`, `/Users/...`, `/mnt/...`, `/srv/...`, `C:\Users\...` | everywhere |
| Real data identifiers | night directories (`/20240808/`), run numbers (`stack_3071431.fit`), observation / plate directories (`/12958/`), real target names (`LWVE_...`) | **module headers and `__main__` demo blocks** of Python files |
| Any absolute path at all | `/some/where/results` | module headers and `__main__` demo blocks (stricter than the rest of the tree) |

## Placeholder conventions

Use these in docs, comments, examples, test data and demo blocks:

| Write | Meaning |
|---|---|
| `<PYAPS_DIR>` (prose, Python strings) / `$PYAPS_HOME` (shell) | the PyAPS checkout / installation |
| `<PYAPS_DATA>` (prose, Python strings) / `$PYAPS_DATA` (shell blocks) | the folder with your L1 / L2 / CAL / CAT data |
| `<OUTPUT_DIR>` | any output folder |
| `<night>`, `<runid>`, `<obid>`, `<target>` | night directory, run number, observation id, target name |
| `<DB_HOST>`, `<DB_USER>`, `<DB_PASSWORD>`, `<token>` | anything that would identify or open a service |
| `~/...`, `$HOME/...` | per-user locations |
| `/data` after a colon in `docker run -v <PYAPS_DATA>:/data:ro` | container-side mount point (not a host path) |

Runtime code never hard-codes a machine path. Resolve locations from, in this order, an explicit option,
an environment variable (`PYAPS_HOME`, `PYAPS_DATA_DIR`, `PYAPS_CONFIGDIR`, `PYAPS_CS_DIR`, ...), then a path
relative to the package or the user's home. Contributor contact details written into FITS headers come from
`PYAPS_CS_MAIL` (or `PYAPS_CS_MAIL_ALFA`, `PYAPS_CS_MAIL_AMY`, `PYAPS_CS_MAIL_SQUEZE`) and are empty by default.

## Demo blocks

The demo settings at the bottom of the `aps_*.py` modules (`if __name__ == '__main__':`) and the usage text
at the top of each file are scanned like everything else, and more strictly. A demo that needs real
locations defines them once, in a clearly marked block read from the environment:

```python
if __name__ == '__main__':
    # --- DEMO settings: edit for your setup (or export PYAPS_DATA_DIR / PYAPS_HOME) ---
    DEMO_DATA = os.environ.get('PYAPS_DATA_DIR', '<PYAPS_DATA>')
    infiles = [os.path.join(DEMO_DATA, 'L1/<night>/stack_<runid>.fit')]
```

## How the guard works

* `tools/check_public_hygiene.py` (standard library only) scans every **tracked** file with the rules above.
  It prints `path:line: RULE: matched text` (secret values are masked) and exits non-zero on any finding.
  * `python tools/check_public_hygiene.py` scans the whole tree (what CI and the test run).
  * `--staged` scans only staged files; `--history` is a report-only scan of every past revision;
    `--list-rules` prints the rules.
* `tests/test_public_hygiene.py` runs the scan inside `pytest` and unit-tests every rule (including that
  secrets are masked and can never be allow-listed, and that headers and `__main__` blocks are scanned).
* `.pre-commit-config.yaml` has a local `public-hygiene` hook (`pre-commit install`).
* `.github/workflows/ci.yml` runs the scanner and the test on every push and pull request to `main`, `dev`
  and `prod`.
* `.gitignore` keeps real host-local files (`PyAPS_dms_config*.json`, `script_params_<site>.yaml`, `*.env`,
  keys) out of the repository. Commit only templates named `*.example` with placeholder values.

## The allow-list

Intentional exceptions live in `tools/public_hygiene_allowlist.json`. Each entry has `rule` (or `*`),
`path` (glob), an optional `match` regular expression applied to the offending line, and a mandatory
`reason`. Secrets (`SECRET-*`, `FORBIDDEN-FILE`) can never be allow-listed. Current entries:

| Entry | Why it is allowed |
|---|---|
| the maintainer address `amolaei@ast.cam.ac.uk` | published contact for security reports and citation |
| the public sites `camcead.ast.cam.ac.uk`, `casu.ast.cam.ac.uk`, `www.ast.cam.ac.uk` | public download / calibration-query / project pages |
| `casuhelp@ast.cam.ac.uk` in `doc/weave_datamodel_v8.md` | public helpdesk address quoted as a FITS keyword value |
| `/home/pyaps`, `/var/lib/apt` in `Dockerfile` | paths inside the container image, not a host |

Vendored third-party bundles (`py/PyAPS/assets/vendor/`) and binary/data files are skipped, and the guard's
own three files (scanner, allow-list, test) are exempt from the pattern rules because they spell the patterns.

## Adding an example safely

1. Write the example with the placeholders above, never with a path you copied from a terminal.
2. Run `python tools/check_public_hygiene.py` (or just `pytest tests/test_public_hygiene.py`).
3. If a legitimate public item is flagged, prefer rewording. Only if that is impossible add a narrow
   allow-list entry (specific `path` and `match`) with a reason, and mention it in the pull request.
4. If you ever commit a secret by mistake, treat it as burned: rotate it first, then tell the maintainer.
   Deleting it in a later commit does not remove it from git history.
