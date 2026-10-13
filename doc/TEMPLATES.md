# PyAPS template sets: versions, checks and download

PyAPS needs a large directory of templates and grids (`PyAPS_templates/`): redshift and archetype
templates, FERRE grids, RVS libraries, stellar-population models. It is far too big for git, so it
is **not** part of this repository. To know which version of it a machine uses, every template set
carries a small file, `TEMPLATES_MANIFEST.json`, that identifies it, and PyAPS checks it.

| Set | What it is | Size | Where from |
|---|---|---|---|
| `full` | the complete library of an operations site | terabytes | on request (contact the PyAPS admin); never published |
| `mini` | essential templates for tests, tutorials and viewer-only installs | about 100 MB | download (below) |

## Quick start

```bash
pyaps templates install --set mini            # downloads the latest mini set to ~/PyAPS/PyAPS_templates
pyaps templates check                        # which set, which version, does it match, is it compatible
```

Without installing PyAPS (from a checkout): `python tools/pyaps_templates.py check`. Everything below
works with either spelling; only the Python standard library is needed.

`install` downloads the zip and its `.sha256` from the download area, checks the checksum, extracts
into a temporary folder next to the destination, verifies **every file** against the manifest inside
the zip, and only then moves it into place. It refuses to touch a folder that already holds
templates unless you add `--upgrade`; then the old folder is **renamed** (`PyAPS_templates.<old
version>.<time>`), never deleted. `--version 2026.6.0` installs a specific version instead of the
latest; `--dry-run` shows what would happen. Manual download: the files are listed at the download
area (`<download area>/mini/latest.json` names the current zip); the previous link
`.../PyAPS_templates.tar.gz` keeps working and is the same content as mini 2026.6.0.

## Where PyAPS looks for the templates

1. `$PYAPS_TEMPLATES_DIR`, if set;
2. the folder named `PyAPS_templates` in the paths of your `script_params.yaml` (`templates_RR`, ...);
3. `$PYAPS_HOME/PyAPS_templates`, `$PYAPS_HOME/PyAPS_local/PyAPS_templates`, `~/PyAPS/PyAPS_templates`.

Use `--dir <folder>` on any command to look somewhere else.

## What identifies a set

* **set**: `full` or `mini`.
* **version**: calendar style `YYYY.MM.N` (`2026.10.0`); compared number by number, so `2026.10.0` is
  newer than `2026.9.0`. A new version means the content changed.
* **layout**: the compatibility *major* number. It changes only when PyAPS needs a different directory
  structure. A PyAPS release works with exactly one layout.
* **tree digest**: one SHA-256 over the whole file list (names, sizes, content hashes). Two copies with
  the same digest are the same set, byte for byte (for big files: head, middle and tail blocks, see below).
* **components**: each top-level folder (`templates_RR`, ...) has its own digest (`digest`) and a
  names-and-sizes digest (`shape`). The tree digest is computed from the component digests, so one
  folder can be checked without reading the others.

`TEMPLATES_MANIFEST.json` also holds the creation time, the file count and bytes, the parent version,
the PyAPS versions it was built for, the exclusion patterns, a change log, and (for the mini set)
every file with its SHA-256. For a big set the file list is the sidecar `TEMPLATES_FILES.json.gz`
(its checksum is in the manifest). Files up to 256 MiB are hashed completely (`sha256:`); a larger
file is fingerprinted from its size and three 1 MiB blocks (`s3:`), unless the manifest was built
with `--full`. Git metadata (`.git`), editor litter (`.DS_Store`, `._*`, `*~`), backups (`*.bak`) and
the bookkeeping files themselves are never part of a set.

## Status

`pyaps templates check` prints one of these states (exit code 0 for `match` and `newer`, 1 for a
warning, 2 for a problem); `pyaps templates status --json` prints the whole record for programs and
`--brief` the one-line form `<set> <version> <state>`.

| State | Meaning | Exit |
|---|---|---|
| `match` | manifest present, the files match it, and the version equals the reference (or none was given) | 0 |
| `newer` | version higher than the reference | 0 |
| `older` | version lower than the reference (a newer release exists) | 1 |
| `modified` | files were added, removed, resized or changed after the stamp; the differences are listed | 1 |
| `unstamped` | the folder has no manifest, so its version is unknown | 1 |
| `incompatible` | wrong layout, version below what this PyAPS needs, or not the set expected | 2 |
| `missing` | no template folder found | 2 |

By default `check` compares names and sizes (seconds, even for 100 000 files). `--level content`
also re-reads every file and compares the hashes. `--reference <manifest or folder>` supplies the
latest released manifest, which is what makes `older` and `newer` possible; the official sets are
published with such a manifest (`mini/latest.json` in the download area names it).

## Which PyAPS needs which templates

`pyaps templates info` prints the row that applies to your PyAPS.

| PyAPS | layout | minimum `full` | minimum `mini` |
|---|---|---|---|
| 2.x (`>=2.0,<3`) | 1 | 2026.10.0 | 2026.6.0 |

## What PyAPS does when it starts

The script generator (`aps_runner`) and `tools/check_env.py` look at the manifest of the template
folder named in the configuration. This reads one small file; it never walks the tree and never
writes anything.

* no folder, or no manifest, or a version below the minimum: **a warning** on stderr, once;
* a different **layout**: the program stops with a message telling you what to install.

`PYAPS_TEMPLATES_CHECK` changes that: `off` (no check), `warn` (only warnings, never stops),
`major` (the default: stops only on a layout mismatch), `strict` (also stops on a version below the
minimum). The aps_explorer shows `Templates: <set> <version>` under the PyAPS version when it finds a
manifest (in `$PYAPS_TEMPLATES_DIR` or `$PYAPS_CONFIGDIR`), and `/templatesz` returns the same as JSON.

## Operators: build, compare and stamp

```bash
pyaps templates manifest <folder> --set mini --version 2026.10.0            # prints the manifest, writes nothing
pyaps templates verify   <folder>                                            # folder against its own manifest
pyaps templates diff     <old manifest or folder> <new manifest or folder>   # components, added/removed/changed files
pyaps templates stamp    <folder> --set full --version 2026.11.0 -m "FR grids rebuilt" [--by <name>]   # dry run
pyaps templates stamp    ... --yes                                           # really writes
```

`stamp` is never silent: it needs the version and a message, shows the plan (parent, digest, which
components changed) and writes only with `--yes`. It refuses a version that is not higher than the
current one, a set name that differs from the existing manifest, and a stamp when nothing changed.
The previous manifest is kept next to the new one (`TEMPLATES_MANIFEST.<old version>.json`), and the
change log accumulates inside the manifest. The big set stays where it lives; only the manifest (a few
kilobytes, plus the sidecar) is versioned and compared.

Rules of thumb for versions: change the **month/number** when files are added, replaced or removed;
change the **layout** only together with a PyAPS release that reads a different structure, and
update the table in `aps_templates.REQUIREMENTS` and above in the same commit.
