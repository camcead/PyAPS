"""
aps_ifu_tools.py
================
Auxiliary tools for the WEAVE IFU pipeline.

This module is intentionally structured as a growing toolbox — each
logical group of functions lives in its own clearly-labelled section.
New tool groups can be appended without touching existing ones.

Current tool groups
-------------------
  SECTION 1 — Patch Table Explorer / Reporter
      explore_patch_table     : auto-find and summarise patch files
      inspect_row             : detailed report for a single row

  SECTION 2 — Patch Table Editor
      set_redshift            : update Z / ZERR / ZWARN for a row
      set_class               : update CLASS for a row
      set_type                : change type flag (C / T / M)
      remove_row              : drop a row by id
      add_mask_region         : append a new mask (type=M) entry
      save_patch_table        : write table back to FITS or ASCII
      edit_patch_table        : interactive CLI editor loop

  SECTION 3 — (reserved for future tools)

Usage examples
--------------
  from PyAPS.aps_ifu_tools import explore_patch_table, edit_patch_table

  # Explore
  result = explore_patch_table('WA_P0001', '<PYAPS_DATA>/L2/<night>/<obid>/')

  # Grab the base table and tweak a row
  tbl = result['targets']
  tbl = set_redshift(tbl, row_id=2, z=0.045, zerr=0.001)
  tbl = set_class(tbl, row_id=2, class_str='GALAXY')
  save_patch_table(tbl, result['targets_path'])

  # Or use the interactive CLI
  edit_patch_table('WA_P0001', '<PYAPS_DATA>/L2/<night>/<obid>/')
"""

import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table

# ---------------------------------------------------------------------------
# Internal import — reuse the existing ASCII read/write helpers.
# If aps_ifu_prepare is not available yet, fall back to inline definitions.
# ---------------------------------------------------------------------------
try:
    from PyAPS.aps_ifu_prepare import read_ascii_patchfile, write_ascii_patchfile
except ImportError:
    # Inline fallbacks so this file can be used standalone
    import pandas as pd
    from astropy.table import hstack

    def _convert_to_csv(column):
        if isinstance(column[0], (list, np.ndarray)):
            return [",".join(map(str, arr)).replace('"', "") for arr in column]
        return column

    def write_ascii_patchfile(patch_table, patch_output):
        ready = {c: _convert_to_csv(patch_table[c]) for c in patch_table.columns}
        from astropy.io import ascii as asc
        asc.write(
            Table(ready), Path(patch_output),
            overwrite=True, quotechar=" ",
            delimiter=" ", format="commented_header",
        )

    def read_ascii_patchfile(patch_file):
        column_names = [
            "id", "RA_icrs", "DEC_icrs", "A_world", "B_world",
            "angle", "flag", "type", "Z", "ZERR", "ZWARN", "CLASS",
        ]
        df = pd.read_csv(
            patch_file, sep=r"\s+", comment="#",
            header=None, names=column_names,
        )
        df["CLASS"] = df["CLASS"].str.split(",")
        df["Z"]     = df["Z"].apply(
            lambda x: np.array(list(map(float, x.split(",")))))
        df["ZERR"]  = df["ZERR"].apply(
            lambda x: np.array(list(map(float, x.split(",")))))
        df["ZWARN"] = df["ZWARN"].apply(
            lambda x: np.array(list(map(float, x.split(",")))))
        pt = Table.from_pandas(df)
        lpt = len(pt)
        _ntop = len(pt[0]["CLASS"])
        pc = Table(
            [
                [[np.nan] * _ntop for _ in range(lpt)],
                [[np.nan] * _ntop for _ in range(lpt)],
                [[np.nan] * _ntop for _ in range(lpt)],
                [[""] * _ntop for _ in range(lpt)],
            ],
            names=("Z", "ZERR", "ZWARN", "CLASS"),
            dtype=("float", "float", "float", "U18"),
        )
        for col in ["CLASS", "Z", "ZWARN", "ZERR"]:
            for r in range(lpt):
                for c in range(_ntop):
                    pc[r][col][c] = pt[r][col][c]
        return hstack(
            [pt[["id", "RA_icrs", "DEC_icrs", "A_world",
                 "B_world", "angle", "flag", "type"]], pc]
        )


# =========================================================================== #
#  SECTION 1 — Patch Table Explorer / Reporter                                #
# =========================================================================== #

# Expected column names for a valid patch table
_PATCH_COLUMNS = [
    "id", "RA_icrs", "DEC_icrs", "A_world", "B_world",
    "angle", "flag", "type", "Z", "ZERR", "ZWARN", "CLASS",
]

# Display width for section separators
_W = 72


def _separator(char="="):
    return char * _W


def _decode(val) -> str:
    """
    Safely convert a CLASS value to a plain str.

    FITS string columns are often read back as bytes (b'GALAXY').
    Handles bytes, numpy bytes_, plain str, and nan-like values.
    """
    if isinstance(val, (bytes, np.bytes_)):
        val = val.decode("utf-8", errors="replace")
    val = str(val).strip()
    # Strip residual b'...' wrapping that sometimes survives str()
    if val.startswith("b'") and val.endswith("'"):
        val = val[2:-1]
    if val.upper() in ("NAN", "NONE", "B'NAN'", "B'NONE'", "B''", ""):
        return ""
    return val


def _load_patch_file(path: Path) -> Table:
    """Load a patch file regardless of format (FITS or ASCII)."""
    if path.suffix in [".fits", ".fit"]:
        tbl = Table.read(str(path))
        # Fill masked columns
        for col in tbl.colnames:
            if hasattr(tbl[col], "mask"):
                tbl[col] = tbl[col].filled(np.nan)
    else:
        tbl = read_ascii_patchfile(str(path))
    return tbl


def _find_patch_files(headname: str, outpath: str) -> dict:
    """
    Search *outpath* for patch files matching *headname*.

    Looks for both FITS and ASCII variants of the base and _mod files.
    Returns a dict:
        {
          'targets':      Path | None,
          'targets_mod':  Path | None,
        }
    """
    base    = Path(outpath)
    results = {"targets": None, "targets_mod": None}

    for suffix in [".fits", ".fit", ".dat", ".txt", ""]:
        if results["targets"] is None:
            p = base / (headname + "_targets" + suffix)
            if p.exists():
                results["targets"] = p

        if results["targets_mod"] is None:
            p = base / (headname + "_targets_mod" + suffix)
            if p.exists():
                results["targets_mod"] = p

    return results


def _validate_table(tbl: Table) -> list:
    """
    Run sanity checks on a patch table.

    Returns a (possibly empty) list of warning strings.
    """
    warnings = []

    # Column check
    missing = [c for c in _PATCH_COLUMNS if c not in tbl.colnames]
    if missing:
        warnings.append(f"Missing columns: {missing}")

    # Duplicate IDs
    ids = list(tbl["id"])
    if len(ids) != len(set(ids)):
        from collections import Counter
        dups = [k for k, v in Counter(ids).items() if v > 1]
        warnings.append(f"Duplicate IDs: {dups}")

    # NaN redshift on non-mask targets
    for row in tbl:
        if str(row["type"]).strip().upper() == "M":
            continue
        z_vals = np.atleast_1d(row["Z"])
        if np.all(np.isnan(z_vals.astype(float))):
            warnings.append(
                f"Row id={row['id']} type={row['type'].strip()}: "
                f"all Z values are NaN"
            )

    # ZWARN > 0
    for row in tbl:
        zw = np.atleast_1d(row["ZWARN"]).astype(float)
        valid_zw = zw[~np.isnan(zw)]
        if len(valid_zw) > 0 and np.any(valid_zw > 0):
            warnings.append(
                f"Row id={row['id']}: ZWARN > 0  ({valid_zw})"
            )

    # Zero-size ellipses
    for row in tbl:
        if float(row["A_world"]) == 0 or float(row["B_world"]) == 0:
            warnings.append(
                f"Row id={row['id']}: zero-size ellipse "
                f"(A={row['A_world']}, B={row['B_world']})"
            )

    return warnings


def _table_summary(tbl: Table, label: str) -> None:
    """Print a compact summary of one patch table."""
    print(_separator())
    print(f"  {label}")
    print(_separator())
    print(f"  Rows      : {len(tbl)}")

    # Type distribution
    types = {}
    for row in tbl:
        t = str(row["type"]).strip().upper()
        types[t] = types.get(t, 0) + 1
    type_str = "  ".join(f"{k}:{v}" for k, v in sorted(types.items()))
    print(f"  Types     : {type_str}")

    # Class distribution (first class entry only)
    classes = {}
    for row in tbl:
        cl = np.atleast_1d(row["CLASS"])
        c  = _decode(cl[0]).upper() if len(cl) > 0 else "?"
        if not c:
            c = "unset"
        classes[c] = classes.get(c, 0) + 1
    class_str = "  ".join(f"{k}:{v}" for k, v in sorted(classes.items()))
    print(f"  Classes   : {class_str}")

    # Redshift stats (non-NaN, non-mask)
    z_vals = []
    for row in tbl:
        if str(row["type"]).strip().upper() == "M":
            continue
        z = np.atleast_1d(row["Z"]).astype(float)
        valid = z[~np.isnan(z)]
        if len(valid) > 0:
            z_vals.append(valid[0])
    if z_vals:
        print(
            f"  Z range   : {min(z_vals):.4f} – {max(z_vals):.4f}  "
            f"(median={float(np.median(z_vals)):.4f})"
        )
    else:
        print("  Z range   : all NaN")

    # ZWARN summary
    n_zwarn = sum(
        1 for row in tbl
        if np.any(np.atleast_1d(row["ZWARN"]).astype(float) > 0)
    )
    print(f"  ZWARN > 0 : {n_zwarn} rows")

    # Validation warnings
    warnings = _validate_table(tbl)
    if warnings:
        print(f"\n  *** {len(warnings)} warning(s) ***")
        for w in warnings:
            print(f"    ! {w}")
    else:
        print("  Validation: OK")

    print()


def _table_detail(tbl: Table) -> None:
    """Print a per-row detail table."""
    print(_separator("-"))
    _a_col = 'A"'
    _b_col = 'B"'
    header = (
        f"  {'id':>4}  {'type':>4}  "
        f"{'RA':>12}  {'Dec':>12}  "
        f"{_a_col:>7}  {_b_col:>7}  "
        f"{'CLASS':<12}  {'Z':>8}  {'ZWARN':>6}"
    )
    print(header)
    print(_separator("-"))

    for row in tbl:
        cl   = np.atleast_1d(row["CLASS"])
        c    = _decode(cl[0]) if len(cl) > 0 else "?"
        if not c:
            c = "—"

        z    = np.atleast_1d(row["Z"]).astype(float)
        zv   = z[~np.isnan(z)]
        zstr = f"{zv[0]:.5f}" if len(zv) > 0 else "NaN"

        zw   = np.atleast_1d(row["ZWARN"]).astype(float)
        zwv  = zw[~np.isnan(zw)]
        zwstr = f"{int(zwv[0])}" if len(zwv) > 0 else "—"

        print(
            f"  {row['id']:>4}  {str(row['type']).strip():>4}  "
            f"{float(row['RA_icrs']):>12.6f}  "
            f"{float(row['DEC_icrs']):>12.6f}  "
            f"{float(row['A_world'])*3600:>7.2f}  "
            f"{float(row['B_world'])*3600:>7.2f}  "
            f"{c:<12}  {zstr:>8}  {zwstr:>6}"
        )
    print()


def explore_patch_table(headname: str, outpath: str, detail: bool = True) -> dict:
    """
    Auto-find and report on patch table files for a given *headname*.

    Searches *outpath* for ``{headname}_targets[_mod][.fits/.dat]``,
    loads each file it finds, prints a human-readable report, and returns
    the loaded tables so you can immediately pass them to the edit functions.

    Parameters
    ----------
    headname : str
        IFU output headname (e.g. ``'WA_P0001'``).
    outpath : str
        Directory to search.
    detail : bool
        If ``True`` (default), print a per-row detail table for each file.

    Returns
    -------
    dict with keys:

    ``targets``
        Loaded base patch table (Astropy Table), or ``None``.
    ``targets_mod``
        Loaded ``_mod`` patch table, or ``None``.
    ``targets_path``
        Path to the base file (str), or ``None``.
    ``targets_mod_path``
        Path to the ``_mod`` file (str), or ``None``.
    """
    print()
    print(_separator())
    print(f"  PATCH TABLE EXPLORER")
    print(f"  headname : {headname}")
    print(f"  outpath  : {outpath}")
    print(_separator())

    found = _find_patch_files(headname, outpath)

    result = {
        "targets":          None,
        "targets_mod":      None,
        "targets_path":     None,
        "targets_mod_path": None,
    }

    if found["targets"] is None and found["targets_mod"] is None:
        print(f"\n  *** No patch files found for headname '{headname}' ***")
        print(f"  Searched: {outpath}")
        print()
        return result

    for key, label in [("targets", "BASE TABLE"), ("targets_mod", "MODIFIED TABLE (_mod)")]:
        path = found[key]
        if path is None:
            print(f"\n  {label}: not found")
            continue

        print(f"\n  {label}: {path.name}")

        try:
            tbl = _load_patch_file(path)
        except Exception as e:
            print(f"  *** Failed to load {path}: {e}")
            continue

        result[key]              = tbl
        result[key + "_path"]    = str(path)

        _table_summary(tbl, f"{label}  [{path.name}]")
        if detail:
            _table_detail(tbl)

    print(_separator())
    print()
    return result


def inspect_row(tbl: Table, row_id: int) -> None:
    """
    Print a detailed report for a single row identified by *row_id*.

    Parameters
    ----------
    tbl : astropy.table.Table
        A loaded patch table.
    row_id : int
        The ``id`` value of the row to inspect (not the array index).
    """
    matches = [i for i, r in enumerate(tbl) if int(r["id"]) == row_id]
    if not matches:
        print(f"  Row id={row_id} not found in table.")
        return

    row = tbl[matches[0]]

    print()
    print(_separator())
    print(f"  ROW DETAIL  id={row_id}")
    print(_separator())
    print(f"  type         : {str(row['type']).strip()}")
    print(f"  RA_icrs      : {float(row['RA_icrs']):.8f} deg")
    print(f"  DEC_icrs     : {float(row['DEC_icrs']):.8f} deg")
    print(f"  A_world      : {float(row['A_world'])*3600:.4f} arcsec")
    print(f"  B_world      : {float(row['B_world'])*3600:.4f} arcsec")
    print(f"  angle        : {float(row['angle']):.3f} deg")
    print(f"  flag         : {row['flag']}")

    # Multi-entry arrays
    for arr_col in ["Z", "ZERR", "ZWARN", "CLASS"]:
        vals = np.atleast_1d(row[arr_col])
        if arr_col == "CLASS":
            formatted = "  ".join(_decode(v) or "—" for v in vals)
        else:
            formatted = "  ".join(str(v).strip() for v in vals)
        print(f"  {arr_col:<12} : [{formatted}]")

    print(_separator())
    print()


# =========================================================================== #
#  SECTION 2 — Patch Table Editor                                             #
# =========================================================================== #

def _row_index(tbl: Table, row_id: int) -> int:
    """Return the array index of the row with id==row_id, or raise."""
    for i, row in enumerate(tbl):
        if int(row["id"]) == row_id:
            return i
    raise KeyError(f"Row id={row_id} not found in patch table")


def set_redshift(
    tbl: Table,
    row_id: int,
    z: float,
    zerr: float,
    zwarn: int = 0,
    slot: int = 0,
) -> Table:
    """
    Update the redshift for a row.

    Parameters
    ----------
    tbl : Table
        Patch table to modify (modified in-place and returned).
    row_id : int
        ``id`` of the row to update.
    z : float
        New redshift value.
    zerr : float
        New redshift uncertainty.
    zwarn : int
        New ZWARN flag (default 0).
    slot : int
        Which slot in the Z/ZERR/ZWARN arrays to update (default 0).

    Returns
    -------
    Table
        The modified table.
    """
    idx = _row_index(tbl, row_id)
    tbl[idx]["Z"][slot]     = float(z)
    tbl[idx]["ZERR"][slot]  = float(zerr)
    tbl[idx]["ZWARN"][slot] = int(zwarn)
    print(
        f"  Row id={row_id} slot={slot}: "
        f"Z={z:.6f}  ZERR={zerr:.6f}  ZWARN={zwarn}"
    )
    return tbl


def set_class(
    tbl: Table,
    row_id: int,
    class_str: str,
    slot: int = 0,
) -> Table:
    """
    Update the CLASS for a row.

    Parameters
    ----------
    tbl : Table
        Patch table to modify.
    row_id : int
        ``id`` of the row to update.
    class_str : str
        New class string (e.g. ``'GALAXY'``, ``'STAR'``, ``'QSO'``).
    slot : int
        Which slot in the CLASS array to update (default 0).

    Returns
    -------
    Table
    """
    class_str = _decode(class_str)   # normalise bytes -> str if needed
    valid = {"GALAXY", "QSO", "STAR", "WD", ""}
    if class_str.upper() not in valid and class_str != "":
        print(
            f"  Warning: class_str='{class_str}' is not a standard value. "
            f"Standard values are: {sorted(valid)}"
        )
    idx = _row_index(tbl, row_id)
    tbl[idx]["CLASS"][slot] = class_str.strip()
    print(f"  Row id={row_id} slot={slot}: CLASS set to '{class_str}'")
    return tbl


def set_type(tbl: Table, row_id: int, type_str: str) -> Table:
    """
    Change the type flag for a row.

    Parameters
    ----------
    tbl : Table
        Patch table to modify.
    row_id : int
        ``id`` of the row to update.
    type_str : str
        New type: ``'C'`` (central), ``'T'`` (target), or ``'M'`` (mask).

    Returns
    -------
    Table
    """
    type_str = type_str.strip().upper()
    if type_str not in ("C", "T", "M"):
        raise ValueError(f"type_str must be C, T, or M — got '{type_str}'")
    idx = _row_index(tbl, row_id)
    tbl[idx]["type"] = type_str
    print(f"  Row id={row_id}: type set to '{type_str}'")
    return tbl


def remove_row(tbl: Table, row_id: int) -> Table:
    """
    Remove a row from the patch table by id.

    Parameters
    ----------
    tbl : Table
    row_id : int

    Returns
    -------
    Table
        New table with the row removed.
    """
    idx = _row_index(tbl, row_id)
    tbl.remove_row(idx)
    print(f"  Row id={row_id} removed  ({len(tbl)} rows remaining)")
    return tbl


def add_mask_region(
    tbl: Table,
    ra: float,
    dec: float,
    radius_arcsec: float,
    angle: float = 0.0,
) -> Table:
    """
    Append a new mask entry (type=M) to the patch table.

    Parameters
    ----------
    tbl : Table
    ra, dec : float
        Centre of the mask region in degrees (ICRS).
    radius_arcsec : float
        Radius of the circular mask in arcsec.
    angle : float
        Position angle in degrees (default 0).

    Returns
    -------
    Table
    """
    new_id    = int(np.max(tbl["id"])) + 1
    radius_deg = radius_arcsec / 3600.0

    # Work out the shape of the CLASS/Z/ZERR/ZWARN arrays from an existing row
    ntop = len(np.atleast_1d(tbl[0]["CLASS"]))

    new_row = {
        "id":       new_id,
        "RA_icrs":  ra,
        "DEC_icrs": dec,
        "A_world":  radius_deg,
        "B_world":  radius_deg,
        "angle":    angle,
        "flag":     0,
        "type":     "M",
        "Z":        [np.nan] * ntop,
        "ZERR":     [np.nan] * ntop,
        "ZWARN":    [np.nan] * ntop,
        "CLASS":    [""] * ntop,
    }
    tbl.add_row(new_row)
    print(
        f"  Mask region added  id={new_id}  "
        f"RA={ra:.6f}  Dec={dec:.6f}  r={radius_arcsec:.1f}\""
    )
    return tbl


def save_patch_table(tbl: Table, path: str, overwrite: bool = True) -> None:
    """
    Write a patch table back to disk.

    Detects format from the file extension: FITS for ``.fits`` / ``.fit``,
    ASCII for everything else.

    Parameters
    ----------
    tbl : Table
    path : str
        Full output path including filename and extension.
    overwrite : bool
        Overwrite an existing file (default True).
    """
    path = Path(path)

    if not overwrite and path.exists():
        raise FileExistsError(
            f"{path} already exists. Pass overwrite=True to overwrite."
        )

    if path.suffix in [".fits", ".fit"]:
        hdu = fits.table_to_hdu(tbl)
        hdu.writeto(str(path), overwrite=overwrite, checksum=True)
    else:
        write_ascii_patchfile(tbl, path)

    print(f"  Patch table saved: {path}")


def edit_patch_table(headname: str, outpath: str) -> None:
    """
    Interactive CLI editor for patch tables.

    Loads the patch files for *headname*, prints the explorer report,
    then enters a command loop that lets you modify rows and save.

    Commands
    --------
    list              — re-print the current table
    inspect <id>      — detailed view of one row
    z <id> <z> <zerr> [zwarn] [slot]
                      — set redshift for a row
    class <id> <CLASS> [slot]
                      — set class for a row
    type <id> <C|T|M> — change type flag
    remove <id>       — remove a row
    mask <ra> <dec> <radius_arcsec>
                      — add a mask region
    save [base|mod]   — save to the base or _mod file (default: base)
    saveas <path>     — save to an arbitrary path
    validate          — re-run validation checks
    quit / exit / q   — exit without saving
    help              — show this list

    Parameters
    ----------
    headname : str
    outpath : str
    """
    result = explore_patch_table(headname, outpath, detail=True)

    # Prefer the _mod table if it exists, fall back to base
    if result["targets_mod"] is not None:
        tbl  = deepcopy(result["targets_mod"])
        path = result["targets_mod_path"]
        print(f"  Editing _mod table: {path}")
    elif result["targets"] is not None:
        tbl  = deepcopy(result["targets"])
        path = result["targets_path"]
        print(f"  Editing base table: {path}")
    else:
        print("  Nothing to edit — no patch files found.")
        return

    print()
    print("  Type 'help' for available commands.")
    print()

    while True:
        try:
            raw = input("  patch> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Interrupted — changes NOT saved.")
            break

        if not raw:
            continue

        parts = raw.split()
        cmd   = parts[0].lower()

        # ---- help ----
        if cmd == "help":
            print("""
  Commands:
    list
    inspect <id>
    z <id> <z> <zerr> [zwarn=0] [slot=0]
    class <id> <CLASS> [slot=0]
    type <id> <C|T|M>
    remove <id>
    mask <ra> <dec> <radius_arcsec>
    save [base|mod]
    saveas <full_path>
    validate
    quit / exit / q
""")

        # ---- list ----
        elif cmd == "list":
            _table_detail(tbl)

        # ---- inspect ----
        elif cmd == "inspect":
            if len(parts) < 2:
                print("  Usage: inspect <id>")
                continue
            try:
                inspect_row(tbl, int(parts[1]))
            except Exception as e:
                print(f"  Error: {e}")

        # ---- z ----
        elif cmd == "z":
            if len(parts) < 4:
                print("  Usage: z <id> <z> <zerr> [zwarn=0] [slot=0]")
                continue
            try:
                rid   = int(parts[1])
                z     = float(parts[2])
                zerr  = float(parts[3])
                zwarn = int(parts[4])   if len(parts) > 4 else 0
                slot  = int(parts[5])   if len(parts) > 5 else 0
                tbl   = set_redshift(tbl, rid, z, zerr, zwarn, slot)
            except Exception as e:
                print(f"  Error: {e}")

        # ---- class ----
        elif cmd == "class":
            if len(parts) < 3:
                print("  Usage: class <id> <CLASS> [slot=0]")
                continue
            try:
                rid      = int(parts[1])
                cls_str  = parts[2]
                slot     = int(parts[3]) if len(parts) > 3 else 0
                tbl      = set_class(tbl, rid, cls_str, slot)
            except Exception as e:
                print(f"  Error: {e}")

        # ---- type ----
        elif cmd == "type":
            if len(parts) < 3:
                print("  Usage: type <id> <C|T|M>")
                continue
            try:
                tbl = set_type(tbl, int(parts[1]), parts[2])
            except Exception as e:
                print(f"  Error: {e}")

        # ---- remove ----
        elif cmd == "remove":
            if len(parts) < 2:
                print("  Usage: remove <id>")
                continue
            try:
                confirm = input(
                    f"  Really remove row id={parts[1]}? [y/N] "
                ).strip().lower()
                if confirm == "y":
                    tbl = remove_row(tbl, int(parts[1]))
                else:
                    print("  Cancelled.")
            except Exception as e:
                print(f"  Error: {e}")

        # ---- mask ----
        elif cmd == "mask":
            if len(parts) < 4:
                print("  Usage: mask <ra> <dec> <radius_arcsec>")
                continue
            try:
                tbl = add_mask_region(
                    tbl,
                    float(parts[1]), float(parts[2]), float(parts[3]),
                )
            except Exception as e:
                print(f"  Error: {e}")

        # ---- save ----
        elif cmd == "save":
            target = parts[1].lower() if len(parts) > 1 else "base"
            if target == "mod":
                save_path = result["targets_mod_path"] or (
                    str(Path(result["targets_path"]).with_name(
                        Path(result["targets_path"]).stem
                        + "_mod"
                        + Path(result["targets_path"]).suffix
                    ))
                )
            else:
                save_path = result["targets_path"]
            try:
                save_patch_table(tbl, save_path)
            except Exception as e:
                print(f"  Error saving: {e}")

        # ---- saveas ----
        elif cmd == "saveas":
            if len(parts) < 2:
                print("  Usage: saveas <full_path>")
                continue
            try:
                save_patch_table(tbl, parts[1])
            except Exception as e:
                print(f"  Error saving: {e}")

        # ---- validate ----
        elif cmd == "validate":
            warnings = _validate_table(tbl)
            if warnings:
                print(f"  {len(warnings)} warning(s):")
                for w in warnings:
                    print(f"    ! {w}")
            else:
                print("  Validation: OK — no issues found.")

        # ---- quit ----
        elif cmd in ("quit", "exit", "q"):
            print("  Exiting editor — unsaved changes will be lost.")
            break

        else:
            print(f"  Unknown command: '{cmd}'.  Type 'help' for the list.")


"""
_patch_utils.py
===============
Shared patch-table utilities used by aps_ifu_prepare, aps_ifu_exgal,
and aps_ifu_gal.  Import from here to avoid duplication.
"""

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Row, Table

# Required columns for a valid patch table
PATCH_COLUMNS = [
    "id", "RA_icrs", "DEC_icrs", "A_world", "B_world",
    "angle", "flag", "type", "Z", "ZERR", "ZWARN", "CLASS",
]

# Class priorities for sorting
CLASS_PRIORITY = {"GALAXY": 0, "QSO": 1, "STAR": 2, "WD": 3}


# ------------------------------------------------------------------ #
# String / bytes normalisation                                        #
# ------------------------------------------------------------------ #

def decode_class(val) -> str:
    """Safely decode a CLASS value to a plain str."""
    if isinstance(val, (bytes, np.bytes_)):
        val = val.decode("utf-8", errors="replace")
    val = str(val).strip()
    if val.startswith("b'") and val.endswith("'"):
        val = val[2:-1]
    if val.upper() in ("NAN", "NONE", "B'NAN'", "B'NONE'", "B''", ""):
        return ""
    return val


def first_valid_class(row) -> str:
    """Return the first non-empty, decoded CLASS value for a row."""
    for v in np.atleast_1d(row["CLASS"]):
        c = decode_class(v).upper()
        if c:
            return c
    return ""


def first_valid_z(row):
    """Return (z, zerr, zwarn) from the first non-NaN slot, or (nan,nan,nan)."""
    z_arr    = np.atleast_1d(row["Z"]).astype(float)
    zerr_arr = np.atleast_1d(row["ZERR"]).astype(float)
    zwarn_arr= np.atleast_1d(row["ZWARN"]).astype(float)
    for i, z in enumerate(z_arr):
        if not np.isnan(z):
            return z, zerr_arr[i], zwarn_arr[i]
    return np.nan, np.nan, np.nan


# ------------------------------------------------------------------ #
# Table construction / normalisation                                  #
# ------------------------------------------------------------------ #

def ensure_table(patch_array, class_ntop: int = 1) -> Table:
    """
    Accept a single patch row in any of these forms and return a
    one-row astropy Table with the correct column structure:

      - dict  {col: scalar_or_list}
      - astropy Row  (tbl[i])
      - single-row astropy Table  (tbl[i:i+1])

    Parameters
    ----------
    patch_array : dict | Row | Table
    class_ntop : int
        Number of CLASS/Z/ZERR/ZWARN slots.  Inferred from the input
        when possible; falls back to this value.

    Returns
    -------
    Table  (one row, all PATCH_COLUMNS present)
    """
    # --- normalise to dict ---
    if isinstance(patch_array, Row):
        d = {c: patch_array[c] for c in patch_array.colnames}
    elif isinstance(patch_array, Table):
        assert len(patch_array) == 1, \
            "patch_array Table must have exactly 1 row"
        d = {c: patch_array[c][0] for c in patch_array.colnames}
    elif isinstance(patch_array, dict):
        d = dict(patch_array)
    else:
        raise TypeError(
            f"patch_array must be dict, astropy Row, or single-row Table, "
            f"got {type(patch_array)}"
        )

    # --- infer ntop from array columns if possible ---
    for key in ("CLASS", "Z", "ZERR", "ZWARN"):
        if key in d:
            v = d[key]
            if isinstance(v, (list, np.ndarray)):
                class_ntop = len(v)
                break

    # --- fill missing columns with sensible defaults ---
    d.setdefault("id",       1)
    d.setdefault("flag",     0)
    d.setdefault("type",     "T")
    d.setdefault("angle",    0.0)

    for key, default in [
        ("Z",     [np.nan]  * class_ntop),
        ("ZERR",  [np.nan]  * class_ntop),
        ("ZWARN", [np.nan]  * class_ntop),
        ("CLASS", [""]      * class_ntop),
    ]:
        if key not in d:
            d[key] = default
        else:
            v = d[key]
            if not isinstance(v, (list, np.ndarray)):
                # scalar → wrap in list
                d[key] = [v] + (
                    [np.nan if key != "CLASS" else ""]
                    * (class_ntop - 1)
                )

    # --- build one-row Table ---
    fixed = Table({
        "id":       [int(d["id"])],
        "RA_icrs":  [float(d["RA_icrs"])],
        "DEC_icrs": [float(d["DEC_icrs"])],
        "A_world":  [float(d["A_world"])],
        "B_world":  [float(d["B_world"])],
        "angle":    [float(d["angle"])],
        "flag":     [int(d["flag"])],
        "type":     [str(d["type"]).strip().upper()],
    })

    ntop = len(d["Z"])
    arr = Table({
        "Z":     np.array([d["Z"]],     dtype=float),
        "ZERR":  np.array([d["ZERR"]],  dtype=float),
        "ZWARN": np.array([d["ZWARN"]], dtype=float),
        "CLASS": np.array([
            [decode_class(c) for c in d["CLASS"]]
        ], dtype="U18"),
    })

    from astropy.table import hstack
    return hstack([fixed, arr])


# ------------------------------------------------------------------ #
# Loading from disk                                                   #
# ------------------------------------------------------------------ #

def load_patch_file(patch_file: str) -> Table:
    """Load a patch file (FITS or ASCII) and fill any masked values."""
    try:
        from PyAPS.aps_ifu_prepare import read_ascii_patchfile
    except ImportError:
        read_ascii_patchfile = None

    path = Path(patch_file)
    assert path.exists(), f"Patch file not found: {patch_file}"

    if path.suffix in (".fits", ".fit"):
        tbl = Table.read(str(path))
        for col in tbl.colnames:
            if hasattr(tbl[col], "mask"):
                tbl[col] = tbl[col].filled(np.nan)
    else:
        assert read_ascii_patchfile is not None, \
            "aps_ifu_prepare not available for ASCII patch files"
        tbl = read_ascii_patchfile(str(path))

    assert tbl.colnames == PATCH_COLUMNS, \
        f"Inconsistent patch_file columns.\n" \
        f"  Expected: {PATCH_COLUMNS}\n" \
        f"  Got     : {tbl.colnames}"
    return tbl


def load_and_split_patch_file(patch_file: str) -> tuple:
    """
    Load a patch file and apply multi-class splitting in memory.

    Returns
    -------
    (wp_table, ctarg_excluded)
    """
    try:
        from PyAPS.aps_ifu_prepare import (
            patch_table_needs_modification,
            update_patch_table,
        )
    except ImportError:
        raise ImportError(
            "aps_ifu_prepare is required for patch file loading"
        )

    tbl = load_patch_file(patch_file)

    ctarg_excluded = "C" not in [str(r["type"]).strip().upper()
                                  for r in tbl]

    needs_mod, reason = patch_table_needs_modification(tbl)
    if needs_mod:
        print(f"INFO: Multi-class split in memory: {reason}")
        tbl = update_patch_table(
            tbl,
            class_lists=[["GALAXY", "QSO"], ["STAR", "WD"]],
            debug=False,
            ctarg_excluded=ctarg_excluded,
            resort=True,
        )
        print(f"INFO: Split table → {len(tbl)} rows")

    return tbl, ctarg_excluded


# ------------------------------------------------------------------ #
# Health check                                                        #
# ------------------------------------------------------------------ #

def test_patch_table(
    wp_table: Table,
    mode: str = "ExGal",      # "ExGal" | "Gal" | "prepare"
    patch_array_mode: bool = False,
) -> bool:
    """
    Validate a patch table before entering the processing loop.

    Parameters
    ----------
    wp_table : Table
    mode : str
        ``'ExGal'`` checks for GALAXY/QSO rows.
        ``'Gal'``   checks for STAR/WD rows.
        ``'prepare'`` skips the class check.
    patch_array_mode : bool
        When True, relaxes the "at least one valid target" check
        (single injected row may not have a class yet).

    Returns
    -------
    bool
        ``True`` if all checks pass.

    Raises
    ------
    ValueError
        With a descriptive message on the first failing check.
    """
    errors = []

    # 1. Required columns
    missing = [c for c in PATCH_COLUMNS if c not in wp_table.colnames]
    if missing:
        errors.append(f"Missing columns: {missing}")

    # 2. At least one row
    if len(wp_table) == 0:
        errors.append("Patch table is empty")

    # 3. CLASS values are plain strings (not bytes)
    for row in wp_table:
        for v in np.atleast_1d(row["CLASS"]):
            raw = str(v)
            if raw.startswith("b'"):
                errors.append(
                    f"Row id={row['id']}: CLASS contains bytes "
                    f"({raw}) — run decode_class() before processing"
                )
                break

    # 4. A_world / B_world > 0 for non-mask rows
    for row in wp_table:
        if str(row["type"]).strip().upper() == "M":
            continue
        if float(row["A_world"]) <= 0 or float(row["B_world"]) <= 0:
            errors.append(
                f"Row id={row['id']}: zero or negative ellipse size "
                f"(A={row['A_world']}, B={row['B_world']})"
            )

    # 5. NaN redshift on non-mask targets (warning only in patch_array_mode)
    nan_rows = []
    for row in wp_table:
        if str(row["type"]).strip().upper() == "M":
            continue
        z, _, _ = first_valid_z(row)
        if np.isnan(z):
            nan_rows.append(int(row["id"]))
    if nan_rows and not patch_array_mode:
        errors.append(
            f"Rows with NaN redshift (non-mask): {nan_rows}. "
            f"Set class_patch=True or provide Z in patch_array."
        )

    # 6. At least one target of the right class (skip in patch_array_mode)
    if not patch_array_mode and mode != "prepare":
        target_classes = (
            {"GALAXY", "QSO"} if mode == "ExGal" else {"STAR", "WD"}
        )
        found = any(
            first_valid_class(row) in target_classes
            for row in wp_table
            if str(row["type"]).strip().upper() != "M"
        )
        if not found:
            errors.append(
                f"No valid {mode} targets found in patch table. "
                f"Expected CLASS in {target_classes}."
            )

    if errors:
        msg = "\n".join(f"  [{i+1}] {e}" for i, e in enumerate(errors))
        raise ValueError(
            f"Patch table health check FAILED ({len(errors)} error(s)):\n"
            + msg
        )

    print(f"  Patch table health check OK  "
          f"({len(wp_table)} rows, mode={mode})")
    return True



def adaptive_min_snr(snr_array, method="percentile", percentile=2,
                     absolute_floor=0.1, absolute_ceiling=1.5):
    """
    Derive a data-driven MIN_SNR threshold from the spaxel SNR distribution.

    PURPOSE
    -------
    Remove technically bad spaxels only:
        - Dead or hot fibres
        - Cosmic ray residuals
        - Negative flux artifacts from sky over-subtraction
        - IFU edge effects where the fibre throughput collapses

    This is NOT intended to remove scientifically faint spaxels —
    that is the job of the SB flux filter (apply_flux_filter=True).

    In particular, for high-z emission-line dominated targets where the
    continuum is undetected, the broadband SNR of real spaxels may be
    as low as 0.2–0.5.  The threshold must stay well below this to
    avoid removing real signal.

    The absolute_ceiling (default 1.5) ensures the adaptive value never
    becomes aggressive enough to cut low-continuum emission-line spaxels.
    The absolute_floor (default 0.1) ensures some minimal quality cut
    is always applied even when the SNR distribution is entirely low.

    METHODS
    -------
    'percentile' (recommended)
        Threshold = Nth percentile of the positive-SNR distribution.
        With percentile=2, removes only the bottom 2% — the obvious
        noise/bad-pixel tail.  Robust for all target types.

    'sigma_clip'
        Threshold = median - k*sigma of the clipped distribution.
        More aggressive for compact high-SNR sources.
        Less suitable for faint emission-line targets.

    'snr_gap'
        Finds the largest gap in the lower tail of the SNR distribution.
        Most physically motivated — identifies the natural separation
        between bad pixels (clustered near zero) and real signal.
        Best for targets with a clear bimodal SNR distribution.
        Falls back to 'percentile' on small samples.

    Parameters
    ----------
    snr_array : ndarray
        Per-spaxel SNR values (broadband mean signal / RMS noise).
    method : str
        'percentile', 'sigma_clip', or 'snr_gap'. Default: 'percentile'.
    percentile : float
        For 'percentile': bottom N% to cut. Default 2.
        Keep low (1–5) to avoid cutting real faint signal.
    absolute_floor : float
        Hard minimum threshold. Never cut above this value.
        Default 0.1 — anything with SNR > 0.1 may be real signal,
        especially for emission-line dominated targets.
    absolute_ceiling : float
        Hard maximum threshold. Never be more aggressive than this.
        Default 1.5 — protects faint continuum and emission-line spaxels.

    Returns
    -------
    float
        Adaptive MIN_SNR threshold, always in [absolute_floor, absolute_ceiling].

    Notes
    -----
    Typical values by target type:

    Bright cluster galaxy (SNR 5–50):
        5th pct ~ 1.5–3.0  →  ceiling clips to 1.5
    Normal field galaxy (SNR 1–10):
        2nd pct ~ 0.5–1.0  →  threshold ~ 0.5–1.0
    Faint emission-line galaxy (SNR 0.2–3):
        2nd pct ~ 0.1–0.3  →  floor clips to 0.1
    Dead fibres / bad pixels (SNR ≤ 0):
        Caught by floor=0.1 regardless of distribution
    """
    snr = np.asarray(snr_array, dtype=float)

    # Only consider finite positive SNR values for the distribution
    # Negative SNR = sky-subtraction artifact = always bad
    # SNR=0 = dead fibre = always bad
    snr_positive = snr[np.isfinite(snr) & (snr > 0)]

    if len(snr_positive) == 0:
        return float(absolute_floor)

    if method == "percentile":
        threshold = float(np.percentile(snr_positive, percentile))

    elif method == "sigma_clip":
        try:
            from astropy.stats import sigma_clipped_stats
            _, median, std = sigma_clipped_stats(snr_positive, sigma=3.0)
            threshold = float(median - 2.0 * std)
        except Exception:
            threshold = float(np.percentile(snr_positive, percentile))

    elif method == "snr_gap":
        sorted_snr = np.sort(snr_positive)
        median_snr = float(np.median(sorted_snr))
        # Look for gap only in the lower quarter — bad pixels cluster here
        lower = sorted_snr[sorted_snr < np.percentile(sorted_snr, 25)]
        if len(lower) < 10:
            # Not enough points in the lower tail — fall back to percentile
            threshold = float(np.percentile(snr_positive, percentile))
        else:
            gaps    = np.diff(lower)
            gap_idx = int(np.argmax(gaps))
            # Threshold just above the gap
            threshold = float(lower[gap_idx + 1])
    else:
        raise ValueError(
            "method must be 'percentile', 'sigma_clip', or 'snr_gap'")

    # Hard limits — these protect faint emission-line targets
    threshold = max(threshold, float(absolute_floor))
    threshold = min(threshold, float(absolute_ceiling))

    return threshold
# --------------------------------------------------------------------------- #
