"""
aps_MOSviewer.py - MOS/fibre-level L2 source viewer library (Dash figure/state layer)
========================================================================================

Library module (no standalone Dash app/CLI of its own — see
`aps_L2explorer.py`, the single interactive entry point for all L2
viewing) for the per-target results in a merged L2 `_APS.fits` file
produced by `aps_L2merge.py:mosL2merge` — the schema used for both true
MOS observations and IFU (LIFU/MIFU) data processed at individual-fibre
level (as opposed to `aps_IFUviewer.py`, which handles the Voronoi-
*patch* IFU schemas). One row per target, keyed by `APS_ID`, across up
to six extensions: `CLASS_TABLE`/`CLASS_SPEC` (Redrock, all ranks),
`STAR_TABLE`/`STAR_SPEC` (RVS+FERRE), `GALAXY_TABLE`/`GALAXY_SPEC`
(PPXF+EMIPPXF+line-strength) — all six always present in a `mosL2merge`
output. A single target routinely has *both* Galactic (RVS/FERRE) and
ExGal (PPXF/EMI) results at once, unlike the IFU-patch case, so the
explorer shows whichever apply side by side rather than choosing
between them.

No new plotting logic for Redrock/RVS/FERRE/PPXF/EMI — this module
reconstructs each of `apsPlot.redrock`, `apsPlot.rvs`, `apsPlot.ferre`,
`apsPlot.ppxf`, `apsPlot.emi`'s `build_figure()` inputs from the final
merged file's columns and calls those functions directly, exactly as
`aps_IFUviewer.py` reuses `apsPlot.spaxel_map`/`apsPlot.spectra`.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table

from dash import dcc, html, dash_table, no_update

from werkzeug.local import LocalProxy

from PyAPS.aps_utils import aps_file_info, get_column_unit
from PyAPS import aps_explorer_session as _sess
from PyAPS.apsPlot import redrock as _rr
from PyAPS.apsPlot import rvs as _rvs
from PyAPS.apsPlot import ferre as _ferre
from PyAPS.apsPlot import ppxf as _ppxf
from PyAPS.apsPlot import emi as _emi
from PyAPS.apsPlot import style
from PyAPS.apsPlot.datatable_export import csv_export_row

C_KMS = 299792.458

# Camera-letter -> full arm name, for nicer FERRE/RVS panel labels
# (STAR_SPEC/CLASS_SPEC columns are keyed by the single letter).
_ARM_NAMES = {"B": "BLUE", "G": "GREEN", "R": "RED"}


# --------------------------------------------------------------------------- #
# Coordinates — CNAME parsing (default, zero external files) and an optional
# L1 FIBTABLE join (checkbox-gated: locates the L1 file via the primary
# header's L1_REF_<arm> keys and joins on APS_ID == FIBREID).
# --------------------------------------------------------------------------- #

# WEAVE CNAME is a FIXED-WIDTH sexagesimal encoding with no literal
# decimal point: RA is HHMMSSss (seconds to hundredths, 8 digits), Dec is
# DDMMSSs (seconds to tenths, 7 digits) — e.g. 'WVE_08495699+1225333' is
# RA 08:49:56.99, Dec +12:25:33.3. An earlier version of this regex used
# an optional '\.?\d*' tail assuming a decimal point might be present;
# since it never is, that greedily over-consumed digits into the seconds
# field (verified wrong against a real L1 FIBTABLE cross-check — off by
# ~20 degrees on a real target). Widths must be matched exactly.
_CNAME_RE = re.compile(
    r"WVE_(\d{2})(\d{2})(\d{2})(\d{2})([+-])(\d{2})(\d{2})(\d{2})(\d)"
)


def _parse_cname_radec(cname):
    """WEAVE sexagesimal CNAME (e.g. 'WVE_08495699+1225333') -> (ra_deg, dec_deg).
    Accurate to ~0.1" — good enough for a click-to-select map, no external
    file dependency. Returns (nan, nan) if the CNAME doesn't parse."""
    m = _CNAME_RE.match(str(cname).strip())
    if not m:
        return np.nan, np.nan
    rh, rm, rs_i, rs_f, sign, dd, dm, ds_i, ds_f = m.groups()
    ra = 15.0 * (int(rh) + int(rm) / 60.0 + (int(rs_i) + int(rs_f) / 100.0) / 3600.0)
    dec = int(dd) + int(dm) / 60.0 + (int(ds_i) + int(ds_f) / 10.0) / 3600.0
    if sign == "-":
        dec = -dec
    return ra, dec


def _l1_dir_for_l2_file(aps_file):
    """Derive the L1 night-directory from the L2 file's own path by
    swapping the 'L2' path segment for 'L1' — the two share the same
    <night> subdirectory name in every case checked against real data."""
    parts = list(Path(aps_file).parts)
    for i, p in enumerate(parts):
        if p == "L2":
            parts[i] = "L1"
            return Path(*parts).parent
    return None


def _load_l1_fibtable_coords(aps_file, header):
    """Optional, checkbox-gated coordinate source: locate the L1 file(s)
    referenced in the primary header (L1_REF_B/G/R) and read their
    FIBTABLE extension, joining on APS_ID == FIBREID (verified exact
    match on real MOS and fibre-level-LIFU files). Returns
    {APS_ID: (ra, dec)} or {} if no L1 file could be found/read — callers
    fall back to CNAME parsing in that case.
    """
    l1_dir = _l1_dir_for_l2_file(aps_file)
    if l1_dir is None or not l1_dir.is_dir():
        return {}
    for arm in ("B", "G", "R"):
        ref = str(header.get(f"L1_REF_{arm}", "")).strip()
        if not ref:
            continue
        l1_path = l1_dir / ref
        if not l1_path.is_file():
            continue
        try:
            fib = fits.getdata(l1_path, extname="FIBTABLE")
        except Exception:
            continue
        return {int(fid): (float(ra), float(dec))
                for fid, ra, dec in zip(fib["FIBREID"], fib["TARGRA"], fib["TARGDEC"])}
    return {}


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def _load_mos_fits(dirprefix):
    """Load a `mosL2merge`-schema `_APS.fits` file. `CLASS_TABLE`,
    `STAR_TABLE`, `GALAXY_TABLE` (and their `*_SPEC` counterparts) each
    carry their own `APS_ID` column with *different row counts and
    order* — always look up by APS_ID via the `*_pos` maps below, never
    assume row position lines up across extensions.
    """
    aps_file = dirprefix + "_APS.fits"
    if not Path(aps_file).is_file():
        raise FileNotFoundError(f"No such file: {aps_file}")

    info = aps_file_info(aps_file)
    if info["schema"] != "mos":
        raise ValueError(
            f"{aps_file} is not a MOS-schema file (schema={info['schema']!r}, "
            f"no CLASS_TABLE extension) — use aps_IFUviewer.py for IFU-patch products."
        )

    st = dict(aps_file=aps_file, file_info=info)

    # CLASS_TABLE must be re-masked: Table.read() alone returns a plain
    # Column, and apsPlot.redrock._rank_title's `.value.data` access is
    # written for a MaskedColumn — without this every rank panel's title
    # silently degrades to a bare "Rank N" (no crash, just lost detail).
    st["class_table"] = Table(Table.read(aps_file, hdu="CLASS_TABLE"), masked=True)
    st["class_spec"] = Table.read(aps_file, hdu="CLASS_SPEC")
    st["class_pos"] = {int(a): i for i, a in enumerate(st["class_table"]["APS_ID"])}
    st["class_spec_pos"] = {int(a): i for i, a in enumerate(st["class_spec"]["APS_ID"])}
    st["arms"] = [c.split("_")[-1] for c in st["class_spec"].colnames if c.startswith("LAMBDA_RR_")]

    hdul = fits.open(aps_file)
    extnames = [h.name for h in hdul]

    st["have_star"] = "STAR_TABLE" in extnames
    st["have_star_spec"] = "STAR_SPEC" in extnames
    if st["have_star"]:
        star_table = fits.getdata(aps_file, extname="STAR_TABLE")
        st["star_table"] = star_table
        st["star_pos"] = {int(a): i for i, a in enumerate(star_table.APS_ID)}
        st["rvs_available"] = "VRAD" in star_table.columns.names
        st["ferre_available"] = "TEFF" in star_table.columns.names
    else:
        st["rvs_available"] = st["ferre_available"] = False
    if st["have_star_spec"]:
        star_spec = fits.getdata(aps_file, extname="STAR_SPEC")
        st["star_spec"] = star_spec
        st["star_spec_pos"] = {int(a): i for i, a in enumerate(star_spec.APS_ID)}

    st["have_galaxy"] = "GALAXY_TABLE" in extnames
    st["have_galaxy_spec"] = "GALAXY_SPEC" in extnames
    if st["have_galaxy"]:
        galaxy_table = fits.getdata(aps_file, extname="GALAXY_TABLE")
        st["galaxy_table"] = galaxy_table
        st["galaxy_pos"] = {int(a): i for i, a in enumerate(galaxy_table.APS_ID)}
        st["ppxf_available"] = "V" in galaxy_table.columns.names
        # 62 real emission lines, derived from FLUX_<name> columns
        # (excluding the FLUX_CLEAN_EMI *_SPEC array column, which has a
        # different naming shape and lives in GALAXY_SPEC, not here).
        st["line_names"] = [c[5:] for c in galaxy_table.columns.names
                             if c.startswith("FLUX_") and not c.startswith("FLUX_CLEAN")]
    else:
        st["ppxf_available"] = False
        st["line_names"] = []
    if st["have_galaxy_spec"]:
        galaxy_spec = fits.getdata(aps_file, extname="GALAXY_SPEC")
        st["galaxy_spec"] = galaxy_spec

    # Coordinates: CNAME-parsed always computed first (zero extra files,
    # the guaranteed fallback), then the L1-FIBTABLE join is attempted
    # automatically right here at load time and used whenever it's
    # actually available — explicit request ("you can [just] use the L1
    # fibtable coordinate if available") to drop the manual toggle this
    # used to need (see the removed on_coord_source_change/
    # coord-source-toggle) in favour of just doing the better thing by
    # default. Silently stays on CNAME coordinates if no L1 file is
    # found, exactly like the removed toggle's own "if available" case.
    cnames = st["class_table"]["CNAME"]
    radec = np.array([_parse_cname_radec(c) for c in cnames])
    st["ra_cname"] = radec[:, 0]
    st["dec_cname"] = radec[:, 1]
    st["ra"] = st["ra_cname"]
    st["dec"] = st["dec_cname"]
    st["coord_source"] = "cname"

    header = fits.getheader(aps_file, 0)
    l1_coords = _load_l1_fibtable_coords(aps_file, header)
    if l1_coords:
        aps_ids = np.asarray(st["class_table"]["APS_ID"])
        ra = np.full(len(aps_ids), np.nan)
        dec = np.full(len(aps_ids), np.nan)
        for i, aid in enumerate(aps_ids):
            coord = l1_coords.get(int(aid))
            if coord is not None:
                ra[i], dec[i] = coord
        st["ra"], st["dec"], st["coord_source"] = ra, dec, "l1"

    return st


def _target_availability(st, aps_id):
    """Per-target (not per-dataset) capability flags — a single target
    routinely has both Galactic and ExGal results at once."""
    avail = dict(rvs=False, ferre=False, ppxf=False, emi=False)
    spos = st.get("star_pos", {}).get(aps_id)
    if spos is not None:
        rec = st["star_table"]
        if st["rvs_available"] and np.isfinite(rec.VRAD[spos]):
            avail["rvs"] = True
        if st["ferre_available"] and np.isfinite(rec.TEFF[spos]):
            avail["ferre"] = True
    gpos = st.get("galaxy_pos", {}).get(aps_id)
    if gpos is not None:
        rec = st["galaxy_table"]
        if st["ppxf_available"] and np.isfinite(rec.V[gpos]):
            avail["ppxf"] = True
        if st["have_galaxy_spec"]:
            gspos = st["galaxy_pos"].get(aps_id)  # GALAXY_SPEC aligned with GALAXY_TABLE
            if gspos is not None and gspos < len(st["galaxy_spec"]):
                emis = st["galaxy_spec"].EMISSION_EMI[gspos]
                if np.any(np.isfinite(emis)) and np.nanmax(np.abs(emis)) > 0:
                    avail["emi"] = True
    return avail


# --------------------------------------------------------------------------- #
# Source (coordinate) map
# --------------------------------------------------------------------------- #

_AVAIL_STYLE = {
    "both": ("Galactic + ExGal", "#2ca02c"),
    "gal": ("Galactic only", "#1f77b4"),
    "exgal": ("ExGal only", "#d62728"),
    "none": ("Neither", "#7f7f7f"),
}

# Continuous alternatives to the default categorical "availability" colour
# scheme for the Aladin catalog overlay — per explicit user request for
# "more options" than colour-by-category alone. Each entry:
# (dropdown label, CLASS_TABLE column, colorscale, legend label).
# Z/SNR are both stored per-Redrock-rank (an array per target, see
# _CLASS_RANK_COLS) — rank 0 (the best fit) is what's shown.
COLOR_BY_OPTIONS = {
    # Label renamed per explicit request ("Availability" -> "Class
    # type") -- the "availability" dict *key* is unchanged (still
    # referenced by STATE.color_by/_color_by_values/etc.), only the
    # dropdown-facing label moved.
    "availability": ("Class type (Gal, ExGal, Both, None)", None, None, None),
    "redshift": ("Redshift (best rank)", "Z", "Viridis", "Z"),
    "snr": ("S/N (best rank)", "SNR", "Plasma", "S/N"),
}


def _color_by_values(st, column):
    """Best-rank (rank 0) value of a per-rank CLASS_TABLE column, for
    every target in APS_ID order — NaN where the column or the target
    itself isn't available, matching _aladin_extra_fields' handling of
    the exact same per-rank-array storage convention.

    Takes the `AppState` itself (`st`, e.g. `mos_mod.STATE`), not the raw
    `st.data` dict — matches `aps_l1_preview._color_by_values`'s own
    signature shape, needed here for the cache below. Cached per
    `column`, invalidated only on a fresh `load()` (`st._data_generation`
    — see that field's own comment for the real concurrent-access bug
    this exact pattern fixes) — added for the multi-panel L2 map
    feature, where several panels can easily end up showing the same
    column at once; this was previously recomputed (a plain Python loop
    over every target) on every single call with no caching at all."""
    cache_key = st._data_generation
    if st._color_by_cache_key != cache_key:
        st._color_by_cache = {}
        st._color_by_cache_key = cache_key
    if column in st._color_by_cache:
        return st._color_by_cache[column]

    ct = st.data["class_table"]
    aps_ids = np.asarray(ct["APS_ID"])
    if column not in ct.colnames:
        out = np.full(len(aps_ids), np.nan)
    else:
        out = np.full(len(aps_ids), np.nan)
        for i in range(len(aps_ids)):
            val = ct[column][i]
            arr = np.asarray(val)
            try:
                out[i] = float(arr[0]) if arr.ndim >= 1 else float(arr)
            except (TypeError, ValueError):
                pass
    st._color_by_cache[column] = out
    return out


def _build_source_map_figure(selected_aps_id=None):
    import plotly.graph_objects as go
    from PyAPS.apsPlot import style

    st = STATE.data
    ra, dec = st["ra"], st["dec"]
    aps_ids = np.asarray(st["class_table"]["APS_ID"])
    valid = np.isfinite(ra) & np.isfinite(dec)

    cos_dec = np.cos(np.radians(np.nanmean(dec[valid]))) if valid.any() else 1.0

    category = np.full(len(aps_ids), "none", dtype=object)
    for i, aid in enumerate(aps_ids):
        av = _target_availability(st, int(aid))
        has_gal = av["rvs"] or av["ferre"]
        has_exgal = av["ppxf"] or av["emi"]
        category[i] = "both" if (has_gal and has_exgal) else "gal" if has_gal else "exgal" if has_exgal else "none"

    fig = go.Figure()
    for key, (label, color) in _AVAIL_STYLE.items():
        mask = valid & (category == key)
        if not mask.any():
            continue
        fig.add_trace(go.Scatter(
            x=ra[mask], y=dec[mask], mode="markers",
            marker=dict(size=7, color=color, line=dict(width=0.5, color="rgba(0,0,0,0.3)")),
            customdata=aps_ids[mask].tolist(), name=label,
            hovertemplate="APS_ID=%{customdata}<br>RA=%{x:.5f}°  Dec=%{y:.5f}°<extra></extra>",
        ))

    if selected_aps_id is not None:
        sel = np.flatnonzero(aps_ids == selected_aps_id)
        if sel.size and valid[sel[0]]:
            i = sel[0]
            fig.add_trace(go.Scatter(
                x=[ra[i]], y=[dec[i]], mode="markers",
                marker=dict(symbol="circle-open", size=16, color="black", line=dict(width=2.5)),
                name="Selected", hoverinfo="skip", showlegend=False,
            ))

    # Static reversed range, not autorange="reversed" — see
    # apsPlot/fiber_map.py's fiber_map_figure for why: autorange staying
    # permanently "on" makes Plotly recompute the range from scratch (and
    # discard any uirevision-preserved user zoom/pan) on every rebuild,
    # e.g. every time a different target is clicked elsewhere in the app.
    if valid.any():
        ra_min, ra_max = float(np.nanmin(ra[valid])), float(np.nanmax(ra[valid]))
    else:
        ra_min, ra_max = 0.0, 1.0
    ra_pad = (ra_max - ra_min) * 0.05 or 0.001

    base_layout = {k: v for k, v in style.BASE_LAYOUT.items() if k not in ("hovermode", "margin")}
    fig.update_layout(
        **base_layout,
        hovermode="closest",
        title="Sources (from CNAME)" if st["coord_source"] == "cname" else "Sources (L1 FIBTABLE)",
        width=650, height=600,
        xaxis=dict(title="RA (deg)", range=[ra_max + ra_pad, ra_min - ra_pad]),
        # cos(dec) correction so equal sky-angle spans look equal on
        # screen (same reasoning as apsPlot/fiber_map.py's fiber map) —
        # applied via scaleratio rather than mutating RA, so RA/Dec in
        # the hover template stay real sky coordinates.
        yaxis=dict(title="Dec (deg)", scaleanchor="x", scaleratio=1.0 / cos_dec if cos_dec else 1.0),
        margin=dict(l=60, r=30, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
    )
    return fig


# --------------------------------------------------------------------------- #
# Figure builders — reconstruct each apsPlot module's inputs from the
# merged file for one target, then call its *existing* build_figure().
# --------------------------------------------------------------------------- #

def _rank0_z(zbest, id_zbest):
    z = zbest["Z"][id_zbest]
    return float(np.ravel(z)[0])


def _redrock_figure(aps_id):
    st = STATE.data
    id_zbest = st["class_pos"][aps_id]
    i = st["class_spec_pos"][aps_id]
    return _rr.build_figure(st["class_table"], st["class_spec"], st["arms"], i, id_zbest)


def _rvs_figure(aps_id):
    st = STATE.data
    spos = st["star_pos"][aps_id]
    sspos = st["star_spec_pos"][aps_id]
    rec, spec = st["star_table"], st["star_spec"]

    specdata, yfit = [], []
    for arm in st["arms"]:
        col_lam, col_flux, col_mod = f"LAMBDA_RVS_{arm}", f"FLUX_RVS_{arm}", f"MODEL_RVS_{arm}"
        if col_lam not in spec.columns.names:
            continue
        specdata.append(_SpecData(lam=np.asarray(spec[col_lam][sspos]), spec=np.asarray(spec[col_flux][sspos])))
        yfit.append(np.asarray(spec[col_mod][sspos]))

    title = (f"FEH={rec.FEH_RVS[spos]:.3f}  TEFF={rec.TEFF_RVS[spos]:.1f}  "
             f"LOGG={rec.LOGG_RVS[spos]:.2f}  ALPHA={rec.ALPHA_RVS[spos]:.2f}  "
             f"Vrad={rec.VRAD[spos]:.2f}+/-{rec.VRAD_ERR[spos]:.2f}")
    return _rvs.build_figure(specdata, yfit, title, flux_unit="Flux")


class _SpecData:
    """Minimal duck-typed stand-in for rvspecfit.spec_fit.SpecData —
    apsPlot.rvs.build_figure only reads .lam/.spec."""
    def __init__(self, lam, spec):
        self.lam, self.spec = lam, spec


def _ferre_figure(aps_id):
    st = STATE.data
    spos = st["star_pos"][aps_id]
    sspos = st["star_spec_pos"][aps_id]
    rec, spec = st["star_table"], st["star_spec"]

    # ferre.build_figure keys columns by setup[0] (first char of each
    # `setups` entry) — passing the full arm name ("BLUE"/"RED") both
    # keys the right LAMBDA_FR_B/LAMBDA_FR_R columns (since "BLUE"[0]=="B")
    # and gives a nicer panel label than the bare letter.
    outdict = {}
    setups = []
    for arm in st["arms"]:
        if f"LAMBDA_FR_{arm}" not in spec.columns.names:
            continue
        outdict[f"LAMBDA_FR_{arm}"] = [np.asarray(spec[f"LAMBDA_FR_{arm}"][sspos])]
        outdict[f"FLUX_FR_{arm}"] = [np.asarray(spec[f"FLUX_FR_{arm}"][sspos])]
        outdict[f"MODEL_FR_{arm}"] = [np.asarray(spec[f"MODEL_FR_{arm}"][sspos])]
        setups.append(_ARM_NAMES.get(arm, arm))
    for key in ("FEH", "TEFF", "LOGG", "ALPHA", "MICRO"):
        outdict[key] = [rec[key][spos]]

    return _ferre.build_figure(outdict, 0, setups, flux_unit="Normalized Flux")


def _ppxf_figure(aps_id):
    st = STATE.data
    gpos = st["galaxy_pos"][aps_id]
    rec, spec = st["galaxy_table"], st["galaxy_spec"]

    ppxf_result = np.array([[rec.V[gpos], rec.SIGMA[gpos], rec.H3[gpos], rec.H4[gpos]]])
    logLam = np.asarray(spec.LOGLAM_PPXF[gpos])[None, :]
    bin_spec = np.asarray(spec.FLUX_PPXF[gpos])[None, :]
    bestfit = np.asarray(spec.MODEL_PPXF[gpos])[None, :]
    goodpix = np.asarray(spec.GOODPIX_PPXF[gpos])[None, :]
    return _ppxf.build_figure(ppxf_result, logLam, bin_spec, bestfit, goodpix, 0, flux_unit="Flux")


def _emi_figure(aps_id):
    st = STATE.data
    gpos = st["galaxy_pos"][aps_id]
    id_zbest = st["class_pos"][aps_id]
    gt, gs, zbest = st["galaxy_table"], st["galaxy_spec"], st["class_table"]

    z_in = _rank0_z(zbest, id_zbest)
    names = st["line_names"]
    fluxes = np.array([gt[f"FLUX_{n}"][gpos] for n in names])
    with np.errstate(invalid="ignore"):
        zl = np.array([gt[f"Z_{n}"][gpos] if f"Z_{n}" in gt.columns.names else np.nan for n in names])
        sl = np.array([gt[f"SIGMA_{n}"][gpos] if f"SIGMA_{n}" in gt.columns.names else np.nan for n in names])
        vl = C_KMS * ((zl + 1.0) / (z_in + 1.0) - 1.0)

    meta = [{"APS_ID": int(aps_id), "TARGID": str(gt.TARGID[gpos]).strip(),
             "CNAME": str(gt.CNAME[gpos]).strip(), "Z": z_in,
             "ZERR": float(np.ravel(zbest["ZERR"][id_zbest])[0])}]
    with warnings.catch_warnings():
        # nanmedian on a target with zero detected lines is legitimately
        # all-NaN — emi.build_figure only uses this as a fallback when
        # mc_results (passed below, with the real per-line values) is
        # absent, so the result is unused in the common case anyway.
        warnings.simplefilter("ignore", RuntimeWarning)
        emi_result = [{"emi": np.array([np.nanmedian(vl), np.nanmedian(sl)]),
                       "stellar": np.array([gt.V[gpos], gt.SIGMA[gpos]])}]
    mc_results = [{"line_velocities": vl, "line_sigmas": sl}]

    logLam = np.asarray(gs.LOGLAM_EMI[gpos])[None, :]
    bin_data = np.asarray(gs.FLUX_EMI[gpos])[None, :]
    emi_bestfit = np.asarray(gs.EMISSION_EMI[gpos])[None, :]      # pure emission model, NOT MODEL_EMI
    stellar_bestfit = np.asarray(gs.MODEL_CLEAN_EMI[gpos])[None, :]
    goodpix = np.asarray(gs.GOODPIX_EMI[gpos])[None, :]

    return _emi.build_figure(
        0, meta, emi_result, [fluxes], [names],
        logLam, bin_data, emi_bestfit,
        stellar_bestfit=stellar_bestfit, goodpixels_array=goodpix,
        mc_results=mc_results, flux_unit="Flux",
    )


# --------------------------------------------------------------------------- #
# Value tables
# --------------------------------------------------------------------------- #

_ARRAY_PREVIEW_N = 4


def _json_safe(v):
    if isinstance(v, (bytes, np.bytes_)):
        return v.decode(errors="replace").strip()
    if isinstance(v, np.ndarray):
        # Value-table cells are one line of text — a full COEFF (10
        # numbers per rank) or a scan-curve array reads as an unreadable
        # wall of digits, so show a short preview + how many were
        # dropped rather than every element.
        flat = v.ravel()
        items = [_json_safe(x) for x in flat[:_ARRAY_PREVIEW_N].tolist()]
        preview = ", ".join(str(x) for x in items)
        if flat.size > _ARRAY_PREVIEW_N:
            preview += f", … (+{flat.size - _ARRAY_PREVIEW_N} more)"
        return f"[{preview}]"
    if isinstance(v, (np.floating, float)):
        v = float(v)
        return None if np.isnan(v) else round(v, 6)
    if isinstance(v, (np.integer, int)):
        return int(v)
    return str(v)


_CLASS_RANK_COLS = {"Z", "ZERR", "ZWARN", "CLASS", "SUBCLASS", "TARGSRVY", "TARGCLASS",
                    "TARGPROG", "SNR", "CHI2", "DELTACHI2", "NCOEFF", "NPIXELS", "SRVY_CLASS"}


def _class_table_rows(aps_id):
    """One row per Redrock rank, rank-varying columns unpacked per rank —
    CLASS_TABLE stores rank as an array axis, not extra rows. Only the
    columns Redrock/aps_rr actually vary by rank (per-rank Z/CLASS/SNR/
    etc., confirmed against real data) are unpacked that way; everything
    else (APS_ID/TARGID/CNAME, and the CZZ_*/CZZ_CHI2_* per-template
    chi2-vs-z scan curves — 1000+ elements each, NOT rank-indexed) is
    shown once, un-exploded, on the rank-0 row only, since guessing
    "rank-varying" from array shape alone misfires on those scan curves
    (long enough that shape[0] > rank by coincidence).
    """
    st = STATE.data
    pos = st["class_pos"].get(aps_id)
    if pos is None:
        return []
    zb = st["class_table"]
    n_ranks = _rr._n_ranks(zb)
    scalar_cols = [c for c in zb.colnames if c not in _CLASS_RANK_COLS and not c.startswith("CZZ_")]
    rows = []
    for rk in range(max(n_ranks, 1)):
        row = {"Rank": rk}
        for col in _CLASS_RANK_COLS:
            if col not in zb.colnames:
                continue
            val = zb[col][pos]
            arr = np.asarray(val)
            row[col] = _json_safe(arr[rk] if arr.ndim >= 1 and arr.shape[0] > rk else val)
        if rk == 0:
            for col in scalar_cols:
                row[col] = _json_safe(zb[col][pos])
        rows.append(row)
    return rows


def _star_table_rows(aps_id):
    st = STATE.data
    if not st["have_star"]:
        return []
    pos = st["star_pos"].get(aps_id)
    if pos is None:
        return []
    rec = st["star_table"]
    return [{"Parameter": c, "Unit": get_column_unit(rec, c) or "", "Value": _json_safe(rec[c][pos])}
            for c in rec.columns.names]


def _galaxy_table_rows(aps_id):
    st = STATE.data
    if not st["have_galaxy"]:
        return []
    pos = st["galaxy_pos"].get(aps_id)
    if pos is None:
        return []
    rec = st["galaxy_table"]
    return [{"Parameter": c, "Unit": get_column_unit(rec, c) or "", "Value": _json_safe(rec[c][pos])}
            for c in rec.columns.names]


_VALUE_TABLE_CELL_CONDITIONAL = [
    {"if": {"column_id": "Parameter"}, "textAlign": "left", "fontWeight": "600"},
    {"if": {"column_id": "Unit"}, "color": "var(--pyaps-ink-faint)", "width": "70px"},
]


def _value_tables_panel():
    return html.Div([
        html.H4("Source Data", style={"marginBottom": "4px"}),
        html.Div([
            html.Div([
                html.H5("Redrock (CLASS_TABLE) — all ranks"),
                csv_export_row("class-values-table", dash_table.DataTable(
                    id="class-values-table",
                    sort_action="native", filter_action="native", page_size=5,
                    **{**style.DATATABLE_KWARGS,
                       "style_cell": {**style.DATATABLE_STYLE_CELL, "fontFamily": "monospace", "fontSize": 11}},
                )),
            ]),
            html.Div([
                html.Div([
                    html.H5("Stellar Results (STAR_TABLE)"),
                    csv_export_row("star-values-table", dash_table.DataTable(
                        id="star-values-table",
                        columns=[{"name": "Parameter", "id": "Parameter"},
                                 {"name": "Unit", "id": "Unit"},
                                 {"name": "Value", "id": "Value"}],
                        sort_action="native", filter_action="native", page_size=12,
                        **{**style.DATATABLE_KWARGS,
                           "style_cell": {**style.DATATABLE_STYLE_CELL, "fontFamily": "monospace"},
                           "style_cell_conditional": _VALUE_TABLE_CELL_CONDITIONAL},
                    )),
                ], style={"width": "48%", "display": "inline-block", "verticalAlign": "top"}),
                html.Div([
                    html.H5("Galaxy Results (GALAXY_TABLE, incl. Line Strength)"),
                    csv_export_row("galaxy-values-table", dash_table.DataTable(
                        id="galaxy-values-table",
                        columns=[{"name": "Parameter", "id": "Parameter"},
                                 {"name": "Unit", "id": "Unit"},
                                 {"name": "Value", "id": "Value"}],
                        sort_action="native", filter_action="native", page_size=12,
                        **{**style.DATATABLE_KWARGS,
                           "style_cell": {**style.DATATABLE_STYLE_CELL, "fontFamily": "monospace"},
                           "style_cell_conditional": _VALUE_TABLE_CELL_CONDITIONAL},
                    )),
                ], style={"width": "48%", "display": "inline-block", "verticalAlign": "top", "marginLeft": "4%"}),
            ], style={"marginTop": "16px"}),
        ]),
    ], style={"marginTop": "24px", "borderTop": "1px solid #ddd", "paddingTop": "12px"})


# --------------------------------------------------------------------------- #
# Layout pieces and callback logic — library functions only. This module no
# longer owns a Dash `app`/layout/CLI entry point of its own: PyAPS.aps_L2explorer
# is the single Dash app for all L2 viewing now, and imports this module
# purely for its state (AppState/STATE), loader (_load_mos_fits), figure
# builders, and the layout/callback functions kept below (still plain,
# undecorated functions — reused by the explorer's own callbacks).
# --------------------------------------------------------------------------- #


class AppState:
    def __init__(self):
        self.data = None
        self.selected_aps_id = None
        # What the Aladin catalog overlay is coloured by — "availability"
        # (the original, categorical Gal/ExGal/Both/None view) or a
        # continuous quantity (see _COLOR_BY_OPTIONS). color_vmin/vmax only
        # apply to the continuous options; ignored for "availability".
        self.color_by = "availability"
        self.color_vmin = None
        self.color_vmax = None
        self.color_scale = "linear"  # see aps_explorer.COLOR_SCALE_OPTIONS
        self._color_by_cache = {}       # see _color_by_values()
        self._color_by_cache_key = None  # self._data_generation it was computed for
        # Bumped once, as the very *last* thing load() does — same
        # proven-safe pattern (and same real bug it fixes) as
        # aps_l1_preview.py's own AppState._data_generation; see that
        # field's own comment for the full "why" (a concurrent reader
        # landing mid-load must never see a cache keyed off data that
        # hasn't actually finished updating yet).
        self._data_generation = 0

    def loaded(self):
        return self.data is not None

    def load(self, outpath, headname):
        dirprefix = str(Path(outpath) / headname)
        self.data = _load_mos_fits(dirprefix)
        self.selected_aps_id = int(self.data["class_table"]["APS_ID"][0])
        self.color_by = "availability"
        self.color_vmin = None
        self.color_vmax = None
        self.color_scale = "linear"
        self._data_generation += 1


# A LocalProxy, not a bare instance — see aps_l1_preview.py's own
# STATE = LocalProxy(...) for the full rationale (identical pattern,
# aps_explorer_session.py's own module docstring for the full design).
STATE = LocalProxy(lambda: _sess.current_bundle().get_or_create("mos", AppState))


def _load_form():
    return html.Div([
        html.H3("Load dataset"),
        html.Label("Output/results directory (--outpath)"),
        dcc.Input(id="lf-outpath", type="text", style={"width": "100%"}),
        html.Label("Head name (--headname)"),
        dcc.Input(id="lf-headname", type="text", style={"width": "100%"}),
        html.Button("Load", id="lf-run", n_clicks=0, style={"marginTop": "10px"}),
        html.Div(id="lf-status", style={"color": "crimson", "marginTop": "6px"}),
    ], style={"padding": "20px", "maxWidth": "600px"})


def _file_info_panel():
    st = STATE.data
    if st is None:
        return None
    fi = st["file_info"]
    return html.Div(
        f"schema={fi['schema']}  obsmode={fi['obsmode']}  resolution={fi['resolution']}  "
        f"arms={','.join(fi['arms'])}  PyAPS v{fi['pyaps_version']}  CPS v{fi['cps_version']}  "
        f"targets={len(st['class_table'])}",
        style={"fontSize": "12px", "color": "var(--pyaps-ink-muted)", "marginBottom": "8px"},
    )


def _available_tabs(st, aps_id):
    """Per-TARGET (not per-dataset) tab list — Redrock always applies;
    Stellar/Galaxy tabs vary target to target, and a target commonly has
    both at once."""
    def _tab(label, value):
        return dcc.Tab(label=label, value=value, style=style.TAB_STYLE,
                        selected_style=style.TAB_SELECTED_STYLE)

    tabs = [_tab("Redrock", "redrock")]
    if aps_id is not None:
        av = _target_availability(st, aps_id)
        if av["rvs"]:
            tabs.append(_tab("Stellar (RVS)", "rvs"))
        if av["ferre"]:
            tabs.append(_tab("Stellar (FERRE)", "ferre"))
        if av["ppxf"]:
            tabs.append(_tab("Galaxy (PPXF)", "ppxf"))
        if av["emi"]:
            tabs.append(_tab("Galaxy (EMI)", "emi"))
    return tabs


def _settings_row():
    """"Go to APS_ID" entry and the Aladin "colour by" choice — kept as
    its own function (mirroring aps_IFUviewer._settings_panel) so it
    can be swapped into a shared "settings panel" container per
    dataset kind. The L1-FIBTABLE-coordinates toggle that used to live
    here is gone — explicit request to drop it in favour of always
    using L1 coordinates automatically when available (see
    _load_mos_fits) rather than requiring an opt-in click.

    Real flexbox row (alignItems: "flex-end"), matching
    aps_IFUviewer._value_tables_panel's own identical fix — explicit
    request ("the text box for go to APS_ID is really large and not
    aligned"/"make sure all these select menus are aligned together").
    display:inline-block + verticalAlign:top (the previous approach)
    aligned each group's own *top* edge, so a taller dcc.Dropdown and a
    plain dcc.Input never lined up at the control itself, only at the
    label above it — flex-end aligns every group's own *bottom* edge
    instead."""
    return html.Div([
        html.Div([
            html.Label("Go to APS_ID:", style={"display": "block", "marginBottom": "4px"}),
            html.Div([
                dcc.Input(id="target-id-input", type="number",
                          style={"width": "100px", **style.SOFT_INPUT_HEIGHT_STYLE}),
                html.Button("Go", id="target-id-go", n_clicks=0,
                            style={"marginLeft": "6px", **style.SOFT_BUTTON_STYLE}),
            ]),
        ], style={"marginRight": "24px"}),
        # Wrapped in its own id (shared with aps_l1_preview.py's version
        # of this same control) so aps_explorer.update_map_mode can hide
        # it specifically in 3D mode — see that id's own comment there
        # for the explicit request/reasoning.
        html.Div([
            html.Label("Colour Aladin points by:", style={"display": "block", "marginBottom": "4px"}),
            dcc.Dropdown(
                id="mos-color-by",
                options=[{"label": label, "value": key} for key, (label, *_) in COLOR_BY_OPTIONS.items()],
                value=STATE.color_by, clearable=False, style={"width": "260px", "fontSize": "12px"},
            ),
        ], id="aladin-color-by-row"),
    ], style={"padding": "10px 0", "display": "flex", "alignItems": "flex-end", "flexWrap": "wrap"})


def on_map_click(click_data):
    """Resolve a source-map click to its APS_ID — unlike the IFU map, the
    MOS/fibre-level map's customdata's own [0] entry already *is* the
    APS_ID (one point per target, not per spaxel-within-a-bin), so no
    extra lookup needed. customdata itself is `[index, ra, dec]` (see
    apsPlot.source_map.source_map_figure's own docstring for why ra/dec
    ride along too — the multi-panel L2 map feature's own cross-panel
    selection sync needs them client-side)."""
    if not click_data or not STATE.loaded():
        return no_update
    customdata = click_data["points"][0].get("customdata")
    if customdata is None:
        return no_update
    aps_id = customdata[0] if isinstance(customdata, (list, tuple)) else customdata
    return int(aps_id)


_TAB_VIEWPORT_HEIGHT = "650px"
_GRAPH_CONFIG = style.GRAPH_CONFIG


def _scrollable_graph(fig):
    """Fixed-height, scrollable wrapper for a tab's figure. Several of
    these (Redrock especially — up to 3 ranks x 2 arms x flux+IVAR = up
    to 12 panel-rows) compute a natural height taller than any fixed
    dcc.Graph size; without this the figure would overflow past a
    fixed-height Graph div into (and behind) the value tables below it.
    Letting the Graph render at its own figure.layout.height and constraining
    the *wrapper* instead means every panel stays fully visible via scroll,
    and short figures
    (e.g. PPXF's single panel) just don't fill the scroll area.

    `fill_container_width` clears the fixed pixel width these figures
    default to for their other life as static PNG pipeline-diagnostic
    exports (see its own docstring) — this is this app's own live view,
    so it should fill whatever width its tab actually has instead of
    leaving empty space beside a fixed-width plot.

    A real, confirmed Dash bug this docstring's own previous claim ("the
    Graph render[s] at its own figure.layout.height") turned out NOT to
    actually hold, found by reading `dcc.Graph`'s own async-graph.js
    bundle directly (not assumed) after a live report: "loading takes
    too long... unusually unstable... only for Redrock, not RVS/FERRE/
    PPXF... all 6 rows fit... but then when I click or mouse... only see
    1 rank." `dcc.Graph`'s `responsive` prop defaults to `'auto'`, which
    (per its own `isResponsive()`) treats *any* figure with `layout.
    autosize` truthy and *either* `layout.height` or `layout.width`
    unset as "fully responsive" — and unconditionally clears *both* on
    that path (`getLayoutOverride`'s `case true` branch), not just the
    one that was actually unset. `fill_container_width` above sets
    `width=None` (by design, for the fill-the-tab-width behaviour) while
    leaving `height` genuinely set — but that's enough on its own to
    trigger the "fully responsive" path regardless, silently discarding
    the deliberately-computed height and replacing it with `height:100%`
    of this function's own *wrapper*, i.e. `_TAB_VIEWPORT_HEIGHT`
    (650px) every time, not the figure's real content height. Short
    figures (RVS/FERRE/PPXF, a few panels) squish/stretch to 650px and
    happen to still look reasonable; Redrock's 6-12-row figure squishes
    to the same 650px and becomes unreadable — and, since this is a
    timing-sensitive client-side detection re-run on every `plot()` call
    rather than a one-time layout decision, which of the "squished" vs
    "the wrapper's own overflow-scroll finally kicks in" states wins at
    any given moment is genuinely race-prone, matching the reported
    instability.
    None of `dcc.Graph`'s three `responsive` modes give "responsive
    width, fixed height" independently (checked directly, not assumed):
    `False` forces `autosize:false` too, losing the fill-width behaviour
    entirely (Plotly falls back to its own hardcoded ~700px default
    width); `True`/`'auto'` (the default, and this bug) forces both
    height *and* width cleared. Real, already-proven fix used elsewhere
    in this exact codebase (`aps_l1_preview._stacked_graph`, confirmed
    via its own passing test): give the `dcc.Graph` component itself an
    explicit `style={"height": ...}` in pixels, matching the figure's own
    already-computed `layout.height`. That style lives on the *outer*
    div `dcc.Graph` renders, one level up from the inner plot div Dash's
    "responsive" logic sets to `height:100%` — so the inner div's 100%
    now resolves against *this* real, deliberate pixel value instead of
    against `_TAB_VIEWPORT_HEIGHT`, with no further client-side
    detection/timing involved at all.

    `pyaps-scrollbox` (see its CSS rules in aps_explorer.py's
    `index_string`) replaces the OS's own auto-hide overlay scrollbar with
    an always-rendered one whenever this box's content genuinely overflows
    — explicit user report: on some platforms the default overlay
    scrollbar is invisible except while actively scrolling, easy to miss
    entirely."""
    fixed_fig = style.fill_container_width(fig)
    real_height = fixed_fig.layout.height or 320
    return html.Div(
        dcc.Graph(figure=fixed_fig, config=_GRAPH_CONFIG,
                  style={"height": f"{real_height}px"}),
        className="pyaps-scrollbox",
        style={"height": _TAB_VIEWPORT_HEIGHT, "overflowY": "auto",
               "border": "1px solid #ddd", "borderRadius": "4px"},
    )


def update_tab_content(tab, aps_id, version):
    if not STATE.loaded() or aps_id is None:
        return html.Div("No source selected.")
    try:
        if tab == "redrock":
            return _scrollable_graph(_redrock_figure(aps_id))
        if tab == "rvs":
            return _scrollable_graph(_rvs_figure(aps_id))
        if tab == "ferre":
            return _scrollable_graph(_ferre_figure(aps_id))
        if tab == "ppxf":
            return _scrollable_graph(_ppxf_figure(aps_id))
        if tab == "emi":
            return _scrollable_graph(_emi_figure(aps_id))
    except Exception as e:
        return html.Div(f"Error rendering {tab}: {e}")
    return html.Div()


def update_value_tables(aps_id, version):
    if not STATE.loaded() or aps_id is None:
        return [], [], [], []
    class_rows = _class_table_rows(aps_id)
    class_cols = []
    if class_rows:
        zb = STATE.data["class_table"]
        for c in class_rows[0].keys():
            unit = get_column_unit(zb, c) if c != "Rank" else None
            class_cols.append({"name": f"{c} ({unit})" if unit else c, "id": c})
    return class_rows, class_cols, _star_table_rows(aps_id), _galaxy_table_rows(aps_id)
