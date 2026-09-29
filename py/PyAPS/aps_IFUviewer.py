"""
aps_IFUviewer.py - WEAVE IFU L2 viewer library (Dash figure/state layer)
=========================================================================

Library module (no standalone Dash app/CLI of its own — see
`aps_L2explorer.py`, the single interactive entry point for all L2
viewing) for the two IFU Voronoi-*patch* `_APS.fits` schemas — ExGal
(PPXF/EMIPPXF/line-strength) and Gal (RVS/FERRE) — as opposed to the
MOS/fibre-level schema `aps_IFUviewer.py`'s sibling `aps_MOSviewer.py`
handles. Provides: `AppState`/`STATE` (the loaded-dataset singleton),
`_load_aps_fits` (the loader), spatial-map and spectral-fit figure
builders (all on `PyAPS.apsPlot`, formerly PyQt5/embedded-matplotlib),
and the layout-piece/callback-logic functions `aps_L2explorer.py` wires
up into its own single Dash app.

Scope note: only what's functional in the legacy PyQt5 `EnhancedMapviewer`
app is ported — SFH maps, PPXF H5/H6, DIAGNOSTIC mode, contour overlays,
and "Compare Analysis Results" were all already dead/placeholder code
there (SFH is never populated by the current data loader; H5/H6 have no
plotMap branch; DIAGNOSTIC mode is a literal print-statement stub;
contours and compare-maps are unused/stubbed), so none of that is
carried over. Emission-line spectral fits are BIN-level only, matching
what the current data loader actually populates (legacy's SPAXEL-level
code path reads arrays the loader never sets either).
"""

from __future__ import annotations

import base64
import glob
import re
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from astropy.io import fits
from astropy.table import Table

from dash import dcc, html, dash_table, no_update

from werkzeug.local import LocalProxy

from PyAPS.mapviewer.loadData import emifile
from PyAPS.apsPlot.spaxel_map import spaxel_map_figure, select_colorscale
from PyAPS.apsPlot.spectra import spectrum_overlay_figure
from PyAPS.apsPlot import style
from PyAPS.apsPlot.datatable_export import csv_export_row
from PyAPS.aps_utils import get_column_unit
from PyAPS import aps_explorer_session as _sess

_GRAPH_CONFIG = style.GRAPH_CONFIG


# --------------------------------------------------------------------------- #
# Data loading — port of mapviewer/loadData.py's loadData(self), rewritten
# as a plain function returning a state dict instead of setting attributes
# on a QMainWindow. FITS reading/array logic is otherwise unchanged.
# --------------------------------------------------------------------------- #

def _load_aps_fits(dirprefix):
    """`_APS.fits` files come in two mutually-exclusive flavours, chosen
    upstream by `aps_L2merge.py`'s dispatcher based on which IFU worker
    ran (`ifuExGalL2merge` vs `ifuGalL2merge`) — a target either gets a
    *galaxy* (PPXF/EMIPPXF/line-strength) fit or a *stellar* (RVS/FERRE)
    fit, never both in the same file:

    - ExGal (galaxy) mode: `PATCH_TABLE`, `PATCH_BINSPEC` (raw spectra),
      `GALAXY_TABLE` (per-bin V/SIGMA/H3/H4/LAMBDA_R, per-line results,
      line-strength indices), optional `GALAXY_SPEC` (fitted curves —
      omitted via `no_spec_ext` to save disk space on large runs).
    - Gal (stellar) mode: `PATCH_TABLE`, `STAR_TABLE` (per-bin RVS *and*
      FERRE parameters merged into one table — RVS can succeed with
      FERRE skipped entirely, or vice versa, so availability is checked
      per-column, not per-extension), optional `STAR_SPEC` (fitted
      curves, same `no_spec_ext` convention). Note: no `PATCH_BINSPEC`
      in this mode — RVS/FERRE's own "observed" spectrum in `STAR_SPEC`
      (`FLUX_RVS`/`FLUX_NORM_FERRE`) is the closest thing to raw data.

    Both modes always have `PATCH_TABLE`, so spatial maps of whichever
    per-bin results exist always work; only the fitted-model spectral
    curves need the optional *_SPEC extension.
    """
    aps_file = dirprefix + "_APS.fits"
    if not Path(aps_file).is_file():
        raise FileNotFoundError(f"No such file: {aps_file}")

    hdul = fits.open(aps_file)
    extnames = [h.name for h in hdul]
    have_galaxy_table = "GALAXY_TABLE" in extnames
    have_binspec = "PATCH_BINSPEC" in extnames
    have_spec = "GALAXY_SPEC" in extnames
    have_star_table = "STAR_TABLE" in extnames
    have_star_spec = "STAR_SPEC" in extnames
    aps_gal_spec = hdul["GALAXY_SPEC"].data if have_spec else None
    aps_gal_table, aps_gal_table_head = (
        fits.getdata(aps_file, extname="GALAXY_TABLE", header=True) if have_galaxy_table else (None, {})
    )

    st = dict(aps_file=aps_file, have_spec=have_spec, have_star_spec=have_star_spec)
    # PPXF stellar-kinematics *results* live directly in GALAXY_TABLE
    # (always present); only the fitted model curve needs GALAXY_SPEC.
    st["ppxf_available"] = have_galaxy_table and "V" in aps_gal_table.columns.names and "SIGMA" in aps_gal_table.columns.names
    st["emippxf_available"] = have_galaxy_table and "N_LINES_ALL" in aps_gal_table_head
    st["ls_available"] = have_galaxy_table and "LS_RES" in aps_gal_table_head
    st["ls_res"] = aps_gal_table_head.get("LS_RES") if st["ls_available"] else None

    table = fits.open(aps_file)["PATCH_TABLE"].data
    # X is already sign-flipped and cos(dec)-corrected upstream, at cube
    # construction time (aps_ifu_v0.py: `cube['x'] = -1 * dx_deg * 3600 *
    # cos(dec)`), and passed straight through by ExGalPrepare.py/
    # aps_L2merge.py into PATCH_TABLE.X — confirmed on real data, where
    # X and Y already span the same ~85.7" range. The legacy PyQt5
    # mapviewer's loadData.py re-applied `* -1 * cos(dec)` on load, which
    # was correct for the old pre-merge `_table.fits` format but is a
    # double correction against current `_APS.fits` data — it shrinks X
    # by an extra cos(dec) factor (~0.5x at this target's +59.6 dec),
    # collapsing the map. Do not reapply it here.
    st["table"] = dict(
        X=table.X, Y=table.Y, XBIN=table.XBIN, YBIN=table.YBIN, BIN_ID=table.BIN_ID,
        FLUX=getattr(table, "FLUX", None), SNR=getattr(table, "SNR", None),
        SNRBIN=getattr(table, "SNRBIN", None), NSPAX=getattr(table, "NSPAX", None),
    )
    # Raw per-spaxel/per-bin record arrays, kept as-is (every column, not
    # just the ones the map/spectral panels use) so the "values" tables
    # below the tabs can show everything available for the current
    # selection without needing to know column names up front.
    st["patch_table_rec"] = table
    st["galaxy_table_rec"] = aps_gal_table
    try:
        st["pixelsize"] = float(fits.open(aps_file)["PATCH_TABLE"].header["PIXSIZE"])
    except Exception:
        st["pixelsize"] = 0.20

    # One CNAME for the *whole field* (every bin/spaxel shares it — IFU
    # patches don't have a per-bin CNAME the way MOS fibres/L1 targets do)
    # — same primary-header convention aps_utils.py's gen_targlist(IFU)
    # branch already uses when building the raw L1 target list
    # (BUNDLEID==0 -> CCNAME1, else CCNAME<BUNDLEID>). Purely informational
    # (shown in the Aladin info box); no code here depends on it.
    st["field_cname"] = None
    try:
        primary_header = hdul[0].header
        bundle_id = primary_header.get("BUNDLEID", 0)
        bndid = "1" if not bundle_id else str(int(bundle_id)).replace(" ", "")
        st["field_cname"] = primary_header.get(f"CCNAME{bndid}")
    except Exception:
        pass

    _, idx_short_to_long = np.unique(np.abs(table.BIN_ID), return_inverse=True)
    st["idx_short_to_long"] = idx_short_to_long
    # GALAXY_TABLE/GALAXY_SPEC/PATCH_BINSPEC/STAR_TABLE/STAR_SPEC each
    # carry their own BIN_ID column, and in practice all share the same
    # row order (bin_id N sits at row N) — but that's a convention, not
    # a guarantee (BIN_ID can be negative for spaxels outside the
    # Voronoi region, per idx_short_to_long's `np.abs`), so every per-bin
    # lookup below goes through one of these maps rather than trusting
    # bin_id == row position.
    if have_galaxy_table:
        st["galaxy_table_pos"] = {int(b): i for i, b in enumerate(aps_gal_table.BIN_ID)}

    if have_binspec:
        aps_patch_binspec = fits.getdata(aps_file, extname="PATCH_BINSPEC")
        st["spectra"] = aps_patch_binspec.SPEC
        st["lambda_"] = np.array(aps_patch_binspec.LOGLAM)[0, :]
        st["binspec_pos"] = {int(b): i for i, b in enumerate(aps_patch_binspec.BIN_ID)}
    if have_spec:
        st["spec_pos"] = {int(b): i for i, b in enumerate(aps_gal_spec.BIN_ID)}

    median_v_stellar = 0.0
    if st["ppxf_available"]:
        ppxf = np.array([aps_gal_table.V, aps_gal_table.SIGMA, aps_gal_table.H3,
                          aps_gal_table.H4, aps_gal_table.LAMBDA_R]).T
        median_v_stellar = float(np.nanmedian(ppxf[:, 0]))
        ppxf[:, 0] -= median_v_stellar
        # Pre-broadcast, per-bin (GALAXY_TABLE row order) — for lookups
        # keyed by bin ID (e.g. the PPXF-fit panel's V/sigma title),
        # indexed via galaxy_table_pos[bin_id] rather than the per-spaxel
        # broadcast below, which is only valid indexed by spaxel position.
        st["ppxf_by_bin"] = ppxf
        st["ppxf_results"] = ppxf[idx_short_to_long, :]
        if have_spec and "MODEL_PPXF" in aps_gal_spec.columns.names:
            st["ppxf_bestfit"] = aps_gal_spec.MODEL_PPXF
            st["ppxf_lambda"] = aps_gal_spec.LOGLAM_PPXF[0, :]
            st["ppxf_goodpix"] = aps_gal_spec.GOODPIX_PPXF

    if st["emippxf_available"]:
        emi_table = emifile(aps_file, fits.open(aps_file).index_of("GALAXY_TABLE"))
        line_names = [line["name"] for line in emi_table]
        line_lambdas = [line["_lambda"] for line in emi_table]
        n_lines = len(line_names)
        n_table_bins = len(aps_gal_table)

        vel = np.full((n_table_bins, n_lines), np.nan)
        sigma = np.full((n_table_bins, n_lines), np.nan)
        flux = np.full((n_table_bins, n_lines), np.nan)
        ampl = np.full((n_table_bins, n_lines), np.nan)
        aon = np.full((n_table_bins, n_lines), np.nan)

        for i, line_name in enumerate(line_names):
            if emi_table["action"][i] != "f":
                continue
            line_lambda = emi_table["_lambda"][i]
            candidates = list(dict.fromkeys([
                line_name, f"{line_name.split('_')[0]}_{line_lambda:.2f}",
                f"{line_name.split('_')[0]}_{line_lambda:.1f}",
                f"{line_name.split('_')[0]}_{line_lambda:.0f}",
            ]))
            for name in candidates:
                found = False
                for arr, prefix in ((vel, "V"), (sigma, "SIGMA"), (flux, "FLUX"),
                                     (ampl, "AMPL"), (aon, "AON")):
                    col = f"{prefix}_{name}"
                    if col in aps_gal_table.columns.names:
                        arr[:, i] = aps_gal_table[col]
                        found = True
                if found:
                    break

        emi_results = np.stack([vel, sigma, flux, ampl, aon], axis=1)
        gandalf_results = emi_results[idx_short_to_long, :, :]
        if st["ppxf_available"]:
            gandalf_results[:, 0, :] -= median_v_stellar
        st["gandalf_results"] = gandalf_results
        # Pre-broadcast, per-bin (not per-spaxel) version — indexed via
        # galaxy_table_pos[bin_id], for lookups keyed by bin ID rather
        # than by spaxel (e.g. the AoN-threshold emission-line markers).
        st["emi_results_by_bin"] = emi_results
        st["line_names"] = np.array(line_names)
        st["line_lambdas"] = np.array(line_lambdas)

        # Current pipeline naming is "_EMI" (EMIPPXF module) — legacy
        # mapviewer/loadData.py still looked for "_GAND" (GANDALF-era
        # naming), which no longer exists in any real output. EMISSION_EMI
        # is the pure emission-line model; MODEL_EMI is the *total*
        # (stellar+emission) fit — same distinction established when
        # porting apsPlot/emi_bin.py earlier (MODEL_EMI - EMISSION_EMI =
        # the stellar/continuum component).
        if have_spec and "EMISSION_EMI" in aps_gal_spec.columns.names:
            st["emission_subtracted_bin"] = np.array(aps_gal_spec.FLUX_CLEAN_EMI)
            st["gandalf_bestfit"] = aps_gal_spec.EMISSION_EMI
            st["gandalf_lambda"] = aps_gal_spec.LOGLAM_EMI[0, :]
            st["gandalf_goodpix"] = aps_gal_spec.GOODPIX_EMI
    else:
        st["gandalf_results"] = None
        st["line_names"] = np.array([])

    if st["ls_available"]:
        ls_cols = [c for c in aps_gal_table.columns.names
                   if not c.startswith(("V_", "SIGMA_", "FLUX_", "AMPL_", "AON_", "ERR_"))
                   and c not in ("BIN_ID", "X", "Y", "SNR", "H3", "H4", "LAMBDA_R")]
        if ls_cols:
            ls_data = np.array([aps_gal_table[c] for c in ls_cols]).T
            st["line_strength"] = ls_data[idx_short_to_long, :]
            st["ls_list"] = np.array(ls_cols)
        else:
            st["ls_available"] = False

    # Gal (stellar) mode: STAR_TABLE merges RVS and FERRE parameters into
    # one table via an outer join on BIN_ID (aps_L2merge.py:ifuGalL2merge)
    # — RVS can succeed with FERRE skipped entirely (or vice versa), so
    # each is gated on its own columns existing, not on STAR_TABLE as a
    # whole. VRAD/VRAD_ERR are the only RVS columns with no "_RVS" suffix.
    st["rvs_available"] = False
    st["ferre_available"] = False
    if have_star_table:
        aps_star_table = fits.getdata(aps_file, extname="STAR_TABLE")
        st["star_table_rec"] = aps_star_table
        st["star_table_pos"] = {int(b): i for i, b in enumerate(aps_star_table.BIN_ID)}
        st["rvs_available"] = "VRAD" in aps_star_table.columns.names
        st["ferre_available"] = "TEFF_FERRE" in aps_star_table.columns.names

    if have_star_spec:
        aps_star_spec = fits.getdata(aps_file, extname="STAR_SPEC")
        st["star_spec_pos"] = {int(b): i for i, b in enumerate(aps_star_spec.BIN_ID)}
        if "MODEL_RVS" in aps_star_spec.columns.names:
            # RVS's array columns are variable-length FITS P-columns (rows
            # can differ in length) — `aps_star_spec.LAMBDA_RVS` is an
            # object array of 1-D arrays, so index-per-row (`[pos]`) works
            # directly; a plain 2-D slice would not.
            st["rvs_lambda_rec"] = aps_star_spec.LAMBDA_RVS
            st["rvs_flux_rec"] = aps_star_spec.FLUX_RVS
            st["rvs_model_rec"] = aps_star_spec.MODEL_RVS
        if "MODEL_FERRE" in aps_star_spec.columns.names:
            st["ferre_lambda_rec"] = aps_star_spec.LAMBDA_FERRE
            st["ferre_flux_rec"] = aps_star_spec.FLUX_NORM_FERRE
            st["ferre_model_rec"] = aps_star_spec.MODEL_FERRE

    return st


# --------------------------------------------------------------------------- #
# maptype resolution — port of mapviewer/plotData.py's plotMap dispatch,
# minus the matplotlib rendering (kept purely as data lookup here).
# --------------------------------------------------------------------------- #

def _list_maptypes(st):
    """All maptype strings selectable in this dataset, grouped for the dropdown."""
    groups = []
    table_opts = [(f"Table_{k}", k) for k in ("FLUX", "SNR", "SNRBIN", "NSPAX", "BIN_ID")
                  if st["table"].get(k) is not None]
    if table_opts:
        # "Table" -> "Patch table" per explicit request: this group is
        # PATCH_TABLE's own columns (FLUX/SNR/SNRBIN/NSPAX/BIN_ID), and
        # the bare word "Table" read as meaningless/generic next to the
        # other groups' real names (Stellar, Emission lines, ...). The
        # maptype keys themselves (Table_FLUX etc., used throughout
        # _resolve_maptype and by callers matching on them) are
        # unchanged -- only this dropdown-facing group label moved.
        groups.append(("Patch table", table_opts))
    if st["ppxf_available"]:
        groups.append(("Stellar (PPXF)", [(f"PPXF_{p}", p) for p in ("V", "SIGMA", "H3", "H4", "LAMBDAR")]))
    if st["gandalf_results"] is not None and len(st["line_names"]):
        emi_opts = []
        for line in st["line_names"]:
            for prefix in ("V", "SIGMA", "FLUX", "AMPL", "AON"):
                emi_opts.append((f"{prefix}_{line}", f"{prefix} {line}"))
        groups.append(("Emission lines", emi_opts))
    if st.get("ls_available") and "ls_list" in st:
        groups.append(("Line Strength", [(f"LS_{ls}", ls) for ls in st["ls_list"]]))
    if st.get("rvs_available"):
        groups.append(("Stellar (RVS)", [(f"RVS_{p}", p) for p in
                                          ("VRAD", "TEFF", "LOGG", "FEH", "ALPHA", "VSINI", "SNR")]))
    if st.get("ferre_available"):
        groups.append(("Stellar (FERRE)", [(f"FERRE_{p}", p) for p in
                                            ("TEFF", "LOGG", "FEH", "ALPHA", "MICRO", "SNR", "CHISQ")]))
    return groups


def _broadcast_by_bin_id(bin_id_col, pos_map, values):
    """Per-spaxel array built by looking up each spaxel's own BIN_ID in
    `pos_map` (bin_id -> row position in `values`'s source table) — the
    per-bin-table analogue of `idx_short_to_long`, used for STAR_TABLE
    since its row order isn't tied to GALAXY_TABLE's (it's built by its
    own independent outer join, see `_load_aps_fits`). Spaxels whose bin
    has no matching row (shouldn't normally happen, but PATCH_TABLE and
    STAR_TABLE come from separate join steps upstream) get NaN.
    """
    idx = np.array([pos_map.get(int(b), -1) for b in bin_id_col])
    out = np.full(len(idx), np.nan, dtype=np.float64)
    valid = idx >= 0
    out[valid] = np.asarray(values, dtype=np.float64)[idx[valid]]
    return out


def _resolve_maptype(st, maptype):
    """Returns (data, title, colorbar_label) for a maptype string, or
    (None, None, None) if unavailable — same branching as legacy `plotMap`."""
    table = st["table"]
    if maptype == "Table_FLUX":
        return table["FLUX"], "Total Flux", "Flux"
    if maptype == "Table_BIN_ID":
        return table["BIN_ID"], "Bin ID", "Bin ID"
    if maptype == "Table_SNR":
        return table["SNR"], "Signal-to-Noise Ratio", "S/N"
    if maptype == "Table_SNRBIN":
        return table["SNRBIN"], "Bin S/N Ratio", "Bin S/N"
    if maptype == "Table_NSPAX":
        return table["NSPAX"], "Spaxels per Bin", "N spaxels"

    if maptype.startswith("PPXF_"):
        if not st["ppxf_available"]:
            return None, None, None
        param = maptype[len("PPXF_"):]
        col = {"V": 0, "SIGMA": 1, "H3": 2, "H4": 3, "LAMBDAR": 4}.get(param)
        if col is None:
            return None, None, None
        labels = {"V": ("Stellar Velocity", "V [km/s]"), "SIGMA": ("Stellar Velocity Dispersion", "σ [km/s]"),
                  "H3": ("H3 Moment", "H3"), "H4": ("H4 Moment", "H4"), "LAMBDAR": ("Lambda R", "λ_R")}
        title, cbar = labels[param]
        return st["ppxf_results"][:, col], title, cbar

    if maptype.split("_", 1)[0] in ("V", "SIGMA", "FLUX", "AMPL", "AON") and st["gandalf_results"] is not None:
        data_type, line_name = maptype.split("_", 1)
        line_names = list(st["line_names"])
        if line_name not in line_names:
            return None, None, None
        line_idx = line_names.index(line_name)
        col = {"V": 0, "SIGMA": 1, "FLUX": 2, "AMPL": 3, "AON": 4}[data_type]
        labels = {"V": (f"{line_name} Velocity", "V [km/s]"), "SIGMA": (f"{line_name} Velocity Dispersion", "σ [km/s]"),
                  "FLUX": (f"{line_name} Flux", "Flux"), "AMPL": (f"{line_name} Amplitude", "Amplitude"),
                  "AON": (f"{line_name} AoN", "AoN")}
        title, cbar = labels[data_type]
        return st["gandalf_results"][:, col, line_idx], title, cbar

    if maptype.startswith("LS_") and st.get("ls_available"):
        param = maptype[len("LS_"):]
        ls_list = list(st["ls_list"])
        if param not in ls_list:
            return None, None, None
        return st["line_strength"][:, ls_list.index(param)], f"Line Strength: {param}", param

    if maptype.startswith("RVS_"):
        if not st.get("rvs_available"):
            return None, None, None
        param = maptype[len("RVS_"):]
        # VRAD/VRAD_ERR are the only RVS columns with no "_RVS" suffix.
        col_labels = {
            "VRAD": ("VRAD", "Radial Velocity (RVS)", "V [km/s]"),
            "TEFF": ("TEFF_RVS", "Effective Temperature (RVS)", "Teff [K]"),
            "LOGG": ("LOGG_RVS", "Surface Gravity (RVS)", "log g"),
            "FEH": ("FEH_RVS", "Metallicity (RVS)", "[Fe/H]"),
            "ALPHA": ("ALPHA_RVS", "Alpha Abundance (RVS)", "[α/Fe]"),
            "VSINI": ("VSINI_RVS", "Rotational Velocity (RVS)", "v sin i [km/s]"),
            "SNR": ("SNR_RVS", "S/N (RVS)", "S/N"),
        }
        if param not in col_labels:
            return None, None, None
        col, title, cbar = col_labels[param]
        rec = st["star_table_rec"]
        if col not in rec.columns.names:
            return None, None, None
        return _broadcast_by_bin_id(table["BIN_ID"], st["star_table_pos"], rec[col]), title, cbar

    if maptype.startswith("FERRE_"):
        if not st.get("ferre_available"):
            return None, None, None
        param = maptype[len("FERRE_"):]
        col_labels = {
            "TEFF": ("TEFF_FERRE", "Effective Temperature (FERRE)", "Teff [K]"),
            "LOGG": ("LOGG_FERRE", "Surface Gravity (FERRE)", "log g"),
            "FEH": ("FEH_FERRE", "Metallicity (FERRE)", "[Fe/H]"),
            "ALPHA": ("ALPHA_FERRE", "Alpha Abundance (FERRE)", "[α/Fe]"),
            "MICRO": ("MICRO_FERRE", "Microturbulence (FERRE)", "ξ [dex]"),
            "SNR": ("SNR_FERRE", "S/N (FERRE)", "S/N"),
            "CHISQ": ("CHISQ_FERRE", "χ² (FERRE)", "χ²"),
        }
        if param not in col_labels:
            return None, None, None
        col, title, cbar = col_labels[param]
        rec = st["star_table_rec"]
        if col not in rec.columns.names:
            return None, None, None
        return _broadcast_by_bin_id(table["BIN_ID"], st["star_table_pos"], rec[col]), title, cbar

    return None, None, None


def _auto_select_initial_maptype(st):
    for candidate in ("Table_FLUX", "Table_SNR", "Table_BIN_ID", "PPXF_V"):
        data, _, _ = _resolve_maptype(st, candidate)
        if data is not None:
            return candidate
    return None


# --------------------------------------------------------------------------- #
# App state + figure builders
# --------------------------------------------------------------------------- #

class AppState:
    def __init__(self):
        self.data = None
        self.current_maptype = None
        self.aon_threshold = 4.0
        self.marker_color = "red"
        self.outpath = None
        self.headname = None
        self.color_vmin = None  # None = auto (1st/99th percentile, see spaxel_map_figure)
        self.color_vmax = None
        self.color_scale = "linear"  # see aps_explorer.COLOR_SCALE_OPTIONS

    def loaded(self):
        return self.data is not None

    def load(self, outpath, headname):
        dirprefix = str(Path(outpath) / headname)
        self.data = _load_aps_fits(dirprefix)
        self.current_maptype = _auto_select_initial_maptype(self.data)
        self.outpath = outpath
        self.headname = headname
        self.color_vmin = None
        self.color_vmax = None
        self.color_scale = "linear"


# A LocalProxy, not a bare instance — see aps_l1_preview.py's own
# STATE = LocalProxy(...) for the full rationale (identical pattern,
# aps_explorer_session.py's own module docstring for the full design).
STATE = LocalProxy(lambda: _sess.current_bundle().get_or_create("ifu", AppState))


def _build_map_figure(maptype, selected_bin=None):
    st = STATE.data
    data, title, cbar_label = _resolve_maptype(st, maptype)
    if data is None:
        return spaxel_map_figure([], [], [], [], title="No data")
    # BIN_ID itself, not a plain positional index -- matches
    # aps_explorer.py's own extra-map-panel figure builder (same real
    # bug, same fix; see that call site's own comment) and on_map_click
    # below, which now reads customdata[0] as the BIN_ID directly.
    indices = st["table"]["BIN_ID"]
    selected_xy = None
    if selected_bin is not None:
        # Highlight the bin's centroid (XBIN/YBIN, same value repeated
        # across every spaxel belonging to that bin — matches legacy's
        # `binMarker`), not an arbitrary member spaxel's raw X/Y.
        match = np.flatnonzero(st["table"]["BIN_ID"] == selected_bin)
        if match.size:
            i = match[0]
            selected_xy = (st["table"]["XBIN"][i], st["table"]["YBIN"][i])
    return spaxel_map_figure(
        st["table"]["X"], st["table"]["Y"], data, indices,
        title=title, colorbar_label=cbar_label, selected_xy=selected_xy,
        marker_color=STATE.marker_color,
        # select_colorscale matches against the raw maptype string
        # (e.g. "PPXF_V"), not the human-readable title ("Stellar
        # Velocity") — passing title here would silently fall through
        # to the Inferno default for every quantity.
        colorscale=select_colorscale(maptype),
        vmin=STATE.color_vmin, vmax=STATE.color_vmax,
    )


def _bin_spectrum_figure(bin_id):
    st = STATE.data
    pos = st["binspec_pos"][bin_id]
    wave = np.exp(st["lambda_"])
    flux = st["spectra"][pos, :]
    match = np.flatnonzero(st["table"]["BIN_ID"] == bin_id)
    snr_bin = st["table"].get("SNRBIN")
    title = f"Spectrum — Bin {bin_id}"
    if snr_bin is not None and match.size:
        title += f"  (S/N: {snr_bin[match[0]]:.1f})"
    return spectrum_overlay_figure(
        ["bin"], {"bin": wave}, {"bin": flux}, {"bin": None},
        data_label="Flux", flux_unit="Flux", wave_label="Wavelength [Å]",
        figure_title=title, percentile_clip=(1, 99),
    )


def _ppxf_fit_figure(bin_id):
    st = STATE.data
    spec_pos = st["spec_pos"][bin_id]
    ppxf_wave = np.exp(st["ppxf_lambda"])
    ppxf_fit = st["ppxf_bestfit"][spec_pos, :]
    obs_wave = np.exp(st["lambda_"])
    observed = st["spectra"][st["binspec_pos"][bin_id], :]
    observed_interp = (observed if len(obs_wave) == len(ppxf_wave)
                        else np.interp(ppxf_wave, obs_wave, observed))
    goodpix_idx = st["ppxf_goodpix"][spec_pos, :]
    mask = np.ones(len(ppxf_wave), dtype=bool)
    valid = goodpix_idx[goodpix_idx >= 0]
    valid = valid[valid < len(ppxf_wave)]
    mask[valid] = False

    residual = observed_interp - ppxf_fit
    title = f"PPXF Fit — Bin {bin_id}"
    if "ppxf_by_bin" in st:
        bin_pos = st["galaxy_table_pos"][bin_id]
        v, sigma = st["ppxf_by_bin"][bin_pos, 0], st["ppxf_by_bin"][bin_pos, 1]
        title += f"<br>V = {v:.1f} km/s, σ = {sigma:.1f} km/s"

    return spectrum_overlay_figure(
        ["bin"], {"bin": ppxf_wave}, {"bin": observed_interp}, {"bin": ppxf_fit},
        residuals={"bin": residual}, residual_zero_line=True,
        mask={"bin": mask}, mask_color="gray",
        data_label="Observed", rank_labels=["PPXF Best Fit"],
        flux_unit="Flux", wave_label="Wavelength [Å]",
        figure_title=title, percentile_clip=(1, 99),
    )


def _emission_fit_figure(bin_id):
    st = STATE.data
    spec_pos = st["spec_pos"][bin_id]
    wave = np.exp(st["gandalf_lambda"])
    model = st["gandalf_bestfit"][spec_pos, :]
    clean = st.get("emission_subtracted_bin")
    clean = clean[spec_pos, :] if clean is not None else None
    mask = None
    if "gandalf_goodpix" in st:
        goodpix = st["gandalf_goodpix"][spec_pos, :].astype(bool)
        mask = ~goodpix
    fig = spectrum_overlay_figure(
        ["bin"], {"bin": wave}, {"bin": clean if clean is not None else model}, {"bin": model},
        mask=({"bin": mask} if mask is not None else None), mask_color="gray",
        data_label="Continuum-subtracted", rank_labels=["Emission Model"],
        flux_unit="Flux", wave_label="Wavelength [Å]",
        figure_title=f"Emission-line Fit — Bin {bin_id}", percentile_clip=(1, 99),
    )
    _add_emission_line_markers(fig, bin_id)
    return fig


def _add_emission_line_markers(fig, bin_id):
    """Mark detected emission lines above the AoN threshold — port of
    legacy `mapviewer/plotData.py:addEmissionLineMarkers`. Lines at
    AoN >= 2x threshold are drawn solid/bold; AoN >= threshold but below
    that are drawn as light dotted lines, matching the legacy styling
    that made strong vs. marginal detections visually distinct."""
    st = STATE.data
    if "line_lambdas" not in st or st.get("emi_results_by_bin") is None:
        return
    threshold = STATE.aon_threshold
    bin_pos = st["galaxy_table_pos"][bin_id]
    aon_col = st["emi_results_by_bin"][bin_pos, 4, :]
    for line_name, lam, aon in zip(st["line_names"], st["line_lambdas"], aon_col):
        if not np.isfinite(aon) or aon < threshold or not np.isfinite(lam):
            continue
        strong = aon >= 2 * threshold
        fig.add_vline(
            x=float(lam),
            line_width=1.5 if strong else 1,
            line_dash="solid" if strong else "dot",
            line_color="rgba(80,80,80,0.8)" if strong else "rgba(150,150,150,0.6)",
            annotation_text=line_name, annotation_position="top",
            annotation_font_size=9 if strong else 8,
            annotation_textangle=-90,
        )


def _ls_spectrum_figure(bin_id):
    st = STATE.data
    wave = np.exp(st["lambda_"])
    flux = st["spectra"][st["binspec_pos"][bin_id], :]
    return spectrum_overlay_figure(
        ["bin"], {"bin": wave}, {"bin": flux}, {"bin": None},
        data_label="Flux", flux_unit="Flux", wave_label="Wavelength [Å]",
        figure_title=f"Line Strength Analysis — Bin {bin_id}", percentile_clip=(1, 99),
    )


def _rvs_fit_figure(bin_id):
    """RVS fit panel (Gal/stellar mode) — RVS's own "observed" spectrum
    (`FLUX_RVS`, the RVSPECFIT-prepared input) overlaid with its best-fit
    model (`MODEL_RVS`); wavelengths in `STAR_SPEC` are already linear
    Angstrom (RVS/FERRE run on `_BINSpectra_linear.fits`, unlike PPXF/EMI
    which run on the log-rebinned `PATCH_BINSPEC` — no `np.exp()` here)."""
    st = STATE.data
    pos = st["star_spec_pos"][bin_id]
    wave = np.asarray(st["rvs_lambda_rec"][pos], dtype=np.float64)
    observed = np.asarray(st["rvs_flux_rec"][pos], dtype=np.float64)
    model = np.asarray(st["rvs_model_rec"][pos], dtype=np.float64)

    title = f"RVS Fit — Bin {bin_id}"
    bin_pos = st["star_table_pos"].get(bin_id)
    if bin_pos is not None:
        rec = st["star_table_rec"]
        vrad, teff, logg, feh = (rec.VRAD[bin_pos], rec.TEFF_RVS[bin_pos],
                                  rec.LOGG_RVS[bin_pos], rec.FEH_RVS[bin_pos])
        title += (f"<br>V_rad = {vrad:.1f} km/s, Teff = {teff:.0f} K, "
                  f"log g = {logg:.2f}, [Fe/H] = {feh:.2f}")

    return spectrum_overlay_figure(
        ["bin"], {"bin": wave}, {"bin": observed}, {"bin": model},
        data_label="Observed", rank_labels=["RVS Best Fit"],
        flux_unit="Flux", wave_label="Wavelength [Å]",
        figure_title=title, percentile_clip=(1, 99),
    )


def _ferre_fit_figure(bin_id):
    """FERRE fit panel (Gal/stellar mode) — FERRE's normalized input
    spectrum (`FLUX_NORM_FERRE`) overlaid with its best-fit model
    (`MODEL_FERRE`); linear-Angstrom wavelengths, same as RVS."""
    st = STATE.data
    pos = st["star_spec_pos"][bin_id]
    wave = np.asarray(st["ferre_lambda_rec"][pos], dtype=np.float64)
    observed = np.asarray(st["ferre_flux_rec"][pos], dtype=np.float64)
    model = np.asarray(st["ferre_model_rec"][pos], dtype=np.float64)

    title = f"FERRE Fit — Bin {bin_id}"
    bin_pos = st["star_table_pos"].get(bin_id)
    if bin_pos is not None:
        rec = st["star_table_rec"]
        teff, logg, feh, alpha = (rec.TEFF_FERRE[bin_pos], rec.LOGG_FERRE[bin_pos],
                                   rec.FEH_FERRE[bin_pos], rec.ALPHA_FERRE[bin_pos])
        title += (f"<br>Teff = {teff:.0f} K, log g = {logg:.2f}, "
                  f"[Fe/H] = {feh:.2f}, [α/Fe] = {alpha:.2f}")

    return spectrum_overlay_figure(
        ["bin"], {"bin": wave}, {"bin": observed}, {"bin": model},
        data_label="Normalized Input", rank_labels=["FERRE Best Fit"],
        flux_unit="Normalized Flux", wave_label="Wavelength [Å]",
        figure_title=title, percentile_clip=(1, 99),
    )


# --------------------------------------------------------------------------- #
# Value tables — raw per-spaxel (PATCH_TABLE) and per-bin (GALAXY_TABLE)
# column dumps for the current selection, below the spectral-fit tabs.
# --------------------------------------------------------------------------- #

def _json_safe(v):
    """FITS record-array scalars (numpy float32/64, int16/32, bytes,
    NaN) aren't directly what Dash's JSON encoder wants for DataTable
    cells — normalize to plain Python str/int/float/None."""
    if isinstance(v, (bytes, np.bytes_)):
        return v.decode(errors="replace").strip()
    if isinstance(v, (np.floating, float)):
        v = float(v)
        return None if np.isnan(v) else round(v, 6)
    if isinstance(v, (np.integer, int)):
        return int(v)
    return str(v)


def _spaxel_table_data(bin_id):
    """(column defs, rows) for every PATCH_TABLE spaxel belonging to
    `bin_id` — the "which spaxels make up this Voronoi bin" mapping.
    Column *display names* get a "(unit)" suffix where the FITS file
    actually has one — `id` stays the bare column name either way, so
    row-data lookups elsewhere are unaffected."""
    st = STATE.data
    rec = st["patch_table_rec"]
    cols = rec.columns.names
    col_defs = [{"name": f"{c} ({get_column_unit(rec, c)})" if get_column_unit(rec, c) else c, "id": c}
                for c in cols]
    match = np.flatnonzero(rec.BIN_ID == bin_id)
    rows = [{c: _json_safe(rec[c][i]) for c in cols} for i in match]
    return col_defs, rows


def _bin_params_table_data(bin_id):
    """(rows, source_label) Parameter/Unit/Value rows for every column of
    whichever per-bin results table this dataset has — `GALAXY_TABLE`
    (ExGal mode: PPXF/EMIPPXF/line-strength) or `STAR_TABLE` (Gal mode:
    RVS/FERRE), see `_load_aps_fits` for why a file only ever has one of
    the two. Every fitted parameter available for the bin, not just what
    the current map/tab happens to show."""
    st = STATE.data
    if "galaxy_table_rec" in st and st["galaxy_table_rec"] is not None:
        rec, pos_map, label = st["galaxy_table_rec"], st["galaxy_table_pos"], "Bin results (GALAXY_TABLE)"
    elif "star_table_rec" in st:
        rec, pos_map, label = st["star_table_rec"], st["star_table_pos"], "Bin results (STAR_TABLE)"
    else:
        return [], "Bin results"
    pos = pos_map.get(bin_id)
    if pos is None:
        return [], label
    return [{"Parameter": c, "Unit": get_column_unit(rec, c) or "", "Value": _json_safe(rec[c][pos])}
            for c in rec.columns.names], label


def _value_tables_panel():
    return html.Div([
        html.H4("Bin & Spaxel Data", style={"marginBottom": "4px"}),
        html.Div([
            html.Div([
                html.H5("Spaxels in this bin (PATCH_TABLE)"),
                csv_export_row("spaxel-values-table", dash_table.DataTable(
                    id="spaxel-values-table",
                    sort_action="native",
                    page_size=10,
                    **style.DATATABLE_KWARGS,
                )),
            ], style={"width": "58%", "display": "inline-block", "verticalAlign": "top"}),
            html.Div([
                html.H5(id="bin-values-title", children="Bin results"),
                csv_export_row("bin-values-table", dash_table.DataTable(
                    id="bin-values-table",
                    columns=[{"name": "Parameter", "id": "Parameter"}, {"name": "Unit", "id": "Unit"},
                             {"name": "Value", "id": "Value"}],
                    sort_action="native",
                    filter_action="native",
                    page_size=12,
                    **{**style.DATATABLE_KWARGS,
                       "style_cell_conditional": [{"if": {"column_id": "Parameter"}, "textAlign": "left",
                                                    "fontWeight": "600"},
                                                   {"if": {"column_id": "Unit"}, "color": "var(--pyaps-ink-faint)", "width": "70px"}]},
                )),
            ], style={"width": "40%", "display": "inline-block", "verticalAlign": "top", "marginLeft": "2%"}),
        ]),
    ], style={"marginTop": "24px", "borderTop": "1px solid #ddd", "paddingTop": "12px"})


# --------------------------------------------------------------------------- #
# Layout pieces and callback logic — library functions only. This module no
# longer owns a Dash `app`/layout/CLI entry point of its own: PyAPS.aps_L2explorer
# is the single Dash app for all L2 viewing now, and imports this module
# purely for its state (AppState/STATE), loader (_load_aps_fits), figure
# builders, and the layout/callback functions kept below (still plain,
# undecorated functions — reused by the explorer's own callbacks).
# --------------------------------------------------------------------------- #


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


def _maptype_category_for(st, maptype):
    """Which _list_maptypes group a maptype value belongs to — falls back
    to the first available category if the value isn't found in any (e.g.
    a stale current_maptype left over from a previous, differently-shaped
    dataset). Never returns None if any category exists at all."""
    groups = _list_maptypes(st)
    for label, opts in groups:
        if any(v == maptype for v, _ in opts):
            return label
    return groups[0][0] if groups else None


def _available_tabs(st):
    """Which spectral-panel tabs make sense for the loaded dataset — ExGal
    (galaxy) and Gal (stellar) mode files never share the same extensions
    (see `_load_aps_fits`), and within Gal mode RVS/FERRE are each
    independently optional, so this can't be a fixed list."""
    def _tab(label, value):
        return dcc.Tab(label=label, value=value, style=style.TAB_STYLE,
                        selected_style=style.TAB_SELECTED_STYLE)

    tabs = []
    if "spectra" in st:
        tabs.append(_tab("Spectrum", "spectrum"))
    if st.get("ppxf_available"):
        tabs.append(_tab("Stellar (PPXF)", "stellar"))
    if st.get("gandalf_results") is not None:
        tabs.append(_tab("Emission", "emission"))
    if st.get("ls_available"):
        tabs.append(_tab("Line Strength", "ls"))
    if st.get("rvs_available"):
        tabs.append(_tab("Stellar (RVS)", "rvs"))
    if st.get("ferre_available"):
        tabs.append(_tab("Stellar (FERRE)", "ferre"))
    if not tabs:
        tabs.append(_tab("No spectral data", "none"))
    tabs.append(_tab("Processing History", "history"))
    return tabs


def _settings_panel():
    st = STATE.data
    # A single flat "Map" dropdown got genuinely unusable on a large
    # dataset — one real file had 787 selectable quantities (Table/PPXF/
    # one V-SIGMA-FLUX-AMPL-AON group per detected emission line/Line
    # Strength index) in one list, which read as "I can't see the options
    # for map... it only shows Flux, no way to select other options" even
    # though every option really was there. Split into a Category dropdown
    # (Table/Stellar PPXF/Emission lines/Line Strength/Stellar RVS/Stellar
    # FERRE — whichever the dataset actually has, same grouping
    # _list_maptypes already computed) plus a Map dropdown scoped to just
    # that category, so no single list is ever more than one category's
    # worth of options — each category also stays independently
    # searchable (Dash's dcc.Dropdown default) for the ones that are still
    # long (e.g. "Emission lines" alone).
    groups = _list_maptypes(st)
    current_category = _maptype_category_for(st, STATE.current_maptype)
    category_options = [{"label": f"{label} ({len(opts)})", "value": label} for label, opts in groups]
    quantity_options = [{"label": label, "value": value}
                         for value, label in dict(groups).get(current_category, [])]
    # A real flexbox row with alignItems: "flex-end" -- explicit
    # request ("Go to bin id is not aligned with table selector and map
    # selector"). display:inline-block + verticalAlign:top (the
    # previous approach) aligns each group's own *top* edge, but
    # dcc.Dropdown renders visibly taller than a plain dcc.Input, so the
    # actual controls (not just the labels above them) landed at
    # different heights. flex-end instead aligns every group's own
    # *bottom* edge, which is what actually reads as "these controls
    # are in one aligned row" regardless of each group's total height.
    return html.Div([
        html.Div([
            html.Label("Map category:", style={"display": "block", "marginBottom": "4px"}),
            dcc.Dropdown(id="maptype-category-dd", options=category_options, value=current_category,
                         clearable=False, style={"width": "200px"}),
        ], style={"marginRight": "20px"}),
        html.Div([
            html.Label("Map:", style={"display": "block", "marginBottom": "4px"}),
            dcc.Dropdown(id="maptype-dd", options=quantity_options, value=STATE.current_maptype,
                         clearable=False, searchable=True,
                         placeholder=f"Type to search ({len(quantity_options)} options)...",
                         style={"width": "320px"}),
        ], style={"marginRight": "20px"}),
        html.Div([
            html.Label("Go to bin ID:", style={"display": "block", "marginBottom": "4px"}),
            html.Div([
                dcc.Input(id="bin-id-input", type="number",
                          style={"width": "100px", **style.SOFT_INPUT_HEIGHT_STYLE}),
                html.Button("Go", id="bin-id-go", n_clicks=0,
                            style={"marginLeft": "6px", **style.SOFT_BUTTON_STYLE}),
            ]),
        ], style={"marginRight": "20px"}),
        # The AoN-threshold control (which detected emission lines get
        # labelled on the Emission tab) was removed per explicit user
        # request ("not necessary") — _add_emission_line_markers still
        # uses STATE.aon_threshold internally, just fixed at its default
        # (4.0) rather than user-adjustable, so the underlying labelling
        # behaviour is unchanged, only the control is gone.
        # Colour range (Min/Max/Reset) used to live here — moved to sit
        # directly under the Aladin colour legend instead (see
        # aps_explorer.py's _aladin_color_range_control), right where the
        # colours it controls are actually shown, rather than in a
        # same-looking-as-everything-else row up in the settings panel
        # with no visual link to the legend at all.
        #
        # IFU deliberately has no #aladin-color-by-row of its own (no
        # single "Colour Aladin points by" dropdown the way L1/MOS have —
        # Category/Map above serves a broader purpose across several
        # tabs, not just Aladin's own colouring) — safe to just not have
        # the id at all: aps_explorer.update_map_mode's 3D-mode hiding of
        # that row is a pure CSS rule keyed off #main-panel's own
        # className (`.mode-3d #aladin-color-by-row`, see that
        # callback's own comment), which is inert wherever the id
        # doesn't exist rather than erroring the way a direct Output at
        # a missing id would.
    ], style={"padding": "10px 0", "display": "flex", "alignItems": "flex-end", "flexWrap": "wrap"})


# --------------------------------------------------------------------------- #
# Processing History — pipeline-only artifacts (target detection, segmentation,
# Voronoi-binning-stage diagnostics) that can't be regenerated live from the
# merged patch `_APS.fits`, unlike the per-bin PPXF/EMI/RVS/FERRE fits above
# (those are already reproduced live from the final file). These are written
# once per *pointing* by aps_ifu_prepare.py, before per-patch/class splitting
# — so they live under the parent (pre-"_P####") headname, not this patch's.
# --------------------------------------------------------------------------- #

def _parent_headname(headname):
    return re.sub(r"_P\d+$", "", headname)


def _segmentation_map_figure(seg, white=None):
    if white is not None:
        fig = go.Figure(
            data=[go.Heatmap(z=white, colorscale="Greys", showscale=False, xaxis="x", yaxis="y"),
                  go.Heatmap(z=seg, colorscale="Turbo", showscale=True, xaxis="x2", yaxis="y2")],
        )
        fig.update_layout(
            grid={"rows": 1, "columns": 2, "pattern": "independent"},
            title="White-light image (left) / Segmentation map (right)",
            height=420, margin={"l": 40, "r": 20, "t": 40, "b": 30},
        )
    else:
        fig = go.Figure(data=[go.Heatmap(z=seg, colorscale="Turbo", showscale=True)])
        fig.update_layout(title="Segmentation map", height=420, margin={"l": 40, "r": 20, "t": 40, "b": 30})
    fig.update_yaxes(scaleanchor="x")
    return fig


def _png_gallery(figdir, patch_headname):
    if not Path(figdir).is_dir():
        return []
    suffixes = ("_prep_raw.png", "_prep_snrcut.png", "_prep_spatialbin.png", "_prep_voronoi.png",
                "_snr_stages.png", "_voronoi_overview.png", "_voronoi_cube.png")
    found = [Path(figdir) / (patch_headname + suf) for suf in suffixes
             if (Path(figdir) / (patch_headname + suf)).is_file()]
    found += sorted(Path(p) for p in glob.glob(str(Path(figdir) / (patch_headname + "_sbin_*.png"))))
    children = []
    for p in found:
        try:
            b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        except Exception:
            continue
        # One per row (block, full-width) rather than the previous
        # inline-block flow — at a fixed 480px max-width, several of these
        # used to sit side by side and each looked tiny (the Voronoi
        # overview plot in particular has a lot of fine detail); full-width
        # single-column reads far better for exactly that kind of image.
        children.append(html.Div([
            html.Div(str(p), style={"fontSize": "11px", "color": "var(--pyaps-ink-muted)", "fontFamily": "monospace"}),
            html.Img(src=f"data:image/png;base64,{b64}",
                     style={"maxWidth": "100%", "border": "1px solid #ddd", "borderRadius": "4px"}),
        ], style={"display": "block", "marginBottom": "18px"}))
    return children


def _prep_dir(outpath, aps_file):
    """The intermediate aps_ifu_prepare/aps_ifu_ExGal|Gal working directory
    (where segments/targets/white/figs_* live) is one level below the final
    merge outpath, named after the observation-block ID (e.g.
    ".../20230514/5011/" vs the merged file's ".../20230514/") — read
    straight from the merged file's own primary header (OBID_B/G/R,
    whichever arm is present) rather than guessing/globbing subdirectories.
    Falls back to `outpath` itself if no OBID is found or that subdirectory
    doesn't exist (e.g. a differently-organised run)."""
    try:
        header = fits.getheader(aps_file)
        for key in ("OBID_B", "OBID_G", "OBID_R"):
            obid = header.get(key)
            if obid not in (None, "None", ""):
                cand = Path(outpath) / str(obid)
                if cand.is_dir():
                    return cand
    except Exception:
        pass
    return Path(outpath)


def _source_path_line(path, locked=False):
    """Small monospace "found this at ..." line — added under every
    Processing-History artifact per explicit user request ("print where
    you found these files from"), since the intro sentence alone only
    ever named the *targets* file's directory, not each artifact's own.

    `locked` — same server-mode-with-a-fixed-dataset flag used throughout
    this app to keep real server paths out of a browser in that mode
    (see aps_explorer._is_locked()/aps_l1_preview._primary_header_tab_
    content's own docstring); this tab shows a real filesystem path in
    three places, this being one, so it needs the same treatment."""
    if locked:
        text = "🔒 Dataset fixed for this session — opened via a direct link."
    else:
        text = f"Source: {path}"
    return html.Div(text, style={
        "fontSize": "10px", "color": "var(--pyaps-ink-faint)", "fontFamily": "monospace",
        "marginBottom": "6px", "wordBreak": "break-all",
    })


def _processing_history_tab_content(locked=False):
    if not STATE.loaded() or not STATE.outpath or not STATE.headname:
        return html.Div("Processing-history location unknown for this dataset.")
    patch_headname = STATE.headname
    parent = _parent_headname(patch_headname)
    outpath = _prep_dir(STATE.outpath, STATE.data["aps_file"])
    # Real server directory — never shown as text when locked (server/
    # multi-session mode with a dataset fixed via a direct link); the
    # masked placeholder replaces every spot outpath would otherwise be
    # interpolated into a sentence below. Standalone (not locked) keeps
    # showing the real path, unchanged — explicit request: "for
    # standalone it is fine."
    outpath_display = ("🔒 dataset fixed for this session" if locked else outpath)

    children = [
        html.P(
            f"Generated from \"{parent}\" (in {outpath_display}) at the target-selection/segmentation/binning "
            f"stage (before per-patch splitting) — not reproducible from this patch's merged _APS.fits.",
            style={"fontStyle": "italic", "color": "var(--pyaps-ink-muted)"},
        ),
    ]

    targets_file = None
    for cand in (f"{parent}_targets_mod.fits", f"{parent}_targets.fits"):
        if (Path(outpath) / cand).is_file():
            targets_file = Path(outpath) / cand
            break
    if targets_file is not None:
        try:
            tab = Table.read(str(targets_file))
            rows = [{c: _json_safe(tab[c][i]) for c in tab.colnames} for i in range(len(tab))]
            children.append(html.H5(f"Detected targets ({targets_file.name})"))
            children.append(_source_path_line(targets_file, locked=locked))
            children.append(csv_export_row("ifu-processing-history-table", dash_table.DataTable(
                id="ifu-processing-history-table",
                data=rows, columns=[{"name": c, "id": c} for c in tab.colnames],
                sort_action="native", filter_action="native", page_size=10,
                **style.DATATABLE_KWARGS,
            )))
        except Exception as e:
            children.append(html.Div(f"Could not read {targets_file.name}: {e}"))
    else:
        children.append(html.Div(f"No target-detection table found for \"{parent}\" in {outpath_display}."))

    seg_file = Path(outpath) / f"{parent}_segments.fits"
    white_file = Path(outpath) / f"{parent}_white.fits"
    if seg_file.is_file():
        try:
            # FITS arrays are big-endian; Plotly's orjson serializer chokes
            # on non-native byte order ("numpy array is not native-endianness"),
            # so convert to native before handing off to go.Heatmap.
            seg = fits.getdata(str(seg_file))
            seg = seg.astype(seg.dtype.newbyteorder("="))
            white = None
            if white_file.is_file():
                white = fits.getdata(str(white_file))
                white = white.astype(white.dtype.newbyteorder("="))
            children.append(dcc.Graph(figure=_segmentation_map_figure(seg, white), config=_GRAPH_CONFIG))
        except Exception as e:
            children.append(html.Div(f"Could not read {seg_file.name}: {e}"))
    else:
        children.append(html.Div(f"No segmentation map found for \"{parent}\" in {outpath_display}."))

    gallery = (_png_gallery(Path(outpath) / "figs_ExGal", patch_headname)
               or _png_gallery(Path(outpath) / "figs_Gal", patch_headname))
    if gallery:
        children.append(html.H5(f"Target-selection / spatial-binning diagnostics ({patch_headname})"))
        children.append(html.Div(gallery))
    else:
        children.append(html.Div(f"No target-selection/binning diagnostic images found for \"{patch_headname}\"."))

    return html.Div(children)


def on_map_click(click_data):
    """Resolve a spaxel-map click to its BIN_ID — the map's customdata is
    `[index, x, y]` (see apsPlot.spaxel_map.spaxel_map_figure's own
    docstring for why x/y ride along too — the multi-panel L2 map
    feature's own cross-panel selection sync needs them client-side).
    `index` at [0] is the BIN_ID itself directly (both call sites now
    pass `st["table"]["BIN_ID"]` as `spaxel_map_figure`'s own `indices`
    — was a plain positional index into PATCH_TABLE, requiring a second
    lookup here; that mismatch is exactly what made the multi-panel L2
    map feature's own cross-panel selection restyle land on the wrong
    point, since that clientside code has no way to do this same lookup
    — see aps_explorer.py's own extra-map-panel figure builder for the
    full story)."""
    if not click_data or not STATE.loaded():
        return no_update
    customdata = click_data["points"][0].get("customdata")
    if customdata is None:
        return no_update
    idx = customdata[0] if isinstance(customdata, (list, tuple)) else customdata
    return int(idx)


# Same fixed-height/overflow-auto pattern aps_l1_preview.py/aps_MOSviewer.py
# already use for their own tab content — IFU never had it, and its
# "Processing History" tab (a target-detection table + a segmentation map
# + a whole gallery of full-width diagnostic PNGs, potentially many) could
# make the page arbitrarily tall, pushing the always-visible "Bin & Spaxel
# Data" panel below the tabs far out of view — reported directly ("all
# tables which are at the bottom goes to very low and out of sight"), with
# an explicit ask for consistent page structure regardless of which tab
# is open or how much content it has. Applied to every tab, not just
# History, so the tab-content area always occupies the same slot on the
# page — a plain 560px dcc.Graph sits comfortably inside a 650px box with
# a little headroom, no scrollbar needed; History's own content scrolls
# within it instead of growing the page.
_TAB_VIEWPORT_HEIGHT = "650px"


def _scrollable_tab(children):
    """`pyaps-scrollbox` (see its CSS rules in aps_explorer.py's
    `index_string`) replaces the OS's own auto-hide overlay scrollbar with
    an always-rendered one whenever this box's content genuinely overflows
    — explicit user report: on some platforms the default overlay
    scrollbar is invisible except while actively scrolling, easy to miss
    entirely."""
    return html.Div(
        children, className="pyaps-scrollbox",
        style={"height": _TAB_VIEWPORT_HEIGHT, "overflowY": "auto",
               "border": "1px solid #ddd", "borderRadius": "4px"},
    )


def _fit_graph(fig):
    """One of this tab's spectral-fit figures (Spectrum/Stellar/Emission/
    LS/RVS/FERRE) — a single `dcc.Graph`, always the sole occupant of its
    `_scrollable_tab`, so unlike aps_l1_preview's stacked Spectra/FWHM
    graphs there's no risk of one plot's header overlapping another's
    x-axis here. But a hardcoded "560px" style height had the *other* half
    of that same bug class: some of these figures compute a natural
    height taller than that (more panel-rows than the fixed value assumed
    fit), which would silently overflow past its own box uncontained.
    `_scrollable_tab`'s own fixed-height scrolling wrapper still keeps
    the tab's footprint on the page consistent regardless.

    `fill_container_width` clears the fixed pixel width these figures
    default to for their other life as static PNG pipeline-diagnostic
    exports (see its own docstring) — this app's own live view should
    fill its tab's actual width instead of leaving space empty beside a
    fixed-width plot.

    Leaving the Graph's own `style["height"]` unset does NOT reliably let
    it render at its own `figure.layout.height`, despite that being this
    function's own previous assumption — a real, confirmed Dash bug (see
    aps_MOSviewer._scrollable_graph's own docstring for the full
    live-diagnosed account, found reading `dcc.Graph`'s async-graph.js
    bundle directly): `responsive` defaulting to `'auto'` treats
    `fill_container_width`'s own `width=None` as reason enough to also
    silently discard the real `height`, replacing it with `height:100%`
    of whatever `_scrollable_tab`'s fixed-height wrapper happens to be —
    most visible on this app's own tallest fit figures (RVS/FERRE's
    2-arm-plus-model panels). Same already-proven fix as
    `aps_l1_preview._stacked_graph`: an explicit pixel `style["height"]`
    on the Graph component itself, matching the figure's own computed
    height, so the inner plot div's "responsive" 100% resolves against
    that real value instead."""
    fixed_fig = style.fill_container_width(fig)
    real_height = fixed_fig.layout.height or 320
    return dcc.Graph(figure=fixed_fig, config=_GRAPH_CONFIG,
                      style={"height": f"{real_height}px"})


def update_tab_content(tab, bin_id, version, locked=False):
    if not STATE.loaded() or bin_id is None:
        return html.Div("No bin selected.")
    st = STATE.data
    try:
        if tab == "spectrum":
            content = _fit_graph(_bin_spectrum_figure(bin_id))
        elif tab == "stellar":
            if not st["ppxf_available"]:
                content = html.Div("PPXF stellar kinematics not available for this dataset.")
            elif "ppxf_bestfit" not in st:
                content = html.Div("This file has no spectral-fit extension (GALAXY_SPEC) — "
                                    "parameter maps are still available, but not the fitted curve.")
            else:
                content = _fit_graph(_ppxf_fit_figure(bin_id))
        elif tab == "emission":
            if st["gandalf_results"] is None:
                content = html.Div("No emission-line results in this dataset.")
            elif "gandalf_bestfit" not in st:
                content = html.Div("This file has no spectral-fit extension (GALAXY_SPEC) — "
                                    "per-line parameter maps are still available, but not the fitted curve.")
            else:
                content = _fit_graph(_emission_fit_figure(bin_id))
        elif tab == "ls":
            if not st.get("ls_available"):
                content = html.Div("Line-strength indices not available for this dataset.")
            else:
                content = _fit_graph(_ls_spectrum_figure(bin_id))
        elif tab == "rvs":
            if not st.get("rvs_available"):
                content = html.Div("RVS stellar parameters not available for this dataset.")
            elif "rvs_model_rec" not in st:
                content = html.Div("This file has no spectral-fit extension (STAR_SPEC) — "
                                    "parameter maps are still available, but not the fitted curve.")
            else:
                content = _fit_graph(_rvs_fit_figure(bin_id))
        elif tab == "ferre":
            if not st.get("ferre_available"):
                content = html.Div("FERRE stellar parameters not available for this dataset.")
            elif "ferre_model_rec" not in st:
                content = html.Div("This file has no spectral-fit extension (STAR_SPEC) — "
                                    "parameter maps are still available, but not the fitted curve.")
            else:
                content = _fit_graph(_ferre_fit_figure(bin_id))
        elif tab == "history":
            content = _processing_history_tab_content(locked=locked)
        else:
            content = html.Div()
    except Exception as e:
        content = html.Div(f"Error rendering {tab}: {e}")
    return _scrollable_tab([content])


def update_value_tables(bin_id, version):
    if not STATE.loaded() or bin_id is None:
        return [], [], [], "Bin results"
    spaxel_col_defs, spaxel_rows = _spaxel_table_data(bin_id)
    bin_rows, bin_label = _bin_params_table_data(bin_id)
    return spaxel_rows, spaxel_col_defs, bin_rows, bin_label
