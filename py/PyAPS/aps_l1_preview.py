"""
aps_l1_preview.py - WEAVE L1 preview library (Dash figure/state layer)
=========================================================================

Library module (no standalone Dash app/CLI of its own — see
`aps_explorer.py`, the single interactive entry point for all L1/L2
viewing) for exploring and QC-ing WEAVE spectroscopic L1 data —
interactive fiber selection, spectral visualization, and FWHM
diagnostics. Built on PyAPS.apsPlot (Plotly) + Dash, replacing the
previous PyQt5/pyqtgraph desktop implementation; the underlying
data-loading (`APSOB`) logic is unchanged. Provides: `AppState`/`STATE`
(the loaded-dataset singleton), `_load_dataset`/`load_from_form_fields`
(loaders), figure builders, and the layout-piece/callback-logic
functions `aps_explorer.py` wires up into its own single Dash app.

FEATURES
--------
1. Interactive RA/DEC fiber map, colour-coded by integrated flux
2. Spectral visualization (flux + inverse variance, all arms)
3. FWHM overview (all-fiber "cloud" + global curve + selected-fiber
   highlight) and detailed per-fiber inspection (interpolated curve +
   raw arc-line measurements)
4. Primary FITS header (searchable table)

Aladin sky-view integration (clicking a fiber recenters an embedded
Aladin Lite panel and vice versa) lives in `aps_explorer.py` itself, not
here — this module no longer talks to Aladin/SAMP directly (removed;
the embedded Aladin Lite panel supersedes the old SAMP-based link to a
separate desktop Aladin application).
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.spatial import cKDTree as _cKDTree

from dash import dcc, html, dash_table, no_update

from werkzeug.local import LocalProxy

import PyAPS
from PyAPS import aps_constants
from PyAPS import aps_explorer_session as _sess
from PyAPS.apsPlot import style
from PyAPS.aps_utils import (
    APSOB,
    none_or_str,
    str2bool,
    l1_fileinfo,
    read_infiles_list,
    gen_targlist,
)
from PyAPS.apsPlot.fiber_map import fiber_map_figure
from PyAPS.apsPlot.fwhm import fwhm_overview_figure, fwhm_detail_figure
from PyAPS.apsPlot.spectra import spectrum_overlay_figure
from PyAPS.apsPlot.slit_explorer import slit_explorer_figure
from PyAPS.apsPlot.datatable_export import csv_export_row

APSVERS = PyAPS.__version__

#: How many of _primary_header_tab_content()'s per-input-file header
#: tables get a real, unique id/export button — real WEAVE L1 loads are
#: practically always 1-2 arms, so this is generous headroom, not a
#: tight bound. aps_explorer.py's own registration loop pre-wires
#: exactly this many export callbacks at app startup (callback
#: registration can't happen dynamically per-request), so a dataset
#: with more input files than this just doesn't get an export button on
#: its extra header tables' sections beyond this count — not a crash.
HEADER_TABLE_MAX_FILES = 10


# --------------------------------------------------------------------------- #
# Dataset loading — same APSOB-construction logic as legacy l1_preview(),
# now a plain function returning a state dict rather than driving Qt window
# construction directly.
# --------------------------------------------------------------------------- #

def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Interactively view the spectra from L1 files, v." + str(APSVERS),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--infiles", nargs="+", type=str, default=None, required=False,
                         help="The input stack/stackcube filename(s)")
    parser.add_argument("--infiles_list", type=none_or_str, default=None, required=False,
                         help="Optional file containing infile name per line")
    parser.add_argument("--l1_reference", type=none_or_str, default=None, required=False,
                         help="L1 reference catalog generated from L1 stack/stackcube files to visualise on Aladin")
    parser.add_argument("--l2_reference", type=none_or_str, default=None, required=False,
                         help="L2 reference catalog to show the L2 parameters")
    parser.add_argument("--aps_ids", type=none_or_str, default=None, required=False,
                         help="Return the spectra ONLY for these comma-separated source IDs")
    parser.add_argument("--targsrvy", type=none_or_str, default=None, required=False,
                         help="Return the spectra ONLY for these comma-separated Survey types")
    parser.add_argument("--targclass", type=none_or_str, default=None, required=False,
                         help="Return the spectra ONLY for these comma-separated Target classes")
    parser.add_argument("--mask_aps_ids", type=none_or_str, default=None, required=False,
                         help="Mask spectra for these comma-separated source IDs")
    parser.add_argument("--decimate_stride", type=int, default=None, required=False,
                         help="Load only 1 out of every N spaxels/fibres (explorer speed-up for very "
                              "large IFU/LIFU/MIFU cubes) — a uniform stride over the full candidate "
                              "list (after every other target filter above already applies), so the "
                              "on-sky spatial extent/coverage is unchanged, only the density. None/1 "
                              "(default) loads every spaxel, unchanged from before this option existed.")
    parser.add_argument("--area", type=none_or_str, default=None, required=False,
                         help="RA_CENT(deg),DEC_CENT(deg),A(arcsec),B(arcsec),ANGLE(deg CCW) — "
                              "see aps_utils.py's gen_targlist for the exact 5-value ellipse format")
    parser.add_argument("--mask_areas", nargs="+", type=none_or_str, default=None, required=False,
                         help="One or more regions, same 5-value format as --area: "
                              "RA_CENT(deg),DEC_CENT(deg),A(arcsec),B(arcsec),ANGLE(deg CCW)")
    parser.add_argument("--wlranges", nargs="+", type=none_or_str, default=None, required=False,
                         help="wlmin,wlmax covering range for each input file")
    parser.add_argument("--sens_corr", type=str2bool, default=True,
                         help="Apply sensitivity correction to the flux")
    parser.add_argument("--safe_mask_gaps", type=str2bool, default=False,
                         help="Ignore pixels around gap when applying masks")
    parser.add_argument("--mask_gaps", type=str2bool, default=False, help="Mask all pixels around the gap")
    parser.add_argument("--tellurics", type=str2bool, default=False, help="Mask STRONG telluric bands")
    parser.add_argument("--vacuum", type=str2bool, default=False, help="Report wavelengths in vacuum (air by default)")
    parser.add_argument("--fill_gap", type=str2bool, default=False, help="Fill the gap with NaNs in the final stacked spectra")
    parser.add_argument("--arms_ratio", type=none_or_str, default=None, required=False, help="arms flux scaling ratio")
    parser.add_argument("--join_arms", type=str2bool, default=False, help="Join the spectral arms into a SINGLE spectrum")
    parser.add_argument("--crr", type=str2bool, default=False, help="Run the cosmic-ray rejection")
    parser.add_argument("--catdir", type=none_or_str, default=None, required=False, help="Directory to keep WEAVE INPUT Catalogs")
    parser.add_argument("--caldir", type=none_or_str, default=None, required=False, help="Directory to keep WEAVE INPUT CALIBRATIONS")
    parser.add_argument("--configdir", type=none_or_str, default=None, required=False, help="Directory to keep WEAVE Config files")
    parser.add_argument("--collapse", type=str2bool, default=False, required=False,
                         help="Collapse the spectra within the area to a single spectrum")
    # Previously hardcoded/inaccessible APSOB() parameters — added so every
    # option APSOB.__init__ actually accepts is reachable from this form,
    # not just the subset that happened to be exposed first (per explicit
    # user request: "take a look at aps_utils and make sure all of them
    # are available"). Grouped under "Advanced processing options" in the
    # web form (_load_form below) rather than the main checklist, since
    # these are real but rarely-touched knobs.
    parser.add_argument("--skysub", type=str2bool, default=True,
                         help="Use the sky-subtracted extension (vs raw sky-included)")
    parser.add_argument("--split_arms", type=str2bool, default=False,
                         help="When joining arms, keep them split in the output (only relevant with --join_arms)")
    parser.add_argument("--mask_bad_overlap", type=str2bool, default=True,
                         help="Mask IVAR where joined arms show a significant flux-jump in their overlap (only relevant with --join_arms)")
    parser.add_argument("--use_resolution_deconvolution", type=str2bool, default=True,
                         help="Mask edge pixels affected by resolution-matrix deconvolution")
    parser.add_argument("--edge_pixels_to_mask", type=int, default=5,
                         help="Number of edge pixels to mask per side when --use_resolution_deconvolution is on")
    parser.add_argument("--template_sigma0_angstrom", type=float, default=0.5,
                         help="Template instrumental resolution sigma, in Angstrom, for resolution deconvolution")
    parser.add_argument("--rereplace_binned_cal", type=str2bool, default=True,
                         help="Fall back to unbinned calibration files if the binned ones aren't found")
    parser.add_argument("--lsftype", type=none_or_str, default="LSF", choices=["LSF", "FWHM"],
                         help="LSF source: LSF (pre-computed B-spline lsf_ files) or FWHM (fit from wave_ arc-line files)")
    parser.add_argument("--normalize_ivar", type=str2bool, default=False,
                         help="Normalize IVAR across arms so each contributes proportionally to chi-square")
    parser.add_argument("--ivar_normalization_mode", type=none_or_str, default="balanced",
                         choices=["balanced", "pixels", "hybrid"],
                         help="IVAR normalization mode when --normalize_ivar is on")
    parser.add_argument("--skysub_mask_residuals", type=str2bool, default=False,
                         help="Mask sky residuals in the OH airglow forest region")
    parser.add_argument("--spaxel_weighted_lsf", type=str2bool, default=False,
                         help="IFU/LIFU/MIFU cubes only: per-spaxel LSF/FWHM weighted from the "
                              "underlying dithered single-exposure fibres' own real per-fibre "
                              "LSF/FWHM curves (geometric circle-overlap weighting), instead of "
                              "the flat all-fibre global value every spaxel gets by default. "
                              "See aps_ifu_spaxel_contrib.py. Off by default -- a spaxel with no "
                              "contributing fibre found still falls back to the global value even "
                              "when this is on.")
    parser.add_argument("--extinction_corr", type=str2bool, default=False,
                         help="Opt-in Galactic (SFD98+Fitzpatrick99) dust extinction correction, "
                              "per-target from its own sky position, full strength "
                              "(extinction_ebv_scale=1.0) -- see doc/aps_rr.md's Galactic Extinction "
                              "Correction section. Off by default, same as the pipeline itself.")
    parser.add_argument("--offset_gap_pix", type=int, default=aps_constants.offset_gap_pix,
                         help="Gap-masking dilation width, in pixels")
    parser.add_argument("--funit", type=float, default=aps_constants.funit,
                         help="Flux unit scaling factor")
    parser.add_argument("--port", type=int, default=8060, required=False, help="Dash server port")
    parser.add_argument("--debug", action="store_true", default=False, help="Run the Dash server in debug mode")
    return parser


def _load_dataset(args):
    """Build the target list + APSOB(s) + reference catalogs for one run
    of CLI/form options. Same logic as legacy `l1_preview()`'s loading
    loop — returns a plain dict instead of constructing GUI windows.
    """
    if args.infiles is None and args.infiles_list is None:
        raise ValueError("Either --infiles or --infiles_list must be specified")

    if args.infiles_list is not None:
        infiles_list = read_infiles_list(args.infiles_list)
    else:
        infiles_list = [args.infiles]

    targ_list_master = []
    apsob_master = None

    for c_infiles, i_infiles in enumerate(infiles_list):
        print(f"[{c_infiles + 1}/{len(infiles_list)}] Preparing targets for {i_infiles}")
        wlranges = None
        if args.wlranges is not None and args.wlranges[0] is not None:
            wlranges = [[float(x) for x in args.wlranges[i].split(",")] for i in range(len(i_infiles))]

        arms_ratio = None
        if args.arms_ratio is not None:
            arms_ratio = [float(x) for x in args.arms_ratio.split(",")]
            assert len(arms_ratio) == len(i_infiles), "length of arms_ratio(s) and infiles must be identical"

        i_infiles_info = l1_fileinfo(i_infiles, wlranges=wlranges, arms_ratio=arms_ratio)
        i_infiles = i_infiles_info["infiles"]
        wlranges = i_infiles_info["wlranges"]
        arms_ratio = i_infiles_info["arms_ratio"]

        join_arms = args.join_arms
        if len(i_infiles) < 2:
            join_arms = False

        aps_ids = [int(x) for x in args.aps_ids.split(",")] if args.aps_ids else None
        targsrvy = [str(x) for x in args.targsrvy.split(",")] if args.targsrvy else None
        targclass = [str(x) for x in args.targclass.split(",")] if args.targclass else None
        mask_aps_ids = [int(x) for x in args.mask_aps_ids.split(",")] if args.mask_aps_ids else None

        area = [float(x) for x in args.area.split(",")] if args.area else None
        mask_areas = None
        if args.mask_areas is not None and args.mask_areas[0] is not None:
            mask_areas = [[float(x) for x in a.split(",")] for a in args.mask_areas]

        # Explicit request: "loading all is unnecessary... is there a
        # smart way that it only load for example one out of each 3 or 4
        # or whatever and use that one for explorer only? ... however I
        # do not want that change the Spatial." Resolves the *same*
        # fully-filtered candidate aps_id list `APSOB` itself would use
        # (via the identical `gen_targlist` call, same filters/aps_id_sum)
        # cheaply up front — gen_targlist is headers/table-only, no flux
        # data read at all, so calling it a second time here (APSOB calls
        # it again internally) costs nothing meaningful even on a large
        # cube — then keeps a uniform stride through that list, not the
        # first N or some cropped subset, so the on-sky spatial extent is
        # unchanged (still spans the full field, just less densely
        # sampled); only *density* changes, per the explicit requirement.
        decimate_stride = getattr(args, "decimate_stride", None)
        if decimate_stride and decimate_stride > 1:
            full_candidate_ids, _, _, _, _ = gen_targlist(
                i_infiles[0], i_infiles_info["mode"],
                aps_ids=aps_ids, targsrvy=targsrvy, targclass=targclass,
                mask_aps_ids=mask_aps_ids, area=area, mask_areas=mask_areas,
                aps_id_sum=int(100007 * c_infiles),
            )
            n_before = len(full_candidate_ids)
            aps_ids = full_candidate_ids[::decimate_stride].tolist()
            print(f">>> Decimating: loading 1 of every {decimate_stride} spaxels/fibres "
                  f"({len(aps_ids)} of {n_before}) — spatial extent unchanged, density reduced.")

        targs_i = APSOB(
            i_infiles, aps_ids=aps_ids, targsrvy=targsrvy, targclass=targclass,
            mask_aps_ids=mask_aps_ids, area=area, mask_areas=mask_areas, wlranges=wlranges,
            sens_corr=args.sens_corr, mask_gaps=args.mask_gaps, safe_mask_gaps=args.safe_mask_gaps,
            vacuum=args.vacuum, tellurics=args.tellurics, fill_gap=args.fill_gap,
            arms_ratio=arms_ratio, join_arms=join_arms,
            funit=getattr(args, "funit", None) or aps_constants.funit,
            offset_gap_pix=getattr(args, "offset_gap_pix", None) or aps_constants.offset_gap_pix,
            collapse=args.collapse, crr=args.crr, aps_id_sum=int(100007 * c_infiles),
            catdir=args.catdir, caldir=args.caldir, configdir=args.configdir,
            skysub=getattr(args, "skysub", True),
            split_arms=getattr(args, "split_arms", False),
            mask_bad_overlap=getattr(args, "mask_bad_overlap", True),
            use_resolution_deconvolution=getattr(args, "use_resolution_deconvolution", True),
            edge_pixels_to_mask=getattr(args, "edge_pixels_to_mask", None) or 5,
            template_sigma0_angstrom=getattr(args, "template_sigma0_angstrom", None) or 0.5,
            rereplace_binned_cal=getattr(args, "rereplace_binned_cal", True),
            lsftype=getattr(args, "lsftype", None) or "LSF",
            normalize_ivar=getattr(args, "normalize_ivar", False),
            ivar_normalization_mode=getattr(args, "ivar_normalization_mode", None) or "balanced",
            skysub_mask_residuals=getattr(args, "skysub_mask_residuals", False),
            spaxel_weighted_lsf=getattr(args, "spaxel_weighted_lsf", False),
            extinction_corr=getattr(args, "extinction_corr", False),
        )
        targ_list_master.extend(targs_i._targetlist)
        if apsob_master is None:
            apsob_master = targs_i

    if len(targ_list_master) == 0:
        raise ValueError("No target passed the filter to preview")

    l2_ref_table = Table.read(args.l2_reference, format="fits") if args.l2_reference else None
    l1_ref = None
    if args.l1_reference:
        l1_ref = {"data": Table.read(args.l1_reference, format="fits"), "path": args.l1_reference}

    return dict(
        targs=targ_list_master, apsob=apsob_master,
        l1_reference=l1_ref, l2_reference=l2_ref_table,
    )


# --------------------------------------------------------------------------- #
# Provenance: a stacked/superstacked/cube L1 file's own PROV#### header
# cards record exactly which file(s) it was formed from (WEAVE's own
# convention, confirmed directly on real data — PROV0000 is the output
# file's own name, PROV0001.. list each contributing file by its bare
# filename, sitting in the same directory as the file that references
# it). Used for the optional "show each contributing exposure's own
# fibre positions" Aladin overlay (aps_explorer._contrib_exposures_payload)
# — explicit user request: "I need to see the coordinates of each
# individual single file[s] contributing on the coordinate map... you
# can find it in the provenance properties."
# --------------------------------------------------------------------------- #

# Provenance-chain walk + per-file FIBTABLE read used to live here as
# private functions; both are now shared with aps_utils.py's opt-in
# spaxel-weighted LSF/FWHM feature (which needs the exact same PROV####
# walk to find an IFU cube's underlying single-exposure files), so they
# were extracted to aps_ifu_spaxel_contrib.py rather than kept as two
# copies. Aliased back to their original private names here so every
# existing call site below is unchanged.
from PyAPS.aps_ifu_spaxel_contrib import (
    resolve_single_exposure_files as _resolve_single_exposure_files,
    read_fibtable_positions as _read_fibtable_positions_full,
)


def _read_fibtable_positions(path):
    """RA/Dec/status/targuse straight from one file's own FIBTABLE
    extension (see `aps_ifu_spaxel_contrib.read_fibtable_positions` for
    the full docstring) — this wrapper just drops the NSPEC column that
    function also returns, since nothing in this module needs it."""
    _nspec, ra, dec, status, targuse = _read_fibtable_positions_full(path)
    return ra, dec, status, targuse


# --------------------------------------------------------------------------- #
# App state (single-user local tool — see plan) + figure/table builders
# --------------------------------------------------------------------------- #

class AppState:
    def __init__(self):
        self.targs = None
        self.apsob = None
        self.targs_apstoid = {}
        self.l1_reference = None
        self.l2_reference = None
        self.coord_arr = None
        self.aps_id_arr = None
        self.fwhm_cache = None  # {"arm_names", "wave_grids", "fiber_fwhm", "global_fwhm"}
        self._fwhm_figure_cache = {}      # see _build_fwhm_figures()
        self._fwhm_figure_cache_key = None  # self._data_generation it was computed for
        self.load_args = None  # the argparse Namespace this dataset was loaded with
        self.color_vmin = None  # None = auto (fiber_map_figure's own default range)
        self.color_vmax = None
        self.color_by = "flux"  # see COLOR_BY_OPTIONS / _color_by_values
        self.color_scale = "linear"  # see apsPlot.style.COLOR_SCALE_OPTIONS
        self._contrib_cache = None      # see contributing_fibre_positions()
        self._contrib_cache_key = None  # self._data_generation it was computed for
        self._color_by_cache = {}       # see _color_by_values()
        self._color_by_cache_key = None  # self._data_generation it was computed for
        # Bumped once, as the very *last* thing load() does (after
        # self.targs/coord_arr/aps_id_arr/fwhm_cache are all fully set) --
        # see load()'s own comment for why this exists instead of the
        # id(self.load_args)-based cache-key pattern the three caches
        # above used to use.
        self._data_generation = 0

    def loaded(self):
        return self.targs is not None

    def is_fibre_level(self):
        """Whether the loaded data has a genuine, physically-meaningful
        NSPEC (slit position) per target — true for single-exposure MOS/
        MOSLIFU/MOSMIFU L1 files (one row per real fibre, straight from
        FIBTABLE), false for a stacked/co-added IFU cube (plain LIFU/MIFU
        — aps_utils.gen_targlist's IFU/LIFU/MIFU branch fakes
        NSPEC=APS_ID for those since each spatial position there is built
        from potentially many different fibres across dithered exposures,
        not one). Drives whether the Slit Explorer panel has anything
        meaningful to show — see aps_explorer._slit_explorer_container."""
        if not self.loaded():
            return False
        mode = self.targs[0].meta[0].get("mode") if self.targs[0].meta else None
        return str(mode).upper() in ("MOS", "MOSLIFU", "MOSMIFU")

    def contributing_fibre_positions(self):
        """Per-contributing-single-exposure-file fibre positions for a
        stacked/superstacked/cube L1 dataset — explicit user request: "I
        need to see the coordinates of each individual single file[s]
        contributing on the coordinate map." Only meaningful (and only
        ever computed) for the non-fibre-level case — the whole point is
        seeing the individual exposures a *stack* was built from; a
        genuine single-exposure fibre-level load has nothing to resolve
        here (its own fibre positions already *are* what's shown).

        Lazy and memoized (computed the first time this is called after a
        load, cached until the next one) rather than eagerly computed
        inside `load()` itself — even though a FIBTABLE-only read is
        cheap per file, a real stack can combine over a dozen exposures,
        so this still opens that many files; doing it only if/when the
        feature is actually used (the map's "Show contributing exposures"
        checkbox) means a user who never asks for it never pays any of
        this cost at all.

        Fibre *positions* are only ever read from the *first* loaded
        arm's own provenance chain — explicit request: "no need to load
        the fibre position for each arm" (positions don't depend on which
        arm recorded the light; only the wavelength/flux data differs).
        Every *other* loaded arm's own provenance chain is still resolved
        (headers-only — no FIBTABLE read, just walking each one's own
        PROV#### cards, same as `_resolve_single_exposure_files` already
        does) purely to recover its matching exposure's *filename*, so it
        can be shown alongside the primary arm's name — explicit request:
        "not bad that you write the name of both single blue and red next
        to the [checkbox]". Pairing is by *position* in each arm's own
        chain (their own PROV0001/PROV0002/... order) — the same exposure
        sequence recorded simultaneously by every arm, so index N in one
        arm's chain and index N in another's are the same real exposure
        epoch; if a chain comes back a different length than the primary
        one (shouldn't happen for two arms of the same OB, but not
        assumed), pairing just stops at the shorter of the two rather
        than guessing.

        Also resolves, for each contributing fibre, which row of the
        already-loaded *stacked* target list (`self.targs`/
        `self.coord_arr`) sits nearest to it on the sky (a `scipy`
        `cKDTree` query, not a per-point Python loop) — explicit request:
        "I want the color code [to] come again from the total flux or
        S/N or whatever the main plot is... do not color code them based
        on the single files". This is what lets the caller colour each
        contributing-exposure point by the *stacked* target's own flux/
        S/N value (whatever `COLOR_BY_OPTIONS`/`color_by` currently is)
        without re-reading any spectra from the single-exposure files
        themselves — the nearest final stacked position is treated as
        "the same real target," which is exact for the common case (one
        contributing fibre per final position) and a reasonable nearest-
        neighbour approximation wherever the stack genuinely resampled
        several dithered fibres into one output position.

        Returns `None` if not applicable (fibre-level data, nothing
        loaded, or no PROV#### provenance resolves to anything) or
        `{"arm_used": <basename>, "n_arms_loaded": int, "files": [
        {"file": <basename>, "camera": str|None, "other_files":
        [{"file": <basename>, "camera": str|None}, ...], "ra": array,
        "dec": array, "status": array, "targuse": array, "nearest_idx":
        int array (index into self.targs/self.coord_arr)} or
        {"file": <basename>, "error": str}, ...]}`.
        """
        if not self.loaded() or self.is_fibre_level():
            return None
        cache_key = self._data_generation
        if self._contrib_cache_key == cache_key:
            return self._contrib_cache
        self._contrib_cache_key = cache_key
        self._contrib_cache = None

        infiles = list(getattr(self.load_args, "infiles", None) or [])
        if not infiles:
            return None
        arm_used = infiles[0]
        single_paths = _resolve_single_exposure_files(arm_used)
        if not single_paths:
            return None

        # Camera ("BLUE"/"RED"/...) per loaded arm file, purely cosmetic
        # (for the "single blue and red" label) — l1_fileinfo is a cheap,
        # headers-only inspector already used throughout this module's
        # own loader, not a new/expensive dependency. Called once *per
        # file*, deliberately not once with the whole `infiles` list —
        # l1_fileinfo's own docstring says it reorders its result to a
        # canonical "blue first, then red" order regardless of input
        # order, confirmed directly against real data (feeding it
        # [RED-arm-file, BLUE-arm-file] returned camera=['BLUE','RED'],
        # silently mismatched against the original positional order) —
        # a one-file-at-a-time call has nothing to reorder, so each
        # result is unambiguously that one file's own camera.
        cameras = []
        for f in infiles:
            try:
                cameras.append(l1_fileinfo([f]).get("camera", [None])[0])
            except Exception:
                cameras.append(None)

        # Other loaded arms' own provenance chains, resolved purely for
        # their filenames (never their FIBTABLEs) — paired by position
        # with the primary chain resolved above.
        other_arm_chains = []
        for other_path in infiles[1:]:
            try:
                other_arm_chains.append(_resolve_single_exposure_files(other_path))
            except Exception:
                other_arm_chains.append([])

        # Nearest already-loaded stacked-target row per contributing
        # fibre, for colouring by the main plot's own current colour-by
        # value later (see this method's own docstring) — built once
        # here (not per-file) since it's the same tree for every file.
        tree = None
        if self.coord_arr is not None and len(self.coord_arr):
            finite_stack = np.isfinite(self.coord_arr[:, 0]) & np.isfinite(self.coord_arr[:, 1])
            stack_idx_map = np.flatnonzero(finite_stack)
            if stack_idx_map.size:
                dec0 = float(np.nanmean(self.coord_arr[finite_stack, 1]))
                cos_dec = np.cos(np.radians(dec0)) or 1.0
                stack_xy = np.column_stack([
                    self.coord_arr[finite_stack, 0] * cos_dec, self.coord_arr[finite_stack, 1],
                ])
                tree = _cKDTree(stack_xy)

        files = []
        for i, path in enumerate(single_paths):
            try:
                ra, dec, status, targuse = _read_fibtable_positions(path)
            except Exception as e:
                files.append({"file": os.path.basename(path), "error": str(e)})
                continue

            other_files = []
            for arm_i, chain in enumerate(other_arm_chains):
                if i < len(chain):
                    other_files.append({
                        "file": os.path.basename(chain[i]),
                        "camera": cameras[arm_i + 1] if arm_i + 1 < len(cameras) else None,
                    })

            nearest_idx = np.full(len(ra), -1, dtype=int)
            if tree is not None:
                finite = np.isfinite(ra) & np.isfinite(dec)
                if finite.any():
                    xy = np.column_stack([ra[finite] * cos_dec, dec[finite]])
                    _, nn = tree.query(xy)
                    nearest_idx[finite] = stack_idx_map[nn]

            files.append({
                "file": os.path.basename(path), "camera": cameras[0] if cameras else None,
                "other_files": other_files,
                "ra": ra, "dec": dec, "status": status, "targuse": targuse,
                "nearest_idx": nearest_idx,
            })
        result = {
            "arm_used": os.path.basename(arm_used),
            "n_arms_loaded": len(infiles),
            "files": files,
        }
        self._contrib_cache = result
        return result

    def load(self, args):
        self.load_args = args
        self.color_vmin = None
        self.color_vmax = None
        self.color_by = "flux"
        self.color_scale = "linear"
        data = _load_dataset(args)
        self.targs = data["targs"]
        self.apsob = data["apsob"]
        self.targs_apstoid = {t.aps_id: i for i, t in enumerate(self.targs)}
        self.l1_reference = data["l1_reference"]
        self.l2_reference = data["l2_reference"]

        self.coord_arr = np.array([(t.targra, t.targdec) for t in self.targs], dtype=np.float64)
        self.aps_id_arr = np.array([t.aps_id for t in self.targs], dtype=int)

        self._build_fwhm_cache()

        # Real, reproduced-in-production bug this fixes: this app's
        # session state is per-session (see STATE's own LocalProxy), but
        # nothing serializes concurrent requests *within* one session --
        # gunicorn's own --threads config (see wsgi.py) means two
        # callbacks for the same session can genuinely run on different
        # threads at the same moment, and this load() call alone can take
        # 30+ seconds on a real large cube (see the DOF/masking timing
        # this app's own log already prints). The three caches below used
        # to invalidate on `id(self.load_args)` changing -- but
        # self.load_args was set at the very *top* of this method, well
        # before self.targs/coord_arr/aps_id_arr were actually updated a
        # dozen-plus seconds later, so a concurrent read landing in that
        # window (a real crash, live: "IndexError: index 31920 is out of
        # bounds for axis 0 with size 31920" in _bucket_by_color) saw the
        # *new* load_args (already updated) but still-*old* self.targs
        # (not updated yet) -- computed and cached a values array sized
        # for the wrong dataset, keyed under the new load's own cache
        # key, so a later *correct* read paired that stale cached array
        # with the genuinely-new coord_arr/aps_id_arr and crashed on the
        # size mismatch. Bumping this generation counter here, as the
        # very last statement of load() (strictly after every field the
        # three caches actually depend on has already been fully
        # reassigned), and having those caches key off *this* instead of
        # id(self.load_args), closes the window: a concurrent read during
        # the slow part of a reload now consistently sees either the
        # complete old (self.targs, self._data_generation) pair or the
        # complete new one, never a torn mix of the two. Each cache's own
        # `if <key> != <cache_key>: <clear cache>` line (right where each
        # one reads self._data_generation) is what actually re-derives a
        # fresh value once the generation *has* moved on -- no separate
        # early reset needed here any more.
        self._data_generation += 1

    def load_summary(self, locked=False):
        """Every option actually used for the current load, as
        {"Parameter", "Value"} rows — for the always-visible dataset-info
        panel, rendered as a real (fixed-height, scrollable) table by
        aps_explorer.py rather than a wall of text, since previously the
        only feedback was the plot itself, with no way to tell which
        file(s)/options produced it. Returns [] if nothing is loaded.

        `locked` — same server-mode-with-a-fixed-dataset flag `_load_form`
        already takes (see that function's own docstring) — masks Files/
        caldir/catdir/configdir here too, matching that form's own
        already-established wording exactly. This is a genuinely separate
        display (the always-visible summary table sits above the load
        form, built from a copy of the same `self.load_args` rather than
        going through `_load_form` at all), so it needed the identical
        treatment applied a second time here — explicit report that the
        real server-side paths were still showing up here even after
        `_load_form`'s own fields were masked."""
        a = self.load_args
        if a is None:
            return []

        def yn(attr, default=False):
            return "Yes" if getattr(a, attr, default) else "No"

        if locked:
            files_value = "🔒 Dataset fixed for this session — opened via a direct link."
            dir_value = "(using this server's default)"
        else:
            infiles = a.infiles if a.infiles else (f"(from --infiles_list {a.infiles_list})" if a.infiles_list else None)
            files_value = ", ".join(infiles) if infiles else "(none)"
            dir_value = None  # computed per-row below instead

        rows = [
            {"Parameter": "Files", "Value": files_value},
            {"Parameter": "Mode", "Value": "RAW (unmodified)" if getattr(a, "raw_mode", False) else "Processed"},
            {"Parameter": "caldir", "Value": dir_value or a.caldir or "(not set)"},
            {"Parameter": "catdir", "Value": dir_value or a.catdir or "(not set)"},
            {"Parameter": "configdir", "Value": dir_value or a.configdir or "(not set)"},
            {"Parameter": "L1 reference", "Value": a.l1_reference or "(none)"},
            {"Parameter": "L2 reference", "Value": a.l2_reference or "(none)"},
            {"Parameter": "APS IDs", "Value": a.aps_ids or "(all)"},
            {"Parameter": "Target survey", "Value": a.targsrvy or "(all)"},
            {"Parameter": "Target class", "Value": a.targclass or "(all)"},
            {"Parameter": "Mask APS IDs", "Value": a.mask_aps_ids or "(none)"},
            {"Parameter": "Decimation (1 of every N)",
             "Value": f"1 of every {a.decimate_stride}" if getattr(a, "decimate_stride", None) else "(none — every spaxel)"},
            {"Parameter": "Area", "Value": a.area or "(whole field)"},
            {"Parameter": "Mask areas", "Value": ", ".join(a.mask_areas) if a.mask_areas else "(none)"},
            {"Parameter": "Wavelength ranges", "Value": ", ".join(a.wlranges) if a.wlranges else "(default)"},
            {"Parameter": "Arms ratio", "Value": a.arms_ratio or "(default)"},
            {"Parameter": "Sensitivity correction", "Value": yn("sens_corr")},
            {"Parameter": "Join arms", "Value": yn("join_arms")},
            {"Parameter": "Vacuum wavelengths", "Value": yn("vacuum")},
            {"Parameter": "Mask gaps", "Value": yn("mask_gaps")},
            {"Parameter": "Safe mask gaps", "Value": yn("safe_mask_gaps")},
            {"Parameter": "Tellurics", "Value": yn("tellurics")},
            {"Parameter": "Fill gap", "Value": yn("fill_gap")},
            {"Parameter": "Cosmic-ray rejection", "Value": yn("crr")},
            {"Parameter": "Collapse to single spectrum", "Value": yn("collapse")},
            {"Parameter": "Sky subtraction", "Value": yn("skysub", True)},
            {"Parameter": "Split arms", "Value": yn("split_arms")},
            {"Parameter": "Mask bad overlap", "Value": yn("mask_bad_overlap", True)},
            {"Parameter": "Resolution-deconvolution masking", "Value": yn("use_resolution_deconvolution", True)},
            {"Parameter": "Edge pixels masked", "Value": str(getattr(a, "edge_pixels_to_mask", 5))},
            {"Parameter": "Re-replace binned cal files", "Value": yn("rereplace_binned_cal", True)},
            {"Parameter": "LSF source", "Value": getattr(a, "lsftype", None) or "LSF"},
            {"Parameter": "Normalize IVAR across arms", "Value": yn("normalize_ivar", False)},
            {"Parameter": "IVAR normalization mode", "Value": getattr(a, "ivar_normalization_mode", None) or "balanced"},
            {"Parameter": "Mask sky residuals (OH forest)", "Value": yn("skysub_mask_residuals")},
            {"Parameter": "Gap-masking width (px)", "Value": str(getattr(a, "offset_gap_pix", None) or aps_constants.offset_gap_pix)},
            {"Parameter": "Flux unit scale", "Value": str(getattr(a, "funit", None) or aps_constants.funit)},
            {"Parameter": "Fibres loaded", "Value": str(len(self.targs)) if self.targs else "0"},
        ]
        return rows

    def _build_fwhm_cache(self):
        """Precompute FWHM curves once at load time — same wavelength
        grids/interpolator-evaluation approach as legacy's
        plot_fwhm_all_fibers, done once instead of per-window-open."""
        self.fwhm_cache = None
        if self.apsob is None:
            return
        try:
            n_setups = len(self.targs[0].spectra)
            global_funcs = self.apsob.get_fwhm(aps_id=None, fwhm_key="gfwhm")
            if global_funcs is None or not all(f is not None for f in global_funcs):
                return
        except Exception:
            return

        is_joined = getattr(self.apsob, "join_arms", False)
        if is_joined and n_setups == 1:
            arm_names = ["Joined Arms"]
        else:
            setups = getattr(self.apsob, "_setups", None)
            arm_names = [setups[i] if setups and i < len(setups) else f"Setup {i + 1}"
                         for i in range(n_setups)]

        wave_grids = []
        for setup_idx in range(n_setups):
            wmin = min(t.spectra[setup_idx].wave.min() for t in self.targs)
            wmax = max(t.spectra[setup_idx].wave.max() for t in self.targs)
            wave_grids.append(np.linspace(wmin, wmax, 1000))

        global_fwhm = []
        for setup_idx in range(n_setups):
            try:
                global_fwhm.append(global_funcs[setup_idx](wave_grids[setup_idx]))
            except Exception:
                global_fwhm.append(None)

        fiber_fwhm = {}
        for t in self.targs:
            try:
                funcs = self.apsob.get_fwhm(aps_id=t.aps_id, fwhm_key="fwhm")
            except Exception:
                continue
            if funcs is None:
                continue
            vals = []
            for setup_idx in range(n_setups):
                if setup_idx < len(funcs) and funcs[setup_idx] is not None:
                    try:
                        vals.append(funcs[setup_idx](wave_grids[setup_idx]))
                    except Exception:
                        vals.append(None)
                else:
                    vals.append(None)
            fiber_fwhm[t.aps_id] = vals

        mode_label = "UNKNOWN"
        try:
            gfwhm_meta = self.targs[0].meta[0].get("gfwhm")
            if isinstance(gfwhm_meta, dict):
                mode_label = ("FAST (all fibres use global)" if gfwhm_meta.get("uses_global_fit")
                              else "INDIVIDUAL (per-fibre interpolation)")
        except Exception:
            pass

        self.fwhm_cache = dict(arm_names=arm_names, wave_grids=wave_grids,
                                fiber_fwhm=fiber_fwhm, global_fwhm=global_fwhm, mode_label=mode_label)

    def raw_fwhm_points(self, aps_id):
        """Per-arm raw arc-line measurement dicts for the detail view —
        same `meta[setup]['fwhm']['fiber_file_fits']` lookup as legacy."""
        idx = self.targs_apstoid.get(aps_id)
        if idx is None or self.fwhm_cache is None:
            return []
        target = self.targs[idx]
        n_arms = len(self.fwhm_cache["arm_names"])
        out = []
        for setup_idx in range(n_arms):
            pt = None
            try:
                meta = target.meta[setup_idx]
                fwhm_meta = meta.get("fwhm") if isinstance(meta, dict) else None
                fits_by_file = fwhm_meta.get("fiber_file_fits") if isinstance(fwhm_meta, dict) else None
                if isinstance(fits_by_file, dict):
                    for file_data in fits_by_file.values():
                        if isinstance(file_data, dict) and file_data.get("wavelengths") is not None:
                            pt = file_data
                            break
            except Exception:
                pt = None
            out.append(pt)
        return out

    def primary_headers(self):
        """One entry per input file this dataset was loaded from:
        `{"file": path, "cards": [(keyword, value, comment), ...]}` (or
        `{"file": path, "error": str(e)}` if that particular file
        couldn't be opened). Headers-only read — `fits.open(...,
        memmap=True, lazy_load_hdus=True)` never touches `.data`, and only
        the primary HDU's header is ever accessed — same pattern
        `aps_utils.aps_file_info()` already uses for exactly this reason,
        so this stays cheap and fast regardless of the file's actual data
        size. Deliberately independent of `self.apsob`/`self.targs` (which
        only exist after `APSOB`'s full processing pipeline has already
        transformed the data) — this reads the *original* file straight
        off disk, so it works the same way regardless of what processing
        options were used, and needs nothing extra retained from the load
        itself beyond the file paths already in `self.load_args`.

        Uses `header.cards` (each individual card in on-disk order), not
        `header.keys()`/dict-style access — a FITS header can have
        repeated keywords (`COMMENT`/`HISTORY`/blank), which dict-style
        access collapses into one combined pseudo-value; iterating cards
        instead shows every line exactly as it exists in the file."""
        if self.load_args is None:
            return []
        a = self.load_args
        # `--infiles` is the common interactive-viewer path (a flat list
        # of 1+ arm files for one pointing); `--infiles_list` is the
        # alternative batch-load path (a text file, one row per group of
        # arm files, each row comma-separated) that `_load_dataset` reads
        # via `read_infiles_list` — flatten every row's files together so
        # either path ends up with the same simple flat list here.
        if getattr(a, "infiles", None):
            infiles = list(a.infiles)
        elif getattr(a, "infiles_list", None):
            infiles = [f for group in read_infiles_list(a.infiles_list) for f in group]
        else:
            infiles = []
        out = []
        for path in infiles:
            try:
                with fits.open(path, memmap=True, lazy_load_hdus=True) as hdul:
                    header = hdul[0].header
                    cards = [(c.keyword, c.value, c.comment) for c in header.cards]
                out.append({"file": path, "cards": cards})
            except Exception as e:
                out.append({"file": path, "error": str(e)})
        return out


# What the Aladin catalog overlay (and its legend/colour-range control, both
# in aps_explorer.py) can colour fibres by — "flux" was the only option
# before ("at the moment the colour... is colour-coded by [flux]... can we
# have more options?"); "snr" reuses the same per-arm SNR aps_utils.py
# already computes into target.meta[i]["SNR"] (aps_utils.py:4774) rather
# than recomputing it. label -> Aladin legend/dropdown text, colorscale ->
# Plotly colorscale name, default_vmin -> matches this option's own sane
# floor (0 for S/N; the historical 12.0 floor for flux, unchanged).
COLOR_BY_OPTIONS = {
    "flux": ("Total flux (all arms)", "Jet", 12.0),
    "snr": ("S/N (mean over arms)", "Plasma", 0.0),
}


def _color_by_values(st, color_by):
    """Per-target array for whichever COLOR_BY_OPTIONS key is active, in
    st.targs order (same order as st.aps_id_arr/st.coord_arr) — shared by
    aps_explorer.py's _aladin_catalog_points and _aladin_legend_info so the
    plotted colours and the legend can never disagree.

    Cached per `color_by` value, invalidated only on a fresh load() (same
    `self._data_generation`-keyed pattern already used by
    contributing_fibre_positions() just above, for the same reason — see
    `AppState.load()`'s own comment on that field for a real concurrent-
    access bug this specific pattern was changed to fix): this
    function's actual result depends only on st.targs, which is fixed for
    the life of one loaded dataset, but it used to be called fresh, with
    no caching at all, on *every* Aladin catalog rebuild — including every
    single "Show contributing exposures" checkbox toggle (explicit user
    report: "really slow and laggy to plot provinces or even remove them
    from the plot"). Profiled directly (cProfile against a real
    32,490-target stacked L1 cube, not guessed) to this function alone:
    7.3 of the payload callback's 8.9 total seconds, almost entirely
    `np.nanmean` called once *per target* (64,981 individual calls) in a
    plain Python loop rather than vectorized — a real fix for that inner
    loop is a separate, larger change (each target's own `t.spectra[i]
    .flux`/`t.meta` arrays aren't guaranteed the same length across
    targets, so a single vectorized reduction isn't a small edit); caching
    the already-correct result is the safe, immediate fix for the actual
    complaint (repeated recomputation on every toggle), without touching
    that inner logic at all.

    (A wavelength-sub-range-restricted variant of "flux" briefly lived
    here, driven by a 2D "Apply range to 2D colour" button — removed per
    explicit follow-up: "remove apply to 2d feature for wavelenght
    range as it does not work well and it is not what I want.")"""
    cache_key = st._data_generation
    if st._color_by_cache_key != cache_key:
        st._color_by_cache = {}
        st._color_by_cache_key = cache_key
    if color_by in st._color_by_cache:
        return st._color_by_cache[color_by]

    if color_by == "snr":
        out = np.full(len(st.targs), np.nan)
        for i, t in enumerate(st.targs):
            vals = [m.get("SNR") for m in t.meta if isinstance(m, dict) and m.get("SNR") is not None]
            if vals:
                out[i] = np.nanmean(vals)
    else:
        # "flux" (default/fallback)
        out = np.array([
            sum(np.nanmean(t.spectra[i].flux) for i in range(len(t.spectra)))
            for t in st.targs
        ])
    st._color_by_cache[color_by] = out
    return out


def _build_slit_explorer_figure(selected_item):
    """The Slit Explorer panel's figure, or None if there's nothing
    meaningful to show (not loaded, or loaded but not fibre-level — see
    AppState.is_fibre_level). Reuses the exact same colour-by/vmin/vmax/
    scale STATE the Aladin overlay itself uses (COLOR_BY_OPTIONS/
    _color_by_values), so the two views are always in visual agreement."""
    st = STATE
    if not st.is_fibre_level():
        return None
    nspec = np.array([t.meta[0].get("NSPEC", np.nan) for t in st.targs], dtype=np.float64)
    if not np.isfinite(nspec).any():
        return None
    value_label, colorscale, default_vmin = COLOR_BY_OPTIONS[st.color_by]
    values = _color_by_values(st, st.color_by)
    finite_vals = values[np.isfinite(values)]
    vmin = st.color_vmin if st.color_vmin is not None else default_vmin
    vmax = st.color_vmax if st.color_vmax is not None else (
        float(np.nanmax(finite_vals)) if finite_vals.size else 1.0)
    status = [str(t.meta[0].get("FIB_STATUS", "")) for t in st.targs]
    targuse = [str(t.meta[0].get("TARGUSE", "")) for t in st.targs]
    return slit_explorer_figure(
        nspec, st.aps_id_arr, values, vmin=vmin, vmax=vmax, colorscale=colorscale,
        scale=st.color_scale, selected_aps_id=selected_item, status=status, targuse=targuse,
        value_label=value_label.split(" (")[0],
    )


def slit_info_text(selected_item):
    """Always-visible "where am I on the slit" readout for the currently-
    selected fibre — explicit user request: the plot's own hover tooltip
    only shows info for whichever point the mouse happens to be over (and
    disappears the moment it isn't), not necessarily the actual selection,
    so a genuine "I changed fibre, now show me its slit info" summary
    needs its own persistent text, the same fix aladin-info-box got
    earlier for the identical reason. None if there's nothing to show
    (not fibre-level, or no selection)."""
    st = STATE
    if not st.is_fibre_level() or selected_item is None:
        return None
    idx = st.targs_apstoid.get(selected_item)
    if idx is None:
        return None
    t = st.targs[idx]
    nspec = t.meta[0].get("NSPEC")
    value_label, _, _ = COLOR_BY_OPTIONS[st.color_by]
    values = _color_by_values(st, st.color_by)
    value = values[idx] if idx < len(values) else None
    value_str = f"{value:.3g}" if value is not None and np.isfinite(value) else "n/a"
    status = str(t.meta[0].get("FIB_STATUS", "")).strip() or "—"
    targuse = str(t.meta[0].get("TARGUSE", "")).strip() or "—"
    return (f"NSPEC (slit pos): {nspec}   {value_label.split(' (')[0]}: {value_str}   "
            f"Status: {status}   Use: {targuse}")


# A LocalProxy, not a bare instance — per-browser-session in
# aps_explorer.py's --multi-session mode, one shared instance otherwise
# (standalone usage, or outside any Flask request at all, e.g. this
# repo's own test suite). See aps_explorer_session.py's module docstring
# for the full design; this module has no other reason to import it.
STATE = LocalProxy(lambda: _sess.current_bundle().get_or_create("l1", AppState))

_FLUX_UNIT = "1e-18 erg/s/cm²/Å"
_GRAPH_CONFIG = style.GRAPH_CONFIG


def _build_fiber_map_figure(selected_aps_id=None):
    flux = np.array([
        sum(np.nanmean(t.spectra[i].flux) for i in range(len(t.spectra)))
        for t in STATE.targs
    ])
    use = np.array([t.targuse for t in STATE.targs])
    return fiber_map_figure(
        STATE.coord_arr[:, 0], STATE.coord_arr[:, 1], flux, use, STATE.aps_id_arr,
        selected_aps_id=selected_aps_id, title="Fibre selection (click a fibre)",
        vmin=STATE.color_vmin if STATE.color_vmin is not None else 12.0,
        vmax=STATE.color_vmax,
    )


def _build_spectra_figure(aps_id):
    idx = STATE.targs_apstoid[aps_id]
    targ = STATE.targs[idx]
    n_arms = len(targ.spectra)
    arms = [f"arm{i}" for i in range(n_arms)]
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    wave_d, flux_d, ivar_d = {}, {}, {}
    for i, arm in enumerate(arms):
        wave_d[arm] = targ.spectra[i].wave
        flux_d[arm] = targ.spectra[i].flux
        ivar_d[arm] = targ.spectra[i].ivar
    # SENS_CORR (per-target, per-arm metadata written by aps_utils.py's
    # loader) reflects whether sensitivity correction was *actually*
    # applied — the ground truth, rather than trusting the requested
    # --sens_corr flag blindly. Without it, flux is raw, unscaled detector
    # counts (aps_utils.vectorized_sensitivity_correction only multiplies
    # by the funit-scaled sensitivity function when sens_corr=True), so
    # labelling it "1e-18 erg/s/cm²/Å" would be simply wrong.
    sens_applied = all(
        bool(targ.meta[i].get("SENS_CORR", 0)) for i in range(n_arms) if isinstance(targ.meta[i], dict)
    )
    flux_unit = _FLUX_UNIT if sens_applied else "counts (uncalibrated — no sensitivity correction)"
    flux_fig = spectrum_overlay_figure(
        arms, wave_d, flux_d, {a: None for a in arms},
        data_label="Flux", flux_unit=flux_unit, wave_label="λ [Å]",
        figure_title=f"Fibre {aps_id} — Flux", percentile_clip=(1, 99),
    )
    ivar_fig = spectrum_overlay_figure(
        arms, wave_d, ivar_d, {a: None for a in arms},
        data_label="IVAR", flux_unit="IVAR", wave_label="λ [Å]",
        figure_title=f"Fibre {aps_id} — Inverse Variance", percentile_clip=(1, 99),
    )
    # This app's own live view, not the static PNG export path — see
    # fill_container_width's own docstring for why that distinction means
    # the fixed default width is cleared here rather than at the source.
    return style.fill_container_width(flux_fig), style.fill_container_width(ivar_fig)


def _build_fwhm_figures(aps_id):
    """Cached per `aps_id`, invalidated only on a fresh load() (same
    `self._data_generation`-keyed pattern already used by
    `_color_by_values()`/`contributing_fibre_positions()` for the same
    reason — see `AppState.load()`'s own comment on that field for a real
    concurrent-access bug this specific pattern was changed to fix) —
    explicit follow-up report: "when I am going from spectra
    to fwhm tab it takes a few seconds to load and then if for the same
    selected target I get back to spectra tab and then fwhm, it again
    reload[s] the whole plotting system for fwhm... that information is
    already there and I have not changed anything." `fwhm_overview_
    figure` in particular rebuilds one real Plotly trace per "cloud"
    fibre (up to `_MAX_CLOUD_FIBERS`, see that module's own docstring)
    from scratch on every call — genuinely expensive, and, for a fixed
    `aps_id` against the same loaded dataset, entirely deterministic
    (every input — `STATE.fwhm_cache`'s own contents, `STATE.aps_id_arr`
    — is fixed for the life of one loaded dataset; only `aps_id` varies
    call to call). Caching the already-correct built figures is the
    direct fix, mirroring `_color_by_values`'s own precedent exactly."""
    if STATE.fwhm_cache is None:
        return None, None
    cache_key = STATE._data_generation
    if STATE._fwhm_figure_cache_key != cache_key:
        STATE._fwhm_figure_cache = {}
        STATE._fwhm_figure_cache_key = cache_key
    if aps_id in STATE._fwhm_figure_cache:
        return STATE._fwhm_figure_cache[aps_id]

    c = STATE.fwhm_cache
    overview = fwhm_overview_figure(
        c["arm_names"], c["wave_grids"], c["fiber_fwhm"], c["global_fwhm"],
        STATE.aps_id_arr, highlighted_aps_id=aps_id, mode_label=c["mode_label"],
    )
    detail = None
    if aps_id is not None and aps_id in c["fiber_fwhm"]:
        detail = fwhm_detail_figure(
            c["arm_names"], c["wave_grids"], c["fiber_fwhm"][aps_id], c["global_fwhm"],
            STATE.raw_fwhm_points(aps_id), aps_id,
        )
        detail = style.fill_container_width(detail)
    result = (style.fill_container_width(overview), detail)
    STATE._fwhm_figure_cache[aps_id] = result
    return result


# --------------------------------------------------------------------------- #
# Layout pieces and callback logic — library functions only. This module no
# longer owns a Dash `app`/layout/CLI entry point of its own: PyAPS.aps_explorer
# is the single Dash app for all L1/L2 viewing now, and imports this module
# purely for its state (AppState/STATE), loader (_load_dataset),
# figure builders, and the layout/callback functions kept below (still
# plain, undecorated functions — reused by the explorer's own callbacks).
# --------------------------------------------------------------------------- #


_SECTION_STYLE = {
    "fontSize": "13px", "fontWeight": "bold", "color": "var(--pyaps-ink)",
    "marginTop": "18px", "marginBottom": "8px", "paddingTop": "12px",
    "borderTop": "1px solid #e0e0e0", "textTransform": "uppercase", "letterSpacing": "0.03em",
}
_LABEL_STYLE = {"fontSize": "11px", "color": "var(--pyaps-ink-muted)", "display": "block", "marginBottom": "3px"}
# Explicit background/color/border (there wasn't one before) so these
# fields — used throughout the load-form drawer — don't stay stark
# white-on-black inside an otherwise dark-mode drawer; var(--pyaps-paper) is
# a shade darker than the drawer's own var(--pyaps-paper-raised) background
# in both themes, which is what already reads as "an input field" (a
# subtle sunken look) rather than truly needing a border to read as one.
_INPUT_STYLE = {
    "width": "100%", "fontSize": "13px", "boxSizing": "border-box", "padding": "4px 6px",
    "backgroundColor": "var(--pyaps-paper)", "color": "var(--pyaps-ink)", "border": "1px solid var(--pyaps-line-strong)",
    "borderRadius": "4px",
}
_GRID_STYLE = {
    "display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(190px, 1fr))",
    "gap": "12px", "marginBottom": "4px",
}


def _section(title):
    return html.Div(title, style=_SECTION_STYLE)


def _field(label, comp):
    return html.Div([html.Label(label, style=_LABEL_STYLE), comp])


def _browse_button(target_id, mode="dir"):
    """Opens the shared server-side path-browser modal (defined in
    aps_explorer.py, since it's reused by both the L1 and L2 sections)
    targeting this specific text field. A native OS file-picker isn't an
    option here — this app often runs on a remote server, so the
    browser's own local filesystem is the wrong machine entirely; every
    path field here is a *server-side* path, hence a small server-side
    directory browser instead. The text field stays freely editable by
    hand either way — this is a shortcut, not a replacement."""
    return html.Button(
        "📁", id={"type": "path-browse-btn", "target": target_id, "mode": mode},
        n_clicks=0, title="Browse server files/directories...", type="button",
        style={"marginLeft": "4px", "fontSize": "13px", "padding": "3px 9px",
               "cursor": "pointer", "flexShrink": "0"},
    )


def _field_browsable(label, input_id, placeholder, mode="dir", value=None, area_style=False):
    """Same as _field(), but the input is paired with a _browse_button and
    the two are aligned as a single row."""
    comp = (dcc.Textarea if area_style else dcc.Input)
    kwargs = dict(id=input_id, style={**_INPUT_STYLE, "flex": "1", **({"height": 45} if area_style else {})},
                  placeholder=placeholder, value=value)
    if not area_style:
        kwargs["type"] = "text"
    return html.Div([
        html.Label(label, style=_LABEL_STYLE),
        html.Div([comp(**kwargs), _browse_button(input_id, mode=mode)],
                 style={"display": "flex", "alignItems": "flex-start"}),
    ])


def _v(current, attr, default=None):
    """Safe attribute pull from the argparse Namespace a dataset was last
    loaded with (STATE.load_args), for prefilling the form when it's
    reopened — None (not just a missing attr) also falls through to
    `default`, since that's how "use the built-in default" is represented
    throughout this form's own fields."""
    if current is None:
        return default
    val = getattr(current, attr, default)
    return default if val is None else val


def _v_lines(current, attr, default=None):
    """Same as _v, but for fields the *_load_form stores as a list of
    strings (infiles, mask_areas, wlranges) — rejoined one-per-line to
    match the Textarea format the field itself uses."""
    val = _v(current, attr, None)
    if val is None:
        return default
    return "\n".join(val) if isinstance(val, (list, tuple)) else val


def _v_flags(current, flag_names, default):
    """Same idea, for the boolean processing-option flags stored as plain
    attributes on the Namespace (sens_corr=True/False, etc.) — rebuilds
    the dcc.Checklist `value` list of currently-True flag names."""
    if current is None:
        return default
    selected = [f for f in flag_names if getattr(current, f, False)]
    if getattr(current, "raw_mode", False):
        selected.append("raw")
    return selected


_ADVANCED_FLAG_NAMES = (
    "skysub", "split_arms", "mask_bad_overlap", "use_resolution_deconvolution",
    "rereplace_binned_cal", "normalize_ivar", "skysub_mask_residuals",
    "spaxel_weighted_lsf", "extinction_corr",
)
_ADVANCED_FLAG_DEFAULTS = [
    "skysub", "mask_bad_overlap", "use_resolution_deconvolution",
    "rereplace_binned_cal",
    # normalize_ivar deliberately NOT on by default any more — explicit
    # request: "By default I do not want the ivar normalisation for the
    # PyAPS data explorer... make default to False."
    #
    # spaxel_weighted_lsf (IFU/LIFU/MIFU only) also deliberately NOT on
    # by default -- explicit request: "after it loaded with default
    # behaviour which is disable single spaxel lsf, I can enable it by
    # myself in the setting and input params and check it." Load once
    # with it off (today's existing global-FWHM-only behaviour,
    # unchanged), tick it on in "Advanced processing options", then
    # click "Load / Reload" again to reprocess the same dataset with it
    # on -- see aps_ifu_spaxel_contrib.py / aps_utils.APSOB's own
    # spaxel_weighted_lsf parameter.
]


def _load_form(current=None, locked=False):
    """`current` — the argparse Namespace (`STATE.load_args`) the
    currently-loaded dataset (if any) was loaded with, or None if nothing
    is loaded yet. When given, every field below shows that dataset's
    actual value instead of an empty box, so reopening "Load dataset"
    after a load shows what's actually running rather than a blank form —
    change whichever field(s) you want and hit Load again in place.

    `locked` — server mode (`aps_explorer_session.MULTI_SESSION`) with a
    dataset already recorded for this session (see
    `aps_explorer_auth.check_load_authorized`'s own docstring): the
    actual input file path(s) must never be written into the page at
    all — not even as a disabled field's `value=`, which page source
    would still reveal. `lf-infiles`/`lf-infiles-list` stay present in
    the tree (kept empty) purely so Dash's own State resolution for
    `handle_l1_load` has something to bind to; the callback itself
    ignores their submitted value whenever a dataset is already
    recorded, reloading the session's own known-good files instead —
    what's rendered here is cosmetic, not the actual security boundary.

    `caldir`/`catdir`/`configdir` get the same treatment when locked,
    but for a different reason and a lighter touch: unlike the input
    files, these are *meant* to stay genuinely editable even once
    locked (see the design note this whole locking mechanism was built
    from: "every processing param stays editable... except the
    infiles") — a real user really might want to point at a different
    calibration set. What changed is not editability, just prefill: a
    real user report ("I still can see the full path directory for CAL
    and CAT which is dangerous... I do not see that for input file
    which is good") confirmed the *value* itself was leaking the real
    server directory layout into the page the same way infiles used to.
    Left blank (with a placeholder explaining why) rather than
    prefilled with the real path — submitting it blank still resolves
    correctly via `handle_l1_load`'s own `caldir or _sess.DEFAULT_CALDIR`
    fallback (built earlier this same project), so this is purely a
    prefill/display change, not a behavior change for the common case
    (a weaveOR handoff relying on the deployment default) — an explicit
    override is still one keystroke away if someone actually needs one."""
    if locked:
        input_files_section = html.Div([
            _section("Input files"),
            html.Div("🔒 Dataset fixed for this session — opened via a direct link.",
                      style={"fontSize": "12px", "color": "var(--pyaps-ink-muted)", "fontStyle": "italic",
                             "marginBottom": "8px"}),
            dcc.Textarea(id="lf-infiles", style={"display": "none"}, value=""),
            dcc.Input(id="lf-infiles-list", style={"display": "none"}, value=""),
        ])
    else:
        input_files_section = html.Div([
            _section("Input files"),
            html.Label("One file per line", style=_LABEL_STYLE),
            html.Div([
                dcc.Textarea(id="lf-infiles", style={**_INPUT_STYLE, "height": 70, "flex": "1"},
                             placeholder="<PYAPS_DATA>/L1/20250707/stack_3097460.fit",
                             value=_v_lines(current, "infiles")),
                _browse_button("lf-infiles", mode="file-append"),
            ], style={"display": "flex", "alignItems": "flex-start"}),
            html.Label("...or a file listing one infile per line (--infiles_list, alternative to the above)",
                       style=_LABEL_STYLE),
            _field_browsable("", "lf-infiles-list", "/path/to/infiles_list.txt", mode="file",
                              value=_v(current, "infiles_list")),
        ])

    return html.Div([
        html.H3("Load L1 dataset", style={"marginTop": 0}),

        input_files_section,

        _section("Directories"),
        html.Div([
            _field_browsable("Calibration dir (--caldir)", "lf-caldir",
                              "(using this server's default)" if locked else "<PYAPS_DATA>/CAL",
                              mode="dir", value=None if locked else _v(current, "caldir")),
            _field_browsable("Catalog dir (--catdir)", "lf-catdir",
                              "(using this server's default)" if locked else "<PYAPS_DATA>/CAT",
                              mode="dir", value=None if locked else _v(current, "catdir")),
            _field_browsable("Config dir (--configdir)", "lf-configdir",
                              "(using this server's default)" if locked else "<PYAPS_DIR>/configs/ExGal_configs",
                              mode="dir", value=None if locked else _v(current, "configdir")),
        ], style=_GRID_STYLE),
        html.Div([
            _field_browsable("L1 reference catalog (optional)", "lf-l1ref",
                              "/path/to/l1_reference.fits", mode="file", value=_v(current, "l1_reference")),
            _field_browsable("L2 reference catalog (optional)", "lf-l2ref",
                              "/path/to/l2_reference.fits", mode="file", value=_v(current, "l2_reference")),
        ], style=_GRID_STYLE),

        _section("Target filters"),
        html.Div([
            _field("APS IDs", dcc.Input(id="lf-apsids", type="text", style=_INPUT_STYLE,
                                         placeholder="100,101,102", value=_v(current, "aps_ids"))),
            _field("Target survey", dcc.Input(id="lf-targsrvy", type="text", style=_INPUT_STYLE,
                                               placeholder="WS2022A2-002,WS2022A2-003",
                                               value=_v(current, "targsrvy"))),
            _field("Target class", dcc.Input(id="lf-targclass", type="text", style=_INPUT_STYLE,
                                              placeholder="GALAXY,QSO,STAR", value=_v(current, "targclass"))),
            _field("Mask APS IDs", dcc.Input(id="lf-maskids", type="text", style=_INPUT_STYLE,
                                              placeholder="200,201,202", value=_v(current, "mask_aps_ids"))),
            _field("Load 1 of every N spaxels/fibres (speeds up very large cubes; "
                   "empty/1 = load every spaxel)",
                   dcc.Input(id="lf-decimate-stride", type="number", min=1, step=1,
                              style=_INPUT_STYLE, placeholder="e.g. 3",
                              value=_v(current, "decimate_stride"))),
        ], style=_GRID_STYLE),

        _section("Spatial / wavelength selection (optional)"),
        html.Div([
            _field("Area: RA(deg),DEC(deg),A(arcsec),B(arcsec),PA(deg CCW)",
                   dcc.Input(id="lf-area", type="text", style=_INPUT_STYLE,
                             placeholder="246.29,40.90,60.0,60.0,0.0", value=_v(current, "area"))),
            _field("Arms ratio (comma-sep., one per infile)",
                   dcc.Input(id="lf-arms-ratio", type="text", style=_INPUT_STYLE, placeholder="1.0,1.0",
                             value=_v(current, "arms_ratio"))),
        ], style=_GRID_STYLE),
        html.Label("Mask areas — one \"RA(deg),DEC(deg),A(arcsec),B(arcsec),PA(deg CCW)\" per line "
                   "(same format as Area above)", style=_LABEL_STYLE),
        dcc.Textarea(id="lf-mask-areas", style={**_INPUT_STYLE, "height": 45},
                     placeholder="246.0,40.0,30.0,30.0,0.0\n247.0,41.0,20.0,20.0,45.0",
                     value=_v_lines(current, "mask_areas")),
        html.Label("Wavelength ranges — one \"wlmin,wlmax\" per line, one per infile", style=_LABEL_STYLE),
        dcc.Textarea(id="lf-wlranges", style={**_INPUT_STYLE, "height": 45},
                     placeholder="3800,5950\n5900,9270", value=_v_lines(current, "wlranges")),

        _section("Processing options"),
        dcc.Checklist(
            id="lf-flags",
            options=[
                {"label": " Sensitivity correction", "value": "sens_corr"},
                {"label": " Join arms", "value": "join_arms"},
                {"label": " Vacuum wavelengths", "value": "vacuum"},
                {"label": " Mask gaps", "value": "mask_gaps"},
                {"label": " Safe mask gaps", "value": "safe_mask_gaps"},
                {"label": " Tellurics", "value": "tellurics"},
                {"label": " Fill gap", "value": "fill_gap"},
                {"label": " Cosmic-ray rejection", "value": "crr"},
                {"label": " Collapse to single spectrum", "value": "collapse"},
                {"label": " Load raw/unmodified L1 data (deselects every option above)",
                 "value": "raw"},
            ],
            value=_v_flags(current, ("sens_corr", "join_arms", "vacuum", "mask_gaps", "safe_mask_gaps",
                                      "tellurics", "fill_gap", "crr", "collapse"),
                            default=["sens_corr", "safe_mask_gaps"]),
            style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))",
                   "gap": "4px 12px", "fontSize": "12px"},
            labelStyle={"display": "block"},
        ),

        _section("Advanced processing options"),
        html.Div("Every remaining option APSOB itself accepts — real, but rarely touched.",
                 style={"fontSize": "11px", "color": "var(--pyaps-ink-faint)", "marginBottom": "6px"}),
        dcc.Checklist(
            id="lf-flags-advanced",
            options=[
                {"label": " Sky subtraction", "value": "skysub"},
                {"label": " Split arms (with Join arms above)", "value": "split_arms"},
                {"label": " Mask bad overlap (with Join arms above)", "value": "mask_bad_overlap"},
                {"label": " Resolution-deconvolution edge masking", "value": "use_resolution_deconvolution"},
                {"label": " Re-replace binned cal files if unbinned missing", "value": "rereplace_binned_cal"},
                {"label": " Normalize IVAR across arms", "value": "normalize_ivar"},
                {"label": " Mask sky residuals (OH forest)", "value": "skysub_mask_residuals"},
                {"label": " Spaxel-weighted LSF/FWHM (IFU/LIFU/MIFU cubes only — per-spaxel "
                           "weighted from dithered fibres' real LSF/FWHM, instead of the flat "
                           "global value)", "value": "spaxel_weighted_lsf"},
                {"label": " Galactic extinction correction (automatic, per-target SFD98+"
                           "Fitzpatrick99 dereddening from each target's own sky position — "
                           "same correction validated for REDROCK/ExGal, see doc/aps_rr.md; "
                           "full strength, off by default)", "value": "extinction_corr"},
            ],
            value=_v_flags(current, _ADVANCED_FLAG_NAMES, default=_ADVANCED_FLAG_DEFAULTS),
            style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(260px, 1fr))",
                   "gap": "4px 12px", "fontSize": "12px", "marginBottom": "10px"},
            labelStyle={"display": "block"},
        ),
        html.Div([
            _field("LSF source", dcc.Dropdown(
                id="lf-lsftype",
                options=[{"label": "LSF (pre-computed B-spline lsf_ files)", "value": "LSF"},
                         {"label": "FWHM (fit from wave_ arc-line files)", "value": "FWHM"}],
                value=_v(current, "lsftype", "LSF"), clearable=False, style={"fontSize": "12px"})),
            _field("IVAR normalization mode", dcc.Dropdown(
                id="lf-ivar-norm-mode",
                options=[{"label": "Balanced (equal chi-square contribution)", "value": "balanced"},
                         {"label": "Pixels (weighted by wavelength coverage)", "value": "pixels"},
                         {"label": "Hybrid", "value": "hybrid"}],
                value=_v(current, "ivar_normalization_mode", "balanced"), clearable=False,
                style={"fontSize": "12px"})),
        ], style=_GRID_STYLE),
        html.Div([
            _field("Edge pixels to mask", dcc.Input(id="lf-edge-pixels", type="number", style=_INPUT_STYLE,
                                                      placeholder="5", value=_v(current, "edge_pixels_to_mask"))),
            _field("Gap-masking width (px)", dcc.Input(id="lf-gap-offset-pix", type="number", style=_INPUT_STYLE,
                                                         placeholder="10", value=_v(current, "offset_gap_pix"))),
            _field("Flux unit scale", dcc.Input(id="lf-funit", type="number", style=_INPUT_STYLE,
                                                 placeholder="1e18", value=_v(current, "funit"))),
            _field("Template σ₀ (Å) — currently has no effect (reserved)",
                   dcc.Input(id="lf-template-sigma0", type="number", style=_INPUT_STYLE,
                             placeholder="0.5", value=_v(current, "template_sigma0_angstrom"))),
        ], style=_GRID_STYLE),

        html.Button("Load / Reload", id="lf-run-l1", n_clicks=0,
                    style={"marginTop": "14px", "fontSize": "13px", "padding": "6px 16px"}),
        html.Div(id="lf-status-l1", style={"color": "crimson", "marginTop": "6px", "fontSize": "12px"}),
    ], style={"padding": "20px", "fontSize": "13px"})


def load_from_form_fields(infiles_text, infiles_list, l1ref, l2ref, apsids, targsrvy, targclass, maskids,
                           area, mask_areas_text, wlranges_text, arms_ratio, caldir, catdir, configdir, flags,
                           advanced_flags=None, lsftype=None, ivar_normalization_mode=None,
                           edge_pixels_to_mask=None, offset_gap_pix=None, funit=None,
                           template_sigma0_angstrom=None, decimate_stride=None):
    """Build an argparse-style Namespace from the "Load dataset" form's
    raw field values (mirrors the CLI options 1:1) and load it into
    STATE. Raises on a validation failure or a load error — the caller
    (a real Dash callback in aps_explorer.py) decides how to surface
    that as a status message.

    The `advanced_flags`/`lsftype`/... kwargs default to None so every
    *existing* caller (including every test in this repo) keeps working
    unchanged — they only need to be passed by callers that actually
    expose the "Advanced processing options" section (aps_explorer.py's
    own handle_l1_load callback).
    """
    infiles = [f.strip() for f in (infiles_text or "").splitlines() if f.strip()]
    if not infiles and not infiles_list:
        raise ValueError("At least one input file (or --infiles_list) is required.")

    parser = _build_arg_parser()
    args = parser.parse_args([])
    args.infiles = infiles or None
    args.infiles_list = infiles_list or None
    args.l1_reference = l1ref or None
    args.l2_reference = l2ref or None
    args.aps_ids = apsids or None
    args.targsrvy = targsrvy or None
    args.targclass = targclass or None
    args.mask_aps_ids = maskids or None
    args.decimate_stride = int(decimate_stride) if decimate_stride else None
    args.area = area or None
    mask_areas = [ln.strip() for ln in (mask_areas_text or "").splitlines() if ln.strip()]
    args.mask_areas = mask_areas or None
    wlranges = [ln.strip() for ln in (wlranges_text or "").splitlines() if ln.strip()]
    args.wlranges = wlranges or None
    args.arms_ratio = arms_ratio or None
    args.caldir = caldir or None
    args.catdir = catdir or None
    args.configdir = configdir or None
    flags = flags or []
    for f in ("sens_corr", "join_arms", "vacuum", "mask_gaps", "safe_mask_gaps", "tellurics",
              "fill_gap", "crr", "collapse"):
        setattr(args, f, f in flags)
    args.raw_mode = "raw" in flags
    if args.raw_mode:
        for f in ("sens_corr", "join_arms", "vacuum", "mask_gaps", "safe_mask_gaps", "tellurics", "fill_gap"):
            setattr(args, f, False)

    advanced_flags = advanced_flags if advanced_flags is not None else list(_ADVANCED_FLAG_DEFAULTS)
    for f in _ADVANCED_FLAG_NAMES:
        setattr(args, f, f in advanced_flags)
    if lsftype:
        args.lsftype = lsftype
    if ivar_normalization_mode:
        args.ivar_normalization_mode = ivar_normalization_mode
    if edge_pixels_to_mask is not None:
        args.edge_pixels_to_mask = int(edge_pixels_to_mask)
    if offset_gap_pix is not None:
        args.offset_gap_pix = int(offset_gap_pix)
    if funit is not None:
        args.funit = float(funit)
    if template_sigma0_angstrom is not None:
        args.template_sigma0_angstrom = float(template_sigma0_angstrom)

    STATE.load(args)


# L1's settings row used to hold only the Colour range (Min/Max/Reset)
# control — moved to sit directly under the Aladin colour legend instead
# (see aps_explorer.py's _aladin_color_range_control). It now holds the
# "Colour by" dropdown instead (item 8: "can we have more options?"),
# mirroring aps_MOSviewer.py's own "mos-color-by" control.

def _settings_row():
    return html.Div([
        # Wrapped in its own id (shared with aps_MOSviewer.py's version
        # of this same control) so aps_explorer.update_map_mode can hide
        # it specifically in 3D mode — explicit request: "'COlor aladin
        # point by' shoujlkd be only visible in 2d mode as in 3d it is
        # not releavant" (the 3D cube always colours by summed flux
        # regardless of this setting, see flux_cube_figure's own
        # docstring for why it's deliberately not wired to it at all).
        html.Div([
            html.Label("Colour Aladin points by:"),
            dcc.Dropdown(
                id="l1-color-by",
                options=[{"label": label, "value": key} for key, (label, _, _) in COLOR_BY_OPTIONS.items()],
                value=STATE.color_by, clearable=False, style={"width": "280px"},
            ),
        ], id="aladin-color-by-row", style={"display": "inline-block", "verticalAlign": "top"}),
    ], style={"padding": "10px 0"})


def on_fiber_click(click_data):
    if not click_data or not STATE.loaded():
        return no_update
    point = click_data["points"][0]
    aps_id = point.get("customdata")
    if aps_id is None:
        return no_update
    return int(aps_id)


_TAB_VIEWPORT_HEIGHT = "650px"


def _scrollable_tab(children):
    """Fixed-height, scrollable wrapper for a tab's content — same pattern
    as aps_MOSviewer._scrollable_graph. Without this, the FWHM tab's two
    ~420px graphs (plus Spectra's two 320px ones) just grow the page
    instead of scrolling within their own area, which on a long page pushed
    the always-visible log panel far enough down that it looked like the
    plots were "overlaying"/hiding it.

    `pyaps-scrollbox` (see its CSS rules in aps_explorer.py's
    `index_string`) replaces the OS's own auto-hide overlay scrollbar with
    an always-rendered one whenever this box's content genuinely overflows
    — explicit user report: on some platforms the default overlay
    scrollbar is invisible except while actively scrolling, easy to miss
    entirely, so it looked like there was no way to reach the rest of a
    tall plot."""
    return html.Div(
        children, className="pyaps-scrollbox",
        style={"height": _TAB_VIEWPORT_HEIGHT, "overflowY": "auto",
               "border": "1px solid #ddd", "borderRadius": "4px"},
    )


def _stacked_graph(fig, *, last=False):
    """One graph in a vertically-stacked set (Spectra's Flux+IVAR, FWHM's
    overview+detail).

    Two bugs, fixed together here: the dcc.Graph style used to hardcode a
    height ("320px"/"420px") independent of the figure's own actual
    height, which `spectrum_overlay_figure`/`fwhm_*_figure` compute from
    however many panel-rows that particular figure has (arm count, rank
    count, ...) — whenever that computed height exceeded the hardcoded
    one, the plot overflowed past its own box, uncontained, straight into
    the next graph below it: exactly the reported "header of the lower
    plot overlaps the x-axis of the plot above" (a fixed number could
    never have been right for every dataset, only ones that happened to
    have few enough rows to fit). Reading `fig.layout.height` back off the
    already-built figure instead means the box is always exactly as tall
    as its own content, so there is nothing left *to* overflow. A
    marginBottom spacer between graphs (not after the last one) is added
    on top of that fix, not instead of it, per the explicit request for
    "proper spacer between each set"."""
    style_dict = {"height": f"{fig.layout.height}px"}
    if not last:
        style_dict["marginBottom"] = "22px"
    return dcc.Graph(figure=fig, style=style_dict, config=_GRAPH_CONFIG)


def _primary_header_tab_content(locked=False):
    """One section per input file this dataset was loaded from — a small
    monospace "File: /full/path/..." label (same convention as the IFU
    Processing History tab's own source-path line) above a sortable/
    filterable Keyword/Value/Comment DataTable of that file's raw primary
    FITS header. Multiple sections (not just the first file) so nothing
    is hidden on a multi-arm load — real WEAVE L1 primary headers are a
    few hundred cards at most, not large enough that showing every input
    file's is unreasonable.

    `locked` — same server-mode-with-a-fixed-dataset flag `load_summary`
    already takes (see that method's own docstring): a real server
    filesystem path has no business reaching a browser in that mode, even
    tucked away in a rarely-opened tab — explicit report: "The tab show
    the header still show the full path of L1 files... make sure you fix
    that one as well to not show the full path in server mode..for
    standalone it is fine." Masks the path line only; the header
    table itself (keyword/value/comment, no path info) is unaffected."""
    files = STATE.primary_headers()
    if not files:
        return html.Div("No primary header available.")

    sections = []
    for i, f in enumerate(files):
        file_label = ("🔒 Dataset fixed for this session — opened via a direct link."
                      if locked else f"File: {f['file']}")
        sections.append(html.Div(
            file_label,
            style={"fontFamily": "monospace", "fontSize": "11px", "color": "var(--pyaps-ink-muted)",
                   "marginTop": "14px" if i else "0", "marginBottom": "4px"},
        ))
        if "error" in f:
            sections.append(html.Div(f"Could not read header: {f['error']}",
                                      style={"color": "crimson", "fontSize": "12px"}))
            continue
        data = [{"Keyword": k, "Value": "" if v is None else str(v), "Comment": c}
                for k, v, c in f["cards"]]
        table = dash_table.DataTable(
            data=data,
            columns=[{"name": "Keyword", "id": "Keyword"}, {"name": "Value", "id": "Value"},
                     {"name": "Comment", "id": "Comment"}],
            sort_action="native", filter_action="native",
            **{**style.DATATABLE_KWARGS,
               "style_table": {"maxHeight": "400px", "overflowY": "auto"},
               "style_cell": {**style.DATATABLE_STYLE_CELL, "fontFamily": "monospace"}},
        )
        # One table per input file (see this function's own docstring) —
        # each needs its own unique id/export button, only available for
        # the first HEADER_TABLE_MAX_FILES of them (aps_explorer.py's own
        # registration loop pre-wires exactly that many export callbacks
        # at app startup — callback registration can't happen dynamically
        # per-request). Beyond that (an extreme, unlikely-in-practice
        # multi-file load), the table itself still renders correctly,
        # just without an id or export button — NOT clamped onto a
        # reused id, which would be a real duplicate-component-id bug.
        if i < HEADER_TABLE_MAX_FILES:
            table_id = f"l1-header-table-{i}"
            table.id = table_id
            sections.append(csv_export_row(table_id, table))
        else:
            sections.append(table)
    return _scrollable_tab(sections)


def update_tab_content(tab, aps_id, locked=False):
    if not STATE.loaded():
        return html.Div("No fibre selected.")

    # Whole-dataset, not per-fibre — unlike every other tab here, doesn't
    # need aps_id at all, so it's handled before the aps_id-None guard
    # below (the aps_id can genuinely be None only in edge cases, since a
    # default selection is always made on load, but there is no reason to
    # couple the header display to a fibre selection that doesn't affect it).
    if tab == "header":
        return _primary_header_tab_content(locked=locked)

    if aps_id is None:
        return html.Div("No fibre selected.")

    if tab == "spectra":
        flux_fig, ivar_fig = _build_spectra_figure(aps_id)
        return _scrollable_tab([
            _stacked_graph(flux_fig),
            _stacked_graph(ivar_fig, last=True),
        ])

    if tab == "fwhm":
        overview, detail = _build_fwhm_figures(aps_id)
        if overview is None:
            return html.Div("FWHM data not available for this dataset.")
        children = [_stacked_graph(overview, last=detail is None)]
        if detail is not None:
            children.append(_stacked_graph(detail, last=True))
        return _scrollable_tab(children)

    return html.Div()
