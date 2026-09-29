"""
Shared CLI argument-group registry for the WEAVE APS pipeline scripts.

Every `aps_*.py` processing script (aps_rr.py, aps_rvs.py, aps_ferre.py,
the IFU family, aps_mosExGal.py, aps_squeze.py, aps_feswi.py, aps_amy.py,
aps_alfa-neat.py, aps_cubepreview.py, aps_l1_multi_plot.py) independently
re-declared its own `argparse.ArgumentParser()` with a large, near-identical
block of ~20 shared flags (--infiles, --aps_ids, --targsrvy, --targclass,
--mask_aps_ids, --wlranges, --area, --mask_areas, --sens_corr, --mask_gaps,
--safe_mask_gaps, --tellurics, --vacuum, --fill_gap, --arms_ratio,
--join_arms, --caldir, --catdir, --configdir, --outpath, --headname,
--overwrite), followed by a near-identical "resolve" block converting the
raw CLI strings into real Python types and normalizing infiles/wlranges/
arms_ratio via `l1_fileinfo()`. Explicit request: "I hate this
repea[t]ation in all codes... make something that handle[s] reading the
param[s] and profe[ss]ional[l]y print[s] them out at the beginning...
handling extra or less param[eter]s and also respect[ing] that some codes
use the default values and some use specific param[eters] for specific
purposes."

A direct audit of all 14 scripts found this was not just cosmetic
duplication -- it had already silently drifted into several real, latent
bugs: aps_amy.py/aps_squeze.py parsed mask_aps_ids/aps_ids with a plain
`int()` instead of the `np.int32` dtype every other script uses;
aps_ifu_ExGal.py/aps_ifu_Gal.py/aps_cubepreview.py have no
`assert len(arms_ratio) == len(infiles)` validation at all;
aps_ifu_Gal.py is missing the `join_arms=False` correction for
`len(infiles) < 2` entirely; aps_cubepreview.py and aps_l1_multi_plot.py
never call `print_args()` at all (no logging of what actually ran);
aps_l1_multi_plot.py computes a corrected `join_arms` value into a local
variable and then never uses it. Migrating a script onto this module's
shared `resolve_common_args()` fixes all of the above as a disclosed
side effect of that migration.

**Existing flag names are a real external contract, not just internal
convention** -- confirmed by audit: aps_runner.py builds its actual
SLURM/bash pipeline by hardcoding every one of these scripts' exact flag
names into subprocess command strings; orchestration layers built on
top of PyAPS shell out to aps_runner.py the same way; README.md documents these scripts' flags
directly and states operators can invoke them by hand. No migrated
script's flag *name*, `type`, `default`, `required`, or `nargs` may
change as a result of using this module -- only the mechanism generating
them is shared. Help text is reproduced verbatim per script via
`overrides` wherever it genuinely differs (including pre-existing typos
like "wavelenght"/"lenghtes"/"whill") specifically so nothing about a
script's own `--help` output changes as an unintended side effect of
this refactor.

Usage, per script (see aps_rr.py/aps_rvs.py/aps_mosExGal.py for the first
three migrated as a worked example):

    from PyAPS.aps_common_args import build_common_parser, resolve_common_args

    parser = build_common_parser(
        description="...",
        groups=["target_selection", "spatial_selection", "wavelength",
                "l1_processing", "caldirs", "output"],
        overrides={"vacuum": {"default": False}, "outpath": {"help": "..."}},
        exclude=["overwrite"],  # only if this one script genuinely lacks a flag its groups would otherwise add
        extra_args=[
            (("--ntop",), dict(type=int, default=1, required=False, help="...")),
            ...  # every flag unique to this one script, verbatim
        ],
    )
    args = parser.parse_args(options) if options else parser.parse_args(demo_options)
    resolved = resolve_common_args(args)
    # resolved.aps_ids / .targsrvy / .targclass / .mask_aps_ids / .area /
    # .mask_areas / .wlranges / .arms_ratio are the real, converted values
    # (matching each script's own previous local-variable names exactly);
    # args.infiles/.wlranges/.arms_ratio/.area/.mask_areas/.join_arms are
    # also updated in place on `args` itself, matching the exact subset
    # every one of the "baseline" scripts already wrote back onto `args`
    # for print_args()'s own reporting -- args.aps_ids/.targsrvy/
    # .targclass/.mask_aps_ids deliberately stay as their original raw
    # comma-separated strings on `args` (never overwritten), exactly
    # matching current behaviour -- only `resolved.*` carries the
    # converted values for those four.
"""

from __future__ import absolute_import, division, print_function

import numpy as np

from PyAPS.aps_utils import none_or_str, str2bool, l1_fileinfo


# --------------------------------------------------------------------------- #
# Argument spec groups. Each entry is (flags, kwargs) exactly as it would be
# passed to `parser.add_argument(*flags, **kwargs)`. kwargs here are the
# most common/"canonical" version found across the scripts that share a
# given flag (verified directly against every script's own current source
# before being captured here) -- a specific script overrides only the
# individual kwargs that genuinely differ for it via `overrides=`, so no
# script's own observable --help output changes as a result of migrating.
# --------------------------------------------------------------------------- #

_GROUPS = {
    "target_selection": [
        (("--infiles",), dict(type=none_or_str, default=None, required=True,
                               help="input files", nargs='*')),
        (("--aps_ids",), dict(type=none_or_str, default=None, required=False,
                               help="comma-separated list of WEAVE APS_IDs")),
        (("--targsrvy",), dict(type=none_or_str, default=None, required=False,
                                help="comma-separated list of surveys to be considered")),
        (("--targclass",), dict(type=none_or_str, default=None, required=False,
                                 help="comma-separated list of classtypes to be considered")),
        (("--mask_aps_ids",), dict(type=none_or_str, default=None, required=False,
                                    help="comma-separated list of APS_IDs to be masked")),
    ],
    # Opt-in, not part of "target_selection" -- the IFU family and
    # aps_cubepreview.py select by patch instead and take neither flag.
    "spatial_selection": [
        (("--area",), dict(type=none_or_str, default=None, required=False,
                            help="The area in [RA(deg), DEC(deg), Radius(arcsec)] to be considered in analysis")),
        (("--mask_areas",), dict(type=none_or_str, default=None, required=False,
                                  help="The multi area(s) in [RA(deg), DEC(deg), Radius(arcsec)] to be excluded from analysis",
                                  nargs='*')),
    ],
    "wavelength": [
        (("--wlranges",), dict(type=none_or_str, default=None, required=False,
                                help="wavelength range array for each elements of the setup", nargs='*')),
    ],
    "l1_processing": [
        (("--sens_corr",), dict(type=str2bool, default=True, required=False,
                                 help="apply sensitivity function to the flux and ivar")),
        (("--mask_gaps",), dict(type=str2bool, default=True, required=False,
                                 help="mask the gaps between CCD by putting ivar= 0 (auto gap detection mode)")),
        (("--safe_mask_gaps",), dict(type=str2bool, default=True, required=False,
                                      help="mask the gaps between chips (read from lookup table) by putting ivar= 0")),
        (("--tellurics",), dict(type=str2bool, default=False, required=False,
                                 help="put large errors for regions affected by tellurics")),
        (("--vacuum",), dict(type=str2bool, default=True, required=False,
                              help="transform wavelength from air to vacuum")),
        (("--fill_gap",), dict(type=str2bool, default=False, required=False,
                                help="fill CCD gaps by interpolated values for flux")),
        (("--arms_ratio",), dict(type=none_or_str, default=None, required=False,
                                  help="Correction (flux and ivar/error) faction for each arms")),
        (("--join_arms",), dict(type=str2bool, default=False, required=False,
                                 help="Stitch two arms")),
    ],
    "caldirs": [
        (("--catdir",), dict(type=none_or_str, default=None, required=False,
                              help="Directory to keep WEAVE INPUT Catalogs")),
        (("--caldir",), dict(type=none_or_str, default=None, required=False,
                              help="Directory to keep WEAVE INPUT CALIBRATIONS")),
        (("--configdir",), dict(type=none_or_str, default=None, required=False,
                                 help="Directory to keep WEAVE Config files")),
    ],
    "output": [
        (("--outpath",), dict(type=none_or_str, default=None, required=True,
                               help="Output directory")),
        (("--headname",), dict(type=none_or_str, default="headname", required=True,
                                help="Output headname. The output filenames will be generated based on this")),
        (("--overwrite",), dict(type=str2bool, default=False,
                                 help="If True, overwrites the existing products, otherwise it will skip them")),
    ],
}


def build_common_parser(description, groups, overrides=None, exclude=None, extra_args=None):
    """Build an `argparse.ArgumentParser` from the shared spec groups above.

    Parameters
    ----------
    description : str
        Passed straight to `argparse.ArgumentParser(description=...)`.
    groups : list[str]
        Which bundles from `_GROUPS` to include, in order (only affects
        `--help` listing order, not parsing behaviour).
    overrides : dict[str, dict], optional
        `{flag_name: {kwarg: value}}` (flag_name without the leading
        `--`) -- merged into that flag's own canonical kwargs for THIS
        script only, e.g. `{"vacuum": {"default": False}}`. Use this for
        any genuine per-script difference (default, required, help text,
        including preserving a pre-existing typo) rather than silently
        picking one script's wording as correct for everyone.
    exclude : list[str], optional
        Flag names (without `--`) to skip even though their group would
        otherwise add them -- e.g. a script whose group includes
        `overwrite` but which genuinely has no such flag today.
    extra_args : list[tuple[tuple[str, ...], dict]], optional
        This script's own unique flags, in the same `(flags, kwargs)`
        shape as the registry above, added after every shared group --
        e.g. `aps_rr.py`'s `--ntop`/`--gpu`/`--zall`, or the IFU family's
        `--seg2d_*` options. Copied verbatim from each script's own
        current source, never generalized into the shared registry
        (nothing else uses them).
    """
    import argparse
    parser = argparse.ArgumentParser(description=description)
    overrides = overrides or {}
    exclude = set(exclude or [])
    seen = set()
    for group_name in groups:
        for flags, kwargs in _GROUPS[group_name]:
            name = flags[-1].lstrip("-")
            if name in exclude or name in seen:
                continue
            seen.add(name)
            kw = dict(kwargs)
            if name in overrides:
                kw.update(overrides[name])
            parser.add_argument(*flags, **kw)
    for flags, kwargs in (extra_args or []):
        parser.add_argument(*flags, **kwargs)
    return parser


def _int32_list(csv):
    return np.array(csv.split(","), dtype=np.int32).tolist()


def _str_list(csv):
    return [str(x) for x in csv.split(",")]


class _Resolved(object):
    """Plain attribute bag for resolve_common_args's return value -- one
    attribute per resolved field, `None` for any field the calling
    script's own parser never defined (so e.g. `resolved.area` is simply
    `None` for the IFU family rather than raising `AttributeError`)."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def resolve_common_args(args):
    """The shared "post-parse" step -- generalizes the exact logic already
    used by aps_rr.py/aps_rvs.py/aps_mosExGal.py (confirmed by audit to be
    the most complete/correct version among the 14 scripts: `np.int32`
    dtype for id lists, a real assert message on `arms_ratio` length,
    `infiles`/`wlranges`/`arms_ratio` normalized via `l1_fileinfo()`, and
    the `join_arms=False`-when-`len(infiles)<2` correction) to only touch
    fields that actually exist on `args` at all (`getattr(args, X, None)`
    throughout), so a script with no `--area`/`--mask_areas` (the IFU
    family) is simply unaffected for those two fields.

    Mutates `args` in place for exactly the same subset every baseline
    script already did (`args.infiles`, `args.wlranges`, `args.arms_ratio`,
    `args.area`, `args.mask_areas`, and the `args.join_arms` correction --
    all so `print_args()`'s own report shows the real, resolved values,
    not the raw CLI strings) and returns a separate `_Resolved` object
    carrying every converted value, including the four
    (`aps_ids`/`targsrvy`/`targclass`/`mask_aps_ids`) that every baseline
    script deliberately keeps as local variables rather than writing back
    onto `args` -- reproducing that exact split, not just "resolve
    everything the same way," since `print_args()`'s own report content
    depends on it.
    """
    wlranges = None
    if getattr(args, "wlranges", None) and args.wlranges[0] is not None:
        wlranges = [[float(x) for x in w.split(",")] for w in args.wlranges]

    arms_ratio = None
    if getattr(args, "arms_ratio", None) is not None:
        arms_ratio = [float(x) for x in args.arms_ratio.split(",")]
        assert len(arms_ratio) == len(args.infiles), \
            "lengths of arms_ratio(s) and infiles must be identical"

    infiles_info = l1_fileinfo(args.infiles, wlranges=wlranges, arms_ratio=arms_ratio)
    args.infiles = infiles_info["infiles"]
    wlranges = infiles_info["wlranges"]
    arms_ratio = infiles_info["arms_ratio"]
    args.wlranges = wlranges
    args.arms_ratio = arms_ratio

    if hasattr(args, "join_arms") and len(args.infiles) < 2:
        args.join_arms = False

    aps_ids = _int32_list(args.aps_ids) if getattr(args, "aps_ids", None) is not None else None
    targsrvy = _str_list(args.targsrvy) if getattr(args, "targsrvy", None) is not None else None
    targclass = _str_list(args.targclass) if getattr(args, "targclass", None) is not None else None
    mask_aps_ids = _int32_list(args.mask_aps_ids) if getattr(args, "mask_aps_ids", None) is not None else None

    area = None
    if getattr(args, "area", None) is not None:
        area = [float(x) for x in args.area.split(",")]
        args.area = area

    mask_areas = None
    if getattr(args, "mask_areas", None) and args.mask_areas[0] is not None:
        mask_areas = [[float(x) for x in m.split(",")] for m in args.mask_areas]
        args.mask_areas = mask_areas

    return _Resolved(
        infiles=args.infiles, wlranges=wlranges, arms_ratio=arms_ratio,
        aps_ids=aps_ids, targsrvy=targsrvy, targclass=targclass, mask_aps_ids=mask_aps_ids,
        area=area, mask_areas=mask_areas,
    )
