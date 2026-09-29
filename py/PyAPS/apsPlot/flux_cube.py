"""
"Flux cube" — a 3D alternative to the Aladin coordinate-map panel.
x/y are the same sky coordinates the 2D map itself uses; z is real
wavelength; colour is the summed flux within each (item, wavelength-bin)
cell. Aladin Lite has no 3D mode at all, so this is a plain Plotly 3D
scene instead — genuinely rotatable/zoomable in the browser the same way
any Plotly 3D figure is.

Framework-agnostic: plain arrays in, `go.Figure` out — same convention
as `fiber_map.py`/`spectra.py`/`slit_explorer.py`. Explicit user request:
"can we have an option that instead of the aladin coordinate map, we have
a 3d data, so x and y are coordinates and z could be the sum flux over a
range of wavelength... I want something that I can rotate or change FOV
or angle and go inside the cube and out. However for selection, it must
be only by selecting from the xy coordinate as it is now."
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from . import style

# A real L1 IFU stackcube can have tens of thousands of spaxels; at even a
# modest wavelength-bin count each one becomes a whole ring/box per depth
# layer, so the item count needs a much tighter cap here than e.g.
# fwhm.py's own _MAX_CLOUD_FIBERS (500) — WebGL orbit/zoom starts to
# stutter well before that multiplied-out geometry count would. Evenly
# subsampled (not truncated), same reasoning as that precedent. Explicit
# follow-up acknowledging this exact tradeoff for the "connected cubic
# spaxels" feature below: "...unless we have selected a masked wavelength
# or masked area or we have [the] option to show 1 out of N spaxels to
# speed up" — this cap *is* that "1 out of N" mechanism (automatic rather
# than a separate user-facing toggle), and the resulting gaps on a
# dataset larger than this cap are exactly the acknowledged exception.
MAX_ITEMS = 1200

DEFAULT_N_WAVE_BINS = 30

# User-facing wavelength-binning control, replacing a raw bin *count* —
# explicit follow-up: "I guess it is better if user be able to change the
# width in wavelength range where those bins stacked (sum) together to
# create one cross section... I do not know the golden number for the
# default binned size... do it yourself." 100 Å is the chosen default: a
# conventional, "nice round number" narrow-band-imaging bin width in
# astronomy (many real narrowband filters are themselves ~50-100 Å FWHM),
# and it lands in the same practical bin-*count* ballpark the previous
# fixed `DEFAULT_N_WAVE_BINS=30` did across WEAVE's own real wavelength
# ranges (a single HR arm spans ~2000-2400 Å -> ~20-24 bins; a full
# LR LIFU/MIFU blue+red range spans ~5900 Å -> ~59 bins) — familiar
# territory, not a wild guess. See `wave_bin_width_angstrom`'s own
# docstring for how this is applied.
DEFAULT_WAVE_BIN_WIDTH_ANGSTROM = 100.0

# Defensive cap on the *derived* bin count from a user-chosen bin width —
# an intentionally small width (e.g. 1 Å) on a wide wavelength range could
# otherwise request thousands of bins, each multiplying total vertex count
# by that many again; same "protect WebGL responsiveness" reasoning as
# MAX_ITEMS's own comment, just for the z-axis instead of the item count.
MAX_WAVE_BINS = 300

# shape="circle" (real fibre aperture — L1 fibre-level, MOS): each item's
# own column through the cube is a real *filled* disc per wavelength bin
# (a `go.Mesh3d` fan-triangulated from a ring of vertices around a real
# fibre-aperture radius, plus its own centre vertex — see
# flux_cube_figure's own circle-branch comment for the full mesh
# construction). Design history, oldest to current: originally a
# scattered ring of disconnected points ("For Fibre level data at 3d I
# see a ring[]e of circular points for each fibre at each sec[t]ion...
# it should be a connected circle not a set of points forming a
# circle"), then a connected but hollow outline, then this — explicit
# follow-up request: "instead of showing circles as fibre[s] in 3d view
# I want filled circles with the same filling as they are already in
# their circular borders... fill up the circles... it makes more
# sense." _RING_POINTS is the polygon's own vertex count (10 — enough to
# read as genuinely circular, not a decagon-ish outline).
_RING_POINTS = 10

# shape="square" (genuinely pixel-based data — IFU spaxel, stacked/
# co-added L1 cube's own WCS grid cell): each (item, wavelength-bin) cell
# is a real, filled, connected cuboid — explicit follow-up request: "is
# it possible tha[t] for 3d data when viewing at 3d level, we have
# cubic[] spa[x]els instead of circular? You can get the x y size of the
# cube from the data based on the sampling and also the z as the high[t]
# of each cubic spaxel from the wavelength cut you apply. It means, all
# these cubic spaxels must[] be connected." A real `go.Mesh3d` (Plotly's
# own solid-surface 3D trace) rather than another ring-of-points illusion
# — there's no equivalent trick for a literal filled box. See
# flux_cube_figure's own square-branch comment for the full geometry.

# The original marker-size default (2.5, later 4.5) read as "very
# separated... very scatery" for the old scattered-point rings — no
# longer relevant to either shape's *main* trace now that both are real
# solid `go.Mesh3d` surfaces (no gaps between adjacent vertices the way
# isolated marker sprites did), but the "selected item" highlight
# overlay (both shapes) still uses a small marker cloud for a simple,
# consistent "this one's currently selected" emphasis, sized here.
_SELECTED_MARKER_SIZE = 5

# Low-signal points fading toward invisible, high-signal points reading
# as solidly opaque — explicit request: "some level of transparency in
# 3D mode so if signal (total flux) is lower [it's] more transparent, so
# the 3d map is not dominated by [low-signal] bins." `scatter3d.marker.
# opacity` (and, confirmed later, `mesh3d.opacity`) turned out to be
# scalar-only properties (confirmed directly — passing an array raises a
# plotly.py ValueError immediately) — two real gl3d/3D-trace API
# limitations caught this project by testing rather than assuming, after
# the plotly_click one.
#
# First attempt baked transparency directly into a literal RGBA colour
# *string* per point (`marker.color=[...144,000 strings...]`), bypassing
# `cmin`/`cmax`/`colorscale` entirely. That rendered correctly but turned
# out to make the whole callback take 7+ seconds on a real ~20,000-spaxel
# IFU dataset — profiled directly (cProfile against real data, not
# guessed) to Plotly.py's own colour-array validator: a *string* array is
# validated one element at a time (regex `fullmatch` + a `dir()` call per
# element — confirmed 576,000+ `dir()` calls for 144,000 points), while a
# *numeric* array of the same size validates in bulk (confirmed directly:
# 0.2s vs 3.4s for an identical 144,000-point trace, string vs numeric).
#
# Fixed by going back to a genuinely *numeric* colour array (`marker.
# color`/`line.color`/`intensity` depending on trace type — all three
# confirmed to accept the same numeric-array + colorscale/cmin/cmax
# mechanism directly, not assumed) and baking the alpha gradient into the
# *colourscale* instead: a small (`_N_ALPHA_STOPS`, not one-per-point)
# set of stops whose colours come from the requested named colourscale
# with an alpha channel blended in, `_MIN_ALPHA` at the low end to
# `_MAX_ALPHA` at the high end — Plotly.js linearly interpolates all four
# RGBA channels between stops client-side (same mechanism it already
# uses for hue), so the browser-visible result is the same continuous
# colour+transparency gradient a per-point string array would give, at a
# small, fixed cost instead of one that scales with point count.
_MIN_ALPHA = 0.05
_MAX_ALPHA = 0.9
_N_ALPHA_STOPS = 32

# RA/Dec (degrees) and wavelength (Angstrom) have no shared physical unit
# to compare directly, so there's no "natural" 1:1:1 aspect ratio at all —
# the scene used to force a perfect visual cube (`aspectmode="cube"`)
# purely for simplicity. Explicit follow-up request: "probably better if
# we scale it along z to show longer along z... I do not know what scale
# is good, but give it a try and tell me." `aspectmode="manual"` +
# `aspectratio` is Plotly's own documented mechanism for an explicit,
# non-cubic bounding-box shape; 2.5 (z drawn 2.5x as "long" as x/y) is a
# first, reasoned attempt — noticeably elongated/depth-reading along
# wavelength without going so thin that orbiting it becomes awkward or
# individual slices get hard to make out. A single named constant, easy
# to retune from this one spot once there's feedback on whether 2.5 reads
# well or wants to be larger/smaller.
_Z_ASPECT_STRETCH = 2.5

# Square-mode cells are genuine flat *plates* (a thin 2D sheet at one z),
# not extruded solid cuboids — explicit follow-up, after live-testing the
# extruded-voxel design this replaced: "I do not like the making a cube
# out of x and y and z... instead of cubes we just have a plate along x
# and y (ra and dec) and for z, then when for example you stack data from
# 3000 to 3100 then we have a plane representing that slice at the middle
# of 3000-3100 which is 3050." Each (item, bin) is drawn at a single
# z = that bin's own real centre wavelength (`bin_centers_wave`), with no
# z-extent at all — the visible gap between consecutive slices this
# project earlier added via a shrink factor is now automatic by
# construction (two flat sheets at different z simply don't touch,
# nothing to explicitly shrink), and this halves both vertex count (4 vs
# 8 per element) and triangle count (2 vs 12) versus the extruded design,
# a genuine performance win alongside the requested visual change.


def _auto_square_radius_deg(ra, dec, cos_dec, fallback=None):
    """Real half-width (degrees) of one square voxel — measured directly
    from the actual loaded item positions, not trusted from any header/
    physical-fibre value. Explicit request: "You can get the x y size of
    the cube from the data based on the sampling."

    This also fixes a real, confirmed bug: for a stacked/co-added L1
    cube, the caller used to pass the single-*fibre*'s own physical
    aperture diameter (a real WEAVE fibre's footprint on the sky) as this
    size — but a stacked cube's "square" cell is the *reconstructed
    cube's* own WCS pixel spacing, a different (and typically much
    smaller) number entirely; using the fibre diameter made every voxel
    several times too large, so neighbouring voxels heavily overlapped
    instead of tiling cleanly — confirmed live: "I guess if you select
    the right spaxel size these should not have overlapped at all as
    they are blocks." Measuring the real spacing directly from the data
    itself sidesteps needing to know which header field (if any) means
    the right thing for a given dataset/shape.

    Uses each item's own nearest-neighbour distance via a
    `scipy.spatial.cKDTree` (RA pre-multiplied by `cos_dec` so the tree
    operates in a locally flat, isotropic degree frame — same convention
    `flux_cube_figure`'s own RA/Dec-offset math already uses elsewhere),
    and takes the *median* across every item, not the minimum — a
    handful of unusually close/duplicate points would otherwise collapse
    every voxel down to an unrealistically tiny size. Falls back to
    `fallback` (or a small fraction of the field's own on-sky spread,
    matching `flux_cube_figure`'s own `fibre_radius_deg=None` fallback)
    whenever fewer than 2 finite points exist to measure a spacing from
    at all.
    """
    n = len(ra)
    if n >= 2:
        from scipy.spatial import cKDTree
        xy = np.column_stack([ra * cos_dec, dec])
        tree = cKDTree(xy)
        dists, _ = tree.query(xy, k=2)
        nn = dists[:, 1]
        nn = nn[np.isfinite(nn) & (nn > 0)]
        if nn.size:
            return float(np.median(nn)) / 2.0
    if fallback is not None:
        return fallback
    span = max(float(np.nanmax(ra) - np.nanmin(ra)) * cos_dec if n else 0.0,
               float(np.nanmax(dec) - np.nanmin(dec)) if n else 0.0, 1e-6)
    return span / 60.0


def flux_cube_figure(ra, dec, wave, flux_matrix, items, *, fibre_radius_deg=None,
                      n_wave_bins=DEFAULT_N_WAVE_BINS, wave_bin_width_angstrom=None,
                      wave_min=None, wave_max=None,
                      colorscale="Jet", shape="circle", depth_mode="slice",
                      scale="linear", vmin=None, vmax=None, transparent=False,
                      selected_item=None, max_items=MAX_ITEMS, height=600, camera=None):
    """Build the 3D flux-cube figure.

    Parameters
    ----------
    ra, dec : array, shape (n_items,)
        Sky position per item (fibre/target/bin) — the same x/y the 2D
        Aladin overlay itself uses, in degrees.
    wave : array, shape (n_wave,)
        Shared wavelength grid every row of `flux_matrix` is sampled on
        (Angstrom, whatever convention the caller's own spectra use —
        only relative spacing matters here, not absolute calibration).
    flux_matrix : array, shape (n_items, n_wave)
        Per-item flux at each wavelength in `wave`. NaNs are treated as
        zero contribution to a bin's sum (`np.nansum`), not propagated.
    items : array, shape (n_items,)
        Per-item ID (APS_ID/BIN_ID) — attached as `customdata` on every
        single vertex in the figure (every ring/box vertex, every
        wavelength bin), so a click *anywhere* in an item's own column
        resolves to the same ID the 2D map's own click handler already
        expects — explicit requirement: "for selection, it must be only
        by selecting from the xy coordinate as it is now."
    fibre_radius_deg : float, optional
        Real angular radius (degrees) of one item's own footprint (WEAVE
        fibre core radius — whatever the caller's "true fibre/spaxel
        size" logic already resolves this to for the same dataset). Used
        only as a *fallback*, for both shapes (see `_auto_square_radius_
        deg`'s own docstring), when fewer than 2 finite items exist to
        measure a real spacing from — the normal case instead measures
        each disc's/plate's own real radius/half-width directly from the
        actual loaded item positions (so two genuinely-adjacent WCS
        pixels' plates touch edge-to-edge with no gap or overlap, and
        `shape="circle"`'s own discs render at a size that's actually
        visible on screen). This literal value is deliberately NOT used
        as-is for `shape="circle"` any more — confirmed directly that a
        real WEAVE fibre's true footprint is routinely ~1/10,000th of a
        typical multi-object field's own on-sky span, meaning a *filled*
        disc at that true size would be sub-pixel and invisible at any
        normal zoom (unlike the outline design this replaced, whose
        visibility came from a fixed-*screen*-pixel line width,
        independent of data-space scale — see the circle branch's own
        comment for the full account). `None` falls back to a small
        fixed fraction of the field's own on-sky spread.
    n_wave_bins : int
        Number of wavelength bins along z — each is a *sum*, not a mean,
        of every native wavelength sample that falls inside it (matching
        how a real integrated-flux narrow-band image would be built).
        Overridden by `wave_bin_width_angstrom` when that's given (see
        its own docstring) — this raw bin-*count* form is kept mainly for
        direct/programmatic callers (and this module's own tests) that
        want an exact bin count regardless of the real wavelength span.
    wave_bin_width_angstrom : float, optional
        Wavelength width (Å) of one bin/cross-section — the user-facing
        control this project settled on in place of a raw bin count,
        explicit follow-up: "I guess it is better if user be able to
        change the width in wavelength range where those bins stacked
        (sum) together to create one cross section... add the width of
        the z axis bin in angstrom to the top of the 3d panel so users
        can set it[,] but give a default one." When given (and positive),
        `n_wave_bins` is derived from it (`round(wave_span /
        wave_bin_width_angstrom)`, clamped to `[1, MAX_WAVE_BINS]` — see
        that constant's own comment) rather than taken as a literal count
        — so the same width produces a comparable number of slices
        regardless of how wide a given dataset's own wavelength coverage
        happens to be, instead of a fixed bin *count* silently meaning a
        very different real width from one dataset to the next. `None`
        (the default) leaves `n_wave_bins` as given/defaulted, unchanged.
    wave_min, wave_max : float, optional
        Restrict the cube to just this wavelength sub-range (Å, same
        convention as `wave`) before any binning happens — a real
        selection of which native samples participate at all, not a
        post-hoc display crop, so a narrow window still gets finely
        binned across just that span rather than reusing the full
        range's coarser bins. Explicit request: "scroll through the
        wavelength range and select a start and end." Either or both
        `None` (the default) leaves that end of the range wide open,
        i.e. the dataset's own full native coverage — unchanged from
        before this was added. Swapped automatically if `wave_min` is
        given larger than `wave_max` rather than silently producing an
        empty cube.
    colorscale : str
        Plotly colorscale name — sampled into a small (`_N_ALPHA_STOPS`)
        set of colourscale stops (see `_MIN_ALPHA`/`_MAX_ALPHA`'s own
        comment for why), not passed to Plotly as a bare name directly,
        since the alpha gradient needs to be blended into those same
        stops.
    scale : "linear", "log", "sqrt", "power", or "asinh"
        The same astronomical-image "stretch" options the 2D Aladin
        overlay's own "Scale" dropdown offers (`apsPlot.style.
        apply_color_scale`) — explicit follow-up request: "in 3d view can
        we have other scaling as we have in 2d like log, sinh, power
        etc?" Implemented as *non-uniform colourscale stop positions*
        (each stop still spans the same `cmin`..`cmax` numeric range and
        the colour array stays the real, genuinely numeric flux value —
        see `_MIN_ALPHA`'s own comment for why that matters for
        performance) rather than remapping the flux values themselves,
        so hover text and the colourbar's own axis both keep showing
        real flux values, not a 0-1 stretched fraction — only *where*
        each colour sits along that real range changes.
    vmin, vmax : float, optional
        Explicit colour range override — `None` (either or both) falls
        back to the 1st/99th percentile of the actual binned flux values
        (the same convention `_l1_active_color_params`/`_aladin_legend_
        info` use for the 2D view). Deliberately *not* wired to the
        shared 2D "Colour range" Min/Max control by the caller
        (`aps_explorer.update_map_mode`) even though `scale` is — that
        control's numbers describe whatever quantity the 2D catalog is
        currently coloured by (e.g. S/N, a handful to a few hundred), a
        different quantity entirely from this figure's own per-bin summed
        flux (routinely 1e5-1e7 on real data), so reusing them here would
        silently saturate/blank the whole cube. Exposed as a real
        parameter anyway (not simply removed) since a future caller with
        a scale-appropriate range in hand should still be able to use it.
    transparent : bool
        `False` (the default) renders every vertex at full, constant
        opacity — explicit report: "when I look at the cross section of
        3d data cubes, I do not see what I usually see in 2d maps...
        probably due to transparency issue" — a 2D map's own points are
        always fully opaque regardless of signal strength, so a 3D
        cross-section faded by signal (the *previous* default) genuinely
        doesn't match what the same data looks like in 2D. `True` restores
        the original low-signal-fades-toward-invisible behaviour (still
        useful for seeing through a dense cube to find bright structure),
        via `_MIN_ALPHA`/`_MAX_ALPHA` exactly as before.
    shape : "circle" or "square"
        `"circle"` draws each item's own column as a real *filled* disc
        (a `go.Mesh3d` fan-triangulated from a ring at its true angular
        radius plus its own centre vertex), one disc per wavelength bin
        — a genuine fibre aperture, for real fibre data (L1 fibre-level,
        MOS). Originally a hollow outline (explicit follow-up: "fill up
        the circles... it makes more sense"). `"square"` draws each (item,
        wavelength-bin) cell as a real, flat *plate* (a `go.Mesh3d`
        quad) sitting at that bin's own centre wavelength — for
        genuinely pixel-based data with no physical aperture at all (an
        IFU spaxel, or a stacked/co-added L1 cube's own WCS grid cell).
        Originally a solid extruded cuboid (explicit request: "is it
        possible tha[t] for 3d data... we have cubic[] spa[x]els instead
        of circular?"); replaced with a flat plate after live testing,
        explicit follow-up: "I do not like the making a cube out of x
        and y and z... instead of cubes we just have a plate along x and
        y (ra and dec) and for z... a plane representing that slice at
        the middle of [the bin's own wavelength range]." `depth_mode`
        below brings the extruded form back as an *opt-in* alternative
        rather than reverting this default.
    depth_mode : "slice" or "cube"
        `"slice"` (the default, unchanged from the shape docstring
        above) draws each (item, bin) with no z-extent at all — a flat
        plate or a flat disc, sitting exactly at that bin's own centre
        wavelength. `"cube"` extrudes the *same* (item, bin) cells to
        genuinely fill their bin's real depth instead — explicit
        request: "I have a radio botton that be select between slice or
        cube. IF slice we show wavelenght slices as thery are..and if
        cube. then instead of a slice, I want you to add a dept to each
        element along z ...for examle if a slice is coverig 3000-3100 I
        want a cube that fill this space." `shape="square"` becomes a
        real solid box (top/bottom/4 side walls — the original
        extruded-cuboid geometry from before the flat-plate switch
        above, reconstructed here as an option rather than the only
        choice); `shape="circle"` becomes a real solid cylinder (top/
        bottom rings + side wall, same fan-triangulated-disc convention
        as the flat "circle" case for each cap). Both stay driven by the
        exact same `bin_centers_wave`/`wave_edges` this function already
        computes for "slice" — the *centre* used for "slice" is just the
        midpoint between the same two edges "cube" extrudes to, so
        switching modes never changes which real wavelength range a
        given element represents, only whether it's drawn with real
        depth or not.
    selected_item : int, optional
        Currently-selected item — drawn as a second, small-marker trace
        (both shapes) so the current selection stays visible while
        rotating/zooming, the same "always show what's selected"
        convention as every other panel in this app.
    max_items : int
        Defensive cap — see MAX_ITEMS's own comment.
    height : int
        Figure height in px.
    camera : dict, optional
        Explicit `scene.camera` (`{"eye": {...}, "up": {...}, "center":
        {...}}`) to render with instead of this figure's own default
        top-down view — explicit report: "when I click on a map[[a
        point]] in 3d view, it reload[s] the image prob[a]bly because it
        want[s] to put that point in the centre... I do not want that any
        click on the ma[p] end up with a reload of the 3d view [...] it
        change[s] the 3d orientation every time i click on a point." Every
        click/selection/colour-range/scale/palette/transparency change
        rebuilds this figure from scratch server-side (a brand-new
        `go.Figure`), and Plotly.js's `Plotly.react()` (what `dcc.Graph`
        calls under the hood on every `figure` prop change) resets the
        camera back to whatever `scene.camera` the new figure specifies
        on *every single one* of those rebuilds unless told otherwise —
        reading exactly like an unwanted "reload/recentre."

        `layout.uirevision` — Plotly's own *documented* mechanism for
        exactly this — was tried first and confirmed NOT to work for this
        `dcc.Graph` in this app's installed Dash/Plotly.js combination
        (4.4.1/3.1.1): reading `dcc.Graph`'s own async-graph.js bundle
        directly shows its `plotly_relayout` handler unconditionally
        writes the *live* `gd.layout[key]` value for every changed,
        non-autosize/width/height layout section (which includes `scene`,
        i.e. the camera) back into the component's own `figure` *prop* —
        this is the exact same "`dcc.Graph` corrupts `uirevision`'s own
        bookkeeping by writing zoomed state back into `figure` on every
        interaction" gotcha this project already hit and fixed a
        different way for the (since-removed) 2D Plotly map. Since that
        write-back happens regardless, the caller doesn't need `uirevision`
        at all — it can just *read back* that same live camera value
        (`aps_explorer.update_map_mode` does this via `State("flux-cube-
        graph", "figure")`, the same "read back the graph's own current
        state and manually reapply it" shape the 2D map's own fix used)
        and pass it straight back in here, so every rebuild explicitly
        re-renders with wherever the user last rotated/zoomed to instead
        of silently snapping back to the hardcoded default. `None` (the
        default, and what a genuinely *new* dataset load passes) uses the
        default top-down `camera=dict(eye=dict(x=0,y=0,z=2.5), ...)`
        below — matching "default orientation is top-down on first switch
        to 3D."

    Returns
    -------
    plotly.graph_objects.Figure
    """
    ra = np.asarray(ra, dtype=np.float64)
    dec = np.asarray(dec, dtype=np.float64)
    wave = np.asarray(wave, dtype=np.float64)
    flux_matrix = np.asarray(flux_matrix, dtype=np.float64)
    # FITS-derived integer columns (BIN_ID/APS_ID straight off
    # fits.getdata()) are commonly big-endian, same well-established
    # gotcha as flux/wave above (both already fixed via the explicit
    # dtype=np.float64 casts, which — like this one — produce native
    # byte order because no explicit endianness character was given) —
    # orjson (Plotly's JSON backend) refuses to serialize a non-native
    # array at all ("numpy array is not native-endianness"), caught
    # directly against a real 20,161-item IFU dataset before this ever
    # reached a test.
    items = np.asarray(items).astype(np.int64)

    finite_items = np.isfinite(ra) & np.isfinite(dec)
    ra, dec, flux_matrix, items = ra[finite_items], dec[finite_items], flux_matrix[finite_items], items[finite_items]
    n_items = len(ra)

    # A real sub-range *selection*, not just a display crop — samples
    # outside [wave_min, wave_max] are dropped before binning even sees
    # them, so a narrow range genuinely produces a few finely-binned
    # slices spanning just that window rather than the same coarse bins
    # as the full range with everything outside faded out. Explicit
    # request: "scroll through the wavelength range and select a start
    # and end" — see aps_explorer.py's own flux-cube-wave-min/-max
    # Inputs for where these come from; `None` (either or both, the
    # default) leaves that end of the range wide open, i.e. the full
    # native coverage, unchanged from before this was added.
    if len(wave) and (wave_min is not None or wave_max is not None):
        lo = wave_min if wave_min is not None else -np.inf
        hi = wave_max if wave_max is not None else np.inf
        if lo > hi:
            lo, hi = hi, lo
        wave_mask = (wave >= lo) & (wave <= hi)
        wave = wave[wave_mask]
        flux_matrix = flux_matrix[:, wave_mask]

    fig = go.Figure()
    if n_items == 0 or flux_matrix.shape[1] == 0:
        fig.update_layout(**style.BASE_LAYOUT, height=height,
                           title="No data available for the flux cube.")
        return fig

    # Evenly subsample items (not the wavelength axis) if there are more
    # than max_items — a real thinning, not a truncation to "the first
    # N," so the subsample still spans the whole field.
    n_finite_items = n_items
    if n_items > max_items:
        keep = np.linspace(0, n_items - 1, max_items).round().astype(int)
        ra, dec, flux_matrix, items = ra[keep], dec[keep], flux_matrix[keep], items[keep]
        n_items = len(ra)
        subsampled = True
    else:
        subsampled = False

    n_wave = len(wave)
    # A real, positive bin *width* (Å) overrides the raw bin count —
    # see wave_bin_width_angstrom's own docstring. Derived count is
    # clamped to MAX_WAVE_BINS (a defensive cap independent of the
    # count-based path's own implicit n_wave cap just below) so an
    # unreasonably small requested width can't blow up vertex count.
    if wave_bin_width_angstrom is not None and wave_bin_width_angstrom > 0 and n_wave >= 2:
        wave_span = float(wave[-1] - wave[0])
        n_wave_bins = max(1, min(MAX_WAVE_BINS, round(wave_span / wave_bin_width_angstrom)))
    n_bins = max(1, min(n_wave_bins, n_wave))
    # Bin by native-sample index (np.array_split), not by physical
    # wavelength value — for a "fun," visually-cubic exploration tool,
    # equal-*sample-count* bins are simpler and cheap to compute
    # (np.add.reduceat) and are close enough to equal-width in practice
    # (WEAVE's own wavelength sampling is close to linear already).
    edges = np.linspace(0, n_wave, n_bins + 1).round().astype(int)
    edges = np.unique(edges)
    n_bins = len(edges) - 1
    # Real wavelength as the z-axis throughout (both shapes) — explicit
    # request for the square/voxel case specifically ("z [should be] the
    # high[t] of each cubic spaxel from the wavelength cut you apply"),
    # applied uniformly to circle too for one consistent z convention:
    # Plotly's own default numeric tick generation now works directly
    # (no tickvals/ticktext translation needed the way a bare bin index
    # required), and both shapes' "selected item" highlight/default-
    # camera logic share the same z values.
    # `edges` are sample *indices* 0..n_wave inclusive (`np.linspace`'s
    # own endpoint is always exactly n_wave) — `wave` only has valid
    # indices 0..n_wave-1, so the final edge can't be used to index it
    # directly (an out-of-bounds IndexError, caught directly against
    # real data before this ever reached a test). Clip for indexing, then
    # extrapolate the true upper boundary one sample-spacing past the
    # last real sample, rather than silently under-sizing the final bin
    # to stop exactly on it.
    wave_edges = wave[np.minimum(edges, n_wave - 1)].astype(np.float64)
    if n_wave >= 2:
        wave_edges[-1] = wave[-1] + (wave[-1] - wave[-2])
    bin_centers_wave = (wave_edges[:-1] + wave_edges[1:]) / 2.0
    # np.nansum per bin, vectorized across every item at once via
    # np.add.reduceat on a nan-zeroed copy (nansum itself has no
    # reduceat equivalent).
    flux_zeroed = np.nan_to_num(flux_matrix, nan=0.0)
    binned = np.add.reduceat(flux_zeroed, edges[:-1], axis=1)
    # add.reduceat's last segment runs to the array's end regardless of
    # the true final edge whenever that edge already equals n_wave (the
    # common case here) — harmless (the true last bin's own edges[-1]
    # already *is* n_wave), so no extra correction needed.

    dec0 = float(np.nanmean(dec)) if n_items else 0.0
    cos_dec = np.cos(np.radians(dec0)) or 1.0
    if fibre_radius_deg is None:
        span = max(float(np.nanmax(ra) - np.nanmin(ra)) * cos_dec if n_items else 0.0,
                    float(np.nanmax(dec) - np.nanmin(dec)) if n_items else 0.0, 1e-6)
        fibre_radius_deg = span / 60.0  # a small, always-sane fallback footprint

    finite_c = np.isfinite(binned)
    if vmin is None:
        vmin = float(np.nanpercentile(binned[finite_c], 1)) if finite_c.any() else 0.0
    if vmax is None:
        vmax = float(np.nanpercentile(binned[finite_c], 99)) if finite_c.any() else 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    # A small (_N_ALPHA_STOPS, not one-per-vertex) colourscale with the
    # requested named colourscale's own colours plus a blended-in alpha
    # gradient — see _MIN_ALPHA/_MAX_ALPHA's own comment for why this
    # replaces per-vertex RGBA colour strings. The colour array passed to
    # each trace below stays genuinely numeric (the real flux values), so
    # Plotly's normal fast bulk-array validation applies, and its own
    # cmin/cmax/colorscale/colorbar machinery works unmodified.
    #
    # `scale` (log/sqrt/power/asinh) is applied to the stops' own
    # *positions* along cmin..cmax, not to the flux values themselves —
    # Plotly.js linearly interpolates colour between whichever two stops
    # bracket a vertex's own (still perfectly linear) cmin/cmax-
    # normalized position, so concentrating stops non-uniformly (e.g.
    # bunched near cmin for "log") produces the same *visual* stretch a
    # value-remapping would, without ever touching the flux values/hover
    # text — those all stay real, honest numbers throughout.
    import plotly.colors as pc
    frac_uniform = np.linspace(0.0, 1.0, _N_ALPHA_STOPS)
    stop_pos = np.clip(style.apply_color_scale(frac_uniform, scale), 0.0, 1.0)
    stop_pos[0], stop_pos[-1] = 0.0, 1.0  # guard float round-off at the ends
    stop_rgb = pc.sample_colorscale(colorscale, frac_uniform.tolist())
    # Full, constant opacity by default — see `transparent`'s own
    # docstring for why (a 2D map's own points are always fully opaque,
    # so a faded-by-signal 3D cross-section didn't match). `transparent=
    # True` restores the original low-signal-fades gradient.
    stop_alpha = (_MIN_ALPHA + (_MAX_ALPHA - _MIN_ALPHA) * frac_uniform if transparent
                  else np.ones_like(frac_uniform))
    alpha_colorscale = [
        [float(p), rgb.replace("rgb(", "rgba(").rstrip(")") + f", {a:.3f})"]
        for p, rgb, a in zip(stop_pos, stop_rgb, stop_alpha)
    ]
    # Horizontal, sitting under the scene — explicit request: "make sure
    # the color bar is the same shape as the 2d map['s legend] which
    # sit[s] at the bottom" (the 2D view's own colourbar is a thin
    # horizontal CSS-gradient bar directly under the map, not Plotly's
    # own default vertical/right-hand colorbar this used before).
    colorbar = dict(title=dict(text="Flux/bin", side="top"), orientation="h",
                     thickness=14, len=0.8, x=0.5, xanchor="center", y=-0.08, yanchor="top")

    if shape == "square":
        # Real, filled, connected flat *plates* — one per (item,
        # wavelength bin), sitting at that bin's own real centre
        # wavelength, not an extruded solid box. Originally a solid
        # cuboid (explicit follow-up request: "is it possible tha[t] for
        # 3d data... we have cubic[] spa[x]els instead of circular? You
        # can get the x y size of the cube from the data based on the
        # sampling and also the z as the high[t] of each cubic spaxel
        # from the wavelength cut you apply"), replaced after live
        # testing per explicit follow-up: "I do not like the making a
        # cube out of x and y and z... instead of cubes we just have a
        # plate along x and y (ra and dec) and for z, then when for
        # example you stack data from 3000 to 3100 then we have a plane
        # representing that slice at the middle of 3000-3100 which is
        # 3050." A genuine `go.Mesh3d` (Plotly's own solid-surface 3D
        # trace — the ring-of-points trick used for "circle" only ever
        # *looked* cylindrical from a distance; there's no equivalent
        # illusion for a literal filled sheet), one flat quad per (item,
        # wavelength bin): x/y half-width = the item's own real angular
        # pixel/spaxel size (when two items are genuinely adjacent WCS
        # pixels this half-width is exactly half their spacing, so
        # neighbouring plates' edges coincide with no gap or overlap),
        # z = `bin_centers_wave[b]` (that bin's own real centre
        # wavelength, shared with the circle branch's own ring-placement
        # convention below) — consecutive bins' plates naturally don't
        # touch (nothing to explicitly shrink; a flat sheet has no
        # z-extent to shrink in the first place), giving the requested
        # "slice" reading automatically. "Unless masked/decimated" is
        # automatic, not a special case: a plate is only ever drawn for
        # an (item, bin) combination that actually exists in the input
        # arrays, so a masked-out area (fewer items) or the item-count
        # subsampling above (fewer items than the real field has) both
        # show up as real gaps, exactly as expected, with no extra logic
        # needed.
        #
        # Mesh3d.opacity turned out to be scalar-only too (confirmed
        # directly, the same limitation already found for Scatter3d.
        # marker.opacity) — `intensity` (numeric, fast — see _MIN_ALPHA's
        # own comment for why a *string* array would be a serious
        # performance trap here, confirmed directly against a real mesh
        # before this was ever written this way) + the same alpha-
        # blended colourscale stops mechanism already established for
        # the line/marker cases sidesteps it the same way.
        #
        # x/y half-width measured directly from the actual loaded item
        # positions — see _auto_square_radius_deg's own docstring for
        # why this replaced trusting `fibre_radius_deg` directly (a real,
        # confirmed overlap bug for stacked L1 cubes).
        r = _auto_square_radius_deg(ra, dec, cos_dec, fallback=fibre_radius_deg)
        # Flat-quad corner order: counter-clockwise viewed from +z (same
        # winding convention the old cuboid's own bottom/top faces used),
        # all four at the *same* z — nothing "top"/"bottom" about it any
        # more, see the fixed triangle-index table below.
        dx4 = np.array([-1.0, 1.0, 1.0, -1.0]) * r / cos_dec
        dy4 = np.array([-1.0, -1.0, 1.0, 1.0]) * r

        if depth_mode == "cube":
            # The extruded-cuboid geometry described (but no longer
            # present in code) by this branch's own docstring history —
            # rebuilt as an opt-in `depth_mode`, not a reversion of the
            # flat-plate default. See `depth_mode`'s own top-of-function
            # docstring for the exact request. 8 vertices/12 triangles
            # per (item, bin), the same count the historical flat-plate
            # switch's own commit message cited as what it *halved* —
            # confirms this reconstruction matches the original design,
            # not just a plausible guess at one.
            z_lo_arr = wave_edges[:-1]
            z_hi_arr = wave_edges[1:]
            dx8 = np.concatenate([dx4, dx4])
            dy8 = np.concatenate([dy4, dy4])
            full_shape8 = (n_items, n_bins, 8)
            box_x = np.broadcast_to(ra[:, None, None] + dx8[None, None, :], full_shape8)
            box_y = np.broadcast_to(dec[:, None, None] + dy8[None, None, :], full_shape8)
            z8 = np.concatenate([
                np.broadcast_to(z_lo_arr[:, None], (n_bins, 4)),
                np.broadcast_to(z_hi_arr[:, None], (n_bins, 4)),
            ], axis=1)
            box_z = np.broadcast_to(z8[None, :, :], full_shape8)
            color_grid = np.broadcast_to(binned[:, :, None], full_shape8)
            item_grid = np.broadcast_to(items[:, None, None], full_shape8)

            x_flat = box_x.ravel()
            y_flat = box_y.ravel()
            z_flat = box_z.ravel()
            color_flat = color_grid.ravel()
            item_flat = item_grid.ravel()
            text = [f"{c:.3g}" for c in color_flat]

            n_plates = n_items * n_bins
            # Local vertex indices 0-3 = bottom face (z_lo, same
            # dx4/dy4 corner order as the flat-plate case above), 4-7 =
            # top face (z_hi, same corner order). Bottom face wound
            # opposite the top face (mirrors the flat plate's own CCW-
            # from-+z convention) so each face's normal points away from
            # the box's interior; the 4 side walls each get their own
            # outward-facing pair.
            tri_i = np.array([0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3])
            tri_j = np.array([2, 3, 5, 6, 1, 5, 2, 6, 3, 7, 0, 4])
            tri_k = np.array([1, 2, 6, 7, 5, 4, 6, 5, 7, 6, 4, 7])
            plate_offset = (np.arange(n_plates) * 8)[:, None]
            i_idx = (plate_offset + tri_i[None, :]).ravel()
            j_idx = (plate_offset + tri_j[None, :]).ravel()
            k_idx = (plate_offset + tri_k[None, :]).ravel()
        else:
            full_shape = (n_items, n_bins, 4)
            plate_x = np.broadcast_to(ra[:, None, None] + dx4[None, None, :], full_shape)
            plate_y = np.broadcast_to(dec[:, None, None] + dy4[None, None, :], full_shape)
            plate_z = np.broadcast_to(bin_centers_wave[None, :, None], full_shape)
            color_grid = np.broadcast_to(binned[:, :, None], full_shape)
            item_grid = np.broadcast_to(items[:, None, None], full_shape)

            x_flat = plate_x.ravel()
            y_flat = plate_y.ravel()
            z_flat = plate_z.ravel()
            color_flat = color_grid.ravel()
            item_flat = item_grid.ravel()
            text = [f"{c:.3g}" for c in color_flat]

            n_plates = n_items * n_bins
            # 2 triangles per flat quad, local vertex indices into that one
            # plate's own 4 corners (see the dx4/dy4 comment above for the
            # corner-order convention this relies on).
            tri_i = np.array([0, 0])
            tri_j = np.array([1, 2])
            tri_k = np.array([2, 3])
            plate_offset = (np.arange(n_plates) * 4)[:, None]
            i_idx = (plate_offset + tri_i[None, :]).ravel()
            j_idx = (plate_offset + tri_j[None, :]).ravel()
            k_idx = (plate_offset + tri_k[None, :]).ravel()

        fig.add_trace(go.Mesh3d(
            x=x_flat, y=y_flat, z=z_flat, i=i_idx, j=j_idx, k=k_idx,
            intensity=color_flat, cmin=vmin, cmax=vmax, colorscale=alpha_colorscale,
            # A flat 1D array (one scalar per vertex) — same convention
            # as the circle branch/this module's original marker design;
            # see aps_explorer.py's own clientside click-handling comment
            # for why click can't just be `Input("flux-cube-graph",
            # "clickData")` the way every other click-driven selection in
            # this app is.
            customdata=item_flat,
            hovertemplate="Item=%{customdata}<br>Flux/bin=%{text}<extra></extra>",
            text=text,
            colorbar=colorbar, name="flux cube",
        ))

        if selected_item is not None:
            sel_mask = items == selected_item
            if sel_mask.any():
                si = int(np.flatnonzero(sel_mask)[0])
                sel_x = np.tile(ra[si] + dx4, n_bins)
                sel_y = np.tile(dec[si] + dy4, n_bins)
                sel_z = np.repeat(bin_centers_wave, 4)
                fig.add_trace(go.Scatter3d(
                    x=sel_x, y=sel_y, z=sel_z, mode="markers",
                    marker=dict(size=_SELECTED_MARKER_SIZE, color="red", opacity=1.0,
                                line=dict(width=1, color="white")),
                    hoverinfo="skip", showlegend=False, name="selected",
                ))
    else:
        # "circle" — a real *filled* disc per (item, bin), a `go.Mesh3d`
        # fan-triangulated from a ring plus its own centre vertex (same
        # "real, solid surface, not a disconnected-points illusion"
        # reasoning "square" mode's own Mesh3d already uses) — explicit
        # follow-up request: "instead of showing circles as fibre[s] in
        # 3d view I want filled circles with the same filling as they
        # are already in their circular borders... fill up the
        # circles... it makes more sense." Superseded design, for
        # context: originally a scattered ring of disconnected points
        # ("it should be a connected circle not a set of points forming
        # a circle"), then a connected but hollow ring outline
        # (`go.Scatter3d(mode="lines")`) — this replaces that
        # outline-only trace outright, not just adds a fill on top.
        #
        # Deliberately NOT drawn at the real, literal `fibre_radius_deg`
        # — confirmed directly (a scratch render at the true MOS fibre
        # radius against real data) that a real WEAVE fibre footprint is
        # roughly 1/10,000th of a typical multi-object field's own
        # on-sky span, meaning a *filled* disc at that true size is
        # genuinely sub-pixel and invisible at any normal zoom (a bare
        # outline stayed visible purely because Plotly's `line.width` is
        # a fixed *screen*-pixel width, independent of data-space scale
        # — filling loses that free visibility, since a Mesh3d surface's
        # on-screen size is purely a function of its real data-space
        # extent). Same tension this app's own 2D Aladin view already
        # solves for "true fibre/spaxel size" (default off, a larger
        # fixed symbol used instead, since true-size markers there are
        # similarly too small to see/click at normal zoom) — the same
        # `_auto_square_radius_deg` "square" mode already uses (real
        # median nearest-neighbour spacing across the actual loaded
        # items, robust to the occasional close/duplicate pair) gives a
        # visually sensible, always-on-screen-visible size here too,
        # falling back to the real `fibre_radius_deg` only when too few
        # items exist to measure a spacing from at all.
        r = _auto_square_radius_deg(ra, dec, cos_dec, fallback=fibre_radius_deg)
        theta = np.linspace(0, 2 * np.pi, _RING_POINTS, endpoint=False)
        ring_dra = (r * np.cos(theta)) / cos_dec
        ring_ddec = r * np.sin(theta)

        if depth_mode == "cube":
            # Real filled cylinder per (item, bin) — the "circle" shape's
            # own counterpart to "square"'s extruded box above, same
            # `depth_mode` request. Vertex layout: `_RING_POINTS` ring
            # points at z_lo (local 0..N-1), the same ring again at z_hi
            # (local N..2N-1), then the bottom centre (2N) and top centre
            # (2N+1) — bottom cap + top cap (each a fan, same convention
            # the flat disc below already uses, bottom's winding mirrored
            # so its normal points down) + a side wall (2 triangles per
            # ring segment) closes it into a real solid.
            n = _RING_POINTS
            z_lo_arr = wave_edges[:-1]
            z_hi_arr = wave_edges[1:]
            n_verts_per_cyl = 2 * n + 2
            dra_ring2 = np.concatenate([ring_dra, ring_dra])
            ddec_ring2 = np.concatenate([ring_ddec, ring_ddec])
            dra_full = np.concatenate([dra_ring2, [0.0, 0.0]])
            ddec_full = np.concatenate([ddec_ring2, [0.0, 0.0]])

            full_shape = (n_items, n_bins, n_verts_per_cyl)
            cyl_ra = np.broadcast_to(ra[:, None, None] + dra_full[None, None, :], full_shape)
            cyl_dec = np.broadcast_to(dec[:, None, None] + ddec_full[None, None, :], full_shape)
            z_ring2 = np.concatenate([
                np.broadcast_to(z_lo_arr[:, None], (n_bins, n)),
                np.broadcast_to(z_hi_arr[:, None], (n_bins, n)),
            ], axis=1)
            z_centres = np.stack([z_lo_arr, z_hi_arr], axis=1)  # (n_bins, 2)
            z_per_bin = np.concatenate([z_ring2, z_centres], axis=1)  # (n_bins, 2n+2)
            cyl_z = np.broadcast_to(z_per_bin[None, :, :], full_shape)
            color_grid = np.broadcast_to(binned[:, :, None], full_shape)
            item_grid = np.broadcast_to(items[:, None, None], full_shape)

            x_flat = cyl_ra.ravel()
            y_flat = cyl_dec.ravel()
            z_flat = cyl_z.ravel()
            color_flat = color_grid.ravel()
            item_flat = item_grid.ravel()
            text = [f"{c:.3g}" for c in color_flat]

            n_cyls = n_items * n_bins
            k_range = np.arange(n)
            k_next = (k_range + 1) % n
            bottom_centre, top_centre = 2 * n, 2 * n + 1
            tri_i = np.concatenate([
                np.full(n, bottom_centre), np.full(n, top_centre), k_range, k_range,
            ])
            tri_j = np.concatenate([
                k_next, n + k_range, k_next, n + k_next,
            ])
            tri_k = np.concatenate([
                k_range, n + k_next, n + k_next, n + k_range,
            ])
            cyl_offset = (np.arange(n_cyls) * n_verts_per_cyl)[:, None]
            i_idx = (cyl_offset + tri_i[None, :]).ravel()
            j_idx = (cyl_offset + tri_j[None, :]).ravel()
            k_idx = (cyl_offset + tri_k[None, :]).ravel()
        else:
            n_verts_per_disc = _RING_POINTS + 1  # ring points + centre
            dra_full = np.concatenate([ring_dra, [0.0]])
            ddec_full = np.concatenate([ring_ddec, [0.0]])

            full_shape = (n_items, n_bins, n_verts_per_disc)
            disc_ra = np.broadcast_to(ra[:, None, None] + dra_full[None, None, :], full_shape)
            disc_dec = np.broadcast_to(dec[:, None, None] + ddec_full[None, None, :], full_shape)
            z_grid = np.broadcast_to(bin_centers_wave[None, :, None], full_shape)
            color_grid = np.broadcast_to(binned[:, :, None], full_shape)
            item_grid = np.broadcast_to(items[:, None, None], full_shape)

            x_flat = disc_ra.ravel()
            y_flat = disc_dec.ravel()
            z_flat = z_grid.ravel()
            color_flat = color_grid.ravel()
            item_flat = item_grid.ravel()
            text = [f"{c:.3g}" for c in color_flat]

            n_discs = n_items * n_bins
            # Fan triangulation: RING_POINTS triangles per disc, each
            # (centre, ring[k], ring[k+1 wrapped]) — same fixed local
            # -vertex-index-table-plus-per-element-offset construction
            # "square" mode's own quad triangulation already uses, just a
            # fan instead of a pair.
            center_local_idx = _RING_POINTS
            tri_i = np.full(_RING_POINTS, center_local_idx)
            tri_j = np.arange(_RING_POINTS)
            tri_k = (np.arange(_RING_POINTS) + 1) % _RING_POINTS
            disc_offset = (np.arange(n_discs) * n_verts_per_disc)[:, None]
            i_idx = (disc_offset + tri_i[None, :]).ravel()
            j_idx = (disc_offset + tri_j[None, :]).ravel()
            k_idx = (disc_offset + tri_k[None, :]).ravel()

        fig.add_trace(go.Mesh3d(
            x=x_flat, y=y_flat, z=z_flat, i=i_idx, j=j_idx, k=k_idx,
            intensity=color_flat, cmin=vmin, cmax=vmax, colorscale=alpha_colorscale,
            # A flat 1D array (one scalar per vertex, including the
            # centre vertex — its own item id is real and reachable,
            # unlike the old outline design's unreachable break/closing
            # vertices) — see aps_explorer.py's own clientside click
            # -handling comment for why click can't just be
            # `Input("flux-cube-graph", "clickData")` the way every
            # other click-driven selection in this app is.
            customdata=item_flat,
            hovertemplate="Item=%{customdata}<br>Flux/bin=%{text}<extra></extra>",
            text=text,
            colorbar=colorbar, name="flux cube",
        ))

        if selected_item is not None:
            sel_mask = items == selected_item
            if sel_mask.any():
                si = int(np.flatnonzero(sel_mask)[0])
                sel_ra = ra[si] + ring_dra
                sel_dec = dec[si] + ring_ddec
                sel_x = np.tile(sel_ra, n_bins)
                sel_y = np.tile(sel_dec, n_bins)
                sel_z = np.repeat(bin_centers_wave, _RING_POINTS)
                fig.add_trace(go.Scatter3d(
                    x=sel_x, y=sel_y, z=sel_z, mode="markers",
                    marker=dict(size=_SELECTED_MARKER_SIZE, color="red", opacity=1.0,
                                line=dict(width=1, color="white")),
                    hoverinfo="skip", showlegend=False, name="selected",
                ))

    title = "3D flux cube — drag to rotate, scroll to zoom, right-drag to pan"
    if subsampled:
        title += f" (showing {n_items:,} of {n_finite_items:,} items)"
    # Explicit request: "add the width of the z axis bin in angstrom to
    # the top of the 3d panel so users can set it[,] but give a default
    # one" — the input control itself lives in aps_explorer.py's own UI
    # (above this graph); this echoes back the *realized* bin width
    # (rounding a requested width to an integer bin count means the two
    # can differ slightly) so what's actually plotted is never ambiguous.
    # A real line break (Plotly title text accepts a little HTML, `<br>`
    # among it) before this clause, not just another " — " continuation
    # — explicit follow-up: "font size for the 3d titke is good but as
    # it is very long better if you break it from where you say for
    # exmapk 55 X 100 A wavelenght bins..and centre align it." Two
    # shorter centred lines read far better than one very long one at
    # this figure's fairly narrow top margin.
    if n_bins >= 1 and n_wave >= 2:
        realized_width = float(wave[-1] - wave[0]) / n_bins
        title += f"<br>{n_bins} × ~{realized_width:.0f} Å wavelength bins"
    # BASE_LAYOUT's own margin/hovermode are both 2D-axis-oriented
    # (hovermode="x unified" has no meaning for a 3D scene) — everything
    # else (template/font) still applies fine to a 3D figure.
    # A fixed z-axis (wavelength) range, spanning this call's own full
    # bin-centre extent with a little padding — explicit reports: "3d
    # plot when I want to play with the scrol through wavelenght range
    # reset the camera to along z mode...I want it does not change or
    # reload the plot and ust scrol through z using the same camera fov
    # I am in" and "When I play with wavelenght range in 3d view, first
    # it reset the camera view... if I am zoom in along z, and change
    # the range, we may lost that slice in the view as the cammera may
    # be outer or inner that one." The *camera itself* was already
    # confirmed correctly preserved across every server rebuild (see
    # `camera`'s own docstring/mechanism above) — the actual cause is
    # one level deeper: the live wavelength-range *scroll* is a purely
    # client-side operation (aps_explorer.py's own clientside callback
    # NaNs out whichever vertices fall outside [Start, End] via
    # Plotly.restyle(), never rebuilding this figure at all — see that
    # callback's own docstring), and with no explicit zaxis range set
    # here, Plotly re-autoranges the z axis to whatever subset of data
    # is still visible after every one of those restyle calls. Camera
    # `eye`/`up`/`center` are defined in the scene's own *normalized*
    # unit-cube coordinates, not real data units, so even an unchanged
    # camera ends up framing a genuinely different apparent view once
    # the axis it's normalized against has silently shifted underneath
    # it — reading exactly like "the camera reset" or "lost the slice
    # I was looking at," even though nothing about the camera itself
    # ever actually changed. Pinning the range once, here, to this
    # call's own full bin-centre span (always the dataset's true full
    # native coverage in the live app — see update_map_mode's own
    # docstring for why it never passes wave_min/wave_max to this
    # function at all, only the purely-client-side filter does) keeps
    # that coordinate frame stable regardless of how much of the cube a
    # later client-side filter hides.
    if n_bins >= 1:
        z_lo = float(bin_centers_wave.min())
        z_hi = float(bin_centers_wave.max())
        z_pad = max((z_hi - z_lo) * 0.04, 1.0)
        zaxis_range = [z_lo - z_pad, z_hi + z_pad]
    else:
        zaxis_range = None

    # x/y (RA/Dec) get the identical treatment, for the identical reason
    # — explicit follow-up report: "when I use the start end for
    # wavelenght range, it keep the view angle of camera as I want but
    # the zoom reset to the default zoom...I want the zoom also stuck to
    # what it is." The wavelength-range client-side filter (see zaxis_
    # range's own comment above) NaNs out a hidden vertex's x/y *and* z
    # together, not just z (aps_explorer.py's own restyle callback sets
    # all three to NaN in the same branch) — so narrowing the range can
    # shrink the x/y extent Plotly autoranges against too, whenever the
    # surviving subset of vertices happens to span less sky than the
    # full item set does, which combined with the fixed aspectratio
    # below reads as a real zoom/scale change even though the camera's
    # own eye/up/center genuinely never moved (confirmed directly,
    # unrelated to this). `r` is whichever shape branch above actually
    # ran (`_auto_square_radius_deg`'s own result, in scope here either
    # way) — the real angular half-width already used to place every
    # element's own edge, so the padded bounding box matches exactly
    # how far a full-range render already extends.
    if n_items > 0:
        xaxis_range = [float(np.nanmin(ra)) - r / cos_dec, float(np.nanmax(ra)) + r / cos_dec]
        yaxis_range = [float(np.nanmin(dec)) - r, float(np.nanmax(dec)) + r]
    else:
        xaxis_range = None
        yaxis_range = None

    base_layout = {k: v for k, v in style.BASE_LAYOUT.items() if k not in ("margin", "hovermode")}
    fig.update_layout(
        **base_layout,
        # Explicit report: "The font on 3d plot saying scrol to zoom and
        # pan and ... on the top of the 3d plot is very large font and
        # overstack the frame...make it much smaller" — Plotly's default
        # title font (unset here previously) is sized independently of,
        # and noticeably larger than, BASE_LAYOUT's own 12px body font;
        # against this figure's fairly short top margin (t=40, below)
        # that meant the (often two-line, once the item-count/bin-width
        # suffixes above are appended) title text ran into the scene
        # itself. Matches the small-annotation-label size (11px) used
        # elsewhere in this codebase (e.g. fwhm.py's arm-name badges).
        # x=0.5/xanchor="center" — explicit follow-up: "...centre align
        # it" — centres both the always-present first line and the now-
        # separate (see title's own `<br>` above) bin-count/width line
        # under it.
        title=dict(text=title, font=dict(size=11), x=0.5, xanchor="center"),
        height=height,
        # t=52 (was 40) — the title is now reliably two lines (the
        # `<br>` above always fires whenever n_bins/n_wave allow it,
        # which is every real render), so the top margin needs to fit
        # both lines without the scene creeping up under them again.
        margin=dict(l=0, r=0, t=52, b=0),
        scene=dict(
            # Full dicts (not the xaxis_title-style bare-string shorthand
            # this used before) — needed to also carry `range`/
            # `autorange` alongside `title` here; plotly.py's magic-
            # underscore expansion of a bare `xaxis_title=`/`yaxis_title=`
            # kwarg and an explicit `xaxis=dict(...)`/`yaxis=dict(...)` in
            # the same call would collide over which one actually defines
            # `scene.xaxis`/`scene.yaxis`. See xaxis_range/yaxis_range's
            # own comment above for why this needs to be fixed at all.
            xaxis=dict(title="RA (deg)", range=xaxis_range, autorange=xaxis_range is None),
            yaxis=dict(title="Dec (deg)", range=yaxis_range, autorange=yaxis_range is None),
            zaxis=dict(title="Wavelength (Å)", range=zaxis_range, autorange=zaxis_range is None),
            # See _Z_ASPECT_STRETCH's own comment — a non-cubic bounding
            # box, z drawn longer than x/y, rather than the previous
            # "cube" aspect (RA/Dec degrees and wavelength Angstrom have
            # no shared unit to compare 1:1 in the first place).
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=_Z_ASPECT_STRETCH),
            # Explicit request: "change the default orientation to the
            # case we see x and y and line of sight along z like a 2d
            # map, and we can rotate then" — looking straight down the
            # wavelength axis (camera positioned high on +z, "up"
            # pointing along +y) makes the very first view of the cube
            # look exactly like the familiar 2D RA/Dec map, wavelength
            # receding directly away from the viewer, before any
            # rotation — same standard `layout.scene.camera` mechanism
            # every Plotly 3D figure supports, not a custom addition.
            # Plotly normalizes each axis to the scene's own unit cube
            # for camera-positioning purposes regardless of the axis's
            # real data range, so this needed no adjustment when z
            # switched from a small integer bin index to real wavelength
            # (now in the thousands of Angstrom).
            #
            # `camera or {...default...}` — see `camera`'s own docstring
            # above for why this (explicitly re-rendering with whatever
            # camera the caller read back from the graph's own live
            # state), not `uirevision`, is what actually keeps the user's
            # current view from resetting on every rebuild in this app's
            # Dash/Plotly.js combination.
            camera=camera or dict(eye=dict(x=0.0, y=0.0, z=2.5), up=dict(x=0.0, y=1.0, z=0.0)),
        ),
    )
    return fig
