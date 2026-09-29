# aps_explorer — Complete Documentation

**The single interactive Dash web app for viewing WEAVE L1 (raw/reduced per-fibre
spectra) and L2 (`_APS.fits` analysis products) data.** No need to pick the right
script by hand: give it a dataset — via CLI args or the in-browser "Load dataset"
form — and it detects which of L1, L2 MOS/fibre-level, or L2 IFU Voronoi-patch it is
and routes to the matching view automatically.

This document is the complete feature reference. For install/quick-start, see the
main [README.md](../README.md#-aps_explorer--unified-l12-explorer).

## Table of Contents

1. [Architecture](#architecture)
2. [Loading a dataset](#loading-a-dataset)
3. [The spatial view (Aladin panel)](#the-spatial-view-aladin-panel)
4. [2D vs 3D mode](#2d-vs-3d-mode)
5. [The 3D flux cube](#the-3d-flux-cube)
6. [Tabs, per dataset kind](#tabs-per-dataset-kind)
7. [Additional map panels (L2 IFU/MOS)](#additional-map-panels-l2-ifumos)
8. [The Slit Explorer (L1 only)](#the-slit-explorer-l1-only)
9. [Value tables and CSV export](#value-tables-and-csv-export)
10. [The live log panel](#the-live-log-panel)
11. [Server / multi-user deployment](#server--multi-user-deployment)
12. [Environment variable reference](#environment-variable-reference)
13. [Known limitations / experimental features](#known-limitations--experimental-features)

---

## Architecture

`aps_explorer.py` owns the one real Dash `app` and every callback; it holds no
data-loading or figure-building logic of its own. `aps_l1_preview.py`,
`aps_IFUviewer.py`, and `aps_MOSviewer.py` — each once a standalone PyQt5/Dash app in
its own right — are imported purely as libraries now (their own `AppState`/`STATE`
singletons, loaders, figure builders, and layout-piece/callback-logic functions are
reused directly, undecorated, no `Dash()` app or CLI of their own left). Where a
concern is genuinely shared across all three kinds (the map, the tabs, ID-driven
selection) there is one set of component IDs and one callback here that branches on
which kind is loaded; where a kind produces structurally different panels (settings
row, value tables, the load form itself) the shared container's *children* are
swapped wholesale, reusing each library's own layout-piece function unchanged.

Dataset-kind detection for L2 uses `aps_utils.aps_file_info()`'s minimal-I/O,
headers-only inspection (`CLASS_TABLE` present → MOS/fibre-level; `PATCH_TABLE` +
`GALAXY_TABLE`/`STAR_TABLE` → IFU ExGal/Gal) — never the filename or headname.

## Loading a dataset

A single "☰ Dataset & Settings" button (always present, top-left banner) opens an
in-browser sidebar covering both L1 and L2 loading. It remembers whatever values
were last used and lets you tweak one parameter and reload in place, without
restarting the process or losing your browser tab.

**L1 section** — every `aps_l1_preview` CLI option is exposed: `--infiles`/
`--infiles_list`, `--caldir`/`--catdir` (for LSF/FWHM diagnostics),
`--l1ref`/`--l2ref`, target-selection filters (`--apsids`/`--targsrvy`/`--targclass`/
`--maskids`), spatial/wavelength masking (`--area`/`--mask_areas`/`--wlranges`),
`--arms_ratio`, processing flags (`sens_corr`, `safe_mask_gaps`, `mask_gaps`,
`tellurics`, `join_arms`, `vacuum`, `fill_gap`), advanced flags (LSF type, IVAR
normalisation mode, edge-pixel masking, gap-offset, flux-unit scale, template
sigma₀, Galactic extinction correction — see below), and a one-click **"Load
raw/unmodified L1 data"** toggle that forces off every processing flag above
regardless of what the individual checkboxes say.

**Galactic extinction correction** (`extinction_corr`, under "Advanced
processing options", off by default): the same opt-in SFD98+Fitzpatrick99
per-target dereddening validated for REDROCK and default-on for ExGal — see
`doc/aps_rr.md`'s "Galactic Extinction Correction" section for the method,
the real validation numbers, and why full strength (`ebv_scale=1.0`, not
separately exposed here) isn't automatically the right choice for every
target class. Ticking it and reloading applies the real correction through
the same `APSOB(extinction_corr=True)` code path the pipeline itself uses —
not a separate display-only approximation — so the flux/ivar shown are
genuinely dereddened, not just visually adjusted.

**L2 section** — either `--outpath`+`--headname`, or a full `_APS.fits` path
pasted directly.

Both sections' path fields have a **📁 browse button** next to them: a small
in-browser server-side directory listing (not a native OS file picker — this app
typically runs on a remote server, so the browser's own local filesystem is the
wrong machine entirely). Click it, navigate, click a file/directory to fill the
field — the field stays freely editable by hand either way.

Once a dataset is loading, a **full-page white loading overlay** covers everything
except the log panel (see [below](#the-live-log-panel)), so it's visible that
something is happening instead of the page looking frozen — shown from the instant
"Load / Reload" is clicked (both L1's and L2's own separate buttons wire this
independently, since Dash's `dcc.Tabs` only mounts whichever tab is currently
active), hidden again the moment the new dataset is actually ready. A client-side
30-second safety timer clears it even if a load fails outright, so a failed load
never leaves the page stuck white.

## The spatial view (Aladin panel)

All three dataset kinds share one spatial "click-to-select" view: an embedded
**Aladin Lite** sky panel (not a Plotly figure) showing:

- A clickable catalog overlay — one point per spaxel/fibre/target, coloured by
  whichever quantity is currently selected in "Colour by" (discrete colour buckets
  approximating a continuous colourmap for IFU/L1's own flux/S-N choices, exact
  per-category colours for MOS's Class-type/redshift/S-N choices).
- A colour legend underneath the panel — a gradient bar with min/max labels for
  continuous quantities, discrete swatches for MOS's categorical "Class type".
- **DSS background imagery** — real Digitized Sky Survey (`P/DSS2/color`) tiles
  behind the catalog points, giving real sky context instead of a blank field.
  **On by default**; its own checkbox toggles it off per session if not wanted.
- **"True fibre/spaxel size"** — off by default (a fixed-pixel marker stays
  comfortably clickable at any zoom); when checked, each catalog point is drawn at
  its real angular footprint instead (WEAVE fibre core diameter for MOS/L1
  fibre-level data, the IFU pixel/spaxel scale for IFU) — the scientifically
  meaningful view, but individual points can render very small at typical zoom.
- An info box showing the currently-selected point's ID and RA/Dec.
- A **"Go to bin/target ID"** box — types straight to a selection, same effect as
  clicking that point on the map.
- **"Show contributing exposures"** (L1 stacked/superstacked/cube loads only — not
  meaningful for fibre-level data, hidden otherwise) — overlays every single
  exposure that went into the stack, each with its own checkbox to show/hide
  individually; purely client-side once built, no server round trip per toggle.

Clicking a point selects it — updating every tab and value table, and recentring
the view on its exact coordinates.

Colour range (min/max) and stretch (**Linear / Sqrt / Log / Power / Asinh** — the
same "astronomical image stretch" family `astropy.visualization`/DS9 offer) and
palette are all adjustable below the panel, and apply identically whether looking
at the 2D map or the [3D flux cube](#the-3d-flux-cube).

## 2D vs 3D mode

A radio toggle at the top of the spatial-view column switches between the 2D
Aladin coordinate map (default) and a genuine, rotatable/zoomable **3D flux cube**
— real 3D since Aladin Lite itself has no 3D mode. Switching to 3D for the first
time on a given dataset builds the cube lazily (a real, if usually sub-second,
cost nobody should pay for unless they ask for it); switching back to 2D just
hides it, no rebuild needed to switch back again.

## The 3D flux cube

x/y are the same sky coordinates the 2D map uses; z is real wavelength; colour is
the summed flux within each (item, wavelength-bin) cell.

- **Shape**: fibre data (L1 fibre-level, MOS) renders as a real **filled disc**
  per item per wavelength bin (a `go.Mesh3d` fan-triangulated from a ring plus its
  own centre vertex); pixel-based data (IFU spaxels, a stacked/co-added L1 cube's
  own WCS grid) renders as a real **filled flat plate** at each bin's own centre
  wavelength. Disc/plate size is measured directly from the real spacing between
  neighbouring items in the loaded dataset, not the literal physical fibre
  aperture or pixel scale — a real WEAVE fibre footprint is routinely ~1/10,000th
  of a typical multi-object field's own on-sky span, so filling at the true
  physical size would render as genuinely sub-pixel and invisible at normal zoom.
- **Wavelength bin width (Å)**: how many native wavelength samples get summed into
  one z-slice — a user-set width (not a raw bin count), so the same width means a
  comparable slice count regardless of how wide a given dataset's own wavelength
  coverage happens to be. A default is provided; edit and the cube rebuilds.
- **Wavelength range (Start – End)**: restricts the cube to a sub-range of the
  full native coverage. The server always builds the *full* range once per genuine
  rebuild (dataset load, new selection, colour/scale/bin-width change); scrolling
  the range itself is **entirely client-side** — every vertex already carries its
  own real wavelength, so narrowing/widening/shifting the window just shows/hides
  already-downloaded geometry in the browser, with zero server round trip and no
  re-render delay. Prefilled with the dataset's own real full coverage on every
  fresh load.
  - **◀ / ▶** buttons scroll the whole window back/forward by one bin width.
  - **"Lock range width to bin width"** ties End to Start + the bin-width field
    automatically, so scrolling through fixed-width windows is just editing Start.
- **Camera position** (rotation/zoom/pan) survives every rebuild caused by a
  click/colour-range/scale/palette/transparency/2D↔3D-toggle change — captured
  directly off Plotly's own `plotly_relayout` event, not reset to the default
  top-down view until a genuinely *different* dataset loads.
- **Transparency by signal** — off by default (full, constant opacity, matching
  how the 2D map's own points always look); when on, low-signal bins fade toward
  invisible so the cube isn't dominated by low-signal noise.
- **"Enable click-to-select in 3D view"** — off by default (experimental; can be
  slow on large datasets, since resolving a click to an item requires hit-testing
  many points client-side). Selecting by clicking a point on the 2D map, or the
  "Go to ID" box, always works regardless of this toggle.
- A defensive item-count cap (evenly subsampled, not truncated, so the visible
  subsample still spans the whole field) and a wavelength-bin-count cap both exist
  to keep very large datasets responsive in-browser.

## Tabs, per dataset kind

- **L1**: **Spectra** (flux + inverse-variance panels per arm), **FWHM** (a
  dataset-wide overview plot — every fibre's own FWHM curve faintly overlaid, the
  selected fibre's highlighted, up to a capped number of "cloud" fibres for
  large datasets — plus a per-fibre detail plot of the raw arc-line/twilight
  measurements the interpolator was built from), **Header** (primary FITS header,
  one section per input file this dataset was loaded from). FWHM figures are
  built once per target and cached (`_build_fwhm_figures`) — revisiting an
  already-viewed target's FWHM tab is instant, not a rebuild, since the
  underlying cloud/curve data is fixed for the life of one loaded dataset.
- **L2 MOS/fibre-level**: **Redrock** (all fitted ranks, always present — every
  target has a classification/redshift attempt), plus whichever of **Stellar
  (RVS)**, **Stellar (FERRE)**, **Galaxy (PPXF)**, **Galaxy (EMI)** actually have
  results for the currently-selected target (a single target commonly has both a
  stellar and a galaxy result at once — tabs for both are shown, not a forced
  either/or).
- **L2 IFU Voronoi-patch**: **Spectrum** (raw per-bin spectrum), plus whichever of
  **Stellar (PPXF)**, **Emission** (with an AoN-threshold-filtered line overlay),
  **Line Strength**, **Stellar (RVS)**, **Stellar (FERRE)** apply to this
  dataset's own schema (ExGal datasets get PPXF/Emission/Line-Strength; Gal
  datasets get RVS/FERRE — never both, per the underlying pipeline's own two
  mutually-exclusive `_APS.fits` flavours), and always a **Processing History**
  tab — the pipeline-only target-detection table, segmentation map, and
  target-selection/Voronoi-binning diagnostic images that can't be regenerated
  live from the final merged `_APS.fits` (unlike the per-bin fit tabs, which can).

Every tab-content area is held to the same fixed on-page footprint via a
scrollable box (`pyaps-scrollbox`, an always-visible custom scrollbar rather than
an OS overlay one that can be invisible except while actively scrolling) — a tab
whose real content is short just doesn't fill it; one whose content is taller
scrolls inside it rather than pushing the rest of the page down.

## Additional map panels (L2 IFU/MOS)

Below the master map + spectrum row, IFU and MOS datasets (not L1 — there's no
equivalent concept for per-fibre raw data) get an optional grid of extra 2D
quantity maps, each fully independent of the master Aladin view above it. Nothing
shows until you ask for it: **Rows** and **Columns** dropdowns both default to
`0`, and setting either one away from `0` automatically bumps the other up to `1`
if it's still `0` — so a single dropdown change is enough to see a panel. The
grid caps at 3 rows × 6 columns (18 panels).

Each panel has its own, fully independent controls:

- **Map** — any quantity: for IFU, the same large set of per-bin/per-spaxel
  quantities the main map itself can show (a single flat, searchable dropdown
  here, not the main map's own two-level Category+Map split — this feature's
  own panels are meant to stay simple); for MOS, the same "Colour by" list
  (redshift, S/N, etc.).
- **Scale** — Linear/Sqrt/Log/Power/Asinh, the same stretch family the main map
  and 3D cube use.
- **Palette** and **Min/Max range** — independent per panel, so several panels
  can show the same quantity with deliberately different colour ranges, or
  completely different quantities side by side.

Each panel is a plain Plotly scatter map, not an embedded Aladin instance —
deliberately no DSS/sky-background imagery here: an Aladin instance is a
separate WebGL context plus live CDS tile-server traffic, and multiplying that
by up to 18 simultaneous panels would be a real, avoidable cost these panels
don't need (they complement the master map's own sky context rather than
replacing it). There's likewise no transparency-by-signal toggle or BIN-ID info
panel underneath these — kept deliberately simpler than the master map.

**Clicking a point in any panel selects it exactly like clicking the master
map** — every other open panel's own highlight marker, the master Aladin view,
the tabs, and the value tables all update to match, and selecting a point
anywhere else in the app (the master map, "Go to ID", another panel) moves
every open panel's own marker in turn. This sync is entirely client-side (a
lookup cached per panel plus a `Plotly.restyle()` call, not a server round
trip or a figure rebuild), so it stays reasonably responsive regardless of how
many panels are open; at the grid's own maximum size (18 panels) the update is
spread across a few browser animation frames rather than done all at once, so
the page keeps responding to input while it ripples through instead of
freezing. No panel — nor opening more of them, including from several
concurrent server-mode sessions at once — ever re-reads anything from disk;
every panel is built purely from the same in-memory dataset the master map
already loaded.

**Save layout** / **Load layout** save and restore the whole panel
configuration — grid size, and each panel's own Map/Scale/Palette/Min/Max — as
a small JSON file. Clicking **Save layout** prompts for a filename at that
moment (the closest a web page can get to a native "Save As" dialog — there's
no way for a browser page to offer a real filesystem save-location picker);
cancelling the prompt produces no download. **Load layout** uploads a
previously-saved file and rebuilds the grid to match, after checking its
structure (rows/cols in range, panels a real list, each panel's own fields the
right type) and showing a specific error next to the button instead of
silently producing a broken grid for anything that fails that check. A saved
file describes only the *view* — grid shape and each panel's own display
settings — never any data, so it can be freely reused against a different
dataset of the same kind (IFU or MOS); a saved Map that doesn't exist in
whatever is currently loaded falls back to that panel's first available option
rather than erroring. This works identically in standalone single-user mode
and in a shared server deployment — it's a plain browser download/upload, with
no server-side file storage or extra CLI flag involved either way.

## The Slit Explorer (L1 only)

A compact strip showing every fibre's own position on the spectrograph slit
(NSPEC) — click a point to select that fibre (same effect as clicking it on the
Aladin map), or drag to scroll along the slit. Sits directly above the tabs.

## Value tables and CSV export

Every table anywhere in the app — L1 header/metadata, L2 value tables (MOS
Class/Stellar/Galaxy parameters, IFU Spaxel/Bin data), the top dataset-info panel
— has its own **⬇ Export CSV** button (directly above the table) that saves it
exactly as currently shown: respects whatever sorting/filtering/hidden columns are
active in that specific table at the moment of export.

For L1/Raw target tables specifically (occasionally hundreds to thousands of
rows), pagination is client-side: the full target list is sent once and paged
through in the browser, rather than the server re-querying per page.

## The live log panel

A `Pre`-formatted panel at the bottom of the page tees every `print()`/traceback
from the loading pipeline (APSOB, calibration, LSF/FWHM building, etc.) into the
browser live, polled every 700ms — otherwise long loads (well over a minute for
some LIFU cubes) give zero visible feedback in a plain terminal-less deployment.
Server-mode deployments (see below) collapse any absolute filesystem path down to
just its basename before a line ever reaches the browser, so a shared server never
leaks its own directory layout to a session that didn't set it. Stays above the
full-page loading overlay (a higher stacking order) so it's readable throughout a
load, not just before/after.

## Server / multi-user deployment

See the main README's own
[Server / multi-user deployment](../README.md#server--multi-user-deployment-docker),
[Sharing one host across several projects](../README.md#sharing-one-host-across-several-projects),
and
[Authenticated handoff from a trusted upstream app](../README.md#authenticated-handoff-from-a-trusted-upstream-app)
sections for the full Docker/session-isolation/idle-timeout/weaveOR-handoff design
— reproduced here only as a environment-variable quick reference below, so this
document alone is enough to look up what a given deployment flag does.

## Environment variable reference

| Variable | Default | Effect |
|---|---|---|
| `PYAPS_EXPLORER_MULTI_SESSION` | off | Each browser gets its own isolated session (dataset, selection, log) instead of one shared global. Implied by `PYAPS_EXPLORER_REQUIRE_WEAVEOR_AUTH`. |
| `PYAPS_EXPLORER_URL_PREFIX` | `/` (unset) | Mounts the whole app under a sub-path (e.g. `/weave/`) instead of the host root — for sharing one reverse-proxied host across several tools. `/healthz` always stays unprefixed. |
| `PYAPS_EXPLORER_REQUIRE_WEAVEOR_AUTH` | off | Only a signed token from a trusted upstream app (e.g. weaveOR) may load a dataset; the loaded session is then locked to that one dataset for the rest of the visit. Requires `PYAPS_EXPLORER_WEAVEOR_SECRET`. |
| `PYAPS_EXPLORER_WEAVEOR_SECRET` | — | HMAC secret shared with the token-issuing app (`itsdangerous.URLSafeTimedSerializer`). Server refuses to start under `REQUIRE_WEAVEOR_AUTH` without it. |
| `PYAPS_EXPLORER_WEAVEOR_TOKEN_MAX_AGE` | `300` (seconds) | How long a minted handoff token stays valid before the explorer rejects it as expired. |
| `PYAPS_EXPLORER_WEAVEOR_URL` | unset | Where "back to the upstream app" links in a locked session point; unset shows an explanation without a redirect. |
| `PYAPS_EXPLORER_DEFAULT_CALDIR` / `_DEFAULT_CATDIR` | — | Deployment-wide fallback `caldir`/`catdir` used when a load doesn't specify its own — lets a shared server hide these paths from a locked session's own load form while still resolving LSF/calibration files correctly. |
| `PYAPS_EXPLORER_IDLE_TIMEOUT_MINUTES` | `10` | Minutes of genuine client-side inactivity (no mouse/keyboard/click/scroll) before a session's server-side memory is freed. |
| `PYAPS_EXPLORER_SESSION_TTL_SECONDS` | `14400` (4h) | Passive backstop session lifetime for the rarer case that client-side JS never runs at all (disabled JS, a tab killed outright). |
| `PYAPS_CONFIGDIR` | — | Real, writable, persistent config/cache directory — needed whenever the repo-bundled `configs/ExGal_configs` isn't shipped with a deployment. Also where LSF/FWHM interpolator cache pickles end up. |

## Known limitations / experimental features

- **3D click-to-select** is opt-in and can be slow on large datasets (see above) —
  prefer the 2D map or "Go to ID" for routine selection.
- **MOS Redrock loading**, previously reported as unusually slow/unstable, had two
  real, independent causes found and fixed: a Plotly.py performance trap in the
  masked-region-shading code (O(N²) in the number of shaded regions, fixed by
  batching), and a one-time Plotly import/template-resolution cost that always
  landed on whichever tab renders first in a freshly-started server process
  (fixed by warming it up at server startup instead of on a real user's first
  click). Both server-side load and revisit-caching are otherwise expected to be
  fast; a client-side render taking noticeably longer than ~1s for a very
  large-population dataset is more likely genuine browser/WebGL rendering cost
  for a large number of traces than a server-side issue.
