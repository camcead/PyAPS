# Testing `aps_explorer.py`

Two complementary ways to check the explorer is working: an automated terminal test suite
(no browser needed), and a manual click-through checklist for the web interface itself.

---

## 1. Terminal tests (`pytest`)

```bash
cd <PYAPS_DIR>
pip install -e ".[dev]"
pytest tests/ -v
```

Some tests exercise the real loading/routing code paths against real WEAVE reference data
(L1 exposures, calibration files, L2 products). Point them at your copy of the data:

```bash
export PYAPS_TEST_DATA=<PYAPS_DATA>     # directory containing L1/, L2/, CAL/, CAT/
export PYAPS_HOME=<PYAPS_DIR>           # only if your checkout is not ~/PyAPS
pytest tests/ -v
```

Tests `skip` (not fail) if a given reference file isn't found - a `skipped` result means
the data isn't where expected, not that something is broken; a `FAILED` result is a
real regression. Tests that need no data (CLI flag registry, session isolation, token
handoff, config directory resolution) always run.

| File | Covers |
|---|---|
| `test_aps_explorer.py` | Explorer loading/routing paths (no Dash server, no browser): L2 schema detection, L1/L2 routing, filename/filter parsing, LSF/FWHM handling, tabs, Aladin-Lite dispatch, colour controls, error forwarding |
| `test_l1_fwhm_lsf.py`, `test_ifu_bin_lsf.py`, `test_spaxel_weighted_lsf.py` | LSF/FWHM interpolators (per-fibre, global, merged-arm, spaxel-weighted) |
| `test_aps_common_args.py` | The shared CLI argument-group registry used by the `aps_*.py` scripts: no script may add or drop a flag unintentionally |
| `test_aps_explorer_session.py` | Per-browser-session isolation behind the multi-session / Docker deployment |
| `test_aps_explorer_auth.py` | The signed-token handoff mechanism behind `--require-weaveor-auth` |
| `test_aps_utils_configdir.py` | `configdir` / `PYAPS_CONFIGDIR` resolution |

To cover another dataset, add a `test_...` function following the existing pattern - each
test loads its own dataset, so they don't depend on each other or on order.

---

## 2. Interface (browser) checklist

### Reach the app

```bash
cd <PYAPS_DIR>/py/PyAPS
python aps_explorer.py --port 8080
```

Open **http://localhost:8080**. If the explorer runs on another machine, forward the port
first: `ssh -L 8080:localhost:8080 <user>@<host>`.

**Always fully restart the server and hard-refresh the browser** (not just reload) before
retesting a fix - this app bakes its client-side JavaScript into the page's initial load,
and every response is served with `Cache-Control: no-store` so a stale browser tab is never
the reason something looks unfixed.

### Testing the weaveOR handoff (`--require-weaveor-auth`)

Off by default — the checklist above (no `--require-weaveor-auth`) is
completely unaffected. To test the authenticated-handoff feature itself,
you stand in for weaveOR (which does the real minting in production) by
hand-minting a test token — see `py/PyAPS/aps_explorer_auth.py`'s own
module docstring for the full design this exercises.

1. Pick a real L1 file and find its own actual survey codes — needed for
   the L1 handoff to pass its defense-in-depth check (`allowed_surveys`
   must cover *every* survey the file actually contains, not just the
   one you care about):
   ```bash
   cd <PYAPS_DIR>/py/PyAPS && source ~/venv/bin/activate
   python3 -c "
   from PyAPS.aps_utils import l1_fileinfo
   info = l1_fileinfo(['<PYAPS_DATA>/L1/20250707/stack_3097460.fit'],
                       catdir='<PYAPS_DATA>/CAT', caldir='<PYAPS_DATA>/CAL')
   print(info['srvys'])
   "
   ```

2. Start the explorer with a test secret and the auth flag (a *test*
   secret — never reuse a real production one for this):
   ```bash
   export PYAPS_EXPLORER_WEAVEOR_SECRET=dev-test-secret-not-for-production
   python aps_explorer.py --require-weaveor-auth --port 8080
   ```

3. In a second terminal (same venv, same secret exported), mint a token
   for that file:
   ```bash
   python3 -c "
   from itsdangerous import URLSafeTimedSerializer
   token = URLSafeTimedSerializer('dev-test-secret-not-for-production',
                                   salt='pyaps-explorer-weaveor-handoff').dumps({
       'user': 'your-test-username',
       'allowed_surveys': ['WS2022A2-002', 'WD'],  # from step 1
       'kind': 'l1',
       'infiles': ['<PYAPS_DATA>/L1/20250707/stack_3097460.fit'],
       'caldir': '<PYAPS_DATA>/CAL',
       'catdir': '<PYAPS_DATA>/CAT',
   })
   print(token)
   "
   ```
   For an L2 dataset instead, use `{'kind': 'l2', 'outpath': ...,
   'headname': ...}` — no `allowed_surveys` check applies there (IFU L2
   products carry no survey field at all to check against; see the
   module docstring for why).

4. Open `http://localhost:8080/?token=<paste the token>` (through the
   same SSH tunnel as usual). Check:
   - The dataset loads, and `?token=...` disappears from the address bar.
   - The toolbar (top-right) reads "Logged in as your-test-username (via
     WeaveOR)".
   - "☰ Load dataset" → try a *different* file → rejected with "You're
     not authorized to load a different dataset here", and the
     originally-loaded dataset is still exactly what's shown (check the
     info panel's own table, not just that the form field still has
     what you typed into it).
   - A tampered token (`?token=garbage`), an expired one (wait past
     `PYAPS_EXPLORER_WEAVEOR_TOKEN_MAX_AGE`, default 300s — or lower it
     for a faster test), or no token at all should all leave the app on
     the load form or show a rejection — never open, unauthenticated
     access.

5. Restart without `--require-weaveor-auth` (or unset
   `PYAPS_EXPLORER_WEAVEOR_SECRET`) to confirm standalone behaviour is
   completely unaffected.

### Current architecture, in brief

There is **one spatial view for all three kinds (L1/MOS/IFU): an embedded
Aladin Lite panel**, not a Plotly figure — a Plotly coordinate/map plot
existed through several earlier rounds and was dropped entirely by explicit
request once the Aladin panel's catalog overlay (points coloured the same
way the map was) made it redundant. Aladin shows:
- A clickable catalog overlay (BIN_ID/APS_ID per point, ~16 discrete colour
  buckets for continuous quantities, exact colours per category for MOS).
- A colour legend underneath the panel — a horizontal CSS-gradient bar with
  min/max labels for continuous quantities (IFU's current map, L1's flux),
  or a row of coloured swatches with labels for MOS's four availability
  categories. Always describes exactly what the catalog overlay above it is
  currently showing — recomputed on every load, maptype switch, and
  colour-range change, off the same `aladin-catalog-data` Store the overlay
  itself redraws from.
- Optional DSS background imagery, **off by default**, toggled via its own
  "Show DSS background" checkbox (not just Aladin's own built-in layers
  control).
- A small info box underneath showing the selected point's ID and RA/Dec.

Selecting a point works both ways: clicking a point in Aladin selects it
(updating the info box, the tabs, the value tables) and recentres the view
on its exact coordinates (the *bin's* centroid for IFU, not an arbitrary
member spaxel); the "Go to bin/target ID" text boxes do the same. There is
no continuous zoom/pan syncing with anything else — recentre-on-select only.

### Checklist

**Loading**
- [ ] Click "Load" on an empty/invalid path → a spinner appears over the
      load form while it's working, and a red error message appears.
- [ ] A dark **Log** panel is visible at the bottom of the page at all
      times, even before anything is loaded, and auto-scrolls to keep the
      latest line visible (but stays where you scrolled it if you've
      scrolled up to read earlier lines).
- [ ] Trigger a load error (e.g. clearly wrong outpath) → the log panel
      shows the full traceback, not just a one-line message.
- [ ] Open the browser console (F12) and force a client-side error (e.g.
      briefly go offline so an Aladin/DSS tile request fails) → a
      corresponding `[browser] [console.error] ...` (or `[uncaught]`/
      `[unhandledrejection]`) line appears in the **Log** panel within a
      couple of seconds — this is what makes a purely client-side failure
      (one that never touches Python at all) visible here instead of only
      in the browser console; repeating the *exact same* error rapidly
      logs it once, not dozens of times.
- [ ] Load via outpath+headname, then separately via a full `_APS.fits`
      path in the "Full L2 filename" field — both work.
- [ ] There is exactly **one** "☰ Load dataset" button (top-left, always
      present, before or after anything is loaded) — no second "Load
      different dataset…" button anywhere (removed as redundant).
- [ ] With a dataset loaded, click "☰ Load dataset" → the current view
      stays visible underneath, and the form shows exactly the values you
      loaded with. Change one field and click Load again → view updates in
      place.
- [ ] Load L1, then a MOS file, then an IFU file, then back to L1 — the
      Aladin panel, tabs, and settings all rebuild correctly at every step,
      no leftover content from the previous kind, no browser console
      errors (F12 → Console).
- [ ] The load form is two `dcc.Tabs`, "L1" and "L2 (IFU / MOS)" — with an
      L2 dataset loaded, "☰ Load dataset" opens back on the **L2** tab (not
      L1); with an L1 dataset loaded, it opens back on **L1**. You can no
      longer press Load/Reload for the wrong kind by mistake because the
      two forms' Run buttons are on separate tabs.
- [ ] With an L1 dataset already loaded, reopen "☰ Load dataset" → every
      field (infiles, caldir/catdir/configdir, filters, area/mask/
      wavelength ranges, every checkbox in "Advanced processing options",
      the LSF-source and IVAR-normalization-mode dropdowns, edge-pixels/
      gap-width/flux-unit/template-sigma0 numbers) shows the value that
      dataset was actually loaded with, not blank/default — change one and
      reload → takes effect.
- [ ] Every directory/file field (infiles, caldir, catdir, configdir, L2
      full filename, L2 outpath) has a small 📁 button next to it. Clicking
      it opens a file-browser modal (Up button, typeable path box, a
      scrollable listing of the server's own filesystem); double... single-
      clicking a directory row navigates into it; clicking "Use this
      directory" (dir fields) or a file row (file fields) writes the choice
      back into the field and closes the modal. For `infiles` specifically,
      browsing a second time **appends** another line rather than replacing
      the first. The field stays freely typeable by hand either way — the
      browser is a shortcut, not the only way in.
- [ ] Example/placeholder text in every input (e.g. the Area field's
      `246.29,40.90,...` example) renders small, grey, and italic — never
      mistakable for a real pre-filled current value, and never overflows
      its own box (check the colour-range "min"/"max" boxes specifically).
- [ ] **"Normalize IVAR across arms" is UNCHECKED by default** on a fresh
      "Load L1 dataset" form (Advanced processing options) — explicit
      request: "By default I do not want the ivar normalisation for the
      PyAPS data explorer... make default to False." (Still available,
      just opt-in now — tick it if you specifically want it.)
- [ ] **"Load 1 of every N spaxels/fibres" field** (Target filters
      section) — explicit request: on a very large IFU/LIFU/MIFU cube
      (tens of thousands of spaxels), "loading all is unnecessary... is
      there a smart way that it only load[s] for example one out of each
      3 or 4... make the cubes much faster... however I do not want that
      change the Spatial." Leave it empty → loads every spaxel, unchanged
      from before. Enter e.g. `4` on a large stacked cube and Load → the
      Log panel shows a line like "Decimating: loading 1 of every 4
      spaxels/fibres (N of M)"; the dataset-info table's own "Decimation"
      row confirms it; the load itself is visibly faster (fewer targets
      for the expensive per-target processing loop to churn through); and
      the field of view/coverage on the map is the **same** as an
      undecimated load of the same file — just visibly less densely
      sampled, not cropped to a smaller region.

**L1-specific**
- [ ] Fill in an infile, leave caldir/catdir blank, load → FWHM tab says
      diagnostics aren't available (no crash). Fill in
      `--caldir <PYAPS_DATA>/CAL` / `--catdir <PYAPS_DATA>/CAT` too
      → FWHM tab now shows real curves.
- [ ] The dataset-info panel (always visible once something's loaded) is a
      real, fixed-height (~220px), independently scrollable table — one row
      per load option (Files, Mode, caldir/catdir/configdir, every
      processing flag including the new advanced ones, LSF source, IVAR
      mode, gap width, flux unit, fibres loaded, …) — not a wall of
      one-after-another text.
- [ ] The "Colour Aladin points by" dropdown (below the map area) offers
      "Total flux (all arms)" (default) and "S/N (mean over arms)" —
      switching it recolours the catalog points and updates the legend's
      label/gradient/numbers to match, and the colour-range Min/Max/Reset
      control (below the legend) now applies to whichever one is active.
- [ ] On a genuinely large IFU stackcube (tens of thousands of spaxels, not
      a fibre-table file) with caldir/catdir set, open the FWHM tab → it
      renders (doesn't crash the server or hang the browser tab
      indefinitely) and, if the fibre count is large enough to trigger
      subsampling, the title says "showing N of M fibres."
- [ ] Check "Load raw/unmodified L1 data" → every other processing-option
      checkbox unchecks itself immediately; flux differs noticeably from a
      normal load of the same file, and the Spectra tab's y-axis label
      switches between `1e-18 erg/s/cm²/Å` and
      `counts (uncalibrated — no sensitivity correction)` accordingly.
- [ ] Area field placeholder reads `246.29,40.90,60.0,60.0,0.0` (arcsec, not
      arcmin); Mask areas similarly, one per line.
- [ ] A **Header** tab sits right after **FWHM** (Spectra / Metadata / FWHM
      / Header) — opening it shows the loaded file's own raw primary FITS
      header as a sortable/filterable Keyword/Value/Comment table (not a
      per-fibre summary — the same header regardless of which fibre is
      selected). On a multi-arm load, one such table per input file, each
      under its own "File: /full/path/..." label.
- [ ] Click the **Metadata** tab several times in a row (ideally as the
      very first thing you do in a fresh browser tab, on a freshly
      restarted server — that's the specific combination that exposed a
      real `ChunkLoadError` bug: L1 is the *only* kind with no
      `dash_table.DataTable` anywhere in its initial layout) → it must
      actually switch to and stay on Metadata, showing a real Key/Value
      table, not silently stay on/revert to Spectra.
- [ ] **Slit Explorer** — a genuinely *separate* panel, below everything
      else in the Aladin column (map, legend, colour-range, DSS/Flip/True-
      size checkboxes, and the APS_ID/RA/Dec info box), not interleaved
      among them; a horizontal rule/gap should make it visually read as
      its own section. L1 only. Load a single-exposure, fibre-level file
      (MOS or MOSLIFU/MOSMIFU — e.g. a plain `single_*.fit`/small MOS
      `stack_*.fit`, not a big multi-thousand-spaxel `stackcube_*.fit`) →
      a row of **squares** appears, x-axis labelled "Slit position
      (NSPEC)", each square the same size, coloured by the same flux/S-N
      the Aladin overlay itself uses — colour is the only thing that
      varies point to point now (an earlier version also scaled marker
      *size* by value; dropped by explicit request after live testing —
      a bigger circle looked like it was sitting between its same-size
      neighbours rather than in line with them). The label "**Slit
      Explorer**" itself renders bold (like the Log panel's own title);
      the description sentence right after it ("fibre position on the
      spectrograph slit...") does **not**.
- [ ] Change the **Scale** dropdown (Linear/Log/Sqrt/Power/Asinh, next to
      the colour-range Min/Max/Reset control) → the Slit Explorer's own
      square colours update to match immediately, not just the Aladin
      overlay's. Same check for typing a new Min/Max or clicking Reset —
      all three used to only redraw the Aladin side, leaving the Slit
      Explorer visibly out of sync with whatever "Colour Aladin points
      by"/Scale/range was actually showing until some unrelated click
      happened to refresh it too.
- [ ] The strip shows a **fixed ~50-NSPEC-wide window** centred on the
      current selection, not all 600-1000+ fibres squashed into one view.
      Click a fibre far away on the slit (in Aladin, or via "Go to
      APS_ID") → the window re-centres on its NSPEC. **Drag inside the
      strip** → it pans smoothly to reveal fibres outside the current
      window. **Scroll the mouse wheel over it** → nothing happens (no
      zoom) — the window width never changes, only its position via
      dragging/panning.
- [ ] Click a fibre in Aladin → a dashed red guide line + triangle marker
      jumps to that fibre's position in the strip, **and** the small text
      box directly below the strip updates with that fibre's NSPEC/flux
      (or S-N)/status/use — an always-visible readout, not something you
      only see by hovering. Click several different targets in a row
      (a real report: this sometimes "dropped the connection" and stopped
      picking up the new NSPEC) — every click should update the strip;
      if one ever doesn't, check the **Log panel** first (the whole
      rebuild is now wrapped in try/except, so a genuine failure logs a
      specific "Slit Explorer error: ..." line instead of silently
      leaving the old fibre's marker in place).
- [ ] Click a dot *in the strip itself* → the same fibre gets selected
      everywhere else (Aladin — which must **recentre on it**, tabs, info
      box) — selection is fully bidirectional both ways.
- [ ] Hovering a dot (still works alongside the always-visible readout
      above) shows APS_ID, NSPEC, the flux/S-N value, and `status`/`use`
      (FIB_STATUS/TARGUSE). A fibre with a non-`'A'` (non-active) status
      has a reddish marker border instead of the default grey one.
- [ ] Load a **stacked/co-added** L1 cube instead (a big `stackcube_*.fit`
      pair, `OBSMODE=LIFU`/`MIFU`) → the Slit Explorer strip is replaced by
      a short italic explanation that it isn't available for this dataset,
      not a broken/empty plot.
- [ ] On that same stacked/co-added load, a **"Show contributing
      exposures"** checkbox appears below Flip/True-size (only for this
      case — not for a fibre-level L1 load, not for L2). Check it → small
      **circles at real fibre size** (same physical aperture as "True
      fibre/spaxel size" mode elsewhere — zoom in/out and confirm they
      genuinely scale with the sky, not a constant screen size) appear
      scattered across the field, one per contributing exposure's own
      fibre — real WEAVE data: check a file whose primary header has more
      than one `PROV####` card, e.g. `stackcube_3117934.fit` (16 of
      them). **Colour must match whatever the main "Colour Aladin points
      by" dropdown is currently set to** (total flux / S-N) — switch that
      dropdown and confirm this overlay's own colours change too, not
      just the main catalog's. (This went through two earlier, rejected
      designs first: fixed colour-per-file was "not distinguishable at
      all... because color has been us[ed] for the flux and SNR", and
      shape-per-file markers were rejected outright — "they cannot
      reflect the real shape of the fibre and its diameter... so they
      are useless." Both explicit live-testing reports, not guesses.)
      **Click one of these small circles** → nothing changes in the
      tabs/info box/selection at all — only clicking the normal (larger,
      brighter) stacked-target points still selects anything; this
      overlay is decoration only, confirming the explicit "the click must
      stay only a feature of the stack, not the single files"
      requirement.
- [ ] The note underneath the checkbox is a real **clickable legend**
      (explicit request: "I want to have the option to select and
      deselect each contributed single file in the plot... a[checkbox]
      next to the name of each single one... so I can manage it by
      myself") — one checkbox per contributing exposure, all checked by
      default, plain text labels (no colour swatch any more — colour is
      data-driven now, not per-file). **Uncheck one file** → only that
      file's circles disappear from the map (everything else stays,
      still coloured the same way); **re-check it** → its circles
      reappear. **Every tick/untick should feel genuinely instant** (well
      under half a second, no server round-trip at all — purely a
      client-side show()/hide() call on that file's own already-built
      overlay) — this went through *two* rounds of real, user-reported
      lag, each root-caused differently: (1) "really slow and laggy to
      plot provinces or even remove them from the plot," a 6-7s lag on
      *every* toggle, traced (profiled against a real 32,490-target
      stacked cube) to per-target colour values being recomputed from
      scratch on every call with no caching — fixed by caching; (2)
      after that fix, still reported "still... very slow and laggy" —
      traced (a live JS-timing profile, `console.time` bracketing the
      actual draw loop, not guessed) to the *client-side* Aladin redraw
      itself: tearing down and rebuilding thousands of real `A.circle`
      shapes on every single toggle measured 1-2 seconds and *grew* on
      each successive toggle (a real, apparently cumulative cost in
      Aladin Lite's own `addOverlay`/`removeLayer`, not this app's own
      computation) — fixed by building every file's own overlay once (a
      real, if multi-second, cost the *first* time the main checkbox is
      turned on) and toggling visibility afterward via Aladin's own
      confirmed-working `overlay.show()`/`.hide()` methods, never
      rebuilding anything again for a mere visibility change. If a toggle
      ever feels sluggish again, check both mechanisms — a server round
      trip reappearing in the browser's network tab points at (1), a
      slow-but-no-network-activity toggle points at (2). If the loaded dataset has two arm files (e.g. red
      + blue), each checklist entry should show **both** filenames
      together (e.g. "single_X.fit (RED) + single_Y.fit (BLUE)"), not
      just the one whose fibre positions were actually read. Uncheck the
      *main* "Show contributing exposures" box → everything disappears
      and the per-file legend checklist itself goes away too (not left
      showing with nothing to control).
- [ ] Load an L2 (MOS or IFU) dataset → no Slit Explorer strip and no
      leftover placeholder text at all (this feature is L1-only).

**L2-specific (MOS and IFU)**
- [ ] The dataset-info panel's filename, "PyAPS v...", and "CPS v..." all
      render **bold** — the rest of that summary line (schema/obsmode/
      resolution/arms/target count/dates) stays plain, so the important
      identifying parameters stand out rather than the whole panel being
      uniformly bold.
- [ ] Load a MOS/fibre-level file → Redrock/Stellar/Galaxy tabs as
      appropriate per target.
- [ ] Load an IFU file → Spectrum/PPXF/Emission/Line-Strength (or RVS/
      FERRE) tabs, plus a **Processing History** tab (target-detection
      table, segmentation map, target-selection/Voronoi-binning diagnostic
      images — doesn't regress back to the previous tab's content).
- [ ] Every tab that plots something (Spectrum, Stellar/PPXF, Emission,
      Line Strength, Redrock, Stellar RVS/FERRE, FWHM) actually shows the
      plot, not a blank area — open the browser console (F12) first and
      confirm there's no `ChunkLoadError: Loading chunk ... failed`. This
      broke every plot-bearing tab for one round when the always-present
      main-map `dcc.Graph` was removed from the initial layout; a hidden
      placeholder `dcc.Graph` now guards against it — should never recur,
      but is the single most severe regression this app has had, so it's
      worth actually checking rather than assuming.
- [ ] Every value table (IFU's Spaxel/Bin-results tables, MOS's Redrock/
      Stellar/Galaxy tables) shows a **Unit** column (or a "(unit)" suffix
      on the column header for IFU's wide spaxel table) pulled straight
      from the FITS file's own `TUNIT` header — blank, not a made-up value,
      for columns the file's writer never gave a unit.
- [ ] IFU's **Processing History** tab: the detected-targets table, the
      target-selection image, and the Voronoi-binning diagnostic image are
      stacked in a single column (not a cramped 2-column grid), each at
      full width, and a small monospace "Source: /full/path/..." line
      appears above the targets table showing exactly which file it came
      from.
- [ ] Every IFU tab (Spectrum/PPXF/Emission/Line Strength/RVS/FERRE/
      **Processing History**) sits inside its own fixed-height (~650px),
      independently-scrollable box — switching tabs never changes the
      overall page height or where the "Bin & Spaxel Data" panel below the
      tabs sits, even on a dataset whose Processing History gallery has
      many diagnostic images (matches how L1's Spectra/FWHM tabs and MOS's
      Redrock/Stellar/Galaxy tabs already behaved before this).

**IFU/MOS settings panel**
- [ ] There is no "Marker colour" dropdown (removed as vestigial once the
      Plotly map, its only consumer, was dropped) and no "AoN threshold"
      input either (removed per explicit user request — "not necessary");
      the Emission tab's line markers still work (solid/bold above 2× the
      threshold, light dotted between 1× and 2×, unmarked below), just at
      a fixed internal default, no longer user-adjustable.
- [ ] IFU's Map control is now **two** dropdowns: "Map category" (Table /
      Stellar (PPXF) / Emission lines / Line Strength / Stellar (RVS) /
      Stellar (FERRE), whichever the dataset actually has, each showing
      its option count) and "Map", scoped to just the selected category —
      on a large real dataset (hundreds of emission-line/line-strength
      quantities) this must stay usably short and searchable, never one
      giant flat list. Switching category picks that category's first
      quantity and recolours the Aladin catalog/legend to match.
- [ ] MOS's settings row has a "Colour Aladin points by" dropdown:
      "Availability" (default — the four Gal/ExGal/Both/Neither swatches,
      no colour-range control shown), "Redshift (best rank)", or "S/N (best
      rank)" (either continuous option shows the same colour-range Min/Max/
      Reset control IFU/L1 use, and the legend switches from swatches to a
      gradient bar). Redshift/S/N values are each target's **rank-0**
      (best-fit) value — not an array/string dump of all Redrock ranks.

**The Aladin panel**
- [ ] It fills the left-hand column at a sensible size (matching where the
      old coordinate map used to sit) — not oversized, not misaligned, not
      spanning the whole page/viewport.
- [ ] Aladin's own built-in fullscreen/"maximize" control is **not**
      present at all (disabled at the source, `showFullscreenControl:
      false`, plus a CSS fallback) — clicking anywhere near where it used
      to sit does nothing, and no text box anywhere on the page ends up
      overlaid/hidden behind an expanded Aladin view.
- [ ] Immediately after clicking a new point/bin on the map (or via "Go to
      bin ID"/"Go to APS_ID"), a spinner appears over the plot area while
      the new tab content loads — it should never be ambiguous whether
      your click registered.
- [ ] On a **fresh load** (before clicking anything), the selected point
      is near the spatial **centre** of the field, not an arbitrary
      corner/edge point.
- [ ] DSS imagery is **hidden** by default on a fresh load — only the
      coloured catalog points are visible.
- [ ] Checking "Show DSS background" makes the sky image appear behind the
      points; unchecking it hides it again. No page reload needed either
      way.
- [ ] Click a point in Aladin → the info box below updates with that
      point's ID and RA/Dec, **plus** CNAME/Survey/Class where the loaded
      kind actually has them (L1/MOS), the tabs (Spectra/Metadata/PPXF/
      etc.) update to match, and the view recentres on that exact point.
      The whole info box renders in **bold** — check it's genuinely eye-
      catching against the page, not just technically bold-weight.
- [ ] For an **IFU** dataset specifically: click two different, clearly
      separated bins in turn — Aladin must recentre on *different*
      coordinates each time (matching each bin's own centroid), not stay
      on the same spot regardless of which bin was clicked.
- [ ] Change the colour-range Min/Max (typing, or the number input's
      spinner arrows) or click Reset → the catalog points' colours update
      to match, without any error or the page reloading.
- [ ] A colour legend is visible directly under the Aladin panel: a
      horizontal gradient bar with min/max numbers for IFU/L1 (matching
      the current Map dropdown's quantity — check the label changes when
      you switch maptype), or four coloured swatches ("Galactic + ExGal" /
      "Galactic only" / "ExGal only" / "Neither") for MOS. The legend's
      label/numbers and the swatch labels all render in a **larger, bold**
      font — meant to be eye-catching, not fine print.
- [ ] Typing a new colour-range Min/Max, or clicking Reset, updates the
      legend's numbers to match — they should never show a different range
      than what the catalog points are actually coloured by.
- [ ] Next to Min/Max/Reset, a "Scale" dropdown offers Linear/Log/Sqrt/
      Power/Asinh — switching it recolours the catalog points (the
      legend's own gradient bar stays a plain linear palette either way,
      matching how DS9/most viewers separate "palette" from "stretch";
      only the min/max *numbers* and which points get which colour
      change). Available for IFU/L1 always, and for MOS whenever its own
      "Colour Aladin points by" isn't "Availability."
- [ ] "Flip" mirrors the Aladin view horizontally.
- [ ] "True fibre/spaxel size" checkbox (next to Flip): off by default. On
      a MOS or fibre-level L1 (single-exposure MOSLIFU/MOSMIFU) dataset,
      checking it draws each point as a real angular-size circle (WEAVE's
      actual fibre core: 1.3″ for MOS/mIFU, 2.6″ for LIFU) instead of the
      fixed-pixel marker — zoom in/out and confirm the circles genuinely
      grow/shrink with the sky (an angular size), not stay a constant
      screen size. On an IFU (L2) dataset, it should use that file's own
      spaxel/pixel scale (`PIXSIZE`) instead. Points must still be
      **clickable** in this mode (click-to-select keeps working — the
      circles are a visual layer on top of the normal, still-present
      click targets, not a replacement for them). On a **stacked** L1
      cube, the checkbox has no visible effect (no single fibre size to
      show there) — confirm it doesn't error.
- [ ] **Several back-to-back clicks anywhere on the Aladin panel** (not
      just in true-size mode — a real report: this kept turning the
      Aladin background white even after the try/catch mitigation below).
      A genuine dual-Aladin-instance race was found and fixed this round
      (`createAndGo` now guards against ever running more than once — see
      the History entry below) — please retest specifically by clicking
      rapidly several times right after a fresh page load/dataset
      load, which is when the race window was open. If it still happens:
      open the browser console (F12) and check the **Log panel** — every
      Aladin call in this code path is wrapped in try/catch with
      `console.error` logging (forwarded into the Log panel by the app's
      own existing error interceptor), so a genuine failure should now
      show up as a specific `"PyAPS: ..."` line there instead of a silent
      blank canvas. Please report back either way (fixed, or what the Log
      panel/console showed) — this one couldn't be fully confirmed fixed
      without live browser access.
- [ ] Zoom/pan *inside* Aladin itself (scroll or drag), then click a
      different point — Aladin recentres on the new point but keeps your
      zoom level (does not reset to the dataset-wide default field of
      view).
- [ ] Scroll-zoom (mouse wheel) works smoothly everywhere in the app (the
      spectra/FWHM/Redrock tab plots too) without jitter or snapping back
      to a default view mid-gesture.
- [ ] Watch the browser tab title for a while with nothing happening — it
      stays "PyAPS Explorer" and does not flicker to "Updating…" on its
      own.

**3D flux cube (all three kinds — L1, MOS, IFU)**
- [ ] Directly above the Aladin panel, a **"2D (coordinate map)" / "3D
      (data cube)"** radio selector is visible, **2D selected by
      default**. Switching to 3D hides the Aladin panel and everything
      specific to it (legend, colour-range, DSS/Flip/True-size, "Show
      contributing exposures") and shows a new 3D scene in the exact same
      slot instead — the info box and Slit Explorer below stay visible in
      both modes.
- [ ] The 3D scene shows a genuine cube of **connected geometry**, not
      scattered points — one column per fibre/target/bin, each drawn as
      a real solid shape stacked through depth (not flat dots) — coloured
      by summed flux per wavelength bin (a colourbar labelled "Flux/bin"
      sits beside it). **Drag inside the plot** → the whole scene rotates
      smoothly (orbit, not pan). **Scroll** → zooms in/out; zoom in far
      enough and the camera genuinely goes *inside* the cube, not just
      closer to it. **Right-click-drag** → pans.
- [ ] **Click on a point anywhere in the cube** (any depth/wavelength
      bin, not just the "front" layer) → the info box, tabs, and Slit
      Explorer (L1) all update to that same fibre/target/bin, exactly as
      a 2D Aladin click would — confirming "for selection, it must be
      only by selecting from the xy coordinate as it is now." This is
      **not** an instant/guaranteed hit the way a 2D click is — Plotly's
      own 3D click support has real quirks (see below); if a click
      doesn't seem to register, try clicking more precisely on a visible
      point rather than empty space between them.
- [ ] **Clicking (including double-clicking) a point does NOT reset your
      current camera angle/zoom** — rotate/zoom to some non-default view
      first, then click (or double-click — several real reports of this
      were specifically double-clicks) a different point: the
      highlighted selection updates, but the view you rotated to stays
      *exactly* where it was, rather than snapping back to the default
      top-down orientation. Explicit report: "when I click on a map[[a
      point]] in 3d view, it reload[s] the image... it still reload[s]
      the whole window every time I click to select a spaxel... very few
      times it tried to save the camera angle... Also by click I mean
      double click to select a point." The same holds for a Colour
      range/Scale/Palette/Transparency/wavelength-bin-width change, or
      flipping to 2D and back to 3D, made while viewing the *same*
      dataset — the camera should stay put through all of those.
      Confirmed live via a scripted mouse-drag-then-double-click
      Playwright test, repeated 5 times in a row on different points:
      the rotated camera's own eye/up/center values compared bit-for-bit
      identical before and after each one.
- [ ] **Double-clicking a point in 3D does NOT do anything beyond
      selecting it** — no camera reset, no "jump to this point" motion.
      Plotly's own `doubleClick` config option (which defaults to
      resetting the 3D camera on double-click) is disabled specifically
      for this graph. Explicit report: "in all coordinate in 2d and 3d
      plots, can you disable the click to centre behaviour... in 3d when
      I double click on a point, it make[s] it cent[]re of the field
      which is annoying."
- [ ] **Double-clicking anywhere in the 2D Aladin panel does NOT recentre
      the view either** — Aladin Lite has its own separate, undocumented
      native double-click-to-recentre behaviour (confirmed by reading its
      own bundled JS directly: a `dblclick` listener on its canvas calls
      `pointTo()`/`setRotation(0)`), intercepted and blocked at the
      source. Zoom/pan to some specific view, double-click anywhere in
      the Aladin panel (on a catalog point or empty sky) → the view
      should not move at all from the double-click itself (a genuine
      catalog-point click still selects that point and may recentre via
      the app's own existing, separate "recentre on selection" logic —
      that part is unaffected and still expected).
- [ ] **...but loading a genuinely NEW dataset while in 3D mode DOES
      reset the camera** to the default top-down view — rotate the cube
      for the currently-loaded dataset, then use "☰ Load dataset" to load
      a *different* file while still in 3D mode: the new dataset's cube
      should render from the default top-down angle, not inherit the
      previous dataset's rotated view. Confirmed live the same way (a
      real second dataset load via Playwright, camera compared back to
      the exact default `eye={x:0,y:0,z:2.5}` values afterward).
- [ ] Select a target/bin in **2D** mode, then switch to **3D** → that
      same item's own column is highlighted (larger, red-outlined
      points) among the rest of the cube, so the current selection is
      never ambiguous after switching modes.
- [ ] Switch to 3D, then back to **2D** → the Aladin panel is still
      fully working (clickable, zoomable, not a blank/broken canvas) —
      switching modes must never destroy the underlying Aladin instance.
- [ ] On a genuinely large dataset (tens of thousands of spaxels/fibres),
      3D mode's title says "showing N of M items" — a defensive
      subsampling cap keeps the scene responsive to rotate/zoom rather
      than trying to render every single one.
- [ ] Switching to 3D mode shows a brief spinner (`dcc.Loading`, same
      style used elsewhere in the app) while the cube builds — a real,
      multi-second wait on a large IFU dataset even after optimisation
      (see the perf note below), not an instant flip.
- [ ] **Transparency tracks signal strength**: points/columns with low
      summed flux in a bin render visibly more transparent (faded) than
      high-flux ones — the cube should read as dominated by the bright
      (opaque) structure, not swamped by low-signal noise at full
      opacity. Explicit request: "so the 3d map is not dominated by [low-
      signal] bins."
- [ ] **Default orientation is top-down**: the very first view when
      switching into 3D mode looks like the familiar 2D RA/Dec map (x/y
      laid out flat, wavelength receding away from the viewer along z) —
      *before* any manual rotation. Dragging then rotates freely from
      that starting point.
- [ ] **Square vs circular cross-sections — both genuinely connected
      geometry, not points arranged to merely suggest a shape**: explicit
      report on the original points-in-a-ring design — "For Fibre level
      data at 3d I see a ring[]e of circular points for each fibre at
      each sec[t]ion... it is wrong... it should be a connected circle
      not a set of points forming a circle." For L1 fibre-level data
      (single-exposure MOS/MOSLIFU/MOSMIFU) and MOS, each column is a
      real closed-loop **line** (Plotly `Scatter3d(mode="lines")`) traced
      around the fibre's true angular radius at each wavelength bin, each
      loop visibly a continuous ring, not a dashed/dotted circle of
      separate dots — zoom in close on one column and confirm the outline
      is unbroken. For IFU and any stacked/co-added L1 cube (no single
      real fibre per position), each (item, bin) is a genuine **flat,
      filled plate** (`Mesh3d`) sitting at that bin's own real centre
      wavelength — originally a solid extruded cuboid (explicit request:
      "is it possible tha[t]... we have cubic spa[x]els instead of
      circular?"), replaced after live testing per explicit follow-up:
      "I do not like the making a cube out of x and y and z... instead of
      cubes we just have a plate along x and y (ra and dec) and for
      z... a plane representing that slice at the middle of [the bin's
      own range]." Zoom into a dense area at the **default top-down
      view** and confirm neighbouring spaxels' plates visibly touch/tile
      edge-to-edge sideways with no gaps or overlap — real x/y size is
      measured directly from the actual loaded spaxel positions'
      own nearest-neighbour spacing (a real, confirmed overlap bug on the
      *previous* header-based sizing: "I guess if you select the right
      spaxel size these should not have overlapped at all as they are
      blocks" — check a stacked/co-added L1 cube specifically, the
      dataset type that bug was reported on). Rotate to a side-on view
      and confirm each wavelength bin reads as its own distinct flat
      layer with genuinely empty space between it and its neighbours
      (automatic now — a flat plate has no depth to shrink or overlap in
      the first place, unlike the earlier extruded-cuboid design).
- [ ] **"Wavelength bin width (Å)" control at the top of the 3D panel**
      — explicit request: "I guess it is better if user be able to
      change the width in wavelength range where those bins stacked
      (sum) together to create one cross section... add the width of the
      z axis bin in angstrom to the top of the 3d panel so users can set
      it[,] but give a default one." A number field defaulting to **100**
      sits directly above the 3D graph (only visible in 3D mode). Typing
      a new value (e.g. 50) and clicking/tabbing away rebuilds the cube
      with visibly more/thinner slices (fewer/thicker for a larger
      value); the plot's own title updates to show the *realized* bin
      count/width (e.g. "47 × ~50 Å wavelength bins") so what's actually
      plotted is never ambiguous versus what's typed in the box. Try
      clearing the field completely and retyping a new number (select-all
      then type is the most natural way to replace a number field) —
      this must work correctly on the very first attempt; a real, found-
      and-fixed Dash bug (see the History entry below) used to silently
      ignore the new value in exactly this scenario.
- [ ] **The 3D scene is visibly "longer" along the wavelength axis, not
      a perfect cube** — explicit follow-up: "probably better if we scale
      it along z to show longer along z... give it a try and tell me."
      Rotate to a 3/4 view (not looking straight down z) and confirm the
      wavelength axis reads as noticeably stretched/elongated compared to
      the RA/Dec plane, giving a real sense of depth through the
      wavelength direction — not a cube-shaped bounding box the way it
      used to be. This is a first attempt at a good stretch factor, so
      report back whether it reads well or should be larger/smaller.
- [ ] **On a fresh browser tab, against an already-loaded dataset**
      (no reload — just open a new tab/window to the running app and
      switch straight to 3D mode): the cube renders correctly, same as
      right after a fresh load. This specific scenario went through two
      rounds of live-diagnosed race bugs, both now fixed: (1) the panel's
      visibility toggle and the figure build used to be two separate Dash
      callbacks sharing one trigger, landing as independent React
      commits — fixed by merging into one callback; (2) that merged
      callback's own automatic mount-time firing (Dash fires every
      callback once at page load using each Input's default value) could
      still land *after* a later, genuine 3D click's response and
      silently revert the container back to hidden — fixed with
      `prevent_initial_call=True` on that callback (safe here since the
      static layout already bakes in the correct 2D-mode defaults with no
      callback needed). If this ever regresses, a blank 3D panel
      specifically on a fresh tab/page-load (while a normal same-page
      2D→3D switch still works) is the signature to look for.
- [ ] **Colour range/Scale control works identically in 2D and 3D
      mode** — it now lives *above* both panels (not just inside the 2D
      one), so switching to 3D doesn't hide it. Changing Scale
      (linear/log/sqrt/power/asinh) while in 3D mode rebuilds the cube
      with a visibly different colour distribution (log/asinh especially
      should visibly brighten/spread out what looked like a small hot
      spot under linear) — the *stretch* is shared with 2D's own control,
      but not the literal Min/Max numbers (those describe a different
      quantity — whatever the 2D map is coloured by — from the cube's own
      per-bin summed flux; the cube keeps computing its own auto range
      regardless of what's typed into Min/Max).
- [ ] **Slices read as a genuinely solid, "picture-like" surface**, not
      isolated scattered dots — explicit follow-up: "the points are very
      separated... shall we use a larger points... so each slice be more
      picture like rather than very scatery." This request is now met by
      the connected-geometry rework above (real closed-loop lines for
      circle mode, real solid tiled flat plates for square mode) rather
      than by larger/denser markers — a dense IFU dataset's square-spaxel
      cross-section should read as one continuous tiled surface with no
      gaps, and a fibre column's outline should read as an unbroken ring,
      not a cloud of dots pretending to be either.
- [ ] **Transparency is OFF by default** — switching to 3D mode shows
      every point at the same solid opacity, so a cross-section through
      the cube reads the same way the 2D map does at that location, not
      washed-out/faded. A **"Transparency by signal (low signal fades)"**
      checkbox sits in the shared Colour range/Scale/Palette control
      block, right next to Scale/Palette below the colour bar (moved here
      this round, explicit request: "Add the transparency by signal
      o[p]tion next to the col[o]r range setting below the col[o]r bar" —
      it used to sit directly above the cube itself) — ticking it
      restores the old low-signal-fades-toward-invisible gradient (useful
      for seeing through a dense cube to find bright structure), unticking
      it goes straight back to full, constant opacity. Present (and
      togglable) even while in 2D mode or while MOS is coloured by its
      categorical "Availability" swatches (no Min/Max/Scale/Palette row
      shown there) — it's a 3D-cube-only setting, independent of
      whatever the 2D catalog's own colour-by mode is. Explicit report
      motivating the toggle itself: "when I look at the cross section of
      3d data cubes, I do not see what I usually see in 2d maps...
      probably due to transparency issue."
- [ ] **Colour palette selector — "Default" / "Blue → Red (Jet)" /
      "Black → White (mono)" / "White → Red (hot)"** — a "Palette"
      dropdown sits right next to Scale (same shared, both-2D-and-3D-
      visible control). Switching to "Black → White (mono)" changes the
      colour of *every* currently-visible map (2D Aladin catalog points,
      the Aladin legend bar, and the 3D cube if that's the active mode)
      to a genuine black-to-white gradient — **low signal must render
      dark/black, high signal white** (not the reverse — a common
      colourmap convention pitfall). Switching to "White → Red (hot)"
      instead shows **low signal as white, high signal as deep red** —
      explicit follow-up: "low signal is white and high signal is
      he[a]t[]ing red... because the background is alwa[y]s white for 3d
      and if I have a pallete that shows low signal with wh[i]te it is
      more be[a]u[t]iful to show." This palette is especially visible in
      3D mode since the scene's own background is white — low-signal
      voxels/rings should nearly disappear into it while high-signal ones
      stand out in red. Switch back to "Default" → every map returns to
      its own normal per-quantity colourscale (Jet for flux, Plasma for
      S/N, etc.) exactly as before this control existed. Explicit
      request (mono palette): "is it possible to... use something that is
      black for low signal and white for high signal... something like
      monochromic data so make it more realistic... Can I have it in all
      maps either 2D or 3D."
- [ ] **3D colourbar is a thin horizontal bar at the bottom, matching the
      2D legend's own shape** — explicit request: "make sure the color
      bar is the same shape as the 2[d] map[']s [legend] which sit[s] at
      the bottom." Switch to 3D mode and confirm the "Flux/bin" colourbar
      is a horizontal strip centred under the cube (not Plotly's default
      tall vertical bar down the right-hand side).
- [ ] **Colour range/Scale/Palette control sits below whichever panel
      (2D or 3D) is currently visible, centred** — explicit request:
      "for both 2d and 3d make sure the col[o]r range and scale and
      pallete are below the color bar and cent[]red." Switch between 2D
      and 3D mode repeatedly and confirm this one shared control block
      relocates to sit directly beneath the active panel each time (never
      floating above it, never stuck in the other mode's old position,
      never duplicated).

  **A real, live-tested perf note, not a guess**: building the cube for a
  large real IFU dataset (~20,000 spaxels, ~144,000 rendered points after
  subsampling) profiled at 7+ seconds server-side before an optimisation
  pass — almost entirely Plotly.py's own per-point colour-string
  validation (setting `marker.color` to one RGBA string per point turns
  out to validate each one individually; a numeric array of the same size
  validates in bulk). Fixed by keeping `marker.color` numeric and baking
  the transparency gradient into a small custom colourscale instead,
  cutting it to ~3 seconds — still a real, multi-second wait on the
  largest datasets (hence the loading spinner above), just no longer
  looking indistinguishable from broken. A follow-up point-density
  increase (for the "picture like" request above) was *also* live-tested
  directly against real IFU data before being kept — an initial attempt
  at that same density genuinely looked stuck/broken in testing, but
  turned out to be the mount-time race bug above (fixed separately), not
  a real rendering-cost problem; re-confirmed via screenshot once that
  fix was in place.

  **A real, live-tested caveat, not a guess**: Plotly.js's own native
  `plotly_click` event does not fire at all for `scatter3d` traces in
  this app's installed Plotly.js version (3.1.1) — confirmed directly
  via real mouse clicks against the running app in a headless browser;
  `plotly_hover` fires correctly for the exact same points, `plotly_click`
  never does. This is a documented Plotly.js gl3d limitation, not a bug
  in this app's own selection wiring. Worked around by tracking the most
  recently *hovered* point client-side and reading that back on a native
  (non-Plotly) DOM click listener, confirmed live to work reliably —
  but if clicking in 3D mode ever seems unresponsive in your own
  browser/Plotly.js version, this is the mechanism to check first, and
  is worth reporting back either way.

**Table/tab styling**
- [ ] Every `dash_table.DataTable` in the app (dataset-info, IFU spaxel/
      bin-results, MOS Redrock/Stellar/Galaxy, IFU Processing History
      targets table, L1 Metadata) has a solid dark-blue header (not plain
      black-on-white text) with sortable/filterable columns still working.
- [ ] Every `dcc.Tabs` bar in the app (the L1/L2 load-form tabs, and each
      kind's main content tabs — Spectrum/PPXF/Emission/…, Redrock/
      Stellar/Galaxy, Spectra/Metadata/FWHM) shares the same modern look:
      a light-grey unselected tab, white selected tab with a dark-blue
      bottom border and bold blue text — not the plain default Dash tab
      styling.

**Spectra plot interactivity**
- [ ] Every single-column plot (L1's Spectra tab Flux/IVAR, L1's FWHM tab,
      MOS's Redrock/RVS/FERRE/PPXF/EMI tabs, IFU's Spectrum/Stellar/
      Emission/LS/RVS/FERRE tabs) fills the **full width** of its tab —
      no empty space down the right side, at any browser window width
      (a fixed 1100px/900px figure width used to leave a gap on wider
      screens).
- [ ] On the Spectra and FWHM tabs specifically (the only ones with two
      stacked plots), there's a clear visible **gap** between the two
      graphs — the second plot's title must never sit on top of or
      overlapping the first plot's x-axis, regardless of how many
      arms/panel-rows either figure has (each graph's box now always
      matches its own figure's real height, so nothing overflows into
      the one below it).
- [ ] Whenever a tab's plot(s) are taller than that tab's fixed-height
      box (check a dataset with several arms on FWHM/Spectra, or
      Redrock's up to-6-panel-row worst case, or a Processing History
      gallery with several diagnostic images), a **scrollbar is visibly
      present the whole time**, not only while actively scrolling/
      hovering — this was a real report on some platforms (macOS/touch-
      device "auto-hide" scrollbar settings especially) where the
      scrollbar was there but invisible until interacted with, easy to
      miss that a plot extended below the visible box at all.
- [ ] Every plot's toolbar (Spectra, FWHM, PPXF/Emission/Line-Strength,
      Redrock, Stellar/Galaxy) is **always visible**, not just on hover,
      and does **not** show the Plotly logo.
- [ ] The toolbar includes shape-drawing tools (draw line/open path/
      rectangle, erase shape) in addition to the usual zoom/pan/box-select/
      lasso-select/autoscale/reset-axes/download-PNG buttons — draw a
      rectangle over part of a spectrum to mark a wavelength range, then
      drag one of its edges to resize it.
- [ ] Toggling "Compare data on hover" (via the toolbar or right-clicking
      the plot) and "Show closest data on hover" both work; toggling spike
      lines (toolbar) shows a line across all subplots at the hovered
      x-position.
- [ ] Double-clicking a plot resets it to the full/auto view.
- [ ] Downloading a plot as PNG (toolbar camera icon) produces a
      higher-resolution image than a bare default export (2x scale).

**Banner and toolbar**
- [ ] A banner strip spans the full width of the page at the very top: the
      WEAVE logo on the left (110px tall), then immediately beside it (not
      centred across the banner) "WEAVE Data Explorer" (28px, bold) /
      "PyAPS v<version>" (17px) left-aligned (the version must match the
      installed package — compare against
      `python -c "import PyAPS; print(PyAPS.__version__)"`), then a
      flexible gap, then the camCEAD logo on the right (95px tall, the
      `camCEAD_main_logo.png` mark — clean white-ish mark, no dark box/
      rectangle around it, reads cleanly against the banner's blue). Both
      logos render cleanly — no white/black box or halo around either.
- [ ] A slim toolbar sits directly under the banner, containing only the
      "☰ Load dataset" button (white pill button, blue border/text).
- [ ] Scroll down a long page (e.g. Processing History open on a dataset
      with several diagnostic images) — the banner and toolbar **stick to
      the top of the viewport** and stay fully visible/usable the whole
      time; at no point does either ever overlap the other or get
      overlapped by page content, at any window width.
- [ ] Opening "☰ Load dataset" still shows the sidebar as a full-height
      overlay that visually covers the banner/toolbar underneath it (the
      sidebar's own `×` still closes it).

---

