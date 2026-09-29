/* aps_explorer.py UI behaviours that are pure client state and don't
 * need a Dash server round-trip: the light/auto/dark theme toggle, the
 * settings-drawer backdrop's click-to-close and Escape-to-close, and
 * (defined now, inert until a later pass) per-card
 * collapse/expand.
 *
 * Everything below is delegated on `document` rather than attached to
 * individual elements. This isn't stylistic — Dash replaces tab-content
 * (and, less often, other parts of the layout) wholesale via its own
 * Output-driven re-renders, which silently orphans any listener
 * attached directly to a node that later gets swapped out. A single
 * document-level listener has nothing to orphan: it keeps working
 * regardless of how many times Dash re-renders around it.
 */
(function () {
  "use strict";

  var THEME_KEY = "pyaps-explorer-theme";

  function applyTheme(choice) {
    var root = document.documentElement;
    if (choice === "light") root.setAttribute("data-theme", "light");
    else if (choice === "dark") root.setAttribute("data-theme", "dark");
    else root.removeAttribute("data-theme");
    document.querySelectorAll("[data-theme-choice]").forEach(function (b) {
      b.classList.toggle("active", b.dataset.themeChoice === choice);
    });
    // schedulePlotTheming (not a bare themePlotlyFigures()) so a theme
    // click that lands while a figure is still mid-render (e.g. right
    // after switching tabs) gets the same staggered-retry safety net as
    // a freshly-created plot does — see its own comment below.
    schedulePlotTheming();
  }

  /* Plotly figures are static JSON handed to Plotly.js — none of them
   * can see assets/style.css's --pyaps-* custom properties at all (a
   * Python-built `template="plotly_white"` has no way to react to a
   * client-side theme toggle it doesn't know exists). Confirmed live:
   * every figure in this app stayed white-background regardless of
   * page theme, most visibly reported for the Slit Explorer ("the
   * background of the slit explorer for both bright and dark mode is
   * white...adopt the whole plot for each mode... not only the
   * background but also all elements on it") and, at a whole-page
   * level, for L2 previews generally ("the defaul[t] mode for L2
   * preview seems to be still the bright and not dark mode") — L2's
   * default tab is a large, plot-dominated Redrock/spectrum view, so an
   * un-themed white plot there reads as "the page is still light" even
   * though the surrounding chrome (banner/tables/page background) was
   * already correctly dark; L1's default Spectra plot has the exact
   * same un-themed-white-background issue, it's just proportionally
   * smaller on screen and so less visually dominant.
   *
   * Re-themed here instead, after the fact, on every currently-rendered
   * figure via Plotly.relayout()'s `template` — a template only
   * supplies *default* colours (paper/plot background, gridlines, axis
   * lines/ticks, font, legend/hoverlabel background), which this app's
   * figures never set explicitly (confirmed by grep before writing
   * this: no figure builder sets paper_bgcolor/plot_bgcolor/gridcolor
   * anywhere), so nothing here fights with each figure's own real
   * content (traces, titles, axis ranges, shapes, annotations are all
   * untouched).
   *
   * A single `layout.xaxis`/`layout.yaxis` key in a Plotly template
   * applies as the *default* for every numbered axis variant (xaxis2,
   * xaxis3, ...) automatically — this doesn't need to enumerate axis
   * keys per figure (Redrock's 4-stacked-panel figure picks this up the
   * same way with one shared template object).
   *
   * Explicit trace-level colours that happened to be near-black
   * (invisible against a new dark plot background) are handled
   * separately, below (themeTraceColours) — NOT by changing the shared
   * Python constants (apsPlot.style.DATA_COLOR etc.) the way an earlier
   * pass did: those are also the *pipeline's own static diagnostic PNG
   * exporters* (apsPlot.redrock.make_rrplot/RVS/FERRE/PPXF/EMI, all
   * imported by aps_rr.py and friends, always rendered on a plain white
   * background outside this app entirely) — explicit follow-up request:
   * "make sure if you touch any plot, it does not disturb the basic
   * plotting function use[d] by aps modules like aps_exGal or aps_Gal
   * or aps_rr or so." Doing the near-black detection/remap here instead
   * means those Python constants — and every static export built from
   * them — never change at all; only what a live browser actually
   * paints is touched. */
  var DARK_PLOT_LINE_COLOR = "#94a3b8"; // theme-neutral slate, reads clearly on the dark card background below
  var DARK_MASK_BORDER_COLOR = "rgba(148,163,184,0.55)";

  function isDarkTheme(cs) {
    // Luminance of the page's own resolved --pyaps-paper, rather than
    // hardcoding/duplicating a literal hex to compare against — stays
    // correct even if style.css's actual token values change later.
    var m = /^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(cs.getPropertyValue("--pyaps-paper").trim());
    if (!m) return true; // default to dark's own assumption if somehow unparseable
    var r = parseInt(m[1], 16), g = parseInt(m[2], 16), b = parseInt(m[3], 16);
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255 < 0.5;
  }

  // Dims a resolved --pyaps-* colour's own contrast against whatever it
  // sits on, by converting it to an rgba() at less than full alpha,
  // rather than hand-picking a separate, harder-to-keep-in-sync darker
  // hex — explicit design critique: "The plot gridlines are a little
  // assertive. Lower their opacity/contrast by about 20-30%; the data
  // should remain the brightest structured element." 0.72 lands in the
  // middle of that requested range. Reads the *live* colour via
  // parseColorRGB (defined below, hoisted — function declarations in
  // this file's own IIFE are already called ahead of their textual
  // definition elsewhere, e.g. schedulePlotTheming/themePlotlyFigures)
  // rather than a literal duplicate, so this can never drift from
  // whatever --pyaps-line-strong actually resolves to.
  function withAlpha(hexOrRgb, alpha) {
    var rgb = parseColorRGB(hexOrRgb);
    if (!rgb) return hexOrRgb;
    return "rgba(" + rgb[0] + "," + rgb[1] + "," + rgb[2] + "," + alpha + ")";
  }

  // 2D figures — normal theme-following template. Explicit follow-up
  // report on the first pass of this ("everything look very flat and
  // plots are not nice looking in dark mode"): using the *page's own*
  // background (--pyaps-paper) for every plot's paper/plot background
  // too meant a plot never visually separated from the page behind it —
  // no "card" feeling at all, unlike every other panel on the page
  // (which all sit on --pyaps-paper-raised). Dark mode's plot
  // background now uses that same raised tone, with axis lines/ticks
  // (not gridlines — see withAlpha's own comment for why those are
  // separately dimmed) bumped from the subtle --pyaps-line to the more
  // visible --pyaps-line-strong to match. Light mode is unchanged from
  // the first pass — never reported as a problem, so left alone.
  function currentPlotlyTemplate(cs, dark) {
    var v = function (name) { return cs.getPropertyValue(name).trim(); };
    var paper = dark ? v("--pyaps-paper-raised") : v("--pyaps-paper");
    var ink = v("--pyaps-ink");
    var line = dark ? v("--pyaps-line-strong") : v("--pyaps-line");
    var lineStrong = v("--pyaps-line-strong");
    var raised = v("--pyaps-paper-raised");
    var axis2d = {
      gridcolor: dark ? withAlpha(lineStrong, 0.72) : line,
      linecolor: lineStrong, zerolinecolor: lineStrong, tickcolor: lineStrong,
    };
    return {
      layout: {
        paper_bgcolor: paper,
        plot_bgcolor: paper,
        font: { color: ink },
        xaxis: axis2d,
        yaxis: axis2d,
        legend: { bgcolor: raised, bordercolor: lineStrong, font: { color: ink } },
        hoverlabel: { bgcolor: raised, font: { color: ink } },
      },
    };
  }

  // 3D flux-cube scene — explicit request: "The background for 3d plot
  // I want to be always white regardless of the theme...It makes things
  // more visible." Deliberately a fixed, non-reactive template (plain
  // literals, not --pyaps-* reads) — this one is NOT supposed to follow
  // the page theme at all, in either direction, unlike every other
  // figure in the app.
  var FIXED_LIGHT_3D_TEMPLATE = {
    layout: {
      paper_bgcolor: "#ffffff",
      font: { color: "#1a2733" },
      scene: {
        xaxis: { gridcolor: "#dbe2e8", linecolor: "#c7d3de", zerolinecolor: "#c7d3de", backgroundcolor: "#ffffff", color: "#1a2733" },
        yaxis: { gridcolor: "#dbe2e8", linecolor: "#c7d3de", zerolinecolor: "#c7d3de", backgroundcolor: "#ffffff", color: "#1a2733" },
        zaxis: { gridcolor: "#dbe2e8", linecolor: "#c7d3de", zerolinecolor: "#c7d3de", backgroundcolor: "#ffffff", color: "#1a2733" },
      },
    },
  };

  function isScene3D(gd) {
    return !!((gd.layout && gd.layout.scene) || (gd._fullLayout && gd._fullLayout.scene));
  }

  // Parses "#rrggbb", "rgb(...)", "rgba(...)" and the literal "black" —
  // the handful of forms this app's own figure builders actually use
  // for near-black trace/marker-border colours (confirmed by grep
  // before writing this, not guessed).
  function parseColorRGB(c) {
    if (!c || typeof c !== "string") return null;
    if (c === "black") return [0, 0, 0];
    var m = /^#([0-9a-f]{6})$/i.exec(c);
    if (m) {
      var n = parseInt(m[1], 16);
      return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
    }
    m = /^rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)/i.exec(c);
    if (m) return [parseFloat(m[1]), parseFloat(m[2]), parseFloat(m[3])];
    return null;
  }

  function isNearBlack(c) {
    var rgb = parseColorRGB(c);
    if (!rgb) return false;
    return (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255 < 0.22;
  }

  // Dark-mode-only trace-colour visibility fix — explicit reports: "I
  // want plots with spectra and fits be more visible in both dark and
  // bright themes...Right now they are not very clear" and (further
  // back) the Slit Explorer's active-fibre marker border being "barely
  // there" on a dark background. Detects near-black `line.color`/
  // `marker.line.color` generically (by actual colour, not by trace
  // name or which figure builder produced it) rather than needing to
  // know that Spectra/Redrock/RVS/FERRE/PPXF/EMI's shared "Data" trace or
  // Slit Explorer's own marker border exist at all — the same mechanism
  // covers any current or future figure with the same pattern.
  //
  // Pristine colours are cached per graph div (gd.__pyapsPristineTraces,
  // refreshed whenever trace count changes, i.e. only on a genuine new
  // figure) so light mode can always restore the exact original rather
  // than drifting after repeated dark<->light toggles — same "cache the
  // real, unfiltered original once, always compute the current state
  // from *that*" shape as the pre-existing flux-cube wavelength-range
  // pristine cache (aps_explorer.py's own clientside callback).
  function themeTraceColours(gd, dark) {
    var traces = gd._fullData || gd.data;
    if (!traces || !traces.length) return;
    var freshCache = !gd.__pyapsPristineTraces || gd.__pyapsPristineTraces.length !== traces.length;
    if (freshCache) {
      gd.__pyapsPristineTraces = traces.map(function (tr) {
        return {
          line: tr.line && tr.line.color,
          markerLine: tr.marker && tr.marker.line && tr.marker.line.color,
        };
      });
    }
    // Idempotency guard — same load-bearing reason as themePlotlyFigures'
    // own (see its comment): Plotly.restyle's internal DOM work can
    // itself trigger the class-attribute mutation plotObserver watches
    // for, which would otherwise re-invoke this on every one of its own
    // calls forever. Skipping once this exact theme has already been
    // applied to this exact (still-fresh) trace set breaks that after
    // one real restyle. A brand new figure (freshCache true) always
    // re-applies regardless, since it's real, unstyled data.
    if (!freshCache && gd.__pyapsTraceThemeApplied === dark) return;
    var lineIdx = [], lineVal = [], mlIdx = [], mlVal = [];
    gd.__pyapsPristineTraces.forEach(function (orig, i) {
      if (orig.line && isNearBlack(orig.line)) {
        lineIdx.push(i);
        lineVal.push(dark ? DARK_PLOT_LINE_COLOR : orig.line);
      }
      if (orig.markerLine && isNearBlack(orig.markerLine)) {
        mlIdx.push(i);
        mlVal.push(dark ? DARK_MASK_BORDER_COLOR : orig.markerLine);
      }
    });
    try {
      if (lineIdx.length) Plotly.restyle(gd, { "line.color": lineVal }, lineIdx);
      if (mlIdx.length) Plotly.restyle(gd, { "marker.line.color": mlVal }, mlIdx);
    } catch (err) { /* best-effort */ }
    gd.__pyapsTraceThemeApplied = dark;
  }

  // Dark-mode-only masked/highlighted-region visibility fix — same
  // report as themeTraceColours ("the color for masked regions and so"
  // not very clear). apsPlot.spectra's mask/highlight shading
  // (_vrect/_shade_regions) is drawn as low-opacity (0.12-0.15) rect
  // shapes — deliberately subtle against the old plain-white
  // background, but reads as barely-there against the new, slightly
  // brighter dark card background. Same pristine-cache-per-gd shape as
  // themeTraceColours, applied to `layout.shapes[].opacity` via
  // Plotly.relayout (shapes aren't trace data, so restyle doesn't apply
  // to them) rather than changing apsPlot.spectra's own shared opacity
  // constants, for the identical "don't touch the pipeline's static
  // exports" reason.
  function themeMaskShapes(gd, dark) {
    var shapes = (gd._fullLayout && gd._fullLayout.shapes) || (gd.layout && gd.layout.shapes);
    if (!shapes || !shapes.length) return;
    var freshCache = !gd.__pyapsPristineShapeOpacity || gd.__pyapsPristineShapeOpacity.length !== shapes.length;
    if (freshCache) {
      gd.__pyapsPristineShapeOpacity = shapes.map(function (s) { return s.opacity; });
    }
    // Idempotency guard — see themeTraceColours' own comment for why
    // this is load-bearing, not optional (a real, previously-hit
    // infinite-loop risk, not a hypothetical one).
    if (!freshCache && gd.__pyapsShapeThemeApplied === dark) return;
    var update = {};
    var changed = false;
    gd.__pyapsPristineShapeOpacity.forEach(function (orig, i) {
      if (typeof orig !== "number" || orig <= 0 || orig > 0.2) return; // only the subtle mask/highlight fills
      update["shapes[" + i + "].opacity"] = dark ? Math.min(0.5, orig * 2.2) : orig;
      changed = true;
    });
    if (changed) {
      try { Plotly.relayout(gd, update); } catch (err) { /* best-effort */ }
    }
    gd.__pyapsShapeThemeApplied = dark;
  }

  function themePlotlyFigures() {
    if (!window.Plotly) return;
    var cs = getComputedStyle(document.documentElement);
    var dark = isDarkTheme(cs);
    var template2d = currentPlotlyTemplate(cs, dark);
    document.querySelectorAll(".js-plotly-plot").forEach(function (gd) {
      // .layout is only present once Plotly.newPlot has actually
      // finished initializing this div — a plot mid-render shouldn't
      // block theming the rest of the page (best-effort, see the catch
      // below too).
      if (!gd.layout) return;
      var scene3d = isScene3D(gd);
      var template = scene3d ? FIXED_LIGHT_3D_TEMPLATE : template2d;
      var targetPaper = template.layout.paper_bgcolor;
      // Idempotency guard — load-bearing, not just an optimization.
      // Plotly.relayout()'s own internal DOM work touches attributes on
      // the .js-plotly-plot container itself while redrawing, which
      // plotObserver below (watching class mutations, to catch a plot
      // that finishes initializing slowly — see its own comment) picks
      // right back up; without this check that reliably built an
      // infinite relayout loop that pegged a CPU core and froze the tab
      // (confirmed live: even trivial calls like counting DOM nodes
      // stopped responding at all once this was live). Skipping once
      // the figure already carries the target paper colour breaks the
      // loop after exactly one real relayout, since the observer's
      // callback always runs as a separate, later microtask — never
      // synchronously inside the relayout call that triggered it — so
      // _fullLayout is already updated by the time this check runs
      // again.
      if (!(gd._fullLayout && gd._fullLayout.paper_bgcolor === targetPaper)) {
        try {
          Plotly.relayout(gd, { template: template });
        } catch (err) { /* best-effort styling only — never break the page over it */ }
      }
      // The 3D scene stays fixed-light always (see FIXED_LIGHT_3D_TEMPLATE's
      // own comment) — none of the dark-mode-only trace/shape visibility
      // adjustments below apply to it, since its own near-black-on-white
      // traces (e.g. the flux cube has none currently, but this keeps the
      // rule general) were never a problem there in the first place.
      if (!scene3d) {
        themeTraceColours(gd, dark);
        themeMaskShapes(gd, dark);
      }
    });
  }

  // Dash builds/replaces dcc.Graph output via its own Plotly.newPlot
  // calls (tab switches, a new fibre/spaxel selection, a fresh dataset
  // load) — each one is a brand-new .js-plotly-plot node the theming
  // above hasn't seen yet. Delegated via MutationObserver for the same
  // reason every other listener in this file is delegated on
  // `document`/`document.body` rather than attached per-element (see
  // the module docstring): individually-created graph divs can't be
  // subscribed to ahead of time. Plotly's own internal relayout/restyle
  // DOM writes (including the ones themePlotlyFigures() above just
  // made) never add a *new* .js-plotly-plot-classed node — that class
  // is set once by newPlot on the outermost div — so this can't
  // feedback-loop on its own re-theming.
  //
  // Confirmed live (L2's Redrock tab specifically — its own real,
  // pre-existing "very slow to render client-side" cost, see the
  // separate FWHM/Redrock investigation in project_pyaps_explorer_
  // redesign.md) that watching childList alone misses it: for a figure
  // this slow to build, react-plotly.js/Dash's Graph component first
  // mounts a plain, not-yet-"js-plotly-plot"-classed div (a childList
  // mutation my observer *does* see, but the class check on it
  // correctly finds nothing yet, so nothing schedules a re-theme), and
  // only *after* Plotly.newPlot finishes its slow internal work does
  // that div's className get rewritten to add "js-plotly-plot" — a
  // separate `attributes` mutation on an *already-present* node, which
  // childList-only observation can never see. Smaller/faster figures
  // (confirmed: the flux-cube placeholder, the hidden preload-warm-up
  // graph) happened to already carry the class by the time they're
  // inserted, which is why this only ever showed up on the one
  // genuinely slow figure rather than every plot in the app.
  //
  // Also keeps the staggered-retry safety net from the first attempt at
  // this fix: `gd.layout`/`gd._fullLayout` scaffolding can exist
  // *before* Plotly's own initial draw is actually done (confirmed via
  // a real "Calling _doPlot as if redrawing but this container doesn't
  // yet have a plot" console warning from Plotly.js itself when
  // relayout was attempted too early) — themePlotlyFigures()'s
  // `if (!gd.layout) return` guard doesn't catch that half-ready state.
  var PLOT_THEME_RETRY_DELAYS_MS = [50, 150, 400, 900, 1800];

  function schedulePlotTheming() {
    themePlotlyFigures();
    PLOT_THEME_RETRY_DELAYS_MS.forEach(function (ms) {
      setTimeout(themePlotlyFigures, ms);
    });
  }

  var plotObserver = new MutationObserver(function (mutations) {
    for (var i = 0; i < mutations.length; i++) {
      var mut = mutations[i];
      if (mut.type === "attributes") {
        var t = mut.target;
        if (t.nodeType === 1 && t.classList && t.classList.contains("js-plotly-plot")) {
          schedulePlotTheming();
          return;
        }
        continue;
      }
      var added = mut.addedNodes;
      for (var j = 0; j < added.length; j++) {
        var node = added[j];
        if (node.nodeType !== 1) continue;
        if ((node.classList && node.classList.contains("js-plotly-plot")) ||
            (node.querySelector && node.querySelector(".js-plotly-plot"))) {
          schedulePlotTheming();
          return;
        }
      }
    }
  });
  plotObserver.observe(document.body, {
    childList: true, subtree: true, attributes: true, attributeFilter: ["class"],
  });

  // The banner (and its theme-toggle buttons) is part of Dash's initial
  // layout, hydrated client-side by dash-renderer — this script can run
  // before that hydration has produced the actual DOM nodes. Poll
  // briefly rather than assume an ordering that isn't guaranteed; give
  // up after 2s (40 * 50ms) so a genuinely missing banner (e.g. a
  // future layout change) doesn't spin forever.
  // "dark" (not "auto") is the fallback for a visitor with nothing
  // saved yet — explicit request (2026-08-16) to make dark mode the
  // default, not just "whatever the OS prefers." Mirrors
  // assets/style.css's own base :root now holding the dark values
  // (light became the override, triggered by an explicit choice or an
  // OS light preference) — the two are meant to always agree.
  var themeInitTries = 0;
  var themeInitTimer = setInterval(function () {
    themeInitTries += 1;
    if (document.querySelector("[data-theme-choice]") || themeInitTries > 40) {
      clearInterval(themeInitTimer);
      applyTheme(localStorage.getItem(THEME_KEY) || "dark");
    }
  }, 50);

  // Keeps the toggle's own .active button in sync if the user is on
  // "auto" and their OS theme flips while the page is already open.
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function () {
    var saved = localStorage.getItem(THEME_KEY) || "dark";
    if (saved === "auto") applyTheme(saved);
  });

  document.addEventListener("click", function (e) {
    var themeBtn = e.target.closest("[data-theme-choice]");
    if (themeBtn) {
      var choice = themeBtn.dataset.themeChoice;
      localStorage.setItem(THEME_KEY, choice);
      applyTheme(choice);
      return;
    }

    // Backdrop click closes the drawer by clicking the real close
    // button, so it goes through the same server callback
    // (on_close_load_panel) as every other close path rather than
    // duplicating that logic here.
    if (e.target.id === "load-panel-backdrop") {
      var closeBtn = document.getElementById("close-load-panel");
      if (closeBtn) closeBtn.click();
      return;
    }

    // Card collapse/expand — matches nothing yet (no .card-classed
    // content exists in any tab yet), see the module docstring above.
    var head = e.target.closest(".card-head");
    if (head && !e.target.closest(".plot-toolbar")) {
      var card = head.closest(".card");
      if (card) card.classList.toggle("collapsed");
    }
  });

  document.addEventListener("keydown", function (e) {
    if (e.key !== "Escape") return;
    var panel = document.getElementById("load-panel");
    var closeBtn = document.getElementById("close-load-panel");
    if (!panel || !closeBtn) return;
    // Only fire while the drawer is actually open, so Escape doesn't
    // steal focus from something else (e.g. the path-browser modal)
    // while the drawer is closed. Checking for the absence of the
    // closed-state's -100% offset (rather than an exact match against
    // the open value) sidesteps any browser-specific formatting of the
    // zero value in "translateX(0)".
    if (panel.style.transform.indexOf("-100%") === -1) {
      closeBtn.click();
    }
  });
})();
