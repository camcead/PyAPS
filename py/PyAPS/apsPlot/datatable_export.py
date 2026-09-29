"""
Custom CSV export for every `dash_table.DataTable` in the explorer.

**Why this exists instead of `DataTable`'s own native `export_format`
prop** — that was the first thing tried (zero code needed beyond
setting one kwarg), and was reverted after live investigation: clicking
it deterministically fails in this app with a real browser-side
`ChunkLoadError` —

    Loading chunk 404 failed.
    (missing: .../_dash-component-suites/dash/dash_table/async-export.js)

— confirmed via direct Playwright reproduction against the actual
running server, not a guess. This is the same *class* of bug as two
earlier `ChunkLoadError`s already fixed elsewhere in this app (a
component type's lazily-loaded JS chunk only gets requested once the
client renders a first real instance of it — see `aps_explorer.py`'s
`_datatable_preload_dummy` comment) — except this one didn't respond to
that same fix (a hidden dummy table with `export_format="csv"` set,
present from the very first page load, still didn't trigger an eager
fetch of `async-export.js`; confirmed via a live network trace that only
the button's actual `onClick` ever requests that chunk, and that request
then genuinely hangs — never receives a response, success or failure —
until the browser-side webpack runtime times out and throws), and unlike
those two, this one reproduces 100% deterministically on every attempt,
not as an intermittent first-load race.

Root cause not fully pinned down despite substantial live bisection —
ruled out via direct testing, each in an isolated minimal Dash app using
the *exact* same installed `dash`/`dash_table` version in this same
environment (all worked correctly in isolation): this app's `Dash(...)`
constructor kwargs, its custom `index_string` (including the
`console.error`-forwarding interceptor), a function-based vs static
`app.layout`, multiple simultaneous `DataTable` instances (hidden +
visible), and callback-inserted-after-load vs static-initial-layout
timing. A minimal *isolated* Dash app with the single native
`export_format="csv"` prop works perfectly (confirmed) — something
about this specific, much larger app's full runtime state is the
trigger, not the feature itself or any one identified piece of this
app's own setup.

Given that, rather than keep chasing an exact root cause for a fragile,
lazily-loaded third-party JS mechanism, this sidesteps it entirely:
`csv_export_row()` wraps an already-built `DataTable` with a small,
properly styled "Export CSV" button (matching this app's own established
button visual language — see `aps_explorer.py`'s
`_load_panel_toggle()`/"☰ Load dataset" button) and a `dcc.Download` —
both plain, always-loaded dash-core-components with no separate lazy
chunk of their own, confirmed live to work reliably. This also directly
fixes the second, independent complaint ("very ugly... no proper
alignment... no space around it") about `DataTable`'s own default
export button, which has almost no built-in styling and can't be
cleanly restyled — building the button ourselves gives full control.

The actual data-to-CSV callback (`register_csv_export()`) lives in
`aps_explorer.py` instead of here, since that's the only file that owns
the Dash `app` object — this module only builds layout (pure function,
no app dependency), matching the existing convention that
`aps_l1_preview.py`/`aps_IFUviewer.py`/`aps_MOSviewer.py` are
layout-piece-only library modules with no callbacks of their own.
"""

from __future__ import annotations

from dash import dcc, html

from PyAPS.apsPlot.style import TABLE_HEADER_COLOR

#: Small pill-shaped button matching aps_explorer.py's own established
#: button style (raised-surface background, TABLE_HEADER_COLOR
#: text/border, rounded) — see e.g. the "☰ Load dataset" button — just
#: smaller, since this one sits directly above a table rather than in
#: the page toolbar. "var(--pyaps-paper-raised)" resolves to plain white in
#: light mode (unchanged behavior) but the correct raised-dark surface
#: in dark mode — see py/PyAPS/assets/style.css.
CSV_EXPORT_BUTTON_STYLE = {
    "fontSize": "11px", "fontWeight": "600", "padding": "3px 10px",
    "backgroundColor": "var(--pyaps-paper-raised)", "color": TABLE_HEADER_COLOR,
    "border": f"1px solid {TABLE_HEADER_COLOR}", "borderRadius": "14px",
    "cursor": "pointer",
}


def csv_export_row(table_id, table, label="⬇ Export CSV"):
    """Wrap `table` (an already-built `dash_table.DataTable` whose own
    `id` is the plain string `table_id` — every real table in this app
    already has one, unique across the whole app) with a right-aligned
    "Export CSV" button directly above it (a flex row with its own
    bottom margin, so the button never crowds the table underneath it —
    the exact spacing/alignment complaint the native button had) and a
    matching hidden `dcc.Download`. Pairs with `aps_explorer.py`'s
    `register_csv_export(table_id)`, which wires the actual callback —
    call that once per `table_id` used here, or the button does nothing.
    """
    return html.Div([
        html.Div(
            html.Button(label, id={"type": "csv-export-btn", "table": table_id},
                        n_clicks=0, style=CSV_EXPORT_BUTTON_STYLE),
            style={"display": "flex", "justifyContent": "flex-end", "marginBottom": "6px"},
        ),
        table,
        dcc.Download(id={"type": "csv-download", "table": table_id}),
    ])
