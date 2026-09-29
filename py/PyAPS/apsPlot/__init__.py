"""
Framework-agnostic Plotly figure builders shared across PyAPS.

Each submodule exposes pure functions that take fit-result data and return
a `plotly.graph_objects.Figure` — no file I/O, no web framework, no Qt.
Callers decide what to do with the figure: static PNG/PDF export via
`fig.write_image()` for pipeline diagnostics, or live display in a Dash
page / the WEAVE Operational Hub.

Per-module wrappers (e.g. `viz.redrock`) adapt that module's own result
tables into the shared primitives in `viz.spectra`.
"""
