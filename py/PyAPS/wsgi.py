"""
Gunicorn entrypoint for the server/Docker deployment of aps_explorer.py.

Standalone usage (`python aps_explorer.py --infiles ...` or the
`aps-explorer` console-script) never imports this file at all — it goes
straight through `aps_explorer.explorer_worker()`'s own argparse +
`app.run(...)` (the Werkzeug dev server), unaffected by anything here.

This module exists only for the multi-user path: gunicorn imports it
directly (not `aps_explorer.py`'s own `__main__` block, which never
runs), so setting `PYAPS_EXPLORER_MULTI_SESSION` here — before
`aps_explorer` itself is imported and its module-level
`aps_explorer_session` read happens — is the only mode switch this path
has. See `aps_explorer_session.py`'s module docstring for the full
per-session-isolation design this flag turns on.

Run with (see the project Dockerfile for the containerized version of
this same command):

    gunicorn --workers 1 --worker-class gthread --threads 32 --timeout 600 \\
        --bind 0.0.0.0:8080 PyAPS.wsgi:server

`--workers 1` is not a tunable — session state is a plain in-process
dict (aps_explorer_session.py), so more than one worker process would
each hold a *different* copy of it, silently corrupting a single user's
own session depending on which worker load-balancing happened to route
a given request to. `--threads` (concurrency within that one process)
is safe to tune, and deliberately raised well past gunicorn's own
default given a host with real spare CPU/memory to give it (see the
Dockerfile's own comment where this value is set for the reasoning and
the explicit request behind it). `--timeout` must stay comfortably above
the slowest real Load callback — a full LIFU-cube LSF/FWHM cache build
has taken up to ~140s; gunicorn's own default (30s) would kill that
request mid-flight.
"""

import os

os.environ.setdefault("PYAPS_EXPLORER_MULTI_SESSION", "1")

from PyAPS import aps_explorer  # noqa: E402  (must follow the env var default above)

server = aps_explorer.app.server
