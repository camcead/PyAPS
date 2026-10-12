"""Worker process of ``PyAPS.apsPlot.export``: renders figures to image files, one request at a time.

It is started by the parent (``python _export_worker.py``) in its own session, so the parent can kill it and every
browser process below it when a render does not come back. It imports nothing from PyAPS (only the standard library and
Plotly), reads one JSON request per line on stdin and answers one JSON line on stdout:

    {"cmd": "render", "id": 7, "fig": "<figure json>", "path": "<file>", "kwargs": {"scale": 2}}
        -> {"id": 7, "ok": true}   or   {"id": 7, "ok": false, "error": "<text>"}
    {"cmd": "ping", "id": 0}     -> {"id": 0, "ok": true}      (renders a tiny figure: proves the browser works)
    {"cmd": "quit"}              -> the worker ends

One Kaleido server (one headless browser) serves all requests, which is what makes the worker faster than starting a
browser per image. The image is written under a temporary name next to the target and renamed when complete, so a
render that is killed half way never leaves a truncated picture under the real name.

``SIGUSR1`` makes the worker print the stack of all its threads to stderr (the parent sends it before it kills a worker
that did not answer, so the log shows where it was stuck).
"""
import faulthandler
import json
import os
import signal
import sys


def _reply(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _render(request):
    import plotly.io as pio

    fig = pio.from_json(request["fig"])
    path = request["path"]
    stem, ext = os.path.splitext(path)
    tmp = f"{stem}.part{ext}"
    kwargs = dict(request.get("kwargs") or {})
    try:
        fig.write_image(tmp, **kwargs)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _ping():
    import plotly.graph_objects as go
    import plotly.io as pio

    fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1]))
    pio.to_image(fig, format="png", width=40, height=40)


def main():
    faulthandler.register(signal.SIGUSR1, all_threads=True, file=sys.stderr)
    try:
        import kaleido
    except Exception as e:                      # reported on the first request
        kaleido = None
        startup_error = f"kaleido cannot be imported: {e}"
    else:
        startup_error = None
        try:
            kaleido.Kaleido()                   # raises when no Chrome is installed (the server thread would die silently)
            kaleido.start_sync_server(silence_warnings=True)
        except Exception as e:
            startup_error = f"the Kaleido server did not start: {e}"
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except ValueError as e:
            _reply({"id": None, "ok": False, "error": f"bad request: {e}"})
            continue
        cmd = request.get("cmd")
        if cmd == "quit":
            break
        if startup_error:
            _reply({"id": request.get("id"), "ok": False, "error": startup_error, "fatal": True})
            continue
        try:
            if cmd == "ping":
                _ping()
            elif cmd == "render":
                _render(request)
            else:
                raise ValueError(f"unknown command {cmd!r}")
        except Exception as e:
            _reply({"id": request.get("id"), "ok": False, "error": f"{type(e).__name__}: {e}"[:500]})
        else:
            _reply({"id": request.get("id"), "ok": True})
    if kaleido is not None and startup_error is None:
        try:
            kaleido.stop_sync_server(silence_warnings=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
