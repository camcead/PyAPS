"""Static image export of Plotly figures that can never block a science product.

``write_image(fig, path)`` replaces ``fig.write_image(path)`` in every diagnostic-plot call of the pipeline.

Why it exists. Plotly renders static images through Kaleido, which drives a headless Chrome. Kaleido 1.x waits for the
browser without any deadline while it starts, navigates and closes (only the render itself has a 90 s limit). On the two
APS pods roughly one job in thirty hit a render that never came back: the browser was alive and idle, Python was waiting
for it, the job used no CPU and ran into the SLURM wall limit two days later, and the night it belonged to stayed
PROCESSING. A diagnostic picture must never cost a night.

What it does.
* The figure is rendered by a separate worker process (``_export_worker.py``) in its own session. The worker keeps one
  browser for many images (about ten times faster than the browser-per-image start of ``fig.write_image``).
* Every request has a deadline (``PYAPS_PLOT_TIMEOUT``, default 120 s; the first request after a start gets
  ``PYAPS_PLOT_START_TIMEOUT``, default 180 s). When it expires the worker's stack is written to the log, the whole
  process group (worker and browser) is killed, the picture is skipped (a warning in the log, ``write_image`` returns
  False) and the next picture starts a new worker.
* After ``PYAPS_PLOT_MAX_FAILURES`` (default 3) timeouts or worker deaths in a row the pictures of this process are
  switched off (one warning) so a broken browser costs minutes, not hours.
* The worker is replaced after ``PYAPS_PLOT_RECYCLE`` images (default 200) to bound the browser's memory.
* A picture that fails with an ordinary error (bad figure, unwritable folder) is skipped with a warning and does not
  count towards the switch-off.

``write_image`` never raises and never blocks beyond its deadline. It returns True when the file was written.

``PYAPS_PLOT_EXPORT``: ``off`` (DEFAULT since 2026-11-04: no static images at all), ``worker`` (the process described above; also
written ``on``) or ``inprocess`` (the plain ``fig.write_image``, errors logged and skipped, no deadline: for debugging only).

Why the default is ``off``. The pictures are diagnostics; the headless Chrome behind them was most of the wall time and CPU of the
L2 jobs (one RR job of the SLURM audit: 81 min and 520 CPU-minutes with pictures, 18 min and 117 CPU-minutes without; RVS and
IFU_ExGal also lost the 13-22 GB of Chrome). No data product depends on a picture (``write_image`` never raises and no caller
reads its result). The runner (``aps_runner.py --plot_export``, the script_params key ``plot_export``) writes the chosen mode into
every job script and prints it at the top of each job log: ``resolve_mode`` below is the single place that decides.
"""
from __future__ import annotations

import atexit
import collections
import json
import logging
import os
import select
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

try:                                                     # the package, or the file run as a script
    from PyAPS import aps_dms_settings as dms_settings
except ImportError:                                      # pragma: no cover
    import aps_dms_settings as dms_settings

log = logging.getLogger(__name__)

_WORKER = Path(__file__).with_name("_export_worker.py")
_STDERR_TAIL = 60                      # lines of the worker's stderr kept for the timeout report


def _env_float(name, default):
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return float(default)


def _env_int(name, default):
    return int(_env_float(name, default))


DEFAULT_MODE = "off"
MODES = ("off", "worker", "inprocess")
_ALIASES = {"on": "worker", "true": "worker", "yes": "worker", "1": "worker", "auto": None, "default": None, "": None,
            "false": "off", "no": "off", "0": "off", "none": "off"}


def normalize_mode(value):
    """``off`` / ``worker`` / ``inprocess`` for a user's word (``on``, ``true``, ``0``...), None when it says nothing (unset, ``auto``).

    Raises ValueError for a word that means nothing at all."""
    if value is None:
        return None
    word = str(value).strip().lower()
    if word in MODES:
        return word
    if word in _ALIASES:
        return _ALIASES[word]
    raise ValueError("plot export mode %r is not one of off, on (= worker), worker, inprocess" % (value,))


def resolve_mode(cli=None, conf=None, env=None, dms=None):
    """``(mode, source)`` of a run: the option (``--plot_export``) > ``PYAPS_PLOT_EXPORT`` > the ``plot_export`` key of the DMS config of the pod
    (``PyAPS.aps_dms_settings``) > the ``plot_export`` key of script_params > ``off``.

    A word that means nothing is ignored (with source ``<where> (ignored: ...)`` appended) and the next source decides."""
    env = os.environ if env is None else env
    ignored = ""
    for where, value in dms_settings.chain("plot_export", cli, conf, env, dms):
        try:
            mode = normalize_mode(value)
        except ValueError:
            ignored += " (ignored %s=%r)" % (where, value)
            continue
        if mode is not None:
            return mode, where + ignored
    return DEFAULT_MODE, "default" + ignored


def describe(cli=None, conf=None, env=None, dms=None):
    """One line for the top of a job log."""
    mode, source = resolve_mode(cli, conf, env, dms)
    hint = ("static diagnostic images are NOT written; plot_export: 'on' in script_params or PYAPS_PLOT_EXPORT=worker turns them on"
            if mode == "off" else "static diagnostic images are written by Chrome through Kaleido (slow; PYAPS_PLOT_EXPORT=off to skip)")
    return "PyAPS plot export: mode=%s source=%s; %s" % (mode, source, hint)


def _mode():
    try:
        return normalize_mode(os.environ.get("PYAPS_PLOT_EXPORT")) or DEFAULT_MODE
    except ValueError:
        return DEFAULT_MODE


def _worker_command():
    """The command that starts a worker (tests replace this function)."""
    return [sys.executable, str(_WORKER)]


class _Worker:
    """One worker process and the pipe protocol to it."""

    def __init__(self):
        self.proc = subprocess.Popen(
            _worker_command(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True, close_fds=True)
        self.pid = self.proc.pid
        self.owner = os.getpid()
        self.served = 0
        self.warm = False
        self._buffer = b""
        self._tail = collections.deque(maxlen=_STDERR_TAIL)
        self._reader = threading.Thread(target=self._drain_stderr, daemon=True)
        self._reader.start()

    def _drain_stderr(self):
        try:
            for raw in iter(self.proc.stderr.readline, b""):
                self._tail.append(raw.decode("utf-8", "replace").rstrip())
        except (OSError, ValueError):
            pass

    def stderr_tail(self, n=25):
        return list(self._tail)[-n:]

    def alive(self):
        return self.proc.poll() is None

    def send(self, obj):
        data = (json.dumps(obj) + "\n").encode()
        self.proc.stdin.write(data)
        self.proc.stdin.flush()

    def receive(self, request_id, deadline):
        """The reply with this id, or None when the deadline passes. Raises EOFError when the worker ended."""
        fd = self.proc.stdout.fileno()
        while True:
            while b"\n" in self._buffer:
                line, self._buffer = self._buffer.split(b"\n", 1)
                try:
                    reply = json.loads(line.decode("utf-8", "replace"))
                except ValueError:
                    continue                                  # a stray line of a library on stdout
                if isinstance(reply, dict) and reply.get("id") == request_id:
                    return reply
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            ready, _, _ = select.select([fd], [], [], min(remaining, 1.0))
            if not ready:
                continue
            chunk = os.read(fd, 65536)
            if not chunk:
                raise EOFError("the export worker ended")
            self._buffer += chunk

    def kill(self, report=False):
        """Stop the worker and everything below it. ``report`` first asks it for its stack (written to its stderr)."""
        if report and self.alive():
            try:
                os.kill(self.pid, signal.SIGUSR1)
                time.sleep(0.5)
            except OSError:
                pass
        try:
            os.killpg(self.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass
        try:
            self.proc.kill()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        for pipe in (self.proc.stdin, self.proc.stdout):
            try:
                pipe.close()
            except (OSError, ValueError):
                pass

    def close(self):
        """Orderly end (at exit); falls back to the kill."""
        if self.alive():
            try:
                self.send({"cmd": "quit"})
                self.proc.wait(timeout=5)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
        self.kill()


class _State:
    worker = None
    failures = 0                       # timeouts / worker deaths in a row
    disabled = False
    next_id = 0
    sent = 0
    written = 0
    skipped = 0
    timeouts = 0


_state = _State()
_lock = threading.Lock()


def status():
    """Counters of this process: images written, skipped, timeouts, whether the export was switched off."""
    return {"mode": _mode(), "written": _state.written, "skipped": _state.skipped, "timeouts": _state.timeouts,
            "disabled": _state.disabled}


def shutdown():
    """End the worker (registered at exit)."""
    with _lock:
        worker, _state.worker = _state.worker, None
    if worker is not None and worker.owner == os.getpid():
        worker.close()


atexit.register(shutdown)


def reset():
    """Forget counters and kill the worker (tests; a long-lived process that wants a fresh start)."""
    shutdown()
    _state.failures = 0
    _state.disabled = False
    _state.sent = _state.written = _state.skipped = _state.timeouts = 0


def _skip(path, why, quiet=False):
    _state.skipped += 1
    if quiet:
        if _state.skipped == 1:                # one line per process: with the default (off) every picture is skipped
            log.info("static images are off (%s); the first skipped picture is %s", why, path)
        return False
    log.warning("image %s not written: %s", path, why)
    return False


def _timed_out(worker, path, what, seconds):
    _state.timeouts += 1
    _state.failures += 1
    tail = worker.stderr_tail()
    worker.kill(report=True)
    tail = worker.stderr_tail() or tail
    log.warning("image %s not written: the export worker gave no answer for %s within %.0f s; worker and browser killed, "
                "the job continues without this picture (%d in a row)", path, what, seconds, _state.failures)
    if tail:
        log.warning("last lines of the export worker:\n%s", "\n".join(tail))
    _state.skipped += 1
    return False


def _get_worker():
    """A live worker of this process (a worker inherited through fork belongs to the parent and is never used)."""
    worker = _state.worker
    if worker is not None and (worker.owner != os.getpid() or not worker.alive()
                               or worker.served >= max(1, _env_int("PYAPS_PLOT_RECYCLE", 200))):
        if worker.owner == os.getpid():
            worker.close()
        _state.worker = worker = None
    if worker is None:
        _state.worker = worker = _Worker()
    return worker


def write_image(fig, path, *, scale=None, timeout=None, **kwargs):
    """Write ``fig`` to the image file ``path``. Returns True when written, False when the picture was skipped.

    ``scale`` (when given) and any further keyword argument go to ``Figure.write_image``. ``timeout`` (seconds) overrides
    ``PYAPS_PLOT_TIMEOUT`` for this picture. Never raises, never blocks longer than the deadline.
    """
    path = os.fspath(path)
    if scale is not None:
        kwargs["scale"] = scale
    mode = _mode()
    if mode == "off":
        return _skip(path, "PYAPS_PLOT_EXPORT=off, the default; PYAPS_PLOT_EXPORT=worker writes them", quiet=True)
    if mode == "inprocess":
        try:
            fig.write_image(path, **kwargs)
        except Exception as e:                     # a picture is never worth a night
            return _skip(path, f"{type(e).__name__}: {e}")
        _state.written += 1
        return True

    with _lock:
        if _state.disabled:
            return _skip(path, "static images are switched off in this process after repeated export failures")
        try:
            payload = {"cmd": "render", "fig": fig.to_json(), "path": path, "kwargs": kwargs}
        except Exception as e:
            return _skip(path, f"the figure cannot be serialised: {type(e).__name__}: {e}")
        try:
            worker = _get_worker()
        except OSError as e:
            _state.failures += 1
            return _skip(path, f"the export worker cannot be started: {e}")
        _state.next_id += 1
        request_id = payload["id"] = _state.next_id
        seconds = timeout if timeout is not None else _env_float("PYAPS_PLOT_TIMEOUT", 120)
        if not worker.warm:
            seconds = max(seconds, _env_float("PYAPS_PLOT_START_TIMEOUT", 180))
        deadline = time.monotonic() + seconds
        try:
            worker.send(payload)
            reply = worker.receive(request_id, deadline)
        except (OSError, EOFError, ValueError) as e:
            tail = worker.stderr_tail(8)
            worker.kill()
            _state.worker = None
            _state.failures += 1
            _skip(path, f"the export worker died ({type(e).__name__}); last lines: {' | '.join(tail)[:400]}")
            reply = False
        else:
            if reply is None:
                _state.worker = None
                _timed_out(worker, path, "this image", seconds)
                reply = False
            else:
                worker.served += 1
                worker.warm = True
                if reply.get("ok"):
                    _state.failures = 0
                    _state.written += 1
                    return True
                if reply.get("fatal"):                 # no browser at all: every further picture would fail the same way
                    _state.failures = _env_int("PYAPS_PLOT_MAX_FAILURES", 3)
                else:                                  # the worker answered: it is healthy, the figure or the folder is not
                    _state.failures = 0
                _skip(path, reply.get("error") or "the export worker reported an error")
                reply = False
        if _state.failures >= max(1, _env_int("PYAPS_PLOT_MAX_FAILURES", 3)) and not _state.disabled:
            _state.disabled = True
            log.warning("static images are switched off in this process: %d export failures in a row (the pictures "
                        "are diagnostics; the data products are not affected)", _state.failures)
        return bool(reply)
