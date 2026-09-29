"""
Per-browser-session state isolation for aps_explorer.py — the mechanism
that lets one running Dash server be used safely by several people at
once (a Docker/gunicorn deployment), while leaving today's single-user
standalone usage (`python aps_explorer.py --infiles ...`) completely
unchanged.

**The problem this solves.** `aps_explorer.py` and the three viewer
library modules it imports (`aps_l1_preview.py`/`aps_IFUviewer.py`/
`aps_MOSviewer.py`) each keep the entire loaded dataset — selection,
camera state, colour-range prefs, the log panel — in a single
module-level global (`EXPLORER = ExplorerState()`, `STATE = AppState()`
in each viewer module, `LOG = _LogBuffer()`). That's invisible with one
person, one browser tab, SSH-tunneled in — the only usage pattern to
date. The moment two people hit the same running server, they'd
silently stomp on each other's state with no error.

**The fix, deliberately NOT threading a session_id through 56 callback
signatures.** `EXPLORER`/`STATE`/`LOG` become `werkzeug.local.LocalProxy`
objects (the exact mechanism Flask itself uses internally for
`flask.g`/`flask.request`) bound to a `contextvars.ContextVar` that
`aps_explorer.py`'s `before_request`/`teardown_request` hooks point at
the *current request's* `SessionBundle` for the duration of that
request. Every one of the ~215 existing `EXPLORER.foo`/`STATE.foo` call
sites keeps working completely unchanged — a proxy forwards every
attribute get/set to whatever the context var currently resolves to.

**Per-session state is a live Python object, never pickled.** The real
loaded dataset (`AppState.targs`/`.apsob`/`.fwhm_cache`, etc.) can be
multi-GB and up to ~140s to build for a large LIFU cube. Round-tripping
that through an external cache (Redis/disk) on every single callback —
a colour-range tweak, a tab click — would regress hard-won load-time
performance for no real benefit at this tool's actual scale (a handful
of concurrent astronomers, not a public service). Instead each
`SessionBundle` just sits in this module's own `dict`, in server RAM,
for as long as that session is active.

**Consequence: this module assumes a single worker process.** An
in-process dict only isolates sessions correctly if a given session's
requests always land on the same process — so the server deployment
must run gunicorn with `--workers 1` (many *threads* via
`--worker-class gthread --threads N` for concurrency, not many worker
processes). If real horizontal scaling beyond one host is ever needed,
that's a follow-up (sticky sessions at a proxy, or a genuinely shared
store) — not built here.

**Mode gate — `MULTI_SESSION`, default off.** Without an explicit gate,
even single-user standalone mode would break: `explorer_worker()`'s
CLI-preloaded dataset (`--infiles`/`--outpath`+`--headname`) runs
*before* any real browser request exists, so under an always-on
per-session scheme the first real request would mint a brand-new
*empty* bundle and the browser would show "nothing loaded" despite the
terminal saying otherwise. With `MULTI_SESSION` off (the default —
standalone usage, including the `aps-explorer` console-script), the
session-binding hook in `aps_explorer.py` is a no-op: every access
resolves to one persistent default bundle, exactly today's behaviour.
Only set `PYAPS_EXPLORER_MULTI_SESSION=1` (done automatically by
`wsgi.py` for the Docker/gunicorn path) or pass `--multi-session`
together with no CLI-preload for the real multi-user case.

This same default-bundle fallback is also what keeps
`tests/test_aps_explorer.py` working completely unmodified — those
tests call functions directly with no live Flask request at all, so
they always see the one default bundle, matching today's exact
"one shared global" behaviour.
"""

from __future__ import annotations

import contextvars
import os
import threading
import time
import uuid


def _env_flag(name, default=False):
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() not in ("", "0", "false", "no", "off")


# Mutable at runtime (aps_explorer.py's --multi-session flips this after
# argv parsing, before app.run()) — not a constant, so plain module-level
# reads elsewhere must go through this module rather than caching the
# value at import time.
MULTI_SESSION = _env_flag("PYAPS_EXPLORER_MULTI_SESSION", False)


def _normalize_url_prefix(raw):
    """Dash's own `url_base_pathname` requires a leading *and* trailing
    slash (`"/weave/"`, not `"weave"`/`"/weave"`) — normalize whatever a
    human typed into the env var into that exact shape, collapsing any
    accidental repeated slashes along the way. Empty/whitespace-only
    input means "not mounted under a subpath" (`"/"`), matching Dash's
    own default."""
    raw = (raw or "").strip()
    if not raw or raw == "/":
        return "/"
    parts = [p for p in raw.split("/") if p]
    return "/" + "/".join(parts) + "/"


# Read once at import time, deliberately NOT mutable via a CLI flag the
# way MULTI_SESSION/REQUIRE_WEAVEOR_AUTH are: `url_base_pathname` is a
# Dash *constructor* argument (`aps_explorer.py`'s `app = Dash(...)`,
# itself a module-level statement) — by the time `explorer_worker()`
# parses argv, Dash has already registered its own internal routes under
# whatever prefix was in effect at import time, and there is no way to
# re-register them afterward. Set `PYAPS_EXPLORER_URL_PREFIX` in the
# environment *before* the process starts (exactly like `wsgi.py` already
# does for `PYAPS_EXPLORER_MULTI_SESSION`) — a real shell env var is
# naturally available before any import happens, which a CLI flag parsed
# inside `explorer_worker()` cannot be. Only relevant to server/Docker
# deployments sharing one host across several projects (e.g.
# `hub.example.org/weave/`) — standalone/single-project usage leaves this
# unset and gets the unprefixed `"/"` it always has.
URL_PREFIX = _normalize_url_prefix(os.environ.get("PYAPS_EXPLORER_URL_PREFIX"))

SESSION_COOKIE_NAME = "pyaps_sid"
SESSION_TTL_SECONDS = int(os.environ.get("PYAPS_EXPLORER_SESSION_TTL_SECONDS", 4 * 3600))
_SWEEP_INTERVAL_SECONDS = 300

# Minutes of genuine client-side inactivity (no mouse/keyboard/click/scroll
# event — NOT the same as SESSION_TTL_SECONDS above, which only measures
# time since the last *request*: aps_explorer.py's log panel polls the
# server every 700ms for as long as a tab stays open, including a tab
# left forgotten in the background, so request-recency alone never goes
# stale on its own). aps_explorer.py injects a client-side timer using
# this value (only when MULTI_SESSION is on) that explicitly ends the
# session via end_session() below once truly idle — SESSION_TTL_SECONDS
# remains a passive backstop for the case that JS never gets to run at
# all (disabled JS, a tab killed without a chance to fire), not the
# primary mechanism.
IDLE_TIMEOUT_MINUTES = int(os.environ.get("PYAPS_EXPLORER_IDLE_TIMEOUT_MINUTES", 10))

# A deployment-wide fallback caldir/catdir for L1 loads that don't
# specify one -- concretely, a weaveOR handoff token today only carries
# `infiles` (weaveOR has no clean source for caldir/catdir to put in the
# token -- see aps_explorer_handoff's own weaveOR-side docstring), so
# without this, every L1 dataset opened via a real handoff loaded with
# no LSF/FWHM diagnostics at all, even though the container genuinely
# has read access to the real CAL/CAT archive. Unlike infiles/outpath+
# headname (genuinely different per dataset), caldir/catdir point at one
# fixed archive root for an entire deployment (confirmed against the
# real archive: CAL/CAT are top-level directories the underlying code
# searches by date internally, e.g. CAL/20230512/, not per-dataset
# paths) -- so a single env-var default, unlike a per-token field, is
# both sufficient and the simpler fix. Unset by default, exactly like
# PYAPS_CONFIGDIR -- a deployment that doesn't set these keeps today's
# behavior (no default, caldir/catdir stay None unless the token/form
# explicitly provides one) with zero change.
DEFAULT_CALDIR = os.environ.get("PYAPS_EXPLORER_DEFAULT_CALDIR") or None
DEFAULT_CATDIR = os.environ.get("PYAPS_EXPLORER_DEFAULT_CATDIR") or None


class SessionBundle:
    """One browser session's worth of state. Slots are populated lazily,
    one per owner module (`aps_explorer.py` uses "explorer"/"log",
    `aps_l1_preview.py`/`aps_IFUviewer.py`/`aps_MOSviewer.py` use
    "l1"/"ifu"/"mos" respectively) — this class deliberately never
    imports `ExplorerState`/`AppState` itself, so this module stays a
    leaf with no circular-import risk against the four files that import
    *it*.
    """

    __slots__ = ("slots", "last_accessed", "_lock")

    def __init__(self):
        self.slots = {}
        self.last_accessed = time.time()
        self._lock = threading.Lock()

    def get_or_create(self, key, factory):
        with self._lock:
            if key not in self.slots:
                self.slots[key] = factory()
            return self.slots[key]


# The one persistent bundle used whenever MULTI_SESSION is off, or when
# there's no active Flask request at all (the test suite's calling
# convention) — this IS the ContextVar's own default, so it's returned
# without ever calling bind_request()/.set() at all in that case.
_DEFAULT_BUNDLE = SessionBundle()

_current_bundle = contextvars.ContextVar(
    "pyaps_explorer_session_bundle", default=_DEFAULT_BUNDLE
)

_sessions: dict[str, SessionBundle] = {}
_sessions_lock = threading.Lock()
_last_sweep = 0.0

# Last-known-owner mapping for identified (weaveOR-token) users only --
# see claim_session_for_user() below. Deliberately just a plain dict, no
# separate cleanup pass: an entry pointing at an already-evicted/expired
# sid is harmless (the next claim for that user finds nothing to evict
# in _sessions and just overwrites the stale mapping) rather than
# something that needs active maintenance.
_sessions_by_user: dict[str, str] = {}


def current_bundle():
    """The `SessionBundle` for whatever request (if any) is in progress
    on the calling thread — the one function every `LocalProxy` in
    `aps_explorer.py`/the three viewer modules is built on top of."""
    return _current_bundle.get()


def _sweep_locked(now):
    stale = [sid for sid, b in _sessions.items() if now - b.last_accessed > SESSION_TTL_SECONDS]
    for sid in stale:
        del _sessions[sid]


def end_session(sid):
    """Immediately evict `sid`'s `SessionBundle`, freeing its (potentially
    multi-GB — see the module docstring's own note on `AppState.targs`/
    `.apsob`/`.fwhm_cache`) memory right away rather than waiting for the
    next `SESSION_TTL_SECONDS` sweep. Called from aps_explorer.py's
    idle-kill callback once a client-side inactivity timer decides a tab
    has genuinely gone unused (see IDLE_TIMEOUT_MINUTES above) — a no-op,
    not an error, if `sid` is already gone (already swept, or never
    existed)."""
    with _sessions_lock:
        _sessions.pop(sid, None)


def claim_session_for_user(user, sid):
    """At most one live session per identified weaveOR user — confirmed
    live as a real problem, not a hypothetical: opening two different
    dataset links from weaveOR in two browser tabs (both target="_blank")
    lands both in the *same* pyaps_sid cookie/session (browsers share
    cookies across same-origin tabs), so this alone doesn't cause
    duplicate sessions -- the real duplication happens when the *same*
    user opens links in two genuinely separate browser contexts (a
    second browser, a private/incognito window, a colleague's machine
    logged in as them, etc.), each getting its own fresh cookie. Without
    this, every such click accumulates another full session's worth of
    server memory (potentially multi-GB each) with no way for the old
    one to ever get reclaimed except the passive SESSION_TTL_SECONDS
    sweep.

    Called from aps_explorer.py's handle_url_handoff right after a real
    weaveOR token verifies (never for the plain deep-link path, which
    has no real `user` identity to key on at all -- see this function's
    own `if not user` no-op below). `sid` is *this* request's own
    session id (flask_g._pyaps_session_id) -- if `user` already owns a
    *different* sid, that old session is evicted immediately; if it's
    the same sid (this tab/cookie loading a fresh dataset for itself --
    the existing, intentional "new token in the same tab" flow), this is
    correctly a no-op, not a self-eviction."""
    if not user:
        return
    with _sessions_lock:
        old_sid = _sessions_by_user.get(user)
        if old_sid and old_sid != sid:
            _sessions.pop(old_sid, None)
        _sessions_by_user[user] = sid


def bind_request(cookie_sid):
    """Look up (or create) the `SessionBundle` for `cookie_sid` — the
    value of the `pyaps_sid` cookie on the incoming request, or `None`
    if it wasn't set (first visit, or cookies cleared) — and make it the
    current one for this thread until `unbind_request()` is called.

    Gating on `MULTI_SESSION` is the caller's job (`aps_explorer.py`'s
    `before_request` hook), not this function's — this always does a
    real bind, so it's independently testable regardless of the app's
    own mode flag.

    Returns `(bundle, sid, token)`: `sid` is the id to send back as the
    cookie (echoing `cookie_sid` if it matched a live session, otherwise
    a freshly minted one); `token` is what `unbind_request()` needs.
    """
    global _last_sweep
    now = time.time()
    with _sessions_lock:
        if now - _last_sweep > _SWEEP_INTERVAL_SECONDS:
            _sweep_locked(now)
            _last_sweep = now
        sid = cookie_sid if cookie_sid in _sessions else (cookie_sid or uuid.uuid4().hex)
        if sid not in _sessions:
            _sessions[sid] = SessionBundle()
        bundle = _sessions[sid]
        bundle.last_accessed = now
    token = _current_bundle.set(bundle)
    return bundle, sid, token


def unbind_request(token):
    """Undo `bind_request()`'s `.set()` — restores whatever the context
    var resolved to before (the default bundle, at the top level of a
    single request; a previous bundle, if bind_request was nested, which
    the app itself never does but a test might)."""
    _current_bundle.reset(token)
