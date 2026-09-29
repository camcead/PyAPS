"""
tests/test_aps_explorer_session.py - Tests for aps_explorer_session.py,
the per-browser-session isolation mechanism that lets one running
aps_explorer.py server be used safely by several people at once (the
Docker/gunicorn deployment path — see wsgi.py/Dockerfile) while leaving
standalone/single-user usage completely unchanged (the mode gate,
`MULTI_SESSION`, defaults off).

No browser/live server needed — matches this repo's existing
`test_aps_explorer.py` philosophy of calling the underlying mechanism
directly. One test additionally uses Flask's own in-process
`test_client()` (still no real socket) to prove the actual cookie
round-trip through `aps_explorer.py`'s registered hooks, on top of the
direct-call coverage of `aps_explorer_session.py` itself.

Run with:
    cd <PYAPS_DIR> && pytest tests/test_aps_explorer_session.py -v
"""

from __future__ import annotations

import pytest


def test_two_sessions_get_independent_bundles():
    from PyAPS import aps_explorer_session as sess

    b1, sid1, tok1 = sess.bind_request(None)
    sess.unbind_request(tok1)
    b2, sid2, tok2 = sess.bind_request(None)
    sess.unbind_request(tok2)

    assert sid1 != sid2
    assert b1 is not b2


def test_same_cookie_returns_same_bundle():
    from PyAPS import aps_explorer_session as sess

    b1, sid1, tok1 = sess.bind_request(None)
    sess.unbind_request(tok1)
    b2, sid2, tok2 = sess.bind_request(sid1)
    sess.unbind_request(tok2)

    assert sid1 == sid2
    assert b1 is b2


def test_no_cookie_state_leakage_between_sessions():
    """The actual bug this whole module exists to prevent: two different
    browsers loading two different datasets on the same running server
    must never see each other's state."""
    from PyAPS import aps_explorer as ex
    from PyAPS import aps_explorer_session as sess

    _, sid1, tok1 = sess.bind_request(None)
    ex.EXPLORER.kind = "l1"
    ex.l1_mod.STATE.color_by = "snr"
    sess.unbind_request(tok1)

    _, sid2, tok2 = sess.bind_request(None)
    try:
        assert ex.EXPLORER.kind is None
        assert ex.l1_mod.STATE.color_by == "flux"  # AppState's own default, not leaked
    finally:
        sess.unbind_request(tok2)

    _, _, tok1b = sess.bind_request(sid1)
    try:
        assert ex.EXPLORER.kind == "l1"
        assert ex.l1_mod.STATE.color_by == "snr"
    finally:
        sess.unbind_request(tok1b)


def test_bind_session_is_noop_when_multi_session_off(monkeypatch):
    """With --multi-session off (the default — standalone usage), the
    before_request hook must not touch flask.request at all (there may
    be no active Flask request context, e.g. this call itself)."""
    from PyAPS import aps_explorer as ex
    from PyAPS import aps_explorer_session as sess

    monkeypatch.setattr(sess, "MULTI_SESSION", False)
    ex._bind_session()  # must not raise despite no active Flask request


def test_session_bundle_slot_constructed_once():
    from PyAPS.aps_explorer_session import SessionBundle

    calls = []
    bundle = SessionBundle()
    first = bundle.get_or_create("x", lambda: calls.append(1) or object())
    second = bundle.get_or_create("x", lambda: calls.append(1) or object())

    assert len(calls) == 1
    assert first is second


def test_idle_session_evicted(monkeypatch):
    from PyAPS import aps_explorer_session as sess

    monkeypatch.setattr(sess, "SESSION_TTL_SECONDS", 0)
    monkeypatch.setattr(sess, "_SWEEP_INTERVAL_SECONDS", 0)

    b1, sid, tok1 = sess.bind_request(None)
    sess.unbind_request(tok1)
    b2, sid2, tok2 = sess.bind_request(sid)
    sess.unbind_request(tok2)

    # TTL=0 forces the sweep to have evicted b1 before this second bind —
    # a brand new SessionBundle is created, even though the cookie value
    # itself is reused (no need to force a new Set-Cookie just because
    # the old session's *data* expired).
    assert b1 is not b2


def test_end_session_evicts_immediately():
    """The client-side-inactivity-timer-driven counterpart to the
    passive TTL sweep above — used when a session should be reclaimed
    right away (aps_explorer.py's handle_idle_kill callback) rather than
    waiting for the much longer SESSION_TTL_SECONDS sweep."""
    from PyAPS import aps_explorer_session as sess

    b1, sid, tok1 = sess.bind_request(None)
    sess.unbind_request(tok1)
    assert sid in sess._sessions

    sess.end_session(sid)
    assert sid not in sess._sessions

    # A brand new bundle on the next request with the same (now-stale)
    # cookie, not a crash or a resurrected old one.
    b2, sid2, tok2 = sess.bind_request(sid)
    sess.unbind_request(tok2)
    assert b2 is not b1


def test_end_session_missing_or_none_sid_is_a_safe_noop():
    from PyAPS import aps_explorer_session as sess

    sess.end_session("this-sid-was-never-created")
    sess.end_session(None)  # never actually reached with None in practice
    # (handle_idle_kill guards on it), but must not raise either way.


def test_claim_session_for_user_evicts_a_different_prior_session():
    """The real bug this guards against: a weaveOR user opening dataset
    links in two genuinely separate browser contexts (not just two tabs
    of the same browser -- those share one cookie already) would
    otherwise accumulate one full session's worth of server memory per
    click with no way to reclaim the old ones except the passive TTL
    sweep."""
    from PyAPS import aps_explorer_session as sess

    sess._sessions.clear()
    sess._sessions_by_user.clear()
    try:
        sess.claim_session_for_user("alice", "sidA")
        sess._sessions["sidA"] = sess.SessionBundle()

        sess.claim_session_for_user("alice", "sidB")
        sess._sessions["sidB"] = sess.SessionBundle()

        assert "sidA" not in sess._sessions, "the old session for the same user must be evicted"
        assert "sidB" in sess._sessions
    finally:
        sess._sessions.clear()
        sess._sessions_by_user.clear()


def test_claim_session_for_user_same_sid_is_not_self_eviction():
    """The *existing*, intentional flow -- the same tab/cookie loading a
    fresh dataset via a new token -- must keep working exactly as
    before; claiming must never evict the very session making the
    claim."""
    from PyAPS import aps_explorer_session as sess

    sess._sessions.clear()
    sess._sessions_by_user.clear()
    try:
        sess.claim_session_for_user("bob", "sidX")
        sess._sessions["sidX"] = sess.SessionBundle()

        sess.claim_session_for_user("bob", "sidX")
        assert "sidX" in sess._sessions
    finally:
        sess._sessions.clear()
        sess._sessions_by_user.clear()


def test_claim_session_for_user_noop_without_a_real_identity():
    """The plain (non-token) deep-link path always passes user=None to
    record_dataset_loaded -- there's no real identity to enforce
    "one session per user" against, so this must be a complete no-op,
    never evicting anything based on a shared falsy "user"."""
    from PyAPS import aps_explorer_session as sess

    sess._sessions.clear()
    sess._sessions_by_user.clear()
    try:
        sess._sessions["sidC"] = sess.SessionBundle()
        sess.claim_session_for_user(None, "sidD")
        sess.claim_session_for_user("", "sidE")
        assert "sidC" in sess._sessions
        assert "sidD" not in sess._sessions_by_user
        assert "" not in sess._sessions_by_user
    finally:
        sess._sessions.clear()
        sess._sessions_by_user.clear()


def test_env_flag_parsing(monkeypatch):
    from PyAPS.aps_explorer_session import _env_flag

    monkeypatch.setenv("X", "1")
    assert _env_flag("X") is True
    monkeypatch.setenv("X", "true")
    assert _env_flag("X") is True
    monkeypatch.setenv("X", "0")
    assert _env_flag("X") is False
    monkeypatch.setenv("X", "off")
    assert _env_flag("X") is False
    monkeypatch.delenv("X", raising=False)
    assert _env_flag("X", default=True) is True
    assert _env_flag("X", default=False) is False


def test_normalize_url_prefix():
    """URL_PREFIX (PYAPS_EXPLORER_URL_PREFIX) backs the multi-project-host
    deployment shape (e.g. hub.example.org/weave/, a second project at
    hub.example.org/otherproject/) — Dash's own url_base_pathname
    requires an exact leading+trailing slash, so whatever a human typed
    into the env var needs normalizing into that shape first."""
    from PyAPS.aps_explorer_session import _normalize_url_prefix

    assert _normalize_url_prefix(None) == "/"
    assert _normalize_url_prefix("") == "/"
    assert _normalize_url_prefix("   ") == "/"
    assert _normalize_url_prefix("/") == "/"
    assert _normalize_url_prefix("weave") == "/weave/"
    assert _normalize_url_prefix("/weave") == "/weave/"
    assert _normalize_url_prefix("weave/") == "/weave/"
    assert _normalize_url_prefix("/weave/") == "/weave/"
    assert _normalize_url_prefix("//weave//") == "/weave/"
    assert _normalize_url_prefix("a/b") == "/a/b/"


def test_url_prefix_wires_into_dash_app_and_stays_off_by_default():
    """Real subprocess, not a monkeypatched attribute: URL_PREFIX is read
    once at import time (app = Dash(..., url_base_pathname=...) is
    itself a module-level statement — see aps_explorer_session.py's own
    comment on why this can't be a post-import CLI flag like
    MULTI_SESSION), so proving it actually took effect needs a genuinely
    fresh interpreter, matching this repo's existing subprocess-based
    testing pattern (test_aps_common_args.py's _live_flags)."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    pyaps_dir = Path(__file__).resolve().parent.parent / "py" / "PyAPS"

    def _probe(url_prefix=None):
        env = os.environ.copy()
        if url_prefix is None:
            env.pop("PYAPS_EXPLORER_URL_PREFIX", None)
        else:
            env["PYAPS_EXPLORER_URL_PREFIX"] = url_prefix
        code = (
            "from PyAPS import aps_explorer as ex\n"
            "print('PREFIX=' + repr(ex.app.config.get('url_base_pathname')))\n"
            "print('EXEMPT=' + repr(ex._SESSION_EXEMPT_PREFIXES))\n"
            "print('HEALTHZ=' + repr(any(r.rule == '/healthz' "
            "for r in ex.app.server.url_map.iter_rules())))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, cwd=str(pyaps_dir), timeout=60, env=env,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout

    # Default (unset): unaffected, matches every existing standalone test
    # in this suite that assumes an unprefixed app.
    out = _probe(None)
    assert "PREFIX='/'" in out
    assert "EXEMPT=('/assets/', '/_dash-component-suites/', '/_favicon')" in out
    assert "HEALTHZ=True" in out

    # Set: Dash's own routes, and this app's session-exempt matching,
    # both carry the prefix; /healthz (a plain Flask route, not a Dash
    # one) deliberately does not.
    out = _probe("/weave/")
    assert "PREFIX='/weave/'" in out
    assert "EXEMPT=('/weave/assets/', '/weave/_dash-component-suites/', '/weave/_favicon')" in out
    assert "HEALTHZ=True" in out


def test_session_cookie_roundtrip_via_test_client(monkeypatch):
    """Integration-level, on top of the direct-call tests above: proves
    aps_explorer.py's registered before_request/after_request hooks
    actually set the cookie on a real (in-process, no socket) request,
    and that /healthz stays exempt even with --multi-session on."""
    from PyAPS import aps_explorer as ex
    from PyAPS import aps_explorer_session as sess

    monkeypatch.setattr(sess, "MULTI_SESSION", True)
    client = ex.app.server.test_client()

    r1 = client.get("/")
    assert r1.status_code == 200
    assert sess.SESSION_COOKIE_NAME in r1.headers.get("Set-Cookie", "")

    r2 = client.get("/healthz")
    assert r2.status_code == 200
    assert r2.get_json() == {"status": "ok"}
    assert sess.SESSION_COOKIE_NAME not in r2.headers.get("Set-Cookie", "")


def test_idle_kill_trigger_evicts_this_session_via_real_callback_dispatch(monkeypatch):
    """End-to-end proof of aps_explorer.py's handle_idle_kill callback —
    the server-side half of the client-side inactivity timer injected
    into index_string (MULTI_SESSION-only; see IDLE_TIMEOUT_MINUTES's
    own comment above for why this can't just be a longer
    SESSION_TTL_SECONDS: the log panel's own 700ms poll would keep any
    merely-open tab looking "active" forever). Dispatches a real
    idle-kill-trigger.data update through Dash's own callback endpoint,
    exactly as the injected client-side JS would."""
    from PyAPS import aps_explorer as ex
    from PyAPS import aps_explorer_session as sess

    monkeypatch.setattr(sess, "MULTI_SESSION", True)
    client = ex.app.server.test_client()

    r1 = client.get("/")
    sid = r1.headers.get("Set-Cookie", "").split(f"{sess.SESSION_COOKIE_NAME}=")[1].split(";")[0]
    assert sid in sess._sessions

    resp = client.post("/_dash-update-component", json={
        "output": "idle-kill-status.children",
        "outputs": {"id": "idle-kill-status", "property": "children"},
        "inputs": [{"id": "idle-kill-trigger", "property": "data", "value": 1}],
        "changedPropIds": ["idle-kill-trigger.data"],
    })
    assert resp.status_code == 200
    assert sid not in sess._sessions
