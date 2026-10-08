"""
tests/test_aps_explorer_auth.py - Tests for aps_explorer_auth.py, the
cross-system authentication/authorization handoff from a trusted
upstream app (e.g. weaveOR) into aps_explorer.py — see that module's own
docstring for the full design (a capability-scoped session: a verified
token authorizes exactly the one dataset it names, not general browsing).

No browser/live server needed — matches this repo's existing
`test_aps_explorer_session.py` philosophy of calling the underlying
mechanism directly. `tests/test_aps_explorer.py` has one further,
higher-level test exercising this through the real `handle_url_handoff`
Dash callback end to end.

Run with:
    cd <PYAPS_DIR> && pytest tests/test_aps_explorer_auth.py -v
"""

from __future__ import annotations

import pytest
from itsdangerous import URLSafeTimedSerializer

from PyAPS import aps_explorer_auth as auth
from PyAPS import aps_explorer_session as sess
from PyAPS.aps_explorer_session import SessionBundle

_TEST_SECRET = "test-secret-only-used-in-this-file"


def _mint(monkeypatch, claims, secret=_TEST_SECRET):
    """Mint a token exactly the way a real token-issuing app would —
    using the module's own salt, so it's a genuine round-trip test of
    `verify_token`, not a shortcut around it."""
    monkeypatch.setattr(auth, "_SECRET", secret)
    return URLSafeTimedSerializer(secret, salt=auth._SALT).dumps(claims)


def test_token_round_trip(monkeypatch):
    claims = {"user": "alice", "allowed_surveys": ["WL2022A1"], "kind": "l2",
              "outpath": "/fake_root/L2/20230514", "headname": "LWVE_123"}
    token = _mint(monkeypatch, claims)
    decoded = auth.verify_token(token)
    assert decoded == claims


def test_verify_token_bad_signature(monkeypatch):
    token = _mint(monkeypatch, {"user": "alice", "kind": "l1", "infiles": ["a.fit"]})
    # Alter a character well inside the signature: the final base64 character can carry
    # padding bits only, so changing it would sometimes decode to the same signature.
    i = len(token) - 6
    tampered = token[:i] + ("a" if token[i] != "a" else "b") + token[i + 1:]
    with pytest.raises(auth.TokenError, match="invalid"):
        auth.verify_token(tampered)


def test_verify_token_expired(monkeypatch):
    token = _mint(monkeypatch, {"user": "alice", "kind": "l1", "infiles": ["a.fit"]})
    # -1 forces "elapsed > max_age" on the very first check, no sleep needed.
    monkeypatch.setattr(auth, "_TOKEN_MAX_AGE_SECONDS", -1)
    with pytest.raises(auth.TokenError, match="expired"):
        auth.verify_token(token)


def test_verify_token_no_secret_configured(monkeypatch):
    monkeypatch.setattr(auth, "_SECRET", None)
    with pytest.raises(RuntimeError, match="WEAVEOR_SECRET"):
        auth.verify_token("anything")


def test_require_secret_configured(monkeypatch):
    monkeypatch.setattr(auth, "_SECRET", None)
    with pytest.raises(RuntimeError, match="WEAVEOR_SECRET"):
        auth.require_secret_configured()
    monkeypatch.setattr(auth, "_SECRET", _TEST_SECRET)
    auth.require_secret_configured()  # must not raise


def test_dataset_identity_l2():
    assert auth.dataset_identity("l2", outpath="/a/b", headname="h") == ("l2", "/a/b", "h")


def test_dataset_identity_l1_is_order_independent():
    a = auth.dataset_identity("l1", infiles=["blue.fit", "red.fit"])
    b = auth.dataset_identity("l1", infiles=["red.fit", "blue.fit"])
    assert a == b == ("l1", ("blue.fit", "red.fit"))


def test_dataset_identity_unknown_kind_raises():
    with pytest.raises(ValueError):
        auth.dataset_identity("xyz")


def test_apply_token_and_current_auth(monkeypatch):
    bundle = SessionBundle()
    assert auth.current_auth(bundle) is None

    claims = {"user": "alice", "allowed_surveys": ["WL2022A1", "WL2022A2"],
              "kind": "l2", "outpath": "/fake_root/L2/x", "headname": "h1"}
    auth.apply_token(bundle, claims)
    stored = auth.current_auth(bundle)
    assert stored["user"] == "alice"
    assert stored["allowed_surveys"] == {"WL2022A1", "WL2022A2"}
    assert stored["authorized_dataset"] == ("l2", "/fake_root/L2/x", "h1")

    # A second token replaces the first — moving to a new dataset, not
    # accumulating permissions.
    auth.apply_token(bundle, {"user": "alice", "allowed_surveys": ["WL2022A3"],
                               "kind": "l1", "infiles": ["a.fit", "b.fit"]})
    stored2 = auth.current_auth(bundle)
    assert stored2["authorized_dataset"] == ("l1", ("a.fit", "b.fit"))


def test_check_load_authorized_noop_when_mode_off(monkeypatch):
    monkeypatch.setattr(auth, "REQUIRE_WEAVEOR_AUTH", False)
    bundle = SessionBundle()  # no auth applied at all
    assert auth.check_load_authorized(bundle, "l2", outpath="/anything", headname="h") is None


def test_check_load_authorized_rejects_when_no_auth_on_bundle(monkeypatch):
    monkeypatch.setattr(auth, "REQUIRE_WEAVEOR_AUTH", True)
    bundle = SessionBundle()
    error = auth.check_load_authorized(bundle, "l2", outpath="/x", headname="h")
    assert error is not None
    assert "WeaveOR" in error


def test_check_load_authorized_allows_matching_dataset(monkeypatch):
    monkeypatch.setattr(auth, "REQUIRE_WEAVEOR_AUTH", True)
    bundle = SessionBundle()
    auth.apply_token(bundle, {"user": "alice", "allowed_surveys": ["WL2022A1"],
                               "kind": "l2", "outpath": "/fake_root/L2/x", "headname": "h1"})
    assert auth.check_load_authorized(bundle, "l2", outpath="/fake_root/L2/x", headname="h1") is None


def test_check_load_authorized_rejects_different_dataset(monkeypatch):
    monkeypatch.setattr(auth, "REQUIRE_WEAVEOR_AUTH", True)
    bundle = SessionBundle()
    auth.apply_token(bundle, {"user": "alice", "allowed_surveys": ["WL2022A1"],
                               "kind": "l2", "outpath": "/fake_root/L2/x", "headname": "h1"})
    # Same kind, different headname — must NOT be allowed just because a
    # valid auth exists on the session at all.
    error = auth.check_load_authorized(bundle, "l2", outpath="/fake_root/L2/x", headname="h2")
    assert error is not None
    assert "not authorized" in error

    # Editing L1 *parameters* (not part of the dataset identity) around
    # an L1 authorization must still be allowed — only infiles matter.
    auth.apply_token(bundle, {"user": "alice", "allowed_surveys": ["WL2022A1"],
                               "kind": "l1", "infiles": ["a.fit", "b.fit"]})
    assert auth.check_load_authorized(bundle, "l1", infiles=["a.fit", "b.fit"]) is None
    assert auth.check_load_authorized(bundle, "l1", infiles=["c.fit"]) is not None


def test_record_dataset_loaded():
    """The non-token counterpart to apply_token — used by
    aps_explorer.py's handle_url_handoff for its plain deep-link branch
    (REQUIRE_WEAVEOR_AUTH off). Same _AUTH_SLOT shape, empty user/
    allowed_surveys since no real token was ever presented."""
    bundle = SessionBundle()
    assert auth.current_auth(bundle) is None

    auth.record_dataset_loaded(bundle, "l1", infiles=["b.fit", "a.fit"])
    stored = auth.current_auth(bundle)
    assert stored["user"] is None
    assert stored["allowed_surveys"] == set()
    assert stored["authorized_dataset"] == ("l1", ("a.fit", "b.fit"))  # sorted

    # Replaces, same as apply_token — a fresh deep link is a new
    # navigation event, not an accumulation.
    auth.record_dataset_loaded(bundle, "l2", outpath="/fake_root/L2/x", headname="h1")
    stored2 = auth.current_auth(bundle)
    assert stored2["authorized_dataset"] == ("l2", "/fake_root/L2/x", "h1")


def test_check_load_authorized_multi_session_no_record_rejected(monkeypatch):
    """REQUIRE_WEAVEOR_AUTH off, MULTI_SESSION on, nothing loaded yet for
    this session at all: the sidebar form must never be the thing that
    first picks which files a server-mode session shows — only a URL
    (token or deep link) may."""
    monkeypatch.setattr(auth, "REQUIRE_WEAVEOR_AUTH", False)
    monkeypatch.setattr(sess, "MULTI_SESSION", True)
    bundle = SessionBundle()
    error = auth.check_load_authorized(bundle, "l1", infiles=["a.fit"])
    assert error is not None
    assert "direct link" in error


def test_check_load_authorized_multi_session_matching_record_allowed(monkeypatch):
    """The "editable params, same files" flow, now generalized to plain
    deep links (no real weaveOR token) — matches record_dataset_loaded's
    own recorded identity exactly."""
    monkeypatch.setattr(auth, "REQUIRE_WEAVEOR_AUTH", False)
    monkeypatch.setattr(sess, "MULTI_SESSION", True)
    bundle = SessionBundle()
    auth.record_dataset_loaded(bundle, "l1", infiles=["a.fit", "b.fit"])
    assert auth.check_load_authorized(bundle, "l1", infiles=["a.fit", "b.fit"]) is None
    # Order-independence, same as the token-based case.
    assert auth.check_load_authorized(bundle, "l1", infiles=["b.fit", "a.fit"]) is None


def test_check_load_authorized_multi_session_mismatched_record_rejected(monkeypatch):
    monkeypatch.setattr(auth, "REQUIRE_WEAVEOR_AUTH", False)
    monkeypatch.setattr(sess, "MULTI_SESSION", True)
    bundle = SessionBundle()
    auth.record_dataset_loaded(bundle, "l2", outpath="/fake_root/L2/x", headname="h1")
    error = auth.check_load_authorized(bundle, "l2", outpath="/fake_root/L2/x", headname="h2")
    assert error is not None
    assert "different dataset" in error


def test_check_load_authorized_multi_session_honors_real_token_too(monkeypatch):
    """A real weaveOR token (apply_token) populates the exact same
    _AUTH_SLOT record_dataset_loaded does — the MULTI_SESSION branch
    doesn't care which of the two populated it."""
    monkeypatch.setattr(auth, "REQUIRE_WEAVEOR_AUTH", False)
    monkeypatch.setattr(sess, "MULTI_SESSION", True)
    bundle = SessionBundle()
    auth.apply_token(bundle, {"user": "alice", "allowed_surveys": ["WL2022A1"],
                               "kind": "l2", "outpath": "/fake_root/L2/x", "headname": "h1"})
    assert auth.check_load_authorized(bundle, "l2", outpath="/fake_root/L2/x", headname="h1") is None
    assert auth.check_load_authorized(bundle, "l2", outpath="/fake_root/L2/x", headname="h2") is not None


def test_check_load_authorized_standalone_unaffected_by_multi_session_flag(monkeypatch):
    """Both REQUIRE_WEAVEOR_AUTH and MULTI_SESSION off: always None,
    completely unrestricted, exactly as before either mode existed —
    the literal "editing standalone mode should be provably impossible"
    requirement this whole design has followed throughout."""
    monkeypatch.setattr(auth, "REQUIRE_WEAVEOR_AUTH", False)
    monkeypatch.setattr(sess, "MULTI_SESSION", False)
    bundle = SessionBundle()  # nothing recorded, nothing ever will be
    assert auth.check_load_authorized(bundle, "l1", infiles=["anything.fit"]) is None
    assert auth.check_load_authorized(bundle, "l2", outpath="/x", headname="y") is None


def test_locked_load_form_never_renders_the_real_path(monkeypatch):
    """The actual privacy property server mode needs, checked directly
    against aps_explorer.py's own _load_form() output rather than just
    the authorization *decision* tested above: once a dataset is
    recorded for a session, its real file path must not appear anywhere
    in the rendered component tree at all -- not even as a disabled
    field's value, which page source would still reveal. Uses the
    default bundle directly (no live server/test client needed) since
    _load_form() reads aps_explorer_session.current_bundle() itself."""
    from PyAPS import aps_explorer as ex
    from PyAPS.aps_explorer_session import _DEFAULT_BUNDLE

    monkeypatch.setattr(sess, "MULTI_SESSION", True)
    # _DEFAULT_BUNDLE is a module-level singleton (used whenever there's
    # no real Flask request bound, exactly this test's situation) --
    # must not leak this test's recorded dataset into any other test
    # that happens to read the default bundle afterward.
    saved_slots = dict(_DEFAULT_BUNDLE.slots)
    try:
        locked_path = "/fake_root/L1/20260101/super_secret_stack_9999999.fit"
        auth.record_dataset_loaded(_DEFAULT_BUNDLE, "l1", infiles=[locked_path])

        rendered = str(ex._load_form())
        assert locked_path not in rendered
        assert "Dataset fixed for this session" in rendered
        # The component still exists (empty) -- Dash's own State
        # resolution for handle_l1_load needs it present, even though
        # it's never populated with anything real once locked.
        assert "lf-infiles" in rendered
    finally:
        _DEFAULT_BUNDLE.slots.clear()
        _DEFAULT_BUNDLE.slots.update(saved_slots)


def test_log_panel_redacts_real_paths_in_server_mode_only(monkeypatch):
    """Explicit request: "the log... shows where we are reading the data
    from the server data directory and also... reveal many secure information...
    fine for standalone but not here." The pipeline's own print()s bake
    real absolute paths directly into their text -- LOG.add() is the one
    place every single one of them (this file's own prints, and every
    print()/traceback anywhere in APSOB/aps_calib/etc., all funneled
    through _StreamTee) passes through before ever reaching the browser,
    so that's where this is fixed, not by chasing individual print call
    sites. Real log lines from live testing, used verbatim."""
    from PyAPS import aps_explorer as ex

    monkeypatch.setattr(sess, "MULTI_SESSION", True)
    cases = {
        ">>> Loading L1 dataset: ['/fake_root/L1/20240105/single_3039505.fit', "
        "'/fake_root/L1/20240105/single_3039506.fit']":
            ">>> Loading L1 dataset: ['single_3039505.fit', 'single_3039506.fit']",
        "[ARM 0] Starting vectorized processing: /fake_root/L1/20240105/single_3039506.fit":
            "[ARM 0] Starting vectorized processing: single_3039506.fit",
        ">>> Loading L2 product: /fake_root/L2/20240105/single_3039506__single_3039505_APS.fits":
            ">>> Loading L2 product: single_3039506__single_3039505_APS.fits",
        # A real WEAVE filename contains "+" (declination sign) -- must survive.
        "Analysing /fake_root/L1/20240105/LWVE_08370302+6946308_01_BR_L1_P0001_APS.fits file":
            "Analysing LWVE_08370302+6946308_01_BR_L1_P0001_APS.fits file",
        "a line with no path at all": "a line with no path at all",
    }
    for line, expected in cases.items():
        assert ex._redact_paths(line) == expected

    monkeypatch.setattr(sess, "MULTI_SESSION", False)
    unredacted = "/fake_root/L1/20240105/single_3039506.fit"
    assert ex._redact_paths(unredacted) == unredacted, "standalone mode must stay completely unaffected"


def test_log_buffer_add_applies_redaction_in_server_mode(monkeypatch):
    """One level up from _redact_paths itself -- proves LOG.add() (the
    real thing every print()/traceback funnels through) actually calls
    it, not just that the helper function works in isolation."""
    from PyAPS import aps_explorer as ex

    monkeypatch.setattr(sess, "MULTI_SESSION", True)
    buf = ex._LogBuffer()
    buf.add("", "Loading /fake_root/L1/20240105/single_3039506.fit now")
    snapshot = buf.snapshot()
    assert len(snapshot) == 1
    assert "/fake_root" not in snapshot[0]
    assert "single_3039506.fit" in snapshot[0]


def test_l1_deep_link_falls_back_to_default_caldir_catdir(monkeypatch):
    """The real bug this guards against: a weaveOR handoff (token or
    plain deep link) has no clean source for caldir/catdir at all (see
    weave/lib/aps_explorer_handoff.py's own docstring on the weaveOR
    side) -- confirmed live, every L1 dataset opened this way loaded
    with zero LSF/FWHM diagnostics even though the container genuinely
    has read access to the real archive. Checks the plain deep-link
    branch directly (mocking _load_l1 to capture what it was actually
    called with, rather than needing real L1 data to load) -- the token
    branch a few lines above it in the same function applies the exact
    same `claims.get(...) or _sess.DEFAULT_CALDIR` expression, so this
    is a direct, not just analogous, proof of that logic too."""
    from PyAPS import aps_explorer as ex
    from PyAPS.aps_explorer_session import _DEFAULT_BUNDLE

    monkeypatch.setattr(sess, "MULTI_SESSION", False)
    monkeypatch.setattr(auth, "REQUIRE_WEAVEOR_AUTH", False)
    monkeypatch.setattr(sess, "DEFAULT_CALDIR", "/fake_root/CAL")
    monkeypatch.setattr(sess, "DEFAULT_CATDIR", "/fake_root/CAT")

    captured = {}

    def fake_load_l1(args):
        captured["caldir"] = args.caldir
        captured["catdir"] = args.catdir

    monkeypatch.setattr(ex, "_load_l1", fake_load_l1)
    monkeypatch.setattr(ex, "_central_item", lambda kind: "fake-item")

    # record_dataset_loaded (called unconditionally by this branch) writes
    # into the shared default bundle's own slots (outside any Flask
    # request, current_bundle() resolves to _DEFAULT_BUNDLE) -- save/
    # restore exactly like test_load_handlers_reject_cleanly_... above,
    # so this doesn't leak "a dataset is loaded" state into later tests.
    saved_slots = dict(_DEFAULT_BUNDLE.slots)
    try:
        # No caldir/catdir in the deep link at all -> must fall back.
        ex.handle_url_handoff("?kind=l1&infiles=/tmp/a.fit", 0)
        assert captured["caldir"] == "/fake_root/CAL"
        assert captured["catdir"] == "/fake_root/CAT"

        # An explicit caldir/catdir in the deep link must still win over
        # the default -- this is a fallback, not an override.
        captured.clear()
        ex.handle_url_handoff("?kind=l1&infiles=/tmp/a.fit&caldir=/explicit/cal&catdir=/explicit/cat", 0)
        assert captured["caldir"] == "/explicit/cal"
        assert captured["catdir"] == "/explicit/cat"
    finally:
        _DEFAULT_BUNDLE.slots.clear()
        _DEFAULT_BUNDLE.slots.update(saved_slots)


def test_l1_form_load_falls_back_to_default_caldir_catdir(monkeypatch):
    """Same fallback, the sidebar Load/Reload form's own path -- a
    blank caldir/catdir submitted from the browser (the first load
    after a handoff that never had one to prefill, or a manually
    cleared field) shouldn't silently lose LSF/FWHM either."""
    from PyAPS import aps_explorer as ex
    from PyAPS import aps_l1_preview as l1_mod
    from PyAPS.aps_explorer_session import _DEFAULT_BUNDLE

    monkeypatch.setattr(sess, "MULTI_SESSION", False)
    monkeypatch.setattr(auth, "REQUIRE_WEAVEOR_AUTH", False)
    monkeypatch.setattr(sess, "DEFAULT_CALDIR", "/fake_root/CAL")
    monkeypatch.setattr(sess, "DEFAULT_CATDIR", "/fake_root/CAT")

    captured = {}

    def fake_load_from_form_fields(infiles_text, infiles_list, l1ref, l2ref, apsids, targsrvy, targclass,
                                    maskids, area, mask_areas, wlranges, arms_ratio, caldir, catdir,
                                    configdir, flags, **kwargs):
        captured["caldir"] = caldir
        captured["catdir"] = catdir

    monkeypatch.setattr(l1_mod, "load_from_form_fields", fake_load_from_form_fields)
    monkeypatch.setattr(ex, "_central_item", lambda kind: "fake-item")

    # handle_l1_load sets EXPLORER.kind="l1" itself on a successful load
    # (aps_explorer.py:3564, right after load_from_form_fields returns) --
    # on the shared default bundle here (no Flask request context), so
    # this must be saved/restored exactly like the kind-mismatch test
    # above, or a later test building the real layout (e.g.
    # test_session_cookie_roundtrip_via_test_client) inherits
    # EXPLORER.kind == "l1" with no real L1 state ever loaded behind it
    # and crashes in _central_item reading a None coord_arr. Confirmed
    # this is a real, not hypothetical, failure mode -- caught by the
    # full suite before this guard was added.
    saved_slots = dict(_DEFAULT_BUNDLE.slots)
    try:
        ex.handle_l1_load(
            run_clicks=1, infiles_text="/tmp/a.fit", infiles_list=None, l1ref=None, l2ref=None,
            apsids=None, targsrvy=None, targclass=None, maskids=None, decimate_stride=None,
            area=None, mask_areas=None, wlranges=None, arms_ratio=None, caldir=None, catdir=None,
            configdir=None, flags=None, advanced_flags=None, lsftype=None, ivar_norm_mode=None,
            edge_pixels=None, gap_offset_pix=None, funit=None, template_sigma0=None, version=1,
        )
        assert captured["caldir"] == "/fake_root/CAL"
        assert captured["catdir"] == "/fake_root/CAT"
    finally:
        _DEFAULT_BUNDLE.slots.clear()
        _DEFAULT_BUNDLE.slots.update(saved_slots)


def test_load_handlers_reject_cleanly_on_kind_mismatch_instead_of_crashing(monkeypatch):
    """A real crash hit during live testing, not a hypothetical: browsers
    share one pyaps_sid cookie across every same-origin tab, so opening
    an L1 dataset link and an L2 dataset link from weaveOR in two
    different tabs (both target="_blank") lands both in the *same*
    session -- whichever loaded second overwrites _AUTH_SLOT (apply_token/
    record_dataset_loaded both replace, not accumulate). The *other*,
    now-stale tab's own Load/Reload button would then try to unpack the
    new kind's authorized_dataset tuple as if it were its own shape --
    confirmed live: "ValueError: too many values to unpack" (L2's
    3-tuple read as L1's 2-tuple). Both handlers must reject cleanly
    instead. Calls the real handlers directly (Dash's own @app.callback
    registers but does not wrap -- the plain function stays callable)."""
    from PyAPS import aps_explorer as ex
    from PyAPS.aps_explorer_session import _DEFAULT_BUNDLE

    monkeypatch.setattr(sess, "MULTI_SESSION", True)
    saved_slots = dict(_DEFAULT_BUNDLE.slots)
    try:
        # Session currently recorded for L1 -> the L2 handler must reject,
        # not crash trying to unpack a 2-tuple as (kind, outpath, headname).
        auth.record_dataset_loaded(_DEFAULT_BUNDLE, "l1", infiles=["/tmp/a.fit"])
        result = ex.handle_l2_load(1, None, "/some/outpath", "headname", 1)
        error_msg = result[-1]
        assert "different dataset" in error_msg

        # Session currently recorded for L2 -> the L1 handler must reject,
        # not crash trying to unpack a 3-tuple as (kind, infiles_tuple).
        auth.record_dataset_loaded(_DEFAULT_BUNDLE, "l2", outpath="/x", headname="h")
        result = ex.handle_l1_load(
            run_clicks=1, infiles_text="/tmp/b.fit", infiles_list=None, l1ref=None, l2ref=None,
            apsids=None, targsrvy=None, targclass=None, maskids=None, decimate_stride=None,
            area=None, mask_areas=None, wlranges=None, arms_ratio=None, caldir=None, catdir=None,
            configdir=None, flags=None, advanced_flags=None, lsftype=None, ivar_norm_mode=None,
            edge_pixels=None, gap_offset_pix=None, funit=None, template_sigma0=None, version=1,
        )
        error_msg = result[-1]
        assert "different dataset" in error_msg
    finally:
        _DEFAULT_BUNDLE.slots.clear()
        _DEFAULT_BUNDLE.slots.update(saved_slots)


def test_verify_l1_surveys_allowed(monkeypatch):
    monkeypatch.setattr(auth, "REQUIRE_WEAVEOR_AUTH", False)
    # No-op when the mode is off, regardless of content.
    assert auth.verify_l1_surveys_allowed(["WS9999"], ["WL2022A1"]) is None

    monkeypatch.setattr(auth, "REQUIRE_WEAVEOR_AUTH", True)
    assert auth.verify_l1_surveys_allowed([], ["WL2022A1"]) is None  # nothing to check
    assert auth.verify_l1_surveys_allowed(["WL2022A1"], ["WL2022A1", "WL2022A2"]) is None
    error = auth.verify_l1_surveys_allowed(["WL2022A1", "WS9999"], ["WL2022A1"])
    assert error is not None
    assert "WS9999" in error


def test_weaveor_entry_redirect_via_real_test_client(monkeypatch):
    """A direct, tokenless visit to the base page while
    REQUIRE_WEAVEOR_AUTH is on used to render the normal (empty,
    useless without a dataset) explorer shell -- confirmed live, real
    user complaint: "entering .../weave/ on a non-login browser
    redirects me to the aps_explorer page with empty contents". Now it
    shows a small interstitial pointing back to WeaveOR instead.
    Real Flask test_client() dispatch, matching
    test_session_cookie_roundtrip_via_test_client's own established
    pattern in test_aps_explorer_session.py, rather than calling the
    before_request hook directly -- this is specifically testing that
    hook ordering/registration wiring works, not just its own logic in
    isolation."""
    from PyAPS import aps_explorer as ex
    from PyAPS import aps_explorer_session as sess

    monkeypatch.setattr(sess, "MULTI_SESSION", True)
    monkeypatch.setattr(auth, "REQUIRE_WEAVEOR_AUTH", False)
    client = ex.app.server.test_client()

    # Off (the default state): unchanged,
    # normal Dash page, no interstitial.
    r_off = client.get(sess.URL_PREFIX)
    assert r_off.status_code == 200
    assert b"This page isn't accessible directly" not in r_off.data

    monkeypatch.setattr(auth, "REQUIRE_WEAVEOR_AUTH", True)
    monkeypatch.setenv("PYAPS_EXPLORER_WEAVEOR_URL", "https://upstream.example.org/app/")

    # On, no token, no prior auth recorded for this session: redirected
    # to the interstitial, not the normal (empty) app shell.
    r_notoken = client.get(sess.URL_PREFIX)
    assert r_notoken.status_code == 200
    assert b"This page isn't accessible directly" in r_notoken.data
    assert b"https://upstream.example.org/app/" in r_notoken.data

    # On, with a token in the query string: a real handoff in progress
    # -- must NOT be intercepted here, handle_url_handoff gets to verify
    # it for real (a garbage token just fails there, same as before this
    # feature existed; this check only cares that it wasn't pre-empted).
    r_token = client.get(sess.URL_PREFIX + "?token=garbage")
    assert r_token.status_code == 200
    assert b"This page isn't accessible directly" not in r_token.data

    # On, but this session already has a real dataset authorized
    # (e.g. reloading the page after a genuine handoff already
    # succeeded): must NOT be redirected away from its own loaded data.
    # sess.current_bundle() can't be used directly here -- called
    # outside a real request (as this test function body is), it
    # resolves to _DEFAULT_BUNDLE, not this test_client's own
    # cookie-bound session -- confirmed the hard way, this test's own
    # first draft failed exactly this way. Same sid-from-Set-Cookie
    # pattern test_idle_kill_trigger_evicts_this_session_via_real_
    # callback_dispatch already established for the identical problem.
    sid = r_off.headers.get("Set-Cookie", "").split(f"{sess.SESSION_COOKIE_NAME}=")[1].split(";")[0]
    bundle = sess._sessions[sid]
    auth.record_dataset_loaded(bundle, "l2", outpath="/x", headname="h")
    try:
        r_authed = client.get(sess.URL_PREFIX)
        assert r_authed.status_code == 200
        assert b"This page isn't accessible directly" not in r_authed.data
    finally:
        bundle.slots.pop(auth._AUTH_SLOT, None)


def test_weaveor_entry_redirect_only_gates_the_base_page(monkeypatch):
    """Assets and /healthz must stay reachable even when
    REQUIRE_WEAVEOR_AUTH is on and this session has no token -- a
    container orchestrator's liveness probe, and the page's own CSS/JS,
    would otherwise get the interstitial HTML back instead of what they
    actually asked for."""
    from PyAPS import aps_explorer as ex
    from PyAPS import aps_explorer_session as sess

    monkeypatch.setattr(sess, "MULTI_SESSION", True)
    monkeypatch.setattr(auth, "REQUIRE_WEAVEOR_AUTH", True)
    client = ex.app.server.test_client()

    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.get_json() == {"status": "ok"}
