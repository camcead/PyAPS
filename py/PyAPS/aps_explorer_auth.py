"""
Cross-system authentication/authorization handoff from weaveOR.

**The problem.** `aps_explorer.py` has no authentication of its own by
design (see `aps_explorer_session.py`'s own module docstring —
standalone/local-install usage must stay zero-config). weaveOR (a
separate, existing web app — TurboGears2 + repoze.who + Beaker session
cookies, classic server-rendered stack; confirmed directly: no API
tokens/JWT/OAuth exist in it today) already authenticates its users and
knows which survey/programme each one may access. It needs to hand an
already-logged-in user off directly into a specific dataset view in the
explorer, without a second login — and **without that handoff turning
into a way to then browse arbitrary *other* data via the explorer's own
"Load different dataset" sidebar form** (explicit requirement: "users
cannot explore all data unless they are allowed or be member of
different groups").

**Why this is a capability-scoped session, not a general "logged in ->
browse anything" grant.** The explorer cannot, in general, independently
determine "what survey does this dataset belong to" for an arbitrary
path someone types into the sidebar form: `aps_utils.l1_fileinfo(...)
["srvys"]` gives a fast, catalog-only set of TARGSRVY codes for an *L1*
file, but L2 IFU-schema products (ExGal/Gal) carry no survey/programme
field anywhere in them at all (confirmed by direct code audit), and
MOS-schema L2 files only expose TARGSRVY via a real `CLASS_TABLE` read,
not the fast routing path. So authorization here is scoped to
*specific datasets weaveOR already vetted*, not reconstructed from file
introspection: a session that receives a valid token may load (and
freely re-load with edited parameters — caldir/catdir/wlranges/
sens_corr/etc., preserving the existing "editable params after load"
flow) only the *one* dataset that token named. Viewing a second dataset
needs a second token (a fresh handoff link from weaveOR) — this module
enforces that, function by function, but does not decide what counts as
"allowed" beyond what a token already claims.

**Token format**: `itsdangerous.URLSafeTimedSerializer`, HMAC-signed
with a shared secret (`PYAPS_EXPLORER_WEAVEOR_SECRET`, configured
identically on both sides — symmetric signing is deliberate: this is two
internally-trusted apps run by the same team, not a public multi-tenant
OAuth scenario, so asymmetric keys would be real over-engineering).
Already a transitive Flask dependency (confirmed: `itsdangerous` 2.2.0
installed, used internally by Flask itself for session-cookie signing)
— no new dependency. Short-lived (`PYAPS_EXPLORER_WEAVEOR_TOKEN_MAX_AGE`,
default 300s) — enough time to click through a redirect, not a standing
credential; the *session* (via `aps_explorer_session.py`'s existing
`SessionBundle`), not the token, is what persists authorization across
a page refresh — see `aps_explorer.py`'s `handle_url_handoff`, which
clears the token from the URL once consumed.

**Mode gate**: `REQUIRE_WEAVEOR_AUTH` (env var
`PYAPS_EXPLORER_REQUIRE_WEAVEOR_AUTH`, or `aps_explorer.py
--require-weaveor-auth`), default off — mirrors
`aps_explorer_session.MULTI_SESSION`'s exact pattern, so standalone
usage is provably unaffected until explicitly turned on.
`check_load_authorized()` is a pure no-op (always returns `None`,
meaning "allowed") whenever this is off.

Expected usage, per session, once a token has been verified:

    from PyAPS import aps_explorer_auth as _auth
    from PyAPS import aps_explorer_session as _sess

    claims = _auth.verify_token(token)          # raises TokenError
    _auth.apply_token(_sess.current_bundle(), claims)

    # ... later, on every load attempt (sidebar form or a fresh token):
    error = _auth.check_load_authorized(_sess.current_bundle(), "l2",
                                         outpath=outpath, headname=headname)
    if error:
        # show `error` to the user instead of loading
        ...
"""

from __future__ import annotations

import os

from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired

from PyAPS import aps_explorer_session as _sess


def _env_flag(name, default=False):
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() not in ("", "0", "false", "no", "off")


# Mutable at runtime (aps_explorer.py's --require-weaveor-auth flips this
# after argv parsing, before app.run()) — not a constant, so reads
# elsewhere go through this module rather than caching the value.
REQUIRE_WEAVEOR_AUTH = _env_flag("PYAPS_EXPLORER_REQUIRE_WEAVEOR_AUTH", False)

_SECRET = os.environ.get("PYAPS_EXPLORER_WEAVEOR_SECRET")
_TOKEN_MAX_AGE_SECONDS = int(os.environ.get("PYAPS_EXPLORER_WEAVEOR_TOKEN_MAX_AGE", 300))
_SALT = "pyaps-explorer-weaveor-handoff"

# Where a verified token's claims live on a SessionBundle (see
# aps_explorer_session.SessionBundle — a plain `.slots` dict; this key is
# set directly, bypassing get_or_create, since it's not lazily computed —
# it only exists once a token has actually been verified).
_AUTH_SLOT = "weaveor_auth"


class TokenError(Exception):
    """Raised for any invalid/expired/malformed handoff token. The
    message is written to be shown to the user directly (not a stack
    trace) — callers should catch this and surface `str(exc)`."""


def require_secret_configured():
    """Raises `RuntimeError` if `PYAPS_EXPLORER_WEAVEOR_SECRET` isn't
    set — called once at startup (`aps_explorer.explorer_worker()`) so
    an auth-required deployment fails fast and loudly rather than
    silently rejecting every token later with no clue why."""
    if not _SECRET:
        raise RuntimeError(
            "PYAPS_EXPLORER_REQUIRE_WEAVEOR_AUTH is on but "
            "PYAPS_EXPLORER_WEAVEOR_SECRET is not set — refusing to start "
            "an auth-required deployment with no way to verify tokens."
        )


def _serializer():
    require_secret_configured()
    return URLSafeTimedSerializer(_SECRET, salt=_SALT)


def verify_token(token):
    """Verify and decode a weaveOR handoff token. Returns the claims
    dict (`user`, `allowed_surveys`, `kind`, plus kind-specific dataset
    fields) on success; raises `TokenError` otherwise.

    Deliberately does NOT catch a missing-secret `RuntimeError` from
    `_serializer()` — that's a deployment misconfiguration, not a bad
    token, and must propagate loudly rather than get relabelled as
    "this link is invalid" (a real bug caught by this module's own test
    suite: an earlier version's `except Exception` was too broad and
    silently absorbed exactly this case)."""
    serializer = _serializer()
    try:
        return serializer.loads(token, max_age=_TOKEN_MAX_AGE_SECONDS)
    except SignatureExpired:
        raise TokenError("This link has expired — please go back to WeaveOR and open it again.")
    except BadSignature:
        raise TokenError("This link is invalid.")
    except Exception as e:
        raise TokenError(f"This link could not be read: {e}")


def dataset_identity(kind, **fields):
    """A hashable, comparable identity for "which dataset" — used both
    to record what a session was authorized for and to check a later
    load request against it.

    L2: `(kind, outpath, headname)`. L1: `(kind, tuple(sorted(infiles)))`
    — `caldir`/`catdir`/etc. are load *parameters*, not part of the
    dataset's own identity, so editing them (the existing "editable
    params after load" flow) doesn't require a new token; only loading a
    genuinely *different* set of input files does.
    """
    if kind == "l2":
        return ("l2", fields["outpath"], fields["headname"])
    if kind == "l1":
        infiles = fields.get("infiles") or []
        return ("l1", tuple(sorted(infiles)))
    raise ValueError(f"unknown dataset kind {kind!r}")


def apply_token(bundle, claims):
    """Store a freshly-verified token's claims onto `bundle` (an
    `aps_explorer_session.SessionBundle`), authorizing that session for
    exactly the one dataset the token names — replacing any previous
    authorization (a session moves to viewing a new dataset by
    presenting a new token, not by accumulating permissions)."""
    kind = claims["kind"]
    dataset_fields = {k: v for k, v in claims.items() if k not in ("user", "allowed_surveys", "kind")}
    bundle.slots[_AUTH_SLOT] = {
        "user": claims.get("user"),
        "allowed_surveys": set(claims.get("allowed_surveys") or []),
        "authorized_dataset": dataset_identity(kind, **dataset_fields),
        "kind": kind,
    }


def record_dataset_loaded(bundle, kind, **fields):
    """Records which dataset this session is now showing, *without* a
    verified weaveOR token — used by `aps_explorer.py`'s
    `handle_url_handoff` for its plain `?kind=l1&infiles=...` deep-link
    branch (only reachable when `REQUIRE_WEAVEOR_AUTH` is off). Writes
    the exact same `_AUTH_SLOT` shape `apply_token` does, just with
    empty `user`/`allowed_surveys` — `check_load_authorized`'s
    `MULTI_SESSION` branch below doesn't care which of the two populated
    it, only that *something* did, so the file-selection lock works
    identically whether a session arrived via a signed token or a plain
    deep link. Like `apply_token`, replaces any previous record — a
    fresh deep link is a new navigation event, not an accumulation."""
    bundle.slots[_AUTH_SLOT] = {
        "user": None,
        "allowed_surveys": set(),
        "authorized_dataset": dataset_identity(kind, **fields),
        "kind": kind,
    }


def current_auth(bundle):
    """The verified `{"user", "allowed_surveys", "authorized_dataset",
    "kind"}` dict for `bundle`, or `None` if this session has never
    presented a valid token *or* (see `record_dataset_loaded` above)
    loaded anything via a plain server-mode deep link."""
    return bundle.slots.get(_AUTH_SLOT)


def check_load_authorized(bundle, kind, **fields):
    """The one gate every load entry point (sidebar form or URL
    handoff) must call before actually loading data. Returns `None` if
    the load is allowed; returns a user-facing error string otherwise
    (never raises — callers show the string directly, e.g. into the
    existing `lf-status`/`lf-status-l1` output slots).

    `REQUIRE_WEAVEOR_AUTH` on: unchanged, strict weaveOR-token-only
    behavior. Off but `aps_explorer_session.MULTI_SESSION` on: a session
    may only load whatever dataset is already recorded for it (via a
    real token or a plain deep link — see `record_dataset_loaded`) with
    *different parameters* — never a genuinely different set of files;
    the sidebar "Load / Reload" form can adjust processing options
    (caldir/catdir/wlranges/sens_corr/vacuum/skysub/...) but can never be
    the thing that first picks, or later switches, which files a server-
    mode session shows — only a URL (token or deep link) can. Both off
    (standalone): always `None`, completely unrestricted, exactly as
    before this mode existed.
    """
    if REQUIRE_WEAVEOR_AUTH:
        auth = current_auth(bundle)
        if not auth:
            return "Please open this dataset from WeaveOR — direct access isn't available."
        requested = dataset_identity(kind, **fields)
        if requested != auth["authorized_dataset"]:
            return "You're not authorized to load a different dataset here — open it from WeaveOR instead."
        return None
    if _sess.MULTI_SESSION:
        auth = current_auth(bundle)
        if not auth:
            return "This server only opens datasets via a direct link — please use the link you were given."
        requested = dataset_identity(kind, **fields)
        if requested != auth["authorized_dataset"]:
            return "You can't switch to a different dataset here — open the new one via its own link."
        return None
    return None


def verify_l1_surveys_allowed(found_surveys, allowed_surveys):
    """L1-specific defense-in-depth, on top of `check_load_authorized`'s
    dataset-identity check: confirm the file's own survey codes (from a
    fast, catalog-only `aps_utils.l1_fileinfo(...)["srvys"]` read —
    kept in `aps_explorer.py`, which already imports that heavier chain;
    this module stays a leaf) are a subset of what the token actually
    allows. Catches drift between what weaveOR *thinks* it authorized
    and what the file actually contains — no equivalent check is
    possible for L2 IFU data, which carries no survey field at all (see
    this module's own docstring).

    Returns `None` if allowed (including when `found_surveys` is empty
    — nothing to check), or a user-facing error string otherwise.
    """
    if not REQUIRE_WEAVEOR_AUTH:
        return None
    if not found_surveys:
        return None
    disallowed = set(found_surveys) - set(allowed_surveys or [])
    if disallowed:
        return (f"This file contains survey(s) you're not authorized for "
                 f"({', '.join(sorted(disallowed))}) — open it from WeaveOR instead.")
    return None
