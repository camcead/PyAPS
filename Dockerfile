# syntax=docker/dockerfile:1
#
# PyAPS Explorer — server/Docker image for aps_explorer.py's multi-user
# mode. Standalone/single-user usage (`pip install PyAPS`, then
# `aps-explorer` or `python aps_explorer.py`) never touches this file at
# all — see wsgi.py and aps_explorer_session.py for the mode this image
# turns on.
#
# Build:
#   docker build -t pyaps-explorer .
#
# Run (bind-mount the real data root read-only rather than baking it in —
# swap the host path for wherever your data lives):
#   docker run -d --name pyaps-explorer -p 8080:8080 \
#       -v <PYAPS_DATA>:/data:ro \
#       pyaps-explorer
# (add -e PYAPS_EXPLORER_URL_PREFIX=/pyaps/ too if this is sharing a host
# with other projects — see below.)
#
# Requiring an authenticated handoff from a trusted upstream app (e.g.
# weaveOR) instead of open access — see aps_explorer_auth.py's own
# module docstring for the full design — needs two more env vars:
#   -e PYAPS_EXPLORER_REQUIRE_WEAVEOR_AUTH=1 \
#   -e PYAPS_EXPLORER_WEAVEOR_SECRET=<same value the token-issuing app signs with> \
# Off by default. Even off, MULTI_SESSION mode (the default in this
# image) never lets the sidebar form itself pick which files a session
# shows -- only a URL (a real token, or the plain ?kind=l1&infiles=...
# deep link this off-mode still accepts) can; the form can only reload
# whatever's already loaded with different *processing* parameters. See
# aps_explorer_auth.check_load_authorized's own docstring.
#
# PYAPS_CONFIGDIR: a writable, persistent config/cache directory --
# needed whenever this image is built from a checkout that doesn't ship
# the repo-bundled configs/ExGal_configs (this is also where LSF/FWHM
# interpolator cache pickles land, kept off any read-only data mount):
#   -v <PYAPS_CONFIGDIR>:<PYAPS_CONFIGDIR> \
#   -e PYAPS_CONFIGDIR=<PYAPS_CONFIGDIR> \
#
# PYAPS_EXPLORER_IDLE_TIMEOUT_MINUTES (default 10): minutes of genuine
# client-side inactivity (no mouse/keyboard/click/scroll -- not just
# "tab still open", which the log panel's own 700ms poll would keep
# looking active forever) before a session's memory is freed. Only
# active in MULTI_SESSION mode.
#
# Sharing one host across several projects, each under its own path
# (e.g. hub.example.org/pyaps/ for this deployment, hub.example.org/
# otherproject/ for another one entirely) — see aps_explorer_session.py's
# own comment on PYAPS_EXPLORER_URL_PREFIX for why this has to be an env
# var, not a CLI flag:
#   -e PYAPS_EXPLORER_URL_PREFIX=/pyaps/ \
# Unset (default "/"): the app is mounted at the host's own root, exactly
# as documented above — this is the only knob here that changes what
# your reverse proxy in front of this container needs to do. With a
# prefix set, proxy that exact path straight through unchanged (no
# rewriting needed — Dash itself already generates every internal
# asset/callback URL under the configured prefix):
#   location /pyaps/ { proxy_pass http://<this-container>:8080/pyaps/; }
# /healthz (used by this image's own HEALTHCHECK below) is a plain Flask
# route outside Dash's own routing and always stays unprefixed
# (http://<container>:8080/healthz) regardless of PYAPS_EXPLORER_URL_PREFIX
# — add a second, separate proxy rule for it if your reverse proxy itself
# (rather than something hitting the container directly) needs it too.
#
# Multi-stage: the build toolchain (gfortran, for the astronomy packages'
# compiled extensions) isn't retained in the final image.

FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gfortran \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml version.txt README.md ./
COPY py/ py/

# The "[server]" extra installs the explorer's own lean dependency set
# (see pyproject.toml) plus gunicorn -- deliberately NOT "[pipeline]":
# PyQt5/pyqtgraph/redrock/rvspecfit/ppxf and the rest of the CLI pipeline's
# dependencies are never imported by the explorer's code path and would only
# bloat this image.
RUN pip install --no-cache-dir --prefix=/install ".[server]"

FROM python:3.11-slim AS runtime

# libgomp1: OpenMP runtime some of the compiled scientific deps link
# against (scipy/scikit-learn wheels commonly need it even though it's
# not an explicit Python dependency).
#
# The container runs as an unprivileged user. If your data mount is group- or
# owner-restricted, build with the UID/GID that can read it:
#   docker build --build-arg PYAPS_UID=<uid> --build-arg PYAPS_GID=<gid> -t pyaps-explorer .
ARG PYAPS_UID=1000
ARG PYAPS_GID=1000
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid ${PYAPS_GID} pyaps \
    && useradd --create-home --uid ${PYAPS_UID} --gid ${PYAPS_GID} pyaps

COPY --from=builder /install /usr/local

ENV PYAPS_EXPLORER_MULTI_SESSION=1 \
    PYTHONUNBUFFERED=1

USER pyaps
WORKDIR /home/pyaps
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8080/healthz', timeout=3)"

# --workers 1 is not a tunable -- see wsgi.py's own docstring for why
# (session state is one in-process dict).
# --timeout 600 gives real headroom over the ~140s observed worst-case
# LSF/FWHM cache build (gunicorn's default 30s would kill that request).
# --threads is the one concurrency knob that is safe to raise: threads share
# the single process's session dict. Tune it to your host's core count.
ENV PYAPS_GUNICORN_THREADS=8
CMD ["sh", "-c", "exec gunicorn --workers 1 --worker-class gthread --threads ${PYAPS_GUNICORN_THREADS} --timeout 600 --bind 0.0.0.0:8080 PyAPS.wsgi:server"]
