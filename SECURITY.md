# Security Policy

## Supported versions
Security fixes are provided for the latest released version (currently 2.x).

## Reporting a vulnerability
Please **do not** open a public issue. Use GitHub's private reporting
("Security" tab -> "Report a vulnerability") or e-mail **amolaei@ast.cam.ac.uk** with the details
and, if possible, a way to reproduce. You will get an acknowledgement within a few working days and
we will keep you informed until a fix is released.

## Scope and deployment notes
* The interactive explorer (`aps-explorer`) is a Dash web application. Run it on `localhost` for
  personal use. For a shared deployment use the provided Docker image behind a reverse proxy with
  authentication (see the "Server / multi-user deployment" section of the README); the built-in
  token handoff (`--require-weaveor-auth`) needs a strong `PYAPS_EXPLORER_WEAVEOR_SECRET`.
* PyAPS reads and writes FITS files on paths you provide. Do not run it on untrusted files as a
  privileged user.
* Never commit credentials, tokens or real survey data to this repository.
