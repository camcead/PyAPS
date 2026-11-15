# pyaps-standalone

Install and use **PyAPS** (the WEAVE Advanced Processing System) on your own machine through Claude, with a local MCP server. Public code: it carries no secret and installs only public PyAPS.

* Members of the WEAVE project install the branch **`casper`** of `github.com/camcead/PyAPS`: the latest prod commit that is running and has passed its tests (`docs/CASPER_BRANCH.md`).
* Administrators of the WEAVE project may choose any branch or tag (`prod`, `dev`, `main`, ...), see `docs/ACCESS.md`.
* Every install is a plan first and a second confirmed call; long steps run as background jobs.

This folder is `standalone/` on the `casper` branch. Install: `docs/INSTALL.md`. Tests: `python -m pytest tests`.
