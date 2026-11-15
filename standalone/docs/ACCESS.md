# Who may install what (pyaps-standalone)

| Access level | Set how | May install |
|---|---|---|
| `member` (default) | nothing to do | the branch `casper` of `github.com/camcead/PyAPS`: the stable public version (see `CASPER_BRANCH.md`) |
| `admin` | `python -m pyaps_standalone.cli set-access admin` (writes `~/.config/casper-pyaps-standalone/config.json`) | any branch or tag the remote has, e.g. `casper`, `prod`, `dev`, `main` (the frozen branch of the PyAPS paper); `pyaps_refs` lists them |

Use `admin` only if you are an administrator of the WEAVE project. CASPER's setup skill for WEAVE administrators does it for them; members are never told how.

**What this rule is and is not.** The GitHub repository `camcead/PyAPS` is public, and so are its `dev` and `prod` branches (checked 2026-11-15 with the unauthenticated GitHub API: both answer 200). Nothing can hide them, so the member rule does not protect secrets; it keeps members on released code and keeps the operational branches for the people who run them. A member who edits the config file or runs `git` by hand can install any branch; that is allowed by the licence and not a breach.
The private operator layer `PyAPS_admin` is **not** handled by this tool at all.

Operational branches carry a warning in the plan (`prod` is what the pipeline runs now, `dev` is the next release on trial).
