# The `casper` branch of github.com/camcead/PyAPS

`casper` is the branch that members of the WEAVE project install: **the latest prod commit that has proved itself**.

| Branch | What it is |
|---|---|
| `main` | frozen until the PyAPS paper is approved; the version the paper describes. Not touched. |
| `dev` | the next release on trial. Operational branch. |
| `prod` | what the WEAVE pipeline runs now. Operational branch. |
| `casper` | the stable public version for CASPER users: a prod commit promoted by hand (below), plus the folder `standalone/` that holds this tool |

All of them are public on GitHub. What keeps unfinished work out of `casper` is the promotion rule, not secrecy.

## Promotion rule (dev, then prod, then casper)

A prod commit is promoted to `casper` only when **all three** hold:
1. it is the commit the Imperial pod is running now (read its head with the version-coherence check or `pod_code_state`);
2. the test suite gives the same result as the last known-good run of prod;
3. the owner says "promote".

Nobody commits to `casper` directly, except the folder `standalone/` (this tool). The move is `tools/promote_casper.sh` of the weave-standalone repository: dry run by default; it needs `--pod-head <sha>` (the head read from the pod) and refuses if that is not the commit being promoted. It merges the prod commit into `casper`, so `standalone/` stays.

No GitHub *release* is made from `casper` (only the branch), so the Zenodo record of the paper is not triggered.
