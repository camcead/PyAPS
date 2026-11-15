# Prod: work that is deliberately NOT here yet

Last updated 2026-11-15. This is the prod copy of the list. The checklist with the steps lives on
branch `dev` in the file of the same name; tick items there, and remove the entry here when it is promoted.

## 1. FERRE metallicity columns `FEH` / `FEH_ERR` renamed `MH` / `MH_ERR` (dev commit `c1cac86`)

- **State on prod:** not applied. `aps_ferre` still writes `FEH` / `FEH_ERR` in `FR_TABLE` of
  `ferre_<head>.fits` and in the L2 `STAR_TABLE`. Everything that reads those files (FESWI, weaveOR,
  validator, WAS) keeps working. Do not change that without the steps below.
- **Why it is held back now:** the checks are done (the FESWI check on a real OB, the readers, the pod tests; see the dev copy of this
  file) and the owner said go on 2026-11-15. What is left is to deploy the CASPER pod tools first (they must read `MH` before the first file
  with it exists), then push the prepared branch `ready-promote-mh` (worktree `wt_integ_prod`, commit 8a85d82) to prod and update the pods.
- **Do not** cherry-pick `c1cac86` or merge `dev` wholesale into prod while this entry is open; promote
  other dev work commit by commit, and check `git diff origin/prod origin/dev --stat` before every merge.
- **FESWI (to do on prod):** on dev FESWI keeps working through `aps_feswi.feswi_param_file` (a temporary copy
  of `ferre_<head>.fits` with `FEH` / `FEH_ERR` added) and `doc/feswi_accept_mh.patch` (the same fix inside FESWI).
  Neither is on prod yet. They must go to prod in the same cherry-pick as the rename: `py/PyAPS/aps_feswi.py`,
  `tests/test_feswi_mh_compat.py`, `doc/feswi_accept_mh.patch`, the `CS/README.md` paragraph. Before that, one
  real `aps_feswi` run on the dev pod must give the same `*-feswi.fits` as before the rename.
- **When it may come here:** after every box in the dev copy of this file is ticked and the owner says go.
  The promotion is: cherry-pick `c1cac86` plus its follow-ups, run `tests/test_ferre_mh_columns.py` and the
  FERRE and L2 merge tests on the pod, push, then `pyaps-admin pod update` (pauses and resumes the runner).
- **If a user asks why `FEH` is still there on prod:** the rename is planned, not rejected; it waits for FESWI.
