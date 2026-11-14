# Prod: work that is deliberately NOT here yet

Last updated 2026-11-14. This is the prod copy of the list. The checklist with the steps lives on
branch `dev` in the file of the same name; tick items there, and remove the entry here when it is promoted.

## 1. FERRE metallicity columns `FEH` / `FEH_ERR` renamed `MH` / `MH_ERR` (dev commit `c1cac86`)

- **State on prod:** not applied. `aps_ferre` still writes `FEH` / `FEH_ERR` in `FR_TABLE` of
  `ferre_<head>.fits` and in the L2 `STAR_TABLE`. Everything that reads those files (FESWI, weaveOR,
  validator, WAS) keeps working. Do not change that without the steps below.
- **Why it is held back:** the owner asked on 2026-11-14 to keep it on dev. FESWI reads `FEH` / `FEH_ERR`
  and would fail with a KeyError on files written with the new names.
- **Do not** cherry-pick `c1cac86` or merge `dev` wholesale into prod while this entry is open; promote
  other dev work commit by commit, and check `git diff origin/prod origin/dev --stat` before every merge.
- **When it may come here:** after every box in the dev copy of this file is ticked and the owner says go.
  The promotion is: cherry-pick `c1cac86` plus its follow-ups, run `tests/test_ferre_mh_columns.py` and the
  FERRE and L2 merge tests on the pod, push, then `pyaps-admin pod update` (pauses and resumes the runner).
- **If a user asks why `FEH` is still there on prod:** the rename is planned, not rejected; it waits for FESWI.
