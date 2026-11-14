# Dev-only work waiting for promotion to prod

Last updated 2026-11-14. Work goes dev -> prod; public `main` is only for public releases.
This file lists what is on `dev` and deliberately NOT on `prod` yet. Remove an entry when it is promoted.

## 1. FERRE metallicity columns `FEH` / `FEH_ERR` renamed `MH` / `MH_ERR`

- **Where:** dev commit `c1cac86` (thastro, 2026-11-12) with the test `tests/test_ferre_mh_columns.py`,
  CHANGELOG entry "FERRE: metallicity columns renamed", `doc/aps_ferre.md` section 12,
  `doc/weave_datamodel_v8.md`.
- **Decision:** the owner asked on 2026-11-14 to keep it on dev for now. Not promoted.
- **What changes for users:** `FR_TABLE` of `ferre_<head>.fits` and the L2 `STAR_TABLE` (TTYPE cards
  included) get `MH` / `MH_ERR` instead of `FEH` / `FEH_ERR`. Values, units and positions are unchanged.
  Readers in PyAPS accept both names, so old L2 files still work.

### Before promoting (check each, then tick)

- [ ] **FESWI.** The FESWI code installed from `CS/README.md` reads `FEH` / `FEH_ERR` and fails with a
      KeyError on files written with the new names. Its reader must accept `MH` / `MH_ERR` first,
      or FESWI must be pinned to the old names. Ask the FESWI owner.
- [ ] **Other readers of L2 `STAR_TABLE`.** WEAVE-ICD-027 v8.00 still lists `FEH` / `FEH_ERR`.
      Check weaveOR (spectrum and target pages), the WAS/validator checks, weavehub and the CASPER tools
      for any use of the old column names (`grep -rn "FEH"` in each repo) and decide whether an ICD
      change request is needed.
- [ ] **Data already on disk.** Existing L2 files keep `FEH`; new ones get `MH`. Decide whether
      to leave both generations or reprocess. Do NOT reprocess without an owner decision.
- [ ] **Name.** The commit message says `M_H`; the code uses `MH`. Confirm `MH` / `MH_ERR` with thastro.
- [ ] **Tests on the pod** (`~/venv`, `PYTHONPATH=py`): `tests/test_ferre_mh_columns.py` plus the FERRE and
      L2 merge tests. The 78 golden SLURM-resource failures are the known baseline.
- [ ] **Owner says go.** Then: cherry-pick `c1cac86` and its follow-up commits onto `prod`, push,
      and update the pod with `pyaps-admin pod update` (it pauses the runner and resumes it).

## Notes

- On 2026-11-14 the tree difference between prod and dev (before this file) was exactly the rename:
  16 files (CHANGELOG, four L2 template FITS, docs, FERRE and its three readers, the new test).
  Everything else on dev (FERRE worker fix, dynamic ExGal widths, graphify files) is on prod under
  other commit hashes. Compare trees, not hashes: `git diff origin/prod origin/dev --stat`.
