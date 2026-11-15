# Dev: work waiting for promotion to prod (the checklist)

Last updated 2026-11-15. Work goes dev -> prod; public `main` is only for public releases.
This file lists what is on `dev` and deliberately NOT on `prod` yet. Tick the boxes here. Prod has its own
copy of the file (what is absent there and why); remove the entry from both when it is promoted.

## 1. FERRE metallicity columns `FEH` / `FEH_ERR` renamed `MH` / `MH_ERR`

- **Where:** dev commit `c1cac86` (thastro, 2026-11-12) with the test `tests/test_ferre_mh_columns.py`,
  CHANGELOG entry "FERRE: metallicity columns renamed", `doc/aps_ferre.md` section 12,
  `doc/weave_datamodel_v8.md`.
- **Decision:** the owner asked on 2026-11-14 to keep it on dev for now; on 2026-11-15 the owner said go ahead. All checks below are done except the name with thastro; the push to prod waits for the CASPER pod-tool deploy (step 1 below).
- **What changes for users:** `FR_TABLE` of `ferre_<head>.fits` and the L2 `STAR_TABLE` (TTYPE cards
  included) get `MH` / `MH_ERR` instead of `FEH` / `FEH_ERR`. Values, units and positions are unchanged.
  Readers in PyAPS accept both names, so old L2 files still work.

### Before promoting (check each, then tick)

- [x] **FESWI (dev, 2026-11-14; real OB 2026-11-15).** FESWI reads `FEH` / `FEH_ERR`. `aps_feswi.feswi_param_file` gives it a
      copy with those names (`tests/test_feswi_mh_compat.py`); `doc/feswi_accept_mh.patch` is the same fix inside FESWI for its
      maintainers. **Real-OB check done 2026-11-15:** OB 26140 (night 20261106, head `single_3155960__single_3155959`), the
      production `CS_FESWI` command line: the `*-feswi.fits` of the renamed file equals the one of the old file in all four
      HDUs (only DATETIME and the checksums that follow from it differ); see the CHANGELOG entry "Investigated (FESWI on a
      real OB ...)". Found on the way and not caused by the rename: FESWI does not finish on the pod venv with pandas 3.0.6
      (`delim_whitespace`); patch for FESWI's maintainers `doc/feswi_pandas3.patch`; no FESWI job has run since 2026-10-25.
- [x] **Other readers of L2 `STAR_TABLE` (2026-11-15).** No use of the column in weaveOR (local and prod checkout on apm63),
      the validator, wdap, weavehub or the operational database (no `FEH` column in the data model reference 8.00_dyn).
      Only the CASPER `weave_files` library named it (L2 fit-parameter view, column classifier, key-column check): now reads
      both names, on casper `main` 9cb39369 / casper-mcp-servers 73f1e72 (CHANGELOGs 2026-11-15). **Still to do: deploy the
      pod tools** (`casper_deploy.py deploy arcus imperial --yes`; the classifier asked for the owner's own go) BEFORE the first
      file with `MH` exists, otherwise `weave_check_*` reports the metallicity as a missing key column. Also still to do,
      later: regenerate the packaged weave contract (`tools/sync_pyaps_contract.py`) once the rename is public and then drop
      `RENAMED_COLUMNS` in `tools_scan_w3.py`. **Open, for people, not code:** WEAVE-ICD-027 v8.00 still lists `FEH` / `FEH_ERR`
      in `STAR_TABLE`: decide with the WAS team whether a change request is needed.
- [x] **Data already on disk (default taken 2026-11-15).** Existing L2 files keep `FEH`, new ones get `MH`; every reader in
      PyAPS accepts both, so the two generations can stay side by side. NO reprocessing was started; that stays an owner decision.
- [ ] **Name.** The commit message says `M_H`; the code, the tests and the docs say `MH` / `MH_ERR` everywhere. Confirm with
      thastro that `MH` is the final name (cannot be done from here).
- [x] **Tests on the pod (2026-11-15, dev pod 128.232.226.106, `~/venv`, `PYTHONPATH=py`):** `tests/test_ferre_mh_columns.py`,
      `tests/test_feswi_mh_compat.py`, `tests/test_ifu_ferre_parallel_bins.py`, `tests/test_l2_job_failures.py`: 39 passed.
- [x] **Owner says go (2026-11-15: "go ahead with all still opens").** The promotion is PREPARED, not pushed: local branch
      `ready-promote-mh` (in the worktree `wt_integ_prod`, on the pod host workstation only, not on origin) = the dev tree of the 21 files plus an emptied prod
      todo; its tree equals dev except this file.

### How to finish the promotion (in this order)

1. Owner (or the session with the owner's go): deploy the CASPER pod tools, see above.
2. `git fetch origin && git rebase origin/prod ready-promote-mh` (it should be a no-op or touch only CHANGELOG), check
   `git diff origin/dev ready-promote-mh --stat` shows only `doc/PROMOTION_TODO.md`, then `git push origin ready-promote-mh:prod`.
3. `pyaps-admin pod update` on both pods (it pauses the runner and resumes it); wait for a quiet moment.
4. Delete this entry from this file, and write the date and the two commit ids into the CHANGELOG.

## Notes

- On 2026-11-14 the tree difference between prod and dev (before this file) was exactly the rename:
  16 files (CHANGELOG, four L2 template FITS, docs, FERRE and its three readers, the new test).
  Everything else on dev (FERRE worker fix, dynamic ExGal widths, graphify files) is on prod under
  other commit hashes. Compare trees, not hashes: `git diff origin/prod origin/dev --stat`.
