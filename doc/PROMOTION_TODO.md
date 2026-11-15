# Prod: work that is deliberately NOT here yet

Last updated 2026-11-15. This is the prod copy of the list. The checklist with the steps lives on
branch `dev` in the file of the same name; tick items there, and remove the entry here when it is promoted.

Nothing is held back at the moment. The FERRE metallicity columns `FEH` / `FEH_ERR` were renamed `MH` / `MH_ERR`
(dev commit `c1cac86`) and promoted together with the FESWI compatibility (`66ee04f`: `aps_feswi.feswi_param_file`,
`tests/test_feswi_mh_compat.py`, `doc/feswi_accept_mh.patch`). Old L2 files keep `FEH`; every reader in PyAPS accepts both.
