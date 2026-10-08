# PyAPS 2.0 - final publication snapshot (tag `v2.0-rasti`)

The version label is **2.0**, the version cited in the paper. Two archives carry that label:

| | Tag | Commit | Date | Zenodo |
|---|---|---|---|---|
| Earlier public release | `v2.0` (unchanged) | 1684edfe3e532a7e1cc16059285598e38d6e2195 | 2026-09-29 | 10.5281/zenodo.23042511 |
| **Final publication snapshot** | `v2.0-rasti` | 20cd0b6fb131119379faf5c32edd4ccb2d12da65 | 2026-10-09 | **10.5281/zenodo.23250780** (concept DOI 10.5281/zenodo.23042510) |

The snapshot adds, on top of the earlier release: the corrected Redrock per-arm chi-square implementation with the `--rr_solver perarm|joint` switch
(`py/PyAPS/aps_rr.py`, `tests/test_rr_perarm_patch.py`); the one aperture convention (A_world/B_world are full axis lengths) and the explicit
`gal_aperture_factor` option (`tests/test_aperture_convention.py`); the NumPy 2 and cache fixes of the L2 jobs; the environment checker
(`tools/check_env.py`); and the configuration templates.

## Python and scientific engines

* Supported: Python >= 3.12 (`requires-python`; classifiers 3.12 and 3.13).
* Tested with this snapshot (frozen set `requirements-lock-20261006.txt`, `pip install --no-deps -e .`):
  Python 3.12.12 (macOS arm64) and 3.14.0 for the test suite; the production pods run Python 3.12.3.
* Engines of the production pods and of the test environment: Redrock 0.22.2.post1195 (desihub/redrock commit 1119657), RVSpecFit 0.9.4.dev14+g24b00e2c7,
  pPXF 9.5.0, FERRE 5.1.3 (pods only; the test suite does not run FERRE), NumPy 2.5.3, SciPy 1.18.1, Astropy 8.0.1, numba 0.68.0, desiutil 4.0.1.
* The paper's historical benchmarks and stellar comparisons were produced with the stack named in the paper (Redrock 0.22.x, RVSpecFit 0.9.3,
  FERRE 5.1.1, pPXF 9.4.x, NumPy 2.3.4, SciPy 1.16.2, Astropy 7.1.1). They were NOT rerun with this snapshot, and this document does not claim they were.
  Installing the pre-2026-10 stack needs the older dependency pins of the `v2.0` tag (Python >= 3.9, NumPy >= 1.21).

## Tests of the snapshot (Python 3.12.12, frozen lock, editable install)

Full suite: 219 passed, 1 skipped (PyQt5 not installed), 115 expected failures (xfail), 0 failed.
Redrock per-arm, aperture, L2-job and common-argument tests: 54 passed, 14 skipped (only without the editable install), 8 xfailed.

Note: the Zenodo archive `PyAPS-v2.0-rasti.zip` was created from commit 20cd0b6 (the tag `v2.0-rasti`), i.e. BEFORE this documentation commit. The `CITATION.cff` inside that archive still carries the older DOI and the spelling "Advance"; the files on `main` (this commit and later) are the corrected ones, and 10.5281/zenodo.23250780 is the DOI to cite for the snapshot.
