# IFU seg2d: robust sky, calibrated threshold, patch reporting (2026-11-06)

What changed in IFU_PR (`aps_ifu_prepare.py`, `aps_ifu_seg3d.py`) and in IFU_ExGal / IFU_Gal after **OB 26133** (night 20261104, LIFU low-res, one 900 s exposure) produced **226 patch products from one super-bright central source** and took 8.4 h, while every job said COMPLETED.

## 1. What went wrong

IFU_PR collapses the cube to a white-light image, segments it with SEP and writes one patch (`P####`) per detection. With `seg2d_ext_thresh` and `seg2d_minarea` set to `None` (the shipped default) it estimates both from the image (`estimate_seg2d_params`). On OB 26133 that estimate chose `minarea = 3` and `ext_thresh = 1.2` and found 148 "sources" (209 patch rows, 349 after the class split, 226 products). About 135 of the 148 were noise. Three causes, all measured on the real files:

1. **The white-light image is a mosaic of single-fibre footprints with gaps** (62% blank, ~544 footprints of ~22 px at 0.5"/px), not a smooth image. `sep.Background` cannot build a mesh from such sparse tiles: on this image even the plain call returned one constant for all pixels.
2. **The sky was biased by the two very bright sources** (4472 and 2555 sigma). `sep.Background` returned back = 3111 and rms = 7428; the sigma-clipped sky of the same pixels is median 8527, std 9530. The threshold `back + 1.2*rms` was only **0.37 sigma above the true sky**: half of the fibres exceeded it by chance.
3. **The sign-flip purity check was fooled by the same background.** Its negative detections were counted on the biased image (23 negative against 167 positive at 1.0, 11 against 148 at 1.2: purity 0.93), so the lowest threshold of the grid was recommended. With the sky re-centred the same test gives 63 against 64 at 1.0 and 52 against 35 at 1.2 (purity 0.33); no threshold below about 5 reaches 0.9. In addition the "PSF" measured from the most compact round detections was a single fibre footprint (FWHM 1.8 px), so `minarea` was 3 px: one fibre.

The consequences downstream: Redrock fitted noise (junk classes, `class_ntop = 3` split ambiguous rows), IFU_ExGal and IFU_Gal logged "No valid spaxels remaining after filters!" for 222 of the 226 patches and still ended COMPLETED.

Same estimator on the other LIFU OBs of the night (before the fix): 24311 (138 patches) has a sane background and zero negative detections above 1.8 sigma, so its sources are real; 25931 (68) is mostly real; 24266 (1 patch) correctly has nothing. The fix must, and does, leave those alone (section 5).

## 2. What changed

| | Change | Where |
|---|---|---|
| F1 | One **robust sky** for all seg2d steps: iterative 3-sigma clipping of the finite pixels gives the level and the noise; for a sparse fibre mosaic (< 90% finite pixels) the sky is that constant, for a well populated image (e.g. PS1) a `sep.Background` mesh built from the surviving pixels, used only if it agrees with the clipped level. The PSF estimate, the purity calibration, the final extraction (`object_extract`) and `sep_catalog` (seg3d) all use it, so the calibrated threshold is applied to the image it was calibrated on. | `aps_ifu_seg3d.robust_background`, `prepare_white`; `aps_ifu_prepare.object_extract` |
| F2 | **Guards on the calibration**: grid extended to 5, 6, 8 sigma; a threshold is accepted only if purity >= 0.9 **and** at least 3 more positive than negative detections; when none is accepted the calibration is reported as FAILED (`calibration_ok = false`, a WARNING, the largest threshold is used) instead of falling back silently. | `calibrate_sep_purity_ex`, `SEG2D_THRESH_GRID`, `SEG2D_MIN_EXCESS` |
| F3 | **Fibre-footprint floor** for `minarea`: on a fibre mosaic (>= 100 footprints, median <= 200 px) `minarea >= 0.5 x footprint` (11 px for the 22 px LIFU footprint), because the "PSF" measured on such an image is one footprint. Other images keep the PSF-derived value. | `fibre_footprint_px`, `estimate_seg2d_params`, `SEG2D_FIBRE_MIN_FRAC` |
| F4 | **Cap and summary**: new option `--seg2d_max_targets` (default 150, 0 = off): above it the brightest detections (by flux) are kept and a WARNING says so (the largest genuine count seen is 121, OB 24311). IFU_PR prints one `SEG2D_SUMMARY {...}` line and writes `<headname>_seg2d_summary.json` (thresholds, sky, plain-sep sky, calibration_ok, detections kept/dropped). This is a cap with a warning, not a stop of the OB: a hard stop before IFU_ExGal/IFU_Gal would turn an unusual field into a failed OB. | `ifu_seg2d.object_extract` |
| F5 | **Patch reporting** in IFU_ExGal and IFU_Gal: every run prints `IFU_PATCH_SUMMARY {"attempted":..,"written":..,"no_valid_spaxels":..,"failed":..}`, writes `<headname>_<stage>_patches.json`, warns when less than half of the attempted patches produced a product, and **fails the job** when at least 10 patches were attempted and none produced a product. | `aps_ifu_patch_report.py`, `aps_ifu_ExGal.py`, `aps_ifu_Gal.py` |
| F6 | **Class-split rows: not changed.** `update_patch_table` splitting a source into one GALAXY/QSO row and one STAR/WD row (`class_ntop = 3`) is by design (PyAPS RASTI paper, Table 5): the same object is analysed under both hypotheses. After F1 to F4 it multiplies at most a handful of real sources. A rule such as "require a minimum Redrock significance before splitting" changes science output and needs the survey scientists' decision. | - |

`aps_ifu_v0.py` (the legacy single-file IFU script) still has its own copy of the old code; it is not used by the runner.

## 3. Result on the real data

Re-running the pipeline's own `ifu_seg2d` on the real 26133 stackcubes (scratch directory, nothing written into the OB):

```
robust sky: level=8527 rms=9530 (plain sep.Background would give back=3111 rms=7428; 6.2% of the pixels rejected)
fibre-footprint image (footprint 22 px): minarea floor 0.5 x footprint -> minarea=11
ext_thresh=5.0: n_pos=8 n_neg=0 purity=1.000 <- recommended
SEG2D_SUMMARY {"calibration_ok": true, "ext_thresh": 5.0, "minarea": 11, "n_detected": 8, "n_kept": 8, ...}
```
8 detections (peaks 4472, 2555, 122, 66, 41, 28, 14, 9 sigma: the central target, a second bright source, six fainter ones) instead of 148. The same code on the night's other OBs:

| OB | before (detections / patches) | after | note |
|---|---|---|---|
| 26133 | 148 / 226 | 8 | the problem OB |
| 24311 | 123 / 138 | 114 | real sources, unchanged in kind |
| 25931 | 47 / 68 | 39 | mostly real |
| 24316 | 18 | 23 | |
| 24282 | 3 | 1 (calibration_ok = false: only sources above 8 sigma are kept) | sky field |
| 24266 | 0 / 1 | 0 (calibration_ok = false) | unchanged |

## 4. How to read the new output

* In the IFU_PR stdout: `estimate_seg2d_params` block (PSF, fibre footprint, **robust sky vs the plain sep sky**, the purity table, `<- recommended` or a WARNING when the calibration failed), the `seg2d sky:` line and `SEG2D_SUMMARY {...}`.
* `<OB>/<headname>_seg2d_summary.json`: the same as JSON. Red flags: `calibration_ok: false` with many detections, `n_dropped_by_cap > 0`, `plain_sep_back` far from `sky_level` (the old bias).
* IFU_ExGal / IFU_Gal stdout and `<headname>_ExGal_patches.json`, `<headname>_Gal_patches.json`: attempted / written / no_valid_spaxels / failed.

## 5. Settings and compatibility

* `--seg2d_ext_thresh` / `--seg2d_minarea` explicit numbers still win; `None` (shipped default) = auto as above. Explicit thresholds are in units of the **robust noise** (clipped std), no longer of `sep.Background.globalrms`: an old explicit value now corresponds to a lower absolute threshold on images where the plain rms was too small, and the same on images where it was fine.
* `--seg2d_max_targets` (default 150) is a command-line option of `aps_ifu_prepare.py`; the runner does not pass it, so the default applies in production.
* Existing runs are not changed; only new IFU_PR runs use the new code. To redo an OB: retract and reprocess it with the usual tools.
* Roll back: the previous behaviour is the commit before this change (`git revert`); nothing in the data model changed.

## 6. Tests

`tests/test_seg2d_robust.py` (synthetic fibre mosaic with two very bright fibres: robust sky unbiased, calibration ok, few detections, footprint floor only for a mosaic, a pure-noise field fails the calibration loudly and yields at most one detection, the patch tally and its failure rule, the pipeline wiring); `tests/test_aps_common_args.py` (the new flag). The synthetic image cannot reproduce sep's degenerate background, so the old-versus-new comparison is documented from the real data (section 3), not unit-tested.
