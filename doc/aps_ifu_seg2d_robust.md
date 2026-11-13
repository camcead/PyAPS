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

## 7. Crowded fields: `seg2d_deblend_mode` (2026-11-13)

### 7.1 The problem

OB 14699 (night 20260315, globular cluster M3, white-light image 185 x 171 px at 0.5 arcsec/px): `sep.extract` with `deblend_nthresh = 4` gave 44 detections. The 28 smallest are single stars (a = 1 arcsec); the 15 largest are **blends** of 2 to 13 stars each (the largest: a chain of 8 stars, a = 12 arcsec, elongation 4.8). Each detection becomes one IFU patch, and the Galactic stage bins every patch (2 x 2 arcsec bins) and runs FERRE on each bin, so the blends cost most of the run time. A finer deblending of the whole image (64 levels, `deblend_cont` 1e-4) splits the blends, but it would also tear a galaxy apart (noise ripples and star-forming knots become "objects"), and in the collapsed image a compact galaxy and a pair of stars look alike (fibre footprint 2.6 arcsec).

### 7.2 What the parameter does

| `seg2d_deblend_mode` | meaning |
|---|---|
| `fixed` | exactly the code before this change: `sep.extract` with `seg2d_deblend_nthresh`. Nothing else runs. |
| `auto` (default) | the same extraction first (its detections are the *parents*); then, **only if the field passes the crowding gate**, a fine pass finds the children of each parent and a parent is replaced by its children **only if every rule of section 7.3 holds**. Everything else is kept exactly as sep returned it. When no parent is split the arrays are returned untouched, so the result is bit for bit the `fixed` result. |

`seg2d_deblend_nthresh` keeps its meaning (the coarse pass; `None` = 4). Where to set it: `--seg2d_deblend_mode fixed|auto` of `aps_ifu_prepare.py`; `"seg2d_deblend_mode"` in the `IFU_params` JSON (all shipped IFU configs say `"auto"`) or `seg2d_deblend_mode:` in `script_params.yaml`; the runner passes it on (JSON > YAML > default `auto`). **To turn it off: `"seg2d_deblend_mode": "fixed"`** (per survey in the JSON, or `--seg2d_deblend_mode fixed`). An unknown value is an error, not a silent default.

### 7.3 The rules (all must hold; constants are in `aps_ifu_deblend.AUTO_PARAMS`)

0. **PSF size**: the median core size (FWHM of the disc that has the area of the pixels above half of the peak, in the filtered image) of the detections with a peak of at least 10 sigma, kept within 2.0 to 3.2 arcsec (a LIFU fibre is 2.6 arcsec across; 3.1 arcsec on OB 14699). The core size, not the isophotal size, because the latter grows with the brightness of a star.
1. **Crowding gate**: at least **10** compact (core size <= 1.5 x PSF), round (b/a >= 0.5), measurable (peak >= 5 sigma) detections. A sparse field or an isolated galaxy has few; the gate is the first thing computed, and below it nothing else runs (36 on OB 14699).
2. **Real children**: a fine-pass child counts only if it holds >= 7 % of the parent's flux, has a topographic prominence (its peak minus the saddle to a brighter neighbour) of >= 10 % of its own peak **and** >= 6 sigma of the filtered image, and its peak is >= 1.5 x PSF (4.6 arcsec) away from every brighter real child (closer peaks are one source: fibre-to-fibre noise makes one bright star look like several peaks, and a pair that close cannot be told from one star here). Noise ripples of a smooth object fail this. At least two are needed.
3. **The parent is not extended**: at most 35 % of its flux may lie outside the real children (the rest being fragments), the fine pass must not break it into 25 or more pieces, and it must not touch (within 2 pixels) an object that is extended in that sense (more than 50 % of the flux outside the real children, or 25 or more pieces).
4. **The children are point-like**: each real child, with the fragments around it given to it (by the highest saddle), has a core size <= 2 x PSF and covers at least one PSF disc, and none holds more than 70 % of the parent's flux (one dominant child = one object).
5. **The split pays**: it saves at least 10 Galactic-stage bins (estimated as pi/4 (A/2)(B/2)/4 with A = a x radii_factor x pixel scale, B likewise) **and** 15 % of the parent's bins.

Children are measured from their own pixels with the formulas of sep (checked against sep: identical for an undeblended object), so each child gets its own `a`, `b`, `theta`, `x`, `y`, `flux`, `npix`, `peak`, ... and its own `A_world`/`B_world` in the patch table. The segmentation map is relabelled consistently (label = row + 1); the pixels of the other detections are untouched.

### 7.3a Changing a threshold: `seg2d_deblend_params`

Every constant of section 7.3 can be set from the configuration, without a code change: `"seg2d_deblend_params": "gate_min_compact=8,min_separation=1.2"` in the `IFU_params` JSON, `seg2d_deblend_params:` in `script_params.yaml`, or `--seg2d_deblend_params` of `aps_ifu_prepare.py` (JSON > YAML > default `none` = all defaults). The value is `name=value` pairs separated by commas (no spaces); the names are the keys of `aps_ifu_deblend.AUTO_PARAMS` (`gate_min_compact`, `gate_compact_size`, `child_min_flux_frac`, `child_min_prominence`, `child_min_prom_sigma`, `min_separation`, `max_diffuse`, `extended_diffuse`, `extended_n_fine`, `touch_px`, `child_max_size`, `dominant_flux_frac`, `min_bins_saved`, `min_saving`, `psf_*`, `fine_nthresh`, `fine_cont`, ...). A name that does not exist or a value that is not a number stops IFU_PR with the list of valid names (a typo never leaves the old threshold in force unnoticed). The overrides are written to the segmentation summary (`deblend_auto.params_overridden`) so a product always says which thresholds made it. Loosening the rules splits more blends and also risks tearing a galaxy: section 7.4 gives the measured cost of the loosest setting that was tried (3 more fields changed, all with galaxy groups).

### 7.4 Galaxies, knots and why the rules look as they do

* A **large galaxy** has smooth light: the fine pass breaks it into dozens of noise fragments (28 to 115 pieces in the real fields that were checked) and most of its flux is in none of them. Rule 3 keeps it whole and **also protects whatever touches it**: the knots and cores of a galaxy are separate detections of the coarse pass that touch the galaxy's envelope.
* **Star-forming knots** inside a galaxy are unresolved at 2.6 arcsec, so a compact galaxy with two knots looks like two stars. Nothing in the collapsed image tells them apart for certain. The defence is the combination of rules 1, 2 and 5: crowded field, peaks at least 4.6 arcsec apart and at least 6 sigma above their saddle, no dominant child, and a clear saving. In the real data this combination left every galaxy field unchanged (section 7.6); relaxing it (peaks >= 3 arcsec apart, 3 sigma, 5 bins) changes 3 more fields (OB 16231, 26125, 16940, all with galaxy groups or extended galaxies in them) and saves only 9 more bins on OB 14699.
* A blend of stars next to a galaxy, in a crowded field, is split if it passes the rules and the galaxy stays one detection (test `test_crowded_field_with_a_big_galaxy_splits_the_stars_and_keeps_the_galaxy`).
* If you work on a survey of **galaxies only** and want the old behaviour guaranteed, set `fixed` in that survey's `IFU_params`.

### 7.5 Cap, summary and log

* `seg2d_max_targets` (default 150) is applied **after** the split, brightest first as before. A split is never allowed to push a field over the cap: parents are split brightest first, and a split that would exceed the cap is skipped (`n_skipped_by_cap`), so a split never costs a faint detection.
* IFU_PR prints `seg2d deblend auto: crowded field; 44 parents, 2 split into 7 children (36 compact detections, gate 10, PSF 3.1 arcsec)` and `SEG2D_SUMMARY {...}`; `<head>_seg2d_summary.json` has new keys: `deblend_mode`, `n_parents` (detections of the coarse pass), `n_parents_split`, `n_children`, `n_detected` (after the split, before the cap), and in `auto` mode `deblend_auto` with `gate_passed`, `n_compact`, `gate_min_compact`, `psf_size_arcsec`, `n_parents_candidates`, `n_skipped_by_cap`, `n_rejected` (why parents with two or more fine children were not split: `too_few_real`, `extended`, `diffuse`, `next_to_extended`, `child_too_small`, `not_pointlike`, `dominant`, `no_saving`; `one_child` = parents the fine pass does not split at all) and the list `splits` (position, size, number of children). `fixed` writes `deblend_auto: null`.
* The step is optional: any exception in it is caught, a WARNING is printed, and the normal extraction is used (the summary then has `deblend_auto.error`).

### 7.6 Result on the real data

OB 14699, the pipeline's own parameters (`ext_thresh` 1.0, `minarea` 8, `deblend_nthresh` 4), checked with the real `ifu_seg2d.object_extract`:

| | detections | largest a | estimated bins |
|---|---|---|---|
| `fixed` | 44 | 12.0 arcsec (chain of 8 stars) | 494 |
| `auto` | 49 (2 parents split into 7 children) | 5.0 arcsec | 412 |

The 12-arcsec chain became 3 pieces and a 5.4-arcsec blend 4. The other blends were left whole by the rules (a dominant bright star with fainter neighbours, peaks closer than 4.6 arcsec, savings below the limits); the loosest setting (peaks >= 3 arcsec apart, 3 sigma, 5 bins, no dominance rule) splits one more parent and saves 15 more bins (397), at the price of the 3 other fields above. All 108 different finished white-light images on the Imperial pod (all the L2 nights there, the thresholds from the OB's own `seg2d_summary.json` or, if absent, from `estimate_seg2d_params`) were run in both modes: **107 gave exactly the same detections and segmentation map, OB 14699 is the only one that changed** (44 of the 108 pass the crowding gate; among them the fields with the largest galaxies, e.g. OB 12878 with a 56-arcsec-long tail, OB 16161, 24316, 26735).

### 7.7 Tests

`tests/test_seg2d_deblend.py` (synthetic images: a chain of five stars in a crowded field is split into the five with the right positions; a sparse field, an isolated galaxy with knots, and a galaxy with knots in a crowded field are untouched, the very same arrays; a crowded field with a galaxy splits the stars and keeps the galaxy; the cap; `fixed` pinned against a copy of the code before the change; the measured columns against sep; summary keys; wiring) and, where `PYAPS_TEST_DATA` has OB 14699, the regression above.
