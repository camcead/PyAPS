# PyAPS Redrock Strategy and Archetype Fallback System

## Overview

PyAPS uses [Redrock](https://github.com/desihub/redrock) for target classification
and redshift estimation. Redrock was originally developed for DESI and works
exceptionally well for that survey. For WEAVE, several adaptations are required
due to differences in sky subtraction quality, detector geometry, and the target
populations observed. This document describes the full strategy, the known failure
modes, and the archetype fallback system developed to recover correct redshifts.

---

## Redrock Two-Stage Fitting

Redrock fits spectra in two stages:

### Stage 1 — Coarse PCA scan

Redrock evaluates chi2 on a fixed redshift grid using PCA (Principal Component
Analysis) templates. These templates are flexible eigenvectors that span a wide
range of spectral shapes — they are not physically motivated but can reproduce
unusual SEDs that physical templates cannot.

```
Templates used: GALAXY, QSO:::LOZ, QSO:::HIZ,
                STAR:::A, STAR:::B, STAR:::CV,
                STAR:::F, STAR:::G, STAR:::K,
                STAR:::M, STAR:::WD

Grid spacing: ~dz=0.002 (varies by template)
Output: best z per template, ranked by chi2
```

The coarse scan is the most reliable step for **class identification** because
the PCA templates cover a broader range of SEDs than physical archetypes.

### Stage 2 — Archetype fine fitting (fitz)

Redrock refines the top `nminima` redshift candidates using physically motivated
archetype templates (LRG, BGS, ELG for galaxies; M/K/F star spectra; QSO composites).
A Legendre polynomial continuum correction is applied per camera arm.

```
Archetypes used: GALAXY (LRG, BGS, ELG variants)
                 QSO (LOZ, HIZ, BAL variants)
                 STAR (per spectral type)

Per-camera Legendre correction: deg_legendre=2
n_nearest archetypes blended:   n_nearest=3
```

The archetype stage gives more precise redshifts and physically meaningful
subtypes, but it can fail for unusual SEDs where the archetype set is incomplete.

---

## PyAPS Production Parameters

```python
# zfind call parameters
nminima          = 5        # explore top 5 chi2 minima
n_nearest        = 3        # blend 3 archetypes per fit
deg_legendre     = 2        # quadratic continuum correction per arm
per_camera       = True     # independent Legendre per arm
prior_sigma      = 0.1      # Legendre regularisation
zminfit_npoints  = 25       # points for parabola fit
ncamera          = dynamic  # set per observation mode

# Spectral processing
tellurics        = False    # do NOT mask telluric bands
                            # CRITICAL for extragalactic targets:
                            # telluric mask removes Hα for z=0.07-0.16
vacuum           = True     # output wavelengths in vacuum
safe_mask_gaps   = True     # mask CCD gaps from lookup table
skusub_residual_mask = True # smart handling of sky residuals
# Wavelength ranges (LR mode)
wlranges = [[3800.0, 5925.0],   # blue arm
            [5925.0, 9280.0]]   # red arm — cut at 9280Å
                                # removes OH forest (8200-9380Å) lines through smart filterring
                                # which causes M/K star contamination
```

### Why `tellurics=False`

With `tellurics=True`, the O2 A-band mask (7600-7700Å) and its wings
(6800-7600Å) remove a large fraction of the red arm. For A2142 cluster
galaxies at z≈0.073, Hα falls at 7040Å — directly in the telluric mask.
Masking this region makes the galaxy spectrum featureless in the critical
wavelength range, causing M/K star templates to win on continuum shape alone.

### Why keep the 8200-–9380Å region

The 8200–9380Å region contains a dense forest of OH sky emission lines and is therefore particularly sensitive to imperfections in sky subtraction. In the current WEAVE data products, strong sky residuals are not always fully reflected in the IVAR arrays, causing Redrock to treat some contaminated pixels as valid spectral information.

Rather than removing or masking the entire wavelength range, APS now applies a dedicated sky-residual filtering stage that attempts to identify and suppress individual residual features directly through modifications of the IVAR array. This approach preserves as much astrophysical information as possible while reducing the impact of residual sky contamination on the fit.

The primary motivation is to retain valuable spectral features that fall within this wavelength range, most notably Hα for galaxies at z \approx 0.25-0.37. Completely masking the region would remove one of the strongest redshift diagnostics for these objects, whereas targeted residual filtering allows the majority of the wavelength coverage to remain available for classification and redshift determination.

While this significantly improves robustness, the effectiveness of the approach ultimately depends on the quality of the upstream sky subtraction and variance estimation. Residual contamination in this region remains one of the main limitations of automatic redshift estimation for current WEAVE data products.

### IVAR normalisation

Two-arm data uses `mode='balanced'` normalisation, which equalises the
chi2 contribution of each arm (50/50). This prevents the higher-IVAR
red arm from dominating the fit and ensures galaxy features in the blue
arm (Hβ, [OIII] for z≈0.07-0.10) receive equal weight.

---

## Known Failure Modes

### Failure Mode 1 — Class mismatch (coarse correct, archetype wrong)

The coarse PCA scan correctly identifies the target class and redshift,
but the archetype fine fitting converges to the wrong class.

**Root cause:** The galaxy has an unusual SED (dust-reddened, AGN contamination,
TP-AGB dominated) that the LRG/BGS/ELG archetype set cannot reproduce. The M/K
star archetype fits the rising red continuum better than the galaxy archetype,
winning on chi2 even though the class is wrong.

**Example:**
```
Coarse: GALAXY z=0.073  chi2=41,751  ← correct
Arch:   STAR   z=0.002  chi2=81,300  ← wrong (ratio=1.95)
```

**Detection:** `arch_class != coarse_class` where STAR is involved — *unless*
targeting corroboration applies (see below).

**Targeting corroboration (added 27 Aug 2026).** The rule above was
originally blanket and direction-blind: any STAR-involving mismatch
reverted to the coarse scan, full stop. This protects the case in the
example above (a genuine galaxy spuriously flipped to STAR), but it is
symmetric where the real failure isn't: it also silently discards cases
where the *archetype* is right and the *coarse scan* is wrong. Confirmed
on a real GA-LRDISC OB (20260122, `stack_3134457__stack_3134456`,
b≈−1°, i.e. significantly reddened): genuine `TARGCLASS='STAR'` targets
lose the PCA coarse scan to GALAXY outright, independent of archetypes
— the 10-coefficient GALAXY PCA basis's extra shape freedom out-fits the
5-coefficient STAR basis's fixed, unreddened continuum shape at this
Galactic latitude — and the archetype correctly found STAR for several
of these, but the blanket rule was discarding it.

Fix: `_targeting_class_hint()` (TARGCLASS > TARGPROG > TARGSRVY, same
priority as `_ensure_srvy_class_represented`) is checked as a
corroborating signal. When the archetype's class agrees with targeting
*and* the coarse scan's does not, the class-mismatch trigger is
suppressed and the ratio trigger's threshold is widened (not removed) by
`targeting_ratio_relax` (default 3×):

```python
targeting_override = (
    targeting_hint is not None and
    arch_class == targeting_hint and
    coarse_class != targeting_hint)

if arch_class == coarse_class:
    class_mismatch = False
elif pair in {('GALAXY','QSO'), ('QSO','GALAXY')}:
    class_mismatch = |arch_z - coarse_z| > 0.01
elif targeting_override:
    class_mismatch = False          # archetype corroborated by targeting
else:
    class_mismatch = True

effective_ratio_threshold = (
    1.5 * targeting_ratio_relax if targeting_override else 1.5)
ratio_trigger = degradation > effective_ratio_threshold
```

Verified end-to-end (real pipeline, archetypes on) on 8 real GA-LRDISC
targets: 4/6 previously-misclassified targets (APS_ID 1, 3, 7, 581) are
now correctly STAR; the two genuine catastrophic-class regression cases
from the original WC motivating dataset (`catastrophic_redshifts_apsmod.txt`
in `PyAPS_local/<investigation_dir>`, APS_ID 791/938) are **unaffected** — no
regression, since `TARGCLASS='GALAXY'` there and the archetype found
STAR, so `targeting_override` never applies. Those two remain unfixed by
any current mechanism (see Unfixable Failure below) — coarse PCA and
archetype independently agree on the wrong class there, so there is no
disagreement for this function to arbitrate either way.

---

### Failure Mode 2 — Ratio degradation (same class, wrong fit quality)

The archetype found the right class but its chi2 is significantly worse
than the coarse PCA chi2, indicating the archetype fit is poor.

**Root cause:** Unusual SED within the correct class — the available archetypes
do not span the full diversity of the observed population.

**Example:**
```
Coarse: STAR z=0.000  chi2=59,023  ← correct
Arch:   STAR z=0.000  chi2=88,940  ← right class, degraded fit (ratio=1.51)
```

**Detection:** `arch_chi2 / coarse_chi2 > 1.5`

---

### Failure Mode 3 — Redshift alias (same class, wrong z)

The archetype correctly identifies the class but converges to a different
chi2 minimum than the coarse scan — a spurious high-z alias.

**Root cause:** Multiple chi2 minima exist within the same template class.
The coarse PCA grid finds the correct minimum, but the archetype fine fitting
converges to a different (wrong) minimum with lower chi2.

**Example:**
```
Coarse: GALAXY z=0.049  chi2=31,902  ← correct
Arch:   GALAXY z=1.644  chi2=33,749  ← wrong alias (dv=455,000 km/s)
```

**Detection:** `arch_class == coarse_class` but `dv(arch, coarse) > 3000 km/s`
— *unless* the archetype's fit is at least as good as the coarse scan's own
best (see below).

**Archetype z improvement (added 27 Aug 2026).** The unmodified trigger
always reverted `z_reference` to the coarse scan's own z on a large same-
class jump, *even when the archetype's chi2 was decisively better* — i.e.
even when the archetype had genuinely found a better redshift, not
aliased to a spurious one. Confirmed on real WC data (same OB as the
Unfixable Failure example below, APS_ID 348 and 618): the archetype
found z matching literature almost exactly (0.0432 vs Z_LIT=0.04323;
0.0425 vs Z_LIT=0.04247) with *lower* chi2 than the coarse scan's own
best, but the trigger discarded it and landed on a third, unrelated,
still-wrong z≈0.40 by searching near the coarse scan's own worse
reference z.

Fix: when `arch_chi2 <= coarse_chi2` at the same class
(`archetype_improved_same_class`), the z-shift trigger is suppressed and
`z_reference` is set to the archetype's own z instead of the coarse/PCA-
refined one:

```python
archetype_improved_same_class = (
    arch_class == coarse_class and arch_chi2 <= coarse_chi2)

if archetype_improved_same_class:
    z_reference = arch_z

z_shift_trigger = (
    (arch_class == coarse_class) and
    (dv_arch_coarse > 3000) and
    not archetype_improved_same_class)
```

Verified end-to-end: APS_ID 348's dV-to-literature drops from 102,909 to
3.0 km/s; 618 from 102,913 to 1.3 km/s. The two catastrophic-class cases
in the same batch (791, 938) are unaffected either way, as expected —
this only touches same-class redshift disagreements.

---

### Unfixable Failure — Coarse scan wrong

The coarse scan itself never finds a competitive candidate anywhere near
the true redshift, in any class. This occurs when the spectrum does not
contain enough discriminating features at the true redshift — typically
passive galaxies where the blue arm continuum accidentally matches a
high-z galaxy template better than the correct low-z solution. Unlike
Failure Modes 1-3 above, no post-hoc rank substitution can recover the
correct *redshift* here, because no rank anywhere in `zfitall` carries
it — this is the one failure mode that is about the coarse scan's own
chi2(z) landscape, not about which of its candidates the archetype/
fallback machinery ends up preferring.

```
True z=0.091, SNR≈6-9:
  Coarse: GALAXY z≈1.16-1.70  ← wrong z (pure continuum degeneracy)
  Arch:   STAR    z=0.000     ← wrong class too (as originally seen)
  No post-processing can recover the correct redshift.
```

This is a real pair of targets: WC OB 20250630/16287,
`stack_3095664__stack_3095663`, APS_ID 938 and 791
(`catastrophic_redshifts_apsmod.txt` in `PyAPS_local/<investigation_dir>`).
**Investigation status (28 Aug 2026, re-verified via a real `aps_rr.py`
run against current `develop`):** the *redshift* remains unrecoverable
for both — the coarse scan's own ranked candidate list (ZNUM 0-8, i.e.
`--nminima 15`, well above the production default of 3) contains no
GALAXY entry anywhere near the true z for either target, so this is not
a "too few minima kept" truncation; the true minimum is either never
sampled by the coarse redshift grid at all, or genuinely isn't a
competitive local minimum in the coarse scan's own chi2(z) curve at
these grid points. Raising `nminima` to 15 changed nothing (coarse
GALAXY answer identical, chi2 unchanged). Whether this is a
redshift-grid sampling-density issue specifically has not been confirmed
by dumping the actual chi2(z) curve. Extinction does not explain this
pair either: E(B-V)_SFD is only ≈0.05 at this field's higher Galactic
latitude, and forcing the fit to the true z=Z_LIT by hand already shows
GALAXY beating every STAR template there even on raw, undereddened
data — so the failure is specifically about the coarse scan never
finding that z, not about which template wins once you're at the right
redshift.

The *classification* half of this failure, however, is no longer present
for 938 on current `develop`: the class-mismatch trigger (3.7/3.8) now
correctly swaps rank 0 to GALAXY (z=1.1616, still the wrong z, but the
right class) because the coarse-scan's own GALAXY chi2 (18097) already
beat the archetype's STAR chi2 (22901) and a same-class candidate passed
the tier-1 ZWARN filter. 791 lands on QSO at rank 0 instead (coarse and
archetype both agree on QSO there), with GALAXY injected into rank 2 by
`_ensure_srvy_class_represented`. Neither is "fixed" in the sense of
getting the right redshift, but neither is a silent STAR misclassification
of a galaxy any more either — see Version History 3.11 for the specific
fix (BAD_MINFIT-relaxed fallback search) that resolved the *remaining*
STAR-classification failures in this same 43-target batch (APS_ID 948,
931), which had the classification bug 938 used to have.

---

## The Archetype Fallback System

The fallback system operates between the Redrock fit and the final zbest
table construction. It detects the three failure modes above and corrects
them by substituting a better rank from the zfitall table.

### Processing order in `gen_zbest_multiple`

```
Step 1: _apply_archetype_fallback
        → detects and corrects failure modes 1, 2, 3
        → sets APS-extended ZWARN bits (see below)

Step 2: ntop stacking
        → takes top ntop ranks per target
        → no ZWARN-aware promotion (disabled — causes wrong swaps)

Step 2b: _ensure_srvy_class_represented
         → ensures targeting class appears in at least one rank
         → injects best matching rank into last slot if absent

Step 3: Survey classification (SRVY_CLASS)
        → assigns SRVY_CLASS from TARGCLASS/TARGPROG/TARGSRVY
        → purely informational, does not affect Z or CLASS

Step 4: Finalise and write
```

### Trigger conditions

Any one of the three conditions is sufficient to trigger the fallback:

**Condition 1 — Class mismatch (pair-aware)**

```python
pair = (coarse_class, arch_class)

if arch_class == coarse_class:
    class_mismatch = False          # same class → no trigger

elif pair in {('GALAXY','QSO'), ('QSO','GALAXY')}:
    # Physically plausible (AGN host) — only trigger if z disagrees
    class_mismatch = |arch_z - coarse_z| > 0.01

else:
    # STAR involved → always a genuine mismatch
    class_mismatch = True
```

**Condition 2 — Ratio trigger**

```python
ratio_trigger = (arch_chi2 / coarse_chi2) > 1.5
```

**Condition 3 — Redshift shift within same class**

```python
dv = |arch_z - coarse_z| / (1 + coarse_z) × c
z_shift_trigger = (arch_class == coarse_class) and (dv > 3000 km/s)
```

### Fallback rank selection

When a trigger fires, the system searches `zfitall` for the best replacement,
in two tiers:

**Tier 1 (strict, as originally designed):**
1. Find all ranks matching `coarse_class`
2. Exclude ranks with Redrock ZWARN > 2 (Z_FITLIMIT, BAD_MINFIT etc.)
3. Among remaining ranks, select the one with z closest to `z_reference`
4. ZWARN=0 ranks preferred via a 0.5 dz penalty for ZWARN=1,2

**Tier 2 (relaxed, only runs if tier 1 finds nothing — added 28 Aug 2026):**
Retry allowing ranks whose *only* extra Redrock flag beyond bits 0-1 is
`BAD_MINFIT` (bit 10 — "bad parabola fit to the chi2 minimum"). This flag
means Redrock is unsure of the *formal error bar* on that z, not that the
z itself is wrong; the original strict-only search treated it the same as
`NODATA`/`UNPLUGGED`/`BAD_TARGET` and silently gave up, even when the
same-class candidate had unambiguously *better* chi2 than the archetype's
wrong-class answer that ends up kept by default. Real case this fixes: WC
OB 20250630/16287 (`stack_3095664__stack_3095663`), APS_ID 948
(TARGCLASS=GALAXY): coarse scan already correctly preferred GALAXY
(chi2=18097) over the archetype's STAR (chi2=18551), i.e. the trigger
condition and z_reference were already right — but every GALAXY row in
`zfitall` for this target carried `BAD_MINFIT`, so tier-1-only search
found nothing and kept the wrong STAR answer despite the better candidate
sitting right there. Verified end-to-end via a real `aps_rr.py` run on the
full 43-target GALAXY-hinted regression batch (see Version
History 3.11): fixes APS_ID 948 (STAR→GALAXY) and 931 (QSO→GALAXY), zero
change to the other 41 targets (35 already-correct GALAXY, 5 QSO,
including the harder 791/938 cases which are unaffected either way — see
"Unfixable Failure" below). Also re-ran the full GA-LRDISC population
batch (STAR-dominated, the survey `_apply_archetype_fallback` was
originally built for) to confirm tier 2 doesn't regress the opposite
direction — see Version History.

Tagged with `APS_MINFIT_RELAXED` (bit 26) whenever tier 2 is the one that
found the substitute, so it's auditable which fixes came from a
BAD_MINFIT-flagged candidate — see ZWARN bit table below.

```
z_reference selection:
  z_pca  if PCA parabola fit confirms coarse grid point
         (dv(coarse, z_pca) < 1000 km/s)
  coarse_best_z  otherwise
```

If no valid same-class rank exists in *either* tier → no swap, keep rank 0,
set `APS_FALLBACK_NO_VALID_RANK` bit.

### PCA fine fit reliability

A parabola is fitted around the coarse chi2 minimum to:

1. Obtain sub-grid z precision (`z_pca`)
2. Assess whether the minimum is well-defined

```
Reliability checks:
  a) Parabola opens upward (genuine minimum)
  b) Minimum is within the fit window
  c) chi2 at minimum agrees with coarse chi2 (<5% difference)
  d) Curvature is above noise floor
  e) dv(coarse, z_pca) < 1000 km/s (parabola confirms grid point)

If reliable → use z_pca as reference for fallback selection
If unreliable → use coarse_best_z as reference
```

A flat chi2 landscape (low curvature, `pca_reliable=False`) indicates
the target redshift is genuinely ambiguous — the spectrum does not contain
enough discriminating information. These targets are flagged but not
corrected.

---

## ZWARN Extended Bit Definitions

The `ZWARN` column in APS L2 products extends the Redrock native flags
with APS-specific bits. **This is not the original Redrock ZWARN.**

### Redrock native bits (0-9, read-only in APS)

| Bit | Value | Name | Description |
|-----|-------|------|-------------|
| 0 | 1 | SKY | Sky fiber |
| 1 | 2 | LITTLE_COVERAGE | Too little wavelength coverage |
| 2 | 4 | SMALL_DELTA_CHI2 | chi2 of best fit too close to 2nd best |
| 3 | 8 | NEGATIVE_MODEL | Synthetic spectrum is negative |
| 4 | 16 | MANY_OUTLIERS | Fraction of >5-sigma outliers too large (>0.05) |
| 5 | 32 | Z_FITLIMIT | chi2 minimum at edge of the redshift fit range |
| 6 | 64 | NEGATIVE_EMISSION | QSO line exhibits negative emission |
| 7 | 128 | UNPLUGGED | Fiber unplugged/broken, no spectrum |
| 8 | 256 | BAD_TARGET | Catastrophically bad targeting data |
| 9 | 512 | NODATA | No data for this fiber (ivar=0 everywhere) |
| 10 | 1024 | BAD_MINFIT | Bad parabola fit to the chi2 minimum |
| 11 | 2048 | POORDATA | Poor input data quality but fit anyway |

(source: `redrock.zwarning.ZWarningMask` — corrected 28 Aug 2026; the
values previously listed here didn't match the installed redrock version
and would have misidentified which bit is which. `_apply_archetype_fallback`'s
tier-2 BAD_MINFIT relaxation, `APS_MINFIT_RELAXED` bit 26 below, tests
`redrock_zwarn & ~1024`, i.e. bit 10.) Bits 12-15 are reserved for future
Redrock use and are never set by APS.

### APS extended bits (16-23)

| Bit | Value | Name | Description |
|-----|-------|------|-------------|
| 16 | 65536 | APS_FALLBACK_CLASS_MISMATCH | Class mismatch trigger fired |
| 17 | 131072 | APS_FALLBACK_RATIO | Ratio trigger fired |
| 18 | 262144 | APS_FALLBACK_Z_SHIFT | Z-shift trigger fired |
| 19 | 524288 | APS_FALLBACK_SWAP_APPLIED | Fallback swap was made |
| 20 | 1048576 | APS_FALLBACK_NO_VALID_RANK | Trigger fired, no valid rank |
| 21 | 2097152 | APS_PCA_UNRELIABLE | Coarse PCA minimum unreliable |
| 22 | 4194304 | APS_PCA_CONFIRMS_COARSE | Parabola confirmed coarse z |
| 23 | 8388608 | APS_Z_REFERENCE_IS_PCA | z_pca used as fallback reference |
| 24 | 16777216 | APS_TARGETING_OVERRIDE | Class-mismatch trigger suppressed / ratio threshold widened because targeting corroborated the archetype's class over the coarse scan's |
| 25 | 33554432 | APS_ARCHETYPE_Z_IMPROVED | Z-shift trigger suppressed, z_reference set to the archetype's own z, because its chi2 was at least as good as the coarse scan's own best |
| 26 | 67108864 | APS_MINFIT_RELAXED | Fallback rank substitution found no same-class candidate passing the strict ZWARN<=2 filter, but found one whose only extra flag is BAD_MINFIT (bit 10) — used as a second-tier candidate rather than keeping a worse-chi2 wrong-class answer |

### Extracting Redrock-only ZWARN

```python
# Mask to Redrock bits only (bits 0-9):
redrock_zwarn = zwarn & 0x3FF        # mask bits 0-9

# Check if APS fallback was applied:
fallback_applied = (zwarn & 524288) > 0

# Check if target needs inspection (trigger but no fix):
needs_inspection = (zwarn & 1048576) > 0

# Check if PCA landscape is unreliable:
pca_unreliable = (zwarn & 2097152) > 0

# Find all APS-modified targets:
aps_touched = (zwarn >> 16) > 0

# Clean targets (Redrock clean AND no APS intervention):
fully_clean = (zwarn == 0)
```

---

## Per-arm chi2 modification (monkey patch)

PyAPS replaces Redrock's `calc_zchi2_batch` (CPU, multi-arm only) at import.
Upstream solves ONE template-coefficient vector jointly for all arms at each
trial redshift. PyAPS solves one vector **per arm**:

```
upstream : c*   = argmin_c  sum_a (f_a - T_a c  )^T W_a (f_a - T_a c  )
PyAPS    : c_a* = argmin_ca       (f_a - T_a c_a)^T W_a (f_a - T_a c_a)    (per arm)
           reported chi2   = sum_a  residual_chi2(c_a*)
           returned coeffs = mean over arms of c_a*   (see below)
```

Because it replaces the function by name, the scope is **every** call: coarse
scan, fine-fit refinement (`fitz`), archetype fits (`get_best_archetype`,
nearest-neighbour model, per-camera Legendre solve). Single-spectrum input and
GPU mode use the upstream function unchanged, and `--rr_solver joint`
(or `APS_RR_SOLVER=joint`) selects the upstream solve for CPU multi-arm input
too, for A/B comparison. Default is `perarm`.

### What the modification does and does not do (verified, see `tests/test_rr_perarm_patch.py`)

- **More freedom.** `n_arm x n_basis` coefficients instead of `n_basis`.
  Unregularised, the minimum chi2 can only be lower than the joint fit. A lower
  chi2 is not by itself evidence of a better answer; a wrong redshift or
  template can also profit from relaxing the shared spectral shape.
- **Coefficients are invariant to a uniform IVAR rescale of an arm; the score is
  not.** Rescaling one arm's IVAR leaves its per-arm coefficients unchanged
  (no prior) but changes its contribution to the summed chi2 proportionally.
  Candidate rankings and DELTACHI2 therefore still depend on arm weighting
  (`--arms_ratio`, IVAR `balanced` normalisation). The patch does **not** make
  results "independent of IVAR scaling" and does not remove "joint coefficient
  bias" in any demonstrated sense; legitimately higher S/N should still weigh
  more in the likelihood.
- **Absorbs relative flux calibration / scale errors between arms.** A
  multiplicative mismatch between arms is absorbed by the independent
  coefficients (in synthetic tests a 15% offset removes most of the extra
  chi2). This is the effect that actually changes behaviour on real data, and
  it is not a calibration of the offset: the offset is simply not penalised.
- **The returned coefficients do not reproduce the reported chi2.** `zcoeff`
  is the equal-weight arithmetic mean of the per-arm vectors; applying it to
  every arm gives a chi2 >= the reported one. Redrock stores the mean as
  `COEFF`, and PCA-mode model spectra (`gen_zspec`) are drawn from it, so the
  plotted/stored PCA model is not the model that produced the reported chi2.
  Equal averaging is not an unbiased estimator of a shared coefficient vector.
- **Priors.** A supplied prior matrix (archetype fits: `prior_sigma`) is
  scaled in each arm by that arm's fraction of the total weight. This is not
  equivalent to regularising one shared vector, and the reported chi2 contains
  no penalty term either way (as upstream).
- **Arms with no template support** (all-zero template columns in that arm)
  contribute `sum(w f^2)` (model = 0) as in the upstream chi2; arms with zero
  total weight are omitted; solver failure or no valid arm returns `HUGE_CHI2`.
- **Per-camera (arm-specific) columns** (archetype mode: one Legendre block
  per arm). Each column is solved on the arms where it is non-zero and averaged
  only over those arms (fix of 2 Oct 2026; before it the Legendre coefficients
  stored in `COEFF` were diluted by `1/n_arm` and, with the prior, the other
  arm's columns were solved against a prior-only block). Reported chi2 and
  rankings were not affected by that bug; the stored Legendre coefficients were
  (PyAPS re-fits the Legendre terms when drawing archetype models).
- **Archetype amplitudes.** In archetype mode each arm also gets its own
  archetype amplitude(s), so arm scale errors are absorbed there too.

Diagnostics: set `aps_rr._RR_DIAG = []` (single process) and each multi-arm
call appends the reported chi2, the chi2 of the returned mean coefficients and
the coefficients.

The historical motivation text ("fixes joint PCA coefficient bias for WEAVE
blue+red arms", "fitz starting point") is not supported by a documented failing
case or by the call graph (fitz stores the mean coefficients as the result, it
does not use them as a seed). See `PyAPS_local/PyAPS_redrock_concern` audit
and the A/B results recorded in the Version History below.

---

## Galactic Extinction Correction (opt-in, added 27 Aug 2026)

Investigating the GA-LRDISC failures above led to checking whether
uncorrected Milky Way foreground dust extinction was a contributing
cause: GA-LRDISC targets sit at b≈−1° (E(B-V)_SFD≈1.2-1.5, substantial),
and the observed spectra show exactly the signature a rigid, unreddened
STAR template can't absorb but a flexible GALAXY template can (a real,
measured blue-flux deficit relative to model, ~30% of expected at
3800Å recovering to ~100% by 5000Å, present equally for both templates).

`aps_utils.py` now provides `get_sfd_ebv()`, `dust_transmission_curve()`,
and `deredden_spectrum()` (SFD98 dust map via `desiutil.dust` — already a
redrock dependency, no new package — + Fitzpatrick99 extinction law).
`APSOB` gained `extinction_corr` (default `False`), `extinction_ebv_scale`
(default `1.0`), and `extinction_mapdir` constructor parameters, applied
per-target (via each target's own TARGRA/TARGDEC) as a new step 7b in
`_process_single_arm_vectorized`, immediately after sensitivity
correction. Threaded through to `aps_rr.py`'s `--extinction_corr`/
`--extinction_ebv_scale`/`--extinction_mapdir` CLI flags. **Off by
default** — existing behaviour is unchanged unless explicitly requested.
Real SFD maps (desiutil only ships small test fixtures) are at
`PyAPS_data/DUST/SFD_dust_4096_{ngp,sgp}.fits` (not version-controlled,
same as CAL/CAT; see `deredden_spectrum` docstring for the source if they
need re-fetching).

### Why `extinction_ebv_scale` exists, and why 1.0 isn't automatically right for stars

SFD gives the *total* Galactic dust column to effectively infinite
distance — correct for a background source (GALAXY/QSO) but potentially
an over-correction for a foreground Milky Way star sitting at a finite
distance in front of only part of that column. A first, naive
hand-reconstructed test (not through the real pipeline — see caveat
below) suggested full-strength correction made GA-LRDISC's STAR-vs-GALAXY
gap *worse*, not better. Re-tested properly end-to-end through the real
pipeline (real Rcsr resolution-matrix convolution and `balanced` IVAR
normalisation both matter materially here and were missing from the
hand-reconstruction) on 7 GA-LRDISC targets with the targeting-override
fix from above already in place:

| Configuration | Correct (of 7) | Notes |
|---|---:|---|
| Baseline (no extinction corr.) | 6/7 | APS_ID 613 still wrong |
| `ebv_scale=0.25` | 5/7 | **Regressed** APS_ID 7 (was correct at baseline) |
| `ebv_scale=1.0` (full SFD) | 7/7 | Fixed 613 too, no regressions in this batch |

So, on this small real-pipeline test, full-strength correction was the
best option — reversing the initial (hand-reconstructed, methodologically
incomplete) caution against it, and a partial correction was actively
worse than doing nothing.

**GALAXY/QSO (WC) side, tested 27 Aug 2026** on the same 44-target
catastrophic-redshift batch used above (`PyAPS_local/<investigation_dir>`,
`ebv_scale=1.0`, WC field E(B-V)_SFD≈0.05-0.06, much smaller than
GA-LRDISC's ≈1.2-1.5): **5 genuine fixes** (APS_ID 685, 499, 467, 28, 959
— dV-to-literature dropped from hundreds of thousands of km/s to tens)
but **3 real regressions**, one of them a genuine class flip: APS_ID 997
(dV worsened 76,258→442,940 km/s), APS_ID 214 (correct QSO flipped to
wrong GALAXY, dV 4,179→107,584 km/s), APS_ID 864 (mild, 19→47 km/s). The
two hardest catastrophic cases (791, 938) were unaffected either way, as
expected given the low E(B-V) there. **Net positive on this batch (5
fixed vs 3 regressed) but not unconditionally safe even for GALAXY/QSO
targets** — same pattern as the STAR side (partial correction on GA-LRDISC
regressed a different target even while being directionally right).

**Population-scale GA-LRDISC result (27 Aug 2026, all 765 GA-LRDISC
targets in the OB, `ebv_scale=1.0`):**

| | N correct (of 765) | % |
|---|---:|---:|
| Baseline (no extinction corr.) | 513 | 67.1% |
| Full extinction correction | 660 | 86.3% |

165 targets fixed, 18 regressed (net +147, ~9:1 ratio) — a much larger
and more statistically robust net win than the small-sample tests above
suggested, and a real, substantial step toward the WEAVE requirement of
≥95% correctly classified (still short of it, but 67%→86% on this one
survey/field is a large, real gain). The GALAXY/QSO (WC) 44-target batch
above remains the only population check on that side (5 fixed/3
regressed, smaller N).

**Conclusion so far: `--extinction_corr` is real and gives a large net
positive at population scale on GA-LRDISC, and a smaller net positive on
the WC sample tested — but individual regressions are real on both
sides (STAR: 18/765 here; GALAXY/QSO: 3/44, including one class flip).**
It should stay opt-in until either (a) a way to identify *which* targets
it will help vs. regress is found, or (b) a similarly large population
check on the GALAXY/QSO side confirms the same favourable ratio seen for
GA-LRDISC. The full investigation trail is kept with the APS development notes.

### Reused for ExGal (PPXF/EMIPPXF/LS) — default-on, added 1 Sep 2026

The same `get_sfd_ebv`/`dust_transmission_curve` machinery above is also
used by the ExGal (extragalactic stellar-population/emission-line/
line-strength) pipeline, via `ExGalPrepare.resolve_ebmv_extinction` and
the `EBmV` key already present in every `configs/ExGal_configs/*.json`
(previously read but never actually applied to anything -- see below).
Unlike the REDROCK/classification use above, ExGal is **default-on**,
not opt-in, and always uses full-strength correction (`ebv_scale=1.0`)
with no partial-fraction caveat: `aps_mosExGal.py`'s `proc_mosExGaL`
hardcodes `working_classlist = ['GALAXY','QSO']`, so every ExGal target
is background/extragalactic by construction -- there is no finite-distance
foreground-star ambiguity to hedge against here, so SFD's "integrated to
infinity" assumption is simply correct, not a compromise.

**Why one fix reaches PPXF, EMIPPXF, and LS at once**: `proc_mosExGaL`
(and IFU's `ifu_ExGal_prepare`) build a single `APSOB(...)` object, copy
its flux/ivar into `exgal_targ['spec']`/`['error']`, and write that once
to an on-disk SPEC file that `MOSExGalPPXF`/`MOSExGalEMIPPXF`/`MOSExGalLS`
(and IFU equivalents) all read independently -- none of them touch raw
flux or do their own extinction handling. So dereddening once, at the
point `APSOB` builds the spectrum, reaches continuum fitting, emission-line
ratios, and line-strength indices identically, with no risk of the three
disagreeing.

**`EBmV` config semantics** (read by `resolve_ebmv_extinction`, argument >
config > default precedence, same pattern as `SPAXEL_WEIGHTED_LSF`):
- `"None"` (every bundled config's current value) -> automatic Galactic
  extinction correction, per-target SFD sky-position lookup (MOS) or one
  value for the whole patch evaluated at the cube's own WCS reference
  point CRVAL1/CRVAL2 (IFU -- SFD is effectively constant at IFU angular
  scales anyway, and this avoids needing a full APSOB build just to find
  a patch centre). This is the new default behaviour -- extinction
  correction is now genuinely applied for every ExGal run using the
  bundled configs, where previously `EBmV` was read into a FITS header
  and never used for anything (see below).
- a number -> that fixed E(B-V) \[mag\] applied to every target in the
  run, bypassing the automatic lookup (e.g. a manually-verified value
  for one field).

Verified via a real `proc_mosExGaL(...)` run through the full production
path (`APSOB` -> Prepare -> PPXF -> EMIPPXF -> LS) on real WC GALAXY
targets (APS_ID 118/618/348, `stack_3095664__stack_3095663`): auto mode
correctly resolved a per-target median E(B-V)_SFD=0.059 and completed
PPXF cleanly; a fixed `EBmV=0.5` config value correctly resolved to
exactly that value and completed PPXF+EMIPPXF+LS cleanly end to end.

**Two real, separate bugs found and fixed while implementing this**:
1. `EBmV` itself was already being read (`MOSExGalEMIPPXF.py`, one line)
   but *only* to stamp its raw, unresolved value into an output FITS
   header (`'De-redden the spectra for the Galactic extinction'`) --
   nothing anywhere ever consumed it to actually correct anything. Since
   the JSON literal is the string `"None"` (not JSON `null`), the header
   ended up literally storing the text `"None"`. Fixed: the header now
   also carries `EBMVAPPL`, the *actually applied* value (`'OFF'`, a
   fixed float, or `'AUTO (per-target SFD98)'`), set by `proc_mosExGaL`
   from what `resolve_ebmv_extinction` decided; the original `EBmV` key
   is kept (now stringified) for backward compatibility with older
   output but documented as "raw config value, see EBMVAPPL".
2. The `REDDENING` config key (`[0.1, 0.1]` in most bundled configs) is
   *also* dead -- it is never read from the config dict anywhere in the
   codebase. It corresponds to ppxf's own optional `reddening=` free-fit
   parameter (a *different* mechanism from `EBmV`: a physical dust law
   fit *during* the stellar continuum fit, capturing whatever residual
   attenuation -- host-galaxy internal dust included -- is needed to
   match the model, as opposed to `EBmV`'s pre-fit Galactic-foreground-only
   correction). **Formally retired (1 Sep 2026), not fixed**: the actual
   `ppxf(...)` call in `MOSExGalPPXF.py`/`IFUExGalPPXF.py`'s `run_ppxf`
   already uses `mdegree=<config MDEG>` (a multiplicative polynomial) to
   absorb residual continuum-shape mismatch, and ppxf's own documentation
   is explicit that combining `mdegree>0` with `reddening=` in the same
   fit is a degenerate double-parameterisation of the same effect.
   Enabling `REDDENING` alongside the existing `mdegree` would risk
   destabilising the stellar-kinematics fits, not cleanly add a missing
   correction -- a real methodological swap (drop to `mdegree=0` and use
   `reddening=` instead), not a blind wire-up, and the user's explicit
   choice was to keep `mdegree` as the continuum-shape mechanism and
   retire `REDDENING` rather than make that swap. To stop this being
   silently forgotten again the way `EBmV` was, `ExGalPrepare.
   note_reddening_unused(configs)` now prints an explicit one-line notice
   (`aps_mosExGal.py`/`aps_ifu_ExGal.py`, called once per run right after
   loading the config) whenever a run's config has a non-null `REDDENING`
   value -- verified via a real `proc_mosExGaL` run on the unmodified
   bundled `MOSLR11.json`. If the physical dust-law-fit swap is wanted
   later, it needs its own validation pass first, the same way the
   `EBmV` full-vs-partial correction was tested end to end above -- not
   assumed to be an improvement.

**Not investigated**: a distance-resolved (3D) dust map (e.g. Green et
al. 2019 "Bayestar19", Lallement et al. 2019 "STILISM") would let the
STAR/REDROCK `ebv_scale` stopgap above be replaced with a principled,
per-star value instead of an empirically-tuned global fraction -- a real
future improvement for the *stellar* classification case specifically.
Not relevant to ExGal (galaxies/QSOs sit beyond the full dust column
regardless of distance, so 2D SFD is already exactly correct there).

### Emission-line equivalent widths, and real E(B-V) in output headers (1 Sep 2026)

Two related additions, both team/user requests:

**Equivalent widths.** `EW_<line>`/`ERR_EW_<line>` columns (units:
Angstrom) added to both MOS (`MOSExGalEMIPPXF.py`'s `save_ppxf_as_emippxf`,
the `_emippxf_BIN.fits`/`_emippxf_spec_BIN.fits`-writing function -- see
"Two nearly-identical output functions" warning below) and IFU
(`IFUExGalEMIPPXF.py`'s `save_emi_kinematics_emippxf_format`) EMIPPXF
output, via a new shared `ExGalPrepare.compute_equivalent_width` helper:
integrated line flux divided by the local fitted stellar continuum at
the line's observed (redshifted) wavelength, sampled from ppxf's own
continuum model by linear interpolation. **Sign convention: positive for
genuine emission** -- the opposite convention from the Lick-style
stellar-absorption indices in the LS module (`Hbeta`/`Mgb`/`Fe5270`/etc,
positive for absorption). Error propagation is flux-error-only
(`ERR_EW = ERR_FLUX / continuum`, continuum treated as fixed) -- not a
full covariance propagation, documented as a real simplification.

Also propagated to: the MOS EMIPPXF diagnostic PNG plot's per-line text
box (`apsPlot/emi.py`, rest wavelength parsed from each line's own name
via the codebase's own `<label>_<rest_wavelength>` convention rather
than threading a new parameter through); `aps_MOSviewer.py`'s live
diagnostic plot (reuses the same `emi.build_figure`, so this came for
free); both MOS and IFU viewers' generic "Value Table" panels (`for c in
rec.columns.names` -- a fully generic column dump, so any new FITS
column shows up automatically with zero viewer code changes); and the
L2-merge step (`aps_L2merge.py`'s `gen_hdu`/astropy `join()` is a
generic column passthrough, not a hardcoded allowlist, so new columns
reach the final `_APS.fits` `GALAXY_TABLE` automatically too). **Not
propagated**: IFU's own per-bin diagnostic PNG (`apsPlot/emi_bin.py`,
called from inside `IFUExGalEMIPPXF.py`'s multiprocessing worker
`run_ppxf_emi` during the fit itself) -- deliberately left alone rather
than risk threading a new field through that worker's positional
queue-item packing/unpacking, a much higher-risk change than everywhere
else this landed cleanly; IFU viewer's separate emission-line-marker
plot (`aps_IFUviewer.py`'s `_emission_fit_figure`, a different, simpler
builder with no per-line text-box infrastructure to extend).

Verified via a real `proc_mosExGaL` PPXF+EMIPPXF+LS run: real, sensible
EW values (correctly NaN exactly where FLUX is NaN, correct sign,
magnitude consistent with the fitted continuum level). IFU's code path
is syntax/import-verified only -- no real IFU L1 data available locally
this session, same caveat as the IFU extinction wiring earlier.

**Two nearly-identical output functions, easy to edit the wrong one.**
Both MOS and IFU EMIPPXF each have two output-writing functions with
near-duplicate internal logic and similar-but-different array/variable
names (MOS: `save_emi_kinematics_mos` vs `save_ppxf_as_emippxf`; IFU:
`save_emi_kinematics` vs `save_emi_kinematics_emippxf_format`) -- one
writing files nothing downstream reads (`_emippxf.fits`/
`_emippxf_spec.fits`, no "_BIN" suffix), one writing the real,
LS/plotting/viewer-consumed files (`_emippxf_BIN.fits`/
`_emippxf_spec_BIN.fits`). Confirmed via `grep` that only the "_BIN"
files have any downstream reader anywhere in the codebase. This bit
during implementation: an early edit landed in MOS's dead-consumer
`save_emi_kinematics_mos` (which doesn't even receive a stellar
continuum parameter, so it would have crashed at runtime -- caught by
the real end-to-end test, not a code review) before being found and
redone in the correct function. Worth knowing before touching either of
these files again.

**Real E(B-V) in output headers.** The Galactic extinction correction
already applied upstream (see "Reused for ExGal" above) is now also
recorded in the PPXF, EMIPPXF, and LS output headers via a shared
`ExGalPrepare.write_ebmv_header(header, configs)`: `EBMVAPPL` (the
actually-applied E(B-V) in mag, or `'OFF'`), and for MOS's automatic
per-target mode, `EBMVMIN`/`EBMVMAX` giving the real spread across the
batch (`EBMVAPPL` there is the median -- explicit request: show the real
measured SFD value, not just an "AUTO" mode label, "so people can see
what absorption correction is already applied"). For MOS, this needed
`aps_mosExGal.py`'s `EBmV_APPLIED` finalisation moved to *after*
`APSOBJ` is built and a second (cheap) `get_sfd_ebv` call on the batch's
RA/Dec, since the per-target values themselves live inside `APSOB`'s
internals and weren't otherwise exposed back to the caller. IFU already
had the real single-patch value available (per its own WCS-centre
lookup design) with no extra computation needed.

While updating `doc/weave_datamodel_v8.md`'s `GALAXY_TABLE` column list
to match, found it already documented a per-line `EBMV_<line>`/
`ERR_EBMV_<line>` pair that was never actually implemented -- almost
certainly aspirational for the `REDDENING`/ppxf-`reddening=` fit
retired above, not the Galactic-foreground correction here. Removed
from that doc (real output tables carry no such column) rather than
left to mislead a future reader.

### Dead ExGal config keys removed (1 Sep 2026)

Tracing `EBmV`/`REDDENING` above prompted a full audit of every key in
every bundled `configs/ExGal_configs/*.json` (all 30 real mode files --
MOS/MOSLIFU*/MOSMIFU* follow one schema, LIFU*/MIFU* follow another),
checked against real, non-commented reads in the actual MOS and IFU
production pipeline files (not just "does this string appear anywhere
in the repo", which has false positives from dev/legacy scripts and
comments). Removed, since confirmed genuinely dead for the schema they
were removed from:

- **From all 18 MOS-schema configs**: `ORIGIN` (zero hits anywhere in
  the codebase, not even a comment), `PIXELSIZE` (MOS's only reference
  was commented out -- real uses are all IFU-side), `SPBIN_SIZE` (zero
  real MOS hits; its only occurrence anywhere was a coincidental
  same-name variable in an unrelated IFU dev script -- MOS has no
  spatial-binning concept, one spectrum per fibre), `HZ_LMIN_PPXF`/
  `HZ_LMAX_PPXF`/`HZ_LMIN_EMI`/`HZ_LMAX_EMI` (real, but only consumed by
  `aps_ifu_v0.py` -- a separate IFU-only script, not any MOS file or the
  current `aps_ifu_ExGal.py`).
- **From all 12 IFU-schema configs**: `PIXELSIZE` (real uses all
  MOS-side), `SETMODE` (real, but only one narrow use anywhere -- MOS's
  `MOSExGalEMIPPXF.py`, building an LSF config filename; the other 3
  MOS references are also just comments; never read by any IFU script),
  `FOR_ERRORS` (MOS-only real use).

**Not removed**, despite initially looking dead when checked only
against `aps_ifu_ExGal.py`: `EMIPPXF_LEVEL`, `FERRE_GRID_IDS`,
`FERRE_GRID_PREFIX`, `MIN_SNR_GAL`, `SPBIN_SIZE_GAL`, `TARGET_SNR_GAL`,
`VORONOI_GAL`. These are real, read by `aps_ifu_Gal.py` -- the
**Galactic/stellar** IFU pipeline, a sibling to `aps_ifu_ExGal.py` that
turns out to load its config from the exact same `IFU_params` JSON files
(the `_GAL`-suffixed keys are that pipeline's counterparts to the ExGal
ones -- `configs/ExGal_configs/` is a dual-purpose directory despite the
name). `REDDENING`/`EBmV` were not touched by this cleanup -- see above,
already handled deliberately (retired with a runtime notice, and fixed
respectively).

Verified safe two ways: (1) every value in every file was diffed
programmatically against the pre-cleanup version to confirm byte-for-byte
identical content apart from the removed keys (no accidental value
changes from the JSON re-serialisation); (2) a real `proc_mosExGaL` run
(PPXF+EMIPPXF+LS) through the actual cleaned `MOSLR11.json` completed
with no error, `REDDENING`'s retirement notice and the extinction
correction message both still firing correctly. Re-serialising every
file with `json.dump(..., indent=4)` also normalised previously
inconsistent tab/space alignment across files to one consistent style --
a side effect, not the goal, but a real (harmless) formatting change
worth knowing about if diffing an old export against a new one.

---

## TARGCLASS-Based Coarse-Scan Redshift Prior (opt-in, added 28 Aug 2026)

All fixes above act *after* the coarse+archetype fit already ran. Redrock
itself has a native mechanism that can act *during* the coarse scan:
`priors` (`redrock.priors.Priors`, a FITS file of TARGETID/Z/SIGMA/
FUNCTION rows) adds a chi2 penalty as a function of `(TARGETID, z)`,
identical across every template class for that target, applied
immediately after the coarse scan and before minima selection. A tight
Gaussian prior centred at z=0 for a target independently known to be a
STAR penalises any high-z GALAXY/QSO solution directly at the source.

New `--star_z_prior_sigma <dz>` (default `None` = off; ignored if
`--priors` is also explicitly set): auto-generates this file for every
target whose `TARGCLASS`/`TARGPROG`/`TARGSRVY` resolves to STAR (via
`_targeting_class_hint`, using each redrock `Target`'s own `.id`/`.meta`
— no extra I/O). `sigma≈0.0067` corresponds to ~2000 km/s.

**Important: this is a complementary, additive lever, not a substitute
for the archetype/targeting fixes above.** Verified on the same 8-target
GA-LRDISC batch:

| Configuration | Correct (of 8) |
|---|---:|
| Archetypes + targeting fix, no prior | 6/8 (APS_ID 6, 613 wrong) |
| Archetypes + targeting fix + z-prior | **7/8** (APS_ID 6 fixed too) |
| z-prior alone, archetypes off | Does **not** flip class — moves the coarse GALAXY answer off the spurious z~1.6 trap onto a *different*, still-wrong z~0.16-0.20 GALAXY minimum |

The prior alone doesn't flip the class because the underlying
template-flexibility asymmetry (GALAXY's 10 PCA coefficients vs. STAR's
5) still lets GALAXY win at *some* redshift on its own — the prior just
removes one particular escape route. Fixes APS_ID 6 specifically because
that case is `coarse=GALAXY, archetype=QSO` — the targeting-corroboration
fix above cannot touch it at all (it only fires when the archetype's
*own* class already agrees with targeting; QSO never does for a STAR
target). APS_ID 613 remains wrong with or without the prior: its coarse
GALAXY answer is already at z≈0 (a genuine continuum-shape degeneracy,
not a high-z excursion), so a z=0 prior has nothing to penalise there.

---

## Other Preprocessing Chain Findings (28 Aug 2026 full-chain review)

Reviewing `aps_utils.py`'s `APSOB`/`_process_single_arm_vectorized`
preprocessing chain end to end (the actual input-preparation steps
`aps_rr.py` relies on before any redrock call) for anything else that
might help GALAXY or STAR classification:

- **Sky-residual masking is genuinely active**, not dead code as a stray
  commented-out block inside `_process_single_arm_vectorized` first
  suggested — the real call is `_apply_sky_residual_masking`, invoked
  once per observation (across all arms) in the calling method, *after*
  per-arm processing and *before* IVAR normalisation. Consistent with
  this doc's and the team email's description of it as an active feature.
- **`skysub_mask_residuals` CLI flag was dead** — fixed above (see
  Version History 3.10).
- **Not yet investigated as part of this review** (candidates for a
  follow-up pass, not yet checked): whether `arms_ratio=[1.0, 1.0]` (used
  throughout all the verification runs in this document) is actually the
  right calibrated value for LR mode, or whether a properly-calibrated
  non-unity ratio (the CLI docstring's own worked example uses `0.83`
  for the red arm in one older config) would reduce the kind of
  arm-color imbalance that Galactic extinction correction is currently
  compensating for; whether the `wlranges`/CCD-gap (`gap_bands`) and
  telluric-band definitions in `aps_constants.py` are current and
  complete; and whether cosmic-ray removal / gap-filling parameters have
  any measurable effect on the classification failure modes documented
  above (as opposed to redshift precision, which is what they were
  originally tuned for).

---

## CCD Gap Handling (verified 28 Aug 2026)

Redrock was designed for DESI, which has no physical inter-chip CCD gaps
in its wavelength coverage. WEAVE does (each arm's detector is a mosaic
of chips with real dead wavelength ranges between them). This raised the
question of whether gaps need special handling before reaching redrock,
beyond what redrock does natively.

**Short answer: no special redrock-side handling is needed, and the
existing preprocessing already does the right thing.** Verified two ways:

1. **By construction**: redrock's coarse scan and archetype refit are
   both strictly IVAR-weighted least squares (`M = Tb.T @ (w[:,None]*Tb)`,
   `y = Tb.T @ (w*flux)`, `chi2 = sum((f-model)^2 * w)` — see the
   `calc_zchi2_batch` patch above). A pixel with `ivar=0` contributes
   exactly zero to every one of these sums, regardless of what its flux
   value is. This means redrock already handles arbitrarily-placed
   zero-weight pixels correctly -- it doesn't need to know a *gap*
   exists as a distinct concept, as long as the pixels inside it carry
   `ivar=0`. DESI relies on the same mechanism for its own bad-pixel/
   cosmic-ray masking; WEAVE's gaps are just larger contiguous blocks of
   the same thing.

2. **Empirically, on real data**: `aps_utils.py`'s preprocessing chain
   zeroes `ivar` across gaps two ways, both applied by default
   (`mask_gaps=True`, `safe_mask_gaps=True`):
   - `mask_gaps` (`vectorized_gap_masking`) auto-detects wherever the
     L1 data's own `ivar` is already exactly 0 and dilates that mask by
     `offset_gap_pix` pixels on each side (a small buffer against
     reduced-quality pixels right at a gap edge that aren't already
     flagged `ivar=0`).
   - `safe_mask_gaps` additionally zeroes `ivar` across fixed wavelength
     bands from `aps_constants.gap_bands` (a per-instrument-mode lookup
     table), as a safety net independent of whatever the L1 data itself
     happened to flag.

   Checked directly against a real MOS-LR observation (WC OB
   20250630/16287, APS_ID 938): the blue arm shows `ivar=0` runs at
   5451.5-5568.8 Å (470 px) and 5921.6-5951.4 Å (120 px); the red arm at
   7542.1-7699.4 Å (630 px) -- matching `gap_bands['MOSLR_BLUE']`
   (`[5450,5565]`, `[5920,6000]`) and `['MOSLR_RED']` (`[7540,7680]`)
   almost exactly (redrock's own `Z_FITLIMIT`-style wavelength-range
   trimming accounts for the small differences at the very ends). Gap
   masking is real and active, not a dead/inert setting.

3. **Fill_gap is orthogonal and doesn't affect any of this**: `fill_gap`
   linearly interpolates *flux* across `ivar<=0` regions but does NOT
   touch `ivar` itself -- those pixels stay at `ivar=0` and so still
   contribute nothing to any redrock fit either way. It exists for other
   consumers (continuity in diagnostic plots, other pipeline stages that
   may not be purely IVAR-weighted), not for redrock's benefit.

**One real, pre-existing gap not investigated further here**:
`aps_constants.py`'s `gap_bands` table carries an explicit `to-do:
Update the table for all modes` comment and was last updated using data
"up to July 2024" -- if the instrument's actual gap locations have
drifted or been recalibrated since, this lookup table (the `safe_mask_gaps`
safety net specifically -- `mask_gaps`'s own auto-detection is
data-driven and doesn't depend on this table) would be stale. Confirming
current accuracy requires input from the instrument/calibration team, not
something checkable from the pipeline code alone -- flagged, not fixed,
same as in "Other Preprocessing Chain Findings" above.

---

## Collapse Mode (IFU) — Verified 28 Aug 2026

WEAVE's IFU (LIFU/mIFU) classification/redshift pipeline
(`aps_ifu_prepare.py`, `aps_ifu_v0.py`) feeds redrock a single
*collapsed* spectrum per patch -- `APSOB(..., collapse=True)` combines
every spaxel in a patch into one IVAR-weighted-mean spectrum
(`APS_ID=-999`, `CNAME='collapsed'`) before `rrweave_worker` ever runs.
None of `aps_rr.py`'s CLI exposes `--collapse` (it's only reachable by
calling `rrweave_worker`/`rrweave` directly, which is how the IFU
callers use it), so every fix documented above needed checking against
this mode specifically rather than assumed to carry over.

**Verified via a real end-to-end `rrweave_worker(..., collapse=True,
extinction_corr=True)` call** (5-target WC GALAXY collapse, APS_ID
118/938/618/348/168 combined): ran cleanly, `_apply_archetype_fallback`'s
Step 1 executed and printed all ZWARN bits 16-26 as normal, extinction
correction applied without error, final classification GALAXY z=0.0428
(consistent with the individual members' own z~0.04-0.09), SRVY_CLASS
correctly resolved via TARGCLASS. Collapse-mode processing order is:
per-spaxel corrections (gap masking, sensitivity, extinction, sky-residual
masking, cosmic-ray removal -- `crr` is force-enabled for collapse,
since combining more exposures raises real cosmic-ray risk) all happen
in `_process_arms_serial`/`_process_single_arm_vectorized` **before**
collapse combines spaxels, so collapse always operates on already-corrected
data -- confirmed directly: the collapsed spectrum's gap `ivar=0` runs
are pixel-for-pixel identical to a single member target's own (see "CCD
Gap Handling" above), not weakened or shifted by the averaging. Every fix
added this session (extinction correction, the archetype-fallback/
targeting/z-shift/BAD_MINFIT-relaxation logic, `star_z_prior_sigma`) is
generic over `Target`/`zfitall`/`scandata` structures with no MOS-specific
assumptions, so none of it needed collapse-specific changes.

**One real bug found and fixed while checking this** (`aps_utils.py`,
`APSOB.__init__`'s collapse block): the collapsed target's `TARGSRVY`
was being silently overwritten with a copy of its own `TARGCLASS` string
(`targsrvy = targsrvy` as a no-op, immediately followed by
`targsrvy = targclass` -- almost certainly a copy-paste mistake, given
the surrounding lines' pattern of deliberate resets). Verified directly:
before the fix, a collapsed WC GALAXY patch reported `TARGSRVY='GALAXY'`
instead of the real survey code `'WS2023A1-022'`. `TARGCLASS`/`TARGSRVY`/
`TARGPROG` all feed `_targeting_class_hint` (checked in that priority
order) and `classify_target_comprehensive`'s SRVY_CLASS assignment, so a
corrupted `TARGSRVY` specifically breaks the *TARGSRVY* fallback tier of
those lookups for every collapsed target -- not the `TARGCLASS` tier the
archetype-fallback fixes mostly rely on (checked first, and untouched by
this bug, which is presumably why it went unnoticed), but a real, silent
metadata corruption regardless. Fixed by simply not touching `targsrvy`
(same treatment as `targclass` just above it, which was already left
alone) -- verified fixed via the same live collapse test. `TARGPROG` is
still deliberately blanked to `""` for collapsed targets (a legitimate
design choice: a collapsed patch may span multiple programs, so leaving
it unset is more honest than picking one arbitrarily) and was not
changed.

**Not yet investigated**: whether `TARGCLASS` itself should be a
computed consensus across the collapsed group rather than simply
inherited from whichever spaxel happens to be first in `self._targetlist`
-- for `aps_ifu_prepare.py`'s actual usage pattern (spaxels belonging to
one astrophysical target within an IFU patch) this is expected to
normally be uniform anyway, but has not been checked against real,
possibly-mixed-class IFU patches.

---

## ZWARN-Aware Promotion — Disabled

An earlier version of PyAPS promoted the first ZWARN=0 rank to rank 0 when
the chi2 winner had ZWARN≠0. This is now **disabled** because:

- `ZWARN=64` (APS internal flag, now removed) caused wrong promotions
- `ZWARN=4` (BAD_MINFIT) promoted rank was often worse than rank 0
- `ZWARN=1024` (Z_FITLIMIT) at z=1.19 was promoted to z=0.000 (wrong)

The archetype fallback (Step 1) handles all genuine failures. ZWARN
is now treated as purely informational in rank selection.

---

## Known Limitations

**1. Coarse scan degeneracy**

When the spectrum contains no discriminating features at the true redshift
(passive galaxies, very low SNR, stellar contamination), the coarse PCA
scan may find a wrong global minimum. The fallback cannot correct this
because it trusts the coarse scan class and z as ground truth. See the
Unfixable Failure example above (APS_ID 938/791) for a real case under
active investigation — `nminima` truncation has been ruled out as the
sole cause; redshift-grid sampling density is the leading remaining
hypothesis, not yet confirmed.

**2. High-z galaxies (z > 1.6)**

GALAXY archetype ranks for z > 1.6 frequently have `ZWARN=Z_FITLIMIT`
(Redrock's redshift grid ends near z=1.7). These ranks are excluded by the
ZWARN>2 filter, preventing the fallback from finding a valid high-z galaxy
rank when the coarse scan correctly identifies z>1.6. The tier-2
BAD_MINFIT relaxation added in 3.11 (see "Fallback rank selection" above)
does *not* help here — it only tolerates `BAD_MINFIT` (bit 10), not
`Z_FITLIMIT` (bit 5); the two flags mean different things (an uncertain
parabola vertex vs. the minimum sitting at the edge of the searched grid,
which is a real "we may not have found the true minimum" warning) and
were not conflated. Extending tier 2 to also tolerate `Z_FITLIMIT` is a
plausible future addition but has not been verified against a real case
the way the BAD_MINFIT relaxation was (WC APS_ID 948, 931) — do not add
it without the same kind of real, evidence-backed check.

**3. SNR < 3**

Below SNR≈3, the chi2 landscape is dominated by noise. Any template can
win depending on noise realisation. Catastrophic redshifts are expected
and statistically inevitable at these SNR levels. The SNR_APSMOD column
in the zbest output allows downstream filtering.

**4. Archetype coverage gaps**

The DESI archetype set does not cover all galaxy SED types observed by
WEAVE — specifically dust-reddened galaxies and TP-AGB dominated populations
with rising red continua. For these targets the fallback corrects the class
but the archetype z precision is reduced (the best available archetype z
may differ from the true z by hundreds of km/s).

---

## Recommended Downstream Usage

```python
import numpy as np
from astropy.table import Table

zbest = Table.read('zbest_output.fits')

# Reliable redshifts: Redrock clean + no APS issues
# (most conservative cut)
mask_clean = (zbest['ZWARN'] == 0)

# Reliable redshifts: include APS-corrected targets
# where the fallback swap was applied
mask_reliable = (
    ((zbest['ZWARN'] & 0x3FF) == 0) |     # Redrock clean
    ((zbest['ZWARN'] & 524288) > 0)        # APS swap applied
)

# Targets needing inspection:
# trigger fired but no valid rank could be found
mask_inspect = (zbest['ZWARN'] & 1048576) > 0

# SNR-quality cut (remove noise-dominated targets):
mask_snr = zbest['SNR'] >= 3.0

# Recommended extragalactic sample:
mask_gal = (
    mask_reliable &
    mask_snr &
    ~mask_inspect &
    (zbest['CLASS'] == 'GALAXY')
)
```

---

## Debug Mode

When `debug=True` is passed to `gen_zbest_multiple`, the `APSFallbackDiagnostic`
class records full per-target fallback history and writes two output files:

```
<outpath>/<headname>_fallback_diag.fits   ← full machine-readable table
<outpath>/<headname>_fallback_diag.txt    ← human-readable ASCII
```

Columns recorded per target:

| Column | Description |
|--------|-------------|
| `aps_id` | APS target ID |
| `coarse_class/z/chi2` | Coarse PCA scan winner |
| `arch_class/z/chi2/dchi2` | Archetype rank 0 result |
| `pca_reliable` | Whether coarse PCA minimum is well-defined |
| `z_pca` | Parabola-refined z |
| `dv_pca_coarse` | Velocity offset coarse vs parabola (km/s) |
| `curvature_norm` | Normalised chi2 curvature (sharper = more reliable) |
| `class_mismatch` | Class mismatch trigger fired |
| `ratio_trigger` | Ratio trigger fired |
| `z_shift_trigger` | Z-shift trigger fired |
| `degradation` | arch_chi2 / coarse_chi2 |
| `dv_arch_coarse` | Velocity offset arch vs coarse z (km/s) |
| `fallback_applied` | Whether swap was made |
| `fallback_z/class/chi2` | Fallback rank result |
| `no_valid_rank` | Trigger fired but no ZWARN<=2 rank found |
| `final_zwarn` | Final ZWARN after APS bits set |

The diagnostic is thread-safe — multiple parallel workers can record
simultaneously via an internal threading lock.

---

## Version History

| Version | Change |
|---------|--------|
| 1.0 | Initial Redrock integration |
| 2.0 | Per-arm chi2 monkey patch |
| 2.5 | IVAR balanced normalisation |
| 3.0 | tellurics=False|
| 3.1 | Archetype fallback v1 (class mismatch + ratio trigger) |
| 3.2 | ZWARN>2 exclusion, no Pass 2, ZWARN promotion disabled |
| 3.3 | Z-shift trigger (Condition 3) with velocity threshold |
| 3.4 | PCA fine fit reliability check, z_reference selection |
| 3.5 | APS-extended ZWARN bits (16-23), APSFallbackDiagnostic |
| 3.6 | Diagnostic plotting moved to `PyAPS.viz.redrock` (Plotly), shared with the new cross-module PyAPS.viz platform. Fixed `gen_zspec` computing only the rank-0 model spectrum: `MODEL_RR_<arm>` is now `float[NRANK,NW]` (was `float[NW]`), so each rank panel in the fit plot shows its own genuine best-fit model instead of all ranks sharing the rank-0 curve. See `doc/weave_datamodel_v8.md`. |
| 3.7 | Targeting-corroboration carve-out in the class-mismatch trigger (`_targeting_class_hint`, `targeting_override`, ZWARN bit 24) — fixes genuine STAR targets previously reverted to GALAXY by the blanket rule, without regressing the original catastrophic-galaxy-redshift cases it protects against. See "Failure Mode 1" above. |
| 3.8 | Archetype-z-improvement trust in the z-shift trigger (`archetype_improved_same_class`, ZWARN bit 25) — stops the fallback discarding a genuinely better archetype redshift within the same class just because it's far from the coarse scan's own (worse) z. See "Failure Mode 3" above. |
| 3.9 | Opt-in Galactic (SFD98+Fitzpatrick99) extinction correction, off by default (`--extinction_corr`/`--extinction_ebv_scale`/`--extinction_mapdir`). See "Galactic Extinction Correction" above. Population-scale GA-LRDISC result: 67.1%->86.3% correctly STAR (513->660 of 765, 165 fixed/18 regressed). WC/GALAXY side (44-target batch): 5 fixed/3 regressed, including one class flip. Large net positive on both, but individual regressions are real on both sides -- stays opt-in, not a default. |
| 3.10 | Opt-in TARGCLASS-based coarse-scan z=0 redshift prior for STAR targets (`--star_z_prior_sigma`), off by default. See "TARGCLASS-Based Coarse-Scan Redshift Prior" above -- complementary to the archetype/targeting fixes (3.7), fixes an additional case (APS_ID 6, coarse=GALAXY/archetype=QSO) they cannot reach alone. Also fixed `read_spectra`'s dead `skysub_mask_residuals` parameter (was hardcoded `True`, ignoring the value actually passed in). |
| 3.11 | Second-tier BAD_MINFIT relaxation in the fallback rank-substitution search (default-on, not opt-in -- see "Fallback rank selection" above), gated to `class_mismatch and not targeting_override` after real testing caught an ungated version actively regressing 20 correct STAR classifications in the GA-LRDISC population batch. Verified: WC 43-target batch fixes APS_ID 948 (STAR->GALAXY) and 931 (QSO->GALAXY), zero change to the other 41; GA-LRDISC 765-target population batch shows zero STAR-count change (601->601) and only 9 lateral QSO<->GALAXY relabellings among targets already wrong on both sides. Also corrected the Redrock-native ZWARN bit-value table in this document, which didn't match the installed redrock version at all. Also: verified CCD-gap handling needs no redrock-side changes (ivar=0 masking is structurally sufficient for redrock's IVAR-weighted fitting) and verified/fixed collapse (IFU) mode -- see "CCD Gap Handling" and "Collapse Mode (IFU)" above; found and fixed a real bug where a collapsed target's TARGSRVY was silently overwritten with its own TARGCLASS string. |
| 3.12 | Branch `redrock-perarm-audit` (not merged): per-arm chi2 patch audited and corrected (per-camera columns no longer diluted by 1/n_arm in the stored coefficients; arm with no template support adds its full weighted flux^2 instead of HUGE_CHI2), `--rr_solver perarm\|joint` switch (default unchanged), docs rewritten, 14 synthetic tests. Real-data A/B (OB 20250630/16287, 840 fibres): final galaxy z within 1000 km/s of reference 94.8% per-arm vs 82.4% joint; see PyAPS_local/PyAPS_redrock_concern/requested_outputs/FINDINGS.md. |
