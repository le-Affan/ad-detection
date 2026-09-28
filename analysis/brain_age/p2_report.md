# P2 — DeepBrainNet brain-age gap: computation, sanity checks, ablation

Branch `brain-age`. Date 2026-09-28. Train and val only. **No test subject's
scan was loaded, processed or evaluated;** test brain age is computed once in
P5 with this frozen pipeline. Rules: `p2_preregistration.md`, committed as
`2936320` before any result below existed.

## Verdict

**Card only.** Against the pre-registered rule, the brain-age gap is not
adopted as a fusion input:

| Pair | CV macro-F1 diff, 95% CI | Val diff | Result |
|---|---|---|---|
| E − B (clinical + ALZ + gap vs clinical + ALZ) | −0.002 [−0.011, +0.006] | −0.016 | no reliable gain |
| E′ − B′ (without the defining scores) | +0.004 [−0.007, +0.016] | −0.002 | no reliable gain |

The gap itself is a sound brain-health measure on this cohort:
- It rises monotonically CN → EMCI → LMCI → AD.
- It's uncorrelated with age after correction.
- It separates AD from CN with a CV AUC of 0.80 on its own.
- It's about 60% independent of the FreeSurfer volumes and ALZ.

It adds no staging accuracy on top of the clinical features, as ALZ didn't
either. It belongs on the assessment card as independent imaging evidence.

---

## 1. Computation

- **Model:** DeepBrainNet `Models/DBN_model.h5`, repo `505f4e4`, sha256
  `9257e98e…41ea` (checked at start of run).
  - **This is the repo's released VGG16-based network, not the
    Inception-ResNet-v2 described in the paper.**
  - The accuracy figures below are our own measurements, not the paper's 3.7.
- **Pipeline:** unchanged from the smoke test. One adaptation (a
  world-coordinate trilinear resample of the m1 `mni/` volume onto the
  182×218×182 MNI 1 mm grid), then the repo's steps: 185/p97 scaling, axial
  slices 45–124, JPEG, 256×256 nearest, /255, median of 80 slice predictions.
- **Runtime:** CPU only (TensorFlow has no GPU here). 687 scans in ~44 min
  (~3.5 s/scan). **0 failures** (`brainage_failures.csv` is empty).
- **Reproducibility:** the 40 smoke-test subjects reproduce to within
  0.005 years.
- **Chronological age at scan:** ADSL `AGE` (age at enrolment, 0.1-year
  resolution) + (`MRI_EXAMDATE` from `dataset/mri_manifest/mri_manifest.csv`
  − ADSL `ENRLDT`)/365.25.
  - The P1 smoke tests anchored on `BASELINE_EXAMDATE` (screening) instead,
    about 40 days earlier: a ~0.1-year difference, used for P2 onward.
- **Output:** `brainage_trainval.csv`: RID, PTID, split, DX_bl, label,
  age_at_scan, pred_age, gap_raw, has_brainage, missing_reason (plus
  slice-prediction IQR and the age source columns).

### Coverage

| Split | Subjects | With brain age | Missing (NaN, not imputed) |
|---|---|---|---|
| train | 569 | **566** | 027_S_4966 (EMCI, m1 registration failed); 098_S_4095 (AD, no scan); 126_S_2360 (EMCI, no scan) |
| val | 122 | **121** | 941_S_4187 (LMCI, no scan) |
| test | 123 | not computed (P5) | — |

Predicted ages span 52.3–84.8 (train mean 73.8 vs chronological 72.5; val
73.4 vs 73.1). The median within-scan IQR of slice predictions is 3.0 years.

## 2. Sanity checks (train only for anything label-based; `p2_sanity.md`, `p2_sanity_plots.png`)

### 2.1 Accuracy on CN train (n = 133)

Bias fit: predicted = **0.582 · age + 29.97**. The slope is well below 1:
strong compression towards the mean age.

| | MAE | Mean error | r with age | SD pred / SD age |
|---|---|---|---|---|
| Raw | **3.16** | −0.72 | 0.745 | 0.781 |
| Bias-corrected | 3.97 | +0.00 | 0.745 | 1.343 |

**Our measured CN MAE is 3.16 years (raw).**
- Correction removes the age dependence but raises MAE. Dividing by α = 0.58
  scales the residual noise by 1.7×.
- This is the known cost of the α/β correction when α is well below 1.

### 2.2 Corrected gap by diagnosis (train)

| Diagnosis | n | Mean gap | SD | Median |
|---|---|---|---|---|
| CN | 133 | +0.00 | 5.47 | +0.69 |
| EMCI | 215 | +2.20 | 5.10 | +2.67 |
| LMCI | 114 | +4.02 | 5.13 | +4.01 |
| AD | 104 | **+6.80** | 6.26 | +6.34 |

- **Kruskal–Wallis H = 76.0, p = 2.2 × 10⁻¹⁶.**
- CN ≈ 0 holds by construction. The rise is monotonic, as expected.
- The steps are about 2 years each, including EMCI vs CN.

### 2.3 Gap vs age

- **CN train: r(corrected gap, age) = 0.000** (it was −0.626 for the raw gap).
- Across all train diagnoses, r = −0.17: younger patients carry larger gaps.

### 2.4 Site effect (CN train; site = PTID prefix)

44 sites; only 6 have ≥ 5 CN subjects. Overall CN gap SD is 5.47.

| Site | CN n | Mean gap | z (vs SE) | z (vs individual SD) |
|---|---|---|---|---|
| **016** | 6 | **−8.68** | **−3.89 (FLAG)** | −1.59 |
| 073 | 6 | −2.95 | −1.32 | −0.54 |
| 003 | 6 | −0.58 | −0.26 | −0.11 |
| 011 | 5 | −0.25 | −0.10 | −0.04 |
| 031 | 5 | +2.53 | +1.03 | +0.46 |
| 094 | 5 | +3.96 | +1.62 | +0.73 |

- **Site 016 is flagged:** its 6 CN subjects average 8.7 years younger-looking
  than CN elsewhere. This could be a scanner or protocol effect, or chance
  with n = 6.
- No site is more than 2 SD of individual CN gaps from the mean.
- With most sites holding fewer than 5 CN subjects, site effects can't be
  assessed more finely. **The gap is not harmonised across sites.**

### 2.5 How much of the gap is new information (train, Spearman ρ)

| | ALZ | Ventricles | Hippocampus | WholeBrain | Entorhinal | Fusiform | MidTemp | ICV | Age |
|---|---|---|---|---|---|---|---|---|---|
| All train | +0.35 | +0.34 | −0.25 | −0.09 | −0.19 | −0.21 | −0.20 | +0.07 | −0.13 |
| CN train | +0.25 | +0.36 | −0.05 | +0.02 | +0.05 | −0.11 | −0.02 | +0.13 | −0.01 |

- A linear model of the gap on the 7 volumes, ALZ and age explains
  R² = 0.40 in-sample, so **~60% of the gap's variance is not linearly
  explained by inputs we already have.**
- Its strongest overlap is with ventricular volume and ALZ.

### 2.6 Standalone signal (the gap as the score; 5-fold CV on train, per-fold CN bias fit)

| Contrast | AUC |
|---|---|
| AD vs CN | **0.797** |
| AD vs rest | 0.713 (ALZ on the same subjects: 0.764) |
| LMCI vs CN | 0.707 |
| EMCI vs CN | 0.620 |

## 3. Ablation (`p2_ablation.py`, reusing `analysis/fusion_feasibility/ablation.py`)

Protocol as pre-registered:
- CatBoost with a fixed 265 trees, seeds 42–46.
- CV: stratified 5-fold over the 569 clinical-train subjects. Bias correction
  is refitted in every fold on that fold's training CN subjects only (α
  0.55–0.65, β 24.7–32.3, 106–107 CN per fold).
- Val: train on 569, evaluate on 122, with correction fitted on all 133 train
  CN (α 0.582, β 29.97).
- BAG is NaN for the 4 subjects without brain age.

**Harness check:** arms A, B, A′ and B′ reproduce fusion-feasibility's
per-seed metrics exactly (max |diff| 1.1 × 10⁻¹⁶).

**Cross-validation, 569 clinical-train (primary), mean ± SD over 5 seeds**

| Arm | Macro-F1 | CN F1 | EMCI F1 | LMCI F1 | AD F1 | Bal. acc | AD AUC |
|---|---|---|---|---|---|---|---|
| A | 0.752 ± 0.006 | 0.947 ± 0.002 | 0.786 ± 0.003 | 0.410 ± 0.019 | 0.865 ± 0.003 | 0.750 ± 0.004 | 0.982 ± 0.001 |
| B | 0.752 ± 0.006 | 0.949 ± 0.002 | 0.788 ± 0.007 | 0.405 ± 0.015 | 0.864 ± 0.012 | 0.749 ± 0.006 | 0.981 ± 0.001 |
| **E** | 0.749 ± 0.008 | 0.947 ± 0.002 | 0.782 ± 0.006 | 0.406 ± 0.022 | 0.863 ± 0.009 | 0.747 ± 0.007 | 0.982 ± 0.001 |
| A′ | 0.620 ± 0.006 | 0.704 ± 0.010 | 0.667 ± 0.007 | 0.351 ± 0.013 | 0.758 ± 0.021 | 0.621 ± 0.006 | 0.958 ± 0.001 |
| B′ | 0.629 ± 0.010 | 0.697 ± 0.021 | 0.666 ± 0.010 | 0.374 ± 0.024 | 0.778 ± 0.011 | 0.629 ± 0.009 | 0.958 ± 0.002 |
| **E′** | 0.633 ± 0.009 | 0.713 ± 0.012 | 0.684 ± 0.003 | 0.371 ± 0.029 | 0.765 ± 0.011 | 0.634 ± 0.009 | 0.958 ± 0.001 |
| G | 0.750 ± 0.006 | 0.947 ± 0.002 | 0.790 ± 0.005 | 0.405 ± 0.012 | 0.859 ± 0.012 | 0.748 ± 0.005 | 0.981 ± 0.001 |

**Validation, train 569 → 122 clinical-val, mean ± SD over 5 seeds**

| Arm | Macro-F1 | CN F1 | EMCI F1 | LMCI F1 | AD F1 | Bal. acc | AD AUC |
|---|---|---|---|---|---|---|---|
| A | 0.778 ± 0.020 | 0.982 ± 0.000 | 0.818 ± 0.021 | 0.395 ± 0.058 | 0.917 ± 0.014 | 0.785 ± 0.016 | 0.995 ± 0.001 |
| B | 0.796 ± 0.017 | 0.978 ± 0.008 | 0.828 ± 0.007 | 0.456 ± 0.047 | 0.921 ± 0.016 | 0.799 ± 0.012 | 0.995 ± 0.001 |
| **E** | 0.780 ± 0.016 | 0.982 ± 0.000 | 0.827 ± 0.012 | 0.406 ± 0.046 | 0.905 ± 0.010 | 0.788 ± 0.012 | 0.994 ± 0.001 |
| A′ | 0.644 ± 0.009 | 0.778 ± 0.030 | 0.712 ± 0.023 | 0.307 ± 0.028 | 0.778 ± 0.011 | 0.652 ± 0.012 | 0.963 ± 0.002 |
| B′ | 0.625 ± 0.025 | 0.758 ± 0.014 | 0.685 ± 0.020 | 0.295 ± 0.067 | 0.760 ± 0.022 | 0.632 ± 0.021 | 0.964 ± 0.002 |
| **E′** | 0.623 ± 0.011 | 0.752 ± 0.020 | 0.665 ± 0.024 | 0.304 ± 0.026 | 0.770 ± 0.027 | 0.632 ± 0.010 | 0.963 ± 0.003 |
| G | 0.788 ± 0.019 | 0.982 ± 0.000 | 0.830 ± 0.019 | 0.432 ± 0.057 | 0.909 ± 0.010 | 0.794 ± 0.015 | 0.995 ± 0.002 |

**Subjects with brain age only** (566 CV / 121 val): every arm moves by
≤ 0.005 and every difference by ≤ 0.001. Full tables, including accuracy, are
in `p2_ablation_tables.md`.

### Paired macro-F1 differences (2,000 bootstrap resamples of subjects, mean over 5 seeds)

| Pair | Eval | Diff | 95% CI | Per-seed (42–46) |
|---|---|---|---|---|
| **E − B** | CV | −0.002 | [−0.011, +0.006] | +0.002 −0.009 −0.009 +0.006 −0.001 |
| **E − B** | val | −0.016 | [−0.046, +0.009] | −0.024 −0.028 +0.019 −0.015 −0.030 |
| **E′ − B′** | CV | +0.004 | [−0.007, +0.016] | +0.010 +0.001 +0.001 −0.003 +0.012 |
| **E′ − B′** | val | −0.002 | [−0.019, +0.017] | −0.009 −0.040 +0.026 −0.013 +0.027 |
| G − A (secondary) | CV | −0.001 | [−0.010, +0.007] | −0.007 +0.001 −0.011 +0.005 +0.006 |
| G − A (secondary) | val | +0.010 | [−0.013, +0.034] | −0.021 +0.046 +0.031 −0.000 −0.005 |

Val bootstraps are descriptive.

### Confusion matrices on val (mean over 5 seeds; rows true, columns predicted)

| B | CN | EMCI | LMCI | AD | | E | CN | EMCI | LMCI | AD |
|---|---|---|---|---|---|---|---|---|---|---|
| CN | 27.0 | 1.0 | 0 | 0 | | CN | 27.0 | 1.0 | 0 | 0 |
| EMCI | 0.2 | 42.8 | 3.0 | 0 | | EMCI | 0 | 43.0 | 3.0 | 0 |
| LMCI | 0 | 13.6 | 8.6 | 2.8 | | LMCI | 0 | 14.0 | 7.4 | 3.6 |
| AD | 0 | 0 | 1.0 | 22.0 | | AD | 0 | 0 | 1.0 | 22.0 |

| B′ | CN | EMCI | LMCI | AD | | E′ | CN | EMCI | LMCI | AD |
|---|---|---|---|---|---|---|---|---|---|---|
| CN | 22.2 | 5.8 | 0 | 0 | | CN | 22.4 | 5.6 | 0 | 0 |
| EMCI | 7.4 | 34.0 | 4.6 | 0 | | EMCI | 8.0 | 32.4 | 5.6 | 0 |
| LMCI | 1.0 | 12.6 | 6.0 | 5.4 | | LMCI | 1.2 | 12.2 | 6.2 | 5.4 |
| AD | 0 | 0.8 | 4.8 | 17.4 | | AD | 0 | 1.2 | 4.0 | 17.8 |

**Adding the gap moves no error pattern.**
- B → E: LMCI→EMCI 13.6 → 14.0, LMCI→AD 2.8 → 3.6.
- B′ → E′: AD→LMCI 4.8 → 4.0, EMCI→CN 7.4 → 8.0.
- All shifts are within one subject on average, i.e. within seed noise.

### SHAP of the gap (val; models trained on all 569)

| Arm | Gap rank (per seed) | Mean \|SHAP\| | ALZ rank (mean) |
|---|---|---|---|
| E | 26.6 (22, 29, 31, 25, 26) of 38 | 0.027 ± 0.005 | 16.0 |
| E′ | 19.4 (22, 19, 17, 24, 15) of 35 | 0.057 ± 0.007 | 22.6 |
| G | 22.4 (22, 25, 29, 15, 21) of 37 | 0.031 ± 0.009 | — |

- With the full clinical set, the gap is a low-ranked feature.
- Without the defining scores (E′), it moves up to about 19th and outranks
  ALZ, but it still doesn't change macro-F1.

## 4. Limitations

- **Architecture.** The released model is VGG16-based, not the paper's
  Inception-ResNet-v2. We cite our own measured accuracy (CN train MAE 3.16
  raw, r = 0.75), not the paper's 3.7.
- **Our input volumes differ from the model's training input:**
  - About 5% of brain voxels are saturated at 1.0 by m1's 5–95% truncation.
  - Skull stripping uses the ICBM152 template mask, not a subject-specific
    mask, leaving some skull-base tissue in the lowest slices.
  - Registration is ANTs affine to ICBM152 2009a, not FLIRT.
  - The smoke test showed the model tolerates these (stable to ±10%
    intensity). A systematic offset can't be excluded, but the CN-only bias
    correction absorbs any global shift.
- **Compression towards the mean age.** α = 0.58: raw predictions
  under-estimate the old and over-estimate the young.
  - The correction removes the age dependence (r = 0.000 in CN) at the cost
    of 1.7× noise amplification (corrected CN MAE 3.97).
  - Individual gaps are noisy (SD ≈ 5.5 years in CN), so the gap is a group-
    or trend-level measure, not a precise individual one.
- **Outliers.** Two CN train subjects have corrected gaps of about −23 and
  −25 years (raw predictions 57–59 at ages 71–73; see
  `p2_sanity_plots.png`). They have not been inspected; scan quality is a
  plausible cause. They were kept, as pre-registered.
- **Site.** No harmonisation. One site (016) is flagged for a large negative
  mean in its 6 CN subjects. Most sites have too few CN subjects to check.
- **Age anchor.** ADSL `AGE` has 0.1-year resolution and is anchored at
  enrolment, then shifted to the scan date.
- **Cohort size.** Val has 25 LMCI and 23 AD subjects, so val differences of
  ±0.02 macro-F1 are within seed-to-seed spread.

## 5. Files

| File | Content |
|---|---|
| `p2_preregistration.md` | the rules (commit `2936320`) |
| `compute_brainage.py`, `compute_brainage.log`, `brainage_failures.csv` | Step 1 (resumable; `.partial.csv` is its checkpoint) |
| `brainage_trainval.csv` | brain age for train + val |
| `p2_sanity.py`, `p2_sanity.md`, `p2_site_effects.csv`, `p2_sanity_plots.png` | Step 3 |
| `p2_ablation.py`, `p2_ablation_tables.md`, `p2_ablation_runs.csv`, `p2_ablation_oof.npz`, `p2_bias_fits.csv`, `p2_ablation.log` | Step 4 |
