# Fusion feasibility pass — does MRI's ALZ add value to the clinical stream?

Branch: `fusion-feasibility` (off `multivisit-static-augmentation` at `ddea918`;
not pushed). Date: 2026-09-27.
Validation and cross-validation only. The test split was not read or evaluated:
only its RID list (`test_rids.csv`) was opened, for the split audit.

Plan: `docs/mri_role_refined_plan.md`, Section 7.

---

## Part 1 — Investigation

### 1. Clinical confusion matrix on validation

Saved model `models/step5_catboost.cbm` (depth 6, l2 10, lr 0.1, 265 trees),
on `step4_val.csv` (122 subjects). Nothing retrained.

| True ↓ / Pred → | CN | EMCI | LMCI | AD | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| CN (28) | 27 | 1 | 0 | 0 | 1.000 | 0.964 | 0.982 |
| EMCI (46) | 0 | 44 | 2 | 0 | 0.759 | 0.957 | 0.846 |
| LMCI (25) | 0 | **13** | 9 | **3** | 0.750 | 0.360 | 0.486 |
| AD (23) | 0 | 0 | 1 | 22 | 0.880 | 0.957 | 0.917 |

Accuracy 0.836, macro-F1 0.808, balanced accuracy 0.809, AD AUC 0.996.

- **LMCI errors go mostly to EMCI:** 13 of 16. Only 3 go to AD.
- **AD has one error, and it goes to LMCI.** There is no AD↔EMCI or AD↔CN
  confusion.
- So only about 4 of 122 subjects sit on the LMCI/AD boundary, where anatomy
  could help. The main error mass is on the LMCI/EMCI boundary, which the plan
  expects MRI to help little with.
- These validation figures are optimistic. `step5_catboost_shap.py` chose its
  hyperparameters on val accuracy and early-stopped on val.

### 2. Split audit

PTID's numeric suffix equals RID for all 814 manifest rows and all 811 MRI
index rows, and labels agree between the streams for every subject.

- **The streams' splits are identical.** `m2_index.py` inherits the split from
  `dataset/processed/*_rids.csv`. Of the 811 MRI subjects, 0 sit in a
  different split from the clinical split (567 train / 121 val / 123 test).
  Path A's 690 subjects also have 0 mismatches.
- **Clinical subjects with no MRI (3):** 126_S_2360 (train, EMCI),
  098_S_4095 (train, AD), 941_S_4187 (val, LMCI).
- **MRI subjects with no clinical record:** none. All 123 clinical-test
  subjects have a scan.
- **104 vs 121.** The 104 are the clinical-val subjects in the Path A
  follow-up download (690 of 811 subjects). The other 17 (CN 1, EMCI 8,
  LMCI 2, AD 6) were never in `MANIFEST_patha_followup.csv`. The 104 are a
  strict subset of the 121; one of them (016_S_2007) has no usable Path A
  baseline.
- **121 vs 122.** MRI val lacks 941_S_4187, the val subject without a scan.

### 3. Seed-study status

Finished (8 seeds × {R0, R2}, `docs/mv_static_report.md`).
- **Class weighting: no reliable gain.** Macro-F1 +0.032, 95% CI
  [−0.029, +0.092], R2 won on 6 of 8 seeds.
- ALZ AUC is unaffected: 0.788 ± 0.027 (R0) vs 0.787 ± 0.037 (R2).
- The CNN config for this pass is therefore v9 lr_3e-5 **without** class
  weighting.

---

## Part 2 — Pre-registration (written and committed before the ablation ran)

### Cross-fitted ALZ

- 567 clinical-train subjects with MRI, 5 folds stratified by label
  (`make_folds.py`, seed 42), baseline scan only.
- Per fold: train on the other 4 folds minus a stratified 15% inner holdout;
  early stopping and best-val-loss selection on that inner holdout only.
  The predicted fold and clinical val are never seen.
- CNN: v9 lr_3e-5 settings (resnet18w, batch 32, lr 3e-5, wd 1e-2, 60 epochs,
  patience 15, default augmentation, auto lambdas and pos_weight), no class
  weighting. Seeds 42–44 per fold.
- ALZ = sigmoid of the ALZ logit. A fold model's ALZ is the mean of its 3
  seeds' probabilities; DEMO likewise.
- **Clinical-val ALZ/DEMO:** the mean of the 5 fold models (each already a
  3-seed mean). Train-side ALZ comes from ~80%-data models, so val ALZ comes
  from the same models to keep the ALZ distribution consistent. No model is
  trained on all clinical-train for this pass.
- Subjects without MRI get NaN, never an imputed value.
- Frozen v9 embeddings are not used.

### Arms

All arms use the existing step-4 features and the step-5 hyperparameters
(depth 6, l2_leaf_reg 10, lr 0.1, MultiClass, PTGENDER categorical) with a
**fixed 265 trees and no early stopping.** 265 was originally chosen on
validation (step 5's early stopping); it is applied identically to every arm,
so it cannot favour one arm, and no stopping decision uses an evaluation set.

| Arm | Features |
|---|---|
| A | 36 clinical features (step 4, incl. CogIndex) |
| B | A + ALZ |
| C | A + ALZ + DEMO — run only if DEMO is not collapsed (rule below) |
| A′ | A minus the diagnosis-defining instruments (list below): 33 features |
| B′ | A′ + ALZ |

Arm D (Path C hippocampal slope) is **not run**: no slope file exists in the
repo or its history, so it is not easy to join.

**DEMO collapse rule (for arm C).** Cross-fitted DEMO on clinical-train counts
as *not collapsed* only if both hold: Spearman ρ(DEMO, label) ≥ 0.2, and the
per-class mean DEMO spans ≥ 0.25 (max class mean − min class mean; targets
span 0–2). Per-class spread is reported either way.

### A′ — columns removed and why

ADNI assigns CN / EMCI / LMCI / AD from MMSE, CDR (global, with CDR-SB
consistent) and Logical Memory II delayed recall (education-adjusted cutoffs;
this is what separates EMCI from LMCI).

| Removed | Why |
|---|---|
| `CDRSB` | CDR Sum of Boxes: a diagnosis-defining instrument |
| `MMSE` | Diagnosis-defining instrument |
| `CogIndex` | Derived from them: normalized MMSE − CDRSB − ADAS13 (step 4) |

Checked and absent from the feature set, so nothing to remove:
CDR global (`CDGLOBAL`), Logical Memory (`LDELTOTAL`, `LIMMTOTAL`).
`step2_manifest.json` lists the 35 source features; no other feature is
computed from CDR, MMSE or Logical Memory.

**Kept, and why A′ is still not cognition-free:** ADAS11/13, MoCA, RAVLT (4),
FAQ and the 14 ECog items are cognitive or functional scales, but not ADNI's
diagnostic criteria. They correlate strongly with the removed ones, so A′
measures MRI's value without the *label-defining* scores, not without
cognition. The FreeSurfer volumes (Ventricles, Hippocampus, WholeBrain,
Entorhinal, Fusiform, MidTemp, ICV) stay in every arm, so **no arm is
imaging-free** (see the note at the end).

### Evaluation protocol

- **Cross-validation (primary):** all 569 clinical-train subjects (the 2
  without MRI keep NaN ALZ). For each CatBoost seed s ∈ {42..46}: stratified
  5-fold CV with `random_state = s` (redrawn per seed; the same folds for every
  arm at that seed), out-of-fold predictions for all 569. Metrics on the pooled
  OOF predictions per seed; reported as mean ± SD over the 5 seeds.
- **Validation:** train on all 569 clinical-train, evaluate on the 122
  clinical-val subjects, seeds 42–46, mean ± SD.
- Every metric is also reported restricted to subjects with MRI (567 CV / 121
  val), from the same predictions.
- **Metrics.** Primary: 4-class macro-F1. Secondary: LMCI F1, AD F1, balanced
  accuracy, AD AUC (AD-vs-rest from P(AD)). Per-class F1 for all arms.

### Decision rule

Applied separately to **B vs A** and **B′ vs A′**:

- Paired bootstrap on the CV evaluation: 2,000 resamples of the 569
  clinical-train subjects (seed 0). In each resample, compute macro-F1(B) −
  macro-F1(A) on the resampled OOF predictions for each CatBoost seed and
  average over the 5 seeds. The 95% CI is the 2.5th–97.5th percentile.
- **B counts as an improvement over A only if** the 95% CI of the CV
  macro-F1 difference excludes zero **AND** the validation difference (mean
  over the 5 seeds of macro-F1(B) − macro-F1(A)) has the same sign.
- Otherwise the verdict is **"no reliable gain"**.
- A significant *negative* CV difference with a negative val difference is
  reported as "reliable loss".
- The same bootstrap on val (122 subjects) is reported descriptively; it is not
  part of the rule.

---

## Part 2 — Results

_Filled in after the runs; nothing below this line existed when the section
above was committed (31462eb)._

### Verdict

**No reliable gain from ALZ, in either comparison.**

| Pair | CV macro-F1 diff, 95% CI | Val diff | Rule |
|---|---|---|---|
| B − A (full clinical) | +0.000 [−0.009, +0.008] | +0.018 | CI includes 0 → **no reliable gain** |
| B′ − A′ (no defining scores) | +0.009 [−0.003, +0.021] | −0.019 | CI includes 0, and the signs disagree → **no reliable gain** |

Taking out the label-defining scores (A′) costs about 0.13 macro-F1, but even
then ALZ doesn't reliably recover any of it. By the plan's Section 7 decision,
this is the "no gain" branch: MRI provides independent anatomical corroboration
and an AD-specific signal, but little extra staging accuracy over cognitive
scores.

### 5. Cross-fitted ALZ — sanity check

15 CNN runs (5 folds × seeds 42–44), 19:07–19:39 locally on the RTX 5060.
Best epochs ranged from 4 to 40. The 68-subject inner holdouts give a noisy
val loss, so several runs stopped early at epoch 4–5.

| | n | ALZ AUC (AD vs rest) |
|---|---|---|
| Clinical-train, cross-fitted (pooled) | 567 | **0.764** |
| Clinical-train, per fold | 113–114 | 0.759 / 0.709 / 0.806 / 0.866 / 0.777 (mean 0.783) |
| Clinical-val, 5-fold-model ensemble | 121 | **0.778** |
| Single-seed fold models, on val | 121 | 0.631–0.800 |

- **Both are in the expected range (0.75–0.82), and the cross-fitted AUC isn't
  above the val AUC**, so there's no sign of leakage.
- The pooled train AUC is below the per-fold mean, because the 5 fold models
  are calibrated slightly differently. CatBoost sees that shift as noise.
- **The val ALZ comes from the 5-fold-model ensemble**, recorded as
  `fold = "full"` in `alz_crossfit.csv`, as decided. Averaging narrows the
  distribution: val ALZ SD is 0.173, against 0.216 for cross-fitted train (the
  means are 0.447 and 0.448). This mismatch is mild but real; it affects only
  the val evaluation, not the primary CV.
- Per class, cross-fitted train ALZ (mean ± SD): CN 0.379 ± 0.172,
  EMCI 0.386 ± 0.191, LMCI 0.490 ± 0.220, AD 0.617 ± 0.209.
  LMCI sits between EMCI and AD.
- **DEMO isn't collapsed by the pre-registered rule** (ρ = 0.345, class-mean
  span 0.343), so arm C ran. It is heavily shrunk toward the mean, though:
  CN 0.665 → AD 1.008 against targets of 0 → 2.

| DEMO (cross-fitted train) | Mean | SD | Min | Max |
|---|---|---|---|---|
| CN | 0.665 | 0.238 | 0.047 | 1.412 |
| EMCI | 0.690 | 0.259 | 0.197 | 1.587 |
| LMCI | 0.841 | 0.329 | 0.174 | 1.836 |
| AD | 1.008 | 0.355 | 0.170 | 1.799 |

Files:
- `alz_crossfit.csv`: RID, PTID, clin_split, fold, ALZ, DEMO, has_mri, label.
  591 rows (569 train + 122 val). 3 subjects without MRI have NaN.
- `alz_per_seed.csv`: every fold model × seed prediction.

### 7. Ablation

All values are mean ± SD over CatBoost seeds 42–46.

**Cross-validation, 569 clinical-train (primary)**

| Arm | Macro-F1 | LMCI F1 | AD F1 | Bal. acc | AD AUC | CN F1 | EMCI F1 |
|---|---|---|---|---|---|---|---|
| A | 0.752 ± 0.006 | 0.410 ± 0.019 | 0.865 ± 0.003 | 0.750 ± 0.004 | 0.982 ± 0.001 | 0.947 ± 0.002 | 0.786 ± 0.003 |
| B | 0.752 ± 0.006 | 0.405 ± 0.015 | 0.864 ± 0.012 | 0.749 ± 0.006 | 0.981 ± 0.001 | 0.949 ± 0.002 | 0.788 ± 0.007 |
| C | 0.749 ± 0.007 | 0.401 ± 0.020 | 0.861 ± 0.005 | 0.747 ± 0.007 | 0.981 ± 0.001 | 0.949 ± 0.002 | 0.783 ± 0.011 |
| A′ | 0.620 ± 0.006 | 0.351 ± 0.013 | 0.758 ± 0.021 | 0.621 ± 0.006 | 0.958 ± 0.001 | 0.704 ± 0.010 | 0.667 ± 0.007 |
| B′ | 0.629 ± 0.010 | 0.374 ± 0.024 | 0.778 ± 0.011 | 0.629 ± 0.009 | 0.958 ± 0.002 | 0.697 ± 0.021 | 0.666 ± 0.010 |

**Validation, train on 569 → 122 clinical-val**

| Arm | Macro-F1 | LMCI F1 | AD F1 | Bal. acc | AD AUC | CN F1 | EMCI F1 |
|---|---|---|---|---|---|---|---|
| A | 0.778 ± 0.020 | 0.395 ± 0.058 | 0.917 ± 0.014 | 0.785 ± 0.016 | 0.995 ± 0.001 | 0.982 ± 0.000 | 0.818 ± 0.021 |
| B | 0.796 ± 0.017 | 0.456 ± 0.047 | 0.921 ± 0.016 | 0.799 ± 0.012 | 0.995 ± 0.001 | 0.978 ± 0.008 | 0.828 ± 0.007 |
| C | 0.793 ± 0.018 | 0.453 ± 0.045 | 0.918 ± 0.020 | 0.798 ± 0.013 | 0.994 ± 0.001 | 0.978 ± 0.008 | 0.824 ± 0.026 |
| A′ | 0.644 ± 0.009 | 0.307 ± 0.028 | 0.778 ± 0.011 | 0.652 ± 0.012 | 0.963 ± 0.002 | 0.778 ± 0.030 | 0.712 ± 0.023 |
| B′ | 0.625 ± 0.025 | 0.295 ± 0.067 | 0.760 ± 0.022 | 0.632 ± 0.021 | 0.964 ± 0.002 | 0.758 ± 0.014 | 0.685 ± 0.020 |

**Subjects with MRI only** (567 CV / 121 val): every arm moves by ≤ 0.005
and every difference by ≤ 0.001. Only 2 CV and 1 val subjects lack a scan.
Full tables are in `ablation_tables.md`.

**Paired macro-F1 differences** (2,000 bootstrap resamples, mean over 5 seeds)

| Pair | Eval | Diff | 95% CI | Per-seed (42–46) |
|---|---|---|---|---|
| B − A | CV | +0.000 | [−0.009, +0.008] | −0.011 +0.006 −0.004 +0.002 +0.007 |
| B − A | val | +0.018 | [−0.002, +0.045] | −0.028 +0.028 +0.024 +0.025 +0.040 |
| B′ − A′ | CV | +0.009 | [−0.003, +0.021] | +0.001 +0.004 +0.029 +0.012 −0.001 |
| B′ − A′ | val | −0.019 | [−0.042, +0.001] | −0.010 +0.010 −0.071 −0.006 −0.018 |

The val bootstrap is descriptive only; it is not part of the rule.
C vs A wasn't pre-registered for the decision. C is at or below B on every
metric, in both CV and val, so DEMO adds nothing beyond ALZ.

**Reading the numbers**

- **With the full clinical set, ALZ changes nothing.** The CV difference is
  exactly zero, with a CI of ±0.009, even though CatBoost does use ALZ
  (see SHAP). The val +0.018 comes mostly from LMCI F1 (+0.06, about 1.4
  more LMCI subjects correct on average). But it has the size and spread
  expected of a 25-LMCI val set, and the 4.7× larger CV shows no trace of it.
- **Without the defining scores, ALZ nudges the anatomical classes in CV:**
  LMCI F1 +0.023 and AD F1 +0.020. The LMCI↔AD errors drop a little: 23.8 →
  21.8 LMCI→AD and 19.8 → 17.2 AD→LMCI. This is the direction the plan
  predicted. But the macro-F1 CI still includes zero, and val goes the other
  way, so it doesn't pass.
- **ALZ adds nothing to AD AUC** (0.981–0.995 with clinical features). The
  clinical features already separate AD almost perfectly.
- **The saved step-5 model's val macro-F1 (0.808) is optimistic.** Arm A with
  a fixed 265 trees and no val early stopping gives 0.778 ± 0.020 on val and
  0.752 in CV.

### Confusion matrices on validation (mean over 5 seeds; rows true, cols predicted)

| A | CN | EMCI | LMCI | AD | | B | CN | EMCI | LMCI | AD |
|---|---|---|---|---|---|---|---|---|---|---|
| CN | 27.0 | 1.0 | 0 | 0 | | CN | 27.0 | 1.0 | 0 | 0 |
| EMCI | 0 | 42.8 | 3.2 | 0 | | EMCI | 0.2 | 42.8 | 3.0 | 0 |
| LMCI | 0 | 14.8 | 7.2 | 3.0 | | LMCI | 0 | 13.6 | 8.6 | 2.8 |
| AD | 0 | 0 | 1.0 | 22.0 | | AD | 0 | 0 | 1.0 | 22.0 |

| A′ | CN | EMCI | LMCI | AD | | B′ | CN | EMCI | LMCI | AD |
|---|---|---|---|---|---|---|---|---|---|---|
| CN | 22.4 | 5.6 | 0 | 0 | | CN | 22.2 | 5.8 | 0 | 0 |
| EMCI | 6.2 | 35.8 | 4.0 | 0 | | EMCI | 7.4 | 34.0 | 4.6 | 0 |
| LMCI | 1.0 | 12.4 | 6.0 | 5.6 | | LMCI | 1.0 | 12.6 | 6.0 | 5.4 |
| AD | 0 | 0.8 | 4.0 | 18.2 | | AD | 0 | 0.8 | 4.8 | 17.4 |

**Did ALZ move LMCI/AD errors on val? Barely.**
- A → B: LMCI→AD 3.0 → 2.8 and AD→LMCI unchanged at 1.0. LMCI→EMCI falls from
  14.8 to 13.6, which is the boundary MRI isn't expected to help with, and it
  is within seed noise.
- A′ → B′: LMCI→AD 5.6 → 5.4, AD→LMCI 4.0 → 4.8, EMCI→CN 6.2 → 7.4, so the
  val set got slightly worse.
- CV confusion matrices for all arms are in `ablation_tables.md`.

### SHAP importance of ALZ

Mean |SHAP| is taken over subjects and classes, for the models trained on all
569 subjects. Values are mean over seeds, with the per-seed ranks in brackets.

| Arm | Scored on | ALZ rank | ALZ mean abs SHAP | Top 5 (seed 42) |
|---|---|---|---|---|
| B | val | 15.0 (9, 19, 16, 14, 17) of 37 | 0.049 ± 0.010 | CDRSB, CogIndex, MMSE, RAVLT_perc_forgetting, ADAS13 |
| B | train (in-sample) | 13.8 of 37 | 0.052 ± 0.010 | same |
| C | val | 17.4 of 38 | 0.041 ± 0.003 | CDRSB, CogIndex, ADAS13, MMSE, RAVLT_perc_forgetting |
| B′ | val | 21.0 (21, 22, 21, 21, 20) of 34 | 0.049 ± 0.004 | ADAS11, EcogSPMem, ADAS13, FAQ, MOCA |
| B′ | train (in-sample) | 20.0 of 34 | 0.053 ± 0.004 | same |

- ALZ is a mid-ranked feature. CatBoost splits on it, but the splits don't
  change enough predictions to move macro-F1.
- In B′, removing the defining scores does not promote ALZ. The model uses
  ADAS, ECog, FAQ and MoCA instead, and these still carry most of the label
  information.

### Imaging is already in every arm

The clinical feature set contains 7 FreeSurfer volumes: Ventricles,
Hippocampus, WholeBrain, Entorhinal, Fusiform, MidTemp and ICV. Fusiform is
SHAP rank #9 in the step-5 model. **Arm A is therefore not imaging-free**,
and this ablation measures what a whole-brain CNN's AD probability adds on
top of regional volumes plus cognition. It does not measure what MRI adds on
top of cognition alone. The added value of the CNN is smaller for this
reason, and it also conflicts with the rule that clinical explanations never
mention MRI (plan §6.4).

### Caveats

- **Arm D was not run:** no Path C slope file exists.
- **Val ALZ distribution:** the 5-model ensemble is ~20% narrower than the
  cross-fitted train ALZ (SD 0.173 vs 0.216), which slightly handicaps
  B/B′/C on val only.
- **The fixed 265 trees** were originally selected on validation by step 5.
  They are applied identically to all arms.
- **Step-4 preprocessing** (median imputation, z-scoring) was fit on all 569
  clinical-train subjects, so the CV folds share it. Trees ignore z-scoring,
  at most 5.5% of any feature was imputed, and this affects all arms equally.
- **Seed variance in the CNN:** single fold models score 0.63–0.80 AUC on
  val. The 3-seed averaging was chosen to reduce this; a better ALZ (e.g.
  more seeds) is unlikely to change a CV difference of exactly 0.000.

### Reproduce

```
.venv-mri/bin/python    analysis/fusion_feasibility/make_folds.py
                        analysis/fusion_feasibility/run_folds.sh      # 15 runs, ~32 min
.venv-mri/bin/python    analysis/fusion_feasibility/predict_alz.py
.venv-fusion/bin/python analysis/fusion_feasibility/ablation.py       # ~3 min
```

`.venv-fusion` (outside the repo, `LYP/.venv-fusion`) is Python 3.12 with
catboost 1.2.10, shap 0.52.0, pandas 3.0.5 and scikit-learn 1.8.0. Weights
under `runs/*/` aren't tracked.
