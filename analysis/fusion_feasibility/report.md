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
above was committed._
