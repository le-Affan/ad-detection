# P2 pre-registration — brain-age gap as a fusion input

Branch `brain-age`. Written and committed **before** any bias correction,
sanity check or ablation result existed. At commit time, only the raw
DeepBrainNet predictions were being computed (Step 1); none had been
summarised by diagnosis. Test subjects are never loaded (P5).

## Inputs

- **Brain age:** `analysis/brain_age/brainage_trainval.csv`.
  - DeepBrainNet `DBN_model.h5` (VGG16 release, sha256 `9257e98e…41ea`),
    pipeline as in the smoke test.
  - Chronological age at scan = ADSL `AGE` (age at enrolment) + (manifest
    `MRI_EXAMDATE` − ADSL `ENRLDT`)/365.25.
  - Missing (NaN, never imputed): the 3 subjects without a scan, and
    027_S_4966 (failed registration).
- **ALZ:** `analysis/fusion_feasibility/alz_crossfit.csv`, unchanged.
- **Clinical features:** `step4_{train,val}.csv`, unchanged. A′ removes
  CDRSB, MMSE, CogIndex (the fusion-feasibility list).

## a. Bias correction

- Fit `predicted_age = α·age + β` by ordinary least squares on **CN subjects
  only** (with brain age).
- Corrected brain age = (predicted − β)/α. **Corrected gap (BAG)** = corrected
  brain age − age.
- **CV:** in each fold, α and β are fitted on that fold's **training** CN
  subjects only, then applied to both the fold's training and evaluation
  subjects. Evaluation-fold labels are never used.
- **Validation:** α and β are fitted on all clinical-train CN subjects, then
  applied to train and val.
- **Sanity checks (section e):** one fit on all clinical-train CN subjects.
  Label-based checks use train subjects only.
- All α, β values are saved.

## b. Ablation arms

CatBoost exactly as in fusion-feasibility (`analysis/fusion_feasibility/ablation.py`
helpers):
- depth 6, l2 10, lr 0.1, a **fixed 265 trees**, PTGENDER categorical;
- seeds 42–46;
- CV: `StratifiedKFold(5, shuffle=True, random_state=seed)` over the 569
  clinical-train subjects, OOF predictions;
- val: train on 569, evaluate on 122.

| Arm | Features |
|---|---|
| A | clinical (36) |
| B | A + ALZ |
| **E** | A + ALZ + BAG |
| A′ | A − {CDRSB, MMSE, CogIndex} (33) |
| B′ | A′ + ALZ |
| **E′** | A′ + ALZ + BAG |
| G (secondary) | A + BAG |

**Harness check:** A, B, A′ and B′ use the same data, folds and seeds as
fusion-feasibility, so they must reproduce its per-seed metrics exactly. Any
mismatch is reported, and the run is investigated before any verdict.

## c. Decision rule

**The brain-age gap is adopted as a fusion input only if E beats B, OR E′
beats B′.** "Beats" means both of:

1. **CV:** the paired bootstrap 95% CI of the macro-F1 difference excludes
   zero. Method: 2,000 resamples of the 569 clinical-train subjects (seed 0);
   in each resample, compute macro-F1 on the OOF predictions for each seed
   and average the difference over the 5 seeds; take the 2.5th–97.5th
   percentiles.
2. **Val:** the difference (mean over the 5 seeds) has the same sign.

Otherwise: **card only.** The gap is displayed on the assessment card, not
used as a model input.

G − A and all val bootstraps are reported descriptively. They are not part of
the rule.

## d. Secondary metrics (reported, not decisive)

- Per-class F1 (LMCI especially), balanced accuracy, AD AUC.
- Confusion matrices: B vs E and B′ vs E′ on val.
- SHAP rank and mean |SHAP| of BAG in E, E′ and G.
- Everything on all subjects and on the subjects with brain age.

## e. Sanity checks (train subjects for anything label-based)

1. **CN train:** MAE and r(predicted, age), raw and bias-corrected; α and β.
2. **Mean BAG by diagnosis on train,** with a Kruskal–Wallis test. Expected:
   CN ≈ 0 by construction, rising towards AD.
3. **r(BAG, age) on CN train.** Should be ≈ 0.
4. **Site effect.** Site = PTID prefix. Mean BAG per site among CN train
   subjects. **Flag** any site with ≥ 5 CN subjects whose mean is more than 2
   standard errors from the overall CN mean (SE = SD of CN BAG / √n_site).
   Sites more than 2 SD of individual CN gaps away are also listed.
5. **Overlap with existing inputs (train):** Spearman r of BAG with ALZ and
   with each FreeSurfer volume in the clinical features (Ventricles,
   Hippocampus, WholeBrain, Entorhinal, Fusiform, MidTemp, ICV).
6. **Standalone signal, CV on train:** use the seed-42 folds with per-fold
   bias correction, taking BAG as the score directly. Report OOF AUC for AD
   vs CN and AD vs rest.

Val is used only for the ablation's val arm and for coverage counts.
