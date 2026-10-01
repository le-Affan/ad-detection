# KAN fusion — pre-registration

Branch `kan-fusion` (off `mri-morphometry` at `ad81ecc`). Date 2026-10-01.
Plan: `docs/pipeline_blueprint.md` v2, sections 3 and 6.

**Written and committed before any model result exists.** No CatBoost or KAN model
has been fitted on the fusion dataset. Train and val only. No test subject's scan
or features is read; the test set is evaluated once, in Prompt 3, with the frozen
pipeline.

What exists at commit time:
- the fusion dataset (`fusion_trainval.csv`, `build_fusion_dataset.py`);
- the discordance thresholds (item g), computed from train ALZ alone;
- the brain-age reliability counts (item h). These are a scan-quality
  description, not a model result.

The KAN architecture and training are specified in Prompt 2. They go into an
addendum to this file, committed before any KAN result exists. The addendum may
add KAN details but may not change items a–h.

---

## 0. Data

`analysis/kan_fusion/fusion_trainval.csv`: 569 clinical-train and 122
clinical-val subjects (the clinical split is the master split).

| Split | CN | EMCI | LMCI | AD | Total | With a scan (ALZ) | FreeSurfer missing |
|---|---|---|---|---|---|---|---|
| train | 133 | 217 | 114 | 105 | 569 | 567 | 1 |
| val | 28 | 46 | 25 | 23 | 122 | 121 | 0 |

- **Clinical stream:** 29 features, which is arm A0 of the morphometry gate (the
  36 step-4 features minus Ventricles, Hippocampus, WholeBrain, Entorhinal,
  Fusiform, MidTemp and ICV). These are taken unchanged from `step4_{train,val}.csv`:
  median/mode imputation and z-scoring fitted on clinical-train.
- **MRI stream, 7 inputs:**
  - `Ventricles_ICV`, `Hippocampus_ICV`, `WholeBrain_ICV`, `Entorhinal_ICV`,
    `Fusiform_ICV`, `MidTemp_ICV`: raw ADNI FreeSurfer volume divided by FreeSurfer
    ICV. A missing ratio gets the clinical-train median ratio (the step-4 rule),
    then each ratio is z-scored with the clinical-train mean and SD. Values are in
    `fusion_manifest.json`. Nothing is fitted on val.
  - `ALZ`: cross-fitted. Train ALZ is out-of-fold; val ALZ is the mean of the 5
    fold models (`analysis/fusion_feasibility/alz_crossfit.csv`). It is NaN for
    the 3 subjects without a scan (126_S_2360 train EMCI, 098_S_4095 train AD,
    941_S_4187 val LMCI) and is never imputed in the dataset.
- **Only missing FreeSurfer row:** 098_S_4095 (train AD). It is also one of the
  scanless subjects, so this subject has no imaging evidence at all; its six
  ratios are train medians.
- **Not model inputs:** brain-age gap and SynthSeg volumes (card only).

## a. Input sets

| Set | Inputs | Role |
|---|---|---|
| **C** | clinical top-K (from the 29) | clinical only |
| **F** | the same top-K + the 7 MRI inputs | the proposed system |
| **R** | the full 36 step-4 features, FreeSurfer included (= arm A) | reference |

- C and F use the same clinical columns within every fold, so the C-vs-F
  difference is the 7 MRI inputs alone.
- R is a CatBoost reference. KAN on R may be reported descriptively; no rule
  depends on it.
- ALZ alone (AD-vs-rest AUC) is reported descriptively.

## b. Top-K

1. For each seed s ∈ {42, …, 46} and each fold of `StratifiedKFold(5,
   shuffle=True, random_state=s)` over the 569 train subjects (the same folds as
   the fusion-feasibility harness):
   - Fit CatBoost (settings in c) on that fold's **training portion only**, with
     all 29 clinical features.
   - Compute SHAP (TreeExplainer) on the same training-portion rows.
   - Rank features by mean |SHAP| over subjects and the 4 classes. Ties go to the
     earlier column in `fusion_trainval.csv`.
   - The held-out fold never enters the ranking.
2. For each K ∈ {8, 10, 12}, take that fold's top-K list. Fit CatBoost **C** on
   the training portion and predict the held-out fold.
3. **K\*** = the K with the highest CV macro-F1 of CatBoost C: pooled
   out-of-fold predictions per seed, then the mean over the 5 seeds. An exact tie
   goes to the smaller K.
   - K is chosen on CatBoost C only (not on F, not on KAN), so the choice cannot
     favour MRI or KAN.
   - K\* is then used for C and F in both models.
4. **Final feature list** (for val, and frozen for test): fit CatBoost on all 569
   train subjects with the 29 features, once per seed. Rank by mean |SHAP| on the
   train rows, averaged over the 5 seeds, and take the top K\*.
5. Record K\*, the final list and how often each feature entered a fold-level
   top-K\* (out of 25 fold-seeds).

## c. Models

- **CatBoost:**
  - harness settings: MultiClass, depth 6, l2_leaf_reg 10, learning rate 0.1;
  - a fixed 265 trees, no early stopping;
  - PTGENDER categorical;
  - ALZ NaN handled natively.
- **KAN:** specified in Prompt 2 (addendum), under these constraints:
  - same rows, folds, seeds (42–46) and input columns as CatBoost;
  - any scaling, encoding (PTGENDER) or NaN handling (ALZ for 3 subjects) is
    fitted on the training portion only (fold-training for CV, all 569 train for
    val);
  - val is never used for early stopping, model selection or tuning. Any tuning
    uses inner CV on the training portion.
- **Evaluation:** every model × input set is fitted per seed:
  - CV: out-of-fold predictions for all 569 train subjects;
  - val: train on all 569, predict the 122.

## d. Metrics

All metrics are reported per model × input set × {CV, val}, mean ± SD over the 5
seeds:
- 4-class macro-F1 (**primary**);
- accuracy and balanced accuracy;
- per-class F1, with **LMCI and AD highlighted**;
- AD AUC (AD vs rest from P(AD));
- confusion matrices (mean over seeds);
- stage-distance error (mean |predicted − true stage|);
- the severity metrics in f.

The same metrics restricted to subjects with a scan (567 CV / 121 val) are
reported alongside.

## e. Decision rules

**Bootstrap machinery:** the harness's `macro_f1_boot`, with 2,000 resamples of
the 569 train subjects (rng seed 0). In each resample, compute the macro-F1
difference per seed on the resampled out-of-fold predictions, then average over
the 5 seeds. The 95% CI is the 2.5th–97.5th percentile. The same resample indices
are used for every comparison.

**e1. MRI contribution (F vs C), for CatBoost and for KAN separately.**
- **Reliable gain:** the CV 95% CI of macro-F1(F) − macro-F1(C) lies entirely
  above 0 AND the val difference (mean over seeds) is > 0.
- **Reliable loss:** the CI lies entirely below 0 AND the val difference is < 0.
- **No reliable difference:** anything else.
- Reported whatever the outcome, as the row "MRI contribution" in the final
  table, with the CI.

**e2. KAN vs CatBoost (final decision layer), on input set F.**
- **KAN is the final decision layer if it is non-inferior to CatBoost on CV:**
  the lower bound of the 95% CI of CV macro-F1(KAN, F) − macro-F1(CatBoost, F) is
  above −0.02.
- Otherwise CatBoost is the final decision layer, and KAN is reported as the
  interpretability comparison.
- The point estimate, the val difference and the same comparison on C are
  reported descriptively. Val is not part of this rule.

**e3. Descriptive only:** C − R and F − R (CatBoost), with the same CIs. These
show what moving the FreeSurfer volumes into the MRI stream and cutting the
clinical features to K\* costs or gains.

## f. Severity index

- s = Σ_k k · p(stage k), with k = 0, 1, 2, 3 for CN, EMCI, LMCI, AD. Computed
  from each model's probabilities.
- **Primary object:** the final decision layer on F. Other models are reported
  alongside.
- **Monotonicity check (val):** on seed-averaged probabilities, the mean s must
  be strictly increasing CN < EMCI < LMCI < AD. Reported as pass/fail, plus the
  number of seeds (of 5) that pass individually.
- **Spearman ρ(s, true stage):** val (primary) and CV out-of-fold.
- **Spearman ρ(s, CDRSB):** descriptive only. CDRSB is a model input and helps
  define the labels, so this is **not independent validation**, and the report
  says so.

## g. Discordance flag (thresholds from train only)

**Thresholds,** fixed now from the cross-fitted ALZ of the 104 train AD subjects
with a scan (`numpy.percentile`, linear):
- P10 = **0.2981**
- P90 = **0.8751**

**Clinical-only prediction:** argmax of the seed-averaged probabilities of
**CatBoost C** at K\*. For val, the models are trained on all 569; for train, the
CV out-of-fold predictions are used. CatBoost C is used whichever model becomes
the final decision layer, so the flag does not depend on e2.

**Flag a subject when either holds:**
1. C predicts AD and ALZ < 0.2981;
2. C predicts CN or EMCI and ALZ > 0.8751.

Subjects without ALZ are "not evaluable".

**Report** (val primary; train out-of-fold alongside):
- flag rate per true class and per predicted class;
- the error rate of flagged vs unflagged subjects, for CatBoost C and for the
  final decision layer on F.

This is descriptive only, not a performance claim.

**Known property of the thresholds** (from the ALZ distribution, before any
model): about 1% of train CN/EMCI and 0% of val CN/EMCI have ALZ > 0.8751. The
second branch is therefore expected to fire rarely; most flags will come from
branch 1.

## h. Brain-age reliability rule (card)

**Rule:** a scan's brain age is **unreliable** if, inside the ICBM152 template
brain mask, the m1 `mni/` volume has **no zero-intensity voxels**, or **more than
10% saturated voxels** (intensity ≥ 0.999).

Computed for all 688 train/val scans by `brainage_reliability.py`; output in
`brainage_reliability.csv`.

| Split | Scans | No-zero clause | > 10% saturated | Unreliable |
|---|---|---|---|---|
| train | 567 | 0 | 3 | **3** |
| val | 121 | 0 | 0 | **0** |

| Flagged scan | Split | Diagnosis | Saturated | Zero voxels in mask |
|---|---|---|---|---|
| 016_S_4121 | train | CN | 23.2% | 2 (of 1,572,286) |
| 098_S_4275 | train | CN | 23.3% | 1 |
| 094_S_2216 | train | EMCI | 23.5% | 1 |

- **The other 685 scans are 2.8–5.1% saturated** (median 5.0%), so the
  threshold sits in a wide empty gap between 5.1% and 23.2%. The three flagged
  scans are all GE Sag IR-SPGR. Every unflagged scan has at least 4.7% of its
  mask at zero.
- **Read literally, the no-zero clause never fires at full resolution:** each
  failed scan keeps 1–2 zero voxels in the mask. The m1 `mask_fill` = 1.000 seen
  in Part 1 was measured on the 96³ downsample, where those voxels average away.
  The saturation clause catches the same three scans on its own. The rule is kept
  as written; the clause is redundant on this cohort.
- The card also shows 016_S_4097 (corrected gap −25.0, saturation 4.4%, not
  flagged) with an "extreme value" note, per the blueprint.

## Limitations stated in advance

- **Shared preprocessing across folds:** the step-4 imputation and z-scoring and
  the ratio imputation and z-scoring are fitted on all 569 train subjects, so CV
  folds share them. This is arm-symmetric and affects at most 5.5% of any
  clinical feature and 1 subject's ratios.
- **K\* is chosen on the same CV** that evaluates C and F. The optimism applies
  equally to both, so e1 is unaffected; absolute C/F numbers are slightly
  optimistic.
- **Val is mildly optimistic for every CatBoost model:** the 265 trees were
  originally chosen on val in step 5.
- **The val ALZ ensemble is narrower** than the cross-fitted train ALZ (SD 0.173
  vs 0.216; fusion-feasibility report). This handicaps F on val only.
- **Val is small:** 25 LMCI and 23 AD subjects, so val differences of ±0.02
  macro-F1 are within seed spread.
