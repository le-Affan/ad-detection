# Step 0 gate — pre-registration

Branch `mri-morphometry` (off `brain-age` at `80d5af2`). Date 2026-09-30.
Written and committed **before** `step0_gate.py` was run. Train and val only;
no test subject's data is read.

## Question

Do ADNI's FreeSurfer volumes, which sit inside the "clinical" feature set,
add staging accuracy? If they do, morphometry has a real role, and the MRI
stream should supply it from our own scans (Steps 4–10). If they don't, the
morphometry plan is unlikely to add accuracy either.

## Step 1 — MRI-derived columns in the clinical feature set

Checked against `dataset/processed/step4_train.csv` (36 features) and
`step2_manifest.json` (35 source features + CogIndex from step 4).

| Column | What it is |
|---|---|
| `Ventricles` | FreeSurfer ventricular volume (ADNIMERGE) |
| `Hippocampus` | FreeSurfer hippocampal volume |
| `WholeBrain` | FreeSurfer whole-brain volume |
| `Entorhinal` | FreeSurfer entorhinal cortex volume |
| `Fusiform` | FreeSurfer fusiform gyrus volume |
| `MidTemp` | FreeSurfer middle temporal gyrus volume |
| `ICV` | FreeSurfer intracranial volume |

These 7 are the "FreeSurfer columns". No other feature is MRI-derived:
the rest are demographics (AGE, PTGENDER, PTEDUCAT), APOE4 (genetic),
cognitive/functional scales (CDRSB, ADAS11/13, MMSE, RAVLT ×4, FAQ, MOCA,
ECog ×14) and CogIndex (normalised MMSE − CDRSB − ADAS13). No PET or CSF
column is in the feature set.

SHAP ranks are produced by the Step 0 run itself (arms A and A′, full-train
models, all 5 seeds, scored on val and in-sample on train). The existing
`models/step5_shap_values.npz` is **not** used, because it was computed on the
test split.

## Arms

Same harness (`analysis/fusion_feasibility/ablation.py`), unchanged: CatBoost
MultiClass, depth 6, l2_leaf_reg 10, lr 0.1, a fixed 265 trees, no early
stopping, PTGENDER categorical, seeds 42–46, stratified 5-fold CV over the
569 clinical-train subjects with `random_state = seed` (same folds for every
arm at a seed), and train-all-569 → 122 clinical-val.

| Arm | Features | n |
|---|---|---|
| A | clinical (current 36) | 36 |
| A0 | A minus the 7 FreeSurfer columns | 29 |
| A′ | A minus CDRSB, MMSE, CogIndex (existing diagnosis-defining list) | 33 |
| A0′ | A′ minus the 7 FreeSurfer columns | 26 |

No ALZ, DEMO or brain-age in any arm.

## Decision rule

**Morphometry "matters" if A beats A0 OR A′ beats A0′.**

X beats Y when both hold:
1. the paired-bootstrap 95% CI of the CV macro-F1 difference X − Y
   (2,000 resamples of subjects, rng seed 0, mean over the 5 seeds, as in the
   harness) lies entirely above zero; and
2. the val macro-F1 difference X − Y (mean over seeds) is also above zero.

A CI entirely below zero (FreeSurfer hurts) is reported but is not "beats".

## Also reported (descriptive)

Per-class F1 (all four, with LMCI and AD highlighted), balanced accuracy,
AD AUC (AD-vs-rest from P(AD)), accuracy, for every arm, CV and val,
mean ± SD over seeds; paired differences for these; SHAP ranks of the 7
FreeSurfer columns in A and A′.

## Note

Val figures stay optimistic for every arm: step 5 chose the 265 trees on val.
This applies equally to all arms.
