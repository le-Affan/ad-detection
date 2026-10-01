# Pipeline Blueprint v2 — Multimodal AD Assessment System

Date: 2026-10-01
Status: current plan. Replaces blueprint v1 (2026-09-28) and supersedes
`docs/mri_role_refined_plan.md` wherever they differ.
Rule for every experiment: decision rules are written and committed BEFORE the
results exist. Test subjects are read once, in the frozen test evaluation.

---

## 1. System in one paragraph

A stage-aware, interpretable assessment system for ADNI-GO/2 subjects. A
clinical stream (cognition, function, demographics) and an MRI stream
(regional volumes and a CNN-derived AD probability) feed a fusion layer that
outputs a 4-stage classification (CN / EMCI / LMCI / AD), an AD-vs-rest
probability and a continuous severity index. An assessment card shows the
outputs with explanations (SHAP, Grad-CAM, KAN edge functions, a
clinical-vs-imaging discordance flag, and brain-age as supporting imaging
evidence). The test set has not been read.

---

## 2. Evidence log (what has been tested and what it showed)

All results are validation / cross-validation. The test set is untouched.

| # | Experiment | Branch | Result |
|---|---|---|---|
| 1 | Temporal model over multi-visit MRI (Path A) | `path-a-multivisit` | No temporal signal: sequence models do not beat a last-scan-only model. DX_bl temporal variant is worse than static v9 (macro-F1 0.20 vs 0.34). Tentative: latest scan helps on a dementia-at-last-visit target (0.83–0.85 vs 0.819 AUC), within noise, untested on test |
| 2 | Follow-up scans as extra training data | `multivisit-static-augmentation` | No gain (R1 vs R0 within ±0.015); with class weighting it hurts (R3 vs R2 −0.067 acc4). Model overfits to subject identity (567 unique brains) |
| 3 | v9 reproducibility, class weighting (8 seeds) | same | Class weighting: no reliable gain (+0.032 macro-F1, CI [−0.029, +0.092], 6/8 seeds). v9 expected val macro-F1 is **0.284 ± 0.047**; the logged 0.349 was an upper-tail single run |
| 4 | Cross-fitted ALZ added to clinical CatBoost | `fusion-feasibility` | B − A macro-F1 +0.000 CV [−0.009, +0.008]; B′ − A′ +0.009 [−0.003, +0.021], val −0.019. No reliable gain. ALZ AUC 0.764 (train, cross-fitted) / 0.778 (val) |
| 5 | Brain-age gap (DeepBrainNet) added | `brain-age` | E − B −0.002 [−0.011, +0.006]; E′ − B′ +0.004 [−0.007, +0.016]. No reliable gain. Strong stage gradient: corrected gap CN 0.0 → EMCI +2.2 → LMCI +4.0 → AD +6.8 years (KW p = 2×10⁻¹⁶); AD vs CN AUC 0.80 alone |
| 6 | Remove the 7 FreeSurfer volumes from the clinical features | `mri-morphometry` | A − A0 +0.002 [−0.010, +0.015]; A′ − A0′ −0.003 [−0.021, +0.015]. Volumes add nothing on CV. SynthSeg Part 2 deliberately not run |

**Reading of the evidence.** The imaging signals are valid: ALZ AUC ~0.77,
brain-age AUC 0.80 with a clean stage gradient, hippocampus ~10% smaller in
AD. They add no measurable staging accuracy once cognitive and functional
assessments are available, because the CN/EMCI/LMCI/AD labels are defined by
those assessments. Subject count (~570 train) bounds what a 3D CNN can learn
beyond that. The report states this plainly and treats it as a finding.

Clinical stream reference numbers: test accuracy 0.813, macro-F1 0.786
(untouched since). Fixed-tree retrain: val macro-F1 0.778, CV 0.752. Do not
quote the saved model's validation 0.808 (its settings were tuned on val).
Clinical confusion (val): 13 of 16 LMCI errors go to EMCI.

---

## 3. Architecture

```
 Clinical tables                         T1 MRI scan
        │                                      │
 Clinical stream [built]            ┌──────────┼───────────────┐
 cognition, function, age           ▼          ▼               ▼
 (36 features minus the       Volumes/ICV   3D CNN         Brain-age
  7 FreeSurfer columns)       [built]       ALZ score      [built]
        │                          │        (cross-fitted)   │
        │                          │           │             │ card only,
        └──────────┬───────────────┴───────────┘             │ not a model
                   ▼                                          │ input
          Top-K clinical + 7 MRI inputs                       │
                   ▼                                          │
        Fusion layer [to build]                               │
        KAN, compared with CatBoost                           │
                   │                                          │
      ┌────────────┼──────────────┐                           │
      ▼            ▼              ▼                           │
  Stage (COG)   AD vs rest    Severity index                  │
  4-class       probability   Σ k·p(stage k)                  │
      └────────────┴──────┬───────┘                           │
                          ▼                                   │
                  Assessment card ◄───────────────────────────┘
        outputs, SHAP, Grad-CAM, KAN curves, flags
```

### 3.1 Inputs
- **Clinical stream:** current 36 features minus Ventricles, Hippocampus,
  WholeBrain, Entorhinal, Fusiform, MidTemp, ICV (arm A0). Top-K by SHAP, K ∈
  {8, 10, 12}, chosen on training data only.
- **MRI stream (7 inputs):** the six FreeSurfer volumes each divided by ICV
  (Ventricles, Hippocampus, WholeBrain, Entorhinal, Fusiform, MidTemp) plus the
  cross-fitted ALZ score.
- **Not model inputs:** brain-age gap, SynthSeg volumes, Path C slopes
  (future information), DEMO (auxiliary CNN loss only).
- Moving the volumes into the MRI stream is a structural choice. Step 0 shows it
  costs no accuracy. State this when presenting it.

### 3.2 Outputs
| Output | Definition |
|---|---|
| Stage (COG) | 4-class softmax over CN / EMCI / LMCI / AD |
| AD vs rest (ADD) | Binary AD probability (ADNI has no non-AD dementias) |
| Severity index | s = Σ k·p(stage k), k = 0..3. Ordered by construction; checked for monotonic increase across stages. Spearman with CDR-SB is descriptive only, since CDR-SB is an input |

### 3.3 Card contents and rules
- SHAP (clinical inputs), Grad-CAM (CNN; wording: periventricular/ventricular
  attribution, not hippocampal), KAN edge-function plots.
- **Discordance flag**, thresholds fixed from train only: clinical prediction AD
  with ALZ below the 10th percentile of train-AD ALZ, or clinical prediction
  CN/EMCI with ALZ above the 90th percentile. Reported descriptively.
- **Brain-age on the card:** show raw and corrected gap. Mark as unreliable when
  the brain mask has no zero-intensity voxels or more than 10% saturated
  voxels (catches the GE IR-SPGR failures, e.g. 098_S_4275 with 23% saturated).
  Show 016_S_4097 (gap −25, plausible scan, inflated by the bias correction) with an
  "extreme value" note.

---

## 4. Components and status

| Component | Status | Notes |
|---|---|---|
| Clinical stream (CatBoost + SHAP) | Built | Test untouched |
| 3D CNN, ALZ | Built | v9 lr_3e-5 config, no class weighting; 5 folds × 3 seeds cross-fit; val/test ALZ = mean of the 5 fold models |
| FreeSurfer volumes (ADNI tables) | Available | Higher quality than SynthSeg on our volumes (SynthSeg cortical agreement 0.27–0.45) |
| Brain-age (DeepBrainNet) | Built, card only | VGG16 release (not the paper's Inception-ResNet-v2); cite our measured MAE 3.16 years on CN, not the paper's. Bias correction α = 0.58, fitted on CN train only (inside each CV fold) |
| KAN fusion + CatBoost comparator | **To build** | Prompts 1–2 |
| Severity index, discordance flag | **To build** | Prompt 1–2 |
| Frozen test evaluation | **To run** | Prompt 3 |
| Assessment card, demo notebook, report, slides | **To do** | After test evaluation |

---

## 5. Remaining plan

| Step | Work | Gate |
|---|---|---|
| Prompt 1 | Push `mri-morphometry`; branch `kan-fusion`; build fusion dataset; pre-register KAN rules | Pre-registration committed |
| Prompt 2 | KAN + CatBoost on input sets C (clinical only) and F (clinical + MRI); severity; discordance; edge-function plots and pruning. One day, train/val only | KAN is the final decision layer only if within 0.02 macro-F1 of CatBoost on CV; otherwise CatBoost is final and KAN is the interpretability comparison |
| Prompt 3 | Freeze models and rules, then read test once | Freeze commit precedes any test access |
| Card + demo | Assessment card, update `demo/phase1_phase2_demo.ipynb` | — |
| Report + slides | Corrected numbers, honest MRI framing, limitations | — |

Optional: regenerate Path C slopes for a longitudinal panel (the file is lost);
standalone 3-class MRI model with the Liu et al. (2020) design fixes.

---

## 6. Evaluation protocol

- **Splits:** clinical split is master: 567 train / 122 val / 123 test,
  identical across streams (verified). Subject-level; no subject crosses splits.
- **Models:** CatBoost at a fixed 265 trees (chosen on val originally, applied
  identically to every arm); KAN as specified in Prompt 2.
- **CV:** repeated stratified 5-fold over clinical-train, 5 seeds, paired
  bootstrap (2,000 resamples).
- **Imaging contribution:** F (clinical + MRI inputs) vs C (clinical only),
  reported as its own results-table row for each model, with the CI.
- **Final test, run once:** (1) clinical CatBoost C, (2) CatBoost on F, (3) the
  final decision layer, (4) ALZ alone, (5) reference R (full 36 features).
  Metrics: accuracy, macro-F1, balanced accuracy, per-class F1, AD AUC,
  confusion matrices, severity monotonicity, stage-distance error.

---

## 7. Traps and handling

| Trap | Handling |
|---|---|
| In-sample MRI scores are overfit | ALZ cross-fitted; brain-age model never saw ADNI |
| Cross-stream split mismatch | Verified identical |
| Top-K chosen on the full data | SHAP ranking inside each fold, training portion only |
| Validation used for tuning, then quoted | Quote CV and fixed-tree numbers |
| Future information | Path C slopes excluded from baseline-diagnosis inputs |
| Brain-age age bias | Correction fitted on CN train subjects only, inside each fold |
| Brain-age scan failures | Reliability rule above |
| Label-defining scores dominate | A′/B′/E′-style arms (scores removed) reported |
| Normalisation fit across CV folds | Step-4 normalisation/imputation was fit on all clinical-train rows (small, arm-symmetric); state as a limitation |
| Rules changed after results | Every rule committed before its experiment |

---

## 8. Limitations to state

- Imaging adds no measured staging accuracy; the labels are defined by cognitive
  assessments, and the cohort has ~570 training subjects.
- The clinical features contain FreeSurfer volumes (now moved to the MRI
  stream); a new patient's raw scan would need its own segmentation. SynthSeg is
  reliable for subcortical structures (hippocampus 0.79, ventricles 0.98
  agreement with FreeSurfer) and weak for cortical ones.
- Our MRI volumes saturate ~5% of brain voxels, use a template-mask skull strip,
  and three GE IR-SPGR scans have a severe saturation failure.
- Scanners are unharmonised (site 016 shows a brain-age offset).
- Brain-age model compresses towards the mean (α = 0.58); released model is
  VGG16, not the published architecture.
- One cohort; no external validation; test set of 123 subjects with few LMCI.
- Path A raw-scan coverage is 690 of 811 subjects, and the missing subjects are
  AD-heavy (55 of 121).

---

## 9. Numbers to correct in slides and report

| Item | Old | New |
|---|---|---|
| MRI v9 macro-F1 | 0.349 | 0.284 ± 0.047 (8 seeds, val) |
| Clinical val macro-F1 | 0.808 | Do not quote. Use test 0.813 / 0.786; CV 0.752 |
| Grad-CAM wording | Hippocampus / temporal lobe | Periventricular / ventricular regions |
| MRI stream role | Staging | Volumes + ALZ supply imaging evidence; staging and severity come from fusion |
| Progression modelling claim | Modelled by the pipeline | Cross-sectional; Path A tested temporal modelling and found no gain |

---

## 10. File and branch map

- Docs: `docs/pipeline_blueprint.md` (this file), `docs/mri_role_refined_plan.md`
  (partly superseded), `docs/path_a_roadmap.md`,
  `docs/path_a_stages_abc_report.md`, `docs/mv_static_report.md`
- Reports: `analysis/fusion_feasibility/report.md`,
  `analysis/brain_age/p2_report.md`, `analysis/mri_morphometry/part1_report.md`
- Branches: `path-a-multivisit`, `multivisit-static-augmentation`,
  `fusion-feasibility`, `brain-age` (pushed); `mri-morphometry` (local until
  Prompt 1); `kan-fusion` (to create). Branches only; never push to master.
