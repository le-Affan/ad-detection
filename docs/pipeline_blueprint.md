# Pipeline Blueprint v3 — Multimodal AD Assessment System

Date: 2026-10-01
Status: current. v3 records the built and frozen system (`fusion-v3`) and its single test run.
Replaces v2 (same date) and v1 (2026-09-28); supersedes
`docs/mri_role_refined_plan.md` wherever they differ.
Rule for every experiment: decision rules are written and committed BEFORE the
results exist. Test subjects are read once, in the frozen test evaluation.

---

## 1. System in one paragraph

A stage-aware, interpretable assessment system for ADNI-GO/2 subjects. A
clinical specialist (CatBoost on cognition, function, demographics) and an MRI
specialist (logistic model on regional volumes/ICV, a CNN-derived AD
probability, brain-age gap, age and sex) each output 4-class probabilities; a
KAN fusion layer combines these stream-level scores and outputs a 4-stage classification (CN / EMCI / LMCI / AD), an AD-vs-rest
probability and a continuous severity index. An assessment card shows the
outputs with explanations (SHAP, Grad-CAM, KAN edge functions, a
clinical-vs-imaging discordance flag, and brain-age as supporting imaging
evidence). The frozen system was evaluated on test once (`fusion-v3`, freeze
commit `8f351b6`).

---

## 2. Evidence log (what has been tested and what it showed)

Rows 1–7 are validation / cross-validation. Row 8 includes the single frozen test run.

| # | Experiment | Branch | Result |
|---|---|---|---|
| 1 | Temporal model over multi-visit MRI (Path A) | `path-a-multivisit` | No temporal signal: sequence models do not beat a last-scan-only model. DX_bl temporal variant is worse than static v9 (macro-F1 0.20 vs 0.34). Tentative: latest scan helps on a dementia-at-last-visit target (0.83–0.85 vs 0.819 AUC), within noise, untested on test |
| 2 | Follow-up scans as extra training data | `multivisit-static-augmentation` | No gain (R1 vs R0 within ±0.015); with class weighting it hurts (R3 vs R2 −0.067 acc4). Model overfits to subject identity (567 unique brains) |
| 3 | v9 reproducibility, class weighting (8 seeds) | same | Class weighting: no reliable gain (+0.032 macro-F1, CI [−0.029, +0.092], 6/8 seeds). v9 expected val macro-F1 is **0.284 ± 0.047**; the logged 0.349 was an upper-tail single run |
| 4 | Cross-fitted ALZ added to clinical CatBoost | `fusion-feasibility` | B − A macro-F1 +0.000 CV [−0.009, +0.008]; B′ − A′ +0.009 [−0.003, +0.021], val −0.019. No reliable gain. ALZ AUC 0.764 (train, cross-fitted) / 0.778 (val) |
| 5 | Brain-age gap (DeepBrainNet) added | `brain-age` | E − B −0.002 [−0.011, +0.006]; E′ − B′ +0.004 [−0.007, +0.016]. No reliable gain. Strong stage gradient: corrected gap CN 0.0 → EMCI +2.2 → LMCI +4.0 → AD +6.8 years (KW p = 2×10⁻¹⁶); AD vs CN AUC 0.80 alone |
| 6 | Remove the 7 FreeSurfer volumes from the clinical features | `mri-morphometry` | A − A0 +0.002 [−0.010, +0.015]; A′ − A0′ −0.003 [−0.021, +0.015]. Volumes add nothing on CV. SynthSeg Part 2 deliberately not run |
| 7 | KAN vs CatBoost on raw clinical top-12 (C) and + 7 MRI inputs (F) | `kan-fusion` | CatBoost F − C +0.014 CV [−0.001, +0.029]: no reliable MRI gain. KAN on raw features −0.058 vs CatBoost on F (not non-inferior); MRI inputs hurt the raw-feature KAN |
| 8 | Stream specialists + KAN fusion on stream scores (35 configurations, nested stacking), frozen and tested once | `fusion-v3` | Final KAN K2 CV 0.756 vs clinical specialist 0.744 (+0.013 [−0.006, +0.034]) and CatBoost F 0.758 (−0.002). MRI share inside the fusion +0.007 [−0.006, +0.019]. **Test:** KAN 0.764, clinical 0.771, CatBoost F 0.746, CatBoost R 0.784; all differences within ±0.08 CIs. MRI specialist alone: AD-vs-rest AUC 0.880 CV / 0.873 test; 4-class macro-F1 0.42 / 0.44 (majority 0.14) |

Post-hoc, not in the freeze (train CV / val only): MRI specialist input-set controls
(`analysis/fusion_v3/mri_controls.md`). Age + sex alone give AD-vs-rest AUC 0.574; volumes/ICV alone 0.867; the
frozen specialist 0.880 (M4 − M0 +0.306 [+0.239, +0.370]; M4 − M2 +0.009 [−0.003, +0.020] CV). AD detection comes
from the regional volumes, not from age.

**Reading of the evidence.** The imaging signals are valid: ALZ AUC ~0.77,
brain-age AUC 0.80 with a clean stage gradient, hippocampus ~10% smaller in
AD. They add no measurable staging accuracy once cognitive and functional
assessments are available, because the CN/EMCI/LMCI/AD labels are defined by
those assessments. Subject count (~570 train) bounds what a 3D CNN can learn
beyond that. The report states this plainly and treats it as a finding.

Clinical stream reference numbers: original step-5 model test accuracy 0.813,
macro-F1 0.786; CatBoost R refitted on 691 (fusion-v3 test run) 0.813 / 0.784.
Fixed-tree retrain: val macro-F1 0.778, CV 0.752. Do not
quote the saved model's validation 0.808 (its settings were tuned on val).
Clinical confusion (val): 13 of 16 LMCI errors go to EMCI.

---

## 3. Architecture

```
 Clinical tables                         T1 MRI scan
        │                                      │
        │                       ┌──────────────┼──────────────┐
        │                       ▼              ▼              ▼
        │                 Volumes/ICV     3D CNN ALZ      Brain-age gap
        │                 (FreeSurfer)   (cross-fitted)   (DeepBrainNet,
        │                       │              │           reliability rule)
        ▼                       └──────┬───────┴──────────────┘
 Clinical specialist                   ▼            (+ age, sex)
 CatBoost, top-12 of 29        MRI specialist
 → p_clin (4)                  multinomial logistic → p_mri (4), P(AD)
        │                              │
        └──────────────┬───────────────┘
                       ▼
     KAN fusion K2 (residual): logit = log p_clin + KAN(clin logits, MRI logits,
     has_mri, age, CDRSB, CogIndex, MMSE); modality dropout 15%
                       │
      ┌────────────────┼──────────────────┐
      ▼                ▼                  ▼
  Stage (COG)     AD vs rest (ADD)   Severity s = Σ k·p(stage k)
      └────────────────┴────────┬─────────┘
                                ▼
                 Assessment card: outputs, SHAP, Grad-CAM, KAN edge curves,
                 MRI specialist standalone, discordance flag, brain age
```

### 3.1 Inputs (as frozen in `analysis/fusion_v3/freeze.md`)
- **Clinical specialist:** the 29 arm-A0 clinical features (36 minus the 7
  FreeSurfer columns); top 12 by SHAP inside every fit (K = 12 chosen on train
  CV); CatBoost, harness settings, 265 trees.
- **MRI specialist:** six FreeSurfer volumes / ICV, cross-fitted ALZ, corrected
  brain-age gap (NaN if the reliability rule fails), age, sex; multinomial
  logistic regression (C = 0.1), chosen from 6 candidates by train-CV log-loss.
- **KAN fusion:** stream-level scores only (centred log-probabilities of both
  specialists), the has_mri flag, plus age and the clinical top-3 (CDRSB,
  CogIndex, MMSE). Stream scores are out-of-fold (nested inside CV).
- **Not used:** SynthSeg volumes, Path C slopes (future information), DEMO.
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
- **KAN edges on the card:** only MRI edges consistent in all 25 val-stage and all 25 frozen KANs (MRI CN → AD,
  MRI EMCI → LMCI). Grad-CAM wording: the map is too coarse to localise a structure; cohort-level attribution is
  periventricular/ventricular.
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
| Clinical specialist (CatBoost + SHAP) | Built, frozen | Test macro-F1 0.771 (5-seed average, refitted on 691) |
| 3D CNN, ALZ | Built | v9 lr_3e-5 config, no class weighting; 5 folds × 3 seeds cross-fit; val/test ALZ = mean of the 5 fold models |
| FreeSurfer volumes (ADNI tables) | Available | Higher quality than SynthSeg on our volumes (SynthSeg cortical agreement 0.27–0.45) |
| Brain-age (DeepBrainNet) | Built, card only | VGG16 release (not the paper's Inception-ResNet-v2); cite our measured MAE 3.16 years on CN, not the paper's. Bias correction α = 0.58, fitted on CN train only (inside each CV fold) |
| MRI specialist | Built, frozen | Logistic, AD-vs-rest AUC 0.873 on test; weak at staging (4-class macro-F1 0.44) |
| KAN fusion (K2) + CatBoost comparators | Built, frozen (`fusion-v3`) | Matches CatBoost F / clinical specialist within CIs; does not beat them |
| Severity index, discordance flag | Built | Severity monotonic on CV, val and test; flag thresholds from train (kan-fusion) |
| Frozen test evaluation | **Done once** | Freeze `8f351b6`, test outputs `adc4ccb`; no re-runs |
| Assessment card + demo notebook | **Built** | `src/card/`, `demo/assessment_demo.ipynb` (HTML + PNG, any of 814 subjects, precomputed outputs only); see `demo/README.md` |
| Report, slides | **To do** | — |

---

## 5. Remaining plan

| Step | Work | Gate |
|---|---|---|
| Prompt 1 | Push `mri-morphometry`; branch `kan-fusion`; fusion dataset; pre-registration | **Done** |
| Prompt 2 | KAN vs CatBoost on raw features (C / F) | **Done**: CatBoost F won; KAN not non-inferior |
| fusion-v3 | Stream specialists + KAN fusion on stream scores; freeze; single test run | **Done**: KAN K2 frozen as the decision layer; test 0.764 macro-F1 |
| Card + demo | Assessment card from frozen artefacts; demo notebook | **Done** (`fusion-v3`): showcase set chosen by a committed rule |
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
- **Final test, run once (done, `fusion-v3`):** clinical specialist 0.771,
  MRI specialist (standalone / MRI-only mode) 0.438 with AD AUC 0.873, KAN
  fusion K2 (final decision layer) 0.764, CatBoost F 0.746, CatBoost R 0.784
  (macro-F1; 5-seed averages refitted on all 691 train+val subjects). Metrics:
  accuracy, macro-F1, balanced accuracy, per-class F1, AD AUC, 3-class,
  confusion matrices, severity monotonicity, paired bootstrap CIs.

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
- One cohort; no external validation; test set of 123 subjects with few LMCI
  (24): test differences of ±0.07 macro-F1 are within the CIs.
- Fusion configurations (35) were selected on the same train CV that reports
  them; the test run is the unbiased check.
- The KAN's MRI-logit edges are not separately identifiable (centred logits sum
  to zero); read them jointly.
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
  `fusion-feasibility`, `brain-age`, `mri-morphometry`, `kan-fusion`, `fusion-v3`
  (all pushed). Branches only; never push to master.
- fusion-v3: `analysis/fusion_v3/report.md`, `freeze.md`, `config_log.csv`, `mri_controls.md` (post-hoc).
- Card: `src/card/` (builders + renderer + showcase rule), `demo/card_data/`, `demo/output/`, `demo/README.md`.
