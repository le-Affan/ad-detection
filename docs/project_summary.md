# Alzheimer's Disease & Stage Detection Using Deep Learning — Project Summary

Group 12 · Department of I.T. (AI & Data Science) · K J Somaiya School of Engineering
Team: Affan Shaikh, Aneeket Banerjee, Ankon Mukherjee, Eshani Kane
Guide: Dr. Sujata Pathak
Date: 2026-10-04
Numbers in this document come from the audited results file (`docs/final_results.md`).

---

## 1. What the project is

A multimodal, stage-aware and interpretable assessment system for Alzheimer's
disease, built on ADNI data. For one subject at one visit, it takes clinical
and cognitive records plus a structural brain MRI, and returns:

- a **stage**: Cognitively Normal (CN), Early MCI (EMCI), Late MCI (LMCI) or
  Alzheimer's Disease (AD), with class probabilities
- an **AD-vs-rest probability**
- a **continuous severity index** (0 to 3)
- an **assessment card** with explanations and flags

The project goes beyond binary AD detection: the target is the ordered
progression CN → EMCI → LMCI → AD, with a severity score and explanations.

## 2. Problem and objectives

Existing deep-learning systems mostly classify AD versus healthy, do not model
severity or progression, rely on one modality, and give weak explanations.
The project's functional requirements: accept clinical and MRI inputs, work at
the visit level, classify four stages, compute a continuous severity index,
explain predictions at feature level, and visualise severity across stages.
Non-functional requirements: interpretable, reproducible, moderate compute,
feasible on academic hardware, consistent with disease progression.

## 3. Data

| Item | Value |
|---|---|
| Source | ADNI (ADNI-GO and ADNI2 only, to keep the EMCI/LMCI distinction) |
| Clinical tables | 9 domain tables (ADNIMERGE, DXSUM, ADAS, CDR, MMSE, MoCA, RAVLT, FAQ, ECog, MRI ROIs); 404 raw variables |
| Cohort funnel | 16,015 visit records → 5,166 joined → **814 valid subjects**: CN 190, EMCI 310, LMCI 163, AD 151 |
| MRI | 811 subjects with a baseline T1-weighted scan |
| Split (clinical split is master, identical across streams) | train 569 (567 with a scan) · val 122 (121) · test 123 (123) |
| Multi-visit MRI | 690 subjects, visits bl / m06 / m12 / m24 (2,556 usable scans) |
| Leakage controls | Subject-level splits everywhere; 17 diagnosis-proxy columns blacklisted; train-only fitting of every scaler, imputer and selector; a subject-overlap check in `m3_train.py` |

Diagnosis labels (CN/EMCI/LMCI/AD) are defined in ADNI by cognitive
assessments (MMSE, CDR, memory tests). This single fact explains most of the
results below.

## 4. System architecture

```
 Clinical tables                         T1 MRI scan, ADNI tables
        │                                          │
 Top-K clinical features (12, SHAP)    ┌───────┬───┴────┬─────────┐
        │                              ▼       ▼        ▼         ▼
        │                          6 volume  3D CNN   Brain-age  Age, sex
        │                          ratios    ALZ      gap
        ▼                              └───────┴────┬───┴─────────┘
 Clinical specialist                                ▼
 CatBoost, 265 trees                         MRI specialist
        │                                  regularised classifier
        └──────────────┬────────────────────────────┘
                       ▼
              Residual KAN fusion layer
                       │
        ┌──────────────┼───────────────┐
        ▼              ▼               ▼
   Stage (COG)    AD vs rest     Severity index
        └──────────────┴───────┬───────┘
                               ▼
                       Assessment card
```

### 4.1 Clinical stream
- Six-step pipeline: ingestion → harmonisation → cohort and QC → feature
  processing → CatBoost → SHAP.
- 36 features: 35 clinical plus one engineered composite (CogIndex). Seven are
  FreeSurfer MRI volumes (Ventricles, Hippocampus, WholeBrain, Entorhinal,
  Fusiform, MidTemp, ICV); they now belong to the MRI stream.
- The final clinical specialist uses the SHAP top-12 clinical features
  (cognition, function, demographics) in CatBoost with 265 fixed trees.
- Top SHAP features in the original model: CDRSB, CogIndex, MMSE, ADAS13, ADAS11,
  RAVLT forgetting, ECog memory, FAQ.

### 4.2 MRI stream
**Preprocessing (per scan):** N4 bias correction → reorient → resample to
1×1×1.2 mm → affine registration (head, then brain) to the ICBM152 template →
template brain mask → clip and truncate intensities → min–max scaling. The
result is a 189×233×164 volume; a 96³ downsample feeds the CNN.

**Inputs to the MRI specialist:**

| Input | Source |
|---|---|
| Six regional volumes divided by ICV | ADNI FreeSurfer tables |
| ALZ score | 3D CNN (ResNet-style trunk, AD probability), **cross-fitted**: 5 folds × 3 seeds, so no training subject is scored by a model that saw it |
| Brain-age gap | DeepBrainNet (pretrained on healthy brains, not on ADNI), bias-corrected with a line fitted on cognitively normal training subjects only; set to missing when a scan fails the reliability rule |
| Age and sex | ADNI tables |

The MRI specialist is a regularised classifier chosen by nested CV log-loss
(exact model family: see `analysis/fusion_v3/config_log.csv`). Its outputs are
4-class scores and an AD probability.

### 4.3 Fusion layer
- A **residual KAN**: final logit = clinical logit + a KAN correction computed
  from the MRI scores. Inputs: clinical logits, MRI logits, age and the top-3
  clinical features.
- Trained with a class-balanced loss and modality dropout (the MRI stream is
  hidden in 15% of rows, so subjects without a scan are handled).
- An ensemble of 25 seed models; final specialists and KAN were refit on train +
  val (691 subjects) for the single test evaluation.

### 4.4 Outputs
| Output | Definition |
|---|---|
| Stage (COG) | 4-class probabilities; predicted class = argmax |
| AD vs rest (ADD) | Binary AD probability |
| Severity index | s = Σ k·p(stage k), k = 0..3; monotonic across stages on CV, val and test (Spearman with true stage 0.88 on test) |
| Assessment card | Outputs, clinical SHAP evidence, MRI evidence (volumes, Grad-CAM), brain age with reliability and extreme-value notes, KAN correction, exploratory discordance flag |

## 5. Results

All test numbers come from one evaluation, run after a committed freeze
(`8f351b6`). Test has 123 subjects (24 LMCI), so test intervals are about ±0.07.

| System | CV macro-F1 | Test macro-F1 |
|---|---|---|
| Clinical specialist | 0.744 | 0.771 |
| MRI specialist (subjects with a scan) | 0.424 | 0.438 |
| **KAN fusion (final)** | **0.756** | **0.764** |
| CatBoost, clinical + MRI inputs | 0.758 | 0.746 |
| CatBoost, all 36 features | 0.752 | 0.784 |
| Always predicting EMCI | 0.138 | 0.138 |

- KAN fusion − clinical specialist: CV +0.013 [−0.006, +0.034]; test −0.008
  [−0.081, +0.065]. **The fusion matches the best single-stream model; no
  difference is resolved.**
- Original clinical stream: test accuracy 0.813, macro-F1 0.786 (a separate,
  earlier pipeline).

### 5.1 What the MRI stream does
- **Detects AD without cognitive tests:** AD-vs-rest AUC 0.880 (CV) and 0.873
  (test), against 0.995 for the clinical stream with tests.
- **Where that comes from (post-hoc controls):** age and sex alone 0.574; six
  volume ratios alone 0.867; full set 0.880. The regional volumes carry almost
  all of it.
- **Cannot stage MCI:** LMCI F1 is 0.142 on CV and 0.071 on test.
- **Inside the fusion:** the MRI share is +0.007 [−0.006, +0.019], not reliable.
- **Brain-age gap:** clean stage gradient (CN 0.0, EMCI +2.2, LMCI +4.0, AD +6.8
  years, p ≈ 2×10⁻¹⁶); AD-vs-CN AUC 0.80 on its own.

### 5.2 When the fusion and the MRI change the stage (post-hoc, train/val only)
- **Fusion changed the stage** (final stage ≠ clinical specialist's):
  - 68 of 569 training subjects, out-of-fold: 31 corrected, 32 made wrong;
    net accuracy −0.2% [−2.8%, +2.6%];
  - validation: 12 of 122, 6 corrected and 6 made wrong.
- **MRI changed the stage** (final stage ≠ the stage with the scan treated as
  missing):
  - 10 of 121 held-out validation subjects: 6 corrected, 4 made wrong; net
    +1.7% [−3.3%, +6.6%].
- **Neither shows a measurable benefit,** consistent with the fusion matching
  the clinical specialist.

## 6. What was tested on the way

Every experiment's decision rule was committed before its results.

| Experiment | Outcome |
|---|---|
| Temporal model over multi-visit MRI | No gain over the latest scan alone |
| Follow-up scans as extra training data | No gain; overfits to subject identity |
| Class weighting; v9 reproducibility | No reliable gain; v9 macro-F1 is 0.284 ± 0.047 over 8 seeds (the logged 0.349 was a lucky single run) |
| Cross-fitted ALZ added to clinical model | No reliable gain |
| Brain-age gap added | No reliable gain |
| Removing FreeSurfer volumes from clinical features | No change |
| KAN on raw features | Failed the non-inferiority margin (−0.058); redesigned as stream specialists + residual KAN |

**Reading:** imaging signals are valid, but add no measurable staging accuracy
once cognitive assessments are present, because the stage labels are defined by
those assessments. With about 570 training subjects, the CNN is limited by
subject count, not by architecture or number of scans.

## 7. Explainability

Every explanation was evaluated with pre-registered checks:
- **Stability:** across seeds, models or refits.
- **Plausibility:** whether attributions point the expected way for CDR-SB,
  MMSE, hippocampus and ventricle volume.
- **Faithfulness:** removing the most-attributed inputs should change the
  output more than removing random ones.
- **Sanity checks, for the CNN:** does the map change when the trained weights
  are randomised?

All rules were committed before any result. Only train and validation subjects
were used, and no test metric was computed. Full report:
`analysis/explainability/report.md`.

| Component | What was measured | Status on the card |
|---|---|---|
| SHAP on the clinical specialist | <ul><li>Stable across the 5 seeds (0.90); faithful.</li><li>CDR-SB fails the pre-registered sign criterion (0.36).</li><li>A post-hoc check shows its SHAP toward AD does rise with CDR-SB (Spearman 0.61 train, 0.55 val).</li></ul> | WEAK, both CDR-SB results stated |
| SHAP on the MRI specialist (new) | <ul><li>Exact.</li><li>Volumes carry 66% of the AD attribution, ALZ + brain-age gap 27%, age + sex 6%, agreeing with the input-set controls.</li><li>Faithful.</li><li>Ventricles point the unexpected way (12% plausible), from the frozen model's coefficient.</li><li>Stability across refits 0.60.</li></ul> | WEAK; shown first in the MRI panel |
| CNN attribution (LayerCAM, replaces Grad-CAM) | <ul><li>A deletion rule chose layer1 (24³). Its top voxels remove 1.08 ALZ logits vs 0.31 for random occlusion. The old 3³ Grad-CAM was no better than random.</li><li>Both sanity checks fail: the maps change little when the weights are randomised (0.81 correlation), and occluding the top region lowers the AD score less than control regions.</li><li>The maps barely differ between AD and CN.</li></ul> | WEAK; labelled "not anatomically reliable"; no region named |
| KAN fusion (Shapley values, new) | <ul><li>Exact, all 25 models.</li><li>89 of 144 effects have the same sign in all 25 models, including every MRI effect.</li><li>78 of the 89 replicate on held-out validation subjects.</li><li>The MRI correction to the AD logit is ordered by stage: CN −0.45, EMCI −0.11, LMCI +0.23, AD +0.54.</li></ul> | WEAK (CDR-SB and MMSE plausibility); only consistent effects shown |
| "Fusion / MRI changed the stage" (new) | No measurable benefit (§5.2) | Shown with its training and held-out evidence |
| Discordance flag | 13 of 567 training subjects flagged; error rate 23% flagged vs 24% unflagged | Exploratory, not validated |

**Overall:** none of the four explanation types passes every check, and the
card states this for each one. The MRI specialist's attributions and the
fusion's MRI effects are consistent and agree with the ablations. The CNN
heatmap does not support any anatomical claim.

## 8. Methodology and integrity

- Pre-registered decision rules, committed before each experiment's results.
- Subject-level splits; the clinical split is the master split for both streams.
- A frozen test evaluation: 109 hashed artefacts, and test-time scripts that
  refuse the test split unless the freeze is committed.
- Test-set history: the original clinical stream reported a test result before
  this work; the fusion pipeline's evaluation was a separate single run.
- A number-by-number audit of the results file against saved predictions
  (`analysis/fusion_v3/results_audit.md`).
- Every tuning configuration logged (35 of 40).

## 9. Deliverables

| Deliverable | Where |
|---|---|
| Code and reports | GitHub `le-Affan/ad-detection`, branches `fusion-v3` (final system), `path-a-multivisit` (multi-visit preprocessing and temporal models), and experiment branches |
| Assessment card and demo notebook | `src/card/`, `demo/assessment_demo.ipynb` (about 15 s run) |
| Blueprint and results | `docs/pipeline_blueprint.md`, `docs/final_results.md` |
| Experiment reports | `analysis/fusion_feasibility/`, `brain_age/`, `mri_morphometry/`, `kan_fusion/`, `fusion_v3/` |

Progress against the original plan: every implementation segment is done
(clinical stream, MRI stream, severity scoring, staging, fusion, KAN,
evaluation). Documentation is partly done; the defense deck, written report and
Q&A are still to do.

## 10. Limitations

- Imaging adds no measured staging accuracy; the labels are defined by
  cognitive tests; about 570 training subjects.
- Regional volumes come from ADNI's FreeSurfer tables; a new scan would need its
  own segmentation (SynthSeg is reliable for subcortical structures, weak for
  cortical ones).
- The preprocessed volumes saturate about 5% of brain voxels and use a template
  brain mask; three GE scans fail badly; scanners are not harmonised.
- The brain-age model compresses toward the mean age and is the released VGG16
  variant, not the published architecture.
- The CNN heatmaps are not anatomical evidence. They fail
  weight-randomisation and region-deletion checks, show a stride artefact, and
  use a template atlas with affine registration. Both earlier wordings
  (hippocampal, then periventricular) are withdrawn.
- No explanation type passes every pre-registered check:
  - MRI-specialist SHAP has an implausible ventricle sign;
  - the CDR-SB plausibility checks are weak or fail;
  - KAN Shapley values explain each model's logits, not the averaged
    probability.
- Neither the fusion's nor the MRI's stage changes show a measurable benefit.
- One cohort, no external validation; the test set is small (123 subjects).
- Multi-visit raw-scan coverage is 690 of 811 subjects, and the missing subjects
  are AD-heavy.

## 11. What is left

1. Defense deck and written report, built from `docs/final_results.md`.
2. Defense preparation: updated Q&A, numbers to know cold, likely examiner
   questions.
3. Explainability fix (prompt ready) and, optionally, the resolution diagnostic.
4. Optional extras: a longitudinal severity panel (requires regenerating the
   lost atrophy-slope file) and a higher-resolution Grad-CAM.
5. Consolidating branches into `master`.

## 12. How to explain the project in one paragraph

We built a system that combines a clinical stream and an MRI stream to classify
Alzheimer's stage (CN, EMCI, LMCI, AD), estimate AD probability and a continuous
severity score, and explain each result. The clinical stream carries staging
accuracy because the stage labels are defined by cognitive tests. The MRI stream
detects AD from scans alone (AUC 0.87) and supplies imaging evidence, but it
cannot stage MCI. A residual KAN fuses the two streams and matches the best
single stream on an untouched test set (macro-F1 0.764 against 0.771). Every
design decision was tested under pre-registered rules, and every number in the
results was audited against its source.
