# Pipeline Blueprint — Revised Multimodal AD Assessment System

Date: 2026-09-28
Status: proposed plan. Supersedes the MRI-staging parts of the original design.
Related docs: `docs/mri_role_refined_plan.md`, `analysis/fusion_feasibility/report.md`,
`docs/mv_static_report.md`, `docs/path_a_roadmap.md`

---

## 1. Purpose

Deliver the system promised in the project proposal (stage classification,
continuous severity, AD probability, explanations), with each component doing
what the evidence shows it can do:

- The **clinical stream** carries staging accuracy.
- The **MRI stream** supplies independent imaging evidence as a small number
  of scalar inputs: ALZ probability and, if validated, brain-age gap.
- The **fusion layer (KAN)** produces the final outputs and makes their
  relationships inspectable.

Core principle, unchanged: honest, pre-registered results over report-promised
results.

---

## 2. Why the plan changed

| Finding | Evidence | Consequence |
|---|---|---|
| MRI 4-class staging hits a ceiling | Static CNN macro-F1 0.284 ± 0.047 (8 seeds); v9's 0.349 was an upper-tail single run | Drop MRI staging as a goal |
| More scans of the same subjects don't help | Temporal modelling (Path A) and extra-scans-as-training both negative | The bottleneck is subject count (~570), not architecture or scan count |
| The labels are defined by cognitive tests | ADNI assigns CN/EMCI/LMCI/AD from MMSE, CDR, Logical Memory | Clinical features dominate staging; EMCI/LMCI is a test-score boundary |
| ALZ adds no measurable accuracy to clinical data | Cross-fitted ablation: B − A macro-F1 +0.000 CV; B′ − A′ +0.009, CI crosses 0 | MRI must bring information the clinical features lack |
| The clinical model's main error is LMCI → EMCI | 13 of 16 LMCI validation errors | MRI is unlikely to fix staging; its role is evidence and AD-specific signal |
| The design never required MRI staging | The original fusion vector was `[DEMO, ALZ, clinical top-K]`; the 4-class output belongs to fusion | The new role is consistent with the original architecture |

The rigorous literature agrees. Leakage-free studies report ~0.90 AUC for AD vs.
healthy and ~63–68% accuracy for 3-class CN/MCI/AD, and rarely attempt
EMCI vs. LMCI from MRI at all (Wen et al., 2020; Liu et al., 2020).

---

## 3. Target architecture

```
                 ADNI visit (subject, baseline)
                            │
          ┌─────────────────┴─────────────────┐
          ▼                                   ▼
   Raw T1 MRI scan                      Clinical tables
          │                                   │
   Preprocessing [built]              Ingestion + QC [built]
   N4 → register → skull-strip        36 features (incl. FreeSurfer volumes)
          │                                   │
     ┌────┴──────────┐                CatBoost + SHAP [built]
     ▼               ▼                        │
  3D CNN [built]   Brain-age model [to test]  ▼
  ALZ probability  Brain-age gap          Top-K features
  (cross-fitted)   (pretrained, 1 value)      │
     │               │                        │
     └───────┬───────┴────────────────────────┘
             ▼
     Fusion layer [to build]
     KAN, compared with CatBoost on identical inputs
             │
     ┌───────┼──────────────┐
     ▼       ▼              ▼
  Stage    AD vs rest    Severity index
  (COG)    (ADD)         (expected stage, 0–3)
     └───────┴──────┬───────┘
                    ▼
            Assessment card  ◄── Explanations (not model inputs):
                                  SHAP, Grad-CAM, discordance flag
```

### 3.1 Fusion input vector

```
[ clinical top-K features ,  ALZ (cross-fitted) ,  brain-age gap (if it passes the gate) ]
```

- K is chosen on training data only (a SHAP ranking inside each CV fold, or on
  the full train split for the final model).
- DEMO is **not** a fusion input. It stays as an auxiliary training loss inside
  the CNN.
- Path C atrophy slopes are **not** inputs to the baseline model, because they
  use post-baseline visits.

### 3.2 Outputs

| Output | Definition | Notes |
|---|---|---|
| Stage (COG) | 4-class softmax: CN / EMCI / LMCI / AD | Primary output |
| AD vs rest (ADD) | Binary AD probability | ADNI has no non-AD dementias, so ADD is AD vs. everything else |
| Severity index | Σ k·p(stage k), k = 0..3 | Continuous, ordered by construction. Validated for monotonic increase across stages. Not regressed on CDR-SB, which is an input |
| Explanations | SHAP (clinical), Grad-CAM (MRI), KAN edge-function plots, discordance flag | Displayed on the card; not model inputs |

### 3.3 Discordance flag

Raised when the clinical prediction and the imaging evidence disagree, for
example predicted LMCI with high ALZ or a large brain-age gap, or predicted AD
with low ALZ. Thresholds are set on training data. It is reported
descriptively (rate, which classes), **not** as a performance claim.

---

## 4. Component specifications

### 4.1 Clinical stream (built)
- CatBoost MultiClass, 36 features (35 clinical + CogIndex), SHAP top-K.
- Test result: accuracy 0.813, macro-F1 0.786 (test set untouched since).
- Fixed-tree retrain (265 trees): val macro-F1 0.778, CV 0.752. Quote these,
  not the saved model's validation-tuned 0.808.
- Note: the features include FreeSurfer volumes, so the stream is not
  imaging-free. The "clinical explanations never mention MRI" rule must be
  reworded, or the volumes moved to the imaging stream.

### 4.2 MRI stream: ALZ (built)
- DualHeadNet, v9 lr_3e-5 config, no class weighting.
- Cross-fitted: 5 folds over clinical-train × 3 seeds each; val and test ALZ =
  mean of the 5 fold models.
- Cross-fitted train AUC 0.764, val AUC 0.778.
- Output file: `analysis/fusion_feasibility/alz_crossfit.csv`.

### 4.3 MRI stream: brain-age gap (to test)
- **Model selection criteria:**
  1. Publicly available pretrained weights.
  2. Training data excludes ADNI, or at minimum none of our subjects.
     Verify, don't assume.
  3. Input format compatible with our full-resolution MNI outputs
     (`mni/` folders, pre-96³). Check resolution, orientation, skull-stripping
     and intensity normalisation.
- **Output:** gap = predicted age − chronological age at scan, one value per
  subject.
- **Sanity checks before use:**
  - MAE on CN subjects close to the model's published MAE.
  - Mean gap increases CN < EMCI < LMCI < AD.
  - No large offset by scanner or site.
  - The gap is not simply a proxy for age (correlation with age near zero
    after any bias correction).
- **Bias correction:** if the model shows the usual regression-to-the-mean
  bias, fit the correction on **CN training subjects only**.

### 4.4 Fusion layer (to build)
- **KAN** (pykan), 2 layers, small width. Inputs are the Section 3.1 vector,
  standardised on training data. Outputs are the COG and ADD heads; severity is
  derived from COG.
- **Loss:** λ1·CE(COG) + λ2·BCE(ADD). λ values are tuned on validation within
  a small grid.
- **Comparator:** CatBoost with identical inputs and targets (planned ablation
  "H": KAN vs. a conventional decision model).
- **Time box:** about one day. If KAN doesn't train stably, the CatBoost
  decision layer becomes the final model and KAN is reported as an attempted
  comparison.
- **Interpretability deliverables:** edge-function plots for ALZ, brain-age
  gap and the top clinical features, plus pruning results (which inputs KAN
  keeps).

### 4.5 Standalone MRI result (for the defense, separate from fusion)
- 3-class CN / MCI / AD, merging EMCI and LMCI, matching the literature.
- Apply the Liu et al. (2020) design findings: instance normalisation, less
  early downsampling, a wider network, age as an input.
- Same multi-seed, validation-only protocol. Answers "what can MRI do alone?"

---

## 5. Implementation phases

| Phase | Work | Gate / decision | Est. effort |
|---|---|---|---|
| **P0** | Housekeeping: commit untracked docs; push branches; PR the leakage check to master; update slide numbers | — | 1–2 h |
| **P1** | Brain-age model search and vetting (training data, input format, licence) | A suitable model exists. If not, drop brain-age and go to P3 | 2–4 h |
| **P2** | Compute brain-age gap for all 811 subjects, run sanity checks, run the pre-registered ablation arms | Gap passes the rule (Section 6) → include in fusion. Otherwise card display only | ~1 day |
| **P3** | KAN fusion build + CatBoost comparator + severity index + discordance flag | Time-boxed to ~1 day | 1–1.5 days |
| **P4** | Standalone 3-class MRI model with design fixes | Optional if time is short | ~1 day |
| **P5** | Pre-registered one-time test evaluation | Model list frozen **before** running | 2–3 h |
| **P6** | Assessment card, demo notebook update, report and slides | — | 1–2 days |

Optional, only if the longitudinal panel is wanted: regenerate Path C slopes
from the FreeSurfer tables (~half a day) for a returning-patient view on the
card.

---

## 6. Evaluation protocol

**Splits.** The clinical split is the master split: 567 train, 122 val, 123
test, identical across streams (verified). Subject-level; no subject crosses
splits.

**Brain-age gap ablation (P2), pre-registered before running:**

| Arm | Inputs |
|---|---|
| A | Clinical |
| B | Clinical + ALZ |
| E | Clinical + ALZ + brain-age gap |
| A′ | Clinical minus diagnosis-defining scores |
| B′ | A′ + ALZ |
| E′ | A′ + ALZ + brain-age gap |

- **Decision rule:** E beats B (or E′ beats B′) only if the paired bootstrap
  95% CI of the macro-F1 difference in CV excludes zero **and** the validation
  difference has the same sign.
- **Protocol:** CatBoost at a fixed 265 trees, 5 seeds, repeated stratified
  5-fold CV over clinical-train, plus validation. Report on all subjects and on
  the MRI-only subset.

**Final test evaluation (P5).** Run once, on 123 subjects, with the model list
frozen in a commit before running:
1. Clinical CatBoost (fixed trees)
2. Clinical + MRI inputs, CatBoost
3. KAN fusion (final)
4. ALZ alone, for reference

Metrics: accuracy, macro-F1, balanced accuracy, per-class F1, AD AUC,
confusion matrix, severity monotonicity, stage-distance error.

---

## 7. Known traps and how they are handled

| Trap | Handling |
|---|---|
| In-sample MRI scores are overfit | ALZ is cross-fitted. Brain-age uses a model never trained on ADNI |
| Cross-stream split mismatch | Verified identical; clinical split is the master |
| Feature selection on the full data | Top-K chosen on training data only |
| Validation used for tuning, then reported | Quote CV and fixed-tree numbers; test stays untouched until P5 |
| Future information | Path C slopes excluded from the baseline model |
| Brain-age model trained on ADNI | Verified in P1 before any use |
| Brain-age age bias | Correction fitted on CN training subjects only |
| Circularity with label-defining scores | A′/B′/E′ arms reported alongside A/B/E |
| Changing rules after seeing results | Every decision rule committed before its experiment runs |

---

## 8. What is dropped or demoted

- MRI 4-class staging as a goal: **dropped**.
- DEMO as a fusion input: **dropped** (auxiliary CNN loss only).
- DEMO Plan A fix: **optional**, low priority.
- Path A temporal model: **reported as a controlled negative result**. The
  latest-scan finding on the dementia-at-last-visit target is kept as a
  secondary, test-pending result.
- Multi-visit scans as extra training data: **reported as a negative result**.
- Class weighting: **no reliable gain**, not used.

---

## 9. How the report should frame it

- **Contribution:** a stage-aware, interpretable multimodal framework in which
  each stream's role is justified by controlled experiments.
- **MRI:** a 3D CNN on ~570 training subjects yields a real but redundant AD
  signal (AUC ~0.77). Staging accuracy is bounded by subject count and by
  cognitively defined labels, supported by four pre-registered experiments.
  Imaging contributes independent evidence (ALZ, brain-age gap, Grad-CAM),
  whether or not it raises accuracy.
- **Brain-age gap:** described as an established imaging biomarker
  incorporated as a fusion input, not as a novel metric.
- **KAN:** its value is interpretability (plottable edge functions), measured
  against CatBoost rather than assumed better.

**Numbers to correct in slides:**

| Item | Old | New |
|---|---|---|
| MRI v9 macro-F1 | 0.349 | 0.284 ± 0.047 (8 seeds) |
| Clinical validation macro-F1 | 0.808 | Don't quote. Use test 0.813 / 0.786, CV 0.752 |
| Grad-CAM wording | Hippocampus / temporal lobe | Periventricular / ventricular regions |
| MRI outputs | Staging | ALZ + brain-age gap; staging comes from fusion |

---

## 10. Open decisions

1. Which pretrained brain-age model (P1 outcome).
2. Whether to build the Path C longitudinal panel.
3. Whether P4 (standalone 3-class MRI) fits the timeline.
4. Final K for the clinical top-K features (chosen on training data).
