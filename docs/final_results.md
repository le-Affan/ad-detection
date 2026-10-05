# Final Results — Multimodal AD Assessment (ADNI-GO/2)

Date: 2026-10-04 (audited against the sources on 2026-10-04: `analysis/fusion_v3/results_audit.md`)
Source: `analysis/fusion_v3/report.md`, `analysis/fusion_v3/mri_controls.md`,
`analysis/kan_fusion/report.md`, `analysis/brain_age/p2_report.md`,
`analysis/mri_morphometry/part1_report.md`, `analysis/fusion_feasibility/report.md`,
`docs/mv_static_report.md`, `docs/path_a_stages_abc_report.md`.
Section 2 was filled from the saved prediction files (no model was run).
Every number was checked against its source; see Section 8.

---

## 0. Protocol

| Item | Value |
|---|---|
| Splits (clinical split is master, identical across streams) | train 569 (567 with a scan), val 122 (121), test 123 (123) |
| CV | Repeated stratified 5-fold over clinical-train, 5 seeds |
| Confidence intervals | Paired bootstrap over subjects, 2,000 resamples, 95% (exception: experiment 3 in Section 5 uses a t-interval over 8 seeds) |
| Final fusion system | Specialists and KAN refit on train + val (691 subjects) |
| Test read | Once, after the freeze commit `8f351b6`. Test-time scripts refuse the test split unless the freeze is committed |
| Tuning budget | 35 of 40 configurations, all in `analysis/fusion_v3/config_log.csv` |

**Test-set history.** The original clinical stream (`step5`) already reported a
test result (accuracy 0.813, macro-F1 0.786) before this work. The freeze
governs the fusion-v3 pipeline, which has its own single test evaluation. Say
this when asked whether the test set was used before.

---

## 1. Headline: final system versus alternatives

Macro-F1, 4 classes (CN / EMCI / LMCI / AD). CV: 569 clinical-train subjects,
mean over 5 seeds (MRI specialist: the 567 with a scan). Test: 123 subjects
(24 LMCI; all 123 have a scan), single frozen run, so test intervals are about
±0.07 wide.

| System | CV macro-F1 | Test macro-F1 |
|---|---|---|
| Clinical specialist | 0.744 | 0.771 |
| MRI specialist (subjects with a scan) | 0.424 | 0.438 |
| **KAN fusion (final, residual variant)** | **0.756** | **0.764** |
| CatBoost on F (clinical + MRI inputs) | 0.758 | 0.746 |
| CatBoost on R (all 36 features, reference) | 0.752 | 0.784 |
| Always predicting EMCI (baseline; subjects with a scan) | 0.138 (567) | 0.138 (123) |
| Historical: original clinical stream (different pipeline; hyperparameters chosen on val) | — | 0.786 (accuracy 0.813) |

Paired differences (macro-F1):

| Comparison | CV [95% CI] | Test [95% CI] | Reading |
|---|---|---|---|
| KAN fusion − clinical specialist | +0.013 [−0.006, +0.034] | −0.008 [−0.081, +0.065] | No difference resolved |
| KAN fusion − CatBoost F | −0.002 [−0.021, +0.018] | +0.018 [−0.045, +0.083] | No difference resolved |
| MRI share inside the fusion (final KAN vs the same KAN with MRI inputs masked; CV only) | +0.007 [−0.006, +0.019] | — (not run on test) | Not reliable |

**Reading:** the fusion matches the best single-stream model and the
conventional alternatives. Every interval includes zero. CatBoost R has the
highest test point estimate (0.784), which is also within noise of the others.

---

## 2. Full metrics

Computed from the saved prediction files (`analysis/fusion_v3/streams.npz`,
`kan_runs/32.npz`, `analysis/kan_fusion/catboost_preds.npz` for CV;
`analysis/fusion_v3/frozen_run/test_*.npz` for test). No model was run. CV is
the mean over seeds 42–46 of per-seed metrics (± SD); test is the single
frozen system (each system's seed average). AD-vs-rest AUC uses P(AD) of the
4-class output. Stage-distance error is the mean |predicted stage − true
stage| (0–3).

### 2.1 Final system (KAN fusion, residual variant K2)

| Metric | CV (569) | Test (123) |
|---|---|---|
| Accuracy | 0.756 ± 0.013 | 0.764 |
| Balanced accuracy | 0.756 ± 0.013 | 0.773 |
| Macro-F1 | 0.756 ± 0.014 | 0.764 |
| F1 CN / EMCI / LMCI / AD | 0.923 / 0.740 / 0.495 / 0.866 | 1.000 / 0.727 / 0.471 / 0.857 |
| AD-vs-rest AUC (stage output) | 0.984 ± 0.002 | 0.992 |
| AD-vs-rest AUC (separate ADD head) | 0.984 | 0.992 |
| Stage-distance error | 0.253 ± 0.015 | 0.252 |

### 2.2 All systems

**CV** (mean over 5 seeds):

| System | n | Accuracy | Bal. acc. | Macro-F1 | F1 CN | F1 EMCI | F1 LMCI | F1 AD | AD AUC | Stage-dist. error |
|---|---|---|---|---|---|---|---|---|---|---|
| Clinical specialist | 569 | 0.763 | 0.740 | 0.744 | 0.945 | 0.776 | 0.407 | 0.846 | 0.983 | 0.245 |
| MRI specialist | 567 | 0.489 | 0.441 | 0.424 | 0.394 | 0.599 | 0.142 | 0.561 | 0.880 | 0.629 |
| **KAN fusion (final)** | 569 | 0.756 | 0.756 | 0.756 | 0.923 | 0.740 | 0.495 | 0.866 | 0.984 | 0.253 |
| CatBoost F | 569 | 0.774 | 0.753 | 0.758 | 0.946 | 0.780 | 0.439 | 0.865 | 0.983 | 0.236 |
| CatBoost R | 569 | 0.774 | 0.750 | 0.752 | 0.947 | 0.786 | 0.410 | 0.865 | 0.982 | 0.232 |

**Test** (single frozen run):

| System | n | Accuracy | Bal. acc. | Macro-F1 | F1 CN | F1 EMCI | F1 LMCI | F1 AD | AD AUC | Stage-dist. error |
|---|---|---|---|---|---|---|---|---|---|---|
| Clinical specialist | 123 | 0.797 | 0.780 | 0.771 | 1.000 | 0.788 | 0.400 | 0.898 | 0.995 | 0.211 |
| MRI specialist | 123 | 0.504 | 0.472 | 0.438 | 0.542 | 0.554 | 0.071 | 0.586 | 0.873 | 0.634 |
| **KAN fusion (final)** | 123 | 0.764 | 0.773 | 0.764 | 1.000 | 0.727 | 0.471 | 0.857 | 0.992 | 0.252 |
| CatBoost F | 123 | 0.772 | 0.759 | 0.746 | 1.000 | 0.763 | 0.341 | 0.880 | 0.997 | 0.236 |
| CatBoost R | 123 | 0.813 | 0.791 | 0.784 | 1.000 | 0.804 | 0.432 | 0.898 | 0.993 | 0.203 |

**KAN fusion − CatBoost F, macro-F1** (paired bootstrap, 2,000 resamples, rng
seed 0): CV −0.002 [−0.021, +0.018] (seed-averaged within each resample);
test +0.018 [−0.045, +0.083].

### 2.3 Confusion matrices

Rows are the true class, columns the predicted class, both in the order CN /
EMCI / LMCI / AD. CV cells are mean counts per seed (569 subjects; MRI
specialist 567); test cells are counts (123).

| System | CV | Test |
|---|---|---|
| Clinical specialist | 120.2 12.8 0.0 0.0 / 1.2 184.0 30.8 1.0 / 0.0 57.0 39.8 17.2 / 0.0 3.6 11.0 90.4 | 29 0 0 0 / 0 39 7 1 / 0 13 8 3 / 0 0 1 22 |
| MRI specialist | 43.6 80.4 4.6 4.4 / 31.2 160.2 8.8 15.8 / 11.0 56.2 11.0 35.8 / 2.2 22.4 17.0 62.4 | 13 15 0 1 / 5 31 3 8 / 1 13 1 9 / 0 6 0 17 |
| **KAN fusion (final)** | 120.0 12.8 0.2 0.0 / 5.8 160.4 49.0 1.8 / 1.2 41.2 57.0 14.6 / 0.0 2.0 10.2 92.8 | 29 0 0 0 / 0 32 13 2 / 0 9 12 3 / 0 0 2 21 |
| CatBoost F | 120.0 12.4 0.6 0.0 / 0.6 185.2 30.0 1.2 / 0.0 56.0 43.2 14.8 / 0.0 4.0 8.8 92.2 | 29 0 0 0 / 0 37 9 1 / 0 13 7 4 / 0 0 1 22 |
| CatBoost R | 120.0 13.0 0.0 0.0 / 0.4 188.8 27.0 0.8 / 0.0 58.8 38.6 16.6 / 0.0 3.0 8.8 93.2 | 29 0 0 0 / 0 41 5 1 / 0 13 8 3 / 0 1 0 22 |

---

## 3. What the MRI stream does

### 3.1 Standalone (MRI specialist: scan-derived inputs plus age and sex)

| Metric | CV (567 with a scan) | Test (123) |
|---|---|---|
| AD-vs-rest AUC | 0.880 | 0.873 |
| 4-class macro-F1 | 0.424 | 0.438 |
| LMCI F1 | 0.142 | 0.071 |

It detects AD well and cannot stage MCI.

### 3.2 MRI-only mode (scans plus age and sex, no cognitive tests) versus clinical

Same subjects (those with a scan). The MRI-only mode is the MRI specialist.

| Metric | MRI-only mode, CV (567) | Clinical specialist, CV (567) | MRI-only mode, test (123) | Clinical specialist, test (123) |
|---|---|---|---|---|
| AD-vs-rest AUC | 0.880 | 0.983 | 0.873 | 0.995 |
| 4-class accuracy | 0.489 | 0.763 | 0.504 | 0.797 |

### 3.3 Inside the fusion

| Finding | Value |
|---|---|
| MRI share, macro-F1 (CV; final KAN vs the same KAN with MRI masked) | +0.007 [−0.006, +0.019], not reliable |
| LMCI→AD errors, clinical specialist → KAN fusion (CV, mean count per seed, 569 subjects) | 17.2 → 14.6; the same KAN with MRI masked gives 15.6, so 1.0 of the 2.6 is attributable to MRI. Test: 3 → 3 |
| Clinical specialist → KAN fusion on test | LMCI recovered 8 → 12; EMCI called LMCI 7 → 13 (a trade, not a gain). On CV this shift appears when the class-balanced loss is adopted (config log #26/#28) |
| KAN edges consistent across all 25 seed models (the 25 KANs trained on the 569 train subjects) | MRI evidence for "normal" lowers the AD logit; MRI evidence for AD lowers the CN logit |
| Not usable as an explanation | The direct MRI-AD → AD edge is non-monotone in 18 of 25 models |
| `has_mri` flag | Pruned in all 25 models; missing scans handled by modality dropout |

### 3.4 Where the MRI specialist's AD detection comes from (post-hoc controls, train CV and val)

| Set | Inputs | CV AD AUC | Val AD AUC | CV macro-F1 | Val macro-F1 |
|---|---|---|---|---|---|
| M0 | age + sex | 0.574 ± 0.009 | 0.558 | 0.189 ± 0.008 | 0.160 |
| M1 | 6 volumes/ICV | 0.867 ± 0.005 | 0.781 | 0.379 ± 0.013 | 0.298 |
| M2 | volumes + age + sex | 0.871 ± 0.004 | 0.781 | 0.412 ± 0.007 | 0.362 |
| M3 | M2 + ALZ | 0.876 ± 0.005 | 0.787 | 0.415 ± 0.007 | 0.373 |
| **M4** | **M3 + brain-age gap (frozen specialist)** | **0.880 ± 0.005** | **0.811** | **0.424 ± 0.005** | **0.391** |
| M5 | ALZ + brain age only | 0.788 ± 0.001 | 0.794 | 0.304 ± 0.008 | 0.284 |

- M4 − M0 (CV): AUC +0.306 [+0.239, +0.370]; macro-F1 +0.235 [+0.196, +0.274].
- M4 − M2: AUC +0.009 CV [−0.003, +0.020]; +0.030 val [+0.002, +0.060]
  (121 subjects, 23 AD). Post-hoc, so it is not a pre-registered claim.
- **Reading:** age and sex barely detect AD. The scan-derived inputs supply the
  rest, almost all from the six FreeSurfer volume ratios. The CNN and brain-age
  add a small increment.

### 3.5 Imaging signals on their own

| Signal | Result |
|---|---|
| CNN ALZ (cross-fitted) | AD-vs-rest AUC 0.764 (train, cross-fitted), 0.778 (val) |
| Brain-age gap (DeepBrainNet, bias-corrected) | MAE 3.16 years raw (3.97 after bias correction) on the 133 CN train subjects; not cross-validated, since DeepBrainNet is a pretrained external model. Corrected gap on train: CN 0.0 → EMCI +2.2 → LMCI +4.0 → AD +6.8 years (n 133 / 215 / 114 / 104; Kruskal–Wallis H = 76.0, p = 2.2×10⁻¹⁶). Gap as a score, 5-fold CV on train with per-fold CN bias fit: AD-vs-CN AUC 0.80, AD-vs-rest 0.71 |
| Hippocampus, CN vs AD (smoke sample: 10 CN + 10 AD train subjects) | ADNI FreeSurfer: 9% smaller in AD (AD/CN 0.91), Mann–Whitney p = 0.054, Cohen's d ≈ 1.1. SynthSeg on our volumes, same 20 subjects: 12% smaller (0.88), p = 0.045, d ≈ 0.9. No cohort-wide FreeSurfer estimate is reported |

---

## 4. Severity index

s = Σ k·p(stage k), k = 0..3. Class means rise CN < EMCI < LMCI < AD on CV, val
and test (in all 5 seeds on CV and val; test is the single seed-averaged
system). Spearman with true stage: 0.88 on test. For the final KAN, EMCI and
LMCI means sit only 0.37 apart on test (0.42 on CV), which mirrors the
classifier's main error.
CDR-SB correlation is descriptive only (CDR-SB is an input).

---

## 5. Experiment log (all rules committed before running)

| # | Experiment | Result |
|---|---|---|
| 1 | Temporal model over multi-visit MRI | No temporal signal; sequence models do not beat last-scan-only. DX_bl variant macro-F1 0.20 vs static 0.34 (val, Path A subset, 17 AD) |
| 2 | Follow-up scans as extra training data | No gain (R1 vs R0 within ±0.015); with class weighting −0.067 acc4 (R3 vs R2). Val (121). Overfits to subject identity |
| 3 | v9 reproducibility (8 seeds) and class weighting | v9 expected val macro-F1 **0.284 ± 0.047** (logged 0.349 was an upper-tail single run). Class weighting +0.032 [−0.029, +0.092] (val; t-interval over 8 seeds, not a bootstrap), 6/8 seeds |
| 4 | Cross-fitted ALZ added to clinical CatBoost | B − A +0.000 [−0.009, +0.008]; B′ − A′ +0.009 [−0.003, +0.021] (CV) |
| 5 | Brain-age gap added | E − B −0.002 [−0.011, +0.006]; E′ − B′ +0.004 [−0.007, +0.016] (CV) |
| 6 | Remove 7 FreeSurfer volumes from clinical features | A − A0 +0.002 [−0.010, +0.015]; A′ − A0′ −0.003 [−0.021, +0.015] (CV) |
| 7 | KAN on raw features (v1) | KAN F − CatBoost F −0.058 [−0.078, −0.038] (CV); failed the non-inferiority margin. Led to the stream-specialist redesign |
| 8 | Stream specialists + KAN fusion (v3, final) | Matches clinical specialist and CatBoost (Section 1) |

(A′/B′/E′: diagnosis-defining scores removed. A0: FreeSurfer columns removed.)

Clinical reference: fixed-tree retrain val macro-F1 0.778, CV 0.752. The saved
model's validation 0.808 is optimistic (settings tuned on val).

---

## 6. Suggested wording

- "The fusion system matches the best single-stream model (test macro-F1 0.764
  vs 0.771; difference −0.008, 95% CI [−0.081, +0.065])."
- "The MRI stream detects AD without cognitive testing (AUC 0.87 on test),
  driven mainly by regional volumes. It cannot stage MCI."
- "Imaging adds no measurable staging accuracy once cognitive assessments are
  available, because the stage labels are defined by those assessments."
- "The KAN's consistent learned effects show how MRI evidence shifts the clinical
  prediction at the CN/AD ends of the scale."

Avoid: "fusion improves accuracy", "MRI improves staging", "KAN outperforms",
and attributing the 0.87 AD detection to the CNN.

---

## 7. Numbers to correct in slides and report

| Item | Old | New |
|---|---|---|
| MRI v9 macro-F1 | 0.349 | 0.284 ± 0.047 (8 seeds, val) |
| Clinical val macro-F1 | 0.808 | Do not quote. Use test 0.771 (frozen specialist), CV 0.744 |
| Original clinical test result | "untouched" | Reported once by the original pipeline (0.813 / 0.786); the fusion freeze is a separate evaluation |
| Grad-CAM wording | Hippocampus / temporal lobe; later "periventricular / ventricular" | Neither. The card names no region and labels the CNN map "not anatomically reliable" (§9): both sanity checks fail and the maps barely differ between AD and CN. Say: "the CNN attribution cannot support an anatomical claim". The old layer4 Grad-CAM removed no more ALZ evidence than random occlusion |
| MRI stream role | Staging | AD detection from scan-derived inputs, evidence and explanations; staging comes from fusion |
| Progression modelling | Modelled | Cross-sectional; temporal modelling tested, no gain |
| Final decision layer | KAN direct | Stream specialists + residual KAN fusion |

---

## 8. Audit status

Audited on 2026-10-04 against the source reports, JSON/CSV summaries and saved
prediction files: `analysis/fusion_v3/results_audit.md`. Macro-F1 for all five
systems was recomputed from the saved predictions on CV and test, and matches
to within 0.0005. No **[fill]** cells remain.

Section 9 (explainability, added 2026-10-05 on branch `explainability`) was
filled from `analysis/explainability/results/` and its report. It contains no
test metric.

---

## 9. Explainability (post-hoc, train/val only)

**Protocol.**
- Branch `explainability`; the full write-up is `analysis/explainability/report.md`.
- Every rule and threshold was pre-registered in
  `analysis/explainability/preregistration.md` (commit `050ca8e`) before any
  result existed.
- Nothing frozen was retrained or modified (`MANIFEST.sha256`, 109 files,
  verifies).
- No test metric was computed: train is primary, val the replication.
- The only fits were in-memory copies, never saved: 200 bootstrap refits of
  the MRI specialist, and the 25 train-only KANs, regenerated behind an exact
  reproduction gate.

**CNN attribution (P1).**
- **Stage choice.** LayerCAM at layer1 (24³) was chosen by the deletion rule:
  - its top 10% of voxels removes 1.08 ALZ logits, against 0.31 for random
    occlusion;
  - layer2 is also eligible;
  - layer4, the old card's level, is not (−0.02 [−0.09, 0.06] above random).
- **Naming rule.** It names 7 deep regions (amygdala, thalamus, hippocampus,
  ventricles, basal ganglia, cerebral white matter, brainstem; densities
  1.1–1.3× the brain average in AD).
- **But the same pattern appears in every class.** AD − CN differences are
  ≤ 0.06, and AD is lower than CN for ventricles.
- **Both sanity checks fail:**
  - maps correlate 0.81 with the original after randomising the head and
    layer4, and 0.72 with everything randomised;
  - occluding the top region lowers ALZ less than control regions (−0.12
    [−0.15, −0.08]).
- **Conclusion:** the CNN attribution supports no anatomical claim. The card
  names no region and labels the map "not anatomically reliable"
  (pre-registration Addendum 1, made after the results; it removes claims).
- **Known limitation of the naming rule:** its "also present in non-AD
  subjects" qualifier is not applied where AD < CN significantly.

**MRI-specialist SHAP (P2).**
- Exact linear SHAP; additivity error 1e-15.
- Share of the AD attribution:
  - volumes 66% [65, 68];
  - ALZ + brain-age gap 27%;
  - age + sex 6%.
- This agrees with the M0–M5 controls.
- The frozen ventricle coefficient points away from AD: plausibility 12%, FAIL.

**KAN Shapley (P3).**
- Exact; efficiency error 1e-6.
- 89 of 144 cohort cells are consistent in all 25 KANs, including all 36 MRI
  cells.
- On held-out val with the train-only KANs: 78 replicated, 7 partially, 4 not.
- The MRI group's correction to the AD logit is ordered by stage: CN −0.45,
  EMCI −0.11, LMCI +0.23, AD +0.54.

**Stage changes (P4).**

| Indicator | Data | Fired | Corrected / made wrong | Net accuracy [95% CI] |
|---|---|---|---|---|
| Fusion changed the stage | train out-of-fold (569) | 68 | 31 / 32 | −0.2% [−2.8%, +2.6%] |
| Fusion changed the stage | val (122) | 12 | 6 / 6 | 0.0% [−5.7%, +5.7%] |
| MRI changed the stage (scan vs scan treated as missing) | val, train-only KANs (121) | 10 | 6 / 4 | +1.7% [−3.3%, +6.6%] |

No measurable benefit. The discordance flag stays exploratory: on train
out-of-fold, flagged subjects have error rate 23% vs 24% unflagged.

**Evaluation (P5).** Thresholds: PASS ≥ 0.80, WEAK 0.60–0.80.

| Explanation | Stability | Plausibility | Faithfulness | Sanity | Overall |
|---|---|---|---|---|---|
| CNN region attribution | PASS 0.91 | PASS (trivially: all classes > 1) | PASS | **FAIL** (cascade, region deletion) | WEAK |
| MRI-specialist SHAP | WEAK 0.60 | hippocampus PASS 0.85; ventricles **FAIL** 0.12 | PASS | — | WEAK |
| Clinical SHAP | PASS 0.90 | CDR-SB **FAIL** 0.36 (post-hoc ρ 0.61, see below); MMSE WEAK 0.71 | PASS | — | WEAK |
| KAN Shapley | PASS 0.88 | CDR-SB WEAK 0.60; MMSE WEAK 0.79 | PASS | — | WEAK |

**Post-hoc CDR-SB check (not pre-registered).**
- The pre-registered CDR-SB criterion compares each subject with the CN-train
  median, which is near zero. EMCI and LMCI subjects therefore count as
  "above".
- Checked differently, the clinical SHAP value of CDR-SB toward AD rises with
  CDR-SB: Spearman 0.61 [0.54, 0.68] on train, 0.55 [0.37, 0.70] on val.
- The FAIL verdict is unchanged.

**Card.** The card shows:
- MRI SHAP drivers first, then the CNN map labelled "not anatomically
  reliable", with no region names;
- KAN effects only where all 25 KANs agree;
- both stage-change indicators;
- a VALIDATED/WEAK status, with the failed checks, on every explanation panel.

**Suggested wording.** "Every explanation was evaluated against pre-registered
checks; none passed all of them. The MRI specialist's attributions agree with
the input-set ablations, and the fusion's MRI correction is consistent across
all 25 models. The CNN heatmaps fail model-randomisation checks and are shown
as non-anatomical."

