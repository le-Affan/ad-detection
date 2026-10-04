# Audit of docs/final_results.md

Date 2026-10-04. Branch `fusion-v3`. No model was run and no prediction was computed: values were
read from reports, JSON/CSV summaries and saved prediction files (`.npz`). One row per reported number
(a value with its CI, or a mean ± SD, is one number). Duplicated mentions in later sections are audited
where they first appear unless their label differs.

## Summary

| Status | Count |
|---|---|
| verified | 105 |
| mismatch | 6 |
| basis differs | 5 |
| not found | 1 |
| **total** | **117** |

## Step 2: macro-F1 recomputed from saved predictions

CV: `analysis/fusion_v3/streams.npz` (clinical, MRI), `kan_runs/32.npz` (KAN), `analysis/kan_fusion/catboost_preds.npz` (CatBoost F, R); mean of per-seed macro-F1 over seeds 42–46, 569 subjects (MRI 567).
Test: `analysis/fusion_v3/frozen_run/test_streams.npz` (seed-averaged) and `test_kan.npz`, 123 subjects. Threshold for STOP: |difference| > 0.002.

| System | File CV | Recomputed CV | File test | Recomputed test | Result |
|---|---|---|---|---|---|
| Clinical specialist | 0.744 | 0.7435 | 0.771 | 0.7715 | match |
| MRI specialist | 0.424 | 0.4239 | 0.438 | 0.4382 | match |
| KAN fusion (final) | 0.756 | 0.7562 | 0.764 | 0.7638 | match |
| CatBoost F | 0.758 | 0.7577 | 0.746 | 0.7461 | match |
| CatBoost R | 0.752 | 0.7517 | 0.784 | 0.7836 | match |

Largest difference 0.0005. Step 2 passed. The KAN − CatBoost F bootstrap recomputed with the original procedures (rng seed 0, 2,000 resamples) reproduces the stored intervals exactly: CV −0.0015 [−0.0212, +0.0178], test +0.0177 [−0.0448, +0.0833].

## Answers to the specific checks

- **a. Date:** was 2026-10-02; set to 2026-10-04.
- **b. MRI LMCI F1 "about 0.1":** CV 0.142 ± 0.008 (567), test 0.071 (123).
- **c. Section 3.2:** all four values in the file are test (123 subjects, all with a scan). CV equivalents (567): MRI-only AUC 0.880 / accuracy 0.489; clinical 0.983 / 0.763. Both added.
- **d. LMCI→AD 17.2 → 14.6:** clinical specialist → final KAN fusion (K2), CV, mean count per seed over 5 seeds (569 subjects). It is not "with vs without MRI": the same KAN with MRI masked gives 15.6, so MRI accounts for 1.0 of the 2.6 reduction. Test: 3 → 3.
- **e. Hippocampus:** FreeSurfer (ADNI tables) on the smoke 20 (10 CN + 10 AD, train): CN 7,053 ± 509 vs AD 6,436 ± 653 mm³, 9% smaller (AD/CN 0.91), Mann–Whitney p = 0.054, Cohen's d ≈ 1.1. SynthSeg on our volumes, same 20: 9,954 ± 1,423 vs 8,794 ± 1,064 mm³, 12% smaller (0.88), p = 0.045, d ≈ 0.9. The file's "about 10%" for FreeSurfer has no cohort-wide source (it traces to the blueprint's summary wording).
- **f. Brain age:** the 3.16-year MAE is raw MAE on the 133 CN train subjects (not CV; DeepBrainNet is pretrained). The AUCs (AD vs CN 0.797, AD vs rest 0.713) are 5-fold CV on train with the CN bias fit refitted per fold. The gap means and Kruskal–Wallis test are on train.
- **g. Always-EMCI 0.14:** 0.138 on test, all 123 test subjects (= those with a scan); also 0.138 on CV over the 567 train subjects with a scan.
- **h. CIs in Sections 1 and 5:** Section 1: KAN − clinical CV [−0.006, +0.034] and test [−0.081, +0.065]; KAN − CatBoost F CV [−0.021, +0.018] and test [−0.045, +0.083] (added); MRI share CV [−0.006, +0.019]. Section 5: experiments 4–7 are CV paired bootstraps; experiment 3's [−0.029, +0.092] is a two-sided t-interval over 8 seeds on val.
- **i. MRI share +0.007 [−0.006, +0.019]:** CV (final K2, config #32, vs the same K2 with MRI masked, #34). It was not run on test.

## All numbers

| # | Section | Item | Value in file | Value in source | Source | Status | Change |
|---|---|---|---|---|---|---|---|
| 1 | Header | Date | 2026-10-02 | audit date 2026-10-04 (file saved 2026-10-04) | task instruction; file mtime | mismatch | set to 2026-10-04 |
| 2 | 0 | Train subjects (with a scan) | 569 (567) | 569 (567) | analysis/fusion_v3/freeze.md; eval_tables.md n | verified |  |
| 3 | 0 | Val subjects (with a scan) | 122 (121) | 122 (121) | analysis/fusion_v3/eval_tables.md | verified |  |
| 4 | 0 | Test subjects (with a scan) | 123 (123) | 123 (n_with_scan 123) | analysis/fusion_v3/frozen_run/test_results.json | verified |  |
| 5 | 0 | CV folds × seeds | 5-fold, 5 seeds | StratifiedKFold(5), seeds 42–46 | analysis/fusion_v3/eval_tables.md | verified |  |
| 6 | 0 | Bootstrap resamples / level | 2,000, 95% | 2,000, 2.5–97.5 percentiles | analysis/fusion_v3/evaluate.py, test_evaluate.py | verified | experiment 3 (Section 5) is a t-interval; exception added |
| 7 | 0 | Final training set | 691 | 691 (569 + 122) | analysis/fusion_v3/freeze.md | verified |  |
| 8 | 0 | Freeze commit | 8f351b6 | 8f351b6 | analysis/fusion_v3/freeze.md; git log | verified |  |
| 9 | 0 | Tuning budget | 35 of 40 | 35 rows | analysis/fusion_v3/config_log.csv | verified |  |
| 10 | 0 | Original step-5 test accuracy | 0.813 | 0.8130 | models/step5_metrics.json (test_accuracy) | verified |  |
| 11 | 0 | Original step-5 test macro-F1 | 0.786 | 0.786 (mean of per-class F1 1.00 / 0.82 / 0.45 / 0.875) | models/step5_metrics.json (test_per_class) | verified |  |
| 12 | 1 | Test LMCI count | 24 | labels [29, 47, 24, 23] | analysis/fusion_v3/frozen_run/test_results.md | verified |  |
| 13 | 1 | Test interval width | ±0.07 | half-widths 0.073 / 0.064 / 0.058 for the three KAN comparisons | analysis/fusion_v3/frozen_run/test_results.md | verified |  |
| 14 | 1 | Clinical specialist macro-F1, CV | 0.744 | 0.7435 (recomputed); 0.744 ± 0.011 | analysis/fusion_v3/eval_tables.md | verified | basis CV, 569, mean of 5 seeds |
| 15 | 1 | Clinical specialist macro-F1, test | 0.771 | 0.7715 (recomputed) | analysis/fusion_v3/frozen_run/test_results.md | verified |  |
| 16 | 1 | MRI specialist macro-F1, CV | 0.424 | 0.4239 (recomputed), 567 with a scan | analysis/fusion_v3/eval_tables.md | verified |  |
| 17 | 1 | MRI specialist macro-F1, test | 0.438 | 0.4382 (recomputed), 123 | analysis/fusion_v3/frozen_run/test_results.md | verified |  |
| 18 | 1 | KAN fusion macro-F1, CV | 0.756 | 0.7562 (recomputed) | analysis/fusion_v3/eval_tables.md | verified |  |
| 19 | 1 | KAN fusion macro-F1, test | 0.764 | 0.7638 (recomputed) | analysis/fusion_v3/frozen_run/test_results.md | verified |  |
| 20 | 1 | CatBoost F macro-F1, CV | 0.758 | 0.7577 (recomputed) | analysis/kan_fusion/catboost_preds.npz; eval_tables.md | verified |  |
| 21 | 1 | CatBoost F macro-F1, test | 0.746 | 0.7461 (recomputed) | analysis/fusion_v3/frozen_run/test_results.md | verified |  |
| 22 | 1 | CatBoost R macro-F1, CV | 0.752 | 0.7517 (recomputed) | analysis/kan_fusion/catboost_preds.npz; eval_tables.md | verified |  |
| 23 | 1 | CatBoost R macro-F1, test | 0.784 | 0.7836 (recomputed) | analysis/fusion_v3/frozen_run/test_results.md | verified |  |
| 24 | 1 | Always-EMCI baseline | 0.14 (test column) | 0.138 on test, all 123 test subjects (= the subjects with a scan); 0.138 on CV (567 train subjects with a scan) | analysis/fusion_v3/frozen_run/test_results.md; analysis/fusion_v3/eval_tables.md (MRI roles) | verified | subject set stated; CV value added |
| 25 | 1 | Historical clinical stream test | 0.786 (acc 0.813) | 0.786 / 0.8130 | models/step5_metrics.json | verified |  |
| 26 | 1 | KAN − clinical, CV | +0.013 [−0.006, +0.034] | +0.0126 [−0.0063, +0.0339], CV | analysis/fusion_v3/eval_tables.md | verified | basis CV |
| 27 | 1 | KAN − clinical, test | −0.008 [−0.081, +0.065] | −0.0077 [−0.0808, +0.0650], test | analysis/fusion_v3/frozen_run/test_results.md | verified | basis test |
| 28 | 1 | KAN − CatBoost F, CV | −0.002 (no CI) | −0.0015 [−0.0212, +0.0178], CV; recomputed identically | analysis/fusion_v3/eval_tables.md | verified | CI added |
| 29 | 1 | KAN − CatBoost F, test | +0.018 (no CI) | +0.0177 [−0.0448, +0.0833], test; recomputed identically | analysis/fusion_v3/frozen_run/test_results.md | verified | CI added |
| 30 | 1 | MRI share | +0.007 [−0.006, +0.019] (CV column) | +0.0067 [−0.0062, +0.0193]: CV, final K2 (#32) vs K2 with MRI masked (#34); not run on test | analysis/fusion_v3/eval_tables.md | verified | basis CV confirmed; labelled 'CV only' |
| 31 | 1 | CatBoost R highest test point estimate | 0.784 | 0.784 (highest of the four) | analysis/fusion_v3/frozen_run/test_results.md | verified |  |
| 32 | 3.1 | MRI AD-vs-rest AUC, CV | 0.880 | 0.8796, 567 | analysis/fusion_v3/eval_tables.md | verified |  |
| 33 | 3.1 | MRI AD-vs-rest AUC, test | 0.873 | 0.873, 123 | analysis/fusion_v3/frozen_run/test_results.md | verified |  |
| 34 | 3.1 | MRI macro-F1, CV / test | 0.424 / 0.438 | 0.424 / 0.438 | analysis/fusion_v3/eval_tables.md; analysis/fusion_v3/frozen_run/test_results.md | verified | duplicate of Section 1 |
| 35 | 3.1 | MRI LMCI F1, CV | about 0.1 | 0.142 ± 0.008 | analysis/fusion_v3/eval_tables.md | mismatch | replaced with 0.142 |
| 36 | 3.1 | MRI LMCI F1, test | about 0.1 | 0.071 | analysis/fusion_v3/frozen_run/test_results.md | mismatch | replaced with 0.071 |
| 37 | 3.2 | MRI-only AD AUC | 0.873 (test) | 0.873, test, 123 | analysis/fusion_v3/frozen_run/test_results.md | verified | CV 0.880 added |
| 38 | 3.2 | Clinical AD AUC (same subjects) | 0.995 (test) | 0.995, test, 123 | analysis/fusion_v3/frozen_run/test_results.md | verified | CV 0.983 (567) added |
| 39 | 3.2 | MRI-only 4-class accuracy | 0.50 (test) | 0.504, test | analysis/fusion_v3/frozen_run/test_results.md | verified | CV 0.489 added |
| 40 | 3.2 | Clinical 4-class accuracy | 0.80 (test) | 0.797, test | analysis/fusion_v3/frozen_run/test_results.md | verified | CV 0.763 (567) added |
| 41 | 3.3 | MRI share (repeat) | +0.007 [−0.006, +0.019] | CV, #32 vs #34 | analysis/fusion_v3/eval_tables.md | verified | labelled CV |
| 42 | 3.3 | LMCI→AD errors | 17.2 → 14.6 'with MRI inputs' (CV) | 17.2 = clinical specialist, 14.6 = KAN fusion K2: CV, mean count per seed (5 seeds), 569 subjects; KAN with MRI masked (#34) = 15.6 | eval_tables.md (MRI roles); kan_runs/34.npz (read only) | mismatch | relabelled: clinical → fusion; MRI-attributable part 1.0 |
| 43 | 3.3 | LMCI→AD on test | stays at 3 | 3 → 3 (clinical → KAN) | analysis/fusion_v3/frozen_run/test_results.md | verified |  |
| 44 | 3.3 | LMCI recovered on test | 8 → 12 ('class-balanced loss on test') | 8 → 12, clinical specialist vs KAN fusion on test (the whole fusion, not the loss in isolation) | analysis/fusion_v3/frozen_run/test_results.md | basis differs | relabelled clinical → fusion; CV config log attribution kept as context |
| 45 | 3.3 | EMCI called LMCI on test | 7 → 13 | 7 → 13, same comparison | analysis/fusion_v3/frozen_run/test_results.md | basis differs | same relabel |
| 46 | 3.3 | Edge consistency model count | 25 | 25 KANs trained on the 569 train subjects (5 outer × 5 KAN seeds) | analysis/fusion_v3/kan_interp.md | verified | basis stated (not the 25 frozen KANs) |
| 47 | 3.3 | MRI-AD → AD non-monotone | 18 of 25 | 0 up, 7 down, 18 non-monotone of 25 | analysis/fusion_v3/kan_interp.md | verified |  |
| 48 | 3.3 | has_mri pruned | all 25 | removed in all 25 | analysis/fusion_v3/report.md §5 | verified |  |
| 49 | 3.4 | M0 CV AD AUC | 0.574 ± 0.009 | 0.574 ± 0.009 | analysis/fusion_v3/mri_controls.json | verified |  |
| 50 | 3.4 | M0 val AD AUC | 0.558 | 0.558 | analysis/fusion_v3/mri_controls.json | verified |  |
| 51 | 3.4 | M0 CV macro-F1 | 0.189 ± 0.008 | 0.189 ± 0.008 | analysis/fusion_v3/mri_controls.json | verified |  |
| 52 | 3.4 | M0 val macro-F1 | 0.160 | 0.160 | analysis/fusion_v3/mri_controls.json | verified |  |
| 53 | 3.4 | M1 CV AD AUC | 0.867 ± 0.005 | 0.867 ± 0.005 | analysis/fusion_v3/mri_controls.json | verified |  |
| 54 | 3.4 | M1 val AD AUC | 0.781 | 0.781 | analysis/fusion_v3/mri_controls.json | verified |  |
| 55 | 3.4 | M1 CV macro-F1 | 0.379 ± 0.013 | 0.379 ± 0.013 | analysis/fusion_v3/mri_controls.json | verified |  |
| 56 | 3.4 | M1 val macro-F1 | 0.298 | 0.298 | analysis/fusion_v3/mri_controls.json | verified |  |
| 57 | 3.4 | M2 CV AD AUC | 0.871 ± 0.004 | 0.871 ± 0.004 | analysis/fusion_v3/mri_controls.json | verified |  |
| 58 | 3.4 | M2 val AD AUC | 0.781 | 0.781 | analysis/fusion_v3/mri_controls.json | verified |  |
| 59 | 3.4 | M2 CV macro-F1 | 0.412 ± 0.007 | 0.412 ± 0.007 | analysis/fusion_v3/mri_controls.json | verified |  |
| 60 | 3.4 | M2 val macro-F1 | 0.362 | 0.362 | analysis/fusion_v3/mri_controls.json | verified |  |
| 61 | 3.4 | M3 CV AD AUC | 0.876 ± 0.005 | 0.876 ± 0.005 | analysis/fusion_v3/mri_controls.json | verified |  |
| 62 | 3.4 | M3 val AD AUC | 0.787 | 0.787 | analysis/fusion_v3/mri_controls.json | verified |  |
| 63 | 3.4 | M3 CV macro-F1 | 0.415 ± 0.007 | 0.415 ± 0.007 | analysis/fusion_v3/mri_controls.json | verified |  |
| 64 | 3.4 | M3 val macro-F1 | 0.373 | 0.373 | analysis/fusion_v3/mri_controls.json | verified |  |
| 65 | 3.4 | M4 CV AD AUC | 0.880 ± 0.005 | 0.880 ± 0.005 | analysis/fusion_v3/mri_controls.json | verified |  |
| 66 | 3.4 | M4 val AD AUC | 0.811 | 0.811 | analysis/fusion_v3/mri_controls.json | verified |  |
| 67 | 3.4 | M4 CV macro-F1 | 0.424 ± 0.005 | 0.424 ± 0.005 | analysis/fusion_v3/mri_controls.json | verified |  |
| 68 | 3.4 | M4 val macro-F1 | 0.391 | 0.391 | analysis/fusion_v3/mri_controls.json | verified |  |
| 69 | 3.4 | M5 CV AD AUC | 0.788 ± 0.001 | 0.788 ± 0.001 | analysis/fusion_v3/mri_controls.json | verified |  |
| 70 | 3.4 | M5 val AD AUC | 0.794 | 0.794 | analysis/fusion_v3/mri_controls.json | verified |  |
| 71 | 3.4 | M5 CV macro-F1 | 0.304 ± 0.008 | 0.304 ± 0.008 | analysis/fusion_v3/mri_controls.json | verified |  |
| 72 | 3.4 | M5 val macro-F1 | 0.284 | 0.284 | analysis/fusion_v3/mri_controls.json | verified |  |
| 73 | 3.4 | M4 − M0 AUC (CV) | +0.306 [+0.239, +0.370] | +0.306 [+0.239, +0.370] | analysis/fusion_v3/mri_controls.json | verified | basis CV stated |
| 74 | 3.4 | M4 − M0 macro-F1 (CV) | +0.235 [+0.196, +0.274] | +0.235 [+0.196, +0.274] | analysis/fusion_v3/mri_controls.json | verified |  |
| 75 | 3.4 | M4 − M2 AUC, CV | +0.009 [−0.003, +0.020] | +0.009 [−0.003, +0.020] | analysis/fusion_v3/mri_controls.json | verified |  |
| 76 | 3.4 | M4 − M2 AUC, val | +0.030 [+0.002, +0.060] | +0.030 [+0.002, +0.060] | analysis/fusion_v3/mri_controls.json | verified |  |
| 77 | 3.4 | Val subjects / AD | 121, 23 AD | 121 with a scan; labels [28, 46, 24, 23] | analysis/fusion_v3/mri_controls.md | verified |  |
| 78 | 3.5 | ALZ AUC, train cross-fitted | 0.764 | 0.764 (567, pooled cross-fit) | analysis/fusion_feasibility/report.md | verified |  |
| 79 | 3.5 | ALZ AUC, val | 0.778 | 0.778 (121, 5-fold-model ensemble) | analysis/fusion_feasibility/report.md | verified |  |
| 80 | 3.5 | Brain-age MAE | 'CV MAE on CN 3.16 years raw' | 3.16 raw MAE on the 133 CN train subjects; no CV (pretrained external model); 3.97 after correction | analysis/brain_age/p2_report.md §2.1 | basis differs | relabelled: CN train, not CV |
| 81 | 3.5 | Corrected gap CN | 0.0 | +0.00 (train, n 133) | analysis/brain_age/p2_report.md §2.2 | verified | split: train |
| 82 | 3.5 | Corrected gap EMCI | +2.2 | +2.20 (train, n 215) | analysis/brain_age/p2_report.md §2.2 | verified |  |
| 83 | 3.5 | Corrected gap LMCI | +4.0 | +4.02 (train, n 114) | analysis/brain_age/p2_report.md §2.2 | verified |  |
| 84 | 3.5 | Corrected gap AD | +6.8 | +6.80 (train, n 104) | analysis/brain_age/p2_report.md §2.2 | verified |  |
| 85 | 3.5 | Kruskal–Wallis p | 2×10⁻¹⁶ | 2.2 × 10⁻¹⁶ (H = 76.0) | analysis/brain_age/p2_report.md §2.2 | mismatch | rounding; set to 2.2×10⁻¹⁶, H added |
| 86 | 3.5 | Gap AD-vs-CN AUC | 0.80 | 0.797, 5-fold CV on train, per-fold CN bias fit | analysis/brain_age/p2_report.md §2.6 | verified | basis (CV, train) stated |
| 87 | 3.5 | Gap AD-vs-rest AUC | 0.71 | 0.713, same basis | analysis/brain_age/p2_report.md §2.6 | verified |  |
| 88 | 3.5 | FreeSurfer hippocampus | about 10% smaller in AD | no cohort-wide figure; FreeSurfer on the smoke 20 (10 CN + 10 AD train): 9% smaller (AD/CN 0.91), p = 0.054, d ≈ 1.1 | analysis/mri_morphometry/part1_report.md; smoke_summary.md | not found | replaced with the sourced smoke-sample figure |
| 89 | 3.5 | SynthSeg hippocampus | 12% | 12% (0.88), Mann–Whitney p = 0.045, d ≈ 0.9, same 20 | analysis/mri_morphometry/part1_report.md; smoke_summary.md | verified |  |
| 90 | 3.5 | SynthSeg p | 0.045 | 0.045 | analysis/mri_morphometry/smoke_summary.md | verified |  |
| 91 | 3.5 | Smoke sample size | 20 subjects | 20 (10 CN + 10 AD, train) | analysis/mri_morphometry/part1_report.md | verified | composition added |
| 92 | 4 | Monotonic 'in every seed' | every seed | 5/5 seeds on CV and val; test is one seed-averaged system | analysis/fusion_v3/eval_tables.md; analysis/fusion_v3/frozen_run/test_results.md | basis differs | reworded |
| 93 | 4 | Spearman with stage, test | 0.88 | 0.884 (KAN) | analysis/fusion_v3/frozen_run/test_results.md | verified |  |
| 94 | 4 | EMCI–LMCI severity gap | about 0.35 | KAN: 0.37 test (1.77 − 1.40), 0.42 CV, 0.38 val | analysis/fusion_v3/frozen_run/test_results.md; analysis/fusion_v3/eval_tables.md | mismatch | set to 0.37 test (0.42 CV) |
| 95 | 5 | Exp. 1 DX_bl vs static macro-F1 | 0.20 vs 0.34 | 0.203 ± 0.046 vs 0.344; val (Path A subset, 17 AD) | docs/path_a_stages_abc_report.md | verified | basis val added |
| 96 | 5 | Exp. 2 R1 vs R0 | within ±0.015 | every metric within ±0.015 (val, 121) | docs/mv_static_report.md | verified |  |
| 97 | 5 | Exp. 2 R3 vs R2 acc4 | −0.067 | −0.067 (val) | docs/mv_static_report.md | verified |  |
| 98 | 5 | Exp. 3 v9 expected val macro-F1 | 0.284 ± 0.047 | 0.284 ± 0.047 (8 seeds, val) | docs/mv_static_report.md | verified |  |
| 99 | 5 | Exp. 3 logged single run | 0.349 | 0.349 | docs/mv_static_report.md | verified |  |
| 100 | 5 | Exp. 3 class weighting | +0.032 [−0.029, +0.092] | +0.032, 95% CI [−0.029, +0.092], two-sided t over 8 seeds (df 7), val | docs/mv_static_report.md | basis differs | CI is a t-interval, not paired bootstrap; labelled |
| 101 | 5 | Exp. 3 seeds won | 6/8 | 6 of 8 | docs/mv_static_report.md | verified |  |
| 102 | 5 | Exp. 4 B − A | +0.000 [−0.009, +0.008] | same, CV | analysis/fusion_feasibility/report.md | verified | CV added |
| 103 | 5 | Exp. 4 B′ − A′ | +0.009 [−0.003, +0.021] | same, CV | analysis/fusion_feasibility/report.md | verified |  |
| 104 | 5 | Exp. 5 E − B | −0.002 [−0.011, +0.006] | same, CV | analysis/brain_age/p2_report.md | verified | CV added |
| 105 | 5 | Exp. 5 E′ − B′ | +0.004 [−0.007, +0.016] | same, CV | analysis/brain_age/p2_report.md | verified |  |
| 106 | 5 | Exp. 6 A − A0 | +0.002 [−0.010, +0.015] | same, CV | analysis/mri_morphometry/part1_report.md | verified | CV added |
| 107 | 5 | Exp. 6 A′ − A0′ | −0.003 [−0.021, +0.015] | same, CV | analysis/mri_morphometry/part1_report.md | verified |  |
| 108 | 5 | Exp. 7 KAN F − CatBoost F | −0.058 [−0.078, −0.038] | same, CV | analysis/kan_fusion/report.md | verified | CV added |
| 109 | 5 | Clinical reference val macro-F1 | 0.778 | 0.778 ± 0.020 (arm A, 265 trees, val) | analysis/fusion_feasibility/report.md | verified |  |
| 110 | 5 | Clinical reference CV macro-F1 | 0.752 | 0.752 ± 0.006 (arm A, CV) | analysis/fusion_feasibility/report.md | verified |  |
| 111 | 5 | Saved step-5 model val macro-F1 | 0.808 | 0.808 | analysis/fusion_feasibility/report.md | verified |  |
| 112 | 6 | Wording: test 0.764 vs 0.771, −0.008 [−0.081, +0.065] | as stated | −0.0077 [−0.0808, +0.0650], test | analysis/fusion_v3/frozen_run/test_results.md | verified |  |
| 113 | 6 | Wording: AUC 0.87 on test | 0.87 | 0.873 | analysis/fusion_v3/frozen_run/test_results.md | verified |  |
| 114 | 7 | v9 macro-F1 old / new | 0.349 / 0.284 ± 0.047 | same | docs/mv_static_report.md | verified |  |
| 115 | 7 | Clinical val 0.808; use test 0.771, CV 0.744 | as stated | 0.808; 0.771 test; 0.744 CV | analysis/fusion_v3/frozen_run/test_results.md; analysis/fusion_v3/eval_tables.md | verified |  |
| 116 | 7 | Original clinical test 0.813 / 0.786 | as stated | 0.8130 / 0.786 | models/step5_metrics.json | verified |  |
| 117 | 7 | Grad-CAM map size | 3×3×3 | layer4 output 3×3×3 at 96³ input | src/card/build_gradcam.py; src/mri/model.py | verified |  |

## Non-numeric label corrections

- Section 3.1 title said "scan-derived inputs only"; the MRI specialist also uses age and sex (`freeze.md`). Corrected.
- Section 3.3 "Class-balanced loss on test" row compared the clinical specialist with the whole fusion; relabelled.
