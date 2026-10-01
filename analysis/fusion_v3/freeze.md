# fusion-v3 — FREEZE (committed before any test subject's scan or features is read)

Branch `fusion-v3`. Date 2026-10-01.
After this commit the test set is run **once**, by the scripts below, with no
re-runs after any test metric exists. Only purely mechanical failures (crash,
missing file) may be fixed, and every such fix is logged in the report.

## Chosen system

**Final decision layer: KAN fusion K2 (residual), config 32 of `config_log.csv`.**
It has the highest train-CV macro-F1 of the 35 logged configurations:
0.7562 ± 0.0135 (5 seeds, nested stacking).

| Item | Value |
|---|---|
| **Clinical specialist** | CatBoost MultiClass, depth 6, l2_leaf_reg 10, lr 0.1, 265 trees, PTGENDER categorical, on the top 12 of the 29 arm-A0 clinical features, ranked by mean \|SHAP\| from a 29-feature CatBoost fitted on the same rows (ranking redone inside every fit) |
| **MRI specialist** | multinomial logistic regression, L2, C = 0.1, lbfgs, max_iter 5000, on 6 FreeSurfer volume/ICV ratios + cross-fitted ALZ + corrected brain-age gap + age (AGE) + sex. Median imputation and z-scoring fitted on its training rows. Fitted on and applied to subjects with a scan only |
| **Brain-age gap** | (pred_age − β)/α − age_at_scan. α, β from OLS of pred_age on age_at_scan over the training rows' CN subjects with a reliable brain age. NaN (then imputed by the specialist's training median) if the scan fails the reliability rule (no zero voxels in the template mask, or > 10% voxels ≥ 0.999), has no brain age, or fails registration (m1 corr_vs_mni < 0.15) |
| **KAN inputs (13)** | clinical centred log-probabilities (4), MRI centred log-probabilities (4), has_mri flag, AGE, CDRSB, CogIndex, MMSE. z-scored on the KAN training rows (MRI columns over rows with a scan). Missing MRI → neutral logits (centred log of the training class prior), flag 0. No temperature |
| **KAN** | pykan 0.2.8 MultKAN width [13, 5], grid 5, spline order 3, SiLU base, seed = KAN seed. Output = log p_clinical + KAN outputs 0–3 (COG); output 4 = ADD logit |
| **Loss** | class-balanced CE (weights n / (4·n_k) from the fitting rows) + 0.5 × squared EMD between cumulative predicted and true stage distributions + 0.5 × BCE(ADD) + 1e-3 × pykan reg('edge_forward_spline_n', 1, 2, 0, 0) |
| **Training** | Adam lr 0.01, full batch, ≤ 400 steps, early stopping on a stratified 15% holdout (patience 40, best state restored), grid updated from the fitting rows at step 0 and every 5 steps to step 50, modality dropout 15% of fitting rows per step |
| **Seeds** | outer seeds 42–46 (specialist seeds and stream-score fold seeds); KAN seeds 42–46 per outer seed |
| **Heads** | COG = softmax of the residual logits; ADD = sigmoid(output 4); severity s = Σ k·p(stage k) from COG |

## Final training data

**All 691 train+val subjects** (569 + 122), not 689 as the task text says:
the clinical split has 569 train and 122 val. Of these, 688 have a scan and 684
a reliable brain age.

1. **Features.** `frozen/feature_stats.json` is fitted on the 691:
   - the step-4 rules (`src/step4_processing.py`): median/mode imputation,
     z-scoring, CogIndex;
   - FreeSurfer ratios: median imputation, z-scoring.
   
   `final_features.py` asserts that the same code fitted on train alone
   reproduces `step4_train.csv` and the fusion-dataset ratios exactly.
2. **Scan inputs.**
   - ALZ: train = cross-fitted (out-of-fold) ALZ; val = mean of the 5 fold
     models. The ALZ CNNs are not retrained.
   - Brain age: `analysis/brain_age/brainage_trainval.csv`.
   - Reliability flags: `analysis/kan_fusion/brainage_reliability.csv`.
3. **Specialists** (`freeze_fit_specialists.py`): for each seed s:
   - the clinical and MRI specialists are fitted on all 691;
   - 5-fold out-of-fold stream scores over the 691 (StratifiedKFold(5,
     shuffle, random_state = s)) train the KANs;
   - reference CatBoost F (kan-fusion frozen top-12 + 7 MRI inputs) and R (36
     step-4 features) are fitted on all 691, harness settings, seed s.
4. **KAN** (`freeze_fit_kan.py`): 25 KANs (outer seed s × KAN seed). Each
   reload reproduces its in-memory predictions (< 1e-5), and retraining is
   bit-identical.

## Test-time procedure (run in this order, once)

```
.venv-mri/bin/python            analysis/fusion_v3/scan_inputs.py   --split test   # ALZ = mean of the 15 fold CNNs; reliability rule
TF_USE_LEGACY_KERAS=1 .venv-deepbrainnet/bin/python analysis/fusion_v3/brainage_split.py --split test
.venv-fusion/bin/python         analysis/fusion_v3/test_streams.py  --split test   # frozen stats + specialists + CatBoost F/R
.venv-kan/bin/python            analysis/fusion_v3/test_kan.py      --split test   # 25 frozen KANs, averaged
.venv-fusion/bin/python         analysis/fusion_v3/test_evaluate.py --split test
```

- **Freeze guard:** every script refuses `--split test` unless this file is
  committed and unmodified.
- **Dry run on val** (val is in the final training data, so its numbers are not
  estimates):
  - `scan_inputs.py` reproduces val ALZ to 6e-7 and the val reliability flags;
  - `brainage_split.py` reproduces val brain age to 0.0004 years;
  - the full chain runs end to end.
- **Hashes:** every frozen artefact, the code above, the input tables and the
  15 ALZ checkpoints are listed with their sha256 in `frozen/MANIFEST.sha256`
  (its own hash is given in the commit message).

## Systems and metrics reported on test

**Systems:**
1. **Clinical specialist:** the 5-seed average.
2. **MRI specialist,** standalone, which is also the MRI-only mode (MRI inputs +
   age + sex, no cognitive tests). Scored on subjects with a scan, against the
   majority class (EMCI) and against the clinical specialist on the same
   subjects.
3. **Chosen KAN fusion:** the average of the 25 KANs.
4. **CatBoost F and CatBoost R:** 5-seed averages, as references.

**Metrics:**
- macro-F1, accuracy and balanced accuracy;
- per-class F1 (LMCI and AD highlighted);
- AD AUC (from P(AD) of COG), and the ADD-head AUC for the KAN;
- 3-class macro-F1 and accuracy (CN / MCI / AD);
- confusion matrices;
- LMCI→AD, AD→LMCI, LMCI→EMCI and EMCI→LMCI counts, clinical vs fusion;
- severity: class means, monotonicity, Spearman with stage, and Spearman with
  CDRSB (descriptive; CDRSB is an input);
- paired bootstrap 95% CIs (2,000 resamples of the 123 test subjects, rng
  seed 0) for KAN − clinical specialist, KAN − CatBoost F and KAN − CatBoost R.

**No decision depends on the test numbers.** They are reported next to the
train-CV estimates.
