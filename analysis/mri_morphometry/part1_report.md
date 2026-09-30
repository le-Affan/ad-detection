# MRI morphometry — Part 1: Step 0 gate, SynthSeg smoke test, brain-age outliers

Branch `mri-morphometry` (off `brain-age` at `80d5af2`). Date 2026-09-30.
Train and val only. **No test subject's scan or features were loaded,
processed or evaluated.** `run_synthseg.py` asserts this against
`test_rids.csv`. The step-5 SHAP file (`models/step5_shap_values.npz`) was not
used because it was computed on test.

## Summary

| Item | Result |
|---|---|
| **Step 0 verdict** | **Morphometry does not matter** under the pre-registered rule. A − A0: CV +0.002 [−0.010, +0.015], val +0.026. A′ − A0′: CV −0.003 [−0.021, +0.015], val +0.008. |
| SynthSeg | FreeSurfer 7.4.1 `mri_synthseg` (SynthSeg 2.0 + parcellation + QC), CPU, ~20 s/scan |
| Input | `mni/` volumes: no native-space intermediate exists. **All volumes are in affine-normalised ICBM152 space.** |
| Smoke QC | 22/22 segmented, 0 failures, all QC scores ≥ 0.70 |
| FreeSurfer agreement | Ventricles ρ 0.95–0.98, hippocampus 0.79 (vs FS/ICV), whole brain 0.74 (vs FS/ICV); cortical ROIs weak (entorhinal 0.41, fusiform 0.27, middle temporal 0.45); ICV meaningless (ρ −0.18) |
| Hippocampus CN vs AD | 9,954 vs 8,794 mm³: 12% smaller in AD, p = 0.045. FreeSurfer on the same 20 subjects: 9%, p = 0.054. |
| Brain-age outliers | 098_S_4275 (−23 y) is a **preprocessing intensity failure: flag it**. 016_S_4097 (−25 y) is a **plausible scan**: show it, with an extreme-value note. The site-016 flag is partly one more intensity failure (016_S_4121) and partly real anatomy. |

---

## Step 1 — MRI-derived columns in the clinical features

The feature list was checked in `step4_train.csv` (36 features) and
`step2_manifest.json`. **Seven** columns are MRI measurements, all ADNI
FreeSurfer volumes: `Ventricles`, `Hippocampus`, `WholeBrain`, `Entorhinal`,
`Fusiform`, `MidTemp`, `ICV`. No other feature is imaging-derived. The rest are
demographics, APOE4, cognitive/functional scales and CogIndex. There are no PET
or CSF columns.

SHAP rank: mean |SHAP| over subjects and classes, from the full-train models
of each seed. The rank is the mean over 5 seeds, with per-seed ranks in
brackets.

| Column | A (of 36), val | A, train in-sample | A′ (of 33), val | A′, train in-sample |
|---|---|---|---|---|
| Fusiform | **10.8** (8, 7, 12, 15, 12) | 11.2 | 11.0 (13, 12, 9, 14, 7) | 11.2 |
| Hippocampus | **11.0** (10, 15, 7, 9, 14) | 12.2 | 14.8 (14, 15, 14, 16, 15) | 15.2 |
| Entorhinal | 12.2 (13, 12, 9, 14, 13) | 14.2 | **6.4** (6, 4, 6, 8, 8) | 7.4 |
| WholeBrain | 21.6 (19, 25, 24, 19, 21) | 21.4 | 25.4 (26, 25, 26, 24, 26) | 25.4 |
| MidTemp | 22.2 (29, 16, 20, 26, 20) | 23.2 | 21.2 (20, 20, 24, 21, 21) | 21.2 |
| Ventricles | 28.0 (25, 30, 25, 30, 30) | 28.0 | 25.2 (25, 22, 27, 25, 27) | 25.0 |
| ICV | 32.4 (34, 33, 32, 32, 31) | 32.4 | 29.6 (30, 30, 30, 30, 28) | 29.6 |

The model uses them in mid-table positions. Without the defining scores,
Entorhinal rises to about #6. High SHAP rank did not translate into accuracy
(Step 3).

## Steps 2–3 — Step 0 gate

Pre-registered in `step0_preregistration.md`, committed as `207d3ea` before
the run. Script: `step0_gate.py`, which reuses the fusion-feasibility harness
unchanged. Full tables are in `step0_tables.md`.

**Harness check:** arms A and A′ reproduce the P2 / fusion-feasibility
numbers exactly (CV 0.752 / 0.620, val 0.778 / 0.644).

**Cross-validation, 569 clinical-train (primary), mean ± SD over 5 seeds**

| Arm | n | Macro-F1 | CN F1 | EMCI F1 | LMCI F1 | AD F1 | Bal. acc | AD AUC |
|---|---|---|---|---|---|---|---|---|
| A | 36 | 0.752 ± 0.006 | 0.947 | 0.786 | 0.410 ± 0.019 | 0.865 ± 0.003 | 0.750 | 0.982 |
| A0 | 29 | 0.750 ± 0.008 | 0.946 | 0.785 | 0.402 ± 0.024 | 0.864 ± 0.006 | 0.747 | 0.982 |
| A′ | 33 | 0.620 ± 0.006 | 0.704 | 0.667 | 0.351 ± 0.013 | 0.758 ± 0.021 | 0.621 | 0.958 |
| A0′ | 26 | 0.623 ± 0.013 | 0.719 | 0.676 | 0.351 ± 0.025 | 0.745 ± 0.004 | 0.624 | 0.958 |

**Validation, train 569 → 122 clinical-val**

| Arm | Macro-F1 | CN F1 | EMCI F1 | LMCI F1 | AD F1 | Bal. acc | AD AUC |
|---|---|---|---|---|---|---|---|
| A | 0.778 ± 0.020 | 0.982 | 0.818 | 0.395 ± 0.058 | 0.917 ± 0.014 | 0.785 | 0.995 |
| A0 | 0.752 ± 0.020 | 0.982 | 0.818 | 0.320 ± 0.066 | 0.889 ± 0.013 | 0.767 | 0.994 |
| A′ | 0.644 ± 0.009 | 0.778 | 0.712 | 0.307 ± 0.028 | 0.778 ± 0.011 | 0.652 | 0.963 |
| A0′ | 0.635 ± 0.012 | 0.749 | 0.713 | 0.307 ± 0.020 | 0.772 ± 0.015 | 0.642 | 0.966 |

**Paired macro-F1 differences (2,000 bootstrap resamples, mean over seeds)**

| Pair | Eval | Diff | 95% CI | Per-seed | ΔLMCI F1 | ΔAD F1 | ΔBal. acc | ΔAD AUC |
|---|---|---|---|---|---|---|---|---|
| A − A0 | CV | +0.002 | [−0.010, +0.015] | −.003 +.006 +.008 +.004 −.005 | +0.007 | +0.000 | +0.002 | −0.000 |
| A − A0 | val | +0.026 | [−0.011, +0.067] | +.042 +.028 −.027 +.048 +.038 | +0.075 | +0.028 | +0.018 | +0.001 |
| A′ − A0′ | CV | −0.003 | [−0.021, +0.015] | +.009 +.014 −.011 −.016 −.010 | +0.000 | +0.012 | −0.003 | −0.000 |
| A′ − A0′ | val | +0.008 | [−0.012, +0.028] | +.007 +.007 +.025 −.010 +.014 | +0.000 | +0.006 | +0.010 | −0.003 |

**Verdict: A does not beat A0; A′ does not beat A0′. Morphometry does not
matter under the rule.**

What the numbers say:
- On CV, the primary evaluation, removing all seven FreeSurfer volumes changes
  macro-F1 by at most ±0.003, and both CIs sit tightly around zero.
- The val gain for A − A0 (+0.026, of which LMCI F1 accounts for +0.075, or
  about 1–2 LMCI subjects) is not reproduced on CV. It has the same size and
  the same "val-only" pattern as the earlier ALZ result.
- The cognitive scales carry the label information that the volumes also
  carry, so the model gets no net gain from the volumes.
- **Implication for Part 2:** if ADNI's own FreeSurfer volumes add nothing
  measurable, our own morphometry is unlikely to (it is noisier; see Step 4).
  The Part 2 ablation would most likely give a fifth null result, and its
  "recovered fraction" is undefined, because A − A0 is not meaningfully above
  zero.

## Step 4 — SynthSeg setup and smoke test

### 4.1 Installation

- **Tool:** FreeSurfer **7.4.1** `mri_synthseg` (SynthSeg 2.0 robust-mode
  code with the 2.0 segmentation, cortical-parcellation and QC networks).
  - It runs from a minimal `FREESURFER_HOME` at `LYP/third_party/fs_synthseg/`:
    `bin/mri_synthseg` from the freesurfer GitHub repo at tag `v7.4.1`, plus
    `models/`. It is not a full FreeSurfer install.
- **Environment:** `LYP/.venv-synthseg`: Python 3.10.21, tensorflow-cpu
  2.12.0, keras 2.12.0, numpy 1.23.5, nibabel 5.1.0, scipy 1.9.1, surfa 0.4.2.
  These are FreeSurfer 7.4.1's build pins. No other environment was touched.
- **Weights:** fetched from FreeSurfer's public git-annex
  (`surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/repo/annex.git`) by the
  annex keys in the v7.4.1 tree. Each sha256 matches its key:

  | File | sha256 |
  |---|---|
  | synthseg_2.0.h5 | `f190bfd742f450ef3ca2c9df9ed4d2e0232b3a74471da5e51b7770bacdf80c3e` |
  | synthseg_parc_2.0.h5 | `83bb1de76fb6f173c6dacacd433f81209fc6abb1dbc179a930ec06ecabbeb684` |
  | synthseg_qc_2.0.h5 | `fb92e37daa0842bdd8a03bc06a3b36b0d38810e31c714659cedd09119c0abfb7` |
  | bin/mri_synthseg | `38f9b429a7766d330a6174940dfba6701501247f9276e0db668f5e7d6d67599d` |

- **Why not the standalone release:**
  - Its model download links (UCL SharePoint) return 404.
  - Its pinned TF 2.2 stack aborts with `std::bad_alloc` in the parcellation
    network on CPU. The segmentation alone runs.
  - The full FreeSurfer tarball (9.5 GB) downloads here at ~0.75 MB/s.
  - The clone (`third_party/SynthSeg`, `2a2aa3b`) is left in place but unused.
- **Resources:**
  - About 20 s per scan on 16 CPU threads, plus ~20 s of model loading per
    batch. The GPU (RTX 5060) is not usable with TF 2.12.
  - Peak RAM is ~11.5 GB of 15 GB, so SynthSeg must run alone. Running the QC
    script alongside it got the process OOM-killed once. Resuming picked up
    the remaining subjects with no duplicates.
  - The full train+val run (688 scans) would take ~4 h.

### 4.2 Input choice: `mni/`

- The m1 pipeline (`src/mri/m1_preprocess.py`) writes only `mni/` (after
  truncation and min-max) and `vol96/`. **No native-space N4 intermediate was
  saved.**
- The raw baseline downloads are gone: `dataset/mri_raw/` holds only
  manifests, as does the Windows-side copy.
- The only native scans on disk are Path A's follow-up download (ADNI
  MT1/GradWarp/N3m). That covers the selected baseline image for just 50/567
  train and 14/121 val subjects.
- By your rule, the input is therefore **`mni/`: affine-normalised to ICBM152
  2009a (roughly head-size normalised), template-mask skull strip, 5–95th
  percentile truncation, min-max [0, 1].** Every later report must say so.
- **Crop:** each volume is cropped to the fixed template-mask bounding box
  plus 3 voxels. This is lossless (only zeros are removed; the check is
  asserted), and it lets SynthSeg run on a 160×192×160 patch. A centred crop of
  the uncropped grid would clip the cerebellum, because the brain sits
  off-centre in that grid.

### 4.3 Skull-stripped input

- SynthSeg's README lists robustness to "scans with or without preprocessing:
  bias field correction, skull stripping, normalisation".
- During training, skull and extra-cerebral tissue are unlabelled background
  with randomised intensities, so their absence is within the training
  distribution.
- **Consequence:** SynthSeg's "total intracranial" is the sum of all
  non-background posteriors. On our input that is essentially the template
  brain mask filled with brain + CSF (~1.9 L, including ~0.46 L labelled
  "csf"). It is **not an ICV**: its CV is 5.4% vs 11.2% for FreeSurfer ICV,
  and ρ with FreeSurfer ICV is −0.18. The Part 2 W-score plan correctly omits
  ICV for mni/ input.

### 4.4 Smoke test

The sample is 10 CN + 10 AD clinical-train subjects (random, seed 0), plus the
2 brain-age outliers, all train. The list is in `smoke_subjects.csv`; tables
are in `smoke_summary.md`, volumes in `smoke_volumes.csv`.

**Runtime:** median 20.4 s/scan (range 19.3–23.0), 0 failures.

**QC scores (SynthSeg's own):** all 22 subjects score ≥ 0.70 in every
structure group. The lowest is general grey matter (median 0.74, minimum 0.703
on 099_S_4086); hippocampus+amygdala has median 0.85 and minimum 0.79. No
subject is below 0.65 (the flagging cut-off used in the SynthSeg 2.0 paper;
not in the tool's help, so to be verified). The saturated outlier 098_S_4275
still scores ≥ 0.75, because SynthSeg is contrast-agnostic.

**Visual check** (`qc_smoke/overlay_*.png`): 2 CN, 2 AD and both outliers,
each with coronal, axial and sagittal views and hippocampus, amygdala, lateral
and inferior lateral ventricles overlaid.
- The hippocampus follows the medial temporal lobe in all six, with the
  amygdala anterior and the temporal horn lateral.
- The ventricles are traced cleanly.
- The AD cases show visibly larger temporal horns.
- No gross failures.

**Agreement with ADNI FreeSurfer (n = 20)**

| Structure | ρ vs raw FS | ρ vs FS / FS ICV | Comment |
|---|---|---|---|
| Ventricles | 0.95 | **0.98** | excellent |
| Hippocampus | 0.63 | **0.79** | adequate; our space is head-size normalised, so FS/ICV is the fair comparison |
| WholeBrain | 0.02 | **0.74** | same reason |
| MidTemp | 0.47 | 0.45 | weak |
| Entorhinal | 0.46 | 0.41 | weak |
| Fusiform | 0.43 | 0.27 | poor |
| ICV | −0.18 | — | not an ICV (4.3) |

- n = 20 gives wide intervals (ρ = 0.45 has a 95% CI of roughly 0.0–0.75).
- Even so, the cortical parcellation agrees clearly worse than the
  subcortical segmentation.
- Likely causes are the 1.2 mm axis, the truncated and saturated intensities,
  and the template-mask edge cutting cortex. The GE scans show a bright rim at
  the mask boundary.

**Hippocampal volume, CN vs AD (smoke 20):**
- CN 9,954 ± 1,423 vs AD 8,794 ± 1,064 mm³: **12% smaller in AD**,
  Mann–Whitney p = 0.045.
- On the same subjects, FreeSurfer gives 9% smaller (p = 0.054). This AD
  sample is mildly atrophied, and our effect size is similar (Cohen's d ≈ 0.9
  vs ≈ 1.1).
- The inferior lateral ventricles are 55% larger in AD (p = 0.045).
- Entorhinal shows no AD difference (ratio 1.01), consistent with its weak
  agreement with FreeSurfer.
- "Clearly smaller in AD" holds in direction, but not decisively at n = 20.

## Step 5 — Brain-age outlier check

Images:
- `qc_outliers/outlier_<PTID>.png`: each outlier with two age-matched CN
  references (gap ≈ 0), one MPRAGE and one GE.
- `qc_outliers/site016_cn.png`
- `qc_outliers/cn_gap_vs_anatomy.png`

Numbers are in `outlier_qc.csv`. Anatomy is checked independently of our
pipeline: ADNI FreeSurfer ventricles/ICV and hippocampus/ICV, as
age-adjusted z-scores among train CN.

| Subject | Gap | Registration (corr_vs_mni; cohort IQR 0.49–0.56) | Saturated voxels (cohort 5.0%) | FS ventricles z / hippocampus z | Assessment |
|---|---|---|---|---|---|
| **098_S_4275** | −22.8 | 0.54, fine | **23.3%** | +0.6 / −0.6 (average) | **Preprocessing failure: flag as unreliable** |
| **016_S_4097** | −25.0 | 0.48, fine | 4.4% | −1.3 / +1.7 (young-looking) | **Plausible scan: show, with an extreme-value note** |
| 016_S_4121 | −11.4 | 0.54 | **23.2%** | −1.1 / +0.7 | intensity failure: flag |
| 016_S_4638 | +3.8 | 0.54 | 4.6% | +0.5 / −0.5 | normal |
| 016_S_4688 | −1.8 | 0.55 | 5.0% | −1.1 / +1.0 | normal |
| 016_S_4951 | −7.9 | 0.48 | 4.8% | −0.6 / +1.3 | normal, young-looking |
| 016_S_4952 | −10.1 | 0.42 (lowest; outline still tracks) | 4.9% | −1.0 / +1.6 | normal, young-looking |

### 098_S_4275: preprocessing failure

The intensity histogram has no zero floor. Every voxel inside the mask is
non-zero (m1 `mask_fill` = 1.000), 23% are saturated at 1.0, and white matter
is washed out, so grey/white contrast is crushed. Registration is fine. ADNI's
FreeSurfer anatomy is average for age, so nothing supports a brain 23 years
younger. The gap comes from the intensity artefact.

### 016_S_4097: plausible

Intensity, contrast and registration match the references. FreeSurfer
independently shows small ventricles and a large hippocampus for age 70.7, so
the direction is real. The magnitude is not reliable:
- The raw gap is −14.2 years. The α = 0.58 bias correction multiplies it by
  1.7.
- It is a GE IR-SPGR scan, and CN gaps on GE are twice as variable as on
  MPRAGE (SD 7.9 vs 4.1 years, train CN).
- I'd show it on the card with a note such as "younger-looking than expected;
  value beyond the calibrated range", rather than as a precise number.

### Site 016

- All 6 CN are GE Sag IR-SPGR scans.
- One (016_S_4121) is the same intensity failure as 098_S_4275.
- The other five are normal scans, and FreeSurfer agrees four of them look
  young for their age (hippocampus z +1.0 to +1.7). So the site's −8.7 mean is
  partly one artefact and partly a genuinely healthy-looking CN group.
- Scanner contrast may add to it, but that can't be separated with n = 6.

### Wider check

Across all 688 train/val scans, exactly **3** show this lifted-floor failure
(m1 `mask_fill` = 1.000): 016_S_4121 (CN), 098_S_4275 (CN) and 094_S_2216
(EMCI, gap −4.3). All three are GE Sag IR-SPGR. Proposed card rule: **flag
brain age as unreliable when `mask_fill` ≥ 0.9999 or the saturated fraction is
> 10%**. Part 2 should carry the same flag into the SynthSeg table. Their QC
scores look fine, but the inputs are abnormal.

## Files

| File | What |
|---|---|
| `step0_preregistration.md`, `step0_gate.py` | pre-registration (`207d3ea`) and script |
| `step0_tables.md`, `step0_runs.csv`, `step0_shap.csv`, `step0_oof.npz`, `step0_gate.log` | Step 0 results |
| `run_synthseg.py` | resumable SynthSeg runner (crop, per-run CSVs, failure log, test guard) |
| `smoke_subjects.csv`, `smoke_analysis.py`, `smoke_summary.md`, `smoke_volumes.csv`, `smoke_synthseg.log`, `qc_smoke/` | smoke test |
| `outlier_qc.py`, `outlier_qc.csv`, `qc_outliers/` | Step 5 |

Segmentations and cropped inputs are in `dataset/mri_prep/synthseg/`
(git-ignored: DUA data).

Environments used: `.venv-fusion` for Step 0, `.venv-synthseg` for SynthSeg
and the smoke analysis, `.venv-mri` for the outlier QC.
