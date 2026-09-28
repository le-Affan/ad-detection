# P1 — Brain-age model search, vetting and smoke test

Branch: `brain-age` (off `fusion-feasibility` at `3e51c32`; not pushed).
Date: 2026-09-28. Blueprint: `docs/pipeline_blueprint.md` §4.3, §5, §7.
Smoke test: 20 clinical-**train** subjects only. No val/test subject was read,
no full-cohort run, no bias correction, no ablation.

## Verdict

**SFCN (UK Biobank), the top-ranked candidate, is not usable on our volumes.**
Predictions are compressed into 60–71 years, subjects over 80 are
under-predicted by 10–24 years, AD shows no positive gap relative to CN, and
predictions move by up to 7 years under a 20–25% intensity rescale. Making its
input faithful would need a different preprocessing from raw (FSL
BET/FAST/FLIRT, no intensity truncation), which this task says to stop at, and
its 42–82 output range would still cap predictions for our >80s.

---

## Part 1 — Input volumes on this machine

| Location | What exists | Usable baseline |
|---|---|---|
| `dataset/mri_prep/mni/` | 811 full-res volumes (4.0 GB), all 811 passed m1 | **810**: 027_S_4966 failed registration (corr_vs_mni 0.054); 099_S_4086, 021_S_2077, 094_S_4282 are low (0.21–0.24), worth a visual check |
| `dataset/mri_prep/vol96/` | 811 × 96³ model inputs | — |
| `dataset/mri_prep_patha/mni/` | 2,555 visit volumes, same m1 chain | **683** bl (train 475 / val 103 / test 105). 690 Path A subjects: 6 bl not in cohort manifest, 1 REG_FAILED (027_S_4966 again) |
| `dataset/mri_raw/` | scripts and scan-selection CSVs only | **No raw baseline NIfTIs**, confirming the earlier report |
| `LYP/datasetnew_extracted/ADNI/` (outside repo, 180 GB) | Path A raw downloads (IDA `MT1__N3m` / `MT1__GradWarp__N3m`), incl. bl | raw bl for the 683 Path A subjects |

**Coverage of the 811:** 810 have a usable full-res baseline volume now. A model
that needs *raw* input could reach only the 683 Path A baselines; the other 128
(121 never in Path A + 7 without a usable Path A bl) would need their raw scans
re-downloaded from IDA (the IDs are in `dataset/mri_raw/selected_image_ids.txt`).

### What the `mni/` volumes are (measured, checked against `m1_preprocess.py`)

| Property | Value |
|---|---|
| Shape / storage | 189 × 233 × 164, float32; array axes = S, P, R (axis0 I→S, axis1 A→P, axis2 L→R) |
| Voxel spacing | 1.0 × 1.0 × **1.2 mm** (the 1.2 mm is left–right) |
| Space | ICBM152 2009a (via nilearn), affine-only: head→head then brain→brain, ANTs `Affine` ×2. World coords in MNI mm (x −98…98, y −134…98, z −72…116) |
| Skull stripping | ICBM152 **template** brain mask applied after registration (not a subject-specific mask); 20.7% of voxels non-zero |
| Bias correction | ANTs N4, native space, before registration |
| Intensity | clip negatives → `TruncateIntensity` 5–95% over the whole masked volume → min–max to [0, 1]. Background is 0, so the 5th percentile truncates nothing; the 95th **saturates 5.3% of brain voxels at 1.0** (bright white matter) |
| Path A copies | same shape, affine and intensity statistics as the original cohort (checked on 3 subjects) |

The chain in the brief (N4 → reorient → resample → affine to ICBM152 → mask →
clip → 5–95% truncation → min–max) matches `m1_preprocess.py` exactly. Two
details matter for brain-age models: the mask is the template's, and the top
truncation clips white matter.

---

## Part 2 — Candidates

| Candidate | 1. Weights / licence | 2. ADNI in training? (source) | 3. Expected input | 4. Match to our `mni/` | 5. Framework / runs here | 6. Published MAE, training ages | 7. Status |
|---|---|---|---|---|---|---|---|
| **SFCN** (Peng et al., 2021) — `ha-ha-ha-han/UKBiobank_deep_pretrain` | Yes, in repo (11.8 MB). MIT | **No**: UK Biobank only, 12,949 training subjects (repo README; Peng et al. bioRxiv 10.1101/2019.12.17.879346) | UKB `T1_brain_to_MNI`: BET skull strip, FAST bias correction, FLIRT **linear** to FSL MNI152 1 mm (182×218×182); ÷ volume mean; centre crop 160×192×160 | Resample + re-normalise gets the shape and space; skull strip, bias correction, template and intensity (our WM saturation) all differ | PyTorch; runs on the RTX 5060 (8 GB) at batch 1 | 2.14 y (UKB test); ages **44–80**; output bins **42–82** | Repo last commit 2024-09; one checkpoint, no updates |
| **DeepBrainNet** (Bashyam et al., 2020) — `vishnubashyam/DeepBrainNet` | Yes, `DBN_model.h5` in repo. No licence file | **No** (contrary to the brief's note): training set "LifespanCN", 13 studies (ADC, AIBL, BLSA, CARDIA, PAC ×3, PING, PNC, PENN PMC, SHIP, UKB); ADNI-1/2 used only as disease cohorts (*Brain* 143(7):2312, Table 1). **Read via a page summariser; please confirm Table 1 yourself** | Skull-stripped, linearly registered T1 (its slicer assumes a 1 mm MNI grid; 80 axial slices 45–124); intensity × 185 / p97 | Same resample as SFCN; p97-based scaling would hit our saturated WM plateau | Keras `.h5` (TF-1-era); 2D, CPU is fine; TF GPU on Blackwell not tested | MAE not verified here; training ages 3–95 (covers our 55–91) | Repo untouched since 2020-09 |
| **brainageR** (Cole) — `james-cole/brainageR` | Yes. LGPL-3.0 | **No**: 3,377 subjects from AIBL, DLBS, GSP, IXI, NKI-RS, OASIS-1, SALD (README) | **Raw native-space** T1; does its own SPM12 segmentation and normalisation | **Incompatible**: needs raw scans (683/811 on disk) | R + SPM12 + MATLAB, or Octave in the v2.1 Docker image | 3.93 y; ages 18–92 | v2.1 (2022); little recent activity |
| **pyment** SFCN-reg (Leonardsen et al., 2022) — `estenhl/pyment-public` | Current repo ships **2025** weights (`reg-2025`, `multi-2025`), CC-BY-NC-4.0; 2022 weights only in old releases | **Unverified.** 2022 paper: 21 training datasets listed only in Supplementary Table 1; ADNI/AIBL controls were in the *external* test set, "scanners not used in the reference dataset", which suggests no ADNI in training but doesn't prove it. 2025 models: 114,257 scans, dataset list not reachable (medRxiv blocked) | FastSurfer v2 crops (current); 2022: FreeSurfer skull strip, FLIRT 6-dof to MNI152 1 mm, crop [6:173, 2:214, 0:160], min–max [0,1] | **Incompatible**: needs FastSurfer from raw | TensorFlow 2.21 / Keras 3 | 2022 SFCN-reg: 2.47 test / 3.90 external; 2025 reg: 5.19 on ds000030; ages 3–95 | Active (commits 2026-07) |
| **SynthBA** (Puglisi et al., 2024) — `LemuelPuglisi/SynthBA` | Yes (Hugging Face) | **Yes → disqualified**: internal training data aggregates ADNI (784 healthy subjects, 3,345 T1w scans), AIBL, IXI, HCP, CoRR (arXiv 2406.00365, Table I and §III-A) | Does its own SynthStrip + affine to MNI 1 mm; `--skip-prep` for already-aligned inputs | — | PyTorch | 4.7 ± 0.6 y (avg across test sets); ages 6–95 | Active |
| **Westman group** (Dartora et al., 2024) — `westman-neuroimaging-group/brainage-prediction-mri` | Yes (GitHub releases). MIT | **Yes → disqualified**: ADNI, AIBL, AddNeuroMed, GENIC, J-ADNI, UK Biobank (*Front. Aging Neurosci.* 10.3389/fnagi.2023.1303036) | Minimally processed T1 + FSL registration | — | PyTorch | 2.67–3.08 y; ages 32–96 | — |

### Ranked recommendation (no ADNI > compatible input > PyTorch > MAE in our age range)

1. **SFCN** — only candidate that is verified ADNI-free, PyTorch and reachable
   from our existing volumes. **Failed the smoke test (Part 3).**
2. **DeepBrainNet** — ADNI-free per its paper (please confirm Table 1), covers
   ages 3–95, needs only the same resample. Keras/2D, older code; its MAE isn't
   verified here. **The best next smoke test.**
3. **brainageR** — ADNI-free and well documented, but needs raw scans plus an
   SPM12/MATLAB or Octave stack: new preprocessing, 683/811 coverage.
4. **pyment** — strong external MAE in 2022, but the training-set ADNI status
   is unverified, and it needs FastSurfer from raw and TensorFlow.
5. SynthBA, Westman/Dartora: disqualified (ADNI in training).

---

## Part 3 — SFCN smoke test

Environment: `LYP/.venv-brainage` (Python 3.12, torch 2.11.0+cu128, nibabel
5.4.2, scipy 1.18.1). SFCN code and weights in `LYP/third_party/UKBiobank_deep_pretrain`
@ `4e56dc1`, weights sha256 `f9197c92…21ad3`. `.venv-mri` and `.venv-fusion`
untouched.

**Subjects.** 10 CN + 10 AD drawn at random (seed 0) from clinical-train with
`corr_vs_mni ≥ 0.3`. Age at scan = ADSL `AGE` (enrolment) + (MRI date −
clinical baseline date); the gap is 0–63 days.

**Input adaptation (all steps, nothing else):**
1. Resample by world coordinates from our grid onto SFCN's FSL MNI152 1 mm grid
   (182×218×182, affine diag(−1,1,1), origin (90,−126,−72)); trilinear, zero
   outside the field of view. 100% of the brain's intensity mass lands in the
   grid; the 160×192×160 crop keeps 99.7–99.8%.
2. `data / data.mean()` (SFCN's normalisation).
3. Centre crop to 160×192×160 (SFCN's `crop_center`).

Orientation was checked visually (`qc_003_S_4644.png`, `qc_016_S_4963.png`):
axis 0 sagittal, 1 coronal, 2 axial, as in UKB/FSL arrays.

### Results (`smoke_test.csv`)

| Group | n | Age at scan | Predicted | MAE | Mean gap (SD) |
|---|---|---|---|---|---|
| CN | 10 | 74.4 (63.5–89.0) | 66.0 (62.3–70.8) | 8.39 | **−8.39** (7.89) |
| AD | 10 | 71.7 (59.4–83.2) | 65.8 (60.1–70.9) | 7.40 | **−5.92** (6.87) |
| CN aged ≤ 80 (inside SFCN's training range) | 6 | | | 3.32 | |

- Corr(predicted, chronological) = 0.53; SD of predictions across all 20
  subjects is **3.3 years** against ~8 years of true age spread.
- The four CN subjects over 80 (80.7, 83.9, 85.6, 89.0) are predicted 65–71:
  gaps of −10 to −24 years.
- AD's mean gap (−5.9) is *not* above CN's (−8.4). AD is also younger here, so
  this comparison is confounded by the compression, but nothing in the raw
  numbers shows the expected AD > CN shift.
- The output distributions are wide (per-subject SD 3.7–9.7 years).

### Robustness diagnostic (`sfcn_sensitivity.py`, `smoke_sensitivity.csv`)

Perturbations applied *after* SFCN's own normalisation; an in-domain model
should barely move.

| Perturbation | Mean shift | Max \|shift\| | Corr with unperturbed |
|---|---|---|---|
| intensity × 0.8 | −2.45 | 6.96 | 0.711 |
| intensity × 1.25 | −0.80 | 5.53 | 0.463 |
| gamma 1.5 | −0.76 | 3.67 | 0.905 |
| left–right flip | +0.43 | 4.19 | 0.840 |

### Flags

- **Predictions clustered at one age:** yes, 60–71 for everyone.
- **CN MAE far above published:** 8.4 overall against the published 2.14. For
  CN inside the 44–80 training range it's 3.3, which is plausible for an
  external scanner, so part of the failure is the age range.
- **No AD signal:** the AD gap is not positive.
- **Out-of-domain input:** a 20–25% intensity change moves predictions by up
  to 7 years. Our input differs from UKB's in skull stripping (template mask vs
  BET), bias correction (N4 vs FAST), template (ICBM 2009a ANTs affine vs FSL
  FLIRT) and intensity (5–95% truncation saturating WM). None of these can be
  undone from the `mni/` volumes; the WM saturation is lossy.

**Verdict: not usable.** Neither as-is nor with a documented adaptation of
our existing volumes. A faithful input needs a new preprocessing from raw, and
the 42–82 output range would still compress our oldest subjects (16% of the
cohort is over 80; 25% of AD).

---

## Decisions needed before P2

1. **Next candidate.** Recommended: smoke-test **DeepBrainNet** on the same 20
   train subjects with the same resample. It's ADNI-free per its paper and
   covers ages 3–95. Before that, please confirm *Brain* 143(7):2312 Table 1
   yourself, since I read it through a summariser. Alternatives:
   - **SFCN with a faithful input**: build a UKB-style chain from raw (BET,
     FAST, FLIRT, no truncation). This is new preprocessing, reaches 683/811
     subjects without a re-download, and the 82-year ceiling remains.
   - **brainageR** from raw via SPM12/Octave: new stack, 683/811.
   - **Drop brain-age** and go to P3, the blueprint's gate if no suitable
     model exists.
2. **Raw re-download** for the 128 subjects without raw baselines. Needed only
   if a raw-input route is chosen.
3. **027_S_4966** (failed registration): leave its brain-age gap as NaN rather
   than re-registering.
4. **pyment 2025 weights**: whether to pursue verifying their training data
   (email the author, or read the medRxiv supplement in a browser). Only worth
   it if a FastSurfer route becomes acceptable.

---

# DeepBrainNet smoke test

Added 2026-09-28. Clinical-train subjects only; no val/test subject read.

## Step 1 — Weights and input

### Which weight file, and how it was verified

- Repo `vishnubashyam/DeepBrainNet` pinned at `505f4e4` (2020-09-04) in
  `LYP/third_party/DeepBrainNet`. No licence file.
- **The repo has one weight file, `Models/DBN_model.h5`** (Git LFS; fetched
  from GitHub's LFS media endpoint because `git-lfs` isn't installed).
  sha256 `9257e98e9aff88bfcd83acdb1e74ffde762967503ef6df16c9270aa39d9941ea`,
  182,983,936 bytes, **identical to the oid in the LFS pointer**.
- **It is the brain-age regressor, not an AD/MCI classifier:**
  - README: "Convolutional Neural Network trained for age prediction using a
    large (n=11,729) set of MRI scans". 11,729 is the LifespanCN count.
  - `Script/test.sh` uses it to "perform brain age prediction".
  - Its saved config: `Sequential[vgg16 → GlobalAveragePooling2D →
    Dense(1024, relu) → Dropout(0.9) → Dense(1, linear)]`, trained with
    `loss: mean_squared_error`, `metrics: [mae]`. A single linear output with
    MSE is an age regression; an AD/MCI classifier would have a sigmoid or
    softmax output with a cross-entropy loss.
- **History:** the repo only ever held age-prediction files. Commit `5ffbbfd`
  added `DeepBrainNet_{DenseNet169,InceptionResnetv2,Resnet50,VGG16}.h5`;
  `505f4e4` moved those "other models to Box". The Box folder renders its file
  list with JavaScript and couldn't be listed from here. No AD/MCI
  transfer-learning weights were used or downloaded.
- **Discrepancy to note:** the released model's backbone is **VGG16**, not
  the Inception-ResNet-v2 the paper describes as DeepBrainNet. It's the
  repo's official inference model, and its provenance (LifespanCN,
  n = 11,729) is as above. Whether it is the exact network behind the paper's
  published MAE isn't verifiable from the repo.

### Environment

`LYP/.venv-deepbrainnet`: Python 3.12, tensorflow 2.18.1 (CPU), tf_keras
2.18.0 (`TF_USE_LEGACY_KERAS=1` loads the Keras 2.2.4 file), nibabel, scipy,
pillow, matplotlib. **TensorFlow found no usable GPU** (CUDA libraries not
installed for TF; the RTX 5060 is unused), so everything runs on CPU:
**~5 s per scan** (80 slices), ~15 s per subject including the two
perturbation runs.

**Environment check:** the repo's 4 example inputs (`Data/`, ages in
`Sample_Data.csv`) predict 58.1, 46.5, 56.7, 61.8 for ages 57.8, 45.4, 52.2,
58.3 (MAE 2.3). The pipeline reproduces as intended. These examples may be
training subjects, so this is a check of the code, not an accuracy estimate.

### Expected input vs our `mni/` volumes

| | DeepBrainNet (repo scripts + example inputs) | Our `mni/` |
|---|---|---|
| Grid | 182×218×182, 1 mm isotropic, affine diag(−1,−1,1), origin (90, 91, −72): the FSL MNI152 1 mm grid in L-P-S storage | 189×233×164, 1×1×1.2 mm, ICBM152 2009a world coords, S-P-R storage |
| Registration | linear (paper: FLIRT affine to a common atlas) | ANTs affine ×2 to ICBM152 2009a |
| Skull strip | subject-specific (paper: multi-atlas); example brains 26.7% of grid | ICBM152 template mask; 25.6% of the DBN grid after resampling |
| Intensity | raw (int16), smooth bright tail; the script scales by 185/p97 | N4, 5–95% truncation, min–max [0,1]; top 5.3% of brain voxels flat at 1.0 |
| Slices | axial 45–124 (z = −27…+52 mm) | — |

**Adaptations applied: one.** A world-coordinate trilinear resample onto the
example grid. Everything else is the repo's own pipeline, unchanged:
- `× 185/p97`
- slices 45–124
- PIL float → RGB → JPEG
- Keras `load_img` 256×256 nearest-neighbour, then ÷255
- median over slices

No preprocessing from raw is needed.

**Residual differences** that can't be removed from `mni/` (single subject
003_S_4644 vs repo Subject1, after 185/p97; see `dbn_input_compare.png`):
- **Grey matter is darker:** brain median 125 vs 146.
- **Top 3.5% of brain voxels form a plateau at the maximum (193);** the
  example's tail runs smoothly to 273. This is the m1 truncation.
- **The template mask leaves some skull-base tissue** in the lowest slices.
- Anatomical level and orientation of slices 50, 85 and 120 match.

## Step 2 — Pre-registered pass criteria (committed before the smoke test ran)

Smoke-test set: the 20 P1 subjects (10 CN, 10 AD) plus 10 CN and 10 AD more,
2 per within-class age quintile (seed 1). 40 clinical-train subjects,
`corr_vs_mni ≥ 0.3`. Gap = predicted − chronological age at scan, uncorrected.

The model **passes only if all four hold:**

| | Criterion | Computed on |
|---|---|---|
| a1 | Pearson r(predicted, chronological) ≥ 0.6 | CN (n = 20) |
| a2 | SD(predicted) ≥ 0.6 × SD(chronological) | CN (n = 20) |
| b | MAE ≤ 6 years | CN (n = 20) |
| c | mean gap(AD) − mean gap(CN) ≥ 2 years | all 40 |
| d | median over subjects of \|Δ prediction\| ≤ 2 years, separately for intensity × 0.9 and × 1.1 | all 40 |

**How d is applied.** The factor multiplies the volume *after* DeepBrainNet's
own `185/p97` scaling. Scaling *before* it would be cancelled by the p97
normalisation, making the test vacuous. This mirrors the SFCN diagnostic,
where the perturbation was also applied after that model's normalisation.
The max |Δ| is reported descriptively.

## Step 3 — Results

40 clinical-train subjects (`dbn_smoke_test.csv`): the 20 P1 subjects plus 20
added across age quintiles. 7 are over 80 (4 CN, 3 AD). Per-subject age,
prediction, gap and the two perturbed predictions are in the CSV; the plots
are in `dbn_smoke_plots.png`. Scored by `dbn_evaluate.py` into
`dbn_criteria.csv`.

(Before the run, one code fix: pandas 3's `groupby.apply` drops the group
columns, so the added-subject draw now uses `groupby.sample(2, random_state=1)`.
Same rule; no results existed at that point.)

| | Criterion | Value | Threshold | Result |
|---|---|---|---|---|
| a1 | r(predicted, chronological), CN | **0.830** | ≥ 0.6 | PASS |
| a2 | SD(pred) / SD(age), CN | **0.616** (4.8 / 7.8) | ≥ 0.6 | PASS (narrowly) |
| b | MAE, CN | **3.36 y** | ≤ 6 | PASS |
| c | mean gap AD − CN | **+4.12 y** (+3.58 vs −0.54) | ≥ 2 | PASS |
| d | median \|Δ\|, intensity × 0.9 | **0.31 y** (max 1.11) | ≤ 2 | PASS |
| d | median \|Δ\|, intensity × 1.1 | **0.23 y** (max 0.94) | ≤ 2 | PASS |

**Verdict: PASS.** All pre-registered criteria hold.

**Descriptive (not criteria):**

| | n | Age | Predicted | MAE | Gap |
|---|---|---|---|---|---|
| CN | 20 | 73.5 ± 7.8 (61.3–89.0) | 72.9 ± 4.8 (60.8–83.5) | 3.36 | −0.54 ± 4.68 |
| AD | 20 | 72.6 ± 7.5 (56.0–83.2) | 76.2 ± 3.3 (70.5–80.6) | 5.27 | +3.58 ± 6.71 |

- **CN MAE 3.36 matches the published 3.7.** For CN ≤ 80 it's 2.44
  (r = 0.861).
- **Regression toward the mean is present,** as expected before
  correction: CN slope of predicted on chronological age is 0.51.
  - The over-80s are under-predicted (mean gap −6.1, n = 7; predictions
    75–84). This is milder than SFCN, which had a hard 81.5 ceiling and put
    the same 89-year-old at 65.
  - The youngest AD subjects get the largest gaps (56.0 → 78.4,
    59.4 → 74.4).
  - Both patterns are what the P2 bias correction (fitted on CN train only)
    is for.
- **The AD signal survives an age adjustment:** with gap regressed on age
  within CN, the AD − CN residual gap is +3.71. This is descriptive; it is
  not the criterion.
- **Robust to intensity mis-scaling:** ×0.9 and ×1.1 correlate 0.997 with
  the unperturbed predictions. SFCN correlated 0.46–0.71 under a larger
  perturbation.
- **Same 20 P1 subjects, head to head:** r with age is 0.673 for DeepBrainNet
  vs 0.529 for SFCN; the two models' predictions correlate 0.58.

**Caveats for P2:**
- **Small set.** a2 cleared by 0.016, and 20 CN is a small set.
- **Architecture.** The released model is VGG16, not the paper's
  Inception-ResNet-v2 (see Step 1).
- **Input differences.** Our input still differs from its training input: WM
  plateau from m1 truncation, darker GM after 185/p97, template-mask skull
  strip. The model tolerated them here.
- **Runtime.** CPU only: ~5 s per scan, so ~70 min for 811 scans, or ~3.5 h
  with the two perturbation passes (which P2 doesn't need).
