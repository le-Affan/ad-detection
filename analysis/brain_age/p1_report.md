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
