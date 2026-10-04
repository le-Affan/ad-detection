# PR: Multi-visit MRI pipeline, pre-registered experiments, stream-specialist + KAN fusion, assessment card

Branch: `fusion-v3` → `master`
Draft date: 2026-10-04. Items marked **[verify]** must be checked against `git`
before posting (see the verification prompt).

## Summary

This branch adds, on top of the existing clinical and MRI streams:

1. A multi-visit MRI preprocessing pipeline (bl / m06 / m12 / m24, 690 subjects).
2. A series of pre-registered experiments on whether imaging signals improve
   staging (temporal modelling, extra scans, class weighting, ALZ, brain-age,
   volumes). Each experiment's decision rule was committed before its results.
3. The final fusion system: a clinical specialist and an MRI specialist, fused
   by a residual KAN, frozen and evaluated once on the test split.
4. An assessment card and demo notebook.

**The final system matches the single-stream baselines; it does not beat them.**
Test macro-F1: fusion 0.764, clinical specialist 0.771, with every interval
including zero. The MRI stream's value is AD detection from scan-derived inputs
and imaging evidence on the card, not staging accuracy. Negative results are
reported as results.

## What is and is not in this PR

Included: code, analysis scripts, reports, docs, and tracked result tables.

Excluded on purpose:
- preprocessed volumes, checkpoints, raw ADNI downloads (outside the repo)
- `demo/card_data/` and `demo/output/` (subject-level; gitignored, regenerated
  locally by the scripts documented in `demo/README.md`)

Branch lineage: **[verify]** whether `fusion-v3` contains `kan-fusion`,
`mri-morphometry`, `brain-age`, `fusion-feasibility`,
`multivisit-static-augmentation` and `path-a-multivisit`, or only some of them.
Commit range: **[verify]** `master..fusion-v3`.

## Code changes by area  **[verify file lists with `git diff --stat master..fusion-v3`]**

| Area | Files | What it does |
|---|---|---|
| Multi-visit preprocessing | `src/mri/p1_visits.py`, `p2_preprocess.py`, `p3_index.py` | Visit table from the IDA manifest, parallel preprocessing with the unchanged m1 chain, index of usable visits |
| Path A model code | `src/mri/dataset_visits.py`, `model_temporal.py`, `p4_temporal_c1.py`, `p5_train_temporal.py`, `test_path_a.py` | Masked visit-sequence dataset, temporal heads, frozen-embedding and fine-tuning experiments, 12 masking and shape tests |
| Shared MRI code | `src/mri/m1_preprocess.py`, `m3_train.py` | See the next section |
| Experiment harnesses | `analysis/fusion_feasibility/`, `brain_age/`, `mri_morphometry/`, `kan_fusion/`, `fusion_v3/` | Cross-fitted ALZ, brain-age, morphometry gate, KAN builds, frozen test pipeline |
| Card and demo | `src/card/`, `demo/assessment_demo.ipynb`, `demo/README.md` | Assessment card renderer and notebook, from frozen artefacts only |
| Docs | `docs/` | Blueprint, results tables, reports |
| Repo config | `.gitattributes`, `.gitignore` | LF line endings; ignore rules for generated data |

## Changes that affect shared behaviour (please review these)

- **`src/mri/m1_preprocess.py`:** the per-scan body moved into `process_one()`
  and the settings into `chain_params()`. Intended: CLI and outputs unchanged.
  **[verify]** with a small before/after run on a few baseline scans.
- **`src/mri/m3_train.py`:** adds a subject-disjointness check on train/val/test
  and new flags. Intended: defaults unchanged. **[verify]** with
  `--limit 4`.
- **`.gitattributes`:** `* text=auto` with binary types marked. This prevents
  the CRLF/LF phantom diffs that appeared on re-extraction; it may show
  one-time line-ending noise on a Windows checkout.
- **`.gitignore`:** additions for `dataset/mri_prep_patha/` and the card data.
  Existing tracking of `mri_prep/` and `kaggle_output/` is untouched.

## Results (CV on train; test read once)

Macro-F1, 4 classes. Test has 123 subjects (24 LMCI), so test intervals are
about ±0.07.

| System | CV | Test |
|---|---|---|
| Clinical specialist | 0.744 | 0.771 |
| MRI specialist (subjects with a scan) | 0.424 | 0.438 |
| KAN fusion (final) | 0.756 | 0.764 |
| CatBoost, clinical + MRI inputs | 0.758 | 0.746 |
| CatBoost, all 36 features (reference) | 0.752 | 0.784 |

- KAN fusion − clinical specialist: CV +0.013 [−0.006, +0.034]; test −0.008
  [−0.081, +0.065].
- The MRI specialist detects AD without cognitive tests (AUC 0.880 CV, 0.873
  test), mostly from six regional volume ratios (0.867 alone; age and sex alone
  0.574). It cannot stage MCI (LMCI F1 0.142 CV, 0.071 test).
- Earlier experiments (temporal modelling, extra scans, class weighting, ALZ,
  brain-age, volumes) found no reliable staging gain. The v9 MRI macro-F1
  originally logged as 0.349 is 0.284 ± 0.047 over 8 seeds.
- Full tables and the number-by-number audit: `docs/final_results.md`,
  `analysis/fusion_v3/results_audit.md`.

## Test-set integrity

The fusion pipeline was frozen at `8f351b6` (109 artefacts hashed) before any
test data was read, and the test-time scripts refuse the test split unless the
freeze is committed. Test outputs are in `adc4ccb`. The original clinical stream
had already reported a test result before this work (accuracy 0.813); the
freeze governs the fusion pipeline only.

## Running it

Separate environments live outside the repo (`.venv-mri`, `.venv-fusion`,
`.venv-brainage`, `.venv-deepbrainnet`, `.venv-synthseg`, `.venv-demo`); versions
are recorded in the reports. The demo needs local ADNI data. See
`demo/README.md` ("Regenerating card data") and the branch docs.

## Data note

The repo already tracks raw ADNI tables and subject-named QC images. ADNI's data
use agreement restricts redistribution, so the repo should stay private and be
shared only with people covered by the agreement. This PR adds no new
subject-level data to the tree. Earlier commits on this branch did include some
card data before it was removed (`6311595`); it remains in this branch's history.

## Limitations

- Imaging adds no measured staging accuracy; the stage labels are defined by
  cognitive assessments, and training has about 570 subjects.
- Regional volumes come from ADNI's FreeSurfer tables; a new scan would need its
  own segmentation (SynthSeg is reliable for subcortical structures only).
- Volumes saturate about 5% of brain voxels and use a template-mask skull strip;
  three GE scans fail badly. Scanners are not harmonised.
- Brain-age compresses toward the mean age and uses the released VGG16 model.
- One cohort, no external validation.

## Merge request

- Please use a **merge commit**, not a squash. The reports cite the freeze commit
  and the showcase-selection rule commit (`0c7074e`) as evidence of ordering.
- Optional: the leakage check in `m3_train.py` and `.gitattributes` are small and
  general. I can split them into a separate PR so they don't wait on review of
  the rest.
