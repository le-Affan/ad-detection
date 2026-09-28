# Path A (multi-visit MRI) — Stages A, B, C report

Branch: `path-a-multivisit` (head `a72f68d`, pushed). `master` is untouched.
Date: 2026-09-27.

## Summary

The multi-visit pipeline is built and verified end to end: 2,554 usable visits
(bl / m06 / m12 / m24) across all 690 Path A subjects are preprocessed with
the unchanged m1 chain, indexed, and served by a masked sequence dataset to a
temporal model.

The modelling result is negative for temporal signal at this cohort size.
Modelling change across visits does not beat a model that sees only the
latest scan. Seeing a later scan does help, but only on a progression-type
label (dementia at the last visit), where it improves on the baseline-only
model. The test split has not been touched.

## Setup and repository

| Planned | What happened |
|---|---|
| Commit outstanding working-tree changes before starting | Changed. The 308 "modified" files were CRLF-only noise from a Windows-made copy (the repo stores LF). The working tree was restored rather than committed. |
| — | Added `.gitattributes` (`* text=auto`, binary formats marked) so the noise cannot recur. |
| Work on a feature branch, merge via PR | As planned. All work is on `path-a-multivisit`; nothing pushed to `master`, no PR opened. |
| — | Added a Linux environment (`LYP/.venv-mri`, outside the repo); the only existing one was a Windows venv. It uses torch 2.11 built for CUDA 12.8, not the 2.13 CPU build pinned in `requirements-mri.txt`, which is unchanged. |

## Stage A — visit table (`src/mri/p1_visits.py`)

**As planned**
- New module; `dataset/mri_raw/select_scans.py` is untouched.
- Reads `MANIFEST_patha_followup.csv` with an exact schema check and copies
  each file path verbatim — no file is re-selected. 082_S_4224 m24 is pinned
  to I1051225 (not the corrupt duplicate I1051042).
- Dense grid: 690 subjects × 4 slots = 2,760 rows. A missing visit is marked
  `present=False` with a reason; nothing is imputed and no subject is dropped.
- Validation: every OK path exists at its recorded size, every NIfTI header
  opens, every subject has a cohort entry and a clinical split, scan dates
  increase across slots.

**Deviations and findings**
- Five visits masked beyond what the manifest marked missing, each named in code:
  - 4 × `DUPLICATE_SCAN` — the ±60-day date fallback had placed a scan
    already owned by another slot into a second slot (016_S_2284, 018_S_2138,
    068_S_4424, 136_S_4189).
  - 123_S_4806 m12 `DATE_OUT_OF_ORDER` — scanned before its m06.
  - 002_S_4237 m06 `REG_FAILED_LAYOUT` (added during Stage B) — voxel layout
    contradicts the header.
- About 170 subjects have no m24 row at all (rather than a MISSING row);
  these are marked `NO_ROW`.
- Paths are stored relative to the extraction folder (`raw_relpath`), not
  absolute, matching how `m2_index.py` stores paths.
- 41 visits are more than 4 months off their nominal slot (e.g. an "m24"
  scan at 36 months). The model therefore uses actual days since the first
  scan, not slot months.
- Result: 2,555 present visits (against 2,561 OK rows in the manifest).

## Stage B — preprocessing

Files: `src/mri/m1_preprocess.py` (refactor), `src/mri/p2_preprocess.py`,
`src/mri/p3_index.py`.

**As planned**
- Minimal m1 refactor: the per-scan body became `process_one()` and the
  settings record `chain_params()`. m1's CLI, outputs and report columns are
  unchanged.
- `p2_preprocess.py` runs every present visit through the identical chain in
  parallel, writing to `dataset/mri_prep_patha/{mni,vol96,qc}/<PTID>__<slot>`.
  `dataset/mri_prep/` is never written to. Resumable; failures and QC-suspect
  scans are written to their own files, never skipped silently.
- Staged rollout: dry run → baseline reproduction check → full run → index.

| Step | Result |
|---|---|
| Dry run (11 edge-case scans) | 11 OK, 0 suspect |
| Baseline reproduction (20 bl scans vs the original cohort's volumes) | Same-pipeline median r = 0.987 (0.951–0.998). Two identical runs agree at r = 0.997, so the remaining gap is IDA's different derivative file for the same session. |
| Full run | 2,524 scans in 631 min (15.0 s/scan wall, 4 workers × 4 ITK threads); 0 failed, 21 suspect. Planned estimate was 7–10 h. |
| Index (`visits_index.csv`) | 2,554 usable visits, 690 subjects. 4 visits: 490; 3: 194; 2: 6. 7 subjects without a usable baseline; none without any visit. Template correlation median 0.517 (original cohort 0.518). |

**Deviations**
- **128_S_4842 bl**: voxel axes 0 and 1 swapped relative to the header, and
  axis 0 mirrored; registration failed deterministically (corr_vs_mni 0.15 /
  0.12). A full-sphere orientation search (`ants.affine_initializer`) made it
  worse, since a swap is a reflection no rotation reaches. Fixed by a named
  single-scan correction (`RAW_FIXES` in `p2_preprocess.py`) before the
  unchanged chain: corr_vs_mni 0.488, r = 0.91 against the subject's own
  correctly labelled baseline. Without undoing the mirror the template metric
  looked fine (0.49) but the brain was left–right reversed (r = 0.57).
- **027_S_4966 bl** masked as `REG_FAILED` after preprocessing (corr 0.045,
  tilted, scalp inside the mask). It failed identically in the original
  cohort (0.054), so the fault is in the file.
- **The other 20 suspect scans are kept.** Their low scores are consistent
  within each subject and match the original cohort — anatomy (e.g. enlarged
  ventricles), not misregistration.
- Throughput: ~57 s per scan per worker under parallel load (21.5 s alone),
  hence the longer run.

## Stage C — temporal model

Files: `src/mri/dataset_visits.py`, `src/mri/model_temporal.py`,
`src/mri/test_path_a.py`, `src/mri/p4_temporal_c1.py`,
`src/mri/p5_train_temporal.py`, `analysis/patha/`.

**As planned**
- `ADNIVisitSequences`: one item per subject — `(4,1,96,96,96)` volumes, a
  bool mask, days since the first *usable* visit, DX_bl targets imported from
  `dataset.py`. Fixed shape, so default collation works; masked slots are
  never read from disk.
- `TemporalDualHeadNet`: the existing `ResNet3DTrunk` runs on usable visits
  only (gather → encode → scatter), then a mask-aware temporal module
  (`delta` / `gru` / `attn`), then DEMO/ALZ heads with DualHeadNet's output
  keys. `load_trunk()` initialises from a static checkpoint. `model.py` and
  `dataset.py` are unchanged.

**Added**: `test_path_a.py`, 12 checks, all passing. The central one: random
values written into masked slots leave the output unchanged for all three
modules, across every visit pattern in the cohort. Others cover split
coverage, masked slots never opened, the day clock, target parity with the
static dataset, and a real GPU training step.

| Planned | What happened |
|---|---|
| Scripts `p4_train_temporal.py` + `p5_eval_static.py` | Changed: `p4_temporal_c1.py` (C1) and `p5_train_temporal.py` (C2). The static baseline is scored inside C1 on validation; `p5_eval_static.py` is not written yet (belongs with Stage E). |
| Initialise from the v9 checkpoint | Started with lr_5e-5 (lowest val loss, m3's selection rule); switched to lr_3e-5, whose embeddings scored ~0.03–0.04 AUC higher throughout. |
| Target: DX_bl | Added `dx_last` (Dementia at the last usable visit, from `DXSUM.csv`) after C1 showed DX_bl cannot reveal temporal signal. A leakage-free MCI→dementia conversion target was rejected: only 4–5 converters in validation. |
| C1: compare the three modules on frozen embeddings | Done, plus first-visit-only and last-visit-only controls. The first run (lr 1e-3) overfit at epoch 0; rerun at lr 1e-4, and its attention result (AUC 0.872) was discarded. |
| C2: end-to-end fine-tuning of the best module | `gru`, plus a matched last-visit-only control and a DX_bl variant for the fusion layer: 3 configurations × 3 seeds instead of a 2-run budget, since an epoch took ~13 s and all 9 runs finished in 24 min. |
| Same augmentation for all visits of a subject | Refined: geometry (rotation/translation) shared across a subject's visits; intensity jitter per visit. m3's `augment_batch` reused unchanged. |

### Results (validation, 104 subjects)

Dementia at last visit — 24 positives:

| Model | AUC | Balanced accuracy |
|---|---|---|
| Static v9 lr_3e-5 (baseline scan only) | 0.819 | 0.719 |
| C1 frozen: first visit only | 0.808 | 0.700 |
| C1 frozen: last visit only | **0.846** | 0.729 |
| C1 frozen: `gru` | 0.838 | 0.752 |
| C2 end-to-end: `gru`, all visits | 0.833 ± 0.009 | 0.757 ± 0.055 |
| C2 end-to-end: last visit only (control) | 0.836 ± 0.017 | **0.783 ± 0.015** |

Baseline diagnosis (DX_bl) — 17 AD:

| Model | AD AUC | Macro-F1 | DEMO MAE |
|---|---|---|---|
| Static v9 lr_3e-5 | 0.804 | **0.344** | 0.522 |
| C1 frozen: first / last / `gru` | 0.789 / 0.804 / 0.808 | 0.255 / 0.273 / 0.306 | 0.517 / 0.492 / 0.473 |
| C2 end-to-end: `gru`, all visits | 0.813 ± 0.014 | 0.203 ± 0.046 | 0.479 ± 0.038 |

C1 rows are means of 5 seeds, C2 rows means of 3 seeds, all selected at the
best-validation-loss epoch. With 17–24 positives, AUC differences under about
0.05 are within noise, and the ± covers seed variation only.

### Conclusions
1. The multi-visit gain comes from the latest scan, not from change between
   scans: the last-visit control matches or beats the sequence models, frozen
   or fine-tuned.
2. End-to-end fine-tuning did not beat frozen features; runs peak at epochs
   1–11.
3. On the progression-type label, any model that sees the latest scan beats
   the baseline-only model (0.83–0.85 vs 0.819 AUC) — the one place Path A
   adds value, and within noise until tested.
4. The DX_bl temporal variant does not replace static v9 for the 4-class
   output the fusion layer needs (macro-F1 0.20 vs 0.34).

## Verification steps from the plan not carried out
- End-to-end run of the refactored m1 on baseline data: the raw baseline
  scans are not on this machine. The same `process_one()` produced all 2,555
  Path A volumes.
- Regression runs of `m2_index.py` and `m3_train.py --limit 4`: not run; both
  files are unmodified.
- The 8-subject overfit test: not run; superseded by the C1/C2 runs and the
  masking tests.
- `p5_eval_static.py` reproducing v9's logged metrics: deferred to Stage E.

## Open decisions
- **Stage D**: what Path A claims in the report.
- **Stage E**: which checkpoints get the one-time test evaluation. Proposed:
  static v9 lr_3e-5, `gru_dx_last_last`, `gru_dx_last_all`, on the same 106
  test subjects.
- `docs/path_a_roadmap.md` is still untracked.

## Commits on `path-a-multivisit`

| Commit | Content |
|---|---|
| `06f225c` | .gitignore rules for Path A outputs; session 8 handoff |
| `383b5ea` | Stage A visit table |
| `4e41ab4` | `.gitattributes` (LF line endings) |
| `6d0c34a` | Stage B driver, m1 refactor, dry run |
| `7a00814` | Baseline reproduction check; 128_S_4842 fix; 002_S_4237 mask |
| `50b367c` | Stage B4 index, full preprocessing run outputs |
| `50e1a77` | Stage C dataset, temporal model, masking tests |
| `1f8ddd2` | C1 on frozen v9 embeddings |
| `b06a91c` | C1 dementia-at-last-visit target; trunk comparison |
| `a72f68d` | C2 end-to-end fine-tuning and results |
