# PR verification: `docs/pr_description.md` against `fusion-v3`

Date 2026-10-04. This is read-only verification: no model was run.

**Status: complete.** The PR text is now in `docs/pr_description.md` (draft dated
2026-10-04). Sections 1–6 below record the repository facts. Section 7 goes
through each `[verify]` item and other claim, and section 8 gives the exact
corrections to make in the PR text.

**Main finding:** a large part of the PR text describes **`path-a-multivisit`**,
which `fusion-v3` does **not** contain:
- the multi-visit preprocessing code;
- the Path A model code;
- the `m1_preprocess.py` refactor into `process_one()` / `chain_params()`;
- `.gitattributes`.

Before posting, either:
- (A) remove those parts from the PR text, or
- (B) merge `path-a-multivisit` into `fusion-v3` first and re-verify.

The corrections in section 8 assume (A).

## 1. Commit range and ancestry

`git log --oneline master..fusion-v3` lists 31 commits, oldest first:

```
b471660 chore: import Path A visit index for the multi-visit static experiment
0abab24 feat: add the multi-visit train index for the static model (mv1)
f8b7de5 feat: let m3 train on the multi-visit index, with class weighting
ec684ad feat: add the multi-visit static run queue and val-only comparison
a4a0076 feat: add multi-visit static results and report (val only)
02dafac docs: pre-register the v9 reproduction and class-weighting rules
ddea918 feat: add the 8-seed R0/R2 comparison and v9 reproduction verdict
31462eb docs: pre-register the fusion feasibility ablation and build cross-fit folds
8d74423 feat: add cross-fitted ALZ and the fusion feasibility ablation (val/CV only)
3e51c32 docs: add the MRI role plan, pipeline blueprint and Path A roadmap/report
7d916fe feat: add the P1 brain-age model vetting and SFCN smoke test (train only)
1954763 docs: pre-register the DeepBrainNet smoke-test criteria and verify its weights
537ed8f feat: add the DeepBrainNet smoke test results (train only): PASS
2936320 docs: pre-register P2 brain-age bias correction, sanity checks and ablation
80d5af2 feat: add P2 brain-age gap for train+val, sanity checks and ablation: card only
207d3ea docs: pre-register the Step 0 FreeSurfer gate (A vs A0, A' vs A0')
ba4afa6 feat: add the Step 0 FreeSurfer gate results: morphometry does not matter
f1dcf9b feat: add the SynthSeg runner, smoke-test analysis and brain-age outlier QC scripts
ad81ecc feat: add Part 1 SynthSeg smoke test, brain-age outlier QC and report
1de9120 docs: pipeline blueprint v2 (evidence log, MRI stream as volumes/ICV + ALZ, KAN plan)
1c4eba1 feat: build the KAN fusion dataset (...) and apply the brain-age reliability rule
8da778c docs: pre-register the KAN fusion experiment (...)
c8788a7 docs: pre-registration addendum: KAN architecture, training, nested tuning, ...
471282a feat: KAN fusion results: CatBoost F is the final decision layer (...)
8f351b6 freeze: fusion-v3 final system (KAN K2 config 32) fitted on 691 train+val; test not yet read
adc4ccb test: single run of the frozen fusion-v3 system (freeze 8f351b6); outputs only
ad32acf docs: fusion-v3 report (config log, CV/val/test, MRI roles, KAN edges) and blueprint v3
0c7074e post-hoc MRI specialist input-set controls (...); card showcase selection rule, ...
8de1fcd feat: assessment card (src/card) and demo notebook (...)
6311595 chore: untrack DUA-restricted card data; strip notebook outputs
4c15da6 docs: final_results.md audited against sources and filled from saved predictions; results_audit.md
```

Since then: `e67bdda` (blueprint wording and the first, partial verification) and the commit adding this file.

`git merge-base --is-ancestor <branch> fusion-v3`, checked for both the local
and the `origin/` ref:

| Branch | Contained in fusion-v3 |
|---|---|
| kan-fusion | yes |
| mri-morphometry | yes |
| brain-age | yes |
| fusion-feasibility | yes |
| multivisit-static-augmentation | yes |
| path-a-multivisit | **no** |

- **path-a-multivisit is not merged.** Its merge base with fusion-v3 is
  `b717aee`, which is also `master`'s head.
- **It has 10 commits that fusion-v3 does not contain:** Path A stages B4–C2,
  including `50b367c`, `50e1a77`, `1f8ddd2`, `b06a91c` and `a72f68d`.
- **fusion-v3 carries one file from it:** `b471660` imports
  `dataset/mri_prep_patha/visits_index.csv` as a plain file, not as a merge.
- **Docs written on that branch are added here directly:**
  `docs/path_a_roadmap.md` and `docs/path_a_stages_abc_report.md`, both from
  `3e51c32`.
- **A PR description that says it includes or merges path-a-multivisit is
  wrong.**

## 2. Diff by area

`git diff --stat master..fusion-v3`: **423 files changed, 58,121 insertions,
18 deletions.** 420 files are added and 3 modified (`.gitignore`,
`src/mri/dataset.py`, `src/mri/m3_train.py`).

| Area | Files | Contents |
|---|---|---|
| `analysis/fusion_v3/` | 166 | specialists, stacking, KAN runs (`kan_runs/`), freeze (`frozen/`), test run (`frozen_run/`), report, post-hoc controls, audit |
| `analysis/mv/` | 88 | multi-visit static experiment runs and logs |
| `analysis/fusion_feasibility/` | 64 | cross-fit folds, ALZ, ablation |
| `analysis/brain_age/` | 30 | P1 vetting, DeepBrainNet smoke test, P2 gap and ablation |
| `analysis/kan_fusion/` | 26 | fusion dataset, pre-registration, CatBoost and KAN stages, report |
| `analysis/mri_morphometry/` | 26 | Step 0 gate, SynthSeg smoke test, outlier QC |
| `src/card/` | 6 | `__init__.py`, `build_card_data.py`, `build_gradcam.py`, `build_kan_terms.py`, `render.py`, `showcase.py` |
| `src/mri/` | 4 | modified: `dataset.py`, `m3_train.py`; added: `mv1_index.py`, `test_multiscan.py` |
| `docs/` | 6 | `final_results.md`, `mri_role_refined_plan.md`, `mv_static_report.md`, `path_a_roadmap.md`, `path_a_stages_abc_report.md`, `pipeline_blueprint.md` |
| `demo/` | 2 | `README.md`, `assessment_demo.ipynb` (outputs stripped) |
| `dataset/mri_prep_mv/` | 3 | `train_scans.csv`, `train_scans_excluded.csv`, `train_scans_summary.json` |
| `dataset/mri_prep_patha/` | 1 | `visits_index.csv` |
| repo root | 1 | `.gitignore` |

**Not in the diff:** `src/mri/m1_preprocess.py`, `.gitattributes`,
`dataset/mri_prep/` and `analysis/kaggle_output/` are all unchanged; see items
3 and 4.

**Cross-check with the PR's "Code changes by area" table:** see section 7.
The table names 8 files that are not in this diff, all on `path-a-multivisit`:
`p1_visits.py`, `p2_preprocess.py`, `p3_index.py`, `dataset_visits.py`,
`model_temporal.py`, `p4_temporal_c1.py`, `p5_train_temporal.py` and
`test_path_a.py`. It also names `m1_preprocess.py`, which is unchanged, and
`.gitattributes`, which is absent. It omits `analysis/mv/`,
`dataset/mri_prep_mv/`, `dataset/mri_prep_patha/visits_index.csv`,
`src/mri/dataset.py`, `src/mri/mv1_index.py` and `src/mri/test_multiscan.py`.

## 3. Shared-behaviour files

**`src/mri/m1_preprocess.py`: no diff.** `git diff master..fusion-v3 --
src/mri/m1_preprocess.py` is empty, so `process_one` and its outputs are
unchanged by construction. A before/after run on 3 volumes would compare
identical code, so it was not needed and was not run.

**`src/mri/m3_train.py`: modified, 103 insertions and 18 deletions.**

Options that are off by default:
- `--train-index PATH` (default `None`) swaps the train split for a
  one-row-per-scan index (`mv1_index.py`). Val and test still come from
  `mri_index.csv`.
- `--class-weight {none,inverse}` (default `none`): `inverse` weights the DEMO
  MSE per sample by N / (4·n_class) from the train rows. The weights apply to
  the train loss, the val loss and the gradient-balance measurement.
- With both off, the loss is unchanged:
  - with no ALZ positive weight, the original `dual_head_loss` path is used;
  - with an ALZ positive weight, `demo_mse(..., class_w=None)` returns exactly
    `F.mse_loss`.

Changes that also apply with default settings:
1. **Subject-disjointness check on every run.** `assert_subject_disjoint`
   raises `RuntimeError` if any PTID or RID is shared between train, val and
   test. It reads index metadata only. With the default index it passes, so
   training proceeds as before; a run whose index leaks subjects now aborts.
2. **`--limit` selects by subject, not by row.** It keeps the first N unique
   PTIDs. With the default one-scan-per-subject index these are the same rows
   as before.
3. **Console output:** the "subjects" line reports unique train PTIDs, which
   equals the row count for the default index. The loss line adds
   "class-weighted " only when class weighting is on.
4. **`train_config.json` gains two keys,** `train_index_info` and
   `class_weights`, both `null` by default. This is an output-format change:
   extra keys, with no existing key altered.
5. **Internal change:** the gradient-balance batches now carry labels. The
   loss is unchanged unless class weighting is on.
6. **New import:** `hashlib`, used only with `--train-index`.

**Verdict for `m3_train.py`:** training behaviour and checkpoints are
unchanged with default flags. The visible differences are the leak check,
which can only abort a leaking run, and the two `null` keys in
`train_config.json`.

**`src/mri/dataset.py` (also shared): 1 change.** With `data_root` set, the
volume file name is `r.scan_key` if the index has that column, otherwise
`r.PTID` as before. The default index has no `scan_key`, so default behaviour
is unchanged.

## 4. Repository files

- **`.gitattributes`:** does not exist on `master` or on `fusion-v3`. Any PR
  claim about its content is incorrect.
- **`.gitignore`:** 32 lines added at the end, nothing removed.
  - `dataset/mri_prep_patha/*` except `visits_index.csv`; `analysis/patha/`.
  - `*:Zone.Identifier` (Windows download stubs).
  - `analysis/mv/**/*.pt`.
  - `analysis/fusion_feasibility/**/*.pt` and
    `analysis/fusion_feasibility/folds/*/vol96`.
  - `demo/card_data/*` and `demo/output/*`: DUA-restricted subject-level card
    data, added in `6311595`.
- **`dataset/mri_prep/` and `analysis/kaggle_output/`:** no diff against
  master, and 843 tracked files on both branches. Tracking is unchanged.

## 5. Quoted hashes

| Hash | Quoted as | What the commit actually is |
|---|---|---|
| `8f351b6` | freeze | Freeze of the final system: KAN K2, config 32, specialists and 25 KANs fitted on 691 train+val; `freeze.md`, `frozen/`, `MANIFEST.sha256`. Its message says "test not yet read" |
| `adc4ccb` | test outputs | The single frozen test run's outputs: `frozen_run/test_*`, scan inputs and brain age for test. Outputs only |
| `0c7074e` | showcase rule | **Two things in one commit:** the post-hoc MRI input-set controls (`mri_controls.py`, `.md`, `.json`) and the card showcase selection rule (`src/card/showcase.py`). It precedes `8de1fcd`, the first card output. Describe it as "showcase rule, committed together with the post-hoc MRI controls" |
| `6311595` | card-data cleanup | Untracks `demo/card_data/` and `demo/output/` and adds them to `.gitignore`; strips the notebook's outputs and widget state; removes the per-subject showcase table and example subject IDs. The files remain in `8de1fcd` in the pushed history |
| `4c15da6` | audit | `docs/final_results.md` audited and filled from saved predictions; `analysis/fusion_v3/results_audit.md` (117 numbers) |

All five exist on `fusion-v3` and `origin/fusion-v3`.

## 6. GitHub visibility

`gh auth status` reports: "You are not logged into any GitHub hosts."
Visibility can't be confirmed. An anonymous API request for the repo returns
404, which is consistent with a private repo.

To check:
```
gh auth login
gh repo view le-Affan/ad-detection --json visibility
```

## 7. Each `[verify]` item and claim in the PR text

| PR claim (line) | Found | Status |
|---|---|---|
| Lineage: contains kan-fusion, mri-morphometry, brain-age, fusion-feasibility, multivisit-static-augmentation and path-a-multivisit (34–36) | All except **path-a-multivisit**: not an ancestor, with 10 unmerged commits (section 1) | correct for 5; **wrong for path-a-multivisit** |
| Commit range `master..fusion-v3` (37) | 33 commits, `b471660` … `4c15da6`, then `e67bdda` and this verification commit | fill in |
| Multi-visit preprocessing: `src/mri/p1_visits.py`, `p2_preprocess.py`, `p3_index.py` (43) | Not in `fusion-v3`, nor on master; they exist only on `path-a-multivisit` | **wrong for this branch** |
| Path A model code: `dataset_visits.py`, `model_temporal.py`, `p4_temporal_c1.py`, `p5_train_temporal.py`, `test_path_a.py` (44) | Same: only on `path-a-multivisit`. `test_path_a.py` has **11** tests there, not 12 | **wrong for this branch**; the count is also off |
| Shared MRI code: `m1_preprocess.py`, `m3_train.py` (45) | `m1_preprocess.py` is unchanged against master. Changed: `m3_train.py`, **`dataset.py`**. Added: **`mv1_index.py`**, **`test_multiscan.py`** | **partly wrong** |
| Experiment harnesses: `analysis/fusion_feasibility/`, `brain_age/`, `mri_morphometry/`, `kan_fusion/`, `fusion_v3/` (46) | All present (64 / 30 / 26 / 26 / 166 files). **`analysis/mv/` (88 files) is missing from the table** | incomplete |
| Card and demo: `src/card/`, `demo/assessment_demo.ipynb`, `demo/README.md` (47) | Present (6 + 2 files) | verified |
| Docs: `docs/` (48) | 6 files: blueprint, final_results, mri_role_refined_plan, mv_static_report, path_a_roadmap, path_a_stages_abc_report. `pr_description.md` is added by the verification commit | verified |
| Repo config: `.gitattributes`, `.gitignore` (49) | **`.gitattributes` does not exist on `fusion-v3` or master.** It was added on `path-a-multivisit` (`4e41ab4`). `.gitignore` is changed | **wrong for `.gitattributes`** |
| Not in the table | `dataset/mri_prep_mv/` (3 files: multi-visit train index), `dataset/mri_prep_patha/visits_index.csv`, `analysis/mv/`, `src/mri/dataset.py`, `src/mri/mv1_index.py`, `src/mri/test_multiscan.py` | missing rows |
| `m1_preprocess.py`: per-scan body moved into `process_one()` and settings into `chain_params()`; [verify] a before/after run (53–55) | The refactor is on `path-a-multivisit`; on `fusion-v3` the file is identical to master. A before/after run is **not possible here**: `process_one()` takes a raw image, and no raw scans are on disk (`dataset/mri_raw/` holds only manifests). Preprocessed vol96/mni volumes are outputs, not valid inputs | **not applicable to this PR**; not runnable |
| `m3_train.py`: subject-disjointness check and new flags; defaults unchanged; [verify] with `--limit 4` (56–58) | Code inspection (no run; models may not be run here): the defaults leave training unchanged. Default-path differences (section 3): the disjointness check runs on every run and aborts only on a leak; `--limit` selects by subject, giving the same rows for the default index; `train_config.json` gains two `null` keys. A `--limit 4` smoke run was not done because it trains a model | verified by inspection; run not done |
| `.gitattributes`: `* text=auto` with binary types (59–61) | Content matches the `path-a-multivisit` file (`* text=auto` plus 16 binary patterns), but the file is not in this PR | **wrong for this branch** |
| `.gitignore`: additions for `dataset/mri_prep_patha/` and the card data (62) | True, but incomplete. Also added: `analysis/patha/`, `*:Zone.Identifier`, `analysis/mv/**/*.pt`, `analysis/fusion_feasibility/**/*.pt`, `analysis/fusion_feasibility/folds/*/vol96` | incomplete |
| Tracking of `mri_prep/` and `kaggle_output/` untouched (63) | No diff; 843 tracked files on both branches | verified |
| Summary 1: multi-visit preprocessing pipeline, bl/m06/m12/m24, 690 subjects (11) | The pipeline code is on `path-a-multivisit`. This branch carries only its visit index (`visits_index.csv`, 2,760 rows) and the multi-visit train index built from it. 690 matches the blueprint's "Path A raw-scan coverage is 690 of 811" | **wrong for this branch** as worded |
| Results table and bullets (70–85) | Every number matches `docs/final_results.md` and `results_audit.md`, which were recomputed from the saved predictions | verified |
| Freeze `8f351b6`, 109 artefacts hashed; test outputs `adc4ccb` (91–93) | `MANIFEST.sha256` has 109 lines; the hashes match (section 5) | verified |
| Original clinical stream accuracy 0.813 (94) | Verified (`models/step5_metrics.json`); macro-F1 0.786 could be added | verified |
| Environments (99–100) | Also needed and present: **`.venv-kan`** (pykan, used for the KAN), missing from the list | incomplete |
| "This PR adds no new subject-level data to the tree" (108–109) | **False.** Against master, the branch adds at least **40 CSVs keyed by RID or PTID**: e.g. `analysis/kan_fusion/fusion_trainval.csv`, `analysis/brain_age/brainage_trainval.csv`, `analysis/fusion_feasibility/alz_crossfit.csv`, `analysis/fusion_v3/scan_inputs_test.csv`, `brainage_test.csv`, `dataset/mri_prep_patha/visits_index.csv`, `dataset/mri_prep_mv/train_scans.csv`. It also adds **10 brain-image PNGs named by subject ID** (`analysis/brain_age/qc_*.png`, `analysis/mri_morphometry/qc_outliers/`, `qc_smoke/`) and per-subject prediction `.npz` files. Only the card data was excluded | **wrong** |
| Card data remains in branch history (109–110) | True: `8de1fcd` | verified |
| Merge request: `0c7074e` is the showcase-selection rule commit (126) | It is, but the same commit also adds the post-hoc MRI controls | imprecise |
| Optional split of the leakage check and `.gitattributes` (127–129) | The leakage check is in this branch's `m3_train.py`; `.gitattributes` is not in this branch | partly wrong |

## 8. Exact corrections to the PR text (option A: post without path-a-multivisit)

1. **Summary, item 1.** Replace "A multi-visit MRI preprocessing pipeline (bl / m06 / m12 / m24, 690 subjects)." with:

   > Multi-visit MRI experiments using the Path A visit index (`dataset/mri_prep_patha/visits_index.csv`) and a multi-visit train index (`src/mri/mv1_index.py`). The multi-visit preprocessing and temporal-model code itself is on `path-a-multivisit`, which is not part of this PR.

2. **Branch lineage.** Replace the `[verify]` sentence with:

   > `fusion-v3` contains `kan-fusion`, `mri-morphometry`, `brain-age`, `fusion-feasibility` and `multivisit-static-augmentation`. It does not contain `path-a-multivisit`; only that branch's visit index and two of its reports are included, as files.

3. **Commit range.** Replace the `[verify]` with:

   > `master..fusion-v3`: 33 commits, `b471660` … plus the verification commit; 423+ files changed, almost all added.

4. **Code changes table.**
   - Delete the rows "Multi-visit preprocessing" and "Path A model code".
   - Replace the "Shared MRI code" row with:

     > `src/mri/m3_train.py`, `src/mri/dataset.py` (modified); `src/mri/mv1_index.py`, `src/mri/test_multiscan.py` (new): optional multi-visit train index and class weighting, a subject-disjointness check, and `scan_key` volume names.

   - Add a row:

     > Multi-visit static experiment | `analysis/mv/`, `dataset/mri_prep_mv/`, `dataset/mri_prep_patha/visits_index.csv` | runs, train index, visit index

   - Change "Repo config" to `.gitignore` only.

5. **Shared-behaviour section.**
   - Delete the `m1_preprocess.py` bullet. The file is unchanged in this PR.
   - Replace the `m3_train.py` bullet with:

     > adds an always-on subject-disjointness check (aborts only if train/val/test share a PTID or RID) and two flags, `--train-index` and `--class-weight`, that are off by default. With default flags the loss and training are unchanged; `--limit` now selects by subject (same rows for the default index), and `train_config.json` gains two keys that are `null` by default. Verified by code inspection, not by a run.

   - Add:

     > `src/mri/dataset.py`: with `data_root` set, the volume file is named by `scan_key` if the index has that column, otherwise by PTID as before.

   - Delete the `.gitattributes` bullet.
   - Extend the `.gitignore` bullet with: `analysis/patha/`, `*:Zone.Identifier`, and `.pt` weights and fold `vol96` symlinks under `analysis/mv/` and `analysis/fusion_feasibility/`.

6. **Test-set integrity.** Change "(accuracy 0.813)" to "(accuracy 0.813, macro-F1 0.786)".
7. **Running it.** Add `.venv-kan` to the list of environments.
8. **Data note.** Replace "This PR adds no new subject-level data to the tree." with:

   > This PR adds per-subject analysis tables (about 40 CSVs keyed by RID/PTID, including test-split scan inputs and brain-age), per-subject prediction files, and 10 subject-named QC brain images, in line with what the repo already tracks. The assessment-card data (`demo/card_data/`, `demo/output/`) is excluded.

9. **Merge request.**
   - Change "the showcase-selection rule commit (`0c7074e`)" to "the commit that fixed the showcase-selection rule before any card output (`0c7074e`, together with the post-hoc MRI controls)".
   - In the optional-split bullet, drop `.gitattributes` (it is not in this branch): "the leakage check in `m3_train.py` is small and general; I can split it into a separate PR".

**If you choose option B instead** (merge `path-a-multivisit` into `fusion-v3`
first): rows 43–45 and the `.gitattributes` bullets become applicable. Re-run
this verification after the merge. The m1 before/after run would then need the
raw baseline scans, which are not on this machine. The test count for
`test_path_a.py` is 11, not 12.
## 9. Blueprint fix (Part 3)

`docs/pipeline_blueprint.md` (v3) contains no literal "untouched" or "has not
been read" statement. Those were in v2 (`1de9120`) and were removed in
`ad32acf`. The only remaining statement implying test was never read before is
line 8, "Test subjects are read once, in the frozen test evaluation." It is
replaced with:

> The original clinical stream (step5) reported a test result (accuracy 0.813,
> macro-F1 0.786) before this work; the fusion-v3 pipeline was evaluated once
> on test, after the freeze commit `8f351b6`.

Nothing else in the blueprint was changed. §6 still says "567 train", which
should be 569; it was left as it is, because the task allows no other edits.
