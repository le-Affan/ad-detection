# PR verification: `fusion-v3` vs `master`

Date 2026-10-04. This file is read-only verification: no model was run.

**Status: partial.** The PR text (`docs/pr_description.md`) was not provided.
It is not in the repo, on any branch, or elsewhere on disk, and it was not
attached to the request.

- **Done in full:** everything that can be checked against the repo alone
  (items 1 and 3–6, and the item 2 diff grouping).
- **Still pending:** the comparison with the PR's "Code changes by area" table,
  each `[verify]` item, and the exact corrections to the PR text. These need
  the PR text.

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

(The pipeline-blueprint commit for this verification follows `4c15da6`.)

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

**Cross-check with the PR's "Code changes by area" table: pending (no PR
text).** Any file the table names that is not in this list, and any area above
the table omits, is a correction to make.

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

## 7. Corrections needed in the PR text

**Pending until the PR text is provided.** From the findings above, check the
PR text for any of these and correct them:
- that it merges or contains `path-a-multivisit`: it does not;
- any `.gitattributes` content: the file doesn't exist;
- that `m1_preprocess.py` changed: it didn't;
- that `m3_train.py` defaults are fully unchanged: the leak check and the two
  `null` keys in `train_config.json` are default-path changes;
- that `0c7074e` contains only the showcase rule: it also contains the
  post-hoc MRI controls;
- that card data is absent from history: it is still in `8de1fcd`;
- that the test set was untouched before fusion-v3: step5 reported a test
  result (accuracy 0.813, macro-F1 0.786).

## 8. Blueprint fix (Part 3)

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
