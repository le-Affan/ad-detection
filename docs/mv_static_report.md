# Multi-visit data for the static 4-class model — report

Branch: `multivisit-static-augmentation` (off `master` at `b717aee`; not pushed).
Date: 2026-09-27. Validation only — the test split has not been read.

## Summary

Path A's follow-up scans were added to the static model's training set as
independent labelled examples: 567 baseline scans became 1,726 scans from the
same 567 train subjects. Model, config and validation set were unchanged.

**The result is negative.** Across 3 seeds, more scans did not improve any
4-class metric. Combined with class weighting, they made it worse.
The extra scans come from the same 567 brains. The model fits them faster
(lower train loss, earlier best epochs, far noisier val loss) without
generalising better to new subjects. Within this cohort the 4-class head's
weakness is not a shortage of scans; the limit is the number of subjects.

A second finding matters for Stage E: v9 lr_3e-5's logged val numbers
(macro-F1 0.349) sit above all three local reproductions of the same config
(0.211–0.304). v9's figure is likely a favourable draw, not the config's
expected performance.

## Data

- **Per-visit labels.** ADNI has no per-visit CN/EMCI/LMCI/AD label.
  - DXSUM's per-visit diagnosis is CN / MCI / Dementia.
  - EMCI/LMCI exists only as the enrollment arm.
- **Follow-up labels.** A follow-up therefore takes the subject's DX_bl, and
  is kept only when that visit's DXSUM diagnosis agrees at the three-class
  level: EMCI and LMCI must still be MCI, CN still CN, AD still Dementia.
- **Exclusions.** 142 follow-ups were excluded. 139 are converters and
  reverters whose diagnosis had changed; 3 had no diagnosis for that visit.
- **Path A's own baseline scan** is not used: it is the same session as the
  static baseline.
- **Label sanity.**
  - About 2% of train subjects oscillate between adjacent categories.
  - There are no CN↔Dementia jumps.
  - DXSUM has no conflicting duplicate rows.

| Train set | Scans | CN | EMCI | LMCI | AD |
|---|---|---|---|---|---|
| Baseline only (v9) | 567 | 23.5% | 38.1% | 20.1% | 18.3% |
| Multi-visit (`mv1_index.py`) | 1,726 | 25.9% | 40.8% | 19.2% | 14.1% |

Adding follow-ups dilutes AD. AD subjects have fewer follow-ups, and 40 of
the 87 train subjects without Path A data are AD. Class weighting was
therefore tested alongside.

**Split discipline.**
- Train rows are re-derived from the clinical split files by RID. Only train
  subjects are written to the index.
- Validation is v9's own set: 121 baseline scans, one per subject.
- `m3_train.py` now aborts on any PTID or RID shared between splits.
- `test_multiscan.py` passes 11/11. This includes a negative test: an
  injected val subject is caught.

## Runs

A 2×2 design at v9 lr_3e-5's config: resnet18w, batch 32, lr 3e-5, wd 1e-2,
60 epochs, patience 15, default augmentation. Seeds 42–44 were used, run
locally on an RTX 5060. The metrics below are from the best-val-loss epoch
(m3's selection rule) on 121 val subjects, as mean ± sd over 3 seeds.

| Arm | Train data | Class weight | acc4 | Macro-F1 | Bal. acc | AD AUC | DEMO MAE | Best epoch |
|---|---|---|---|---|---|---|---|---|
| R0 `bl_non` | baseline | none | 0.317 ± 0.019 | 0.254 ± 0.047 | 0.290 ± 0.026 | 0.769 ± 0.023 | 0.522 | 8.3 |
| R1 `mv_non` | multi-visit | none | 0.325 ± 0.072 | 0.240 ± 0.082 | 0.286 ± 0.070 | 0.770 ± 0.047 | 0.513 | 4.0 |
| R2 `bl_inv` | baseline | inverse | 0.331 ± 0.014 | **0.279 ± 0.052** | **0.314 ± 0.027** | 0.764 ± 0.039 | 0.542 | 9.0 |
| R3 `mv_inv` | multi-visit | inverse | 0.264 ± 0.014 | 0.217 ± 0.055 | 0.271 ± 0.019 | 0.742 ± 0.027 | 0.587 | 1.7 |
| v9 lr_3e-5 (Kaggle, 1 run) | baseline | none | 0.364 | 0.349 | 0.360 | 0.819 | 0.526 | 15 |

Per-class F1, as means over seeds:

| Arm | CN | EMCI | LMCI | AD |
|---|---|---|---|---|
| R0 `bl_non` | 0.097 | 0.453 | 0.290 | 0.176 |
| R1 `mv_non` | 0.020 | 0.478 | 0.289 | 0.174 |
| R2 `bl_inv` | 0.108 | 0.458 | 0.312 | 0.238 |
| R3 `mv_inv` | 0.031 | 0.327 | 0.278 | 0.232 |

**Other readouts.** These come from `compare.py`, which re-scores every
`best.pt` in fp32.

| Arm | 104-subject Path A subset, macro-F1 | Fitted-cutpoint macro-F1 ceiling |
|---|---|---|
| R0 | 0.233 | 0.408 |
| R1 | 0.217 | 0.409 |
| R2 | 0.252 | 0.412 |
| R3 | 0.215 | 0.354 |
| v9 | 0.344 | 0.501 |

- The fitted ceiling refits the 3 cutpoints on the same 121 subjects, so it
  is an upper bound, not a score.
- Re-scoring reproduces every logged number, including v9's.

## Reading (pre-registered: differences under ~0.05 are noise at n = 121)

1. **Multi-visit data alone has no effect.** R1 vs R0: every metric is
   within ±0.015, and seed spread doubles.
2. **Multi-visit data with class weighting is worse.** R3 vs R2:
   - acc4 is −0.067 and macro-F1 −0.063. Both are past the noise threshold,
     and the losses are consistent across seeds.
   - EMCI F1 collapses from 0.46 to 0.33.
   - The fitted ceiling also drops (0.41 to 0.35), so the damage is in the
     representation, not only the binning.
3. **Class weighting alone gives a small gain.** R2 vs R0: macro-F1 +0.025,
   balanced accuracy +0.024, and AD F1 +0.06. This is within noise and
   directionally consistent, so it is worth keeping in mind for the static
   model independently of this experiment.
4. **Why the extra scans don't help.**
   - Multi-visit runs end at train loss about 0.5, against about 0.85 for
     baseline-only runs.
   - Their best epochs come at 1–4 instead of 7–13.
   - Their val loss swings about 3× more from epoch to epoch (mean
     |Δval_loss| 1.0 vs 0.33).

   Follow-ups of one subject are near-copies of the same anatomy under the
   same label, so the model can fit subject identity instead of disease
   signal. Effective sample size stays at 567 subjects.
5. **A pipeline difference was ruled out.**
   - Same-session static and Path A volumes of the 103 val subjects have
     identical intensity statistics: brain mean 0.591 vs 0.589, std 0.302 vs
     0.302, foreground fraction 0.218 vs 0.218.
   - Every model scores the same on either copy.
6. **v9's logged result is optimistic.** The same config reproduced locally
   gives macro-F1 0.254 ± 0.047; v9's 0.349 is about 2 sd above that mean.
   - This is a Kaggle T4 vs local GPU comparison, so hardware
     nondeterminism can't be separated from seed luck.
   - Either way, one run is not a stable baseline for this config.

## Consistency with Path A

This matches Path A's C1/C2 conclusion from another direction. Extra visits
of the same subjects did not help the DX_bl 4-class target, whether used as
a sequence (C2 macro-F1 0.20) or as independent scans (here). The later
scans carried signal only for a label that changes over time (dementia at
the last visit).

## Files

| File | Purpose |
|---|---|
| `src/mri/mv1_index.py` | Builds the train-only, one-row-per-scan index and its exclusion log |
| `src/mri/test_multiscan.py` | Split-discipline, label-policy, path and weighting checks |
| `src/mri/m3_train.py` | Adds `--train-index`, `--class-weight`, the disjointness guard and subject-based `--limit`. Defaults are unchanged. |
| `src/mri/dataset.py` | Resolves `scan_key` in `--data-root` mode |
| `analysis/mv/run_queue.sh`, `compare.py` | Run the 2×2 queue and compare on val only |
| `analysis/mv/*/epoch_log.csv`, `train_config.json`, `*.log`, `summary_*.csv`, `preds/` | Run artefacts. Weights are not tracked. |

## Open decisions

- **Stage E.** Nothing from this experiment is recommended for the one-time
  test pass. If the static 4-class model goes to test, the numbers above
  argue for evaluating several seeds rather than v9's single run.
- **Class weighting on the static baseline (R2).** It is a small,
  within-noise gain. Whether to pursue it, with more seeds or alongside the
  DEMO head fix (roadmap stage F), is a separate decision.
- **Branch.** Whether to push `multivisit-static-augmentation`, and whether
  the `m3_train.py` changes (the guard in particular) should reach `master`.
