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

---

# Follow-up: v9 reproducibility and class weighting (8 seeds)

Validation only; the test split is not read. Added 2026-09-27.

## Step 1 — checks before training

**a. Validation set identity.**
- v9 lr_3e-5's logged macro-F1 0.349 was computed on the **same 121-subject
  validation set** the mv_static runs use (`mri_index.csv`, split `val`).
- Its epoch log records class supports CN 28 / EMCI 46 / LMCI 24 / AD 23 at
  every epoch, identical to the local set.
- Re-scoring the checkpoint locally reproduces every logged metric to three
  decimals (see b), which a different subject set could not do.
- Path A's 104 subjects are a subset of these 121.
- So 0.349 and R0's 0.254 are directly comparable, and the validation set
  stays unchanged.

**b. The v9 checkpoint scored locally, with no training.**
`analysis/kaggle_output/v9/lr_3e-5/best.pt` is the epoch-15 checkpoint.

| Val set | acc4 | Macro-F1 | Bal. acc | AD AUC | F1 CN | F1 EMCI | F1 LMCI | F1 AD |
|---|---|---|---|---|---|---|---|---|
| 121 (v9's own) | 0.364 | 0.349 | 0.360 | 0.820 | 0.176 | 0.430 | 0.257 | 0.533 |
| 104 (Path A subset) | 0.346 | 0.344 | 0.352 | 0.804 | 0.182 | 0.415 | 0.233 | 0.545 |

- fp32 and AMP give the same results.
- The logged number is real for that checkpoint. The gap to R0 therefore
  lies in training (run-to-run variance), not in evaluation or data.

**c. Which epoch 0.349 came from.**
- It is epoch 15, the best-val-loss epoch, i.e. m3's selection rule.
  All metrics are read from that one checkpoint; there was no per-metric
  epoch picking.
- Per-metric maxima in the same run would have given:
  - macro-F1 0.385 at epoch 29
  - acc4 0.430 at epoch 30
  - bal_acc 0.383 at epoch 23
  - AD AUC 0.832 at epoch 23

  None of these was reported.

## Step 2 — pre-registered decision rules (committed before any Step 3 run)

**Arms.**
- R0 is baseline scans only, no class weighting, at exact v9 lr_3e-5 settings.
- R2 is the same, with `--class-weight inverse`.
- Both run on the same 8 seeds: 42–49. Seeds 42–44 are reused from the
  mv_static runs. Their configs match v9 in every setting; the only
  differences are data location, a no-op `--resume`, and the runtime-measured
  auto-lambdas.

**Metrics.**
- Every metric comes from the best-val-loss epoch of each run.
- All metrics are scored on the 121-subject validation set.

**v9 reproduction.**
- The reportable number for this config is R0's 8-seed mean ± SD of
  macro-F1 at the best-val-loss epoch, on the 121-subject validation set.
- The single Kaggle run (0.349) is reported only as a historical reference.
  Its position within the R0 distribution is stated descriptively: its
  z-score and how many of the 8 seeds reach it.

**Class weighting.** It counts as a real gain only if BOTH hold:
1. The mean paired macro-F1 gain (R2 − R0, same seed) is ≥ 0.05, **or**
   the two-sided 95% t-interval of the 8 paired differences (df = 7,
   t = 2.365) excludes zero.
2. R2 beats R0 on at least 7 of 8 paired seeds. A win is strictly higher
   macro-F1; a tie counts as not a win.

Otherwise the result is reported as **"no reliable gain"**.

## Step 3 — runs

- 16 runs: R0 and R2 × seeds 42–49. The 10 new runs (seeds 45–49) took
  about 2.5 min each, locally on the RTX 5060, with the same config as seeds
  42–44.
- The queue is `SEEDS="45 46 47 48 49" ARMS="bl:none bl:inverse"
  analysis/mv/run_queue.sh`.
- The rules are applied by `analysis/mv/seeds8.py`. Its outputs are
  `seeds8_runs.csv` and `seeds8_paired.csv`.

## Step 4 — results

All numbers are mean ± SD over 8 seeds, at the best-val-loss epoch, on the
121-subject validation set.

| Arm | acc4 | Macro-F1 | Bal. acc | F1 CN | F1 EMCI | F1 LMCI | F1 AD | AD AUC | Best epoch |
|---|---|---|---|---|---|---|---|---|---|
| R0 (v9 settings) | 0.332 ± 0.035 | 0.284 ± 0.047 | 0.302 ± 0.033 | 0.190 ± 0.132 | 0.451 ± 0.054 | 0.267 ± 0.050 | 0.226 ± 0.122 | 0.788 ± 0.027 | 11.3 |
| R2 (+ class weighting) | 0.351 ± 0.045 | 0.316 ± 0.071 | 0.336 ± 0.046 | 0.190 ± 0.192 | 0.456 ± 0.060 | 0.299 ± 0.039 | 0.317 ± 0.132 | 0.787 ± 0.037 | 11.4 |
| v9 Kaggle run (historical, 1 run) | 0.364 | 0.349 | 0.360 | 0.176 | 0.430 | 0.257 | 0.533 | 0.819 | 15 |

Paired macro-F1, R2 − R0, same seed:

| Seed | 42 | 43 | 44 | 45 | 46 | 47 | 48 | 49 |
|---|---|---|---|---|---|---|---|---|
| R0 | 0.304 | 0.211 | 0.247 | 0.244 | 0.314 | 0.272 | 0.333 | 0.344 |
| R2 | 0.338 | 0.239 | 0.261 | 0.354 | 0.259 | 0.374 | 0.436 | 0.263 |
| R2 − R0 | +0.033 | +0.028 | +0.013 | +0.110 | −0.055 | +0.102 | +0.104 | −0.080 |

- Mean difference: +0.032 (SD 0.072).
- 95% CI: [−0.029, +0.092], two-sided t, df = 7.
- R2 wins on 6 of 8 seeds.

## Verdicts (against the Step 2 rules)

**Class weighting: no reliable gain.**
- Rule (1) is not met. The mean gain is +0.032, below 0.05, and the 95% CI
  includes zero.
- Rule (2) is not met: R2 wins on 6 of 8 seeds, and 7 were required.
- R2's means are higher on acc4, macro-F1, balanced accuracy, LMCI F1 and
  AD F1 (+0.09). But the seed-to-seed variation, including two seeds where
  R2 loses by 0.055 and 0.080, is as large as the effect.
- This result supersedes point 3 of the "Reading" section above.

**v9 reproduction: the 0.349 is a real checkpoint score and an upper-tail
run, not the config's expected performance.**
- The checkpoint reproduces exactly (Step 1b), on the same validation set
  (Step 1a), at the correct selection epoch (Step 1c).
- Eight local runs of the same settings give 0.284 ± 0.047, ranging from
  0.211 to 0.344.
- None of the 8 reaches 0.349; it sits at z = +1.38.
- Whether Kaggle-vs-local hardware adds to that gap cannot be separated from
  seed variance with one Kaggle run. The variance alone (SD 0.047) is enough
  to produce it.
- The 3-seed R0 figure used earlier in this report (0.254 ± 0.047) is
  superseded by the 8-seed figure.

## The number to use in place of 0.349

**Static 4-class MRI model (v9 lr_3e-5 settings): validation macro-F1
0.284 ± 0.047.** This is the mean ± SD over 8 seeds, at the best-val-loss
epoch, on the 121-subject validation split (`mri_index.csv`, CN 28 / EMCI 46
/ LMCI 24 / AD 23).

The companion numbers on the same basis are:
- acc4 0.332 ± 0.035
- balanced accuracy 0.302 ± 0.033
- AD AUC 0.788 ± 0.027

Where the single run is cited, label it as "one Kaggle run, macro-F1 0.349;
upper tail of an 8-seed distribution".
