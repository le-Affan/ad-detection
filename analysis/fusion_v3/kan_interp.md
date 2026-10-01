# Final KAN (K2, config 32) — edge functions and pruning

25 models (outer seeds 42–46 × KAN seeds 42–46), trained on the 569 train OOF stream scores; failed fits: 0.

## MRI-correction edges (additive corrections to the clinical log-probabilities)

| Input | Output logit | Median amplitude (P5–P95) | Shape across 25 models |
|---|---|---|---|
| mri_logit_CN | CN | 0.673 | not consistent (9 up, 0 down, 16 non-monotone of 25) |
| mri_logit_CN | EMCI | 0.308 | not consistent (3 up, 0 down, 22 non-monotone of 25) |
| mri_logit_CN | LMCI | 0.252 | not consistent (0 up, 20 down, 5 non-monotone of 25) |
| mri_logit_CN | AD | 0.935 | monotone decreasing, consistent |
| mri_logit_EMCI | CN | 0.413 | monotone decreasing, consistent |
| mri_logit_EMCI | EMCI | 0.445 | not consistent (18 up, 0 down, 7 non-monotone of 25) |
| mri_logit_EMCI | LMCI | 0.513 | monotone decreasing, consistent |
| mri_logit_EMCI | AD | 0.347 | not consistent (0 up, 13 down, 12 non-monotone of 25) |
| mri_logit_LMCI | CN | 0.960 | not consistent (0 up, 24 down, 1 non-monotone of 25) |
| mri_logit_LMCI | EMCI | 0.428 | not consistent (1 up, 22 down, 2 non-monotone of 25) |
| mri_logit_LMCI | LMCI | 0.188 | not consistent (0 up, 21 down, 4 non-monotone of 25) |
| mri_logit_LMCI | AD | 0.834 | not consistent (20 up, 0 down, 5 non-monotone of 25) |
| mri_logit_AD | CN | 0.624 | monotone decreasing, consistent |
| mri_logit_AD | EMCI | 0.365 | not consistent (0 up, 19 down, 6 non-monotone of 25) |
| mri_logit_AD | LMCI | 0.462 | not consistent (1 up, 0 down, 24 non-monotone of 25) |
| mri_logit_AD | AD | 0.506 | not consistent (0 up, 7 down, 18 non-monotone of 25) |

has_mri flag jump φ(1) − φ(0), median [min, max] over 25 models, per output CN / EMCI / LMCI / AD: CN +0.003 [-0.050, +0.056]; EMCI +0.007 [-0.031, +0.195]; LMCI -0.006 [-0.143, +0.158]; AD +0.001 [-0.077, +0.028]

## Other inputs (edges to the LMCI and AD outputs)

| Input | Output | Median amplitude | Shape |
|---|---|---|---|
| clin_logit_CN | LMCI | 0.773 | monotone decreasing, consistent |
| clin_logit_CN | AD | 0.493 | not consistent (0 up, 6 down, 19 non-monotone of 25) |
| clin_logit_EMCI | LMCI | 0.619 | not consistent (0 up, 0 down, 25 non-monotone of 25) |
| clin_logit_EMCI | AD | 0.441 | not consistent (0 up, 18 down, 7 non-monotone of 25) |
| clin_logit_LMCI | LMCI | 0.833 | monotone decreasing, consistent |
| clin_logit_LMCI | AD | 0.737 | not consistent (0 up, 1 down, 24 non-monotone of 25) |
| clin_logit_AD | LMCI | 0.544 | not consistent (11 up, 0 down, 14 non-monotone of 25) |
| clin_logit_AD | AD | 0.698 | not consistent (1 up, 13 down, 11 non-monotone of 25) |
| AGE | LMCI | 0.544 | monotone decreasing, consistent |
| AGE | AD | 0.934 | not consistent (0 up, 22 down, 3 non-monotone of 25) |
| CDRSB | LMCI | 0.336 | not consistent (1 up, 10 down, 14 non-monotone of 25) |
| CDRSB | AD | 1.202 | not consistent (5 up, 0 down, 20 non-monotone of 25) |
| CogIndex | LMCI | 0.454 | monotone decreasing, consistent |
| CogIndex | AD | 0.761 | monotone decreasing, consistent |
| MMSE | LMCI | 0.677 | not consistent (14 up, 0 down, 11 non-monotone of 25) |
| MMSE | AD | 1.089 | monotone decreasing, consistent |

## Attribution and pruning

| Input | Attribution share, mean ± SD | Kept by prune_input (of 25) |
|---|---|---|
| clin_logit_LMCI | 0.111 ± 0.009 | 25 |
| CDRSB | 0.111 ± 0.011 | 25 |
| clin_logit_EMCI | 0.092 ± 0.017 | 25 |
| MMSE | 0.090 ± 0.021 | 25 |
| clin_logit_AD | 0.090 ± 0.014 | 25 |
| clin_logit_CN | 0.086 ± 0.027 | 25 |
| CogIndex | 0.079 ± 0.022 | 25 |
| AGE | 0.078 ± 0.010 | 25 |
| mri_logit_CN | 0.067 ± 0.011 | 25 |
| mri_logit_LMCI | 0.067 ± 0.013 | 25 |
| mri_logit_AD | 0.065 ± 0.008 | 25 |
| mri_logit_EMCI | 0.063 ± 0.007 | 25 |
| has_mri | 0.001 ± 0.000 | 0 |

Smallest attribution score of any input in any model: 0.0009 (threshold 0.01).
