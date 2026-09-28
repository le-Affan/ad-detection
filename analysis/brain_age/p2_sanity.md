# P2 sanity checks

## Coverage

| Split | Subjects | With brain age | Missing |
|---|---|---|---|
| train | 569 | 566 | 027_S_4966 (EMCI, m1 registration failed (corr_vs_mni 0.054)); 098_S_4095 (AD, no MRI scan); 126_S_2360 (EMCI, no MRI scan) |
| val | 122 | 121 | 941_S_4187 (LMCI, no MRI scan) |

## 1. CN train: accuracy, raw vs bias-corrected

Bias fit on 133 CN train subjects: predicted = 0.582·age + 29.97

| | MAE | mean error | r with age | SD pred / SD age |
|---|---|---|---|---|
| raw | 3.16 | -0.72 | 0.745 | 0.781 |
| corrected | 3.97 | +0.00 | 0.745 | 1.343 |

## 2. Corrected gap by diagnosis (train)

| Diagnosis | n | Mean BAG | SD | Median | Raw gap mean |
|---|---|---|---|---|---|
| CN | 133 | +0.00 | 5.47 | +0.69 | -0.72 |
| EMCI | 215 | +2.20 | 5.10 | +2.67 | +1.47 |
| LMCI | 114 | +4.02 | 5.13 | +4.01 | +2.05 |
| AD | 104 | +6.80 | 6.26 | +6.34 | +2.84 |

Kruskal–Wallis across the 4 groups: H = 75.99, p = 2.23e-16

## 3. Corrected gap vs age

CN train: r(BAG, age) = 0.000 (p = 1.00). Raw gap in CN: r = -0.626. All train diagnoses: r(BAG, age) = -0.173.

## 4. Site effect (CN train, site = PTID prefix)

44 sites; 6 have ≥ 5 CN subjects. CN mean BAG +0.00, SD 5.47.

| Site | CN n | Mean BAG | z (SE) | z (SD) | Flag |
|---|---|---|---|---|---|
| 016 | 6 | -8.68 | -3.89 | -1.59 | **FLAG** |
| 073 | 6 | -2.95 | -1.32 | -0.54 |  |
| 003 | 6 | -0.58 | -0.26 | -0.11 |  |
| 011 | 5 | -0.25 | -0.10 | -0.04 |  |
| 031 | 5 | +2.53 | +1.03 | +0.46 |  |
| 094 | 5 | +3.96 | +1.62 | +0.73 |  |

Flagged (≥5 CN, |z_SE| > 2): 016. Any site with |mean − overall| > 2 SD of individual CN gaps: none.

## 5. Overlap with existing inputs (train, Spearman)

| Variable | ρ with BAG (all train) | ρ with BAG (CN train) |
|---|---|---|
| ALZ | +0.351 | +0.245 |
| Ventricles | +0.343 | +0.358 |
| Hippocampus | -0.251 | -0.054 |
| WholeBrain | -0.087 | +0.017 |
| Entorhinal | -0.193 | +0.045 |
| Fusiform | -0.209 | -0.108 |
| MidTemp | -0.198 | -0.023 |
| ICV | +0.070 | +0.130 |
| AGE | -0.127 | -0.012 |

In-sample R² of BAG on the 7 volumes + ALZ + AGE (linear, train): 0.399. So about 60% of BAG's variance is not linearly explained by them.

## 6. Standalone signal (corrected gap as the score, 5-fold CV, per-fold correction)

- AD vs CN: AUC 0.797 (n = 237)
- AD vs rest: AUC 0.713 (n = 566)
- EMCI vs CN: AUC 0.620
- LMCI vs CN: AUC 0.707
- For reference, cross-fitted ALZ on the same subjects: AD vs rest AUC 0.764

