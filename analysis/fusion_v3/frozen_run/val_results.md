# Frozen system on val (n = 122; labels CN/EMCI/LMCI/AD [28, 46, 25, 23])

| System | macro_f1 | acc | bal_acc | f1_CN | f1_EMCI | f1_LMCI | f1_AD | ad_auc | macro_f1_3 | acc_3 |
|---|---|---|---|---|---|---|---|---|---|---|
| Clinical specialist | 0.941 | 0.943 | 0.931 | 0.982 | 0.939 | 0.864 | 0.979 | 1.000 | 0.982 | 0.984 |
| KAN fusion (final, K2) | 0.874 | 0.877 | 0.874 | 0.947 | 0.863 | 0.727 | 0.958 | 0.999 | 0.957 | 0.959 |
| CatBoost F (reference) | 0.961 | 0.959 | 0.951 | 0.982 | 0.948 | 0.913 | 1.000 | 1.000 | 0.992 | 0.992 |
| CatBoost R (reference) | 0.961 | 0.959 | 0.951 | 0.982 | 0.948 | 0.913 | 1.000 | 1.000 | 0.992 | 0.992 |
| MRI specialist (MRI-only mode), scan subjects | 0.411 | 0.455 | 0.421 | 0.353 | 0.513 | 0.207 | 0.571 | 0.843 | 0.478 | 0.587 |
| Clinical specialist, scan subjects | 0.939 | 0.942 | 0.929 | 0.982 | 0.939 | 0.857 | 0.979 | 1.000 | 0.982 | 0.983 |
| Majority class (EMCI), scan subjects | 0.138 | 0.380 | 0.250 | 0.000 | 0.551 | 0.000 | 0.000 | 0.500 | 0.244 | 0.579 |

KAN ADD-head AUC 1.000

- KAN − Clinical specialist: -0.0667 [-0.1204, -0.0192]
- KAN − CatBoost F (reference): -0.0868 [-0.1449, -0.0403]
- KAN − CatBoost R (reference): -0.0868 [-0.1449, -0.0403]

Errors clinical → fusion: {'LMCI_to_AD': [1, 2], 'AD_to_LMCI': [0, 0], 'LMCI_to_EMCI': [5, 7], 'EMCI_to_LMCI': [0, 3]}

- severity Clinical specialist: means [0.07, 1.12, 1.69, 2.91], monotonic True, ρ stage 0.938, ρ CDRSB 0.854
- severity KAN fusion (final, K2): means [0.04, 1.21, 1.71, 2.91], monotonic True, ρ stage 0.920, ρ CDRSB 0.857
- severity CatBoost F (reference): means [0.08, 1.11, 1.71, 2.88], monotonic True, ρ stage 0.949, ρ CDRSB 0.834
- severity CatBoost R (reference): means [0.07, 1.1, 1.74, 2.89], monotonic True, ρ stage 0.953, ρ CDRSB 0.847

- confusion Clinical specialist: [[27, 1, 0, 0], [0, 46, 0, 0], [0, 5, 19, 1], [0, 0, 0, 23]]
- confusion KAN fusion (final, K2): [[27, 1, 0, 0], [2, 41, 3, 0], [0, 7, 16, 2], [0, 0, 0, 23]]
- confusion CatBoost F (reference): [[27, 1, 0, 0], [0, 46, 0, 0], [0, 4, 21, 0], [0, 0, 0, 23]]
- confusion CatBoost R (reference): [[27, 1, 0, 0], [0, 46, 0, 0], [0, 4, 21, 0], [0, 0, 0, 23]]
- confusion MRI specialist: [[9, 18, 0, 1], [12, 29, 1, 4], [2, 12, 3, 7], [0, 8, 1, 14]]
