# Frozen system on test (n = 123; labels CN/EMCI/LMCI/AD [29, 47, 24, 23])

| System | macro_f1 | acc | bal_acc | f1_CN | f1_EMCI | f1_LMCI | f1_AD | ad_auc | macro_f1_3 | acc_3 |
|---|---|---|---|---|---|---|---|---|---|---|
| Clinical specialist | 0.771 | 0.797 | 0.780 | 1.000 | 0.788 | 0.400 | 0.898 | 0.995 | 0.972 | 0.976 |
| KAN fusion (final, K2) | 0.764 | 0.764 | 0.773 | 1.000 | 0.727 | 0.471 | 0.857 | 0.992 | 0.936 | 0.943 |
| CatBoost F (reference) | 0.746 | 0.772 | 0.759 | 1.000 | 0.763 | 0.341 | 0.880 | 0.997 | 0.963 | 0.967 |
| CatBoost R (reference) | 0.784 | 0.813 | 0.791 | 1.000 | 0.804 | 0.432 | 0.898 | 0.993 | 0.972 | 0.976 |
| MRI specialist (MRI-only mode), scan subjects | 0.438 | 0.504 | 0.472 | 0.542 | 0.554 | 0.071 | 0.586 | 0.873 | 0.554 | 0.634 |
| Clinical specialist, scan subjects | 0.771 | 0.797 | 0.780 | 1.000 | 0.788 | 0.400 | 0.898 | 0.995 | 0.972 | 0.976 |
| Majority class (EMCI), scan subjects | 0.138 | 0.382 | 0.250 | 0.000 | 0.553 | 0.000 | 0.000 | 0.500 | 0.244 | 0.577 |

KAN ADD-head AUC 0.992

- KAN − Clinical specialist: -0.0077 [-0.0808, +0.0650]
- KAN − CatBoost F (reference): +0.0177 [-0.0448, +0.0833]
- KAN − CatBoost R (reference): -0.0198 [-0.0758, +0.0398]

Errors clinical → fusion: {'LMCI_to_AD': [3, 3], 'AD_to_LMCI': [1, 2], 'LMCI_to_EMCI': [13, 9], 'EMCI_to_LMCI': [7, 13]}

- severity Clinical specialist: means [0.04, 1.27, 1.66, 2.85], monotonic True, ρ stage 0.888, ρ CDRSB 0.905
- severity KAN fusion (final, K2): means [0.0, 1.4, 1.77, 2.87], monotonic True, ρ stage 0.884, ρ CDRSB 0.901
- severity CatBoost F (reference): means [0.05, 1.28, 1.64, 2.86], monotonic True, ρ stage 0.883, ρ CDRSB 0.901
- severity CatBoost R (reference): means [0.05, 1.26, 1.64, 2.82], monotonic True, ρ stage 0.891, ρ CDRSB 0.898

- confusion Clinical specialist: [[29, 0, 0, 0], [0, 39, 7, 1], [0, 13, 8, 3], [0, 0, 1, 22]]
- confusion KAN fusion (final, K2): [[29, 0, 0, 0], [0, 32, 13, 2], [0, 9, 12, 3], [0, 0, 2, 21]]
- confusion CatBoost F (reference): [[29, 0, 0, 0], [0, 37, 9, 1], [0, 13, 7, 4], [0, 0, 1, 22]]
- confusion CatBoost R (reference): [[29, 0, 0, 0], [0, 41, 5, 1], [0, 13, 8, 3], [0, 1, 0, 22]]
- confusion MRI specialist: [[13, 15, 0, 1], [5, 31, 3, 8], [1, 13, 1, 9], [0, 6, 0, 17]]
