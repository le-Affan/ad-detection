# KAN F interpretability (5 all-569 models)

Flat = median over seeds of the severity partial-dependence range over the input's train P5–P95 < 0.1 stage units (addendum K4).

| Input | Stream | s range, median [min, max] | Verdict | Attribution share | Mean rank | Kept by prune_input (of 5) |
|---|---|---|---|---|---|---|
| CDRSB | clinical | 1.20 [0.73, 1.34] | non-trivial | 0.147 | 1.0 | 5 |
| MMSE | clinical | 0.36 [0.21, 0.44] | non-trivial | 0.081 | 2.4 | 5 |
| CogIndex | clinical | 0.33 [0.21, 0.59] | non-trivial | 0.078 | 3.4 | 5 |
| FAQ | clinical | 0.12 [0.06, 0.20] | non-trivial | 0.071 | 4.4 | 5 |
| EcogSPDivatt | clinical | 0.35 [0.23, 0.38] | non-trivial | 0.055 | 7.0 | 5 |
| EcogSPMem | clinical | 0.18 [0.06, 0.38] | non-trivial | 0.055 | 7.2 | 5 |
| Fusiform_ICV | MRI | 0.36 [0.22, 0.43] | non-trivial | 0.052 | 7.4 | 5 |
| EcogSPPlan | clinical | 0.17 [0.14, 0.26] | non-trivial | 0.052 | 8.6 | 5 |
| ADAS13 | clinical | 0.39 [0.08, 0.44] | non-trivial | 0.050 | 8.4 | 5 |
| RAVLT_learning | clinical | 0.23 [0.04, 0.26] | non-trivial | 0.043 | 11.4 | 5 |
| ADAS11 | clinical | 0.26 [0.10, 0.38] | non-trivial | 0.041 | 11.6 | 5 |
| Hippocampus_ICV | MRI | 0.17 [0.13, 0.22] | non-trivial | 0.040 | 12.0 | 5 |
| RAVLT_perc_forgetting | clinical | 0.27 [0.07, 0.40] | non-trivial | 0.038 | 13.4 | 5 |
| PTEDUCAT | clinical | 0.36 [0.22, 0.43] | non-trivial | 0.037 | 13.4 | 5 |
| WholeBrain_ICV | MRI | 0.10 [0.07, 0.17] | flat | 0.036 | 14.2 | 5 |
| Ventricles_ICV | MRI | 0.08 [0.06, 0.25] | flat | 0.035 | 14.4 | 5 |
| Entorhinal_ICV | MRI | 0.17 [0.08, 0.26] | non-trivial | 0.034 | 15.4 | 5 |
| ALZ | MRI | 0.07 [0.06, 0.13] | flat | 0.030 | 16.6 | 5 |
| MidTemp_ICV | MRI | 0.14 [0.05, 0.16] | non-trivial | 0.026 | 17.8 | 5 |

prune_input(threshold=1e-2), no retraining:
- seed 42: kept 19 of 19; dropped []; pruned val macro-F1 0.672
- seed 43: kept 19 of 19; dropped []; pruned val macro-F1 0.735
- seed 44: kept 19 of 19; dropped []; pruned val macro-F1 0.744
- seed 45: kept 19 of 19; dropped []; pruned val macro-F1 0.730
- seed 46: kept 19 of 19; dropped []; pruned val macro-F1 0.732

Smallest attribution score of any input in any seed: 0.0846 (threshold 0.01).
