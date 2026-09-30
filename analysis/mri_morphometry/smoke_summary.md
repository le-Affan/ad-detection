# SynthSeg smoke test — summary tables

20 clinical-train subjects (10 CN, 10 AD) + 2 brain-age outliers (CN, reported separately). Volumes in mm³ of affine-normalised ICBM152 space.

## QC scores (SynthSeg 2.0 automated QC, 0-1; 0.65 is the flagging cut-off used in the SynthSeg 2.0 paper, Billot et al. PNAS 2023, not stated in the tool's own help -- to verify)

| Structure group | min | median | max | n < 0.65 |
|---|---|---|---|---|
| general white matter | 0.798 | 0.849 | 0.884 | 0 |
| general grey matter | 0.703 | 0.743 | 0.782 | 0 |
| general csf | 0.815 | 0.888 | 0.950 | 0 |
| cerebellum | 0.841 | 0.862 | 0.887 | 0 |
| brainstem | 0.860 | 0.894 | 0.913 | 0 |
| thalamus | 0.808 | 0.850 | 0.871 | 0 |
| putamen+pallidum | 0.845 | 0.875 | 0.916 | 0 |
| hippocampus+amygdala | 0.788 | 0.847 | 0.895 | 0 |

Lowest subject minimum: 099_S_4086 (0.703).

## Agreement with ADNI FreeSurfer (n = 20 smoke subjects)

| Structure | Spearman ρ (raw FS) | Pearson r (raw FS) | Spearman ρ (FS / FS ICV) | SynthSeg mean | FS mean |
|---|---|---|---|---|---|
| Hippocampus | +0.63 | +0.61 | +0.79 | 9,374 | 6,744 |
| Ventricles | +0.95 | +0.98 | +0.98 | 68,421 | 43,748 |
| Entorhinal | +0.46 | +0.40 | +0.41 | 5,854 | 3,458 |
| Fusiform | +0.43 | +0.37 | +0.27 | 23,346 | 17,943 |
| MidTemp | +0.47 | +0.48 | +0.45 | 23,334 | 19,155 |
| WholeBrain | +0.02 | +0.35 | +0.74 | 1,365,832 | 1,013,361 |
| ICV | -0.18 | +0.12 | +nan | 1,894,631 | 1,483,002 |

SynthSeg ICV coefficient of variation 0.054 vs FreeSurfer ICV 0.112.

## CN vs AD (smoke subjects)

| Volume | CN mean ± SD | AD mean ± SD | AD / CN | Mann–Whitney p |
|---|---|---|---|---|
| ss_Hippocampus | 9,954 ± 1,423 | 8,794 ± 1,064 | 0.88 | 4.5e-02 |
| ss_Amygdala | 4,245 ± 546 | 3,911 ± 571 | 0.92 | 2.1e-01 |
| ss_InfLatVent | 2,398 ± 1,129 | 3,717 ± 1,617 | 1.55 | 4.5e-02 |
| ss_Ventricles | 59,488 ± 35,767 | 77,355 ± 39,641 | 1.30 | 3.4e-01 |
| ss_Entorhinal | 5,810 ± 607 | 5,897 ± 508 | 1.01 | 9.1e-01 |
| fs_Hippocampus | 7,053 ± 509 | 6,436 ± 653 | 0.91 | 5.4e-02 |

## Brain-age outliers

| PTID | Hippocampus | Ventricles | InfLatVent | min QC |
|---|---|---|---|---|
| 016_S_4097 | 11,605 | 20,450 | 1,527 | 0.782 |
| 098_S_4275 | 9,162 | 56,497 | 1,683 | 0.751 |

CN smoke range: hippocampus 6,813–11,110, ventricles 24,469–149,116.

