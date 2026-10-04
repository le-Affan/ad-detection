# MRI specialist: input-set controls

**Post-hoc analysis, not in the freeze.** Run after the single test run. It
uses train CV and val only, adds no test numbers, and changes nothing in the
frozen system. Script: `mri_controls.py`; numbers: `mri_controls.json`.

## Setup

- **Model:** the frozen MRI specialist configuration for every input set:
  - multinomial logistic regression, L2, C = 0.1;
  - median imputation and z-scoring fitted on the training rows;
  - brain-age α and β refitted on the training rows' CN subjects.
- **CV:** the step-1 folds. That is StratifiedKFold(5, shuffle, random_state =
  seed) over the 569 train subjects, seeds 42–46. Models are fitted and scored
  on the subjects with a scan (567).
- **Val:** fitted on the 567 train subjects with a scan, scored on the 121 val
  subjects with a scan. Logistic regression is deterministic, so val is one
  number with no seed SD.
- **Reproduction check:** M4 reproduces the frozen specialist's step-1 CV
  out-of-fold probabilities exactly (asserted in the script). Its val numbers
  equal the report's (0.811 / 0.391).
- **Bootstrap:** paired, 2,000 resamples of subjects, rng seed 0. On CV the
  metric is averaged over the 5 seeds inside each resample.

## Results

| Set | Inputs | CV AD-vs-rest AUC | Val AD AUC | CV 4-class macro-F1 | Val macro-F1 | CV F1 CN / EMCI / LMCI / AD |
|---|---|---|---|---|---|---|
| M0 | age + sex only | 0.574 ± 0.009 | 0.558 | 0.189 ± 0.008 | 0.160 | 0.16 / 0.55 / 0.00 / 0.05 |
| M1 | six volumes/ICV only | 0.867 ± 0.005 | 0.781 | 0.379 ± 0.013 | 0.298 | 0.24 / 0.58 / 0.13 / 0.57 |
| M2 | volumes/ICV + age + sex | 0.871 ± 0.004 | 0.781 | 0.412 ± 0.007 | 0.362 | 0.37 / 0.59 / 0.13 / 0.56 |
| M3 | volumes/ICV + ALZ + age + sex | 0.876 ± 0.005 | 0.787 | 0.415 ± 0.007 | 0.373 | 0.37 / 0.59 / 0.13 / 0.57 |
| **M4** | **M3 + corrected brain-age gap (frozen specialist)** | **0.880 ± 0.005** | **0.811** | **0.424 ± 0.005** | **0.391** | 0.39 / 0.60 / 0.14 / 0.56 |
| M5 | ALZ + brain-age gap only | 0.788 ± 0.001 | 0.794 | 0.304 ± 0.008 | 0.284 | 0.15 / 0.57 / 0.01 / 0.49 |

CV values are mean ± SD over 5 seeds. For reference, the majority class
(EMCI) has a macro-F1 of about 0.14.

**Paired bootstrap differences (95% CI)**

| Comparison | CV AD AUC | Val AD AUC | CV macro-F1 | Val macro-F1 |
|---|---|---|---|---|
| M4 − M0 (scan inputs added to age + sex) | +0.306 [+0.239, +0.370] | +0.253 [+0.114, +0.401] | +0.235 [+0.196, +0.274] | +0.231 [+0.150, +0.309] |
| M4 − M2 (ALZ + brain age added to volumes + age + sex) | +0.009 [−0.003, +0.020] | +0.030 [+0.002, +0.060] | +0.012 [−0.010, +0.036] | +0.029 [−0.051, +0.110] |

## Plain reading

- **Age and sex give almost no AD detection.** On their own they reach a CV
  AUC of 0.574 (val 0.558), against 0.5 for chance.
  - Measured as AUC above chance (0.5), age and sex alone reach 0.074 of the
    0.380 the full specialist reaches, about a fifth. The scan-derived inputs
    supply the rest: M4 − M0 is +0.306 AUC on CV, CI [+0.239, +0.370].
  - Adding age and sex to the volumes moves the AUC only from 0.867 to 0.871.
    Their main contribution is to 4-class staging, mostly through CN F1
    (0.24 → 0.37), not to AD detection.
- **The six FreeSurfer volume ratios carry almost all of the AD-detection
  signal.** Volumes alone reach a CV AUC of 0.867. The frozen specialist reaches
  0.880.
- **ALZ and brain age add little once the volumes are present.**
  - On CV, M4 − M2 is +0.009 AUC with a CI that includes zero.
  - On val it is +0.030 [+0.002, +0.060]. That is the only interval here that
    excludes zero, and it rests on 121 subjects (23 AD).
  - On their own, ALZ and brain age reach an AUC of 0.788 (val 0.794). Their
    signal is real, but it largely overlaps with the volumes.
- **Staging (4-class macro-F1) comes from the same source.** Age and sex alone
  give 0.19, close to the majority class. Volumes alone give 0.38, and the full
  specialist 0.42. LMCI stays near chance (F1 ≤ 0.14) with every input set.
- **Conclusion:** the MRI specialist's AD detection is a genuine imaging
  result, and it comes mainly from the regional volumes. It is not an age
  effect.
- **Caveat:** ALZ on train is cross-fitted and brain age comes from an
  external model, so neither has seen these labels. The FreeSurfer volumes come
  from the ADNI tables and are not learned.
