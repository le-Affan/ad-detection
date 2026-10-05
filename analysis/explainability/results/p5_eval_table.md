# Explanation evaluation (P5; pre-registered thresholds, train primary, val replication)

| Explanation | stability | plausibility: hippocampus | plausibility: ventricles | plausibility: CDRSB | plausibility: MMSE | faithfulness | sanity: cascading randomisation | sanity: region deletion | overall |
|---|---|---|---|---|---|---|---|---|---|
| CNN region attribution | PASS | PASS | PASS | N/A | N/A | PASS | FAIL | FAIL | WEAK |
| MRI-specialist SHAP | WEAK | PASS | FAIL | N/A | N/A | PASS | N/A | N/A | WEAK |
| Clinical SHAP | PASS | N/A | N/A | FAIL | WEAK | PASS | N/A | N/A | WEAK |
| KAN Shapley | PASS | N/A | N/A | WEAK | WEAK | PASS | N/A | N/A | WEAK |

Details:
- **CNN region attribution** — WEAK; weak/failed: sanity: cascading randomisation (FAIL), sanity: region deletion (FAIL)
  - stability: PASS (train 0.91, val 0.93)
  - plausibility: hippocampus: PASS (train 1.00, val 1.00)
  - plausibility: ventricles: PASS (train 0.98, val 0.96)
  - faithfulness: PASS (val AD (n=23): deletion -0.14, insertion 0.12; train AD PASS (selection-biased))
  - sanity: cascading randomisation: FAIL (median Spearman after head + layer4 0.81)
  - sanity: region deletion: FAIL (top-region minus control drop -0.12)
- **MRI-specialist SHAP** — WEAK; weak/failed: stability (WEAK), plausibility: ventricles (FAIL)
  - stability: WEAK (train 0.60, val 0.73)
  - plausibility: hippocampus: PASS (train 0.85, val 0.83)
  - plausibility: ventricles: FAIL (train 0.12, val 0.15)
  - faithfulness: PASS (train PASS (deletion -0.08, insertion 0.08); val PASS)
- **Clinical SHAP** — WEAK; weak/failed: plausibility: CDRSB (FAIL), plausibility: MMSE (WEAK)
  - stability: PASS (train 0.90, val 0.90)
  - plausibility: CDRSB: FAIL (train 0.36, val 0.33)
  - plausibility: MMSE: WEAK (train 0.71, val 0.78)
  - faithfulness: PASS (train PASS (deletion -0.19, insertion 0.23); val PASS)
- **KAN Shapley** — WEAK; weak/failed: plausibility: CDRSB (WEAK), plausibility: MMSE (WEAK)
  - stability: PASS (train 0.88, val 0.88)
  - plausibility: CDRSB: WEAK (train 0.60, val 0.59)
  - plausibility: MMSE: WEAK (train 0.79, val 0.84)
  - faithfulness: PASS (train PASS (deletion -0.18, insertion 0.19); val PASS)
