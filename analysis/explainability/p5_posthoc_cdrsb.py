"""POST-HOC check (LYP/.venv-fusion), added 2026-10-05 after the P5 results: NOT pre-registered.

The pre-registered plausibility criterion for CDR-SB (sign of the AD SHAP value relative to the CN-train median)
failed for clinical SHAP (0.36 on train). A likely reason is that CDR-SB's CN-train median is near zero, so most
EMCI / LMCI subjects count as "above the median", while a multiclass model pushes them toward EMCI / LMCI rather
than AD. This check asks a different question: does the AD-class SHAP value of CDR-SB rise with the CDR-SB value?
Statistic: Spearman correlation between the raw CDR-SB value and the 5-seed mean CDR-SB SHAP value on the AD
centred target (the same values P5 used), over subjects with a recorded CDR-SB; bootstrap 95% CI (2,000
resamples, rng 0). Train and val. It does not replace or change the pre-registered criterion, its threshold or
the P5 verdict; it is reported next to it, labelled post-hoc.
Output: results/p5_posthoc_cdrsb.json.
"""
import json
import os
import sys

import numpy as np
from catboost import CatBoostClassifier, Pool
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import tab_data as TD  # noqa: E402

SEEDS = (42, 43, 44, 45, 46)
AD = 3


def main():
    C.verify_manifest()
    d = TD.trainval()
    top = TD.columns()["clinical_top12"]
    phi = np.zeros((len(SEEDS), len(d), 4))
    for si, s in enumerate(SEEDS):
        m = CatBoostClassifier()
        m.load_model(os.path.join(C.FROZEN, f"clinical_{s}.cbm"))
        sv = m.get_feature_importance(Pool(d[top[str(s)]]), type="ShapValues")      # (n, 4, 13)
        if "CDRSB" in top[str(s)]:
            phi[si] = sv[:, :, top[str(s)].index("CDRSB")]
    phi = phi - phi.mean(2, keepdims=True)                                           # centred class targets
    p_ad = phi[:, :, AD].mean(0)
    out = {"label": "POST-HOC (not pre-registered)", "statistic": "Spearman(raw CDR-SB, CDR-SB SHAP on the AD centred target)",
           "pre_registered_criterion": json.load(open(os.path.join(C.RESULTS, "p5_rows.json")))["clinical_shap"]["plausibility"]["CDRSB"], "pre_registered_verdict": "FAIL (unchanged)"}
    for split in ("train", "val"):
        rows = ((d.split == split) & d.raw_CDRSB.notna()).to_numpy()
        x, y = d.raw_CDRSB.to_numpy()[rows], p_ad[rows]
        rho = spearmanr(x, y).correlation
        idx = C.boot_idx(len(x))
        bs = [spearmanr(x[i], y[i]).correlation for i in idx]
        out[split] = {"n": int(rows.sum()), "spearman": float(rho), "ci": [float(np.nanpercentile(bs, 2.5)),
                                                                           float(np.nanpercentile(bs, 97.5))]}
        print(split, out[split])
    C.write_json(out, "p5_posthoc_cdrsb.json")
    C.verify_manifest()


if __name__ == "__main__":
    main()
