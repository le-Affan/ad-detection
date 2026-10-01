"""Freeze, part 1 (run with LYP/.venv-fusion): final specialists and references on all 691 train+val subjects.

Never reads test. Steps:
 0. Check: final_features fitted on train alone reproduces step4_train.csv
    (all 36 features) and the fusion_trainval.csv ratios.
 1. Statistics fitted on the 691 subjects (frozen/feature_stats.json); the
    feature frame for the 691 is built with them.
 2. For each seed s in 42-46:
    - the clinical specialist (top-12 ranking inside) and the MRI specialist
      (logreg C = 0.1) fitted on all 691 -> frozen/clinical_s.cbm (+ top list),
      frozen/mri_s.pkl;
    - 5-fold OOF stream scores over the 691 (StratifiedKFold(5, shuffle,
      random_state = s)), which train the final KAN;
    - reference CatBoost F (kan-fusion frozen top-12 + 7 MRI inputs) and R (the
      36 step-4 features) fitted on all 691.
Outputs in analysis/fusion_v3/frozen/: feature_stats.json, streams_final.npz,
models. Run once; the freeze records their hashes.
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import final_features as FF  # noqa: E402
import metrics as M  # noqa: E402
import specialists as S  # noqa: E402

FROZEN = os.path.join(HERE, "frozen")
MRI_CHOICE = "logreg_C0.1"
EXTRAS = ["AGE", "CDRSB", "CogIndex", "MMSE"]


def check_builder():
    tr = list(pd.read_csv(os.path.join(FF.PROCESSED, "train_rids.csv")).RID)
    d = FF.build(tr, FF.fit_stats(tr), "fit").set_index("RID")
    s4 = pd.read_csv(os.path.join(FF.PROCESSED, "step4_train.csv")).set_index("RID")
    num = [c for c in s4.columns if c not in ("label", "PTGENDER")]
    assert np.allclose(d.loc[s4.index, num].to_numpy(dtype=float), s4[num].to_numpy(dtype=float), atol=1e-9)
    assert (d.loc[s4.index, "PTGENDER"] == s4.PTGENDER).all()
    fu = pd.read_csv(os.path.join(ROOT, "analysis", "kan_fusion", "fusion_trainval.csv")).set_index("RID")
    fu = fu[fu.split == "train"]
    assert np.allclose(d.loc[fu.index, FF.RATIOS].to_numpy(), fu[FF.RATIOS].to_numpy(), atol=1e-9)
    assert np.allclose(d.loc[fu.index, "ALZ"].fillna(-1), fu.ALZ.fillna(-1))
    print("builder check: reproduces step4_train.csv and fusion_trainval ratios/ALZ")


def main():
    check_builder()
    os.makedirs(FROZEN, exist_ok=True)
    rids = list(pd.read_csv(os.path.join(FF.PROCESSED, "train_rids.csv")).RID) + \
        list(pd.read_csv(os.path.join(FF.PROCESSED, "val_rids.csv")).RID)
    stats = FF.fit_stats(rids)
    FF.save_stats(stats, os.path.join(FROZEN, "feature_stats.json"))
    d = FF.build(rids, stats, "fit")
    y = d.label.to_numpy()
    kind, hp = S.MRI_CANDIDATES[MRI_CHOICE]
    topk = json.load(open(os.path.join(ROOT, "analysis", "kan_fusion", "topk.json")))
    f_cols, r_cols = topk["final_list"] + topk["mri_inputs"], topk["r_cols"]
    out = {"rids": d.RID.to_numpy(), "y": y, "has": d.has_mri.to_numpy(), "extras": d[EXTRAS].to_numpy(dtype=float)}
    tops = {}
    for seed in M.SEEDS:
        c = S.Clinical(seed).fit(d)
        c.model.save_model(os.path.join(FROZEN, f"clinical_{seed}.cbm"))
        tops[seed] = c.top
        m = S.MRI(kind, seed, **hp).fit(d)
        pickle.dump(m, open(os.path.join(FROZEN, f"mri_{seed}.pkl"), "wb"))
        pc, pm = np.zeros((len(d), 4)), np.full((len(d), 4), np.nan)
        for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(d, y):
            pc[b] = S.Clinical(seed).fit(d.iloc[a]).predict_proba(d.iloc[b])
            pm[b] = S.MRI(kind, seed, **hp).fit(d.iloc[a]).predict_proba(d.iloc[b])
        out[f"clin_oof_{seed}"], out[f"mri_oof_{seed}"] = pc, pm
        for name, cols in (("F", f_cols), ("R", r_cols)):
            S._cb(d[cols], y, seed).save_model(os.path.join(FROZEN, f"catboost_{name}_{seed}.cbm"))
        print(f"seed {seed}: clinical top-12 {c.top}; OOF macro-F1 clin "
              f"{M.metrics(y, pc)['macro_f1']:.3f}, MRI AD AUC {M.metrics(y[out['has']], pm[out['has']])['ad_auc']:.3f}",
              flush=True)
    np.savez(os.path.join(FROZEN, "streams_final.npz"), **out)
    json.dump({"clinical_top12": {str(k): v for k, v in tops.items()}, "catboost_F_cols": f_cols,
               "catboost_R_cols": r_cols, "extras": EXTRAS, "mri_choice": MRI_CHOICE},
              open(os.path.join(FROZEN, "columns.json"), "w"), indent=1)
    print(f"691 subjects: {int(d.has_mri.sum())} with a scan, {int(d.ba_ok.sum())} with a reliable brain age")


if __name__ == "__main__":
    main()
