"""Step 2: nested out-of-fold stream scores (run with LYP/.venv-fusion). Never reads test.

For each seed s (42-46) and outer fold f of StratifiedKFold(5, shuffle,
random_state=s) over the 569 clinical-train subjects (the harness folds):
  - both specialists are fitted on the whole outer-train set and score the
    outer-test fold, giving clin_oof[s], mri_oof[s] over all 569;
  - inside the outer-train set, StratifiedKFold(5, shuffle, random_state=s)
    builds inner out-of-fold scores (clin_inner, mri_inner). These are the only
    stream scores the fusion layer is trained on for that outer fold.
For val (per seed): the fusion layer trains on clin_oof[s] / mri_oof[s] (OOF
over 569), and specialists fitted on all 569 score the 122 val subjects.

MRI specialist = the step-1 selection (logreg C = 0.1). MRI scores are NaN
for subjects without a scan.
Check: clin_oof must reproduce kan-fusion's CatBoost C OOF predictions (same
procedure and seeds).
Output: streams.npz.
"""
import json
import os
import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import metrics as M  # noqa: E402
import specialists as S  # noqa: E402

MRI_CHOICE = "logreg_C0.1"


def main():
    d = S.load_trainval()
    tr = d[d.split == "train"].reset_index(drop=True)
    va = d[d.split == "val"].reset_index(drop=True)
    y = tr.label.to_numpy()
    kind, hp = S.MRI_CANDIDATES[MRI_CHOICE]
    n_s = len(M.SEEDS)
    out = {"clin_oof": np.zeros((n_s, len(tr), 4)), "mri_oof": np.full((n_s, len(tr), 4), np.nan),
           "clin_val": np.zeros((n_s, len(va), 4)), "mri_val": np.full((n_s, len(va), 4), np.nan)}
    folds = {}
    for si, seed in enumerate(M.SEEDS):
        for fi, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(tr, y)):
            otr = tr.iloc[a].reset_index(drop=True)
            out["clin_oof"][si, b] = S.Clinical(seed).fit(otr).predict_proba(tr.iloc[b])
            out["mri_oof"][si, b] = S.MRI(kind, seed, **hp).fit(otr).predict_proba(tr.iloc[b])
            ci, mi = np.zeros((len(a), 4)), np.full((len(a), 4), np.nan)
            for ia, ib in StratifiedKFold(5, shuffle=True, random_state=seed).split(otr, otr.label):
                itr = otr.iloc[ia]
                ci[ib] = S.Clinical(seed).fit(itr).predict_proba(otr.iloc[ib])
                mi[ib] = S.MRI(kind, seed, **hp).fit(itr).predict_proba(otr.iloc[ib])
            out[f"clin_inner_{seed}_{fi}"], out[f"mri_inner_{seed}_{fi}"] = ci, mi
            out[f"idx_train_{seed}_{fi}"], out[f"idx_test_{seed}_{fi}"] = a, b
        out["clin_val"][si] = S.Clinical(seed).fit(tr).predict_proba(va)
        out["mri_val"][si] = S.MRI(kind, seed, **hp).fit(tr).predict_proba(va)
        print(f"seed {seed} done", flush=True)

    old = np.load(os.path.join(ROOT, "analysis", "kan_fusion", "catboost_preds.npz"))
    dc, dv = np.abs(out["clin_oof"] - old["oof_C"]).max(), np.abs(out["clin_val"] - old["val_C"]).max()
    print(f"clinical specialist vs kan-fusion CatBoost C: max |diff| OOF {dc:.2e}, val {dv:.2e}")
    np.savez(os.path.join(HERE, "streams.npz"), train_rids=tr.RID.to_numpy(), val_rids=va.RID.to_numpy(), **out)

    s = M.cv_summary(y, out["clin_oof"])
    M.log_config("clinical_specialist", "Clinical specialist (CatBoost, top-12 of 29, 265 trees)",
                 "clinical top-12 (per-fold SHAP ranking)", {"K": 12, "trees": 265, "depth": 6, "l2": 10, "lr": 0.1},
                 s, decision="used", notes=f"K = 12 chosen on train CV in kan-fusion; reproduces kan-fusion C "
                                           f"(max |diff| {dc:.1e})")
    print(json.dumps({k: round(v, 4) for k, v in s.items()}))


if __name__ == "__main__":
    main()
