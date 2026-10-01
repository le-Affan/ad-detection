"""Step 1: choose the MRI specialist by train CV (run with LYP/.venv-fusion). Never reads test.

Each candidate in specialists.MRI_CANDIDATES gets repeated stratified 5-fold CV
over the 569 clinical-train subjects (StratifiedKFold(5, shuffle,
random_state=seed), seeds 42-46, the harness folds). It is fitted on the
training folds' subjects with a scan and evaluated on the 567 train subjects
with a scan.

Selection: lowest mean CV multi-class log-loss. The specialist's job is to give
the fusion layer calibrated probabilities, so log-loss is the selection
criterion; macro-F1 and the other metrics are logged too. Every candidate is
logged in config_log.csv.

Output: mri_select_oof.npz (OOF probabilities per candidate).
"""
import os
import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import metrics as M  # noqa: E402
import specialists as S  # noqa: E402


def main():
    d = S.load_trainval()
    tr = d[d.split == "train"].reset_index(drop=True)
    y, mri = tr.label.to_numpy(), tr.has_mri.to_numpy()
    out, res = {}, {}
    for name, (kind, hp) in S.MRI_CANDIDATES.items():
        P = np.full((len(M.SEEDS), len(tr), 4), np.nan)
        for si, seed in enumerate(M.SEEDS):
            for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(tr, y):
                P[si, b] = S.MRI(kind, seed, **hp).fit(tr.iloc[a]).predict_proba(tr.iloc[b])
        s = M.cv_summary(y, P, mri)
        res[name], out[name] = s, P
        print(f"{name:14s} logloss {s['logloss']:.4f}  macro-F1 {s['macro_f1']:.3f} ± {s['macro_f1_sd']:.3f}  "
              f"acc {s['acc']:.3f}  AD AUC {s['ad_auc']:.3f}  LMCI {s['f1_LMCI']:.3f}  AD {s['f1_AD']:.3f}", flush=True)
    best = min(res, key=lambda k: res[k]["logloss"])
    for name, (kind, hp) in S.MRI_CANDIDATES.items():
        M.log_config("mri_specialist", f"MRI specialist {name}", "+".join(S.MRI_FEATURES), {"kind": kind, **hp},
                     res[name], decision="selected" if name == best else "not selected",
                     notes="selection = lowest CV log-loss; evaluated on 567 train subjects with a scan")
    np.savez(os.path.join(HERE, "mri_select_oof.npz"), **out)
    prior = np.bincount(y[mri], minlength=4) / mri.sum()
    print(f"selected: {best}; class-prior log-loss {-np.mean(np.log(prior[y[mri]])):.4f}")


if __name__ == "__main__":
    main()
