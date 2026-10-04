"""Post-hoc analysis, not in the freeze: input-set controls for the MRI specialist
(run with LYP/.venv-fusion). Train CV and val only; never reads test.

The frozen MRI specialist configuration (multinomial logistic, L2, C = 0.1,
median imputation and z-scoring fitted on the training rows, brain-age α/β
refitted on the training rows' CN subjects) is retrained with six input sets:
  M0  age + sex
  M1  six volumes/ICV
  M2  volumes/ICV + age + sex
  M3  volumes/ICV + ALZ + age + sex
  M4  M3 + corrected brain-age gap (the frozen MRI specialist)
  M5  ALZ + brain-age gap
CV: the step-1 folds (StratifiedKFold(5, shuffle, random_state = seed), seeds
42-46) over the 569 train subjects, fitted and scored on subjects with a scan
(567). Val: fitted on the 567 train subjects with a scan, scored on the 121 val
subjects with a scan (logistic regression is deterministic, so the val result
is the same for every seed).
Paired bootstrap (2,000 resamples of subjects, rng seed 0): M4 − M0 and M4 − M2
for AD-vs-rest AUC and 4-class macro-F1; on CV the metric is averaged over the
5 seeds inside each resample.
Output: mri_controls.md, mri_controls.json.
"""
import json
import os
import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import metrics as M  # noqa: E402
import specialists as S  # noqa: E402

R = S.RATIOS
SETS = {"M0": ["AGE", "SEX"],
        "M1": R,
        "M2": R + ["AGE", "SEX"],
        "M3": R + ["ALZ", "AGE", "SEX"],
        "M4": R + ["ALZ", "BAG", "AGE", "SEX"],
        "M5": ["ALZ", "BAG"]}
DESC = {"M0": "age + sex only", "M1": "six volumes/ICV only", "M2": "volumes/ICV + age + sex",
        "M3": "volumes/ICV + ALZ + age + sex (no brain-age)",
        "M4": "M3 + corrected brain-age gap (= frozen MRI specialist)", "M5": "ALZ + brain-age gap only"}
HP = S.MRI_CANDIDATES["logreg_C0.1"]


class MRISubset(S.MRI):
    """S.MRI with a chosen input subset; fitting is otherwise unchanged."""

    def __init__(self, cols, seed):
        super().__init__(HP[0], seed, **HP[1])
        self.cols = cols

    def _X(self, df):
        X = df[R + ["ALZ", "AGE", "SEX"]].copy()
        X["BAG"] = np.where(df.ba_ok, (df.pred_age - self.beta) / self.alpha - df.age_at_scan, np.nan)
        return X[self.cols].to_numpy(dtype=float)


def auc_fast(y, s):
    """AD-vs-rest AUC via ranks (ties averaged); y boolean."""
    from scipy.stats import rankdata
    r = rankdata(s)
    n1 = y.sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def mf1(y, p):
    return M.f1s(M.confusion(y, p))[0].mean()


def boot(y, P, idx):
    """P: (n_seeds, n, 4). Returns (B,) seed-averaged AUC and macro-F1 per resample."""
    auc = np.array([[auc_fast(y[i] == 3, p[i, 3]) for p in P] for i in idx]).mean(1)
    f1 = M.macro_f1_boot(y, P.argmax(2), idx).mean(1)
    return auc, f1


def main():
    d = S.load_trainval()
    tr = d[(d.split == "train") & d.has_mri].reset_index(drop=True)
    va = d[(d.split == "val") & d.has_mri].reset_index(drop=True)
    tr_all = d[d.split == "train"].reset_index(drop=True)
    keep = tr_all.has_mri.to_numpy()
    y, yv = tr.label.to_numpy(), va.label.to_numpy()
    cv, val = {}, {}
    for name, cols in SETS.items():
        P = np.full((len(M.SEEDS), len(tr_all), 4), np.nan)
        for si, seed in enumerate(M.SEEDS):
            # step-1 folds: split over all 569 train subjects, then restricted to those with a scan
            for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(tr_all, tr_all.label):
                P[si, b] = MRISubset(cols, seed).fit(tr_all.iloc[a]).predict_proba(tr_all.iloc[b])
        cv[name] = P[:, keep]
        val[name] = MRISubset(cols, 42).fit(tr).predict_proba(va)[None]
    # M4 must reproduce the frozen-config CV out-of-fold predictions of step 1
    ref = np.load(os.path.join(HERE, "mri_select_oof.npz"))["logreg_C0.1"][:, keep]
    assert np.abs(cv["M4"] - ref).max() < 1e-9, "M4 does not reproduce the MRI specialist"

    res = {"note": "post-hoc analysis, not in the freeze", "n_cv": int(len(y)), "n_val": int(len(yv)), "sets": {}}
    for name in SETS:
        per = [M.metrics(y, p) for p in cv[name]]
        mv = M.metrics(yv, val[name][0])
        res["sets"][name] = {"inputs": SETS[name], "description": DESC[name],
                             "cv_auc": [float(np.mean([m["ad_auc"] for m in per])), float(np.std([m["ad_auc"] for m in per], ddof=1))],
                             "cv_f1": [float(np.mean([m["macro_f1"] for m in per])), float(np.std([m["macro_f1"] for m in per], ddof=1))],
                             "cv_logloss": float(np.mean([m["logloss"] for m in per])),
                             "cv_f1_by_class": [float(np.mean([m[f"f1_{c}"] for m in per])) for c in M.NAMES],
                             "val_auc": float(mv["ad_auc"]), "val_f1": float(mv["macro_f1"]),
                             "val_logloss": float(mv["logloss"])}

    rng = np.random.default_rng(0)
    idx = {"cv": rng.integers(0, len(y), size=(2000, len(y))),
           "val": rng.integers(0, len(yv), size=(2000, len(yv)))}
    B = {ev: {n: boot(yy, (cv if ev == "cv" else val)[n], idx[ev]) for n in ("M0", "M2", "M4")}
         for ev, yy in (("cv", y), ("val", yv))}
    res["boot"] = {}
    for ev in ("cv", "val"):
        for ref_ in ("M0", "M2"):
            for mi, met in enumerate(("auc", "f1")):
                a, b = res["sets"]["M4"][f"{ev}_{met}"], res["sets"][ref_][f"{ev}_{met}"]
                point = a[0] - b[0] if ev == "cv" else a - b
                dd = B[ev]["M4"][mi] - B[ev][ref_][mi]
                res["boot"][f"{ev} M4 − {ref_} {met}"] = [float(point), float(np.percentile(dd, 2.5)),
                                                          float(np.percentile(dd, 97.5))]
    json.dump(res, open(os.path.join(HERE, "mri_controls.json"), "w"), indent=1)
    for n, s in res["sets"].items():
        print(f"{n} {s['description']:52s} CV AUC {s['cv_auc'][0]:.3f} ± {s['cv_auc'][1]:.3f}  "
              f"CV F1 {s['cv_f1'][0]:.3f} ± {s['cv_f1'][1]:.3f}  val AUC {s['val_auc']:.3f}  val F1 {s['val_f1']:.3f}  "
              f"F1 by class {[round(x, 2) for x in s['cv_f1_by_class']]}")
    for k, v in res["boot"].items():
        print(f"{k}: {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]")


if __name__ == "__main__":
    main()
