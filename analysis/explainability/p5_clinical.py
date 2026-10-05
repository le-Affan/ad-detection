"""P5, clinical-SHAP row (LYP/.venv-fusion): the card's existing explainer, evaluated without refitting.

Explainer: CatBoost ShapValues of the 5 frozen clinical specialists (frozen/clinical_<seed>.cbm, each on its own
top-12 features), mapped onto the union of their features (15; 0 where a seed omits one), centred over the 4
classes (the raw multiclass scores are shift-invariant).
Stability: global importance per seed = mean |phi| over the split, averaged over the 4 centred class targets;
  median pairwise Spearman over the 10 seed pairs. Per subject: median over seed pairs of the Spearman of the
  15-vectors on the predicted-class target.
Plausibility: 5-seed mean phi of CDRSB and MMSE on the AD centred target vs the CN-train median of the raw value.
Faithfulness: P(clinical predicted class) of the 5-seed ensemble; deletion / insertion over the 15 union features
  (replacement by 50 train background rows, rng 0) vs 50 random orders (seed SeedSequence([RID, 1])).
Output: results/p5_rows.json["clinical_shap"], results/p5_clinical_curves.csv.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import tab_data as TD  # noqa: E402
import tab_faith as TF  # noqa: E402

SEEDS = (42, 43, 44, 45, 46)
AD = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    if a.dry:
        C.set_dry()
    C.verify_manifest()
    t0 = time.time()
    d = TD.trainval()
    tr = (d.split == "train").to_numpy()
    top = TD.columns()["clinical_top12"]
    union = sorted(set(sum(top.values(), [])))
    models, cols = [], []
    phi = np.zeros((len(SEEDS), len(d), len(union), 4))
    for si, s in enumerate(SEEDS):
        m = CatBoostClassifier()
        m.load_model(os.path.join(C.FROZEN, f"clinical_{s}.cbm"))
        assert m.feature_names_ == top[str(s)]
        sv = m.get_feature_importance(Pool(d[top[str(s)]]), type="ShapValues")      # (n, 4, 13)
        add_err = np.abs(sv.sum(2) - m.predict(d[top[str(s)]], prediction_type="RawFormulaVal")).max()
        assert add_err < 1e-4, add_err
        for j, c in enumerate(top[str(s)]):
            phi[si, :, union.index(c), :] = sv[:, :, j]
        models.append(m)
        cols.append([union.index(c) for c in top[str(s)]])
    phi = phi - phi.mean(3, keepdims=True)                                          # centred class targets
    X = d[union].to_numpy(float)

    def predict(Z):
        return np.mean([m.predict_proba(Z[:, c], thread_count=16) for m, c in zip(models, cols)], 0)
    P = predict(X)
    pred = P.argmax(1)
    row = {"explanation": "Clinical SHAP", "detail": "CatBoost ShapValues, 5 frozen seeds, centred class scores",
           "features": union}
    for split, rows in (("train", tr), ("val", ~tr)):
        imp = np.abs(phi[:, rows]).mean(1).mean(-1)                                 # (5, 15)
        row[f"stability_{split}"], row[f"stability_{split}_pairs"] = C.median_pairwise_spearman(imp)
        ps = []
        for i in np.flatnonzero(rows):
            v = phi[:, i, :, pred[i]]
            ps.append(np.nanmedian([C.spearman(v[x], v[y]) for x in range(5) for y in range(x + 1, 5)]))
        row[f"stability_per_subject_{split}"] = float(np.nanmedian(ps))
    pl = {}
    for f, hi in (("CDRSB", True), ("MMSE", False)):
        med = TD.cn_train_median(d, f"raw_{f}")
        j = union.index(f)
        out = {}
        for split, rows in (("train", tr), ("val", ~tr)):
            e = TD.expected_sign(d.loc[rows, f"raw_{f}"], med, hi)
            p = phi[:, rows, j, AD].mean(0)
            ok = (e != 0) & (np.abs(p) > 1e-6)
            out[split] = float((np.sign(p[ok]) == e[ok]).mean()) if ok.any() else np.nan
            out[f"n_{split}"] = int(ok.sum())
        pl[f] = out
    row["plausibility"] = pl
    print({k: v for k, v in row.items() if k != "features"}, f"{time.time() - t0:.0f}s", flush=True)
    B = X[np.sort(np.random.default_rng(0).choice(np.flatnonzero(tr), 50, replace=False))]
    sets = {"train": np.flatnonzero(tr), "val": np.flatnonzero(~tr)}
    if a.dry:
        sets = {"train": np.flatnonzero(d.PTID.isin(C.dry_run_ptids()).to_numpy()), "val": np.flatnonzero(~tr)[:3]}
    faith, curves = {}, []
    for split, ix in sets.items():
        res = [TF.subject_faithfulness(X[i], B, phi[:, i, :, pred[i]].mean(0), d.RID.iloc[i], 1, predict, pred[i])
               for i in ix]
        dd, ii = np.array([r["del_diff"] for r in res]), np.array([r["ins_diff"] for r in res])
        dm, dlo, dhi = C.boot_ci(dd)
        im, ilo, ihi = C.boot_ci(ii)
        dp, ip, v = TF.verdict(dhi, ilo)
        faith[split] = {"n": len(ix), "deletion_diff": dm, "deletion_ci": [dlo, dhi], "deletion_pass": dp,
                        "insertion_diff": im, "insertion_ci": [ilo, ihi], "insertion_pass": ip, "verdict": v}
        for k in ("del_attr", "del_rand", "ins_attr", "ins_rand"):
            for step, y in enumerate(np.mean([r[k] for r in res], 0)):
                curves.append({"split": split, "curve": k, "step": step, "mean_prob": y})
        print(split, faith[split], f"{time.time() - t0:.0f}s", flush=True)
    row["faithfulness"] = faith
    C.write_aggregate(pd.DataFrame(curves), "p5_clinical_curves.csv", ["train", "val"])
    C.update_json("p5_rows.json", "clinical_shap", row)
    C.verify_manifest()


if __name__ == "__main__":
    main()
