"""CatBoost stage of the KAN-fusion experiment (preregistration.md items b, c). Never reads test.

1. Per seed (42-46) and fold of StratifiedKFold(5, shuffle, random_state=seed)
   over the 569 clinical-train subjects (the harness folds): CatBoost on the
   training portion with the 29 clinical features, SHAP (TreeExplainer) on the
   same rows, ranking by mean |SHAP| over subjects and classes. Ties go to the
   earlier column. C is then fitted for K in {8, 10, 12} on that fold's top-K.
2. K* = argmax of the CV macro-F1 of CatBoost C (pooled OOF per seed, mean over
   seeds; an exact tie goes to the smaller K).
3. Final list: CatBoost on all 569 with the 29 features, one per seed; mean
   |SHAP| on the train rows averaged over seeds; top K*.
4. C (top-K*), F (top-K* + 7 MRI inputs) and R (the 36 step-4 features, = arm A)
   for CV and val (train all 569 -> 122 val).

CatBoost settings are the harness's (analysis/fusion_feasibility/ablation.py,
imported unchanged): depth 6, l2 10, lr 0.1, a fixed 265 trees, PTGENDER
categorical when present, ALZ NaN handled natively.

Outputs: catboost_preds.npz (OOF and val probabilities), topk.json (fold-level
and final lists, K*), catboost_stage.log (printed). Run with LYP/.venv-fusion.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "analysis", "fusion_feasibility"))
import ablation as H  # noqa: E402

KS = (8, 10, 12)
MRI_INPUTS = ["Ventricles_ICV", "Hippocampus_ICV", "WholeBrain_ICV", "Entorhinal_ICV",
              "Fusiform_ICV", "MidTemp_ICV", "ALZ"]


def load():
    d = pd.read_csv(os.path.join(HERE, "fusion_trainval.csv"))
    clinical = json.load(open(os.path.join(HERE, "fusion_manifest.json")))["clinical_features"]
    tr, va = d[d.split == "train"].reset_index(drop=True), d[d.split == "val"].reset_index(drop=True)
    for df, s in ((tr, "train"), (va, "val")):                    # R: the 36 step-4 features
        s4 = pd.read_csv(os.path.join(ROOT, "dataset", "processed", f"step4_{s}.csv"))
        assert (s4.RID.to_numpy() == df.RID.to_numpy()).all()
        for c in ("Ventricles", "Hippocampus", "WholeBrain", "Entorhinal", "Fusiform", "MidTemp", "ICV"):
            df[c] = s4[c].to_numpy()
    r_cols = [c for c in s4.columns if c not in ("RID", "label")]
    assert len(r_cols) == 36
    return tr, va, clinical, r_cols


def fit(X, y, seed):
    m = CatBoostClassifier(random_seed=seed, cat_features=[c for c in ("PTGENDER",) if c in X.columns],
                           **H.PARAMS)
    m.fit(X, y)
    return m


def shap_importance(m, X):
    sv = np.abs(np.asarray(shap.TreeExplainer(m).shap_values(X)))
    return sv.mean(axis=(0, 2)) if sv.ndim == 3 else sv.mean(0)    # as in the harness


def ranking(imp, cols):
    return [cols[i] for i in np.argsort(-imp, kind="stable")]       # stable: ties keep column order


def macro_f1(y, P):
    return H.metrics(y, P)["macro_f1"]


def main():
    tr, va, clinical, r_cols = load()
    ytr, yva = tr.label.to_numpy(), va.label.to_numpy()
    seeds = H.SEEDS
    n_s = len(seeds)

    # 1. fold rankings and C at each K
    fold_rank, fold_idx = {}, {}
    oof_ck = {k: np.zeros((n_s, len(tr), 4)) for k in KS}
    for si, seed in enumerate(seeds):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fi, (a, b) in enumerate(skf.split(tr, ytr)):
            m = fit(tr.iloc[a][clinical], ytr[a], seed)
            rank = ranking(shap_importance(m, tr.iloc[a][clinical]), clinical)
            fold_rank[f"{seed}_{fi}"], fold_idx[f"{seed}_{fi}"] = rank, (a, b)
            for k in KS:
                mk = fit(tr.iloc[a][rank[:k]], ytr[a], seed)
                oof_ck[k][si, b] = mk.predict_proba(tr.iloc[b][rank[:k]])
        print(f"seed {seed}: rankings + C_K done", flush=True)

    k_score = {k: float(np.mean([macro_f1(ytr, oof_ck[k][si]) for si in range(n_s)])) for k in KS}
    k_star = max(KS, key=lambda k: (round(k_score[k], 12), -k))
    print("CV macro-F1 of CatBoost C by K:", {k: round(v, 4) for k, v in k_score.items()}, "-> K* =", k_star)

    # 3. final list
    imps = []
    for seed in seeds:
        m = fit(tr[clinical], ytr, seed)
        imps.append(shap_importance(m, tr[clinical]))
    mean_imp = np.mean(imps, axis=0)
    final_rank = ranking(mean_imp, clinical)
    final = final_rank[:k_star]
    freq = pd.Series([f for r in fold_rank.values() for f in r[:k_star]]).value_counts()
    print("final top-K*:", final)
    print("fold-level top-K* frequency (of 25):", freq.to_dict())

    # 4. C, F, R
    sets = {"C": None, "F": None, "R": r_cols}
    oof = {s: np.zeros((n_s, len(tr), 4)) for s in sets}
    val = {s: np.zeros((n_s, len(va), 4)) for s in sets}
    for si, seed in enumerate(seeds):
        for fi in range(5):
            a, b = fold_idx[f"{seed}_{fi}"]
            top = fold_rank[f"{seed}_{fi}"][:k_star]
            for s, cols in (("C", top), ("F", top + MRI_INPUTS), ("R", r_cols)):
                if s == "C":
                    oof[s][si, b] = oof_ck[k_star][si, b]          # same model as in the K selection
                    continue
                oof[s][si, b] = fit(tr.iloc[a][cols], ytr[a], seed).predict_proba(tr.iloc[b][cols])
        for s, cols in (("C", final), ("F", final + MRI_INPUTS), ("R", r_cols)):
            val[s][si] = fit(tr[cols], ytr, seed).predict_proba(va[cols])
        print(f"seed {seed}: C/F/R done", flush=True)

    for s in sets:
        cv = [macro_f1(ytr, oof[s][si]) for si in range(n_s)]
        vv = [macro_f1(yva, val[s][si]) for si in range(n_s)]
        print(f"{s}: CV macro-F1 {np.mean(cv):.4f} ± {np.std(cv, ddof=1):.4f}   val {np.mean(vv):.4f} ± {np.std(vv, ddof=1):.4f}")

    np.savez(os.path.join(HERE, "catboost_preds.npz"), train_rids=tr.RID.to_numpy(), val_rids=va.RID.to_numpy(),
             **{f"oof_{s}": oof[s] for s in sets}, **{f"val_{s}": val[s] for s in sets},
             **{f"oof_C{k}": oof_ck[k] for k in KS})
    folds = {key: {"train": a.tolist(), "test": b.tolist()} for key, (a, b) in fold_idx.items()}
    json.dump({"k_scores": k_score, "k_star": k_star, "final_list": final,
               "final_ranking_29": final_rank, "final_mean_abs_shap": dict(zip(clinical, mean_imp.tolist())),
               "fold_rankings": fold_rank, "fold_topk_frequency": freq.to_dict(), "folds": folds,
               "mri_inputs": MRI_INPUTS, "r_cols": r_cols},
              open(os.path.join(HERE, "topk.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
