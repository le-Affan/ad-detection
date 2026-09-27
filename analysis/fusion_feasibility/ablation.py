"""CatBoost ablation: does cross-fitted ALZ add value? Never reads test.

Implements the protocol pre-registered in report.md (commit 31462eb):
  arms A, B, (C), A', B'; step-5 hyperparameters with a fixed 265 trees and no
  early stopping; 5 CatBoost seeds; repeated stratified 5-fold CV over the 569
  clinical-train subjects (primary) and train-all -> clinical-val; paired
  bootstrap of the macro-F1 difference for B-A and B'-A'.

Inputs:  dataset/processed/step4_{train,val}.csv, alz_crossfit.csv
Outputs: ablation_runs.csv (every seed x arm x eval x subset), ablation_oof.npz,
         ablation_tables.md (the report tables), printed summary.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PROCESSED = os.path.join(ROOT, "dataset", "processed")

NAMES = ["CN", "EMCI", "LMCI", "AD"]
PARAMS = dict(loss_function="MultiClass", depth=6, l2_leaf_reg=10, learning_rate=0.1,
              iterations=265, verbose=False,
              allow_writing_files=False)  # keep step 5's tracked catboost_info/ intact
SEEDS = (42, 43, 44, 45, 46)
OUT = HERE
N_BOOT = 2000
DEFINING = ["CDRSB", "MMSE", "CogIndex"]          # A' removals, see report.md
PAIRS = [("B", "A"), ("B'", "A'")]


# --- metrics ------------------------------------------------------------------
def confusion(y, p):
    return np.bincount(4 * y + p, minlength=16).reshape(4, 4)


def f1s_from_cm(cm):
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros(4), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros(4), where=cm.sum(1) > 0)
    return np.divide(2 * prec * rec, prec + rec, out=np.zeros(4), where=(prec + rec) > 0), rec


def metrics(y, proba):
    p = proba.argmax(1)
    f1, rec = f1s_from_cm(confusion(y, p))
    out = {"macro_f1": f1.mean(), "bal_acc": rec.mean(), "acc": (p == y).mean(),
           "ad_auc": roc_auc_score(y == 3, proba[:, 3])}
    out.update({f"f1_{n}": f1[i] for i, n in enumerate(NAMES)})
    return out


def macro_f1_boot(y, preds, idx):
    """preds: (n_seeds, n) class predictions; idx: (B, n) resample indices.
    Returns (B, n_seeds) macro-F1 values."""
    out = np.empty((len(idx), len(preds)))
    for s, p in enumerate(preds):
        codes = 4 * y[idx] + p[idx]                       # (B, n)
        cms = np.stack([np.bincount(c, minlength=16) for c in codes]).reshape(-1, 4, 4)
        tp = np.diagonal(cms, axis1=1, axis2=2).astype(float)
        ps, rs = cms.sum(1), cms.sum(2)
        prec = np.divide(tp, ps, out=np.zeros_like(tp), where=ps > 0)
        rec = np.divide(tp, rs, out=np.zeros_like(tp), where=rs > 0)
        f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
        out[:, s] = f1.mean(1)
    return out


# --- data -----------------------------------------------------------------------
def load(crossfit):
    cf = pd.read_csv(crossfit)
    tr = pd.read_csv(os.path.join(PROCESSED, "step4_train.csv"))
    va = pd.read_csv(os.path.join(PROCESSED, "step4_val.csv"))
    for df, s in ((tr, "train"), (va, "val")):
        c = cf[cf.clin_split == s].set_index("RID")
        assert set(c.index) == set(df.RID), f"{s}: crossfit RIDs != step4 RIDs"
        assert (c.label.reindex(df.RID).values == df.label.values).all()
        df["ALZ"] = df.RID.map(c.ALZ)
        df["DEMO"] = df.RID.map(c.DEMO)
        df["has_mri"] = df.RID.map(c.has_mri).astype(bool)
    return tr, va


def demo_rule(tr):
    m = tr[tr.has_mri]
    rho = spearmanr(m.DEMO, m.label).statistic
    means = m.groupby("label").DEMO.mean()
    span = means.max() - means.min()
    return {"spearman_rho": float(rho), "class_mean_span": float(span),
            "per_class": m.groupby("label").DEMO.agg(["mean", "std", "min", "max"]).round(3)
                          .rename(index=dict(enumerate(NAMES))).to_dict("index"),
            "not_collapsed": bool(rho >= 0.2 and span >= 0.25)}


def fit(X, y, seed):
    m = CatBoostClassifier(random_seed=seed, cat_features=["PTGENDER"], **PARAMS)
    m.fit(X, y)
    return m


# --- main -------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="code check: 1 seed, 5 trees, 50 boots")
    ap.add_argument("--crossfit", default=os.path.join(HERE, "alz_crossfit.csv"))
    ap.add_argument("--out", default=HERE)
    args = ap.parse_args()
    global OUT
    OUT = args.out
    seeds, n_boot = (SEEDS[:1], 50) if args.smoke else (SEEDS, N_BOOT)
    if args.smoke:
        PARAMS["iterations"] = 5

    tr, va = load(args.crossfit)
    base = [c for c in tr.columns if c not in ("RID", "label", "ALZ", "DEMO", "has_mri")]
    assert len(base) == 36 and all(c in base for c in DEFINING)
    reduced = [c for c in base if c not in DEFINING]
    dr = demo_rule(tr)
    print("DEMO rule:", json.dumps({k: v for k, v in dr.items() if k != "per_class"}))
    arms = {"A": base, "B": base + ["ALZ"]}
    if dr["not_collapsed"]:
        arms["C"] = base + ["ALZ", "DEMO"]
    arms["A'"] = reduced
    arms["B'"] = reduced + ["ALZ"]
    print({a: len(c) for a, c in arms.items()})

    ytr, yva = tr.label.to_numpy(), va.label.to_numpy()
    mri_tr, mri_va = tr.has_mri.to_numpy(), va.has_mri.to_numpy()
    oof = {a: np.zeros((len(seeds), len(tr), 4)) for a in arms}
    vpr = {a: np.zeros((len(seeds), len(va), 4)) for a in arms}
    shap_rows = []
    for si, seed in enumerate(seeds):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fi, (a_idx, b_idx) in enumerate(skf.split(tr, ytr)):
            for a, cols in arms.items():
                m = fit(tr.iloc[a_idx][cols], ytr[a_idx], seed)
                oof[a][si, b_idx] = m.predict_proba(tr.iloc[b_idx][cols])
        for a, cols in arms.items():
            m = fit(tr[cols], ytr, seed)
            vpr[a][si] = m.predict_proba(va[cols])
            if "ALZ" in cols:
                for split, df in (("val", va), ("train (in-sample)", tr)):
                    sv = np.abs(np.asarray(shap.TreeExplainer(m).shap_values(df[cols])))
                    imp = sv.mean(axis=(0, 2)) if sv.ndim == 3 else sv.mean(0)
                    order = np.argsort(-imp)
                    shap_rows.append({"arm": a, "seed": seed, "split": split,
                                      "alz_mean_abs_shap": float(imp[cols.index("ALZ")]),
                                      "alz_rank": int(np.where(np.array(cols)[order] == "ALZ")[0][0]) + 1,
                                      "n_features": len(cols),
                                      "top5": ", ".join(np.array(cols)[order][:5])})
        print(f"seed {seed} done")

    # per-seed metrics
    runs = []
    for a in arms:
        for si, seed in enumerate(seeds):
            for ev, y, P, mask in (("cv", ytr, oof[a][si], mri_tr), ("val", yva, vpr[a][si], mri_va)):
                for sub, mk in (("all", np.ones_like(mask)), ("mri", mask)):
                    runs.append({"arm": a, "seed": seed, "eval": ev, "subset": sub, "n": int(mk.sum()),
                                 **metrics(y[mk], P[mk])})
    runs = pd.DataFrame(runs)
    runs.to_csv(os.path.join(OUT, "ablation_runs.csv"), index=False)
    np.savez(os.path.join(OUT, "ablation_oof.npz"), train_rids=tr.RID.to_numpy(),
             val_rids=va.RID.to_numpy(), **{f"oof_{a}": oof[a] for a in arms},
             **{f"val_{a}": vpr[a] for a in arms})

    # paired bootstrap
    rng = np.random.default_rng(0)
    boots = {}
    for ev, y, P, mask in (("cv", ytr, oof, mri_tr), ("val", yva, vpr, mri_va)):
        for sub, mk in (("all", np.ones_like(mask)), ("mri", mask)):
            yy = y[mk]
            idx = rng.integers(0, len(yy), size=(n_boot, len(yy)))
            for b, a in PAIRS:
                pb = P[b][:, mk].argmax(-1)
                pa = P[a][:, mk].argmax(-1)
                d = (macro_f1_boot(yy, pb, idx) - macro_f1_boot(yy, pa, idx)).mean(1)
                point = np.mean([f1s_from_cm(confusion(yy, pb[s]))[0].mean()
                                 - f1s_from_cm(confusion(yy, pa[s]))[0].mean()
                                 for s in range(len(seeds))])
                boots[(ev, sub, b, a)] = {"diff": point, "lo": np.percentile(d, 2.5),
                                          "hi": np.percentile(d, 97.5),
                                          "per_seed": [float(x) for x in
                                                       runs.query("eval==@ev and subset==@sub and arm==@b").macro_f1.values
                                                       - runs.query("eval==@ev and subset==@sub and arm==@a").macro_f1.values]}

    verdicts = {}
    for b, a in PAIRS:
        cv, v = boots[("cv", "all", b, a)], boots[("val", "all", b, a)]
        excl = cv["lo"] > 0 or cv["hi"] < 0
        same = np.sign(cv["diff"]) == np.sign(v["diff"]) and v["diff"] != 0
        if excl and same:
            verdicts[f"{b} vs {a}"] = "improvement" if cv["diff"] > 0 else "reliable loss"
        else:
            verdicts[f"{b} vs {a}"] = "no reliable gain"

    # confusion matrices on val, mean over seeds
    cms = {a: np.mean([confusion(yva, vpr[a][s].argmax(1)) for s in range(len(seeds))], 0) for a in arms}
    cms_cv = {a: np.mean([confusion(ytr, oof[a][s].argmax(1)) for s in range(len(seeds))], 0) for a in arms}

    write_tables(runs, boots, verdicts, cms, cms_cv, pd.DataFrame(shap_rows), dr, arms, seeds, n_boot)
    print(open(os.path.join(OUT, "ablation_tables.md")).read())


def write_tables(runs, boots, verdicts, cms, cms_cv, shap_df, dr, arms, seeds, n_boot):
    L = []
    cols = ["macro_f1", "f1_LMCI", "f1_AD", "bal_acc", "ad_auc", "f1_CN", "f1_EMCI", "acc"]
    hdr = ["Macro-F1", "LMCI F1", "AD F1", "Bal. acc", "AD AUC", "CN F1", "EMCI F1", "Acc"]
    for ev, title in (("cv", "Cross-validation (569 clinical-train, OOF, 5 seeds)"),
                      ("val", "Validation (train on 569, evaluate on 122 clinical-val, 5 seeds)")):
        for sub in ("all", "mri"):
            g = runs[(runs["eval"] == ev) & (runs.subset == sub)]
            n = int(g.n.iloc[0])
            L.append(f"#### {title} — {'all subjects' if sub == 'all' else 'subjects with MRI only'} (n = {n})\n")
            L.append("| Arm | " + " | ".join(hdr) + " |")
            L.append("|---|" + "---|" * len(hdr))
            for a in arms:
                r = g[g.arm == a]
                L.append(f"| {a} | " + " | ".join(f"{r[c].mean():.3f} ± {r[c].std():.3f}" for c in cols) + " |")
            L.append("")
    L.append(f"#### Paired macro-F1 differences ({n_boot} bootstrap resamples, mean over {len(seeds)} seeds)\n")
    L.append("| Pair | Eval | Subset | Diff | 95% CI | Per-seed diffs |")
    L.append("|---|---|---|---|---|---|")
    for (ev, sub, b, a), r in boots.items():
        L.append(f"| {b} − {a} | {ev} | {sub} | {r['diff']:+.4f} | [{r['lo']:+.4f}, {r['hi']:+.4f}] | "
                 + " ".join(f"{x:+.3f}" for x in r["per_seed"]) + " |")
    L.append("")
    L.append("#### Verdicts against the pre-registered rule\n")
    for k, v in verdicts.items():
        L.append(f"- **{k}: {v}**")
    L.append("")
    for tag, cmd in (("val (122)", cms), ("CV (569)", cms_cv)):
        L.append(f"#### Confusion matrices, {tag}, mean over seeds (rows true, cols pred)\n")
        for a in arms:
            cm = cmd[a]
            L.append(f"**{a}**\n")
            L.append("| | " + " | ".join(NAMES) + " |")
            L.append("|---|---|---|---|---|")
            for i, n in enumerate(NAMES):
                L.append(f"| {n} | " + " | ".join(f"{x:.1f}" for x in cm[i]) + " |")
            L.append("")
    L.append("#### SHAP importance of ALZ (mean |SHAP| over subjects and classes)\n")
    L.append("| Arm | Split | ALZ rank (of n) | ALZ mean abs SHAP | Top 5 (seed 42) |")
    L.append("|---|---|---|---|---|")
    for (a, sp), g in shap_df.groupby(["arm", "split"], sort=False):
        L.append(f"| {a} | {sp} | {g.alz_rank.mean():.1f} ({', '.join(map(str, g.alz_rank))}) of {g.n_features.iloc[0]} | "
                 f"{g.alz_mean_abs_shap.mean():.4f} ± {g.alz_mean_abs_shap.std():.4f} | {g.top5.iloc[0]} |")
    L.append("")
    L.append("#### DEMO collapse check (cross-fitted, clinical-train with MRI)\n")
    L.append(f"Spearman ρ(DEMO, label) = {dr['spearman_rho']:.3f}; class-mean span = "
             f"{dr['class_mean_span']:.3f} → {'not collapsed: arm C run' if dr['not_collapsed'] else 'collapsed: arm C not run'}\n")
    L.append("| Class | Mean | SD | Min | Max |")
    L.append("|---|---|---|---|---|")
    for n, r in dr["per_class"].items():
        L.append(f"| {n} | {r['mean']:.3f} | {r['std']:.3f} | {r['min']:.3f} | {r['max']:.3f} |")
    with open(os.path.join(OUT, "ablation_tables.md"), "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
