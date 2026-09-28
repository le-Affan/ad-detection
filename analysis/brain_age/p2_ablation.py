"""P2 ablation: does the bias-corrected brain-age gap (BAG) add value? Never reads test.

Implements p2_preregistration.md (commit 2936320). Reuses the fusion-feasibility
harness (analysis/fusion_feasibility/ablation.py: load, fit, metrics, paired
bootstrap, fixed-265-tree CatBoost, seeds, CV scheme); the only additions are
the brain-age join, the per-fold CN-only bias correction and the new arms.

Outputs (analysis/brain_age/): p2_ablation_runs.csv, p2_ablation_oof.npz,
p2_bias_fits.csv, p2_ablation_tables.md.
"""
import os
import sys

import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FF = os.path.join(ROOT, "analysis", "fusion_feasibility")
sys.path.insert(0, FF)
import ablation as H  # noqa: E402  the fusion-feasibility harness

BRAINAGE = os.path.join(HERE, "brainage_trainval.csv")
PAIRS = [("E", "B"), ("E'", "B'"), ("G", "A")]       # G - A is descriptive only
DECISIVE = [("E", "B"), ("E'", "B'")]


# --- bias correction (pre-registration a) ----------------------------------------
def fit_bias(df):
    """OLS predicted = alpha*age + beta on CN rows with brain age."""
    cn = df[(df.label == 0) & df.pred_age.notna()]
    alpha, beta = np.polyfit(cn.age_at_scan, cn.pred_age, 1)
    return float(alpha), float(beta), len(cn)


def bag(df, alpha, beta):
    return (df.pred_age - beta) / alpha - df.age_at_scan


def load():
    tr, va = H.load(os.path.join(FF, "alz_crossfit.csv"))
    ba = pd.read_csv(BRAINAGE).set_index("RID")
    for df, s in ((tr, "train"), (va, "val")):
        b = ba[ba.split == s]
        assert set(b.index) == set(df.RID), f"{s}: brain-age RIDs != step4 RIDs"
        df["pred_age"] = df.RID.map(b.pred_age)
        df["age_at_scan"] = df.RID.map(b.age_at_scan)
        df["has_brainage"] = df.RID.map(b.has_brainage).astype(bool)
    return tr, va


def arms_for(tr):
    base = [c for c in tr.columns if c not in
            ("RID", "label", "ALZ", "DEMO", "has_mri", "pred_age", "age_at_scan", "has_brainage", "BAG")]
    assert len(base) == 36 and all(c in base for c in H.DEFINING)
    red = [c for c in base if c not in H.DEFINING]
    return {"A": base, "B": base + ["ALZ"], "E": base + ["ALZ", "BAG"],
            "A'": red, "B'": red + ["ALZ"], "E'": red + ["ALZ", "BAG"], "G": base + ["BAG"]}


def main():
    tr, va = load()
    arms = arms_for(tr)
    ytr, yva = tr.label.to_numpy(), va.label.to_numpy()
    sub_tr, sub_va = tr.has_brainage.to_numpy(), va.has_brainage.to_numpy()
    oof = {a: np.zeros((len(H.SEEDS), len(tr), 4)) for a in arms}
    vpr = {a: np.zeros((len(H.SEEDS), len(va), 4)) for a in arms}
    fits, shap_rows = [], []

    for si, seed in enumerate(H.SEEDS):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fi, (a_idx, b_idx) in enumerate(skf.split(tr, ytr)):
            t_in, t_ev = tr.iloc[a_idx].copy(), tr.iloc[b_idx].copy()
            al, be, n = fit_bias(t_in)                      # training-fold CN only
            fits.append({"eval": "cv", "seed": seed, "fold": fi, "alpha": al, "beta": be, "n_cn": n})
            t_in["BAG"], t_ev["BAG"] = bag(t_in, al, be), bag(t_ev, al, be)
            for a, cols in arms.items():
                m = H.fit(t_in[cols], ytr[a_idx], seed)
                oof[a][si, b_idx] = m.predict_proba(t_ev[cols])
        al, be, n = fit_bias(tr)                            # all clinical-train CN
        fits.append({"eval": "val", "seed": seed, "fold": -1, "alpha": al, "beta": be, "n_cn": n})
        t_all, v_all = tr.assign(BAG=bag(tr, al, be)), va.assign(BAG=bag(va, al, be))
        for a, cols in arms.items():
            m = H.fit(t_all[cols], ytr, seed)
            vpr[a][si] = m.predict_proba(v_all[cols])
            if "BAG" in cols:
                sv = np.abs(np.asarray(shap.TreeExplainer(m).shap_values(v_all[cols])))
                imp = sv.mean(axis=(0, 2)) if sv.ndim == 3 else sv.mean(0)
                order = list(np.array(cols)[np.argsort(-imp)])
                shap_rows.append({"arm": a, "seed": seed, "bag_rank": order.index("BAG") + 1,
                                  "n_features": len(cols), "bag_mean_abs_shap": float(imp[cols.index("BAG")]),
                                  "alz_rank": order.index("ALZ") + 1 if "ALZ" in cols else None,
                                  "top5": ", ".join(order[:5])})
        print(f"seed {seed} done", flush=True)

    runs = []
    for a in arms:
        for si, seed in enumerate(H.SEEDS):
            for ev, y, P, mask in (("cv", ytr, oof[a][si], sub_tr), ("val", yva, vpr[a][si], sub_va)):
                for sub, mk in (("all", np.ones_like(mask)), ("brainage", mask)):
                    runs.append({"arm": a, "seed": seed, "eval": ev, "subset": sub, "n": int(mk.sum()),
                                 **H.metrics(y[mk], P[mk])})
    runs = pd.DataFrame(runs)
    runs.to_csv(os.path.join(HERE, "p2_ablation_runs.csv"), index=False)
    pd.DataFrame(fits).to_csv(os.path.join(HERE, "p2_bias_fits.csv"), index=False)
    np.savez(os.path.join(HERE, "p2_ablation_oof.npz"), train_rids=tr.RID.to_numpy(), val_rids=va.RID.to_numpy(),
             **{f"oof_{a}": oof[a] for a in arms}, **{f"val_{a}": vpr[a] for a in arms})

    # harness check: A, B, A', B' must reproduce fusion-feasibility exactly
    ff = pd.read_csv(os.path.join(FF, "ablation_runs.csv"))
    ff = ff[ff.subset == "all"].set_index(["arm", "seed", "eval"])
    mine = runs[(runs.subset == "all") & runs.arm.isin(["A", "B", "A'", "B'"])].set_index(["arm", "seed", "eval"])
    cols = ["macro_f1", "bal_acc", "ad_auc", "f1_CN", "f1_EMCI", "f1_LMCI", "f1_AD"]
    maxdiff = float((mine[cols] - ff.loc[mine.index, cols]).abs().max().max())
    print(f"harness check vs fusion-feasibility (A, B, A', B'): max |diff| = {maxdiff:.2e}")

    rng = np.random.default_rng(0)
    boots = {}
    for ev, y, P, mask in (("cv", ytr, oof, sub_tr), ("val", yva, vpr, sub_va)):
        for sub, mk in (("all", np.ones_like(mask)), ("brainage", mask)):
            yy = y[mk]
            idx = rng.integers(0, len(yy), size=(H.N_BOOT, len(yy)))
            for b, a in PAIRS:
                pb, pa = P[b][:, mk].argmax(-1), P[a][:, mk].argmax(-1)
                d = (H.macro_f1_boot(yy, pb, idx) - H.macro_f1_boot(yy, pa, idx)).mean(1)
                per_seed = [H.f1s_from_cm(H.confusion(yy, pb[s]))[0].mean()
                            - H.f1s_from_cm(H.confusion(yy, pa[s]))[0].mean() for s in range(len(H.SEEDS))]
                boots[(ev, sub, b, a)] = {"diff": float(np.mean(per_seed)), "lo": float(np.percentile(d, 2.5)),
                                          "hi": float(np.percentile(d, 97.5)), "per_seed": per_seed}
    verdicts = {}
    for b, a in DECISIVE:
        cv, v = boots[("cv", "all", b, a)], boots[("val", "all", b, a)]
        excl = cv["lo"] > 0 or cv["hi"] < 0
        same = np.sign(cv["diff"]) == np.sign(v["diff"]) and v["diff"] != 0
        verdicts[f"{b} vs {a}"] = ("beats" if cv["diff"] > 0 else "reliably worse") if (excl and same) else "no reliable gain"
    adopted = any(v == "beats" for v in verdicts.values())

    cms = {a: np.mean([H.confusion(yva, vpr[a][s].argmax(1)) for s in range(len(H.SEEDS))], 0) for a in arms}
    write_tables(runs, boots, verdicts, adopted, cms, pd.DataFrame(shap_rows), pd.DataFrame(fits), maxdiff)
    print(open(os.path.join(HERE, "p2_ablation_tables.md")).read())


def write_tables(runs, boots, verdicts, adopted, cms, shap_df, fits, maxdiff):
    L = []
    cols = ["macro_f1", "f1_CN", "f1_EMCI", "f1_LMCI", "f1_AD", "bal_acc", "ad_auc", "acc"]
    hdr = ["Macro-F1", "CN F1", "EMCI F1", "LMCI F1", "AD F1", "Bal. acc", "AD AUC", "Acc"]
    arms = list(dict.fromkeys(runs.arm))
    for ev, title in (("cv", "CV (569 clinical-train, OOF, 5 seeds)"), ("val", "Val (train 569 → 122 clinical-val, 5 seeds)")):
        for sub in ("all", "brainage"):
            g = runs[(runs["eval"] == ev) & (runs.subset == sub)]
            L.append(f"#### {title} — {'all subjects' if sub == 'all' else 'subjects with brain age'} (n = {int(g.n.iloc[0])})\n")
            L.append("| Arm | " + " | ".join(hdr) + " |")
            L.append("|---|" + "---|" * len(hdr))
            for a in arms:
                r = g[g.arm == a]
                L.append(f"| {a} | " + " | ".join(f"{r[c].mean():.3f} ± {r[c].std():.3f}" for c in cols) + " |")
            L.append("")
    L.append("#### Paired macro-F1 differences (2,000 bootstrap resamples, mean over 5 seeds)\n")
    L.append("| Pair | Eval | Subset | Diff | 95% CI | Per-seed (42–46) |")
    L.append("|---|---|---|---|---|---|")
    for (ev, sub, b, a), r in boots.items():
        L.append(f"| {b} − {a} | {ev} | {sub} | {r['diff']:+.4f} | [{r['lo']:+.4f}, {r['hi']:+.4f}] | "
                 + " ".join(f"{x:+.3f}" for x in r["per_seed"]) + " |")
    L.append("")
    L.append("#### Verdicts (pre-registered rule)\n")
    for k, v in verdicts.items():
        L.append(f"- **{k}: {v}**")
    L.append(f"- **Brain-age gap: {'FUSION INPUT' if adopted else 'CARD ONLY'}**\n")
    L.append(f"Harness check (A, B, A′, B′ vs fusion-feasibility per-seed metrics): max |diff| = {maxdiff:.2e}\n")
    L.append("#### Val confusion matrices, mean over seeds (rows true, cols predicted)\n")
    for a in arms:
        L.append(f"**{a}**\n")
        L.append("| | " + " | ".join(H.NAMES) + " |")
        L.append("|---|---|---|---|---|")
        for i, n in enumerate(H.NAMES):
            L.append(f"| {n} | " + " | ".join(f"{x:.1f}" for x in cms[a][i]) + " |")
        L.append("")
    L.append("#### SHAP of BAG (val, models trained on all 569)\n")
    L.append("| Arm | BAG rank (per seed) | BAG mean abs SHAP | ALZ rank (mean) | Top 5 (seed 42) |")
    L.append("|---|---|---|---|---|")
    for a, g in shap_df.groupby("arm", sort=False):
        L.append(f"| {a} | {g.bag_rank.mean():.1f} ({', '.join(map(str, g.bag_rank))}) of {g.n_features.iloc[0]} | "
                 f"{g.bag_mean_abs_shap.mean():.4f} ± {g.bag_mean_abs_shap.std():.4f} | "
                 f"{'' if g.alz_rank.isna().all() else f'{g.alz_rank.mean():.1f}'} | {g.top5.iloc[0]} |")
    L.append("")
    f = fits[fits["eval"] == "cv"]
    L.append(f"Bias fits: CV folds alpha {f.alpha.mean():.3f} (range {f.alpha.min():.3f}–{f.alpha.max():.3f}), "
             f"beta {f.beta.mean():.2f} (range {f.beta.min():.2f}–{f.beta.max():.2f}), CN per fold {f.n_cn.min()}–{f.n_cn.max()}; "
             f"val fit alpha {fits[fits['eval'] == 'val'].alpha.iloc[0]:.3f}, beta {fits[fits['eval'] == 'val'].beta.iloc[0]:.2f}.")
    with open(os.path.join(HERE, "p2_ablation_tables.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
