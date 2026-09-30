"""Step 0 gate: do the ADNI FreeSurfer volumes add staging accuracy? Never reads test.

Implements step0_preregistration.md. Reuses the fusion-feasibility harness
(analysis/fusion_feasibility/ablation.py: load, fit, metrics, paired bootstrap,
fixed-265-tree CatBoost, seeds 42-46, repeated stratified 5-fold CV over the
569 clinical-train subjects, train-all -> clinical-val). The only additions
are the four arms and the SHAP ranks of the FreeSurfer columns.

Outputs (analysis/mri_morphometry/): step0_runs.csv, step0_oof.npz,
step0_shap.csv, step0_tables.md.
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

FREESURFER = ["Ventricles", "Hippocampus", "WholeBrain", "Entorhinal", "Fusiform", "MidTemp", "ICV"]
PAIRS = [("A", "A0"), ("A'", "A0'")]


def arms_for(tr):
    base = [c for c in tr.columns if c not in ("RID", "label", "ALZ", "DEMO", "has_mri")]
    assert len(base) == 36 and all(c in base for c in H.DEFINING + FREESURFER)
    red = [c for c in base if c not in H.DEFINING]
    return {"A": base, "A0": [c for c in base if c not in FREESURFER],
            "A'": red, "A0'": [c for c in red if c not in FREESURFER]}


def shap_ranks(m, df, cols):
    sv = np.abs(np.asarray(shap.TreeExplainer(m).shap_values(df[cols])))
    imp = sv.mean(axis=(0, 2)) if sv.ndim == 3 else sv.mean(0)
    order = list(np.array(cols)[np.argsort(-imp)])
    return {c: (order.index(c) + 1, float(imp[cols.index(c)])) for c in cols}


def main():
    tr, va = H.load(os.path.join(FF, "alz_crossfit.csv"))
    arms = arms_for(tr)
    print({a: len(c) for a, c in arms.items()})
    ytr, yva = tr.label.to_numpy(), va.label.to_numpy()
    oof = {a: np.zeros((len(H.SEEDS), len(tr), 4)) for a in arms}
    vpr = {a: np.zeros((len(H.SEEDS), len(va), 4)) for a in arms}
    shap_rows = []

    for si, seed in enumerate(H.SEEDS):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for a_idx, b_idx in skf.split(tr, ytr):
            for a, cols in arms.items():
                m = H.fit(tr.iloc[a_idx][cols], ytr[a_idx], seed)
                oof[a][si, b_idx] = m.predict_proba(tr.iloc[b_idx][cols])
        for a, cols in arms.items():
            m = H.fit(tr[cols], ytr, seed)
            vpr[a][si] = m.predict_proba(va[cols])
            if a in ("A", "A'"):
                for split, df in (("val", va), ("train (in-sample)", tr)):
                    for c, (rank, imp) in shap_ranks(m, df, cols).items():
                        shap_rows.append({"arm": a, "seed": seed, "split": split, "feature": c,
                                          "rank": rank, "mean_abs_shap": imp, "n_features": len(cols)})
        print(f"seed {seed} done")

    runs = []
    for a in arms:
        for si, seed in enumerate(H.SEEDS):
            for ev, y, P in (("cv", ytr, oof[a][si]), ("val", yva, vpr[a][si])):
                runs.append({"arm": a, "seed": seed, "eval": ev, "n": len(y), **H.metrics(y, P)})
    runs = pd.DataFrame(runs)
    runs.to_csv(os.path.join(HERE, "step0_runs.csv"), index=False)
    np.savez(os.path.join(HERE, "step0_oof.npz"), train_rids=tr.RID.to_numpy(), val_rids=va.RID.to_numpy(),
             **{f"oof_{a}": oof[a] for a in arms}, **{f"val_{a}": vpr[a] for a in arms})
    shap_df = pd.DataFrame(shap_rows)
    shap_df.to_csv(os.path.join(HERE, "step0_shap.csv"), index=False)

    # paired bootstrap, as in the harness (rng seed 0, 2000 resamples of subjects)
    rng = np.random.default_rng(0)
    boots = {}
    for ev, y, P in (("cv", ytr, oof), ("val", yva, vpr)):
        idx = rng.integers(0, len(y), size=(H.N_BOOT, len(y)))
        for b, a in PAIRS:
            pb, pa = P[b].argmax(-1), P[a].argmax(-1)
            d = (H.macro_f1_boot(y, pb, idx) - H.macro_f1_boot(y, pa, idx)).mean(1)
            per_seed = (runs.query("eval==@ev and arm==@b").macro_f1.values
                        - runs.query("eval==@ev and arm==@a").macro_f1.values)
            boots[(ev, b, a)] = {"diff": float(per_seed.mean()), "lo": np.percentile(d, 2.5),
                                 "hi": np.percentile(d, 97.5), "per_seed": per_seed}

    verdicts = {}
    for b, a in PAIRS:
        cv, v = boots[("cv", b, a)], boots[("val", b, a)]
        verdicts[f"{b} vs {a}"] = "beats" if cv["lo"] > 0 and v["diff"] > 0 else "does not beat"
    matters = any(v == "beats" for v in verdicts.values())

    write_tables(runs, boots, verdicts, matters, shap_df, arms)
    print(open(os.path.join(HERE, "step0_tables.md")).read())


def write_tables(runs, boots, verdicts, matters, shap_df, arms):
    L = []
    cols = ["macro_f1", "f1_CN", "f1_EMCI", "f1_LMCI", "f1_AD", "bal_acc", "ad_auc", "acc"]
    hdr = ["Macro-F1", "CN F1", "EMCI F1", "LMCI F1", "AD F1", "Bal. acc", "AD AUC", "Acc"]
    for ev, title in (("cv", "Cross-validation (569 clinical-train, OOF, 5 seeds)"),
                      ("val", "Validation (train on 569, evaluate on 122 clinical-val, 5 seeds)")):
        g = runs[runs["eval"] == ev]
        L.append(f"#### {title}\n")
        L.append("| Arm | n feat | " + " | ".join(hdr) + " |")
        L.append("|---|---|" + "---|" * len(hdr))
        for a in arms:
            r = g[g.arm == a]
            L.append(f"| {a} | {len(arms[a])} | " + " | ".join(f"{r[c].mean():.3f} ± {r[c].std():.3f}" for c in cols) + " |")
        L.append("")
    L.append(f"#### Paired differences (with − without FreeSurfer; {H.N_BOOT} bootstrap resamples, mean over 5 seeds)\n")
    L.append("| Pair | Eval | Macro-F1 diff | 95% CI | Per-seed | LMCI F1 diff | AD F1 diff | Bal. acc diff | AD AUC diff |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for (ev, b, a), r in boots.items():
        gb = runs.query("eval==@ev and arm==@b").reset_index()
        ga = runs.query("eval==@ev and arm==@a").reset_index()
        dd = {c: (gb[c] - ga[c]).mean() for c in ("f1_LMCI", "f1_AD", "bal_acc", "ad_auc")}
        L.append(f"| {b} − {a} | {ev} | {r['diff']:+.4f} | [{r['lo']:+.4f}, {r['hi']:+.4f}] | "
                 + " ".join(f"{x:+.3f}" for x in r["per_seed"])
                 + f" | {dd['f1_LMCI']:+.3f} | {dd['f1_AD']:+.3f} | {dd['bal_acc']:+.3f} | {dd['ad_auc']:+.4f} |")
    L.append("")
    L.append("#### Verdict against the pre-registered rule\n")
    for k, v in verdicts.items():
        L.append(f"- {k}: **{v}**")
    L.append(f"- **Morphometry {'matters' if matters else 'does not matter'}** (rule: A beats A0 OR A′ beats A0′)")
    L.append("")
    L.append("#### SHAP ranks of the FreeSurfer columns (full-train models; rank mean over 5 seeds, per-seed in brackets)\n")
    L.append("| Feature | A, val | A, train in-sample | A′, val | A′, train in-sample |")
    L.append("|---|---|---|---|---|")
    for f in FREESURFER:
        cells = []
        for a in ("A", "A'"):
            for sp in ("val", "train (in-sample)"):
                g = shap_df[(shap_df.arm == a) & (shap_df.split == sp) & (shap_df.feature == f)]
                cells.append(f"{g['rank'].mean():.1f} ({', '.join(map(str, g['rank']))}) of {g.n_features.iloc[0]}")
        L.append(f"| {f} | " + " | ".join(cells) + " |")
    L.append("")
    with open(os.path.join(HERE, "step0_tables.md"), "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
