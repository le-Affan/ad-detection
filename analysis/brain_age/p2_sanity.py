"""P2 sanity checks (p2_preregistration.md, section e). Label-based checks use
clinical-TRAIN subjects only; val only for coverage. Never reads test.
Writes p2_sanity.md, p2_site_effects.csv and p2_sanity_plots.png."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import kruskal, pearsonr, spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p2_ablation as P  # noqa: E402

NAMES = ["CN", "EMCI", "LMCI", "AD"]
VOLS = ["Ventricles", "Hippocampus", "WholeBrain", "Entorhinal", "Fusiform", "MidTemp", "ICV"]


def main():
    ba = pd.read_csv(P.BRAINAGE)
    L = ["# P2 sanity checks\n"]

    # coverage
    L.append("## Coverage\n")
    L.append("| Split | Subjects | With brain age | Missing |")
    L.append("|---|---|---|---|")
    for s, g in ba.groupby("split"):
        miss = g[~g.has_brainage]
        L.append(f"| {s} | {len(g)} | {int(g.has_brainage.sum())} | "
                 + "; ".join(f"{r.PTID} ({r.DX_bl}, {r.missing_reason})" for r in miss.itertuples()) + " |")
    L.append("")

    tr, _ = P.load()
    tr = tr[tr.has_brainage].copy()
    al, be, n = P.fit_bias(tr)
    tr["BAG"] = P.bag(tr, al, be)
    tr["corr_age"] = (tr.pred_age - be) / al
    cn = tr[tr.label == 0]

    # 1. CN accuracy, raw and corrected
    L.append("## 1. CN train: accuracy, raw vs bias-corrected\n")
    L.append(f"Bias fit on {n} CN train subjects: predicted = {al:.3f}·age + {be:.2f}\n")
    L.append("| | MAE | mean error | r with age | SD pred / SD age |")
    L.append("|---|---|---|---|---|")
    for name, col in (("raw", "pred_age"), ("corrected", "corr_age")):
        e = cn[col] - cn.age_at_scan
        L.append(f"| {name} | {e.abs().mean():.2f} | {e.mean():+.2f} | {pearsonr(cn[col], cn.age_at_scan)[0]:.3f} | "
                 f"{cn[col].std() / cn.age_at_scan.std():.3f} |")
    L.append("")

    # 2. BAG by diagnosis
    L.append("## 2. Corrected gap by diagnosis (train)\n")
    L.append("| Diagnosis | n | Mean BAG | SD | Median | Raw gap mean |")
    L.append("|---|---|---|---|---|---|")
    groups = []
    for k, nm in enumerate(NAMES):
        g = tr[tr.label == k]
        groups.append(g.BAG.to_numpy())
        L.append(f"| {nm} | {len(g)} | {g.BAG.mean():+.2f} | {g.BAG.std():.2f} | {g.BAG.median():+.2f} | "
                 f"{(g.pred_age - g.age_at_scan).mean():+.2f} |")
    H, p = kruskal(*groups)
    L.append(f"\nKruskal–Wallis across the 4 groups: H = {H:.2f}, p = {p:.2e}\n")

    # 3. BAG vs age in CN
    r3, p3 = pearsonr(cn.BAG, cn.age_at_scan)
    r3all = pearsonr(tr.BAG, tr.age_at_scan)[0]
    L.append("## 3. Corrected gap vs age\n")
    L.append(f"CN train: r(BAG, age) = {r3:.3f} (p = {p3:.2f}). "
             f"Raw gap in CN: r = {pearsonr(cn.pred_age - cn.age_at_scan, cn.age_at_scan)[0]:.3f}. "
             f"All train diagnoses: r(BAG, age) = {r3all:.3f}.\n")

    # 4. site effect
    cn = cn.assign(site=cn.RID.map(pd.read_csv(P.BRAINAGE).set_index("RID").PTID).str[:3])
    mu, sd = cn.BAG.mean(), cn.BAG.std()
    site = cn.groupby("site").BAG.agg(["count", "mean"]).reset_index()
    site["z_se"] = (site["mean"] - mu) / (sd / np.sqrt(site["count"]))
    site["z_sd"] = (site["mean"] - mu) / sd
    site["flag"] = (site["count"] >= 5) & (site.z_se.abs() > 2)
    site.sort_values("count", ascending=False).round(3).to_csv(os.path.join(HERE, "p2_site_effects.csv"), index=False)
    big = site[site["count"] >= 5].sort_values("z_se")
    L.append("## 4. Site effect (CN train, site = PTID prefix)\n")
    L.append(f"{cn.site.nunique()} sites; {len(big)} have ≥ 5 CN subjects. CN mean BAG {mu:+.2f}, SD {sd:.2f}.\n")
    L.append("| Site | CN n | Mean BAG | z (SE) | z (SD) | Flag |")
    L.append("|---|---|---|---|---|---|")
    for r in big.itertuples():
        L.append(f"| {r.site} | {r.count} | {r.mean:+.2f} | {r.z_se:+.2f} | {r.z_sd:+.2f} | {'**FLAG**' if r.flag else ''} |")
    L.append(f"\nFlagged (≥5 CN, |z_SE| > 2): {', '.join(big[big.flag].site) or 'none'}. "
             f"Any site with |mean − overall| > 2 SD of individual CN gaps: "
             f"{', '.join(site[(site['count'] >= 5) & (site.z_sd.abs() > 2)].site) or 'none'}.\n")

    # 5. overlap with ALZ and FreeSurfer volumes
    L.append("## 5. Overlap with existing inputs (train, Spearman)\n")
    L.append("| Variable | ρ with BAG (all train) | ρ with BAG (CN train) |")
    L.append("|---|---|---|")
    for v in ["ALZ"] + VOLS + ["AGE"]:
        a = tr[[v, "BAG"]].dropna()
        c = cn[[v, "BAG"]].dropna()
        L.append(f"| {v} | {spearmanr(a[v], a.BAG)[0]:+.3f} | {spearmanr(c[v], c.BAG)[0]:+.3f} |")
    from sklearn.linear_model import LinearRegression
    X = tr[VOLS + ["ALZ", "AGE"]].to_numpy()
    r2 = LinearRegression().fit(X, tr.BAG).score(X, tr.BAG)
    L.append(f"\nIn-sample R² of BAG on the 7 volumes + ALZ + AGE (linear, train): {r2:.3f}. "
             f"So about {100 * (1 - r2):.0f}% of BAG's variance is not linearly explained by them.\n")

    # 6. standalone signal, CV with per-fold correction (seed-42 folds)
    full, _ = P.load()
    y = full.label.to_numpy()
    oof = np.full(len(full), np.nan)
    for a_idx, b_idx in StratifiedKFold(5, shuffle=True, random_state=42).split(full, y):
        fa, fb, _ = P.fit_bias(full.iloc[a_idx])
        oof[b_idx] = P.bag(full.iloc[b_idx], fa, fb)
    ok = ~np.isnan(oof)
    adcn = ok & np.isin(y, [0, 3])
    L.append("## 6. Standalone signal (corrected gap as the score, 5-fold CV, per-fold correction)\n")
    L.append(f"- AD vs CN: AUC {roc_auc_score(y[adcn] == 3, oof[adcn]):.3f} (n = {int(adcn.sum())})")
    L.append(f"- AD vs rest: AUC {roc_auc_score(y[ok] == 3, oof[ok]):.3f} (n = {int(ok.sum())})")
    for k in (1, 2):
        m = ok & np.isin(y, [0, k])
        L.append(f"- {NAMES[k]} vs CN: AUC {roc_auc_score(y[m] == k, oof[m]):.3f}")
    alz = full.ALZ.to_numpy()
    m = ok & ~np.isnan(alz)
    L.append(f"- For reference, cross-fitted ALZ on the same subjects: AD vs rest AUC {roc_auc_score(y[m] == 3, alz[m]):.3f}\n")

    with open(os.path.join(HERE, "p2_sanity.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    colors = ["#2a7ab9", "#6aa84f", "#e6a23c", "#c8553d"]
    for k, nm in enumerate(NAMES):
        g = tr[tr.label == k]
        ax[0].scatter(g.age_at_scan, g.pred_age, s=9, alpha=0.6, color=colors[k], label=nm)
    xs = np.array([55, 92])
    ax[0].plot(xs, xs, "--", color="0.6", lw=1, label="identity")
    ax[0].plot(xs, al * xs + be, color="k", lw=1, label=f"CN fit ({al:.2f}·age + {be:.1f})")
    ax[0].set(xlabel="Age at scan", ylabel="DeepBrainNet predicted age", title="Raw predictions (train)")
    ax[0].legend(fontsize=7, frameon=False)
    ax[1].boxplot([tr[tr.label == k].BAG for k in range(4)], showfliers=False)
    ax[1].set_xticks(range(1, 5), NAMES)
    ax[1].axhline(0, color="0.6", ls="--", lw=1)
    ax[1].set(ylabel="Corrected gap (years)", title=f"Corrected gap by diagnosis (train), KW p={p:.1e}")
    ax[2].scatter(cn.age_at_scan, cn.BAG, s=10, color=colors[0])
    ax[2].axhline(0, color="0.6", ls="--", lw=1)
    ax[2].set(xlabel="Age at scan", ylabel="Corrected gap (years)", title=f"CN train: corrected gap vs age (r={r3:.2f})")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "p2_sanity_plots.png"), dpi=100)


if __name__ == "__main__":
    main()
