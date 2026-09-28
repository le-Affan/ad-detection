"""Score dbn_smoke_test.csv against the pre-registered criteria (p1_report.md,
commit 1954763) and draw the two plots. Reads only the smoke-test CSV."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(os.path.join(HERE, "dbn_smoke_test.csv"))
cn, ad = d[d.DX_bl == "CN"], d[d.DX_bl == "AD"]

res = []
r = np.corrcoef(cn.pred_age, cn.age_at_scan)[0, 1]
res.append(("a1", "r(pred, age), CN", r, ">= 0.6", r >= 0.6))
ratio = cn.pred_age.std() / cn.age_at_scan.std()
res.append(("a2", "SD(pred)/SD(age), CN", ratio, ">= 0.6", ratio >= 0.6))
mae = cn.gap.abs().mean()
res.append(("b", "MAE, CN (years)", mae, "<= 6", mae <= 6))
dg = ad.gap.mean() - cn.gap.mean()
res.append(("c", "mean gap AD - CN (years)", dg, ">= 2", dg >= 2))
for k in ("x0.9", "x1.1"):
    m = (d[f"pred_{k}"] - d.pred_age).abs().median()
    res.append(("d", f"median |Δpred|, intensity {k} (years)", m, "<= 2", m <= 2))
res = pd.DataFrame(res, columns=["crit", "measure", "value", "threshold", "pass"])
print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print("\nOVERALL:", "PASS" if res["pass"].all() else "FAIL")

print("\nDescriptive")
for dx, g in ((("CN", cn)), ("AD", ad)):
    print(f"  {dx}: n={len(g)} age {g.age_at_scan.mean():.1f} ± {g.age_at_scan.std():.1f} "
          f"({g.age_at_scan.min():.1f}-{g.age_at_scan.max():.1f})  pred {g.pred_age.mean():.1f} ± {g.pred_age.std():.1f} "
          f"({g.pred_age.min():.1f}-{g.pred_age.max():.1f})  MAE {g.gap.abs().mean():.2f}  gap {g.gap.mean():+.2f} ± {g.gap.std():.2f}")
    print(f"      slope pred~age {np.polyfit(g.age_at_scan, g.pred_age, 1)[0]:.2f}  r {np.corrcoef(g.pred_age, g.age_at_scan)[0, 1]:.3f}")
for k in ("x0.9", "x1.1"):
    s = d[f"pred_{k}"] - d.pred_age
    print(f"  {k}: mean Δ {s.mean():+.2f}  max |Δ| {s.abs().max():.2f}  corr with base {np.corrcoef(d[f'pred_{k}'], d.pred_age)[0, 1]:.3f}")
over80 = d[d.age_at_scan > 80]
print(f"  >80: n={len(over80)} ({', '.join(f'{a}:{p:.0f}' for a, p in zip(over80.DX_bl, over80.pred_age))})  mean gap {over80.gap.mean():+.2f}")
cn80 = cn[cn.age_at_scan <= 80]
print(f"  CN <=80: n={len(cn80)} MAE {cn80.gap.abs().mean():.2f}  r {np.corrcoef(cn80.pred_age, cn80.age_at_scan)[0, 1]:.3f}")
# age-matched view of c (descriptive): residual of gap on age, fitted on CN only
b1, b0 = np.polyfit(cn.age_at_scan, cn.gap, 1)
resid = d.gap - (b0 + b1 * d.age_at_scan)
print(f"  gap regressed on age within CN (slope {b1:.2f}): AD - CN residual gap "
      f"{resid[d.DX_bl == 'AD'].mean() - resid[d.DX_bl == 'CN'].mean():+.2f}  (descriptive only, not the criterion)")
p1 = pd.read_csv(os.path.join(HERE, "smoke_test.csv"))[["PTID", "pred_age"]].rename(columns={"pred_age": "sfcn"})
m = d.merge(p1, on="PTID")
print(f"  P1 overlap n={len(m)}: DBN vs SFCN pred r {np.corrcoef(m.pred_age, m.sfcn)[0, 1]:.3f}; "
      f"r with age DBN {np.corrcoef(m.pred_age, m.age_at_scan)[0, 1]:.3f} vs SFCN {np.corrcoef(m.sfcn, m.age_at_scan)[0, 1]:.3f}")
res.to_csv(os.path.join(HERE, "dbn_criteria.csv"), index=False)

colors = {"CN": "#2a7ab9", "AD": "#c8553d"}
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
lo, hi = 50, 95
ax[0].plot([lo, hi], [lo, hi], color="0.6", lw=1, ls="--", label="identity")
for dx, g in d.groupby("DX_bl"):
    ax[0].scatter(g.age_at_scan, g.pred_age, s=28, color=colors[dx], label=f"{dx} (n={len(g)})")
ax[0].set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Chronological age at scan (years)",
          ylabel="DeepBrainNet predicted age (years)", title="Predicted vs chronological age")
ax[0].legend(frameon=False, fontsize=8)
for i, dx in enumerate(("CN", "AD")):
    g = d[d.DX_bl == dx].gap
    ax[1].boxplot(g, positions=[i], widths=0.5, showfliers=False)
    ax[1].scatter(np.full(len(g), i) + np.random.default_rng(0).uniform(-0.12, 0.12, len(g)), g, s=22, color=colors[dx])
ax[1].axhline(0, color="0.6", lw=1, ls="--")
ax[1].set(xticks=[0, 1], xticklabels=["CN", "AD"], ylabel="Gap = predicted − chronological (years)",
          title="Brain-age gap by diagnosis (uncorrected)")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "dbn_smoke_plots.png"), dpi=110)
