"""Interpretability figures and tables for the KAN F models (addendum K4). Never reads test.

Reads kan_interp.json (from kan_stage.py: the 5 all-569 F models, one per seed).
Writes:
  kan_edges.png       for ALZ, Hippocampus/ICV, Ventricles/ICV and the first 3 clinical
                      features of the frozen list: the 8 layer-0 edge functions (seed 42),
                      and the partial dependence of severity s and P(AD), 5 seeds overlaid
  kan_attribution.png pykan input attribution, mean over seeds
  kan_interp.md       flat / non-trivial verdict per input, attribution, pruning
x axes are in the inputs' natural units: ALZ probability, raw FreeSurfer
volume / ICV (via fusion_manifest.json), clinical scores via step4_manifest.json
(CogIndex is a composite of z-scores and stays in z units).
Run with LYP/.venv-kan.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FLAT = 0.10


def to_units(name, x):
    fm = json.load(open(os.path.join(HERE, "fusion_manifest.json")))
    sm = json.load(open(os.path.join(ROOT, "dataset", "processed", "step4_manifest.json")))["normalization"]
    x = np.asarray(x)
    if name in fm["ratio_zscore_train_mean"]:
        return x * fm["ratio_zscore_train_std"][name] + fm["ratio_zscore_train_mean"][name], f"{name.replace('_', ' / ')} (raw ratio)"
    if name in sm["mean"]:
        return x * sm["std"][name] + sm["mean"][name], f"{name} (raw score)"
    return x, ("ALZ (probability)" if name == "ALZ" else f"{name} (z units)")


def main():
    I = json.load(open(os.path.join(HERE, "kan_interp.json")))
    seeds = sorted(I, key=int)
    top3 = json.load(open(os.path.join(HERE, "topk.json")))["final_list"][:3]
    plot_inputs = ["ALZ", "Hippocampus_ICV", "Ventricles_ICV"] + top3
    cols = I[seeds[0]]["cols"]

    fig, ax = plt.subplots(len(plot_inputs), 3, figsize=(14, 3.0 * len(plot_inputs)))
    for r, c in enumerate(plot_inputs):
        cur = I[seeds[0]]["curves"][c]
        x, xl = to_units(c, cur["plot"]["x"])
        for e in cur["edges"]:
            ax[r, 0].plot(x, e, lw=1)
        ax[r, 0].set_title(f"{c}: 8 edge functions φ(x), seed {seeds[0]}", fontsize=9)
        for s in seeds:
            cs = I[s]["curves"][c]
            xs, _ = to_units(c, cs["plot"]["x"])
            ax[r, 1].plot(xs, cs["plot"]["severity"], lw=1, label=f"seed {s}")
            ax[r, 2].plot(xs, cs["plot"]["p_ad"], lw=1)
        rng = np.median([I[s]["curves"][c]["severity_range_p5_p95"] for s in seeds])
        ax[r, 1].set_title(f"severity s vs {c} (others at mean); P5–P95 range {rng:.2f} → "
                           f"{'flat' if rng < FLAT else 'non-trivial'}", fontsize=9)
        ax[r, 2].set_title(f"P(AD) vs {c}", fontsize=9)
        ax[r, 1].set_ylim(0, 3)
        ax[r, 2].set_ylim(0, 1)
        for k in range(3):
            ax[r, k].set_xlabel(xl, fontsize=8)
    ax[0, 1].legend(fontsize=7)
    fig.suptitle("KAN F (all-569 train models): learned layer-0 edge functions and partial dependence. "
                 "x range = train 1st–99th percentile", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "kan_edges.png"), dpi=90)
    plt.close(fig)

    # attribution, flatness, pruning
    att = pd.DataFrame({s: I[s]["attribution"] for s in seeds}).loc[cols]
    share = att / att.sum(0)
    rows = []
    for c in cols:
        rngs = [I[s]["curves"][c]["severity_range_p5_p95"] for s in seeds]
        rows.append({"input": c, "stream": "MRI" if c.endswith("_ICV") or c == "ALZ" else "clinical",
                     "severity_range_median": np.median(rngs), "severity_range_min": min(rngs),
                     "severity_range_max": max(rngs),
                     "verdict": "flat" if np.median(rngs) < FLAT else "non-trivial",
                     "attribution_share_mean": share.loc[c].mean(),
                     "attribution_rank_mean": att.rank(ascending=False).loc[c].mean(),
                     "kept_by_prune_input": sum(c in I[s]["pruned_kept"] for s in seeds)})
    t = pd.DataFrame(rows).sort_values("attribution_share_mean", ascending=False)
    t.to_csv(os.path.join(HERE, "kan_interp_summary.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    col = ["tab:orange" if s == "MRI" else "tab:blue" for s in t.stream]
    ax.barh(t.input[::-1], t.attribution_share_mean[::-1], color=col[::-1],
            xerr=share.loc[t.input[::-1]].std(1), capsize=2)
    ax.set_xlabel("pykan input attribution, share of total (mean ± SD over 5 seeds)")
    ax.set_title("KAN F: input attribution (orange = MRI stream)")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "kan_attribution.png"), dpi=90)
    plt.close(fig)

    L = ["# KAN F interpretability (5 all-569 models)\n",
         f"Flat = median over seeds of the severity partial-dependence range over the input's train "
         f"P5–P95 < {FLAT} stage units (addendum K4).\n",
         "| Input | Stream | s range, median [min, max] | Verdict | Attribution share | Mean rank | Kept by prune_input (of 5) |",
         "|---|---|---|---|---|---|---|"]
    for _, r in t.iterrows():
        L.append(f"| {r.input} | {r.stream} | {r.severity_range_median:.2f} [{r.severity_range_min:.2f}, "
                 f"{r.severity_range_max:.2f}] | {r.verdict} | {r.attribution_share_mean:.3f} | "
                 f"{r.attribution_rank_mean:.1f} | {r.kept_by_prune_input} |")
    L.append("\nprune_input(threshold=1e-2), no retraining:")
    for s in seeds:
        k = I[s]["pruned_kept"]
        L.append(f"- seed {s}: kept {len(k)} of {len(cols)}; dropped {[c for c in cols if c not in k]}; "
                 f"pruned val macro-F1 {I[s]['pruned_val_macro_f1']:.3f}")
    score_min = att.min().min()
    L.append(f"\nSmallest attribution score of any input in any seed: {score_min:.4f} (threshold 0.01).")
    with open(os.path.join(HERE, "kan_interp.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(att.round(3).to_string())


if __name__ == "__main__":
    main()
