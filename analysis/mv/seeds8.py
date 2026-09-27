"""8-seed R0 vs R2 on validation: v9 reproduction and the class-weighting verdict.

Applies the rules pre-registered in docs/mv_static_report.md (commit 02dafac)
mechanically, so the verdict is computed rather than read off a table:

  class weighting is a real gain only if BOTH
    (1) mean paired macro-F1 gain >= 0.05, or the two-sided 95% t-interval of
        the paired differences (df = 7) excludes zero; and
    (2) R2 beats R0 (strictly) on >= 7 of 8 seeds.

Every metric is read from the best-val-loss epoch of each run's epoch_log.csv,
on the 121 val subjects. Val only: nothing here builds a dataset at all.
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.join(ROOT, "analysis", "mv")
SEEDS = list(range(42, 50))
ARMS = {"R0": "bl_non", "R2": "bl_inv"}
V9_LOGGED = 0.349
METRICS = ["acc4", "macro_f1", "bal_acc", "f1_CN", "f1_EMCI", "f1_LMCI", "f1_AD", "alz_auc"]


def best_row(run):
    e = pd.read_csv(os.path.join(HERE, run, "epoch_log.csv"))
    r = e.loc[e.val_loss.idxmin()]
    return {"epoch": int(r.epoch), **{m: float(r["val_" + m]) for m in METRICS}}


def main():
    rows = []
    for arm, prefix in ARMS.items():
        for s in SEEDS:
            run = f"{prefix}_s{s}"
            if not os.path.exists(os.path.join(HERE, run + ".log")) or \
                    "best val loss" not in open(os.path.join(HERE, run + ".log")).read():
                raise SystemExit(f"{run} has not finished")
            rows.append({"arm": arm, "seed": s, **best_row(run)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "seeds8_runs.csv"), index=False)

    print("per-arm, mean ± SD over 8 seeds (best-val-loss epoch, 121 val subjects)\n")
    tab = df.groupby("arm")[METRICS + ["epoch"]].agg(["mean", "std"])
    for arm in ARMS:
        print(f"{arm}: " + "  ".join(
            f"{m} {tab.loc[arm, (m, 'mean')]:.3f}±{tab.loc[arm, (m, 'std')]:.3f}"
            for m in METRICS + ["epoch"]))

    p = df.pivot(index="seed", columns="arm", values="macro_f1")
    d = (p.R2 - p.R0).to_numpy()
    n = len(d)
    mean, sd = d.mean(), d.std(ddof=1)
    t = stats.t.ppf(0.975, n - 1)
    lo, hi = mean - t * sd / np.sqrt(n), mean + t * sd / np.sqrt(n)
    wins = int((d > 0).sum())
    print("\npaired macro-F1, R2 - R0:")
    for s, a, b, x in zip(p.index, p.R0, p.R2, d):
        print(f"  seed {s}: R0 {a:.3f}  R2 {b:.3f}  diff {x:+.3f}")
    print(f"  mean {mean:+.3f}  sd {sd:.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
          f"(t={t:.3f}, df={n - 1})  wins {wins}/{n}")
    c1 = mean >= 0.05 or lo > 0 or hi < 0
    c2 = wins >= 7
    verdict = "real gain" if (c1 and c2) else "no reliable gain"
    print(f"  rule (1) {'MET' if c1 else 'not met'}; rule (2) {'MET' if c2 else 'not met'}"
          f"  ->  class weighting: {verdict.upper()}")

    r0 = df[df.arm == "R0"].macro_f1
    z = (V9_LOGGED - r0.mean()) / r0.std(ddof=1)
    reach = int((r0 >= V9_LOGGED).sum())
    print(f"\nv9 reproduction: R0 macro-F1 {r0.mean():.3f} ± {r0.std(ddof=1):.3f} "
          f"(range {r0.min():.3f}–{r0.max():.3f}); Kaggle v9 {V9_LOGGED} is z = {z:+.2f}, "
          f"reached by {reach}/8 seeds")

    pd.DataFrame({"seed": p.index, "R0": p.R0.values, "R2": p.R2.values, "diff": d}).to_csv(
        os.path.join(HERE, "seeds8_paired.csv"), index=False)


if __name__ == "__main__":
    main()
