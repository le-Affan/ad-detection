"""Threshold-refit diagnostic: is acc4 limited by binning or by representation?

For every per-checkpoint CSV in analysis/threshold_refit/, compares the 4-class
metrics under the cutpoints m3_train.py implies (midpoints of DEMO_VALUES, i.e.
0.25 / 0.75 / 1.50) against cutpoints grid-searched on val to maximise macro-F1.

THE FITTED CUTPOINTS ARE A CEILING, NOT A SCORE. Three parameters fitted on the
same 121 subjects they are scored on cannot be read as generalisation.

Binning convention matches demo_to_label exactly: np.argmin over |x - value|
breaks ties toward the lower class, so a boundary value falls in the LOWER bin.
That is `np.digitize(x, cuts, right=True)`.
"""
import glob
import json
import os

import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(ROOT, "analysis", "threshold_refit")
CLASSES = ["CN", "EMCI", "LMCI", "AD"]
CURRENT = (0.25, 0.75, 1.50)


def confusion(true, pred):
    m = np.zeros((4, 4), dtype=int)
    for t, p in zip(true, pred):
        m[t, p] += 1
    return m


def macro_f1_from_conf(c):
    """c[true, pred]. Returns macro-F1 over the 4 classes, zero_division=0."""
    tp = np.diag(c).astype(float)
    pred_tot = c.sum(0).astype(float)
    true_tot = c.sum(1).astype(float)
    den = pred_tot + true_tot
    f1 = np.divide(2 * tp, den, out=np.zeros(4), where=den > 0)
    return f1.mean(), f1


def _sweep(cum, tot, grid, want_max=None):
    """One vectorised pass over all c1<c2<c3 triples.

    Returns (best_mf1, best_triple) when want_max is None, otherwise the number
    of triples whose macro-F1 equals want_max (to expose a flat optimum).
    """
    M = len(grid)
    pb, pc = np.triu_indices(M, k=1)
    best_v, best_t, n_tie = -1.0, None, 0
    for a in range(M - 2):
        sel = pb > a
        b, c = pb[sel], pc[sel]
        if b.size == 0:
            continue
        n0 = np.repeat(cum[:, a][:, None], b.size, axis=1)
        n1 = cum[:, b] - n0
        n2 = cum[:, c] - cum[:, b]
        n3 = tot[:, None] - cum[:, c]
        conf = np.stack([n0, n1, n2, n3], axis=2)            # (4, T, 4)
        tp = np.stack([conf[k, :, k] for k in range(4)], axis=1).astype(float)
        den = conf.sum(0).astype(float) + tot[None, :]
        f1 = np.divide(2 * tp, den, out=np.zeros_like(den), where=den > 0)
        mf1 = f1.mean(1)
        if want_max is None:
            i = int(np.argmax(mf1))
            if mf1[i] > best_v:
                best_v, best_t = float(mf1[i]), (grid[a], grid[b[i]], grid[c[i]])
        else:
            n_tie += int((np.abs(mf1 - want_max) < 1e-12).sum())
    return (best_v, best_t) if want_max is None else n_tie


def fit_cutpoints(demo, labels, step=0.01):
    """Grid-search c1<c2<c3 maximising macro-F1 over the observed range.

    Uses cumulative per-class counts so each candidate triple costs O(1): the
    whole 4x4 confusion matrix is differences of cumulative counts, which makes
    the full C(M,3) sweep vectorisable instead of a per-triple relabelling.
    """
    lo = np.floor(demo.min() * 100) / 100
    hi = np.ceil(demo.max() * 100) / 100
    grid = np.round(np.arange(lo, hi + step / 2, step), 10)
    tot = np.array([(labels == k).sum() for k in range(4)], dtype=np.int64)
    cum = np.empty((4, len(grid)), dtype=np.int64)
    for k in range(4):
        d = np.sort(demo[labels == k])
        cum[k] = np.searchsorted(d, grid, side="right")   # count of <= grid[m]
    best_v, best_t = _sweep(cum, tot, grid)
    n_tie = _sweep(cum, tot, grid, want_max=best_v)
    return best_t, best_v, len(grid), n_tie


def describe(demo, labels):
    rows = [("ALL", demo)]
    for k in range(4):
        rows.append((CLASSES[k], demo[labels == k]))
    out = []
    for name, v in rows:
        out.append((name, len(v), v.min(), v.max(), v.mean(), v.std(ddof=0)))
    return out


def main():
    manifest = json.load(open(os.path.join(DIR, "_manifest.json")))
    order = {m["tag"]: m for m in manifest}
    results = []
    for path in sorted(glob.glob(os.path.join(DIR, "*.csv"))):
        tag = os.path.basename(path)[:-4]
        d = np.genfromtxt(path, delimiter=",", names=True)
        labels = d["true_class"].astype(int)
        demo = d["demo_pred"].astype(float)

        pred_cur = np.digitize(demo, CURRENT, right=True)
        c_cur = confusion(labels, pred_cur)
        mf1_cur, f1_cur = macro_f1_from_conf(c_cur)
        acc_cur = float((pred_cur == labels).mean())

        cuts, mf1_fit, gridN, nopt = fit_cutpoints(demo, labels)
        pred_fit = np.digitize(demo, cuts, right=True)
        c_fit = confusion(labels, pred_fit)
        _, f1_fit = macro_f1_from_conf(c_fit)
        acc_fit = float((pred_fit == labels).mean())

        rho, pval = spearmanr(demo, labels)
        results.append(dict(
            tag=tag, meta=order.get(tag, {}),
            acc_cur=acc_cur, mf1_cur=mf1_cur, f1_cur=f1_cur, conf_cur=c_cur,
            acc_fit=acc_fit, mf1_fit=mf1_fit, f1_fit=f1_fit, conf_fit=c_fit,
            cuts=cuts, grid=gridN, nopt=nopt,
            rho=float(rho), pval=float(pval),
            stats=describe(demo, labels),
            frac_gt_150=float((demo > 1.50).mean()),
            n_gt_150=int((demo > 1.50).sum()),
            demo=demo, labels=labels))
    np.save(os.path.join(DIR, "_results.npy"),
            np.array(results, dtype=object), allow_pickle=True)

    # ---- summary table -------------------------------------------------
    print("=" * 118)
    print(f"{'checkpoint':30s} {'arch':10s} {'ep':>3s} | {'acc4':>6s} {'mF1':>6s} "
          f"| {'acc4*':>6s} {'mF1*':>6s} | {'c1':>5s} {'c2':>5s} {'c3':>5s} "
          f"| {'rho':>6s} {'p':>9s} | {'>1.5':>5s}")
    print("-" * 118)
    for r in sorted(results, key=lambda x: x["tag"]):
        m = r["meta"]
        print(f"{r['tag']:30s} {str(m.get('arch','?')):10s} {str(m.get('epoch','?')):>3s} "
              f"| {r['acc_cur']:6.3f} {r['mf1_cur']:6.3f} "
              f"| {r['acc_fit']:6.3f} {r['mf1_fit']:6.3f} "
              f"| {r['cuts'][0]:5.2f} {r['cuts'][1]:5.2f} {r['cuts'][2]:5.2f} "
              f"| {r['rho']:6.3f} {r['pval']:9.2e} | {r['n_gt_150']:5d}")
    print("=" * 118)
    print("acc4*/mF1* = fitted-on-val cutpoints: a diagnostic CEILING, not a score.")
    return results


if __name__ == "__main__":
    main()
