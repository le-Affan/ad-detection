"""Frozen system, step 3 (run with LYP/.venv-fusion): metrics on one split.

Every system is its 5-seed average (the final KAN averages 25 KANs).
Systems:
  - clinical specialist;
  - MRI specialist standalone (= MRI-only mode: MRI inputs + age + sex, no
    cognitive tests), on subjects with a scan;
  - KAN fusion (final);
  - CatBoost F and CatBoost R (references).
Metrics:
  - macro-F1, accuracy, balanced accuracy, per-class F1, AD AUC (COG P(AD)),
    the ADD head's AUC, 3-class (CN / MCI / AD), confusion matrices;
  - severity checks;
  - paired bootstrap CIs (2,000 resamples of subjects, rng seed 0) for KAN −
    clinical, KAN − CatBoost F and KAN − CatBoost R;
  - MRI roles (a)–(c).
--split test only after freeze.md is committed and unmodified; run once.
Output: frozen_run/<split>_results.json and <split>_results.md.
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import metrics as M  # noqa: E402
from freeze_guard import assert_frozen  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["val", "test"])
    a = ap.parse_args()
    if a.split == "test":
        print("freeze commit:", assert_frozen())
    z = np.load(os.path.join(HERE, "frozen_run", f"{a.split}_streams.npz"))
    k = np.load(os.path.join(HERE, "frozen_run", f"{a.split}_kan.npz"))
    y, has = z["y"], z["has"]
    avg = lambda key: np.mean([z[f"{key}_{s}"] for s in M.SEEDS], 0)          # noqa: E731
    P = {"Clinical specialist": avg("clin"), "KAN fusion (final, K2)": k["p"],
         "CatBoost F (reference)": avg("cbF"), "CatBoost R (reference)": avg("cbR")}
    mri = avg("mri")
    res = {"n": int(len(y)), "n_with_scan": int(has.sum()), "labels": np.bincount(y, minlength=4).tolist(),
           "n_reliable_brain_age": int(z["ba_ok"].sum()), "systems": {}, "confusion": {}}
    for s, p in P.items():
        res["systems"][s] = {**M.metrics(y, p), **M.metrics3(y, p)}
        res["confusion"][s] = M.confusion(y, p.argmax(1)).tolist()
    res["systems"]["KAN fusion (final, K2)"]["add_head_auc"] = float(roc_auc_score(y == 3, k["add"]))
    res["systems"]["MRI specialist (MRI-only mode), scan subjects"] = {**M.metrics(y[has], mri[has]),
                                                                      **M.metrics3(y[has], mri[has])}
    res["confusion"]["MRI specialist"] = M.confusion(y[has], mri[has].argmax(1)).tolist()
    res["systems"]["Clinical specialist, scan subjects"] = {**M.metrics(y[has], P["Clinical specialist"][has]),
                                                           **M.metrics3(y[has], P["Clinical specialist"][has])}
    maj = 1                                                                   # train majority class (EMCI)
    pm = np.full((has.sum(), 4), 1e-9)
    pm[:, maj] = 1
    res["systems"]["Majority class (EMCI), scan subjects"] = {**M.metrics(y[has], pm), **M.metrics3(y[has], pm)}

    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(y), size=(2000, len(y)))
    fin = P["KAN fusion (final, K2)"].argmax(1)[None]
    res["boot"] = {}
    for ref in ("Clinical specialist", "CatBoost F (reference)", "CatBoost R (reference)"):
        r = P[ref].argmax(1)[None]
        d = (M.macro_f1_boot(y, fin, idx) - M.macro_f1_boot(y, r, idx))[:, 0]
        res["boot"][f"KAN − {ref}"] = {"diff": res["systems"]["KAN fusion (final, K2)"]["macro_f1"]
                                       - res["systems"][ref]["macro_f1"],
                                       "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5))}
    cf, cc = np.array(res["confusion"]["KAN fusion (final, K2)"]), np.array(res["confusion"]["Clinical specialist"])
    res["fusion_vs_clinical_errors"] = {"LMCI_to_AD": [int(cc[2, 3]), int(cf[2, 3])],
                                        "AD_to_LMCI": [int(cc[3, 2]), int(cf[3, 2])],
                                        "LMCI_to_EMCI": [int(cc[2, 1]), int(cf[2, 1])],
                                        "EMCI_to_LMCI": [int(cc[1, 2]), int(cf[1, 2])]}
    res["severity"] = {}
    for s, p in P.items():
        sv = p @ np.arange(4)
        means = [float(sv[y == c].mean()) for c in range(4)]
        res["severity"][s] = {"means": means, "monotonic": bool(np.all(np.diff(means) > 0)),
                              "rho_stage": float(spearmanr(sv, y)[0]), "rho_cdrsb": float(spearmanr(sv, z["cdrsb"])[0])}
    json.dump(res, open(os.path.join(HERE, "frozen_run", f"{a.split}_results.json"), "w"), indent=1, default=float)

    cols = ["macro_f1", "acc", "bal_acc", "f1_CN", "f1_EMCI", "f1_LMCI", "f1_AD", "ad_auc", "macro_f1_3", "acc_3"]
    L = [f"# Frozen system on {a.split} (n = {len(y)}; labels CN/EMCI/LMCI/AD {res['labels']})\n",
         "| System | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for s, m in res["systems"].items():
        L.append(f"| {s} | " + " | ".join(f"{m[c]:.3f}" for c in cols) + " |")
    L.append(f"\nKAN ADD-head AUC {res['systems']['KAN fusion (final, K2)']['add_head_auc']:.3f}\n")
    for kk, b in res["boot"].items():
        L.append(f"- {kk}: {b['diff']:+.4f} [{b['lo']:+.4f}, {b['hi']:+.4f}]")
    L.append(f"\nErrors clinical → fusion: {res['fusion_vs_clinical_errors']}\n")
    for s, v in res["severity"].items():
        L.append(f"- severity {s}: means {[round(x, 2) for x in v['means']]}, monotonic {v['monotonic']}, "
                 f"ρ stage {v['rho_stage']:.3f}, ρ CDRSB {v['rho_cdrsb']:.3f}")
    L.append("")
    for s, c in res["confusion"].items():
        L.append(f"- confusion {s}: {c}")
    open(os.path.join(HERE, "frozen_run", f"{a.split}_results.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
