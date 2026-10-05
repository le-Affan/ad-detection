"""P2 (LYP/.venv-fusion): exact linear SHAP for the frozen MRI specialist, its checks, and its P5 row.

Explainer (mri_shap_lib): interventional linear SHAP of the centred class scores; background = the 567 train
subjects with a scan in the specialist's own preprocessing. The 5 frozen seed pickles are asserted identical.
P2-R1 additivity: |sum phi + base - centred score| < 1e-6, and shap.LinearExplainer agrees within 1e-8.
P2-R2 comparison with M0-M5: groups volumes / ALZ+BAG / age+sex, train mean |phi| on the AD centred score.
P5 row: stability from 200 bootstrap refits of an in-memory copy (P2-R4; never saved); plausibility for
hippocampus and ventricles; deletion / insertion faithfulness on P(predicted class).
Outputs: results/p2_global_{train,val}.csv, results/p2_additivity.json, results/p2_vs_controls.{csv,md},
figures/p2_global_AD.png, results/p5_rows.json["mri_shap"]; subject_level/p2_mri_shap.npz (gitignored).
"""
import argparse
import os
import pickle
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import mri_shap_lib as MS  # noqa: E402
import tab_data as TD  # noqa: E402
import tab_faith as TF  # noqa: E402

sys.path.insert(0, C.V3)
import specialists as S  # noqa: E402

N_REFIT = 200
AD = 3


def load_frozen():
    ms = [pickle.load(open(os.path.join(C.FROZEN, f"mri_{s}.pkl"), "rb")) for s in (42, 43, 44, 45, 46)]
    for m in ms[1:]:
        assert np.array_equal(m.model.coef_, ms[0].model.coef_) and np.array_equal(m.med, ms[0].med)
    return ms[0]


def importance(phi):
    """global importance: mean |phi| over subjects, averaged over the 4 class targets -> (10,)"""
    return np.abs(phi).mean(0).mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    if a.dry:
        C.set_dry()
    C.verify_manifest()
    t0 = time.time()
    mri = load_frozen()
    d = TD.trainval()
    d = d[d.has_mri].reset_index(drop=True)
    tr = (d.split == "train").to_numpy()
    assert tr.sum() == 567 and (~tr).sum() == 121
    z, miss = MS.design(mri, d)
    z_bg = z[tr]
    phi, base = MS.shap_values(mri, z, z_bg)
    sc = MS.centred_scores(mri, z)

    # P2-R1 additivity
    err = float(np.abs(phi.sum(1) + base - sc).max())
    import shap
    Wc, bc = MS.centred_coef(mri)
    ex = shap.LinearExplainer((Wc, bc), shap.maskers.Independent(z_bg, max_samples=len(z_bg)))
    sv = np.asarray(ex.shap_values(z))
    sv = sv if sv.shape == phi.shape else np.moveaxis(sv, 0, -1)
    err_shap = float(np.abs(sv - phi).max())
    add = {"max_abs_additivity_error": err, "pass_1e-6": err < 1e-6, "max_abs_diff_vs_LinearExplainer": err_shap,
           "pass_1e-8": err_shap < 1e-8, "n_subjects": len(d), "seed_copies_identical": True}
    print(add, flush=True)
    assert add["pass_1e-6"] and add["pass_1e-8"], "P2-R1 failed: bug"
    C.write_json(add, "p2_additivity.json")
    np.savez_compressed(C.subj_path("p2_mri_shap.npz"), ptid=d.PTID.to_numpy(), split=d.split.to_numpy(),
                        phi=phi.astype(np.float32), base=base, imputed=miss, features=np.array(MS.FEATURES))

    # global importance per class
    for split, rows in (("train", tr), ("val", ~tr)):
        out = []
        for k, nm in enumerate(C.NAMES):
            for j, f in enumerate(MS.FEATURES):
                m, lo, hi = C.boot_ci(np.abs(phi[rows, j, k]))
                out.append({"class_target": nm, "feature": f, "mean_abs_phi": m, "ci_lo": lo, "ci_hi": hi,
                            "mean_phi": float(phi[rows, j, k].mean()), "n": int(rows.sum()),
                            "n_imputed": int(miss[rows, j].sum())})
        C.write_aggregate(pd.DataFrame(out), f"p2_global_{split}.csv", [split])

    # P2-R2 groups on the AD centred score
    gi = {g: [MS.FEATURES.index(f) for f in fs] for g, fs in MS.GROUPS.items()}
    absAD = np.abs(phi[tr, :, AD])
    grp = {g: absAD[:, ix].sum(1) for g, ix in gi.items()}
    tot = absAD.sum(1)
    rows = []
    for g, v in grp.items():
        m, lo, hi = C.boot_ci(v)
        rows.append({"group": g, "mean_abs_phi_AD": m, "ci_lo": lo, "ci_hi": hi})
    idx = C.boot_idx(tr.sum())
    for r in rows:
        v = grp[r["group"]]
        share = v.sum() / tot.sum()
        bs = np.array([v[i].sum() / tot[i].sum() for i in idx])
        r.update(share=share, share_ci_lo=np.percentile(bs, 2.5), share_ci_hi=np.percentile(bs, 97.5))
    gt = pd.DataFrame(rows)
    order = list(gt.sort_values("mean_abs_phi_AD", ascending=False).group)
    agree = order == ["volumes", "ALZ+BAG", "age+sex"]
    gt["controls_standalone_cv_auc"] = gt.group.map({"volumes": 0.867, "ALZ+BAG": 0.788, "age+sex": 0.574})
    C.write_aggregate(gt, "p2_vs_controls.csv", ["train"])
    md = ["# MRI-specialist SHAP vs the M0–M5 input-set controls (P2-R2)", "",
          "Train subjects with a scan (n = 567). Mean |SHAP| on the AD centred score, summed per input group; "
          "share of the total with bootstrap 95% CI.", "",
          "| Group | Mean \\|φ\\| (AD) [95% CI] | Share [95% CI] | Controls: standalone CV AD AUC |", "|---|---|---|---|"]
    for r in gt.itertuples():
        md.append(f"| {r.group} | {r.mean_abs_phi_AD:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}] | {r.share:.2f} "
                  f"[{r.share_ci_lo:.2f}, {r.share_ci_hi:.2f}] | {r.controls_standalone_cv_auc:.3f} |")
    md += ["", f"SHAP order: {' > '.join(order)}. Controls order: volumes > ALZ+BAG > age+sex. "
               f"**{'Agree' if agree else 'Disagree'}** (pre-registered rule P2-R2)."]
    open(os.path.join(C.RESULTS, "p2_vs_controls.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md[-1:]), flush=True)

    # figure: train mean |phi| for AD with CI
    g = pd.read_csv(os.path.join(C.RESULTS, "p2_global_train.csv"))
    g = g[g.class_target == "AD"].sort_values("mean_abs_phi")
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    col = ["#2a78d6" if v > 0 else "#eb6834" for v in g.mean_phi]
    ax.barh([MS.LABELS[f] for f in g.feature], g.mean_abs_phi, xerr=[g.mean_abs_phi - g.ci_lo, g.ci_hi - g.mean_abs_phi],
            color="#2a78d6", height=0.6, error_kw={"lw": 0.8})
    ax.set_xlabel("mean |SHAP| on the AD centred score (train, n = 567)")
    ax.set_title("MRI specialist: what drives its AD score", fontsize=9)
    fig.tight_layout()
    os.makedirs(C.FIGURES, exist_ok=True)
    fig.savefig(os.path.join(C.FIGURES, "p2_global_AD.png"), dpi=110)
    plt.close(fig)
    del col

    # --- P5 row ---------------------------------------------------------------------------------------
    # stability: 200 bootstrap refits of an in-memory copy (P2-R4), never saved
    rng = np.random.default_rng(0)
    dtr = d[tr].reset_index(drop=True)
    n_refit = 5 if a.dry else N_REFIT
    imps, imps_val, ps = [], [], []
    dva = d[~tr].reset_index(drop=True)
    imp_frozen = importance(phi[tr])
    for b in range(n_refit):
        samp = dtr.iloc[rng.integers(0, len(dtr), len(dtr))].reset_index(drop=True)
        cp = S.MRI("logreg", 42, C=0.1).fit(samp)                          # in-memory copy; never written
        zb, _ = MS.design(cp, dtr)
        ph, _ = MS.shap_values(cp, zb, zb)
        imps.append(importance(ph))
        zv, _ = MS.design(cp, dva)
        imps_val.append(importance(MS.shap_values(cp, zv, zb)[0]))
        ps.append(np.median([C.spearman(ph[i, :, AD], phi[tr][i, :, AD]) for i in range(0, len(dtr), 7)]))
    st, npairs = C.median_pairwise_spearman(np.array(imps))
    st_val = C.median_pairwise_spearman(np.array(imps_val))[0]
    vs_frozen = float(np.median([C.spearman(v, imp_frozen) for v in imps]))
    print(f"stability: median pairwise {st:.3f} ({npairs} pairs), vs frozen {vs_frozen:.3f}", flush=True)

    # plausibility (raw ratios vs CN-train median)
    pl = {}
    for f, col, hi in (("Hippocampus_ICV", "raw_Hippocampus_ICV", False), ("Ventricles_ICV", "raw_Ventricles_ICV", True)):
        med = TD.cn_train_median(d, col)
        j = MS.FEATURES.index(f)
        out = {}
        for split, rows in (("train", tr), ("val", ~tr)):
            e = TD.expected_sign(d.loc[rows, col], med, hi)
            p = phi[rows, j, AD]
            ok = (e != 0) & (np.abs(p) > 1e-6)
            out[split] = float((np.sign(p[ok]) == e[ok]).mean()) if ok.any() else np.nan
            out[f"n_{split}"] = int(ok.sum())
        pl[f.replace("_ICV", "").lower()] = out
    print("plausibility", pl, flush=True)

    # faithfulness on P(predicted class)
    brng = np.random.default_rng(0)
    B = z_bg[brng.choice(len(z_bg), 50, replace=False)]
    W, b0 = mri.model.coef_, mri.model.intercept_

    def predict(Z):
        return MS.softmax(Z @ W.T + b0)
    pred = predict(z).argmax(1)
    faith = {}
    sel_rows = {"train": np.flatnonzero(tr), "val": np.flatnonzero(~tr)}
    if a.dry:
        sel_rows = {"train": np.flatnonzero(tr & d.PTID.isin(C.dry_run_ptids()).to_numpy()), "val": np.flatnonzero(~tr)[:3]}
    curves = []
    for split, ix in sel_rows.items():
        res = [TF.subject_faithfulness(z[i], B, phi[i, :, pred[i]], d.RID.iloc[i], 2, predict, pred[i]) for i in ix]
        dd = np.array([r["del_diff"] for r in res])
        ii = np.array([r["ins_diff"] for r in res])
        dm, dlo, dhi = C.boot_ci(dd)
        im, ilo, ihi = C.boot_ci(ii)
        dp, ip, v = TF.verdict(dhi, ilo)
        faith[split] = {"n": len(ix), "deletion_diff": dm, "deletion_ci": [dlo, dhi], "deletion_pass": dp,
                        "insertion_diff": im, "insertion_ci": [ilo, ihi], "insertion_pass": ip, "verdict": v}
        for k in ("del_attr", "del_rand", "ins_attr", "ins_rand"):
            for step, y in enumerate(np.mean([r[k] for r in res], 0)):
                curves.append({"split": split, "curve": k, "step": step, "mean_prob": y})
        print(split, faith[split], flush=True)
    C.write_aggregate(pd.DataFrame(curves), "p5_mri_curves.csv", ["train", "val"])
    row = {"explanation": "MRI-specialist SHAP", "detail": "exact linear SHAP, centred class scores, background 567 train",
           "stability_train": st, "stability_train_pairs": npairs, "stability_source": f"{n_refit} bootstrap refits",
           "stability_vs_frozen_median": vs_frozen, "stability_per_subject_train": float(np.median(ps)),
           "stability_val": st_val, "plausibility": pl, "faithfulness": faith,
           "group_agreement_with_controls": agree, "group_order": order}
    C.update_json("p5_rows.json", "mri_shap", row)
    print(f"done {time.time() - t0:.0f}s")
    C.verify_manifest()


if __name__ == "__main__":
    main()
