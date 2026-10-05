"""P5, CNN row (LYP/.venv-mri, GPU): stability, plausibility and faithfulness of the chosen-stage LayerCAM.

Stability: global importance per model = mean d_r over the subjects it explains (13 regions).
  train: the 3 seed models within each fold, over that fold's train subjects (15 pairs);
  val: all 15 models over the 121 val subjects (105 pairs). Median pairwise Spearman; PASS >= 0.80, WEAK
  0.60-0.80, FAIL < 0.60. Per-subject: median over a subject's model pairs of the Spearman of d vectors.
Plausibility: fraction of train AD subjects with d_r > 1 for hippocampus, and for ventricles (same tiers).
Faithfulness: deletion / insertion of 0, 2, 5, 10, 20, 30, 40, 50% of brain voxels (CN-train mean replaces;
  insertion starts from the CN-train mean) in the order of the subject's map, vs 10 smoothed-noise orders
  (chosen stage's sigma; seed SeedSequence([RID, 100 + stage index, draw])). Output = mean ALZ probability over
  the subject's models. Per-subject normalised area difference (attribution - random). Deletion PASS if CI
  upper < 0; insertion PASS if CI lower > 0. Primary set: val AD (23); train AD reported (selection-biased).
Output: results/p5_rows.json["cnn"], results/p5_cnn_curves.csv (mean curves), figures/p5_cnn_faithfulness.png.
"""
import argparse
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import p1_lib as L  # noqa: E402

FRACS = np.array([0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50])
N_RAND = 10


def area(y):
    return np.trapezoid(y, FRACS) / (FRACS[-1] - FRACS[0])


def curves(models, vol, order, base, mask_size):
    """deletion and insertion curves (mean ALZ probability over models) for one voxel order."""
    dels, ins = [], []
    start = base.copy()
    for f in FRACS:
        k = int(round(f * mask_size))
        idx = order[:k]
        v = vol.copy().ravel()
        v[idx] = base.ravel()[idx]
        dels.append(v.reshape(vol.shape))
        w = start.copy().ravel()
        w[idx] = vol.ravel()[idx]
        ins.append(w.reshape(vol.shape))
    batch = np.stack(dels + ins)
    pr = np.mean([1 / (1 + np.exp(-L.alz_logit(m, batch))) for m in models], 0)
    return pr[:len(FRACS)], pr[len(FRACS):]


def faithfulness(d, z, pair, stage, mask, base, label):
    sel = d.reset_index(drop=True)
    rows, mean_curves = [], {"del_attr": [], "del_rand": [], "ins_attr": [], "ins_rand": []}
    si = L.STAGES.index(stage)
    groups = {}
    for i, r in sel.iterrows():
        groups.setdefault(tuple(L.models_of(r)), []).append(i)
    vols = L.load_volumes(list(sel.path))
    for mids, idx in groups.items():
        models = [L.get_model(m) for m in mids]
        for i in idx:
            r = sel.iloc[i]
            up = L.upsample(np.stack([z[f"layercam_{stage}"][pair[(r.PTID, m)]] for m in mids]).astype(np.float32),
                            mask).mean(0)
            order = L.brain_order(up, mask)
            da, ia = curves(models, vols[i], order, base, mask.sum())
            dr, ir = [], []
            for k in range(N_RAND):
                o = L.brain_order(L.noise_field((int(r.RID), 100 + si, k), L.SIGMA[stage]), mask)
                a_, b_ = curves(models, vols[i], o, base, mask.sum())
                dr.append(a_)
                ir.append(b_)
            dr, ir = np.mean(dr, 0), np.mean(ir, 0)
            rows.append({"PTID": r.PTID, "set": label, "del_diff": area(da) - area(dr), "ins_diff": area(ia) - area(ir)})
            for k, v in (("del_attr", da), ("del_rand", dr), ("ins_attr", ia), ("ins_rand", ir)):
                mean_curves[k].append(v)
        del models
        torch.cuda.empty_cache()
    return pd.DataFrame(rows), {k: np.mean(v, 0) for k, v in mean_curves.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    if a.dry:
        C.set_dry()
    C.verify_manifest()
    t0 = time.time()
    stage = json.load(open(os.path.join(C.RESULTS, "card_rules.json")))["cnn_stage"]["stage"]
    _, info = L.atlas()
    regions = info["regions"]
    dcols = [f"d_{g}" for g in regions]

    # stability
    pr = pd.read_csv(C.subj_path("p1_regions_pair.csv"))
    C.assert_no_test(L.subjects().set_index("PTID").loc[pr.PTID.unique()].RID)
    tr = pr[pr.split == "train"]
    pairs_tr = []
    for fold, g in tr.groupby("fold"):
        imp = g.groupby("model")[dcols].mean().to_numpy()
        for i in range(len(imp)):
            for j in range(i + 1, len(imp)):
                pairs_tr.append(C.spearman(imp[i], imp[j]))
    va = pr[pr.split == "val"]
    imp_va = va.groupby("model")[dcols].mean().to_numpy() if len(va) else np.empty((0, len(dcols)))
    st_val = C.median_pairwise_spearman(imp_va)[0] if len(imp_va) > 1 else np.nan

    def per_subject(g):
        v = g[dcols].to_numpy()
        return np.median([C.spearman(v[i], v[j]) for i in range(len(v)) for j in range(i + 1, len(v))])
    ps_tr = float(np.median([per_subject(g) for _, g in tr.groupby("PTID")]))
    ps_va = float(np.median([per_subject(g) for _, g in va.groupby("PTID")])) if len(va) else np.nan
    st_tr = float(np.median(pairs_tr))

    # plausibility
    sub = pd.read_csv(C.subj_path("p1_regions_subject.csv"))
    ch = sub[(sub.method == "layercam") & (sub.stage == stage)]
    ad_tr = ch[(ch.split == "train") & (ch.label == 3)]
    ad_va = ch[(ch.split == "val") & (ch.label == 3)]
    pl = {r: float((ad_tr[f"d_{r}"] > 1).mean()) for r in ("hippocampus", "ventricles")}
    pl_val = {r: float((ad_va[f"d_{r}"] > 1).mean()) if len(ad_va) else np.nan for r in ("hippocampus", "ventricles")}

    # faithfulness
    z = np.load(C.cache_path("cams.npz"))
    pair = {(p, m): i for i, (p, m) in enumerate(zip(z["pair_ptid"], z["pair_model"]))}
    mask = L.brain_mask()
    base = L.cn_train_mean()
    d = L.subjects()
    sets = {"val_AD": d[(d.split == "val") & (d.label == 3)], "train_AD": d[(d.split == "train") & (d.label == 3)]}
    if a.dry:
        sets = {"val_AD": d[d.PTID.isin(C.dry_run_ptids()) & (d.label == 3)].iloc[:1],
                "train_AD": d[d.PTID.isin(C.dry_run_ptids()) & (d.label == 3)]}
    faith, curve_rows = {}, []
    for name, g in sets.items():
        C.assert_no_test(g.RID)
        res, mc = faithfulness(g, z, pair, stage, mask, base, name)
        res.to_csv(C.subj_path(f"p5_cnn_faith_{name}.csv"), index=False)
        dm, dlo, dhi = C.boot_ci(res.del_diff.to_numpy())
        im, ilo, ihi = C.boot_ci(res.ins_diff.to_numpy())
        dp, ip = bool(dhi < 0), bool(ilo > 0)
        faith[name] = {"n": len(res), "deletion_diff": dm, "deletion_ci": [dlo, dhi], "deletion_pass": dp,
                       "insertion_diff": im, "insertion_ci": [ilo, ihi], "insertion_pass": ip,
                       "verdict": "PASS" if dp and ip else ("WEAK" if dp or ip else "FAIL")}
        for k, v in mc.items():
            for f, y in zip(FRACS, v):
                curve_rows.append({"set": name, "curve": k, "fraction": f, "mean_alz_prob": y})
        print(name, faith[name], f"{time.time() - t0:.0f}s", flush=True)
    cv = pd.DataFrame(curve_rows)
    C.write_aggregate(cv, "p5_cnn_curves.csv", ["train", "val"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, kind in zip(axes, ("del", "ins")):
        for name, col in (("val_AD", "#2a78d6"), ("train_AD", "#eb6834")):
            for curve, ls in ((f"{kind}_attr", "-"), (f"{kind}_rand", "--")):
                c = cv[(cv.set == name) & (cv.curve == curve)]
                ax.plot(c.fraction, c.mean_alz_prob, ls, color=col,
                        label=f"{name.replace('_', ' ')} {'attribution' if 'attr' in curve else 'random'}")
        ax.set_xlabel("fraction of brain voxels replaced" if kind == "del" else "fraction of brain voxels inserted")
        ax.set_ylabel("mean ALZ probability")
        ax.set_title("Deletion" if kind == "del" else "Insertion (from the CN-train mean)", fontsize=9)
    axes[1].legend(fontsize=7, frameon=False)
    fig.suptitle(f"CNN faithfulness, LayerCAM {stage}", fontsize=9)
    fig.tight_layout()
    os.makedirs(C.FIGURES, exist_ok=True)
    fig.savefig(os.path.join(C.FIGURES, "p5_cnn_faithfulness.png"), dpi=110)
    plt.close(fig)

    row = {"explanation": "CNN region attribution", "detail": f"LayerCAM {stage} ({L.STAGE_SIZE[stage]}^3)",
           "stability_train": st_tr, "stability_train_pairs": len(pairs_tr), "stability_val": st_val,
           "stability_per_subject_train": ps_tr, "stability_per_subject_val": ps_va,
           "plausibility_train": pl, "plausibility_val": pl_val, "faithfulness": faith}
    C.update_json("p5_rows.json", "cnn", row)
    print(json.dumps(row, indent=1, default=float)[:1500])
    C.verify_manifest()


if __name__ == "__main__":
    main()
