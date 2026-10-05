"""P1 step 3 (LYP/.venv-mri, CPU): region attribution, per-class tables and the naming rule (P1-R2, P1-R3).

Per subject (train and val), for every method x stage: the subject map = mean over its models of the upsampled,
brain-masked, unit-mass model maps. share_r = mass in region r / brain mass; d_r = share_r / (voxels_r /
voxels_brain). Region "other" is tabulated but never named.
Aggregates (committed): results/p1_regions_train.csv, results/p1_regions_val.csv (mean share and d_r per true
class, bootstrap CIs), results/p1_ad_vs_cn.csv (AD - CN difference in mean d_r, train), results/p1_naming.json
and card_rules.json["cnn_naming"] (chosen stage, LayerCAM). Figures: figures/p1_group_maps.png (train per-class
mean maps, chosen stage) and figures/p1_group_maps_gradcam_layer4.png (the previous card method, for reference).
Subject-level (gitignored): subject_level/p1_regions_subject.csv, subject_level/p1_regions_pair.csv (per model,
chosen stage, for P5 stability).
"""
import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import p1_lib as L  # noqa: E402


def region_stats(maps, atl, regions, vox, nb):
    """maps (n, 96^3) unit mass -> shares (n, R), densities (n, R)."""
    flat = atl.ravel()
    sh = np.stack([maps.reshape(len(maps), -1)[:, flat == i + 1].sum(1) for i in range(len(regions))], 1)
    tot = sh.sum(1, keepdims=True)
    sh = sh / np.where(tot > 0, tot, 1)
    dens = sh / (np.array([vox[r] for r in regions]) / nb)[None]
    return sh, dens


def diff_ci(a, b, B=C.B_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    bs = np.array([a[rng.integers(0, len(a), len(a))].mean(0) - b[rng.integers(0, len(b), len(b))].mean(0)
                   for _ in range(B)])
    return a.mean(0) - b.mean(0), np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)


def group_figure(path, maps_by_class, tmpl, slices, title):
    fig, axes = plt.subplots(4, 3, figsize=(9, 11.5))
    for r, (cls, m) in enumerate(maps_by_class):
        top = m.max()
        for c, (sl, origin, nm) in enumerate(((lambda v: v[slices["axial_S"]], "upper", "axial"),
                                              (lambda v: v[:, slices["coronal_P"]], "lower", "coronal"),
                                              (lambda v: v[:, :, slices["sagittal_R"]], "lower", "sagittal (left)"))):
            ax = axes[r, c]
            ax.imshow(sl(tmpl), cmap="gray", origin=origin)
            ax.imshow(np.ma.masked_where(sl(m) <= 0, sl(m)), cmap="inferno", origin=origin, alpha=0.6, vmin=0,
                      vmax=top)
            ax.axis("off")
            ax.set_title(f"{cls} · {nm}", fontsize=8)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    if a.dry:
        C.set_dry()
    C.verify_manifest()
    chosen = json.load(open(os.path.join(C.RESULTS, "card_rules.json")))["cnn_stage"]["stage"]
    z = np.load(C.cache_path("cams.npz"))
    d = L.subjects()
    d = d[d.PTID.isin(set(z["pair_ptid"]))].reset_index(drop=True)
    C.assert_no_test(d.RID)
    atl, info = L.atlas()
    regions, vox, nb = info["regions"], info["voxels"], info["brain_voxels"]
    mask = L.brain_mask()
    pairs_of = pd.Series(np.arange(len(z["pair_ptid"]))).groupby(z["pair_ptid"]).apply(list).to_dict()
    sub_rows, pair_rows = [], []
    class_sum = {(m, s): np.zeros((4, 96, 96, 96)) for m in L.METHODS for s in L.STAGES}
    class_n = np.zeros(4)
    for i, r in enumerate(d.itertuples()):
        idx = pairs_of[r.PTID]
        if r.split == "train":
            class_n[r.label] += 1
        for meth in L.METHODS:
            for s in L.STAGES:
                up = L.upsample(z[f"{meth}_{s}"][idx].astype(np.float32), mask)
                subj = up.mean(0)
                tot = subj.sum()
                subj = subj / tot if tot > 0 else subj
                sh, de = region_stats(subj[None], atl, regions, vox, nb)
                row = {"PTID": r.PTID, "split": r.split, "label": r.label, "method": meth, "stage": s,
                       "empty": bool(tot <= 0)}
                row.update({f"share_{g}": sh[0, j] for j, g in enumerate(regions)})
                row.update({f"d_{g}": de[0, j] for j, g in enumerate(regions)})
                sub_rows.append(row)
                if r.split == "train":
                    class_sum[(meth, s)][r.label] += subj
                if meth == "layercam" and s == chosen:
                    shp, dep = region_stats(up, atl, regions, vox, nb)
                    for k, pi in enumerate(idx):
                        pr = {"PTID": r.PTID, "split": r.split, "label": r.label, "fold": r.fold,
                              "model": int(z["pair_model"][pi])}
                        pr.update({f"d_{g}": dep[k, j] for j, g in enumerate(regions)})
                        pair_rows.append(pr)
        if i % 100 == 0:
            print(f"{i}/{len(d)}", flush=True)
    sub = pd.DataFrame(sub_rows)
    pairs = pd.DataFrame(pair_rows)
    sub.to_csv(C.subj_path("p1_regions_subject.csv"), index=False)
    pairs.to_csv(C.subj_path("p1_regions_pair.csv"), index=False)

    # --- per-class tables -------------------------------------------------------------------------------
    for split in ("train", "val"):
        out = []
        for (meth, s), g in sub[sub.split == split].groupby(["method", "stage"], sort=False):
            for lab in range(4):
                h = g[(g.label == lab) & ~g.empty]
                for reg in regions:
                    ms, ls, hs = C.boot_ci(h[f"share_{reg}"].to_numpy())
                    md, ld, hd = C.boot_ci(h[f"d_{reg}"].to_numpy())
                    out.append({"method": meth, "stage": s, "chosen": meth == "layercam" and s == chosen,
                                "class": C.NAMES[lab], "n": len(h), "n_empty_maps": int(((g.label == lab) & g.empty).sum()),
                                "region": reg, "mean_share": ms, "share_ci_lo": ls, "share_ci_hi": hs,
                                "mean_density": md, "density_ci_lo": ld, "density_ci_hi": hd})
        C.write_aggregate(pd.DataFrame(out), f"p1_regions_{split}.csv", [split])

    # --- AD vs CN, naming (chosen stage, LayerCAM) --------------------------------------------------------
    ch = sub[(sub.method == "layercam") & (sub.stage == chosen)]
    tr_ad = ch[(ch.split == "train") & (ch.label == 3)]
    tr_cn = ch[(ch.split == "train") & (ch.label == 0)]
    va_ad = ch[(ch.split == "val") & (ch.label == 3)]
    dcols = [f"d_{g}" for g in regions]
    dif, dlo, dhi = diff_ci(tr_ad[dcols].to_numpy(), tr_cn[dcols].to_numpy())
    C.write_aggregate(pd.DataFrame({"region": regions, "ad_minus_cn_density": dif, "ci_lo": dlo, "ci_hi": dhi}),
                      "p1_ad_vs_cn.csv", ["train"])
    names = []
    table = []
    for j, reg in enumerate(regions):
        m, lo, hi = C.boot_ci(tr_ad[f"d_{reg}"].to_numpy() - 1)
        val_mean = float(va_ad[f"d_{reg}"].mean())
        nameable = reg != "other" and lo > 0
        rec = {"region": reg, "train_ad_mean_density": m + 1, "ci_lo": lo + 1, "ci_hi": hi + 1,
               "nameable": bool(nameable), "val_ad_mean_density": val_mean, "replicated_on_val": val_mean > 1,
               "ad_minus_cn": float(dif[j]), "ad_minus_cn_ci": [float(dlo[j]), float(dhi[j])],
               "also_in_non_ad": bool(dlo[j] <= 0 <= dhi[j])}
        table.append(rec)
        if nameable:
            names.append(rec)
    names.sort(key=lambda x: -x["train_ad_mean_density"])
    naming = {"stage": chosen, "method": "layercam", "n_train_ad": len(tr_ad), "n_val_ad": len(va_ad),
              "rule": "P1-R3: named iff the bootstrap 95% CI of mean(d_r - 1) over train AD excludes 0 (positive)",
              "named": [{"region": x["region"], "density": x["train_ad_mean_density"],
                         "replicated_on_val": x["replicated_on_val"], "also_in_non_ad": x["also_in_non_ad"]}
                        for x in names],
              "approximate_names": info["approximate_names"], "table": table}
    C.write_json(naming, "p1_naming.json")
    C.update_json("card_rules.json", "cnn_naming", {k: naming[k] for k in ("stage", "method", "named",
                                                                            "approximate_names")})
    print(pd.DataFrame(table)[["region", "train_ad_mean_density", "ci_lo", "ci_hi", "nameable",
                               "val_ad_mean_density", "ad_minus_cn", "also_in_non_ad"]].round(3).to_string(index=False))

    # --- group figures (train, >= 104 subjects per class) ---------------------------------------------------
    tmpl = L.template96()
    os.makedirs(C.FIGURES, exist_ok=True)
    for (meth, s), fname in (((("layercam", chosen)), "p1_group_maps.png"),
                             ((("gradcam", "layer4")), "p1_group_maps_gradcam_layer4.png")):
        maps = [(C.NAMES[k], class_sum[(meth, s)][k] / class_n[k]) for k in range(4)]
        group_figure(os.path.join(C.FIGURES, fname), maps, tmpl, info["card_slices"],
                     f"Mean {meth} map ({s}, {L.STAGE_SIZE[s]}³), train subjects by true class "
                     f"(n = {', '.join(str(int(x)) for x in class_n)}); slices through the template hippocampus")
    C.verify_manifest()


if __name__ == "__main__":
    main()
