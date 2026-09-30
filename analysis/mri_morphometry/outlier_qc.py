"""Step 5: QC of the brain-age outliers (train CN only; never reads test).

Subjects: the two CN with corrected brain-age gaps near -25 and -23 years
(016_S_4097, 098_S_4275) and the six site-016 CN (all clinical-train).

For each subject this saves a slice panel of the m1 mni/ volume (the exact
input DeepBrainNet saw, before its own resample) with the template brain-mask
contour, plus reference CN scans of similar age with gaps near zero, one per
protocol (MPRAGE and Sag IR-SPGR). It also tabulates the checks that separate
a preprocessing failure from a plausible young-looking brain:
  - registration: m1 corr_vs_mni (the pipeline's alignment check; < 0.30 suspect)
  - intensity: fraction of in-mask voxels saturated at 1.0, in-mask median
  - anatomy, independent of our pipeline: ADNI FreeSurfer Ventricles/ICV and
    Hippocampus/ICV, as age-adjusted z-scores among clinical-train CN
    (OLS on age). A genuinely young-looking brain has small ventricles and a
    large hippocampus for its age; a pipeline artefact does not.

Run with LYP/.venv-mri (ants, nilearn, nibabel, matplotlib).
Outputs: qc_outliers/*.png and outlier_qc.csv in analysis/mri_morphometry/.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src", "mri"))
OUT = os.path.join(HERE, "qc_outliers")
MNI = os.path.join(ROOT, "dataset", "mri_prep", "mni")
ALPHA, BETA = 0.582, 29.97                     # P2 val-protocol CN bias fit (all train CN)

OUTLIERS = ["016_S_4097", "098_S_4275"]
# Slices in the 189x233x164 (S, P, R) grid: axial through the lateral
# ventricles, coronal through the hippocampal body, sagittal off-midline.
AX, COR, SAG = 105, 130, 60


def load_table():
    ba = pd.read_csv(os.path.join(ROOT, "analysis", "brain_age", "brainage_trainval.csv"))
    ba["gap"] = (ba.pred_age - BETA) / ALPHA - ba.age_at_scan
    sc = pd.read_csv(os.path.join(ROOT, "dataset", "mri_raw", "selected_scans.csv"))[["PTID", "protocol"]]
    fs = pd.read_csv(os.path.join(ROOT, "dataset", "processed", "step2_features.csv"))[
        ["RID", "Ventricles", "Hippocampus", "ICV"]]
    m1 = pd.read_csv(os.path.join(ROOT, "dataset", "mri_prep", "m1_report_full.csv"))[["PTID", "corr_vs_mni"]]
    d = ba.merge(sc, on="PTID", how="left").merge(fs, on="RID", how="left").merge(m1, on="PTID", how="left")
    cn = d[(d.label == 0) & (d.split == "train")].copy()
    for v in ("Ventricles", "Hippocampus"):
        r = cn[v] / cn.ICV
        ok = r.notna()
        b = np.polyfit(cn.age_at_scan[ok], r[ok], 1)
        res = r - np.polyval(b, cn.age_at_scan)
        cn[f"{v}_ICV_agez"] = res / res[ok].std()
    return cn


def intensity(ptid, mask):
    a = nib.load(os.path.join(MNI, f"{ptid}.nii.gz")).get_fdata(dtype="float32")
    x = a[mask]
    return a, {"sat_frac": float((x >= 0.999).mean()), "median_in_mask": float(np.median(x))}


def panel(ax_row, a, mask, title):
    views = ((a[AX], mask[AX]), (a[:, COR], mask[:, COR]), (a[:, :, SAG], mask[:, :, SAG]))
    for ax, (img, m) in zip(ax_row[:3], views):
        ax.imshow(np.rot90(img), cmap="gray", vmin=0, vmax=1)
        ax.contour(np.rot90(m), levels=[0.5], colors="r", linewidths=0.5)
        ax.axis("off")
    ax_row[0].set_title(title, fontsize=8, loc="left")


def main():
    import m1_preprocess as m1
    os.makedirs(OUT, exist_ok=True)
    _, mask_w, _, _, _ = m1.load_template(96)
    mask = mask_w.numpy() > 0.5

    cn = load_table()
    site016 = sorted(cn[cn.PTID.str.startswith("016_")].PTID)
    targets = sorted(set(OUTLIERS) | set(site016))
    rows = []
    for p in cn.PTID:                                   # intensity stats for every train CN
        if not os.path.exists(os.path.join(MNI, f"{p}.nii.gz")):
            continue
        _, st = intensity(p, mask)
        rows.append({"PTID": p, **st})
    cn = cn.merge(pd.DataFrame(rows), on="PTID", how="left")

    def refs(ptid):
        """Two reference CN: nearest age with |gap| < 2, one per protocol family."""
        age = cn.set_index("PTID").age_at_scan[ptid]
        out = []
        for fam in ("MPRAGE", "SPGR"):
            c = cn[cn.protocol.str.contains(fam) & (cn.gap.abs() < 2) & ~cn.PTID.isin(targets)]
            out.append(c.iloc[(c.age_at_scan - age).abs().argsort()].PTID.iloc[0])
        return out

    def label(p):
        r = cn.set_index("PTID").loc[p]
        return (f"{p}  age {r.age_at_scan:.1f}  gap {r.gap:+.1f}  {r.protocol}\n"
                f"corr_vs_mni {r.corr_vs_mni:.2f}  sat {100 * r.sat_frac:.1f}%  "
                f"FS vent/ICV z {r.Ventricles_ICV_agez:+.1f}  hipp/ICV z {r.Hippocampus_ICV_agez:+.1f}")

    for p in OUTLIERS:
        subs = [p] + refs(p)
        fig, axes = plt.subplots(len(subs), 4, figsize=(13, 3.3 * len(subs)))
        for row, s in zip(axes, subs):
            a, _ = intensity(s, mask)
            panel(row, a, mask, ("OUTLIER  " if s == p else "reference  ") + label(s))
            row[3].hist(a[mask], bins=100, range=(0, 1), color="0.3")
            row[3].set_yticks([])
            row[3].set_xlabel("in-mask intensity", fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"outlier_{p}.png"), dpi=90)
        plt.close(fig)

    fig, axes = plt.subplots(len(site016), 4, figsize=(13, 3.3 * len(site016)))
    for row, s in zip(axes, site016):
        a, _ = intensity(s, mask)
        panel(row, a, mask, label(s))
        row[3].hist(a[mask], bins=100, range=(0, 1), color="0.3")
        row[3].set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "site016_cn.png"), dpi=90)
    plt.close(fig)

    # anatomy vs gap for all train CN, targets highlighted
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for i, (x, xl) in enumerate((("Ventricles_ICV_agez", "FS ventricles/ICV, age-adjusted z"),
                                 ("Hippocampus_ICV_agez", "FS hippocampus/ICV, age-adjusted z"),
                                 ("sat_frac", "saturated fraction in mask"))):
        ge = cn.protocol.str.contains("SPGR")
        ax[i].scatter(cn[x][~ge], cn.gap[~ge], s=10, c="tab:blue", label="MPRAGE CN")
        ax[i].scatter(cn[x][ge], cn.gap[ge], s=10, c="tab:orange", label="IR-SPGR (GE) CN")
        t = cn[cn.PTID.isin(targets)]
        ax[i].scatter(t[x], t.gap, s=40, facecolors="none", edgecolors="k", label="016 / outliers")
        for _, r in t.iterrows():
            ax[i].annotate(r.PTID, (r[x], r.gap), fontsize=6)
        ax[i].set_xlabel(xl)
        ax[i].set_ylabel("corrected brain-age gap (y)")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cn_gap_vs_anatomy.png"), dpi=90)
    plt.close(fig)

    cols = ["PTID", "age_at_scan", "gap", "protocol", "corr_vs_mni", "sat_frac", "median_in_mask",
            "Ventricles_ICV_agez", "Hippocampus_ICV_agez"]
    cn["target"] = cn.PTID.isin(targets)
    cn[cols + ["target"]].to_csv(os.path.join(HERE, "outlier_qc.csv"), index=False)
    print(cn[cn.target][cols].round(3).to_string(index=False))
    print("\ntrain CN reference (median [IQR]):")
    for c in ("corr_vs_mni", "sat_frac", "median_in_mask"):
        q = cn[c].quantile([.25, .5, .75])
        print(f"  {c}: {q[.5]:.3f} [{q[.25]:.3f}, {q[.75]:.3f}]")
    print("\nFS age-adjusted z vs gap (Spearman, train CN):",
          {v: round(cn[[v, "gap"]].corr("spearman").iloc[0, 1], 3)
           for v in ("Ventricles_ICV_agez", "Hippocampus_ICV_agez")})


if __name__ == "__main__":
    main()
