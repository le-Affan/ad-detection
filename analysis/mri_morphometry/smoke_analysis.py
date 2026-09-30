"""Step 4 smoke test: summarise SynthSeg on the 20 clinical-train subjects (+2 outliers).

Reads dataset/mri_prep/synthseg/{vol,qc}_smoke_*.csv (from run_synthseg.py) and
the ADNI FreeSurfer values (dataset/processed/step2_features.csv, raw units).
Train only. Run with LYP/.venv-synthseg.

Outputs (analysis/mri_morphometry/): smoke_volumes.csv, smoke_summary.md,
qc_smoke/overlay_<PTID>.png.

Structure matching (SynthSeg -> ADNI FreeSurfer column):
  Hippocampus  L + R hippocampus
  Ventricles   L + R lateral + inferior lateral ventricle
  Entorhinal   ctx-lh/rh-entorhinal
  Fusiform     ctx-lh/rh-fusiform
  MidTemp      ctx-lh/rh-middletemporal
  WholeBrain   every brain label except csf and the four ventricle labels
  ICV          SynthSeg "total intracranial"
Our volumes are in affine-normalised ICBM152 space, so they are roughly
head-size normalised; FreeSurfer's are native mm^3. Agreement is therefore
reported both against raw FreeSurfer values and against FreeSurfer / ICV.
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from nibabel.processing import resample_from_to  # noqa: E402
from scipy.stats import mannwhitneyu, pearsonr, spearmanr  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
WORK = os.path.join(ROOT, "dataset", "mri_prep", "synthseg")
OUT = os.path.join(HERE, "qc_smoke")

VENT = ["left lateral ventricle", "right lateral ventricle",
        "left inferior lateral ventricle", "right inferior lateral ventricle"]
NON_BRAIN = VENT + ["3rd ventricle", "4th ventricle", "csf", "total intracranial"]


def derive(v):
    d = pd.DataFrame({"PTID": v.subject})
    d["ss_Hippocampus"] = v["left hippocampus"] + v["right hippocampus"]
    d["ss_Ventricles"] = v[VENT].sum(1)
    for fs, ss in (("Entorhinal", "entorhinal"), ("Fusiform", "fusiform"), ("MidTemp", "middletemporal")):
        d[f"ss_{fs}"] = v[f"ctx-lh-{ss}"] + v[f"ctx-rh-{ss}"]
    d["ss_WholeBrain"] = v.drop(columns=["subject"] + NON_BRAIN).sum(1)
    d["ss_ICV"] = v["total intracranial"]
    d["ss_InfLatVent"] = v[VENT[2:]].sum(1)
    d["ss_Amygdala"] = v["left amygdala"] + v["right amygdala"]
    return d


def overlay(ptid, title):
    seg = nib.load(os.path.join(WORK, "seg", f"{ptid}_synthseg.nii.gz"))
    img = resample_from_to(nib.load(os.path.join(WORK, "input", f"{ptid}.nii.gz")), seg, order=1)
    s, a = np.asarray(seg.dataobj).astype(int), img.get_fdata()
    hip = np.isin(s, (17, 53))
    amy = np.isin(s, (18, 54))
    lat = np.isin(s, (4, 43))
    inf = np.isin(s, (5, 44))
    # grid is (S, P, R) as in mni/: axis 1 is coronal, axis 0 axial
    cor = int(np.argmax(hip.sum((0, 2))))
    axi = int(np.argmax(lat.sum((1, 2))))
    sag = int(np.argmax(hip[:, :, : s.shape[2] // 2].sum((0, 1))))
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.6))
    for i, (sl, name) in enumerate(((np.s_[:, cor, :], f"coronal {cor}"), (np.s_[axi, :, :], f"axial {axi}"),
                                    (np.s_[:, :, sag], f"sagittal {sag} (left)"))):
        ax[i].imshow(np.rot90(a[sl]), cmap="gray", vmin=0, vmax=1)
        rgba = np.zeros(a[sl].shape + (4,))
        for m, c in ((hip, (1, .8, 0)), (amy, (0, 1, 1)), (lat, (1, 0, 0)), (inf, (1, 0, 1))):
            rgba[m[sl]] = (*c, .55)
        ax[i].imshow(np.rot90(rgba))
        ax[i].set_title(name, fontsize=8)
        ax[i].axis("off")
    fig.suptitle(f"{title}   yellow hippocampus, cyan amygdala, red lateral ventricles, magenta inferior lateral",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"overlay_{ptid}.png"), dpi=85)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    subs = pd.read_csv(os.path.join(HERE, "smoke_subjects.csv"))
    v = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(os.path.join(WORK, "vol_smoke_*.csv"))))
    qc = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(os.path.join(WORK, "qc_smoke_*.csv"))))
    qc = qc.rename(columns={"subject": "PTID"})
    assert v.subject.is_unique and qc.PTID.is_unique
    fs = pd.read_csv(os.path.join(ROOT, "dataset", "processed", "step2_features.csv"))
    fsc = ["Hippocampus", "Ventricles", "Entorhinal", "Fusiform", "MidTemp", "WholeBrain", "ICV"]
    d = subs.merge(derive(v), on="PTID").merge(qc, on="PTID").merge(
        fs[["RID"] + fsc].rename(columns={c: f"fs_{c}" for c in fsc}), on="RID")
    d.to_csv(os.path.join(HERE, "smoke_volumes.csv"), index=False)
    sm = d[d.set == "smoke"]

    L = ["# SynthSeg smoke test — summary tables\n",
         f"{len(sm)} clinical-train subjects ({(sm.DX_bl == 'CN').sum()} CN, {(sm.DX_bl == 'AD').sum()} AD) "
         f"+ {len(d) - len(sm)} brain-age outliers (CN, reported separately). "
         "Volumes in mm³ of affine-normalised ICBM152 space.\n",
         "## QC scores (SynthSeg 2.0; the tool's docs treat < 0.65 as a likely failure)\n"]
    qcc = [c for c in qc.columns if c != "PTID"]
    L.append("| Structure group | min | median | max | n < 0.65 |")
    L.append("|---|---|---|---|---|")
    for c in qcc:
        L.append(f"| {c} | {d[c].min():.3f} | {d[c].median():.3f} | {d[c].max():.3f} | {(d[c] < .65).sum()} |")
    L.append(f"\nLowest subject minimum: {d.set_index('PTID')[qcc].min(1).idxmin()} "
             f"({d[qcc].min(1).min():.3f}).\n")

    L.append(f"## Agreement with ADNI FreeSurfer (n = {len(sm)} smoke subjects)\n")
    L.append("| Structure | Spearman ρ (raw FS) | Pearson r (raw FS) | Spearman ρ (FS / FS ICV) | "
             "SynthSeg mean | FS mean |")
    L.append("|---|---|---|---|---|---|")
    for c in fsc:
        x, y = sm[f"ss_{c}"], sm[f"fs_{c}"]
        ok = x.notna() & y.notna()
        rn = spearmanr(x[ok], (y / sm.fs_ICV)[ok]).statistic if c != "ICV" else float("nan")
        L.append(f"| {c} | {spearmanr(x[ok], y[ok]).statistic:+.2f} | {pearsonr(x[ok], y[ok])[0]:+.2f} | "
                 f"{rn:+.2f} | {x.mean():,.0f} | {y.mean():,.0f} |")
    L.append(f"\nSynthSeg ICV coefficient of variation {sm.ss_ICV.std() / sm.ss_ICV.mean():.3f} "
             f"vs FreeSurfer ICV {sm.fs_ICV.std() / sm.fs_ICV.mean():.3f}.\n")

    L.append("## CN vs AD (smoke subjects)\n")
    L.append("| Volume | CN mean ± SD | AD mean ± SD | AD / CN | Mann–Whitney p |")
    L.append("|---|---|---|---|---|")
    cn, ad = sm[sm.DX_bl == "CN"], sm[sm.DX_bl == "AD"]
    for c in ("ss_Hippocampus", "ss_Amygdala", "ss_InfLatVent", "ss_Ventricles", "ss_Entorhinal",
              "fs_Hippocampus"):
        L.append(f"| {c} | {cn[c].mean():,.0f} ± {cn[c].std():,.0f} | {ad[c].mean():,.0f} ± {ad[c].std():,.0f} | "
                 f"{ad[c].mean() / cn[c].mean():.2f} | {mannwhitneyu(cn[c], ad[c]).pvalue:.1e} |")
    L.append("")

    ol = d[d.set == "outlier"]
    L.append("## Brain-age outliers\n")
    L.append("| PTID | Hippocampus | Ventricles | InfLatVent | min QC |")
    L.append("|---|---|---|---|---|")
    for _, r in ol.iterrows():
        L.append(f"| {r.PTID} | {r.ss_Hippocampus:,.0f} | {r.ss_Ventricles:,.0f} | {r.ss_InfLatVent:,.0f} | "
                 f"{r[qcc].min():.3f} |")
    L.append(f"\nCN smoke range: hippocampus {cn.ss_Hippocampus.min():,.0f}–{cn.ss_Hippocampus.max():,.0f}, "
             f"ventricles {cn.ss_Ventricles.min():,.0f}–{cn.ss_Ventricles.max():,.0f}.\n")
    with open(os.path.join(HERE, "smoke_summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))

    # overlays: youngest-looking and median CN, median and most atrophied AD, and the two outliers
    picks = [cn.sort_values("ss_Hippocampus").PTID.iloc[-1], cn.sort_values("ss_Hippocampus").PTID.iloc[4],
             ad.sort_values("ss_Hippocampus").PTID.iloc[5], ad.sort_values("ss_Hippocampus").PTID.iloc[0]]
    for p in picks + list(ol.PTID):
        r = d.set_index("PTID").loc[p]
        overlay(p, f"{p} {r.DX_bl} ({r.set})  hippocampus {r.ss_Hippocampus:,.0f} mm³")
    print("overlays:", picks + list(ol.PTID))


if __name__ == "__main__":
    main()
