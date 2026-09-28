"""P1 smoke test: pretrained SFCN brain age (Peng et al., 2021) on 20 clinical-TRAIN subjects.

Never reads val or test: subjects are drawn from dataset/processed/train_rids.csv
only, and that is asserted.

Model: ha-ha-ha-han/UKBiobank_deep_pretrain @ 4e56dc1 (MIT), weights
brain_age/run_20190719_00_epoch_best_mae.p (sha256 f9197c92...). UK Biobank only,
ages 44-80; output is a 40-bin soft classification over 42-82 years, so a
prediction cannot leave [42.5, 81.5]. Pass the repo path with --sfcn-repo.

Input adaptation -- the ONLY steps applied to our m1 output (dataset/mri_prep/mni):
  1. Resample, by world (MNI mm) coordinates, from our grid (189x233x164,
     1 x 1 x 1.2 mm, SPR storage) onto the FSL MNI152 1 mm grid SFCN was
     trained on (182x218x182, affine diag(-1,1,1), origin (90,-126,-72)).
     Trilinear, zero outside the field of view.
  2. data / data.mean()   (SFCN's own normalisation, whole volume)
  3. centre crop to 160x192x160   (SFCN's own crop_center)
Our volumes are already skull-stripped (template mask), bias-corrected (N4)
and affinely registered to ICBM152 2009a, so no re-preprocessing is done.
Residual mismatches are documented in p1_report.md.
"""
import argparse
import os
import sys

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from nibabel.processing import resample_from_to

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PROCESSED = os.path.join(ROOT, "dataset", "processed")

FSL_SHAPE = (182, 218, 182)
FSL_AFFINE = np.array([[-1, 0, 0, 90], [0, 1, 0, -126], [0, 0, 1, -72], [0, 0, 0, 1]], float)
CROP = (160, 192, 160)
BIN_RANGE, BIN_STEP = (42, 82), 1
N_PER_CLASS = 10
MIN_CORR = 0.3          # excludes m1 registration failures (027_S_4966 r = 0.05)


def pick_subjects(seed):
    train = set(pd.read_csv(os.path.join(PROCESSED, "train_rids.csv")).RID)
    mi = pd.read_csv(os.path.join(ROOT, "dataset", "mri_prep", "mri_index.csv"))
    qc = pd.read_csv(os.path.join(ROOT, "dataset", "mri_prep", "m1_report_full.csv"))[["PTID", "corr_vs_mni"]]
    man = pd.read_csv(os.path.join(ROOT, "dataset", "mri_manifest", "mri_manifest.csv"))
    age = pd.read_csv(os.path.join(PROCESSED, "step2_features.csv"), usecols=["RID", "AGE"])
    df = (mi.merge(qc, on="PTID").merge(age, on="RID")
            .merge(man[["RID", "BASELINE_EXAMDATE", "MRI_EXAMDATE"]], on="RID"))
    df = df[df.RID.isin(train) & (df.split == "train") & df.label.isin([0, 3]) & (df.corr_vs_mni >= MIN_CORR)]
    # ADSL AGE is age at enrolment; shift by the baseline-visit -> MRI-date gap
    gap = (pd.to_datetime(df.MRI_EXAMDATE) - pd.to_datetime(df.BASELINE_EXAMDATE)).dt.days / 365.25
    df = df.assign(age_at_scan=df.AGE + gap, mri_minus_baseline_days=(gap * 365.25).round().astype(int))
    pick = pd.concat([g.sample(N_PER_CLASS, random_state=seed) for _, g in df.groupby("label")])
    assert pick.RID.isin(train).all()
    return pick.reset_index(drop=True)


def to_sfcn_input(path):
    img = nib.load(path)
    res = resample_from_to(img, (FSL_SHAPE, FSL_AFFINE), order=1, mode="constant", cval=0.0)
    data = np.asarray(res.dataobj, dtype=np.float32)
    fov_brain_kept = float(data.sum() / (np.asarray(img.dataobj, np.float32).sum() * 1.2))
    data = data / data.mean()
    x, y, z = [(s - c) // 2 for s, c in zip(FSL_SHAPE, CROP)]
    crop = data[x:x + CROP[0], y:y + CROP[1], z:z + CROP[2]]
    crop_brain_kept = float(crop.sum() / data.sum())
    return crop, {"fov_mass_kept": fov_brain_kept, "crop_mass_kept": crop_brain_kept}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sfcn-repo", default=os.path.join(ROOT, "..", "..", "third_party", "UKBiobank_deep_pretrain"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--qc-png", action="store_true")
    args = ap.parse_args()
    sys.path.insert(0, os.path.abspath(args.sfcn_repo))
    from dp_model.model_files.sfcn import SFCN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SFCN()
    sd = torch.load(os.path.join(args.sfcn_repo, "brain_age", "run_20190719_00_epoch_best_mae.p"),
                    map_location="cpu")
    model.load_state_dict({k.removeprefix("module."): v for k, v in sd.items()}, strict=True)
    model.to(device).eval()
    centers = BIN_RANGE[0] + BIN_STEP / 2 + BIN_STEP * np.arange((BIN_RANGE[1] - BIN_RANGE[0]) // BIN_STEP)

    subj = pick_subjects(args.seed)
    rows = []
    for r in subj.itertuples():
        x, qc = to_sfcn_input(os.path.join(ROOT, "dataset", "mri_prep", "mni", f"{r.PTID}.nii.gz"))
        with torch.no_grad():
            logp = model(torch.from_numpy(x)[None, None].to(device))[0].reshape(-1).double().cpu().numpy()
        p = np.exp(logp)
        pred = float(p @ centers)
        rows.append({"RID": r.RID, "PTID": r.PTID, "DX_bl": r.DX_bl, "age_at_scan": round(r.age_at_scan, 2),
                     "mri_minus_baseline_days": r.mri_minus_baseline_days,
                     "pred_age": round(pred, 2), "gap": round(pred - r.age_at_scan, 2),
                     "p_top_bin_81.5": round(float(p[-1]), 3), "p_ge_78": round(float(p[centers >= 78].sum()), 3),
                     "pred_sd": round(float(np.sqrt(p @ (centers - pred) ** 2)), 2),
                     "corr_vs_mni": round(r.corr_vs_mni, 3), **{k: round(v, 4) for k, v in qc.items()}})
        if args.qc_png and len(rows) in (1, N_PER_CLASS + 1):
            save_qc(x, r.PTID)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(HERE, "smoke_test.csv"), index=False)
    print(out.to_string(index=False))
    summarise(out)


def summarise(out):
    print()
    for dx, g in out.groupby("DX_bl", sort=False):
        print(f"{dx:3s} n={len(g)}  age {g.age_at_scan.mean():.1f} ({g.age_at_scan.min():.1f}-{g.age_at_scan.max():.1f})"
              f"  pred {g.pred_age.mean():.1f} ({g.pred_age.min():.1f}-{g.pred_age.max():.1f})"
              f"  MAE {g.gap.abs().mean():.2f}  mean gap {g.gap.mean():+.2f} (sd {g.gap.std():.2f})")
    cn = out[out.DX_bl == "CN"]
    y = cn[cn.age_at_scan <= 80]
    print(f"CN aged <=80 (inside SFCN's 44-80 training range): n={len(y)} MAE {y.gap.abs().mean():.2f}")
    print(f"corr(pred, age) all: {np.corrcoef(out.pred_age, out.age_at_scan)[0, 1]:.3f}   "
          f"pred SD across subjects: {out.pred_age.std():.2f}")


def save_qc(x, ptid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = [s // 2 for s in x.shape]
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for a, sl, t in zip(ax, (x[c[0]], x[:, c[1]], x[:, :, c[2]]), ("sagittal (axis0)", "coronal (axis1)", "axial (axis2)")):
        a.imshow(np.rot90(sl), cmap="gray")
        a.set_title(t)
        a.axis("off")
    fig.suptitle(f"{ptid}: SFCN input 160x192x160 (FSL MNI152 1mm grid, cropped)")
    fig.savefig(os.path.join(HERE, f"qc_{ptid}.png"), dpi=80)
    plt.close(fig)


if __name__ == "__main__":
    main()
