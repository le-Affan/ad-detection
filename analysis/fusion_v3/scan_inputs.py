"""Scan-derived inputs for one split: ALZ and the brain-age reliability rule (run with LYP/.venv-mri).

--split val   reproducibility check: must reproduce val ALZ in
              kan_fusion/../fusion_feasibility/alz_crossfit.csv and the val rows of
              kan_fusion/brainage_reliability.csv.
--split test  allowed only after analysis/fusion_v3/freeze.md is committed and
              unmodified (checked against git).

ALZ = mean over the 5 cross-fit fold models of each model's 3-seed mean
sigmoid(ALZ logit). This is exactly how val ALZ was built
(fusion_feasibility/predict_alz.py), with the same checkpoints (best.pt).
Reliability: inside the ICBM152 template brain mask, the m1 mni/ volume has no
zero voxels, or more than 10% of voxels >= 0.999 -> unreliable
(kan_fusion/brainage_reliability.py).
Output: scan_inputs_<split>.csv (RID, PTID, ALZ, has_mri, zero voxels,
saturated fraction, unreliable).
"""
import argparse
import os
import sys

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from freeze_guard import assert_frozen

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src", "mri"))
sys.path.insert(0, os.path.join(ROOT, "analysis", "fusion_feasibility"))
PROCESSED = os.path.join(ROOT, "dataset", "processed")
RUNS = os.path.join(ROOT, "analysis", "fusion_feasibility", "runs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["val", "test"])
    a = ap.parse_args()
    if a.split == "test":
        print("freeze commit:", assert_frozen())
    from dataset import ADNIVolumes
    from predict_alz import load_model, score
    import m1_preprocess as m1

    rids = set(pd.read_csv(os.path.join(PROCESSED, f"{a.split}_rids.csv")).RID)
    ds = ADNIVolumes(split=None, strict=True)
    ds.df = ds.df[ds.df.RID.isin(rids)].reset_index(drop=True)
    print(f"{a.split}: {len(rids)} subjects, {len(ds.df)} with a scan")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vols = torch.stack([ds[i]["volume"] for i in range(len(ds.df))])
    fold_means = []
    for k in range(5):
        seeds = []
        for seed in (42, 43, 44):
            model, _ = load_model(os.path.join(RUNS, f"fold{k}_s{seed}", "best.pt"), device)
            seeds.append(score(model, vols, device)[1])
            del model
        fold_means.append(np.mean(seeds, 0))
    alz = np.mean(fold_means, 0)

    _, mask_w, _, _, _ = m1.load_template(96)
    mask = mask_w.numpy() > 0.5
    rows = []
    for p in ds.df.PTID:
        x = nib.load(os.path.join(ROOT, "dataset", "mri_prep", "mni", f"{p}.nii.gz")).get_fdata(dtype="float32")[mask]
        rows.append({"PTID": p, "n_zero": int((x == 0).sum()), "sat_frac": float((x >= 0.999).mean())})
    out = pd.DataFrame({"RID": ds.df.RID, "PTID": ds.df.PTID, "ALZ": alz}).merge(pd.DataFrame(rows), on="PTID")
    out["unreliable"] = (out.n_zero == 0) | (out.sat_frac > 0.10)
    allr = pd.DataFrame({"RID": sorted(rids)})
    out = allr.merge(out, on="RID", how="left")
    out["has_mri"] = out.ALZ.notna()
    out.to_csv(os.path.join(HERE, f"scan_inputs_{a.split}.csv"), index=False)
    print(f"ALZ for {int(out.has_mri.sum())}; unreliable brain-age scans: {int(out.unreliable.fillna(False).sum())}")

    if a.split == "val":
        cf = pd.read_csv(os.path.join(ROOT, "analysis", "fusion_feasibility", "alz_crossfit.csv"))
        cf = cf[cf.clin_split == "val"].set_index("RID").ALZ
        d = (out.set_index("RID").ALZ - cf).abs()
        rel = pd.read_csv(os.path.join(ROOT, "analysis", "kan_fusion", "brainage_reliability.csv")).set_index("RID")
        agree = (out.set_index("RID").unreliable.dropna() == rel.unreliable.reindex(out.dropna().RID)).all()
        print(f"val check: max |ALZ - alz_crossfit| = {d.max():.2e}; reliability agrees: {agree}")


if __name__ == "__main__":
    main()
