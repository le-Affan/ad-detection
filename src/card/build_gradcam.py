"""Card data, step 3 of 3 (run with LYP/.venv-mri): Grad-CAM of the ALZ CNN for every
subject with a CNN volume. Uses the existing cross-fit checkpoints only
(analysis/fusion_feasibility/runs/fold<k>_s<seed>/best.pt); nothing is trained.

Which models explain which subject (the models that produced the subject's ALZ):
  - train subjects: the 3 seed models of the subject's own cross-fit fold
    (the models that never trained on that subject);
  - val and test subjects: all 15 fold models.
Grad-CAM targets the ALZ logit at trunk.layer4 (3 x 3 x 3 at 96^3 input):
channel weights = mean gradient, map = ReLU(weighted sum), trilinear upsampling
to 96^3, each model's map scaled to max 1, then averaged over models.
The map is that coarse, so it localises to a broad region only; the card's
wording is "periventricular / ventricular attribution", not a structure.
Stored per subject: the three central orthogonal slices (axial, coronal,
sagittal at voxel 48 of 96), the scan and the map as uint8. Central slices, not
the map's peak: the coarse map often peaks in a corner cell, and fixed slices are
comparable across subjects. The averaged map's peak voxel is stored too.
Output: demo/card_data/gradcam.npz.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src", "mri"))
sys.path.insert(0, os.path.join(ROOT, "analysis", "fusion_feasibility"))
from dataset import ADNIVolumes  # noqa: E402
from predict_alz import load_model  # noqa: E402

RUNS = os.path.join(ROOT, "analysis", "fusion_feasibility", "runs")
OUT = os.path.join(ROOT, "demo", "card_data")
PROCESSED = os.path.join(ROOT, "dataset", "processed")


def gradcam(model, vols, device):
    acts = {}
    h1 = model.trunk.layer4.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    out = []
    for s in range(0, len(vols), 8):
        x = vols[s:s + 8].to(device)
        o = model(x)["alz_logit"]
        a = acts["a"]
        g, = torch.autograd.grad(o.sum(), a)
        cam = Fn.relu((g.mean((2, 3, 4), keepdim=True) * a).sum(1, keepdim=True))
        cam = Fn.interpolate(cam, size=x.shape[2:], mode="trilinear", align_corners=False)[:, 0]
        cam = cam / cam.flatten(1).max(1).values.clamp_min(1e-12)[:, None, None, None]
        out.append(cam.detach().cpu().numpy())
    h1.remove()
    return np.concatenate(out)


def u8(x):
    return np.clip(np.round(x * 255), 0, 255).astype(np.uint8)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_of = {int(r): s for s in ("train", "val", "test")
                for r in pd.read_csv(os.path.join(PROCESSED, f"{s}_rids.csv")).RID}
    folds = pd.read_csv(os.path.join(ROOT, "analysis", "fusion_feasibility", "folds", "folds.csv"))
    fold_of = dict(zip(folds.RID, folds.fold))
    ds = ADNIVolumes(split=None, strict=True)
    ds.df = ds.df[ds.df.RID.isin(split_of)].sort_values("PTID").reset_index(drop=True)
    vols = torch.stack([ds[i]["volume"] for i in range(len(ds.df))])
    sp = ds.df.RID.map(split_of).to_numpy()
    fk = ds.df.RID.map(fold_of).to_numpy()
    print(f"{len(ds.df)} volumes on {device}", flush=True)
    cam_sum = np.zeros((len(ds.df),) + tuple(vols.shape[2:]), np.float32)
    n_models = np.zeros(len(ds.df), int)
    for k in range(5):
        rows = np.flatnonzero((sp != "train") | (fk == k))
        for seed in (42, 43, 44):
            model, _ = load_model(os.path.join(RUNS, f"fold{k}_s{seed}", "best.pt"), device)
            cam_sum[rows] += gradcam(model, vols[rows], device)
            n_models[rows] += 1
            del model
        print(f"fold {k} done", flush=True)
    assert set(n_models[sp == "train"]) == {3} and set(n_models[sp != "train"]) == {15}
    cam = cam_sum / n_models[:, None, None, None]
    v = vols[:, 0].numpy()
    peak = np.array([np.unravel_index(np.argmax(c), c.shape) for c in cam])
    bg, cm = [], []
    a = b = c = v.shape[1] // 2
    for i in range(len(v)):
        # vol96 axes are (S, P, R): axial = fixed S -> (P, R); coronal = fixed P -> (S, R); sagittal = fixed R -> (S, P)
        bg.append([v[i][a, :, :], v[i][:, b, :], v[i][:, :, c]])
        cm.append([cam[i][a, :, :], cam[i][:, b, :], cam[i][:, :, c]])
    np.savez_compressed(os.path.join(OUT, "gradcam.npz"), ptid=ds.df.PTID.to_numpy(), split=sp,
                        n_models=n_models, peak=peak, bg=u8(np.array(bg)), cam=u8(np.array(cm)))
    print(f"saved {len(ds.df)} subjects; size {os.path.getsize(os.path.join(OUT, 'gradcam.npz')) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
