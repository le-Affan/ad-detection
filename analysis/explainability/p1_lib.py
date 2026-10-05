"""P1 shared code (LYP/.venv-mri): volumes, the frozen ALZ CNNs, LayerCAM / Grad-CAM, occlusion, noise masks.

Frozen models are loaded with analysis/fusion_feasibility/predict_alz.load_model (CPU load, eval mode) and are
never written. Which models explain a subject: train -> the 3 seed models of its own cross-fit fold (out-of-fold);
val and test -> all 15. Maps per model: hooked stage output A, target = ALZ logit,
  LayerCAM  M = ReLU(sum_c ReLU(dy/dA_c) * A_c)
  Grad-CAM  M = ReLU(sum_c mean(dy/dA_c) * A_c)
cached at native resolution (float16). upsample(): trilinear to 96^3 (align_corners=False), masked to the
template brain, unit positive mass.
"""
import os
import sys

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

sys.path.insert(0, os.path.join(C.ROOT, "src", "mri"))
sys.path.insert(0, os.path.join(C.ROOT, "analysis", "fusion_feasibility"))
from predict_alz import load_model  # noqa: E402

RUNS = os.path.join(C.ROOT, "analysis", "fusion_feasibility", "runs")
STAGES = ["layer1", "layer2", "layer4"]
STAGE_SIZE = {"layer1": 24, "layer2": 12, "layer4": 3}
SIGMA = {s: 96 / n for s, n in STAGE_SIZE.items()}                  # noise smoothing, voxels: 4, 8, 32
METHODS = ["layercam", "gradcam"]
MODELS = [(k, s) for k in range(5) for s in (42, 43, 44)]          # model id = index in this list
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def model_path(mid):
    k, s = MODELS[mid]
    return os.path.join(RUNS, f"fold{k}_s{s}", "best.pt")


def get_model(mid):
    m, _ = load_model(model_path(mid), DEVICE)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


# --- subjects -------------------------------------------------------------------------------
def subjects(include_test=False):
    """One row per subject with a CNN volume: PTID, RID, split, label, fold (train only), path."""
    idx = pd.read_csv(os.path.join(C.ROOT, "dataset", "mri_prep", "mri_index.csv"))
    sp = C.split_of()
    idx["split"] = idx.RID.map(sp)
    keep = {"train", "val"} | ({"test"} if include_test else set())
    d = idx[idx.split.isin(keep)].copy()
    fo = C.fold_of()
    d["fold"] = d.RID.map(fo)
    d["path"] = [os.path.join(C.ROOT, *str(p).replace("\\", "/").split("/")) for p in d.vol_path]
    d = d.sort_values("PTID").reset_index(drop=True)
    assert d[d.split == "train"].fold.notna().all()
    if not include_test:
        C.assert_no_test(d.RID)
    return d[["PTID", "RID", "split", "label", "fold", "path"]]


def models_of(row):
    if row.split == "train":
        return [i for i, (k, _) in enumerate(MODELS) if k == int(row.fold)]
    return list(range(len(MODELS)))


def load_volumes(paths):
    out = np.empty((len(paths), 96, 96, 96), np.float32)
    for i, p in enumerate(paths):
        v = np.asarray(nib.load(p).dataobj, dtype=np.float32)
        assert v.shape == (96, 96, 96) and v.min() >= -1e-4 and v.max() <= 1 + 1e-4, p
        out[i] = v
    return out


def brain_mask():
    a = nib.load(os.path.join(C.ATLAS, "atlas96.nii.gz")).get_fdata()
    return a > 0                                                     # atlas is restricted to the template mask


def atlas():
    import json
    a = np.asarray(nib.load(os.path.join(C.ATLAS, "atlas96.nii.gz")).dataobj).astype(np.int16)
    info = json.load(open(os.path.join(C.ATLAS, "regions.json")))
    return a, info


def template96():
    """The masked ICBM152 template on the 96^3 grid (written by atlas/atlas_build.py; regenerable)."""
    return np.asarray(nib.load(os.path.join(C.ATLAS, "work", "template96.nii.gz")).dataobj, dtype=np.float32)


def cn_train_mean():
    """Voxel-wise mean of the CN-train scans (occlusion baseline); cached (gitignored)."""
    path = C.cache_path("cn_train_mean.npy")
    if os.path.exists(path):
        return np.load(path)
    d = subjects()
    d = d[(d.split == "train") & (d.label == 0)]
    assert len(d) == 133, len(d)
    m = load_volumes(list(d.path)).mean(0)
    np.save(path, m)
    return m


# --- attribution ------------------------------------------------------------------------------
def cams(model, vols, stages=STAGES, batch=8):
    """vols: (n, 96, 96, 96) float32 numpy. Returns ({(method, stage): (n, s, s, s) float32}, alz_logit (n,))."""
    acts = {}
    hooks = [getattr(model.trunk, s).register_forward_hook(lambda m, i, o, s=s: acts.__setitem__(s, o))
             for s in stages]
    out = {(m, s): [] for m in METHODS for s in stages}
    logits = []
    try:
        for b in range(0, len(vols), batch):
            x = torch.from_numpy(vols[b:b + batch]).unsqueeze(1).to(DEVICE).requires_grad_(True)
            y = model(x)["alz_logit"]
            A = [acts[s] for s in stages]
            G = torch.autograd.grad(y.sum(), A)
            for s, a, g in zip(stages, A, G):
                lc = Fn.relu((Fn.relu(g) * a).sum(1))
                gc = Fn.relu((g.mean((2, 3, 4), keepdim=True) * a).sum(1))
                out[("layercam", s)].append(lc.detach().float().cpu().numpy())
                out[("gradcam", s)].append(gc.detach().float().cpu().numpy())
            logits.append(y.detach().double().cpu().numpy())
    finally:
        for h in hooks:
            h.remove()
    return {k: np.concatenate(v) for k, v in out.items()}, np.concatenate(logits)


def upsample(maps, mask):
    """(n, s, s, s) native maps -> (n, 96, 96, 96) on the brain, unit positive mass (all-zero maps stay zero)."""
    t = torch.from_numpy(np.ascontiguousarray(maps, dtype=np.float32)).unsqueeze(1)
    u = Fn.interpolate(t, size=(96, 96, 96), mode="trilinear", align_corners=False)[:, 0].numpy()
    u = np.clip(u, 0, None) * mask
    s = u.reshape(len(u), -1).sum(1)
    return u / np.where(s > 0, s, 1)[:, None, None, None]


@torch.no_grad()
def alz_logit(model, vols, batch=16):
    out = []
    for b in range(0, len(vols), batch):
        x = torch.from_numpy(np.ascontiguousarray(vols[b:b + batch])).unsqueeze(1).to(DEVICE)
        out.append(model(x)["alz_logit"].double().cpu().numpy())
    return np.concatenate(out)


def top_fraction_mask(score, mask, frac):
    """Boolean mask of the top `frac` of brain voxels by score; ties broken by (flat) voxel index."""
    flat = np.flatnonzero(mask.ravel())
    k = int(round(frac * len(flat)))
    order = flat[np.lexsort((flat, -score.ravel()[flat]))]      # descending score, then ascending index
    m = np.zeros(mask.size, bool)
    m[order[:k]] = True
    return m.reshape(mask.shape)


def brain_order(score, mask):
    """Brain voxels (flat indices) in descending score, ties by index."""
    flat = np.flatnonzero(mask.ravel())
    return flat[np.lexsort((flat, -score.ravel()[flat]))]


def noise_field(seed_key, sigma):
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(np.random.SeedSequence(list(seed_key)))
    return gaussian_filter(rng.standard_normal((96, 96, 96)).astype(np.float64), sigma)


def occlude(vol, m, base):
    v = vol.copy()
    v[m] = base[m]
    return v
