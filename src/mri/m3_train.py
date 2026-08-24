"""Training loop for the dual-head 3D ResNet-18 (m3).

Consumes `model.py` (trunk + DEMO/ALZ heads + `dual_head_loss`) and
`dataset.py` (ADNIVolumes). The only dataset change was making volume paths
resolvable away from this repo, since the index records native paths and this
trains on Linux.

RUNS IN TWO PLACES, SAME CODE. Locally the dataset resolves through the repo;
on Kaggle pass `--data-root /kaggle/input/adni-mri-prep`, which rebuilds each
path from PTID. Nothing here hardcodes a subject list or count, matching the
rest of the MRI stream (and deliberately unlike preprocessing.py, which pins
expected shapes to fail fast on a fixed cohort).

THE DEFAULTS NOW FIGHT OVERFITTING, BECAUSE THE BASELINE MEASURED IT.
The first run (resnet18, no augmentation, dropout 0, wd 1e-4) memorised the
train split outright: train loss 0.014 against val loss 9.34, best checkpoint
at epoch 9 of 24, and AD F1 exactly 0.000 at that checkpoint. Three defaults
changed in response -- a smaller backbone (`--arch resnet10`), dropout 0.3, and
train-only augmentation -- plus weight decay raised from 1e-4 to AdamW's own
reference 1e-2.

THE BASELINE REMAINS EXACTLY REPRODUCIBLE:

    --arch resnet18 --dropout 0.0 --weight-decay 1e-4 --no-aug

`DualHeadNet()` with no arguments is still the original 33,161,026-parameter
model, so nothing downstream of model.py changed meaning.

THE TWO LOSSES ARE NOT ON A COMMON SCALE, AND THE LAMBDAS CORRECT FOR IT.
The objective is `lam_demo * MSE(DEMO) + lam_alz * BCE(ALZ)`. Left at 1.0/1.0
these terms are not comparable: DEMO targets are {0.0, 0.5, 1.0, 2.0} so an
untrained model sits at MSE ~1.24, while ALZ starts at BCE ~0.693 = ln(2).

Both lambdas therefore default to `auto`: at startup, the per-task gradient
norms are MEASURED on the initialised model and the lambdas are set so each
task contributes equally, normalised so they average 1.0 (keeping total loss
magnitude comparable to the old 1.0/1.0 default).

  - Norms are taken over the SHARED TRUNK only. That is where the tasks
    actually compete for capacity; head gradients are task-private and cannot
    conflict, so including them would dilute the measurement.
  - Measured in fp32 with autocast and GradScaler off. The scaler multiplies
    gradients by ~65536, which would corrupt a norm comparison.
  - Measured AFTER the ALZ pos_weight is fixed, since that weight changes the
    ALZ gradient magnitude.

WHY FIXED LAMBDAS AND NOT RUNNING NORMALISATION. Dividing each term by its own
running average keeps the balance correct throughout training, not just at
initialisation, which is strictly better for optimisation. It was rejected
anyway: it makes the objective non-stationary, so `val_loss` at epoch 40 is not
comparable to `val_loss` at epoch 5. Early stopping and best-checkpoint
selection both compare val loss across epochs, so a moving normaliser would
quietly corrupt the two mechanisms this script relies on to decide which
weights to keep. Fixed lambdas buy a comparable, reproducible objective at the
cost of the balance drifting as MSE falls faster than BCE. That drift is the
known limitation; it is recorded in train_config.json along with the measured
norms so it can be revisited.

ALZ IS IMBALANCED and `--alz-pos-weight` defaults to `auto`, which computes
neg/pos FROM THE TRAIN SPLIT AT RUNTIME (currently 463/104 = 4.4519) and feeds
it to BCEWithLogitsLoss. It is computed rather than hardcoded on purpose: the
MRI stream holds no subject counts anywhere, which is what let it scale from 5
subjects to 811 without a code change, and a baked-in 4.4519 would go stale
the moment the cohort moves.

FOUR-CLASS METRICS COME FROM THE DEMO HEAD. Neither head emits a 4-class
prediction, so DEMO is snapped to the nearest of its four target values. The
mapping is imported from dataset.py rather than restated, so changing the
spacing there changes it here too. This is a reporting convenience for
watching CN/EMCI/LMCI/AD separately -- the KAN downstream consumes the raw
DEMO/ALZ outputs, not these bins.

CHECKPOINTS ASSUME INTERRUPTION. Kaggle sessions get killed. `last.pt` is
written every epoch, `best.pt` on every val-loss improvement, and
`epoch_XXX.pt` every `--ckpt-every`. Each carries optimizer, scheduler, AMP
scaler, RNG and early-stopping state, so `--resume` continues rather than
restarts. The epoch CSV is flushed every epoch for the same reason: a run
killed at epoch 40 keeps 40 rows.
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import ADNIVolumes, DEMO_BY_LABEL  # noqa: E402
from model import ARCHS, DualHeadNet, dual_head_loss  # noqa: E402

CLASSES = ["CN", "EMCI", "LMCI", "AD"]
# nearest-value targets for snapping DEMO back to a class; single-sourced
DEMO_VALUES = np.array([DEMO_BY_LABEL[i] for i in range(4)], dtype=np.float32)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def demo_to_label(demo):
    """Snap continuous DEMO predictions to the nearest class target."""
    return np.abs(np.asarray(demo)[:, None] - DEMO_VALUES[None, :]).argmin(1)


def _rotation_matrices(angles):
    """(B, 3) radians about x/y/z -> (B, 3, 3) composed rotation."""
    ax, ay, az = angles[:, 0], angles[:, 1], angles[:, 2]
    z, o = torch.zeros_like(ax), torch.ones_like(ax)
    ca, sa = torch.cos(ax), torch.sin(ax)
    cb, sb = torch.cos(ay), torch.sin(ay)
    cg, sg = torch.cos(az), torch.sin(az)
    rx = torch.stack([o, z, z, z, ca, -sa, z, sa, ca], 1).view(-1, 3, 3)
    ry = torch.stack([cb, z, sb, z, o, z, -sb, z, cb], 1).view(-1, 3, 3)
    rz = torch.stack([cg, -sg, z, sg, cg, z, z, z, o], 1).view(-1, 3, 3)
    return rz @ ry @ rx


def augment_batch(vol, rot_deg=0.0, trans_vox=0.0, brightness=0.0, contrast=0.0,
                  flip_lr=False):
    """Random affine + intensity jitter, per sample, on whatever device vol is on.

    TRAIN SPLIT ONLY. This is called from run_epoch under `if train`, so val and
    test cannot receive it by construction rather than by convention.

    WHAT IS DELIBERATELY ABSENT, and why -- both would corrupt the label:

      LEFT-RIGHT FLIP is opt-in (`flip_lr`) and off by default. Medial-temporal
      atrophy in AD is asymmetric, typically left-lateralised in amnestic
      presentations, so hemisphere asymmetry is itself diagnostic. Mirroring
      teaches the model to be invariant to exactly that signal. The volumes are
      MNI-registered so a flip looks anatomically plausible, which is precisely
      what makes it a tempting mistake.

      SCALING is not offered at all. Brain volume IS the atrophy signal -- it is
      why WholeBrain/Hippocampus/ICV are features in the first place -- so scale
      jitter would perturb the thing being measured.

    Rotation and translation are kept small because these volumes are already
    affine-registered to ICBM152. Augmentation is meant to model residual
    registration error, not to undo the registration.

    Background is exactly 0 by construction (m1 min-max normalises after the
    clip fix), so grid_sample's zero padding is anatomically consistent -- newly
    exposed corners look like background, not like a border artefact.
    """
    b = vol.size(0)
    dev = vol.device

    if flip_lr:
        m = torch.rand(b, device=dev) < 0.5
        if m.any():
            vol = vol.clone()
            vol[m] = torch.flip(vol[m], dims=[2])   # (B,C,X,Y,Z): X is L-R

    if rot_deg > 0 or trans_vox > 0:
        ang = (torch.rand(b, 3, device=dev) * 2 - 1) * math.radians(rot_deg)
        rot = _rotation_matrices(ang)
        # affine_grid works in normalised [-1, 1] coords, so a voxel offset is
        # 2 * v / extent along each axis
        extent = torch.tensor(vol.shape[2:], device=dev, dtype=vol.dtype)
        tr = (torch.rand(b, 3, device=dev) * 2 - 1) * trans_vox
        tr = 2.0 * tr / extent
        theta = torch.cat([rot.to(vol.dtype), tr.unsqueeze(-1)], dim=2)
        grid = torch.nn.functional.affine_grid(theta, list(vol.shape),
                                               align_corners=False)
        vol = torch.nn.functional.grid_sample(vol, grid, mode="bilinear",
                                              padding_mode="zeros",
                                              align_corners=False)

    if contrast > 0:
        c = 1 + (torch.rand(b, 1, 1, 1, 1, device=dev) * 2 - 1) * contrast
        mean = vol.mean(dim=(1, 2, 3, 4), keepdim=True)
        vol = (vol - mean) * c + mean
    if brightness > 0:
        vol = vol + (torch.rand(b, 1, 1, 1, 1, device=dev) * 2 - 1) * brightness
    if contrast > 0 or brightness > 0:
        vol = vol.clamp_(0.0, 1.0)   # dataset guarantees [0,1]; keep it true
    return vol


def compute_metrics(demo_pred, alz_logit, labels):
    """Per-class and ALZ metrics. Pure numpy/sklearn, no torch."""
    from sklearn.metrics import (balanced_accuracy_score,
                                 precision_recall_fscore_support, roc_auc_score)
    labels = np.asarray(labels)
    pred4 = demo_to_label(demo_pred)
    alz_prob = 1.0 / (1.0 + np.exp(-np.asarray(alz_logit, dtype=np.float64)))
    alz_true = (labels == 3).astype(int)
    alz_pred = (alz_prob >= 0.5).astype(int)

    p, r, f, s = precision_recall_fscore_support(
        labels, pred4, labels=[0, 1, 2, 3], average=None, zero_division=0)

    m = {
        "demo_mae": float(np.mean(np.abs(np.asarray(demo_pred) - DEMO_VALUES[labels]))),
        "acc4": float((pred4 == labels).mean()),
        "macro_f1": float(f.mean()),
        "bal_acc": float(balanced_accuracy_score(labels, pred4)),
        "alz_acc": float((alz_pred == alz_true).mean()),
    }
    # AUC is undefined if a split ever holds one ALZ class only
    m["alz_auc"] = (float(roc_auc_score(alz_true, alz_prob))
                    if len(np.unique(alz_true)) == 2 else float("nan"))
    tp = int(((alz_pred == 1) & (alz_true == 1)).sum())
    tn = int(((alz_pred == 0) & (alz_true == 0)).sum())
    m["alz_sens"] = tp / max(int(alz_true.sum()), 1)
    m["alz_spec"] = tn / max(int((alz_true == 0).sum()), 1)
    for i, c in enumerate(CLASSES):
        m["prec_" + c] = float(p[i])
        m["rec_" + c] = float(r[i])
        m["f1_" + c] = float(f[i])
        m["sup_" + c] = int(s[i])
    return m


def run_epoch(model, loader, device, lam_demo, lam_alz, pos_weight,
              optimizer=None, scaler=None, amp=False, aug=None):
    """One pass. Training when `optimizer` is given, evaluation otherwise.

    `aug` is a kwargs dict for augment_batch and is applied only when training,
    so val/test are structurally incapable of receiving augmentation.
    """
    train = optimizer is not None
    model.train(train)
    tot = tot_mse = tot_bce = 0.0
    n = 0
    demo_all, alz_all, lab_all = [], [], []
    pw = None if pos_weight is None else torch.as_tensor(float(pos_weight), device=device)

    for batch in loader:
        vol = batch["volume"].to(device, non_blocking=True)
        tgt = {"demo": batch["demo"].to(device, non_blocking=True),
               "alz": batch["alz"].to(device, non_blocking=True)}
        bs = vol.size(0)
        if train and aug:
            # fp32, outside autocast: grid_sample and the intensity ops are
            # cheap here and stay exact
            with torch.no_grad():
                vol = augment_batch(vol, **aug)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=amp):
                out = model(vol)
                if pw is None:
                    loss, parts = dual_head_loss(out, tgt, lam_demo, lam_alz)
                else:
                    # same objective, but BCE carries a positive-class weight
                    mse = torch.nn.functional.mse_loss(out["demo"], tgt["demo"])
                    bce = torch.nn.functional.binary_cross_entropy_with_logits(
                        out["alz_logit"], tgt["alz"], pos_weight=pw)
                    loss = lam_demo * mse + lam_alz * bce
                    parts = {"mse": float(mse.detach()), "bce": float(bce.detach())}

        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        tot += float(loss.detach()) * bs
        tot_mse += parts["mse"] * bs
        tot_bce += parts["bce"] * bs
        n += bs
        demo_all.append(out["demo"].detach().float().cpu().numpy())
        alz_all.append(out["alz_logit"].detach().float().cpu().numpy())
        lab_all.append(batch["label"].numpy())

    metrics = compute_metrics(np.concatenate(demo_all), np.concatenate(alz_all),
                              np.concatenate(lab_all))
    metrics["loss"] = tot / n
    metrics["mse"] = tot_mse / n
    metrics["bce"] = tot_bce / n
    return metrics


def measure_grad_balance(model, loader, device, pos_weight, n_batches=8):
    """Per-task gradient norms over the shared trunk, on the initialised model.

    Returns (g_demo, g_alz): the L2 norm of the trunk gradient of each task's
    MEAN loss across `n_batches`. Run in fp32 with no autocast and no
    GradScaler -- see the module docstring for why both would corrupt this.

    NORM OF THE MEAN GRADIENT, NOT THE MEAN OF PER-BATCH NORMS. The first
    implementation averaged per-batch norms and was measurably unstable: the
    estimate rose monotonically with n_batches (18.5 -> 33.9 -> 46.2 at 2/4/8
    batches) instead of converging, because per-batch norms carry the noise of
    a 4-subject sample and norms are not linear, so averaging them estimates
    E||g_batch|| rather than ||E g||. Training descends the mean loss, so the
    mean gradient is the quantity the lambdas should balance. Accumulating
    scaled backward passes and taking one norm at the end converges properly.

    BN BUFFERS ARE RESTORED afterwards. These extra forward passes advance
    BatchNorm running stats. That does not change train-mode forwards (which
    normalise by batch statistics), but it would make the model handed to
    training depend on --balance-batches, which should never influence results.
    """
    import copy
    model.train()
    bn_backup = copy.deepcopy({k: v for k, v in model.state_dict().items()
                               if "running_" in k or "num_batches_tracked" in k})
    pw = None if pos_weight is None else torch.as_tensor(float(pos_weight), device=device)

    batches = []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        batches.append((batch["volume"].to(device),
                        batch["demo"].to(device), batch["alz"].to(device)))
    if not batches:
        return 1.0, 1.0

    norms = {}
    for task in ("demo", "alz"):
        model.zero_grad(set_to_none=True)
        for vol, d_t, a_t in batches:
            out = model(vol)
            if task == "demo":
                loss = torch.nn.functional.mse_loss(out["demo"], d_t)
            else:
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    out["alz_logit"], a_t, pos_weight=pw)
            (loss / len(batches)).backward()   # accumulate the mean gradient
        sq = sum(float((p.grad.detach().float() ** 2).sum())
                 for p in model.trunk.parameters() if p.grad is not None)
        norms[task] = sq ** 0.5

    model.zero_grad(set_to_none=True)
    model.load_state_dict(bn_backup, strict=False)
    return norms["demo"], norms["alz"]


def balanced_lambdas(g_demo, g_alz):
    """Lambdas equalising task gradient contribution, normalised to mean 1.0.

    lam_i proportional to 1/g_i makes lam_demo*g_demo == lam_alz*g_alz; the
    scale factor pins (lam_demo + lam_alz) / 2 == 1 so the total loss stays in
    the same range as the old 1.0/1.0 default and the printed numbers remain
    readable against it.
    """
    eps = 1e-12
    inv_d, inv_a = 1.0 / max(g_demo, eps), 1.0 / max(g_alz, eps)
    s = 2.0 / (inv_d + inv_a)
    return s * inv_d, s * inv_a


def save_ckpt(path, model, opt, sched, scaler, epoch, best, bad, args, full=True):
    """Write a checkpoint.

    SIZE MATTERS HERE. A full checkpoint is ~398 MB: 133 MB of fp32 weights
    plus 265 MB of AdamW exp_avg/exp_avg_sq buffers. `last.pt` and `best.pt`
    are full, because resuming needs the optimizer moments -- dropping them
    restarts Adam cold and puts a visible kink in the loss curve.

    Periodic `epoch_XXX.pt` snapshots are weights-only (~133 MB). They exist to
    look back at, not to resume from, and at 60 epochs the full form would push
    output past 5 GB for no benefit.
    """
    blob = {"epoch": epoch, "best_val_loss": best, "epochs_no_improve": bad,
            "model": model.state_dict(), "args": vars(args)}
    if full:
        blob.update({
            "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict(),
            "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                    "python": random.getstate()},
        })
    torch.save(blob, path)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Train the dual-head 3D ResNet-18.")
    ap.add_argument("--data-root", default=None,
                    help="e.g. /kaggle/input/adni-mri-prep; omit to use the repo layout")
    ap.add_argument("--out-dir",
                    default="/kaggle/working" if os.path.isdir("/kaggle/working")
                    else os.path.join(here, "..", "..", "runs"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", default="auto",
                    help="integer, or 'auto' to probe the GPU for the largest safe size")
    ap.add_argument("--probe-sizes", default="4,8,16,24,32",
                    help="candidates for --batch-size auto")
    ap.add_argument("--probe-safety", type=float, default=0.85,
                    help="max fraction of VRAM the auto batch size may reserve")
    ap.add_argument("--lr", type=float, default=1e-4)
    # 1e-2 is AdamW's own reference default; the 1e-4 used for the baseline was
    # borrowed from Adam-with-L2 habits and is weak for decoupled weight decay.
    # Raised deliberately after the baseline memorised the train split.
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--warmup-epochs", type=int, default=3)
    ap.add_argument("--arch", default="resnet10", choices=sorted(ARCHS),
                    help="backbone size; resnet18 is the original baseline")
    ap.add_argument("--dropout", type=float, default=0.3,
                    help="on the pooled feature vector, before both heads")
    # --- augmentation (train split only) ---
    ap.add_argument("--no-aug", action="store_true",
                    help="disable all augmentation (baseline behaviour)")
    ap.add_argument("--aug-rot-deg", type=float, default=5.0,
                    help="max rotation per axis; small because data is registered")
    ap.add_argument("--aug-trans-vox", type=float, default=2.0,
                    help="max translation in voxels (~1.9 mm each at 96^3)")
    ap.add_argument("--aug-brightness", type=float, default=0.05)
    ap.add_argument("--aug-contrast", type=float, default=0.05)
    ap.add_argument("--aug-flip-lr", action="store_true",
                    help="OFF by default on purpose: AD atrophy is asymmetric, "
                         "so mirroring destroys lateralised signal")
    ap.add_argument("--lam-demo", type=float, default=None,
                    help="omit for gradient-balanced auto; give both to pin manually")
    ap.add_argument("--lam-alz", type=float, default=None)
    ap.add_argument("--balance-batches", type=int, default=16,
                    help="batches sampled when measuring gradient balance; the "
                         "estimate converges from above, so do not lower this")
    ap.add_argument("--alz-pos-weight", default="auto",
                    help="float, 'auto' for train-split neg/pos, or 1.0 to disable")
    ap.add_argument("--patience", type=int, default=15, help="early stop on val loss")
    ap.add_argument("--ckpt-every", type=int, default=5)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    ap.add_argument("--resume", action="store_true", help="continue from last.pt")
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke test: cap subjects per split")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (not args.no_amp) and device.type == "cuda"

    print("device      : " + str(device)
          + (" (" + torch.cuda.get_device_name(0) + ")" if device.type == "cuda" else ""))
    print("amp         : " + str(amp))
    print("out-dir     : " + args.out_dir)
    print("data-root   : " + (args.data_root or "(repo layout)"))

    sets = {s: ADNIVolumes(split=s, data_root=args.data_root, strict=True)
            for s in ("train", "val", "test")}
    if args.limit:
        for s in sets:
            sets[s].df = sets[s].df.head(args.limit).reset_index(drop=True)
        print("LIMIT       : {} subjects/split (smoke test, not a real run)".format(args.limit))

    counts = sets["train"].class_counts()
    n_ad = counts.get(3, 0)
    n_tr = len(sets["train"])
    print("subjects    : train {}  val {}  test {}".format(
        n_tr, len(sets["val"]), len(sets["test"])))
    print("train counts: " + "  ".join(
        "{}={}".format(c, counts.get(i, 0)) for i, c in enumerate(CLASSES)))

    # --- ALZ positive weight: measured from the split, never hardcoded --------
    if str(args.alz_pos_weight).lower() == "auto":
        args.alz_pos_weight = (n_tr - n_ad) / max(n_ad, 1)
        src = "auto (train neg/pos)"
    else:
        args.alz_pos_weight = float(args.alz_pos_weight)
        src = "explicit"
    print("alz balance : {}/{} positive ({:.2f}%)  pos_weight = {:.6f}  [{}]".format(
        n_ad, n_tr, 100 * n_ad / max(n_tr, 1), args.alz_pos_weight, src))
    pw = args.alz_pos_weight if args.alz_pos_weight != 1.0 else None

    # --- batch size: probed on the real GPU, not guessed ----------------------
    if str(args.batch_size).lower() == "auto":
        if device.type != "cuda":
            args.batch_size = 4
            print("batch size  : 4  [auto unavailable on CPU -- probe needs a GPU]")
        else:
            from m3_probe_batch import probe_one
            total = torch.cuda.get_device_properties(0).total_memory
            chosen, rows = None, []
            for cand in [int(s) for s in args.probe_sizes.split(",") if s.strip()]:
                ok, res, _alloc, sec, err = probe_one(cand, device, amp)
                if not ok:
                    rows.append("{}:OOM".format(cand))
                    break
                frac = res / total
                rows.append("{}:{:.0%}".format(cand, frac))
                if frac <= args.probe_safety:
                    chosen = cand
            args.batch_size = chosen or 2
            print("batch probe : " + "  ".join(rows))
            print("batch size  : {}  [auto, <= {:.0%} of {:.1f} GiB]".format(
                args.batch_size, args.probe_safety, total / 2**30))
    else:
        args.batch_size = int(args.batch_size)
        print("batch size  : {}  [explicit]".format(args.batch_size))

    pin = device.type == "cuda"
    loaders = {s: DataLoader(sets[s], batch_size=args.batch_size,
                             shuffle=(s == "train"), num_workers=args.num_workers,
                             pin_memory=pin, drop_last=False,
                             persistent_workers=args.num_workers > 0)
               for s in sets}

    model = DualHeadNet(arch=args.arch, dropout=args.dropout).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    _base = 33161026
    print("arch        : {}  ({:,} params{})".format(
        args.arch, n_par,
        "" if args.arch == "resnet18"
        else ", {:.1f}x smaller than resnet18".format(_base / n_par)))
    print("dropout     : {}".format(args.dropout))
    print("weight decay: {}".format(args.weight_decay))
    n_par = n_par

    # --- loss weights: measured gradient balance, then frozen ----------------
    grad_norms = None
    if args.lam_demo is None or args.lam_alz is None:
        gen = torch.Generator().manual_seed(args.seed)
        bl = DataLoader(sets["train"], batch_size=args.batch_size, shuffle=True,
                        generator=gen, num_workers=0)
        g_d, g_a = measure_grad_balance(model, bl, device, pw, args.balance_batches)
        lam_d, lam_a = balanced_lambdas(g_d, g_a)
        args.lam_demo = lam_d if args.lam_demo is None else args.lam_demo
        args.lam_alz = lam_a if args.lam_alz is None else args.lam_alz
        grad_norms = {"trunk_grad_norm_demo": g_d, "trunk_grad_norm_alz": g_a,
                      "ratio_demo_over_alz": g_d / max(g_a, 1e-12)}
        print("grad norms  : demo {:.4f}   alz {:.4f}   ratio {:.3f}x  "
              "({} batches, fp32)".format(g_d, g_a, g_d / max(g_a, 1e-12),
                                          args.balance_batches))
        print("lambdas     : demo {:.4f}   alz {:.4f}   [auto, gradient-balanced]".format(
            args.lam_demo, args.lam_alz))
    else:
        print("lambdas     : demo {:.4f}   alz {:.4f}   [explicit]".format(
            args.lam_demo, args.lam_alz))
    print("loss        : {:.4f} * MSE(DEMO) + {:.4f} * BCE(ALZ, pos_weight={})".format(
        args.lam_demo, args.lam_alz,
        "off" if pw is None else "{:.4f}".format(pw)))

    aug = None if args.no_aug else {
        "rot_deg": args.aug_rot_deg, "trans_vox": args.aug_trans_vox,
        "brightness": args.aug_brightness, "contrast": args.aug_contrast,
        "flip_lr": args.aug_flip_lr}
    if aug is None:
        print("augment     : OFF (baseline)")
    else:
        print("augment     : rot +-{}deg  trans +-{}vox  bright +-{}  "
              "contrast +-{}  flip_lr {}  [TRAIN SPLIT ONLY]".format(
                  args.aug_rot_deg, args.aug_trans_vox, args.aug_brightness,
                  args.aug_contrast, args.aug_flip_lr))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    def lr_lambda(e):
        if e < args.warmup_epochs:
            return (e + 1) / max(args.warmup_epochs, 1)
        prog = (e - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
        return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    start_epoch, best, bad = 0, float("inf"), 0
    last_path = os.path.join(args.out_dir, "last.pt")
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1
        best = ck["best_val_loss"]
        bad = ck["epochs_no_improve"]
        rs = ck["rng"]["torch"]
        torch.set_rng_state(rs.cpu() if hasattr(rs, "cpu") else rs)
        np.random.set_state(ck["rng"]["numpy"])
        random.setstate(ck["rng"]["python"])
        # THE SCHEDULER IS REBUILT, NOT RESTORED -- deliberately.
        # opt.load_state_dict() brings back whatever LR the previous run left in
        # the optimizer, which is the LR computed for the *next* epoch under the
        # *previous* --epochs. Resume with a different --epochs and that value is
        # wrong; resume from a run that reached its cosine minimum and it is
        # exactly 0.0, so the first resumed epoch trains at zero LR and silently
        # does nothing. Replaying step() makes the LR a pure function of
        # (--lr, --epochs, --warmup-epochs, epoch index) instead.
        for _ in range(start_epoch):
            sched.step()
        print("resumed     : from epoch {} (best val loss {:.4f}, lr {:.2e})".format(
            ck["epoch"], best, opt.param_groups[0]["lr"]))

    csv_path = os.path.join(args.out_dir, "epoch_log.csv")
    fields = (["epoch", "lr", "train_loss", "train_mse", "train_bce",
               "val_loss", "val_mse", "val_bce", "val_demo_mae", "val_acc4",
               "val_macro_f1", "val_bal_acc", "val_alz_auc", "val_alz_acc",
               "val_alz_sens", "val_alz_spec"]
              + ["val_{}_{}".format(k, c)
                 for k in ("f1", "prec", "rec", "sup") for c in CLASSES]
              + ["train_seconds", "val_seconds", "epoch_seconds"])
    new_file = not (args.resume and os.path.exists(csv_path))
    fcsv = open(csv_path, "a", newline="")
    writer = csv.DictWriter(fcsv, fieldnames=fields)
    if new_file:
        writer.writeheader()
        fcsv.flush()

    with open(os.path.join(args.out_dir, "train_config.json"), "w") as f:
        json.dump({**vars(args), "device": str(device), "amp": amp,
                   "params": n_par, "train_counts": counts, "augment": aug,
                   "grad_balance": grad_norms}, f, indent=2)

    print("\n{:>3} {:>8} {:>8} {:>8} {:>6} {:>6} {:>6} {:>6} {:>6}".format(
        "ep", "lr", "train", "val", "mae", "acc4", "mF1", "auc", "sec"))
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        tr = run_epoch(model, loaders["train"], device, args.lam_demo,
                       args.lam_alz, pw, optimizer=opt, scaler=scaler,
                       amp=amp, aug=aug)
        t1 = time.time()
        va = run_epoch(model, loaders["val"], device, args.lam_demo,
                       args.lam_alz, pw, amp=amp)
        t2 = time.time()
        lr_now = opt.param_groups[0]["lr"]
        sched.step()

        row = {"epoch": epoch, "lr": lr_now,
               "train_loss": tr["loss"], "train_mse": tr["mse"], "train_bce": tr["bce"],
               "val_loss": va["loss"], "val_mse": va["mse"], "val_bce": va["bce"],
               "val_demo_mae": va["demo_mae"], "val_acc4": va["acc4"],
               "val_macro_f1": va["macro_f1"], "val_bal_acc": va["bal_acc"],
               "val_alz_auc": va["alz_auc"], "val_alz_acc": va["alz_acc"],
               "val_alz_sens": va["alz_sens"], "val_alz_spec": va["alz_spec"],
               "train_seconds": t1 - t0, "val_seconds": t2 - t1,
               "epoch_seconds": t2 - t0}
        for k in ("f1", "prec", "rec", "sup"):
            for c in CLASSES:
                row["val_{}_{}".format(k, c)] = va["{}_{}".format(k, c)]
        writer.writerow(row)
        fcsv.flush()

        improved = va["loss"] < best - 1e-6
        if improved:
            best, bad = va["loss"], 0
            save_ckpt(os.path.join(args.out_dir, "best.pt"), model, opt, sched,
                      scaler, epoch, best, bad, args)
        else:
            bad += 1
        save_ckpt(last_path, model, opt, sched, scaler, epoch, best, bad, args)
        if args.ckpt_every and (epoch + 1) % args.ckpt_every == 0:
            save_ckpt(os.path.join(args.out_dir, "epoch_{:03d}.pt".format(epoch)),
                      model, opt, sched, scaler, epoch, best, bad, args, full=False)

        print("{:>3} {:>8.2e} {:>8.4f} {:>8.4f} {:>6.3f} {:>6.3f} {:>6.3f} "
              "{:>6.3f} {:>6.1f}{}".format(
                  epoch, lr_now, tr["loss"], va["loss"], va["demo_mae"],
                  va["acc4"], va["macro_f1"], va["alz_auc"], t2 - t0,
                  "  *best" if improved else ""))

        if bad >= args.patience:
            print("\nearly stop: no val-loss improvement in {} epochs".format(args.patience))
            break

    fcsv.close()
    print("\nbest val loss: {:.4f}".format(best))
    print("artefacts    : " + args.out_dir)


if __name__ == "__main__":
    main()
