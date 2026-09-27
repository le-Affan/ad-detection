"""Path A, stage C2: end-to-end training of the temporal model.

C1 froze the v9 trunk and found that, on its features, a visit sequence adds
little beyond the latest scan. C2 asks whether a trunk FINE-TUNED on sequences
learns change the frozen one could not see. Three runs, one command each:

    --target dx_last --visits all    GRU over every usable visit
    --target dx_last --visits last   matched control: identical model and
                                     training, but only the last usable visit
                                     is loaded -- so any gap between these two
                                     is temporal signal, not "saw a later scan"
    --target dx_bl   --visits all    v9's own targets (DEMO severity + ALZ),
                                     the output the fusion layer consumes

TARGETS.
  dx_bl   DEMO and ALZ from DX_bl exactly as dataset.py defines them, trained
          with v9 lr_3e-5's own lambdas and ALZ pos_weight (train_config.json).
  dx_last binary: Dementia at the subject's last usable MRI visit (DXSUM, via
          p4_temporal_c1.dx_last_labels). ALZ head only, class weight from this
          target's train split. The DEMO head is untrained and not reported.

INITIALISATION AND OPTIMISATION. The trunk starts from v9 lr_3e-5's best.pt
(C1: the stronger of the three) and trains at that run's own learning rate,
3e-5; the new temporal module and heads train at --head-lr (1e-4, where C1
converged). AdamW, weight decay 1e-2, linear warmup then cosine, AMP, early
stopping on val loss -- m3's recipe, so the only new variable is the sequence.

AUGMENTATION reuses m3_train.augment_batch unchanged, applied twice so it means
the same thing per visit that it meant per scan:
  geometry   one rotation/translation per SUBJECT, shared by all its visits
             (the visits are stacked as channels of one volume). Independent
             jitter would add visit-to-visit misalignment that looks exactly
             like the change the model is supposed to learn.
  intensity  brightness/contrast per VISIT, on the usable volumes only, as m3
             does per scan -- scanner gain genuinely varies between sessions,
             and masked zero slots never enter a contrast mean.

CHECKPOINTS assume interruption, as m3's do: last.pt every epoch, best.pt on
each val-loss improvement, the epoch CSV flushed every epoch, and --resume
continues from last.pt with optimizer, scheduler, scaler and early-stop state.
Weights go to analysis/patha/<name>/ (*.pt git-ignored).

SELECTION is on val, as for v9's best.pt; test is not touched here.
"""
import argparse
import copy
import csv
import json
import math
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dataset import ROOT  # noqa: E402
from dataset_visits import ADNIVisitSequences  # noqa: E402
from m3_train import augment_batch, compute_metrics, set_seed  # noqa: E402
from model_temporal import TemporalDualHeadNet  # noqa: E402
from p4_temporal_c1 import binary_metrics, dx_last_labels  # noqa: E402

V9 = os.path.join(ROOT, "analysis", "kaggle_output", "v9")
RUNS = os.path.join(ROOT, "analysis", "patha")


class LastVisitOnly(torch.utils.data.Dataset):
    """Wraps ADNIVisitSequences so only each subject's last usable visit exists.

    The other slots are masked BEFORE loading, so they are never read, and the
    item is otherwise identical -- same shape, same targets -- which is what
    makes the last-visit run a matched control rather than a different model.
    """

    def __init__(self, ds):
        self.ds = ds
        g = ds.df.groupby("PTID")
        last = g.apply(lambda x: x[x.present].slot_idx.max(), include_groups=False)
        ds.df["present"] = ds.df.apply(lambda r: bool(r.present and r.slot_idx == last[r.PTID]), axis=1)
        ds._rows = {p: x for p, x in ds.df.groupby("PTID")}

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        return self.ds[i]


def make_split(split, target, visits):
    ds = ADNIVisitSequences(split)
    if visits == "last":
        ds = LastVisitOnly(ds)
    base = ds.ds if isinstance(ds, LastVisitOnly) else ds
    y = None
    if target == "dx_last":
        y = dx_last_labels(base.ptids)
        keep = np.flatnonzero(~np.isnan(y))
        return Subset(ds, keep), {base.ptids[i]: float(y[i]) for i in keep}
    return ds, None


def targets(batch, ylast, device):
    if ylast is None:
        return batch["demo"].to(device), batch["alz"].to(device), batch["label"].numpy()
    y = torch.tensor([ylast[p] for p in batch["ptid"]], dtype=torch.float32)
    return torch.zeros_like(y).to(device), y.to(device), y.numpy()


def run_epoch(net, loader, ylast, cfg, opt, sched, scaler, device, aug, train):
    net.train(train)
    pw = torch.tensor(cfg["alz_pos_weight"], device=device)
    tot = n = 0.0
    demo_p, logit_p, truth = [], [], []
    for batch in loader:
        vols = batch["volumes"].to(device, non_blocking=True)      # (B,4,1,D,H,W)
        mask = batch["mask"].to(device)
        days = batch["days"].to(device)
        demo, alz, truth_b = targets(batch, ylast, device)
        if train and aug:
            b, s = vols.shape[:2]
            vols = augment_batch(vols.view(b, s, *vols.shape[3:]),
                                 rot_deg=aug["rot_deg"], trans_vox=aug["trans_vox"]).view_as(vols)
            used = augment_batch(vols[mask], brightness=aug["brightness"], contrast=aug["contrast"])
            vols = vols.clone()
            vols[mask] = used
            vols[~mask] = 0
        with torch.set_grad_enabled(train), torch.autocast("cuda", enabled=device == "cuda"):
            out = net(vols, mask, days)
        mse = torch.nn.functional.mse_loss(out["demo"].float(), demo)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(out["alz_logit"].float(), alz, pos_weight=pw)
        loss = cfg["lam_demo"] * mse + cfg["lam_alz"] * bce
        if train:
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        bs = len(alz)
        tot += float(loss.detach()) * bs
        n += bs
        demo_p.append(out["demo"].detach().float().cpu())
        logit_p.append(out["alz_logit"].detach().float().cpu())
        truth.append(truth_b)
    d, lg, t = torch.cat(demo_p).numpy(), torch.cat(logit_p).numpy(), np.concatenate(truth)
    met = (binary_metrics(d, lg, t) if ylast is not None else compute_metrics(d, lg, t))
    met["loss"] = tot / n
    return met


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--target", required=True, choices=["dx_bl", "dx_last"])
    ap.add_argument("--visits", default="all", choices=["all", "last"])
    ap.add_argument("--temporal", default="gru")
    ap.add_argument("--init", default="lr_3e-5", help="v9 run whose best.pt initialises the trunk")
    ap.add_argument("--name", help="run directory name under analysis/patha/")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8, help="subjects per batch")
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--warmup-epochs", type=int, default=2)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="train subjects cap, for smoke tests")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    a.name = a.name or f"{a.temporal}_{a.target}_{a.visits}"
    out = os.path.join(RUNS, a.name)
    os.makedirs(out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(a.seed)

    v9cfg = json.load(open(os.path.join(V9, a.init, "train_config.json")))
    tr, ytr = make_split("train", a.target, a.visits)
    va, yva = make_split("val", a.target, a.visits)
    ylast = {**(ytr or {}), **(yva or {})} or None
    if a.target == "dx_bl":
        cfg = {k: v9cfg[k] for k in ("lam_demo", "lam_alz", "alz_pos_weight")}
    else:
        pos = sum(ytr.values())
        cfg = {"lam_demo": 0.0, "lam_alz": 1.0, "alz_pos_weight": (len(ytr) - pos) / pos}
    if a.limit:
        tr = Subset(tr, range(min(a.limit, len(tr))))
    aug = None if a.no_aug else v9cfg["augment"]
    trunk_lr = v9cfg["lr"]

    net = TemporalDualHeadNet(a.temporal, arch=v9cfg["arch"], hidden=a.hidden, dropout=a.dropout)
    net.load_trunk(os.path.join(V9, a.init, "best.pt"))
    net = net.to(device)
    trunk_p = list(net.trunk.parameters())
    other_p = [p for n_, p in net.named_parameters() if not n_.startswith("trunk.")]
    opt = torch.optim.AdamW([{"params": trunk_p, "lr": trunk_lr},
                             {"params": other_p, "lr": a.head_lr}], weight_decay=a.weight_decay)
    loader = dict(batch_size=a.batch_size, num_workers=a.num_workers, pin_memory=True,
                  persistent_workers=a.num_workers > 0)
    dl_tr = DataLoader(tr, shuffle=True, drop_last=True, **loader)
    dl_va = DataLoader(va, shuffle=False, **loader)
    steps, warm = a.epochs * len(dl_tr), a.warmup_epochs * len(dl_tr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(steps - warm, 1))))
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    start, best, bad = 0, float("inf"), 0
    if a.resume and os.path.exists(os.path.join(out, "last.pt")):
        ck = torch.load(os.path.join(out, "last.pt"), map_location=device, weights_only=False)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start, best, bad = ck["epoch"] + 1, ck["best"], ck["bad"]
        print(f"resumed at epoch {start}")

    conf = {**vars(a), **cfg, "trunk_lr": trunk_lr, "arch": v9cfg["arch"], "augment": aug,
            "n_train": len(tr), "n_val": len(va), "device": device,
            "params": sum(p.numel() for p in net.parameters())}
    json.dump(conf, open(os.path.join(out, "train_config.json"), "w"), indent=2)
    print(f"{a.name}: train {len(tr)} / val {len(va)} subjects, {len(dl_tr)} steps/epoch, "
          f"lam_demo {cfg['lam_demo']:.3f} lam_alz {cfg['lam_alz']:.3f} pos_weight {cfg['alz_pos_weight']:.3f}, "
          f"trunk lr {trunk_lr:g} head lr {a.head_lr:g}", flush=True)

    log = os.path.join(out, "epoch_log.csv")
    for ep in range(start, a.epochs):
        t0 = time.time()
        trm = run_epoch(net, dl_tr, ylast, cfg, opt, sched, scaler, device, aug, True)
        t1 = time.time()
        with torch.no_grad():
            vam = run_epoch(net, dl_va, ylast, cfg, opt, sched, scaler, device, None, False)
        row = {"epoch": ep, "lr_trunk": opt.param_groups[0]["lr"], "train_loss": trm["loss"],
               **{f"val_{k}": v for k, v in vam.items()}, "train_s": t1 - t0, "val_s": time.time() - t1}
        with open(log, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if f.tell() == 0:
                w.writeheader()
            w.writerow(row)
        improved = vam["loss"] < best - 1e-4
        if improved:
            best, bad = vam["loss"], 0
            torch.save({"model": net.state_dict(), "args": conf, "epoch": ep, "val": vam},
                       os.path.join(out, "best.pt"))
        else:
            bad += 1
        torch.save({"model": net.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": ep, "best": best, "bad": bad, "args": conf}, os.path.join(out, "last.pt"))
        extra = (f"macro_f1 {vam['macro_f1']:.3f} demo_mae {vam['demo_mae']:.3f}"
                 if a.target == "dx_bl" else f"bal_acc {vam['bal_acc']:.3f}")
        print(f"ep {ep:2d}  train {trm['loss']:.4f}  val {vam['loss']:.4f}  auc {vam['alz_auc']:.3f}  "
              f"{extra}  {row['train_s']:.0f}s+{row['val_s']:.0f}s{'  *best' if improved else ''}", flush=True)
        if bad >= a.patience:
            print(f"early stop: no val-loss improvement for {a.patience} epochs")
            break
    print(f"done. best val loss {best:.4f} -> {os.path.join(out, 'best.pt')}")


if __name__ == "__main__":
    main()
