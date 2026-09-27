"""Path A, stage C1: compare temporal modules on frozen trunk embeddings.

End-to-end training of a 3D CNN over four visits is the expensive step, so the
question it would answer first -- is there temporal signal at all, and which
module reads it best -- is asked here cheaply. The static v9 trunk is frozen,
every usable visit is encoded ONCE, and only the temporal module and heads are
trained, in minutes.

WHAT IS COMPARED, all on the same Path A validation subjects:

  static   the v9 checkpoints as trained -- DualHeadNet on each subject's
           original baseline volume (dataset/mri_prep), untouched. This is the
           number Path A has to beat.
  first    a temporal head trained on the FIRST usable visit only
  last     the same, on the LAST usable visit only
  delta / gru / attn   the three temporal modules on every usable visit

`first` is the control that matters. It retrains a head on the same frozen
embeddings with one visit, so any gap between it and the sequence models is
temporal signal, not the benefit of a retrained head or of this script's
optimiser. `last` separates "the later scan is more atrophied, so easier" from
"the change between scans is informative".

SAME OBJECTIVE AS THE BASELINE. lam_demo, lam_alz and the ALZ pos_weight are
read from the chosen checkpoint's train_config.json, not re-measured, so every
model here optimises exactly v9's loss. Metrics are m3_train.compute_metrics.

SELECTION ON VALIDATION. Each run keeps its lowest-val-loss epoch and reports
val metrics at that epoch -- as v9's best.pt was chosen -- so the numbers are
optimistic by the same mechanism for every row, and comparable to each other.
The test split is not touched. Every row is 5 seeds; with 104 val subjects
(17 AD) a difference inside a couple of standard deviations is noise.

Embeddings are z-scored per feature with statistics from TRAIN usable visits
only; masked slots stay exactly zero.
"""
import argparse
import copy
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dataset import ROOT, ADNIVolumes  # noqa: E402
from dataset_visits import ADNIVisitSequences  # noqa: E402
from m3_train import compute_metrics  # noqa: E402
from model import DualHeadNet  # noqa: E402
from model_temporal import TemporalDualHeadNet  # noqa: E402

KO = os.path.join(ROOT, "analysis", "kaggle_output", "v9")
OUT_DIR = os.path.join(ROOT, "dataset", "mri_prep_patha")
SPLITS = ("train", "val", "test")
MODES = ("first", "last", "delta", "gru", "attn")
REPORT = ["alz_auc", "macro_f1", "bal_acc", "acc4", "demo_mae", "alz_sens", "alz_spec",
          "f1_CN", "f1_EMCI", "f1_LMCI", "f1_AD"]


def ckpt_path(run):
    return os.path.join(KO, run, "best.pt")


def load_static(run, device):
    ck = torch.load(ckpt_path(run), map_location="cpu", weights_only=False)
    a = ck["args"]
    net = DualHeadNet(arch=a["arch"], dropout=a["dropout"])
    net.load_state_dict(ck["model"])
    return net.to(device).eval(), a


@torch.no_grad()
def build_cache(run, device, path):
    """Frozen-trunk embedding of every usable visit, all splits."""
    net, a = load_static(run, device)
    out = {"emb": [], "mask": [], "days": [], "label": [], "ptid": [], "split": []}
    for split in SPLITS:
        ds = ADNIVisitSequences(split)
        for i in range(len(ds)):
            it = ds[i]
            m = it["mask"]
            f = net.trunk(it["volumes"][m].to(device)).float().cpu()
            e = torch.zeros(4, f.shape[1])
            e[m] = f
            for k, v in (("emb", e), ("mask", m), ("days", it["days"]),
                         ("label", it["label"]), ("ptid", it["ptid"]), ("split", split)):
                out[k].append(v)
        print(f"  cached {split}: {len(ds)} subjects", flush=True)
    np.savez(path, emb=torch.stack(out["emb"]).numpy(), mask=torch.stack(out["mask"]).numpy(),
             days=torch.stack(out["days"]).numpy(), label=np.array([int(x) for x in out["label"]]),
             ptid=np.array(out["ptid"]), split=np.array(out["split"]), arch=a["arch"], run=run)


@torch.no_grad()
def static_metrics(run, device, ptids):
    """v9 as trained, on its own baseline volumes, restricted to `ptids`."""
    net, _ = load_static(run, device)
    ds = ADNIVolumes("val")
    ds.df = ds.df[ds.df.PTID.isin(ptids)].reset_index(drop=True)
    demo, logit, lab = [], [], []
    for i in range(len(ds)):
        it = ds[i]
        o = net(it["volume"].unsqueeze(0).to(device))
        demo.append(float(o["demo"]))
        logit.append(float(o["alz_logit"]))
        lab.append(int(it["label"]))
    return compute_metrics(np.array(demo), np.array(logit), np.array(lab)), len(ds)


def restrict(mask, mode):
    """Keep only the first or last usable visit of each subject."""
    if mode not in ("first", "last"):
        return mask
    idx = torch.arange(4).expand_as(mask)
    pick = (torch.where(mask, idx, 4).min(1).values if mode == "first"
            else torch.where(mask, idx, -1).max(1).values)
    return torch.nn.functional.one_hot(pick, 4).bool()


def fit(mode, data, arch, cfg, seed, device, lr=1e-3, hidden=128, epochs=300, patience=40, bs=64):
    torch.manual_seed(seed)
    np.random.seed(seed)
    kind = "delta" if mode in ("first", "last") else mode   # one visit: delta is a plain MLP
    net = TemporalDualHeadNet(kind, arch=arch, hidden=hidden, dropout=0.3)
    net.trunk = None                                           # frozen and cached: not needed
    net = net.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    pw = torch.tensor(cfg["alz_pos_weight"], device=device)

    def batches(s, shuffle):
        emb, mask, days, demo, alz, lab = data[s]
        mask = restrict(mask, mode)
        n = len(lab)
        order = torch.randperm(n) if shuffle else torch.arange(n)
        for i in range(0, n, bs):
            j = order[i:i + bs]
            yield (emb[j].to(device), mask[j].to(device), days[j].to(device),
                   demo[j].to(device), alz[j].to(device), lab[j])

    def loss_of(o, demo, alz):
        mse = torch.nn.functional.mse_loss(o["demo"], demo)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(o["alz_logit"], alz, pos_weight=pw)
        return cfg["lam_demo"] * mse + cfg["lam_alz"] * bce

    best, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        net.train()
        for e, m, d, demo, alz, _ in batches("train", True):
            opt.zero_grad()
            loss_of(net.forward_embeddings(e, m, d), demo, alz).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            tot, n = 0.0, 0
            for e, m, d, demo, alz, _ in batches("val", False):
                tot += float(loss_of(net.forward_embeddings(e, m, d), demo, alz)) * len(demo)
                n += len(demo)
        if tot / n < best - 1e-4:
            best, best_state, bad, best_ep = tot / n, copy.deepcopy(net.state_dict()), 0, ep
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    dp, lg, lb = [], [], []
    with torch.no_grad():
        for e, m, d, _, _, lab in batches("val", False):
            o = net.forward_embeddings(e, m, d)
            dp.append(o["demo"].cpu())
            lg.append(o["alz_logit"].cpu())
            lb.append(lab)
    met = compute_metrics(torch.cat(dp).numpy(), torch.cat(lg).numpy(), torch.cat(lb).numpy())
    met.update(val_loss=best, best_epoch=best_ep)
    return met


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default="lr_5e-5", help="v9 run whose frozen trunk is cached")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from dataset import DEMO_BY_LABEL, ALZ_BY_LABEL

    cache = os.path.join(OUT_DIR, f"c1_emb_v9_{a.run}.npz")
    if not os.path.exists(cache):
        print(f"caching frozen {a.run} trunk embeddings -> {cache}")
        build_cache(a.run, device, cache)
    z = np.load(cache)
    cfg = json.load(open(os.path.join(KO, a.run, "train_config.json")))

    emb = torch.tensor(z["emb"])
    mask = torch.tensor(z["mask"])
    tr = z["split"] == "train"
    usable = emb[torch.tensor(tr)][mask[torch.tensor(tr)]]
    mu, sd = usable.mean(0), usable.std(0).clamp_min(1e-6)
    emb = torch.where(mask.unsqueeze(-1), (emb - mu) / sd, torch.zeros_like(emb))

    data = {}
    for s in SPLITS:
        k = torch.tensor(z["split"] == s)
        lab = torch.tensor(z["label"])[k]
        data[s] = (emb[k], mask[k], torch.tensor(z["days"])[k],
                   torch.tensor([DEMO_BY_LABEL[int(x)] for x in lab], dtype=torch.float32),
                   torch.tensor([ALZ_BY_LABEL[int(x)] for x in lab], dtype=torch.float32), lab)
    val_ptids = list(z["ptid"][z["split"] == "val"])
    print(f"train {int(tr.sum())} / val {len(val_ptids)} subjects; objective from {a.run}: "
          f"lam_demo {cfg['lam_demo']:.3f}, lam_alz {cfg['lam_alz']:.3f}, pos_weight {cfg['alz_pos_weight']:.3f}")

    rows = []
    for run in ("lr_1e-4", "lr_5e-5", "lr_3e-5"):
        m, n = static_metrics(run, device, val_ptids)
        rows.append({"model": f"static v9 {run}", "seed": 0, **m})
    print(f"static baselines evaluated on {n} val subjects (their original baseline volumes)")

    for mode in a.modes.split(","):
        for seed in range(a.seeds):
            m = fit(mode, data, str(z["arch"]), cfg, seed, device, lr=a.lr, hidden=a.hidden)
            rows.append({"model": mode, "seed": seed, **m})
        r = pd.DataFrame([x for x in rows if x["model"] == mode])
        print(f"  {mode:<6s} alz_auc {r.alz_auc.mean():.3f}±{r.alz_auc.std():.3f}  "
              f"macro_f1 {r.macro_f1.mean():.3f}±{r.macro_f1.std():.3f}  "
              f"demo_mae {r.demo_mae.mean():.3f}  best_epoch ~{r.best_epoch.median():.0f}", flush=True)

    df = pd.DataFrame(rows)
    tag = f"{a.run}_lr{a.lr:g}_h{a.hidden}"
    df.to_csv(os.path.join(OUT_DIR, f"c1_results_{tag}.csv"), index=False)
    g = df.groupby("model", sort=False)[REPORT]
    summ = g.mean().round(3).astype(str) + " ± " + g.std().fillna(0).round(3).astype(str)
    print("\n" + summ[["alz_auc", "macro_f1", "bal_acc", "demo_mae", "alz_sens", "alz_spec"]].to_string())
    print("\nper-class F1 (mean):\n" + g.mean()[["f1_CN", "f1_EMCI", "f1_LMCI", "f1_AD"]].round(3).to_string())
    print(f"\nwrote {os.path.join(OUT_DIR, f'c1_results_{tag}.csv')}")


if __name__ == "__main__":
    main()
