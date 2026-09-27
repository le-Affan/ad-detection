"""Cross-fitted ALZ/DEMO from the fold models, plus the sanity check. Never reads test.

For fold k, the 3 seed checkpoints (best val loss, m3's selection rule) score:
  - fold k's own subjects (clinical-train, never trained or selected on), and
  - the 121 clinical-val subjects with MRI.
A fold model's ALZ is the mean of its seeds' sigmoid(ALZ logit); DEMO likewise.
Clinical-train subjects get their fold model's value ("fold" = k); clinical-val
subjects get the mean over the 5 fold models ("fold" = "full", i.e. the
fold-model ensemble -- see report). Subjects without MRI get NaN.

Writes alz_crossfit.csv, alz_per_seed.csv and prints the AUC sanity check.
Only clinical-train and clinical-val RIDs are ever loaded; test RIDs are
asserted absent.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src", "mri"))
from dataset import ADNIVolumes  # noqa: E402
from model import DualHeadNet  # noqa: E402

PROCESSED = os.path.join(ROOT, "dataset", "processed")
K = 5
SEEDS = (42, 43, 44)


def load_model(ckpt, device):
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    a = blob["args"]
    m = DualHeadNet(arch=a["arch"], dropout=float(a.get("dropout") or 0.0))
    m.load_state_dict(blob["model"], strict=True)
    return m.to(device).eval(), int(blob["epoch"])


@torch.no_grad()
def score(model, vols, device):
    demo, logit = [], []
    for s in range(0, len(vols), 16):
        o = model(vols[s:s + 16].to(device))
        demo.append(o["demo"].double().cpu().numpy())
        logit.append(o["alz_logit"].double().cpu().numpy())
    return np.concatenate(demo), 1 / (1 + np.exp(-np.concatenate(logit)))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_of = {int(r): s for s in ("train", "val", "test")
                for r in pd.read_csv(os.path.join(PROCESSED, f"{s}_rids.csv")).RID}
    folds = pd.read_csv(os.path.join(HERE, "folds", "folds.csv"))

    ds = ADNIVolumes(split=None, strict=True)
    ds.df = ds.df[ds.df.RID.map(split_of).isin(["train", "val"])].reset_index(drop=True)
    assert not set(ds.df.RID) & {r for r, s in split_of.items() if s == "test"}
    ds.df["fold"] = ds.df.RID.map(folds.set_index("RID").fold)
    ds.df["clin_split"] = ds.df.RID.map(split_of)
    assert ds.df[ds.df.clin_split == "train"].fold.notna().all()
    val_rows = np.flatnonzero(ds.df.clin_split == "val")
    print(f"scoring: {int((ds.df.clin_split == 'train').sum())} train (cross-fit), "
          f"{len(val_rows)} val (fold-model ensemble)")

    per_seed, val_alz, val_demo = [], [], []
    tr_out = []
    for k in range(K):
        rows = np.concatenate([np.flatnonzero(ds.df.fold == k), val_rows])
        vols = torch.stack([ds[i]["volume"] for i in rows])
        demos, alzs = [], []
        for seed in SEEDS:
            model, ep = load_model(os.path.join(HERE, "runs", f"fold{k}_s{seed}", "best.pt"), device)
            d, a = score(model, vols, device)
            demos.append(d)
            alzs.append(a)
            sub = ds.df.iloc[rows]
            per_seed.append(pd.DataFrame({"RID": sub.RID.values, "fold_model": k, "seed": seed,
                                          "best_epoch": ep, "ALZ": a, "DEMO": d,
                                          "clin_split": sub.clin_split.values}))
            del model
        demo, alz = np.mean(demos, 0), np.mean(alzs, 0)
        n = int((ds.df.fold == k).sum())
        tr_out.append(pd.DataFrame({"RID": ds.df.RID.values[rows[:n]], "fold": str(k),
                                    "ALZ": alz[:n], "DEMO": demo[:n]}))
        val_alz.append(alz[n:])
        val_demo.append(demo[n:])
        print(f"fold{k}: scored {n} held-out + {len(val_rows)} val")

    va = pd.DataFrame({"RID": ds.df.RID.values[val_rows], "fold": "full",
                       "ALZ": np.mean(val_alz, 0), "DEMO": np.mean(val_demo, 0)})
    cf = pd.concat(tr_out + [va])

    # every clinical train/val subject, MRI or not
    allr = pd.DataFrame({"RID": [r for r, s in split_of.items() if s in ("train", "val")]})
    allr["clin_split"] = allr.RID.map(split_of)
    out = allr.merge(cf, on="RID", how="left")
    out["has_mri"] = out.ALZ.notna()
    out["PTID"] = out.RID.map(pd.read_csv(os.path.join(ROOT, "dataset", "mri_manifest",
                                                       "mri_manifest.csv")).set_index("RID").PTID)
    lab = pd.read_csv(os.path.join(ROOT, "dataset", "processed", "step2_features.csv"),
                      usecols=["RID", "label"]).set_index("RID").label
    out["label"] = out.RID.map(lab)
    out = out[["RID", "PTID", "clin_split", "fold", "ALZ", "DEMO", "has_mri", "label"]]
    out.sort_values(["clin_split", "RID"]).to_csv(os.path.join(HERE, "alz_crossfit.csv"), index=False)
    ps = pd.concat(per_seed)
    ps.to_csv(os.path.join(HERE, "alz_per_seed.csv"), index=False)

    # --- sanity check ---------------------------------------------------------
    m = out[out.has_mri]
    print("\nALZ AUC, AD vs rest")
    for s in ("train", "val"):
        g = m[m.clin_split == s]
        print(f"  {s:5s} ({'cross-fitted' if s == 'train' else '5-fold ensemble'}, n={len(g)}): "
              f"{roc_auc_score(g.label == 3, g.ALZ):.4f}")
    t = m[m.clin_split == "train"]
    for k in range(K):
        g = t[t.fold == str(k)]
        print(f"    train fold{k} (n={len(g)}): {roc_auc_score(g.label == 3, g.ALZ):.4f}")
    print("  single-seed, single-fold-model AUCs:")
    for (k, seed), g in ps.merge(out[["RID", "label"]], on="RID").groupby(["fold_model", "seed"]):
        h = g[g.clin_split == "train"]
        v = g[g.clin_split == "val"]
        print(f"    fold{k} s{seed} ep{g.best_epoch.iloc[0]:>2}: held-out {roc_auc_score(h.label == 3, h.ALZ):.3f}"
              f"  val {roc_auc_score(v.label == 3, v.ALZ):.3f}")
    print("\nper-class ALZ / DEMO (mean ± sd)")
    print(m.groupby(["clin_split", "label"])[["ALZ", "DEMO"]].agg(["mean", "std"]).round(3).to_string())


if __name__ == "__main__":
    main()
