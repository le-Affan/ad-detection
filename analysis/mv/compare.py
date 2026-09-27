"""Compare the multi-visit static arms on VALIDATION. Never reads test.

Two readouts per run, both at the checkpoint m3 selected (best val loss):

  1. LOGGED -- the best-val-loss row of the run's own epoch_log.csv, exactly as
     m3 reported it (121 val subjects, AMP forward). This is the like-for-like
     comparison with v9's logged numbers.
  2. RE-SCORED -- best.pt re-run in fp32 over the 121 val baseline volumes,
     scored with m3's own compute_metrics on all 121 and on the 104-subject
     Path A val subset (for cross-reference with the C1/C2 tables), plus the
     fitted-cutpoint macro-F1 from refit_thresholds.py. The fitted value is a
     CEILING (3 cutpoints fitted on the same subjects they are scored on), not
     a score.

v9 lr_3e-5's own best.pt goes through the same re-scoring, so every row of
the re-scored table comes from identical code.

Val only by construction: the only dataset built here is ADNIVolumes("val").
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.join(ROOT, "analysis", "mv")
sys.path.insert(0, os.path.join(ROOT, "src", "mri"))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from dataset import ADNIVolumes  # noqa: E402
from m3_train import compute_metrics  # noqa: E402
from model import DualHeadNet  # noqa: E402
from refit_thresholds import fit_cutpoints  # noqa: E402

V9 = os.path.join(ROOT, "analysis", "kaggle_output", "v9", "lr_3e-5")
VISITS_INDEX = os.path.join(ROOT, "dataset", "mri_prep_patha", "visits_index.csv")
ARMS = ["bl_non", "mv_non", "bl_inv", "mv_inv"]
KEYS = ["acc4", "macro_f1", "bal_acc", "alz_auc", "demo_mae",
        "f1_CN", "f1_EMCI", "f1_LMCI", "f1_AD"]


def finished(name):
    """A run counts once m3 has printed its closing line; in-flight runs don't."""
    log = os.path.join(HERE, name + ".log")
    return os.path.exists(log) and any(l.startswith("best val loss") for l in open(log))


def runs():
    out = []
    for log in sorted(glob.glob(os.path.join(HERE, "*", "epoch_log.csv"))):
        name = os.path.basename(os.path.dirname(log))
        m = re.fullmatch(r"(.+)_s(\d+)", name)
        if m and m.group(1) in ARMS and finished(name):
            out.append((name, m.group(1), int(m.group(2)), os.path.dirname(log)))
    return out


def logged_best(run_dir):
    e = pd.read_csv(os.path.join(run_dir, "epoch_log.csv"))
    r = e.loc[e.val_loss.idxmin()]
    row = {"epoch": int(r.epoch), "epochs_run": len(e), "val_loss": r.val_loss}
    row.update({k: r["val_" + k] for k in KEYS})
    return row


def load_val(device):
    ds = ADNIVolumes(split="val")
    vols = torch.stack([ds[i]["volume"] for i in range(len(ds))])
    labels = ds.df.label.to_numpy()
    vi = pd.read_csv(VISITS_INDEX)
    sub = ds.df.PTID.isin(set(vi[vi.split == "val"].PTID)).to_numpy()
    return vols, labels, sub, ds.df.PTID.to_numpy()


def rescore(ckpt, vols, labels, sub, device):
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    a = blob["args"]
    model = DualHeadNet(arch=a["arch"], dropout=float(a.get("dropout") or 0.0))
    model.load_state_dict(blob["model"], strict=True)
    model.to(device).eval()
    demo, logit = [], []
    with torch.no_grad():
        for s in range(0, len(vols), 16):
            o = model(vols[s:s + 16].to(device))
            demo.append(o["demo"].double().cpu().numpy())
            logit.append(o["alz_logit"].double().cpu().numpy())
    demo, logit = np.concatenate(demo), np.concatenate(logit)
    full = compute_metrics(demo, logit, labels)
    part = compute_metrics(demo[sub], logit[sub], labels[sub])
    _, fitted, _, _ = fit_cutpoints(demo, labels)
    row = {"epoch": int(blob["epoch"])}
    row.update({k: full[k] for k in KEYS})
    row.update({"sub104_" + k: part[k] for k in ("acc4", "macro_f1", "bal_acc", "alz_auc")})
    row["fitted_mf1_ceiling"] = fitted
    return row, demo, logit


def arm_table(df, cols):
    g = df.groupby("arm")[cols]
    mean, sd = g.mean(), g.std(ddof=1)
    out = mean.map(lambda v: f"{v:.3f}").astype(str) + " ± " + sd.map(lambda v: f"{v:.3f}")
    out.insert(0, "n", df.groupby("arm").size())
    return out.reindex([a for a in ARMS if a in out.index])


def main():
    rs = runs()
    if not rs:
        raise SystemExit("no finished runs under analysis/mv/")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pd.set_option("display.width", 200)

    logged = pd.DataFrame([{"run": n, "arm": a, "seed": s, **logged_best(d)} for n, a, s, d in rs])
    v9 = {"run": "v9_lr_3e-5 (Kaggle)", "arm": "v9", "seed": 42, **logged_best(V9)}
    head = ["acc4", "macro_f1", "bal_acc", "alz_auc", "demo_mae"]
    print("LOGGED (epoch_log, best-val-loss epoch, 121 val subjects)\n")
    print(arm_table(logged, head + ["epoch"]).to_string())
    print("\nv9 lr_3e-5 logged: " + "  ".join(f"{k} {v9[k]:.3f}" for k in head)
          + f"  epoch {v9['epoch']}")
    print("\nper-class F1 (mean over seeds):")
    print(logged.groupby("arm")[["f1_CN", "f1_EMCI", "f1_LMCI", "f1_AD"]].mean()
          .reindex([a for a in ARMS if a in set(logged.arm)]).round(3).to_string())

    vols, labels, sub, ptids = load_val(device)
    print(f"\nRE-SCORED best.pt, fp32 ({len(labels)} val subjects; {int(sub.sum())} in the Path A subset)\n")
    rows = []
    os.makedirs(os.path.join(HERE, "preds"), exist_ok=True)
    for n, a, s, d in rs + [("v9_lr_3e-5", "v9", 42, V9)]:
        row, demo, logit = rescore(os.path.join(d, "best.pt"), vols, labels, sub, device)
        rows.append({"run": n, "arm": a, "seed": s, **row})
        pd.DataFrame({"PTID": ptids, "label": labels, "demo_pred": demo, "alz_logit": logit,
                      "in_patha_subset": sub}).to_csv(os.path.join(HERE, "preds", n + ".csv"), index=False)
    res = pd.DataFrame(rows)
    cols = head + ["sub104_acc4", "sub104_macro_f1", "sub104_alz_auc", "fitted_mf1_ceiling"]
    print(arm_table(res[res.arm != "v9"], cols).to_string())
    r9 = res[res.arm == "v9"].iloc[0]
    print("\nv9 lr_3e-5 re-scored: " + "  ".join(f"{k} {r9[k]:.3f}" for k in cols))

    logged.to_csv(os.path.join(HERE, "summary_logged.csv"), index=False)
    res.to_csv(os.path.join(HERE, "summary_rescored.csv"), index=False)
    print("\nwrote summary_logged.csv, summary_rescored.csv, preds/ under analysis/mv/")


if __name__ == "__main__":
    main()
