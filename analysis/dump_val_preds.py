"""Dump raw val-split predictions for every downloaded checkpoint.

ANALYSIS ONLY. Nothing here trains, tunes or writes into src/mri. The model is
constructed from the `arch` recorded inside each checkpoint's own `args` blob,
never from a default, and the parameter count is asserted against the tensors
actually present in the state dict before a single forward pass runs.

AUGMENTATION IS STRUCTURALLY ABSENT. m3_train.py applies augment_batch inside
run_epoch under `if train`; this script does not import or call it at all, and
runs under model.eval() + torch.no_grad().

The 121 val volumes are read once into RAM (~428 MB fp32) and reused across all
checkpoints -- loading dominates the cost otherwise.
"""
import csv
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "mri"))
from dataset import ADNIVolumes, DEMO_BY_LABEL  # noqa: E402
from model import ARCHS, DualHeadNet  # noqa: E402

CKPT_ROOT = os.path.join(ROOT, "analysis", "kaggle_output")
OUT_DIR = os.path.join(ROOT, "analysis", "threshold_refit")


def find_checkpoints():
    out = []
    for dirpath, _d, files in os.walk(CKPT_ROOT):
        for f in sorted(files):
            if f.endswith(".pt"):
                out.append(os.path.join(dirpath, f))
    return sorted(out)


def tag_for(path):
    rel = os.path.relpath(path, CKPT_ROOT).replace("\\", "/")
    return rel[:-3].replace("/", "__")


def main():
    torch.set_num_threads(os.cpu_count() or 4)
    ds = ADNIVolumes(split="val")
    n = len(ds)
    print(f"val subjects: {n}")
    if n != 121:
        raise SystemExit(f"expected 121 val subjects, got {n}")

    print("loading volumes into RAM ...", flush=True)
    vols = torch.empty(n, 1, 96, 96, 96, dtype=torch.float32)
    rids, labels = [], []
    for i in range(n):
        r = ds[i]
        vols[i] = r["volume"]
        rids.append(r["rid"])
        labels.append(int(r["label"]))
    labels = np.asarray(labels)
    print(f"loaded {n} volumes, {vols.numel() * 4 / 2**20:.0f} MiB")

    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = []
    for path in find_checkpoints():
        tag = tag_for(path)
        blob = torch.load(path, map_location="cpu", weights_only=False)
        args = blob.get("args", {})
        sd = blob["model"]

        # v7 predates the --arch flag, so its checkpoint records no arch at all.
        # Rather than defaulting (which would silently build the wrong trunk),
        # identify it from the tensors present: the arch whose state_dict has
        # identical keys AND identical shapes is the one that was trained.
        arch = args.get("arch")
        dropout = float(args.get("dropout", 0.0) or 0.0)
        if arch is None:
            cands = []
            for name in sorted(ARCHS):
                ref = DualHeadNet(arch=name).state_dict()
                if (set(ref) == set(sd)
                        and all(ref[k].shape == sd[k].shape for k in ref)):
                    cands.append(name)
            if len(cands) != 1:
                raise SystemExit(f"{tag}: cannot identify arch, matches={cands}")
            arch = cands[0]
            arch_src = "inferred from state_dict"
        else:
            arch_src = "recorded in checkpoint args"

        model = DualHeadNet(arch=arch, dropout=dropout)
        # strict=True is the real check: every key and every shape must line up.
        # A raw element count over the state dict would NOT be that check -- it
        # also sweeps up BatchNorm running_mean/running_var, which are buffers,
        # not parameters (9,600 of them for resnet18, 4,800 for resnet18w).
        model.load_state_dict(sd, strict=True)
        n_model = sum(p.numel() for p in model.parameters())
        recorded = args.get("params")
        if recorded is not None and int(recorded) != n_model:
            raise SystemExit(f"{tag}: params {n_model} != recorded {recorded}")
        model.eval()

        demo = np.empty(n, dtype=np.float64)
        logit = np.empty(n, dtype=np.float64)
        with torch.no_grad():
            for s in range(0, n, 4):
                o = model(vols[s:s + 4])
                demo[s:s + 4] = o["demo"].double().numpy()
                logit[s:s + 4] = o["alz_logit"].double().numpy()
        prob = 1.0 / (1.0 + np.exp(-logit))

        csv_path = os.path.join(OUT_DIR, tag + ".csv")
        with open(csv_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["rid", "true_class", "demo_target",
                        "demo_pred", "alz_logit", "alz_prob"])
            for i in range(n):
                w.writerow([rids[i], labels[i], DEMO_BY_LABEL[labels[i]],
                            f"{demo[i]:.8f}", f"{logit[i]:.8f}", f"{prob[i]:.8f}"])
        manifest.append({"tag": tag, "ckpt": os.path.relpath(path, ROOT),
                         "arch": arch, "arch_source": arch_src,
                         "dropout": dropout, "params": n_model,
                         "epoch": blob.get("epoch"),
                         "best_val_loss": blob.get("best_val_loss"),
                         "rows": n, "csv": os.path.relpath(csv_path, ROOT)})
        print(f"{tag:34s} arch={arch:10s} ({arch_src:28s}) "
              f"params={n_model:,} epoch={blob.get('epoch')} rows={n}", flush=True)
        del blob, sd, model

    with open(os.path.join(OUT_DIR, "_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nwrote {len(manifest)} CSVs to {OUT_DIR}")


if __name__ == "__main__":
    main()
