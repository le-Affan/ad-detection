"""P1 step 1 (LYP/.venv-mri, GPU): LayerCAM and Grad-CAM of the ALZ logit at layer1 / layer2 / layer4 for every
train and val subject with a scan, one map per (subject, model), cached at native resolution.

Models per subject: train -> its fold's 3 out-of-fold models; val -> all 15 (p1_lib.models_of).
Output (gitignored): cache/cams.npz (or cache/dry/cams.npz with --dry): for every method x stage a float16 array
(n_pairs, s, s, s), plus pair_ptid, pair_model, pair_logit (ALZ logit of that model for that subject).
No test subject is processed here (test maps are made by src/card/build_cam.py for the card only).
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import p1_lib as L  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    if a.dry:
        C.set_dry()
    C.verify_manifest()
    torch.manual_seed(0)
    d = L.subjects()
    if a.dry:
        d = d[d.PTID.isin(C.dry_run_ptids())].reset_index(drop=True)
        assert len(d) == 10
    t0 = time.time()
    vols = L.load_volumes(list(d.path))
    print(f"{len(d)} subjects loaded in {time.time() - t0:.0f}s", flush=True)
    store = {(m, s): [] for m in L.METHODS for s in L.STAGES}
    pp, pm, pl = [], [], []
    for mid in range(len(L.MODELS)):
        rows = [i for i, r in enumerate(d.itertuples()) if mid in L.models_of(r)]
        if not rows:
            continue
        model = L.get_model(mid)
        maps, logit = L.cams(model, vols[rows])
        for k, v in maps.items():
            assert np.isfinite(v).all()
            store[k].append(v.astype(np.float16))
        pp += list(d.PTID.iloc[rows])
        pm += [mid] * len(rows)
        pl.append(logit)
        del model
        torch.cuda.empty_cache()
        print(f"model {mid} {L.MODELS[mid]}: {len(rows)} subjects  {time.time() - t0:.0f}s", flush=True)
    out = {f"{m}_{s}": np.concatenate(store[(m, s)]) for m, s in store}
    out.update(pair_ptid=np.array(pp), pair_model=np.array(pm), pair_logit=np.concatenate(pl))
    zero = {k: int((v.reshape(len(v), -1).max(1) <= 0).sum()) for k, v in out.items() if k.startswith(("layercam", "gradcam"))}
    print("all-zero maps per method/stage:", zero)
    path = C.cache_path("cams.npz")
    np.savez(path, **out)
    print(f"saved {path}; {len(pp)} pairs; peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.2f} GB; "
          f"{time.time() - t0:.0f}s")
    C.verify_manifest()


if __name__ == "__main__":
    main()
