"""Frozen system, step 2 (run with LYP/.venv-kan): the frozen KAN fusion on a split's stream scores.

Final COG / ADD probabilities = mean over the 25 frozen KANs (outer seed s x
KAN seed); KAN (s, k) takes the seed-s specialists' scores.
--split test only after freeze.md is committed and unmodified.
Output: frozen_run/<split>_kan.npz.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kan_fusion as KF  # noqa: E402
import metrics as M  # noqa: E402
from freeze_guard import assert_frozen  # noqa: E402

FROZEN = os.path.join(HERE, "frozen")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["val", "test"])
    a = ap.parse_args()
    if a.split == "test":
        print("freeze commit:", assert_frozen())
    cfg = json.load(open(os.path.join(FROZEN, "kan_config.json")))
    z = np.load(os.path.join(HERE, "frozen_run", f"{a.split}_streams.npz"))
    P, A = [], []
    for s in M.SEEDS:
        for ks in KF.KSEEDS:
            m = KF.build(cfg, 9 + len(cfg["extras"]), ks)
            m.load_state_dict(torch.load(os.path.join(FROZEN, f"kan_{s}_{ks}.pt")))
            inp = pickle.load(open(os.path.join(FROZEN, f"kan_inputs_{s}_{ks}.pkl"), "rb"))
            with torch.no_grad():
                o = m(inp(z[f"clin_{s}"], z[f"mri_{s}"], z["has"], z["extras"]))
                P.append(torch.softmax(KF.cog_logits(cfg, o, inp.base(z[f"clin_{s}"])), 1).numpy())
                A.append(torch.sigmoid(o[:, 4]).numpy())
    np.savez(os.path.join(HERE, "frozen_run", f"{a.split}_kan.npz"), p=np.mean(P, 0), add=np.mean(A, 0),
             p_each=np.array(P))
    print(f"{a.split}: KAN fusion applied ({len(P)} KANs)")


if __name__ == "__main__":
    main()
