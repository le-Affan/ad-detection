"""Freeze, part 2 (run with LYP/.venv-kan): the final KAN fusion layer on all 691 train+val subjects.

Config 32 (K2 residual, G 5, mu 1e-3, scaled inputs, extras AGE / CDRSB /
CogIndex / MMSE, class-balanced CE + 0.5 x squared-EMD ordinal term,
modality dropout 15%). For every outer seed s (42-46) it trains KAN seeds
42-46 on the 691 OOF stream scores of frozen/streams_final.npz (seed-s folds).
The final prediction is the mean over these 25 KANs, each applied to the
seed-s specialists' scores.
Each KAN's state_dict and fitted Inputs (scalers, temperatures, neutral
logits) are saved. Reloading must reproduce the in-memory predictions.
Never reads test.
"""
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

FROZEN = os.path.join(HERE, "frozen")
CID = 32


def main():
    torch.set_num_threads(4)
    cfg = {**KF.DEFAULT, **json.loads(str(np.load(os.path.join(HERE, "kan_runs", f"{CID}.npz"))["cfg"]))}
    z = np.load(os.path.join(FROZEN, "streams_final.npz"))
    y, has, ex = z["y"], z["has"], z["extras"]
    json.dump(cfg, open(os.path.join(FROZEN, "kan_config.json"), "w"), indent=1)
    probs = []
    for s in M.SEEDS:
        tr = {"pc": z[f"clin_oof_{s}"], "pm": z[f"mri_oof_{s}"], "has": has, "y": y, "extras": ex}
        for ks in KF.KSEEDS:
            p, _, diag, (model, inp) = KF.fit_predict(cfg, tr, tr, ks)
            torch.save(model.state_dict(), os.path.join(FROZEN, f"kan_{s}_{ks}.pt"))
            pickle.dump(inp, open(os.path.join(FROZEN, f"kan_inputs_{s}_{ks}.pkl"), "wb"))
            m2 = KF.build(cfg, 9 + len(cfg["extras"]), ks)
            m2.load_state_dict(torch.load(os.path.join(FROZEN, f"kan_{s}_{ks}.pt")))
            inp2 = pickle.load(open(os.path.join(FROZEN, f"kan_inputs_{s}_{ks}.pkl"), "rb"))
            with torch.no_grad():
                o = m2(inp2(tr["pc"], tr["pm"], tr["has"], tr["extras"]))
                p2 = torch.softmax(KF.cog_logits(cfg, o, inp2.base(tr["pc"])), 1).numpy()
            assert np.abs(p - p2).max() < 1e-5, "reload mismatch"
            assert not diag["failed"]
            probs.append(p)
        print(f"outer seed {s}: 5 KANs saved and reload-checked", flush=True)
    P = np.mean(probs, 0)
    print("in-sample (691, not an estimate):", {k: round(v, 3) for k, v in M.metrics(y, P).items()})


if __name__ == "__main__":
    main()
