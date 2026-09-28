"""Robustness diagnostic for the P1 smoke test: same 20 clinical-TRAIN subjects.

Perturbs the SFCN input after its own mean normalisation (x0.8, x1.25,
gamma 1.5, left-right flip) and reports how far predictions move. A model
operating in-domain should be nearly invariant to these. Never reads val/test.
Writes smoke_sensitivity.csv.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.join(ROOT, "..", "..", "third_party", "UKBiobank_deep_pretrain")
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)
import sfcn_smoke as S  # noqa: E402
from dp_model.model_files.sfcn import SFCN  # noqa: E402

m = SFCN()
sd = torch.load(os.path.join(REPO, "brain_age", "run_20190719_00_epoch_best_mae.p"), map_location="cpu")
m.load_state_dict({k.removeprefix("module."): v for k, v in sd.items()}, strict=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m.to(dev).eval()
centers = 42.5 + np.arange(40)

rows = []
for r in S.pick_subjects(0).itertuples():
    x, _ = S.to_sfcn_input(os.path.join(ROOT, "dataset", "mri_prep", "mni", f"{r.PTID}.nii.gz"))
    row = {"PTID": r.PTID, "DX_bl": r.DX_bl, "age_at_scan": r.age_at_scan}
    for name, v in {"base": x, "x0.8": x * 0.8, "x1.25": x * 1.25,
                    "gamma1.5": (x / x.max()) ** 1.5 * x.max(), "flipLR": x[::-1]}.items():
        with torch.no_grad():
            lp = m(torch.from_numpy(np.ascontiguousarray(v))[None, None].to(dev))[0].reshape(-1)
        row[name] = float(np.exp(lp.cpu().numpy()) @ centers)
    rows.append(row)
d = pd.DataFrame(rows)
d.round(2).to_csv(os.path.join(HERE, "smoke_sensitivity.csv"), index=False)
for k in ["x0.8", "x1.25", "gamma1.5", "flipLR"]:
    s = d[k] - d.base
    print(f"{k:9s} mean shift {s.mean():+.2f}  max |shift| {s.abs().max():.2f}  "
          f"corr with base {np.corrcoef(d[k], d.base)[0, 1]:.3f}")
