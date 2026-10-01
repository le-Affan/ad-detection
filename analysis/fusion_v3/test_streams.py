"""Frozen system, step 1 (run with LYP/.venv-fusion): features and stream scores for one split.

--split test only after freeze.md is committed and unmodified. --split val is the
mechanical dry run: val is part of the final training data, so its numbers are
not estimates.
Applies frozen/feature_stats.json (fitted on the 691) to the split's raw rows
and scan inputs (scan_inputs_<split>.csv, brainage_<split>.csv), then, for each
seed, the frozen clinical specialist, MRI specialist and reference CatBoost F / R.
Output: frozen_run/<split>_streams.npz.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import final_features as FF  # noqa: E402
import metrics as M  # noqa: E402
from freeze_guard import assert_frozen  # noqa: E402

FROZEN = os.path.join(HERE, "frozen")


def cb(path):
    m = CatBoostClassifier()
    m.load_model(path)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["val", "test"])
    a = ap.parse_args()
    if a.split == "test":
        print("freeze commit:", assert_frozen())
    rids = list(pd.read_csv(os.path.join(FF.PROCESSED, f"{a.split}_rids.csv")).RID)
    d = FF.build(rids, FF.load_stats(os.path.join(FROZEN, "feature_stats.json")), a.split)
    cols = json.load(open(os.path.join(FROZEN, "columns.json")))
    out = {"rids": d.RID.to_numpy(), "y": d.label.to_numpy(), "has": d.has_mri.to_numpy(),
           "ba_ok": d.ba_ok.to_numpy(), "extras": d[cols["extras"]].to_numpy(dtype=float),
           "cdrsb": d.CDRSB.to_numpy(dtype=float)}
    for s in M.SEEDS:
        out[f"clin_{s}"] = cb(os.path.join(FROZEN, f"clinical_{s}.cbm")).predict_proba(d[cols["clinical_top12"][str(s)]])
        out[f"mri_{s}"] = pickle.load(open(os.path.join(FROZEN, f"mri_{s}.pkl"), "rb")).predict_proba(d)
        out[f"cbF_{s}"] = cb(os.path.join(FROZEN, f"catboost_F_{s}.cbm")).predict_proba(d[cols["catboost_F_cols"]])
        out[f"cbR_{s}"] = cb(os.path.join(FROZEN, f"catboost_R_{s}.cbm")).predict_proba(d[cols["catboost_R_cols"]])
    os.makedirs(os.path.join(HERE, "frozen_run"), exist_ok=True)
    np.savez(os.path.join(HERE, "frozen_run", f"{a.split}_streams.npz"), **out)
    print(f"{a.split}: {len(d)} subjects, {int(d.has_mri.sum())} with a scan, {int(d.ba_ok.sum())} reliable brain age; "
          f"label counts {np.bincount(d.label, minlength=4).tolist()}")


if __name__ == "__main__":
    main()
