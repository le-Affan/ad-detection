"""Model-ready frames for train and val, exactly as the card builder makes them (LYP/.venv-fusion).

final_features.build(train + val RIDs, frozen/feature_stats.json, source "fit"): the tables the frozen system
was fitted on. Raw (unimputed) values come from final_features.raw_rows. Never loads a test subject.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

sys.path.insert(0, C.V3)
import final_features as FF  # noqa: E402


def trainval():
    stats = FF.load_stats(os.path.join(C.FROZEN, "feature_stats.json"))
    tr, va = C.rids("train"), C.rids("val")
    d = FF.build(tr + va, stats, "fit")
    sp = {r: "train" for r in tr} | {r: "val" for r in va}
    d["split"] = d.RID.map(sp)
    d = d.sort_values("PTID").reset_index(drop=True)
    C.assert_no_test(d.RID)
    raw = FF.raw_rows(d.RID).set_index("RID").loc[d.RID].reset_index()
    for v in FF.VOLUMES:
        d[f"raw_{v}_ICV"] = (raw[v] / raw["ICV"]).to_numpy()
    for c in ("CDRSB", "MMSE"):
        d[f"raw_{c}"] = raw[c].to_numpy()
    return d


def cn_train_median(d, col):
    x = d.loc[(d.split == "train") & (d.label == 0), col].dropna()
    return float(x.median())


def expected_sign(values, median, higher_is_ad):
    """+1 where the subject's raw value points toward AD, -1 away, 0 at the median or missing."""
    v = np.asarray(values, float)
    s = np.sign(v - median) * (1 if higher_is_ad else -1)
    s[np.isnan(v)] = 0
    return s


def columns():
    return json.load(open(os.path.join(C.FROZEN, "columns.json")))
