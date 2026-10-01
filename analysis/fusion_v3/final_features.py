"""Feature construction for the frozen system (pandas only; any env).

fit_stats(rids): every imputation/normalisation statistic, fitted on the given
  subjects (the final training set, all 691 train+val subjects):
  - the step-4 rules of src/step4_processing.py on the 35 source features
    (median/mode imputation, z-scoring; CogIndex = z MMSE - z CDRSB - z ADAS13);
  - FreeSurfer volume / ICV ratios: median imputation, z-scoring.
build(rids, stats, source): the model-ready frame for those subjects.
  source "fit": scan inputs from the train/val tables (cross-fitted train ALZ +
    val fold-ensemble ALZ, brainage_trainval.csv, kan_fusion reliability flags);
  source "val" / "test": scan inputs from scan_inputs_<split>.csv and
    brainage_<split>.csv (written by scan_inputs.py / brainage_split.py).
Every table is filtered to the requested RIDs as it is read.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))
import step4_processing as S4  # noqa: E402

PROCESSED = os.path.join(ROOT, "dataset", "processed")
VOLUMES = ["Ventricles", "Hippocampus", "WholeBrain", "Entorhinal", "Fusiform", "MidTemp"]
RATIOS = [f"{v}_ICV" for v in VOLUMES]
NAMES = ["CN", "EMCI", "LMCI", "AD"]


def raw_rows(rids):
    f = pd.read_csv(os.path.join(PROCESSED, "step2_features.csv"), low_memory=False)
    return f[f.RID.isin(set(rids))].sort_values("RID").reset_index(drop=True)


def fit_stats(rids):
    raw = raw_rows(rids)
    feats = [c for c in raw.columns if c not in S4.ID_COLUMNS]
    cat = [c for c in S4.CATEGORICAL_FEATURES if c in feats]
    num = [c for c in feats if c not in cat]
    med, mode = S4.fit_imputation(raw, num, cat)
    imp = S4.apply_imputation(raw, med, mode)
    mean, std = S4.fit_normalization(imp, num)
    r = pd.DataFrame({n: raw[v] / raw["ICV"] for v, n in zip(VOLUMES, RATIOS)})
    rmed = {c: float(r[c].median()) for c in RATIOS}
    r = r.fillna(rmed)
    return {"n_fit": len(raw), "num": num, "cat": cat, "median": med, "mode": mode, "mean": mean, "std": std,
            "ratio_median": rmed, "ratio_mean": {c: float(r[c].mean()) for c in RATIOS},
            "ratio_std": {c: float(r[c].std()) for c in RATIOS}}


def scan_tables(rids, source):
    rids = set(rids)
    if source == "fit":
        cf = pd.read_csv(os.path.join(ROOT, "analysis", "fusion_feasibility", "alz_crossfit.csv"))
        cf = cf[cf.RID.isin(rids)][["RID", "PTID", "ALZ"]]
        ba = pd.read_csv(os.path.join(ROOT, "analysis", "brain_age", "brainage_trainval.csv"))
        ba = ba[ba.RID.isin(rids)][["RID", "pred_age", "age_at_scan"]]
        rel = pd.read_csv(os.path.join(ROOT, "analysis", "kan_fusion", "brainage_reliability.csv"))
        rel = rel[rel.RID.isin(rids)][["RID", "unreliable"]]
        return cf.merge(ba, on="RID", how="left").merge(rel, on="RID", how="left")
    si = pd.read_csv(os.path.join(HERE, f"scan_inputs_{source}.csv"))
    ba = pd.read_csv(os.path.join(HERE, f"brainage_{source}.csv"))
    si, ba = si[si.RID.isin(rids)], ba[ba.RID.isin(rids)]
    out = si[["RID", "PTID", "ALZ", "unreliable"]].merge(ba[["RID", "pred_age", "age_at_scan"]], on="RID", how="left")
    if out.PTID.isna().any():                                   # subjects without a scan: PTID from brain-age table
        out["PTID"] = out.PTID.fillna(out.RID.map(ba.set_index("RID").PTID))
    return out


def build(rids, stats, source):
    raw = raw_rows(rids)
    imp = S4.apply_imputation(raw, stats["median"], stats["mode"])
    df = S4.add_cogindex(S4.apply_normalization(imp, stats["mean"], stats["std"]))
    for v, n in zip(VOLUMES, RATIOS):
        r = (raw[v] / raw["ICV"]).fillna(stats["ratio_median"][n])
        df[n] = (r - stats["ratio_mean"][n]) / stats["ratio_std"][n]
    sc = scan_tables(rids, source)
    df = df.merge(sc, on="RID", how="left")
    df["has_mri"] = df.ALZ.notna()
    df["ba_ok"] = df.pred_age.notna() & (df.unreliable != True)              # noqa: E712
    df["SEX"] = (df.PTGENDER == "Male").astype(float)
    df["diagnosis"] = df.label.map(dict(enumerate(NAMES)))
    assert len(df) == len(set(rids)) and df.RID.is_unique
    return df


def save_stats(stats, path):
    json.dump(stats, open(path, "w"), indent=1)


def load_stats(path):
    return json.load(open(path))


if __name__ == "__main__":
    tv = list(pd.read_csv(os.path.join(PROCESSED, "train_rids.csv")).RID) + \
        list(pd.read_csv(os.path.join(PROCESSED, "val_rids.csv")).RID)
    st = fit_stats(tv)
    d = build(tv, st, "fit")
    print(len(d), d.has_mri.sum(), d.ba_ok.sum(), d[RATIOS + ["ALZ", "AGE", "CogIndex"]].describe().round(2))
    print(np.isnan(d[RATIOS].to_numpy()).sum())
