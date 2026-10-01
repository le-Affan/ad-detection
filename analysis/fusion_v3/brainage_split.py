"""DeepBrainNet brain age for one split (run with TF_USE_LEGACY_KERAS=1 in LYP/.venv-deepbrainnet).

Pipeline, weights (sha256 checked) and age-at-scan definition are those of
analysis/brain_age/compute_brainage.py, unchanged.
Exclusion rule (written down for the test split): a scan whose m1 registration
check corr_vs_mni < 0.15 gets no brain age. On train/val this reproduces the
original exclusion exactly: 027_S_4966 at 0.054 is the only scan below 0.15;
the next lowest is 0.207 and has a valid brain age.
--split val --limit N   reproducibility check against brainage_trainval.csv
--split test            only after freeze.md is committed and unmodified.
Output: brainage_<split>.csv (RID, PTID, age_at_scan, pred_age, excluded).
"""
import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "analysis", "brain_age"))
sys.path.insert(0, HERE)
import compute_brainage as CB  # noqa: E402
import dbn_smoke as D  # noqa: E402
from freeze_guard import assert_frozen  # noqa: E402


def subjects(split):
    rids = set(pd.read_csv(os.path.join(CB.PROCESSED, f"{split}_rids.csv")).RID)
    man = pd.read_csv(os.path.join(ROOT, "dataset", "mri_manifest", "mri_manifest.csv"))
    adsl = pd.read_csv(os.path.join(ROOT, "dataset", "raw", "adnimerge2_tables", "ADSL.csv"),
                       usecols=["SUBJID", "AGE", "ENRLDT"], low_memory=False)
    adsl["RID"] = adsl.SUBJID.astype(str).str.split("_S_").str[-1].astype(int)
    df = pd.DataFrame({"RID": sorted(rids)}).merge(man[["RID", "PTID", "MRI_EXAMDATE"]], on="RID", how="left") \
        .merge(adsl[["RID", "AGE", "ENRLDT"]], on="RID", how="left")
    df["age_at_scan"] = df.AGE + (pd.to_datetime(df.MRI_EXAMDATE) - pd.to_datetime(df.ENRLDT)).dt.days / 365.25
    mi = pd.read_csv(os.path.join(ROOT, "dataset", "mri_prep", "mri_index.csv"))
    m1 = pd.read_csv(os.path.join(ROOT, "dataset", "mri_prep", "m1_report_full.csv"), usecols=["PTID", "corr_vs_mni"])
    df = df.merge(m1, on="PTID", how="left")
    df["has_scan"] = df.PTID.isin(mi.PTID)
    df["excluded"] = df.has_scan & (df.corr_vs_mni < 0.15)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.split == "test":
        print("freeze commit:", assert_frozen())
    with open(os.path.join(D.DBN, "Models", "DBN_model.h5"), "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == CB.SHA256
    df = subjects(a.split)
    todo = df[df.has_scan & ~df.excluded]
    if a.limit:
        todo = todo.head(a.limit)
    model = D.load_model()
    preds = {}
    for i, r in enumerate(todo.itertuples(), 1):
        vol = D.to_dbn_grid(os.path.join(ROOT, "dataset", "mri_prep", "mni", f"{r.PTID}.nii.gz"))
        preds[r.PTID] = D.predict(model, vol)[0]
        if i % 20 == 0:
            print(f"{i}/{len(todo)}", flush=True)
    df["pred_age"] = df.PTID.map(preds)
    out = df if not a.limit else df[df.PTID.isin(preds)]
    out[["RID", "PTID", "age_at_scan", "pred_age", "has_scan", "excluded", "corr_vs_mni"]].to_csv(
        os.path.join(HERE, f"brainage_{a.split}.csv"), index=False)
    print(f"{a.split}: {len(df)} subjects, {int(df.has_scan.sum())} scans, {int(df.excluded.sum())} excluded, "
          f"{len(preds)} predicted")
    if a.split == "val":
        ref = pd.read_csv(os.path.join(ROOT, "analysis", "brain_age", "brainage_trainval.csv")).set_index("PTID")
        o = out.set_index("PTID")
        print(f"val check: max |pred_age diff| {np.abs(o.pred_age - ref.pred_age.reindex(o.index)).max():.4f}, "
              f"max |age_at_scan diff| {np.abs(o.age_at_scan - ref.age_at_scan.reindex(o.index)).max():.2e}")


if __name__ == "__main__":
    main()
