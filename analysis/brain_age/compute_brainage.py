"""P2 step 1: DeepBrainNet brain age for clinical TRAIN and VAL subjects only.

Test subjects are never loaded: the subject list is built from train_rids.csv
and val_rids.csv, and every scan path is checked against the test RIDs.
Run with TF_USE_LEGACY_KERAS=1 in LYP/.venv-deepbrainnet.

Pipeline = dbn_smoke.py, unchanged: world-coordinate resample of the m1 mni/
volume onto the 182x218x182 MNI 1 mm grid, then the repo's own steps
(185/p97, axial slices 45-124, JPEG, 256x256 nearest, /255, median of the
80 slice predictions). Weights: DBN_model.h5, sha256 9257e98e...41ea (checked
at start).

Chronological age at scan = ADSL AGE (age at enrolment, 0.1 y resolution)
+ (MRI_EXAMDATE - ADSL ENRLDT) / 365.25. MRI_EXAMDATE is from
dataset/mri_manifest/mri_manifest.csv.

No brain age (NaN, never imputed): subjects without an MRI scan, and
027_S_4966 (m1 registration failed, corr_vs_mni 0.054).

Resumable: finished subjects are appended to brainage_trainval.partial.csv
and skipped on rerun; failures go to brainage_failures.csv and are re-tried
on the next run, never dropped. The final CSV is written only when every
scan has a result or a logged failure.
"""
import hashlib
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dbn_smoke as D  # noqa: E402

ROOT = D.ROOT
PROCESSED = os.path.join(ROOT, "dataset", "processed")
SHA256 = "9257e98e9aff88bfcd83acdb1e74ffde762967503ef6df16c9270aa39d9941ea"
EXCLUDE = {"027_S_4966": "m1 registration failed (corr_vs_mni 0.054)"}
PARTIAL = os.path.join(HERE, "brainage_trainval.partial.csv")
FAILS = os.path.join(HERE, "brainage_failures.csv")
OUT = os.path.join(HERE, "brainage_trainval.csv")


def subjects():
    split_of = {int(r): s for s in ("train", "val", "test")
                for r in pd.read_csv(os.path.join(PROCESSED, f"{s}_rids.csv")).RID}
    feats = pd.read_csv(os.path.join(PROCESSED, "step2_features.csv"), usecols=["RID", "label"])
    df = feats[feats.RID.map(split_of).isin(["train", "val"])].copy()
    df["split"] = df.RID.map(split_of)
    man = pd.read_csv(os.path.join(ROOT, "dataset", "mri_manifest", "mri_manifest.csv"))
    adsl = pd.read_csv(os.path.join(ROOT, "dataset", "raw", "adnimerge2_tables", "ADSL.csv"),
                       usecols=["SUBJID", "AGE", "ENRLDT"], low_memory=False)
    adsl["RID"] = adsl.SUBJID.astype(str).str.split("_S_").str[-1].astype(int)
    df = df.merge(man[["RID", "PTID", "DX_bl", "MRI_EXAMDATE"]], on="RID").merge(
        adsl[["RID", "AGE", "ENRLDT"]], on="RID", how="left")
    df["age_at_scan"] = df.AGE + (pd.to_datetime(df.MRI_EXAMDATE) - pd.to_datetime(df.ENRLDT)).dt.days / 365.25
    mi = pd.read_csv(os.path.join(ROOT, "dataset", "mri_prep", "mri_index.csv"))
    df["has_scan"] = df.PTID.isin(mi.PTID)
    assert not set(df.RID) & {r for r, s in split_of.items() if s == "test"}
    assert df[df.has_scan].age_at_scan.notna().all()   # no-scan subjects: no scan date, age NaN
    return df


def main():
    with open(os.path.join(D.DBN, "Models", "DBN_model.h5"), "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == SHA256, "weight file hash mismatch"
    df = subjects()
    done = pd.read_csv(PARTIAL) if os.path.exists(PARTIAL) else pd.DataFrame(columns=["PTID", "pred_age"])
    todo = df[df.has_scan & ~df.PTID.isin(EXCLUDE) & ~df.PTID.isin(done.PTID)]
    print(f"train+val subjects {len(df)}; with scan {int(df.has_scan.sum())}; excluded {len(EXCLUDE)}; "
          f"done {len(done)}; to run {len(todo)}", flush=True)

    model = D.load_model() if len(todo) else None
    fails = []
    for i, r in enumerate(todo.itertuples(), 1):
        t0 = time.time()
        try:
            vol = D.to_dbn_grid(os.path.join(ROOT, "dataset", "mri_prep", "mni", f"{r.PTID}.nii.gz"))
            pred, per_slice = D.predict(model, vol)
            row = pd.DataFrame([{"PTID": r.PTID, "pred_age": round(pred, 3),
                                 "slice_pred_iqr": round(float(np.subtract(*np.percentile(per_slice, [75, 25]))), 3),
                                 "seconds": round(time.time() - t0, 1)}])
            row.to_csv(PARTIAL, mode="a", header=not os.path.exists(PARTIAL), index=False)
        except Exception as e:  # logged, retried next run, never skipped silently
            fails.append({"PTID": r.PTID, "error": repr(e), "trace": traceback.format_exc()[-500:]})
            print(f"FAIL {r.PTID}: {e!r}", flush=True)
        if i % 25 == 0:
            print(f"{i}/{len(todo)}  {time.strftime('%H:%M:%S')}", flush=True)
    pd.DataFrame(fails, columns=["PTID", "error", "trace"]).to_csv(FAILS, index=False)

    done = pd.read_csv(PARTIAL)
    missing = set(df[df.has_scan & ~df.PTID.isin(EXCLUDE)].PTID) - set(done.PTID)
    if missing:
        print(f"{len(missing)} scans still without a result (see {FAILS}); final CSV not written")
        return
    out = df.merge(done[["PTID", "pred_age", "slice_pred_iqr"]], on="PTID", how="left")
    out["gap_raw"] = out.pred_age - out.age_at_scan
    out["has_brainage"] = out.pred_age.notna()
    out["missing_reason"] = np.where(out.has_brainage, "",
                                     np.where(~out.has_scan, "no MRI scan", out.PTID.map(EXCLUDE).fillna("")))
    cols = ["RID", "PTID", "split", "DX_bl", "label", "age_at_scan", "pred_age", "gap_raw",
            "has_brainage", "missing_reason", "slice_pred_iqr", "AGE", "ENRLDT", "MRI_EXAMDATE"]
    out[cols].sort_values(["split", "PTID"]).to_csv(OUT, index=False)
    print(f"wrote {OUT}: {int(out.has_brainage.sum())} with brain age, "
          f"{int((~out.has_brainage).sum())} NaN ({out[~out.has_brainage].missing_reason.value_counts().to_dict()})")


if __name__ == "__main__":
    main()
