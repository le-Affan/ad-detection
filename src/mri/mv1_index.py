"""Build the multi-visit TRAIN index for the static model: one row per scan.

The static model (m3_train.py) sees one baseline scan per subject. Path A
preprocessed the m06/m12/m24 follow-ups of 690 of those subjects; this module
turns the train split's follow-ups into extra, separately labelled examples for
the SAME static model. It is not a sequence model -- every row is an
independent (scan, label) pair -- and m3_train.py consumes the output through
`--train-index`, leaving val and test exactly as they were.

TRAIN ROWS ONLY, BY CONSTRUCTION. Val and test follow-ups are never written
here, so nothing the train loader reads can contain them. The split of every
row is re-derived from `m2_index.load_splits()` by RID -- the one split reader
the MRI stream has -- and both input indexes' own `split` columns are asserted
to agree with it rather than trusted.

WHAT A FOLLOW-UP'S LABEL IS. ADNI has no per-visit CN/EMCI/LMCI/AD label:
DXSUM's per-visit DIAGNOSIS is CN / MCI / Dementia, and EMCI vs LMCI exists
only as the enrollment arm. So a follow-up carries its subject's DX_bl, and
under the default `consistent` policy it is kept only when that visit's own
DXSUM diagnosis still agrees with DX_bl at the three-class level:

    DX_bl   kept when the visit's DIAGNOSIS is
    CN      CN
    EMCI    MCI
    LMCI    MCI
    AD      Dementia

A converter's post-conversion scans, and a reverter's, are therefore dropped,
not relabelled, and written to `train_scans_excluded.csv` with the reason. The
`dx_bl` policy keeps every follow-up under DX_bl regardless; it exists for
ablation only.

The DXSUM row is matched by the scan's own VISCODE2, as p4_temporal_c1's
dx_last_labels does, so an m06 slot filled from an m03 visit takes the m03
diagnosis.

THE PATH A `bl` SLOT IS NOT USED. It is the same session as the static
baseline scan (median r = 0.987 against it; IDA served a different derivative
file). The static file is the one kept, so the 567 baseline rows here are
exactly v9's train set, byte for byte the same paths.

Outputs, under dataset/mri_prep_mv/:

    train_scans.csv           the index m3_train.py reads
    train_scans_excluded.csv  follow-ups left out, and why
    train_scans_summary.json  counts, for the record and for the tests
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m2_index import load_splits  # noqa: E402  -- the one split reader

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MRI_INDEX = os.path.join(ROOT, "dataset", "mri_prep", "mri_index.csv")
VISITS_INDEX = os.path.join(ROOT, "dataset", "mri_prep_patha", "visits_index.csv")
DXSUM = os.path.join(ROOT, "dataset", "raw", "adnimerge2_tables", "DXSUM.csv")
OUT_DIR = os.path.join(ROOT, "dataset", "mri_prep_mv")

CLASSES = ["CN", "EMCI", "LMCI", "AD"]
# DX_bl -> the DXSUM per-visit diagnosis it is consistent with
COLLAPSE = {"CN": "CN", "EMCI": "MCI", "LMCI": "MCI", "AD": "Dementia"}
FOLLOW_UPS = ("m06", "m12", "m24")
POLICIES = ("consistent", "dx_bl")
COLUMNS = ["RID", "PTID", "scan_key", "slot", "viscode2", "DX_bl", "label",
           "visit_dx", "split", "vol_path", "source"]


def load_visit_dx(path=DXSUM):
    """(PTID, VISCODE2) -> DXSUM DIAGNOSIS, refusing conflicting duplicates."""
    dx = pd.read_csv(path, dtype=str, usecols=["PTID", "VISCODE2", "DIAGNOSIS"]).dropna()
    if dx.groupby(["PTID", "VISCODE2"]).DIAGNOSIS.nunique().gt(1).any():
        raise ValueError("DXSUM has conflicting diagnoses for one (PTID, VISCODE2)")
    return dx.drop_duplicates(["PTID", "VISCODE2"]).set_index(["PTID", "VISCODE2"]).DIAGNOSIS


def check_inputs(mi, vi, split_of):
    """Both indexes must agree with load_splits(), and with each other."""
    for name, df in (("mri_index", mi), ("visits_index", vi)):
        derived = df.RID.map(split_of)
        if derived.isna().any():
            raise ValueError(f"{name}: {int(derived.isna().sum())} rows have no split assignment")
        if (derived != df.split).any():
            bad = df[derived != df.split].PTID.unique()
            raise ValueError(f"{name}: split column disagrees with load_splits() for {list(bad[:5])}")

    pairs = pd.concat([mi[["PTID", "RID"]], vi[["PTID", "RID"]]]).drop_duplicates()
    if pairs.PTID.duplicated().any() or pairs.RID.duplicated().any():
        raise ValueError("PTID <-> RID is not one-to-one across the two indexes")

    missing = set(vi.PTID) - set(mi.PTID)
    if missing:
        raise ValueError(f"{len(missing)} Path A subjects are absent from mri_index: {sorted(missing)[:5]}")
    lab = mi.set_index("PTID")[["DX_bl", "label"]]
    v = vi.drop_duplicates("PTID").set_index("PTID")[["DX_bl", "label"]]
    if not v.equals(lab.loc[v.index]):
        raise ValueError("visits_index and mri_index disagree on DX_bl/label for some subject")


def build(policy="consistent", out_dir=OUT_DIR):
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}")
    split_of = load_splits()
    mi = pd.read_csv(MRI_INDEX)
    vi = pd.read_csv(VISITS_INDEX)
    check_inputs(mi, vi, split_of)
    visit_dx = load_visit_dx()

    # --- baseline rows: v9's train set, unchanged ---------------------------
    bl = mi[mi.RID.map(split_of) == "train"].copy()
    bl["scan_key"] = bl.PTID
    bl["slot"] = "bl"
    bl["viscode2"] = "bl"
    bl["visit_dx"] = [visit_dx.get((p, "bl")) for p in bl.PTID]
    bl["source"] = "mri_prep"
    bl["vol_path"] = bl.vol_path.str.replace("\\", "/", regex=False)

    # --- follow-up rows ---------------------------------------------------
    fu = vi[(vi.RID.map(split_of) == "train") & vi.present.astype(bool)
            & vi.slot.isin(FOLLOW_UPS)].copy()
    fu["scan_key"] = fu.prep_key
    fu["visit_dx"] = [visit_dx.get((p, c)) for p, c in zip(fu.PTID, fu.viscode2)]
    fu["source"] = "mri_prep_patha"
    fu["vol_path"] = fu.vol_path.str.replace("\\", "/", regex=False)
    fu["reason"] = None
    fu.loc[fu.visit_dx.isna(), "reason"] = "no_visit_dx"
    disagree = fu.visit_dx.notna() & (fu.visit_dx != fu.DX_bl.map(COLLAPSE))
    fu.loc[disagree, "reason"] = "visit_dx_disagrees"

    if policy == "consistent":
        excluded = fu[fu.reason.notna()]
        fu = fu[fu.reason.isna()]
    else:
        excluded = fu.iloc[0:0]

    df = pd.concat([bl[COLUMNS], fu[COLUMNS]], ignore_index=True)
    df["split"] = df.RID.map(split_of)
    df = df.sort_values(["PTID", "slot"]).reset_index(drop=True)

    # --- the guarantees this file exists to give ----------------------------
    if (df.split != "train").any():
        raise AssertionError("non-train row in the train index")
    held_out = mi[mi.RID.map(split_of) != "train"]
    if set(df.PTID) & set(held_out.PTID) or set(df.RID) & set(held_out.RID):
        raise AssertionError("train index shares a subject with val/test")
    if df.duplicated(["PTID", "slot"]).any() or df.scan_key.duplicated().any():
        raise AssertionError("duplicate (PTID, slot) or scan_key")
    base = df[df.slot == "bl"].set_index("PTID")[["RID", "label", "vol_path"]].sort_index()
    ref = bl.set_index("PTID")[["RID", "label", "vol_path"]].sort_index()
    if len(base) != len(mi[mi.split == "train"]) or not base.equals(ref):
        raise AssertionError("baseline rows are not exactly mri_index's train split")
    absent = df[~df.vol_path.map(lambda p: os.path.exists(os.path.join(ROOT, p)))]
    if len(absent):
        raise FileNotFoundError(f"{len(absent)} volumes missing on disk: {list(absent.scan_key[:5])}")

    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "train_scans.csv"), index=False)
    excluded[COLUMNS + ["reason"]].to_csv(
        os.path.join(out_dir, "train_scans_excluded.csv"), index=False)

    counts = df.DX_bl.value_counts().reindex(CLASSES, fill_value=0)
    per_subject = df.groupby("PTID").size().value_counts().sort_index()
    summary = {
        "policy": policy,
        "rows": len(df),
        "subjects": int(df.PTID.nunique()),
        "rows_by_source": df.source.value_counts().to_dict(),
        "class_counts": {k: int(v) for k, v in counts.items()},
        "scans_per_subject": {str(k): int(v) for k, v in per_subject.items()},
        "excluded": {f"{d}:{r}": int(n) for (d, r), n in
                     excluded.groupby(["DX_bl", "reason"]).size().items()},
    }
    with open(os.path.join(out_dir, "train_scans_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"policy {policy}: {len(df)} scans from {summary['subjects']} train subjects "
          f"-> {os.path.join(out_dir, 'train_scans.csv')}")
    print("  by source   : " + "  ".join(f"{k}={v}" for k, v in summary["rows_by_source"].items()))
    print("  by class    : " + "  ".join(
        f"{k}={v} ({100 * v / len(df):.1f}%)" for k, v in counts.items()))
    print("  scans/subj  : " + "  ".join(f"{k}:{v}" for k, v in summary["scans_per_subject"].items()))
    print(f"  excluded    : {len(excluded)} follow-ups")
    for k, v in summary["excluded"].items():
        print(f"      {k:32s} {v}")
    return df, excluded


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--policy", default="consistent", choices=POLICIES)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()
    build(args.policy, args.out_dir)


if __name__ == "__main__":
    main()
