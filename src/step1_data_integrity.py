"""
Step 1 — Data integrity: cohort filter and 4-class label assignment.

Input: dataset/processed/step0_joined.csv (RID-keyed, already one row per
subject — step0 collapses every source table to its own baseline row
before joining, so there is no baseline-visit selection left to do here).

Rules (see docs/context.txt):
  - Cohort filter: ORIGPROT in {ADNIGO, ADNI2} only — EMCI/LMCI is only
    valid for subjects who originally enrolled under these phases.
    ORIGPROT is used (not the per-visit PHASE column) because PHASE
    drifts as a subject gets followed into later study phases while
    their enrollment cohort never changes. step0 coalesces ORIGPROT from
    ADSL by explicit priority, so this reads the authoritative
    subject-level value rather than whichever source merged first.
  - Drop SMC (Significant Memory Concern) subjects.
  - Drop rows with missing DX_bl.
  - Drop ADNI1-plain-MCI rows: MCI subjects with no EMCI/LMCI split
    (structurally impossible for an ADNIGO/ADNI2 cohort member since the
    arm-derived split always resolves EMCI or LMCI for real GO/2 MCI
    enrollees — this guards against any stray plain-MCI row).
  - Map: CN=0, EMCI=1, LMCI=2, AD=3.
"""
import argparse
import os

import pandas as pd

COHORT_PHASES = {"ADNIGO", "ADNI2"}
LABEL_MAP = {"CN": 0, "EMCI": 1, "LMCI": 2, "AD": 3}

MRI_COLS = [
    "Ventricles", "Hippocampus", "Entorhinal", "Fusiform",
    "MidTemp", "ICV", "WholeBrain",
]
COVERAGE_CHECK_COLS = ["CDRSB", "CDGLOBAL"] + MRI_COLS


def filter_cohort(df):
    return df[df["ORIGPROT"].isin(COHORT_PHASES)].copy()


def drop_invalid_labels(df):
    df = df[df["DX_bl"] != "SMC"]
    df = df.dropna(subset=["DX_bl"])
    df = df[df["DX_bl"] != "MCI"]  # ADNI1-plain, no EMCI/LMCI split
    return df.copy()


def assign_label(df):
    df["label"] = df["DX_bl"].map(LABEL_MAP)
    return df


def run(in_path, out_path):
    df = pd.read_csv(in_path, dtype=str, low_memory=False)

    if df["RID"].duplicated().any():
        raise ValueError(
            "step0 output has duplicate RIDs; it is expected to be "
            "one row per subject."
        )

    df = filter_cohort(df)
    df = drop_invalid_labels(df)
    df = assign_label(df)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def main():
    parser = argparse.ArgumentParser(description="Step 1: data integrity")
    parser.add_argument("--in-path", default="dataset/processed/step0_joined.csv")
    parser.add_argument("--out", default="dataset/processed/step1_labeled.csv")
    args = parser.parse_args()

    df = run(args.in_path, args.out)
    n = len(df)
    print(f"Output: {n} rows, {df.shape[1]} cols -> {args.out}")
    print(f"Unique subjects: {df['RID'].nunique()}")

    print("\nKey feature coverage:")
    for c in COVERAGE_CHECK_COLS:
        if c not in df.columns:
            print(f"  {c:12s} COLUMN ABSENT")
            continue
        nn = df[c].notna().sum()
        print(f"  {c:12s} non-null {nn:4d}/{n}  ({nn / n * 100:5.1f}%)")

    if "WholeBrain" in df.columns:
        wb = pd.to_numeric(df["WholeBrain"], errors="coerce")
        print(f"\nWholeBrain mean: {wb.mean():,.0f}")

    print("\nClass distribution:")
    counts = df["DX_bl"].value_counts()
    for cls, label in LABEL_MAP.items():
        print(f"  {cls} ({label}): {counts.get(cls, 0)}")


if __name__ == "__main__":
    main()
