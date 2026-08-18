"""
Step 3 — Subject-level train/val/test split.

Input:  dataset/processed/step2_features.csv (RID + label + features, one
        row per subject — read as-is, no re-filtering)
Output: dataset/processed/train_rids.csv
        dataset/processed/val_rids.csv
        dataset/processed/test_rids.csv
        (each a single-column RID list, so the split is reusable without
        recomputation)

Split: 70% train / 15% val / 15% test, stratified on the 4-class label,
seed 42. Two-stage: carve off test first, then split the remainder into
train/val. A subject (RID) never appears in more than one split.
"""
import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split

LABEL_NAMES = {0: "CN", 1: "EMCI", 2: "LMCI", 3: "AD"}

SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15


def run(in_path, out_dir):
    df = pd.read_csv(in_path, usecols=["RID", "label"])

    train_val, test = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["label"], random_state=SEED,
    )
    # val_size as a fraction of the remaining (train+val) pool.
    train, val = train_test_split(
        train_val, test_size=VAL_SIZE / (1 - TEST_SIZE),
        stratify=train_val["label"], random_state=SEED,
    )

    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for name, split in [("train", train), ("val", val), ("test", test)]:
        path = os.path.join(out_dir, f"{name}_rids.csv")
        split[["RID"]].to_csv(path, index=False)
        paths[name] = path

    return train, val, test, paths


def report(df, train, val, test):
    print(f"Total subjects: {len(df)}")
    print(f"Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")

    print("\nClass distribution per split:")
    dist = pd.DataFrame({
        "train": train["label"].value_counts(),
        "val": val["label"].value_counts(),
        "test": test["label"].value_counts(),
    }).fillna(0).astype(int)
    dist.index = dist.index.map(LABEL_NAMES)
    dist = dist.reindex(["CN", "EMCI", "LMCI", "AD"])
    print(dist.to_string())

    train_ids = set(train["RID"])
    val_ids = set(val["RID"])
    test_ids = set(test["RID"])

    overlaps = {
        "train&val": train_ids & val_ids,
        "train&test": train_ids & test_ids,
        "val&test": val_ids & test_ids,
    }
    for name, ov in overlaps.items():
        assert not ov, f"RID overlap between {name}: {ov}"

    total = len(train_ids) + len(val_ids) + len(test_ids)
    assert total == len(df), "split RID counts do not sum to total subjects"

    print(f"\nRID overlap check: train&val={len(overlaps['train&val'])}, "
          f"train&test={len(overlaps['train&test'])}, "
          f"val&test={len(overlaps['val&test'])} -- all zero, confirmed disjoint.")


def main():
    parser = argparse.ArgumentParser(description="Step 3: subject-level split")
    parser.add_argument("--in-path", default="dataset/processed/step2_features.csv")
    parser.add_argument("--out-dir", default="dataset/processed")
    args = parser.parse_args()

    df = pd.read_csv(args.in_path, usecols=["RID", "label"])
    train, val, test, paths = run(args.in_path, args.out_dir)

    print(f"Split written:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    print()

    report(df, train, val, test)


if __name__ == "__main__":
    main()
