"""
Step 4 — Feature processing (train-only fit).

Input:  dataset/processed/step2_features.csv (RID + label + 35 features)
        dataset/processed/{train,val,test}_rids.csv (step3 split)
Output: dataset/processed/step4_{train,val,test}.csv
        dataset/processed/step4_manifest.json

Everything (imputation values, missingness-indicator thresholds,
normalization mean/std) is fit on the train split only, then applied
unchanged to val/test.

Per docs/context.txt Step 4:
- median imputation (numeric), mode imputation (categorical: PTGENDER)
- missingness indicator columns for features with >10% NaN in train
- CogIndex = (normalized MMSE) - (normalized CDRSB) - (normalized ADAS13)
- z-score normalization, fit on train only

PTGENDER is the registered CatBoost categorical (step2 manifest) — it is
imputed by mode but never z-normalized or one-hot encoded. RID and label
are passed through untouched.
"""
import argparse
import json
import os

import pandas as pd

ID_COLUMNS = ["RID", "label"]
CATEGORICAL_FEATURES = ["PTGENDER"]
MISSINGNESS_THRESHOLD = 0.10
COGINDEX_COMPONENTS = ("MMSE", "CDRSB", "ADAS13")


def load_splits(features_path, split_dir):
    features = pd.read_csv(features_path, low_memory=False)
    splits = {}
    for name in ("train", "val", "test"):
        rids = pd.read_csv(os.path.join(split_dir, f"{name}_rids.csv"))["RID"]
        splits[name] = features[features["RID"].isin(rids)].copy()
    return splits


def fit_imputation(train, numeric_features, categorical_features):
    medians = {f: float(train[f].median()) for f in numeric_features}
    modes = {f: train[f].mode(dropna=True).iloc[0] for f in categorical_features}
    return medians, modes


def apply_imputation(df, medians, modes):
    df = df.copy()
    for f, v in medians.items():
        df[f] = df[f].fillna(v)
    for f, v in modes.items():
        df[f] = df[f].fillna(v)
    return df


def fit_missingness_indicators(train, numeric_features, threshold):
    rates = train[numeric_features].isna().mean()
    return [f for f in numeric_features if rates[f] > threshold], rates


def add_missingness_indicators(df, original, indicator_features):
    df = df.copy()
    for f in indicator_features:
        df[f"{f}_missing"] = original[f].isna().astype(int)
    return df


def fit_normalization(train, numeric_features):
    means = {f: float(train[f].mean()) for f in numeric_features}
    stds = {f: float(train[f].std()) for f in numeric_features}
    return means, stds


def apply_normalization(df, means, stds):
    df = df.copy()
    for f, mean in means.items():
        std = stds[f]
        df[f] = (df[f] - mean) / std if std else 0.0
    return df


def add_cogindex(df):
    df = df.copy()
    df["CogIndex"] = df["MMSE"] - df["CDRSB"] - df["ADAS13"]
    return df


def run(features_path, split_dir, out_dir, threshold):
    splits = load_splits(features_path, split_dir)
    train = splits["train"]

    all_features = [c for c in train.columns if c not in ID_COLUMNS]
    categorical_features = [c for c in CATEGORICAL_FEATURES if c in all_features]
    numeric_features = [c for c in all_features if c not in categorical_features]

    # 1. Imputation, fit on train only.
    medians, modes = fit_imputation(train, numeric_features, categorical_features)

    # 2. Missingness indicators: threshold evaluated on raw (pre-impute) train NaN
    # rates. current data tops out at 5.5% (EcogSPOrgan), so this is expected to
    # produce zero indicator columns.
    indicator_features, train_missing_rates = fit_missingness_indicators(
        train, numeric_features, threshold,
    )

    imputed = {}
    for name, split in splits.items():
        imputed[name] = add_missingness_indicators(
            apply_imputation(split, medians, modes), split, indicator_features,
        )

    # 4. Z-score normalization, fit on train only. RID/label/categoricals and
    # the *_missing indicators are excluded.
    means, stds = fit_normalization(imputed["train"], numeric_features)

    processed = {}
    for name, split in imputed.items():
        normalized = apply_normalization(split, means, stds)
        # 3. CogIndex, computed from the already-normalized components.
        processed[name] = add_cogindex(normalized)

    os.makedirs(out_dir, exist_ok=True)
    out_paths = {}
    for name, df in processed.items():
        path = os.path.join(out_dir, f"step4_{name}.csv")
        df.to_csv(path, index=False)
        out_paths[name] = path

    manifest = {
        "imputation": {
            "median": medians,
            "mode": modes,
        },
        "missingness_threshold": threshold,
        "missingness_indicators_created": [f"{f}_missing" for f in indicator_features],
        "train_missing_rates": {f: float(r) for f, r in train_missing_rates.items()},
        "cogindex_formula": "CogIndex = (normalized MMSE) - (normalized CDRSB) - (normalized ADAS13)",
        "normalization": {
            "mean": means,
            "std": stds,
        },
        "categorical_features": categorical_features,
        "row_counts": {name: int(len(df)) for name, df in processed.items()},
    }
    manifest_path = os.path.join(out_dir, "step4_manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)

    return processed, manifest, out_paths, manifest_path


def main():
    parser = argparse.ArgumentParser(description="Step 4: feature processing")
    parser.add_argument("--features-path", default="dataset/processed/step2_features.csv")
    parser.add_argument("--split-dir", default="dataset/processed")
    parser.add_argument("--out-dir", default="dataset/processed")
    parser.add_argument("--threshold", type=float, default=MISSINGNESS_THRESHOLD)
    args = parser.parse_args()

    processed, manifest, out_paths, manifest_path = run(
        args.features_path, args.split_dir, args.out_dir, args.threshold,
    )

    print("Split written:")
    for name, path in out_paths.items():
        print(f"  {name}: {path}")
    print(f"  manifest: {manifest_path}")

    print("\nPer-split row counts:")
    for name, count in manifest["row_counts"].items():
        print(f"  {name}: {count}")

    print("\nImputation/normalization stats: fit on train only (confirmed by "
          "construction -- fit_imputation/fit_normalization take only the train "
          "split as input).")

    if manifest["missingness_indicators_created"]:
        print(f"\nMissingness indicators created ({len(manifest['missingness_indicators_created'])}):")
        for f in manifest["missingness_indicators_created"]:
            print(f"  {f}")
    else:
        print(f"\nNo feature exceeded {args.threshold:.0%} NaN in train -- "
              f"zero missingness indicators created (expected, not a bug).")

    print(f"\nCogIndex formula applied: {manifest['cogindex_formula']}")


if __name__ == "__main__":
    main()
