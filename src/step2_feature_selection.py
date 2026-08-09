"""
Step 2 — Feature selection.

Input:  dataset/processed/step1_labeled.csv (one row per subject, labelled)
Output: dataset/processed/step2_features.csv  — RID + label + kept features
        dataset/processed/step2_manifest.json — kept/dropped features and
                                                the CatBoost categorical list

Builds the doc's target feature set, drops columns above the missingness
threshold, and hard-excludes label-leaking columns.

Naming: the doc's feature list uses classic ADNIMERGE names (MMSE, ADAS13,
PTGENDER, ...). This dump has no ADNIMERGE table, so each is mapped to the
equivalent column reconstructed in step0 (see FEATURE_MAP). Output columns
use the doc's names so downstream steps and the fusion interface match the
spec.
"""
import argparse
import json
import os

import pandas as pd

# ---------------------------------------------------------------------
# Label-leak exclusions
# ---------------------------------------------------------------------

# Enumerated explicitly rather than matched on a substring like "DX" —
# a keyword match would both miss ARM/ACTARM (which carry the diagnosis
# without a "DX" in the name) and over-match innocent columns.
#
# ARM is the important one: the EMCI/LMCI/SMC granularity in DX_bl was
# derived FROM the enrollment arm, so ARM is very close to a direct
# encoding of the target. It is excluded on leakage grounds, not because
# it happens to be empty for this cohort — an ADSL refresh that populated
# it would otherwise silently leak the label into the model.
LABEL_LEAK_COLUMNS = frozenset({
    "DX_bl",       # label source column
    "label",       # the assigned target itself
    "ARM",         # enrollment arm — supplied EMCI/LMCI/SMC to DX_bl
    "ACTARM",      # actual arm; same information as ARM
    "ARMCD",       # coded arm
    "ACTARMCD",
    "ARM_CAT",     # step0 intermediate: ARM mapped to a label category
    "DXSUM_CAT",   # step0 intermediate: DXSUM diagnosis mapped to a category
    "DX",          # per-visit diagnosis (CN / MCI / Dementia)
    "DXCHANGE",    # absent from this export; excluded if ever reintroduced
    "DXCURREN",
    "DXCONV",
    "DXCONTYP",
    "DXREV",
    "DIAGNOSIS",   # raw DXSUM diagnosis field
    "TRTA",        # treatment arm (actual / planned) — tracks stage
    "TRTP",
})

# ---------------------------------------------------------------------
# Target feature set (doc name -> column produced by step0)
# ---------------------------------------------------------------------
FEATURE_MAP = {
    # Demographics
    "AGE": "AGE",
    "PTGENDER": "SEX",
    "PTEDUCAT": "EDUC",
    "APOE4": "APOE4",
    # Core cognitive
    "CDRSB": "CDRSB",
    "ADAS11": "ADASTT11",
    "ADAS13": "ADASTT13",
    "MMSE": "MMSCORE",
    "RAVLT_immediate": "RAVLTIMM",
    "RAVLT_learning": "RAVLTLRN",
    "RAVLT_forgetting": "RAVLTFG",
    "RAVLT_perc_forgetting": "RAVLTFGP",
    "FAQ": "FAQTOTAL",
    "MOCA": "MOCA",
    # ECog — participant and study partner
    **{f"Ecog{side}{dom}": f"Ecog{side}{dom}"
       for side in ("Pt", "SP")
       for dom in ("Mem", "Lang", "Visspat", "Plan", "Organ", "Divatt", "Total")},
    # MRI volumetrics
    "Ventricles": "Ventricles",
    "Hippocampus": "Hippocampus",
    "WholeBrain": "WholeBrain",
    "Entorhinal": "Entorhinal",
    "Fusiform": "Fusiform",
    "MidTemp": "MidTemp",
    "ICV": "ICV",
}

# Passed to CatBoost as cat_features — never one-hot encoded.
CATEGORICAL_FEATURES = ["PTGENDER"]

MISSINGNESS_THRESHOLD = 0.50
ID_COLUMNS = ["RID", "label"]


def build_feature_frame(df):
    missing_sources = [src for src in FEATURE_MAP.values() if src not in df.columns]
    if missing_sources:
        raise KeyError(f"step1 output is missing expected columns: {missing_sources}")

    out = df[ID_COLUMNS].copy()
    for name, source in FEATURE_MAP.items():
        out[name] = df[source]
    return out


def assert_no_leakage(feature_names):
    """Fail loudly if any leak column reached the candidate set."""
    leaked = sorted(set(feature_names) & LABEL_LEAK_COLUMNS)
    if leaked:
        raise AssertionError(f"label-leaking columns in candidate set: {leaked}")
    return True


def drop_high_missingness(df, feature_names, threshold):
    rates = df[feature_names].isna().mean()
    kept = [f for f in feature_names if rates[f] <= threshold]
    dropped = [(f, float(rates[f])) for f in feature_names if rates[f] > threshold]
    return kept, dropped, rates


def run(in_path, out_path, manifest_path, threshold):
    df = pd.read_csv(in_path, low_memory=False)

    features = build_feature_frame(df)
    candidates = [c for c in features.columns if c not in ID_COLUMNS]

    # Leak check runs against the candidate set AND the raw source frame,
    # so an accidental passthrough is caught either way.
    assert_no_leakage(candidates)
    assert_no_leakage([FEATURE_MAP[c] for c in candidates])

    kept, dropped, rates = drop_high_missingness(features, candidates, threshold)
    out = features[ID_COLUMNS + kept]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)

    manifest = {
        "n_subjects": int(len(out)),
        "missingness_threshold": threshold,
        "features_kept": kept,
        "features_dropped": [{"feature": f, "missing_rate": r} for f, r in dropped],
        "categorical_features": [c for c in CATEGORICAL_FEATURES if c in kept],
        "label_leak_excluded": sorted(LABEL_LEAK_COLUMNS),
        "feature_source_map": {k: FEATURE_MAP[k] for k in kept},
    }
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)

    return out, manifest, rates


def main():
    parser = argparse.ArgumentParser(description="Step 2: feature selection")
    parser.add_argument("--in-path", default="dataset/processed/step1_labeled.csv")
    parser.add_argument("--out", default="dataset/processed/step2_features.csv")
    parser.add_argument("--manifest", default="dataset/processed/step2_manifest.json")
    parser.add_argument("--threshold", type=float, default=MISSINGNESS_THRESHOLD)
    args = parser.parse_args()

    out, manifest, rates = run(args.in_path, args.out, args.manifest, args.threshold)

    print(f"Feature matrix: {out.shape[0]} subjects x {len(manifest['features_kept'])} "
          f"features -> {args.out}")
    print(f"Manifest -> {args.manifest}")

    print(f"\nLabel-leak exclusion: {len(LABEL_LEAK_COLUMNS)} columns on the drop list; "
          f"none present in the candidate set (asserted).")
    print(f"CatBoost categorical features: {manifest['categorical_features']}")

    if manifest["features_dropped"]:
        print(f"\nDropped for >{args.threshold:.0%} missingness:")
        for d in manifest["features_dropped"]:
            print(f"  {d['feature']:24s} {d['missing_rate']:.1%} missing")
    else:
        print(f"\nNo feature exceeded the {args.threshold:.0%} missingness threshold.")

    print("\nKept features (missing rate):")
    for f in manifest["features_kept"]:
        print(f"  {f:24s} {rates[f]:5.1%}")

    print("\nClass distribution:")
    print(out["label"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
