"""Build the KAN-fusion dataset for clinical-train and clinical-val. Never reads test.

Design (docs/pipeline_blueprint.md v2, section 3.1):
  Clinical stream: the 36 step-4 features minus the 7 FreeSurfer columns
    (arm A0 of the morphometry gate, 29 features). Taken unchanged from
    dataset/processed/step4_{train,val}.csv: median/mode imputation and
    z-scoring already fit on clinical-train only.
  MRI stream, 7 inputs: Ventricles, Hippocampus, WholeBrain, Entorhinal,
    Fusiform, MidTemp, each divided by ICV (raw ADNI FreeSurfer values from
    step2_features.csv), plus cross-fitted ALZ from
    analysis/fusion_feasibility/alz_crossfit.csv (train: out-of-fold;
    val: mean of the 5 fold models).
  Missing values, following step 4: a ratio missing on train or val is
    replaced by the clinical-train median of that ratio, then every ratio is
    z-scored with the clinical-train mean/SD. Nothing is fit on val.
    ALZ stays NaN for subjects without a scan (never imputed).
  Not inputs: brain-age gap, SynthSeg volumes (card only).

step2_features.csv is filtered to train/val RIDs as it is read; test rows are
dropped before any processing, and test_rids.csv is opened only to assert
that no test RID appears in the output.

Output: analysis/kan_fusion/fusion_trainval.csv and fusion_manifest.json.
Run with LYP/.venv-fusion.
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PROCESSED = os.path.join(ROOT, "dataset", "processed")
CROSSFIT = os.path.join(ROOT, "analysis", "fusion_feasibility", "alz_crossfit.csv")

NAMES = ["CN", "EMCI", "LMCI", "AD"]
FREESURFER = ["Ventricles", "Hippocampus", "WholeBrain", "Entorhinal", "Fusiform", "MidTemp", "ICV"]
VOLUMES = FREESURFER[:-1]
RATIOS = [f"{v}_ICV" for v in VOLUMES]
MRI_INPUTS = RATIOS + ["ALZ"]


def main():
    rids = {s: set(pd.read_csv(os.path.join(PROCESSED, f"{s}_rids.csv")).RID) for s in ("train", "val")}
    keep = rids["train"] | rids["val"]

    raw = pd.read_csv(os.path.join(PROCESSED, "step2_features.csv"), low_memory=False)
    raw = raw.loc[raw.RID.isin(keep), ["RID"] + FREESURFER].set_index("RID")

    cf = pd.read_csv(CROSSFIT)
    parts = []
    for s in ("train", "val"):
        d = pd.read_csv(os.path.join(PROCESSED, f"step4_{s}.csv"))
        assert set(d.RID) == rids[s]
        c = cf[cf.clin_split == s].set_index("RID")
        assert set(c.index) == rids[s], f"{s}: crossfit RIDs != split RIDs"
        assert (c.label.reindex(d.RID).to_numpy() == d.label.to_numpy()).all()
        d.insert(1, "PTID", d.RID.map(c.PTID))
        d.insert(2, "split", s)
        d["ALZ"] = d.RID.map(c.ALZ)
        d["has_mri"] = d.RID.map(c.has_mri).astype(bool)
        assert d.loc[~d.has_mri, "ALZ"].isna().all() and d.loc[d.has_mri, "ALZ"].notna().all()
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)

    base = [c for c in parts[0].columns if c not in ("RID", "PTID", "split", "label", "ALZ", "has_mri")]
    assert len(base) == 36 and set(FREESURFER) <= set(base)
    clinical = [c for c in base if c not in FREESURFER]
    assert len(clinical) == 29

    # MRI stream: volume / ICV from the raw FreeSurfer values
    r = raw.reindex(df.RID)
    for v, name in zip(VOLUMES, RATIOS):
        df[name] = (r[v] / r["ICV"]).to_numpy()
    df["fs_missing"] = df[RATIOS].isna().any(axis=1).to_numpy()
    tr = df.split == "train"
    median = {c: float(df.loc[tr, c].median()) for c in RATIOS}
    for c in RATIOS:
        df[c] = df[c].fillna(median[c])
    mean = {c: float(df.loc[tr, c].mean()) for c in RATIOS}
    std = {c: float(df.loc[tr, c].std()) for c in RATIOS}
    for c in RATIOS:
        df[c] = (df[c] - mean[c]) / std[c]

    df.insert(4, "diagnosis", df.label.map(dict(enumerate(NAMES))))
    out = df[["RID", "PTID", "split", "label", "diagnosis"] + clinical + MRI_INPUTS + ["has_mri", "fs_missing"]]
    test = set(pd.read_csv(os.path.join(PROCESSED, "test_rids.csv")).RID)
    assert not out.RID.isin(test).any()
    assert out.RID.is_unique and out[clinical + RATIOS].notna().all().all()
    out.to_csv(os.path.join(HERE, "fusion_trainval.csv"), index=False)

    manifest = {
        "rows": out.split.value_counts().to_dict(),
        "clinical_features": clinical,
        "mri_inputs": MRI_INPUTS,
        "ratio_definition": "ADNI FreeSurfer volume / ADNI FreeSurfer ICV (raw mm^3, step2_features.csv)",
        "ratio_imputation_train_median": median,
        "ratio_zscore_train_mean": mean,
        "ratio_zscore_train_std": std,
        "alz_source": "analysis/fusion_feasibility/alz_crossfit.csv (train out-of-fold; val = mean of 5 fold models)",
        "fs_missing": {s: int(out[(out.split == s)].fs_missing.sum()) for s in ("train", "val")},
        "fs_missing_rids": out.loc[out.fs_missing, "RID"].tolist(),
        "no_mri": out.loc[~out.has_mri, ["RID", "PTID", "split", "diagnosis"]].to_dict("records"),
    }
    with open(os.path.join(HERE, "fusion_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"{len(out)} rows, {len(clinical)} clinical + {len(MRI_INPUTS)} MRI inputs")
    print(pd.crosstab(out.split, out.diagnosis)[NAMES].assign(total=lambda x: x.sum(axis=1)))
    print(out.groupby("split")[["has_mri", "fs_missing"]].sum())
    print("ALZ NaN:", out.groupby("split").ALZ.apply(lambda s: int(s.isna().sum())).to_dict())
    print("no MRI:", manifest["no_mri"])
    print("FS missing:", out.loc[out.fs_missing, ["PTID", "split", "diagnosis", "has_mri"]].to_dict("records"))
    print("train ratio medians:", {k: round(v, 5) for k, v in median.items()})
    print(out.groupby("diagnosis")[MRI_INPUTS].mean().loc[NAMES].round(3).to_string())


if __name__ == "__main__":
    main()
