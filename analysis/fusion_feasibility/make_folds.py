"""Build the cross-fitting folds for ALZ, over the CLINICAL train split.

The clinical split (dataset/processed/{train,val,test}_rids.csv) is the master
split. Only clinical-train subjects with a baseline MRI enter the folds;
clinical val and test subjects are never written to any fold index, so no fold
model can train on, early-stop on, or select with them.

For each fold k (5, stratified by 4-class label, one fold per subject):
    test  = fold k                       (predicted, never trained or selected on)
    val   = 15% of the other 4 folds     (early stopping / best-val-loss only)
    train = the remaining 85%

Each fold gets a data root, folds/fold{k}/, holding its own mri_index.csv and a
vol96 symlink to dataset/mri_prep/vol96, so m3_train.py runs unchanged via
--data-root and its subject-disjointness guard covers all three parts.

Writes folds/folds.csv (RID, PTID, label, fold, role per fold) for the record.
"""
import os

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PROCESSED = os.path.join(ROOT, "dataset", "processed")
MRI_INDEX = os.path.join(ROOT, "dataset", "mri_prep", "mri_index.csv")
VOL96 = os.path.join(ROOT, "dataset", "mri_prep", "vol96")
OUT = os.path.join(HERE, "folds")

K = 5
SEED = 42
INNER_VAL = 0.15


def clinical_split():
    return {int(r): s for s in ("train", "val", "test")
            for r in pd.read_csv(os.path.join(PROCESSED, f"{s}_rids.csv")).RID}


def main():
    split_of = clinical_split()
    mi = pd.read_csv(MRI_INDEX)
    mi["clin_split"] = mi.RID.map(split_of)
    assert mi.clin_split.notna().all(), "MRI subject missing from the clinical split"
    tr = mi[mi.clin_split == "train"].reset_index(drop=True)
    print(f"clinical-train subjects with MRI: {len(tr)}")

    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
    tr["fold"] = -1
    for k, (_, te) in enumerate(skf.split(tr, tr.label)):
        tr.loc[te, "fold"] = k

    record = tr[["RID", "PTID", "label", "fold"]].copy()
    cols = ["RID", "PTID", "DX_bl", "label", "PHASE", "split", "vol_path", "mni_path"]
    for k in range(K):
        rest = tr[tr.fold != k]
        inner_tr, inner_va = train_test_split(
            rest, test_size=INNER_VAL, stratify=rest.label, random_state=SEED + k)
        idx = pd.concat([inner_tr.assign(split="train"), inner_va.assign(split="val"),
                         tr[tr.fold == k].assign(split="test")])[cols]
        assert idx.RID.is_unique and set(idx.RID) == set(tr.RID)
        assert set(idx.RID.map(split_of)) == {"train"}, "non-train subject in a fold index"
        d = os.path.join(OUT, f"fold{k}")
        os.makedirs(d, exist_ok=True)
        idx.to_csv(os.path.join(d, "mri_index.csv"), index=False)
        link = os.path.join(d, "vol96")
        if not os.path.islink(link):
            os.symlink(VOL96, link)
        record[f"role_fold{k}"] = record.RID.map(idx.set_index("RID").split)
        print(f"fold{k}: " + "  ".join(
            f"{s} {len(g)} {g.label.value_counts().sort_index().tolist()}"
            for s, g in idx.groupby("split", sort=False)))

    record.to_csv(os.path.join(OUT, "folds.csv"), index=False)


if __name__ == "__main__":
    main()
