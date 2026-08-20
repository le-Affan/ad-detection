"""Build the index the MRI dataset reads: what is preprocessed, and its split.

Disk is the source of truth. This globs whatever m1 has actually produced and
inner-joins it against the cohort manifest and the clinical split files, so the
index always describes the subjects that exist right now. Nothing here knows
how many subjects there are supposed to be.

THE SPLIT IS REUSED, NEVER RECOMPUTED. `dataset/processed/{train,val,test}_rids.csv`
were fixed at the clinical stage over all 814 subjects, subject-level and
stratified. The MRI subset inherits that assignment, which is what keeps the
two streams leak-free and lets them be fused later. Re-deriving a split here
from the subjects that happen to be downloaded would silently break that.

A subject is indexed only if BOTH artefacts exist -- the full-res MNI volume and
the model-input volume. The full-res directory is the one globbed, since it is
the canonical output of registration; a subject with a full-res volume but no
downsample is a half-finished run and is reported rather than quietly indexed.
"""
import argparse
import glob
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PREP = os.path.join(ROOT, "dataset", "mri_prep")
MNI_DIR = os.path.join(PREP, "mni")
VOL_DIR = os.path.join(PREP, "vol96")
MANIFEST = os.path.join(ROOT, "dataset", "mri_manifest", "mri_manifest.csv")
PROCESSED = os.path.join(ROOT, "dataset", "processed")
OUT = os.path.join(PREP, "mri_index.csv")

SPLITS = ("train", "val", "test")


def load_splits():
    """RID -> split name, from the clinical split files."""
    m = {}
    for s in SPLITS:
        rids = pd.read_csv(os.path.join(PROCESSED, f"{s}_rids.csv")).RID.astype(int)
        for r in rids:
            if r in m:
                raise ValueError(f"RID {r} appears in more than one split file")
            m[int(r)] = s
    return m


def build(out=OUT):
    found = sorted(glob.glob(os.path.join(MNI_DIR, "*.nii.gz")))
    rows = [{"PTID": os.path.basename(p)[:-len(".nii.gz")],
             "mni_path": os.path.relpath(p, ROOT)} for p in found]
    df = pd.DataFrame(rows)
    print(f"preprocessed volumes on disk : {len(df)}")
    if df.empty:
        raise SystemExit("nothing preprocessed yet -- run m1_preprocess.py first")

    df["vol_path"] = df.PTID.map(lambda p: os.path.join("dataset", "mri_prep", "vol96", f"{p}.nii.gz"))
    missing_vol = df[~df.vol_path.map(lambda p: os.path.exists(os.path.join(ROOT, p)))]
    if len(missing_vol):
        print(f"  WARNING: {len(missing_vol)} have a full-res volume but no {os.path.basename(VOL_DIR)} "
              f"volume; excluded as half-finished: {list(missing_vol.PTID)[:5]}")
        df = df[~df.PTID.isin(missing_vol.PTID)]

    man = pd.read_csv(MANIFEST, dtype=str)
    man["RID"] = man.RID.astype(int)
    man["label"] = man.label.astype(int)
    before = len(df)
    df = df.merge(man[["PTID", "RID", "DX_bl", "label", "PHASE"]], on="PTID", how="inner")
    if len(df) != before:
        raise ValueError(f"{before - len(df)} preprocessed subjects are absent from the manifest")

    split_of = load_splits()
    df["split"] = df.RID.map(split_of)
    unassigned = df[df.split.isna()]
    if len(unassigned):
        raise ValueError(f"{len(unassigned)} subjects have no split assignment: "
                         f"{list(unassigned.PTID)[:5]}")

    df = df[["RID", "PTID", "DX_bl", "label", "PHASE", "split", "vol_path", "mni_path"]]
    df = df.sort_values(["split", "PTID"]).reset_index(drop=True)
    df.to_csv(out, index=False)

    print(f"indexed {len(df)} subjects -> {out}\n")
    order = ["CN", "EMCI", "LMCI", "AD"]
    tab = pd.crosstab(df.split, df.DX_bl).reindex(index=list(SPLITS)).reindex(columns=order, fill_value=0)
    tab = tab.fillna(0).astype(int)
    tab["total"] = tab.sum(axis=1)
    print(tab.to_string())
    print(f"\ntotal {len(df)}   (label/DX_bl agreement: "
          f"{int((df.label == df.DX_bl.map({k: i for i, k in enumerate(order)})).sum())}/{len(df)})")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=OUT)
    build(ap.parse_args().out)


if __name__ == "__main__":
    main()
