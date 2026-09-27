"""Path A, stage B4: build the index the multi-visit dataset reads.

visits.csv (stage A) says which visits exist in the manifest; this says which
of them are usable as model input right now. It is the Path A counterpart of
m2_index.py and, like it, treats disk as the source of truth: a visit is usable
only if BOTH its full-res MNI volume and its 96^3 volume exist. A present visit
missing either is reported and masked as PREP_FAILED, not silently dropped.

Each visit also carries its QC numbers from p2's reports. A key can appear in
several reports (dry run, baseline check, reruns, the raw fix, the full run);
the one from the most recently written report wins, since that run produced
the volume now on disk.

POST-PREPROCESSING MASKS. Visits that preprocessed without error but whose
volume is unusable are masked here, by name, with the evidence alongside. The
grid stays dense and the subject stays in; only the slot goes dark.

Output: dataset/mri_prep_patha/visits_index.csv -- visits.csv plus
vol_path, mni_path, prep_ok, corr_vs_mni, mask_fill, suspect, raw_fix, with
`present`/`missing_reason` updated by the masks above.
"""
import argparse
import glob
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from p1_visits import OUT_DIR, ROOT  # noqa: E402

VISITS = os.path.join(OUT_DIR, "visits.csv")
OUT = os.path.join(OUT_DIR, "visits_index.csv")
REL = os.path.relpath(OUT_DIR, ROOT)

# 027_S_4966 bl: affine misregistration -- brain tilted against the template,
# scalp inside the brain mask; corr_vs_mni 0.045. The same scan failed the same
# way in the original baseline cohort (0.054), so the fault is in the file.
# Its m06/m12/m24 register normally (median 0.439). Decided 2026-09-27.
POST_MASKED = {("027_S_4966", "bl"): "REG_FAILED"}


def latest_qc():
    """prep_key -> QC row, taking each key from its most recent report."""
    paths = sorted(glob.glob(os.path.join(OUT_DIR, "p2_report_*.csv")), key=os.path.getmtime)
    if not paths:
        raise SystemExit("no p2 reports -- run p2_preprocess.py first")
    qc = pd.concat([pd.read_csv(p).assign(report=os.path.basename(p)) for p in paths])
    qc = qc.drop_duplicates("prep_key", keep="last").set_index("prep_key")
    if "raw_fix" not in qc:
        qc["raw_fix"] = ""
    return qc


def build(out=OUT):
    v = pd.read_csv(VISITS)
    qc = latest_qc()

    def rel(sub, k):
        return None if pd.isna(k) else f"{REL}/{sub}/{k}.nii.gz"

    v["vol_path"] = v.prep_key.map(lambda k: rel("vol96", k))
    v["mni_path"] = v.prep_key.map(lambda k: rel("mni", k))
    on_disk = v.vol_path.notna() & v.vol_path.map(lambda p: pd.notna(p) and os.path.exists(os.path.join(ROOT, p))) \
        & v.mni_path.map(lambda p: pd.notna(p) and os.path.exists(os.path.join(ROOT, p)))
    v["prep_ok"] = on_disk

    failed = v[v.present & ~on_disk]
    if len(failed):
        print(f"WARNING: {len(failed)} present visits have no complete volume pair -> PREP_FAILED:")
        print("  " + ", ".join(failed.prep_key.head(20)))
    v.loc[v.present & ~on_disk, "missing_reason"] = "PREP_FAILED"
    v.loc[v.present & ~on_disk, "present"] = False

    for (ptid, slot), reason in POST_MASKED.items():
        hit = (v.PTID == ptid) & (v.slot == slot) & v.present
        if hit.sum() != 1:
            raise ValueError(f"POST_MASKED {ptid}/{slot} does not match exactly one usable visit")
        v.loc[hit, ["present", "missing_reason"]] = [False, reason]

    for c in ("corr_vs_mni", "mask_fill", "suspect", "raw_fix", "report"):
        v[c] = v.prep_key.map(qc[c]) if c in qc else pd.NA
    v["suspect"] = v.suspect.fillna("")
    v["raw_fix"] = v.raw_fix.fillna("")
    # a masked slot is not model input: no path the dataset could follow
    v.loc[~v.present, ["vol_path", "mni_path"]] = pd.NA

    no_qc = v[v.present & v.corr_vs_mni.isna()]
    if len(no_qc):
        raise ValueError(f"{len(no_qc)} usable visits have no QC row: {list(no_qc.prep_key[:5])}")

    # volumes on disk that no usable visit claims -- stale or masked leftovers
    claimed = set(v[v.present].prep_key)
    orphans = sorted(os.path.basename(p)[:-len(".nii.gz")]
                     for p in glob.glob(os.path.join(OUT_DIR, "vol96", "*.nii.gz")))
    orphans = [k for k in orphans if k not in claimed]

    v.to_csv(out, index=False)
    report(v, orphans, out)
    return v


def report(v, orphans, out):
    use = v[v.present]
    subj = v.groupby("PTID").agg(split=("split", "first"), DX_bl=("DX_bl", "first"),
                                 n=("present", "sum"), has_bl=("present", "first"))
    print(f"usable visits {len(use)} of {len(v)} slots, {v.PTID.nunique()} subjects")
    print("\nslot x missing_reason:")
    print(pd.crosstab(v.slot, v.missing_reason).reindex(["bl", "m06", "m12", "m24"]).to_string())
    print("\nusable visits per subject:")
    print(subj.n.value_counts().sort_index().to_string())
    print(f"subjects without a usable bl: {int((~subj.has_bl).sum())}")
    print(f"subjects with zero usable visits: {int((subj.n == 0).sum())}")
    print("\nsplit x DX_bl (subjects):")
    print(pd.crosstab(subj.split, subj.DX_bl)
          .reindex(index=["train", "val", "test"], columns=["CN", "EMCI", "LMCI", "AD"]).to_string())
    print(f"\nusable visits flagged suspect (kept): {int((use.suspect != '').sum())}")
    print(f"raw-fixed visits: {list(use[use.raw_fix != ''].prep_key)}")
    print(f"corr_vs_mni over usable visits: median {use.corr_vs_mni.median():.3f}, "
          f"min {use.corr_vs_mni.min():.3f}")
    if orphans:
        print(f"\nvolumes on disk not used by any usable visit: {orphans}")
    print(f"\nwrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=OUT)
    build(ap.parse_args().out)


if __name__ == "__main__":
    main()
