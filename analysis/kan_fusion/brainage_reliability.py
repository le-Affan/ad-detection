"""Brain-age reliability rule for the assessment card, applied to every train/val scan.

Rule (pre-registered in preregistration.md, item h): a scan's brain age is
UNRELIABLE if, inside the ICBM152 template brain mask, the m1 mni/ volume has
  (1) no zero-intensity voxels, or
  (2) more than 10% saturated voxels (intensity >= 0.999 after m1's min-max
      scaling to [0, 1]).
The rule comes from the mri-morphometry Part 1 outlier QC
(analysis/mri_morphometry/part1_report.md): GE Sag IR-SPGR scans with a
washed-out intensity profile fill the whole mask and saturate ~23% of it.
It is a card rule only; brain age is not a model input.

The file list comes from clinical-train/val PTIDs (fusion_trainval.csv); a test
scan is never opened, and this is asserted against test_rids.csv.
Mask and saturation threshold are those of analysis/mri_morphometry/outlier_qc.py.

Run with LYP/.venv-mri (ants, nilearn, nibabel).
Output: analysis/kan_fusion/brainage_reliability.csv, printed counts.
"""
import os
import sys

import nibabel as nib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src", "mri"))
MNI = os.path.join(ROOT, "dataset", "mri_prep", "mni")
SAT = 0.999
MAX_SAT_FRAC = 0.10


def main():
    import m1_preprocess as m1
    _, mask_w, _, _, _ = m1.load_template(96)
    mask = mask_w.numpy() > 0.5

    subs = pd.read_csv(os.path.join(HERE, "fusion_trainval.csv"),
                       usecols=["RID", "PTID", "split", "diagnosis", "has_mri"])
    test = set(pd.read_csv(os.path.join(ROOT, "dataset", "processed", "test_rids.csv")).RID)
    assert not subs.RID.isin(test).any() and (subs.PTID.str[-4:].astype(int) == subs.RID).all()
    ba = pd.read_csv(os.path.join(ROOT, "analysis", "brain_age", "brainage_trainval.csv"),
                     usecols=["PTID", "has_brainage"])
    m1r = pd.read_csv(os.path.join(ROOT, "dataset", "mri_prep", "m1_report_full.csv"), usecols=["PTID", "mask_fill"])

    rows = []
    for p in subs.PTID:
        path = os.path.join(MNI, f"{p}.nii.gz")
        if not os.path.exists(path):
            rows.append({"PTID": p, "has_scan": False})
            continue
        x = nib.load(path).get_fdata(dtype="float32")[mask]
        rows.append({"PTID": p, "has_scan": True, "n_mask": int(x.size), "n_zero": int((x == 0).sum()),
                     "zero_frac": float((x == 0).mean()), "sat_frac": float((x >= SAT).mean())})
    d = subs.merge(pd.DataFrame(rows), on="PTID").merge(ba, on="PTID", how="left").merge(m1r, on="PTID", how="left")
    d["rule_no_zero"] = d.has_scan & (d.n_zero == 0)
    d["rule_saturated"] = d.has_scan & (d.sat_frac > MAX_SAT_FRAC)
    d["unreliable"] = d.rule_no_zero | d.rule_saturated
    d.to_csv(os.path.join(HERE, "brainage_reliability.csv"), index=False)

    s = d[d.has_scan]
    print(f"scans checked: {len(s)} ({(s.split == 'train').sum()} train, {(s.split == 'val').sum()} val); "
          f"no scan: {d.loc[~d.has_scan, 'PTID'].tolist()}")
    print(s.groupby("split")[["rule_no_zero", "rule_saturated", "unreliable"]].sum())
    print(s[s.unreliable][["PTID", "split", "diagnosis", "zero_frac", "sat_frac", "mask_fill",
                           "has_brainage"]].to_string(index=False))
    print("\nnear misses: zero_frac < 0.001 or sat_frac > 0.08 among reliable scans")
    nm = s[~s.unreliable & ((s.zero_frac < 1e-3) | (s.sat_frac > 0.08))]
    print(nm[["PTID", "split", "diagnosis", "zero_frac", "sat_frac", "mask_fill"]].to_string(index=False)
          if len(nm) else "  none")
    q = s.sat_frac.quantile([.5, .95, .99])
    print(f"\nsat_frac median {q[.5]:.3f}, p95 {q[.95]:.3f}, p99 {q[.99]:.3f}; "
          f"zero_frac median {s.zero_frac.median():.4f}")
    print("agreement with m1 mask_fill >= 0.9999:",
          pd.crosstab(s.rule_no_zero, s.mask_fill >= 0.9999).to_dict())


if __name__ == "__main__":
    main()
