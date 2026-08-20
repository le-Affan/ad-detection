"""
MRI acquisition manifest for the 814-subject clinical cohort.

Read-only with respect to the pipeline: it reads step1_labeled.csv (the
cohort source of truth) and two raw ADNI tables, and writes only into
dataset/mri_manifest/. Nothing under dataset/processed/, models/, or src/
is opened for writing.

Purpose: give the LONI IDA portal the subject IDs and visit/date context
needed to pull the matching baseline scans.

Column provenance
-----------------
PTID is already present in step1_labeled.csv, coalesced by step0 from
CDR (first in CANONICAL_PRIORITY["PTID"]). It is re-verified here against
raw dataset/raw/CDR.csv rather than trusted.

There is no single cohort-wide baseline visit code: step0 deliberately
collapses each source table to *its own* baseline code before joining
(CDR uses 'sc', ADAS/ECog use 'bl', UCSFFSX51 uses 'scmri'). The manifest
therefore carries two visit blocks instead of pretending there is one:

  BASELINE_*  the clinical baseline every subject shares -- CDR screening
              ('sc' for all 814), the visit the cohort and CDRSB come
              from. Complete, zero nulls.
  MRI_*       the FreeSurfer (UCSFFSX51) row step0 actually joined, i.e.
              the visit an actual scan was processed from. This is the
              date to match on the portal, and it is NOT complete.
"""
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STEP1 = os.path.join(ROOT, "dataset", "processed", "step1_labeled.csv")
CDR_RAW = os.path.join(ROOT, "dataset", "raw", "CDR.csv")
FSX_RAW = os.path.join(ROOT, "dataset", "raw", "adnimerge2_tables", "UCSFFSX51.csv")
OUT_DIR = os.path.join(ROOT, "dataset", "mri_manifest")

NA_VALUES = ["NA", ""]
LABEL_NAMES = {"0": "CN", "1": "EMCI", "2": "LMCI", "3": "AD"}
EXPECTED_CLASSES = {"CN": 190, "EMCI": 310, "LMCI": 163, "AD": 151}
# UCSFFSX51 codes its baseline-equivalent visit as 'scmri'; 'bl'/'sc' are
# accepted too so the check does not depend on one export's convention.
FS_BASELINE_CODES = {"scmri", "bl", "sc"}


def report(ok, msg):
    print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    return ok


def build():
    s1 = pd.read_csv(STEP1, dtype=str, low_memory=False, na_values=NA_VALUES)

    man = pd.DataFrame({
        "RID": s1["RID"],
        "PTID": s1["PTID"],
        "DX_bl": s1["DX_bl"],
        "label": s1["label"],
        "PHASE": s1["PHASE"],
        "ORIGPROT": s1["ORIGPROT"],
        "BASELINE_VISCODE": s1["VISCODE__CDR"],
        "BASELINE_VISCODE2": s1["VISCODE2__CDR"],
        "BASELINE_EXAMDATE": s1["VISDATE__CDR"],
        "MRI_VISCODE": s1["VISCODE__UCSFFSX51"],
        "MRI_VISCODE2": s1["VISCODE2__UCSFFSX51"],
        "MRI_EXAMDATE": s1["EXAMDATE"],
    })
    man["HAS_FS_ROW"] = man["MRI_EXAMDATE"].notna()
    man["HAS_FS_BASELINE"] = man["MRI_VISCODE2"].isin(FS_BASELINE_CODES)
    man = man.sort_values("RID", key=lambda s: s.astype(int)).reset_index(drop=True)
    return s1, man


def checks(s1, man):
    results = []
    print("\n--- Section 2: cohort sanity checks ---")

    results.append(report(len(man) == 814, f"row count is {len(man)}, expected 814"))
    results.append(report(man["RID"].is_unique, f"RID unique ({man['RID'].nunique()} distinct)"))
    results.append(report(
        man["PTID"].isna().sum() == 0,
        f"PTID nulls = {man['PTID'].isna().sum()}"))
    results.append(report(
        man["BASELINE_EXAMDATE"].isna().sum() == 0,
        f"BASELINE_EXAMDATE nulls = {man['BASELINE_EXAMDATE'].isna().sum()}"))
    results.append(report(man["PTID"].is_unique, f"PTID unique ({man['PTID'].nunique()} distinct)"))
    results.append(report(
        bool(man["PTID"].str.match(r"^\d{3}_S_\d{4}$").all()),
        "every PTID matches NNN_S_NNNN"))

    # Trailing numeric segment of PTID vs RID. Compared as integers so a
    # zero-padded PTID tail is not counted as a mismatch.
    tail = man["PTID"].str.split("_").str[-1].astype(int)
    mism = man[tail != man["RID"].astype(int)]
    results.append(report(len(mism) == 0, f"PTID trailing segment == RID for all rows ({len(mism)} mismatches)"))
    if len(mism):
        print("        mismatching RIDs (reported, NOT dropped):")
        for _, r in mism.iterrows():
            print(f"          RID {r['RID']} -> PTID {r['PTID']}")

    named = man["label"].map(LABEL_NAMES)
    results.append(report(
        bool((named == man["DX_bl"]).all()),
        "label code agrees with DX_bl string on every row"))
    dist = named.value_counts().to_dict()
    for cls, want in EXPECTED_CLASSES.items():
        got = dist.get(cls, 0)
        results.append(report(got == want, f"class {cls:4s} = {got:3d}, expected {want}"))

    # PTID provenance re-verified against the raw table it came from.
    cdr = pd.read_csv(CDR_RAW, dtype=str, low_memory=False, na_values=NA_VALUES)
    cdr = cdr[cdr["RID"].isin(set(man["RID"]))].dropna(subset=["PTID"])
    per_rid = cdr.groupby("RID")["PTID"].nunique()
    results.append(report(
        bool((per_rid == 1).all()),
        f"raw CDR.csv gives one distinct PTID per RID ({int((per_rid > 1).sum())} conflicts)"))
    lookup = cdr.drop_duplicates("RID").set_index("RID")["PTID"]
    joined = man["RID"].map(lookup)
    results.append(report(
        bool((joined == man["PTID"]).all()) and joined.notna().all(),
        f"PTID matches raw CDR.csv for {int((joined == man['PTID']).sum())} / {len(man)} subjects"))

    return all(results)


def mri_coverage(man):
    """FreeSurfer coverage as an availability INDICATOR (see printout)."""
    print("\n--- Section 3: FreeSurfer (UCSFFSX51) coverage ---")
    fs = pd.read_csv(FSX_RAW, dtype=str, low_memory=False, na_values=NA_VALUES,
                     usecols=["RID", "VISCODE", "VISCODE2", "EXAMDATE"])
    cohort = set(man["RID"])
    any_visit = cohort & set(fs["RID"])
    at_bl = cohort & set(fs[fs["VISCODE2"].isin(FS_BASELINE_CODES)]["RID"])

    print(f"  in UCSFFSX51 at the BASELINE visit : {len(at_bl)} / 814")
    print(f"  in UCSFFSX51 at ANY visit          : {len(any_visit)} / 814")

    missing_bl = sorted(cohort - at_bl, key=int)
    missing_any = sorted(cohort - any_visit, key=int)
    print(f"\n  RIDs with NO baseline FreeSurfer row ({len(missing_bl)}):")
    sub = man.set_index("RID")
    for rid in missing_bl:
        row = sub.loc[rid]
        note = "no FreeSurfer row at all" if rid in missing_any else \
               f"nearest processed scan at {row['MRI_VISCODE2']} ({row['MRI_EXAMDATE']})"
        print(f"    RID {rid:>5s}  PTID {row['PTID']:<11s} {row['DX_bl']:<4s}  {note}")

    print("\n  NOTE: UCSFFSX51 is a derived FreeSurfer table, so a row in it")
    print("  proves a scan existed AND passed processing. Absence is an")
    print("  INDICATOR ONLY, not proof that no scan exists on the IDA portal:")
    print("  a scan can be present but unprocessed, QC-failed, or excluded")
    print("  from this particular export. Confirm all 814 against IDA itself.")
    return len(at_bl), len(any_visit), missing_bl, missing_any


def main():
    s1, man = build()
    ok = checks(s1, man)
    at_bl, any_visit, missing_bl, missing_any = mri_coverage(man)

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "mri_manifest.csv")
    txt_path = os.path.join(OUT_DIR, "ptid_list.txt")
    man.to_csv(csv_path, index=False)
    with open(txt_path, "w") as fh:
        fh.write("\n".join(man["PTID"]) + "\n")

    print("\n--- Section 4/5: outputs ---")
    print(f"  {csv_path}  ({len(man)} rows x {man.shape[1]} cols)")
    print(f"  {txt_path}  ({len(man)} PTIDs, no header)")
    print(f"\n  PHASE          : {man['PHASE'].value_counts().to_dict()}")
    print(f"  clinical dates : {man['BASELINE_EXAMDATE'].min()} -> {man['BASELINE_EXAMDATE'].max()}")
    md = man['MRI_EXAMDATE'].dropna()
    print(f"  MRI scan dates : {md.min()} -> {md.max()}  ({len(md)} dated)")
    print(f"\n  ALL SANITY CHECKS {'PASSED' if ok else 'DID NOT PASS -- see FAIL lines above'}")


if __name__ == "__main__":
    main()
