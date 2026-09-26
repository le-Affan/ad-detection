"""Path A, stage A: turn the multi-visit manifest into a per-subject visit table.

This is the multi-visit counterpart of dataset/mri_raw/select_scans.py and does
NOT replace it. select_scans.py picks one baseline scan per subject from an IDA
search export; here every selection has already been made, upstream, in
`MANIFEST_patha_followup.csv`. This module therefore never chooses a file. It
copies `File Path` verbatim, validates it, and lays the result out on a fixed
slot grid.

THE MANIFEST'S PATH IS FINAL. Some series have duplicates on disk -- 082_S_4224
m24 has a corrupt copy (I1051042) next to the good one (I1051225) the manifest
names. Anything that re-derives a path from a directory walk can land on the
wrong one, so nothing here looks at the directory tree except to confirm the
manifest's own path exists.

THE GRID IS DENSE: every subject gets every slot. A slot is absent for one of
two reasons, and they are kept apart in `missing_reason`:

    MISSING_NOT_IN_COHORT  the manifest has a row saying the scan is unavailable
    NO_ROW                 the manifest has no row at all for that slot (most
                           m24 gaps -- those subjects simply had no m24 visit)

Absent slots are marked `present=False` and nothing else -- no imputation, no
nearest-visit fill, no subject dropped. What to do with a masked slot is the
model's decision, not this module's.

Output, both under dataset/mri_prep_patha/:

    visits.csv          one row per (subject, slot): 4 x n_subjects rows
    visits_summary.csv  one row per subject: how many slots, which pattern

`prep_key` (<PTID>__<slot>) is the file stem stage B writes under, so a visit's
preprocessed volume is found from this table alone.
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m2_index import load_splits  # noqa: E402  -- reuse the one split reader

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The multi-visit extraction lives outside the repo (DUA data, ~hundreds of GB).
PATHA_ROOT = os.environ.get(
    "PATHA_ROOT", os.path.normpath(os.path.join(ROOT, "..", "..", "datasetnew_extracted")))
COHORT = os.path.join(ROOT, "dataset", "mri_manifest", "mri_manifest.csv")
OUT_DIR = os.path.join(ROOT, "dataset", "mri_prep_patha")

# slot -> (index on the grid, nominal months from baseline)
SLOTS = {"bl": (0, 0), "m06": (1, 6), "m12": (2, 12), "m24": (3, 24)}
STATUSES = {"OK", "MISSING_NOT_IN_COHORT"}
COLUMNS = ["Subject ID", "Slot", "VISCODE2", "Match Type", "Scan Date", "File Path",
           "Image ID", "Pipeline", "Size Bytes", "Status"]

# Pinned manifest decisions that must survive any rewrite of this module.
PINNED = {("082_S_4224", "m24"): "I1051225"}

# Visits the manifest lists as OK that are masked here, by explicit decision
# (2026-09-26). Each is named individually so a NEW ordering problem still
# fails validation instead of being masked silently.
#   123_S_4806 m12: exact VISCODE2 match, but scanned 2012-08-27 -- two months
#   after bl and before m06 (2013-01-10). Most likely a VISCODE mislabel
#   upstream; slot order and time order cannot both hold, so the slot is masked.
#   002_S_4237 m06: its NIfTI stores the thick (1.2 mm) axis second, not first
#   (256x169x256), one of only two such scans in the cohort. The m1 chain's
#   affine cannot recover the resulting ~90-degree misorientation (the other
#   such scan, 128_S_4842 bl, failed deterministically: corr_vs_mni 0.15/0.12).
#   m06 is non-critical, so it is masked rather than given a special path.
MASKED = {("123_S_4806", "m12"): "DATE_OUT_OF_ORDER",
          ("002_S_4237", "m06"): "REG_FAILED_LAYOUT"}


def raw_abspath(relpath, patha_root=PATHA_ROOT):
    """Absolute path of a visits.csv `raw_relpath` on this machine."""
    return os.path.join(patha_root, *str(relpath).replace("\\", "/").split("/"))


def read_manifest(path):
    m = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(m.columns) != COLUMNS:
        raise ValueError(f"manifest columns {list(m.columns)} != expected {COLUMNS}")
    bad = sorted(set(m.Slot) - set(SLOTS))
    if bad:
        raise ValueError(f"unknown slots {bad}")
    bad = sorted(set(m.Status) - STATUSES)
    if bad:
        raise ValueError(f"unknown statuses {bad}")
    dup = m[m.duplicated(["Subject ID", "Slot"], keep=False)]
    if len(dup):
        raise ValueError(f"{len(dup)} duplicate (subject, slot) rows, e.g. "
                         f"{dup[['Subject ID', 'Slot']].head(3).values.tolist()}")
    return m


def check_files(ok, root, errors):
    """Every OK row must point at an existing file of the recorded size."""
    abs_path = ok["File Path"].map(lambda p: os.path.join(root, p))
    exists = abs_path.map(os.path.exists)
    for _, r in ok[~exists].iterrows():
        errors.append(f"missing file: {r['Subject ID']}/{r.Slot} -> {r['File Path']}")
    size = abs_path[exists].map(os.path.getsize)
    wrong = ok[exists][size != ok[exists]["Size Bytes"].astype(int)]
    for _, r in wrong.iterrows():
        errors.append(f"size mismatch: {r['Subject ID']}/{r.Slot} "
                      f"expected {r['Size Bytes']} got {os.path.getsize(os.path.join(root, r['File Path']))}")
    return abs_path


def check_headers(paths, errors):
    """--deep: open every NIfTI header, catching truncated or corrupt files."""
    import nibabel as nib
    for key, p in paths.items():
        try:
            shape = nib.load(p).shape
            if len(shape) < 3 or min(shape[:3]) < 16:
                errors.append(f"implausible shape {shape}: {key}")
        except Exception as e:  # noqa: BLE001 -- report every failure, not just the first
            errors.append(f"unreadable header {key}: {type(e).__name__}: {e}")


def build(manifest_path, out_dir=OUT_DIR, deep=False):
    root = os.path.dirname(os.path.abspath(manifest_path))
    m = read_manifest(manifest_path)
    errors, warnings = [], []

    for (ptid, slot), iid in PINNED.items():
        got = m[(m["Subject ID"] == ptid) & (m.Slot == slot)]["Image ID"].tolist()
        if got != [iid]:
            errors.append(f"pinned {ptid}/{slot} should be {iid}, manifest has {got}")

    ok = m[m.Status == "OK"].copy()
    ok["raw_path"] = check_files(ok, root, errors)

    if deep:
        check_headers({f"{r['Subject ID']}/{r.Slot}": r.raw_path
                       for _, r in ok.iterrows() if os.path.exists(r.raw_path)}, errors)

    # dense grid: every subject x every slot, then attach what the manifest has
    subjects = sorted(m["Subject ID"].unique())
    grid = pd.DataFrame([(p, s) for p in subjects for s in SLOTS], columns=["PTID", "slot"])
    m = m.merge(ok[["Subject ID", "Slot", "raw_path"]], on=["Subject ID", "Slot"], how="left")
    df = grid.merge(m.rename(columns={"Subject ID": "PTID", "Slot": "slot"}),
                    on=["PTID", "slot"], how="left")

    df["slot_idx"] = df.slot.map(lambda s: SLOTS[s][0])
    df["months"] = df.slot.map(lambda s: SLOTS[s][1])
    df["present"] = df.Status.eq("OK")
    df["missing_reason"] = df.Status.fillna("NO_ROW")

    # One scan must never fill two slots: it would read as a visit pair with
    # zero progression. It happens when the +/-60-day fallback lands on a scan
    # an exact VISCODE2 match already owns; the exact slot keeps it and the
    # fallback slot is masked. Any other duplicate is unexplained and fails.
    dup = df[df.present & df.duplicated(["PTID", "Image ID"], keep=False)]
    for (ptid, iid), g in dup.groupby(["PTID", "Image ID"]):
        fb = g[g["Match Type"] == "fallback"]
        if len(g) == 2 and len(fb) == 1:
            df.loc[fb.index, ["present", "missing_reason"]] = [False, "DUPLICATE_SCAN"]
        else:
            errors.append(f"{ptid}: {iid} fills slots {list(g.slot)} and no single "
                          f"fallback row explains it")
    for (ptid, slot), reason in MASKED.items():
        hit = (df.PTID == ptid) & (df.slot == slot) & df.present
        if hit.sum() != 1:
            errors.append(f"MASKED entry {ptid}/{slot} no longer matches a present visit")
        df.loc[hit, ["present", "missing_reason"]] = [False, reason]
    df["scan_date"] = pd.to_datetime(df["Scan Date"].replace("", None), format="%Y-%m-%d")
    df.loc[~df.present, "scan_date"] = pd.NaT   # a masked visit carries no date
    first = df[df.present].groupby("PTID").scan_date.min()
    df["days_from_first"] = (df.scan_date - df.PTID.map(first)).dt.days.astype("Int64")
    df["prep_key"] = df.PTID + "__" + df.slot
    # Stored relative to PATHA_ROOT, like m2_index's repo-relative paths, so
    # the table survives the extraction moving; resolve with raw_abspath().
    df["raw_relpath"] = df["File Path"]
    df.loc[~df.present, ["raw_relpath", "prep_key"]] = pd.NA

    # scan dates must strictly increase along the present slots of a subject
    for ptid, g in df[df.present].sort_values("slot_idx").groupby("PTID"):
        if not g.scan_date.is_monotonic_increasing or g.scan_date.duplicated().any():
            errors.append(f"non-increasing scan dates for {ptid}: "
                          f"{list(zip(g.slot, g.scan_date.dt.date.astype(str)))}")

    # cohort metadata and the fixed clinical split
    cohort = pd.read_csv(COHORT, dtype=str).set_index("PTID")
    unknown = sorted(set(subjects) - set(cohort.index))
    if unknown:
        errors.append(f"{len(unknown)} subjects absent from {COHORT}: {unknown[:5]}")
    df["RID"] = df.PTID.map(cohort.RID).astype("Int64")
    df["DX_bl"] = df.PTID.map(cohort.DX_bl)
    df["label"] = df.PTID.map(cohort.label).astype("Int64")
    split_of = load_splits()
    df["split"] = df.RID.map(lambda r: split_of.get(int(r)) if pd.notna(r) else None)
    nosplit = sorted(df[df.split.isna()].PTID.unique())
    if nosplit:
        errors.append(f"{len(nosplit)} subjects have no split: {nosplit[:5]}")

    fallback = df[df["Match Type"].eq("fallback")]
    if len(fallback):
        warnings.append(f"{len(fallback)} visits matched by the +/-60-day date fallback "
                        f"rather than exact VISCODE2")

    if errors:
        print(f"VALIDATION FAILED ({len(errors)} errors):", file=sys.stderr)
        for e in errors[:50]:
            print("  " + e, file=sys.stderr)
        raise SystemExit(1)

    df = df.rename(columns={"Image ID": "image_id", "Pipeline": "pipeline",
                            "Match Type": "match_type", "VISCODE2": "viscode2"})
    df["scan_date"] = df.scan_date.dt.date.astype("string")
    df = df[["RID", "PTID", "slot", "slot_idx", "months", "present", "missing_reason",
             "image_id", "scan_date", "days_from_first", "pipeline", "match_type",
             "viscode2", "raw_relpath", "prep_key", "split", "DX_bl", "label"]]
    df = df.sort_values(["PTID", "slot_idx"]).reset_index(drop=True)
    # blank strings from the manifest's MISSING rows -> real missing values
    df = df.replace({"": pd.NA})

    summ = df.groupby("PTID").agg(
        RID=("RID", "first"), split=("split", "first"), DX_bl=("DX_bl", "first"),
        n_present=("present", "sum"),
        pattern=("present", lambda s: "".join("1" if v else "0" for v in s)),
        has_bl=("present", "first")).reset_index()

    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "visits.csv"), index=False)
    summ.to_csv(os.path.join(out_dir, "visits_summary.csv"), index=False)
    report(df, summ, warnings, out_dir)
    return df, summ


def report(df, summ, warnings, out_dir):
    print(f"subjects {df.PTID.nunique()}   grid rows {len(df)}   present {int(df.present.sum())}")
    print("\nslot x missing_reason:")
    print(pd.crosstab(df.slot, df.missing_reason).reindex(list(SLOTS)).to_string())
    print("\nvisits per subject:")
    print(summ.n_present.value_counts().sort_index().to_string())
    print("\npattern (bl m06 m12 m24):")
    print(summ.pattern.value_counts().to_string())
    print(f"\nsubjects without bl: {int((~summ.has_bl).sum())}")
    print("\nsplit x DX_bl (subjects):")
    print(pd.crosstab(summ.split, summ.DX_bl)
          .reindex(index=["train", "val", "test"], columns=["CN", "EMCI", "LMCI", "AD"])
          .to_string())
    pres = df[df.present]
    print(f"\nactual months from first scan, by slot (median [min, max]):")
    for s, g in pres.groupby("slot", sort=False):
        mo = g.days_from_first.astype(float) / 30.44
        print(f"  {s:<4s} {mo.median():5.1f}  [{mo.min():5.1f}, {mo.max():5.1f}]  n={len(g)}")
    for w in warnings:
        print(f"\nnote: {w}")
    print(f"\nwrote {os.path.join(out_dir, 'visits.csv')}")
    print(f"wrote {os.path.join(out_dir, 'visits_summary.csv')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", default=os.path.join(PATHA_ROOT, "MANIFEST_patha_followup.csv"))
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--deep", action="store_true", help="also open every NIfTI header (needs nibabel)")
    a = ap.parse_args()
    build(a.manifest, a.out_dir, a.deep)


if __name__ == "__main__":
    main()
