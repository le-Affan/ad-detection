"""
Unpack the IDA zips into dataset/mri_raw/<PTID>/ (one NIfTI per subject) and
verify the result against the manifest and the selection.

Download and verification only -- no preprocessing, no conversion, no model code.
Reads dataset/mri_manifest/mri_manifest.csv and dataset/mri_raw/selected_scans.csv;
writes only inside dataset/mri_raw/.

IDA zips lay files out as  ADNI/<PTID>/<Description>/<date>/<ImageID>/<file>.nii
so the subject and image id are both recoverable from the archive path; each
extracted file is matched back to the selection by Image ID, which is the key
we actually chose on, rather than by filename text.
"""
import os
import re
import shutil
import zipfile

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, "dataset", "mri_raw")
ZIPS = os.path.join(RAW, "_zips")
STAGE = os.path.join(RAW, "_stage")
SEL = os.path.join(RAW, "selected_scans.csv")
MANIFEST = os.path.join(ROOT, "dataset", "mri_manifest", "mri_manifest.csv")

IMGID_RE = re.compile(r"(?:^|[/_])I(\d{4,})(?:[/_.]|$)")
PTID_RE = re.compile(r"(\d{3}_S_\d{4})")
def extract():
    zips = sorted(z for z in os.listdir(ZIPS) if z.endswith(".zip"))
    if not zips:
        raise SystemExit("no zips found in " + ZIPS)
    os.makedirs(STAGE, exist_ok=True)
    total = 0
    for z in zips:
        path = os.path.join(ZIPS, z)
        with zipfile.ZipFile(path) as zf:
            members = [m for m in zf.namelist() if m.lower().endswith((".nii", ".nii.gz"))]
            zf.extractall(STAGE, members=members)
            total += len(members)
        print(f"  {z}: {len(members)} image files")
    print(f"extracted {total} image files")
    return total


def image_id_from(path):
    m = IMGID_RE.search(path.replace("\\", "/"))
    return m.group(1) if m else None


def ptid_from(path):
    m = PTID_RE.search(path.replace("\\", "/"))
    return m.group(1) if m else None


def organise():
    """Match each extracted file back to a selected subject.

    Image ID is the key the selection was actually made on, so it wins.
    PTID parsed from the archive path is the fallback, because IDA has used
    more than one on-disk layout over the years and a filename that omits the
    image id is still unambiguous if the subject directory is present.
    """
    sel = pd.read_csv(SEL, dtype=str)
    by_imgid = sel.set_index("Image ID")["PTID"].to_dict()
    wanted = set(sel.PTID)

    placed, unmatched, how = {}, [], {"image_id": 0, "ptid_path": 0}
    for dirpath, _dirs, files in os.walk(STAGE):
        for fn in files:
            if not fn.lower().endswith((".nii", ".nii.gz")):
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, STAGE)

            iid = image_id_from(rel) or image_id_from(fn)
            ptid = by_imgid.get(iid) if iid else None
            if ptid is not None:
                how["image_id"] += 1
            else:
                cand = ptid_from(rel)
                if cand in wanted:
                    ptid, how["ptid_path"] = cand, how["ptid_path"] + 1

            if ptid is None:
                unmatched.append(rel)
                continue

            dest_dir = os.path.join(RAW, ptid)
            os.makedirs(dest_dir, exist_ok=True)
            dest = os.path.join(dest_dir, fn)
            if os.path.exists(dest):
                dest = os.path.join(dest_dir, f"{iid or 'dup'}_{fn}")
            shutil.move(full, dest)
            placed.setdefault(ptid, []).append(dest)

    print(f"  matched by Image ID: {how['image_id']}, by PTID path: {how['ptid_path']}")
    if unmatched:
        print(f"  UNMATCHED ({len(unmatched)}), first 5:")
        for u in unmatched[:5]:
            print("    " + u)
    return placed, unmatched


def verify(placed, unmatched):
    sel = pd.read_csv(SEL, dtype=str)
    man = pd.read_csv(MANIFEST, dtype=str)
    info = man.set_index("PTID")

    expected = set(sel.PTID)
    got = {p for p, f in placed.items() if f}
    missing = sorted(expected - got, key=lambda p: int(info.loc[p, "RID"]))
    extra = sorted(got - expected)

    print("\n=== VERIFICATION ===")
    print(f"expected subjects (obtainable) : {len(expected)}")
    print(f"subjects with >=1 file on disk  : {len(got)}")
    print(f"subjects MISSING a file         : {len(missing)}")
    print(f"unexpected subjects             : {len(extra)}")
    print(f"files that matched no selection : {len(unmatched)}")

    multi = {p: f for p, f in placed.items() if len(f) > 1}
    print(f"subjects with >1 file           : {len(multi)}")

    if missing:
        print("\n--- MISSING (no file downloaded) ---")
        for p in missing:
            r = info.loc[p]
            print(f"  RID {r.RID:>5s}  {p:<11s} {r.DX_bl:<4s} {r.PHASE}")

    excluded = sorted(set(man.PTID) - expected, key=lambda p: int(info.loc[p, "RID"]))
    print("\n--- EXCLUDED BY DESIGN (no baseline structural T1 in IDA) ---")
    for p in excluded:
        r = info.loc[p]
        print(f"  RID {r.RID:>5s}  {p:<11s} {r.DX_bl:<4s} {r.PHASE}  (not retried)")

    sel["downloaded"] = sel.PTID.isin(got)
    sel["file"] = sel.PTID.map(
        lambda p: os.path.relpath(placed[p][0], ROOT) if placed.get(p) else "")
    sel["bytes"] = sel.PTID.map(
        lambda p: os.path.getsize(placed[p][0]) if placed.get(p) else 0)
    out = os.path.join(RAW, "downloaded_manifest.csv")
    sel.to_csv(out, index=False)
    print(f"\nwrote {out}")

    print("\n--- pipeline / protocol of what landed on disk ---")
    dl = sel[sel.downloaded]
    print(dl.pipeline.value_counts().to_string())
    print("\ntop protocols:")
    print(dl.protocol.value_counts().head(12).to_string())
    print(f"\naccelerated protocol used: {int(dl.accelerated.astype(str).eq('True').sum())}")
    print(f"total bytes on disk: {dl.bytes.sum()/2**30:.2f} GiB")
    return missing


def main():
    print("=== extracting ===")
    extract()
    print("\n=== organising into dataset/mri_raw/<PTID>/ ===")
    placed, unmatched = organise()
    print(f"placed files for {len(placed)} subjects")
    verify(placed, unmatched)
    if os.path.isdir(STAGE) and not any(
            f for _d, _s, fs in os.walk(STAGE) for f in fs):
        shutil.rmtree(STAGE, ignore_errors=True)
        print("removed empty staging dir")


if __name__ == "__main__":
    main()
