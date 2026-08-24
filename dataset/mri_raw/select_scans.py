"""
Pick exactly one baseline T1 NIfTI per subject from the IDA advanced-search
export, using the agreed selection rule. Read-only: reads the IDA export and
dataset/mri_manifest/mri_manifest.csv, writes only into dataset/mri_raw/.

Rule (in order):
  1. Pipeline: prefer "MT1; GradWarp; N3m", else "MT1; N3m".
     Exclude "Spatially Normalized, Masked and N3 corrected T1 image"
     (already skull-stripped + MNI-normalized) and the HHP/HarP boutique
     pipelines -- mixing those in would put a preprocessing covariate that
     correlates with site straight into the training data.
  2. Protocol: prefer non-accelerated (plain MPRAGE / IR-SPGR) over
     accelerated (GRAPPA2 / SENSE2 / "Accelerated" / "w/acceleration").
  3. Tie: scan whose Study Date is closest to MRI_EXAMDATE in the manifest.
  4. Final determinism guard: lowest Image ID, so reruns are identical.
"""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEARCH = os.path.join(ROOT, "dataset", "mri_raw", "ida_search_results_baselineT1.csv")
MANIFEST = os.path.join(ROOT, "dataset", "mri_manifest", "mri_manifest.csv")
OUT = os.path.join(ROOT, "dataset", "mri_raw", "selected_scans.csv")

KEEP_PIPELINES = {"MT1; GradWarp; N3m": 0, "MT1; N3m": 1}   # value = rank, lower wins
ACCEL_TOKENS = ("grappa", "sense", "accelerated", "acceleration")


def main():
    d = pd.read_csv(SEARCH, dtype=str)
    man = pd.read_csv(MANIFEST, dtype=str)

    d = d[d["Type"] == "Pre-processed"].copy()
    parts = d["Description"].str.split("<-", n=1, expand=True)
    d["pipeline"] = parts[0].str.strip()
    d["protocol"] = parts[1].str.strip() if parts.shape[1] > 1 else None

    excluded = d[~d["pipeline"].isin(KEEP_PIPELINES)]
    d = d[d["pipeline"].isin(KEEP_PIPELINES)].copy()

    d["pipeline_rank"] = d["pipeline"].map(KEEP_PIPELINES)
    d["accelerated"] = d["protocol"].fillna("").str.lower().apply(
        lambda s: any(t in s for t in ACCEL_TOKENS))
    d["protocol_rank"] = d["accelerated"].astype(int)

    d["scan_date"] = pd.to_datetime(d["Study Date"], format="%m/%d/%Y", errors="coerce")
    ref = man.assign(ref=pd.to_datetime(man["MRI_EXAMDATE"], errors="coerce")) \
             .set_index("PTID")["ref"]
    d["ref_date"] = d["Subject ID"].map(ref)
    # Subjects with no MRI_EXAMDATE fall back to their own earliest scan, so the
    # tiebreak stays defined rather than going NaN and sorting unpredictably.
    fallback = d.groupby("Subject ID")["scan_date"].transform("min")
    d["ref_date"] = d["ref_date"].fillna(fallback)
    d["date_gap"] = (d["scan_date"] - d["ref_date"]).abs().dt.days
    d["date_gap"] = d["date_gap"].fillna(10**6)

    d["image_id_num"] = pd.to_numeric(d["Image ID"], errors="coerce").fillna(10**12)

    d = d.sort_values(
        ["Subject ID", "pipeline_rank", "protocol_rank", "date_gap", "image_id_num"],
        kind="mergesort")
    win = d.drop_duplicates(subset="Subject ID", keep="first").copy()

    info = man.set_index("PTID")
    win["RID"] = win["Subject ID"].map(info["RID"])
    win["DX_bl"] = win["Subject ID"].map(info["DX_bl"])
    win["PHASE"] = win["Subject ID"].map(info["PHASE"])
    win["MRI_EXAMDATE"] = win["Subject ID"].map(info["MRI_EXAMDATE"])
    win["n_candidates"] = win["Subject ID"].map(d.groupby("Subject ID").size())

    out = win[[
        "RID", "Subject ID", "DX_bl", "PHASE", "Visit", "Image ID",
        "pipeline", "protocol", "accelerated", "Study Date", "MRI_EXAMDATE",
        "date_gap", "n_candidates", "Description",
    ]].rename(columns={"Subject ID": "PTID", "Study Date": "SCAN_DATE"})
    out = out.sort_values("RID", key=lambda s: s.astype(int))
    out.to_csv(OUT, index=False)

    print(f"pre-processed rows considered : {len(d)}")
    print(f"rows excluded by pipeline rule : {len(excluded)}")
    if len(excluded):
        print("   " + ", ".join(f"{k} ({v})" for k, v in
                                excluded['pipeline'].value_counts().items()))
    print(f"subjects selected              : {len(out)}")
    print(f"\npipeline chosen:\n{out['pipeline'].value_counts().to_string()}")
    print(f"\naccelerated protocol chosen: {int(out['accelerated'].sum())} "
          f"of {len(out)}")
    print(f"\ncandidates per subject:\n"
          f"{out['n_candidates'].value_counts().sort_index().to_string()}")
    print(f"\nwrote {OUT}")

    ids = out["Image ID"].tolist()
    idfile = os.path.join(ROOT, "dataset", "mri_raw", "selected_image_ids.txt")
    with open(idfile, "w") as fh:
        fh.write(",".join(ids))
    print(f"wrote {idfile}  ({len(ids)} ids, {len(','.join(ids))} chars)")


if __name__ == "__main__":
    main()
