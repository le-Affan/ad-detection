"""Path A, stage B: preprocess every present visit with the unchanged m1 chain.

The chain itself is not restated here -- each scan goes through
m1_preprocess.process_one, the same code that produced the baseline cohort, so
a follow-up volume is interchangeable with a baseline one. What differs is only
bookkeeping:

    input    dataset/mri_prep_patha/visits.csv (stage A), present rows only.
             The raw path is the manifest's own, never re-derived.
    output   dataset/mri_prep_patha/{mni,vol96}/<PTID>__<slot>.nii.gz
             dataset/mri_prep_patha/qc/<tag>/<PTID>__<slot>.png
             Keyed by visit, since one subject now has up to four scans, and in
             a separate tree so dataset/mri_prep/ is never written to.

PARALLELISM. ANTs registration is CPU-bound and does not use the GPU. One scan
on all 16 threads takes ~21 s; ITK's scaling flattens well before 16, so
several processes with a few threads each finish the cohort sooner than one
process with all of them. Workers are started with `spawn` and the thread cap
is put in the environment first, so ITK reads it before it initialises. Each
worker loads the template once.

NOTHING FAILS SILENTLY. process_one never raises; a failure comes back as a row
with ok=False and the error text. Every finished scan is appended to the report
CSV as it completes, so an interrupted run keeps what it did, and the run ends
by writing the failures and the QC-suspect scans to their own files and
printing their counts. Scans whose volume already exists are skipped, so
re-running resumes; a failed scan has no volume and is retried automatically.

SUSPECT, NOT FAILED. corr_vs_mni < 0.30 (m1's calibration: a 15-voxel
misregistration scores ~0.37, 30 voxels ~0.22), mask_fill < 0.90, or values
outside [0, 1]. These still produce a volume; they are listed for a look at the
QC render, not excluded.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from p1_visits import OUT_DIR, raw_abspath  # noqa: E402

VISITS = os.path.join(OUT_DIR, "visits.csv")
MNI_DIR = os.path.join(OUT_DIR, "mni")
VOL_DIR = os.path.join(OUT_DIR, "vol96")
QC_DIR = os.path.join(OUT_DIR, "qc")

SUSPECT_CORR = 0.30
SUSPECT_FILL = 0.90

_template = None


def _init_worker():
    global _template
    from m1_preprocess import load_template
    _template = load_template()


def _work(job):
    from m1_preprocess import process_one
    key, path, qc_dir = job
    try:
        res = process_one(path, key, MNI_DIR, VOL_DIR, qc_dir, _template)
    except Exception as e:  # noqa: BLE001 -- e.g. an unreadable header, before process_one's own guard
        res = {"native": "", "ok": False, "seconds": 0.0, "error": f"{type(e).__name__}: {e}"}
    return key, res


def suspect_reason(r):
    why = []
    if r.get("corr_vs_mni", 1.0) < SUSPECT_CORR:
        why.append(f"corr_vs_mni {r['corr_vs_mni']:.3f} < {SUSPECT_CORR}")
    if r.get("mask_fill", 1.0) < SUSPECT_FILL:
        why.append(f"mask_fill {r['mask_fill']:.3f} < {SUSPECT_FILL}")
    if r.get("in_unit_range") is False:
        why.append(f"range [{r['min']:.3f}, {r['max']:.3f}] outside [0, 1]")
    return "; ".join(why)


def run(keys=None, workers=4, threads=4, force=False, qc=True, tag="patha", limit=0):
    v = pd.read_csv(VISITS)
    todo = v[v.present].copy()
    if keys:
        want = set(keys)
        todo = todo[todo.prep_key.isin(want) | todo.PTID.isin(want)]
        unknown = want - set(todo.prep_key) - set(todo.PTID)
        if unknown:
            raise SystemExit(f"not present visits or subjects: {sorted(unknown)}")
    todo = todo.sort_values(["PTID", "slot_idx"])
    for d in (MNI_DIR, VOL_DIR):
        os.makedirs(d, exist_ok=True)
    qc_dir = os.path.join(QC_DIR, tag) if qc else None
    if qc_dir:
        os.makedirs(qc_dir, exist_ok=True)

    done = todo.prep_key.map(lambda k: os.path.exists(os.path.join(VOL_DIR, f"{k}.nii.gz")))
    if not force:
        print(f"{int(done.sum())} of {len(todo)} already preprocessed, skipped")
        todo = todo[~done]
    if limit:
        todo = todo.head(limit)
    if todo.empty:
        print("nothing to do")
        return None
    info = todo.set_index("prep_key")
    jobs = [(k, raw_abspath(p), qc_dir) for k, p in zip(todo.prep_key, todo.raw_relpath)]

    report = os.path.join(OUT_DIR, f"p2_report_{tag}.csv")
    with open(os.path.join(OUT_DIR, f"p2_params_{tag}.json"), "w") as f:
        from m1_preprocess import chain_params
        json.dump({**chain_params(), "workers": workers, "itk_threads_per_worker": threads,
                   "suspect": {"corr_vs_mni_lt": SUSPECT_CORR, "mask_fill_lt": SUSPECT_FILL}},
                  f, indent=2)

    # ITK reads this when it first initialises, so it must be in the
    # environment before a spawned worker imports ants.
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(threads)
    print(f"{len(jobs)} scans, {workers} workers x {threads} ITK threads -> {report}")

    rows, t0 = [], time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init_worker) as pool:
        for n, (key, res) in enumerate(pool.imap_unordered(_work, jobs), 1):
            r = info.loc[key]
            row = {"prep_key": key, "RID": int(r.RID), "PTID": r.PTID, "slot": r.slot,
                   "image_id": r.image_id, "pipeline": r.pipeline, "DX_bl": r.DX_bl, **res}
            row["suspect"] = suspect_reason(res) if res["ok"] else ""
            rows.append(row)
            # append as we go: an interrupted run keeps every finished row
            pd.DataFrame([row]).to_csv(report, mode="a", index=False,
                                       header=not os.path.exists(report))
            el = time.time() - t0
            eta = el / n * (len(jobs) - n)
            if res["ok"]:
                msg = (f"{res['seconds']:5.1f}s fill={res['mask_fill']:.3f} "
                       f"corr={res['corr_vs_mni']:.3f}" + (f"  SUSPECT: {row['suspect']}" if row["suspect"] else ""))
            else:
                msg = f"FAILED -- {res['error']}"
            print(f"[{n}/{len(jobs)} {el / 60:5.1f}m eta {eta / 60:5.1f}m] {key:<18s} {msg}", flush=True)

    df = pd.DataFrame(rows)
    fails = df[~df.ok]
    sus = df[df.ok & (df.suspect != "")]
    fails.to_csv(os.path.join(OUT_DIR, f"p2_failures_{tag}.csv"), index=False)
    sus.to_csv(os.path.join(OUT_DIR, f"p2_suspect_{tag}.csv"), index=False)
    wall = time.time() - t0
    print(f"\n{len(df)} scans in {wall / 60:.1f} min ({wall / len(df):.1f} s/scan wall)  "
          f"ok {int(df.ok.sum())}  FAILED {len(fails)}  suspect {len(sus)}")
    if len(fails):
        print(fails[["prep_key", "error"]].to_string(index=False))
    if len(sus):
        print(sus[["prep_key", "suspect"]].to_string(index=False))
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--keys", help="comma-separated prep_keys (PTID__slot) or PTIDs; default all present")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4, help="ITK threads per worker")
    ap.add_argument("--limit", type=int, default=0, help="process at most N scans")
    ap.add_argument("--force", action="store_true", help="redo scans that already have output")
    ap.add_argument("--no-qc", action="store_true", help="skip QC PNGs")
    ap.add_argument("--tag", default="patha", help="names the report and QC subdirectory")
    a = ap.parse_args()
    run(keys=[k.strip() for k in a.keys.split(",")] if a.keys else None,
        workers=a.workers, threads=a.threads, force=a.force, qc=not a.no_qc,
        tag=a.tag, limit=a.limit)


if __name__ == "__main__":
    main()
