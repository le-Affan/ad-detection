"""Run SynthSeg 2.0 (+ cortical parcellation, volumes, QC) on m1 mni/ volumes.

Run with LYP/.venv-synthseg (Python 3.10, tensorflow-cpu 2.12.0, keras 2.12.0,
numpy 1.23.5, nibabel 5.1.0, scipy 1.9.1, surfa 0.4.2: FreeSurfer 7.4.1's build
pins). The tool is FreeSurfer 7.4.1's `mri_synthseg` script, run from a
minimal FREESURFER_HOME (LYP/third_party/fs_synthseg: bin/mri_synthseg plus
models/ with the 2.0 weights and label files from FreeSurfer's git-annex, see
smoke_report.md for hashes). The standalone SynthSeg repo's TF 2.2 stack was
tried first and aborts (std::bad_alloc) in the parcellation network on CPU.

Input choice: the m1 pipeline keeps no native-space intermediate, so the input
is dataset/mri_prep/mni/<PTID>.nii.gz (affine to ICBM152 2009a, template-mask
skull strip, 5-95th percentile truncation, min-max to [0, 1]). Volumes are
therefore in affine-normalised space.

Crop. Every mni/ volume is zero outside the same template brain mask, whose
bounding box is fixed (MASK_BOX, voxels in the 189x233x164 grid). Each input is
cropped to that box plus a 3-voxel margin before SynthSeg: lossless (only
zeros are removed; the affine is updated), and it lets SynthSeg run with a
160x192x160 analysis patch instead of its 192^3 default, which exhausts the
15 GB of RAM here. A centred 160 crop of the uncropped grid would clip the
inferior cerebellum, because the brain sits off-centre in that grid.

Resumable: subjects with a segmentation already on disk are skipped, and the
batch folder is rebuilt with pending subjects only. Each run writes its own
vol_<tag>_<time>.csv / qc_<tag>_<time>.csv; readers concatenate vol_<tag>_*.csv.
Run it alone: at ~11.5 GB peak it leaves no room for other jobs on 15 GB.
Failures are logged to synthseg_failures.csv and never imputed.

Never reads a test subject: --subjects must list clinical-train/val PTIDs
only, and this is asserted against dataset/processed/test_rids.csv.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

import nibabel as nib
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
LYP = os.path.dirname(os.path.dirname(ROOT))
FSHOME = os.path.join(LYP, "third_party", "fs_synthseg")
MNI = os.path.join(ROOT, "dataset", "mri_prep", "mni")
WORK = os.path.join(ROOT, "dataset", "mri_prep", "synthseg")      # git-ignored (DUA data)

MASK_BOX = ((0, 155), (25, 206), (22, 143))                        # template brain mask bbox
MARGIN = 3
PATCH = (160, 192, 160)


def crop(ptid, out_dir):
    dst = os.path.join(out_dir, f"{ptid}.nii.gz")
    if os.path.exists(dst):
        return dst
    img = nib.load(os.path.join(MNI, f"{ptid}.nii.gz"))
    sl = tuple(slice(max(lo - MARGIN, 0), min(hi + MARGIN, n)) for (lo, hi), n in zip(MASK_BOX, img.shape))
    data = img.get_fdata(dtype="float32")
    assert (data != 0).sum() == (data[sl] != 0).sum(), f"{ptid}: signal outside the mask box"
    nib.save(img.slicer[sl], dst)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", required=True, help="CSV with a PTID column")
    ap.add_argument("--tag", required=True, help="batch name for the CSV outputs")
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()

    subs = pd.read_csv(a.subjects)
    test = set(pd.read_csv(os.path.join(ROOT, "dataset", "processed", "test_rids.csv")).RID)
    assert not subs.PTID.str[-4:].astype(int).isin(test).any(), "test subject in the list"

    crop_dir, seg_dir = os.path.join(WORK, "input"), os.path.join(WORK, "seg")
    run_id = f"{a.tag}_{time.strftime('%Y%m%d-%H%M%S')}"          # each run keeps its own CSVs
    batch_in = os.path.join(WORK, f"batch_{a.tag}")
    shutil.rmtree(batch_in, ignore_errors=True)                     # links to pending subjects only
    for d in (crop_dir, seg_dir, batch_in):
        os.makedirs(d, exist_ok=True)

    todo = [p for p in subs.PTID if not os.path.exists(os.path.join(seg_dir, f"{p}_synthseg.nii.gz"))]
    print(f"{len(subs)} subjects, {len(todo)} to run")
    fails = []
    for p in todo:
        try:
            src = crop(p, crop_dir)
            link = os.path.join(batch_in, f"{p}.nii.gz")
            if not os.path.exists(link):
                os.symlink(src, link)
        except Exception as e:                                        # noqa: BLE001
            fails.append({"PTID": p, "stage": "crop", "error": f"{type(e).__name__}: {e}"})
    if not todo:
        return

    t0 = time.time()
    cmd = [sys.executable, os.path.join(FSHOME, "bin", "mri_synthseg"),
           "--i", batch_in, "--o", seg_dir, "--parc", "--cpu", "--threads", str(a.threads),
           "--crop", *map(str, PATCH),
           "--vol", os.path.join(WORK, f"vol_{run_id}.csv"), "--qc", os.path.join(WORK, f"qc_{run_id}.csv")]
    print(" ".join(cmd))
    rc = subprocess.call(cmd, env={**os.environ, "FREESURFER_HOME": FSHOME})
    print(f"SynthSeg exit {rc}, {time.time() - t0:.0f} s for {len(todo)} scans")

    for p in todo:
        if not os.path.exists(os.path.join(seg_dir, f"{p}_synthseg.nii.gz")) and \
                not any(f["PTID"] == p for f in fails):
            fails.append({"PTID": p, "stage": "synthseg", "error": f"no output (exit {rc})"})
    if fails:
        path = os.path.join(HERE, "synthseg_failures.csv")
        pd.DataFrame(fails).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
        print(f"{len(fails)} failures logged")

    # per-scan runtime from output mtimes (SynthSeg writes each scan's segmentation in turn)
    segs = sorted(glob.glob(os.path.join(seg_dir, "*_synthseg.nii.gz")), key=os.path.getmtime)
    times = [os.path.getmtime(s) for s in segs if os.path.basename(s)[:10] in set(todo)]
    if len(times) > 1:
        gaps = pd.Series(times).diff().dropna()
        print(f"per-scan seconds (steady state): median {gaps.median():.1f}, min {gaps.min():.1f}, max {gaps.max():.1f}")


if __name__ == "__main__":
    main()
