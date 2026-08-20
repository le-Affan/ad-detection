"""Preprocess raw ADNI T1 NIfTIs into model-ready volumes in MNI space.

Chain, in order:

    1. N4 bias field correction        (native space -- see note below)
    2. reorient to IAL
    3. resample to 1.0 x 1.0 x 1.2 mm
    4. Affine registration to the MNI152 template
    5. multiply by the MNI brain mask
    6. truncate intensity to the 5th-95th percentile
    7. min-max normalise to [0, 1]
    8. downsample to 96 x 96 x 96

WHY N4 RUNS FIRST. The Toqa277 reference applies N4 *after* registration and
masking. The bias field is a native-space scanner artefact, so correcting it
after the image has been interpolated into template space both smears the
field and lets a bias-corrupted image drive the registration. Running it first
is the standard order and is what this module does.

TEMPLATE, AND WHY NOT THE BUNDLED ONE. ANTsPy's bundled `mni` is MNI152 1mm
geometry but is already SKULL-STRIPPED, and so is nilearn's
`load_mni152_template`. Registering our whole-head scans to a brain-only
template fails in a specific, measurable way: the affine shrinks the entire
head to fit the template's brain, leaving skull and scalp inside the brain
mask and the brain itself 10-15% undersized. Measured against a known-good
affine (Colin27 to template, r = 0.61), that route scored r = 0.20, which the
same yardstick puts at roughly a 30-voxel misregistration.

This module therefore uses the whole-head ICBM152 2009a T1 plus its published
brain mask -- the same pairing the reference gets from FSL's MNI152_T1_1mm.nii
and MNI152_T1_1mm_Brain_Mask.nii. nilearn fetches and caches them.

Registration then runs in two stages:

    1. whole head -> whole head, so skull matches skull
    2. mask the result, then refine brain -> brain

Stage 1 is what fixes the problem (r 0.20 -> 0.39); stage 2 removes any
residual pull from the skull on the final fit (r -> 0.42).

A NOTE ON READING THAT NUMBER. r does not approach 1.0 even when registration
is perfect, because an affine cannot match individual anatomy to a smooth
population template -- and it must not. An atrophied brain with enlarged
ventricles legitimately scores lower than a healthy one, and that difference
is the signal the classifier is meant to find. Use r to catch gross failures,
not to rank subjects.

INTERPOLATION. `ants.resample_image` defaults to interp_type=1, which is
NEAREST NEIGHBOUR -- the reference's downsample step inherits that default and
aliases badly going from ~182^3 to 64^3. Every resample here passes an explicit
interpolator: b-spline for the upstream 1.2mm resample, and linear for the
final downsample. Linear is required at that last step because it is a convex
combination and so cannot overshoot the [0, 1] range established in step 7,
whereas b-spline and sinc both ring past it.

Two artefacts are written per subject:

    dataset/mri_prep/mni/<PTID>.nii.gz    full-res MNI space, post-step 7
    dataset/mri_prep/vol96/<PTID>.nii.gz  the 96^3 model input, post-step 8

The full-res volume is kept because registration is by far the most expensive
and least reversible part of the chain; keeping it means changing the model
input size is a cheap resample rather than a full re-registration.

The subject list is read from disk at run time -- whatever
`downloaded_manifest.csv` marks as downloaded -- and subjects whose output
already exists are skipped. Re-running as more scans arrive picks up only the
new ones, with no code change.
"""
import argparse
import json
import os
import time

import ants
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, "dataset", "mri_raw")
MANIFEST = os.path.join(RAW, "downloaded_manifest.csv")
PREP = os.path.join(ROOT, "dataset", "mri_prep")
MNI_DIR = os.path.join(PREP, "mni")
VOL_DIR = os.path.join(PREP, "vol96")
QC_DIR = os.path.join(PREP, "qc")

WORK_SPACING = (1.0, 1.0, 1.2)
ORIENTATION = "IAL"
TRUNC_LO, TRUNC_HI = 0.05, 0.95
OUT_SIZE = 96

LINEAR, BSPLINE = 0, 4


def load_template(size=OUT_SIZE):
    """Fetch the whole-head ICBM152 template and its brain mask.

    nilearn caches these under ~/nilearn_data after the first call, so only the
    first run touches the network. Returns the head and mask on the working
    grid, plus the masked head and mask at output scale for QC.
    """
    from nilearn import datasets
    icbm = datasets.fetch_icbm152_2009()

    head = ants.image_read(icbm["t1"], reorient=ORIENTATION)
    mask = ants.image_read(icbm["mask"], reorient=ORIENTATION)
    head_w = ants.resample_image(head, WORK_SPACING, False, BSPLINE)
    mask_w = ants.threshold_image(
        ants.resample_image(mask, WORK_SPACING, False, LINEAR), 0.5, 1e12)
    brain_w = head_w * mask_w
    tmpl_o = ants.resample_image(brain_w, (size,) * 3, True, LINEAR)
    mask_o = ants.resample_image(mask_w, (size,) * 3, True, LINEAR)
    return head_w, mask_w, brain_w, tmpl_o, mask_o


def preprocess(path, head_w, mask_w, brain_w, size=OUT_SIZE):
    """Run the full chain on one image, returning (full_res, downsampled).

    Both returned images are already normalised to [0, 1]; the downsample is
    linear so the range survives it.
    """
    img = ants.image_read(path)

    # 1. N4 in native space, before any interpolation touches the bias field.
    img = ants.n4_bias_field_correction(img)

    # 2-3. into a common orientation and voxel grid
    img = ants.reorient_image2(img, ORIENTATION)
    img = ants.resample_image(img, WORK_SPACING, False, BSPLINE)

    # 4a. whole head to whole head, so the skull has something to match
    aligned = ants.registration(
        fixed=head_w, moving=img, type_of_transform="Affine")["warpedmovout"]
    # 4b. strip, then refine brain to brain so the skull stops influencing the fit
    aligned = ants.registration(
        fixed=brain_w, moving=aligned * mask_w,
        type_of_transform="Affine")["warpedmovout"]

    # 5. brain mask
    brain = aligned * mask_w

    # 6. percentile truncation, 7. min-max to [0, 1]
    brain = ants.iMath(brain, "TruncateIntensity", TRUNC_LO, TRUNC_HI)
    arr = brain.numpy()
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        raise ValueError(f"degenerate intensity range after masking: [{lo}, {hi}]")
    full = brain.new_image_like((arr - lo) / (hi - lo))

    # 8. downsample, linear so [0, 1] cannot be overshot
    small = ants.resample_image(full, (size,) * 3, True, LINEAR)
    return full, small


def qc_metrics(small, tmpl_o, mask_o):
    """Alignment and range diagnostics for one preprocessed volume.

    corr_vs_mni is the alignment check: intensity correlation against the
    template inside the brain mask. Calibration on this data -- a known-good
    affine of a different real brain scores 0.61, a deliberate 15-voxel
    misalignment 0.37, and 30 voxels 0.22 -- so treat < ~0.30 as suspect and
    investigate the QC image rather than trusting the number alone.

    mask_fill is NOT an independent alignment check and must not be read as
    one. The volume has already been multiplied by the mask, so it can only
    report how much of the mask ended up non-zero; it catches a blank or
    badly undersized volume and nothing subtler.
    """
    a = small.numpy()
    m = mask_o.numpy() > 0.5
    b = a > 1e-6

    x, y = a[m], tmpl_o.numpy()[m]
    corr = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")

    inside = a[m]
    return {
        "shape": "x".join(str(s) for s in a.shape),
        "min": float(a.min()), "max": float(a.max()),
        "mean": float(a.mean()), "std": float(a.std()),
        "frac_zero": float((a == 0).mean()),
        "brain_mean": float(inside.mean()), "brain_std": float(inside.std()),
        "mask_fill": float((b & m).sum()) / float(max(m.sum(), 1)),
        "frac_outside_mask": float((b & ~m).sum()) / float(max(b.sum(), 1)),
        "corr_vs_mni": corr,
        "in_unit_range": bool(a.min() >= -1e-6 and a.max() <= 1 + 1e-6),
    }


def save_qc_png(ptid, small, mask_o, out_dir):
    """Three orthogonal slices with the MNI brain mask outline drawn on top.

    The contour is the point of the figure: alignment is judged by whether the
    subject's brain edge tracks the template outline, which a bare slice cannot
    show.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a, m = small.numpy(), mask_o.numpy()
    fig, ax = plt.subplots(1, 3, figsize=(11, 4))
    for i in range(3):
        sl = [slice(None)] * 3
        sl[i] = a.shape[i] // 2
        ax[i].imshow(np.rot90(a[tuple(sl)]), cmap="gray", vmin=0, vmax=1)
        ax[i].contour(np.rot90(m[tuple(sl)]), levels=[0.5], colors="r", linewidths=0.7)
        # IAL: axis 0 is inferior-superior, axis 1 anterior-posterior,
        # axis 2 left-right, so fixing each axis gives axial/coronal/sagittal.
        ax[i].set_title(["axial", "coronal", "sagittal"][i], fontsize=9)
        ax[i].axis("off")
    fig.suptitle(f"{ptid}  (red = MNI brain mask)", fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{ptid}.png")
    fig.savefig(path, dpi=95)
    plt.close(fig)
    return path


def subjects_on_disk():
    """Subjects with a raw NIfTI present, read from the manifest at run time."""
    dm = pd.read_csv(MANIFEST, dtype=str)
    dl = dm[dm.downloaded == "True"].copy()
    dl["path"] = dl.file.map(lambda f: os.path.join(ROOT, f))
    return dl[["RID", "PTID", "DX_bl", "path"]].sort_values("PTID").reset_index(drop=True)


def run(ptids=None, size=OUT_SIZE, force=False, qc=True, tag="qc"):
    os.makedirs(MNI_DIR, exist_ok=True)
    os.makedirs(VOL_DIR, exist_ok=True)
    qc_dir = os.path.join(QC_DIR, tag)
    if qc:
        os.makedirs(qc_dir, exist_ok=True)

    todo = subjects_on_disk()
    if ptids:
        todo = todo[todo.PTID.isin(ptids)].reset_index(drop=True)
        found = set(todo.PTID)
        for p in ptids:
            if p not in found:
                print(f"  WARNING: {p} requested but not on disk")

    print(f"loading template (output {size}^3) ...")
    head_w, mask_w, brain_w, tmpl_o, mask_o = load_template(size)
    print(f"  working grid {head_w.shape} @ {tuple(round(s, 3) for s in head_w.spacing)}"
          f"  brain voxels {int((mask_w.numpy() > 0.5).sum())}")

    rows = []
    for i, r in todo.iterrows():
        out_small = os.path.join(VOL_DIR, f"{r.PTID}.nii.gz")
        out_full = os.path.join(MNI_DIR, f"{r.PTID}.nii.gz")
        if os.path.exists(out_small) and not force:
            print(f"[{i + 1}/{len(todo)}] {r.PTID}  skip (exists)")
            continue

        native = ants.image_header_info(r.path)["dimensions"]
        t0 = time.time()
        try:
            full, small = preprocess(r.path, head_w, mask_w, brain_w, size)
        except Exception as e:
            dt = time.time() - t0
            print(f"[{i + 1}/{len(todo)}] {r.PTID}  FAILED in {dt:.1f}s -- {type(e).__name__}: {e}")
            rows.append({"RID": r.RID, "PTID": r.PTID, "DX_bl": r.DX_bl,
                         "native": "x".join(str(s) for s in native),
                         "ok": False, "seconds": round(dt, 1),
                         "error": f"{type(e).__name__}: {e}"})
            continue

        # Write atomically so an interrupted run cannot leave a half-written
        # volume that a later run would skip as "already done". The temp name
        # has to keep the .nii.gz suffix -- ITK picks its writer from the
        # extension and refuses anything it does not recognise.
        for img, dest in ((full, out_full), (small, out_small)):
            tmp = os.path.join(os.path.dirname(dest), f".tmp_{os.path.basename(dest)}")
            ants.image_write(img, tmp)
            os.replace(tmp, dest)

        dt = time.time() - t0
        met = qc_metrics(small, tmpl_o, mask_o)
        png = save_qc_png(r.PTID, small, mask_o, qc_dir) if qc else ""
        rows.append({"RID": r.RID, "PTID": r.PTID, "DX_bl": r.DX_bl,
                     "native": "x".join(str(s) for s in native),
                     "ok": True, "seconds": round(dt, 1), "error": "",
                     **met, "png": os.path.relpath(png, ROOT) if png else ""})
        print(f"[{i + 1}/{len(todo)}] {r.PTID}  {r.DX_bl:<4s} "
              f"{rows[-1]['native']:>12s} -> {met['shape']}  {dt:6.1f}s  "
              f"fill={met['mask_fill']:.3f} corr={met['corr_vs_mni']:.3f} "
              f"range=[{met['min']:.3f},{met['max']:.3f}]")

    if not rows:
        print("nothing to do")
        return None

    df = pd.DataFrame(rows)
    out = os.path.join(PREP, f"m1_report_{tag}.csv")
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")

    with open(os.path.join(PREP, f"m1_params_{tag}.json"), "w") as f:
        json.dump({"orientation": ORIENTATION, "work_spacing": WORK_SPACING,
                   "transform": "Affine x2 (head->head, then brain->brain)", "truncate": [TRUNC_LO, TRUNC_HI],
                   "normalise": "min-max to [0,1] before downsample",
                   "downsample": {"size": size, "interp": "linear"},
                   "n4": "first, in native space",
                   "template": "ICBM152 2009a whole-head T1 + published brain mask (via nilearn)"},
                  f, indent=2)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ptids", help="comma-separated PTIDs; default = everything on disk")
    ap.add_argument("--size", type=int, default=OUT_SIZE)
    ap.add_argument("--force", action="store_true", help="redo subjects that already have output")
    ap.add_argument("--no-qc", action="store_true", help="skip QC PNGs")
    ap.add_argument("--tag", default="qc", help="names the report and QC subdirectory")
    a = ap.parse_args()
    run(ptids=[p.strip() for p in a.ptids.split(",")] if a.ptids else None,
        size=a.size, force=a.force, qc=not a.no_qc, tag=a.tag)


if __name__ == "__main__":
    main()
