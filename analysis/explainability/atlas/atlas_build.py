"""Atlas, step 3 of 3 (run with LYP/.venv-mri): the 14-region atlas on the CNN's 96^3 grid, and its QC.

Input: atlas/work/template_synthseg.nii.gz (SynthSeg 2.0 --parc on the masked ICBM152 2009a template in the
m1 working grid; step 2, run alone in LYP/.venv-synthseg).
  1. label map -> m1 working grid (genericLabel, through world coordinates);
  2. labels grouped into the 14 pre-registered regions (preregistration.md §2, Appendix A);
  3. one-hot per region (plus background) -> 96^3 with m1's own step-8 call
     (ants.resample_image(img, (96,)*3, True, LINEAR)), argmax; restricted to the 96^3 template mask (> 0.5);
     mask voxels whose argmax is background become "other".
QC (pre-registered): Dice(atlas brain, template mask) >= 0.90 at 96^3. Fit to individual anatomy: per-region
Dice of the template atlas against the 22 smoke subjects' existing SynthSeg segmentations (all train), processed
identically; reported as medians (aggregate only).
Outputs (committed; template-derived, no subject data): atlas96.nii.gz, atlas96_fractional.npz, regions.json,
atlas_qc.png, ../results/atlas_qc.csv.
"""
import json
import os
import sys

import ants
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import common as C  # noqa: E402

sys.path.insert(0, os.path.join(C.ROOT, "src", "mri"))
import m1_preprocess as M1  # noqa: E402

WORK = os.path.join(HERE, "work")
SMOKE_SEG = os.path.join(C.ROOT, "dataset", "mri_prep", "synthseg", "seg")
REGIONS = ["hippocampus", "amygdala", "ventricles", "thalamus", "basal ganglia", "cerebellum", "brainstem",
           "cerebral white matter", "frontal lobe", "temporal lobe", "parietal lobe", "occipital lobe", "other"]
# FreeSurfer / SynthSeg labels (left, right)
SUBCORT = {"hippocampus": (17, 53), "amygdala": (18, 54), "ventricles": (4, 5, 14, 15, 43, 44),
           "thalamus": (10, 49), "basal ganglia": (11, 12, 13, 26, 50, 51, 52, 58),
           "cerebellum": (7, 8, 46, 47), "brainstem": (16,), "cerebral white matter": (2, 41),
           "other": (24, 28, 60, 31, 63, 3, 42)}
LOBES = {"frontal lobe": ["superiorfrontal", "rostralmiddlefrontal", "caudalmiddlefrontal", "parsopercularis",
                          "parstriangularis", "parsorbitalis", "lateralorbitofrontal", "medialorbitofrontal",
                          "precentral", "paracentral", "frontalpole"],
         "parietal lobe": ["superiorparietal", "inferiorparietal", "supramarginal", "postcentral", "precuneus"],
         "temporal lobe": ["superiortemporal", "middletemporal", "inferiortemporal", "bankssts", "fusiform",
                           "transversetemporal", "entorhinal", "temporalpole", "parahippocampal"],
         "occipital lobe": ["lateraloccipital", "lingual", "cuneus", "pericalcarine"],
         "other": ["rostralanteriorcingulate", "caudalanteriorcingulate", "posteriorcingulate", "isthmuscingulate",
                   "insula"]}


def label_to_region():
    names = np.load(os.path.join(C.LYP, "third_party", "fs_synthseg", "models", "synthseg_parcellation_names.npy"))
    labs = np.load(os.path.join(C.LYP, "third_party", "fs_synthseg", "models", "synthseg_parcellation_labels.npy"))
    m = {}
    for r, ls in SUBCORT.items():
        for lab in ls:
            m[lab] = REGIONS.index(r) + 1
    dk = {}
    for lab, nm in zip(labs, names):
        nm = str(nm)
        if nm.startswith("ctx-"):
            dk[int(lab)] = nm.split("-", 2)[2]
    for lab, parcel in dk.items():
        hit = [r for r, ps in LOBES.items() if parcel in ps]
        assert len(hit) == 1, (lab, parcel)
        m[lab] = REGIONS.index(hit[0]) + 1
    return m                                                       # label -> 1..13 ; 0 = background


def to_96(seg_path, target_w, l2r):
    """SynthSeg label map -> region index map on the 96^3 grid (0 = outside brain), plus fractional maps."""
    seg = ants.image_read(seg_path)
    seg_w = ants.resample_image_to_target(seg, target_w, interp_type="genericLabel").numpy().astype(int)
    unknown = sorted(set(np.unique(seg_w)) - set(l2r) - {0})
    reg_w = np.zeros_like(seg_w)
    for lab, r in l2r.items():
        reg_w[seg_w == lab] = r
    reg_w[np.isin(seg_w, unknown)] = REGIONS.index("other") + 1     # any unlisted label: "other"
    frac = []
    for r in range(len(REGIONS) + 1):
        img = target_w.new_image_like((reg_w == r).astype(np.float32))
        frac.append(ants.resample_image(img, (96,) * 3, True, M1.LINEAR).numpy())
    frac = np.stack(frac)                                           # (14, 96, 96, 96); channel 0 = background
    return frac, unknown


def dice(a, b):
    return 2 * (a & b).sum() / max(a.sum() + b.sum(), 1)


def main():
    C.verify_manifest()
    head_w, mask_w, brain_w, tmpl_o, mask_o = M1.load_template(96)
    l2r = label_to_region()
    mask96 = mask_o.numpy() > 0.5
    ref = nib.load(os.path.join(C.ROOT, "dataset", "mri_prep", "vol96", "002_S_2010.nii.gz"))
    ants.image_write(tmpl_o, os.path.join(WORK, "template96.nii.gz"))
    assert np.allclose(nib.load(os.path.join(WORK, "template96.nii.gz")).affine, ref.affine, atol=1e-3), \
        "96^3 template grid differs from vol96"

    frac, unknown = to_96(os.path.join(WORK, "template_synthseg.nii.gz"), brain_w, l2r)
    print("labels mapped to other (unlisted):", unknown)
    am = frac.argmax(0)
    brain_atlas = am > 0
    d_mask = dice(brain_atlas, mask96)
    atlas = np.where(mask96, np.where(am == 0, REGIONS.index("other") + 1, am), 0).astype(np.uint8)
    out = nib.Nifti1Image(atlas, ref.affine)
    nib.save(out, os.path.join(HERE, "atlas96.nii.gz"))
    np.savez_compressed(os.path.join(HERE, "atlas96_fractional.npz"), frac=frac.astype(np.float16),
                        channels=np.array(["background"] + REGIONS))

    vox = {r: int((atlas == i + 1).sum()) for i, r in enumerate(REGIONS)}
    S, P, R = np.indices(atlas.shape)
    hip = atlas == REGIONS.index("hippocampus") + 1
    left = hip & (R < 48)
    centroid = [int(round(x[hip].mean())) for x in (S, P, R)]
    centroid_left = [int(round(x[left].mean())) for x in (S, P, R)]
    slices = {"axial_S": centroid[0], "coronal_P": centroid[1], "sagittal_R": centroid_left[2]}

    # fit to individual anatomy: 22 smoke subjects (train), template atlas vs own SynthSeg
    smoke = pd.read_csv(os.path.join(C.ROOT, "analysis", "mri_morphometry", "smoke_subjects.csv"))
    C.assert_no_test(smoke.RID)
    rows = []
    for p in smoke.PTID:
        f = os.path.join(SMOKE_SEG, f"{p}_synthseg.nii.gz")
        if not os.path.exists(f):
            continue
        fs, _ = to_96(f, brain_w, l2r)
        a = fs.argmax(0)
        a = np.where(mask96, np.where(a == 0, REGIONS.index("other") + 1, a), 0)
        rows.append([dice(atlas == i + 1, a == i + 1) for i in range(len(REGIONS))])
    dsm = np.array(rows)
    qc = pd.DataFrame({"region": REGIONS, "voxels_96": [vox[r] for r in REGIONS],
                       "share_of_brain": [vox[r] / mask96.sum() for r in REGIONS],
                       "smoke_dice_median": np.median(dsm, 0), "smoke_dice_p25": np.percentile(dsm, 25, 0),
                       "smoke_dice_p75": np.percentile(dsm, 75, 0)})
    C.write_aggregate(qc, "atlas_qc.csv", ["train"])
    hip_med = float(np.median(dsm[:, REGIONS.index("hippocampus")]))
    info = {"regions": REGIONS, "index": {r: i + 1 for i, r in enumerate(REGIONS)}, "voxels": vox,
            "brain_voxels": int(mask96.sum()), "dice_atlas_vs_mask": float(d_mask),
            "acceptance_dice_ge_0.90": bool(d_mask >= 0.90), "n_smoke_subjects": len(rows),
            "smoke_hippocampus_dice_median": hip_med, "approximate_names": bool(hip_med < 0.5),
            "card_slices": slices, "hippocampus_centroid": centroid, "left_hippocampus_centroid": centroid_left,
            "unlisted_labels_as_other": [int(u) for u in unknown],
            "source": "SynthSeg 2.0 (FreeSurfer 7.4.1 mri_synthseg --parc) on ICBM152 2009a sym (masked), "
                      "m1 working grid, one-hot linear resampling to 96^3 + argmax",
            "license": "Derived from the MNI ICBM152 2009a template; see ~/nilearn_data/icbm152_2009/COPYING "
                       "(Copyright (C) 1993-2009 Louis Collins, McConnell Brain Imaging Centre, Montreal "
                       "Neurological Institute, McGill University)."}
    json.dump(info, open(os.path.join(HERE, "regions.json"), "w"), indent=1)

    t = tmpl_o.numpy()
    from matplotlib.colors import ListedColormap
    pal = ["#e6194b", "#f58231", "#4363d8", "#911eb4", "#3cb44b", "#9a6324", "#808000", "#e8e8e8", "#fabed4",
           "#ffe119", "#42d4f4", "#469990", "#a9a9a9"]
    cmap = ListedColormap(["#000000"] + pal)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    views = [(t[slices["axial_S"]], atlas[slices["axial_S"]], "axial", "upper"),
             (t[:, slices["coronal_P"]], atlas[:, slices["coronal_P"]], "coronal", "lower"),
             (t[:, :, slices["sagittal_R"]], atlas[:, :, slices["sagittal_R"]], "sagittal (left)", "lower")]
    for ax, (bg, lab, name, origin) in zip(axes, views):
        ax.imshow(bg, cmap="gray", origin=origin)
        ax.imshow(np.ma.masked_where(lab == 0, lab), cmap=cmap, vmin=0, vmax=len(REGIONS), alpha=0.45,
                  origin=origin, interpolation="nearest")
        for r, col in (("hippocampus", "white"), ("ventricles", "yellow")):
            ax.contour(lab == REGIONS.index(r) + 1, levels=[0.5], colors=col, linewidths=0.8, origin=origin)
        ax.set_title(name)
        ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(i + 1)) for i in range(len(REGIONS))]
    fig.legend(handles, REGIONS, loc="lower center", ncol=7, fontsize=7, frameon=False)
    fig.suptitle(f"Template atlas on the CNN's 96³ grid (ICBM152 2009a, SynthSeg). Dice vs mask {d_mask:.3f}; "
                 f"smoke-subject hippocampus Dice median {hip_med:.2f}\n"
                 "outlines: hippocampus (white), ventricles (yellow)", fontsize=9)
    fig.savefig(os.path.join(HERE, "atlas_qc.png"), dpi=110, bbox_inches="tight")
    print(json.dumps({k: info[k] for k in ("dice_atlas_vs_mask", "acceptance_dice_ge_0.90", "n_smoke_subjects",
                                          "smoke_hippocampus_dice_median", "approximate_names", "card_slices")}))
    print(qc.round(3).to_string(index=False))
    C.verify_manifest()


if __name__ == "__main__":
    main()
