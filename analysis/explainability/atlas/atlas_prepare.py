"""Atlas, step 1 of 3 (run with LYP/.venv-mri): the template in the m1 working grid, ready for SynthSeg.

Calls m1_preprocess.load_template() unchanged, so the template sits in exactly the grid every subject's
mni/ volume was registered to (ICBM152 2009a sym, IAL, 1 x 1 x 1.2 mm). Writes:
  atlas/work/template_mni.nii.gz    masked template (brain_w), full working grid
  atlas/work/template_crop.nii.gz   the same, cropped to run_synthseg.MASK_BOX + 3 voxels (lossless; SynthSeg input)
  atlas/work/mask_mni.nii.gz        template brain mask (mask_w)
Template data only (no subject data).
"""
import os
import sys

import ants
import nibabel as nib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import common as C  # noqa: E402

sys.path.insert(0, os.path.join(C.ROOT, "src", "mri"))
sys.path.insert(0, os.path.join(C.ROOT, "analysis", "mri_morphometry"))
import m1_preprocess as M1  # noqa: E402

WORK = os.path.join(HERE, "work")
MASK_BOX = ((0, 155), (25, 206), (22, 143))      # run_synthseg.MASK_BOX (template brain-mask bbox, mni/ grid)
MARGIN = 3


def main():
    C.verify_manifest()
    os.makedirs(WORK, exist_ok=True)
    head_w, mask_w, brain_w, tmpl_o, mask_o = M1.load_template(96)
    print("working grid", brain_w.shape, brain_w.spacing, brain_w.orientation)
    ref = nib.load(os.path.join(C.ROOT, "dataset", "mri_prep", "mni", "002_S_2010.nii.gz"))
    assert brain_w.shape == ref.shape, (brain_w.shape, ref.shape)
    ants.image_write(brain_w, os.path.join(WORK, "template_mni.nii.gz"))
    ants.image_write(mask_w, os.path.join(WORK, "mask_mni.nii.gz"))
    img = nib.load(os.path.join(WORK, "template_mni.nii.gz"))
    assert np.allclose(img.affine, ref.affine, atol=1e-3), "template grid differs from the subjects' mni/ grid"
    m = nib.load(os.path.join(WORK, "mask_mni.nii.gz")).get_fdata() > 0.5
    nz = np.argwhere(m)
    lo, hi = nz.min(0), nz.max(0)
    for a, (b0, b1) in enumerate(MASK_BOX):
        assert b0 <= lo[a] and hi[a] <= b1, ("mask bbox", lo, hi)
    sl = tuple(slice(max(b0 - MARGIN, 0), min(b1 + MARGIN + 1, img.shape[a])) for a, (b0, b1) in enumerate(MASK_BOX))
    crop = img.slicer[sl]
    data = crop.get_fdata().astype(np.float32)
    nib.save(nib.Nifti1Image(data / data.max(), crop.affine), os.path.join(WORK, "template_crop.nii.gz"))
    print("crop", crop.shape, "written to", WORK)
    C.verify_manifest()


if __name__ == "__main__":
    main()
