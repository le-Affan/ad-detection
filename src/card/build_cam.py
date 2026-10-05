"""Card data, step 3 of 3 (run with LYP/.venv-mri): CNN attribution for every subject with a CNN volume.

Method fixed by the explainability analysis (analysis/explainability, pre-registered in preregistration.md):
LayerCAM of the ALZ logit at the stage chosen by the deletion rule P1-R1 (results/card_rules.json
["cnn_stage"]). Uses the existing cross-fit checkpoints only; nothing is trained.
Which models explain which subject (the models that produced the subject's ALZ):
  - train subjects: the 3 seed models of the subject's own cross-fit fold (never trained on that subject);
  - val and test subjects: all 15 fold models.
Per model: map upsampled trilinearly to 96^3, masked to the template brain, unit positive mass; averaged over
the subject's models. Region density d_r = share of attribution in atlas region r / its share of brain voxels
(template atlas, analysis/explainability/atlas). Card slices are fixed for every subject: axial and coronal
through the template hippocampus centroid, sagittal through the left hippocampus centroid.
Stored per subject: the three slices of the scan and of the map (uint8; map scaled to its maximum over the
three slices), d_r for the 13 regions, n_models. Volumes are loaded per model (peak RAM about 2 GB).
Output: demo/card_data/cam.npz. (Replaces the earlier layer4 Grad-CAM, gradcam.npz.)
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
EXPL = os.path.join(ROOT, "analysis", "explainability")
sys.path.insert(0, EXPL)
import common as C  # noqa: E402
import p1_lib as L  # noqa: E402

OUT = os.path.join(ROOT, "demo", "card_data")


def u8(x):
    return np.clip(np.round(x * 255), 0, 255).astype(np.uint8)


def main():
    C.verify_manifest()
    stage = json.load(open(os.path.join(C.RESULTS, "card_rules.json")))["cnn_stage"]["stage"]
    atl, info = L.atlas()
    regions, vox, nb = info["regions"], info["voxels"], info["brain_voxels"]
    sl = info["card_slices"]
    mask = L.brain_mask()
    d = L.subjects(include_test=True)                     # card only: test subjects are rendered, never aggregated
    print(f"{len(d)} subjects on {L.DEVICE}; LayerCAM at {stage}", flush=True)
    flat = atl.ravel()
    reg_idx = [np.flatnonzero(flat == i + 1) for i in range(len(regions))]
    # Memory: volumes are loaded per model, and only what the card needs is accumulated per subject:
    # region masses (13) and the three fixed slices of the summed map. Unit-mass maps make both linear in the sum.
    mass = np.zeros((len(d), len(regions)))
    sl_sum = np.zeros((len(d), 3, 96, 96), np.float32)
    n_models = np.zeros(len(d), int)

    def slices(v):
        return np.stack([v[sl["axial_S"]], v[:, sl["coronal_P"]], v[:, :, sl["sagittal_R"]]])
    for mid in range(len(L.MODELS)):
        rows = [i for i, r in enumerate(d.itertuples()) if mid in L.models_of(r)]
        model = L.get_model(mid)
        for c in range(0, len(rows), 32):
            chunk = rows[c:c + 32]
            vols = L.load_volumes(list(d.path.iloc[chunk]))
            maps, _ = L.cams(model, vols, stages=[stage])
            up = L.upsample(maps[("layercam", stage)], mask)
            for j, i in enumerate(chunk):
                f = up[j].ravel()
                mass[i] += [f[ix].sum() for ix in reg_idx]
                sl_sum[i] += slices(up[j])
            n_models[chunk] += 1
        del model
        torch.cuda.empty_cache()
        print(f"model {mid}: {len(rows)} subjects", flush=True)
    assert set(n_models[d.split == "train"]) == {3} and set(n_models[d.split != "train"]) == {15}
    share = mass / mass.sum(1, keepdims=True)
    dens = share / (np.array([vox[r] for r in regions]) / nb)[None]
    bg, cm = [], []
    for i in range(len(d)):
        v = L.load_volumes([d.path.iloc[i]])[0]
        # vol96 axes are (S, P, R): axial = fixed S -> (P, R); coronal = fixed P -> (S, R); sagittal = fixed R -> (S, P)
        bg.append(slices(v))
        cs = sl_sum[i] / n_models[i]
        cm.append(cs / cs.max() if cs.max() > 0 else cs)
    np.savez_compressed(os.path.join(OUT, "cam.npz"), ptid=d.PTID.to_numpy(), split=d.split.to_numpy(),
                        n_models=n_models, bg=u8(np.array(bg)), cam=u8(np.array(cm)), density=dens.astype(np.float32),
                        regions=np.array(regions), stage=stage, map_size=L.STAGE_SIZE[stage])
    print(f"saved {len(d)} subjects; {os.path.getsize(os.path.join(OUT, 'cam.npz')) / 1e6:.1f} MB")
    C.verify_manifest()


if __name__ == "__main__":
    main()
