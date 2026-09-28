"""DeepBrainNet smoke test (Bashyam et al., 2020) on clinical-TRAIN subjects only.

Never reads val or test: subjects come from dataset/processed/train_rids.csv,
asserted. Run with TF_USE_LEGACY_KERAS=1 in LYP/.venv-deepbrainnet.

Model: vishnubashyam/DeepBrainNet @ 505f4e4, Models/DBN_model.h5 (Git LFS
object, sha256 9257e98e...41ea): the LifespanCN brain-age regressor (VGG16
backbone, 1 linear output, MSE loss). See p1_report.md for the verification.

The repo's inference (Script/Slicer.py + Model_Test.py), reproduced exactly:
  volume * (185 / p97(volume))  ->  axial slices 45..124 of the 182x218x182
  array  ->  PIL Image.fromarray(float).convert('RGB') -> JPEG  ->  keras
  load_img(target_size=(256, 256), nearest)  ->  / 255  ->  VGG16  ->  median
  over the 80 slice predictions.

Input adaptation of our m1 output (dataset/mri_prep/mni), the only step added:
  resample by world (MNI mm) coordinates onto the grid of the repo's example
  inputs (182x218x182, 1 mm, affine diag(-1,-1,1), origin (90, 91, -72)),
  trilinear, zero outside the field of view.

Modes:
  --examples   the repo's 4 example subjects (environment check, no resample)
  (default)    the smoke-test subjects -> dbn_smoke_test.csv (+ sensitivity)
"""
import argparse
import os
import tempfile
import time

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.processing import resample_from_to
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PROCESSED = os.path.join(ROOT, "dataset", "processed")
DBN = os.path.abspath(os.path.join(ROOT, "..", "..", "third_party", "DeepBrainNet"))

GRID_SHAPE = (182, 218, 182)
GRID_AFFINE = np.array([[-1, 0, 0, 90], [0, -1, 0, 91], [0, 0, 1, -72], [0, 0, 0, 1]], float)
SLICES = range(45, 125)
MIN_CORR = 0.3
PERTURB = {"x0.9": 0.9, "x1.1": 1.1}   # applied after DBN's own 185/p97 scaling


# --- repo pipeline -----------------------------------------------------------------
def load_model():
    import tf_keras
    return tf_keras.models.load_model(os.path.join(DBN, "Models", "DBN_model.h5"), compile=False)


def slices_to_batch(vol, tmp, factor=1.0):
    """Slicer.py + keras load_img, exactly; `factor` rescales after 185/p97."""
    from tf_keras.utils import img_to_array, load_img
    data = (vol * (185.0 / np.percentile(vol, 97)) * factor).astype(np.float32)   # PIL mode 'F'
    batch = []
    for k in SLICES:
        p = os.path.join(tmp, f"s{k}.jpg")
        Image.fromarray(np.ascontiguousarray(data[:, :, k])).convert("RGB").save(p)
        batch.append(img_to_array(load_img(p, target_size=(256, 256), interpolation="nearest")) / 255.0)
    return np.stack(batch)


def predict(model, vol, factor=1.0):
    with tempfile.TemporaryDirectory() as tmp:
        x = slices_to_batch(vol, tmp, factor)
    per_slice = model.predict(x, batch_size=80, verbose=0).reshape(-1)
    return float(np.median(per_slice)), per_slice


def to_dbn_grid(path):
    img = nib.load(path)
    res = resample_from_to(img, (GRID_SHAPE, GRID_AFFINE), order=1, mode="constant", cval=0.0)
    return np.asarray(res.dataobj, dtype=np.float32)


# --- subjects -------------------------------------------------------------------------
def cohort():
    train = set(pd.read_csv(os.path.join(PROCESSED, "train_rids.csv")).RID)
    mi = pd.read_csv(os.path.join(ROOT, "dataset", "mri_prep", "mri_index.csv"))
    qc = pd.read_csv(os.path.join(ROOT, "dataset", "mri_prep", "m1_report_full.csv"))[["PTID", "corr_vs_mni"]]
    man = pd.read_csv(os.path.join(ROOT, "dataset", "mri_manifest", "mri_manifest.csv"))
    age = pd.read_csv(os.path.join(PROCESSED, "step2_features.csv"), usecols=["RID", "AGE"])
    df = (mi.merge(qc, on="PTID").merge(age, on="RID")
            .merge(man[["RID", "BASELINE_EXAMDATE", "MRI_EXAMDATE"]], on="RID"))
    df = df[df.RID.isin(train) & (df.split == "train") & df.label.isin([0, 3]) & (df.corr_vs_mni >= MIN_CORR)]
    gap = (pd.to_datetime(df.MRI_EXAMDATE) - pd.to_datetime(df.BASELINE_EXAMDATE)).dt.days / 365.25
    return df.assign(age_at_scan=df.AGE + gap), train


def pick_subjects():
    """The 20 P1 subjects (SFCN smoke test) + 10 CN and 10 AD more, 2 per age
    quintile of that class (seed 1), so the set spans the age range."""
    df, train = cohort()
    p1 = pd.read_csv(os.path.join(HERE, "smoke_test.csv"))
    first = df[df.PTID.isin(p1.PTID)].assign(source="P1")
    assert len(first) == 20
    rest = df[~df.PTID.isin(p1.PTID)].copy()
    rest["q"] = rest.groupby("label").age_at_scan.transform(lambda a: pd.qcut(a, 5, labels=False))
    extra = rest.groupby(["label", "q"], group_keys=False).apply(
        lambda g: g.sample(2, random_state=1)).assign(source="added")
    out = pd.concat([first, extra.drop(columns="q")]).reset_index(drop=True)
    assert out.RID.isin(train).all() and out.RID.is_unique
    return out


# --- runs ---------------------------------------------------------------------------
def run_examples(model):
    ages = pd.read_csv(os.path.join(DBN, "Sample_Data.csv")).set_index("ID").Age
    for sid, age in ages.items():
        vol = np.asarray(nib.load(os.path.join(DBN, "Data", f"{sid}_T1_BrainAligned.nii.gz")).dataobj, np.float32)
        pred, _ = predict(model, vol)
        print(f"{sid}: age {age:.2f}  pred {pred:.2f}  gap {pred - age:+.2f}")


def run_smoke(model):
    subj = pick_subjects()
    rows = []
    for r in subj.itertuples():
        t0 = time.time()
        vol = to_dbn_grid(os.path.join(ROOT, "dataset", "mri_prep", "mni", f"{r.PTID}.nii.gz"))
        pred, per_slice = predict(model, vol)
        sec = time.time() - t0
        row = {"RID": r.RID, "PTID": r.PTID, "DX_bl": r.DX_bl, "source": r.source,
               "age_at_scan": round(r.age_at_scan, 2), "pred_age": round(pred, 2),
               "gap": round(pred - r.age_at_scan, 2), "slice_pred_iqr": round(float(np.subtract(*np.percentile(per_slice, [75, 25]))), 2),
               "nonzero_frac": round(float((vol > 0).mean()), 4), "seconds": round(sec, 1)}
        for k, f in PERTURB.items():
            row[f"pred_{k}"] = round(predict(model, vol, f)[0], 2)
        rows.append(row)
        print(f"{r.PTID} {r.DX_bl} age {r.age_at_scan:5.1f} pred {pred:5.1f} ({sec:.1f}s)", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(HERE, "dbn_smoke_test.csv"), index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", action="store_true")
    args = ap.parse_args()
    model = load_model()
    if args.examples:
        run_examples(model)
    else:
        run_smoke(model)


if __name__ == "__main__":
    main()
