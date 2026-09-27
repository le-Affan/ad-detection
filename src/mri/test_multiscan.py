"""Checks for the multi-visit train index (mv1_index.py).

Run directly (`python src/mri/test_multiscan.py`) or under pytest. Reads the
real train_scans.csv, so run mv1_index.py first.

The central property is SUBJECT-LEVEL SPLIT DISCIPLINE: every row belongs to a
train subject, re-derived from load_splits() by RID, and no val or test subject
appears in any form. A leak here would inflate every number the experiment
reports without any visible symptom, so it is tested directly rather than
inferred from the builder having run.
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from m2_index import load_splits  # noqa: E402
from mv1_index import (COLLAPSE, FOLLOW_UPS, MRI_INDEX, OUT_DIR,  # noqa: E402
                       ROOT, VISITS_INDEX)

TRAIN_SCANS = os.path.join(OUT_DIR, "train_scans.csv")


def _load():
    return pd.read_csv(TRAIN_SCANS), pd.read_csv(MRI_INDEX)


def test_every_row_is_a_train_subject():
    df, _ = _load()
    split_of = load_splits()
    assert (df.RID.map(split_of) == "train").all()
    assert (df.split == "train").all()


def test_no_subject_shared_with_val_or_test():
    df, mi = _load()
    split_of = load_splits()
    held = mi[mi.RID.map(split_of) != "train"]
    vi = pd.read_csv(VISITS_INDEX)
    held_pa = vi[vi.RID.map(split_of) != "train"]
    assert not set(df.PTID) & (set(held.PTID) | set(held_pa.PTID))
    assert not set(df.RID) & (set(held.RID) | set(held_pa.RID))
    # scan_key embeds the PTID; guard against a follow-up keyed to another subject
    assert (df.scan_key.str.split("__").str[0] == df.PTID).all()


def test_baseline_rows_are_exactly_v9_train():
    df, mi = _load()
    bl = df[df.slot == "bl"].set_index("PTID").sort_index()
    ref = mi[mi.split == "train"].set_index("PTID").sort_index()
    assert list(bl.index) == list(ref.index)
    assert (bl.label == ref.label).all() and (bl.RID == ref.RID).all()
    assert (bl.vol_path == ref.vol_path.str.replace("\\", "/", regex=False)).all()
    assert (bl.source == "mri_prep").all()


def test_no_path_a_baseline_slot():
    """Path A's bl is the static baseline's session; using both double-counts it."""
    df, _ = _load()
    assert not ((df.slot == "bl") & (df.source != "mri_prep")).any()
    assert set(df[df.source == "mri_prep_patha"].slot) <= set(FOLLOW_UPS)


def test_follow_up_labels_are_consistent():
    df, mi = _load()
    fu = df[df.slot != "bl"]
    assert fu.visit_dx.notna().all()
    assert (fu.visit_dx == fu.DX_bl.map(COLLAPSE)).all()
    # the label is the subject's DX_bl label, never a per-visit relabel
    lab = mi.set_index("PTID").label
    assert (df.label == df.PTID.map(lab)).all()


def test_unique_and_on_disk():
    df, _ = _load()
    assert not df.duplicated(["PTID", "slot"]).any()
    assert not df.scan_key.duplicated().any()
    assert df.vol_path.map(lambda p: os.path.exists(os.path.join(ROOT, p))).all()


def test_ptid_rid_one_to_one():
    df, _ = _load()
    assert (df.groupby("PTID").RID.nunique() == 1).all()
    assert (df.groupby("RID").PTID.nunique() == 1).all()


def _sets(train_index=TRAIN_SCANS):
    """The three splits exactly as m3_train.py builds them with --train-index."""
    from dataset import ADNIVolumes
    return {s: ADNIVolumes(split=s, index_path=train_index if s == "train" else None)
            for s in ("train", "val", "test")}


def test_m3_guard_passes_on_real_index():
    from m3_train import assert_subject_disjoint
    assert_subject_disjoint(_sets())


def test_m3_guard_catches_an_injected_leak():
    """The negative test: one val scan slipped into the train index must abort."""
    import tempfile
    from m3_train import assert_subject_disjoint
    df, mi = _load()
    victim = mi[mi.split == "val"].iloc[0]
    leak = df.iloc[[0]].copy()
    leak[["RID", "PTID", "scan_key"]] = [victim.RID, victim.PTID, victim.PTID]
    leak["split"] = "train"          # a mislabelled split column must not hide it
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "leaky.csv")
        pd.concat([df, leak]).to_csv(path, index=False)
        try:
            assert_subject_disjoint(_sets(path))
        except RuntimeError as e:
            assert "SUBJECT LEAK" in str(e)
        else:
            raise AssertionError("guard did not catch a val subject in the train index")


def test_scan_key_resolves_under_data_root():
    """Relocated (Kaggle) layout: vol96/<scan_key>.nii[.gz], one file per scan."""
    import shutil
    import tempfile
    from dataset import ADNIVolumes
    df, mi = _load()
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "vol96"))
        rows = pd.concat([df[df.slot == "bl"].head(1), df[df.slot != "bl"].head(2)])
        for r in rows.itertuples():
            shutil.copy(os.path.join(ROOT, r.vol_path), os.path.join(d, "vol96", r.scan_key + ".nii.gz"))
        rows.to_csv(os.path.join(d, "train_scans.csv"), index=False)
        ds = ADNIVolumes("train", index_path=os.path.join(d, "train_scans.csv"), data_root=d)
        paths = [ds.volume_path(r) for r in ds.df.itertuples()]
        assert len(set(paths)) == 3 and all(os.path.exists(p) for p in paths)
        assert ds[1]["volume"].shape == (1, 96, 96, 96)
    # the default index has no scan_key and still resolves by PTID
    ds = ADNIVolumes("val", index_path=MRI_INDEX, data_root="/x")
    assert ds.volume_path(ds.df.iloc[0]).startswith("/x/vol96/" + ds.df.PTID.iloc[0])


def test_class_weighting_math():
    import torch
    from m3_train import demo_mse, inverse_class_weights
    df, _ = _load()
    counts = df.label.value_counts().to_dict()
    w = inverse_class_weights(counts)
    per_row = torch.tensor(w)[torch.tensor(df.label.values)]
    assert abs(float(per_row.mean()) - 1.0) < 1e-6          # loss scale unchanged
    for i in range(4):                                     # each class sums equal
        assert abs(w[i] * counts[i] - len(df) / 4) < 1e-6
    p, t = torch.randn(16), torch.randn(16)
    lab = torch.randint(0, 4, (16,))
    assert torch.equal(demo_mse(p, t, lab), torch.nn.functional.mse_loss(p, t))
    assert torch.allclose(demo_mse(p, t, lab, torch.ones(4)), torch.nn.functional.mse_loss(p, t))


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001 -- report every failure, not the first
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
