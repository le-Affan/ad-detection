"""PyTorch Dataset over Path A visit sequences: one item per SUBJECT.

The multi-visit counterpart of dataset.py, which is unchanged and still serves
the single-scan baseline. Reads `visits_index.csv` (p3_index.py) and returns,
per subject, a fixed four-slot sequence (bl, m06, m12, m24):

    volumes  (4, 1, 96, 96, 96) float32   zeros in masked slots
    mask     (4,)  bool                   True = usable visit
    days     (4,)  float32                days since the subject's first USABLE
                                          visit; 0 in masked slots
    months   (4,)  float32                nominal slot months (0/6/12/24)
    demo, alz, label, ptid, rid           exactly as dataset.py defines them

FIXED SHAPE, SO NO COLLATE FUNCTION. Every item is four slots whether the
subject has four visits or two, so torch's default collate stacks a batch into
(B, 4, 1, 96, 96, 96) plus a (B, 4) mask. Padding lives in the data, and the
mask says which of it is real. The model must use the mask, never the zeros:
a zero volume is a legal (if odd) input to a CNN, so "it's zero, it won't
matter" is not a masking scheme.

MASKED SLOTS ARE NEVER READ. A masked visit has no vol_path in the index, and
this class does not open a file for it. Zeros are written there only so the
batch can stack.

DAYS ARE RECOMPUTED HERE, not taken from stage A's `days_from_first`. That
column is measured from the first visit present in the MANIFEST; a visit
masked later (027_S_4966 bl, REG_FAILED) would leave its subject's clock
starting at a visit the model never sees. Measuring from the first usable
visit keeps t = 0 on real input for every subject.

TARGETS ARE THE BASELINE DIAGNOSIS (DX_bl), matching the static model, so
Stage E compares like with like. They are imported from dataset.py, not
restated.
"""
import os

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from dataset import ALZ_BY_LABEL, DEMO_BY_LABEL, EXPECTED_SHAPE, ROOT

INDEX = os.path.join(ROOT, "dataset", "mri_prep_patha", "visits_index.csv")
N_SLOTS = 4


class ADNIVisitSequences(Dataset):
    """Per-subject four-slot visit sequences for one split.

    Parameters mirror ADNIVolumes: `split=None` loads every subject (QC only,
    never training); `data_root` rebuilds each path as
    <data_root>/vol96/<PTID>__<slot>.nii[.gz] for a relocated copy (Kaggle);
    `strict` validates shape and range on every load.
    """

    def __init__(self, split="train", index_path=None, strict=True, data_root=None):
        self.data_root = data_root
        if index_path is None:
            index_path = os.path.join(data_root, "visits_index.csv") if data_root else INDEX
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"{index_path} not found -- run src/mri/p3_index.py first")
        df = pd.read_csv(index_path)
        if split is not None:
            if split not in {"train", "val", "test"}:
                raise ValueError(f"unknown split {split!r}")
            df = df[df.split == split]
        df = df.sort_values(["PTID", "slot_idx"])

        slots = df.groupby("PTID").slot_idx.apply(list)
        bad = slots[slots.map(lambda s: s != list(range(N_SLOTS)))]
        if len(bad):
            raise ValueError(f"{len(bad)} subjects lack the dense 4-slot grid: {list(bad.index[:5])}")
        empty = df.groupby("PTID").present.sum()
        if (empty == 0).any():
            raise ValueError(f"subjects with no usable visit: {list(empty[empty == 0].index[:5])}")

        self.df = df.reset_index(drop=True)
        self.ptids = list(slots.index)
        self._rows = {p: g for p, g in self.df.groupby("PTID")}
        self.split = split
        self.strict = strict

    def __len__(self):
        return len(self.ptids)

    def class_counts(self):
        """Subject-level label counts, as ADNIVolumes.class_counts."""
        lab = self.df.groupby("PTID").label.first()
        return lab.value_counts().reindex(range(4), fill_value=0).to_dict()

    def visit_counts(self):
        """How many subjects have 1, 2, 3, 4 usable visits."""
        return self.df.groupby("PTID").present.sum().value_counts().sort_index().to_dict()

    def volume_path(self, r):
        if self.data_root:
            base = os.path.join(self.data_root, "vol96", str(r.prep_key))
            for ext in (".nii.gz", ".nii"):
                if os.path.exists(base + ext):
                    return base + ext
            return base + ".nii.gz"
        return os.path.join(ROOT, *str(r.vol_path).replace("\\", "/").split("/"))

    def load_volume(self, r):
        path = self.volume_path(r)
        vol = np.asarray(nib.load(path).dataobj, dtype=np.float32)
        if self.strict:
            if vol.shape != EXPECTED_SHAPE:
                raise ValueError(f"{r.prep_key}: shape {vol.shape}, expected {EXPECTED_SHAPE}")
            lo, hi = float(vol.min()), float(vol.max())
            if lo < -1e-4 or hi > 1 + 1e-4:
                raise ValueError(f"{r.prep_key}: intensity [{lo}, {hi}] outside [0, 1]")
        return vol

    def __getitem__(self, i):
        g = self._rows[self.ptids[i]]
        vols = np.zeros((N_SLOTS, 1) + EXPECTED_SHAPE, dtype=np.float32)
        mask = g.present.to_numpy(dtype=bool, copy=True)
        dates = pd.to_datetime(g.scan_date).to_numpy()
        first = dates[mask].min()
        days = np.zeros(N_SLOTS, dtype=np.float32)
        for k, (_, r) in enumerate(g.iterrows()):
            if not mask[k]:
                continue                       # never opened: no path, no read
            vols[k, 0] = self.load_volume(r)
            days[k] = (dates[k] - first) / np.timedelta64(1, "D")

        r0 = g.iloc[0]
        label = int(r0.label)
        return {
            "volumes": torch.from_numpy(vols),                         # (4, 1, 96, 96, 96)
            "mask": torch.from_numpy(mask),                            # (4,)
            "days": torch.from_numpy(days),                            # (4,)
            "months": torch.tensor(g.months.to_numpy(dtype=np.float32)),
            "demo": torch.tensor(DEMO_BY_LABEL[label], dtype=torch.float32),
            "alz": torch.tensor(ALZ_BY_LABEL[label], dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
            "ptid": r0.PTID,
            "rid": int(r0.RID),
        }
