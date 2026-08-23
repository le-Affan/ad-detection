"""PyTorch Dataset over the preprocessed ADNI volumes.

Reads `mri_index.csv` and filters by split. The index is built by m2_index.py
from whatever is on disk, so this class never holds a subject list or count of
its own -- as more scans finish preprocessing, rebuilding the index is the only
step needed.

TARGETS. The SDD gives the MRI stream two heads, and the 4-class label maps
onto them like this:

    DX_bl   label   DEMO    ALZ
    CN        0      0.0     0
    EMCI      1      0.5     0
    LMCI      2      1.0     0
    AD        3      2.0     1

DEMO is the continuous 0-2 severity score, on the CDR-like spacing: the
CN->EMCI->LMCI steps are half-points and AD sits a full point above LMCI,
mirroring how CDR global scores actually distribute rather than spreading the
four classes evenly. ALZ is AD-vs-rest.

Both mappings are single constants below and are the only place either target
is defined, so changing the spacing or making ALZ e.g. AD+LMCI-vs-rest is a
one-line edit. The 4-class label is returned alongside them, unused by the two
heads but needed by the downstream KAN's COG head and useful for stratified
metrics.

Volumes are stored as .nii.gz written by m1 and are already normalised to
[0, 1] and shaped 96^3, so loading does no rescaling -- it only adds the
channel axis the 3D convolutions expect. Anything outside [0, 1] or off-shape
means an m1 artefact is stale or corrupt, so it raises rather than silently
feeding bad data to the model.
"""
import os

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INDEX = os.path.join(ROOT, "dataset", "mri_prep", "mri_index.csv")

# label -> continuous severity for the DEMO head (CDR-like spacing)
DEMO_BY_LABEL = {0: 0.0, 1: 0.5, 2: 1.0, 3: 2.0}
# label -> binary target for the ALZ head (AD vs rest)
ALZ_BY_LABEL = {0: 0.0, 1: 0.0, 2: 0.0, 3: 1.0}

EXPECTED_SHAPE = (96, 96, 96)


class ADNIVolumes(Dataset):
    """Preprocessed baseline T1 volumes for one split.

    Parameters
    ----------
    split : "train" | "val" | "test" | None
        None loads every indexed subject, which is useful for QC sweeps but
        must never be used to train.
    index_path : str
        Defaults to the index m2_index.py writes.
    strict : bool
        Validate shape and intensity range on every load. Cheap next to the
        file read, and left on by default so a bad volume surfaces at once.
    """

    def __init__(self, split="train", index_path=None, strict=True, data_root=None):
        self.data_root = data_root
        if index_path is None:
            index_path = (os.path.join(data_root, "mri_index.csv")
                          if data_root else INDEX)
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"{index_path} not found -- run src/mri/m2_index.py first")
        df = pd.read_csv(index_path)
        if split is not None:
            if split not in set(df.split.unique()) | {"train", "val", "test"}:
                raise ValueError(f"unknown split {split!r}")
            df = df[df.split == split]
        self.df = df.reset_index(drop=True)
        self.split = split
        self.strict = strict

    def __len__(self):
        return len(self.df)

    def class_counts(self):
        """Label counts for this split, for class weighting or reporting.

        Computed from the rows actually present, so it stays correct as the
        cohort grows instead of encoding today's numbers.
        """
        return self.df.label.value_counts().reindex(range(4), fill_value=0).to_dict()

    def volume_path(self, r):
        """Absolute path to a row's model-input volume.

        Two resolution modes. With `data_root` set (Kaggle, or any relocated
        copy of the preprocessed data) the path is rebuilt from PTID against
        that root, because the index's `vol_path` is only meaningful relative
        to the repo it was generated in. Without it, `vol_path` is used as
        recorded -- but separators are normalised first: m2_index.py writes
        native paths, so an index built on Windows carries backslashes, which
        are ordinary filename characters on Linux and would not resolve.
        """
        if self.data_root:
            return os.path.join(self.data_root, "vol96", f"{r.PTID}.nii.gz")
        rel = str(r.vol_path).replace("\\", "/")
        return os.path.join(ROOT, *rel.split("/"))

    def __getitem__(self, i):
        r = self.df.iloc[i]
        path = self.volume_path(r)
        vol = np.asarray(nib.load(path).dataobj, dtype=np.float32)

        if self.strict:
            if vol.shape != EXPECTED_SHAPE:
                raise ValueError(f"{r.PTID}: shape {vol.shape}, expected {EXPECTED_SHAPE}")
            lo, hi = float(vol.min()), float(vol.max())
            if lo < -1e-4 or hi > 1 + 1e-4:
                raise ValueError(f"{r.PTID}: intensity [{lo}, {hi}] outside [0, 1]")

        label = int(r.label)
        return {
            "volume": torch.from_numpy(vol).unsqueeze(0),  # (1, 96, 96, 96)
            "demo": torch.tensor(DEMO_BY_LABEL[label], dtype=torch.float32),
            "alz": torch.tensor(ALZ_BY_LABEL[label], dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
            "ptid": r.PTID,
            "rid": int(r.RID),
        }
