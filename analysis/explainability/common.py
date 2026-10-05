"""Shared helpers for the explainability analyses (numpy / pandas only; importable from every env).

Rules enforced here (analysis/explainability/preregistration.md):
  - verify_manifest(): the frozen fusion-v3 artefacts (frozen/MANIFEST.sha256, 109 files) are unchanged.
    Every script calls it at start and end.
  - assert_no_test(rids): no test subject reaches an aggregate, a selection or a threshold. Test subjects
    are only ever processed by the card builders in src/card/.
  - write_aggregate(): committed tables carry no subject identifiers and only train/val subjects.
  - boot_ci(): 2,000 percentile-bootstrap resamples of subjects, numpy default_rng(0).
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
LYP = os.path.dirname(os.path.dirname(ROOT))
V3 = os.path.join(ROOT, "analysis", "fusion_v3")
FROZEN = os.path.join(V3, "frozen")
PROCESSED = os.path.join(ROOT, "dataset", "processed")
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")
CACHE = os.path.join(HERE, "cache")
SUBJ = os.path.join(HERE, "subject_level")
ATLAS = os.path.join(HERE, "atlas")
NAMES = ["CN", "EMCI", "LMCI", "AD"]
B_BOOT = 2000


# --- frozen guard -----------------------------------------------------------------------
def verify_manifest():
    """sha256 of every file listed in frozen/MANIFEST.sha256; raises on any mismatch."""
    n = 0
    with open(os.path.join(FROZEN, "MANIFEST.sha256")) as f:
        for line in f:
            if not line.strip():
                continue
            digest, rel = line.split(maxsplit=1)
            rel = rel.strip().lstrip("*")
            h = hashlib.sha256()
            with open(os.path.join(ROOT, rel), "rb") as g:
                for chunk in iter(lambda: g.read(1 << 20), b""):
                    h.update(chunk)
            if h.hexdigest() != digest:
                raise RuntimeError(f"frozen artefact changed: {rel}")
            n += 1
    assert n == 109, f"manifest lists {n} files, expected 109"
    return n


# --- splits -------------------------------------------------------------------------------
def rids(split):
    return [int(r) for r in pd.read_csv(os.path.join(PROCESSED, f"{split}_rids.csv")).RID]


def split_of():
    return {r: s for s in ("train", "val", "test") for r in rids(s)}


def fold_of():
    f = pd.read_csv(os.path.join(ROOT, "analysis", "fusion_feasibility", "folds", "folds.csv"))
    return dict(zip(f.RID.astype(int), f.fold.astype(int)))


def assert_no_test(rid_list):
    test = set(rids("test"))
    bad = test & {int(r) for r in rid_list}
    assert not bad, f"{len(bad)} test subjects reached an aggregate or selection"


def labels():
    """RID -> 4-class label (0 CN .. 3 AD) for train and val, from the fusion dataset."""
    d = pd.read_csv(os.path.join(ROOT, "analysis", "kan_fusion", "fusion_trainval.csv"), usecols=["RID", "label"])
    return dict(zip(d.RID.astype(int), d.label.astype(int)))


def dry_run_ptids():
    """Pre-registered dry-run set: first 3 AD, 3 CN, 2 EMCI, 2 LMCI train subjects with a scan, by PTID."""
    d = pd.read_csv(os.path.join(ROOT, "analysis", "kan_fusion", "fusion_trainval.csv"),
                    usecols=["PTID", "split", "label", "has_mri"])
    d = d[(d.split == "train") & d.has_mri].sort_values("PTID")
    out = []
    for lab, k in ((3, 3), (0, 3), (1, 2), (2, 2)):
        out += list(d[d.label == lab].PTID.iloc[:k])
    return out


# --- statistics ---------------------------------------------------------------------------
def boot_idx(n, B=B_BOOT, seed=0):
    return np.random.default_rng(seed).integers(0, n, size=(B, n))


def boot_ci(x, B=B_BOOT, seed=0, stat=np.mean):
    """Point estimate and percentile 95% CI of stat over subjects (axis 0)."""
    x = np.asarray(x, float)
    idx = boot_idx(len(x), B, seed)
    bs = np.array([stat(x[i], axis=0) for i in idx]) if stat is not np.mean else x[idx].mean(1)
    return stat(x, axis=0), np.percentile(bs, 2.5, axis=0), np.percentile(bs, 97.5, axis=0)


def tier(v, pass_at=0.80, weak_at=0.60):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return "PASS" if v >= pass_at else ("WEAK" if v >= weak_at else "FAIL")


def spearman(a, b):
    from scipy.stats import spearmanr
    r = spearmanr(a, b).correlation
    return float(r) if np.isfinite(r) else np.nan


def median_pairwise_spearman(vectors):
    """vectors: (n_models, n_items). Median Spearman over all pairs."""
    v = np.asarray(vectors, float)
    rs = [spearman(v[i], v[j]) for i in range(len(v)) for j in range(i + 1, len(v))]
    return float(np.nanmedian(rs)), len(rs)


# --- outputs ------------------------------------------------------------------------------
FORBIDDEN_COLS = {"PTID", "RID", "ptid", "rid", "subject"}


def write_aggregate(df, name, splits_used):
    """Committed table: no subject identifiers; only train/val subjects contributed."""
    assert not (FORBIDDEN_COLS & set(df.columns)), f"{name}: subject identifier column in an aggregate table"
    assert set(splits_used) <= {"train", "val"}, f"{name}: splits {splits_used}"
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    df.to_csv(path, index=False)
    return path


def write_json(obj, name, splits_used=("train", "val")):
    assert set(splits_used) <= {"train", "val"}
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    json.dump(obj, open(path, "w"), indent=1, default=_jsonable)
    return path


def update_json(name, key, value):
    """Merge one section into a results JSON (e.g. card_rules.json, eval_rows.json)."""
    path = os.path.join(RESULTS, name)
    obj = json.load(open(path)) if os.path.exists(path) else {}
    obj[key] = value
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(obj, open(path, "w"), indent=1, default=_jsonable)
    return path


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def subj_path(name):
    os.makedirs(SUBJ, exist_ok=True)
    return os.path.join(SUBJ, name)


def cache_path(name):
    os.makedirs(CACHE, exist_ok=True)
    return os.path.join(CACHE, name)
