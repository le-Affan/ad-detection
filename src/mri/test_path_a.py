"""Checks for the Path A dataset and temporal model.

Run directly (`python src/mri/test_path_a.py`) or under pytest. The dataset
tests read the real visits_index.csv and volumes; the model tests use random
tensors, plus one real batch on the GPU when one is available.

The central property is MASK INVARIANCE: whatever sits in a masked slot --
zeros, noise, another subject's scan -- must not change a subject's output.
Zero-padding alone would pass a weaker test ("output is sane") while still
letting padding leak into the prediction, so the test writes random values
into masked slots and requires the output to be identical.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dataset_visits import INDEX, ADNIVisitSequences  # noqa: E402
from model_temporal import TEMPORAL, TemporalDualHeadNet  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
V9 = os.path.join(ROOT, "analysis", "kaggle_output", "v9", "lr_1e-4", "best.pt")


def _masks():
    """Every visit pattern present in the cohort, plus the extremes."""
    pats = ["1111", "1110", "1011", "1101", "0111", "1100", "1010", "0110", "1000", "0001"]
    return torch.tensor([[c == "1" for c in p] for p in pats])


# ---------------------------------------------------------------- dataset

def test_dataset_grid_and_splits():
    idx = pd.read_csv(INDEX)
    seen = set()
    for split in ("train", "val", "test"):
        ds = ADNIVisitSequences(split)
        assert not (seen & set(ds.ptids)), "a subject is in two splits"
        seen |= set(ds.ptids)
    assert seen == set(idx.PTID), "splits do not cover every subject"
    assert len(seen) == 690


def test_masked_slots_are_never_opened():
    ds = ADNIVisitSequences(split=None)
    opened = []
    real = ds.load_volume
    ds.load_volume = lambda r: (opened.append(r.prep_key), real(r))[1]
    s = pd.read_csv(INDEX).groupby("PTID").present.apply(lambda x: "".join("1" if v else "0" for v in x))
    for ptid in [s[s == "1011"].index[0], s[s == "0111"].index[0], "027_S_4966"]:
        opened.clear()
        item = ds[ds.ptids.index(ptid)]
        m = item["mask"]
        assert len(opened) == int(m.sum()), f"{ptid}: opened {len(opened)} for {int(m.sum())} usable"
        assert torch.all(item["volumes"][~m] == 0)
        assert torch.all(item["days"][~m] == 0)
        assert all(k.startswith(ptid) for k in opened)


def test_item_shapes_and_days():
    ds = ADNIVisitSequences("val")
    it = ds[0]
    assert it["volumes"].shape == (4, 1, 96, 96, 96) and it["volumes"].dtype == torch.float32
    assert it["mask"].dtype == torch.bool and it["mask"].shape == (4,)
    d = it["days"][it["mask"]]
    assert d[0] == 0 and torch.all(torch.diff(d) > 0), "days must start at 0 and increase"


def test_days_restart_at_first_usable_visit():
    # 027_S_4966's bl was masked after stage A; its clock must start at m06
    ds = ADNIVisitSequences(split=None)
    it = ds[ds.ptids.index("027_S_4966")]
    assert not it["mask"][0] and it["mask"][1]
    assert it["days"][1] == 0


def test_targets_match_static_dataset():
    from dataset import ADNIVolumes
    stat = ADNIVolumes("val").df.set_index("PTID")
    ds = ADNIVisitSequences("val")
    for i in range(0, len(ds), 25):
        it = ds[i]
        assert int(it["label"]) == int(stat.label[it["ptid"]])


# ---------------------------------------------------------------- model

def _random_inputs(mask, d, seed=0):
    g = torch.Generator().manual_seed(seed)
    emb = torch.randn(mask.shape + (d,), generator=g)
    days = torch.cumsum(torch.rand(mask.shape, generator=g) * 400 + 100, dim=1)
    # days relative to first usable visit, 0 where masked (as the dataset does)
    first = torch.where(mask, days, torch.inf).min(dim=1, keepdim=True).values
    days = torch.where(mask, days - first, torch.zeros_like(days))
    return emb, days


def test_mask_invariance_all_temporal():
    mask = _masks()
    for kind in TEMPORAL:
        torch.manual_seed(0)
        net = TemporalDualHeadNet(kind, arch="resnet10w").eval()
        d = net.trunk.out_features
        emb, days = _random_inputs(mask, d)
        emb = emb * mask.unsqueeze(-1)                       # what encode() produces
        with torch.no_grad():
            a = net.forward_embeddings(emb, mask, days)
            noisy_emb = torch.where(mask.unsqueeze(-1), emb, torch.randn_like(emb) * 50)
            noisy_days = torch.where(mask, days, torch.rand_like(days) * 5000)
            b = net.forward_embeddings(noisy_emb, mask, noisy_days)
        for k in ("demo", "alz_logit"):
            assert torch.allclose(a[k], b[k], atol=1e-5), f"{kind}: masked slots leak into {k}"


def test_single_visit_equals_any_slot():
    # one usable visit: the output must not depend on WHICH slot holds it
    for kind in TEMPORAL:
        net = TemporalDualHeadNet(kind, arch="resnet10w").eval()
        e = torch.randn(1, net.trunk.out_features)
        outs = []
        for s in range(4):
            mask = torch.zeros(1, 4, dtype=torch.bool)
            mask[0, s] = True
            emb = torch.zeros(1, 4, e.shape[1])
            emb[0, s] = e
            with torch.no_grad():
                outs.append(net.forward_embeddings(emb, mask, torch.zeros(1, 4))["demo"])
        assert torch.allclose(torch.stack(outs), outs[0].expand(4, 1), atol=1e-5), kind


def test_encode_skips_masked_volumes():
    net = TemporalDualHeadNet("gru", arch="resnet10w").eval()
    calls = []
    net.trunk.register_forward_hook(lambda m, i, o: calls.append(i[0].shape[0]))
    mask = _masks()[:3]
    vols = torch.rand(3, 4, 1, 32, 32, 32)
    with torch.no_grad():
        emb = net.encode(vols, mask)
    assert calls == [int(mask.sum())], "trunk saw masked volumes"
    assert torch.all(emb[~mask] == 0)


def test_rejects_empty_subject():
    net = TemporalDualHeadNet("delta", arch="resnet10w")
    try:
        net.forward_embeddings(torch.zeros(1, 4, 128), torch.zeros(1, 4, dtype=torch.bool), torch.zeros(1, 4))
    except ValueError:
        return
    raise AssertionError("an all-masked subject must raise")


def test_load_v9_trunk():
    if not os.path.exists(V9):
        print("  (skipped: v9 checkpoint not on disk)")
        return
    net = TemporalDualHeadNet("gru", arch="resnet18w")
    args = net.load_trunk(V9)
    assert args.get("arch") == "resnet18w"
    ck = torch.load(V9, map_location="cpu", weights_only=False)["model"]
    assert torch.equal(net.trunk.stem[0].weight, ck["trunk.stem.0.weight"])


def test_real_batch_gpu_step():
    if not torch.cuda.is_available():
        print("  (skipped: no GPU)")
        return
    from torch.utils.data import DataLoader
    from model import dual_head_loss
    ds = ADNIVisitSequences("val")
    batch = next(iter(DataLoader(ds, batch_size=2, shuffle=False)))
    net = TemporalDualHeadNet("attn", arch="resnet18w").cuda()
    dev = {k: v.cuda() for k, v in batch.items() if torch.is_tensor(v)}
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda"):
        out = net(dev["volumes"], dev["mask"], dev["days"])
    loss, parts = dual_head_loss({k: v.float() for k, v in out.items()}, dev)
    loss.backward()
    assert np.isfinite(loss.item())
    assert net.trunk.stem[0].weight.grad is not None
    print(f"  batch of 2 subjects ({int(dev['mask'].sum())} visits): loss {loss.item():.3f}, "
          f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    fails = 0
    for name, fn in [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{fails} failed")
    sys.exit(1 if fails else 0)
