"""Exact interventional Shapley values for the K2 KAN fusion (LYP/.venv-kan).

K2 is a single additive layer [13 -> 5] (node scale 1, bias 0, asserted at load): out_j = sum_i e_ij(x_i).
COG logits z = log p_clinical + out[:4] = Lc - lse(Lc) + out[:4], with Lc the centred clinical log-probabilities
(the raw clinical-logit inputs; temperature 1). Centring over the 4 classes removes lse, so every target below
is a sum of per-input terms and Shapley values are exact and closed-form:
  T_i(x) for target t; phi_i = T_i(x) - mean_bg T_i(b); group value = sum over the group's inputs.
Targets (9): centred final COG logits (CN, EMCI, LMCI, AD), centred residual (out[:4]) (4), ADD logit (out[4]).
Groups: clinical logits (4); MRI logits (4) + has_mri; AGE; CDRSB + CogIndex + MMSE.
"""
import json
import os
import pickle
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

sys.path.insert(0, C.V3)
import kan_fusion as KF  # noqa: E402

INPUTS = ["clin_logit_CN", "clin_logit_EMCI", "clin_logit_LMCI", "clin_logit_AD", "mri_logit_CN", "mri_logit_EMCI",
          "mri_logit_LMCI", "mri_logit_AD", "has_mri", "AGE", "CDRSB", "CogIndex", "MMSE"]
GROUPS = {"clinical logits": [0, 1, 2, 3], "MRI logits + has_mri": [4, 5, 6, 7, 8], "age": [9],
          "CDRSB + CogIndex + MMSE": [10, 11, 12]}
GROUP_NAMES = list(GROUPS)
TARGETS = [f"final_{n}" for n in C.NAMES] + [f"residual_{n}" for n in C.NAMES] + ["ADD"]
MODELS = [(s, k) for s in (42, 43, 44, 45, 46) for k in KF.KSEEDS]


def cfg():
    return json.load(open(os.path.join(C.FROZEN, "kan_config.json")))


def load_frozen(s, k, config=None):
    config = config or cfg()
    m = KF.build(config, 9 + len(config["extras"]), k)
    m.load_state_dict(torch.load(os.path.join(C.FROZEN, f"kan_{s}_{k}.pt")))
    inp = pickle.load(open(os.path.join(C.FROZEN, f"kan_inputs_{s}_{k}.pkl"), "rb"))
    for p in (m.node_scale + m.subnode_scale):
        assert torch.allclose(p, torch.ones_like(p))
    for p in (m.node_bias + m.subnode_bias):
        assert torch.allclose(p, torch.zeros_like(p))
    return m, inp


def terms(m, inp, pc, pm, has, ex):
    """Per-input target terms T (n, 13, 9), plus the model's own outputs for checks."""
    X = inp(pc, pm, has, ex)
    with torch.no_grad():
        out = m(X).numpy().astype(np.float64)
        E = m.spline_postacts[0].numpy().astype(np.float64).transpose(0, 2, 1)      # (n, 13 inputs, 5 outputs)
    assert np.abs(E.sum(1) - out).max() < 1e-4, "edge sum != output"
    Lc = inp.raw(pc, pm, has, ex)[:, :4]
    fin = E[:, :, :4].copy()
    fin[:, np.arange(4), np.arange(4)] += Lc
    fin -= fin.mean(2, keepdims=True)
    res = E[:, :, :4] - E[:, :, :4].mean(2, keepdims=True)
    T = np.concatenate([fin, res, E[:, :, 4:5]], 2)
    base = inp.base(pc).numpy().astype(np.float64)
    z = out[:, :4] + base
    f = np.concatenate([z - z.mean(1, keepdims=True), out[:, :4] - out[:, :4].mean(1, keepdims=True), out[:, 4:5]], 1)
    return T, f, out, base


def shapley(T, T_bg):
    """per-input phi (n, 13, 9) and E_bg f (9,)"""
    return T - T_bg.mean(0, keepdims=True), T_bg.sum(1).mean(0)


def group(phi):
    return np.stack([phi[:, ix].sum(1) for ix in GROUPS.values()], 1)              # (n, 4 groups, 9)


def brute_force_group_shapley(m, inp, x, bg):
    """Independent check: 16-coalition interventional values over the 4 groups for one subject.
    x / bg: dicts with pc, pm, has, ex (x: 1 row; bg: background rows). Returns (4 groups, 9 targets)."""
    from itertools import combinations
    from math import factorial
    nb = len(bg["pc"])

    def v(S):
        take = np.zeros(13, bool)
        for g in S:
            take[GROUPS[GROUP_NAMES[g]]] = True
        Xr_x = inp.raw(x["pc"], x["pm"], x["has"], x["ex"])
        Xr_b = inp.raw(bg["pc"], bg["pm"], bg["has"], bg["ex"])
        Xr = np.where(take[None], np.repeat(Xr_x, nb, 0), Xr_b)
        X = torch.tensor((Xr - inp.mu) / inp.sd, dtype=torch.float32)
        with torch.no_grad():
            out = m(X).numpy().astype(np.float64)
        Lc = Xr[:, :4]
        z = out[:, :4] + Lc
        f = np.concatenate([z - z.mean(1, keepdims=True), out[:, :4] - out[:, :4].mean(1, keepdims=True),
                            out[:, 4:5]], 1)
        return f.mean(0)
    n = 4
    vals = {S: v(S) for r in range(n + 1) for S in combinations(range(n), r)}
    phi = np.zeros((n, 9))
    for g in range(n):
        for S, val in vals.items():
            if g in S:
                continue
            w = factorial(len(S)) * factorial(n - len(S) - 1) / factorial(n)
            phi[g] += w * (vals[tuple(sorted(S + (g,)))] - val)
    return phi


def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)
