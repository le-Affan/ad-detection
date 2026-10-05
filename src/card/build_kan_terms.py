"""Card data, step 2 of 3 (run with LYP/.venv-kan): the frozen KAN fusion's outputs
and per-edge terms for every subject. No model is refitted; no metric is computed.

For each of the 25 frozen KANs (outer seed s x KAN seed k), applied to the
seed-s specialists' scores (demo/card_data/streams_all.npz):
  - COG probabilities and the ADD probability; the final output is the mean over
    the 25 KANs (as in test_kan.py; val/test must reproduce frozen_run);
  - the residual (KAN outputs 0-3, added to log p_clinical) and every edge's
    output phi_ij(x_i). K2 is a single layer, so output j = sum_i phi_ij(x_i)
    (pykan's affine node parameters are fixed at scale 1, bias 0; checked).
Edge curves (kan_edges.json): the MRI-logit edges that kan_interpret.py found
monotone and consistent in all 25 val-stage KANs (kan_interp.json), re-checked
with the same rule on the 25 frozen KANs over the 691 training inputs, with each
passing edge's curve (mean, min and max over the 25 KANs) on a grid over the raw
input's P1-P99. They are kept as a record and are NOT drawn on the card: the card
explains the fusion with group-level Shapley values (kan_shapley.npz below),
showing a group only where all 25 KANs agree in sign.
Shapley values (analysis/explainability, P3): exact interventional Shapley values of each frozen KAN over
the 4 input groups (clinical logits; MRI logits + has_mri; age; CDRSB + CogIndex + MMSE) for the centred final
COG logits, the centred residual and the ADD logit; background = the train subjects' prediction-time inputs.
Also the ensemble prediction with the MRI inputs masked (the missing-scan input: neutral logits, has_mri = 0),
for the card's "MRI changed the stage" line (P4-B).
Outputs (demo/card_data/): kan_terms.npz (final p, ADD, residual, per-edge
terms, clinical base, raw inputs; means over the 25 KANs), kan_edges.json, kan_shapley.npz.
"""
import json
import os
import pickle
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
V3 = os.path.join(ROOT, "analysis", "fusion_v3")
sys.path.insert(0, V3)
import kan_fusion as KF  # noqa: E402
import metrics as M  # noqa: E402
from kan_interpret import edge_stats, input_names  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "analysis", "explainability"))
import common as XC  # noqa: E402
import kan_shap_lib as KS  # noqa: E402

FROZEN = os.path.join(V3, "frozen")
OUT = os.path.join(ROOT, "demo", "card_data")
NEVER_SHOW = {("mri_logit_AD", "AD")}


def load(s, ks, cfg):
    m = KF.build(cfg, 9 + len(cfg["extras"]), ks)
    m.load_state_dict(torch.load(os.path.join(FROZEN, f"kan_{s}_{ks}.pt")))
    inp = pickle.load(open(os.path.join(FROZEN, f"kan_inputs_{s}_{ks}.pkl"), "rb"))
    for p in (m.node_scale + m.subnode_scale):
        assert torch.allclose(p, torch.ones_like(p))
    for p in (m.node_bias + m.subnode_bias):
        assert torch.allclose(p, torch.zeros_like(p))
    return m, inp


def main():
    torch.set_num_threads(4)
    cfg = json.load(open(os.path.join(FROZEN, "kan_config.json")))
    names = input_names(cfg)
    z = np.load(os.path.join(OUT, "streams_all.npz"), allow_pickle=True)
    fin = np.load(os.path.join(FROZEN, "streams_final.npz"))
    has, ex = z["has"], z["extras"]
    P, A, RES, EDGE, BASE, XRAW = [], [], [], [], [], []
    curves, display = {}, {}
    grid = None
    for s in M.SEEDS:
        for ks in KF.KSEEDS:
            m, inp = load(s, ks, cfg)
            with torch.no_grad():
                o = m(inp(z[f"clin_{s}"], z[f"mri_{s}"], has, ex))
                e = m.spline_postacts[0][:, :, :].numpy()                    # (n, 5 outputs, 13 inputs)
                base = inp.base(z[f"clin_{s}"])
                P.append(torch.softmax(KF.cog_logits(cfg, o, base), 1).numpy())
                A.append(torch.sigmoid(o[:, 4]).numpy())
            assert np.abs(e.sum(2) - o.numpy()).max() < 1e-4, "edge sum != output"
            RES.append(o[:, :4].numpy())
            EDGE.append(e[:, :4, :])
            BASE.append(base.numpy())
            XRAW.append(inp.raw(z[f"clin_{s}"], z[f"mri_{s}"], has, ex))
            if grid is None:                                                  # display grid: raw P1-P99, 691 with a scan
                r = inp.raw(fin[f"clin_oof_{s}"], fin[f"mri_oof_{s}"], fin["has"], fin["extras"])[fin["has"]]
                grid = {i: np.linspace(*np.percentile(r[:, i], [1, 99]), 60) for i in range(4, 8)}
            for i in range(4, 8):                                             # edge curves on the raw display grid
                X = torch.zeros((60, len(names)))
                X[:, i] = torch.tensor((grid[i] - inp.mu[i]) / inp.sd[i], dtype=torch.float32)
                with torch.no_grad():
                    m(X)
                for j in range(4):
                    display.setdefault((names[i], M.NAMES[j]), []).append(m.spline_postacts[0][:, j, i].numpy())
            # edge curves over the frozen KAN's own training inputs (691), for the consistency re-check
            Xtr = inp(fin[f"clin_oof_{s}"], fin[f"mri_oof_{s}"], fin["has"], fin["extras"]).numpy()
            for i, nm in enumerate(names):
                if not nm.startswith("mri_logit"):
                    continue
                p5, p95 = np.percentile(Xtr[fin["has"], i], [5, 95])
                X = torch.zeros((60, len(names)))
                X[:, i] = torch.linspace(float(p5), float(p95), 60)
                with torch.no_grad():
                    m(X)
                y = m.spline_postacts[0][:, :4, i].numpy().T
                for j in range(4):
                    curves.setdefault((nm, M.NAMES[j]), []).append(y[j])
    p = np.mean(P, 0)
    pos = {r: i for i, r in enumerate(z["rids"])}
    for sp in ("val", "test"):                                               # card = frozen-run outputs
        k = np.load(os.path.join(V3, "frozen_run", f"{sp}_kan.npz"))
        rows = np.array([pos[r] for r in np.load(os.path.join(V3, "frozen_run", f"{sp}_streams.npz"))["rids"]])
        # val ALZ here comes from the train/val tables (frozen_run used scan_inputs_val.csv; |diff| <= 6e-7)
        assert np.abs(p[rows] - k["p"]).max() < 1e-4 and np.abs(np.mean(A, 0)[rows] - k["add"]).max() < 1e-4
    analysed = {(r["input"], r["output"]): r for r in json.load(open(os.path.join(V3, "kan_interp.json")))["table"]}
    edges = []
    for (nm, out), cs in curves.items():
        med, verdict = edge_stats(cs)
        a = analysed[(nm, out)]
        ok = ("consistent" in a["verdict"] and "not consistent" not in a["verdict"]
              and "consistent" in verdict and "not consistent" not in verdict and (nm, out) not in NEVER_SHOW)
        edges.append({"input": nm, "output": out, "input_index": names.index(nm), "output_index": M.NAMES.index(out),
                      "verdict_val_stage": a["verdict"], "verdict_frozen": verdict,
                      "amplitude_frozen": med, "shown": bool(ok)})
        if ok:
            ys = np.array(display[(nm, out)])
            edges[-1]["curve"] = {"x": grid[names.index(nm)].tolist(), "y_mean": ys.mean(0).tolist(),
                                  "y_min": ys.min(0).tolist(), "y_max": ys.max(0).tolist()}
    json.dump({"inputs": names, "edges": edges}, open(os.path.join(OUT, "kan_edges.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(OUT, "kan_terms.npz"), rids=z["rids"], p=p.astype(np.float32),
                        add=np.mean(A, 0).astype(np.float32), residual=np.mean(RES, 0).astype(np.float32),
                        edge=np.mean(EDGE, 0).astype(np.float32), base=np.mean(BASE, 0).astype(np.float32),
                        x_raw=np.mean(XRAW, 0).astype(np.float32))
    shapley(z)
    for e in edges:
        if "consistent" in e["verdict_val_stage"] and "not" not in e["verdict_val_stage"] or e["shown"]:
            print(f"{e['input']:>14s} -> {e['output']:4s} val-stage: {e['verdict_val_stage']:34s} "
                  f"frozen: {e['verdict_frozen']:40s} shown {e['shown']}")


def shapley(z):
    """Per-subject group Shapley values of the 25 frozen KANs and the MRI-masked ensemble prediction."""
    sp = XC.split_of()
    has, ex = z["has"].astype(bool), z["extras"].astype(float)
    bg = np.array([sp[int(r)] == "train" for r in z["rids"]])
    config = KS.cfg()
    G, P_mask = [], []
    for s, k in KS.MODELS:
        m, inp = KS.load_frozen(s, k, config)
        T, f, out, base = KS.terms(m, inp, z[f"clin_{s}"], z[f"mri_{s}"], has, ex)
        phi, ebg = KS.shapley(T, T[bg])
        assert np.abs(phi.sum(1) + ebg - f).max() < 1e-4, "efficiency"
        G.append(KS.group(phi))
        with torch.no_grad():
            o2 = m(inp(z[f"clin_{s}"], z[f"mri_{s}"], np.zeros_like(has), ex)).numpy()
        P_mask.append(KS.softmax(o2[:, :4] + base.astype(np.float64)))
    np.savez_compressed(os.path.join(OUT, "kan_shapley.npz"), rids=z["rids"], phi_group=np.array(G, np.float32),
                        p_mri_masked=np.mean(P_mask, 0).astype(np.float32), groups=np.array(KS.GROUP_NAMES),
                        targets=np.array(KS.TARGETS), n_background=int(bg.sum()))
    print(f"kan_shapley.npz: {len(G)} models x {len(z['rids'])} subjects; background {int(bg.sum())} train")


if __name__ == "__main__":
    main()
