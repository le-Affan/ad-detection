"""Edge functions and pruning for the final KAN (config 32, K2) — run with LYP/.venv-kan. Never reads test.

Refits the 25 val-stage KANs (outer seeds 42-46 x KAN seeds 42-46), trained
on the 569 train OOF stream scores exactly as in kan_fusion.py.
K2 is a single-layer KAN, [13 -> 5], added to log p_clinical. Each edge
phi_ij(x_i) is therefore an additive correction to output logit j.
For the MRI-correction inputs (the 4 MRI logits and the has_mri flag):
  - plot every edge to the 4 COG outputs, all 25 models overlaid, x in raw
    centred-logit units over the train P1-P99;
  - per edge: amplitude (max - min over train P5-P95); monotone if, over
    P5-P95, the curve never moves against its overall direction by more than
    5% of its amplitude; consistent if all 25 models are monotone in the same
    direction. Edges with median amplitude < 0.05 logit units count as flat.
  - flag: the jump phi(1) - phi(0) per output.
Pruning: pykan prune_input(threshold=1e-2) on each model; surviving inputs counted.
Outputs: kan_edges_mri.png, kan_edges_all.png, kan_interp.md, kan_interp.json.
"""
import json
import os
import sys
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kan_fusion as KF  # noqa: E402
import metrics as M  # noqa: E402

CID = 32
OUT_NAMES = ["CN", "EMCI", "LMCI", "AD"]


def input_names(cfg):
    return ([f"clin_logit_{n}" for n in OUT_NAMES] + [f"mri_logit_{n}" for n in OUT_NAMES] + ["has_mri"]
            + cfg["extras"])


def one(arg):
    cfg, seed, ks = arg
    trd, ted, trdf, tedf = KF.unit_data(seed, "val")
    for d, df in ((trd, trdf), (ted, tedf)):
        d["extras"] = df.iloc[d["rows"]][cfg["extras"]].to_numpy(dtype=float) if cfg["extras"] else None
    _, _, diag, (model, inp) = KF.fit_predict(cfg, trd, ted, ks)
    Xtr = inp(trd["pc"], trd["pm"], trd["has"], trd["extras"]).numpy()
    names = input_names(cfg)
    res = {"edges": {}, "flag": None}
    for i, nm in enumerate(names):
        if nm == "has_mri":
            X = torch.zeros((2, len(names)))
            X[:, i] = torch.tensor([0.0, 1.0])
            with torch.no_grad():
                model(X)
            e = model.spline_postacts[0][:, :4, i].detach().numpy()
            res["flag"] = (e[1] - e[0]).tolist()
            continue
        rows = trd["has"] if nm.startswith("mri") else np.ones(len(Xtr), bool)
        lo, hi = np.percentile(Xtr[rows, i], [1, 99])
        p5, p95 = np.percentile(Xtr[rows, i], [5, 95])
        xs = np.linspace(lo, hi, 80)
        X = torch.zeros((80, len(names)))
        X[:, i] = torch.tensor(xs, dtype=torch.float32)
        with torch.no_grad():
            model(X)
        e = model.spline_postacts[0][:, :4, i].detach().numpy().T                       # 4 outputs x 80
        raw = xs * inp.sd[i] + inp.mu[i]
        inner = (xs >= p5) & (xs <= p95)
        res["edges"][nm] = {"x": raw.tolist(), "y": e.tolist(), "inner": inner.tolist()}
    with torch.no_grad():
        model(torch.tensor(Xtr, dtype=torch.float32))
    model.attribute(plot=False)
    res["attribution"] = dict(zip(names, model.node_scores[0].detach().numpy().tolist()))
    with torch.no_grad():
        model(torch.tensor(Xtr, dtype=torch.float32))
    pr = model.prune_input(threshold=1e-2, log_history=False)
    res["kept"] = [names[j] for j in pr.input_id.tolist()]
    res["seed"], res["kseed"], res["failed"] = seed, ks, diag["failed"]
    return res


def edge_stats(curves):
    """curves: list of (y over inner range). Returns amplitude median, direction per model, verdict."""
    amps, dirs = [], []
    for y in curves:
        y = np.asarray(y)
        amp = y.max() - y.min()
        net = y[-1] - y[0]
        d = np.sign(net) if abs(net) > 1e-9 else 0
        dy = np.diff(y)
        against = -dy[dy * d < 0].sum() * 1 if d != 0 else amp
        mono = d != 0 and against <= 0.05 * amp + 1e-12
        amps.append(amp)
        dirs.append(int(d) if mono else 0)
    med = float(np.median(amps))
    if med < 0.05:
        verdict = "flat"
    elif all(x == 1 for x in dirs):
        verdict = "monotone increasing, consistent"
    elif all(x == -1 for x in dirs):
        verdict = "monotone decreasing, consistent"
    else:
        verdict = (f"not consistent ({dirs.count(1)} up, {dirs.count(-1)} down, "
                   f"{dirs.count(0)} non-monotone of {len(dirs)})")
    return med, verdict


def main():
    z = np.load(os.path.join(HERE, "kan_runs", f"{CID}.npz"))
    cfg = json.loads(str(z["cfg"]))
    names = input_names(cfg)
    with Pool(15, initializer=KF.init_worker) as pool:
        R = pool.map(one, [(cfg, s, ks) for s in M.SEEDS for ks in KF.KSEEDS])

    # MRI-correction edges
    mri_in = [n for n in names if n.startswith("mri_logit")]
    fig, ax = plt.subplots(4, 4, figsize=(15, 13), sharey=True)
    table = []
    for r, nm in enumerate(mri_in):
        for j in range(4):
            ins = []
            for res in R:
                e = res["edges"][nm]
                ax[r, j].plot(e["x"], e["y"][j], lw=0.6, alpha=0.6,
                              color=plt.cm.tab10(M.SEEDS.index(res["seed"])))
                ins.append(np.asarray(e["y"][j])[np.asarray(e["inner"])])
            med, verdict = edge_stats(ins)
            table.append({"input": nm, "output": OUT_NAMES[j], "amplitude_median": med, "verdict": verdict})
            ax[r, j].set_title(f"{nm} → {OUT_NAMES[j]} logit\namp {med:.2f}; {verdict}", fontsize=7)
            ax[r, j].axhline(0, color="0.7", lw=0.5)
            if r == 3:
                ax[r, j].set_xlabel("MRI centred logit (raw units)", fontsize=8)
    fig.suptitle("Final KAN (K2, config 32): MRI-correction edges, 25 models "
                 "(colour = outer seed; 5 KAN seeds each). Additive corrections to log p_clinical.", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "kan_edges_mri.png"), dpi=85)
    plt.close(fig)

    # all other inputs, edge to the AD and LMCI outputs (context)
    others = [n for n in names if n not in mri_in and n != "has_mri"]
    fig, ax = plt.subplots(len(others), 2, figsize=(10, 2.3 * len(others)))
    for r, nm in enumerate(others):
        for c, j in enumerate((2, 3)):
            ins = []
            for res in R:
                e = res["edges"][nm]
                ax[r, c].plot(e["x"], e["y"][j], lw=0.6, alpha=0.6, color=plt.cm.tab10(M.SEEDS.index(res["seed"])))
                ins.append(np.asarray(e["y"][j])[np.asarray(e["inner"])])
            med, verdict = edge_stats(ins)
            table.append({"input": nm, "output": OUT_NAMES[j], "amplitude_median": med, "verdict": verdict})
            ax[r, c].set_title(f"{nm} → {OUT_NAMES[j]}: amp {med:.2f}; {verdict}", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "kan_edges_all.png"), dpi=80)
    plt.close(fig)

    flag = np.array([res["flag"] for res in R])
    att = np.array([[res["attribution"][n] for n in names] for res in R])
    share = att / att.sum(1, keepdims=True)
    kept = {n: sum(n in res["kept"] for res in R) for n in names}
    L = ["# Final KAN (K2, config 32) — edge functions and pruning\n",
         f"25 models (outer seeds 42–46 × KAN seeds 42–46), trained on the 569 train OOF stream scores; "
         f"failed fits: {sum(r['failed'] for r in R)}.\n",
         "## MRI-correction edges (additive corrections to the clinical log-probabilities)\n",
         "| Input | Output logit | Median amplitude (P5–P95) | Shape across 25 models |", "|---|---|---|---|"]
    for t in table:
        if t["input"].startswith("mri"):
            L.append(f"| {t['input']} | {t['output']} | {t['amplitude_median']:.3f} | {t['verdict']} |")
    L.append(f"\nhas_mri flag jump φ(1) − φ(0), median [min, max] over 25 models, per output CN / EMCI / LMCI / AD: "
             + "; ".join(f"{OUT_NAMES[j]} {np.median(flag[:, j]):+.3f} [{flag[:, j].min():+.3f}, {flag[:, j].max():+.3f}]"
                         for j in range(4)))
    L += ["\n## Other inputs (edges to the LMCI and AD outputs)\n", "| Input | Output | Median amplitude | Shape |", "|---|---|---|---|"]
    for t in table:
        if not t["input"].startswith("mri"):
            L.append(f"| {t['input']} | {t['output']} | {t['amplitude_median']:.3f} | {t['verdict']} |")
    L += ["\n## Attribution and pruning\n", "| Input | Attribution share, mean ± SD | Kept by prune_input (of 25) |", "|---|---|---|"]
    for i in np.argsort(-share.mean(0)):
        L.append(f"| {names[i]} | {share[:, i].mean():.3f} ± {share[:, i].std():.3f} | {kept[names[i]]} |")
    L.append(f"\nSmallest attribution score of any input in any model: {att.min():.4f} (threshold 0.01).")
    open(os.path.join(HERE, "kan_interp.md"), "w").write("\n".join(L) + "\n")
    json.dump({"table": table, "flag": flag.tolist(), "attribution_share_mean": dict(zip(names, share.mean(0).tolist())),
               "kept": kept}, open(os.path.join(HERE, "kan_interp.json"), "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
