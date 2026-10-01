"""KAN stage of the KAN-fusion experiment (preregistration.md, addendum K1-K4). Never reads test.

For input sets C and F, seeds 42-46 and outer units (the 5 CV folds of each
seed, as in catboost_stage.py's topk.json, plus "val" = all 569 train -> 122
val), one job does:
  nested choice of (G, lambda, mu) by inner StratifiedKFold(3) macro-F1 on the
  outer training portion (K3), refit of the chosen config on the whole outer
  training portion with a 15% early-stopping holdout (K2), prediction of the
  held-out rows. For F/val it also computes the interpretability outputs (K4):
  edge functions, partial dependence of severity and P(AD), input attribution,
  prune_input.

Jobs run in parallel processes with one torch thread each.
Outputs: kan_preds.npz, kan_runs.json, kan_interp.json. Run with LYP/.venv-kan.
"""
import argparse
import itertools
import json
import os
import tempfile
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = (42, 43, 44, 45, 46)
GRID = list(itertools.product((3, 5), (0.5, 1.0), (0.0, 1e-3)))     # (G, lambda, mu), K1 order
HIDDEN, MAX_STEPS, PATIENCE, LR, HOLDOUT = 8, 400, 40, 0.01, 0.15
GRID_UPDATE_EVERY, GRID_UPDATE_UNTIL = 5, 50
PLOT_MRI = ["ALZ", "Hippocampus_ICV", "Ventricles_ICV"]

D = TOPK = None


def init_worker():
    global D, TOPK
    torch.set_num_threads(1)
    D = pd.read_csv(os.path.join(HERE, "fusion_trainval.csv"))
    TOPK = json.load(open(os.path.join(HERE, "topk.json")))


def macro_f1(y, p):
    f = []
    for k in range(4):
        tp = np.sum((p == k) & (y == k))
        pr, rc = tp / max(np.sum(p == k), 1), tp / max(np.sum(y == k), 1)
        f.append(0 if pr + rc == 0 else 2 * pr * rc / (pr + rc))
    return float(np.mean(f))


class Prep:
    """K2 step 1: fitted on the training rows only."""

    def __init__(self, df, cols):
        self.cols = cols
        x = self.raw(df)
        self.median = np.nanmedian(x, axis=0)
        x = np.where(np.isnan(x), self.median, x)
        self.mu, self.sd = x.mean(0), x.std(0, ddof=1)
        self.sd[self.sd == 0] = 1.0

    def raw(self, df):
        x = df[self.cols].copy()
        if "PTGENDER" in self.cols:
            x["PTGENDER"] = (x["PTGENDER"] == "Male").astype(float)
        return x.to_numpy(dtype=float)

    def __call__(self, df):
        x = self.raw(df)
        x = np.where(np.isnan(x), self.median, x)
        return torch.tensor((x - self.mu) / self.sd, dtype=torch.float32)


def make_model(n_in, G, seed, ckpt):
    from kan import MultKAN
    torch.manual_seed(seed)
    return MultKAN(width=[n_in, HIDDEN, 5], grid=G, k=3, seed=seed, auto_save=False,
                   symbolic_enabled=False, save_act=True, ckpt_path=ckpt)


def task_loss(out, y, lam):
    return Fn.cross_entropy(out[:, :4], y) + lam * Fn.binary_cross_entropy_with_logits(out[:, 4], (y == 3).float())


def fit_one(df_tr, cols, cfg, seed, ckpt):
    """K2: one fit on the training rows df_tr. Returns (model, prep, diagnostics)."""
    G, lam, mu = cfg
    prep = Prep(df_tr, cols)
    y_all = df_tr.label.to_numpy()
    i_fit, i_ho = train_test_split(np.arange(len(df_tr)), test_size=HOLDOUT, stratify=y_all, random_state=seed)
    X = prep(df_tr)
    xf, xh = X[i_fit], X[i_ho]
    yf, yh = torch.tensor(y_all[i_fit]), torch.tensor(y_all[i_ho])

    # reference: predicting the fitting rows' class frequencies
    freq = torch.bincount(yf, minlength=4).float() / len(yf)
    p_ad = (yf == 3).float().mean()
    prior_loss = float(Fn.nll_loss(torch.log(freq)[None].expand(len(yh), 4), yh)
                       + lam * Fn.binary_cross_entropy(p_ad.expand(len(yh)), (yh == 3).float()))

    model = make_model(len(cols), G, seed, ckpt)
    model.update_grid(xf)
    opt = torch.optim.Adam(model.get_params(), lr=LR)
    best, best_step, best_state, finite = np.inf, -1, None, True
    for step in range(MAX_STEPS):
        if 0 < step <= GRID_UPDATE_UNTIL and step % GRID_UPDATE_EVERY == 0:
            model.update_grid(xf)
        opt.zero_grad()
        out = model(xf)
        loss = task_loss(out, yf, lam)
        if mu > 0:
            loss = loss + mu * model.get_reg("edge_forward_spline_n", 1.0, 2.0, 0.0, 0.0)
        if not torch.isfinite(loss):
            finite = False
            break
        loss.backward()
        opt.step()
        with torch.no_grad():
            h = float(task_loss(model(xh), yh, lam))
        if not np.isfinite(h):
            finite = False
            break
        if h < best:
            best, best_step = h, step
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif step - best_step >= PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    failed = (not finite) or best_step <= 0 or not (best < prior_loss)
    reason = "non-finite" if not finite else "best step 0" if best_step <= 0 else \
        "no better than class prior" if not (best < prior_loss) else ""
    return model, prep, {"best_step": int(best_step), "stopped_step": int(step), "best_holdout_loss": float(best),
                         "prior_loss": prior_loss, "failed": bool(failed), "reason": reason}


def predict(model, prep, df):
    with torch.no_grad():
        out = model(prep(df))
    return torch.softmax(out[:, :4], 1).numpy(), torch.sigmoid(out[:, 4]).numpy()


def interpret(model, prep, df_tr, df_va, cols, clinical_top3):
    """K4 on one all-569 F model."""
    out = {"cols": cols}
    raw_tr = prep.raw(df_tr)
    raw_tr = np.where(np.isnan(raw_tr), prep.median, raw_tr)
    curves = {}
    for c in cols:
        i = cols.index(c)
        lo, hi = np.percentile(raw_tr[:, i], [1, 99])
        p5, p95 = np.percentile(raw_tr[:, i], [5, 95])
        res = {}
        for name, (a, b, n) in (("plot", (lo, hi, 100)), ("flat", (p5, p95, 50))):
            xs = np.linspace(a, b, n) if hi > lo else np.full(n, lo)
            X = torch.zeros((n, len(cols)))
            X[:, i] = torch.tensor((xs - prep.mu[i]) / prep.sd[i], dtype=torch.float32)
            with torch.no_grad():
                o = model(X)
            p = torch.softmax(o[:, :4], 1).numpy()
            res[name] = {"x": xs.tolist(), "severity": (p @ np.arange(4)).tolist(), "p_ad": p[:, 3].tolist()}
            if name == "plot" and c in PLOT_MRI + clinical_top3:
                res["edges"] = model.spline_postacts[0][:, :, i].numpy().T.tolist()    # 8 x n
        res["severity_range_p5_p95"] = float(np.ptp(res["flat"]["severity"]))
        del res["flat"]
        curves[c] = res
    out["curves"] = curves

    xtr = prep(df_tr)
    with torch.no_grad():
        model(xtr)
    model.attribute(plot=False)
    score = model.node_scores[0].detach().numpy()
    out["attribution"] = dict(zip(cols, score.tolist()))
    with torch.no_grad():
        model(xtr)
    pruned = model.prune_input(threshold=1e-2, log_history=False)
    kept = [cols[i] for i in pruned.input_id.tolist()]
    with torch.no_grad():
        pv = torch.softmax(pruned(prep(df_va))[:, :4], 1).numpy()
    out["pruned_kept"] = kept
    out["pruned_val_macro_f1"] = macro_f1(df_va.label.to_numpy(), pv.argmax(1))
    return out


def job(spec):
    s, seed, unit = spec
    t0 = time.time()
    tr = D[D.split == "train"].reset_index(drop=True)
    va = D[D.split == "val"].reset_index(drop=True)
    if unit == "val":
        top = TOPK["final_list"]
        df_tr, df_te = tr, va
    else:
        key = f"{seed}_{unit}"
        top = TOPK["fold_rankings"][key][:TOPK["k_star"]]
        f = TOPK["folds"][key]
        df_tr, df_te = tr.iloc[f["train"]].reset_index(drop=True), tr.iloc[f["test"]].reset_index(drop=True)
    cols = top + (TOPK["mri_inputs"] if s == "F" else [])
    ckpt = tempfile.mkdtemp(prefix="kanckpt_")

    # K3: inner 3-fold choice
    y = df_tr.label.to_numpy()
    inner = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(df_tr, y))
    scores, inner_fail = {}, 0
    for cfg in GRID:
        pred = np.zeros(len(df_tr), dtype=int)
        for a, b in inner:
            m, prep, dg = fit_one(df_tr.iloc[a].reset_index(drop=True), cols, cfg, seed, ckpt)
            inner_fail += dg["failed"]
            pred[b] = predict(m, prep, df_tr.iloc[b])[0].argmax(1)
        scores[str(cfg)] = macro_f1(y, pred)
    best_cfg = max(GRID, key=lambda c: scores[str(c)])               # max() keeps the first on ties

    model, prep, diag = fit_one(df_tr, cols, best_cfg, seed, ckpt)
    p_cog, p_add = predict(model, prep, df_te)
    res = {"set": s, "seed": seed, "unit": unit, "cols": cols, "config": list(best_cfg), "inner_scores": scores,
           "inner_failures": int(inner_fail), "refit": diag, "p_cog": p_cog.tolist(), "p_add": p_add.tolist(),
           "test_idx": (list(range(len(va))) if unit == "val" else TOPK["folds"][f"{seed}_{unit}"]["test"]),
           "seconds": round(time.time() - t0, 1)}
    if s == "F" and unit == "val":
        res["interp"] = interpret(model, prep, df_tr, df_te, cols, TOPK["final_list"][:3])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()
    specs = [(s, seed, u) for s in ("C", "F") for seed in SEEDS for u in ["val", 0, 1, 2, 3, 4]]
    t0 = time.time()
    results = []
    with Pool(a.workers, initializer=init_worker) as pool:
        for r in pool.imap_unordered(job, specs):
            results.append(r)
            print(f"[{len(results)}/{len(specs)}] {r['set']} seed {r['seed']} {r['unit']}: cfg {r['config']} "
                  f"best step {r['refit']['best_step']} failed {r['refit']['failed']} "
                  f"inner fails {r['inner_failures']}/24  {r['seconds']} s", flush=True)
    print(f"total {time.time() - t0:.0f} s")

    d = pd.read_csv(os.path.join(HERE, "fusion_trainval.csv"))
    n_tr, n_va = (d.split == "train").sum(), (d.split == "val").sum()
    arrays = {}
    for s in ("C", "F"):
        for kind, n in (("oof", n_tr), ("val", n_va)):
            arrays[f"{kind}_{s}"] = np.full((len(SEEDS), n, 4), np.nan)
            arrays[f"{kind}_add_{s}"] = np.full((len(SEEDS), n), np.nan)
    for r in results:
        si = SEEDS.index(r["seed"])
        kind = "val" if r["unit"] == "val" else "oof"
        arrays[f"{kind}_{r['set']}"][si, r["test_idx"]] = r["p_cog"]
        arrays[f"{kind}_add_{r['set']}"][si, r["test_idx"]] = r["p_add"]
    assert all(np.isfinite(v).all() for v in arrays.values())
    np.savez(os.path.join(HERE, "kan_preds.npz"), **arrays)
    runs = [{k: v for k, v in r.items() if k not in ("p_cog", "p_add", "test_idx", "interp")} for r in results]
    runs.sort(key=lambda r: (r["set"], r["seed"], str(r["unit"])))
    json.dump(runs, open(os.path.join(HERE, "kan_runs.json"), "w"), indent=1)
    interp = {r["seed"]: r["interp"] for r in results if "interp" in r}
    json.dump(interp, open(os.path.join(HERE, "kan_interp.json"), "w"))


if __name__ == "__main__":
    main()
