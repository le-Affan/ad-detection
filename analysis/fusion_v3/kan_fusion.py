"""KAN fusion on stream scores (run with LYP/.venv-kan). Never reads test.

Inputs: clinical logits (4), MRI logits (4), has_mri flag (+ optional extras).
  Logit = log(clip(p, 1e-4)) centred per row. Optional per-stream temperature T,
  fitted on the training rows by NLL (grid of 241 values, 0.25-4), gives
  logit / T.
  MRI rows without a scan get the "neutral" logits: centred log of the training
  rows' class prior. Their flag is 0.
  scale=True: every non-flag input is z-scored with the training rows' mean and
  SD (MRI columns over rows with a scan), and the KAN grid is set from the
  training rows (update_grid at step 0 and every 5 steps up to 50).
  scale=False: raw logits, pykan's default grid [-1, 1], no grid update.
Variants:
  K1  MultKAN [n_in, 5]: no hidden layer.
  K2  residual: COG logits = log p_clin (temperature-scaled if temp) + KAN[n_in, 5] outputs 0-3.
  K3  MultKAN [n_in, 4, 5], sparsity penalty mu > 0 required.
  Output 4 is the ADD logit (AD vs rest), with loss weight 0.5.
no_mri=True (ablation): the MRI stream is masked (neutral, flag 0) for every row,
  train and test, so the KAN sees clinical logits (+ extras) only.
Loss: CE(COG; label smoothing ls, class-balanced weights if cb)
  + ord * squared EMD between cumulative predicted and true distributions
  + 0.5 * BCE(ADD) + mu * pykan reg('edge_forward_spline_n', 1, 2, 0, 0).
Training: Adam lr 0.01, full batch, at most 400 steps, early stopping on a
  stratified 15% holdout of the training rows (patience 40, best state
  restored; holdout scored without dropout). Modality dropout: each step, 15%
  of the fitting rows get neutral MRI inputs and flag 0.
Seeds: for every outer unit, KAN seeds 42-46 are trained and their
  probabilities averaged (seed ensemble).
CV: for outer seed s, fold f, the KAN trains on the inner OOF stream scores of
  the outer-train set and predicts the outer-test fold from the full-fit
  specialists' scores (streams.npz).
Val: the KAN trains on clin_oof[s] / mri_oof[s] (569) and predicts val.

Usage: python kan_fusion.py --configs configs.json   (a list of config dicts)
Each config is logged to config_log.csv; predictions go to kan_runs/<config_id>.npz.
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import metrics as M  # noqa: E402

KSEEDS = (42, 43, 44, 45, 46)
LR, MAX_STEPS, PATIENCE, HOLDOUT, DROPOUT, LAM_ADD = 0.01, 400, 40, 0.15, 0.15, 0.5
DEFAULT = dict(variant="K1", G=3, mu=1e-3, scale=True, temp=False, extras=[], ls=0.0, cb=False, ord=0.0, no_mri=False)
T_GRID = np.exp(np.linspace(np.log(0.25), np.log(4), 241))

ST = DATA = None


def init_worker():
    global ST, DATA
    torch.set_num_threads(1)
    ST = dict(np.load(os.path.join(HERE, "streams.npz")))
    d = pd.read_csv(os.path.join(ROOT, "analysis", "kan_fusion", "fusion_trainval.csv"))
    DATA = {s: d[d.split == s].reset_index(drop=True) for s in ("train", "val")}


# --- inputs ---------------------------------------------------------------------------
def centred_logit(p):
    L = np.log(np.clip(p, 1e-4, 1))
    return L - L.mean(1, keepdims=True)


def fit_temperature(L, y):
    best, bt = np.inf, 1.0
    for t in T_GRID:
        z = L / t
        z = z - z.max(1, keepdims=True)
        nll = -np.mean(z[np.arange(len(y)), y] - np.log(np.exp(z).sum(1)))
        if nll < best:
            best, bt = nll, t
    return bt


class Inputs:
    """Fitted on the training rows only; transforms any rows."""

    def __init__(self, cfg, pc, pm, has, y, extras):
        self.cfg = cfg
        Lc, Lm = centred_logit(pc), centred_logit(np.where(np.isnan(pm), 0.25, pm))
        self.Tc = fit_temperature(Lc, y) if cfg["temp"] else 1.0
        self.Tm = fit_temperature(Lm[has], y[has]) if cfg["temp"] and has.any() else 1.0
        ref = has if has.any() else np.ones(len(y), bool)          # no_mri: every row is neutral
        prior = np.bincount(y[ref], minlength=4) / ref.sum()
        self.neutral = centred_logit(prior[None])[0] / self.Tm
        X = self.raw(pc, pm, has, extras)
        k = X.shape[1]
        self.mu, self.sd = np.zeros(k), np.ones(k)
        if cfg["scale"]:
            for j in range(k):
                if j == 8:                                                    # flag stays 0/1
                    continue
                rows = has if 4 <= j < 8 else np.ones(len(X), bool)
                if not rows.any():
                    continue
                self.mu[j], self.sd[j] = X[rows, j].mean(), X[rows, j].std() + 1e-8

    def raw(self, pc, pm, has, extras):
        Lc = centred_logit(pc) / self.Tc
        Lm = np.where(has[:, None], centred_logit(np.where(np.isnan(pm), 0.25, pm)) / self.Tm, self.neutral)
        return np.column_stack([Lc, Lm, has.astype(float)] + ([extras] if extras is not None else []))

    def __call__(self, pc, pm, has, extras):
        return torch.tensor((self.raw(pc, pm, has, extras) - self.mu) / self.sd, dtype=torch.float32)

    def neutral_scaled(self):
        return torch.tensor((self.neutral - self.mu[4:8]) / self.sd[4:8], dtype=torch.float32)

    def base(self, pc):
        """K2 residual base: log softmax of the (temperature-scaled) clinical logits."""
        z = centred_logit(pc) / self.Tc
        return torch.tensor(z - np.log(np.exp(z).sum(1, keepdims=True)), dtype=torch.float32)


# --- model ----------------------------------------------------------------------------
def build(cfg, n_in, seed):
    from kan import MultKAN
    torch.manual_seed(seed)
    width = [n_in, 4, 5] if cfg["variant"] == "K3" else [n_in, 5]
    return MultKAN(width=width, grid=cfg["G"], k=3, seed=seed, auto_save=False, symbolic_enabled=False,
                   save_act=True, ckpt_path=os.path.join("/tmp", f"kanv3_{os.getpid()}"))


def cog_logits(cfg, out, base):
    return out[:, :4] + base if cfg["variant"] == "K2" else out[:, :4]


def loss_fn(cfg, out, base, y, w):
    z = cog_logits(cfg, out, base)
    loss = Fn.cross_entropy(z, y, weight=w, label_smoothing=cfg["ls"])
    if cfg["ord"] > 0:
        cp = torch.cumsum(torch.softmax(z, 1), 1)[:, :3]
        ct = torch.cumsum(Fn.one_hot(y, 4).float(), 1)[:, :3]
        loss = loss + cfg["ord"] * ((cp - ct) ** 2).sum(1).mean()
    return loss + LAM_ADD * Fn.binary_cross_entropy_with_logits(out[:, 4], (y == 3).float())


def fit_predict(cfg, tr, te, seed):
    """tr / te: dicts with pc, pm, has, y (tr), extras. Returns (p_cog, p_add, diag) for te."""
    y = tr["y"]
    inp = Inputs(cfg, tr["pc"], tr["pm"], tr["has"], y, tr["extras"])
    X = inp(tr["pc"], tr["pm"], tr["has"], tr["extras"])
    B = inp.base(tr["pc"])
    i_fit, i_ho = train_test_split(np.arange(len(y)), test_size=HOLDOUT, stratify=y, random_state=seed)
    yt = torch.tensor(y)
    w = None
    if cfg["cb"]:
        cnt = np.bincount(y[i_fit], minlength=4)
        w = torch.tensor(len(i_fit) / (4 * cnt), dtype=torch.float32)
    model = build(cfg, X.shape[1], seed)
    if cfg["scale"]:
        model.update_grid(X[i_fit])
    opt = torch.optim.Adam(model.get_params(), lr=LR)
    gen = torch.Generator().manual_seed(seed)
    neutral = inp.neutral_scaled()
    best, best_step, state, finite = np.inf, -1, None, True
    for step in range(MAX_STEPS):
        if cfg["scale"] and 0 < step <= 50 and step % 5 == 0:
            model.update_grid(X[i_fit])
        xf = X[i_fit].clone()
        drop = torch.rand(len(i_fit), generator=gen) < DROPOUT
        xf[drop, 4:8] = neutral
        xf[drop, 8] = 0.0
        opt.zero_grad()
        out = model(xf)
        loss = loss_fn(cfg, out, B[i_fit], yt[i_fit], w)
        if cfg["mu"] > 0:
            loss = loss + cfg["mu"] * model.get_reg("edge_forward_spline_n", 1.0, 2.0, 0.0, 0.0)
        if not torch.isfinite(loss):
            finite = False
            break
        loss.backward()
        opt.step()
        with torch.no_grad():
            h = float(loss_fn(cfg, model(X[i_ho]), B[i_ho], yt[i_ho], w))
        if not np.isfinite(h):
            finite = False
            break
        if h < best:
            best, best_step = h, step
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif step - best_step >= PATIENCE:
            break
    if state is not None:
        model.load_state_dict(state)
    with torch.no_grad():
        out = model(inp(te["pc"], te["pm"], te["has"], te["extras"]))
        p = torch.softmax(cog_logits(cfg, out, inp.base(te["pc"])), 1).numpy()
        a = torch.sigmoid(out[:, 4]).numpy()
    diag = {"best_step": best_step, "failed": bool((not finite) or best_step <= 0)}
    return p, a, diag, (model, inp)


# --- jobs -----------------------------------------------------------------------------
def unit_data(seed, unit):
    """Training and test stream scores for one outer unit (fold index or 'val')."""
    tr, va = DATA["train"], DATA["val"]
    si = M.SEEDS.index(seed)

    def pack(df, rows, pc, pm):
        sub = df.iloc[rows]
        return {"pc": pc, "pm": pm, "has": sub.has_mri.to_numpy(), "y": sub.label.to_numpy(), "rows": rows}

    if unit == "val":
        a = np.arange(len(tr))
        trd = pack(tr, a, ST["clin_oof"][si], ST["mri_oof"][si])
        ted = pack(va, np.arange(len(va)), ST["clin_val"][si], ST["mri_val"][si])
        return trd, ted, tr, va
    a, b = ST[f"idx_train_{seed}_{unit}"], ST[f"idx_test_{seed}_{unit}"]
    trd = pack(tr, a, ST[f"clin_inner_{seed}_{unit}"], ST[f"mri_inner_{seed}_{unit}"])
    ted = pack(tr, b, ST["clin_oof"][si][b], ST["mri_oof"][si][b])
    return trd, ted, tr, tr


def job(arg):
    cfg, seed, unit = arg
    trd, ted, trdf, tedf = unit_data(seed, unit)
    for d, df in ((trd, trdf), (ted, tedf)):
        if cfg["no_mri"]:                                                 # ablation: MRI stream masked everywhere
            d["has"] = np.zeros_like(d["has"])
        d["extras"] = df.iloc[d["rows"]][cfg["extras"]].to_numpy(dtype=float) if cfg["extras"] else None
    ps, as_, fails, steps = [], [], 0, []
    for ks in KSEEDS:
        p, a, dg, _ = fit_predict(cfg, trd, ted, ks)
        ps.append(p)
        as_.append(a)
        fails += dg["failed"]
        steps.append(dg["best_step"])
    return seed, unit, ted["rows"], np.mean(ps, 0), np.mean(as_, 0), fails, steps


def run_config(cfg, pool, stage="kan", notes=""):
    cfg = {**DEFAULT, **cfg}
    assert not (cfg["variant"] == "K3" and cfg["mu"] <= 0)
    n_tr, n_va = len(DATA["train"]), len(DATA["val"])
    P = {"cv": np.zeros((5, n_tr, 4)), "val": np.zeros((5, n_va, 4))}
    A = {"cv": np.zeros((5, n_tr)), "val": np.zeros((5, n_va))}
    fails, steps = 0, []
    t0 = time.time()
    for seed, unit, rows, p, a, f, st in pool.imap_unordered(
            job, [(cfg, s, u) for s in M.SEEDS for u in [0, 1, 2, 3, 4, "val"]]):
        k = "val" if unit == "val" else "cv"
        P[k][M.SEEDS.index(seed), rows] = p
        A[k][M.SEEDS.index(seed), rows] = a
        fails += f
        steps += st
    y = DATA["train"].label.to_numpy()
    s = M.cv_summary(y, P["cv"])
    inputs = "clin logits(4)+MRI logits(4)+has_mri" + ("+" + "+".join(cfg["extras"]) if cfg["extras"] else "")
    hp = {k: v for k, v in cfg.items() if k != "extras"}
    cid = M.log_config(stage, f"KAN fusion {cfg['variant']}", inputs, hp, s,
                       notes=(notes + f"; failed fits {fails}/150; best steps {min(steps)}-{max(steps)}").lstrip("; "))
    os.makedirs(os.path.join(HERE, "kan_runs"), exist_ok=True)
    np.savez(os.path.join(HERE, "kan_runs", f"{cid}.npz"), cv=P["cv"], val=P["val"], cv_add=A["cv"],
             val_add=A["val"], cfg=json.dumps(cfg))
    yv = DATA["val"].label.to_numpy()
    v = M.cv_summary(yv, P["val"])
    print(f"[{cid}] {cfg['variant']} G{cfg['G']} mu{cfg['mu']} scale{int(cfg['scale'])} temp{int(cfg['temp'])} "
          f"extras{len(cfg['extras'])} ls{cfg['ls']} cb{int(cfg['cb'])} ord{cfg['ord']}: "
          f"CV macro-F1 {s['macro_f1']:.4f} ± {s['macro_f1_sd']:.4f}  LMCI {s['f1_LMCI']:.3f} AD {s['f1_AD']:.3f}  "
          f"logloss {s['logloss']:.3f} | val {v['macro_f1']:.4f}  fails {fails}  {time.time() - t0:.0f}s", flush=True)
    return cid, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True)
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--stage", default="kan")
    a = ap.parse_args()
    cfgs = json.load(open(a.configs))
    init_worker()
    with Pool(a.workers, initializer=init_worker) as pool:
        for c in cfgs:
            notes = c.pop("_notes", "")
            run_config(c, pool, a.stage, notes)


if __name__ == "__main__":
    main()
