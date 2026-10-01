"""Metrics and config logging shared by every fusion-v3 script (numpy/sklearn only).

The formulas are those of analysis/fusion_feasibility/ablation.py (confusion,
per-class F1, macro-F1 bootstrap); evaluate.py asserts they agree with it.
"""
import csv
import json
import os
import time

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ["CN", "EMCI", "LMCI", "AD"]
SEEDS = (42, 43, 44, 45, 46)
LOG = os.path.join(HERE, "config_log.csv")
LOG_FIELDS = ["config_id", "stage", "system", "inputs", "hyperparameters", "n_eval", "cv_macro_f1",
              "cv_macro_f1_sd", "cv_acc", "cv_bal_acc", "cv_f1_LMCI", "cv_f1_AD", "cv_ad_auc", "cv_logloss",
              "decision", "notes", "time"]


def confusion(y, p, k=4):
    return np.bincount(k * y + p, minlength=k * k).reshape(k, k)


def f1s(cm):
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros(len(tp)), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros(len(tp)), where=cm.sum(1) > 0)
    return np.divide(2 * prec * rec, prec + rec, out=np.zeros(len(tp)), where=(prec + rec) > 0), rec


def metrics(y, proba):
    p = proba.argmax(1)
    f1, rec = f1s(confusion(y, p))
    out = {"macro_f1": f1.mean(), "bal_acc": rec.mean(), "acc": (p == y).mean(),
           "ad_auc": roc_auc_score(y == 3, proba[:, 3]),
           "logloss": float(-np.mean(np.log(np.clip(proba[np.arange(len(y)), y], 1e-12, 1)))),
           "stage_dist": float(np.abs(p - y).mean())}
    out.update({f"f1_{n}": f1[i] for i, n in enumerate(NAMES)})
    return out


def metrics3(y, proba):
    """3-class CN / MCI (EMCI + LMCI) / AD."""
    m = np.array([0, 1, 1, 2])
    y3 = m[y]
    p3 = np.stack([proba[:, 0], proba[:, 1] + proba[:, 2], proba[:, 3]], 1)
    p = p3.argmax(1)
    f1, rec = f1s(confusion(y3, p, 3))
    return {"macro_f1_3": f1.mean(), "acc_3": (p == y3).mean(), "bal_acc_3": rec.mean(),
            **{f"f1_3_{n}": f1[i] for i, n in enumerate(["CN", "MCI", "AD"])}}


def macro_f1_boot(y, preds, idx):
    """preds: (n_seeds, n) class predictions; idx: (B, n). Returns (B, n_seeds)."""
    out = np.empty((len(idx), len(preds)))
    for s, p in enumerate(preds):
        codes = 4 * y[idx] + p[idx]
        cms = np.stack([np.bincount(c, minlength=16) for c in codes]).reshape(-1, 4, 4)
        tp = np.diagonal(cms, axis1=1, axis2=2).astype(float)
        ps, rs = cms.sum(1), cms.sum(2)
        prec = np.divide(tp, ps, out=np.zeros_like(tp), where=ps > 0)
        rec = np.divide(tp, rs, out=np.zeros_like(tp), where=rs > 0)
        f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
        out[:, s] = f1.mean(1)
    return out


def cv_summary(y, P, mask=None):
    """P: (n_seeds, n, 4). Mean (and SD of macro-F1) over seeds."""
    if mask is not None:
        y, P = y[mask], P[:, mask]
    ms = [metrics(y, p) for p in P]
    agg = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
    agg["macro_f1_sd"] = float(np.std([m["macro_f1"] for m in ms], ddof=1))
    agg["n"] = int(len(y))
    return agg


def next_id():
    if not os.path.exists(LOG):
        return 1
    with open(LOG) as f:
        return sum(1 for _ in f)              # header counts as row 0


def log_config(stage, system, inputs, hp, s, decision="", notes=""):
    new = not os.path.exists(LOG)
    cid = next_id()
    with open(LOG, "a", newline="") as f:
        w = csv.DictWriter(f, LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow({"config_id": cid, "stage": stage, "system": system, "inputs": inputs,
                    "hyperparameters": json.dumps(hp, sort_keys=True), "n_eval": s["n"],
                    "cv_macro_f1": round(s["macro_f1"], 4), "cv_macro_f1_sd": round(s["macro_f1_sd"], 4),
                    "cv_acc": round(s["acc"], 4), "cv_bal_acc": round(s["bal_acc"], 4),
                    "cv_f1_LMCI": round(s["f1_LMCI"], 4), "cv_f1_AD": round(s["f1_AD"], 4),
                    "cv_ad_auc": round(s["ad_auc"], 4), "cv_logloss": round(s["logloss"], 4),
                    "decision": decision, "notes": notes, "time": time.strftime("%Y-%m-%d %H:%M")})
    return cid
