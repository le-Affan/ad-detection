"""Evaluate fusion-v3 systems on train CV and val (run with LYP/.venv-fusion). Never reads test.

Systems: CatBoost R and F (kan-fusion predictions, same folds and seeds), the
clinical specialist, the MRI specialist, KAN K1 / K2 / K3 (the best config of
each variant from config_log.csv), and the no-MRI control of the final KAN.
Paired bootstraps use the harness's macro-F1 bootstrap (2,000 resamples,
rng seed 0, the same indices for every comparison).
MRI roles: (a) the MRI specialist standalone vs the majority-class baseline;
(b) MRI-only mode (MRI specialist, which includes age and sex) vs the clinical
specialist on the same subjects; (c) fusion vs the clinical specialist.
Outputs: eval_tables.md, eval_summary.json, eval_runs.csv.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "analysis", "fusion_feasibility"))
import ablation as H  # noqa: E402
import metrics as M  # noqa: E402

BEST = {"KAN K1": 26, "KAN K2 (final)": 32, "KAN K3": 33, "KAN K2, MRI masked (control)": 34}
FINAL = "KAN K2 (final)"


def load():
    d = pd.read_csv(os.path.join(ROOT, "analysis", "kan_fusion", "fusion_trainval.csv"))
    tr, va = d[d.split == "train"].reset_index(drop=True), d[d.split == "val"].reset_index(drop=True)
    st = np.load(os.path.join(HERE, "streams.npz"))
    cb = np.load(os.path.join(ROOT, "analysis", "kan_fusion", "catboost_preds.npz"))
    assert (st["train_rids"] == tr.RID.to_numpy()).all() and (cb["train_rids"] == tr.RID.to_numpy()).all()
    P = {"cv": {"CatBoost R": cb["oof_R"], "CatBoost F (previous final)": cb["oof_F"],
                "Clinical specialist": st["clin_oof"], "MRI specialist": st["mri_oof"]},
         "val": {"CatBoost R": cb["val_R"], "CatBoost F (previous final)": cb["val_F"],
                 "Clinical specialist": st["clin_val"], "MRI specialist": st["mri_val"]}}
    add = {}
    for name, cid in BEST.items():
        z = np.load(os.path.join(HERE, "kan_runs", f"{cid}.npz"))
        P["cv"][name], P["val"][name] = z["cv"], z["val"]
        add[name] = {"cv": z["cv_add"], "val": z["val_add"]}
    return tr, va, P, add


def ms(v):
    return f"{np.mean(v):.3f} ± {np.std(v, ddof=1):.3f}"


def main():
    tr, va, P, add = load()
    y = {"cv": tr.label.to_numpy(), "val": va.label.to_numpy()}
    has = {"cv": tr.has_mri.to_numpy(), "val": va.has_mri.to_numpy()}
    cdr = {"cv": tr.CDRSB.to_numpy(), "val": va.CDRSB.to_numpy()}
    systems = list(P["cv"])

    # metric module agrees with the harness
    p0 = P["cv"]["Clinical specialist"][0]
    a, b = M.metrics(y["cv"], p0), H.metrics(y["cv"], p0)
    assert all(abs(a[k] - b[k]) < 1e-12 for k in b)

    rows = []
    for ev in ("cv", "val"):
        for s in systems:
            mk = has[ev] if s == "MRI specialist" else np.ones(len(y[ev]), bool)
            for si, seed in enumerate(M.SEEDS):
                r = {**M.metrics(y[ev][mk], P[ev][s][si][mk]), **M.metrics3(y[ev][mk], P[ev][s][si][mk])}
                if s in add:
                    r["add_head_auc"] = roc_auc_score(y[ev] == 3, add[s][ev][si])
                rows.append({"system": s, "eval": ev, "seed": seed, "n": int(mk.sum()), **r})
    runs = pd.DataFrame(rows)
    runs.to_csv(os.path.join(HERE, "eval_runs.csv"), index=False)

    # bootstraps (all subjects; MRI specialist excluded since it has no output without a scan)
    rng = np.random.default_rng(0)
    idx = {ev: rng.integers(0, len(y[ev]), size=(H.N_BOOT, len(y[ev]))) for ev in ("cv", "val")}

    def boot(a, b, ev):
        pa, pb = P[ev][a].argmax(-1), P[ev][b].argmax(-1)
        d = (M.macro_f1_boot(y[ev], pa, idx[ev]) - M.macro_f1_boot(y[ev], pb, idx[ev])).mean(1)
        point = np.mean([M.f1s(M.confusion(y[ev], pa[s]))[0].mean() - M.f1s(M.confusion(y[ev], pb[s]))[0].mean()
                         for s in range(len(pa))])
        return {"diff": float(point), "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5))}

    comps = [(k, ref) for k in BEST if "control" not in k for ref in ("Clinical specialist", "CatBoost F (previous final)")]
    comps += [(FINAL, "KAN K2, MRI masked (control)"), ("KAN K2, MRI masked (control)", "Clinical specialist"),
              (FINAL, "CatBoost R")]
    boots = {f"{a} − {b}": {ev: boot(a, b, ev) for ev in ("cv", "val")} for a, b in comps}

    # confusion matrices (mean over seeds)
    cms = {ev: {s: np.mean([M.confusion(y[ev][has[ev] if s == "MRI specialist" else slice(None)],
                                        P[ev][s][si][has[ev] if s == "MRI specialist" else slice(None)].argmax(1))
                            for si in range(5)], 0) for s in systems} for ev in ("cv", "val")}

    # MRI roles
    roles = {}
    for ev in ("cv", "val"):
        mk = has[ev]
        yy = y[ev][mk]
        maj = np.bincount(y["cv"][has["cv"]]).argmax()                       # majority class of train (EMCI)
        pm = np.zeros((len(yy), 4))
        pm[:, maj] = 1
        base = {**M.metrics(yy, pm + 1e-9), **M.metrics3(yy, pm + 1e-9)}
        base["ad_auc"] = 0.5
        mri = runs[(runs.system == "MRI specialist") & (runs["eval"] == ev)]
        clin = [{**M.metrics(yy, P[ev]["Clinical specialist"][si][mk]),
                 **M.metrics3(yy, P[ev]["Clinical specialist"][si][mk])} for si in range(5)]
        roles[ev] = {"n": int(mk.sum()), "majority": base,
                     "mri": {k: float(mri[k].mean()) for k in ("macro_f1", "acc", "bal_acc", "f1_CN", "f1_EMCI",
                                                               "f1_LMCI", "f1_AD", "ad_auc", "macro_f1_3", "acc_3")},
                     "mri_sd": {k: float(mri[k].std(ddof=1)) for k in ("macro_f1", "ad_auc", "acc")},
                     "clinical": {k: float(np.mean([c[k] for c in clin])) for k in ("macro_f1", "acc", "ad_auc",
                                                                                     "macro_f1_3", "acc_3")}}
        cf, cc = cms[ev][FINAL], cms[ev]["Clinical specialist"]
        roles[ev]["fusion_vs_clinical"] = {
            "macro_f1_diff": boots[f"{FINAL} − Clinical specialist"][ev],
            "LMCI_to_AD": [float(cc[2, 3]), float(cf[2, 3])], "AD_to_LMCI": [float(cc[3, 2]), float(cf[3, 2])],
            "LMCI_to_EMCI": [float(cc[2, 1]), float(cf[2, 1])], "LMCI_correct": [float(cc[2, 2]), float(cf[2, 2])]}

    # severity
    sev = {}
    for s in systems:
        if s == "MRI specialist":
            continue
        sev[s] = {}
        for ev in ("cv", "val"):
            sv = P[ev][s].mean(0) @ np.arange(4)
            means = [float(sv[y[ev] == k].mean()) for k in range(4)]
            per = sum(bool(np.all(np.diff([(P[ev][s][si] @ np.arange(4))[y[ev] == k].mean() for k in range(4)]) > 0))
                      for si in range(5))
            sev[s][ev] = {"means": means, "monotonic": bool(np.all(np.diff(means) > 0)), "seeds": per,
                          "rho_stage": float(spearmanr(sv, y[ev])[0]), "rho_cdrsb": float(spearmanr(sv, cdr[ev])[0])}

    S = {"boots": boots, "roles": roles, "severity": sev,
         "confusion": {ev: {s: c.round(1).tolist() for s, c in cms[ev].items()} for ev in cms}}
    json.dump(S, open(os.path.join(HERE, "eval_summary.json"), "w"), indent=1)

    L = ["# fusion-v3 — CV and val tables\n",
         "Mean ± SD over seeds 42–46. CV: 569 clinical-train, pooled OOF per seed (MRI specialist: the 567 "
         "with a scan). Val: 122 (MRI specialist: 121). KAN predictions are 5-KAN-seed ensembles.\n"]
    cols = ["macro_f1", "acc", "bal_acc", "f1_LMCI", "f1_AD", "ad_auc", "macro_f1_3", "logloss"]
    for ev in ("cv", "val"):
        L += [f"## {ev.upper()}\n", "| System | n | " + " | ".join(cols) + " | ADD-head AUC |", "|---" * (len(cols) + 3) + "|"]
        for s in systems:
            g = runs[(runs.system == s) & (runs["eval"] == ev)]
            L.append(f"| {s} | {g.n.iloc[0]} | " + " | ".join(ms(g[c]) for c in cols) +
                     f" | {ms(g.add_head_auc) if s in add else '—'} |")
        L.append("")
    L += ["## Paired macro-F1 differences (2,000 resamples; mean over seeds)\n",
          "| Comparison | CV diff [95% CI] | val diff [95% CI] |", "|---|---|---|"]
    for k, b in boots.items():
        L.append(f"| {k} | {b['cv']['diff']:+.4f} [{b['cv']['lo']:+.4f}, {b['cv']['hi']:+.4f}] | "
                 f"{b['val']['diff']:+.4f} [{b['val']['lo']:+.4f}, {b['val']['hi']:+.4f}] |")
    L.append("\n## Severity (seed-averaged probabilities)\n")
    L += ["| System | eval | mean s CN/EMCI/LMCI/AD | monotonic | seeds | ρ stage | ρ CDRSB |", "|---|---|---|---|---|---|---|"]
    for s, o in sev.items():
        for ev in ("val", "cv"):
            x = o[ev]
            L.append(f"| {s} | {ev} | {' / '.join(f'{v:.2f}' for v in x['means'])} | {x['monotonic']} | "
                     f"{x['seeds']}/5 | {x['rho_stage']:.3f} | {x['rho_cdrsb']:.3f} |")
    L.append("\n## MRI roles\n")
    L.append("```\n" + json.dumps(roles, indent=1, default=float) + "\n```\n")
    L.append("## Confusion matrices (mean over seeds; rows true CN/EMCI/LMCI/AD)\n")
    for ev in ("val", "cv"):
        for s, c in cms[ev].items():
            L.append(f"- {ev} {s}: " + " | ".join(" ".join(f"{v:5.1f}" for v in r) for r in c))
        L.append("")
    open(os.path.join(HERE, "eval_tables.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
