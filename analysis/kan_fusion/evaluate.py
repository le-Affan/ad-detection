"""Evaluate CatBoost and KAN on C / F (and CatBoost R) as pre-registered. Never reads test.

Reads catboost_preds.npz, kan_preds.npz, kan_runs.json, topk.json and
fusion_trainval.csv. Applies preregistration.md items d (metrics), e (decision
rules e1, e2, e3), f (severity) and g (discordance), using the harness's metric
and bootstrap functions (analysis/fusion_feasibility/ablation.py, unchanged).

Outputs: eval_runs.csv (per seed x model x eval x subset), eval_tables.md,
eval_summary.json, discordance_val.csv. Run with LYP/.venv-fusion.
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
sys.path.insert(0, os.path.join(ROOT, "analysis", "fusion_feasibility"))
import ablation as H  # noqa: E402

NAMES = H.NAMES
MODELS = ["CatBoost C", "CatBoost F", "CatBoost R", "KAN C", "KAN F"]
P10, P90 = 0.2981, 0.8751                                          # item g, fixed in the pre-registration
NI_MARGIN = -0.02
MAX_FAIL = 0.20


def load():
    d = pd.read_csv(os.path.join(HERE, "fusion_trainval.csv"))
    tr, va = d[d.split == "train"].reset_index(drop=True), d[d.split == "val"].reset_index(drop=True)
    cb, kn = np.load(os.path.join(HERE, "catboost_preds.npz")), np.load(os.path.join(HERE, "kan_preds.npz"))
    assert (cb["train_rids"] == tr.RID.to_numpy()).all() and (cb["val_rids"] == va.RID.to_numpy()).all()
    P = {"cv": {}, "val": {}}
    for s in ("C", "F", "R"):
        P["cv"][f"CatBoost {s}"], P["val"][f"CatBoost {s}"] = cb[f"oof_{s}"], cb[f"val_{s}"]
    for s in ("C", "F"):
        P["cv"][f"KAN {s}"], P["val"][f"KAN {s}"] = kn[f"oof_{s}"], kn[f"val_{s}"]
    add = {"cv": {s: kn[f"oof_add_{s}"] for s in ("C", "F")}, "val": {s: kn[f"val_add_{s}"] for s in ("C", "F")}}
    return tr, va, P, add


def severity(p):
    return p @ np.arange(4)


def seed_metrics(y, p, cdrsb):
    m = H.metrics(y, p)
    s = severity(p)
    m["stage_dist"] = float(np.abs(p.argmax(1) - y).mean())
    m["sev_rho_stage"] = float(spearmanr(s, y)[0])
    m["sev_rho_cdrsb"] = float(spearmanr(s, cdrsb)[0])
    return m


def boot_diff(y, Pa, Pb, idx):
    """Harness bootstrap: mean over seeds of macro-F1(a) - macro-F1(b) per resample."""
    pa, pb = Pa.argmax(-1), Pb.argmax(-1)
    d = (H.macro_f1_boot(y, pa, idx) - H.macro_f1_boot(y, pb, idx)).mean(1)
    point = float(np.mean([H.f1s_from_cm(H.confusion(y, pa[s]))[0].mean()
                           - H.f1s_from_cm(H.confusion(y, pb[s]))[0].mean() for s in range(len(pa))]))
    per_seed = [H.f1s_from_cm(H.confusion(y, pa[s]))[0].mean() - H.f1s_from_cm(H.confusion(y, pb[s]))[0].mean()
                for s in range(len(pa))]
    return {"diff": point, "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)),
            "per_seed": [round(float(x), 4) for x in per_seed]}


def e1_verdict(cv, val):
    if cv["lo"] > 0 and val["diff"] > 0:
        return "reliable gain"
    if cv["hi"] < 0 and val["diff"] < 0:
        return "reliable loss"
    return "no reliable difference"


def main():
    tr, va, P, add = load()
    y = {"cv": tr.label.to_numpy(), "val": va.label.to_numpy()}
    mri = {"cv": tr.has_mri.to_numpy(), "val": va.has_mri.to_numpy()}
    cdr = {"cv": tr.CDRSB.to_numpy(), "val": va.CDRSB.to_numpy()}
    alz = {"cv": tr.ALZ.to_numpy(), "val": va.ALZ.to_numpy()}
    n_s = len(H.SEEDS)

    # --- d. metrics per seed ---------------------------------------------------
    rows = []
    for ev in ("cv", "val"):
        for sub, mk in (("all", np.ones_like(mri[ev])), ("mri", mri[ev])):
            for m in MODELS:
                for si, seed in enumerate(H.SEEDS):
                    r = seed_metrics(y[ev][mk], P[ev][m][si][mk], cdr[ev][mk])
                    if m.startswith("KAN"):
                        r["add_head_auc"] = float(roc_auc_score(y[ev][mk] == 3, add[ev][m[-1]][si][mk]))
                    rows.append({"model": m, "eval": ev, "subset": sub, "seed": seed, "n": int(mk.sum()), **r})
    runs = pd.DataFrame(rows)
    runs.to_csv(os.path.join(HERE, "eval_runs.csv"), index=False)

    # --- e. bootstrap comparisons ------------------------------------------------
    rng = np.random.default_rng(0)
    idx = {ev: rng.integers(0, len(y[ev]), size=(H.N_BOOT, len(y[ev]))) for ev in ("cv", "val")}
    comps = [("MRI contribution, CatBoost (F − C)", "CatBoost F", "CatBoost C"),
             ("MRI contribution, KAN (F − C)", "KAN F", "KAN C"),
             ("KAN − CatBoost on F (e2)", "KAN F", "CatBoost F"),
             ("KAN − CatBoost on C", "KAN C", "CatBoost C"),
             ("CatBoost C − R", "CatBoost C", "CatBoost R"),
             ("CatBoost F − R", "CatBoost F", "CatBoost R")]
    boots = {name: {ev: boot_diff(y[ev], P[ev][a], P[ev][b], idx[ev]) for ev in ("cv", "val")}
             for name, a, b in comps}
    # same pairs restricted to subjects with a scan (descriptive), using the CV resample of the subset
    rng_m = np.random.default_rng(0)
    boots_mri = {}
    for name, a, b in comps[:3]:
        boots_mri[name] = {}
        for ev in ("cv", "val"):
            mk = mri[ev]
            im = rng_m.integers(0, mk.sum(), size=(H.N_BOOT, mk.sum()))
            boots_mri[name][ev] = boot_diff(y[ev][mk], P[ev][a][:, mk], P[ev][b][:, mk], im)

    kan_runs = json.load(open(os.path.join(HERE, "kan_runs.json")))
    fails = {s: sum(r["refit"]["failed"] for r in kan_runs if r["set"] == s and r["unit"] != "val") for s in ("C", "F")}
    inner_fails = sum(r["inner_failures"] for r in kan_runs)
    stable = all(f / 25 <= MAX_FAIL for f in fails.values())
    e2 = boots["KAN − CatBoost on F (e2)"]["cv"]
    kan_ni = e2["lo"] > NI_MARGIN
    final = "KAN F" if (kan_ni and stable) else "CatBoost F"
    verdicts = {
        "e1 CatBoost": e1_verdict(boots["MRI contribution, CatBoost (F − C)"]["cv"],
                                  boots["MRI contribution, CatBoost (F − C)"]["val"]),
        "e1 KAN": e1_verdict(boots["MRI contribution, KAN (F − C)"]["cv"], boots["MRI contribution, KAN (F − C)"]["val"]),
        "e2 KAN non-inferior (CI lower bound > -0.02)": bool(kan_ni),
        "KAN stable (<= 20% failed outer fits)": bool(stable),
        "KAN refit failures (of 25 outer CV fits)": fails,
        "KAN inner-fit failures (of 1440)": int(inner_fails),
        "final decision layer": final,
    }

    # --- f. severity -------------------------------------------------------------
    sev = {}
    for m in MODELS:
        out = {}
        for ev in ("cv", "val"):
            pm = P[ev][m].mean(0)
            s = severity(pm)
            means = [float(s[y[ev] == k].mean()) for k in range(4)]
            per_seed_mono = sum(bool(np.all(np.diff([severity(P[ev][m][si])[y[ev] == k].mean() for k in range(4)]) > 0))
                                for si in range(n_s))
            out[ev] = {"class_means": means, "monotonic": bool(np.all(np.diff(means) > 0)),
                       "seeds_monotonic": per_seed_mono,
                       "rho_stage": float(spearmanr(s, y[ev])[0]), "rho_cdrsb": float(spearmanr(s, cdr[ev])[0]),
                       "class_sd": [float(s[y[ev] == k].std()) for k in range(4)]}
        sev[m] = out

    # --- g. discordance -----------------------------------------------------------
    disc = {}
    for ev in ("val", "cv"):
        pc = P[ev]["CatBoost C"].mean(0).argmax(1)
        pf = P[ev][final].mean(0).argmax(1)
        a = alz[ev]
        evaluable = ~np.isnan(a)
        b1 = evaluable & (pc == 3) & (a < P10)
        b2 = evaluable & np.isin(pc, (0, 1)) & (a > P90)
        flag = b1 | b2
        df = pd.DataFrame({"true": np.array(NAMES)[y[ev]], "pred_C": np.array(NAMES)[pc],
                           "pred_final": np.array(NAMES)[pf], "ALZ": a, "evaluable": evaluable,
                           "branch1": b1, "branch2": b2, "flag": flag,
                           "err_C": pc != y[ev], "err_final": pf != y[ev]})
        if ev == "val":
            df.insert(0, "PTID", va.PTID)
            df.to_csv(os.path.join(HERE, "discordance_val.csv"), index=False)
        e = df[df.evaluable]
        disc[ev] = {
            "n": len(df), "not_evaluable": int((~evaluable).sum()), "flagged": int(flag.sum()),
            "branch1": int(b1.sum()), "branch2": int(b2.sum()),
            "rate_by_true": {k: f"{int(g.flag.sum())}/{len(g)}" for k, g in e.groupby("true")},
            "rate_by_pred_C": {k: f"{int(g.flag.sum())}/{len(g)}" for k, g in e.groupby("pred_C")},
            "err_C_flagged": f"{int(e[e.flag].err_C.sum())}/{int(e.flag.sum())}",
            "err_C_unflagged": f"{int(e[~e.flag].err_C.sum())}/{int((~e.flag).sum())}",
            "err_final_flagged": f"{int(e[e.flag].err_final.sum())}/{int(e.flag.sum())}",
            "err_final_unflagged": f"{int(e[~e.flag].err_final.sum())}/{int((~e.flag).sum())}",
            "flagged_subjects": (df[df.flag][["PTID", "true", "pred_C", "pred_final", "ALZ"]].round(3)
                                 .to_dict("records") if ev == "val" else None),
        }

    alz_auc = {ev: float(roc_auc_score((y[ev] == 3)[~np.isnan(alz[ev])], alz[ev][~np.isnan(alz[ev])]))
               for ev in ("cv", "val")}
    cms = {ev: {m: np.mean([H.confusion(y[ev], P[ev][m][si].argmax(1)) for si in range(n_s)], 0).round(1).tolist()
                for m in MODELS} for ev in ("cv", "val")}
    summary = {"verdicts": verdicts, "boots": boots, "boots_mri_subset": boots_mri, "severity": sev,
               "discordance": disc, "alz_alone_ad_auc": alz_auc, "confusion": cms}
    json.dump(summary, open(os.path.join(HERE, "eval_summary.json"), "w"), indent=1)
    write_tables(runs, summary)
    print(open(os.path.join(HERE, "eval_tables.md")).read())


def ms(x):
    return f"{x.mean():.3f} ± {x.std(ddof=1):.3f}"


def write_tables(runs, S):
    L = ["# KAN fusion — evaluation tables\n", "Mean ± SD over seeds 42–46. CV = 569 clinical-train (pooled OOF per seed); "
         "val = train 569 → 122 clinical-val.\n"]
    cols = ["macro_f1", "acc", "bal_acc", "f1_CN", "f1_EMCI", "f1_LMCI", "f1_AD", "ad_auc", "stage_dist"]
    for sub, title in (("all", "all subjects"), ("mri", "subjects with a scan (567 / 121)")):
        for ev in ("cv", "val"):
            L.append(f"## {ev.upper()}, {title}\n")
            L.append("| Model | " + " | ".join(cols) + " | ADD-head AUC |")
            L.append("|---" * (len(cols) + 2) + "|")
            for m in MODELS:
                g = runs[(runs.model == m) & (runs["eval"] == ev) & (runs.subset == sub)]
                add = ms(g.add_head_auc) if m.startswith("KAN") else "—"
                L.append(f"| {m} | " + " | ".join(ms(g[c]) for c in cols) + f" | {add} |")
            L.append("")
    L.append("## Paired macro-F1 differences (2,000 resamples, mean over seeds)\n")
    L.append("| Comparison | CV diff [95% CI] | val diff [95% CI, descriptive] | CV per seed |")
    L.append("|---|---|---|---|")
    for name, b in S["boots"].items():
        c, v = b["cv"], b["val"]
        L.append(f"| {name} | {c['diff']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}] | "
                 f"{v['diff']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}] | {' '.join(f'{x:+.3f}' for x in c['per_seed'])} |")
    L.append("\nSubjects with a scan only:\n")
    L.append("| Comparison | CV diff [95% CI] | val diff [95% CI] |")
    L.append("|---|---|---|")
    for name, b in S["boots_mri_subset"].items():
        c, v = b["cv"], b["val"]
        L.append(f"| {name} | {c['diff']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}] | {v['diff']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}] |")
    L.append("\n## Verdicts\n")
    for k, v in S["verdicts"].items():
        L.append(f"- {k}: **{v}**")
    L.append(f"\nALZ alone, AD vs rest AUC: CV (cross-fitted) {S['alz_alone_ad_auc']['cv']:.3f}, "
             f"val {S['alz_alone_ad_auc']['val']:.3f}\n")
    L.append("## Severity s = Σ k·p(k) (seed-averaged probabilities)\n")
    L.append("| Model | Eval | mean s CN / EMCI / LMCI / AD | monotonic | seeds monotonic | ρ(s, stage) | ρ(s, CDRSB) |")
    L.append("|---|---|---|---|---|---|---|")
    for m, o in S["severity"].items():
        for ev in ("val", "cv"):
            x = o[ev]
            L.append(f"| {m} | {ev} | {' / '.join(f'{v:.2f}' for v in x['class_means'])} | {x['monotonic']} | "
                     f"{x['seeds_monotonic']}/5 | {x['rho_stage']:.3f} | {x['rho_cdrsb']:.3f} |")
    L.append("\n## Discordance flag (CatBoost C prediction; P10 0.2981, P90 0.8751)\n")
    for ev, d in S["discordance"].items():
        L.append(f"**{ev}**: {d['flagged']} flagged of {d['n'] - d['not_evaluable']} evaluable "
                 f"({d['not_evaluable']} without ALZ); branch 1 (pred AD, low ALZ) {d['branch1']}, "
                 f"branch 2 (pred CN/EMCI, high ALZ) {d['branch2']}.")
        L.append(f"- by true class: {d['rate_by_true']}; by C prediction: {d['rate_by_pred_C']}")
        L.append(f"- errors, CatBoost C: flagged {d['err_C_flagged']}, unflagged {d['err_C_unflagged']}; "
                 f"final layer on F: flagged {d['err_final_flagged']}, unflagged {d['err_final_unflagged']}")
        if d["flagged_subjects"]:
            L.append(f"- flagged: {d['flagged_subjects']}")
        L.append("")
    L.append("## Confusion matrices (mean over seeds; rows true CN/EMCI/LMCI/AD, columns predicted)\n")
    for ev in ("val", "cv"):
        for m, cm in S["confusion"][ev].items():
            L.append(f"- {ev} {m}: " + " | ".join(" ".join(f"{v:5.1f}" for v in r) for r in cm))
        L.append("")
    with open(os.path.join(HERE, "eval_tables.md"), "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
