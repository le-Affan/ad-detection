"""P3 (LYP/.venv-kan, CPU): exact Shapley values of the 25 frozen KANs, checks, sign consistency, held-out
replication with the regenerated train-only KANs, the MRI-masked predictions for P4-B, and the KAN P5 row.

Inputs at prediction time: demo/card_data/streams_all.npz (the frozen specialists' scores, written by
src/card/build_card_data.py); only its train and val rows are used here. Background = the 569 train rows.
P3-R0  efficiency |sum phi + E_bg f - f(x)| < 1e-4 (every subject, model, target); brute-force 16-coalition
       group values on 60 train subjects (10 dry-run + 50 by default_rng(0)) agree within 1e-4.
P3-R1  cells (group, target, true-class stratum on train): mean phi per model; consistent iff the same sign in
       all 25 models and |median| >= 0.05.
P3-R3  regenerate the 25 train-only KANs in memory (kan_fusion.unit_data(seed, "val") + fit_predict, as
       kan_interpret.py); gate: per outer seed, mean val probabilities and ADD match kan_runs/32.npz within 1e-4.
       If passed: val-stratum cell signs over the 25 train-only KANs (background = their own 569 training rows):
       replicated (25/25 agree), partially (>= 13), not replicated.
P4-B   ensemble probabilities with the scan and with the MRI inputs masked (has_mri = 0, neutral logits):
       frozen KANs on train and val (in-sample), train-only KANs on val (held out, if the gate passes).
P5     stability (25 models), plausibility (CDRSB, MMSE per-input phi on the AD final target), faithfulness
       (ensemble P(final predicted class), deletion / insertion of the 13 inputs vs 50 random orders).
Outputs: results/p3_efficiency.json, results/p3_sign_consistency.csv, results/p3_global.csv,
results/p3_replication.json, card_rules.json["kan"], p5_rows.json["kan"]; subject-level arrays (gitignored).
Nothing is written to analysis/fusion_v3/.
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import kan_shap_lib as K  # noqa: E402
import tab_faith as TF  # noqa: E402

sys.path.insert(0, C.V3)
import final_features as FF  # noqa: E402
import kan_fusion as KF  # noqa: E402

EFF_TOL, MIN_EFFECT, GATE_TOL = 1e-4, 0.05, 1e-4


# --- data -------------------------------------------------------------------------------------
def load_rows():
    z = np.load(os.path.join(C.ROOT, "demo", "card_data", "streams_all.npz"), allow_pickle=True)
    sp, lab = C.split_of(), C.labels()
    rid = z["rids"].astype(int)
    keep = np.array([sp[r] in ("train", "val") for r in rid])
    rid = rid[keep]
    C.assert_no_test(rid)
    d = {"rid": rid, "split": np.array([sp[r] for r in rid]), "y": np.array([lab[r] for r in rid]),
         "has": z["has"][keep].astype(bool), "ex": z["extras"][keep].astype(float)}
    for s in (42, 43, 44, 45, 46):
        d[f"pc_{s}"], d[f"pm_{s}"] = z[f"clin_{s}"][keep], z[f"mri_{s}"][keep]
    raw = FF.raw_rows(rid).set_index("RID").loc[rid]
    d["raw_CDRSB"], d["raw_MMSE"] = raw.CDRSB.to_numpy(float), raw.MMSE.to_numpy(float)
    return d


def sub(d, idx):
    return {k: (v[idx] if isinstance(v, np.ndarray) and len(v) == len(d["rid"]) else v) for k, v in d.items()}


# --- train-only KANs (P3-R3) -------------------------------------------------------------------
def regen(arg):
    config, seed, ks = arg
    trd, ted, trdf, tedf = KF.unit_data(seed, "val")
    for dd, df in ((trd, trdf), (ted, tedf)):
        dd["extras"] = df.iloc[dd["rows"]][config["extras"]].to_numpy(dtype=float)
    p, a, diag, (model, inp) = KF.fit_predict(config, trd, ted, ks)
    return seed, ks, p, a, {k: v.clone() for k, v in model.state_dict().items()}, inp, trd, ted, \
        tedf.RID.to_numpy()[ted["rows"]]


def cells_table(phi_models, strata, n_models):
    """phi_models: (n_models, n, 4 groups, 9); strata: true class (n,). -> DataFrame of cells."""
    rows = []
    for g, gn in enumerate(K.GROUP_NAMES):
        for t, tn in enumerate(K.TARGETS):
            for c, cn in enumerate(C.NAMES):
                e = phi_models[:, strata == c, g, t].mean(1)                     # (n_models,)
                npos, nneg = int((e > 0).sum()), int((e < 0).sum())
                med = float(np.median(e))
                rows.append({"group": gn, "target": tn, "stratum": cn, "n_models": n_models,
                             "n_positive": npos, "n_negative": nneg, "median_effect": med,
                             "min_effect": float(e.min()), "max_effect": float(e.max()),
                             "consistent": bool((npos == n_models or nneg == n_models) and abs(med) >= MIN_EFFECT)})
    return pd.DataFrame(rows)


def faith_kan(Ux, Ub, target, order_attr, rid):
    """Ux (25, 13, 4), Ub (25, 50, 13, 4): ensemble P(target) curves for deletion and insertion."""
    orders = np.vstack([order_attr[None], TF.random_orders(rid, 3, 13)])
    keep = np.zeros((len(orders), 14, 13))
    for o in range(len(orders)):
        for k in range(1, 14):
            keep[o, k, orders[o, :k]] = 1
    D = Ub - Ux[:, None]                                                        # (25, 50, 13, 4)
    Sx, Sb = Ux.sum(1), Ub.sum(2)                                               # (25, 4), (25, 50, 4)
    Ld = Sx[None, None, :, None] + np.einsum("oki,mbij->okmbj", keep, D)
    Li = Sb[None, None] - np.einsum("oki,mbij->okmbj", keep, D)
    pd_ = K.softmax(Ld)[..., target].mean(2).mean(-1)                           # mean over models, then bg
    pi_ = K.softmax(Li)[..., target].mean(2).mean(-1)
    ad, ai = TF.area(pd_), TF.area(pi_)
    return ad[0] - ad[1:].mean(), ai[0] - ai[1:].mean(), pd_[0], pd_[1:].mean(0), pi_[0], pi_[1:].mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    if a.dry:
        C.set_dry()
    C.verify_manifest()
    torch.set_num_threads(4)
    t0 = time.time()
    config = K.cfg()
    d = load_rows()
    tr = d["split"] == "train"
    assert tr.sum() == 569 and (~tr).sum() == 122
    n = len(d["rid"])
    nm = len(K.MODELS)

    # --- frozen KANs: Shapley, efficiency, masked predictions --------------------------------------
    PHI = np.zeros((nm, n, 13, 9), np.float32)
    EBG = np.zeros((nm, 9))
    U = np.zeros((nm, n, 13, 4))
    P_scan, P_mask = np.zeros((nm, n, 4)), np.zeros((nm, n, 4))
    eff = 0.0
    frozen = {}
    for mi, (s, k) in enumerate(K.MODELS):
        m, inp = K.load_frozen(s, k, config)
        frozen[(s, k)] = (m, inp)
        pc, pm = d[f"pc_{s}"], d[f"pm_{s}"]
        T, f, out, base = K.terms(m, inp, pc, pm, d["has"], d["ex"])
        phi, ebg = K.shapley(T, T[tr])
        eff = max(eff, float(np.abs(phi.sum(1) + ebg - f).max()))
        PHI[mi], EBG[mi] = phi, ebg
        Lc = inp.raw(pc, pm, d["has"], d["ex"])[:, :4]
        with torch.no_grad():                                                    # raw (uncentred) edge terms for P5
            m(inp(pc, pm, d["has"], d["ex"]))
            Eraw = m.spline_postacts[0].numpy().astype(np.float64).transpose(0, 2, 1)[:, :, :4]
        u = Eraw.copy()
        u[:, np.arange(4), np.arange(4)] += Lc
        U[mi] = u
        P_scan[mi] = K.softmax(out[:, :4] + base)
        with torch.no_grad():
            nomri = np.zeros_like(d["has"])
            o2 = m(inp(pc, pm, nomri, d["ex"])).numpy()
        P_mask[mi] = K.softmax(o2[:, :4] + base)
    print(f"frozen KANs done {time.time() - t0:.0f}s; max efficiency error {eff:.2e}", flush=True)

    # brute force on 60 train subjects (dry: the 10 dry-run subjects)
    ptid_of = pd.read_csv(os.path.join(C.ROOT, "analysis", "kan_fusion", "fusion_trainval.csv"),
                          usecols=["RID", "PTID"]).set_index("RID").PTID
    dry_rids = set(ptid_of[ptid_of.isin(C.dry_run_ptids())].index)
    tr_idx = np.flatnonzero(tr)
    bf_idx = [i for i in tr_idx if d["rid"][i] in dry_rids]
    if not a.dry:
        rest = np.setdiff1d(tr_idx, bf_idx)
        bf_idx += list(np.sort(np.random.default_rng(0).choice(rest, 50, replace=False)))
    bf_models = K.MODELS if not a.dry else K.MODELS[:3]
    bf_err = 0.0
    for mi, (s, k) in enumerate(K.MODELS):
        if (s, k) not in bf_models:
            continue
        m, inp = frozen[(s, k)]
        bg = {"pc": d[f"pc_{s}"][tr], "pm": d[f"pm_{s}"][tr], "has": d["has"][tr], "ex": d["ex"][tr]}
        G = K.group(PHI[mi].astype(np.float64))
        for i in bf_idx:
            x = {"pc": d[f"pc_{s}"][i:i + 1], "pm": d[f"pm_{s}"][i:i + 1], "has": d["has"][i:i + 1],
                 "ex": d["ex"][i:i + 1]}
            bf_err = max(bf_err, float(np.abs(K.brute_force_group_shapley(m, inp, x, bg) - G[i]).max()))
    effj = {"max_abs_efficiency_error": eff, "pass": eff < EFF_TOL, "brute_force_subjects": len(bf_idx),
            "brute_force_models": len(bf_models), "max_abs_brute_force_diff": bf_err, "brute_force_pass": bf_err < EFF_TOL,
            "n_models": nm, "n_subjects": int(n), "background": "569 train subjects, prediction-time inputs"}
    print(effj, f"{time.time() - t0:.0f}s", flush=True)
    assert effj["pass"] and effj["brute_force_pass"], "P3-R0 failed: bug"
    C.write_json(effj, "p3_efficiency.json")

    G = np.stack([K.group(PHI[mi].astype(np.float64)) for mi in range(nm)])        # (25, n, 4, 9)
    np.savez_compressed(C.subj_path("p3_kan_shapley.npz"), rid=d["rid"], split=d["split"], y=d["y"],
                        phi_group=G.astype(np.float32), phi_input=PHI, ebg=EBG, models=np.array(K.MODELS),
                        groups=np.array(K.GROUP_NAMES), targets=np.array(K.TARGETS), inputs=np.array(K.INPUTS))

    # --- P3-R1 cells on train; descriptive val (frozen, in-sample) ------------------------------------
    cells = cells_table(G[:, tr], d["y"][tr], nm)
    va = cells_table(G[:, ~tr], d["y"][~tr], nm)
    cells["val_frozen_mean_effect"] = [float(G[:, ~tr][:, d["y"][~tr] == C.NAMES.index(r.stratum),
                                         K.GROUP_NAMES.index(r.group), K.TARGETS.index(r.target)].mean())
                                       for r in cells.itertuples()]
    del va

    # global (train): mean |phi| per group and target
    glob = []
    for g, gn in enumerate(K.GROUP_NAMES):
        for t, tn in enumerate(K.TARGETS):
            v = np.abs(G[:, tr, g, t]).mean(1)
            glob.append({"group": gn, "target": tn, "mean_abs_phi": float(v.mean()), "min_over_models": float(v.min()),
                         "max_over_models": float(v.max())})
    C.write_aggregate(pd.DataFrame(glob), "p3_global.csv", ["train"])

    # --- P3-R3 train-only KANs ------------------------------------------------------------------------
    rep = {"attempted": True}
    kr = np.load(os.path.join(C.V3, "kan_runs", "32.npz"))
    jobs = [(config, s, k) for s, k in K.MODELS]
    if a.dry:
        jobs = [j for j in jobs if j[1] == 42]
    KF.init_worker()
    with Pool(min(len(jobs), 12), initializer=KF.init_worker) as pool:
        res = pool.map(regen, jobs)
    res.sort(key=lambda r: (r[0], r[1]))
    gate = []
    for s in sorted({r[0] for r in res}):
        rs = [r for r in res if r[0] == s]
        p = np.mean([r[2] for r in rs], 0)
        ad = np.mean([r[3] for r in rs], 0)
        si = (42, 43, 44, 45, 46).index(s)
        gate.append({"seed": s, "max_abs_p_diff": float(np.abs(p - kr["val"][si]).max()),
                     "max_abs_add_diff": float(np.abs(ad - kr["val_add"][si]).max())})
    passed = all(g["max_abs_p_diff"] < GATE_TOL and g["max_abs_add_diff"] < GATE_TOL for g in gate)
    rep.update(gate=gate, passed=bool(passed), tolerance=GATE_TOL, n_models=len(res))
    print("reproduction gate", rep, f"{time.time() - t0:.0f}s", flush=True)
    val_rid = res[0][8]
    val_pos = {r: i for i, r in enumerate(d["rid"])}
    vix = np.array([val_pos[r] for r in val_rid])
    assert (d["split"][vix] == "val").all()
    P_val_scan = P_val_mask = None
    if rep["passed"]:
        Gv, Pvs, Pvm = [], [], []
        for s, k, p, adp, state, inp, trd, ted, _ in res:
            m = KF.build(config, 13, k)
            m.load_state_dict(state)
            Tv, fv, outv, basev = K.terms(m, inp, ted["pc"], ted["pm"], ted["has"], ted["extras"])
            Tb, _, _, _ = K.terms(m, inp, trd["pc"], trd["pm"], trd["has"], trd["extras"])
            phv, ebv = K.shapley(Tv, Tb)
            assert np.abs(phv.sum(1) + ebv - fv).max() < EFF_TOL
            assert np.abs(K.softmax(outv[:, :4] + basev) - p).max() < 1e-5
            Gv.append(K.group(phv))
            Pvs.append(p)
            with torch.no_grad():
                o2 = m(inp(ted["pc"], ted["pm"], np.zeros_like(ted["has"]), ted["extras"])).numpy()
            Pvm.append(K.softmax(o2[:, :4] + basev))
        Gv = np.stack(Gv)
        yv = d["y"][vix]
        nmv = len(Gv)
        rcol = []
        for r in cells.itertuples():
            if not r.consistent:
                rcol.append(None)
                continue
            e = Gv[:, yv == C.NAMES.index(r.stratum), K.GROUP_NAMES.index(r.group), K.TARGETS.index(r.target)].mean(1)
            agree = int((np.sign(e) == np.sign(r.median_effect)).sum())
            rcol.append("replicated" if agree == nmv else ("partially replicated" if agree >= 13 else "not replicated"))
        cells["val_trainonly_replication"] = rcol
        cells["val_trainonly_models_agreeing"] = [
            None if not r.consistent else int((np.sign(Gv[:, yv == C.NAMES.index(r.stratum), K.GROUP_NAMES.index(r.group),
                                                         K.TARGETS.index(r.target)].mean(1)) == np.sign(r.median_effect)).sum())
            for r in cells.itertuples()]
        P_val_scan, P_val_mask = np.mean(Pvs, 0), np.mean(Pvm, 0)
        rep["n_val"] = int(len(vix))
    else:
        cells["val_trainonly_replication"] = None
        rep["note"] = "reproduction gate failed: held-out replication skipped; P4-B uses the frozen models only"
    C.write_json(rep, "p3_replication.json")
    C.write_aggregate(cells, "p3_sign_consistency.csv", ["train", "val"])
    cons = cells[cells.consistent]
    print(f"consistent cells: {len(cons)} of {len(cells)}", flush=True)
    C.update_json("card_rules.json", "kan", {
        "consistent_cells": cons[["group", "target", "stratum", "median_effect"] +
                                 (["val_trainonly_replication"] if "val_trainonly_replication" in cons else [])
                                 ].to_dict("records"),
        "min_effect": MIN_EFFECT, "groups": K.GROUP_NAMES, "replication_gate_passed": rep["passed"]})

    # P4-B arrays (subject level)
    np.savez_compressed(C.subj_path("p4_mri_masked.npz"), rid=d["rid"], split=d["split"], y=d["y"], has=d["has"],
                        frozen_scan=P_scan.mean(0), frozen_mask=P_mask.mean(0), val_rid=val_rid,
                        trainonly_val_scan=P_val_scan if P_val_scan is not None else np.empty(0),
                        trainonly_val_mask=P_val_mask if P_val_mask is not None else np.empty(0))

    # --- P5 row -----------------------------------------------------------------------------------------
    P_ens = P_scan.mean(0)
    pred = P_ens.argmax(1)
    row = {"explanation": "KAN Shapley", "detail": "exact interventional Shapley, 25 frozen KANs, 13 inputs / 4 groups"}
    for split, rows in (("train", tr), ("val", ~tr)):
        imp = np.abs(PHI[:, rows, :, :4]).mean(1).mean(-1)                         # (25, 13)
        row[f"stability_{split}"], row[f"stability_{split}_pairs"] = C.median_pairwise_spearman(imp)
        from scipy.stats import rankdata
        psub = []
        for i in np.flatnonzero(rows):
            v = PHI[:, i, :, pred[i]]
            rk = np.apply_along_axis(rankdata, 1, v)
            cm = np.corrcoef(rk)
            psub.append(np.nanmedian(cm[np.triu_indices(nm, 1)]))
        row[f"stability_per_subject_{split}"] = float(np.nanmedian(psub))
    pl = {}
    for nmf, idx, hi in (("CDRSB", 10, True), ("MMSE", 12, False)):
        col = d[f"raw_{nmf}"]
        med = float(np.nanmedian(col[tr & (d["y"] == 0)]))
        out = {}
        for split, rows in (("train", tr), ("val", ~tr)):
            e = np.sign(col[rows] - med) * (1 if hi else -1)
            e[np.isnan(col[rows])] = 0
            p = PHI[:, rows, idx, 3].mean(0)
            ok = (e != 0) & (np.abs(p) > 1e-6)
            out[split] = float((np.sign(p[ok]) == e[ok]).mean()) if ok.any() else np.nan
            out[f"n_{split}"] = int(ok.sum())
        pl[nmf] = out
    row["plausibility"] = pl
    bg = np.sort(np.random.default_rng(0).choice(np.flatnonzero(tr), 50, replace=False))
    faith, curves = {}, []
    sets = {"train": np.flatnonzero(tr), "val": np.flatnonzero(~tr)}
    if a.dry:
        sets = {"train": np.array(bf_idx), "val": np.flatnonzero(~tr)[:3]}
    for split, ix in sets.items():
        res_f = []
        for i in ix:
            phi_t = PHI[:, i, :, pred[i]].mean(0)
            res_f.append(faith_kan(U[:, i], U[:, bg], pred[i], TF.attribution_order(phi_t), d["rid"][i]))
        dd = np.array([r[0] for r in res_f])
        ii = np.array([r[1] for r in res_f])
        dm, dlo, dhi = C.boot_ci(dd)
        im, ilo, ihi = C.boot_ci(ii)
        dp, ip, v = TF.verdict(dhi, ilo)
        faith[split] = {"n": len(ix), "deletion_diff": dm, "deletion_ci": [dlo, dhi], "deletion_pass": dp,
                        "insertion_diff": im, "insertion_ci": [ilo, ihi], "insertion_pass": ip, "verdict": v}
        for j, kname in enumerate(("del_attr", "del_rand", "ins_attr", "ins_rand")):
            for step, y in enumerate(np.mean([r[2 + j] for r in res_f], 0)):
                curves.append({"split": split, "curve": kname, "step": step, "mean_prob": y})
        print(split, faith[split], f"{time.time() - t0:.0f}s", flush=True)
    row["faithfulness"] = faith
    C.write_aggregate(pd.DataFrame(curves), "p5_kan_curves.csv", ["train", "val"])
    C.update_json("p5_rows.json", "kan", row)
    print({k: v for k, v in row.items() if k != "faithfulness"})
    print(f"done {time.time() - t0:.0f}s")
    C.verify_manifest()


if __name__ == "__main__":
    main()
