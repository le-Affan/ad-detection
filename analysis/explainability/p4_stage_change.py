"""P4 (LYP/.venv-fusion; numpy / pandas): stage-change indicators.

P4-A "Fusion changed the stage": argmax(final) != argmax(clinical specialist).
  train out-of-fold: streams.npz clin_oof vs kan_runs/32.npz cv; val: clin_val vs kan_runs/32.npz val.
  Primary: mean of the 5 seeds' probabilities, argmax; per-seed range reported.
P4-B "MRI changed the stage": argmax(ensemble, with the scan) != argmax(ensemble, MRI inputs masked), subjects
  with a scan. Held out: the train-only KANs on val (P3-R3, if the reproduction gate passed). In-sample,
  descriptive: the 25 frozen KANs on train and val. Arrays from p3_kan_shapley.py (subject_level).
Outcomes among firings (reference = clinical for A, MRI-masked for B): corrected, broken, both wrong closer /
farther / same distance. Net effects (bootstrap CIs over subjects): (corrected - broken) / N and the mean
reduction of |stage error| over all N; by transition.
Discordance flag (exploratory, kan-fusion rule) tabulated on train out-of-fold and val.
Outputs: results/p4_summary.csv, results/p4_transitions.csv, results/p4_discordance.csv,
card_rules.json["stage_change"].
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

P10, P90 = 0.2981, 0.8751
MIN_TRANSITION_N = 10


def outcomes(ref, fin, y):
    fired = ref != fin
    corrected = fired & (ref != y) & (fin == y)
    broken = fired & (ref == y) & (fin != y)
    both = fired & (ref != y) & (fin != y)
    dr, df = np.abs(ref - y), np.abs(fin - y)
    return {"fired": fired, "corrected": corrected, "broken": broken, "both_closer": both & (df < dr),
            "both_farther": both & (df > dr), "both_same": both & (df == dr), "err_reduction": dr - df}


def summarise(name, split, ref, fin, y, per_seed=None):
    o = outcomes(ref, fin, y)
    n = len(y)
    net = (o["corrected"].astype(float) - o["broken"].astype(float))
    m, lo, hi = C.boot_ci(net)
    em, elo, ehi = C.boot_ci(o["err_reduction"].astype(float))
    fm, flo, fhi = C.boot_ci(o["fired"].astype(float))
    row = {"indicator": name, "split": split, "n": n, "fired": int(o["fired"].sum()), "fire_rate": fm,
           "fire_rate_ci_lo": flo, "fire_rate_ci_hi": fhi, "corrected": int(o["corrected"].sum()),
           "broken": int(o["broken"].sum()), "both_wrong_closer": int(o["both_closer"].sum()),
           "both_wrong_farther": int(o["both_farther"].sum()), "both_wrong_same": int(o["both_same"].sum()),
           "moved_toward": int(o["corrected"].sum() + o["both_closer"].sum()),
           "moved_away": int(o["broken"].sum() + o["both_farther"].sum()),
           "net_accuracy_change": m, "net_ci_lo": lo, "net_ci_hi": hi,
           "mean_stage_error_reduction": em, "err_ci_lo": elo, "err_ci_hi": ehi}
    if per_seed is not None:
        nets = [float((outcomes(r, f, y)["corrected"].astype(float) - outcomes(r, f, y)["broken"]).mean())
                for r, f in per_seed]
        fires = [float((r != f).mean()) for r, f in per_seed]
        row.update(per_seed_net_min=min(nets), per_seed_net_max=max(nets), per_seed_fire_min=min(fires),
                   per_seed_fire_max=max(fires))
    return row, o


def transitions(name, split, ref, fin, y):
    o = outcomes(ref, fin, y)
    rows = []
    for a in range(4):
        for b in range(4):
            if a == b:
                continue
            s = (ref == a) & (fin == b)
            if not s.any():
                continue
            rows.append({"indicator": name, "split": split, "from": C.NAMES[a], "to": C.NAMES[b], "n": int(s.sum()),
                         "corrected": int((o["corrected"] & s).sum()), "broken": int((o["broken"] & s).sum()),
                         "both_wrong_closer": int((o["both_closer"] & s).sum()),
                         "both_wrong_farther": int((o["both_farther"] & s).sum()),
                         "both_wrong_same": int((o["both_same"] & s).sum())})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    if a.dry:
        C.set_dry()
    C.verify_manifest()
    st = np.load(os.path.join(C.V3, "streams.npz"))
    kr = np.load(os.path.join(C.V3, "kan_runs", "32.npz"))
    fv = pd.read_csv(os.path.join(C.ROOT, "analysis", "kan_fusion", "fusion_trainval.csv"))
    tr_df, va_df = fv[fv.split == "train"].reset_index(drop=True), fv[fv.split == "val"].reset_index(drop=True)
    assert (st["train_rids"] == tr_df.RID.to_numpy()).all() and (st["val_rids"] == va_df.RID.to_numpy()).all()
    C.assert_no_test(list(tr_df.RID) + list(va_df.RID))
    summ, trans = [], []
    data = {"train_oof": (st["clin_oof"], kr["cv"], tr_df), "val": (st["clin_val"], kr["val"], va_df)}
    for split, (clin, fin, df) in data.items():
        y = df.label.to_numpy()
        ref, f = clin.mean(0).argmax(1), fin.mean(0).argmax(1)
        row, _ = summarise("A: fusion changed the stage", split, ref, f, y,
                           per_seed=[(clin[i].argmax(1), fin[i].argmax(1)) for i in range(5)])
        summ.append(row)
        trans += transitions("A: fusion changed the stage", split, ref, f, y)

    # P4-B
    mm = np.load(C.subj_path("p4_mri_masked.npz"), allow_pickle=True)
    has = mm["has"].astype(bool)
    for split in ("train", "val"):
        s = (mm["split"] == split) & has
        ref, f, y = mm["frozen_mask"][s].argmax(1), mm["frozen_scan"][s].argmax(1), mm["y"][s]
        row, _ = summarise("B: MRI changed the stage (frozen KANs, in-sample, descriptive)", split, ref, f, y)
        summ.append(row)
        trans += transitions("B: MRI changed the stage (frozen, in-sample)", split, ref, f, y)
    if mm["trainonly_val_scan"].size:
        pos = {r: i for i, r in enumerate(mm["rid"])}
        idx = np.array([pos[r] for r in mm["val_rid"]])
        s = has[idx]
        ref = mm["trainonly_val_mask"][s].argmax(1)
        f = mm["trainonly_val_scan"][s].argmax(1)
        y = mm["y"][idx][s]
        row, _ = summarise("B: MRI changed the stage (train-only KANs, held out)", "val", ref, f, y)
        summ.append(row)
        trans += transitions("B: MRI changed the stage (train-only, held out)", "val", ref, f, y)
    summ = pd.DataFrame(summ)
    trans = pd.DataFrame(trans)
    C.write_aggregate(summ, "p4_summary.csv", ["train", "val"])
    C.write_aggregate(trans, "p4_transitions.csv", ["train", "val"])
    print(summ[["indicator", "split", "n", "fired", "corrected", "broken", "both_wrong_closer", "both_wrong_farther",
                "both_wrong_same", "net_accuracy_change", "net_ci_lo", "net_ci_hi"]].round(4).to_string(index=False))

    # discordance flag (exploratory), train out-of-fold and val
    alz = pd.read_csv(os.path.join(C.ROOT, "analysis", "fusion_feasibility", "alz_crossfit.csv")).set_index("RID").ALZ
    disc = []
    for split, (clin, fin, df) in data.items():
        y = df.label.to_numpy()
        cp, fp = clin.mean(0).argmax(1), fin.mean(0).argmax(1)
        al = df.RID.map(alz).to_numpy(float)
        ev = ~np.isnan(al)
        flag = ev & (((cp == 3) & (al < P10)) | (np.isin(cp, (0, 1)) & (al > P90)))
        for who, p in (("clinical specialist", cp), ("final (fusion)", fp)):
            err = (p != y).astype(float)
            idx = C.boot_idx(len(y))
            bs = [err[i][flag[i]].mean() - err[i][~flag[i] & ev[i]].mean() if flag[i].any() else np.nan for i in idx]
            disc.append({"split": split, "model": who, "evaluable": int(ev.sum()), "flagged": int(flag.sum()),
                         "error_rate_flagged": float(err[flag].mean()) if flag.any() else np.nan,
                         "error_rate_unflagged": float(err[~flag & ev].mean()),
                         "difference_ci_lo": float(np.nanpercentile(bs, 2.5)) if flag.any() else np.nan,
                         "difference_ci_hi": float(np.nanpercentile(bs, 97.5)) if flag.any() else np.nan})
    disc = pd.DataFrame(disc)
    C.write_aggregate(disc, "p4_discordance.csv", ["train", "val"])
    print(disc.round(3).to_string(index=False))

    trA = trans[(trans.indicator.str.startswith("A")) & (trans.split == "train_oof")]
    rules = {"min_transition_n": MIN_TRANSITION_N,
             "A_train_oof": summ[(summ.indicator.str.startswith("A")) & (summ.split == "train_oof")].iloc[0].to_dict(),
             "A_val": summ[(summ.indicator.str.startswith("A")) & (summ.split == "val")].iloc[0].to_dict(),
             "A_transitions_train_oof": trA.to_dict("records"),
             "B_heldout_val": (summ[summ.indicator.str.contains("held out")].iloc[0].to_dict()
                               if summ.indicator.str.contains("held out").any() else None),
             "B_frozen_train": summ[(summ.indicator.str.contains("frozen")) & (summ.split == "train")].iloc[0].to_dict()}
    C.update_json("card_rules.json", "stage_change", rules)
    C.verify_manifest()


if __name__ == "__main__":
    main()
