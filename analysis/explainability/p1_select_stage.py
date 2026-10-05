"""P1 step 2 (LYP/.venv-mri, GPU): stage selection by the pre-registered deletion rule (P1-R1).

Subjects: 100 train AD with a scan, default_rng(0).choice of the 104 sorted by PTID (without replacement).
For each subject, each of its 3 out-of-fold models m and each stage s:
  D      = ALZ logit_m(x) - ALZ logit_m(x with the top 10% of brain voxels by model m's LayerCAM map
           replaced by the CN-train mean volume)
  D_rand = mean over 10 masks (top 10% of brain voxels of white noise smoothed with sigma 4 / 8 / 32 voxels,
           seed SeedSequence([RID, stage index, draw])), same replacement, same model.
Subject values = mean over its 3 models. Eligible: bootstrap 95% CI of mean(D - D_rand) has lower bound > 0.
Chosen = eligible stage with the largest mean D (ties within 1e-6 -> finer stage); if none is eligible, the
largest mean D, and CNN faithfulness is recorded as FAIL.
Grad-CAM is computed the same way for reference only (never selected).
Outputs: results/p1_stage_selection.csv, results/card_rules.json["cnn_stage"]; per-subject drops in
subject_level/p1_deletion.csv (gitignored). --dry: the dry-run AD subjects; all outputs under cache/dry/.
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import p1_lib as L  # noqa: E402

N_RAND, FRAC = 10, 0.10


def rand_masks(args):
    rid, mask = args
    return [[L.top_fraction_mask(L.noise_field((rid, si, k), L.SIGMA[s]), mask, FRAC) for k in range(N_RAND)]
            for si, s in enumerate(L.STAGES)]


def chosen_subjects(dry):
    d = L.subjects()
    if dry:
        return d[d.PTID.isin(C.dry_run_ptids()) & (d.label == 3)].reset_index(drop=True)
    ad = d[(d.split == "train") & (d.label == 3)].sort_values("PTID").reset_index(drop=True)
    assert len(ad) == 104, len(ad)
    pick = np.sort(np.random.default_rng(0).choice(len(ad), 100, replace=False))
    return ad.iloc[pick].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    if a.dry:
        C.set_dry()
    C.verify_manifest()
    t0 = time.time()
    d = chosen_subjects(a.dry)
    C.assert_no_test(d.RID)
    z = np.load(C.cache_path("cams.npz"))
    pair = {(p, m): i for i, (p, m) in enumerate(zip(z["pair_ptid"], z["pair_model"]))}
    mask = L.brain_mask()
    base = L.cn_train_mean()
    vols = L.load_volumes(list(d.path))
    with ProcessPoolExecutor(8) as ex:
        rmasks = list(ex.map(rand_masks, [(int(r), mask) for r in d.RID]))
    print(f"{len(d)} subjects; random masks ready {time.time() - t0:.0f}s", flush=True)
    rows = []
    for fold in sorted(d.fold.unique()):
        mids = [i for i, (k, _) in enumerate(L.MODELS) if k == int(fold)]
        models = {m: L.get_model(m) for m in mids}
        for i in np.flatnonzero(d.fold.to_numpy() == fold):
            r = d.iloc[i]
            for m in mids:
                batch, tags = [vols[i]], [("orig", None, None, -1)]
                for si, s in enumerate(L.STAGES):
                    for meth in L.METHODS:
                        mp = L.upsample(z[f"{meth}_{s}"][pair[(r.PTID, m)]][None].astype(np.float32), mask)[0]
                        if mp.sum() <= 0:                       # all-zero map (Grad-CAM only): nothing to occlude
                            tags.append(("empty", s, meth, -1))
                            batch.append(vols[i])
                            continue
                        batch.append(L.occlude(vols[i], L.top_fraction_mask(mp, mask, FRAC), base))
                        tags.append(("attr", s, meth, -1))
                    for k in range(N_RAND):
                        batch.append(L.occlude(vols[i], rmasks[i][si][k], base))
                        tags.append(("rand", s, None, k))
                lg = L.alz_logit(models[m], np.stack(batch))
                for (kind, s, meth, k), v in zip(tags[1:], lg[1:]):
                    rows.append({"PTID": r.PTID, "model": m, "kind": kind, "stage": s, "method": meth, "draw": k,
                                 "dlogit": float(lg[0] - v)})
        del models
        torch.cuda.empty_cache()
        print(f"fold {fold} done {time.time() - t0:.0f}s", flush=True)
    raw = pd.DataFrame(rows)
    per = []                                                         # subject x stage x method: D, D_rand
    for s in L.STAGES:
        rnd = raw[(raw.kind == "rand") & (raw.stage == s)].groupby("PTID").dlogit.mean()
        for meth in L.METHODS:
            att = raw[(raw.kind.isin(["attr", "empty"])) & (raw.stage == s) & (raw.method == meth)]
            att = att.assign(dlogit=np.where(att.kind == "empty", 0.0, att.dlogit)).groupby("PTID").dlogit.mean()
            per.append(pd.DataFrame({"PTID": att.index, "stage": s, "method": meth, "D": att.values,
                                     "D_rand": rnd.loc[att.index].values}))
    per = pd.concat(per, ignore_index=True)
    out = []
    for (s, meth), g in per.groupby(["stage", "method"], sort=False):
        mD, loD, hiD = C.boot_ci(g.D.to_numpy())
        mE, loE, hiE = C.boot_ci((g.D - g.D_rand).to_numpy())
        out.append({"stage": s, "map_size": L.STAGE_SIZE[s], "method": meth, "n_subjects": len(g),
                    "mean_D": mD, "D_ci_lo": loD, "D_ci_hi": hiD, "mean_D_rand": g.D_rand.mean(),
                    "mean_excess": mE, "excess_ci_lo": loE, "excess_ci_hi": hiE,
                    "eligible": bool(loE > 0) if meth == "layercam" else None})
    tab = pd.DataFrame(out)
    lc = tab[tab.method == "layercam"].copy()
    pool = lc[lc.eligible] if lc.eligible.any() else lc
    best = pool.mean_D.max()
    tied = pool[pool.mean_D >= best - 1e-6]
    chosen = tied.sort_values("map_size", ascending=False).iloc[0]                # ties -> finer stage
    tab["chosen"] = (tab.method == "layercam") & (tab.stage == chosen.stage)
    print(tab.round(4).to_string(index=False))
    print(f"chosen stage: {chosen.stage} (eligible stages: {list(lc[lc.eligible].stage)}); {time.time() - t0:.0f}s; "
          f"peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    raw.to_csv(C.subj_path("p1_deletion_raw.csv"), index=False)
    per.to_csv(C.subj_path("p1_deletion.csv"), index=False)
    C.write_aggregate(tab, "p1_stage_selection.csv", ["train"])
    C.update_json("card_rules.json", "cnn_stage", {
        "stage": chosen.stage, "map_size": int(chosen.map_size), "method": "layercam",
        "eligible_stages": list(lc[lc.eligible].stage), "faithfulness_rule_fail": not bool(lc.eligible.any()),
        "sigma_voxels": L.SIGMA[chosen.stage]})
    C.verify_manifest()


if __name__ == "__main__":
    main()
