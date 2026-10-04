"""Card data, step 1 of 3 (run with LYP/.venv-fusion): per-subject inputs, stream
outputs and clinical SHAP from the frozen fusion-v3 artefacts. No model is
refitted; no metric is computed.

For every train, val and test subject:
  - features built with frozen/feature_stats.json (final_features.build):
    train/val from the train/val scan tables (source "fit", the tables the
    frozen system was fitted on), test from scan_inputs_test.csv and
    brainage_test.csv (source "test", written during the single test run);
  - clinical specialist and MRI specialist probabilities for each seed
    (frozen/clinical_s.cbm, frozen/mri_s.pkl); for val and test these must
    reproduce frozen_run/<split>_streams.npz;
  - clinical SHAP (CatBoost ShapValues, raw multiclass log-odds) per seed,
    mapped onto the 29 clinical features (0 where a seed's top 12 omits a
    feature) and averaged over the 5 seeds;
  - brain age: raw gap, corrected gap with the frozen MRI specialist's alpha,
    beta; status reliable / unreliable (pre-registered rule) / not available;
    "extreme value" note when |corrected gap| >= EXTREME_GAP years (rule fixed
    before any card output);
  - discordance flag (pre-registered, kan-fusion): clinical specialist
    predicts AD and ALZ < P10, or predicts CN/EMCI and ALZ > P90;
  - FreeSurfer volume / ICV ratios against CN-train reference percentiles.
Outputs (demo/card_data/): subjects.csv, clinical_shap.npz, streams_all.npz,
meta.json.
"""
import json
import os
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
V3 = os.path.join(ROOT, "analysis", "fusion_v3")
sys.path.insert(0, V3)
import final_features as FF  # noqa: E402
import metrics as M  # noqa: E402
import specialists as S  # noqa: E402

FROZEN = os.path.join(V3, "frozen")
OUT = os.path.join(ROOT, "demo", "card_data")
FREEZE_COMMIT = "8f351b6"
EXTREME_GAP = 15.0                      # years; |corrected gap| at or above this gets the "extreme value" note
P10, P90 = 0.2981, 0.8751               # discordance thresholds, kan-fusion preregistration item g
NAMES = M.NAMES
LABELS = {
    "AGE": "Age (years)", "PTGENDER": "Sex", "PTEDUCAT": "Education (years)", "APOE4": "APOE ε4 alleles",
    "CDRSB": "CDR sum of boxes", "ADAS11": "ADAS-Cog 11", "ADAS13": "ADAS-Cog 13", "MMSE": "MMSE",
    "RAVLT_immediate": "RAVLT immediate recall", "RAVLT_learning": "RAVLT learning",
    "RAVLT_forgetting": "RAVLT forgetting", "RAVLT_perc_forgetting": "RAVLT % forgetting", "FAQ": "FAQ (function)",
    "MOCA": "MoCA", "EcogPtMem": "ECog self: memory", "EcogPtLang": "ECog self: language",
    "EcogPtVisspat": "ECog self: visuospatial", "EcogPtPlan": "ECog self: planning",
    "EcogPtOrgan": "ECog self: organisation", "EcogPtDivatt": "ECog self: divided attention",
    "EcogPtTotal": "ECog self: total", "EcogSPMem": "ECog partner: memory", "EcogSPLang": "ECog partner: language",
    "EcogSPVisspat": "ECog partner: visuospatial", "EcogSPPlan": "ECog partner: planning",
    "EcogSPOrgan": "ECog partner: organisation", "EcogSPDivatt": "ECog partner: divided attention",
    "EcogSPTotal": "ECog partner: total", "CogIndex": "Cognitive index (z MMSE − z CDR-SB − z ADAS13)",
}
REGION = {"Ventricles_ICV": "Ventricles", "Hippocampus_ICV": "Hippocampus", "WholeBrain_ICV": "Whole brain",
          "Entorhinal_ICV": "Entorhinal cortex", "Fusiform_ICV": "Fusiform gyrus", "MidTemp_ICV": "Middle temporal gyrus"}


def rids(split):
    return list(pd.read_csv(os.path.join(FF.PROCESSED, f"{split}_rids.csv")).RID)


def cb(path):
    m = CatBoostClassifier()
    m.load_model(path)
    return m


def versions():
    import catboost
    import sklearn
    v = {"catboost": catboost.__version__, "scikit-learn": sklearn.__version__}
    try:
        v["git_head"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                                       text=True).stdout.strip()
    except OSError:
        pass
    return v


def main():
    os.makedirs(OUT, exist_ok=True)
    stats = FF.load_stats(os.path.join(FROZEN, "feature_stats.json"))
    cols = json.load(open(os.path.join(FROZEN, "columns.json")))
    feats = S.clinical_features()
    tv = FF.build(rids("train") + rids("val"), stats, "fit")
    te = FF.build(rids("test"), stats, "test")
    split_of = {r: s for s in ("train", "val", "test") for r in rids(s)}
    d = pd.concat([tv, te], ignore_index=True)
    d["split"] = d.RID.map(split_of)
    d = d.sort_values("PTID").reset_index(drop=True)
    raw = FF.raw_rows(d.RID).set_index("RID").loc[d.RID].reset_index()
    n = len(d)

    # --- streams and SHAP ------------------------------------------------------------
    streams = {"rids": d.RID.to_numpy(), "has": d.has_mri.to_numpy(), "extras": d[cols["extras"]].to_numpy(float)}
    shap = np.zeros((len(M.SEEDS), n, len(feats), 4))
    mris = []
    for si, s in enumerate(M.SEEDS):
        top = cols["clinical_top12"][str(s)]
        m = cb(os.path.join(FROZEN, f"clinical_{s}.cbm"))
        streams[f"clin_{s}"] = m.predict_proba(d[top])
        sv = m.get_feature_importance(Pool(d[top], cat_features=[c for c in top if c == "PTGENDER"]),
                                      type="ShapValues")                     # (n, 4, 12 + 1)
        for j, c in enumerate(top):
            shap[si, :, feats.index(c), :] = sv[:, :, j]
        mri = pickle.load(open(os.path.join(FROZEN, f"mri_{s}.pkl"), "rb"))
        mris.append(mri)
        streams[f"mri_{s}"] = mri.predict_proba(d)
    for sp in ("val", "test"):                                               # card = frozen-run outputs
        z = np.load(os.path.join(V3, "frozen_run", f"{sp}_streams.npz"))
        idx = pd.Series(np.arange(n), index=d.RID).loc[z["rids"]].to_numpy()
        for s in M.SEEDS:
            assert np.abs(streams[f"clin_{s}"][idx] - z[f"clin_{s}"]).max() < 1e-6
            assert np.nanmax(np.abs(streams[f"mri_{s}"][idx] - z[f"mri_{s}"])) < 1e-4
        assert (d.has_mri.to_numpy()[idx] == z["has"]).all()
    np.savez_compressed(os.path.join(OUT, "streams_all.npz"), **streams)
    np.savez_compressed(os.path.join(OUT, "clinical_shap.npz"), ptid=d.PTID.to_numpy(), features=np.array(feats),
                        shap=shap.mean(0).astype(np.float32))

    # --- per-subject table -----------------------------------------------------------
    pc = np.mean([streams[f"clin_{s}"] for s in M.SEEDS], 0)
    pm = np.mean([streams[f"mri_{s}"] for s in M.SEEDS], 0)
    alpha, beta = mris[0].alpha, mris[0].beta
    assert all(abs(m.alpha - alpha) < 1e-12 and abs(m.beta - beta) < 1e-12 for m in mris)
    t = pd.DataFrame({"PTID": d.PTID, "RID": d.RID, "split": d.split, "true": d.label.map(dict(enumerate(NAMES))),
                      "has_mri": d.has_mri, "alz": d.ALZ})
    for k, nm in enumerate(NAMES):
        t[f"clin_p_{nm}"] = pc[:, k]
        t[f"mri_p_{nm}"] = pm[:, k]
    t["clin_pred"] = np.array(NAMES)[pc.argmax(1)]
    t["mri_pred"] = np.where(d.has_mri, np.array(NAMES)[np.nan_to_num(pm).argmax(1)], None)

    # brain age
    ba_tv = pd.read_csv(os.path.join(ROOT, "analysis", "brain_age", "brainage_trainval.csv"),
                        usecols=["RID", "missing_reason"])
    ba_te = pd.read_csv(os.path.join(V3, "brainage_test.csv"), usecols=["RID", "has_scan", "excluded"])
    reason = dict(zip(ba_tv.RID, ba_tv.missing_reason))
    reason.update({r: ("registration check failed" if e else ("no scan" if not h else None))
                   for r, h, e in zip(ba_te.RID, ba_te.has_scan, ba_te.excluded)})
    t["ba_age_at_scan"], t["ba_pred_age"] = d.age_at_scan, d.pred_age
    t["ba_raw_gap"] = d.pred_age - d.age_at_scan
    t["ba_corr_gap"] = (d.pred_age - beta) / alpha - d.age_at_scan
    unrel = d.unreliable.fillna(False).astype(bool)
    t["ba_unreliable"] = unrel
    t["ba_status"] = np.where(d.pred_age.isna(), "not available", np.where(unrel, "unreliable", "reliable"))
    t["ba_missing_reason"] = [None if pd.notna(p) else (reason.get(r) if isinstance(reason.get(r), str) else
                                                        ("no scan" if not h else "not available"))
                              for r, p, h in zip(d.RID, d.pred_age, d.has_mri)]
    t["ba_extreme"] = t.ba_corr_gap.abs() >= EXTREME_GAP

    # discordance (clinical specialist prediction vs ALZ)
    a = d.ALZ.to_numpy()
    ev = ~np.isnan(a)
    b1 = ev & (pc.argmax(1) == 3) & (a < P10)
    b2 = ev & np.isin(pc.argmax(1), (0, 1)) & (a > P90)
    t["discordance_evaluable"] = ev
    t["discordant"] = b1 | b2
    t["discordance_branch"] = np.where(b1, "clinical AD, low ALZ", np.where(b2, "clinical CN/EMCI, high ALZ", ""))

    # volumes / ICV, raw (not imputed); CN-train reference percentiles
    cn_tr = (d.split == "train") & (d.label == 0)
    ref = {}
    for v, r in zip(FF.VOLUMES, FF.RATIOS):
        ratio = (raw[v] / raw["ICV"]).to_numpy()
        t[f"vol_{r}"] = ratio
        x = ratio[cn_tr.to_numpy()]
        x = x[~np.isnan(x)]
        ref[r] = {"label": REGION[r], "n": int(len(x)),
                  "pct": dict(zip(("p5", "p25", "p50", "p75", "p95"), np.percentile(x, [5, 25, 50, 75, 95]).tolist()))}
        t[f"volpct_{r}"] = [np.nan if np.isnan(u) else float((x < u).mean() * 100) for u in ratio]

    # raw clinical values (as recorded; NaN = missing, imputed for the model)
    for c in feats:
        t[f"raw_{c}"] = raw[c].to_numpy() if c in raw else d[c].to_numpy()
    t["raw_CogIndex"] = d.CogIndex.to_numpy()                                 # derived composite (z units)
    t.to_csv(os.path.join(OUT, "subjects.csv"), index=False)

    ev_sum = json.load(open(os.path.join(V3, "eval_summary.json")))
    meta = {"freeze_commit": FREEZE_COMMIT, "names": NAMES, "clinical_features": feats, "feature_labels": LABELS,
            "volume_reference": ref, "brain_age": {"alpha": float(alpha), "beta": float(beta),
                                                   "extreme_gap_years": EXTREME_GAP},
            "discordance": {"P10": P10, "P90": P90},
            "severity_stage_means_cv": ev_sum["severity"]["KAN K2 (final)"]["cv"]["means"],
            "versions": versions(),
            "n": {s: int((t.split == s).sum()) for s in ("train", "val", "test")}}
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
    print(f"{n} subjects ({meta['n']}); with a scan {int(t.has_mri.sum())}; brain age "
          f"{t.ba_status.value_counts().to_dict()}; discordant {int(t.discordant.sum())}; extreme gaps "
          f"{int(t.ba_extreme.sum())}")


if __name__ == "__main__":
    main()
