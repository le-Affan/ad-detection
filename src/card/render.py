"""Assessment card renderer (numpy, pandas, matplotlib only).

Reads the precomputed per-subject outputs in demo/card_data/ (written by
build_card_data.py, build_kan_terms.py and build_cam.py from the frozen
fusion-v3 artefacts) and the committed aggregate rules of the explainability
analysis (analysis/explainability/results/card_rules.json, eval_status.json),
and renders one subject's card as HTML and PNG. It never fits or runs a model,
so any train, val or test subject renders in about a second.

Explanation text is generated from the pre-registered rules, never fixed:
  - CNN: no region names; the panel is labelled not anatomically reliable (Addendum 1);
  - KAN contributions are shown only where all 25 KANs agree in sign (P3-R2) and
    cohort effects only for consistent cells (P3-R1);
  - every explanation panel carries its P5 validation status (eval_status.json).

    from card import CardData, render_card
    data = CardData()                      # finds demo/card_data from this file's location
    render_card(data, ptid, "demo/output")   # ptid: any PTID in data.ptids
"""
import base64
import html
import io
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

NAMES = ["CN", "EMCI", "LMCI", "AD"]
DATA_DIR = Path(__file__).resolve().parents[2] / "demo" / "card_data"
RULES_DIR = Path(__file__).resolve().parents[2] / "analysis" / "explainability" / "results"
PREREG_COMMIT = "050ca8e"
VOXEL_MM = (1.979, 2.442, 2.059)                     # vol96 voxel size (S, P, R)
MIN_EFFECT = 0.05
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE, ORANGE, GREY, RED = "#2a78d6", "#eb6834", "#b9b8b2", "#e34948"
SHOW_TOP = 6
NOT_CLINICAL = "Research prototype, not for clinical use."

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlecolor": INK,
                     "font.family": "DejaVu Sans"})


# --- data ---------------------------------------------------------------------------
class CardData:
    def __init__(self, data_dir=None, rules_dir=None):
        d = Path(data_dir) if data_dir else DATA_DIR
        rd = Path(rules_dir) if rules_dir else RULES_DIR
        self.dir = d
        self.subjects = pd.read_csv(d / "subjects.csv")
        self.meta = json.load(open(d / "meta.json"))
        sh = np.load(d / "clinical_shap.npz", allow_pickle=True)
        self.features, self.shap = list(sh["features"]), sh["shap"]
        k = np.load(d / "kan_terms.npz")
        self.kan = {key: k[key] for key in k.files}
        self.edges = json.load(open(d / "kan_edges.json"))
        assert (sh["ptid"] == self.subjects.PTID.to_numpy()).all()
        assert (k["rids"] == self.subjects.RID.to_numpy()).all()
        ms = np.load(d / "mri_shap.npz", allow_pickle=True)
        assert (ms["ptid"] == self.subjects.PTID.to_numpy()).all()
        self.mshap = {key: ms[key] for key in ms.files}
        ks = np.load(d / "kan_shapley.npz", allow_pickle=True)
        assert (ks["rids"] == self.subjects.RID.to_numpy()).all()
        self.kshap = {key: ks[key] for key in ks.files}
        self.rules = json.load(open(rd / "card_rules.json"))
        self.eval = json.load(open(rd / "eval_status.json"))
        ph = rd / "p5_posthoc_cdrsb.json"
        self.posthoc_cdrsb = json.load(open(ph)) if ph.exists() else None
        self.cam = None
        if (d / "cam.npz").exists():
            g = np.load(d / "cam.npz", allow_pickle=True)
            self.cam = {key: g[key] for key in g.files}
            self.cam_index = {p: i for i, p in enumerate(self.cam["ptid"])}
        self.index = {p: i for i, p in enumerate(self.subjects.PTID)}
        p = self.kan["p"]
        self.subjects["pred"] = np.array(NAMES)[p.argmax(1)]

    @property
    def ptids(self):
        return list(self.subjects.PTID)

    def table(self):
        """One row per subject: what the showcase rule needs (PTID, split, true, pred, flags)."""
        return self.subjects[["PTID", "split", "true", "pred", "discordant", "ba_unreliable", "has_mri"]].copy()

    def subject(self, ptid):
        if ptid not in self.index:
            raise KeyError(f"unknown subject {ptid!r}; see CardData().ptids")
        i = self.index[ptid]
        r = self.subjects.iloc[i]
        p = self.kan["p"][i].astype(float)
        clin = np.array([r[f"clin_p_{n}"] for n in NAMES])
        kc = int(clin.argmax())
        sv = self.shap[i][:, kc]
        order = [j for j in np.argsort(-np.abs(sv)) if abs(sv[j]) > 1e-6][:SHOW_TOP]
        shap_rows = [{"feature": self.features[j], "label": self.meta["feature_labels"][self.features[j]],
                      "value": fmt_value(self.features[j], r[f"raw_{self.features[j]}"]), "shap": float(sv[j])}
                     for j in order]
        cam = None
        if self.cam is not None and ptid in self.cam_index:
            j = self.cam_index[ptid]
            cam = {"bg": self.cam["bg"][j], "cam": self.cam["cam"][j], "n_models": int(self.cam["n_models"][j]),
                   "density": dict(zip(self.cam["regions"], self.cam["density"][j].astype(float)))}
        pred = NAMES[int(p.argmax())]
        mri_p = np.array([r[f"mri_p_{n}"] for n in NAMES])
        return {"row": r, "p": p, "pred": pred, "add": float(self.kan["add"][i]),
                "severity": float(p @ np.arange(4)), "clin": clin, "clin_pred": NAMES[kc], "shap": shap_rows,
                "mri": mri_p, "residual": self.kan["residual"][i].astype(float),
                "base": self.kan["base"][i].astype(float), "cam": cam,
                "mri_shap": self.mri_shap_rows(i, r, mri_p) if r.has_mri else None,
                "kan_shap": self.kan_shap(i, pred),
                "mri_masked_pred": NAMES[int(self.kshap["p_mri_masked"][i].argmax())] if r.has_mri else None}

    def mri_shap_rows(self, i, r, mri_p):
        """P2-R3: the 6 largest |phi| for the MRI specialist's predicted class, then the 3 largest for AD."""
        phi, imp = self.mshap["phi"][i].astype(float), self.mshap["imputed"][i]
        feats, labels = list(self.mshap["features"]), list(self.mshap["labels"])
        k = int(mri_p.argmax())
        out = {"pred": NAMES[k], "rows": [], "ad_rows": []}
        for tgt, n, key in ((k, 6, "rows"), (3, 3, "ad_rows")):
            if key == "ad_rows" and k == 3:
                break
            for j in np.argsort(-np.abs(phi[:, tgt]), kind="stable")[:n]:
                out[key].append({"feature": feats[j], "label": labels[j], "value": mri_value(feats[j], r, imp[j]),
                                 "phi": float(phi[j, tgt]), "imputed": bool(imp[j])})
        return out

    def kan_shap(self, i, pred):
        """Group Shapley values (25 KANs) for the final and residual logits of the predicted stage (P3-R2)."""
        G = self.kshap["phi_group"][:, i].astype(float)                              # (25, 4 groups, 9 targets)
        tg = list(self.kshap["targets"])
        out = {}
        for key in (f"final_{pred}", f"residual_{pred}"):
            t = tg.index(key)
            rows = []
            for g, gn in enumerate(self.kshap["groups"]):
                v = G[:, g, t]
                same = bool((v > 0).all() or (v < 0).all())
                rows.append({"group": str(gn), "mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()),
                             "consistent": same and abs(v.mean()) >= MIN_EFFECT})
            out[key] = rows
        return out


def mri_value(feat, r, imputed):
    if imputed:
        return "missing (imputed)" if feat != "BAG" else f"not used ({r.ba_status}; imputed)"
    if feat.endswith("_ICV"):
        return f"{r[f'volpct_{feat}']:.0f}th pct of CN-train"
    if feat == "ALZ":
        return f"{r.alz:.2f}"
    if feat == "BAG":
        return f"{r.ba_corr_gap:+.1f} y"
    if feat == "AGE":
        return f"{r.raw_AGE:.0f} y"
    if feat == "SEX":
        return str(r.raw_PTGENDER)
    return ""


def fmt_value(feat, v):
    if feat == "PTGENDER":
        return str(v)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if np.isnan(v):
        return "missing (imputed)"
    if feat == "CogIndex":
        return f"{v:+.2f} (z)"
    if feat in ("APOE4", "PTEDUCAT", "MMSE", "MOCA") or float(v).is_integer():
        return f"{v:.0f}"
    return f"{v:.1f}" if feat.startswith(("AGE", "CDRSB", "ADAS", "RAVLT", "FAQ")) else f"{v:.2f}"


# --- text -----------------------------------------------------------------------------
def header(data, s, category=None):
    r = s["row"]
    split = r.split
    sample = ("held-out test subject (outputs from the single frozen test run)" if split == "test" else
              "in the frozen system's training data (fitted on all 691 train+val subjects): outputs are in-sample, "
              "not performance estimates")
    return {"Subject": r.PTID, "Split": f"{split}: {sample}", "True diagnosis": r.true if isinstance(r.true, str) else
            "unknown", "MRI scan available": "yes" if r.has_mri else "no", **({"Showcase category": category}
                                                                             if category else {})}


def brain_age_lines(data, s):
    r = s["row"]
    if r.ba_status == "not available":
        why = r.ba_missing_reason if isinstance(r.ba_missing_reason, str) else "not available"
        return ["Not available: " + why + ".", "The MRI specialist imputes the gap from its training median."]
    L = [f"Age at scan {r.ba_age_at_scan:.1f} y; predicted brain age {r.ba_pred_age:.1f} y (DeepBrainNet).",
         f"Raw gap {r.ba_raw_gap:+.1f} y; corrected gap {r.ba_corr_gap:+.1f} y "
         f"(bias correction fitted on CN training subjects, α = {data.meta['brain_age']['alpha']:.2f}).",
         "Brain age is supporting imaging evidence; it adds no measured staging accuracy."]
    if r.ba_status == "unreliable":
        L.insert(0, "UNRELIABLE: this scan fails the pre-registered reliability rule (no zero-intensity voxels in "
                    "the brain mask, or > 10% saturated voxels). The gap is not used by the MRI specialist.")
    if r.ba_extreme:
        L.append(f"Extreme value: |corrected gap| ≥ {data.meta['brain_age']['extreme_gap_years']:.0f} y. The bias "
                 "correction divides by α and amplifies deviations; treat this gap with caution.")
    return L


def flag_lines(data, s):
    r = s["row"]
    th = data.meta["discordance"]
    rule = (f"Rule (thresholds from train only): clinical specialist predicts AD with ALZ < {th['P10']:.3f} "
            f"(P10 of train AD), or predicts CN/EMCI with ALZ > {th['P90']:.3f} (P90).")
    if not r.discordance_evaluable:
        return ["Discordance flag (exploratory): not evaluable, no scan.", rule]
    if r.discordant:
        return [f"Discordance flag (exploratory): FLAGGED, {r.discordance_branch} "
                f"(clinical {s['clin_pred']}, ALZ {r.alz:.3f}).", rule,
                "Exploratory and descriptive only; it is not validated as a predictor of error."]
    return [f"Discordance flag (exploratory): not flagged (clinical {s['clin_pred']}, ALZ {r.alz:.3f}).", rule]


def mri_lines(s):
    r = s["row"]
    if not r.has_mri:
        return ["No MRI scan. The MRI stream is treated as missing: the MRI specialist gives no output, the fusion "
                "layer receives neutral MRI inputs (training class-prior logits, has_mri = 0; it was trained with "
                "15% modality dropout), and the CNN attribution and brain age are not available."] + (
            ["The FreeSurfer volumes shown come from the ADNI tables; they are displayed for reference but are not "
             "used, because the MRI specialist runs only for subjects with a scan."]
            if any(not np.isnan(r[k]) for k in r.index if k.startswith("vol_")) else [])
    m = s["mri"]
    return [f"MRI specialist (volumes/ICV, ALZ, brain-age gap, age, sex): {NAMES[int(m.argmax())]} "
            f"({m.max():.0%}); P(AD) {m[3]:.0%}. CNN AD probability (ALZ) {r.alz:.2f}.",
            "The MRI specialist detects AD well (AD-vs-rest AUC 0.87 on test) but is weak at staging MCI. Its AD "
            "detection comes mostly from the six volume ratios (train CV: volumes alone AUC 0.867; ALZ alone 0.764)."]


def validation(data, key):
    e = data.eval[key]
    if e["overall"] == "VALIDATED":
        return "Validation: VALIDATED (all pre-registered checks passed)"
    return "Validation: WEAK. Weak or failed checks: " + "; ".join(e["weak_or_failed"])


def clinical_validation_lines(data):
    """P5 status plus the post-hoc CDR-SB check (reported next to, not instead of, the pre-registered criterion)."""
    L = [validation(data, "clinical_shap")]
    ph = data.posthoc_cdrsb
    if ph:
        pre = ph["pre_registered_criterion"]
        L.append(f"CDR-SB plausibility: pre-registered sign criterion {pre['train']:.2f} of training subjects "
                 f"(FAIL, unchanged). Post-hoc (not pre-registered): the CDR-SB SHAP value toward AD rises with "
                 f"CDR-SB, Spearman {ph['train']['spearman']:.2f} [{ph['train']['ci'][0]:.2f}, "
                 f"{ph['train']['ci'][1]:.2f}] on training subjects, {ph['val']['spearman']:.2f} on validation.")
    return L


def mri_shap_lines(s):
    ms = s["mri_shap"]
    if ms is None:
        return []
    L = [f"Drivers of the MRI specialist's {ms['pred']} score (SHAP on the centred class score, log-odds units; "
         "background = training subjects with a scan):"]
    L += [f"{x['label']} = {x['value']}: {'towards' if x['phi'] > 0 else 'away from'} {ms['pred']} ({abs(x['phi']):.2f})"
          for x in ms["rows"]]
    if ms["ad_rows"]:
        L.append("For AD: " + "; ".join(f"{x['label']} {'towards' if x['phi'] > 0 else 'away from'} AD "
                                         f"({abs(x['phi']):.2f})" for x in ms["ad_rows"]) + ".")
    return L


def cell_mm(size):
    lo, hi = min(VOXEL_MM) * 96 / size, max(VOXEL_MM) * 96 / size
    return f"{size}³ grid (cells ≈ {lo:.0f}–{hi:.0f} mm)"


def cnn_region_text(data, s):
    """No region names (preregistration.md, Addendum 1, 2026-10-05: made after the results; removes claims).
    The naming rule P1-R3 and its region tables are kept in analysis/explainability/report.md."""
    st = data.rules["cnn_stage"]
    return [f"CNN attribution: LayerCAM of the ALZ CNN at {st['stage']} ({cell_mm(st['map_size'])}), "
            f"{s['cam']['n_models']} models; stage chosen by the pre-registered deletion rule.",
            "NOT ANATOMICALLY RELIABLE: the map changes little when the CNN's trained weights are randomised, "
            "occluding its most-attributed region does not lower the AD score more than other regions, and the "
            "maps barely differ between AD and CN subjects. No brain region is named."]


CAM_NOTE = ("Overlay: the whole map on brain voxels (brighter = higher); fixed slices through the template "
            "hippocampus. The regular dot pattern is a stride artefact of gradient-based maps at this layer, "
            "not anatomy.")
GROUP_LABEL = {"clinical logits": "clinical specialist", "MRI logits + has_mri": "MRI specialist",
               "age": "age", "CDRSB + CogIndex + MMSE": "CDR-SB / CogIndex / MMSE"}


def kan_lines(data, s):
    L = [f"Final = clinical log-probabilities + KAN correction (K2 residual, mean of 25 KANs). Total correction "
         f"to each logit: " + ", ".join(f"{n} {v:+.2f}" for n, v in zip(NAMES, s["residual"])) + "."]
    pred = s["pred"]
    for key, what in ((f"final_{pred}", f"final {pred} logit"), (f"residual_{pred}", f"KAN correction to {pred}")):
        rows = s["kan_shap"][key]
        ok = [x for x in rows if x["consistent"]]
        bad = [GROUP_LABEL[x["group"]] for x in rows if not x["consistent"]]
        L.append(f"Contributions to the {what} (Shapley, centred, mean of 25 KANs): "
                 + (", ".join(f"{GROUP_LABEL[x['group']]} {x['mean']:+.2f}" for x in ok) if ok else "none consistent")
                 + "." + (f" Not shown (sign differs across the 25 KANs or |value| < {MIN_EFFECT}): "
                          + ", ".join(bad) + "." if bad else ""))
    cells = [c for c in data.rules["kan"]["consistent_cells"]
             if c["target"] in (f"final_{pred}", f"residual_{pred}") and c["stratum"] == pred]
    if cells:
        def rep(c):
            v = c.get("val_trainonly_replication")
            return "" if v in (None, "replicated") else f"; {v} on held-out validation subjects"
        L.append(f"In training subjects whose true stage is {pred}, consistent in all 25 KANs: " + "; ".join(
            f"{GROUP_LABEL[c['group']]} {'raises' if c['median_effect'] > 0 else 'lowers'} the "
            f"{'final' if c['target'].startswith('final') else 'correction to the'} {pred} logit "
            f"(median {c['median_effect']:+.2f}{rep(c)})" for c in cells) + ".")
    if not s["row"].has_mri:
        L.append("No scan: the MRI inputs are the neutral missing-scan values.")
    L.append("Shapley values explain each KAN's logits; the probabilities shown average 25 softmaxes.")
    return L


def stage_change_lines(data, s):
    sc = data.rules["stage_change"]
    clin, fin = s["clin_pred"], s["pred"]
    L = []
    if clin != fin:
        L.append(f"Fusion changed the stage: clinical specialist {clin} → final {fin}.")
        t = [x for x in sc["A_transitions_train_oof"] if x["from"] == clin and x["to"] == fin]
        n = t[0]["n"] if t else 0
        if n >= sc["min_transition_n"]:
            x = t[0]
            both = x["both_wrong_closer"] + x["both_wrong_farther"] + x["both_wrong_same"]
            L.append(f"On training subjects (out-of-fold), the change {clin} → {fin} occurred {n} times: "
                     f"{x['corrected']} corrected, {x['broken']} made wrong, {both} wrong before and after.")
        else:
            L.append(f"Too few training cases ({n}) of the change {clin} → {fin} to judge it.")
    else:
        L.append(f"Fusion did not change the clinical specialist's stage ({clin}).")
    a = sc["A_train_oof"]
    L.append(f"All fusion stage changes on training subjects (out-of-fold): {a['fired']} of {a['n']}, "
             f"net accuracy change {a['net_accuracy_change']:+.1%} [{a['net_ci_lo']:+.1%}, {a['net_ci_hi']:+.1%}].")
    if s["row"].has_mri:
        m = s["mri_masked_pred"]
        L.append(f"MRI changed the stage: without the MRI inputs (scan treated as missing) the system would predict "
                 f"{m} instead of {fin}." if m != fin else
                 "The MRI inputs do not change the predicted stage (same stage with the scan treated as missing).")
        b = sc.get("B_heldout_val")
        if b:
            L.append(f"On held-out validation subjects (train-only KANs), MRI changed the stage in {b['fired']} of "
                     f"{b['n']}: {b['corrected']} corrected, {b['broken']} made wrong (net "
                     f"{b['net_accuracy_change']:+.1%} [{b['net_ci_lo']:+.1%}, {b['net_ci_hi']:+.1%}]).")
        else:
            L.append("The 'MRI changed the stage' line is descriptive (in-sample frozen models).")
    return L


def footer(data):
    v = data.meta["versions"]
    return [f"Frozen system fusion-v3, freeze commit {data.meta['freeze_commit']}. Clinical specialist: CatBoost "
            f"{v.get('catboost', '')} (265 trees, top-12 by SHAP, 5 seeds). MRI specialist: multinomial logistic "
            f"regression C = 0.1 (scikit-learn {v.get('scikit-learn', '')}). Fusion: KAN K2 residual, config 32, "
            "25 models (pykan 0.2.8). ALZ: 3D ResNet CNN, 15 cross-fit checkpoints. Brain age: DeepBrainNet (VGG16). "
            f"Explanations: analysis/explainability (pre-registered, commit {PREREG_COMMIT}); report "
            "analysis/explainability/report.md.",
            NOT_CLINICAL]


# --- panels ---------------------------------------------------------------------------
def _bars(ax, p, title, highlight, color=BLUE):
    y = np.arange(4)[::-1]
    ax.barh(y, p, color=[color if n == highlight else GREY for n in NAMES], height=0.62)
    for yy, v in zip(y, p):
        ax.text(min(v, 1) + 0.02, yy, f"{v:.0%}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y, NAMES)
    ax.set_xlim(0, 1.15)
    ax.set_xticks([0, 0.5, 1], ["0", "50%", "100%"])
    ax.set_title(title, loc="left")


def panel_stage(ax, s):
    _bars(ax, s["p"], f"Predicted stage: {s['pred']}", s["pred"])


def panel_severity(ax, data, s):
    means = data.meta["severity_stage_means_cv"]
    ax.set_xlim(-0.05, 3.05)
    ax.set_ylim(0, 1)
    ax.imshow(np.linspace(0, 1, 256)[None], extent=(0, 3, 0.35, 0.6), aspect="auto", cmap="Blues", vmin=-0.3,
              vmax=1.1)
    for n, m in zip(NAMES, means):
        ax.plot([m, m], [0.3, 0.65], color=INK2, lw=1)
        ax.text(m, 0.72, n, ha="center", fontsize=7.5, color=INK2)
    ax.plot([s["severity"]], [0.2], marker="^", ms=11, color=INK)
    ax.text(s["severity"], 0.02, f"{s['severity']:.2f}", ha="center", fontsize=8.5, color=INK, weight="bold")
    ax.set_yticks([])
    ax.set_xticks([0, 1, 2, 3])
    ax.spines["left"].set_visible(False)
    ax.set_title(f"Severity index {s['severity']:.2f} (0–3)   ·   AD vs rest {s['add']:.0%}", loc="left")
    ax.text(3.05, 0.9, "ticks: stage means (train CV)", ha="right", fontsize=7, color=MUTED)


def panel_shap(ax, s):
    rows = s["shap"][::-1]
    if not rows:
        ax.axis("off")
        return
    v = [r["shap"] for r in rows]
    ax.barh(range(len(rows)), v, color=[BLUE if x > 0 else ORANGE for x in v], height=0.6)
    ax.set_yticks(range(len(rows)), [f"{r['label'].split(' (')[0]} = {r['value']}" for r in rows], fontsize=7.8)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel(f"SHAP (log-odds) for {s['clin_pred']}: blue towards, orange away")
    ax.set_title(f"Clinical evidence: clinical specialist predicts {s['clin_pred']}\n"
                 "(cognitive, functional and demographic inputs)", loc="left")


def panel_mri(ax, s):
    if not s["row"].has_mri:
        ax.axis("off")
        ax.text(0, 0.5, "MRI specialist: no output (no scan)", fontsize=9, color=INK2)
        return
    _bars(ax, s["mri"], f"MRI specialist: {NAMES[int(s['mri'].argmax())]} · P(AD) {s['mri'][3]:.0%}",
          NAMES[int(s["mri"].argmax())], color=ORANGE)


def panel_volumes(ax, data, s):
    """Each region relative to its CN-train median (100%), so all six share one axis."""
    ref = data.meta["volume_reference"]
    keys = list(ref)
    r = s["row"]
    for k, key in enumerate(keys[::-1]):
        q = {a: b / ref[key]["pct"]["p50"] * 100 for a, b in ref[key]["pct"].items()}
        ax.plot([q["p5"], q["p95"]], [k, k], color=GREY, lw=2, solid_capstyle="round")
        ax.plot([q["p25"], q["p75"]], [k, k], color="#8fb7e8", lw=7, solid_capstyle="butt")
        ax.plot([100], [k], marker="|", ms=10, color=INK2)
        v = r[f"vol_{key}"]
        if np.isnan(v):
            ax.text(1.02, k, "missing", transform=ax.get_yaxis_transform(), fontsize=7.5, va="center", color=MUTED)
        else:
            x = v / ref[key]["pct"]["p50"] * 100
            ax.plot([min(max(x, 42), 265)], [k], marker="o", ms=7, color=INK, markeredgecolor="white",
                    markeredgewidth=1.5)
            ax.text(1.02, k, f"{r[f'volpct_{key}']:.0f}th pct", transform=ax.get_yaxis_transform(), fontsize=7.5,
                    va="center", color=INK2)
    ax.set_yticks(range(len(keys)), [ref[k]["label"] for k in keys[::-1]], fontsize=7.8)
    ax.set_xlim(40, 270)
    ax.set_xlabel("volume / ICV, % of CN-train median")
    ax.set_title("Volumes / ICV vs CN-train reference\n(grey P5–P95, blue P25–P75; dot = subject)", loc="left")


def panel_cam(axes, s):
    gc = s["cam"]
    top = gc["cam"].max() / 255 if gc is not None else 1.0
    for ax, t in zip(axes, ("axial", "coronal", "sagittal (left)")):
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        if gc is None:
            ax.text(0.5, 0.5, "no CNN volume", ha="center", va="center", transform=ax.transAxes, color=MUTED)
            continue
        k = ("axial", "coronal", "sagittal (left)").index(t)
        # stored arrays: axial (P, R), coronal (S, R), sagittal (S, P); vol96 axes run toward S, P, R
        origin = "upper" if t == "axial" else "lower"                       # anterior up / superior up
        ax.imshow(gc["bg"][k], cmap="gray", origin=origin)
        cam = gc["cam"][k].astype(float) / 255
        show = gc["bg"][k] > 12                                             # brain voxels; whole map, no threshold
        ax.imshow(np.ma.masked_where(~show, cam), cmap="inferno", origin=origin, alpha=0.55, vmin=0, vmax=top)
        ax.set_title(t, fontsize=8, weight="normal", color=INK2)


def panel_mri_shap(ax, s):
    ms = s["mri_shap"]
    if ms is None:
        ax.axis("off")
        return
    rows = ms["rows"][::-1]
    v = [r["phi"] for r in rows]
    ax.barh(range(len(rows)), v, color=[BLUE if x > 0 else ORANGE for x in v], height=0.6)
    ax.set_yticks(range(len(rows)), [f"{r['label']} = {r['value']}" for r in rows], fontsize=7.6)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel(f"SHAP (log-odds) for {ms['pred']}: blue towards, orange away", fontsize=8)
    ax.set_title(f"MRI specialist drivers (predicts {ms['pred']})", loc="left")


def panel_kan(ax, s):
    clin = np.exp(s["base"])
    clin = clin / clin.sum()
    x = np.arange(4)
    ax.bar(x - 0.19, clin, 0.36, color=GREY, label="clinical specialist")
    ax.bar(x + 0.19, s["p"], 0.36, color=BLUE, label="final (with KAN correction)")
    ax.set_xticks(x, NAMES)
    ax.set_ylim(0, 1.35)
    ax.set_yticks([0, 0.5, 1], ["0", "50%", "100%"])
    ax.legend(frameon=False, fontsize=7.5, loc="upper right", ncol=2)
    ax.set_title("How the fusion changes the prediction", loc="left")


def panel_kan_shapley(ax, s):
    pred = s["pred"]
    rows_f, rows_r = s["kan_shap"][f"final_{pred}"], s["kan_shap"][f"residual_{pred}"]
    labels = [GROUP_LABEL[x["group"]] for x in rows_f]
    y = np.arange(len(labels))[::-1]
    for off, rows, col, nm in ((0.18, rows_f, BLUE, f"final {pred} logit"), (-0.18, rows_r, ORANGE,
                                                                          f"KAN correction to {pred}")):
        for yy, x in zip(y, rows):
            if x["consistent"]:
                ax.barh(yy + off, x["mean"], height=0.34, color=col)
            else:
                ax.text(0, yy + off, "  inconsistent across KANs", va="center", fontsize=6.8, color=MUTED)
        ax.barh([], [], color=col, label=nm)
    ax.set_yticks(y, labels, fontsize=7.8)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2)
    ax.set_xlabel("Shapley value (centred logit), mean of 25 KANs", fontsize=8)
    ax.set_title(f"What moves the {pred} logit\n(only where all 25 KANs agree)", loc="left", fontsize=8.8)


# --- PNG ------------------------------------------------------------------------------
def _textbox(ax, title, lines, size=8.2, width=118):
    """Wrapped text from the top of ax; line spacing in points so it never overlaps."""
    import textwrap
    ax.axis("off")
    y = 0.0
    if title:
        ax.annotate(title, (0, 1), xycoords="axes fraction", xytext=(0, y), textcoords="offset points",
                    fontsize=9.5, weight="bold", color=INK, va="top")
        y -= 9.5 * 1.7
    for ln in lines:
        w = textwrap.fill(ln, width) if ln else ""
        ax.annotate(w, (0, 1), xycoords="axes fraction", xytext=(0, y), textcoords="offset points",
                    fontsize=size, color=INK2, va="top")
        y -= size * 1.32 * (w.count("\n") + 1) + size * 0.55


def render_png(data, ptid, path, category=None):
    s = data.subject(ptid)
    fig = plt.figure(figsize=(11, 30), dpi=110)
    fig.patch.set_facecolor("white")
    gs = GridSpec(13, 6, figure=fig, height_ratios=[0.8, 1.5, 0.02, 1.9, 1.6, 1.7, 1.5, 0.85, 0.95, 1.8, 1.5, 1.2, 1.2],
                  hspace=0.8, wspace=0.6, left=0.17, right=0.95, top=0.98, bottom=0.015)
    h = header(data, s, category)
    ax = fig.add_subplot(gs[0, :])
    ax.axis("off")
    ax.text(0, 1, f"Assessment card · {h['Subject']}", fontsize=15, weight="bold", color=INK, va="top",
            transform=ax.transAxes)
    rest = [f"{k}: {v}" for k, v in h.items() if k != "Subject"]
    for k, ln in enumerate(rest):
        ax.annotate(ln, (0, 1), xycoords="axes fraction", xytext=(0, -26 - 13 * k), textcoords="offset points",
                    fontsize=8.5, color=INK2, va="top")
    panel_stage(fig.add_subplot(gs[1, :3]), s)
    panel_severity(fig.add_subplot(gs[1, 3:]), data, s)
    ax = fig.add_subplot(gs[2, :])
    ax.axis("off")
    panel_shap(fig.add_subplot(gs[3, 1:4]), s)
    _textbox(fig.add_subplot(gs[3, 4:]), None, [f"{r['label']}: {r['value']}; "
                                               f"{'towards' if r['shap'] > 0 else 'away from'} {s['clin_pred']} "
                                               f"({abs(r['shap']):.2f})" for r in s["shap"]]
             + [""] + clinical_validation_lines(data), size=7.0, width=44)
    panel_mri(fig.add_subplot(gs[4, :2]), s)
    if s["mri_shap"]:
        panel_mri_shap(fig.add_subplot(gs[4, 3:]), s)
    panel_volumes(fig.add_subplot(gs[5, 1:3]), data, s)
    _textbox(fig.add_subplot(gs[5, 3:]), "MRI evidence", mri_lines(s) + (
        ["", validation(data, "mri_shap")] if s["mri_shap"] else []), size=7.3, width=64)
    axes = [fig.add_subplot(gs[6, k]) for k in range(3)]
    panel_cam(axes, s)
    _textbox(fig.add_subplot(gs[6, 3:]), "CNN attribution", (cnn_region_text(data, s) + [
        CAM_NOTE, validation(data, "cnn")]) if s["cam"] else ["Not available (no CNN volume)."],
        size=7.0, width=66)
    _textbox(fig.add_subplot(gs[7, :]), "Brain age", brain_age_lines(data, s), size=7.8)
    _textbox(fig.add_subplot(gs[8, :]), "Stage changes", stage_change_lines(data, s), size=7.4, width=150)
    panel_kan(fig.add_subplot(gs[9, :3]), s)
    panel_kan_shapley(fig.add_subplot(gs[9, 4:]), s)
    _textbox(fig.add_subplot(gs[10, :]), "How the fusion changes the prediction",
             kan_lines(data, s) + [validation(data, "kan")], size=7.3, width=150)
    _textbox(fig.add_subplot(gs[11:, :]), "Flags", flag_lines(data, s) + [""] + footer(data), size=7.4, width=150)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# --- HTML -----------------------------------------------------------------------------
def _img(draw, size):
    fig, ax = plt.subplots(figsize=size, dpi=110)
    draw(ax)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    plt.close(fig)
    return f'<img alt="" src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def _img_multi(draw, n, size):
    fig, axes = plt.subplots(1, n, figsize=size, dpi=110)
    draw(list(np.atleast_1d(axes)))
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    plt.close(fig)
    return f'<img alt="" src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


CSS = """
:root{--bg:#f6f5f2;--card:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8984;--line:#e4e3df;--warn:#b3261e;--accent:#2a78d6}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#121211;--card:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#9a998f;--line:#33332f;--warn:#ff8a80;--accent:#3987e5}}
:root[data-theme="dark"]{--bg:#121211;--card:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#9a998f;--line:#33332f;--warn:#ff8a80;--accent:#3987e5}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:980px;margin:0 auto;padding:16px}
h1{font-size:22px;margin:8px 0 4px} h2{font-size:15px;margin:0 0 8px;color:var(--ink)}
section{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:12px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px;align-items:start}
img{max-width:100%;height:auto;border-radius:6px;background:#fff}
table{border-collapse:collapse;width:100%;font-size:13px} td,th{padding:4px 6px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{color:var(--ink2);font-weight:600} p{margin:4px 0;color:var(--ink2)} .warn{color:var(--warn);font-weight:600}
.ok{color:var(--accent);font-weight:600} h3{font-size:13px;margin:12px 0 6px;color:var(--ink)}
.kv td:first-child{color:var(--ink2);width:180px} .note{font-size:12px;color:var(--muted)}
footer{font-size:12px;color:var(--muted);padding:8px 2px 24px}
"""


def chip(data, key):
    return "ok" if data.eval[key]["overall"] == "VALIDATED" else "warn"


def render_html(data, ptid, path, category=None):
    s = data.subject(ptid)
    e = html.escape
    h = header(data, s, category)
    P = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
         f"content='width=device-width,initial-scale=1'><title>Assessment card {e(ptid)}</title><style>{CSS}</style>"
         f"</head><body><main><h1>Assessment card · {e(ptid)}</h1>",
         "<section><h2>1. Subject</h2><table class='kv'>" +
         "".join(f"<tr><td>{e(k)}</td><td>{e(str(v))}</td></tr>" for k, v in h.items()) + "</table></section>",
         "<section><h2>2. Outputs</h2><div class='grid'>" + _img(lambda ax: panel_stage(ax, s), (4.4, 2.3)) +
         _img(lambda ax: panel_severity(ax, data, s), (4.6, 1.9)) + "</div>"
         f"<p>Predicted stage <b>{s['pred']}</b> ({s['p'].max():.0%}); AD-vs-rest probability (ADD head) "
         f"<b>{s['add']:.0%}</b>; severity index <b>{s['severity']:.2f}</b> on 0–3 (Σ k·p(stage k)).</p></section>"]
    rows = "".join(f"<tr><td>{e(r['label'])}</td><td>{e(r['value'])}</td><td>"
                   f"{'towards' if r['shap'] > 0 else 'away from'} {s['clin_pred']}</td><td>{abs(r['shap']):.2f}</td></tr>"
                   for r in s["shap"])
    P.append("<section><h2>3. Clinical evidence</h2><p>The clinical specialist predicts <b>"
             f"{s['clin_pred']}</b> ({s['clin'].max():.0%}) from cognitive, functional and demographic inputs. "
             "Top contributions (SHAP, log-odds, mean of 5 seeds):</p><div class='grid'>" +
             _img(lambda ax: panel_shap(ax, s), (5.2, 2.6)) +
             f"<table><tr><th>Feature</th><th>Value</th><th>Direction</th><th>Magnitude</th></tr>{rows}</table>"
             "</div>" + "".join(f"<p class='{chip(data, 'clinical_shap') if k == 0 else 'note'}'>{e(x)}</p>"
                               for k, x in enumerate(clinical_validation_lines(data))) + "</section>")
    mri = "".join(f"<p>{e(x)}</p>" for x in mri_lines(s))
    imgs = "<div class='grid'>" + (_img(lambda ax: panel_mri(ax, s), (4.4, 2.3)) if s["row"].has_mri else "")
    if s["mri_shap"]:
        ml = mri_shap_lines(s)
        imgs += _img(lambda ax: panel_mri_shap(ax, s), (5.4, 2.6)) + "</div>" + \
            "".join(f"<p class='note'>{e(x)}</p>" for x in ml) + \
            f"<p class='{chip(data, 'mri_shap')}'>{e(validation(data, 'mri_shap'))}</p><div class='grid'>"
    imgs += _img(lambda ax: panel_volumes(ax, data, s), (5.4, 2.6)) + "</div>"
    if s["row"].has_mri and s["cam"]:
        ct = cnn_region_text(data, s)
        imgs += ("<h3>CNN attribution (not anatomically reliable)</h3>" +
                 _img_multi(lambda axs: panel_cam(axs, s), 3, (7.5, 2.6)) +
                 "".join(f"<p>{e(x)}</p>" for x in ct) +
                 f"<p class='note'>{e(CAM_NOTE)} Subject's right on the right.</p>"
                 f"<p class='{chip(data, 'cnn')}'>{e(validation(data, 'cnn'))}</p>")
    P.append(f"<section><h2>4. MRI evidence</h2>{mri}{imgs}</section>")
    ba = "".join(f"<p class='{'warn' if x.startswith(('UNRELIABLE', 'Extreme')) else ''}'>{e(x)}</p>"
                 for x in brain_age_lines(data, s))
    P.append(f"<section><h2>5. Brain age</h2>{ba}</section>")
    kl = kan_lines(data, s)
    sc = stage_change_lines(data, s)
    P.append("<section><h2>6. How the fusion changes the prediction</h2><div class='grid'>" +
             _img(lambda ax: panel_kan(ax, s), (4.6, 2.4)) + _img(lambda ax: panel_kan_shapley(ax, s), (5.2, 2.6)) +
             "</div>" + "".join(f"<p>{e(x)}</p>" for x in kl[:-1]) + f"<p class='note'>{e(kl[-1])}</p>"
             f"<p class='{chip(data, 'kan')}'>{e(validation(data, 'kan'))}</p>"
             "<h3>Stage changes</h3>" + "".join(f"<p>{e(x)}</p>" for x in sc) + "</section>")
    fl = flag_lines(data, s)
    P.append(f"<section><h2>7. Flags</h2><p class='{'warn' if s['row'].discordant else ''}'>{e(fl[0])}</p>" +
             "".join(f"<p class='note'>{e(x)}</p>" for x in fl[1:]) + "</section>")
    ft = footer(data)
    P.append(f"<footer><p>{e(ft[0])}</p><p><b>{e(ft[1])}</b></p></footer></main></body></html>")
    Path(path).write_text("\n".join(P), encoding="utf-8")
    return path


def render_card(data, ptid, out_dir, category=None):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"card_{ptid}")
    return render_html(data, ptid, base + ".html", category), render_png(data, ptid, base + ".png", category)
