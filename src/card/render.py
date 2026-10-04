"""Assessment card renderer (numpy, pandas, matplotlib only).

Reads the precomputed per-subject outputs in demo/card_data/ (written by
build_card_data.py, build_kan_terms.py and build_gradcam.py from the frozen
fusion-v3 artefacts) and renders one subject's card as HTML and PNG. It never
fits or runs a model, so any train, val or test subject renders in about a
second.

    from card import CardData, render_card
    data = CardData()                      # finds demo/card_data from this file's location
    render_card(data, "002_S_4229", "demo/output")
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
    def __init__(self, data_dir=None):
        d = Path(data_dir) if data_dir else DATA_DIR
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
        self.gc = None
        if (d / "gradcam.npz").exists():
            g = np.load(d / "gradcam.npz", allow_pickle=True)
            self.gc = {key: g[key] for key in g.files}
            self.gc_index = {p: i for i, p in enumerate(self.gc["ptid"])}
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
        shown = [e for e in self.edges["edges"] if e["shown"]]
        edge_terms = [{**e, "x": float(self.kan["x_raw"][i, e["input_index"]]),
                       "contrib": float(self.kan["edge"][i, e["output_index"], e["input_index"]])} for e in shown]
        gc = None
        if self.gc is not None and ptid in self.gc_index:
            j = self.gc_index[ptid]
            gc = {"bg": self.gc["bg"][j], "cam": self.gc["cam"][j], "n_models": int(self.gc["n_models"][j])}
        return {"row": r, "p": p, "pred": NAMES[int(p.argmax())], "add": float(self.kan["add"][i]),
                "severity": float(p @ np.arange(4)), "clin": clin, "clin_pred": NAMES[kc], "shap": shap_rows,
                "mri": np.array([r[f"mri_p_{n}"] for n in NAMES]), "residual": self.kan["residual"][i].astype(float),
                "base": self.kan["base"][i].astype(float), "edges": edge_terms, "gradcam": gc}


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
                "15% modality dropout), and Grad-CAM and brain age are not available."] + (
            ["The FreeSurfer volumes shown come from the ADNI tables; they are displayed for reference but are not "
             "used, because the MRI specialist runs only for subjects with a scan."]
            if any(not np.isnan(r[k]) for k in r.index if k.startswith("vol_")) else [])
    m = s["mri"]
    return [f"MRI specialist (volumes/ICV, ALZ, brain-age gap, age, sex): {NAMES[int(m.argmax())]} "
            f"({m.max():.0%}); P(AD) {m[3]:.0%}. CNN AD probability (ALZ) {r.alz:.2f}.",
            "The MRI specialist detects AD well (AD-vs-rest AUC 0.87 on test) but is weak at staging MCI."]


def edge_text(e):
    a, b = e["input"].replace("mri_logit_", ""), e["output"]
    direction = "decreasing" if "decreasing" in e["verdict_frozen"] else "increasing"
    return f"MRI {a} logit → {b} logit ({direction} in all 25 + 25 models)"


def kan_lines(s):
    shown = sum(e["contrib"] for e in s["edges"]) if s["edges"] else 0
    L = [f"Final = clinical log-probabilities + KAN correction (K2 residual, mean of 25 KANs). Total correction "
         f"to each logit: " + ", ".join(f"{n} {v:+.2f}" for n, v in zip(NAMES, s["residual"])) + "."]
    for e in s["edges"]:
        L.append(f"{edge_text(e)}: input {e['x']:+.2f}, contribution {e['contrib']:+.2f} to the {e['output']} logit.")
    L.append("Only MRI edges monotone in the same direction in all 25 val-stage and all 25 frozen KANs are shown; "
             "the rest of the correction comes from the other edges (age, CDR-SB, CogIndex, MMSE, clinical logits, "
             "and MRI edges without a consistent shape, including MRI AD → AD). MRI logits are centred and "
             "sum to zero, so their edges are read jointly.")
    if not s["row"].has_mri:
        L.append("No scan: MRI inputs are the neutral values, so the MRI edges contribute their neutral-point value.")
    del shown
    return L


def footer(data):
    v = data.meta["versions"]
    return [f"Frozen system fusion-v3, freeze commit {data.meta['freeze_commit']}. Clinical specialist: CatBoost "
            f"{v.get('catboost', '')} (265 trees, top-12 by SHAP, 5 seeds). MRI specialist: multinomial logistic "
            f"regression C = 0.1 (scikit-learn {v.get('scikit-learn', '')}). Fusion: KAN K2 residual, config 32, "
            "25 models (pykan 0.2.8). ALZ: 3D ResNet CNN, 15 cross-fit checkpoints. Brain age: DeepBrainNet (VGG16).",
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


def panel_gradcam(axes, s):
    gc = s["gradcam"]
    top = gc["cam"].max() / 255 if gc is not None else 1.0
    for ax, t in zip(axes, ("axial", "coronal", "sagittal")):
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        if gc is None:
            ax.text(0.5, 0.5, "no CNN volume", ha="center", va="center", transform=ax.transAxes, color=MUTED)
            continue
        k = ("axial", "coronal", "sagittal").index(t)
        # stored arrays: axial (P, R), coronal (S, R), sagittal (S, P); vol96 axes run toward S, P, R
        origin = "upper" if t == "axial" else "lower"                       # anterior up / superior up
        ax.imshow(gc["bg"][k], cmap="gray", origin=origin)
        cam = gc["cam"][k].astype(float) / 255
        show = (cam >= 0.6 * top) & (gc["bg"][k] > 12)                      # brain voxels, upper 40% of this map
        ax.imshow(np.ma.masked_where(~show, cam), cmap="autumn_r", origin=origin, alpha=0.5, vmin=0.6 * top,
                  vmax=top)
        ax.set_title(t, fontsize=8, weight="normal", color=INK2)


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


def panel_edge(ax, e):
    c = e["curve"]
    ax.fill_between(c["x"], c["y_min"], c["y_max"], color="#cfe0f5", lw=0, label="range over 25 KANs")
    ax.plot(c["x"], c["y_mean"], color=BLUE, lw=2, label="mean")
    ax.plot([e["x"]], [e["contrib"]], marker="o", ms=8, color=INK, markeredgecolor="white", markeredgewidth=1.5)
    ax.axhline(0, color=GRID, lw=0.8)
    a = e["input"].replace("mri_logit_", "")
    ax.set_xlabel(f"MRI {a} logit (centred)", fontsize=8)
    ax.set_ylabel(f"→ {e['output']} logit", fontsize=8)
    ax.set_title(f"MRI {a} → {e['output']}: {e['contrib']:+.2f}", loc="left", fontsize=8.5)


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
    fig = plt.figure(figsize=(11, 21.5), dpi=110)
    fig.patch.set_facecolor("white")
    gs = GridSpec(10, 6, figure=fig, height_ratios=[0.8, 1.5, 0.02, 1.9, 1.6, 1.4, 0.9, 1.8, 1.5, 1.45],
                  hspace=0.75, wspace=0.6, left=0.17, right=0.95, top=0.975, bottom=0.02)
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
                                               f"({abs(r['shap']):.2f})" for r in s["shap"]], size=7.4, width=42)
    panel_mri(fig.add_subplot(gs[4, :2]), s)
    panel_volumes(fig.add_subplot(gs[4, 3:]), data, s)
    axes = [fig.add_subplot(gs[5, k]) for k in range(3)]
    panel_gradcam(axes, s)
    _textbox(fig.add_subplot(gs[5, 3:]), "MRI evidence", mri_lines(s) + (
        [f"Grad-CAM of the ALZ CNN ({s['gradcam']['n_models']} models), central slices (right on the right; overlay = brain voxels in the top 40% of this subject's map, yellow = highest). The map is "
         "coarse (3×3×3 before upsampling) and cannot localise a specific structure such as the hippocampus; across "
         "the cohort the attribution is mainly periventricular / ventricular."] if s["gradcam"] else []), size=7.6, width=62)
    _textbox(fig.add_subplot(gs[6, :]), "Brain age", brain_age_lines(data, s), size=7.8)
    panel_kan(fig.add_subplot(gs[7, :3]), s)
    _textbox(fig.add_subplot(gs[7, 3:]), None, kan_lines(s), size=7.2, width=62)
    for k, e in enumerate(s["edges"][:2]):
        panel_edge(fig.add_subplot(gs[8, 3 * k:3 * k + 2]), e)
    _textbox(fig.add_subplot(gs[9, :]), "Flags", flag_lines(data, s) + [""] + footer(data), size=7.4, width=150)
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
.kv td:first-child{color:var(--ink2);width:180px} .note{font-size:12px;color:var(--muted)}
footer{font-size:12px;color:var(--muted);padding:8px 2px 24px}
"""


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
             "</div></section>")
    mri = "".join(f"<p>{e(x)}</p>" for x in mri_lines(s))
    imgs = ("<div class='grid'>" + (_img(lambda ax: panel_mri(ax, s), (4.4, 2.3)) if s["row"].has_mri else "") +
            _img(lambda ax: panel_volumes(ax, data, s), (5.4, 2.6)) + "</div>")
    if s["row"].has_mri:
        if s["gradcam"]:
            imgs += (_img_multi(lambda axs: panel_gradcam(axs, s), 3, (7.5, 2.6)) +
                     f"<p class='note'>Grad-CAM of the ALZ CNN ({s['gradcam']['n_models']} models), central slices (subject's right on the right; overlay = brain voxels in the top 40% of this subject's map, yellow = highest). "
                     "The map is coarse (3×3×3 before upsampling) and cannot localise a specific "
                     "structure such as the hippocampus; across the cohort the attribution is mainly periventricular / "
                     "ventricular.</p>")
    P.append(f"<section><h2>4. MRI evidence</h2>{mri}{imgs}</section>")
    ba = "".join(f"<p class='{'warn' if x.startswith(('UNRELIABLE', 'Extreme')) else ''}'>{e(x)}</p>"
                 for x in brain_age_lines(data, s))
    P.append(f"<section><h2>5. Brain age</h2>{ba}</section>")
    kl = kan_lines(s)
    edges = "".join(_img(lambda ax, ee=ee: panel_edge(ax, ee), (3.6, 2.4)) for ee in s["edges"])
    P.append("<section><h2>6. How MRI changes the prediction</h2><div class='grid'>" +
             _img(lambda ax: panel_kan(ax, s), (4.6, 2.4)) + f"<div>{''.join(f'<p>{e(x)}</p>' for x in kl[:-1])}</div>"
             f"</div><div class='grid'>{edges}</div><p class='note'>{e(kl[-1])}</p></section>")
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
