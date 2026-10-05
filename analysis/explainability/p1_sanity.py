"""P1 step 4 (LYP/.venv-mri, GPU): sanity checks of the chosen stage's LayerCAM (P1-R4).

(a) Cascading randomisation. Subjects: the first 50 train subjects with a scan by PTID, each with its 3
    out-of-fold models. On in-memory deep copies (never saved), cumulatively from the top: ALZ head, layer4,
    layer3, layer2, layer1, stem are re-initialised with the model's own initialisers (conv: Kaiming normal,
    fan_out; BN: weight 1, bias 0, running stats reset; Linear: reset_parameters), torch seed 0 per model copy.
    Similarity = Spearman of the brain-masked 96^3 subject maps (mean of its models' unit-mass maps),
    original vs randomised. An all-zero randomised map counts as similarity 0.
    PASS if the median similarity after head + layer4 < 0.5; WEAK 0.5-0.8; FAIL >= 0.8.
(b) Region deletion. Each of the 104 train AD subjects: occlude (CN-train mean) its highest-d_r region
    (chosen stage, excluding "other"); control = mean drop over the other regions (excluding "other") whose
    voxel count is within a factor 2 (else the single nearest in size). Drops are ALZ-logit drops averaged over
    the subject's 3 models. PASS if the CI of the mean paired difference > 0; WEAK if mean > 0 but the CI
    includes 0; FAIL otherwise.
Outputs: results/p1_sanity_cascade.csv, results/p1_sanity_region.csv, results/p5_rows.json["cnn_sanity"],
figures/p1_cascade.png; subject-level values in subject_level/ (gitignored).
"""
import argparse
import copy
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import p1_lib as L  # noqa: E402

CASCADE = ["alz_head", "layer4", "layer3", "layer2", "layer1", "stem"]


def reinit(module):
    for m in module.modules():
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
            m.reset_running_stats()
        elif isinstance(m, nn.Linear):
            m.reset_parameters()


def block(model, name):
    return model.alz_head if name == "alz_head" else (model.trunk.stem if name == "stem" else getattr(model.trunk, name))


def subject_map(models, vol, stage, mask):
    ups = []
    for m in models:
        maps, _ = L.cams(m, vol[None], stages=[stage], batch=1)
        ups.append(L.upsample(maps[("layercam", stage)], mask)[0])
    s = np.mean(ups, 0)
    return s / s.sum() if s.sum() > 0 else s


def spearman_masked(a, b, mask):
    x, y = a[mask], b[mask]
    if y.max() <= 0 or x.max() <= 0:
        return 0.0
    rx, ry = rankdata(x), rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def verdict_cascade(med):
    return "PASS" if med < 0.5 else ("WEAK" if med < 0.8 else "FAIL")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    if a.dry:
        C.set_dry()
    C.verify_manifest()
    t0 = time.time()
    stage = json.load(open(os.path.join(C.RESULTS, "card_rules.json")))["cnn_stage"]["stage"]
    d = L.subjects()
    mask = L.brain_mask()
    base = L.cn_train_mean()
    atl, info = L.atlas()
    regions = info["regions"]

    # (a) cascading randomisation
    tr = d[d.split == "train"].sort_values("PTID")
    cas = tr[tr.PTID.isin(C.dry_run_ptids())] if a.dry else tr.iloc[:50]
    C.assert_no_test(cas.RID)
    vols = L.load_volumes(list(cas.path))
    rows = []
    for fold in sorted(cas.fold.unique()):
        mids = [i for i, (k, _) in enumerate(L.MODELS) if k == int(fold)]
        orig = [L.get_model(m) for m in mids]
        sel = np.flatnonzero(cas.fold.to_numpy() == fold)
        o_maps = {i: subject_map(orig, vols[i], stage, mask) for i in sel}
        rand = [copy.deepcopy(m) for m in orig]                    # in-memory copies; the frozen files are untouched
        for step in CASCADE:
            for c in rand:
                if step == CASCADE[0]:
                    torch.manual_seed(0)
                reinit(block(c, step))
                c.eval()
            for i in sel:
                r_map = subject_map(rand, vols[i], stage, mask)
                rows.append({"PTID": cas.PTID.iloc[i], "step": step, "spearman": spearman_masked(o_maps[i], r_map, mask)})
        del orig, rand
        torch.cuda.empty_cache()
        print(f"cascade fold {fold} done {time.time() - t0:.0f}s", flush=True)
    cr = pd.DataFrame(rows)
    cr.to_csv(C.subj_path("p1_cascade_subject.csv"), index=False)
    ctab = []
    for step in CASCADE:
        v = cr[cr.step == step].spearman.to_numpy()
        med = float(np.median(v))
        ctab.append({"randomised_through": step, "n": len(v), "median_spearman": med,
                     "p25": float(np.percentile(v, 25)), "p75": float(np.percentile(v, 75))})
    ctab = pd.DataFrame(ctab)
    c_med = float(ctab.set_index("randomised_through").loc["layer4", "median_spearman"])
    c_ver = verdict_cascade(c_med)
    C.write_aggregate(ctab, "p1_sanity_cascade.csv", ["train"])
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(range(len(CASCADE)), ctab.median_spearman, marker="o", color="#2a78d6")
    ax.fill_between(range(len(CASCADE)), ctab.p25, ctab.p75, color="#cfe0f5")
    ax.axhline(0.5, color="#8a8984", lw=0.8, ls="--")
    ax.set_xticks(range(len(CASCADE)), [f"+{s}" for s in CASCADE], rotation=30, fontsize=8)
    ax.set_ylabel("Spearman vs original map")
    ax.set_ylim(-0.1, 1.05)
    ax.set_title(f"Cascading randomisation, LayerCAM {stage} (n = {len(cas)} train)", fontsize=9)
    fig.tight_layout()
    os.makedirs(C.FIGURES, exist_ok=True)
    fig.savefig(os.path.join(C.FIGURES, "p1_cascade.png"), dpi=110)
    plt.close(fig)
    print(ctab.round(3).to_string(index=False), flush=True)

    # (b) region deletion
    sub = pd.read_csv(C.subj_path("p1_regions_subject.csv"))
    sub = sub[(sub.method == "layercam") & (sub.stage == stage) & (sub.split == "train") & (sub.label == 3)]
    ad = d[d.PTID.isin(sub.PTID)].reset_index(drop=True)
    C.assert_no_test(ad.RID)
    named = [r for r in regions if r != "other"]
    vox = {r: info["voxels"][r] for r in named}
    vols = L.load_volumes(list(ad.path))
    rrows = []
    for fold in sorted(ad.fold.unique()):
        mids = [i for i, (k, _) in enumerate(L.MODELS) if k == int(fold)]
        models = [L.get_model(m) for m in mids]
        for i in np.flatnonzero(ad.fold.to_numpy() == fold):
            p = ad.PTID.iloc[i]
            s = sub[sub.PTID == p].iloc[0]
            top = max(named, key=lambda r: s[f"d_{r}"])
            ctrl = [r for r in named if r != top and 0.5 <= vox[r] / vox[top] <= 2]
            if not ctrl:
                ctrl = [min((r for r in named if r != top), key=lambda r: abs(np.log(vox[r] / vox[top])))]
            batch = [vols[i]] + [L.occlude(vols[i], atl == info["index"][r], base) for r in [top] + ctrl]
            drops = np.mean([(lambda lg: lg[0] - lg[1:])(L.alz_logit(m, np.stack(batch))) for m in models], 0)
            rrows.append({"PTID": p, "top_region": top, "drop_top": float(drops[0]),
                          "drop_control": float(drops[1:].mean()), "n_control": len(ctrl)})
        del models
        torch.cuda.empty_cache()
    rr = pd.DataFrame(rrows)
    rr.to_csv(C.subj_path("p1_region_deletion_subject.csv"), index=False)
    m, lo, hi = C.boot_ci((rr.drop_top - rr.drop_control).to_numpy())
    r_ver = "PASS" if lo > 0 else ("WEAK" if m > 0 else "FAIL")
    rtab = pd.DataFrame([{"n": len(rr), "mean_drop_top": rr.drop_top.mean(), "mean_drop_control": rr.drop_control.mean(),
                          "mean_difference": m, "ci_lo": lo, "ci_hi": hi, "verdict": r_ver}])
    tops = rr.top_region.value_counts().rename_axis("top_region").reset_index(name="n_subjects")
    C.write_aggregate(rtab, "p1_sanity_region.csv", ["train"])
    C.write_aggregate(tops, "p1_sanity_region_tops.csv", ["train"])
    C.update_json("p5_rows.json", "cnn_sanity", {
        "stage": stage, "cascade_median_spearman_head_layer4": c_med, "cascade_verdict": c_ver,
        "region_deletion_mean_difference": m, "region_deletion_ci": [lo, hi], "region_deletion_verdict": r_ver})
    print(rtab.round(4).to_string(index=False))
    print(f"cascade {c_ver} ({c_med:.3f}); region deletion {r_ver}; {time.time() - t0:.0f}s; "
          f"peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    C.verify_manifest()


if __name__ == "__main__":
    main()
