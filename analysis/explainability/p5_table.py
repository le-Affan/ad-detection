"""P5 collector (any env; json / pandas): the evaluation table, with the pre-registered tiers.

Reads results/p5_rows.json (written by p1_eval_cnn.py, p1_sanity.py, p2_mri_shap.py, p5_clinical.py,
p3_kan_shapley.py) and results/card_rules.json["cnn_stage"]. Rules (preregistration.md §7):
  stability / plausibility: PASS >= 0.80, WEAK 0.60-0.80, FAIL < 0.60;
  faithfulness: PASS if deletion and insertion pass, WEAK if one, FAIL if neither;
  train is primary and val the replication: a PASS not replicated on val is downgraded to WEAK
  (CNN faithfulness: val AD is primary, train AD is reported only);
  CNN: if no stage was eligible under P1-R1, faithfulness is FAIL; sanity cells from P1-R4;
  overall VALIDATED iff every applicable cell is PASS, otherwise WEAK with the cells listed.
Outputs: results/p5_eval_table.csv, results/p5_eval_table.md, results/eval_status.json.
"""
import argparse
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402


def tier(v):
    return C.tier(v)


def replicate(primary, replication):
    if primary == "PASS" and replication not in ("PASS", "N/A"):
        return "WEAK"
    return primary


def fmt(v):
    return "—" if v is None or (isinstance(v, float) and v != v) else f"{v:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    if a.dry:
        C.set_dry()
    rows = json.load(open(os.path.join(C.RESULTS, "p5_rows.json")))
    stage = json.load(open(os.path.join(C.RESULTS, "card_rules.json")))["cnn_stage"]
    table, status = [], {}
    for key, name in (("cnn", "CNN region attribution"), ("mri_shap", "MRI-specialist SHAP"),
                      ("clinical_shap", "Clinical SHAP"), ("kan", "KAN Shapley")):
        r = rows[key]
        cells = {}
        # stability
        st = replicate(tier(r["stability_train"]), tier(r.get("stability_val")))
        cells["stability"] = (st, f"train {fmt(r['stability_train'])}, val {fmt(r.get('stability_val'))}")
        # plausibility
        if key == "cnn":
            for inp in ("hippocampus", "ventricles"):
                t, v = r["plausibility_train"][inp], r["plausibility_val"][inp]
                cells[f"plausibility: {inp}"] = (replicate(tier(t), tier(v)), f"train {fmt(t)}, val {fmt(v)}")
            for inp in ("CDRSB", "MMSE"):
                cells[f"plausibility: {inp}"] = ("N/A", "not an input")
        else:
            pl = r["plausibility"]
            for inp, src in (("CDRSB", "CDRSB"), ("MMSE", "MMSE"), ("hippocampus", "hippocampus"),
                             ("ventricles", "ventricles")):
                if src in pl:
                    t, v = pl[src]["train"], pl[src]["val"]
                    cells[f"plausibility: {inp}"] = (replicate(tier(t), tier(v)), f"train {fmt(t)}, val {fmt(v)}")
                else:
                    cells[f"plausibility: {inp}"] = ("N/A", "not an input")
        # faithfulness
        f = r["faithfulness"]
        if key == "cnn":
            fv = f["val_AD"]["verdict"]
            if stage.get("faithfulness_rule_fail"):
                fv = "FAIL"
            cells["faithfulness"] = (fv, f"val AD (n={f['val_AD']['n']}): deletion {fmt(f['val_AD']['deletion_diff'])}, "
                                         f"insertion {fmt(f['val_AD']['insertion_diff'])}; train AD {f['train_AD']['verdict']} "
                                         "(selection-biased)")
        else:
            fv = replicate(f["train"]["verdict"], f["val"]["verdict"])
            cells["faithfulness"] = (fv, f"train {f['train']['verdict']} (deletion {fmt(f['train']['deletion_diff'])}, "
                                         f"insertion {fmt(f['train']['insertion_diff'])}); val {f['val']['verdict']}")
        # sanity (CNN)
        if key == "cnn":
            s = rows["cnn_sanity"]
            cells["sanity: cascading randomisation"] = (s["cascade_verdict"],
                                                        f"median Spearman after head + layer4 "
                                                        f"{fmt(s['cascade_median_spearman_head_layer4'])}")
            cells["sanity: region deletion"] = (s["region_deletion_verdict"],
                                                f"top-region minus control drop {fmt(s['region_deletion_mean_difference'])}")
        else:
            cells["sanity: cascading randomisation"] = ("N/A", "CNN only")
            cells["sanity: region deletion"] = ("N/A", "CNN only")
        bad = [c for c, (v, _) in cells.items() if v not in ("PASS", "N/A")]
        overall = "VALIDATED" if not bad else "WEAK"
        status[key] = {"name": name, "overall": overall, "weak_or_failed": [f"{c} ({cells[c][0]})" for c in bad],
                       "cells": {c: {"verdict": v, "detail": dtl} for c, (v, dtl) in cells.items()}}
        table.append({"explanation": name, **{c: v for c, (v, _) in cells.items()}, "overall": overall})
    t = pd.DataFrame(table)
    C.write_aggregate(t, "p5_eval_table.csv", ["train", "val"])
    C.write_json(status, "eval_status.json")
    cols = [c for c in t.columns if c != "explanation"]
    md = ["# Explanation evaluation (P5; pre-registered thresholds, train primary, val replication)", "",
          "| Explanation | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for r in t.itertuples(index=False):
        md.append("| " + " | ".join(str(x) for x in r) + " |")
    md += ["", "Details:"]
    for k, s in status.items():
        md.append(f"- **{s['name']}** — {s['overall']}" + (f"; weak/failed: {', '.join(s['weak_or_failed'])}"
                                                            if s["weak_or_failed"] else ""))
        for c, v in s["cells"].items():
            if v["verdict"] != "N/A":
                md.append(f"  - {c}: {v['verdict']} ({v['detail']})")
    open(os.path.join(C.RESULTS, "p5_eval_table.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
