"""Full per-checkpoint detail behind the summary table."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refit_thresholds import (CLASSES, CURRENT, confusion, describe,
                              fit_cutpoints, macro_f1_from_conf, DIR)
from scipy.stats import spearmanr


def show(tag, meta):
    d = np.genfromtxt(os.path.join(DIR, tag + ".csv"), delimiter=",", names=True)
    lab = d["true_class"].astype(int); demo = d["demo_pred"].astype(float)
    print("=" * 78)
    print(f"{tag}   arch={meta.get('arch')}  epoch={meta.get('epoch')}  "
          f"params={meta.get('params'):,}  n={len(lab)}")
    print("=" * 78)
    cuts, mf1f, gN, nties = fit_cutpoints(demo, lab)
    for name, cs in [("CURRENT  (0.25/0.75/1.50)", CURRENT), (f"FITTED   ({cuts[0]:.2f}/{cuts[1]:.2f}/{cuts[2]:.2f})", cuts)]:
        p = np.digitize(demo, cs, right=True)
        c = confusion(lab, p); mf1, f1 = macro_f1_from_conf(c)
        print(f"\n{name}   acc4={float((p==lab).mean()):.4f}  macroF1={mf1:.4f}")
        print("        pred:   CN  EMCI  LMCI    AD  |  true tot")
        for k in range(4):
            print(f"   true {CLASSES[k]:5s}  {c[k,0]:4d}  {c[k,1]:4d}  {c[k,2]:4d}  {c[k,3]:4d}  |  {c[k].sum():4d}")
        print(f"   pred tot     {c[:,0].sum():4d}  {c[:,1].sum():4d}  {c[:,2].sum():4d}  {c[:,3].sum():4d}")
        print("   per-class F1: " + "  ".join(f"{CLASSES[i]}={f1[i]:.3f}" for i in range(4)))
    rho, pv = spearmanr(demo, lab)
    print(f"\n   Spearman(demo_pred, true_class) = {rho:.4f}   p = {pv:.3e}")
    print(f"   grid points={gN}  optimal triples={nties}")
    print(f"   demo_pred > 1.50 : {int((demo>1.50).sum())}/{len(demo)} = {float((demo>1.50).mean()):.4f}")
    print(f"\n   {'group':6s} {'n':>4s} {'min':>8s} {'max':>8s} {'mean':>8s} {'std':>8s}")
    for nm, n, mn, mx, mu, sd in describe(demo, lab):
        print(f"   {nm:6s} {n:4d} {mn:8.4f} {mx:8.4f} {mu:8.4f} {sd:8.4f}")
    print()


man = {m["tag"]: m for m in json.load(open(os.path.join(DIR, "_manifest.json")))}
sel = sys.argv[1:] or sorted(man)
for t in sel:
    show(t, man[t])
