# Assessment card demo

This demo renders a one-subject assessment card, as HTML and PNG, for the
frozen fusion-v3 system (freeze commit `8f351b6`). The card shows:

- the predicted stage with its 4-class probabilities;
- the AD probability;
- the severity index;
- clinical SHAP evidence;
- MRI evidence: the MRI specialist, volumes against CN reference ranges, and
  Grad-CAM;
- brain age;
- the KAN's MRI correction;
- the exploratory discordance flag.

**Research prototype, not for clinical use.**

## Run it (about 15 s)

```bash
python3 -m venv .venv-demo          # once: numpy, pandas, matplotlib, jupyter, ipywidgets
.venv-demo/bin/pip install numpy pandas matplotlib jupyter ipywidgets
cd demo
../.venv-demo/bin/jupyter nbconvert --to notebook --execute assessment_demo.ipynb --output assessment_demo.ipynb
# or open assessment_demo.ipynb in Jupyter and run all cells
```

On this machine the environment is `LYP/.venv-demo` (`../../.venv-demo` from the
repo root).

- **Paths:** the notebook finds the repo root from its working directory, and
  uses no absolute paths.
- **What it does:** it applies the showcase rule, renders the 8 showcase cards
  and one test subject into `demo/output/`, and displays them.
- **Other subjects:** set `SUBJECT` to any train, val or test PTID, or use the
  picker in interactive Jupyter.
- **From Python** (with `src/` on `sys.path`):
  ```python
  from card import CardData, render_card
  render_card(CardData(), "003_S_4136", "demo/output")
  ```

## What it reads

The notebook only reads `demo/card_data/`. It never fits or runs a model, and it
computes no metric. That directory was written once from the frozen artefacts
by three scripts, which need the project environments and, for Grad-CAM, the
MRI volumes:

| Step | Command (from the repo root) | Output |
|---|---|---|
| 1 | `../../.venv-fusion/bin/python src/card/build_card_data.py` | `subjects.csv` (inputs, specialist outputs, brain age, flags, volumes), `clinical_shap.npz`, `streams_all.npz`, `meta.json` |
| 2 | `../../.venv-kan/bin/python src/card/build_kan_terms.py` | `kan_terms.npz` (final probabilities, ADD, KAN residual and edge terms), `kan_edges.json` |
| 3 | `../../.venv-mri/bin/python src/card/build_gradcam.py` | `gradcam.npz` (central slices of scan and Grad-CAM; about 1.5 min on GPU) |

**Steps 1 and 2 check themselves against the frozen run:** the val and test
outputs must reproduce `analysis/fusion_v3/frozen_run/` (specialists < 1e-6,
KAN < 1e-4).

**Subjects:** cards cover all 814 subjects, train, val and test.
- **Train and val** subjects are in the frozen system's training data, so their
  outputs are in-sample, and the card says so.
- **Test** outputs are those of the single frozen test run.

## Card rules

1. **Header:**
   - subject ID, split and in-sample status;
   - the true diagnosis (known for every ADNI subject here);
   - whether a scan is available.
2. **Outputs:**
   - the final KAN fusion stage and its probabilities;
   - the ADD-head AD probability;
   - the severity index s = Σ k·p(k) on a 0–3 gauge, marked with the final
     KAN's train-CV stage means.
3. **Clinical evidence:**
   - the top 6 SHAP contributions of the clinical specialist towards its own
     predicted stage: CatBoost ShapValues in log-odds, averaged over 5 seeds,
     with features outside a seed's top 12 counted as 0;
   - shown as value, direction and magnitude;
   - its inputs are cognitive, functional and demographic only.
4. **MRI evidence:**
   - the MRI specialist's stage probabilities and P(AD), and ALZ;
   - the six FreeSurfer volume/ICV ratios against CN-train P5 / P25 / P50 /
     P75 / P95, with the subject's percentile;
   - Grad-CAM of the ALZ CNN on the central slices:
     - **train** subjects use their own fold's 3 checkpoints, which never saw
       them; **val and test** subjects use all 15;
     - the map is 3×3×3 before upsampling, so the card says it cannot localise
       a structure, and that the cohort-level attribution is periventricular /
       ventricular.
5. **Brain age:**
   - raw and corrected gap, with the frozen α and β;
   - marked **unreliable** when the pre-registered reliability rule fires;
   - "not available" with its reason (no scan, or registration check failed);
   - an **"extreme value"** note when |corrected gap| ≥ 15 years. This
     threshold was fixed in `build_card_data.py` before any card was rendered,
     and it fires for 20 of the 814 subjects.
6. **How MRI changes the prediction:**
   - clinical specialist probabilities vs final probabilities;
   - the total KAN correction to each logit;
   - the MRI edges, but only those monotone in the same direction in **all 25
     val-stage KANs** (`kan_interp.json`) **and all 25 frozen KANs** (re-checked
     with the same rule). Two edges pass:
     - MRI CN → AD (decreasing);
     - MRI EMCI → LMCI (decreasing).
   - Two edges consistent in the val-stage analysis are not shown, because the
     frozen KANs fall short:
     - MRI AD → CN, in 22 of 25;
     - MRI EMCI → CN, in 24 of 25.
   - The non-monotone MRI AD → AD edge is never shown.
7. **Flags:** the discordance flag, labelled exploratory, using the
   pre-registered train-only thresholds:
   - clinical specialist predicts AD and ALZ < 0.2981; or
   - it predicts CN/EMCI and ALZ > 0.8751.
8. **Footer:** model and library versions, the freeze commit, and "research
   prototype, not for clinical use".

## Showcase set

**The rule** (`src/card/showcase.py`) was committed in `0c7074e`, before any card
output existed:
- **Source:** the val split, ordered by PTID; take the first subject in each
  category below.
- **Duplicates:** a subject already chosen is skipped.
- **Empty categories** fall back to the train split.

| Category | PTID | Split | True | Final | Clinical specialist | Note |
|---|---|---|---|---|---|---|
| correct CN | 003_S_4900 | val | CN | CN | CN | corrected brain-age gap −20.9 y ("extreme value" note) |
| correct EMCI | 002_S_2010 | val | EMCI | EMCI | EMCI | |
| correct LMCI | 002_S_4654 | val | LMCI | LMCI | LMCI | MRI specialist says AD |
| correct AD | 003_S_4136 | val | AD | AD | AD | gap +17.4 y ("extreme value" note) |
| misclassified LMCI | 006_S_4713 | val | LMCI | **EMCI** | LMCI | the fusion moved a correct clinical LMCI to EMCI |
| discordance flag | 135_S_4954 | val | AD | AD | AD | flag fires (clinical AD, ALZ 0.24); the only flagged val subject |
| unreliable brain-age scan | 016_S_4121 | **train** (fallback) | CN | CN | CN | no val scan fails the rule; 23% of brain voxels saturated |
| no scan | 941_S_4187 | val | LMCI | LMCI | LMCI | the only val subject without a scan; neutral MRI inputs |

Val and train predictions are in-sample for the frozen system.
