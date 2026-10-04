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

**The card data is not in git.** `demo/card_data/` and `demo/output/` hold
DUA-restricted, subject-level ADNI data:
- per-subject tables: IDs, diagnoses, scores, volumes and model outputs;
- brain-slice and Grad-CAM images.

Both are git-ignored. Build them locally first; see "Regenerating card data"
below. The notebook is committed with its outputs stripped. Do not commit an
executed copy.

## Run it (about 15 s, after regenerating the card data)

```bash
python3 -m venv .venv-demo          # once: numpy, pandas, matplotlib, jupyter, ipywidgets
.venv-demo/bin/pip install numpy pandas matplotlib jupyter ipywidgets
cd demo
../.venv-demo/bin/jupyter nbconvert --to notebook --execute assessment_demo.ipynb --output-dir output
# writes the executed copy to demo/output/ (git-ignored); or open the notebook in Jupyter and run all cells
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
  data = CardData()
  render_card(data, data.ptids[0], "demo/output")
  ```

## What it reads

The notebook only reads `demo/card_data/`. It never fits or runs a model, and it
computes no metric. That directory is written from the frozen artefacts by three
scripts, which need the project environments and, for Grad-CAM, the MRI volumes:

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

The notebook prints the selected subjects: category, ID, split, and whether the
category fell back to train. This README does not list them, because they are
subject-level data. One category does fall back to train: no val scan fails the
brain-age reliability rule, so the unreliable-scan subject comes from train.

Val and train predictions are in-sample for the frozen system.

## Regenerating card data

This needs local ADNI data, which is not in git:
- the clinical tables under `dataset/`;
- the preprocessed MRI volumes `dataset/mri_prep/vol96/`, for Grad-CAM;
- the ALZ cross-fit checkpoints `analysis/fusion_feasibility/runs/fold*_s*/best.pt`;
- the frozen fusion-v3 artefacts in `analysis/fusion_v3/frozen/` and
  `frozen_run/`.

On this machine the environments live in `LYP/`. Run from the repo root, in
this order:

```bash
../../.venv-fusion/bin/python src/card/build_card_data.py      # ~5 s; writes subjects.csv, clinical_shap.npz, streams_all.npz, meta.json
../../.venv-kan/bin/python    src/card/build_kan_terms.py      # ~5 s; needs streams_all.npz; writes kan_terms.npz, kan_edges.json
../../.venv-mri/bin/python    src/card/build_gradcam.py        # ~1.5 min on GPU; writes gradcam.npz
cd demo && ../../../.venv-demo/bin/jupyter nbconvert --to notebook --execute assessment_demo.ipynb --output-dir output
```

- **Self-checks:** steps 1 and 2 assert that their val and test outputs
  reproduce `analysis/fusion_v3/frozen_run/`, and they fail if that is not the
  case.
- **Nothing is retrained**, and no metric is computed.
- **Output:** `demo/card_data/` (about 14 MB) and the rendered cards in
  `demo/output/`. Both stay local.
