# Explainability of the frozen fusion-v3 system: report

Branch `explainability`, from `fusion-v3` @ `468a010`. Date 2026-10-05.

| Commit | Content |
|---|---|
| `050ca8e` | Pre-registration (`preregistration.md`), before any analysis code or result existed |
| `9d22334` | Atlas |
| `39fec00` | Analysis code + dry run / go-no-go (`dryrun_gonogo.md`) |
| `7b0ed68` | Full-run results |
| `b8ca90a` | Card |

**What this is.** A post-hoc analysis.
- Nothing frozen was retrained or modified. `frozen/MANIFEST.sha256` (109 files)
  verifies before and after every script, and `freeze.md` is untouched.
- **No test metric was computed.** Every selection, threshold and evaluation uses
  train (primary) and val (replication).
- Test subjects appear only on rendered cards.
- The only models fitted were in-memory, never-saved copies:
  - the 200 bootstrap refits of the MRI specialist for stability (P2-R4);
  - the 25 regenerated train-only KANs (P3-R3).

Per-subject outputs are gitignored. Everything under `results/` and `figures/`
is an aggregate over train and/or val.

## Summary

| | Result |
|---|---|
| **P1 CNN attribution** | The deletion rule chose **LayerCAM at layer1 (24³)**, the only eligible stages being layer1 and layer2. The naming rule names 7 regions, but **both sanity checks fail**: <ul><li>the maps barely change when the trained weights are randomised;</li><li>occluding the most-attributed region lowers ALZ *less* than occluding similar-sized control regions.</li></ul> The maps are almost identical across CN, EMCI, LMCI and AD. **They should not be read anatomically.** The card names regions only because the pre-registered rule says so, and it labels the panel WEAK with the failed checks. |
| **P2 MRI-specialist SHAP** | Exact (additivity error 1e-15). Volumes carry 66% of the AD attribution, ALZ + brain age 27%, and age + sex 6%. This **agrees** with the M0–M5 controls. Weak points: <ul><li>ventricle plausibility **fails** (12%), because the frozen model's ventricle coefficient points away from AD;</li><li>stability is WEAK (0.60).</li></ul> |
| **P3 KAN Shapley** | Exact (efficiency error 1e-6; brute force agrees to 5e-7). <ul><li>89 of 144 cohort cells are consistent in all 25 KANs. All 36 cells of the MRI group are consistent.</li><li>On held-out val with the regenerated train-only KANs, 78 consistent cells replicate, 7 partially and 4 do not.</li><li>The train-only KANs reproduce the stored val predictions exactly (Δ = 0).</li></ul> |
| **P4 Stage changes** | <ul><li>**Fusion changed the stage** for 68 of 569 train subjects (out-of-fold): 31 corrected and 32 made wrong. The net accuracy change is −0.2% [−2.8%, +2.6%]. On val it changed 12 of 122: 6 corrected, 6 made wrong.</li><li>**MRI changed the stage** for 10 of 121 held-out val subjects: 6 corrected and 4 made wrong. Net +1.7% [−3.3%, +6.6%].</li><li>**No measurable benefit either way.**</li></ul> |
| **P5 Evaluation** | **All four explanation types are WEAK**, each for a different reason (table below). Faithfulness passes for all four, and stability for three. |

## 0. Setup

| Item | Value |
|---|---|
| Environments | `.venv-mri` (P1, CNN rows of P5, `build_cam.py`) · `.venv-fusion` (P2, P4, clinical P5, `p5_table.py`, `build_card_data.py`) · `.venv-kan` (P3, `build_kan_terms.py`) · `.venv-synthseg` (atlas step 2, run alone: 39 s, 10.9 GB peak) · `.venv-demo` (notebook) |
| Full-run compute | <ul><li>GPU lane: 14.6 min</li><li>CPU lane: 6 min</li><li>card builders: about 3 min</li></ul> Peak RAM was 2.8 GB in both lanes, under the pre-registered 10 GB (and below that again in the separate card builders). |
| Bootstrap | 2,000 resamples of subjects, percentile 95% CI, `default_rng(0)` |
| Splits | train 569 (567 with a scan) · val 122 (121) |

## 1. Atlas

The template is ICBM152 2009a (sym), the template every scan was affinely
registered to. SynthSeg 2.0 was run on it in the m1 working grid, then
resampled to the CNN's 96³ grid with m1's own call (one-hot linear + argmax).

**Acceptance:** Dice of the atlas brain vs the template mask = **0.989** (pre-registered threshold 0.90).

**Fit to individual anatomy.** The template atlas was compared, per region, with
the existing SynthSeg segmentations of 22 train smoke subjects (median Dice):

| Region | Median Dice |
|---|---|
| cerebellum | 0.89 |
| brainstem | 0.86 |
| thalamus | 0.76 |
| amygdala | 0.72 |
| basal ganglia | 0.66 |
| **hippocampus** | **0.66** |
| cerebral white matter | 0.65 |
| temporal lobe | 0.59 |
| **ventricles** | **0.54** |
| frontal lobe | 0.52 |
| parietal lobe | 0.47 |
| occipital lobe | 0.47 |

- The hippocampus median is ≥ 0.5, so the pre-registered "approximate" label is
  not triggered.
- Cortical lobes are coarse, as expected, and are always labelled approximate.

Files: `atlas/atlas96.nii.gz`, `atlas/regions.json`, `atlas/atlas_qc.png`, `results/atlas_qc.csv`.

## 2. P1: CNN attribution

### 2.1 Stage choice (P1-R1)

100 train AD subjects, top 10% of brain voxels replaced with the CN-train mean, ALZ-logit drop:

| Stage | Map | LayerCAM drop D | Random drop | Excess [95% CI] | Eligible |
|---|---|---|---|---|---|
| **layer1** | 24³ | **1.08** | 0.31 | **+0.77 [0.66, 0.89]** | **yes (chosen)** |
| layer2 | 12³ | 0.73 | 0.29 | +0.44 [0.36, 0.53] | yes |
| layer4 | 3³ | 0.23 | 0.24 | −0.02 [−0.09, 0.06] | no |

- **Grad-CAM (reference)** excess: layer1 +0.06, layer2 −0.02, layer4 −0.04.
- **The previous card map** (layer4 Grad-CAM) removes no more ALZ evidence than random occlusion.

Table: `results/p1_stage_selection.csv`.

### 2.2 Regions and the naming rule (P1-R2, P1-R3)

Density d_r is a region's share of attribution divided by its share of brain
voxels; 1 = the brain average. Train AD (n = 104):

| Region | Density [95% CI] | Named | Val AD | AD − CN [95% CI] |
|---|---|---|---|---|
| amygdala | 1.34 [1.32, 1.35] | yes | 1.33 | +0.04 [0.02, 0.07] |
| thalamus | 1.25 [1.24, 1.26] | yes | 1.25 | −0.02 [−0.03, 0.00] |
| hippocampus | 1.24 [1.23, 1.26] | yes | 1.24 | +0.04 [0.02, 0.06] |
| ventricles | 1.20 [1.18, 1.21] | yes | 1.19 | **−0.06 [−0.08, −0.04]** |
| basal ganglia | 1.20 [1.19, 1.21] | yes, "also present in non-AD subjects" | 1.19 | +0.00 [−0.01, 0.02] |
| cerebral white matter | 1.11 [1.10, 1.11] | yes | 1.10 | +0.01 [0.00, 0.01] |
| brainstem | 1.10 [1.09, 1.12] | yes | 1.10 | −0.03 [−0.05, −0.02] |
| parietal / frontal / temporal / occipital lobe | 0.97 / 0.95 / 0.93 / 0.86 | no | — | — |
| cerebellum | 0.77 | no | — | — |

**What this shows.** The rule names 7 deep regions, and every one replicates on
val. But the AD − CN differences are tiny (|Δ| ≤ 0.06 on densities around
1.2). The mean maps of CN, EMCI, LMCI and AD look almost identical
(`figures/p1_group_maps.png`): the concentration is a property of the map, not
of AD. For ventricles the density is significantly *lower* in AD than in CN.

**Wording gap (flagged, not fixed).** As literally pre-registered, the card
qualifier "also present in non-AD subjects" is added only when the AD − CN CI
includes zero. It is therefore not added for regions where AD < CN significantly
(ventricles, brainstem, thalamus up to rounding), although the concentration is
present in non-AD subjects there too. This needs a sign-off decision
(§8 below). The card's WEAK label and failed checks appear next to the text in
every case.

**The lattice.** Layer1 LayerCAM maps show a regular dot pattern with a period
of 2 cells (about 16–20 mm) in every subject. It is the gradient checkerboard
left by the downstream stride-2 convolutions. It was seen in the dry run and,
as pre-registered, not acted on. The card says it is not anatomy.

**The old map.** Layer4 Grad-CAM (`figures/p1_group_maps_gradcam_layer4.png`)
does differ by class: AD maps are brighter centrally and periventricularly. But
P1-R1 shows that its top voxels carry no more ALZ evidence than random
occlusion. The earlier card wording ("periventricular") therefore described a
real but unfaithful pattern.

### 2.3 Sanity checks (P1-R4)

**(a) Cascading randomisation** (50 train subjects): median Spearman
correlation with the original map after cumulatively randomising each block:

| Randomised through | Median Spearman |
|---|---|
| ALZ head | 0.85 |
| **+ layer4** | **0.81** |
| + layer3 | 0.81 |
| + layer2 | 0.80 |
| + layer1 | 0.73 |
| + stem | 0.72 |

- The pre-registered criterion (< 0.5 after head + layer4) **fails**.
- Even a fully randomised network gives maps correlated 0.72 with the trained
  ones. The map is driven mostly by the input image and the architecture, not
  by what the CNN learned.

**(b) Region deletion** (104 train AD):
- occluding each subject's highest-density region (amygdala in 78, thalamus in
  17) lowers the ALZ logit by 0.04;
- size-matched control regions lower it by 0.16;
- difference −0.12 [−0.15, −0.08]: **fails**.

**Meaning for the claim.** Voxel-level faithfulness passes (§6): the top voxels
of the map, a sparse lattice, matter more for ALZ than smoothed random voxels.
But region-level attribution and dependence on the learned weights do not hold
up. The CNN attribution cannot support any anatomical statement about what the
CNN uses. This is consistent with the CNN being the minor MRI input (P2).

Figures: `figures/p1_cascade.png`, `figures/p5_cnn_faithfulness.png`.

## 3. P2: MRI-specialist SHAP

- **Method:** exact linear SHAP of the frozen logistic regression's centred
  class scores.
- **Background:** 567 train subjects with a scan.
- **Additivity:** error 1.1e-15, and `shap.LinearExplainer` agrees exactly.
- The 5 seed pickles are identical, which was asserted.

**Group comparison with the M0–M5 controls (P2-R2):**

| Group | Mean \|φ\| on the AD score [CI] | Share [CI] | Controls' standalone CV AUC |
|---|---|---|---|
| volumes (6) | 1.257 [1.207, 1.305] | 0.66 [0.65, 0.68] | 0.867 |
| ALZ + brain-age gap | 0.519 [0.493, 0.544] | 0.27 [0.26, 0.29] | 0.788 |
| age + sex | 0.115 [0.113, 0.117] | 0.06 [0.06, 0.06] | 0.574 |

**Agree.** SHAP and the input-set controls rank the groups the same way.

**Largest single drivers of the AD score (train mean |φ|):**
1. hippocampus 0.40
2. brain-age gap 0.31
3. entorhinal 0.29
4. middle temporal 0.24
5. ALZ 0.21
6. fusiform 0.16
7. ventricles 0.13

**Known failure.** The frozen model's AD-row ventricle coefficient is negative
(disclosed in the pre-registration). The ventricle contribution to the AD score
therefore has the unexpected sign in 88% of train subjects: larger-than-median
ventricles push *away* from AD, and smaller ones toward it. This is visible on the cards; for example, a correct-AD showcase
subject at the 100th ventricle percentile gets −0.6 toward AD. The most likely
cause is collinearity with the whole-brain ratio. Nothing is changed, because
the model is frozen.

## 4. P3: KAN Shapley

**Groups:**
1. clinical logits;
2. MRI logits + has_mri;
3. age;
4. CDR-SB + CogIndex + MMSE.

**Method.** Because K2 is additive per input and softmax ignores the log-sum-exp
shift, Shapley values are exact and closed-form. The brute-force 16-coalition
check on 60 train subjects × 25 KANs agrees to 5e-7, and the efficiency error
is 1e-6.

**Consistent cells (P3-R1)** in all 25 frozen KANs, with |median| ≥ 0.05:

| Group | Final logits | Residual | ADD |
|---|---|---|---|
| MRI + has_mri | 16/16 | 16/16 | 4/4 |
| clinical logits | 14/16 | 7/16 | 3/4 |
| CDR-SB / CogIndex / MMSE | 10/16 | 10/16 | 3/4 |
| age | 3/16 | 3/16 | 0/4 |
| **Total** | | | **89/144** |

The MRI group's correction is ordered by the true stage. For example, its
correction to the AD logit has a median of:
- CN subjects −0.45;
- EMCI −0.11;
- LMCI +0.23;
- AD +0.54.

The cell-level edge inconsistency reported in `fusion_v3/kan_interp.md` thus
does not carry over to the group level: the group effects are stable.

**Held-out replication (P3-R3).**
- **Gate:** the 25 train-only KANs regenerated in memory reproduce
  `kan_runs/32.npz` val probabilities and ADD exactly (max |Δ| = 0.0, all 5
  seeds).
- **Outcome of the 89 consistent cells:**
  - 78 replicated;
  - 7 partially replicated;
  - 4 not replicated: MRI → EMCI logit in LMCI subjects (final and residual),
    and age → CN logit in EMCI subjects (final and residual).

**On the card:**
- a group's value is shown only if all 25 KANs agree in sign and |mean| ≥ 0.05;
- cohort cells are shown only if consistent, and carry their replication
  status.

**Caveat.** Shapley values explain each KAN's logits; the card's probabilities
are the mean of 25 softmaxes.

Tables: `results/p3_sign_consistency.csv`, `p3_global.csv`, `p3_efficiency.json`, `p3_replication.json`.

## 5. P4: Stage changes

**A. "Fusion changed the stage"** (final ≠ clinical specialist; 5-seed mean, train out-of-fold / val):

| Split | n | Fired | Corrected | Made wrong | Both wrong (closer / farther) | Net accuracy [95% CI] |
|---|---|---|---|---|---|---|
| train OOF | 569 | 68 | 31 | 32 | 3 / 2 | −0.2% [−2.8%, +2.6%] |
| val | 122 | 12 | 6 | 6 | 0 / 0 | 0.0% [−5.7%, +5.7%] |

**By transition (train out-of-fold):**
- EMCI → LMCI: 47 (20 corrected, 24 made wrong);
- AD → LMCI: 7 (4 / 3);
- LMCI → EMCI: 6 (5 / 1);
- EMCI → CN: 4 (0 / 3);
- LMCI → AD: 4 (2 / 1).

Only EMCI → LMCI reaches the card's n ≥ 10.

**B. "MRI changed the stage"** (final ≠ final with the scan treated as missing; subjects with a scan):

| Models | Split | n | Fired | Corrected | Made wrong | Net [95% CI] |
|---|---|---|---|---|---|---|
| **train-only KANs (held out)** | val | 121 | 10 | 6 | 4 | +1.7% [−3.3%, +6.6%] |
| frozen KANs (in-sample) | train | 567 | 30 | 12 | 18 | −1.1% [−3.0%, +0.7%] |
| frozen KANs (in-sample) | val | 121 | 6 | 4 | 2 | +1.7% [−1.7%, +5.8%] |

**Reading.** Neither indicator shows a measurable benefit. The fusion's stage
changes are about as often wrong as right. MRI changes the stage for about 8% of
held-out subjects, with a CI that spans zero.

**Discordance flag** (exploratory, kept as is):
- train OOF: 13 of 567 flagged; error rate 23% flagged vs 24% unflagged;
- val: 1 of 121 flagged, so no estimate is possible. The bootstrap CI printed
  in `p4_discordance.csv` is degenerate with n = 1.

**Caveat.** Config 32 was the best of 35 on the same train CV, so the
out-of-fold estimate is slightly optimistic.

## 6. P5: Evaluation table

The table is generated in `results/p5_eval_table.md`. Tiers: PASS ≥ 0.80,
WEAK 0.60–0.80, FAIL < 0.60. Train is primary and val the replication.

| Explanation | Stability | Plausibility | Faithfulness (del / ins) | Sanity | Overall |
|---|---|---|---|---|---|
| CNN region attribution | PASS (0.91 / val 0.93) | hippocampus PASS (1.00); ventricles PASS (0.98) | PASS (val AD: −0.14 / +0.12) | cascade **FAIL** (0.81); region deletion **FAIL** | **WEAK** |
| MRI-specialist SHAP | **WEAK** (0.60 / val 0.73; 200 refits) | hippocampus PASS (0.85); ventricles **FAIL** (0.12) | PASS (−0.08 / +0.08) | N/A | **WEAK** |
| Clinical SHAP | PASS (0.90) | CDR-SB **FAIL** (0.36); MMSE WEAK (0.71) | PASS (−0.19 / +0.23) | N/A | **WEAK** |
| KAN Shapley | PASS (0.88) | CDR-SB WEAK (0.60); MMSE WEAK (0.79) | PASS (−0.18 / +0.19) | N/A | **WEAK** |

**Reading the failures honestly:**

- **CNN plausibility passes trivially.** Every class has densities above 1 in
  these regions (§2.2), so the check does not discriminate. The sanity checks
  are the informative ones, and they fail.
- **CDR-SB plausibility, clinical and KAN.** The criterion compares each
  subject's raw value with the CN-train median and asks for a push toward AD.
  CDR-SB's CN median is very low, so most EMCI and LMCI subjects count as
  "above". For them a multiclass model plausibly pushes toward EMCI or LMCI
  rather than AD.
  - This is the likely reason for the low fractions; it was not tested
    separately.
  - The criterion was fixed in advance and is reported as it stands.
- **MRI stability** is the agreement of global feature-importance rankings
  across 200 bootstrap refits.
  - It is 0.60 on train (val 0.73), and 0.70 against the frozen model's ranking.
  - Per-subject agreement of the AD attributions with the frozen model is high:
    median 0.93.
  - Which inputs swap ranks was not analysed.

## 7. Card changes

These are in `src/card/`. The card reads the committed `results/card_rules.json`
and `results/eval_status.json`.

1. **MRI evidence.** MRI-specialist SHAP drivers come first (6 for its
   predicted class, then 3 for AD, imputed inputs marked). Then the volume
   reference. Then the CNN attribution: LayerCAM at layer1, whole-map overlay on
   fixed hippocampal slices. Its region text is generated by the naming rule and
   carries the lattice note.
2. **Stage changes.** Both indicators, with the out-of-fold transition counts
   and the held-out val result.
3. **How the fusion changes the prediction.** Group Shapley values for the
   predicted stage's final logit and its KAN correction, consistent ones only;
   consistent cohort cells with their replication status. The KAN edge curves
   are no longer drawn.
4. **Validation status.** Every explanation panel states VALIDATED or WEAK with
   the failed checks. All four are WEAK.
5. **Discordance flag.** Unchanged, labelled exploratory.

**Showcase rule** (`showcase.py`, commit `0c7074e`): unchanged.

**Display change.** The CNN overlay now shows the whole map instead of its top
40%, because the threshold picked out only lattice peaks. This is a display
choice, not a pre-registered rule.

## 8. Open items for sign-off

1. **Naming-rule qualifier (§2.2).** The literal rule omits "also present in
   non-AD subjects" where AD < CN significantly. The proposed fix, as a dated
   addendum, is to apply the qualifier whenever the AD − CN CI does not lie
   entirely above zero. On current numbers this would add the qualifier to
   ventricles, thalamus and brainstem, and leave amygdala, hippocampus and
   cerebral white matter unqualified. Not done without your approval, because
   it would be a post-results rule change.
2. **CNN panel on the card.** With both sanity checks failing, you may prefer
   the card to show the CNN map without region names. That would also be a
   change to the pre-registered card rule, so it is not done.

## 9. Verification

| Check | Result |
|---|---|
| `python analysis/explainability/audit.py` | manifest (109) verifies; freeze.md and frozen/ unchanged vs fusion-v3; pre-registration precedes all results commits; no subject-level file tracked; no identifier columns; no test split in any results file; no frozen artefact written or refitted |
| Card builders | still reproduce `frozen_run/` for val and test |
| Notebook | executes end to end and renders the 8 showcase cards |
