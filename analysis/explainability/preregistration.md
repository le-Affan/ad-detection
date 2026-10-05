# Explainability of the frozen fusion-v3 system — pre-registration

Branch `explainability`, created from `fusion-v3` @ `468a010`. Date 2026-10-05.

This file is written and committed before any result of the analyses below
exists.

**Post-hoc only.**
- No frozen artefact is retrained, refitted or modified. This is verified with
  `analysis/fusion_v3/frozen/MANIFEST.sha256` (109 files) at the start and end
  of every script.
- `analysis/fusion_v3/freeze.md` is not edited.
- The only models fitted are the in-memory, never-saved copies named in §4
  (P2-R4) and §5 (P3-R3).

**No test metric is computed.** Every method, layer, threshold and naming
decision uses TRAIN subjects and is replicated on VAL. Test subjects appear
only on rendered cards.

## 0. What was seen before writing this

Planning inspected the code and existing reports, including:
- the 3×3×3 Grad-CAM and its periventricular appearance at cohort level;
- one showcase card whose map peaks in frontal cortex;
- `kan_interp.md` (edge consistency);
- the kan-fusion discordance results: 1 flag on val, and no error enrichment on train;
- the MRI specialist's frozen coefficients. In the AD row, the Ventricles_ICV
  coefficient is negative (−0.17), and the 5 seed copies are identical.

The ventricle plausibility check for MRI-specialist SHAP (P5) is therefore
expected to fail. Its threshold is not changed because of this.

No attribution map, SHAP value or Shapley value from the new methods had been
computed.

## 1. Common rules

- **Bootstrap:** 2,000 resamples of subjects, percentile 95% CI,
  numpy `default_rng(0)`.
- **Splits:**
  - "train" = the 569 clinical-train subjects, 567 with a scan
    (CN 133, EMCI 216, LMCI 114, AD 104);
  - "val" = 122 subjects, 121 with a scan (AD 23).
- **Status tiers:** PASS / WEAK / FAIL, as defined for each check below.
- **Failures:** a failed check never removes a component. It is reported, and
  the card labels the panel "weak" and names the failed checks.
- **Dry run:** the first 3 AD, 3 CN, 2 EMCI and 2 LMCI train subjects with a
  scan, ordered by PTID (10 subjects).
  - It is checked for execution, shapes, additivity, efficiency, memory and
    runtime only.
  - No threshold or rule is revisited after it.
  - Any change after the dry run needs a dated addendum to this file, committed
    before the full run.
- **Go/no-go after the dry run:**
  - every script exits 0, and the manifest verifies before and after;
  - the P2 additivity and P3 efficiency tolerances are met;
  - maps have the expected shapes, contain no NaN, and every subject has
    positive mass > 0 at every stage;
  - peak RAM < 10 GB and VRAM < 7 GB;
  - the extrapolated full runtime is < 4 h;
  - the test-RID guard was exercised;
  - the atlas QC (§2) passed.

## 2. Atlas (prerequisite)

- **Segmentation:** SynthSeg 2.0 (FreeSurfer 7.4.1 `mri_synthseg --parc`) on
  the masked ICBM152 2009a (sym) template, in the m1 working grid
  (`m1_preprocess.load_template`).
- **Resampling:**
  1. to the working grid with genericLabel;
  2. to 96³ by one-hot linear resampling, with m1's own step-8 call, then
     argmax;
  3. restricted to the 96³ template mask (> 0.5).
- **Regions (14):**
  - hippocampus;
  - amygdala;
  - ventricles (lateral, inferior lateral, 3rd, 4th);
  - thalamus;
  - basal ganglia (caudate, putamen, pallidum, accumbens);
  - cerebellum (cortex + white matter);
  - brainstem;
  - cerebral white matter;
  - frontal, temporal, parietal and occipital lobes (Appendix A);
  - other (cingulate, insula, ventral DC, CSF, choroid plexus, unlabelled mask
    voxels). "Other" is never named.
- **Cortical lobes are coarse.** SynthSeg's cortical agreement with FreeSurfer
  on our volumes is weak (blueprint §8), so lobe names are reported as
  approximate.
- **Acceptance:** Dice(atlas brain, template mask) ≥ 0.90 at 96³.
- **Fit to individual subjects:**
  - I compute the per-region Dice of the template atlas against the existing
    SynthSeg segmentations of the 22 smoke subjects (train/val), processed the
    same way, and report it.
  - If the median hippocampus Dice is < 0.5, every region name on the card
    carries "approximate (template atlas, affine registration)".

## 3. P1 — CNN attribution

**Candidate stages:** the outputs of `trunk.layer1` (24³), `trunk.layer2` (12³)
and `trunk.layer4` (3³).

**Methods:**
- LayerCAM: M = ReLU(Σ_c ReLU(∂y/∂A_c) ⊙ A_c). This is the primary method.
- Grad-CAM: M = ReLU(Σ_c mean(∂y/∂A_c) · A_c). It is a reference only: reported
  at every stage, never selected.

**Target:** the ALZ logit, with models in eval mode and fp32.

**Map per model:**
1. trilinear upsampling to 96³ (`align_corners=False`);
2. masked to the template brain;
3. normalised to unit positive mass.

**Map per subject:** the mean over the subject's models. Train subjects use the
3 out-of-fold models of their own cross-fit fold; val and test subjects use all
15.

### P1-R1 Stage selection (deletion rule)

**Subjects:** 100 train AD subjects with a scan, drawn by `default_rng(0)` from
the 104, sorted by PTID, without replacement.

**Attribution drop D.** For each subject, each of its 3 models m and each stage s:
1. take the top 10% of brain voxels by model m's own LayerCAM map (ties broken
   by voxel index);
2. replace them with the CN-train mean volume (the voxel-wise mean of the 133
   CN-train scans);
3. D = model m's ALZ logit before − after.

**Random control D_rand.** For each subject and stage:
1. make 10 random masks, each the top 10% of brain voxels of Gaussian-smoothed
   white noise. Sigma = 96 / map size voxels, i.e. 4, 8 and 32; the seed is
   `SeedSequence([subject index, stage index, draw])`;
2. apply the same replacement;
3. D_rand = the mean over the 10 masks.

Subject values are the means over the subject's 3 models.

**Selection:**
- A stage is eligible if the bootstrap 95% CI of mean(D − D_rand) has a lower
  bound > 0.
- The chosen stage is the eligible stage with the largest mean D. Ties within
  1e-6 go to the finer stage.
- If no stage is eligible, the stage with the largest mean D is chosen, and CNN
  faithfulness is recorded as FAIL.

### P1-R2 Region attribution

- **Share:** share_r = positive mass in region r / mass in the brain.
- **Density:** d_r = share_r / (voxels_r / voxels_brain).
- **Per true class on train:** mean share_r and mean d_r, with bootstrap CIs.
  The same is reported on val.
- **Reported, not used for naming:** the AD − CN difference in mean d_r, with
  CI.

### P1-R3 Naming rule

- **Nameable:** a region r (excluding "other") may be named in card text iff,
  over train AD subjects, the bootstrap 95% CI of mean(d_r − 1) excludes zero
  on the positive side.
- **Val replication:** if the val-AD mean d_r is ≤ 1, the name carries "not
  replicated on validation subjects".
- **Specificity:** if the CI of the AD − CN difference includes zero, the text
  says the concentration is "also present in non-AD subjects".
- **Order:** names are ordered by train-AD mean d_r.
- **Nothing else is named.** Card text is generated only from this rule's
  output.

### P1-R4 Sanity checks (chosen stage, LayerCAM)

**(a) Cascading randomisation.**
- **Subjects:** the first 50 train subjects with a scan, by PTID, each with its
  own out-of-fold models.
- **Procedure:** re-initialise, cumulatively from the top: ALZ head, layer4,
  layer3, layer2, layer1, stem. This is done on in-memory deep copies with the
  model's own initialisers:
  - conv weights: Kaiming normal;
  - BN weight 1 and bias 0, with running stats reset (mean 0, variance 1);
  - Linear layers: `reset_parameters`.
  The torch seed is 0.
- **Similarity:** the Spearman correlation of the brain-masked 96³ subject maps
  between the original and the randomised models.
- **Verdict**, from the median Spearman after randomising the head + layer4:
  - PASS if < 0.5;
  - WEAK if 0.5–0.8;
  - FAIL if ≥ 0.8.

**(b) Region deletion.**
- **Subjects:** each of the 104 train AD subjects.
- **Occlusion:** occlude (with the CN mean) the subject's highest-d_r region,
  excluding "other". The drop is computed as in P1-R1, averaged over the
  subject's 3 models.
- **Control:** the mean drop over the other regions whose voxel count is within
  a factor of 2 of that region. If there are none, the single nearest region in
  size is used.
- **Verdict:**
  - PASS if the CI of the mean paired difference (top − control) is > 0;
  - WEAK if the mean is > 0 but the CI includes 0;
  - FAIL otherwise.

## 4. P2 — MRI-specialist SHAP

**Method:**
- Exact linear SHAP (interventional) of the centred class scores of the frozen
  logistic regression (decision_function minus its row mean).
- **Background:** the 567 train subjects with a scan, after the specialist's
  own imputation and scaling.
- **Imputed inputs** are explained at their imputed value and marked "imputed".

### P2-R1 Additivity
- For every subject and class: |Σφ + base − centred score| < 1e-6.
- The analytic values and `shap.LinearExplainer` (Independent masker, the same
  background) must agree within 1e-8.
- A failure is a bug and is fixed before any result is used.

### P2-R2 Comparison with the M0–M5 controls (`mri_controls.md`)
- **Groups:** volumes (6), ALZ + BAG, age + sex.
- **"Agree"** iff the train mean |φ| on the AD centred score, summed per group,
  orders volumes > ALZ + BAG > age + sex. That is the order of the controls'
  standalone AUCs: 0.867 > 0.788 > 0.574.
- **Otherwise "disagree",** reported with each group's share and its CI.

### P2-R3 Card
The 6 largest |φ| for the MRI specialist's predicted class, then the 3 largest
for AD if AD is not that class. Imputed inputs are marked.

### P2-R4 Stability refits (P5 only)
- 200 bootstrap samples of the 567 train subjects with a scan (`default_rng(0)`).
- For each sample, a copy of the frozen configuration is refitted in memory and
  never saved:
  - α and β on the sample's CN subjects with a reliable brain age;
  - median imputation and z-scoring;
  - LogisticRegression with C = 0.1 and max_iter = 5000.
- Its SHAP values are computed on the 567 train subjects.

## 5. P3 — KAN Shapley

**Groups:**
1. clinical logits (4);
2. MRI logits (4) + has_mri;
3. AGE;
4. CDRSB + CogIndex + MMSE.

**Targets, per model:**
- the centred final COG logits (log p_clinical + residual, minus the row mean);
- the centred residual;
- the ADD logit.

**Method:**
- Exact interventional Shapley. K2 is additive over inputs, and the clinical
  base depends only on group 1.
- **Background:** the 569 train subjects' prediction-time inputs.
- **Models:** all 25 frozen KANs.
- **Caveat:** these values explain each KAN's logits. The deployed output is
  the mean of 25 softmaxes, which is not additive in the groups.

### P3-R0 Efficiency
- For every subject, model and target: |Σ_G φ_G + E_bg f − f(x)| < 1e-4.
- A brute-force 16-coalition evaluation on 60 train subjects must agree within
  1e-4. The 60 are the 10 dry-run subjects plus 50 drawn by `default_rng(0)`.

### P3-R1 Cohort cells
- A cell is a (group, target, true-class stratum on train) combination. Its
  effect is the mean φ over the stratum's subjects, computed per model.
- **Consistent** iff the sign is the same in all 25 models and |median over
  models| ≥ 0.05 logits.
- Only consistent cells are shown on the card. The rest are listed as
  inconsistent in the report.

### P3-R2 Per-subject display
- A group's φ for a target is shown iff its sign is the same in all 25 models
  and |mean| ≥ 0.05.
- Otherwise it is listed as "inconsistent across models".

### P3-R3 Held-out replication (train-only KANs)
- **Regeneration:** the 25 train-only ("val-stage") KANs are regenerated in
  memory exactly as `kan_interpret.py` does: `kan_fusion.unit_data(seed, "val")`
  and `fit_predict` with KAN seeds 42–46. They are never written to disk.
- **Reproduction gate:** for every outer seed, the mean val probabilities over
  its 5 KANs must match `kan_runs/32.npz` `val` within 1e-4, and the mean ADD
  probability must match `val_add` within 1e-4.
  - If the gate fails, the held-out replication is skipped and the report says
    so.
  - P4 then falls back to the in-sample frozen models only.
- **If the gate passes:**
  - Shapley values are computed for the val subjects, with each KAN's own
    training inputs as background (the 569 train out-of-fold inputs).
  - Each P3-R1 consistent cell is evaluated on the matching val stratum. It is:
    - "replicated" if the sign agrees in all 25 train-only KANs;
    - "partially replicated" if it agrees in ≥ 13;
    - "not replicated" otherwise.
  - On the card, consistent cells that are not replicated carry "not
    replicated on held-out validation subjects".

## 6. P4 — Stage-change indicators

### P4-A "Fusion changed the stage"
- **Data:**
  - train out-of-fold: `streams.npz` `clin_oof` and `kan_runs/32.npz` `cv`;
  - val: `clin_val` and `kan_runs/32.npz` `val`.
- **Predictions:** the mean of the 5 seeds' probabilities, then argmax. The
  per-seed range is reported.
- **Fires** iff argmax(final) ≠ argmax(clinical).

### P4-B "MRI changed the stage"
- **Fires** iff argmax(ensemble, with the scan) ≠ argmax(ensemble, with the MRI
  inputs set to the system's missing-scan input: neutral logits, has_mri = 0).
- Evaluated only for subjects with a scan.
  - **Primary (held out):** the train-only KANs (P3-R3) on val, if the
    reproduction gate passes.
  - **Descriptive (in-sample):** the 25 frozen KANs on train and val.

### Outcomes (both indicators)
- **Among firings,** with the reference being clinical (A) or MRI-masked (B):
  - corrected: the reference is wrong, the final is right;
  - broken: the reference is right, the final is wrong;
  - both wrong: closer, farther, or at the same distance from the true stage.
- **Net effects, with bootstrap CIs:**
  - (corrected − broken) / N;
  - the mean reduction of |stage error| over all N.
- **Breakdown:** by transition (reference stage → final stage).

### Card
- Both indicators are shown for the subject.
- For a firing subject, the transition's train out-of-fold counts (A) are shown
  if that transition fired ≥ 10 times on train. Otherwise the card says "too
  few training cases (n) to judge".
- B is labelled "descriptive" unless the held-out val result exists.
- The existing discordance flag is retained, labelled exploratory, and
  tabulated the same way on train out-of-fold predictions.

## 7. P5 — Evaluation table and thresholds

**Rows:** CNN region attribution (chosen stage), MRI-specialist SHAP, clinical
SHAP (CatBoost ShapValues of the 5 frozen seeds) and KAN Shapley (25 frozen
KANs).

**Replication:** train is primary, and val is the replication. A PASS that is
not replicated on val is downgraded to WEAK.

### Stability
- **Measure:** the median pairwise Spearman correlation of global importance
  between seeds, models or refits. Global importance is the mean |attribution|
  per feature or region over the split, averaged over the four class targets
  (the CNN has a single target).
- **Sources:**
  - CNN: region d_r of each model's own maps. On train these are the 3 seed
    models within each fold (15 pairs); on val, all 15 models (105 pairs).
  - MRI: the 200 refits (P2-R4).
  - Clinical: the 5 seeds, over the union of their top-12 features.
  - KAN: the 25 models, over the 13 inputs.
- **Verdict:**
  - PASS if ≥ 0.80;
  - WEAK if 0.60–0.80;
  - FAIL if < 0.60.
- The per-subject median Spearman is reported too.

### Plausibility
- **Expected directions,** relative to the CN-train median, with ties excluded:
  - CDRSB above → pushes toward AD;
  - MMSE below → toward AD;
  - hippocampus/ICV below → toward AD;
  - ventricles/ICV above → toward AD;
  - and the converse on the other side of the median.
- **Tabular rows:** the fraction of train subjects (non-missing, |φ| > 1e-6)
  whose φ on the AD centred target has the expected sign. For the KAN this uses
  the per-input φ of CDRSB and MMSE, as the 25-model mean.
- **CNN row:** the fraction of train AD subjects with d_r > 1 for the
  hippocampus, and separately for the ventricles.
- **Verdict, per input:**
  - PASS if ≥ 0.80;
  - WEAK if 0.60–0.80;
  - FAIL if < 0.60.
- N/A where the explainer has no such input.

### Faithfulness
- **Measure:** deletion and insertion curves on the probability shown on the
  card:
  - CNN: the mean ALZ probability over the subject's models;
  - MRI: P(predicted class);
  - clinical: the 5-seed mean P(predicted class);
  - KAN: the ensemble P(final predicted class).
- **Tabular rows:**
  - Features are removed, or inserted, in descending order of their φ toward
    the explained class.
  - A removed feature is replaced by the interventional expectation over 50
    train background rows (`default_rng(0)`).
  - The comparison is against 50 random orders.
  - Clinical uses the union of the seeds' top-12 features; MRI uses its 10
    inputs; the KAN uses its 13 inputs.
- **CNN row:**
  - 0, 2, 5, 10, 20, 30, 40 and 50% of brain voxels are replaced with the CN
    mean in attribution order. Insertion starts from the CN mean.
  - The comparison is against 10 smoothed-noise orders, with the chosen
    stage's sigma.
- **Statistic:** the per-subject difference in normalised curve area
  (trapezoid over the step or fraction axis, divided by its length), attribution
  minus random.
- **Verdict:**
  - deletion PASS if the CI upper bound is < 0;
  - insertion PASS if the CI lower bound is > 0;
  - faithfulness PASS if both pass, WEAK if one passes, FAIL if neither does.
- **CNN faithfulness set:** the primary set is val AD (23 subjects, not used in
  stage selection). Train AD is reported, but it is selection-biased.

### Sanity checks
CNN only: P1-R4.

### Overall
- VALIDATED iff every applicable cell is PASS. Otherwise WEAK, listing the weak
  or failed cells.
- The result is written to `results/eval_status.json` and shown on the card.

## Appendix A — DK parcels to lobes

| Lobe | DK parcels |
|---|---|
| Frontal | superiorfrontal, rostralmiddlefrontal, caudalmiddlefrontal, parsopercularis, parstriangularis, parsorbitalis, lateralorbitofrontal, medialorbitofrontal, precentral, paracentral, frontalpole |
| Parietal | superiorparietal, inferiorparietal, supramarginal, postcentral, precuneus |
| Temporal | superiortemporal, middletemporal, inferiortemporal, bankssts, fusiform, transversetemporal, entorhinal, temporalpole, parahippocampal |
| Occipital | lateraloccipital, lingual, cuneus, pericalcarine |
| Other | rostralanteriorcingulate, caudalanteriorcingulate, posteriorcingulate, isthmuscingulate, insula |

## Appendix B — Sign-off decisions (2026-10-05)

- **Q1:** both "Fusion changed the stage" (P4-A) and the MRI-masked "MRI changed
  the stage" (P4-B).
- **Q2:** stability from bootstrap refits of an in-memory copy (P2-R4).
- **Q3:** the train-only KANs are regenerated in memory behind a reproduction
  gate (P3-R3). Nothing is written to disk.
- **Q4:** `docs/project_summary.md` is not created by this work. If it exists
  after execution, its §5, §7 and §10 are updated.
- **Q5:** the candidate stages are layer1, layer2 and layer4 only.
