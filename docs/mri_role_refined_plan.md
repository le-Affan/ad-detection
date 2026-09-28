# MRI Stream Role — Refined Plan

Date: 2026-09-27
Status: agreed direction; the fusion feasibility pass (Section 7) is the next task.

## 1. Decision in one paragraph

Stop pursuing 4-class CN/EMCI/LMCI/AD staging from MRI. The MRI stream's job
is to supply **independent anatomical evidence**, mainly an **AD probability
(ALZ)**, to the fusion layer. The fusion layer produces the 4-class COG output
and the continuous severity score, as the original design intended. MRI's
contribution is measured by the project's own ablation: clinical-only vs.
clinical + MRI signals.

## 2. Why: the design never required MRI staging

The documented architecture has the MRI stream pass exactly two values to
fusion:

- **DEMO**: continuous severity, 0–2
- **ALZ**: AD probability, 0–1

The fused vector is `[DEMO, ALZ, clinical top-K]`. The **4-class output
belongs to the KAN fusion layer**, not to the MRI stream. `DualHeadNet` has no
4-class head. The "acc4" tracked in MRI experiments is DEMO regression binned
into 4 classes. It's a proxy for DEMO quality, and it was never a requirement.

## 3. Why MRI 4-class staging hits a ceiling

In ADNI, CN/EMCI/LMCI/AD are **defined by cognitive tests**: MMSE, CDR and
Logical Memory delayed recall. The EMCI/LMCI split is essentially a cutoff on
a memory test score.

- The clinical stream stages well (acc 0.813, macro-F1 0.786) partly because
  its top features (CDRSB, MMSE, ADAS) are close to the variables that define
  the labels.
- MRI would have to predict a test-score boundary from anatomy. Early and late
  MCI differ little in structure, and published MRI-only EMCI results are
  modest even with much larger cohorts than ~480 training subjects.
- Both Path A experiments reached the same limit from different directions:
  - temporal modelling over visits did not beat the latest scan alone;
  - extra follow-up scans as training data did not help.
  Performance is bounded by the number of distinct subjects, not by
  architecture or scan count.

**Defense framing:** the weak MRI staging result is the expected ceiling for
cognitively defined labels at this cohort size. It is not a pipeline failure,
and two controlled experiments support that conclusion.

Related defense note: because the labels are defined by cognitive tests, the
clinical stream's accuracy partly reflects label definition. Expect a question
on this.

## 4. Where MRI can still add value

Clinical stream per-class F1: CN 1.0 · EMCI 0.82 · **LMCI 0.45** · AD 0.875.

- LMCI is the weak class. **Where LMCI errors go decides whether MRI can
  help COG:**
  - LMCI ↔ AD confusion: the boundary is partly anatomical, so ALZ can help.
  - LMCI ↔ EMCI confusion: the boundary is a memory score, so MRI won't help much.
- ADNI has no non-AD dementias, so the ADD head reduces to AD vs. everything
  else. That is exactly what ALZ targets.
- The continuous severity score is a **system-level** requirement (eval deck:
  "Compute a continuous disease severity index"). The fusion model's severity
  head produces it from clinical features plus ALZ. It does not have to come
  from MRI's DEMO head.

## 5. MRI signals: what to use

| Signal | Status | Role | Verdict |
|---|---|---|---|
| **ALZ probability** | Works, AD AUC ~0.76–0.82 | ADD head, AD/LMCI boundary, independent evidence | **Primary fusion input** |
| **DEMO** | Mean collapse, unfixed | Severity | Auxiliary training loss only. Fusion input only if the ablation shows value. Plan A fix is low priority |
| **Path C hippocampal atrophy slope** | Built and validated (stage gradient, p = 0.004) | Progression and severity reporting | **Not a baseline-diagnosis fusion input** (see 6.3). Use for the progression analysis and the "longitudinal" claim |
| **Latest-scan model** (Path A C1/C2) | Tentative, pending test | Dementia-at-last-visit target | Secondary finding |
| **4-class MRI staging** | At its expected ceiling | — | **Dropped as a goal** |

## 6. Technical traps (these must be handled before fusion)

### 6.1 In-sample ALZ scores are overfit
The CNN trained on the train split gives overconfident ALZ scores for those
subjects. A fusion model trained on them learns to trust ALZ too much, and that
collapses on test.

**Fix: cross-fitted ALZ.** Retrain the CNN k times (k = 5) over the train
split, each time predicting ALZ for the held-out fold. Every training subject
then has an ALZ score from a model that never saw it.

- A frozen-embedding shortcut is **not** enough. The cached trunk embeddings
  come from a trunk that was itself trained on the train subjects, so a k-fold
  head on top still leaks.
- Local static runs take ~3 min each, so 5 fold models are cheap.
- Validation and test subjects get ALZ from a model trained on the full train
  split (or the mean of the fold models; choose one and record it).

### 6.2 Split alignment between streams
The clinical stream has 814 subjects and its own split. The MRI stream has 811
subjects and the split in `mri_index.csv`. Two different MRI validation sets
have appeared in reports (104 and 121 subjects). If a fusion test subject was
a training subject in either stream, the fused test result is leaked.

**Fix: treat the clinical split as the master split.** Cross-fitting (6.1)
runs over the clinical train split, and the final ALZ model is trained on the
clinical train split. Clinical results stay unchanged, and alignment holds by
construction. Subjects without an MRI scan get a missing ALZ value, which
CatBoost handles natively and KAN will need a mask for.

### 6.3 Path C slopes use future visits
A subject's atrophy slope is computed from visits after baseline. For a
baseline-diagnosis target, that is information from the future. It isn't
available when a new patient is first assessed, so it can't be a deployable
input to baseline-diagnosis fusion. It stays valid for:
- predicting status at a later visit, with slopes computed only from visits up
  to that point;
- descriptive progression and severity analysis.

If it appears in the ablation at all, label it as a **future-information upper
bound**, not a candidate input.

### 6.4 The clinical stream already contains MRI volumes
FreeSurfer volumes (e.g. Fusiform, SHAP rank #9) are among the clinical
features. The "clinical-only" arm is therefore not imaging-free, which shrinks
the measured added value of the CNN. Report this in the ablation. It also
conflicts with the rule that clinical explanations never mention MRI, so that
rule's wording needs revisiting.

## 7. Next task: fusion feasibility pass (validation only, no test)

1. **Clinical confusion matrix** on validation: where do LMCI errors go?
2. **Split audit** across the clinical stream, `mri_index.csv`, and the
   validation sets used in MRI reports (104 vs. 121 subjects).
3. **Cross-fitted ALZ** for every clinical-train subject, plus ALZ for
   validation subjects from a full-train model.
4. **CatBoost ablation** (a quick check, not the KAN):
   - A: clinical only
   - B: clinical + ALZ
   - C: clinical + ALZ + DEMO (only if DEMO carries signal)
   - D (optional, labelled "upper bound, not deployable"): + Path C slope
   Evaluate on validation and by cross-validation over the train split,
   multiple seeds, with a paired comparison against A.

**Decision after the pass:**
- B clearly beats A → build the KAN with ALZ as the MRI input.
- No gain → report honestly: "MRI provides independent anatomical
  corroboration and an AD-specific signal, but little extra staging accuracy
  over cognitive scores." Then decide how much KAN work is still justified.

## 8. What changes elsewhere

- **Report/slides:** remove any claim that MRI performs stage classification.
  The MRI stream outputs ALZ (and DEMO as an auxiliary), and staging is
  fusion's job.
- **DEMO Plan A fix:** optional polish, not a blocker.
- **Path A:** reported as a well-controlled negative result on temporal
  modelling, plus the tentative latest-scan finding.
- **v9 reproducibility / class-weighting seed study:** its outcome decides the
  CNN configuration used for the cross-fitted ALZ models.
- **Grad-CAM wording** (periventricular, not hippocampal) still needs fixing
  before the defense.
