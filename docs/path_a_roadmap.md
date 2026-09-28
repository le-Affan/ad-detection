# Path A Roadmap — Multi-Visit MRI Progression Modeling

Last updated: 2026-09-26

## Where things stand

Multi-visit MRI data ingestion is complete:
- Cohort `patha_bl_m6_m12_m24` (IDA ID 521634): 690 subjects, slots bl/m06/m12/m24,
  5,227 images
- Extracted at `/home/ankon/projects/LYP/datasetnew_extracted/`: 4,231 NIfTI files,
  690 subject folders
- Master index: `MANIFEST_patha_followup.csv` — 2,573 rows (one per subject/slot),
  99.53% OK / 0.47% MISSING_NOT_IN_COHORT
- `dataset/` and `dataset/mri_raw/` (original baseline-only data) left untouched
- Raw source zips (~154GB) deleted after verification, once extraction was confirmed

Known findings to keep in mind:
- Visit labeling on IDA is not a flat VISCODE2 scheme — it branches on New-Pt vs.
  Continuing-Pt status (New-Pt: Year 1 Visit = m12, Year 2 Visit = m24;
  Continuing-Pt: ADNI2 Initial Visit-Cont Pt = m12, Year 1 Visit = m24). Should be
  documented in `docs/` if not already.
- m36 was dropped from the slot schedule — only 118/690 subjects had it (real
  ADNI2 vs. ADNIGO protocol split, not a data gap).
- Matching used exact VISCODE2 as primary key, ±60-day date fallback only for
  subjects missing the exact code (fallback fired on 0.3% of matched scans).
- 082_S_4224/m24 has a corrupted duplicate series (I1051042) still on disk — the
  manifest deliberately routes around it to the good series (I1051225).
- 7 (subject, slot) pairs from the original target list were never found on IDA —
  likely recatalogued under a different ADNI phase (unconfirmed hypothesis).

Still unbuilt/unconfirmed from earlier audit:
- DEMO head Plan A fix (`m4_train.py`, equal-interval relabeling, freq-weighted
  MSE) — does not exist yet, independent of Path A.
- Path A gate (`check_patha_gate.py`) — does not exist yet. Was meant to require
  the static 4-class COG head to clear F1 thresholds on AD/LMCI before building
  Path A. COG head is currently weak (~0.36–0.44 acc, near majority-class floor).
  **This needs an explicit decision (Stage D below) before/alongside Path A
  training — not a silent bypass.**
- KAN fusion layer — entirely unbuilt.
- Grad-CAM report language — claims hippocampal/temporal attribution, actual
  heatmaps show periventricular attribution. Needs correcting before defense.

## Roadmap

| Stage | What it is | Depends on | Expected output |
|---|---|---|---|
| **A. `select_scans.py` multi-visit rewrite** | Replace single-baseline selection with manifest-driven multi-visit loading — read `MANIFEST_patha_followup.csv`, produce per-subject sequences of (slot → scan path), with masking for subjects missing a slot | Manifest (done) | Per-subject visit-sequence structure consumable by `dataset.py` — correctly shaped data, no training yet |
| **B. Preprocess new scans** | Run existing `m1_preprocess.py` pipeline (N4 → registration → skull-strip → clip → downsample → normalize) on newly-ingested `.nii` files | Can run in parallel with A — needs file list, not the sequence structure | Preprocessed tensors for ~2,561 new scans, same shape/format as existing baseline-only preprocessed data |
| **C. Temporal architecture** | Extend `DualHeadNet` with a temporal component over the visit sequence (shared CNN backbone per visit + GRU/transformer/embedding-delta) | A + B | Trainable multi-visit model — architecturally complete, unvalidated |
| **D. Gate decision — fix or bypass?** | Explicit decision: (a) improve static COG head first and formally check the Path A gate, or (b) consciously proceed to Path A anyway, gate treated as advisory/skipped — documented either way | Can be decided any time before E | Written decision + rationale (user/guide call, not automated) |
| **E. Train Path A model** | Actual training run of the temporal model on the multi-visit dataset | C, D | Metrics on whether multi-visit input improves on the weak static baseline — the real test of Path A |
| **F. DEMO head fix (Plan A)** | Independent, smaller fix — equal-interval relabeling + freq-weighted MSE | Independent of Path A | Fixed DEMO regression head, checked against documented pass/fail thresholds |
| **G. KAN fusion layer** | Combine clinical + MRI (incl. Path A signal) into final COG/ADD dual-output fusion | Clinical (done), MRI stream stable (gated on E), likely gated on F | End-to-end multimodal model — capstone centerpiece, currently unbuilt |
| **H. Grad-CAM/report correction** | Fix claimed hippocampal/temporal attribution to match actual periventricular attribution | Independent | Corrected slides/report language |
| **I. Defense prep** | Q&A docs, numbers-to-know-cold, trap questions, literature cross-reference | Everything above, ideally | Defense-ready materials |

## Open risk worth flagging

Path A assumes multi-visit temporal signal will fix what the weak static 4-class
head couldn't. That assumption is untested. If the COG head's weakness is a
data/label issue rather than "insufficient temporal context," Path A's
architecture won't fix it — worth a quick diagnostic pass on *why* the COG head
is stuck near the floor before or alongside Stage C, since that could change
what Stage E's results actually mean.

## Immediate next action

Stage A (`select_scans.py` / manifest-consumption rewrite), since B and C can't
meaningfully proceed without it. B can start in parallel once the raw file list
is available.
