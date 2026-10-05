# Dry run and go/no-go (2026-10-05)

Dry-run set (pre-registered): the first 3 AD, 3 CN, 2 EMCI and 2 LMCI train subjects with a scan, by PTID.
All outputs went to `cache/dry/` (gitignored). The dry run was checked for execution only; no rule or
threshold was changed after it.

**Scope note.** The tabular scripts (P2, P3 cells and stability, P4, P5 clinical) are cheap, and their dry
mode ran over all train and val subjects for the parts that do not depend on the dry set. Those outputs
also went to `cache/dry/` only. Every rule governing them was committed beforehand (`050ca8e`).

| Criterion | Result |
|---|---|
| Every script exits 0; manifest verifies before and after | yes: p1_cam, p1_select_stage, p1_regions, p1_sanity, p1_eval_cnn, p2_mri_shap, p3_kan_shapley, p4_stage_change, p5_clinical, p5_table |
| P2 additivity < 1e-6; LinearExplainer within 1e-8 | 1.1e-15; 0.0 |
| P3 efficiency < 1e-4; brute force within 1e-4 | 1.0e-6; 2.7e-7 (10 subjects × 3 models) |
| Maps: expected shapes, no NaN, positive mass > 0 at every stage | LayerCAM: yes at all 3 stages. Grad-CAM (reference only) gives all-zero maps at layer4 for 3 of 30 subject-model pairs, which is expected when the weighted evidence is negative |
| Peak RAM < 10 GB; VRAM < 7 GB | 1.7 GB; 0.6 GB |
| Extrapolated full runtime < 4 h | about 45 min |
| Test-RID guard exercised | yes: `assert_no_test`, plus the identifier and split guards in `write_aggregate`, all fire on a test RID |
| Atlas QC | Dice vs mask 0.989 (≥ 0.90); smoke-subject hippocampus Dice median 0.66 |
| Train-only KAN reproduction gate (seed 42 in the dry run) | max |Δp| 0.0, max |ΔADD| 0.0 |

**Observation (not acted on):** layer1 LayerCAM maps show a regular lattice with a period of 2 cells.
This is the checkerboard pattern that stride-2 convolutions leave in gradients. The method is unchanged,
as pre-registered. The sanity and faithfulness checks judge the maps.

**Decision: GO.**
