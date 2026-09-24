# Go2 categorical-radial N32 V3 result

Date: 2026-07-11

Status: authoritative seed 20260710 completed; fit gate failed; seed 20260711,
both train-role holdouts, full training, G2, calibration, runtime, and promotion
remain unauthorized.

## Bound experiment

The run followed the frozen token-width binding in
`docs/lewm_go2_categorical_radial_n32_v3_token_width_binding_2026-07-11.md`
(SHA-256
`a9898d349d82f65ce35443192b555aac4386136032c8fe70c115eda5a788a5ad`)
and the pre-run implementation manifest in
`docs/lewm_go2_categorical_radial_n32_v3_implementation_manifest_2026-07-11.md`
(SHA-256
`200c1d9d8944fb0252828b659f1c32c6176cf7d32d2e4e89cf10abb5d2ca1877`).

The sole intervention was `token_feature_dim: 24 -> 32`. The 64-channel
context, encoder, geometry, factorization, data, loss, controls, optimizer,
500-epoch exposure, and exact gates remained fixed. Of 133 state entries, 130
were bit-identical to the seed-corresponding V2 initialization and exactly
three changed shape. No trained V2 state was loaded.

Authoritative artifact:

- path:
  `.generated/go2_categorical_radial_n32/v3/seed_20260710_result.json`;
- file SHA-256:
  `0f3eb212afe54a38d7a81a1fc51ca544dfab667a94a836be742d3ea3e2298d85`;
- canonical content SHA-256:
  `ec8dd8450fb34bee3a5ba1c5a5b532339d281241560c8ed9ac07a48d2c2bea4e`;
- execution: AMD Radeon AI PRO R9700, direct batch 80, FP32, 2,000 updates,
  500 effective epochs;
- model: 2,891,171 parameters.

The independent strict V3 finalizer accepted the artifact, bound evidence,
32-entry source map, initialization proof, schedules, controls, access
reconciliation, recomputed gates, and decision.

## Result

The aggregate fit report passed every fixed threshold at steps 1,800, 1,900,
and 2,000, but the mandatory all-family rule failed at all three evaluations.

At step 2,000 the aggregate correct-view metrics were:

- hierarchical balanced NLL: `0.012135`;
- UNKNOWN/known balanced accuracy: `0.991675`;
- FREE/OCCUPIED balanced accuracy: `0.999853`;
- UNKNOWN/FREE/OCCUPIED recall: `0.985061 / 0.999512 / 0.982108`.

Only `medium_enclosed_maze` passed every family gate at step 2,000. The
remaining failures were:

- `large_enclosed_maze`: OCCUPIED recall `0.979255`;
- `open_obstacle_field`: OCCUPIED recall `0.960720` and UNKNOWN/known balanced
  accuracy `0.989983`;
- `rough_local_dynamics`: UNKNOWN recall `0.972079` and UNKNOWN/known balanced
  accuracy `0.985557`;
- `small_enclosed_maze`: OCCUPIED recall `0.975620`.

Correct-view metrics continued improving late in training and wrong-view
controls remained separated, but the fixed terminal requirement was not met.

## Comparison and interpretation

The width-32 candidate underperformed the width-24 V2 reference at the final
checkpoint:

- aggregate NLL: `0.011052 -> 0.012135`;
- UNKNOWN recall: `0.986321 -> 0.985061`;
- OCCUPIED recall: `0.987779 -> 0.982108`;
- rough-terrain UNKNOWN recall: `0.974518 -> 0.972079`;
- open-field OCCUPIED recall: `0.978723 -> 0.960720`.

This rejects the specific hypothesis that the learned `192 -> 24` projection
was discarding the visual cues responsible for the N32 residual. It does not
show that arbitrary extra capacity cannot help, but it removes token width as
the next rational lever. The stable failure is still visibility/occlusion
classification along rays, especially UNKNOWN versus observed space in rough
terrain. A new experiment must be justified and frozen around that structured
failure rather than adding exposure or selecting another width post hoc.

## Access and decision

The run decoded exactly 320 fit images and opened exactly 20 fit label shards.
It recorded 179,200 fit-image requests and 166,400 fit-target requests. It
opened zero same-scene holdout, cross-scene holdout, checkpoint-selection,
calibration, non-train, G2, or sealed payload bytes and produced zero outputs
on those roles.

The immutable decision is `fit_gate_failed` and `favorable=false`. Seed
20260711 is forbidden. No shared-JEPA full-training candidate, G2 evaluation,
runtime, G3, or promotion license exists.
