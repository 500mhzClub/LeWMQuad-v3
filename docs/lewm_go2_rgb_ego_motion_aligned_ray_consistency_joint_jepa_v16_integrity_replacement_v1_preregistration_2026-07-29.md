# V16 integrity replacement V1 preregistration

Date: 2026-07-29

Status: preregistered science-identical integrity replacement only. No
replacement reservation, scientific input access, GPU training, checkpoint,
qualification, calibration, G2, navigation, or held-out access has occurred.

## Trigger and terminal predecessor

The original V16 attempt is terminal in commit
`12d5b77d2707f4cf263786286ba6118880e835c0` at
`docs/lewm_go2_rgb_ego_motion_aligned_ray_consistency_joint_jepa_v16_terminal_failure_result_2026-07-29.md`,
SHA-256
`99343e18fba804fd93f752aa69a644926eb7eacfe6161640d0062e0589a9f6eb`,
byte count `2856`.

It completed initialization and update-0 evaluation, then failed at
`train_update_1` before its first optimizer or EMA update. No scientific gate
or recovery/development checkpoint occurred. The immutable exception-message
SHA-256
`9d0a8542eeef7e9cbaca31ac20af5086142eab97506a7876a3608b4088f86faf`
exactly identifies `current_evidence floating evidence must be float32`.

The source cause is closed: V16 correctly consumes float32 learned hazards,
within-bin offsets, and ground-survival logits, but its adapter also required
the existing, unused V14 query-geometry UV and distance metadata to be
float32. Those metadata are intentionally float64. A source-only
production-shape reproduction confirmed the exact dtype split without opening
scientific data.

The original attempt therefore did not test the V16 mechanism. This is a
distinct science-identical integrity replacement, not a retry or resume of the
consumed attempt.

## Sole implementation correction

- Keep the exact float32 requirement for the learned hazard, offset, and
  ground-survival tensors consumed by the V16 loss.
- Accept the inherited float64 query-geometry UV and target-distance metadata;
  retain their exact shapes/device validation. They are not inputs to the V16
  raster, warp, mask, weight, or KL computation.
- Add one source-only production-shape regression proving that exact V14
  auxiliary evidence reaches the V16 helper and returns finite loss/support.
- Change only replacement evidence selectors, preregistration identity,
  schema prefix, and fresh output root required to distinguish this attempt.

No other validator, tensor operation, model parameter, architecture, data
field, loss, coefficient, seed, schedule element, optimizer setting, EMA
operation, observation, metric, control, threshold, stopping rule, or
accounting multiplier may change.

## Frozen scientific identity

Preserve original V16 preregistration commit
`2792343e14d3376add9d6adbda7f29346a3e9e29` exactly:

- unchanged V14 RGB-only model, online encoder, unified ray-survival evidence,
  semantic state, action-conditioned predictor, and EMA target;
- unchanged symmetric stop-gradient metric ray-consistency loss
  `C = C_base + 0.1*M` in the same joint-JEPA update;
- unchanged training-only realized SE(2), with no motion input at inference;
- fresh N320 initialization and the same constructor, schedule, experiment,
  bootstrap, and projection seeds;
- the same 4,262-pair train and 495-pair checkpoint-selection roles and exact
  first 16,000-presentation schedule;
- four microbatches of four, float32 AdamW, parameter groups, learning rates,
  clipping, one optimizer step, and one EMA update per completed update;
- observations at updates `0`, `100`, `400`, and `1000`;
- the exact V16 update-400, final, and continuation-eligibility gates;
- milestone full-state recovery publication only after a passing update-400
  gate and an eligible or fully passing update-1000 gate; and
- exactly 1,000 maximum updates and 16,000 maximum presentations.

There is one replacement seed and one replacement attempt. There is no loss
search, topology change, automatic retry, automatic resume, or automatic
extension.

## Replacement identity and authority

- Schema/evidence prefix:
  `lewm_go2_rgb_ego_motion_aligned_ray_consistency_joint_jepa_v16_integrity_replacement_v1`.
- Fresh attempt root:
  `.generated/go2_rgb_ego_motion_aligned_ray_consistency_joint_jepa_v16_integrity_replacement_v1/attempt_v1`.
- The original V16 output is closed; only its committed terminal result is
  admissible as source-only identity evidence.
- Source implementation, focused regression, recursive closure, independent
  review, narrow clean-export certification, and one-shot authority must be
  frozen before reservation or scientific input access.
- Any replacement source, authority, reservation, custody, exception, or gate
  failure is terminal. A second integrity replacement is not preregistered.

Probability calibration, G2, navigation, held-out, sealed, production,
promotion, deployment, retry, resume, and extension remain unauthorized unless
earned later under separate ordered authority.
