# V17 delayed-onset ego-motion-aligned ray-consistency joint-JEPA preregistration

Date: 2026-07-29

Status: preregistered one-shot development probe only. No V17 reservation,
scientific input access, GPU training, checkpoint, probability calibration,
G2, navigation, or held-out access has occurred.

## Trigger and scientific question

V16 integrity replacement V2 is terminal in commit
`5f5092f528beafe5c3c8ded67b0e368f1a2d992e` at
`docs/lewm_go2_rgb_ego_motion_aligned_ray_consistency_joint_jepa_v16_integrity_replacement_v2_scientific_result_2026-07-29.json`,
file SHA-256
`e858c0473ea6f159a1697732d658ae495ec5ee88813d2dd9b5f4769ea5ef92e8`,
content SHA-256
`3506a34b3afea85b56eab1ee749e10fd83d7104fd8ff9fe8129d9db5cbcb59c2`,
and byte count `10681`.

V16 was a valid scientific negative at update 400. Structural integrity and
all twelve causal-control checks passed, but it achieved only 71 physical
margins, total shortfall 70.02022160058146, and rough depth p95
1.929633975028991 m against exact requirements of at least 72,
less than 68.96964862816927, and less than 1.8582415819168085 m. It was
slightly worse than the matched V14/V15 prefix on every selected physical
summary at updates 100 and 400. V16 is consumed and may not be retried,
resumed, extended, or used as initialization.

The single V17 hypothesis is that applying self-consistency from update 1
made two immature, inaccurate learned ray fields agree before supervised
camera grounding stabilized. V17 tests whether delaying the unchanged
consistency term until after the update-100 observation removes that early
interference and produces strict improvement by update 400.

## Sole scientific change

For completed training update `u`, use the fixed ray-consistency coefficient

`w_M(u) = 0.0` for `1 <= u <= 100`, and

`w_M(u) = 0.1` for `101 <= u <= 1000`.

The Camera loss is exactly `C = C_base + w_M(u)*M`. The exact frozen V16
rasterization, SE(2) warp, symmetric stop-gradient Bernoulli KL, masks,
weights, support checks, and receipt are still computed at every update.
Before onset, `M` contributes zero parameter gradient; from update 101 it is
the unchanged V16 term at coefficient 0.1.

There is no ramp, coefficient search, alternative onset, EMA-teacher change,
architecture change, topology change, extra head, new supervision, loss
replacement, threshold change, or horizon extension.

## Frozen scientific identity

Preserve the reviewed V16 model and execution identity exactly except for the
coefficient schedule above:

- unchanged V14 RGB-only online encoder, unified ray-survival evidence,
  semantic state, action-conditioned predictor, and EMA target;
- unchanged learned float32 hazard/offset/ground evidence and inherited
  float64 query-geometry validation;
- unchanged training-only realized SE(2), with RGB-only inference;
- fresh N320 initialization and the same constructor, experiment, bootstrap,
  projection, and schedule seeds;
- the same 4,262-pair train and 495-pair checkpoint-selection roles and exact
  first 16,000-presentation order;
- four microbatches of four, float32 AdamW, parameter groups, learning rates,
  clipping, one optimizer step, and one EMA update per completed update;
- unchanged Navigation joint-JEPA loss `N`, Camera base loss `C_base`, gradient
  routes, target stop-gradient behavior, and accounting multipliers;
- observations at updates `0`, `100`, `400`, and `1000`;
- the exact V16 update-400, final, and continuation-eligibility gates;
- full-state recovery publication only after a passing update-400 gate and an
  eligible or fully passing update-1000 gate; and
- exactly 1,000 maximum updates and 16,000 maximum presentations.

Update 100 is a fixed pre-onset control. V17 must then beat the original V16
update-400 comparators without any threshold relaxation. If any update-400
conjunct fails, V17 terminates and the temporal ray-consistency family is
closed; no second timing or coefficient variant is authorized.

## Identity and authority

- Schema/evidence prefix:
  `lewm_go2_rgb_delayed_onset_ego_motion_aligned_ray_consistency_joint_jepa_v17`.
- Fresh certified source root:
  `/home/andrewknowles/Workspace/LeWMQuad-v3-v17-delayed-ray-consistency-source`.
- Fresh attempt root:
  `.generated/go2_rgb_delayed_onset_ego_motion_aligned_ray_consistency_joint_jepa_v17/attempt_v1`.
- Original V16 and both integrity-replacement outputs remain terminal
  documentary evidence only. No model, optimizer, EMA, RNG, trace, metric, or
  checkpoint state may be reused.
- Source implementation, focused boundary tests, recursive closure,
  independent review, narrow clean-export certification, and one-shot
  authority must be frozen before reservation or scientific input access.
- Any source, authority, reservation, custody, exception, or gate failure is
  terminal. Retry, resume, alternate onset, coefficient search, and automatic
  extension are unauthorized.

Probability calibration, G2, navigation, held-out, sealed, production,
promotion, deployment, and any further temporal-consistency experiment remain
unauthorized unless separately earned after this terminal result.
