# Go2 categorical-radial N32 V2 result

Date: 2026-07-11

Status: authoritative seed 20260710 completed; fit gate failed; seed 20260711,
train-role holdouts, full training, G2, calibration, runtime, and promotion remain
unauthorized.

## Bound experiment

The run followed the frozen exposure-matched binding in
`docs/lewm_go2_categorical_radial_n32_v2_exposure_binding_2026-07-11.md`
(SHA-256
`4164ec011910cb2d1d2fbea5beaad81eb13ea6b506e063ebf13a66a41e14fb6f`).
It kept the V1 N32 model, data, controls, and gates fixed while transferring the
successful N16 schedule dimensionlessly: batch 80, 2,000 optimizer updates,
500 effective epochs, and the V3 cosine learning-rate schedule from `2e-4` to
`1e-5`.

Authoritative artifact:

- path:
  `.generated/go2_categorical_radial_n32/v2/seed_20260710_result.json`;
- file SHA-256:
  `0a5f8a822d7fec8287a30103125fca1a4927f0413e2f0906db431cef54ec2265`;
- canonical content SHA-256:
  `e070cc96d69b76e1f85f533fa1d94221225963a2b66a491f0c2a867c008b97ef`;
- execution: AMD Radeon AI PRO R9700, FP32, 2026-07-11 00:12:37 to
  03:01:38 UTC;
- model: unchanged full-ray categorical-radial model, 2,887,067 parameters.

The independent strict V2 finalizer accepted the artifact, source commitments,
schedule, controls, access reconciliation, and decision.

## Result

The terminal fit rule failed at steps 1,800, 1,900, and 2,000. Aggregate
performance passed every fixed threshold at all three terminal evaluations,
but the contract also requires every scene family to pass.

At step 2,000 the aggregate correct-view metrics were:

- hierarchical balanced NLL: `0.011052` (gate `<= 0.03`);
- UNKNOWN/known balanced accuracy: `0.992535` (gate `>= 0.99`);
- FREE/OCCUPIED balanced accuracy: `0.999874` (gate `>= 0.99`);
- UNKNOWN/FREE/OCCUPIED recall: `0.986321 / 0.999588 / 0.987779`;
- FREE recall at 1-2 m, 2-3 m, and beyond 3 m:
  `0.999718 / 0.999616 / 0.999538`.

Three of five scene families passed every gate. The stable terminal failures
were:

- `open_obstacle_field`: OCCUPIED recall `0.978723` at step 2,000, with
  598/611 occupied cells correct; the fixed threshold is `0.98`;
- `rough_local_dynamics`: UNKNOWN recall `0.974518` and UNKNOWN/known balanced
  accuracy `0.986932`; the fixed thresholds are `0.98` and `0.99`.

The same two failures were present at steps 1,800 and 1,900. Correct-view NLL
continued to improve, and wrong-view controls remained strongly separated, so
this is neither output collapse nor a failure to bind RGB to geometry.

## Access and decision

The run decoded exactly 320 fit images and opened exactly 20 fit label shards.
It recorded 179,200 fit-image requests and 166,400 fit-target requests. It
opened zero same-scene holdout, cross-scene holdout, checkpoint-selection,
calibration, non-train, G2, or sealed payload bytes and produced zero outputs
on those roles.

The frozen decision is therefore `fit_gate_failed` and `favorable=false`.
Seed 20260711 is forbidden, and no categorical-radial full-training candidate
is licensed. Per the V2 binding, the exhausted exposure-only hypothesis now
licenses one preregistered representation/capacity intervention at N32. That
intervention must preserve the data roles, causal wrong-view controls, exact
terminal gates, and conditional holdout rule, and must be frozen before its
first dataset-backed model output.
