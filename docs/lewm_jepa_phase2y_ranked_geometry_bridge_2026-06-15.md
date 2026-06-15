# Phase 2Y Ranked Geometry Bridge

Date registered: 2026-06-15

Status: failed bounded ROCm GPU smoke; train and validation data only; no
`test_id` or `test_hard` metric use.

## Trigger

Phase 2X showed that single-frame RGB can reconstruct the sanitized swept-state
target within coarse error thresholds but still cannot select useful first
primitives. Phase 2Y tests a narrower control: can explicit local metric
geometry, similar to a deployable depth/ray observation, pass the primitive gate
when given the same light primitive-ranking objective that helped Phase 2W?

## Implementation

The existing Phase 2R geometry trainer now supports an optional primitive
ranking loss:

```text
--primitive-ranking-loss-weight
--primitive-ranking-regression-weight
```

Defaults preserve Phase 2R behavior. The Phase 2Y smoke used:

```text
geometry features: 16 local rays + source pose + goal-relative geometry
primitive-ranking-loss-weight: 0.10
optimization_steps: 512
device: cuda
```

## Result

The ROCm GPU smoke completed with finite metrics but failed:

```text
primitive_match_rate: 0.390625
mean_target_utility_regret: 0.097959344
selected_max_primitive_fraction: 0.328125
oracle_max_primitive_fraction: 0.3515625
```

Primitive action-only prior:

```text
primitive_match_rate: 0.1640625
mean_target_utility_regret: 0.058599013
selected_max_primitive_fraction: 1.0
```

Gate:

```text
passed: false
failure_reasons:
  - primitive_match_rate_below_threshold
  - regret_not_below_action_only_baseline
```

## Interpretation

Adding a ranking term to source-local metric geometry did not make the geometry
bridge pass. This weakens the idea that a small local ray vector plus source and
goal pose is enough.

Together with Phase 2W and Phase 2X, the current evidence says:

- the sanitized swept-geometry teacher target is usable when the action-conditioned
  consequence geometry is supplied;
- single-frame RGB does not recover that target well enough for action choice;
- source-local metric rays without action-conditioned swept rollout are still
  insufficient.

The next candidate should use a richer deployable state: temporal memory,
occupancy/local map features, explicit depth-derived swept occupancy, or
factorized slots that are supervised by the Phase 2W sanitized target.

## Artifact Hashes

```text
233bb1d866330870ac8a00569256f38239878705bd0a3738473f1483a74425c3  phase2y_geometry_affordance_rank010_smoke.json
47ae424c7e11249de1c32e844b6d495e44e1854e8987b90fe92f1ec0c343c72f  phase2y_geometry_affordance_rank010_smoke_gate.json
```
