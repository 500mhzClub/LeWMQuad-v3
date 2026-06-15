# Phase 2O Factorized Primitive Affordance Targets

Date registered: 2026-06-15

Status: target contract implemented and audited; train and validation data only;
no `test_id` or `test_hard` access.

## Trigger

Phase 2M and Phase 2N both failed the primitive-affordance promotion gate.
Phase 2M showed that source images contain some immediate-action signal, but
the scalar primitive utility objective produced worse regret than the
source-independent primitive prior. Phase 2N showed that class balancing did
not solve the problem; it slightly improved primitive match but made utility
regret worse.

The next target must represent why a primitive is good or bad, not only which
primitive was the scalar oracle winner.

## Target Contract

Phase 2O builds factorized first-primitive targets from the existing generator
consequence labels. For each source state and first primitive, it uses the
utility-best continuation from the Phase 2M contract, then records:

```text
safe_recoverable
task_gain_norm
p05_clearance_norm
minimum_clearance_norm
unsafe_sample_fraction
heading_alignment
```

Target schema:

```text
phase2o_factorized_first_primitive_affordance_v0
```

Core factors:

```text
safe_recoverable
task_gain_norm
p05_clearance_norm
minimum_clearance_norm
unsafe_sample_fraction
```

`heading_alignment` remains an optional tie-breaker because some rows do not
have a defined target heading.

## Implemented Files

```text
lewm/benchmarks/phase2o_factorized_affordance.py
scripts/audit_jepa_phase2o_factorized_affordance.py
lewm/tests/test_phase2o_factorized_affordance.py
```

## Focused Quality Gate

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .generated/venvs/genesis_render_vulkan/bin/python \
  -m pytest lewm/tests/test_phase2o_factorized_affordance.py \
  lewm/tests/test_phase2m_primitive_affordance.py -q

Result: 8 passed
```

## Real-Data Audit Command

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .generated/venvs/genesis_render_vulkan/bin/python \
  scripts/audit_jepa_phase2o_factorized_affordance.py \
  --train-data .generated/jepa_counterfactual/phase2d_min_sources/train_spatial_future_v1.jsonl \
  --validation-data .generated/jepa_counterfactual/phase2d_min_sources/validation_spatial_future_v1.jsonl \
  --output .generated/jepa_counterfactual/phase2d_min_sources/phase2o_factorized_affordance_audit.json
```

## Audit Result

```text
train source states: 512
validation source states: 256
train valid primitive targets: 4,608
validation valid primitive targets: 2,304
train core factors complete: true
validation core factors complete: true
train all factors complete: false
validation all factors complete: false
```

The only incomplete factor is `heading_alignment`:

```text
train heading_alignment labels: 3,564 / 4,608
validation heading_alignment labels: 1,656 / 2,304
```

All safety, task-gain, clearance, and unsafe-fraction factors are complete for
every source/primitive target.

Safe-positive counts by primitive on validation:

```text
arc_left: 97 / 256
arc_right: 93 / 256
backward: 133 / 256
forward_fast: 79 / 256
forward_medium: 86 / 256
forward_slow: 98 / 256
hold: 140 / 256
yaw_left: 140 / 256
yaw_right: 140 / 256
```

Artifact hash:

```text
5e0951c02e86e1466d12f1264d9064de830f141044fee750d3387a5a7609bdb1  .generated/jepa_counterfactual/phase2d_min_sources/phase2o_factorized_affordance_audit.json
```

## Decision

The Phase 2O factorized target contract is viable for a bounded model pilot.
It does not require new data generation.

The next bounded implementation should train a factorized primitive affordance
head and select primitives with an explicit safety-first score:

1. reject or heavily penalize predicted unsafe/unrecoverable primitives;
2. among predicted-safe primitives, rank by clearance and task gain;
3. use heading alignment only where the target exists.

Promotion must still use the Phase 2M/2N primitive gate:

- primitive match rate at least `0.50`;
- regret below the source-independent primitive prior;
- selected primitive distribution not more collapsed than oracle by more than
  `0.20`;
- no `test_id` or `test_hard` access.
