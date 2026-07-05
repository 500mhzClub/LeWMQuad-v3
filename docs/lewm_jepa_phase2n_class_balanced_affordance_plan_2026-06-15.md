# Phase 2N Class-Balanced Primitive Affordance Pilot

Date registered: 2026-06-15

Status: failed bounded GPU smoke; train and validation data only; no `test_id`
or `test_hard` access.

## Trigger

Phase 2M showed that the source-only primitive affordance model learned some
source signal but did not produce a promotable utility state:

```text
primitive_match_rate: 0.3125
primitive_action_only_prior_match_rate: 0.1640625
mean_target_utility_regret: 0.108387
primitive_action_only_prior_regret: 0.058599
selected_max_primitive_fraction: 0.6875
oracle_max_primitive_fraction: 0.3515625
```

The model improved primitive match but selected too narrowly, mostly
`backward` and `yaw_left`, and had worse regret than the action-only prior.

## Hypothesis

The Phase 2M soft scalar-utility objective may still allow oracle class
imbalance and utility-scale outliers to dominate. A class-balanced hard-oracle
primitive objective should reduce selected primitive collapse if the source
image contains enough information for immediate primitive choice.

This is a bounded objective test, not a new architecture claim.

## Objective Amendment

Phase 2N keeps the Phase 2M source-only model and data contract. It changes
only the ranking objective:

```text
oracle_primitive = argmax primitive_utility(source, primitive)
class_weight = inverse oracle primitive frequency in the training split
loss = class_weight[oracle_primitive] * cross_entropy(predicted_scores, oracle_primitive)
```

Weights are clipped by `primitive_class_weight_max` and normalized to mean one
over observed oracle classes.

Registered smoke settings:

```text
primitive_ranking_loss = hard_ce
primitive_class_balance = oracle_inverse_frequency
primitive_class_weight_max = 5.0
primitive_regression_weight = 0.0
```

The regression term is disabled for this bounded test to isolate whether
balanced hard-oracle supervision fixes the selection-collapse failure.

## Promotion Gate

Use the unchanged Phase 2M executable gate:

- primitive match rate at least `0.50`;
- primitive match rate strictly above the source-independent primitive prior;
- mean target-utility regret strictly below the source-independent primitive
  prior;
- selected primitive distribution not more collapsed than the oracle primitive
  distribution by more than `0.20`;
- finite metrics and finite gradient norm throughout the bounded run;
- no `test_id` or `test_hard` access.

Passing this gate would justify a bounded JEPA integration pilot with an
explicit primitive-affordance state. Failing this gate means RGB CLS-only
primitive affordance supervision is not enough and the next target should be
geometry-derived.

## Implemented Changes

```text
lewm/models/primitive_affordance.py
lewm/benchmarks/phase2m_primitive_affordance.py
scripts/train_jepa_phase2m_primitive_affordance.py
lewm/tests/test_phase2m_primitive_affordance.py
```

The trainer remains the Phase 2M executable because Phase 2N is an objective
amendment over the same data/model contract.

## Pre-Smoke Quality Gate

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .generated/venvs/genesis_render_vulkan/bin/python \
  -m pytest lewm/tests/test_phase2m_primitive_affordance.py \
  lewm/tests/test_phase2i_utility.py -q

Result: 12 passed
```

## Bounded GPU Smoke Command

```bash
env -u HSA_OVERRIDE_GFX_VERSION \
  ROCM_PATH=/opt/rocm-7.1.1 \
  HIP_VISIBLE_DEVICES=0 \
  PATH=/opt/rocm-7.1.1/lib/llvm/bin:/opt/rocm-7.1.1/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  /home/andrewknowles/TinyQuadJEPA/bin/python \
  scripts/train_jepa_phase2m_primitive_affordance.py \
  --train-data .generated/jepa_counterfactual/phase2d_min_sources/train_spatial_future_v1.jsonl \
  --validation-data .generated/jepa_counterfactual/phase2d_min_sources/validation_spatial_future_v1.jsonl \
  --output .generated/jepa_counterfactual/phase2d_min_sources/phase2n_class_balanced_primitive_affordance_smoke.pt \
  --run-class smoke \
  --optimization-steps 256 \
  --evaluation-interval 128 \
  --source-states-per-batch 16 \
  --seed 20260615 \
  --device cuda \
  --lr 1e-4 \
  --max-grad-norm 1.0 \
  --primitive-ranking-loss hard_ce \
  --primitive-regression-weight 0.0 \
  --primitive-class-balance oracle_inverse_frequency \
  --primitive-class-weight-max 5.0 \
  --log-every 64
```

Gate command:

```bash
.generated/venvs/genesis_render_vulkan/bin/python \
  scripts/check_jepa_phase2m_primitive_affordance_gate.py \
  --report .generated/jepa_counterfactual/phase2d_min_sources/phase2n_class_balanced_primitive_affordance_smoke.json \
  --output .generated/jepa_counterfactual/phase2d_min_sources/phase2n_class_balanced_primitive_affordance_smoke_gate.json
```

## Bounded GPU Smoke Results

The ROCm GPU smoke completed with finite metrics:

```text
device: cuda
trainable_parameters: 84,345
optimization_steps: 256
primitive_ranking_loss: hard_ce
primitive_class_balance: oracle_inverse_frequency
primitive_regression_weight: 0.0
```

Validation summary:

```text
primitive_match_rate: 0.33203125
mean_target_utility_regret: 0.252526
selected_max_primitive_fraction: 0.59765625
oracle_max_primitive_fraction: 0.3515625
```

Compared with Phase 2M:

```text
primitive_match_rate: 0.3125 -> 0.33203125
mean_target_utility_regret: 0.108387 -> 0.252526
selected_max_primitive_fraction: 0.6875 -> 0.59765625
```

Compared with the primitive action-only prior:

```text
prior primitive_match_rate: 0.1640625
prior mean_target_utility_regret: 0.058599
prior selected_max_primitive_fraction: 1.0
```

Selected primitive counts:

```text
arc_left: 24
backward: 153
forward_fast: 79
```

Executable gate result:

```text
passed: false
failure_reasons:
- primitive_match_rate_below_threshold
- selected_primitive_distribution_more_collapsed_than_oracle
- regret_not_below_action_only_baseline
```

Artifact hashes:

```text
e038e1fbb6b91fe36874362c6b4c7ed623e05ceca9c1f06c8355ffd3b081f695  .generated/jepa_counterfactual/phase2d_min_sources/phase2n_class_balanced_primitive_affordance_smoke.json
61f551c4ee0cba5378c233e00e48d17ae432630565afd5da583a9842054e2fef  .generated/jepa_counterfactual/phase2d_min_sources/phase2n_class_balanced_primitive_affordance_smoke.pt
7375f1fe80efae44fcb0f74bfb586be6d05499052946fbdaf4fcedff45a45532  .generated/jepa_counterfactual/phase2d_min_sources/phase2n_class_balanced_primitive_affordance_smoke_gate.json
```

## Interpretation

Phase 2N is not promotable.

Class balancing marginally improved primitive match and reduced the worst
selected-primitive concentration, but it made utility regret much worse. The
model learned to select more rare oracle classes, especially `forward_fast`,
without learning the safety/clearance conditions that make those classes
locally appropriate.

This rejects oracle class imbalance as the main remaining explanation. The
failure now points to target semantics: the RGB CLS scalar primitive objective
does not encode the geometry needed to avoid bad-regret primitive choices.

## Decision

Stop RGB CLS-only scalar primitive-affordance objective variants.

The next bounded step should create factorized geometry-derived primitive
affordance targets, separating at least:

- unsafe entry/end/recoverability risk;
- swept clearance;
- task progress or clearance gain;
- heading alignment only as a tie-breaker.

The next model should be evaluated with a safety-first selection rule before
any JEPA latent integration.
