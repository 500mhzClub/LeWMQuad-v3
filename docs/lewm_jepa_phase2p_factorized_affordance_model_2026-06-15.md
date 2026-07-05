# Phase 2P Factorized Primitive Affordance Model

Date registered: 2026-06-15

Status: failed bounded GPU smoke; train and validation data only; no `test_id`
or `test_hard` access.

## Trigger

Phase 2M and Phase 2N both failed the primitive-affordance gate. They showed
that source images contain some immediate-action signal, but scalar primitive
utility supervision produced either selected-primitive collapse, worse utility
regret than the primitive action-only prior, or both.

Phase 2O then audited the factorized target contract and found complete
train/validation coverage for the core geometry-derived primitive factors:

```text
safe_recoverable
task_gain_norm
p05_clearance_norm
minimum_clearance_norm
unsafe_sample_fraction
```

`heading_alignment` is partially available and remains an optional tie-breaker.

## Research Question

Can a source-image model predict enough factorized immediate affordance
structure to select first primitives with lower validation regret and less
selection collapse than the scalar utility heads?

This is a prerequisite diagnostic, not a JEPA world-model result. It asks
whether the current source observation can support immediate safety/progress
factor prediction before those factors are integrated into a latent dynamics
objective.

## Model Contract

Phase 2P uses a source-only factorized primitive affordance model:

```text
RGB source observation -> diagnostic ViT encoder -> primitive x factor logits
```

The output shape is:

```text
batch x primitive_count x factor_count
```

The model predicts the six Phase 2O factors for every first primitive. Runtime
selection uses transformed factor values:

```text
safe_recoverable: sigmoid
task_gain_norm: tanh
p05_clearance_norm: sigmoid
minimum_clearance_norm: sigmoid
unsafe_sample_fraction: sigmoid
heading_alignment: sigmoid
```

Training loss:

```text
loss = safety_loss_weight * BCEWithLogits(safe_recoverable)
       + value_loss_weight * masked_MSE(transformed_continuous_factors)
```

The loss is masked per factor so partially missing `heading_alignment` labels
do not invalidate the batch.

## Safety-First Selection Rule

For each source state, Phase 2P scores valid first primitives as follows:

1. keep candidates whose predicted `safe_recoverable >= 0.5` and predicted
   `unsafe_sample_fraction <= 0.5`;
2. among kept candidates, maximize:

```text
0.75 * task_gain_norm
+ 1.25 * p05_clearance_norm
+ 0.75 * minimum_clearance_norm
- 2.00 * unsafe_sample_fraction
+ 0.05 * heading_alignment
```

3. if no candidate passes the predicted safety gate, fall back to:

```text
safe_recoverable - unsafe_sample_fraction + 0.05 * score
```

This rule is intentionally explicit. It tests whether learned factor values are
usable for source-local action selection without hiding the decision in another
learned head.

## Promotion Gate

Use the unchanged Phase 2M executable primitive gate on validation:

- primitive match rate at least `0.50`;
- primitive match rate strictly above the source-independent primitive prior;
- mean target-utility regret strictly below the source-independent primitive
  prior;
- selected primitive distribution not more collapsed than the oracle primitive
  distribution by more than `0.20`;
- finite metrics and finite gradient norm throughout the bounded run;
- no `test_id` or `test_hard` access.

Passing this gate would justify a bounded JEPA integration pilot with an
explicit factorized affordance state. It would not yet justify a full
confirmatory JEPA matrix.

## Implemented Files

```text
lewm/benchmarks/phase2o_factorized_affordance.py
lewm/models/primitive_affordance.py
scripts/train_jepa_phase2p_factorized_affordance.py
lewm/tests/test_phase2p_factorized_affordance.py
```

## Pre-Smoke Quality Gate

Focused tests and trainer CLI:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .generated/venvs/genesis_render_vulkan/bin/python \
  -m pytest lewm/tests/test_phase2p_factorized_affordance.py \
  lewm/tests/test_phase2o_factorized_affordance.py -q

Result: 5 passed

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .generated/venvs/genesis_render_vulkan/bin/python \
  scripts/train_jepa_phase2p_factorized_affordance.py --help

Result: CLI parsed successfully
```

## Bounded GPU Smoke Command

```bash
env -u HSA_OVERRIDE_GFX_VERSION \
  ROCM_PATH=/opt/rocm-7.1.1 \
  HIP_VISIBLE_DEVICES=0 \
  PATH=/opt/rocm-7.1.1/lib/llvm/bin:/opt/rocm-7.1.1/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  /home/andrewknowles/TinyQuadJEPA/bin/python \
  scripts/train_jepa_phase2p_factorized_affordance.py \
  --train-data .generated/jepa_counterfactual/phase2d_min_sources/train_spatial_future_v1.jsonl \
  --validation-data .generated/jepa_counterfactual/phase2d_min_sources/validation_spatial_future_v1.jsonl \
  --output .generated/jepa_counterfactual/phase2d_min_sources/phase2p_factorized_affordance_smoke.pt \
  --run-class smoke \
  --optimization-steps 256 \
  --evaluation-interval 128 \
  --source-states-per-batch 16 \
  --seed 20260615 \
  --device cuda \
  --lr 1e-4 \
  --max-grad-norm 1.0 \
  --safety-loss-weight 1.0 \
  --value-loss-weight 1.0 \
  --safe-threshold 0.5 \
  --unsafe-threshold 0.5 \
  --task-gain-weight 0.75 \
  --p05-clearance-weight 1.25 \
  --minimum-clearance-weight 0.75 \
  --unsafe-penalty-weight 2.0 \
  --heading-weight 0.05 \
  --log-every 64
```

Gate command:

```bash
.generated/venvs/genesis_render_vulkan/bin/python \
  scripts/check_jepa_phase2m_primitive_affordance_gate.py \
  --report .generated/jepa_counterfactual/phase2d_min_sources/phase2p_factorized_affordance_smoke.json \
  --output .generated/jepa_counterfactual/phase2d_min_sources/phase2p_factorized_affordance_smoke_gate.json
```

## Bounded GPU Smoke Results

The ROCm GPU smoke completed with finite metrics:

```text
device: cuda
trainable_parameters: 88,710
train source states: 512
validation source states: 256
valid primitive targets per source: 9
optimization_steps: 256
```

Final validation metrics:

```text
factorized_affordance_loss: 0.750319
factorized_safety_bce_loss: 0.673972
factorized_value_mse_loss: 0.076347
```

Final validation primitive-selection summary:

```text
primitive_match_rate: 0.13671875
mean_target_utility_regret: 0.124758
selected_max_primitive_fraction: 0.609375
oracle_max_primitive_fraction: 0.3515625
uniform_random_expected_primitive_match_rate: 0.111111
```

Source-independent primitive prior:

```text
primitive_match_rate: 0.1640625
mean_target_utility_regret: 0.058599
selected_max_primitive_fraction: 1.0
selected_primitive: yaw_left for all 256 validation source states
```

Selected primitive counts:

```text
arc_left: 37
backward: 46
hold: 17
yaw_right: 156
```

Oracle primitive counts:

```text
arc_left: 34
arc_right: 14
backward: 90
forward_fast: 47
forward_medium: 3
forward_slow: 1
hold: 3
yaw_left: 42
yaw_right: 22
```

Executable gate result:

```text
passed: false
failure_reasons:
- primitive_match_rate_below_threshold
- selected_primitive_distribution_more_collapsed_than_oracle
- primitive_match_rate_not_above_action_only_baseline
- regret_not_below_action_only_baseline
```

Artifact hashes:

```text
b62634f198cf968f940606c70c939bc799140a61da8fd6ce7f7e869f8259c0c2  .generated/jepa_counterfactual/phase2d_min_sources/phase2p_factorized_affordance_smoke.json
57cbbc16b59080e8b791c83a46845c1f81b6cf544464084e7bdc4dc520b2eae3  .generated/jepa_counterfactual/phase2d_min_sources/phase2p_factorized_affordance_smoke.pt
76ac702823b84f8a70848b7362bfb3c0abfb6793beccee9b0d2893896dce36b1  .generated/jepa_counterfactual/phase2d_min_sources/phase2p_factorized_affordance_smoke_gate.json
```

## Interpretation

Phase 2P is not promotable.

The factorized targets and safety-first selection rule did not recover a useful
source-local primitive selector with the current small RGB source-only encoder.
The model remained finite and did not collapse to one primitive, but it scored
worse than the primitive action-only prior on both exact primitive match and
utility regret. It also selected `yaw_right` for `156/256` validation source
states, exceeding the oracle maximum primitive fraction by `0.2578125`, above
the registered `0.20` tolerance.

This rejects the narrow hypothesis that factorized geometry-derived labels plus
a hand-written safety-first selector are sufficient when trained through the
same small RGB CLS diagnostic encoder. It does not reject factorized
affordance supervision itself.

## Decision Rule

Phase 2P failed. Do not integrate this factorized source-only head into JEPA.

The next step should stop using the small RGB source-only diagnostic encoder as
the affordance substrate. The next candidate should expose geometry to the
learned state more directly, for example factorized local affordance slots,
ray/clearance tokens, or motion-equivariant tokens, before returning to JEPA
latent rollout training.
