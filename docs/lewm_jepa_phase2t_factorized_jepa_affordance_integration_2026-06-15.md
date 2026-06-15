# Phase 2T Factorized JEPA Affordance Integration

Date registered: 2026-06-15

Status: failed bounded ROCm GPU integration smoke; train and validation data
only; no `test_id` or `test_hard` metric use.

## Trigger

Phase 2S passed the primitive-affordance gate when supplied
action-conditioned swept-geometry features:

```text
primitive_match_rate: 0.53125
mean_target_utility_regret: 0.049968756
selected_max_primitive_fraction: 0.4140625
```

That result made the next question specific: can the existing normalized
spatial JEPA path carry factorized affordance consequences through imagined
future tokens?

## Research Question

If the current C2 spatial JEPA model predicts future spatial tokens from
observation and action, can an auxiliary factorized consequence head on those
predicted futures pass both:

- the Phase 2D spatial action-identifiability/persistence gate; and
- the Phase 2M/2P primitive-affordance gate?

This is a JEPA integration smoke, not a confirmatory full run.

## Model Contract

Phase 2T keeps the existing C2 spatial JEPA controls:

```text
EMA target encoder
normalized spatial token prediction
hard-negative action control
zero-action control
spatial variance floor
appearance SIGReg
```

It adds an external auxiliary head:

```text
predicted spatial futures -> mean pooled latent -> six Phase 2O factor logits
```

The target is sequence-level factorized consequence supervision derived from
the existing Phase 2O candidate labels:

```text
safe_recoverable
task_gain_norm
p05_clearance_norm
minimum_clearance_norm
unsafe_sample_fraction
heading_alignment
```

At validation time, the model scores all 81 candidate sequences per source
state with the same safety-first factor rule and evaluates the selected first
primitive against the primitive oracle.

## Implemented Files

```text
lewm/benchmarks/phase2t_factorized_jepa_affordance.py
scripts/train_jepa_phase2t_factorized_jepa_affordance.py
lewm/tests/test_phase2t_factorized_jepa_affordance.py
```

## Data Hygiene

The valid Phase 2T smoke uses only:

```text
.generated/jepa_counterfactual/phase2d_min_sources/train_spatial_future_v1.jsonl
.generated/jepa_counterfactual/phase2d_min_sources/validation_spatial_future_v1.jsonl
```

No `test_id` or `test_hard` metric is reported or used for model selection.

## Quality Gate

Focused tests, CLI parser, and whitespace gate:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .generated/venvs/genesis_render_vulkan/bin/python \
  -m pytest lewm/tests/test_phase2t_factorized_jepa_affordance.py -q

Result: 2 passed

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .generated/venvs/genesis_render_vulkan/bin/python \
  scripts/train_jepa_phase2t_factorized_jepa_affordance.py --help

Result: CLI parsed successfully

git diff --check

Result: clean
```

A tiny two-source GPU path smoke also completed successfully before the full
bounded smoke. It was used only to validate the training path, not as research
evidence.

## Bounded GPU Smoke Command

```bash
env -u HSA_OVERRIDE_GFX_VERSION \
  ROCM_PATH=/opt/rocm-7.1.1 \
  HIP_VISIBLE_DEVICES=0 \
  PATH=/opt/rocm-7.1.1/lib/llvm/bin:/opt/rocm-7.1.1/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  /home/andrewknowles/TinyQuadJEPA/bin/python \
  scripts/train_jepa_phase2t_factorized_jepa_affordance.py \
  --train-data .generated/jepa_counterfactual/phase2d_min_sources/train_spatial_future_v1.jsonl \
  --validation-data .generated/jepa_counterfactual/phase2d_min_sources/validation_spatial_future_v1.jsonl \
  --output .generated/jepa_counterfactual/phase2d_min_sources/phase2t_factorized_jepa_affordance_smoke.pt \
  --run-class smoke \
  --optimization-steps 32 \
  --evaluation-interval 32 \
  --source-states-per-batch 2 \
  --seed 20260615 \
  --device cuda \
  --lr 3e-4 \
  --head-lr 1e-3 \
  --weight-decay 1e-4 \
  --max-grad-norm 1.0 \
  --factor-loss-lambda 1.0 \
  --safety-loss-weight 1.0 \
  --value-loss-weight 1.0 \
  --head-hidden-dim 96 \
  --detach-action-control-state \
  --safe-threshold 0.5 \
  --unsafe-threshold 0.5 \
  --task-gain-weight 0.75 \
  --p05-clearance-weight 1.25 \
  --minimum-clearance-weight 0.75 \
  --unsafe-penalty-weight 2.0 \
  --heading-weight 0.05 \
  --log-every 8
```

Gate commands:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .generated/venvs/genesis_render_vulkan/bin/python \
  scripts/check_jepa_phase2m_primitive_affordance_gate.py \
  --report .generated/jepa_counterfactual/phase2d_min_sources/phase2t_factorized_jepa_affordance_smoke.json \
  --output .generated/jepa_counterfactual/phase2d_min_sources/phase2t_factorized_jepa_affordance_smoke_primitive_gate.json

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .generated/venvs/genesis_render_vulkan/bin/python \
  scripts/check_jepa_phase2d_smoke_gate.py \
  --report .generated/jepa_counterfactual/phase2d_min_sources/phase2t_factorized_jepa_affordance_smoke.json \
  --output .generated/jepa_counterfactual/phase2d_min_sources/phase2t_factorized_jepa_affordance_smoke_phase2d_gate.json
```

## Bounded GPU Smoke Result

The ROCm GPU smoke completed with finite losses, but failed both gates:

```text
device: cuda
trainable_parameters: 381,334
train candidate rows: 41,472
validation candidate rows: 20,736
optimization_steps: 32
final pre-clip gradient_norm: 1.48134477824e11
```

Final validation primitive-affordance summary:

```text
primitive_match_rate: 0.109375
mean_target_utility_regret: 0.283293713
selected_max_primitive_fraction: 0.375
oracle_max_primitive_fraction: 0.3515625
```

Primitive action-only prior:

```text
primitive_match_rate: 0.1640625
mean_target_utility_regret: 0.058599013
selected_max_primitive_fraction: 1.0
```

Primitive gate:

```text
passed: false
failure_reasons:
  - primitive_match_rate_below_threshold
  - primitive_match_rate_not_above_action_only_baseline
  - regret_not_below_action_only_baseline
```

Final Phase 2D spatial gate:

```text
passed: false
stability_pass: false
hard_negative_action_advantage: -820.373914
zero_action_advantage: 0.002180
one_step_rollout_persistence_ratio: 1842.874351
```

Phase 2D gate failure reasons:

```text
stability_failed
hard_negative_action_advantage_below_threshold
zero_action_advantage_below_threshold
persistence_ratio_not_below_threshold
```

Artifact hashes:

```text
e9205c0983f5cf1dcb58221ff3ddb10c04dd7b6dfac5a21f456b2564dd8f105a  .generated/jepa_counterfactual/phase2d_min_sources/phase2t_factorized_jepa_affordance_smoke.json
0d2872a16e6b7a5c2b5d5c48696b1e48fbaff958d3872e1bcd547b1f6bb1784e  .generated/jepa_counterfactual/phase2d_min_sources/phase2t_factorized_jepa_affordance_smoke.pt
1b31f7228922d23893708dcdddd21c8108466980c012550f3cfdd522694c3439  .generated/jepa_counterfactual/phase2d_min_sources/phase2t_factorized_jepa_affordance_smoke_primitive_gate.json
b3bc49947736cd076b959df58fefd23935829e379104f2b18d9e2c8b90bde905  .generated/jepa_counterfactual/phase2d_min_sources/phase2t_factorized_jepa_affordance_smoke_phase2d_gate.json
```

## Interpretation

Phase 2T separates two claims:

1. The Phase 2S swept-geometry affordance state is a useful target substrate.
2. The current image-aligned spatial JEPA can predict and organize that
   substrate through imagined futures.

The Phase 2S result supports the first claim. The Phase 2T result rejects the
second claim for the tested C2 patch-token integration smoke.

The failure is not subtle. The spatial targets remained unstable/collapsed by
the registered gate, real actions were worse than hard-negative actions, zero
actions were not sufficiently worse than real actions, and persistence was far
better than the rollout. The auxiliary factor head also selected worse first
primitives than the primitive action-only prior.

## Decision

Do not launch a full JEPA training matrix from Phase 2T.

Do not spend more compute tuning this patch-token auxiliary-head integration.
The next fix should change the target state, not only add another head:

- train/predict a compact swept-affordance state directly;
- or distill RGB observations into Phase 2S-style action-conditioned state
  before recurrent JEPA training;
- or replace image-aligned patch targets with factorized affordance slots whose
  axes are explicitly tied to action-conditioned consequence geometry.

Full training remains blocked until a bounded integration smoke passes both the
Phase 2D spatial action-identifiability/persistence gate and the primitive
affordance gate.
