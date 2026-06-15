# Phase 2U Source/Action Factorized Affordance Bridge

Date registered: 2026-06-15

Status: failed bounded ROCm GPU bridge smoke; train and validation data only;
no `test_id` or `test_hard` metric use.

## Trigger

Phase 2S passed when supplied privileged action-conditioned swept geometry.
Phase 2T failed when the current C2 patch-token JEPA was asked to carry
factorized affordance consequences through predicted futures.

Phase 2U tests a narrower bridge: before another JEPA target redesign, can a
source RGB observation plus candidate action sequence directly predict the
factorized consequence labels well enough to pass the primitive gate?

## Model Contract

Phase 2U uses the existing small source/action ranker pattern, but predicts six
factorized consequence logits instead of a scalar utility:

```text
source RGB observation + two-block action sequence
  -> SourceActionFactorizedAffordanceModel
  -> safe_recoverable, task_gain_norm, p05_clearance_norm,
     minimum_clearance_norm, unsafe_sample_fraction, heading_alignment
```

The validation selector scores all 81 candidate sequences per source state
using the same safety-first factor rule and evaluates the selected first
primitive against the primitive oracle.

This is a bridge diagnostic, not a JEPA world model.

## Implemented Files

```text
lewm/models/source_action_utility.py
scripts/train_jepa_phase2u_source_action_factorized_affordance.py
lewm/tests/test_phase2u_source_action_factorized_affordance.py
```

## Data Hygiene

The valid Phase 2U smoke uses only:

```text
.generated/jepa_counterfactual/phase2d_min_sources/train_spatial_future_v1.jsonl
.generated/jepa_counterfactual/phase2d_min_sources/validation_spatial_future_v1.jsonl
```

No `test_id` or `test_hard` metric is reported or used for model selection.

## Quality Gate

Focused tests, CLI parser, and whitespace gate:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .generated/venvs/genesis_render_vulkan/bin/python \
  -m pytest lewm/tests/test_phase2u_source_action_factorized_affordance.py -q

Result: 2 passed

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .generated/venvs/genesis_render_vulkan/bin/python \
  scripts/train_jepa_phase2u_source_action_factorized_affordance.py --help

Result: CLI parsed successfully

git diff --check

Result: clean
```

## Bounded GPU Smoke Command

```bash
env -u HSA_OVERRIDE_GFX_VERSION \
  ROCM_PATH=/opt/rocm-7.1.1 \
  HIP_VISIBLE_DEVICES=0 \
  PATH=/opt/rocm-7.1.1/lib/llvm/bin:/opt/rocm-7.1.1/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  /home/andrewknowles/TinyQuadJEPA/bin/python \
  scripts/train_jepa_phase2u_source_action_factorized_affordance.py \
  --train-data .generated/jepa_counterfactual/phase2d_min_sources/train_spatial_future_v1.jsonl \
  --validation-data .generated/jepa_counterfactual/phase2d_min_sources/validation_spatial_future_v1.jsonl \
  --output .generated/jepa_counterfactual/phase2d_min_sources/phase2u_source_action_factorized_affordance_smoke.pt \
  --run-class smoke \
  --optimization-steps 256 \
  --evaluation-interval 128 \
  --source-states-per-batch 4 \
  --seed 20260615 \
  --device cuda \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --max-grad-norm 1.0 \
  --safety-loss-weight 1.0 \
  --value-loss-weight 1.0 \
  --input-mode source_action \
  --fusion-mode film_interaction \
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
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .generated/venvs/genesis_render_vulkan/bin/python \
  scripts/check_jepa_phase2m_primitive_affordance_gate.py \
  --report .generated/jepa_counterfactual/phase2d_min_sources/phase2u_source_action_factorized_affordance_smoke.json \
  --output .generated/jepa_counterfactual/phase2d_min_sources/phase2u_source_action_factorized_affordance_smoke_gate.json
```

## Bounded GPU Smoke Result

The ROCm GPU smoke completed with finite metrics:

```text
device: cuda
trainable_parameters: 105,954
train candidate rows: 41,472
validation candidate rows: 20,736
optimization_steps: 256
```

Final validation metrics:

```text
factorized_affordance_loss: 0.743217
factorized_safety_bce_loss: 0.652399
factorized_value_mse_loss: 0.090818
```

Final validation primitive-selection summary:

```text
primitive_match_rate: 0.21484375
mean_target_utility_regret: 0.190293744
selected_max_primitive_fraction: 0.45703125
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
  - regret_not_below_action_only_baseline
```

Artifact hashes:

```text
5e3ae9d356c8caee6b7432053fb7472f2387619146d6f65c337a7528add18d14  .generated/jepa_counterfactual/phase2d_min_sources/phase2u_source_action_factorized_affordance_smoke.json
aee9efc4f87815b151b8d9763e7d5d18386a8f33e23014feb042798218008ef1  .generated/jepa_counterfactual/phase2d_min_sources/phase2u_source_action_factorized_affordance_smoke.pt
c07e4ba819e461df41e308676a07bfc34c8de94297dfee64b56d9f10e012e554  .generated/jepa_counterfactual/phase2d_min_sources/phase2u_source_action_factorized_affordance_smoke_gate.json
```

## Interpretation

Phase 2U shows that the current small RGB source/action encoder does not recover
the Phase 2S affordance substrate. It improves primitive match over the
source-independent primitive prior, but still fails the required `0.50` match
threshold and has much worse utility regret than the primitive prior.

This keeps the Phase 2S conclusion intact: action-conditioned swept geometry is
useful when available. The failure is in getting that state from current RGB
and patch-token machinery.

## Decision

Do not launch a full JEPA training run from Phase 2U.

The next bounded fix should not be another shallow head on the current encoder.
It should introduce stronger state supervision or a different target geometry:

- RGB-to-swept-geometry/state distillation with explicit pose/geometry teacher;
- factorized affordance slots supervised directly by Phase 2S features;
- or a non-image-aligned compact state whose dimensions are tied to swept
  clearance, safety, progress, and heading.

Full training remains blocked until a bounded model passes the primitive gate
and, for JEPA integration, the spatial action-identifiability/persistence gate.
