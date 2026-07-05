# Phase 2S Action-Conditioned Swept Geometry Affordance

Date registered: 2026-06-15

Status: passed bounded ROCm GPU primitive-affordance smoke; train and
validation data only; no `test_id` or `test_hard` metric use.

## Trigger

Phase 2R showed that source-local geometry alone was not enough. It improved
primitive match over RGB source-only Phase 2P, but still failed the gate:

```text
primitive_match_rate: 0.38671875
mean_target_utility_regret: 0.096113720
selected_max_primitive_fraction: 0.32421875
```

The failure pointed to the missing variable: action-conditioned consequence
geometry. The Phase 2S test therefore exposes swept geometry for each candidate
primitive and its two-block continuation grid before learning.

## Research Question

If the model is supplied with explicit action-conditioned swept geometry, can
it learn the Phase 2O factor targets well enough to pass the unchanged
primitive-affordance gate?

This is still a diagnostic control. It is not an RGB policy, not a JEPA world
model, and not a deployable navigation result.

## Model Contract

For each source state and first primitive, Phase 2S computes per-primitive
features from:

- the first primitive's executed command block;
- a kinematic rollout of the first primitive;
- kinematic two-block rollouts over all continuations beginning with that first
  primitive;
- scene geometry from the referenced scene manifest;
- optional goal pose from referenced frame metadata.

Feature schema:

```text
phase2s_action_conditioned_swept_geometry_v0
```

The feature vector contains:

```text
primitive one-hot
first-block command and displacement features
first-block swept clearance, unsafe fraction, goal progress, heading alignment
two-block continuation aggregate clearance, unsafe fraction, goal progress
best-progress continuation second-primitive one-hot
```

The learned model is a shared per-primitive MLP:

```text
source x primitive x swept_geometry_features
  -> source x primitive x Phase 2O factor logits
```

The trainer uses the same factorized loss and the same safety-first selector as
Phase 2P and Phase 2R. Phase 2O labels are used only as supervision, not as
input features.

## Implemented Files

```text
lewm/benchmarks/phase2s_swept_geometry_affordance.py
lewm/models/primitive_affordance.py
scripts/train_jepa_phase2s_swept_geometry_affordance.py
lewm/tests/test_phase2s_swept_geometry_affordance.py
```

## Data Hygiene

The valid Phase 2S smoke uses only:

```text
.generated/jepa_counterfactual/phase2d_min_sources/train_spatial_future_v1.jsonl
.generated/jepa_counterfactual/phase2d_min_sources/validation_spatial_future_v1.jsonl
```

Pose metadata is joined through referenced train/validation render summary,
render replay plan, and frame metadata files. Scene geometry is read only from
the scene manifests referenced by those train/validation rows.

No `test_id` or `test_hard` metric is reported or used for model selection.

## Quality Gate

Focused tests, CLI parser, and whitespace gate:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .generated/venvs/genesis_render_vulkan/bin/python \
  -m pytest lewm/tests/test_phase2s_swept_geometry_affordance.py -q

Result: 2 passed

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .generated/venvs/genesis_render_vulkan/bin/python \
  scripts/train_jepa_phase2s_swept_geometry_affordance.py --help

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
  scripts/train_jepa_phase2s_swept_geometry_affordance.py \
  --train-data .generated/jepa_counterfactual/phase2d_min_sources/train_spatial_future_v1.jsonl \
  --validation-data .generated/jepa_counterfactual/phase2d_min_sources/validation_spatial_future_v1.jsonl \
  --output .generated/jepa_counterfactual/phase2d_min_sources/phase2s_swept_geometry_affordance_smoke.pt \
  --run-class smoke \
  --optimization-steps 512 \
  --evaluation-interval 128 \
  --source-states-per-batch 64 \
  --seed 20260615 \
  --device cuda \
  --command-dt-s 0.1 \
  --max-ray-m 4.0 \
  --unsafe-clearance-m 0.02 \
  --hidden-dim 128 \
  --mlp-depth 3 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
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
  --log-every 128
```

Gate command:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .generated/venvs/genesis_render_vulkan/bin/python \
  scripts/check_jepa_phase2m_primitive_affordance_gate.py \
  --report .generated/jepa_counterfactual/phase2d_min_sources/phase2s_swept_geometry_affordance_smoke.json \
  --output .generated/jepa_counterfactual/phase2d_min_sources/phase2s_swept_geometry_affordance_smoke_gate.json
```

## Bounded GPU Smoke Result

The ROCm GPU smoke completed with finite metrics:

```text
device: cuda
trainable_parameters: 40,296
train source states: 512
validation source states: 256
feature_count: 49
train primitive feature rows: 4,608
validation primitive feature rows: 2,304
validation source_pose_found: 256 / 256
validation goal_pose_found: 184 / 256
minimum_continuations_per_primitive: 9
optimization_steps: 512
```

Final validation loss metrics:

```text
factorized_affordance_loss: 0.243200
factorized_safety_bce_loss: 0.225494
factorized_value_mse_loss: 0.017706
```

Final validation primitive-selection summary:

```text
primitive_match_rate: 0.53125
mean_target_utility_regret: 0.049968756
selected_max_primitive_fraction: 0.4140625
oracle_max_primitive_fraction: 0.3515625
uniform_random_expected_primitive_match_rate: 0.111111
```

Source-independent primitive prior:

```text
primitive_match_rate: 0.1640625
mean_target_utility_regret: 0.058599013
selected_max_primitive_fraction: 1.0
selected primitive: yaw_left for all 256 validation source states
```

Primitive gate:

```text
passed: true
failure_reasons: []
min_primitive_match_rate: 0.50
max_selected_primitive_excess: 0.20
```

Artifact hashes:

```text
7c3f7761837972e1f3e231cf4b241bb24e7bfb0d2a68b599848e8bb6bd02bf1e  .generated/jepa_counterfactual/phase2d_min_sources/phase2s_swept_geometry_affordance_smoke.json
e32e8b38bcb7c82075e061f0ef57b1a40b9ca2b6460ec209d5bce06cda68eb1b  .generated/jepa_counterfactual/phase2d_min_sources/phase2s_swept_geometry_affordance_smoke.pt
109e16a29bdd383b7bb008775e5722662eb8c3e4704a907dd717f9bc1bb49a85  .generated/jepa_counterfactual/phase2d_min_sources/phase2s_swept_geometry_affordance_smoke_gate.json
```

## Interpretation

Phase 2S supports the specific hypothesis that the missing state variable is
action-conditioned swept consequence geometry, not merely source-local obstacle
layout.

The model passed all primitive-affordance smoke thresholds:

- primitive match exceeded `0.50`;
- primitive match exceeded the primitive action-only prior;
- utility regret improved over the primitive action-only prior;
- selected primitive distribution was within the registered collapse bound.

The result does not validate RGB perception or JEPA latent prediction. It does
show that the Phase 2O factor targets become learnable when the representation
contains the right action-conditioned geometry.

## Decision

Keep Phase 2S as the first passed learned affordance-state diagnostic.

The next implementation step should be a JEPA integration smoke, not a broader
diagnostic sweep:

- expose or predict a factorized swept-affordance state from observation and
  action;
- keep the Phase 2S primitive gate as a mandatory validation monitor;
- retain the Phase 2D action-identifiability, zero-action, shuffled-action, and
  persistence gates;
- run train/validation only until a checkpoint candidate passes all preregistered
  gates.

Do not claim navigation is solved. The supported claim is narrower: the
action-conditioned swept-geometry state is a viable target substrate for the
next JEPA integration smoke.
