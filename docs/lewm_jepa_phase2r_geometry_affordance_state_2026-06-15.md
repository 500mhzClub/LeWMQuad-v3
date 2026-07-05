# Phase 2R Geometry-Exposed Affordance State

Date registered: 2026-06-15

Status: failed bounded ROCm GPU smoke; train and validation data only; no
`test_id` or `test_hard` metric use.

## Trigger

Phase 2Q showed that the Phase 2O factor targets plus the Phase 2P
safety-first selector form a coherent train/validation ceiling. With true
factor values, the selector passed the primitive gate:

```text
validation primitive_match_rate: 0.8671875
validation mean_target_utility_regret: 0.001419918
validation selected_max_primitive_fraction: 0.33984375
validation oracle_max_primitive_fraction: 0.3515625
```

Phase 2P, however, failed when those factors had to be predicted from the
source RGB observation. Phase 2R therefore tests a sharper hypothesis: whether
privileged source-local geometry features are sufficient to learn the Phase 2O
factors, before attempting another RGB or JEPA integration.

## Research Question

Can a learned model supplied with explicit source pose, local obstacle
geometry, and optional goal-relative geometry predict enough factorized
primitive affordance structure to pass the unchanged primitive gate?

This is a diagnostic control. It is not a deployable RGB policy, not a JEPA
world model, and not a planning result.

## Model Contract

Phase 2R constructs a deterministic source-local geometry feature vector:

```text
16 normalized radial obstacle distances
source obstacle clearance
source x/y within scene bounds
source yaw sin/cos
goal_present
goal distance, forward offset, left offset
goal bearing sin/cos
```

Feature schema:

```text
phase2r_source_goal_geometry_features_v0
```

The learned model is a small MLP:

```text
geometry_features -> primitive_count x Phase 2O factor_count logits
```

It uses the same factorized loss and the same safety-first selection rule as
Phase 2P. The only intentional change is the source representation.

## Implemented Files

```text
lewm/benchmarks/phase2r_geometry_affordance.py
lewm/models/primitive_affordance.py
scripts/train_jepa_phase2r_geometry_affordance.py
lewm/tests/test_phase2r_geometry_affordance.py
```

## Data Hygiene

The valid Phase 2R smoke uses only:

```text
.generated/jepa_counterfactual/phase2d_min_sources/train_spatial_future_v1.jsonl
.generated/jepa_counterfactual/phase2d_min_sources/validation_spatial_future_v1.jsonl
```

Pose metadata is joined through the referenced train/validation render summary,
render replay plan, and frame metadata files. Scene geometry is read from the
scene manifests referenced by those same train/validation rows.

No `test_id` or `test_hard` metric is reported or used for model selection.

## Quality Gate

Focused tests and trainer CLI:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .generated/venvs/genesis_render_vulkan/bin/python \
  -m pytest lewm/tests/test_phase2r_geometry_affordance.py \
  lewm/tests/test_phase2q_factorized_ceiling.py -q

Result: 4 passed

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .generated/venvs/genesis_render_vulkan/bin/python \
  scripts/train_jepa_phase2r_geometry_affordance.py --help

Result: CLI parsed successfully
```

## Bounded GPU Smoke Command

```bash
env -u HSA_OVERRIDE_GFX_VERSION \
  ROCM_PATH=/opt/rocm-7.1.1 \
  HIP_VISIBLE_DEVICES=0 \
  PATH=/opt/rocm-7.1.1/lib/llvm/bin:/opt/rocm-7.1.1/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  /home/andrewknowles/TinyQuadJEPA/bin/python \
  scripts/train_jepa_phase2r_geometry_affordance.py \
  --train-data .generated/jepa_counterfactual/phase2d_min_sources/train_spatial_future_v1.jsonl \
  --validation-data .generated/jepa_counterfactual/phase2d_min_sources/validation_spatial_future_v1.jsonl \
  --output .generated/jepa_counterfactual/phase2d_min_sources/phase2r_geometry_affordance_smoke.pt \
  --run-class smoke \
  --optimization-steps 512 \
  --evaluation-interval 128 \
  --source-states-per-batch 64 \
  --seed 20260615 \
  --device cuda \
  --ray-count 16 \
  --max-ray-m 4.0 \
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
  --report .generated/jepa_counterfactual/phase2d_min_sources/phase2r_geometry_affordance_smoke.json \
  --output .generated/jepa_counterfactual/phase2d_min_sources/phase2r_geometry_affordance_smoke_gate.json
```

## Bounded GPU Smoke Result

The ROCm GPU smoke completed with finite metrics:

```text
device: cuda
trainable_parameters: 43,628
train source states: 512
validation source states: 256
feature_count: 27
validation source_pose_found: 256 / 256
validation goal_pose_found: 184 / 256
optimization_steps: 512
```

Final validation loss metrics:

```text
factorized_affordance_loss: 1.403183
factorized_safety_bce_loss: 1.358852
factorized_value_mse_loss: 0.044331
```

Final validation primitive-selection summary:

```text
primitive_match_rate: 0.38671875
mean_target_utility_regret: 0.096113720
selected_max_primitive_fraction: 0.32421875
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
passed: false
failure_reasons:
  - primitive_match_rate_below_threshold
  - regret_not_below_action_only_baseline
min_primitive_match_rate: 0.50
max_selected_primitive_excess: 0.20
```

Artifact hashes:

```text
066fa5a998437030944e41380b2f5c5ca8316a769dba3ec2d12546241b2f21dd  .generated/jepa_counterfactual/phase2d_min_sources/phase2r_geometry_affordance_smoke.json
971be720771809c0fc7154f6cfd1f2b7bddb2a91aa4b330d7c12e5ee5c80e266  .generated/jepa_counterfactual/phase2d_min_sources/phase2r_geometry_affordance_smoke.pt
118f4720ffaa1c80b1a4436bc2f8bfeaded961c2cd67d26c223409313336b1ab  .generated/jepa_counterfactual/phase2d_min_sources/phase2r_geometry_affordance_smoke_gate.json
```

## Interpretation

Phase 2R separates two hypotheses:

1. The learned model only needed explicit source-local geometry.
2. The learned model needs action-conditioned consequence geometry, not only
   source-local obstacle and goal features.

The bounded smoke weakens the first hypothesis. Source-local geometry improved
primitive match over Phase 2P and over the primitive action-only prior, and it
did not fail the selected-primitive distribution collapse check. However, it
still failed the promotion gate because match remained below `0.50` and utility
regret was worse than the primitive prior.

The failure mode is informative. The Phase 2Q true-factor ceiling remains
strong, while Phase 2R's source-only geometry features do not recover it. The
missing signal is likely the action-conditioned swept consequence state:
clearance along the primitive's path, collision/recovery after executing that
primitive, and task progress after the primitive, not just rays around the
current pose.

## Decision

Do not launch a full JEPA training run from Phase 2R.

Do not integrate the Phase 2R geometry-feature MLP into the navigation stack.

The next bounded fix should expose action-conditioned geometry before learning:

- per-primitive swept clearance and collision/recovery features;
- per-primitive goal-progress deltas in the source frame;
- factor-table or slot-state distillation from Phase 2O true factors;
- the same primitive gate before any JEPA full training run.

The acceptance gate remains unchanged:

- primitive match rate at least `0.50`;
- primitive match rate above the primitive action-only prior;
- mean target-utility regret below the primitive action-only prior;
- selected primitive distribution not more collapsed than oracle by more than
  `0.20`;
- train/validation only until a selected checkpoint candidate exists.
