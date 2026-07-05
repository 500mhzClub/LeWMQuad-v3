# Phase 2M Structured First-Primitive Affordance Pilot

Date registered: 2026-06-15

Status: failed bounded GPU smoke; train and validation data only; no `test_id`
or `test_hard` access.

## Trigger

Phases 2I through 2L tested source-conditioned utility rankers over the full
two-block action sequence. Those pilots completed on the ROCm GPU runtime but
did not produce a promotable affordance state:

- Phase 2I and 2J selected `backward/backward` for all 256 validation source
  states.
- Phase 2K removed the action-only identity path, but the source-action model
  still selected `backward/backward` for all validation source states.
- Phase 2L changed hard one-hot utility ranking to a soft utility-distribution
  objective, but still selected a backward-first primitive for all validation
  source states and failed exact top-1 selection.

The immediate failure is therefore not rollout horizon. The system still lacks
a source-local immediate affordance state.

## Research Question

From the same current observation, can a model identify which first primitive
has the best available continuation under the generator-derived safety-first
utility?

This is deliberately narrower than the Phase 2D JEPA world-model question. If
the current image observation cannot support immediate primitive affordance
prediction, a full latent future-prediction run is not scientifically justified.

## Label Contract

Phase 2M collapses the 81 two-block candidates for each source state into a
primitive-level target:

```text
utility(source, first_primitive)
  = max utility(source, first_primitive, second_primitive)
```

The max-over-continuations rule asks whether the first primitive is locally
promising, independent of the exact second-block continuation. It preserves the
Phase 2G safety-first utility target and changes only the target geometry.

The implemented target schema is:

```text
phase2m_source_local_first_primitive_max_utility_v0
```

## Model Contract

The Phase 2M diagnostic model is source-only:

```text
RGB source observation -> ViT-Tiny diagnostic encoder -> primitive utility vector
```

There is no candidate action input. This prevents the previous sequence-ranker
shortcut where global action priors dominated source conditioning.

Training loss:

```text
soft target distribution = softmax((utility - source_mean_utility) / temperature)
loss = soft_cross_entropy(predicted_scores, soft target distribution)
       + centered_utility_regression
```

The centered regression term focuses the model on source-local primitive
ordering instead of global scene difficulty.

## Promotion Gate

Phase 2M is promotable only if the executable gate passes on validation:

- primitive match rate at least `0.50`;
- primitive match rate strictly above the source-independent primitive prior;
- mean target-utility regret strictly below the source-independent primitive
  prior;
- selected primitive distribution is not more collapsed than the oracle
  primitive distribution by more than `0.20`;
- finite metrics and finite gradient norm throughout the bounded run;
- no `test_id` or `test_hard` access.

Passing this gate would justify a bounded JEPA integration pilot with a
structured affordance token/state. It would not yet justify a full confirmatory
matrix.

## Implemented Files

```text
lewm/benchmarks/phase2m_primitive_affordance.py
lewm/models/primitive_affordance.py
scripts/train_jepa_phase2m_primitive_affordance.py
scripts/check_jepa_phase2m_primitive_affordance_gate.py
lewm/tests/test_phase2m_primitive_affordance.py
```

## Pre-Smoke Quality Gate

Focused tests:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .generated/venvs/genesis_render_vulkan/bin/python \
  -m pytest lewm/tests/test_phase2m_primitive_affordance.py \
  lewm/tests/test_phase2i_utility.py -q

Result: 11 passed
```

Real-data CPU contract smoke:

```text
train subset: 162 candidate rows
validation subset: 162 candidate rows
optimization steps: 1
device: cpu
result: finite train/validation metrics; executable report and gate schema verified
scientific gate: failed, as expected for a two-source contract check
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
  --output .generated/jepa_counterfactual/phase2d_min_sources/phase2m_primitive_affordance_smoke.pt \
  --run-class smoke \
  --optimization-steps 256 \
  --evaluation-interval 128 \
  --source-states-per-batch 16 \
  --seed 20260615 \
  --device cuda \
  --lr 1e-4 \
  --max-grad-norm 1.0 \
  --primitive-ranking-loss soft_ce \
  --primitive-regression-weight 1.0 \
  --primitive-softmax-temperature 0.25 \
  --log-every 64
```

Gate command:

```bash
.generated/venvs/genesis_render_vulkan/bin/python \
  scripts/check_jepa_phase2m_primitive_affordance_gate.py \
  --report .generated/jepa_counterfactual/phase2d_min_sources/phase2m_primitive_affordance_smoke.json \
  --output .generated/jepa_counterfactual/phase2d_min_sources/phase2m_primitive_affordance_smoke_gate.json
```

## Bounded GPU Smoke Results

The ROCm GPU smoke completed with finite metrics:

```text
device: cuda
trainable_parameters: 84,345
train source states: 512
validation source states: 256
valid primitive targets per source: 9
optimization_steps: 256
```

Validation summary:

```text
primitive_match_rate: 0.3125
mean_target_utility_regret: 0.108387
selected_max_primitive_fraction: 0.6875
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
backward: 176
hold: 5
yaw_left: 70
yaw_right: 5
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
- regret_not_below_action_only_baseline
```

Artifact hashes:

```text
87a60d9182343c6d93779d44cc63871d1f1fe7565d31db6fb6e1611aa965e2bd  .generated/jepa_counterfactual/phase2d_min_sources/phase2m_primitive_affordance_smoke.json
805965d297216c6c5275eb3799b0ed76f8cb8563c0292ad8b75b3ec253a47cc4  .generated/jepa_counterfactual/phase2d_min_sources/phase2m_primitive_affordance_smoke.pt
122dc5781f22a9a76dc41a32524aa3891d63c8c46c763a136f414c968e515e61  .generated/jepa_counterfactual/phase2d_min_sources/phase2m_primitive_affordance_smoke_gate.json
```

## Interpretation

Phase 2M is informative but not promotable.

The source-only primitive affordance model improved primitive match over the
source-independent primitive prior (`0.3125` vs `0.1641`), so the source image
does contain some immediate-action signal. However, utility regret became
worse than the prior (`0.1084` vs `0.0586`) and the model still concentrated
selection into a small subset of primitives, especially `backward`.

The current scalar utility-vector objective is therefore not sufficient as the
structured affordance state. It can learn a coarse biased classifier but not a
calibrated source-local utility ordering.

## Decision Rule

Phase 2M failed. Do not integrate this source-only scalar primitive-affordance
head into JEPA.

The next bounded implementation should explicitly attack the observed failure:
selected primitive collapse and poor utility regret despite improved primitive
match. The immediate low-cost test is a class-balanced hard-oracle primitive
objective using the same source-only model. If that cannot reduce collapse and
regret, the next step should stop using RGB CLS-only supervision and introduce
geometry-derived local affordance targets, for example occupancy/ray or
swept-clearance features, before re-entering JEPA latent prediction.

If Phase 2M passes, the next bounded JEPA experiment should add a dedicated
primitive-affordance state/token and require both:

- the Phase 2M primitive affordance gate to remain passed; and
- the Phase 2D one-step persistence/action-identifiability gate to pass.
