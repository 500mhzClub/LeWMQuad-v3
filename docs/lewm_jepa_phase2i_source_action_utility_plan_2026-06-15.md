# Phase 2I Source-Conditioned Action-Utility Pilot

Date registered: 2026-06-15

Status: implementation pilot; train and validation data only; no `test_id` or
`test_hard` access.

## Trigger

Phase 2H found that the Phase 2G utility labels are usable and not explained by
source-independent action-only priors:

```text
validation_full_sequence_action_only_top1: 0.0078125
validation_first_primitive_action_only_top1: 0.01171875
uniform_random_expected_top1: 0.012346
validation_valid_utility_rows: 20736
```

The failed Phase 2G utility head therefore points back to the collapsing
predicted spatial-token representation. The next bounded test is whether a
source-conditioned utility state can learn the action ranking at all.

## Hypothesis

The current observation plus candidate action sequence contains enough
information to predict source-local action utility better than action-only
baselines. If this fails, a JEPA full run is not justified because the
supervised affordance target itself is not learnable under the current data and
model scale. If it passes, the next JEPA redesign should add a dedicated
source-conditioned affordance/utility state rather than attaching utility to
mean-pooled image-token futures.

## Model Contract

The Phase 2I ranker consumes:

- the current RGB source observation;
- the candidate action sequence;
- masked source-local utility targets from `phase2g_oracle_order_utility_v0`.

It trains with the same source-grouped listwise cross-entropy used by Phase 2G,
plus the same small masked utility regression term. It does not predict future
image tokens and is not a JEPA world model. It is a prerequisite diagnostic for
the next JEPA architecture.

## Promotion Gate

A bounded smoke may be promoted to a source-conditioned affordance/utility
model family only if:

- all metrics and gradients remain finite;
- validation top-1 utility match is at least `0.25`;
- validation first-primitive utility match is at least `0.50`;
- validation top-1 utility match exceeds both Phase 2H action-only baselines;
- validation first-primitive match exceeds both Phase 2H action-only baselines;
- validation utility regret is lower than both Phase 2H action-only baselines;
- no `test_id` or `test_hard` access is used.

Passing this gate is not sufficient to launch a JEPA full training run. It only
permits integrating a dedicated affordance/utility state into the next bounded
JEPA pilot.

## Bounded Smoke Command

```bash
env -u HSA_OVERRIDE_GFX_VERSION \
  ROCM_PATH=/opt/rocm-7.1.1 \
  HIP_VISIBLE_DEVICES=0 \
  PATH=/opt/rocm-7.1.1/lib/llvm/bin:/opt/rocm-7.1.1/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  /home/andrewknowles/TinyQuadJEPA/bin/python \
  scripts/train_jepa_phase2i_source_action_utility.py \
  --train-data .generated/jepa_counterfactual/phase2d_min_sources/train_spatial_future_v1.jsonl \
  --validation-data .generated/jepa_counterfactual/phase2d_min_sources/validation_spatial_future_v1.jsonl \
  --baseline-audit .generated/jepa_counterfactual/phase2d_min_sources/phase2h_action_utility_audit.json \
  --output .generated/jepa_counterfactual/phase2d_min_sources/phase2i_source_action_utility_smoke.pt \
  --run-class smoke \
  --optimization-steps 256 \
  --evaluation-interval 128 \
  --source-states-per-batch 2 \
  --seed 20260614 \
  --device cuda \
  --lr 1e-4 \
  --max-grad-norm 1.0
```

The utility-gate checker must then be run against the emitted JSON report.

## Implementation Gates Before Smoke

Focused tests:

```text
39 passed
```

Diff hygiene and JSON validation:

```text
git diff --check passed
jq empty docs/lewm_jepa_claims_registry_2026-06-14.json passed
```

## Bounded Smoke Results

Status: failed. Do not integrate this Phase 2I ranker into a JEPA pilot.

The ROCm GPU smoke completed and wrote artifacts with finite metrics and
gradients.

Final validation utility summary:

```text
top1_match_rate: 0.214844
first_primitive_match_rate: 0.3515625
mean_target_utility_regret: 0.323505
mean_selected_target_utility: -0.292631
mean_oracle_target_utility: 0.030874
source_state_count: 256
```

Action-only baseline reference:

```text
full_sequence_mean_top1: 0.0078125
full_sequence_mean_first_primitive: 0.1640625
full_sequence_mean_regret: 0.195299

first_primitive_mean_top1: 0.01171875
first_primitive_mean_first_primitive: 0.3515625
first_primitive_mean_regret: 0.243493
```

Executable utility-gate result:

```text
passed: false
failure_reasons:
- top1_match_rate_below_threshold
- first_primitive_match_rate_below_threshold
- first_primitive_match_rate_not_above_action_only_baselines
- regret_not_below_action_only_baselines
```

Selection-distribution failure mode:

```text
selected_first_primitive:
backward: 256

oracle_first_primitive:
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

Artifact hashes:

```text
68669e48fb791ec901a5b88c03029201aadb594678df25f3ed62e15e902e5fdb  .generated/jepa_counterfactual/phase2d_min_sources/phase2i_source_action_utility_smoke.json
ae03869dbb7c2220bb94e503fe5462d4ac8f86e87ea838681e3a06b47bda0a31  .generated/jepa_counterfactual/phase2d_min_sources/phase2i_source_action_utility_smoke.pt
f9df34b956df445522fb1e11b13c0fc2ca7f43463644838739c6db0ba5034b30  .generated/jepa_counterfactual/phase2d_min_sources/phase2i_source_action_utility_smoke_gate.json
```

### Matched Trainable Action-Only Control

I then ran the same trainer with `--input-mode action_only`, using the same
seed, optimizer, schedule, and gate. This controls for whether the
source-conditioned ranker used source images at all.

The action-only control produced the same validation selection summary:

```text
top1_match_rate: 0.214844
first_primitive_match_rate: 0.3515625
mean_target_utility_regret: 0.323505
mean_selected_target_utility: -0.292631
mean_oracle_target_utility: 0.030874
```

Executable utility-gate result:

```text
passed: false
failure_reasons:
- top1_match_rate_below_threshold
- first_primitive_match_rate_below_threshold
- first_primitive_match_rate_not_above_action_only_baselines
- regret_not_below_action_only_baselines
```

Control artifact hashes:

```text
8a6ea32249ba6425de6ee552e27aa217902b2dbe9db31cc0af2dafef08b52832  .generated/jepa_counterfactual/phase2d_min_sources/phase2i_action_only_utility_control_smoke.json
9dffd8833c521b2701802b72a524122402b90c3b9e6bd1fb95e5ab00ccad1d44  .generated/jepa_counterfactual/phase2d_min_sources/phase2i_action_only_utility_control_smoke.pt
7675641adeecf3159f49127d24a26dd4c8d06aa954100d16ad3d649b3bbcd72d  .generated/jepa_counterfactual/phase2d_min_sources/phase2i_action_only_utility_control_smoke_gate.json
```

## Interpretation

The source-conditioned ranker learned a strong global `backward/backward`
preference. That shortcut gives exact top-1 matches on the subset of validation
states where `backward/backward` is truly the oracle sequence, but it fails to
adapt to source geometry and produces worse mean utility regret than the
action-only baselines.

The matched trainable action-only control reached the same validation result.
Therefore, the source-conditioned concatenation model did not measurably use
the source observation.

This is not evidence that source-conditioned affordance learning is impossible.
It rejects this implementation: concatenating source CLS tokens and action
embeddings is too weak or too easy to ignore under the current source-local
ranking objective.

## Decision

Stop this Phase 2I pilot at bounded smoke. It is not promotable.

The next bounded fix must make source-action interaction explicit and penalize
global action-prior collapse. Candidate changes:

- use a FiLM or bilinear source-conditioned action scorer;
- center candidate scores per source group before listwise loss;
- add a selected-primitive diversity or anti-prior diagnostic, not as a
  deployment objective but as a smoke-gate warning;
- compare against an action-only trainable ranker under the same architecture.
