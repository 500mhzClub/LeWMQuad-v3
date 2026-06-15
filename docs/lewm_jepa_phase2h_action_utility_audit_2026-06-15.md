# Phase 2H Action-Utility Label And Action-Only Baseline Audit

Date registered: 2026-06-15

Status: completed audit; train and validation data only; no `test_id` or
`test_hard` access.

## Trigger

Phase 2G added direct source-local action-utility supervision to predicted
future spatial tokens, but the bounded smoke failed:

```text
top1_match_rate: 0.20
first_primitive_match_rate: 0.20
hard_negative_action_advantage: -30.852806
one_step_rollout_persistence_ratio: 117.979802
collapse_warning: true
effective_rank_warning: true
```

Before building another model, the utility target itself needs an audit. The
question is whether the labels are genuinely source-conditioned or whether a
source-independent action-sequence prior already explains validation choices.

## Audit Contract

Use the Phase 2G target version:

```text
phase2g_oracle_order_utility_v0
```

The audit measures:

- target coverage by row and source state;
- within-source utility range;
- oracle top-tie fraction;
- validation performance of train-derived full-sequence action-only priors;
- validation performance of train-derived first-primitive action-only priors;
- uniform random expected top-1 and first-primitive match rates.

The audit is not a model-training run. It is a data and confounding gate.

## Command

```bash
.generated/venvs/genesis_render_vulkan/bin/python \
  scripts/audit_jepa_phase2h_action_utility.py \
  --train-data .generated/jepa_counterfactual/phase2d_min_sources/train_spatial_future_v1.jsonl \
  --validation-data .generated/jepa_counterfactual/phase2d_min_sources/validation_spatial_future_v1.jsonl \
  --output .generated/jepa_counterfactual/phase2d_min_sources/phase2h_action_utility_audit.json
```

## Implementation Gates

Focused tests:

```text
35 passed
```

Diff hygiene and JSON validation:

```text
git diff --check passed
jq empty docs/lewm_jepa_claims_registry_2026-06-14.json passed
```

## Results

Utility-target coverage:

```text
train_source_states: 512
train_valid_utility_rows: 41472
train_valid_row_fraction: 1.0
train_mean_candidate_rows_per_source: 81.0
train_mean_utility_range_per_source: 1.233588
train_mean_top_tie_fraction: 0.012635

validation_source_states: 256
validation_valid_utility_rows: 20736
validation_valid_row_fraction: 1.0
validation_mean_candidate_rows_per_source: 81.0
validation_mean_utility_range_per_source: 1.326726
validation_mean_top_tie_fraction: 0.013455
```

Train-derived action-only validation baselines:

```text
baseline: full_sequence_mean
top1_match_rate: 0.0078125
first_primitive_match_rate: 0.1640625
mean_target_utility_regret: 0.195299
uniform_random_expected_top1_rate: 0.012346
uniform_random_expected_first_primitive_match_rate: 0.111111

baseline: first_primitive_mean
top1_match_rate: 0.01171875
first_primitive_match_rate: 0.3515625
mean_target_utility_regret: 0.243493
uniform_random_expected_top1_rate: 0.012346
uniform_random_expected_first_primitive_match_rate: 0.111111
```

Validation oracle first-primitive distribution:

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

Artifact hash:

```text
d061a338cc71c52440057bef98bc1e0d7ccaa439f32892d91d22b61fd66d6b18  .generated/jepa_counterfactual/phase2d_min_sources/phase2h_action_utility_audit.json
```

## Interpretation

The utility target has complete coverage on the registered-minimum train and
validation splits and a non-trivial within-source range. Top ties are rare, so
the source-local ranking target is well-defined.

The full-sequence action-only baseline is below uniform random top-1. The
first-primitive baseline improves first-primitive match rate, mainly because
some primitives are more often safe in this corpus, but it does not improve
top-1 selection and has higher utility regret than the full-sequence prior.

Therefore, the utility target is not explained by a source-independent action
sequence prior. The failed Phase 2G utility head should be attributed to the
collapsing predicted spatial-token representation, not to an obviously
confounded utility label.

## Decision

Proceed to a bounded Phase 2I source-conditioned affordance/utility pilot.

The Phase 2I model should predict source-local action utility from the current
observation and candidate action sequence directly. It should be evaluated
against the Phase 2H action-only baselines before it is used to justify another
JEPA world-model full run.

Promotion requires:

- all metrics and gradients finite;
- validation top-1 utility match above both action-only baselines and at least
  `0.25`;
- validation first-primitive match above both action-only baselines and at
  least `0.50`;
- mean utility regret below both action-only baselines;
- no `test_id` or `test_hard` access.
