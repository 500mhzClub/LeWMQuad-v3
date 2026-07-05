# Phase 2G Source-Local Action Utility Pilot

Date registered: 2026-06-15

Status: implementation pilot; train and validation data only; no `test_id` or
`test_hard` access.

## Trigger

Phase 2F showed that a mean-pooled sequence-level consequence head can be
trained in isolation, but it did not prevent latent collapse and did not make
the predicted future identify the correct action. A full Phase 2F matrix is not
justified.

The next bounded question is narrower: can the model learn to rank the
available counterfactual actions from the same source state by navigation
utility when those utilities are supervised directly?

## Hypothesis

The current latent future may not need to reconstruct all image-aligned future
tokens before it can support navigation. A direct source-local action-utility
head may reveal whether the predicted latent future contains enough information
to choose the best action among the same source state's candidate sequences.

This is still not a deployment controller. It is a diagnostic bridge between
JEPA future prediction and navigation-relevant action choice.

## Utility Target

Each candidate row receives a masked scalar utility target derived from the
privileged generator consequence labels. The target is only valid when the row
has:

- `target_progress_m` or `clearance_gain_m`;
- `p05_swept_configuration_clearance_m`;
- `unsafe_sample_fraction`;
- `enters_grid_unsafe`;
- `ends_grid_unsafe`;
- `target_recoverable`.

The target version is:

```text
phase2g_oracle_order_utility_v0
```

The utility proxy is safety-first and source-local:

```text
utility = (
  2.0 * clipped_task_gain
  + clipped_p05_clearance
  - 8.0 * enters_grid_unsafe
  - 6.0 * ends_grid_unsafe
  - 4.0 * unrecoverable
  - 2.0 * clipped_unsafe_fraction
  - 0.25 * heading_penalty
) / 10.0
```

The score is not an absolute reward claim. It is a differentiable proxy for the
counterfactual benchmark's oracle ordering and is valid only for comparing
candidates from the same source observation.

## Model Amendment

The Phase 2G amendment adds:

- `action_utility_targets`, `action_utility_mask`, and
  `action_utility_group_ids` to each source-grouped batch;
- an action-utility head over mean-pooled predicted future spatial tokens;
- source-grouped cross-entropy that selects the highest target utility within
  each source state;
- a small masked MSE term to preserve utility scale;
- validation records comparing predicted selected action against the oracle
  selected action for each source state.

The JEPA latent prediction objective, EMA target, spatial variance floor, and
registered Phase 2D validation gate remain active. The utility head is an
auxiliary diagnostic; it does not replace the latent world-model gate.

## Promotion Gate

A bounded smoke may be promoted to a full pilot manifest only if:

- all trainer metrics and gradients remain finite;
- source-local utility records are emitted on validation;
- utility selection is non-degenerate: top-1 match rate is at least `0.25` or
  first-primitive match rate is at least `0.50`;
- no collapse, effective-rank, or near-static-target warning is raised;
- real action beats hard-negative actions by at least `0.10` of target change;
- real action beats zero action by at least `0.10` of target change;
- one-step real prediction beats persistence with ratio `< 1.0`;
- the trainer reports `checkpoint_selection_permitted: true`;
- the executable smoke gate passes.

If utility selection improves but the latent gate fails, do not launch a full
Phase 2G world-model run. That result would support a split architecture:
dedicated affordance/utility state for action choice plus a separate dynamics
state, rather than another image-aligned future-token objective.

## Bounded Smoke Command

```bash
env -u HSA_OVERRIDE_GFX_VERSION \
  ROCM_PATH=/opt/rocm-7.1.1 \
  HIP_VISIBLE_DEVICES=0 \
  PATH=/opt/rocm-7.1.1/lib/llvm/bin:/opt/rocm-7.1.1/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  /home/andrewknowles/TinyQuadJEPA/bin/python \
  scripts/train_jepa_phase2d.py \
  --train-data .generated/jepa_counterfactual/phase2d_min_sources/train_spatial_future_v1.jsonl \
  --validation-data .generated/jepa_counterfactual/phase2d_min_sources/validation_spatial_future_v1.jsonl \
  --output .generated/jepa_counterfactual/phase2d_min_sources/phase2g_action_utility_c1_lr1e4_lam1_smoke.pt \
  --cell C1 \
  --run-class smoke \
  --optimization-steps 128 \
  --evaluation-interval 64 \
  --source-states-per-batch 2 \
  --max-validation-rows 400 \
  --seed 20260614 \
  --device cuda \
  --lr 1e-4 \
  --max-grad-norm 1.0 \
  --action-utility-loss-lambda 1.0
```

The smoke-gate checker must then be run against the emitted JSON report.

## Implementation Gates Before Smoke

Focused tests:

```text
32 passed
```

Diff hygiene:

```text
git diff --check passed
```

## Bounded Smoke Results

Status: failed. Do not launch a full Phase 2G training matrix from this pilot.

The bounded ROCm GPU smoke completed and wrote artifacts, but failed both the
utility-selection promotion gate and the latent world-model gate.

Final validation utility-selection summary:

```text
source_state_count: 5
mean_candidate_rows: 80.0
top1_match_rate: 0.20
first_primitive_match_rate: 0.20
mean_target_utility_regret: 0.358380
mean_selected_target_utility: -0.266398
mean_oracle_target_utility: 0.091983
```

Final validation latent gate:

```text
checkpoint_selection_permitted: false
gate_pass: false
stability_pass: false
collapse_warning: true
effective_rank_warning: true
mean_feature_std: 0.031231
effective_rank: 2.734216
effective_rank_fraction: 0.056963
hard_negative_action_advantage: -30.852806
zero_action_advantage: 0.006918
one_step_rollout_persistence_ratio: 117.979802
```

Executable smoke-gate result:

```text
passed: false
failure_reasons:
- stability_failed
- hard_negative_action_advantage_below_threshold
- zero_action_advantage_below_threshold
- persistence_ratio_not_below_threshold
```

Artifact hashes:

```text
9051a1a303c9617cda7504417c033a0ba8887e4cda10c820af5023c93e96be1a  .generated/jepa_counterfactual/phase2d_min_sources/phase2g_action_utility_c1_lr1e4_lam1_smoke.json
dd19d13556571e7ea54c945754a09aa246c7fc609e46460a6e6e7bd515371fed  .generated/jepa_counterfactual/phase2d_min_sources/phase2g_action_utility_c1_lr1e4_lam1_smoke.pt
fec14dce4a1d9279ac3ba9bb566d929ab3fc0f9fb5737e036f75d793cf6fa596  .generated/jepa_counterfactual/phase2d_min_sources/phase2g_action_utility_c1_lr1e4_lam1_smoke_gate.json
```

## Interpretation

The utility head stayed numerically finite, but its cross-entropy remained near
the entropy of an approximately 80-way source-local candidate set. The final
top-1 utility match rate was one source state out of five and the
first-primitive match rate was also one out of five. This is not enough evidence
to claim useful action-utility learning.

More importantly, the latent state still collapsed and the real predicted
future lost to persistence by `117.98x`. Direct utility supervision attached to
mean-pooled predicted future tokens did not repair the representation failure.

This does not reject action-utility supervision. It rejects this implementation:
a source-local utility head trained on the same collapsing image-aligned latent
future.

## Decision

Stop Phase 2G at bounded smoke. No full Phase 2G run is justified.

The next diagnostic must audit the utility labels themselves before another
model variant is trained:

- measure utility-target coverage and spread per source state;
- measure how much validation utility selection can be explained by
  source-independent action-only baselines;
- only proceed to a source-conditioned affordance/utility model if action-only
  baselines do not already explain the target.
