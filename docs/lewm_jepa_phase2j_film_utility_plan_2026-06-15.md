# Phase 2J FiLM Source-Action Utility Interaction Pilot

Date registered: 2026-06-15

Status: implementation pilot; train and validation data only; no `test_id` or
`test_hard` access.

## Trigger

Phase 2I showed that a source/action concatenation utility ranker did not
measurably use the source observation. A matched trainable action-only control
reached the same validation result:

```text
top1_match_rate: 0.214844
first_primitive_match_rate: 0.3515625
mean_target_utility_regret: 0.323505
selected_first_primitive: backward for all 256 validation source states
```

Therefore, simple source/action concatenation is insufficient. The next bounded
fix must force source-action interaction and keep a matched action-only control.

## Hypothesis

A FiLM-style source-conditioned action scorer can use source observation
features to modulate candidate action embeddings and reduce global action-prior
collapse. If it still matches the action-only control, source-conditioned
utility prediction is not ready to be integrated into a JEPA world model.

## Model Amendment

Phase 2J changes only the utility-ranker fusion path:

- encode the current source observation into a source vector;
- encode the candidate action sequence into an action vector;
- predict FiLM parameters from the source vector;
- score `[source, conditioned_action, source * action]`.

The matched action-only control uses the same architecture with the source
vector zeroed before conditioning. This preserves a fair trainable action-prior
comparison.

## Promotion Gate

Use the existing Phase 2I executable utility gate:

- validation top-1 utility match at least `0.25`;
- validation first-primitive utility match at least `0.50`;
- validation top-1 and first-primitive match above action-only baselines;
- validation regret below action-only baselines;
- source-action FiLM smoke must exceed its matched FiLM action-only control;
- no `test_id` or `test_hard` access.

Passing this gate would permit a bounded JEPA affordance-state integration
pilot, not a full JEPA training matrix.

## Bounded Smoke Commands

Source-action FiLM:

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
  --output .generated/jepa_counterfactual/phase2d_min_sources/phase2j_film_source_action_utility_smoke.pt \
  --run-class smoke \
  --optimization-steps 256 \
  --evaluation-interval 128 \
  --source-states-per-batch 2 \
  --seed 20260614 \
  --device cuda \
  --lr 1e-4 \
  --max-grad-norm 1.0 \
  --fusion-mode film_interaction \
  --log-every 64
```

Matched action-only FiLM control:

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
  --output .generated/jepa_counterfactual/phase2d_min_sources/phase2j_film_action_only_utility_control_smoke.pt \
  --run-class smoke \
  --optimization-steps 256 \
  --evaluation-interval 128 \
  --source-states-per-batch 2 \
  --seed 20260614 \
  --device cuda \
  --lr 1e-4 \
  --max-grad-norm 1.0 \
  --input-mode action_only \
  --fusion-mode film_interaction \
  --log-every 64
```

## Implementation Gates Before Smoke

Focused tests must pass, `git diff --check` must pass, and the claims registry
must remain valid JSON before the GPU smokes.

## Bounded Smoke Results

Status: failed. Do not integrate this Phase 2J FiLM ranker into a JEPA pilot.

Both ROCm GPU smokes completed with finite metrics and wrote train/validation
artifacts. The source-action FiLM model and its matched action-only FiLM
control produced the same final validation utility summary:

```text
top1_match_rate: 0.214844
first_primitive_match_rate: 0.3515625
mean_target_utility_regret: 0.323505
mean_selected_target_utility: -0.292631
mean_oracle_target_utility: 0.030874
source_state_count: 256
```

Executable utility-gate result for both runs:

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
selected_sequence:
backward/backward: 256

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
6043230e494c7279522e6949cfb307a49afa1323ebe05b17b05109e215fce812  .generated/jepa_counterfactual/phase2d_min_sources/phase2j_film_source_action_utility_smoke.json
05e9792a864d68ec1489336cd94c9105669a5ae961128ddc663bb90615eb0619  .generated/jepa_counterfactual/phase2d_min_sources/phase2j_film_source_action_utility_smoke.pt
cf7c8f2d2d54c3425137c2b30edf8c9ecc86182a94d27b614214bc1fd12dbbf7  .generated/jepa_counterfactual/phase2d_min_sources/phase2j_film_source_action_utility_smoke_gate.json

1e07daea66bea26b0f896c4fb77bf6f86543f240a6f6cb9fd692129a73fe5615  .generated/jepa_counterfactual/phase2d_min_sources/phase2j_film_action_only_utility_control_smoke.json
8fcc2c98587c7b9b488e6f6fe0d499bd5053f0f931cab6075abb78837c862701  .generated/jepa_counterfactual/phase2d_min_sources/phase2j_film_action_only_utility_control_smoke.pt
09d6b49eb34d1c086879e50fd3f7bea7946f3b05b7f175753ad9cf263711d9d9  .generated/jepa_counterfactual/phase2d_min_sources/phase2j_film_action_only_utility_control_smoke_gate.json
```

## Interpretation

The FiLM interaction did not force use of source observation. The source-action
model and the matched trainable action-only control converge to the same global
`backward/backward` policy, matching the Phase 2I failure signature.

This rejects FiLM plus standalone action features as a sufficient
source-conditioned utility scorer. It does not reject source-conditioned
affordance learning. It shows the scorer must remove or explicitly control
standalone action-prior paths before any JEPA integration.

## Decision

Stop this Phase 2J pilot at bounded smoke. It is not promotable.

The next bounded fix is Phase 2K: an interaction-only source-action scorer where
the utility head receives only source-action interaction features, not standalone
action features. Its matched action-only control should be intentionally unable
to rank by action identity; passing then requires the source-conditioned model
to learn source-local action utility from the observation/action interaction.
