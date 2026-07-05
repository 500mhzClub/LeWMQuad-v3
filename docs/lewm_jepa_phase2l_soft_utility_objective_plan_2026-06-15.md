# Phase 2L Soft Utility-Distribution Objective Pilot

Date registered: 2026-06-15

Status: implementation pilot; train and validation data only; no `test_id` or
`test_hard` access.

## Trigger

Phase 2K removed the standalone action-identity path from the action-only
control, but the source-action interaction-only model still selected
`backward/backward` for all 256 validation source states:

```text
source_action_top1_match_rate: 0.214844
source_action_first_primitive_match_rate: 0.3515625
source_action_mean_target_utility_regret: 0.323505

interaction_only_action_only_top1_match_rate: 0.0078125
interaction_only_action_only_first_primitive_match_rate: 0.01171875
interaction_only_action_only_mean_target_utility_regret: 0.195885
```

The control result shows the architecture restriction worked. The remaining
failure points to the objective: hard one-hot cross-entropy rewards only the
single argmax per source group and can still prefer a frequent global winner
even when regret is poor on many source states.

## Hypothesis

Training against the full within-source utility distribution should reduce
global argmax collapse. A soft utility-distribution loss plus stronger utility
regression may preserve near-miss and regret information that hard argmax CE
throws away.

## Objective Amendment

Phase 2L keeps the Phase 2K interaction-only source-action scorer and changes
only the utility objective:

```text
ranking_loss = soft_ce
target_distribution = softmax((utility_targets - mean) / temperature)
loss = cross_entropy(target_distribution, predicted_scores)
regression_weight = 1.0
temperature = 0.25
```

The default hard-CE objective remains unchanged for earlier registered
experiments.

## Promotion Gate

Use the existing Phase 2I executable utility gate:

- validation top-1 utility match at least `0.25`;
- validation first-primitive utility match at least `0.50`;
- validation top-1 and first-primitive match above action-only baselines;
- validation regret below action-only baselines;
- source-action soft-objective smoke must exceed its matched action-only
  control;
- no `test_id` or `test_hard` access.

Passing this gate would permit a bounded JEPA affordance-state integration
pilot. It would not yet justify a full JEPA training matrix.

## Bounded Smoke Commands

Source-action interaction-only with soft utility objective:

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
  --output .generated/jepa_counterfactual/phase2d_min_sources/phase2l_soft_interaction_source_action_utility_smoke.pt \
  --run-class smoke \
  --optimization-steps 256 \
  --evaluation-interval 128 \
  --source-states-per-batch 2 \
  --seed 20260614 \
  --device cuda \
  --lr 1e-4 \
  --max-grad-norm 1.0 \
  --fusion-mode interaction_only \
  --action-utility-ranking-loss soft_ce \
  --action-utility-regression-weight 1.0 \
  --action-utility-softmax-temperature 0.25 \
  --log-every 64
```

Matched interaction-only action-only control:

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
  --output .generated/jepa_counterfactual/phase2d_min_sources/phase2l_soft_interaction_action_only_utility_control_smoke.pt \
  --run-class smoke \
  --optimization-steps 256 \
  --evaluation-interval 128 \
  --source-states-per-batch 2 \
  --seed 20260614 \
  --device cuda \
  --lr 1e-4 \
  --max-grad-norm 1.0 \
  --input-mode action_only \
  --fusion-mode interaction_only \
  --action-utility-ranking-loss soft_ce \
  --action-utility-regression-weight 1.0 \
  --action-utility-softmax-temperature 0.25 \
  --log-every 64
```

## Implementation Gates Before Smoke

Focused tests must pass, `git diff --check` must pass, and the claims registry
must remain valid JSON before the GPU smokes.

Focused pre-smoke gates:

```text
lewm/tests/test_phase2d_spatial_lewm.py lewm/tests/test_phase2i_utility.py: 26 passed
git diff --check: passed
jq empty docs/lewm_jepa_claims_registry_2026-06-14.json: passed
```

## Bounded Smoke Results

Status: failed. Do not integrate this Phase 2L objective into a JEPA pilot.

The source-action interaction-only soft-objective ROCm GPU smoke completed with
finite metrics and wrote train/validation artifacts. It changed the exact
sequence failure mode but did not pass the utility gate:

```text
top1_match_rate: 0.0
first_primitive_match_rate: 0.3515625
mean_target_utility_regret: 0.243999
mean_selected_target_utility: -0.213125
mean_oracle_target_utility: 0.030874
source_state_count: 256
```

Selection distribution:

```text
selected_sequence:
backward/yaw_left: 1
backward/yaw_right: 255

selected_first_primitive:
backward: 256
```

Executable utility-gate result:

```text
passed: false
failure_reasons:
- top1_match_rate_below_threshold
- first_primitive_match_rate_below_threshold
- top1_match_rate_not_above_action_only_baselines
- first_primitive_match_rate_not_above_action_only_baselines
- regret_not_below_action_only_baselines
```

The matched interaction-only action-only control remained degenerate as
expected:

```text
selected_sequence: hold/hold for all 256 validation source states
top1_match_rate: 0.0078125
first_primitive_match_rate: 0.01171875
mean_target_utility_regret: 0.195885
```

Control gate result:

```text
passed: false
failure_reasons:
- top1_match_rate_below_threshold
- first_primitive_match_rate_below_threshold
- top1_match_rate_not_above_action_only_baselines
- first_primitive_match_rate_not_above_action_only_baselines
- regret_not_below_action_only_baselines
```

Artifact hashes:

```text
03021283fa4692f47c6dd9887d42f39251297ded9a3aa5de5a43a75b08366e96  .generated/jepa_counterfactual/phase2d_min_sources/phase2l_soft_interaction_source_action_utility_smoke.json
d4af73b1359bfdd8a4804e0df50c94d9a5cf9ae35106605ba5f7df0609a55245  .generated/jepa_counterfactual/phase2d_min_sources/phase2l_soft_interaction_source_action_utility_smoke.pt
786cc1787ad9e3e1112ee7e019181156e328bbaeae047c42ba684f1bb81361a4  .generated/jepa_counterfactual/phase2d_min_sources/phase2l_soft_interaction_source_action_utility_smoke_gate.json

0851ba8a37697fa8f67e3dd51c052c0755538c451c4bd319f3fcfd967e7ca1fd  .generated/jepa_counterfactual/phase2d_min_sources/phase2l_soft_interaction_action_only_utility_control_smoke.json
0d1242ef97f82f982834721acc53189fa82fe869537fdc54e32c4d8e2749f7e4  .generated/jepa_counterfactual/phase2d_min_sources/phase2l_soft_interaction_action_only_utility_control_smoke.pt
eb80e79bce313b481cc4315f9afe89833a1ca1f3b99fb8b04d2cc6d11ccbf869  .generated/jepa_counterfactual/phase2d_min_sources/phase2l_soft_interaction_action_only_utility_control_smoke_gate.json
```

## Interpretation

The soft utility-distribution objective prevented the exact `backward/backward`
sequence from dominating, but it did not make action selection source-local.
The model still chose a backward-first action for every validation source, exact
top-1 fell to zero, and regret was slightly worse than the Phase 2H
first-primitive action-only baseline.

This rejects hard-CE versus soft-CE as the main missing ingredient. The current
RGB source encoder and source-local utility supervision are not producing a
usable source-conditioned affordance ranker at this bounded scale.

## Decision

Stop this Phase 2L pilot at bounded smoke. It is not promotable.

Do not launch a full JEPA training matrix from Phases 2I through 2L. The next
registered fix should stop treating the current RGB CLS utility ranker as the
affordance state. It should introduce a structured, source-local affordance
state with explicit primitive-balanced supervision and geometry-derived
calibration before re-entering JEPA latent prediction.
