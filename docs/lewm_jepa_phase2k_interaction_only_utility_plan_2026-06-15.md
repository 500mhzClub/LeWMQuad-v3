# Phase 2K Interaction-Only Source-Action Utility Pilot

Date registered: 2026-06-15

Status: implementation pilot; train and validation data only; no `test_id` or
`test_hard` access.

## Trigger

Phase 2I and Phase 2J both failed by learning the same global action-prior
shortcut:

```text
selected_sequence: backward/backward for all 256 validation source states
top1_match_rate: 0.214844
first_primitive_match_rate: 0.3515625
mean_target_utility_regret: 0.323505
```

In Phase 2J, a source-action FiLM ranker and its matched trainable action-only
control produced identical validation summaries. Therefore, the scorer still
had a usable standalone action-identity path.

## Hypothesis

A utility scorer whose head receives only source-action interaction features can
remove the standalone action-prior shortcut. If the source-action model still
fails, the current RGB source encoder plus masked utility target are not enough
under this bounded training scale. If it passes, the next JEPA pilot should use
a dedicated interaction-based affordance state rather than a mean-pooled
predicted-token utility head.

## Model Amendment

Phase 2K adds `--fusion-mode interaction_only` to the existing Phase 2I trainer.
The ranker computes:

```text
source = encoder(start_rgb)[CLS]
action = action_encoder(candidate_action_sequence)
features = source * action
utility = head(features)
```

The utility head does not receive standalone `source` or standalone `action`
features. In the matched `--input-mode action_only` control, the source vector
is zeroed before interaction, so all candidate features are zero and the model
cannot rank actions by candidate identity. This control is intentionally
degenerate; it verifies that any non-trivial action ranking must come from
source-action interaction.

## Promotion Gate

Use the existing Phase 2I executable utility gate:

- validation top-1 utility match at least `0.25`;
- validation first-primitive utility match at least `0.50`;
- validation top-1 and first-primitive match above action-only baselines;
- validation regret below action-only baselines;
- source-action interaction-only smoke must exceed its matched interaction-only
  action-only control;
- no `test_id` or `test_hard` access.

Passing this gate would permit a bounded JEPA affordance-state integration
pilot. It would not yet justify a full JEPA training matrix.

## Bounded Smoke Commands

Source-action interaction-only:

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
  --output .generated/jepa_counterfactual/phase2d_min_sources/phase2k_interaction_source_action_utility_smoke.pt \
  --run-class smoke \
  --optimization-steps 256 \
  --evaluation-interval 128 \
  --source-states-per-batch 2 \
  --seed 20260614 \
  --device cuda \
  --lr 1e-4 \
  --max-grad-norm 1.0 \
  --fusion-mode interaction_only \
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
  --output .generated/jepa_counterfactual/phase2d_min_sources/phase2k_interaction_action_only_utility_control_smoke.pt \
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
  --log-every 64
```

## Implementation Gates Before Smoke

Focused tests must pass, `git diff --check` must pass, and the claims registry
must remain valid JSON before the GPU smokes.

Focused pre-smoke gates:

```text
lewm/tests/test_phase2i_utility.py: 6 passed
git diff --check: passed
jq empty docs/lewm_jepa_claims_registry_2026-06-14.json: passed
```

## Bounded Smoke Results

Status: failed. Do not integrate this Phase 2K interaction-only ranker into a
JEPA pilot.

The source-action interaction-only ROCm GPU smoke completed with finite metrics
and wrote train/validation artifacts, but reproduced the same final validation
utility summary as Phases 2I and 2J:

```text
top1_match_rate: 0.214844
first_primitive_match_rate: 0.3515625
mean_target_utility_regret: 0.323505
mean_selected_target_utility: -0.292631
mean_oracle_target_utility: 0.030874
source_state_count: 256
selected_first_primitive: backward for all 256 validation source states
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

The matched interaction-only action-only control behaved as intended: because
the source vector is zeroed before multiplication, all candidate features are
zero and the model cannot rank by action identity. It selected `hold/hold` for
all validation sources:

```text
top1_match_rate: 0.0078125
first_primitive_match_rate: 0.01171875
mean_target_utility_regret: 0.195885
mean_selected_target_utility: -0.165011
selected_first_primitive: hold for all 256 validation source states
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
fff5aec1957001485a97cb3094cd004dc6580d701fe4df60f57d984cf1868d05  .generated/jepa_counterfactual/phase2d_min_sources/phase2k_interaction_source_action_utility_smoke.json
ff28e9f9cb7cec9e293e6878f11fdfd522a1a38e86f5acd9f70a83a31b59cc73  .generated/jepa_counterfactual/phase2d_min_sources/phase2k_interaction_source_action_utility_smoke.pt
1ac56689ff10f0b23ffd9100ff5e51173b229a714bc6fc7ed5d50e7936e97b7d  .generated/jepa_counterfactual/phase2d_min_sources/phase2k_interaction_source_action_utility_smoke_gate.json

8b56a04b57df04b4ecd5cca5befa4e7ba3992046471373908b4c5385630681f8  .generated/jepa_counterfactual/phase2d_min_sources/phase2k_interaction_action_only_utility_control_smoke.json
c6ad4ada7668e79010ff75bcadfc60550f6d10f7bba5136c2a4b3dec31904f27  .generated/jepa_counterfactual/phase2d_min_sources/phase2k_interaction_action_only_utility_control_smoke.pt
cce58a00d398593411afffd42f0d3e2eb15a7f19d2dc56b73188158dd14e9b30  .generated/jepa_counterfactual/phase2d_min_sources/phase2k_interaction_action_only_utility_control_smoke_gate.json
```

## Interpretation

The architectural restriction worked for the control: standalone action identity
was no longer available when source evidence was removed. However, the
source-action model still learned a global `backward/backward` shortcut through
the interaction path. This implies the next bottleneck is not only the fusion
module; the hard argmax ranking objective can still reward a frequent global
winner even when its regret is poor on many source states.

## Decision

Stop this Phase 2K pilot at bounded smoke. It is not promotable.

The next bounded fix is Phase 2L: keep the interaction-only scorer, but replace
the hard one-hot utility cross-entropy with a soft utility-distribution loss and
increase the utility regression weight. The goal is to train against the full
within-source utility shape rather than only the single argmax.
