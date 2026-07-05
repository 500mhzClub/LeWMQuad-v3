# Phase 2Q Factorized Affordance Ceiling Audit

Date registered: 2026-06-15

Status: true-factor ceiling passed on train/validation; no learned model; no
`test_id` or `test_hard` result use.

## Trigger

Phase 2P failed the primitive-affordance gate despite using factorized
geometry-derived labels and a safety-first selector:

```text
primitive_match_rate: 0.13671875
mean_target_utility_regret: 0.124758
selected_max_primitive_fraction: 0.609375
primitive_action_only_prior_match_rate: 0.1640625
primitive_action_only_prior_regret: 0.058599
```

Before changing the learned architecture, the selector and target geometry need
a ceiling check. If true factor targets cannot pass the primitive gate, then
the failure is in the factor contract or hand-written selection rule. If true
factor targets pass, then Phase 2P failed because the current RGB source-only
diagnostic encoder did not learn the necessary factors.

## Research Question

Using the true Phase 2O factor targets as the selector input, does the
safety-first primitive selector pass the same validation gate that learned
models must pass?

This is an oracle/ceiling audit. It is not a learned model, not a JEPA result,
and not a deployable runtime policy.

## Implemented Files

```text
lewm/benchmarks/phase2q_factorized_ceiling.py
scripts/audit_jepa_phase2q_factorized_affordance_ceiling.py
lewm/tests/test_phase2q_factorized_ceiling.py
```

## Data Hygiene

The valid Phase 2Q artifact reads only:

```text
.generated/jepa_counterfactual/phase2d_min_sources/train_spatial_future_v1.jsonl
.generated/jepa_counterfactual/phase2d_min_sources/validation_spatial_future_v1.jsonl
```

During exploratory implementation, one unscoped filesystem search printed paths
and metadata from `.generated/datagen_full/rollout/test_hard`. Those values are
not used in the Phase 2Q code, report, decision, thresholds, or claims. The
Phase 2Q JSON artifact contains only the registered train/validation inputs and
split-overlap check. No `test_id` or `test_hard` metrics are reported or used
for model selection.

## Quality Gate

Focused tests and CLI check:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .generated/venvs/genesis_render_vulkan/bin/python \
  -m pytest lewm/tests/test_phase2q_factorized_ceiling.py \
  lewm/tests/test_phase2p_factorized_affordance.py \
  lewm/tests/test_phase2o_factorized_affordance.py -q

Result: 7 passed

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .generated/venvs/genesis_render_vulkan/bin/python \
  scripts/audit_jepa_phase2q_factorized_affordance_ceiling.py --help

Result: CLI parsed successfully
```

## Real-Data Audit Command

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .generated/venvs/genesis_render_vulkan/bin/python \
  scripts/audit_jepa_phase2q_factorized_affordance_ceiling.py \
  --train-data .generated/jepa_counterfactual/phase2d_min_sources/train_spatial_future_v1.jsonl \
  --validation-data .generated/jepa_counterfactual/phase2d_min_sources/validation_spatial_future_v1.jsonl \
  --output .generated/jepa_counterfactual/phase2d_min_sources/phase2q_factorized_affordance_ceiling_audit.json \
  --seed 20260615 \
  --require-gate-pass
```

## Result

The true-factor ceiling passed the executable primitive gate.

Validation true-factor selector:

```text
primitive_match_rate: 0.8671875
mean_target_utility_regret: 0.001419918
selected_max_primitive_fraction: 0.33984375
oracle_max_primitive_fraction: 0.3515625
uniform_random_expected_primitive_match_rate: 0.111111
```

Validation primitive action-only prior:

```text
primitive_match_rate: 0.1640625
mean_target_utility_regret: 0.058599013
selected_max_primitive_fraction: 1.0
selected_primitive: yaw_left for all 256 validation source states
```

Validation gate:

```text
passed: true
failure_reasons: []
min_primitive_match_rate: 0.50
max_selected_primitive_excess: 0.20
```

Train true-factor selector:

```text
primitive_match_rate: 0.869140625
mean_target_utility_regret: 0.003351515
selected_max_primitive_fraction: 0.326171875
oracle_max_primitive_fraction: 0.314453125
```

Artifact hash:

```text
3140c18125fc625ae30c40a8c84a4801f879ff4927651a848c1b5d1468d3fc62  .generated/jepa_counterfactual/phase2d_min_sources/phase2q_factorized_affordance_ceiling_audit.json
```

## Interpretation

Phase 2Q separates two hypotheses:

1. The Phase 2O factor target plus Phase 2P safety-first selector may be
   incoherent.
2. The Phase 2P RGB source-only model may be unable to learn the factors.

The ceiling result rejects the first hypothesis for the registered
train/validation protocol. With true factor targets, the same selector strongly
passes the primitive gate and nearly eliminates validation utility regret.

Therefore, training Phase 2P longer is not the right next step. The next
learned pilot must improve how geometry enters the learned state.

## Decision

Keep the Phase 2O factor targets and Phase 2P safety-first selector as the
primitive-affordance gate contract.

Do not launch a full JEPA training run yet. The next bounded implementation
should be Phase 2R: a geometry-exposed learned affordance state. Acceptable
forms include:

- local ray/clearance tokens derived from scene geometry and source pose;
- factorized affordance slots supervised by the Phase 2O labels;
- a privileged geometry-feature ceiling/control before any RGB-only runtime
  claim;
- a subsequent RGB-to-geometry distillation test only if the geometry-exposed
  state passes the primitive gate.

The Phase 2R promotion gate should remain the unchanged primitive gate:

- primitive match rate at least `0.50`;
- primitive match rate above the primitive action-only prior;
- mean target-utility regret below the primitive action-only prior;
- selected primitive distribution not more collapsed than oracle by more than
  `0.20`;
- train/validation only until a selected checkpoint candidate exists.
