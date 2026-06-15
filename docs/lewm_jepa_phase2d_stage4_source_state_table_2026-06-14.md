# JEPA Phase 2D Source-State Prediction Table Implementation

Date: 2026-06-14

Branch: `jepa-spatial-world-model-nav`

Registration:
`docs/lewm_jepa_phase2d_preregistered_research_plan_2026-06-14.md`

Trainer/statistics foundation:
`docs/lewm_jepa_phase2d_stage3_trainer_statistics_2026-06-14.md`

Split/run guard continuation:
`docs/lewm_jepa_phase2d_stage5_split_run_readiness_2026-06-14.md`

## Scope

This increment implements the trainer-side per-source-state prediction/control
table required by the Phase 2D statistical analysis plan. It remains a smoke
and infrastructure result, not a confirmatory model result.

## Implemented

`lewm/benchmarks/phase2d_training.py` now emits:

- candidate-step prediction/control rows for every valid transition;
- primary one-step source-state rows aggregated over candidate actions;
- checkpoint-rule records derived from source-state records and stability
  diagnostics;
- JSON-safe `null` values for undefined ratios when target change is zero.

The candidate-step rows include:

- seed, scene, source state, candidate index, row index, and horizon step;
- primitive name and zero-action flag;
- valid-transition, hard-negative, and zero-action eligibility;
- real-action MSE;
- persistence MSE;
- mean same-source wrong-action MSE;
- zero-action MSE;
- target-change MSE;
- rollout/persistence ratio;
- hard-negative and zero-action advantages;
- advantages normalized by target change when defined.

The primary source-state table aggregates step-one candidate rows into the
registered experimental unit. Candidate rows are not treated as independent
samples.

## Smoke Evidence

Command:

```bash
.generated/venvs/genesis_render_vulkan/bin/python \
  scripts/train_jepa_phase2d.py \
  --train-data .generated/jepa_counterfactual/phase2b_train_8scene_spatial_v1.jsonl \
  --validation-data .generated/jepa_counterfactual/phase2b_eval_8scene_spatial_v1.jsonl \
  --output .generated/jepa_counterfactual/phase2d_stage4_c2_source_state_table_smoke.pt \
  --cell C2 \
  --run-class smoke \
  --optimization-steps 1 \
  --evaluation-interval 1 \
  --source-states-per-batch 1 \
  --max-train-rows 9 \
  --max-validation-rows 9 \
  --device cpu
```

Observed validation output:

| Artifact field | Count |
| --- | ---: |
| candidate-step prediction/control records | `18` |
| primary source-state records | `1` |
| checkpoint-rule records | `1` |

The saved JSON report contains no `Infinity` or `NaN` tokens. Undefined ratios
are represented as `null`.

The checkpoint-rule record for this one-step smoke reports:

- `stability_pass=false`;
- hard-negative action advantage over target change: `-101.5425`;
- one-step rollout/persistence ratio: `201.1876`;
- source-state count: `1`.

This is expected for an untrained one-step smoke and is not a model-quality
claim.

Report hash:

`648fe0c04a1751f45092efd90047a5dd09626f431d9dd47e89ec81806954fad9`

Checkpoint hash:

`8fe51f8cd2edb22e1c693338b21527104900cd4524d515da90f8a75d7f68049c`

Verified manifest:

`.generated/jepa_counterfactual/phase2d_stage4_source_state_table_smoke_manifest.json`

Manifest hash:

`1ec4056c55bfbf33818d0c72930e682c5d91e3f8a0d23547682bc5028a8c7020`

## Verification

Focused tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .generated/venvs/genesis_render_vulkan/bin/python -m pytest \
  lewm/tests/test_phase2d_training.py \
  lewm/tests/test_phase2d_statistics.py -q
```

Result: `9 passed`.

The tests verify:

- candidate-step row values;
- target-change-normalized advantages;
- source-state aggregation;
- eligible wrong-action and zero-action counts;
- checkpoint-rule record construction;
- paired bootstrap utilities remain compatible with source-state records.

Repository command:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .generated/venvs/genesis_render_vulkan/bin/python -m pytest lewm/tests -q
```

Result: `124 passed`, `3 subtests passed`, with six existing
`belief_encoder.py` nested-tensor warnings.

## Gate Decision

The per-source-state table implementation gate passes for smoke/pilot
infrastructure.

Confirmatory Phase 2D remains blocked until:

1. full four-way data splits exist with registered scene/source-state sizes;
2. every selected source state has 81 unique two-block candidates;
3. immutable split lineage manifests verify topology, visual seed, source
   state, and file hashes;
4. C0/C1/C2 manifests are frozen before validation results are inspected;
5. the split/readiness guards pass with frozen selected checkpoints.
