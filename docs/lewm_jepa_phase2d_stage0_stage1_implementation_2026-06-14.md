# JEPA Phase 2D Stage 0/1 Foundation Implementation

Date: 2026-06-14

Branch: `jepa-spatial-world-model-nav`

Registration:
`docs/lewm_jepa_phase2d_preregistered_research_plan_2026-06-14.md`

## Scope

This increment implements the reusable Stage 0 and Stage 1 foundations that can
be completed before generating new confirmatory train, validation, test-ID, and
test-hard data.

It deliberately does not start Phase 2D training or silently change historical
Phase 2B/2C trainer behavior.

## Implemented

### Content-Addressed Experiment Manifests

`lewm/benchmarks/experiment_manifest.py` and
`scripts/create_jepa_experiment_manifest.py` now provide:

- SHA-256 and byte-size identity for inputs and artifacts;
- Git commit, branch, dirty-state, and worktree status;
- Python, platform, and core-package environment fingerprint;
- exact run command, seeds, and structured configuration;
- post-run verification that every recorded file still matches its hash.

### Explicit Per-Slot Data Contract

`lewm/benchmarks/phase2_data.py` now provides:

- loading modes for all rows, any-valid-transition rows, and complete rows;
- future-slot validity derived from explicit observation metadata;
- teacher-forced transition validity that requires valid current and future
  observations;
- deterministic placeholder materialization for invalid images, paired with an
  authoritative transition mask;
- deterministic source-grouped batches that never split a matched source state.

Invalid placeholder images are never valid prediction targets. The interface is
implemented for the corrected Phase 2D trainer; legacy trainers remain
behavior-locked until the registered model and loss corrections are implemented
together.

### Valid Hard-Negative Construction

The Phase 2 data module now builds exhaustive hard-negative indexes with this
contract:

- same source state;
- positive transition is valid;
- positive action is non-zero for the primary action-identifiability estimand;
- negative action vector differs at the evaluated step;
- duplicate negative action vectors are removed;
- construction is independent of row or batch order.

Every exclusion reason and negative-coverage statistic is emitted.

### Reusable Data And Control Audit

`scripts/audit_jepa_phase2_data.py` emits:

- file hashes and load provenance;
- all-row versus complete-row counts;
- per-slot and teacher-forced transition validity;
- action and family distributions before and after masks;
- candidates per source state;
- exhaustive hard-negative coverage;
- reproduction of the legacy batch-rolled action-control contamination;
- pairwise scene and source-state split overlap;
- separate foundation and confirmatory-data gates.

### Frozen Checkpoint Train/Validation Diagnostic

`scripts/evaluate_jepa_phase2_checkpoint.py` loads historical pooled or spatial
Phase 2 checkpoints and evaluates the identical frozen model on any number of
named complete-valid datasets. It records checkpoint and dataset hashes and
emits the same persistence, action, collapse, and selection diagnostics used by
the historical trainer.

This converts the earlier one-off training-set diagnostic into a reproducible
analysis. It remains post-hoc evidence and cannot retroactively pass a
registered gate.

### Tracked Claims Registry

`docs/lewm_jepa_claims_registry_2026-06-14.json` records the current claim
status, evidence class, permitted wording, prohibited wording, and artifact
lineage for the central Phase 2 conclusions.

## Actual Legacy Phase 2B Audit

This Stage 0/1 audit is retained as the original foundation artifact. The
stricter registered data gate added later is documented in
`docs/lewm_jepa_phase2d_stage3_trainer_statistics_2026-06-14.md`.

Command:

```bash
python3 scripts/audit_jepa_phase2_data.py \
  --dataset train=.generated/jepa_counterfactual/phase2b_train_8scene_spatial_v1.jsonl \
  --dataset validation=.generated/jepa_counterfactual/phase2b_eval_8scene_spatial_v1.jsonl \
  --legacy-batch-size 8 \
  --output .generated/jepa_counterfactual/phase2_stage0_stage1_audit.json
```

Result:

- foundation gate: **pass**;
- confirmatory-data gate: **fail**, as expected for the legacy bounded data.

### Foundation Checks

| Check | Result |
| --- | --- |
| train/validation scene overlap | none |
| train/validation source-state overlap | none |
| identical actions admitted as constructed hard negatives | zero |
| eligible non-hold valid positives with at least one hard negative | `100%` train and validation |

### Data Retention

| Split | Planned rows | Complete-valid rows | Rows with any valid transition | Valid step-one transitions | Valid step-two transitions |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 576 | 394 | 423 | 423 | 394 |
| validation | 576 | 422 | 447 | 447 | 422 |

The corrected per-slot interface makes `29` additional training rows and `25`
additional validation rows available for at least one valid transition without
mislabeling invalid observations.

### Hard-Negative Eligibility

| Split | Eligible non-hold valid step-one positives | Unique hard negatives | Mean negatives per positive |
| --- | ---: | ---: | ---: |
| train | 289 | 1,094 | 3.79 |
| validation | 302 | 1,229 | 4.07 |

This shows that the existing candidate data can support a clean bounded
step-one action-identifiability diagnostic. It does not satisfy the
confirmatory full-candidate or action-balance contract.

### Confirmatory Failures

- every source has nine candidates, not the registered full 81;
- eligible first-action coverage is highly imbalanced;
- only eight train and eight validation scenes exist;
- no unopened test-ID or test-hard splits exist;
- no artifact and seed-lineage manifest exists for confirmatory access.

The rarest eligible step-one action is `forward_medium`, with one example in
each split. This fails the registered minimum action-share requirement.

### Legacy Control Reproduction

The reusable audit exactly reproduces the previously identified contamination:

| Split | Same-source batch-rolled control | Same step-one action | Real step-one action is zero |
| --- | ---: | ---: | ---: |
| train complete subset | `79.70%` | `42.13%` | `32.23%` |
| validation complete subset | `79.15%` | `40.52%` | `32.94%` |

### Frozen Checkpoint Train/Validation Result

| Cell | Train step-one / persistence | Validation step-one / persistence | Train step-two / persistence | Validation step-two / persistence |
| --- | ---: | ---: | ---: | ---: |
| pooled | `2.28x` | `2.07x` | `4.51x` | `4.60x` |
| regularized spatial | `3.14x` | `2.69x` | `3.72x` | `3.68x` |

Both final checkpoints lose to persistence on their own complete-valid training
rows. This supports an optimization/objective-failure interpretation rather
than a pure held-out-generalization explanation.

Generated diagnostic hashes:

- pooled:
  `f7a9aec5b5ab1cb7c60e6572b4ce90bd29f2bd5b5e4fcc1fd0f72fdd1b2f1fe3`;
- regularized spatial:
  `44405e39b75ed6206a2c02dce0308d5c799d3fbec7f1e4ed0651e0f7e5ef70ea`.

## Verification

Focused tests cover:

- per-slot and transition validity;
- partial-row retention;
- deterministic invalid-slot substitution and masking;
- same-source non-identical hard negatives;
- source-grouped batches;
- split-overlap detection independent of split labels;
- content-addressed manifests and change detection;
- separation of foundation and confirmatory audit gates.

Verification command:

```bash
python3 -m pytest \
  lewm/tests/test_phase2_data.py \
  lewm/tests/test_experiment_manifest.py \
  lewm/tests/test_audit_jepa_phase2_data.py -q
```

Result: `10 passed`.

Repository-level regression command:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .generated/venvs/genesis_render_vulkan/bin/python -m pytest lewm/tests -q
```

Result: `100 passed`, `3 subtests passed`, with six existing
`belief_encoder.py` nested-tensor warnings.

The first repository-level attempt was blocked before test collection by an
auto-loaded ROS `launch_testing` plugin whose environment lacked `lark`.
Disabling unrelated third-party pytest plugin auto-loading allowed the
repository suite to run without changing dependencies.

## Remaining Before Phase 2D Training

1. Generate full-81-candidate source states across all four registered splits.
2. Emit topology and visual lineage fields for every selected source state.
3. Freeze confirmatory split manifests that pass the registered readiness gate.
4. Freeze immutable C0/C1/C2 run manifests before inspecting confirmatory
   validation results.

The corrected spatial model and diagnostics are implemented and verified in
`docs/lewm_jepa_phase2d_stage2_corrected_model_2026-06-14.md`.

The corrected trainer, diagnostic controls, registered data gate, and
cluster-aware statistics utilities are implemented and verified in
`docs/lewm_jepa_phase2d_stage3_trainer_statistics_2026-06-14.md`.

The trainer-side per-source-state prediction/control table is implemented and
verified in
`docs/lewm_jepa_phase2d_stage4_source_state_table_2026-06-14.md`.

The split-manifest and run-readiness guards are implemented and verified in
`docs/lewm_jepa_phase2d_stage5_split_run_readiness_2026-06-14.md`.

The registered decision remains: do not start Phase 2D training until these
remaining gates pass.
