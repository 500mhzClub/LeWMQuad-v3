# JEPA Phase 2D Trainer, Controls, And Statistics Implementation

Date: 2026-06-14

Branch: `jepa-spatial-world-model-nav`

Registration:
`docs/lewm_jepa_phase2d_preregistered_research_plan_2026-06-14.md`

Corrected model foundation:
`docs/lewm_jepa_phase2d_stage2_corrected_model_2026-06-14.md`

Source-state table continuation:
`docs/lewm_jepa_phase2d_stage4_source_state_table_2026-06-14.md`

## Scope And Evidence Status

This increment implements the corrected Phase 2D pilot trainer, source-grouped
batch materialization, diagnostic state-only and action-only controls, the full
registered data gate, and cluster-aware statistical utilities.

It still does not run a confirmatory Phase 2D experiment. The only trainer
executions here are one-step smoke runs on one legacy source group from train
and one source group from validation. They verify integration and artifact
lineage; their losses are not research evidence.

## Implemented

### Source-Grouped Batch Contract

`lewm/benchmarks/phase2d_training.py` provides:

- fixed registered cell definitions for `C0`, `C1`, `C2`, `state_only`, and
  `action_only`;
- deterministic RGB tensor materialization through the per-slot placeholder
  contract;
- batch tensors for vision, real actions, transition masks, same-source
  non-identical wrong actions, wrong-action masks, and non-hold masks;
- rejection of any batch that omits hard negatives because a source state was
  split;
- compact batch-contract audits.

The trainer no longer relies on row-wise shuffling or batch-rolled action
controls.

### Diagnostic Controls

`Phase2DSpatialLeWorldModel` now supports three prediction input modes:

| Mode | Contract |
| --- | --- |
| `state_action` | normal C0/C1/C2 action-conditioned predictor |
| `state_only` | real current state, action input fixed to zero |
| `action_only` | real action, learned constant spatial state token |

Diagnostic controls cannot optimize action-identifiability losses and are
marked as not participating in C0-C2 checkpoint selection.

### Pilot Trainer

`scripts/train_jepa_phase2d.py` trains only `smoke` and `pilot` runs. It does
not expose confirmatory execution. It records:

- fixed model constants;
- optimizer constants;
- seed and device;
- cell configuration;
- train and validation data audits;
- source-grouped batch contracts;
- per-step loss and mask metrics;
- validation interface diagnostics;
- model checkpoint and JSON report.

Validation values emitted by this pilot trainer are interface diagnostics. The
confirmatory per-source-state table is implemented by the subsequent Stage 4
increment and is the authority for registered source-state comparisons.

### Registered Data Gate

`confirmatory_data_gate` in `lewm/benchmarks/phase2_data.py` now requires:

- all four splits: `train`, `validation`, `test_id`, `test_hard`;
- no scene or source-state overlap between split pairs;
- registered minimum scene counts;
- registered minimum source states per scene;
- full 81 unique two-block action sequences per source state;
- nine distinct first-action vectors per source state;
- at least 70% eligible hard-negative coverage;
- at least 5% eligible first-action share;
- explicit artifact and seed-lineage verification.

The legacy data now fails the registered gate mechanically, not only by prose.

### Statistical Utilities

`lewm/benchmarks/phase2d_statistics.py` provides:

- candidate-row aggregation to source-state experimental units;
- exact paired source-state matching between cells;
- paired differences and ratios;
- hierarchical bootstrap over scenes and source states;
- equal-weight aggregation over matched optimization seeds;
- deterministic bootstrap seeds;
- approximate cluster-aware power from bootstrap standard error;
- the registered validation checkpoint rule:
  reject unstable checkpoints, maximize hard-negative action advantage, break
  near-ties by one-step persistence ratio, then choose the earlier epoch.

Candidate rows are explicitly not bootstrapped independently.

## Real-Data Smoke Evidence

### Registered Gate Audit

Command:

```bash
.generated/venvs/genesis_render_vulkan/bin/python \
  scripts/audit_jepa_phase2_data.py \
  --dataset train=.generated/jepa_counterfactual/phase2b_train_8scene_spatial_v1.jsonl \
  --dataset validation=.generated/jepa_counterfactual/phase2b_eval_8scene_spatial_v1.jsonl \
  --legacy-batch-size 8 \
  --output .generated/jepa_counterfactual/phase2d_registered_gate_audit.json
```

Exit status: `1`, expected because the registered gate fails.

Key gate result:

| Check | Result |
| --- | --- |
| foundation gate | pass |
| required split set | fail, missing `test_id` and `test_hard` |
| train scenes | `8`, requires `32` |
| validation scenes | `8`, requires `16` |
| train source states per scene | `8`, requires `16` |
| validation source states per scene | `8`, requires `16` |
| full 81 unique two-block sequences | fail |
| eligible first-action minimum share | `0.35%` train, `0.33%` validation |
| lineage verified | fail |

Artifact hash:

`cab9c86e5c71ed3f2f89c419061343a13ae73a7cf56b32e601738b7f371c47de`

### C2 Trainer Smoke

Command:

```bash
.generated/venvs/genesis_render_vulkan/bin/python \
  scripts/train_jepa_phase2d.py \
  --train-data .generated/jepa_counterfactual/phase2b_train_8scene_spatial_v1.jsonl \
  --validation-data .generated/jepa_counterfactual/phase2b_eval_8scene_spatial_v1.jsonl \
  --output .generated/jepa_counterfactual/phase2d_stage3_c2_trainer_smoke.pt \
  --cell C2 \
  --run-class smoke \
  --optimization-steps 1 \
  --evaluation-interval 1 \
  --source-states-per-batch 1 \
  --max-train-rows 9 \
  --max-validation-rows 9 \
  --device cpu
```

Observed contract:

- rows: `9`;
- horizon: `2`;
- command dimension: `15`;
- valid transitions: `18`;
- non-hold valid transitions: `14`;
- eligible wrong-action pairs: `80`;
- eligible wrong-action transitions: `14`;
- invalid transitions in the selected smoke source group: `0`;
- all materialized frames finite: yes.

Report hash:

`25b64ae83403d69dc6b64ceebd0692a0c257f7f37a305f9f1f5a2bf08cf3a140`

Checkpoint hash:

`0b8dbd32ce9bb729f0fd35b99499e9d5d910d8acd7232815e695a09aa7923bb8`

The validation stability diagnostic triggered `collapse_warning=True`. Because
this was one untrained smoke step, the warning is evidence that the diagnostic
path is active, not evidence about model quality.

### Diagnostic-Control Smokes

The same one-step smoke command was run for `state_only` and `action_only`.

| Cell | Checkpoint selection | report hash | checkpoint hash |
| --- | --- | --- | --- |
| `state_only` | false | `5926000104e7fae4b4d14fd6db960aab4f4e1d8ded490112cbd72f7cf024ce0a` | `f31c9b01870266e47f87b0ccb11f91c7d6b892aec773a808fde13ded7af31aaf` |
| `action_only` | false | `185274f756a822bd2c6a0846989e6cbbb99016642be25840f8875af9e5149b6d` | `8c3a524e024aabd8fb4da8e40f85c715af02d6ecb105daeafe6e738114479178` |

Both controls emitted the same corrected batch contract as C2 and did not
optimize action-identifiability terms.

### Manifest

Verified manifest:

`.generated/jepa_counterfactual/phase2d_stage3_trainer_controls_smoke_manifest.json`

Manifest hash:

`d10f97333cf1a84d85d41ea1a146057e42c89e0bfe38d24274d44dd908d5be95`

The manifest verifies all tracked input data, trainer code, model code, batch
contract code, diagnostics code, smoke reports, checkpoints, and the strict
registered-gate audit.

## Verification

Focused command:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .generated/venvs/genesis_render_vulkan/bin/python -m pytest \
  lewm/tests/test_phase2_data.py \
  lewm/tests/test_audit_jepa_phase2_data.py \
  lewm/tests/test_phase2d_statistics.py \
  lewm/tests/test_phase2d_training.py \
  lewm/tests/test_phase2d_spatial_lewm.py \
  lewm/tests/test_rollout_diagnostics.py -q
```

Result: `33 passed`.

The focused tests cover:

- registered gate failures for missing splits, split sizes, balance, and
  lineage;
- strict separation between preliminary full-81 checks and the registered
  confirmatory gate;
- source-grouped batch materialization;
- rejection of split source groups;
- state-only and action-only input invariants;
- paired hierarchical bootstrap;
- seed-equal aggregation;
- checkpoint selection rule.

Repository regression is recorded in the final verification section of this
research increment.

Repository command:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .generated/venvs/genesis_render_vulkan/bin/python -m pytest lewm/tests -q
```

Result: `123 passed`, `3 subtests passed`, with six existing
`belief_encoder.py` nested-tensor warnings.

## Gate Decision

The trainer/control/statistics implementation gate passes for smoke and pilot
infrastructure:

- C0/C1/C2 and diagnostic controls are constructible;
- C2 consumes exhaustive wrong-action and zero-action masks;
- source-grouped batch materialization works on real legacy data;
- strict registered data gate exists and blocks legacy data;
- cluster-aware statistical utilities are implemented and tested.

Confirmatory Phase 2D remains blocked.

## Remaining Before Confirmatory Training

1. Generate full confirmatory data with four splits, 81 unique two-block
   candidates per source state, minimum scenes, and minimum source-state counts.
2. Add immutable split manifests with topology, visual seed, source-state, file
   hash, and lineage verification.
3. Integrate the hierarchical bootstrap and checkpoint rule into the final
   validation report.
4. Add `test_id` and `test_hard` evaluation commands that refuse to run before
   frozen C0/C1/C2 checkpoints and manifests exist.
