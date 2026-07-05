# JEPA Phase 2D Split And Run Readiness Guards

Date: 2026-06-14

Branch: `jepa-spatial-world-model-nav`

Registration:
`docs/lewm_jepa_phase2d_preregistered_research_plan_2026-06-14.md`

Previous increment:
`docs/lewm_jepa_phase2d_stage4_source_state_table_2026-06-14.md`

## Scope

This increment implements the guardrails that prevent confirmatory validation,
test-ID, or test-hard access without verified split lineage and frozen C0/C1/C2
run manifests.

It does not generate the missing confirmatory data and does not open held-out
test results.

## Decisions

### Split Lineage Is Strict

The split manifest requires source-state lineage fields for both topology and
visual seed. The accepted field aliases are intentionally narrow:

- topology: `topology_seed`, `topology_id`, `topology_hash`, `maze_seed`,
  `layout_seed`;
- visual: `visual_seed`, `visual_id`, `visual_hash`, `texture_seed`,
  `material_seed`.

If future data uses different field names, the generator must either emit one
of these fields or the protocol must be amended before any confirmatory result
is inspected.

### Test-Hard Is Gated Behind Test-ID

`test_hard` readiness requires a verified `test_id` report manifest. This
implements the preregistered access rule that test-hard is opened only after
test-ID reporting is complete.

### Cell Manifests Must Be Confirmatory

Readiness requires C0, C1, and C2 manifests with:

- matching `config.cell`;
- `config.run_class == "confirmatory"`;
- a registered `config.checkpoint_rule`;
- a verified `selected_checkpoint` artifact.

Pilot and smoke manifests cannot unlock held-out evaluation.

## Implemented

`lewm/benchmarks/phase2d_readiness.py` provides:

- canonical split-name normalization;
- source-state topology/visual lineage audits;
- optional referenced-image hashing;
- immutable split manifest construction;
- split manifest verification;
- run/test readiness checks for validation, test-ID, and test-hard.

Scripts:

- `scripts/create_jepa_phase2d_split_manifest.py`;
- `scripts/check_jepa_phase2d_readiness.py`.

## Legacy Guard Evidence

Command:

```bash
.generated/venvs/genesis_render_vulkan/bin/python \
  scripts/create_jepa_phase2d_split_manifest.py \
  --split train=.generated/jepa_counterfactual/phase2b_train_8scene_spatial_v1.jsonl \
  --split validation=.generated/jepa_counterfactual/phase2b_eval_8scene_spatial_v1.jsonl \
  --output .generated/jepa_counterfactual/phase2d_stage5_legacy_split_manifest.json
```

Exit status: `1`, expected.

The manifest records that the legacy split files hash correctly, but fail the
registered gate because:

- `test_id` and `test_hard` are missing;
- train has `8` scenes, not `32`;
- validation has `8` scenes, not `16`;
- available scenes have `8` source states, not `16`;
- no source has 81 unique two-block sequences;
- eligible first-action share is below `5%`;
- all `64` train and all `64` validation source states lack topology and
  visual lineage fields.

Readiness command:

```bash
.generated/venvs/genesis_render_vulkan/bin/python \
  scripts/check_jepa_phase2d_readiness.py \
  --split-manifest .generated/jepa_counterfactual/phase2d_stage5_legacy_split_manifest.json \
  --requested-stage test_id \
  --output .generated/jepa_counterfactual/phase2d_stage5_legacy_test_id_readiness.json
```

Exit status: `1`, expected.

The readiness artifact blocks test-ID access because:

- split manifest verification does not pass the registered gate;
- C0, C1, and C2 selected-checkpoint manifests are missing.

## Hashes

Legacy split manifest:

`0623d8a6e257268faf94ee3644f1c000bff0d1ddab5381dd8fd48bdaeec302b2`

Legacy test-ID readiness report:

`63646b51248ab987e757b4ffce68c0bd908de68f7421c03564bb7c7d9394f9ad`

Verified Stage 5 artifact manifest:

`.generated/jepa_counterfactual/phase2d_stage5_readiness_guard_manifest.json`

Manifest hash:

`094512575c88e8ec7529df8ea2d2a993ba32d0e3d90c21c1915edfa5b88e1434`

## Verification

Focused command:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .generated/venvs/genesis_render_vulkan/bin/python -m pytest \
  lewm/tests/test_phase2d_readiness.py \
  lewm/tests/test_phase2_data.py -q
```

Result: `12 passed`.

The tests verify:

- split-name canonicalization;
- topology/visual lineage failure and success;
- split manifest hash verification distinct from gate success;
- missing C2 manifest failure;
- test-ID readiness success for synthetic complete manifests;
- test-hard failure until a verified test-ID report manifest is present.

Repository command:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .generated/venvs/genesis_render_vulkan/bin/python -m pytest lewm/tests -q
```

Result: `128 passed`, `3 subtests passed`, with six existing
`belief_encoder.py` nested-tensor warnings.

## Gate Decision

The split/run-readiness guard implementation passes for smoke/pilot
infrastructure.

Stage 6 adds generator support for the full sequence-grid and lineage fields.
Confirmatory Phase 2D remains blocked until generated split files actually
contain:

1. four registered splits;
2. topology and visual lineage fields for every source state;
3. 81 unique two-block candidates for every selected source state;
4. enough scenes and source states per scene;
5. frozen C0/C1/C2 selected-checkpoint manifests after validation-only
   checkpoint selection.
