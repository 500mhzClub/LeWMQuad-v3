# JEPA Phase 2D Generation Contract

Date: 2026-06-14

Branch: `jepa-spatial-world-model-nav`

Registration:
`docs/lewm_jepa_phase2d_preregistered_research_plan_2026-06-14.md`

Previous increment:
`docs/lewm_jepa_phase2d_stage5_split_run_readiness_2026-06-14.md`

## Scope

This increment makes the Phase 2D counterfactual data-generation contract
explicit and mechanically testable before any confirmatory training run.

It does not generate the four registered confirmatory splits and does not
inspect validation, test-ID, or test-hard model results.

## Problem

The Stage 5 readiness gate correctly rejects legacy spatial-future data because
the flattened rows do not contain:

- 81 unique two-block candidates for every source state;
- nine distinct first-action alternatives for every source state;
- topology and visual lineage fields for every source state.

The decision benchmark builder already used a factorial product internally, but
that fact was not recorded in the row contract, and later render/spatial joins
dropped topology and visual lineage. This made it possible to confuse a bounded
nine-candidate pilot artifact with a Phase 2D-ready source-state grid.

## Decisions

### Full-Factorial Grid Is A First-Class Contract

`lewm/benchmarks/phase2d_generation.py` now defines the registered primitive
set and two-block horizon:

- primitives: `hold`, `forward_slow`, `forward_medium`, `forward_fast`,
  `backward`, `yaw_left`, `yaw_right`, `arc_left`, `arc_right`;
- horizon: `2` action blocks;
- expected sequences per source state: `81`.

The helper `sequence_grid_audit` records expected, observed, unique, missing,
unexpected, duplicate, and first-action counts.

### Lineage Must Survive Every File Boundary

The benchmark builder now emits canonical `topology_seed` and `visual_seed`
fields plus `phase2d_source_state_lineage`.

The render-plan builder copies topology and visual lineage into
`counterfactual_context`.

The spatial-future dataset builder writes topology and visual lineage into the
final rows consumed by Phase 2D training and readiness checks.

### Strict Lineage Is Opt-In At Generation Time

`scripts/build_jepa_counterfactual_benchmark.py` accepts
`--require-phase2d-lineage`. With this flag, generation fails if a source state
cannot be assigned both topology and visual lineage. Without it, pilot/legacy
paths can still run, but the output rows and summaries record missing-lineage
counts.

## Implemented

New module:

- `lewm/benchmarks/phase2d_generation.py`.

Updated scripts:

- `scripts/build_jepa_counterfactual_benchmark.py`;
- `scripts/build_jepa_counterfactual_render_plans.py`;
- `scripts/build_jepa_spatial_future_dataset.py`.

New and updated tests:

- `lewm/tests/test_phase2d_generation.py`;
- `lewm/tests/test_counterfactual_render_plan.py`;
- `lewm/tests/test_spatial_future_dataset.py`.

## Verification

Focused command:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .generated/venvs/genesis_render_vulkan/bin/python -m pytest \
  lewm/tests/test_phase2d_generation.py \
  lewm/tests/test_counterfactual_render_plan.py \
  lewm/tests/test_spatial_future_dataset.py -q
```

Result: `12 passed`.

The tests verify:

- the registered primitive grid produces exactly `81` unique two-block
  sequences;
- every first action appears exactly nine times in the full grid;
- duplicate primitive names are rejected;
- missing and unexpected sequence entries are reported;
- topology and visual lineage are recovered from rows, row scene metadata, or
  scene manifests;
- render plans propagate lineage into `counterfactual_context`;
- flattened spatial-future rows preserve lineage and the sequence-grid audit.

Benchmark-script import smoke:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .generated/venvs/genesis_render_vulkan/bin/python \
  scripts/build_jepa_counterfactual_benchmark.py --help
```

Result: exit status `0`; `--require-phase2d-lineage` is exposed.

Repository command:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .generated/venvs/genesis_render_vulkan/bin/python -m pytest lewm/tests -q
```

Result: `135 passed`, `3 subtests passed`, with six existing
`belief_encoder.py` nested-tensor warnings.

Verified Stage 6 artifact manifest:

`.generated/jepa_counterfactual/phase2d_stage6_generation_contract_manifest.json`

Manifest verification: pass.

Manifest hash:

`ae0c62e19bc1b49fefed21a8f2bbab548810ab2a06f80db6b3c2ca6d56805f09`

## Real-Data Strict-Lineage Smoke

Command:

```bash
.generated/venvs/genesis_render_vulkan/bin/python \
  scripts/build_jepa_counterfactual_benchmark.py \
  --input .generated/task_aligned_decisions/train32_v2_scored.jsonl \
  --output .generated/jepa_counterfactual/phase2d_stage6_strict_lineage_generator_smoke.jsonl \
  --max-rows 1 \
  --require-phase2d-lineage
```

Result: exit status `0`.

Observed summary:

- generated source rows: `1`;
- lineage verified rows: `1`;
- missing lineage rows: `0`;
- sequences per row: `81`;
- candidate count: `81`;
- first-action count: `9`;
- full-factorial grid: pass;
- Phase 2D 81 two-block grid: pass.

Render-plan command:

```bash
.generated/venvs/genesis_render_vulkan/bin/python \
  scripts/build_jepa_counterfactual_render_plans.py \
  --input .generated/jepa_counterfactual/phase2d_stage6_strict_lineage_generator_smoke.jsonl \
  --output-root .generated/jepa_counterfactual/phase2d_stage6_strict_lineage_plans \
  --max-candidates-per-row 0 \
  --overwrite
```

Result: exit status `0`; `81` candidates and `162` endpoint frames planned.
The generated plan and first frame context preserve the same topology and
visual seeds and mark `phase2d_lineage_verified == true`.

Strict-lineage smoke manifest:

`.generated/jepa_counterfactual/phase2d_stage6_strict_lineage_smoke_manifest.json`

Manifest verification: pass.

Manifest hash:

`d4b94f078aec16e267e15f12f845f6db2a87f5d0c736da043b1f635b1d817e6e`

## Research Decision

The Stage 6 contract closes a provenance gap but does not provide confirmatory
evidence. It makes future evidence harder to mislabel:

- a Phase 2D-ready source state must contain the full two-block factorial grid;
- each row must expose topology and visual lineage to the split manifest;
- pilot downsampling remains allowed only when documented as pilot data.

Confirmatory Phase 2D remains blocked until the full four-way data generation
run exists and the Stage 5 split/readiness gates pass.

## Next Step

Completed by:
`docs/lewm_jepa_phase2d_stage7_training_start_gate_2026-06-14.md`.

The remaining next step is to generate enough strict-lineage data for all four
registered splits and run the training-start preflight before any full
confirmatory training launch.
