# Phase 2D Stage 8: Source Selection And Render Readiness

Date: 2026-06-14

## Status

Stage 8 completed the registered-minimum source-state selection and full
counterfactual render-plan construction needed before Phase 2D training data can
be materialized.

This document records the source-selection, render-planning, and render-resume
increment. It is superseded for final training readiness by:

```text
docs/lewm_jepa_phase2d_stage9_training_ready_2026-06-14.md
```

The initial Stage 8 render-readiness gate correctly failed before full
rendering. The later Stage 9 gate records the completed render, rebuilt
spatial-future datasets, verified split manifest, C2 training-start preflight,
and frozen primary run manifests.

No Phase 2D model has been trained or evaluated on the held-out splits in this
stage. Held-out files were opened only to build preregistered source indices,
verify lineage/disjointness, and write render plans.

## Research Purpose

The Phase 2D confirmatory experiment requires a dataset in which each source
observation supports all two-block action sequences under a common physical
source state. The purpose of this stage was to remove remaining ambiguity before
expensive rendering:

- every split must have enough task-aligned source states;
- source states must have valid local target observations when target labels are
  present;
- topology and visual lineage must be present and split-disjoint;
- selected source states must meet the registered minimum exactly;
- every selected source state must expand to the full `9 x 9 = 81` two-block
  counterfactual action grid;
- training must remain blocked until all planned renders exist and are valid.

## Implemented Gate

New source-index readiness code:

- `lewm/benchmarks/phase2d_source_indices.py`;
- `scripts/check_jepa_phase2d_source_indices.py`;
- `lewm/tests/test_phase2d_source_indices.py`.

This gate verifies:

- all confirmatory splits are present;
- split labels canonicalize correctly;
- source rows with target labels have a local target frame;
- scene manifests are present;
- Phase 2D topology and visual lineage verify;
- each split meets minimum scene and source-state counts;
- scene IDs, source-state keys, and topology/visual seed pairs do not overlap
  across splits.

New deterministic source selection code:

- `lewm/benchmarks/phase2d_source_selection.py`;
- `scripts/select_jepa_phase2d_source_rows.py`;
- `lewm/tests/test_phase2d_source_selection.py`.

Selection is deterministic under seed `20260614`. It balances scene choice by
family where available, shuffles eligible rows within each selected scene, and
writes exactly the registered minimum:

| Split | Scenes | Source states per scene | Source rows |
| --- | ---: | ---: | ---: |
| train | 32 | 16 | 512 |
| validation | 16 | 16 | 256 |
| test_id | 16 | 16 | 256 |
| test_hard | 16 | 16 | 256 |

New render-readiness code:

- `lewm/benchmarks/phase2d_render_readiness.py`;
- `scripts/check_jepa_phase2d_render_readiness.py`;
- `lewm/tests/test_phase2d_render_readiness.py`.

This gate verifies, for every confirmatory split:

- plan summary exists and has schema
  `jepa_counterfactual_render_plan_summary_v0`;
- render root summary exists and has schema
  `jepa_counterfactual_render_root_summary_v0`;
- render root summary references the expected plan root and output root;
- rendered scene count matches planned scene count;
- rendered frame count matches planned frame count;
- per-scene frame sums match the root summary;
- invalid frame counts reconcile between per-scene summaries and the root
  summary;
- per-scene render return codes are accepted accounting statuses.

Important correction: the render-readiness gate is an accounting gate, not a
claim that every counterfactual future observation is visually valid. The
spatial-future dataset contract explicitly records invalid future observations
and masks them from prediction loss. Therefore all planned frames must be
accounted for, but invalid all-candidate futures are allowed and are audited
separately through `all_rendered_frames_valid` and downstream data coverage
statistics.

## Source-Index Readiness

Command:

```bash
.generated/venvs/genesis_render_vulkan/bin/python \
  scripts/check_jepa_phase2d_source_indices.py \
  --source train=.generated/task_aligned_decisions/train32_v2_scored.jsonl \
  --source validation=.generated/task_aligned_decisions/val32_v2_scored.jsonl \
  --source test_id=.generated/task_aligned_decisions/test_id16_phase2d_v2_scored.jsonl \
  --source test_hard=.generated/task_aligned_decisions/test_hard16_phase2d_v2_scored.jsonl \
  --output .generated/jepa_counterfactual/phase2d_stage8_source_index_readiness_all_splits.json
```

Result: exit status `0`;
`ready_for_counterfactual_generation: true`.

Hash:

```text
024b620602c0c3a43943213fed586cb7a30e361322450390503f8123ff741035  .generated/jepa_counterfactual/phase2d_stage8_source_index_readiness_all_splits.json
```

Observed source-index counts:

| Split | Rows | Eligible rows | Scenes | Min eligible rows per scene | Missing local-target rows skipped | Lineage |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| train | 16,384 | 14,890 | 32 | 250 | 1,494 | verified |
| validation | 16,384 | 14,381 | 32 | 255 | 2,003 | verified |
| test_id | 7,380 | 6,762 | 16 | 305 | 618 | verified |
| test_hard | 7,234 | 6,847 | 16 | 264 | 387 | verified |

Pairwise scene, source-state, and topology/visual lineage overlap checks passed.

## Selected Source Rows

Command:

```bash
.generated/venvs/genesis_render_vulkan/bin/python \
  scripts/select_jepa_phase2d_source_rows.py \
  --source train=.generated/task_aligned_decisions/train32_v2_scored.jsonl \
  --source validation=.generated/task_aligned_decisions/val32_v2_scored.jsonl \
  --source test_id=.generated/task_aligned_decisions/test_id16_phase2d_v2_scored.jsonl \
  --source test_hard=.generated/task_aligned_decisions/test_hard16_phase2d_v2_scored.jsonl \
  --output-dir .generated/task_aligned_decisions/phase2d_selected_sources \
  --output-summary .generated/task_aligned_decisions/phase2d_selected_sources/summary.json
```

Result: exit status `0`; `passes_registered_minimum: true`.

Hash:

```text
e9cedc913cad474dc4fdbddb0162e40915c6133d4841876e79b06e00a0754812  .generated/task_aligned_decisions/phase2d_selected_sources/summary.json
```

Selected-source readiness command:

```bash
.generated/venvs/genesis_render_vulkan/bin/python \
  scripts/check_jepa_phase2d_source_indices.py \
  --source train=.generated/task_aligned_decisions/phase2d_selected_sources/train_phase2d_sources.jsonl \
  --source validation=.generated/task_aligned_decisions/phase2d_selected_sources/validation_phase2d_sources.jsonl \
  --source test_id=.generated/task_aligned_decisions/phase2d_selected_sources/test_id_phase2d_sources.jsonl \
  --source test_hard=.generated/task_aligned_decisions/phase2d_selected_sources/test_hard_phase2d_sources.jsonl \
  --output .generated/jepa_counterfactual/phase2d_stage8_selected_source_index_readiness.json
```

Result: exit status `0`;
`ready_for_counterfactual_generation: true`.

Hash:

```text
dcb04d501315e62cac88c33bf2fe2ae0e792f896983a61b31108ccf12694e4cb  .generated/jepa_counterfactual/phase2d_stage8_selected_source_index_readiness.json
```

The selected rows contain no skipped local-target rows and verified lineage in
all four splits.

## Counterfactual Decision Benchmarks

Each selected source state was expanded into all `81` two-block action
sequences using the strict lineage contract.

| Split | Source rows | Scenes | Candidates | Summary hash |
| --- | ---: | ---: | ---: | --- |
| train | 512 | 32 | 41,472 | `7b7090fdc94abd3a20b7a903f7e9e282bec3430e96860b37e7ab4b9678b87252` |
| validation | 256 | 16 | 20,736 | `0bb2d6c8ed900109b362a7bb5a2b63f4dc7531ee544532efeb55aeff0a796a29` |
| test_id | 256 | 16 | 20,736 | `16a2e1f22fff3b95aaffc5107129b56246cc318889553e5de07e064b23236675` |
| test_hard | 256 | 16 | 20,736 | `bef2ccafb446d94db22e31437f2cdaf3195dc1cbeebb34d883a9ce03abef6da6` |

Total: `1,280` source rows, `80` scenes, `103,680` candidate sequences.

## Render Plans

All candidates were planned for rendering with `--max-candidates-per-row 0`.

| Split | Plan scenes | Candidate sequences | Future frames | Summary hash |
| --- | ---: | ---: | ---: | --- |
| train | 32 | 41,472 | 82,944 | `08007cbc02f537a01fba1bdf53757f883549835f1fa8818e071bfe9a720b6831` |
| validation | 16 | 20,736 | 41,472 | `d295d7281e420c6d9fe8461bb5bb9c9d9c6a2de277b411f24c615dfa5ec240d4` |
| test_id | 16 | 20,736 | 41,472 | `84570215517c3af08cff3b7198860abc027d7923c5dbf49047cab2d62aebf556` |
| test_hard | 16 | 20,736 | 41,472 | `5df02c216cfce361a6d8621ba1ae66d5f9af728f1c0d55d5fa464938201cdc74` |

Total planned render load: `207,360` future frames.

## Render-Readiness Result

Command:

```bash
PYTHONPATH=/home/andrewknowles/Workspace/LeWMQuad-v3 \
.generated/venvs/genesis_render_vulkan/bin/python \
  scripts/check_jepa_phase2d_render_readiness.py \
  --plan-root train=.generated/jepa_counterfactual/phase2d_min_sources/train_plans \
  --plan-root validation=.generated/jepa_counterfactual/phase2d_min_sources/validation_plans \
  --plan-root test_id=.generated/jepa_counterfactual/phase2d_min_sources/test_id_plans \
  --plan-root test_hard=.generated/jepa_counterfactual/phase2d_min_sources/test_hard_plans \
  --render-root train=.generated/jepa_counterfactual/phase2d_min_sources/train_render \
  --render-root validation=.generated/jepa_counterfactual/phase2d_min_sources/validation_render \
  --render-root test_id=.generated/jepa_counterfactual/phase2d_min_sources/test_id_render \
  --render-root test_hard=.generated/jepa_counterfactual/phase2d_min_sources/test_hard_render \
  --output .generated/jepa_counterfactual/phase2d_stage8_render_readiness_before_full_render.json
```

Result: exit status `1`, expected.

Hash:

```text
2eb142c6338d7641c99c72049714e725354c1b511451c55b2d67e90430234936  .generated/jepa_counterfactual/phase2d_stage8_render_readiness_before_full_render.json
```

The report confirms:

- all required plan roots are present;
- all required render-root paths are named;
- no full render root summary exists yet for train, validation, test-ID, or
  test-hard;
- `ready_for_spatial_future_join: false`.

This is a correct fail. The repo is ready to commence the full rendering job,
not the full training job.

## Full Render Attempt And Correction

Initial full-render launch:

```bash
EGL_DEVICE_ID=0 PYOPENGL_PLATFORM=egl \
GENESIS_ROCM_PYTHON=$PWD/.generated/venvs/genesis_render_vulkan/bin/python \
.generated/venvs/genesis_render_vulkan/bin/python \
  scripts/render_jepa_counterfactual_plan_root.py \
  --plan-root .generated/jepa_counterfactual/phase2d_min_sources/train_plans \
  --output-root .generated/jepa_counterfactual/phase2d_min_sources/train_render \
  --scene-corpus .generated/scene_corpus/minimum_tex_20260520T211541Z \
  --backend vulkan \
  --camera-mode replay \
  --replay-env-mode single \
  --rgb-format png \
  --store-resolution training
```

This wrote train render data under the workspace filesystem and stopped with
`OSError: [Errno 28] No space left on device` during train scene `23/32`.

Partial output was moved, not deleted:

```text
/tmp/lewm_phase2d_min_sources_render_20260614/train_render
```

Partial render evidence:

- completed scene summaries: `22`;
- rendered future frames in completed summaries: `57,024`;
- invalid future frames in completed summaries: `15,939`;
- low-info frames: `15,102`;
- camera-safety unresolved frames: `2,841`;
- partial output size: `69G`.

The first completed train scene had `2,592` frames and `910` invalid frames,
mostly `near_forward_geometry`, `near_wall_depth`, `low_rgb_texture`, and
`camera_safety_unresolved`. This exposed an error in the first version of the
render-readiness gate: zero invalid frames is too strict for an all-candidate
counterfactual dataset because unsafe or visually degenerate futures are
legitimate outcomes. The corrected gate requires complete accounting and leaves
valid-transition sufficiency to the dataset audit.

Corrected partial-readiness command:

```bash
PYTHONPATH=/home/andrewknowles/Workspace/LeWMQuad-v3 \
.generated/venvs/genesis_render_vulkan/bin/python \
  scripts/check_jepa_phase2d_render_readiness.py \
  --plan-root train=.generated/jepa_counterfactual/phase2d_min_sources/train_plans \
  --plan-root validation=.generated/jepa_counterfactual/phase2d_min_sources/validation_plans \
  --plan-root test_id=.generated/jepa_counterfactual/phase2d_min_sources/test_id_plans \
  --plan-root test_hard=.generated/jepa_counterfactual/phase2d_min_sources/test_hard_plans \
  --render-root train=/tmp/lewm_phase2d_min_sources_render_20260614/train_render \
  --render-root validation=/tmp/lewm_phase2d_min_sources_render_20260614/validation_render \
  --render-root test_id=/tmp/lewm_phase2d_min_sources_render_20260614/test_id_render \
  --render-root test_hard=/tmp/lewm_phase2d_min_sources_render_20260614/test_hard_render \
  --output .generated/jepa_counterfactual/phase2d_stage8_render_readiness_after_partial_render.json
```

Result: exit status `1`, expected. The train split has per-scene summaries but
no split-level `root_summary.json`; validation, test-ID, and test-hard are not
rendered yet.

Hash:

```text
9c58206c1d2ca2fb9263831e31cf31a3172fc22287a1ee9164293a2c14d060ab  .generated/jepa_counterfactual/phase2d_stage8_render_readiness_after_partial_render.json
```

## Parallel Render Resume

The first resume used the original split renderer, which processes scenes
sequentially. CPU utilization stayed low because each scene render is an
independent subprocess-bound workload. I added a scene-level parallel wrapper:

```text
scripts/render_jepa_counterfactual_plan_root_parallel.py
```

Operational contract:

- the output schema remains `jepa_counterfactual_render_root_summary_v0`;
- existing per-scene `summary.json` files are reused unless `--overwrite` is
  passed;
- reused scenes infer `render_return_code` from `invalid_frame_count`, matching
  the sequential wrapper's accounting convention;
- newly rendered scene stdout/stderr is written to
  `<render-root>/_parallel_logs/<split>/<family>/<scene>.log`;
- accepted scene render exit codes remain `0` and `2`;
- unexpected exit codes are recorded in the split root summary `failures` list
  and make the wrapper return non-zero.

I also added a reproducible Phase 2D runner:

```text
scripts/run_jepa_phase2d_min_sources_parallel_render.sh
```

Default command:

```bash
scripts/run_jepa_phase2d_min_sources_parallel_render.sh
```

The runner defaults to rendering the remaining validation, test-ID, and
test-hard splits under:

```text
/tmp/lewm_phase2d_min_sources_render_20260614
```

It then runs `scripts/check_jepa_phase2d_render_readiness.py` over train,
validation, test-ID, and test-hard. Train is included in the readiness gate even
though it was already complete, because training must be blocked unless all four
split roots are fully accounted.

Concurrency decision:

- `--jobs 16` is the current bounded default;
- the host has a 16-core/32-thread CPU, but each Genesis/Vulkan scene process also
  consumes GPU/driver memory and writes thousands of PNG/depth files;
- `16` matches physical cores while leaving SMT, IO, and driver overhead as
  headroom;
- `32` is not the first confirmed setting because it would launch one full
  Genesis/Vulkan renderer per hardware thread, which is more likely to saturate
  render memory or disk bandwidth than to halve wall-clock time;
- if this pass is stable, future render jobs can raise the explicit `--jobs`
  knob rather than changing code.

Validation at relaunch:

- train split root summary already complete: `32` scenes, `82,944` frames,
  `25,230` invalid frames;
- validation reused `7/16` completed scene summaries from the sequential
  attempt;
- the first parallel attempt used `--jobs 4`; after confirming the host exposes
  `32` hardware threads, about `67-70 GiB` available memory during the active
  batch, and roughly `485 GiB` free on `/tmp`, it was superseded by a
  sixteen-worker run.

First confirmed parallel result:

- validation completed with `16/16` scenes, `41,472` frames, `14,382` invalid
  frames, and `failure_count: 0`;
- validation wrote
  `/tmp/lewm_phase2d_min_sources_render_20260614/validation_render/root_summary.json`;
- test-ID then launched all `16/16` scenes concurrently;
- during the full test-ID fan-out, memory still had approximately `40 GiB`
  available and `/tmp` had approximately `458 GiB` free.

Second confirmed parallel result:

- test-ID completed with `16/16` scenes, `41,472` frames, `11,604` invalid
  frames, and `failure_count: 0`;
- test-ID wrote
  `/tmp/lewm_phase2d_min_sources_render_20260614/test_id_render/root_summary.json`;
- test-hard then launched all `16/16` scenes concurrently;
- during the test-hard fan-out, memory still had approximately `41 GiB`
  available and `/tmp` had approximately `419 GiB` free.

Verification before launch:

```bash
python3 -m pytest lewm/tests/test_counterfactual_render_root.py
bash -n scripts/run_jepa_phase2d_min_sources_parallel_render.sh
scripts/run_jepa_phase2d_min_sources_parallel_render.sh --dry-run
```

Focused result: `4 passed`; runner syntax and dry-run command expansion passed.

Stage 9 records the completed terminal sequence after final render readiness:

1. build spatial-future datasets for train, validation, test-ID, and test-hard;
2. create the Phase 2D split manifest;
3. run the training-start preflight;
4. freeze full primary run manifests before confirmatory training.

## Quality Gates

Stage 8 artifact manifest:

```text
4f87ecd43dcc97b0c414bab31d493bf22ca8031b1d0f85e1985a6813d843ebcf  .generated/jepa_counterfactual/phase2d_stage8_source_selection_render_readiness_manifest.json
```

Manifest verification:

- file artifacts verified: `36`;
- failed file-artifact hashes: `0`;
- render-plan directory tree hashes recorded for train, validation, test-ID,
  and test-hard.
- the Stage 8 note itself is intentionally excluded from the manifest to avoid
  a self-referential manifest-hash record.

Focused Stage 8 tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
.generated/venvs/genesis_render_vulkan/bin/python -m pytest \
  lewm/tests/test_phase2d_source_indices.py \
  lewm/tests/test_phase2d_source_selection.py \
  lewm/tests/test_phase2d_render_readiness.py \
  -q
```

Current focused result after correcting render readiness: source-index,
selection, and render-readiness tests passed with `8 passed`.

Local repository suite:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
.generated/venvs/genesis_render_vulkan/bin/python -m pytest lewm/tests -q
```

Result: `144 passed, 6 warnings, 3 subtests passed`.

Diff hygiene:

```bash
git diff --check
```

Result: passed.

Unrestricted root-level `pytest` is not a valid gate in this workspace because
it collects installed package trees under `install/`, `lewm_genesis/`, and
`lewm_worlds` that are not importable under the current Phase 2D venv layout.
The enforced gate for this increment is the local `lewm/tests` suite plus the
explicit generated-artifact readiness checks above.
