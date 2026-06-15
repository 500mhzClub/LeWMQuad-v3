# Phase 2D Stage 9: Training-Ready Gate

Date: 2026-06-14

## Status

Phase 2D is ready to commence the registered primary full-training matrix for
C0, C1, and C2 on train and validation data only.

No full Phase 2D training run has been started in this stage. No test-ID or
test-hard result has been opened. The only trainer execution was a one-step C2
interface smoke that used capped train/validation rows and is explicitly marked
as non-confirmatory result evidence.

## Parallel Render Decision

The first full render was launched sequentially and stopped when the workspace
filesystem ran out of space. The partial train render was moved to:

```text
/tmp/lewm_phase2d_min_sources_render_20260614
```

The render was then resumed with a scene-level parallel wrapper:

```text
scripts/render_jepa_counterfactual_plan_root_parallel.py
```

The correct default for this host is now `--jobs 16`, not `8`.

Rationale:

- the host exposes `32` hardware threads;
- each job launches a full Genesis/Vulkan scene-render subprocess;
- `16` keeps all physical cores busy while leaving SMT, filesystem, memory, and
  graphics-driver headroom;
- the completed full render stayed stable at `16`;
- `32` remains an explicit future throughput experiment, not the registered
  default, because it may saturate render memory or disk bandwidth.

The reproducible runner is:

```text
scripts/run_jepa_phase2d_min_sources_parallel_render.sh
```

## Render Accounting Gate

Final render-readiness artifact:

```text
.generated/jepa_counterfactual/phase2d_stage8_render_readiness_after_full_render.json
```

Hash:

```text
2bd8e172feb63a6541649254da5da01f4cc000b751aa7d908397f8c53073b5d4
```

Result:

- `ready_for_spatial_future_join: true`;
- all required split plan roots are present;
- all required render roots are present;
- all split renders are complete and accounted;
- all per-scene metadata files are present;
- scene metadata frame sums match split-root frame counts.

Frame accounting:

| Split | Scenes | Rendered Frames | Invalid Frames | Metadata Relocations |
| --- | ---: | ---: | ---: | ---: |
| train | 32 | 82,944 | 25,230 | 22 |
| validation | 16 | 41,472 | 14,382 | 0 |
| test-ID | 16 | 41,472 | 11,604 | 0 |
| test-hard | 16 | 41,472 | 14,366 | 0 |

Invalid frames are not a render-readiness failure. Phase 2D is an all-candidate
counterfactual dataset, so unsafe or degenerate futures are valid outcomes as
long as they are accounted for and masked from token prediction loss.

Render storage:

- render root: `/tmp/lewm_phase2d_min_sources_render_20260614`;
- render size: `247G`;
- `/tmp` free space after render: about `370G`;
- workspace free space after generated metadata: about `68G`.

## Stale-Path Correction

Moving the partial train render exposed stale absolute paths inside generated
render metadata. The dataset joiner and render-readiness gate now resolve moved
render roots by local scene layout:

- `frames_rendered_jsonl` can be resolved beside a scene `summary.json`;
- stale `rgb_path` entries can be resolved to the local scene `rgb/` directory;
- render-readiness checks per-scene metadata presence and frame counts, not only
  split-root summaries.

Code:

- `scripts/build_jepa_spatial_future_dataset.py`;
- `lewm/benchmarks/phase2d_render_readiness.py`;
- `lewm/tests/test_spatial_future_dataset.py`;
- `lewm/tests/test_phase2d_render_readiness.py`.

## Spatial-Future Datasets

All four split datasets were rebuilt after the stale-path correction.

| Split | Source Rows | Candidate Rows | Complete Valid | Incomplete Or Invalid | Unplanned Skipped | Rendered Frames Indexed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 512 | 41,472 | 27,237 | 14,235 | 0 | 82,944 |
| validation | 256 | 20,736 | 12,752 | 7,984 | 0 | 41,472 |
| test-ID | 256 | 20,736 | 14,163 | 6,573 | 0 | 41,472 |
| test-hard | 256 | 20,736 | 12,686 | 8,050 | 0 | 41,472 |

Dataset hashes:

| Split | Dataset Hash | Summary Hash |
| --- | --- | --- |
| train | `c75bd0830fd632ce9ac77c85856db15456be76029768085528eb37295fba532c` | `57de1f7b82ca99c6e5db3f886a492916bf16406c1e13893c2e26f162a4c0a91a` |
| validation | `e33eeb095522b7ec1369497999fb50658c0e5f00f106f33c1de63bbbf4bbbd4a` | `4e6c4d0d92b758ac9ed78de0fd4aa1786607ae7318a736b8cd705e39f8cb6ebd` |
| test-ID | `e442adcdf062db214c689b82c2dd98ff947fd868cb2a52c5ba45e78ed7d7a70a` | `1641305c1854ea70df697c338ef86dd3f29281343fe06be4ca3329189ff7b072` |
| test-hard | `e06dfb0fac6790f8427ac99d7312387528d6877b54caa19296b5ec6ca0292136` | `0939ef51072347afb84fc32082a624f1a5078c84aac9045da5194ec0bcfa2647` |

## Split Manifest And Training-Start Gate

Split manifest:

```text
.generated/jepa_counterfactual/phase2d_min_sources/phase2d_split_manifest.json
```

Hash:

```text
c20e82e5917cf898d8c63f291f219b3c95623284668bfcbda04aa84b1498cda7
```

Gate result:

- confirmatory data gate passes;
- all required splits are present;
- all split scene/source pairs are disjoint;
- all required dataset files verify by hash;
- topology and visual lineage verify;
- every selected source state has the full `81` two-block sequence grid;
- eligible non-hold hard-negative coverage is `1.0` in every split;
- eligible first-action minimum share is above `5%` in every split.

C2 training-start preflight:

```text
.generated/jepa_counterfactual/phase2d_min_sources/c2_confirmatory_training_start_preflight.json
```

Hash:

```text
9b57587f2c79d494d541feaa44d836bc2773acb2baf75278700ae2f0ff1927c9
```

Result: passed.

Access scope remains:

```text
training_start_only_no_validation_or_test_result_access_granted
```

## Trainer Smoke

One bounded C2 training-start smoke was run:

- cell: `C2`;
- run class: `confirmatory`;
- optimization steps: `1`;
- capped train rows: `162`;
- capped validation rows: `162`;
- device: `cpu`.

Artifacts:

```text
.generated/jepa_counterfactual/phase2d_min_sources/c2_confirmatory_training_start_smoke.pt
.generated/jepa_counterfactual/phase2d_min_sources/c2_confirmatory_training_start_smoke.json
```

Hashes:

```text
1688da61ab19d7252bfa8562b0bdf8538861128e173d1b4f5bc20e490b0add43
263471c81c22622a47ff7add7e78b8153fa869e7c83b436d779c39cb1c1c3b61
```

Smoke contract:

- batch rows: `162`;
- horizon: `2`;
- command dimension: `15`;
- valid transitions: `64`;
- non-hold valid transitions: `51`;
- eligible wrong pairs: `408`;
- all materialized frame tensors finite: `true`.

The validation diagnostic from this smoke is not a confirmatory result and must
not be used for H1/H2 decisions.

## Full Primary Training Manifests

Frozen manifest generator:

```text
scripts/create_jepa_phase2d_full_run_manifests.py
```

Generated matrix:

```text
.generated/jepa_counterfactual/phase2d_min_sources/full_run_manifests/summary.json
```

Registered primary cells:

- `C0`;
- `C1`;
- `C2`.

Registered optimization seeds:

- `20260614`;
- `20260615`;
- `20260616`.

Frozen schedule:

| Field | Value |
| --- | ---: |
| train source states | 512 |
| source states per batch | 2 |
| steps per epoch | 256 |
| epochs | 3 |
| optimization steps | 768 |
| evaluation interval | 256 |

Rationale:

- one source state contains exactly `81` candidate rows;
- a batch of `2` source states contains `162` candidate rows, matching the
  passed C2 smoke contract;
- `256` steps is one full pass over the `512` train source states;
- validation is evaluated at epoch boundaries;
- the same schedule is frozen for C0, C1, and C2.

Each manifest records:

- exact train and validation dataset hashes;
- split-manifest hash;
- cell;
- seed;
- optimizer constants;
- checkpoint rule;
- exact launch command;
- train/validation-only access scope.

Example C2 seed command:

```bash
.generated/venvs/genesis_render_vulkan/bin/python scripts/train_jepa_phase2d.py \
  --train-data .generated/jepa_counterfactual/phase2d_min_sources/train_spatial_future_v1.jsonl \
  --validation-data .generated/jepa_counterfactual/phase2d_min_sources/validation_spatial_future_v1.jsonl \
  --output .generated/jepa_counterfactual/phase2d_min_sources/full_run_checkpoints/C2_seed_20260614.pt \
  --cell C2 \
  --run-class confirmatory \
  --split-manifest .generated/jepa_counterfactual/phase2d_min_sources/phase2d_split_manifest.json \
  --optimization-steps 768 \
  --evaluation-interval 256 \
  --source-states-per-batch 2 \
  --seed 20260614 \
  --device auto
```

Environment note: the current venv reports PyTorch `2.12.0+cu130` with
`cuda_available: false`, so `--device auto` resolves to CPU in this environment
unless a different torch/driver environment is activated.

## Remaining Guardrail

Starting the C0/C1/C2 full-training matrix is now admissible. Held-out result
access is still not admissible.

Before any test-ID access:

1. complete all registered C0/C1/C2 seed runs;
2. apply the validation-only checkpoint rule;
3. create selected-checkpoint manifests containing verified
   `selected_checkpoint` artifacts;
4. run the Stage 5 `test_id` readiness gate.

Before any test-hard access:

1. produce and freeze the test-ID report manifest;
2. run the Stage 5 `test_hard` readiness gate.

Diagnostic state-only and action-only controls remain required for the complete
research report, but they do not unlock C0/C1/C2 checkpoint selection or
held-out access.

## Quality Gates

Focused checks after the run-manifest implementation:

```bash
python3 -m py_compile \
  lewm/benchmarks/phase2d_run_manifest.py \
  scripts/create_jepa_phase2d_full_run_manifests.py

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
.generated/venvs/genesis_render_vulkan/bin/python -m pytest \
  lewm/tests/test_phase2d_run_manifest.py \
  lewm/tests/test_phase2d_readiness.py \
  -q
```

Result:

```text
7 passed
```

Prior focused checks for the render/data corrections:

```text
13 passed
```

Local repository suite:

```text
152 passed, 6 warnings, 3 subtests passed
```

No unrestricted root-level `pytest` result is claimed because this workspace
contains non-importable installed package trees outside the local `lewm/tests`
contract.
