# Phase 2D Stage 10: Full Training Launch

Date: 2026-06-14

## Status

The registered Phase 2D primary full-training matrix reached launch, exposed a
validation-diagnostic OOM in the first C0 wave, and was stopped before any
checkpoint was produced. The diagnostic was fixed and smoke-tested. A subsequent
CPU-only relaunch was identified as the wrong runtime for this workstation and
was stopped before first validation.

The corrected ROCm GPU relaunch completed all C0 and C1 seeds. All completed
C0/C1 runs failed the registered one-step persistence gate, and final C1
stability diagnostics also failed. The original C2 cell failed before producing
usable seed results: one concurrent C2 job hit GPU OOM, and the remaining C2
jobs entered non-finite training dynamics around the first epoch.

On 2026-06-15, a bounded C2 stabilization pilot with gradient clipping and
detached action-control branches stayed numerically finite for 128 steps, but
failed the scientific checkpoint gate by collapse, low effective rank, strongly
negative hard-negative action advantage, and a one-step persistence ratio far
above `1.0`. It is therefore a failed stabilization pilot, not a replacement
confirmatory C2 result.

No test-ID or test-hard result has been opened. All evidence in this document
uses train and validation data under the Stage 9 access scope.

## Pre-Launch Runtime Fix

Before launch, I added a batch-local image tensor cache to:

```text
lewm/benchmarks/phase2d_training.py
```

Reason:

- each source state expands to `81` candidate rows;
- those rows repeat the same current observation path;
- invalid future slots substitute the most recent valid observation;
- the previous materializer decoded and resized duplicated paths repeatedly.

The cache is local to one materialized batch. It changes no row order, masks,
actions, hard negatives, model inputs, losses, or metrics. It only avoids
duplicated PNG decode/resize work inside one batch.

Guard test:

```text
lewm/tests/test_phase2d_training.py::test_materialized_phase2d_batch_caches_duplicate_image_paths
```

## Manifest Hardening

The full-run manifests were regenerated after the runtime fix. Each manifest
now hashes the core code inputs in addition to the split/data inputs:

- `scripts/train_jepa_phase2d.py`;
- `lewm/benchmarks/phase2_data.py`;
- `lewm/benchmarks/phase2d_training.py`;
- `lewm/benchmarks/rollout_diagnostics.py`;
- `lewm/models/phase2d_spatial_lewm.py`;
- `lewm/models/spatial_predictor.py`.

This prevents a stale frozen manifest from silently pointing at changed trainer
code.

Full-run matrix summary:

```text
.generated/jepa_counterfactual/phase2d_min_sources/full_run_manifests/summary.json
```

Hash:

```text
03629b6eaca5392979c7ba9e4408e30bd1a0df20e0ac8f079ae1af246e3b6833
```

Example regenerated run manifest hash:

```text
7be2f5cfb073e32973f638f67651c3e05565f4e15e773317ef1e4e955047691a  .generated/jepa_counterfactual/phase2d_min_sources/full_run_manifests/C0_seed_20260614_manifest.json
```

## Matrix Runner

New runner:

```text
scripts/run_jepa_phase2d_full_training_matrix.py
```

Responsibilities:

- verify every run manifest before launch;
- skip already existing checkpoints unless `--overwrite` is passed;
- run a bounded number of independent manifests concurrently;
- stop launching queued manifests after the first failure unless
  `--continue-on-failure` is explicitly passed;
- set fixed CPU thread environment per process;
- write one log per run;
- write an incremental machine-readable launch report.

Dry-run command:

```bash
.generated/venvs/genesis_render_vulkan/bin/python \
  scripts/run_jepa_phase2d_full_training_matrix.py \
  --summary .generated/jepa_counterfactual/phase2d_min_sources/full_run_manifests/summary.json \
  --log-dir .generated/jepa_counterfactual/phase2d_min_sources/full_run_logs_v2_dry_run \
  --report .generated/jepa_counterfactual/phase2d_min_sources/full_run_matrix_dry_run_report_v2.json \
  --jobs 3 \
  --threads-per-job 10 \
  --dry-run
```

Dry-run result:

- all nine manifests verified;
- all generated commands parsed;
- default fail-fast policy recorded as `continue_on_failure: false`;
- runner report passed.

Dry-run report hash:

```text
1f075453ca6b8bbc36c6b1dc7d47bd25ecf9c2b040b634652468cabb0c56803e
```

Dry-run report:

```text
.generated/jepa_counterfactual/phase2d_min_sources/full_run_matrix_dry_run_report_v2.json
```

## Attempt 1 Failure

Initial launch session:

```text
phase2d_full_training
```

Initial active jobs:

- `C0_seed_20260614`;
- `C0_seed_20260615`;
- `C0_seed_20260616`.

All three C0 jobs failed at the first validation boundary with return code `1`.
No checkpoint was written.

Root cause:

```text
lewm/benchmarks/rollout_diagnostics.py
```

The old `summarize_spatial_stability` implementation computed:

```python
(states[:, None] - states[None, :]).square().mean(dim=2)
```

For the Phase 2D validation population, this attempted to allocate:

```text
190210142896128 bytes
```

This was an O(N^2 * D) diagnostic broadcast. It does not invalidate the train
metrics already emitted before validation, but it prevents those runs from
being counted as completed training runs.

The stopped attempt is retained as evidence:

```text
.generated/jepa_counterfactual/phase2d_min_sources/full_run_logs/
.generated/jepa_counterfactual/phase2d_min_sources/full_run_matrix_report.json
```

## OOM Fix

The stability diagnostic now:

- keeps the full token-level feature standard deviation, norm, and covariance
  effective-rank diagnostics;
- reshapes only for state-level pairwise discrimination;
- samples at most `1024` state rows deterministically;
- computes pairwise squared distance as
  `||x||^2 + ||y||^2 - 2 x y^T`, avoiding the broadcasted `(N, N, D)` tensor;
- records the population size, sample size, and sampling policy in every
  stability report.

New reported fields:

```text
pairwise_state_population
pairwise_state_sample_size
pairwise_state_sampling
```

Guard tests:

```text
lewm/tests/test_rollout_diagnostics.py::test_spatial_stability_bounds_pairwise_state_sample
lewm/tests/test_rollout_diagnostics.py::test_spatial_stability_rejects_invalid_pairwise_sample_size
```

Full-validation smoke after the fix:

```text
.generated/jepa_counterfactual/phase2d_min_sources/phase2d_oomfix_full_validation_smoke.json
.generated/jepa_counterfactual/phase2d_min_sources/phase2d_oomfix_full_validation_smoke.pt
```

Smoke hashes:

```text
b6be89ed9ae94da9ddd3e6edd34920a46c0eb1198e5e367c6e3f2437ea436704  .generated/jepa_counterfactual/phase2d_min_sources/phase2d_oomfix_full_validation_smoke.json
247c5c72319e7631ece2e68c66a8e74105e941f5763385039a3b6c4930cc3bce  .generated/jepa_counterfactual/phase2d_min_sources/phase2d_oomfix_full_validation_smoke.pt
```

Smoke stability summary:

```text
pairwise_state_population: 62208
pairwise_state_sample_size: 1024
pairwise_state_sampling: deterministic_stride
mean_pairwise_state_mse: 0.0026608523912727833
collapse_warning: true
effective_rank_warning: true
```

The collapse/rank warnings are expected for a one-step untrained C0 smoke and
are not confirmatory model evidence. The gate being tested here was whether the
full validation diagnostic executes without the previous allocation failure.

## Superseded CPU Relaunch

After the OOM fix, a v2 matrix was launched from:

```text
.generated/jepa_counterfactual/phase2d_min_sources/full_run_manifests/
```

with:

```text
.generated/venvs/genesis_render_vulkan/bin/python
```

That environment is the rendering/Vulkan environment and reports no usable
PyTorch GPU device on this host. The v2 launch therefore ran on CPU. This was a
runtime-selection error, not a research result. The v2 session was stopped
before first validation and before any completed checkpoint was produced.

Retained superseded evidence:

```text
.generated/jepa_counterfactual/phase2d_min_sources/full_run_logs_v2/
.generated/jepa_counterfactual/phase2d_min_sources/full_run_matrix_report_v2.json
```

## GPU Runtime Correction

The correct local training runtime is:

```text
/home/andrewknowles/TinyQuadJEPA/bin/python
```

Host ROCm check, outside the sandboxed device namespace:

```text
torch: 2.10.0.dev20250926+rocm6.3
hip: 6.3.42131-fa1d09cbd
cuda_available: True
device_count: 2
device_name[0]: AMD Radeon AI PRO R9700
```

Required GPU environment:

```bash
export ROCM_PATH=/opt/rocm-7.1.1
export PATH=/opt/rocm-7.1.1/lib/llvm/bin:/opt/rocm-7.1.1/bin:$PATH
export HIP_VISIBLE_DEVICES=0
unset HSA_OVERRIDE_GFX_VERSION
```

The `genesis_render_vulkan` environment remains valid for rendering tasks, but
it is not the Phase 2D training runtime.

## GPU Manifest Set

The registered GPU manifests were regenerated with `--device cuda` and GPU
checkpoint paths:

```text
.generated/jepa_counterfactual/phase2d_min_sources/full_run_manifests_gpu/
.generated/jepa_counterfactual/phase2d_min_sources/full_run_checkpoints_gpu/
```

Summary hash:

```text
a3a6f748cf2f56d5723c7406c16ab25bf0ceff899fb853c9626f4e5f6b268769  .generated/jepa_counterfactual/phase2d_min_sources/full_run_manifests_gpu/summary.json
```

Example GPU run manifest hash:

```text
c245c532a95825b86149f4be32688b691097529736bb2f784ffbac6e3abf5865  .generated/jepa_counterfactual/phase2d_min_sources/full_run_manifests_gpu/C0_seed_20260614_manifest.json
```

Every GPU run command uses:

```text
/home/andrewknowles/TinyQuadJEPA/bin/python scripts/train_jepa_phase2d.py ... --device cuda
```

## GPU Prelaunch Gates

GPU smoke command scope:

- one optimization step;
- full validation path enabled;
- C0 cell;
- seed `20260614`;
- device `cuda`;
- no test-ID or test-hard access.

GPU smoke hashes:

```text
179e522a8c8c735851a39de3bd947b5757fe759d16d1351eb4045e0011421bae  .generated/jepa_counterfactual/phase2d_min_sources/phase2d_gpu_cuda_smoke.json
40dd1cbf91799fa833b21a43adf92d234278e889dee224b73fce7858af8664dd  .generated/jepa_counterfactual/phase2d_min_sources/phase2d_gpu_cuda_smoke.pt
```

GPU smoke stability summary:

```text
pairwise_state_population: 600
pairwise_state_sample_size: 600
pairwise_state_sampling: full_population
mean_pairwise_state_mse: 0.0005273325950838625
collapse_warning: true
effective_rank_warning: true
```

The smoke warnings are expected for a one-step untrained run. The gate was CUDA
execution and full validation-path execution, not model quality.

GPU dry-run report:

```text
.generated/jepa_counterfactual/phase2d_min_sources/full_run_matrix_dry_run_report_gpu.json
```

GPU dry-run hash:

```text
e8fcd8505f7926d4460818bde65aa2621bb1f625302c5b1b15aab960143faa89
```

Dry-run result:

- all nine GPU manifests verified;
- all generated commands parsed;
- default fail-fast policy recorded as `continue_on_failure: false`;
- no checkpoint overwrite.

## GPU Launch Policy

Launch policy:

- parallel jobs: `3`;
- CPU threads per job: `4`;
- total requested CPU worker threads: `12`;
- host hardware threads: `32`;
- checkpoint overwrite: disabled;
- device argument: `cuda`;
- visible GPU: `HIP_VISIBLE_DEVICES=0`;
- fail-fast: stop launching queued manifests after first failure.

Rationale:

- the R9700 has enough VRAM for three concurrent Phase 2D jobs;
- the GPU should be saturated by training while CPU threads feed data and
  logging;
- limiting CPU threads per job leaves headroom for IO, tmux, monitoring, and
  the OS;
- three jobs maps to the three optimization seeds per C0/C1/C2 wave.

Launch command:

```bash
export ROCM_PATH=/opt/rocm-7.1.1
export PATH=/opt/rocm-7.1.1/lib/llvm/bin:/opt/rocm-7.1.1/bin:$PATH
export HIP_VISIBLE_DEVICES=0
unset HSA_OVERRIDE_GFX_VERSION
tmux new-session -d \
  -s phase2d_full_training_gpu \
  -c /home/andrewknowles/Workspace/LeWMQuad-v3 \
  "/home/andrewknowles/TinyQuadJEPA/bin/python scripts/run_jepa_phase2d_full_training_matrix.py --summary .generated/jepa_counterfactual/phase2d_min_sources/full_run_manifests_gpu/summary.json --log-dir .generated/jepa_counterfactual/phase2d_min_sources/full_run_logs_gpu --report .generated/jepa_counterfactual/phase2d_min_sources/full_run_matrix_report_gpu.json --jobs 3 --threads-per-job 4"
```

Live session:

```text
phase2d_full_training_gpu
```

Live report:

```text
.generated/jepa_counterfactual/phase2d_min_sources/full_run_matrix_report_gpu.json
```

Run logs:

```text
.generated/jepa_counterfactual/phase2d_min_sources/full_run_logs_gpu/
```

Early GPU resource evidence:

- GPU 0 utilization reached `100%`;
- GPU 0 VRAM used about `12.8 GiB` of about `34.2 GiB`;
- GPU 1 remained unused;
- active C0 jobs were `C0_seed_20260614`, `C0_seed_20260615`, and
  `C0_seed_20260616`.

## GPU C0 Completion Check

At `2026-06-14 23:30 BST`, the C0 GPU wave had completed and C1 had launched.
The runner recorded all three C0 jobs as `completed` with return code `0` and
manifest verification passing.

Runtime:

| Run | Elapsed seconds |
| --- | ---: |
| `C0_seed_20260614` | `925.911` |
| `C0_seed_20260615` | `926.119` |
| `C0_seed_20260616` | `929.440` |

Final validation interface diagnostic summary:

| Run | Hard-negative advantage | Persistence ratio | Stability pass |
| --- | ---: | ---: | --- |
| `C0_seed_20260614` | `-0.3799` | `17.3162` | false |
| `C0_seed_20260615` | `10.7137` | `20.0288` | true |
| `C0_seed_20260616` | `3.3573` | `5.8561` | false |

These values are validation-interface diagnostics for the C0 control wave. They
are not held-out test results and are not sufficient for model promotion. The
persistence ratios remain far above the registered `< 1.0` success criterion in
this wave.

C0 artifact hashes:

```text
22cc938759c411f14bcc724e00808563019c5705aab0bad0e7018ebd4b0af3bf  .generated/jepa_counterfactual/phase2d_min_sources/full_run_checkpoints_gpu/C0_seed_20260614.pt
8c6f01caeacf2d49db33b19d62abb555d26f1871d7cb8f723bbc91382cf039f4  .generated/jepa_counterfactual/phase2d_min_sources/full_run_checkpoints_gpu/C0_seed_20260614.json
0d7eea45fb5e01280ea7777042c21b3b68754c3707ad71120dcc2d2daace3a01  .generated/jepa_counterfactual/phase2d_min_sources/full_run_checkpoints_gpu/C0_seed_20260615.pt
d4398051a6181d616f57f8821f81bdc318b9f027ca97768479827333e90a8fec  .generated/jepa_counterfactual/phase2d_min_sources/full_run_checkpoints_gpu/C0_seed_20260615.json
4691e03d7acd0f257b7274fc6c144e9fa1dafdad700db8dccbc53505d45aa8ac  .generated/jepa_counterfactual/phase2d_min_sources/full_run_checkpoints_gpu/C0_seed_20260616.pt
28feface0bbe1bbfa98d1dd30bcdad38e39ed3d6aa9de7327f4130338f715131  .generated/jepa_counterfactual/phase2d_min_sources/full_run_checkpoints_gpu/C0_seed_20260616.json
```

## GPU C1 Completion Check

At `2026-06-14 23:43 BST`, the C1 GPU wave had completed and C2 had launched.
The runner recorded all three C1 jobs as `completed` with return code `0`.

Runtime:

| Run | Elapsed seconds |
| --- | ---: |
| `C1_seed_20260614` | `791.313` |
| `C1_seed_20260615` | `774.728` |
| `C1_seed_20260616` | `783.907` |

Final validation interface diagnostic summary:

| Run | Hard-negative advantage | Persistence ratio | Stability pass |
| --- | ---: | ---: | --- |
| `C1_seed_20260614` | `3.5859` | `3.9770` | false |
| `C1_seed_20260615` | `-3.6378` | `9.9963` | false |
| `C1_seed_20260616` | `4.2759` | `3.3150` | false |

C1 improves the persistence ratio relative to most C0 runs but still fails the
registered `< 1.0` persistence criterion and all final C1 stability checks fail.
This remains validation-interface evidence only.

C1 artifact hashes:

```text
ae45ac2054b25a3060c837c12149ca0e65e9b98f1847080c6a29b10a844da3e8  .generated/jepa_counterfactual/phase2d_min_sources/full_run_checkpoints_gpu/C1_seed_20260614.pt
eb7564ffa4371a54fd5359e9a7a1ab36b320ebd6974740df549247d625f3e7d6  .generated/jepa_counterfactual/phase2d_min_sources/full_run_checkpoints_gpu/C1_seed_20260614.json
fac511200e37b43623af8adbad43b834f2856218f7aa76434864552a05b1898d  .generated/jepa_counterfactual/phase2d_min_sources/full_run_checkpoints_gpu/C1_seed_20260615.pt
b761c7c243a2a23d7d808544915a0f00f54213045d21ff844c06b470f74b364a  .generated/jepa_counterfactual/phase2d_min_sources/full_run_checkpoints_gpu/C1_seed_20260615.json
734f3e265e16eafcec6ee1360e62b71e89190369e1379534e8a36b07b6672211  .generated/jepa_counterfactual/phase2d_min_sources/full_run_checkpoints_gpu/C1_seed_20260616.pt
4e8e04e97b12c13d2be28a191808707d01b4a2201f1afbca15dc965777ed0fd1  .generated/jepa_counterfactual/phase2d_min_sources/full_run_checkpoints_gpu/C1_seed_20260616.json
```

## C2 Failure

C2 adds hard-negative and zero-action contrastive losses. Three concurrent C2
jobs exceeded available R9700 VRAM. `C2_seed_20260616` failed before completing
an optimization step:

```text
status: failed
elapsed_seconds: 14.050
return_code: 1
exception: torch.OutOfMemoryError
failed allocation: 244 MiB
reported free GPU memory: 0 bytes
```

This first failure was a resource-scheduling failure, not a model result. The
two already-running C2 jobs then continued far enough to expose a separate
model/objective failure: training metrics became non-finite around the first
epoch boundary, and validation later failed in the stability diagnostic because
the feature covariance was non-finite or ill-conditioned.

Observed original C2 outcome:

- `C2_seed_20260616`: GPU OOM before training;
- `C2_seed_20260614`: non-finite training dynamics before a usable final
  validation result;
- `C2_seed_20260615`: non-finite training dynamics before a usable final
  validation result.

Decision:

- the original registered C2 cell has failed;
- no original C2 checkpoint may be selected or treated as confirmatory;
- any further C2-like run must be documented as an amended stabilization pilot,
  not as a continuation of the preregistered C2 result.

## C2 Stabilization Pilot

On 2026-06-15, I implemented and smoke-tested two engineering safeguards before
considering any C2 recovery launch:

- fail-fast checks for non-finite train and validation metrics;
- robust stability diagnostics that report non-finite or ill-conditioned
  features without raising a linear-algebra exception;
- optional gradient clipping with recorded pre-clip gradient norm;
- optional detachment of the current spatial state for wrong-action and
  zero-action contrast branches.

Gradient clipping alone did not stabilize the original objective:

- `--max-grad-norm 1.0` failed with infinite gradient norm;
- `--lr 1e-4 --max-grad-norm 1.0` failed with non-finite gradient norm.

The detached-control pilot command:

```bash
/home/andrewknowles/TinyQuadJEPA/bin/python \
  scripts/train_jepa_phase2d.py \
  --train-data .generated/jepa_counterfactual/phase2d_min_sources/train_spatial_future_v1.jsonl \
  --validation-data .generated/jepa_counterfactual/phase2d_min_sources/validation_spatial_future_v1.jsonl \
  --output .generated/jepa_counterfactual/phase2d_min_sources/phase2d_c2_detach_clip1_smoke.pt \
  --cell C2 \
  --run-class smoke \
  --optimization-steps 128 \
  --evaluation-interval 64 \
  --source-states-per-batch 2 \
  --max-validation-rows 400 \
  --seed 20260614 \
  --device cuda \
  --max-grad-norm 1.0 \
  --detach-action-control-state
```

Detached-control pilot artifacts:

```text
.generated/jepa_counterfactual/phase2d_min_sources/phase2d_c2_detach_clip1_smoke.json
.generated/jepa_counterfactual/phase2d_min_sources/phase2d_c2_detach_clip1_smoke.pt
```

Artifact hashes:

```text
57ffad62862c89bcb0c58462211c6353dc8edde0b55223cfb15cc0f8984e4073  .generated/jepa_counterfactual/phase2d_min_sources/phase2d_c2_detach_clip1_smoke.json
d72cd891229688e9e78037fb7c5a1b8dd9e0f3b428ca4cdecf22c7bc965022ff  .generated/jepa_counterfactual/phase2d_min_sources/phase2d_c2_detach_clip1_smoke.pt
```

Final detached-control validation gate:

```text
stability_pass: false
collapse_warning: true
effective_rank_warning: true
mean_feature_std: 0.019723
effective_rank: 1.759245
effective_rank_fraction: 0.036651
hard_negative_action_advantage: -80.072174
one_step_rollout_persistence_ratio: 247.687668
```

Interpretation:

- the amended run is numerically finite, but it is scientifically invalid;
- the representation collapsed under the registered stability diagnostic;
- persistence is still overwhelmingly better than the learned prediction;
- the real action is worse than hard-negative actions on the primary
  normalized action-identifiability estimand;
- no full detached-control C2 run is justified.

## Checkpoint Gate Amendment

The trainer now separates cell participation from actual checkpoint
selectability. A validation checkpoint is selectable only when all registered
conditions pass:

- stability diagnostics pass;
- real action beats hard-negative actions by at least `0.10` of target change;
- real action beats zero action by at least `0.10` of target change;
- one-step real prediction beats persistence with ratio `< 1.0`.

The trainer records this as:

```text
final_validation_gate
checkpoint_selection_permitted
```

The reusable checker is:

```bash
.generated/venvs/genesis_render_vulkan/bin/python \
  scripts/check_jepa_phase2d_smoke_gate.py \
  --report .generated/jepa_counterfactual/phase2d_min_sources/phase2d_c2_detach_clip1_smoke.json \
  --output .generated/jepa_counterfactual/phase2d_min_sources/phase2d_c2_detach_clip1_smoke_gate.json
```

Result:

```text
passed: false
failure_reasons:
- stability_failed
- hard_negative_action_advantage_below_threshold
- missing_zero_action_advantage
- persistence_ratio_not_below_threshold
```

Gate report hash:

```text
32b74b2fb367ff13f0b4470c396220d4eadb31da9d67b554738bc126463a163c  .generated/jepa_counterfactual/phase2d_min_sources/phase2d_c2_detach_clip1_smoke_gate.json
```

The matrix runner now requires `--required-smoke-gate-report` for any non-dry
launch. A dry-run against the old GPU manifests now fails manifest verification
because the trainer/model/diagnostic code hashes changed after the original
GPU launch:

```text
.generated/jepa_counterfactual/phase2d_min_sources/full_run_matrix_gate_dry_run_report_gpu.json
```

Dry-run report hash:

```text
bba44fca64afbc8337569e5f4a71cbe55e8332b8ef71a6bb7def28d51f341e7f  .generated/jepa_counterfactual/phase2d_min_sources/full_run_matrix_gate_dry_run_report_gpu.json
```

This prevents numerically finite but collapsed runs from being promoted and
prevents stale pre-amendment manifests from being relaunched accidentally.

## Quality Gates

Focused checks:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
.generated/venvs/genesis_render_vulkan/bin/python -m pytest \
  lewm/tests/test_rollout_diagnostics.py \
  lewm/tests/test_phase2d_run_manifest.py \
  lewm/tests/test_phase2d_training.py \
  lewm/tests/test_phase2d_spatial_lewm.py \
  -q
```

Result:

```text
31 passed
```

Full local suite:

```text
154 passed, 6 warnings, 3 subtests passed
```

Diff hygiene:

```text
git diff --check passed
```

## Next Gates

The original matrix does not proceed to selected-checkpoint manifests because
C2 has no valid checkpoint candidate. The next gates are:

1. update the preregistered plan with the original C2 failure and the failed
   detached-control pilot;
2. do not open `test_id` or `test_hard`;
3. implement the next architecture-level fix rather than launching another C2
   full run;
4. require a bounded smoke to pass the new explicit validation gate before any
   full training launch;
5. keep all amended runs labelled as pilots until a new preregistration is
   frozen.

## Phase 2E Follow-Up

On 2026-06-15, the first architecture-level follow-up tested learned slot
target/state geometry as a bounded Phase 2E pilot:

```text
docs/lewm_jepa_phase2e_target_geometry_plan_2026-06-15.md
```

The run completed on the ROCm GPU runtime and avoided the original C2
non-finite-gradient failure, but it failed the registered smoke gate:

```text
checkpoint_selection_permitted: false
stability_pass: false
hard_negative_action_advantage: -9.082974
zero_action_advantage: 0.050411
one_step_rollout_persistence_ratio: 24.960993
```

Gate report:

```text
passed: false
failure_reasons:
- stability_failed
- hard_negative_action_advantage_below_threshold
- zero_action_advantage_below_threshold
- persistence_ratio_not_below_threshold
```

Artifact hashes:

```text
8db813c81d57f9058b9816ab76683b34a23385f731394d4b5fe11aadece472ae  .generated/jepa_counterfactual/phase2d_min_sources/phase2e_slot_c2_smoke.json
0ba04c3730c3cccaca6af32df3ee18fb258d15b5f602f0533078a8163ba7d9dc  .generated/jepa_counterfactual/phase2d_min_sources/phase2e_slot_c2_smoke.pt
12e2103337e083a19e313af585e52a05300e4240a7ea263298a1bd8a9e69d759  .generated/jepa_counterfactual/phase2d_min_sources/phase2e_slot_c2_smoke_gate.json
```

Decision: no full Phase 2E slot-geometry matrix is justified. The next candidate
must change the represented target family, not only the token pooling geometry.
