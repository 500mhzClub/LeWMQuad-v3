# JEPA Phase 2D Training-Start Gate

Date: 2026-06-14

Branch: `jepa-spatial-world-model-nav`

Registration:
`docs/lewm_jepa_phase2d_preregistered_research_plan_2026-06-14.md`

Previous increment:
`docs/lewm_jepa_phase2d_stage6_generation_contract_2026-06-14.md`

## Scope

This increment adds the final pre-training quality gate needed before a full
Phase 2D training run can be launched.

It does not start confirmatory training. It makes confirmatory training
unlaunchable unless the split manifest, lineage, data scale, registered cell,
and train/validation path checks pass first.

## Problem

Stage 5 guarded validation/test access after C0/C1/C2 selected-checkpoint
manifests exist. That is necessary, but it is not sufficient as a training
start gate. A full training launcher also needs to reject:

- direct JSONL paths that do not match the frozen split manifest;
- diagnostic cells used as if they were primary confirmatory cells;
- split manifests whose file hashes verify but whose confirmatory gate fails;
- missing validation split data;
- accidental validation/test-result access implied by a training command.

## Decisions

### Training Start Is Separate From Held-Out Access

`phase2d_training_start_readiness` verifies permission to start training only.
It does not grant validation, test-ID, or test-hard result access. Held-out
access remains controlled by `phase2d_run_readiness`.

### Confirmatory Training Must Use A Frozen Split Manifest

Confirmatory training requires:

- a split manifest whose files hash correctly;
- verified topology/visual lineage;
- a passing confirmatory data gate;
- present train and validation splits;
- explicit train/validation paths that match the manifest;
- a registered primary cell: `C0`, `C1`, or `C2`.

Diagnostic cells remain available for smoke/pilot work, but cannot be launched
as confirmatory cells.

### The Trainer Enforces The Gate

`scripts/train_jepa_phase2d.py --run-class confirmatory` now requires
`--split-manifest`. If the preflight fails, the script exits before loading
data, constructing the model, or running optimization.

## Implemented

Updated module:

- `lewm/benchmarks/phase2d_readiness.py`.

New script:

- `scripts/check_jepa_phase2d_training_start.py`.

Updated trainer:

- `scripts/train_jepa_phase2d.py`.

Updated tests:

- `lewm/tests/test_phase2d_readiness.py`.

## End-To-End Strict-Lineage Smoke

The Stage 6 one-source strict-lineage benchmark was rendered and joined into
spatial-future rows.

Render command:

```bash
EGL_DEVICE_ID=0 PYOPENGL_PLATFORM=egl \
GENESIS_ROCM_PYTHON=$PWD/.generated/venvs/genesis_render_vulkan/bin/python \
.generated/venvs/genesis_render_vulkan/bin/python \
  scripts/render_jepa_counterfactual_plan_root.py \
  --plan-root .generated/jepa_counterfactual/phase2d_stage6_strict_lineage_plans \
  --output-root .generated/jepa_counterfactual/phase2d_stage6_strict_lineage_render \
  --scene-corpus .generated/scene_corpus/minimum_tex_20260520T211541Z \
  --backend vulkan \
  --camera-mode replay \
  --replay-env-mode single \
  --rgb-format png \
  --store-resolution training \
  --overwrite
```

Result: exit status `0`; `162` frames rendered; `0` invalid frames.

The same command with `EGL_DEVICE_ID=1` failed because only one EGL device was
visible in this session. The successful command therefore records
`EGL_DEVICE_ID=0` as the current local smoke setting, not as a scientific model
condition.

Spatial join command:

```bash
.generated/venvs/genesis_render_vulkan/bin/python \
  scripts/build_jepa_spatial_future_dataset.py \
  --benchmark .generated/jepa_counterfactual/phase2d_stage6_strict_lineage_generator_smoke.jsonl \
  --plan-root .generated/jepa_counterfactual/phase2d_stage6_strict_lineage_plans \
  --render-root .generated/jepa_counterfactual/phase2d_stage6_strict_lineage_render \
  --output .generated/jepa_counterfactual/phase2d_stage6_strict_lineage_spatial_v1.jsonl
```

Observed summary:

- candidate sequences written: `81`;
- complete-valid candidate sequences: `81`;
- future observation slots written: `162`;
- rendered frames indexed: `162`;
- missing topology seed rows: `0`;
- missing visual seed rows: `0`.

Row contract check:

- rows: `81`;
- unique primitive sequences: `81`;
- distinct first actions: `9`;
- complete-valid rows: `81`;
- lineage verified: true;
- Phase 2D full 81 two-block grid: true.

## Gate Evidence

Registered data audit command:

```bash
.generated/venvs/genesis_render_vulkan/bin/python \
  scripts/audit_jepa_phase2_data.py \
  --dataset train=.generated/jepa_counterfactual/phase2d_stage6_strict_lineage_spatial_v1.jsonl \
  --legacy-batch-size 8 \
  --output .generated/jepa_counterfactual/phase2d_stage6_strict_lineage_data_gate_audit.json
```

Exit status: `1`, expected.

The audit passes:

- foundation gate;
- all-sources-have-81-candidates check;
- eligible first-action minimum share;
- non-hold hard-negative coverage.

It fails the confirmatory gate because only one train smoke split exists.

Split manifest command:

```bash
.generated/venvs/genesis_render_vulkan/bin/python \
  scripts/create_jepa_phase2d_split_manifest.py \
  --split train=.generated/jepa_counterfactual/phase2d_stage6_strict_lineage_spatial_v1.jsonl \
  --output .generated/jepa_counterfactual/phase2d_stage6_strict_lineage_split_manifest.json
```

Exit status: `1`, expected.

The manifest verifies source-state lineage and file hashes, but fails because
validation, test-ID, and test-hard splits are missing and the train split has
only one scene/source state.

Validation-readiness command:

```bash
.generated/venvs/genesis_render_vulkan/bin/python \
  scripts/check_jepa_phase2d_readiness.py \
  --split-manifest .generated/jepa_counterfactual/phase2d_stage6_strict_lineage_split_manifest.json \
  --requested-stage validation \
  --output .generated/jepa_counterfactual/phase2d_stage6_strict_lineage_validation_readiness.json
```

Exit status: `1`, expected. The readiness layer blocks because the split gate
does not pass and C0/C1/C2 selected-checkpoint manifests are absent.

Training-start command:

```bash
.generated/venvs/genesis_render_vulkan/bin/python \
  scripts/check_jepa_phase2d_training_start.py \
  --split-manifest .generated/jepa_counterfactual/phase2d_stage6_strict_lineage_split_manifest.json \
  --train-data .generated/jepa_counterfactual/phase2d_stage6_strict_lineage_spatial_v1.jsonl \
  --validation-data .generated/jepa_counterfactual/phase2d_stage6_strict_lineage_spatial_v1.jsonl \
  --cell C2 \
  --run-class confirmatory \
  --output .generated/jepa_counterfactual/phase2d_stage7_training_start_preflight_smoke.json
```

Exit status: `1`, expected.

The preflight confirms:

- run class is supported;
- C2 is a registered primary cell;
- split-manifest files verify;
- lineage verifies;
- train data matches the manifest.

It blocks because:

- the confirmatory data gate does not pass;
- no validation split exists in the manifest;
- the requested validation path does not match any manifest validation split.

Trainer guard command:

```bash
.generated/venvs/genesis_render_vulkan/bin/python \
  scripts/train_jepa_phase2d.py \
  --train-data .generated/jepa_counterfactual/phase2d_stage6_strict_lineage_spatial_v1.jsonl \
  --validation-data .generated/jepa_counterfactual/phase2d_stage6_strict_lineage_spatial_v1.jsonl \
  --output .generated/jepa_counterfactual/phase2d_stage7_confirmatory_guard_should_not_train.pt \
  --cell C2 \
  --run-class confirmatory \
  --split-manifest .generated/jepa_counterfactual/phase2d_stage6_strict_lineage_split_manifest.json \
  --optimization-steps 1 \
  --evaluation-interval 1 \
  --source-states-per-batch 1 \
  --device cpu
```

Exit status: `1`, expected. The trainer exits before optimization with the same
preflight failure report. No
`.generated/jepa_counterfactual/phase2d_stage7_confirmatory_guard_should_not_train.pt`
checkpoint file was written.

## Verification

Focused command:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .generated/venvs/genesis_render_vulkan/bin/python -m pytest \
  lewm/tests/test_phase2d_readiness.py \
  lewm/tests/test_phase2d_training.py -q
```

Result: `9 passed`.

The tests verify:

- training-start readiness passes for a frozen manifest, C2, and matching
  train/validation paths;
- confirmatory diagnostic-cell launch is rejected;
- manifest/path mismatch is rejected;
- confirmatory gate failure is rejected;
- existing run-readiness held-out access rules still pass/fail as before.

Repository command:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .generated/venvs/genesis_render_vulkan/bin/python -m pytest lewm/tests -q
```

Result: `136 passed`, `3 subtests passed`, with six existing
`belief_encoder.py` nested-tensor warnings.

Verified Stage 7 artifact manifest:

`.generated/jepa_counterfactual/phase2d_stage7_training_start_gate_manifest.json`

Manifest verification: pass.

Manifest hash:

`9191985b5f668afe42e2f0b3582869597dc851eccaa4a86d26660bc7caa8b0cd`

## Current Full-Training Blockers

Supersession note: this blocker list was correct at Stage 7. The current
terminal readiness state is recorded in
`docs/lewm_jepa_phase2d_stage9_training_ready_2026-06-14.md`.

Full confirmatory training is not yet admissible. The next generated data must
provide:

1. train, validation, test-ID, and test-hard split files;
2. at least `32` train scenes and `16` validation/test scenes;
3. at least `16` source states per scene;
4. exactly `81` complete two-block candidates per selected source state;
5. verified topology and visual lineage for every source state;
6. frozen split manifest hashes before training starts.

After those pass, C0/C1/C2 confirmatory training may be launched with
`--run-class confirmatory --split-manifest ...`. Validation/test result access
remains blocked until selected-checkpoint manifests are frozen and the Stage 5
readiness gate passes.
