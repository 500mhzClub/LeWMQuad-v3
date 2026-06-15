# Phase 2V/2W/2X Swept-State Bridge Results

Date registered: 2026-06-15

Status: Phase 2W sanitized privileged target passed; Phase 2V and Phase 2X
RGB bridges failed. Train and validation data only. No `test_id` or
`test_hard` metric use.

## Research Question

Phase 2S showed that an action-conditioned swept-geometry state can support
first-primitive selection when privileged geometry is supplied. The remaining
question is whether a deployable model can recover a useful version of that
state from source RGB, and whether any non-deployable target fields were
responsible for the Phase 2S pass.

The tested bridge was:

```text
source RGB
  -> per-primitive swept-state prediction
  -> factorized affordance head
  -> safety-first primitive selector
```

## Implemented Changes

```text
lewm/models/source_action_utility.py
lewm/benchmarks/phase2v_swept_state_distillation.py
scripts/train_jepa_phase2v_swept_state_distillation.py
scripts/check_jepa_phase2v_swept_state_distillation_gate.py
lewm/tests/test_phase2v_swept_state_distillation.py
scripts/train_jepa_phase2s_swept_geometry_affordance.py
```

The Phase 2V trainer caches decoded source images, supports ROCm GPU training,
records finite metric checks, and reports both primitive selection and swept
state reconstruction error.

The Phase 2S trainer now supports feature-subset ablations with default
behavior unchanged. It also supports an optional primitive-ranking term with
default `0.0`.

## Quality Checks

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .generated/venvs/genesis_render_vulkan/bin/python \
  -m pytest lewm/tests/test_phase2v_swept_state_distillation.py \
  lewm/tests/test_phase2s_swept_geometry_affordance.py -q

Result: 5 passed

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .generated/venvs/genesis_render_vulkan/bin/python \
  scripts/train_jepa_phase2v_swept_state_distillation.py --help

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .generated/venvs/genesis_render_vulkan/bin/python \
  scripts/train_jepa_phase2s_swept_geometry_affordance.py --help

git diff --check

Result: clean
```

## Phase 2V: Full RGB-to-Swept-State Distillation

Phase 2V trained source RGB to predict all Phase 2S swept-state features, then
used the predicted state to predict factorized affordance labels.

Two bounded ROCm GPU variants were run:

1. plain state distillation plus factor prediction;
2. anchored factor head plus primitive-ranking loss.

Both failed the promotion gate.

Final plain variant:

```text
primitive_match_rate: 0.1484375
mean_target_utility_regret: 0.137367196
swept_state_mean_absolute_error: 0.151248738
swept_state_max_feature_mae: 0.462368429
worst_feature_name: best_progress_second_onehot_forward_fast
```

Final anchored/ranked variant:

```text
primitive_match_rate: 0.1328125
mean_target_utility_regret: 0.180375570
swept_state_mean_absolute_error: 0.141483277
swept_state_max_feature_mae: 0.456580222
worst_feature_name: best_progress_second_onehot_forward_fast
```

Gate failures included primitive match below `0.50`, regret worse than the
primitive action-only prior, and excessive max feature MAE.

## Phase 2W: Sanitized Privileged Target Audit

Phase 2W asked whether the explicit continuation-choice feature
`best_progress_second_onehot_*` was necessary for the privileged swept-state
target to pass.

The successful Phase 2W cell removed only that feature family and added a light
primitive-ranking term:

```text
--exclude-feature-prefix best_progress_second_onehot_
--primitive-ranking-loss-weight 0.10
```

This cell passed the primitive gate:

```text
primitive_match_rate: 0.51171875
mean_target_utility_regret: 0.036494391
selected_max_primitive_fraction: 0.41015625
oracle_max_primitive_fraction: 0.3515625
primitive action-only prior match: 0.1640625
primitive action-only prior regret: 0.058599013
```

This means the Phase 2S pass did not require the explicit best-second-action
onehot. The deployable target should exclude that feature family.

Two negative controls were also run:

```text
no-second-onehot without ranking:
  primitive_match_rate: 0.5078125
  mean_target_utility_regret: 0.063395312
  gate: failed regret

first-block-only target:
  primitive_match_rate: 0.23046875
  mean_target_utility_regret: 0.059671987
  gate: failed match and regret
```

The first-block-only failure shows that continuation aggregate information is
still important. The target should remove explicit second-action identity, not
all continuation-derived evidence.

## Phase 2X: Sanitized RGB Bridge

Phase 2X repeated the RGB bridge using the passed Phase 2W sanitized target and
the same `0.10` primitive-ranking weight.

It failed the gate:

```text
primitive_match_rate: 0.125
mean_target_utility_regret: 0.130230964
selected_max_primitive_fraction: 0.27734375
swept_state_mean_absolute_error: 0.130972728
swept_state_max_feature_mae: 0.350509197
worst_feature_name: continuation_max_heading_alignment
```

The state reconstruction thresholds passed, but primitive selection failed and
was worse than the primitive action-only prior.

## Interpretation

Phase 2W gives a cleaner target state for future work: action-conditioned swept
geometry with continuation aggregates, but without explicit best-second-action
identity.

Phase 2X shows that the current single-source RGB encoder still does not recover
that target well enough for action choice on scene-disjoint validation. The
failure is no longer explained by the second-action onehot. It points to the
perception/state bridge: the model can reduce average reconstruction error but
does not preserve the action-relevant geometry needed by the selector.

## Decision

Do not start a full JEPA training sweep yet.

The next bounded implementation must strengthen the deployable state bridge
before JEPA integration. Reasonable next candidates are:

1. predict the sanitized swept state from explicit local metric observations
   such as depth/rays or a learned occupancy-style intermediate state;
2. add temporal/memory observations rather than single-frame RGB, because local
   swept affordances may be partially unobservable from one image;
3. train a factorized slot state whose slots are tied to safety, clearance,
   goal progress, and heading, using the Phase 2W sanitized target as teacher.

Any candidate must pass the primitive gate before a full JEPA sweep. A JEPA
integration candidate must also pass the spatial action-identifiability,
zero/shuffled-action, stability, and persistence gates.

## Artifact Hashes

```text
82d7ebc0a7d2916d8378b73f1ecd0e31f9e786077d6a20864a258c62af5919a4  phase2v_swept_state_distillation_smoke.json
3236f081cbd80c1f031f19dc4dce3b9e6d0ab43d72b309aa342fc7816ce5d42d  phase2v_swept_state_distillation_smoke_gate.json
60165c5166f2327d32ba25c59c66b656ae16816af9b991b479fd65323fe3648e  phase2v_swept_state_distillation_ranked_smoke.json
a5b9c6db0811f4753c17461ad276606420d7cac7d6b7fef508a764aa46d967a9  phase2v_swept_state_distillation_ranked_smoke_gate.json
733f21996f727cd90d9a168e88d6b827fb79bb4566d6a7bac2b35faa0dbe685a  phase2w_swept_geometry_no_second_onehot_rank010_smoke.json
f78106f840ebe38b021f45bbb5e44c349aacd00ea0c9f1cfef3611be533c8bd2  phase2w_swept_geometry_no_second_onehot_rank010_smoke_gate.json
815d68cad1beecf256716096e4b415dcec78f6d89cbcecbf55951a8e4e4e8813  phase2x_source_sanitized_swept_state_rank010_smoke.json
8e76f12e2e6758e797ca014125fc88d026f4b953d479e5cab2c205002fd144a1  phase2x_source_sanitized_swept_state_rank010_smoke_gate.json
```
