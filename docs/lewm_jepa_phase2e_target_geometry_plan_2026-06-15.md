# Phase 2E Target Geometry Pilot Plan

Date registered: 2026-06-15

Status: implementation pilot; train and validation data only; no `test_id` or
`test_hard` access.

## Trigger

Phase 2D failed before a valid selected checkpoint:

- C0/C1 completed on the corrected ROCm GPU runtime but failed the one-step
  persistence gate;
- original C2 failed through GPU concurrency OOM and non-finite objective
  dynamics;
- detached-control C2 stayed finite for a bounded smoke but collapsed, failed
  hard-negative action identifiability, and lost to persistence by `247.69x`.

This activates the preregistered Phase 2D stop rule: stop direct image-aligned
patch dynamics as the primary state and test redesigned target geometry.

## Hypothesis

The current patch-token objective is dominated by image-aligned appearance
persistence. A smaller learned slot geometry may reduce patch-grid persistence
dominance and make action-conditioned consequences more measurable under the
same corrected Phase 2D controls.

This is not an object-discovery claim. The learned slots are only a bounded
target/state-geometry pilot.

## Pilot Cell

Use the existing Phase 2D trainer, data, source grouping, masks, hard negatives,
EMA target, action-identifiability objective, persistence baseline, stability
diagnostics, and smoke gate.

Change only model-side target/state geometry:

- `target_geometry=slot`;
- `num_target_slots=16` by default;
- learned query attention pools encoder patch tokens into slots;
- the predictor, EMA target projection, action controls, losses, and
  diagnostics operate on slots instead of patch tokens.

The patch C2 result remains the reference. Slot geometry is an amended Phase 2E
pilot, not a replacement Phase 2D confirmatory result.

## Gate

A bounded smoke may be promoted to a full pilot manifest only if the executable
smoke gate passes:

- no collapse, effective-rank, or near-static-target warning;
- real action beats hard-negative actions by at least `0.10` of target change;
- real action beats zero action by at least `0.10` of target change;
- one-step real prediction beats persistence with ratio `< 1.0`;
- the trainer reports `checkpoint_selection_permitted: true`.

If the bounded smoke fails, do not launch a full slot-geometry run. Record the
failure and move to the next target-family candidate: factorized
affordance/dynamics/event state.

## Bounded Smoke Command

The first smoke uses one seed and validation truncation only to test numerical
and scientific gate viability:

```bash
/home/andrewknowles/TinyQuadJEPA/bin/python \
  scripts/train_jepa_phase2d.py \
  --train-data .generated/jepa_counterfactual/phase2d_min_sources/train_spatial_future_v1.jsonl \
  --validation-data .generated/jepa_counterfactual/phase2d_min_sources/validation_spatial_future_v1.jsonl \
  --output .generated/jepa_counterfactual/phase2d_min_sources/phase2e_slot_c2_smoke.pt \
  --cell C2 \
  --run-class smoke \
  --optimization-steps 128 \
  --evaluation-interval 64 \
  --source-states-per-batch 2 \
  --max-validation-rows 400 \
  --seed 20260614 \
  --device cuda \
  --max-grad-norm 1.0 \
  --target-geometry slot \
  --num-target-slots 16
```

The smoke-gate checker must then be run against the emitted JSON report.

## Quality Gates Before Smoke

Focused implementation gate:

```text
27 passed
```

Diff hygiene:

```text
git diff --check passed
```

## Bounded Smoke Result

Status: failed the registered smoke gate. Do not launch a full slot-geometry
training run from this pilot.

The smoke completed on the ROCm GPU runtime and did not reproduce the original
C2 infinite-gradient failure. The learned-slot geometry therefore improved
numerical behavior relative to the original C2 objective, but it did not solve
the research problem.

Final validation gate:

```text
checkpoint_selection_permitted: false
gate_pass: false
stability_pass: false
collapse_warning: true
effective_rank_warning: true
mean_feature_std: 0.047641
effective_rank: 1.772106
effective_rank_fraction: 0.036919
zero_action_advantage: 0.050411
hard_negative_action_advantage: -9.082974
one_step_rollout_persistence_ratio: 24.960993
```

Executable smoke-gate result:

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

## Interpretation

The pilot separates two effects:

- learned slots reduce the numerical instability observed in the original C2
  run;
- learned slots still produce a low-rank state whose action-conditioned future
  loses badly to persistence and does not rank the real action above hard
  negatives.

This falsifies the narrow hypothesis that a smaller learned target/state
geometry is sufficient to escape image-aligned persistence dominance under the
current Phase 2D objective. It does not falsify JEPA navigation or learned-slot
state models generally.

## Decision

Stop Phase 2E slot geometry at the bounded smoke. No full slot-geometry matrix
is justified.

The next implementation target is a factorized affordance/dynamics/event state.
The next bounded smoke must test whether action-conditioned predictions encode
navigation-relevant events or affordance changes directly enough to beat:

- zero action by at least `0.10` of target change;
- hard-negative actions by at least `0.10` of target change;
- one-step persistence with ratio `< 1.0`;
- collapse and low-rank stability diagnostics.
