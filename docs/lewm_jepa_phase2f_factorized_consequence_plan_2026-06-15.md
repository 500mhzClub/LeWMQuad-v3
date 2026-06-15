# Phase 2F Factorized Consequence Target Pilot

Date registered: 2026-06-15

Status: implementation pilot; train and validation data only; no `test_id` or
`test_hard` access.

## Trigger

Phase 2E learned-slot target geometry completed on the ROCm GPU runtime but
failed the registered smoke gate:

```text
stability_pass: false
hard_negative_action_advantage: -9.082974
zero_action_advantage: 0.050411
one_step_rollout_persistence_ratio: 24.960993
checkpoint_selection_permitted: false
```

This means generic image/slot target prediction remains dominated by low-rank
appearance persistence. A full slot-geometry run is not justified.

## Hypothesis

Navigation-relevant action consequences may need to be represented as a
factorized target family rather than only as predicted future image tokens.

The first bounded Phase 2F test adds an auxiliary sequence-level consequence
head trained from privileged generator labels already present in the Phase 2D
JSONL rows. These labels are training/evaluation signals only. They are not
runtime inputs.

## Target Contract

Each candidate sequence may provide a masked normalized consequence vector:

- target progress;
- clearance gain;
- minimum swept clearance;
- p05 swept clearance;
- unsafe sample fraction;
- enters unsafe grid state;
- ends unsafe grid state;
- target recoverability;
- heading alignment.

Nullable fields remain masked. Continuous fields are clipped into comparable
ranges before MSE loss. Boolean fields are represented as `0.0` or `1.0`.

## Pilot Cell

Use the corrected Phase 2D trainer and validation gate with these amendments:

- `cell=C2`;
- `--detach-action-control-state`;
- `--max-grad-norm 1.0`;
- `--consequence-loss-lambda 1.0`;
- default patch target geometry.

The JEPA latent prediction objective, hard-negative action objective,
zero-action objective, EMA target, spatial variance floor, and persistence gate
remain active. The consequence head is auxiliary; it does not replace the
latent world-model gate.

## Promotion Gate

A bounded smoke may be promoted to a full pilot manifest only if:

- all trainer metrics and gradients remain finite;
- consequence labels are present and the consequence loss is finite;
- no collapse, effective-rank, or near-static-target warning is raised;
- real action beats hard-negative actions by at least `0.10` of target change;
- real action beats zero action by at least `0.10` of target change;
- one-step real prediction beats persistence with ratio `< 1.0`;
- the trainer reports `checkpoint_selection_permitted: true`;
- the executable smoke gate passes.

If the smoke fails, do not launch a full Phase 2F run. Record whether the
failure is numerical, event-target-only, or still a latent gate failure.

## Bounded Smoke Command

```bash
/home/andrewknowles/TinyQuadJEPA/bin/python \
  scripts/train_jepa_phase2d.py \
  --train-data .generated/jepa_counterfactual/phase2d_min_sources/train_spatial_future_v1.jsonl \
  --validation-data .generated/jepa_counterfactual/phase2d_min_sources/validation_spatial_future_v1.jsonl \
  --output .generated/jepa_counterfactual/phase2d_min_sources/phase2f_consequence_c2_smoke.pt \
  --cell C2 \
  --run-class smoke \
  --optimization-steps 128 \
  --evaluation-interval 64 \
  --source-states-per-batch 2 \
  --max-validation-rows 400 \
  --seed 20260614 \
  --device cuda \
  --max-grad-norm 1.0 \
  --detach-action-control-state \
  --consequence-loss-lambda 1.0
```

The smoke-gate checker must then be run against the emitted JSON report.

## Implementation Gates Before Smoke

Focused tests:

```text
28 passed
```

Diff hygiene:

```text
git diff --check passed
```

## Bounded Smoke Results

Status: failed. Do not launch a full Phase 2F training matrix from these pilots.

### C2 Consequence Pilot

The registered C2 consequence smoke failed numerically before the first
validation:

```text
output: .generated/jepa_counterfactual/phase2d_min_sources/phase2f_consequence_c2_smoke.pt
cell: C2
lr: 3e-4
consequence_loss_lambda: 1.0
failure_step: 51
error: nonfinite_phase2d_gradient_norm
gradient_norm: Infinity
```

No checkpoint or JSON report was written.

### C2 Stabilization Variant

A lower-learning-rate, lower-consequence-weight variant also failed before
validation:

```text
output: .generated/jepa_counterfactual/phase2d_min_sources/phase2f_consequence_c2_lr1e4_lam0p1_smoke.pt
cell: C2
lr: 1e-4
consequence_loss_lambda: 0.1
failure_step: 62
error: nonfinite_phase2d_gradient_norm
gradient_norm: Infinity
```

No checkpoint or JSON report was written.

Interpretation: adding the factorized consequence head does not stabilize the
C2 hard-negative hinge objective. The C2-plus-consequence path remains
numerically invalid under bounded smoke conditions.

### C1 Consequence-Isolation Variant

To isolate whether consequence supervision itself was unstable, I ran a C1
variant with EMA, the auxiliary consequence target, and the same validation
controls, but without training the C2 hard-negative and zero-action hinge
losses.

Command amendment:

```text
cell: C1
lr: 1e-4
consequence_loss_lambda: 1.0
max_grad_norm: 1.0
```

This run completed and wrote artifacts, but failed the executable gate:

```text
checkpoint_selection_permitted: false
gate_pass: false
stability_pass: false
collapse_warning: true
effective_rank_warning: true
mean_feature_std: 0.021959
effective_rank: 2.713607
effective_rank_fraction: 0.056533
hard_negative_action_advantage: -53.045634
zero_action_advantage: 0.033336
one_step_rollout_persistence_ratio: 229.692053
final_train_consequence_loss: 0.185720
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
a5f373bc863da92479923e1eddc9326ed5133c32defa7503b3d81196470d85ec  .generated/jepa_counterfactual/phase2d_min_sources/phase2f_consequence_c1_lr1e4_lam1_smoke.json
1ae8f7da454a7266bbf9c3dd784b092d920b7fdb812cd4db821209d0a5592a8d  .generated/jepa_counterfactual/phase2d_min_sources/phase2f_consequence_c1_lr1e4_lam1_smoke.pt
d74ae516dae6771872df3680d08e024ba3df11c86fc77f53fe461a68150ada3c  .generated/jepa_counterfactual/phase2d_min_sources/phase2f_consequence_c1_lr1e4_lam1_smoke_gate.json
```

## Interpretation

The factorized consequence head is trainable in isolation from the C2 hinge:
the C1 variant stayed finite and reduced consequence loss on training batches.
However, the latent state still collapsed and the action-conditioned latent
prediction still lost badly to persistence. The auxiliary event target did not
make the learned latent future identify action consequences under the registered
validation controls.

This is a narrower failure than "event targets cannot help JEPA navigation." It
only rejects this implementation: a pooled sequence-level consequence head
attached to the same image-aligned latent prediction objective.

## Decision

Stop Phase 2F at bounded smoke. No full Phase 2F run is justified.

The next target-family candidate must change where consequence structure enters
the world model. Plausible next options are:

- predict consequences from dedicated affordance/dynamics tokens, not from a
  mean-pooled latent future;
- supervise per-action candidate utilities directly across the same source
  state's 81-way counterfactual set;
- replace the image-prediction target with a navigation-event target for the
  dynamics branch while keeping appearance tokens separate;
- add an uncertainty or low-rank penalty that is tied to action-conditioned
  consequence variation rather than global token variance.

Any next candidate must keep the same no-test-access rule and must pass a
bounded smoke before full training.
