# Position units × optimization budget: bounded development comparison V1

One final same-room optimization diagnostic before independent-layout collection.
This protocol is fixed after the negative neural pilot, successful aggregate
action/time baseline and first-batch gradient diagnosis. It is development-only,
not preregistered independent generalization evidence.

## Fixed intervention and matching

Compare `raw` (the exact frozen training_loss and optimizer implementation) with
`position_6cm`: divide only predicted and target XY by 0.06 metres inside the
four-channel SmoothL1 mean. This physical scale is the existing external local
arrival tolerance in `lewm/goal_region_pulse_servo_development.py`, not a fitted
target statistic or a search result. Leave unit sin/cos, BCE, EMA latent MSE,
variance/covariance penalties and their coefficients unchanged. This tests XY
unit scaling only, not a complete task-metric loss or a new angle formulation.
The Huber transition is now 6 cm per XY coordinate; this does not certify a
6 cm planar bound or compensate for state-dependent dynamics.

Three conditions: direct, supervised_rollout, jepa. Three paired initialization
seeds: 2026090721, 2026090722, 2026090723. Two objectives: raw and position_6cm.
Eighteen fresh fits, fixed order seed → condition → objective. Same architecture,
latent width32, AdamW lr0.001, weight decay0, global gradient clip1, EMA0.99,
CPU one thread, deterministic algorithms. No residual baseline, input scaling,
architecture, camera, action vocabulary or inference-interface changes.

Each fit receives exactly the original12 batches of6 draws repeated10 times:
120 updates,720 draws,62 distinct windows;2160 updates overall. Cache those12
already-materialized training batches, without adding labels or frames. The
12-update snapshot is the prefix of the same fresh120-update trajectory, not
a second independently trained model or a resumed old attempt. Snapshot and
score only updates12 and120. Verify each raw12 model tensor hash matches its
completed predecessor, without loading predecessor weights into training.

Score restored snapshot weights on all185 train-role windows/917valid targets
with exact2.2/2.5s endpoints and independent masks. Save predictions and report
all trained direct/recursive heads, zero and the frozen72-draw empirical control,
by action, actual time and recorded condition. Repeating every draw10times leaves
the empirical fitted means unchanged. No best-checkpoint selection, additional
schedule, early success stop, failed-seed exclusion or scientific resume.

## Integrity and resource limits

Bind the frozen pilot launch/result, empirical/gradient launch/results and all
inherited source/input identities before materializing data. Freeze these new
source/test/protocol identities in the exclusive launch. Existing sources,
checkpoints and outputs remain unchanged; no clean source export or sealed access.
Write exclusively to `.generated/go2_pulse_position_scale_budget_v1_attempt_001`.
Preserve10GiB free storage and allow1GiB for this diagnostic. No GPU, new physics,
robot commands, online control integration or deployment authority is inferred.

Save immutable per-update records, one aggregate per completed fit, and both
model/optimizer snapshots with metrics/prediction files. Reload and verify tensor
identity before scoring. A failure stops this attempt and preserves completed
outputs; it does not authorize overwrite, retry or resume. Verify source/input
identities after completion and independently check saved predictions/metrics,
optimizer step counts and matched schedule exposure.

## Interpretation and exit from this diagnostic stage

At each budget, compare scaling versus raw within the same condition/seed. Within
each objective, compare12 versus120 updates. Compare JEPA versus matched direct/
supervised heads only at equal objective and budget. Report every seed and auxiliary
head; do not select a favorable condition or hide low friction. All scores remain
one-room training-role resubstitution, not independent-layout statistics.

If scaling helps, it is evidence for this optimization intervention under this
fixed budget/data, not proof of general sensor utility. If longer raw training
helps as much, the12-step pilot was insufficient to support a scaling explanation.
If both fail the empirical baseline, preserve the negative result and inspect
capacity/input dependence on new data rather than start a scale search. In every
case the next stage is independent connected-layout/state/support/action/contact
collection and sensor/action ablations. Do not install the winner as a controller
or keep extending this room experiment. Local tracking/execution, online memory/
physical backtracking, matched maze studies, realistic deadlines/body sweep and
bounded hardware remain outstanding parts of the original goal.
