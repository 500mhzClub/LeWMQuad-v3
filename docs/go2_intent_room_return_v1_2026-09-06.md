# Prospective multi-reference and persistent-intent room-return assay V1

This fresh development experiment combines two evidence-motivated engineering
changes: bounded sensor-only keyframe fallback and persistent return heading
intent. It is simulated physical execution, not robot hardware, JEPA control,
autonomous maze exploration, final evaluation or a matched causal ablation.
Preserve the predecessor coupled assay at 0/3 complete returns, with 12/13
native pose holds and 13/13 signed-winding holds. Its right corner boundary
failure at 0.06005187983247706 m remains a failure.

## Fixed collection

Run nominal_left, nominal_right, lower_friction_left once each, in that order.
Left spawn (.05,-.38,.375), yaw +.055; physics seed 2026090701 and appearance
seed 2026090705. Right spawn (-.05,.35,.375), yaw -.065; seeds 2026090702 and
2026090706. Low friction shares the left seeds/geometry but uses .15 rather
than 1.0 on both robot and floor. All four walls, physical floor, procedural
texture algorithm, frozen learned gait and checkpoint gains remain unchanged.
New starts and textures prohibit attributing predecessor differences causally
to either controller change. Subsequent scientific comparisons require paired
starts, scenes, sensor streams where applicable, and independent layouts/seeds.

Keep the predecessor seven stages, all .4 m forward requests, signed quarter
and half turns, stored observed corner/home, and one uninterrupted pose frame.
Use the frozen IntentRoomReturn and MultiReferenceVisualLedMotion sources.
The latest reference is tried first. Only its failure permits up to seven
retained alternatives under unchanged matching, six-cell support, gyro and
increment gates. Conflicting qualified poses (> .02 m or .10 rad pairwise)
stop; these diagnostic thresholds are not calibrated uncertainty. Rank valid
alternatives using sensor support/residual, never evaluator pose; promote the
current qualified frame after fallback. No failure reset or native pose input.

Persistent return intent keeps the approach heading across clipped .4 m
subgoals. Only the actual final-home subgoal requests heading zero. A tiny
corner residual cannot invent a reversed approach heading. No intermediate
leg or memory-stack event establishes home identity or mission success.

The six-cell empirical table remains fitted only to the two old nominal pulse
episodes; no current data fit or online adaptation. Keep coupled beam width256,
horizon min(24,remaining35 pulses), first-pulse feedback, forward .20 and yaw
+/-.45 commands for 2/5 ticks, minimum20 braking and10 quiet intervals,
maximum40 braking intervals. Retain all existing local and global limits:
1000 ticks/35 pulses per leg, 1 m excursion/.08 m overshoot, 3600 mission ticks,
140 pulses and36 legs. Terminal nonphysical failures drain10 guarded zeros;
native stops terminate physics immediately. Never increase budgets after a
failure, relaunch this attempt, resume its controller or modify frozen inputs.

## Storage and integrity

Exclusive fresh output:
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_intent_room_return_v1_attempt_001`.

This is new generated development data only. No source export, old-data move,
symlink redirect or custody-root search. The explicit output helper validates
the owned nonsymlink root and canonical ordinary artifact names. Repository
source/input bindings remain repository-relative under their existing guard;
external artifact bindings are separate and validated before/after auditing.
The exact output root is in the launch. Retain predecessor and recorded-replay
identities, recursive new-source/test bindings and installed dependencies.

Budget15 GiB for three trials plus40 GiB reserve on the recovery filesystem;
require55 GiB initially and45 GiB before each trial. At each decision record
free bytes and stop below40 GiB using the ordinary failure/zero-drain path.
This is an estimate, not a compression guarantee. Preserve partial data and
all old evidence. No data deletion or source copying to the external root.

## Independent acceptance

Reconstruct raw ideal sensors from native state and compare packet bytes,
raw depth/geometric rays, timestamps, native contacts/materials/gains, actual
new initial prefixes, command/slew/phase tape and complete runtime/memory replay.
The separate cached reader returns copied current packets only and admits all
3611 potential frames (initial +3600 commands +10 terminal drain). It changes
neither sensor generation nor decision causality; no frames may be dropped to
fit the predecessor reader's narrower bound.

Each completed local hold must satisfy the unchanged native limits at all501
physics poses over1 s: position .06 m, yaw .05 rad, speed .02 m/s, yaw rate
.05 rad/s. Separately verify signed unwrapped yaw from the observed goal
anchor. Full return requires all seven stages, every local pose and winding
hold, no native stop, and the independent full home XY(0,0)/yaw0 hold. A visual
candidate alone cannot pass. Report failures, stages, fallback events, native
error/path/duration and measured compute time without tolerance slack.

The completed recorded replay (4986/4986 frames, two fallbacks) is evidence for
the observer on old trajectories only; it is not an executed recovery. This
assay must generate and audit its own new trajectories. Ideal hidden-robot
RGB-D/gyro, level floor and paused compute remain limitations. No clearance,
uncertainty, realtime or deployment claim follows. The ultimate goal still
requires live branch/marker exploration and physical backtracking, useful
memory, matched direct/supervised/JEPA predictive-training and online-rollout
studies across independent mazes/training seeds, realistic sensor/timing tests,
and bounded hardware evidence when available.
