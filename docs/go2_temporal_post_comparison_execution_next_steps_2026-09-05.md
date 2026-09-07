# Next execution steps after the temporal comparison

The [completed result](go2_temporal_rgb_body_learning_comparison_development_v1_result_2026-09-05.md)
provides no positive JEPA utility claim. Preserve all nine checkpoints and the
negative evidence. Continue toward an effective RGB/body-sensor navigation
system and a defensible answer about JEPA; scientific success does not require
the JEPA arm to win.

## 1. Close the coordinate and live-input interface

Update: this stage is complete as a recorded-input integration check; see the
[adapter/orientation result](go2_temporal_online_adapter_replay_development_v1_result_2026-09-05.md).
The six-method adapter uses all three fixed seeds, and 22 focused tests plus
4,080 comparable prediction replays pass. This does not establish policy utility.

The live four-frame history buffer has passed recorded-stream replay. Next add
a causal relative-orientation tracker using the qualified gyro integration
convention, without the turn controller's action-specific 12-second deadline.
It must track every new body-gyro sample exactly once, reject rewritten/missing
data and reset explicitly. Verify it against recorded physical orientations in
an evaluation-only replay; world orientation must never enter its input.

This is needed to keep an externally supplied **initial-frame direction** fixed
while the body turns. A constant leftward body-frame cue rotates with the robot;
silently calling that the same world-fixed target is a task-semantics error.
Relative heading is not translational odometry and cannot locate a fixed point
goal. Do not claim point-goal arrival using orientation alone.

Then implement an online temporal adapter using the audited final checkpoints,
all three seeds per arm, and the live four-frame buffer. Average probabilities,
not logits. Expose direct and rollout heads separately, exact candidate plans,
input/prediction/model bindings and timing. Never substitute canonical training
RGB for a live image. Invalid inputs must trigger an explicit safe command at
the caller; an exception alone does not stop the robot.

## 2. Execute a bounded successive-decision diagnostic

Update: the [fixed 144-trial protocol](go2_successive_choice_maze_development_v1_2026-09-05.md)
and runner are implemented, 18 additional integration/metric tests pass, and the
one-shot fresh panel has been launched. Do not alter the bound protocol/source
or restart the attempt. Interpret results only after independent raw audit.

Prepare a new fixed development protocol before execution. Proposed scope:
eight fresh topology-disjoint procedural layouts (candidate seeds2026091800–07),
three supplied initial-frame directions, and six methods: all stop; direct-only
direct head; supervised-direct; supervised-rollout; JEPA-direct; JEPA-rollout.
This gives 144 physical trials with fixed three-seed ensembles, not selected
individual seeds. Validate topology disjointness without inspecting protected
material or resampling on outcome.

Use the corrected gait and an explicitly declared common teacher initialization.
After initialization, use only current RGB/body/control history and the supplied
direction. Replan every .5 seconds for eight decisions, using the first .5-second
prediction horizon of the existing five-action bank. Remaining tensor slots are
unknown/masked, not zero commands to execute. Record every100-ms packet so the
history and orientation tracker remain current between replans. Finish with the
specified zero-command release and native simulator emergency stopping.

Fix the ranking cost, duration, safety/continuation measures, trial ordering,
prefix validity, geometry and source bindings in that protocol. A reasonable
continuation of the old ranking is 10×predicted contact probability plus distance
to a .8-m directional cue transported into the current body frame. This is an
instantaneous directional cue, **not a fixed point goal**. Primary physical
reports should separate contact/failure incidence from signed displacement along
the supplied initial direction, lateral deviation, duration and release motion.
Do not define a favorable post-hoc scalar by hiding no-progress or stopped trials.

Report actual action changes, switching from moving commands to stop/reverse,
prediction error at the executed .5-second endpoints and the contribution of
each observed failure mode. Score all trials, including unavailable prefixes and
sensor failures. Compare paired outcomes by layout; one-step prediction accuracy
or a contact-free stop does not establish task completion. This panel is a
conditional directional-continuation diagnostic, not autonomous maze exploration.

The .5-second horizon is supported at all derived offsets for the executed
continuation action, but switched candidates and later policy states remain a
new distribution assumption. State that limitation upfront. The diagnostic
exists to test it, not certify it from the data loader's passing tests.

## 3. Repair the identified failure, not a convenient metric

If action switches fail, collect actual counterfactual suffixes from identical
moving prefixes, including braking and changing curvature. Keep all windows of
a layout in its assigned role; balance past-action/future-action combinations
and report prefix failures. Do not impute unexecuted alternatives. Prefer this
targeted causal coverage over another loss-weight/seed search on the same eight
validation layouts.

If stopping dominates despite clear observed exits, distinguish perception/risk
error, action-bank limitations, uncertainty and the lack of an exploration/time
objective. Add separately qualified look/reorientation decisions and observed
exit hypotheses; do not secretly feed a gyro controller's future realized
feedback commands into the current fixed-action predictor.

## 4. Complete the actual maze system and final evidence

Integrate relative translation estimates, uncertain place association, observed
exits/beacons and frontier selection with the directed traversal graph. Keep
unconfirmed reverse paths distinct from demonstrated traversals and account for
wrong merges and missed revisits. Compare the same controller with and without
persistent memory on discovery and return, with no oracle graph or unseen goal
image. A recorded route retrace alone is not evidence of general exploration.

Finally freeze methods and task thresholds, evaluate genuinely independent maze
layouts and shifts with maze-level uncertainty, then conduct bounded real-Go2
work with actual sensor calibration/timestamps and human safety supervision.
Hardware access remains necessary for the transfer deliverable, but it does not
block the remaining software/simulation work. Keep the ultimate goal active;
none of these intermediate diagnostics substitutes for it.
