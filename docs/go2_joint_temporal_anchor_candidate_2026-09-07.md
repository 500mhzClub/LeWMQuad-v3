# Joint-rotation continuity candidate: implemented, not adopted

The completed [post-hoc analysis](go2_tracking_posthoc_raw_accuracy_result_2026-09-07.md)
exposes accepted translation error up to 287.510 mm under shared gyro bias.
The gyro-conditioned anchor and previous-frame rotations share that bias by
construction. This motivates the conditional joint-RGB-D comparison already
described in `go2_independent_heading_followup_plan_2026-09-07.md`.

`lewm/joint_temporal_anchor_continuity_development.py` implements a separate
`JointTemporalAnchorRGBDPose` and `JointTemporalAnchorVisualLedMotion`.
Both retained-reference and previous-frame candidate calculations use the
existing `joint_rgbd_rigid_pose_development.register(..., mode='joint')`.
There is no new rigid solver, global monkeypatch or frozen-source modification.
The wrapper correctly labels visual rotation and gyro consistency monitoring.

The candidate inherits descriptor matching, consensus rules, reference ranking
and retention, motion envelopes, ten-frame bridge budget, anchor/increment
disagreement gates, and terminal sensor failure semantics. It records each
qualified local fitted rotation, its reference rotation, composed rotation,
gyro comparison, frame/time identity and inlier support. Selected anchor and
increment witnesses are explicit and separately nullable. A failed measurement
does not claim an incremental rotation witness; qualified conflicting witnesses
can remain evidence without creating an accepted pose. Snapshot copies do not
alias mutable estimator state. It estimates no gyro bias or calibrated bound.

The 50-test candidate/joint-solver/original-continuity suite passed in 11.86 s.
Tests include actual synthetic rendered motion, signed shared gyro bias,
anchor loss/rejoin, an unchanged terminal eleventh bridge, missing current RGB,
depth or gyro, invalid clocks/identity and witness nonmutation. Matrix products
and rotation disagreement are independently reconstructed in the synthetic
tests. This is not yet a full independent reader of real candidate outputs.

The initial run had two test failures: it incorrectly expected exact identity
rotation from identical images despite SIFT/LK interpolation producing about
1.53e-9 rad of numerical motion. The corrected test requires exact equality
between the same joint visual fit with and without the signed gyro perturbation.
No estimator gate or physical acceptance tolerance changed. This tests the
fitting mechanism, not a claim that image matching is noiseless.

No retained tape replay, native collection, control adoption or hardware motion
has been performed for this candidate. Original source, failed attempts and the
completed post-hoc result remain unchanged. The next unfinished experiment is
the matched candidate evaluation, followed by fresh closed-loop execution only
if the development trade-off supports it.

## Fixed scope for the next development comparison

Use all eight explicitly admitted tracking tapes, including the two baffle
tracking failures and three strict depth failures. These exposed tapes are
development data. Compare gyro continuity and joint continuity under nominal,
positive and negative 0.02 rad/s timestamp-consistent shared body-z gyro bias,
and one missing-current-RGB frame. Bias and RGB onset are frame 84. Retain every
443-frame stream and each arm's terminal rows; do not reset or restart an arm.
Keep matching, reference management, sensors, fault timing and acceptance gates
matched. The negative bias condition is new; it requires its own gyro baseline.

Previously saved nominal/positive-bias/missing-RGB gyro-continuity outputs may
be reused only after exact source, packet, transformation and stream binding
checks. Disclose any such reuse and its historical timings. New candidate
timing is not a contemporaneous paired speed comparison against saved timing.
No original raw audit needs to be repeated merely to score this candidate.
All new sensor streams must be accounted for before evaluator-only native truth
is supplied to any scorer. Use the explicit sensor-representation interface
and preserve the strict visibility failures.

The development continuation criterion must be fixed before inspecting the
candidate outputs: no loss of nominal availability on any tape, no accepted
nominal or biased pose beyond the existing empirical 20 mm / 2 degree allocation,
and a reduction in paired bias-induced error on shared support for both signs.
Report complete-tape failures independently of shared-support errors. Missing
current RGB must remain terminal. A failed criterion retains this candidate's
negative result; it does not trigger changed thresholds, a favorable subset,
or controller adoption. Even meeting these development criteria requires fresh
validation observations and full-loop timing before deployment claims.

Recheck hardware and benchmark bounded representative per-tape concurrency
before launching that new workload. Implement and test its writer/admission/
numerical interface with a finite derived output root. The existing numerical
pose checker can check joint matrices, but the old continuity checker explicitly
requires gyro-mode rows; do not falsely present it as an unchanged joint-mode
verification. The source candidate alone is not the completed experiment.

The long-term goal is active. The handoff's section 9 is complete; continuous
sensor-feedback execution, informative visual action selection, learned online
planning, memory/backtracking, generalization and deployment evidence remain.
