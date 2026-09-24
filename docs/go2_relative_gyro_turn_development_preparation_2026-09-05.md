# Relative gyro-based turning: source preparation

Maze exploration needs actions that inspect a branch and turn back at a dead
end. The existing learned bank has forward curves/reverse but no qualified
turn-in-place/reorientation primitive. A prototype now uses causal 50-Hz body
gyros to control a bounded turn without simulator yaw. It is a new component,
not a modification of any completed predictor, adapter or physical experiment.

The controller integrates ordered body angular velocity on SO(3), using the
average adjacent samples over each 20-ms interval. Heading is the current forward
axis projected into the **initial body frame**; its rate includes all three body
gyro axes through that relative rotation. This is not world yaw or global
odometry. No translation, gravity alignment, absolute pose, map or future sample
enters the controller. Gyro bias and discretization error can accumulate.

Requested forward/lateral velocity remain zero. Yaw request is 1.5×wrapped heading
error, clipped to ±0.35 rad/s, using the existing actuator-side slew contract.
The source prototype uses heading tolerance 0.08 rad, projected heading-rate
tolerance 0.1 rad/s, and a 0.3-s zero-command dwell. A turn must finish within
12 s; entering tolerance at the deadline cannot extend that budget. Relative
targets are bounded to ±pi. Missing/invalid/rewritten gyro samples, clock/episode
discontinuity or an ill-defined forward projection stop the controller. Chunk
integration is transactional: bad input cannot leave partially integrated state.

Thirteen synthetic tests pass: handedness/orthogonality, both quarter-turns and a
half-turn under an ideal command-following rate model, nonplanar constant body
rates, overlap handling, failures, timeout and deadline dwell. These tests do not
establish Go2 translation drift, stopping, heading accuracy, contact safety,
calibration or hardware performance. The component is **not physically qualified**.

## Next bounded physical question

Specify a paired development assay before execution: gyro-feedback versus nominal
timed turn, three relative targets (+90°, −90°, +180°), three initial headings,
fixed fresh seeds and the corrected gait in the known large enclosed arena.
That would be 18 trials, not an independent maze benchmark. Keep the same yaw
cap, per-trial maximum budget and post-controller release. The timed baseline's
duration comes from target/requested rate, not fitted assay outcomes.

Record live RGB/gyro/control packets and every controller decision; keep native
contact/stability monitors outside the policy boundary. Independently compare
integrated relative orientation against raw physical relative rotation at the
same timestamps, and measure actual heading error, translation drift and release
motion. Freeze the physical acceptance tolerances and stopping rules before
execution. Preserve failures rather than extending duration or gain-searching.

This primitive can support looking/reorientation, but it must not be inserted
into the old JEPA as if its future feedback-generated command tape were known at
selection. A predictor for a feedback primitive needs its declared primitive
parameters/policy as conditioning, or another explicitly causal model. Broader
training-state/temporal coverage and online memory are still required for
successive decisions and full exploration/return.
