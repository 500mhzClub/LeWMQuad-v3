# Observed-image arrival recognition with the dense world-model planner

Status: **COMPLETE: 2/2 local final arrivals, no contact**. Native sessions
38708 and 27621 exited 0; reader session 6325 exited 0. Each run used the
full 20-decision budget and five final quiet ticks. No process remains live.

| Exposed geometry | Parent final XY / yaw | With arrival recognition final XY / yaw | Final within 3 cm / 5 degrees | Contact |
|---|---:|---:|---|---|
| cluster 02, family_episode_026 | 2.99 cm / 6.73 degrees | 2.62 cm / 3.66 degrees | Yes | No |
| cluster 03, family_episode_003 | 3.33 cm / 8.06 degrees | 2.62 cm / 3.66 degrees | Yes | No |

Both runs preserved the original planner's first four left-arc decisions and
latched hold at tick 30, after two seconds of active approach. At that instant,
actual position error was 0.390 cm and heading error 0.139 degrees. Every
subsequent camera frame remained inside tolerance for the remaining 8.5 seconds
of execution. The longest within-goal streak was 87 frames in both cases.
The supplied images differ across scenes, but action sequences and resulting
final physical errors are identical. These are two related visual settings,
not independent dynamics replications.

The intervention adds a training-only signed visual-goal readout to the parent
adapted predictor and scalar goal metric. Its inputs are the current RGB and
supplied goal RGB. On its first estimate within 3 cm and 5 degrees, the
controller latches zero-command hold, continues observing, and stops invoking
the predictor. There is no simulator-pose arrival trigger. Native pose is used
only afterward by the evaluator. The fixed budget is not shortened.

The reader verified identical RGB through tick 30 against each original parent
run, matched pre-latch actions and forecast costs, all subsequent hold commands,
actual goal membership at latch and every remaining frame. The observed-command
checks passed for the four predicted action blocks and subsequent hold blocks.
Run wall times were 37.12 and 38.00 seconds; physics paused for computation.

This resolves the observed local departure-after-arrival failure in these two
development tasks. It does not improve the predictor itself or show that
prediction beats direct visual feedback. The readout previously missed seven
of nine sampled within-goal states; the latch succeeds here because the robot
passes through a state it recognizes. These tasks informed the intervention,
and a latched false positive could prevent recovery on another task. No claim
of independent generalisation, complete maze navigation, isolated JEPA-training
advantage, real-time operation or hardware validation follows.

Next compare the frozen planner plus this arrival rule against direct visual
feedback using the same encoder, signed goal readout, goal images, command
limits, observation timing and execution budget. Define that comparator before
testing on fresh development layouts/goals; do not tune it against these two
successes. A wider unseen-maze evaluation, persistent routing memory and
backtracking remain required for the full goal.

Follow-up feedback comparison is complete:
`go2_direct_visual_feedback_pilot_2026-09-17.md`. Direct feedback succeeds on
one task and falsely latches arrival on the other, exposing a shared readout/
latch limitation despite the two successful planner trajectories here.

Before execution, 73 GiB RAM was available and no competing experiment was
running. Two processes used CPU cores 4-7 and 8-11 and shared the R9700 GPU.
The root-volume preparation did not create an attempt because capacity was
insufficient. Before freezing, output was moved to the dedicated experiment
volume, with 1,047,384,064 bytes available versus a measured-size requirement
of 1,036,621,019 bytes (512 MiB reserve plus 2.5 reference-case sizes).
Both completed; approximately 599 MiB remained. All RGB, depth, physics,
commands and outcomes are retained. Further native collection needs headroom.

The inherited plan field `goals_commands_timing_budget_stops_unchanged`
refers to the task setup and physical safety stops; the controller's arrival
rule explicitly changed, as recorded by `stopping_rule_added: true`.

Plan: `go2_dense_visual_arrival_pilot_plan_2026-09-17.json`.
Result: `go2_dense_visual_arrival_pilot_result_2026-09-17.json`.
Controller: `lewm/dense_visual_arrival_control_development.py`.
Runner/reader: `scripts/run_go2_dense_visual_arrival_pilot_development.py` and
`scripts/read_go2_dense_visual_arrival_pilot_development.py`.
Output: `/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1/go2_dense_visual_arrival_pilot_v1_attempt_001`.
