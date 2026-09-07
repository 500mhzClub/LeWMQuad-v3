# Next: bounded pulse feedback, then persistent navigation memory

## Current handoff

Pulse-feedback V1 completed with an [audited0/4 result](go2_pulse_feedback_servo_result_2026-09-06.md).
The distinct [goal-region successor](go2_goal_region_pulse_servo_result_2026-09-06.md)
now has3/3 nominal full move/turn/hold successes and0/1 low-friction success,
confirmed by raw replay and native full-second scoring;2,388tests pass.
Proceed to the [continuous-execution/online-memory plan](go2_local_execution_to_online_memory_next_steps_2026-09-06.md),
preserving the failure history and all sensing/terrain/generalization limits.
The original implementation plan below remains as chronology.

The [command-pulse collection and audit](go2_command_pulse_response_result_2026-09-06.md) completed all four fixed development schedules.
Use its raw audit and per-event sensor/native response reports, not requested
twist multiplied by time, to design the next local execution baseline. These
are four exposed episodes with two starts/orderings, not held-out maze results.

## Finish local execution with the action interface now measured

1. Build a distinct sensor-feedback controller around finite drive pulses and
   observed braking. Use the unchanged gait and declared bank commands
   forward .20m/s and yaw ±.45rad/s. The measured 2- and 5-tick durations are
   candidate action choices; neither is a fixed physical displacement. Preserve
   the actual command-slew sequence and all stop/partial-failure records.
2. Separate action-specific response from retrospective course. A forward
   pulse's observed net displacement can inform subsequent forward steering;
   translation from a turn must not be used as predicted forward course.
   Start with a transparent geometric/empirical response baseline using ONLY
   past sensor poses and executed actions. If fitting response priors from
   collection, use visual labels, report condition/state dispersion and freeze
   them before fresh physical validation. No online simulator friction label.
3. Bound each correction in duration and then re-enter zero-command actual-goal
   verification. Keep the full .4m forward / +.3rad yaw task and .06m/.05rad
   final tolerances, with ten observed quiet 100ms intervals and independent
   full-interval native scoring. Do not require indefinite convergence to the
   previous controller's optional .015rad internal margin. Report all motion
   during stopping, not only the endpoint at command cessation.
4. Freeze a total time/pulse budget appropriate to drive-plus-brake execution,
   observed excursion limits and the unchanged native safety stops before
   validation. Do not inherit the prior 35s limit implicitly if the algorithm
   deliberately adds braking after every pulse; a declared new time budget
   does not retroactively change any failed old trial. Record time/energy and
   inefficiency alongside success, rather than hiding the cost of stop-start
   control. Fixed two-second waits are not themselves proof of settling: the
   collection endpoint diagnostic checks only its final 100ms.
5. Run the complete task from several genuinely new nominal starts with one
   frozen controller, plus a retained low-friction challenge. Preserve all
   outcomes. Short forward pulses can move backwards/sideways on low friction;
   nominal progress does not establish that robustness. Do not condition the
   controller on evaluator friction labels or silently omit that result.

## Do not stop at a successful local demonstration

With repeatable local execution in a declared condition, integrate persistent
place/branch memory and actual return-to-place/return-home execution. Validate
complete exploration, backtracking, goal and home missions in novel mazes.
Use sensed place evidence and executed connections, never evaluator topology
or native pose as online state. Open-loop collection or static graph tests do
not prove useful online memory. Preserve camera/terrain assumptions until tested.

Then use matched geometric/persistence, supervised action-prediction and JEPA
representation/prediction arms with equal sensors, gait, action vocabulary,
training data, compute and controller budgets. Separately test predictive
training, genuine online multistep rollout and persistent memory contributions
across independent layouts and training seeds. Predict drive AND stopping
consequences; action sampling and labels must reflect actual slew/latency.
The current 64 events are engineering characterization, not sufficient evidence
for broad predictive generalization or JEPA advantage. Collect task-diverse
sequences prospectively for those studies, with proper custody and fresh tests.

Full-loop wall-clock timing, camera aperture/raster, near-field/body-sweep
observability and sensor calibration remain unresolved. Paused ideal simulation
is not deployment-valid evidence. Seek bounded hardware evidence only when
actual access permits; the full scientific goal remains active until supported.
