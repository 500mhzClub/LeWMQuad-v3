# Validation B: observer stop, mixed response-model evidence

The fixed B run executed 24 of 28 scheduled targets, then stopped at 5.3 s
because the continuous observer exhausted its unchanged position-error proxy
budget. Independent raw replay reproduced every decision and the failure.
The three actual zero-command tail ticks ran through 5.6 s. This is a failed
bounded validation execution, not a completed schedule or a new maze success.
Full discovery/return remains 0/2; JEPA advantage remains unestablished.

## What ran and what was verified

The A-only fit was completed and frozen before B implementation: 28 sensor-target
transitions, 300 affine/ridge coefficients, no exclusions, no simulator-pose or
contact training labels. Body and per-joint design matrices have full column
rank (11 and 7), but the data contain only nonnegative yaw commands. B's negative
yaw is therefore an extrapolation challenge; full rank is not adequate coverage.
In-sample normalized body RMSE 0.06664 is not validation accuracy.

B retained the predetermined command schedule, unchanged learned low-level gait,
startup, same continuous observer/memory, calibration-region conditions and
native guards. The three predictors saw identical sensor/action inputs and did
not influence command selection. Acquisition recorded 2,800 physics samples,
42 RGB-D observations and 39 decisions, with every expected artifact present.
No disallowed contact, non-foot ground contact, instability, speed-cap or padded
region violation was found by raw audit. Final-window maximum linear/angular
speeds were 0.02145 m/s and 0.03169 rad/s. The final three zero ticks still moved
the base 1.177 mm. No controller restart or retry occurred.

The two forward segments displaced the base 24.288 and 18.676 mm; net
post-handoff displacement through the tail was 23.918 mm. Maximum active base
speed was 0.06972 m/s. These evaluator-only measurements also demonstrate that
command targets are not measured body velocities.

`physical_stop_reason` and `sensor_stop_reason` are null in the collector result,
because this failure was a returned, fault-latched controller state, not a thrown
collector exception. The authoritative terminal states are `FAILED_OBSERVER`
and `FAILED_HANDOFF_SENSOR_OR_CONDITION`. Null exception fields do not mean a
successful run. The independent audit's completion flags are false.

## The motion model improved body translation, not all articulated errors

All three models have 96 fully recorded, actually executed forecasts: 24
overlapping windows at each horizon. The terminal zero tail matches the zero
commands forecast near the end. There are no truncated/changed-command cases
among these issued forecasts, but four scheduled decisions were never reached.
This is a stopped-trajectory comparison, not 96 independent trials or a completed
B protocol. Do not impute forecasts at the missing decisions.

At 0.4 s:

| Predictor | Mean/max body-position error | Mean/max worst primitive-centre error | Max primitive-point error upper bound |
| --- | ---: | ---: | ---: |
| Ideal body + joint-velocity persistence | 14.46 / 25.70 mm | 38.47 / 109.05 mm | 117.55 mm |
| Ideal body + joint-position persistence | 14.46 / 25.70 mm | 26.91 / 43.50 mm | 46.65 mm |
| Frozen sensor-supervised response | 10.16 / 17.99 mm | 28.61 / 52.46 mm | 55.95 mm |

The learned response improves body translation at every scored horizon. At
0.1 s its mean/max body error is 1.86/4.61 mm versus 4.27/8.50 mm for the ideal
body baseline. Its 0.4-s mean/max joint error is 0.0632/0.1318 rad versus
0.0679/0.1728 rad for fixed posture and 0.1100/0.6273 rad for qdot persistence.
Nevertheless, fixed posture has lower mean worst primitive-centre error at all
four horizons. The learned response exceeds the illustrative 50-mm centre error
at one 0.3-s and two 0.4-s forecasts. Fixed posture stays below 50 mm here but
exceeded it on A, so neither gains a universal 50-mm error guarantee.

Keep both simple baselines and the frozen response. Do not select another fit
using B and relabel its B score independent validation. This supervised model is
not JEPA, a learned high-level policy, or evidence that predictive lookahead
improves navigation: these forecasts did not choose the commands.

## Why the observer stopped

Depth translation was rank 3 through 4.0 s, then rank 2 from 4.1 s onward after
the negative turn. At 5.2 s the combined position scale was 76.967 mm. At 5.3 s,
after 1.3 consecutive weak seconds, a fresh offline reconstruction gives
73.523 mm inherited proxy plus 12.000 mm transported initial-velocity allowance
= 85.523 mm, above the fixed 80-mm budget. The weak direction in the previous
body frame was approximately [0.1234, 0.9915, 0.0410].

Evaluator-only actual position error at that instant was 8.331 mm. The proxy is
explicitly uncalibrated: this one small actual error neither invalidates the
stop rule nor certifies a larger threshold. The learned response did not cause
the stop and must not be substituted for a missing depth observation. Restoring
rank also must not erase previously accumulated global-pose uncertainty.

The next causal-estimation question is whether RGB correspondence plus aligned
depth can observe this missing component when plane-normal depth registration
cannot. Reject low-texture, repeated-texture, occlusion, depth-discontinuity and
inconsistent matches; do not manufacture a full-rank observation. Where evidence
is inadequate, the mission controller needs an observation maneuver before its
uncertainty reserve is exhausted. This is more relevant than refitting the same
small response model or running further static floor-mask probes.

## Timing and scope

There are 37 completed loops containing fresh capture and actual command
execution: minimum 179.784, median 234.344, maximum 243.186 ms; all exceed the
100-ms cadence. Mean components are capture/depth observer 62.263 ms, controller
97.057 ms, execution 62.178 ms, other in-loop overhead 2.229 ms. The first packet
was captured before the outer timer for setup admission and is explicitly
excluded. The terminal decision has no command execution and is not counted as
a moving-loop latency result. Entire acquisition took 32.233 s wall time.

These measurements include diagnostic three-model prediction and the inherited
current-posture comparator. They identify implementation cost, not the minimum
possible planner cost. Profile/remove duplication with reference equivalence
tests before claiming real-time performance. Inherited native COM/joint-limit
warnings, contact/support validity, sensor-error calibration and hardware remain
unresolved. No GPU training, sealed access or hardware actuation ran.

## Next work and retained evidence

Follow [the post-B whole-mission plan](go2_post_B_observability_and_whole_mission_plan_2026-09-06.md).
The A/B local response comparison has now yielded its evidence. Preserve it and
address RGB-D motion observability, integrate a fresh complete mission, then
test matched supervised/JEPA, true online multistep rollout and memory effects
on independent layouts/seeds/robustness. Do not equate further local tests with
the final goal.

Output: `.generated/go2_action_motion_validation_development_v1_attempt_001`.
Launch SHA-256 `5f6131be108b1a7ad118108eca7ebecb1dccb5e15daf85e6f391d5c321addae2`;
result `9c2c7810f511e052466368fddf5833d84abb945a2a34db166a64fbc6e3714bbc`;
raw audit `ba9504f5c3c001a3e0f5c55f45f9f847e0f4d1c97806fc3b7041734dd68b63f9`.
The launch binds 420 source paths, 4,479 inputs and 16 native identities, checked
before/after acquisition and raw audit. Launched files/results remain unchanged.

[The terminal-fusion diagnostic](go2_action_motion_validation_development_v1_diagnostic_2026-09-06.json)
replays only through the failed decision, not the failed controller or its tail;
SHA-256 `7efde2de45638b0e36ceaac5fd35d28db510d76805b28c13e6aef0f4718df953`.

Verification: focused 91173 passed 10 response tests; frozen-fit check 25299
passed 415 sources/4,475 inputs; focused 25215 passed 30 tests in 14.47 s;
B preflight 39477 passed with fresh output; regression 21788 passed 1,816 tests
across 147 explicit files in 153.47 s. Acquisition 67968 and raw audit 10166
exited 0 (audit reconstructs scientific failure); diagnostic 29414 exited 0.
No tested/launched source edits occurred during those executions.
