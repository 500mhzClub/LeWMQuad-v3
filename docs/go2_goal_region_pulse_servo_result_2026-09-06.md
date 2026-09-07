# Goal-region pulse V1: three nominal local successes, friction failure retained

The fixed move/turn/hold task now succeeds in all three new nominal-start
simulations, including independent checks throughout the final one-second hold.
The low-friction challenge fails. This is a bounded local engineering result,
not broad reliability, novel-maze navigation, a learned high-level policy or a
JEPA contribution.

Output: `.generated/go2_goal_region_pulse_servo_v1_attempt_001`.
Protocol: [goal-region pulse V1](go2_goal_region_pulse_servo_v1_2026-09-06.md).
Next: [continuous execution and online memory](go2_local_execution_to_online_memory_next_steps_2026-09-06.md).

## Complete-task evidence

The task requires initial-body XY(.4,0)m and yaw+.3rad, with final planar error
<=.06m and yaw error<=.05rad, plus ten new quiet zero-command intervals. Native
evaluation checks all501 poses during that second, speed<=.02m/s and yaw
rate<=.05rad/s. All criteria were frozen before these trials.

| Trial | Full task | Pulses | Total simulated time | Worst hold position error | Worst hold yaw error |
|---|---|---:|---:|---:|---:|
| nominal_a | Pass | 10 | 27.7 s | .039379 m | .038566 rad |
| nominal_b | Pass | 10 | 27.7 s | .040682 m | .037719 rad |
| nominal_c | Pass | 10 | 27.7 s | .040350 m | .016328 rad |
| lower_friction_a | Fail: visual pose unavailable | 3 | 9.9 s | No completed hold | No completed hold |

Maximum speeds during successful holds are .002915/.003031/.002059m/s;
maximum yaw rates are .006716/.006863/.003272rad/s. Nominal native path lengths
are .511718/.510223/.508911m. Each nominal episode uses262command ticks,
263RGB-D observations and13,850physics samples; 1.5s of its27.7s is setup.
This stop-start controller is slow; repeatability here is not efficient control.

The low-friction episode loses current visual evidence at8.9s and retains its
guarded1s zero-command drain. It ends with native position error .383418m and
yaw error .691515rad. No native contact/speed/domain stop occurs in any trial.
All four outcomes remain in the denominator. The stage change does not solve
low-friction sideways motion or robust tracking during larger maze turns.

## What was fixed, and what is not inferred

The [predecessor](go2_pulse_feedback_servo_result_2026-09-06.md) recorded 0/4
successes because of visual failures or pulse exhaustion. Its internal2.5cm approach threshold
could demand further steering even inside the actual6cm goal region. This
successor selects final orientation/hold whenever inside that actual region.
It preserves the gait, pulse sizes, braking, final tolerances, safety stops,
failure latch and total time/pulse limits. No visual estimator reset or fitting.

New world starts are(-.62,-.28,.015),(-.52,.16,.065),(-.72,.26,-.045) in x/y/yaw,
plus matching first-start friction1/.15 challenge. Physics seeds2026090671..673;
appearance2026090675..677. Actual spawn and changed750sample prefixes were
verified. These nearby deterministic development starts share one scene and
similar responses; they are not three independent layouts or a statistically
established reliability estimate. Different predecessor starts also prevent
claiming a matched causal improvement rate from0/4 to3/4.

## Audit and software verification

Collection53454 and raw audit43515 both terminated exit0. The audit replays all
864sensor/controller decisions exactly, checks all874RGB-D observations, and
verifies commands/slew, clocks, native contacts/guards, friction, gains and starts.
Total870command ticks,46,500physics samples and93.0simulated seconds. No expected
artifacts are absent. Closure:626sources/28,316inputs plus inherited native
bindings, frozen before collection. All launch/result/audit bindings rechecked.

Focused40tests passed (8811,1.97s). Full2,388tests across190 explicitly enumerated
files passed (41518,196.58s). These counts are software evidence only.

A separate normalized-Hamilton-quaternion computation reproduces all four
native endpoints and all three full-hold scores (52191,exit0). An initial check
(13026) omitted quaternion normalization that SciPy applies, producing a
5.3e-10m discrepancy against its1e-12m comparison threshold. Correcting that
checker convention, without changing data or tolerances, passes; raw quaternion
norm errors reach approximately1.17e-7. This is not a physical rerun or a change
to sensor estimates. The first check had already verified all artifact bindings.

- launch.json: `77a038b8515b15a412c05722342168b6030549c47b77e09b012bc31d0856cf3b`
- result.json: `05f8ba6785e23151fb69e988c71f11ff859b0e4f055b8f213cba02e31443dcdf`
- raw_servo_audit_launch.json: `a31bac087c22da2c8d1f1be140561431eae6523e0a5e5775634068dc0d65ae64`
- raw_servo_audit.json: `27e43fae55c2c71a5398adbbff5d6061f4803ed6f6bde93617a2486803ddfb6b`

## Remaining scientific requirements

This hand-written high-level controller uses classical RGB-D/gyro pose and an
existing learned locomotion gait. It does not use a JEPA predictor. Continuous
multi-leg execution, useful persistent memory, observed place/branch/home
association, complete independent-layout maze missions, matched predictive-
training/online-rollout/memory comparisons and multiple training seeds remain.

Controlled level floor, hidden-robot ideal RGB-D, ideal gyro and paused physics
during compute remain explicit assumptions. Camera/body-sweep observability,
sensor availability/calibration, real-time timing, terrain robustness and
bounded hardware evidence are not established. Torque/power was not recorded;
mechanical energy is unavailable. The full scientific goal remains active.
