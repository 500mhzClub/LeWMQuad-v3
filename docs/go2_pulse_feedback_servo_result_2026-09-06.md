# Pulse-feedback V1: audited negative result

All four complete local tasks failed. The raw audit passed; passing acquisition
and replay checks does not turn the failures into navigation successes.

Output: `.generated/go2_pulse_feedback_servo_v1_attempt_001`.
Protocol: [fixed pulse-feedback trial](go2_pulse_feedback_servo_v1_2026-09-06.md).

| Trial | Terminal cause | Pulses | Simulated duration, including failure drain | Native final position/yaw error |
|---|---|---:|---:|---|
| nominal_a | Visual pose unavailable | 18 | 45.2 s | .01277 m / 1.00398 rad |
| nominal_b | Pulse limit | 35 | 87.2 s | .01999 m / .07942 rad |
| nominal_c | Visual pose unavailable | 18 | 44.5 s | .02605 m / 1.19119 rad |
| lower_friction_a | Visual pose unavailable | 6 | 17.3 s | .37613 m / 1.36139 rad |

No native stop occurred. Total: 1,846 controller decisions, 1,882 command ticks,
97,100 physics samples and 1,886 RGB-D observations, representing194.2 simulated
seconds. All decisions replay exactly; all1,886 raw depth checks pass the1mm
diagnostic. Actual changed starts, clocks, commands/slew, contacts, material and
gains pass their checks. Maximum visual position errors over available estimates
are .004317/.008323/.006905/.003351m respectively. These maxima exclude missing
estimates and are descriptive, not calibrated uncertainty bounds.

## What changed the next action

The controller's internal approach target was .025m although the actual task
requires .06m final position error. Nominal_b at tick250 already had visual
position error .042116m and yaw error .032199rad, both inside the real criteria,
yet remained in approach. It subsequently steered unnecessarily and exhausted
its pulse budget. Briefly meeting pose criteria is not a verified final hold.

This is an implementation/design error, not evidence the task is physically
infeasible. The distinct goal-region successor uses the actual .06m position
region to choose orientation/hold while preserving .05rad yaw and every final
hold/stop criterion. Its separate fixed fresh-start protocol is
[goal-region pulse V1](go2_goal_region_pulse_servo_v1_2026-09-06.md).

The three sensor failures retain their terminal consensus/grid/displacement
rejection. No particular one of those combined predicates is isolated by the
saved exception alone. Removing unnecessary steering may reduce exposure, but
does not establish robust tracking during maze turns. No sensor-gate relaxation
or post-failure estimator reset was applied.

## Reproducibility and scope

Audit process39164 terminated exit0. Post-audit process33182 verified all launch,
result artifact and audit-input/source bindings. Predecessor sources were not
edited. Its prior full regression was2,366 tests; the successor-inclusive suite
now passes2,388 tests across190 explicit files (41518, exit0,196.58s).

- launch.json: `f29a6309ca592248f07de6eae42d58c26a254ffe627abf5a9fd36c1ff6f9d786`
- result.json: `3812126265bec1fb770bcf82d752636385ee6b88475cb9468b5080fdd4903992`
- raw_servo_audit_launch.json: `a3b3dab216b94979d2b32a98b5dd5cf27abc334d55f44a1cd273e21777c6575c`
- raw_servo_audit.json: `840f9f4d916e6c0c48cd42d8de9675c1dd4b3d6ad853c7277846ff31ba98d58f`

This remains engineered high-level control with a learned low-level gait on a
continuous level floor, using hidden-robot ideal RGB-D and ideal gyro. Physics
pauses during computation. It establishes neither JEPA benefit, useful maze
memory, deployment sensor validity, real-time operation nor hardware success.
