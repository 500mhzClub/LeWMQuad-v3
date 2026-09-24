# Closed-loop visual servo executed: both tasks failed, acquisition audit passed

The controller now chooses actual simulated Go2 commands from current RGB-D/gyro
pose evidence. This is a meaningful integration step beyond replay. However,
neither of the two fixed target sequences completed: 0/2 engineering successes.
Do not relabel movement, clean stops or passing tests as target completion.

The high-level controller is engineered; the locomotion policy is pretrained.
No JEPA training or learned high-level navigation contribution was tested here.

## Fixed experiment and outcomes

The task was 0.4m forward in the initial observation frame, settle, turn to
0.3rad, then settle. Targets and gains were frozen before execution. Native
state supplied acquisition, startup checks, external stops and later evaluation,
not commands. The only prelude was 15 zero-command settling ticks.

| Outcome | Nominal friction | Lower friction |
| --- | ---: | ---: |
| Controller result | Time limit during turn | Forward excursion limit |
| Control decisions | 201 | 147 |
| Executed command ticks, including failure drain | 210 | 156 |
| Recorded physics samples | 11,250 | 8,550 |
| RGB-D captures | 211 | 157 |
| Total simulated duration, including settling/drain (s) | 22.5 | 17.1 |
| Native physical stop | None | None |
| Final native planar target error (m) | 0.01992 | 0.10166 |
| Final native yaw (rad), target +0.3 | -0.35755 | -2.10330 |
| Final native yaw error (rad) | 0.65755 | 2.40330 |
| Maximum visual position error at decisions (m) | 0.00707 | 0.00460 |
| Final 200ms maximum native speed (m/s) | 0.01004 | 0.00237 |

After each controller failure, the fixed one-second zero-command drain executed
without a native stop. This is recorded stopping behavior, not recovery or
task success. No retry, threshold change or replacement of either result.

The actual first physical position/heading match the new spawn
(-0.65m, +0.15m, +0.12rad), within the preregistered numerical checks. Both
750-sample startup pose traces differ from the old fitting prefix. The paired
arms still share one geometry and starting condition; this is not an independent
layout or training-seed generalization result.

## What the failures show

Nominally, forward acquisition took 15.8s. Its visual heading was already about
-0.844rad when the controller entered forward braking. Braking took another
1.6s, leaving only 2.6s before the 20s control deadline. The subsequent turn
made progress but could not reach +0.3rad before timeout. Native forward-brake
drift was 7.10mm; the terminal drain drift was 11.11mm.

Under low friction the forward controller followed a curved trajectory and hit
its x>0.48m observed excursion limit before reaching the planar target region.
It never entered the intended forward-brake/turn stages. Native terminal drain
drift was 14.31mm. Visual error is small relative to the final task error; this
is not explained by a visual false-arrival declaration.

A post-hoc native diagnostic gives the following phase means. These are observed
responses on correlated trajectories, not calibrated coefficients or causal
effects of an independently excited action model:

| Forward-phase mean | Nominal | Lower friction |
| --- | ---: | ---: |
| Requested/applied forward command (m/s) | 0.08095 | 0.08678 |
| Actual body forward velocity (m/s) | 0.02493 | 0.03040 |
| Actual body lateral velocity (m/s) | 0.00589 | 0.03160 |
| Applied yaw command (rad/s) | -0.07819 | -0.14932 |
| Actual body yaw rate (rad/s) | -0.05361 | -0.14353 |

Requested forward speed is not achieved forward speed. More importantly, body
heading is not the course of motion when lateral velocity is substantial. The
controller steers the body toward the target bearing without compensating for
that difference. These observations motivate course/action-response feedback;
they do not prove that one gain change will solve it. Merely extending the old
timeout would leave the low-friction tracking failure unaddressed.

## Verification

Raw audit PASS for both conditions. It checks body/gyro acquisition and clocks,
native contacts, all 368 RGB-D depth checks, command application, stage timing,
material coefficients/ratios, unchanged gains and native guard rows. Every one
of the 348 controller inputs/outputs replays exactly from the acquired sensor
packets. Final pose and braking measurements are independently scored from native
pose/twist. Changed spawn is checked physically, not inferred from the seed.

All 36 focused tests passed. Full regression passed 2,295 tests across 183
explicit files in 188.77 seconds. Tests overlapped collection, and physics paused
during image capture and inference. No real-time claim follows from wall timing.
The simulator's existing URDF inertia and neutral-position warnings remain.

## Scope and next step

This is explicitly a controlled continuous-floor, ideal hidden-robot-camera
simulation stage. It neither clears the camera aperture/raster failures nor
establishes supporting terrain from unseen pixels. There is no new hardware
calibration or execution, complete novel-maze task, persistent-memory benefit,
JEPA advantage or genuine multistep result. The full goal is active/unachieved.

Follow the [course-aware action-response plan](go2_course_aware_visual_control_next_steps_2026-09-06.md).
Use these exposed data for development, freeze the new controller/model before
new physical validation, and keep both V1 failures in every comparison.

## Artifacts

Output: `.generated/go2_bounded_visual_servo_v1_attempt_001`.
Launch: 582 sources and 12,798 inputs plus inherited native identities.
The raw audit has a separately bound source/input manifest. No source export.

| File | SHA-256 |
| --- | --- |
| launch.json | 52275754c7f12902aa120a17252bc8155d23e58b8b983b383086777baf6e1b90 |
| result.json | 84de6410fa19eb489eb34d1399a80f541e0370203db03f7b04bddf3f1f3613cc |
| raw_servo_audit_launch.json | 3821fadb3e1139f5e75edbaf0cec7f3f88fc341af0cd3f6123905aa71a376ee5 |
| raw_servo_audit.json | 291e19821a0bc88608d4dc153061a687af5bcd2012a797198d87db39af2b8f01 |
