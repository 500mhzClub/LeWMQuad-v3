# Course-aware control and response fitting: useful negatives, no task success

Two pieces of work completed: a small supervised sensor-only response baseline,
and actual course-aware closed-loop control at a genuinely new physical start.
Both new control trials failed their complete target sequence. The prior V1
failures remain unchanged; neither the fit nor the new controller is JEPA.

## Supervised response baseline

The fit uses 195 nominal and 141 low-friction transitions from the already
exposed heading-servo recordings. Features are proposed forward/yaw command,
their product and past visual motion over 0.5s. Targets are next-interval visual
body displacement and yaw change divided by 0.1s. Native pose and simulator
friction labels are not features or targets. These are development recordings,
not independent validation. The fitted model does not control the robot.

| Train → evaluate | Fitted planar-velocity RMSE (m/s) | Past-motion persistence RMSE (m/s) | Fitted yaw-rate RMSE (rad/s) | Persistence yaw RMSE (rad/s) |
| --- | ---: | ---: | ---: | ---: |
| Nominal → nominal, in-sample | 0.01541 | 0.02078 | 0.03015 | 0.07964 |
| Low friction → low friction, in-sample | 0.01591 | 0.01681 | 0.03671 | 0.04776 |
| Nominal → low friction | 0.04256 | 0.01681 | 0.04726 | 0.04776 |
| Low friction → nominal | 0.02882 | 0.02078 | 0.15681 | 0.07964 |

Both standardized feature matrices have rank 7; their action submatrices have
rank 4. Full algebraic rank is not causal identification: actions were generated
by feedback, conditions have different state/phase coverage, and no independently
randomized counterfactual intervention was fitted. Cross-condition performance
does not justify deploying this model. A separate augmented-SVD least-squares
calculation reproduces the ridge solution within 7.55e-12 per coefficient.

## Course-aware physical experiment

The new estimator uses six causal poses to measure course independently from
body heading. It explicitly reports incomplete/low-speed windows. Transport of
course/body-yaw offset assumes slowly varying slip; it is not a validated
instantaneous motion model. The controller removes V1's persistent minimum
forward speed, caps forward command at 0.12m/s and uses a predeclared 35s limit.
Targets, final tolerances, excursion stops and zero-command failure drain remain.

The actual new start (-0.55,-0.20,0.375)m, yaw -0.08rad was verified, and both
startup pose traces differ from the old fitting and heading-servo traces. Both
friction conditions share the new start and geometry. Controller, speed limit,
time budget and starting conditions changed together: this is not a controlled
ablation proving a course-feedback benefit.

| Outcome | Nominal friction | Lower friction |
| --- | ---: | ---: |
| Complete target-sequence success | No | No |
| Failure | Final target/brake drift | Time limit in forward stage |
| Sensor-selected decisions | 137 | 351 |
| Command ticks, including failure drain | 146 | 360 |
| Physics samples | 8,050 | 18,750 |
| RGB-D captures | 147 | 361 |
| Total simulated duration (s) | 16.1 | 37.5 |
| Native physical stop | None | None |
| Final native planar error (m) | 0.01342 | 0.09814 |
| Final native yaw error (rad) | 0.06015 | 0.75463 |
| Maximum visual position error at decisions (m) | 0.00325 | 0.00476 |
| Final 200ms maximum native speed (m/s) | 0.000418 | 0.002129 |

Nominally, forward braking began at 10.4s and the final brake at 14.1s. The final
brake moved only 1.20mm, but yaw drifted +0.02114rad during its one-second quiet
window. At the controller's final decision, visual yaw error was 0.050885rad,
just beyond the unchanged 0.050rad tolerance. It correctly declared failure.
The subsequent fixed failure drain added about 0.00925rad of native yaw drift;
the table reports the end of the full retained trace. Do not widen the tolerance
or report nominal success because the miss was small.

Low friction never reached the intended forward-arrival/brake/turn sequence.
Its course estimates often changed as forward commands were reduced and yaw
commands reversed. A course measured while a turn causes translation need not
predict the direction of a subsequent forward command. The half-second delay,
action dependence and low-speed switching are plausible contributors, not an
independently isolated causal diagnosis. Small course-fit residuals do not prove
predictive validity.

## Verification and limits

Raw acquisition/command audit PASS for both trials: all 488 sensor-interface-
controller decisions replay exactly, and all 508 RGB-D depth checks meet the
existing diagnostic criterion. Gains, actual robot+floor friction/ratios,
contact/clock/command application, native guards, changed spawn and target/brake
scores were checked. No launched source, result or protocol was changed.

Final focused tests: 18 passed. Full regression: 2,303 passed across 185 explicit
files in 187.75s. Regression overlapped collection. Physics pauses during capture
and inference; no real-time claim. The existing URDF warnings, ideal sensor
assumptions and hidden-robot forward-camera aperture/raster limitations remain.

This is still a controlled continuous-floor/ideal-camera engineering stage, not
sensor-qualified novel-maze navigation. No JEPA contribution, useful online-memory
result, genuine multistep benefit or hardware evidence has been achieved. The
full objective remains active. Follow the
[action-conditioned stopping/holding plan](go2_action_conditioned_stopping_next_steps_2026-09-06.md).

## Frozen artifacts

Response output: `.generated/go2_visual_course_response_v1_attempt_001`;
588 source paths / 13,964 inputs plus inherited native identities.
Control output: `.generated/go2_course_aware_visual_servo_v1_attempt_001`;
594 source paths / 13,968 inputs plus inherited native identities, separate raw
audit manifest. These are normal new implementation files, not a source export.

| Artifact | SHA-256 |
| --- | --- |
| Response launch.json | 7453054ac7af37ef13c5b7bcbf35128bf41fca7ec79b06925bbcd5a7843d69e8 |
| Response result.json | aa7ebf98e145a65cb4564c9bf318c0bc05ea733a12810e88d781e08edf767eed |
| Control launch.json | d6db89d7ba1c473b5f7fac46011ea7364291f101676d61f278ab90b08895808b |
| Control result.json | be1dd7c12be514dadb268d0ad52fc05410bc368afa2f8a3fee931f7477ad64fe |
| raw_servo_audit_launch.json | 16472c5927371313b8ec0bee0d5d4e14a766c34f8377477158f9248f25824838 |
| raw_servo_audit.json | 3eba40b1298cf14644203d4ab452f4338f6bf774d89d57931d9c72e25f3fde60 |
