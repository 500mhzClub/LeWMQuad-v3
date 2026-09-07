# Friction challenge: physical collection succeeds, contact odometry drops out

Both predeclared 24-second simulation collections completed and passed the raw
acquisition/support audit. Total:24,000physics samples,452RGB-D captures and
2,252live support predictions. No reported native physical stop, shadow fault,
retry, estimator-selected command or controller-gate change occurred.

## Intervention and custody

The zero-step preflight verified actual solver friction on all27robot+1ground
geometries, including all28per-environment multipliers. Nominal coefficient1.0
and lower coefficient0.15 are effective on BOTH sides of foot–ground contact;
the simulator combines their coefficients using maximum. This is not merely
a floor-property change: non-foot robot coefficients also change. Walls and
visual-mesh bytes remain unchanged. All multipliers equal1.0. Each condition
has228coefficient/readback checks through termination. Gait gains are unchanged.

The two arms use common physics seed2026090641 and appearance seed2026090643,
one matched geometry and the same requested tape. They do not represent an
independent-layout or multi-training-seed study. No protected benchmark was used.
The existing URDF inertia/neutral-position warnings remain; this did not validate
the robot's physical inertial parameters or camera aperture.

## Important independence correction

A separately bound post-hoc check found that the nominal physics prefix through
14.5s (7,250samples) is EXACTLY identical to the earlier fitting collection in
every trace field: root pose/twist, q/dq, commands, clocks and phase/contact
fields. A changed seed did not randomize those deterministic dynamics. This is
a new execution and appearance recording, but that nominal prefix is replication,
not fresh physical generalization evidence. The lower-friction intervention and
later changed command sequence remain distinct observations; do not inflate the
number of independent trials or treat the repeated prefix as reserved validation.

## Main result: availability must be reported with error

| Measurement | Nominal | Lower friction |
| --- | ---: | ---: |
| Contact-consensus unavailable /1,126 observations | 2 | 815 |
| Forward consensus unavailable /600 observations | 2 | 541 |
| Forward rolling-model mean error, available observations only (mm/s) | 2.453 | 4.325 |
| Forward rolling-model maximum error, available only (mm/s) | 67.601 | 31.722 |
| Forward derived contact tangential speed, mean (m/s) | 0.01481 | 0.14151 |
| Forward derived contact tangential speed,95th percentile (m/s) | 0.02587 | 0.60579 |

Both fixed hypotheses require at least two feet whose resultant load exceeds5N
for20ms. Under low friction that condition frequently fails. Its smaller
conditional maximum error is NOT an improvement: it excludes most forward
observations. Report availability and matched valid intervals; do not replace
missing velocities by zero, lower the dwell gate post-hoc or restart histories.

The tangential speed diagnostic uses native body/joint kinematics and solver
contact positions, solely after sensor prediction. It is evidence of increased
simulated contact motion, but contact-position timing and physical slip have not
been independently calibrated. The recorded forces are ideal three-axis net
contact transducers, not verified vendor scalar foot sensors.

## Actual motion and braking

| Executed segment measurement | Nominal | Lower friction |
| --- | ---: | ---: |
| Forward net translation (m) | 0.56785 | 0.69869 |
| Forward yaw change (rad) | -0.02279 | -0.33615 |
| Forward-brake net displacement (mm) | 4.877 | 9.002 |
| Left-turn yaw change (rad) | 0.52135 | 0.69240 |
| Right-turn yaw change (rad) | -0.67366 | -0.74514 |
| Right-brake net displacement (mm) | 8.773 | 15.795 |

The same command is not the same physical action. The intervention changes
trajectory, yaw drift and stopping, so command integration cannot replace a
validated action-response model. Successful supervised execution despite missing
contact-odometry consensus also shows that this static-foot diagnostic is not
a general locomotion-admission criterion. It does not prove unsupported steps
safe or justify removing terrain/body-clearance checks.

## Verification

The raw audit reconstructs body/gyro histories, contacts, RGB-D packets, command
timing, applied commands, guards, material records and all live support outputs.
It checked594,527nominal and586,145lower-friction interior depth rays under the
existing hidden-robot rendering contract; no new visual qualification follows.
All96,000foot-force vectors conserve native side-wise force sums; maximum
coordinate errors are2.607e-13 and1.880e-13simulatedN. Both1,126-row support
histories replay exactly. The shared models/URDF remain an independence limit.

All48focused tests passed; full regression passed2,255tests across180explicit
files in183.04s. Tests overlapped collection, so wall timing from this run is not
a real-time qualification. No tested or launched source changed during execution.

## Next action and remaining science

Follow the [missing-contact multimodal integration plan](go2_contact_dropout_fusion_next_steps_2026-09-06.md):
first test frozen RGB-D estimators on these new recordings, explicitly including
the missing-contact intervals, then develop a separately named sensor fusion
and action-response procedure with truly fresh validation. Do not optimize the
already exposed nominal prefix or promote contact-only odometry.

No learned high-level policy, JEPA advantage, complete maze exploration/return,
online-memory benefit, genuine multistep benefit, independent-layout/seed result,
real-time qualification or hardware sensing/execution has been established here.
Camera aperture/clipping and two prior ray discrepancies remain. The full goal
is active and unachieved.

## Artifacts

Preflight: `.generated/go2_support_friction_native_preflight_v1_attempt_001`;
562source paths and11,357inputs. Collection:
`.generated/go2_support_friction_collection_v1_attempt_001`;
564source paths and11,365inputs, plus inherited native identities.
Raw and prefix audits have separate manifests; no source export occurred.

| Collection artifact | SHA-256 |
| --- | --- |
| launch.json | 1602091d4f49713495798cb1ecd354294a164ca0a79cb003217348b6753a2980 |
| result.json | 39afad5afcc546f8017d1920b79cd524566c3c51f12a0ea0b8cc819ced38f175 |
| raw_support_audit_launch.json | a7628861af989a145b9b4d0c69d54464c93254888d57a294c4fec4487ca23198 |
| raw_support_audit.json | 8144a8bdd84dbd5534c7c166a043fd077101c371d44b97036e6337c3a9501673 |
| prefix_identity_audit.json | 08e5a159eff5cf6b9e8edc237d3ee0b8444a5726c7ad3667590b02b4f5caf58f |
