# Visual-led motion interface: implemented and replayed

`VisualLedMotion` now wraps the unchanged joint-rigid and gyro-conditioned RGB-D
estimators. Optional contact diagnostics cannot update, erase or authorize the
visual pose. The interface explicitly separates measured pose, missing contact,
historical observations and motion permission.

## What is implemented

- Explicit episode and calibration identities, observation clocks, keyframe
  provenance, image/depth identities and unknown physical error bounds.
- Typed ideal-vector contact velocity samples with source identity, clocks,
  rotation and calibration checks. These are not stock-Go2 raw force counts.
- Six-sample contact integration in a fixed gyro anchor, compared with visual
  displacement in the current body frame. Missing values remain None;
  malformed optional windows produce a contact-specific rejection.
- Latched visual failure without reinitialization. Current pose is supplied only
  at its image timestamp; later queries receive a historical pose, not assumed
  zero motion. Returned snapshots do not alias mutable internal state.
- No command-integrated position, contact fusion weight, covariance fitted from
  feature scatter, slip exclusion or motion/terrain permission.

## Executed integration evidence

The interface processed the original RGB-D/body/fast-gyro packets in both exposed
friction recordings, with their previously acquired support diagnostic stream.
All 904 visual output/provenance records exactly match the frozen visual-only
witness. This is integration verification, not another independent accuracy
comparison or a new physical trial.

| Per-model observations | Nominal | Lower friction |
| --- | ---: | ---: |
| Exact visual output records | 226 | 226 |
| Contact comparisons available | 221 | 41 |
| Contact comparisons unavailable | 4 | 184 |
| Initial frame without a prior comparison interval | 1 | 1 |
| Intermediate queries explicitly lacking current pose | 900 | 900 |

Across both models and conditions, all 3,600 intermediate 20ms queries preserve
historical provenance and report current translation AND orientation unobserved.
This version does not asynchronously integrate gyro between camera updates.
It supports a 10Hz navigation boundary, not a 50Hz current-pose controller.

A separate read-only scalar long-double calculation checked all 524 available
contact/visual comparisons and all 376 missing comparisons. Maximum disagreement
with the implementation was 2.033e-18 m. This checks arithmetic and missingness,
not contact physics, feature associations or sensor uncertainty.

The final focused suite passed 46 tests, including 20 new interface tests.
Full regression passed 2,285 tests across 182 explicit files in 185.50 seconds.
Tests covered optional-channel dropout, delayed/stale/misaligned identities,
visual terminal faults, clock ordering, fixed-frame integration and snapshot
isolation. No launched source or original estimator was changed during replay.

## Important remaining limits

Publication time is the simulation decision clock; processing latency is
explicitly unaccounted for. Image acquisition and compute latency still need
separate treatment before asynchronous or real-time control. The contact gyro
anchor is an upstream acquisition convention, not independently calibrated
hardware orientation. Invalid contact construction must be handled at that
acquisition boundary; raw counts cannot masquerade as a valid typed sample.

The interface has not yet chosen any physical command. It does not solve current
support, future landing terrain, body/leg sweep, braking response or camera
aperture validity. Existing forward RGB-D remains an idealized hidden-robot
compatibility channel. There is no JEPA contribution, completed maze/memory task
or new hardware evidence here. The full scientific goal remains active.

## Next

Follow the [bounded closed-loop visual-control plan](go2_bounded_visual_servo_next_steps_2026-09-06.md).
Use the interface in actual simulated control, with a clear distinction between
a conditional controlled-floor/ideal-camera engineering experiment and a
sensor-qualified novel-maze claim. Do not require perfect formal terrain proof
to test a labeled simulation stage, and do not promote that stage as evidence
that unseen real terrain is safe. Resolve optical and sensing limitations on
the path to the latter claim.

## Artifacts

Output: `.generated/go2_visual_led_motion_integration_v1_attempt_001`.
Launch binds 575 source paths and 12,795 inputs plus inherited native identities.
No source export, training, protected-data access or new physical execution.

| File | SHA-256 |
| --- | --- |
| launch.json | 3e7903e671ded08b0b9725d6b2cafde9c53e29469dd8452e197de412d2916e8a |
| motion_evidence.json | d995275d4411622b0b57fceed651bd09a9da3a531b03b277eee956b82f5950d2 |
| result.json | 8d4602353daebf14dfa52e0d84106bd01e104d29481526e538d3fafd0a4b3ceb |
