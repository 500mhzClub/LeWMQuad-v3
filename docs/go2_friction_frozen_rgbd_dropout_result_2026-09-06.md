# Visual motion remained available during contact-odometry dropout

The frozen joint rigid RGB-D and gyro-conditioned RGB-D estimators each produced
all 226 observations in both audited friction recordings, with no terminal
failure or reinitialization. The two estimators made 904 predictions total.
No model parameters, keyframe thresholds, commands or sensor gates changed.
Both conditions' predictions were saved before native/support scoring.

This closes the immediate question from the contact challenge: on these tapes,
loss of contact odometry did not imply loss of visual motion estimation. It does
not establish a fused controller, safe terrain, hardware transfer or JEPA benefit.

## Availability and accuracy

All numbers below are native-evaluated simulation diagnostics, not physical
error bounds. “Joint” means joint rigid rotation/translation estimation, not a
learned model or joint-angle fusion.

| Measurement | Nominal joint | Nominal gyro | Low-friction joint | Low-friction gyro |
| --- | ---: | ---: | ---: | ---: |
| Visual observations available / 226 | 226 | 226 | 226 | 226 |
| Contact-dropout camera intervals / 225 | 4 | 4 | 184 | 184 |
| Visual intervals unavailable within dropout | 0 | 0 | 0 | 0 |
| Mean 100 ms displacement error during dropout (mm) | 0.232 | 0.572 | 0.257 | 0.562 |
| Maximum 100 ms displacement error during dropout (mm) | 0.306 | 0.738 | 1.880 | 1.690 |
| Maximum position drift over the full recording (mm) | 4.688 | 4.456 | 8.954 | 5.213 |
| Maximum orientation error over the full recording (mrad) | 1.523 | 0.577 | 2.445 | 0.490 |

The contact diagnostic has 2 missing samples in 2 spans nominally and 815 missing
samples in 67 spans under lower friction. Every span is retained and matched to
camera intervals in `evaluation.json`. Each interval uses six endpoint-inclusive
50 Hz contact observations; a missing shared endpoint can affect two intervals.
All 120 forward and 30 left-turn intervals in the low-friction recording contain
at least one missing contact estimate. Both visual estimators remain available.

Joint RGB-D has lower average incremental translation error, but gyro-conditioned
RGB-D has lower maximum accumulated position/orientation drift. Those are
different metrics; neither result establishes universal superiority or learned
prediction benefit. Do not select weights using these exposed outcomes and then
call the same recordings validation.

## Image evidence and checks

The nominal recording has 156,538–180,573 valid depth pixels per frame; the
lower-friction recording has 158,345–183,788. Minimum accepted inlier counts are
92 for each nominal model, and 84 joint / 85 gyro under lower friction. Accepted
correspondences span at least six grid cells. These are algorithmic support
diagnostics, not proof of correct feature identity or calibrated covariance.

The separately bound audit checks raw depth lifting for 136,107 accepted inlier
pairs, independent quaternion-eigenvector rigid fitting, gyro/keyframe pose
composition, native scores using a separate rotation implementation, dropout
membership, all phase/availability strata and every missing-sample span.
Maximum independent fit-coordinate disagreement is 3.997e-15; maximum native
score disagreement is 1.572e-13. Audit status: PASS. Estimators were not rerun.
The audit does not independently prove feature associations, contact physics or
sensor uncertainty.

The focused suite passed 49 tests. Full regression passed 2,265 tests across
181 explicit files in 184.25 seconds. Replay runtime averaged about 45–48 ms per observer,
excluding dataset reading and downstream control; observed maxima reached
153 ms and 223 ms. Regression overlapped replay, so these are neither isolated
timings nor a full-loop real-time qualification.

## Limits that still matter

- The nominal dynamics through 14.5 seconds exactly repeat earlier fitting data.
  The lower-friction arm is a single controlled intervention, not independent
  layouts, training seeds or deployment validation.
- RGB-D uses the existing hidden-robot forward-camera compatibility renderer.
  Camera aperture, clipping and prior raster discrepancies remain unresolved.
- Gyro and depth are ideal simulated channels; foot loads are hypothetical
  three-axis transducers, not calibrated stock-Go2 foot-force counts.
- Accurate 10 Hz endpoints do not establish continuous motion, safe future
  footfalls, braking sweep, loaded support or terrain clearance between frames.
- No learned high-level policy, JEPA advantage, complete novel-maze task, useful
  online memory, genuine multistep benefit or hardware result was achieved here.

## Decision and next action

Follow [visual-led motion integration](go2_visual_led_motion_integration_next_steps_2026-09-06.md).
Do not require continuous contact-odometry availability as a precondition for
visual pose updates. Keep missing contact explicit and avoid adding unsupported
contact weights merely to call the system multimodal. RGB-D plus gyro already
provides a multimodal baseline; extra sensing must earn its contribution.

## Frozen artifacts

Output: `.generated/go2_friction_frozen_rgbd_dropout_v1_attempt_001`.
Launch binds 570 source paths and 12,789 inputs, with inherited native identities.
The audit has an additional explicit source binding. No source export occurred.

| File | SHA-256 |
| --- | --- |
| launch.json | 3642c3e4ac6ccc6f11113d1306c4cf03bf36be1d2ec2b834e97fe387a6267647 |
| predictions.json | b359514632e07f90ae072c8b2978bbab0ac3a916ae2d17066559aa28d9e479a1 |
| evaluation.json | 7a724cc2ebe8cb797d3cc44d1daad418e7d83763e96734fc8c75bdf999b08c9b |
| result.json | fd3bb8615f0c5518369664de74220196dfa9ce716f91442d76489b469395f2a3 |
| pose_dropout_audit_launch.json | 9c0da125b3dc4c3095e0d3e5784ce85d42ba7d9564b5ca866e3f64b306154195 |
| pose_dropout_audit.json | 4ce1d13f19bd35016a4fc2166b197a9a3745365e021503669400d24531682bd8 |
