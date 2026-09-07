# Joint RGB-D rigid-pose development result

The ten-case fitting-only replay and its reference audit are complete. Joint
RGB-D rotation/translation preserves tracking through the recorded gyro-bias
conditions, but does not improve nominal accuracy over gyro-conditioned fitting.
This is geometric state estimation, not learned navigation or a JEPA result.

| Input condition | Joint admitted / 336 | Gyro admitted / 336 | Joint max position error | Gyro max position error |
| --- | ---: | ---: | ---: | ---: |
| Nominal | 336 | 336 | 6.904 mm | 5.340 mm |
| Gyro z +.001 rad/s | 336 | 275 | 6.904 mm | 88.021 mm |
| Gyro z -.001 rad/s | 336 | 336 | 6.904 mm | 105.863 mm |
| Blank RGB | 1 | 1 | Anchor only | Anchor only |
| Existing depth-noise member | 202 | 202 | 4.240 mm | 4.113 mm |

Errors above cover each model's admitted history. The matched positive-bias
comparison uses 275 common frames; its joint maximum is also 6.904 mm. Joint
nominal maximum orientation error is .00214444 rad versus .000624686 rad for
nominal gyro. Joint uses 33 keyframes including the initial anchor. Both blank
members reject frame 1; depth-noise members reject frame 202 because a perturbed
pixel marked valid exceeds the 5-m range. The malformed input remains unchanged,
without clipping, restart or replacement. Gyro positive-bias rejects frame 275
on the registration consensus/support/displacement gate.

The robust proposal/inlier procedure is shared between joint and gyro modes.
Joint RGB-D estimates rotation; gyro is only a disagreement monitor. Exact pose
invariance to the two admitted gyro perturbations is consequently expected by
construction, not evidence that gyro bias was estimated or corrected. Nor is
this evidence of general robustness to scene, calibration or sensor shifts.

## Audit and scope

The independent quaternion-eigen solver reconstructs accepted joint fits to
3.56e-15 maximum transform-entry difference from the production SVD solver.
An independent quaternion/native-pose scorer agrees within 2.41e-13. The audit
checks 2,361 admitted states, 317,801 raw-depth lifted point pairs and 227 actual
reference promotions across all ten cases, along with terminal latches,
reprojection/residual/conditioning gates, parent composition and motion limits.
Forty synthetic planar and volumetric transforms additionally check the audit
solver. Feature correspondence identity and proposal optimality are not
independently proved. Both pose-error bounds remain explicitly unknown.

Pose-only nominal replay timing: median 45.59 ms, sampled p95 48.79 ms, maximum
50.35 ms. This excludes capture, evidence queries, planning and actuation and is
not a real-time whole-loop qualification. No validation images were loaded by
this replay, and no physics or navigation occurred.

The 16 focused estimator tests passed. Subsequent combined estimator/new-collection
tests passed 33/33. A fresh 172-file regression passed all 2,156 tests in 188.31 s;
eight additional longer-acquisition audit tests passed separately. This does not
infer a result for the earlier unavailable regression handle.

## Bound artifacts

Root: `.generated/go2_joint_rgbd_rigid_pose_development_v1_attempt_001`.

| Artifact | SHA-256 |
| --- | --- |
| launch.json | 6b4c5b9d64b021549e5707f6dfc80cb17f531f23ab23becd9e655fc55c0531ae |
| result.json | 77b3aef4d941eccb94947a10f46707b2efa18d1376e30fb1a7ebd01b974ae8b5 |
| predictions.json | ebde3b6fe74a6c15f22ac3acfc24d7a89bf9c706abd3955a4b4be4c3c27643f1 |
| evaluation.json | 11883222a9850fb72953868c9063810aaac020170c53b3fc81c29a43f1d0a97f |
| rigid_pose_reference_audit.json | 28bac2e0e69a5b2866670fbf72db12a3150344f79474dda6c953dd206756a5f9 |

The original launch binds 517 sources and 7,710 inputs plus native/OpenCV;
the reference audit additionally binds its own source and input artifacts.
Launched source, protocol and output identities remain immutable.

## Next scientific action

Move to [fresh longer physical collection](go2_longer_observed_floor_motion_development_v1_2026-09-06.md)
with frozen nominal joint/gyro comparison before new validation exposure. Check
actual full-body floor coverage, turn/brake data and transfer of pose tracking.
Do not select another estimator on the same fitting tape or convert its maximum
error into a clearance allowance. Then establish independently validated
relative body/surface uncertainty and prospective actuation; complete the
[mission and JEPA plan](go2_floor_factored_navigation_next_steps_2026-09-06.md).
Reliable navigation, memory benefit, JEPA/multistep benefit, independent maze/seed
evidence, full-loop timing and hardware remain unproved.
