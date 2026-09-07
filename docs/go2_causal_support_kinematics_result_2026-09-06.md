# Causal support kinematics: useful nominal motion, unresolved contact errors

The fitting-only diagnostic completed 2,926 sensor observations from 1.5 to60s.
Both fixed hypotheses use only co-timed ideal foot-load, joint and IMU data.
All predictions were saved before native velocity or frozen RGB-D comparison
was loaded. No physical trial, model training, control gate or hardware changed.

The two hypotheses are stationary foot centre and a rolling spherical foot with
a nominal contact normal along IMU-derived up. The latter is a conditional local
contact model, NOT proof of a level floor between/beyond the current feet.
The 1.3–1.5s gravity initialization uses gyro-transported specific-force samples;
no native orientation is used for prediction. Physical error bounds remain unknown.

## Results against simulator reference

| Segment | Centre mean velocity error (mm/s) | Rolling mean error (mm/s) | Centre / rolling maximum (mm/s) |
| --- | ---: | ---: | ---: |
| Sustained forward | 5.519 | 2.810 | 64.525 /67.601 |
| Forward brake | 2.035 | 0.639 | 8.061 /2.972 |
| Left turn | 4.374 | 3.207 | 14.437 /17.376 |
| Post-turn forward | 3.206 | 1.693 | 12.249 /12.201 |
| Right turn | 2.925 | 2.346 | 37.078 /36.994 |
| Right brake | 1.623 | 1.081 | 23.219 /25.085 |

Mean error improves with the rolling correction, but worst-case error does not
uniformly improve. Do not promote the model or use these maxima as future bounds.
Ten observations have only one persistently loaded foot, so neither model supplies
a consensus. There are 470 observations with two selected feet, 1,536 with three
and 910 with four. These unavailable intervals are retained, not bridged by a
fabricated zero velocity or observer reset.

Post-hoc inspection locates the worst velocity error at4.8s: only FL and RR meet
the fixed dwell rule. Their rolling-model errors are10.77 and130.65mm/s; the mean
has67.60mm/s error and63.43mm/s cross-foot disagreement. The current RR upward
load is9.37simulatedN. This is evidence that persistent nonzero load does not
make a foot-motion hypothesis accurate, not a calibrated slip detector.

Conditional up's maximum error is0.932mrad against native up. Selected-foot height
spread along that direction peaks at1.405mm. Neither small number establishes
terrain continuity, support on an unknown surface, or robustness to acceleration.

## RGB-D consistency, not independent truth

Of585100-ms intervals,571 have complete kinematic consensus windows;14 are
unavailable. Using gyro-only rotations to integrate those predictions, mean
displacement disagreement with frozen joint RGB-D is0.471mm for centre and
0.320mm for rolling. Against gyro RGB-D it is0.506mm and0.351mm. RGB-D does not
feed back into the kinematic predictions. Shared sensors and a ground-only
fitting scene mean this consistency is not independent generalization evidence.

## Audit and regression

The independent audit checked all23,408per-foot hypothesis vectors by directional
numerical differentiation of the URDF forward kinematics, without using the
analytical Jacobians. Maximum prediction difference was4.846e-10m/s under the
fixed2e-7m/s audit tolerance. Separate quaternion integration reconstructs all
gyro-relative rotations within3.176e-14 per coordinate and up within6.017e-15.
Load selection and all available consensus means match. URDF forward kinematics
are shared, so this is a numerical implementation check, not robot calibration.

All42focused tests passed, including17new support-kinematic tests. Full regression
passed2,249tests across179explicit files in180.76s. No tested or launched source
changed during execution. Original unavailable intervals and error peaks remain.

## Next action

Run the [fresh contact-motion challenge preparation](go2_support_motion_challenge_next_steps_2026-09-06.md)
with both hypotheses frozen. Prioritize a nominal control and changed-friction
condition to expose contact-model failure. Do not tune a contact threshold to
this one4.8s outlier and claim success on the same recording.

The ideal three-axis load modality remains a simulation capability hypothesis,
not verified stock-Go2 sensing. Existing camera aperture/clipping and two ray
disagreements remain unresolved. No current support, future landing, body sweep
or stopping predicate has been qualified for autonomous control. Short sensor-only
execution, online branch memory, full maze discovery/backtracking/home return,
matched JEPA/predictive-training/multistep/memory comparisons across independent
layouts/seeds, real-time robustness and bounded hardware evidence remain required.

## Identities

Root: `.generated/go2_causal_support_kinematics_development_v1_attempt_001`.
Launch binds555source paths and11,351inputs plus inherited native identities;
the derivative/up audit is separately bound. No source export or sealed access.

| Artifact | SHA-256 |
| --- | --- |
| launch.json | 88b6cdbf38cf20b6d2af3537474fe2e1e1f75b7221e7cbcb10d66b558b8f4132 |
| result.json | 957dbafe351941c4e78f8be12a07778381526e607d99bc1fa52a444d484763f1 |
| predictions.json | 4b64b842d6c412285b4c0f08372dba59f984dbb43883371ca47f52011793fa23 |
| derivative_up_audit_launch.json | 968fcc8b51619e8a54a5ee77290c3dc811de72b932460a444115895851f57fb0 |
| derivative_up_audit.json | 98a32092377497091fd9afb9c8bdded3d984b947f73b42e4f32862683f0cb639 |
