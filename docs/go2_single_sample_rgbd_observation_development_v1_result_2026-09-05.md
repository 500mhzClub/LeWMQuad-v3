# Single-sample RGBD: actual visual-surface check passes

Collector22303 completed both fixed cases. Full audit32948 passed both cases
and all ten depth checks, exit0. The final source suite89300 passed1,068 tests
across93 files in74.67 s. This closes the ideal simulated visual-depth interface
check, not navigation, collision-clearance or hardware qualification.

The visible case's maximum checked absolute error over five frames is
0.000393204 m; the occluded case's maximum is0.000123277 m. Both are below the
prospective5-mm tolerance. Each frame has4,193–4,683 eligible interior rays.
Visible background rays remain invalid; the occluded marker has no visible
panel rays. RGB marker responses remain5/5 visible and0/5 occluded. The actual
OpenGL readback confirms a single-sample depth target, zero sample buffers and
samples, disabled multisampling and pixel scale1. No physics step or camera
transform change occurred between paired RGB and depth captures.

Actual native mesh and collision readback confirms rendered floor z=-.005 m
and collision floor z=0. All ten separate physical-floor-reference checks still
fail: maximum optical-depth differences reach30.912 mm in the visible case
and9.800 mm in the occluded case. A5-mm vertical surface offset can produce a
larger optical-depth difference on oblique rays. No runtime compensation using
floor truth was introduced. Original RGBD V1 remains unchanged and failed.

Evidence includes1,900 native physics/fast samples,190 ordinary sensor samples,
ten actual RGB/depth pairs, exact causal packet/history replay, native contacts,
camera geometry, actuator gains and artifact bindings. All commands were zero;
no contact or body-limit stop occurred. This is two correlated stationary
development configurations, not a noise model, moving-sensor validation or an
independent test set.

Root: `.generated/go2_single_sample_rgbd_observation_development_v1_attempt_001`.
Launch SHA-256: `088587ce5db2e7aec2e885b6f7bef0f74428ad6a38a540ede6c1114274e909e5`.
Result SHA-256: `0a54ba38f1d98c30c6fb9b49a1dd753c63e50bb65dc841886fe1a77b508a6146`.
Audit SHA-256: `47d945747f6a0bf8799c5aa6f6b822e3c7e6afa028561e886380c63a3fdfcb55`.
All260 source,192 input, two gait and ten native source bindings remain fixed.
No collection or audit remains running. Do not rerun or edit bound sources.

Next: extract observed local wall segments and occlusion boundaries, explicitly
test unobservable translation in featureless corridors, and connect measured
arrival/turning-clearance evidence to the continuous discovery-and-return task.
Forward depth alone must not certify unseen side/rear volume as free.
