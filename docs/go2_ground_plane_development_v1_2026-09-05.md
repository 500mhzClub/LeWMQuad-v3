# Body-sensor ground hypothesis: fixed replay on26 streams

Test a new deployment-input-only ground-plane hypothesis on all eight old
multi-junction routes and18 old turns, without new physics, fitting or controller
changes. Initialize only from the first actual zero-command post-settling
packet. Require complete specific-force and command histories, all zero commands,
mean force magnitude8–12 m/s² and gyro RMS<=.5 rad/s. These broad consistency
checks do not certify zero acceleration or hardware calibration. Propagate the
initial normalized mean specific-force direction with the existing relative
gyro integrator, without resetting at route edges.

Use current ordered joint angles and the exact bundled Go2 URDF geometry:
hip origins x=±.1934,y=±.0465 m; lateral thigh offset±.0955 m; two.213-m leg
segments; foot collision-sphere centre offset−.002 m in calf x and radius.022 m.
The source URDF SHA is bound in the launch. Compute all four sphere centres and
assume the lowest sphere touches a common flat plane. This estimates height;
it does not measure support or detect flight, stairs, slip or uneven terrain.
The runtime result remains explicitly unqualified. Pixel-to-plane projection
uses the fixed calibrated camera mounting/intrinsics, not camera-world pose.

Evaluation alone uses the already-audited raw body pose and camera transform to
report every observed height, up-direction and camera-height error. Independently
verify closed-form runtime foot positions against generic homogeneous URDF chain
composition at every consumed packet (maximum coordinate difference1e-12 m).
All26 attempts remain: report initialization or later sensor rejection as
unavailable with its exact frame, rather than substituting world gravity or
discarding it. No noise covariance, height correction or acceptance threshold
is fit from these errors. Original route/turn task failures remain represented.

Fixed fresh output: `.generated/go2_ground_plane_development_v1_attempt_001`.
Before output creation, preflight rejected the virtual-environment URDF path
because it resolves into RecoveryStorage. The exact resolved file was inspected
read-only, its SHA matched, and its explicit absolute path was bound instead.
No source tree or asset was copied and no alternate geometry was substituted.
Bind predecessor source/result/audit identities and the new estimator/test/replay/
protocol/URDF bytes before execution, then verify afterward. Retain failures
without retry. This measures a candidate RGB-plus-body geometry component, not
an exit/clearance qualification or permission to merge places by estimated pose.
