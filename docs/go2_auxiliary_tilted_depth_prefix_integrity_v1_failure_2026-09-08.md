# Auxiliary depth segmentation-integrity V1 acquisition-order failure

The first integrity successor decoded background and link labels successfully,
saved one complete auxiliary RGB/depth/segmentation frame at physical sample
749, and executed one zero command through sample 799. It then stopped because
the collector required primary observation 1 before calling the session's lazy
`capture_current()` operation. Physics had advanced correctly; the primary
manifest still contained only observation 0. The frozen `sensor_packets()`
method normally performs that capture, but this sensor-only loop did not call
it. This is an acquisition-order bug, not a pixel, coverage or physical failure.

The completed auxiliary frame contains 307,200 valid depth pixels and zero
robot-label pixels, with the robot present in the visual/segmentation roster.
One frame does not characterize the proposed approach-view coverage. All
partial data and both failures remain immutable.

- Attempt: `go2_auxiliary_tilted_depth_prefix_integrity_v1_attempt_001`
- Launch: `d3ddd15fdc2439e3ef506336b07686ce7ef483f3b7e260b7c1183cb1739a2a8a`
- Failure: `ffc55133b4008d8ee590169380af265fa210dbebf24c7847db32b2fb2dacba81`
- One-command tape: `db5cd70a5819f926df94d4f2659234976e903ffd8f76e78d9d5837bc5b61461e`
- One-frame auxiliary audit: `f427fa5032417f72a33484ccd7554b60b1a8b3059c125f3c3b5f5c69ef78d6b6`

Nine segmentation/scope tests passed in 1.84 s before this launch. They covered
the background fix and unchanged collection/audit bodies, but did not exercise
lazy per-tick acquisition. The next distinct integrity successor must add that
test and explicitly acquire the current primary frame before checking its
index and capturing the auxiliary view. Preserve all camera, physics, command,
renderer, analysis and resource settings. No retry or resume of either failed
root is allowed, and neither failure provides navigation evidence.
