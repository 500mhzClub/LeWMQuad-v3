# Continuous gyro processing independent of camera selection

`lewm/camera_independent_gyro_development.py` now provides a small acquisition
component that processes every 100 ms gyro packet, including all fifty 2 ms
intervals, even when a visual consumer selects fewer camera frames. It retains
the latest sixteen exact-time rotation snapshots for camera association.

The existing `FastRelativeOrientation` performs the integration and sensor
checks. Packet gaps, rewritten boundaries, invalid measurements and identity
changes retain their existing rejection behavior. Camera queries cannot
interpolate, request future data or recover an expired snapshot. A gyro-stream
failure prevents further camera queries. Returned snapshots own their values.
This is a single-owner component; no thread, process or simulator is created.

Three focused tests passed in 0.25 seconds. They cover varying-axis rotation
with irregular camera queries, failure on a missing gyro packet, exact-time
retention and independent returned values.

A recorded check consumed frames 0–30 of the completed stop-conditioned maze02
run. All 31 accumulated rotations matched the original recorded tracker gyro
rotations exactly. It integrated 1,500 intervals while simulated camera
consumers requested only frames 0, 3, 7, 12, 20 and 30. Every intervening gyro
packet was still consumed. Median processing time was 0.598 ms and maximum
was 0.659 ms per packet in this small shared-host check, excluding packet reads.
The check completed in session 20175 with exit code 0. Per-frame equality and
timing results, query timestamps and input/source identities are retained in
`go2_camera_independent_gyro_recorded_2026-09-13.json`.

This closes one input-handling prerequisite for selecting fresh camera frames:
the intervening inertial measurements need not be lost. It does not yet make
the visual tracker accept gaps. That work must preserve real acquisition and
reference timestamps through visual continuity, measured-plane association and
pose availability checks, rather than relabeling a longer interval as 100 ms.
The existing image-chain fallback also requires its actual intervening images;
missing images cannot be manufactured from gyro measurements.

No visual tracker, map, planner or native command was executed in this check.
Gyro rotation remains consistency evidence, not a substitute for observed
visual pose or translation. The sensor calibration remains the existing ideal
development calibration. Continuous physical execution and hardware readiness
remain unproven. No live or queued navigation implementation was changed.
