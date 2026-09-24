# Complete chained-native history: single-pass timing comparison

Compare `MeasuredPlaneChainedAnchorController` with
`MeasuredPlaneChainedSinglePassController` on every observation from the exact
completed `go2_measured_plane_chained_maze02_v1_attempt_001` run. Bind native
launch SHA-256
`0f2963636abc4f1fc738e9323d2a74db02888ca7062dc1afa21023f8c443b8ff`
and require its original parent and worker to have ended, its complete raw
audit and physical prefix checks to have finished, and its actual result hash.
A fully audited negative outcome is admissible. No growing stream, favorable
frame subset, different native attempt, retry or resume is admissible.

The current collection reports 4,014 observations and mission-budget exhaustion;
its audit is still pending when this protocol is written. The actual completed
result, not that preliminary receipt, must determine the admitted population.
The existing 4–4,014-observation bound and the original physical command
endpoint rules remain in force. This experiment makes no new physical
navigation claim and does not change the native controller.

Use two independent evaluation-only instances of the originally assigned
corrected no-RGB direct model. RGB/depth/gyro public packets still drive visual
perception in both controllers. Alternate execution order at each frame.
Reproduce every complete original decision and compare the complete normalized
candidate decision, including model forecasts, actions, tracking, map, gates,
mission, failure and terminal receipts. Stop at a mismatch before consuming the
next observation. Check equal actual model-forward counts and unchanged model
weights with no gradients. No training, GPU execution or new simulation occurs.

Both controllers use the exact chained visual-motion class. Preserve all its
types and fields, including image caches and reacquisition witnesses, and the
complete mission state. Use only the existing ten map/residual/history type
normalizations and the one existing registration implementation type
normalization. Compare complete retained-state fingerprints at frames 0, 3,
61, 122, 255, 511, 1023, 2047 and 3071 when present, plus the actual final
observation. Preserve all OpenCV keypoint fields through the existing explicit
serializer. No additional scientific field or observer type is normalized.

Time complete controller `observe` calls, excluding input reconstruction,
fingerprinting, retained-state checks and output writing. Report all-observation
and actual-model-forward populations, totals, medians, p95, counts over 100 ms
and the existing fixed windows [0,256), [256,1024), [1024,2048), [2048,3072)
and [3072,4014). Do not transfer the earlier 53% improvement to this controller
pair without this experiment. Instrumented component profiling is outside this
protocol, and the result remains non-isolated timing evidence.

Reauthenticate the complete native artifact roster before and after replay,
reconstruct its original physical prefix, and independently reconstruct every
saved comparison row against original decisions, sensor packets, timestamps
and command endpoints. Recompute the final report from saved rows and state
checks. Bind sources, protocol, owner, boot, hardware, exact input admission and
all output hashes. Preserve any failure and partial evidence.

Before dispatch require ended native owners, no active native attempt or named
repository CPU replay, at least 64 GiB available RAM and 43 GiB artifact space.
During replay retain the existing 8 GiB available-RAM stop floor, 41 GiB disk
reserve and 2 GiB comparison-output cap. OpenCV, BLAS and Torch remain in the
original deterministic single-thread CPU environment. Source-only preflight
reports hardware without admitting runtime resources or executing the pair.

Use the exclusive output
`go2_measured_plane_chained_single_pass_full_history_v1_attempt_001`.
The launcher is
`scripts/replay_go2_measured_plane_chained_single_pass_full_history_v1.py` with
`--chained-native-result-sha256` set to the actual completed native result hash.
This is a paired recorded-history performance experiment. Native adoption,
independent-maze reliability, real-time execution, hardware and deployment
remain unproven by it.
