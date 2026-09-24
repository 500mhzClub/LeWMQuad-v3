# Complete measured-plane native history: single-pass timing comparison

Compare the original `MeasuredPlaneResidualController` with the already defined
`MeasuredPlaneSinglePassController` on every sensor observation from the completed
`go2_measured_plane_dispatch_recovery_v1_attempt_001` native episode. Its exact
parent is PID 2916106, creation time 1789162140.42, and launch SHA-256 is
`93d0af54af1d07eb1981338e17313f5e08bca00f257a57c3a4bee055ef44aacb`.
Do not consume its growing decision stream or infer completion from timeouts.
Require the original parent to end and its full raw worker audit to complete.
Either navigation success or a fully audited scientific failure is admissible.

The earlier 123-observation combined-controller prefix completed with result
`5d0e1b47ca80b8c19474dd7a371957867c71ed9835cc97bc342b19752a22f087`.
Its approximately 41% time reduction is short-prefix evidence only. This new
experiment uses the revised estimator's own physical history, including all
available late history and the original ending. No frames, phases or failures
are selected for a favorable timing result. It performs no physical simulation
and makes no new navigation claim.

Both controllers receive independent instances of the same originally assigned,
corrected, evaluation-only no-RGB direct model. Original RGB/depth/gyro packets
remain unchanged; RGB still drives perception in both controllers. Alternate
controller order at each observation. Time only each complete `observe` call,
excluding input reconstruction, fingerprinting, state checks and output writing.
Reproduce the complete original native decision and compare every normalized
candidate decision, including action, forecasts, gates, map, mission and failure
receipts. Observe equal actual model call counts and unchanged model state.
Stop and preserve a failure immediately on any mismatch; never continue onto
sensors that would follow a different candidate action.

Compare the declared retained map/residual/history state at frames 0, 3, 61, 122,
255, 511, 1023, 2047 and 3071 when present, plus the actual final observation.
Use the existing ten exact retained-state type-path normalizations. In addition
compare the entire motion, registration and mission state. Normalize only the
registration implementation tag from the existing `TiledDensityFloorRegistration`
to `MeasuredFloorTransportRegistration`; reject all other registration types and
retain every registration field. Motion and mission types remain exact. This
eleventh explicit path was identified by actual image-state testing before the
experiment was frozen; no observed values are excluded.
Serialize retained OpenCV keypoints with all point coordinates, size, angle,
response, octave and class ID; do not omit opaque feature objects from the check.
Report all-observation and model-forward timing populations, fixed frame windows
[0,256), [256,1024), [1024,2048), [2048,3072), and [3072,4014), medians, p95,
totals and counts over 100 ms. Empty populations are explicit, not extrapolated.

The replay reauthenticates the complete original raw artifact roster before and
after execution and reconstructs every saved row against the closed original
decision stream and public sensor packets. Output includes ordered comparison
rows, state hashes, resource monitoring and the complete report. The preceding
reactive prefix must be complete and its owner ended. Source discovery follows
the explicit import closure and preserves every existing frozen source.

One full CPU replay may overlap one separately owned native comparator. This is
not an isolated benchmark. Require at least 64 GiB available RAM and 43 GiB
artifact space before dispatch. Keep the existing 41 GiB native disk reserve,
an 8 GiB available-RAM stop floor and a 2 GiB comparison-output cap during replay.
OpenCV and BLAS remain single-threaded; no training or GPU workload is started.
If dispatch resources are temporarily unavailable, the registered waiter keeps
observing resource state rather than starting an undersized attempt.

Output is `go2_measured_plane_single_pass_full_history_v1_attempt_001`. There is
no automatic retry, resume, native-controller substitution, realtime
qualification, new independent-maze result or goal-completion claim. A faster
replay must still meet realistic sensing and continuous-physics timing before
supporting deployment claims.
