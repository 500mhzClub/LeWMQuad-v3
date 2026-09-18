# Contact-worker direct-flow observer prefix V1

Replay the actual public camera, depth, body and fast-gyro packets of the
completed contact-scoring worker, beginning at frame zero. Instantiate fresh
DualCameraVisualMotion and existing DirectFlowDualCameraVisualMotion observers.
Require every original visual evidence record to reproduce completely. The
candidate may differ only through its declared association fallback; stop at
the first changed evidence, either observer's terminal failure, or frame 561.
Do not consume a subsequent observation after any such boundary.

Every current-pose result must pass the original current_dual_camera_pose
contract against its actual public packets. Both observers must leave those
packets unchanged. The existing candidate retains the original rigid, gyro,
continuity and bridge gates, checks qualified original measurements for
conflicts, and does not reset tracking history. The replay records all evidence
and failure receipts, including early or unsuccessful fallback outcomes.

The replay authenticates the original completed worker and raw artifact roster,
the raw-match diagnosis and the fixed-pair association probe. Final contact
parent completion is not claimed or needed to diagnose its ended worker. It
does not replay floor registration, mapping, learned model predictions or
command selection, and cannot establish native navigation success.

The sole prior CPU replay is the exact original sustained-turn raw replay
PID 2813368, creation time 1789115323.91, on the recorded boot and command line.
Wait for that process to end, then verify its completed result, fixed launch,
source closure, all output artifacts and declared 407-observation boundary.
Do not restart it on a timeout or failure. Waiting is bounded to 48 hours,
polled every 30 seconds. No observer is instantiated while waiting. A source
preflight verifies source and resources without reading the completed worker's
runtime artifacts or creating output. Full startup creates an exclusive
observer output and records the waiting process and prerequisite identity.

Use single-thread CPU OpenCV with OpenCL disabled, 40 GiB available RAM for
observer plus concurrent native work, 40 GiB artifact reserve and a 256 MiB
observer output limit. At most 562 observations are replayed. Preserve failure
outputs and do not retry, replace or resume the execution. A successful pose
boundary would justify a separately checked controller replay and prospective
native follow-up; it does not select an independent-study policy.
