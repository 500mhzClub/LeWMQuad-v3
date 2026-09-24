# Spatial feature support candidate: complete bootstrap trace replay V1

The completed learned-goal bootstrap failed its left case because all qualified
rigid-consensus inliers covered only three current image cells; six are required.
The right case tracked but its learned controller only turned. This experiment
addresses visual support only and cannot repair or relabel either mission.

Freeze one candidate before reading its performance on either trace. Detect
default SIFT points in the current public-depth-valid image mask, reject points
which the unchanged depth lifting rejects, and keep up to50 strongest points
in each existing4x3 image cell, at most600 descriptors. Use deterministic
response/coordinate/scale/octave/angle ordering. Fail if the detector returns
more than50,000 points. Descriptor, matching, LK, lifting, rigid proposal,
consensus, coverage, gyro consistency, reference retention, bridge and pose
continuity rules remain unchanged. No native state enters the observer.

The new observer copies only the frozen observe method to replace the feature
constructor and make the existing joint-mode labels explicit. A source-AST
test checks this exact scope; registration, continuity and reference methods
are inherited without replacement or runtime monkeypatching. Synthetic tests
exercise feature bounds, valid-depth coverage, deterministic output, input
ownership, rendered RGB-D pose admission and latched failure.

Authenticate the complete bootstrap result
`71601c660272b579db520ba631af661d76f869267cf1c4a26fb60b17cdfbb7f1`
and its launch and all1,146 bound artifacts. Replay both complete recorded
episodes,16 and254 frames including terminal zero drains, in order. Run a fresh
original observer and a fresh candidate from frame zero, with no reinitializing
after failure. Preserve every frame and failure; no selected-frame scoring.
The old controller stopped querying its observer at terminal, so full-drain
observer replay is a separate diagnostic and is not old command replay.

Only after observer processing, compare accepted poses with recorded native
body motion relative to the first camera frame. Record completeness, exact
failure, XY and rotation errors, feature coverage and observation wall time.
Candidate eligibility for a separately frozen native probe requires all270
frames to supply admitted current joint poses, no terminal failure, maximum
XY error at most.02m and maximum rotation error at most.05rad in each trace.
These are diagnostic thresholds, not calibrated uncertainty or navigation
qualification. Original baseline failure at left frame5 must reproduce.

Assess available CPU/RAM/GPU/storage and competing processes before replay.
Use serial arms with OpenCV, Torch and BLAS limited to one thread for comparable
observer timing; no physics, tensor training or GPU workload is launched.
Reserve256MiB output plus40GiB free storage and require4GiB available RAM.
Write new receipts and complete evidence only to the exclusive root
`go2_spatial_support_observer_replay_v1_attempt_001`. Bind sources before
execution and verify sources/input bytes after both traces. No adaptive feature
counts, repeated attempt, threshold adjustment or production adoption.
