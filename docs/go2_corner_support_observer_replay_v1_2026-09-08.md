# Corner detector with upright SIFT: complete bootstrap replay V1

The preceding spatial-SIFT quota candidate remains failed and ineligible:
result `1a8097c3ba3340ce00f3c4b20d995471b67f25a3743c9d8671f71f2e6ab027d9`.
Its left view had fewer than150 liftable SIFT detections, below every quota;
the inspected image contains large texture-block corners. This is a different
detector experiment, not an extension or resumed attempt.

Freeze Shi-Tomasi corner detection at quality0.01, minimum distance5 pixels,
block size3, current public-depth-valid mask, and maximum50,000 detections.
Reject corners failing the unchanged depth lift. Select up to50 per existing
4x3 image cell, maximum600, ordered by minimum-eigenvalue response then image
coordinates. Compute upright SIFT descriptors at size8 pixels and angle0.
Do not adjust these values using replay results. Mutual SIFT ratio0.7, LK
forward/back checks, lifting, rigid fitting, six-cell coverage, gyro consistency,
retained references, bridge budgets and failure latching remain unchanged.
No optical-flow-only fallback, native input or runtime monkeypatch is added.

Source-AST and synthetic tests check narrow observer transcription, detector
bounds/determinism, unknown-depth exclusion, real rendered RGB-D pose admission,
failure latching, coarse-block translation through unchanged registration and
repeated-checker descriptor rejection. This is an unqualified sensor candidate.

Run fresh original and corner observers on both complete authenticated
bootstrap traces,16 and254 frames including drains, from frame zero with no
reset after failure. The fixed input launch/result and every artifact are
verified before/after replay. The original left-frame5 failure must reproduce.
No commands, physics, fitting, checkpoint selection or mission rescoring.

Only after observer processing, use native traces for XY/rotation error. Record
all failures, current-pose missingness and active observer wall times. Eligibility
for a separately frozen native probe requires all270 current poses, no terminal
failure, maximum XY error at most0.02m and rotation error at most0.05rad in each
trace. This diagnostic threshold does not establish calibrated uncertainty,
novel-maze navigation or control timing. Keep the failed mission outcomes.

Inspect hardware/RAM/GPU/competing jobs/storage before launch. Use serial arms,
one OpenCV/Torch/BLAS thread, at least4GiB available RAM and256MiB output plus
40GiB storage reserve. Freeze new sources before computing results. Write only
to exclusive `go2_corner_support_observer_replay_v1_attempt_001`; no retry,
resume, feature-count search, gate changes or production adoption.
