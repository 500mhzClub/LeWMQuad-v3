# Gyro-seeded correspondence development comparison V1

Question: does tracking reference features directly, without requiring current
descriptor redetection, improve whole-stream visual pose availability under
unchanged rigid registration and motion gates? The completed registration
diagnosis does not establish that discarded matches were correct or justify
weakening spatial-support gates.

## Fixed method and limitations

Use existing SIFT reference locations, deterministic half-pixel deduplication,
at most600 unique reference points, observed reference depth and measured
relative gyro rotation. Rotate reference body points into the current body with
a zero-translation initial guess; project with existing camera extrinsics and
pixel-centre conventions. This seed is not a measured correspondence and
contains neither native pose nor a commanded-displacement prior.

Run bidirectional LK with the existing window/pyramid/termination parameters and
0.5-pixel round-trip check. Reject invalid/out-of-image forward endpoints before
reverse tracking. Deduplicate current positions and apply existing bilinear
depth validity/discontinuity checks. Use exactly the existing robust rigid
registration, reprojection, inlier-fraction, six-cell support, translation and
increment gates, primary-first recent-reference selection, promotion and terminal
failure behaviour. No alternate-method search, original fallback, reanchor,
restart, post-failure extrapolation or per-trajectory frontend selection.

The correspondence checks **do change**: mutual descriptor association and
proximity to an independently descriptor-matched endpoint are replaced by
rotation-seeded tracking. There is no one-pixel limit around the rotation-only
seed, since translation is unknown. The final bidirectional rigid-reprojection
check remains one pixel. Keeping it does not establish equal association
reliability or calibrated pose uncertainty.

Known-transform fixtures include camera lever arms, body rotation, translation
and depth changes. Negative/limitation cases include blank or unrelated images,
occlusion, repeated texture, depth discontinuity/unknown rays, duplicate
locations, wrong point associations, gyro errors and clock/identity/image-binding
failures. A periodic plane can produce identical images under zero and nonzero
translation; retain that accepted-but-ambiguous case rather than manufacture a
rejection guarantee. A small common depth bias can also pass the gates; a larger
bias can be rejected. These remain limitations.

## Complete paired population

Use all three inner-arrival and all three persistent-intent recordings with
both original and gyro-seeded observers:7,433 recorded frames,14,866 frame/arm
observations. Authenticate the completed predecessor launch/result, all its
output hashes, source/native/input bindings and six exact original estimate
streams. Original pose, reference selection and terminal failure must match
every predecessor estimate, including drain/no-op frames. Do not shorten streams
on candidate failure or score only failing intervals.

Save and bind all six paired sensor streams before parsing native coordinates.
Only then evaluate exact samples749+50*frame with matching timestamps. Report3D
position and orientation error in the initial body frame, with available-arm
and paired common-frame distributions. Preserve both/original-only/candidate-only/
neither availability counts, first failures, all frames and per-stage candidate
evidence. Native truth cannot affect matching, selection, reference promotion,
method choice or output estimates.

This is reused development data, not independent validation. Availability gain
alone cannot qualify the candidate for control. Changing the estimator in closed
loop would change motion, imagery and future keyframes. Report new failures and
worse paired errors alongside gains. Do not select thresholds, revise this
method mid-run or claim successful physical recovery.

Timing is untraced per-observer wall time excluding packet loading, complete
control and physics. Original/candidate calls share each packet sequentially,
original first, and may overlap collection. This is not an isolated timing
benchmark, real-time control or hardware qualification.

## Execution and next decision

One exclusive output at
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_gyro_seeded_correspondence_replay_v1_attempt_001`.
Use the existing2GiB output allowance and40GiB free-space reserve, exclusive
durable streaming writes, bounded metadata/rows and verified source/input/output
bindings before and after. Retain terminal failure and partial artifacts on any
fault; no retry/resume/replacement. CPU/OpenCV single-thread, exact development
venv, PYTHONDONTWRITEBYTECODE=1, PYTHONHASHSEED=0,
PYTHONPATH=.:lewm_genesis:lewm_worlds and OMP/MKL/OPENBLAS_NUM_THREADS=1.
No GPU, new simulation, training or hardware operation. Preserve the running
771-source collector and reviewed786-source matched-study definition.

After complete verified results, decide whether the fixed method warrants
additional independent scene/sensor/error challenges. Prior periodic/common-bias
counterexamples remain even if all recorded trajectories improve. Closed-loop
adoption requires that additional evidence; no replay is a new room return.
The separate low-friction dynamics failure and pending36-fit independent-layout
JEPA/baseline study remain in scope, followed by useful online rollout,
memory/backtracking, unfamiliar-maze missions, realistic sensing, real-time
execution and bounded hardware evidence when access permits.
