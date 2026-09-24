# Whole-stream registration conditioning witnesses V1

This diagnostic does not change an estimator, registration gate, recorded image,
physical result, controller, model or benchmark role. Its question is whether
the latest tracking failures arise at descriptor matching, flow, depth lifting,
robust consensus or a final spatial-support gate, and how those sensor-only
witnesses relate descriptively to actual recorded pose errors. No rejected
candidate may be fed back to tracking or control.

## Fixed population and separation

Use all three terminal-audited inner-arrival streams with the original frontend,
and all three terminal-audited persistent-intent streams with the original and
balanced frontends: nine estimator streams, all 12,138 frame/arm observations.
Input roots, launch/result/audit identities and the six original paired replay
estimate hashes are explicit in the executable. Verify complete original
artifact bindings before and after processing, and verify all source/native
bindings. No directory or checkpoint discovery is permitted.

Preserve every frame, including setup/initial references, rejected observations
and failure-latched terminal drain frames. Never restart, reanchor, choose a
frontend by native error, skip a difficult scene or substitute available-pose
counts for full-stream denominators. Original current poses must reproduce all
recorded physical decisions exactly. Both persistent-intent frontends must also
reproduce their complete previously saved pose, selection and failure streams.
Those checks cover the old balanced-left failure, not only its old right recovery.

The existing observer and exact matching/registration/candidate functions run
unchanged. A narrowly scoped, single-thread Python tracer reads only their
actual arguments/local return or failure witnesses. It does not replace
functions, alter arrays, proposal seeds, thresholds, keyframes, reference
selection or outputs. Existing tracers cannot be replaced. Trace errors stop
the diagnostic; they cannot become estimator acceptance or disappear into an
ordinary visual failure.

Persist all nine sensor-only estimate/witness streams and their hashes before
parsing any native coordinate arrays for evaluation. Original native files may
be hashed beforehand as provenance, but their values do not enter the observer,
the witness extraction or reference decisions. After the sensor phase is bound,
evaluate exact native samples 749 + 50*frame with verified timestamps and the
original frame-zero body coordinate anchor. Store native error separately for
available poses and converged diagnostic candidates, including rejected ones.
Do not call a stale intermediate fit a converged rejected candidate.

## What the witnesses mean

Count the actual keypoints, ratio/mutual matches, duplicate-location removals,
bidirectional-flow survivors, per-view and paired depth survivors, robust
consensus/pruning counts and final grid/fraction/translation gates. Preserve
original rejection strings and candidate qualification separately from selected
reference identity. A qualified alternative is not necessarily selected.

With gyro-fixed rotation, translation is the mean of a_i - R b_i. Its conditional
translation normal matrix is N*I regardless of image-grid occupancy. This
algebra assumes correct static matches; it does not prove the five-cell
rejections safe or invalidate the six-cell gate. Report spatial scatter and
centered angular normal eigenvalues separately, with their own units. Do not
report a unit-mixed pose condition number as an error certificate.

Report the first-order sensitivity of the fitted translation to common gyro
angle error: the spectral norm of R*[mean(b)]_cross is norm(mean(b)), in metres
per radian. Correct-correspondence/rotation assumptions and correlated sensor
errors remain explicit. Low residuals, many matches and full algebraic rank do
not bound common-mode point bias. No covariance or confidence calibration is
performed. Analytic tests must demonstrate this distinction rather than only
checking matrix shapes.

Report complete per-frame errors and descriptive candidate groups: qualified,
grid-only rejected and other rejected. Keep pair counts distinct from frames
and independent trials. Aggregate maxima are diagnostic, not a selected
acceptance threshold. Do not fit a new confidence gate on these same failures
or claim generalization from reused room recordings.

Measure instrumented whole-observer and matched-pair matching/registration wall
times, with available-frame and all-frame summaries separate. Tracing changes
execution overhead, terminal no-ops change timing populations, and collection
may overlap this replay. These are not isolated uninstrumented measurements,
hardware latency, real-time control or a benchmark of physical simulation speed.

## Execution bounds and next decision

The exclusive output is the owned RecoveryStorage child
`go2_registration_conditioning_replay_v1_attempt_001`. Keep a 40 GiB free-space
reserve, a 2 GiB total artifact allowance, bounded metadata/individual rows,
exclusive writes and durable streaming output. Retain partial evidence on any
failure; do not retry, resume or replace an attempt. CPU/OpenCV single-thread
replay only; no GPU, new physics, fitting or hardware operation. The active
collection and reviewed matched-study source closures must remain unchanged.

If witnesses reproduce all original estimates, use the full error/conditioning
population to choose a distinct estimator or uncertainty experiment. A new
qualification must be evaluated on additional independent sensor/scene/error
challenges before closed-loop adoption. Do not merely lower the grid threshold,
repeat the failed same-room tolerance change or retrospectively choose frontends.
The low-friction dynamics failure is separate and remains for the matched
prediction study and later physical execution tests. Useful online rollout,
memory/backtracking, whole unfamiliar-maze missions, deployment-valid sensing,
real-time operation and bounded hardware evidence remain unfinished goals.
