# Fixed causal gravity-feedback ground comparison

## Question and fixed design

The completed ray diagnostic found point errors up to1.18 m and accumulated
normal errors despite accurate initial gravity. Test whether causal force
feedback improves the normal/height/ray estimate on all8 preserved route traces
and all1,498 actual packets. No new physics, fitting, JEPA-input change, source
edit of an executed predecessor, radius search or navigation qualification.
Retain all4 original task failures. These are correlated reused development
routes, not independent new mazes or hardware trials.

Three fixed modes: unchanged `gyro_only`; `body_mean_feedback`, which averages
the20 force vectors directly; and `transported_feedback`, which first transports
each force vector to the current body frame using the co-timed50-Hz gyro history
and the existing midpoint SO(3) increment. The body-mean arm is a diagnostic of
frame mixing, not an assertion that vectors from distinct frames are equivalent.
Initialize all modes identically from the existing zero-command ground estimator.
Correct only the up vector, not heading/yaw or measured gyro values. Recompute
the same lowest-foot-sphere height hypothesis with the corrected normal.

For each100-ms packet, transport the previous estimated up vector using the
unchanged relative gyro increment. When the fixed gates accept, blend it with
the normalized force-history mean using alpha =1-exp(-0.1/2.0), then normalize.
The time constant is2 s, fixed before this experiment. Require valid co-timed
force/gyro history, five valid recent100-ms applied commands with each component
range at most0.05 (m/s,m/s,rad/s respectively), mean force magnitude within0.75
m/s² of9.81, and vector residual RMS at most3 m/s². Reject unavailable/dynamic
force as feedback, retain gyro propagation, and count each rejection reason.
Reject rewritten force/command/gyro histories or invalid policy clocks with a
latched sensor failure. No current ground-truth pose enters the estimator.

These heuristic gates are not calibrated confidence tests. Constant unknown
acceleration can satisfy all gates and tilt the estimate; a synthetic test
explicitly preserves that limitation. They do not certify gravity/acceleration
separation, contact stance, absolute heading, camera calibration or hardware
accuracy. The sensor model is still ideal body-origin force from a backward
velocity difference, without real-IMU lever arm, bias, noise or delay.

## Evaluation and preservation

Use separate audited actual body/camera geometry and the same stride8 visible
floor references. Require every gyro-only frame's visible-floor count, nominal
valid count, point-error sum and maximum to reproduce the completed ray study
exactly. Its physical bytes are not regenerated. An unexpected baseline failure
is integrity FAIL, not a favorable missing comparator.

Report per-frame and per-route normal angular error, body-height error and
nominal visible-floor point error mean/max/valid counts. The primary descriptive
endpoints are route-macro normal angular error and the fraction of actual
visible-floor rays whose point is unavailable or has error above0.25 m. Missing
projections count as failures in that fraction; do not improve it by censoring
unavailable rays. Retain the ordinary conditional point-error mean with its
explicit denominator as a secondary endpoint. Report every mode and all gate
counts, not only the best mode or a favorable route.

If a feedback estimator fails, retain its fault and count all subsequent source
frames/pixels as unavailable. Do not compute a full-route paired normal/point
mean by silently dropping failed frames. Require complete matched routes for
the reported mean contrasts; otherwise report them unavailable. Report the
available counts and failure fraction in all cases. No significance claim,
bootstrap radius selection, interim tuning or online action authorization.

## Execution

Exact fresh root:
`.generated/go2_gravity_feedback_ground_development_v1_attempt_001`.
Runner: `scripts/analyze_go2_gravity_feedback_ground_development_v1.py`.
Bind the exact prior ray launch/result, its full source/input dependency set,
and new runtime/runner/protocol plus22 synthetic estimator/metric tests before
actual replay. Verify current policy/RGB/pose/camera files against original route
artifacts, and source-generated geometry against launched specs. Reverify all
bindings at completion. Any unexpected integrity/runtime error yields retained
FAIL; do not retry, resume or edit executed source. No protected material,
whole-tree export, GPU, physics or hardware execution. One CPU thread, separate
from the fixed18-model coverage training whose source/inputs remain untouched.

## Consequence

A positive result would support testing this sensor estimator on fresh dynamic
excitation and then using it in a separately scoped temporal observation/exit
component. It would not repair palette dependence, the current camera near-field
gap, unobserved body volume, place ambiguity, or unqualified traversal/arrival.
A negative result remains informative about force contamination and gyro drift;
do not introduce a tuned variant under the same experiment identity. Full maze
exploration/discovery/directed return and real Go2 evidence remain unfinished.
