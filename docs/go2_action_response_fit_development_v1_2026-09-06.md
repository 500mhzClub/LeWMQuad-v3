# Frozen specification: A-only sensor-supervised response fit

One fit at `.generated/go2_action_response_fit_development_v1_attempt_001`.
Input is only the fixed, verified identification A acquisition. No B acquisition
exists or is consulted; no fit selection or hyperparameter sweep is allowed.
Native trace files are integrity-hashed, not parsed for features or targets.
Native truth remains evaluation-only. Targets come from future deployment-valid
depth/gyro estimates and joint observations, not simulator pose/contact labels.

Replay A's same continuous state and all recorded decisions. Train on its 28
scheduled 100-ms transitions; exclude a transition only when future depth
translation is not rank-3 observed, and record every exclusion. Require >=14 rows.
Do not include stopping-tail transitions as extra unreported fitting samples.

Fit an affine body response with 11 features: intercept, current estimated body
velocity/0.3, current measured gyro/0.5, previous and next applied forward/yaw
commands scaled by0.3/0.5. Its 12 targets are next-interval displacement/0.03,
gyro-derived rotation vector/0.05, next estimated body velocity/0.3 and next
measured gyro/0.5, all with explicit current/next body-frame conventions.

Fit each joint independently with seven features: intercept, current position
relative to fixed [0,.8,-1.5] per joint class divided by0.5, measured velocity/5,
and the same four command features. Targets are position increment/0.5 and
next measured joint velocity/5. Use ridge0.01 on all non-intercept coefficients,
no penalty on the intercept, no learned normalization, clipping, spectral-radius
projection, feature search or model selection. Total coefficients:300.

Forecast by propagating predicted pose, joints, joint velocities, body velocity
and gyro through each of four future applied commands. Never insert future
observations during rollout. Reject nonfinite output, but do not hide finite
large errors by clipping. This is a small supervised response model, not JEPA,
a learned navigation policy, or a calibrated contact/swept-volume model.

Freeze model file hash, canonical coefficient identity, training-row hash,
source/input/native bindings and fit outcome before B implementation/launch.
Report in-sample fit as in-sample. The fixed B schedule remains the separate
development validation; its predictions must include joint-velocity and
joint-position persistence controls on the same observations/commands.
Score every fully recorded executed horizon at0.1/0.2/0.3/0.4s, reporting body,
joint and primitive-point errors, failures and truncated/changed-command cases.
No B outcome may select a refit on the same B evidence. Neither passing this
fit nor a single validation trace establishes independent-maze or JEPA benefit.
