# Causal load/joint/IMU kinematics V1 — fitting diagnostic only

Freeze both hypotheses before outcomes: (A) stationary spherical-foot centre,
`v = -(omega_body cross r + J_linear dq)`; (B) stationary material contact point
at the sphere's nominal bottom along conditional IMU up,
`v = A + .022 * (omega_body + J_angular dq) cross up`.
The latter assumes a locally level contact normal, not an observed continuous
floor. Rolling/slip, different normals and IMU error may invalidate either.

Use ONLY the new ideal-load predictions and existing50Hz body/500Hz gyro sensor
arrays for fitting data. Strict same-stream identity/calibration, current valid
q/dq/gyro/specific force, and11co-timed2ms load samples are required. Select feet
whose valid unsaturated resultant exceeds5N throughout the20ms window; do not
filter with ground labels, native contact position, world pose or guard results.
Report per-foot predictions for both hypotheses; use unweighted mean only when
at least two feet are selected. Report disagreement, not a safety threshold.

Initialize a conditional up hypothesis from11specific-force samples at1.3..1.5s,
each gyro-transported into the1.3s anchor frame. Normalize their mean and transport
it causally with subsequent500Hz midpoint gyro integration. This assumes quiet
initialization; acceleration contamination and gyro bias remain unknown errors.
No native up/orientation enters prediction. Score from1.5s through60s at50Hz.

Compute exact revolute-joint linear/angular Jacobians from the URDF chain,
with synthetic central-difference tests. Report foot-height spread along the
conditional up direction and load components along it; do not turn them into
plane, ground, no-slip or future-sweep permission. Missing load values are
explicitly unavailable, not observed zero force.

Persist complete predictions BEFORE native scoring or RGB-D comparison. Score
body-velocity error, up error, cross-foot disagreement and loaded-foot residuals
by original motion phase, not just startup. Compare each eligible100ms integrated
kinematic displacement against frozen joint-RGBD and gyro-RGBD pose differences
in the current body frame; keep unavailable windows. Use gyro-only relative
rotations for kinematic integration, not RGB-D or native pose feedback.

No fitting, threshold adaptation, physical replay, validation-role selection,
learned update, controller integration, retry or overwrite. Preserve missing
evidence and failures. A later frozen fresh challenge must test slip, uneven
supports, weak sensing and commanded forward/turn/brake response before policy
use. Ideal three-axis loads are not calibrated vendor channels. Full maze,
memory, JEPA/multistep/layout/seed/timing/hardware aims remain unachieved.
