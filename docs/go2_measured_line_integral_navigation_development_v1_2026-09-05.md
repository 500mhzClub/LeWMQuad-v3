# Measured-line / integral navigation V1: fixed development experiment

## Question and frozen comparison

Does fixed-lookahead line guidance plus bounded integral alignment advance the
continuous exploration/discovery/return controller past the audited measured-
region V1 failures? This is a joint two-component engineering intervention,
not a component ablation, learned navigation policy or JEPA benefit experiment.
Preserve the predecessor0/2 outcome, rank loss and four failed depth checks.

Exactly two original development layouts, north_dogleg and south_branch, retain
their physical construction, seeds, episodic memory, gait/gains, sensor/render
paths and fixed360-s/36-leg budgets. Method is `measured_line_integral`.
Output is `.generated/go2_measured_line_integral_navigation_development_v1_attempt_001`.
No retry, source edit, threshold change or alternate coefficient after launch.
This population is not independent evaluation or hardware evidence.

## Changed commands only

At selection of each measured local target, retain the observed approach origin
and heading in the initial body reference. Project heading into the observed
gravity plane. Compute signed cross-track error to that line; desired direction
is approach minus its left-normal times cross-track/0.5 m, clipped to±0.5.
Yaw request is1.5 times observed angular error, clipped to±0.25 rad/s. The fixed
spatial response length does not shrink as the endpoint approaches. No future
pose, true cell, target marker coordinate or simulator velocity is supplied.
Retain predecessor target selection, forward speed, measured braking, settling,
sampled clearance, arrival tolerance and terminal failures unchanged.

Alignment retains the exact0.02-rad heading and0.1-rad/s rate acceptance,
0.3-s dwell,12-s timeout and±0.35-rad/s cap. Add integral command at0.4 times
heading error per second, capped at±0.12 rad/s. Do not integrate farther into
saturation. Reset at sign changes, inside the heading tolerance and at terminal.
No-response trajectories must still fail; no acceptance relaxation is permitted.
Synthetic deadzone tests validate implementation, not actual Go2 response.

The wrapper replaces only newly constructed child/alignment operators before
their first observation. No active controller history is discarded. The entire
collector and core physical-audit function bodies are unchanged; only the
explicit imported controller differs. Full RGB/depth/gyro/state replay, native
physics, contacts, actuator identity, marker, memory and physical scoring remain.

## Interpretation and next actions

Require current measured translation and retain the original rank threshold.
Missing cumulative state stops control. Line guidance might preserve useful
surface coverage, but cannot make an unobservable translation component known.
Do not retrospectively claim the new commands would fix predecessor traces.
Any recurrent rank loss requires independently informative deployment-valid
sensing/fusion with explicit uncertainty, not commanded-zero substitution.

Report all full-task outcomes, first arrival, scan/alignment progress, marker
discovery, actual home return, false home claims, contact/faults, and every depth
and motion check. Full audit PASS certifies fidelity, not physical success.
Nominal sampled turning clearance is still not continuous-volume/future-gait
or hardware qualification. Preserve invalid rays and all failures.

After reliable execution, matched online-memory, supervised-versus-JEPA and
genuine multi-step rollout comparisons on independent layouts/seeds and sensor
robustness remain required, followed by bounded actual Go2 evidence when possible.
