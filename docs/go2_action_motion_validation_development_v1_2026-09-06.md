# Frozen-response command validation B

One new CPU-physics/software-EGL acquisition at
`.generated/go2_action_motion_validation_development_v1_attempt_001`, seed
2026090604. No retry, overwrite, refit, clipping, model search or B-based tuning.
This is development command validation in the same calibration-arena geometry,
not independent-maze evaluation, a sealed benchmark, or deployment qualification.

The A-only sensor-response fit completed before this B source was implemented:
28 transitions, no exclusions, 300 coefficients. Fit launch SHA-256
`9cb3cfc1e0da625c940e00f3c37ecb91a6f3cbe309af96a29b0c40cbcd0f656f`;
result `5e61b02d27258da30bcc7439bf6f8d1b496d5b8f200c7756a864aab39ba6cf3e`;
model file `090c0227e50969db07b6083f23ea76e11a20276c03a54a972e8813f0f81f0f3f`;
canonical model `119b3612887ec19e293d402b0991c51459483d9cbe369ae7cc5815cdd31258c1`;
training rows `819244cb39f280afbe54f2bb972506d9a79a7172a6d2816f2afa8a6a55ea8639`.
Normalized body training RMSE 0.06663738381747766 is in-sample, not validation.
Sources, inputs and native bindings must pass before and after B and raw audit.

Reuse the unchanged startup, measured stopping tail, continuous MotionState,
physical guards, low-level learned gait, calibrated-arena setup assumptions and
all sensor/evaluator separation of A. New region is [-1.25,1.25]^3 in initial
body coordinates, valid from 1.5 through 8 s; initial velocity 0 +/-0.02 m/s.
These are explicitly native-checked setup conditions, not onboard scene facts.
The declared B schedule already existed in A's frozen source: six 100-ms ticks
forward at 0.10 m/s, four zero, four yaw at -0.35 rad/s, four zero, six forward
at 0.08 m/s, four zero, followed by three real zero-command tail ticks. Requested
commands are not measured velocities. Limit 61 decisions; no physics after a
native stop. Unexpected sensor-contract errors retain partial artifacts and
stop without an implicit tail/restart. Returned controller terminal states use
the same bounded three-tick tail as A, with native guards still active.

At every scheduled command, retain the original ideal-body/joint-velocity
persistence, ideal-body/joint-position persistence and frozen learned response
forecasts for the next four commands. All see identical current observations
and future requested/slew-limited commands. No prediction selects or authorizes
commands; neither the learned model nor persistence error is a safety guarantee.
The response rollout receives no future observations or simulator truth.

Independently reconstruct raw sensors, all decisions, commands, physical guards,
setup, stopping and terminal native identities. Score every fully recorded,
actually executed 0.1/0.2/0.3/0.4-s forecast, retaining truncated/changed-command
cases and requiring matched populations for all three models. Report per-case
and horizon mean/max body translation, rotation, joint, primitive-centre error
and rigid primitive material-point error upper bounds. Counts above the earlier
illustrative 50-mm assumption are descriptive, not a changed acceptance rule.
The 28 overlapping forecasts per horizon are not independent experimental units.
Any negative result remains intact; B cannot justify selecting a revised fit
and rescoring that fit as independently validated on B.

Add a monotonic outer timer before each capture/packet assembly and end after
decision, printing, command execution and its bookkeeping. Retain exceptional
loops. Mark the first loop's already-captured admission packet explicitly;
exclude it from complete-capture latency summaries. Timer finalization and
appending its own row, setup, final tail, and final artifact persistence are
outside these decision-loop intervals; whole acquisition time is also recorded.
Keep component sums separately and never label them full-loop timing. No
real-time, contact/support, calibrated-error or hardware qualification follows.

Following B, use the recorded errors to decide what local motion model remains
needed, then integrate continuous full discovery/marker/return missions. The
goal still requires matched RGB-plus-sensor supervised/JEPA/geometry comparisons,
genuine multistep prediction affecting navigation, memory contribution, novel
layouts/seeds/robustness and bounded real-platform evidence when available.
