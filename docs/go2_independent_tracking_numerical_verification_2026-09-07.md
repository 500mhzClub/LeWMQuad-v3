# Independent tracking measurement reconstruction

This adds an independent numerical check for the pending fixed tracking
challenge. It does not launch an experiment, change its commands, alter its
success criteria, or replace raw sensor/contact/geometry validation. The original
learning collector and 36-fit study remain the execution priority.

Implementation:
`lewm/independent_tracking_numerical_verification_development.py`.
It has no file-access or experiment-launch interface. A later complete-result
verifier must authenticate the eight-trial base and 88-stream stress sensor
phase **before** loading native arrays and calling these functions.

## Measured physical execution

`verify_coverage` reconstructs the nine segment boundaries from a separately
transcribed fixed schedule, checks the complete 500 Hz native prefix and its
relation to completed/partial command intervals, and retains early stops.
Direction must be supplied separately from the authenticated trial specification;
the checker does not take the report's own direction as its authority.

Yaw change is reconstructed from the world projection of body-forward and
principal incremental heading changes, rather than calling the production
Euler-yaw/unwrap calculation. Displacement and path length are distinct. A long
path returning to its start does not satisfy the 20 cm translation requirement.
The two measured turns must have the requested signs and reach 150 degrees.
Every sample in the final second contributes to the stopping maxima; one bad
sample prevents the stopping claim. A complete command tape without that
motion is still a negative result.

The native quaternion contract permits 1e-7 norm roundoff. The independent
rotation implementation normalizes quaternions, whereas the production yaw
formula uses their raw components. Coverage scalar comparison therefore uses
1e-6 absolute tolerance. Boolean threshold outcomes remain exact: a threshold
disagreement is a verification failure, not permission to round a failure into
a pass. This is numerical tolerance, not a physical accuracy or safety bound.

## Pose error and availability

`verify_pose_stream` accepts bounded iterators for one base or stress stream.
It independently joins each frame to native sample `749 + 50 * frame` and the
exact 100 ms observation clock. It reconstructs initial-body-frame truth,
absolute displacement/orientation error, and consecutive-estimate incremental
errors. Rotation error uses matrix skew/trace `atan2`, not the production
quaternion-composition error routine. Summary means and linearly interpolated
quantiles are independently reduced from all reconstructed errors.

The check preserves both observer populations, the common-availability subset,
null errors after failure, first-failure indices, observer-only timing and the
bridge-error subset. An observer cannot resume after terminal unavailability.
Truncation cannot improve a complete-stream score. Missing or empty populations
do not meet the empirical 2 cm / 2 degree pose allocation. Saved row errors,
summary counts, means, quantiles, maxima and allocation booleans are compared;
all qualification flags remain false. Pose scalar comparisons use 1e-7 absolute
tolerance, with exact population, boolean and field checks.

## Verification scope and remaining work

The initial 48 synthetic numerical tests passed. They exercised both turn
directions, early and partial stops, wrong-way/stationary motion, path versus
displacement, whole-second stopping, 3D noncommuting rotations, near-zero and
near-pi angles, tracker loss, bridging, and deliberately corrupted results.
The final four-file regression passed **142 tests** in 112.96 seconds. An
independent JUnit parse confirmed 142 tests with zero failures, errors or skips.
It includes 51 new numerical/integration tests and the existing challenge,
base-scoring and stress-cohort tests. Source hashes were unchanged through the
run. The JUnit artifact is
`.generated/navigation-development-staging.m6MDz1/independent_tracking_numerical_verification_v1.xml`
(SHA-256 `3deab93f26739fd08721bd9115e2c914cb5d227f935cbac84aa75c2465334af7`).

The integration fixture uses the real paired observers and production scorers
for all eight base plus 88 stress streams, but its tapes contain only three
synthetic frames. It tests numerical/schema compatibility; it does **not**
reach the stress onset, simulate physical movement, or prove robustness.

This module does not reconstruct observer inference, complete reference/rejoin
semantics, actual stress injections, raw contact/geometry/visibility evidence,
or the 48 predecessor comparisons. It also cannot establish source/data
authenticity on its own. The complete-result verifier must still join those
pieces, preserve all failures and distinguish exercised interventions from
scheduled-but-unreached ones before any adoption decision.

The fixed challenge remains unlaunched. Reliable closed-loop turns/returns,
low-friction execution, full-loop timing, predictive planning benefit, online
memory/backtracking and novel-maze mission success remain unproved. These
numerical checks are preparation for measuring those outcomes, not substitutes
for them.
