# Explicit coverage-status successor to failed frozen-pose transfer

The original frozen transfer attempt is terminal and immutable. Its collector
coupled pose replay to a diagnostic floor query; the latter raised
`proper finite relative transform and supported backend required` after the
fit-frame-200 progress marker. No predictions/evaluation were persisted. Do not
infer an exact terminal frame or pose score from that truncated progress log.

Original output: `.generated/go2_longer_motion_frozen_pose_development_v1_attempt_001`.
Launch SHA-256 dda0037e6cef3f990492b866ca0705ad768e76f2ba89d8cf95440455aa7229c7;
failure SHA-256 64f7578718d02f2ca85d76b420d04657287553bc0ab815354f0152d36cfb2548.

The pose estimator accepts proper-rotation numerical defects up to 1e-8, while
the unchanged floor consumer requires 1e-12. This mismatch must be exposed,
not silently normalized, threshold-relaxed or treated as observed clearance.
The original transfer's inability to record surviving pose states is an
experiment-coordination failure, not a measured estimator failure.

This separately named diagnostic keeps the exact same four nominal estimator
instances, datasets, model rules, native scorer and coverage predicate. It adds
ONLY explicit per-frame coverage outcomes: pose unavailable, surface unavailable,
contract rejected, or conditional zero-additional-error coverage. A coverage
SensorContractError records the reason and rotation numerical defects, returns
unknown coverage and does not terminate an independent pose estimator. Unexpected
exceptions still terminate the experiment. A failed pose model is never reinvoked.
No fallback projection, pose reset, geometric tolerance change or motion occurs.

Predictions and coverage statuses for both trials are written before native
scoring. The fit/validation estimator comparison remains the one reserved before
collection; no coefficients, thresholds, variants or results-based selections
are introduced. Native coverage mismatches are scored only when both queries
exist; report rejected and missing queries separately, not as correct negatives.
Any missing coverage is ineligible for action permission.

Bind this protocol, coordinator, coverage helper and focused tests along with
the original launch/failure and all inherited sources/inputs before launch.
Exclusive output:
`.generated/go2_longer_motion_frozen_pose_coverage_status_v1_attempt_001`.
This is an explicit failure-accounting successor, not a silent retry, not an
additional independent scientific trial and not a physical-collection extension.
No launched predecessor edits, sealed access, threshold relaxation, model fitting,
navigation, hardware actuation, or recovery of the failed attempt is permitted.

After the result, independently diagnose any numerical accumulation or coverage
disagreements before changing a pose/geometry interface. Relative uncertainty,
prospective gait/braking, memory and the full matched novel-maze JEPA science
remain unfinished regardless of this diagnostic's outcome.
