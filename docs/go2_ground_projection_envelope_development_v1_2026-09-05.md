# Fixed ray-projection envelope development diagnostic

## New question

The completed ground-hypothesis study measured body height and normal errors,
not their induced pixel-to-ground point errors. This separate diagnostic asks
how nominal range error depends on viewing distance, and how much observation
is lost or made imprecise when explicitly carrying a plane-error hypothesis.
It does not retry the earlier study, refit its estimator, change any physical
trajectory, train a model, or certify RGB place/exit/arrival or robot clearance.

Use all8 original audited multi-junction routes and all1,498 current RGB/body
packets, including the4 final task failures. Route variants are correlated,
not8 new independent mazes. Initialize the unchanged CausalGroundPlane at the
first actual zero-command packet and propagate its actual causal gyro/joints.
Runtime sees current RGB/body plus fixed camera/robot calibration only.
Independent camera/body poses and wall geometry are evaluation-only.

## Conditional geometry, not calibrated uncertainty

For a unit plane normal n within an angular cap around nominal n0, and body
height h in a declared interval, a calibrated body ray r from camera origin o
intersects the plane at optical depth lambda = -(h + n·o)/(n·r).
Compute exact minimum/maximum dot products over the cap for both o and r.
Bound numerator and denominator independently, making a conservative ratio
interval when camera height stays positive and every ray direction in the cap
points downward. Abstain when the ray could face the horizon/upward, or the full
interval is not within the calibrated0.05–200 m optical clip. Never turn an
unbounded ray into a finite safe distance by clipping it.

Fixed sensitivity grid: height radii0.01,0.03,0.05 m × angular radii0.025,0.05,
0.10 rad. These values are motivated by previously observed development errors;
they are not preregistered against those old errors, held-out calibration,
probabilistic coverage levels or guaranteed hardware error bounds. Report every
cell; do not choose a winning radius or tune it after this ray-level result.
The camera lever arm is included, and ignored normal/height correlation can
make the intervals unnecessarily wide. Camera/extrinsic errors, non-flat terrain,
palette mistakes and unobserved overhang/body volume remain outside this model.

## Fixed pixel evidence and outcomes

Native640×480 RGB, stride8 pixel centers, identical to the previous analytic
floor reference. Use the unchanged greenish-floor palette baseline and unbroken
bottom-connected columns. Negative pixels mean unknown. Visible floor is not a
swept-footprint guarantee, nor does a point behind an occluder count as visible.
Evaluate all reference-valid pixels; report near-clip ambiguity separately.
Keep positive-but-nonfloor pixels and any accepted false-surface projections
in explicit counts rather than deleting them from the reported population.

Independently ray-cast actual camera geometry against the known ground and
audited wall boxes. Check current body-to-optical mount against actual camera
pose. For nominal projections on actual visible floor, report point error sum,
mean, maximum and counts exceeding0.05,0.1,0.25,0.5 m. Stratify by true optical
depth [0,0.5),[0.5,1),[1,2),[2,4),[4,infinity). These evaluation-only bins never
enter runtime or select observations. Keep a missing nominal intersection as
missing with an explicit valid-point denominator.

For each uncertainty family, report accepted visible-floor rays, false-surface
acceptances, abstentions, true-range coverage and interval widths, including
counts with width at most0.1,0.25,0.5,1,2,5 m. These are descriptive observation
widths, not control thresholds. Report whether the actual plane lies in the
declared family for each frame. If it does and an accepted true-floor ray falls
outside its interval (absolute depth tolerance1e-10 m), fail the implementation
check. Empirical misses when assumptions fail remain scientific observations.
Do not confuse conditional geometric containment with empirical sensor coverage.

## Execution and integrity

Exact fresh root:
`.generated/go2_ground_projection_envelope_development_v1_attempt_001`.
Runner: `scripts/analyze_go2_ground_projection_envelope_development_v1.py`.
Require exact completed ground/floor launch and result identities, their bound
source/input dependencies, and the original route source/result/audit bindings.
Validate every actual policy/RGB/raw-camera/pose file against the route result.
Reconstruct source route specs and require equality to the originally launched
specs before output creation. No protected inputs, source export or new physics.

Bind new runtime, runner,19 synthetic geometry/accounting tests and this protocol
before analysis; reverify all sources/inputs at completion. Store per-frame and
per-route accounting for all nine sensitivity families. Sensor failures retain
the affected route and explicit unprocessed-frame count. Unexpected integrity
failure produces retained FAIL; no retry or source edit after launch. This study
may use one CPU thread while the separate fixed learning comparison trains;
do not alter that training's29 bound source paths or read/select interim outcomes.

## Navigation consequence

Use these results to judge whether projected floor is a useful local observation
component and where it is too imprecise. No envelope cell licenses an edge,
place merge, beacon identity or physical traversal. The next actual navigation
integration must combine observation reliability, footprint/occlusion evidence,
measured execution and explicit arrival/localization uncertainty. The palette
shortcut's appearance failure remains a transfer limitation regardless of range
accuracy under this renderer.
