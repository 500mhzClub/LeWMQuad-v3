# Fusion-aware ray memory: implemented prototype and integration limits

This turn implemented a fusion-aware memory consumer and completed its focused
and regression tests plus read-only actual-packet diagnostics. The preceding
moment-fusion goal turn was progress; this turn adds actual uncertainty-dependent
clearance behavior. The prototype is not yet connected to navigation and is not
a launched, source-frozen scientific experiment. The prior 333 launched sources
and all completed results remain unchanged.

## Implemented behavior

`lewm/uncertain_ray_memory_development.py` owns a moment-aware integrator. Its
input remains the original depth observer record: identity, RGB/depth hashes and
timestamps are checked; predicted motion is never disguised as rank-3 depth.
The consumer latches faults and rejects pose-proxy budget exhaustion before
adding another usable view. Subsequent queries cannot use stale retained evidence
after the fault. This is now an operational consumer check, not merely a reported
`usable` flag that a caller can ignore.

For historical views, the current implementation uses the sum of endpoint
position-error scales plus endpoint orientation scales times a conservative
lever arm. This is a conditional envelope calculation using **uncalibrated
proxies**, not validated covariance or a deterministic bound. Current-frame
pose transport cancels exactly and receives zero historical transport radius.

Each transported point's pose ball is enclosed in a box and projected through
the fixed camera model. The query checks every sampled depth pixel in the
resulting rectangle, not just the nominal ray. All required pixels must be valid
before that view can approve free space. An observed nearby surface remains a
conflict even if another part of the rectangle is missing or outside the image.
The latest stationary observation can contradict an older keyframe.

Ground support stays separate from free space and is allowed only for explicit
low-body support roles. Per-pixel normals and local planarity are computed
independently of query radius. The full query window must meet those predicates,
the approximate gravity-height consistency check, and the height tolerance with
the declared radius included. Retained static-scene evidence remains bounded to
64 keyframes plus the current frame. No continuous-volume, future-gait, hardware
or calibrated-uncertainty certificate is returned.

## Tests and actual-packet evidence

- Focused session 21491 passed 21 tests. These cover off-ray thin obstacles,
  missing lateral pixels, monotonic loss of free/support evidence as radius
  grows, preservation of conflicts at image boundaries, randomized projection
  cases, empty queries, ground roles, depth/prediction separation, identity/
  history faults, current-view cancellation and latched budget stops.
- Full session 98036 passed 1,255 tests across 115 files in 84.63 s. No source
  edits occurred during that full run.
- Read-only diagnostic 6813 processed 181 original north depth-floor-hold RGBD/
  depth-observer records, with predecessor source/input/artifact bindings checked
  before and after. This is integration/latency evidence, not a newly executed
  trajectory or registered evaluation.
- Diagnostic 48326 processed the recorded narrow-depth cases with no added bias,
  0.02-m/s² bias, and 0.2-m/s² bias through this consumer. All three latch at the
  recorded proxy-budget boundary, tick 93. This does not repair the previously
  documented large-bias proxy exceedance at tick 85.

## Two unresolved integration limits

### Query latency

Initial actual-packet query times at ticks 80/140/180 were approximately
232/384/406 ms. Vectorized projection and rejection of envelopes too near the
camera to intersect any valid return reduced them to approximately 78/155/169 ms
in diagnostic 39159, with unchanged unknown/conflict counts. These timings cover
the memory query alone, not the full camera/state/geometry/control pipeline.
Memory observation itself was about 16.5 ms in the first diagnostic. The later
queries still exceed the 100-ms control period; no real-time claim is justified.

### Ground-support rejection under historical pose envelopes

At tick 180 there are 931 nominal volume samples, including 405 ground-support
roles. The new query rejects 256 samples, all ground-support roles. Every
non-ground sample is supported. The largest historical transport radius is
approximately 39.51 mm. This is not a successful clearance decision.

Read-only diagnostic 71620 separately queried the same evidence at zero transport
radius: all 931 samples were supported and there were no conflicts. The runtime
radii and source were not altered by that diagnostic. It identifies the pose-
envelope contribution to the rejection; it does not authorize zero uncertainty
or a relaxed ground tolerance.

The sum of absolute endpoint scales does not exploit errors shared by poses
along one integrated history. Some common reference error cancels in relative
transport, but the correct remaining uncertainty depends on the estimator's
composition, gyro transport, weak-interval velocity and bias. Simply subtracting
the current scalar variance proxies as though they were independent calibrated
covariances would introduce a new unjustified assumption. Likewise, a current
floor plane is a useful measured height constraint, not automatic proof that
unseen ground is present or that an old ground patch is still associated.

## Next actions toward actual navigation and the JEPA question

1. Derive and implement explicit interval-relative error accounting, including
   shared-reference cancellation, gyro/lever-arm effects, weak-direction velocity
   and persistent bias. Test exact cancellation and adversarial correlated errors
   against an independently integrated reference. Evaluate relative point/plane
   transport on actual packets. Do not silently replace the absolute-envelope
   contract with an independence assumption or erase unknown-bias failures.
2. Use measured current/retained floor constraints where their association is
   supported, preserving ground-coverage evidence and discontinuity/step/hole
   negatives. Recheck all nominal samples; an unsupported result must remain
   unsupported until additional evidence or a justified model changes it.
3. Profile and optimize exact rectangle reductions/retained-view processing;
   keep the missing-pixel, thin-obstacle and conflict behavior unchanged. Check
   the complete observation-to-command latency, not only an isolated query.
4. Integrate the resulting consumer into a separately named controller with an
   explicit fused-speed interface and information-seeking observation actions.
   Preserve original depth records, current physical guards and whole-task
   metrics. Account for measured yaw-gait translation when choosing a view or
   repositioning. Execute fresh complete missions once the integration is
   meaningful; another estimator-only repeat is not the navigation milestone.
5. Give action-conditioned supervised and JEPA prediction the same sensors,
   candidate tapes, training data and budgets. Test whether predicted turn drift
   and future observation quality change actions and physical outcomes. Then
   require actual return-memory and genuine multistep-rollout comparisons,
   independent layouts/seeds, robustness and bounded real-platform evidence.

Whole-task success remains 0/2 in the latest executed simulation. There is no
new learned navigation, JEPA-benefit, independent-maze or hardware result here.
The full goal remains active, with meaningful implementation work available.
