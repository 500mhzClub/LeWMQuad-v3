# Residual clearance is now connected to retained depth

The new articulated-configuration consumer combines the supplied starting-region
condition with actual depth evidence for uncovered boxes. Its recorded queries
produce both positive residual clearance and retained negatives. This is a
sensor/evidence interface, not a learned gait predictor or executed navigation.
No new physical mission or training ran; full-mission success remains **0/2**.

## Implemented behavior

`lewm/observed_setup_configuration_development.py` accepts a fresh continuous
state owner and a supplied articulated configuration. It constructs all 27
physical primitives, expands the initial-frame query by the declared point
uncertainty and 4-cm padding, and partitions it against the finite starting
region. Every whole physical query is inspected in every retained depth view,
even when the starting-region condition covers it completely. Every residual
box needs positive complete-box evidence from at least one retained view; merely
seeing part of it, an occluder or no intersecting return does not establish it.

Prepared/raw depth hashes, measured plane-hypothesis identities and observation
clocks are checked. Per-piece positive witnesses and per-shape obstacle/ground
veto witnesses retain their observation times. All observed whole-query vetoes
override supplied clearance and positive evidence from other views. A conflict
in an enlarged residual enclosure blocks that residual's positive evidence; it
is not misrepresented as a direct observation of the smaller physical shape.

Initial-frame configurations and current-body-relative configurations have
different uncertainty semantics. Arbitrary initial-frame boxes use conservative
stored-pose allowances. Current-body-relative whole queries preserve the existing
common-pose cancellation in the current camera and the existing relative
transport allowance for older observations. The initial-region partition still
contains global pose uncertainty; that uncertainty is not discarded just because
the current-view relation cancels it. Residual boxes fixed to the initial-region
boundary retain conservative absolute-pose accounting.

The consumer reuses the physical primitive/plane relation checks. Observed
plane-family returns can be exempted only for a separated non-foot primitive or
an exact foot primitive straddling that plane under the supplied error model.
Observed whole-footprint coverage and a compatible foot relation yield only a
contact *candidate*. They do not establish ground-support/contact permission.
The query explicitly denies execution-prediction, swept-volume, future-gait and
navigation-action qualification.

## Fixed body-axis recorded diagnostic

The bound continuous owner reproduced the 12 startup decisions and all 15
relative observations from the saved trial, consuming the three stopping-tail
frames without restart. At 2.9 s, five rigid translations of the measured final
posture were queried through 3.3 s; the original setup expiry remains 3.5 s.
These poses were not executed and do not bound the space between them.

| Forward offset | Conditionally clear primitives | Required residual boxes | Residuals with depth clearance |
| --- | ---: | ---: | ---: |
| 0 m | 27/27 | 0 | 0 |
| 0.25 m | 27/27 | 0 | 0 |
| 0.50 m | 27/27 | 0 | 0 |
| 0.75 m | 19/27 | 8 | 4 |
| 1.00 m | 20/27 | 15 | 13 |

At 0–0.5 m all positive clearance is conditional on the supplied starting
region, not newly observed ground. At 0.75/1.00 m the front lower-leg primitives
and foot spheres retain obstacle vetoes. The 0.75-m query also has an FL-foot
plane-penetration flag; the 1.00-m query has FL/FR-foot flags. These are geometric
query outcomes under the declared plane/error model, **not physical collisions**.
All five compiled/reference query results match exactly. Partial compiled query
times ranged from about 24 to 198 ms while the regression ran concurrently;
these are not isolated full-loop timing measurements or real-time qualification.

## Separate exploratory gravity-tangent diagnostic

The body-axis negatives motivated a separately declared diagnostic using the
repository's existing sensor-derived `gravity_basis`. The measured body-forward
axis has an up component of -0.0161964: translating its fixed posture by one
metre moves it down by 16.1964 mm relative to estimated gravity. The tangent
forward direction has up component approximately 1.53e-17.

The original body-axis results were retained. Tangent queries produce 27/27
conditional clearances at 0–0.5 m, 20/27 at 0.75 m and 19/27 at 1.00 m. Residual
coverage is 4/8 and 10/15 at the two farther offsets. Foot-penetration flags are
absent in all five tangent configurations, while four front lower-leg vetoes
remain at the farther offsets. At 1.00 m, FL/FR feet become observed contact
candidates, still with no contact/action permission. All five tangent
compiled/reference outputs also match exactly.

This explains one geometric contributor; it does not validate a command-to-
motion model, show that the leg vetoes are physical obstacles, or establish a
continuous traversable route. A fixed pitched posture translated in space is
not a walking quadruped. The tangent diagnostic is exploratory on the same data,
not independent evaluation or a replacement for the original negative result.

## Verification and custody

- 26105: 16 initial focused tests passed in 12.64 s.
- 94961: 17 passed, one failed because the new test subtracted a float from a
  geometry list. The fixture was explicitly converted to arrays; no threshold
  or scientific acceptance condition was relaxed.
- 89382: all 19 focused tests passed in 14.85 s.
- 6658: **1,745 tests across 141 explicit files**, passed in 118.25 s.
- 60658: fixed recorded configuration diagnostic completed, exit 0.
- 36076: separate gravity-tangent diagnostic completed, exit 0.

Tests cover region-plus-observed residuals, missing pixels in all views,
occlusion, out-of-view queries, an observed wall overriding a deliberately false
starting-region claim, expiry, submerged feet, stale/faulted/unbound sources,
compiled/reference equivalence, inverse-pose enclosures and common-pose
cancellation without shrinking the global setup error.

Each diagnostic verifies the frozen 385-source/inherited-input/74-artifact and
16-native bindings plus three fixed result identities before/after replay, and
separately binds its nine or eleven explicit development sources/protocols.
No tested source was edited during a live test or diagnostic. The two diagnostic
scripts are executed read-only checks, not additional files in the 141-file test
suite. All listed handles are terminal. Frozen physical trial sources, inputs,
protocols, result and raw audit remain unchanged; no sealed access occurred.

## Next work toward the complete mission

First trace the four remaining lower-leg vetoes to their individual observed
plane relations and pose-error terms. Distinguish a measured non-floor obstacle
from an unresolved ground-intersection possibility; do not call either a physical
collision without execution evidence or simply drop the veto. Establish the
support/contact model as a separate channel from non-floor clearance.

Then implement action-conditioned body/foot configurations **over time**, using
the observed-gravity command basis and measured command history. Validate their
prediction errors, between-sample swept motion and braking against actual
development execution, keeping current posture and empirical model coverage
distinct from guaranteed reachability. Connect that action/evidence interface
to the continuous owner and complete discovery/marker/return controller in a
fresh declared mission. More geometric pose probes cannot substitute for this
execution step.

The full requirements remain: matched geometry/supervised/JEPA predictors, real
multistep online rollout, memory ablations, independent layouts/seeds and
robustness, full-loop latency, and bounded real-platform evidence when available.
No new JEPA advantage, learned navigation or full-goal completion is claimed.
