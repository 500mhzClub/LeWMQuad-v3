# Observation-bound primitive floor and obstacle memory

The preceding goal turn made progress on physical floor-gap semantics. This
turn adds a current-posture consumer with causal observation bindings, separate
floor/non-floor evidence, and all-view vetoes. It also exposes and corrects two
geometric overrejections and adds an exact incomplete-view early-out. This is
an unlaunched development consumer, not a qualified navigation controller.

## Observation boundary and shared work

`PrimitiveObstacleMemory` owns the current measured joint positions, original
nominal depth record input, fused relative pose, and retained depth observations.
It checks fresh depth and joint timestamps and binds the floor cache's depth
hash to the nominal observer record. Sensor/identity/budget failures inherited
from the fusion/ray consumer, changed depth bytes, duplicate records or missing
current joint values latch failure and discard queryable prepared history.
Queries require the current clock and cannot accept an externally supplied
clearance flag, pose or posture. Results bind their contributing view times and
depth hashes. Native scene/contact classifications are not inputs.

`PreparedFloorFrame` retains immutable depth/valid/up copies, a measured floor-cell
index and the minimum valid depth. Both standalone floor queries and the new
consumer use the same implementation. Repeated queries reuse the index without
rebuilding it. This does not remove the older ray consumer's separate geometry
work; the overall pipeline still has duplication and timing problems.

Each view uses the first located observed footprint among the four current
physical foot centres to select its floor-plane family. The same family is used
for complete physical-primitive floor footprints and for classifying measured
ray returns. If there is no seed, no floor classification is invented: nearby
unclassified returns remain potential non-floor conflicts. This seed limitation
is important in the recorded negative result below.

## Non-floor evidence and all-view composition

The checked boxes contain the physical primitive AABBs, the unchanged
per-primitive transport-error proxies, and 40-mm isotropic geometry padding
(once). This padding is not a foot-penetration allowance. The declared range
allowance is 1 mm; normal/up/plane-offset hypotheses remain 0.002/0.001/1 mm.
All remain uncalibrated development assumptions.

For every intersecting camera cell, the consumer checks its four measured
corner returns. They must either belong to the selected observed floor-plane
family or all lie beyond the box with the range allowance. Missing returns
prevent clearance. A return interval that can meet the box and is not classified
as floor supplies a veto, including from incomplete views. A nearer occluder is
unknown rather than free. These are potential geometric conflicts under the
model, not proof that a physical collision occurred.

Floor separation, complete floor coverage and non-floor clearance must agree
within a single view before it contributes conditional primitive clearance.
Every retained view can veto with a potential non-floor intersection or an
observed-floor penetration under every supplied model. Older positive evidence
cannot overwrite that veto. Exact foot-sphere contact candidates remain separate
and never become contact permission. The query assesses only the latest joint
posture at the current estimated body pose, not a future commanded sweep.

The sampled/interpolated camera representation, static-scene assumption and
uncalibrated error sets remain explicit. No continuous physical-scene, contact,
future-gait, real-time, navigation or hardware qualification is emitted.

## Two geometric corrections

The initial rectangle-by-global-depth construction combined extrema that need
not occur together. Probe 56417 reproduced many body and head conflicts on the
recorded prefix. The successor preserves angular/depth coupling: for each pixel
cell, it intersects the optical box with the beam between its four centre rays.
At positive depth z, its x interval is [a*z,b*z], so overlap requires

    a*z <= Xhi, -b*z <= -Xlo,

with analogous y inequalities and the box's z interval. Four linear inequalities
give the beam's depth interval without ray sampling. Four-corner minimum/maximum
depth bounds then remain conservative across that cell. This removes impossible
rectangle/depth combinations without dropping views or reducing padding.

Probe 13110 confirmed fewer body/head vetoes but retained lower-leg vetoes. A
synthetic all-floor partial view reproduced a second issue: the last image row
had been treated as non-floor merely because the cell below it is unavailable.
Boundary returns now use the fully observed incident cell on the inside for
classification. The unavailable outside corners remain unavailable, so this
does NOT make the partial view clear. Image-edge wall and step returns still
veto. Probe 99227 confirmed removal of all vetoes from views with an available
floor seed in the three recorded checkpoints; seedless-view negatives remain.

Finally, an incomplete view whose entire box lies nearer than the frame's
minimum valid depth (including range error) cannot contain a nearby measured
return. It can therefore skip a potentially whole-camera scan while remaining
UNKNOWN. This is especially useful when an old camera origin lies inside a
now-nearby primitive box. Probe 12459's primitive clearance, contact-candidate,
conflict, penetration and aggregate-clearance outputs exactly match probe 99227
at all three checkpoints; only work counts and timings change.

## Recorded consumer results and remaining negatives

All probes are read-only on the first 181 original north packets, with saved
nominal observer records. They do not rerun physics or rescore a mission.
Final probe 12459 reports:

| Tick | Retained views | Conditionally clear primitives | Foot candidates | Vetoed primitives | Query time | Scanned cells |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 80 | 21 | 10 | 0 | 4 | 1,731 ms | 4,666,825 |
| 140 | 29 | 19 | 2 | 6 | 2,698 ms | 7,570,544 |
| 180 | 30 | 21 | 2 | 4 | 3,168 ms | 8,441,233 |

Counts need not sum to 27: some primitives can remain unknown without a veto.
The two candidates at ticks140/180 are the rear foot spheres. At tick180 the
remaining vetoes affect FL_calflower, FL_calflower1, FL_foot and FR_foot, from
the views at 8.2 and 8.5 seconds. Neither view has a located current-foot floor
seed. That explains the missing classification, but does not establish that
the returns are floor or justify deleting them. There are no definite floor-
penetration findings in these three queries and no all-primitive-clear pass.

The earlier corrected query took 3,861/4,821/5,545 ms and scanned
14,322,762/18,662,530/19,571,494 cells. The exact early-out cuts work substantially,
but the query remains orders of magnitude over the 100-ms full-loop target.
The final median observation subset is 53.96 ms. These timings exclude live
nominal depth registration, acquisition, gait and remaining control. Full
regressions overlapped the later probes; this is not an isolated timing study
or a fair full-pipeline speed claim.

## Verification and chronology

- Session 61048 passed 32 focused tests before the image-edge extension;
  11572 passed 34 afterward.
- Session 92131 passed 36 after beam geometry; full 29069 passed 1,462 tests
  across 126 explicit files in94.34s.
- Synthetic commands c8aa89/f2bca9 separated a non-intersecting partial floor
  case from the boundary-floor counterexample. Session 79166 then passed 37
  focused tests, and full14259 passed1,463 in93.97s after the incident-cell fix.
- Session 51278 passed 38 focused tests after the exact early-out. Final full
  session46371 passed1,464 tests across126 explicit files in93.77s, with no
  concurrent source edits. All test/probe handles are terminal.

The 28 new tests include 10,000 sampled optical-box containment checks supporting
the analytic beam derivation, a global-depth false-overlap negative, wall and
horizontal-overhang depth fixtures, missing rays, occlusion, range-error growth,
partial-view/image-edge negatives, observed boundary floor, order-independent
all-view vetoes, shape identities, cache reuse/isolation, stale queries and
observation-fault latching. Synthetic containment samples test implementation;
they do not substitute for the finite-region derivation or real sensor validation.

All four completed probes verify the 333 frozen predecessor source bindings
and their bound inputs/artifacts before and after processing. No experiment
directory was created and original physical results are unchanged.

## Next actions toward execution and the scientific comparison

1. Resolve the seedless-view classification limitation using observed floor
   hypotheses with explicit sensor/posture provenance and error accounting.
   Investigate a query-independent per-frame ground hypothesis checked against
   observed physical geometry/contact assumptions. Do not simply discard the
   two views, call all upward-facing surfaces floor, or borrow a plane across
   uncertain poses without propagating its errors. Preserve wall/overhang and
   missing-floor negatives.
2. Share per-view plane-family masks and use bounded/compiled beam reductions
   to remove millions of repeated Python/array operations. Validate exact
   decisions against this reference, then measure the complete loop. The current
   reference cannot serve as a 10-Hz real-time controller.
3. State and test a modelled foot-contact admissibility rule separately from
   claiming measured contact. Couple it to prospective commanded motion and
   existing physical stop/audit mechanisms in a separately named development
   controller. A conditional simulation experiment need not claim hardware
   calibration, but its assumptions and failures must be explicit. Do not turn
   ambiguous feet into a retrospective mission success.
4. Execute fresh complete discovery/return missions, then matched supervised/
   JEPA predictive-training, memory and genuine multistep online-rollout
   comparisons on independent layouts/seeds. Current-posture geometry tests,
   replay diagnostics and these repeated development views cannot substitute
   for those mission-level results.

Whole-task success remains 0/2. No new learned-navigation, JEPA advantage,
independent-maze or hardware evidence was produced. The full goal remains active.
