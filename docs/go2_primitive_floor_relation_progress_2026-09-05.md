# Physical primitive floor separation and explicit contact candidates

The previous goal turn made progress on coupled footprint geometry and exposed
the symmetric height-band rejection. This turn implements separate physical
primitive gaps, whole-primitive observed floor footprints, and contact-candidate
semantics. All five new source/test/probe paths are unlaunched. No frozen
controller, uncertainty radius, original threshold or physical result changed.

## What changed

`primitive_floor_relation_development.py` computes the exact nominal minimum
plane gap for each of the verified Go2's 27 collision primitives using its
support function. This is the minimum over the unpadded physical shape, not the
range between its bottom and top, and not a sampled turn-envelope point.

For g0=min_p n·(p−a), normal error en, anchored plane-offset error eb and a
per-primitive point-displacement allowance ep, the minimum residual lies within

    g0 ± [en*L + eb + (||n||+en)*ep],

where L bounds max||p−a|| using the primitive's AABB. This follows by bounding
the residual change at every physical point before taking the minimum. The
actual plane normals are assumed unit. Point allowances must include the
intended kinematic and relative-pose errors; this function does not validate
them. Numerical outward allowance is separate from physical error.

Forty-millimetre isotropic padding is reported separately. Padding overlap with
floor does not establish physical penetration and cannot be spent as a
penetration tolerance. Positive lower gap establishes separation only under
the supplied model; negative upper gap means some part of the primitive is
below the plane under every model in the supplied family. An interval containing
zero leaves contact/intersection possible.

Only the four exact `FL/FR/RL/RR_foot:0` sphere identities are foot-contact
candidates. Calf geometry, arbitrary ground flags, padded envelope samples and
mixed groups do not inherit that role. A possible foot contact is not declared
observed or permitted. Physical separation, complete observed floor coverage,
and non-floor clearance are independent inputs to the conditional combiner.
A missing input prevents clearance. Its evidence provenance is not yet enforced
by an online clock/hash-bound consumer, so no navigation approval is emitted.

`primitive_floor_observation_development.py` supplies the floor-coverage side.
It takes an eligible measured seed cell and the mean relative sensor-frame
transform, encloses each entire physical primitive AABB including its supplied
point/plane/up errors, and checks every projected floor cell and both of its
triangle planes. It does not impose a proximity limit on the primitive's top:
being farther above floor is not penetration. A seed plane cannot bridge a
missing return, step, wall or out-of-camera footprint. This remains conditional
coverage of an interpolated measured mesh, not continuous real-scene coverage
or evidence that the space above that mesh is free of obstacles.

## Recorded result

Probe 14848 completed a read-only reconstruction of the first 181 original
north observations. At tick180, it assessed the 20 measured joint vectors in the
current packet, each placed at the SAME current estimated body pose, as a
posture library. These are not 20 reconstructed historical body poses and not
a future yaw sweep. The plane seed comes from the first retained camera view;
the whole floor footprint of each primitive is then checked independently.

The unchanged transport-radius proxy is maximized over each primitive AABB.
Illustrative normal/up/offset allowances remain 0.002/0.001/1 mm, uncalibrated.
No evaluator pose, map, oracle plane or visual/physical floor-offset correction
enters the calculation.

Results for this one observed posture library and plane family:

- All 540 primitive/posture floor footprints have measured-cell coverage and
  plane-family agreement in the selected retained view.
- All 460 non-foot primitive/posture cases have a strictly positive physical
  minimum-gap lower bound. The smallest is about 9.94 mm at FR_calflower1.
- All 80 foot-sphere cases have gap intervals containing zero. Across them,
  nominal minima are 3.27–5.10 mm and the overall interval extrema are about
  −19.96 to +28.14 mm. None proves physical separation or certain penetration.
- Eight lower-calf primitives overlap the floor with their padded envelopes
  across all 20 joint vectors, despite positive physical gap bounds. This
  illustrates why padding and physical contact cannot share one predicate.
- The four previous padded samples still fail the old ±60-mm band. Their
  coupled lower heights remain positive at 17.22–17.24 mm. Their old result is
  preserved, not rescored as successful navigation.

Non-floor clearance is deliberately false in this diagnostic: checking floor
footprints is not checking walls, overhangs or the entire future body trajectory.
Foot contact remains candidate-only. The selected floor plane/point-error family
is not calibrated merely because it agrees with these observed cells.

The existing policy packet contains gyro, specific force, and 12 joint positions
plus 12 joint velocities; it contains no joint torques or foot-load/contact
measurement. A new contact observer must not pretend otherwise or read privileged
evaluator contact classifications as though they were onboard measurements.

## Verification and chronology

Focused session 46123 passed the initial 29 primitive tests. Initial full session
65576 passed 1,425 tests across 124 files before the observation-side extension.
Probe 3111 completed calculations and final predecessor binding checks but
failed while serializing a NumPy counter. Public gap endpoints were converted
to Python scalars and JSON serialization regression coverage was added.

Focused session 62159 passed 39 and failed one new synthetic penetration test:
the fixture assumed leg-interleaved joint order, whereas the actual packet is
joint-type-major (four hips, then four thighs, then four calves). The fixture
was corrected to the declared ordering; the penetration assertion was not
weakened. Final focused session 32976 passed all 40 tests.

Tests include independent closed-form foot kinematics, exact support under 200
perturbed plane/translation models across all 27 primitives, attained fixed-
normal error extrema, padding independence, physical penetration, foot/mixed-
link identity negatives, required evidence, and JSON boundaries. Actual depth-
fixture tests cover missing returns and steps inside whole foot footprints,
invalid seeds, behind-camera floor, proper transforms and penetration without
contact permission. Boolean non-floor negatives test the combiner only; a new
non-floor observation producer still needs geometric wall/overhang tests.

Probe 14848 verified all 333 predecessor source bindings and bound inputs and
artifacts before and after processing. No experiment directory was created.
Final full session 89732 passed 1,436 tests across 125 explicitly enumerated
files in 90.68 s, with no concurrent source edits. All test/probe handles are
terminal and the tracked worktree remains unchanged.

## Next implementation and scientific milestones

1. Implement a clock/hash/pose-bound observation consumer that keeps measured
   floor, actual foot-contact candidates and non-floor obstacles distinct.
   Establish non-floor clearance with complete ray/envelope coverage and retain
   contradictions from every view; a floor-plane match must never clear a wall
   or overhang. Reuse per-frame geometry/indexes rather than rebuilding them for
   every posture or query.
2. Specify contact admissibility separately from contact estimation. A planner
   can propose a modelled foot contact without claiming it is already measured,
   but its allowed geometry, unilateral constraint, dynamics and penetration/
   compliance assumptions must be explicit and tested. The current 80 ambiguous
   foot cases cannot become clearance merely by changing a flag. If additional
   contact information is needed, use a separately validated deployment-valid
   sensor/observer interface, not evaluator contact labels. Validate the error
   assumptions on distinct development trajectories and sensor disturbances.
3. Evaluate the actual prospective commanded motion, not only historical joint
   vectors at a fixed pose. Integrate explicit fused speed and observation/
   repositioning actions; measure the complete acquisition-to-command loop.
   Previous 139–157-ms subset timing and the 100-ms target remain unresolved.
4. Run a separately named full discovery/return successor, then matched
   supervised/JEPA predictive-training, memory and genuine multistep-rollout
   comparisons on independent layouts/seeds. Neither the 540 geometry cases
   nor repeated use of this north prefix is an independent navigation trial.

Whole-task success remains 0/2. No new learned-navigation, JEPA advantage,
independent-maze or hardware evidence was produced. The full goal remains active.
