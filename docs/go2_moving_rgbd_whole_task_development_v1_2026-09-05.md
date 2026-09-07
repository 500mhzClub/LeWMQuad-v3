# Moving RGBD relative state inside the continuous mission, V1

## Purpose and fixed population

Integrate verified single-sample depth and causal gyro-relative motion into the
continuous exploration/discovery/return execution path, initially as an observer
that cannot change commands. This moving-state measurement check precedes using
its estimates for arrival or turning. It is not another stationary population,
a navigation improvement claim or a matched JEPA comparison.

Run exactly the north_dogleg and south_branch development layouts from the
failed whole-task pilot, using its fixed_forward controller and episodic memory
arm, same seeds, geometry, initial state, gains, gait, timing, budgets and release.
Only scene IDs and output root differ. The collector function and controller
are unchanged; its session subclass adds actual depth and observer acquisition.
No retry, old outcome change, hidden pose/controller geometry, teacher route or
new learned checkpoint is introduced. The maximum remains360 s per mission.

Root: `.generated/go2_moving_rgbd_whole_task_development_v1_attempt_001`.
All new sources and predecessor bindings are fixed by the launch record.

## Measured state

Use the verified actual RGB render plus depth-only render at the same camera
transform and physics time. Retain raw optical depth, causal policy depth/masks,
floor visual/collision identity and actual framebuffer readback for every capture.
Ordinary body sensing and fast500-Hz gyro histories are unchanged. Depth remains
an ideal simulated modality without a calibrated hardware counterpart.

The local surface observer preserves body-height wall support, unknown and
unmodelled returns, depth discontinuities and conditional weak directions.
Separately, the motion estimator uses full-view depth at stride8, with normals
from neighbouring finite returns. Reject neighbourhood edges over0.25 m and
nonplanarity over3 mm. Rotate current points into previous body axes using only
the causal gyro integration. Match static surfaces within0.2 m, requiring absolute
normal agreement >=.95 and at least100 correspondences. Solve point-to-plane
translation with1-cm Huber weighting for at most15 iterations, convergence10 um.
Weak eigen-directions remain unknown (normal-Gram eigenvalue threshold max.005,
.01 times the largest). Accept the observable projection only if converged,
unweighted correspondence RMS <=1 cm and displacement <=.15 m per100 ms.

These are prospective development assumptions, not calibrated probability or
uncertainty bounds. Do not use command integration to fill missing directions.
A missing full translation invalidates cumulative position for the rest of that
episode; it does not silently reset to zero. Point correspondences can be wrong
in repeated or dynamic geometry even when the solver converges. Native contacts
between policy ticks retain terminal RGBD but do not invent a full decision
history or update odometry from an incomplete gyro interval.

## Fixed measurement endpoints and full audit

Evaluate against raw actual physics only on the evaluator side. Full interval
translation is scored in previous body axes; partial estimates are separately
scored after projecting truth into the declared observable subspace. The moving
state check requires >=90% fully observed intervals, maximum accepted full-step
error <=1 cm, and an unbroken cumulative estimate with final error <=5 cm. A
missing final estimate fails, even if every available short estimate is accurate.
Always report availability and every failure, not just errors on survivors.

Check every moving depth frame at the same native pixel centres, stride8,
2-cm box-edge exclusion, .22–4.98-m eligibility and5-mm tolerance as the completed
interface. Require >=1000 eligible rays. Marker-specific stationary visibility
criteria do not apply to the whole moving scene. Actual visual floor z=-.005 m
is checked against native floor identity, never silently equated with collision0.

The full raw audit reuses the complete original whole-task audit on its exact
artifact subset: physical contacts, gains, command tape/slew, RGB/body/gyro,
camera transforms, controller replay, memory replay and independent discovery/
return metric. It independently checks all added depth artifacts, full live
observer replay, and both measurement reductions. Audit PASS is separate from
moving-state/depth check PASS and from unchanged whole-task success/failure.

## Next decision

If moving-state accuracy/availability is adequate, next use tracked observed
opening boundaries and full articulated-body support for arrival, with full-ray
local clearance and explicit unknown side/rear volume for turns. If it fails,
use the recorded moving evidence to address the observed estimator failure;
do not fit a new threshold and rescore this attempt. Neither outcome makes a
successful scan, beacon acquisition, return, memory advantage, JEPA planning
advantage, independent generalisation or hardware result true by itself.
