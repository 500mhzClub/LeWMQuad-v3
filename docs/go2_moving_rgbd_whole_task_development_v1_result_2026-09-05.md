# Moving relative state passes; navigation remains unchanged and unsuccessful

Both continuous missions completed. Full audit37845 passed all RGB/body/gyro,
physical/controller/memory checks and exact live relative-state replay. The
new observer measured motion accurately in both existing development layouts.
It did not influence commands: the original image-change arrival failure and
zero discovery/return outcomes remain unchanged.

## Fixed outcomes

| Layout | Full motion intervals | Worst step error | Worst cumulative error | Final position error | Moving depth frames passing | Whole task |
|---|---:|---:|---:|---:|---:|---:|
| North dogleg |104/104|0.951 mm|0.674 mm|0.139 mm|105/105|Fail|
| South branch |104/104|0.956 mm|0.684 mm|0.164 mm|104/105|Fail|

Both pass the prospective moving-state check: >=90% full availability, maximum
step error <=1 cm and final accumulated error <=5 cm without filling missing
components. These are results for ideal simulated depth and gyro, one short
mostly-forward physical trajectory under two scene configurations. They are
not calibrated uncertainty bounds or evidence of robustness to turns, long
loops, texture changes, dynamics, sensor noise or hardware timing errors.

The two trials have exactly identical physics arrays to their respective
episodic RGB-only predecessors, verified read-only by diagnostic88220 against
recorded artifact hashes. They also reproduce the same physical trajectory
across the two layouts. Do not count them as independent physical replications.
Both run10.4 s after settling, end1.572881 m from home and stop with
FAILED_LOCAL_FAILED_NO_VISUAL_CHANGE, no marker discovery, no return attempt,
no contact, no native body stop and no sensor fault. Whole-task success is0/2.

## One moving-depth failure is retained

North's maximum checked visual-depth error is1.411 mm. South frame75 has one
sampled ray at row316, column436 exceeding5 mm: expected optical depth2.262126 m
to the visual floor, rendered2.231210 m, a30.916-mm difference. All other104
south frames pass. Neither the pixel nor frame was excluded from the recorded
metric, and the study's all-moving-depth claim remains false.

A read-only diagnostic places that rendered ray endpoint at approximately
(3.560000235,-0.728792932,-0.000000173) m: just0.173 micrometres below the
lower edge of two overlapping wall boxes, with x on their front face. The
nominal analytic ray narrowly misses those boxes and reaches the floor behind.
This is consistent with numerical raster/analytic boundary inclusion ambiguity,
not proof of a new uniform depth-scale error. The mechanism is not independently
reproduced here. The unchanged motion normal/edge filter excludes this sampled
ray; its nearest retained motion-cloud point is45.345 mm away. This exclusion
was already in the predeclared estimator, not added after the result.

The original visual/collision floor offset remains5 mm, with actual native
identity checked throughout. Range discontinuities and uncertain silhouette
returns need explicit treatment in clearance reasoning; a perfect-depth or
collision-surface guarantee cannot be inferred from accurate odometry.

## What the live surface observations reveal

North has zero reported discrete depth jumps across105 views; south has29 views
with one jump each. North nevertheless has visible perpendicular wall supports
and corners. Therefore a depth-jump-only portal detector would miss relevant
structure. Fitted line support endpoints can be field-of-view-censored, split
at corners or limited by filtering; they are not automatically wall ends.

The next controller must track visible corners/plane intersections and accumulated
ray evidence across motion, distinguish censored endpoints from observed physical
boundaries, and retain uncertainty about unseen opening extent. This result
supports replacing command-distance proxies with measured relative state, but
does not itself supply whole-body arrival or clearance for a360-degree turn.

## Execution and verification

Collector98173 completed2, exit0. Full audit37845 passed2, exit0, replaying200
original controller decisions and210 new live observer records. Evidence includes
11,900 native physics/fast-gyro rows,1,190 ordinary sensor rows and210 actual
RGB/depth pairs, including the zero-command release. Observer state is computed
at acquisition from policy-valid inputs; raw physics and maze geometry enter
only the independent evaluator.

Final source suite17806 passes1,089 tests across96 explicitly selected files
in75.24 s. Focused65898 previously passed9 tests; the final suite includes the
additional launch-schema regression. Initial full24905 passed1,088 tests before
source review found an inherited launch-schema string; that metadata and its
audit were corrected before execution, without changing scientific parameters.
Final preflight61655 passed272 source,195 input, two gait and ten installed
native source bindings. No bound source changed during or after collection.

Root: `.generated/go2_moving_rgbd_whole_task_development_v1_attempt_001`.
Launch SHA-256: `cd49c6bfb43aba3ddb2d586345e74ccf619342a4bcf5e8b596f08ff0ff29e542`.
Result SHA-256: `68225efeb63ef78a9299b6ec02d40ff69ba5f3d86040a045c2c3cc12877c83e9`.
Audit SHA-256: `d492770e248936e6c4244f16aabf4b71881c4f1eb5b5e0e80d0e023a1991c4aa`.
All12 newly included source/test/protocol/witness paths, including the earlier
local-surface module and report, are now bound. Do not edit or rerun them.

Post-document guard77538 passes272 source,195 input, two gait and ten native
bindings plus the three exact terminal identities, exit0. No experiment, audit,
test or verification process remains running.

Next: a separately named measured-geometry control successor using tracked
opening/corner evidence and accumulated full-view rays for a viable observation/
turning region. Preserve the old0/4 and current0/2 navigation results. Do not
substitute a fitted travel distance or image-change threshold for that state.
Reliable discovery and physical return, matched memory/JEPA comparisons,
independent mazes and real-platform evidence remain required.
