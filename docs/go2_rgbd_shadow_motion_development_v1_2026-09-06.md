# Fresh RGB-D shadow-motion V1

One new three-arm physical acquisition, output
`.generated/go2_rgbd_shadow_motion_development_v1_attempt_001`. Preserve all
predecessors. Use the verified fresh6m enclosure+angled partition, spawn
(-.25,-.2,.375), heading0.27, physics/topology2026090606, appearance2026090607.
Arms neutral/repeated/distinctive share collision shapes, gait and fixed tape.
This is one fresh layout with matched appearance conditions, not three independent
layout trials, not B recovery and not a learned navigation policy.

Keep original frozen gait checkpoint/config and checkpoint gains, 2ms physics,
20ms body sensing/policy, 100ms commands/images and500Hz gyro. Fresh initializer
changes only scene construction. All visual-only floor/wall links are excluded
from physical contact attribution. Use actual body-mounted calibrated optical
camera pose, separate single-sample RGB/depth renders, immutable static visual
triangle checks and finite-floor-domain checks; no old pixels or native pose
input to fusion. Raw native trace is evaluator/simulated-sensor source only.

Settle15zero ticks. Independently check the new initial velocity0±.02m/s prior,
27padded body primitives, supplied nonfloor box[-1.25,1.25]^3 in initial-body axes
through7.5s, native foot identities and instantaneous four-group support.
This supplied region is bounded collection supervision, not a maze map or a
deployment observation. After admission, stop physical acquisition on disallowed
contact, stability violation, nonfoot-ground contact, speed>.3m/s, padded-region
exit or expiry. Guards are evaluator-owned and count as external supervision.

Fixed50tick tape:6forward(.08,0,0),4zero,6yaw(0,0,.3),4zero,
6reverse(-.06,0,0),4zero,6yaw(0,0,-.3),4zero,6forward(.10,0,0),4zero.
Then three actual zero-command tail ticks. Expected3400physics samples and
54RGB-D frames if complete. Both reverse and both yaw signs are included.
Zero target is not rest; independently score the full final100ms twist and tail
displacement. No shadow result changes, shortens or extends the fixed tape.

Run the frozen complementary RGBD/ray-memory owner in shadow from1.5s. Explicit
point hypotheses2mm step/10mm agreement reproduce the already tested development
example; they are not calibrated, hardware-valid or action-admission thresholds.
No fitting, tuning or promotion based on this run. If the shadow estimator or
its budget fails, record its terminal failure and never invoke it again. The
independently guarded data collector may finish the declared tape; every later
frame is labelled NOT_REINVOKED_AFTER_SHADOW_FAILURE. No stale state is a success.

Persist raw physics/contacts, all body/gyro samples and histories, actual RGB-D,
native/static/actuator identities, command acknowledgements, shadow tape and full
outer tick timing including execution/capture/observer/bookkeeping. Source/input/
native/OpenCV identities are checked before and after; all partial outputs and
missing expected artifacts remain visible on failure. No retry, resume, threshold
relaxation, hardware actuation or final-benchmark access.

Independent audit reconstructs sensor packets from raw native trace separately,
then replays the estimator with sensor-only inputs, checks identical prediction/
failure chronology, all physical guards/tape/stop outcomes, RGB-depth calibration,
new visual roles and matched-arm physics. Only afterward score position drift,
point displacement errors and original depth ranks. The errors are development
evidence, not an independently calibrated uncertainty bound. Next validate the
error/relative-transport model on separate motions/layouts and integrate complete
discovery/marker/return, genuine predictive-training/rollout/memory comparisons
and bounded hardware when available. Whole scientific goal remains unachieved.
