# Qualified feature-overlap retention replay V1

Use the exact completed retained-patch direct-039 result and its 147 public
RGB/depth/body/gyro packets. Compare two fresh sequential observers: the frozen
corner-support observer, then the separately named overlap-retention observer.
Reconstruct every one of the 137 available original observer evidence rows
exactly, including terminal failure at frame 136. The remaining ten original
mission rows are drain rows without new observer evidence.

The candidate inherits feature selection, pair matching, rigid registration,
gyro consistency, displacement envelopes, reference choice, eight-reference
capacity, failure latch and ten-frame bridge limit. Only after an accepted
ANCHOR_MEASUREMENT, retain its current view if no existing promotion occurred
and accepted inliers are at most half the selected features in that reference.
The half-overlap rule is fixed before this replay, with no threshold sweep.
Never promote a bridge-only pose. Retain the measured parent pose and all
evidence; no pose reset, inertial extrapolation, native state or command input.
More frequent anchor composition still accumulates uncalibrated error.

Save every evidence row and retention receipt. Open native pose only after each
complete observer replay for evaluator-only XY and rotation errors. Eligibility
for a separately declared native probe requires all 147 current poses, no
terminal failure, maximum XY error at most 20 mm and rotation error at most
0.05 rad. These development diagnostics are not calibrated uncertainty bounds.
The original ten drain frames contain the original stop commands; replay does
not infer any changed controller's unexecuted outcomes.

Verify the full native source/input/artifact bindings before and after, and
bind this source, focused tests and predecessor result report. Use an exclusive
new root with durable terminal failure on error. At least 4 GiB RAM and
40 GiB free artifact storage plus 256 MiB allowance are required. Inspect and
record CPU/RAM/GPU/storage/jobs before and after each arm. Run the two causal
arms sequentially with one thread for comparable uncontended observer timing;
there is one affected history and no independent native job to parallelize.
No training, native execution, old-artifact changes, hardware or navigation
qualification. The end-to-end navigation goal remains active.
