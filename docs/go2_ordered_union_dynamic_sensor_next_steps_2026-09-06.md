# Next execution: moving sensor pairing before independent-layout learning

The static RGB draw-order mechanism is now reproduced and corrected under fixed
order. The next experiment must exercise mounted sensing during actual Go2
motion, not accumulate more static demonstrations. The full objective remains
independent-layout JEPA prediction, useful online rollout and memory, reliable
novel-maze execution and deployment-valid sensing/hardware.

## Bounded dynamic pilot to implement and freeze

Use the existing l00 inventory definitions without changing layout roles, spawn,
physics seed, appearance seed, command histories or action durations. Use new
union visuals with explicitly fixed floor-first native JIT ordering. Retain the
same native Go2/gait, collision geometry, 5mm camera near plane, native stops,
actual mounted poses, sensor timing and command-independent collection.

Eight distinct scene builds are proposed: two repeats of each of these four
inventory conditions, in a prospectively fixed order:

- `l00_junction_recent_forward_nominal_a0`
- `l00_junction_recent_forward_nominal_a3`
- `l00_near_wall_recent_forward_lower_friction_a0`
- `l00_near_wall_recent_forward_lower_friction_a1`

The junction pair exercises the motion/context of the retained silhouette
failure; the near-wall/low-friction pair adds a context absent from the stopped
batch and can expose different dynamics or contacts. Keep any setup failure,
early stop and missing observation. Do not replace either with an easier scene.
Repeats are repeatability diagnostics, not independent layouts or training data.

Implement distinct initialization/session/collector/auditor sources, preserving
the frozen ancestors. Before launch bind source/native identities and the exact
eight specifications, artifact roster, exclusive output root, 1GiB total budget,
40GiB free-space reserve and terminal failure behavior. No hardware control,
learned-model fitting, new layout role or later layout launch is included.

## Measurements and decision

Record and verify native renderer ordering after every RGB/depth capture; also
record actual native raster precision/sampling state. Require exact native and
full sensor-history prefixes across candidate actions and exact complete streams
between repeats of the same condition where observations exist. Report missing
comparisons separately, not as matches. Reconstruct physical setup, command tape,
native contact outcomes and censoring from raw observations.

Retain the unchanged strict physical visibility score on every frame. Separately
report the new geometry-only pixel-footprint diagnostic: stable-interior error,
boundary uncertainty/coverage, and near-plane failures. This diagnostic does not
certify boundary pixels or license training. Freeze the measurement contract
before launch; do not create a passing gate by masking newly observed failures.
Future policy-side boundary handling must use deployment-valid measured sensors,
never evaluator geometry, and must not interpret ambiguous pixels as free space.

Only after this native evidence is reviewed should a distinct full collection be
frozen over the existing independent roles. Then execute the already prepared
matched three-seed direct/supervised-rollout/JEPA study, action/time baseline and
RGB/history/action ablations. Actual positive hazard coverage, robust execution,
rollout and memory/backtracking benefits, real-time operation and hardware remain
required scientific work, not consequences of a passing sensor pilot.

This is an implementation plan, not an executed or launch-frozen experiment.
