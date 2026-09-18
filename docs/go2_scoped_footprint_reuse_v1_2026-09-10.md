# Exact footprint reuse within one selection

The original recovery chain first checks a correction to the first forecast
point, then checks anchored continuation if the first correction does not
produce a better action. Both use the same corrected first XY point and
unchanged first yaw. The surface filter queries only that first point, although
the separate nominal path constraint checks all eight forecast points. Repeated
surface work can therefore be reused without skipping the later path checks.

The new controller calls the complete original selector through a temporary
memory view. It reuses only exact footprint requests in that one synchronous
selection. Keys retain every float64 bit, including signed zero, both XY
coordinates, yaw, timestamp and persistence. The geometry object must be the
same object. Non-Python-float yaw values retain their original numerical
dispatch through uncached queries. No rounding, spatial tolerance, query
approximation or observation-to-observation reuse is introduced.

The supported production integration requires exactly
`MeasuredFloorTransportMemory` and `ArticulatedCollisionGeometry`, with no
instance footprint/support overrides. Other types use the existing frozen
receipt view and forward every query. The reviewed production selection owns
a fixed observed memory and fixed geometry during contact filtering and
recovery. The helper is not a concurrency mechanism or a general mutation
detector. In-place edits to map contents or geometry inside an owned selection
are outside its contract. The original current-memory guard still runs on
every query, including hits, preserving stale-clock and latched-failure stops.

At most 18 complete receipts are cached; overflow recomputes. Exceptions and
unsupported custom/cyclic/non-JSON receipt graphs are never cached. Each hit
constructs an independent frozen container graph. Final public detachment
retains aliases within one receipt but does not create aliases between distinct
query receipts. The cache and target references are cleared on normal or
exceptional scope exit. No cache survives in controller or map state.

This is a separately named development candidate. The frozen original
controller, ongoing late-history profiler, native workers and queue are
unchanged. Source tests cover original recovery gates, exact query distinctions,
receipt ownership, failure behavior, cache lifetime, unknown stateful surfaces,
and actual contact calculations on a synthetic observed dual-camera memory
with articulated Go2 geometry. They are not native navigation evidence.

Before any native use, require paired replay from frame zero with the original
assigned JEPA model, complete decisions and forecasts, original command-tape
endpoints, unchanged public packet arrays and model state, and no gradients.
Rebuild late history rather than inferring its state from saved receipts. Use
the existing first 1,428 observations, covering the fixed early, repeated-hold
and late windows. Preserve the original strict visibility failure at frame
1173 and the zero-round-trip result. Metadata normalization must be limited to
the declared controller identity/optimization flag. Verify retained state at
fixed checkpoints, with no unreviewed state differences.

Assess helper cost and the completed late-history profile before spending the
paired replay. Shared-host timings establish no isolated speedup or real-time
qualification. Keep one native scene, and finish the already running CPU
profile before starting another full replay. The goal still requires useful
closed-loop navigation, matched comparisons, realistic timing and sensing,
and bounded real-platform evidence.
