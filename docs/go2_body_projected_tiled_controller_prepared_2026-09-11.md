# Observation-local body projection: controller and replay harness prepared

The lower-level geometry implementation now reuses each camera's optical-to-body
projection within one map observation. All original depth validation, pose
validation, floor classification, height bands, plane fitting, coverage checks,
retained patch prefixes and floor-evidence receipts remain in their original
order. The six explicit source derivatives are recorded in
`docs/go2_body_projected_floor_geometry_source_derivatives_2026-09-11.json`;
tests compare their executable syntax with the original methods after only
those declared substitutions.

`BodyProjectedRecordingFloorGeometry` composes the new geometry with the original
recording context and tiled floor index. The original `LaterResolvedFloorMap.observe`
body is privately bound to this context, retaining its exception and finally
paths. Both the index cache and body projection cache close at observation end.
The existing public index-cache counts remain unchanged; projection counts are
internal. A fresh-map installation preserves every existing map field object,
memory alias, registration, selector, residual, history and patch store.

This candidate is independently based on `TiledDensityProgressiveFloorController`.
It does not adopt the separately running primitive-leaf freezing experiment.
Registration and map floor classifications are not shared: their up-vector
dependencies differ. The floor index's floating-point unprojection expression
also differs from the mapper's expression and remains unchanged. Only the
mapper's identical full-image body projection expressions are reused.

## Validation completed

- Geometry suite: 50 passed in 7.47 seconds, tool session 66111, exit zero.
  Complete output and retained-prefix comparisons cover floor heights, poses,
  repeated and changed depth, and all five consumers' original validation paths.
- Controller suite: 11 passed in 5.05 seconds, tool session 82271, exit zero.
  Real public packet fixtures and articulated robot footprints match the tiled
  baseline. These are fixture observations, not a saved native sensor stream or
  model-inference comparison. Successful observation cleanup and an injected
  late mapping failure both empty and close the caches. Initial state, aliases,
  original sensor failure and nested decision evidence are preserved.
- Full-history harness suite: 16 passed in 4.88 seconds, tool session 64268,
  exit zero. Synthetic 1,428-row fixtures exercise alternating execution order,
  separate model instances, seven original retained-state witnesses and
  corruption rejection. No actual full-history replay was executed here.

The standalone projection component's earlier 23 tests and synthetic benchmark
remain recorded in `docs/go2_frame_body_projection_cache_component_preparation_2026-09-11.json`.
Its synthetic speed reductions do not establish an integrated controller gain.

## Next actual experiment

Prepare and check an exclusive paired launcher for
`scripts/body_projected_tiled_replay_development.py`, comparing this controller
with the completed tiled baseline on all 1,428 original observations and 1,425
forecasts. Its reserved output is
`go2_body_projected_tiled_late_history_v1_attempt_001` under the existing artifact
root. The harness has no launch entry point or input-admission authority.

Bind the baseline's actual result
`9f83c8e10db48c31918d386ba4867818a3e143625dde191e565aa6b796fb7307`,
launch `8ad6b00ce598b26b2058c9c904a51a908f2f63f4398b6eb5b46e8484b6ca45dc`,
and completion `543ff2e72fc8f31b1f241f5f905212ca8d952348c3216ca6f73af7666b564556`.
Reconstruct its complete report, all original raw/model bindings, seven state
witnesses and negative sensing scope before and after execution. Preserve the
original ten normalized state type paths; selector class identity remains
outside that original comparison scope. Require actual ended-owner completion
verification, all timings including regressions and over-100-ms counts, fixed
threads, resource checks and no retry or resume.

The existing full CPU replay owner 2871712 was still live while these checks
completed; last observed paired frame was 1241. Its completion watcher 2872194
owns final verification. Do not launch this new full replay or a duplicate
checker while those processes are live. The extended-budget native worker
2867880 was also live, last complete timing row tick 1739. Those progress
observations are historical, not completion evidence. Recheck authoritative
process identities before acting.

No queued native controller was changed. This work establishes neither a new
navigation outcome nor real-time, independent-maze, JEPA-advantage or hardware
qualification. The full navigation goal remains active.
