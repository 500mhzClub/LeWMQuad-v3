# Mapping projection component result and next integration

Source inspection rejected a blanket registration/mapping floor-index cache:
registration calls `measured_candidates` with the original visual orientation
and initial gravity, whereas mapping queries use the registered map transform.
Their up vectors need not be byte-identical. Reusing a classification merely
because depth matches would be unsound. No such sharing was implemented.

The mapping methods instead repeat the same depth-to-body projection. The verified
tiled-controller profile has 100 `numpy.indices` calls in ten late-history
observations, including 20 floor-coverage calls, 40 sampled-patch calls, 20 retained
patch appends and 20 `measured_points` calls. Source review confirms the same
optical-grid expression in these four consumers. The floor-index kernel uses a
different floating-point multiplication/division order and is excluded from this
reuse proposal.

## Tested standalone component

- Source: `lewm/frame_body_projection_cache_development.py`.
- Tests: `lewm/tests/test_frame_body_projection_cache_development.py`;
  23 passed in 1.53 s, session 94146.
- Protocol: `docs/go2_frame_body_projection_cache_component_v1_2026-09-11.md`.
- Benchmark: `scripts/benchmark_go2_frame_body_projection_cache_v1.py`;
  session 9115 exited zero, 2,357 source bindings.
- Benchmark result: `docs/go2_frame_body_projection_cache_component_benchmark_2026-09-11.json`,
  SHA-256 `70560685bb2697ec9d4ffcc5b2f015d4063daca9b5e099d9a622cfdef924eaba`.
- Preparation: `docs/go2_frame_body_projection_cache_component_preparation_2026-09-11.json`,
  SHA-256 `6bd6f8fff4a3f20d6574125728c25f02449e1f7f025fd35dc3f72c4aad45d904`,
  2,358 source bindings. No controller integration has been performed.

Each predefined synthetic workload made ten projection calls on two depth grids,
with twenty alternating paired repetitions after warmup. Every complete projected
grid matched byte-for-byte and both source grids remained unchanged. Timings
include cache construction, complete key bytes, immutable backing and close;
output comparison is outside the timing interval. The observed component
total-time reductions were:

| Synthetic workload | Reduction |
| --- | ---: |
| float32, grouped cameras | 73.6031% |
| float32, interleaved cameras | 73.3414% |
| float64, grouped cameras | 56.1575% |
| float64, interleaved cameras | 55.3477% |

These results do not establish a whole-controller gain. The workload reflects
an aggregate projection count, not recorded actual call arguments. Shared-host
conditions remain. No model, raw recorded sensor data or independent layout was
consumed by this benchmark.

## Required integration scope

The next candidate should attach a `FrameBodyProjectionCache` to the existing
single-observation floor-geometry context. It should replace only the repeated
optical/body expression in `FloorFrameGeometry.floor_coverage`,
`sampled_floor_patch`, `append_patch` and the mapping `measured_points` helper.
Calls from `primary_floor_plane` and `confirm_auxiliary_floor` must reach the
same observation-local projection provider. Preserve the original arithmetic
after body projection, including each separate height, floor-plane, up-vector,
coverage, prefix and classification calculation. Preserve the existing tiled
index implementation and its public cache-count receipt.

The helper performs no sensor, validity, range, pose or floor-support validation.
Integration must retain every original consumer validation before cache access.
Tests must cover complete consumer outputs, changed transforms and depths,
failure before/after an earlier successful query, immutable ownership, overflow,
scope closure on exceptions and no retained projection across observations.
Actual public packets and robot-geometry checks must follow. A full-history
paired replay remains necessary before any controller-speed claim or adoption.
Existing executed sources and running controllers must remain unchanged.

## Existing jobs remain active

The freeze-only paired replay is still PID 2871712 / creation 1789141465.68,
session 17139, with completion watcher 2872194 / creation 1789141631.66,
session 59978. Latest complete paired row was frame 618. Do not start another
full CPU replay until this one ends; its watcher will verify completion once.

Extended-budget native worker 2867880 / creation 1789139673.31 is collecting
under original launcher 2843773. Latest complete stream-timing row was tick 1075.
This is live progress, not an audited outcome. Preserve the existing native queue.
There remains no verified round trip, independent-maze reliability, predictive
planning advantage, real-time qualification or hardware result. The full goal
remains active.
