# Observation-local mapping body-projection reuse

The completed tiled-controller profile contains 100 calls to `numpy.indices` in
ten late-history decisions. Source inspection identifies repeated identical
multiply-then-divide optical-grid construction in mapping floor coverage,
sampled patches, retained patch append and measured primary-plane points.

Registration and mapping do not generally share a floor-index key: registration
uses the original visual rotation and initial gravity direction, while mapping
uses the registered map transform. Their up vectors and therefore classification
results must remain distinct. The floor-index kernels also use a different
multiply/divide order when constructing optical points. This prototype does not
reuse those kernel results or their body-point calculation.

`lewm/frame_body_projection_cache_development.py` implements a separate bounded
cache of only the mapping `optical @ BODY_FROM_OPTICAL.rotation.T + translation`
grid. The original expression, including multiplication/division order, is
unchanged. Keys include complete depth bytes, shape, dtype, camera-transform
bytes and focal constant. There are at most two entries and every cache is owned
by one observation. Unsupported representations use the original expression;
capacity overflow recomputes; close clears all entries and rejects later use.
Cached arrays have immutable byte backing. This cache performs no validation of
sensor validity, range, pose or floor support; original consumer validation must
remain before its use in any future integration.

Twenty-three tests passed in 1.53 s in session 94146. They verify byte identity
against the original mapping expression for float32/float64/integer/boolean
arrays and contiguous/strided/Fortran layouts, AST identity of the optical
expression in all four consumers, signed-zero and small-input distinction,
source mutation, immutable ownership, bounded overflow, close/new-observation
lifetime, changed calibration, malformed shape and array-subclass fallback.

The source-only component benchmark is
`scripts/benchmark_go2_frame_body_projection_cache_v1.py`, writing exclusively
`docs/go2_frame_body_projection_cache_component_benchmark_2026-09-11.json`.
It uses two fixed-seed synthetic grids, independently for float32 and float64,
with grouped and interleaved call orders. Each synthetic observation makes five
calls per grid, reflecting the aggregate projection count in the motivating
profile; this is not a replay of actual call arguments. Twenty paired repetitions
alternate baseline/candidate order after two warmup pairs. Cache construction,
key copies, immutable backing and close are timed. Every complete output grid is
byte-checked outside timing; source grids must remain unchanged. Preserve every
timing, including any negative result, under shared-host conditions.

No running controller is modified, no model or recorded raw sensor is loaded,
and no additional full CPU replay or native scene is started. No up-vector,
classification, height, projected coverage, retained patch, clearance or mission
calculation is cached by this component. Controller integration must retain each
original validation and geometry expression, check observation-scope closure
and failure paths, and pass complete-history equivalence before any timing gain
is attributed to the controller. No real-time, navigation or hardware claim is
established by this prototype or its synthetic benchmark.
