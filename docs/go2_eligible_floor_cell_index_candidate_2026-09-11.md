# Eligible-cell floor-index optimization

The completed receipt-copy profile identified four floor-index constructions
per observation: two during measured-plane registration and two during mapping.
The kernel accounted for 1.34 seconds over ten early profiled observations and
2.24 seconds over ten late profiled observations. These are shared-host timings
with profiler overhead, not isolated costs or additive whole-controller gains.
The profiler files were checked against the completed profile's artifact hashes
before inspecting their call counts and callers.

`lewm/eligible_floor_cell_index_development.py` preserves the original depth,
validity, unit-up, four-corner height, normal length/alignment, and fourth-point
planarity gates. It computes triangle normals only for cells that survive the
preceding gates. Ambiguous floating-point boundary comparisons use the original
dense implementation. Outputs retain the original mask, integer prefix sum,
up vector, ownership, and read-only behavior. No threshold is widened and no
new cell is approved without the original tests.

All 32 focused tests passed in 2.50 seconds, session 35029. They compare complete
output bytes for rooms, missing and unstructured returns, tilted up vectors,
depth discontinuities, float32 and strided inputs, noisy floors, alignment and
planarity threshold neighbours, invalid inputs and output ownership. Two initial
rejection fixtures accidentally rounded float64 boundary values into the valid
float32 range; the fixtures were corrected to use nextafter in their own dtype.
That initial test session (85762) recorded 22 passes and two fixture failures.

The initial benchmark script failed before executing any pair because its
source-discovery call omitted the required inherited-source argument. The
original script and failure were preserved in
`docs/go2_eligible_floor_cell_index_microbenchmark_v1_preflight_failure_2026-09-11.json`.
The V2 script corrects that call, binds the predecessor and failure record, and
completed successfully in session 77149 with 119 source paths. Its result is
`docs/go2_eligible_floor_cell_index_microbenchmark_v2_2026-09-11.json`, SHA-256
`974d4e59ef64c6f7aaa8c783aef9c50ba2166aa9b55d11fdf9faaec4da8801d1`.

Each fixed synthetic fixture used 20 alternating paired repetitions, with
identical output bytes and unchanged input arrays on every pair:

| Fixture | Original total ms | Candidate total ms | Reduction |
| --- | ---: | ---: | ---: |
| Room | 610.14 | 447.12 | 26.72% |
| Missing returns | 652.39 | 237.28 | 63.63% |
| All missing | 655.52 | 119.84 | 81.72% |
| Unstructured returns | 804.53 | 253.10 | 68.54% |
| Tilted room | 637.54 | 207.58 | 67.44% |

These are component timings on synthetic inputs. They establish neither a
complete-controller speedup nor real-time operation. Existing simulation
controllers and queued source closures were not changed.

The next check uses the original tracking simulation's recorded public packets
and visual evidence over frames 0–853, through two fresh registration objects.
Private function bindings change only the floor-index implementation. The
unchanged registration bodies, independent globals, initial state and failure
latch passed two additional tests. Every resulting registration receipt must
match the original recorded evidence, and all retained registration state and
public input bytes must match. Mapping integration and whole-controller replay
are still required before any adoption in a future simulation.
