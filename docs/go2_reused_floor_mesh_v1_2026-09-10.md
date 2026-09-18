# Reuse an exact measured depth mesh within an observation

The optimized-controller profile has written both fixed windows and all 405
decision comparisons; its final input audit is still pending at preparation.
The early window contains 40 floor-index constructions across ten observations,
with 3.400738634 cumulative profiled seconds in `observed_floor_cell_index`.
Cumulative timings overlap and include profiler overhead. This motivates a
bounded source experiment, not a real-time or completed-profile claim.

Keep the exact existing depth mesh calculation. Separate the points, three
triangle cross products, lengths and fourth-point planarity mask from the
tests involving the supplied up vector. Reuse those private intermediate arrays
only for exactly equal depth dtype, shape and bytes and valid-mask dtype, shape
and bytes during one observation. Keep the original complete-index keys,
including exact up bytes, and the original hit/miss/uncached counters. Both
caches admit at most eight entries and clear on normal exit or failure.

Preserve the original validation, expression order within each numerical
calculation, thresholds, output array dtypes and summed invalid-cell index.
Do not round poses, approximate geometry, reuse across observations, infer
unobserved floor or exempt contacts. Original public index results keep their
immutable byte backing. Private mesh arrays own their data and are not exposed
to consumers or retained after the observation. No new dependency is needed.

The candidate uses the already verified frozen-footprint selector unchanged.
Its map inherits the original measured-floor transport map. Its observation
method changes only the temporary geometry provider; ordered primary/auxiliary
floor witnesses, retained memory, failure latches and cache cleanup remain.
It adds one controller identity and one mesh-reuse flag to the frozen-footprint
metadata. Original models, forecast horizons, scoring and feasibility rules
remain in place. Existing live and completed source files are unchanged.

Initial checks: 26 mesh tests passed in 8.70 seconds; five controller integration
tests passed in 8.30 seconds. Synthetic controller tests cover three warmup
observations, including missing-floor transport, complete decisions and retained
state, input preservation and cleanup. They do not execute model forecasts.
The isolated workload used two synthetic depth images and two distinct exact
up vectors per image. After two warmups, six alternating-order repetitions
gave median 0.22673196252435446 seconds for the original frame cache and
0.16289117594715208 seconds for mesh reuse, with all four complete indices
byte-equal. This is a shared-host helper measurement, not a controller speedup.

Before accepting this implementation, complete the original optimized-profile
audit and authenticate its result. Then run a separately named, source-bound
paired replay on the same 405 original public observations. Require all 402
fresh model forecasts, complete decisions, command endpoints, public arrays,
unchanged weights/gradients and the four existing retained-state checks. The
original retained-state population uses the unchanged memory class, floor and
occupied dictionaries, residual and history; no type/value normalization is
needed there. Synthetic whole-controller comparison additionally permits only
the declared controller and mapper class changes. No simulator use, real-time
qualification, independent-layout success or hardware claim follows from this
source experiment.

The prepared raw runner is `scripts/replay_go2_reused_floor_mesh_prefix_v1.py`,
using the exclusive root `go2_reused_floor_mesh_prefix_v1_attempt_001`.
Source preflight may inspect the frozen optimized-profile launch and completed
predecessor, but must not admit original raw inputs, load models or create the
output root. Actual replay requires `--profile-result-sha256` and rejects an
incomplete or failed optimized profile before raw input admission. Bind all six
profile artifacts and the exact profile launch
`41b564dae8ec07a43c0c9d23068010b8f520692bb49465afbeb7ca216ba0b6d1`.
Require 48 GiB available RAM, 41 GiB artifact space and four physical CPU cores;
run one CPU replay alongside the existing single native worker.

The runner reuses the original paired replay's unchanged code object with only
the candidate constructor, metadata normalization, output root and progress
label substituted in a private globals dictionary. Imported module globals
remain unchanged. Remove only the new mesh flag, the existing frozen-footprint
flag and the top-level controller identity when comparing complete decisions.
Keep every nested evidence field. Preserve all original source/input/artifact
checks before and after execution, fail on any mismatch and retain failures
without automatic retry. Eleven focused runner checks passed in 2.31 seconds.

Timing compares the original controller with the combined frozen-footprint and
mesh-reuse candidate. It does not isolate the incremental mesh contribution
against the already optimized controller. The isolated helper measurement above
and the future paired controller measurement must remain separate.

Before execution the profiler completed as
`c636eb55c13f02624b73680295ab3f70d7faac00680d7e66dd820b870cfb9866`.
Its independent recorded verification
`bc432a67568521ee173057ecd6b0fda714c8281ca88480e203d4df439dafea56`
checks all 405 comparison rows, source/artifact bindings and both raw profiler
summaries. The final runner binds these exact identities, includes that
verification and final profile report in its source closure, and rejects any
other profile result. Source preflight also checks the verification identity.
