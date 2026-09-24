# Packed voxel grouping with owned bounds: exact component benchmark

The existing batched insertion candidate had already reproduced311 full
controller decisions in an earlier benchmark, but was not adopted into the
current maze controller. It stores bounds as views into insertion-batch arrays.
The new PackedOwnedMeasuredSampleBoundsIndex preserves the original insertion
and inherited query semantics while grouping biased signed12-bit voxel keys
with one-dimensional np.unique and copying each final2x3 bounds array into
independently owned storage. Original implementations remain unchanged.

Ten focused tests passed1.23s in4591: exact signed/boundary grouping, inverse
indices and counts; accumulated bound bytes, query outputs, witness isolation;
invalid input and capacity rejection; missing-witness partial-failure behavior;
and no surviving reference to old batch-array backing storage.

Component benchmark8906 CLOSED exit0. Root
go2_packed_owned_maze_bounds_benchmark_v1_attempt_001;
resulte3f603f7baed193e23b53116fe14f46a94bc6a98ad4b5eebca0af604dbe9a872;
launchfdbeffa26ed3cbefc3655eb00be372c312f8dea5508c93fba775cb508cb809ff;
1546 sources. It reconstructs exactly128 primary public clouds from the
completed ninth pilot using its saved validated registered visual poses and
unchanged lifting/transform operations. No native pose, model or simulator
input. All source/input/cloud identities pass before/after; every insertion's
key ordering, bound bytes, counts, witnesses, latest frames and sampled
box/sphere queries match the original. Both repeats retain20894 cells.

| Repeat and order | Original insertion total | Previous batched total | Packed-owned total | Original/packed ratio |
| --- | ---: | ---: | ---: | ---: |
| 0: original, batched, packed | 5036.806595ms | 2319.836981ms | 1647.204532ms | 3.057790637 |
| 1: packed, batched, original | 5084.492858ms | 2320.751707ms | 1716.011970ms | 2.962970508 |

Median insertion durations were39.739081/40.425605ms original,
18.097539/17.849543ms previous batched and13.034107/13.481170ms packed-owned.
These are two opposite-order component passes, not a full-controller latency
distribution or isolated-hardware benchmark.

At the end of each pass, distinct bounds-array backing storage totals were
1002912 bytes for original and packed-owned, versus42277536 bytes for the old
batched candidate. This counts array data only and excludes Python dictionaries,
array headers and other process memory. It establishes retention of unused
batch backing storage in this recorded workload, not a universal RSS ratio.

Hardware admission recorded16 physical/32 logical CPUs,3.6% aggregate CPU,
77,133,922,304 available RAM bytes,118,429,171,712 artifact-free bytes and the
independent settling replay as the only substantial competing Python job.
One benchmark process/one numerical thread; no GPU computation or new scene.

## Full-controller equivalence is still running

The separate PackedOwnedLaterFloorController replaces only eight empty
persistent bound indices: primary all/floor/other, auxiliary all/floor/other and
confirmed auxiliary floor/other. It retains original observer, registration,
map/ledger, selector, mission, query methods and exact decision labels.
Two integration tests passed3.80s in44922, including all eight distinct index
types and exact full decisions/bounds through the actual public packet path.

Preflight1055 CLOSED exit0:1550 sources,76,662,919,168 available RAM bytes and
118,392,938,496 artifact-free bytes.8GiB RAM and1GiB output above40GiB reserve
passed. Full replay94114 is LIVE:
go2_packed_owned_maze_controller_replay_v1_attempt_001,
launch90dc683b552a22f90cf2c920e2a1e041d2ae9a6ae1ff0a7c438ea08bbfb97b54.
Last reported100; no mismatch reported. It must reproduce all1881 ninth-run
decisions, including the1870 visual failure and terminal drain, then revalidate
all inputs/sources/model state. Do not restart or edit its frozen sources.

This optimization is not installed in the prepared settled-boundary native
experiment. No complete-controller speedup, real-time operation, improved
navigation or overall-goal completion has yet been established by it.
