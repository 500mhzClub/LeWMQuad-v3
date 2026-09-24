# Single-pass bounds queries: exact component evidence, roughly2x faster

Benchmark89167 completed with exit0 in29.499661802s. It reconstructed the same
128 public clouds as the prior insertion benchmark, checked their exact hashes,
and inserted them into two evolving indices in each of two passes. All index
state and every timed query receipt matched the original. Final input/source
checks passed; no controller or simulator was executed.

There were36,864 timed queries per implementation across the two opposite-order
passes. Original/single-pass total query times were2015.179901/1008.838212ms
in the first pass and2032.235606/992.775856ms in the second. Median paired
144-query batch speed ratios were2.003319 and2.049575. Both final indices
retained20894cells; the complete reference-receipt digest was identical in
both passes:b8107c99187f05961d0fbf89c94be305e7d95ad24a179810abc8a3d93398a245.

This measures warm repeated component queries at fixed-rule centers on recorded
clouds. It is not the actual controller query distribution or a whole-controller
speed measurement. Shared-machine timing ran beside the native settling pilot.
The existing11 candidate tests passed0.76s; all nine benchmark query definitions
also passed a small direct original/candidate check before launch.

Output:go2_single_pass_maze_queries_v1_attempt_001.
Result:570a3978c98b26d8c47d25a31f646c9278eae8221a86af2cd8ca6538e8991d79.
Launch:bd063784b959d499e2da79fb402fdea79e316846372657aa03d167ea20e562db.
1550 source bindings. Preflight62196 passed: RAM78,949,433,344 and artifact-free
116,849,684,480 bytes. One CPU process/one numerical thread;4GiB RAM and64MiB
output above40GiB reserve admissions. No native adoption or qualification claim.

The next bounded integration is SinglePassLaterFloorController, replacing only
the eight empty persistent bound indices and preserving original output labels.
Two wiring/public-packet tests passed3.83s. A complete1881-decision ninth-maze
replay must validate all actual controller decisions, including failure1870
and terminal drain, before claiming trajectory equivalence. The running native
settling scene retains its original query implementation throughout.
