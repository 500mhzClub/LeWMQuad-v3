# Frame-cache and record-copy performance work

The floor-input diagnosis completed with101 exact controller decisions and
unchanged model state. Result SHA-256:
`49de94b98890885d0c47bf78313d2383e62c0d3fc2583795bd05cc1a81291c60`;
launch `1f816b62e3001cbbab65a883281b3badd2a702d817d4a482fe30d8a9ff136da0`.
Session82830 closed successfully. At each of20,60,100, nine calls have exactly
two distinct byte-input groups: four primary and five auxiliary. Grouping used
actual bytes/shapes/dtypes, not hash equality. Calls include floor coverage,
sampled floor patches, retained-patch append and auxiliary confirmation.

The separate frame-cache implementation is in
`lewm/frame_floor_index_cache_development.py`,
`lewm/frame_cached_floor_geometry_development.py`, and
`lewm/frame_cached_floor_map_development.py`. It reuses immutable results only
within one explicit map observation and closes the scope even on failure.
13 individual derivatives are checked against SHA-bound original functions with
only declared provider routing changes. Eight distinct cached inputs maximum;
further distinct calls compute normally. Sensor validation, camera receipts,
pose registration, map and historical patch values, contact policy, constraints
and model are retained. No source or function in the native attempt was edited.

24 tests passed5.92s (44905 closed). The first test run had23 passing and one
AST comparison failing solely on multiline docstring indentation introduced by
moving a function into a class; the check now normalizes docstrings only while
requiring every executable AST statement to match. No runtime mismatch was
observed by those tests. Preflight79379 passed1508 sources, RAM76,860,948,480
bytes, artifactfree125,396,819,968 bytes. No new native scene is admitted.

Exact960-observation cache replay88913 launched as
`go2_frame_cached_floor_prefix_v1_attempt_001`, launch SHA-256
`6e16b5963432fc4f0ed4ab131c1e8d21d0e5ae8e93f3557a9cb66b44fe61db6a`.
Source bindings are frozen. It compares every complete decision against the
completed common-floor candidate prefix and stops before the changed command's
unexecuted outcome. Last confirmed progress at this note's creation: frame900.
At410 cache counts were7hits/2misses/0uncached. This is pending full equivalence,
not adoption or a controlled end-to-end speed comparison.

The separate `lewm/receipt_copy_development.py` helper optimizes exact builtin
container/atomic copying and uses standard deepcopy fallback for other types.
It preserves memo sharing, cycles, custom copying order and NumPy independence.
Nine tests passed0.14s. It is not installed in any controller or running replay.

Paired component benchmark18249 completed, root
`go2_receipt_copy_benchmark_v1_attempt_001`,1505 sources. Result SHA-256
`fc17e49bb2ded6236ff8adb95051284f5d99d849a68793fe5f678237703001c5`;
launch `9e49a80b95bce98709281e4a6beff29386fff2373d0784848fce910742f7bebc`.
Eight interleaved repetitions per function per saved selection, alternating
execution order, all canonical output bytes exact:

| Frame | Serialized bytes | deepcopy median ms | copy_receipt median ms | Component ratio |
| --- | --- | --- | --- | --- |
|20|1410238|11.410479|5.862880|1.9462|
|60|1410168|12.058539|6.477970|1.8615|
|100|1447544|11.848091|6.210778|1.9077|
|959|1499198|12.931234|7.473751|1.7302|

Saved JSON records do not preserve live object alias distributions; the
separate semantic tests address that limitation only for their cases. Native
audit and cache replay were competing during this small benchmark. It shows
a component improvement on these records, not a full-controller speedup or
100ms execution. Any integration needs separately frozen source and full raw
decision equality. The full navigation/comparison goal remains unachieved.
