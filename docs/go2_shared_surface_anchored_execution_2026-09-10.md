# Surface receipt sharing replay in initial input verification

Completed update, 2026-09-10 15:02 UTC: the original replay and its queued
independent checker have both exited zero. All original final input verification
passed. Result `97624a042d388cf1ce988c3007e1886ed25db001e834de8a7a80359f4fe1a385`;
verification `03eb57a49fb357a6939e1c11a0bab30b29bb185a8e28a47ca51ceb2b6e8b9cc5`.
See [the completed result](go2_shared_surface_anchored_prefix_result_2026-09-10.md)
for final scope and timings. The observations below are retained history and
are no longer the current process state.

Latest observation, 2026-09-10T14:46:47.004020+00:00: all 405 paired comparisons
are recorded, with no failure or final result. The exact original replay owner
remains live after reaching the end of its comparison stream. Final original
input verification is still required; the independently prepared checker will
run only after that owner exits successfully.

Provisional fixed-window totals recomputed from the 405 saved timing rows:
402 post-warmup calls total 339.537167456 s original versus
329.617644308 s candidate
(2.92% lower).
Their medians are 0.800636029 s and
0.800612296 s. Repeated-hold medians are
1.622400191 s and 1.316040264 s
(18.88% lower).
Early-window totals are 7.006320178 s and
7.355871943 s. All 402 candidate
post-warmup calls exceed 100 ms, as do all 402 original
calls. These are controller-only, shared-workstation timings and remain
provisional until original completion and independent recorded-evidence checks.
No navigation or real-time qualification follows from these measurements.

The independent verification waiter is PID 2707248, creation epoch
1789051304.83, session 8557. It pins the original replay owner and launch and
checks sources before invoking the separately tested verifier once. See
`go2_shared_surface_anchored_verification_preparation_2026-09-10.md`.

The following entry is the retained initial execution observation.

At 2026-09-10T14:17:32.157470+00:00, the original new replay process was live:
PID 2705097, creation epoch 1789049815.58, tool session 23493.
It emitted `SHARED_SURFACE_ANCHORED_INPUT_ADMISSION_STARTED`. Its reserved
output root, `go2_shared_surface_anchored_prefix_v1_attempt_001`, had not yet been created.
This is initial input verification, not a failed or completed replay; do not
restart because its output is absent or progress is quiet.

The candidate keeps the exact recovery function code and all original decisions
and gates. It borrows only disjoint ordinary surface receipts while those
functions read them, and detaches borrowed receipts before returning a changed
public result. Cross-field aliases, custom values and cycles use the original
copy behavior. There is no persistent workspace and no change to active native
sources. The complete design and fixed prospective replay are in
`go2_shared_surface_anchored_v1_2026-09-10.md` and
`go2_shared_surface_anchored_prefix_v1_2026-09-10.md`.

Preparation passed 30 tests in 2.57 seconds (session 56512), including full
synthetic recovery decisions, vetoes/exceptions, source code identity, private
bindings, alias ownership, mutation isolation, unchanged complete state
serialization and source-only preflight exclusion of runtime work. Source-only
preflight session 96228 passed all 1,991 bindings. The prelaunch machine
assessment found 16 physical CPUs, 73,397,911,552 available RAM bytes and
644,731,559,936 free artifact bytes. The replay uses one numerical thread and
creates no native scene; the original supervised audit remains live.

The new replay must complete original transitive input admission, both fresh
controllers on all 405 fixed public observations, 402 forecasts, full decision
comparison and four retained-state checks, final full input admission and
source/output verification before any completed equivalence or timing claim.
The fixed 402-call total, early and repeated-hold windows must be retained,
including regressions. Neither test success nor scheduling establishes a
speedup, real-time operation, a new navigation outcome or hardware readiness.

The JSON execution record binds the new source bytes and owner identity:
`go2_shared_surface_anchored_execution_2026-09-10.json`, SHA-256
`16f276b808d27e55954c7a85568fa74b78f21fc281eb6b35bf6ca17fc10eec97`.
