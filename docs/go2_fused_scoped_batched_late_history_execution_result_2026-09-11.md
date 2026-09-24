# Fused receipt comparison launched

The new receipt implementation and paired replay passed component checks
(73 tests, session 55825, 6.55 s) and replay/component checks (51 tests,
session 33349, 6.34 s). These suites overlap and their counts are not additive.
Source preflight passed with 2,164 frozen bindings in session 36763.

Preparation:
`docs/go2_fused_scoped_batched_late_history_preparation_2026-09-11.json`,
SHA-256 `5d77e53861a5590928d4298a3f97b6b0d59e1e39f160dcf6b5c8caa270786ed4`.

The original process is PID 2780519, creation time 1789090638.22, execution
session 36692. It completed input admission and started reconstructing the
old causal history. Launch SHA-256:
`45bab28d786c146a5d2f66bf7d8d1d429f8a30e7b2007dfde0425be6ce47e6da`.
Execution record:
`docs/go2_fused_scoped_batched_late_history_execution_2026-09-11.json`,
SHA-256 `a253a456d0a9ecdf13af3aca6991f297fe1540559ca93035f65c9370c3432b26`.
Launch verification exited 0 in session 16743 and checked all frozen sources,
the preparation identity, exact live owner and launch input/reference bindings.

The comparison must finish all 1,428 observations, 1,425 forecasts and seven
retained-state checks before equivalence or performance improvement is
established. Completion and final input verification are pending. The
prospective protocol is
`docs/go2_fused_scoped_batched_late_history_v1_2026-09-11.md`.
The native comparison remains in its original sixth worker, and its four
follow-up native experiments retain their existing order. No new native
command or independent-layout data was produced by this replay launch.
