# Incremental packed-index paired replay on the completed fused reference

Compare FusedScopedBatchedController with PackedFusedScopedController on the
same fixed 1,428 old observations, using the original paired loop and two
independent assigned models in alternating order. Only eight persistent bounds
indices change to the already tested packed-owned insertion. All 1,425
forecasts, complete normalized decisions and public arrays must remain exact.
Seven retained-state hashes must match the completed fused predecessor. The
only state-type normalization paths are the two existing patch-store tags and
eight explicit bounds-index tags; all numerical state remains compared.

The preceding fused replay has fixed original launch SHA-256
`45bab28d786c146a5d2f66bf7d8d1d429f8a30e7b2007dfde0425be6ce47e6da`
and owner PID 2780519, creation time 1789090638.22. It must end before runtime
admission. The CLI requires the completed result's explicit SHA-256. Admission
checks the original source/output identities, complete rows, timing summary,
model, state checkpoints and preceding combined-reference identities. Missing,
failed, incomplete or altered evidence prevents execution.

Actual original raw worker artifacts and bound model inputs are rehashed
before and after; completed full training provenance is reused without
rerunning its ancestry. Original visibility failure at observation 1173 and
the failed round trip remain. No new model fit, raw scene or physical command
is produced. Performance is an incremental measurement on a shared host;
earlier reductions must not be added or multiplied, and the 100 ms deadline
must still be reported honestly.

Source preflight requires the prepared source closure and measured 64 GiB
available RAM, 41 GiB artifact space and four physical CPUs, with single-thread
numerical settings, fixed hash seed and OpenCL disabled. It does not admit raw
inputs or run a controller. Runtime uses only the available full CPU replay
slot. The existing native scene and queue remain. The new output root is
exclusive; partial output and terminal failure are retained with no automatic
retry, resume or replacement. Entry point:
`scripts/replay_go2_packed_fused_scoped_late_history_v1.py`.
