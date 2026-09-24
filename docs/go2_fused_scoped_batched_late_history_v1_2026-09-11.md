# Fused receipt construction: prospective paired development replay

The completed combined-controller profile has result SHA-256
`c1890e862509753457c1df2fca03555064dbfa1e678f93433572b0ba8f08363e`.
Its late ten-observation window spends 2.8591 s cumulatively constructing
frozen receipts out of 16.9683 s total exclusive profiled time. These times
include profiling overhead and do not establish isolated performance.

This experiment compares the completed ScopedBatchedFootprintController with
FusedScopedBatchedController. It changes only receipt construction: validation
and freezing share one traversal; cache hits clone the validated frozen graph
directly. Every separately returned receipt still owns separate containers,
internal aliases remain, and final public detachment is unchanged. Unsupported
or cyclic input graphs retain forwarding behavior. Cache keys, lifetime,
capacity, stale/failure checks and supported memory/geometry types remain.

The exact old paired loop reconstructs all 1,428 observations and 1,425
forecasts with two independent models and alternating execution order. Every
public input and complete normalized decision must match the original and
completed combined replay. Seven retained memory/map/residual/history hashes
must match the predecessor; only the existing two patch-store type paths are
normalized for that identity comparison. No new physical command is executed.

The completed profile and combined replay outputs, frozen sources, actual old
raw worker artifacts and bound model inputs are rehashed before and after.
Completed full training provenance is reused; full training ancestry is not
reexecuted. Original visibility failure at observation 1173 and failed round
trip remain. Neither performance improvement nor navigation success is assumed.

One CPU replay slot is used only after the exact original profile owner has
ended. Requirements are 64 GiB available RAM, 41 GiB artifact space and four
physical CPUs, with single-thread numerical settings, fixed hash seed and
OpenCL disabled. The single existing native scene and its queue are untouched.
Attempt output is exclusive. Failure and partial evidence remain; no automatic
retry, resume or replacement is provided. The entrypoint is
`scripts/replay_go2_fused_scoped_batched_late_history_v1.py`; source preflight
must pass before execution. This is development replay evidence, with no
independent-layout collection, training, hardware or deployment claim.
