# Paired late-history test of selection-local footprint reuse

Compare `FrozenFootprintAnchoredController` with
`ScopedFootprintAnchoredController` on the same original full-sensor JEPA
maze-2 observations, starting at frame zero and ending at frame 1427. This is
an incremental comparison against the already implemented frozen-receipt
optimization. Each controller uses a fresh independently stored copy of the
original assigned model, with unchanged state and no gradients.

Require the original late-history profiler to finish under its exact process
identity, with successful terminal result and all eight bound output artifacts.
Its launch SHA is
`0a87515a70adc3ee27039d01ec70a79809329cea77afcd2baae49c327a8599b3`.
The replay command requires that completed result's explicit SHA. The original
negative strict visibility result, hard-failure frame 1173 and zero round trip
remain bound. Failure at 1173 is inside the replayed history; this is diagnostic
development data, not a qualified sensing prefix.

Perform complete original worker/input admission before and after replay.
Reconstruct both full decisions at all 1,428 observations and all 1,425 raw
forecasts. Match each original command endpoint, decision, and public-packet
hash, plus the frozen-controller decision hashes from the completed profile.
Verify arrays after each controller call so mutation by the first controller
cannot contaminate the second. Do not consume observation 1428.

Compare retained observed state at frames 3, 12, 395, 404, 1173, 1418 and 1427.
The exact state scope is memory, floor cells, occupied cells, residual state
and model history. No type-path normalization is allowed for those objects.
The selector class identity is outside that state scope. Complete decision
normalization removes only the candidate's declared top-level optimization
flag and replaces its top-level controller identity; nested evidence remains
unchanged. Do not claim equality of all controller internals from this scope.

Alternate baseline-first and candidate-first execution by frame parity. Time
only `controller.observe`, without profiling. Reconstruction, hashing,
serialization and state snapshots remain outside the timed calls. Preserve
all timings, including regressions. Report the complete navigation prefix
and the fixed windows 3–12, 395–404 and 1418–1427. Use medians, totals and
counts above 100 ms; no isolated-host, acquisition, full-loop, real-time or
native navigation claim follows from a faster replay.

Require at least 64 GiB available RAM, 41 GiB artifact headroom and four
physical CPUs. Refresh admission immediately before replay. These are
available-resource checks, not OS limits or reservations. Fix OpenCV, Torch,
BLAS and the Python hash seed as in the original profile. One CPU replay may
coexist with the originally owned single native scene; do not overlap this
replay with the predecessor CPU profile or alter the native queue.

Create the fresh `go2_scoped_footprint_late_history_v1_attempt_001` root only
after successful input admission. Preserve a comparison stream and terminal
failure on runtime mismatch; no retry or resume. Source preflight creates no
artifact root, loads no model, and requires no completed profile. Before
execution, review the completed profile and the candidate's helper costs as
specified in its preparation protocol.
