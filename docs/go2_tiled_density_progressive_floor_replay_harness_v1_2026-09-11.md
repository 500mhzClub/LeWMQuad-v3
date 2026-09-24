# Tiled-density full-history replay harness

`scripts/tiled_density_progressive_floor_replay_development.py` prepares the
paired replay without a launch entry point. It privately binds the unchanged
original 1,428-observation replay body to ProgressiveBatchedFloorController and
TiledDensityProgressiveFloorController. Both retain progressive patch batching;
the candidate changes only the dense branch of floor-index construction in
registration and mapping.

The harness retains original ordered packet and command-tape checks, all
normalized decisions, every predecessor candidate decision hash, alternating
execution order, 1,425 forecast comparisons, independent model tensor storage,
unchanged model state and gradients, and seven original retained-state checks.
Only the existing ten type-normalization paths and the new controller name and
implementation flag are normalized. It consumes no observation 1428 and runs
no physics. The original strict sensing failure remains part of the reference.

Sixteen harness tests passed in 4.96 s (session 79354, exit zero), including the
complete synthetic 1,428-frame loop and corruption of receipts, input arrays,
metadata, terminal status, retained state, gradients, weights, model/storage
sharing, command endpoints and predecessor input/decision hashes. These tests
establish harness behavior, not actual recorded-controller equivalence.

Before invoking this harness, a launcher must:

1. Observe the exact progressive owner ended and preserve any failure.
2. Run its prepared completion checker with the actual completed result hash.
3. Bind that completion witness and reconstruct the entire original progressive
   result and timing rows, including its raw/model admission and negative scope.
4. Check source bindings, original single-thread environment, at least 64 GiB
   available RAM, 41 GiB artifact space and four physical CPUs. Preserve one
   full CPU replay at a time.
5. Create the exclusive artifact root
   `go2_tiled_density_progressive_floor_late_history_v1_attempt_001`, record
   actual launch/owner identities, invoke the harness once, and verify all
   inputs, sources and predecessor evidence again before writing completion.

The launcher has not yet been created because the progressive result and its
completion witness are not available. This preparation launches nothing and
does not substitute tests for a successful actual replay. The next independent
navigation decision still depends on completion and review of the existing
native diagnostic queue.
