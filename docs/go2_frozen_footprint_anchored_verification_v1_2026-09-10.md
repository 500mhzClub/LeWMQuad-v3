# Independent frozen-footprint replay verification V1

Verify only the completed original fixed-root replay, using explicit expected
launch and result SHA-256 arguments. A missing final result, retained failure,
wrong launch, incomplete population, changed input or source binding, or
expanded scientific claim must fail. The checker creates one exclusive
verification JSON under docs and never starts or retries the replay.

The checker is derived from the completed surface-sharing recorded-evidence
checker, source SHA-256
`6bbc8709e3cbf29c6af16777e71f82159a638c8e420a5790b591f29b368cdb6d`,
bound in its completed independent verification
`03eb57a49fb357a6939e1c11a0bab30b29bb185a8e28a47ca51ceb2b6e8b9cc5`.
It preserves the recorded population, hash, command endpoint and timing checks.
The explicit intervention declarations now require frozen footprint receipts
alone; earlier recovery sharing, custom copying and batched queries remain
disabled. Both launch identity and final result identity are explicit inputs.

Authenticate the exact completed surface-sharing predecessor and its completed
batching, copying and profile chain. Check the unchanged assigned JEPA model,
all 405 original/candidate saved hashes, 402 forecast-bearing observations,
405 completed original command endpoints, matching public-input hashes across
the completed profile and copy replay, and all four reported retained-state
hashes at observations 3, 12, 395 and 404. Recompute all timing windows from
finite, positive per-observation times and the original alternating order.
Verify the original decision stream and command tape before and after reading.

The checker authenticates the complete report and saved evidence. It does not
rerun neural inference, independently reconstruct hidden controller state or
reload raw sensor packets. Those computations belong to the original paired
replay. No state types or values are normalized. Verify the checker's own
recursive source closure before and after result checking and record it in the
exclusive output `go2_frozen_footprint_anchored_prefix_verification_2026-09-10.json`.

Synthetic tests cover incomplete or corrupted comparisons, changed public
inputs and commands, unfinished or shifted command intervals, invalid timings,
incorrect execution order, altered states or models, incomplete forecasts,
mixed optimization flags, missing frozen-receipt declarations, and unsupported
navigation, real-time or goal-completion claims. They do not establish the
actual replay result. That evidence can only be checked after the original
replay completes. The native queue and frozen runtime sources remain unchanged.
