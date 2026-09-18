# Reusing measured image-link associations

The batched tracker still accumulated 7.785 s of optimistic FIFO observation
age on the full recorded journey. The frame-990 profile showed repeated
tracking through overlapping retained-anchor image chains. This candidate
caches only exact image-link associations for the same owned read-only gray
and depth arrays and the same ordered source keypoints. Writable feature
frames bypass the cache. Entries hold strong input references, are capped at
256, and expire outside the existing 3.2 s image history when chains run.
Returned arrays and receipts are independent copies.

The original chain algorithm still checks every timestamp, retains original
pixel identities, excludes lost tracks, and directly lifts both endpoints.
No pose, rigid fit, or controller decision is cached. Four focused association
tests pass (1.92 s), including progressive reuse, mutation bypass, intermediate
depth loss, invalid clocks, independent outputs, expiry and capacity.

The first recorded replay, `go2_cached_chain_recorded_tracker_v1_attempt_001`,
failed output equality at frame 1. Its top-level candidate override bypassed
the measured-plane refinement wrapper. The failure and difference artifacts
remain unchanged. The corrected class composition inserts caching beneath
that wrapper. A three-frame moving-image comparison against the batched
tracker passes (2.37 s).

The corrected full 4,740-observation replay completed in
`go2_cached_chain_recorded_tracker_v1_attempt_002`, owner PID 3204845,
creation time 1789290078.84 (session 25494, exit zero). All 4,740 recorded
outputs match except the existing proposal-work counter. The cache records
91 hits and 235 misses. Median tracking time is 81.44 ms, p95 141.47 ms,
maximum 721.15 ms, and total 421.62 s. There are 1,010 calls over 100 ms.
The earlier batched tracker total was 426.62 s and maximum 711.63 ms; these
noncontemporaneous shared-host traces do not establish an isolated cache
speedup. The active native comparison continues with its original controller.

A one-worker FIFO calculation at 10 Hz reaches 6.964 s maximum observation
age at frame 2397, with 2,027 outputs older than 800 ms. Median age is 240 ms;
the queue clears by the end. Acquisition, mapping and planning are assigned
zero cost in this optimistic calculation. The result is recorded in
`docs/go2_cached_chain_tracker_fifo_timing_2026-09-13.json`. Source result hash:
`1fb20b3c5d02a562574274ae9c31f66b170fc05123f3aee8967ee9d17bd9140f`.

This closes the cache experiment without adoption. The limited reuse and
unchanged worst-call scale do not warrant a new native run on their own.
Tracking optimization alone cannot establish whole-loop timing: mapping and
candidate-footprint work remain substantial. Continuous physics and
delayed-observation control remain outstanding.

## Complete-controller paired prefix

The combined sampled-plane, full-consensus, batched-patch and cached-chain
candidate also completed a 13-observation paired replay against the original
stop-conditioned controller. Every complete decision matches after removing
only the sampled-plane marker and proposal-work counts. Execution order
alternates; the first three observations are warmup. The ten timed calls total
5.247 s for the original and 4.234 s for the candidate (19.30% reduction).
Median complete-decision time is 503.77 versus 421.88 ms. This small shared-host
prefix is not a whole-journey speed claim or an estimate of the cache's isolated
benefit. The full tracker replay was running concurrently.

Mean exclusive candidate phase times are 84.30 ms tracking, 33.33 ms floor
registration, 203.07 ms mapping, 85.34 ms action selection, and 7.84 ms neural
inference. Nested inference is excluded from the selection number. Acquisition
is excluded. Mapping is the largest measured component; even eliminating
tracking would leave this decision loop far over its 100 ms period.

The result is `docs/go2_cached_chain_early_controller_2026-09-13.json`, produced
by `scripts/compare_cached_chain_early_controller_development.py` (session
54924, exit zero). Model parameters remained unchanged. The next whole-loop
timing work should target mapping or a design that processes perception and
planning at different rates, retaining actual acquisition and command times.
