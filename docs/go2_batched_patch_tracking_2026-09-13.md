# Batched photometric comparison on recorded image chains

The frame-990 profile identified thousands of scalar patch-agreement tests as
the largest part of its retained-image-chain search. The new helper retains
the original `getRectSubPix` uint8 patch samples, then batches float64 means,
standard deviations and correlation calculations. At decision boundaries it
uses the original scalar calculation. Image borders, texture and correlation
thresholds, LK, depth lifting, track identities and rigid admission are unchanged.

Three focused tests pass (0.23 s), covering mixed valid/invalid patches,
subpixel samples, borders, exact decision-boundary fallback and complete
moving-image association equality. A separate tracker integration test passes
(2.04 s), confirming that the pose fallback actually uses the batched helper
on real image pairs when descriptor support is unavailable.

The recorded benchmark reconstructs the six primary-camera chains ending at
frame 990, starting at frames 978, 975, 973, 971, 968 and 966. They contain 109
image intervals in total, matching the profiled calls. Five paired repetitions
alternate execution order. Every endpoint array and association receipt matches
exactly. Mean total time for the six chains falls from 686.32 to 421.30 ms:
38.61% less association time. This is a benchmark of selected problematic
chains, not an overall tracker speedup or a navigation result.

Exact measurements and source hashes are in
`docs/go2_batched_patch_recorded_chains_2026-09-13.json`. The batched helper is
composed with the verified full-consensus tracker, retaining eager auxiliary
descriptors. The full 4,740-observation comparison completed in
`go2_batched_patch_recorded_tracker_v1_attempt_001`. Every recorded tracker
output matches except the proposal-work counter. Total tracking time is
426.62 s, median 82.24 ms, p95 148.47 ms, and maximum 711.63 ms. There are
1,067 calls above the 100 ms camera period. Total tracking time is only 0.7%
below the earlier full-consensus trace; these shared-host full runs were not
contemporaneous. The paired 29-call timed prefix improves by 5.32%.

A one-worker FIFO calculation with every camera observation arriving at 10 Hz
still reaches 7.785 s maximum completion age, with 2,233 observations older
than 800 ms on completion. Median age is 477 ms and final age 1.284 s. This
optimistically assigns zero cost to acquisition, mapping and planning. The
trace improves on the earlier 8.619 s peak backlog, but does not resolve
continuous timing. Measurements and assumptions are in
`docs/go2_batched_patch_tracker_fifo_timing_2026-09-13.json`.

No active native controller adopts this candidate. Whole-controller timing,
continuous physics and deployment qualification remain unproven.
