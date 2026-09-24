# Original-rate tracking with a full-consensus early exit

After the 5-Hz camera experiment failed during the recorded return transition,
this experiment retains every 100 ms camera observation. It changes only
proposal-search work in rigid registration: once all input correspondences are
inliers, every maximum-count hypothesis has the same all-true mask. The original
final fit and monotonic pruning therefore receive the same mask. Residual
tie-breaking between full-consensus proposals cannot change that final fit.

Random proposals are generated lazily in their original seeded order. If no
full-consensus proposal appears, the original search completes. Geometry,
minimum support, reprojection, gyro-disagreement and displacement checks remain
in force. The diagnostic `valid_proposals` count records the work actually done.
The original implementation used by native runs is unchanged.

Five focused tests pass (1.32 s), covering exact pose, inlier-mask and quality
equality with and without outliers in both fitting modes, and final gyro
disagreement rejection even after full consensus.

The first 32 recorded tracker observations match the original evidence except
for `valid_proposals`. Alternating baseline/candidate order and excluding three
warmup observations gives 2.9952 s baseline versus 2.5865 s candidate total:
13.65% less tracking time. Median call time falls from 102.09 to 88.35 ms.
This is a short, paired comparison on a shared host, not a real-time guarantee.

The full 4,740-observation check completed against the reused-maze recording.
Every tracker output matches after removing only the `valid_proposals` work
count. Median tracker time is 82.93 ms, p95 148.31 ms, maximum 983.21 ms, and
1,101 calls exceed 100 ms. Total tracker time is 429.61 s, compared with 461.08 s
in the earlier sampled-plane run; those full traces were measured at different
times on a shared host. The short paired comparison above is the direct timing
comparison. No model, map, planner or native scene was executed. The artifact root is
`go2_full_consensus_recorded_tracker_v1_attempt_001`.

A separate cProfile sample of ten calls after three warmup observations finds
sampled plane extraction still costs about 32.8 ms per observation, feature
construction 22.2 ms (including 16.8 ms SIFT computation), and rigid registration
17.0 ms. These are profiled inclusive costs, with instrumentation overhead and
some overlapping work; they should not be added as an unprofiled timing claim.
The profile is `docs/go2_full_consensus_tracker_profile_2026-09-13.json`.

Reusing the full measured timing trace in a 10-Hz FIFO calculation gives median
completion age 647.34 ms and a maximum of 8.619 s (frame 4459), versus 13.342 s
maximum for the preceding sampled-plane trace. It still accumulates stale
output despite mean service time being below 100 ms. Acquisition, mapping and
planning are excluded, and no delayed physical behavior is inferred. The exact
results are in `docs/go2_full_consensus_tracker_fifo_timing_2026-09-13.json`.

This candidate is not installed in the active navigation runs. Full-controller
timing and continuous physics remain outstanding evidence.

The separate deferred-descriptor experiment defers auxiliary corner descriptors until a fit
needs that camera, while owning its measured RGB and depth immediately. Two
focused tests pass, including exact fallback equality after a primary-camera
failure and preservation of pixels after the caller mutates its input arrays.
Its full-history check completed after this one in
`go2_lazy_auxiliary_recorded_tracker_v1_attempt_001`. The first 32 frames match;
the 29 timed calls take 5.33% less total time, although the median worsens from
87.41 to 90.06 ms. This mixed short result does not establish a useful overall
speedup. All 4,740 outputs match except work counts. Total tracker time was
436.42 s, median 84.39 ms, p95 142.81 ms, maximum 986.38 ms, and 1,245 calls
exceeded 100 ms. These full traces are not contemporaneous paired measurements.
The fifteen known auxiliary-camera frames took 1,187.15 ms total versus
1,088.97 ms in the preceding full-consensus run; their maximum rose from
132.53 to 143.63 ms. There is no clear overall benefit, so the next candidate
retains eager descriptors. The deferred result is preserved and not adopted.
