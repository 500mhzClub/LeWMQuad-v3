# Local correspondence endpoint depth

The gyro-initial estimator completes 3/4 noisy 601-frame recordings. At the
remaining layout-2 frame-221 failure, the fitted correspondences pass the 3-D
residual bound but too few pass the unchanged image reprojection checks. The
fixed 5x5 local inverse-depth estimator restored clean-depth inlier counts on
that adjacent pair without changing any threshold.

Test `lewm/local_feature_depth_consensus_development.py` in complete sequences.
Descriptor, direct-flow and final chained-flow endpoints lift locally estimated
depths. Feature detection and intermediate chain links keep original depth and
their existing eligibility/photometry checks. All camera packets remain intact.
The floor path still derives its local estimates from original depth, so it is
not filtered twice. Gyro-initial proposals, retained-acquisition equivalence,
floor refinement, final gyro consensus and temporal/bridge gates remain fixed.
Receipts distinguish derived image geometry from raw pixel depth.

A cache lives only for one synchronous observation and retains its source
arrays, avoiding repeated local filtering without stale reuse across frames.
No pose or fitted transform is cached. The actual noisy tracker and registration
passed a 12-frame startup check before this study. The earlier paired probe
constructed features with filtered depth; this sequence variant deliberately
keeps the existing raw-depth detector population and filters the fit endpoints.
It can exclude unsafe endpoint neighborhoods, so sequence outcomes remain open.

Fix all eight assignments: four layouts with 2-mm noise, then four with zero
noise, frames 0–600 and seed 2026091414. Keep source/settings fixed until every
assignment finishes. Use the completed gyro-initial study as the matched control.
Launcher: `scripts/replay_go2_local_feature_depth_noise_development.py`.
Each persistent-routing recording owns an exclusive
`depth_noise_{0|2}mm_local_feature_depth_tracking_601_v1/` directory containing
source hashes, poses, floor availability and all terminal evidence. Load physics
only after estimation to score error. This is fixed-input synthetic-noise replay;
neither hardware calibration nor closed-loop navigation is established.

Before launch, CPU and both GPUs were idle, available RAM 76 GiB, and free
artifact-volume space 21 GiB. Use the successfully measured four-replay scheduling
from the preceding study: disjoint groups 0–3/16–19, 4–7/20–23, 8–11/24–27,
12–15/28–31 with single-thread numerical libraries. Record throughput and memory;
no sensor archive is duplicated.

## Four noisy sequences complete

All four noisy owners exited 0 after accepting every one of 601 frames, with
floor availability throughout. Maximum position errors were
4.494/4.103/4.354/3.978 mm; median errors 1.922/1.980/1.752/2.376 mm.
Elapsed times were 74.14/70.91/75.76/74.17 seconds. Observed RSS was about
1.2 GiB per owner with 73 GiB RAM available. The matched gyro-initial control
completed 3/4 noisy prefixes; the original estimator completed 1/4.

The four zero-noise controls launched next unchanged. Complete journey lengths
were read from retained metadata and agree with saved pose counts:
3,106/3,525/3,105/3,027 frames. After finishing the fixed short controls, the
next test should use those entire recordings with the same frozen estimator
and noise settings. Sixty-second survival is not full-journey robustness.

## All eight short assignments complete

All four zero-noise controls also completed 601 frames, with floor planes
throughout. Maximum position errors were 5.826/3.520/3.541/7.533 mm and elapsed
times 64.8–69.6 seconds. All eight owners exited 0 and all source-hash sets agree.
Combined results are saved in
`go2_local_feature_depth_noise_four_layout_summary_v1_attempt_001/result.json`.

This estimator completes 4/4 noisy and 4/4 clean 60-second prefixes. It improves
on 3/4 noisy prefixes for the matched gyro-initial control and 1/4 for the
original tracker. These revisited development recordings are not independent
new-layout navigation evidence. No native controller has adopted the estimator.
Proceed with the full-recording study defined in
`docs/go2_local_feature_depth_full_journey_2026-09-14.md`.
