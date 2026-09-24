# Tracking cost after the goal on transfer layout 2

The completed fixed comparison's fitted-motion layout 2 failed on return with
a full 32-entry tracking queue. Its full recording is retained at
`go2_stopping_projection_transfer_pose_command_noise_2mm_native_layout02_4800_v1_attempt_001`.
The original failure remains part of the twelve-assignment result.

The unchanged local-reference tracker and robust registration replay accepted
all 1,280 captured frames and exactly matched all 1,246 recorded raw poses.
Maximum registered position error remained 6.994 mm. The replay completed in
154.087 s; it is estimator-only evidence, not a successful alternative navigation
execution. Outputs are in `gyro_coherent_floor_local_view_revisit_replay_v1`.

Per-call wall timers exclude packet loading, decompression and noise recreation.
`tracking_cost_replay_segments_v1.json` gives:

| Frames | Tracking median | Registration median | Revisit attempts / selected |
| --- | ---: | ---: | ---: |
| 0–999 | 54.927 ms | 10.669 ms | 108 / 108 |
| 1000–1090 | 51.999 ms | 9.092 ms | 18 / 4 |
| 1091–1245 | 106.924 ms | 9.056 ms | 155 / 0 |
| 1246–1279 | 117.132 ms | 9.100 ms | 34 / 0 |

All 189 attempts after frame 1090 selected the same candidate reference, frame
1044, but never used it for the accepted pose. The tracker continued to produce
valid poses using another reference. The return-phase median exceeds the 100-ms
camera period even without simulator concurrency or process transport. This
localizes a sustained cost problem, while the earlier successful revisits show
why the reference mechanism should not simply be removed.

A second exact replay profiles only motion calls on frames 1100–1131 with
`cProfile`; it retains the full preceding estimator history. Directory:
`gyro_coherent_floor_local_view_revisit_profile_1100_32_v1`. Source:
`scripts/replay_go2_gyro_coherent_floor_development.py --variant local_view_revisit --profile-from-frame 1100`.
Profiled-call wall times include profiler overhead and must not replace the
first replay's uninstrumented timing estimates. Check all recorded raw-pose
matches before using the profile to choose an optimization. No tracker,
acceptance threshold, reference policy or native controller has changed yet.

## Call profile complete

The profiled replay accepted all 1,280 frames and matched all 1,246 recorded
raw poses again. It profiled exactly frames 1100–1131 and completed in 155.842 s.
The 32 motion calls took 4.918 s including profiler overhead. There were 496
camera/reference candidate attempts and 992 optical-flow calls. Candidate
processing accounted for 4.022 s cumulatively, including 2.111 s in image
consensus registration. Deep copying accounted for 0.740 s cumulatively.
Paired-floor fitting accounted for only 0.231 s and raw-floor retrieval for
0.107 s. These nested cumulative times must not be added as disjoint costs.

This narrows the next optimization to repeated candidate/consensus work. Merely
caching floor clouds would address a small fraction of the profiled cost. The
association with failed local revisits does not by itself isolate their entire
causal cost: the existing primary-first camera selector may test many reference
candidates before using the auxiliary camera. Preserve camera disagreement,
consensus, temporal and pose-acceptance checks when optimizing or testing a
different reference-attempt policy. No optimization has been implemented yet.

## Batched proposal candidate

`lewm/batched_gyro_consensus_development.py` batches the scatter SVDs for the
same 128 ordered gyro-conditioned three-point proposals. It retains the initial
all-point fit and full-consensus early exit, exact proposal seeds/order, scalar
translation and inlier scoring, tie-breaking, final fitting, pruning and all
acceptance thresholds. The unchanged gyro rotation is checked at entry rather
than twice for every proposal. Joint-mode proposals retain scalar fitting.
`lewm/batched_consensus_tracking_development.py` binds this fitter into the
existing single- and paired-camera tracker without changing reference policy.
The original tracker and fixed twelve-assignment launcher remain unchanged.

The focused numerical suite passed 25 tests, including exact pose/mask/receipt
agreement with outliers, failed consensus, degenerate geometry and invalid gyro.
The complete retained fitted-layout-2 recording is being replayed under variant
`batched_consensus`; full raw-pose and registered-pose agreement will be checked,
not just the selected summary fields in the replay result.

A prospective follow-up launcher is prepared:
`scripts/run_go2_batched_consensus_followup_development.py --layout-index 2 --condition pose_command`.
It retains the fitted controller, physics/noise seeds, 100-ms camera cadence,
32-entry queue, reference policy, physical limits and 4,800-tick budget. This is
an exposed-layout follow-up and never replaces the original comparison failure.
Run it alone after replay supports the optimization; success alone would not
isolate the optimization from the change from paired to single-owner host load.

## Exact replay result

The batched replay completed with all 1,280 frames accepted, all 1,246 native
recorded raw poses matched, and all 1,280 full raw poses, registered positions,
floor constraints and revisit attempts exactly equal to the unchanged replay.
There were the same 315 revisit attempts and 112 selections. Maximum position
error stayed 6.994 mm. Total replay wall time was 147.076 s versus 154.087 s;
packet I/O is excluded from the per-call times below.

| Frames | Original tracking median | Batched median | Original / batched over 100 ms |
| --- | ---: | ---: | ---: |
| 0–999 | 54.927 ms | 53.490 ms | 10 / 5 |
| 1000–1090 | 51.999 ms | 48.583 ms | 5 / 0 |
| 1091–1245 | 106.924 ms | 84.265 ms | 92 / 1 |
| 1246–1279 | 117.132 ms | 90.130 ms | 32 / 2 |

Return-section p95 fell from 127.604 to 95.111 ms. This is promising replay
speed evidence, not real-time navigation or a repaired native outcome. Report:
`batched_consensus_replay_comparison_v1.json` in the original failure root.
The prepared exposed-layout native follow-up has now launched alone.

## Native follow-up complete: independently verified round trip

`go2_batched_consensus_pose_command_noise_2mm_native_layout02_4800_v1_attempt_001`
exited 0 after 4:04.38 including startup and recording, with no swap. It accepted
all 1,661 camera poses, independently passed goal frame 1148 and home frame 1659,
and had zero disallowed contacts. Maximum physical distance during the one-second
quiet arrival dwells was 14.706 mm at the goal and 20.002 mm at home; maximum
registered pose error was 5.054 mm. Simulated duration was 166.44 s. All 406
selected plans verify the intended fitted XY/command-yaw treatment, with 401
on time. The physical arrival, navigation summary, saved XY/yaw forecasts and
actual-treatment evaluations are complete. No tracking queue overflow occurred.

Recorded tracking service median/p95 was 59.947/71.424 ms over all 1,661 frames.
The exact-replay comparison isolates unchanged pose outputs on the old recording;
this new closed-loop trajectory differs and ran alone, while the original ran
with another native owner. Do not attribute its complete outcome solely to the
batch optimization or replace the original failed comparison assignment.

The comparison artifact is
`go2_batched_consensus_tracking_followup_comparison_v1_attempt_001/result.json`.
Shared controller/sensor settings and all 152 common source hashes match;
three new source paths select the batched fitter. PNG/SVG trajectory comparisons
are saved beside it. Keep the original failure and this first successful
batched-tracker recording in full.

Camera acquisition remains about 42 ms median and host/simulation lag still
exceeds the wall-mode deadline on many intervals. This is a measured-simulation
success, not host-real-time or hardware qualification. The camera/command
scheduling issue is separate and remains open. Future runs may use this batched
tracker with explicit treatment annotations; the original twelve-run controller
and comparison stay frozen.

The saved trajectory PNG was visually inspected: the original stops at the goal,
while the follow-up completes its measured return to home. No native or replay
owner remains live from this follow-up.
