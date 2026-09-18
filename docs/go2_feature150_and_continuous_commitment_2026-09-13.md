# 150-feature tracking and longer command windows

The candidate retains at most 150 spatially distributed corners per camera,
with at most 13 per cell. The existing descriptor constructor computes up to
156 descriptors (13 times 12 cells), then the strength-ordered population is
trimmed to 150 and cell counts recomputed. Retained descriptors exactly match
their 300-feature counterparts in the focused test (1.79 s). Geometric and
temporal pose-admission thresholds are unchanged.

The first attempt failed at frame zero because the original constructor caps
per cell, not the total: its 156 possible descriptors exceeded the attempted
150 limit. Preserve `go2_feature150_300_201frames_recorded_v1_attempt_001`.
This was an implementation error, not evidence of insufficient visual support.

The corrected 201-frame paired comparison completed in
`go2_feature150_300_201frames_recorded_v1_attempt_002` (session 41945, exit zero).
Both trackers accepted all frames; execution order alternated. Timing after
the first three frames was 14.135 versus 16.360 seconds, a 13.60% reduction.
Median tracking time was 68.71 versus 81.00 ms. Maximum position error was
5.17 versus 3.44 mm; median was 4.22 versus 2.56 mm. Maximum rotation error
was 0.003128 versus 0.001983 rad. Native poses were loaded only after tracking
stopped. This is a measured speed/accuracy tradeoff, not pose equivalence.

The full 3,440-frame paired recorded journey completed in
`go2_feature150_300_3440frames_recorded_v1_attempt_001`, session 19574, exit zero.
Both trackers accepted all frames. The 150-feature tracker used 247.85 seconds
versus 292.20 seconds for 300 features, a 15.18% reduction. Median tracking time
was 69.52 versus 82.33 ms. Median/maximum position error was 5.02/10.02 mm
versus 5.61/11.90 mm. Native poses were loaded only after estimation finished.
This is a matched recorded-journey comparison on a shared host, not a fresh
navigation trajectory or real-time qualification.

## Fresh 60-second native run

Root: `go2_feature150_60s_native_layout00_v1_attempt_001`, session 46367,
exit zero. Result SHA-256:
`3cb47f8c985c8b4f48853692acc4cddc0031437a939f5f0862326059ffac866b`.

All 601 frames completed without tracking failure or disallowed contact.
105 of 150 plans were on time, versus 16 in the earlier 300-feature run.
There were 520 nonzero 20 ms services, all forward requests, and one initial
stale-observation veto. Native XY displacement was 56.53 cm, versus 12.71 cm
in the preceding 300-feature episode. These are shared-host, prospective
development runs; do not attribute every difference solely to feature count.
The main interval was 60.1 simulated seconds; zero draining brought the total
after settling to 60.64 seconds. Timed host work was 62.98 seconds. No host
real-time, full raw-sensor audit or maze-goal success is claimed.

## Continuous commitment experiment

The next runtime plans every 400 ms and requests one 400 ms command window
after its 300 ms prefix. The first prefix is zero. Later prefixes come from
commands already committed at the observation boundary, allowing successive
windows to join without a mandatory zero gap. The model still predicts eight
100 ms horizons; candidate progress and contact cost are scored through
700 ms, with a zero final interval. No model retraining is performed.

The immutable commitment ledger retains the observation-time prefix. Before
dispatch, all fifteen actual 20 ms requests in that prefix must match it.
A mid-prefix obstacle veto therefore invalidates the next forecast rather
than rewriting its input. Current depth can still veto at every policy step,
and rejected windows remain zero. Ledger and scheduling tests passed (seven
tests, 2.55 s; two earlier ledger tests passed in 2.03 s).

The new 201-frame native experiment completed in
`go2_continuous_commitment_20s_native_layout00_v1_attempt_001`, session 64530,
exit zero. Result SHA-256:
`2a56b35f8c86a9e3d4288e82e323adb59a31fecc354bc35188a46686a961c3b0`.

There were 26 on-time and 24 late plans. Twenty-four windows completed all
400 ms; two had fewer nonzero requests (one veto and the bounded run end).
All 26 nonzero windows began exactly at their scheduled simulation boundary,
and all fifteen actual prefix requests for each match the model's committed
prefix. All main-loop requested and clipped commands match the native trace.
The longest uninterrupted nonzero sequence was 3.8 seconds. Commands included
296 forward, 80 left-arc, 110 left-turn and 20 right-turn policy services.

Native XY displacement was 1.1865 m over the 20.1-second main interval,
with delta XY approximately [0.9946, 0.6469] m. No disallowed contact or
tracking failure was recorded. Post-estimation native-pose comparison over
all 201 observations gave median/maximum position error 1.81/3.12 mm and
rotation error 0.000900/0.002094 rad. This supports the observed trajectory's
tracking quality; it is not a complete raw sensor or visibility audit.

This establishes model-selected forward and turning motion with continuous
command segments under measured-latency simulation. It does not establish
reliable timing, arrival at the public goal, a return mission, independent
layout reliability or host-real-time/hardware qualification. Next work should
extend this controller toward measured goal arrival and return while retaining
the explicit command-prefix checks and investigating remaining missed plans.
