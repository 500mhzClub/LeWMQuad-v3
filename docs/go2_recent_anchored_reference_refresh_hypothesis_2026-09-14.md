# Recent anchored reference refresh: inactive tracking hypothesis

The stronger reactive layout-3 failure reproduced exactly from public sensors:
996 accepted raw poses matched, then frame 996 exhausted the ten-frame bridge
because all eight retained references lacked sufficient rigid-pose matches.
Incremental image/gyro tracking remained available. The last retained reference
was frame 979; accepted frames 980–985 did not promote new references, followed
by ten bridge frames. At failure, the most recent primary/auxiliary feature
witnesses still contained 36/69 selected features. This is a reference-association
failure, not simply absence of all current image features.

`lewm/recent_anchored_reference_refresh_development.py` prepares a separate
variant. After an already accepted, non-bridge anchored pose, retain its current
paired image/depth features if no reference has been added for 400 ms, the
existing planning period. Reuse the accepted pose and measured floor evidence;
keep the preferred stable anchor and existing eight-reference bound. Preserve
all image/geometry/continuity acceptance rules and the ten-frame bridge limit.
Never promote a bridge pose or reset a failed tracker. This can alter later
associations and accumulated error; additional successful tracking alone will
not establish accurate localization or navigation.

Five focused tests passed in 2.29 s: accepted-pose refresh without altering its
pose or preferred anchor; no promotion from a bridge, duplicate promotion or
young reference; and preservation of the eight-reference limit. These test
retention mechanics, not actual visual matching. A separate saved-sensor replay
must measure tracking survival and pose error before any native experiment.
No replay or native run has used this variant yet. The fixed routing-memory
study retains its unchanged tracker throughout all eight assignments.

## Saved-sensor replay result

The separate replay accepted and registered all 1,018 recorded frames, versus
the original tracker stopping after 996. It took 119.16 s. New reference
refreshes occurred at frames 4, 8, 12, 640, 652, 798, 839, 983 and 1010; the
first raw position difference was frame 985. Native physics was loaded only
after estimation: median/maximum/final position errors were 3.042/5.891/4.115 mm.
No bridge allowance or matching threshold changed. The result is saved under
the failed stronger-reactive layout-3 root in
`recent_anchored_reference_refresh_replay_v1/result.json`.

This clears the particular recorded tracking failure and preserves accuracy
over this short recording. Only 22 recorded frames extend past the original
accepted prefix; there is no counterfactual goal arrival, new native trial,
long-duration reliability or hardware result. Keep the entire fixed memory
experiment on its original tracker. Assess this variant prospectively afterward.

## Native comparison prepared after replay

Prepared `scripts/run_go2_recent_reference_refresh_reactive_development.py`.
Its sole behavioral change from the completed stronger-reactive controller is
the tracking initializer selecting the accepted-anchor reference-refresh model
from frame zero. Both camera geometry and depth validity, pose acceptance,
registration, routing, heading-first terminal control, dispatch, mission budget,
arrival tests, physics and capture remain unchanged. Launch annotation checks
the original stronger-reactive record before adding the treatment and two new
source identities. Four annotation configurations match the existing controls;
the initializer smoke check selects the intended model at frame zero.

Fix four runs on the same development layouts, 0/1 then 2/3, after all eight
memory assignments complete and archive. Keep the current CPU groups and
4,800-tick budget, with no replacement of failures or interim tuning. These
are development revisits; the four completed stronger-reactive runs provide
the original-tracker control, including its layout-3 tracking failure. Record
accepted-pose count, tracking failures, reference refresh events, independent
pose error, verified goals/round trips, contacts and timing across all four.
The first three layouts test whether earlier successful goal arrivals regress;
layout 3 tests whether the replay improvement survives live feedback. No
native refresh trial has launched, and the memory study retains its original
tracker. Any learned-controller adoption remains a separate prospective step.

## First native pair launched

The fixed memory study completed all eight assignments: persistent routing
achieved 4/4 verified round trips and reduced routing 0/4, with all comparisons
saved. Its final owners exited before this study began. Prelaunch CPU was
0.3% busy, about 76 GiB RAM and 30 GiB artifact-volume space were available,
both GPUs idle. Retain the tested two-owner CPU profile and compact capture.

Refresh layouts 0/1 launched around 22:20 local log time in sessions
47833/11857, PIDs 3491474/3491493. Both live launch records identify the
reactive assignment and the 400-ms accepted-anchor reference refresh. The
original tracker controls and every completed memory run remain unchanged.
Finish and archive both owners before dispatching refresh layouts 2/3;
no interim tuning or failure replacement is planned.

## Layout 0 completed with a verified round trip

Owner 47833 exited 0 and archived all 2,718 camera pairs and registered poses.
Independent evaluation verified goal/home arrivals at frames 1864/2716. The
one-second physical dwell distances were 6.998–10.710 mm and 10.178–17.384 mm,
with maximum 100-ms speeds 0.007125/0.031630 m/s and all requested commands
zero. No disallowed contacts occurred. Median/maximum pose errors were
3.206/7.795 mm, path 24.058 m, and 661/667 selected plans were on time.
Owner elapsed time was 390.14 s, maximum RSS 15,322,356 KiB, zero swaps.

The tracker made 27 accepted-anchor age refreshes, with no global history
reset or pipeline fault. The original stronger-reactive control reached the
goal but failed to return. Both had valid tracking throughout their respective
recordings, so this is not simply recovery from a tracking stop. Their routes
diverge, and a single pair does not separate robustness from closed-loop timing
and trajectory variation. Complete all four fixed assignments before drawing
a study-level conclusion or adopting the tracker in the learned controller.

Evaluation, summary and resource records are saved in the layout-0 run root.
The comparison and visually inspected trajectory figures are in
`go2_recent_reference_refresh_comparison_layout00_v1_attempt_001`.
Layout 1 remains active; layouts 2/3 have not launched.

A separate short saved-frame depth-noise sensitivity probe completed while
this pair ran; it changes no trial source. Its scope and results are recorded
in `docs/go2_initial_floor_depth_noise_sensitivity_2026-09-14.md`.

## Layout 1 completed; second pair launched

Layout 1 exited 0 after its full budget, with 4,805 camera pairs and registered
poses, no arrival and zero disallowed contacts. Median/maximum pose errors were
10.559/17.530 mm; path 22.071 m; 1,168/1,188 selected plans were on time.
There were 19 accepted-anchor refreshes. Owner elapsed time was 704.42 s,
maximum RSS 25,172,112 KiB, zero swaps. The original tracker control reached
the goal, so this new run is a goal-completion regression despite uninterrupted
tracking. The first pair totals one goal and one round trip; the four-run
study remains incomplete and the tracker is not adopted in the learned arm.

The terminal failure is physically observable. During the final 180 seconds,
437/444 selected plans requested right turns, seven held, and 438 were on time.
Actual pure-right-turn requests lasted 170.922 seconds. Evaluator physics
recorded 79.030 rad absolute yaw travel and a roughly 34.908-mm median-radius
orbit; its fitted centre was 17.344 mm from the goal. Physical goal distances
ranged 6.269–61.608 mm without a completed dwell. Public registered poses also
showed the orbit, so tracking drift alone does not explain it. The precise
gait-pivot mechanism and a successful remedy remain unproven. Observation and
physical diagnostics, evaluation, summary/resources and the visually inspected
comparison figure are retained in the corresponding run/comparison roots.

Before the second pair, both preceding owners were absent, CPU was 0.4% busy,
about 76 GiB RAM and 25 GiB artifact-volume space were available, and both
GPUs were idle. Fixed refresh layouts 2/3 launched in sessions 41084/28470
with the same CPU groups and unchanged source, controller and settings.

## Layout 3 completed with a verified round trip

Owner 28470 exited 0 after 313.29 s, maximum RSS 12,292,116 KiB, zero swaps.
All 2,053 pairs have registered poses. Independent evaluation verified goal
and home at frames 1445/2051: physical dwell ranges 12.406–14.643 mm and
9.489–10.686 mm, maximum 100-ms speeds 0.017607/0.013272 m/s, all requests
zero. There were no disallowed contacts. Median/maximum position errors were
2.419/7.043 mm; path 19.617 m; 494/506 selected plans were on time.

The tracker recorded 27 accepted-anchor refreshes and no global reset or
pipeline fault. The original controller stopped with tracking failure after
996 registered poses and no verified arrival. The new native run follows a
different feedback trajectory, so it is not an exact replay of that failure;
the earlier saved-input replay supplies the separate same-input evidence.
Together they support further investigation of the retention change, while
the layout-1 goal regression and unfinished layout 2 prevent a blanket
reliability claim. Evaluations, summaries and resources are saved; the layout-3
comparison figures were visually inspected.

Layout 2 remains active, last observed beyond frame 3200, about 1.1 m from
its outbound goal. No replacement run or tracker/controller tuning is planned.

## Four-run study complete

Layout 2 exited 0 after exhausting its mission budget: 4,805 camera pairs and
registered poses, no arrival, no disallowed contact. Median/maximum position
errors were 3.099/6.870 mm, path 12.973 m, closest goal distance 1.111 m.
There were 20 accepted-anchor refreshes and 1,195/1,200 selected plans were on
time. Owner elapsed time was 668.21 s, maximum RSS 25,082,428 KiB, zero swaps.

During its last 180 seconds all 451 selected plans were on time and requested
right turns, but every actual dispatch was zero: 451 missing-observation
decisions and 8,571 latched vetoes. Primary depth was populated at sampled frame
1500 and empty at 2000, 3000 and 4800, while auxiliary depth remained populated.
These diagnostics identify another close-wall observation failure despite
continuous accurate pose tracking. All full recordings and failures remain.

All four comparisons and visually inspected figures are complete. Combined
results: `go2_recent_reference_refresh_four_layout_summary_v1_attempt_001/result.json`.
The original stronger-reactive controls achieved 3/4 goals and 0/4 round trips;
reference refresh achieved 2/4 goals and 2/4 round trips. All eight recorded
zero disallowed contacts. The change improved return completion and eliminated
the recorded tracking-stop outcome, while goal completion regressed on layouts
1 and 2. These mixed development results do not justify a general reliability
claim or automatic adoption in the learned controller. Neither this comparison
nor the memory study establishes JEPA-specific advantage or hardware readiness.

Next is the fixed saved-sequence depth-noise study described in
`docs/go2_initial_floor_depth_noise_sensitivity_2026-09-14.md`. It uses the
original learned controller's tracker and completed memory-study recordings;
the reference-refresh variant remains separate.
