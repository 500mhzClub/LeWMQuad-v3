# Independent obstacle-floor estimation under noisy depth

The preceding tracker/registration upgrade completed full noisy recorded
journeys, but all four subsequent live noisy runs stopped during their initial
surveys. Tracking remained accurate, with no disallowed contacts. Independent
current obstacle evidence was unavailable on 92–97% of frames because its raw
floor-candidate detector had fewer than 100 usable points. See
`docs/go2_live_local_feature_depth_noise_2026-09-15.md` for all results.

The existing local inverse-depth estimator restored floor support on all eight
selected failure frames across the four noisy recordings, preserving four
startup fits. All twelve original plane receipts reproduced exactly. A
51-frame observer sequence through the first layout-0 failure reproduced the
original observer and verified that the candidate's obstacle cells still come
from original noisy depth points. These are diagnostic results, not navigation.

After all eight assignments of the current clean/noisy comparison finish,
run one prospective noisy trial on each of the same four layouts with
`scripts/run_go2_live_local_floor_obstacle_noise_development.py --layout-index I`.
Fix all four assignments now; run 0/1 then 2/3, waiting for both archives and
owner exits between pairs. Do not tune between outcomes or replace failures.
Output roots are `go2_live_local_floor_obstacle_noise_2mm_native_layoutXX_4800_v1_attempt_001`.

Only the independent obstacle-floor candidate source changes, to the existing
5x5 local inverse-depth estimate. Keep the original noisy obstacle points,
unknown-ray rules, height band, grid/crop, floor fit and acceptance thresholds,
gyro integration and current-observation requirement. Local estimates are
derived geometry, not raw measured pixels. Keep tracking/registration, mapping,
learned model/residual, persistent routing memory, mission, actions, footprint,
arrival conditions and 4,800-tick budget unchanged. Noise remains 2-mm iid
Gaussian depth noise with seed 2026091414 and identical layout/frame/camera
indexing. Preserve every outcome and timing cost; do not bypass missing evidence.

Use the established two native CPU groups (0–7/16–23 and 8–15/24–31),
single-thread numerical libraries, measured-simulation timing and compact
native-pixel-plus-noise-recipe recordings. Warm the floor kernel before timed
capture in the independent obstacle worker, as already done in the tracker
and registration workers. Check actual RAM/CPU/GPU/storage before dispatch;
use the authorized depth-retention policy if needed for completed old probes.

The primary comparison is the four new noisy runs against the four original
noisy failures: the independent floor-estimation treatment and its measured
runtime cost are the intended differences. The four preceding clean runs are
context with the original independent floor detector, not clean tests of the
new detector. Do not describe this follow-up as a new eight-run noise study.
Evaluate reported goal/return arrivals against saved physics, contacts, pose
errors, floor availability, command holds and planning/release timing. These
are development revisits, not new independent layouts or final qualification.
No hardware noise calibration, strict real-time operation, JEPA-specific
advantage or real-robot success follows from this experiment alone.

The launcher is prepared but has not been run. Complete the current comparison
before activating it.

The first two preceding clean controls subsequently ended on the unchanged
0.3-m/s full-3-D speed guard, with one verified outbound goal and no contacts.
This is a separate execution-envelope problem. The proposed four-run
independent-floor study remains fixed; do not combine a speed-limit or action
change with its floor treatment.

Preparation checks passed: the complete writer chain records the fixed
2-mm noise, unchanged tracker/persistent routing, original obstacle points
and the intended independent-floor treatment. The worker initializer warms
the kernel and selects LocalFloorIndependentObstacles. No native run has
used this launcher yet.

## First pair launched after the complete eight-run comparison

The preceding comparison is terminal and independently evaluated: clean
3/4 goals and 2/4 round trips (two speed-stop failures), noisy 0/4 goals and
round trips, zero disallowed contacts across all eight. All common source
hashes and paired non-noise settings matched. Its combined result is
`go2_live_local_feature_depth_noise_eight_run_summary_v1_attempt_001/result.json`.

Immediately before dispatch, there were no other native/replay/probe owners,
76 GiB RAM available, roughly 12 GiB artifact space and idle GPUs. The completed
local terminal-control diagnostic pair was reviewed for depth retirement;
5.657 GiB was reclaimed while preserving all comparison and non-depth records.
Both new native trials launched on the declared disjoint CPU groups: layout 0
session 63129, layout 1 session 1375. Results are pending. Preserve both through
archive and exit, evaluate them, then complete layouts 2/3 unchanged.

Both first-pair owners were confirmed live with the intended launcher
settings: PIDs 3534927 and 3534948. Their launch records select
LocalFloorIndependentObstacles, preserve original obstacle points and record
unchanged independent-floor thresholds. They reached frames 900/800 with
active OUTBOUND missions. No goal, return or full-run floor-availability
outcome is established yet.

## First pair complete; second pair running

Layouts 0/1 exited 0 with complete archives after 672.67/709.39 seconds,
maximum RSS 25,424,788/25,420,600 KiB and zero swaps. Independent physics
evaluation found no goals, returns or disallowed contacts. Both exhausted
the mission budget. Median/maximum position errors were 2.852/5.873 mm and
6.166/7.695 mm. Preserve these navigation failures.

The floor treatment achieved availability on all 4,806/4,805 receipts, and
both initial panoramas completed all nine views. Planning then repeatedly
returned `VIEW_BUDGET_EXHAUSTED` (1,081/1,074 records). Only 119/126 records
selected actions, of which 114/115 were on time. Thus the dispatch reason
`NO_ON_TIME_PLAN` mainly reflects absent plans after view-budget exhaustion,
not measured inference deadline overruns. No pipeline faults were recorded.
Per-root `floor_and_route_failure_diagnostic_v1.json`, navigation summaries
and independent arrival evaluations preserve these distinctions. Map coverage
under noisy depth is the next component hypothesis, not yet a proven cause.

With both owners exited, unchanged source hashes, 76 GiB RAM available,
7.4 GiB artifact space and idle GPUs, layouts 2/3 launched unchanged in
sessions 53326/27742 on the declared CPU groups. This completes the assignment
roster; retain both through archive and exit before drawing the four-run result.

While the second pair ran, fixed-pose map probes on completed layouts 0/1
reproduced all 114/122 saved mapping-count witnesses through first view-budget
exhaustion. Applying the existing local estimator to floor coverage restored
a base-proposer frontier route in both cases with identical obstacle sets;
these component counterfactuals do not establish live navigation. The separate
prospective mapping study is prepared in
`docs/go2_live_local_floor_mapping_noise_2026-09-15.md` and remains inactive
until this four-run population is complete.

## Final four-run result

Layouts 2/3 exited 0 with complete archives after 673.33/709.83 seconds,
maximum RSS 25,400,632/25,436,508 KiB and zero swaps. Physics evaluation
found no arrivals or contacts. Median/maximum pose errors were 1.668/3.938 mm
and 1.748/5.196 mm. Both completed all nine initial-survey views, kept floor
evidence available on all 4,805 frames, then repeatedly returned
`VIEW_BUDGET_EXHAUSTED` (1,101/1,093 planning records). Of 99/107 selected
plans, 99/104 were on time; neither run recorded a pipeline fault.

All four assignments are terminal and evaluated: 0/4 goals and round trips,
zero disallowed contacts, floor availability 19,221/19,221 and initial-survey
completion 4/4. The predecessor noisy arm also achieved no goals or returns;
its floor availability was only 130/358/190/386 of 4,805 frames per layout,
and none completed its initial panorama. The floor treatment resolves this
component failure but does not resolve end-to-end navigation.

All shared baseline/treatment source hashes and all four treatment source
sets match. Non-treatment launch settings match. Combined evidence is
`go2_live_local_floor_obstacle_noise_four_layout_summary_v1_attempt_001/result.json`.
The separately fixed mapping-floor experiment can now start. All current
independent-floor recordings remain fully retained.
