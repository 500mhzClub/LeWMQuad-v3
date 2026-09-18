# Smaller visual feature population

The candidate retains at most 25 upright-SIFT corners per spatial cell and
300 per camera, versus the original 50 and 600. Detection, selected-corner
descriptors, RGB/depth sampling, geometric admission, measured-plane refinement,
gyro checks and temporal thresholds are otherwise unchanged. The pose output
is expected to differ. Two focused tests pass (2.36 s): selected descriptors
match their original counterparts, and the moving-image observer retains the
plane-refinement wrapper and bounded feature population.

The first 201 independent-maze observations all produced accepted poses.
Across the 58 paired timed observations, tracking takes 4.963 versus 6.195 s
(19.88% reduction). Candidate median tracking time across 201 frames is
82.68 ms, maximum 216.44 ms. Native poses were opened only after estimation
stopped. Candidate median/maximum position errors are 2.56/3.44 mm; the original
recorded tracker has 0.82/1.82 mm. Maximum rotation errors are 0.001983 versus
0.001049 rad. This is a short-segment speed–accuracy tradeoff, not evidence of
unchanged accuracy or complete-journey reliability.

Prefix root: `go2_feature_budget300_recorded_prefix_v1_attempt_001` (session
6168, exit zero).

## Paced result and dispatch correction

The 61-frame process-separated paced replay completed with 10 on-time and
five late plans, compared with one on-time and 14 late plans for 600 features.
Maximum tracking completion age fell to 202 ms; registered-pose age reached
251 ms. Maximum candidate position error in this shorter prefix was 2.58 mm.
There was one skipped 20 ms request tick. The model remained unchanged.

The original paced result contains 22 nonzero shadow requests across nine
command windows. Subsequent inspection found that they first became nonzero
40–60 ms after their scheduled starts: an initial stale-observation veto was
retried within the same window. These outputs do not establish the intended
100 ms command execution or successful on-time dispatch. The original result
is retained in `go2_feature_budget300_paced_recorded_prefix_v1_attempt_001`
(session 72087, exit zero).

The runtime now latches a veto for the remainder of that command window.
A focused regression test passes (1.99 s), demonstrating that a later fresh
observation cannot restart a rejected window. Subsequent corrected paced
results are recorded below.

The early obstacle check now uses the tracker's current paired-depth measured
plane to classify above-floor returns in current body coordinates, before the
separate floor-registration stage. Missing current plane or paired returns
produce no usable obstacle observation. The 0.45 m nominal disk is unchanged;
this remains a sampled obstacle veto, not proof of whole-body clearance,
ground support or free unobserved space.

Putting this extraction inside the serial tracking process completed all 61
frames but missed all 15 planning deadlines. Tracking mean service was
112.89 ms and maximum completion age 898.90 ms. The result is preserved in
`go2_early_obstacle_paced_recorded_prefix_v1_attempt_001` (session 82729).
An isolated 13-frame profile measured roughly 7.2–8.9 ms per obstacle update;
the paced timing difference cannot all be attributed to this cost because
these were shared-host runs.

Moving extraction to the existing registration worker, ahead of floor
registration, permits overlap with the next tracking update. This completed
all 61 frames with 10 on-time and five late plans. Four windows each emitted
five nonzero shadow requests; first requests started 0.376–0.653 ms after
their scheduled dispatch clocks. Six other windows were initially stale and
remained latched to zero. No request ticks were skipped. Maximum tracking
completion age was 199.14 ms, obstacle publication age 208.57 ms, and planning
completion age 362.66 ms. Obstacle extraction mean service was 8.64 ms.
Result: `go2_overlapped_obstacle_paced_recorded_prefix_v1_attempt_001`, session
41941, result SHA-256
`cf79f850b3ff7c086f7ba538df5af433ec4cf21ab5acc8bcfe3ac3ae251eaa53`.

Both new runs retained the 300-feature prefix accuracy and unchanged model.
The focused obstacle-veto and latch tests passed (three tests, 1.91 s).
The corrected pipeline now demonstrates timely shadow dispatch in some
windows, but only four of fifteen proposed windows were usable. Acquisition
was preloaded, requests did not control the recording, and no prospective
navigation outcome or reliable continuous execution is established.

## Full recorded journey

The complete 3,440-observation independent out-and-back recording finished
with all observations accepted in
`go2_feature_budget300_full_journey_v1_attempt_001` (session 46953).
Native poses were loaded only after estimation stopped. Candidate position
error median/maximum was 5.61/11.90 mm versus original 6.70/11.17 mm;
rotation error median/maximum was 0.001348/0.004867 rad versus original
0.002345/0.009873 rad. Median tracking time was 80.41 ms, maximum 346.58 ms.
The 58 paired timing observations showed a 19.04% reduction (5.285 versus
6.528 s). Total experiment wall time was 431.55 s on the shared host.
The complete recorded journey supports proceeding with the smaller feature
population, but acquisition timing is excluded and spikes remain. There is
no new native navigation result or real-time qualification.
