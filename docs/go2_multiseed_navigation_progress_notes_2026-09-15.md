# Three training methods across three seeds on two new development mazes

Before any native outcome, fix 22 assignments: full-input JEPA, direct and
supervised-rollout models at seeds 2026091001, 2026091401 and 2026091402, plus
fitted-motion and instantaneous reactive controls, each on both new layouts.
No seed or layout is selected using these navigation outcomes. Preserve every
failure in the fixed denominator; do not tune or replace assignments mid-study.

The layout inventory is `go2_multiseed_navigation_layout_inventory_2026-09-15.json`,
SHA-256 `bf831e45cee5435c982b34a817e059cc16c351de17a698c432878c42e586d512`.
Its two graphs are distinct from the explicit 76-layout development registry;
they remain the same maze family, not final or sealed evaluation.
The nine-model registry is `go2_multiseed_navigation_models_2026-09-15.json`,
SHA-256 `2ed1cb433886b0f29c1f4add68b69ad3eba2ea0446272bf59b3c95575922ccff`.
Each model must match its state identity and its own frozen motion correction.
All corrections use the same earlier training/validation recordings and fitting
procedure. This compares training plus that fixed correction procedure, not
the isolated effect of uncorrected neural forecasts.

All arms share BatchedConsensusMotion, robust floor registration with missing
observation/reacquisition handling, routing memory, current-plane mapping,
committed camera views, physical guards, 2-mm synthetic depth noise and ideal
gyro. Predictive arms share stopping projection and the same planner. Reactive
uses no learned model or predicted candidate outcomes, and differs in predictive
clearance and recovery. Thus this is not an isolated prediction-ranking ablation.
Fitted motion computes the seed-2026091001 supervised model but scores fitted
pose-command XY and command yaw. Learned arms score their corrected learned XY
and learned yaw. Contact scoring is disabled in predictive arms.

Use measured-simulation scheduling, 480-second navigation budget, 300-ms planning
delay, 100-ms cameras, 20-ms gait/command service and 2-ms physics. This permits
two parallel scientific simulations without claiming actual-host-deadline or
hardware validation. Layout 0 uses CPUs 0-7,16-23; layout 1 uses 8-15,24-31.
Wait for both owners and recording completion before advancing to the next pair.

Fixed pair order, abbreviated seed/method; S denotes supervised rollout:

| Pair | Layout 0 | Layout 1 |
| --- | --- | --- |
| 1 | 1001 JEPA | 1401 S |
| 2 | 1001 direct | 1402 JEPA |
| 3 | 1001 S | 1402 direct |
| 4 | 1401 JEPA | 1402 S |
| 5 | 1401 direct | fitted motion |
| 6 | 1401 S | reactive |
| 7 | 1402 JEPA | 1001 JEPA |
| 8 | 1402 direct | 1001 direct |
| 9 | 1402 S | 1001 S |
| 10 | fitted motion | 1401 JEPA |
| 11 | reactive | 1401 direct |

Launcher: `scripts/run_go2_multiseed_navigation_development.py --layout-index N
--arm seed_SEED_full_METHOD` (or `pose_command`, `reactive`). Evaluate completed
owners with `scripts/evaluate_go2_multiseed_navigation_development.py` using the
same arguments. Check actual saved model/correction and final scored forecast
assignments, independent goal/home arrival, contacts, outage/recovery and mission
duration. Report every assigned result, including pipeline and budget failures.
Retain full failures and current reference successes; retire redundant successful
depth only after comparisons and useful diagnosis, retaining all non-depth data.


The first pair launched with JEPA seed 2026091001 on layout 0 (PID 3679846,
session 14187) and supervised rollout seed 2026091401 on layout 1 (PID 3679845,
session 70225). All nine assigned models and fits matched their registry;
intentional mismatched-seed loads were rejected. Actual launch metadata binds
the correct seeds, model assignments, corrections, tracker and registration.
Both owners entered physical navigation. Final arrivals and forecasts remain
pending. Retirement of two superseded early success depth recordings reclaimed
16.577 GB while they run; current references and all failures remain intact.


## First pair independently verified; second pair running

Both first owners exited 0, with no swaps. JEPA seed 2026091001 on layout 0
passed goal/home at frames 1552/2246: 2,248 poses, maximum physical quiet-dwell
distances 19.013/18.934 mm, maximum pose error 5.904 mm, zero contacts and
224.92 simulated seconds. All 552 selected forecasts match its assigned
model/correction and learned XY/yaw; 537 plans were on time. Owner wall time
including recording was 5:40.40, maximum RSS 13,528,696 KiB.

Supervised rollout seed 2026091401 on layout 1 passed goal/home at frames
854/1356: 1,358 poses, quiet-dwell maxima 18.113/4.286 mm, maximum pose error
6.864 mm, zero contacts and 136.00 simulated seconds. All 332 selected forecasts
match the additional seed's assigned model/correction and learned XY/yaw;
319 plans were on time. Owner wall time including recording was 3:53.85,
maximum RSS 9,262,032 KiB. Physical, actual-treatment, XY and yaw evaluations
are complete for both. Neither had a rejected floor observation or reacquisition
hold, so neither demonstrates live recovery from missing floor observations.

The inspected first-pair PNG/SVG and result are in
`go2_multiseed_navigation_first_pair_outcomes_v1_attempt_001`. The two mazes
have different routes; their 26.5/15.0-m paths and durations do not rank the
methods. The full fixed comparison is 2/22 complete, both successful so far.
Preserve these first current full success recordings.

The second fixed pair launched unchanged: direct seed 2026091001 on layout 0
(PID 3681468, session 99721), JEPA seed 2026091402 on layout 1 (PID 3681478,
session 81502). Both entered physical navigation with the correct recorded
model/fit bindings. Their final results are pending. Wait for both exits,
evaluate both, then dispatch pair 3: supervised seed 2026091001 on layout 0
and direct seed 2026091402 on layout 1. No runtime sources or fitted parameters
changed between the first two pairs.


Complete-cohort aggregation is prepared in
`scripts/compare_go2_multiseed_navigation_development.py`. It requires all 22
saved evaluations, preserves failed assignments, compares shared settings and
runtime sources within each maze, and reports the seed-by-maze outcome table.
Six learned-method executions span two independent mazes, not six independent
mazes. No statistical superiority or isolated prediction-ranking conclusion
follows from the aggregate alone.


## Four assignments verified; third pair running

Both second-pair owners exited 0 with no swaps. Direct seed 2026091001 on
layout 0 passed goal/home at frames 1745/2504: 2,506 poses, quiet-dwell maxima
15.636/13.197 mm, maximum pose error 9.527 mm, zero contacts and 250.78
simulated seconds. All 611 selected forecasts matched the assigned direct
model/correction and learned XY/yaw; 596 plans were on time. Owner wall time
was 5:58.97 including recording, maximum RSS 14,622,628 KiB.

JEPA seed 2026091402 on layout 1 passed goal/home at frames 915/1479:
1,481 poses, quiet-dwell maxima 15.164/20.883 mm, maximum pose error 5.731 mm,
zero contacts and 148.42 simulated seconds. All 355 selected forecasts matched
the correct additional-seed model/correction; 340 plans were on time. Owner
wall time was 3:54.88 including recording, maximum RSS 9,678,792 KiB.
Physical, treatment, XY and yaw evaluations are complete for both. Neither
had floor rejection or reacquisition holds. The fixed cohort is 4/22 complete,
with four verified round trips and zero contacts; this remains a partial result.

`go2_multiseed_navigation_first_four_outcomes_v1_attempt_001/result.json`
records behavior and within-layout comparisons. All 159 common runtime source
identities and shared settings match in both pairs. On layout 0, the same-seed
JEPA/direct runs travelled 26.453/27.123 m in 224.92/250.78 simulated seconds.
Direct used 87.50 seconds of turn-only commands versus JEPA's 63.92 seconds,
while translation times were 144.72/143.74 seconds. This explains most of the
observed duration difference in this one paired execution; it is not evidence
of a general JEPA advantage. Layout-1 results currently use different seeds.
Keep all four full recordings through the fixed comparison and useful diagnosis.

The third pair launched with unchanged runtime sources and fits: supervised
rollout seed 2026091001 on layout 0 (PID 3682937, session 61869), and direct
seed 2026091402 on layout 1 (PID 3682942, session 21357). Launch metadata binds
the correct models/corrections. Wait for both owners to exit and evaluate both;
then run pair 4: JEPA seed 2026091401 on layout 0 and supervised rollout seed
2026091402 on layout 1. Final-cohort aggregation remains pending all 22 outcomes.


## Six evaluated: five round trips and one tracking-queue failure

Supervised rollout seed 2026091001 on layout 0 exited 0 in 4:58.58, no swaps,
maximum RSS 12,448,424 KiB. Independent goal/home checks passed at frames
1379/2059, quiet-dwell maxima 14.509/12.310 mm: 2,061 poses, maximum pose
error 7.108 mm, zero contacts, 206.46 simulated seconds. All 508 selected
forecasts matched its model/correction and learned XY/yaw; 502 were on time.

Direct seed 2026091402 on layout 1 exited 1 in 1:27.53, no swaps, maximum RSS
4,927,744 KiB. The tracking queue filled before any arrival: 460 captured
frames, 426 registered poses, maximum pose error 2.869 mm, zero contacts.
All 106 selected forecasts matched the assigned additional-seed model and
correction. Physical, actual-treatment, XY and yaw evaluations are complete.
Its original `Full()` failure remains in the denominator and its full recording
is retained. Secondary clock-closed exceptions occurred during teardown.

`terminal_tracking_queue_diagnostic_v1.json` records tracking service medians
58 ms on frames 0-299, then 113/117/116 ms on frames 300-349/350-399/400-425,
against 100-ms camera cadence. Completion age reached 3.318 seconds. These
are measured-simulation service receipts, not isolated CPU measurements.

An unchanged BatchedConsensusMotion sensor replay accepted all 460 captured
frames in 59.61 seconds. All 426 recorded complete raw-pose fields and registered
positions matched exactly. Saved result and timings are under the failure's
`gyro_coherent_floor_batched_consensus_replay_v1/`, with full equality and segment
costs in `full_pose_match_and_cost_segments_v1.json`. Packet I/O is excluded
from call timings. Tracking medians were 46.38 ms on frames 0-299 and about
101-102 ms thereafter. Every frame 350-459 attempted a local revisit without
selecting it. Thus sustained estimator work already reaches/exceeds the camera
period, before process transport and simulator scheduling. This establishes
backlog on valid poses, not a terminal geometric rejection or a repaired native
outcome. A focused 32-call profile at frame 350 is a useful next diagnosis when
an assigned CPU group is idle; do not change the frozen cohort to address it.

The seed-2026091001 three-method comparison on layout 0 is complete in
`go2_multiseed_navigation_seed2026091001_layout00_comparison_v1_attempt_001`.
All 159 common source identities and shared settings match. All three passed:
supervised rollout 206.46 s / 26.170 m, JEPA 224.92 s / 26.453 m, direct
250.78 s / 27.123 m. Supervised/JEPA/direct turn-only durations were
55.30/63.92/87.50 s; translation durations 141.14/143.74/144.72 s. The PNG/SVG
trajectories were inspected. This is one maze and one seed, not a general
method ranking; the fixed 22-assignment comparison remains incomplete.
None of the six outcomes exercised floor-reacquisition holds.

Pair 4 is running unchanged: JEPA seed 2026091401 on layout 0 (PID 3684196,
session 80122), supervised rollout seed 2026091402 on layout 1 (PID 3684197,
session 28054). Both entered physical navigation with the correct model/fit
bindings. Wait for both owner exits and complete their evaluations. Pair 5 is
direct seed 2026091401 on layout 0 and fitted motion on layout 1. Keep the full
new failure and its replay input; all earlier failures remain preserved.


## Eight evaluated: six round trips, two retained failures; pair 5 running

JEPA seed 2026091401 on layout 0 exited 0 in 6:03.65, no swaps, maximum RSS
14,684,148 KiB. Goal/home passed at frames 1816/2521: 2,523 poses, quiet-dwell
maxima 18.949/17.886 mm, maximum pose error 4.831 mm, zero contacts and
252.60 simulated seconds. All 604 selected forecasts matched the assigned
model/correction and learned XY/yaw; 591 were on time. Evaluations are complete.

Supervised rollout seed 2026091402 on layout 1 exhausted the 480-second budget
without any arrival. Its owner exited 0 in 11:43.32, no swaps, maximum RSS
25,349,364 KiB; this is a scientific failure despite clean process completion.
All 4,805 camera frames were saved; 1,534 registered poses had maximum error
5.131 mm, zero contacts. All 274 selected forecasts matched its assigned
model/correction, 263 were on time. Physical, treatment and XY/yaw evaluations
are complete. Retain the full failed recording and all diagnoses.

The floor recovery mechanism was exercised naturally: 3,271 measurements were
rejected, beginning at frame 567. Rejected poses were absent from publication
and reset arrival dwell. There were 3,699 hold frames and 18,538 zero-command
hold intervals. Planning resumed 58 times, each after four accepted poses,
starting at frame 796; 133 selected plans followed that first resumption, but
there were **zero nonzero requested commands after it**. This demonstrates
reacquisition of floor poses and planning, not restored physical navigation.
Receipts: `floor_reacquisition_behavior_diagnostic_v1.json`.

The first resumed window selected on-time left turns at frames 796/800/804,
but dispatch vetoed missing current obstacle observations and latched those
windows. The independent observer requires a nonempty cloud from each camera,
even when a combined floor plane is available. Replaying the original noisy
packets found 173,864 valid primary pixels at frame 560, only 22 at 566 and
zero at 567. Primary depth remained empty in all checked resumed frames
796/798/800/804 and at 4804, while auxiliary clouds still contained about
9,642-10,603 valid stride-4 points. The recorded floor plane was available in
those resumed examples. Thus recovering floor pose alone cannot clear the
independent obstacle gate when the primary camera is blind. Exact sampled
receipts and the responsible source are in
`current_obstacle_unavailability_depth_diagnostic_v1.json`. No missing depth
was filled or treated as free space. The cause of the primary depth loss and
a safe physical recovery strategy remain to diagnose; do not modify this cohort.

The earlier direct-seed-1402 queue failure's 32-call profile (frames 350-381)
finished in an otherwise idle CPU group while the supervised mission ran.
All 460 replay raw poses, registered positions, paired-floor constraints and
revisit receipts match the unprofiled replay exactly. The 32 instrumented
tracking calls totalled 4.185 seconds: 272 candidate attempts accumulated
3.015 seconds, deep copying 0.863 seconds, paired-floor fitting 0.799 seconds,
and 544 optical-flow calls 0.403 seconds. Batched consensus registration itself
accounted for only 0.127 seconds. Cumulative costs overlap and profiler overhead
is included. Profile group was CPUs 0-7,16-23, unlike the uninstrumented
reference's 8-15,24-31; use the uninstrumented replay for service-cost estimates.
The profile is a concrete guide to repeated candidate/floor/copy costs, not a
second consensus optimization or a repaired mission. Output directory:
`gyro_coherent_floor_batched_consensus_profile_350_32_v1/`, with
`profile_diagnostic_v1.json`. No profile or replay remains running.

Pair 5 launched unchanged: direct seed 2026091401 on layout 0 (PID 3686319,
session 46084), fitted motion on layout 1 (PID 3686320, session 71839).
Wait for both owner exits and evaluate both. Pair 6 is supervised rollout seed
2026091401 on layout 0 and reactive on layout 1. Fixed result so far is 6/8
verified round trips, not the final 22-assignment result; all eight were contact-free.


## Ten evaluated: eight round trips; first fitted-motion control passed

Direct seed 2026091401 on layout 0 exited 0 in 5:19.37, no swaps, maximum RSS
13,102,076 KiB. Goal/home passed at frames 1537/2192: 2,195 poses, quiet-dwell
maxima 18.001/7.392 mm, maximum pose error 8.734 mm, zero contacts and
219.92 simulated seconds. All 541 selected forecasts matched the assigned
model/correction; 526 plans were on time. Physical, treatment and XY/yaw
analyses are complete. On this maze direct is faster than JEPA for seed 1401,
whereas the ordering was reversed for seed 1001. These single executions
illustrate why a method conclusion should await the complete seed comparison.

The fitted-motion control on layout 1 exited 0 in 4:38.93, no swaps, maximum RSS
11,115,632 KiB. Goal/home passed at frames 1011/1780: 1,782 poses, quiet-dwell
maxima 14.355/17.839 mm, maximum pose error 5.480 mm, zero contacts and
178.44 simulated seconds. All 429 selected forecasts used fitted pose-command
XY and command yaw, with the correct separately recorded seed-1001 supervised
neural alternative; 409 plans were on time. Physical, actual-treatment and
XY/yaw analyses are complete. Keep this first full current fitted success.
The incomplete fixed total is 8/10 verified round trips and zero contacts.

The supervised-seed-1402 camera-blindness cause is now directly identified in
`primary_depth_raw_range_diagnostic_v1.json`. Primary RGB frames 560 and 796
were inspected and show a close wall. Raw native optical depth at frame 567
is entirely 171-189 mm, at resumed frame 796 entirely 156-187 mm, and at final
frame 4804 entirely 145-184 mm: every one of 307,200 pixels lies below the
fixed 200-mm sensor minimum. At frame 566 only 44 native pixels reach 200 mm,
and the delivered noisy packet retains only 22 valid pixels. The valid range
is unchanged; no out-of-range depth is fed to the controller. The robot can
remain contact-free yet face a wall too close for its forward depth camera.
This is a sensor-range recovery/standoff issue in addition to floor handling,
not unexplained missing data. It remains a retained failed mission, and no
runtime modification or replacement run is introduced into this cohort.

Pair 6 launched unchanged: supervised rollout seed 2026091401 on layout 0
(session 65250) and the model-free reactive control on layout 1 (session 2457).
Wait for both exits, evaluate both, then run pair 7: JEPA seed 2026091402 on
layout 0 and JEPA seed 2026091001 on layout 1. The source and fits remain fixed.


## Pair 6 status: nine verified round trips; reactive reached its budget

Supervised rollout seed 2026091401 on layout 0 exited 0 in 5:11.93,
no swaps, maximum RSS 12,875,892 KiB. Its physical and treatment evaluations
passed: 2,146 poses, goal/home frames 1428/2144, zero contacts, maximum pose
error 5.575 mm, 214.82 simulated seconds. All 529 selected forecasts used
the assigned learned XY/yaw model and correction; 514 were on time.
Quiet-dwell maximum physical distances were 16.645/11.039 mm.
The fully evaluated total is now 9 round trips and 2 failures from 11 runs.

The reactive layout-1 simulation reached its 480.8-second budget with no
observed arrivals and no reported contact: 4,805 camera frames, 419 plans
(415 on time), and 15,734 floor-reacquisition hold command intervals.
Its owner (session 2457) was still saving artifacts at this status check.
Do not treat the terminal log as completed physical/treatment evaluation
or assign the earlier near-range blindness diagnosis to this run without
checking its saved evidence. Wait for owner exit and evaluate this run,
then launch fixed pair 7 (JEPA seed 1402 layout 0 and JEPA seed 1001 layout 1).
Ten of the 22 assigned simulations remain unstarted. Artifact filesystem
space was about 15 GiB; workspace about 2.1 GiB.


## Reactive failure evaluated; pair 7 in progress

Reactive layout 1 exited 0 after 11:38.71, no swaps, maximum RSS 25,269,956
KiB. Evaluation confirms a budget failure with no arrivals, zero contacts,
1,984 registered poses and maximum pose error 6.776 mm. All 419 selected
plans were model-free and used no predicted outcomes; 415 were on time.
After pair 6 the complete evaluated total is 9/12 round trips, three failures.

Its saved evidence confirms the same near-range failure mechanism as the
supervised-seed-1402 layout-1 run. There were 2,821 floor rejections starting
at frame 1082, and 50 planning resumptions starting at frame 1287, each after
four accepted poses. All 15,734 floor-hold requests were zero; 149 plans were
selected after the first resumption but no nonzero command was requested.
At frame 1076 all primary native depth lies at 178-196 mm, below the fixed
200-mm minimum; delivered primary depth is empty while the recorded joint
floor plane remains available. Frames 1081, 1082, 1287, 1292 and 4804 also
have entirely subminimum primary depth. At resumed frame 1287 the auxiliary
camera still has 172,742 valid pixels and the joint floor plane is available,
but the independent obstacle observer has no current cells. Diagnostics:
`floor_reacquisition_behavior_diagnostic_v1.json` and
`primary_depth_raw_range_diagnostic_v1.json`. Retain the full failure.

Pair 7 launched unchanged: JEPA seed 1402 layout 0, PID 3689393/session 13961,
and JEPA seed 1001 layout 1, PID 3689394/session 29945. The latter exited 0
in 5:20.99, no swaps, maximum RSS 12,504,544 KiB. Its physical and treatment
evaluations passed: 2,073 poses, goal/home frames 1459/2071, zero contacts,
maximum pose error 8.348 mm, 207.68 simulated seconds; all 501 selected plans
used the assigned learned XY/yaw model and correction, with 482 on time.
Maximum physical dwell distances were 17.034/17.567 mm. The evaluated total
is now 10 round trips and 3 failures from 13 runs. Layout 0 reported an
observed round trip and was still archiving; evaluate after owner exit.
Next fixed pair is direct seed 1402 layout 0 and direct seed 1001 layout 1.

The isolated `cached_pair_floor` replay variant is not used by this cohort.
Its focused tests passed (12 including the original paired-floor tests).
Its 460-frame failed-tracker replay is running in the now-idle layout-1 CPU
group, session 45180; compare every raw/registered pose, paired constraint
and revisit receipt against the existing uncached replay before interpreting
any timing change. Do not change the frozen native sources.


## Fourteen evaluated: eleven round trips; pair 8 launched

JEPA seed 1402 layout 0 exited 0 in 6:43.81, no swaps, maximum RSS
16,062,344 KiB. All physical and treatment evaluations passed: 2,818 poses,
goal/home frames 1867/2816, zero contacts, maximum pose error 6.782 mm,
281.92 simulated seconds. All 685 selected forecasts used the assigned
learned XY/yaw model and correction; 663 were on time. Quiet-dwell maxima
were 17.413/12.896 mm. All 14 evaluated outcomes remain contact-free.
The within-layout shared settings and all common source hashes match across
seven evaluated arms per maze: 159 common sources on layout 0 and 144 on
layout 1, where the model-free reactive arm reduces the intersection. Receipt:
`go2_multiseed_navigation_first_fourteen_source_comparison_v1_attempt_001/result.json`.

Pair 8 is direct seed 1402 layout 0 (session 1475) and direct seed 1001
layout 1 (session 29684), with unchanged sources/models/fits. Wait for both
owner exits and evaluate, then run pair 9: supervised seed 1402 layout 0
and supervised seed 1001 layout 1. Six fixed assignments remain unstarted.

The isolated cache replay completed with all 460 raw poses, registered
positions, paired-floor constraints and revisit receipts exactly equal to
the uncached replay, including all 426 published original raw poses. Only
96 of 1,687 paired-fit requests hit the cache (1,591 misses). Later frames
400-459 had motion-call median 95.569 ms versus 101.686 ms in the reference,
with 5/60 versus 41/60 calls over 100 ms; earlier frames 0-299 were slightly
slower (48.349 versus 46.382 ms). Both used CPUs 8-15,24-31; the other group
was archiving during the cached replay. Total replay time was 60.181 s,
versus 59.608 s previously. This does not demonstrate an overall speedup or
a repaired native queue failure. Keep this variant isolated; most fitted
pairs are distinct, so a cache alone is not a convincing solution. Exact
comparison and timings are in the failed direct-seed-1402 layout-1 root,
`gyro_coherent_floor_cached_pair_floor_replay_v1/exact_match_and_cost_comparison_v1.json`.
Neither replay is now running.


## Eighteen evaluated: fifteen round trips; pair 10 launched

Both pairs 8 and 9 completed with verified round trips, zero contacts,
correct assigned learned XY/yaw forecasts and motion corrections.

| Arm / layout | Poses | Goal / home frames | Simulated seconds | Plans on time / total | Max pose error mm | Owner wall time |
| --- | ---: | --- | ---: | --- | ---: | --- |
| Direct seed 1402 / 0 | 2281 | 1573 / 2279 | 228.42 | 549 / 562 | 6.658 | 5:30.19 |
| Direct seed 1001 / 1 | 1814 | 1252 / 1812 | 181.56 | 386 / 442 | 4.667 | 4:45.90 |
| Supervised seed 1402 / 0 | 2398 | 1688 / 2396 | 239.94 | 567 / 579 | 7.028 | 5:44.67 |
| Supervised seed 1001 / 1 | 1610 | 1120 / 1608 | 161.14 | 377 / 395 | 7.739 | 4:13.50 |

All four owners exited 0, without swaps. Maximum RSS was respectively
13,521,524; 11,277,172; 14,085,696; and 10,297,428 KiB. Physical quiet-dwell
maximum distances (goal/home) were respectively 1.842/12.811, 12.694/16.152,
18.613/7.851 and 19.337/15.214 mm. Native arrival, actual treatment and saved
XY/yaw evaluations are complete for each. Direct seed 1001 layout 1 had
56 late plans and a 4.286-second maximum measured host lag; its round trip
is not a host real-time qualification.

Pair 10 launched unchanged: fitted motion on layout 0 (session 97646), and
JEPA seed 1401 on layout 1 (session 9280). Wait for both owner exits and
evaluate both. Pair 11 is reactive layout 0 and direct seed 1401 layout 1.
Then run `scripts/compare_go2_multiseed_navigation_development.py` for the
complete fixed aggregate and prepare checked trajectory comparisons. All
three failures and all current cohort depth recordings remain retained.

The auxiliary-only turn recovery is a separate prepared follow-up, documented
in `docs/go2_auxiliary_turn_recovery_2026-09-15.md`. Eight focused tests passed.
Six saved-state guard probes allowed five turns and kept one unavailable
because of its rejected floor. A complete 820-frame sequential observer
replay matched every original floor/gyro receipt and all 567 originally
available obstacle outputs; it supplied auxiliary-only evidence on 29 more
frames and left 224 unavailable. Runtime method order and both model/fit
annotations were checked. None of this demonstrates physical recovery yet.
The fixed follow-up order is reactive layout 1, then supervised seed 1402
layout 1, each with the new observer and turn-only dispatch. Both use the
original layout-1 CPU group sequentially and follow the completed 22-run
comparison. No observer replay remains running.


## Twenty evaluated: sixteen round trips; final pair launched

Fitted motion on layout 0 exited 0 in 5:02.91, no swaps, maximum RSS
12,607,232 KiB. Its physical and treatment evaluations passed: 2,093 poses,
goal/home frames 1392/2091, 209.70 simulated seconds, zero contacts and
maximum pose error 8.202 mm. All 515 selected plans used fitted XY and
command yaw with the correctly recorded neural alternative; 503 were on time.
Physical quiet-dwell maxima were 18.725/9.059 mm.

JEPA seed 1401 layout 1 failed with `queue.Full` in the 32-entry tracking
queue. Owner exit 1, 1:32.45, no swaps, maximum RSS 5,093,384 KiB. The full
494-frame recording is retained. Evaluation verifies no arrivals, zero
contacts, 460 registered poses, maximum error 3.386 mm, and all 114 selected
plans using the correct learned XY/yaw model and correction. This is the
fourth failure in the fixed comparison, not a replacement candidate.

Tracking service median rose from 56 ms over frames 0-299 to 117 ms over
350-399 and 400 onward; the camera period is 100 ms. Completion age reached
over 3.3 seconds. The final zero-duration tracking event and registration/
tracking clock-closed exceptions belong to teardown after the queue failure,
not an additional scientific outcome. Receipt:
`terminal_tracking_queue_diagnostic_v1.json` in that failed root. These are
measured-simulation stage receipts, not a CPU profile.

Pair 11, the final fixed pair, launched unchanged: reactive on layout 0
(session 51172), direct seed 1401 on layout 1 (session 36447). Wait for both
owner exits, evaluate both, then run the complete comparison. All 22 fixed
assignments have now been launched; there are no unstarted assignments.
Free artifact space was about 9.3 GiB before this pair.

A second isolated tracking optimization is prepared in
`lewm/deferred_registration_copy_development.py`. It removes only the first
of two consecutive deep copies during gyro consensus refitting; the final
copy still supplies independent output evidence. Eight focused tests passed,
including exact primary/auxiliary/joint pruned fits and mutable-output
isolation. It is not used by the native comparison or camera-recovery
follow-ups. The new replay variant is `deferred_registration_copy`; run the
460-frame direct-seed-1402 failure replay on an idle CPU group, then compare
all raw/registered poses, floor constraints and revisit receipts and assess
its timing before considering any native use. No such replay has yet run.
