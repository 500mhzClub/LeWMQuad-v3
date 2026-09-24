# Prospective short-pulse navigation comparison

Use the three frozen pulse-trained residual models directly in navigation.
The offline study improved direct/supervised pulse predictions but regressed
JEPA pulse prediction and yaw. Preserve every method; do not choose a favorable
training seed or add an externally fitted neural correction.

The fixed plan has fourteen assignments: JEPA, direct, supervised rollout,
pose-command fit, matched-data command-history fit, instantaneous ranking,
and reactive control on each of two new development maze layouts. Maze 0 uses
that order; maze 1 reverses it. Geometry construction rejects matches to the
explicit 82-layout development registry, and is fixed before native outcomes.
These are two new units, not fourteen independent mazes or final evaluation.

All conditions share public RGB-D/gyro perception, mapping, mission arrival
rules and actual dispatch guards. Predictive conditions share routing, recovery,
clearance and stopping checks. The instantaneous condition uses supervised
forecasts in those checks while replacing the main ranking with current
waypoint utilities; it isolates ranking rather than all prediction. The
reactive condition consumes no future forecasts and also loses predictive
guards; it is a broader controller comparison. Existing memory results remain
separate, and this experiment does not establish a new memory contribution.

Neural predictors receive their original four causal RGB/body/control packets
and eight prospective commands. The command-history control has the same
training data, draw schedule and future commands, but no camera/body inputs.
The pose-command control additionally consumes four causal registered poses
and was trained on earlier closed-loop recordings; it is a strong practical
comparator, not an isolated architecture comparison. All predictive arms
compute the alternatives for comparable bookkeeping and recorded diagnostics;
only their assigned forecast reaches the shared scorer and guards. Contact
scores remain disabled, not interpreted as contact-free predictions.

The nominal command-composed neural forecast is already the trained model's
absolute output. No predecessor translation bias or closed-loop residual
correction is applied. Use learned XY/yaw together for neural arms, fitted
pose XY with integrated-command yaw for the pose control, and fitted XY/yaw
for the command-history control. Keep the original terminal coordinate metric
across the comparison; do not simultaneously promote the coordinate follow-up.

Run native owners sequentially using the established CPU groups. Parallel
paused-physics data collection speedups do not apply to paced navigation.
Keep the 4,800-tick budget, 300-ms planning delay, 400-ms planning cadence,
100-ms terminal translation pulses, 20-mm observed arrival radius, independent
40-mm physical arrival check and one-second quiet dwell. Sensing remains
synthetic 2-mm depth noise with ideal gyro; measured simulation is neither
calibrated sensing nor a hardware/hard-real-time claim.

Report every terminal failure, arrival, contact, elapsed simulated/wall time,
late plan and measured forecast error. Evaluate the prior assignment before
launching the next. Keep all results and diagnosed failure records. Routine
retirement of completed, diagnosed depth may support later assignments; active
debugging data and selected sensor-replay references remain retained.

Launcher: `scripts/run_go2_short_pulse_navigation_development.py --assignment N`.
Plan: `docs/go2_short_pulse_navigation_plan_2026-09-16.json`.
Inventory: `docs/go2_short_pulse_navigation_layout_inventory_2026-09-16.json`.

## Recorded progress

Assignment 1 (JEPA, maze 0) is complete and independently evaluated. The owner
exited zero; the mission exhausted its fixed 4,800-tick budget. It achieved a
physically verified outbound goal, but no return arrival: zero disallowed
contacts, 19.75 m travelled, final native home distance 2.840 m. Outbound dwell
at frame 3552 stayed within 18.344 mm of the target with zero requests and
maximum measured 100-ms speed 1.232 mm/s. This is an incomplete round trip,
not a successful navigation assignment or an infrastructure crash.

The robot first observed the goal within 10 cm at frame 2774, then took 77.8 s
to certify arrival. Of 188 selected plans in that interval, 171 were left turns,
eight holds, five left arcs and four forward; 184 were on time. Thus terminal
turning remains material even where most plans met their deadlines. The saved
trajectory shows outbound travel around the long wall and partial physical
backtracking along the same corridor. It does not show a completed return.

Further recorded-selection diagnosis: arrival-entry gating changed 159 of
those 188 near-goal decisions, including 157 reversions from translation to
turn. The preceding position-only preference had found predicted progress,
but the selected translation did not predict reaching the 20-mm circle in one
step, so the gate restored heading guidance. The stopping-projection guard
changed none of these 188 choices. This is a rule interaction, not evidence
that any unexecuted translation would have succeeded. On the same 157
reverted candidates, the recorded pose-command alternative predicted entry
zero times and command history once; simply replacing the neural forecast
does not obviously remove this gate condition on those recorded states.
All recorded neural common-prefix spreads were exactly zero, so candidate
disagreement before dispatch is not the explanation in this run. See
`terminal_selection_diagnosis.json` and
`terminal_alternative_arrival_forecasts.json` in the assignment-1 analysis root.
These findings motivate a later isolated controller experiment; the active
fourteen-assignment comparison remains unchanged.

On 425 matching executed 700-ms windows, position RMSE was 19.908 mm for the
raw nominal-composed JEPA model, 9.137 mm for the recorded pose-command
alternative, 11.995 mm for command history and 21.982 mm for nominal integration.
Yaw RMSE was 2.632 degrees learned, 0.641 command history and 1.457 nominal.
Only seven windows were translation pulses. These overlapping, selected
trajectory windows compare forecasts, not alternative policies or independent
maze trials. The actual treatment check confirms raw neural XY/yaw reached the
controller without an external correction.

Timing remains a limitation: 1,023/1,193 plans were on time; 170 were late.
The simulation loop reported 622.456 wall seconds for 480.82 simulated seconds
and peak simulator lag 139.724 s; recording persistence added further wall time.
Median measured camera service was 82.555 ms and planning 106.290 ms. The earlier
first-seed full-JEPA maze-0 reference had medians 42.243 and 78.029 ms, but both
maze and runtime differ, so this comparison does not isolate a cause. No
real-time claim is supported.

Artifacts: `go2_short_pulse_navigation_jepa_noise_2mm_native_layout00_4800_v1_attempt_001`
contains physical, actual-treatment and same-window forecast evaluations.
`go2_short_pulse_navigation_assignment01_analysis_v1_attempt_001` contains the
inspected trajectory figure, stage timing and terminal-approach counts. The
first evaluator invocation preceded completion of native persistence and stopped
on missing camera metadata; after the same owner's clean exit, evaluation
completed. The native experiment was not restarted.

Assignment 2 (direct, maze 0) failed on a disallowed physical contact after
253.286 simulated seconds following settling. The owner exited one and saved
the complete failure recording. Independent evaluation found no arrivals and
one contact sample. The rear-left calf contacted
`independent_round_trip_wall_0_1_0`, with measured force magnitude 175.787 N,
while a right-turn request was active. All failures remain in the denominator;
no retry or model/controller adjustment replaced this assignment.

The final plan was on time and allowed stepwise reserve recovery, with minimum
stored-map predicted clearance 0.450243 m (nominal footprint radius 0.45 m;
full reserve 0.48 m). Evaluator-only geometry gives actual body-centre-to-wall
horizontal distance 0.386953 m at that plan and 0.387298 m at contact. The
last dispatch check reported nearest current observed obstacle at 0.847172 m.
Maximum registered-position error over the whole run was only 4.251 mm,
median 1.310 mm: gross localization drift does not explain this clearance
discrepancy. In the final ten seconds, 490/500 requests were zero; the remaining
ten were right-turn requests. The exact perception/map/guard mechanism still
needs raw-depth diagnosis. This is not yet proof of a particular repair or a
failure attributable solely to the neural predictor.

Of 632 selected plans, 590 were on time and 42 late. On 451 matching executed
700-ms windows, direct-model XY RMSE was 9.548 mm, versus pose-command 5.681,
command-history 6.749 and nominal integration 10.977 mm. Learned yaw was
0.626 degrees, command history 0.427 and nominal 0.761. These windows differ
from JEPA's executed trajectory and must not be treated as paired accuracy
measurements. The model's improved short-motion prediction did not suffice
for successful navigation on this assignment.

`go2_short_pulse_navigation_through02_analysis_v1_attempt_001` contains the
inspected two-condition trajectory figure, both physical summaries and the
contact diagnosis. Primary artifacts remain under the assignment-2 root.
The tracking/obstacle faults saying the simulation clock closed occurred
during shutdown after the physical stop; the recorded primary failure is
`DISALLOWED_CONTACT`.

The raw-depth map diagnosis is complete. Replaying the 632 actual map updates
through frame 2524, using reconstructed live-noise packets and saved registered
poses, reproduced all 4,899 stored fine obstacle cells by count and the final
recorded start clearance exactly (0.45281328256941267 m). This replay read no
native pose or geometry. A separate evaluator then placed the saved cells in
world coordinates: near the contacted wall end they cover the previously
visible face around world y=0.60–0.61 m, while the physical wall extends from
y=0.61 to 0.69 m. The opposite face and solid thickness were absent from that
local stored-cell population. Body position error at the last plan was only
1.169 mm. The resulting map/physical clearance discrepancy is 65.860 mm.

This explains why the observed-surface clearance test could pass while the
nominal disk overlapped the actual wall. The diagnostic supports missing
occluded wall volume as a mechanism; it does not certify an alternative
controller. A later repair must obtain adequate public sensing or represent
unobserved volume conservatively, without inserting simulator wall thickness
or ground-truth geometry into control. The frozen comparison remains unchanged.
`go2_short_pulse_direct_contact_map_replay_v1_attempt_001` contains the replay
result, stored cells, separate geometry diagnosis and inspected PNG/SVG figure.
Replay implementation: `scripts/diagnose_go2_short_pulse_direct_contact_development.py`.

A separate floor-coverage probe saved the actual accumulated floor cells and
replayed the same 632 map updates. Complete observed-floor coverage of the
nominal 45-cm disk held at only 70/632 recorded states. At startup 290/292 disk
cells were unobserved; immediately before the contact, 86/291 were unobserved.
The first evaluator-confirmed nominal-disk overlap with the contacted wall
occurred at frame 1512; six newly entered unobserved floor cells were present
at that update. This is useful warning evidence, but requiring complete disk
floor coverage everywhere would reject startup and 562/632 recorded states.
That blanket rule is not a demonstrated usable repair. Footprint area newly
entering unknown space and deliberate observation before cornering remain
hypotheses requiring prospective testing; no alternative controller was run.
Artifacts: `go2_short_pulse_direct_contact_floor_replay_v1_attempt_001`.
These bounded replays used the spare CPU group while native owners remained
sequential; wall-time comparisons are still shared-host measured simulation,
not isolated hardware timing benchmarks.

The disabled contact head also failed to give a useful warning for the actual
direct-model collision. Its last selected right-turn forecast assigned contact
probabilities 0.0508%, 0.0555%, 0.0608% and 0.0666% at 500–800 ms; the physical
contact occurred 486 ms after that observation. The actual command sequence
matched through the event, so positive cumulative-contact labels remain valid
despite the subsequent physical stop. These are four overlapping horizons for
one event, not four independent contacts. Across the direct run's 4,057 matching
negative labels, 26 predictions exceeded 50% contact probability. Corresponding
counts were JEPA 165/5,361, supervised 132/5,426, and the unused supervised
reference on pose-control trajectories 203/5,101. This does not establish
probability calibration and provides no basis for simply re-enabling contact
scoring as the collision repair.

The readout is
`go2_short_pulse_contact_forecasts_first_four_v1_attempt_002/result.json`, from
`scripts/read_go2_short_pulse_contact_forecasts_development.py`. It compares
recorded physics requests with the exact float32 command representation used
by the runner. The first readout used a 1e-8 tolerance against unquantized
commands, inadvertently excluding turns (0.45 versus 0.44999998807907104).
It remains preserved and explicitly superseded. No native run or prediction
was changed; all matched labels use only executed command prefixes and native
contact is evaluator-only.

Assignment 3 (supervised rollout, maze 0) completed and was physically
evaluated after owner exit zero. It reached the outbound goal at frame 3744,
then exhausted the fixed mission budget without returning. The physical dwell
stayed within 14.283 mm, with zero requests and maximum measured 100-ms speed
15.504 mm/s. There were zero disallowed contacts. Final native home distance
was 4.104 m; total path length was 18.345 m. The result remains an incomplete
round trip.

The supervised model first observed the goal within 10 cm at frame 2658 and
took another 108.6 s to certify arrival. In that interval it selected 170 right
turns, 20 left turns, 55 right arcs and 17 forward actions; 253/262 plans were
on time. Arrival-entry gating changed 212 selections, while stopping projection
changed none. Across the mission, 951/1,190 plans were on time and 239 were late;
peak simulator lag was 136.447 s. This reinforces the terminal-selection and
timing limitations without establishing the benefit of an untested repair.

On its 434 matching executed 700-ms windows, supervised-model XY RMSE was
16.687 mm versus pose-command 9.095, command-history 12.526 and nominal 21.766.
Yaw RMSE was learned 1.204 degrees, command-history 0.621 and nominal 1.251.
The three-condition figure and summaries are in
`go2_short_pulse_navigation_through03_analysis_v1_attempt_001`; the per-run
terminal counts are `terminal_approach_diagnosis_v1.json` in assignment 3.

The completed prefix is **0/3 round trips, 2/3 verified outbound goals and one
contact failure**, all on the first new maze. This does not isolate a training
method effect or justify a reliability estimate. It also differs from the old
35/36 pilot in geometry, model training and removal of external correction, so
the old/new difference cannot be attributed to geometry alone.

Assignment 4 (pose-command, maze 0) completed the first independently verified
round trip in this comparison: 429.6 simulated seconds, 23.070 m travelled,
zero contacts, owner exit zero. Outbound/home dwell maxima were 18.624/15.608
mm, with zero requested commands and maximum measured 100-ms speeds
8.438/9.311 mm/s. The baseline used fitted pose-command XY and integrated-command
yaw, not neural forecasts for control. The common neural reference was computed
as specified by the prospective plan.

It took 13.1 s from first observing the outbound target within 10 cm to arrival,
compared with JEPA's 77.8 s and supervised rollout's 108.6 s. Home settling
took 2.0 s from the first 10-cm observation. Its arrival-entry gate changed ten
of 26 near-outbound-goal decisions; the shared rule therefore does not prevent
all successful stopping, although it interacts poorly with the neural forecasts
in the recorded trajectories. This is one run per method, not an isolated
causal estimate of the rule or general method superiority.

On 452 matching executed windows, applied XY RMSE was 9.747 mm versus the
unused supervised neural reference 18.749, command history 13.038 and nominal
24.800 mm. Applied yaw was 1.356 degrees, neural reference 1.288 and command
history 0.619. Of 1,062 plans, 867 were on time and 195 late; peak simulator
lag was 120.202 s. The initial actual-treatment evaluation compared a recorded
float64 fit directly with its float32 assigned output and rejected ordinary
rounding. The evaluator now checks the exact float32 conversion, passing every
recorded plan. No runtime, model or native outcome changed and no run was
repeated. Physical arrival evaluation then passed both dwell checks.

The four-condition figure/summaries are in
`go2_short_pulse_navigation_through04_analysis_v1_attempt_001`. The completed
prefix is now **1/4 round trips, 3/4 outbound goals and one contact failure**.
The only round trip is the simpler fitted-pose predictor; there is no learned
prediction advantage established on this new maze.

Assignment 5 (command-history, maze 0) launched next with the unchanged plan,
models and controller. Runtime session 14943 at launch. Check the live owner
and terminal artifacts before taking another action; this paragraph is a
checkpoint, not evidence that the process remains alive. Next: evaluate that
same owner after exit, then continue assignment 6 (instantaneous ranking).
Keep all four new navigation recordings in full for current diagnosis.

Routine retirement reclaimed redundant completed pulse-collection depth and
four superseded successful RGB-comparison depth recordings. All model inputs,
weights, outcomes and non-depth records remain, including all new incomplete
mission data and the earlier unresolved tracking failure. Detailed inventories
are linked from the development retention policy. Free space before assignment
2 was 7,356,522,496 bytes.

Before assignment 4, the completed third-seed coordinate-reference pair and
first-seed JEPA reference pair had their redundant depth retired under the
updated retention policy; all non-depth evidence remains. The earlier tracking
failure and its comparator stay full, as do all new navigation recordings.
Free space before assignment 4 was 6,733,127,680 bytes. A read-only review of
completed family/switch training depth found shared hard links, and its
single-link inventory check stopped before creating an inventory or deleting
anything; that original training depth remains unchanged.

A later, narrower retention pass retired only 3,996 single-link depth leaves
from 43 completed family/switch recordings whose physical, acquisition,
visibility and local-motion feature checks passed. Shared hard links, failed
recordings, one full reference per eligible source/geometry, current RGB/body/
command inputs, derived features and every non-depth file remain. This changes
the earlier read-only review's final retention state only for those inventoried
leaves. The inventory and explicit partial-retirement markers are recorded in
the retention policy; 3,274,797,056 allocated bytes were reclaimed, leaving
10,000,039,936 bytes before assignment-4 persistence. Space was about 6.3 GiB
before assignment 5.

Assignment 5 has now finished persistence, and its native owner is no longer
running. Physical evaluation passed the outbound arrival at frame 3171:
maximum native distance 18.199 mm during the one-second dwell, all requested
commands zero, maximum measured 100-ms speed 7.627 mm/s. It exhausted the
480.92-s simulation budget without returning home, with zero contacts.
Actual assigned command-history treatment passed for all 1,197 selected plans;
963 were on time and 234 late. The completed prefix is now **1/5 round trips,
4/5 verified outbound goals and one contact failure**, all on the first new
maze. The only round trip remains the pose-command baseline.

At this status check no native simulation is running. Available filesystem
space is about 2.8 GiB, below the launcher's 4-GiB reserve. Assignment 6
(instantaneous ranking) is next after routine eligible depth retirement;
nine of the fourteen fixed assignments remain. No further run has launched
at this checkpoint. The goal remains active and no learned-planning advantage
or hardware timing qualification has been established.

The storage prerequisite is now resolved for assignment 6. The diagnosed
supervised maze-0 timeout's depth was retired under the updated retention
policy: 9,610 leaves, 3,132,776,448 allocated bytes. Every outcome and non-depth
record remains; 32 JSON hashes and 9,651 other identities were checked. Its
maximum registered-position error was 6.954 mm, with no tracking or contact
failure. No pending raw-depth replay consumes it. This explicitly ends that
recording's earlier full-depth pin; JEPA, the direct collision and the
pose-command successful comparator retain their depth. Free space before
assignment 6 was 6,049,976,320 bytes.

Assignment 6 (instantaneous ranking, maze 0) launched with the unchanged
prospective controller/model settings: session 56404, PID 3883407. The live
process and worker log confirmed progress through camera frame 600, about
60 simulated seconds, at the launch checkpoint. Evaluate only after owner
exit and recording persistence; assignment 7 (reactive) follows it. These
identifiers are a checkpoint, not evidence of continued liveness.

The first-five comparison and inspected trajectory figure are saved in
`go2_short_pulse_navigation_through05_analysis_v1_attempt_001`. Its
`phase_and_frontier_diagnosis.json` separates outbound/return planning and
request counts and records exploration-view event times. Command-history
control took 317.1 s to the outbound arrival, versus pose-command's 273.2 s.
Its final near-goal approach took only 2.8 s. The first frontier-view event
lasted 72.4 s, of which only 9.2 s followed the directed-view commitment;
this differs from the neural runs' prolonged terminal approaches.

Command-history then had 163.1 s of observed return phase, versus the
pose-command success's 155.9 s, yet ended 3.980 m from home. It issued nonzero
requests for 2,062/8,161 return intervals (25.3%), versus pose-command's
3,214/7,800 (41.2%). Veto-latched intervals were 3,680 versus 2,397; missing
on-time-plan intervals were 2,126 versus 1,899. These are request-interval
counts, not measured physical-motion duration, and the executed trajectories
differ. The diagnosis rules out insufficient remaining time alone as a full
description; it does not isolate the causal benefit of an untested repair.

On command-history's 441 matching executed 700-ms windows, XY RMSE was
11.127 mm, versus recorded pose-command 8.418, unused neural reference 16.081
and nominal 20.392 mm. Its yaw RMSE was 0.635 degrees, versus neural 1.241
and nominal 1.465. Alternative forecasts are evaluated on executed actions;
they are not alternative navigation outcomes. Only one independent new maze
is represented so far.

The first-five dispatch diagnosis reconstructs the original stop from the
actual requests in each committed window, rather than trusting the stored
`initial_veto` field, which the commitment wrapper can overwrite with the
latch reason itself. Every JEPA and direct latched window began with an
unavailable/stale obstacle observation. In the return phase, stale observations
account for JEPA 2,561/2,561 latched intervals, supervised 1,874/1,893,
pose-command 2,397/2,397 and command-history 3,661/3,680. The remaining
supervised and command-history intervals followed one stopping-margin veto
each. Return stale-trigger ages were 220 ms, against the unchanged 200-ms
limit. Counts include suppressed requests, not independently measured motion.

Across these five missions, median camera acquisition was about 82–83 ms,
median observation-to-obstacle completion 112–114 ms, and its 95th percentile
122–124 ms. With 100-ms acquisitions and 20-ms command intervals, the recorded
tail delays repeatedly leave the preceding obstacle sample too old. This is
an observed sensing/processing timing mechanism, not evidence to relax the
freshness limit or proof that a timing repair alone fixes navigation. Readout
and its exact source are retained as `dispatch_stall_diagnosis.json` and
`dispatch_stall_readout.py` in the first-five analysis root. The direct failure
did not persist `measured_latency_releases.json`; its stage events are used
and the missing service-time population is explicitly marked unavailable.

A bounded public-packet replay of direct-run frames 0–127 reproduced every
independent-obstacle receipt exactly. Profiling frames 16–127 found 0.478 s
in NumPy cell uniqueness, of 1.677 s total profiled service; height-cluster
candidate work was another major cost. Artifacts:
`go2_short_pulse_obstacle_profile_v1_attempt_001`. This is an initial-scan
profile on the shared host, excluding acquisition and subprocess transfer;
it is not a full-mission or hardware latency qualification. No frozen runtime
source changed.

An isolated alternate cell-uniqueness calculation then replayed those same
128 public frames with two independent observers, alternating reference and
variant call order. Packing bounded integer XY cell pairs into scalar keys
before uniqueness preserved the exact obstacle objects and every recorded
receipt. After the first 16 frames, median service was 13.665 ms reference
versus 10.119 ms variant (25.9% lower); p95 was 15.142 versus 11.107 ms.
The saved `benchmark.py` in
`go2_short_pulse_obstacle_unique_cells_benchmark_v1_attempt_001` contains the
candidate implementation, overflow fallback and timing procedure. It is not
imported by any navigation runtime. These 112 timed frames cover only the
initial scan; full-route equivalence, end-to-end dispatch timing and prospective
navigation improvement remain untested. This gives a concrete lower-level
perception optimization to test after the fixed cohort without changing its
sensing thresholds or observed geometry.

The full-direct replay extended exact equality to all 2,532 recorded obstacle
receipts, frames 0–2531, and reference/variant obstacle objects through that
same prefix. The readout then exited one at frame 2532: there are 2,533 camera
frames but only 2,532 recorded obstacle receipts because the final camera
frame was captured during contact shutdown. Indexing a nonexistent reference
receipt raised `IndexError`. The saved source, terminal exception and inferred
completed assertion population are in
`go2_short_pulse_obstacle_unique_cells_full_direct_v1_attempt_001/result.json`,
explicitly marked `INCOMPLETE_READOUT`. The final camera frame was not paired;
the end-of-loop timing summary was not saved. This broadens output-equivalence
evidence for the consumed public stream, but supplies no full-route speedup
estimate or prospective navigation improvement. No runtime was changed.

Assignment 6 (instantaneous ranking, maze 0) finished with owner exit zero,
and physical evaluation verified the outbound arrival at frame 3451. During
the one-second dwell, maximum native distance was 13.671 mm, requests were
zero and maximum measured 100-ms speed was 17.477 mm/s. It exhausted the
480.9-s budget without returning home, with zero contacts and final native
home distance 3.969 m. Total path length was 18.396 m. Actual ranking and
predictive-guard treatment passed all 1,197 selected plans; 1,004 were on time
and 193 late. Peak simulator lag was 138.917 s.

It first observed the outbound goal within 10 cm at frame 2190, then needed
126.1 s to settle. In that interval, 300/312 plans were on time; selected
actions were 150 right turns, 104 right arcs, 50 forward and eight left turns.
Terminal-position priority changed 248 selections, arrival-entry gating 245,
and stopping projection/arrival hold none. This main-ranking intervention did
not remove the prolonged terminal approach. It retained supervised forecasts
for clearance and terminal gates, so it is not a complete prediction-off test.
On 384 matched executed 700-ms windows, neural XY RMSE was 17.305 mm versus
pose-command 10.958, command-history 13.369 and nominal 22.131; learned yaw
was 1.722 degrees versus command-history 0.872 and nominal 2.248. Different
executed trajectories prevent interpreting these as paired policy outcomes.

The completed prefix is now **1/6 round trips, 5/6 verified outbound goals
and one contact failure**, still all on the first new maze. Assignment 7
(reactive, maze 0) launched next in session 75748. Check its live process and
terminal persistence before evaluation. Eight assignments remain including
that active mission. The command-history recording's diagnosed depth was
retired before this launch, reclaiming 3,086,675,968 allocated bytes while
preserving all 32 JSON hashes and 9,651 non-depth identities. About 5.1 GiB
remained after assignment-6 persistence, before assignment 7. Neither the
cell-uniqueness candidate nor any terminal-rule change is active in this cohort.

The corrected full-direct perception benchmark completed using the actual
2,532-record observer population; shutdown-only camera frame 2532 is explicitly
excluded because it has no recorded observer output. Reference and scalar-key
cell calculations matched every obstacle object and recorded receipt. With
alternating call order and the first 16 frames excluded from timing, median
in-process service was 13.579 ms reference versus 10.088 ms candidate (25.7%
lower), and p95 was 15.507 versus 11.229 ms. All 2,516 per-frame timings and
the exact implementation are saved in
`go2_short_pulse_obstacle_unique_cells_full_direct_v1_attempt_002`; the failed
first readout remains preserved. Replay took 187.014 wall seconds. This covers
the recorded scan, navigation and collision approach, but excludes camera
acquisition and subprocess transfer and shares the host with assignment 7.
It establishes a broader equivalent-output compute improvement, not a measured
end-to-end dispatch or navigation improvement. The candidate remains offline.

Assignment 6's dispatch readout is also complete: its return phase had 147
latched windows, all triggered by stale obstacle observations, suppressing
2,521 further request intervals. Maximum registered-position error was
7.100 mm. Its completed, diagnosed depth was retired under the updated policy,
reclaiming 3,024,207,872 allocated bytes; all 33 preserved JSON hashes and
9,652 other-file identities match. Original full sensor replay is unavailable
for assignment 6; every result and non-depth record remains. Free space was
8,488,873,984 bytes while assignment 7 was still running, before persistence.

Assignment 7 (reactive, maze 0) completed with owner exit zero and both arrivals
independently physically verified: outbound frame 1259 and home frame 2244.
Dwell maximum distances were 18.777/21.563 mm, requested commands were zero,
and maximum measured 100-ms speeds were 20.597/15.273 mm/s. There were no
contacts. Round-trip simulation time was 224.8 s, compared with pose-command's
429.6 s; path length was 21.431 m and final native home distance 19.567 mm.
Of 551 plans, 547 were on time and four late. Median measured planning service
was 25.484 ms (p95 61.877 ms); independent-obstacle service remained 27.415 ms
median. Peak simulator lag was still 58.801 s, so this is not wall-clock or
hardware qualification. Near-goal/home settling took 7.9/12.4 s from first
observing the target within 10 cm.

The first independent new maze's fixed seven-controller comparison is complete:
**2/7 round trips, 6/7 verified outbound goals, one contact failure**. Reactive
and pose-command succeeded; JEPA, supervised rollout, command-history and
instantaneous ranking reached the goal but timed out returning; direct
prediction contacted the wall before arrival. This is evidence against a
practical learned-prediction advantage in this setting so far, not a general
method ranking: one execution per method on one layout, differing trajectories,
and a reactive comparator that also removes forecast-based guards and compute.
The exact comparison and seven-condition trajectory figure are in
`go2_short_pulse_navigation_maze00_complete_v1_attempt_001`.

Assignment 8 (reactive, maze 1) launched next with the fixed reverse order,
session 73872, CPU group 8–15 and 24–31. Check current liveness before acting;
evaluate after native-owner exit and persistence. Assignment 9 is instantaneous
ranking on that same second maze. Seven of fourteen assignments remain,
including the active one. All models, thresholds and runtime implementations
remain unchanged; the perception optimization is still an offline candidate.

An additional 256-frame benchmark exercised the actual `AcquiredFrame` payload
and `independent_depth_process_development.observe` request/response boundary
in separate spawned one-worker pools. It alternated reference/candidate call
order on CPU group 0–7/16–23 while assignment 8 used the other group. Every
obstacle object and recorded receipt matched. Excluding the first 16 frames,
median worker round-trip time was 16.524 ms reference versus 10.690 ms candidate;
p95 was 17.423 versus 12.868 ms. This includes worker request/response transfer,
but excludes acquisition and does not reproduce full live pipeline contention.
Artifacts and exact source are in
`go2_short_pulse_obstacle_ipc_benchmark_v1_attempt_001` (25.534 s wall time).
The tested candidate is loaded only in that benchmark's isolated worker;
no navigation implementation or threshold changed. These timings should not
be directly substituted for the live 27-ms obstacle-service measurements.

Assignment 8 (reactive, maze 1) finished with owner exit zero and both physical
arrival checks passed. Outbound/home frames were 1902/2573; maximum native
dwell distances were 25.600/20.393 mm, with zero requests and maximum measured
100-ms speeds 0.927/14.199 mm/s. There were no contacts. The round trip took
257.7 simulated seconds, travelled 20.086 m and ended 20.574 mm from home.
Of 634 plans, 616 were on time and 18 late. Peak simulator lag was 77.870 s;
median measured planning service was 28.455 ms (p95 66.637 ms). Outbound/home
approaches from first entering 10 cm took 44.4/1.8 s.

Reactive control is therefore **2/2 verified round trips, zero contacts** on
the two independent new mazes, with one run per layout. This is a successful
current-sensor/persistent-map baseline, not learned-world-model evidence or a
general reliability estimate. The paired outcome/trajectory artifact is
`go2_short_pulse_reactive_two_maze_result_v1_attempt_001`. The plot helper's
default title is now evaluated only when a custom title is absent, allowing
the two-layout figure without an irrelevant integer-format error. No runtime
source changed.

Actual reactive terminal pulses are clarified in
`reactive_actual_terminal_pulse_treatment_v1.json` under both reactive roots.
The original launch label `terminal_translation_pulses=false` was incorrect
for this inherited controller: recorded decisions use measured heading
alignment followed by 100-ms forward pulses. Original launch files remain
preserved. Thus reactive differs in its terminal rule as well as removing
forecasts and forecast-based guards. On maze 1, 919 of the 955 `NO_ON_TIME_PLAN`
intervals near the outbound goal were intentional pulse zero tails; all 106
near-goal plans were on time. That reason must not be read as pure planning
lateness. Across the whole maze-1 run, 919/1,335 no-plan intervals were pulse
tails; on maze 0, 134/265. Dispatch-stale veto diagnoses remain based on their
separate actual trigger records. No controller or observed outcome changed.

The completed prefix is **3/8 round trips, 7/8 verified outbound goals and one
contact failure**. Assignment 9 (instantaneous ranking, maze 1) is active:
session 5660, PID 3888761 at launch, with about 4.5 GiB free before launch.
Check liveness and wait for owner exit/persistence before evaluation.
Assignment 10 is command-history on maze 1. Six assignments remain including
the active mission; all three neural methods still await their second-maze
assignments. The goal remains incomplete.

The first-eight pulse-gap readout is complete in
`go2_short_pulse_navigation_pulse_gap_readout_v1_attempt_001`. It intersects
`NO_ON_TIME_PLAN` requests with the recorded on-time committed pulse's 300-ms
zero tail; it does not equate all remaining intervals with planning lateness.
For maze-0 JEPA/direct/supervised/pose-command/command-history/instantaneous,
planned tails explain respectively 120/3,573, 0/850, 964/5,858, 116/4,093,
30/4,726 and 2,291/6,167 no-plan intervals. The latter is 37.1%, so raw
no-plan counts overstate the instantaneous controller's missed-compute time.
Its separate plan records still contain 193 late plans. Return-phase no-plan
intervals for JEPA, supervised, command-history and instantaneous contain no
planned pulse tails in this population. Original dispatch reason counts remain
unchanged; this supplies their more precise interpretation.

Completed pose-command/reactive maze-0 success depths were retired after all
first-maze and reactive-pair analyses, ending their earlier full-depth pins.
Exactly 13,080 leaves reclaimed 4,164,550,656 allocated bytes; all 63 preserved
JSON hashes and 13,161 non-depth identities match. Their outcomes and all
non-depth inputs remain. Maze-1 reactive success, maze-0 JEPA and the direct
collision retain full depth, as do older active tracking-failure inputs.
Available space was 8,990,797,824 bytes while assignment 9 was running.

An isolated neural-inference benchmark evaluated the same 12 evenly spaced
public JEPA maze-0 histories and actual committed command sequences with all
three frozen models. Reference inference reproduced recorded JEPA forecasts
within 1e-5. The candidate encodes the six candidates' shared history once,
then expands its latent result before the unchanged action-conditioned
transitions and heads. With alternating call order and 48 timed forwards per
model/variant, reference/shared median milliseconds were JEPA 6.253/2.281,
direct 6.271/2.294 and supervised 6.279/2.304. Forecast validity/horizon tensors
were identical, but floating-point outcomes were not bitwise equal: maximum
absolute differences were 1.907e-6, 9.537e-7 and 9.537e-7 respectively across
outcome channels. Controller-action equality was not tested. All sources and
per-context measurements are in
`go2_short_pulse_shared_history_benchmark_v1_attempt_001`.

This saves roughly four milliseconds in an isolated forward pass; it does
not explain or remove most of the recorded roughly 100-ms planning-stage cost.
No input preparation, routing or prediction-based guard timing is included.
It remains an offline candidate and no navigation model or implementation
changed. At the latest assignment-9 checkpoint, the live owner had reached
camera frame 3600 (360 simulated seconds) but remained near its starting
position, about 4.12 m from the outbound goal. Preserve the full fixed-budget
outcome and diagnose its recorded decisions after completion.

Assignment 9 (instantaneous ranking, maze 1) finished with owner exit zero
and complete physical evaluation: no arrivals, zero contacts, budget exhausted
at 480.86 simulated seconds. It ended 4.118 m from the outbound goal and
0.139 m from home; accumulated 10-Hz path length was 1.800 m. Position error
was 3.804 mm median and 4.567 mm maximum. All 1,200 selected plans passed the
actual-treatment evaluation; 1,192 were on time and only eight late. Nonzero
requests occupied 456 twenty-millisecond intervals (9.12 s), not necessarily
9.12 s of measured physical motion.

Every decision remained in the initial panorama phase. There were 227 left-turn
selections and 973 holds, with no translation or frontier events. Four of nine
survey views completed. Frame 912 was the last non-hold selection. From frame
916 onward, all 972 decisions reported `NO_CLEAR_CANDIDATE_ZERO_REQUESTED`;
969 of these were on time. The instantaneous scorer continued preferring a
left turn, but the retained forecast-clearance filter rejected all candidates.
`initial_survey_stall_diagnosis_v1.json` preserves the survey state and exact
first-blocked/last-moving/final decisions.

The new public-only replay
`scripts/diagnose_go2_short_pulse_initial_survey_stall_development.py` rebuilt
229 map updates through frame 912, reproducing the 1,986 stored fine cells by
count and all six candidates' eight segment clearances to 1e-12 at decision
frame 916. Body-centre stored-map clearance was 0.453833 m. The shared zero-
command prefix's predicted motion reduced the first-segment clearance to
0.449124 m, below the 0.45-m nominal radius, so every candidate failed before
its distinct action interval. The nearest stored cell was [5,58], first added
at map frame 504. Replay used saved public depth and registered estimates,
not native pose or geometry; runtime code was unchanged.

A separate native evaluator found actual body-to-nearest-wall horizontal
clearance 0.471491 m at frame 916 and registered-position error 4.421 mm.
The nearest map cell spans approximately world y=0.595–0.606 m over the
retained obstacle-height range, versus the wall face at y=0.610 m. Thus the
map clearance was already 17.658 mm more conservative than physical geometry;
the forecast reduced it by another 4.709 mm. This identifies a compound
map/forecast threshold mechanism for the initial stop, not an isolated main-
ranking failure or permission to weaken the safety threshold. No alternative
controller was executed. The map-cell measurement/quantization contributions
and recovery treatment require further diagnosis. Keep this recording's depth
for that work. Artifacts:
`go2_short_pulse_initial_survey_stall_map_replay_v1_attempt_001`, including
the stored map and separate `geometry_diagnosis.json` (24.818 s public replay).

On 976 matched executed windows, neural XY RMSE was 5.839 mm versus pose-command
4.338, command-history 3.486 and nominal 3.485; learned yaw was 0.593 degrees
versus command-history 0.309 and nominal 0.315. This stalled, predominantly
hold population does not represent successful navigation prediction accuracy.

The completed prefix is **3/9 round trips, 7/9 outbound goals and one contact
failure**. Instantaneous ranking is 0/2 round trips; reactive is 2/2. Assignment
10 (command-history, maze 1) launched next in session 96305, PID 3890802,
with about 4.7 GiB free. It is the only native owner; check its current process
and wait for exit/persistence before evaluation. Assignment 11 is pose-command
on maze 1. Five fixed assignments remain including the active mission.

The nearest blocking cell's first contributing pixels are now traced in
`first_cell_pixel_attribution.json` in the survey-stall replay root. Exactly
two primary/auxiliary sampled pixels at frame 504 first populated cell [5,58].
Using native camera-frame body pose only in the evaluator, their unperturbed
rays land on the physical wall at world y=0.610000 m. The actual delivered
noisy rays land at y=0.602314–0.602489 m: about 7.5–7.7 mm nearer the robot.
Applying the saved estimated capture pose places them at y=0.604159–0.604302 m.
At decision frame 916, the nearest of these stored points is 0.462545 m from
the estimated body in map coordinates; treating its entire 1-cm cell as occupied
reduces that to 0.453833 m, another 8.712 mm. Current registered-position
error and the already measured forecast drift contribute separately.

This shows the cell originated in noisy returns from the real wall, rather
than a distinct physical obstacle. The attribution covers the pixels that
first populated this one cell, not every later observation or every retained
cell. Whether later observations support or contradict it is not established
by this readout. A future map treatment should investigate temporal surface
evidence and uncertainty; neither deleting this cell nor relaxing the radius
has been tested as a controller repair. Native truth remains evaluator-only.

Assignment 10 (command-history, maze 1) has now exited zero with complete
recording persistence and physical evaluation. It exhausted the 480.92-s
simulation budget with no arrivals and zero contacts. All 1,200 selected plans
passed actual-treatment verification; 819 were on time and 381 late. Maximum
simulator lag was 145.782 s. This is a completed unsuccessful mission, not a
running simulation or infrastructure exception.

The fixed comparison is now **10/14 evaluated: 3/10 round trips, 7/10 outbound
goals and one contact failure**. Four assignments remain: pose-command,
supervised, direct and JEPA on maze 1, in that order. No native owner is active
at this checkpoint. Disk availability is about 1.1 GiB; routine authorized
retirement of eligible completed depth is needed before the next launch's
4-GiB headroom requirement. Preserve the active collision and survey-stall
diagnosis inputs. No runtime or frozen study treatment changed.

The public-only blocking-cell reobservation readout also completed in
`go2_short_pulse_initial_survey_stall_map_replay_v1_attempt_001/`
`blocking_cell_reobservation_counts.json`: 59 sampled pixel hits populated
cell [5,58] across 32 of the 229 replayed map updates through frame 912.
Thus the earlier first-two-pixel attribution is not the cell's entire
observation history. Missing hits alone are not free-space evidence; this
readout does not establish that the cell should have been removed.

Assignment 10's dispatch readout reconstructs 616 latched windows, all from
obstacle-observation staleness at 220 ms versus the 200-ms limit. These explain
all 10,588 latched request intervals; 607 windows selected non-hold actions.
Initial panorama completion took 138.0 s versus reactive's 32.8 s on this
layout. During that panorama, 337/344 plans were on time but only 1,025/6,900
requests were nonzero; reactive had 80/81 on-time plans and 1,140/1,640 nonzero
requests. Thus initial scanning was heavily interrupted even with timely
planning. Requested intervals do not prove measured motion duration.

After the scan, command-history had only 482/856 on-time plans, with all
decisions still routing to or viewing frontiers. All 1,200 plans reported a
clear candidate, unlike assignment 9's persistent clearance rejection.
Twelve frontier-view events completed; the ninth-to-tenth routing progression
included a 164.8-s frontier event. Minimum physical goal distance was 1.410 m,
final distance 3.186 m, path length 9.044 m and maximum pose error 6.770 mm.
The trajectory follows the exploratory detour and begins the route around
the wall, but runs out of budget before reaching the goal corridor.

Median measured planning service was 110.364 ms versus reactive's 28.455 ms;
obstacle service was similar at 28.640/28.966 ms. On 180 matched executed
700-ms windows, command-history XY RMSE was 15.576 mm versus unused neural
19.491, pose-command 11.254 and nominal 28.485. These diagnoses establish
interruption and timing differences, not a counterfactual successful mission.
Artifacts under assignment 10: `dispatch_stall_diagnosis_v1.json`, its saved
readout source, and `phase_comparison_with_reactive_v1.json`. The inspected
three-control trajectory figure and summaries are in
`go2_short_pulse_navigation_maze01_through10_v1_attempt_001`.

The diagnosed maze-0 JEPA raw-depth pin was ended under the retention policy:
9,612 leaves reclaimed 3,111,563,264 allocated bytes; all 31 JSON hashes and
9,652 non-depth identities match. Outcomes and current model inputs remain.
Assignment 11 (pose-command, maze 1) launched in session 46976, PID 3892729,
with 4,307,427,328 bytes free. Its process was confirmed active at three
minutes elapsed, camera frame 1200. Wait for actual owner exit and complete
persistence before evaluation. No frozen runtime/model treatment changed.

After its completed diagnosis, assignment 10's depth was also retired under
the recorded policy: 9,610 leaves reclaimed 3,145,342,976 allocated bytes,
with all 33 JSON hashes and 9,653 non-depth identities preserved. Its failed
mission outcome remains in the comparison. Available space is now about
6.94 GiB while assignment 11 runs; direct-collision, initial-survey-stall and
reactive maze-1 reference depths remain full. Latest verified native progress:
assignment 11 process 3892729 active at 4:39 elapsed, camera frame 2000,
200.02 simulated seconds, outbound goal distance 2.765 m. Session 46976 is
the existing owner; do not restart while it is live.

Assignment 11 (pose-command, maze 1) has exited zero with full recording
persistence and physical evaluation. It exhausted the 480.94-s simulation
budget with no arrivals and zero contacts. All 1,200 plans passed actual-
treatment verification; 818 were on time and 382 late. Minimum physical
goal distance was 1.357 m, final goal distance 3.146 m, final home distance
2.339 m and 10-Hz path length 9.113 m. Registered-position error was 2.043 mm
median and 4.678 mm maximum. Maximum simulator lag was 144.571 s.

Its initial panorama took 130.0 s (295/324 plans on time, 858/6,500 nonzero
request intervals); after the scan, 523/876 plans were on time. All subsequent
plans routed to or viewed frontiers; no route-to-goal-cell phase occurred.
There were 1,193 clear-candidate selections and seven hold-relative clearance
recovery selections, with no persistent all-candidate rejection. Fourteen
frontier events are recorded. All 629 latched windows began with stale obstacle
observations at 220 ms, accounting for all 10,792 latched request intervals;
622 windows selected non-hold actions. The phase partition ends at the final
camera mission observation, so its counts exclude the remaining request tail.
Median measured planning service was 108.560 ms, obstacle service 28.517 ms,
and acquisition 82.429 ms. The source and readouts are preserved in
`dispatch_stall_readout.py`, `dispatch_stall_diagnosis_v1.json` and
`phase_diagnosis_v1.json` under assignment 11.

On 161 matched 700-ms executed windows, applied pose-command XY RMSE was
11.362 mm versus unused neural 20.657, command-history 16.441 and nominal
29.861. Applied nominal yaw RMSE was 1.475 degrees versus unused neural 1.221
and command-history 0.591. These are overlapping windows from an unsuccessful
mission, not alternative navigation outcomes. The second-maze control outcome
summaries and trajectory figure are in
`go2_short_pulse_navigation_maze01_through11_v1_attempt_001`.

The completed comparison prefix is **3/11 round trips, 7/11 outbound goals
and one contact failure**. Pose-command is now 1/2 round trips, command-history
0/2, instantaneous 0/2, reactive 2/2; all neural methods still await their
second-maze runs. Both fitted predictors made substantially less progress
than reactive on maze 1, with major dispatch interruption and planning-delay
differences. Their failed missions do not establish a timing-only explanation
or isolate the utility of learned ranking.

Assignment 11 depth was retired after diagnosis under the retention policy:
9,610 leaves reclaimed 3,180,675,072 allocated bytes; all 33 JSON hashes and
9,653 non-depth identities match. The failed outcome and other evidence remain.
Assignment 12 (supervised rollout, maze 1) launched next, session 89873,
PID 3895289, with 6,771,920,896 bytes free. Its process was confirmed active
42 seconds after launch. Assignments 13 direct and 14 JEPA follow. Wait for
owner exit and complete persistence before evaluation; no frozen treatment
or runtime source changed. The broad goal remains incomplete.

While assignment 12 runs, the previously benchmarked integer-XY grouping
optimization has been implemented as a separate candidate observer in
`lewm/packed_cell_obstacles_development.py`. It uses private function bindings
for primary/partial and auxiliary obstacle extraction; it does not mutate
NumPy globally or modify any existing launcher/observer used by this cohort.
Ten focused tests passed in 1.63 s, covering duplicates, negative/strided/empty
inputs, signed dtypes, large-coordinate/key overflow fallback and the general
NumPy API. No navigation integration or benefit is claimed.

The actual candidate class then reproduced every obstacle object and original
receipt on all 2,532 recorded observer frames of the retained direct collision
run (191.909 s). Global NumPy remained unchanged. This population contains no
auxiliary-only or unavailable-obstacle frames, so equivalence on those branches
is not established by this replay. Shutdown-only camera frame 2,532 is excluded
because it had no recorded observer result. Evidence and exact candidate source
are in `go2_packed_cell_obstacles_full_direct_equivalence_v1_attempt_001`;
source SHA-256:
`8f0d48d305f6228090355f86d7d4e250c5fa1d0efcdb9605fef7d0115a34b9d8`.
The prior prototype's service-time savings remain an offline result. This
prepares a subsequent sensor-timing experiment; the fourteen fixed missions
retain their original perception and controller code.

The candidate's missing-primary recovery branch also matched in a bounded
synthetic check: initialize both observers on original public frame 0, then
mask all primary depth returns to invalid zeros on frames 1–4 while preserving
the actual auxiliary packets and other public inputs. All four modified frames
produced available auxiliary-only evidence with identical obstacle objects and
receipts. `synthetic_primary_dropout_equivalence.json` in the equivalence root
records this protocol and result. This is a modified-packet check, not a native
dropout/navigation experiment. The ten focused tests now reside at
`tests/test_packed_cell_obstacles_development.py`; only their location and
module docstring changed after the passing run.

Assignment 12 (supervised rollout, maze 1) exited zero with complete persistence
and physical evaluation: budget exhausted at 480.90 simulated seconds, no
arrivals, zero contacts. All 1,200 selections passed actual-treatment checks;
810 plans were on time and 390 late. Minimum physical goal distance was
1.407 m, final goal distance 3.297 m, final home distance 2.151 m and path
length 9.204 m. Position error was 2.307 mm median and 5.875 mm maximum;
maximum simulator lag was 148.080 s. No wall-clock qualification follows.

Initial panorama took 126.4 s, with 309/315 plans on time and 877/6,320
nonzero request intervals. After the panorama, 501/885 plans were on time;
all decisions still routed to or viewed frontiers. All 1,200 plans reported
a clear candidate. Fifteen frontier events completed, but no goal-cell route
phase occurred. All 614 latched windows began at observation age 220 ms,
explaining all 10,472 latched request intervals; 608 windows selected non-hold
actions. Median acquisition/obstacle/planning service was 82.878/28.663/109.768
ms. The original source and readouts are retained as `dispatch_stall_readout.py`,
`dispatch_stall_diagnosis_v1.json` and `phase_diagnosis_v1.json` in this run.
These repeated timing and progress patterns support testing sensor delivery,
but do not prove that a faster observer alone would complete navigation.

On 165 matched executed 700-ms windows, applied neural XY RMSE was 20.154 mm
versus pose-command 10.664, command-history 15.264 and nominal 29.025. Applied
yaw RMSE was 1.261 degrees versus nominal 1.412 and command-history 0.640.
Unused alternatives are predictions on the executed trajectory, not executed
alternative policies. The supervised model is now 0/2 round trips and 1/2
outbound arrivals on these two development mazes.

The evaluated prefix is **3/12 round trips, 7/12 outbound arrivals and one
contact failure**. Assignment 12's diagnosed depth was retired: 9,610 leaves
reclaimed 3,134,046,208 allocated bytes; all 33 JSON hashes and 9,653 non-depth
identities match. The budget failure and all non-depth evidence remain.

Assignment 13 (direct, maze 1) is active in session 74412, PID 3897664,
confirmed live 13 seconds after launch with its launch output observed.
Available space before launch was 6,096,474,112 bytes. Assignment 14 JEPA is
last. Wait for owner exit and complete recording persistence before evaluation.
The packed-cell candidate is not installed in either remaining fixed mission.
The broad goal remains incomplete, with no established JEPA advantage.

The follow-up timing experiment is prepared in
`scripts/run_go2_obstacle_grouping_navigation_development.py` and
`docs/go2_obstacle_grouping_navigation_2026-09-16.md`. It is a fixed original-
then-packed observer pair on exposed maze 1 using the same supervised model
and controller. The reference is repeated with spare-core analysis paused for
both conditions, since historical assignment 12 included concurrent analysis.
The launcher refuses execution before assignment 14 is evaluated, preserves
separate outputs, and uses the existing physical/treatment evaluator. Its CLI
loaded successfully; neither prospective mission has been launched. This is
a timing/continuity repair test, not new-layout reliability or JEPA attribution.

Assignment 13 (direct, maze 1) exited zero with complete recording persistence
and physical evaluation. It exhausted the 480.94-s budget without arrivals or
contacts. All 1,200 selections passed actual-treatment verification; 920 plans
were on time and 280 late. Minimum physical goal distance was 1.488 m, final
goal distance 2.184 m, final home distance 2.070 m and path length 6.029 m.
Position error was 1.379 mm median and 5.508 mm maximum. Maximum simulator lag
was 146.885 s. Direct is now 0/2 round trips and 0/2 outbound arrivals, with
the distinct maze-0 contact failure retained.

The opening panorama took 178.0 s, despite 434/444 plans being on time.
Only 1,010/8,900 requests in that phase were nonzero. Afterward, 486/756
plans were on time, all routing to or viewing frontiers. Nine frontier events
completed and all 1,200 plans had a clear candidate. All 790 latched windows
began at observation age 220 ms, accounting for 13,670 stopped request intervals;
784 selected non-hold actions. Median acquisition/obstacle/planning service was
83.229/28.383/106.187 ms. Phase and dispatch readouts and the dispatch readout
source are preserved in the run directory.

On 99 matched executed 700-ms windows, direct XY RMSE was 20.888 mm versus
unused pose-command 9.006, command-history 13.393 and nominal 25.552. Direct
yaw RMSE was 0.923 degrees versus nominal 1.306 and command-history 0.484.
These errors do not represent executed alternative policies.

A new five-controller timing readout checks each stale request against actual
recorded obstacle-stage completion. Across reactive/command-history/pose-command/
supervised/direct maze-1 runs, there were zero newer obstacle stages recorded
strictly before a stale request. Stages recorded at the same timestamp numbered
90/217/240/224/222, respectively; same-time records do not establish delivery
before dispatch. Direct's next result was only 2–6 ms late at 538/790 triggers;
supervised's at 375/614. The per-trigger evidence and longer-delay distributions
are in assignment 13's `stale_obstacle_completion_comparison_v1.json`. This
supports the planned equivalent-observer speedup test, not a delivery-bug claim
or counterfactual navigation result. No freshness threshold was changed.

The evaluated prefix is **3/13 round trips, 7/13 outbound arrivals and one
contact failure**. Diagnosed assignment-13 depth retirement reclaimed
3,125,456,896 allocated bytes from 9,610 leaves; all 34 JSON hashes and 9,654
non-depth identities match. The distinct direct maze-0 collision remains full.

Assignment 14 (JEPA, maze 1), the last fixed mission, is active in session
3620, PID 3900311. Its live process and launch output were confirmed 26 seconds
after launch; available space before launch was 5,434,552,320 bytes. Wait for
actual owner exit and complete persistence, then evaluate assignment 14 and
summarize the full fourteen-run comparison. Only afterward may the prepared
original-versus-packed observer experiment begin. The broad goal is incomplete.

Assignment 14 (JEPA, maze 1) exited zero with complete persistence and physical
evaluation: budget exhausted at 480.88 simulated seconds, no arrivals, zero
contacts. All 1,200 selections passed actual-treatment verification; 659 plans
were on time and 541 late. Minimum physical goal distance was 1.459 m, final
goal distance 2.879 m, final home distance 2.763 m and path length 9.142 m.
Position error was 2.185 mm median and 4.557 mm maximum. Maximum simulator lag
was 147.659 s. Initial panorama took 98.8 s (242/246 plans on time); afterward
only 417/954 plans were on time. Sixteen frontier events completed, with no
goal-cell route phase. There were 1,199 clear-candidate selections and one
hold-relative clearance recovery, not a persistent all-candidate block.

All 459 latched windows began with stale observations, accounting for 7,778
stopped request intervals; 453 selected non-hold actions. Trigger age median
was 220 ms, maximum 300 ms. Median acquisition/obstacle/planning service was
82.508/28.292/112.784 ms. On 176 matched 700-ms executed windows, JEPA XY RMSE
was 26.735 mm versus pose-command 11.188, command-history 16.063 and nominal
29.097; yaw RMSE was 3.007 degrees versus nominal 1.439 and command-history
0.600. Phase/dispatch readouts and source remain under this recording.

All fourteen outcomes are now evaluated. The complete result is **3/14 round
trips, 7/14 outbound arrivals and one contact failure**. Reactive is 2/2 round
trips, pose-command 1/2, and every other method 0/2. The three neural methods
together are 0/6 round trips and 2/6 outbound arrivals. These are two independent
development mazes, one execution per method/layout and one training seed.
There is no demonstrated raw learned-planner or JEPA advantage here.

`scripts/read_go2_short_pulse_navigation_complete_development.py` produced
`go2_short_pulse_navigation_complete_v1_attempt_001/result.json` and the full
second-maze aggregate. Its seven-controller PNG/SVG figure was generated and
inspected. The concise scientific report, limitations and next test are in
`docs/go2_short_pulse_navigation_complete_result_2026-09-16.md`. The broad goal
remains incomplete; completing this negative experiment does not complete it.

Diagnosed JEPA maze-1 depth was retired after the full comparison: 9,610 leaves
reclaimed 3,139,473,408 allocated bytes, preserving all 33 JSON hashes and
9,653 non-depth identities. Its failed mission remains in the results.

The observer-grouping **reference** mission has now launched, session 82456,
PID 3902304, under
`go2_obstacle_grouping_navigation_reference_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001`.
Its live process and launch output were confirmed 16 seconds after launch;
available space was 4,745,621,504 bytes. No spare-core replay, training or
performance analysis may run during this mission or the subsequent packed
condition. After actual owner exit/persistence, evaluate with
`scripts.run_go2_obstacle_grouping_navigation_development --condition reference --evaluate`,
then proceed to the packed condition using the same CPU allocation and native
environment. Preserve both outcomes. See the dedicated observer-grouping plan
for the current experiment; the fourteen-run cohort is complete.
