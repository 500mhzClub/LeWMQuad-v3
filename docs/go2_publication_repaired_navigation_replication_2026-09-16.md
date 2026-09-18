# Matched navigation after the recovery-publication repair

Run a new fixed 15-assignment cohort with the corrected publisher in every arm.
The predecessor stopped at four outcomes on layout 0: JEPA and supervised
successes, a pose-command tracking failure and an instantaneous execution stall.
Its sixteen unexecuted assignments are not counted as failures. The subsequent
single exposed-maze instantaneous validation exercised the corrected publisher
twice and passed the complete physical round trip. Neither set of outcomes is
pooled into this new cohort.

Use only layouts 1, 2 and 3 of the already generated inventory in
`docs/go2_recovery_repaired_navigation_replication_layout_inventory_2026-09-16.json`.
No controller has executed those layouts. Keep their original construction,
physics, appearance and noise seeds, and all geometry unchanged. Exclude exposed
layout 0 because it has already supplied outcomes and debugging observations.
No fresh generation or selection based on navigation results is performed.

Each layout receives JEPA, same-data supervised rollout, fitted pose-command
prediction, instantaneous ranking with predictive guards, and reactive feedback
without forecast selection. Preserve the original rotated order: S/P/I/R/J on
layout 1, P/I/R/J/S on layout 2, I/R/J/S/P on layout 3. Keep the shared startup
repair, original heading-release behavior, persistent routing memory and the
new measured-recovery publisher in all five runtime compositions.

Primary outcome: physically verified quiet goal-and-home round trip without
contact. Preserve every failure, report per-layout outcomes and physical
backtracking, and report deadline behavior, survey deferral/recovery publications
and prediction errors. Do not alter the controller or add retries during the
batch. JEPA versus supervised tests training objective; instantaneous ranking
retains predictive guards; reactive feedback changes a broader controller
package. All arms retain memory, so this cohort does not isolate memory value.

Models, training seed 2026091001, six candidates, 0.8-second horizon, 400-ms
planning cadence, 300-ms deadline, 20-ms extra publication wait, 4800-tick budget,
2-mm depth noise, ideal gyro and physical arrival/clearance requirements stay
fixed. One execution per arm/layout and one model seed provide limited
development evidence, not broad reliability or hardware readiness. Broader
environment types remain deferred as requested.

Before launch the host had no navigation/training job, 64 GiB RAM available,
idle CPU and GPUs, and about 8.9 GiB free artifact space after retiring three
superseded plane-coverage successes. About 1.5 GiB workspace space remains.
Reuse the measured CPU/software-rendering setup and per-layout CPU groups.
Run native missions sequentially with no heavy concurrent analysis, since
scheduling affects deadline and trajectory outcomes. Evaluate each run before
the next; retire diagnosed redundant success depth under the standing policy,
preserving active failures and the exercised publication-fix reference.

Launcher: `scripts/run_go2_publication_repaired_navigation_replication_development.py`.
Prepare once with `--prepare`, then execute `--assignment N` for N=1–15 and
evaluate each with `--assignment N --evaluate` after the owner exits. The three
publication regression tests already passed; no new model or scientific selector
implementation is introduced here.

Assignment 1 (supervised, layout 1) terminated normally at the 480.88-second
simulation budget. The goal arrival at frame 1891 passed physical distance and
quiet-motion checks, with zero contacts, but home was not reached. There were
1066/1195 on-time plans, 696 visual-recovery plans, ten recovery publications
and no pipeline faults. Startup completed all nine views without deferral.
Eleven unique outbound corridor edges were traversed; no return corridor edge
was traversed. Preserve the full recording as an active navigation failure.

The exploratory late-return diagnostic covers frames 2200–4800: all 651 plans
requested measured-view recovery, 647 were on time, and actions were 647 holds,
one right arc and three pure turns. Every held plan had both turns clear of the
nominal footprint but blocked by additional reserve/recovery checks. In every
held plan at least one arc passed the recorded predictive-clearance and stopping
checks. This identifies a possible conflict between view recovery and clearance
recovery; it does not prove an unexecuted arc safe or a different mission
successful. Do not change the policy during this cohort. Detailed receipts are
in `return_visual_recovery_stall_v1.json` in the run root.

One of fifteen assignments is evaluated. The aggregate is
`go2_publication_repaired_replication_readout_v1_attempt_001/result.json`.
Continue with assignment 2, pose-command prediction on the same layout.

Assignment 2 (pose-command, layout 1) lost visual tracking after 619 camera
frames, before a goal arrival, with zero contacts. All 153 plans were on time;
53 requested visual recovery. The initial survey deferred after three completed
views, and fourteen recovery publications occurred without an execution stall.
The final three plans requested right turns for visual recovery, but tracking
still failed. This failure occurred despite both the startup repair and the
publication-lock repair. Preserve its full recording for later perception and
recovery diagnosis. Two assignments are evaluated; neither completed a round
trip, and the supervised arm alone has a verified goal arrival so far.

Assignment 3 (instantaneous ranking, layout 1) passed both physical quiet arrivals
in 366.58 simulated seconds with zero contacts. Its return reversed all eleven
unique outbound corridor edges, with no invalid transitions. The initial survey
deferred after two completed views; nine recovery publications and 34 recovery
plans occurred, with no pipeline fault. This is the first verified round trip
of the corrected cohort, while the other two outcomes remain failures.

Only 572/905 plans were on time. The main delay interval was outbound frames
800–2399: 82/397 plans were on time, median route computation was 66.6–67.9 ms
and median whole-plan wall time including release wait was about 290 ms. Outside
that interval, route medians were 19.0–25.0 ms; model-forward medians across all
intervals stayed between 8.6 and 11.2 ms. The fixed 20-ms extra wait crossed 131
deadlines on the recorded trajectory. These profiles locate the slowdown without
establishing a unique causal decomposition or a counterfactual navigation time.
Receipts are in `planning_delay_interval_readout_v1.json`.

After evaluation and diagnosis, assignment 3's redundant depth was retired:
7326 leaves and 2,336,083,968 allocated bytes reclaimed, with all 7409 non-depth
files and 42 JSON identities preserved. No raw replay is pending; the exercised
publication-fix reference and both current failures remain full. Exact sensor
replay of this success is no longer available. Three of fifteen assignments
are evaluated; proceed to reactive feedback on layout 1.

Assignment 4 (reactive feedback, layout 1) exhausted the 480.80-second budget
after a physically verified goal arrival at frame 2635, without reaching home.
There were zero contacts, no pipeline faults and 1112/1196 on-time plans.
The return reversed eight observed outbound corridor edges, with no invalid
graph transitions; twelve unique directed outbound edges were recorded.
Startup deferred after three completed views. There were 186 visual-recovery
plans and 34 recovery publications. The full failure recording is retained.

The final frames 3600–4800 contain 301 hold plans, 290 on time. All requested
an additional view, all marked the current nominal disk blocked, and all six
candidate actions were ineligible. Stored clearance ranged from 0.4279 to
0.4377 m, below the unchanged 0.45-m footprint radius. No visual-support
recovery was active in this interval, and forecast values were not used for
selection. This distinguishes the observed stall from a missed-deadline or
publication-lock stall. It does not establish whether stored obstacles were
accurate or whether any unexecuted alternative would have been safe.
Receipts are in `return_clearance_stall_v1.json` in the run root.

Four of fifteen assignments are evaluated: one round trip, two goal-only
budget failures and one tracking failure. Continue with JEPA on layout 1,
keeping the frozen batch unchanged. To retain this reactive failure in full
and make room, retired two superseded, fully analyzed gyro-estimator success
depth recordings under the standing policy: 11432 leaves, 3366072320 allocated
bytes reclaimed. All non-depth identities and JSON hashes were preserved.
Inventories are `.generated/depth_retirement_superseded_gyro_comparison_success00_2026-09-17/`
and the corresponding `success01` directory. Neither original sensor replay
remains available; the completed estimator replay, outcomes and all failures remain.

Assignment 5 (JEPA, layout 1) launched in session 93270, owner PID 4064524,
on the fixed odd-layout CPU group. Its output root is
`go2_publication_repaired_replication_jepa_noise_2mm_native_layout01_4800_v1_attempt_001`.
Outcome is pending; monitor the existing owner rather than launching a duplicate.

Assignment 5 completed with owner exit 0 and independent evaluation: no goal
arrival, budget exhausted at 480.94 simulated seconds, zero contacts and
1163/1200 on-time plans. Startup deferred after three completed views; 61
visual-recovery plans and sixteen recovery publications occurred without a
pipeline fault. Preserve the full failure recording.

Frames 2000–4800 contain 701 consecutive hold plans, 680 on time. Every route
requested an additional view and all six predicted paths failed nominal
footprint clearance. Evaluator-only native base-centre distance to the closest
physical wall was median 0.45523 m, range 0.44734–0.45729 m; 72/701 samples
were below the 0.45-m nominal radius. These are current physical distances,
not the model's future-path clearances. The evidence identifies a sustained
clearance/view-recovery stall, but does not isolate how it entered this state
or establish a safe alternative. Full receipts are in
`outbound_clearance_stall_v1.json`.

Additional reactive-failure geometry analysis: over frames 3600–4800, the
physical wall clearance was median 0.44307 m, range 0.44111–0.45104 m.
It was below the nominal radius in 283/301 samples. Stored-map clearance
underestimated physical clearance by median 13.17 mm. This is not solely a
false map obstruction: the physical centre also entered the nominal margin.
The nominal disk differs from articulated-body collision geometry, and zero
contacts do not establish that an unexecuted turn or escape would be safe.
Details: `return_stall_physical_wall_clearance_v1.json` in assignment 4.

The first complete five-arm layout now has these fixed outcomes:

| Controller | Goal | Home | Contacts | On-time plans |
| --- | --- | --- | ---: | ---: |
| Supervised rollout | Yes | No, budget | 0 | 1066/1195 |
| Pose-command | No, tracking failure | No | 0 | 153/153 |
| Instantaneous ranking | Yes | Yes | 0 | 572/905 |
| Reactive feedback | Yes | No, budget | 0 | 1112/1196 |
| JEPA | No, budget | No | 0 | 1163/1200 |

This single layout supplies no JEPA advantage and shows that high on-time
fractions do not ensure navigation success. Complete the two remaining
layouts without tuning or replacing these outcomes. Five of fifteen runs
are evaluated; next is pose-command on layout 2, assignment 6.

Routine retention reclaimed 6225154048 allocated bytes from three older,
completed and diagnosed successes: shared-recovery reactive layout 0,
heading-repeatability repetition 1 layout 0, and survey-reposition layout 1.
Every non-depth identity and JSON hash was preserved. Their exact sensor
replay is no longer available. Inventories are
`.generated/depth_retirement_superseded_shared_recovery_success00_2026-09-17/`
and `.generated/depth_retirement_superseded_heading_survey_success00_2026-09-17/`
plus the corresponding `success01` directory. All failures and current raw
debugging references remain full; artifact free space is about 6.26 GiB.

Assignment 6 (pose-command, layout 2) launched in session 43397, owner PID
4067692, on CPUs 0–7,16–23. Its output root is
`go2_publication_repaired_replication_pose_command_noise_2mm_native_layout02_4800_v1_attempt_001`.
Outcome is pending; continue monitoring this owner.

Assignment 6 completed with owner exit 0. Both physical quiet arrivals passed:
goal frame 1121 and home frame 1605, in 160.88 simulated seconds, with zero
contacts. All six unique outbound corridor edges were traversed in reverse,
with no invalid graph transitions. There were 384/393 on-time plans, nine
visual-recovery plans and two recovery publications, without pipeline faults.
The initial survey completed all nine views without deferral. The fixed 20-ms
extra wait crossed six deadlines on the recorded trajectory.

After arrival, forecast, timing and backtracking evaluation, retired the
redundant success depth: 3214 leaves and 948391936 allocated bytes reclaimed,
preserving all 3295 non-depth files and 40 JSON identities. Inventory:
`.generated/depth_retirement_publication_repaired_replication_assignment06_2026-09-17/`.
Exact original sensor replay is unavailable; all failures and active debugging
references remain full. Six of fifteen assignments are evaluated, with two
round trips. Next is instantaneous ranking on layout 2, assignment 7.

Assignment 7 launched in session 14961, owner PID 4068900, on CPUs 0–7,16–23.
Root: `go2_publication_repaired_replication_instantaneous_noise_2mm_native_layout02_4800_v1_attempt_001`.
Outcome is pending; monitor this existing owner.

Assignment 7 completed with owner exit 0 and verified physical quiet arrivals
at goal frame 924 and home frame 1913. Duration was 191.70 simulated seconds,
with zero contacts, 456/469 on-time plans and no pipeline faults. The physical
return reversed all six unique outbound edges, with no invalid transitions.
Startup completed all nine survey views without deferral; 45 visual-recovery
plans and 21 recovery publications occurred. The extra 20-ms wait crossed six
deadlines on the recorded trajectory.

The instantaneous arm now has two round trips from two assignments, while
retaining predictive clearance guards. On this layout it took 30.82 seconds
longer than pose-command, despite an earlier goal arrival; this is one execution
per condition, with different trajectories, and not a general ranking claim.
After all routine evaluations, retired its redundant depth: 3830 leaves,
1151565824 allocated bytes, all 3912 non-depth files and 41 JSON identities
preserved. Inventory:
`.generated/depth_retirement_publication_repaired_replication_assignment07_2026-09-17/`.
Exact sensor replay is no longer available. Seven of fifteen runs are evaluated,
with three round trips; proceed to reactive feedback on layout 2, assignment 8.

Assignment 8 launched in session 73938, owner PID 4070214, on CPUs 0–7,16–23.
Root: `go2_publication_repaired_replication_reactive_feedback_noise_2mm_native_layout02_4800_v1_attempt_001`.
Outcome is pending; monitor the existing owner.

Assignment 8 completed with owner exit 0 and independently verified goal and
home arrivals at frames 1293 and 1773. Duration was 177.76 simulated seconds,
with zero contacts, 417/432 on-time plans and no pipeline faults. All six
unique outbound edges were reversed on return, with no invalid transitions.
Startup completed nine survey views without deferral. There were thirteen
visual-recovery plans and two recovery publications; the extra 20-ms wait
crossed six deadlines on the recorded trajectory.

Pose-command, instantaneous ranking and reactive feedback have all completed
layout 2; JEPA and supervised remain. Their single-run times so far are
160.88, 191.70 and 177.76 simulated seconds respectively. These are complete
controller outcomes with different trajectories, not isolated causal effects
of forecasting or evidence of general superiority.

Retired assignment 8's completed, diagnosed redundant depth: 3550 leaves,
1084100608 allocated bytes, all 3632 non-depth files and 41 JSON identities
preserved. Inventory:
`.generated/depth_retirement_publication_repaired_replication_assignment08_2026-09-17/`.
Exact original sensor replay is unavailable. Eight of fifteen assignments are
evaluated, with four round trips. Next is JEPA on layout 2, assignment 9.

Assignment 9 launched in session 43887, owner PID 4071531, on CPUs 0–7,16–23.
Root: `go2_publication_repaired_replication_jepa_noise_2mm_native_layout02_4800_v1_attempt_001`.
Outcome is pending; monitor this owner before evaluation or further dispatch.

Assignment 9 completed with owner exit 0 and verified physical quiet arrivals
at goal frame 1339 and home frame 1839, in 184.48 simulated seconds. Zero
contacts, no pipeline faults, 442/453 on-time plans. The physical return reversed
all six unique outbound edges without invalid transitions. Startup completed
all nine survey views without deferral; no visual-recovery plans or recovery
publications were needed. The extra 20-ms wait crossed five deadlines on the
recorded trajectory. This success remains separate from JEPA's layout-1 failure.

After physical, forecast, timing and backtracking evaluation, retired the
redundant success depth: 3682 leaves, 1118760960 allocated bytes, all 3763
non-depth files and 40 JSON identities preserved. Inventory:
`.generated/depth_retirement_publication_repaired_replication_assignment09_2026-09-17/`.
Exact original sensor replay is unavailable. Nine of fifteen assignments are
evaluated, with five round trips. All four completed layout-2 conditions passed;
supervised rollout is next, assignment 10. No controller settings changed.

Assignment 10 launched in session 8552, owner PID 4072839, on CPUs 0–7,16–23.
Root: `go2_publication_repaired_replication_supervised_rollout_noise_2mm_native_layout02_4800_v1_attempt_001`.
Outcome is pending; continue this owner rather than launching another attempt.

Assignment 10 completed with owner exit 0 and verified physical quiet arrivals
at goal frame 849 and home frame 1456. Duration was 146.02 simulated seconds,
zero contacts, 347/357 on-time plans and no pipeline faults. All six unique
outbound edges were reversed without invalid transitions. Startup completed
all nine views without deferral; ten visual-recovery plans and five recovery
publications occurred. The extra 20-ms wait crossed six deadlines on the
recorded trajectory.

The second layout's complete five-arm result:

| Controller | Verified round trip | Simulated seconds | Contacts | On-time plans |
| --- | --- | ---: | ---: | ---: |
| Pose-command | Yes | 160.88 | 0 | 384/393 |
| Instantaneous ranking | Yes | 191.70 | 0 | 456/469 |
| Reactive feedback | Yes | 177.76 | 0 | 417/432 |
| JEPA | Yes | 184.48 | 0 | 442/453 |
| Supervised rollout | Yes | 146.02 | 0 | 347/357 |

Supervised was fastest in this single set and 38.46 seconds faster than JEPA;
the fitted pose-command baseline also completed faster than JEPA. All five
reversed the six outbound edges. These single executions on one layout do
not establish statistical superiority or isolate memory's causal contribution.
Across the first two layouts there are six round trips in ten assignments;
instantaneous ranking has two, each other arm one. Every failure is included.

After evaluation, retired assignment 10's redundant depth: 2916 leaves,
855142400 allocated bytes, all 2997 non-depth files and 40 JSON identities
preserved. Inventory:
`.generated/depth_retirement_publication_repaired_replication_assignment10_2026-09-17/`.
Exact sensor replay is unavailable. Next is the third layout, with unchanged
order instantaneous, reactive, JEPA, supervised, pose-command (assignments 11–15).

Assignment 11 launched in session 10570, owner PID 4074065, on CPUs 8–15,24–31.
Root: `go2_publication_repaired_replication_instantaneous_noise_2mm_native_layout03_4800_v1_attempt_001`.
Outcome is pending; monitor this existing owner before evaluating or dispatching.

Assignment 11 exited 1 after visual tracking failed (`measured visual pose
unavailable`), with 1379 acquired camera pairs and 1374 registered poses.
Independent evaluation found no verified arrival and zero contacts. There were
335/343 on-time plans, thirty visual-recovery plans and seven recovery
publications. Startup deferred after six completed survey views. The extra
20-ms wait crossed four deadlines on the recorded trajectory. Retain the full
failure recording; do not retry or replace it in this cohort.

During goal-area frames 900–1372, all 119 plans were on time. Selected actions
were 72 right turns, 43 left turns, two right arcs and two forward actions;
26 plans requested visual recovery. The final three support records show zero
selected primary-camera features, with 44, 52 and 48 auxiliary features. Full-run
registered-position error was median 5.59 mm, maximum 7.48 mm. These observations
locate a terminal-approach/visual-recovery failure despite timely plans; they do
not isolate the estimator's rejection cause or prove another action would work.
Receipts: `terminal_goal_area_tracking_failure_v1.json` in the run root.

Eleven of fifteen assignments are evaluated, with six round trips. The
instantaneous arm finishes its three fixed layouts at two round trips and one
tracking failure. Continue with reactive feedback on layout 3, assignment 12.

Assignment 12 launched in session 78973, owner PID 4075339, on CPUs 8–15,24–31.
Root: `go2_publication_repaired_replication_reactive_feedback_noise_2mm_native_layout03_4800_v1_attempt_001`.
Outcome is pending; monitor the existing owner before evaluation.

Assignment 12 completed with owner exit 0 and verified physical quiet arrivals
at goal frame 1364 and home frame 1844. Duration was 184.78 simulated seconds,
zero contacts, 436/452 on-time plans and no pipeline faults. The return reversed
six observed outbound corridor edges, with no invalid transitions; eight unique
directed outbound edges were recorded. Startup deferred after four completed
views. Twenty-one visual-recovery plans and four recovery publications occurred.
The extra 20-ms wait crossed two deadlines on the recorded trajectory.

Reactive feedback finishes the fixed cohort with two round trips and one
goal-only budget failure. Twelve of fifteen assignments are evaluated, with
seven round trips. JEPA on layout 3 is next, assignment 13.

Retired assignment 12's completed, diagnosed redundant depth: 3692 leaves,
1114759168 allocated bytes, all 3774 non-depth files and 41 JSON identities
preserved. Inventory:
`.generated/depth_retirement_publication_repaired_replication_assignment12_2026-09-17/`.
Also retired the superseded matched-training supervised layout-2 success after
checking its completed twelve-trial comparison: 6164 leaves, 1778184192 allocated
bytes, all 6202 non-depth files and 28 JSON identities preserved. Inventory:
`.generated/depth_retirement_superseded_matched_supervised_layout02_2026-09-17/`.
Their exact historical sensor replay is unavailable. All failures and current
raw debugging references remain full; recording headroom is restored.

Assignment 13 launched in session 74882, owner PID 4076860, on CPUs 8–15,24–31.
Root: `go2_publication_repaired_replication_jepa_noise_2mm_native_layout03_4800_v1_attempt_001`.
Outcome is pending; monitor this owner before evaluation or further dispatch.

Assignment 13 completed with owner exit 0 and verified physical quiet arrivals
at goal frame 860 and home frame 1345, in 134.84 simulated seconds. Zero
contacts, no pipeline faults and 311/330 on-time plans. The return reversed six
observed outbound corridor edges, with no invalid transitions; eight unique
directed outbound edges were recorded. Startup completed all nine views without
deferral. No visual-recovery plan or publication was needed. The extra 20-ms
wait crossed nine deadlines on the recorded trajectory.

JEPA finishes the three fixed layouts with two round trips and one no-goal
budget failure. Its third-layout time was 49.94 seconds shorter than reactive
feedback's, but this is one execution per condition, with different trajectories;
the two remaining arms and full-cohort comparison are still pending.

After evaluation, retired the redundant success depth: 2694 leaves,
816988160 allocated bytes, all 2775 non-depth files and 40 JSON identities
preserved. Inventory:
`.generated/depth_retirement_publication_repaired_replication_assignment13_2026-09-17/`.
Exact sensor replay is unavailable. Thirteen of fifteen runs are evaluated,
with eight round trips. Supervised rollout on layout 3 is next, assignment 14.

Assignment 14 launched in session 82116, owner PID 4077983, on CPUs 8–15,24–31.
Root: `go2_publication_repaired_replication_supervised_rollout_noise_2mm_native_layout03_4800_v1_attempt_001`.
Outcome is pending; monitor the existing owner before evaluation.

Assignment 14 completed with owner exit 0 and verified physical quiet arrivals
at goal frame 928 and home frame 1635, in 164.24 simulated seconds. Zero
contacts, no pipeline faults and 387/397 on-time plans. All six unique outbound
edges were reversed on return, with no invalid graph transitions. Startup
deferred after six completed views; five visual-recovery plans and one recovery
publication occurred. The extra 20-ms wait crossed four deadlines on the
recorded trajectory.

Supervised finishes at two round trips and one goal-only budget failure. It
was 29.40 seconds slower than JEPA on layout 3, while 38.46 seconds faster on
layout 2. Both have two round trips from three mazes; these observations do
not establish a consistent training-objective advantage.

After evaluation, retired the redundant success depth: 3276 leaves,
991100928 allocated bytes, all 3357 non-depth files and 40 JSON identities
preserved. Inventory:
`.generated/depth_retirement_publication_repaired_replication_assignment14_2026-09-17/`.
Exact sensor replay is unavailable. Fourteen of fifteen assignments are
evaluated, with nine round trips. The final run is pose-command on layout 3,
assignment 15, before the complete-cohort scientific readout.

Assignment 15 launched in session 60043, owner PID 4079272, on CPUs 8–15,24–31.
Root: `go2_publication_repaired_replication_pose_command_noise_2mm_native_layout03_4800_v1_attempt_001`.
Outcome is pending; monitor this owner before evaluation and cohort finalization.

Assignment 15 completed with owner exit 0 and verified physical quiet arrivals
at goal frame 896 and home frame 1349, in 135.26 simulated seconds. Zero
contacts, no pipeline faults, 319/330 on-time plans. All six outbound edges
were reversed without invalid transitions. Startup completed all nine views;
no visual-recovery plan or publication was needed. The extra wait crossed four
deadlines and was measured at 20–22 ms.

All fifteen fixed assignments are complete: ten round trips, twelve goal
arrivals, zero contacts, two tracking failures and three budget failures.
Each arm completed two of three mazes. The complete scientific readout and
visually inspected outcome figure are saved; see
`docs/go2_publication_repaired_navigation_replication_complete_result_2026-09-17.md`.
No further run belongs to this fixed cohort. The broad navigation goal remains
incomplete; the next work targets the documented clearance/view-recovery conflict.

Assignment 15's diagnosed redundant depth was retired after evaluation:
2702 leaves, 808472576 allocated bytes, all 2783 non-depth files and 40 JSON
identities preserved. Inventory:
`.generated/depth_retirement_publication_repaired_replication_assignment15_2026-09-17/`.
All ten successes now have depth-retirement markers; all five failures remain
full. Exact historical sensor replay is unavailable for retired successes.
