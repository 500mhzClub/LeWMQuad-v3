# Fixed fresh-maze sparse-corner comparison

## Completed comparison

All five fixed assignments completed independently verified goal-and-home
round trips, with zero contacts, zero pipeline faults, and all nine outbound
corridor edges retraced on return. This is five treatments on **one** previously
unexecuted maze, not five independent-layout successes.

| Treatment | Round-trip simulated seconds | Plans on time | Local turn-memory plans |
| --- | ---: | ---: | ---: |
| jepa | 203.30 | 94.61% | 16 |
| supervised_rollout | 262.12 | 74.85% | 0 |
| command_history | 257.70 | 85.33% | 5 |
| reactive_feedback | 192.54 | 96.19% | 0 |
| jepa_no_route_turn_memory | 235.50 | 84.23% | 0 |

The sparse-corner tracker now supports successful navigation for every tested
controller on this layout. Reactive feedback has the shortest recorded time;
JEPA superiority is not established. The successful local-memory ablation
shows that this particular mechanism was not necessary in that execution;
it does not test navigation without the persistent map. Repeatability across
independent layouts and the incremental value of learned prediction remain
open scientific questions.

On JEPA's 465 matched, overlapping 700-ms executed windows, neural XY RMSE
was 10.20 mm versus 7.36 mm for pose-command and 8.18 mm for command-history;
neural yaw RMSE was 0.936 degrees versus 0.803 for command-history. These
matched prediction scores also do not establish a JEPA advantage.

Machine-readable outcomes and all five forecast summaries:
`docs/go2_sparse_corner_comparison_complete_result_2026-09-17.json`.
All five full recordings remain retained. The entries below record the
progression of the fixed batch; their pending statements are historical.

The sparse-corner JEPA pilot completed a verified round trip on exposed
transfer layout 1 in 255.42 s, without contacts or pipeline faults. It retained
all pose gates and the original strong-corner recovery signal. Its 81.79%
on-time plan fraction and the lack of independent-layout replication remain
limitations. This new comparison freezes that tracker/controller treatment on
previously unexecuted transfer inventory layout 0.

Run these five assignments in order, once each, without intermediate controller
or model changes or additional attempts to obtain a favorable result:

| Assignment | Treatment | Question |
| --- | --- | --- |
| 1 | Frozen JEPA readout | Current learned planner on a fresh maze |
| 2 | Frozen supervised readout | Contribution associated with training objective |
| 3 | Fitted command-history prediction | Learned versus simple motion prediction with shared forecast consumers |
| 4 | Current-waypoint reactive feedback | Predictive controller package versus a non-predictive controller |
| 5 | JEPA without interrupted-route-turn memory | Contribution of this particular local memory mechanism |

Every arm shares sparse-corner tracking, the strong-count visual recovery
signal, observed-floor/obstacle mapping, route search, six candidate primitives,
initial survey, current-depth command dispatch and the same physical maze.
The reactive controller computes the JEPA model for workload control but uses
no forecast values to select commands. It replaces predictive feasibility,
recovery, arrival and stopping selection with existing current-waypoint feedback;
it is therefore a controller-package comparison, not an isolated scoring toggle.
Its nominal current-clearance rule differs from the predictive 3-cm reserve.
The local-memory ablation skips only interrupted-route-turn memory; it retains
the persistent observed map and measured visual recovery references. Do not
describe it as removing all memory.

Use the same frozen model/readout identities, one training seed, 4800 ticks,
2-mm depth noise, ideal gyro, CPU group 8–15/24–31, 300-ms planning deadline
and 20-ms extra wait. Native runs are sequential to avoid cross-run timing
contention. Two independent offline replays previously ran usefully in parallel;
no heavy analysis runs alongside these timed native missions. Each launch
captures current hardware. No robot hardware motion is involved.

Primary outcome: independently verified goal-and-home arrival without
disallowed contact. Report every failure, goal-only outcome, physical
backtracking, deadline fraction, exercised memory decision and matched
executed-window forecast error. One run per treatment on one new maze cannot
establish reliable generalization, statistical superiority or hardware readiness.
The simulation pauses physical time during computation and is not a validated
wall-clock real-time deployment test. Non-maze tests and increasing the number
of candidate futures remain deferred.

Source inventory: `docs/go2_route_turn_memory_transfer_layout_inventory_2026-09-17.json`.
Plan: `docs/go2_sparse_corner_comparison_plan_2026-09-17.json`.
Launcher/evaluator: `scripts/run_go2_sparse_corner_comparison_development.py`.
Root pattern:
`/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1/go2_sparse_corner_comparison_{number:02d}_{arm}_noise_2mm_native_layout00_4800_v1_attempt_001`.

The launch writer explicitly records the actual inventory, avoiding the prior
pilot's inherited corridor-metadata error. Keep its corrected V2 result and full
recording as the reference. Keep this comparison's failures and full recordings
through analysis. Before starting, routine retirement of four completed older
successes reclaimed 7.72 GiB, leaving about 10.8 GiB on the output volume.
Check available headroom before each run; storage cleanup must not alter the
fixed scientific assignment order or dispose of failures/current inputs.

Assignment 1 (JEPA) launched in session 71787, owner PID 4159247. Launch
metadata identifies `CompletionRuntime`, `SparseCornerCompletionPose`, layout
index 0 and the correct `route-turn-memory-transfer-development-v1-00` inventory
entry. The fixed source/plan was prepared in session 94327, exit zero.
No outcome is yet established; assignments 2–5 remain pending.

## Assignment 1: JEPA completed

Owner session 71787 exited zero after full archival. Evaluator session 38791
exited zero: **verified round trip in 203.30 simulated seconds, zero contacts,
zero pipeline faults**. Goal/home frames were 1204/2028; maximum physical dwell
distances were 9.62/25.12 mm. Both one-second quiet arrival checks passed.
All nine unique return corridor edges reversed outbound edges, with no invalid
transitions. Tracking remained available for all 2031 camera pairs.

There were 501 plans, 474 on time (94.61%) and 27 late. Live wall time before
archival was 266.45 s. Interrupted-route-turn memory was exercised on 16 plans.
This is a successful prospective JEPA navigation result on a new maze with the
fixed completed-corner tracker. It remains one execution; comparisons and
reliability are not established until the remaining evidence is collected.
The output drive had 9.4 GiB free afterward. Assignment 2 proceeds unchanged.

## Assignment 2: supervised completed

Native session 33563 (owner PID 4160076) exited zero after persistence;
evaluation session 89949 exited zero. **Verified round trip in 262.12 simulated
seconds, zero contacts and no pipeline faults.** All nine unique return edges
reverse outbound edges, with no invalid transitions. Goal/home frames were
1860/2616; maximum physical one-second dwell distances were 12.63/11.55 mm.
Tracking remained available throughout 2619 camera pairs.

There were 648 plans, 485 on time (74.85%) and 163 late. Live wall time before
archival was 340.60 s. No route-turn-memory selection was exercised. The final
outbound approach took substantially longer than JEPA's recorded approach;
this single asynchronous pair does not isolate a training-objective effect.
Both learned arms have completed one verified round trip. Three fixed controls
remain pending. The output drive had 7.6 GiB free afterward.

## Assignment 3: command-history prediction completed

Native session 41431 (owner PID 4160908) exited zero after persistence;
evaluation session 17900 exited zero. **Verified round trip in 257.70 simulated
seconds, zero contacts and no pipeline faults.** Goal/home frames were
1577/2572; maximum physical dwell distances were 6.55/9.64 mm. Both quiet
arrival checks passed and all 2574 camera poses remained available.

All 9 unique return corridor edges reversed outbound edges, with zero invalid
transitions. There were 634 plans, 541 on time (85.33%) and 93 late. Live wall
time before archival was 335.19 s; local route-turn memory was exercised on
5 plans. All three predictive treatments have now succeeded once; the
learned forecast is not shown necessary for navigation on this layout. The
reactive and local-memory controls remain pending.

Between runs, retiring depth from the superseded interrupted-view pilot
reclaimed another 1.842 GiB (6120 leaves), leaving 7.68 GiB. All its non-depth
files and every current recording/failure are preserved. This does not change
the frozen treatment order.

## Assignment 4: reactive feedback completed

Native session 49765 (owner PID 4161819) exited zero after archival; evaluator
session 84750 exited zero. **Verified round trip in 192.54 simulated seconds,
zero contacts and no pipeline faults.** All nine unique return edges reversed
outbound edges; no invalid transitions. Goal/home frames were 1092/1921;
maximum physical dwell distances were 9.08/7.66 mm. Both quiet arrival checks
passed. Tracking covered all 1923 camera pairs.

The saved-selector reconstruction verified all 472 reactive decisions: no
forecast-based feasibility/selection consumer was retained; the model still
computed for workload control, and the actual current-depth dispatch checks
remained. 454 plans were on time (96.19%), 18 late; live wall time before
archival was 249.10 s. No route-turn-memory selection was exercised.

All four completed treatments have succeeded once. Reactive feedback currently
has the shortest recorded time, but this is one execution of each controller
package on one maze. Different predictive reserve/recovery rules and selector
costs prevent interpreting it as an isolated cost-of-prediction experiment.
It provides no evidence of JEPA superiority. The local-memory ablation remains
pending. The dedicated volume has 6.4 GiB free.

## Assignment 5: JEPA without local route-turn memory completed

The native owner completed archival. The saved independent evaluation verifies
**a round trip in 235.50 simulated seconds, zero contacts and no pipeline
faults**. Goal/home frames were 1554/2347; maximum physical dwell distances
were 18.44/18.74 mm. Both one-second quiet arrival checks passed. All nine
return edges reversed outbound edges, with no invalid graph transitions.
The recording contains 2349 camera pairs.

There were 577 plans, 486 on time (84.23%) and 91 late. No route-turn-memory
selection was exercised, as required by the ablation. Persistent mapping and
visual recovery references remained active. The final saved result completes
the five-assignment batch; no additional attempts are part of this comparison.
