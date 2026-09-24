# Sparse-corner navigation replication

Current result: **all 15 assignments complete: fourteen successes and one failure**.
Command-history failed before any arrival on replication layout 1 after a
turn/recovery loop followed by loss of visual pose. Its full recording is kept.
The completed five-treatment set on replication layout 0 has verified round
trips, zero contacts,
zero pipeline faults and all nine return corridor edges reversing outbound
edges. Including the preceding fixed study gives **19 successful round trips
in 20 runs on four prospective layouts**, with zero contacts in all runs.
These are four independent layouts, not twenty independent-layout trials.

| Treatment | Preceding comparison, seconds | Replication layout 0, seconds | Replication layout 1, seconds | Replication layout 2, seconds | Round trips |
| --- | ---: | ---: | ---: | ---: | ---: |
| JEPA | 203.30 | 234.08 | 366.12 | 207.72 | 4/4 |
| Supervised | 262.12 | 253.68 | 230.52 | 237.54 | 4/4 |
| Command-history | 257.70 | 240.90 | Failed: no arrival, tracking lost | 260.90 | 3/4 |
| Reactive feedback | 192.54 | 254.90 | 222.54 | 191.86 | 4/4 |
| JEPA without local turn memory | 235.50 | 262.08 | 239.70 | 225.68 | 4/4 |

Reactive was fastest on three of four layouts, JEPA on replication layout 0.
All treatments succeeded except command-history on replication layout 1.
The four other arms are each 4/4; command-history is 3/4. This improves
end-to-end navigation evidence but does not
establish JEPA superiority, a learned-prediction necessity, broad reliability
or a persistent-memory contribution. No local turn-memory selections occurred
on the first two replication layouts. On the final layout JEPA selected local
memory in 25 plans; 19 reached recorded command execution. The local ablation
does not remove map memory.
Matched executed-window translation errors favor the pose-command predictor
over the neural predictor in all 20 run summaries. These overlapping windows
describe executed paths, not the accuracy of unexecuted candidate futures.

Machine-readable first-layout results:
`docs/go2_sparse_corner_replication_layout00_result_2026-09-17.json`.
Second-layout results: `docs/go2_sparse_corner_replication_layout01_result_2026-09-17.json`.
Final-layout results: `docs/go2_sparse_corner_replication_layout02_result_2026-09-17.json`.
Complete results: `docs/go2_sparse_corner_replication_complete_result_2026-09-17.json`.
All native jobs and independent evaluations have finished. Historical launch
and pending statements below record progress rather than current batch status.

The preceding fixed five-treatment comparison completed five verified round
trips without contacts on one new maze. Reactive feedback was fastest and the
matched executed-window prediction scores did not establish a JEPA advantage.
The next experiment tests whether this navigation performance transfers to
three additional same-family mazes with no controller or model changes.

The new inventory excludes the explicit 100-layout development registry by
the existing topology/embedding criteria. Its construction seed is 2026091761;
it is selected without runtime outcomes. Run all five treatments once on each
layout. Fixed orders are JEPA/supervised/command-history/reactive/no-local-memory
on layout 0; command-history/reactive/no-local-memory/JEPA/supervised on layout 1;
and no-local-memory/JEPA/supervised/command-history/reactive on layout 2. These
three cyclic orders vary execution position but do not fully balance it.

Reuse the completed sparse-corner tracker, original strong-corner recovery
signal, polygon floor map, axis routing, six candidate primitives, frozen model
readouts, 2-mm depth noise, ideal gyro, 4800 ticks, CPU group 8–15/24–31, 300-ms
planning deadline and 20-ms extra wait. Native jobs run sequentially; capture
hardware before each launch. No heavy analysis runs during timed navigation.
Evaluate each archived run before starting its successor. Retain failures and
report every assigned outcome; no extra attempts to obtain favorable results.

Primary outcome is independently verified physical goal-and-home arrival with
no disallowed contact. Report backtracking, tracking, deadline fractions,
local memory selections and matched executed-window prediction errors.
Reactive remains a controller-package comparison. The memory ablation removes
only interrupted-route-turn memory, retaining mapping and visual recovery.
One training seed, three layouts and differing asynchronous trajectories limit
causal interpretation. Simulation pauses physics for computation; no realistic
gyro, wall-clock deployment or robot hardware validation is claimed.

Plan: `docs/go2_sparse_corner_replication_plan_2026-09-17.json`.
Inventory: `docs/go2_sparse_corner_replication_layout_inventory_2026-09-17.json`.
Launcher/evaluator: `scripts/run_go2_sparse_corner_replication_development.py`.
The plan fixes 15 assignments before any native execution. Check recording
headroom before each run. Completed new success depth may be retired after
per-run analysis under the existing policy, retaining the preceding complete
five-arm comparison as the full reference until batch review. Every failure
remains full. This avoids accumulating unnecessary successful depth recordings.

Preparation completed in session 63869, exit zero. All three initial geometry
candidates were accepted, with three distinct abstract topology/embedding
groups disjoint from the explicit 100-layout registry. No runtime outcomes
were used in their selection. After retirement of two superseded success
recordings, 7.85 GiB was available. Assignment 1 (JEPA, layout 0) launched in
session 69035; its outcome is pending.

## Assignment 1 completed: JEPA, layout 0

Native session 69035 (owner PID 4164208) and evaluation session 35848 exited
zero. Independently verified round trip in **234.08 simulated seconds**, zero
contacts and pipeline faults. Tracking covered all 2339 camera pairs. Goal/home
frames were 1593/2337; maximum physical one-second dwell distances were
21.82/6.02 mm, and both quiet-arrival checks passed. Nine unique return edges
reversed outbound edges (11 unique outbound edges); no invalid transitions.

548 of 576 plans were on time (95.14%), with 28 late. No local route-turn-memory
selection was exercised. Live wall time before archival was 309.91 seconds.
This adds a second prospective layout with successful fixed sparse-corner
JEPA navigation, counting the preceding comparison separately. It establishes
neither broad reliability nor superiority over the pending matched controls.
The output volume had 6.18 GiB free after evaluation.

On 545 matched overlapping 700-ms executed windows, JEPA XY RMSE was 9.55 mm
versus pose-command 6.96 mm and command-history 7.75 mm; neural yaw RMSE was
0.861 degrees versus command-history 0.759 degrees. Prediction accuracy did
not establish a JEPA advantage on its executed path.

## Assignment 2 completed: supervised, layout 0

Native session 15923 and evaluation session 74492 exited zero. Independently
verified round trip in **253.68 simulated seconds**, zero contacts and pipeline
faults. Tracking covered all 2534 pairs. Goal/home frames were 1425/2532;
maximum physical one-second dwell distances were 15.08/9.01 mm. Both quiet
arrival checks passed. All nine return edges reversed outbound edges (11
unique outbound edges), with no invalid graph transitions. No local turn-memory
selection was exercised. 580 of 620 plans were on time (93.55%), 40 late; live
wall time before archival was 331.92 seconds.

On 573 matched overlapping 700-ms executed windows, neural XY RMSE was
9.80 mm versus pose-command 7.12 mm and command-history 8.84 mm; neural yaw
RMSE was 0.849 degrees versus command-history 0.712 degrees. Different executed
paths limit cross-arm prediction comparisons. Both learned arms succeeded;
the three controls on this layout and all ten later-layout assignments remain.
The output volume had 4.42 GiB free after evaluation.

## Assignment 3 completed: command-history, layout 0

Native session 9180 and evaluator session 26243 exited zero. Verified round
trip in **240.90 simulated seconds**, zero contacts and pipeline faults.
All 2407 camera pairs remained tracked. Goal/home frames were 1696/2405;
maximum physical dwell distances were 6.65/4.55 mm, with both quiet-arrival
checks passed. Nine return edges reversed outbound edges (11 outbound), with
zero invalid graph transitions. No local route-turn-memory selections occurred.
524 of 595 plans were on time (88.07%), 71 late. Live wall time before archival
was 315.86 seconds.

On 520 matched overlapping 700-ms windows, applied command-history XY RMSE
was 7.59 mm, neural 9.55 mm and pose-command 6.71 mm. Applied yaw RMSE was
0.721 degrees versus neural 0.860 degrees. All three predictive arms have now
succeeded on this layout; learned prediction has not been shown necessary.
The reactive and local-memory controls remain pending. Free space was 2.75 GiB.

Per the prospective storage policy, assignment 2 and 3 success depth is now
eligible for retirement: physical/tracking/forecast/backtracking analyses are
complete, and neither is a raw replay or training input. Preserve all non-depth
data, assignment 1 as the full new-layout reference, and the preceding full
five-arm comparison. No outcome or controller change accompanies retirement.
Completed retirement reclaimed 2.80 GiB, leaving 5.54 GiB before assignment 4.

## Assignment 4 completed: reactive feedback, layout 0

Native session 85288 (owner PID 4166966) and evaluator session 4829 exited
zero. Verified round trip in **254.90 simulated seconds**, zero contacts and
pipeline faults. Goal/home frames were 1177/2545; maximum physical dwell
distances were 10.37/10.45 mm, with both quiet-arrival checks passed. Tracking
covered 2547 pairs. All nine return edges reversed outbound edges (11 outbound),
with zero invalid transitions. The saved reactive-selector verification passed.
No local route-turn-memory selections occurred. 595 of 618 plans were on time
(96.28%), 23 late; live wall time before archival was 332.99 seconds.

The reactive controller reached the goal first but spent longer at the final
home approach. JEPA finished faster on this layout; reactive finished faster
on the preceding comparison layout. Neither ordering establishes superiority.
On 571 overlapping 700-ms windows, unused neural XY RMSE was 11.34 mm versus
pose-command 7.91 mm and command-history 9.37 mm. Unused neural yaw RMSE was
0.793 degrees versus command-history 0.688 degrees. These forecasts did not
select this reactive run's commands.

Free space was 3.79 GiB. Retire this completed success's depth under the
prospective policy after its completed analysis; retain all non-depth evidence,
the preceding full five-treatment comparison and the full new-layout JEPA run.
Because the JEPA run used no local turn-memory selections, the pending local
ablation cannot establish a benefit of that mechanism on this particular path.
Depth retirement reclaimed 1.43 GiB, leaving 5.22 GiB for assignment 5.

## Assignment 5 completed: JEPA without local turn memory, layout 0

Native session 4825 and evaluator session 76917 exited zero. Verified round
trip in **262.08 simulated seconds**, zero contacts and pipeline faults.
Tracking covered 2619 pairs. Goal/home frames were 1805/2617; maximum physical
dwell distances were 3.98/11.68 mm. Both quiet-arrival checks passed. All nine
return edges reversed outbound edges (11 outbound); no invalid transitions.
No local turn-memory selection was exercised, as required. 605 of 648 plans
were on time (93.36%), 43 late. Live wall time before archival was 342.61 seconds.

On 599 overlapping matched 700-ms windows, neural XY RMSE was 9.43 mm versus
pose-command 6.74 mm and command-history 7.77 mm. Neural yaw RMSE was 0.866
degrees versus command-history 0.737 degrees. This completes the first layout
of the fixed three-layout replication batch, with no treatment changes.

The output volume had 3.37 GiB free. This completed success's depth is eligible
under the prospective per-run policy: all analyses are complete, no raw replay
or training input is pending, and the preceding full comparison plus this
layout's full JEPA reference remain. Preserve every non-depth artifact.
Completed retirement reclaimed 1.52 GiB, leaving 4.88 GiB. Assignment 6
(command-history, replication layout 1) launched in session 30287. The fixed
next order on that layout is reactive, no-local-memory JEPA, JEPA, supervised.

## Assignment 6 completed: command-history failure, layout 1

Native owner PID 4168907 / session 30287 exited 1 with a tracking fault;
evaluation session 41607 exited zero and confirmed **no goal arrival, no round
trip, zero contacts**. The saved failure records 1926 acquired camera pairs and
9629 requested steps. Last planning frame was 1920; the tracking fault is
`measured visual pose unavailable`. No simulated round-trip duration exists.
439 of 480 plans were on time (91.46%). Preserve this full failed recording.

The post-hoc `failure_decision_triage_v1.json` summarizes the 256 saved plans
from frame 900 onward: 136 frontier-route and 120 weak-visual-support recovery
decisions; 142 left turns, 109 right turns, four holds and one right arc.
224/256 were on time. No local turn-memory selection occurred anywhere in the
run. This indicates a repeated turn/recovery loop before tracking failed; it
does not yet identify a causal controller or tracker defect. No controller or
model change is made during the fixed batch. Reactive feedback is next.

With the first layout's five-arm analysis complete, release its assignment-1
success depth pin to allow continuation. Keep the original full five-arm
comparison as the successful reference and this new failure in full. The
first-layout JEPA outcome, all physical/prediction/decision records and all
other non-depth files remain; no raw replay or training input is pending.

## Assignment 7 completed: reactive feedback, layout 1

Native session 56965 (owner PID 4170122) and evaluator session 72301 exited
zero. Verified round trip in **222.54 simulated seconds**, zero contacts and
pipeline faults. Goal/home frames were 1153/2219; maximum physical dwell
distances were 12.21/8.84 mm. Both quiet-arrival checks passed. Tracking covered
2222 pairs. All eight return edges reversed outbound edges (10 outbound),
with no invalid graph transitions. The reactive-selector verification passed.
No local turn-memory selections occurred. 470 of 548 plans were on time
(85.77%), 78 late. Live wall time before archival was 290.36 seconds.

On 455 overlapping matched 700-ms windows, unused neural XY RMSE was 11.18 mm
versus pose-command 7.61 mm and command-history 8.90 mm. Unused neural yaw
RMSE was 0.910 degrees versus command-history 0.804 degrees. These forecasts
did not select reactive commands. The reactive package succeeded where the
command-history run failed, on differing trajectories; causality is unresolved.

Keep assignments 6 and 7 in full as the current failed/successful same-layout
diagnostic pair. Free space was 3.38 GiB. A storage review replaces the older
comparison's full reactive reference with this new one; this is an explicit
exception to retaining the entire predecessor depth population through batch
review. All predecessor results/non-depth records and its other four full
recordings remain. No model/controller/order or outcome is changed.

## Assignment 8 completed: JEPA without local turn memory, layout 1

Native session 51987 and evaluator session 96047 exited zero. Verified round
trip in **239.70 simulated seconds**, zero contacts and pipeline faults.
Goal/home frames were 1731/2392; maximum physical dwell distances were
21.17/13.17 mm, with both quiet-arrival checks passed. Tracking covered 2394
pairs. All eight return edges reversed outbound edges (10 outbound), with no
invalid transitions. No local turn-memory selections occurred, as required.
539 of 592 plans were on time (91.05%), 53 late; live wall time was 312.82 s.

On 528 overlapping matched 700-ms windows, neural XY RMSE was 9.51 mm versus
pose-command 6.92 mm and command-history 8.06 mm; neural yaw RMSE was 0.817
degrees versus command-history 0.694 degrees. Standard JEPA remains next.
Keep this full recording for the current same-layout learned/memory comparison.
Free space was 2.78 GiB. Replace the predecessor no-local-memory success depth
pin with this new full reference; all its results/non-depth records remain.

## Assignment 9 completed: JEPA, layout 1

Native session 17239 and evaluator session 79259 exited zero. Verified round
trip in **366.12 simulated seconds**, zero contacts and pipeline faults.
Goal/home frames were 2624/3657; maximum physical dwell distances were
16.91/24.90 mm, with both quiet-arrival checks passed. Tracking covered 3659
pairs. All eight return edges reversed outbound edges (10 outbound), with no
invalid transitions. 788 of 907 plans were on time (86.88%), 119 late; live
wall time before archival was 479.70 seconds.

No local turn-memory selection occurred. Thus the large timing difference
versus the successful local-memory ablation does not demonstrate a cost or
benefit from exercised memory decisions. Asynchronous timing and trajectories
differ; no mechanism-level causal effect is established. Standard JEPA now has
three prospective-layout successes including the preceding comparison.

On 772 overlapping matched 700-ms windows, neural XY RMSE was 9.31 mm versus
pose-command 6.91 mm and command-history 8.39 mm; neural yaw RMSE was 0.944
degrees versus command-history 0.784 degrees. The learned prediction advantage
remains unproven. Supervised prediction completes this layout next.

Free space was 1.52 GiB. Keep the full current layout's failed command-history
run and successful reactive, ablated-JEPA and JEPA runs. End the predecessor
comparison's completed JEPA and supervised success-depth pins: no raw replay
or training input needs either, and all their results/analyses/non-depth files
remain. Its command-history success remains full as an earlier successful
same-arm counterpart for the current failure. No controller/model change occurs.

## Assignment 10 completed: supervised, layout 1

Native session 24390 and evaluator session 45008 exited zero. Verified round
trip in **230.52 simulated seconds**, zero contacts and pipeline faults.
Goal/home frames were 1372/2301; maximum physical dwell distances were
11.50/13.38 mm, with both quiet-arrival checks passed. Tracking covered 2303
pairs. All eight return edges reversed outbound edges (10 outbound), with no
invalid transitions. 532 of 567 plans were on time (93.83%), 35 late; live
wall time before archival was 301.99 seconds. No local turn-memory selections
occurred. All four learned/reactive arms succeeded on this layout; only
command-history failed. These single executions do not establish superiority,
isolated prediction benefit or a memory effect.

On 527 overlapping matched 700-ms windows, neural XY RMSE was 10.87 mm versus
pose-command 7.61 mm and command-history 10.00 mm; neural yaw RMSE was 0.799
degrees versus command-history 0.719 degrees.

Free space was 2.51 GiB. Retain current assignments 6 (failure), 7 (reactive),
9 (JEPA), and 10 (supervised) in full for diagnosis. The local-memory paired
analysis is complete, with zero exercised memory selections in either run;
retire assignment 8 success depth, preserving every non-depth record. Also
release the predecessor command-history success depth pin: its completed
analysis is retained and no further raw replay or training input needs it.
These reviews revise only those two depth pins; every failure remains full.
Retirement reclaimed 2.82 GiB, leaving 5.33 GiB. Assignment 11 (JEPA without
local turn memory, replication layout 2) launched in session 14251. The final
layout order remains no-local-memory JEPA, JEPA, supervised, command-history,
reactive. Five assignments remain including that active run.

## Assignment 11 completed: JEPA without local turn memory, layout 2

Native session 14251 (owner PID 4174113) and evaluator session 81438 exited
zero. Verified round trip in **225.68 simulated seconds**, zero contacts and
pipeline faults. Goal/home frames were 1328/2252; maximum physical dwell
distances were 13.06/22.55 mm. Both quiet-arrival checks passed. Tracking
covered 2254 pairs. All seven return edges reversed outbound edges, with no
invalid transitions. No local turn-memory selections occurred, as required.
515 of 549 plans were on time (93.81%), 34 late; live wall time was 299.17 s.

On 504 overlapping matched 700-ms windows, neural XY RMSE was 10.01 mm versus
pose-command 7.05 mm and command-history 7.93 mm; neural yaw RMSE was 0.831
degrees versus command-history 0.683 degrees. Keep this full recording for
the pending standard-JEPA comparison on the final layout.

Free space was 3.73 GiB. End the completed layout-1 supervised success depth
pin: its full five-arm study, physical/tracking/backtracking and forecast
analyses are complete, with no pending raw replay or training input. Its
results, poses, commands and decisions remain. Keep layout-1 command-history
failure and JEPA/reactive diagnostic controls in full, plus this new ablation.

## Assignment 12 completed: JEPA, layout 2

Native session 97989 and evaluator session 23120 exited zero. Verified round
trip in **207.72 simulated seconds**, zero contacts and pipeline faults.
Goal/home frames were 1160/2072; maximum physical dwell distances were
14.89/6.35 mm, with both quiet-arrival checks passed. Tracking covered 2074
pairs. All seven return edges reversed outbound edges, with zero invalid
transitions. 429 of 504 plans were on time (85.12%), 75 late; live wall time
before archival was 272.48 seconds. Standard JEPA is now 4/4 on prospective
layouts with this fixed controller, including the preceding comparison.

Local turn memory was selected on 25 left-turn plans, frames 1208–1304, in the
return phase. Nineteen were on time. Joining saved dispatch provenance to
those plan observation timestamps identifies 379 intervals from 19 plans:
364 applied positive-yaw commands and 15 applied zero commands. The saved
`local_route_turn_memory_execution_v1.json` records this evidence. Memory
choices actually reached execution; the 207.72-s versus 225.68-s ablation
comparison still does not isolate causal benefit from differing trajectories
and deadlines. Preserve both final-layout memory-arm recordings in full.

On 423 overlapping matched 700-ms windows, neural XY RMSE was 10.19 mm versus
pose-command 6.61 mm and command-history 8.65 mm; neural yaw RMSE was 0.860
degrees versus command-history 0.751 degrees. The remaining supervised,
command-history and reactive treatments complete the fixed batch.

Free space was 3.57 GiB. End the completed preceding-layout reactive success
depth pin: its selector, physical, tracking, backtracking and forecast analyses
are complete and no raw replay is pending. Preserve all its non-depth records,
the same-layout command-history failure and JEPA diagnostic control in full,
and the current exercised memory/ablation pair. This replaces the earlier
three-recording diagnostic depth population with the predictive pair plus the
current memory pair; every failure remains full.

## Assignment 13 completed: supervised, layout 2

Native session 83841 and evaluator session 64542 exited zero. Verified round
trip in **237.54 simulated seconds**, zero contacts and pipeline faults.
Goal/home frames were 1420/2367; maximum physical dwell distances were
16.91/14.54 mm, with both quiet-arrival checks passed. Tracking covered 2369
pairs. All seven return edges reversed outbound edges, with zero invalid
transitions. 477 of 582 plans were on time (81.96%), 105 late; live wall time
before archival was 310.14 seconds. Supervised is now 4/4 on prospective
layouts with this controller, including the preceding comparison.

Local turn memory selected 25 left-turn plans, frames 1664–1760. Twenty were
on time; dispatch provenance joins identify 398 intervals from those plans,
368 applying positive-yaw commands and 30 zero commands. The saved execution
readout records the counts. This is exercised memory, not a demonstrated
causal navigation advantage. Different models/trajectories/timing preclude
treating the JEPA-versus-supervised difference as an isolated memory effect.

On 471 overlapping matched 700-ms windows, neural XY RMSE was 10.02 mm versus
pose-command 6.88 mm and command-history 9.63 mm; neural yaw RMSE was 0.955
degrees versus command-history 0.822 degrees.

Free space was 3.15 GiB. Per the prospective per-run policy, retire this
fully analysed success's depth; no raw replay or training input is pending.
Keep all its outcome, model, pose, decision, forecast and execution records.
The full current JEPA/ablation pair retains the exercised-memory comparison,
and the earlier full failure/JEPA diagnostic pair remains intact.

## Assignment 14 completed: command-history, layout 2

Native session 57793 and evaluator session 61043 exited zero. Verified round
trip in **260.90 simulated seconds**, zero contacts and pipeline faults.
Goal/home frames were 1432/2604; maximum physical dwell distances were
16.94/16.82 mm, with both quiet-arrival checks passed. Tracking covered 2606
pairs. All seven return edges reversed outbound edges, with zero invalid
transitions. 545 of 635 plans were on time (85.83%), 90 late; live wall time
before archival was 341.46 seconds. Command-history is 3/4 across prospective
layouts, including its preserved earlier tracking failure.

Local turn memory selected 18 left-turn plans, frames 1484–1552; 17 were on
time. All 340 joined dispatch intervals from those 17 plans applied positive
yaw commands. Its saved execution readout preserves the evidence; differing
layouts and trajectories prevent interpreting the failure/success contrast as
a causal memory effect.

On 542 overlapping matched 700-ms windows, applied command-history XY RMSE
was 8.22 mm versus unused neural 9.64 mm and pose-command 6.74 mm; applied
yaw RMSE was 0.678 degrees versus unused neural 0.807 degrees.

Free space was 2.65 GiB. Retire this fully analysed success's depth under the
prospective policy, preserving all non-depth outcomes, decisions and execution
records. No raw replay or training input is pending for it. The full failed
command-history run, its JEPA diagnostic control, and current JEPA/ablation
memory pair remain. Reactive feedback is the final fixed assignment.

## Assignment 15 completed: reactive feedback, layout 2

Native session 38999 and evaluator session 78082 exited zero. Verified round
trip in **191.86 simulated seconds**, zero contacts and pipeline faults.
Goal/home frames were 1277/1911; maximum physical dwell distances were
16.79/7.31 mm, with both quiet-arrival checks passed. Tracking covered 1913
pairs. All seven return edges reversed outbound edges, with zero invalid
transitions. Reactive selector verification passed. 455 of 470 plans were
on time (96.81%), 15 late; live wall time before archival was 249.81 seconds.
No local turn-memory selections occurred.

On 444 overlapping matched 700-ms windows, unused neural XY RMSE was 11.75 mm
versus pose-command 7.57 mm and command-history 8.62 mm; neural yaw RMSE was
0.864 degrees versus command-history 0.743 degrees. These neural forecasts
did not select the reactive controller's commands.

The frozen batch is complete. No additional attempts were added. The remaining
scientific questions concern the preserved tracking failure, wider reliability,
the contribution of learned prediction and persistent spatial memory, and
realistic sensing/timing. No JEPA superiority or hardware readiness is established.
