# Corrected-runtime fresh-maze comparison: complete result

All fifteen prospectively assigned missions are evaluated: **ten physically
verified round trips, five failures and zero disallowed contacts**. Each of the
five controllers completed two of three new development mazes. Reliable transfer
and a JEPA advantage remain unproven.

The fixed plan is
[`go2_publication_repaired_navigation_replication_plan_2026-09-16.json`](go2_publication_repaired_navigation_replication_plan_2026-09-16.json).
The [execution journal](go2_publication_repaired_navigation_replication_2026-09-16.md)
records every outcome, diagnosis and depth retirement. This cohort uses the three
previously unexecuted layouts from the earlier four-layout inventory, excluding
the layout exposed during publication-deadlock diagnosis. Its results are not
pooled with that interrupted predecessor or the exposed repair verification.

## Complete navigation outcomes

Successful times are simulated seconds, including goal and return quiet dwells.

| Controller | Maze 1 | Maze 2 | Maze 3 | Round trips |
| --- | --- | --- | --- | ---: |
| JEPA | No goal; budget 480.94 s | 184.48 s | 134.84 s | 2/3 |
| Supervised rollout | Goal only; budget 480.88 s | 146.02 s | 164.24 s | 2/3 |
| Pose-command | Tracking failure | 160.88 s | 135.26 s | 2/3 |
| Instantaneous ranking | 366.58 s | 191.70 s | Tracking failure | 2/3 |
| Reactive feedback | Goal only; budget 480.80 s | 177.76 s | 184.78 s | 2/3 |

Twelve missions reached the goal; ten also returned home. Every successful
arrival passed the independent physical-radius and one-second quiet-motion
checks. All ten successful returns physically reversed previously traversed
outbound corridor edges, with no invalid graph transitions. The reactive maze-1
failure also backtracked partway before stalling. This demonstrates route reuse
in the implemented controller, without isolating a causal memory advantage.

JEPA and supervised completed the same two mazes. Supervised was 38.46 seconds
faster on maze 2; JEPA was 29.40 seconds faster on maze 3. Pose-command was
23.60 seconds faster than JEPA on maze 2 and 0.42 seconds slower on maze 3.
There is no consistent JEPA advantage in completion or speed. Differences come
from single executions with different exploration, recovery and terminal
approach trajectories; they are not repeated independent speed estimates.

Instantaneous ranking completed maze 1 where the others failed, but lost
tracking near the maze-3 goal. It retains predictive clearance guards, so its
two successes do not demonstrate navigation without prediction. Reactive
feedback removes forecast-based selection and predictive feasibility rules
while retaining measured dispatch guards; it is a broader controller-package
comparison. Neural forecasts still run in all arms for workload control, but
total computation is not identical. All arms retain persistent observed memory.

## What failed

- **Supervised, maze 1:** reached the goal, then spent most of the remaining
  budget in view recovery. Frames 2200–4800 contain 651 recovery plans, including
  647 holds; 647 plans were on time. Both turns passed nominal-footprint clearance
  in the held plans but failed additional reserve/recovery checks. At least one
  arc passed the recorded memory-clearance and planned-stopping checks in every
  held plan. An unexecuted arc's safety or success is not established.
- **Pose-command, maze 1:** tracking failed after 619 camera frames, without an
  arrival, despite all 153 plans being on time. Startup deferred after three
  views and visual recovery did not prevent loss of the measured pose.
- **Reactive, maze 1:** reached the goal and backtracked eight corridor edges,
  then held for the final 120 seconds because stored clearance rejected all
  actions. Of 301 final plans, 290 were on time. Stored clearance underestimated
  physical wall clearance by median 13.17 mm, but the physical base centre was
  itself within the configured 0.45-m nominal margin in 283/301 samples. This is
  not merely a false map obstruction; a safe escape remains unproven.
- **JEPA, maze 1:** no arrival; all 701 plans in frames 2000–4800 held, and 680
  were on time. All six predicted paths failed nominal clearance. Current
  physical wall clearance was median 0.45523 m and below 0.45 m in 72/701
  samples. Current native clearance is not the predicted future-path clearance.
  The entry cause and any safe alternative are not yet isolated.
- **Instantaneous, maze 3:** tracking failed near the goal before an arrival,
  after 1379 camera pairs. All 119 final goal-area plans were on time; 115
  selected pure turns and 26 requested visual recovery. The final three support
  records had zero primary-camera selected features. Registered-position error
  before failure remained below 7.49 mm. The underlying tracking rejection still
  needs diagnosis; feature counts are not calibrated confidence.

All five full failure recordings remain available. The publication deadlock did
not recur. Neither that fix nor startup-survey deferral resolved the remaining
navigation and tracking failures. Zero contacts does not imply reliable mission
completion or certify the nominal circular footprint as articulated-body safety.

## Timing and prediction

Of 8605 plans, 7900 were on time (**91.8%**). The largest delay concentration was
the successful instantaneous maze-1 run, with 572/905 on time. Its outbound
route-computation slowdown is recorded separately. Most failure intervals had
high on-time fractions, so missed deadlines alone do not explain this cohort.
The fixed extra 20-ms publication wait crossed 207 deadlines on recorded
trajectories; that count does not predict outcomes without the wait.

On matched executed 700-ms windows, the fitted pose-command alternative had
lower planar endpoint RMSE than the recorded neural forecast in **all fifteen
recordings**. The command-history yaw alternative also had lower endpoint yaw
RMSE than the neural forecast in all fifteen. Per-run metrics and action-group
breakdowns are preserved in the complete scientific readout. These overlapping
windows are trajectory-conditional and do not evaluate unexecuted candidates.
The neural reference in pose-command and reactive runs is the supervised model,
computed but unused for selection. These are prediction diagnostics, not
counterfactual navigation results or a calibrated reserve guarantee.

## Evidence and scope

Artifacts live under
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/`.
The directory `go2_publication_repaired_replication_readout_v1_attempt_001/`
contains `result.json`, `complete_scientific_readout_v1.json`, and the visually
inspected `complete_outcomes_v1.png` / `.svg`. Reproduce the final readout using
`scripts/read_go2_publication_repaired_navigation_replication_complete_development.py`.

Only three mazes from one family, one neural training seed and one execution
per condition/layout are represented. Six action candidates, the 0.8-second
prediction horizon, shared controller settings and model weights stayed fixed.
The ideal gyro, synthetic 2-mm depth noise and measured-simulation deadlines
remain unqualified for real sensors and real-time operation. No hardware run,
sealed evaluation, broader-environment generalization or deployment claim follows.

Completed, diagnosed success depth was retired under the standing retention
policy; all non-depth outcomes, RGB, poses, commands, timing, forecasts, physics
and comparison evidence remain. Exact original depth replay is unavailable for
those successes. Full failures and current debugging references remain retained.

## Next scientific step

Finish diagnosing the shared clearance/view-recovery conflict before another
large replication. Start with the supervised maze-1 stall, where the saved
records identify arcs passing existing clearance and stopping checks while the
view objective repeatedly selects hold. Inspect the actual selector composition
and test a narrowly defined recovery intervention on this exposed failure,
preserving the model, six candidates and existing nominal/reserve/dispatch limits.
Do not infer its efficacy from saved scores: any proposed change needs a new
closed-loop run and later prospective replication.

The JEPA nominal-clearance stall and the two tracking losses may need different
interventions; do not assume one repair solves all five failures. Candidate-bank
expansion and broader environment types remain deferred. Multi-seed comparisons,
visual-dependence controls, causal memory tests with the current runtime,
realistic sensing/full-loop timing and bounded physical Go2 evidence remain
part of the active overall goal.
