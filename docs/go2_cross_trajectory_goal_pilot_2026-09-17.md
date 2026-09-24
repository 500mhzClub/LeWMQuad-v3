# Cross-trajectory goal metric: completed local control comparison

The matched-budget change to goal-pair supervision corrects the opposing-turn
distance defect but does not establish reliable navigation: **0/4 final
arrivals, one transient goal visit, one contact, one false arrival latch**.
The encoder, action-conditioned predictor, six candidate actions, 500-ms
horizon and visual arrival rule remain unchanged. All failures are retained.

| Task | Final position error (cm) | Final heading error (degrees) | Contact | False arrival latch |
|---|---:|---:|---|---|
| fresh_00, left | 3.040 | 0.640 | No | No |
| fresh_01, right | 15.461 | 39.517 | Yes | No |
| fresh_02, left | 8.302 | 14.934 | No | No |
| fresh_03, right | 3.153 | 1.961 | No | Yes |

Success still requires final position within 3 cm, heading within 5 degrees,
completed budget and no contact. Task 00 visits the target and drifts just
outside the position threshold after its hold latch. Task 01 selects forward
four times, then contacts during a right arc. Task 02 approaches and then
chooses hold while outside the goal. Task 03 now turns right but falsely
latches arrival and finishes outside position tolerance. No threshold was
changed after observing these results.

| Controller on these same four tasks | Final arrivals | Transient visits | Contacts | False latches |
|---|---:|---:|---:|---:|
| Predictor + original within-recording metric | 0/4 | 1/4 | 0 | 0 |
| Direct visual feedback | 0/4 | 0/4 | 1 | 2 |
| Predictor + signed visual pose cost | 0/4 | 1/4 | 1 | 1 |
| Predictor + mixed within/cross-recording metric | 0/4 | 1/4 | 1 | 1 |

The first three rows reuse retained results; only the final row consists of
four new trials. All layouts are now exposed development cases. They share
the training obstacle family, appearance and dynamics, and are not full mazes
or independent confirmation of the intervention. The experiment changes
supervised goal grounding, not the JEPA representation objective.

Execution: cases 0/1 had completed before this continuation; their terminal
results were verified rather than rerun. Cases 2/3 completed in parallel
(sessions 61147 and 70580, both exit 0). Reader session 26014 exited 0;
initial eleven RGB frames match the reference tasks exactly, and task 01's
first forecast costs reproduce the fixed evaluation. RGB, physics, actions
and results are retained; unused depth was not recorded.

Training coverage diagnosis (`go2_dense_start_action_coverage_2026-09-17.json`)
finds 594/3,518 examples with completely quiet past command history: 524
continue holding and 70 contain some future motion. For each of the five
moving primitives, only two match the full deployment 500-ms applied tape.
They cover the two left-opening training geometries, both with appearance
seed 2026090940; there are no exact right-opening start examples. The other
60 motion examples use partial starts or short pulses. This is a concrete
coverage imbalance, not proof that it caused the right-task failure. In
particular, the current predictor does select right arc on task 03.

The next predictor intervention should test this coverage hypothesis with
training-layout data and matched action-conditioned/action-blind budgets,
keeping the encoder and goal metric fixed. Assess forecast action ranking
and prospective control separately. Additional goal-head fitting alone is
not justified by these results. Contact avoidance and longer-horizon route
selection also remain missing from this local terminal-goal-cost experiment;
better pixel-feature prediction alone cannot establish those capabilities.

Authoritative results: `go2_cross_trajectory_goal_pilot_result_2026-09-17.json`.
Fixed ranking evaluation: `go2_cross_trajectory_goal_metric_evaluation_2026-09-17.json`.
No complete-maze, real-time or hardware claim is supported. The full goal
remains active.
