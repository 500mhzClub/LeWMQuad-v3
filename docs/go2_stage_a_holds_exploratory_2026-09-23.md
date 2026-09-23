# Stage A holds — exploratory read-only analysis

The four assigned runs are complete and unchanged. Each has 1,198 selected plans and exhausted its 480.32-s budget without reaching goal/home; none recorded a disallowed contact or pipeline fault. This analysis reads their saved planning and 20-ms dispatch traces. It runs no models or physics and computes no audit regret.

| Layout / matched head | Hold plans | No eligible movement | Eligible movement lost score/tie | Explicit override | Insufficient evidence |
|---|---:|---:|---:|---:|---:|
| 00 / old data | 987 | 898 | 88 | 1 | 0 |
| 00 / maze data | 979 | 875 | 19 | 84 | 1 |
| 02 / maze data | 1,140 | 1,138 | 1 | 1 | 0 |
| 02 / old data | 957 | 614 | 97 | 244 | 2 |

Categories are mutually exclusive. Explicit logged overrides take precedence; then no eligible movement; then a finite recorded hold score at least as high as every eligible moving candidate; otherwise insufficient evidence. There were no exact score ties in the classified score-loss cases. The inspected selector takes the first maximum in canonical candidate order, with hold first.

Eligibility here means the logged scan/view action subset, recorded memory-path gate (including reserve recovery where recorded), and any recorded stopping-projection gate. It is **not physical acceptability**. No new forecasts or clearance tests were reconstructed. Missing evidence was not converted into an inferred override.

Overrides were stopping projection in 1/84/0/12 cases respectively, and a blocked latched recovery turn in 0/0/1/232 cases. Three unexplained holds remain explicitly unclassified.

| Layout / head | Holds with observation-only action restriction | Holds with motion-dependent clearance exclusion | Holds with stopping-projection exclusion |
|---|---:|---:|---:|
| 00 / old data | 1 | 899 | 2 |
| 00 / maze data | 837 | 959 | 0 |
| 02 / maze data | 1,131 | 1,139 | 1 |
| 02 / old data | 293 | 859 | 12 |

These flags overlap and do not sum to hold totals. Observation-only coverage is limited to the identifiable scan/view action-space restriction; other observation validity rules are not fully reconstructible from these logs. Motion-dependent flags use logged predicted path and stopping checks against observed geometry. Candidate-level exclusion counts are retained in JSON. Stopping-projection flags in this table count candidates surviving the preceding recorded scan subset: a recovery override can select a translation outside that subset and then be stopped. Thus 84 stopping overrides and zero such candidate flags in 00/maze are not contradictory counts.

Zero requested commands at 20-ms service calls are separate from planning decisions:

| Layout / head | Zero-request calls | Mission settling/terminal | No on-time plan | Current nominal obstacle test passed | Latched command-window veto | Prefix not executed | Stale |
|---|---:|---:|---:|---:|---:|---:|---:|
| 00 / old data | 19,796 | 16 | 60 | 19,720 | 0 | 0 | 0 |
| 00 / maze data | 19,636 | 16 | 60 | 19,560 | 0 | 0 | 0 |
| 02 / maze data | 22,886 | 16 | 60 | 22,760 | 29 | 20 | 1 |
| 02 / old data | 19,216 | 16 | 60 | 19,100 | 39 | 0 | 1 |

“Obstacle test passed” on a zero request does not mean that a moving candidate was available; the planner may already have requested hold. The service counts are not additional hold plans. Two planning rows per run without a selection are excluded from the 1,198-plan denominator.

The trajectories and observations differ between runs. These counts cannot establish an isolated readout effect, prove that rejected movement would have been safe, or quantify decision headroom. They show that logged clearance restrictions dominate many holds, while score choice and explicit recovery/dispatch behavior also matter.

[Per-hold categories, input hashes and full dispatch reasons](go2_stage_a_holds_exploratory_2026-09-23.json) were produced by [the bounded reader](../scripts/analyse_go2_stage_a_holds_readonly_development.py). Selector semantics were inspected in `memory_forecast_clearance_development.py`, `clearance_turn_recovery_development.py`, `planned_stopping_projection_development.py`, `paced_multirate_controller_development.py` and the hold-relative recovery modules. The [Stage A result](go2_maze_view_readout_navigation_2026-09-23.md) remains the outcome record.
