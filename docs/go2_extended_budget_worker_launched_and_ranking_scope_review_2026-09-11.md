# Extended-budget worker launched; predictive ranking intervention scope

This follows `go2_tiled_controller_verified_profile_v2_active_2026-09-11.md`.
The extended-budget launcher's initial input admission has now finished. It
created its immutable launch and spawned its original native worker. The worker
performs another `verify_inputs(launch)` before loading the assigned model and
calling `pipeline.collect`; an active worker does not establish that simulation
has begun. At this observation the episode directory was absent, the worker log
was empty and no worker-terminal record existed.

- Native root: `go2_no_rgb_direct_extended_budget_maze02_pilot_v1_attempt_001`.
- Launch SHA-256: `7380fbd8c83306dfe544fd9760e13a9eb8a0b70cb93ade06e1972096044c8113`;
  all 2,141 launch source bindings were rechecked without repeating raw-input admission.
- Original launcher: 2843773 / creation 1789128335.77, now parent of worker
  2867880 / creation 1789139673.31 and resource tracker 2867879.
- Case: `no_rgb_direct_extended_budget_anchored_maze_02`, original model
  `56799c99e2bb1fd5e5a591009b931c5b2e04741d3a3744e2346ce9896a2160dd`.
- The launch retains 4,000 navigation ticks, the original controller, fresh
  memory, paused physics during computation and no queued controller changes.
- Observation receipt:
  `docs/go2_extended_budget_native_worker_launch_observation_2026-09-11.json`,
  SHA-256 `ed8d0c8a6a98f8de2c9335e0847c8adbf9642adbd216d18a95b3954cd80326ed`.

The full-history profile V2 remains live as PID 2866966 / creation 1789139321.48,
session 29134, and has printed frame 450. Its automatic completion watcher
2867206 / creation 1789139457.92, session 80413, remains live. No profile result
or failure was present. Do not start another full CPU replay. The subsequent
native queue remains sustained turn, contact/flow and chained anchors.

## Ranking-only comparison requires more than changing one final score

The existing four-arm independent definition remains unchanged and unexecuted.
The prior comparison-scope review correctly identifies its reactive arm as a
whole-method comparison. This additional source inspection identifies the
specific intervention points that a future ranking-only comparison must cover:

1. `RoundTripMissionSelector.choose` in
   `lewm/observed_round_trip_controller_development.py` calls final-goal execution
   scoring and then `correct_final_goal_score`. The latter uses residual-corrected
   first-point predicted translation, predicted first-point yaw and predicted
   final-horizon contact when ranking feasible actions.
2. `ViewReentrySelector.choose` in
   `lewm/view_reentry_round_trip_controller_development.py` first applies the
   translation reentry rule, then calls `score_waypoint_execution`. That function
   in `lewm/executed_waypoint_score_development.py` applies only to its declared
   intermediate-waypoint scope, excluding final-goal and nominal-reentry cases.
   It ranks with a 100 ms predicted pose and an 800 ms contact score while keeping
   the longer predicted-path constraints.
3. `ResidualFirstIntervalSelector.choose` and
   `ResidualAnchoredContinuationSelector.choose` subsequently run correction
   policies. The anchored continuation function first invokes hold feasibility
   reconsideration and may then apply its own anchored-hold reconsideration.
   The anchored-hold code checks the original score contract, causal predicted
   scoring positions and utilities before selecting among eligible actions.

Consequently, replacing a final returned action or changing only intermediate
waypoint scoring would not implement a complete, consistently labelled ranking
intervention. It could miss final-goal decisions or bypass the recovery sequence.
Reusing the original predicted-utility contract for a nonpredictive score would
also misdescribe the resulting receipts.

Before any ranking-only treatment is executed, a prospective successor must
define its observed-data ranking rule, preserve the shared forecast/feasibility
and mission machinery, identify each scoring branch and its receipt contract,
and exercise final-goal, intermediate-waypoint, reentry, hold-recovery and empty
feasible-set cases. A ranking-only treatment that retains forecast-based vetoes
must be described as conditional on those vetoes, not as a fully nonpredictive
controller. The existing whole-method reactive comparison remains a separate
question. Actual matched closed-loop execution is required to establish benefit;
source inspection or rescoring recorded decisions cannot establish it.

No new treatment, model assignment, population size or comparison family was
selected by this review. The pending eight-diagnostic review must consider the
actual native outcomes before freezing the independent-study definition. There
is still no verified round trip, independent-maze reliability, predictive-planning
advantage, real-time qualification or hardware result.
