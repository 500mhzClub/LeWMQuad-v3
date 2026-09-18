# Exact final mission-target prefix V1

Authenticate the completed observed-floor-contact native pair and readout.
Keep fixed corrected seed-2026091001 JEPA/direct models, all sensing/mapping,
contact semantics, learned forecasts, action plans, scoring formula, nominal
first/eight-step checks, waits, arrival radii/dwell and mission budgets.

Correct the final waypoint only. When the existing route reaches the instructed
goal cell and the chosen waypoint is its last cell centre, form the exact mission
goal in the existing gravity map using map_from_initial. Use that endpoint only
if the closed current-to-goal connector's entire cell supercover is measured
floor and its continuous nominal distance to all observed occupied cells passes
the original 0.45-m check. Otherwise retain the original waypoint and report why.
Never retarget a frontier or bypass a required intermediate waypoint. Preserve
the unchanged model prediction inputs; only the instructed terminal target and
its existing utility scoring change.

The five connector tests cover the actual goal-centre mismatch, intermediate/
frontier retention, unseen endpoint/corner cells, known obstacles and malformed
input. The scope test checks identical eight-step scoring/constraints and
inherited execution. Run two fresh exact replays per model, requiring unchanged
causal observations, map receipts, model forecasts, surface and nominal checks.
Record all selected exact-goal witnesses. Stop at the first requested-command
or terminal difference, including that observation; otherwise use the recorded
tape end. Execute no new command and infer no native counterfactual outcome.

Exclusive go2_exact_mission_target_prefix_v1_attempt_001; freeze/reverify full
source and input bindings. One CPU process/thread, current hardware check,
8 GiB available RAM and 256 MiB output allowance above the 40-GiB reserve.
This is integration evidence, not verified arrival, independent-maze navigation,
backtracking, real-time operation or hardware qualification.
