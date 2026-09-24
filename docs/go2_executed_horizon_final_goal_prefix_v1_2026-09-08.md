# Executed-horizon final-goal controller prefix V1

Use the completed exact-mission-target native integration and its authenticated
readout. Keep both assigned corrected models, current six action tapes, public
sensor streams, observation/map behavior and original arrival/budget contracts.

Change one scoring choice: when the existing measured and nominally clear
final connector selects the exact instructed goal, score distance-and-bearing
potential progress at the first 100-ms predicted pose. Retain the original
800-ms cumulative contact score with coefficient 1.2 and all original first
articulated-surface and eight nominal-segment constraints. Preserve previous
800-ms scores and action as evidence. Intermediate-waypoint and view-acquisition
scoring stay identical. No distance threshold is added; no tolerance is widened.

Replay each fixed model twice from fresh state. Compare causal observations,
map receipts, predictions and all surface/nominal checks exactly. Require each
new selection to equal the explicit pure score transformation of the recorded
selection. Stop at the first command or terminal intervention, including only
the causal observation preceding it. Do not consume subsequent counterfactual
observations or infer outcomes. No native commands or training occur here.

Exclusive `go2_executed_horizon_final_goal_prefix_v1_attempt_001`; one CPU
process/numerical thread, 8-GiB RAM and 256-MiB output allowance above the
40-GiB reserve. Freeze and reverify source, readout, native and model inputs.
Launch after the isolated mapping timing benchmark finishes. Prospective
execution is needed to determine whether shorter scoring actually improves
the approach; early holds remain a possible failure. Independent-maze,
backtracking, baseline, timing and hardware requirements remain open.
