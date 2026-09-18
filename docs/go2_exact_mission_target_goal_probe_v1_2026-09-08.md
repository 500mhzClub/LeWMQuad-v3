# Exact instructed final-target native probe V1

Run the fixed corrected seed-2026091001 JEPA/direct pair on reused
family_episode_039, unchanged seed-2026091501 order. Require full correction
admission/readout, the completed observed-floor-contact native pair/readout
and two exact final-target prefix replays per model, with at least one changed
command and unchanged observations, map, forecasts, surface and nominal checks.

Change only the final waypoint: when the existing route reaches the goal cell,
the selected waypoint is its last cell centre, and the closed current-to-exact-
goal connector has fully observed floor and passes the original continuous
0.45-m nominal check, score actions toward the instructed goal transformed into
the observed map. Otherwise keep the original intermediate/frontier waypoint.
Record the exact decision, connector cells, missing observations and clearance.

Keep the same two robot-visible cameras/calibrations, observer/map, explicit
observed-floor foot-contact semantics, all unknown/non-floor/non-foot obstacle
checks, models/inputs/corrections, six action sequences, learned scoring formula,
first-step/eight-step nominal constraints, bounded waits, 240 navigation ticks,
4-cm controller arrival dwell and native 6-cm one-second quiet/contact goal gate.
Preserve gait, slew, gains, setup, physical stops and ten-command terminal drain.
No native pose, scene labels or unexecuted outcomes enter command selection.

Freeze source and input identities before exclusive output
go2_exact_mission_target_goal_probe_v1_attempt_001. Two one-thread CPU scenes,
current hardware check and prior scaling evidence, 32 GiB available RAM and
8 GiB output allowance above the 40-GiB reserve. Monitor resources and full-loop
time. Require unchanged first-four paired 45-degree packets and 900 native
samples. Replay every command decision with a fresh assigned model and apply
all existing raw/native audits. Reverify every binding after completion and
preserve failures without retry/resume.

Count goals only under the full original gate. A predecessor quiet stop inside
the evaluator radius did not satisfy its controller arrival state. No tolerance
is widened here. Independent mazes, backtracking, matched reactive/nonpredictive/
planning/memory comparisons, realistic timing and hardware remain unqualified.
