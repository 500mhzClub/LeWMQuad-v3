# Executed-horizon final-goal native probe V1

Run the fixed corrected seed-2026091001 full JEPA/direct pair on reused
family_episode_039, unchanged seed-2026091501 order. Require full correction
admission/readout, the completed exact-mission-target native pair/readout and
two exact final-goal scoring prefix replays per model. At least one command
must change; observations, map, forecasts and all clearance checks must match.

Only when the existing observed final-goal connector selects the exact
instructed target, score its distance-and-bearing potential at the executed
100-ms predicted pose. Keep the original 800-ms cumulative contact penalty
with coefficient 1.2, all first articulated-surface checks and all eight
0.45-m nominal segment checks. Preserve the original scored candidates and
action. Intermediate targets and view acquisition keep their original score.
No new distance threshold, command sequence, model fitting or calibration.

Keep both robot-visible cameras/calibrations, observer/map and explicit
observed-floor contact semantics. Keep all unknown/non-floor/non-foot checks,
six action plans, one-step commitment, bounded waits, 240 navigation ticks,
4-cm controller arrival dwell and native 6-cm one-second quiet/contact gate.
Preserve gait, slew, gains, setup, physical stops and ten-command terminal drain.
Native pose, scene labels and unexecuted outcomes never enter selection.
Use original map indices; the concurrent source-only batching optimization
is a separate question and is not part of this scoring experiment.

Freeze source/input identities before exclusive output
`go2_executed_horizon_final_goal_probe_v1_attempt_001`. Two one-thread CPU
scenes, current hardware check and prior scaling evidence; require 32 GiB
available RAM and an 8-GiB output allowance above the 40-GiB reserve. Monitor
resources and complete-loop time. Require unchanged first-four paired packets
and 900 native samples. Reconstruct all raw sensors, replay every command with
a fresh assigned model and retain every existing native audit. Reverify all
bindings; preserve failures without retry/resume.

Count goals only under the complete unchanged gate. Earlier holds or drift
could still defeat the new score. Zero independent-maze or hardware claims
are authorized by this integration test; exploration, physical backtracking,
matched baselines and realistic timing remain required.
