# Cadenced old-view tracking experiment

Run one full supervised-rollout mission on exposed short-pulse maze 1 with
the 250-ms obstacle age bound and old-view bank probes every four frames.
Keep every 100-ms camera/gyro update, the original recent/stable references,
all reference acceptance limits, model, map, dispatch/stopping checks, mission
coordinates and 4800-tick budget. Parent Python switch interval remains 5 ms;
use the same maze-1 CPU allocation, software EGL, 2-mm depth noise and ideal
gyro. Keep the original queue failure and this result regardless of outcome.

The preceding 250-ms attempt eliminated stale vetoes but overflowed its
tracking queue near 195 s. Its retained replay reproduced all 1918 recorded
poses exactly. In frames 1700 onward, unsuccessful old-view probes dominated
and median isolated tracker processing was about 94 ms. A candidate probing
the bank every 400 ms (the existing planning cadence) completed all 1952
public packets. Its median late-frame processing was about 52 ms, with maximum
raw position error 5.098 mm and maximum difference from recorded raw positions
3.587 mm. Native truth was used only by a separate saved-pose evaluator.

This changes reference-selection opportunities. It is not numerically
equivalent, does not reset tracking, and does not remove route memory or the
old-view bank. The replay is a speed/accuracy check on one saved trajectory,
not evidence of new navigation success. Native timing includes transport and
contention absent from replay; faster median time does not guarantee deadlines.

Primary measures are physical arrivals, contacts, progress and failure status.
Also inspect tracking service/lag and queue behavior, late plans, actual
accepted observation ages, pose error and measured command continuity. No
heavy concurrent analysis during simulation. One exposed layout and one
execution cannot establish reliability or a JEPA advantage. Real-time and
hardware qualification remain unproven.

Replay evidence: `go2_cadenced_view_revisit_replay_v1_attempt_001`, including
`physical_accuracy_evaluation_v1.json` and the comparison excluding profiled
reference frames, `matched_late_timing_comparison_v1.json`.
Launcher: `scripts.run_go2_cadenced_view_navigation_development`.
Evaluate with `--evaluate` after owner exit and complete persistence.
Root: `go2_cadenced_view_age_250ms_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001`.

Before launch: 16 physical/32 logical CPUs, approximately 0.3% aggregate CPU
utilization, 64.3 GiB available RAM, 5.35 GiB artifact-volume space; discrete
GPU idle with 1.84 GB of 34.21 GB VRAM used. Preserve the reference's software
EGL and CPU group 8–15,24–31. Execute one native mission; earlier concurrent
native attempts distorted deadlines, so leave independent analysis paused.

Launched in session 97533, owner PID 3913233. Native launch and the live owner
were confirmed at 19 seconds elapsed. The actual pose worker (PID 3913294)
wrote `pose_worker_identity.json`: CadencedViewRevisitMotion,
CadencedViewRevisitPose, old-view period four frames. Poll this same session
and preserve the attempt; evaluate only after owner exit and complete sensor
persistence. The broad goal remains active and incomplete.

## Verified maze-1 result

Owner exited zero and complete physical evaluation verified both arrivals:
goal at camera frame 2164, home at frame 3171, each with a full one-second
quiet dwell and zero requested commands. Maximum physical distance during
the respective dwells was 9.921 mm and 22.133 mm (40-mm requirement); maximum
100-ms speed was 25.195 and 21.786 mm/s (50-mm/s requirement). No contact
occurred. Total simulation time was 317.82 s, path length 19.147 m and final
home distance 15.394 mm. Registered pose error was 3.019 mm median and 7.082 mm
maximum. This is one verified development round trip, not general reliability.

All 785 recorded selections matched the raw supervised world-model treatment
with no external neural XY correction. There were 406 on-time plans and 379
late plans. Initial survey took 26.4 s; tracking service was 66.621 ms median
and 76.051 ms 95th percentile. Median obstacle/planning service was
28.210/114.014 ms. There were two stale vetoes and seven latched request
intervals. Actual age treatment passed: 8099 accepted requests, 305 older than
200 ms, maximum accepted age 240 ms, actual-age charging on every accepted
translation. Maximum simulator lag was 97.877 s, so this is not real-time or
hardware qualification.

On 404 overlapping executed 700-ms windows, neural XY RMSE was 16.635 mm,
versus unused pose-command 8.721, command-history 12.691 and nominal 22.725 mm.
These are forecasts on the executed action, not alternative navigation
outcomes; the result establishes neither an RGB benefit nor a JEPA advantage.

The three-condition trajectory comparison and inspected PNG/SVG are saved in
`go2_pipeline_repair_navigation_comparison_v1_attempt_001`. The original
200-ms failure and 250-ms queue failure remain visible alongside this success.

Next fixed test: use the identical repaired controller, model, sensor settings
and budget on exposed short-pulse maze 0, with its original CPU group
0–7,16–23. Launcher: `scripts.run_go2_cadenced_view_maze00_development`.
Root: `go2_cadenced_view_age_250ms_supervised_rollout_noise_2mm_native_layout00_4800_v1_attempt_001`.
No tuning between the successful maze-1 run and this test; preserve any failure.
This is a second exposed development layout, not a fresh generalization test.
Keep the maze-1 success and the queue-overflow sensor recording in full.

Second-maze preflight: 16 physical/32 logical CPUs, 3.4% aggregate CPU load,
68.84 GB available RAM, GPUs idle (discrete VRAM 1.84/34.21 GB), and
5.39 GB free after routine diagnosed-depth retirement. No competing native
mission. Keep software EGL and CPU group 0–7,16–23; run one mission, with
no concurrent heavy analysis because prior concurrency distorted deadlines.

Maze 0 launched in session 28479, owner PID 3915732; process and native
launch confirmed live. Poll the same owner through persistence, then evaluate.

## Fixed second-maze failure

Owner 3915732 exited 1 with tracking failure after 670 acquired frames
(about 67 simulated seconds). Complete persistence and evaluation are saved:
no arrivals or contacts; 666 registered poses, median/max position error
1.195/3.665 mm. 156/166 plans were on time. Actual 250-ms age treatment
passed (3,094 accepted requests, 90 older than 200 ms, maximum 240 ms).
Thus the repaired supervised controller has one round trip and one tracking
failure on its two fixed exposed mazes. Preserve both outcomes and full sensors.

The cadenced public-sensor replay reproduced all 666 recorded raw poses exactly
and failed at frame 666: no retained or previous-frame image match passed.
The primary camera had no detected features; auxiliary feature support remained
but rigid matching/consensus failed. Original every-frame old-view replay is
being compared on the same fixed recording before changing the tracker.

Both sensor replays are complete. The original every-frame old-view tracker
also fails on this fixed recording, earlier at frame 615. Its old-view probe
substitutes frame 584 and drops retained frame 607 from the attempted set;
the cadenced tracker successfully uses frame 607 at that instant. Common
pre-failure pose differences reach 3.707 mm, so this is not an identical-state
counterfactual. Reverting cadence is not a demonstrated solution. The later
cadenced failure at 666 has no old-view attempt and no accepted recent or
previous-frame measurement. Both primary/auxiliary images were inspected:
primary faces a dark low-texture wall, auxiliary retains sparse floor/wall
features. Preserve the complete failure and both replay diagnostics.

Comparison: `go2_cadenced_maze00_failure_cadenced_replay_v1_attempt_001/tracker_comparison_v1.json`.
Next scientific step: examine the measured feature-support decline and executed
commands before the loss, then test observation-grounded view recovery before
tracking becomes unavailable. Also avoid displacing a useful recent reference
when an optional old-view probe fails. Do not weaken matching thresholds or
claim recovered navigation from this retrospective replay alone. No native
mission is currently running. The broad goal remains incomplete.
