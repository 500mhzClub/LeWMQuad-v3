# Fixed-controller second development maze

Run the successful persistent-local-view supervised world-model controller on
exposed short-pulse maze 1. Do not change model, controller, tracker, thresholds,
sensor treatment, arrival checks or budget between this mission and maze 0.
The recorded local controller source hashes match the completed maze-0 launch.
Only the layout/noise-layout index and its original CPU allocation change.
Use the same 2-mm depth noise, ideal gyro, software EGL, 4800-tick budget,
20-mm observed arrival radius, 40-mm physical radius and one-second quiet dwell.

Maze 0 completed one verified round trip in 355.58 simulated seconds with zero
contacts. Its no-age-limit branch was not exercised; do not attribute that
success to removing expiry. This fixed second-layout test assesses the final
controller without further exposed-maze tuning. Both layouts have been used
in development, so even two successes would not establish unseen generalization.

Preflight: 16 physical/32 logical CPUs, 1.7% aggregate CPU load, 69.42 GB available
RAM, GPUs idle (discrete VRAM 1.84/34.21 GB), 4.51 GB artifact space after routine
retirement of completed diagnosed depth. No competing native mission. Use CPUs
8–15,24–31 with one owner; pause concurrent heavy analysis to preserve timing.

Launcher: `scripts.run_go2_persistent_local_visual_maze01_development`.
Root: `go2_persistent_local_visual_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001`.
Evaluate only after owner exit and complete persistence. Preserve every outcome,
including tracking, contact and budget failures. Then move to fixed prospective
independent-layout comparisons with matched controls; no JEPA/RGB or hardware
claim follows from these exposed-layout results.

Launched session 39231, owner PID 3926063. Native launch and live owner
confirmed at 18 seconds elapsed. Poll this same session through persistence
and exit before evaluation. Broad navigation goal remains incomplete.

## Completed result

Owner exited successfully and recording persistence completed. Post-run physical
evaluation passed both arrivals: goal and home each remained within 40 mm for
one second, with zero requested commands and measured speed below 50 mm/s.
Maximum physical distance during the dwells was 13.024 mm at the goal and
12.490 mm at home; maximum 100-ms speed was 21.740 and 37.942 mm/s respectively.
The round trip took 374.48 simulated seconds, with zero disallowed contact
samples. Of 912 plans, 498 arrived on time. Actual learned forecast treatment
and dispatch observation-age charging passed evaluation.

Together with maze 0, the unchanged final controller has two verified round
trips from two executions on two exposed development layouts, with zero
contacts (355.58 and 374.48 simulated seconds). This demonstrates end-to-end
operation, not reliability on fresh mazes or superiority to simpler controls.
All earlier failures remain part of the development record. JEPA advantage,
isolated predictive-planning benefit, fresh-layout generalization, real-time
operation and hardware performance remain unestablished. Next scientific step:
hold this controller fixed and compare against matched simpler controls on
fresh development layouts.
