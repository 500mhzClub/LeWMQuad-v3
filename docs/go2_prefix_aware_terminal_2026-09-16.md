# Queued-prefix terminal approach experiment

Run one fixed supervised world-model mission on exposed short-pulse maze 0.
Keep the preceding visual-support recovery, cadenced tracker, sensors, model,
4800-tick budget, 20-mm observed arrival radius, 40-mm physical requirement,
one-second quiet dwell and all clearance/dispatch checks.

The previous return approach selected a 400-ms translating arc at frame 4692,
121.5 mm from home. Its queued prefix forecast covered 57.2 mm forward, while
the full selected forecast covered 117.9 mm by 700 ms. At 4696 the robot was
47.9 mm away; the now-selected hold still had a queued prefix forecasting
53.7 mm forward. It crossed home before settling, exceeded the observed radius
after two quiet intervals, and failed the return dwell. These are saved
forecasts/commands and measured poses, not counterfactual motion outcomes.

New terminal-mode entry radius is 0.10 m plus the sum of planar command-speed
magnitudes in the three committed 100-ms prefix intervals, times 0.1 s.
For three 0.16-m/s commands this is 0.148 m. Apply only when the observed-floor
route target is the exact mission endpoint, with no scanning objective.
This selects the existing 100-ms translation candidates earlier; both forecasts
and commitment duration use those same pulse commands. Terminal positioning
and predictive-hold logic enter together. Queued command travel is a mode-switch
heuristic, not an observed pose or calibrated stopping-distance bound.

Twelve focused checks passed for prefix expansion, exact-goal versus frontier/
scan applicability, feature-view recovery and the existing age/stopping gates.
Launcher: `scripts.run_go2_prefix_aware_terminal_development`.
Root: `go2_prefix_aware_terminal_supervised_rollout_noise_2mm_native_layout00_4800_v1_attempt_001`.
One native owner, software EGL, CPUs 0–7,16–23; no concurrent heavy analysis.
Preserve the full outcome, including a failure. Evaluate after owner exit and
complete persistence. This exposed-maze follow-up cannot establish general
reliability, isolated causal improvement, JEPA/RGB benefit or hardware readiness.

Preflight: 16 physical/32 logical CPUs, 0.5% utilization, 69.53 GB available
RAM, GPUs idle (discrete VRAM 1.84/34.21 GB), 5.49 GB artifact space.
Launched in session 78351, owner PID 3921468; native launch and live owner
confirmed at 19 seconds elapsed. Poll this same owner; do not restart on a
poll timeout. No heavy analysis until native terminal. Broad goal incomplete.

## Tracking failure before terminal-mode exposure

Owner exited 1 and persistence/evaluation completed. There were 1,138 acquired
frames, 1,134 registered poses, no arrivals or contacts, and 271/283 on-time
plans. Registered median/max position error was
1.935/5.085 mm. Earlier terminal mode was
never newly enabled, so this attempt does not test its stopping effect.

Public replay reproduced every recorded raw pose exactly and failed at frame
1134 with no qualified image reference, no old-view probe at the failure.
Feature support fell below 48 in both cameras at 1127, briefly rose to 59
at planning frame 1128, then the planning-cadence rule triggered at 1132.
Its turn could not start before 1135, after the loss. The camera-cadence
state update instead retains the warning from 1127 through planning frame
1128 on the fixed recording; this is not a recovered navigation outcome.
Keep this failure and full sensors. Next fixed test updates recovery at each
registered camera frame, binding its state to the matching planning observation.
