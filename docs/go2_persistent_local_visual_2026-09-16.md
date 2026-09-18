# Persistent local visual recovery objective

Run one fixed full mission on exposed short-pulse maze 0. Retain the framewise
attempt's model, tracker, feature thresholds, terminal-prefix rule, command
windows, nominal/predicted clearance checks, sensor settings, arrival checks and
4800-tick budget. Remove only the ten-second maximum age for starting a recovery
toward the last strongly supported local view. Its 0.20-m spatial limit remains.

The preceding attempt failed tracking with no arrivals or contacts. At failure,
the last recorded recovery target was 30.9 seconds old but only 71.1 mm away.
The old age rule would exclude this target even though it remains local. A
historical heading is only an objective: it supplies neither a current pose nor
current feature support nor obstacle clearance. Recovery completion still needs
a newly measured heading within 0.1 rad and at least 48 current features in one
camera. Every action still passes learned forecast and current dispatch gates.
The possibility that an old view no longer has useful features is tested by the
full native outcome, not assumed away.

Fourteen focused checks passed, including persistent local versus remote views,
fresh feature support for completion, camera-time evidence isolation and the
existing terminal/dispatch checks. Preflight: 16 physical/32 logical CPUs,
0.4% CPU utilization, 69.45 GB available RAM, GPUs idle (discrete VRAM
1.84/34.21 GB), 4.60 GB artifact space after routine diagnosed-depth retirement.
Use one native owner, software EGL and CPUs 0–7,16–23; no heavy concurrent work.

Launcher: `scripts.run_go2_persistent_local_visual_development`.
Root: `go2_persistent_local_visual_supervised_rollout_noise_2mm_native_layout00_4800_v1_attempt_001`.
Preserve every failure. Evaluate arrivals, contacts, tracking and actual recovery
after owner exit and complete persistence. This exposed development attempt
does not establish unseen-maze reliability, a JEPA/RGB advantage or hardware
readiness, regardless of its outcome.

Launched session 11900, owner PID 3923944. Native launch and live owner
confirmed at 16 seconds elapsed. Poll this same owner through persistence and
exit, then evaluate. The full navigation goal remains incomplete.

## Verified round trip

Owner exited zero and persistence completed. Goal at frame 2448 and home at
3552 both passed independent one-second quiet zero-command dwell checks.
Maximum physical distances during the goal/home dwells were 10.170/6.804 mm;
maximum 100-ms speeds were 9.681/9.285 mm/s. Both are within the unchanged
40-mm and 50-mm/s requirements. No contacts. Total simulation 355.58 s,
path 24.635 m, final home distance 6.953 mm. All 3,554 camera frames produced
registered poses; median/max position error 3.451/8.488 mm. There were 639
on-time and 241 late plans. Maximum simulator lag 102.017 s: this is not
real-time or hardware validation.

Fourteen recovery triggers covered 74 planning observations. The oldest
reference was 5.0 s old at a trigger and 6.8 s old during active recovery.
Thus the removal of the ten-second limit was NOT exercised. Do not attribute
the success to that change. The trajectory and timing differed from previous
attempts; a single successful execution does not establish reliability.

Earlier terminal mode was exercised once on return: frame 3520, distance
107.065 mm, expanded entry radius 148 mm, selected right turn. Subsequent
plans 3524/3528/3532/3536 selected 100-ms forward pulses, followed by the
verified dwell. This demonstrates actual use, not isolated causal benefit.
The raw supervised world-model selection and actual 250-ms dispatch treatment
were independently verified; no external neural XY correction was used.

`persistent_local_visual_readout_v1.json` records treatment exposure. Two
three-panel comparison artifacts preserve all six maze-0 development outcomes,
including the three tracking failures and two incomplete returns:
`go2_maze00_recovery_sequence_part1_v1_attempt_001` and
`go2_maze00_recovery_sequence_part2_v1_attempt_001`.

Keep this success's full sensors. Next hold the controller fixed: repeat it on
the other exposed development maze, then run a prospective independent-layout
comparison with matched non-neural controls. Do not infer JEPA/RGB advantage,
general reliability or hardware readiness from the exposed successes. No native
owner is running; the broad navigation goal remains incomplete.

Both comparison PNG/SVG pairs were generated and the PNGs visually inspected.
All six outcomes and their physical paths are visible, including failures.
