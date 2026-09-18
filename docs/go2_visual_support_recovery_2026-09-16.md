# Measured visual support recovery experiment

Run one full supervised world-model mission on exposed short-pulse maze 0.
Retain the 250-ms age bound, every-four-frame old-view probes, model, sensors,
CPU allocation, motion/obstacle gates and 4800-tick budget of the failed
cadenced maze-0 attempt. No tracker acceptance threshold changes.

The saved failure's exact replay shows both cameras first below 48 selected
features at frame 566, then intermittent recovery before terminal loss at 666.
The prospective rule first triggers at planning frame 568 on that fixed tape,
using a strongly supported view acquired at frame 552. This is a warning signal
on one exposed trajectory, not calibrated tracking confidence or a recovered
navigation outcome.

At planning observations, remember the latest measured pose where either camera
has at least 96 selected features. When neither has 48, request its heading
only if that reference is no older than 10 seconds and within 0.20 m. Complete
recovery using measured heading within 0.1 rad and at least 48 features in one
camera. Reset references on mission phase change. The thresholds are fixed
development heuristics. They do not establish visibility, reliable matching or
an error bound. No native geometry or pose supplies the recovery target.

Recovery changes the view objective and clears the previous objective's latched
turn direction. Learned future yaw/path forecasts still choose among hold/turn
candidates, and all existing predicted and current clearance, stopping, actual
age, commitment and mission gates remain. Original reference substitution is
unchanged in this experiment so its separate weakness is not conflated with
view recovery. Preserve every failure and run to the full mission terminal.

Seven focused checks passed, covering local/recent reference selection, measured
completion, phase reset, missing evidence, and existing age/stopping dispatch.
Preflight: 16 physical/32 logical CPUs, 0.5% CPU utilization, 68.85 GB available
RAM, GPUs idle with 1.84/34.21 GB discrete VRAM used, 4.87 GB artifact space.
Keep software EGL and CPUs 0–7,16–23. Run one native mission without concurrent
heavy analysis; earlier concurrency distorted timing.

Launcher: `scripts.run_go2_visual_support_recovery_development`.
Root: `go2_visual_support_recovery_supervised_rollout_noise_2mm_native_layout00_4800_v1_attempt_001`.
Evaluate physical arrivals/contacts after owner exit and complete persistence,
then inspect actual recovery triggers, selections, tracking accuracy and latency.
Do not claim unseen-layout reliability, JEPA/RGB benefit or hardware readiness.

Launched in session 8727, owner PID 3918438. Native launch and owner confirmed
live at 47 seconds elapsed. Preserve this attempt and poll through persistence.

## Completed result

Owner exited zero after complete persistence. Physical evaluation verified one
outbound arrival at frame 3871: maximum goal distance during the one-second
zero-command dwell 20.251 mm, maximum 100-ms speed 17.248 mm/s. No contacts.
The full 480.92-second mission ended at its unchanged time limit without a
verified home arrival. All 4,805 camera frames produced registered poses;
median/max position error 3.841/11.692 mm. Path length 26.949 m; final physical
home distance 54.656 mm. There were 965 on-time and 230 late plans. Maximum
simulator lag 138.348 s; no real-time or hardware qualification.

Actual recovery: 20 triggers, 113 planning observations (101 left turns,
12 right turns). First trigger was frame 892, after the previous attempt failed
at 666. Timing and trajectories already differed before the new rule acted;
therefore this is not isolated causal evidence that recovery prevented that
failure. It is one prospective goal-reaching run with continuous tracking,
not a verified round trip or unseen-layout reliability.

On return, observed distance fell from 85.56 mm at frame 4694 to 1.28 mm at
4699 while the body was still moving about 148 mm/s. Zero requests began for
the 4700 interval. The required dwell started at 4702, but observed drift crossed
20 mm at 4704 (20.63 mm), resetting its two quiet intervals. Later turning did
not establish the home dwell before the deadline. Preserve this near-miss as a
failure, not an arrival. Existing terminal control begins short pulses only
within 0.10 m; delayed execution of earlier commands needs examination before
changing either arrival criteria or the mission budget.

The goal and actual model/dispatch treatments are independently evaluated.
`visual_support_navigation_readout_v1.json` preserves recovery and terminal
approach details. All sensor recordings remain full for active diagnosis.
Next: examine the queued commands and predicted approach before the 0.10-m
terminal transition, and test earlier braking/pulse entry with the same physical
arrival requirement. Replication and matched baseline tests remain necessary.
No native owner is running; the broad navigation goal is incomplete.
