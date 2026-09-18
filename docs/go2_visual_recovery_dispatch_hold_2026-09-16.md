# Prompt visual recovery hold: exposed-layout development trial

The retained JEPA/layout-3 failure shows a timing gap between detecting weak
visual support and executing a recovery action. Camera frame 1101 (111.60 s)
triggered recovery; registration published it at 111.778 s. The previous
left turn continued, including a new left-turn plan based on frame 1100 that
finished at 111.786 s. The next sampled planning frame, 1104, selected a right
turn, first requested at 112.20 s. Tracking failed on frame 1105, acquired at
112.00 s. At the first right-turn request the rate-limited applied command
still had positive yaw (+0.10 rad/s). Thus recovery arrived after the failed
image was acquired. This does not prove an earlier stop prevents failure.

Exact stage timestamps, mission publications, plans and 20-ms requests are in
`visual_recovery_dispatch_timing_v1.json` in the original JEPA/layout-3 root.
The negative stored-view and newest-view replays remain unchanged.

Run one prospective closed-loop mission on the same exposed layout 3 using
the same frozen JEPA checkpoint, physics/appearance seeds, CPU allocation,
4800-tick budget and sensor conditions. The sole intervention is to cancel
pre-trigger command windows when registration publishes a camera-cadence
visual recovery trigger. Discard any pre-trigger plan still computing then.
The commitment ledger records cancellation at publication time, not camera
capture time, and rejects forecasts whose assumed prefix was interrupted.
Post-trigger plans retain all existing forecast, clearance, observation-age,
floor, mission and prefix checks. Tracking thresholds, camera cadence,
reference retention, map and recovery view selection remain unchanged.

Implementation: `lewm/visual_recovery_dispatch_hold_development.py`.
Launcher: `scripts/run_go2_visual_recovery_dispatch_hold_development.py`.
Output: `go2_visual_recovery_dispatch_hold_jepa_noise_2mm_native_layout03_4800_v1_attempt_001`.
The original failure is preserved. Record every recovery publication and
cancelled command window, physical contacts, arrivals, full outcome and
failure. Evaluate after the owner exits and native persistence finishes.
Passing the old failure point alone does not count as successful navigation.

Run sequentially: additional native simulation or heavy concurrent analysis
would change deadline behavior. Available artifact storage before launch is
5.42 GB. This is an exposed development intervention, not an independent
reliability result, JEPA advantage, realistic-sensor validation or hardware
evidence. Broader environment-type tests remain deferred.

## Prospective result and control repeat

The hold-enabled run exited normally and physically passed both arrivals,
with zero contacts, in 232.12 simulated seconds. Maximum dwell distances were
8.874 mm outbound and 18.889 mm at home; maximum 100-ms speeds during those
dwells were 20.068 and 0.912 mm/s. Planning met 456/572 deadlines.

However, there were **zero recovery publications**, no active planning-time
recovery states, and all 11,598 requests retained the unset recovery threshold.
The new hold was never exercised. This success is not evidence of a repair;
the original controller also could have followed this trajectory. The original
failure remains in the study population. Exact run sources and the explicit
negative treatment receipt are retained in the output root.

Next, run one original-controller repeat using `--control`, before changing
tracking or recovery again. Keep the same model, layout, sensors, CPU group
and mission budget. Output:
`go2_visual_recovery_original_control_jepa_noise_2mm_native_layout03_4800_v1_attempt_001`.
It tests outcome variability under measured execution schedules; a single
repeat cannot establish reliability or isolate which timing difference matters.
The successful hold-enabled recording has no pending raw-depth replay and may
be retired under the standing policy after retaining its full non-depth evidence.

## Original-controller repeat result

The original controller also completed a physically verified round trip:
216.88 simulated seconds, zero contacts, maximum dwell distances 15.190 mm
outbound and 7.496 mm at home. Maximum dwell speeds were 0.255 and
22.453 mm/s. Both arrivals included one second of zero requested commands.
Planning met 424/529 deadlines. Its existing view recovery was active in
four plans; the new dispatch hold was not enabled.

The original failed run and control repeat first requested different commands
at 29.00 s. Both selected a left arc from frame 272, but the failed run's plan
finished at 28.998 s (on time) while the repeat finished at 29.002 s (late).
The repeat therefore held. The hold-enabled success first diverged from the
failed run at 31.36 s: the original held for stale/unavailable obstacle
evidence while the success continued forward. These differences preceded the
original tracking loss by over a minute. They establish timing-related command
variation, not that any single delayed command caused or prevented the loss.
Identical frozen model/layout identities and unchanged predecessor controller
sources were checked across all three runs.

Combined evidence:
`go2_visual_recovery_timing_comparison_v1_attempt_001/result.json`.
It preserves all three outcomes and first command differences. These follow-ups
do not replace the original 28-mission study or change its 3/4 JEPA result.
The original controller has one failure and one success on this exposed layout;
the unexercised hold-enabled success is a separate development trial.

The hold remains experimental and is not adopted as a demonstrated repair.
Next work should address reliability under execution timing variation and
test recovery on trajectories that actually lose visual support. Repeating
only successful schedules or relaxing tracking acceptance is not evidence of
robustness. Broader environment types and hardware remain untested.

Follow-up: the [fixed 20-ms planning-latency stress pair](go2_planning_latency_stress_2026-09-16.md)
caused both JEPA and supervised planners to exhaust their mission budget,
with tracking intact and zero contacts. Nearly all ordinary routing plans
missed the deadline. This identifies shared timing margin as a concrete
limitation to address before interpreting another successful rerun as robust.
