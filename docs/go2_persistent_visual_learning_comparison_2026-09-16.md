# Frozen training-condition comparison

Eight fixed missions compare the existing pulse-trained JEPA and direct models
on the four locked persistent-visual transfer layouts. Alternate which model
runs first by layout. Reuse the unchanged persistent-local-view controller,
tracker, mapping, safeguards, pulse windows, depth noise, ideal gyro, physical
arrival checks, 4800-tick budget and per-layout CPU groups. Use one native owner
and software EGL; pause heavy concurrent analysis.

Both checkpoints were trained before these layouts were constructed, with the
existing matched initialization/data/update schedule. No retraining, checkpoint
selection or controller tuning uses this batch's outcomes. Load each assigned
model and its corresponding prediction head; retain raw nominal-composed XY/yaw
and disabled contact score. Preserve every outcome, including failures.

Compare with all eight completed supervised/pose-command outcomes. The layouts
are now exposed development layouts, and the controller was developed using
the supervised condition. This single-seed comparison therefore has a possible
controller-development preference and is not a pristine final evaluation or
complete causal attribution of training method. Timing differences also remain
part of each closed-loop execution. Do not equate forecast RMSE with navigation
success or infer alternative-policy outcomes from fixed recorded windows.

Frozen plan: `go2_persistent_visual_learning_plan_2026-09-16.json`.
Same inventory: `go2_persistent_visual_transfer_layout_inventory_2026-09-16.json`.
Launcher: `scripts.run_go2_persistent_visual_learning_comparison_development`,
assignments 1–8. Evaluate each only after owner exit and complete persistence,
using its assignment plus `--evaluate`. The launcher refuses execution until
the previous eight-mission comparison is finished. This experiment addresses
training-condition behavior; reactive/prediction-off and memory attribution,
replication, realistic sensing/timing and hardware evidence remain outstanding.

Plan preparation completed before any assigned-model navigation on these
layouts. The first JEPA/layout-0 mission launched: session 33547, owner PID
3939195. Native launch and live owner confirmed. Recorded checkpoint `jepa.pt`
SHA-256 `3790b3bdef97fbb3c340bc43d6d1fbccc6295fb0da80f015f11d064296303978`.
Preflight: CPU load 0.4%, available RAM 68.78 GB, GPUs idle, discrete VRAM
1.84/34.21 GB, artifact space 6.66 GB, no competing native mission; CPUs
0–7,16–23 and software EGL. Poll this same session through persistence and
owner exit before evaluating assignment 1; then run direct/layout-0 assignment 2.
The broad navigation goal remains incomplete.

Assignment 1 (JEPA/layout 0) completed and passed physical/model/dispatch
evaluation after owner exit and persistence: round trip 157.98 simulated seconds,
zero contacts. Goal/home maximum physical dwell distances were 14.541/15.445 mm,
maximum speeds 12.980/33.331 mm/s, with zero requested commands throughout.
Path 14.739 m, final home distance 9.003 mm; all 1,578 frames registered.
Plans: 319 on time, 67 late; maximum simulator lag 44.230 s. Five visual-recovery
triggers covered 17 plans, with two early terminal entries. On 318 executed
forecast windows, JEPA XY RMSE was 16.683 mm versus pose/command 8.874 mm.
This first result is similar to the supervised arm's 160.92-s successful trip;
one execution does not establish training-method superiority. No settings change.

Assignment 2 (direct/layout 0) launched: session 4631, owner PID 3940181.
Native launch and live owner confirmed. Checkpoint `direct.pt` SHA-256
`01397558c70669c1095efa130c008482e3524ec23f5109066d842a55943da326`.
Preflight: CPU load 0.4%, available RAM 68.65 GB, GPUs idle, discrete VRAM
1.84/34.21 GB, artifact space 5.50 GB, no competing native mission; same CPUs
0–7,16–23 and software EGL. Poll through complete persistence and owner exit,
then evaluate assignment 2 before the next fixed layout.

Assignment 2 completed after owner exit and full persistence. Direct/layout 0
reached a physically verified goal at frame 4225 but exhausted the 480.8-s
budget before returning; zero contacts, all 4,806 poses registered. Goal dwell
maximum distance 20.117 mm, maximum speed 0.505 mm/s. Final home distance
3.858 m. Plans: 1,011 on time, 136 late; maximum simulation lag 141.537 s.
The saved near_goal_hold_diagnosis_v1.json distinguishes two causes: frames
1000–4199 had 746/751 plans on time and 722 hold choices; at frame 2000 all six
actions passed forecast clearance and stopping projection, but hold had the
highest utility. The predictive arrival-hold override was ineligible there.
Frames 4500 onward instead had 0/76 plans on time. The terminal hold preference
and later return lateness are distinct; no raw-depth replay is pending.
Keep this failed return in every aggregate; no controller or model changes
within the frozen batch. Next is assignment 3, direct/layout 1.

The layout-0 four-arm comparison and PNG/SVG are saved at
`go2_persistent_visual_learning_comparison_layout00_v1_attempt_001`; PNG inspected.
Completed JEPA/direct depth retired under standing policy after diagnosis;
all non-depth comparison and failure records preserved.

Assignment 3 (direct/layout 1) launched: session 37392, owner PID 3943032.
Native launch and live owner confirmed. Frozen direct checkpoint unchanged.
Preflight: CPU busy 0.2%, available RAM 69.19 GB, GPUs idle, discrete VRAM
1.84/34.21 GB, recording space 5.87 GB; no competing native mission. CPUs
8–15,24–31 and software EGL, sequential execution to preserve timing comparison.
Broader environment-type testing remains deferred per user instruction.

Assignment 3 completed after owner exit and full persistence: direct/layout 1
passed physical goal and home checks, 219.60 simulated seconds, zero contacts.
Goal/home maximum physical dwell distances 18.563/21.123 mm and maximum speeds
30.101/17.965 mm/s, with zero requested commands throughout each one-second
dwell. Plans: 402 on time, 130 late; maximum simulator lag 66.607 s.
Model treatment and actual dispatch-age treatment passed, with 375 accepted
requests older than 200 ms and maximum accepted age 240 ms. This is direct
1/2 round trips so far, with its layout-0 failed return retained.
Path 16.011 m, final home distance 21.286 mm. On 391 matched executed
forecast windows, direct XY RMSE 18.001 mm versus pose/command 9.331 mm;
these overlapping windows are diagnostic, not independent trials. Completed
depth retired after evaluation; paired comparison needs no raw replay.

Assignment 4 (JEPA/layout 1) launched: session 30753, owner PID 3944128.
Native launch and live owner confirmed, frozen JEPA checkpoint unchanged.
Preflight: 0.5% CPU busy, 69.0 GB RAM available, 5.58 GB artifact space, GPUs idle.
Same CPUs 8–15,24–31 and software EGL; no competing native mission.

Assignment 4 completed after owner exit and full persistence: JEPA/layout 1
passed physical goal and home checks, 177.48 simulated seconds, zero contacts.
Maximum goal/home dwell distances 22.662/12.591 mm, maximum speeds
6.275/12.780 mm/s, zero commands throughout both one-second dwells.
Plans: 342 on time, 93 late; maximum simulator lag 53.588 s. Model and actual
dispatch treatment verified. Path 14.894 m, final home distance 12.52 mm.
On 340 executed windows, neural XY RMSE 17.121 mm versus
pose/command 8.519 mm. Four of eight planned runs complete; JEPA 2/2
round trips, direct 1/2, all zero contacts. This interim single-seed result
does not establish superiority; controller-development and timing caveats
remain. Next fixed assignment: JEPA/layout 2, then direct/layout 2.

Layout-1 four-arm result and PNG/SVG saved at
`go2_persistent_visual_learning_comparison_layout01_v1_attempt_001`; PNG inspected.
Completed JEPA/layout-1 depth retired; all comparison inputs retained.
Assignment 5 (JEPA/layout 2) launched: session 79774, owner PID 3945148.
Native launch and live owner confirmed, frozen model/controller unchanged.
Preflight: 0.3% CPU busy, 68.89 GB RAM available, 5.33 GB artifact space, GPUs idle.
CPUs 0–7,16–23, software EGL, no competing native mission.

Assignment 5 completed after owner exit and full persistence: JEPA/layout 2
passed physical goal/home checks, 159.98 simulated seconds, zero contacts.
Maximum goal/home dwell distances 11.413/20.915 mm, maximum speeds
35.512/1.378 mm/s, zero commands throughout one-second dwells. Plans:
344 on time, 46 late; maximum simulator lag 43.610 s. Model and actual
dispatch treatment verified. Path 15.202 m, final home distance 19.593 mm.
On 343 executed windows neural XY RMSE 13.479 mm versus
pose/command 7.269 mm. JEPA is 3/3 so far; next direct/layout 2.

Completed JEPA/layout-2 depth retired after evaluations, no raw replay pending.
Assignment 6 (direct/layout 2) launched: session 31408, owner PID 3945861.
Native launch and live owner confirmed; fixed model/controller unchanged.
Preflight: 0.3% CPU busy, 68.81 GB RAM available, 5.12 GB artifact space, GPUs idle.
CPUs 0–7,16–23, software EGL, no competing native mission.

Assignment 6 completed after owner exit and persistence: direct/layout 2
passed physical goal/home checks, 186.78 simulated seconds, zero contacts.
Maximum goal/home dwell distances 18.543/20.133 mm, maximum speeds
19.077/22.454 mm/s, zero commands throughout one-second dwells. Plans:
418 on time, 40 late; maximum simulator lag 50.952 s. Model and actual
dispatch treatment verified. Path 16.13 m, final home distance 15.02 mm.
On 411 executed windows neural XY RMSE 14.553 mm versus
pose/command 8.125 mm. Six of eight planned runs complete: JEPA 3/3
round trips, direct 2/3; all goals reached and zero contacts.

Layout-2 four-arm comparison and PNG/SVG saved at
`go2_persistent_visual_learning_comparison_layout02_v1_attempt_001`; PNG inspected.
Completed direct/layout-2 depth retired; no raw replay pending.
Assignment 7 (direct/layout 3) launched: session 14046, owner PID 3946740.
Native launch and live owner confirmed; frozen model/controller unchanged.
Preflight: 0.4% CPU busy, 68.64 GB RAM available, 4.87 GB artifact space, GPUs idle.
CPUs 8–15,24–31, software EGL, no competing native mission.

Assignment 7 completed after owner exit and full persistence: direct/layout 3
passed physical goal/home checks, 228.40 simulated seconds, zero contacts.
Maximum goal/home dwell distances 9.489/26.551 mm, maximum speeds
20.972/18.362 mm/s, zero commands throughout one-second dwells. Plans:
431 on time, 125 late; maximum simulator lag 68.544 s. Model and actual
dispatch treatment verified. Path 17.543 m, final home distance 26.714 mm.
On 413 executed windows neural XY RMSE 17.814 mm versus
pose/command 9.636 mm. Direct complete: 3/4 round trips, 4/4
goals, zero contacts. Its layout-0 return failure remains in the aggregate.

Completed direct/layout-3 depth retired after evaluation, no raw replay pending.
Assignment 8 (JEPA/layout 3) launched: session 38755, owner PID 3947622.
Native launch and live owner confirmed; frozen model/controller unchanged.
Preflight: 1.1% CPU busy, 68.72 GB RAM available, 4.57 GB artifact space, GPUs idle.
CPUs 8–15,24–31, software EGL, no competing native mission.

Assignment 8 ended with exit 1: measured visual pose unavailable, before any
arrival. Full failure persistence completed: 1,109 camera pairs, 1,105 registered
poses through frame 1104, 2,218 depth leaves retained. Physical evaluation found
zero contacts, path 3.049 m, final goal distance 1.608 m; no goal or return.
Model/dispatch treatments passed. Of 276 plans, 260 were on time. Frames
500–1108 contained 149 turn choices and three holds, no translation choices;
98 plans requested visual recovery across 6 triggers. The last route-turn
preference was blocked by the predicted reserve; a subsequent visual-recovery
turn was selected shortly before tracking loss. Failure diagnosis and full
raw sensor evidence retained. Exact cadenced-tracker replay launched in
session 12747 at `go2_persistent_visual_learning_jepa03_failure_cadenced_replay_v1_attempt_001`.
No retry or controller tuning replaces this frozen outcome.

Exact public-sensor replay completed in 108.46 s: all 1,105 recorded raw poses
matched exactly, followed by the same visual failure at frame 1105. Neither
camera supplied an accepted pose; primary references lacked rigid matches,
and auxiliary latest-reference gyro consensus failed. The failure frame had
no old-view revisit attempt under the four-frame cadence. This does not prove
a different tracker would recover or that recovery would complete navigation.
Retain the full raw recording for targeted follow-up. Layout-3 four-arm
comparison and PNG/SVG saved and PNG inspected. All eight fixed missions
are now complete, including both failures, with no replacement runs.

Complete sixteen-outcome aggregate saved at
`go2_persistent_visual_learning_complete_v1_attempt_001/result.json`.
Summary: `docs/go2_persistent_visual_learning_complete_result_2026-09-16.md`.
JEPA/direct each 3/4 round trips; supervised/pose-command each 4/4. All zero
contacts; JEPA 3/4 goals, other models 4/4. No JEPA advantage demonstrated.
No native mission or sensor replay remains live. Next: matched current-controller
reactive/prediction-off controls, bounded visual-failure follow-up, then memory
attribution/replication and sensor/timing realism. Broad goal remains active.
