# Fixed-controller fresh-layout comparison

Eight prospective missions: four new development mazes, each with supervised
learned forecasts and the fitted pose/command predictor. Alternate which arm
runs first by layout. Keep the successful exposed-pair controller unchanged:
camera-cadence local-view recovery, tracker, terminal pulse rules, timing,
clearance guards, arrival checks, 2-mm synthetic depth noise, ideal gyro and
4800-tick budget. All alternative forecasts, including the same supervised
network, are computed in both arms; only the forecast used for selection and
predictive checks differs. This controls computational work without claiming
identical measured scheduling on divergent closed-loop trajectories.

The deterministic generator excludes the explicit 84-layout source registry
and within-batch duplicate topology/embedding. Selection uses structural rules
only, before navigation. Runtime receives the public goal and sensor packets,
not native geometry or evaluator routes. No outcomes may select layouts or
change the controller within this batch. Preserve and report all failures.

Frozen plan: `go2_persistent_visual_transfer_plan_2026-09-16.json`.
Inventory: `go2_persistent_visual_transfer_layout_inventory_2026-09-16.json`.
Launcher: `scripts.run_go2_persistent_visual_transfer_development` with
`--assignment 1` through `8`; evaluate each only after owner exit and persistence
using the same assignment plus `--evaluate`. The completed predecessor pair and
unchanged controller source hashes were checked during plan preparation.

One native owner at a time, software EGL, original CPU groups by layout parity.
Previous concurrency distorted measured deadlines, so additional simultaneous
native runs would change this comparison. Record actual hardware availability
at each launch and pause heavy analysis during native execution. Retire completed
diagnosed depth under the standing policy as needed, retaining active failure
inputs and all non-depth outcomes.

Primary outcome: both physically verified arrivals, each within 40 mm for one
second with zero requested commands and measured speed below 50 mm/s, with no
disallowed contacts. Also report goal-only outcomes, contacts, tracking failures,
budget exhaustion, path length, duration and timing. One execution per condition
and layout is preliminary evidence, not a reliable success-rate estimate.

This batch addresses fresh-layout performance and learned versus simple motion
prediction. JEPA training, prediction-on/off, memory contribution, repetitions,
realistic sensor imperfections and real-time/hardware validation remain further
requirements of the full goal. Neither this batch nor the exposed-pair success
completes that goal.

Assignment 1 launched: session 5119, owner PID 3928852. Native launch confirmed.
Preflight recorded 16 physical/32 logical CPUs, 0.3% CPU load, 69.09 GB available
RAM, idle GPUs (discrete VRAM 1.84/34.21 GB), 6.19 GB artifact space, and no
competing native mission. Layout 0 uses CPUs 0–7,16–23. Poll this same owner
through persistence and exit before physical evaluation; do not restart on an
observation timeout.

Assignment 1 completed and passed physical/treatment evaluation after owner exit
and persistence: verified round trip in 160.92 simulated seconds, zero contacts.
Goal/home maximum physical dwell distances were 13.539/15.739 mm; maximum
100-ms speeds 8.103/17.688 mm/s, with zero requested commands throughout both
one-second dwells. There were 312 on-time and 82 late plans; maximum simulator
lag 45.928 s, so this remains measured simulation rather than real-time evidence.
Actual raw learned forecast and age-charged dispatch treatments passed. This is
one fresh-layout success; the matched control and remaining layouts are pending.

On assignment 1's 312 executed 700-ms windows, XY forecast RMSE was 15.228 mm
for the applied learned model versus 8.122 mm for pose/command, 11.085 mm for
command history and 18.292 mm for nominal command integration. These overlapping
windows are diagnostic, not independent trials or alternative-policy outcomes.
The learned navigation success does not itself demonstrate forecast superiority.

Assignment 2 launched with the same layout and settings: session 39437, owner
PID 3929977. Native launch confirmed; only pose/command XY and command yaw are
used by action selection and predictive checks. Poll through exit/persistence,
then evaluate before proceeding to assignment 3.

Assignment 2 completed and passed physical/model/dispatch evaluation: round trip
445.28 simulated seconds, zero contacts. Goal/home maximum physical dwell
distances 11.977/18.477 mm and speeds 17.827/22.362 mm/s, both zero-command dwells.
Plans: 329 on time, 768 late. Between frames 1200 and 4000, 690 of 693 plans were
late despite selecting forward/right-arc movement. Median observation-to-plan
completion was 324 ms against a 300-ms dispatch deadline. Planning measured
service time rose to 132.968 ms in this interval (104.398 ms before frame 1200,
106.467 ms after frame 4000); tracking medians stayed around 64–66 ms. This
diagnoses the direct reason commands did not execute, not the internal cause
of the planning-service increase. No policy or deadline was changed.

The first fresh-layout pair therefore has 1/1 round trips for each arm and zero
contacts. Learned completion was faster, but timing effects prevent attributing
that difference solely to forecast quality. Preserve both in
`go2_persistent_visual_transfer_comparison_layout00_v1_attempt_001`.
Six assignments on three further fixed layouts remain in this batch.

The paired PNG/SVG trajectory figure was generated and visually inspected. Paths
were 14.861 m (learned) and 15.606 m (pose/command). During the control's pause,
13,940 requests had NO_ON_TIME_PLAN and only 60 passed current dispatch. Its own
328 executed forecast windows gave pose/command XY RMSE 8.066 mm versus neural
16.119 mm; the same-window accuracy ranking therefore agrees across both tapes.
Neither forecast accuracy alone nor a single mission time proves navigation
superiority. Completed pair depth was retired under the standing policy after
comparison; all non-depth evidence remains.

Assignment 3 launched on fresh layout 1, pose/command first as fixed in the plan:
session 46514, owner PID 3932074. Native launch and live owner confirmed.
Preflight: 0.3% aggregate CPU load, 68.93 GB available RAM, GPUs idle,
discrete VRAM 1.84/34.21 GB, artifact space 5.36 GB, no competing native mission.
CPUs 8–15,24–31; software EGL. Poll this same session through complete persistence
and owner exit, evaluate assignment 3, then run assignment 4 (learned/layout 1).
Keep the controller and plan unchanged throughout the remaining batch.

Assignment 3 completed and passed physical/model/dispatch evaluation after owner
exit and persistence. Round trip: 199.28 simulated seconds, zero contacts,
goal/home maximum dwell distances 10.826/10.246 mm and maximum speeds
12.901/20.763 mm/s. Path 15.270 m, final home distance 3.751 mm. All 1,991 camera
frames registered; 350 plans on time, 140 late, maximum simulator lag 63.833 s.
No visual recovery triggers; two early terminal entries. On 348 executed
forecast windows, pose/command XY RMSE was 9.235 mm versus neural 17.856 mm.
No controller change follows from this result. Three of eight missions complete.

Assignment 4 (learned/layout 1) launched: session 97435, owner PID 3933322.
Native launch and live owner confirmed. Preflight: CPU load 0.2%, available RAM
68.92 GB, GPUs idle, discrete VRAM 1.84/34.21 GB, artifact space 5.07 GB,
no competing native mission; same CPUs 8–15,24–31 and software EGL. Await complete
persistence and owner exit before evaluation; no controller/settings changes.

Assignment 4 completed and passed physical/model/dispatch evaluation. Round trip
204.88 simulated seconds, zero contacts; goal/home maximum dwell distances
8.420/10.448 mm, maximum speeds 16.468/22.999 mm/s, zero-command dwells.
Path 15.610 m, final home distance 9.507 mm; all 2,047 frames registered.
Plans: 368 on time, 135 late; maximum simulator lag 61.886 s. One visual-recovery
trigger covered three plans; the expanded terminal threshold added no entries
beyond the original terminal mode. Its 365 executed windows gave learned XY
RMSE 15.781 mm versus pose/command 8.837 mm. Keep this diagnostic separate from
the actual paired navigation outcomes.

Four of eight missions are complete: both learned and pose/command controllers
have 2/2 verified fresh-layout round trips and zero contacts. Layout-1 times are
similar (204.88 versus 199.28 s); layout-0 times differ mainly alongside the
recorded late-plan pause. No general reliability, learning advantage or JEPA
advantage follows. The second pair's results and trajectory comparison are in
`go2_persistent_visual_transfer_comparison_layout01_v1_attempt_001`.

The layout-1 PNG/SVG comparison was generated and visually inspected. Its
completed depth recordings have been retired under the standing policy; all
non-depth comparison evidence remains. Routine retirement of the older completed
rollout-off/layout-0 depth additionally restored room for further missions;
its failure outcome and clearance diagnosis remain unchanged.

Assignment 5 (learned/layout 2) launched: session 27587, owner PID 3934560.
Native launch and live owner confirmed. Preflight: CPU load 0.4%, available RAM
68.97 GB, GPUs idle, discrete VRAM 1.84/34.21 GB, artifact space 7.69 GB,
no competing native mission; CPUs 0–7,16–23 and software EGL. Poll this same
session through complete persistence and owner exit before evaluating assignment
5. Assignment 6 is the matched pose/command controller on the same layout.
The broad navigation goal remains incomplete.

Assignment 5 completed and passed physical/model/dispatch evaluation after owner
exit and persistence. Round trip 148.42 simulated seconds, zero contacts;
goal/home maximum dwell distances 11.895/19.748 mm, maximum speeds
17.765/7.734 mm/s, zero-command dwells. Path 15.063 m, final home distance
19.889 mm; all 1,482 frames registered. Plans: 329 on time, 35 late;
maximum simulator lag 41.540 s. Three visual recovery triggers covered 16 plans;
two early terminal entries. On 328 executed forecast windows, learned XY RMSE
was 14.058 mm versus pose/command 7.346 mm. Five missions complete; learned
controller has 3/3 fresh-layout round trips so far. The third matched control
and both final-layout missions remain pending. No controller change.

Assignment 6 (pose/command/layout 2) launched: session 66259, owner PID 3935489.
Native launch and live owner confirmed. Preflight: CPU load 0.3%, available RAM
68.82 GB, GPUs idle, discrete VRAM 1.84/34.21 GB, artifact space 6.61 GB,
no competing native mission; same CPUs 0–7,16–23 and software EGL. Poll through
persistence and owner exit, then evaluate assignment 6 before the final layout.

Assignment 6 completed and passed physical/model/dispatch evaluation after owner
exit and persistence. Round trip 176.06 simulated seconds, zero contacts;
goal/home maximum dwell distances 7.280/15.123 mm and maximum speeds
20.683/10.261 mm/s, zero-command dwells. Path 15.976 m, final home distance
15.183 mm; all 1,757 frames registered. Plans: 377 on time, 55 late;
maximum simulator lag 48.293 s. Three recovery triggers covered 16 plans, with
two early terminal entries. On its 375 executed windows, pose/command XY RMSE
was 8.797 mm versus neural 15.657 mm.

Third pair: both controllers completed round trips, learned 148.42 s versus
pose/command 176.06 s, zero contacts. Requested translation/turn/zero durations
were 81.22/44.80/22.18 s for learned and 87.18/57.60/30.90 s for pose/command;
this describes divergent executed trajectories, not a same-state causal test.
Pair results, forecast diagnostics, recovery exposure and trajectory comparison:
`go2_persistent_visual_transfer_comparison_layout02_v1_attempt_001`.
Six of eight missions are complete; each arm has 3/3 fresh-layout round trips.
The two fixed final-layout missions remain, with no controller changes.

The layout-2 trajectory PNG/SVG comparison was generated and visually inspected.
Both completed depth recordings were retired under the standing policy; every
non-depth result and comparison input remains. Assignment 7 (pose/command/layout
3) launched: session 18062, owner PID 3936649. Native launch and live owner
confirmed. Preflight: CPU load 0.3%, available RAM 68.94 GB, GPUs idle, discrete
VRAM 1.84/34.21 GB, artifact space 7.25 GB, no competing native mission; CPUs
8–15,24–31 and software EGL. Await complete persistence and owner exit before
evaluation. Assignment 8 is the learned controller on this same final layout.

Assignment 7 completed and passed physical/model/dispatch evaluation after owner
exit and persistence. Round trip 237.22 simulated seconds, zero contacts;
goal/home maximum dwell distances 16.349/16.613 mm, maximum speeds
19.842/31.817 mm/s, zero-command dwells. Path 17.736 m, final home distance
8.471 mm; all 2,370 frames registered. Plans: 436 on time, 149 late;
maximum simulator lag 76.140 s. One recovery trigger covered four plans;
two early terminal entries. On 431 executed windows, pose/command XY RMSE
was 8.560 mm versus neural 15.940 mm. The simple predictor therefore has
4/4 verified fresh-layout round trips with zero contacts. Seven missions complete;
the final learned mission remains. No controller or settings changes.

Assignment 8 (learned/layout 3) launched: session 58653, owner PID 3937970.
Native launch and live owner confirmed. Preflight: CPU load 0.4%, available RAM
68.82 GB, GPUs idle, discrete VRAM 1.84/34.21 GB, artifact space 5.56 GB,
no competing native mission; same CPUs 8–15,24–31 and software EGL. Poll this
session through complete persistence and owner exit before evaluation, then
aggregate all eight fixed outcomes without dropping any failure.

Assignment 8 completed and passed physical/model/dispatch evaluation after owner
exit and persistence. Round trip 201.68 simulated seconds, zero contacts;
goal/home maximum dwell distances 13.766/17.827 mm, maximum speeds
9.998/14.767 mm/s, zero-command dwells. Path 17.197 m, final home distance
17.957 mm; all 2,013 frames registered. Plans: 377 on time, 110 late;
maximum simulator lag 61.588 s. No visual-recovery triggers; two early terminal
entries. On 373 executed windows, neural XY RMSE was 17.191 mm versus
pose/command 9.780 mm.

All eight fixed missions are complete: each arm has 4/4 physically verified
round trips, zero contacts. Shared recorded source/settings bindings matched
the frozen plan for every mission. The pose/command predictor had lower recorded
same-window XY error on all eight trajectories; mission duration is confounded
by differing execution timing, especially the layout-0 control pause. Full
summary: `go2_persistent_visual_transfer_complete_result_2026-09-16.md`.
Aggregate: `go2_persistent_visual_transfer_complete_v1_attempt_001/result.json`.
The broad navigation goal remains incomplete.
