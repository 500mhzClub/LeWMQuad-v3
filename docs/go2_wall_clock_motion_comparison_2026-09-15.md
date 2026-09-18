# Learned and fitted-motion navigation under wall-clock deadlines

The completed learned reference independently passed goal and home on exposed
transfer layout 1 with 2,482 poses, zero contacts and 248.18 simulated seconds.
The follow-up fixes the same scene, depth-noise recipe, local-reference tracker,
map/routing/view memory, command set, predictive clearance/stopping checks,
300-ms planning delay, 20-ms physical service and actual host-deadline limits.
Both missions run alone on the same CPU group, with one physics owner, one
non-stepping snapshot renderer and cyclic GC deferred during bounded execution.

The control changes only the scored motion forecasts: fitted pose-command XY
and integrated command yaw instead of learned corrected XY and learned yaw.
The frozen supervised model still runs in both arms so removing its inference
cost does not masquerade as a scoring effect. This control remains predictive;
it does not establish the effect of predictive planning versus no prediction.
There is one run per arm, no randomization and no general reliability claim.
Neither mission is a fresh-maze, sealed, final or hardware evaluation.

Reference: `go2_async_camera_wall_mission_learned_layout01_4800_v1_attempt_001`.
Control: `go2_async_camera_wall_mission_pose_command_layout01_4800_v1_attempt_001`.
Launcher: `scripts/run_go2_async_wall_fitted_control_development.py`.
The completed learned runtime sources are unchanged. A private factory supplies
the fitted-motion argument to the same runtime, and the final launch annotation
identifies the seven treatment fields consistently. Verify the actual saved
forecasts, both physical arrivals, timing and shared settings after owner exit.
The fitted control ran with the same 4,800-tick mission budget and exited 1
after 5:26.14 wall time, without swap. It independently passed the goal at
frame 1487 (maximum quiet-dwell distance 21.879 mm), but failed on return:
`renderer exceeded two in-flight camera acquisitions`. All 2,073 poses
registered, maximum pose error was 4.718 mm, and there were zero disallowed
contacts. All 508 fitted forecast assignments were verified; 483 were on time.
Two pending camera results were retained at shutdown. Physical-arrival,
forecast, treatment and host-timing evaluations are complete.

The maximum recorded host/simulation lag was 1,189.162 ms, with median/p95
0.545/28.985 ms. Median rendering cost was 22.745 ms and maximum 28.646 ms;
no cyclic-GC events occurred. The underlying long-delay cause remains to be
diagnosed. This is a failed complete mission, not evidence that fitted motion
could not navigate home or that learned forecasts are superior. No simulation
remains running after this control's exit. Preserve its full failure recording.

The paired comparison and inspected trajectory figures are complete under
`go2_wall_clock_motion_sources_comparison_layout01_v1_attempt_001`. Shared
settings and common runtime source hashes match; the queue failure stays in
the original comparison.

Further timing inspection shows accumulated service overload, not a single
one-second blocked call: in simulated seconds 200–207.5, physical service
averaged 20.97 ms (median 20.11 ms) against a 20-ms period. Host lag reached
1.189 s and fell to 0.826 s as physics caught up, before camera queue overflow.
The largest recorded request duration was 19.71 ms and largest gap from
request completion to physical-service start was 0.019 ms. The next full
follow-up uses the same fitted controller with timers around gait inference,
scene stepping and sensor/guard recording plus owner-thread CPU time. It
retains the same queue bounds, clocks, physics and control limits. Timer cost
is an explicit difference; this does not replace the original failure.
Launcher: `scripts/run_go2_wall_physics_timing_followup_development.py`.
Root: `go2_wall_physics_timing_pose_command_layout01_4800_v1_attempt_001`.

The timed follow-up exited 1 after 6:07.62, zero swaps, maximum RSS
13,777,000 KiB. It independently passed goal frame 1620 (quiet-dwell maximum
15.016 mm), then reproduced the camera queue overflow during return after
2,381 poses and 11,915 physical services. Maximum pose error was 4.871 mm,
zero disallowed contacts; all 585 fitted assignments were verified, 540 on
time. Both pending camera results were retained. Physical, treatment, forecast
and host-timing evaluations are complete. Maximum host lag was 771.498 ms.

Component receipts matched every service: ten scene steps, ten sensor/guard
samples and one gait inference each. During the final 220–238.3 simulated
seconds, sensor/guard recording averaged 13.506 ms per service, versus 5.025 ms
in the first 20 seconds. Scene stepping stayed near 3.104 ms and gait inference
0.640 ms. Total service wall time averaged 18.963 ms while owner-thread CPU
time averaged 9.499 ms; initially these were 9.004/8.049 ms. Most added elapsed
time was not owner-thread CPU work. Scheduling, GIL contention and native
worker waits remain possible explanations; the timers do not isolate them.
No cyclic-GC events occurred. The evidence points to delays during sensor/guard
recording rather than an expensive physics solve or neural gait inference.
Keep the complete failed recording and `wall_physics_cost_diagnostic_v1.json`.

The next complete-controller comparison is prepared in
`scripts/run_go2_async_wall_reactive_control_development.py`, root
`go2_async_camera_wall_mission_reactive_layout01_4800_v1_attempt_001`.
It uses the already exercised current-feedback controller with the same
tracking, mapping, routing memory, camera-view policy, sensor/noise recipe,
physical ledger, asynchronous cameras and host limits. Its model assignment
is explicitly `reactive`, with no model loaded or neural inference. Predictive
clearance/stopping anticipation and recovery differ from the learned arm;
this comparison cannot isolate predictive ranking or JEPA. Run alone after
the active physics-cost follow-up exits and has sufficient recording space.
This is one execution on an exposed development maze, not a new-maze cohort.

The reactive owner has launched alone (session 23838, PID 3673297), after the
timed fitted owner's exit and complete evaluations. Before launch there were
4,164,456,448 bytes free; a retained full-budget reactive recording occupies
3,586,146,304 bytes. Only depth from the completed, analyzed GC timing prefix
003 was additionally retired; its source/timing evidence remains. Reactive
arrival and timing results are pending. The timed fitted follow-up matches
all 159 common source identities and the recorded shared settings of its
original fitted reference; its two timing-source additions are explicit in
`original_fitted_comparison_v1.json`.

The reactive run exited 1 after 3:34.14, no swaps, maximum RSS 8,613,384 KiB.
Goal frame 1157 passed independent quiet-dwell checks (maximum distance
9.957 mm). It failed during return at registration with `current measured
candidate conflicts with transported floor reference`: 1,286 delivered camera
frames, 1,285 registered poses, maximum pose error 6.217 mm and zero disallowed
contacts. All 318 selected plans verified the model-free instantaneous treatment
and were on time. Host lag median/p95/maximum was 0.541/8.476/34.386 ms.
It did not reproduce camera overflow before its earlier registration failure;
this shorter run cannot rule out later scheduling overload without inference.
One pending camera result was retained after shutdown. The terminal zero-command
`PIPELINE_FAILURE` row lacks a duplicate `now_ns`; its exact host request time
is present in the separate physical-service ledger. Timing analysis explicitly
uses that recorded value, without modifying or retiming the original records.

All three original full missions are now evaluated and compared in
`go2_wall_three_controls_comparison_layout01_v1_attempt_001`, including inspected
PNG/SVG trajectories. All 142 common runtime source identities and shared
settings match. All reached goal; only learned completed return. The fitted
scheduling failure and reactive floor-registration failure remain in this
one-execution, exposed-maze comparison. There is no statistical, isolated
prediction-ranking or JEPA advantage claim. The extra timed fitted failure
remains a separate diagnostic follow-up.

`scripts/diagnose_go2_wall_reactive_floor_conflict_development.py` is replaying
the original noisy paired observations with the exact LocalViewRevisitMotion
and robust floor registration, retaining the terminal raw pose, floor anchor,
candidate-selection receipt and residuals. This is a sensor-estimator replay,
not a new physical navigation trial. Preserve the full reactive recording.

The replay completed in 143.81 s, matched all 1,285 recorded raw poses and
reproduced the registration rejection at frame 1285. The selected population
was 2,616 auxiliary-camera points, no primary-camera points. Its second
covariance eigenvalue was 0.000390535 m², below the unchanged 0.0004 m²
two-axis extent requirement. Under the transported-normal scalar-height
fallback, 14 points exceeded 3 mm residual; maximum 3.088102 mm and RMS
1.498889 mm. The diagnostic retained the same robust height selector and all
its selected points. This explains the rejection without relaxing thresholds.

The existing ReacquiringFloorRegistration was checked directly on that exact
terminal noisy packet and restored admitted anchor. It consumed frame 1285
as unavailable, published no pose, preserved anchor/reference, and cleared
only the failure latch for subsequent observations. This single-frame check
does not show subsequent recovery. Four existing focused reacquisition tests
passed. A full wall-clock follow-up now uses this existing behavior with the
same reactive controller, local-view tracker, noise and physical/host limits:
`scripts/run_go2_wall_reactive_floor_reacquisition_development.py`, root
`go2_wall_reactive_floor_reacquisition_layout01_4800_v1_attempt_001`.
It cancels pending motion on rejection, clears arrival dwell, excludes rejected
poses from mapping/arrival, and waits for four accepted floor observations
before resuming planning. The original reactive failure remains in its original
comparison. No synthetic gap is inserted and no floor threshold is changed.

Before this follow-up, superseded successful progress-rejoining layout 3 depth
was retired after its completed gyro diagnoses. This reclaimed 15.668 GB and
left 18.879 GB free. All failures, current first successes and the full reactive
replay input remain; no artifact relocation to the Steam drive was needed.

The recovery-enabled reactive follow-up exited 0 after 5:28.08, no swaps,
maximum RSS 12,376,216 KiB. Goal/home arrivals passed at frames 1176/2104,
maximum quiet-dwell distances 9.046/19.132 mm. All 2,106 poses registered,
maximum pose error 7.095 mm, zero disallowed contacts; all 519 actual reactive
plans finished on time. Timed execution was 210.54 simulated seconds and
211.049 wall seconds. Exact physical-command histories and camera timestamps
were checked. Host lag median/p95/maximum was 0.533/11.638/35.811 ms, with
201 lag vetoes and one initial dispatch 1.256 ms late. No hardware or hard
real-time qualification follows.

There were zero floor rejections and zero reacquisition holds. The later
trajectory therefore proves a full model-free reactive wall-clock round trip
with recovery enabled, but does not exercise recovery or attribute success to
that mechanism. The original failure remains. The comparison is
`go2_wall_reactive_floor_reacquisition_comparison_layout01_v1_attempt_001`:
143 common runtime sources match, with registration behavior and its two
additional source files explicitly identified. Keep this first full reactive
wall success in full. Original reactive and learned wall recordings remain.

The exact original failure replay additionally compared all raw-pose fields
and all registered positions across its 1,285 accepted frames; every value
matched, recorded in `full_raw_pose_and_registered_position_match_v1.json`.

The original/follow-up reactive PNG/SVG trajectories were visually inspected.
No native mission remains running. The six additional-seed correction fits
then completed in three CPU pairs; preparation and exact identities are in
`docs/go2_additional_training_seed_motion_fits_2026-09-15.md`. They add no new
navigation outcomes and do not resolve the fitted wall-clock scheduling
failure, the unexercised live recovery mechanism or hardware validation.
