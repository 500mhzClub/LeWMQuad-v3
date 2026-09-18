# 150-feature prospective navigation experiment

The latest 100-feature run lost rigid consensus at frame 58. On its same 66
public sensor frames, 150 and 300 features completed tracking with the
original six-cell/60% support criteria, optional plane refinement and partial
height registration. The 150-feature replay added 2.17 ms median tracking time
and had 6.43 mm maximum position error. The original and support-ablation
100-feature trackers failed identically on that trajectory.

The new run uses 150 features per camera, the original support criteria and
the already tested optional plane refinement. Navigation retains the learned
model, stored-map forecast filter, measured frontier visits, fine obstacle
memory, current-view checks and 1,800-tick mission. This changes feature
budget without using the support-threshold ablation. Its initializer matches
the successful paired replay: the standard 150-feature observer with the
optional-refinement callback.

Native artifact:
`go2_feature150_memory_clearance_native_layout00_v1_attempt_001`.
Session 93509 ended with a visual-tracking failure at frame 338, after 343
camera acquisitions and 1,715 requested policy steps. All 343 camera pairs
were saved. There were 69 on-time plans out of 84, final/minimum observed
goal distance was 1.623 m, and no goal or frontier-view completion. A frontier
view was pending. There was no physical contact stop. This remains a negative
navigation result.

Original-support 150-feature replay reproduces all 338 admitted raw poses
exactly, then fails with the same consensus/grid-support rejection. Its
median/max position error is 5.04/6.43 mm. The combined 150-feature and
conditioned-support variant completes all 343 frames, with first raw pose
difference at frame 84 and median/max position error 5.88/8.07 mm. It retains
the original noncollinear 3-D conditioning, absolute match minimum,
reprojection, gyro, motion, temporal and floor checks; it uses a strict
inlier majority without image-bin counting. Evidence:
`budget150_optional_plane_refinement_replay.json` and
`budget150_conditioned_support_replay.json` in this artifact directory.

The combined variant completed the full 1,805-frame frontier-visit trajectory
(session 16649). Median/max position error was 2.17/6.17 mm, versus the
original 100-feature result's 3.05/8.86 mm. Raw poses differ from frame 1
because the feature budget changes the measurements. This is a positive
replay result, not prospective navigation or real-time qualification.

A fresh combined experiment was run in
`go2_conditioned150_memory_clearance_native_layout00_v1_attempt_001`.
It completed all 1,805 frames and 9,025 policy steps over 181.02 simulated
seconds with no tracking failures or contacts. There were 412 on-time plans
and 38 late plans. All camera pairs were saved and the process exited
successfully. No goal arrival occurred; minimum/final observed goal distance
was 1.903 m. No frontier visit completed.

The planner selected hold 391 times, including 390 `NO_CLEAR_CANDIDATE_ZERO_REQUESTED`
decisions from the memory forecast filter. At the first such event, frame
244, a right arc was already committed for the 300 ms prefix. Every candidate
predicted path overlapped the stored footprint. Current-view dispatch checks
mostly passed (8,178 requests); the principal stall is therefore in the stored
map/planning path, not repeated current-view vetoes. A replay of frames
224–244 is checking predictions, subsequent observed positions, map timing and
actual requested-command execution to distinguish prediction error from a
newly observed obstacle. The same navigation controller and 1,800-tick mission
were retained. This remains a negative navigation result.

All 1,805 admitted poses had median/max post-estimation position error
1.54/7.14 mm; maximum native XY displacement was 1.406 m. Result SHA-256:
`63500fd77b7261bec8330d96283f8deb19a74d9509f30513c2c48aa760b87f05`.

The forecast replay distinguishes this transition from a newly seen obstacle.
All seven expected 100 ms command intervals executed for each inspected plan.
At frame 240, the selected right arc's minimum predicted distance was
0.45223 m, while the subsequent observed path reached 0.43456 m against the
same already available map. Position prediction error reached 19.9 mm at
700 ms; at frame 236 it reached 22.2 mm. Thus nominal mean-path clearance
with only a 0.45 m footprint left too little reserve for this prediction error.
Evidence: `memory_forecast_clearance_diagnostic.json` in the combined native
artifact directory.

The next development treatment adds a fixed 3 cm reserve for translating
candidate paths: they require 0.48 m distance to the same stored obstacle
squares. Hold/turn candidates retain the 0.45 m nominal footprint check.
The reserve is informed by this development error and is not a calibrated
uncertainty bound. Recorded rescoring changes frame 232 from left arc to
left turn and frame 240 from right arc to right turn; no changed-trajectory
success is inferred. Two focused tests passed in 1.74 s.

Completed prospective experiment:
`go2_translation_reserve_native_layout00_v1_attempt_001`, session 95902.
All 1,805 camera pairs were persisted and the owner process is terminal.
It completed 181.06 simulated seconds without contact or pipeline faults,
with 371/450 on-time plans, no arrivals and no completed frontier visits.
Minimum/final goal distance was 2.5107/2.5429 m. There were 369 holds,
including 294 no-clear-candidate decisions beginning at frame 628.
Result SHA-256: `4a9772582cc65514259136dee7654063db6765f1be96e7baf565da84af0ada8d`.

Forecast replay at frames 620 and 624 confirms all seven expected command
intervals executed. Their hold predictions had minimum clearance
0.45199/0.45226 m, but subsequent measured paths against those same maps
reached 0.44801/0.44443 m. Final prediction errors were 9.14/11.46 mm.
Thus translation-only reserve did not prevent drift into nominal overlap.

Public-sensor map reconstruction also identifies an upstream route/waypoint
mismatch (`lookahead_clearance_diagnostic.json`). The radial 0.35 m waypoint
can skip a route bend: at frame 320 its straight shortcut has 0.44609 m
clearance, while earlier route points preserve about 0.482 m. At frame 480
the selected shortcut has 0.45671 m, versus about 0.485 m for a nearer point.
The next experiment checks successive shortcut segments before selecting
the waypoint, preserving available start clearance up to the existing 0.48 m
translation requirement. The model, action scoring, action-path filter and
current-view dispatch checks remain unchanged. This is a prospective
hypothesis, not evidence of navigation success.

The lookahead experiment completed in
`go2_clearance_lookahead_native_layout00_v1_attempt_001`, session 84573,
exit 0 with all 1,805 camera pairs saved. No contacts, pipeline faults or
goal arrivals occurred. All 450 plans found a clear candidate; there were
only 17 holds, two completed frontier views, and minimum/final goal distance
1.4299/1.7383 m. The waypoint changed on 225 decisions and never lacked a
visible route point. This supports the route-shortcut diagnosis on this
trajectory, but does not establish reliable navigation.

Only 82/450 plans met the 300 ms deadline. Median end-to-end planning latency
was 338 ms; after frame 600, median planning-stage duration was 160 ms versus
104 ms in the previous stalled run. Tracking/registration times were similar.
Result SHA-256: `ae1c77940c13d262511ed73f9307f6359f173654cde938708a7bb437a2e79adf`.

An offline profile using frame 600's reconstructed public-sensor map measured
14.00 ms for routing, 1.58 ms for lookahead, 6.35 ms for model inference and
14.75 ms for the memory forecast filter. These isolated measurements do not
reproduce concurrent simulator contention. An exact distance optimization
uses endpoint distances as an upper bound and segment/cell bounding-box
distances as lower bounds, excluding only cells that cannot attain the
minimum. The same frame's memory filter fell to 4.76 ms and lookahead to
0.50 ms, with an exactly equal action and full clearance receipt. Five
focused tests passed, including 200 random full-versus-pruned geometric
comparisons and stationary, crossing and tangent segments.

Completed prospective artifact:
`go2_pruned_clearance_lookahead_native_layout00_v1_attempt_001`.
The optimization changes neither obstacle geometry nor selection criteria.

Session 47750 exited 0 with 1,806 camera pairs saved. Planning met the
deadline on 419/450 decisions, compared with 82/450 before optimization.
No contacts or pipeline faults occurred, but there were no arrivals. One
frontier visit completed; minimum/final goal distance was 1.5158/2.1616 m.
All plans found a clear candidate, but 316 chose hold. Result SHA-256:
`c4ec78e353ca97556d4ef6d26542f4b3b31a7085ad33c6edab218b7103d07eec`.

The stall now reflects a reserve rule that includes the unchangeable prefix.
At frame 448, every candidate shares a 0.4665 m minimum in the first 300 ms.
Forward then predicts increasing clearance, with 0.4951 m minimum during
the 700–800 ms zero-command tail. Nevertheless the fixed 0.48 m full-path
requirement rejects it. This is distinct from nominal footprint overlap.

The next treatment permits a translating reserve-recovery candidate only
when its prefix remains strictly outside 0.45 m, its remaining predicted
path never falls below that prefix's minimum clearance, and its entire
700–800 ms tail clears 0.48 m. It preserves the nominal footprint and requires
restoring the full reserve by the commitment endpoint. The reserve remains
a development heuristic, not a calibrated error bound. On the same recorded
forecasts, this changes frames 432, 440, 448 and 600 from hold to forward;
frames 408, 416 and 424 remain unchanged. Evidence:
`memory_forecast_clearance_diagnostic.json` and `reserve_recovery_rescoring.json`.
No counterfactual navigation success is claimed.

The reserve-recovery experiment completed in
`go2_reserve_recovery_lookahead_native_layout00_v1_attempt_001`, session 65560,
exit 0 with 1,805 saved camera pairs. There were 407/450 on-time plans and
one selected recovery action (forward at frame 188). No contact or pipeline
fault occurred, but no frontier visit or goal arrival completed. Minimum/final
goal distance was 1.6794/1.6910 m. There were 373 no-clear-candidate decisions
and 336 initial current-obstacle vetoes. Result SHA-256:
`18b691a7e848e741f9ff428d186910d59230402e3ca7a6d4cc14e4553103571f`.

The robot stopped during its first frontier inspection. At frame 300 its
predicted and subsequent observed minimum clearance against the same stored
map were both 0.50748 m, with all expected commands executed. At frame 304,
the old-map predicted and observed minima were both 0.51100 m. Newly revealed
wall points then reduced current map clearance to 0.46914 m at frame 308 and
0.44712 m at frame 312. The first current-depth veto at 32.5 s measured a
0.44385 m obstacle distance. Paired RGB at frame 310 shows the close wall.
Thus this failure is newly observed geometry, not simply inaccurate motion
relative to the previously observed map. Evidence:
`memory_forecast_clearance_diagnostic.json` and saved frame-310 camera images.

The next experiment begins a frontier view when remaining route length is
at most 0.50 m, rather than waiting until the robot is within 0.10 m of the
frontier. It aims from the actual viewpoint toward the unknown neighbour.
Route length prevents premature triggering across an intervening bend, and
records distinguish a stand-off view from physical frontier arrival. The
model, reserve recovery, geometry and dispatch checks remain unchanged.
This tests earlier active observation; it is not full-footprint visibility
qualification. Next artifact: `go2_standoff_frontier_native_layout00_v1_attempt_001`.

The stand-off experiment completed with exit 0 and 1,805 saved camera pairs.
It completed three frontier views without the previous current-depth wall
trap. There were 422/450 on-time plans, no contacts or pipeline faults, and
no goal arrivals. Minimum/final goal distance was 1.6091/2.0764 m. A later
stored-map stall produced 310 no-clear-candidate decisions. Result SHA-256:
`213de6aa5d109c0244d75e6aafd451e9db4bdc4d5164930e48b17015414b96ce`.

Same-map replay identifies motion prediction error in this later transition.
At frame 464, all seven expected command intervals executed, but the 700 ms
XY prediction error was 48.5 mm. Predicted minimum clearance was 0.49061 m
versus observed 0.47307 m. At frame 468 those values were 0.47231/0.45551 m
with 29.2 mm XY error and all expected intervals executed. The forward
transition predicted 7.53 cm body-forward displacement at 700 ms versus
3.17 cm observed, alongside a sideways error. These errors motivate a
motion-forecast study rather than another clearance-margin change.

`scripts/fit_closed_loop_motion_residual_development.py` fits a fixed-ridge
XY residual from frozen JEPA forecasts, causal registered pose history and
known commands. Four earlier runs supply fitting windows; this stand-off
run is excluded from fitting and used for development validation. Labels
require the forecast requests to match actual requested commands through
each horizon. Long stationary training tails are thinned by a fixed rule.
No native poses or future observations enter inference features; neural
weights remain frozen. Results separate moving, translating, turn-only and
transition windows. Study artifact:
`go2_closed_loop_motion_residual_study_v1_attempt_001`.

The study completed in 38.0 s, fitting 1,207 windows and evaluating 448
windows from the excluded stand-off run. For the 119 moving validation
windows with all seven command intervals matched, 700 ms position RMSE
fell from 20.85 to 8.03 mm; p95 fell from 44.79 to 15.19 mm. For 67 command
transition windows, RMSE fell from 24.85 to 9.62 mm. For 54 translating
windows it fell from 26.76 to 10.16 mm, and for 65 turn-only windows from
14.18 to 5.69 mm. These are development replay measurements against
registered visual poses, not novel-layout or physical navigation success.

The frozen fit SHA-256 is
`ef7511b29afd0117291f600d7afad6adccdcd90199d0ea1f45519d7b81b01638`.
Runtime integration uses only the exact current and previous three registered
poses; future poses are unnecessary. On recorded frame 464, runtime features
match the saved validation features exactly. Candidate-prefix equality and
unchanged yaw/contact outputs were checked. The residual is applied before
waypoint scoring and stored-map filtering. No neural weights, clearance
thresholds or frontier settings changed.

Next prospective artifact:
`go2_motion_residual_standoff_native_layout00_v1_attempt_001`.

Attempt 001 stopped at initial planning with a tuple/list command-prefix
concatenation error. The failed artifact is preserved. The correction now
accepts the live ledger's tuple representation as well as recorded lists;
the numeric inputs and frozen coefficients are unchanged. Fresh artifact:
`go2_motion_residual_standoff_native_layout00_v1_attempt_002`.

Attempt 002 completed with exit 0, 1,805 saved camera pairs and 431/450
on-time plans. No contacts or pipeline faults occurred. It completed two
stand-off views, but no goal arrival; minimum/final goal distance was
1.7627/1.7706 m. A later turn revealed another close wall, causing 346
no-clear-candidate decisions and 338 initial current-obstacle vetoes.
Result SHA-256:
`885a1b991c010d4c895d9d7afae2f6af19e00f63ccb3b29d4c809cea4b4fc5e9`.

Motion accuracy improved on this fresh trajectory, which was excluded from
both fitting and the prior validation. Among 76 moving windows with all
seven requested intervals matched, raw/corrected 700 ms position RMSE was
24.09/7.46 mm and p95 was 46.66/12.71 mm. Among 49 command-transition windows,
RMSE was 25.98/7.50 mm. These paired predictions share the same actual
trajectory; this does not infer how an uncorrected controller would navigate.
Evidence: `prospective_motion_correction_accuracy.json`.

The corrected same-map replay reproduces recorded raw and corrected XY
forecasts exactly. At frame 408, predicted/observed old-map minimum clearance
was 0.51599/0.51413 m with all seven intervals executed. At frame 412 the
old-map observed path remained 0.51229 m clear, although its later commands
were vetoed. Newly observed geometry then reduced current-map clearance to
0.46351 m at frame 416 and 0.43758 m at frame 420. Earlier single-direction
frontier views had not exposed that wall.

The next treatment keeps the frozen residual and clearance rules, but
requires a panoramic frontier inspection before excluding the frontier.
After acquiring the initial target heading, it advances through eight
45-degree increments, returning to that heading. Each of nine stages needs
measured heading alignment and a subsequently measured map. Earlier stage
receipts remain immutable. This is sequential active observation, not a
full-footprint visibility certificate. Next artifact:
`go2_panoramic_frontier_motion_residual_native_layout00_v1_attempt_001`.

That run completed with exit 0 and all 1,805 camera pairs saved, but stopped
before reaching a frontier. Therefore it did not exercise the panoramic
frontier treatment. There were 418/450 on-time plans, no contacts or pipeline
faults, and no arrivals. Minimum/final goal distance was 2.4772/2.4798 m;
373 no-clear-candidate decisions began at frame 108. Result SHA-256:
`79b97e35c526ac7b58f267ac8e1e76c7d05f66ebc13f16eaea06fdcd4b9bb3f6`.

The initial forward-facing map had nearly 2 m observed clearance. During the
first turn, a newly visible obstacle reduced stored clearance from 1.7305 m
at frame 32 to 0.5877 m at frame 36. Continued turning/drift brought it to
0.4519 m by frame 108. The next experiment adds the same measured panoramic
sequence at the initial pose, before any translating route is selected.
It retains the resulting map and then proceeds with ordinary exploration,
including panoramic frontier inspections. Source tests require all nine
heading/map stages and reject completion from stale maps or wrong headings.
Next artifact: `go2_initial_panorama_motion_residual_native_layout00_v1_attempt_001`.

The initial-panorama run terminated after 94 acquired camera pairs with a
tracking failure at frame 71; only two of nine survey stages completed.
Both cameras still supported incremental motion, but neither independently
supported a retained reference and the ten-frame measured bridge allowance
was exhausted. No physical contact caused the stop. Replaying at 150 and
300 features both reproduced failure at frame 71. Doubling the feature
budget improved accuracy but did not resolve retained-anchor loss.

A joint-camera diagnostic lifted each camera's within-view correspondences
into the calibrated body frame and checked reprojection in its own camera.
Reference 58 to frame 71 passed with 12/19 inliers (seven primary, five
auxiliary), 0.422 mm fitting RMS and 0.00284 rad gyro disagreement. Other
retained references failed. This supplies additional measured support;
the bridge allowance and residual, reprojection, conditioning, gyro and
temporal thresholds remain unchanged.

The joint fit is now a fallback within the existing retained-anchor search
and measured-plane refinement. The first integration replay stopped at
frame 60 because the public witness interface only admitted single-camera
labels; that failed artifact is preserved. The interface now explicitly
represents joint measurements and their per-camera inlier counts, and
overlap retention counts the pooled reference feature population.
`joint_camera_replay_attempt_002.json` accepted all 94 recorded frames,
first differing from the original raw pose at frame 60. Median/maximum
registered position error was 4.603/6.018 mm, evaluated using native state
only after estimation. A synthetic known-motion test also verified the two
camera transforms and rejection of inconsistent auxiliary reprojections.
This is development replay evidence, not a completed panoramic survey or
navigation mission. Fresh prospective artifact:
`go2_joint_camera_initial_panorama_native_layout00_v1_attempt_001`.

That prospective run completed with exit 0 and 1,806 persisted camera pairs.
Tracking survived throughout. All nine initial survey stages completed at
29.5 s simulation time, and one nine-stage frontier panorama completed at
80.3 s. There were no contacts or pipeline faults. Plans were on time for
337/450 decisions. The mission exhausted its 1,800-tick budget with no
arrivals. Minimum/final observed goal distance was 1.6177/1.8100 m.

This run continued exploring instead of entering the previous sustained
clearance stall. Its last fifteen plans had an observed-floor route to the
goal; the final admitted plan selected forward motion on that route.
Native evaluator-only comparison across all 1,806 frames gives registered
position median/p95/maximum errors of 11.12/14.56/16.37 mm; the final error
was 15.24 mm. This demonstrates prospective tracking and survey completion,
not goal-reaching or novel-layout generalization.

The next run keeps the controller and sensing treatment and increases only
the global navigation budget from 1,800 to 3,600 ticks. This allows time for
the newly discovered goal route and return. Existing in-memory capture at
this size is approximately twice the prior run's storage population and
fits available host memory. Fresh artifact:
`go2_joint_camera_extended_round_trip_native_layout00_v1_attempt_001`.

The 3,600-tick attempt terminated early with a floor-registration conflict:
`current measured candidate conflicts with transported floor reference`.
It persisted 752 camera pairs, with registered poses accepted through frame
748. The initial panorama completed; no contact caused the stop and no goal
arrival occurred. The last observed goal distance was 1.6803 m. The changed
execution timing produced a different closed-loop trajectory, so extending
the prospective budget did not reproduce the earlier trajectory verbatim.
A public-sensor replay is diagnosing the rejected floor candidate before
any further native attempt.

The public replay exactly reproduces the original raw poses and registration
failure at frame 749. Its full floor anchor is frame 606. All 4,010 current
candidates come from the auxiliary camera. After the existing scalar height
update, two candidates exceed the 3 mm transported-plane residual limit;
the maximum is 3.0174 mm. This is not evidence of another physical surface:
the current points fit a plane with maximum/RMS residuals of 2.84/0.72
micrometres. Its second-axis RMS spread is 4.653 cm, below the fixed 5 cm
normal-admission cutoff. Native state, loaded only for independent evaluation,
gives current-fit normal error 0.00000448 rad versus transported-normal error
0.007075 rad. The failure is therefore explained by transporting a less
accurate orientation while rejecting the well-fitting current normal.

The next development sensitivity treatment admits a current plane with
second-axis RMS spread at least 2 cm, the existing rigid visual fit's spatial
conditioning scale. It retains the 100-point minimum, all-point 3 mm residual
check, up alignment and correction bounds. The original 5 cm default remains
available; the 2 cm treatment is explicitly selected before observer creation
in every process. This changes geometric admission and is not a calibrated
real-sensor uncertainty rule. Thirteen focused plane/height tests pass,
including collinear and incoherent-point rejection under the new setting.

The 2 cm replay accepted all 752 saved frames, with the first raw pose
difference at frame 607. Median/maximum registered position error was
8.94/12.57 mm against evaluator-only native state. No rejection threshold
was changed after observing this replay. Fresh six-minute prospective run:
`go2_two_cm_floor_joint_camera_round_trip_native_layout00_v1_attempt_001`.

That attempt completed the initial panorama, then failed tracking at frame
574, preserving 588 camera pairs. Registered poses ended at frame 573 and
the last observed goal distance was 1.6672 m. The public replay reproduced
all preceding raw poses exactly. Neither retained references nor either
single-camera previous-frame fit passed. This was not bridge exhaustion.

A direct frame-573/574 diagnostic found six primary and twelve auxiliary
descriptor pairs. Joint calibrated registration accepted 17/18 pairs
(six primary, eleven auxiliary), with 0.906 mm RMS fitting residual.
Joint optical flow independently accepted 31/46 pairs (nine primary,
twenty-two auxiliary), with 0.865 mm RMS. The existing joint-camera path
was limited to retained-reference reacquisition and could not use this
consecutive-frame evidence.

The previous-frame joint-fit extension passed all 588 replay frames, but
its first version also replaced earlier auxiliary-camera successes after
a primary-camera failure. That replay is preserved. The candidate has
been narrowed to the auxiliary fallback path so successful primary and
auxiliary fits take precedence. Bridge limits, promotion rules and
anchor/increment conflict checks remain unchanged.

The narrowed joint previous-frame replay accepted all 588 frames, with
median/maximum registered position error 6.62/12.22 mm. Its first raw pose
difference is frame 56: pooled descriptors can pass before the old separate
optical-flow fallback is needed. Thus this preserves successful single-camera
descriptor fits, not every historical association choice. Fresh six-minute
prospective artifact:
`go2_joint_previous_frame_round_trip_native_layout00_v1_attempt_001`.

Comparison follow-up: the existing six-action reactive selector matches the
command bank, but its controller uses the earlier execution loop. Its old
results are not timing-matched continuous baselines. Prospective comparisons
must retain the current acquisition, pose, obstacle, mapping, survey, command
commitment and mission machinery while replacing the action-selection method.
The recent closed-loop motion residual is an additional learned treatment;
it must be separately ablated or fitted from the same development population
for matched predictive models before attributing a difference to JEPA.
Removing only model packet history is not an ablation of persistent spatial
memory when the observed map and frontier history remain available.

The joint previous-frame native attempt acquired and persisted all 3,614
camera pairs. It accepted 3,612 registered poses and completed the initial
panorama and two frontier panoramas. It reported an outbound arrival at frame
2242 and entered RETURN, then exhausted the 3,600-tick budget 2.5603 m from
home. The owner ultimately exited 1 because its fixed three-second simulated
drain allowance ended before queued sensor work released. Later clock-closed
faults are cleanup consequences, not the primary failure.

The reported outbound arrival is NOT native verified. Its observed distance
was 34.22 mm, but native distance during the associated one-second dwell
was 49.80–55.33 mm, outside the unchanged 40 mm physical requirement.
Registered pose median/p95/maximum error over the run was 8.92/15.12/15.78 mm.
No continuous goal or round-trip success is claimed from this run.

Planning met its deadline for 560/896 decisions, falling to 8/100 in the
last hundred. Median planning-stage cost grew from 74 to 120 ms as the map
grew. The frame-2000 map was reconstructed from public RGB-D and recorded
registered poses; its route status, lookahead target and selected action
reproduced exactly. Profiling isolated repeated obstacle inflation, repeated
fine-cell setup and segment queries. The optimized version caches two exact
cell populations, uses bitmap inflation with the same offsets, and queries
a spatial index with a conservative radius before applying the original
segment-to-square distance calculation. The recorded route, target, action
and every candidate clearance remain identical. Five profiled evaluations
fell from 0.193 to 0.088 s with the same map reused; these are warm-cache
offline timings, not prospective deadline results. Tests additionally match
all-cell distances for 102 diverse segments and original inflation for five
radius/population combinations. Twenty focused geometry and mission tests pass.

Arrival handling is also corrected prospectively: the exact public goal is
appended to an observed route through its containing cell, subject to the
same shortcut-clearance check. A 2 cm observed arrival radius is selected
for this development run; the independent physical requirement remains 4 cm.
This reserves room for the measured pose error without claiming a calibrated
error bound. Final queue draining now continues zero-command physics until
the queues empty or a 20-second wall-time limit expires, rather than stopping
the simulation clock after a fixed three simulated seconds. Fresh artifact:
`go2_indexed_geometry_precise_goal_round_trip_native_layout00_v1_attempt_001`.

That run completed with exit 0, all 2,540 camera pairs and registered poses
saved, no contacts and no pipeline faults. Both native arrival checks passed:
the outbound/home one-second dwell maximum distances were 6.73/16.09 mm,
within the unchanged 40 mm physical requirement. Both dwells had zero
requested commands and native 100 ms speed below 0.05 m/s. The controller
retained its map and tracker while returning. Plans were on time for
594/625 decisions; timed wall/simulation duration was 254.70/254.22 seconds.
This is the first verified continuous development round trip. It remains
ideal-sensor simulation on heavily developed layout 0, not generalization
or hardware evidence. See `go2_first_continuous_round_trip_2026-09-13.md`.
