# Fixed polygon-floor repeatability comparison

The corrected floor mapper completed one verified exposed-maze JEPA round trip
in 289.04 simulated seconds. That pilot is excluded from this batch's totals.
Run four fixed assignments: **JEPA, supervised, supervised, JEPA** on exposed
layout 1, keeping every outcome. This tests repeatability and model transfer of
the geometry correction; two repetitions per model cannot establish reliable
generalization or a causal JEPA contribution.

Plan: `docs/go2_polygon_floor_repeatability_plan_2026-09-17.json`.
Launcher/evaluator: `scripts/run_go2_polygon_floor_repeatability_development.py`.
It reuses the existing frozen-readout runner/evaluator, substituting the
polygon mapper initializer and explicit launch annotations. The existing
`frozen_readout_navigation_readout_v1.json` filename/schema is retained; the
new root, plan, arm and outer batch-assignment fields identify each mission.

All four use `InterruptedViewRuntime`, `ProjectedPolygonFloorRoutingMap`, the
same frozen JEPA/supervised readouts, six candidates, 4800-tick budget, 2-mm
depth noise, ideal gyro, CPU group 8–15/24–31 and the existing 300-ms deadline
plus 20-ms wait. No controller/model changes between assignments and no extra
repetitions to obtain favorable results. Preserve full depth through analysis
and every failure thereafter. Native jobs run sequentially; no heavy work
competes with timed simulations. The output drive had 18 GiB free at preparation.

Primary outcome is independent physical verification of goal-and-home arrivals
with no disallowed contact. Also retain physical backtracking, tracking errors,
coverage-view outcomes, planning deadlines and trajectory-conditional prediction
scores. Wait for the owner to exit after recording persistence, then evaluate
each assignment before launching the next. Hardware readiness, realistic
sensor uncertainty, real-time qualification and fresh-layout reliability remain
unproven. Use only public observations for control; native state stays evaluator-only.

Output root pattern:
`/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1/go2_polygon_floor_repeatability_{number:02d}_{arm}_noise_2mm_native_layout01_4800_v1_attempt_001`.

## Execution

No outcome is included before the corresponding owner and evaluator finish.

Assignment 1 (JEPA) launched in session 86624, owner PID 4137997, verified live.
Launch metadata confirms the four fixed assignments, the polygon mapper,
unchanged pilot controller and JEPA readout state
`f372e75c1a5c4b3933beb9d59ee97158ce17a8a2b567a89c9be59b74cf8112a8`.
Outer log: `/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1/go2_polygon_floor_repeatability_01_launch.log`.
Detailed progress is in the mission root's `worker.log`. No terminal result yet.

Assignment 1's owner exited zero after recording persistence. Evaluation session
41521 exited zero: **verified goal arrival, no home arrival, budget exhausted
at 480.98 simulated seconds, zero disallowed contacts and no pipeline faults**.
The goal dwell's maximum physical distance was 17.27 mm and maximum 100-ms
speed was 0.01097 m/s; all one-second quiet-arrival checks passed. On return,
eleven unique corridor edges reversed outbound edges, with zero invalid graph
transitions. The mission had thirteen unique outbound edges. This is verified
partial physical backtracking, not a completed round trip.

There were 4805 camera pairs and 1187 selections: 929 on time, 258 late
(78.26% on time). Live wall time before archival was 641.67 s. Actions were
49 holds, 210 left turns, 253 right turns, 126 right arcs, 168 left arcs and
381 forward plans. All nine recorded coverage-view completions observed their
requested patches; no fresh-view failure was recorded. There were ten
translation-coverage rejections. Preserve the full failed recording.

Saved timing profiles identify a post-batch diagnostic, without changing this
comparison. In assignment 1, route wall time was median 20.98 ms for on-time
plans and 128.31 ms for late plans (late p95 404.00 ms, maximum 880.30 ms).
Model-forward thread CPU time was median 9.07/8.89 ms respectively. Late plans
comprised 116 frontier-route plans, 141 goal-route plans and one coverage-view
plan; the slowest eight had no active coverage-view target. The successful
pilot showed the same route-time contrast (23.14/154.08 ms medians) but fewer
late plans overall. Whole-plan wall time includes release waits and must not
be equated directly with the recorded deadline condition. These associations
identify where to inspect execution cost; they do not prove that route latency
alone caused the timeout or establish an alternative mission outcome.

Assignment 2 (supervised) launched after assignment 1's owner and evaluator
exited, in session 15889, owner PID 4139664, verified live. Launch metadata
confirms the polygon mapper and supervised readout state
`5f2a862f1f7650c20655b4e99fe74ed5fcac1c6b76b2405ae260988d90e25c44`.
Outer log: `/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1/go2_polygon_floor_repeatability_02_launch.log`.
No controller/model change was made. Assignments 3 (supervised) and 4 (JEPA)
remain unexecuted.

Prepared `scripts/profile_go2_polygon_floor_routing_development.py` while
assignment 2 was running; only syntax was checked, no replay executed. It
requires assignment 4's evaluated result, reconstructs the actual map updates
for selected assignment-1 states (2372, 2644, 2920, 2924), and profiles the
coarse and fine-goal routing subproblems. Full controller/frontier state is
explicitly outside its scope. Existing clearance-preference receipts took only
1.09/1.88 ms at frames 2372/2644 despite route totals near 400 ms, so the replay
will distinguish base proposal work from fine-goal search rather than assuming
the weighted preference is responsible. No comparison source was modified.

Assignment 2's owner exited zero after persistence; evaluation session 75119
exited zero. **Verified round trip in 349.66 simulated seconds, zero disallowed
contacts and no pipeline faults.** Maximum physical goal/home dwell distances
were 19.74/13.68 mm; maximum 100-ms speeds were 0.03025/0.01785 m/s. Both
one-second quiet-arrival checks passed. All eleven return corridor edges
reversed outbound edges, with zero invalid graph transitions.

There were 3494 camera pairs and 866 selections: 712 on time and 154 late
(82.22% on time). Live wall time before archival was 460.40 s. Actions were
16 holds, 138 left turns, 173 right turns, 120 right arcs, 123 left arcs and
296 forward plans. All six recorded coverage-view completions observed their
requested patches; there were no recorded fresh-view failures and seven
translation-coverage rejections. Keep the full recording through batch analysis.
The batch currently has one round trip and one goal-only timeout; the successful
pilot is not pooled into these totals. No repeatability or JEPA advantage claim
is warranted from the two completed assignments.

Assignment 3 (supervised) launched after assignment 2's owner and evaluator
exited, in session 22927, owner PID 4141246, verified live. The launch receipt
confirms the same polygon mapper and supervised readout state as assignment 2.
Outer log: `/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1/go2_polygon_floor_repeatability_03_launch.log`.
Assignment 4 (JEPA) remains unexecuted; the prepared heavy routing replay remains
unexecuted until the batch completes.

Assignment 3's owner exited zero after persistence; evaluation session 72588
exited zero. **Verified round trip in 332.88 simulated seconds, zero disallowed
contacts and no pipeline faults.** Maximum physical goal/home dwell distances
were 5.38/9.05 mm; maximum 100-ms speeds were 0.01118/0.01924 m/s. Both
one-second quiet-arrival checks passed. All eleven return corridor edges
reversed outbound edges, with zero invalid graph transitions; twelve unique
outbound edges were traversed.

There were 3326 camera pairs and 820 selections: 752 on time and 68 late
(91.71% on time). Live wall time before archival was 437.93 s. Actions were
11 holds, 174 left turns, 178 right turns, 134 right arcs, 102 left arcs and
221 forward plans. All seven recorded coverage-view completions observed their
requested patches; there were no fresh-view failures and nine translation-
coverage rejections. Both supervised repetitions succeeded. The unfinished
batch currently has two round trips and one goal-only timeout; the final JEPA
assignment is still required. The output drive had 9.5 GiB free after archival.

Assignment 4 (JEPA) launched after assignment 3's owner and evaluator exited,
in session 55193, owner PID 4142483, verified live. Its launch receipt confirms
the unchanged polygon mapper and JEPA readout state
`f372e75c1a5c4b3933beb9d59ee97158ce17a8a2b567a89c9be59b74cf8112a8`.
Outer log: `/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1/go2_polygon_floor_repeatability_04_launch.log`.
After this final assignment's owner and evaluator finish, summarize the complete
batch and run the prepared routing replay. No additional native repetition is
part of this batch.

## Completed batch

Assignment 4's owner exited one after saving the failed recording; evaluation
session 24641 exited zero. **Verified goal arrival at camera frame 2637, no
home arrival, zero disallowed contacts, then visual tracking failure after
4214 acquired camera pairs.** The physical goal dwell's maximum distance was
10.81 mm and maximum 100-ms speed was 0.01527 m/s; the one-second quiet-arrival
checks passed. The evaluator has no normal terminal simulation time for this
faulted run. Eleven unique outbound corridor edges were traversed, but no
return corridor edge was crossed. Preserve the full failed recording.

There were 1047 selections, 973 on time and 74 late (92.93% on time). Actions
were 12 holds, 401 left turns, 371 right turns, 64 right arcs, 70 left arcs and
129 forward plans. All 393 selections after the goal were turns or holds:
206 left turns, 183 right turns and four holds. Return routing alternated
between an observed-floor goal route (213 plans) and low-visual-support
recovery (180). At the last plan, translations failed predicted obstacle
clearance, the preferred right turn failed the 0.48-m reserve requirement
(0.47550 m predicted), and the selected left turn passed (0.48204 m).
These recorded decisions identify a turning/recovery stall before the eventual
tracking loss; they do not prove which intervention would complete navigation.

| Fixed assignment | Model | Verified goal | Verified home | Outcome |
| --- | --- | --- | --- | --- |
| 1 | JEPA | Yes | No | Budget exhausted at 480.98 s |
| 2 | Supervised | Yes | Yes | Round trip, 349.66 s |
| 3 | Supervised | Yes | Yes | Round trip, 332.88 s |
| 4 | JEPA | Yes | No | Visual tracking failure |

The fixed batch therefore has **four goals, two round trips, zero contacts,
and one tracking failure**. Supervised completed 2/2 round trips; JEPA completed
0/2. The separate successful JEPA pilot remains excluded. This small exposed-
layout comparison does not establish generalization or a causal training
effect, and offers no evidence of a JEPA navigation advantage. The higher
on-time fraction in assignment 4 also prevents attributing both JEPA failures
simply to excessive routing time.

The planned saved-map routing diagnostic launched in session 6704 after the
last owner and evaluator exited. After that diagnostic, inspect the return
turning/recovery interaction as a separate navigation failure mechanism.

### Post-batch diagnosis

The routing replay completed (session 6704, exit zero): 731 actual map updates
were replayed from delivered noisy depth and recorded estimated poses in
97.29 s. Retained floor/fine-obstacle counts matched all four selected plans.
At frames 2372/2644, cold fine-goal queries took 226.22/226.69 ms and found no
route; at 2920/2924 they took 81.18/80.25 ms and found a route. Warm cached
queries took 2.88–2.98 ms. Coarse proposals took 17.71–18.18 ms. These are
subproblem timings, not a reproduction of full controller/frontier state.
Profiles attribute most cold-search time to repeated general segment/square
distance calculations on axis-aligned floor-graph edges.

`lewm/axis_aligned_fine_connectivity_development.py` adds a compiled exact
interval-gap distance for those edges, keeping the original A* implementation,
costs, tie-breaking and continuous endpoint checks. It has not been connected
to a native mission. Two focused tests passed, including general-geometry
comparisons, edge touching, empty geometry, reversed segments, fallback for
diagonal segments and the 0.45-m clearance threshold. The saved-query verifier
completed (session 76953, exit zero), preserving all four query results exactly
apart from timing. With caches reset in both execution orders, original query
times were 85.14–230.18 ms and accelerated times were 30.67–62.54 ms (roughly
2.8–3.7 times faster). The receipt is assignment 1's
`fine_goal_routing_profile_v1/axis_geometry_verification_v1.json`.

The recorded return-decision diagnostic
`scripts/read_go2_polygon_return_stall_development.py` completed in session
62462. Its initial execution failed only at JSON serialization of a NumPy
integer; no output was written. The corrected execution saved assignment 4's
`return_stall_readout_v1.json`. Assignment 4 applied **zero translating command
samples** after arrival and reversed applied turn direction 127 times. Estimated
position stayed within 0.1392 m of the arrival position and unwrapped heading
spanned only 1.981 radians. There were 61 alternative-turn latch events and
34 transitions from an active route-turn latch into visual recovery. None of
the other three returns had that transition. All 74 preferred-turn reserve
rejections in assignment 4 still passed the nominal footprint test; this does
not justify removing the prediction-error reserve. In contrast, assignment 1
made substantial return progress and exhausted its remaining budget.

Next: inspect observed-map forecast clearance at selected assignment-4 turn
conflicts, separating learned forecast drift from view-recovery constraints.
The accelerated routing is verified on saved inputs only; a future native
comparison must explicitly include and warm up that source. No new native
repetition, training, storage deletion or hardware execution occurred during
these post-batch diagnostics.

Subsequent work: the forecast-clearance replay and single prospective route-turn
memory pilot are recorded in `docs/go2_interrupted_route_turn_memory_2026-09-17.md`.
The pilot retains this batch's original routing implementation; the saved-input
axis-distance acceleration remains a separate, not-yet-live-validated change.

### Recorded executed-window forecast scores

The mission evaluators also saved these 700-ms scores, restricted to selected
actions whose full requested command sequence matched the forecast. Each row
compares forecasts on the same windows within that mission; rows have different
trajectories and are not a matched cross-model prediction benchmark.

| Assignment | Windows | Applied neural XY RMSE, mm | Pose-command XY, mm | Command-history XY, mm | Neural yaw RMSE, degrees | Command-history yaw, degrees |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1, JEPA | 906 | 9.256 | 7.096 | 9.070 | 1.058 | 0.858 |
| 2, supervised | 681 | 9.071 | 7.321 | 8.621 | 0.803 | 0.696 |
| 3, supervised | 714 | 8.450 | 6.756 | 7.669 | 0.858 | 0.740 |
| 4, JEPA | 882 | 7.928 | 5.942 | 6.648 | 1.320 | 1.022 |

The fitted pose-command forecast has lower planar error and command history
has lower yaw error than the neural model on each recorded mission. Numbers
of matched windows with any neural path error over 30 mm were 1, 5, 1 and 0.
Overlapping windows are not independent samples, and these selected-action
scores do not verify the unexecuted alternative turns or certify the reserve.
Sources are each root's `saved_short_pulse_same_window_xy_v1.json`,
`saved_short_pulse_yaw_evaluation_v1.json`, and
`saved_executed_motion_forecast_evaluation_v1.json`.
