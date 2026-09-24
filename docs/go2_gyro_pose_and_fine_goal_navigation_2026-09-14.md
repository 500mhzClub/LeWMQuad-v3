# Gyro perception and goal connectivity development

The completed fixed-controller comparison remains 2/4 verified outbound
arrivals for learned versus 0/4 reactive, and 0/4 verified round trips for
both. The following experiments are development revisits, outside that cohort.
Earlier estimator replay results and implementation history are in
`go2_continuous_controller_comparison_2026-09-13.md`.

The first native gyro-consensus experiment,
`go2_gyro_consensus_progress_rejoining_learned_round_trip_native_layout06_4800_v1_attempt_001`,
exited 0 after the eight-minute budget. All 4,805 paired frames and registered
poses were archived. Independent evaluation confirms no arrivals and zero
disallowed contact samples. Registered position median/maximum error was
11.92/18.56 mm. Thus the improved pose survived actual navigation, but did
not yield goal-reaching on the changed trajectory.

Native minimum goal distance was 0.9447 m; final goal/home distances were
3.2816/3.9196 m, and the 10 Hz horizontal path length was 21.437 m.
1,102/1,192 plans were on time (92.4%). From frame 3432 onward the recorded
planner repeatedly reported `OBSERVED_COMPONENT_HAS_NO_FRONTIER`; the final
99 decisions comprised 94 holds and five right turns. The final turn latch
could not select a reserve-clear turn. No arrival threshold or action
clearance rule was relaxed.

A public-sensor map replay used the exact recorded mapping update frames
and registered poses to reconstruct four snapshots. Native pose was not
used. The coarse replay omitted the runtime's visited-frontier exclusions,
so its frontier status does not reproduce the recorded no-frontier status;
it nevertheless failed to route to the already observed goal. The existing
fine-cell goal fallback recovered an observed route in every snapshot:

| Frame | Weighted route length (m) | Minimum continuous clearance (m) | Added route search (ms) |
| --- | --- | --- | --- |
| 3000 | 8.150 | 0.4887 | 188 |
| 3600 | 7.252 | 0.5150 | 252 |
| 4200 | 7.238 | 0.4572 | 175 |
| 4760 | 7.101 | 0.4712 | 170 |

These paths retain the 0.45 m nominal footprint. They do not certify future
action clearance, and late searches remain subject to the existing 300 ms
planning deadline. Artifacts in the first gyro native root:
`outbound_route_connectivity_diagnostic.json`, four
`outbound_map_snapshot_<frame>.npz` files, and
`fine_goal_route_counterfactual_v1.json`.

The next native trial ran in
`go2_gyro_consensus_fine_goal_learned_round_trip_native_layout06_4800_v1_attempt_001`.
It enables the previously implemented `FineGoalRouteRuntime`, retaining
gyro-consensus perception, the learned model and motion correction, the
progress-rejoining action selector, mission, timing, and arrival requirements.
The fallback only operates when the goal cell is observed and the coarse
proposal lacks a goal route. Its source was already covered by focused
geometry tests; the four reconstructed real snapshots above supply the
current applicability check. No new recovery-latch treatment is included.

Before launch: no Python experiment owner remained, CPU 0.6%, 77.04 GiB
available RAM, 57.12 GiB free disk, both GPUs idle. Run one native owner and
four post-run archive writers. Evaluate arrivals independently after terminal
execution and archive completion. At launch there was no result yet, and no
reliable-transfer, JEPA-specific, realistic-sensor or hardware claim is made.

The first finer-routing trial terminated with exit 1 because the tracking
input queue filled. It achieved a verified outbound arrival at frame 1456:
the one-second physical dwell stayed 8.16–12.05 mm from the goal, every
requested command was zero, and maximum 100 ms speed was 0.03772 m/s.
There was no return arrival. Final native home distance was 50.67 mm.
This remains a failed round-trip attempt, with zero disallowed contacts.

All 3,310 acquired camera pairs were archived; 3,276 registered poses were
admitted. Their median/maximum position error was 6.69/9.77 mm. Native path
length was 20.436 m and 720/814 plans were on time (88.5%). The physical
arrival report and `native_navigation_summary.json` are in the trial root.

Tracking usually cost approximately 78 ms per 100 ms camera interval, but
the seven recorded tracking jobs from frame 3270 onward had a median cost
of 798 ms and maximum 854 ms. A 32-frame input backlog then filled. The
subsequent `simulation clock closed before result release` message is a
shutdown effect; the original failure was `queue.Full`. No queue enlargement,
sensor skipping or simulation-clock pause has been applied.

A one-thread replay profiled frames 3260–3279 after reconstructing the
same tracker state from the beginning. It completed all 3,280 frames without
failure and reproduced all 3,276 saved raw poses exactly, using no native pose. Output:
`gyro_tracking_profile_v1/`. Before launch: CPU 0.5%, 76.89 GiB available
RAM, 46.20 GiB free disk, GPUs idle. There is no active native owner.
An inactive exact per-frame registration memo has four focused passing tests
for result identity, mutable-result isolation, changed observations/rules,
and preserved rejection reasons. It is not connected to the tracker yet;
the completed profile instead identifies chained optical flow as the larger cost.

Across the 20 profiled frames, tracking took 8.268 seconds under cProfile.
Repeated chained association consumed 4.610 seconds, including 4,245
short-interval association calls and 8,934 OpenCV optical-flow calls. Rigid
registration consumed 2.084 seconds. The registration memo remains inactive.

The current treatment reuses exact short-interval associations in overlapping
chains. It compares camera grayscale/depth contents again on each observation,
keys links by both image versions, original seed pixels and association rules,
and bounds storage to 128 image entries and 2,048 links. Cached arrays and
receipts are isolated from caller mutation. Original chain clocks, lost-track
behavior, endpoint lifting, pose fitting and acceptance checks still execute.
It is enabled only in `ReusedFlowGyroVisualMotion`; predecessor tracking remains
available without the cache. The new memo and existing association suites pass
22 tests, including changed inputs, rules, eviction and occluded tracks.

`gyro_tracking_reused_flow_profile_v1/` replayed the same 3,280-frame prefix
with the treatment. Before launch: CPU 0.3%, available RAM 76.72 GiB, free disk
46.19 GiB, no experiment Python owner. One CPU replay runs; no native trial is
active during that replay.

The replay completed without failure and matched all 3,276 saved raw poses
exactly. The profiled 20-frame section fell from 8.268 to 4.684 seconds (43.3%
less time); chained association fell from 4.610 to 1.013 seconds. Across the
full prefix the cache served 4,452 hits and computed 897 links. Frames
3270–3279 had median/maximum profiled cost 432.7/756.1 ms, versus 980.1/1058.9
ms previously. Profiling inflates both measurements, and the remaining burst
means a native backlog fix is not yet established.

The next native attempt is
`go2_reused_flow_gyro_fine_goal_learned_round_trip_native_layout06_4800_v1_attempt_001`.
It enables this exact association reuse, retaining gyro-consensus perception,
fine-goal routing, the same learned model, action/arrival requirements and
480-second navigation budget. Queue capacity and camera cadence are unchanged.
One native owner uses the available CPU/GPU resources; four writers archive
camera packets after execution. Independent physical-arrival evaluation remains
required after completion. This is another development revisit, outside the
fixed transfer cohort.

The reused-flow native attempt exited 1 after 3,019 acquired camera pairs.
It admitted 3,016 registered poses, then tracking frame 3016 raised
`ValueError('measured visual pose unavailable')`. The main runtime's
`RuntimeError('runtime stopped')` is the consequent shutdown. This was not
another `queue.Full` failure. All camera pairs were archived, and independent
evaluation found no arrival, no disallowed contact, and registered position
median/maximum error 10.98/16.25 mm. Minimum physical goal distance was
5.608 mm; final goal/home distances were 43.45 mm/1.295 m. Path length was
15.480 m, with 680/750 plans on time (90.7%).

`terminal_approach_and_tracking_diagnostic_v1.json` records the final-approach
diagnosis. The robot first came within 10 cm by its own estimate at frame
1398. Of 401 subsequent plans, 305 were left turns, two right turns, 78 left
arcs and 16 forwards. There were 105 translation-veto viewing requests.
Terminal position priority changed 83 of its 272 eligible planning records,
so that mechanism was active but insufficient to finish the approach.

There were only 12 observed frames inside 2 cm, and the maximum completed
quiet dwell was three intervals. At the closest physical pass, frame 1567,
the observed goal distance was 15.90 mm and observed interval speed was
0.180 m/s. That observation reached the mission controller 180 ms after its
measurement. Frames 1568/1569 remained inside the observed radius while
moving; frame 1570 had already drifted out to 23.76 mm. Thus passing close
to the target did not establish a settled arrival. This supports investigating
earlier braking or finer terminal motion, without relaxing arrival criteria;
it does not prove a particular braking treatment will succeed.

Tracking median cost over the full attempt was 78 ms, maximum 680 ms, and
maximum measurement-to-completion delay 1.28 s. The last 30 jobs cost a median
77 ms and maximum 140 ms. The original profiled bottleneck is cheaper, but
the new trajectory still developed delay and ultimately lost an accepted pose.
`gyro_tracking_failure_replay_v1/` is replaying this new attempt to capture
the exact tracking rejection at frame 3016. One CPU replay runs; no native
owner remains. Before this replay: CPU 0.3%, RAM available 77.18 GiB, disk
free 35.75 GiB. Both failed native attempts and all evaluation records remain.

The failure replay completed with exact agreement on all 3,016 saved raw
poses and reproduced the rejection at frame 3016:
`gyro-conditioned rotation must compose current measured gyro increments`.
The diagnostic is numerical: maximum elementwise pose/gyro rotation difference
grew from 8.35e-13 at frame 100 to 9.99e-11 at frame 3015, just below the
1e-10 check. The gyro matrix orthogonality error was only 1.03e-13, but
repeated retained-reference compositions accumulated it. The existing
rotation tolerance is unchanged. `OrthonormalGyroVisualMotion` adds polar
re-orthogonalization after each originally validated 100-ms gyro integration,
and `orthonormal_gyro_failure_replay_v1/` is testing that treatment on all
3,019 recorded camera pairs. This changes floating-point values; the replay
reports numerical pose differences and reference-choice changes explicitly.

The separate prepared planning treatment `PredictiveArrivalHoldRuntime`
selects the existing hold action if its existing corrected forecast ends
inside the observed arrival radius at both final forecast samples and its
terminal XY speed is at most 0.05 m/s. It applies only on the exact nearby
goal route, preserves the hold clearance check and active turn recovery,
and leaves measured 3-D settling and independent physical arrival unchanged.
The new selector and prior terminal-priority tests pass nine tests.
It is not yet selected by a native launcher.

`predictive_terminal_hold_counterfactual_v1.json` found two eligible changes
in the failed run: frames 1564 and 2088, both on-time left-turn plans. At
1564 the hold forecast ended 16.31/15.44 mm from the goal with terminal
XY speed 0.0132 m/s. The actual run issued a turn at 158.20 s, then only
stopped at 158.38 s when the in-radius observation reached the mission
controller. The counterfactual establishes an actionable prediction/selection
discrepancy, not successful closed-loop arrival under a different action.

The orthonormal-gyro replay completed all 3,019 frames without failure.
It retained every saved reference choice; maximum position difference from
the 3,016 original saved poses was 1.887e-11 m, and maximum rotation-element
difference was 1.0004e-10. It therefore cleared the recorded arithmetic
failure without changing the numerical acceptance tolerance. Two focused
rotation tests also passed, including 10,000 retained-reference compositions.

The next native attempt is
`go2_orthonormal_gyro_predictive_hold_fine_goal_learned_round_trip_native_layout06_4800_v1_attempt_001`.
It combines that numerical fix with the proposed predictive arrival hold.
The learned weights, motion correction, fine-goal routing, action set,
300-ms delay, 400-ms commitment, arrival thresholds and dwell remain fixed.
Both implementation changes and their explicit flags are recorded in the
launch metadata. One native owner runs, followed by four archive writers.
Prelaunch resources: 16 physical/32 logical cores, affinity 32, 75.98 GiB
available RAM while the completing replay was live, 35.74 GiB free disk,
both GPUs idle, and no other experiment owner. This is a development
treatment on layout 6; transfer and causal training comparisons remain open.

The orthonormal-gyro/predictive-hold native attempt completed with exit 0,
archiving all 4,805 camera pairs and admitting all 4,805 registered poses.
Independent evaluation verified outbound arrival at frame 3448: the full
one-second physical dwell stayed 20.43–21.53 mm from the goal, maximum
100-ms speed was 0.01919 m/s, and every requested interval was zero. There
were no disallowed contacts. The mission exhausted its 480-second navigation
budget without a return arrival. Final physical home distance was 53.63 mm.
Registered position median/maximum error was 11.01/18.61 mm. Path length was
27.325 m; 1,111/1,192 plans were on time (93.2%). The output root contains
`continuous_native_arrival_evaluation.json` and `native_navigation_summary.json`.
An evaluator call made before camera archiving finished found missing final
metadata and wrote no result; the evaluation above ran after owner exit and
archive completion.

The arithmetic fix survived the complete native trial. Predictive arrival
holding was insufficient for reliable terminal control: it changed exactly
one committed plan (frame 1808, left turn to hold), documented in
`predictive_arrival_hold_activation_v1.json`. The forecast's final two goal
distances were 19.07/19.80 mm with terminal XY speed 0.02163 m/s. Observed
distances at those endpoints, frames 1815/1816, were 29.14/28.22 mm. The
stop slowed the robot but missed the observed arrival region; the verified
arrival happened much later. This is not evidence that predictive holding
caused the eventual arrival.

The last 100 planning records contained 76 left turns, 17 left arcs, four
forward actions and one right turn (two records had no selected action).
On return the closest observed home distance was 11.97 mm, but no quiet
interval completed. The remaining issue is small terminal corrections and
settling, not a need to widen arrival criteria. No process remains live;
free disk after archive completion is 19.40 GiB.

Next scientific treatment to implement: near the exact goal only, evaluate
100-ms translation pulses followed by zero within the existing 400-ms
planning cadence. Keep the six action types and full turn durations. The
forecaster, frozen motion-correction command features, actual command window
and known-prefix ledger must all describe the same pulse. Score the forecast
after its settling tail (existing 700-ms scoring endpoint), not just at pulse
end. This is an untested control/forecast hypothesis, and the frozen residual
may be inaccurate on shorter pulses. Existing `delayed_candidate_inputs`
already supports one-tick inputs and `ScheduledCommand` supports one-tick
windows; `ContinuousCommitmentLedger.commit` currently hardcodes four ticks,
and `FrozenMotionResidual.correct` currently builds four repeated command
intervals. Use a separate development runtime/ledger so the completed trials
retain their definitions. Preserve all current arrival, geometry and actual
prefix checks. A focused terminal-control experiment can test the hypothesis
before another full maze attempt; available storage must be assessed before
any additional large camera archive.
