# Fixed fresh-maze model comparison

The interrupted route-turn memory pilot completed an exercised, physically
verified JEPA round trip on the exposed maze in 353.66 simulated seconds. The
next comparison fixes **JEPA then supervised**, once each, on newly generated
development layout 1. Both use the same route-turn memory, polygon floor mapper,
six actions, original frozen fitted readouts, sensors and execution checks.
No outcome from either run changes the second assignment or controller.

Generation seed 2026091751 produced two inventory layouts after examining three
candidates. The existing generator rejected one topology already in the explicit
98-layout development registry; both accepted topologies/embeddings are disjoint
from that registry. Layout 1 is fixed for this pair; layout 0 remains unexecuted.
This is a prospective same-family development comparison, not final evaluation
or a test of other environment types. Full maze geometry stays in the physical
session and evaluator; online control receives the ordinary public mission and
sensor observations.

Both arms also use the exact axis-aligned fine-graph distance acceleration.
It preserved the four saved routing-query results and passed the two focused
geometry tests; it is warmed up before native execution. This is its first live
use, so report actual planning deadlines. The paired treatments share this change;
comparisons to the preceding exposed-maze pilot cannot isolate its navigation
effect. No causal memory ablation is included in this two-run model comparison.

Plan: `docs/go2_route_turn_memory_transfer_plan_2026-09-17.json`.
Inventory: `docs/go2_route_turn_memory_transfer_layout_inventory_2026-09-17.json`.
Launcher/evaluator: `scripts/run_go2_route_turn_memory_transfer_development.py`.
Preparation session 55808 exited zero. Each mission has a 4800-tick budget,
2-mm depth noise, ideal gyro, CPU group 8–15/24–31, 300-ms planning deadline
and 20-ms extra wait. No heavy work runs alongside a timed native mission.
The dedicated output drive had 7.78 GiB available after retiring only depth
from two superseded successes. Preserve both new recordings through analysis
and every failure. Evaluate assignment 1 after its owner exits, then run and
evaluate assignment 2 even if assignment 1 fails scientifically.

Root pattern:
`/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1/go2_route_turn_memory_transfer_{number:02d}_{arm}_noise_2mm_native_layout01_4800_v1_attempt_001`.

Primary outcome: independently verified goal-and-home arrival without disallowed
contact. Retain goal-only outcomes, tracking failures, physical backtracking,
planning deadlines, exercised turn-memory decisions and matched executed-window
forecast scores. One fresh maze, one execution per model and one training seed
cannot establish reliable generalization, JEPA advantage or hardware readiness.

Assignment 1 (JEPA) launched in session 77398, owner PID 4149909, verified live.
Launch metadata confirms `TransferRuntime`, `ProjectedPolygonFloorRoutingMap`,
axis acceleration enabled and fresh scene
`route-turn-memory-transfer-development-v1-01`. The frozen JEPA readout remains
`f372e75c1a5c4b3933beb9d59ee97158ce17a8a2b567a89c9be59b74cf8112a8`.
Outer log: `/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1/go2_route_turn_memory_transfer_01_launch.log`.
Detailed progress is in the mission root's `worker.log`. No outcome yet;
assignment 2 (supervised) remains pending.

Use `scripts/evaluate_go2_route_turn_memory_transfer_development.py --assignment N`
for evaluation. This small adapter uses the same physical/forecast evaluator
while correcting the inherited exposed-layout descriptive flag to fresh
development layout. It does not change the frozen native runner or either
treatment. The native launch already records the new inventory correctly.

Assignment 1's owner exited one after tracking failure and full persistence.
Evaluation session 97310 exited zero: **no arrivals, no disallowed contacts,
tracking failure after 1272 acquired camera pairs**. There were 316 selections,
308 on time and eight late (97.47%). The new memory appeared on 33 plans:
frames 204–296 and 1232–1264, all selecting right turns. The final eight plans
were on-time right turns along an observed-floor frontier route. The final
recorded planning feature witnesses were [39,59], [53,82], [66,49] at frames
1256/1260/1264, without active visual recovery; feature count remains an
uncalibrated warning heuristic. Preserve the complete failure and diagnose its
actual tracking rejection after the fixed pair finishes.

Assignment 2 (supervised) launched unchanged in session 48335, owner PID
4150774, verified live. Outer log:
`/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1/go2_route_turn_memory_transfer_02_launch.log`.
The volume had 7.0 GiB available after assignment 1's shorter failed recording.
No source/model change or additional repetition was inserted. The prepared
post-pair replay, `scripts/replay_go2_route_turn_transfer_tracking_failure_development.py`,
reuses the original exact-tracker diagnostic and requires assignment 2's
evaluated result before executing. No heavy replay ran alongside either mission.

## Completed fixed pair

Assignment 2's owner exited zero after persistence. Evaluation session 17452
exited zero: **verified goal-and-home round trip in 298.08 simulated seconds,
zero disallowed contacts and no pipeline faults**. Goal/home frames were
2460/2976. Maximum physical distances throughout their one-second quiet dwells
were 9.48/8.40 mm; maximum 100-ms speeds were 0.01922/0.00924 m/s. Both arrival
checks passed. All seven unique return corridor edges reversed outbound edges,
with zero invalid graph transitions; eleven unique outbound edges were visited.

There were 2978 camera pairs and 736 selections: 710 on time and 26 late
(96.47%). Live wall time before archival was 388.17 s. Actions were 49 holds,
176 left turns, 199 right turns, 95 right arcs, 102 left arcs and 115 forward
plans. Route-turn memory was exercised on 39 plans between frames 204 and 1616,
all on time. Preserve the complete success and failed counterpart through analysis.

| Treatment | Goal | Home | Contacts | Outcome | On-time plans |
| --- | --- | --- | ---: | --- | ---: |
| JEPA | No | No | 0 | Tracking failure, 1272 camera frames | 308/316 |
| Supervised | Yes | Yes | 0 | Round trip, 298.08 s | 710/736 |

The fixed pair is **one round trip out of two missions on one fresh maze**.
It adds verified independent-layout supervised world-model navigation evidence;
the corresponding JEPA run failed. It establishes neither reliable
generalization nor a causal JEPA advantage. Lower routing cost did not remove
tracking vulnerability. The pair had no memory-off or reactive treatment, and
one training seed; earlier distinct comparison populations remain separate.

On each mission's matched executed 700-ms windows, neural versus pose-command
XY RMSE was 7.807/6.235 mm (JEPA) and 8.703/6.949 mm (supervised). Supervised yaw
RMSE was 0.852 degrees versus 0.716 for command history. These are trajectory-
conditional forecast scores, not alternative-policy outcomes.

The unchanged tracking replay launched after both owners and evaluators exited,
in session 65870. Its stdout is assignment 1's
`tracking_failure_replay_stdout.log`; the detailed result will be
`tracking_failure_replay_v1/result.json`. No additional native attempt belongs
to this pair.

## Exact failure replay and image-pair diagnosis

Replay session 65870 exited zero in 123.44 s. All 1267 recorded accepted raw
poses matched array-exactly, including mode and selected reference. Failure
recurred at frame 1267. Seven of the eight recent reference attempts lacked
rigid-pose matches; the latest reference's pooled attempt failed the requirement
for at least three inliers per camera. No old view was eligible, and diagnostic
fits from all eight stored views in both cameras also failed. No plane conflict
was recorded. The failed tracker was not restarted.

`scripts/probe_go2_transfer_failure_correspondences_development.py` inspected
the delivered noisy image pairs 1265→1266 and 1266→1267, with the original
150-corner selector and its existing 600-corner predecessor. The last pair's
original descriptor matcher yielded only 2+6 primary/auxiliary matches. Direct
flow yielded 2+13, with 2+11 pooled inliers: enough total support for the core
fit, but not the per-camera three-inlier requirement. With the larger per-cell
budget, direct flow yielded 2+14 and 2+13 pooled inliers. The pooled fit still
failed the unchanged per-camera rule; it was not admitted.

The separate `failure_pair_depth_gate_probe_v1.json` confirms that the
uncapped sparse auxiliary population supports a **single-camera** core fit
with twelve inliers under the existing thresholds. The capped auxiliary fit
fails after pruning. This was true with either raw or locally estimated depth.
The original primary direct-flow population has six raw-valid correspondences,
reduced to two by the existing local depth estimator. No raw-depth fallback or
admission-threshold relaxation has been implemented. Neither pair-level probe
ran floor refinement or temporal admission, and neither proves a valid full
tracker pose or alternative navigation outcome.

`lewm/sparse_feature_budget_tracking_development.py` now retains all measured
corners only when their liftable population already fits within 150. Dense
images keep the old spatial selection; sparse images no longer discard useful
corners solely because one image region has thirteen. The 150-feature total
cap, matching, floor refinement, pose/temporal gates and sensor packets remain
unchanged. The complete saved-sequence test launched in session 94276 using
`scripts/replay_go2_sparse_feature_budget_development.py`; output is assignment
1's `sparse_feature_budget_replay_v1/`. It does not execute a new navigation
trajectory. Assess its actual result and pose accuracy before any native trial.

The sparse-feature replay completed in 128.51 s: **it still fails at frame
1267**, with 1267 accepted frames and no extension beyond the original tracker.
The latest-reference attempt now fails the downstream gyro-consensus support
check. The image-pair improvement therefore does not survive the full tracker.
This variant is not promoted into native navigation.

Independent physics-only evaluation of the common 1267 poses gives XY RMSE
3.956 mm for the original and 3.734 mm for sparse retention, with maximum
errors 7.422/7.198 mm. Rotation maxima remain 0.05223 degrees. Physics was used
only after replay for scoring. The modest trajectory-specific accuracy change
does not establish a robustness or navigation improvement.

The next diagnosis measures the accepted correspondence support on both fixed
missions using `scripts/replay_go2_transfer_registration_support_development.py`.
It checks exact reproduction of every recorded accepted pose and records the
selected fit's existing camera-time witness. The original feature-count warning
triggered at frame 1265 and was published at 128.162 s, only 38 ms before the
failed frame's 128.200-s acquisition. Several earlier accepted fits had only
12–15 inliers while selected-feature counts remained above the warning level.
Whether a correspondence-based warning is useful requires examining the
successful counterpart too; no controller change follows from counts alone.

For this diagnostic, inspect one fixed correspondence-warning hypothesis:
weak below 18 accepted inliers (1.5 times the existing 12-match minimum),
strong at 24 or more (twice that minimum), retaining the 48/96 feature
thresholds and 0.20-m reference locality. These are uncalibrated development
margins, not fitted confidence limits. Replay the existing feature-only state
for comparison, then examine the hypothetical warning on the recorded paths;
neither path may be described as an executed alternative policy. No threshold
sweep or native run is part of this diagnostic.

Both unchanged support replays completed: JEPA matched all 1267 accepted poses
in 126.04 s and reproduced the failure; supervised matched all 2978 poses in
297.97 s with no failure. Independent CPU groups allowed these untimed sensor
replays to run together; no native mission ran alongside them.

The simple inlier-warning hypothesis is **not selected for a native trial**.
Below-18 support occurs on 416/1266 noninitial JEPA frames (32.9%) and 611/2977
supervised frames (20.5%), despite the latter completing its round trip.
`scripts/probe_go2_registration_support_warning_development.py` reproduced all
316/736 existing planning recovery states. Its hypothetical state first differs
at frame 15 in both missions, far before the terminal failure. Applied to the
unchanged recorded paths, it has 13/40 onsets versus 5/6 originally and remains
active for 1164/2374 frames versus 34/48. Those later hypothetical states are
not executed alternative trajectories; they cannot predict an actual recovery
duty cycle or prove failure. They do show that an inlier-count margin alone
is insufficient evidence for the intended narrowly targeted intervention.

The next saved-sensor test fills spare feature slots with weaker image corners:
`lewm/sparse_corner_completion_development.py` and
`scripts/replay_go2_sparse_corner_completion_development.py`. Sparse images
retain their stronger corners, then add depth-valid corners at quality 0.001
up to the unchanged total of 150. Dense images keep their original selection.
Matching, local depth estimation, floor refinement, gyro consensus and pose
admission thresholds are unchanged. This tests whether currently discarded
image evidence can resolve the failure; it is not yet a navigation treatment.

That completion replay accepted all 1272 saved frames in 134.42 s, including
five after the original failure, with common-frame XY RMSE 3.979 mm (original
3.956 mm) and extension XY RMSE 3.782 mm. It now proceeds to one separately
recorded exposed-maze navigation pilot, documented in
`docs/go2_sparse_corner_completion_2026-09-17.md`. The original transfer pair's
outcomes remain unchanged. The pilot preserves the strong-corner recovery
signal separately from the expanded tracking population.
