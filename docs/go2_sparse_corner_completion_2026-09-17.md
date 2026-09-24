# Sparse-corner completion navigation pilot

The original fresh-maze JEPA mission failed tracking at frame 1267; its matched
supervised counterpart completed a round trip. Retaining uncapped strong
corners alone did not resolve the failure. Filling spare sparse-view slots
with weaker measured corners accepted all 1272 saved sensor frames, including
five beyond the old failure. All matching, depth, floor, gyro-consensus and
temporal pose gates remain unchanged, with at most 150 features per camera.

On the 1267 common frames, original/completed XY RMSE is 3.956/3.979 mm;
maximum error is 7.422/7.386 mm. The five recovered frames have 3.782-mm XY
RMSE and 3.836-mm maximum. Physics was used only by the independent evaluator
after sensor-only replay. This establishes a local replay improvement, not
navigation reliability. Replay took 134.42 s with 55.04-ms median tracker time.

Feature witnesses retain the original detector's strong-threshold
`detected_features` and `liftable_features`; `completion_detected_features`
and `added_weaker_corners` describe the added detection pass. `selected_features`
and `selected_per_cell` describe the completed tracking population. They must
not be interpreted as the population of the original strong detector.

The live integration explicitly separates those populations: visual recovery
uses `original_selected_count` at the unchanged 48/96 thresholds, while
tracking uses the completed features. All 1267 recorded strong-count pairs
matched the original replay exactly. The focused registration integration test
passed and verifies that many faint features do not suppress the old weak-view
trigger or prematurely release recovery. Original pose/map/dispatch checks
remain in place.

Run exactly one JEPA mission on transfer layout 1, now an exposed development
maze. Use the same frozen readout, route-turn memory, polygon floor map, axis
routing acceleration, six candidates, 4800-tick budget, 2-mm depth noise, ideal
gyro, CPU group 8–15/24–31 and 300-ms planning deadline plus 20-ms extra wait.
No extra repetitions to obtain a success. Preserve its full recording, including
any failure. Primary outcome is independently verified goal-and-home arrival
without disallowed contact; also inspect tracking, recovery and deadlines.

Launcher/evaluator: `scripts/run_go2_sparse_corner_completion_development.py`.
Plan: `docs/go2_sparse_corner_completion_plan_2026-09-17.json`.
Root: `/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1/go2_sparse_corner_completion_jepa_noise_2mm_native_layout01_4800_v1_attempt_001`.

This is measured simulation with physics paused during computation. It does
not establish realistic-gyro behavior, wall-clock real-time execution, hardware
readiness, JEPA advantage or independent-layout reliability. No other heavy
job runs alongside the native mission.

The native pilot launched in session 28803, owner PID 4156858. Its live
launch records `CompletionRuntime`; the pose worker records
`SparseCornerCompletionMotion` / `SparseCornerCompletionPose` with the unchanged
four-frame old-view revisit cadence. Original strong-corner recovery counts
are explicitly enabled. The drive had 4.8 GiB free before launch. Outer log:
`/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1/go2_sparse_corner_completion_launch.log`.

## Completed outcome

Owner session 28803 exited zero after full persistence. Physical/model evaluator
session 63451 exited zero: **verified goal-and-home round trip in 255.42
simulated seconds, zero contacts, zero pipeline faults**. All 2549 camera poses
were retained. Goal/home arrival frames were 1321/2547; maximum physical
distances across the one-second quiet dwells were 16.91/17.38 mm. Maximum
100-ms speeds during those dwells were 0.01448/0.02631 m/s. Both checks passed.
Median/max registered position errors over the recording were 2.15/8.18 mm.

The corridor evaluator initially consumed stale `fresh_layout_inventory`
metadata inherited by the new launch writer from an older reference. This
produced six spurious invalid graph transitions. The actual physical session
used the intended transfer maze: all 25 native static-object records match
the completed transfer mission exactly, and the frozen prospective plan binds
the transfer inventory SHA-256
`ad3d628990526ec90adc7c627cbd582314a2ff71da4e7b25c4d30177d1ff6036`.
`scripts/evaluate_go2_sparse_corner_completion_development.py` corrects only
that graph input, preserving launch and V1 results. Session 32183 exited zero.
The authoritative combined readout is
`sparse_corner_completion_navigation_readout_v2.json`; the detailed corridor
readout is `physical_return_corridor_readout_v2.json`. All **seven unique return
edges reverse outbound edges, with zero invalid transitions**. Arrival, contact
and forecast evaluation were unaffected by the metadata correction. Future
launch adapters must explicitly supply the actual inventory, rather than inherit
this pilot's stale descriptive field; do not rewrite its frozen launch/source.

There were 626 plans: 512 on time and 114 late (81.79%). Live wall time before
archival was 333.97 s. Actions: 17 holds, 152 left turns, 157 right turns,
74 left arcs, 87 right arcs and 139 forward plans. Route-turn memory was active
on 40 plans at frames 1504–1660. All 626 planning recovery receipts explicitly
used the original strong-corner counts; 317 had different completed tracking
counts. This confirms that added weak corners did not silently redefine the
visual recovery threshold.

On 497 matched executed 700-ms windows, neural/pose-command/command-history XY
RMSE was 9.188/6.218/7.526 mm; neural/command-history yaw RMSE was
0.876/0.743 degrees. These scores do not demonstrate a JEPA forecast advantage
and are not outcomes of alternative policies.

The one prescribed pilot is complete and its full recording is retained as
the current exercised tracking/navigation reference. The original JEPA failure
and all negative replays remain retained. This successful prospective run on
an exposed maze does not establish repeatable improvement or novel-layout
reliability. No new native job is running. The output drive now has about
3.1 GiB free, below the existing 4-GiB launch headroom; retire only eligible
superseded success depth before another native run, preserving failures and
current comparison/reference recordings.

Next, fix the controller/tracker treatment for an independent-layout matched
comparison (transfer inventory layout 0 has not been executed). Include the
relevant non-predictive/reactive and memory controls in the broader comparison;
do not substitute this tracking success for evidence of JEPA or memory's causal
benefit. Deadline behavior also needs reporting: 18.2% late plans in this pilot
does not establish real-time operation. The original full goal remains open.
