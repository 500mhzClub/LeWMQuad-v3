# Correct projected floor coverage, then test navigation

The completed four-arm frozen-readout comparison produced no arrivals in any
arm. Saved depth explains a common obstruction: the mapper tested every pixel
quad in the axis-aligned rectangle enclosing a projected floor cell, including
corners outside the actual floor square. Raised geometry in those extra corners
could keep an otherwise observed square unknown. Every arm's first failed
view of [40,-14] has this property. No JEPA benefit is established.

## Recorded evidence

`scripts/probe_go2_frozen_readout_floor_views_development.py` reconstructed the
delivered noisy public depth and recorded estimated poses for 16 selected cell
observations across the four retained failures. It reproduced every old
classification, using frame zero to reconstruct the map basis and fixed floor
height. It did not replay the full historical map or use native pose/wall data.

Ten selected observations were rejected. In nine, every pixel quad overlapping
the projected square passed the unchanged validity and 10-mm height checks;
only the surrounding rectangle failed. At the first [40,-14] rejection, the
overlapping quads numbered 2769, 2705, 2948 and 2562 in the four arms, with zero
rejections in each. Outside-square rejected quads numbered 136, 157, 122 and 51.
The one remaining failed observation, command-history frame 2516, contained
172 rejected overlapping quads and must stay unknown. This targeted sample is
not an estimated error rate for all map cells.

Result: `/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1/go2_frozen_readout_floor_view_probe_v1.json`.

## Implemented correction and checks

`lewm/projected_polygon_floor_coverage_development.py` adds an alternative
current-plane mapper. Every image quad intersecting the projected convex floor
square, including boundary contact, must pass the original raw validity and
height-band checks. The fully projected visibility requirement remains. The
unavailable-plane fallback is unchanged; original obstacle sensing and mapping
are unchanged. Previously accepted rectangle coverage remains accepted.

Three focused tests passed in 1.99 s: geometric intersection including touching
boundaries and reversed orientation; compiled predicate agreement with the
independent vectorized predicate at every tested bad-quad location; and a
synthetic depth-plane case that removes an outside-only defect while rejecting
an inside obstacle and inside missing depth.

`scripts/verify_go2_polygon_floor_coverage_development.py` then compared old/new
maps on all 16 queries across 18 updates including initialization. Every query
matched the independent pixel diagnosis; all old floor cells were retained;
map basis, floor height, coarse obstacles and fine obstacles matched exactly.
This is selected current-frame evidence, not a full history replay or a proven
navigation improvement.

The first vectorized per-rectangle implementation took roughly 240–317 ms per
map update; its result remains in `go2_polygon_floor_coverage_saved_views_v1.json`.
A compiled early-exit implementation retained all classifications and reduced
median noninitial map time to 78.07 ms versus 52.63 ms for the old mapper. The
compiled kernel is warmed before live mapping. The optimized comparison,
including source identities, is
`/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1/go2_polygon_floor_coverage_saved_views_v2.json`.
These sequential saved-frame timings are diagnostic, not real-time qualification.

## Prospective navigation test

Plan: `docs/go2_polygon_floor_navigation_plan_2026-09-17.json`.
Launcher/evaluator: `scripts/run_go2_polygon_floor_navigation_development.py`.
One execution is assigned on exposed layout 1 using the same JEPA readout as
the failed second arm, with state
`f372e75c1a5c4b3933beb9d59ee97158ce17a8a2b567a89c9be59b74cf8112a8`.
Only the floor-coverage geometry changes. Six actions, tracking, view planning,
footprint filtering, clearance/stopping checks, 2-mm depth noise, ideal gyro,
CPU group, 4800-tick budget and the 300-ms deadline plus 20-ms wait stay fixed.

The outcome must be independently evaluated after recording persistence; a
saved-pixel fix does not count as a goal or home arrival. Keep the full recording
and every failure. One exposed-maze execution cannot establish repeatability,
fresh-layout generalization, a JEPA contribution or hardware readiness.

Launched in session 29165, owner PID 4135962, verified live. The launch receipt
confirms one assigned mission, `ProjectedPolygonFloorRoutingMap`, unchanged JEPA
readout state and `world_model_changed=false`. Output root:
`/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1/go2_polygon_floor_jepa_readout_noise_2mm_native_layout01_4800_v1_attempt_001`.
Outer log: `/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1/go2_polygon_floor_navigation_launch.log`.
Detailed progress is in the root's `worker.log`. After the owner exits, run the
same launcher with `--evaluate`; retain its terminal result before any next run.

## Completed native outcome

The owner exited zero after complete recording persistence. Evaluation session
13221 exited zero: **verified goal-and-home round trip in 289.04 simulated
seconds, zero disallowed contacts, no pipeline faults**. Maximum physical
distance during the one-second goal/home dwells was 17.76/14.65 mm, within the
40-mm physical acceptance radius. Maximum 100-ms speeds during those dwells
were 0.0100/0.0189 m/s; all requested dwell intervals were zero. The evaluator
reported 2.46-mm median and 7.09-mm maximum registered-position error.

All eleven unique return corridor edges reversed outbound edges, with zero
invalid graph transitions. This establishes physical backtracking in this
mission, not a causal advantage from persistent memory.

There were 2888 camera pairs and 713 plans: 610 on time, 103 late (85.55%
on time). Live wall time was 384.29 s before archival. Actions were 38 holds,
127 left turns, 87 right turns, 108 right arcs, 112 left arcs and 241 forward
plans. Eleven requested coverage patches were observed, with no recorded
fresh-view failures. There were 23 translation-coverage rejections. No explicit
coverage-view request for [40,-14] was recorded. That absence alone does not
identify the first mapped observation of this cell.

The preceding same-readout run exhausted 481.30 simulated seconds without any
arrival, with 998 holds and a persistent [40,-14] coverage request. Together
with the saved-pixel reconstruction, this supports correcting the enclosing
rectangle bug. The two live executions followed different asynchronous
trajectories; they do not establish a repeatable causal navigation effect or
JEPA benefit. The current sources and model remain frozen for this completed
experiment. Keep its full depth as the current successful geometry reference,
and keep all four preceding failure recordings.

Authoritative result: output root's `polygon_floor_navigation_readout_v1.json`,
with arrival details in `continuous_native_arrival_evaluation.json`. Native
state and maze graph were used only by independent evaluation. No hardware
trial has run, gyro remains ideal, and neither full-loop real-time operation
nor realistic sensor uncertainty is established.

Next scientific step: a fixed repeatability comparison of JEPA and supervised
readouts with this geometry, preserving every outcome and avoiding changes
between runs. Then assess transfer to new development layouts. No repeatability
batch has been prepared or launched yet; the one-mission experiment is complete.
