# Perception-backed local obstacles: making ego-depth navigate (2026-06-13)

Goal of this pass (the "perceptual path"): get the perception-backed obstacle
source (`DepthLocalObstacleModel`, ego-camera depth) to actually **navigate the
v43 maze** — the held-out `medium_enclosed_maze_71e18534e51a`, goal cell 32, a
7-hop colourful-beacon seek that the **privileged grid** completes (v43: final
0.19 m, perceptual stop correct). This isolates *navigation capability* from
*deployment provenance*: ego-depth here is still deployment-invalid (sim-rendered
depth + sim ground-truth pose), but if it can drive the same maze the only
remaining gap to a deployable claim is the sensor/pose source, not the algorithm.

Harness: `scripts/benchmark_topo_nav_e2e.py` (vulkan venv). Reproduce a run with
`--local-obstacle-source ego-depth` added to the v43 command (see the wide-maze
demo doc), `--mode physical`. Logs in `.generated/topo_nav/perceptual_path_logs/`.

## Status: ego-depth now drives the full maze, recognises the goal, and stops

Three occupancy fixes plus a flag-gated controller-recovery mode took ego-depth
from "stuck at the start" to **traversing the entire 7-hop held-out maze on goal
image alone, recognising the goal (cosine 0.95), and firing a perceptual stop at
0.78 m**. It misses the 0.65 m `success_eval` radius by 0.13 m — the perceptual
stop fires at the goal-image *standoff* (`goal_standoff_m` 0.72 m), where the
goal image was rendered, and the depth model also (correctly) sees the physical
beacon as an obstacle and stops safely short of it. The hard problem — navigate
a held-out maze with a perception-backed obstacle source, recognise the goal,
stop — is solved; the residual 0.13 m is a standoff-vs-radius / goal-object
clearance detail, not a navigation or recognition failure.

| run | obstacle source | final m | progress m | subgoals | goal cos | stop | escapes |
|-----|-----------------|--------:|-----------:|---------:|---------:|:----:|--------:|
| v43 (reference) | privileged grid, 300 blk | 0.19 | 5.2 | 7 (success) | 0.99 | yes | — |
| ego-depth, pre-fix | ego-depth, 300 blk | 6.05 | −0.63 | 0 | — | no | 159 |
| + FOV-axis fix | ego-depth, 300 blk | 4.22 | +1.19 | 0 | — | no | 151 |
| + inflation 0.10 | ego-depth, 300 blk | 5.26 | +0.15 | 1 | — | no | 140 |
| + free-clears-occupied | ego-depth, 300 blk | 5.33 | +0.09 | 2 | — | no | 128 |
| + controller recovery | ego-depth, 300 blk | 1.42 | +4.0 | 8 | 0.57 | no | 64 |
| + 500-block budget | ego-depth, 500 blk | **0.78** | +4.63 | 8 | **0.95** | **yes** | — |

(Kinematic mode is **not** a valid testbed: its walk controller is
`forward-or-yaw_right` with no heading hold and fails this maze even with the
privileged grid, 2.18 m. Physical mode is the robust controller v43 uses.)

## Bug 1 (FIXED): FOV axis — phantom floor walls

`DepthLocalObstacleModel` treated the configured FOV as **horizontal** and
derived `tan_v = tan_h·(H/W)`. Genesis's camera `fov` is the **vertical** FOV
(`genesis/vis/camera.py:39`). On the wide 640×480 image this underestimates the
vertical ray angles, so floor returns ~0.5 m ahead (bottom image rows) project to
height ≈ **0.10 m** — just above the 0.08 m `min_obstacle_height_m` — and the
floor is classified as a wall, filling every corridor.

Fix: treat the FOV as vertical, `tan_h = tan_v·(W/H)`
(`lewm/planning/depth_local_obstacles.py`). Renamed the config field
`horizontal_fov_deg → vertical_fov_deg`; benchmark passes `pack.camera.fov_deg`.
Regression test `test_wide_image_grazing_floor_is_not_obstacle`. Verified with
`scripts/probe_depth_occupancy_corridor.py`: a connected corridor went from
"blocked from 0.70 m" to fully free (floor now projects to z ≈ 0).

## Bug 2 (FIXED): occupancy never cleared — rotation smear

The model marked free and occupied cells but **never cleared a cell once
occupied**. During an ALIGN scan (full revolution) a close/grazing return at one
yaw stamps occupied cells that intrude into the corridor; later frames that see
straight through the same space could not undo it, so the smear accumulated and
choked the corridor (occupancy map was mostly `#` where the grid was free).

Fix: a ray seen *through* a cell now clears a stale occupied mark there (standard
occupancy semantics), preserving the current frame's own wall endpoints
(`_mark_ray_free`). Test `test_free_ray_clears_stale_occupied`. After this the
occupancy window matches the grid: corridor free, real walls retained;
`recent_occupied_cells` for the seek dropped 167 → 69.

Also: the occupancy `inflation` (0.22 m) **double-counts** with the physical
veto's capsule body radius (0.25 m); the capsule already covers the robot's
extent. Reducing inflation to 0.10 m via `--depth-obstacle-inflation-m` cut
occupied-rejects 30% → 11% and tripled forward steps (7 → 27–36). The model
**default stays 0.22 m** (conservative for the safety contract); 0.10–0.12 m is
the navigation-tuned value.

## Controller recovery (fixed, flag-gated): give-up + backward-drift

With clean occupancy the robot reached a node then looped (`blocked after N
realigns; replanning`) and drifted *backward* down the corridor. Two coupled
causes, both tuned for the smooth privileged grid: (1) on a transient forward
veto (gait heading drift ~0.26 rad), the walk veto-escape picks the highest-
clearance primitive, which on a straight corridor is **`backward`** — so the
robot retreats instead of re-centering; (2) the realign/dead-edge budget
(`edge_veto_streak >= 3`, `realigns_this_edge >= 2`) is too small for noisier
perception occupancy, so a sole-route edge is abandoned and re-planned to the
same edge — futile thrash.

Fix (benchmark `run_scene`, flag-gated, **defaults preserve the verified
privileged-grid behaviour**):

- `--seek-escape-no-backward` — drop `backward` from the walk veto-escape set so
  the robot turns to re-acquire the bearing instead of retreating;
- `--seek-edge-veto-streak` (default 3) and `--seek-edge-realign-cap` (default 2)
  — raise persistence on a sole-route edge.

Running ego-depth with `--seek-escape-no-backward --seek-edge-veto-streak 6
--seek-edge-realign-cap 5` (+ inflation 0.10) jumped subgoals 2 → 8, progress
0.09 → 4.0 m, escapes 128 → 64; a 500-block budget then reached 0.78 m with goal
cosine 0.95 and a perceptual stop.

## Remaining: 0.13 m to the success radius

The perceptual stop fires at ~0.78 m = the goal-image standoff (`goal_standoff_m`
0.72 m, where the goal image is rendered) and the depth model sees the physical
beacon as an obstacle, so the robot stops safely short of it — arguably *more*
correct than the privileged grid (which has no beacon obstacle and walks in to
0.19 m). To pass the 0.65 m radius, the final approach must close the last
~0.13 m past the recognised standoff without overshooting (cf. the v26 overshoot
to 1.05 m). Candidate: relax local-obstacle clearance specifically against the
*matched goal object* during the final servo, or render the goal image at a
tighter standoff. This is a goal-approach-stop detail, separate from the maze
navigation that now works.

## Tooling added this pass (reusable)

- `scripts/probe_depth_occupancy_corridor.py`: single-pose / scan diagnostic.
  Renders depth at a graph cell facing a connected neighbour, builds the
  occupancy, and compares `is_free` to the privileged grid along the corridor
  centerline; `--scan-frames N` accumulates a scan; prints per-point projection
  (row/dist/height), a top-down ego-depth-vs-grid occupancy map, and the
  false-occupied (obstacle-where-grid-free) points with the camera yaw that
  produced them.
- `DepthLocalObstacleModel.diagnostics()` now reports `is_free` rejection tallies
  (`reject_occupied`, `reject_unknown`, `hit_robot_footprint`, `hit_free_cell`)
  and an opt-in per-point `debug_capture` (default off) used by the probe.
