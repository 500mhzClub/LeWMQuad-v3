# Perception-navigation runtime contract — 2026-06-13

## Claim boundary

The current topological-navigation result is perception-driven for place
recognition, localization, route selection, goal matching, and perceptual stop.
It is **not yet a pure-perception runtime result** because the verified v43
seek-time collision avoidance uses an inflated occupancy grid built from the
privileged scene manifest.

There are two distinct deployment commitments:

- no privileged scene geometry or simulator ground-truth pose at runtime;
- the current platform sensor contract is monocular RGB plus onboard
  proprioception/state estimation, not rendered depth.

An ego-depth experiment can satisfy the first geometry-source requirement when
paired with onboard odometry, but it broadens the sensor contract. The current
benchmark's rendered-depth plus simulator-pose wiring satisfies neither full
deployment commitment.

Privileged geometry remains acceptable for:

- mapping-tour construction in this benchmark;
- kinematic simulation and collision enforcement;
- offline labels, diagnostics, and oracle comparisons;
- evaluation-only success and distance metrics.

Privileged geometry is not acceptable for a deployment-valid seek policy's
runtime local-obstacle queries.

## Enforced code boundary

Runtime seek safety now depends on:

`lewm.planning.local_obstacles.LocalObstacleModel`

Available adapters:

- `PrivilegedGridObstacleModel`: wraps the manifest grid and reports
  `deployment_valid=False`;
- `PerceptionObstacleModel`: wraps an onboard perception/local-occupancy
  provider and reports `deployment_valid=True`;
- `DepthLocalObstacleModel`: builds conservative rolling occupancy from depth
  and odometry. Its deployment flag is inherited from the concrete sensor and
  odometry sources.

`scripts/benchmark_topo_nav_e2e.py` keeps its privileged route-construction grid
separate from the runtime local-obstacle model. The following runtime decisions
consume only the local-obstacle interface:

- primitive feasibility and padded-clearance probes;
- latent-servo candidate veto;
- WALK steering and veto escape;
- final goal completion;
- fall-recovery free-point search.

Each trial writes `local_obstacle_contract` into its result artifact and prints
the runtime source at startup.

## Strict deployment-valid mode

Use:

```bash
scripts/benchmark_topo_nav_e2e.py \
  --require-deployment-valid-local-obstacles \
  ...
```

The benchmark rejects both built-in sources in strict mode:

- `privileged-grid` uses manifest geometry;
- `ego-depth` currently uses simulator-rendered depth registered with simulator
  ground-truth pose.

A provider backed by an accepted onboard sensor and onboard odometry must be
injected before a result can claim deployment-valid local collision avoidance.

The guard does not make the benchmark's privileged mapping tour deployable.
It also does not certify the complete autonomous seek stack: the benchmark
currently uses simulator pose as an odometry/proprioception stand-in for
heading control, predicted-trajectory placement, stuck detection, and
evaluation. That pose dependency must be replaced by or validated against
onboard state estimation before claiming a fully deployment-valid seek.

The strict local-obstacle guard is therefore necessary but not sufficient for a
pure-perception navigation claim.

## Current status

The v43 result predates this boundary and used privileged manifest-grid runtime
safety. It remains a valid topological-navigation and controller diagnostic,
but not a pure-perception navigation result.

An experimental ego-depth provider now exists and all seek-time obstacle
queries can run through it without consulting the manifest grid. A 120-block
kinematic experiment
(`.generated/topo_nav/ego_depth_sim_pose_seek120.json`) reached
evaluation-success distance with:

- `0.44 m` progress and `0.61 m` final distance;
- six `forward_medium` blocks, zero falls;
- no perceptual stop;
- unknown space blocked conservatively.

This validates the local-obstacle interface and depth occupancy behavior, not a
deployment claim. The experiment still used simulator-rendered depth and
simulator ground-truth pose. The strict guard now rejects that composition.

To preserve the current monocular-RGB commitment, the next provider must derive
local free space from RGB/LeWM features and onboard egomotion. Alternatively,
adding a real depth sensor requires an explicit platform-contract change.

### Driving the v43 maze with ego-depth (navigation, not provenance)

A separate effort tried to make the ego-depth source actually *navigate* the v43
maze (provenance held invalid). It found and fixed two occupancy bugs (a camera
FOV-axis error that marked the floor as walls; missing occupancy-clearing that
let scan rotations smear phantom walls into corridors), advancing subgoal
progress 0 → 2 of 7 hops with the local map now matching the privileged grid.
Ego-depth still does not finish the maze: the residual blocker is the seek-loop
controller's give-up/replan dynamics (tuned for the exact privileged grid), not
the occupancy model. Full write-up:
`docs/lewm_perception_depth_occupancy_diagnosis_2026-06-13.md`.
