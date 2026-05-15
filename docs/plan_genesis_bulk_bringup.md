# Plan: Genesis Bulk Bring-up

Date: 2026-05-15
Status: Active
Companion to: [decision_pivot_to_genesis.md](decision_pivot_to_genesis.md)

This is the implementation plan for the Genesis-first bulk-generation backend.
The decision doc explains *what* and *why*; this doc explains *how* and *in
what order*. Aligns to [fresh_retrain_data_spec.md](fresh_retrain_data_spec.md)
§4–§10, with lidar dropped.

## Cross-cutting constraints

### ROS 2 contract preserved end-to-end

The trained model must be deployable on a real Go2 through the existing LeWM
ROS 2 message contract. Implementation:

- **Schema layer:** LeWM ROS 2 message types are authoritative. `CommandBlock`,
  `ExecutedCommandBlock`, `BaseState`, `FootContacts`, `ResetEvent`,
  `EpisodeInfo`, plus standard `sensor_msgs` / `nav_msgs` types.
- **Storage layer:** per-scene `.mcap` files via `rosbag2_py` Python API, with
  sim-time clock. Drop-in for a real Go2 bag.
- **Logic layer:** LeWM ROS-node bodies are lifted into a sim-agnostic Python
  module (`lewm_genesis/lewm_contract.py`). Same semantics, callable from the
  Genesis loop. The ROS spinners remain alive for the audit oracle path and
  for real-robot deployment.
- **Deployment swap point:** the real Go2 hardware bridge replaces Genesis;
  everything above (LeWM topics, policy/world-model code) is unchanged.

Trade-off: DDS-induced timing jitter and reordering are not modeled during
bulk capture. If they matter, sample them from the audit oracle path.

### Known-good locomotion policy

Off-the-shelf is the requirement: no training step on our side before bulk
generation. The honest picture:

- **The Genesis pip package ships only the Go2 URDF**, no trained
  checkpoint. The Genesis-team `examples/locomotion/go2_train.py` is a
  training *recipe*, not a pretrained policy. Running it is not off-the-shelf.
- **The realistic off-the-shelf path is unitree_rl_gym's PPO checkpoint**
  (https://github.com/unitreerobotics/unitree_rl_gym). Trained in IsaacGym,
  deployed on real Go2 hardware. Small MLP, public weights.

Tiers:

- **Tier A (off-the-shelf):** load unitree_rl_gym's PPO checkpoint into a
  legged_gym-style policy adapter. Observation layout: body-frame base
  lin/ang velocity, projected gravity, command, joint position deltas from
  default qpos, joint velocities, previous action. Action: 12-dim joint
  position offsets from default qpos, PD-controlled. Accept some sim-to-sim
  drift between IsaacGym (source) and Genesis (target); validate via Phase 2
  gait histogram before committing to a full corpus run.
- **Tier B (if Tier A drifts unacceptably):** run Genesis's own
  `examples/locomotion/go2_train.py` to convergence. Loses the "no training"
  property but produces a Genesis-native policy.
- **Not in scope:** dropping checkpoints from non-Unitree sources
  (legged_gym variants, IsaacLab, MJX) — narrower deployment validation.

### Data spec alignment

Per [fresh_retrain_data_spec.md §5–§10](fresh_retrain_data_spec.md):

- **§5.1 per-step required:** `vision`, `cmd_nominal`, `cmd_executed`, `done`,
  `collision/contact`, `base_position_world`, `base_orientation_world`,
  `joint_position`, `joint_velocity`, `last_low_level_action`,
  `proprio_observation`, `camera_valid`. All captured per env per tick.
- **§5.1 recommended:** body-frame velocities, per-leg foot contacts,
  `command_source`, `controller_mode`, `sim_time` / `control_dt` /
  `physics_dt` / `decimation`. Captured.
- **§5.2 scene metadata:** passes through from `lewm_worlds` scene manifests
  unmodified. `scene_id`, seeds, `scene_family`, `camera_model`,
  `world_bounds`, `obstacles`, `landmarks`, `navigable_graph`.
- **§5.3 derived labels:** out of bulk scope. Computed offline from raw
  rollout + scene manifest.
- **§7 shape:** 64 streams × 800–1200 raw steps per scene at full target.
  `n_envs=64` in `scene.build(n_envs=...)`.
- **§7.3 compute-cut policy:** reduce envs to 32 before reducing scene count.
- **§8 distribution:** consume what `lewm_worlds` produces. Missing families
  (local composite motifs, rough/local dynamics, visual/sensor stress) are
  scene-corpus work in parallel; do not block bulk on them.

Lidar is dropped from the corpus (`/velodyne_points`,
`/unitree_lidar/points`). URDF retains mount bodies for collision only.

## Phase 1 work breakdown

Each item has an exit criterion. Ordered by dependency.

### 1.1 Install Genesis and train Tier A policy

- Install Genesis with ROCm/AMDGPU support on the production GPU
  (Radeon AI Pro 9700, 32 GB).
- Run `examples/locomotion/go2_train.py` (or the current Genesis equivalent)
  to convergence.
- Freeze the resulting checkpoint under
  `models/tier_a_go2_locomotion/<date>.pt`.

**Exit:** checkpoint produces a walking gait for ≥10 s under constant
`(vx=0.3, vy=0, yaw_rate=0)` without falling.

### 1.2 `lewm_genesis/lewm_contract.py`

Port the LeWM ROS-node bodies into a sim-agnostic Python module.

- Command-block expansion (`vx`/`vy`/`yaw_rate` vectors from primitive +
  block_size).
- Safety clipping (absolute + per-tick delta) against the platform manifest.
- `ExecutedCommandBlock` reconstruction with `clipped` /
  `safety_overridden` / `controller_mode` flags.
- `BaseState` computation: quat → rpy, body-frame ↔ world-frame velocity
  rotation.
- Foot contacts: Genesis contact API → `FootContacts` with `fl/fr/rl/rr`
  ordering and optional force magnitudes.
- Reset bookkeeping: `episode_id`, `reset_count`, `EpisodeInfo` emission.

**Exit:** unit tests parity against the ROS node behavior on a recorded
fixture (driven from a smoke-corpus rollout).

### 1.3 `lewm_genesis/scene_loader.py`

Extend [scene_builder.py](../lewm_genesis/lewm_genesis/scene_builder.py) to:

- Consume `.generated/scene_corpus/<name>/<split>/<family>/<scene_id>/genesis_scene.json`.
- Build Genesis scene with `n_envs=64`.
- Attach Go2 URDF via [go2_adapter.py](../lewm_genesis/lewm_genesis/go2_adapter.py).
- Attach camera at the manifest-specified mount pose with intrinsics from
  [config/go2_platform_manifest.yaml](../config/go2_platform_manifest.yaml).

**Exit:** loads one smoke-corpus scene and renders one frame to disk.

### 1.4 `lewm_genesis/rollout.py`

Main bulk loop. Per scene:

- Build scene + load Tier A policy.
- Sample command tape from [config/go2_primitive_registry.yaml](../config/go2_primitive_registry.yaml).
- Step physics at `physics_dt` and control at `control_dt` (decimation from
  platform manifest); render at `camera_hz`.
- Capture state, commands, contacts, RGB per tick.
- Handle resets: terminate on fall/collision per data spec; advance
  `episode_id`; emit `ResetEvent` + `EpisodeInfo`.

Modeled on [../../LeWMQuad-v2/scripts/1_physics_rollout.py](../../LeWMQuad-v2/scripts/1_physics_rollout.py)
(rollout loop at lines 1006–1126).

**Exit:** one scene × 64 envs × 1000 ticks produced in memory.

### 1.5 `lewm_genesis/mcap_writer.py`

Write rosbag2 MCAP per scene with all required LeWM message types.

- `rosbag2_py` Python API, sim-time clock.
- All 64 envs as per-env namespaces (`/env_00/lewm/go2/...` …
  `/env_63/lewm/go2/...`) so each stream is independently replayable.
- Sidecar `summary.json` with scene manifest hash, audit results, env count,
  step count.

**Exit:** generated MCAP plays back via `ros2 bag play` and exposes all
expected topics.

### 1.6 `scripts/genesis_bulk_rollout.{py,sh}`

CLI entry point. Args:

```
--scene-corpus <dir>   # e.g. .generated/scene_corpus/smoke
--split <name>         # train | val | test_id | test_hard
--out <dir>            # output root
--n-envs 64
--steps 1000
--policy <ckpt>        # Tier A checkpoint
```

Iterates the split, emits one MCAP + summary per scene.

**Exit:** end-to-end smoke-corpus run produces valid outputs that pass the
`smoke` data-quality profile.

### 1.7 Audit hooks

Adapt `contract_audit` / `topic_audit` / `data_quality_audit` from
[scripts/convert_smoke_bag_to_raw_rollout.py](../scripts/convert_smoke_bag_to_raw_rollout.py)
to operate on the new per-scene MCAPs. Profiles: `smoke`, `raw_pilot`,
`raw_training` per data spec §4.1.

**Exit:** `smoke` profile passes on a generated MCAP; `raw_pilot` /
`raw_training` surface known gaps for tuning.

## Out of Phase 1 scope

- Tier A gait distribution validation → Phase 2.
- Safety threshold recalibration → Phase 2.
- Genesis-Gazebo parity audit → Phase 2.
- Derived labels (§5.3) → Phase 3 or downstream offline pass.
- Missing scene families (composite motifs, rough/local dynamics,
  visual/sensor stress) → parallel `lewm_worlds` work, not bulk-loop work.
- Tier B policy port → only if Tier A fails Phase 1.1.
- DDS-overhead modeling → audit oracle path only.

## Open questions

- Camera output: per-env separate cameras vs. one camera with batched env
  rendering. Genesis-side ergonomics dictate; resolve in 1.3.
- MCAP namespacing: per-env topic prefix (`/env_NN/...`) vs. envelope field
  on each message. Resolve in 1.5 against rosbag2 tooling expectations.
- Reset policy: data spec §10 needs failed and successful traversals;
  current `reset_manager` resets on collision. Confirm we capture *both*
  pre-collision approach and recovery, not just successful runs.
