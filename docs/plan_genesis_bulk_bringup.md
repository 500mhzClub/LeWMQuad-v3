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
- **§5.3 derived labels:** out of the rollout loop by design. Computed
  offline from raw rollout pose logs plus regenerated scene metadata by
  [scripts/derive_raw_rollout_labels.py](../scripts/derive_raw_rollout_labels.py).
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

### 1.1 Install Genesis and port Tier A policy

- Install Genesis with a verified GPU backend on the production GPU
  (Radeon AI Pro 9700, 32 GB). For the local current-Genesis path, see
  [genesis_rocm_local_audit.md](genesis_rocm_local_audit.md): `genesis-world`
  `0.4.6` can step `gs.amdgpu` with `n_envs=4` under an isolated ROCm PyTorch
  venv, including the Genesis-bundled Go2 URDF and batched 12-DOF leg control.
  Legacy Vulkan notes live in
  [genesis_vulkan_local_audit.md](genesis_vulkan_local_audit.md).
- Fetch the open `unitree_rl_gym` Go2 PPO checkpoint and port it into a
  legged_gym-style policy adapter that consumes the obs layout described in
  "Known-good locomotion policy" above and emits 12-dim joint position offsets
  from the default qpos. PD control via Genesis
  `robot.control_dofs_position`.
- Freeze the ported policy (weights + obs/action adapter) under
  `models/tier_a_go2_locomotion/<date>/`.
- Tier B fallback (training a Genesis-native policy via upstream
  `examples/locomotion/go2_train.py` to convergence and freezing the resulting
  checkpoint at the same path) is only triggered if Tier A's gait validation
  in Phase 2 fails. The local trainer smoke documented in
  [genesis_rocm_local_audit.md](genesis_rocm_local_audit.md) proves the
  training path is wired; it is not the planned Phase 1 step.

**Exit:** ported Tier A policy produces a walking gait for ≥10 s under
constant `(vx=0.3, vy=0, yaw_rate=0)` inside Genesis without falling.

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
- Per-episode each env draws a collector from the data-spec §13 mix via
  [`lewm_genesis.collectors.EpisodeScheduler`](../lewm_genesis/lewm_genesis/collectors/base.py)
  (`route_teacher` 30 %, `frontier` 20 %, `primitive_curriculum` 20 %,
  `ou_noise` 10 %, `recovery` 10 %, `loop_revisit` 10 %). Per block each
  env's assigned collector observes privileged state (base xy/yaw, current
  cell, clearance to walls, last executed cmd) and returns a primitive +
  source tag.
- Step physics at `physics_dt` and control at `control_dt` (decimation from
  platform manifest); render at `camera_hz`.
- Capture state, commands, contacts, RGB per tick. Each emitted
  `CommandBlock` carries `command_source`, `route_target_id`, and
  `next_waypoint_id` privileged-label fields.
- Handle resets: terminate on fall/collision per data spec; advance
  `episode_id`; emit `ResetEvent` + `EpisodeInfo`. Spawn pose is
  re-sampled per episode (random reachable cell + random yaw,
  clearance-gated) via `SceneGraph.sample_spawn_pose`.

Modeled on [../../LeWMQuad-v2/scripts/1_physics_rollout.py](../../LeWMQuad-v2/scripts/1_physics_rollout.py)
(rollout loop at lines 1006–1126).

**Exit:** one scene × 64 envs × 1000 ticks produced in memory, with the
realized §13 mix reported in the rollout stats summary.

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

### 1.8 Offline derived-label pass

Phase A1 is implemented as a post-processor, not as rollout code:

```bash
scripts/derive_raw_rollout_labels.py \
  .generated/genesis_bulk_rollouts/<run>/<scene_id> \
  --out .generated/derived_labels/<run>/<scene_id>
```

The script accepts compact `messages.jsonl` or per-scene rosbag2 MCAP input.
It resolves the scene from `summary.json` (`family`, `topology_seed`, split,
difficulty tier), from `--family + --topology-seed`, from `--scene-manifest`,
or by searching `--scene-corpus`. It writes `labels.jsonl` plus `summary.json`
with per-step `cell_id`, `yaw_bin`, `local_graph_type`,
`nearest_cell_distance_m`, `bfs_distance_to_landmark`, clearance,
traversability, landmark visibility/bearing/range, and integrated body motion.

**Exit:** focused tests cover graph-type classification, BFS landmark labels,
body-motion windows, and command/episode/base-pose joins.

## Next production ramp

There are no known implementation blockers to proceed with the staged
production path. The remaining items are validation gates for scale, not
reasons to keep implementing before the next shard. Use the split-phase path:

1. CPU Genesis physics rollout with MCAP writer and `--no-rgb`.
2. Convert to compact `raw_rollout` with the `raw_training` profile.
3. Plan render replay from the converted raw rollout.
4. Bulk render RGB/depth on AMDGPU.
5. Compute derived labels offline.

Initial shard:

```bash
scripts/genesis_bulk_rollout.sh \
  --scene-corpus .generated/scene_corpus/acceptance \
  --split train \
  --scene-limit 1 \
  --n-envs 512 \
  --n-blocks 20 \
  --backend cpu \
  --no-rgb \
  --out .generated/genesis_bulk_rollouts/cpu_512env_20block_pilot
```

Post-process:

```bash
scripts/convert_smoke_bag_to_raw_rollout.sh \
  .generated/genesis_bulk_rollouts/cpu_512env_20block_pilot/<scene_id> \
  --out .generated/raw_rollouts/cpu_512env_20block_pilot/<scene_id> \
  --quality-profile raw_training

scripts/plan_bulk_render_replay.sh \
  --raw-root .generated/raw_rollouts/cpu_512env_20block_pilot \
  --out-root .generated/rendered_vision_plans/cpu_512env_20block_pilot \
  --camera-hz 10

scripts/render_replay_genesis.sh \
  .generated/rendered_vision_plans/cpu_512env_20block_pilot/<plan_dir>/render_replay_plan.json \
  --backend amdgpu \
  --out .generated/rendered_vision/cpu_512env_20block_pilot/<plan_dir>

scripts/derive_raw_rollout_labels.py \
  .generated/raw_rollouts/cpu_512env_20block_pilot/<scene_id> \
  --scene-corpus .generated/scene_corpus/acceptance \
  --out .generated/derived_labels/cpu_512env_20block_pilot/<scene_id>
```

Gate this shard on:

- `raw_training` conversion pass;
- render `invalid_frame_count=0`;
- derived-label rows matching the base-state pose count, except explained
  sentinels;
- stable disk/write throughput.

Ramp order after a clean shard: `512 x 20 blocks`, then `512 x 80`, then
`512 x 200`. Only test writer-enabled 1024-env shards after the 512-env ramp
is stable.

## Out of Phase 1 scope

- Tier A gait distribution validation → Phase 2.
- Safety threshold recalibration → Phase 2.
- Genesis-Gazebo parity audit → Phase 2.
- Inline derived-label computation inside `RolloutRunner` or `MCAPSceneWriter`
  → intentionally out of scope. §5.3 labels are **landed as a downstream
  offline pass** in
  [lewm_worlds/labels/derived.py](../lewm_worlds/lewm_worlds/labels/derived.py)
  and [scripts/derive_raw_rollout_labels.py](../scripts/derive_raw_rollout_labels.py).
- Missing scene families (composite motifs, rough/local dynamics,
  visual/sensor stress) → **landed** in `lewm_worlds.families`; all eight
  spec families now in the registry.
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
