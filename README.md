# LeWMQuad-v3 Go2 Simulator Bring-up

This repository targets a Unitree Go2 simulation stack for LeWMQuad-v3 data
generation and later H-JEPA training.

> **Backend pivot (2026-05-15):** Genesis is now the authoritative bulk
> data-generation backend. Gazebo is retained as a small audit oracle only.
> Lidar is dropped from the v3 corpus contract. See
> [docs/decision_pivot_to_genesis.md](docs/decision_pivot_to_genesis.md) for
> the full decision, rationale, and rollback condition.

## Backend Roles

- **Genesis (bulk path):** authoritative for physics rollout, RGB/depth
  rendering, and raw_rollout production at scale. Runs on the production GPU
  (Radeon AI Pro 9700, ROCm). The current pilot CLI writes per-scene MCAP
  raw-rollout directories; later bulk shards can be .h5/.npz/Zarr once the
  schema is locked. Implementation lives in [lewm_genesis/](lewm_genesis/) and is being
  brought up against the pattern in
  [../LeWMQuad-v2/scripts/1_physics_rollout.py](../LeWMQuad-v2/scripts/1_physics_rollout.py).
  Tier A locomotion ports the open `unitree_rl_gym` Go2 PPO checkpoint into a
  Genesis-compatible policy adapter (no training on our side). Training a
  Genesis-native policy via Genesis's `go2_train.py` is the Tier B fallback.
- **Gazebo (audit oracle only):** small, periodic dynamics and sensor parity
  spot checks against the Genesis corpus. ROS 2 + CHAMP + ros_gz_bridge stay
  alive only for this role. All Gazebo-specific instructions below describe
  the audit oracle, not the production path.

## Audit Oracle Stack (Gazebo)

Everything from here through the "Known Blockers" section describes the
Gazebo audit oracle: setup, smoke tests, throughput benchmarks, and bag
recording. This is the parity reference, not the production generator.

The audit oracle base is:

- OS: Ubuntu 24.04 Noble or a Noble-derived OS such as Pop!_OS 24.04.
- ROS 2: Jazzy.
- Gazebo: Harmonic / Gazebo Sim.
- Go2 simulator submodule: `https://github.com/khaledgabr77/unitree_go2_ros2.git`.
- Pinned upstream commit: `29bce68480dcc3d3bac8cc0cac983f8ac951e8e3`.
- Go2 locomotion path: CHAMP controller driven by `/cmd_vel`.

Do not use ROS 2 Humble or Gazebo Classic for this workspace unless the platform
plan is explicitly changed. The local scripts sanitize any currently sourced ROS
environment before sourcing Jazzy, which matters on machines that already source
Kilted or another distro by default.

## Repository Layout

- `lewm_genesis`: Genesis-side bulk generation (in progress). Scene building,
  parity checks, and render-replay scaffolding. Becomes the bulk physics +
  rendering loop; cross-references the v2 rollout pattern.
- `lewm_worlds`: canonical scene manifest model and split planner. Consumed by
  both Genesis and the Gazebo audit oracle.
- `lewm_go2_control`: LeWM-specific ROS 2 message and service interfaces.
  Contract is sim-agnostic; runtime form is ROS nodes under the audit oracle
  and Python utilities under Genesis bulk.
- `third_party/unitree_go2_ros2`: selected upstream Go2 simulator, description,
  Gazebo launch, and CHAMP controller packages. Audit oracle only.
- `config/go2_platform_manifest.yaml`: pinned platform, camera, controller, and
  no-data gates.
- `config/go2_primitive_registry.yaml`: initial command primitive bank.
- `docs/decision_pivot_to_genesis.md`: backend-role authority. **Read first.**
- `docs/go2_genesis_gazebo_ros2_bringup_plan.md`: superseded as backend
  authority; LeWM interface specs inside still apply.
- `docs/upstream_go2_sim_audit.md`: upstream fit and license audit for the
  Gazebo audit oracle.
- `scripts/`: install, alignment, build, launch, smoke-test, benchmark, and
  render-replay helpers (audit oracle plus Genesis-side planners).

## Native Setup

Run this from the repository root.

```bash
scripts/install_jazzy_harmonic_deps.sh
```

This installs the Jazzy/Harmonic packages used by the selected upstream stack:

- `ros-jazzy-desktop`
- `ros-jazzy-ros-gz`
- `ros-jazzy-gz-ros2-control`
- `ros-jazzy-xacro`
- `ros-jazzy-robot-localization`
- `ros-jazzy-ros2-control`
- `ros-jazzy-ros2-controllers`
- `ros-jazzy-velodyne-description`
- `ros-jazzy-realsense2-description`

The upstream README mentions `ros-jazzy-gazebo-ros2-control`; this workspace uses
`ros-jazzy-gz-ros2-control`, which is the Gazebo Sim / Harmonic control package.

If sudo asks for a password, enter your normal login password. If your OS does
not expose Noble apt packages despite being Ubuntu 24-derived, install Jazzy and
Gazebo Harmonic through a Noble container or fix the ROS apt source before
continuing.

## Environment Check

After installation:

```bash
scripts/check_ros_gz_alignment.sh
```

This verifies:

- OS codename is `noble`.
- `/opt/ros/jazzy/setup.bash` exists and sources correctly.
- `ros2`, `colcon`, `rosdep`, and `gz` are available.
- Required ROS packages are discoverable.
- The Go2 submodule URL and pinned commit match the manifest.

If this fails because Kilted is currently sourced, open a fresh shell and rerun
the script. The scripts try to sanitize ROS paths, but a clean shell is still the
least confusing way to run ROS workspaces.

## Build

The current `build/` directory may have been created under ROS Kilted during
early scaffold checks. Clean it before the first Jazzy build:

```bash
scripts/build_go2_sim.sh --clean
```

For later incremental builds:

```bash
scripts/build_go2_sim.sh
```

The build target is intentionally scoped to the selected Go2 simulator packages
and the LeWM adapter:

```bash
colcon build --symlink-install \
  --packages-select \
    champ \
    champ_msgs \
    champ_base \
    unitree_go2_description \
    unitree_go2_sim \
    lewm_worlds \
    lewm_genesis \
    lewm_go2_bringup \
    lewm_go2_control
```

## Launch

Launch the upstream stock Go2 sim first. Do this before adding LeWM world
generation, reset managers, or data recording.

```bash
scripts/launch_go2_sim.sh
```

The default wrapper launches:

```bash
ros2 launch lewm_go2_bringup go2_sim.launch.py rviz:=false gui:=false
```

It also starts the first-pass LeWM command adapter, the base state publisher,
the manifest-driven CameraInfo publisher, the episode-bookkeeping reset
manager, the CHAMP foot-contact republisher, the mode/policy service node,
and the feature-check service node by default:

```text
/lewm/go2/command_block -> /cmd_vel
/lewm/go2/executed_command_block
/gazebo/odom -> /lewm/go2/base_state
/rgb_image -> /lewm/go2/camera_info
/lewm/go2/reset (service) -> /lewm/go2/reset_event
/lewm/episode_info
/foot_contacts (CHAMP gait phase) -> /lewm/go2/foot_contacts
/lewm/go2/set_mode (service) -> /lewm/go2/mode
/lewm/go2/set_policy (service) -> /lewm/go2/mode
/lewm/go2/run_feature_check (service)
```

Disable them only when debugging the stock upstream stack:

```bash
scripts/launch_go2_sim.sh lewm_adapter:=false
scripts/launch_go2_sim.sh lewm_base_state:=false
scripts/launch_go2_sim.sh lewm_camera_info:=false
scripts/launch_go2_sim.sh lewm_reset:=false
scripts/launch_go2_sim.sh lewm_foot_contacts:=false
scripts/launch_go2_sim.sh lewm_mode:=false
scripts/launch_go2_sim.sh lewm_feature_check:=false
```

To pass through launch arguments:

```bash
scripts/launch_go2_sim.sh rviz:=false gui:=true
scripts/launch_go2_sim.sh rviz:=true gui:=false
```

To launch a generated canonical smoke world instead of the upstream default:

```bash
scripts/generate_smoke_world.sh --seed 7
scripts/launch_go2_sim.sh --smoke-world 7 rviz:=false gui:=false
```

The smoke world is written under `.generated/worlds/` and uses Gazebo world
name `default` so the reset manager can still call `/world/default/set_pose`.

To generate a multi-family scene corpus instead of a single world:

```bash
# Smoke tier (default): one scene per family across train/val/test_id/test_hard.
scripts/generate_scene_corpus.sh --name smoke

# Spec-aligned plan with custom split totals (data spec section 8 shares).
scripts/generate_scene_corpus.sh --standard \
  --name pilot --plan-seed 1 \
  --train 200 --val 50 --test-id 50 --test-hard 50

# Custom totals JSON.
scripts/generate_scene_corpus.sh --plan path/to/totals.json --name custom
```

Each scene is written to
`.generated/scene_corpus/<name>/<split>/<family>/<scene_id>/` with
`world.sdf`, `manifest.json`, `topology.json`, and `genesis_scene.json`. The
top-level `corpus.json` records the full plan and per-scene manifest
checksums. Splits never share a topology seed for a given plan seed.

On Pop!_OS/Wayland, `scripts/launch_go2_sim.sh` automatically sets
`QT_QPA_PLATFORM=xcb` when a display is available. This avoids common
Gazebo/RViz OGRE GLX window errors while still allowing headless launches. Set
`LEWM_GO2_KEEP_QT_PLATFORM=1` only when deliberately testing another Qt backend.

The upstream launch brings up Gazebo, the Go2 model, `ros_gz_bridge`,
`robot_state_publisher`, CHAMP control, EKF localization, and ROS 2 control
spawners.

## Smoke Test

In a second terminal, from the repository root:

```bash
scripts/smoke_go2_topics.sh
```

Required first-pass topics:

- `/clock`
- `/tf`
- `/joint_states`
- `/imu/data`
- `/odom`
- `/rgb_image`
- `/cmd_vel`

To also send a short forward velocity command:

```bash
scripts/smoke_go2_topics.sh --drive
```

LeWM command-block adapter smoke test. In one terminal:

```bash
scripts/ros2_go2.sh ros2 topic echo /lewm/go2/executed_command_block --once
```

In another terminal:

```bash
scripts/ros2_go2.sh ros2 topic pub /lewm/go2/command_block \
  lewm_go2_control/msg/CommandBlock \
  "{sequence_id: 1, block_size: 5, command_dt_s: 0.1, primitive_name: arc_left}" \
  --once
```

The adapter expands named velocity primitives from
`config/go2_primitive_registry.yaml`, clips against
`config/go2_platform_manifest.yaml`, publishes `/cmd_vel`, then records the
requested and executed command arrays for later audits.

Mode-event primitives such as `recovery_stand` dispatch through
`/lewm/go2/set_mode` and publish a zero-velocity block. Under the current
CHAMP backend this is a conservative stance/stop mapping, not a Unitree
sport-mode fall-recovery behavior.

LeWM mode and policy service smoke tests:

```bash
scripts/ros2_go2.sh ros2 service call /lewm/go2/set_mode \
  lewm_go2_control/srv/SetGo2Mode \
  "{mode: 'hold'}"

scripts/ros2_go2.sh ros2 service call /lewm/go2/set_policy \
  lewm_go2_control/srv/SetPolicy \
  "{backend_id: 'champ_cmd_vel'}"

scripts/ros2_go2.sh ros2 topic echo /lewm/go2/mode --once
```

LeWM base state publisher smoke test:

```bash
scripts/ros2_go2.sh ros2 topic echo /lewm/go2/base_state --once
```

The publisher converts the bridged Gazebo ground-truth `/gazebo/odom` stream
into `BaseState`, preserving the world-frame pose, rotating the body-frame twist
into the world frame, and emitting roll/pitch/yaw alongside the raw `(x, y, z,
w)` quaternion. The upstream EKF still publishes `/odom`.

LeWM CameraInfo publisher smoke test:

```bash
scripts/ros2_go2.sh ros2 topic echo /lewm/go2/camera_info --once
```

The publisher computes the RGB camera intrinsics from
`config/go2_platform_manifest.yaml` (resolution + horizontal FOV) and emits a
`CameraInfo` stamped to match each `/rgb_image` frame, since the upstream
Gazebo camera plugin does not publish one for `/rgb_image`.

LeWM reset manager smoke test. Echo the event topic in one terminal:

```bash
scripts/ros2_go2.sh ros2 topic echo /lewm/go2/reset_event
```

Call the service in another:

```bash
scripts/ros2_go2.sh ros2 service call /lewm/go2/reset \
  lewm_go2_control/srv/ResetEpisode \
  "{scene_id: 1, reason: 'manual_smoke', use_spawn_pose: false}"
```

The reset manager owns the monotonic `episode_id` and `reset_count` counters
used to split training windows on episode boundaries and publishes
`/lewm/episode_info` after each reset. When `use_spawn_pose` is true it also
sends zero `/cmd_vel` commands around a `gz service -s /world/default/set_pose`
teleport. The service response reports whether the teleport succeeded, and the
published `ResetEvent` carries the requested pose for downstream auditing. Set
`enable_teleport:=false` on the node parameters to disable the teleport path
while keeping the bookkeeping.

Limitations to know about:

- The teleport sets the entity pose only; linear and angular velocities are
  not zeroed through a physics-state API, and the joint trajectory in progress
  is not paused. The reset manager now sends zero velocity before and after the
  teleport, but CHAMP's kinematic odometry never tracks a teleport (it is pure
  dead reckoning), and a teleport that lands during a swing phase or with
  a high drop will frequently topple the robot. Cross-reset training
  windows are filtered out by the dataset loader, so the controller
  discontinuity is acceptable for episode boundaries, but callers should
  expect the post-reset frames to look unsteady.
- Pick a spawn `z` slightly above CHAMP's stance height. The launch
  default of `0.375 m` matches upstream and lands cleanly; lower values
  let the feet penetrate the ground at the teleport instant.

LeWM foot contacts smoke test:

```bash
scripts/ros2_go2.sh ros2 topic echo /lewm/go2/foot_contacts --once
```

The publisher republishes CHAMP's kinematic foot-contact booleans as the
LeWM `FootContacts` msg. CHAMP exposes a per-leg `gait_phase()` boolean on
`/foot_contacts` (`champ_msgs/ContactsStamped`) whenever the quadruped
controller runs with `gazebo: False` and `publish_foot_contacts: True`;
the launch sets those parameters here. The signal is the gait planner's
intended stance/swing state, not a physics-derived contact -- it is what
CHAMP itself uses on real hardware that lacks per-foot sensors. The
republisher maps CHAMP's `lf, rf, lh, rh` order to LeWM's
`fl, fr, rl, rr` fields and sets the four `*_force_n` fields to `0.0`,
since the kinematic signal carries no force information.

This is a deliberate retreat from gz `contact` sensors. An earlier slice
declared four `<sensor type="contact">` blocks plus a
`gz-sim-contact-system` plugin in a custom world; that combination
destabilized CHAMP's effort-tracked gait within ~6 s of `/cmd_vel` under
`bullet-featherstone`. The sensor declarations alone were harmless, and
the plugin alone was harmless, but together the per-tick contact
processing was enough to wreck gait stability. The kinematic path here
has zero physics-side cost and restores the upstream's ~30+ s walking
endurance.

Feature-check smoke test:

```bash
scripts/ros2_go2.sh ros2 service call /lewm/go2/run_feature_check \
  lewm_go2_control/srv/RunFeatureCheck \
  "{check_name: 'all', include_optional: false}"
```

The service checks topic presence, live messages, mode/policy services, a
primitive-registry audit, a `hold` command-block round trip, and reset
bookkeeping. Use specific `check_name` values such as `topics`, `messages`,
`mode_services`, `primitive_registry`, `command_round_trip`, or
`reset_bookkeeping` when isolating failures.

Primitive motion audit, after controllers are active:

```bash
scripts/audit_go2_primitives.sh
```

This calls `/lewm/go2/run_feature_check` with `check_name: primitive_motion`,
drives every trainable primitive, and writes a JSON report under
`.generated/audits/`.

End-to-end acceptance gate. This builds the workspace, generates a smoke
scene corpus, launches the simulator headless, runs the full feature check
and primitive audit, records a smoke bag while driving through
`/lewm/go2/command_block`, converts the bag to a `raw_rollout`, then tears the
simulator down:

```bash
scripts/acceptance_go2_contract.sh
```

The run writes per-stage logs and a top-level `summary.json` under
`.generated/acceptance/<run-name>/`. Pass `--skip-build` (or `--skip-launch`
if a simulator is already running) to shorten the loop. The exit code is
non-zero if any non-skipped stage failed.

Gazebo GUI smoke test:

```bash
scripts/launch_go2_sim.sh rviz:=false gui:=true
scripts/smoke_go2_topics.sh --drive
```

RViz smoke test:

```bash
scripts/launch_go2_sim.sh rviz:=true gui:=false
scripts/smoke_go2_topics.sh --drive
```

On this Pop!_OS/Wayland workstation, Gazebo GUI and RViz have been validated
separately. The combined `rviz:=true gui:=true` launch now avoids the original
OGRE/GLX crash after forcing `xcb`, but it is still not reliable enough as a
bring-up gate on this workstation because controller spawners can stall under
the combined GUI load. Use the two separate GUI checks above for now.

For ad hoc ROS commands, use the wrapper so the command always runs with ROS 2
Jazzy and this workspace overlay, even if your shell has another ROS distro
sourced by default:

```bash
scripts/ros2_go2.sh ros2 control list_controllers
scripts/ros2_go2.sh ros2 topic echo /odom --once
scripts/ros2_go2.sh ros2 topic echo /gazebo/odom --once
scripts/ros2_go2.sh ros2 topic echo /joint_states --once
scripts/ros2_go2.sh ros2 topic hz /rgb_image
```

You can manually publish the same command with:

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.15, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" \
  -r 5
```

## First Successful Run Criteria

Before any LeWM data generation work, the stock simulator must pass:

1. Go2 spawns without unstable joint explosions.
2. `/cmd_vel` causes visible forward, backward, yaw-left, and yaw-right motion.
3. `/joint_states`, `/imu/data`, `/odom`, `/gazebo/odom`, `/tf`, and
   `/rgb_image` publish with simulation timestamps.
4. RViz can visualize the TF tree and robot state when launched with
   `rviz:=true`.
5. A short ROS bag can capture the required topics without timestamp gaps.

Example smoke bag:

```bash
scripts/record_smoke_bag.sh --duration 20
scripts/convert_smoke_bag_to_raw_rollout.sh .generated/bags/<bag-dir>
```

`record_smoke_bag.sh` records with an explicit subscription QoS overrides
file at [config/rosbag_record_qos_overrides.yaml](config/rosbag_record_qos_overrides.yaml).
Default rosbag2 subscriptions use depth=10, which at `/clock`'s ~1 kHz is
only ~10 ms of buffer headroom; under any subscriber stall the middleware
reports a `MessageLost` event and the recorder logs it as a transport drop.
The overrides give each hot topic at least ~250 ms of headroom (e.g.
`/clock` depth 1024, `/tf` and `/joint_states` depth 256). Pass
`--no-qos-overrides` to fall back to defaults for comparison, or
`--qos-overrides <path>` to point at a custom file. The recorder also bumps
`--max-cache-size` to 256 MiB to keep RGB frames buffered during writes.

The converter writes a compact smoke `raw_rollout` directory with
`summary.json` and `messages.jsonl`. Large image/point-cloud payload arrays are
omitted from JSONL records while their timestamps and metadata are preserved.
Under the audit oracle role, use `scripts/record_smoke_bag.sh --profile raw`
to capture a state/action-only audit bag without `/rgb_image`. Bulk RGB
generation does not run through this path anymore; it runs through Genesis.

`summary.json` also contains `contract_audit`, `topic_audit`, and
`data_quality_audit` blocks. Every requested
`/lewm/go2/command_block.sequence_id` must reappear as an
`/lewm/go2/executed_command_block.sequence_id`, no `sequence_id` may repeat
on either topic, and `/lewm/go2/reset_event.reset_count` must increase by
exactly one per event. A missing execution, duplicate sequence_id, or reset
gap fails the audit and the converter exits non-zero. Orphan executions
(executions whose command was published before the recording started) are
flagged in the audit but do not fail it.

The default converter policy is `--quality-profile smoke`: only
contract-critical command/reset loss fails the conversion, while per-topic
timing is reported for triage. Before pilot or training data capture, run with
`--quality-profile pilot` or `--quality-profile training`. Those stricter
profiles also fail missing critical streams, timestamp regressions, excessive
topic gaps, and low observed rates on `/clock`, `/tf`, state, odometry, IMU,
RGB, camera-info, contacts, mode, episode-info, and `/cmd_vel`. Pass
`--no-strict` to write the audits without failing — useful when triaging a
known-bad bag or producing a benchmark report.

Gazebo throughput benchmark, with a simulator already running:

```bash
scripts/benchmark_gazebo_throughput.sh --duration 60 --machine-role dev_laptop --capture-profile vision
scripts/benchmark_gazebo_throughput.sh --duration 60 --machine-role production_desktop --capture-profile raw
```

This records a command-block-driven bag, converts it, and writes
`.generated/benchmarks/<run>/throughput.json` with real-time factor, RGB frame
rate, bag write rate, dropped/lost-message log lines, the command-contract
audit, the pilot-profile data-quality audit, conversion duration, and host
metadata. Under the post-pivot scope these benchmarks measure audit-oracle
capacity, not the production generation rate. The production rate is set by
Genesis, not Gazebo. See
[docs/decision_pivot_to_genesis.md](docs/decision_pivot_to_genesis.md).

GPU render-replay planning from a converted `raw_rollout`:

```bash
scripts/plan_gpu_render_replay.sh .generated/benchmarks/<run>/raw_rollout \
  --out .generated/rendered_vision_plans/<run>
```

This writes `render_replay_plan.json` and `frames.jsonl`: the immutable contract
for a renderer consuming an audit-oracle `raw_rollout`. The bulk path skips
this stage entirely — Genesis emits state and RGB in the same pass. The
render-replay scaffold is preserved for the audit-only flow where Gazebo
captures dynamics and a separate renderer (Genesis or otherwise) generates
matching frames for parity comparison.

## Known Blockers Before Data Generation

### Genesis bulk path (production)

- The Genesis bulk loop now has a PPO-backed CLI entry point:

  ```bash
  scripts/genesis_bulk_rollout.sh \
    --scene-corpus .generated/scene_corpus/acceptance \
    --split train \
    --scene-limit 1 \
    --n-envs 1 \
    --n-blocks 1 \
    --backend cpu \
    --out .generated/genesis_bulk_rollouts/ppo_smoke
  ```

  The script writes per-scene MCAP raw-rollout directories through the existing
  `MCAPSceneWriter`.
- On `--backend amdgpu`, the bulk CLI defaults `foot_contact_source=zero`
  because Genesis `get_links_net_contact_force()` triggers an HSA hardware
  exception on the Radeon AI Pro R9700. CPU rollouts still use Genesis-derived
  contact-force labels by default. The AMDGPU path still emits the
  `/lewm/go2/foot_contacts` topic at command-tick cadence, and summaries mark
  the contact source explicitly.
- The validated Genesis-native Go2 PPO artifact is frozen under
  `models/tier_a_go2_locomotion/20260516_contract_ppo/` and referenced by
  `config/go2_platform_manifest.yaml`. It passed the LeWM velocity primitive
  contract for hold, forward, backward, yaw, and arc commands. Nonzero lateral
  `vy_body_mps` remains disabled until trained and validated.
- Tier B training wrappers ship in this repo; on the production GPU host:

  ```bash
  scripts/setup_genesis_rocm_training.sh                      # one-time deps
  scripts/check_genesis_go2_locomotion_train_rocm_smoke.sh    # 1-iter wiring check
  scripts/train_genesis_go2_locomotion_contract.sh            # LeWM command-contract PPO
  scripts/check_genesis_go2_policy_contract.sh --exp-name lewm-go2-contract-20260516T163413Z --ckpt 500
  ```

  Logs land under `.generated/upstream_genesis/locomotion/logs/<exp_name>/`.
  Tune via `LEWM_GO2_NUM_ENVS`, `LEWM_GO2_MAX_ITERS`, `LEWM_GO2_SEED`,
  `LEWM_GO2_EXP_NAME`, and `LEWM_GO2_REWARD_TRACKING_ANG_VEL`. Do **not** set
  `HSA_OVERRIDE_GFX_VERSION` on RDNA4 (Radeon AI Pro 9700 / `gfx1201`); it is
  only needed on the laptop iGPU (`gfx1103`).
- Safety thresholds in `config/go2_platform_manifest.yaml` now reflect the
  validated PPO primitive set; lateral velocity is clamped to zero.
- ROCm-Genesis training and contract validation have passed on the production
  Radeon AI Pro R9700. Local ROCm audit status is tracked in
  [docs/genesis_rocm_local_audit.md](docs/genesis_rocm_local_audit.md):
  `genesis-world==0.4.6` can step `gs.amdgpu` with `n_envs=4` under an isolated
  ROCm PyTorch venv, including the Genesis-bundled Go2 URDF, 12-DOF leg
  control path, PPO-backed MCAP rollout, RGB capture, and `raw_pilot`
  conversion. A two-scene, two-block AMDGPU RGB pilot passed on
  `large_enclosed_maze` and `loop_alias_stress`. Legacy Vulkan status is
  tracked in [docs/genesis_vulkan_local_audit.md](docs/genesis_vulkan_local_audit.md).

### Audit oracle (Gazebo)

- License: the selected upstream repo is technically suitable but has
  incomplete license metadata for `unitree_go2_description` and
  `unitree_go2_sim`. Use it for local evaluation until the license is
  clarified or replaced.
- World generation: `lewm_worlds` now ships per-family generators
  (`open_obstacle_field`, `small_enclosed_maze`, `medium_enclosed_maze`,
  `large_enclosed_maze`, `loop_alias_stress`), a deterministic
  train/val/test_id/test_hard split planner with disjoint seeds, and a corpus
  builder. `scripts/generate_scene_corpus.sh` materializes a smoke or
  spec-aligned plan to `.generated/scene_corpus/<name>/`. The same manifest
  drives both Genesis bulk and the Gazebo audit oracle.
- No explicit gait-endurance blocker on the audit oracle: upstream CHAMP +
  bullet-featherstone delivers ~30+ s of sustained `/cmd_vel` walking on
  this workstation, comfortably above the data spec's 16-24 s per-episode
  minimum
  ([docs/fresh_retrain_data_spec.md](docs/fresh_retrain_data_spec.md)).
  This claim depends on keeping the gz `contact-system` plugin and per-foot
  `<sensor type="contact">` blocks out of the spawned model -- see the
  foot-contacts section above for why those break the gait.

### Dropped from scope

- Lidar (`/velodyne_points`, `/unitree_lidar/points`) is not in the v3 corpus
  contract. The Velodyne-description apt package is still required because
  the upstream URDF references a Velodyne mount body, not because lidar data
  is captured.
- The "use Gazebo until throughput proves a blocker" gate from the original
  bringup plan no longer applies. The 154M-tick spec target made Gazebo a
  bulk-path blocker without further pilot benchmarking.

Resolved on the current branch (kept here as a paper trail; the live state
lives in the launch and the smoke tests above):

- Camera info: `/lewm/go2/camera_info` is now published from the manifest.
- Foot contacts: CHAMP gait-phase contacts + the LeWM republisher publish
  `/lewm/go2/foot_contacts`.
- Reset semantics: `/lewm/go2/reset` + `/lewm/go2/reset_event` advance
  monotonic episode counters, publish `/lewm/episode_info`, and optionally
  teleport via `gz set_pose`.
- Mode/policy/feature services: `/lewm/go2/set_mode`, `/lewm/go2/set_policy`,
  `/lewm/go2/mode`, and `/lewm/go2/run_feature_check` are implemented for the
  CHAMP `/cmd_vel` backend.

## Troubleshooting

Wrong ROS distro:

```bash
echo "$ROS_DISTRO"
```

If this prints `kilted`, `humble`, or anything other than `jazzy`, open a fresh
terminal and use the scripts instead of manually sourcing overlays.

Missing Gazebo:

```bash
gz sim --version
```

If `gz` is missing, rerun:

```bash
scripts/install_jazzy_harmonic_deps.sh
```

Stale build cache:

```bash
scripts/build_go2_sim.sh --clean
```

No GUI:

```bash
scripts/launch_go2_sim.sh rviz:=false gui:=false
```

Submodule repair:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```
