# LeWMQuad-v3 Go2 / ROS 2 / Gazebo / Genesis Bring-up and Rendering Plan

Date: 2026-05-13

## 1. Purpose

This document turns the existing simulator-agnostic v3 documents into a
platform-specific implementation plan for:

- A Unitree Go2 embodiment.
- ROS 2 control and observability.
- Gazebo as the integration simulator.
- Genesis as the high-throughput world instantiation and rendering path, where
  it passes parity against Gazebo.
- A LeWM/H-JEPA data pipeline where the JEPA controls only generic command
  blocks such as forward, backward, turn, hold, recovery, and optional jump
  events. The JEPA does not replace the blind locomotion controller.

The first deliverable is not training. The first deliverable is a fully
instrumented Go2 in simulation, controllable through ROS 2, with all required
state, sensors, resets, and command execution logs validated. Only after that
do we generate data.

## 2. Review of Current Local Documents

The current docs are valuable but intentionally abstract:

- `docs/fresh_retrain_data_spec.md` is the source of truth for dataset schema,
  scene counts, action-block coverage, render-quality gates, reset integrity,
  and training-readiness criteria.
- `docs/v3_hjepa_plan.md` is the source of truth for the H-JEPA diagnostic
  phases, the action-block abstraction, the privileged-leak rule, and the
  portability requirements.
- `docs/architecture_talking_points.md` captures the v2 architecture and
  failure analysis. It is useful for avoiding repeated mistakes, but it still
  describes the Mini Pupper / v2 implementation and should not be treated as a
  Go2 implementation guide.

What is missing before implementation:

- A concrete Go2 robot interface.
- A ROS 2 package layout and topic/service contract.
- A Gazebo instantiation template.
- A Genesis scene/rendering template using the same canonical world metadata.
- A blind locomotion policy contract.
- A bring-up checklist proving the dog, sensors, commands, reset flow, and
  data capture work before any LeWM data generation begins.

## 3. Platform Decisions

### 3.1 Default Versions

Use this as the default simulation stack unless a hardware constraint forces a
change:

- OS: Ubuntu 24.04.
- ROS 2: Jazzy.
- Gazebo: Harmonic.
- DDS: Cyclone DDS where Unitree compatibility matters, otherwise the ROS 2
  default is acceptable for sim-only work.
- Dataset storage: raw ROS bags for bring-up/debug, converted into immutable
  HDF5 or Zarr data products for training.

Reasoning:

- ROS 2 Jazzy officially targets Ubuntu 24.04 and pairs with Gazebo Harmonic.
- Gazebo Harmonic is an LTS release.
- Unitree's official ROS 2 package currently documents Ubuntu 22.04 / Humble as
  the recommended tested ROS 2 path, so the Go2 bridge must be isolated behind
  our own adapter instead of leaking Unitree-specific messages throughout the
  data pipeline.

If real-robot development must stay on Humble, keep the same internal
interfaces and build the Unitree hardware bridge in a Humble overlay. Do not
let Humble-only details define the training schema.

### 3.1.1 Selected Upstream Base

Use `khaledgabr77/unitree_go2_ros2` as the selected upstream base, pending clone,
license, and build audit. It is the closest match to our target because it is
documented as a ROS 2 Jazzy / Ubuntu 24.04 / Gazebo Harmonic Go2 simulator with:

- `unitree_go2_description` for URDF/Xacro, meshes, sensors, and Gazebo
  description.
- `unitree_go2_sim` for launch, Gazebo, ROS 2 control, RViz, and gait/config
  files.
- CHAMP locomotion driven by `/cmd_vel`, which maps cleanly to our
  `[vx_body_mps, vy_body_mps, yaw_rate_radps]` command-block contract.
- Simulated IMU, joint states, lidar/camera options, odometry/state estimation,
  and Gazebo Harmonic integration.

Do not continue developing `go2_description` or a full `lewm_go2_gazebo`
replacement unless the upstream audit fails. Our code should wrap the upstream
stack with LeWM-specific command-block logging, reset/episode management,
scene-manifest world insertion, and dataset conversion.

Keep Unitree's official `unitreerobotics/unitree_ros2` as the real-robot API
reference and hardware-parity bridge, not as the primary Gazebo simulator. It
is official and important for Go2 sport-mode semantics, but it is not a complete
Gazebo/Harmonic simulation stack.

### 3.2 Backend Roles

Use one canonical scene manifest and two backend exporters:

1. `scene_manifest -> Gazebo SDF`: integration, ROS 2 launch, controller
   bring-up, sensor validation, reset semantics, and parity checks.
2. `scene_manifest -> Genesis scene`: high-throughput rollout/replay/rendering
   only after parity checks pass.

Do not make either Gazebo SDF or Genesis code the source of truth for topology.
The scene manifest is the source of truth because derived labels, train/test
splits, graph distances, landmarks, and loop-closure supervision must be
regenerated deterministically without replaying physics.

### 3.3 Controller Boundary

The LeWM/H-JEPA stack commands a blind locomotion system. It does not output
joint targets directly.

The default command vector remains:

```text
[vx_body_mps, vy_body_mps, yaw_rate_radps]
```

Action blocks are fixed-length arrays of these commands:

```text
block = K x 3
active_block = flatten(block)
```

Named behaviors such as `stand`, `recover`, `stop`, `jump`, and `front_jump`
are command events in the adapter layer. They are not added to the LeWM action
space unless the simulated locomotion policy can execute them reliably and the
dataset contains enough support.

## 4. System Architecture

The implementation should have four layers.

### 4.1 Canonical World Layer

Package target:

```text
lewm_worlds/
  manifests/
  scene_families/
  exporters/
    to_gazebo_sdf.py
    to_genesis.py
  labels/
    topology.py
    landmarks.py
    traversability.py
```

Contract:

```python
def build_scene_manifest(scene_seed: int, family: str) -> SceneManifest:
    """Return topology, geometry, materials, landmarks, and physics params."""

def export_gazebo_sdf(manifest: SceneManifest, out_dir: Path) -> Path:
    """Write world.sdf plus model assets."""

def export_genesis_scene(manifest: SceneManifest) -> GenesisSceneSpec:
    """Return objects and materials for deterministic Genesis construction."""
```

Manifest requirements:

- `scene_id`, `topology_seed`, `visual_seed`, `physics_seed`.
- Scene family and difficulty tier.
- Obstacles, walls, ramps, steps, rough terrain, landmarks, and distractors.
- Materials and texture IDs.
- World bounds.
- Navigable graph: node centers, adjacency, widths, graph type, dead ends,
  cycle count, landmark node IDs.
- Camera-validity constraints: minimum wall thickness, near/far plane bounds,
  camera clearance thresholds.

### 4.2 Robot and Control Layer

Package target:

```text
third_party/unitree_go2_ros2/
  unitree_go2_description/
  unitree_go2_sim/
  champ/
  champ_base/
  champ_msgs/

lewm_go2_control/
  msg/
  srv/
  src/
    command_block_adapter.py
    cmd_vel_adapter.py
    executed_command_logger.py
    safety_monitor.py
    reset_manager.py
```

The upstream submodule owns the Go2 robot description, Gazebo launch, ROS 2
control, CHAMP locomotion, and standard sensor topics. `lewm_go2_control` owns
only LeWM-specific translation. The rest of the LeWM stack must depend on the
internal messages below, not directly on CHAMP or Gazebo-specific topics.

Core ROS topics:

```text
/lewm/go2/command_block          lewm_go2_control/msg/CommandBlock
/lewm/go2/executed_command_block lewm_go2_control/msg/ExecutedCommandBlock
/lewm/go2/mode                   lewm_go2_control/msg/Go2ModeState
/lewm/go2/base_state             lewm_go2_control/msg/BaseState
/lewm/go2/foot_contacts          lewm_go2_control/msg/FootContacts
/lewm/go2/reset_event            lewm_go2_control/msg/ResetEvent
/lewm/episode_info               lewm_go2_control/msg/EpisodeInfo

/joint_states                    sensor_msgs/msg/JointState
/tf                              tf2_msgs/msg/TFMessage
/imu/data                        sensor_msgs/msg/Imu
/rgb_image                       sensor_msgs/msg/Image
/d455/camera_info                sensor_msgs/msg/CameraInfo optional
/lewm/go2/camera_info            sensor_msgs/msg/CameraInfo optional if derived
/velodyne_points/points          sensor_msgs/msg/PointCloud2 optional
/clock                           rosgraph_msgs/msg/Clock
```

Core services:

```text
/lewm/go2/set_mode               SetGo2Mode.srv
/lewm/go2/reset                  ResetEpisode.srv
/lewm/go2/set_policy             SetPolicy.srv
/lewm/go2/run_feature_check      RunFeatureCheck.srv
```

The adapter must publish both requested and executed commands. LeWM trains on
the executed block.

### 4.3 Gazebo Integration Layer

Package target:

```text
third_party/unitree_go2_ros2/
  unitree_go2_sim/
    launch/unitree_go2_launch.py
    config/gait/gait.yaml
    config/ros_control/ros_control.yaml
  unitree_go2_description/
    urdf/unitree_go2_robot.xacro
    urdf/unitree_go2_gazebo.xacro
    worlds/default.sdf

lewm_worlds/exporters/to_gazebo_sdf.py
lewm_go2_control/src/*
```

Responsibilities:

- Launch the selected upstream Gazebo Harmonic stack first.
- Override the upstream world with exported LeWM scene manifests once the stock
  launch passes.
- Start the LeWM command-block adapter that converts fixed-length blocks to
  `/cmd_vel`.
- Start the LeWM executed-command logger, reset manager, recorders, and health
  monitors.

Gazebo is the authoritative environment for ROS 2 integration testing. A
rollout is not accepted until Gazebo proves that the exact command/state
contract works under ROS 2.

### 4.4 Genesis Rendering Layer

Package target:

```text
lewm_genesis/
  scene_builder.py
  go2_adapter.py
  render_replay.py
  batch_renderer.py
  parity_checks.py
```

Responsibilities:

- Build a Genesis scene from the same manifest used by Gazebo.
- Load the same Go2 URDF/MJCF asset or a documented Genesis-compatible
  equivalent.
- Attach the same camera model with matching intrinsics and extrinsics.
- Replay raw rollouts or run Genesis-native rollouts when parity is accepted.
- Render RGB, depth, segmentation, and camera-validity diagnostics.
- Write `rendered_vision` without changing reset arrays or executed commands.

Genesis is accepted as the high-throughput data path only after it matches the
Gazebo/ROS 2 contract on geometry, camera pose, command timing, reset semantics,
and basic controller response.

## 5. Step-by-Step Bring-up Plan

### Phase 0: Asset and Policy Lock

Goal: identify every external artifact before building code around it.

Tasks:

1. Choose the Go2 description source: official Unitree description if usable,
   otherwise a clearly versioned third-party URDF/MJCF with checked inertials,
   collision geometry, joint limits, and sensor frames.
2. Choose the locomotion backend:
   - preferred for sim: frozen blind Go2 locomotion policy, exported as ONNX or
     TorchScript, consuming proprioception plus `[vx, vy, yaw_rate]`;
   - fallback for hardware/API parity: Unitree Sport API adapter, exposing
     `Move(vx, vy, vyaw)`, stop, stand, recovery, gait, and optional action
     events;
   - not acceptable for data generation: a kinematic body teleport or a
     controller that cannot produce joint/proprio/foot-contact state.
3. Freeze command timing:
   - policy tick, for example 50 Hz;
   - high-level command tick, for example 10 Hz;
   - LeWM macro block `K = 5` command ticks unless Go2 dynamics require
     retuning.
4. Freeze camera mount:
   - parent link;
   - body-frame translation and rotation;
   - FOV, resolution, near/far planes;
   - rolling/global shutter assumption;
   - RGB encoding and timestamp policy.

Exit gate:

- A checked-in `go2_platform_manifest.yaml` names the robot asset, controller
  artifact, camera parameters, command timing, ROS 2 distro, Gazebo version, and
  Genesis version.

### Phase 1: Internal ROS 2 Interfaces

Goal: create a stable interface that hides Gazebo, Genesis, and Unitree details.

Define these messages.

`CommandBlock.msg`:

```text
std_msgs/Header header
uint32 sequence_id
uint32 block_size
float32 command_dt
string primitive_name
float32[] vx
float32[] vy
float32[] yaw_rate
string event_name
bool event_allowed_in_training
```

`ExecutedCommandBlock.msg`:

```text
std_msgs/Header header
uint32 sequence_id
uint32 block_size
float32 command_dt
string primitive_name
float32[] requested_vx
float32[] requested_vy
float32[] requested_yaw_rate
float32[] executed_vx
float32[] executed_vy
float32[] executed_yaw_rate
bool clipped
bool safety_overridden
string controller_mode
```

`BaseState.msg`:

```text
std_msgs/Header header
geometry_msgs/Pose pose_world
geometry_msgs/Twist twist_world
geometry_msgs/Twist twist_body
float32[4] quat_world_xyzw
float32 roll
float32 pitch
float32 yaw
```

`FootContacts.msg`:

```text
std_msgs/Header header
bool fl
bool fr
bool rl
bool rr
float32 fl_force
float32 fr_force
float32 rl_force
float32 rr_force
```

Exit gate:

- A dummy node can publish a `CommandBlock`.
- The adapter echoes an `ExecutedCommandBlock`.
- A bag records all required topics with consistent timestamps.

### Phase 2: Upstream Gazebo Go2 Bring-up

Goal: instantiate the dog in Gazebo with ROS 2 visibility.

Use the selected upstream repository rather than creating a local Go2 simulator
from scratch:

```text
third_party/unitree_go2_ros2/
  unitree_go2_description/
  unitree_go2_sim/
  champ/
  champ_base/
  champ_msgs/
```

Pinned commit:

```text
29bce68480dcc3d3bac8cc0cac983f8ac951e8e3
```

Initial upstream launch:

```bash
ros2 launch unitree_go2_sim unitree_go2_launch.py rviz:=false
```

Important upstream files to audit:

- `unitree_go2_sim/launch/unitree_go2_launch.py`
- `unitree_go2_sim/config/gait/gait.yaml`
- `unitree_go2_sim/config/ros_control/ros_control.yaml`
- `unitree_go2_description/urdf/unitree_go2_robot.xacro`
- `unitree_go2_description/urdf/unitree_go2_gazebo.xacro`
- `unitree_go2_description/worlds/default.sdf`

The upstream bridge and Gazebo plugins must be verified against the installed
`ros_gz_bridge` message conversion table. Anything unsupported by the bridge
should be emitted by a native ROS 2 node instead of forced through Gazebo
Transport.

Feature checks:

- Model spawns without exploding.
- Joint axes and limits match the asset source.
- `robot_state_publisher` publishes the expected TF tree.
- IMU, joint states, camera image, camera info or manifest-derived intrinsics,
  and optional lidar publish.
- Foot-contact or foot-force estimates publish through CHAMP, Gazebo contact
  sensors, or the LeWM adapter.
- Reset service returns the dog to a deterministic pose.
- Blind policy can stand, walk forward, walk backward, yaw left/right, stop,
  and recover.
- Dynamic behaviors such as jump are either validated or marked unsupported in
  the primitive registry.

Exit gate:

- `ros2 launch unitree_go2_sim unitree_go2_launch.py rviz:=false` brings up
  the model, CHAMP controller, required topics, and a successful feature check.

### Phase 3: Locomotion Adapter

Goal: ensure every command block has a measurable physical interpretation.

Adapter responsibilities:

1. Accept `CommandBlock`.
2. Validate primitive name, vector lengths, speed bounds, and event permission.
3. Apply rate limits and safety bounds.
4. Send commands to the backend:
   - Unitree-style high-level backend: `Move(vx, vy, yaw_rate)` plus mode/event
     calls such as stand, stop, recovery, gait, and optional jump.
   - Policy backend: build the policy observation and publish joint targets.
5. Publish `ExecutedCommandBlock`.
6. Publish controller mode and any clipping/safety override.

Primitive registry:

```yaml
hold:
  type: velocity_block
  vx: 0.0
  vy: 0.0
  yaw_rate: 0.0
  train: true

forward_slow:
  type: velocity_block
  vx: 0.15
  vy: 0.0
  yaw_rate: 0.0
  train: true

forward_medium:
  type: velocity_block
  vx: 0.25
  vy: 0.0
  yaw_rate: 0.0
  train: true

backward:
  type: velocity_block
  vx: -0.20
  vy: 0.0
  yaw_rate: 0.0
  train: true

yaw_left:
  type: velocity_block
  vx: 0.0
  vy: 0.0
  yaw_rate: 0.45
  train: true

yaw_right:
  type: velocity_block
  vx: 0.0
  vy: 0.0
  yaw_rate: -0.45
  train: true

recover:
  type: mode_event
  event: recovery_stand
  train: true

jump:
  type: mode_event
  event: front_jump
  train: false
  enable_only_after_policy_validation: true
```

Exit gate:

- Command histograms show distinct executed movement for every trainable
  primitive.
- Unsupported or unsafe primitives are excluded from the LeWM planner action
  space.

### Phase 4: Canonical World Generator

Goal: generate diverse worlds once and instantiate them in both simulators.

Scene families should inherit the proportions from
`docs/fresh_retrain_data_spec.md`:

- Open obstacle fields.
- Local composite motifs.
- Small, medium, and large enclosed mazes.
- Loop and alias stress mazes.
- Rough/local dynamics variants.
- Visual/sensor stress variants.

Implementation rules:

- All walls, obstacles, landmarks, surfaces, and graph nodes are emitted from
  the manifest.
- The manifest owns random seeds and object IDs.
- Gazebo and Genesis exporters may approximate backend-specific geometry, but
  they must preserve graph topology, camera occlusion, traversability, and
  landmark line of sight.
- Labels are computed from the manifest and rollout state, not from renderer
  pixels.

Exit gate:

- For a fixed seed, both exporters produce the same graph node count, edge
  count, dead-end list, landmark node IDs, and world bounds.
- A visual inspection script renders a top-down debug image for both backends.

### Phase 5: Genesis Instantiation and Rendering Template

Goal: build and render the same world in Genesis.

Template:

```python
import genesis as gs


def build_genesis_scene(manifest, platform):
    gs.init(backend=platform.backend, seed=manifest.physics_seed)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=platform.physics_dt,
            gravity=(0.0, 0.0, -9.81),
        ),
        show_viewer=False,
        renderer=gs.renderers.Rasterizer(),
    )

    scene.add_entity(gs.morphs.Plane())

    for wall in manifest.walls:
        scene.add_entity(
            gs.morphs.Box(
                pos=wall.pos,
                size=wall.size,
                quat=wall.quat,
            ),
            surface=wall.surface,
            name=wall.id,
        )

    robot = scene.add_entity(
        gs.morphs.URDF(
            file=platform.go2_urdf,
            pos=manifest.spawn.pos,
            quat=manifest.spawn.quat_wxyz,
            fixed=False,
        ),
        name="go2",
    )

    camera = scene.add_camera(
        res=platform.camera.resolution,
        pos=platform.camera.initial_world_pos,
        lookat=platform.camera.initial_world_lookat,
        fov=platform.camera.fov_deg,
        near=platform.camera.near_m,
        far=platform.camera.far_m,
        GUI=False,
    )

    scene.build(n_envs=platform.n_envs)
    return scene, robot, camera
```

Rendering requirements:

- RGB is mandatory.
- Depth is mandatory for camera-validity audits even if it is not a model
  input.
- Segmentation is strongly recommended for landmark and obstacle audit clips.
- Camera timestamps must align with command and proprio timestamps.
- Invalid frames are flagged; contact frames are fixed, not silently dropped.

Exit gate:

- A 10-scene Genesis render smoke test produces valid RGB/depth frames,
  matching reset arrays, and invalid-frame rates below the thresholds in
  `docs/fresh_retrain_data_spec.md`.

### Phase 6: Gazebo / Genesis Parity

Goal: prove that Genesis data is acceptable for training if it is used.

Run the same seed, spawn, command script, and camera spec in both backends.

Compare:

- Body displacement per primitive.
- Yaw response per primitive.
- Stop latency.
- Recovery behavior.
- Foot-contact timing at a coarse level.
- Camera pose relative to base.
- Landmark visibility and occlusion.
- Render invalid-frame rate.
- Reset reproducibility.

Acceptance:

- No primitive collapses to indistinguishable motion in either backend.
- Camera observations are qualitatively and quantitatively similar enough for
  domain randomization to cover the gap.
- If Genesis physics diverges materially from Gazebo, use Gazebo for raw
  rollouts and Genesis only for deterministic replay/rendering where replay
  state can be enforced.

### Phase 7: Data Capture

Goal: capture raw rollouts without training-specific shortcuts.

Record:

- Requested command blocks.
- Executed command blocks.
- Controller mode and safety overrides.
- Base pose/twist in world and body frames.
- Joint position, velocity, and command.
- Last low-level action.
- IMU.
- Foot contacts/forces.
- Camera image and camera info.
- Optional lidar.
- Collision/contact diagnostics.
- Reset events and episode IDs.
- Scene manifest hash and seed.

Preferred bring-up flow:

1. Record ROS bags in MCAP for short smoke and pilot rollouts.
2. Convert bags to immutable `raw_rollout` chunks.
3. Render or copy images into `rendered_vision`.
4. Compute `derived_labels` offline from the manifest and rollout state.
5. Build manifests for train/val/test only after scene IDs are fixed.

Exit gate:

- A bag-to-HDF5 conversion round trip preserves timestamp order, reset arrays,
  command arrays, and scene metadata.

### Phase 8: Smoke Dataset

Goal: validate the entire path before scaling.

Smoke tier:

- 10 scenes.
- 8 episodes per scene.
- 200 to 400 raw command ticks per episode.
- All primitive families represented at least once.
- At least one reset per scene.
- At least one contact/recovery case.

Pilot tier:

- 100 scenes.
- 16 to 32 episodes per scene.
- 800 raw command ticks per episode.
- Initial train/val split by scene.
- All render-quality, reset-integrity, and command-support audits run.

Exit gate:

- No cross-reset windows.
- Executed-command histograms are sane.
- Camera invalid-frame rates are below threshold.
- Scene-level split is enforced.
- Derived labels populate at least 99% of graph-scene steps.

### Phase 9: Full Dataset Generation

Only start the full run after the pilot tier passes.

Use the counts and gates in `docs/fresh_retrain_data_spec.md`:

- Full target: 2400 train scenes, 300 validation scenes, 300 test-ID scenes,
  plus hard/transfer splits as needed.
- Minimum serious target: 1000 train scenes, 150 validation scenes, 150 test-ID
  scenes, 150 test-hard scenes.
- Train on executed command blocks.
- Preserve scene-level splits.
- Cap per-scene windows to prevent large scenes dominating.
- Commit dataset manifests, hashes, and audits with the corpus.

## 6. Unitree Go2 Feature Readiness Matrix

Before data generation, the simulated dog must pass this matrix.
For capabilities exposed by the selected Go2 backend, the standard is either
"passes in simulation" or "explicitly marked unsupported with a reason and
excluded from training." Silent partial support is not acceptable.

| Feature | Required for training | Backend contract | Acceptance |
| --- | --- | --- | --- |
| Spawn/reset | Yes | Gazebo and Genesis | Deterministic reset pose and no joint explosion |
| Stand/balance | Yes | Adapter mode | Stable for 30 s sim time |
| Stop/hold | Yes | Velocity block | Displacement below threshold |
| Forward/backward | Yes | `[vx, 0, 0]` | Distinct executed displacement bins |
| Yaw left/right | Yes | `[0, 0, wz]` | Distinct yaw bins |
| Arc turns | Yes | `[vx, 0, wz]` | Curvature matches primitive registry |
| Lateral step | Optional | `[0, vy, 0]` | Include only if policy supports it |
| Recovery stand | Yes | Mode event | Recovers from scripted fall/contact cases |
| Sit/rise | Debug only | Mode event | Works, but excluded from planner unless needed |
| Body Euler/pose | Debug only | Mode event | Works for inspection, not a LeWM action |
| Speed/gait mode | Optional | Adapter parameter | Logged as controller mode |
| Jump/front jump | Optional | Mode event | Excluded until stable and represented in data |
| Low-level motor control | Policy-only | Joint target backend | Never used directly by JEPA |
| IMU | Yes | ROS sensor topic | Timestamp-aligned and nonzero under motion |
| Joint states | Yes | ROS sensor topic | All 12 leg joints present |
| Foot contacts | Yes | ROS/custom topic | Contact toggles under gait |
| Front camera | Yes | ROS/Gazebo and Genesis | Intrinsics/extrinsics match manifest |
| Lidar | Optional | ROS sensor topic | Recorded only if part of target claim |

## 7. Training Action-Space Policy

The primitive bank must be smaller than the physical feature matrix.

Default trainable primitives:

- hold/stop
- forward slow/medium/fast
- backward/backout
- yaw left/right
- arc left/right
- recovery
- wall-follow left/right after wall-follow teacher exists
- frontier/route-following teacher commands for collection only

Optional primitives:

- lateral left/right, only if the Go2 controller executes lateral commands
  distinctly and reliably;
- jump/front-jump, only if the simulated policy executes them safely and they
  are part of the task.

Do not expose every Unitree sport behavior to CEM. The planner should search a
small bank that is both trained and physically meaningful.

## 8. Data Product Contract

The data products remain those in `docs/fresh_retrain_data_spec.md`:

1. `raw_rollout`: ROS/Gazebo/Genesis state, commands, resets, contacts,
   proprioception, and scene metadata.
2. `rendered_vision`: egocentric RGB plus camera validity and copied raw labels.
3. `derived_labels`: graph, landmark, clearance, traversability, body-motion,
   recovery, stuck, frontier, loop, and reachability labels.

Additional Go2-specific metadata:

```yaml
robot_id: unitree_go2
robot_asset_version: ...
locomotion_policy_id: ...
locomotion_policy_hash: ...
adapter_version: ...
ros_distro: jazzy
gazebo_version: harmonic
genesis_version: ...
controller_dt: ...
command_dt: ...
physics_dt: ...
action_block_size: 5
camera_mount:
  parent_link: ...
  xyz_body: [...]
  rpy_body: [...]
```

## 9. No-Training-Until Gates

Do not begin data generation or training until all gates pass:

- Go2 model spawns in Gazebo and publishes TF, joints, IMU, camera, contacts,
  and base state.
- Go2 can be controlled through `CommandBlock`, not ad hoc backend commands.
- `ExecutedCommandBlock` is recorded and differs from requested commands when
  clipping or safety overrides occur.
- Reset events are explicit and produce reset-separated `episode_id`.
- Canonical world manifest exports to both Gazebo and Genesis.
- Camera intrinsics/extrinsics are identical in manifest, Gazebo, and Genesis.
- Genesis render smoke test passes or Genesis is demoted to a non-authoritative
  renderer until parity is fixed.
- Primitive registry excludes unsupported Go2 behaviors.
- Bag-to-training conversion preserves timestamps and reset arrays.
- Smoke and pilot datasets pass data-quality audits.

## 10. Main Risks and Mitigations

| Risk | Failure mode | Mitigation |
| --- | --- | --- |
| Unitree/Humble assumptions leak into Jazzy sim | Training code depends on vendor topics | Internal `CommandBlock` and state messages |
| Gazebo and Genesis disagree | Training data does not match integration sim | Manifest source of truth plus parity tests |
| No real blind policy artifact | Dog moves kinematically or unrealistically | Lock policy artifact before dataset work |
| Advanced sport behaviors are unstable | Jump/recovery corrupts action labels | Primitive registry with train flags |
| Commands collapse physically | JEPA sees different vectors with same outcome | Executed-command histograms and action sensitivity |
| Camera mismatch | Goal/navigation latents do not transfer | One camera manifest, backend camera tests |
| Reset contamination | Impossible transitions enter LeWM | Reset manager and zero cross-reset index gate |
| Renderer artifacts near contact | Model learns invalid wall views | Depth-based camera-validity audit |
| Scene leakage | Inflated validation metrics | Split by scene/topology seed only |
| Privileged leakage | Deployment cannot reproduce metrics | Labels-only boundary preserved |

## 11. External References Checked

- [ROS 2 Jazzy release notes](https://docs.ros.org/en/rolling/Releases/Release-Jazzy-Jalisco.html)
  document Ubuntu 24.04 as a Tier 1 platform and list Gazebo Harmonic as the
  paired Gazebo dependency.
- [Gazebo Harmonic ROS 2 overview](https://gazebosim.org/docs/harmonic/ros2_overview/),
  [ROS 2 bridge guide](https://gazebosim.org/docs/harmonic/ros2_integration/),
  [model spawning guide](https://gazebosim.org/docs/harmonic/ros2_spawn_model/),
  [SDF worlds guide](https://gazebosim.org/docs/harmonic/sdf_worlds/), and
  [URDF spawning guide](https://gazebosim.org/docs/harmonic/spawn_urdf/) define
  the Gazebo-side launch, bridge, world, and robot-spawn assumptions.
- [Unitree's official `unitree_ros2` repository](https://github.com/unitreerobotics/unitree_ros2)
  documents ROS 2 communication for Go2/B2/H1 through Cyclone DDS, including
  sport state, sport requests, and low command examples.
- [Unitree's official SDK2 Go2 `SportClient`](https://github.com/unitreerobotics/unitree_sdk2/blob/main/include/unitree/robot/go2/sport/sport_client.hpp)
  exposes the high-level methods this plan wraps: `Move(vx, vy, vyaw)`,
  `StopMove`, `BalanceStand`, `RecoveryStand`, gait/mode methods, and optional
  dynamic actions such as `FrontJump`.
- [Genesis Hello tutorial](https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/hello_genesis.html),
  [file I/O / URDF loading reference](https://genesis-world.readthedocs.io/en/latest/api_reference/utilities/file_io.html),
  [visualization and rendering guide](https://genesis-world.readthedocs.io/en/v0.3.3/user_guide/getting_started/visualization.html),
  and [camera sensor guide](https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/camera_sensors.html)
  confirm scene creation, URDF loading, camera creation, RGB/depth/segmentation
  rendering, and camera sensor backends including a batch renderer for
  high-throughput work.

## 12. Implementation Order Summary

1. Lock Go2 asset, camera, controller artifact, and platform versions.
2. Define internal ROS 2 messages/services.
3. Build Gazebo Go2 spawn and sensor template.
4. Implement command adapter and blind policy runner.
5. Implement canonical world manifests and Gazebo exporter.
6. Implement Genesis exporter and renderer from the same manifests.
7. Run Gazebo feature checks.
8. Run Genesis render smoke tests.
9. Run Gazebo/Genesis parity tests.
10. Record smoke bags and convert them to training chunks.
11. Run pilot dataset audits.
12. Generate the full corpus only after the no-training gates pass.
