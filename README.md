# LeWMQuad-v3 Go2 Simulator Bring-up

This repository targets a Unitree Go2 simulation stack for LeWMQuad-v3 data
generation and later H-JEPA training.

The current simulator base is:

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

- `third_party/unitree_go2_ros2`: selected upstream Go2 simulator, description,
  Gazebo launch, and CHAMP controller packages.
- `lewm_go2_control`: LeWM-specific ROS 2 message and service interfaces.
- `config/go2_platform_manifest.yaml`: pinned platform, camera, controller, and
  no-data gates.
- `config/go2_primitive_registry.yaml`: initial command primitive bank.
- `docs/go2_genesis_gazebo_ros2_bringup_plan.md`: full simulator and rendering
  plan.
- `docs/upstream_go2_sim_audit.md`: upstream fit and license audit.
- `scripts/`: install, alignment, build, launch, and smoke-test helpers.

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
manager, and the Gazebo foot-contact aggregator by default:

```text
/lewm/go2/command_block -> /cmd_vel
/lewm/go2/executed_command_block
/odom -> /lewm/go2/base_state
/rgb_image -> /lewm/go2/camera_info
/lewm/go2/reset (service) -> /lewm/go2/reset_event
/foot_contacts (CHAMP gait phase) -> /lewm/go2/foot_contacts
```

Disable them only when debugging the stock upstream stack:

```bash
scripts/launch_go2_sim.sh lewm_adapter:=false
scripts/launch_go2_sim.sh lewm_base_state:=false
scripts/launch_go2_sim.sh lewm_camera_info:=false
scripts/launch_go2_sim.sh lewm_reset:=false
scripts/launch_go2_sim.sh lewm_foot_contacts:=false
```

To pass through launch arguments:

```bash
scripts/launch_go2_sim.sh rviz:=false gui:=true
scripts/launch_go2_sim.sh rviz:=true gui:=false
```

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

LeWM base state publisher smoke test:

```bash
scripts/ros2_go2.sh ros2 topic echo /lewm/go2/base_state --once
```

The publisher converts the EKF-fused `/odom` stream into `BaseState`, preserving
the world-frame pose, rotating the body-frame twist into the world frame, and
emitting roll/pitch/yaw alongside the raw `(x, y, z, w)` quaternion.

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
used to split training windows on episode boundaries. When `use_spawn_pose`
is true it also shells out to `gz service -s /world/default/set_pose` to
teleport the Go2 model in Gazebo; the service response reports whether the
teleport succeeded, and the published `ResetEvent` carries the requested
pose for downstream auditing. Set `enable_teleport:=false` on the node
parameters to disable the teleport path while keeping the bookkeeping.

Limitations to know about:

- The teleport sets the entity pose only; linear and angular velocities are
  not explicitly zeroed, and the joint trajectory in progress is not
  paused. CHAMP's kinematic odometry never tracks a teleport (it is pure
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
3. `/joint_states`, `/imu/data`, `/odom`, `/tf`, and `/rgb_image` publish with
   simulation timestamps.
4. RViz can visualize the TF tree and robot state when launched with
   `rviz:=true`.
5. A short ROS bag can capture the required topics without timestamp gaps.

Example smoke bag:

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 bag record -s mcap -o go2_bringup_smoke \
  /clock /tf /joint_states /imu/data /odom /rgb_image /cmd_vel
```

## Known Blockers Before Data Generation

- License: the selected upstream repo is technically suitable but has incomplete
  license metadata for `unitree_go2_description` and `unitree_go2_sim`. Use it
  for local evaluation until the license is clarified or replaced.
- World generation: the upstream world is only the stock bring-up world plus a
  small set of boxes/cylinders for visual variety. Canonical LeWM scene
  manifests and Gazebo/Genesis exporters still need implementation.
- No explicit gait-endurance blocker at the moment: upstream CHAMP +
  bullet-featherstone delivers ~30+ s of sustained `/cmd_vel` walking on
  this workstation, comfortably above the data spec's 16-24 s per-episode
  minimum
  ([docs/fresh_retrain_data_spec.md](docs/fresh_retrain_data_spec.md)).
  This claim depends on keeping the gz `contact-system` plugin and per-foot
  `<sensor type="contact">` blocks out of the spawned model -- see the
  foot-contacts section above for why those break the gait. If a future
  slice needs sensor-derived foot forces, that path needs to be revisited
  (force_torque on the foot joint, an alternative physics backend, or a
  different locomotion stack).

Resolved on the current branch (kept here as a paper trail; the live state
lives in the launch and the smoke tests above):

- Camera info: `/lewm/go2/camera_info` is now published from the manifest.
- Foot contacts: gz contact sensors + the LeWM aggregator publish
  `/lewm/go2/foot_contacts`.
- Reset semantics: `/lewm/go2/reset` + `/lewm/go2/reset_event` advance
  monotonic episode counters and optionally teleport via `gz set_pose`.

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
