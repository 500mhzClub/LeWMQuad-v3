# Go2 ROS 2 Scaffold Runbook

Date: 2026-05-13

This runbook covers only the initial scaffold. It is not a data-generation
pipeline yet. We have selected an upstream Go2 simulator base and should not
continue building a full robot description or Gazebo stack from scratch.

Selected upstream base:

- `https://github.com/khaledgabr77/unitree_go2_ros2`
- ROS 2 Jazzy / Ubuntu 24.04 / Gazebo Harmonic target.
- Go2 description package: `unitree_go2_description`.
- Simulation package: `unitree_go2_sim`.
- Locomotion controller: CHAMP through `/cmd_vel`.

The repository is now cloned as a git submodule at
`third_party/unitree_go2_ros2` and pinned to
`29bce68480dcc3d3bac8cc0cac983f8ac951e8e3`. It still needs license
clarification and a target-environment build audit before it can be treated as
training-ready.

## Files Added

- `config/go2_platform_manifest.yaml`: platform, timing, camera, backend, and
  no-data gates.
- `config/go2_primitive_registry.yaml`: initial command primitive registry.
- `lewm_go2_control`: internal ROS 2 messages and services.
- `third_party/unitree_go2_ros2`: selected upstream Go2 simulator submodule.
- `docs/upstream_go2_sim_audit.md`: technical and license audit notes for the
  pinned upstream commit.

## Expected Workspace Layout

Build these packages from the repository root as a ROS 2 workspace:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

If the repository is later nested under a larger workspace, keep these packages
at workspace package-root depth or add the repository root to `COLCON_IGNORE`
rules deliberately. Do not duplicate the message package in multiple overlays.

The upstream submodule adds packages at `third_party/unitree_go2_ros2`. A full
workspace build will include those packages unless `third_party` is ignored.
That is intentional for the target Jazzy/Harmonic environment, but local
developer machines without `ros_gz`, `xacro`, `gz_ros2_control`, and
`robot_localization` will not be able to launch or build the full stack.

## First Bring-up Command

The next real bring-up should use the selected upstream launch first, not our
placeholder launch. After ROS 2 Jazzy, Gazebo Harmonic, `ros_gz`, `xacro`, and
controller packages are installed:

```bash
ros2 launch unitree_go2_sim unitree_go2_launch.py
```

Record the smoke topics with our LeWM recorder once implemented. Until then,
use `ros2 bag record` directly:

```bash
ros2 bag record -s mcap -o go2_bringup_smoke \
  /clock /tf /joint_states /imu/data /odom /rgb_image \
  /cmd_vel
```

After the LeWM adapter is running, add `/lewm/go2/foot_contacts`,
`/lewm/go2/command_block`, `/lewm/go2/executed_command_block`,
`/lewm/go2/base_state`, `/lewm/go2/reset_event`, and `/lewm/episode_info`.

## Before Any Training Data

Build and audit the selected upstream stack. Then update:

- `config/go2_platform_manifest.yaml`
- upstream commit SHA and license status
- upstream Go2 description package path
- upstream CHAMP gait and control config paths
- LeWM camera mount override, if different from upstream

Then implement only the LeWM-specific adapter nodes behind the
`lewm_go2_control` interfaces:

- command block adapter
- `/cmd_vel` bridge to upstream CHAMP controller
- executed-command logger/reconstructor
- base-state publisher
- foot-contact publisher
- reset manager
- feature-check runner

The first accepted bag must show consistent `/clock`, `/tf`, `/joint_states`,
IMU, camera, reset events, requested command blocks, and executed command
blocks.
