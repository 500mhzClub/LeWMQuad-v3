# Upstream Go2 Simulator Selection Audit

Date: 2026-05-13

## Selected Repository

- Repository: `https://github.com/khaledgabr77/unitree_go2_ros2`
- Local path: `third_party/unitree_go2_ros2`
- Pinned commit: `29bce68480dcc3d3bac8cc0cac983f8ac951e8e3`
- Selection status: technically selected, blocked from training use until
  license and build audits pass.

## Why This Is the Best Technical Fit

This repository is the closest match to the LeWMQuad-v3 platform requirements:

- Targets Ubuntu 24.04, ROS 2 Jazzy, and Gazebo Harmonic.
- Provides `unitree_go2_description` with Go2 URDF/Xacro, meshes, inertials,
  collision geometry, Gazebo plugins, camera, IMU, and lidar definitions.
- Provides `unitree_go2_sim` with `unitree_go2_launch.py`, Gazebo launch,
  ROS 2 control config, gait config, RViz config, and default SDF world.
- Includes CHAMP packages (`champ`, `champ_base`, `champ_msgs`) for blind
  quadruped locomotion.
- Accepts high-level body velocity commands through `/cmd_vel`, matching our
  LeWM action vector `[vx_body_mps, vy_body_mps, yaw_rate_radps]`.

## Relevant Upstream Paths

- Launch: `unitree_go2_sim/launch/unitree_go2_launch.py`
- Robot model: `unitree_go2_description/urdf/unitree_go2_robot.xacro`
- Gazebo plugins and camera: `unitree_go2_description/urdf/unitree_go2_gazebo.xacro`
- Gait config: `unitree_go2_sim/config/gait/gait.yaml`
- ROS 2 control config: `unitree_go2_sim/config/ros_control/ros_control.yaml`
- Joint map: `unitree_go2_sim/config/joints/joints.yaml`
- Link map: `unitree_go2_sim/config/links/links.yaml`

## Interface Notes

The upstream launch starts:

- `robot_state_publisher`
- Gazebo Harmonic through `ros_gz_sim`
- `ros_gz_bridge`
- `champ_base/quadruped_controller_node`
- `champ_base/state_estimation_node`
- `robot_localization` EKF nodes
- controller spawners for joint states and effort trajectory control

Important topics observed in source:

- Command input: `/cmd_vel`
- Joint state: `/joint_states`
- IMU: `/imu/data`
- Odometry: `/odom`, `odom/raw`, `odom/local`
- RGB camera: `/rgb_image`
- D455 camera topics: `/d455/image`, `/d455/depth_image`, `/d455/points`,
  `/d455/camera_info`
- Lidar: `/velodyne_points/points`, `/unitree_lidar/points`
- Foot contacts from CHAMP: `foot_contacts`, but the pinned upstream launch sets
  `publish_foot_contacts: False`, so LeWM must enable this or add Gazebo contact
  sensors before training.
- Joint trajectory output:
  `/joint_group_effort_controller/joint_trajectory`

The upstream gait limits at the pinned commit are conservative:

- `max_linear_velocity_x: 0.3`
- `max_linear_velocity_y: 0.25`
- `max_angular_velocity_z: 0.5`
- `stance_duration: 0.25`
- `nominal_height: 0.225`

The LeWM primitive registry has been clipped to these initial limits. Faster
commands should remain disabled until the gait config is deliberately retuned and
the executed-command histograms show stable distinct motion.

The upstream mono RGB camera is mounted on `camera_link`, publishes `/rgb_image`,
uses 640x480 native resolution, horizontal FOV `1.367` rad, near plane `0.05`,
and far plane `200`. The launch does not bridge a matching mono camera-info
topic at the pinned commit, so the LeWM adapter must either publish
manifest-derived intrinsics or switch to the D455 camera topics.

## License Audit

Current status: blocked pending clarification.

Findings at pinned commit:

- No top-level `LICENSE` file was found.
- `champ/package.xml`, `champ_base/package.xml`, and `champ_msgs/package.xml`
  declare `BSD`.
- `unitree_go2_description/package.xml` declares `TODO: License declaration`.
- `unitree_go2_sim/package.xml` declares `TODO: License declaration`.
- The README has a `## License` heading with no license text below it.
- README acknowledgements say the Go2 robot description is based on
  `unitreerobotics/unitree_ros`, and the controller is based on CHAMP.

Consequence:

- We may use the submodule for local evaluation and audit, but we should not
  redistribute, modify, or build project commitments around it as a clean
  dependency until the license is clarified.
- If the license cannot be clarified, fallback options are:
  1. Use `Unitree-Go2-Robot/go2_robot` for Apache-2.0 ROS 2 interfaces and
     port the simulation pieces to Jazzy/Harmonic.
  2. Use official `unitreerobotics/unitree_ros` Go2 model assets where license
     permits, and keep CHAMP from its original BSD source.
  3. Treat `khaledgabr77/unitree_go2_ros2` as an architectural reference only and
     recreate the minimal glue using clearly licensed upstream assets.

## Required Next Audits

Before data generation:

- Confirm upstream license with the maintainer or replace with a clearly
  licensed source.
- Build under the target stack: Ubuntu 24.04, ROS 2 Jazzy, Gazebo Harmonic.
- Verify `xacro` expansion of `unitree_go2_robot.xacro`.
- Verify launch of `unitree_go2_sim unitree_go2_launch.py`.
- Verify `/cmd_vel` produces distinct forward, backward, yaw, and arc motion.
- Verify `/rgb_image` camera pose, FOV, resolution, near/far plane, frame
  alignment, and camera-info/intrinsics source against the LeWM camera manifest.
- Verify foot-contact topic quality after enabling it; if CHAMP contacts are
  synthetic or unavailable, determine whether Gazebo contact sensors are needed
  for collision/recovery labels.
- Verify reset support. Upstream launch does not provide LeWM reset/episode
  semantics; we still need our reset manager.
- Verify bag capture and conversion into `raw_rollout`.
