#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/ros_env.sh
source "$SCRIPT_DIR/ros_env.sh"

REPO_ROOT="$(lewm_repo_root)"
EXPECTED_SUBMODULE_URL="https://github.com/khaledgabr77/unitree_go2_ros2.git"
EXPECTED_SUBMODULE_SHA="29bce68480dcc3d3bac8cc0cac983f8ac951e8e3"

lewm_check_noble_os
lewm_source_jazzy_underlay

echo "ROS_DISTRO: ${ROS_DISTRO}"

lewm_need_command ros2
lewm_need_command colcon
lewm_need_command rosdep
lewm_need_command gz

required_ros_packages=(
  ros_gz_sim
  ros_gz_bridge
  gz_ros2_control
  xacro
  robot_localization
  robot_state_publisher
  controller_manager
  joint_state_broadcaster
  joint_trajectory_controller
  realsense2_description
  velodyne_description
)

for package_name in "${required_ros_packages[@]}"; do
  lewm_need_ros_package "$package_name"
done

actual_url="$(git config --file "$REPO_ROOT/.gitmodules" --get submodule.third_party/unitree_go2_ros2.url)"
actual_sha="$(git -C "$REPO_ROOT/third_party/unitree_go2_ros2" rev-parse HEAD)"

if [[ "$actual_url" != "$EXPECTED_SUBMODULE_URL" ]]; then
  echo "Submodule URL mismatch: expected $EXPECTED_SUBMODULE_URL, got $actual_url" >&2
  exit 1
fi

if [[ "$actual_sha" != "$EXPECTED_SUBMODULE_SHA" ]]; then
  echo "Submodule SHA mismatch: expected $EXPECTED_SUBMODULE_SHA, got $actual_sha" >&2
  exit 1
fi

echo
echo "Gazebo version:"
gz sim --version

echo
echo "ROS/Gazebo alignment looks correct."

