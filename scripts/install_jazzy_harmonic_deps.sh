#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/ros_env.sh
source "$SCRIPT_DIR/ros_env.sh"

lewm_check_noble_os

if ! command -v sudo >/dev/null 2>&1; then
  echo "sudo is required to install ROS/Gazebo apt packages." >&2
  exit 1
fi

packages=(
  ros-jazzy-desktop
  ros-dev-tools
  python3-rosdep
  python3-colcon-common-extensions
  python3-vcstool
  ros-jazzy-ros-gz
  ros-jazzy-gz-ros2-control
  ros-jazzy-xacro
  ros-jazzy-robot-localization
  ros-jazzy-ros2-control
  ros-jazzy-ros2-controllers
  ros-jazzy-controller-manager
  ros-jazzy-joint-state-broadcaster
  ros-jazzy-joint-trajectory-controller
  ros-jazzy-joint-state-publisher
  ros-jazzy-joint-state-publisher-gui
  ros-jazzy-velodyne
  ros-jazzy-velodyne-description
  ros-jazzy-realsense2-description
  ros-jazzy-teleop-twist-keyboard
)

echo "Installing Jazzy/Harmonic simulator dependencies..."
sudo apt update
sudo apt install -y "${packages[@]}"

if [[ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]]; then
  sudo rosdep init
fi

rosdep update

echo
echo "Installed simulator dependencies."
echo "Next: scripts/check_ros_gz_alignment.sh"

