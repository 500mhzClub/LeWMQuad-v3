#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/ros_env.sh
source "$SCRIPT_DIR/ros_env.sh"

REPO_ROOT="$(lewm_repo_root)"
ROSDEP="${ROSDEP:-1}"
CLEAN="${CLEAN:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean)
      CLEAN=1
      shift
      ;;
    --skip-rosdep)
      ROSDEP=0
      shift
      ;;
    *)
      break
      ;;
  esac
done

cd "$REPO_ROOT"

lewm_source_jazzy_underlay

git submodule sync --recursive
git submodule update --init --recursive

if [[ "$CLEAN" == "1" ]]; then
  echo "Removing build/, install/, and log/ before the Jazzy build..."
  rm -rf build install log
fi

if [[ -f build/lewm_go2_control/CMakeCache.txt ]] && grep -q '/opt/ros/kilted' build/lewm_go2_control/CMakeCache.txt; then
  echo "Existing build/ was configured against ROS Kilted." >&2
  echo "Re-run with: scripts/build_go2_sim.sh --clean" >&2
  exit 1
fi

if [[ "$ROSDEP" == "1" ]]; then
  rosdep update
  rosdep install --from-paths lewm_go2_control third_party/unitree_go2_ros2 --ignore-src -r -y --rosdistro jazzy
fi

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
    lewm_go2_control \
  "$@"

echo
echo "Build complete."
echo "Source the overlay with: source install/setup.bash"
echo "Launch with: scripts/launch_go2_sim.sh"
