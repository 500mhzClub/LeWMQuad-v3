#!/usr/bin/env bash

# Shared helpers for keeping this workspace on ROS 2 Jazzy / Gazebo Harmonic.
# Source this file from the other scripts; do not execute it directly.

lewm_repo_root() {
  git rev-parse --show-toplevel
}

lewm_strip_path_entries() {
  local var_name="$1"
  local current="${!var_name-}"
  local cleaned=""
  local entry

  if [[ -z "$current" ]]; then
    return 0
  fi

  local old_ifs="$IFS"
  IFS=:
  for entry in $current; do
    case "$entry" in
      /opt/ros/*|*/install|*/install/*)
        continue
        ;;
    esac
    if [[ -z "$cleaned" ]]; then
      cleaned="$entry"
    else
      cleaned="$cleaned:$entry"
    fi
  done
  IFS="$old_ifs"

  export "$var_name=$cleaned"
}

lewm_clean_ros_environment() {
  lewm_strip_path_entries PATH
  lewm_strip_path_entries LD_LIBRARY_PATH
  lewm_strip_path_entries PYTHONPATH
  lewm_strip_path_entries CMAKE_PREFIX_PATH
  lewm_strip_path_entries AMENT_PREFIX_PATH
  lewm_strip_path_entries COLCON_PREFIX_PATH
  lewm_strip_path_entries PKG_CONFIG_PATH

  unset ROS_DISTRO
  unset ROS_ETC_DIR
  unset ROS_PACKAGE_PATH
  unset ROS_PYTHON_VERSION
  unset ROS_VERSION
}

lewm_source_jazzy_underlay() {
  local setup_file="/opt/ros/jazzy/setup.bash"
  local had_nounset=0

  lewm_clean_ros_environment

  if [[ ! -f "$setup_file" ]]; then
    echo "Missing $setup_file" >&2
    echo "Install the simulator dependencies first: scripts/install_jazzy_harmonic_deps.sh" >&2
    return 1
  fi

  case "$-" in
    *u*) had_nounset=1; set +u ;;
  esac
  # shellcheck disable=SC1090
  source "$setup_file"
  if [[ "$had_nounset" == "1" ]]; then
    set -u
  fi

  if [[ "${ROS_DISTRO:-}" != "jazzy" ]]; then
    echo "Expected ROS_DISTRO=jazzy after sourcing $setup_file, got '${ROS_DISTRO:-unset}'" >&2
    return 1
  fi
}

lewm_source_workspace_overlay() {
  local repo_root="$1"
  local setup_file="$repo_root/install/setup.bash"
  local had_nounset=0

  if [[ ! -f "$setup_file" ]]; then
    echo "Missing $setup_file" >&2
    echo "Build first: scripts/build_go2_sim.sh" >&2
    return 1
  fi

  case "$-" in
    *u*) had_nounset=1; set +u ;;
  esac
  # shellcheck disable=SC1090
  source "$setup_file"
  if [[ "$had_nounset" == "1" ]]; then
    set -u
  fi
}

lewm_need_command() {
  local command_name="$1"
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Missing command: $command_name" >&2
    return 1
  fi
}

lewm_need_ros_package() {
  local package_name="$1"
  if ! ros2 pkg prefix "$package_name" >/dev/null 2>&1; then
    echo "Missing ROS package: $package_name" >&2
    return 1
  fi
}

lewm_check_noble_os() {
  local codename=""
  local pretty_name=""

  if [[ -f /etc/os-release ]]; then
    # shellcheck disable=SC1091
    source /etc/os-release
    codename="${UBUNTU_CODENAME:-${VERSION_CODENAME:-}}"
    pretty_name="${PRETTY_NAME:-unknown}"
  fi

  if [[ "$codename" != "noble" ]]; then
    echo "Expected an Ubuntu 24.04/Noble-compatible OS, got '$pretty_name' codename '$codename'." >&2
    return 1
  fi

  echo "OS: $pretty_name ($codename)"
}
