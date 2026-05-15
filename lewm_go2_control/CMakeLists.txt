cmake_minimum_required(VERSION 3.8)
project(lewm_go2_control)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/BaseState.msg"
  "msg/CommandBlock.msg"
  "msg/EpisodeInfo.msg"
  "msg/ExecutedCommandBlock.msg"
  "msg/FootContacts.msg"
  "msg/Go2ModeState.msg"
  "msg/ResetEvent.msg"
  "srv/ResetEpisode.srv"
  "srv/RunFeatureCheck.srv"
  "srv/SetGo2Mode.srv"
  "srv/SetPolicy.srv"
  DEPENDENCIES geometry_msgs std_msgs
)

install(
  PROGRAMS
    nodes/command_block_adapter
    nodes/base_state_publisher
    nodes/camera_info_publisher
    nodes/reset_manager
    nodes/foot_contacts_publisher
    nodes/mode_manager
    nodes/feature_check_runner
  DESTINATION lib/${PROJECT_NAME}
)

install(
  FILES
    ../config/go2_platform_manifest.yaml
    ../config/go2_primitive_registry.yaml
  DESTINATION share/${PROJECT_NAME}/config
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
