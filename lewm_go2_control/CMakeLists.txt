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

ament_export_dependencies(rosidl_default_runtime)
ament_package()

