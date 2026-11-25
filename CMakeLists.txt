cmake_minimum_required(VERSION 3.8)
project(adaptive_tactile_gripper_description)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY config meshes mjcf srdf urdf
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
