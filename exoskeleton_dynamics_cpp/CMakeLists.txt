cmake_minimum_required(VERSION 3.8)
project(exoskeleton_dynamics_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

set(CMAKE_CXX_STANDARD 17)

# Pinocchio installed at /opt/openrobots
list(PREPEND CMAKE_PREFIX_PATH "/opt/openrobots")

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(exoskeleton_safety_msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)
set(COAL_DISABLE_HPP_FCL_WARNINGS TRUE)
find_package(pinocchio REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

add_executable(dynamics_node_cpp src/dynamics_node_cpp.cpp)
target_compile_definitions(dynamics_node_cpp PRIVATE COAL_DISABLE_HPP_FCL_WARNINGS)

target_include_directories(dynamics_node_cpp PRIVATE
  ${EIGEN3_INCLUDE_DIR}
  /opt/openrobots/include
)

target_link_libraries(dynamics_node_cpp
  pinocchio::pinocchio
)

ament_target_dependencies(dynamics_node_cpp
  rclcpp
  sensor_msgs
  geometry_msgs
  visualization_msgs
  exoskeleton_safety_msgs
  ament_index_cpp
)

install(TARGETS dynamics_node_cpp
  DESTINATION lib/${PROJECT_NAME}
)

# ── Standalone C++ benchmark (no ROS) ──
add_executable(benchmark_standalone src/benchmark_standalone.cpp)
target_compile_definitions(benchmark_standalone PRIVATE COAL_DISABLE_HPP_FCL_WARNINGS)
target_include_directories(benchmark_standalone PRIVATE
  ${EIGEN3_INCLUDE_DIR}
  /opt/openrobots/include
)
target_link_libraries(benchmark_standalone
  pinocchio::pinocchio
  stdc++fs
)

install(TARGETS benchmark_standalone
  DESTINATION lib/${PROJECT_NAME}
)

# ── ROS C++ benchmark node ──
add_executable(benchmark_dynamics_cpp src/benchmark_dynamics_cpp.cpp)
target_compile_definitions(benchmark_dynamics_cpp PRIVATE COAL_DISABLE_HPP_FCL_WARNINGS)
target_include_directories(benchmark_dynamics_cpp PRIVATE
  ${EIGEN3_INCLUDE_DIR}
  /opt/openrobots/include
)
target_link_libraries(benchmark_dynamics_cpp
  pinocchio::pinocchio
  stdc++fs
)
ament_target_dependencies(benchmark_dynamics_cpp
  rclcpp
  sensor_msgs
  geometry_msgs
  exoskeleton_safety_msgs
  ament_index_cpp
)

install(TARGETS benchmark_dynamics_cpp
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
