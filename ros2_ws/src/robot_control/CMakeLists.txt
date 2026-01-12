# ============================================================================
# CMakeLists.txt - Production-Grade Build Configuration
# Robot Control Package for ML-Based Object Picking Robot
# ============================================================================
# 
# Author: Victor's Production Team
# Date: 2025-01-12
# Version: 1.0.0
# 
# This CMakeLists.txt builds:
# - C++ libraries (motor drivers, servo controllers, gripper control)
# - ROS2 nodes (hardware interface, control nodes)
# - Integration with external libraries (pigpio, libgpiod, i2c)
# 
# ============================================================================

cmake_minimum_required(VERSION 3.8)
project(robot_control)

# ============================================================================
# COMPILER SETTINGS
# ============================================================================

# Set C++ standard
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Optimization flags for production
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_options(-O3 -march=native)
endif()

# ============================================================================
# DEPENDENCIES
# ============================================================================

# Find ROS2 packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(robot_interfaces REQUIRED)

# Find system libraries
find_library(PIGPIO_LIBRARY pigpio REQUIRED)
find_library(GPIOD_LIBRARY gpiod REQUIRED)
find_library(I2C_LIBRARY i2c)

# Check if libraries found
if(NOT PIGPIO_LIBRARY)
  message(FATAL_ERROR "pigpio library not found! Install with: sudo apt-get install pigpio")
endif()

if(NOT GPIOD_LIBRARY)
  message(FATAL_ERROR "libgpiod not found! Install with: sudo apt-get install libgpiod-dev")
endif()

message(STATUS "Found pigpio: ${PIGPIO_LIBRARY}")
message(STATUS "Found libgpiod: ${GPIOD_LIBRARY}")

# ============================================================================
# INCLUDE DIRECTORIES
# ============================================================================

include_directories(
  include
  ${rclcpp_INCLUDE_DIRS}
  ${std_msgs_INCLUDE_DIRS}
  ${sensor_msgs_INCLUDE_DIRS}
  ${geometry_msgs_INCLUDE_DIRS}
  ${diagnostic_msgs_INCLUDE_DIRS}
  ${robot_interfaces_INCLUDE_DIRS}
)

# ============================================================================
# LIBRARIES
# ============================================================================

# Motor Driver Library
add_library(motor_driver
  src/motor_driver.cpp
)

target_include_directories(motor_driver PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(motor_driver
  ${PIGPIO_LIBRARY}
  pthread
)

ament_target_dependencies(motor_driver
  rclcpp
)

# Servo Controller Library
add_library(servo_controller
  src/servo_controller.cpp
)

target_include_directories(servo_controller PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(servo_controller
  ${PIGPIO_LIBRARY}
  ${I2C_LIBRARY}
  pthread
)

ament_target_dependencies(servo_controller
  rclcpp
)

# Gripper Control Library
add_library(gripper_control
  src/gripper_control.cpp
)

target_include_directories(gripper_control PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(gripper_control
  ${PIGPIO_LIBRARY}
)

ament_target_dependencies(gripper_control
  rclcpp
  std_msgs
  sensor_msgs
  robot_interfaces
)

# ============================================================================
# EXECUTABLES - ROS2 NODES
# ============================================================================

# Hardware Interface Node
add_executable(hardware_interface_node
  src/hardware_interface_node.cpp
)

target_link_libraries(hardware_interface_node
  ${GPIOD_LIBRARY}
  ${I2C_LIBRARY}
)

ament_target_dependencies(hardware_interface_node
  rclcpp
  std_msgs
  sensor_msgs
  diagnostic_msgs
)

# Servo Controller Node
add_executable(servo_controller_node
  src/servo_controller_node.cpp
)

target_link_libraries(servo_controller_node
  servo_controller
  ${PIGPIO_LIBRARY}
  ${I2C_LIBRARY}
)

ament_target_dependencies(servo_controller_node
  rclcpp
  std_msgs
  sensor_msgs
  robot_interfaces
)

# Motor Driver Node
add_executable(motor_driver_node
  src/motor_driver_node.cpp
)

target_link_libraries(motor_driver_node
  motor_driver
  ${PIGPIO_LIBRARY}
)

ament_target_dependencies(motor_driver_node
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  robot_interfaces
)

# Gripper Controller Node
add_executable(gripper_controller_node
  src/gripper_control.cpp
)

target_link_libraries(gripper_controller_node
  gripper_control
  ${PIGPIO_LIBRARY}
)

ament_target_dependencies(gripper_controller_node
  rclcpp
  std_msgs
  sensor_msgs
  robot_interfaces
)

# Safety Monitor Node
add_executable(safety_monitor_node
  src/safety_monitor_node.cpp
)

ament_target_dependencies(safety_monitor_node
  rclcpp
  std_msgs
  sensor_msgs
  diagnostic_msgs
  robot_interfaces
)

# System Monitor Node
add_executable(system_monitor_node
  src/system_monitor_node.cpp
)

ament_target_dependencies(system_monitor_node
  rclcpp
  std_msgs
  diagnostic_msgs
)

# ============================================================================
# TESTING
# ============================================================================

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  
  # Linting
  ament_lint_auto_find_test_dependencies()
  
  # Unit tests
  find_package(ament_cmake_gtest REQUIRED)
  
  # Motor driver tests
  ament_add_gtest(test_motor_driver
    test/test_motor_driver.cpp
  )
  
  target_link_libraries(test_motor_driver
    motor_driver
    ${PIGPIO_LIBRARY}
  )
  
  # Servo controller tests
  ament_add_gtest(test_servo_controller
    test/test_servo_controller.cpp
  )
  
  target_link_libraries(test_servo_controller
    servo_controller
    ${PIGPIO_LIBRARY}
  )
  
  # Gripper control tests
  ament_add_gtest(test_gripper_control
    test/test_gripper_control.cpp
  )
  
  target_link_libraries(test_gripper_control
    gripper_control
    ${PIGPIO_LIBRARY}
  )
endif()

# ============================================================================
# INSTALLATION
# ============================================================================

# Install libraries
install(TARGETS
  motor_driver
  servo_controller
  gripper_control
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Install executables
install(TARGETS
  hardware_interface_node
  servo_controller_node
  motor_driver_node
  gripper_controller_node
  safety_monitor_node
  system_monitor_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install header files
install(DIRECTORY include/
  DESTINATION include
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

# Install config files
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}/
)

# ============================================================================
# AMENT PACKAGE
# ============================================================================

# Export dependencies
ament_export_include_directories(include)
ament_export_libraries(
  motor_driver
  servo_controller
  gripper_control
)
ament_export_dependencies(
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  diagnostic_msgs
  robot_interfaces
)

# Package
ament_package()

# ============================================================================
# POST-BUILD INFORMATION
# ============================================================================

message(STATUS "============================================")
message(STATUS "Robot Control Package Configuration")
message(STATUS "============================================")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Install Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
message(STATUS "Libraries:")
message(STATUS "  - motor_driver")
message(STATUS "  - servo_controller")
message(STATUS "  - gripper_control")
message(STATUS "")
message(STATUS "Executables:")
message(STATUS "  - hardware_interface_node")
message(STATUS "  - servo_controller_node")
message(STATUS "  - motor_driver_node")
message(STATUS "  - gripper_controller_node")
message(STATUS "  - safety_monitor_node")
message(STATUS "  - system_monitor_node")
message(STATUS "")
message(STATUS "External Libraries:")
message(STATUS "  - pigpio: ${PIGPIO_LIBRARY}")
message(STATUS "  - libgpiod: ${GPIOD_LIBRARY}")
if(I2C_LIBRARY)
  message(STATUS "  - libi2c: ${I2C_LIBRARY}")
endif()
message(STATUS "============================================")
