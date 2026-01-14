cmake_minimum_required(VERSION 3.8)
project(robot_control)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(control_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(robot_interfaces REQUIRED)

# Check for optional hardware libraries
find_path(PIGPIO_INCLUDE_DIR pigpio.h)
find_library(PIGPIO_LIBRARY pigpio)
if(PIGPIO_INCLUDE_DIR AND PIGPIO_LIBRARY)
  set(HAVE_PIGPIO TRUE)
  message(STATUS "Found pigpio")
else()
  set(HAVE_PIGPIO FALSE)
  message(WARNING "pigpio not found - some features disabled")
endif()

find_path(GPIOD_INCLUDE_DIR gpiod.h)
find_library(GPIOD_LIBRARY gpiod)
if(GPIOD_INCLUDE_DIR AND GPIOD_LIBRARY)
  set(HAVE_GPIOD TRUE)
  message(STATUS "Found gpiod")
else()
  set(HAVE_GPIOD FALSE)
  message(WARNING "gpiod not found - some features disabled")
endif()

include_directories(include)

# Build source list conditionally
set(SOURCES)

# gripper_control.cpp - needs GripperState from robot_interfaces
list(APPEND SOURCES src/gripper_control.cpp)

# hardware_interface_node.cpp - needs gpiod
if(HAVE_GPIOD)
  list(APPEND SOURCES src/hardware_interface_node.cpp)
else()
  message(STATUS "Skipping hardware_interface_node.cpp (needs gpiod)")
endif()

# motor_driver.cpp - needs pigpio
if(HAVE_PIGPIO)
  list(APPEND SOURCES src/motor_driver.cpp)
else()
  message(STATUS "Skipping motor_driver.cpp (needs pigpio)")
endif()

# servo_controller.cpp - needs pigpio
if(HAVE_PIGPIO)
  list(APPEND SOURCES src/servo_controller.cpp)
else()
  message(STATUS "Skipping servo_controller.cpp (needs pigpio)")
endif()

if(SOURCES)
  add_library(${PROJECT_NAME} SHARED ${SOURCES})
  
  ament_target_dependencies(${PROJECT_NAME}
    rclcpp
    std_msgs
    geometry_msgs
    sensor_msgs
    control_msgs
    diagnostic_msgs
    robot_interfaces
  )
  
  if(HAVE_PIGPIO)
    target_include_directories(${PROJECT_NAME} PRIVATE ${PIGPIO_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME} ${PIGPIO_LIBRARY})
  endif()
  
  if(HAVE_GPIOD)
    target_include_directories(${PROJECT_NAME} PRIVATE ${GPIOD_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME} ${GPIOD_LIBRARY})
  endif()
  
  install(TARGETS ${PROJECT_NAME}
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
  )
  
  ament_export_libraries(${PROJECT_NAME})
else()
  message(WARNING "No buildable sources for robot_control")
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include")
  install(DIRECTORY include/
    DESTINATION include
  )
endif()

install(DIRECTORY 
  launch 
  config
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)

ament_export_include_directories(include)
ament_export_dependencies(rclcpp std_msgs geometry_msgs sensor_msgs control_msgs diagnostic_msgs robot_interfaces)
ament_package()
