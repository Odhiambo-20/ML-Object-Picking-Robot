cmake_minimum_required(VERSION 3.8)
project(task_planning)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(robot_interfaces REQUIRED)

include_directories(include)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include")
  install(DIRECTORY include/
    DESTINATION include
  )
endif()

install(DIRECTORY 
  launch 
  config 
  behavior_trees
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)

ament_export_include_directories(include)
ament_export_dependencies(rclcpp std_msgs robot_interfaces)
ament_package()
