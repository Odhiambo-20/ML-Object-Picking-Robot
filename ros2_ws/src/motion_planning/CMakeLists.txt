# CMakeLists.txt for motion_planning ROS2 package
# Production-grade build configuration with full optimization and safety features

cmake_minimum_required(VERSION 3.16)
project(motion_planning)

# Default to C++17 for modern features
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find required dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# ROS2 dependencies
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(control_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

# MoveIt2 dependencies
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_ros_move_group REQUIRED)
find_package(moveit_ros_occupancy_map_monitor REQUIRED)

# Robot dependencies
find_package(kdl_parser REQUIRED)
find_package(orocos_kdl REQUIRED)

# Math and optimization dependencies
find_package(Eigen3 REQUIRED)
find_package(fcl REQUIRED)
find_package(ompl REQUIRED COMPONENTS geometric)

# Threading and performance
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

# Python dependencies (if any Python components)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Set compiler flags for production
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Production optimization flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native")
    
    # Safety and reliability flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Werror")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wconversion -Wsign-conversion")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wdouble-promotion -Wfloat-equal")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wshadow -Wformat=2")
    
    # Performance and debugging flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffunction-sections -fdata-sections")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-common -fstrict-aliasing")
    
    # Security flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORTIFY_SOURCE=2 -fstack-protector-strong")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIE -fPIC")
    
    # Linker flags
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--gc-sections")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,relro,-z,now")
    
    # Enable RTTI and exceptions (required by MoveIt2)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions -frtti")
endif()

# Include directories
include_directories(
    include
    ${EIGEN3_INCLUDE_DIRS}
    ${orocos_kdl_INCLUDE_DIRS}
    ${fcl_INCLUDE_DIRS}
    ${ompl_INCLUDE_DIRS}
)

# Add library
add_library(${PROJECT_NAME}_lib
    src/collision_checker.cpp
    src/inverse_kinematics.cpp
    src/path_optimizer.cpp
    src/trajectory_planner.cpp
    src/motion_control_node.cpp
)

# Link libraries
target_link_libraries(${PROJECT_NAME}_lib
    ${rclcpp_LIBRARIES}
    ${rclcpp_action_LIBRARIES}
    ${rclcpp_components_LIBRARIES}
    moveit::moveit_core
    moveit::moveit_ros_planning
    moveit::moveit_ros_planning_interface
    moveit::moveit_ros_move_group
    ${orocos_kdl_LIBRARIES}
    ${fcl_LIBRARIES}
    ${ompl_LIBRARIES}
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
    Threads::Threads
)

# Set target properties
target_include_directories(${PROJECT_NAME}_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_definitions(${PROJECT_NAME}_lib PUBLIC
    MOTION_PLANNING_PACKAGE
    EIGEN_NO_DEBUG
    EIGEN_MPL2_ONLY
    BOOST_ALL_NO_LIB
    BOOST_SYSTEM_NO_DEPRECATED
)

# Add executable for the motion control node
add_executable(motion_control_node
    src/motion_control_node.cpp
)

target_link_libraries(motion_control_node
    ${PROJECT_NAME}_lib
    ${rclcpp_LIBRARIES}
    ${rclcpp_action_LIBRARIES}
    moveit::moveit_ros_planning_interface
)

# Add tests if building tests
if(BUILD_TESTING)
    find_package(ament_cmake_gtest REQUIRED)
    find_package(ament_lint_auto REQUIRED)
    find_package(ament_cmake_cppcheck REQUIRED)
    find_package(ament_cmake_cpplint REQUIRED)
    find_package(ament_cmake_uncrustify REQUIRED)
    
    # Enable linting
    ament_lint_auto_find_test_dependencies()
    
    # Add unit tests
    ament_add_gtest(test_collision_checker
        test/test_collision_checker.cpp
    )
    target_link_libraries(test_collision_checker
        ${PROJECT_NAME}_lib
        GTest::gtest
        GTest::gtest_main
    )
    
    ament_add_gtest(test_inverse_kinematics
        test/test_inverse_kinematics.cpp
    )
    target_link_libraries(test_inverse_kinematics
        ${PROJECT_NAME}_lib
        GTest::gtest
        GTest::gtest_main
    )
    
    ament_add_gtest(test_trajectory_planner
        test/test_trajectory_planner.cpp
    )
    target_link_libraries(test_trajectory_planner
        ${PROJECT_NAME}_lib
        GTest::gtest
        GTest::gtest_main
    )
    
    # Add performance tests
    add_executable(benchmark_collision_checker
        test/benchmark_collision_checker.cpp
    )
    target_link_libraries(benchmark_collision_checker
        ${PROJECT_NAME}_lib
        ${rclcpp_LIBRARIES}
        benchmark::benchmark
    )
    
    add_executable(benchmark_ik_solver
        test/benchmark_ik_solver.cpp
    )
    target_link_libraries(benchmark_ik_solver
        ${PROJECT_NAME}_lib
        ${rclcpp_LIBRARIES}
        benchmark::benchmark
    )
endif()

# Install targets
install(TARGETS
    ${PROJECT_NAME}_lib
    motion_control_node
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# Install headers
install(DIRECTORY include/
    DESTINATION include/${PROJECT_NAME}
    PATTERN "*.hpp"
    PATTERN "*.h"
)

# Install launch files
install(DIRECTORY launch/
    DESTINATION share/${PROJECT_NAME}/launch
)

# Install config files
install(DIRECTORY config/
    DESTINATION share/${PROJECT_NAME}/config
)

# Install Python modules if any
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/motion_planning)
    ament_python_install_package(${CMAKE_CURRENT_SOURCE_DIR}/motion_planning)
endif()

# Export dependencies
ament_export_dependencies(
    rclcpp
    rclcpp_action
    rclcpp_components
    std_msgs
    geometry_msgs
    sensor_msgs
    trajectory_msgs
    control_msgs
    moveit_msgs
    visualization_msgs
    moveit_core
    moveit_ros_planning
    moveit_ros_planning_interface
    moveit_ros_move_group
    kdl_parser
    orocos_kdl
    fcl
    ompl
    Eigen3
)

# Export include directories
ament_export_include_directories(include)

# Export libraries
ament_export_libraries(${PROJECT_NAME}_lib)

# Finalize
ament_package()

# Add custom targets for development
add_custom_target(format
    COMMAND find ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/include
            -name "*.cpp" -o -name "*.hpp" -o -name "*.h"
            | xargs clang-format -i -style=file
    COMMENT "Formatting source code with clang-format"
    VERBATIM
)

add_custom_target(docs
    COMMAND doxygen ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile
    COMMENT "Generating API documentation"
    VERBATIM
)

add_custom_target(coverage
    COMMAND mkdir -p coverage && cd coverage && gcovr -r ${CMAKE_CURRENT_SOURCE_DIR}
            --exclude-unreachable-branches --exclude-throw-branches
            --html --html-details -o coverage.html
    COMMENT "Generating code coverage report"
    VERBATIM
)

# Performance optimization flags for release builds
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        target_compile_options(${PROJECT_NAME}_lib PRIVATE
            -flto
            -fno-signed-zeros
            -fno-trapping-math
            -fassociative-math
            -frename-registers
            -ftree-vectorize
        )
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -flto")
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(${PROJECT_NAME}_lib PRIVATE
            -flto=thin
            -ffast-math
            -fvectorize
            -fslp-vectorize
        )
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto=thin")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -flto=thin")
    endif()
    
    # Strip symbols for production (optional)
    if(NOT CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -s")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s")
    endif()
endif()

# Add compiler feature detection
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    add_compile_options(-march=native)
endif()

check_cxx_compiler_flag("-mtune=native" COMPILER_SUPPORTS_MTUNE_NATIVE)
if(COMPILER_SUPPORTS_MTUNE_NATIVE)
    add_compile_options(-mtune=native)
endif()

# Add sanitizers for debug builds
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${PROJECT_NAME}_lib PRIVATE
            -fsanitize=address
            -fsanitize=undefined
            -fsanitize=leak
            -fno-omit-frame-pointer
            -fno-optimize-sibling-calls
        )
        target_link_options(${PROJECT_NAME}_lib PRIVATE
            -fsanitize=address
            -fsanitize=undefined
            -fsanitize=leak
        )
    endif()
endif()

# Set RPATH for proper library loading
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Generate compile_commands.json for tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Print configuration summary
message(STATUS "=========================================")
message(STATUS "Motion Planning Package Configuration")
message(STATUS "=========================================")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Eigen3: ${Eigen3_VERSION}")
message(STATUS "OMPL: ${ompl_VERSION}")
message(STATUS "MoveIt2: Found")
message(STATUS "OpenMP: ${OpenMP_CXX_VERSION}")
message(STATUS "Tests: ${BUILD_TESTING}")
message(STATUS "Sanitizers: ${CMAKE_BUILD_TYPE STREQUAL 'Debug'}")
message(STATUS "=========================================")
