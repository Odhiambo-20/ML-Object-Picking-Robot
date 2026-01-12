# CMakeLists.txt for Task Planning Package
# Production-ready configuration for ML-based Object Picking Robot

cmake_minimum_required(VERSION 3.8)
project(task_planning)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Enable C++17 features
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Check for ament_cmake
if(NOT DEFINED ament_cmake_FOUND)
  find_package(ament_cmake REQUIRED)
endif()

# Find ROS2 dependencies
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(action_msgs REQUIRED)

# Find custom message packages
find_package(robot_interfaces REQUIRED)

# Find additional dependencies
find_package(yaml-cpp REQUIRED)
find_package(Threads REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Check for optional dependencies
find_package(OpenCV QUIET COMPONENTS core highgui imgproc)
find_package(PCL QUIET COMPONENTS common io)
find_package(Eigen3 QUIET)
find_package(Boost QUIET COMPONENTS thread system chrono)

# Python dependencies
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 QUIET)

# Set Python install directory
if(WIN32)
  set(PYTHON_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/Lib/site-packages")
else()
  set(PYTHON_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages")
endif()

# Set compiler flags for performance and safety
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # Performance optimizations
  add_compile_options(
    -O3
    -march=native
    -mtune=native
    -ffast-math
    -funroll-loops
    -ftree-vectorize
    -flto
    -fno-fat-lto-objects
  )
  
  # Safety and debugging
  add_compile_options(
    -Wall
    -Wextra
    -Werror
    -Wpedantic
    -Wconversion
    -Wsign-conversion
    -Wnull-dereference
    -Wdouble-promotion
    -Wformat=2
    -Wduplicated-cond
    -Wduplicated-branches
    -Wlogical-op
    -Wuseless-cast
    -Wshadow
    -Wcast-align
    -Wstrict-overflow=5
    -fstack-protector-strong
    -fPIC
    -fno-common
  )
  
  # Link-time optimization
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -flto")
  
  # Security hardening
  add_compile_options(
    -D_FORTIFY_SOURCE=2
    -fstack-clash-protection
    -fcf-protection=full
  )
  
  # Add sanitizers in debug mode
  if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    add_compile_options(
      -fsanitize=address
      -fsanitize=undefined
      -fsanitize=leak
      -fsanitize=float-divide-by-zero
      -fsanitize=float-cast-overflow
      -fno-sanitize-recover=all
    )
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address -fsanitize=undefined -fsanitize=leak")
  endif()
  
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  # MSVC optimizations
  add_compile_options(
    /O2
    /Ob2
    /GF
    /Gy
    /arch:AVX2
    /fp:fast
  )
  
  # Safety and warnings
  add_compile_options(
    /W4
    /WX
    /sdl
    /permissive-
    /guard:cf
  )
  
  # Security
  add_compile_options(
    /GS
    /DYNAMICBASE
    /NXCOMPAT
  )
endif()

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Set build properties based on build type
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_compile_definitions(DEBUG_BUILD)
  add_compile_options(-g3 -Og -D_DEBUG)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_definitions(NDEBUG RELEASE_BUILD)
  add_compile_options(-g0 -DNDEBUG)
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  add_compile_definitions(NDEBUG RELWITHDEBINFO_BUILD)
  add_compile_options(-g2 -DNDEBUG)
elseif(CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
  add_compile_definitions(NDEBUG MINSIZEREL_BUILD)
  add_compile_options(-g0 -Os -DNDEBUG)
endif()

# Add project-specific definitions
add_compile_definitions(
  TASK_PLANNING_PACKAGE
  ROS_DISTRO_HUMBLE
  _USE_MATH_DEFINES
  _CRT_SECURE_NO_WARNINGS
  NOMINMAX  # Windows specific
)

# Include directories
include_directories(
  include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${rclcpp_INCLUDE_DIRS}
  ${std_msgs_INCLUDE_DIRS}
  ${geometry_msgs_INCLUDE_DIRS}
  ${sensor_msgs_INCLUDE_DIRS}
  ${nav_msgs_INCLUDE_DIRS}
  ${visualization_msgs_INCLUDE_DIRS}
  ${tf2_INCLUDE_DIRS}
  ${tf2_ros_INCLUDE_DIRS}
  ${tf2_geometry_msgs_INCLUDE_DIRS}
  ${action_msgs_INCLUDE_DIRS}
  ${robot_interfaces_INCLUDE_DIRS}
  ${yaml-cpp_INCLUDE_DIRS}
  ${Python3_INCLUDE_DIRS}
)

# Add optional dependencies
if(OpenCV_FOUND)
  add_compile_definitions(WITH_OPENCV)
  include_directories(${OpenCV_INCLUDE_DIRS})
  list(APPEND EXTRA_LIBS ${OpenCV_LIBRARIES})
endif()

if(PCL_FOUND)
  add_compile_definitions(WITH_PCL)
  include_directories(${PCL_INCLUDE_DIRS})
  list(APPEND EXTRA_LIBS ${PCL_LIBRARIES})
endif()

if(Eigen3_FOUND)
  add_compile_definitions(WITH_EIGEN)
  include_directories(${EIGEN3_INCLUDE_DIRS})
endif()

if(Boost_FOUND)
  add_compile_definitions(WITH_BOOST)
  include_directories(${Boost_INCLUDE_DIRS})
  list(APPEND EXTRA_LIBS ${Boost_LIBRARIES})
endif()

if(pybind11_FOUND)
  add_compile_definitions(WITH_PYBIND11)
  include_directories(${pybind11_INCLUDE_DIRS})
endif()

# Add system includes
include_directories(SYSTEM
  /usr/include
  /usr/local/include
)

# Set library output directory
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Create library for shared C++ components
add_library(task_planning_common SHARED
  src/common/task_utils.cpp
  src/common/safety_utils.cpp
  src/common/performance_monitor.cpp
)

target_link_libraries(task_planning_common
  ${rclcpp_LIBRARIES}
  ${std_msgs_LIBRARIES}
  ${geometry_msgs_LIBRARIES}
  ${sensor_msgs_LIBRARIES}
  ${tf2_LIBRARIES}
  ${tf2_ros_LIBRARIES}
  ${yaml-cpp_LIBRARIES}
  ${Python3_LIBRARIES}
  Threads::Threads
  ${EXTRA_LIBS}
)

# Add Python module
pybind11_add_module(task_planning_pybind
  src/python_bindings/task_bindings.cpp
  src/python_bindings/safety_bindings.cpp
)

target_link_libraries(task_planning_pybind
  task_planning_common
  ${rclcpp_LIBRARIES}
  ${Python3_LIBRARIES}
  pybind11::module
)

# Install Python package
install(
  DIRECTORY task_planning/
  DESTINATION ${PYTHON_INSTALL_DIR}/task_planning
  PATTERN "*.pyc" EXCLUDE
  PATTERN "__pycache__" EXCLUDE
)

install(
  FILES
    package.xml
    setup.py
    setup.cfg
  DESTINATION share/${PROJECT_NAME}
)

# Install Python scripts
install(
  PROGRAMS
    scripts/task_planner_node.py
    scripts/safety_monitor_node.py
    scripts/task_executor_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(
  DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Install config files
install(
  DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

# Install Python modules
install(
  TARGETS task_planning_pybind
  LIBRARY DESTINATION ${PYTHON_INSTALL_DIR}/task_planning
)

# Install C++ library
install(
  TARGETS task_planning_common
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Install headers
install(
  DIRECTORY include/
  DESTINATION include
)

# Export targets
ament_export_include_directories(include)
ament_export_libraries(task_planning_common)
ament_export_dependencies(
  rclcpp
  rclpy
  std_msgs
  geometry_msgs
  sensor_msgs
  nav_msgs
  visualization_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  action_msgs
  robot_interfaces
  yaml-cpp
)

# Add Python dependencies to package.xml generation
ament_python_install_package(${PROJECT_NAME}
  PACKAGE_DIR task_planning
)

# Add tests if enabled
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
  
  # Enable linting
  ament_lint_auto_find_test_dependencies()
  
  # Add C++ tests
  if(BUILD_CPP_TESTS)
    add_subdirectory(test/cpp)
  endif()
  
  # Add Python tests
  if(BUILD_PYTHON_TESTS)
    add_subdirectory(test/python)
  endif()
  
  # Add integration tests
  if(BUILD_INTEGRATION_TESTS)
    add_subdirectory(test/integration)
  endif()
endif()

# Performance profiling support
option(ENABLE_PROFILING "Enable performance profiling" OFF)
if(ENABLE_PROFILING)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-pg)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
  endif()
  add_compile_definitions(ENABLE_PROFILING)
endif()

# Code coverage support
option(ENABLE_COVERAGE "Enable code coverage reporting" OFF)
if(ENABLE_COVERAGE)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(--coverage -fprofile-arcs -ftest-coverage)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
  endif()
endif()

# Static analysis support
option(ENABLE_STATIC_ANALYSIS "Enable static analysis" ON)
if(ENABLE_STATIC_ANALYSIS)
  find_program(CLANG_TIDY_EXE NAMES "clang-tidy")
  if(CLANG_TIDY_EXE)
    set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE};-checks=*;-warnings-as-errors=*")
    message(STATUS "Enabled clang-tidy static analysis")
  endif()
  
  find_program(CPPCHECK_EXE NAMES "cppcheck")
  if(CPPCHECK_EXE)
    set(CMAKE_CXX_CPPCHECK "${CPPCHECK_EXE};--enable=all;--inconclusive;--suppress=missingIncludeSystem")
    message(STATUS "Enabled cppcheck static analysis")
  endif()
endif()

# Generate compile_commands.json for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set RPATH for proper library loading
if(NOT APPLE)
  set(CMAKE_INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib")
else()
  set(CMAKE_INSTALL_RPATH "@loader_path/@loader_path/../lib")
endif()

# Add custom targets
add_custom_target(format
  COMMAND find ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/include
          -name "*.cpp" -o -name "*.hpp" -o -name "*.h" | xargs clang-format -i
  COMMENT "Formatting C++ source code"
)

add_custom_target(python_format
  COMMAND find ${CMAKE_CURRENT_SOURCE_DIR}/task_planning
          -name "*.py" | xargs black -q
  COMMENT "Formatting Python source code"
)

add_custom_target(lint
  COMMAND find ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/include
          -name "*.cpp" -o -name "*.hpp" -o -name "*.h" | xargs clang-tidy -p ${CMAKE_BINARY_DIR}
  COMMENT "Running clang-tidy on C++ source code"
)

add_custom_target(docs
  COMMAND doxygen ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile
  COMMENT "Generating documentation"
)

# Create version header
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/task_planning/version.hpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/task_planning/version.hpp
)

# Export include directory
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

# Finalize package
ament_package()

# Print configuration summary
message(STATUS "===========================================")
message(STATUS "Task Planning Package Configuration Summary")
message(STATUS "===========================================")
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Python Version: ${Python3_VERSION}")
message(STATUS "Install Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Python Install Dir: ${PYTHON_INSTALL_DIR}")
message(STATUS "OpenCV Support: ${OpenCV_FOUND}")
message(STATUS "PCL Support: ${PCL_FOUND}")
message(STATUS "Eigen Support: ${Eigen3_FOUND}")
message(STATUS "Boost Support: ${Boost_FOUND}")
message(STATUS "Pybind11 Support: ${pybind11_FOUND}")
message(STATUS "Profiling Enabled: ${ENABLE_PROFILING}")
message(STATUS "Coverage Enabled: ${ENABLE_COVERAGE}")
message(STATUS "Static Analysis: ${ENABLE_STATIC_ANALYSIS}")
message(STATUS "===========================================")

# Add post-build checks
add_custom_command(TARGET task_planning_common POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E echo "Build completed: ${PROJECT_NAME}"
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/reports
)

# Create package documentation
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/README.md
  ${CMAKE_BINARY_DIR}/README.md
  COPYONLY
)

# Generate install manifest
install(
  FILES ${CMAKE_BINARY_DIR}/README.md
  DESTINATION share/${PROJECT_NAME}
)
