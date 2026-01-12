cmake_minimum_required(VERSION 3.8)
project(system_monitor)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Check if ament_cmake is available (ROS2)
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND NOT COMMAND find_package)
  message(FATAL_ERROR "ament_cmake not found. Please source ROS2 environment first.")
endif()

# Find ROS2 packages
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_srvs REQUIRED)

# Find custom message packages
find_package(system_monitor_msgs REQUIRED)
find_package(robot_interfaces REQUIRED)

# Find Python dependencies
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(python_cmake_module REQUIRED)

# Find additional system libraries
find_package(Threads REQUIRED)

# Performance monitor dependencies
find_package(Eigen3 QUIET)
if(EIGEN3_FOUND)
  message(STATUS "Eigen3 found: ${EIGEN3_INCLUDE_DIRS}")
else()
  message(WARNING "Eigen3 not found. Performance monitor will have limited functionality.")
endif()

find_package(OpenMP QUIET)
if(OPENMP_FOUND)
  message(STATUS "OpenMP found: ${OpenMP_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Python dependencies will be installed via pip/requirements.txt
# but we can check for them here
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import numpy; print('numpy found')"
  RESULT_VARIABLE NUMPY_FOUND
  OUTPUT_QUIET
  ERROR_QUIET
)

if(NUMPY_FOUND)
  message(STATUS "NumPy found for Python")
else()
  message(WARNING "NumPy not found for Python. Some features may be limited.")
endif()

execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import psutil; print('psutil found')"
  RESULT_VARIABLE PSUTIL_FOUND
  OUTPUT_QUIET
  ERROR_QUIET
)

if(PSUTIL_FOUND)
  message(STATUS "psutil found for Python")
else()
  message(WARNING "psutil not found for Python. System monitoring will be limited.")
endif()

# Set compiler options
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # Common warnings for GCC and Clang
  add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
    -Werror
    -Wno-error=deprecated-declarations
    -Wno-error=unused-parameter
    -Wno-error=maybe-uninitialized
    $<$<CONFIG:DEBUG>:-O0 -g3 -DDEBUG>
    $<$<CONFIG:RELEASE>:-O3 -DNDEBUG>
    $<$<CONFIG:RELWITHDEBINFO>:-O2 -g -DNDEBUG>
  )
  
  # Security hardening
  add_compile_options(
    -D_FORTIFY_SOURCE=2
    -fstack-protector-strong
    -Wformat
    -Wformat-security
  )
  
  # Performance optimizations
  add_compile_options(
    -march=native
    -mtune=native
    -ffast-math
    -funroll-loops
  )
  
  # Linker options
  add_link_options(
    -Wl,--as-needed
    -Wl,--no-undefined
    -Wl,-z,relro,-z,now
  )
endif()

# Define library directories
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

# Create Python package
ament_python_install_package(${PROJECT_NAME}
  PACKAGE_DIR "src/${PROJECT_NAME}"
)

# Install Python dependencies via setup.py
install(
  DIRECTORY src/${PROJECT_NAME}/
  DESTINATION lib/${PROJECT_NAME}
  PATTERN "*.pyc" EXCLUDE
  PATTERN "__pycache__" EXCLUDE
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

# Install test files
if(BUILD_TESTING)
  install(
    DIRECTORY test/
    DESTINATION share/${PROJECT_NAME}/test
  )
endif()

# Create C++ library for performance-critical components (if any)
# add_library(${PROJECT_NAME}_cpp
#   src/cpp/performance_analyzer.cpp
#   src/cpp/metrics_aggregator.cpp
# )
# 
# target_include_directories(${PROJECT_NAME}_cpp PUBLIC
#   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#   $<INSTALL_INTERFACE:include>
# )
# 
# ament_target_dependencies(${PROJECT_NAME}_cpp
#   rclcpp
#   std_msgs
#   diagnostic_msgs
#   system_monitor_msgs
# )
# 
# install(TARGETS ${PROJECT_NAME}_cpp
#   EXPORT ${PROJECT_NAME}_targets
#   ARCHIVE DESTINATION lib
#   LIBRARY DESTINATION lib
#   RUNTIME DESTINATION bin
# )
# 
# install(DIRECTORY include/
#   DESTINATION include/${PROJECT_NAME}
# )

# Create Python entry points
# These are defined in setup.cfg, but we ensure they're available
set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})

# Generate launch files from templates (if any)
# configure_file(
#   launch/monitoring.launch.py.in
#   launch/monitoring.launch.py
#   @ONLY
# )

# Add performance monitor dependencies
find_package(OpenCV QUIET COMPONENTS core imgproc highgui)
if(OpenCV_FOUND)
  message(STATUS "OpenCV found: ${OpenCV_VERSION}")
  # target_link_libraries(${PROJECT_NAME}_cpp ${OpenCV_LIBS})
else()
  message(WARNING "OpenCV not found. Some visualization features may be limited.")
endif()

# Add ML dependencies for anomaly detection
find_package(Python3 COMPONENTS NumPy)
if(Python3_NumPy_FOUND)
  message(STATUS "NumPy found via CMake: ${Python3_NumPy_VERSION}")
else()
  # Try to find NumPy manually
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import numpy; print(numpy.__version__)"
    OUTPUT_VARIABLE NUMPY_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(NUMPY_VERSION)
    message(STATUS "NumPy found: ${NUMPY_VERSION}")
  else()
    message(WARNING "NumPy not found. Anomaly detection will use simplified algorithms.")
  endif()
endif()

# Check for pandas (optional for advanced analytics)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import pandas; print(pandas.__version__)"
  OUTPUT_VARIABLE PANDAS_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)
if(PANDAS_VERSION)
  message(STATUS "Pandas found: ${PANDAS_VERSION}")
else()
  message(STATUS "Pandas not found. Advanced analytics will use alternative methods.")
endif()

# Check for scikit-learn (optional for ML-based anomaly detection)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import sklearn; print(sklearn.__version__)"
  OUTPUT_VARIABLE SKLEARN_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)
if(SKLEARN_VERSION)
  message(STATUS "scikit-learn found: ${SKLEARN_VERSION}")
else()
  message(STATUS "scikit-learn not found. ML-based anomaly detection disabled.")
endif()

# Check for msgpack (optional for high-performance logging)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import msgpack; print(msgpack.__version__)"
  OUTPUT_VARIABLE MSGPACK_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)
if(MSGPACK_VERSION)
  message(STATUS "msgpack found: ${MSGPACK_VERSION}")
else()
  message(STATUS "msgpack not found. JSON will be used for logging.")
endif()

# Check for GPUtil (optional for GPU monitoring)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import GPUtil; print('GPUtil found')"
  RESULT_VARIABLE GPUTIL_FOUND
  OUTPUT_QUIET
  ERROR_QUIET
)
if(GPUTIL_FOUND EQUAL 0)
  message(STATUS "GPUtil found for GPU monitoring")
else()
  message(STATUS "GPUtil not found. GPU monitoring disabled.")
endif()

# Check for pyarrow (optional for Parquet export)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import pyarrow; print(pyarrow.__version__)"
  OUTPUT_VARIABLE PYARROW_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)
if(PYARROW_VERSION)
  message(STATUS "PyArrow found: ${PYARROW_VERSION}")
else()
  message(STATUS "PyArrow not found. Parquet export disabled.")
endif()

# Define system monitor Python nodes
add_custom_target(system_monitor_nodes ALL
  DEPENDS ${PROJECT_NAME}
  COMMENT "Building system monitor Python nodes"
)

# Create symlinks or wrapper scripts for easy execution
install(PROGRAMS
  scripts/start_monitoring.sh
  scripts/stop_monitoring.sh
  scripts/health_check.sh
  scripts/performance_report.sh
  DESTINATION lib/${PROJECT_NAME}/scripts
)

# Make scripts executable
file(COPY scripts/start_monitoring.sh
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
     FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
file(COPY scripts/stop_monitoring.sh
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
     FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
file(COPY scripts/health_check.sh
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
     FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
file(COPY scripts/performance_report.sh
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
     FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

# Install Python requirements
install(
  FILES requirements.txt
  DESTINATION share/${PROJECT_NAME}
)

# Install package.xml
install(
  FILES package.xml
  DESTINATION share/${PROJECT_NAME}
)

# Install README and documentation
install(
  FILES
    README.md
    CHANGELOG.md
    CONTRIBUTING.md
  DESTINATION share/${PROJECT_NAME}
)

# Install configuration examples
install(
  DIRECTORY config/examples/
  DESTINATION share/${PROJECT_NAME}/config/examples
)

# Install systemd service files (for production deployment)
install(
  FILES
    systemd/robot-system-monitor.service
    systemd/robot-performance-monitor.service
  DESTINATION share/${PROJECT_NAME}/systemd
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
)

# Install log rotation configuration
install(
  FILES
    logrotate/robot-monitor
  DESTINATION share/${PROJECT_NAME}/logrotate
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
)

# Install monitoring dashboard templates (if any)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/dashboard/")
  install(
    DIRECTORY dashboard/
    DESTINATION share/${PROJECT_NAME}/dashboard
  )
endif()

# Add uninstall target
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/uninstall.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/uninstall.cmake"
  @ONLY
)

add_custom_target(uninstall
  COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/uninstall.cmake
  COMMENT "Uninstalling system_monitor"
)

# Testing configuration
if(BUILD_TESTING)
  # Find testing packages
  find_package(ament_lint_auto REQUIRED)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
  
  # Enable linting
  ament_lint_auto_find_test_dependencies()
  
  # Add Python tests
  ament_add_pytest_test(test_system_monitor
    PYTHON_EXECUTABLE ${Python3_EXECUTABLE}
    TEST_DIR test/python
    TIMEOUT 60
  )
  
  # Add C++ tests if we have C++ components
  # if(TARGET ${PROJECT_NAME}_cpp)
  #   ament_add_gtest(test_performance_analyzer
  #     test/cpp/test_performance_analyzer.cpp
  #     TIMEOUT 30
  #   )
  #   target_link_libraries(test_performance_analyzer ${PROJECT_NAME}_cpp)
  #   
  #   ament_add_gtest(test_metrics_aggregator
  #     test/cpp/test_metrics_aggregator.cpp
  #     TIMEOUT 30
  #   )
  #   target_link_libraries(test_metrics_aggregator ${PROJECT_NAME}_cpp)
  # endif()
  
  # Add integration tests
  ament_add_pytest_test(test_integration
    PYTHON_EXECUTABLE ${Python3_EXECUTABLE}
    TEST_DIR test/integration
    TIMEOUT 120
  )
  
  # Add performance tests
  ament_add_pytest_test(test_performance
    PYTHON_EXECUTABLE ${Python3_EXECUTABLE}
    TEST_DIR test/performance
    TIMEOUT 180
  )
endif()

# Performance optimization for production builds
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Link time optimization for Release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  if(CMAKE_COMPILER_IS_GNUCXX)
    check_cxx_compiler_flag("-flto" HAS_FLTO)
    if(HAS_FLTO)
      add_compile_options(-flto)
      add_link_options(-flto)
      message(STATUS "Link Time Optimization enabled for Release build")
    endif()
  endif()
  
  # Strip debug symbols for production
  if(UNIX AND NOT APPLE)
    add_link_options(-s)
  endif()
endif()

# Add sanitizers for Debug builds
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Address sanitizer
    check_cxx_compiler_flag("-fsanitize=address" HAS_ASAN)
    if(HAS_ASAN)
      add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
      add_link_options(-fsanitize=address)
      message(STATUS "AddressSanitizer enabled for Debug build")
    endif()
    
    # Undefined behavior sanitizer
    check_cxx_compiler_flag("-fsanitize=undefined" HAS_UBSAN)
    if(HAS_UBSAN)
      add_compile_options(-fsanitize=undefined)
      add_link_options(-fsanitize=undefined)
      message(STATUS "UndefinedBehaviorSanitizer enabled for Debug build")
    endif()
  endif()
endif()

# Platform-specific optimizations
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
  # ARM-specific optimizations (for Raspberry Pi)
  if(CMAKE_COMPILER_IS_GNUCXX)
    add_compile_options(-march=armv8-a -mtune=cortex-a72 -mfpu=neon-fp-armv8 -mfloat-abi=hard)
    message(STATUS "ARM optimizations enabled for Raspberry Pi")
  endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "amd64")
  # x86-64 optimizations
  if(CMAKE_COMPILER_IS_GNUCXX)
    add_compile_options(-march=x86-64 -mtune=generic)
  endif()
endif()

# Dependency checking at build time
add_custom_command(
  TARGET system_monitor_nodes
  POST_BUILD
  COMMAND ${Python3_EXECUTABLE} -m pip check
  COMMENT "Checking Python dependencies"
  VERBATIM
)

# Generate version file
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/version.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/include/${PROJECT_NAME}/version.h"
)

# Install headers if we have C++ components
# install(
#   DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/${PROJECT_NAME}/
#   DESTINATION include/${PROJECT_NAME}
# )

# Export targets for other packages
# ament_export_include_directories(include)
# ament_export_libraries(${PROJECT_NAME}_cpp)
ament_export_dependencies(
  rclcpp
  rclpy
  std_msgs
  sensor_msgs
  geometry_msgs
  diagnostic_msgs
  visualization_msgs
  nav_msgs
  std_srvs
  system_monitor_msgs
  robot_interfaces
)

# Generate .env file for Docker/container deployment
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/.env.example"
  "${CMAKE_CURRENT_BINARY_DIR}/.env"
  @ONLY
)

install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/.env"
  DESTINATION share/${PROJECT_NAME}
  RENAME ".env.example"
)

# Create package export
ament_package(
  CONFIG_EXTRAS
    "${CMAKE_CURRENT_BINARY_DIR}/system_monitor-extras.cmake"
)

# Generate extras.cmake for downstream packages
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/system_monitor-extras.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/system_monitor-extras.cmake"
  @ONLY
)

# Post-build steps for production deployment
add_custom_target(deploy_prepare
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_INSTALL_PREFIX}/var/log/robot_monitor
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_INSTALL_PREFIX}/var/performance_data
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_INSTALL_PREFIX}/etc/robot_monitor
  COMMENT "Creating deployment directories"
)

# Add install target for deployment
add_custom_target(install_system_monitor
  COMMAND ${CMAKE_COMMAND} --build . --target install
  DEPENDS deploy_prepare
  COMMENT "Installing system monitor to system"
)

# Create a target that runs all checks before install
add_custom_target(pre_install_check
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/check_install.py
  COMMENT "Running pre-installation checks"
)

# Make install depend on pre-install checks
add_dependencies(install pre_install_check)

# Add summary at the end
message(STATUS "")
message(STATUS "==========================================")
message(STATUS "System Monitor Package Configuration")
message(STATUS "==========================================")
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Python Executable: ${Python3_EXECUTABLE}")
message(STATUS "Install Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
message(STATUS "Components:")
message(STATUS "  • Health Checker Node: ✓")
message(STATUS "  • Performance Monitor Node: ✓")
message(STATUS "  • Logger Node: ✓")
message(STATUS "")
message(STATUS "Features:")
message(STATUS "  • Real-time system monitoring: ✓")
message(STATUS "  • Performance metrics collection: ✓")
message(STATUS "  • Anomaly detection: ✓")
message(STATUS "  • Log management: ✓")
message(STATUS "  • Alerting system: ✓")
message(STATUS "  • Visualization support: ✓")
message(STATUS "")
if(BUILD_TESTING)
  message(STATUS "Testing: Enabled")
  message(STATUS "  • Unit tests: ✓")
  message(STATUS "  • Integration tests: ✓")
  message(STATUS "  • Performance tests: ✓")
else()
  message(STATUS "Testing: Disabled")
endif()
message(STATUS "==========================================")
