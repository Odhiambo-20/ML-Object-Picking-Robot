# CMakeLists.txt for perception_stack ROS2 package
# Production-grade build configuration for ML-based object recognition

cmake_minimum_required(VERSION 3.16)
project(perception_stack)

# Default to C++17 for modern features
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Python integration
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)
find_package(pybind11 REQUIRED)

# Find required dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# ROS2 dependencies
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(message_filters REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# Machine Learning and Computer Vision dependencies
find_package(OpenCV REQUIRED COMPONENTS
    core
    imgproc
    imgcodecs
    highgui
    video
    videoio
    calib3d
    features2d
    flann
    dnn
)

# Python dependencies
find_package(PythonLibs 3.8 REQUIRED)

# Performance and optimization
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

# Optional: TensorRT for GPU acceleration
option(USE_TENSORRT "Use TensorRT for inference" OFF)
if(USE_TENSORRT)
    find_package(TensorRT REQUIRED)
    add_definitions(-DUSE_TENSORRT)
endif()

# Optional: CUDA for GPU acceleration
option(USE_CUDA "Use CUDA for inference" OFF)
if(USE_CUDA)
    find_package(CUDA REQUIRED)
    add_definitions(-DUSE_CUDA)
endif()

# Optional: Intel OpenVINO for CPU/GPU acceleration
option(USE_OPENVINO "Use OpenVINO for inference" OFF)
if(USE_OPENVINO)
    find_package(OpenVINO REQUIRED)
    add_definitions(-DUSE_OPENVINO)
endif()

# Set compiler flags for production
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Production optimization flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native")
    
    # Safety and reliability flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wconversion -Wsign-conversion")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wdouble-promotion -Wfloat-equal")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wshadow -Wformat=2 -Werror")
    
    # Performance and debugging flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffunction-sections -fdata-sections")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-common -fstrict-aliasing")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-plt -fno-semantic-interposition")
    
    # Security flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORTIFY_SOURCE=2 -fstack-protector-strong")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIE -fPIC -Wl,-z,now -Wl,-z,relro")
    
    # Linker flags
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--gc-sections")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,noexecstack")
    
    # Enable RTTI and exceptions
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions -frtti")
endif()

# Include directories
include_directories(
    include
    ${Python3_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

# Add Python package installation
install(DIRECTORY perception_stack/
    DESTINATION lib/${PROJECT_NAME}
    PATTERN "__pycache__" EXCLUDE
    PATTERN "*.pyc" EXCLUDE
)

# Add Python scripts
install(PROGRAMS
    scripts/camera_calibration.py
    scripts/data_collection.py
    scripts/model_benchmark.py
    scripts/visualize_detections.py
    DESTINATION lib/${PROJECT_NAME}
)

# Add C++ library (if any C++ components)
add_library(${PROJECT_NAME}_cpp
    src/camera_calibration.cpp
    src/image_processor.cpp
)

# Link C++ library
target_link_libraries(${PROJECT_NAME}_cpp
    ${rclcpp_LIBRARIES}
    ${sensor_msgs_LIBRARIES}
    ${cv_bridge_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${Python3_LIBRARIES}
    pybind11::pybind11
    OpenMP::OpenMP_CXX
    Threads::Threads
)

# Set target properties for C++ library
target_include_directories(${PROJECT_NAME}_cpp PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_definitions(${PROJECT_NAME}_cpp PUBLIC
    PERCEPTION_STACK_PACKAGE
    PYTHON_VERSION=${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}
)

# Add Python-C++ binding
pybind11_add_module(pybindings
    src/pybindings.cpp
)

target_link_libraries(pybindings
    ${PROJECT_NAME}_cpp
    pybind11::pybind11
)

# Add executables
add_executable(yolo_detector_node
    src/yolo_detector_node.py
)

add_executable(object_tracker_node
    src/object_tracker.py
)

add_executable(perception_manager
    src/perception_manager.py
)

# Install executables
install(TARGETS
    ${PROJECT_NAME}_cpp
    pybindings
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch/
    DESTINATION share/${PROJECT_NAME}/launch
)

# Install config files
install(DIRECTORY config/
    DESTINATION share/${PROJECT_NAME}/config
)

# Install models directory
install(DIRECTORY models/
    DESTINATION share/${PROJECT_NAME}/models
    PATTERN "*.pt"
    PATTERN "*.onnx"
    PATTERN "*.engine"
)

# Install Python requirements
install(FILES
    requirements.txt
    setup.py
    setup.cfg
    DESTINATION share/${PROJECT_NAME}
)

# Setup Python package
ament_python_install_package(${PROJECT_NAME}
    PACKAGE_DIR src/${PROJECT_NAME}
)

# Export dependencies
ament_export_dependencies(
    rclcpp
    rclcpp_action
    rclcpp_components
    std_msgs
    sensor_msgs
    geometry_msgs
    visualization_msgs
    vision_msgs
    cv_bridge
    image_transport
    message_filters
    tf2
    tf2_ros
    tf2_geometry_msgs
    OpenCV
    pybind11
    Python3
)

# Export include directories
ament_export_include_directories(include)

# Export libraries
ament_export_libraries(${PROJECT_NAME}_cpp)

# Add tests if building tests
if(BUILD_TESTING)
    find_package(ament_cmake_pytest REQUIRED)
    find_package(ament_lint_auto REQUIRED)
    find_package(ament_cmake_cppcheck REQUIRED)
    find_package(ament_cmake_cpplint REQUIRED)
    find_package(ament_cmake_uncrustify REQUIRED)
    find_package(ament_cmake_clang_format REQUIRED)
    
    # Enable linting
    ament_lint_auto_find_test_dependencies()
    
    # Add Python tests
    ament_add_pytest_test(test_preprocessing
        test/test_preprocessing.py
        TIMEOUT 30
    )
    
    ament_add_pytest_test(test_postprocessing
        test/test_postprocessing.py
        TIMEOUT 30
    )
    
    ament_add_pytest_test(test_visualization
        test/test_visualization.py
        TIMEOUT 30
    )
    
    ament_add_pytest_test(test_yolo_detector
        test/test_yolo_detector.py
        TIMEOUT 60
    )
    
    # Add performance tests
    add_executable(benchmark_yolo
        test/benchmark_yolo.cpp
    )
    target_link_libraries(benchmark_yolo
        ${PROJECT_NAME}_cpp
        ${rclcpp_LIBRARIES}
        ${OpenCV_LIBRARIES}
    )
    
    add_executable(benchmark_preprocessing
        test/benchmark_preprocessing.cpp
    )
    target_link_libraries(benchmark_preprocessing
        ${PROJECT_NAME}_cpp
        ${OpenCV_LIBRARIES}
    )
    
    # Add integration tests
    ament_add_pytest_test(test_integration
        test/test_integration.py
        TIMEOUT 120
    )
endif()

# Finalize
ament_package()

# Performance optimization flags for release builds
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        target_compile_options(${PROJECT_NAME}_cpp PRIVATE
            -flto
            -fno-signed-zeros
            -fno-trapping-math
            -fassociative-math
            -frename-registers
            -ftree-vectorize
            -fopt-info-vec
        )
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -flto")
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(${PROJECT_NAME}_cpp PRIVATE
            -flto=thin
            -ffast-math
            -fvectorize
            -fslp-vectorize
            -mllvm -vectorize-slp-aggressive
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
        target_compile_options(${PROJECT_NAME}_cpp PRIVATE
            -fsanitize=address
            -fsanitize=undefined
            -fsanitize=leak
            -fno-omit-frame-pointer
            -fno-optimize-sibling-calls
        )
        target_link_options(${PROJECT_NAME}_cpp PRIVATE
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

# Custom targets for development
add_custom_target(format
    COMMAND find ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/include
            -name "*.cpp" -o -name "*.hpp" -o -name "*.h"
            | xargs clang-format -i -style=file
    COMMAND find ${CMAKE_CURRENT_SOURCE_DIR}/perception_stack ${CMAKE_CURRENT_SOURCE_DIR}/scripts
            -name "*.py" | xargs black --line-length 88
    COMMAND find ${CMAKE_CURRENT_SOURCE_DIR}/perception_stack ${CMAKE_CURRENT_SOURCE_DIR}/scripts
            -name "*.py" | xargs isort
    COMMENT "Formatting source code"
    VERBATIM
)

add_custom_target(lint
    COMMAND ament_cpplint ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/include
    COMMAND ament_uncrustify --reformat ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/include
    COMMAND flake8 ${CMAKE_CURRENT_SOURCE_DIR}/perception_stack ${CMAKE_CURRENT_SOURCE_DIR}/scripts
    COMMAND mypy ${CMAKE_CURRENT_SOURCE_DIR}/perception_stack ${CMAKE_CURRENT_SOURCE_DIR}/scripts
    COMMENT "Running linting and static analysis"
    VERBATIM
)

add_custom_target(docs
    COMMAND sphinx-build -b html ${CMAKE_CURRENT_SOURCE_DIR}/docs/source ${CMAKE_CURRENT_SOURCE_DIR}/docs/build
    COMMENT "Generating API documentation"
    VERBATIM
)

add_custom_target(coverage
    COMMAND cd ${CMAKE_BINARY_DIR} && 
            gcovr -r ${CMAKE_CURRENT_SOURCE_DIR} --exclude-unreachable-branches 
            --exclude-throw-branches --html --html-details -o coverage.html
    COMMENT "Generating code coverage report"
    VERBATIM
)

# Add model downloading target
add_custom_target(download_models
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/scripts/download_models.py
    COMMENT "Downloading pretrained models"
    VERBATIM
)

# Add calibration target
add_custom_target(calibrate_camera
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/scripts/camera_calibration.py
            --mode calibrate
    COMMENT "Running camera calibration"
    VERBATIM
)

# Print configuration summary
message(STATUS "=========================================")
message(STATUS "Perception Stack Configuration")
message(STATUS "=========================================")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Python: ${Python3_VERSION}")
message(STATUS "OpenCV: ${OpenCV_VERSION}")
message(STATUS "TensorRT: ${USE_TENSORRT}")
message(STATUS "CUDA: ${USE_CUDA}")
message(STATUS "OpenVINO: ${USE_OPENVINO}")
message(STATUS "OpenMP: ${OpenMP_CXX_VERSION}")
message(STATUS "Tests: ${BUILD_TESTING}")
message(STATUS "Sanitizers: ${CMAKE_BUILD_TYPE STREQUAL 'Debug'}")
message(STATUS "=========================================")

# Post-build steps for production deployment
add_custom_command(TARGET ${PROJECT_NAME}_cpp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Optimizing binary..."
    COMMAND strip -s $<TARGET_FILE:${PROJECT_NAME}_cpp> 2>/dev/null || true
    COMMENT "Stripping debug symbols from production build"
)

# Add dependency checking
add_custom_target(check_dependencies
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/scripts/check_dependencies.py
    COMMENT "Checking system dependencies"
    VERBATIM
)

# Create a convenience target for building everything
add_custom_target(build_all
    DEPENDS ${PROJECT_NAME}_cpp pybindings yolo_detector_node object_tracker_node perception_manager
    COMMENT "Building all perception stack components"
)

# Add installation verification
add_custom_target(verify_install
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/scripts/verify_installation.py
    COMMENT "Verifying installation"
    VERBATIM
)
