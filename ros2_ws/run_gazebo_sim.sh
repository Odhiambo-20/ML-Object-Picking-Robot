#!/bin/bash

# Fix X11 authorization
export DISPLAY=:0
xhost +local:root 2>/dev/null || true

# Source workspace
cd /workspace/ros2_ws
source install/setup.bash

# Run Gazebo
echo "Starting Gazebo simulation..."
ros2 launch robot_description gazebo.launch.py
