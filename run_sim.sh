#!/bin/bash
cd /workspace/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export DISPLAY=:0
xhost +local:root 2>/dev/null || true
ros2 launch robot_description gazebo.launch.py gui:=true
