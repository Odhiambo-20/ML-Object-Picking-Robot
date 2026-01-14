#!/bin/bash
docker exec -it ml-robot-workspace bash -c '
    source /opt/ros/humble/setup.bash
    cd /workspace/ros2_ws
    [ -f install/setup.bash ] && source install/setup.bash
    echo "🤖 ROS2 Humble workspace ready!"
    echo "Location: /workspace/ros2_ws"
    exec bash
'
