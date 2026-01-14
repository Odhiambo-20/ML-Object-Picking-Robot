#!/bin/bash
set -e

echo "🐳 Starting Docker container..."
docker stop ml-robot-workspace 2>/dev/null || true
docker rm ml-robot-workspace 2>/dev/null || true

docker run -d \
  --name ml-robot-workspace \
  --network host \
  --privileged \
  -v $(pwd):/workspace \
  -v /dev:/dev \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -w /workspace \
  osrf/ros:humble-desktop-full \
  tail -f /dev/null

echo "⏳ Waiting for container to start..."
sleep 3

echo "✅ Container started!"
docker ps | grep ml-robot

echo ""
echo "📦 Installing dependencies..."
docker exec ml-robot-workspace bash -c '
    apt-get update -qq
    apt-get install -y python3-pip python3-colcon-common-extensions python3-rosdep vim nano
    pip3 install numpy==1.26.4 scipy opencv-python transitions matplotlib pyyaml
    echo "✅ Dependencies installed"
'

echo ""
echo "🏗️ Building ROS2 workspace..."
docker exec ml-robot-workspace bash -c '
    source /opt/ros/humble/setup.bash
    cd /workspace/ros2_ws
    rosdep install --from-paths src --ignore-src -r -y 2>/dev/null || true
    colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release
    echo "✅ Build complete"
'

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Enter container with: ./enter.sh"
