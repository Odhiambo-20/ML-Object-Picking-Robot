#!/bin/bash
# This script runs INSIDE the Docker container

cd /workspace/ros2_ws

echo "=== Fixing perception_stack setup.py ==="
cat > /workspace/ros2_ws/src/perception_stack/setup.py <<'EOF'
from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'perception_stack'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Victor',
    maintainer_email='victor@production-team.com',
    description='ML-based perception for object picking robot',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [],
    },
)
EOF

echo "=== Creating resource directory ==="
mkdir -p /workspace/ros2_ws/src/perception_stack/resource
touch /workspace/ros2_ws/src/perception_stack/resource/perception_stack
mkdir -p /workspace/ros2_ws/src/perception_stack/perception_stack
touch /workspace/ros2_ws/src/perception_stack/perception_stack/__init__.py

echo "=== Fixing all other Python package structures ==="
for pkg in robot_control task_planning system_monitor; do
    echo "Fixing ${pkg}..."
    mkdir -p "/workspace/ros2_ws/src/${pkg}/resource"
    touch "/workspace/ros2_ws/src/${pkg}/resource/${pkg}"
    mkdir -p "/workspace/ros2_ws/src/${pkg}/${pkg}"
    touch "/workspace/ros2_ws/src/${pkg}/${pkg}/__init__.py"
    
    # Create minimal setup.py if needed
    if [ ! -f "/workspace/ros2_ws/src/${pkg}/setup.py" ] || grep -q "✓" "/workspace/ros2_ws/src/${pkg}/setup.py" 2>/dev/null; then
        cat > "/workspace/ros2_ws/src/${pkg}/setup.py" <<SETUP_EOF
from setuptools import setup, find_packages

package_name = '${pkg}'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Victor',
    maintainer_email='victor@production-team.com',
    description='${pkg} package',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [],
    },
)
SETUP_EOF
    fi
done

echo ""
echo "=== Cleaning previous build ==="
rm -rf build install log

echo ""
echo "=== Sourcing ROS2 Humble ==="
source /opt/ros/humble/setup.bash

echo ""
echo "=== Building workspace ==="
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "SUCCESS! Build completed successfully!"
    echo "========================================"
    echo ""
    echo "To use the workspace, run:"
    echo "  source /workspace/ros2_ws/install/setup.bash"
else
    echo ""
    echo "========================================"
    echo "Build failed - see errors above"
    echo "========================================"
    exit 1
fi
