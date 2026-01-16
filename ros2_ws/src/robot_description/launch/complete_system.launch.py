#!/usr/bin/env python3
"""
Complete ML Object Picking Robot System Launch
Starts: Gazebo + Robot + YOLO Detection + Pick-Place Coordinator
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import Command
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    
    # Package paths
    robot_desc_pkg = get_package_share_directory('robot_description')
    urdf_file = os.path.join(robot_desc_pkg, 'urdf', 'robot.urdf.xacro')
    
    robot_description = ParameterValue(
        Command(['xacro ', urdf_file]),
        value_type=str
    )
    
    # 1. Launch Gazebo simulation
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(robot_desc_pkg, 'launch', 'gazebo.launch.py')
        )
    )
    
    # 2. YOLO Object Detector (delayed start)
    yolo_detector = TimerAction(
        period=5.0,  # Wait 5s for Gazebo to stabilize
        actions=[
            Node(
                package='perception_stack',
                executable='yolo_detector_node',
                name='yolo_detector',
                output='screen',
                parameters=[{
                    'camera_topic': '/camera/camera/image_raw',
                    'confidence_threshold': 0.6,
                    'model_path': 'yolov5s'
                }]
            )
        ]
    )
    
    # 3. Pick-Place Coordinator (delayed start)
    pick_place_coordinator = TimerAction(
        period=7.0,  # Wait for detector to initialize
        actions=[
            Node(
                package='task_planning',
                executable='pick_place_coordinator',
                name='pick_place_coordinator',
                output='screen',
                parameters=[{
                    'min_confidence': 0.6,
                    'target_classes': ['bottle', 'cup', 'box']
                }]
            )
        ]
    )
    
    return LaunchDescription([
        gazebo_launch,
        yolo_detector,
        pick_place_coordinator
    ])
