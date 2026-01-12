#!/usr/bin/env python3
"""
robot_control.launch.py - Production-grade ROS2 launch file for robot control

This launch file starts all robot control nodes including:
- Servo controller for manipulator arm
- Motor driver for mobile base (if applicable)
- Gripper controller for end effector
- Hardware interface node
- Joint state publisher
- Controller manager
- Safety monitor

Author: Victor's Production Team
Date: 2025-01-12
Version: 1.0.0

Usage:
    ros2 launch robot_control robot_control.launch.py
    ros2 launch robot_control robot_control.launch.py use_sim:=true
    ros2 launch robot_control robot_control.launch.py debug:=true
    ros2 launch robot_control robot_control.launch.py log_level:=debug
"""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    LogInfo,
    RegisterEventHandler,
    TimerAction,
    EmitEvent,
    Shutdown
)
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import (
    OnProcessStart,
    OnProcessExit,
    OnExecutionComplete
)
from launch.events import Shutdown as ShutdownEvent
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PythonExpression,
    PathJoinSubstitution,
    Command,
    FindExecutable
)

from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare
from launch_ros.descriptions import ParameterValue


def generate_launch_description():
    """
    Generate launch description for robot control system.
    
    Returns:
        LaunchDescription: Complete launch configuration
    """
    
    # ========================================================================
    # PACKAGE PATHS
    # ========================================================================
    
    pkg_robot_control = get_package_share_directory('robot_control')
    pkg_robot_description = get_package_share_directory('robot_description')
    
    # ========================================================================
    # CONFIGURATION FILES
    # ========================================================================
    
    # Servo calibration file
    servo_calibration_file = os.path.join(
        pkg_robot_control,
        'config',
        'servo_calibration.yaml'
    )
    
    # Motor parameters file
    motor_params_file = os.path.join(
        pkg_robot_control,
        'config',
        'motor_params.yaml'
    )
    
    # Robot description URDF
    robot_description_file = os.path.join(
        pkg_robot_description,
        'urdf',
        'robot.urdf.xacro'
    )
    
    # Controller configuration
    controller_config_file = os.path.join(
        pkg_robot_control,
        'config',
        'robot_controllers.yaml'
    )
    
    # ========================================================================
    # LAUNCH ARGUMENTS
    # ========================================================================
    
    # Declare launch arguments
    use_sim_arg = DeclareLaunchArgument(
        'use_sim',
        default_value='false',
        description='Use simulation (Gazebo) instead of real hardware'
    )
    
    debug_arg = DeclareLaunchArgument(
        'debug',
        default_value='false',
        description='Enable debug mode with verbose logging'
    )
    
    log_level_arg = DeclareLaunchArgument(
        'log_level',
        default_value='info',
        choices=['debug', 'info', 'warn', 'error', 'fatal'],
        description='ROS logging level'
    )
    
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='false',
        description='Launch RViz for visualization'
    )
    
    robot_name_arg = DeclareLaunchArgument(
        'robot_name',
        default_value='ml_robot',
        description='Name of the robot'
    )
    
    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='Robot namespace'
    )
    
    enable_safety_arg = DeclareLaunchArgument(
        'enable_safety',
        default_value='true',
        description='Enable safety monitoring'
    )
    
    control_rate_arg = DeclareLaunchArgument(
        'control_rate',
        default_value='50',
        description='Control loop rate (Hz)'
    )
    
    # ========================================================================
    # LAUNCH CONFIGURATIONS
    # ========================================================================
    
    use_sim = LaunchConfiguration('use_sim')
    debug = LaunchConfiguration('debug')
    log_level = LaunchConfiguration('log_level')
    use_rviz = LaunchConfiguration('use_rviz')
    robot_name = LaunchConfiguration('robot_name')
    namespace = LaunchConfiguration('namespace')
    enable_safety = LaunchConfiguration('enable_safety')
    control_rate = LaunchConfiguration('control_rate')
    
    # ========================================================================
    # ROBOT DESCRIPTION
    # ========================================================================
    
    # Process robot description with xacro
    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name='xacro')]),
        ' ',
        robot_description_file,
        ' ',
        'use_sim:=', use_sim,
        ' ',
        'robot_name:=', robot_name
    ])
    
    robot_description = {
        'robot_description': ParameterValue(robot_description_content, value_type=str)
    }
    
    # ========================================================================
    # NODES - HARDWARE INTERFACE
    # ========================================================================
    
    # Servo controller node (C++ or Python based on implementation)
    servo_controller_node = Node(
        package='robot_control',
        executable='servo_controller_node',
        name='servo_controller',
        namespace=namespace,
        output='screen',
        emulate_tty=True,
        parameters=[
            servo_calibration_file,
            {
                'use_sim': use_sim,
                'control_rate': control_rate,
                'enable_safety': enable_safety,
            }
        ],
        arguments=['--ros-args', '--log-level', log_level],
        condition=UnlessCondition(use_sim),
        respawn=True,
        respawn_delay=2.0
    )
    
    # Motor driver node (for mobile base or additional motors)
    motor_driver_node = Node(
        package='robot_control',
        executable='motor_driver_node',
        name='motor_driver',
        namespace=namespace,
        output='screen',
        emulate_tty=True,
        parameters=[
            motor_params_file,
            {
                'use_sim': use_sim,
                'control_rate': control_rate,
            }
        ],
        arguments=['--ros-args', '--log-level', log_level],
        condition=UnlessCondition(use_sim),
        respawn=True,
        respawn_delay=2.0
    )
    
    # Gripper controller node
    gripper_controller_node = Node(
        package='robot_control',
        executable='gripper_controller_node',
        name='gripper_controller',
        namespace=namespace,
        output='screen',
        emulate_tty=True,
        parameters=[
            servo_calibration_file,
            {
                'use_sim': use_sim,
                'control_rate': control_rate,
                'enable_slip_detection': True,
                'max_force': 50.0,  # Newtons
            }
        ],
        arguments=['--ros-args', '--log-level', log_level],
        respawn=True,
        respawn_delay=2.0
    )
    
    # Hardware interface node (low-level GPIO/I2C/SPI interface)
    hardware_interface_node = Node(
        package='robot_control',
        executable='hardware_interface_node',
        name='hardware_interface',
        namespace=namespace,
        output='screen',
        emulate_tty=True,
        parameters=[
            {
                'i2c_bus': 1,
                'pca9685_address': 0x40,
                'pwm_frequency': 50,
                'gpio_chip': 'gpiochip0',
            }
        ],
        arguments=['--ros-args', '--log-level', log_level],
        condition=UnlessCondition(use_sim),
        respawn=True,
        respawn_delay=2.0
    )
    
    # ========================================================================
    # NODES - STATE PUBLISHING
    # ========================================================================
    
    # Robot state publisher (publishes TF tree from URDF)
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        namespace=namespace,
        output='screen',
        parameters=[robot_description],
        arguments=['--ros-args', '--log-level', log_level]
    )
    
    # Joint state publisher (aggregates joint states)
    joint_state_publisher_node = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        namespace=namespace,
        output='screen',
        parameters=[
            robot_description,
            {
                'source_list': [
                    'servo_controller/joint_states',
                    'motor_driver/joint_states',
                    'gripper_controller/joint_states'
                ],
                'rate': control_rate,
            }
        ],
        arguments=['--ros-args', '--log-level', log_level]
    )
    
    # ========================================================================
    # NODES - SAFETY AND MONITORING
    # ========================================================================
    
    # Safety monitor node
    safety_monitor_node = Node(
        package='robot_control',
        executable='safety_monitor_node',
        name='safety_monitor',
        namespace=namespace,
        output='screen',
        emulate_tty=True,
        parameters=[
            {
                'enable_safety': enable_safety,
                'emergency_stop_timeout': 2.0,  # seconds
                'joint_velocity_limits': True,
                'collision_detection': True,
                'workspace_limits': True,
                'max_joint_velocities': [60.0, 60.0, 60.0, 60.0, 60.0, 60.0],  # deg/s
            }
        ],
        arguments=['--ros-args', '--log-level', log_level],
        condition=IfCondition(enable_safety),
        respawn=True,
        respawn_delay=1.0
    )
    
    # System monitor node (performance, health checks)
    system_monitor_node = Node(
        package='robot_control',
        executable='system_monitor_node',
        name='system_monitor',
        namespace=namespace,
        output='screen',
        parameters=[
            {
                'monitor_rate': 1.0,  # Hz
                'check_cpu_temp': True,
                'check_voltage': True,
                'check_current': True,
                'cpu_temp_limit': 80.0,  # °C
                'voltage_min': 5.5,  # V
                'voltage_max': 6.5,  # V
            }
        ],
        arguments=['--ros-args', '--log-level', log_level]
    )
    
    # ========================================================================
    # NODES - VISUALIZATION (OPTIONAL)
    # ========================================================================
    
    # RViz visualization
    rviz_config_file = os.path.join(
        pkg_robot_control,
        'config',
        'robot_control.rviz'
    )
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file],
        condition=IfCondition(use_rviz),
        parameters=[
            {'use_sim_time': use_sim}
        ]
    )
    
    # ========================================================================
    # SIMULATION NODES (GAZEBO)
    # ========================================================================
    
    # Gazebo simulation (only if use_sim=true)
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('robot_description'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'world': 'warehouse',
            'robot_name': robot_name,
        }.items(),
        condition=IfCondition(use_sim)
    )
    
    # ========================================================================
    # EVENT HANDLERS
    # ========================================================================
    
    # Log when servo controller starts
    servo_controller_started_handler = RegisterEventHandler(
        OnProcessStart(
            target_action=servo_controller_node,
            on_start=[
                LogInfo(msg='Servo controller started successfully'),
            ]
        )
    )
    
    # Log when gripper controller starts
    gripper_controller_started_handler = RegisterEventHandler(
        OnProcessStart(
            target_action=gripper_controller_node,
            on_start=[
                LogInfo(msg='Gripper controller started successfully'),
            ]
        )
    )
    
    # Handle critical node failures
    critical_node_exit_handler = RegisterEventHandler(
        OnProcessExit(
            target_action=servo_controller_node,
            on_exit=[
                LogInfo(msg='CRITICAL: Servo controller exited! Shutting down system...'),
                EmitEvent(event=ShutdownEvent(reason='Critical node failure'))
            ]
        )
    )
    
    # ========================================================================
    # DELAYED STARTUP (SEQUENTIAL INITIALIZATION)
    # ========================================================================
    
    # Start hardware interface first, then controllers
    delayed_servo_controller = TimerAction(
        period=2.0,  # Wait 2 seconds for hardware to initialize
        actions=[servo_controller_node]
    )
    
    delayed_motor_driver = TimerAction(
        period=2.5,
        actions=[motor_driver_node]
    )
    
    delayed_gripper_controller = TimerAction(
        period=3.0,
        actions=[gripper_controller_node]
    )
    
    # ========================================================================
    # GROUPED ACTIONS
    # ========================================================================
    
    # Hardware nodes group (real hardware only)
    hardware_nodes_group = GroupAction(
        actions=[
            hardware_interface_node,
            delayed_servo_controller,
            delayed_motor_driver,
            delayed_gripper_controller,
        ],
        condition=UnlessCondition(use_sim),
        scoped=True
    )
    
    # State publishing nodes group
    state_publishing_group = GroupAction(
        actions=[
            robot_state_publisher_node,
            joint_state_publisher_node,
        ]
    )
    
    # Monitoring nodes group
    monitoring_group = GroupAction(
        actions=[
            safety_monitor_node,
            system_monitor_node,
        ]
    )
    
    # ========================================================================
    # GLOBAL PARAMETERS
    # ========================================================================
    
    # Set global ROS parameters
    set_use_sim_time = SetParameter(
        name='use_sim_time',
        value=use_sim
    )
    
    # ========================================================================
    # LAUNCH DESCRIPTION
    # ========================================================================
    
    return LaunchDescription([
        # Arguments
        use_sim_arg,
        debug_arg,
        log_level_arg,
        use_rviz_arg,
        robot_name_arg,
        namespace_arg,
        enable_safety_arg,
        control_rate_arg,
        
        # Global parameters
        set_use_sim_time,
        
        # Startup message
        LogInfo(
            msg=[
                '\n',
                '═══════════════════════════════════════════════════════════\n',
                '  ML-Based Object Picking Robot - Control System Launch\n',
                '═══════════════════════════════════════════════════════════\n',
                'Configuration:\n',
                '  • Robot Name: ', robot_name, '\n',
                '  • Simulation: ', use_sim, '\n',
                '  • Debug Mode: ', debug, '\n',
                '  • Log Level: ', log_level, '\n',
                '  • Safety Monitor: ', enable_safety, '\n',
                '  • Control Rate: ', control_rate, ' Hz\n',
                '═══════════════════════════════════════════════════════════\n'
            ]
        ),
        
        # Event handlers
        servo_controller_started_handler,
        gripper_controller_started_handler,
        critical_node_exit_handler,
        
        # Node groups
        hardware_nodes_group,
        state_publishing_group,
        monitoring_group,
        
        # Optional nodes
        rviz_node,
        
        # Simulation
        gazebo_launch,
        
        # Completion message
        LogInfo(
            condition=UnlessCondition(use_sim),
            msg='Robot control system launched on REAL HARDWARE - BE CAREFUL!'
        ),
        LogInfo(
            condition=IfCondition(use_sim),
            msg='Robot control system launched in SIMULATION mode'
        ),
    ])


def main():
    """
    Main entry point (for testing).
    """
    from launch import LaunchService
    
    ls = LaunchService()
    ls.include_launch_description(generate_launch_description())
    return ls.run()


if __name__ == '__main__':
    import sys
    sys.exit(main())
