#!/usr/bin/env python3
# ml-object-picking-robot/ros2_ws/src/motion_planning/launch/motion_planning.launch.py
# Industrial-grade motion planning system launch configuration
# Complete launch file with all motion planning components

import os
import yaml
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    ExecuteProcess,
    RegisterEventHandler,
    LogInfo,
    TimerAction,
    SetEnvironmentVariable,
    OpaqueFunction,
)
from launch.event_handlers import OnProcessStart, OnProcessExit, OnShutdown
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
    EnvironmentVariable,
    Command,
    FindExecutable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node, ComposableNodeContainer, LoadComposableNodes
from launch_ros.substitutions import FindPackageShare
from launch_ros.descriptions import ComposableNode
from launch_ros.parameter_descriptions import ParameterFile
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # ============================================
    # LAUNCH ARGUMENTS
    # ============================================
    
    # Core configuration arguments
    declare_config_package_arg = DeclareLaunchArgument(
        'config_package',
        default_value='motion_planning',
        description='Package containing configuration files'
    )
    
    declare_config_path_arg = DeclareLaunchArgument(
        'config_path',
        default_value='config',
        description='Path to configuration files within package'
    )
    
    declare_environment_arg = DeclareLaunchArgument(
        'environment',
        default_value='production',
        description='Environment configuration: production, staging, development',
        choices=['production', 'staging', 'development']
    )
    
    declare_robot_model_arg = DeclareLaunchArgument(
        'robot_model',
        default_value='ur10e',
        description='Robot model configuration',
        choices=['ur10e', 'ur5e', 'ur3e', 'ur16e']
    )
    
    # Component enable/disable arguments
    declare_enable_ik_arg = DeclareLaunchArgument(
        'enable_ik',
        default_value='true',
        description='Enable inverse kinematics solver'
    )
    
    declare_enable_collision_checking_arg = DeclareLaunchArgument(
        'enable_collision_checking',
        default_value='true',
        description='Enable collision checking system'
    )
    
    declare_enable_path_optimization_arg = DeclareLaunchArgument(
        'enable_path_optimization',
        default_value='true',
        description='Enable path optimization'
    )
    
    declare_enable_trajectory_planning_arg = DeclareLaunchArgument(
        'enable_trajectory_planning',
        default_value='true',
        description='Enable trajectory planning'
    )
    
    declare_enable_visualization_arg = DeclareLaunchArgument(
        'enable_visualization',
        default_value='true',
        description='Enable visualization nodes'
    )
    
    declare_enable_monitoring_arg = DeclareLaunchArgument(
        'enable_monitoring',
        default_value='true',
        description='Enable system monitoring'
    )
    
    # Performance arguments
    declare_use_composition_arg = DeclareLaunchArgument(
        'use_composition',
        default_value='false',
        description='Use composed nodes for better performance'
    )
    
    declare_container_name_arg = DeclareLaunchArgument(
        'container_name',
        default_value='motion_planning_container',
        description='Name of the container for composed nodes'
    )
    
    declare_num_threads_arg = DeclareLaunchArgument(
        'num_threads',
        default_value='4',
        description='Number of threads for parallel processing'
    )
    
    # Safety arguments
    declare_safety_enabled_arg = DeclareLaunchArgument(
        'safety_enabled',
        default_value='true',
        description='Enable safety systems'
    )
    
    declare_emergency_stop_timeout_arg = DeclareLaunchArgument(
        'emergency_stop_timeout',
        default_value='0.1',
        description='Emergency stop response timeout in seconds'
    )
    
    # Debug arguments
    declare_debug_arg = DeclareLaunchArgument(
        'debug',
        default_value='false',
        description='Enable debug output and profiling'
    )
    
    declare_log_level_arg = DeclareLaunchArgument(
        'log_level',
        default_value='info',
        description='Log level for all nodes',
        choices=['debug', 'info', 'warn', 'error', 'fatal']
    )
    
    # ============================================
    # LAUNCH CONFIGURATIONS
    # ============================================
    
    config_package = LaunchConfiguration('config_package')
    config_path = LaunchConfiguration('config_path')
    environment = LaunchConfiguration('environment')
    robot_model = LaunchConfiguration('robot_model')
    
    enable_ik = LaunchConfiguration('enable_ik')
    enable_collision_checking = LaunchConfiguration('enable_collision_checking')
    enable_path_optimization = LaunchConfiguration('enable_path_optimization')
    enable_trajectory_planning = LaunchConfiguration('enable_trajectory_planning')
    enable_visualization = LaunchConfiguration('enable_visualization')
    enable_monitoring = LaunchConfiguration('enable_monitoring')
    
    use_composition = LaunchConfiguration('use_composition')
    container_name = LaunchConfiguration('container_name')
    num_threads = LaunchConfiguration('num_threads')
    
    safety_enabled = LaunchConfiguration('safety_enabled')
    emergency_stop_timeout = LaunchConfiguration('emergency_stop_timeout')
    
    debug = LaunchConfiguration('debug')
    log_level = LaunchConfiguration('log_level')
    
    # ============================================
    # PATH AND FILE RESOLUTION
    # ============================================
    
    # Get package directories
    motion_planning_pkg = FindPackageShare('motion_planning')
    robot_description_pkg = FindPackageShare('robot_description')
    
    # Configuration file paths
    global_config_path = PathJoinSubstitution([
        FindPackageShare('ml_object_picking_robot'),
        'config',
        environment,
        'motion_planning.yaml'
    ])
    
    motion_params_path = PathJoinSubstitution([
        motion_planning_pkg,
        'config',
        'motion_params.yaml'
    ])
    
    kdl_config_path = PathJoinSubstitution([
        motion_planning_pkg,
        'config',
        'kdl_config.yaml'
    ])
    
    safety_limits_path = PathJoinSubstitution([
        motion_planning_pkg,
        'config',
        'safety_limits.yaml'
    ])
    
    # Robot description
    robot_description_path = PathJoinSubstitution([
        robot_description_pkg,
        'urdf',
        'robot.urdf.xacro'
    ])
    
    robot_description = Command([
        'xacro ', robot_description_path,
        ' robot_model:=', robot_model,
        ' environment:=', environment,
        ' safety_enabled:=', safety_enabled
    ])
    
    # ============================================
    # COMPOSABLE NODES CONFIGURATION
    # ============================================
    
    # Inverse Kinematics Solver composable node
    ik_composable_node = ComposableNode(
        package='motion_planning',
        plugin='motion_planning::InverseKinematicsSolver',
        name='inverse_kinematics_solver',
        namespace='motion_planning',
        parameters=[
            # Load multiple parameter files
            ParameterFile(motion_params_path, allow_substs=True),
            ParameterFile(kdl_config_path, allow_substs=True),
            ParameterFile(global_config_path, allow_substs=True),
            {
                'robot_description': robot_description,
                'use_sim_time': False,
                'num_threads': num_threads,
                'log_level': log_level,
                'debug': debug,
                'safety.enabled': safety_enabled,
                'environment': environment,
                'robot_model': robot_model,
                # IK-specific parameters
                'ik.enable_caching': True,
                'ik.cache_size': 10000,
                'ik.max_solutions': 8,
                'ik.position_tolerance': 0.001,
                'ik.orientation_tolerance': 0.01,
                'ik.max_iterations': 500,
                'ik.timeout': 0.1,
            }
        ],
        extra_arguments=[
            {'use_intra_process_comms': True}
        ]
    )
    
    # Collision Checker composable node
    collision_checker_composable_node = ComposableNode(
        package='motion_planning',
        plugin='motion_planning::CollisionChecker',
        name='collision_checker',
        namespace='motion_planning',
        parameters=[
            ParameterFile(motion_params_path, allow_substs=True),
            ParameterFile(safety_limits_path, allow_substs=True),
            ParameterFile(global_config_path, allow_substs=True),
            {
                'robot_description': robot_description,
                'use_sim_time': False,
                'num_threads': num_threads,
                'log_level': log_level,
                'debug': debug,
                'safety.enabled': safety_enabled,
                'environment': environment,
                'robot_model': robot_model,
                # Collision checking specific parameters
                'collision.enable_continuous_checking': True,
                'collision.enable_self_collision': True,
                'collision.enable_environment_collision': True,
                'collision.safety_margin': 0.05,
                'collision.warning_distance': 0.2,
                'collision.emergency_distance': 0.05,
                'collision.max_check_time': 0.01,
                'collision.enable_visualization': enable_visualization,
            }
        ],
        extra_arguments=[
            {'use_intra_process_comms': True}
        ]
    )
    
    # Path Optimizer composable node
    path_optimizer_composable_node = ComposableNode(
        package='motion_planning',
        plugin='motion_planning::PathOptimizer',
        name='path_optimizer',
        namespace='motion_planning',
        parameters=[
            ParameterFile(motion_params_path, allow_substs=True),
            ParameterFile(global_config_path, allow_substs=True),
            {
                'robot_description': robot_description,
                'use_sim_time': False,
                'num_threads': num_threads,
                'log_level': log_level,
                'debug': debug,
                'safety.enabled': safety_enabled,
                'environment': environment,
                'robot_model': robot_model,
                # Path optimization specific parameters
                'optimization.enable_shortcutting': True,
                'optimization.enable_smoothing': True,
                'optimization.enable_simplification': True,
                'optimization.time_optimal.enabled': True,
                'optimization.max_velocity_scale': 0.8,
                'optimization.max_acceleration_scale': 0.7,
                'optimization.max_jerk_scale': 0.6,
                'optimization.enable_caching': True,
                'optimization.cache_size': 5000,
            }
        ],
        extra_arguments=[
            {'use_intra_process_comms': True}
        ]
    )
    
    # Trajectory Planner composable node
    trajectory_planner_composable_node = ComposableNode(
        package='motion_planning',
        plugin='motion_planning::TrajectoryPlanner',
        name='trajectory_planner',
        namespace='motion_planning',
        parameters=[
            ParameterFile(motion_params_path, allow_substs=True),
            ParameterFile(global_config_path, allow_substs=True),
            {
                'robot_description': robot_description,
                'use_sim_time': False,
                'num_threads': num_threads,
                'log_level': log_level,
                'debug': debug,
                'safety.enabled': safety_enabled,
                'environment': environment,
                'robot_model': robot_model,
                # Trajectory planning specific parameters
                'planning.primary_planner': 'rrt_star',
                'planning.max_time': 5.0,
                'planning.goal_tolerance.position': 0.005,
                'planning.goal_tolerance.orientation': 0.01,
                'planning.enable_replanning': True,
                'planning.enable_visualization': enable_visualization,
                'execution.control_frequency': 125.0,
                'execution.enable_preemption': True,
                'execution.realtime_tolerance': 0.001,
            }
        ],
        extra_arguments=[
            {'use_intra_process_comms': True}
        ]
    )
    
    # Motion Planning Manager composable node
    motion_planning_manager_composable_node = ComposableNode(
        package='motion_planning',
        plugin='motion_planning::MotionPlanningManager',
        name='motion_planning_manager',
        namespace='motion_planning',
        parameters=[
            ParameterFile(motion_params_path, allow_substs=True),
            ParameterFile(global_config_path, allow_substs=True),
            {
                'robot_description': robot_description,
                'use_sim_time': False,
                'num_threads': num_threads,
                'log_level': log_level,
                'debug': debug,
                'safety.enabled': safety_enabled,
                'environment': environment,
                'robot_model': robot_model,
                # Manager specific parameters
                'manager.coordination_timeout': 5.0,
                'manager.health_check_interval': 1.0,
                'manager.enable_fault_recovery': True,
                'manager.max_recovery_attempts': 3,
                'manager.enable_performance_monitoring': enable_monitoring,
            }
        ],
        extra_arguments=[
            {'use_intra_process_comms': True}
        ]
    )
    
    # ============================================
    # REGULAR NODES (for non-composed deployment)
    # ============================================
    
    # Inverse Kinematics Solver node
    ik_node = Node(
        condition=UnlessCondition(use_composition),
        package='motion_planning',
        executable='inverse_kinematics_solver',
        name='inverse_kinematics_solver',
        namespace='motion_planning',
        output='screen',
        emulate_tty=True,
        parameters=[
            ParameterFile(motion_params_path, allow_substs=True),
            ParameterFile(kdl_config_path, allow_substs=True),
            ParameterFile(global_config_path, allow_substs=True),
            {
                'robot_description': robot_description,
                'use_sim_time': False,
                'num_threads': num_threads,
                'log_level': log_level,
                'debug': debug,
                'safety.enabled': safety_enabled,
                'environment': environment,
                'robot_model': robot_model,
                'ik.enable_caching': True,
                'ik.cache_size': 10000,
                'ik.max_solutions': 8,
                'ik.position_tolerance': 0.001,
                'ik.orientation_tolerance': 0.01,
                'ik.max_iterations': 500,
                'ik.timeout': 0.1,
            }
        ],
        remappings=[
            ('/joint_states', '/robot/joint_states'),
            ('/robot_description', '/robot/robot_description'),
        ]
    )
    
    # Collision Checker node
    collision_checker_node = Node(
        condition=UnlessCondition(use_composition),
        package='motion_planning',
        executable='collision_checker',
        name='collision_checker',
        namespace='motion_planning',
        output='screen',
        emulate_tty=True,
        parameters=[
            ParameterFile(motion_params_path, allow_substs=True),
            ParameterFile(safety_limits_path, allow_substs=True),
            ParameterFile(global_config_path, allow_substs=True),
            {
                'robot_description': robot_description,
                'use_sim_time': False,
                'num_threads': num_threads,
                'log_level': log_level,
                'debug': debug,
                'safety.enabled': safety_enabled,
                'environment': environment,
                'robot_model': robot_model,
                'collision.enable_continuous_checking': True,
                'collision.enable_self_collision': True,
                'collision.enable_environment_collision': True,
                'collision.safety_margin': 0.05,
                'collision.warning_distance': 0.2,
                'collision.emergency_distance': 0.05,
                'collision.max_check_time': 0.01,
                'collision.enable_visualization': enable_visualization,
            }
        ],
        remappings=[
            ('/joint_states', '/robot/joint_states'),
            ('/robot_description', '/robot/robot_description'),
            ('/collision_object', '/perception/collision_object'),
            ('/planning_scene', '/move_group/planning_scene'),
        ]
    )
    
    # Path Optimizer node
    path_optimizer_node = Node(
        condition=UnlessCondition(use_composition),
        package='motion_planning',
        executable='path_optimizer',
        name='path_optimizer',
        namespace='motion_planning',
        output='screen',
        emulate_tty=True,
        parameters=[
            ParameterFile(motion_params_path, allow_substs=True),
            ParameterFile(global_config_path, allow_substs=True),
            {
                'robot_description': robot_description,
                'use_sim_time': False,
                'num_threads': num_threads,
                'log_level': log_level,
                'debug': debug,
                'safety.enabled': safety_enabled,
                'environment': environment,
                'robot_model': robot_model,
                'optimization.enable_shortcutting': True,
                'optimization.enable_smoothing': True,
                'optimization.enable_simplification': True,
                'optimization.time_optimal.enabled': True,
                'optimization.max_velocity_scale': 0.8,
                'optimization.max_acceleration_scale': 0.7,
                'optimization.max_jerk_scale': 0.6,
                'optimization.enable_caching': True,
                'optimization.cache_size': 5000,
            }
        ],
        remappings=[
            ('/joint_states', '/robot/joint_states'),
            ('/robot_description', '/robot/robot_description'),
            ('/trajectory', '/motion_planning/trajectory'),
        ]
    )
    
    # Trajectory Planner node
    trajectory_planner_node = Node(
        condition=UnlessCondition(use_composition),
        package='motion_planning',
        executable='trajectory_planner',
        name='trajectory_planner',
        namespace='motion_planning',
        output='screen',
        emulate_tty=True,
        parameters=[
            ParameterFile(motion_params_path, allow_substs=True),
            ParameterFile(global_config_path, allow_substs=True),
            {
                'robot_description': robot_description,
                'use_sim_time': False,
                'num_threads': num_threads,
                'log_level': log_level,
                'debug': debug,
                'safety.enabled': safety_enabled,
                'environment': environment,
                'robot_model': robot_model,
                'planning.primary_planner': 'rrt_star',
                'planning.max_time': 5.0,
                'planning.goal_tolerance.position': 0.005,
                'planning.goal_tolerance.orientation': 0.01,
                'planning.enable_replanning': True,
                'planning.enable_visualization': enable_visualization,
                'execution.control_frequency': 125.0,
                'execution.enable_preemption': True,
                'execution.realtime_tolerance': 0.001,
            }
        ],
        remappings=[
            ('/joint_states', '/robot/joint_states'),
            ('/robot_description', '/robot/robot_description'),
            ('/trajectory', '/motion_planning/trajectory'),
            ('/follow_joint_trajectory', '/robot/follow_joint_trajectory'),
        ]
    )
    
    # Motion Planning Manager node
    motion_planning_manager_node = Node(
        condition=UnlessCondition(use_composition),
        package='motion_planning',
        executable='motion_planning_manager',
        name='motion_planning_manager',
        namespace='motion_planning',
        output='screen',
        emulate_tty=True,
        parameters=[
            ParameterFile(motion_params_path, allow_substs=True),
            ParameterFile(global_config_path, allow_substs=True),
            {
                'robot_description': robot_description,
                'use_sim_time': False,
                'num_threads': num_threads,
                'log_level': log_level,
                'debug': debug,
                'safety.enabled': safety_enabled,
                'environment': environment,
                'robot_model': robot_model,
                'manager.coordination_timeout': 5.0,
                'manager.health_check_interval': 1.0,
                'manager.enable_fault_recovery': True,
                'manager.max_recovery_attempts': 3,
                'manager.enable_performance_monitoring': enable_monitoring,
            }
        ],
        remappings=[
            ('/joint_states', '/robot/joint_states'),
            ('/robot_description', '/robot/robot_description'),
        ]
    )
    
    # ============================================
    # VISUALIZATION NODES
    # ============================================
    
    # RViz node for visualization
    rviz_node = Node(
        condition=IfCondition(enable_visualization),
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        namespace='motion_planning',
        output='screen',
        arguments=[
            '-d', PathJoinSubstitution([
                motion_planning_pkg,
                'config',
                'motion_planning.rviz'
            ])
        ],
        parameters=[{
            'use_sim_time': False,
        }]
    )
    
    # Robot State Publisher
    robot_state_publisher_node = Node(
        condition=IfCondition(enable_visualization),
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        namespace='motion_planning',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': False,
            'publish_frequency': 50.0,
            'frame_prefix': 'motion_planning/',
        }]
    )
    
    # Joint State Publisher GUI (for manual testing)
    joint_state_publisher_gui_node = Node(
        condition=IfCondition(PythonExpression([
            enable_visualization, ' and ', debug
        ])),
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        namespace='motion_planning',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'rate': 50.0,
        }],
        remappings=[
            ('/joint_states', '/motion_planning/joint_states'),
        ]
    )
    
    # ============================================
    # MONITORING AND DIAGNOSTICS
    # ============================================
    
    # System Monitor node
    system_monitor_node = Node(
        condition=IfCondition(enable_monitoring),
        package='system_monitor',
        executable='system_monitor_node',
        name='system_monitor',
        namespace='motion_planning',
        output='screen',
        parameters=[
            {
                'monitoring.enabled': True,
                'monitoring.update_rate': 1.0,
                'monitoring.cpu_threshold': 80.0,
                'monitoring.memory_threshold': 85.0,
                'monitoring.disk_threshold': 90.0,
                'monitoring.enable_alerting': True,
                'log_level': log_level,
            }
        ]
    )
    
    # Diagnostics Aggregator
    diagnostics_aggregator_node = Node(
        condition=IfCondition(enable_monitoring),
        package='diagnostic_aggregator',
        executable='aggregator_node',
        name='diagnostics_aggregator',
        namespace='motion_planning',
        output='screen',
        parameters=[
            {
                'analyzers.motion_planning.type': 'diagnostic_aggregator/GenericAnalyzer',
                'analyzers.motion_planning.path': 'Motion Planning',
                'analyzers.motion_planning.contains': [
                    'inverse_kinematics_solver',
                    'collision_checker',
                    'path_optimizer',
                    'trajectory_planner',
                ],
                'log_level': log_level,
            }
        ]
    )
    
    # ============================================
    # COMPOSED CONTAINER
    # ============================================
    
    # Create container for composed nodes
    motion_planning_container = ComposableNodeContainer(
        condition=IfCondition(use_composition),
        name=container_name,
        namespace='motion_planning',
        package='rclcpp_components',
        executable='component_container_mt',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'use_sim_time': False,
        }],
        composable_node_descriptions=[
            ik_composable_node,
            collision_checker_composable_node,
            path_optimizer_composable_node,
            trajectory_planner_composable_node,
            motion_planning_manager_composable_node,
        ]
    )
    
    # Load composable nodes into existing container
    load_composable_nodes = LoadComposableNodes(
        condition=IfCondition(use_composition),
        target_container=container_name,
        composable_node_descriptions=[
            ik_composable_node,
            collision_checker_composable_node,
            path_optimizer_composable_node,
            trajectory_planner_composable_node,
            motion_planning_manager_composable_node,
        ]
    )
    
    # ============================================
    # SAFETY AND EMERGENCY SYSTEMS
    # ============================================
    
    # Emergency Stop Manager
    emergency_stop_manager_node = Node(
        condition=IfCondition(safety_enabled),
        package='safety_system',
        executable='emergency_stop_manager',
        name='emergency_stop_manager',
        namespace='safety',
        output='screen',
        parameters=[
            {
                'emergency_stop.timeout': emergency_stop_timeout,
                'emergency_stop.enable_auto_recovery': True,
                'emergency_stop.recovery_timeout': 2.0,
                'emergency_stop.require_manual_reset': False,
                'log_level': log_level,
            }
        ],
        remappings=[
            ('/emergency_stop', '/safety/emergency_stop'),
            ('/robot_status', '/robot/status'),
        ]
    )
    
    # Safety Monitor
    safety_monitor_node = Node(
        condition=IfCondition(safety_enabled),
        package='safety_system',
        executable='safety_monitor',
        name='safety_monitor',
        namespace='safety',
        output='screen',
        parameters=[
            {
                'monitoring.joint_limits.enabled': True,
                'monitoring.collision.enabled': True,
                'monitoring.velocity.enabled': True,
                'monitoring.torque.enabled': False,
                'monitoring.temperature.enabled': True,
                'monitoring.update_rate': 100.0,
                'log_level': log_level,
            }
        ],
        remappings=[
            ('/joint_states', '/robot/joint_states'),
            ('/collision_status', '/motion_planning/collision_checker/status'),
        ]
    )
    
    # ============================================
    # UTILITY NODES
    # ============================================
    
    # Parameter Blackboard (for shared parameters)
    parameter_blackboard_node = Node(
        package='parameter_blackboard',
        executable='parameter_blackboard_node',
        name='parameter_blackboard',
        namespace='motion_planning',
        output='screen',
        parameters=[
            {
                'shared_parameters': [
                    'robot_model',
                    'environment',
                    'safety_enabled',
                    'log_level',
                    'debug',
                ],
                'update_rate': 10.0,
                'log_level': log_level,
            }
        ]
    )
    
    # ROS Bag Recorder (for debugging)
    rosbag_recorder_node = Node(
        condition=IfCondition(debug),
        package='rosbag2',
        executable='record',
        name='rosbag_recorder',
        namespace='motion_planning',
        output='screen',
        arguments=[
            '-a',
            '-o', PathJoinSubstitution([
                EnvironmentVariable('HOME'),
                '.ros/motion_planning_bags'
            ]),
            '--qos-profile-overrides-path', PathJoinSubstitution([
                motion_planning_pkg,
                'config',
                'rosbag_qos_overrides.yaml'
            ])
        ]
    )
    
    # ============================================
    # STARTUP AND SHUTDOWN SCRIPTS
    # ============================================
    
    # Startup script to verify system readiness
    startup_verification_script = ExecuteProcess(
        cmd=[
            'python3',
            PathJoinSubstitution([
                motion_planning_pkg,
                'scripts',
                'verify_system_readiness.py'
            ]),
            '--environment', environment,
            '--robot-model', robot_model,
            '--timeout', '30'
        ],
        name='startup_verification',
        output='screen',
        shell=True
    )
    
    # Shutdown cleanup script
    shutdown_cleanup_script = ExecuteProcess(
        cmd=[
            'python3',
            PathJoinSubstitution([
                motion_planning_pkg,
                'scripts',
                'cleanup_resources.py'
            ])
        ],
        name='shutdown_cleanup',
        output='screen',
        shell=True
    )
    
    # ============================================
    # EVENT HANDLERS
    # ============================================
    
    # Handler for IK solver startup
    ik_startup_handler = RegisterEventHandler(
        OnProcessStart(
            target_action=ik_node if not use_composition else motion_planning_container,
            on_start=[
                LogInfo(msg='Inverse Kinematics Solver started successfully'),
                TimerAction(
                    period=2.0,
                    actions=[
                        LogInfo(msg='IK Solver initialization complete')
                    ]
                )
            ]
        )
    )
    
    # Handler for collision checker startup
    collision_checker_startup_handler = RegisterEventHandler(
        OnProcessStart(
            target_action=collision_checker_node if not use_composition else motion_planning_container,
            on_start=[
                LogInfo(msg='Collision Checker started successfully'),
                TimerAction(
                    period=2.0,
                    actions=[
                        LogInfo(msg='Collision Checker initialization complete')
                    ]
                )
            ]
        )
    )
    
    # Handler for trajectory planner startup
    trajectory_planner_startup_handler = RegisterEventHandler(
        OnProcessStart(
            target_action=trajectory_planner_node if not use_composition else motion_planning_container,
            on_start=[
                LogInfo(msg='Trajectory Planner started successfully'),
                TimerAction(
                    period=2.0,
                    actions=[
                        LogInfo(msg='Trajectory Planner initialization complete')
                    ]
                )
            ]
        )
    )
    
    # Handler for emergency stop system startup
    safety_startup_handler = RegisterEventHandler(
        OnProcessStart(
            target_action=emergency_stop_manager_node,
            on_start=[
                LogInfo(msg='Safety systems started successfully'),
                TimerAction(
                    period=1.0,
                    actions=[
                        LogInfo(msg='Safety systems initialization complete')
                    ]
                )
            ]
        )
    )
    
    # Handler for system shutdown
    shutdown_handler = RegisterEventHandler(
        OnShutdown(
            on_shutdown=[
                LogInfo(msg='Motion Planning System shutting down'),
                shutdown_cleanup_script,
                LogInfo(msg='Cleanup complete')
            ]
        )
    )
    
    # ============================================
    # LAUNCH DESCRIPTION ASSEMBLY
    # ============================================
    
    ld = LaunchDescription()
    
    # Add launch arguments
    ld.add_action(declare_config_package_arg)
    ld.add_action(declare_config_path_arg)
    ld.add_action(declare_environment_arg)
    ld.add_action(declare_robot_model_arg)
    ld.add_action(declare_enable_ik_arg)
    ld.add_action(declare_enable_collision_checking_arg)
    ld.add_action(declare_enable_path_optimization_arg)
    ld.add_action(declare_enable_trajectory_planning_arg)
    ld.add_action(declare_enable_visualization_arg)
    ld.add_action(declare_enable_monitoring_arg)
    ld.add_action(declare_use_composition_arg)
    ld.add_action(declare_container_name_arg)
    ld.add_action(declare_num_threads_arg)
    ld.add_action(declare_safety_enabled_arg)
    ld.add_action(declare_emergency_stop_timeout_arg)
    ld.add_action(declare_debug_arg)
    ld.add_action(declare_log_level_arg)
    
    # Set environment variables
    ld.add_action(SetEnvironmentVariable(
        name='RCUTILS_COLORIZED_OUTPUT',
        value='1'
    ))
    ld.add_action(SetEnvironmentVariable(
        name='RCUTILS_CONSOLE_OUTPUT_FORMAT',
        value='[{severity}] [{time}] [{name}]: {message}'
    ))
    
    # Add startup verification
    ld.add_action(startup_verification_script)
    
    # Add composed container or individual nodes
    if use_composition:
        ld.add_action(motion_planning_container)
        ld.add_action(load_composable_nodes)
    else:
        # Add individual nodes based on conditions
        if enable_ik:
            ld.add_action(ik_node)
        if enable_collision_checking:
            ld.add_action(collision_checker_node)
        if enable_path_optimization:
            ld.add_action(path_optimizer_node)
        if enable_trajectory_planning:
            ld.add_action(trajectory_planner_node)
        ld.add_action(motion_planning_manager_node)
    
    # Add visualization nodes
    if enable_visualization:
        ld.add_action(robot_state_publisher_node)
        ld.add_action(rviz_node)
        if debug:
            ld.add_action(joint_state_publisher_gui_node)
    
    # Add monitoring nodes
    if enable_monitoring:
        ld.add_action(system_monitor_node)
        ld.add_action(diagnostics_aggregator_node)
    
    # Add safety nodes
    if safety_enabled:
        ld.add_action(emergency_stop_manager_node)
        ld.add_action(safety_monitor_node)
    
    # Add utility nodes
    ld.add_action(parameter_blackboard_node)
    if debug:
        ld.add_action(rosbag_recorder_node)
    
    # Add event handlers
    ld.add_action(ik_startup_handler)
    ld.add_action(collision_checker_startup_handler)
    ld.add_action(trajectory_planner_startup_handler)
    if safety_enabled:
        ld.add_action(safety_startup_handler)
    ld.add_action(shutdown_handler)
    
    # Add final startup message
    ld.add_action(LogInfo(
        msg=f'''
        ============================================
        Motion Planning System Launch Configuration
        ============================================
        Environment: {environment}
        Robot Model: {robot_model}
        Safety Enabled: {safety_enabled}
        Use Composition: {use_composition}
        Threads: {num_threads}
        Log Level: {log_level}
        Debug Mode: {debug}
        
        Components:
        - Inverse Kinematics: {enable_ik}
        - Collision Checking: {enable_collision_checking}
        - Path Optimization: {enable_path_optimization}
        - Trajectory Planning: {enable_trajectory_planning}
        - Visualization: {enable_visualization}
        - Monitoring: {enable_monitoring}
        
        Starting system...
        ============================================
        '''
    ))
    
    return ld


# Helper function to load YAML configuration
def load_yaml_config(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)
    
    try:
        with open(absolute_file_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Warning: Configuration file {absolute_file_path} not found")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {absolute_file_path}: {e}")
        return {}


if __name__ == '__main__':
    generate_launch_description()
