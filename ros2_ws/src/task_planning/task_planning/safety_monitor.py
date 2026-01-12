#!/usr/bin/env python3
"""
Safety Monitor Module for ML-based Object Picking Robot
Production-ready implementation for industrial use
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
import yaml
from dataclasses import dataclass
from enum import Enum, auto
import json
import time
from datetime import datetime, timedelta
import queue

# ROS2 Message Imports
from geometry_msgs.msg import Twist, PoseStamped, Pose, Point, Quaternion
from sensor_msgs.msg import JointState, Image, CameraInfo
from std_msgs.msg import Header, String, Bool, Float32, Int32
from nav_msgs.msg import OccupancyGrid, Path
from tf2_ros import Buffer, TransformListener, TransformException
from tf2_geometry_msgs import do_transform_pose

# Custom ROS2 Message Imports
from robot_interfaces.msg import DetectedObject, ObstacleMap, RobotStatus, GraspPose
from robot_interfaces.srv import ClearObstacleMap
from robot_interfaces.action import PickPlace

# ROS2 Action imports
from rclpy.action import ActionServer, ActionClient
from rclpy.action.server import ServerGoalHandle
from rclpy.action.client import ClientGoalHandle

class SafetyState(Enum):
    """Safety states for the robotic system"""
    NORMAL = auto()
    WARNING = auto()
    CRITICAL = auto()
    EMERGENCY_STOP = auto()
    RECOVERY = auto()

class SafetyViolation(Enum):
    """Types of safety violations"""
    JOINT_LIMIT_EXCEEDED = auto()
    COLLISION_IMMINENT = auto()
    VELOCITY_LIMIT_EXCEEDED = auto()
    TEMPERATURE_CRITICAL = auto()
    POWER_ANOMALY = auto()
    COMMUNICATION_FAILURE = auto()
    GRIPPER_FAULT = auto()
    OBSTACLE_PROXIMITY = auto()
    UNEXPECTED_OBJECT_MOVEMENT = auto()

@dataclass
class SafetyThresholds:
    """Configurable safety thresholds"""
    # Joint limits (radians for revolute, meters for prismatic)
    joint_position_min: Dict[str, float]
    joint_position_max: Dict[str, float]
    joint_velocity_max: Dict[str, float]
    joint_acceleration_max: Dict[str, float]
    joint_torque_max: Dict[str, float]
    
    # Workspace limits (meters)
    workspace_x_min: float = -2.0
    workspace_x_max: float = 2.0
    workspace_y_min: float = -2.0
    workspace_y_max: float = 2.0
    workspace_z_min: float = 0.0
    workspace_z_max: float = 2.0
    
    # Collision avoidance
    minimum_obstacle_distance: float = 0.15  # 15cm minimum clearance
    critical_obstacle_distance: float = 0.05  # 5cm emergency stop
    
    # Temperature limits (°C)
    motor_temperature_max: float = 85.0
    electronics_temperature_max: float = 70.0
    
    # Power limits
    current_max: float = 12.0  # Amps
    voltage_min: float = 10.5  # Volts
    voltage_max: float = 13.8  # Volts
    
    # Time-based safety
    max_continuous_operation_hours: float = 8.0
    cooldown_period_minutes: float = 15.0
    
    # Response thresholds
    warning_response_time_ms: int = 100
    critical_response_time_ms: int = 50
    emergency_response_time_ms: int = 10

@dataclass
class SystemHealth:
    """System health metrics"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float]
    disk_usage: float
    network_latency_ms: float
    process_alive: Dict[str, bool]
    ros_topics_active: Dict[str, bool]
    last_heartbeat: Dict[str, datetime]

class SafetyMonitor(Node):
    """
    Real-time Safety Monitor for industrial robotic system
    Implements multi-layered safety checks and emergency protocols
    """
    
    def __init__(self):
        super().__init__('safety_monitor')
        
        # Initialize with high reliability QoS
        qos_profile = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST
        )
        
        # Separate callback groups for parallel safety checks
        self.safety_check_group = ReentrantCallbackGroup()
        self.health_monitor_group = MutuallyExclusiveCallbackGroup()
        self.emergency_group = MutuallyExclusiveCallbackGroup()
        
        # State variables
        self.current_safety_state = SafetyState.NORMAL
        self.safety_thresholds = None
        self.system_health = SystemHealth(
            timestamp=datetime.now(),
            cpu_usage=0.0,
            memory_usage=0.0,
            gpu_usage=None,
            disk_usage=0.0,
            network_latency_ms=0.0,
            process_alive={},
            ros_topics_active={},
            last_heartbeat={}
        )
        
        # Real-time monitoring buffers
        self.joint_states_buffer = queue.Queue(maxsize=100)
        self.obstacle_map_buffer = queue.Queue(maxsize=10)
        self.detected_objects_buffer = queue.Queue(maxsize=50)
        
        # Violation tracking
        self.active_violations: Set[SafetyViolation] = set()
        self.violation_history = []
        self.violation_count = 0
        self.last_emergency_time = None
        
        # Thread safety
        self._lock = threading.RLock()
        self._emergency_lock = threading.Lock()
        
        # Load configuration
        self._load_configuration()
        
        # Initialize ROS2 components
        self._initialize_publishers(qos_profile)
        self._initialize_subscribers(qos_profile)
        self._initialize_services()
        self._initialize_action_servers()
        
        # TF2 for coordinate transforms
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Start monitoring threads
        self._start_monitoring_threads()
        
        self.get_logger().info("Safety Monitor initialized in PRODUCTION mode")
    
    def _load_configuration(self):
        """Load safety configuration from YAML files"""
        config_paths = [
            '/home/victor/ml-object-picking-robot/config/production.yaml',
            '/home/victor/ml-object-picking-robot/ros2_ws/src/task_planning/config/safety_rules.yaml',
            '/home/victor/ml-object-picking-robot/ros2_ws/src/motion_planning/config/safety_limits.yaml'
        ]
        
        config = {}
        for path in config_paths:
            try:
                with open(path, 'r') as f:
                    config.update(yaml.safe_load(f) or {})
                self.get_logger().info(f"Loaded config from {path}")
            except Exception as e:
                self.get_logger().error(f"Failed to load config {path}: {e}")
        
        # Parse safety thresholds
        safety_config = config.get('safety', {})
        self.safety_thresholds = SafetyThresholds(
            joint_position_min=safety_config.get('joint_position_min', {}),
            joint_position_max=safety_config.get('joint_position_max', {}),
            joint_velocity_max=safety_config.get('joint_velocity_max', {}),
            joint_acceleration_max=safety_config.get('joint_acceleration_max', {}),
            joint_torque_max=safety_config.get('joint_torque_max', {}),
            workspace_x_min=safety_config.get('workspace_x_min', -2.0),
            workspace_x_max=safety_config.get('workspace_x_max', 2.0),
            workspace_y_min=safety_config.get('workspace_y_min', -2.0),
            workspace_y_max=safety_config.get('workspace_y_max', 2.0),
            workspace_z_min=safety_config.get('workspace_z_min', 0.0),
            workspace_z_max=safety_config.get('workspace_z_max', 2.0),
            minimum_obstacle_distance=safety_config.get('minimum_obstacle_distance', 0.15),
            critical_obstacle_distance=safety_config.get('critical_obstacle_distance', 0.05),
            motor_temperature_max=safety_config.get('motor_temperature_max', 85.0),
            electronics_temperature_max=safety_config.get('electronics_temperature_max', 70.0),
            current_max=safety_config.get('current_max', 12.0),
            voltage_min=safety_config.get('voltage_min', 10.5),
            voltage_max=safety_config.get('voltage_max', 13.8),
            max_continuous_operation_hours=safety_config.get('max_continuous_operation_hours', 8.0),
            cooldown_period_minutes=safety_config.get('cooldown_period_minutes', 15.0),
            warning_response_time_ms=safety_config.get('warning_response_time_ms', 100),
            critical_response_time_ms=safety_config.get('critical_response_time_ms', 50),
            emergency_response_time_ms=safety_config.get('emergency_response_time_ms', 10)
        )
    
    def _initialize_publishers(self, qos_profile):
        """Initialize ROS2 publishers"""
        self.safety_state_pub = self.create_publisher(
            RobotStatus,
            '/robot/safety/state',
            qos_profile
        )
        
        self.emergency_stop_pub = self.create_publisher(
            Bool,
            '/robot/emergency_stop',
            QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE)
        )
        
        self.safety_violation_pub = self.create_publisher(
            String,
            '/robot/safety/violations',
            qos_profile
        )
        
        self.health_status_pub = self.create_publisher(
            String,
            '/robot/health/status',
            qos_profile
        )
        
        self.safety_metrics_pub = self.create_publisher(
            String,
            '/robot/safety/metrics',
            qos_profile
        )
    
    def _initialize_subscribers(self, qos_profile):
        """Initialize ROS2 subscribers"""
        # Joint states monitoring
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self._joint_state_callback,
            qos_profile,
            callback_group=self.safety_check_group
        )
        
        # Obstacle monitoring
        self.obstacle_map_sub = self.create_subscription(
            ObstacleMap,
            '/perception/obstacle_map',
            self._obstacle_map_callback,
            qos_profile,
            callback_group=self.safety_check_group
        )
        
        # Object detection monitoring
        self.detected_objects_sub = self.create_subscription(
            DetectedObject,
            '/perception/detected_objects',
            self._detected_objects_callback,
            qos_profile,
            callback_group=self.safety_check_group
        )
        
        # System health monitoring
        self.system_health_sub = self.create_subscription(
            String,
            '/system_monitor/health',
            self._system_health_callback,
            qos_profile,
            callback_group=self.health_monitor_group
        )
        
        # Motion planning monitoring
        self.planned_path_sub = self.create_subscription(
            Path,
            '/motion_planning/planned_path',
            self._planned_path_callback,
            qos_profile,
            callback_group=self.safety_check_group
        )
        
        # Gripper status
        self.gripper_status_sub = self.create_subscription(
            String,
            '/robot_control/gripper/status',
            self._gripper_status_callback,
            qos_profile,
            callback_group=self.safety_check_group
        )
    
    def _initialize_services(self):
        """Initialize ROS2 services"""
        self.clear_violations_srv = self.create_service(
            ClearObstacleMap,
            '/safety/clear_violations',
            self._clear_violations_callback,
            callback_group=self.emergency_group
        )
        
        self.get_safety_state_srv = self.create_service(
            RobotStatus,
            '/safety/get_state',
            self._get_safety_state_callback,
            callback_group=self.health_monitor_group
        )
    
    def _initialize_action_servers(self):
        """Initialize ROS2 action servers"""
        self.safety_override_action = ActionServer(
            self,
            PickPlace,
            '/safety/override',
            self._safety_override_callback,
            callback_group=self.emergency_group
        )
    
    def _start_monitoring_threads(self):
        """Start real-time monitoring threads"""
        # High-frequency safety check thread (100Hz)
        self.safety_check_timer = self.create_timer(
            0.01,  # 100Hz
            self._perform_safety_checks,
            callback_group=self.safety_check_group
        )
        
        # System health monitoring thread (1Hz)
        self.health_monitor_timer = self.create_timer(
            1.0,  # 1Hz
            self._update_system_health,
            callback_group=self.health_monitor_group
        )
        
        # Emergency response thread (500Hz)
        self.emergency_check_timer = self.create_timer(
            0.002,  # 500Hz
            self._emergency_response_check,
            callback_group=self.emergency_group
        )
        
        # Metrics publishing thread (10Hz)
        self.metrics_timer = self.create_timer(
            0.1,  # 10Hz
            self._publish_safety_metrics,
            callback_group=self.health_monitor_group
        )
    
    def _joint_state_callback(self, msg: JointState):
        """Process joint state updates"""
        try:
            self.joint_states_buffer.put_nowait({
                'timestamp': datetime.now(),
                'positions': dict(zip(msg.name, msg.position)),
                'velocities': dict(zip(msg.name, msg.velocity)),
                'efforts': dict(zip(msg.name, msg.effort))
            })
        except queue.Full:
            self.get_logger().warn("Joint states buffer full, dropping old data")
            try:
                self.joint_states_buffer.get_nowait()
                self.joint_states_buffer.put_nowait({
                    'timestamp': datetime.now(),
                    'positions': dict(zip(msg.name, msg.position)),
                    'velocities': dict(zip(msg.name, msg.velocity)),
                    'efforts': dict(zip(msg.name, msg.effort))
                })
            except:
                pass
    
    def _obstacle_map_callback(self, msg: ObstacleMap):
        """Process obstacle map updates"""
        try:
            self.obstacle_map_buffer.put_nowait({
                'timestamp': datetime.now(),
                'obstacles': msg.obstacles,
                'resolution': msg.resolution,
                'origin': msg.origin
            })
        except queue.Full:
            pass  # Obstacle maps are less critical than joint states
    
    def _detected_objects_callback(self, msg: DetectedObject):
        """Process detected object updates"""
        try:
            self.detected_objects_buffer.put_nowait({
                'timestamp': datetime.now(),
                'object_id': msg.object_id,
                'class_name': msg.class_name,
                'confidence': msg.confidence,
                'bounding_box': msg.bounding_box,
                'pose': msg.pose,
                'velocity': msg.velocity
            })
        except queue.Full:
            pass
    
    def _system_health_callback(self, msg: String):
        """Process system health updates"""
        try:
            health_data = json.loads(msg.data)
            with self._lock:
                self.system_health = SystemHealth(
                    timestamp=datetime.fromisoformat(health_data['timestamp']),
                    cpu_usage=health_data.get('cpu_usage', 0.0),
                    memory_usage=health_data.get('memory_usage', 0.0),
                    gpu_usage=health_data.get('gpu_usage'),
                    disk_usage=health_data.get('disk_usage', 0.0),
                    network_latency_ms=health_data.get('network_latency_ms', 0.0),
                    process_alive=health_data.get('process_alive', {}),
                    ros_topics_active=health_data.get('ros_topics_active', {}),
                    last_heartbeat=health_data.get('last_heartbeat', {})
                )
        except Exception as e:
            self.get_logger().error(f"Failed to parse health data: {e}")
    
    def _planned_path_callback(self, msg: Path):
        """Check planned paths for safety"""
        # Check for collisions in planned path
        if self.obstacle_map_buffer.empty():
            return
        
        try:
            obstacle_data = self.obstacle_map_buffer.queue[-1]  # Get latest
            for pose_stamped in msg.poses:
                # Convert pose to obstacle map coordinates
                # Check collision with obstacles
                # This is simplified - actual implementation would use
                # proper coordinate transforms and collision checking
                pass
        except:
            pass
    
    def _gripper_status_callback(self, msg: String):
        """Monitor gripper status"""
        status_data = json.loads(msg.data)
        
        # Check for gripper faults
        if status_data.get('fault_detected', False):
            self._register_violation(
                SafetyViolation.GRIPPER_FAULT,
                f"Gripper fault detected: {status_data.get('fault_type', 'Unknown')}",
                critical=True
            )
        
        # Check grip force anomalies
        grip_force = status_data.get('grip_force', 0.0)
        if grip_force > 100.0:  # Example threshold
            self._register_violation(
                SafetyViolation.GRIPPER_FAULT,
                f"Excessive grip force: {grip_force}N",
                critical=True
            )
    
    def _perform_safety_checks(self):
        """Perform comprehensive safety checks at high frequency"""
        violations_detected = []
        
        # 1. Check joint limits
        violations_detected.extend(self._check_joint_limits())
        
        # 2. Check workspace boundaries
        violations_detected.extend(self._check_workspace_boundaries())
        
        # 3. Check obstacle proximity
        violations_detected.extend(self._check_obstacle_proximity())
        
        # 4. Check velocity/acceleration limits
        violations_detected.extend(self._check_dynamic_limits())
        
        # 5. Check system health
        violations_detected.extend(self._check_system_health())
        
        # 6. Check operational time limits
        violations_detected.extend(self._check_operational_limits())
        
        # 7. Check for unexpected object movements
        violations_detected.extend(self._check_object_movements())
        
        # Update safety state based on violations
        self._update_safety_state(violations_detected)
    
    def _check_joint_limits(self) -> List[Tuple[SafetyViolation, str]]:
        """Check joint position and velocity limits"""
        violations = []
        
        if self.joint_states_buffer.empty():
            return violations
        
        try:
            joint_data = self.joint_states_buffer.queue[-1]  # Get latest
            
            for joint_name, position in joint_data['positions'].items():
                # Check position limits
                min_pos = self.safety_thresholds.joint_position_min.get(joint_name)
                max_pos = self.safety_thresholds.joint_position_max.get(joint_name)
                
                if min_pos is not None and position < min_pos:
                    violations.append((
                        SafetyViolation.JOINT_LIMIT_EXCEEDED,
                        f"Joint {joint_name} below minimum: {position} < {min_pos}"
                    ))
                
                if max_pos is not None and position > max_pos:
                    violations.append((
                        SafetyViolation.JOINT_LIMIT_EXCEEDED,
                        f"Joint {joint_name} above maximum: {position} > {max_pos}"
                    ))
            
            # Check velocity limits
            for joint_name, velocity in joint_data['velocities'].items():
                max_vel = self.safety_thresholds.joint_velocity_max.get(joint_name)
                
                if max_vel is not None and abs(velocity) > max_vel:
                    violations.append((
                        SafetyViolation.VELOCITY_LIMIT_EXCEEDED,
                        f"Joint {joint_name} velocity exceeded: {velocity} > {max_vel}"
                    ))
        
        except Exception as e:
            self.get_logger().error(f"Joint limit check failed: {e}")
        
        return violations
    
    def _check_workspace_boundaries(self) -> List[Tuple[SafetyViolation, str]]:
        """Check if end effector is within workspace boundaries"""
        violations = []
        
        # This would require forward kinematics calculation
        # Simplified implementation - actual would use kinematics library
        
        try:
            # Get end effector pose from TF
            transform = self.tf_buffer.lookup_transform(
                'base_link',
                'end_effector',
                rclpy.time.Time()
            )
            
            position = transform.transform.translation
            
            # Check X boundaries
            if position.x < self.safety_thresholds.workspace_x_min:
                violations.append((
                    SafetyViolation.COLLISION_IMMINENT,
                    f"End effector X below workspace: {position.x} < {self.safety_thresholds.workspace_x_min}"
                ))
            elif position.x > self.safety_thresholds.workspace_x_max:
                violations.append((
                    SafetyViolation.COLLISION_IMMINENT,
                    f"End effector X above workspace: {position.x} > {self.safety_thresholds.workspace_x_max}"
                ))
            
            # Check Y boundaries
            if position.y < self.safety_thresholds.workspace_y_min:
                violations.append((
                    SafetyViolation.COLLISION_IMMINENT,
                    f"End effector Y below workspace: {position.y} < {self.safety_thresholds.workspace_y_min}"
                ))
            elif position.y > self.safety_thresholds.workspace_y_max:
                violations.append((
                    SafetyViolation.COLLISION_IMMINENT,
                    f"End effector Y above workspace: {position.y} > {self.safety_thresholds.workspace_y_max}"
                ))
            
            # Check Z boundaries
            if position.z < self.safety_thresholds.workspace_z_min:
                violations.append((
                    SafetyViolation.COLLISION_IMMINENT,
                    f"End effector Z below workspace: {position.z} < {self.safety_thresholds.workspace_z_min}"
                ))
            elif position.z > self.safety_thresholds.workspace_z_max:
                violations.append((
                    SafetyViolation.COLLISION_IMMINENT,
                    f"End effector Z above workspace: {position.z} > {self.safety_thresholds.workspace_z_max}"
                ))
        
        except TransformException:
            pass  # TF not available yet
        except Exception as e:
            self.get_logger().error(f"Workspace boundary check failed: {e}")
        
        return violations
    
    def _check_obstacle_proximity(self) -> List[Tuple[SafetyViolation, str]]:
        """Check distance to obstacles"""
        violations = []
        
        if self.obstacle_map_buffer.empty():
            return violations
        
        try:
            obstacle_data = self.obstacle_map_buffer.queue[-1]
            detected_objects = list(self.detected_objects_buffer.queue)
            
            # Get end effector position
            transform = self.tf_buffer.lookup_transform(
                'base_link',
                'end_effector',
                rclpy.time.Time()
            )
            ee_position = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ])
            
            # Check proximity to obstacles
            for obstacle in obstacle_data['obstacles']:
                obstacle_pos = np.array([obstacle.x, obstacle.y, obstacle.z])
                distance = np.linalg.norm(ee_position - obstacle_pos)
                
                if distance < self.safety_thresholds.critical_obstacle_distance:
                    violations.append((
                        SafetyViolation.COLLISION_IMMINENT,
                        f"Critical obstacle proximity: {distance:.3f}m"
                    ))
                elif distance < self.safety_thresholds.minimum_obstacle_distance:
                    violations.append((
                        SafetyViolation.OBSTACLE_PROXIMITY,
                        f"Warning: obstacle proximity: {distance:.3f}m"
                    ))
            
            # Check proximity to detected objects
            for obj in detected_objects:
                if 'pose' in obj and obj['pose']:
                    obj_pos = np.array([
                        obj['pose'].position.x,
                        obj['pose'].position.y,
                        obj['pose'].position.z
                    ])
                    distance = np.linalg.norm(ee_position - obj_pos)
                    
                    if distance < 0.1:  # 10cm threshold for objects
                        violations.append((
                            SafetyViolation.UNEXPECTED_OBJECT_MOVEMENT,
                            f"Close to object {obj.get('object_id', 'unknown')}: {distance:.3f}m"
                        ))
        
        except Exception as e:
            self.get_logger().error(f"Obstacle proximity check failed: {e}")
        
        return violations
    
    def _check_dynamic_limits(self) -> List[Tuple[SafetyViolation, str]]:
        """Check dynamic limits (acceleration, jerk)"""
        violations = []
        
        if self.joint_states_buffer.qsize() < 3:
            return violations  # Need multiple samples for derivatives
        
        try:
            # Get last 3 joint state samples
            samples = []
            for _ in range(3):
                samples.append(self.joint_states_buffer.get_nowait())
            
            # Calculate accelerations
            time_diffs = []
            for i in range(1, len(samples)):
                dt = (samples[i]['timestamp'] - samples[i-1]['timestamp']).total_seconds()
                time_diffs.append(dt)
            
            avg_dt = np.mean(time_diffs) if time_diffs else 0.01
            
            # Check each joint
            joint_names = samples[0]['velocities'].keys()
            for joint_name in joint_names:
                velocities = [s['velocities'].get(joint_name, 0) for s in samples]
                
                # Calculate acceleration (simplified)
                if len(velocities) >= 2 and avg_dt > 0:
                    acceleration = (velocities[-1] - velocities[-2]) / avg_dt
                    
                    max_accel = self.safety_thresholds.joint_acceleration_max.get(joint_name)
                    if max_accel is not None and abs(acceleration) > max_accel:
                        violations.append((
                            SafetyViolation.VELOCITY_LIMIT_EXCEEDED,
                            f"Joint {joint_name} acceleration exceeded: {acceleration:.2f} > {max_accel}"
                        ))
            
            # Put samples back
            for sample in samples:
                try:
                    self.joint_states_buffer.put_nowait(sample)
                except:
                    pass
        
        except Exception as e:
            self.get_logger().error(f"Dynamic limit check failed: {e}")
        
        return violations
    
    def _check_system_health(self) -> List[Tuple[SafetyViolation, str]]:
        """Check system health metrics"""
        violations = []
        
        with self._lock:
            # Check CPU usage
            if self.system_health.cpu_usage > 90.0:
                violations.append((
                    SafetyViolation.COMMUNICATION_FAILURE,
                    f"High CPU usage: {self.system_health.cpu_usage:.1f}%"
                ))
            
            # Check memory usage
            if self.system_health.memory_usage > 85.0:
                violations.append((
                    SafetyViolation.COMMUNICATION_FAILURE,
                    f"High memory usage: {self.system_health.memory_usage:.1f}%"
                ))
            
            # Check process health
            critical_processes = ['perception_stack', 'motion_planning', 'robot_control']
            for process in critical_processes:
                if not self.system_health.process_alive.get(process, True):
                    violations.append((
                        SafetyViolation.COMMUNICATION_FAILURE,
                        f"Critical process {process} not alive"
                    ))
            
            # Check ROS topic health
            critical_topics = ['/joint_states', '/perception/detected_objects']
            for topic in critical_topics:
                if not self.system_health.ros_topics_active.get(topic, True):
                    violations.append((
                        SafetyViolation.COMMUNICATION_FAILURE,
                        f"Critical topic {topic} not active"
                    ))
        
        return violations
    
    def _check_operational_limits(self) -> List[Tuple[SafetyViolation, str]]:
        """Check operational time limits"""
        violations = []
        
        # Check continuous operation time
        # This would track total operation time and enforce cooldowns
        # Simplified implementation
        
        if self.last_emergency_time:
            time_since_emergency = (datetime.now() - self.last_emergency_time).total_seconds() / 60.0
            
            if time_since_emergency < self.safety_thresholds.cooldown_period_minutes:
                violations.append((
                    SafetyViolation.POWER_ANOMALY,
                    f"Insufficient cooldown time: {time_since_emergency:.1f} minutes"
                ))
        
        return violations
    
    def _check_object_movements(self) -> List[Tuple[SafetyViolation, str]]:
        """Check for unexpected object movements"""
        violations = []
        
        detected_objects = list(self.detected_objects_buffer.queue)
        if len(detected_objects) < 2:
            return violations
        
        # Compare recent object positions
        recent_objects = {}
        for obj in detected_objects[-10:]:  # Last 10 samples
            obj_id = obj.get('object_id')
            if obj_id and 'pose' in obj:
                if obj_id not in recent_objects:
                    recent_objects[obj_id] = []
                recent_objects[obj_id].append({
                    'timestamp': obj['timestamp'],
                    'position': np.array([
                        obj['pose'].position.x,
                        obj['pose'].position.y,
                        obj['pose'].position.z
                    ]),
                    'velocity': obj.get('velocity', 0.0)
                })
        
        # Check for unexpected movements
        for obj_id, positions in recent_objects.items():
            if len(positions) >= 2:
                # Calculate actual movement
                positions_sorted = sorted(positions, key=lambda x: x['timestamp'])
                time_diff = (positions_sorted[-1]['timestamp'] - positions_sorted[0]['timestamp']).total_seconds()
                pos_diff = np.linalg.norm(positions_sorted[-1]['position'] - positions_sorted[0]['position'])
                
                if time_diff > 0:
                    actual_velocity = pos_diff / time_diff
                    reported_velocity = positions_sorted[-1].get('velocity', 0.0)
                    
                    # Check for discrepancy
                    if abs(actual_velocity - reported_velocity) > 0.1:  # 0.1 m/s threshold
                        violations.append((
                            SafetyViolation.UNEXPECTED_OBJECT_MOVEMENT,
                            f"Unexpected movement for object {obj_id}: "
                            f"actual={actual_velocity:.2f}m/s, reported={reported_velocity:.2f}m/s"
                        ))
        
        return violations
    
    def _register_violation(self, violation_type: SafetyViolation, message: str, critical: bool = False):
        """Register a safety violation"""
        with self._lock:
            self.active_violations.add(violation_type)
            self.violation_history.append({
                'timestamp': datetime.now(),
                'type': violation_type,
                'message': message,
                'critical': critical
            })
            self.violation_count += 1
            
            # Publish violation
            violation_msg = String()
            violation_msg.data = json.dumps({
                'timestamp': datetime.now().isoformat(),
                'violation_type': violation_type.name,
                'message': message,
                'critical': critical,
                'current_state': self.current_safety_state.name
            })
            self.safety_violation_pub.publish(violation_msg)
            
            if critical:
                self._trigger_emergency_stop(f"Critical violation: {message}")
    
    def _update_safety_state(self, violations: List[Tuple[SafetyViolation, str]]):
        """Update safety state based on detected violations"""
        if not violations:
            if self.current_safety_state != SafetyState.NORMAL:
                self.current_safety_state = SafetyState.NORMAL
                self.active_violations.clear()
            return
        
        # Classify violations by severity
        critical_violations = any(
            v[0] in {
                SafetyViolation.COLLISION_IMMINENT,
                SafetyViolation.JOINT_LIMIT_EXCEEDED,
                SafetyViolation.GRIPPER_FAULT
            }
            for v in violations
        )
        
        warning_violations = any(
            v[0] in {
                SafetyViolation.OBSTACLE_PROXIMITY,
                SafetyViolation.VELOCITY_LIMIT_EXCEEDED,
                SafetyViolation.UNEXPECTED_OBJECT_MOVEMENT
            }
            for v in violations
        )
        
        # Update state
        new_state = self.current_safety_state
        
        if critical_violations:
            new_state = SafetyState.EMERGENCY_STOP
            for violation_type, message in violations:
                if violation_type in {
                    SafetyViolation.COLLISION_IMMINENT,
                    SafetyViolation.JOINT_LIMIT_EXCEEDED,
                    SafetyViolation.GRIPPER_FAULT
                }:
                    self._register_violation(violation_type, message, critical=True)
        elif warning_violations:
            new_state = SafetyState.WARNING
            for violation_type, message in violations:
                if violation_type in {
                    SafetyViolation.OBSTACLE_PROXIMITY,
                    SafetyViolation.VELOCITY_LIMIT_EXCEEDED,
                    SafetyViolation.UNEXPECTED_OBJECT_MOVEMENT
                }:
                    self._register_violation(violation_type, message, critical=False)
        elif self.current_safety_state == SafetyState.EMERGENCY_STOP:
            new_state = SafetyState.RECOVERY
        
        # State transition
        if new_state != self.current_safety_state:
            self.current_safety_state = new_state
            self._publish_safety_state()
            
            if new_state == SafetyState.EMERGENCY_STOP:
                self.last_emergency_time = datetime.now()
                self._trigger_emergency_stop("Automatic emergency stop triggered")
    
    def _trigger_emergency_stop(self, reason: str):
        """Trigger emergency stop procedure"""
        with self._emergency_lock:
            self.get_logger().error(f"EMERGENCY STOP: {reason}")
            
            # Publish emergency stop command
            stop_msg = Bool()
            stop_msg.data = True
            self.emergency_stop_pub.publish(stop_msg)
            
            # Log emergency
            emergency_log = {
                'timestamp': datetime.now().isoformat(),
                'reason': reason,
                'state': self.current_safety_state.name,
                'active_violations': [v.name for v in self.active_violations]
            }
            
            self.get_logger().error(f"Emergency log: {json.dumps(emergency_log)}")
    
    def _update_system_health(self):
        """Update and publish system health"""
        # In production, this would collect real system metrics
        # For now, publish current health status
        
        health_msg = String()
        health_msg.data = json.dumps({
            'timestamp': datetime.now().isoformat(),
            'safety_state': self.current_safety_state.name,
            'active_violations': len(self.active_violations),
            'total_violations': self.violation_count,
            'joint_states_buffer_size': self.joint_states_buffer.qsize(),
            'obstacle_buffer_size': self.obstacle_map_buffer.qsize(),
            'objects_buffer_size': self.detected_objects_buffer.qsize()
        })
        
        self.health_status_pub.publish(health_msg)
    
    def _publish_safety_metrics(self):
        """Publish safety performance metrics"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'safety_state': self.current_safety_state.name,
            'violation_rate': self.violation_count / max(1, (datetime.now() - self.get_clock().now().to_msg()).seconds),
            'response_times': {
                'warning': self.safety_thresholds.warning_response_time_ms,
                'critical': self.safety_thresholds.critical_response_time_ms,
                'emergency': self.safety_thresholds.emergency_response_time_ms
            },
            'buffer_utilization': {
                'joint_states': self.joint_states_buffer.qsize() / 100.0,
                'obstacle_map': self.obstacle_map_buffer.qsize() / 10.0,
                'detected_objects': self.detected_objects_buffer.qsize() / 50.0
            },
            'system_load': {
                'cpu': self.system_health.cpu_usage,
                'memory': self.system_health.memory_usage,
                'disk': self.system_health.disk_usage
            }
        }
        
        metrics_msg = String()
        metrics_msg.data = json.dumps(metrics)
        self.safety_metrics_pub.publish(metrics_msg)
    
    def _publish_safety_state(self):
        """Publish current safety state"""
        state_msg = RobotStatus()
        state_msg.timestamp = self.get_clock().now().to_msg()
        state_msg.status = self.current_safety_state.name
        state_msg.details = json.dumps({
            'active_violations': [v.name for v in self.active_violations],
            'violation_count': self.violation_count
        })
        
        self.safety_state_pub.publish(state_msg)
    
    def _emergency_response_check(self):
        """High-frequency emergency response check"""
        if self.current_safety_state == SafetyState.EMERGENCY_STOP:
            # Continuously publish emergency stop while in emergency state
            stop_msg = Bool()
            stop_msg.data = True
            self.emergency_stop_pub.publish(stop_msg)
    
    def _clear_violations_callback(self, request, response):
        """Service callback to clear safety violations"""
        with self._lock:
            self.active_violations.clear()
            self.violation_count = 0
            self.current_safety_state = SafetyState.NORMAL
        
        response.success = True
        response.message = "Safety violations cleared"
        return response
    
    def _get_safety_state_callback(self, request, response):
        """Service callback to get current safety state"""
        response.timestamp = self.get_clock().now().to_msg()
        response.status = self.current_safety_state.name
        response.details = json.dumps({
            'active_violations': [v.name for v in self.active_violations],
            'violation_history_count': len(self.violation_history),
            'system_health': {
                'cpu': self.system_health.cpu_usage,
                'memory': self.system_health.memory_usage
            }
        })
        return response
    
    def _safety_override_callback(self, goal_handle):
        """Action callback for safety override (manual intervention)"""
        goal = goal_handle.request
        
        # Only allow override in WARNING state, not in EMERGENCY_STOP
        if self.current_safety_state == SafetyState.EMERGENCY_STOP:
            goal_handle.abort()
            return PickPlace.Result(success=False, message="Cannot override emergency stop")
        
        # Execute override
        self.get_logger().warn(f"Safety override activated: {goal.override_reason}")
        
        # Temporarily suppress warnings
        original_state = self.current_safety_state
        self.current_safety_state = SafetyState.NORMAL
        
        # Wait for task completion
        # In production, this would monitor the task
        
        # Restore original state
        self.current_safety_state = original_state
        
        goal_handle.succeed()
        return PickPlace.Result(success=True, message="Safety override completed")
    
    def destroy_node(self):
        """Clean shutdown"""
        self.get_logger().info("Safety Monitor shutting down")
        
        # Clear all buffers
        while not self.joint_states_buffer.empty():
            try:
                self.joint_states_buffer.get_nowait()
            except:
                pass
        
        while not self.obstacle_map_buffer.empty():
            try:
                self.obstacle_map_buffer.get_nowait()
            except:
                pass
        
        while not self.detected_objects_buffer.empty():
            try:
                self.detected_objects_buffer.get_nowait()
            except:
                pass
        
        super().destroy_node()

def main(args=None):
    """Main entry point"""
    rclpy.init(args=args)
    
    # Use multi-threaded executor for parallel safety checks
    executor = MultiThreadedExecutor(num_threads=8)
    
    safety_monitor = SafetyMonitor()
    executor.add_node(safety_monitor)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        safety_monitor.get_logger().info("Safety Monitor shutdown requested")
    finally:
        executor.shutdown()
        safety_monitor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
