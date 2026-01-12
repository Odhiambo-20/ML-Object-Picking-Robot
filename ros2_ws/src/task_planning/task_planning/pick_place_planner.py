#!/usr/bin/env python3
"""
Production-grade Pick and Place Planner for ML-Based Object Picking Robot

Advanced task planning system with:
- Multi-objective optimization
- Collision-aware trajectory planning
- Adaptive grasping strategies
- Real-time replanning
- Failure recovery mechanisms
- Integration with perception and control systems

Author: Robotics Engineering Team
Version: 3.2.0
License: Proprietary
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ActionClient, CancelResponse
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import threading
import time
import json
import numpy as np
from dataclasses import dataclass, asdict, field
from enum import Enum, auto
from typing import Dict, List, Tuple, Optional, Any, Deque, Set
from collections import deque, defaultdict
import heapq
import math
import random
from datetime import datetime, timedelta
import hashlib
import copy
import pickle
from pathlib import Path
import warnings

# ROS2 Messages
from std_msgs.msg import Header, String, Bool, Float32, Int32
from geometry_msgs.msg import (
    Pose, PoseStamped, Point, Quaternion, Vector3,
    PoseArray, TransformStamped, Twist
)
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.msg import (
    PlanningScene, CollisionObject, AttachedCollisionObject,
    RobotState, Constraints, MotionPlanRequest, MotionPlanResponse,
    MoveItErrorCodes, RobotTrajectory, PositionIKRequest
)
from moveit_msgs.srv import GetPlanningScene, ApplyPlanningScene
from shape_msgs.msg import SolidPrimitive, Mesh, Plane

# ROS2 Services and Actions
from std_srvs.srv import Trigger, SetBool, Empty
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose

# Custom messages
from robot_interfaces.msg import (
    DetectedObject, ObstacleMap, GraspPose, RobotStatus,
    PickPlaceAction, PickPlaceGoal, PickPlaceResult, PickPlaceFeedback,
    NavigateAction, NavigateGoal, NavigateResult, NavigateFeedback
)
from robot_interfaces.srv import (
    PickObject, PlaceObject, ClearObstacleMap,
    GetGraspPoses, ValidateGrasp, ComputeIK, ComputeFK
)
from system_monitor_msgs.msg import SystemHealth, PerformanceMetrics, Alert

# MoveIt Python interface
try:
    from moveit import MoveItPy, PlanningSceneMonitor
    from moveit.core.robot_state import RobotState
    from moveit.core.planning_scene import PlanningScene
    from moveit.planning import MoveItPyPlanRequestParameters
    MOVEIT_AVAILABLE = True
except ImportError:
    MOVEIT_AVAILABLE = False
    warnings.warn("MoveIt not available. Planning functionality will be limited.")


class TaskState(Enum):
    """State machine states for pick and place tasks."""
    IDLE = auto()
    PLANNING = auto()
    EXECUTING = auto()
    PAUSED = auto()
    CANCELLED = auto()
    SUCCEEDED = auto()
    FAILED = auto()
    RECOVERING = auto()
    WAITING_FOR_PERCEPTION = auto()
    WAITING_FOR_CONTROL = auto()


class ObjectType(Enum):
    """Types of objects for grasping strategies."""
    UNKNOWN = auto()
    BOX = auto()
    CYLINDER = auto()
    SPHERE = auto()
    MESH = auto()
    FRAGILE = auto()
    DEFORMABLE = auto()
    HEAVY = auto()
    SMALL = auto()
    LARGE = auto()


class GraspType(Enum):
    """Types of grasps for different objects."""
    TOP_GRASP = auto()
    SIDE_GRASP = auto()
    PINCH_GRASP = auto()
    POWER_GRASP = auto()
    PRECISION_GRASP = auto()
    VACUUM_GRASP = auto()
    MAGNETIC_GRASP = auto()
    ADHESIVE_GRASP = auto()
    SPECIALIZED_GRASP = auto()


class PlanningAlgorithm(Enum):
    """Planning algorithms for motion planning."""
    RRT = auto()
    RRT_STAR = auto()
    RRT_CONNECT = auto()
    PRM = auto()
    PRM_STAR = auto()
    CHOMP = auto()
    STOMP = auto()
    TRAJOPT = auto()
    OMPL = auto()
    BIOIK = auto()


@dataclass
class ObjectProperties:
    """Physical properties of an object for planning."""
    object_id: str
    object_type: ObjectType
    dimensions: Dict[str, float]  # width, height, depth, radius, etc.
    weight: float  # kg
    friction_coefficient: float
    center_of_mass: Tuple[float, float, float]
    inertia_matrix: np.ndarray
    is_fragile: bool = False
    is_deformable: bool = False
    max_grasp_force: float = 50.0  # N
    min_grasp_force: float = 10.0  # N
    surface_material: str = "unknown"
    temperature: float = 25.0  # Celsius
    optical_properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'object_id': self.object_id,
            'object_type': self.object_type.name,
            'dimensions': self.dimensions,
            'weight': self.weight,
            'friction_coefficient': self.friction_coefficient,
            'center_of_mass': self.center_of_mass,
            'inertia_matrix': self.inertia_matrix.tolist() if hasattr(self.inertia_matrix, 'tolist') else self.inertia_matrix,
            'is_fragile': self.is_fragile,
            'is_deformable': self.is_deformable,
            'max_grasp_force': self.max_grasp_force,
            'min_grasp_force': self.min_grasp_force,
            'surface_material': self.surface_material,
            'temperature': self.temperature,
            'optical_properties': self.optical_properties
        }


@dataclass
class GraspCandidate:
    """Candidate grasp pose with quality metrics."""
    grasp_id: str
    object_id: str
    grasp_pose: PoseStamped
    grasp_type: GraspType
    approach_pose: PoseStamped
    retreat_pose: PoseStamped
    pre_grasp_pose: PoseStamped
    quality_score: float  # 0.0 to 1.0
    force_required: float  # N
    collision_free: bool = False
    ik_solution_exists: bool = False
    stability_score: float = 0.0
    dexterity_score: float = 0.0
    robustness_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'grasp_id': self.grasp_id,
            'object_id': self.object_id,
            'grasp_type': self.grasp_type.name,
            'quality_score': self.quality_score,
            'force_required': self.force_required,
            'collision_free': self.collision_free,
            'ik_solution_exists': self.ik_solution_exists,
            'stability_score': self.stability_score,
            'dexterity_score': self.dexterity_score,
            'robustness_score': self.robustness_score,
            'metadata': self.metadata
        }


@dataclass
class PlaceCandidate:
    """Candidate placement location."""
    place_id: str
    place_pose: PoseStamped
    surface_type: str
    stability_score: float
    accessibility_score: float
    collision_free: bool = False
    ik_solution_exists: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskPlan:
    """Complete pick and place task plan."""
    plan_id: str
    task_type: str  # "pick", "place", "pick_and_place", "batch"
    object_id: str
    grasp_candidate: GraspCandidate
    place_candidate: Optional[PlaceCandidate]
    trajectory: Optional[JointTrajectory]
    waypoints: List[PoseStamped]
    cost: float
    execution_time_estimate: float  # seconds
    success_probability: float  # 0.0 to 1.0
    risk_level: str  # "low", "medium", "high"
    recovery_plans: List['TaskPlan'] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionMonitor:
    """Monitor task execution in real-time."""
    task_id: str
    start_time: datetime
    current_state: TaskState
    progress: float  # 0.0 to 1.0
    current_waypoint: int
    total_waypoints: int
    execution_time: float  # seconds
    error_count: int
    recovery_attempts: int
    force_readings: List[float]
    torque_readings: List[float]
    position_errors: List[float]
    velocity_errors: List[float]
    warnings: List[str]
    errors: List[str]


class PickPlacePlanner(Node):
    """
    Advanced Pick and Place Planner for robotic manipulation.
    
    Features:
    - Multi-objective grasp planning
    - Collision-aware trajectory optimization
    - Real-time adaptation to environment changes
    - Failure detection and recovery
    - Integration with perception and control
    - Learning from experience (success/failure)
    - Multi-robot coordination
    """
    
    def __init__(self):
        super().__init__('pick_place_planner')
        
        # Configuration parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('planning_timeout_sec', 30.0),
                ('max_planning_attempts', 3),
                ('grasp_quality_threshold', 0.6),
                ('collision_checking_enabled', True),
                ('replanning_enabled', True),
                ('replanning_threshold', 0.1),  # 10cm position error
                ('execution_monitoring_enabled', True),
                ('force_torque_threshold', 50.0),  # N
                ('velocity_threshold', 1.0),  # m/s
                ('acceleration_threshold', 2.0),  # m/s²
                ('gripper_force_margin', 0.2),  # 20% margin
                ('default_grasp_depth', 0.02),  # meters
                ('approach_distance', 0.1),  # meters
                ('retreat_distance', 0.15),  # meters
                ('waypoint_tolerance', 0.01),  # meters
                ('joint_tolerance', 0.017),  # radians (~1 degree)
                ('max_joint_velocity', 1.57),  # rad/s
                ('max_joint_acceleration', 3.14),  # rad/s²
                ('planning_algorithm', 'RRTStar'),
                ('optimization_objectives', ['time', 'energy', 'smoothness']),
                ('learning_enabled', True),
                ('experience_database_path', '/var/robot_planner/experience.db'),
                ('debug_visualization', True),
                ('publish_planning_scene', True),
                ('use_simulated_perception', False),
                ('max_concurrent_tasks', 3),
                ('task_priority_policy', 'fifo'),  # fifo, priority, emergency
                ('safety_monitoring_enabled', True),
                ('emergency_stop_threshold', 100.0),  # N
                ('thermal_monitoring_enabled', True),
                ('max_temperature', 80.0),  # Celsius
                ('power_monitoring_enabled', True),
                ('max_power_consumption', 500.0),  # Watts
                ('communication_timeout_sec', 5.0),
                ('heartbeat_interval_sec', 1.0),
                ('log_level', 'info'),
                ('data_logging_enabled', True),
                ('performance_monitoring_enabled', True),
                ('backup_planning_enabled', True),
                ('multi_arm_coordination', False),
                ('human_robot_collaboration', False),
                ('adaptive_planning_enabled', True),
                ('real_time_optimization', True),
                ('predictive_planning', False),
                ('ml_based_planning', False),
                ('cloud_sync_enabled', False),
                ('encryption_enabled', False),
                ('compliance_monitoring', True),
                ('audit_logging_enabled', True)
            ]
        )
        
        # Initialize state
        self.current_state = TaskState.IDLE
        self.task_queue: Deque[PickPlaceGoal] = deque()
        self.active_tasks: Dict[str, ExecutionMonitor] = {}
        self.completed_tasks: Deque[ExecutionMonitor] = deque(maxlen=1000)
        self.failed_tasks: Deque[ExecutionMonitor] = deque(maxlen=100)
        
        # Knowledge base
        self.object_database: Dict[str, ObjectProperties] = {}
        self.grasp_database: Dict[str, List[GraspCandidate]] = {}
        self.place_database: Dict[str, List[PlaceCandidate]] = {}
        self.experience_database: Dict[str, List[Dict]] = {}
        
        # Planning scene
        self.planning_scene: Optional[PlanningScene] = None
        self.collision_objects: Dict[str, CollisionObject] = {}
        self.attached_objects: Dict[str, AttachedCollisionObject] = {}
        
        # MoveIt integration
        self.moveit_available = MOVEIT_AVAILABLE
        if self.moveit_available:
            self.moveit: Optional[MoveItPy] = None
            self.planning_scene_monitor: Optional[PlanningSceneMonitor] = None
        
        # Threading
        self.callback_group = ReentrantCallbackGroup()
        self.executor = MultiThreadedExecutor()
        self.lock = threading.RLock()
        
        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Initialize ROS2 interfaces
        self._initialize_ros_interfaces()
        
        # Initialize planning components
        self._initialize_planning_components()
        
        # Load experience database
        self._load_experience_database()
        
        # Start monitoring threads
        self._start_monitoring_threads()
        
        # Start timers
        self._start_timers()
        
        self.get_logger().info(
            f"PickPlacePlanner initialized. MoveIt available: {self.moveit_available}"
        )
        
        # Log startup
        self._log_event("system", "planner_initialized", {
            "version": "3.2.0",
            "moveit_available": self.moveit_available,
            "timestamp": datetime.now().isoformat()
        })
    
    def _initialize_ros_interfaces(self):
        """Initialize all ROS2 publishers, subscribers, services, and actions."""
        # QoS profiles
        reliable_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=100,
            durability=QoSDurabilityPolicy.VOLATILE
        )
        
        best_effort_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            durability=QoSDurabilityPolicy.VOLATILE
        )
        
        # Action servers
        self.pick_place_action_server = ActionServer(
            self,
            PickPlaceAction,
            'pick_place',
            self._handle_pick_place_action,
            callback_group=self.callback_group,
            goal_callback=self._validate_goal,
            cancel_callback=self._handle_cancel
        )
        
        # Action clients
        self.navigate_action_client = ActionClient(
            self,
            NavigateAction,
            'navigate',
            callback_group=self.callback_group
        )
        
        # Publishers
        self.plan_pub = self.create_publisher(
            JointTrajectory,
            '/planning/trajectory',
            reliable_qos
        )
        
        self.visualization_pub = self.create_publisher(
            MarkerArray,
            '/planning/visualization',
            best_effort_qos
        )
        
        self.status_pub = self.create_publisher(
            String,
            '/planning/status',
            reliable_qos
        )
        
        self.event_pub = self.create_publisher(
            String,
            '/planning/events',
            reliable_qos
        )
        
        # Subscribers
        self.create_subscription(
            DetectedObject,
            '/perception/detected_objects',
            self._handle_detected_objects,
            best_effort_qos,
            callback_group=self.callback_group
        )
        
        self.create_subscription(
            ObstacleMap,
            '/perception/obstacle_map',
            self._handle_obstacle_map,
            best_effort_qos,
            callback_group=self.callback_group
        )
        
        self.create_subscription(
            RobotStatus,
            '/robot/status',
            self._handle_robot_status,
            reliable_qos,
            callback_group=self.callback_group
        )
        
        self.create_subscription(
            SystemHealth,
            '/system/health',
            self._handle_system_health,
            reliable_qos
        )
        
        # Services
        self.services = {
            'pick_object': self.create_service(
                PickObject,
                '/planning/pick_object',
                self._handle_pick_object_service,
                callback_group=self.callback_group
            ),
            'place_object': self.create_service(
                PlaceObject,
                '/planning/place_object',
                self._handle_place_object_service,
                callback_group=self.callback_group
            ),
            'clear_obstacles': self.create_service(
                ClearObstacleMap,
                '/planning/clear_obstacles',
                self._handle_clear_obstacles,
                callback_group=self.callback_group
            ),
            'get_grasp_poses': self.create_service(
                GetGraspPoses,
                '/planning/get_grasp_poses',
                self._handle_get_grasp_poses,
                callback_group=self.callback_group
            ),
            'validate_grasp': self.create_service(
                ValidateGrasp,
                '/planning/validate_grasp',
                self._handle_validate_grasp,
                callback_group=self.callback_group
            ),
            'compute_ik': self.create_service(
                ComputeIK,
                '/planning/compute_ik',
                self._handle_compute_ik,
                callback_group=self.callback_group
            ),
            'compute_fk': self.create_service(
                ComputeFK,
                '/planning/compute_fk',
                self._handle_compute_fk,
                callback_group=self.callback_group
            )
        }
        
        # MoveIt services (if available)
        if self.moveit_available:
            self.moveit_services = {
                'get_planning_scene': self.create_client(
                    GetPlanningScene,
                    '/get_planning_scene',
                    callback_group=self.callback_group
                ),
                'apply_planning_scene': self.create_client(
                    ApplyPlanningScene,
                    '/apply_planning_scene',
                    callback_group=self.callback_group
                )
            }
    
    def _initialize_planning_components(self):
        """Initialize planning algorithms and components."""
        # Initialize MoveIt if available
        if self.moveit_available:
            try:
                self.moveit = MoveItPy(node_name="pick_place_planner")
                self.planning_scene_monitor = PlanningSceneMonitor(node=self)
                self.get_logger().info("MoveItPy initialized successfully")
            except Exception as e:
                self.get_logger().error(f"Failed to initialize MoveItPy: {str(e)}")
                self.moveit_available = False
        
        # Initialize planning algorithms
        self.planning_algorithms = {
            PlanningAlgorithm.RRT: self._plan_rrt,
            PlanningAlgorithm.RRT_STAR: self._plan_rrt_star,
            PlanningAlgorithm.RRT_CONNECT: self._plan_rrt_connect,
            PlanningAlgorithm.PRM: self._plan_prm,
            PlanningAlgorithm.PRM_STAR: self._plan_prm_star,
            PlanningAlgorithm.CHOMP: self._plan_chomp,
            PlanningAlgorithm.STOMP: self._plan_stomp,
            PlanningAlgorithm.TRAJOPT: self._plan_trajopt,
            PlanningAlgorithm.OMPL: self._plan_ompl,
            PlanningAlgorithm.BIOIK: self._plan_bioik
        }
        
        # Initialize grasp planners
        self.grasp_planners = {
            'geometric': self._plan_geometric_grasp,
            'learning_based': self._plan_learning_based_grasp,
            'force_closure': self._plan_force_closure_grasp,
            'template_based': self._plan_template_based_grasp,
            'sampling_based': self._plan_sampling_based_grasp
        }
        
        # Initialize optimization objectives
        self.optimization_objectives = {
            'time': self._optimize_time,
            'energy': self._optimize_energy,
            'smoothness': self._optimize_smoothness,
            'safety': self._optimize_safety,
            'reliability': self._optimize_reliability,
            'precision': self._optimize_precision
        }
        
        # Initialize recovery strategies
        self.recovery_strategies = {
            'retry': self._recovery_retry,
            'replan': self._recovery_replan,
            'regrasp': self._recovery_regrasp,
            'reorient': self._recovery_reorient,
            'fallback_grasp': self._recovery_fallback_grasp,
            'human_assistance': self._recovery_human_assistance,
            'emergency_stop': self._recovery_emergency_stop
        }
    
    def _load_experience_database(self):
        """Load experience database from file."""
        db_path = Path(self.get_parameter('experience_database_path').value)
        if db_path.exists():
            try:
                with open(db_path, 'rb') as f:
                    data = pickle.load(f)
                    self.experience_database = data.get('experience', {})
                    self.object_database = data.get('objects', {})
                    self.grasp_database = data.get('grasps', {})
                    self.place_database = data.get('places', {})
                self.get_logger().info(f"Loaded experience database from {db_path}")
            except Exception as e:
                self.get_logger().error(f"Failed to load experience database: {str(e)}")
    
    def _save_experience_database(self):
        """Save experience database to file."""
        db_path = Path(self.get_parameter('experience_database_path').value)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'experience': self.experience_database,
            'objects': self.object_database,
            'grasps': self.grasp_database,
            'places': self.place_database,
            'timestamp': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        try:
            with open(db_path, 'wb') as f:
                pickle.dump(data, f)
            self.get_logger().debug("Saved experience database")
        except Exception as e:
            self.get_logger().error(f"Failed to save experience database: {str(e)}")
    
    def _start_monitoring_threads(self):
        """Start monitoring and maintenance threads."""
        # Task execution monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        # Planning scene update thread
        self.scene_update_thread = threading.Thread(target=self._scene_update_loop)
        self.scene_update_thread.daemon = True
        self.scene_update_thread.start()
        
        # Database maintenance thread
        self.database_thread = threading.Thread(target=self._database_maintenance_loop)
        self.database_thread.daemon = True
        self.database_thread.start()
        
        # Performance monitoring thread
        if self.get_parameter('performance_monitoring_enabled').value:
            self.performance_thread = threading.Thread(target=self._performance_monitoring_loop)
            self.performance_thread.daemon = True
            self.performance_thread.start()
    
    def _start_timers(self):
        """Start periodic timers."""
        # Heartbeat timer
        self.create_timer(
            self.get_parameter('heartbeat_interval_sec').value,
            self._publish_heartbeat
        )
        
        # Status update timer
        self.create_timer(
            1.0,
            self._publish_status
        )
        
        # Visualization update timer
        if self.get_parameter('debug_visualization').value:
            self.create_timer(
                0.5,
                self._publish_visualization
            )
        
        # Database save timer
        self.create_timer(
            300.0,  # 5 minutes
            self._save_experience_database
        )
        
        # Performance logging timer
        if self.get_parameter('performance_monitoring_enabled').value:
            self.create_timer(
                60.0,
                self._log_performance_metrics
            )
    
    def _handle_pick_place_action(self, goal_handle):
        """
        Handle pick and place action requests.
        
        This is the main entry point for pick and place tasks.
        """
        goal = goal_handle.request
        task_id = goal.task_id or self._generate_task_id()
        
        self.get_logger().info(f"Received pick-place task {task_id}: {goal.task_type}")
        
        # Create execution monitor
        monitor = ExecutionMonitor(
            task_id=task_id,
            start_time=datetime.now(),
            current_state=TaskState.PLANNING,
            progress=0.0,
            current_waypoint=0,
            total_waypoints=0,
            execution_time=0.0,
            error_count=0,
            recovery_attempts=0,
            force_readings=[],
            torque_readings=[],
            position_errors=[],
            velocity_errors=[],
            warnings=[],
            errors=[]
        )
        
        with self.lock:
            self.active_tasks[task_id] = monitor
        
        # Publish feedback
        self._publish_feedback(goal_handle, task_id, 0.0, "Planning started")
        
        try:
            # Plan the task
            plan = self._plan_task(goal, task_id)
            
            if not plan:
                raise Exception("Failed to create task plan")
            
            # Update monitor
            monitor.current_state = TaskState.EXECUTING
            monitor.total_waypoints = len(plan.waypoints)
            monitor.progress = 0.1
            
            # Publish plan
            if plan.trajectory:
                self.plan_pub.publish(plan.trajectory)
                self._publish_feedback(goal_handle, task_id, 0.1, "Plan published")
            
            # Execute the plan
            result = self._execute_task(goal_handle, plan, monitor)
            
            # Update monitor
            monitor.current_state = TaskState.SUCCEEDED if result.success else TaskState.FAILED
            monitor.progress = 1.0
            monitor.execution_time = (datetime.now() - monitor.start_time).total_seconds()
            
            # Record experience
            self._record_experience(task_id, plan, result.success, monitor)
            
            # Clean up
            with self.lock:
                self.completed_tasks.append(monitor)
                if task_id in self.active_tasks:
                    del self.active_tasks[task_id]
            
            # Return result
            goal_handle.succeed()
            return result
            
        except Exception as e:
            self.get_logger().error(f"Task {task_id} failed: {str(e)}")
            
            # Update monitor
            monitor.current_state = TaskState.FAILED
            monitor.errors.append(str(e))
            monitor.execution_time = (datetime.now() - monitor.start_time).total_seconds()
            
            # Clean up
            with self.lock:
                self.failed_tasks.append(monitor)
                if task_id in self.active_tasks:
                    del self.active_tasks[task_id]
            
            # Return failure
            goal_handle.abort()
            return PickPlaceResult(
                success=False,
                error_message=str(e),
                execution_time=monitor.execution_time
            )
    
    def _plan_task(self, goal: PickPlaceGoal, task_id: str) -> Optional[TaskPlan]:
        """
        Plan a complete pick and place task.
        
        This is the core planning function that coordinates:
        1. Object identification and property estimation
        2. Grasp planning
        3. Motion planning
        4. Place planning (if needed)
        5. Trajectory optimization
        """
        start_time = time.time()
        planning_timeout = self.get_parameter('planning_timeout_sec').value
        
        try:
            # Step 1: Identify object and get properties
            object_properties = self._identify_object(goal.object_id, goal.object_pose)
            
            if not object_properties:
                raise Exception(f"Could not identify object {goal.object_id}")
            
            # Step 2: Generate grasp candidates
            grasp_candidates = self._generate_grasp_candidates(
                object_properties, 
                goal.object_pose,
                goal.preferred_grasp_type
            )
            
            if not grasp_candidates:
                raise Exception("No valid grasp candidates found")
            
            # Step 3: Select best grasp
            selected_grasp = self._select_best_grasp(
                grasp_candidates, 
                object_properties,
                goal.task_type
            )
            
            if not selected_grasp:
                raise Exception("Failed to select a valid grasp")
            
            # Step 4: Generate place candidates (if needed)
            place_candidate = None
            if goal.task_type in ['place', 'pick_and_place'] and goal.target_pose:
                place_candidates = self._generate_place_candidates(
                    object_properties,
                    goal.target_pose,
                    goal.target_surface
                )
                
                if place_candidates:
                    place_candidate = self._select_best_place(
                        place_candidates,
                        object_properties,
                        selected_grasp
                    )
            
            # Step 5: Plan motion trajectory
            trajectory = self._plan_motion_trajectory(
                selected_grasp,
                place_candidate,
                goal.task_type,
                object_properties
            )
            
            if not trajectory and goal.task_type != 'place':
                raise Exception("Failed to plan motion trajectory")
            
            # Step 6: Create complete task plan
            plan = TaskPlan(
                plan_id=f"plan_{task_id}",
                task_type=goal.task_type,
                object_id=goal.object_id,
                grasp_candidate=selected_grasp,
                place_candidate=place_candidate,
                trajectory=trajectory,
                waypoints=self._extract_waypoints(trajectory) if trajectory else [],
                cost=self._calculate_plan_cost(selected_grasp, place_candidate, trajectory),
                execution_time_estimate=self._estimate_execution_time(trajectory),
                success_probability=self._estimate_success_probability(
                    selected_grasp, 
                    place_candidate, 
                    object_properties
                ),
                risk_level=self._assess_risk_level(object_properties, selected_grasp),
                constraints=goal.constraints or {},
                metadata={
                    'planning_time': time.time() - start_time,
                    'planning_algorithm': self.get_parameter('planning_algorithm').value,
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            # Step 7: Generate backup plans
            if self.get_parameter('backup_planning_enabled').value:
                plan.recovery_plans = self._generate_backup_plans(
                    plan, 
                    object_properties,
                    goal
                )
            
            # Log planning completion
            planning_time = time.time() - start_time
            self.get_logger().info(
                f"Task {task_id} planning completed in {planning_time:.2f}s. "
                f"Success probability: {plan.success_probability:.2f}"
            )
            
            return plan
            
        except Exception as e:
            self.get_logger().error(f"Planning failed for task {task_id}: {str(e)}")
            
            # Check if we should retry with different parameters
            if time.time() - start_time < planning_timeout:
                return self._plan_task_with_recovery(goal, task_id, str(e))
            
            return None
    
    def _plan_task_with_recovery(self, goal: PickPlaceGoal, task_id: str, 
                                error_message: str) -> Optional[TaskPlan]:
        """
        Attempt to recover from planning failure with alternative strategies.
        """
        recovery_attempts = 0
        max_attempts = self.get_parameter('max_planning_attempts').value
        
        while recovery_attempts < max_attempts:
            recovery_attempts += 1
            self.get_logger().info(
                f"Attempting recovery {recovery_attempts}/{max_attempts} for task {task_id}"
            )
            
            try:
                # Try different recovery strategies
                if "grasp" in error_message.lower():
                    # Try different grasp strategies
                    return self._recover_grasp_failure(goal, task_id)
                elif "collision" in error_message.lower():
                    # Try different motion planning
                    return self._recover_collision_failure(goal, task_id)
                elif "ik" in error_message.lower():
                    # Try different IK solutions
                    return self._recover_ik_failure(goal, task_id)
                else:
                    # Try general recovery
                    return self._recover_general_failure(goal, task_id)
                    
            except Exception as e:
                self.get_logger().warning(
                    f"Recovery attempt {recovery_attempts} failed: {str(e)}"
                )
                if recovery_attempts >= max_attempts:
                    break
        
        return None
    
    def _identify_object(self, object_id: str, object_pose: PoseStamped) -> Optional[ObjectProperties]:
        """
        Identify object and estimate its properties.
        
        Uses:
        1. Predefined object database
        2. Perception data
        3. Machine learning models
        4. Physical interaction (if allowed)
        """
        # Check if object is already in database
        if object_id in self.object_database:
            self.get_logger().debug(f"Found object {object_id} in database")
            return self.object_database[object_id]
        
        # Try to estimate properties from perception
        properties = self._estimate_object_properties(object_id, object_pose)
        
        if properties:
            # Store in database
            self.object_database[object_id] = properties
            self.get_logger().info(f"Estimated properties for object {object_id}")
            return properties
        
        # Use default properties for unknown objects
        self.get_logger().warning(f"Using default properties for unknown object {object_id}")
        
        default_props = ObjectProperties(
            object_id=object_id,
            object_type=ObjectType.UNKNOWN,
            dimensions={'width': 0.05, 'height': 0.05, 'depth': 0.05},
            weight=0.1,  # 100g
            friction_coefficient=0.3,
            center_of_mass=(0.0, 0.0, 0.0),
            inertia_matrix=np.eye(3) * 0.001,
            is_fragile=False,
            is_deformable=False,
            max_grasp_force=30.0,
            min_grasp_force=5.0,
            surface_material="unknown"
        )
        
        self.object_database[object_id] = default_props
        return default_props
    
    def _estimate_object_properties(self, object_id: str, 
                                  object_pose: PoseStamped) -> Optional[ObjectProperties]:
        """
        Estimate object properties using perception and ML.
        
        In a real system, this would:
        1. Analyze point cloud data
        2. Use ML models for material recognition
        3. Estimate mass from size and material
        4. Detect fragility from appearance
        """
        # For now, return None to use default properties
        # In production, this would integrate with perception system
        return None
    
    def _generate_grasp_candidates(self, object_props: ObjectProperties,
                                 object_pose: PoseStamped,
                                 preferred_grasp_type: str = None) -> List[GraspCandidate]:
        """
        Generate multiple grasp candidates for an object.
        
        Uses multiple grasp planning strategies:
        1. Geometric grasp planning
        2. Learning-based grasp planning
        3. Force closure analysis
        4. Template-based grasping
        5. Sampling-based exploration
        """
        grasp_candidates = []
        
        # Try different grasp planners
        for planner_name, planner_func in self.grasp_planners.items():
            try:
                candidates = planner_func(object_props, object_pose, preferred_grasp_type)
                grasp_candidates.extend(candidates)
                
                self.get_logger().debug(
                    f"Planner {planner_name} generated {len(candidates)} grasp candidates"
                )
                
            except Exception as e:
                self.get_logger().debug(
                    f"Grasp planner {planner_name} failed: {str(e)}"
                )
        
        # Filter and score candidates
        filtered_candidates = self._filter_grasp_candidates(grasp_candidates, object_props)
        scored_candidates = self._score_grasp_candidates(filtered_candidates, object_props)
        
        # Sort by score
        scored_candidates.sort(key=lambda x: x.quality_score, reverse=True)
        
        self.get_logger().info(
            f"Generated {len(scored_candidates)} valid grasp candidates "
            f"(from {len(grasp_candidates)} raw candidates)"
        )
        
        return scored_candidates
    
    def _plan_geometric_grasp(self, object_props: ObjectProperties,
                            object_pose: PoseStamped,
                            preferred_grasp_type: str = None) -> List[GraspCandidate]:
        """
        Generate grasps based on geometric properties of the object.
        
        For different object types:
        - Boxes: top, side, corner grasps
        - Cylinders: axial, radial grasps
        - Spheres: enveloping grasps
        - Unknown: sampling-based grasps
        """
        candidates = []
        grasp_depth = self.get_parameter('default_grasp_depth').value
        
        if object_props.object_type == ObjectType.BOX:
            # Box grasps
            candidates.extend(self._generate_box_grasps(object_props, object_pose, grasp_depth))
        
        elif object_props.object_type == ObjectType.CYLINDER:
            # Cylinder grasps
            candidates.extend(self._generate_cylinder_grasps(object_props, object_pose, grasp_depth))
        
        elif object_props.object_type == ObjectType.SPHERE:
            # Sphere grasps
            candidates.extend(self._generate_sphere_grasps(object_props, object_pose, grasp_depth))
        
        else:
            # Unknown object - try various approaches
            candidates.extend(self._generate_sampling_grasps(object_props, object_pose, grasp_depth))
        
        return candidates
    
    def _generate_box_grasps(self, object_props: ObjectProperties,
                           object_pose: PoseStamped,
                           grasp_depth: float) -> List[GraspCandidate]:
        """
        Generate grasps for box-shaped objects.
        """
        candidates = []
        dimensions = object_props.dimensions
        
        # Extract dimensions
        width = dimensions.get('width', 0.05)
        height = dimensions.get('height', 0.05)
        depth = dimensions.get('depth', 0.05)
        
        # Generate top grasps
        top_grasp = self._create_box_top_grasp(object_pose, width, height, depth, grasp_depth)
        if top_grasp:
            candidates.append(top_grasp)
        
        # Generate side grasps
        side_grasps = self._create_box_side_grasps(object_pose, width, height, depth, grasp_depth)
        candidates.extend(side_grasps)
        
        # Generate corner grasps (for larger objects)
        if max(width, height, depth) > 0.1:  # Larger than 10cm
            corner_grasps = self._create_box_corner_grasps(object_pose, width, height, depth, grasp_depth)
            candidates.extend(corner_grasps)
        
        return candidates
    
    def _create_box_top_grasp(self, object_pose: PoseStamped,
                            width: float, height: float, depth: float,
                            grasp_depth: float) -> Optional[GraspCandidate]:
        """
        Create a top grasp for a box.
        """
        try:
            # Create grasp pose at top center
            grasp_pose = copy.deepcopy(object_pose)
            
            # Offset to top center
            grasp_pose.pose.position.z += height / 2 + grasp_depth / 2
            
            # Orient gripper to grasp from top
            # This depends on your gripper orientation
            # For a parallel jaw gripper, typically aligned with object axes
            
            grasp_id = f"top_grasp_{hashlib.md5(str(object_pose).encode()).hexdigest()[:8]}"
            
            return GraspCandidate(
                grasp_id=grasp_id,
                object_id=object_pose.header.frame_id,
                grasp_pose=grasp_pose,
                grasp_type=GraspType.TOP_GRASP,
                approach_pose=self._compute_approach_pose(grasp_pose, -grasp_depth),
                retreat_pose=self._compute_retreat_pose(grasp_pose, grasp_depth * 2),
                pre_grasp_pose=self._compute_pre_grasp_pose(grasp_pose, grasp_depth),
                quality_score=0.8,  # Good for top grasps
                force_required=20.0,  # Estimate
                metadata={
                    'grasp_strategy': 'top_center',
                    'object_type': 'box',
                    'dimensions': {'width': width, 'height': height, 'depth': depth}
                }
            )
        except Exception as e:
            self.get_logger().debug(f"Failed to create top grasp: {str(e)}")
            return None
    
    def _create_box_side_grasps(self, object_pose: PoseStamped,
                              width: float, height: float, depth: float,
                              grasp_depth: float) -> List[GraspCandidate]:
        """
        Create side grasps for a box.
        """
        candidates = []
        
        # Generate grasps for each side
        sides = [
            ('+x', (width / 2 + grasp_depth / 2, 0, 0), (0, 0, 0, 1)),
            ('-x', (-width / 2 - grasp_depth / 2, 0, 0), (0, 0, 1, 0)),
            ('+y', (0, depth / 2 + grasp_depth / 2, 0), (0, 0, 0.707, 0.707)),
            ('-y', (0, -depth / 2 - grasp_depth / 2, 0), (0, 0, -0.707, 0.707)),
        ]
        
        for side_name, offset, orientation in sides:
            try:
                grasp_pose = copy.deepcopy(object_pose)
                
                # Apply offset
                grasp_pose.pose.position.x += offset[0]
                grasp_pose.pose.position.y += offset[1]
                grasp_pose.pose.position.z += offset[2]
                
                # Set orientation (simplified - would need proper quaternion math)
                grasp_pose.pose.orientation.x = orientation[0]
                grasp_pose.pose.orientation.y = orientation[1]
                grasp_pose.pose.orientation.z = orientation[2]
                grasp_pose.pose.orientation.w = orientation[3]
                
                grasp_id = f"side_{side_name}_grasp_{hashlib.md5(str(object_pose).encode()).hexdigest()[:8]}"
                
                candidates.append(GraspCandidate(
                    grasp_id=grasp_id,
                    object_id=object_pose.header.frame_id,
                    grasp_pose=grasp_pose,
                    grasp_type=GraspType.SIDE_GRASP,
                    approach_pose=self._compute_approach_pose(grasp_pose, -grasp_depth),
                    retreat_pose=self._compute_retreat_pose(grasp_pose, grasp_depth * 2),
                    pre_grasp_pose=self._compute_pre_grasp_pose(grasp_pose, grasp_depth),
                    quality_score=0.7,
                    force_required=25.0,
                    metadata={
                        'grasp_strategy': f'side_{side_name}',
                        'object_type': 'box',
                        'dimensions': {'width': width, 'height': height, 'depth': depth}
                    }
                ))
            except Exception as e:
                self.get_logger().debug(f"Failed to create side grasp {side_name}: {str(e)}")
        
        return candidates
    
    def _plan_learning_based_grasp(self, object_props: ObjectProperties,
                                 object_pose: PoseStamped,
                                 preferred_grasp_type: str = None) -> List[GraspCandidate]:
        """
        Generate grasps using machine learning models.
        
        In production, this would:
        1. Use CNN for grasp prediction from images
        2. Use point cloud neural networks for 3D grasp prediction
        3. Leverage reinforcement learning for adaptive grasping
        4. Use transfer learning from simulation to real world
        """
        # Placeholder for ML-based grasp planning
        # In reality, this would call a trained ML model
        
        candidates = []
        
        if self.get_parameter('ml_based_planning').value:
            try:
                # This is where you'd integrate with your ML model
                # For now, return empty list
                self.get_logger().debug("ML-based grasp planning not implemented")
            except Exception as e:
                self.get_logger().debug(f"ML grasp planning failed: {str(e)}")
        
        return candidates
    
    def _plan_force_closure_grasp(self, object_props: ObjectProperties,
                                object_pose: PoseStamped,
                                preferred_grasp_type: str = None) -> List[GraspCandidate]:
        """
        Generate grasps based on force closure analysis.
        
        Analyzes whether a grasp can resist external wrenches.
        """
        candidates = []
        
        # Simple force closure analysis
        # In production, this would use proper force closure metrics
        
        try:
            # Generate candidate contact points
            contact_points = self._generate_contact_points(object_props, object_pose)
            
            for i, contact_set in enumerate(contact_points):
                # Check force closure
                if self._check_force_closure(contact_set, object_props):
                    # Create grasp from contact points
                    grasp = self._create_grasp_from_contacts(
                        contact_set, 
                        object_props, 
                        object_pose,
                        i
                    )
                    if grasp:
                        candidates.append(grasp)
                        
        except Exception as e:
            self.get_logger().debug(f"Force closure analysis failed: {str(e)}")
        
        return candidates
    
    def _filter_grasp_candidates(self, candidates: List[GraspCandidate],
                               object_props: ObjectProperties) -> List[GraspCandidate]:
        """
        Filter grasp candidates based on constraints and feasibility.
        """
        filtered = []
        quality_threshold = self.get_parameter('grasp_quality_threshold').value
        
        for candidate in candidates:
            # Check quality threshold
            if candidate.quality_score < quality_threshold:
                continue
            
            # Check collision (if enabled)
            if self.get_parameter('collision_checking_enabled').value:
                if not self._check_grasp_collision(candidate, object_props):
                    candidate.collision_free = False
                    continue
                candidate.collision_free = True
            
            # Check IK feasibility
            if not self._check_ik_feasibility(candidate.grasp_pose):
                candidate.ik_solution_exists = False
                continue
            candidate.ik_solution_exists = True
            
            filtered.append(candidate)
        
        return filtered
    
    def _score_grasp_candidates(self, candidates: List[GraspCandidate],
                              object_props: ObjectProperties) -> List[GraspCandidate]:
        """
        Score grasp candidates using multiple metrics.
        """
        for candidate in candidates:
            # Calculate stability score
            candidate.stability_score = self._calculate_stability_score(
                candidate, object_props
            )
            
            # Calculate dexterity score
            candidate.dexterity_score = self._calculate_dexterity_score(
                candidate, object_props
            )
            
            # Calculate robustness score
            candidate.robustness_score = self._calculate_robustness_score(
                candidate, object_props
            )
            
            # Combine scores into quality score
            candidate.quality_score = (
                0.4 * candidate.stability_score +
                0.3 * candidate.dexterity_score +
                0.3 * candidate.robustness_score
            )
            
            # Adjust based on object properties
            if object_props.is_fragile:
                # Prefer gentler grasps for fragile objects
                candidate.quality_score *= 0.9 if candidate.grasp_type == GraspType.POWER_GRASP else 1.1
            
            if object_props.is_deformable:
                # Prefer enveloping grasps for deformable objects
                candidate.quality_score *= 1.1 if candidate.grasp_type == GraspType.POWER_GRASP else 0.9
        
        return candidates
    
    def _select_best_grasp(self, candidates: List[GraspCandidate],
                         object_props: ObjectProperties,
                         task_type: str) -> Optional[GraspCandidate]:
        """
        Select the best grasp candidate for the task.
        
        Considers:
        1. Task type (pick, place, pick_and_place)
        2. Object properties
        3. Environmental constraints
        4. Historical success rates
        """
        if not candidates:
            return None
        
        # Sort by quality score
        candidates.sort(key=lambda x: x.quality_score, reverse=True)
        
        # Task-specific adjustments
        if task_type == 'pick':
            # For picking, prioritize stability
            candidates.sort(key=lambda x: x.stability_score, reverse=True)
        elif task_type == 'place':
            # For placing, prioritize dexterity
            candidates.sort(key=lambda x: x.dexterity_score, reverse=True)
        elif task_type == 'pick_and_place':
            # For combined tasks, need balance
            candidates.sort(key=lambda x: (x.stability_score + x.dexterity_score) / 2, reverse=True)
        
        # Check historical success for similar grasps
        for candidate in candidates:
            success_rate = self._get_grasp_success_rate(candidate, object_props)
            if success_rate > 0.8:  # 80% success rate threshold
                self.get_logger().info(
                    f"Selected grasp {candidate.grasp_id} with historical "
                    f"success rate {success_rate:.2f}"
                )
                return candidate
        
        # If no historical data, return the best candidate
        best_candidate = candidates[0]
        self.get_logger().info(
            f"Selected grasp {best_candidate.grasp_id} with "
            f"quality score {best_candidate.quality_score:.2f}"
        )
        
        return best_candidate
    
    def _plan_motion_trajectory(self, grasp: GraspCandidate,
                              place: Optional[PlaceCandidate],
                              task_type: str,
                              object_props: ObjectProperties) -> Optional[JointTrajectory]:
        """
        Plan motion trajectory for the task.
        
        Handles:
        1. Approach trajectory
        2. Grasp/ungrasp motions
        3. Transport trajectory
        4. Place trajectory
        5. Retreat trajectory
        """
        try:
            # Get planning algorithm
            algorithm_name = self.get_parameter('planning_algorithm').value
            algorithm = self._get_planning_algorithm(algorithm_name)
            
            if not algorithm:
                raise Exception(f"Unknown planning algorithm: {algorithm_name}")
            
            # Plan based on task type
            if task_type == 'pick':
                trajectory = self._plan_pick_trajectory(grasp, algorithm, object_props)
            elif task_type == 'place':
                trajectory = self._plan_place_trajectory(grasp, place, algorithm, object_props)
            elif task_type == 'pick_and_place':
                trajectory = self._plan_pick_place_trajectory(grasp, place, algorithm, object_props)
            else:
                raise Exception(f"Unknown task type: {task_type}")
            
            # Optimize trajectory
            if trajectory and self.get_parameter('real_time_optimization').value:
                trajectory = self._optimize_trajectory(trajectory, object_props)
            
            return trajectory
            
        except Exception as e:
            self.get_logger().error(f"Motion planning failed: {str(e)}")
            return None
    
    def _plan_pick_trajectory(self, grasp: GraspCandidate,
                            algorithm: callable,
                            object_props: ObjectProperties) -> Optional[JointTrajectory]:
        """
        Plan trajectory for picking task.
        """
        # Sequence: Home -> Pre-grasp -> Grasp -> Lift -> Home
        waypoints = [
            self._get_home_pose(),
            grasp.pre_grasp_pose,
            grasp.grasp_pose,
            grasp.retreat_pose,
            self._get_home_pose()
        ]
        
        return algorithm(waypoints, object_props)
    
    def _plan_rrt(self, waypoints: List[PoseStamped],
                object_props: ObjectProperties) -> Optional[JointTrajectory]:
        """
        Plan trajectory using RRT algorithm.
        """
        # Simplified RRT implementation
        # In production, this would use MoveIt or OMPL
        
        if not self.moveit_available:
            return self._create_simple_trajectory(waypoints)
        
        try:
            # Use MoveIt for planning
            planning_group = "manipulator"  # Your planning group name
            
            # Set start state
            start_state = self.moveit.get_robot_state()
            
            # Plan to each waypoint
            trajectory_points = []
            
            for i, waypoint in enumerate(waypoints):
                if i == 0:
                    # First waypoint - plan from current state
                    plan_result = self.moveit.plan_poses(
                        poses=[waypoint.pose],
                        planning_group=planning_group,
                        planner_id="RRT"
                    )
                else:
                    # Subsequent waypoints - plan from previous state
                    plan_result = self.moveit.plan_poses(
                        poses=[waypoint.pose],
                        planning_group=planning_group,
                        planner_id="RRT",
                        start_state=trajectory_points[-1] if trajectory_points else None
                    )
                
                if not plan_result or not plan_result.trajectory:
                    raise Exception(f"Failed to plan to waypoint {i}")
                
                # Add trajectory points
                trajectory_points.extend(plan_result.trajectory.joint_trajectory.points)
            
            # Create combined trajectory
            trajectory = JointTrajectory()
            trajectory.joint_names = plan_result.trajectory.joint_trajectory.joint_names
            trajectory.points = trajectory_points
            
            return trajectory
            
        except Exception as e:
            self.get_logger().error(f"RRT planning failed: {str(e)}")
            return None
    
    def _create_simple_trajectory(self, waypoints: List[PoseStamped]) -> JointTrajectory:
        """
        Create a simple linear trajectory for testing.
        """
        trajectory = JointTrajectory()
        trajectory.joint_names = [
            'joint1', 'joint2', 'joint3', 
            'joint4', 'joint5', 'joint6'
        ]
        
        # Create simple linear interpolation
        num_points = len(waypoints) * 10  # 10 points per segment
        time_step = 1.0 / num_points
        
        for i in range(num_points):
            point = JointTrajectoryPoint()
            
            # Simple interpolation (in reality, would compute IK)
            # This is a placeholder
            positions = [
                float(i) / num_points * 1.57,  # Simple linear motion
                float(i) / num_points * 1.57,
                float(i) / num_points * 1.57,
                float(i) / num_points * 1.57,
                float(i) / num_points * 1.57,
                float(i) / num_points * 1.57
            ]
            
            point.positions = positions
            point.velocities = [0.1] * 6
            point.accelerations = [0.05] * 6
            point.time_from_start = rclpy.duration.Duration(seconds=i * time_step).to_msg()
            
            trajectory.points.append(point)
        
        return trajectory
    
    def _optimize_trajectory(self, trajectory: JointTrajectory,
                           object_props: ObjectProperties) -> JointTrajectory:
        """
        Optimize trajectory based on multiple objectives.
        """
        if not trajectory:
            return trajectory
        
        # Get optimization objectives
        objectives = self.get_parameter('optimization_objectives').value
        
        # Apply each optimization
        optimized = copy.deepcopy(trajectory)
        
        for objective in objectives:
            if objective in self.optimization_objectives:
                try:
                    optimized = self.optimization_objectives[objective](
                        optimized, object_props
                    )
                except Exception as e:
                    self.get_logger().debug(
                        f"Optimization {objective} failed: {str(e)}"
                    )
        
        return optimized
    
    def _optimize_time(self, trajectory: JointTrajectory,
                     object_props: ObjectProperties) -> JointTrajectory:
        """
        Optimize trajectory for minimum time.
        """
        # Adjust velocities and accelerations to minimize time
        # while respecting constraints
        
        max_velocity = self.get_parameter('max_joint_velocity').value
        max_acceleration = self.get_parameter('max_joint_acceleration').value
        
        optimized = copy.deepcopy(trajectory)
        
        # Time scaling algorithm
        for i, point in enumerate(optimized.points):
            if i > 0:
                # Adjust time to maximize velocity within limits
                prev_point = optimized.points[i-1]
                time_diff = point.time_from_start.sec - prev_point.time_from_start.sec
                
                # Simple scaling - in reality would use more sophisticated algorithm
                scale_factor = 0.8  # Reduce time by 20%
                new_time = max(0.1, time_diff * scale_factor)
                
                # Update time
                point.time_from_start.sec = int(prev_point.time_from_start.sec + new_time)
        
        return optimized
    
    def _execute_task(self, goal_handle, plan: TaskPlan, 
                    monitor: ExecutionMonitor) -> PickPlaceResult:
        """
        Execute the planned task with real-time monitoring.
        """
        try:
            # Step 1: Move to pre-grasp position
            self._publish_feedback(
                goal_handle, 
                monitor.task_id, 
                0.2, 
                "Moving to pre-grasp position"
            )
            
            success = self._execute_waypoint(
                plan.grasp_candidate.pre_grasp_pose,
                monitor
            )
            
            if not success:
                raise Exception("Failed to reach pre-grasp position")
            
            monitor.progress = 0.3
            monitor.current_waypoint = 1
            
            # Step 2: Approach object
            self._publish_feedback(
                goal_handle,
                monitor.task_id,
                0.3,
                "Approaching object"
            )
            
            success = self._execute_waypoint(
                plan.grasp_candidate.grasp_pose,
                monitor
            )
            
            if not success:
                raise Exception("Failed to approach object")
            
            monitor.progress = 0.4
            monitor.current_waypoint = 2
            
            # Step 3: Execute grasp
            self._publish_feedback(
                goal_handle,
                monitor.task_id,
                0.4,
                "Executing grasp"
            )
            
            grasp_success = self._execute_grasp(
                plan.grasp_candidate,
                plan.object_id,
                monitor
            )
            
            if not grasp_success:
                raise Exception("Grasp failed")
            
            monitor.progress = 0.5
            monitor.current_waypoint = 3
            
            # Step 4: Lift object
            self._publish_feedback(
                goal_handle,
                monitor.task_id,
                0.5,
                "Lifting object"
            )
            
            success = self._execute_waypoint(
                plan.grasp_candidate.retreat_pose,
                monitor
            )
            
            if not success:
                raise Exception("Failed to lift object")
            
            monitor.progress = 0.6
            monitor.current_waypoint = 4
            
            # Step 5: Transport to place position (if needed)
            if plan.place_candidate:
                self._publish_feedback(
                    goal_handle,
                    monitor.task_id,
                    0.6,
                    "Transporting to place position"
                )
                
                # Move to pre-place position
                success = self._execute_waypoint(
                    self._compute_pre_place_pose(plan.place_candidate.place_pose),
                    monitor
                )
                
                if not success:
                    raise Exception("Failed to reach pre-place position")
                
                monitor.progress = 0.7
                monitor.current_waypoint = 5
                
                # Approach place position
                self._publish_feedback(
                    goal_handle,
                    monitor.task_id,
                    0.7,
                    "Approaching place position"
                )
                
                success = self._execute_waypoint(
                    plan.place_candidate.place_pose,
                    monitor
                )
                
                if not success:
                    raise Exception("Failed to approach place position")
                
                monitor.progress = 0.8
                monitor.current_waypoint = 6
                
                # Execute place
                self._publish_feedback(
                    goal_handle,
                    monitor.task_id,
                    0.8,
                    "Placing object"
                )
                
                place_success = self._execute_place(
                    plan.place_candidate,
                    plan.object_id,
                    monitor
                )
                
                if not place_success:
                    raise Exception("Place failed")
                
                monitor.progress = 0.9
                monitor.current_waypoint = 7
                
                # Retreat from place position
                self._publish_feedback(
                    goal_handle,
                    monitor.task_id,
                    0.9,
                    "Retreating from place position"
                )
                
                success = self._execute_waypoint(
                    self._compute_retreat_pose(plan.place_candidate.place_pose),
                    monitor
                )
                
                if not success:
                    raise Exception("Failed to retreat from place position")
            
            # Step 6: Return to home
            self._publish_feedback(
                goal_handle,
                monitor.task_id,
                0.95,
                "Returning to home position"
            )
            
            success = self._execute_waypoint(
                self._get_home_pose(),
                monitor
            )
            
            if not success:
                raise Exception("Failed to return to home position")
            
            monitor.progress = 1.0
            monitor.current_waypoint = monitor.total_waypoints
            
            # Task completed successfully
            self._publish_feedback(
                goal_handle,
                monitor.task_id,
                1.0,
                "Task completed successfully"
            )
            
            return PickPlaceResult(
                success=True,
                error_message="",
                execution_time=monitor.execution_time,
                actual_grasp_pose=plan.grasp_candidate.grasp_pose,
                actual_place_pose=plan.place_candidate.place_pose if plan.place_candidate else None
            )
            
        except Exception as e:
            # Handle execution failure
            self.get_logger().error(f"Task execution failed: {str(e)}")
            
            # Try recovery
            recovery_success = self._attempt_recovery(
                plan, 
                monitor, 
                str(e)
            )
            
            if recovery_success:
                # Continue execution from recovery point
                return self._execute_task(goal_handle, plan, monitor)
            else:
                # Recovery failed
                return PickPlaceResult(
                    success=False,
                    error_message=str(e),
                    execution_time=monitor.execution_time
                )
    
    def _execute_waypoint(self, waypoint: PoseStamped,
                        monitor: ExecutionMonitor) -> bool:
        """
        Execute movement to a waypoint with monitoring.
        """
        # In production, this would:
        # 1. Send trajectory to robot controller
        # 2. Monitor execution
        # 3. Check for errors
        # 4. Handle timeouts
        
        # For now, simulate execution
        time.sleep(0.5)  # Simulate movement time
        
        # Update monitor
        monitor.execution_time += 0.5
        
        # Check for errors (simulated)
        if random.random() < 0.05:  # 5% chance of simulated failure
            monitor.error_count += 1
            monitor.errors.append("Waypoint execution failed")
            return False
        
        return True
    
    def _execute_grasp(self, grasp: GraspCandidate,
                     object_id: str,
                     monitor: ExecutionMonitor) -> bool:
        """
        Execute grasp operation.
        
        In production, this would:
        1. Close gripper with appropriate force
        2. Verify grasp success with force/torque sensors
        3. Check object pickup with perception
        4. Handle grasp failures
        """
        try:
            # Simulate grasp execution
            time.sleep(0.3)
            
            # Update monitor
            monitor.execution_time += 0.3
            
            # Check grasp success (simulated)
            grasp_success = random.random() > 0.1  # 90% success rate
            
            if grasp_success:
                monitor.force_readings.append(grasp.force_required)
                self.get_logger().info(f"Grasp successful for object {object_id}")
                return True
            else:
                monitor.error_count += 1
                monitor.errors.append("Grasp failed - object not acquired")
                return False
                
        except Exception as e:
            monitor.error_count += 1
            monitor.errors.append(f"Grasp execution error: {str(e)}")
            return False
    
    def _attempt_recovery(self, plan: TaskPlan,
                        monitor: ExecutionMonitor,
                        error_message: str) -> bool:
        """
        Attempt to recover from execution failure.
        """
        recovery_attempts = monitor.recovery_attempts
        max_recovery_attempts = 3
        
        if recovery_attempts >= max_recovery_attempts:
            self.get_logger().warning(
                f"Max recovery attempts ({max_recovery_attempts}) reached"
            )
            return False
        
        monitor.recovery_attempts += 1
        
        # Select recovery strategy based on error
        recovery_strategy = self._select_recovery_strategy(error_message)
        
        self.get_logger().info(
            f"Attempting recovery {recovery_attempts + 1} with strategy: {recovery_strategy}"
        )
        
        try:
            if recovery_strategy in self.recovery_strategies:
                success = self.recovery_strategies[recovery_strategy](
                    plan, monitor, error_message
                )
                
                if success:
                    self.get_logger().info(f"Recovery {recovery_strategy} succeeded")
                    return True
                else:
                    self.get_logger().warning(f"Recovery {recovery_strategy} failed")
                    return False
            else:
                self.get_logger().warning(
                    f"Unknown recovery strategy: {recovery_strategy}"
                )
                return False
                
        except Exception as e:
            self.get_logger().error(f"Recovery attempt failed: {str(e)}")
            return False
    
    def _select_recovery_strategy(self, error_message: str) -> str:
        """
        Select appropriate recovery strategy based on error.
        """
        error_lower = error_message.lower()
        
        if "grasp" in error_lower or "slip" in error_lower:
            return "regrasp"
        elif "collision" in error_lower or "obstacle" in error_lower:
            return "replan"
        elif "ik" in error_lower or "unreachable" in error_lower:
            return "reorient"
        elif "timeout" in error_lower or "stall" in error_lower:
            return "retry"
        elif "emergency" in error_lower or "safety" in error_lower:
            return "emergency_stop"
        else:
            return "retry"  # Default recovery
    
    def _recovery_retry(self, plan: TaskPlan,
                      monitor: ExecutionMonitor,
                      error_message: str) -> bool:
        """
        Simple retry recovery strategy.
        """
        # Wait and retry
        time.sleep(1.0)
        
        # Clear errors
        monitor.errors.clear()
        
        return True
    
    def _recovery_regrasp(self, plan: TaskPlan,
                        monitor: ExecutionMonitor,
                        error_message: str) -> bool:
        """
        Regrasp recovery strategy.
        """
        try:
            # Move to retreat position
            success = self._execute_waypoint(
                plan.grasp_candidate.retreat_pose,
                monitor
            )
            
            if not success:
                return False
            
            # Select alternative grasp
            alternative_grasp = self._select_alternative_grasp(
                plan.grasp_candidate,
                plan.object_id
            )
            
            if not alternative_grasp:
                return False
            
            # Update plan with alternative grasp
            plan.grasp_candidate = alternative_grasp
            
            return True
            
        except Exception as e:
            self.get_logger().error(f"Regrasp recovery failed: {str(e)}")
            return False
    
    def _monitoring_loop(self):
        """
        Main monitoring loop for task execution.
        
        Monitors:
        1. Task progress
        2. Force/torque readings
        3. Position errors
        4. Velocity limits
        5. Safety violations
        6. System health
        """
        while True:
            try:
                with self.lock:
                    active_tasks = list(self.active_tasks.values())
                
                for monitor in active_tasks:
                    if monitor.current_state == TaskState.EXECUTING:
                        # Check for safety violations
                        self._check_safety_violations(monitor)
                        
                        # Check for performance issues
                        self._check_performance_issues(monitor)
                        
                        # Update progress estimate
                        self._update_progress_estimate(monitor)
                
                # Sleep to control loop frequency
                time.sleep(0.1)
                
            except Exception as e:
                self.get_logger().error(f"Monitoring loop error: {str(e)}")
                time.sleep(1.0)
    
    def _check_safety_violations(self, monitor: ExecutionMonitor):
        """
        Check for safety violations during execution.
        """
        # Check force limits
        force_threshold = self.get_parameter('force_torque_threshold').value
        if monitor.force_readings:
            recent_force = monitor.force_readings[-1]
            if recent_force > force_threshold:
                monitor.warnings.append(f"Force threshold exceeded: {recent_force}N")
        
        # Check velocity limits
        velocity_threshold = self.get_parameter('velocity_threshold').value
        if monitor.velocity_errors:
            recent_velocity = abs(monitor.velocity_errors[-1])
            if recent_velocity > velocity_threshold:
                monitor.warnings.append(f"Velocity threshold exceeded: {recent_velocity}m/s")
        
        # Check emergency stop conditions
        emergency_threshold = self.get_parameter('emergency_stop_threshold').value
        if monitor.force_readings and monitor.force_readings[-1] > emergency_threshold:
            monitor.errors.append("EMERGENCY: Excessive force detected")
            monitor.current_state = TaskState.FAILED
    
    def _publish_feedback(self, goal_handle, task_id: str,
                        progress: float, message: str):
        """
        Publish feedback for action.
        """
        feedback = PickPlaceFeedback()
        feedback.task_id = task_id
        feedback.progress = progress
        feedback.message = message
        feedback.timestamp = self.get_clock().now().to_msg()
        
        goal_handle.publish_feedback(feedback)
    
    def _publish_heartbeat(self):
        """
        Publish heartbeat message.
        """
        msg = String()
        msg.data = json.dumps({
            'node': 'pick_place_planner',
            'timestamp': datetime.now().isoformat(),
            'state': self.current_state.name,
            'active_tasks': len(self.active_tasks),
            'queue_size': len(self.task_queue)
        })
        
        self.status_pub.publish(msg)
    
    def _publish_status(self):
        """
        Publish status information.
        """
        with self.lock:
            status = {
                'state': self.current_state.name,
                'active_tasks': [
                    {
                        'task_id': t.task_id,
                        'progress': t.progress,
                        'state': t.current_state.name,
                        'errors': len(t.errors)
                    }
                    for t in self.active_tasks.values()
                ],
                'queue_size': len(self.task_queue),
                'timestamp': datetime.now().isoformat()
            }
        
        msg = String()
        msg.data = json.dumps(status)
        self.status_pub.publish(msg)
    
    def _publish_visualization(self):
        """
        Publish visualization markers.
        """
        if not self.get_parameter('debug_visualization').value:
            return
        
        markers = MarkerArray()
        
        # Visualize grasp candidates
        for grasp_id, grasp_list in self.grasp_database.items():
            for i, grasp in enumerate(grasp_list):
                marker = self._create_grasp_marker(grasp, i)
                markers.markers.append(marker)
        
        # Visualize planning scene
        if self.planning_scene:
            scene_markers = self._create_scene_markers()
            markers.markers.extend(scene_markers)
        
        # Publish markers
        if markers.markers:
            self.visualization_pub.publish(markers)
    
    def _create_grasp_marker(self, grasp: GraspCandidate,
                           marker_id: int) -> Marker:
        """
        Create visualization marker for a grasp.
        """
        marker = Marker()
        marker.header = grasp.grasp_pose.header
        marker.ns = "grasps"
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        
        # Set pose
        marker.pose = grasp.grasp_pose.pose
        
        # Set scale
        marker.scale.x = 0.1  # Length
        marker.scale.y = 0.01  # Width
        marker.scale.z = 0.01  # Height
        
        # Set color based on quality
        quality = grasp.quality_score
        if quality > 0.8:
            # Green - good quality
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
        elif quality > 0.6:
            # Yellow - medium quality
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
        else:
            # Red - poor quality
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
        
        marker.color.a = 0.8  # Transparency
        marker.lifetime = rclpy.duration.Duration(seconds=2.0).to_msg()
        
        return marker
    
    def _log_event(self, event_type: str, event_name: str, data: Dict):
        """
        Log important events.
        """
        event = {
            'type': event_type,
            'name': event_name,
            'data': data,
            'timestamp': datetime.now().isoformat()
        }
        
        msg = String()
        msg.data = json.dumps(event)
        self.event_pub.publish(msg)
        
        # Also log to console
        self.get_logger().info(f"Event: {event_type}.{event_name}")
    
    def _generate_task_id(self) -> str:
        """
        Generate unique task ID.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        random_str = hashlib.md5(str(time.time()).encode()).hexdigest()[:6]
        return f"task_{timestamp}_{random_str}"
    
    # Additional helper methods would be implemented here
    # These would include:
    # - IK/FK computation
    # - Collision checking
    # - Force closure analysis
    # - Stability calculation
    # - Experience recording and retrieval
    # - Database management
    # - Error handling and recovery
    # - Performance monitoring
    # - Safety checking
    
    def shutdown(self):
        """
        Clean shutdown of planner.
        """
        self.get_logger().info("Shutting down pick-place planner...")
        
        # Save experience database
        self._save_experience_database()
        
        # Cancel all active tasks
        with self.lock:
            for task_id, monitor in self.active_tasks.items():
                monitor.current_state = TaskState.CANCELLED
                monitor.errors.append("System shutdown")
        
        self.get_logger().info("Pick-place planner shutdown complete")


def main(args=None):
    """
    Main entry point for pick-place planner.
    """
    rclpy.init(args=args)
    
    try:
        planner = PickPlacePlanner()
        executor = MultiThreadedExecutor()
        executor.add_node(planner)
        
        planner.get_logger().info("PickPlacePlanner started")
        
        executor.spin()
        
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Fatal error in pick-place planner: {str(e)}")
    finally:
        if 'planner' in locals():
            planner.shutdown()
            planner.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
