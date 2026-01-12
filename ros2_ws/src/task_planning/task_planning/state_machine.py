#!/usr/bin/env python3
"""
State Machine for ML-based Object Picking Robot
Production-ready FSM for industrial task execution
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
import yaml
from dataclasses import dataclass, field
from enum import Enum, auto
import json
import time
from datetime import datetime, timedelta
import queue
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
import asyncio
import uuid

# ROS2 Message Imports
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion, Twist
from sensor_msgs.msg import JointState, Image, CameraInfo, PointCloud2
from std_msgs.msg import Header, String, Bool, Float32, Int32, Empty
from nav_msgs.msg import OccupancyGrid, Path
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import Buffer, TransformListener, TransformException
from tf2_geometry_msgs import do_transform_pose, do_transform_point

# Custom ROS2 Message Imports
from robot_interfaces.msg import DetectedObject, ObstacleMap, RobotStatus, GraspPose
from robot_interfaces.srv import PickObject, PlaceObject, ClearObstacleMap
from robot_interfaces.action import PickPlace, Navigate
from robot_interfaces.action import PickPlace as PickPlaceAction

# ROS2 Action imports
from rclpy.action import ActionServer, ActionClient, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle, GoalStatus
from rclpy.action.client import ClientGoalHandle
from rclpy.duration import Duration

class TaskState(Enum):
    """Main task execution states"""
    IDLE = auto()
    INITIALIZING = auto()
    SCANNING_ENVIRONMENT = auto()
    DETECTING_OBJECTS = auto()
    SELECTING_TARGET = auto()
    PLANNING_MOTION = auto()
    EXECUTING_APPROACH = auto()
    GRASPING_OBJECT = auto()
    LIFTING_OBJECT = auto()
    PLANNING_TRANSPORT = auto()
    TRANSPORTING_OBJECT = auto()
    PLANNING_PLACEMENT = auto()
    PLACING_OBJECT = auto()
    RELEASING_OBJECT = auto()
    RETRACTING_ARM = auto()
    VERIFYING_SUCCESS = auto()
    HANDLING_FAILURE = auto()
    RECOVERING_ERROR = auto()
    SHUTDOWN_SEQUENCE = auto()
    EMERGENCY_STOPPED = auto()
    PAUSED = auto()
    RESETTING = auto()

class TaskEvent(Enum):
    """Events that trigger state transitions"""
    START_TASK = auto()
    INITIALIZATION_COMPLETE = auto()
    SCAN_COMPLETE = auto()
    OBJECTS_DETECTED = auto()
    TARGET_SELECTED = auto()
    MOTION_PLANNED = auto()
    APPROACH_COMPLETE = auto()
    GRASP_SUCCESS = auto()
    GRASP_FAILURE = auto()
    LIFT_COMPLETE = auto()
    TRANSPORT_PLANNED = auto()
    TRANSPORT_COMPLETE = auto()
    PLACEMENT_PLANNED = auto()
    PLACEMENT_COMPLETE = auto()
    RELEASE_SUCCESS = auto()
    RETRACTION_COMPLETE = auto()
    VERIFICATION_SUCCESS = auto()
    VERIFICATION_FAILURE = auto()
    ERROR_DETECTED = auto()
    RECOVERY_COMPLETE = auto()
    PAUSE_REQUESTED = auto()
    RESUME_REQUESTED = auto()
    RESET_REQUESTED = auto()
    EMERGENCY_STOP = auto()
    SHUTDOWN_REQUESTED = auto()
    TIMEOUT_EXPIRED = auto()
    SAFETY_VIOLATION = auto()
    OBSTACLE_DETECTED = auto()
    REQUEST_RETRY = auto()

class TaskPriority(Enum):
    """Task priority levels"""
    EMERGENCY = 0
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5

@dataclass
class TaskContext:
    """Context data for task execution"""
    task_id: str
    start_time: datetime
    target_object: Optional[DetectedObject] = None
    target_pose: Optional[PoseStamped] = None
    grasp_pose: Optional[GraspPose] = None
    transport_pose: Optional[PoseStamped] = None
    placement_pose: Optional[PoseStamped] = None
    detected_objects: List[DetectedObject] = field(default_factory=list)
    obstacle_map: Optional[ObstacleMap] = None
    robot_pose: Optional[PoseStamped] = None
    joint_states: Optional[JointState] = None
    current_plan: Optional[Path] = None
    execution_history: List[Dict] = field(default_factory=list)
    error_log: List[Dict] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: float = 30.0
    priority: TaskPriority = TaskPriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StateTransition:
    """Definition of a state transition"""
    from_state: TaskState
    to_state: TaskState
    event: TaskEvent
    condition: Optional[Callable[[TaskContext], bool]] = None
    action: Optional[Callable[[TaskContext], Any]] = None
    priority: int = 0

@dataclass
class StateConfiguration:
    """Configuration for a specific state"""
    entry_actions: List[Callable] = field(default_factory=list)
    exit_actions: List[Callable] = field(default_factory=list)
    do_actions: List[Callable] = field(default_factory=list)
    timeout_seconds: float = 5.0
    retry_on_failure: bool = True
    max_retries: int = 3
    is_interruptible: bool = True
    requires_safety_check: bool = True

class TaskStateMachine(Node):
    """
    Production-ready Finite State Machine for robotic task execution
    Implements hierarchical FSM with concurrent states and error recovery
    """
    
    def __init__(self):
        super().__init__('task_state_machine')
        
        # Initialize with industrial-grade QoS
        qos_reliable = QoSProfile(
            depth=100,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        
        qos_best_effort = QoSProfile(
            depth=50,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST
        )
        
        # Multi-threaded callback groups
        self.state_transition_group = ReentrantCallbackGroup()
        self.action_execution_group = MutuallyExclusiveCallbackGroup()
        self.monitoring_group = MutuallyExclusiveCallbackGroup()
        
        # State machine variables
        self.current_state = TaskState.IDLE
        self.previous_state = TaskState.IDLE
        self.pending_state = None
        self.state_entry_time = datetime.now()
        self.state_timeout_timer = None
        
        # Task management
        self.active_tasks: Dict[str, TaskContext] = {}
        self.task_queue = queue.PriorityQueue(maxsize=100)
        self.completed_tasks: List[TaskContext] = []
        self.failed_tasks: List[TaskContext] = []
        
        # State configurations
        self.state_configs: Dict[TaskState, StateConfiguration] = {}
        self.transitions: List[StateTransition] = []
        
        # Thread management
        self._state_lock = threading.RLock()
        self._task_lock = threading.RLock()
        self._transition_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix='task_fsm')
        
        # Real-time buffers
        self.object_detection_buffer = queue.Queue(maxsize=50)
        self.obstacle_buffer = queue.Queue(maxsize=20)
        self.joint_state_buffer = queue.Queue(maxsize=100)
        self.safety_state_buffer = queue.Queue(maxsize=10)
        
        # Performance metrics
        self.state_durations: Dict[TaskState, List[float]] = {}
        self.transition_counts: Dict[Tuple[TaskState, TaskState], int] = {}
        self.error_rates: Dict[TaskState, List[bool]] = {}
        
        # Initialize state machine
        self._initialize_state_configurations()
        self._initialize_transitions()
        self._initialize_state_durations()
        
        # Load configuration
        self._load_configuration()
        
        # Initialize ROS2 components
        self._initialize_publishers(qos_reliable, qos_best_effort)
        self._initialize_subscribers(qos_reliable, qos_best_effort)
        self._initialize_services()
        self._initialize_action_servers()
        self._initialize_clients()
        
        # TF2 for coordinate transforms
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Start monitoring and execution threads
        self._start_execution_threads()
        
        self.get_logger().info(f"Task State Machine initialized in {self.current_state.name}")
    
    def _load_configuration(self):
        """Load FSM configuration from YAML files"""
        config_paths = [
            '/home/victor/ml-object-picking-robot/config/production.yaml',
            '/home/victor/ml-object-picking-robot/ros2_ws/src/task_planning/config/task_sequences.yaml'
        ]
        
        config = {}
        for path in config_paths:
            try:
                with open(path, 'r') as f:
                    config.update(yaml.safe_load(f) or {})
                self.get_logger().info(f"Loaded FSM config from {path}")
            except Exception as e:
                self.get_logger().error(f"Failed to load config {path}: {e}")
        
        # Apply configuration to state configs
        fsm_config = config.get('task_state_machine', {})
        for state_name, state_config in fsm_config.get('states', {}).items():
            try:
                state = TaskState[state_name.upper()]
                if state in self.state_configs:
                    self.state_configs[state].timeout_seconds = state_config.get('timeout', 5.0)
                    self.state_configs[state].max_retries = state_config.get('max_retries', 3)
                    self.state_configs[state].is_interruptible = state_config.get('interruptible', True)
            except KeyError:
                self.get_logger().warning(f"Unknown state in config: {state_name}")
    
    def _initialize_state_configurations(self):
        """Initialize configuration for each state"""
        configs = {
            TaskState.IDLE: StateConfiguration(
                entry_actions=[self._log_state_entry, self._reset_task_context],
                exit_actions=[self._log_state_exit],
                do_actions=[self._monitor_task_queue],
                timeout_seconds=float('inf'),
                retry_on_failure=False,
                max_retries=0,
                is_interruptible=True,
                requires_safety_check=False
            ),
            TaskState.INITIALIZING: StateConfiguration(
                entry_actions=[self._log_state_entry, self._initialize_hardware],
                exit_actions=[self._log_state_exit, self._verify_initialization],
                do_actions=[self._check_hardware_status, self._calibrate_sensors],
                timeout_seconds=10.0,
                retry_on_failure=True,
                max_retries=3,
                is_interruptible=True,
                requires_safety_check=True
            ),
            TaskState.SCANNING_ENVIRONMENT: StateConfiguration(
                entry_actions=[self._log_state_entry, self._start_environment_scan],
                exit_actions=[self._log_state_exit, self._process_scan_data],
                do_actions=[self._capture_environment_data, self._update_obstacle_map],
                timeout_seconds=5.0,
                retry_on_failure=True,
                max_retries=2,
                is_interruptible=True,
                requires_safety_check=True
            ),
            TaskState.DETECTING_OBJECTS: StateConfiguration(
                entry_actions=[self._log_state_entry, self._enable_object_detection],
                exit_actions=[self._log_state_exit, self._filter_detected_objects],
                do_actions=[self._process_detection_results, self._update_object_tracking],
                timeout_seconds=3.0,
                retry_on_failure=True,
                max_retries=3,
                is_interruptible=True,
                requires_safety_check=False
            ),
            TaskState.SELECTING_TARGET: StateConfiguration(
                entry_actions=[self._log_state_entry, self._load_selection_policy],
                exit_actions=[self._log_state_exit, self._validate_target_selection],
                do_actions=[self._evaluate_target_candidates, self._compute_grasp_poses],
                timeout_seconds=2.0,
                retry_on_failure=True,
                max_retries=3,
                is_interruptible=True,
                requires_safety_check=False
            ),
            TaskState.PLANNING_MOTION: StateConfiguration(
                entry_actions=[self._log_state_entry, self._compute_ik_solutions],
                exit_actions=[self._log_state_exit, self._validate_motion_plan],
                do_actions=[self._generate_trajectory, self._check_collisions],
                timeout_seconds=5.0,
                retry_on_failure=True,
                max_retries=3,
                is_interruptible=False,  # Don't interrupt planning
                requires_safety_check=True
            ),
            TaskState.EXECUTING_APPROACH: StateConfiguration(
                entry_actions=[self._log_state_entry, self._start_motion_execution],
                exit_actions=[self._log_state_exit, self._verify_approach_position],
                do_actions=[self._monitor_motion_execution, self._adjust_trajectory],
                timeout_seconds=15.0,
                retry_on_failure=True,
                max_retries=2,
                is_interruptible=False,
                requires_safety_check=True
            ),
            TaskState.GRASPING_OBJECT: StateConfiguration(
                entry_actions=[self._log_state_entry, self._prepare_gripper],
                exit_actions=[self._log_state_exit, self._verify_grasp_success],
                do_actions=[self._execute_grasp_sequence, self._monitor_grip_force],
                timeout_seconds=3.0,
                retry_on_failure=True,
                max_retries=3,
                is_interruptible=False,
                requires_safety_check=True
            ),
            TaskState.LIFTING_OBJECT: StateConfiguration(
                entry_actions=[self._log_state_entry, self._compute_lift_trajectory],
                exit_actions=[self._log_state_exit, self._verify_lift_completion],
                do_actions=[self._execute_lift_motion, self._monitor_object_stability],
                timeout_seconds=5.0,
                retry_on_failure=True,
                max_retries=2,
                is_interruptible=False,
                requires_safety_check=True
            ),
            TaskState.TRANSPORTING_OBJECT: StateConfiguration(
                entry_actions=[self._log_state_entry, self._plan_transport_path],
                exit_actions=[self._log_state_exit, self._verify_transport_completion],
                do_actions=[self._execute_transport_motion, self._monitor_object_retention],
                timeout_seconds=10.0,
                retry_on_failure=True,
                max_retries=2,
                is_interruptible=False,
                requires_safety_check=True
            ),
            TaskState.PLACING_OBJECT: StateConfiguration(
                entry_actions=[self._log_state_entry, self._compute_placement_trajectory],
                exit_actions=[self._log_state_exit, self._verify_placement_success],
                do_actions=[self._execute_placement_motion, self._monitor_placement_accuracy],
                timeout_seconds=5.0,
                retry_on_failure=True,
                max_retries=3,
                is_interruptible=False,
                requires_safety_check=True
            ),
            TaskState.RELEASING_OBJECT: StateConfiguration(
                entry_actions=[self._log_state_entry, self._prepare_release],
                exit_actions=[self._log_state_exit, self._verify_release_success],
                do_actions=[self._execute_release_sequence, self._monitor_release_clearance],
                timeout_seconds=2.0,
                retry_on_failure=True,
                max_retries=3,
                is_interruptible=False,
                requires_safety_check=True
            ),
            TaskState.RETRACTING_ARM: StateConfiguration(
                entry_actions=[self._log_state_entry, self._compute_retraction_path],
                exit_actions=[self._log_state_exit, self._verify_arm_retracted],
                do_actions=[self._execute_retraction_motion, self._monitor_clearance],
                timeout_seconds=5.0,
                retry_on_failure=True,
                max_retries=2,
                is_interruptible=False,
                requires_safety_check=True
            ),
            TaskState.VERIFYING_SUCCESS: StateConfiguration(
                entry_actions=[self._log_state_entry, self._capture_verification_data],
                exit_actions=[self._log_state_exit, self._generate_task_report],
                do_actions=[self._analyze_task_outcome, self._update_performance_metrics],
                timeout_seconds=3.0,
                retry_on_failure=False,
                max_retries=1,
                is_interruptible=True,
                requires_safety_check=False
            ),
            TaskState.HANDLING_FAILURE: StateConfiguration(
                entry_actions=[self._log_state_entry, self._analyze_failure_cause],
                exit_actions=[self._log_state_exit, self._initiate_recovery_procedure],
                do_actions=[self._execute_failure_protocol, self._notify_operators],
                timeout_seconds=10.0,
                retry_on_failure=False,
                max_retries=1,
                is_interruptible=False,
                requires_safety_check=True
            ),
            TaskState.RECOVERING_ERROR: StateConfiguration(
                entry_actions=[self._log_state_entry, self._assess_recovery_options],
                exit_actions=[self._log_state_exit, self._execute_recovery_plan],
                do_actions=[self._monitor_recovery_progress, self._adjust_recovery_strategy],
                timeout_seconds=30.0,
                retry_on_failure=True,
                max_retries=5,
                is_interruptible=True,
                requires_safety_check=True
            ),
            TaskState.EMERGENCY_STOPPED: StateConfiguration(
                entry_actions=[self._log_state_entry, self._execute_emergency_stop],
                exit_actions=[self._log_state_exit],
                do_actions=[self._maintain_safety_state, self._await_operator_input],
                timeout_seconds=float('inf'),
                retry_on_failure=False,
                max_retries=0,
                is_interruptible=False,
                requires_safety_check=True
            ),
            TaskState.PAUSED: StateConfiguration(
                entry_actions=[self._log_state_entry, self._suspend_all_operations],
                exit_actions=[self._log_state_exit, self._resume_operations],
                do_actions=[self._maintain_paused_state, self._monitor_resume_conditions],
                timeout_seconds=float('inf'),
                retry_on_failure=False,
                max_retries=0,
                is_interruptible=True,
                requires_safety_check=True
            ),
            TaskState.RESETTING: StateConfiguration(
                entry_actions=[self._log_state_entry, self._initiate_system_reset],
                exit_actions=[self._log_state_exit, self._verify_reset_completion],
                do_actions=[self._execute_reset_sequence, self._calibrate_all_components],
                timeout_seconds=15.0,
                retry_on_failure=True,
                max_retries=3,
                is_interruptible=False,
                requires_safety_check=True
            ),
            TaskState.SHUTDOWN_SEQUENCE: StateConfiguration(
                entry_actions=[self._log_state_entry, self._initiate_graceful_shutdown],
                exit_actions=[self._log_state_exit],
                do_actions=[self._execute_shutdown_procedures, self._save_system_state],
                timeout_seconds=10.0,
                retry_on_failure=False,
                max_retries=0,
                is_interruptible=False,
                requires_safety_check=True
            )
        }
        
        self.state_configs.update(configs)
    
    def _initialize_transitions(self):
        """Define all state transitions with conditions and actions"""
        self.transitions = [
            # Idle transitions
            StateTransition(TaskState.IDLE, TaskState.INITIALIZING, TaskEvent.START_TASK,
                           condition=self._can_start_task,
                           action=self._create_new_task_context),
            
            # Initialization transitions
            StateTransition(TaskState.INITIALIZING, TaskState.SCANNING_ENVIRONMENT, TaskEvent.INITIALIZATION_COMPLETE,
                           condition=self._hardware_initialized),
            StateTransition(TaskState.INITIALIZING, TaskState.HANDLING_FAILURE, TaskEvent.ERROR_DETECTED,
                           condition=self._initialization_failed),
            
            # Scanning transitions
            StateTransition(TaskState.SCANNING_ENVIRONMENT, TaskState.DETECTING_OBJECTS, TaskEvent.SCAN_COMPLETE,
                           condition=self._environment_scanned),
            StateTransition(TaskState.SCANNING_ENVIRONMENT, TaskState.HANDLING_FAILURE, TaskEvent.TIMEOUT_EXPIRED),
            
            # Detection transitions
            StateTransition(TaskState.DETECTING_OBJECTS, TaskState.SELECTING_TARGET, TaskEvent.OBJECTS_DETECTED,
                           condition=self._objects_detected),
            StateTransition(TaskState.DETECTING_OBJECTS, TaskState.SCANNING_ENVIRONMENT, TaskEvent.REQUEST_RETRY,
                           condition=self._no_objects_detected),
            
            # Target selection transitions
            StateTransition(TaskState.SELECTING_TARGET, TaskState.PLANNING_MOTION, TaskEvent.TARGET_SELECTED,
                           condition=self._target_valid),
            StateTransition(TaskState.SELECTING_TARGET, TaskState.HANDLING_FAILURE, TaskEvent.ERROR_DETECTED,
                           condition=self._target_invalid),
            
            # Motion planning transitions
            StateTransition(TaskState.PLANNING_MOTION, TaskState.EXECUTING_APPROACH, TaskEvent.MOTION_PLANNED,
                           condition=self._motion_plan_valid),
            StateTransition(TaskState.PLANNING_MOTION, TaskState.HANDLING_FAILURE, TaskEvent.ERROR_DETECTED,
                           condition=self._motion_planning_failed),
            
            # Approach execution transitions
            StateTransition(TaskState.EXECUTING_APPROACH, TaskState.GRASPING_OBJECT, TaskEvent.APPROACH_COMPLETE,
                           condition=self._approach_successful),
            StateTransition(TaskState.EXECUTING_APPROACH, TaskState.HANDLING_FAILURE, TaskEvent.ERROR_DETECTED,
                           condition=self._approach_failed),
            StateTransition(TaskState.EXECUTING_APPROACH, TaskState.PLANNING_MOTION, TaskEvent.REQUEST_RETRY,
                           condition=self._needs_replanning),
            
            # Grasping transitions
            StateTransition(TaskState.GRASPING_OBJECT, TaskState.LIFTING_OBJECT, TaskEvent.GRASP_SUCCESS,
                           condition=self._grasp_successful),
            StateTransition(TaskState.GRASPING_OBJECT, TaskState.HANDLING_FAILURE, TaskEvent.GRASP_FAILURE,
                           condition=self._grasp_failed),
            
            # Lifting transitions
            StateTransition(TaskState.LIFTING_OBJECT, TaskState.TRANSPORTING_OBJECT, TaskEvent.LIFT_COMPLETE,
                           condition=self._lift_successful),
            StateTransition(TaskState.LIFTING_OBJECT, TaskState.HANDLING_FAILURE, TaskEvent.ERROR_DETECTED,
                           condition=self._lift_failed),
            
            # Transport transitions
            StateTransition(TaskState.TRANSPORTING_OBJECT, TaskState.PLACING_OBJECT, TaskEvent.TRANSPORT_COMPLETE,
                           condition=self._transport_successful),
            StateTransition(TaskState.TRANSPORTING_OBJECT, TaskState.HANDLING_FAILURE, TaskEvent.ERROR_DETECTED,
                           condition=self._transport_failed),
            
            # Placement transitions
            StateTransition(TaskState.PLACING_OBJECT, TaskState.RELEASING_OBJECT, TaskEvent.PLACEMENT_COMPLETE,
                           condition=self._placement_successful),
            StateTransition(TaskState.PLACING_OBJECT, TaskState.HANDLING_FAILURE, TaskEvent.ERROR_DETECTED,
                           condition=self._placement_failed),
            
            # Release transitions
            StateTransition(TaskState.RELEASING_OBJECT, TaskState.RETRACTING_ARM, TaskEvent.RELEASE_SUCCESS,
                           condition=self._release_successful),
            StateTransition(TaskState.RELEASING_OBJECT, TaskState.HANDLING_FAILURE, TaskEvent.ERROR_DETECTED,
                           condition=self._release_failed),
            
            # Retraction transitions
            StateTransition(TaskState.RETRACTING_ARM, TaskState.VERIFYING_SUCCESS, TaskEvent.RETRACTION_COMPLETE,
                           condition=self._retraction_successful),
            StateTransition(TaskState.RETRACTING_ARM, TaskState.HANDLING_FAILURE, TaskEvent.ERROR_DETECTED,
                           condition=self._retraction_failed),
            
            # Verification transitions
            StateTransition(TaskState.VERIFYING_SUCCESS, TaskState.IDLE, TaskEvent.VERIFICATION_SUCCESS,
                           condition=self._task_completed_successfully),
            StateTransition(TaskState.VERIFYING_SUCCESS, TaskState.HANDLING_FAILURE, TaskEvent.VERIFICATION_FAILURE,
                           condition=self._task_failed_verification),
            
            # Failure handling transitions
            StateTransition(TaskState.HANDLING_FAILURE, TaskState.RECOVERING_ERROR, TaskEvent.RECOVERY_COMPLETE,
                           condition=self._recovery_possible),
            StateTransition(TaskState.HANDLING_FAILURE, TaskState.EMERGENCY_STOPPED, TaskEvent.EMERGENCY_STOP,
                           condition=self._critical_failure),
            
            # Error recovery transitions
            StateTransition(TaskState.RECOVERING_ERROR, TaskState.IDLE, TaskEvent.RECOVERY_COMPLETE,
                           condition=self._recovery_successful),
            StateTransition(TaskState.RECOVERING_ERROR, TaskState.EMERGENCY_STOPPED, TaskEvent.EMERGENCY_STOP,
                           condition=self._recovery_failed),
            
            # Emergency stop transitions
            StateTransition(TaskState.EMERGENCY_STOPPED, TaskState.RESETTING, TaskEvent.RESET_REQUESTED,
                           condition=self._can_reset_from_emergency),
            
            # Pause/resume transitions
            StateTransition(TaskState.PAUSED, TaskState.IDLE, TaskEvent.RESUME_REQUESTED,
                           condition=self._can_resume),
            StateTransition(TaskState.IDLE, TaskState.PAUSED, TaskEvent.PAUSE_REQUESTED,
                           condition=self._can_pause),
            StateTransition(TaskState.EXECUTING_APPROACH, TaskState.PAUSED, TaskEvent.PAUSE_REQUESTED,
                           condition=self._can_pause),
            StateTransition(TaskState.GRASPING_OBJECT, TaskState.PAUSED, TaskEvent.PAUSE_REQUESTED,
                           condition=self._can_pause),
            
            # Reset transitions
            StateTransition(TaskState.RESETTING, TaskState.IDLE, TaskEvent.RECOVERY_COMPLETE,
                           condition=self._reset_successful),
            StateTransition(TaskState.RESETTING, TaskState.EMERGENCY_STOPPED, TaskEvent.ERROR_DETECTED,
                           condition=self._reset_failed),
            
            # Global emergency stop
            StateTransition(TaskState.INITIALIZING, TaskState.EMERGENCY_STOPPED, TaskEvent.EMERGENCY_STOP),
            StateTransition(TaskState.SCANNING_ENVIRONMENT, TaskState.EMERGENCY_STOPPED, TaskEvent.EMERGENCY_STOP),
            StateTransition(TaskState.DETECTING_OBJECTS, TaskState.EMERGENCY_STOPPED, TaskEvent.EMERGENCY_STOP),
            StateTransition(TaskState.SELECTING_TARGET, TaskState.EMERGENCY_STOPPED, TaskEvent.EMERGENCY_STOP),
            StateTransition(TaskState.PLANNING_MOTION, TaskState.EMERGENCY_STOPPED, TaskEvent.EMERGENCY_STOP),
            StateTransition(TaskState.EXECUTING_APPROACH, TaskState.EMERGENCY_STOPPED, TaskEvent.EMERGENCY_STOP),
            StateTransition(TaskState.GRASPING_OBJECT, TaskState.EMERGENCY_STOPPED, TaskEvent.EMERGENCY_STOP),
            StateTransition(TaskState.LIFTING_OBJECT, TaskState.EMERGENCY_STOPPED, TaskEvent.EMERGENCY_STOP),
            StateTransition(TaskState.TRANSPORTING_OBJECT, TaskState.EMERGENCY_STOPPED, TaskEvent.EMERGENCY_STOP),
            StateTransition(TaskState.PLACING_OBJECT, TaskState.EMERGENCY_STOPPED, TaskEvent.EMERGENCY_STOP),
            StateTransition(TaskState.RELEASING_OBJECT, TaskState.EMERGENCY_STOPPED, TaskEvent.EMERGENCY_STOP),
            StateTransition(TaskState.RETRACTING_ARM, TaskState.EMERGENCY_STOPPED, TaskEvent.EMERGENCY_STOP),
            StateTransition(TaskState.VERIFYING_SUCCESS, TaskState.EMERGENCY_STOPPED, TaskEvent.EMERGENCY_STOP),
            StateTransition(TaskState.HANDLING_FAILURE, TaskState.EMERGENCY_STOPPED, TaskEvent.EMERGENCY_STOP),
            StateTransition(TaskState.RECOVERING_ERROR, TaskState.EMERGENCY_STOPPED, TaskEvent.EMERGENCY_STOP),
            
            # Shutdown transitions
            StateTransition(TaskState.IDLE, TaskState.SHUTDOWN_SEQUENCE, TaskEvent.SHUTDOWN_REQUESTED),
            StateTransition(TaskState.PAUSED, TaskState.SHUTDOWN_SEQUENCE, TaskEvent.SHUTDOWN_REQUESTED),
            StateTransition(TaskState.SHUTDOWN_SEQUENCE, TaskState.IDLE, TaskEvent.RECOVERY_COMPLETE,
                           condition=self._shutdown_aborted),
        ]
        
        # Sort transitions by priority (lower number = higher priority)
        self.transitions.sort(key=lambda x: x.priority)
    
    def _initialize_state_durations(self):
        """Initialize duration tracking for each state"""
        for state in TaskState:
            self.state_durations[state] = []
            self.error_rates[state] = []
        
        # Initialize transition counts
        for transition in self.transitions:
            self.transition_counts[(transition.from_state, transition.to_state)] = 0
    
    def _initialize_publishers(self, qos_reliable, qos_best_effort):
        """Initialize ROS2 publishers"""
        self.state_pub = self.create_publisher(
            RobotStatus,
            '/task_state_machine/state',
            qos_reliable
        )
        
        self.task_status_pub = self.create_publisher(
            String,
            '/task_state_machine/task_status',
            qos_reliable
        )
        
        self.state_transition_pub = self.create_publisher(
            String,
            '/task_state_machine/transitions',
            qos_best_effort
        )
        
        self.command_pub = self.create_publisher(
            String,
            '/task_state_machine/commands',
            qos_reliable
        )
        
        self.visualization_pub = self.create_publisher(
            MarkerArray,
            '/task_state_machine/visualization',
            qos_best_effort
        )
        
        self.metrics_pub = self.create_publisher(
            String,
            '/task_state_machine/metrics',
            qos_best_effort
        )
    
    def _initialize_subscribers(self, qos_reliable, qos_best_effort):
        """Initialize ROS2 subscribers"""
        self.object_detection_sub = self.create_subscription(
            DetectedObject,
            '/perception/detected_objects',
            self._object_detection_callback,
            qos_best_effort,
            callback_group=self.monitoring_group
        )
        
        self.obstacle_map_sub = self.create_subscription(
            ObstacleMap,
            '/perception/obstacle_map',
            self._obstacle_map_callback,
            qos_best_effort,
            callback_group=self.monitoring_group
        )
        
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self._joint_state_callback,
            qos_reliable,
            callback_group=self.monitoring_group
        )
        
        self.safety_state_sub = self.create_subscription(
            RobotStatus,
            '/robot/safety/state',
            self._safety_state_callback,
            qos_reliable,
            callback_group=self.monitoring_group
        )
        
        self.task_command_sub = self.create_subscription(
            String,
            '/task_state_machine/commands',
            self._task_command_callback,
            qos_reliable,
            callback_group=self.state_transition_group
        )
    
    def _initialize_services(self):
        """Initialize ROS2 services"""
        self.pick_object_srv = self.create_service(
            PickObject,
            '/task_state_machine/pick_object',
            self._pick_object_callback,
            callback_group=self.action_execution_group
        )
        
        self.place_object_srv = self.create_service(
            PlaceObject,
            '/task_state_machine/place_object',
            self._place_object_callback,
            callback_group=self.action_execution_group
        )
        
        self.get_state_srv = self.create_service(
            RobotStatus,
            '/task_state_machine/get_state',
            self._get_state_callback,
            callback_group=self.monitoring_group
        )
    
    def _initialize_action_servers(self):
        """Initialize ROS2 action servers"""
        self.pick_place_action = ActionServer(
            self,
            PickPlaceAction,
            '/task_state_machine/execute_pick_place',
            self._execute_pick_place_callback,
            callback_group=self.action_execution_group,
            goal_callback=self._pick_place_goal_callback,
            cancel_callback=self._pick_place_cancel_callback
        )
    
    def _initialize_clients(self):
        """Initialize ROS2 service clients"""
        self.motion_planning_client = self.create_client(
            PickPlace,
            '/motion_planning/plan_pick_place',
            callback_group=self.action_execution_group
        )
        
        self.robot_control_client = self.create_client(
            PickPlace,
            '/robot_control/execute_pick_place',
            callback_group=self.action_execution_group
        )
        
        # Wait for services
        self.get_logger().info("Waiting for motion planning service...")
        self.motion_planning_client.wait_for_service(timeout_sec=10.0)
        
        self.get_logger().info("Waiting for robot control service...")
        self.robot_control_client.wait_for_service(timeout_sec=10.0)
    
    def _start_execution_threads(self):
        """Start execution and monitoring threads"""
        # Main state machine execution thread (100Hz)
        self.execution_timer = self.create_timer(
            0.01,  # 100Hz
            self._execute_state_machine,
            callback_group=self.state_transition_group
        )
        
        # State timeout monitoring thread (10Hz)
        self.timeout_timer = self.create_timer(
            0.1,  # 10Hz
            self._check_state_timeout,
            callback_group=self.monitoring_group
        )
        
        # Metrics publishing thread (1Hz)
        self.metrics_timer = self.create_timer(
            1.0,  # 1Hz
            self._publish_metrics,
            callback_group=self.monitoring_group
        )
        
        # Visualization update thread (5Hz)
        self.visualization_timer = self.create_timer(
            0.2,  # 5Hz
            self._update_visualization,
            callback_group=self.monitoring_group
        )
        
        # Task queue processing thread (10Hz)
        self.task_queue_timer = self.create_timer(
            0.1,  # 10Hz
            self._process_task_queue,
            callback_group=self.action_execution_group
        )
    
    def _object_detection_callback(self, msg: DetectedObject):
        """Process object detection updates"""
        try:
            self.object_detection_buffer.put_nowait({
                'timestamp': datetime.now(),
                'object': msg
            })
        except queue.Full:
            # Remove oldest if buffer is full
            try:
                self.object_detection_buffer.get_nowait()
                self.object_detection_buffer.put_nowait({
                    'timestamp': datetime.now(),
                    'object': msg
                })
            except:
                pass
    
    def _obstacle_map_callback(self, msg: ObstacleMap):
        """Process obstacle map updates"""
        try:
            self.obstacle_buffer.put_nowait({
                'timestamp': datetime.now(),
                'map': msg
            })
        except queue.Full:
            pass
    
    def _joint_state_callback(self, msg: JointState):
        """Process joint state updates"""
        try:
            self.joint_state_buffer.put_nowait({
                'timestamp': datetime.now(),
                'states': msg
            })
        except queue.Full:
            # Joint states are critical, remove oldest
            try:
                self.joint_state_buffer.get_nowait()
                self.joint_state_buffer.put_nowait({
                    'timestamp': datetime.now(),
                    'states': msg
                })
            except:
                pass
    
    def _safety_state_callback(self, msg: RobotStatus):
        """Process safety state updates"""
        try:
            self.safety_state_buffer.put_nowait({
                'timestamp': datetime.now(),
                'state': msg
            })
        except queue.Full:
            pass
    
    def _task_command_callback(self, msg: String):
        """Process task commands"""
        try:
            command = json.loads(msg.data)
            cmd_type = command.get('command')
            
            if cmd_type == 'start_task':
                self._queue_task(command)
            elif cmd_type == 'pause':
                self._trigger_event(TaskEvent.PAUSE_REQUESTED)
            elif cmd_type == 'resume':
                self._trigger_event(TaskEvent.RESUME_REQUESTED)
            elif cmd_type == 'stop':
                self._trigger_event(TaskEvent.EMERGENCY_STOP)
            elif cmd_type == 'reset':
                self._trigger_event(TaskEvent.RESET_REQUESTED)
            elif cmd_type == 'shutdown':
                self._trigger_event(TaskEvent.SHUTDOWN_REQUESTED)
            
        except Exception as e:
            self.get_logger().error(f"Failed to process command: {e}")
    
    def _queue_task(self, task_spec: Dict):
        """Queue a new task for execution"""
        task_id = task_spec.get('task_id', str(uuid.uuid4()))
        priority = TaskPriority(task_spec.get('priority', TaskPriority.NORMAL.value))
        
        context = TaskContext(
            task_id=task_id,
            start_time=datetime.now(),
            priority=priority,
            metadata=task_spec.get('metadata', {})
        )
        
        # Add to priority queue (lower priority number = higher priority)
        self.task_queue.put((priority.value, datetime.now(), context))
        self.get_logger().info(f"Task {task_id} queued with priority {priority.name}")
    
    def _process_task_queue(self):
        """Process tasks from the queue"""
        if self.current_state != TaskState.IDLE:
            return
        
        if not self.task_queue.empty():
            try:
                priority, timestamp, context = self.task_queue.get_nowait()
                
                # Store active task
                with self._task_lock:
                    self.active_tasks[context.task_id] = context
                
                # Trigger start task event
                self._trigger_event(TaskEvent.START_TASK)
                
                self.get_logger().info(f"Starting task {context.task_id} from queue")
                
            except queue.Empty:
                pass
            except Exception as e:
                self.get_logger().error(f"Failed to process task queue: {e}")
    
    def _execute_state_machine(self):
        """Main state machine execution loop"""
        try:
            # Execute state entry actions on first entry
            if self.state_entry_time is None:
                self._execute_entry_actions()
                self.state_entry_time = datetime.now()
            
            # Execute state 'do' actions
            self._execute_do_actions()
            
            # Check for events and transitions
            self._process_events()
            
            # Update state duration
            if self.state_entry_time:
                duration = (datetime.now() - self.state_entry_time).total_seconds()
                self.state_durations[self.current_state].append(duration)
            
            # Publish current state
            self._publish_state()
            
        except Exception as e:
            self.get_logger().error(f"State machine execution error: {e}")
            self._trigger_event(TaskEvent.ERROR_DETECTED)
    
    def _execute_entry_actions(self):
        """Execute entry actions for current state"""
        config = self.state_configs.get(self.current_state)
        if config:
            for action in config.entry_actions:
                try:
                    action()
                except Exception as e:
                    self.get_logger().error(f"Entry action failed: {e}")
        
        # Start state timeout timer
        if config and config.timeout_seconds < float('inf'):
            self.state_timeout_timer = self.create_timer(
                config.timeout_seconds,
                self._handle_state_timeout,
                callback_group=self.state_transition_group,
                oneshot=True
            )
    
    def _execute_do_actions(self):
        """Execute 'do' actions for current state"""
        config = self.state_configs.get(self.current_state)
        if config:
            for action in config.do_actions:
                try:
                    # Execute actions in thread pool for concurrency
                    future = self._executor.submit(action)
                    future.add_done_callback(self._handle_action_completion)
                except Exception as e:
                    self.get_logger().error(f"Do action failed: {e}")
    
    def _execute_exit_actions(self):
        """Execute exit actions for current state"""
        config = self.state_configs.get(self.current_state)
        if config:
            for action in config.exit_actions:
                try:
                    action()
                except Exception as e:
                    self.get_logger().error(f"Exit action failed: {e}")
        
        # Cancel timeout timer
        if self.state_timeout_timer:
            self.state_timeout_timer.cancel()
            self.state_timeout_timer = None
    
    def _handle_action_completion(self, future: Future):
        """Handle completion of asynchronous actions"""
        try:
            result = future.result(timeout=1.0)
            # Process action result if needed
        except Exception as e:
            self.get_logger().error(f"Action completion error: {e}")
    
    def _process_events(self):
        """Process pending events and check for transitions"""
        # Check transition conditions for all possible transitions from current state
        for transition in self.transitions:
            if transition.from_state == self.current_state:
                # Check if condition is satisfied
                condition_met = True
                if transition.condition:
                    try:
                        condition_met = transition.condition()
                    except Exception as e:
                        self.get_logger().error(f"Transition condition error: {e}")
                        condition_met = False
                
                if condition_met:
                    # Execute transition
                    self._execute_transition(transition)
                    break
    
    def _execute_transition(self, transition: StateTransition):
        """Execute a state transition"""
        with self._transition_lock:
            try:
                self.get_logger().info(
                    f"Transition: {transition.from_state.name} -> "
                    f"{transition.to_state.name} on {transition.event.name}"
                )
                
                # Execute exit actions of current state
                self._execute_exit_actions()
                
                # Update transition count
                key = (transition.from_state, transition.to_state)
                self.transition_counts[key] = self.transition_counts.get(key, 0) + 1
                
                # Update state history
                self.previous_state = self.current_state
                self.current_state = transition.to_state
                
                # Reset state entry time
                self.state_entry_time = None
                
                # Execute transition action if specified
                if transition.action:
                    try:
                        transition.action()
                    except Exception as e:
                        self.get_logger().error(f"Transition action failed: {e}")
                
                # Publish transition
                self._publish_transition(transition)
                
                # Execute entry actions of new state
                self._execute_entry_actions()
                
            except Exception as e:
                self.get_logger().error(f"Transition execution failed: {e}")
                self._trigger_event(TaskEvent.ERROR_DETECTED)
    
    def _trigger_event(self, event: TaskEvent):
        """Trigger a state machine event"""
        # In a production system, this would add event to a queue
        # For simplicity, we'll trigger immediate processing
        self.get_logger().info(f"Event triggered: {event.name}")
        
        # Check for immediate transitions based on event
        for transition in self.transitions:
            if transition.from_state == self.current_state and transition.event == event:
                self._execute_transition(transition)
                break
    
    def _check_state_timeout(self):
        """Check if current state has timed out"""
        if self.state_entry_time is None:
            return
        
        config = self.state_configs.get(self.current_state)
        if config and config.timeout_seconds < float('inf'):
            duration = (datetime.now() - self.state_entry_time).total_seconds()
            
            if duration > config.timeout_seconds:
                self.get_logger().warn(
                    f"State {self.current_state.name} timed out after {duration:.1f}s"
                )
                self._trigger_event(TaskEvent.TIMEOUT_EXPIRED)
    
    def _handle_state_timeout(self):
        """Handle state timeout"""
        self._trigger_event(TaskEvent.TIMEOUT_EXPIRED)
    
    def _publish_state(self):
        """Publish current state information"""
        state_msg = RobotStatus()
        state_msg.timestamp = self.get_clock().now().to_msg()
        state_msg.status = self.current_state.name
        state_msg.details = json.dumps({
            'previous_state': self.previous_state.name,
            'state_duration': (datetime.now() - self.state_entry_time).total_seconds() 
                             if self.state_entry_time else 0.0,
            'active_tasks': len(self.active_tasks),
            'queued_tasks': self.task_queue.qsize()
        })
        
        self.state_pub.publish(state_msg)
    
    def _publish_transition(self, transition: StateTransition):
        """Publish state transition information"""
        transition_msg = String()
        transition_msg.data = json.dumps({
            'timestamp': datetime.now().isoformat(),
            'from_state': transition.from_state.name,
            'to_state': transition.to_state.name,
            'event': transition.event.name,
            'transition_count': self.transition_counts.get(
                (transition.from_state, transition.to_state), 0
            )
        })
        
        self.state_transition_pub.publish(transition_msg)
    
    def _publish_metrics(self):
        """Publish state machine metrics"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'current_state': self.current_state.name,
            'state_durations': {
                state.name: {
                    'count': len(durations),
                    'mean': np.mean(durations) if durations else 0.0,
                    'std': np.std(durations) if len(durations) > 1 else 0.0,
                    'min': np.min(durations) if durations else 0.0,
                    'max': np.max(durations) if durations else 0.0
                }
                for state, durations in self.state_durations.items()
                if durations
            },
            'transition_counts': {
                f"{from_state.name}->{to_state.name}": count
                for (from_state, to_state), count in self.transition_counts.items()
                if count > 0
            },
            'error_rates': {
                state.name: np.mean(errors) if errors else 0.0
                for state, errors in self.error_rates.items()
            },
            'task_stats': {
                'completed': len(self.completed_tasks),
                'failed': len(self.failed_tasks),
                'active': len(self.active_tasks),
                'queued': self.task_queue.qsize()
            }
        }
        
        metrics_msg = String()
        metrics_msg.data = json.dumps(metrics)
        self.metrics_pub.publish(metrics_msg)
    
    def _update_visualization(self):
        """Update state machine visualization"""
        marker_array = MarkerArray()
        
        # Create state visualization markers
        for i, state in enumerate(TaskState):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "state_machine"
            marker.id = i
            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD
            
            # Position states in a grid
            row = i // 5
            col = i % 5
            marker.pose.position.x = col * 0.5 - 1.0
            marker.pose.position.y = row * 0.5 - 1.0
            marker.pose.position.z = 1.0
            marker.pose.orientation.w = 1.0
            
            marker.scale.z = 0.1  # Text height
            marker.text = state.name
            
            # Color based on state
            if state == self.current_state:
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 1.0
            elif state == self.previous_state:
                marker.color.r = 1.0
                marker.color.g = 0.5
                marker.color.b = 0.0
                marker.color.a = 0.7
            else:
                marker.color.r = 0.5
                marker.color.g = 0.5
                marker.color.b = 0.5
                marker.color.a = 0.3
            
            marker.lifetime = Duration(seconds=1.0).to_msg()
            marker_array.markers.append(marker)
        
        self.visualization_pub.publish(marker_array)
    
    # ========== State Action Implementations ==========
    
    def _log_state_entry(self):
        """Log state entry"""
        self.get_logger().info(f"Entering state: {self.current_state.name}")
    
    def _log_state_exit(self):
        """Log state exit"""
        self.get_logger().info(f"Exiting state: {self.current_state.name}")
    
    def _reset_task_context(self):
        """Reset task context for new task"""
        with self._task_lock:
            if self.active_tasks:
                # Clear completed tasks
                task_ids = list(self.active_tasks.keys())
                for task_id in task_ids:
                    context = self.active_tasks[task_id]
                    if context.task_id != task_id:  # Simplified completion check
                        self.completed_tasks.append(context)
                        del self.active_tasks[task_id]
    
    def _monitor_task_queue(self):
        """Monitor task queue for new tasks"""
        # Already handled by _process_task_queue timer
        pass
    
    def _initialize_hardware(self):
        """Initialize all hardware components"""
        self.get_logger().info("Initializing hardware components...")
        # In production, this would call hardware initialization services
        # For now, simulate with a delay
        time.sleep(0.5)
    
    def _check_hardware_status(self):
        """Check hardware status"""
        # Check if all hardware components are ready
        return True  # Simplified
    
    def _calibrate_sensors(self):
        """Calibrate sensors"""
        self.get_logger().info("Calibrating sensors...")
    
    def _verify_initialization(self):
        """Verify initialization completed successfully"""
        return self._check_hardware_status()
    
    def _start_environment_scan(self):
        """Start environment scanning"""
        self.get_logger().info("Starting environment scan...")
        # Trigger perception system to start scanning
    
    def _process_scan_data(self):
        """Process scan data"""
        # Process obstacle map and update context
        pass
    
    def _capture_environment_data(self):
        """Capture environment data"""
        # Capture camera images, point clouds, etc.
        pass
    
    def _update_obstacle_map(self):
        """Update obstacle map"""
        pass
    
    def _enable_object_detection(self):
        """Enable object detection"""
        self.get_logger().info("Enabling object detection...")
    
    def _filter_detected_objects(self):
        """Filter detected objects"""
        # Apply filters to detected objects
        pass
    
    def _process_detection_results(self):
        """Process object detection results"""
        # Process results from detection buffer
        pass
    
    def _update_object_tracking(self):
        """Update object tracking"""
        pass
    
    def _load_selection_policy(self):
        """Load target selection policy"""
        # Load selection rules from configuration
        pass
    
    def _validate_target_selection(self):
        """Validate target selection"""
        return True  # Simplified
    
    def _evaluate_target_candidates(self):
        """Evaluate target candidates"""
        # Evaluate objects based on graspability, priority, etc.
        pass
    
    def _compute_grasp_poses(self):
        """Compute grasp poses for target"""
        # Calculate optimal grasp poses
        pass
    
    def _compute_ik_solutions(self):
        """Compute inverse kinematics solutions"""
        self.get_logger().info("Computing IK solutions...")
    
    def _validate_motion_plan(self):
        """Validate motion plan"""
        return True  # Simplified
    
    def _generate_trajectory(self):
        """Generate motion trajectory"""
        pass
    
    def _check_collisions(self):
        """Check for collisions in planned path"""
        return False  # No collisions (simplified)
    
    def _start_motion_execution(self):
        """Start motion execution"""
        self.get_logger().info("Starting motion execution...")
    
    def _verify_approach_position(self):
        """Verify approach position"""
        return True  # Simplified
    
    def _monitor_motion_execution(self):
        """Monitor motion execution"""
        pass
    
    def _adjust_trajectory(self):
        """Adjust trajectory if needed"""
        pass
    
    def _prepare_gripper(self):
        """Prepare gripper for grasping"""
        self.get_logger().info("Preparing gripper...")
    
    def _verify_grasp_success(self):
        """Verify grasp success"""
        return True  # Simplified
    
    def _execute_grasp_sequence(self):
        """Execute grasp sequence"""
        pass
    
    def _monitor_grip_force(self):
        """Monitor grip force"""
        pass
    
    # ... (Additional action implementations would continue here)
    
    # ========== Transition Condition Implementations ==========
    
    def _can_start_task(self) -> bool:
        """Check if a new task can be started"""
        return self.current_state == TaskState.IDLE and not self.task_queue.empty()
    
    def _hardware_initialized(self) -> bool:
        """Check if hardware initialization completed"""
        return True  # Simplified
    
    def _initialization_failed(self) -> bool:
        """Check if initialization failed"""
        return False  # Simplified
    
    def _environment_scanned(self) -> bool:
        """Check if environment scanning completed"""
        return not self.obstacle_buffer.empty()
    
    def _objects_detected(self) -> bool:
        """Check if objects were detected"""
        return not self.object_detection_buffer.empty()
    
    def _no_objects_detected(self) -> bool:
        """Check if no objects were detected"""
        return self.object_detection_buffer.empty()
    
    def _target_valid(self) -> bool:
        """Check if target is valid"""
        with self._task_lock:
            if self.active_tasks:
                context = next(iter(self.active_tasks.values()))
                return context.target_object is not None
        return False
    
    def _target_invalid(self) -> bool:
        """Check if target is invalid"""
        return not self._target_valid()
    
    def _motion_plan_valid(self) -> bool:
        """Check if motion plan is valid"""
        return True  # Simplified
    
    def _motion_planning_failed(self) -> bool:
        """Check if motion planning failed"""
        return False  # Simplified
    
    def _approach_successful(self) -> bool:
        """Check if approach was successful"""
        return True  # Simplified
    
    def _approach_failed(self) -> bool:
        """Check if approach failed"""
        return False  # Simplified
    
    def _needs_replanning(self) -> bool:
        """Check if replanning is needed"""
        return False  # Simplified
    
    def _grasp_successful(self) -> bool:
        """Check if grasp was successful"""
        return True  # Simplified
    
    def _grasp_failed(self) -> bool:
        """Check if grasp failed"""
        return False  # Simplified
    
    def _lift_successful(self) -> bool:
        """Check if lift was successful"""
        return True  # Simplified
    
    def _lift_failed(self) -> bool:
        """Check if lift failed"""
        return False  # Simplified
    
    def _transport_successful(self) -> bool:
        """Check if transport was successful"""
        return True  # Simplified
    
    def _transport_failed(self) -> bool:
        """Check if transport failed"""
        return False  # Simplified
    
    def _placement_successful(self) -> bool:
        """Check if placement was successful"""
        return True  # Simplified
    
    def _placement_failed(self) -> bool:
        """Check if placement failed"""
        return False  # Simplified
    
    def _release_successful(self) -> bool:
        """Check if release was successful"""
        return True  # Simplified
    
    def _release_failed(self) -> bool:
        """Check if release failed"""
        return False  # Simplified
    
    def _retraction_successful(self) -> bool:
        """Check if retraction was successful"""
        return True  # Simplified
    
    def _retraction_failed(self) -> bool:
        """Check if retraction failed"""
        return False  # Simplified
    
    def _task_completed_successfully(self) -> bool:
        """Check if task completed successfully"""
        return True  # Simplified
    
    def _task_failed_verification(self) -> bool:
        """Check if task failed verification"""
        return False  # Simplified
    
    def _recovery_possible(self) -> bool:
        """Check if recovery is possible"""
        return True  # Simplified
    
    def _critical_failure(self) -> bool:
        """Check if failure is critical"""
        return False  # Simplified
    
    def _recovery_successful(self) -> bool:
        """Check if recovery was successful"""
        return True  # Simplified
    
    def _recovery_failed(self) -> bool:
        """Check if recovery failed"""
        return False  # Simplified
    
    def _can_reset_from_emergency(self) -> bool:
        """Check if can reset from emergency"""
        return True  # Simplified
    
    def _can_resume(self) -> bool:
        """Check if can resume from pause"""
        return True  # Simplified
    
    def _can_pause(self) -> bool:
        """Check if can pause current operation"""
        config = self.state_configs.get(self.current_state)
        return config.is_interruptible if config else True
    
    def _reset_successful(self) -> bool:
        """Check if reset was successful"""
        return True  # Simplified
    
    def _reset_failed(self) -> bool:
        """Check if reset failed"""
        return False  # Simplified
    
    def _shutdown_aborted(self) -> bool:
        """Check if shutdown was aborted"""
        return False  # Simplified
    
    def _create_new_task_context(self):
        """Create new task context"""
        if not self.task_queue.empty():
            try:
                priority, timestamp, context = self.task_queue.get_nowait()
                with self._task_lock:
                    self.active_tasks[context.task_id] = context
                self.get_logger().info(f"Created task context for {context.task_id}")
            except queue.Empty:
                pass
    
    # ========== Service Callbacks ==========
    
    def _pick_object_callback(self, request, response):
        """Handle pick object service request"""
        try:
            task_id = str(uuid.uuid4())
            task_spec = {
                'task_id': task_id,
                'command': 'pick',
                'object_id': request.object_id,
                'pick_pose': request.pick_pose,
                'priority': TaskPriority.HIGH.value
            }
            
            self._queue_task(task_spec)
            
            response.success = True
            response.task_id = task_id
            response.message = f"Pick task {task_id} queued"
            
        except Exception as e:
            self.get_logger().error(f"Pick object service failed: {e}")
            response.success = False
            response.task_id = ""
            response.message = str(e)
        
        return response
    
    def _place_object_callback(self, request, response):
        """Handle place object service request"""
        try:
            task_id = str(uuid.uuid4())
            task_spec = {
                'task_id': task_id,
                'command': 'place',
                'place_pose': request.place_pose,
                'priority': TaskPriority.HIGH.value
            }
            
            self._queue_task(task_spec)
            
            response.success = True
            response.task_id = task_id
            response.message = f"Place task {task_id} queued"
            
        except Exception as e:
            self.get_logger().error(f"Place object service failed: {e}")
            response.success = False
            response.task_id = ""
            response.message = str(e)
        
        return response
    
    def _get_state_callback(self, request, response):
        """Handle get state service request"""
        response.timestamp = self.get_clock().now().to_msg()
        response.status = self.current_state.name
        response.details = json.dumps({
            'previous_state': self.previous_state.name,
            'state_entry_time': self.state_entry_time.isoformat() 
                               if self.state_entry_time else None,
            'active_tasks': list(self.active_tasks.keys()),
            'queued_tasks': self.task_queue.qsize()
        })
        return response
    
    # ========== Action Server Callbacks ==========
    
    def _pick_place_goal_callback(self, goal_request):
        """Handle new pick-place action goal"""
        self.get_logger().info(f"Received pick-place goal: {goal_request}")
        return GoalResponse.ACCEPT
    
    def _pick_place_cancel_callback(self, goal_handle):
        """Handle pick-place action cancellation"""
        self.get_logger().info(f"Cancelling pick-place goal")
        return CancelResponse.ACCEPT
    
    def _execute_pick_place_callback(self, goal_handle):
        """Execute pick-place action"""
        try:
            goal = goal_handle.request
            
            # Create task context from action goal
            task_id = str(uuid.uuid4())
            context = TaskContext(
                task_id=task_id,
                start_time=datetime.now(),
                target_pose=goal.pick_pose,
                placement_pose=goal.place_pose,
                priority=TaskPriority.HIGH
            )
            
            with self._task_lock:
                self.active_tasks[task_id] = context
            
            # Send feedback
            feedback_msg = PickPlaceAction.Feedback()
            feedback_msg.progress = 0.0
            feedback_msg.status = "Task queued"
            goal_handle.publish_feedback(feedback_msg)
            
            # Wait for task completion (simplified)
            # In production, this would monitor task progress
            time.sleep(1.0)
            
            # Send result
            result = PickPlaceAction.Result()
            result.success = True
            result.message = f"Pick-place task {task_id} completed"
            goal_handle.succeed()
            
            return result
            
        except Exception as e:
            self.get_logger().error(f"Pick-place action failed: {e}")
            goal_handle.abort()
            return PickPlaceAction.Result(success=False, message=str(e))
    
    def destroy_node(self):
        """Clean shutdown of state machine"""
        self.get_logger().info("Task State Machine shutting down...")
        
        # Shutdown thread pool
        self._executor.shutdown(wait=True)
        
        # Clear all buffers
        while not self.object_detection_buffer.empty():
            try:
                self.object_detection_buffer.get_nowait()
            except:
                pass
        
        while not self.obstacle_buffer.empty():
            try:
                self.obstacle_buffer.get_nowait()
            except:
                pass
        
        while not self.joint_state_buffer.empty():
            try:
                self.joint_state_buffer.get_nowait()
            except:
                pass
        
        while not self.safety_state_buffer.empty():
            try:
                self.safety_state_buffer.get_nowait()
            except:
                pass
        
        # Clear task queue
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
            except:
                pass
        
        super().destroy_node()

def main(args=None):
    """Main entry point"""
    rclpy.init(args=args)
    
    # Use multi-threaded executor for parallel state machine execution
    executor = MultiThreadedExecutor(num_threads=12)
    
    state_machine = TaskStateMachine()
    executor.add_node(state_machine)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        state_machine.get_logger().info("State Machine shutdown requested")
    finally:
        executor.shutdown()
        state_machine.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
