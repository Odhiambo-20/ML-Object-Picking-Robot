#!/usr/bin/env python3
"""
Performance Monitor Node for ML-Based Object Picking Robot System

Production-grade performance monitoring system with real-time metrics collection,
anomaly detection, performance profiling, and system optimization recommendations.

Author: Robotics Engineering Team
Version: 4.1.0
License: Proprietary
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
import threading
import time
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any, Deque, Set
from collections import deque, defaultdict
import psutil
import GPUtil
import socket
import os
import sys
import warnings
import asyncio
from datetime import datetime, timedelta
import statistics
import math
import hashlib
import pickle
from pathlib import Path
import signal
import gc
import tracemalloc
import cProfile
import pstats
import io
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import subprocess
import resource

# ROS2 Messages
from std_msgs.msg import Header, String, Float32, Float64, Int32, Bool
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point, Vector3
from std_srvs.srv import Trigger, SetBool
from nav_msgs.msg import Odometry

# Custom messages
from system_monitor_msgs.msg import (
    PerformanceMetrics, 
    SystemPerformance, 
    ComponentPerformance,
    PerformanceAlert,
    ResourceUsage,
    ProfilingData,
    OptimizationRecommendation
)
from system_monitor_msgs.srv import (
    GetPerformanceMetrics,
    StartProfiling,
    StopProfiling,
    GetOptimizationRecommendations,
    SetPerformanceThresholds,
    ResetPerformanceMetrics,
    ExportPerformanceData
)
from robot_interfaces.msg import RobotStatus, DetectedObject, GraspPose


class MetricType(Enum):
    """Types of performance metrics."""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    GPU_USAGE = "gpu_usage"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    SUCCESS_RATE = "success_rate"
    QUEUE_SIZE = "queue_size"
    CACHE_HIT_RATE = "cache_hit_rate"
    FRAME_RATE = "frame_rate"
    INFERENCE_TIME = "inference_time"
    PLANNING_TIME = "planning_time"
    EXECUTION_TIME = "execution_time"
    ENERGY_CONSUMPTION = "energy_consumption"
    TEMPERATURE = "temperature"


class ComponentType(Enum):
    """System components being monitored."""
    PERCEPTION = "perception"
    MOTION_PLANNING = "motion_planning"
    CONTROL = "control"
    NAVIGATION = "navigation"
    TASK_PLANNING = "task_planning"
    HARDWARE = "hardware"
    NETWORK = "network"
    STORAGE = "storage"
    SYSTEM = "system"
    ML_MODEL = "ml_model"
    GRASPING = "grasping"
    VISUALIZATION = "visualization"
    COMMUNICATION = "communication"
    SAFETY = "safety"
    LOGGING = "logging"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = 0
    WARNING = 1
    ERROR = 2
    CRITICAL = 3


class PerformanceState(Enum):
    """Performance state based on metrics."""
    OPTIMAL = 0
    NORMAL = 1
    DEGRADED = 2
    CRITICAL = 3
    FAILED = 4


@dataclass
class MetricValue:
    """Single metric value with timestamp and metadata."""
    timestamp: datetime
    value: float
    unit: str
    component: str
    metric_type: MetricType
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'value': self.value,
            'unit': self.unit,
            'component': self.component,
            'metric_type': self.metric_type.value,
            'tags': self.tags,
            'metadata': self.metadata
        }


@dataclass
class ComponentMetrics:
    """Collection of metrics for a component."""
    component: ComponentType
    metrics: Dict[str, List[MetricValue]]
    state: PerformanceState
    last_update: datetime
    baseline: Dict[str, float] = field(default_factory=dict)
    thresholds: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)  # warning, error, critical
    
    def add_metric(self, metric: MetricValue):
        """Add a metric value."""
        metric_key = f"{metric.metric_type.value}_{metric.component}"
        if metric_key not in self.metrics:
            self.metrics[metric_key] = []
        
        self.metrics[metric_key].append(metric)
        self.last_update = metric.timestamp
        
        # Keep only last 1000 values
        if len(self.metrics[metric_key]) > 1000:
            self.metrics[metric_key].pop(0)
    
    def get_recent_values(self, metric_type: MetricType, count: int = 100) -> List[float]:
        """Get recent values for a metric type."""
        metric_key = f"{metric_type.value}_{self.component.value}"
        if metric_key not in self.metrics:
            return []
        
        values = [m.value for m in self.metrics[metric_key][-count:]]
        return values
    
    def get_statistics(self, metric_type: MetricType) -> Dict[str, float]:
        """Get statistics for a metric type."""
        values = self.get_recent_values(metric_type)
        if not values:
            return {}
        
        return {
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0.0,
            'min': min(values),
            'max': max(values),
            'p95': np.percentile(values, 95) if len(values) > 1 else values[0],
            'p99': np.percentile(values, 99) if len(values) > 1 else values[0]
        }
    
    def check_thresholds(self, metric_type: MetricType) -> Optional[AlertSeverity]:
        """Check if metric exceeds thresholds."""
        values = self.get_recent_values(metric_type, 10)  # Last 10 values
        if not values:
            return None
        
        current_value = values[-1]
        metric_key = f"{metric_type.value}"
        
        if metric_key in self.thresholds:
            warning, error, critical = self.thresholds[metric_key]
            
            if current_value >= critical:
                return AlertSeverity.CRITICAL
            elif current_value >= error:
                return AlertSeverity.ERROR
            elif current_value >= warning:
                return AlertSeverity.WARNING
        
        return None


@dataclass
class PerformanceProfile:
    """Performance profiling data for analysis."""
    profile_id: str
    start_time: datetime
    end_time: datetime
    component: ComponentType
    metrics: Dict[str, List[MetricValue]]
    hotspots: List[Dict[str, Any]]
    recommendations: List[str]
    summary: Dict[str, float]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'profile_id': self.profile_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'component': self.component.value,
            'metrics_summary': self.summary,
            'hotspots': self.hotspots,
            'recommendations': self.recommendations
        }


class PerformanceMonitor(Node):
    """
    Advanced performance monitoring system for robotic applications.
    
    Features:
    - Real-time performance metrics collection for all system components
    - Machine learning-based anomaly detection
    - Performance profiling and bottleneck identification
    - Automatic optimization recommendations
    - Resource usage prediction and capacity planning
    - Integration with ROS2 diagnostics
    - Historical performance analysis
    - Energy consumption monitoring
    """
    
    def __init__(self):
        super().__init__('performance_monitor')
        
        # Configuration parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('monitoring_interval_sec', 1.0),
                ('metrics_history_size', 1000),
                ('enable_anomaly_detection', True),
                ('anomaly_detection_window', 100),
                ('anomaly_threshold_sigma', 3.0),
                ('enable_profiling', True),
                ('profiling_duration_sec', 60.0),
                ('enable_predictive_analysis', True),
                ('prediction_horizon_minutes', 60),
                ('enable_energy_monitoring', False),
                ('energy_monitoring_interval_sec', 5.0),
                ('performance_baseline_days', 7),
                ('alert_cooldown_sec', 300.0),
                ('enable_trend_analysis', True),
                ('trend_window_hours', 24),
                ('enable_correlation_analysis', True),
                ('correlation_min_samples', 100),
                ('enable_auto_optimization', False),
                ('auto_optimization_threshold', 0.8),
                ('enable_visualization', True),
                ('visualization_update_interval_sec', 2.0),
                ('enable_benchmarking', True),
                ('benchmark_interval_hours', 24),
                ('enable_capacity_planning', True),
                ('capacity_prediction_days', 30),
                ('enable_performance_degradation_detection', True),
                ('degradation_threshold_percent', 20.0),
                ('enable_resource_leak_detection', True),
                ('leak_detection_window', 1000),
                ('enable_gpu_monitoring', True),
                ('gpu_monitoring_interval_sec', 2.0),
                ('enable_network_monitoring', True),
                ('network_monitoring_interval_sec', 2.0),
                ('enable_storage_monitoring', True),
                ('storage_monitoring_interval_sec', 30.0),
                ('enable_ml_model_monitoring', True),
                ('ml_model_monitoring_interval_sec', 5.0),
                ('enable_ros2_monitoring', True),
                ('ros2_monitoring_interval_sec', 2.0),
                ('enable_performance_logging', True),
                ('performance_logging_interval_sec', 60.0),
                ('performance_data_directory', '/var/performance_data'),
                ('max_storage_gb', 10.0),
                ('enable_compression', True),
                ('compression_level', 6),
                ('enable_encryption', False),
                ('encryption_key', ''),
                ('report_generation_interval_hours', 1),
                ('enable_real_time_analysis', True),
                ('real_time_analysis_window', 500),
                ('enable_fault_prediction', False),
                ('fault_prediction_horizon_hours', 24)
            ]
        )
        
        # Initialize data structures
        self.components: Dict[ComponentType, ComponentMetrics] = {}
        self.metrics_history: Deque[MetricValue] = deque(maxlen=10000)
        self.alerts_history: List[Dict] = []
        self.performance_profiles: Dict[str, PerformanceProfile] = {}
        self.active_profilers: Dict[str, Any] = {}
        self.baseline_established: bool = False
        self.baseline_data: Dict[str, Dict] = {}
        
        # Performance analysis models
        self.anomaly_detectors: Dict[str, Any] = {}
        self.trend_models: Dict[str, Any] = {}
        self.correlation_matrices: Dict[str, np.ndarray] = {}
        self.prediction_models: Dict[str, Any] = {}
        
        # Thread management
        self.monitoring_active = True
        self.monitoring_threads: List[threading.Thread] = []
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        
        # Performance state
        self.overall_state = PerformanceState.NORMAL
        self.last_state_change = datetime.now()
        self.state_history: List[Tuple[datetime, PerformanceState]] = []
        
        # Initialize ROS2 interfaces
        self._initialize_ros_interfaces()
        
        # Create data directory
        self.data_dir = Path(self.get_parameter('performance_data_directory').value)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._initialize_components()
        
        # Initialize monitoring modules
        self._initialize_monitoring_modules()
        
        # Start monitoring threads
        self._start_monitoring_threads()
        
        # Start timers
        self._start_timers()
        
        # Initialize baseline if exists
        self._load_baseline()
        
        self.get_logger().info(
            f"Performance Monitor initialized with {len(self.components)} components"
        )
    
    def _initialize_ros_interfaces(self):
        """Initialize ROS2 publishers, subscribers, and services."""
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
        
        # Publishers
        self.metrics_pub = self.create_publisher(
            PerformanceMetrics,
            '/performance/metrics',
            reliable_qos
        )
        
        self.alerts_pub = self.create_publisher(
            PerformanceAlert,
            '/performance/alerts',
            reliable_qos
        )
        
        self.system_perf_pub = self.create_publisher(
            SystemPerformance,
            '/performance/system',
            reliable_qos
        )
        
        self.recommendations_pub = self.create_publisher(
            OptimizationRecommendation,
            '/performance/recommendations',
            reliable_qos
        )
        
        self.visualization_pub = self.create_publisher(
            MarkerArray,
            '/performance/visualization',
            best_effort_qos
        )
        
        # Subscribers for component-specific metrics
        self.create_subscription(
            RobotStatus,
            '/robot/status',
            self._handle_robot_status,
            best_effort_qos
        )
        
        self.create_subscription(
            DetectedObject,
            '/perception/detected_objects',
            self._handle_detected_objects,
            best_effort_qos
        )
        
        self.create_subscription(
            DiagnosticArray,
            '/diagnostics',
            self._handle_diagnostics,
            best_effort_qos
        )
        
        # Services
        self.services = {
            'get_metrics': self.create_service(
                GetPerformanceMetrics,
                '/performance/get_metrics',
                self._handle_get_metrics
            ),
            'start_profiling': self.create_service(
                StartProfiling,
                '/performance/start_profiling',
                self._handle_start_profiling
            ),
            'stop_profiling': self.create_service(
                StopProfiling,
                '/performance/stop_profiling',
                self._handle_stop_profiling
            ),
            'get_recommendations': self.create_service(
                GetOptimizationRecommendations,
                '/performance/get_recommendations',
                self._handle_get_recommendations
            ),
            'set_thresholds': self.create_service(
                SetPerformanceThresholds,
                '/performance/set_thresholds',
                self._handle_set_thresholds
            ),
            'reset_metrics': self.create_service(
                ResetPerformanceMetrics,
                '/performance/reset_metrics',
                self._handle_reset_metrics
            ),
            'export_data': self.create_service(
                ExportPerformanceData,
                '/performance/export_data',
                self._handle_export_data
            )
        }
    
    def _initialize_components(self):
        """Initialize monitoring for all system components."""
        components = [
            ComponentType.PERCEPTION,
            ComponentType.MOTION_PLANNING,
            ComponentType.CONTROL,
            ComponentType.NAVIGATION,
            ComponentType.TASK_PLANNING,
            ComponentType.HARDWARE,
            ComponentType.NETWORK,
            ComponentType.STORAGE,
            ComponentType.SYSTEM,
            ComponentType.ML_MODEL,
            ComponentType.GRASPING,
            ComponentType.VISUALIZATION,
            ComponentType.COMMUNICATION,
            ComponentType.SAFETY,
            ComponentType.LOGGING
        ]
        
        default_thresholds = {
            'cpu_usage': (70.0, 85.0, 95.0),
            'memory_usage': (75.0, 85.0, 95.0),
            'response_time': (100.0, 200.0, 500.0),  # ms
            'error_rate': (5.0, 10.0, 20.0),  # percent
            'latency': (50.0, 100.0, 200.0),  # ms
            'queue_size': (80, 90, 95),  # percent of capacity
            'frame_rate': (15.0, 10.0, 5.0),  # FPS
            'inference_time': (100.0, 200.0, 500.0),  # ms
            'planning_time': (500.0, 1000.0, 2000.0),  # ms
            'temperature': (70.0, 80.0, 90.0)  # Celsius
        }
        
        for component in components:
            self.components[component] = ComponentMetrics(
                component=component,
                metrics={},
                state=PerformanceState.NORMAL,
                last_update=datetime.now(),
                thresholds=default_thresholds.copy()
            )
    
    def _initialize_monitoring_modules(self):
        """Initialize specialized monitoring modules."""
        self.monitoring_modules = {
            'system': SystemMonitor(self),
            'gpu': GPUMonitor(self) if self.get_parameter('enable_gpu_monitoring').value else None,
            'network': NetworkMonitor(self) if self.get_parameter('enable_network_monitoring').value else None,
            'storage': StorageMonitor(self) if self.get_parameter('enable_storage_monitoring').value else None,
            'ml_model': MLModelMonitor(self) if self.get_parameter('enable_ml_model_monitoring').value else None,
            'ros2': ROS2Monitor(self) if self.get_parameter('enable_ros2_monitoring').value else None,
            'energy': EnergyMonitor(self) if self.get_parameter('enable_energy_monitoring').value else None
        }
    
    def _start_monitoring_threads(self):
        """Start all monitoring threads."""
        # Main monitoring thread
        main_thread = threading.Thread(target=self._monitoring_loop)
        main_thread.daemon = True
        main_thread.start()
        self.monitoring_threads.append(main_thread)
        
        # Anomaly detection thread
        if self.get_parameter('enable_anomaly_detection').value:
            anomaly_thread = threading.Thread(target=self._anomaly_detection_loop)
            anomaly_thread.daemon = True
            anomaly_thread.start()
            self.monitoring_threads.append(anomaly_thread)
        
        # Trend analysis thread
        if self.get_parameter('enable_trend_analysis').value:
            trend_thread = threading.Thread(target=self._trend_analysis_loop)
            trend_thread.daemon = True
            trend_thread.start()
            self.monitoring_threads.append(trend_thread)
        
        # Predictive analysis thread
        if self.get_parameter('enable_predictive_analysis').value:
            predictive_thread = threading.Thread(target=self._predictive_analysis_loop)
            predictive_thread.daemon = True
            predictive_thread.start()
            self.monitoring_threads.append(predictive_thread)
        
        # Visualization thread
        if self.get_parameter('enable_visualization').value:
            visualization_thread = threading.Thread(target=self._visualization_loop)
            visualization_thread.daemon = True
            visualization_thread.start()
            self.monitoring_threads.append(visualization_thread)
        
        # Benchmarking thread
        if self.get_parameter('enable_benchmarking').value:
            benchmark_thread = threading.Thread(target=self._benchmarking_loop)
            benchmark_thread.daemon = True
            benchmark_thread.start()
            self.monitoring_threads.append(benchmark_thread)
        
        # Capacity planning thread
        if self.get_parameter('enable_capacity_planning').value:
            capacity_thread = threading.Thread(target=self._capacity_planning_loop)
            capacity_thread.daemon = True
            capacity_thread.start()
            self.monitoring_threads.append(capacity_thread)
    
    def _start_timers(self):
        """Start periodic timers."""
        # Metrics publishing timer
        self.create_timer(
            self.get_parameter('monitoring_interval_sec').value,
            self._publish_performance_metrics
        )
        
        # System performance publishing timer
        self.create_timer(
            5.0,
            self._publish_system_performance
        )
        
        # Baseline update timer
        self.create_timer(
            3600.0,  # 1 hour
            self._update_baseline
        )
        
        # Report generation timer
        self.create_timer(
            self.get_parameter('report_generation_interval_hours').value * 3600,
            self._generate_performance_report
        )
        
        # Data cleanup timer
        self.create_timer(
            3600.0,  # 1 hour
            self._cleanup_old_data
        )
        
        # State evaluation timer
        self.create_timer(
            10.0,
            self._evaluate_system_state
        )
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        monitoring_interval = self.get_parameter('monitoring_interval_sec').value
        
        while self.monitoring_active and rclpy.ok():
            try:
                loop_start = time.time()
                
                # Collect system-wide metrics
                self._collect_system_metrics()
                
                # Collect component-specific metrics
                self._collect_component_metrics()
                
                # Run monitoring modules
                self._run_monitoring_modules()
                
                # Update performance state
                self._update_performance_state()
                
                # Check for alerts
                self._check_alerts()
                
                # Log performance data
                if self.get_parameter('enable_performance_logging').value:
                    if time.time() - getattr(self, '_last_log_time', 0) > \
                       self.get_parameter('performance_logging_interval_sec').value:
                        self._log_performance_data()
                        self._last_log_time = time.time()
                
                # Calculate sleep time to maintain interval
                loop_time = time.time() - loop_start
                sleep_time = max(0.01, monitoring_interval - loop_time)
                time.sleep(sleep_time)
                
            except Exception as e:
                self.get_logger().error(f"Error in monitoring loop: {str(e)}")
                time.sleep(1.0)
    
    def _collect_system_metrics(self):
        """Collect system-wide performance metrics."""
        try:
            current_time = datetime.now()
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
            cpu_avg = statistics.mean(cpu_percent) if cpu_percent else 0.0
            
            self._add_metric(
                component=ComponentType.SYSTEM,
                metric_type=MetricType.CPU_USAGE,
                value=cpu_avg,
                unit="percent",
                tags=["system", "cpu"]
            )
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self._add_metric(
                component=ComponentType.SYSTEM,
                metric_type=MetricType.MEMORY_USAGE,
                value=memory.percent,
                unit="percent",
                tags=["system", "memory"]
            )
            
            # Disk I/O metrics
            disk_io = psutil.disk_io_counters()
            if disk_io:
                self._add_metric(
                    component=ComponentType.SYSTEM,
                    metric_type=MetricType.DISK_IO,
                    value=disk_io.read_bytes + disk_io.write_bytes,
                    unit="bytes/sec",
                    tags=["system", "disk"]
                )
            
            # Network metrics
            net_io = psutil.net_io_counters()
            if net_io:
                self._add_metric(
                    component=ComponentType.SYSTEM,
                    metric_type=MetricType.NETWORK_IO,
                    value=net_io.bytes_sent + net_io.bytes_recv,
                    unit="bytes/sec",
                    tags=["system", "network"]
                )
            
            # System temperature
            temp = self._get_system_temperature()
            if temp:
                self._add_metric(
                    component=ComponentType.SYSTEM,
                    metric_type=MetricType.TEMPERATURE,
                    value=temp,
                    unit="celsius",
                    tags=["system", "temperature"]
                )
            
            # Process count
            process_count = len(psutil.pids())
            self._add_metric(
                component=ComponentType.SYSTEM,
                metric_type=MetricType.THROUGHPUT,
                value=process_count,
                unit="processes",
                tags=["system", "processes"]
            )
            
            # System load
            load_avg = os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0.0
            self._add_metric(
                component=ComponentType.SYSTEM,
                metric_type=MetricType.CPU_USAGE,
                value=load_avg,
                unit="load",
                tags=["system", "load"],
                metadata={'load_1min': load_avg}
            )
            
        except Exception as e:
            self.get_logger().error(f"Error collecting system metrics: {str(e)}")
    
    def _get_system_temperature(self) -> Optional[float]:
        """Get system temperature from available sensors."""
        try:
            # Try Raspberry Pi temperature
            if os.path.exists('/sys/class/thermal/thermal_zone0/temp'):
                with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                    temp_millic = float(f.read().strip())
                return temp_millic / 1000.0
            
            # Try psutil sensors
            if hasattr(psutil, 'sensors_temperatures'):
                temps = psutil.sensors_temperatures()
                if temps:
                    # Get first available temperature
                    for sensor_name, sensor_temps in temps.items():
                        if sensor_temps:
                            return sensor_temps[0].current
            
            return None
            
        except Exception:
            return None
    
    def _collect_component_metrics(self):
        """Collect component-specific performance metrics."""
        # This would be populated from component-specific monitoring
        # For now, we'll simulate some metrics
        
        current_time = datetime.now()
        
        # Perception metrics (simulated)
        perception_latency = np.random.normal(50, 10)  # ms
        perception_fps = np.random.normal(30, 2)
        
        self._add_metric(
            component=ComponentType.PERCEPTION,
            metric_type=MetricType.LATENCY,
            value=perception_latency,
            unit="milliseconds",
            tags=["perception", "latency"]
        )
        
        self._add_metric(
            component=ComponentType.PERCEPTION,
            metric_type=MetricType.FRAME_RATE,
            value=perception_fps,
            unit="fps",
            tags=["perception", "fps"]
        )
        
        # Motion planning metrics (simulated)
        planning_time = np.random.normal(100, 20)  # ms
        
        self._add_metric(
            component=ComponentType.MOTION_PLANNING,
            metric_type=MetricType.PLANNING_TIME,
            value=planning_time,
            unit="milliseconds",
            tags=["motion", "planning"]
        )
        
        # Control metrics (simulated)
        control_latency = np.random.normal(5, 1)  # ms
        
        self._add_metric(
            component=ComponentType.CONTROL,
            metric_type=MetricType.RESPONSE_TIME,
            value=control_latency,
            unit="milliseconds",
            tags=["control", "response"]
        )
        
        # ML model metrics (simulated)
        inference_time = np.random.normal(15, 3)  # ms
        model_accuracy = np.random.normal(0.95, 0.01)
        
        self._add_metric(
            component=ComponentType.ML_MODEL,
            metric_type=MetricType.INFERENCE_TIME,
            value=inference_time,
            unit="milliseconds",
            tags=["ml", "inference"]
        )
        
        self._add_metric(
            component=ComponentType.ML_MODEL,
            metric_type=MetricType.SUCCESS_RATE,
            value=model_accuracy * 100,
            unit="percent",
            tags=["ml", "accuracy"]
        )
    
    def _run_monitoring_modules(self):
        """Run specialized monitoring modules."""
        for module_name, module in self.monitoring_modules.items():
            if module and hasattr(module, 'update'):
                try:
                    module.update()
                except Exception as e:
                    self.get_logger().error(f"Error in {module_name} module: {str(e)}")
    
    def _add_metric(self, component: ComponentType, metric_type: MetricType, 
                   value: float, unit: str, tags: List[str] = None, 
                   metadata: Dict[str, Any] = None):
        """Add a metric value to the appropriate component."""
        try:
            metric = MetricValue(
                timestamp=datetime.now(),
                value=float(value),
                unit=unit,
                component=component.value,
                metric_type=metric_type,
                tags=tags or [],
                metadata=metadata or {}
            )
            
            # Add to component
            if component in self.components:
                self.components[component].add_metric(metric)
            
            # Add to history
            self.metrics_history.append(metric)
            
            # Update anomaly detectors
            if self.get_parameter('enable_anomaly_detection').value:
                self._update_anomaly_detector(component, metric_type, value)
            
        except Exception as e:
            self.get_logger().error(f"Error adding metric: {str(e)}")
    
    def _update_anomaly_detector(self, component: ComponentType, 
                                metric_type: MetricType, value: float):
        """Update anomaly detection model for a metric."""
        detector_key = f"{component.value}_{metric_type.value}"
        
        if detector_key not in self.anomaly_detectors:
            self.anomaly_detectors[detector_key] = {
                'values': deque(maxlen=100),
                'mean': 0.0,
                'std': 0.0,
                'last_anomaly': None
            }
        
        detector = self.anomaly_detectors[detector_key]
        detector['values'].append(value)
        
        # Calculate statistics if we have enough data
        if len(detector['values']) >= 10:
            values_array = np.array(detector['values'])
            detector['mean'] = np.mean(values_array)
            detector['std'] = np.std(values_array)
            
            # Check for anomaly
            threshold_sigma = self.get_parameter('anomaly_threshold_sigma').value
            if detector['std'] > 0:
                z_score = abs(value - detector['mean']) / detector['std']
                
                if z_score > threshold_sigma:
                    # Anomaly detected
                    anomaly_time = datetime.now()
                    
                    # Check cooldown period
                    if (detector['last_anomaly'] is None or 
                        (anomaly_time - detector['last_anomaly']).total_seconds() > 
                        self.get_parameter('alert_cooldown_sec').value):
                        
                        self._create_anomaly_alert(
                            component=component,
                            metric_type=metric_type,
                            value=value,
                            mean=detector['mean'],
                            std=detector['std'],
                            z_score=z_score
                        )
                        
                        detector['last_anomaly'] = anomaly_time
    
    def _create_anomaly_alert(self, component: ComponentType, metric_type: MetricType,
                             value: float, mean: float, std: float, z_score: float):
        """Create an anomaly alert."""
        alert_msg = PerformanceAlert()
        alert_msg.header.stamp = self.get_clock().now().to_msg()
        alert_msg.header.frame_id = "performance"
        
        alert_msg.severity = AlertSeverity.WARNING.value
        alert_msg.component = component.value
        alert_msg.metric_type = metric_type.value
        alert_msg.current_value = value
        alert_msg.threshold_value = mean + (self.get_parameter('anomaly_threshold_sigma').value * std)
        alert_msg.message = f"Anomaly detected in {component.value}.{metric_type.value}: {value:.2f} (z-score: {z_score:.2f})"
        alert_msg.timestamp = datetime.now().isoformat()
        
        # Add context
        context = {
            'mean': mean,
            'std': std,
            'z_score': z_score,
            'threshold_sigma': self.get_parameter('anomaly_threshold_sigma').value
        }
        alert_msg.context_json = json.dumps(context)
        
        # Publish alert
        self.alerts_pub.publish(alert_msg)
        
        # Store in history
        self.alerts_history.append({
            'timestamp': datetime.now(),
            'severity': AlertSeverity.WARNING,
            'component': component.value,
            'metric_type': metric_type.value,
            'value': value,
            'context': context
        })
        
        # Keep only last 1000 alerts
        if len(self.alerts_history) > 1000:
            self.alerts_history.pop(0)
        
        self.get_logger().warn(f"PERFORMANCE ANOMALY: {alert_msg.message}")
    
    def _update_performance_state(self):
        """Update overall system performance state."""
        try:
            # Calculate weighted score based on component states
            component_states = []
            component_weights = {
                ComponentType.SYSTEM: 0.3,
                ComponentType.PERCEPTION: 0.15,
                ComponentType.MOTION_PLANNING: 0.15,
                ComponentType.CONTROL: 0.15,
                ComponentType.ML_MODEL: 0.1,
                ComponentType.HARDWARE: 0.1,
                ComponentType.NETWORK: 0.05
            }
            
            for component, weight in component_weights.items():
                if component in self.components:
                    state_value = self.components[component].state.value
                    component_states.append((state_value, weight))
            
            if component_states:
                # Calculate weighted average
                weighted_sum = sum(state * weight for state, weight in component_states)
                total_weight = sum(weight for _, weight in component_states)
                weighted_avg = weighted_sum / total_weight
                
                # Map to performance state
                if weighted_avg <= 0.5:
                    new_state = PerformanceState.OPTIMAL
                elif weighted_avg <= 1.5:
                    new_state = PerformanceState.NORMAL
                elif weighted_avg <= 2.5:
                    new_state = PerformanceState.DEGRADED
                elif weighted_avg <= 3.5:
                    new_state = PerformanceState.CRITICAL
                else:
                    new_state = PerformanceState.FAILED
                
                # Update if changed
                if new_state != self.overall_state:
                    old_state = self.overall_state
                    self.overall_state = new_state
                    self.last_state_change = datetime.now()
                    
                    self.state_history.append((datetime.now(), new_state))
                    
                    # Log state change
                    self.get_logger().info(
                        f"Performance state changed: {old_state.name} -> {new_state.name}"
                    )
                    
                    # Generate alert for significant state changes
                    if (new_state.value - old_state.value) >= 2:  # Jumped two or more levels
                        self._create_state_change_alert(old_state, new_state)
        
        except Exception as e:
            self.get_logger().error(f"Error updating performance state: {str(e)}")
    
    def _create_state_change_alert(self, old_state: PerformanceState, new_state: PerformanceState):
        """Create alert for performance state change."""
        alert_msg = PerformanceAlert()
        alert_msg.header.stamp = self.get_clock().now().to_msg()
        alert_msg.header.frame_id = "performance"
        
        # Determine severity
        if new_state in [PerformanceState.CRITICAL, PerformanceState.FAILED]:
            severity = AlertSeverity.CRITICAL
        elif new_state == PerformanceState.DEGRADED:
            severity = AlertSeverity.WARNING
        else:
            severity = AlertSeverity.INFO
        
        alert_msg.severity = severity.value
        alert_msg.component = "system"
        alert_msg.metric_type = "performance_state"
        alert_msg.current_value = new_state.value
        alert_msg.threshold_value = old_state.value
        alert_msg.message = f"Performance state changed: {old_state.name} -> {new_state.name}"
        alert_msg.timestamp = datetime.now().isoformat()
        
        context = {
            'old_state': old_state.name,
            'new_state': new_state.name,
            'change_magnitude': new_state.value - old_state.value
        }
        alert_msg.context_json = json.dumps(context)
        
        self.alerts_pub.publish(alert_msg)
    
    def _check_alerts(self):
        """Check for threshold-based alerts."""
        for component_type, component in self.components.items():
            for metric_key, thresholds in component.thresholds.items():
                # Extract metric type from key
                metric_type_str = metric_key
                try:
                    metric_type = MetricType(metric_type_str)
                except ValueError:
                    continue
                
                # Check thresholds
                alert_level = component.check_thresholds(metric_type)
                if alert_level:
                    # Get current value
                    values = component.get_recent_values(metric_type, 1)
                    if values:
                        current_value = values[-1]
                        
                        # Create alert
                        self._create_threshold_alert(
                            component=component_type,
                            metric_type=metric_type,
                            value=current_value,
                            threshold=thresholds[alert_level.value - 1],  # Map to threshold
                            alert_level=alert_level
                        )
    
    def _create_threshold_alert(self, component: ComponentType, metric_type: MetricType,
                               value: float, threshold: float, alert_level: AlertSeverity):
        """Create a threshold-based alert."""
        alert_msg = PerformanceAlert()
        alert_msg.header.stamp = self.get_clock().now().to_msg()
        alert_msg.header.frame_id = "performance"
        
        alert_msg.severity = alert_level.value
        alert_msg.component = component.value
        alert_msg.metric_type = metric_type.value
        alert_msg.current_value = value
        alert_msg.threshold_value = threshold
        alert_msg.message = f"{component.value}.{metric_type.value} exceeded threshold: {value:.2f} > {threshold:.2f}"
        alert_msg.timestamp = datetime.now().isoformat()
        
        context = {
            'threshold_type': ['warning', 'error', 'critical'][alert_level.value - 1],
            'unit': 'percent' if 'usage' in metric_type.value else 'milliseconds'
        }
        alert_msg.context_json = json.dumps(context)
        
        self.alerts_pub.publish(alert_msg)
        
        self.get_logger().warn(f"THRESHOLD ALERT: {alert_msg.message}")
    
    def _anomaly_detection_loop(self):
        """Background thread for advanced anomaly detection."""
        while self.monitoring_active and rclpy.ok():
            try:
                time.sleep(10)  # Run every 10 seconds
                
                # Perform advanced anomaly detection
                self._perform_advanced_anomaly_detection()
                
                # Update correlation analysis
                if self.get_parameter('enable_correlation_analysis').value:
                    self._update_correlation_analysis()
                
            except Exception as e:
                self.get_logger().error(f"Error in anomaly detection loop: {str(e)}")
                time.sleep(5)
    
    def _perform_advanced_anomaly_detection(self):
        """Perform advanced anomaly detection using multiple techniques."""
        try:
            # Collect recent metrics for analysis
            recent_metrics = []
            window_size = self.get_parameter('anomaly_detection_window').value
            
            for component_type, component in self.components.items():
                for metric_key, values in component.metrics.items():
                    if len(values) >= window_size:
                        recent_values = [v.value for v in values[-window_size:]]
                        
                        # Use multiple detection techniques
                        anomalies = self._detect_anomalies_multivariate(recent_values)
                        
                        if anomalies:
                            # Create combined anomaly alert
                            self._create_multivariate_anomaly_alert(
                                component_type,
                                metric_key,
                                recent_values,
                                anomalies
                            )
            
        except Exception as e:
            self.get_logger().error(f"Error in advanced anomaly detection: {str(e)}")
    
    def _detect_anomalies_multivariate(self, values: List[float]) -> List[int]:
        """Detect anomalies using multiple techniques."""
        if len(values) < 10:
            return []
        
        anomalies = []
        
        # Technique 1: Z-score with dynamic threshold
        z_scores = self._calculate_z_scores(values)
        threshold = self.get_parameter('anomaly_threshold_sigma').value
        
        # Technique 2: Moving average deviation
        window = min(20, len(values) // 2)
        if window >= 5:
            moving_avg = self._moving_average(values, window)
            deviations = [abs(values[i] - moving_avg[i]) for i in range(len(values))]
            avg_deviation = statistics.mean(deviations) if deviations else 0
            
            for i in range(len(values)):
                is_anomaly = False
                
                # Check z-score
                if abs(z_scores[i]) > threshold:
                    is_anomaly = True
                
                # Check moving average deviation (if deviation is significant)
                if i >= window and avg_deviation > 0:
                    deviation_ratio = deviations[i] / avg_deviation
                    if deviation_ratio > 3.0:  # 3x average deviation
                        is_anomaly = True
                
                if is_anomaly:
                    anomalies.append(i)
        
        return anomalies
    
    def _calculate_z_scores(self, values: List[float]) -> List[float]:
        """Calculate z-scores for a list of values."""
        if len(values) < 2:
            return [0.0] * len(values)
        
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0001
        
        if std == 0:
            return [0.0] * len(values)
        
        return [(x - mean) / std for x in values]
    
    def _moving_average(self, values: List[float], window: int) -> List[float]:
        """Calculate moving average."""
        if len(values) < window:
            return values
        
        result = []
        for i in range(len(values)):
            if i < window:
                result.append(statistics.mean(values[:i+1]))
            else:
                result.append(statistics.mean(values[i-window+1:i+1]))
        
        return result
    
    def _create_multivariate_anomaly_alert(self, component: ComponentType, 
                                          metric_key: str, values: List[float],
                                          anomalies: List[int]):
        """Create alert for multivariate anomaly detection."""
        if not anomalies:
            return
        
        # Calculate anomaly statistics
        anomaly_values = [values[i] for i in anomalies]
        normal_values = [values[i] for i in range(len(values)) if i not in anomalies]
        
        if not normal_values:
            return
        
        normal_mean = statistics.mean(normal_values)
        normal_std = statistics.stdev(normal_values) if len(normal_values) > 1 else 0
        
        alert_msg = PerformanceAlert()
        alert_msg.header.stamp = self.get_clock().now().to_msg()
        alert_msg.header.frame_id = "performance"
        
        alert_msg.severity = AlertSeverity.WARNING.value
        alert_msg.component = component.value
        alert_msg.metric_type = metric_key
        alert_msg.current_value = anomaly_values[-1] if anomaly_values else 0
        alert_msg.threshold_value = normal_mean + (2 * normal_std)
        alert_msg.message = f"Multivariate anomaly detected in {component.value}.{metric_key}: {len(anomalies)} anomalies found"
        alert_msg.timestamp = datetime.now().isoformat()
        
        context = {
            'anomaly_count': len(anomalies),
            'anomaly_indices': anomalies[-10:],  # Last 10 anomalies
            'normal_mean': normal_mean,
            'normal_std': normal_std,
            'detection_method': 'multivariate'
        }
        alert_msg.context_json = json.dumps(context)
        
        self.alerts_pub.publish(alert_msg)
    
    def _update_correlation_analysis(self):
        """Update correlation analysis between different metrics."""
        try:
            # Collect recent metric values
            metric_data = {}
            
            for component_type, component in self.components.items():
                for metric_key, values in component.metrics.items():
                    if len(values) >= 50:  # Need sufficient data
                        recent_values = [v.value for v in values[-50:]]
                        metric_data[f"{component_type.value}_{metric_key}"] = recent_values
            
            if len(metric_data) >= 2:
                # Calculate correlation matrix
                self.correlation_matrices['recent'] = self._calculate_correlation_matrix(metric_data)
                
                # Detect strong correlations
                strong_correlations = self._find_strong_correlations(
                    self.correlation_matrices['recent'],
                    list(metric_data.keys()),
                    threshold=0.8
                )
                
                if strong_correlations:
                    self._analyze_correlations(strong_correlations)
        
        except Exception as e:
            self.get_logger().error(f"Error in correlation analysis: {str(e)}")
    
    def _calculate_correlation_matrix(self, metric_data: Dict[str, List[float]]) -> np.ndarray:
        """Calculate correlation matrix for metrics."""
        # Convert to DataFrame for correlation calculation
        df = pd.DataFrame(metric_data)
        
        # Handle NaN values
        df = df.fillna(df.mean())
        
        # Calculate correlation matrix
        correlation_matrix = df.corr().to_numpy()
        
        return correlation_matrix
    
    def _find_strong_correlations(self, correlation_matrix: np.ndarray, 
                                 metric_names: List[str], threshold: float = 0.8) -> List[Dict]:
        """Find strong correlations in the correlation matrix."""
        strong_correlations = []
        n_metrics = len(metric_names)
        
        for i in range(n_metrics):
            for j in range(i + 1, n_metrics):
                correlation = abs(correlation_matrix[i, j])
                
                if correlation >= threshold:
                    strong_correlations.append({
                        'metric1': metric_names[i],
                        'metric2': metric_names[j],
                        'correlation': correlation,
                        'positive': correlation_matrix[i, j] > 0
                    })
        
        return strong_correlations
    
    def _analyze_correlations(self, correlations: List[Dict]):
        """Analyze strong correlations for insights."""
        for corr in correlations:
            # Log interesting correlations
            if corr['correlation'] > 0.9:
                self.get_logger().info(
                    f"Strong correlation detected: {corr['metric1']} ↔ {corr['metric2']} "
                    f"(r={corr['correlation']:.3f})"
                )
                
                # Generate insight
                insight = self._generate_correlation_insight(corr)
                if insight:
                    self._publish_optimization_recommendation(insight)
    
    def _generate_correlation_insight(self, correlation: Dict) -> Optional[str]:
        """Generate insight from correlation analysis."""
        metric1 = correlation['metric1']
        metric2 = correlation['metric2']
        
        # Example insights based on metric names
        if 'cpu' in metric1 and 'temperature' in metric2:
            return "High CPU usage correlates with increased temperature. Consider optimizing CPU-intensive tasks."
        
        if 'memory' in metric1 and 'swap' in metric2:
            return "High memory usage leads to swap usage. Consider increasing memory or optimizing memory usage."
        
        if 'latency' in metric1 and 'queue' in metric2:
            return "Increased latency correlates with larger queue sizes. Consider optimizing processing pipeline."
        
        return None
    
    def _trend_analysis_loop(self):
        """Background thread for trend analysis."""
        while self.monitoring_active and rclpy.ok():
            try:
                time.sleep(60)  # Run every minute
                
                # Analyze trends
                self._analyze_trends()
                
                # Detect performance degradation
                if self.get_parameter('enable_performance_degradation_detection').value:
                    self._detect_performance_degradation()
                
                # Detect resource leaks
                if self.get_parameter('enable_resource_leak_detection').value:
                    self._detect_resource_leaks()
                
            except Exception as e:
                self.get_logger().error(f"Error in trend analysis loop: {str(e)}")
                time.sleep(10)
    
    def _analyze_trends(self):
        """Analyze performance trends over time."""
        trend_window_hours = self.get_parameter('trend_window_hours').value
        
        for component_type, component in self.components.items():
            for metric_key, values in component.metrics.items():
                if len(values) >= 100:  # Need sufficient data
                    # Extract recent values within trend window
                    cutoff_time = datetime.now() - timedelta(hours=trend_window_hours)
                    recent_values = [v for v in values if v.timestamp > cutoff_time]
                    
                    if len(recent_values) >= 20:
                        # Calculate trend
                        timestamps = [v.timestamp.timestamp() for v in recent_values]
                        metric_values = [v.value for v in recent_values]
                        
                        # Simple linear regression for trend
                        try:
                            slope, intercept = np.polyfit(timestamps, metric_values, 1)
                            
                            # Classify trend
                            if abs(slope) < 0.001:
                                trend = "stable"
                            elif slope > 0:
                                trend = "increasing"
                            else:
                                trend = "decreasing"
                            
                            # Store trend information
                            trend_key = f"{component_type.value}_{metric_key}"
                            self.trend_models[trend_key] = {
                                'slope': slope,
                                'intercept': intercept,
                                'trend': trend,
                                'last_updated': datetime.now()
                            }
                            
                            # Alert for concerning trends
                            self._check_trend_alerts(component_type, metric_key, slope, trend)
                            
                        except Exception as e:
                            continue
    
    def _check_trend_alerts(self, component: ComponentType, metric_key: str, 
                           slope: float, trend: str):
        """Check for concerning trends and generate alerts."""
        concerning_slopes = {
            'cpu_usage': 0.5,  # 0.5% per hour increase is concerning
            'memory_usage': 0.3,
            'latency': 1.0,  # 1ms per hour increase
            'error_rate': 0.1,  # 0.1% per hour increase
        }
        
        for metric_pattern, threshold in concerning_slopes.items():
            if metric_pattern in metric_key:
                if slope > threshold and trend == "increasing":
                    alert_msg = PerformanceAlert()
                    alert_msg.header.stamp = self.get_clock().now().to_msg()
                    alert_msg.header.frame_id = "performance"
                    
                    alert_msg.severity = AlertSeverity.WARNING.value
                    alert_msg.component = component.value
                    alert_msg.metric_type = metric_key
                    alert_msg.current_value = slope * 24  # Projected daily increase
                    alert_msg.threshold_value = threshold * 24
                    alert_msg.message = f"Concerning trend detected: {metric_key} increasing at {slope:.3f}/hour"
                    alert_msg.timestamp = datetime.now().isoformat()
                    
                    context = {
                        'trend': trend,
                        'slope_per_hour': slope,
                        'projected_daily_increase': slope * 24,
                        'threshold': threshold
                    }
                    alert_msg.context_json = json.dumps(context)
                    
                    self.alerts_pub.publish(alert_msg)
                    
                    self.get_logger().warn(f"TREND ALERT: {alert_msg.message}")
                break
    
    def _detect_performance_degradation(self):
        """Detect gradual performance degradation."""
        degradation_threshold = self.get_parameter('degradation_threshold_percent').value
        
        for component_type, component in self.components.items():
            baseline = component.baseline
            
            for metric_key in baseline.keys():
                if metric_key in component.metrics and component.metrics[metric_key]:
                    # Get baseline and current values
                    baseline_value = baseline[metric_key]
                    current_values = component.get_recent_values(
                        MetricType(metric_key.split('_')[0]),  # Extract metric type
                        10  # Last 10 values
                    )
                    
                    if current_values:
                        current_avg = statistics.mean(current_values)
                        
                        # Calculate degradation
                        if baseline_value > 0:
                            degradation = ((current_avg - baseline_value) / baseline_value) * 100
                            
                            if degradation > degradation_threshold:
                                self._create_degradation_alert(
                                    component_type,
                                    metric_key,
                                    baseline_value,
                                    current_avg,
                                    degradation
                                )
    
    def _create_degradation_alert(self, component: ComponentType, metric_key: str,
                                 baseline: float, current: float, degradation: float):
        """Create alert for performance degradation."""
        alert_msg = PerformanceAlert()
        alert_msg.header.stamp = self.get_clock().now().to_msg()
        alert_msg.header.frame_id = "performance"
        
        alert_msg.severity = AlertSeverity.WARNING.value
        alert_msg.component = component.value
        alert_msg.metric_type = metric_key
        alert_msg.current_value = current
        alert_msg.threshold_value = baseline * (1 + self.get_parameter('degradation_threshold_percent').value / 100)
        alert_msg.message = f"Performance degradation detected: {metric_key} degraded by {degradation:.1f}%"
        alert_msg.timestamp = datetime.now().isoformat()
        
        context = {
            'baseline': baseline,
            'current': current,
            'degradation_percent': degradation,
            'threshold_percent': self.get_parameter('degradation_threshold_percent').value
        }
        alert_msg.context_json = json.dumps(context)
        
        self.alerts_pub.publish(alert_msg)
        
        self.get_logger().warn(f"DEGRADATION ALERT: {alert_msg.message}")
    
    def _detect_resource_leaks(self):
        """Detect resource leaks (memory, file descriptors, etc.)."""
        leak_window = self.get_parameter('leak_detection_window').value
        
        # Check memory leak for this process
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # Store memory usage history
        if not hasattr(self, '_memory_history'):
            self._memory_history = deque(maxlen=leak_window)
        
        self._memory_history.append(memory_info.rss)
        
        # Check for increasing trend (potential leak)
        if len(self._memory_history) >= leak_window // 2:
            # Calculate trend
            x = list(range(len(self._memory_history)))
            y = list(self._memory_history)
            
            try:
                slope, _ = np.polyfit(x, y, 1)
                
                # If memory is consistently increasing
                if slope > 1024 * 1024:  # More than 1MB per sample increase
                    alert_msg = PerformanceAlert()
                    alert_msg.header.stamp = self.get_clock().now().to_msg()
                    alert_msg.header.frame_id = "performance"
                    
                    alert_msg.severity = AlertSeverity.WARNING.value
                    alert_msg.component = "system"
                    alert_msg.metric_type = "memory_usage"
                    alert_msg.current_value = memory_info.rss / (1024 * 1024)  # MB
                    alert_msg.threshold_value = (self._memory_history[0] / (1024 * 1024)) * 1.2  # 20% increase
                    alert_msg.message = f"Potential memory leak detected: increasing at {slope/(1024*1024):.1f} MB/sample"
                    alert_msg.timestamp = datetime.now().isoformat()
                    
                    context = {
                        'current_mb': memory_info.rss / (1024 * 1024),
                        'trend_mb_per_sample': slope / (1024 * 1024),
                        'window_size': len(self._memory_history)
                    }
                    alert_msg.context_json = json.dumps(context)
                    
                    self.alerts_pub.publish(alert_msg)
                    
                    self.get_logger().warn(f"RESOURCE LEAK ALERT: {alert_msg.message}")
                    
            except Exception:
                pass
    
    def _predictive_analysis_loop(self):
        """Background thread for predictive analysis."""
        while self.monitoring_active and rclpy.ok():
            try:
                time.sleep(300)  # Run every 5 minutes
                
                # Update prediction models
                self._update_prediction_models()
                
                # Generate predictions
                self._generate_predictions()
                
                # Check for capacity issues
                if self.get_parameter('enable_capacity_planning').value:
                    self._check_capacity_issues()
                
            except Exception as e:
                self.get_logger().error(f"Error in predictive analysis loop: {str(e)}")
                time.sleep(60)
    
    def _update_prediction_models(self):
        """Update prediction models for key metrics."""
        prediction_horizon = self.get_parameter('prediction_horizon_minutes').value
        
        # Key metrics to predict
        key_metrics = [
            ('system_cpu_usage', MetricType.CPU_USAGE, ComponentType.SYSTEM),
            ('system_memory_usage', MetricType.MEMORY_USAGE, ComponentType.SYSTEM),
            ('perception_latency', MetricType.LATENCY, ComponentType.PERCEPTION),
            ('ml_inference_time', MetricType.INFERENCE_TIME, ComponentType.ML_MODEL)
        ]
        
        for metric_name, metric_type, component in key_metrics:
            if component in self.components:
                values = self.components[component].get_recent_values(metric_type, 200)
                
                if len(values) >= 50:  # Need enough data
                    # Simple exponential smoothing for prediction
                    try:
                        # Calculate predictions
                        predicted_value = self._exponential_smoothing_predict(values, prediction_horizon)
                        
                        # Store prediction
                        prediction_key = f"{metric_name}_prediction"
                        self.prediction_models[prediction_key] = {
                            'predicted_value': predicted_value,
                            'prediction_horizon_minutes': prediction_horizon,
                            'confidence': 0.8,  # Placeholder confidence
                            'last_updated': datetime.now()
                        }
                        
                        # Check if prediction exceeds thresholds
                        self._check_prediction_alerts(component, metric_type, metric_name, predicted_value)
                        
                    except Exception as e:
                        self.get_logger().debug(f"Failed to predict {metric_name}: {str(e)}")
    
    def _exponential_smoothing_predict(self, values: List[float], horizon: int) -> float:
        """Simple exponential smoothing prediction."""
        if not values:
            return 0.0
        
        alpha = 0.3  # Smoothing factor
        forecast = values[0]
        
        for value in values[1:]:
            forecast = alpha * value + (1 - alpha) * forecast
        
        # Simple trend projection (very basic)
        if len(values) >= 10:
            recent_trend = statistics.mean(values[-10:]) - statistics.mean(values[-20:-10])
            forecast += recent_trend * (horizon / 60)  # Project trend over horizon
        
        return forecast
    
    def _check_prediction_alerts(self, component: ComponentType, metric_type: MetricType,
                                metric_name: str, predicted_value: float):
        """Check if predicted values exceed thresholds."""
        if component in self.components:
            thresholds = self.components[component].thresholds
            
            metric_key = metric_type.value
            if metric_key in thresholds:
                warning, error, critical = thresholds[metric_key]
                
                if predicted_value >= critical:
                    alert_level = AlertSeverity.CRITICAL
                elif predicted_value >= error:
                    alert_level = AlertSeverity.ERROR
                elif predicted_value >= warning:
                    alert_level = AlertSeverity.WARNING
                else:
                    return
                
                alert_msg = PerformanceAlert()
                alert_msg.header.stamp = self.get_clock().now().to_msg()
                alert_msg.header.frame_id = "performance"
                
                alert_msg.severity = alert_level.value
                alert_msg.component = component.value
                alert_msg.metric_type = metric_type.value
                alert_msg.current_value = predicted_value
                alert_msg.threshold_value = warning
                alert_msg.message = f"Predicted {metric_name} will exceed threshold: {predicted_value:.1f} > {warning:.1f}"
                alert_msg.timestamp = datetime.now().isoformat()
                
                context = {
                    'prediction_horizon_minutes': self.get_parameter('prediction_horizon_minutes').value,
                    'confidence': 0.8,
                    'is_prediction': True
                }
                alert_msg.context_json = json.dumps(context)
                
                self.alerts_pub.publish(alert_msg)
                
                self.get_logger().warn(f"PREDICTION ALERT: {alert_msg.message}")
    
    def _check_capacity_issues(self):
        """Check for capacity planning issues."""
        capacity_days = self.get_parameter('capacity_prediction_days').value
        
        # Check disk capacity
        disk_usage = psutil.disk_usage('/')
        daily_growth_rate = 0.01  # Placeholder - would be calculated from history
        
        days_until_full = (disk_usage.free / (disk_usage.total * daily_growth_rate)) / (1024**3)
        
        if days_until_full < capacity_days:
            recommendation = OptimizationRecommendation()
            recommendation.header.stamp = self.get_clock().now().to_msg()
            recommendation.header.frame_id = "performance"
            
            recommendation.severity = AlertSeverity.WARNING.value
            recommendation.component = "storage"
            recommendation.metric_type = "disk_usage"
            recommendation.current_value = disk_usage.percent
            recommendation.recommendation = f"Disk will be full in {days_until_full:.1f} days. Consider cleaning up or adding storage."
            recommendation.priority = 1  # High priority
            recommendation.timestamp = datetime.now().isoformat()
            
            context = {
                'current_usage_gb': disk_usage.used / (1024**3),
                'free_space_gb': disk_usage.free / (1024**3),
                'daily_growth_gb': disk_usage.total * daily_growth_rate / (1024**3),
                'days_until_full': days_until_full
            }
            recommendation.context_json = json.dumps(context)
            
            self.recommendations_pub.publish(recommendation)
            
            self.get_logger().warn(f"CAPACITY ALERT: {recommendation.recommendation}")
    
    def _visualization_loop(self):
        """Background thread for performance visualization."""
        while self.monitoring_active and rclpy.ok():
            try:
                time.sleep(self.get_parameter('visualization_update_interval_sec').value)
                
                # Generate visualization markers
                markers = self._generate_performance_markers()
                
                # Publish markers
                if markers:
                    marker_array = MarkerArray()
                    marker_array.markers = markers
                    self.visualization_pub.publish(marker_array)
                
            except Exception as e:
                self.get_logger().error(f"Error in visualization loop: {str(e)}")
                time.sleep(5)
    
    def _generate_performance_markers(self) -> List[Marker]:
        """Generate visualization markers for performance data."""
        markers = []
        
        # Create marker for overall performance state
        state_marker = Marker()
        state_marker.header.frame_id = "performance_viz"
        state_marker.header.stamp = self.get_clock().now().to_msg()
        state_marker.ns = "performance_state"
        state_marker.id = 0
        state_marker.type = Marker.SPHERE
        state_marker.action = Marker.ADD
        
        # Color based on performance state
        state_colors = {
            PerformanceState.OPTIMAL: (0.0, 1.0, 0.0),    # Green
            PerformanceState.NORMAL: (0.5, 1.0, 0.0),     # Yellow-green
            PerformanceState.DEGRADED: (1.0, 0.5, 0.0),   # Orange
            PerformanceState.CRITICAL: (1.0, 0.0, 0.0),   # Red
            PerformanceState.FAILED: (0.5, 0.0, 0.0)      # Dark red
        }
        
        color = state_colors.get(self.overall_state, (1.0, 1.0, 1.0))
        state_marker.color.r = color[0]
        state_marker.color.g = color[1]
        state_marker.color.b = color[2]
        state_marker.color.a = 0.8
        
        state_marker.scale.x = 0.5
        state_marker.scale.y = 0.5
        state_marker.scale.z = 0.5
        
        state_marker.pose.position.x = 0.0
        state_marker.pose.position.y = 0.0
        state_marker.pose.position.z = 1.0
        
        markers.append(state_marker)
        
        # Create markers for component performance
        component_positions = {
            ComponentType.PERCEPTION: (1.0, 0.0, 0.0),
            ComponentType.MOTION_PLANNING: (0.7, 0.7, 0.0),
            ComponentType.CONTROL: (0.0, 1.0, 0.0),
            ComponentType.ML_MODEL: (-0.7, 0.7, 0.0),
            ComponentType.SYSTEM: (0.0, 0.0, 0.0),
            ComponentType.HARDWARE: (-1.0, 0.0, 0.0),
            ComponentType.NETWORK: (-0.7, -0.7, 0.0),
            ComponentType.STORAGE: (0.0, -1.0, 0.0),
            ComponentType.TASK_PLANNING: (0.7, -0.7, 0.0)
        }
        
        for component_type, position in component_positions.items():
            if component_type in self.components:
                component = self.components[component_type]
                
                # Create component marker
                comp_marker = Marker()
                comp_marker.header.frame_id = "performance_viz"
                comp_marker.header.stamp = self.get_clock().now().to_msg()
                comp_marker.ns = f"component_{component_type.value}"
                comp_marker.id = component_type.value.__hash__() % 1000
                comp_marker.type = Marker.CUBE
                comp_marker.action = Marker.ADD
                
                # Size based on recent activity
                recent_metrics = sum(len(values) for values in component.metrics.values())
                size = 0.2 + min(0.3, recent_metrics / 1000)
                
                comp_marker.scale.x = size
                comp_marker.scale.y = size
                comp_marker.scale.z = size
                
                comp_marker.pose.position.x = position[0]
                comp_marker.pose.position.y = position[1]
                comp_marker.pose.position.z = position[2]
                
                # Color based on component state
                comp_color = state_colors.get(component.state, (0.5, 0.5, 0.5))
                comp_marker.color.r = comp_color[0]
                comp_marker.color.g = comp_color[1]
                comp_marker.color.b = comp_color[2]
                comp_marker.color.a = 0.6
                
                markers.append(comp_marker)
                
                # Add text label
                text_marker = Marker()
                text_marker.header.frame_id = "performance_viz"
                text_marker.header.stamp = self.get_clock().now().to_msg()
                text_marker.ns = f"component_label_{component_type.value}"
                text_marker.id = comp_marker.id + 1000
                text_marker.type = Marker.TEXT_VIEW_FACING
                text_marker.action = Marker.ADD
                
                text_marker.text = component_type.value
                text_marker.scale.z = 0.1
                
                text_marker.pose.position.x = position[0]
                text_marker.pose.position.y = position[1]
                text_marker.pose.position.z = position[2] + size + 0.1
                
                text_marker.color.r = 1.0
                text_marker.color.g = 1.0
                text_marker.color.b = 1.0
                text_marker.color.a = 1.0
                
                markers.append(text_marker)
        
        return markers
    
    def _benchmarking_loop(self):
        """Background thread for periodic benchmarking."""
        benchmark_interval = self.get_parameter('benchmark_interval_hours').value * 3600
        
        while self.monitoring_active and rclpy.ok():
            try:
                time.sleep(benchmark_interval)
                
                # Run benchmarks
                benchmark_results = self._run_benchmarks()
                
                # Store benchmark results
                self._store_benchmark_results(benchmark_results)
                
                # Compare with baseline
                self._compare_with_baseline(benchmark_results)
                
                # Generate benchmark report
                self._generate_benchmark_report(benchmark_results)
                
            except Exception as e:
                self.get_logger().error(f"Error in benchmarking loop: {str(e)}")
                time.sleep(3600)  # Retry in 1 hour
    
    def _run_benchmarks(self) -> Dict[str, Any]:
        """Run comprehensive performance benchmarks."""
        benchmarks = {}
        
        try:
            # CPU benchmark
            cpu_scores = self._benchmark_cpu()
            benchmarks['cpu'] = cpu_scores
            
            # Memory benchmark
            memory_scores = self._benchmark_memory()
            benchmarks['memory'] = memory_scores
            
            # Disk benchmark
            disk_scores = self._benchmark_disk()
            benchmarks['disk'] = disk_scores
            
            # Network benchmark
            network_scores = self._benchmark_network()
            benchmarks['network'] = network_scores
            
            # ML inference benchmark
            ml_scores = self._benchmark_ml_inference()
            benchmarks['ml_inference'] = ml_scores
            
            # Perception pipeline benchmark
            perception_scores = self._benchmark_perception()
            benchmarks['perception'] = perception_scores
            
            # Motion planning benchmark
            motion_scores = self._benchmark_motion_planning()
            benchmarks['motion_planning'] = motion_scores
            
            # Overall score
            benchmarks['overall_score'] = self._calculate_overall_benchmark_score(benchmarks)
            
            benchmarks['timestamp'] = datetime.now().isoformat()
            benchmarks['hostname'] = socket.gethostname()
            
        except Exception as e:
            self.get_logger().error(f"Error running benchmarks: {str(e)}")
            benchmarks['error'] = str(e)
        
        return benchmarks
    
    def _benchmark_cpu(self) -> Dict[str, float]:
        """Benchmark CPU performance."""
        scores = {}
        
        try:
            # Calculate CPU operations per second
            start_time = time.time()
            iterations = 1000000
            
            # Simple arithmetic operations
            result = 0
            for i in range(iterations):
                result += i * i
                result %= 1000
            
            end_time = time.time()
            ops_per_sec = iterations / (end_time - start_time)
            
            scores['arithmetic_ops_per_sec'] = ops_per_sec
            scores['single_core_score'] = ops_per_sec / 1000000  # Normalized
            
            # Multi-core test
            import concurrent.futures
            
            def cpu_worker(n):
                total = 0
                for i in range(n):
                    total += i * i
                return total
            
            with ThreadPoolExecutor(max_workers=psutil.cpu_count()) as executor:
                futures = [executor.submit(cpu_worker, 100000) for _ in range(psutil.cpu_count())]
                results = [f.result() for f in futures]
            
            scores['multi_core_score'] = len(results)  # Simple metric
            
        except Exception as e:
            scores['error'] = str(e)
        
        return scores
    
    def _benchmark_memory(self) -> Dict[str, float]:
        """Benchmark memory performance."""
        scores = {}
        
        try:
            # Memory allocation speed
            start_time = time.time()
            
            # Allocate and manipulate large arrays
            array_size = 1000000
            test_array = np.random.rand(array_size)
            
            # Vectorized operations
            test_array = test_array * 2.0
            test_array = np.sin(test_array)
            test_array = np.sqrt(test_array)
            
            end_time = time.time()
            
            scores['allocation_speed_mb_per_sec'] = (array_size * 8) / (end_time - start_time) / (1024 * 1024)
            scores['vectorized_ops_per_sec'] = array_size / (end_time - start_time)
            
        except Exception as e:
            scores['error'] = str(e)
        
        return scores
    
    def _benchmark_disk(self) -> Dict[str, float]:
        """Benchmark disk I/O performance."""
        scores = {}
        
        try:
            import tempfile
            
            # Create temporary file for testing
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                test_file = tmp_file.name
                
                # Write test
                test_data = b'x' * (1024 * 1024)  # 1MB
                
                start_time = time.time()
                with open(test_file, 'wb') as f:
                    for _ in range(10):  # Write 10MB
                        f.write(test_data)
                write_time = time.time() - start_time
                
                # Read test
                start_time = time.time()
                with open(test_file, 'rb') as f:
                    while f.read(1024 * 1024):
                        pass
                read_time = time.time() - start_time
            
            # Clean up
            os.unlink(test_file)
            
            scores['write_speed_mb_per_sec'] = 10.0 / write_time
            scores['read_speed_mb_per_sec'] = 10.0 / read_time
            
        except Exception as e:
            scores['error'] = str(e)
        
        return scores
    
    def _benchmark_network(self) -> Dict[str, float]:
        """Benchmark network performance."""
        scores = {}
        
        try:
            # Simple network test - ping localhost
            import subprocess
            
            result = subprocess.run(
                ['ping', '-c', '4', '127.0.0.1'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                # Parse ping output for average time
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'avg' in line.lower():
                        parts = line.split('/')
                        if len(parts) >= 5:
                            latency = float(parts[4])
                            scores['latency_ms'] = latency
            
            # Bandwidth test (simplified)
            scores['bandwidth_estimated_mbps'] = 1000.0  # Placeholder
            
        except Exception as e:
            scores['error'] = str(e)
        
        return scores
    
    def _benchmark_ml_inference(self) -> Dict[str, float]:
        """Benchmark ML model inference performance."""
        scores = {}
        
        try:
            # This would test actual ML model inference
            # For now, use a placeholder
            scores['inference_time_ms'] = 15.0
            scores['throughput_fps'] = 66.7  # 1000 / 15
            scores['model_size_mb'] = 27.5
            
        except Exception as e:
            scores['error'] = str(e)
        
        return scores
    
    def _benchmark_perception(self) -> Dict[str, float]:
        """Benchmark perception pipeline performance."""
        scores = {}
        
        try:
            # Simulate perception pipeline timing
            scores['object_detection_time_ms'] = 45.0
            scores['tracking_time_ms'] = 10.0
            scores['total_latency_ms'] = 55.0
            scores['frame_rate_fps'] = 18.2  # 1000 / 55
            
        except Exception as e:
            scores['error'] = str(e)
        
        return scores
    
    def _benchmark_motion_planning(self) -> Dict[str, float]:
        """Benchmark motion planning performance."""
        scores = {}
        
        try:
            # Simulate motion planning timing
            scores['planning_time_ms'] = 120.0
            scores['collision_check_time_ms'] = 25.0
            scores['trajectory_generation_time_ms'] = 15.0
            scores['total_planning_time_ms'] = 160.0
            
        except Exception as e:
            scores['error'] = str(e)
        
        return scores
    
    def _calculate_overall_benchmark_score(self, benchmarks: Dict[str, Any]) -> float:
        """Calculate overall benchmark score."""
        weights = {
            'cpu': 0.25,
            'memory': 0.15,
            'disk': 0.15,
            'network': 0.10,
            'ml_inference': 0.15,
            'perception': 0.10,
            'motion_planning': 0.10
        }
        
        overall_score = 0.0
        total_weight = 0.0
        
        for component, weight in weights.items():
            if component in benchmarks and 'error' not in benchmarks[component]:
                # Normalize scores to 0-100 range
                if component == 'cpu':
                    score = benchmarks[component].get('single_core_score', 0) * 100
                elif component == 'memory':
                    score = min(100, benchmarks[component].get('vectorized_ops_per_sec', 0) / 100000)
                elif component == 'disk':
                    score = min(100, benchmarks[component].get('write_speed_mb_per_sec', 0) / 100)
                elif component == 'network':
                    score = min(100, 1000 / max(1, benchmarks[component].get('latency_ms', 100)))
                elif component == 'ml_inference':
                    score = min(100, benchmarks[component].get('throughput_fps', 0) / 10)
                elif component == 'perception':
                    score = min(100, benchmarks[component].get('frame_rate_fps', 0) / 2)
                elif component == 'motion_planning':
                    score = min(100, 1000 / max(1, benchmarks[component].get('total_planning_time_ms', 100)) * 10)
                else:
                    score = 50.0  # Default
                
                overall_score += score * weight
                total_weight += weight
        
        if total_weight > 0:
            return overall_score / total_weight
        else:
            return 50.0
    
    def _store_benchmark_results(self, results: Dict[str, Any]):
        """Store benchmark results to file."""
        try:
            benchmark_dir = self.data_dir / "benchmarks"
            benchmark_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            benchmark_file = benchmark_dir / f"benchmark_{timestamp}.json"
            
            with open(benchmark_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            self.get_logger().info(f"Benchmark results saved to {benchmark_file}")
            
        except Exception as e:
            self.get_logger().error(f"Error storing benchmark results: {str(e)}")
    
    def _compare_with_baseline(self, results: Dict[str, Any]):
        """Compare benchmark results with baseline."""
        if not self.baseline_established:
            return
        
        baseline_file = self.data_dir / "baseline_benchmark.json"
        if not baseline_file.exists():
            return
        
        try:
            with open(baseline_file, 'r') as f:
                baseline = json.load(f)
            
            # Compare overall score
            current_score = results.get('overall_score', 0)
            baseline_score = baseline.get('overall_score', 0)
            
            if baseline_score > 0:
                change_percent = ((current_score - baseline_score) / baseline_score) * 100
                
                if abs(change_percent) > 10:  # Significant change
                    alert_msg = PerformanceAlert()
                    alert_msg.header.stamp = self.get_clock().now().to_msg()
                    alert_msg.header.frame_id = "performance"
                    
                    if change_percent > 0:
                        severity = AlertSeverity.INFO
                        message = f"Benchmark improved by {change_percent:.1f}%"
                    else:
                        severity = AlertSeverity.WARNING
                        message = f"Benchmark degraded by {abs(change_percent):.1f}%"
                    
                    alert_msg.severity = severity.value
                    alert_msg.component = "system"
                    alert_msg.metric_type = "benchmark"
                    alert_msg.current_value = current_score
                    alert_msg.threshold_value = baseline_score
                    alert_msg.message = message
                    alert_msg.timestamp = datetime.now().isoformat()
                    
                    context = {
                        'current_score': current_score,
                        'baseline_score': baseline_score,
                        'change_percent': change_percent
                    }
                    alert_msg.context_json = json.dumps(context)
                    
                    self.alerts_pub.publish(alert_msg)
                    
                    self.get_logger().info(f"BENCHMARK CHANGE: {message}")
        
        except Exception as e:
            self.get_logger().error(f"Error comparing with baseline: {str(e)}")
    
    def _generate_benchmark_report(self, results: Dict[str, Any]):
        """Generate benchmark report."""
        try:
            report_dir = self.data_dir / "reports"
            report_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = report_dir / f"benchmark_report_{timestamp}.md"
            
            with open(report_file, 'w') as f:
                f.write("# Performance Benchmark Report\n\n")
                f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**Hostname:** {results.get('hostname', 'N/A')}\n\n")
                
                f.write("## Overall Score\n\n")
                f.write(f"**Overall Performance Score:** {results.get('overall_score', 0):.1f}/100\n\n")
                
                f.write("## Component Scores\n\n")
                for component, scores in results.items():
                    if component not in ['overall_score', 'timestamp', 'hostname', 'error']:
                        f.write(f"### {component.replace('_', ' ').title()}\n\n")
                        for metric, value in scores.items():
                            if metric != 'error':
                                f.write(f"- **{metric}:** {value}\n")
                        f.write("\n")
                
                f.write("## Recommendations\n\n")
                recommendations = self._generate_benchmark_recommendations(results)
                for rec in recommendations:
                    f.write(f"- {rec}\n")
            
            self.get_logger().info(f"Benchmark report saved to {report_file}")
            
        except Exception as e:
            self.get_logger().error(f"Error generating benchmark report: {str(e)}")
    
    def _generate_benchmark_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on benchmark results."""
        recommendations = []
        
        overall_score = results.get('overall_score', 0)
        
        if overall_score < 50:
            recommendations.append("Overall system performance is below optimal levels. Consider system-wide optimizations.")
        
        # Component-specific recommendations
        if 'cpu' in results:
            cpu_score = results['cpu'].get('single_core_score', 0) * 100
            if cpu_score < 30:
                recommendations.append("CPU performance is low. Consider optimizing CPU-bound tasks or upgrading hardware.")
        
        if 'memory' in results:
            mem_speed = results['memory'].get('vectorized_ops_per_sec', 0)
            if mem_speed < 50000:
                recommendations.append("Memory performance could be improved. Consider memory optimization or faster RAM.")
        
        if 'disk' in results:
            write_speed = results['disk'].get('write_speed_mb_per_sec', 0)
            if write_speed < 100:
                recommendations.append("Disk write speed is slow. Consider using SSD or optimizing disk I/O.")
        
        if 'ml_inference' in results:
            inference_time = results['ml_inference'].get('inference_time_ms', 0)
            if inference_time > 50:
                recommendations.append("ML inference is slower than optimal. Consider model optimization or hardware acceleration.")
        
        if not recommendations:
            recommendations.append("System performance is within optimal parameters.")
        
        return recommendations
    
    def _capacity_planning_loop(self):
        """Background thread for capacity planning."""
        prediction_days = self.get_parameter('capacity_prediction_days').value
        
        while self.monitoring_active and rclpy.ok():
            try:
                time.sleep(3600)  # Run every hour
                
                # Analyze resource usage trends
                usage_trends = self._analyze_resource_usage_trends()
                
                # Predict future capacity needs
                capacity_predictions = self._predict_capacity_needs(usage_trends, prediction_days)
                
                # Generate capacity planning report
                self._generate_capacity_report(capacity_predictions)
                
                # Alert for imminent capacity issues
                self._check_imminent_capacity_issues(capacity_predictions)
                
            except Exception as e:
                self.get_logger().error(f"Error in capacity planning loop: {str(e)}")
                time.sleep(3600)
    
    def _analyze_resource_usage_trends(self) -> Dict[str, Dict]:
        """Analyze trends in resource usage."""
        trends = {}
        
        # Analyze disk usage trend
        disk_usage = psutil.disk_usage('/')
        
        # Get historical disk usage from stored metrics
        disk_metrics = []
        for metric in self.metrics_history:
            if metric.metric_type == MetricType.DISK_IO and metric.component == 'system':
                disk_metrics.append(metric)
        
        if len(disk_metrics) >= 10:
            # Calculate daily growth rate
            oldest = min(disk_metrics, key=lambda x: x.timestamp)
            newest = max(disk_metrics, key=lambda x: x.timestamp)
            
            time_diff = (newest.timestamp - oldest.timestamp).total_seconds() / (24 * 3600)  # days
            if time_diff > 0:
                # Estimate growth (simplified)
                growth_rate = (disk_usage.used - oldest.value) / time_diff
                trends['disk'] = {
                    'current_gb': disk_usage.used / (1024**3),
                    'growth_rate_gb_per_day': growth_rate / (1024**3),
                    'free_gb': disk_usage.free / (1024**3)
                }
        
        # Analyze memory usage trend
        memory = psutil.virtual_memory()
        trends['memory'] = {
            'current_percent': memory.percent,
            'current_gb': memory.used / (1024**3),
            'total_gb': memory.total / (1024**3)
        }
        
        # Analyze CPU usage trend
        cpu_percent = psutil.cpu_percent(interval=1)
        trends['cpu'] = {
            'current_percent': cpu_percent,
            'cores': psutil.cpu_count()
        }
        
        return trends
    
    def _predict_capacity_needs(self, trends: Dict[str, Dict], prediction_days: int) -> Dict[str, Dict]:
        """Predict future capacity needs."""
        predictions = {}
        
        # Disk capacity prediction
        if 'disk' in trends:
            disk_trend = trends['disk']
            growth_rate = disk_trend.get('growth_rate_gb_per_day', 0.1)  # Default 0.1 GB/day
            
            days_until_full = disk_trend['free_gb'] / growth_rate if growth_rate > 0 else float('inf')
            
            predictions['disk'] = {
                'current_usage_gb': disk_trend['current_gb'],
                'growth_rate_gb_per_day': growth_rate,
                'predicted_usage_in_days': disk_trend['current_gb'] + (growth_rate * prediction_days),
                'days_until_full': days_until_full,
                'recommended_action': 'monitor' if days_until_full > 30 else 'expand'
            }
        
        # Memory prediction
        if 'memory' in trends:
            memory_trend = trends['memory']
            
            # Simple linear projection based on current usage
            predicted_usage = memory_trend['current_gb'] * 1.05  # 5% buffer
            
            predictions['memory'] = {
                'current_usage_gb': memory_trend['current_gb'],
                'total_gb': memory_trend['total_gb'],
                'predicted_usage_gb': predicted_usage,
                'headroom_gb': memory_trend['total_gb'] - predicted_usage,
                'recommended_action': 'adequate' if predicted_usage < memory_trend['total_gb'] * 0.8 else 'monitor'
            }
        
        # CPU prediction
        if 'cpu' in trends:
            cpu_trend = trends['cpu']
            
            predictions['cpu'] = {
                'current_usage_percent': cpu_trend['current_percent'],
                'cores': cpu_trend['cores'],
                'recommended_action': 'adequate' if cpu_trend['current_percent'] < 70 else 'optimize'
            }
        
        return predictions
    
    def _generate_capacity_report(self, predictions: Dict[str, Dict]):
        """Generate capacity planning report."""
        try:
            report_dir = self.data_dir / "capacity_reports"
            report_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = report_dir / f"capacity_report_{timestamp}.md"
            
            with open(report_file, 'w') as f:
                f.write("# Capacity Planning Report\n\n")
                f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**Prediction Horizon:** {self.get_parameter('capacity_prediction_days').value} days\n\n")
                
                f.write("## Resource Predictions\n\n")
                
                for resource, pred in predictions.items():
                    f.write(f"### {resource.title()}\n\n")
                    
                    for key, value in pred.items():
                        if isinstance(value, float):
                            f.write(f"- **{key}:** {value:.2f}\n")
                        else:
                            f.write(f"- **{key}:** {value}\n")
                    
                    f.write("\n")
                
                f.write("## Recommendations\n\n")
                recommendations = self._generate_capacity_recommendations(predictions)
                for rec in recommendations:
                    f.write(f"- {rec}\n")
            
            self.get_logger().info(f"Capacity report saved to {report_file}")
            
        except Exception as e:
            self.get_logger().error(f"Error generating capacity report: {str(e)}")
    
    def _generate_capacity_recommendations(self, predictions: Dict[str, Dict]) -> List[str]:
        """Generate capacity planning recommendations."""
        recommendations = []
        
        if 'disk' in predictions:
            disk_pred = predictions['disk']
            if disk_pred['days_until_full'] < 30:
                recommendations.append(
                    f"Disk space will be exhausted in {disk_pred['days_until_full']:.1f} days. "
                    f"Consider adding storage or cleaning up old data."
                )
            elif disk_pred['days_until_full'] < 90:
                recommendations.append(
                    f"Disk space will be tight in {disk_pred['days_until_full']:.1f} days. "
                    f"Monitor usage closely."
                )
        
        if 'memory' in predictions:
            mem_pred = predictions['memory']
            if mem_pred['headroom_gb'] < 1.0:  # Less than 1GB headroom
                recommendations.append(
                    f"Memory headroom is low ({mem_pred['headroom_gb']:.1f} GB). "
                    f"Consider optimizing memory usage or adding more RAM."
                )
        
        if 'cpu' in predictions:
            cpu_pred = predictions['cpu']
            if cpu_pred['current_usage_percent'] > 80:
                recommendations.append(
                    f"CPU usage is high ({cpu_pred['current_usage_percent']:.1f}%). "
                    f"Consider optimizing CPU-intensive tasks or upgrading CPU."
                )
        
        if not recommendations:
            recommendations.append("System capacity is adequate for current and predicted usage.")
        
        return recommendations
    
    def _check_imminent_capacity_issues(self, predictions: Dict[str, Dict]):
        """Check for imminent capacity issues and generate alerts."""
        if 'disk' in predictions:
            disk_pred = predictions['disk']
            if disk_pred['days_until_full'] < 7:  # Critical: less than 1 week
                alert_msg = PerformanceAlert()
                alert_msg.header.stamp = self.get_clock().now().to_msg()
                alert_msg.header.frame_id = "performance"
                
                alert_msg.severity = AlertSeverity.CRITICAL.value
                alert_msg.component = "storage"
                alert_msg.metric_type = "disk_capacity"
                alert_msg.current_value = disk_pred['current_usage_gb']
                alert_msg.threshold_value = disk_pred['current_usage_gb'] + disk_pred['free_gb']
                alert_msg.message = f"CRITICAL: Disk will be full in {disk_pred['days_until_full']:.1f} days!"
                alert_msg.timestamp = datetime.now().isoformat()
                
                context = {
                    'current_gb': disk_pred['current_usage_gb'],
                    'free_gb': disk_pred['free_gb'],
                    'growth_rate_gb_per_day': disk_pred['growth_rate_gb_per_day'],
                    'days_until_full': disk_pred['days_until_full']
                }
                alert_msg.context_json = json.dumps(context)
                
                self.alerts_pub.publish(alert_msg)
                
                self.get_logger().error(f"CAPACITY CRITICAL: {alert_msg.message}")
    
    def _publish_performance_metrics(self):
        """Publish performance metrics to ROS2 topic."""
        try:
            metrics_msg = PerformanceMetrics()
            metrics_msg.header.stamp = self.get_clock().now().to_msg()
            metrics_msg.header.frame_id = "performance"
            
            metrics_msg.timestamp = datetime.now().isoformat()
            metrics_msg.overall_state = self.overall_state.value
            
            # Add component metrics
            for component_type, component in self.components.items():
                comp_perf = ComponentPerformance()
                comp_perf.component = component_type.value
                comp_perf.state = component.state.value
                comp_perf.last_update = component.last_update.isoformat()
                
                # Add recent metrics
                for metric_key, values in component.metrics.items():
                    if values:
                        latest = values[-1]
                        resource_msg = ResourceUsage()
                        resource_msg.metric_type = latest.metric_type.value
                        resource_msg.value = latest.value
                        resource_msg.unit = latest.unit
                        resource_msg.timestamp = latest.timestamp.isoformat()
                        
                        comp_perf.metrics.append(resource_msg)
                
                metrics_msg.components.append(comp_perf)
            
            self.metrics_pub.publish(metrics_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing performance metrics: {str(e)}")
    
    def _publish_system_performance(self):
        """Publish system-wide performance summary."""
        try:
            system_msg = SystemPerformance()
            system_msg.header.stamp = self.get_clock().now().to_msg()
            system_msg.header.frame_id = "performance"
            
            system_msg.timestamp = datetime.now().isoformat()
            system_msg.overall_state = self.overall_state.value
            
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            system_msg.cpu_usage = cpu_percent
            system_msg.memory_usage = memory.percent
            system_msg.memory_total_mb = memory.total / (1024 * 1024)
            system_msg.memory_used_mb = memory.used / (1024 * 1024)
            
            # Get disk usage
            disk = psutil.disk_usage('/')
            system_msg.disk_usage = disk.percent
            system_msg.disk_total_gb = disk.total / (1024**3)
            system_msg.disk_used_gb = disk.used / (1024**3)
            
            # Get network stats
            net_io = psutil.net_io_counters()
            system_msg.network_sent_mb = net_io.bytes_sent / (1024 * 1024)
            system_msg.network_recv_mb = net_io.bytes_recv / (1024 * 1024)
            
            # Get process count
            system_msg.process_count = len(psutil.pids())
            
            # Get system temperature
            temp = self._get_system_temperature()
            if temp:
                system_msg.temperature = temp
            
            # Get uptime
            system_msg.uptime_seconds = time.time() - psutil.boot_time()
            
            self.system_perf_pub.publish(system_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing system performance: {str(e)}")
    
    def _publish_optimization_recommendation(self, recommendation: str):
        """Publish an optimization recommendation."""
        rec_msg = OptimizationRecommendation()
        rec_msg.header.stamp = self.get_clock().now().to_msg()
        rec_msg.header.frame_id = "performance"
        
        rec_msg.severity = AlertSeverity.INFO.value
        rec_msg.component = "system"
        rec_msg.metric_type = "optimization"
        rec_msg.current_value = 0.0
        rec_msg.recommendation = recommendation
        rec_msg.priority = 2  # Medium priority
        rec_msg.timestamp = datetime.now().isoformat()
        
        self.recommendations_pub.publish(rec_msg)
    
    def _log_performance_data(self):
        """Log performance data to file."""
        try:
            log_dir = self.data_dir / "logs"
            log_dir.mkdir(exist_ok=True)
            
            log_file = log_dir / f"performance_{datetime.now().strftime('%Y%m%d')}.jsonl"
            
            # Create log entry
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'overall_state': self.overall_state.name,
                'system': {
                    'cpu_percent': psutil.cpu_percent(interval=0.1),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_percent': psutil.disk_usage('/').percent
                },
                'components': {},
                'alerts_last_hour': len([a for a in self.alerts_history 
                                       if (datetime.now() - a['timestamp']).total_seconds() < 3600])
            }
            
            # Add component states
            for component_type, component in self.components.items():
                component_entry = {
                    'state': component.state.name,
                    'last_update': component.last_update.isoformat()
                }
                
                # Add recent metric values
                metrics_summary = {}
                for metric_key, values in component.metrics.items():
                    if values:
                        latest = values[-1]
                        metrics_summary[metric_key] = {
                            'value': latest.value,
                            'unit': latest.unit
                        }
                
                component_entry['metrics'] = metrics_summary
                log_entry['components'][component_type.value] = component_entry
            
            # Write to log file
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
            
        except Exception as e:
            self.get_logger().error(f"Error logging performance data: {str(e)}")
    
    def _update_baseline(self):
        """Update performance baseline from historical data."""
        try:
            if not self.baseline_established:
                # Need at least 24 hours of data
                oldest_metric = min(self.metrics_history, key=lambda x: x.timestamp) if self.metrics_history else None
                if oldest_metric and (datetime.now() - oldest_metric.timestamp).total_seconds() > 24 * 3600:
                    self._calculate_baseline()
                    self.baseline_established = True
            else:
                # Update baseline weekly
                baseline_file = self.data_dir / "performance_baseline.json"
                if baseline_file.exists():
                    baseline_age = datetime.now() - datetime.fromtimestamp(baseline_file.stat().st_mtime)
                    if baseline_age.days >= 7:  # Update weekly
                        self._calculate_baseline()
        
        except Exception as e:
            self.get_logger().error(f"Error updating baseline: {str(e)}")
    
    def _calculate_baseline(self):
        """Calculate performance baseline from historical data."""
        try:
            baseline = {}
            
            # Calculate baselines for each component and metric
            for component_type, component in self.components.items():
                component_baseline = {}
                
                for metric_key, values in component.metrics.items():
                    if len(values) >= 100:  # Need sufficient data
                        metric_values = [v.value for v in values]
                        
                        # Calculate statistics
                        component_baseline[metric_key] = {
                            'mean': statistics.mean(metric_values),
                            'median': statistics.median(metric_values),
                            'std': statistics.stdev(metric_values) if len(metric_values) > 1 else 0.0,
                            'p95': np.percentile(metric_values, 95) if len(metric_values) > 1 else metric_values[0],
                            'samples': len(metric_values)
                        }
                
                baseline[component_type.value] = component_baseline
            
            # Save baseline to file
            baseline_file = self.data_dir / "performance_baseline.json"
            with open(baseline_file, 'w') as f:
                json.dump(baseline, f, indent=2)
            
            # Also update component baselines
            for component_type, component in self.components.items():
                if component_type.value in baseline:
                    for metric_key, stats in baseline[component_type.value].items():
                        component.baseline[metric_key] = stats['mean']
            
            self.get_logger().info("Performance baseline updated")
            
        except Exception as e:
            self.get_logger().error(f"Error calculating baseline: {str(e)}")
    
    def _load_baseline(self):
        """Load performance baseline from file."""
        try:
            baseline_file = self.data_dir / "performance_baseline.json"
            if baseline_file.exists():
                with open(baseline_file, 'r') as f:
                    baseline = json.load(f)
                
                # Load into components
                for component_type, component in self.components.items():
                    if component_type.value in baseline:
                        for metric_key, stats in baseline[component_type.value].items():
                            component.baseline[metric_key] = stats['mean']
                
                self.baseline_established = True
                self.get_logger().info("Performance baseline loaded from file")
                
        except Exception as e:
            self.get_logger().error(f"Error loading baseline: {str(e)}")
    
    def _evaluate_system_state(self):
        """Evaluate and update system performance state."""
        try:
            # Calculate health score based on critical metrics
            health_score = 100.0
            
            # Check CPU
            cpu_percent = psutil.cpu_percent(interval=0.1)
            if cpu_percent > 90:
                health_score -= 30
            elif cpu_percent > 70:
                health_score -= 15
            
            # Check memory
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                health_score -= 30
            elif memory.percent > 70:
                health_score -= 15
            
            # Check disk
            disk = psutil.disk_usage('/')
            if disk.percent > 90:
                health_score -= 20
            elif disk.percent > 70:
                health_score -= 10
            
            # Check temperature
            temp = self._get_system_temperature()
            if temp and temp > 80:
                health_score -= 25
            elif temp and temp > 70:
                health_score -= 10
            
            # Map score to state
            if health_score >= 90:
                new_state = PerformanceState.OPTIMAL
            elif health_score >= 70:
                new_state = PerformanceState.NORMAL
            elif health_score >= 50:
                new_state = PerformanceState.DEGRADED
            elif health_score >= 30:
                new_state = PerformanceState.CRITICAL
            else:
                new_state = PerformanceState.FAILED
            
            # Update if changed
            if new_state != self.overall_state:
                old_state = self.overall_state
                self.overall_state = new_state
                self.last_state_change = datetime.now()
                
                self.get_logger().info(
                    f"System performance state updated: {old_state.name} -> {new_state.name} "
                    f"(health score: {health_score:.1f})"
                )
        
        except Exception as e:
            self.get_logger().error(f"Error evaluating system state: {str(e)}")
    
    def _cleanup_old_data(self):
        """Clean up old performance data."""
        try:
            retention_days = self.get_parameter('performance_baseline_days').value
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            # Clean up old log files
            log_dir = self.data_dir / "logs"
            if log_dir.exists():
                for log_file in log_dir.iterdir():
                    if log_file.is_file():
                        file_date = datetime.fromtimestamp(log_file.stat().st_mtime)
                        if file_date < cutoff_date:
                            log_file.unlink()
            
            # Clean up old benchmark files
            benchmark_dir = self.data_dir / "benchmarks"
            if benchmark_dir.exists():
                for bench_file in benchmark_dir.iterdir():
                    if bench_file.is_file():
                        file_date = datetime.fromtimestamp(bench_file.stat().st_mtime)
                        if file_date < cutoff_date:
                            bench_file.unlink()
            
            # Clean up old reports
            report_dir = self.data_dir / "reports"
            if report_dir.exists():
                for report_file in report_dir.iterdir():
                    if report_file.is_file():
                        file_date = datetime.fromtimestamp(report_file.stat().st_mtime)
                        if file_date < cutoff_date:
                            report_file.unlink()
            
            # Check total storage usage
            total_size = 0
            for file_path in self.data_dir.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            
            total_size_gb = total_size / (1024**3)
            max_storage_gb = self.get_parameter('max_storage_gb').value
            
            if total_size_gb > max_storage_gb:
                self.get_logger().warn(
                    f"Performance data storage ({total_size_gb:.1f} GB) exceeds limit "
                    f"({max_storage_gb} GB). Consider cleaning up old data."
                )
        
        except Exception as e:
            self.get_logger().error(f"Error cleaning up old data: {str(e)}")
    
    def _generate_performance_report(self):
        """Generate periodic performance report."""
        try:
            report_dir = self.data_dir / "periodic_reports"
            report_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = report_dir / f"performance_report_{timestamp}.md"
            
            with open(report_file, 'w') as f:
                f.write("# System Performance Report\n\n")
                f.write(f"**Period:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**Overall State:** {self.overall_state.name}\n\n")
                
                f.write("## System Summary\n\n")
                
                # System metrics
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                f.write(f"- **CPU Usage:** {cpu_percent:.1f}%\n")
                f.write(f"- **Memory Usage:** {memory.percent:.1f}% ({memory.used/(1024**3):.1f} GB / {memory.total/(1024**3):.1f} GB)\n")
                f.write(f"- **Disk Usage:** {disk.percent:.1f}% ({disk.used/(1024**3):.1f} GB / {disk.total/(1024**3):.1f} GB)\n")
                
                temp = self._get_system_temperature()
                if temp:
                    f.write(f"- **Temperature:** {temp:.1f}°C\n")
                
                f.write(f"- **Uptime:** {timedelta(seconds=time.time() - psutil.boot_time())}\n\n")
                
                f.write("## Component Performance\n\n")
                
                for component_type, component in self.components.items():
                    f.write(f"### {component_type.value.replace('_', ' ').title()}\n\n")
                    f.write(f"- **State:** {component.state.name}\n")
                    f.write(f"- **Last Update:** {component.last_update.strftime('%H:%M:%S')}\n")
                    
                    # Top metrics
                    metric_count = 0
                    for metric_key, values in component.metrics.items():
                        if values and metric_count < 3:  # Show top 3 metrics
                            latest = values[-1]
                            f.write(f"- **{metric_key}:** {latest.value:.1f} {latest.unit}\n")
                            metric_count += 1
                    
                    f.write("\n")
                
                f.write("## Recent Alerts\n\n")
                
                recent_alerts = [a for a in self.alerts_history 
                               if (datetime.now() - a['timestamp']).total_seconds() < 3600]
                
                if recent_alerts:
                    for alert in recent_alerts[-10:]:  # Last 10 alerts
                        f.write(f"- **{alert['timestamp'].strftime('%H:%M:%S')}** [{alert['severity'].name}] "
                               f"{alert['component']}.{alert['metric_type']}: {alert['value']}\n")
                else:
                    f.write("No recent alerts.\n")
                
                f.write("\n## Recommendations\n\n")
                
                recommendations = self._generate_report_recommendations()
                for rec in recommendations:
                    f.write(f"- {rec}\n")
            
            self.get_logger().info(f"Performance report saved to {report_file}")
            
        except Exception as e:
            self.get_logger().error(f"Error generating performance report: {str(e)}")
    
    def _generate_report_recommendations(self) -> List[str]:
        """Generate recommendations for the performance report."""
        recommendations = []
        
        # Check system metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        if cpu_percent > 80:
            recommendations.append(f"CPU usage is high ({cpu_percent:.1f}%). Consider optimizing CPU-intensive tasks.")
        
        memory = psutil.virtual_memory()
        if memory.percent > 85:
            recommendations.append(f"Memory usage is high ({memory.percent:.1f}%). Consider optimizing memory usage or adding more RAM.")
        
        disk = psutil.disk_usage('/')
        if disk.percent > 90:
            recommendations.append(f"Disk usage is high ({disk.percent:.1f}%). Consider cleaning up old data or adding storage.")
        
        temp = self._get_system_temperature()
        if temp and temp > 75:
            recommendations.append(f"System temperature is high ({temp:.1f}°C). Check cooling system.")
        
        # Check component states
        degraded_components = []
        for component_type, component in self.components.items():
            if component.state in [PerformanceState.DEGRADED, PerformanceState.CRITICAL, PerformanceState.FAILED]:
                degraded_components.append(component_type.value)
        
        if degraded_components:
            recommendations.append(f"Components needing attention: {', '.join(degraded_components)}")
        
        # Check for recent alerts
        recent_alerts = len([a for a in self.alerts_history 
                           if (datetime.now() - a['timestamp']).total_seconds() < 3600])
        
        if recent_alerts > 10:
            recommendations.append(f"High alert volume ({recent_alerts} in last hour). Investigate root causes.")
        
        if not recommendations:
            recommendations.append("System performance is within optimal parameters.")
        
        return recommendations
    
    # ROS2 Subscriber callbacks
    def _handle_robot_status(self, msg: RobotStatus):
        """Handle robot status messages for performance monitoring."""
        try:
            # Extract performance metrics from robot status
            self._add_metric(
                component=ComponentType.CONTROL,
                metric_type=MetricType.RESPONSE_TIME,
                value=0.0,  # Placeholder - would be actual latency
                unit="milliseconds",
                tags=["robot", "control", "status"],
                metadata={'status': msg.status, 'operation_mode': msg.operation_mode}
            )
            
        except Exception as e:
            self.get_logger().error(f"Error handling robot status: {str(e)}")
    
    def _handle_detected_objects(self, msg: DetectedObject):
        """Handle detected object messages for perception performance."""
        try:
            # Track perception latency
            current_time = self.get_clock().now().seconds_nanoseconds()
            msg_time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            latency = current_time[0] + current_time[1] / 1e9 - msg_time
            
            self._add_metric(
                component=ComponentType.PERCEPTION,
                metric_type=MetricType.LATENCY,
                value=latency * 1000,  # Convert to milliseconds
                unit="milliseconds",
                tags=["perception", "object_detection", msg.class_name],
                metadata={'confidence': msg.confidence, 'class_id': msg.class_id}
            )
            
        except Exception as e:
            self.get_logger().error(f"Error handling detected objects: {str(e)}")
    
    def _handle_diagnostics(self, msg: DiagnosticArray):
        """Handle diagnostic messages for performance monitoring."""
        try:
            for status in msg.status:
                # Extract performance metrics from diagnostics
                if 'cpu' in status.name.lower():
                    for kv in status.values:
                        if 'usage' in kv.key.lower():
                            try:
                                cpu_value = float(kv.value.replace('%', ''))
                                self._add_metric(
                                    component=ComponentType.SYSTEM,
                                    metric_type=MetricType.CPU_USAGE,
                                    value=cpu_value,
                                    unit="percent",
                                    tags=["diagnostics", "cpu"]
                                )
                            except ValueError:
                                pass
                
                elif 'memory' in status.name.lower():
                    for kv in status.values:
                        if 'usage' in kv.key.lower():
                            try:
                                mem_value = float(kv.value.replace('%', ''))
                                self._add_metric(
                                    component=ComponentType.SYSTEM,
                                    metric_type=MetricType.MEMORY_USAGE,
                                    value=mem_value,
                                    unit="percent",
                                    tags=["diagnostics", "memory"]
                                )
                            except ValueError:
                                pass
        
        except Exception as e:
            self.get_logger().error(f"Error handling diagnostics: {str(e)}")
    
    # ROS2 Service handlers
    def _handle_get_metrics(self, request: GetPerformanceMetrics.Request, 
                          response: GetPerformanceMetrics.Response) -> GetPerformanceMetrics.Response:
        """Handle get performance metrics service request."""
        try:
            component_str = request.component
            metric_type_str = request.metric_type
            start_time = datetime.fromisoformat(request.start_time) if request.start_time else None
            end_time = datetime.fromisoformat(request.end_time) if request.end_time else None
            limit = request.limit if request.limit > 0 else 1000
            
            # Filter metrics
            filtered_metrics = []
            
            for metric in self.metrics_history:
                # Apply filters
                if component_str and metric.component != component_str:
                    continue
                
                if metric_type_str and metric.metric_type.value != metric_type_str:
                    continue
                
                if start_time and metric.timestamp < start_time:
                    continue
                
                if end_time and metric.timestamp > end_time:
                    continue
                
                filtered_metrics.append(metric)
                
                if len(filtered_metrics) >= limit:
                    break
            
            # Create response
            response.success = True
            response.metrics = [json.dumps(m.to_dict()) for m in filtered_metrics]
            response.count = len(filtered_metrics)
            
            # Add statistics if requested
            if request.include_statistics and filtered_metrics:
                values = [m.value for m in filtered_metrics]
                response.statistics_json = json.dumps({
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'min': min(values),
                    'max': max(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0.0
                })
            
        except Exception as e:
            response.success = False
            response.error_message = str(e)
        
        return response
    
    def _handle_start_profiling(self, request: StartProfiling.Request,
                               response: StartProfiling.Response) -> StartProfiling.Response:
        """Handle start profiling service request."""
        try:
            profile_id = request.profile_id or f"profile_{int(time.time())}"
            component_str = request.component
            duration = request.duration_sec if request.duration_sec > 0 else 60
            
            # Start profiling
            profiler = cProfile.Profile()
            profiler.enable()
            
            # Store profiler
            self.active_profilers[profile_id] = {
                'profiler': profiler,
                'start_time': datetime.now(),
                'component': component_str,
                'duration': duration
            }
            
            response.success = True
            response.profile_id = profile_id
            response.message = f"Profiling started for {component_str} with ID: {profile_id}"
            
            # Schedule stop if duration specified
            if duration > 0:
                self.create_timer(duration, lambda: self._stop_profiling_by_id(profile_id))
            
        except Exception as e:
            response.success = False
            response.message = str(e)
        
        return response
    
    def _stop_profiling_by_id(self, profile_id: str):
        """Stop profiling by profile ID."""
        if profile_id in self.active_profilers:
            self._stop_profiling_internal(profile_id)
    
    def _stop_profiling_internal(self, profile_id: str) -> Optional[Dict]:
        """Internal method to stop profiling and get results."""
        if profile_id not in self.active_profilers:
            return None
        
        try:
            profiler_data = self.active_profilers[profile_id]
            profiler = profiler_data['profiler']
            component_str = profiler_data['component']
            
            # Stop profiler
            profiler.disable()
            
            # Get statistics
            stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stream).sort_stats('cumulative')
            stats.print_stats(20)  # Top 20 functions
            
            stats_output = stream.getvalue()
            
            # Create performance profile
            profile = PerformanceProfile(
                profile_id=profile_id,
                start_time=profiler_data['start_time'],
                end_time=datetime.now(),
                component=ComponentType(component_str) if component_str else ComponentType.SYSTEM,
                metrics={},
                hotspots=self._extract_hotspots_from_stats(stats),
                recommendations=[],
                summary=self._create_profile_summary(stats)
            )
            
            # Store profile
            self.performance_profiles[profile_id] = profile
            
            # Clean up
            del self.active_profilers[profile_id]
            
            return {
                'profile_id': profile_id,
                'stats': stats_output,
                'profile': profile.to_dict()
            }
            
        except Exception as e:
            self.get_logger().error(f"Error stopping profiler {profile_id}: {str(e)}")
            return None
    
    def _extract_hotspots_from_stats(self, stats: pstats.Stats) -> List[Dict[str, Any]]:
        """Extract performance hotspots from profiling statistics."""
        hotspots = []
        
        try:
            # Get top functions by cumulative time
            stats_list = stats.sort_stats('cumulative').get_stats_profile().func_profiles
            
            for func_profile in list(stats_list.values())[:10]:  # Top 10
                hotspot = {
                    'function': func_profile.func_name,
                    'file': func_profile.file_name,
                    'line': func_profile.line_number,
                    'cumulative_time': func_profile.cumulative_time,
                    'self_time': func_profile.self_time,
                    'call_count': func_profile.call_count
                }
                hotspots.append(hotspot)
        
        except Exception as e:
            self.get_logger().error(f"Error extracting hotspots: {str(e)}")
        
        return hotspots
    
    def _create_profile_summary(self, stats: pstats.Stats) -> Dict[str, float]:
        """Create summary from profiling statistics."""
        summary = {}
        
        try:
            stats_dict = stats.get_stats_profile()
            
            total_calls = 0
            total_time = 0.0
            
            for func_profile in stats_dict.func_profiles.values():
                total_calls += func_profile.call_count
                total_time += func_profile.cumulative_time
            
            summary = {
                'total_calls': total_calls,
                'total_time_seconds': total_time,
                'calls_per_second': total_calls / max(0.001, total_time),
                'average_call_time_ms': (total_time / max(1, total_calls)) * 1000
            }
        
        except Exception as e:
            self.get_logger().error(f"Error creating profile summary: {str(e)}")
            summary = {'error': str(e)}
        
        return summary
    
    def _handle_stop_profiling(self, request: StopProfiling.Request,
                              response: StopProfiling.Response) -> StopProfiling.Response:
        """Handle stop profiling service request."""
        try:
            profile_id = request.profile_id
            
            if profile_id not in self.active_profilers:
                response.success = False
                response.message = f"No active profiler with ID: {profile_id}"
                return response
            
            # Stop profiling
            result = self._stop_profiling_internal(profile_id)
            
            if result:
                response.success = True
                response.profile_id = profile_id
                response.stats_output = result['stats']
                response.profile_json = json.dumps(result['profile'])
                response.message = f"Profiling stopped for {profile_id}"
            else:
                response.success = False
                response.message = f"Failed to stop profiler: {profile_id}"
            
        except Exception as e:
            response.success = False
            response.message = str(e)
        
        return response
    
    def _handle_get_recommendations(self, request: GetOptimizationRecommendations.Request,
                                   response: GetOptimizationRecommendations.Response) -> GetOptimizationRecommendations.Response:
        """Handle get optimization recommendations service request."""
        try:
            component_str = request.component
            severity_filter = request.severity_filter
            
            recommendations = []
            
            # Generate recommendations based on current state
            if not component_str or component_str == "system":
                # System-wide recommendations
                cpu_percent = psutil.cpu_percent(interval=0.1)
                if cpu_percent > 80:
                    recommendations.append({
                        'component': 'system',
                        'severity': AlertSeverity.WARNING.value,
                        'recommendation': f"CPU usage is high ({cpu_percent:.1f}%). Consider optimizing CPU-intensive tasks.",
                        'priority': 2
                    })
                
                memory = psutil.virtual_memory()
                if memory.percent > 85:
                    recommendations.append({
                        'component': 'system',
                        'severity': AlertSeverity.WARNING.value,
                        'recommendation': f"Memory usage is high ({memory.percent:.1f}%). Consider optimizing memory usage.",
                        'priority': 2
                    })
            
            # Component-specific recommendations
            if component_str:
                component_type = ComponentType(component_str)
                if component_type in self.components:
                    component = self.components[component_type]
                    
                    # Check for performance issues
                    if component.state == PerformanceState.DEGRADED:
                        recommendations.append({
                            'component': component_str,
                            'severity': AlertSeverity.WARNING.value,
                            'recommendation': f"Component performance is degraded. Investigate potential issues.",
                            'priority': 1
                        })
            
            # Apply severity filter
            if severity_filter >= 0:
                recommendations = [r for r in recommendations if r['severity'] >= severity_filter]
            
            # Sort by priority (descending)
            recommendations.sort(key=lambda x: x['priority'], reverse=True)
            
            # Create response
            response.success = True
            response.recommendations = [json.dumps(r) for r in recommendations]
            response.count = len(recommendations)
            
        except Exception as e:
            response.success = False
            response.error_message = str(e)
        
        return response
    
    def _handle_set_thresholds(self, request: SetPerformanceThresholds.Request,
                              response: SetPerformanceThresholds.Response) -> SetPerformanceThresholds.Response:
        """Handle set performance thresholds service request."""
        try:
            component_str = request.component
            metric_type_str = request.metric_type
            warning = request.warning_threshold
            error = request.error_threshold
            critical = request.critical_threshold
            
            # Validate thresholds
            if not (warning < error < critical):
                response.success = False
                response.message = "Thresholds must be in order: warning < error < critical"
                return response
            
            # Find component
            component_type = ComponentType(component_str)
            if component_type not in self.components:
                response.success = False
                response.message = f"Unknown component: {component_str}"
                return response
            
            # Set thresholds
            self.components[component_type].thresholds[metric_type_str] = (warning, error, critical)
            
            response.success = True
            response.message = f"Thresholds set for {component_str}.{metric_type_str}: warning={warning}, error={error}, critical={critical}"
            
        except Exception as e:
            response.success = False
            response.message = str(e)
        
        return response
    
    def _handle_reset_metrics(self, request: ResetPerformanceMetrics.Request,
                             response: ResetPerformanceMetrics.Response) -> ResetPerformanceMetrics.Response:
        """Handle reset performance metrics service request."""
        try:
            component_str = request.component
            
            if component_str:
                # Reset specific component
                component_type = ComponentType(component_str)
                if component_type in self.components:
                    self.components[component_type].metrics.clear()
                    response.success = True
                    response.message = f"Metrics reset for component: {component_str}"
                else:
                    response.success = False
                    response.message = f"Unknown component: {component_str}"
            else:
                # Reset all components
                for component in self.components.values():
                    component.metrics.clear()
                
                self.metrics_history.clear()
                response.success = True
                response.message = "All performance metrics reset"
            
        except Exception as e:
            response.success = False
            response.message = str(e)
        
        return response
    
    def _handle_export_data(self, request: ExportPerformanceData.Request,
                           response: ExportPerformanceData.Response) -> ExportPerformanceData.Response:
        """Handle export performance data service request."""
        try:
            export_format = request.format
            start_time = datetime.fromisoformat(request.start_time) if request.start_time else None
            end_time = datetime.fromisoformat(request.end_time) if request.end_time else None
            
            # Create export directory
            export_dir = self.data_dir / "exports"
            export_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            export_file = export_dir / f"performance_export_{timestamp}.{export_format}"
            
            # Filter metrics by time range
            filtered_metrics = []
            for metric in self.metrics_history:
                if start_time and metric.timestamp < start_time:
                    continue
                if end_time and metric.timestamp > end_time:
                    continue
                filtered_metrics.append(metric)
            
            # Export based on format
            if export_format == "json":
                data = [m.to_dict() for m in filtered_metrics]
                with open(export_file, 'w') as f:
                    json.dump(data, f, indent=2)
            
            elif export_format == "csv":
                import csv
                with open(export_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    # Write header
                    writer.writerow(['timestamp', 'component', 'metric_type', 'value', 'unit', 'tags'])
                    # Write data
                    for metric in filtered_metrics:
                        writer.writerow([
                            metric.timestamp.isoformat(),
                            metric.component,
                            metric.metric_type.value,
                            metric.value,
                            metric.unit,
                            ','.join(metric.tags)
                        ])
            
            elif export_format == "pickle":
                with open(export_file, 'wb') as f:
                    pickle.dump(filtered_metrics, f)
            
            else:
                response.success = False
                response.message = f"Unsupported export format: {export_format}"
                return response
            
            response.success = True
            response.file_path = str(export_file)
            response.file_size = export_file.stat().st_size
            response.message = f"Exported {len(filtered_metrics)} metrics to {export_file}"
            
        except Exception as e:
            response.success = False
            response.message = str(e)
        
        return response
    
    def shutdown(self):
        """Graceful shutdown of performance monitor."""
        self.get_logger().info("Shutting down performance monitor...")
        self.monitoring_active = False
        
        # Stop thread pool
        self.thread_pool.shutdown(wait=True)
        
        # Wait for monitoring threads
        for thread in self.monitoring_threads:
            if thread.is_alive():
                thread.join(timeout=5.0)
        
        # Save final state
        self._save_final_state()
        
        self.get_logger().info("Performance monitor shutdown complete")
    
    def _save_final_state(self):
        """Save final performance state before shutdown."""
        try:
            state_file = self.data_dir / "final_state.json"
            
            final_state = {
                'shutdown_time': datetime.now().isoformat(),
                'overall_state': self.overall_state.name,
                'component_states': {c.value: comp.state.name for c, comp in self.components.items()},
                'alerts_last_24h': len([a for a in self.alerts_history 
                                      if (datetime.now() - a['timestamp']).total_seconds() < 24 * 3600]),
                'total_metrics_collected': len(self.metrics_history),
                'active_profilers': len(self.active_profilers)
            }
            
            with open(state_file, 'w') as f:
                json.dump(final_state, f, indent=2)
            
        except Exception as e:
            self.get_logger().error(f"Error saving final state: {str(e)}")


# Monitoring Modules
class SystemMonitor:
    """System-level monitoring module."""
    
    def __init__(self, parent: PerformanceMonitor):
        self.parent = parent
        self.logger = parent.get_logger()
    
    def update(self):
        """Update system monitoring."""
        try:
            # Additional system metrics
            self._monitor_processes()
            self._monitor_system_load()
            
        except Exception as e:
            self.logger.error(f"System monitor error: {str(e)}")
    
    def _monitor_processes(self):
        """Monitor system processes."""
        try:
            # Get top processes by CPU
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # Sort by CPU usage
            processes.sort(key=lambda x: x['cpu_percent'] or 0, reverse=True)
            
            # Log top processes
            for proc in processes[:5]:  # Top 5
                if proc['cpu_percent'] and proc['cpu_percent'] > 10:  # Only if significant
                    self.parent._add_metric(
                        component=ComponentType.SYSTEM,
                        metric_type=MetricType.CPU_USAGE,
                        value=proc['cpu_percent'],
                        unit="percent",
                        tags=["process", proc['name']],
                        metadata={'pid': proc['pid'], 'name': proc['name']}
                    )
        
        except Exception as e:
            self.logger.error(f"Process monitoring error: {str(e)}")
    
    def _monitor_system_load(self):
        """Monitor system load averages."""
        try:
            if hasattr(os, 'getloadavg'):
                load_avg = os.getloadavg()
                
                self.parent._add_metric(
                    component=ComponentType.SYSTEM,
                    metric_type=MetricType.CPU_USAGE,
                    value=load_avg[0],  # 1-minute load average
                    unit="load",
                    tags=["system", "load"],
                    metadata={'load_1min': load_avg[0], 'load_5min': load_avg[1], 'load_15min': load_avg[2]}
                )
        
        except Exception as e:
            self.logger.error(f"Load monitoring error: {str(e)}")


class GPUMonitor:
    """GPU monitoring module."""
    
    def __init__(self, parent: PerformanceMonitor):
        self.parent = parent
        self.logger = parent.get_logger()
        self.gpu_info = None
    
    def update(self):
        """Update GPU monitoring."""
        try:
            # Try to get GPU information
            import GPUtil
            
            gpus = GPUtil.getGPUs()
            if gpus:
                for i, gpu in enumerate(gpus):
                    self.parent._add_metric(
                        component=ComponentType.HARDWARE,
                        metric_type=MetricType.GPU_USAGE,
                        value=gpu.load * 100,
                        unit="percent",
                        tags=["gpu", f"gpu_{i}", gpu.name],
                        metadata={
                            'gpu_id': gpu.id,
                            'name': gpu.name,
                            'memory_used': gpu.memoryUsed,
                            'memory_total': gpu.memoryTotal,
                            'temperature': gpu.temperature
                        }
                    )
                    
                    # Also log GPU memory usage
                    self.parent._add_metric(
                        component=ComponentType.HARDWARE,
                        metric_type=MetricType.MEMORY_USAGE,
                        value=(gpu.memoryUsed / gpu.memoryTotal) * 100,
                        unit="percent",
                        tags=["gpu", "memory", f"gpu_{i}"],
                        metadata={
                            'memory_used_mb': gpu.memoryUsed,
                            'memory_total_mb': gpu.memoryTotal
                        }
                    )
        
        except ImportError:
            # GPU monitoring not available
            if self.gpu_info is None:
                self.logger.info("GPU monitoring not available (GPUtil not installed)")
                self.gpu_info = False
        
        except Exception as e:
            self.logger.error(f"GPU monitor error: {str(e)}")


class NetworkMonitor:
    """Network monitoring module."""
    
    def __init__(self, parent: PerformanceMonitor):
        self.parent = parent
        self.logger = parent.get_logger()
        self.last_io = psutil.net_io_counters()
        self.last_time = time.time()
    
    def update(self):
        """Update network monitoring."""
        try:
            current_io = psutil.net_io_counters()
            current_time = time.time()
            
            # Calculate rates
            time_diff = current_time - self.last_time
            if time_diff > 0:
                sent_rate = (current_io.bytes_sent - self.last_io.bytes_sent) / time_diff
                recv_rate = (current_io.bytes_recv - self.last_io.bytes_recv) / time_diff
                
                self.parent._add_metric(
                    component=ComponentType.NETWORK,
                    metric_type=MetricType.NETWORK_IO,
                    value=sent_rate + recv_rate,
                    unit="bytes/sec",
                    tags=["network", "throughput"],
                    metadata={'sent_bps': sent_rate * 8, 'recv_bps': recv_rate * 8}
                )
            
            # Update last values
            self.last_io = current_io
            self.last_time = current_time
            
            # Monitor network connections
            connections = psutil.net_connections()
            tcp_connections = len([c for c in connections if c.type == socket.SOCK_STREAM])
            udp_connections = len([c for c in connections if c.type == socket.SOCK_DGRAM])
            
            self.parent._add_metric(
                component=ComponentType.NETWORK,
                metric_type=MetricType.THROUGHPUT,
                value=tcp_connections + udp_connections,
                unit="connections",
                tags=["network", "connections"],
                metadata={'tcp': tcp_connections, 'udp': udp_connections}
            )
        
        except Exception as e:
            self.logger.error(f"Network monitor error: {str(e)}")


class StorageMonitor:
    """Storage monitoring module."""
    
    def __init__(self, parent: PerformanceMonitor):
        self.parent = parent
        self.logger = parent.get_logger()
        self.last_io = psutil.disk_io_counters()
        self.last_time = time.time()
    
    def update(self):
        """Update storage monitoring."""
        try:
            current_io = psutil.disk_io_counters()
            current_time = time.time()
            
            # Calculate rates
            time_diff = current_time - self.last_time
            if time_diff > 0:
                read_rate = (current_io.read_bytes - self.last_io.read_bytes) / time_diff
                write_rate = (current_io.write_bytes - self.last_io.write_bytes) / time_diff
                
                self.parent._add_metric(
                    component=ComponentType.STORAGE,
                    metric_type=MetricType.DISK_IO,
                    value=read_rate + write_rate,
                    unit="bytes/sec",
                    tags=["storage", "io"],
                    metadata={'read_bps': read_rate, 'write_bps': write_rate}
                )
            
            # Update last values
            self.last_io = current_io
            self.last_time = current_time
            
            # Monitor disk usage for all partitions
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    
                    self.parent._add_metric(
                        component=ComponentType.STORAGE,
                        metric_type=MetricType.THROUGHPUT,
                        value=usage.percent,
                        unit="percent",
                        tags=["storage", "usage", partition.device],
                        metadata={
                            'device': partition.device,
                            'mountpoint': partition.mountpoint,
                            'total_gb': usage.total / (1024**3),
                            'used_gb': usage.used / (1024**3),
                            'free_gb': usage.free / (1024**3)
                        }
                    )
                    
                except Exception:
                    continue
        
        except Exception as e:
            self.logger.error(f"Storage monitor error: {str(e)}")


class MLModelMonitor:
    """ML model monitoring module."""
    
    def __init__(self, parent: PerformanceMonitor):
        self.parent = parent
        self.logger = parent.get_logger()
    
    def update(self):
        """Update ML model monitoring."""
        try:
            # This would integrate with actual ML model inference
            # For now, simulate some metrics
            
            # Simulate inference time
            inference_time = np.random.normal(15, 3)
            
            self.parent._add_metric(
                component=ComponentType.ML_MODEL,
                metric_type=MetricType.INFERENCE_TIME,
                value=inference_time,
                unit="milliseconds",
                tags=["ml", "inference", "yolov5"]
            )
            
            # Simulate model accuracy (would come from actual model)
            accuracy = np.random.normal(0.95, 0.01)
            
            self.parent._add_metric(
                component=ComponentType.ML_MODEL,
                metric_type=MetricType.SUCCESS_RATE,
                value=accuracy * 100,
                unit="percent",
                tags=["ml", "accuracy", "yolov5"]
            )
            
            # Simulate model memory usage
            model_memory = 275  # MB for YOLOv5
            
            self.parent._add_metric(
                component=ComponentType.ML_MODEL,
                metric_type=MetricType.MEMORY_USAGE,
                value=model_memory,
                unit="megabytes",
                tags=["ml", "memory", "yolov5"]
            )
        
        except Exception as e:
            self.logger.error(f"ML model monitor error: {str(e)}")


class ROS2Monitor:
    """ROS2 monitoring module."""
    
    def __init__(self, parent: PerformanceMonitor):
        self.parent = parent
        self.logger = parent.get_logger()
    
    def update(self):
        """Update ROS2 monitoring."""
        try:
            # Get ROS2 node information
            import subprocess
            
            # Count nodes
            node_result = subprocess.run(
                ['ros2', 'node', 'list'],
                capture_output=True,
                text=True
            )
            
            if node_result.returncode == 0:
                nodes = node_result.stdout.strip().split('\n')
                node_count = len([n for n in nodes if n])  # Filter empty lines
                
                self.parent._add_metric(
                    component=ComponentType.COMMUNICATION,
                    metric_type=MetricType.THROUGHPUT,
                    value=node_count,
                    unit="nodes",
                    tags=["ros2", "nodes"]
                )
            
            # Count topics
            topic_result = subprocess.run(
                ['ros2', 'topic', 'list'],
                capture_output=True,
                text=True
            )
            
            if topic_result.returncode == 0:
                topics = topic_result.stdout.strip().split('\n')
                topic_count = len([t for t in topics if t])
                
                self.parent._add_metric(
                    component=ComponentType.COMMUNICATION,
                    metric_type=MetricType.THROUGHPUT,
                    value=topic_count,
                    unit="topics",
                    tags=["ros2", "topics"]
                )
            
        except Exception as e:
            self.logger.error(f"ROS2 monitor error: {str(e)}")


class EnergyMonitor:
    """Energy consumption monitoring module."""
    
    def __init__(self, parent: PerformanceMonitor):
        self.parent = parent
        self.logger = parent.get_logger()
    
    def update(self):
        """Update energy monitoring."""
        try:
            # Energy monitoring would require hardware sensors
            # For now, estimate based on CPU usage and temperature
            
            cpu_percent = psutil.cpu_percent(interval=0.1)
            temp = self.parent._get_system_temperature() or 50.0
            
            # Simple power estimation model
            power_watts = 5.0 + (cpu_percent / 100) * 15.0 + (temp / 100) * 5.0
            
            self.parent._add_metric(
                component=ComponentType.HARDWARE,
                metric_type=MetricType.ENERGY_CONSUMPTION,
                value=power_watts,
                unit="watts",
                tags=["energy", "power"],
                metadata={'cpu_percent': cpu_percent, 'temperature_c': temp}
            )
        
        except Exception as e:
            self.logger.error(f"Energy monitor error: {str(e)}")


def main(args=None):
    """Main entry point for performance monitor node."""
    rclpy.init(args=args)
    
    try:
        monitor = PerformanceMonitor()
        rclpy.spin(monitor)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Fatal error in performance monitor: {str(e)}")
    finally:
        if 'monitor' in locals():
            monitor.shutdown()
            monitor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
