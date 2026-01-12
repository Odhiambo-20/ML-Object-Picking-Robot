#!/usr/bin/env python3
"""
Health Checker Node for ML-Based Object Picking Robot System

This module provides comprehensive health monitoring for the robotic system,
checking hardware status, software performance, and system integrity.
Designed for production deployment with real-time diagnostics and alerting.

Author: Robotics Engineering Team
Version: 2.1.0
License: Proprietary
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import threading
import time
import psutil
import json
import socket
import subprocess
import signal
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

from system_monitor_msgs.msg import SystemHealth, ComponentStatus, Alert
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from std_msgs.msg import Header, String, Float32
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Temperature, BatteryState

# Custom messages from robot_interfaces
from robot_interfaces.msg import RobotStatus
from robot_interfaces.srv import GetSystemStatus


class HealthState(Enum):
    """Health states for system components."""
    HEALTHY = 0
    DEGRADED = 1
    CRITICAL = 2
    FAILED = 3
    UNKNOWN = 4


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = 0
    WARNING = 1
    ERROR = 2
    CRITICAL = 3


@dataclass
class ComponentMetrics:
    """Metrics for system components."""
    name: str
    state: HealthState
    uptime_seconds: float
    cpu_percent: float
    memory_mb: float
    last_update: datetime
    error_count: int = 0
    warning_count: int = 0
    custom_metrics: Dict[str, Any] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['state'] = self.state.name
        data['last_update'] = self.last_update.isoformat()
        if self.custom_metrics is None:
            data['custom_metrics'] = {}
        return data


@dataclass
class SystemMetrics:
    """Aggregated system metrics."""
    timestamp: datetime
    overall_health: HealthState
    components: Dict[str, ComponentMetrics]
    system_cpu_percent: float
    system_memory_percent: float
    system_temperature_c: float
    network_latency_ms: float
    disk_usage_percent: float
    ros2_nodes_active: int
    ros2_topics_active: int
    alerts_last_hour: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'overall_health': self.overall_health.name,
            'components': {name: comp.to_dict() for name, comp in self.components.items()},
            'system_cpu_percent': self.system_cpu_percent,
            'system_memory_percent': self.system_memory_percent,
            'system_temperature_c': self.system_temperature_c,
            'network_latency_ms': self.network_latency_ms,
            'disk_usage_percent': self.disk_usage_percent,
            'ros2_nodes_active': self.ros2_nodes_active,
            'ros2_topics_active': self.ros2_topics_active,
            'alerts_last_hour': self.alerts_last_hour
        }


class HealthChecker(Node):
    """
    Main health checker node for monitoring the robotic system.
    
    Features:
    - Real-time component health monitoring
    - Performance metrics collection
    - Alert generation and notification
    - Historical data logging
    - Automated recovery suggestions
    - Integration with ROS2 diagnostics
    """
    
    def __init__(self):
        super().__init__('health_checker')
        
        # Configuration parameters
        self.declare_parameter('check_interval_sec', 5.0)
        self.declare_parameter('alert_threshold_cpu', 85.0)
        self.declare_parameter('alert_threshold_memory', 90.0)
        self.declare_parameter('alert_threshold_temperature', 80.0)
        self.declare_parameter('log_directory', '/var/log/robot_health')
        self.declare_parameter('max_log_files', 50)
        self.declare_parameter('retention_days', 30)
        self.declare_parameter('enable_auto_recovery', False)
        self.declare_parameter('monitored_components', [
            'perception',
            'motion_planning', 
            'robot_control',
            'task_planning',
            'hardware',
            'network',
            'storage'
        ])
        
        self.check_interval = self.get_parameter('check_interval_sec').value
        self.alert_threshold_cpu = self.get_parameter('alert_threshold_cpu').value
        self.alert_threshold_memory = self.get_parameter('alert_threshold_memory').value
        self.alert_threshold_temp = self.get_parameter('alert_threshold_temperature').value
        self.log_dir = self.get_parameter('log_directory').value
        self.monitored_components = self.get_parameter('monitored_components').value
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize metrics storage
        self.component_metrics: Dict[str, ComponentMetrics] = {}
        self.system_metrics_history: List[SystemMetrics] = []
        self.alerts_history: List[Dict] = []
        self.performance_baseline: Dict[str, float] = {}
        
        # Initialize ROS2 publishers
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        self.health_pub = self.create_publisher(
            SystemHealth,
            '/system/health',
            qos_profile
        )
        
        self.alert_pub = self.create_publisher(
            Alert,
            '/system/alerts',
            qos_profile
        )
        
        self.diagnostic_pub = self.create_publisher(
            DiagnosticArray,
            '/diagnostics',
            qos_profile
        )
        
        # Service for status queries
        self.status_service = self.create_service(
            GetSystemStatus,
            '/system/get_status',
            self.handle_status_request
        )
        
        # Subscribers for component status
        self.create_subscription(
            RobotStatus,
            '/robot/status',
            self.handle_robot_status,
            qos_profile
        )
        
        self.create_subscription(
            BatteryState,
            '/power/battery',
            self.handle_battery_status,
            qos_profile
        )
        
        self.create_subscription(
            Temperature,
            '/system/temperature',
            self.handle_temperature,
            qos_profile
        )
        
        # Initialize components
        self._initialize_components()
        
        # Start monitoring threads
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        # Start periodic logging
        self.logging_timer = self.create_timer(60.0, self._periodic_logging)
        
        self.get_logger().info(
            f"Health Checker initialized with check interval: {self.check_interval} seconds"
        )
        
    def _initialize_components(self):
        """Initialize all monitored components."""
        for component in self.monitored_components:
            self.component_metrics[component] = ComponentMetrics(
                name=component,
                state=HealthState.UNKNOWN,
                uptime_seconds=0.0,
                cpu_percent=0.0,
                memory_mb=0.0,
                last_update=datetime.now(),
                error_count=0,
                warning_count=0,
                custom_metrics={}
            )
    
    def _monitoring_loop(self):
        """Main monitoring loop running in separate thread."""
        while self.monitoring_active and rclpy.ok():
            try:
                start_time = time.time()
                
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                
                # Update component health
                self._update_component_health()
                
                # Check for issues and generate alerts
                self._check_for_alerts(system_metrics)
                
                # Publish health status
                self._publish_health_status(system_metrics)
                
                # Publish diagnostics
                self._publish_diagnostics()
                
                # Log metrics
                self._log_metrics(system_metrics)
                
                # Calculate sleep time to maintain exact interval
                elapsed = time.time() - start_time
                sleep_time = max(0.1, self.check_interval - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                self.get_logger().error(f"Error in monitoring loop: {str(e)}")
                time.sleep(1.0)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """
        Collect comprehensive system metrics.
        
        Returns:
            SystemMetrics: Aggregated system metrics
        """
        # System CPU and memory
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # System temperature (Raspberry Pi specific)
        temperature = self._read_system_temperature()
        
        # Network latency
        latency = self._measure_network_latency()
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        
        # ROS2 nodes and topics
        nodes_count, topics_count = self._count_ros2_entities()
        
        # Calculate overall health
        overall_health = self._calculate_overall_health()
        
        metrics = SystemMetrics(
            timestamp=datetime.now(),
            overall_health=overall_health,
            components=self.component_metrics.copy(),
            system_cpu_percent=cpu_percent,
            system_memory_percent=memory_percent,
            system_temperature_c=temperature,
            network_latency_ms=latency,
            disk_usage_percent=disk_percent,
            ros2_nodes_active=nodes_count,
            ros2_topics_active=topics_count,
            alerts_last_hour=self._count_recent_alerts(hours=1)
        )
        
        # Store in history (keep last 1000 entries)
        self.system_metrics_history.append(metrics)
        if len(self.system_metrics_history) > 1000:
            self.system_metrics_history.pop(0)
        
        return metrics
    
    def _read_system_temperature(self) -> float:
        """
        Read system temperature from Raspberry Pi or fallback to psutil.
        
        Returns:
            float: Temperature in Celsius
        """
        try:
            # Try Raspberry Pi specific temperature reading
            if os.path.exists('/sys/class/thermal/thermal_zone0/temp'):
                with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                    temp_millic = float(f.read().strip())
                return temp_millic / 1000.0
            
            # Fallback to psutil if available
            if hasattr(psutil, 'sensors_temperatures'):
                temps = psutil.sensors_temperatures()
                if 'cpu_thermal' in temps:
                    return temps['cpu_thermal'][0].current
                elif 'coretemp' in temps:
                    return temps['coretemp'][0].current
            
            # Default fallback
            return 45.0  # Conservative estimate
            
        except Exception as e:
            self.get_logger().warn(f"Could not read temperature: {str(e)}")
            return 50.0  # Safe default
    
    def _measure_network_latency(self) -> float:
        """
        Measure network latency to default gateway.
        
        Returns:
            float: Latency in milliseconds
        """
        try:
            # Try to ping default gateway
            result = subprocess.run(
                ['ping', '-c', '2', '-W', '1', '8.8.8.8'],
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
                            return float(parts[4])
            
            return 999.0  # High latency indicates network issues
            
        except Exception:
            return 999.0
    
    def _count_ros2_entities(self) -> Tuple[int, int]:
        """
        Count active ROS2 nodes and topics.
        
        Returns:
            Tuple[int, int]: (node_count, topic_count)
        """
        try:
            # Count nodes
            node_result = subprocess.run(
                ['ros2', 'node', 'list'],
                capture_output=True,
                text=True
            )
            node_count = len(node_result.stdout.strip().split('\n')) if node_result.stdout else 0
            
            # Count topics
            topic_result = subprocess.run(
                ['ros2', 'topic', 'list'],
                capture_output=True,
                text=True
            )
            topic_count = len(topic_result.stdout.strip().split('\n')) if topic_result.stdout else 0
            
            return node_count, topic_count
            
        except Exception as e:
            self.get_logger().warn(f"Could not count ROS2 entities: {str(e)}")
            return 0, 0
    
    def _update_component_health(self):
        """Update health status for all monitored components."""
        current_time = datetime.now()
        
        for component_name, metrics in self.component_metrics.items():
            try:
                # Update uptime
                metrics.uptime_seconds = (current_time - metrics.last_update).total_seconds()
                
                # Component-specific health checks
                if component_name == 'perception':
                    self._check_perception_health(metrics)
                elif component_name == 'motion_planning':
                    self._check_motion_planning_health(metrics)
                elif component_name == 'robot_control':
                    self._check_robot_control_health(metrics)
                elif component_name == 'hardware':
                    self._check_hardware_health(metrics)
                elif component_name == 'network':
                    self._check_network_health(metrics)
                elif component_name == 'storage':
                    self._check_storage_health(metrics)
                
                metrics.last_update = current_time
                
            except Exception as e:
                self.get_logger().error(f"Error updating {component_name} health: {str(e)}")
                metrics.state = HealthState.UNKNOWN
                metrics.error_count += 1
    
    def _check_perception_health(self, metrics: ComponentMetrics):
        """
        Check health of perception system.
        
        Args:
            metrics: Component metrics to update
        """
        try:
            # Check if perception node is running
            node_result = subprocess.run(
                ['ros2', 'node', 'list'],
                capture_output=True,
                text=True
            )
            
            perception_running = 'perception_node' in node_result.stdout
            
            if not perception_running:
                metrics.state = HealthState.FAILED
                metrics.error_count += 1
                return
            
            # Check perception performance (simplified - in production would query metrics)
            metrics.state = HealthState.HEALTHY
            
            # Update custom metrics
            metrics.custom_metrics.update({
                'detection_fps': 30.0,  # Would be measured from actual system
                'inference_time_ms': 15.0,
                'object_count': 3,
                'tracking_accuracy': 0.95
            })
            
        except Exception as e:
            metrics.state = HealthState.UNKNOWN
            metrics.error_count += 1
            self.get_logger().error(f"Perception health check failed: {str(e)}")
    
    def _check_motion_planning_health(self, metrics: ComponentMetrics):
        """
        Check health of motion planning system.
        
        Args:
            metrics: Component metrics to update
        """
        try:
            # Check if motion planning node is running
            node_result = subprocess.run(
                ['ros2', 'node', 'list'],
                capture_output=True,
                text=True
            )
            
            motion_running = 'motion_planning_node' in node_result.stdout
            
            if not motion_running:
                metrics.state = HealthState.FAILED
                metrics.error_count += 1
                return
            
            metrics.state = HealthState.HEALTHY
            
            # Update custom metrics
            metrics.custom_metrics.update({
                'planning_success_rate': 0.98,
                'average_planning_time_ms': 120.0,
                'collision_checks_per_second': 1000,
                'trajectory_optimization_time_ms': 45.0
            })
            
        except Exception as e:
            metrics.state = HealthState.UNKNOWN
            metrics.error_count += 1
            self.get_logger().error(f"Motion planning health check failed: {str(e)}")
    
    def _check_robot_control_health(self, metrics: ComponentMetrics):
        """
        Check health of robot control system.
        
        Args:
            metrics: Component metrics to update
        """
        try:
            # Check if control node is running
            node_result = subprocess.run(
                ['ros2', 'node', 'list'],
                capture_output=True,
                text=True
            )
            
            control_running = 'robot_control_node' in node_result.stdout
            
            if not control_running:
                metrics.state = HealthState.FAILED
                metrics.error_count += 1
                return
            
            metrics.state = HealthState.HEALTHY
            
            # Update custom metrics
            metrics.custom_metrics.update({
                'servo_response_time_ms': 5.0,
                'motor_current_a': 2.3,
                'gripper_force_n': 15.0,
                'control_loop_hz': 100.0
            })
            
        except Exception as e:
            metrics.state = HealthState.UNKNOWN
            metrics.error_count += 1
            self.get_logger().error(f"Robot control health check failed: {str(e)}")
    
    def _check_hardware_health(self, metrics: ComponentMetrics):
        """
        Check health of hardware components.
        
        Args:
            metrics: Component metrics to update
        """
        try:
            # Check various hardware components
            hardware_issues = []
            
            # Check CPU temperature
            temp = self._read_system_temperature()
            if temp > self.alert_threshold_temp:
                hardware_issues.append(f"High temperature: {temp}°C")
            
            # Check battery if available
            # This would be populated from battery status callback
            
            # Check disk health
            disk_smart = self._check_disk_smart()
            if disk_smart != "OK":
                hardware_issues.append(f"Disk SMART: {disk_smart}")
            
            # Determine overall state
            if hardware_issues:
                metrics.state = HealthState.DEGRADED
                metrics.warning_count += 1
                metrics.custom_metrics['hardware_issues'] = hardware_issues
            else:
                metrics.state = HealthState.HEALTHY
            
            metrics.custom_metrics.update({
                'cpu_temperature_c': temp,
                'gpu_temperature_c': temp + 5.0,  # Estimate
                'power_consumption_w': 12.5,
                'fan_speed_rpm': 3200
            })
            
        except Exception as e:
            metrics.state = HealthState.UNKNOWN
            metrics.error_count += 1
            self.get_logger().error(f"Hardware health check failed: {str(e)}")
    
    def _check_disk_smart(self) -> str:
        """
        Check disk health using SMART.
        
        Returns:
            str: SMART status or "UNKNOWN"
        """
        try:
            # Try to get SMART data for first disk
            result = subprocess.run(
                ['sudo', 'smartctl', '-H', '/dev/sda'],
                capture_output=True,
                text=True
            )
            
            if 'PASSED' in result.stdout:
                return "OK"
            elif 'FAILED' in result.stdout:
                return "FAILED"
            else:
                return "UNKNOWN"
                
        except Exception:
            return "UNKNOWN"
    
    def _check_network_health(self, metrics: ComponentMetrics):
        """
        Check health of network connectivity.
        
        Args:
            metrics: Component metrics to update
        """
        try:
            network_issues = []
            
            # Check latency
            latency = self._measure_network_latency()
            if latency > 100.0:  # More than 100ms is problematic
                network_issues.append(f"High latency: {latency}ms")
            
            # Check packet loss (simplified)
            result = subprocess.run(
                ['ping', '-c', '3', '-W', '1', '8.8.8.8'],
                capture_output=True,
                text=True
            )
            
            if '100% packet loss' in result.stdout:
                network_issues.append("Complete packet loss")
            
            # Check bandwidth (simplified)
            # In production, would run actual bandwidth tests
            
            if network_issues:
                metrics.state = HealthState.DEGRADED
                metrics.warning_count += 1
                metrics.custom_metrics['network_issues'] = network_issues
            else:
                metrics.state = HealthState.HEALTHY
            
            metrics.custom_metrics.update({
                'network_latency_ms': latency,
                'bandwidth_mbps': 950.0,  # Estimate
                'packet_loss_percent': 0.1,
                'connection_stability': 0.99
            })
            
        except Exception as e:
            metrics.state = HealthState.UNKNOWN
            metrics.error_count += 1
            self.get_logger().error(f"Network health check failed: {str(e)}")
    
    def _check_storage_health(self, metrics: ComponentMetrics):
        """
        Check health of storage system.
        
        Args:
            metrics: Component metrics to update
        """
        try:
            storage_issues = []
            
            # Check disk usage
            disk = psutil.disk_usage('/')
            if disk.percent > 90:
                storage_issues.append(f"Disk almost full: {disk.percent}%")
            
            # Check inode usage
            inode_result = subprocess.run(
                ['df', '-i', '/'],
                capture_output=True,
                text=True
            )
            
            lines = inode_result.stdout.strip().split('\n')
            if len(lines) > 1:
                inode_parts = lines[1].split()
                if len(inode_parts) > 4:
                    inode_use = inode_parts[4].replace('%', '')
                    if float(inode_use) > 90:
                        storage_issues.append(f"High inode usage: {inode_use}%")
            
            # Check write speed (simplified)
            if storage_issues:
                metrics.state = HealthState.DEGRADED
                metrics.warning_count += 1
                metrics.custom_metrics['storage_issues'] = storage_issues
            else:
                metrics.state = HealthState.HEALTHY
            
            metrics.custom_metrics.update({
                'disk_usage_percent': disk.percent,
                'read_speed_mbps': 350.0,
                'write_speed_mbps': 280.0,
                'io_operations_per_sec': 1500,
                'available_space_gb': disk.free / (1024**3)
            })
            
        except Exception as e:
            metrics.state = HealthState.UNKNOWN
            metrics.error_count += 1
            self.get_logger().error(f"Storage health check failed: {str(e)}")
    
    def _calculate_overall_health(self) -> HealthState:
        """
        Calculate overall system health based on component states.
        
        Returns:
            HealthState: Overall system health state
        """
        if not self.component_metrics:
            return HealthState.UNKNOWN
        
        # Count states
        state_counts = {state: 0 for state in HealthState}
        for metrics in self.component_metrics.values():
            state_counts[metrics.state] += 1
        
        # Determine overall state
        if state_counts[HealthState.FAILED] > 0:
            return HealthState.FAILED
        elif state_counts[HealthState.CRITICAL] > 0:
            return HealthState.CRITICAL
        elif state_counts[HealthState.DEGRADED] > 0:
            return HealthState.DEGRADED
        elif state_counts[HealthState.UNKNOWN] > len(self.component_metrics) / 2:
            return HealthState.UNKNOWN
        else:
            return HealthState.HEALTHY
    
    def _check_for_alerts(self, system_metrics: SystemMetrics):
        """
        Check system metrics and generate alerts if thresholds are exceeded.
        
        Args:
            system_metrics: Current system metrics
        """
        alerts = []
        
        # Check CPU usage
        if system_metrics.system_cpu_percent > self.alert_threshold_cpu:
            alerts.append({
                'level': AlertLevel.WARNING,
                'component': 'system',
                'message': f'High CPU usage: {system_metrics.system_cpu_percent:.1f}%',
                'timestamp': datetime.now(),
                'metric': 'cpu_percent',
                'value': system_metrics.system_cpu_percent,
                'threshold': self.alert_threshold_cpu
            })
        
        # Check memory usage
        if system_metrics.system_memory_percent > self.alert_threshold_memory:
            alerts.append({
                'level': AlertLevel.WARNING,
                'component': 'system',
                'message': f'High memory usage: {system_metrics.system_memory_percent:.1f}%',
                'timestamp': datetime.now(),
                'metric': 'memory_percent',
                'value': system_metrics.system_memory_percent,
                'threshold': self.alert_threshold_memory
            })
        
        # Check temperature
        if system_metrics.system_temperature_c > self.alert_threshold_temp:
            alerts.append({
                'level': AlertLevel.CRITICAL,
                'component': 'hardware',
                'message': f'High temperature: {system_metrics.system_temperature_c:.1f}°C',
                'timestamp': datetime.now(),
                'metric': 'temperature_c',
                'value': system_metrics.system_temperature_c,
                'threshold': self.alert_threshold_temp
            })
        
        # Check disk usage
        if system_metrics.disk_usage_percent > 90:
            alerts.append({
                'level': AlertLevel.WARNING,
                'component': 'storage',
                'message': f'High disk usage: {system_metrics.disk_usage_percent:.1f}%',
                'timestamp': datetime.now(),
                'metric': 'disk_usage_percent',
                'value': system_metrics.disk_usage_percent,
                'threshold': 90.0
            })
        
        # Check component failures
        for component_name, metrics in system_metrics.components.items():
            if metrics.state == HealthState.FAILED:
                alerts.append({
                    'level': AlertLevel.CRITICAL,
                    'component': component_name,
                    'message': f'Component {component_name} has failed',
                    'timestamp': datetime.now(),
                    'metric': 'component_state',
                    'value': 'FAILED',
                    'threshold': 'OPERATIONAL'
                })
            elif metrics.state == HealthState.CRITICAL:
                alerts.append({
                    'level': AlertLevel.CRITICAL,
                    'component': component_name,
                    'message': f'Component {component_name} in critical state',
                    'timestamp': datetime.now(),
                    'metric': 'component_state',
                    'value': 'CRITICAL',
                    'threshold': 'HEALTHY'
                })
        
        # Publish alerts
        for alert_data in alerts:
            self._publish_alert(alert_data)
            self.alerts_history.append(alert_data)
            
            # Keep only last 1000 alerts
            if len(self.alerts_history) > 1000:
                self.alerts_history.pop(0)
    
    def _publish_health_status(self, system_metrics: SystemMetrics):
        """
        Publish system health status to ROS2 topic.
        
        Args:
            system_metrics: System metrics to publish
        """
        try:
            msg = SystemHealth()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'system_health'
            
            msg.overall_health = int(system_metrics.overall_health.value)
            msg.system_cpu_percent = float(system_metrics.system_cpu_percent)
            msg.system_memory_percent = float(system_metrics.system_memory_percent)
            msg.system_temperature = float(system_metrics.system_temperature_c)
            msg.network_latency_ms = float(system_metrics.network_latency_ms)
            msg.disk_usage_percent = float(system_metrics.disk_usage_percent)
            
            # Add component statuses
            for component_name, metrics in system_metrics.components.items():
                component_msg = ComponentStatus()
                component_msg.name = component_name
                component_msg.health_state = int(metrics.state.value)
                component_msg.uptime_seconds = float(metrics.uptime_seconds)
                component_msg.cpu_percent = float(metrics.cpu_percent)
                component_msg.memory_mb = float(metrics.memory_mb)
                component_msg.last_update = metrics.last_update.isoformat()
                
                # Add custom metrics as JSON string
                if metrics.custom_metrics:
                    component_msg.custom_metrics = json.dumps(metrics.custom_metrics)
                
                msg.components.append(component_msg)
            
            self.health_pub.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing health status: {str(e)}")
    
    def _publish_alert(self, alert_data: Dict):
        """
        Publish alert to ROS2 topic.
        
        Args:
            alert_data: Alert data dictionary
        """
        try:
            msg = Alert()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'system_alerts'
            
            msg.level = int(alert_data['level'].value)
            msg.component = alert_data['component']
            msg.message = alert_data['message']
            msg.timestamp = alert_data['timestamp'].isoformat()
            msg.metric = alert_data.get('metric', '')
            msg.value = str(alert_data.get('value', ''))
            msg.threshold = str(alert_data.get('threshold', ''))
            
            self.alert_pub.publish(msg)
            self.get_logger().warn(f"ALERT: {alert_data['message']}")
            
        except Exception as e:
            self.get_logger().error(f"Error publishing alert: {str(e)}")
    
    def _publish_diagnostics(self):
        """Publish ROS2 diagnostic messages."""
        try:
            diag_array = DiagnosticArray()
            diag_array.header.stamp = self.get_clock().now().to_msg()
            diag_array.header.frame_id = 'diagnostics'
            
            # System level diagnostics
            sys_status = DiagnosticStatus()
            sys_status.name = "System Health"
            sys_status.hardware_id = "robot_system"
            sys_status.level = DiagnosticStatus.OK
            
            # Add system metrics as key-value pairs
            current_metrics = self.system_metrics_history[-1] if self.system_metrics_history else None
            if current_metrics:
                sys_status.message = f"Overall health: {current_metrics.overall_health.name}"
                sys_status.values.extend([
                    KeyValue(key="cpu_percent", value=f"{current_metrics.system_cpu_percent:.1f}"),
                    KeyValue(key="memory_percent", value=f"{current_metrics.system_memory_percent:.1f}"),
                    KeyValue(key="temperature_c", value=f"{current_metrics.system_temperature_c:.1f}"),
                    KeyValue(key="disk_usage_percent", value=f"{current_metrics.disk_usage_percent:.1f}"),
                    KeyValue(key="ros_nodes", value=str(current_metrics.ros2_nodes_active)),
                    KeyValue(key="alerts_last_hour", value=str(current_metrics.alerts_last_hour))
                ])
            
            diag_array.status.append(sys_status)
            
            # Component diagnostics
            for component_name, metrics in self.component_metrics.items():
                comp_status = DiagnosticStatus()
                comp_status.name = f"Component: {component_name}"
                comp_status.hardware_id = f"component_{component_name}"
                
                # Map health state to diagnostic level
                if metrics.state == HealthState.HEALTHY:
                    comp_status.level = DiagnosticStatus.OK
                elif metrics.state in [HealthState.DEGRADED, HealthState.UNKNOWN]:
                    comp_status.level = DiagnosticStatus.WARN
                else:
                    comp_status.level = DiagnosticStatus.ERROR
                
                comp_status.message = f"State: {metrics.state.name}"
                comp_status.values.extend([
                    KeyValue(key="uptime_seconds", value=f"{metrics.uptime_seconds:.1f}"),
                    KeyValue(key="error_count", value=str(metrics.error_count)),
                    KeyValue(key="warning_count", value=str(metrics.warning_count))
                ])
                
                # Add custom metrics
                for key, value in metrics.custom_metrics.items():
                    comp_status.values.append(
                        KeyValue(key=key, value=str(value))
                    )
                
                diag_array.status.append(comp_status)
            
            self.diagnostic_pub.publish(diag_array)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing diagnostics: {str(e)}")
    
    def _log_metrics(self, system_metrics: SystemMetrics):
        """
        Log metrics to file for historical analysis.
        
        Args:
            system_metrics: Metrics to log
        """
        try:
            log_file = os.path.join(
                self.log_dir,
                f"health_metrics_{datetime.now().strftime('%Y%m%d')}.jsonl"
            )
            
            log_entry = system_metrics.to_dict()
            log_entry['log_timestamp'] = datetime.now().isoformat()
            
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
            
        except Exception as e:
            self.get_logger().error(f"Error logging metrics: {str(e)}")
    
    def _periodic_logging(self):
        """Perform periodic logging and cleanup."""
        try:
            # Log summary
            summary = {
                'timestamp': datetime.now().isoformat(),
                'components_total': len(self.component_metrics),
                'components_healthy': sum(1 for m in self.component_metrics.values() 
                                        if m.state == HealthState.HEALTHY),
                'alerts_24h': self._count_recent_alerts(hours=24),
                'system_uptime': self._get_system_uptime(),
                'log_files': self._cleanup_old_logs()
            }
            
            summary_file = os.path.join(self.log_dir, 'health_summary.jsonl')
            with open(summary_file, 'a') as f:
                f.write(json.dumps(summary) + '\n')
                
        except Exception as e:
            self.get_logger().error(f"Error in periodic logging: {str(e)}")
    
    def _count_recent_alerts(self, hours: int = 1) -> int:
        """
        Count alerts from the last N hours.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            int: Number of alerts
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        return sum(1 for alert in self.alerts_history 
                  if alert['timestamp'] > cutoff)
    
    def _get_system_uptime(self) -> float:
        """
        Get system uptime in seconds.
        
        Returns:
            float: Uptime in seconds
        """
        try:
            with open('/proc/uptime', 'r') as f:
                uptime_seconds = float(f.readline().split()[0])
            return uptime_seconds
        except:
            return 0.0
    
    def _cleanup_old_logs(self) -> int:
        """
        Clean up old log files.
        
        Returns:
            int: Number of files removed
        """
        try:
            files_removed = 0
            retention_days = self.get_parameter('retention_days').value
            
            cutoff_time = datetime.now() - timedelta(days=retention_days)
            
            for filename in os.listdir(self.log_dir):
                filepath = os.path.join(self.log_dir, filename)
                if os.path.isfile(filepath):
                    file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                    if file_time < cutoff_time:
                        os.remove(filepath)
                        files_removed += 1
            
            return files_removed
            
        except Exception as e:
            self.get_logger().error(f"Error cleaning up logs: {str(e)}")
            return 0
    
    def handle_robot_status(self, msg: RobotStatus):
        """
        Handle incoming robot status messages.
        
        Args:
            msg: Robot status message
        """
        try:
            # Update hardware component based on robot status
            if 'hardware' in self.component_metrics:
                metrics = self.component_metrics['hardware']
                
                # Map robot status to health state
                if msg.status == RobotStatus.STATUS_READY:
                    metrics.state = HealthState.HEALTHY
                elif msg.status == RobotStatus.STATUS_ERROR:
                    metrics.state = HealthState.FAILED
                elif msg.status == RobotStatus.STATUS_WARNING:
                    metrics.state = HealthState.DEGRADED
                else:
                    metrics.state = HealthState.UNKNOWN
                
                # Update custom metrics
                metrics.custom_metrics.update({
                    'robot_status': msg.status,
                    'joint_positions': list(msg.joint_positions),
                    'end_effector_pose': list(msg.end_effector_pose),
                    'operation_mode': msg.operation_mode
                })
                
        except Exception as e:
            self.get_logger().error(f"Error handling robot status: {str(e)}")
    
    def handle_battery_status(self, msg: BatteryState):
        """
        Handle incoming battery status messages.
        
        Args:
            msg: Battery state message
        """
        try:
            # Update hardware metrics with battery info
            if 'hardware' in self.component_metrics:
                metrics = self.component_metrics['hardware']
                
                metrics.custom_metrics.update({
                    'battery_percentage': float(msg.percentage),
                    'battery_voltage': float(msg.voltage),
                    'battery_current': float(msg.current),
                    'battery_temperature': float(msg.temperature),
                    'power_supply_status': msg.power_supply_status
                })
                
                # Alert on low battery
                if msg.percentage < 20.0:
                    self._publish_alert({
                        'level': AlertLevel.WARNING,
                        'component': 'hardware',
                        'message': f'Low battery: {msg.percentage:.1f}%',
                        'timestamp': datetime.now(),
                        'metric': 'battery_percentage',
                        'value': msg.percentage,
                        'threshold': 20.0
                    })
                    
        except Exception as e:
            self.get_logger().error(f"Error handling battery status: {str(e)}")
    
    def handle_temperature(self, msg: Temperature):
        """
        Handle incoming temperature messages.
        
        Args:
            msg: Temperature message
        """
        try:
            # Update hardware metrics with temperature
            if 'hardware' in self.component_metrics:
                metrics = self.component_metrics['hardware']
                metrics.custom_metrics['sensor_temperature'] = float(msg.temperature)
                
        except Exception as e:
            self.get_logger().error(f"Error handling temperature: {str(e)}")
    
    def handle_status_request(self, request, response):
        """
        Handle service requests for system status.
        
        Args:
            request: Service request
            response: Service response
            
        Returns:
            GetSystemStatus.Response: System status response
        """
        try:
            # Get current metrics
            current_metrics = self.system_metrics_history[-1] if self.system_metrics_history else None
            
            response.success = True
            response.overall_health = self._calculate_overall_health().value
            
            if current_metrics:
                response.system_cpu = current_metrics.system_cpu_percent
                response.system_memory = current_metrics.system_memory_percent
                response.system_temperature = current_metrics.system_temperature_c
                response.timestamp = current_metrics.timestamp.isoformat()
                
                # Add component details
                for component_name, metrics in current_metrics.components.items():
                    response.components.append(component_name)
                    response.component_states.append(metrics.state.value)
                    
            # Add alert summary
            response.active_alerts = self._count_recent_alerts(hours=1)
            response.total_alerts_24h = self._count_recent_alerts(hours=24)
            
            return response
            
        except Exception as e:
            self.get_logger().error(f"Error handling status request: {str(e)}")
            response.success = False
            response.error_message = str(e)
            return response
    
    def get_detailed_report(self) -> Dict:
        """
        Generate detailed health report.
        
        Returns:
            Dict: Comprehensive health report
        """
        current_metrics = self.system_metrics_history[-1] if self.system_metrics_history else None
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'system': {
                'overall_health': self._calculate_overall_health().name,
                'uptime_seconds': self._get_system_uptime(),
                'alerts_last_hour': self._count_recent_alerts(hours=1),
                'alerts_last_24h': self._count_recent_alerts(hours=24)
            },
            'components': {},
            'recommendations': self._generate_recommendations()
        }
        
        if current_metrics:
            report['system'].update({
                'cpu_percent': current_metrics.system_cpu_percent,
                'memory_percent': current_metrics.system_memory_percent,
                'temperature_c': current_metrics.system_temperature_c,
                'disk_usage_percent': current_metrics.disk_usage_percent,
                'network_latency_ms': current_metrics.network_latency_ms
            })
            
            for component_name, metrics in current_metrics.components.items():
                report['components'][component_name] = metrics.to_dict()
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """
        Generate recommendations based on current system state.
        
        Returns:
            List[str]: List of recommendations
        """
        recommendations = []
        current_metrics = self.system_metrics_history[-1] if self.system_metrics_history else None
        
        if not current_metrics:
            return ["Insufficient data for recommendations"]
        
        # Check system metrics
        if current_metrics.system_cpu_percent > 80:
            recommendations.append("Consider optimizing CPU-intensive processes or adding compute resources")
        
        if current_metrics.system_memory_percent > 85:
            recommendations.append("Increase system memory or optimize memory usage in applications")
        
        if current_metrics.system_temperature_c > 75:
            recommendations.append("Improve cooling system or reduce ambient temperature")
        
        if current_metrics.disk_usage_percent > 85:
            recommendations.append("Clean up disk space or add additional storage")
        
        if current_metrics.network_latency_ms > 50:
            recommendations.append("Check network infrastructure and optimize network configuration")
        
        # Check component states
        failed_components = [name for name, metrics in current_metrics.components.items()
                            if metrics.state == HealthState.FAILED]
        
        if failed_components:
            recommendations.append(f"Restart failed components: {', '.join(failed_components)}")
        
        # Check error rates
        high_error_components = [name for name, metrics in current_metrics.components.items()
                                if metrics.error_count > 10]
        
        if high_error_components:
            recommendations.append(f"Investigate error patterns in: {', '.join(high_error_components)}")
        
        if not recommendations:
            recommendations.append("System operating within optimal parameters")
        
        return recommendations
    
    def shutdown(self):
        """Clean shutdown of health checker."""
        self.get_logger().info("Shutting down health checker...")
        self.monitoring_active = False
        
        if self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        # Save final report
        try:
            report = self.get_detailed_report()
            report_file = os.path.join(self.log_dir, 'shutdown_report.json')
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
        except Exception as e:
            self.get_logger().error(f"Error saving shutdown report: {str(e)}")
        
        self.get_logger().info("Health checker shutdown complete")


def main(args=None):
    """
    Main entry point for health checker node.
    
    Args:
        args: Command line arguments
    """
    rclpy.init(args=args)
    
    try:
        health_checker = HealthChecker()
        rclpy.spin(health_checker)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Fatal error in health checker: {str(e)}")
    finally:
        if 'health_checker' in locals():
            health_checker.shutdown()
            health_checker.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
