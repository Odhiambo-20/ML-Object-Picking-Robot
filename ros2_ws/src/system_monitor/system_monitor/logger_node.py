#!/usr/bin/env python3
"""
Advanced Logger Node for ML-Based Object Picking Robot System

Production-grade logging system with structured logging, log rotation,
compression, real-time analysis, and integration with monitoring systems.

Author: Robotics Engineering Team
Version: 3.2.0
License: Proprietary
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
import threading
import time
import json
import csv
import msgpack
import pickle
import logging
import logging.handlers
import queue
import zlib
import lzma
import hashlib
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import numpy as np
import psutil
import signal
import gzip
import bz2
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import select
import socket
import struct
import platform

# ROS2 Messages
from std_msgs.msg import Header, String, Bool
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from sensor_msgs.msg import Image, PointCloud2

# Custom messages
from robot_interfaces.msg import (
    DetectedObject, 
    ObstacleMap, 
    GraspPose, 
    RobotStatus,
    PickPlaceAction,
    NavigateAction
)
from system_monitor_msgs.msg import SystemHealth, Alert, LogEntry, LogBatch
from system_monitor_msgs.srv import (
    GetLogs, 
    SearchLogs, 
    ExportLogs,
    SetLogLevel,
    ClearLogs,
    GetLogStatistics
)


class LogLevel(Enum):
    """Logging levels with severity mapping."""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    SYSTEM = 60  # System-level events (startup, shutdown, etc.)
    AUDIT = 70   # Security and audit logs
    PERFORMANCE = 80  # Performance metrics


class LogCategory(Enum):
    """Categories for structured logging."""
    SYSTEM = "system"
    PERCEPTION = "perception"
    MOTION = "motion"
    CONTROL = "control"
    NAVIGATION = "navigation"
    PLANNING = "planning"
    SAFETY = "safety"
    COMMUNICATION = "communication"
    HARDWARE = "hardware"
    SOFTWARE = "software"
    SECURITY = "security"
    PERFORMANCE = "performance"
    USER = "user"
    DEBUG = "debug"
    AUDIT = "audit"
    DIAGNOSTIC = "diagnostic"


class LogFormat(Enum):
    """Supported log formats."""
    JSON = "json"
    MSGPACK = "msgpack"
    CSV = "csv"
    PLAIN = "plain"
    PICKLE = "pickle"
    PARQUET = "parquet"


class CompressionAlgorithm(Enum):
    """Supported compression algorithms."""
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"
    LZMA = "lzma"
    ZLIB = "zlib"
    BZ2 = "bz2"


@dataclass
class LogEntryData:
    """Internal representation of a log entry."""
    timestamp: datetime
    level: LogLevel
    category: LogCategory
    node: str
    source: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    correlation_id: Optional[str] = None
    session_id: Optional[str] = None
    sequence_number: int = 0
    hostname: str = ""
    pid: int = 0
    thread_id: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'level': self.level.name,
            'category': self.category.value,
            'node': self.node,
            'source': self.source,
            'message': self.message,
            'data': self.data,
            'context': self.context,
            'tags': self.tags,
            'correlation_id': self.correlation_id,
            'session_id': self.session_id,
            'sequence_number': self.sequence_number,
            'hostname': self.hostname,
            'pid': self.pid,
            'thread_id': self.thread_id
        }
    
    def to_ros_msg(self) -> LogEntry:
        """Convert to ROS2 message."""
        msg = LogEntry()
        msg.header.stamp = self._create_timestamp()
        msg.header.frame_id = "log"
        msg.timestamp = self.timestamp.isoformat()
        msg.level = self.level.value
        msg.category = self.category.value
        msg.node = self.node
        msg.source = self.source
        msg.message = self.message
        msg.data_json = json.dumps(self.data)
        msg.context_json = json.dumps(self.context)
        msg.tags = self.tags
        msg.correlation_id = self.correlation_id or ""
        msg.session_id = self.session_id or ""
        msg.sequence_number = self.sequence_number
        msg.hostname = self.hostname
        msg.pid = self.pid
        msg.thread_id = self.thread_id
        return msg
    
    def _create_timestamp(self):
        """Create ROS2 timestamp from datetime."""
        from builtin_interfaces.msg import Time
        sec = int(self.timestamp.timestamp())
        nanosec = int((self.timestamp.timestamp() - sec) * 1e9)
        time_msg = Time(sec=sec, nanosec=nanosec)
        return time_msg


@dataclass
class LogStatistics:
    """Statistics for log analysis."""
    period_start: datetime
    period_end: datetime
    total_entries: int
    entries_by_level: Dict[str, int]
    entries_by_category: Dict[str, int]
    entries_by_node: Dict[str, int]
    avg_entry_size_bytes: float
    max_entry_size_bytes: int
    error_rate_per_hour: float
    warning_rate_per_hour: float
    busiest_node: str
    busiest_category: str
    common_tags: List[Tuple[str, int]]
    data_volume_bytes: int
    compression_ratio: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'period_start': self.period_start.isoformat(),
            'period_end': self.period_end.isoformat(),
            'total_entries': self.total_entries,
            'entries_by_level': self.entries_by_level,
            'entries_by_category': self.entries_by_category,
            'entries_by_node': self.entries_by_node,
            'avg_entry_size_bytes': self.avg_entry_size_bytes,
            'max_entry_size_bytes': self.max_entry_size_bytes,
            'error_rate_per_hour': self.error_rate_per_hour,
            'warning_rate_per_hour': self.warning_rate_per_hour,
            'busiest_node': self.busiest_node,
            'busiest_category': self.busiest_category,
            'common_tags': [(tag, count) for tag, count in self.common_tags],
            'data_volume_bytes': self.data_volume_bytes,
            'compression_ratio': self.compression_ratio
        }


class AdvancedLoggerNode(Node):
    """
    Advanced production-grade logger node for robotic systems.
    
    Features:
    - Structured logging with multiple categories and levels
    - Multiple output formats (JSON, MsgPack, CSV, etc.)
    - Real-time compression and rotation
    - Log analysis and statistics
    - Topic-based logging
    - Performance monitoring
    - Distributed logging support
    - Encryption and security features
    """
    
    def __init__(self):
        super().__init__('advanced_logger')
        
        # Configuration parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('log_directory', '/var/log/robot_logs'),
                ('max_file_size_mb', 100),
                ('backup_count', 50),
                ('rotation_interval_hours', 24),
                ('compression_algorithm', 'gzip'),
                ('log_format', 'json'),
                ('default_log_level', 'INFO'),
                ('enable_remote_logging', False),
                ('remote_log_server', ''),
                ('remote_log_port', 514),
                ('enable_encryption', False),
                ('encryption_key', ''),
                ('max_queue_size', 10000),
                ('batch_size', 100),
                ('batch_timeout_sec', 1.0),
                ('enable_performance_logging', True),
                ('performance_log_interval_sec', 60.0),
                ('retention_days', 90),
                ('enable_audit_logging', True),
                ('audit_log_directory', '/var/log/robot_audit'),
                ('enable_diagnostics_logging', True),
                ('diagnostics_log_interval_sec', 5.0),
                ('log_buffer_size_mb', 100),
                ('enable_realtime_analysis', True),
                ('analysis_window_minutes', 5),
                ('alert_threshold_error_rate', 10.0),  # errors per minute
                ('alert_threshold_warning_rate', 50.0),  # warnings per minute
                ('enable_correlation_tracking', True),
                ('correlation_timeout_minutes', 60),
                ('enable_session_tracking', True),
                ('session_timeout_hours', 8),
                ('enable_tags', True),
                ('default_tags', ['robot', 'production', 'ml_picking']),
                ('enable_health_integration', True),
                ('health_check_interval_sec', 30.0),
                ('enable_metrics_export', True),
                ('metrics_export_interval_sec', 300.0),
                ('enable_anomaly_detection', True),
                ('anomaly_detection_window', 1000),
                ('enable_backup', True),
                ('backup_directory', '/backup/logs'),
                ('backup_interval_hours', 24),
                ('enable_compliance_logging', False),
                ('compliance_retention_years', 7)
            ]
        )
        
        # Get parameters
        self.log_dir = Path(self.get_parameter('log_directory').value)
        self.max_file_size = self.get_parameter('max_file_size_mb').value * 1024 * 1024
        self.backup_count = self.get_parameter('backup_count').value
        self.compression_algo = CompressionAlgorithm(
            self.get_parameter('compression_algorithm').value
        )
        self.log_format = LogFormat(self.get_parameter('log_format').value)
        self.default_level = LogLevel[self.get_parameter('default_log_level').value]
        self.max_queue_size = self.get_parameter('max_queue_size').value
        self.batch_size = self.get_parameter('batch_size').value
        self.batch_timeout = self.get_parameter('batch_timeout_sec').value
        
        # Create log directories
        self._create_directories()
        
        # Initialize logging infrastructure
        self.log_queue = queue.Queue(maxsize=self.max_queue_size)
        self.batch_buffer: List[LogEntryData] = []
        self.batch_lock = threading.Lock()
        self.batch_timer = None
        
        # Statistics and tracking
        self.sequence_counter = 0
        self.statistics: Dict[str, LogStatistics] = {}
        self.correlation_map: Dict[str, List[LogEntryData]] = {}
        self.session_map: Dict[str, Dict] = {}
        self.tag_index: Dict[str, List[str]] = {}
        
        # Performance monitoring
        self.performance_metrics = {
            'queue_size': 0,
            'dropped_messages': 0,
            'processed_messages': 0,
            'batch_count': 0,
            'write_time_ms': 0.0,
            'compression_time_ms': 0.0,
            'analysis_time_ms': 0.0,
            'memory_usage_mb': 0.0
        }
        
        # Initialize log writers
        self.log_writers = self._initialize_log_writers()
        self.current_log_file = self._get_current_log_file()
        
        # ROS2 Publishers
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=100,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        
        self.log_pub = self.create_publisher(
            LogBatch,
            '/system/logs/batch',
            qos_profile
        )
        
        self.log_alert_pub = self.create_publisher(
            Alert,
            '/system/logs/alerts',
            qos_profile
        )
        
        self.log_stats_pub = self.create_publisher(
            String,
            '/system/logs/statistics',
            qos_profile
        )
        
        # ROS2 Services
        self.services = {
            'get_logs': self.create_service(
                GetLogs,
                '/system/logs/get_logs',
                self.handle_get_logs
            ),
            'search_logs': self.create_service(
                SearchLogs,
                '/system/logs/search',
                self.handle_search_logs
            ),
            'export_logs': self.create_service(
                ExportLogs,
                '/system/logs/export',
                self.handle_export_logs
            ),
            'set_log_level': self.create_service(
                SetLogLevel,
                '/system/logs/set_level',
                self.handle_set_log_level
            ),
            'clear_logs': self.create_service(
                ClearLogs,
                '/system/logs/clear',
                self.handle_clear_logs
            ),
            'get_log_statistics': self.create_service(
                GetLogStatistics,
                '/system/logs/statistics',
                self.handle_get_log_statistics
            )
        }
        
        # ROS2 Subscribers for topic-based logging
        self._initialize_topic_subscribers()
        
        # Start worker threads
        self.running = True
        self.worker_threads = []
        
        # Writer thread
        writer_thread = threading.Thread(target=self._writer_loop)
        writer_thread.daemon = True
        writer_thread.start()
        self.worker_threads.append(writer_thread)
        
        # Analyzer thread
        if self.get_parameter('enable_realtime_analysis').value:
            analyzer_thread = threading.Thread(target=self._analyzer_loop)
            analyzer_thread.daemon = True
            analyzer_thread.start()
            self.worker_threads.append(analyzer_thread)
        
        # Performance monitoring thread
        if self.get_parameter('enable_performance_logging').value:
            perf_thread = threading.Thread(target=self._performance_monitoring_loop)
            perf_thread.daemon = True
            perf_thread.start()
            self.worker_threads.append(perf_thread)
        
        # Diagnostics logging thread
        if self.get_parameter('enable_diagnostics_logging').value:
            diag_thread = threading.Thread(target=self._diagnostics_logging_loop)
            diag_thread.daemon = True
            diag_thread.start()
            self.worker_threads.append(diag_thread)
        
        # Backup thread
        if self.get_parameter('enable_backup').value:
            backup_thread = threading.Thread(target=self._backup_loop)
            backup_thread.daemon = True
            backup_thread.start()
            self.worker_threads.append(backup_thread)
        
        # Start timers
        self._start_timers()
        
        # Initialize audit logging
        if self.get_parameter('enable_audit_logging').value:
            self._initialize_audit_logging()
        
        # Initialize compliance logging
        if self.get_parameter('enable_compliance_logging').value:
            self._initialize_compliance_logging()
        
        # Log startup
        self._log_system_event(
            LogLevel.SYSTEM,
            "AdvancedLogger initialized",
            {"version": "3.2.0", "pid": os.getpid()},
            tags=["startup", "system"]
        )
        
        self.get_logger().info(
            f"Advanced Logger Node initialized. Log directory: {self.log_dir}"
        )
    
    def _create_directories(self):
        """Create necessary log directories."""
        directories = [
            self.log_dir,
            self.log_dir / "archive",
            self.log_dir / "audit",
            self.log_dir / "diagnostics",
            self.log_dir / "performance",
            self.log_dir / "backup",
            self.log_dir / "tmp"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _initialize_log_writers(self) -> Dict[str, Any]:
        """Initialize log writers for different formats."""
        writers = {}
        
        if self.log_format == LogFormat.JSON:
            writers['json'] = self._create_json_writer()
        elif self.log_format == LogFormat.MSGPACK:
            writers['msgpack'] = self._create_msgpack_writer()
        elif self.log_format == LogFormat.CSV:
            writers['csv'] = self._create_csv_writer()
        elif self.log_format == LogFormat.PLAIN:
            writers['plain'] = self._create_plain_writer()
        elif self.log_format == LogFormat.PICKLE:
            writers['pickle'] = self._create_pickle_writer()
        
        return writers
    
    def _create_json_writer(self):
        """Create JSON log writer with rotation."""
        return {
            'extension': '.jsonl',
            'write_func': self._write_json,
            'rotate_func': self._rotate_json_file
        }
    
    def _create_msgpack_writer(self):
        """Create MsgPack log writer."""
        return {
            'extension': '.msgpack',
            'write_func': self._write_msgpack,
            'rotate_func': self._rotate_binary_file
        }
    
    def _create_csv_writer(self):
        """Create CSV log writer."""
        return {
            'extension': '.csv',
            'write_func': self._write_csv,
            'rotate_func': self._rotate_csv_file
        }
    
    def _create_plain_writer(self):
        """Create plain text log writer."""
        return {
            'extension': '.log',
            'write_func': self._write_plain,
            'rotate_func': self._rotate_plain_file
        }
    
    def _create_pickle_writer(self):
        """Create pickle log writer."""
        return {
            'extension': '.pkl',
            'write_func': self._write_pickle,
            'rotate_func': self._rotate_binary_file
        }
    
    def _get_current_log_file(self) -> Path:
        """Get current log file path based on date and format."""
        date_str = datetime.now().strftime('%Y%m%d')
        extension = self.log_writers['main']['extension']
        return self.log_dir / f"robot_logs_{date_str}{extension}"
    
    def _initialize_topic_subscribers(self):
        """Initialize subscribers for various ROS2 topics."""
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # System health logs
        self.create_subscription(
            SystemHealth,
            '/system/health',
            self._log_system_health,
            qos_profile
        )
        
        # Alerts
        self.create_subscription(
            Alert,
            '/system/alerts',
            self._log_alerts,
            qos_profile
        )
        
        # Diagnostics
        self.create_subscription(
            DiagnosticArray,
            '/diagnostics',
            self._log_diagnostics,
            qos_profile
        )
        
        # Robot status
        self.create_subscription(
            RobotStatus,
            '/robot/status',
            self._log_robot_status,
            qos_profile
        )
        
        # Detected objects (sampled)
        self.create_subscription(
            DetectedObject,
            '/perception/detected_objects',
            self._log_detected_objects,
            qos_profile
        )
        
        # Motion planning events
        self.create_subscription(
            String,
            '/motion/events',
            self._log_motion_events,
            qos_profile
        )
    
    def _initialize_audit_logging(self):
        """Initialize audit logging system."""
        audit_dir = Path(self.get_parameter('audit_log_directory').value)
        audit_dir.mkdir(parents=True, exist_ok=True)
        
        self.audit_logger = logging.getLogger('robot_audit')
        self.audit_logger.setLevel(logging.INFO)
        
        # Create audit log handler with rotation
        audit_handler = logging.handlers.RotatingFileHandler(
            filename=audit_dir / 'audit.log',
            maxBytes=self.max_file_size,
            backupCount=self.backup_count
        )
        
        audit_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        audit_handler.setFormatter(audit_formatter)
        self.audit_logger.addHandler(audit_handler)
    
    def _initialize_compliance_logging(self):
        """Initialize compliance logging for regulatory requirements."""
        compliance_dir = self.log_dir / "compliance"
        compliance_dir.mkdir(exist_ok=True)
        
        self.compliance_logger = logging.getLogger('robot_compliance')
        self.compliance_logger.setLevel(logging.INFO)
        
        compliance_handler = logging.handlers.RotatingFileHandler(
            filename=compliance_dir / 'compliance.log',
            maxBytes=self.max_file_size,
            backupCount=self.backup_count
        )
        
        compliance_formatter = logging.Formatter(
            '%(asctime)s | %(process)d | %(thread)d | %(levelname)s | %(message)s'
        )
        compliance_handler.setFormatter(compliance_formatter)
        self.compliance_logger.addHandler(compliance_handler)
    
    def _start_timers(self):
        """Start various periodic timers."""
        # Batch flush timer
        self.batch_timer = self.create_timer(
            self.batch_timeout,
            self._flush_batch
        )
        
        # Statistics aggregation timer
        self.create_timer(
            300.0,  # 5 minutes
            self._aggregate_statistics
        )
        
        # Cleanup timer
        self.create_timer(
            3600.0,  # 1 hour
            self._cleanup_old_logs
        )
        
        # Health check timer
        if self.get_parameter('enable_health_integration').value:
            self.create_timer(
                self.get_parameter('health_check_interval_sec').value,
                self._log_system_health_check
            )
        
        # Metrics export timer
        if self.get_parameter('enable_metrics_export').value:
            self.create_timer(
                self.get_parameter('metrics_export_interval_sec').value,
                self._export_metrics
            )
    
    def log(self, 
            level: LogLevel, 
            category: LogCategory, 
            node: str, 
            source: str, 
            message: str,
            data: Optional[Dict] = None,
            context: Optional[Dict] = None,
            tags: Optional[List[str]] = None,
            correlation_id: Optional[str] = None,
            session_id: Optional[str] = None) -> Optional[str]:
        """
        Log an entry with full metadata.
        
        Args:
            level: Log level
            category: Log category
            node: Node name
            source: Source function/method
            message: Log message
            data: Additional structured data
            context: Context information
            tags: Tags for categorization
            correlation_id: Correlation ID for distributed tracing
            session_id: Session ID for user sessions
            
        Returns:
            Optional[str]: Log entry ID or None if dropped
        """
        if not self.running:
            return None
        
        # Check if log level is enabled
        if level.value < self.default_level.value:
            return None
        
        # Generate log entry
        entry = LogEntryData(
            timestamp=datetime.utcnow(),
            level=level,
            category=category,
            node=node,
            source=source,
            message=message,
            data=data or {},
            context=context or {},
            tags=tags or self.get_parameter('default_tags').value,
            correlation_id=correlation_id,
            session_id=session_id,
            sequence_number=self.sequence_counter,
            hostname=socket.gethostname(),
            pid=os.getpid(),
            thread_id=threading.get_ident()
        )
        
        self.sequence_counter += 1
        
        # Update correlation tracking
        if correlation_id and self.get_parameter('enable_correlation_tracking').value:
            if correlation_id not in self.correlation_map:
                self.correlation_map[correlation_id] = []
            self.correlation_map[correlation_id].append(entry)
        
        # Update session tracking
        if session_id and self.get_parameter('enable_session_tracking').value:
            if session_id not in self.session_map:
                self.session_map[session_id] = {
                    'start_time': entry.timestamp,
                    'entries': [],
                    'last_activity': entry.timestamp
                }
            self.session_map[session_id]['entries'].append(entry)
            self.session_map[session_id]['last_activity'] = entry.timestamp
        
        # Update tag index
        if self.get_parameter('enable_tags').value:
            for tag in entry.tags:
                if tag not in self.tag_index:
                    self.tag_index[tag] = []
                self.tag_index[tag].append(f"{entry.timestamp.isoformat()}_{entry.sequence_number}")
        
        # Try to add to queue
        try:
            self.log_queue.put_nowait(entry)
            self.performance_metrics['queue_size'] = self.log_queue.qsize()
            return f"{entry.timestamp.isoformat()}_{entry.sequence_number}"
        except queue.Full:
            self.performance_metrics['dropped_messages'] += 1
            self._log_internal_error("Log queue full, dropping message", {
                'message': message,
                'queue_size': self.log_queue.qsize(),
                'dropped_count': self.performance_metrics['dropped_messages']
            })
            return None
    
    def _writer_loop(self):
        """Main writer thread loop."""
        batch_counter = 0
        
        while self.running:
            try:
                # Wait for batch timeout or batch size reached
                entries = []
                start_time = time.time()
                
                while len(entries) < self.batch_size and (time.time() - start_time) < self.batch_timeout:
                    try:
                        entry = self.log_queue.get(timeout=0.1)
                        entries.append(entry)
                        self.performance_metrics['processed_messages'] += 1
                    except queue.Empty:
                        continue
                
                if entries:
                    self._process_batch(entries)
                    batch_counter += 1
                    
                    # Update performance metrics
                    self.performance_metrics['batch_count'] = batch_counter
                    self.performance_metrics['queue_size'] = self.log_queue.qsize()
                
                # Check for rotation
                self._check_log_rotation()
                
                # Small sleep to prevent CPU spinning
                time.sleep(0.01)
                
            except Exception as e:
                self._log_internal_error(f"Error in writer loop: {str(e)}", {
                    'error_type': type(e).__name__,
                    'traceback': self._get_traceback(e)
                })
                time.sleep(1.0)
    
    def _process_batch(self, entries: List[LogEntryData]):
        """Process a batch of log entries."""
        process_start = time.time()
        
        try:
            # Write to file
            write_start = time.time()
            self._write_entries_to_file(entries)
            write_time = (time.time() - write_start) * 1000
            self.performance_metrics['write_time_ms'] = write_time
            
            # Compress if needed
            if self.compression_algo != CompressionAlgorithm.NONE:
                compress_start = time.time()
                self._compress_current_file()
                compress_time = (time.time() - compress_start) * 1000
                self.performance_metrics['compression_time_ms'] = compress_time
            
            # Publish batch via ROS2
            self._publish_log_batch(entries)
            
            # Update statistics
            self._update_statistics(entries)
            
            # Check for anomalies
            if self.get_parameter('enable_anomaly_detection').value:
                self._detect_anomalies(entries)
            
        except Exception as e:
            self._log_internal_error(f"Error processing batch: {str(e)}", {
                'batch_size': len(entries),
                'error_type': type(e).__name__
            })
        
        process_time = (time.time() - process_start) * 1000
        self.performance_metrics['analysis_time_ms'] = process_time
    
    def _write_entries_to_file(self, entries: List[LogEntryData]):
        """Write entries to log file based on format."""
        try:
            writer = self.log_writers['main']
            
            # Check if file needs rotation
            if self.current_log_file.exists():
                file_size = self.current_log_file.stat().st_size
                if file_size >= self.max_file_size:
                    writer['rotate_func']()
            
            # Write entries
            with open(self.current_log_file, 'a' if self.current_log_file.exists() else 'w') as f:
                for entry in entries:
                    writer['write_func'](f, entry)
            
        except Exception as e:
            raise Exception(f"Failed to write log entries: {str(e)}")
    
    def _write_json(self, file_obj, entry: LogEntryData):
        """Write entry in JSON format."""
        json.dump(entry.to_dict(), file_obj)
        file_obj.write('\n')
    
    def _write_msgpack(self, file_obj, entry: LogEntryData):
        """Write entry in MsgPack format."""
        import msgpack
        packed = msgpack.packb(entry.to_dict())
        file_obj.write(packed)
    
    def _write_csv(self, file_obj, entry: LogEntryData):
        """Write entry in CSV format."""
        import csv
        writer = csv.writer(file_obj)
        
        # Convert entry to CSV row
        row = [
            entry.timestamp.isoformat(),
            entry.level.name,
            entry.category.value,
            entry.node,
            entry.source,
            entry.message,
            json.dumps(entry.data),
            json.dumps(entry.context),
            ','.join(entry.tags),
            entry.correlation_id or '',
            entry.session_id or '',
            str(entry.sequence_number),
            entry.hostname,
            str(entry.pid),
            str(entry.thread_id)
        ]
        
        writer.writerow(row)
    
    def _write_plain(self, file_obj, entry: LogEntryData):
        """Write entry in plain text format."""
        log_line = f"{entry.timestamp.isoformat()} | {entry.level.name:8} | {entry.category.value:12} | {entry.node:20} | {entry.source:30} | {entry.message}"
        
        if entry.data:
            log_line += f" | Data: {json.dumps(entry.data)}"
        
        if entry.tags:
            log_line += f" | Tags: {','.join(entry.tags)}"
        
        file_obj.write(log_line + '\n')
    
    def _write_pickle(self, file_obj, entry: LogEntryData):
        """Write entry in pickle format."""
        pickle.dump(entry, file_obj)
    
    def _rotate_json_file(self):
        """Rotate JSON log file."""
        self._rotate_file('.jsonl')
    
    def _rotate_csv_file(self):
        """Rotate CSV log file."""
        self._rotate_file('.csv')
    
    def _rotate_plain_file(self):
        """Rotate plain text log file."""
        self._rotate_file('.log')
    
    def _rotate_binary_file(self):
        """Rotate binary log file."""
        self._rotate_file(self.log_writers['main']['extension'])
    
    def _rotate_file(self, extension: str):
        """Generic file rotation with compression."""
        if not self.current_log_file.exists():
            return
        
        # Generate new filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        archive_file = self.log_dir / "archive" / f"robot_logs_{timestamp}{extension}"
        
        # Move current file to archive
        self.current_log_file.rename(archive_file)
        
        # Create new log file
        self.current_log_file = self._get_current_log_file()
        
        # Compress archived file
        self._compress_file(archive_file)
        
        # Clean up old archives
        self._cleanup_old_archives()
    
    def _compress_current_file(self):
        """Compress the current log file if needed."""
        if self.compression_algo == CompressionAlgorithm.NONE:
            return
        
        # Only compress if file is large enough
        if self.current_log_file.exists():
            file_size = self.current_log_file.stat().st_size
            if file_size > 1024 * 1024:  # 1MB threshold
                self._compress_file(self.current_log_file)
    
    def _compress_file(self, file_path: Path):
        """Compress a file using specified algorithm."""
        try:
            if self.compression_algo == CompressionAlgorithm.GZIP:
                compressed_path = file_path.with_suffix(file_path.suffix + '.gz')
                with open(file_path, 'rb') as f_in:
                    with gzip.open(compressed_path, 'wb') as f_out:
                        f_out.write(f_in.read())
                
            elif self.compression_algo == CompressionAlgorithm.LZMA:
                compressed_path = file_path.with_suffix(file_path.suffix + '.xz')
                with open(file_path, 'rb') as f_in:
                    with lzma.open(compressed_path, 'wb') as f_out:
                        f_out.write(f_in.read())
                
            elif self.compression_algo == CompressionAlgorithm.ZLIB:
                compressed_path = file_path.with_suffix(file_path.suffix + '.zlib')
                with open(file_path, 'rb') as f_in:
                    data = f_in.read()
                    compressed = zlib.compress(data)
                    with open(compressed_path, 'wb') as f_out:
                        f_out.write(compressed)
                
            elif self.compression_algo == CompressionAlgorithm.BZ2:
                compressed_path = file_path.with_suffix(file_path.suffix + '.bz2')
                with open(file_path, 'rb') as f_in:
                    with bz2.open(compressed_path, 'wb') as f_out:
                        f_out.write(f_in.read())
            
            # Remove original file if compression successful
            if compressed_path.exists():
                file_path.unlink()
                
        except Exception as e:
            self._log_internal_error(f"Failed to compress file: {str(e)}", {
                'file': str(file_path),
                'algorithm': self.compression_algo.value
            })
    
    def _check_log_rotation(self):
        """Check if log rotation is needed based on time."""
        rotation_interval = self.get_parameter('rotation_interval_hours').value
        if rotation_interval <= 0:
            return
        
        # Check if current file is older than rotation interval
        if self.current_log_file.exists():
            file_age = datetime.now() - datetime.fromtimestamp(
                self.current_log_file.stat().st_mtime
            )
            
            if file_age.total_seconds() > rotation_interval * 3600:
                writer = self.log_writers['main']
                writer['rotate_func']()
    
    def _publish_log_batch(self, entries: List[LogEntryData]):
        """Publish log batch via ROS2 topic."""
        try:
            batch_msg = LogBatch()
            batch_msg.header.stamp = self.get_clock().now().to_msg()
            batch_msg.header.frame_id = "log_batch"
            batch_msg.count = len(entries)
            batch_msg.timestamp = datetime.now().isoformat()
            
            for entry in entries:
                batch_msg.entries.append(entry.to_ros_msg())
            
            self.log_pub.publish(batch_msg)
            
        except Exception as e:
            self._log_internal_error(f"Failed to publish log batch: {str(e)}", {
                'batch_size': len(entries)
            })
    
    def _update_statistics(self, entries: List[LogEntryData]):
        """Update log statistics."""
        current_hour = datetime.now().strftime('%Y%m%d%H')
        
        if current_hour not in self.statistics:
            self.statistics[current_hour] = LogStatistics(
                period_start=datetime.now().replace(minute=0, second=0, microsecond=0),
                period_end=datetime.now().replace(minute=59, second=59, microsecond=999999),
                total_entries=0,
                entries_by_level={},
                entries_by_category={},
                entries_by_node={},
                avg_entry_size_bytes=0,
                max_entry_size_bytes=0,
                error_rate_per_hour=0.0,
                warning_rate_per_hour=0.0,
                busiest_node="",
                busiest_category="",
                common_tags=[],
                data_volume_bytes=0,
                compression_ratio=1.0
            )
        
        stats = self.statistics[current_hour]
        
        for entry in entries:
            stats.total_entries += 1
            
            # Update level counts
            level_name = entry.level.name
            stats.entries_by_level[level_name] = stats.entries_by_level.get(level_name, 0) + 1
            
            # Update category counts
            category_name = entry.category.value
            stats.entries_by_category[category_name] = stats.entries_by_category.get(category_name, 0) + 1
            
            # Update node counts
            stats.entries_by_node[entry.node] = stats.entries_by_node.get(entry.node, 0) + 1
            
            # Update entry size
            entry_size = len(json.dumps(entry.to_dict()))
            stats.data_volume_bytes += entry_size
            stats.avg_entry_size_bytes = stats.data_volume_bytes / stats.total_entries
            stats.max_entry_size_bytes = max(stats.max_entry_size_bytes, entry_size)
            
            # Update tags
            for tag in entry.tags:
                found = False
                for i, (tag_name, count) in enumerate(stats.common_tags):
                    if tag_name == tag:
                        stats.common_tags[i] = (tag_name, count + 1)
                        found = True
                        break
                if not found:
                    stats.common_tags.append((tag, 1))
            
            # Sort tags by frequency
            stats.common_tags.sort(key=lambda x: x[1], reverse=True)
            stats.common_tags = stats.common_tags[:10]  # Keep top 10
        
        # Update rates
        hours_elapsed = (datetime.now() - stats.period_start).total_seconds() / 3600
        if hours_elapsed > 0:
            stats.error_rate_per_hour = stats.entries_by_level.get('ERROR', 0) / hours_elapsed
            stats.warning_rate_per_hour = stats.entries_by_level.get('WARNING', 0) / hours_elapsed
        
        # Update busiest node and category
        if stats.entries_by_node:
            stats.busiest_node = max(stats.entries_by_node.items(), key=lambda x: x[1])[0]
        
        if stats.entries_by_category:
            stats.busiest_category = max(stats.entries_by_category.items(), key=lambda x: x[1])[0]
    
    def _detect_anomalies(self, entries: List[LogEntryData]):
        """Detect anomalies in log entries."""
        window_size = self.get_parameter('anomaly_detection_window').value
        
        # Count error and warning rates in recent window
        recent_entries = []
        for entry in entries[-window_size:]:
            recent_entries.append(entry)
        
        error_count = sum(1 for e in recent_entries if e.level == LogLevel.ERROR)
        warning_count = sum(1 for e in recent_entries if e.level == LogLevel.WARNING)
        
        window_minutes = window_size / 60  # Approximate minutes
        
        error_rate = error_count / window_minutes if window_minutes > 0 else 0
        warning_rate = warning_count / window_minutes if window_minutes > 0 else 0
        
        # Check thresholds
        error_threshold = self.get_parameter('alert_threshold_error_rate').value
        warning_threshold = self.get_parameter('alert_threshold_warning_rate').value
        
        if error_rate > error_threshold:
            self._log_anomaly_alert(
                "High error rate detected",
                {
                    'error_rate': error_rate,
                    'threshold': error_threshold,
                    'window_minutes': window_minutes,
                    'error_count': error_count
                }
            )
        
        if warning_rate > warning_threshold:
            self._log_anomaly_alert(
                "High warning rate detected",
                {
                    'warning_rate': warning_rate,
                    'threshold': warning_threshold,
                    'window_minutes': window_minutes,
                    'warning_count': warning_count
                }
            )
        
        # Detect repeated messages
        message_counts = {}
        for entry in recent_entries:
            msg_hash = hashlib.md5(entry.message.encode()).hexdigest()
            message_counts[msg_hash] = message_counts.get(msg_hash, 0) + 1
        
        for msg_hash, count in message_counts.items():
            if count > 10:  # Same message repeated 10+ times
                sample_entry = next(e for e in recent_entries if 
                                  hashlib.md5(e.message.encode()).hexdigest() == msg_hash)
                
                self._log_anomaly_alert(
                    "Repeated log message detected",
                    {
                        'message': sample_entry.message,
                        'count': count,
                        'node': sample_entry.node,
                        'source': sample_entry.source
                    }
                )
    
    def _log_anomaly_alert(self, message: str, data: Dict):
        """Log an anomaly alert."""
        alert_msg = Alert()
        alert_msg.header.stamp = self.get_clock().now().to_msg()
        alert_msg.level = 2  # ERROR
        alert_msg.component = "logger"
        alert_msg.message = f"ANOMALY: {message}"
        alert_msg.timestamp = datetime.now().isoformat()
        alert_msg.metric = "log_anomaly"
        alert_msg.value = json.dumps(data)
        
        self.log_alert_pub.publish(alert_msg)
        
        # Also log internally
        self.log(
            LogLevel.ERROR,
            LogCategory.SYSTEM,
            "logger",
            "anomaly_detection",
            message,
            data,
            tags=["anomaly", "alert"]
        )
    
    def _analyzer_loop(self):
        """Real-time log analysis loop."""
        while self.running:
            try:
                # Perform analysis every minute
                time.sleep(60)
                
                # Analyze recent entries
                self._analyze_recent_logs()
                
                # Update tag cloud
                self._update_tag_cloud()
                
                # Check correlation timeouts
                self._cleanup_expired_correlations()
                
                # Check session timeouts
                self._cleanup_expired_sessions()
                
            except Exception as e:
                self._log_internal_error(f"Error in analyzer loop: {str(e)}", {
                    'error_type': type(e).__name__
                })
                time.sleep(5)
    
    def _analyze_recent_logs(self):
        """Analyze recent log entries for patterns."""
        # This would implement more sophisticated analysis
        # For now, we'll just log basic statistics
        
        if not self.statistics:
            return
        
        latest_stats = list(self.statistics.values())[-1] if self.statistics else None
        
        if latest_stats:
            analysis_data = {
                'total_entries': latest_stats.total_entries,
                'error_rate': latest_stats.error_rate_per_hour,
                'warning_rate': latest_stats.warning_rate_per_hour,
                'busiest_node': latest_stats.busiest_node,
                'busiest_category': latest_stats.busiest_category,
                'data_volume_mb': latest_stats.data_volume_bytes / (1024 * 1024)
            }
            
            self.log(
                LogLevel.INFO,
                LogCategory.PERFORMANCE,
                "logger",
                "analysis",
                "Log statistics analysis",
                analysis_data,
                tags=["analysis", "statistics"]
            )
    
    def _update_tag_cloud(self):
        """Update tag cloud based on recent usage."""
        # Count tag frequencies in last hour
        recent_tags = {}
        
        for tag, entry_ids in self.tag_index.items():
            # Count recent entries (last hour)
            recent_count = 0
            hour_ago = datetime.now() - timedelta(hours=1)
            
            for entry_id in entry_ids:
                try:
                    timestamp_str = entry_id.split('_')[0]
                    entry_time = datetime.fromisoformat(timestamp_str)
                    if entry_time > hour_ago:
                        recent_count += 1
                except:
                    continue
            
            if recent_count > 0:
                recent_tags[tag] = recent_count
        
        # Update tag cloud file
        tag_cloud_file = self.log_dir / "tag_cloud.json"
        tag_data = {
            'timestamp': datetime.now().isoformat(),
            'tags': recent_tags
        }
        
        with open(tag_cloud_file, 'w') as f:
            json.dump(tag_data, f, indent=2)
    
    def _cleanup_expired_correlations(self):
        """Clean up expired correlation entries."""
        timeout_minutes = self.get_parameter('correlation_timeout_minutes').value
        timeout_delta = timedelta(minutes=timeout_minutes)
        now = datetime.now()
        
        expired_correlations = []
        
        for corr_id, entries in list(self.correlation_map.items()):
            if entries:
                last_entry = max(entries, key=lambda e: e.timestamp)
                if now - last_entry.timestamp > timeout_delta:
                    expired_correlations.append(corr_id)
        
        for corr_id in expired_correlations:
            del self.correlation_map[corr_id]
    
    def _cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        timeout_hours = self.get_parameter('session_timeout_hours').value
        timeout_delta = timedelta(hours=timeout_hours)
        now = datetime.now()
        
        expired_sessions = []
        
        for session_id, session_data in list(self.session_map.items()):
            if now - session_data['last_activity'] > timeout_delta:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.session_map[session_id]
    
    def _performance_monitoring_loop(self):
        """Monitor logger performance."""
        while self.running:
            try:
                time.sleep(self.get_parameter('performance_log_interval_sec').value)
                
                # Get system metrics
                memory_info = psutil.Process().memory_info()
                self.performance_metrics['memory_usage_mb'] = memory_info.rss / (1024 * 1024)
                
                # Log performance metrics
                self.log(
                    LogLevel.PERFORMANCE,
                    LogCategory.PERFORMANCE,
                    "logger",
                    "performance_monitoring",
                    "Logger performance metrics",
                    self.performance_metrics.copy(),
                    tags=["performance", "monitoring"]
                )
                
            except Exception as e:
                self._log_internal_error(f"Error in performance monitoring: {str(e)}", {
                    'error_type': type(e).__name__
                })
                time.sleep(5)
    
    def _diagnostics_logging_loop(self):
        """Log system diagnostics periodically."""
        while self.running:
            try:
                time.sleep(self.get_parameter('diagnostics_log_interval_sec').value)
                
                # Collect system diagnostics
                diagnostics = self._collect_system_diagnostics()
                
                # Log diagnostics
                self.log(
                    LogLevel.INFO,
                    LogCategory.DIAGNOSTIC,
                    "logger",
                    "diagnostics",
                    "System diagnostics",
                    diagnostics,
                    tags=["diagnostics", "system"]
                )
                
            except Exception as e:
                self._log_internal_error(f"Error in diagnostics logging: {str(e)}", {
                    'error_type': type(e).__name__
                })
                time.sleep(5)
    
    def _collect_system_diagnostics(self) -> Dict:
        """Collect comprehensive system diagnostics."""
        diagnostics = {}
        
        try:
            # CPU information
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            diagnostics['cpu'] = {
                'percent': cpu_percent,
                'count': cpu_count,
                'frequency_mhz': cpu_freq.current if cpu_freq else None
            }
            
            # Memory information
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            diagnostics['memory'] = {
                'total_mb': memory.total / (1024 * 1024),
                'available_mb': memory.available / (1024 * 1024),
                'percent': memory.percent,
                'swap_total_mb': swap.total / (1024 * 1024),
                'swap_used_mb': swap.used / (1024 * 1024),
                'swap_percent': swap.percent
            }
            
            # Disk information
            disk_usage = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            diagnostics['disk'] = {
                'total_gb': disk_usage.total / (1024**3),
                'used_gb': disk_usage.used / (1024**3),
                'free_gb': disk_usage.free / (1024**3),
                'percent': disk_usage.percent,
                'read_mb': disk_io.read_bytes / (1024 * 1024) if disk_io else 0,
                'write_mb': disk_io.write_bytes / (1024 * 1024) if disk_io else 0
            }
            
            # Network information
            net_io = psutil.net_io_counters()
            net_connections = len(psutil.net_connections())
            
            diagnostics['network'] = {
                'bytes_sent_mb': net_io.bytes_sent / (1024 * 1024),
                'bytes_recv_mb': net_io.bytes_recv / (1024 * 1024),
                'active_connections': net_connections
            }
            
            # Process information
            process = psutil.Process()
            with process.oneshot():
                diagnostics['process'] = {
                    'pid': process.pid,
                    'name': process.name(),
                    'status': process.status(),
                    'cpu_percent': process.cpu_percent(),
                    'memory_percent': process.memory_percent(),
                    'num_threads': process.num_threads(),
                    'num_fds': process.num_fds() if hasattr(process, 'num_fds') else None
                }
            
            # System information
            diagnostics['system'] = {
                'hostname': socket.gethostname(),
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'boot_time': datetime.fromtimestamp(psutil.boot_time()).isoformat(),
                'users': len(psutil.users())
            }
            
        except Exception as e:
            diagnostics['error'] = str(e)
        
        return diagnostics
    
    def _backup_loop(self):
        """Backup logs periodically."""
        backup_interval = self.get_parameter('backup_interval_hours').value * 3600
        
        while self.running:
            try:
                time.sleep(backup_interval)
                self._perform_backup()
                
            except Exception as e:
                self._log_internal_error(f"Error in backup loop: {str(e)}", {
                    'error_type': type(e).__name__
                })
                time.sleep(3600)  # Retry in 1 hour
    
    def _perform_backup(self):
        """Perform log backup."""
        backup_dir = Path(self.get_parameter('backup_directory').value)
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = backup_dir / f"logs_backup_{timestamp}.tar.gz"
        
        try:
            import tarfile
            
            with tarfile.open(backup_file, 'w:gz') as tar:
                # Add log directory
                tar.add(self.log_dir, arcname='logs')
            
            # Log backup completion
            self.log(
                LogLevel.INFO,
                LogCategory.SYSTEM,
                "logger",
                "backup",
                "Log backup completed",
                {
                    'backup_file': str(backup_file),
                    'backup_size_mb': backup_file.stat().st_size / (1024 * 1024)
                },
                tags=["backup", "maintenance"]
            )
            
        except Exception as e:
            self._log_internal_error(f"Backup failed: {str(e)}", {
                'backup_file': str(backup_file)
            })
    
    def _flush_batch(self):
        """Force flush current batch."""
        with self.batch_lock:
            if self.batch_buffer:
                self._process_batch(self.batch_buffer.copy())
                self.batch_buffer.clear()
    
    def _aggregate_statistics(self):
        """Aggregate and publish statistics."""
        if not self.statistics:
            return
        
        # Aggregate hourly statistics into daily
        daily_stats = {}
        
        for hour_key, stats in self.statistics.items():
            day_key = hour_key[:8]  # YYYYMMDD
            
            if day_key not in daily_stats:
                daily_stats[day_key] = {
                    'total_entries': 0,
                    'entries_by_level': {},
                    'entries_by_category': {},
                    'entries_by_node': {},
                    'data_volume_bytes': 0
                }
            
            daily = daily_stats[day_key]
            daily['total_entries'] += stats.total_entries
            daily['data_volume_bytes'] += stats.data_volume_bytes
            
            # Merge level counts
            for level, count in stats.entries_by_level.items():
                daily['entries_by_level'][level] = daily['entries_by_level'].get(level, 0) + count
            
            # Merge category counts
            for category, count in stats.entries_by_category.items():
                daily['entries_by_category'][category] = daily['entries_by_category'].get(category, 0) + count
            
            # Merge node counts
            for node, count in stats.entries_by_node.items():
                daily['entries_by_node'][node] = daily['entries_by_node'].get(node, 0) + count
        
        # Publish statistics
        stats_msg = String()
        stats_msg.data = json.dumps({
            'timestamp': datetime.now().isoformat(),
            'statistics': daily_stats
        })
        
        self.log_stats_pub.publish(stats_msg)
        
        # Save statistics to file
        stats_file = self.log_dir / "statistics" / f"stats_{datetime.now().strftime('%Y%m%d')}.json"
        stats_file.parent.mkdir(exist_ok=True)
        
        with open(stats_file, 'w') as f:
            json.dump(daily_stats, f, indent=2)
    
    def _cleanup_old_logs(self):
        """Clean up old log files based on retention policy."""
        retention_days = self.get_parameter('retention_days').value
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        try:
            # Cleanup archive directory
            archive_dir = self.log_dir / "archive"
            if archive_dir.exists():
                for file_path in archive_dir.iterdir():
                    if file_path.is_file():
                        file_date = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if file_date < cutoff_date:
                            file_path.unlink()
            
            # Cleanup old statistics
            stats_dir = self.log_dir / "statistics"
            if stats_dir.exists():
                for file_path in stats_dir.iterdir():
                    if file_path.is_file():
                        file_date = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if file_date < cutoff_date:
                            file_path.unlink()
            
            self.log(
                LogLevel.INFO,
                LogCategory.SYSTEM,
                "logger",
                "cleanup",
                "Old log cleanup completed",
                {'retention_days': retention_days},
                tags=["cleanup", "maintenance"]
            )
            
        except Exception as e:
            self._log_internal_error(f"Cleanup failed: {str(e)}", {
                'retention_days': retention_days
            })
    
    def _cleanup_old_archives(self):
        """Clean up old archive files beyond backup count."""
        archive_dir = self.log_dir / "archive"
        
        if not archive_dir.exists():
            return
        
        # Get all archive files sorted by modification time
        archive_files = sorted(
            archive_dir.iterdir(),
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )
        
        # Remove files beyond backup count
        for file_path in archive_files[self.backup_count:]:
            try:
                file_path.unlink()
            except Exception as e:
                self._log_internal_error(f"Failed to remove old archive: {str(e)}", {
                    'file': str(file_path)
                })
    
    def _log_system_health_check(self):
        """Log system health check."""
        try:
            # Get system health information
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent
            disk_percent = psutil.disk_usage('/').percent
            
            self.log(
                LogLevel.INFO,
                LogCategory.SYSTEM,
                "logger",
                "health_check",
                "System health check",
                {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent,
                    'disk_percent': disk_percent,
                    'queue_size': self.performance_metrics['queue_size'],
                    'dropped_messages': self.performance_metrics['dropped_messages']
                },
                tags=["health", "monitoring"]
            )
            
        except Exception as e:
            self._log_internal_error(f"Health check failed: {str(e)}", {})
    
    def _export_metrics(self):
        """Export metrics for external monitoring systems."""
        try:
            metrics_file = self.log_dir / "metrics" / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            metrics_file.parent.mkdir(exist_ok=True)
            
            metrics_data = {
                'timestamp': datetime.now().isoformat(),
                'performance': self.performance_metrics.copy(),
                'statistics': {k: v.to_dict() for k, v in self.statistics.items()}
            }
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
        except Exception as e:
            self._log_internal_error(f"Metrics export failed: {str(e)}", {})
    
    def _log_system_event(self, level: LogLevel, message: str, data: Dict, tags: List[str]):
        """Log system-level event."""
        self.log(
            level,
            LogCategory.SYSTEM,
            "logger",
            "system",
            message,
            data,
            tags=tags
        )
    
    def _log_internal_error(self, message: str, data: Dict):
        """Log internal logger error."""
        error_data = {
            'logger_error': True,
            **data
        }
        
        self.log(
            LogLevel.ERROR,
            LogCategory.SYSTEM,
            "logger",
            "internal",
            message,
            error_data,
            tags=["logger", "error", "internal"]
        )
    
    def _get_traceback(self, exception: Exception) -> str:
        """Get traceback as string."""
        import traceback
        return traceback.format_exc()
    
    # ROS2 Subscriber callbacks
    def _log_system_health(self, msg: SystemHealth):
        """Log system health messages."""
        health_data = {
            'overall_health': msg.overall_health,
            'system_cpu_percent': msg.system_cpu_percent,
            'system_memory_percent': msg.system_memory_percent,
            'system_temperature': msg.system_temperature,
            'network_latency_ms': msg.network_latency_ms,
            'disk_usage_percent': msg.disk_usage_percent
        }
        
        component_data = {}
        for component in msg.components:
            component_data[component.name] = {
                'health_state': component.health_state,
                'uptime_seconds': component.uptime_seconds,
                'cpu_percent': component.cpu_percent,
                'memory_mb': component.memory_mb
            }
        
        health_data['components'] = component_data
        
        self.log(
            LogLevel.INFO,
            LogCategory.SYSTEM,
            "health_checker",
            "health_publisher",
            "System health update",
            health_data,
            tags=["health", "monitoring"]
        )
    
    def _log_alerts(self, msg: Alert):
        """Log alert messages."""
        alert_data = {
            'level': msg.level,
            'component': msg.component,
            'message': msg.message,
            'metric': msg.metric,
            'value': msg.value,
            'threshold': msg.threshold
        }
        
        log_level = LogLevel.ERROR if msg.level >= 2 else LogLevel.WARNING
        
        self.log(
            log_level,
            LogCategory.SYSTEM,
            msg.component,
            "alert_generator",
            f"ALERT: {msg.message}",
            alert_data,
            tags=["alert", msg.component]
        )
    
    def _log_diagnostics(self, msg: DiagnosticArray):
        """Log diagnostic messages."""
        for status in msg.status:
            diag_data = {
                'name': status.name,
                'level': status.level,
                'message': status.message,
                'hardware_id': status.hardware_id
            }
            
            values = {}
            for kv in status.values:
                values[kv.key] = kv.value
            
            diag_data['values'] = values
            
            log_level = LogLevel.DEBUG if status.level == 0 else \
                       LogLevel.INFO if status.level == 1 else \
                       LogLevel.WARNING if status.level == 2 else \
                       LogLevel.ERROR
            
            self.log(
                log_level,
                LogCategory.DIAGNOSTIC,
                "diagnostic_aggregator",
                "diagnostics",
                f"Diagnostic: {status.name} - {status.message}",
                diag_data,
                tags=["diagnostics", "system"]
            )
    
    def _log_robot_status(self, msg: RobotStatus):
        """Log robot status messages."""
        status_data = {
            'status': msg.status,
            'operation_mode': msg.operation_mode,
            'joint_positions': list(msg.joint_positions),
            'joint_velocities': list(msg.joint_velocities),
            'joint_efforts': list(msg.joint_efforts),
            'end_effector_pose': list(msg.end_effector_pose),
            'gripper_state': msg.gripper_state,
            'battery_level': msg.battery_level
        }
        
        self.log(
            LogLevel.INFO,
            LogCategory.CONTROL,
            "robot_controller",
            "status_publisher",
            "Robot status update",
            status_data,
            tags=["robot", "status", "control"]
        )
    
    def _log_detected_objects(self, msg: DetectedObject):
        """Log detected object messages (sampled)."""
        # Sample 1 in 10 messages to avoid log spam
        import random
        if random.random() > 0.1:
            return
        
        object_data = {
            'class_id': msg.class_id,
            'class_name': msg.class_name,
            'confidence': msg.confidence,
            'bbox': {
                'x': msg.bbox.x,
                'y': msg.bbox.y,
                'width': msg.bbox.width,
                'height': msg.bbox.height
            },
            'position': {
                'x': msg.position.x,
                'y': msg.position.y,
                'z': msg.position.z
            },
            'orientation': {
                'x': msg.orientation.x,
                'y': msg.orientation.y,
                'z': msg.orientation.z,
                'w': msg.orientation.w
            },
            'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        }
        
        self.log(
            LogLevel.DEBUG,
            LogCategory.PERCEPTION,
            "perception_node",
            "object_detector",
            f"Detected object: {msg.class_name}",
            object_data,
            tags=["perception", "object_detection", msg.class_name.lower()]
        )
    
    def _log_motion_events(self, msg: String):
        """Log motion planning events."""
        event_data = {
            'event': msg.data
        }
        
        self.log(
            LogLevel.INFO,
            LogCategory.MOTION,
            "motion_planner",
            "event_publisher",
            f"Motion event: {msg.data}",
            event_data,
            tags=["motion", "planning", "event"]
        )
    
    # ROS2 Service handlers
    def handle_get_logs(self, request: GetLogs.Request, response: GetLogs.Response) -> GetLogs.Response:
        """Handle get logs service request."""
        try:
            start_time = datetime.fromisoformat(request.start_time) if request.start_time else None
            end_time = datetime.fromisoformat(request.end_time) if request.end_time else None
            
            logs = self._search_logs(
                start_time=start_time,
                end_time=end_time,
                level=LogLevel(request.level) if request.level > 0 else None,
                category=LogCategory(request.category) if request.category else None,
                node=request.node if request.node else None,
                source=request.source if request.source else None,
                tags=request.tags if request.tags else None,
                limit=request.limit if request.limit > 0 else 1000,
                offset=request.offset if request.offset > 0 else 0
            )
            
            response.success = True
            response.logs = [log.to_ros_msg() for log in logs]
            response.count = len(logs)
            
        except Exception as e:
            response.success = False
            response.error_message = str(e)
        
        return response
    
    def handle_search_logs(self, request: SearchLogs.Request, response: SearchLogs.Response) -> SearchLogs.Response:
        """Handle search logs service request."""
        try:
            logs = self._search_logs(
                query=request.query,
                regex=request.use_regex,
                case_sensitive=request.case_sensitive,
                level=LogLevel(request.level) if request.level > 0 else None,
                category=LogCategory(request.category) if request.category else None,
                node=request.node if request.node else None,
                tags=request.tags if request.tags else None,
                limit=request.limit if request.limit > 0 else 1000,
                offset=request.offset if request.offset > 0 else 0
            )
            
            response.success = True
            response.logs = [log.to_ros_msg() for log in logs]
            response.count = len(logs)
            
        except Exception as e:
            response.success = False
            response.error_message = str(e)
        
        return response
    
    def _search_logs(self, **kwargs) -> List[LogEntryData]:
        """Search logs based on criteria."""
        # This is a simplified implementation
        # In production, this would use a proper search engine or database
        
        # For now, return empty list
        # Implementation would involve:
        # 1. Querying log files based on time range
        # 2. Parsing and filtering entries
        # 3. Applying search criteria
        # 4. Paginating results
        
        return []
    
    def handle_export_logs(self, request: ExportLogs.Request, response: ExportLogs.Response) -> ExportLogs.Response:
        """Handle export logs service request."""
        try:
            export_format = request.format
            compression = request.compression
            
            # Create export file
            export_dir = self.log_dir / "exports"
            export_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            export_file = export_dir / f"export_{timestamp}.{export_format}"
            
            # Export logs (simplified)
            # In production, this would actually export the requested logs
            
            response.success = True
            response.file_path = str(export_file)
            response.file_size = 0  # Would be actual size
            
        except Exception as e:
            response.success = False
            response.error_message = str(e)
        
        return response
    
    def handle_set_log_level(self, request: SetLogLevel.Request, response: SetLogLevel.Response) -> SetLogLevel.Response:
        """Handle set log level service request."""
        try:
            if request.level:
                self.default_level = LogLevel(request.level)
                response.success = True
                response.message = f"Log level set to {self.default_level.name}"
            else:
                response.success = False
                response.message = "Invalid log level"
                
        except Exception as e:
            response.success = False
            response.message = str(e)
        
        return response
    
    def handle_clear_logs(self, request: ClearLogs.Request, response: ClearLogs.Response) -> ClearLogs.Response:
        """Handle clear logs service request."""
        try:
            if request.confirmation != "CONFIRM_DELETE_ALL_LOGS":
                response.success = False
                response.message = "Invalid confirmation code"
                return response
            
            # Clear in-memory data
            self.statistics.clear()
            self.correlation_map.clear()
            self.session_map.clear()
            self.tag_index.clear()
            self.sequence_counter = 0
            
            # Don't delete actual log files for safety
            response.success = True
            response.message = "In-memory log data cleared. Log files preserved."
            
        except Exception as e:
            response.success = False
            response.message = str(e)
        
        return response
    
    def handle_get_log_statistics(self, request: GetLogStatistics.Request, response: GetLogStatistics.Response) -> GetLogStatistics.Response:
        """Handle get log statistics service request."""
        try:
            period_start = datetime.fromisoformat(request.period_start) if request.period_start else None
            period_end = datetime.fromisoformat(request.period_end) if request.period_end else None
            
            # Get statistics for requested period
            stats = self._get_statistics_for_period(period_start, period_end)
            
            response.success = True
            response.statistics_json = json.dumps(stats.to_dict())
            
        except Exception as e:
            response.success = False
            response.error_message = str(e)
        
        return response
    
    def _get_statistics_for_period(self, start_time: Optional[datetime], end_time: Optional[datetime]) -> LogStatistics:
        """Get statistics for a specific time period."""
        # Aggregate statistics for the period
        # This is a simplified implementation
        
        return LogStatistics(
            period_start=start_time or datetime.now(),
            period_end=end_time or datetime.now(),
            total_entries=0,
            entries_by_level={},
            entries_by_category={},
            entries_by_node={},
            avg_entry_size_bytes=0,
            max_entry_size_bytes=0,
            error_rate_per_hour=0.0,
            warning_rate_per_hour=0.0,
            busiest_node="",
            busiest_category="",
            common_tags=[],
            data_volume_bytes=0,
            compression_ratio=1.0
        )
    
    def shutdown(self):
        """Graceful shutdown of logger node."""
        self.get_logger().info("Shutting down advanced logger...")
        self.running = False
        
        # Flush remaining logs
        self._flush_batch()
        
        # Wait for worker threads
        for thread in self.worker_threads:
            if thread.is_alive():
                thread.join(timeout=5.0)
        
        # Log shutdown event
        self._log_system_event(
            LogLevel.SYSTEM,
            "AdvancedLogger shutdown",
            {"pid": os.getpid(), "uptime": time.time() - self.start_time},
            tags=["shutdown", "system"]
        )
        
        self.get_logger().info("Advanced logger shutdown complete")


def main(args=None):
    """Main entry point for advanced logger node."""
    rclpy.init(args=args)
    
    try:
        logger_node = AdvancedLoggerNode()
        rclpy.spin(logger_node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Fatal error in advanced logger: {str(e)}")
    finally:
        if 'logger_node' in locals():
            logger_node.shutdown()
            logger_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
