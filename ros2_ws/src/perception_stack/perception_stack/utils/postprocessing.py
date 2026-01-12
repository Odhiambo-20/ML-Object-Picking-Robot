"""
Post-processing utilities for YOLO object detection in industrial robotics.

This module handles:
- Non-Maximum Suppression (NMS)
- Coordinate transformations (image -> robot frame)
- Bounding box filtering and validation
- Obstacle classification and prioritization
- Safety zone checks

Author: Victor's Production Team
License: Proprietary - Industrial Use
"""

import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ObstacleType(Enum):
    """Classification of detected obstacles."""
    STATIC = "static"           # Stationary obstacles
    DYNAMIC = "dynamic"         # Moving obstacles (humans, vehicles)
    PICKABLE = "pickable"       # Objects to be picked
    FORBIDDEN = "forbidden"     # No-go zones
    UNKNOWN = "unknown"         # Unclassified


@dataclass
class DetectionResult:
    """Structured detection result with robot-centric data."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in image coords
    center_pixel: Tuple[int, int]    # (cx, cy) in image coords
    robot_coords: Optional[Tuple[float, float, float]] = None  # (x, y, z) in meters
    obstacle_type: ObstacleType = ObstacleType.UNKNOWN
    is_safe: bool = True
    depth: Optional[float] = None    # Distance in meters
    tracking_id: Optional[int] = None


class PostProcessor:
    """
    Production-grade post-processing for YOLO detections.
    
    Handles coordinate transformations, filtering, and safety checks
    for industrial robotic pick-and-place applications.
    """
    
    def __init__(
        self,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 100,
        img_width: int = 640,
        img_height: int = 480,
        camera_matrix: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
        safety_zone_radius: float = 0.5,  # meters
        dynamic_classes: List[str] = None,
        pickable_classes: List[str] = None
    ):
        """
        Initialize post-processor with production parameters.
        
        Args:
            conf_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            max_detections: Maximum number of detections to process
            img_width: Input image width
            img_height: Input image height
            camera_matrix: 3x3 camera intrinsic matrix
            dist_coeffs: Distortion coefficients
            safety_zone_radius: Safety zone around robot (meters)
            dynamic_classes: List of class names considered dynamic
            pickable_classes: List of class names that can be picked
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.img_width = img_width
        self.img_height = img_height
        
        # Camera calibration
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        
        # Safety parameters
        self.safety_zone_radius = safety_zone_radius
        
        # Object classification
        self.dynamic_classes = set(dynamic_classes or [
            'person', 'car', 'truck', 'bus', 'bicycle', 'motorcycle',
            'dog', 'cat', 'bird', 'horse'
        ])
        
        self.pickable_classes = set(pickable_classes or [
            'bottle', 'cup', 'book', 'cell phone', 'mouse', 'keyboard',
            'scissors', 'teddy bear', 'vase', 'bowl'
        ])
        
        # COCO class names (80 classes)
        self.coco_classes = self._load_coco_classes()
        
        logger.info(f"PostProcessor initialized: conf={conf_threshold}, iou={iou_threshold}")
    
    def _load_coco_classes(self) -> List[str]:
        """Load COCO dataset class names."""
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
    
    def process_detections(
        self,
        predictions: np.ndarray,
        original_shape: Tuple[int, int],
        depth_map: Optional[np.ndarray] = None
    ) -> List[DetectionResult]:
        """
        Process raw YOLO predictions into structured detection results.
        
        Args:
            predictions: Raw YOLO output [N, 6] (x1, y1, x2, y2, conf, class_id)
            original_shape: Original image shape (height, width)
            depth_map: Optional depth map for 3D positioning
            
        Returns:
            List of DetectionResult objects
        """
        if predictions is None or len(predictions) == 0:
            logger.debug("No detections to process")
            return []
        
        # Apply confidence threshold
        predictions = self._filter_by_confidence(predictions)
        
        # Apply Non-Maximum Suppression
        predictions = self._apply_nms(predictions)
        
        # Scale bounding boxes to original image size
        predictions = self._scale_boxes(predictions, original_shape)
        
        # Convert to DetectionResult objects
        detections = self._create_detection_results(predictions, depth_map)
        
        # Classify obstacle types
        detections = self._classify_obstacles(detections)
        
        # Perform safety checks
        detections = self._safety_check(detections)
        
        # Sort by priority (dynamic > pickable > static)
        detections = self._prioritize_detections(detections)
        
        logger.info(f"Processed {len(detections)} valid detections")
        return detections[:self.max_detections]
    
    def _filter_by_confidence(self, predictions: np.ndarray) -> np.ndarray:
        """Filter detections below confidence threshold."""
        mask = predictions[:, 4] >= self.conf_threshold
        return predictions[mask]
    
    def _apply_nms(self, predictions: np.ndarray) -> np.ndarray:
        """
        Apply Non-Maximum Suppression using OpenCV.
        
        Args:
            predictions: [N, 6] (x1, y1, x2, y2, conf, class_id)
            
        Returns:
            Filtered predictions after NMS
        """
        if len(predictions) == 0:
            return predictions
        
        boxes = predictions[:, :4]
        scores = predictions[:, 4]
        
        # OpenCV NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            self.conf_threshold,
            self.iou_threshold
        )
        
        if len(indices) > 0:
            indices = indices.flatten()
            return predictions[indices]
        
        return np.array([])
    
    def _scale_boxes(
        self,
        predictions: np.ndarray,
        original_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Scale bounding boxes from model input size to original image size.
        
        Args:
            predictions: Detection array
            original_shape: (height, width) of original image
            
        Returns:
            Scaled predictions
        """
        if len(predictions) == 0:
            return predictions
        
        orig_h, orig_w = original_shape
        scale_y = orig_h / self.img_height
        scale_x = orig_w / self.img_width
        
        predictions[:, [0, 2]] *= scale_x  # x coordinates
        predictions[:, [1, 3]] *= scale_y  # y coordinates
        
        return predictions
    
    def _create_detection_results(
        self,
        predictions: np.ndarray,
        depth_map: Optional[np.ndarray] = None
    ) -> List[DetectionResult]:
        """Convert raw predictions to DetectionResult objects."""
        results = []
        
        for pred in predictions:
            x1, y1, x2, y2, conf, class_id = pred
            class_id = int(class_id)
            
            # Ensure valid class ID
            if class_id >= len(self.coco_classes):
                logger.warning(f"Invalid class_id: {class_id}")
                continue
            
            class_name = self.coco_classes[class_id]
            
            # Calculate center
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            # Get depth if available
            depth = None
            if depth_map is not None:
                depth = self._get_depth_at_point(depth_map, cx, cy)
            
            # Create detection result
            detection = DetectionResult(
                class_id=class_id,
                class_name=class_name,
                confidence=float(conf),
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                center_pixel=(cx, cy),
                depth=depth
            )
            
            results.append(detection)
        
        return results
    
    def _get_depth_at_point(
        self,
        depth_map: np.ndarray,
        x: int,
        y: int,
        window_size: int = 5
    ) -> Optional[float]:
        """
        Get median depth around a point (more robust than single pixel).
        
        Args:
            depth_map: Depth image
            x, y: Center coordinates
            window_size: Size of sampling window
            
        Returns:
            Median depth in meters or None
        """
        h, w = depth_map.shape
        half_win = window_size // 2
        
        y1 = max(0, y - half_win)
        y2 = min(h, y + half_win + 1)
        x1 = max(0, x - half_win)
        x2 = min(w, x + half_win + 1)
        
        depth_window = depth_map[y1:y2, x1:x2]
        
        # Filter out invalid depths (0 or NaN)
        valid_depths = depth_window[(depth_window > 0) & np.isfinite(depth_window)]
        
        if len(valid_depths) > 0:
            return float(np.median(valid_depths))
        
        return None
    
    def _classify_obstacles(
        self,
        detections: List[DetectionResult]
    ) -> List[DetectionResult]:
        """Classify obstacles based on object class."""
        for detection in detections:
            class_name = detection.class_name.lower()
            
            if class_name in self.dynamic_classes:
                detection.obstacle_type = ObstacleType.DYNAMIC
            elif class_name in self.pickable_classes:
                detection.obstacle_type = ObstacleType.PICKABLE
            else:
                detection.obstacle_type = ObstacleType.STATIC
        
        return detections
    
    def _safety_check(
        self,
        detections: List[DetectionResult]
    ) -> List[DetectionResult]:
        """
        Perform safety checks on detections.
        Mark unsafe if too close or dynamic obstacle detected.
        """
        for detection in detections:
            # Check if dynamic obstacle
            if detection.obstacle_type == ObstacleType.DYNAMIC:
                detection.is_safe = False
                logger.warning(f"Dynamic obstacle detected: {detection.class_name}")
            
            # Check if within safety zone
            if detection.depth is not None:
                if detection.depth < self.safety_zone_radius:
                    detection.is_safe = False
                    logger.warning(
                        f"Object within safety zone: {detection.class_name} "
                        f"at {detection.depth:.2f}m"
                    )
        
        return detections
    
    def _prioritize_detections(
        self,
        detections: List[DetectionResult]
    ) -> List[DetectionResult]:
        """
        Sort detections by priority:
        1. Dynamic obstacles (highest priority - safety)
        2. Pickable objects
        3. Static obstacles
        """
        priority_map = {
            ObstacleType.DYNAMIC: 0,
            ObstacleType.PICKABLE: 1,
            ObstacleType.STATIC: 2,
            ObstacleType.FORBIDDEN: 0,
            ObstacleType.UNKNOWN: 3
        }
        
        return sorted(
            detections,
            key=lambda d: (priority_map[d.obstacle_type], -d.confidence)
        )
    
    def pixel_to_robot_coords(
        self,
        detection: DetectionResult,
        camera_to_robot_transform: np.ndarray
    ) -> DetectionResult:
        """
        Transform pixel coordinates to robot frame coordinates.
        
        Args:
            detection: Detection with pixel coordinates
            camera_to_robot_transform: 4x4 transformation matrix
            
        Returns:
            Detection with updated robot_coords
        """
        if self.camera_matrix is None or detection.depth is None:
            logger.warning("Cannot transform: missing camera matrix or depth")
            return detection
        
        cx, cy = detection.center_pixel
        depth = detection.depth
        
        # Pixel to camera frame (using intrinsic matrix)
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx_cam = self.camera_matrix[0, 2]
        cy_cam = self.camera_matrix[1, 2]
        
        # 3D point in camera frame
        x_cam = (cx - cx_cam) * depth / fx
        y_cam = (cy - cy_cam) * depth / fy
        z_cam = depth
        
        # Transform to robot frame
        point_cam = np.array([x_cam, y_cam, z_cam, 1.0])
        point_robot = camera_to_robot_transform @ point_cam
        
        detection.robot_coords = (
            float(point_robot[0]),
            float(point_robot[1]),
            float(point_robot[2])
        )
        
        return detection
    
    def filter_by_roi(
        self,
        detections: List[DetectionResult],
        roi: Tuple[int, int, int, int]
    ) -> List[DetectionResult]:
        """
        Filter detections within a Region of Interest.
        
        Args:
            detections: List of detections
            roi: (x, y, width, height) of ROI
            
        Returns:
            Filtered detections
        """
        x_roi, y_roi, w_roi, h_roi = roi
        filtered = []
        
        for det in detections:
            cx, cy = det.center_pixel
            
            if (x_roi <= cx <= x_roi + w_roi and
                y_roi <= cy <= y_roi + h_roi):
                filtered.append(det)
        
        return filtered
    
    def visualize_detections(
        self,
        image: np.ndarray,
        detections: List[DetectionResult],
        show_confidence: bool = True,
        show_depth: bool = True
    ) -> np.ndarray:
        """
        Visualize detections on image for debugging/monitoring.
        
        Args:
            image: Input image (BGR)
            detections: List of detections
            show_confidence: Whether to show confidence scores
            show_depth: Whether to show depth information
            
        Returns:
            Annotated image
        """
        vis_img = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            # Color based on obstacle type
            color_map = {
                ObstacleType.DYNAMIC: (0, 0, 255),      # Red - danger
                ObstacleType.PICKABLE: (0, 255, 0),     # Green - target
                ObstacleType.STATIC: (255, 165, 0),     # Orange - caution
                ObstacleType.FORBIDDEN: (128, 0, 128),  # Purple
                ObstacleType.UNKNOWN: (128, 128, 128)   # Gray
            }
            
            color = color_map.get(det.obstacle_type, (255, 255, 255))
            
            # Draw bounding box (thicker if unsafe)
            thickness = 3 if not det.is_safe else 2
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, thickness)
            
            # Prepare label
            label_parts = [det.class_name]
            if show_confidence:
                label_parts.append(f"{det.confidence:.2f}")
            if show_depth and det.depth is not None:
                label_parts.append(f"{det.depth:.2f}m")
            
            label = " | ".join(label_parts)
            
            # Draw label background
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                vis_img,
                (x1, y1 - label_h - baseline - 5),
                (x1 + label_w, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                vis_img,
                label,
                (x1, y1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            
            # Draw center point
            cx, cy = det.center_pixel
            cv2.circle(vis_img, (cx, cy), 5, color, -1)
        
        return vis_img
    
    def export_to_dict(self, detections: List[DetectionResult]) -> List[Dict]:
        """Export detections to dictionary format for ROS messages."""
        return [
            {
                'class_id': det.class_id,
                'class_name': det.class_name,
                'confidence': det.confidence,
                'bbox': det.bbox,
                'center_pixel': det.center_pixel,
                'robot_coords': det.robot_coords,
                'obstacle_type': det.obstacle_type.value,
                'is_safe': det.is_safe,
                'depth': det.depth,
                'tracking_id': det.tracking_id
            }
            for det in detections
        ]


# Utility functions for production use

def create_default_postprocessor(
    production_mode: bool = True
) -> PostProcessor:
    """
    Factory function for creating post-processor with sensible defaults.
    
    Args:
        production_mode: If True, use conservative thresholds for safety
        
    Returns:
        Configured PostProcessor instance
    """
    if production_mode:
        return PostProcessor(
            conf_threshold=0.5,      # Higher threshold for production
            iou_threshold=0.45,
            max_detections=50,
            safety_zone_radius=0.5
        )
    else:
        return PostProcessor(
            conf_threshold=0.25,     # Lower for testing
            iou_threshold=0.45,
            max_detections=100,
            safety_zone_radius=0.3
        )
