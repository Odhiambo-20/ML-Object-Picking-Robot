#!/usr/bin/env python3
"""
Advanced visualization utilities for object detection and perception results.
Production-grade visualization with multiple output formats, real-time overlays,
and comprehensive debugging capabilities.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
import logging
from enum import Enum
import time
from collections import defaultdict
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle, Circle, Polygon
from matplotlib.lines import Line2D
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import json

# Configure logging
logger = logging.getLogger(__name__)

class VisualizationMode(Enum):
    """Visualization output modes."""
    OPENCV = "opencv"          # OpenCV-based visualization
    MATPLOTLIB = "matplotlib"  # Matplotlib-based visualization
    PLOTLY = "plotly"          # Plotly interactive visualization
    PILLOW = "pillow"          # PIL/Pillow-based visualization
    ROS = "ros"               # ROS-compatible visualization
    WEB = "web"               # Web-based interactive visualization

class ColorScheme(Enum):
    """Color schemes for visualization."""
    DEFAULT = "default"        # Default color mapping
    JET = "jet"               # Jet colormap
    VIRIDIS = "viridis"       # Viridis colormap
    PLASMA = "plasma"         # Plasma colormap
    CATEGORICAL = "categorical"  # Categorical colors
    MONOCHROME = "monochrome"  # Monochrome visualization

@dataclass
class BoundingBox:
    """Bounding box with visualization attributes."""
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    confidence: float = 1.0
    class_id: int = 0
    class_name: str = "object"
    track_id: Optional[int] = None
    attributes: Dict[str, Any] = None
    
    @property
    def width(self) -> float:
        return self.xmax - self.xmin
    
    @property
    def height(self) -> float:
        return self.ymax - self.ymin
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.xmin + self.xmax) / 2, (self.ymin + self.ymax) / 2)
    
    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height > 0 else 0.0

@dataclass
class VisualizationConfig:
    """Configuration for visualization."""
    # Output settings
    mode: VisualizationMode = VisualizationMode.OPENCV
    output_format: str = "png"
    output_dpi: int = 300
    output_size: Tuple[int, int] = (1920, 1080)
    
    # Color settings
    color_scheme: ColorScheme = ColorScheme.DEFAULT
    opacity: float = 0.7
    line_thickness: int = 2
    font_size: int = 12
    font_face: str = "Arial"
    
    # Content settings
    show_confidence: bool = True
    show_class_names: bool = True
    show_track_ids: bool = True
    show_timestamps: bool = True
    show_fps: bool = True
    show_legend: bool = True
    show_grid: bool = False
    show_axes: bool = False
    
    # Advanced features
    highlight_top_k: int = 5
    use_gradients: bool = False
    use_shadows: bool = True
    use_rounded_corners: bool = False
    animate_transitions: bool = False
    
    # Performance
    cache_size: int = 100
    enable_caching: bool = True
    parallel_rendering: bool = False
    
    # Debug
    debug_mode: bool = False
    save_debug_frames: bool = False
    debug_output_path: str = "/tmp/visualization_debug"

class Visualizer:
    """
    Advanced visualizer for perception results.
    Supports multiple visualization modes, real-time overlays,
    and comprehensive debugging capabilities.
    """
    
    def __init__(self, config: VisualizationConfig = None):
        """
        Initialize visualizer with configuration.
        
        Args:
            config: Visualization configuration
        """
        self.config = config or VisualizationConfig()
        self.setup_logging()
        self.setup_color_palette()
        self.setup_fonts()
        self.setup_cache()
        
        # Performance tracking
        self.frame_count = 0
        self.rendering_times = []
        self.last_timestamp = time.time()
        
        # State
        self.current_frame = None
        self.current_annotations = []
        self.history = defaultdict(list)
        
        logger.info("Visualizer initialized with config: %s", self.config)
    
    def setup_logging(self):
        """Configure logging for visualizer."""
        log_level = logging.DEBUG if self.config.debug_mode else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def setup_color_palette(self):
        """Setup color palette based on configuration."""
        if self.config.color_scheme == ColorScheme.DEFAULT:
            self.color_palette = self._create_default_palette()
        elif self.config.color_scheme == ColorScheme.JET:
            self.color_palette = self._create_jet_palette()
        elif self.config.color_scheme == ColorScheme.VIRIDIS:
            self.color_palette = self._create_viridis_palette()
        elif self.config.color_scheme == ColorScheme.PLASMA:
            self.color_palette = self._create_plasma_palette()
        elif self.config.color_scheme == ColorScheme.CATEGORICAL:
            self.color_palette = self._create_categorical_palette()
        elif self.config.color_scheme == ColorScheme.MONOCHROME:
            self.color_palette = self._create_monochrome_palette()
        else:
            self.color_palette = self._create_default_palette()
    
    def _create_default_palette(self) -> Dict[int, Tuple[int, int, int]]:
        """Create default color palette."""
        colors = [
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red
            (255, 0, 0),    # Blue
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (255, 255, 0),  # Cyan
            (128, 0, 128),  # Purple
            (0, 128, 128),  # Teal
            (128, 128, 0),  # Olive
            (128, 0, 0),    # Maroon
            (0, 128, 0),    # Dark Green
            (0, 0, 128),    # Navy
        ]
        return {i: colors[i % len(colors)] for i in range(100)}
    
    def _create_jet_palette(self) -> Dict[int, Tuple[int, int, int]]:
        """Create jet colormap palette."""
        cmap = cm.get_cmap('jet')
        return {i: tuple(int(c * 255) for c in cmap(i / 100)[:3]) for i in range(100)}
    
    def _create_viridis_palette(self) -> Dict[int, Tuple[int, int, int]]:
        """Create viridis colormap palette."""
        cmap = cm.get_cmap('viridis')
        return {i: tuple(int(c * 255) for c in cmap(i / 100)[:3]) for i in range(100)}
    
    def _create_plasma_palette(self) -> Dict[int, Tuple[int, int, int]]:
        """Create plasma colormap palette."""
        cmap = cm.get_cmap('plasma')
        return {i: tuple(int(c * 255) for c in cmap(i / 100)[:3]) for i in range(100)}
    
    def _create_categorical_palette(self) -> Dict[int, Tuple[int, int, int]]:
        """Create categorical color palette."""
        colors = sns.color_palette("tab20", 20)
        return {i: tuple(int(c * 255) for c in colors[i % 20]) for i in range(100)}
    
    def _create_monochrome_palette(self) -> Dict[int, Tuple[int, int, int]]:
        """Create monochrome palette."""
        return {i: (i, i, i) for i in range(256)}
    
    def setup_fonts(self):
        """Setup fonts for text rendering."""
        self.fonts = {}
        
        if self.config.mode == VisualizationMode.OPENCV:
            # OpenCV fonts
            self.fonts['small'] = cv2.FONT_HERSHEY_SIMPLEX
            self.fonts['medium'] = cv2.FONT_HERSHEY_DUPLEX
            self.fonts['large'] = cv2.FONT_HERSHEY_COMPLEX
        
        elif self.config.mode == VisualizationMode.PILLOW:
            # PIL fonts
            try:
                self.fonts['small'] = ImageFont.truetype(self.config.font_face, self.config.font_size)
                self.fonts['medium'] = ImageFont.truetype(self.config.font_face, self.config.font_size * 2)
                self.fonts['large'] = ImageFont.truetype(self.config.font_face, self.config.font_size * 3)
            except IOError:
                logger.warning("Font %s not found, using default", self.config.font_face)
                self.fonts['small'] = ImageFont.load_default()
                self.fonts['medium'] = ImageFont.load_default()
                self.fonts['large'] = ImageFont.load_default()
    
    def setup_cache(self):
        """Setup caching system for improved performance."""
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def visualize(
        self,
        image: np.ndarray,
        detections: List[BoundingBox],
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Visualize detections on image.
        
        Args:
            image: Input image
            detections: List of bounding boxes
            timestamp: Current timestamp
            metadata: Additional metadata
            
        Returns:
            Visualized result (format depends on mode)
        """
        start_time = time.time()
        
        try:
            # Validate input
            if image is None or image.size == 0:
                logger.error("Empty image provided for visualization")
                return image
            
            # Update state
            self.current_frame = image.copy()
            self.current_annotations = detections.copy()
            
            # Check cache
            cache_key = self._generate_cache_key(image, detections, timestamp)
            if self.config.enable_caching and cache_key in self.cache:
                self.cache_hits += 1
                result = self.cache[cache_key]
                
                # Update cache LRU
                del self.cache[cache_key]
                self.cache[cache_key] = result
                
                if self.config.debug_mode:
                    logger.debug("Cache hit for key: %s", cache_key[:20])
                
                return result
            
            self.cache_misses += 1
            
            # Apply visualization based on mode
            if self.config.mode == VisualizationMode.OPENCV:
                result = self._visualize_opencv(image, detections, timestamp, metadata)
            elif self.config.mode == VisualizationMode.MATPLOTLIB:
                result = self._visualize_matplotlib(image, detections, timestamp, metadata)
            elif self.config.mode == VisualizationMode.PLOTLY:
                result = self._visualize_plotly(image, detections, timestamp, metadata)
            elif self.config.mode == VisualizationMode.PILLOW:
                result = self._visualize_pillow(image, detections, timestamp, metadata)
            elif self.config.mode == VisualizationMode.ROS:
                result = self._visualize_ros(image, detections, timestamp, metadata)
            elif self.config.mode == VisualizationMode.WEB:
                result = self._visualize_web(image, detections, timestamp, metadata)
            else:
                logger.warning("Unknown visualization mode: %s, using OpenCV", self.config.mode)
                result = self._visualize_opencv(image, detections, timestamp, metadata)
            
            # Update cache
            if self.config.enable_caching:
                if len(self.cache) >= self.config.cache_size:
                    # Remove oldest entry
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                
                self.cache[cache_key] = result
            
            # Update performance metrics
            rendering_time = time.time() - start_time
            self.rendering_times.append(rendering_time)
            self.frame_count += 1
            
            # Update FPS
            current_time = time.time()
            if current_time - self.last_timestamp > 1.0:
                fps = len([t for t in self.rendering_times 
                          if t > current_time - 1.0])
                if self.config.debug_mode:
                    logger.debug("Current FPS: %.2f", fps)
                self.last_timestamp = current_time
            
            # Save debug frame if enabled
            if self.config.save_debug_frames:
                self._save_debug_frame(result, timestamp)
            
            return result
            
        except Exception as e:
            logger.error("Visualization failed: %s", str(e))
            return image
    
    def _visualize_opencv(
        self,
        image: np.ndarray,
        detections: List[BoundingBox],
        timestamp: Optional[float],
        metadata: Optional[Dict[str, Any]]
    ) -> np.ndarray:
        """Visualize using OpenCV."""
        # Create copy of image
        visualized = image.copy()
        
        if len(visualized.shape) == 2:
            visualized = cv2.cvtColor(visualized, cv2.COLOR_GRAY2BGR)
        
        # Sort detections by confidence for highlighting
        sorted_detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
        
        # Draw detections
        for i, detection in enumerate(sorted_detections):
            # Determine if this is a top detection
            is_top = i < self.config.highlight_top_k
            
            # Get color for this detection
            color = self._get_detection_color(detection, is_top)
            
            # Draw bounding box
            visualized = self._draw_opencv_bbox(visualized, detection, color, is_top)
            
            # Draw label
            if self.config.show_class_names or self.config.show_confidence or self.config.show_track_ids:
                visualized = self._draw_opencv_label(visualized, detection, color)
        
        # Draw metadata overlays
        visualized = self._draw_opencv_overlays(visualized, timestamp, metadata, len(detections))
        
        # Apply post-processing effects
        if self.config.use_gradients:
            visualized = self._apply_gradient_overlay(visualized)
        
        if self.config.use_shadows:
            visualized = self._apply_shadow_effect(visualized)
        
        return visualized
    
    def _visualize_matplotlib(
        self,
        image: np.ndarray,
        detections: List[BoundingBox],
        timestamp: Optional[float],
        metadata: Optional[Dict[str, Any]]
    ) -> plt.Figure:
        """Visualize using Matplotlib."""
        # Create figure
        fig, ax = plt.subplots(figsize=(self.config.output_size[0]/100, 
                                       self.config.output_size[1]/100),
                              dpi=self.config.output_dpi)
        
        # Display image
        if len(image.shape) == 2:
            ax.imshow(image, cmap='gray')
        else:
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Draw detections
        for i, detection in enumerate(detections):
            # Determine if this is a top detection
            is_top = i < self.config.highlight_top_k
            
            # Get color for this detection
            color = self._get_detection_color(detection, is_top, normalize=True)
            
            # Draw bounding box
            self._draw_matplotlib_bbox(ax, detection, color, is_top)
            
            # Draw label
            if self.config.show_class_names or self.config.show_confidence or self.config.show_track_ids:
                self._draw_matplotlib_label(ax, detection, color)
        
        # Configure axes
        if not self.config.show_axes:
            ax.axis('off')
        
        if self.config.show_grid:
            ax.grid(True, alpha=0.3)
        
        # Add title and metadata
        title_parts = []
        if timestamp is not None and self.config.show_timestamps:
            title_parts.append(f"Time: {timestamp:.3f}s")
        if len(detections) > 0 and self.config.show_fps:
            fps = self.get_current_fps()
            if fps > 0:
                title_parts.append(f"FPS: {fps:.1f}")
        
        if title_parts:
            ax.set_title(" | ".join(title_parts))
        
        # Add legend if requested
        if self.config.show_legend and detections:
            self._add_matplotlib_legend(ax, detections)
        
        plt.tight_layout()
        return fig
    
    def _visualize_plotly(
        self,
        image: np.ndarray,
        detections: List[BoundingBox],
        timestamp: Optional[float],
        metadata: Optional[Dict[str, Any]]
    ) -> go.Figure:
        """Visualize using Plotly for interactive display."""
        # Create figure
        fig = make_subplots(rows=1, cols=1)
        
        # Add image
        if len(image.shape) == 2:
            # Grayscale
            fig.add_trace(
                go.Heatmap(z=image, colorscale='gray', showscale=False),
                row=1, col=1
            )
        else:
            # Color image
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            fig.add_trace(
                go.Image(z=rgb_image),
                row=1, col=1
            )
        
        # Draw detections
        for i, detection in enumerate(detections):
            # Get color for this detection
            color = self._get_detection_color(detection, normalize=False)
            plotly_color = f'rgb({color[0]}, {color[1]}, {color[2]})'
            
            # Add bounding box
            fig.add_trace(
                go.Scatter(
                    x=[detection.xmin, detection.xmax, detection.xmax, detection.xmin, detection.xmin],
                    y=[detection.ymin, detection.ymin, detection.ymax, detection.ymax, detection.ymin],
                    mode='lines',
                    line=dict(color=plotly_color, width=self.config.line_thickness),
                    fill='none',
                    name=f"{detection.class_name} ({detection.confidence:.2f})",
                    hoverinfo='text',
                    hovertext=self._create_hover_text(detection)
                ),
                row=1, col=1
            )
        
        # Configure layout
        fig.update_layout(
            title=f"Detection Results - {len(detections)} objects" if detections else "No detections",
            showlegend=self.config.show_legend,
            xaxis=dict(showgrid=self.config.show_grid, visible=self.config.show_axes),
            yaxis=dict(showgrid=self.config.show_grid, visible=self.config.show_axes),
            height=self.config.output_size[1],
            width=self.config.output_size[0]
        )
        
        return fig
    
    def _visualize_pillow(
        self,
        image: np.ndarray,
        detections: List[BoundingBox],
        timestamp: Optional[float],
        metadata: Optional[Dict[str, Any]]
    ) -> Image.Image:
        """Visualize using PIL/Pillow."""
        # Convert to PIL Image
        if len(image.shape) == 2:
            pil_image = Image.fromarray(image, mode='L').convert('RGB')
        else:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Create drawing context
        draw = ImageDraw.Draw(pil_image, 'RGBA')
        
        # Draw detections
        for i, detection in enumerate(detections):
            # Determine if this is a top detection
            is_top = i < self.config.highlight_top_k
            
            # Get color for this detection
            color = self._get_detection_color(detection, is_top)
            rgba_color = (*color, int(self.config.opacity * 255))
            
            # Draw bounding box
            self._draw_pillow_bbox(draw, detection, rgba_color, is_top)
            
            # Draw label
            if self.config.show_class_names or self.config.show_confidence or self.config.show_track_ids:
                self._draw_pillow_label(draw, detection, color)
        
        # Draw metadata overlays
        if timestamp is not None or metadata is not None:
            self._draw_pillow_overlays(draw, pil_image.size, timestamp, metadata, len(detections))
        
        return pil_image
    
    def _visualize_ros(
        self,
        image: np.ndarray,
        detections: List[BoundingBox],
        timestamp: Optional[float],
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create ROS-compatible visualization output."""
        import rospy  # Import only if needed
        from sensor_msgs.msg import Image as ROSImage
        from visualization_msgs.msg import MarkerArray, Marker
        from geometry_msgs.msg import Point
        
        # Create ROS Image message
        ros_image = ROSImage()
        ros_image.height = image.shape[0]
        ros_image.width = image.shape[1]
        ros_image.encoding = 'bgr8' if len(image.shape) == 3 else 'mono8'
        ros_image.is_bigendian = False
        ros_image.step = image.strides[0] if hasattr(image, 'strides') else image.shape[1] * (3 if len(image.shape) == 3 else 1)
        ros_image.data = image.tobytes()
        
        # Create MarkerArray for detections
        marker_array = MarkerArray()
        
        for i, detection in enumerate(detections):
            marker = Marker()
            marker.header.stamp = rospy.Time.now() if timestamp is None else rospy.Time.from_sec(timestamp)
            marker.header.frame_id = "camera_frame"
            marker.ns = "detections"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            # Set position (center of bounding box)
            center = detection.center
            marker.pose.position.x = center[0]
            marker.pose.position.y = center[1]
            marker.pose.position.z = 0
            marker.pose.orientation.w = 1.0
            
            # Set scale (size of bounding box)
            marker.scale.x = detection.width
            marker.scale.y = detection.height
            marker.scale.z = 0.1  # Small thickness
            
            # Set color
            color = self._get_detection_color(detection)
            marker.color.r = color[2] / 255.0
            marker.color.g = color[1] / 255.0
            marker.color.b = color[0] / 255.0
            marker.color.a = self.config.opacity
            
            # Set text for labels
            marker.text = self._create_label_text(detection)
            
            marker_array.markers.append(marker)
        
        # Create metadata message
        result = {
            "image": ros_image,
            "markers": marker_array,
            "timestamp": timestamp or time.time(),
            "num_detections": len(detections),
            "metadata": metadata or {}
        }
        
        return result
    
    def _visualize_web(
        self,
        image: np.ndarray,
        detections: List[BoundingBox],
        timestamp: Optional[float],
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create web-compatible visualization output."""
        import base64
        import io
        
        # Convert image to base64
        if len(image.shape) == 2:
            pil_image = Image.fromarray(image, mode='L')
        else:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Create detection data
        detection_data = []
        for detection in detections:
            detection_data.append({
                "bbox": [detection.xmin, detection.ymin, detection.xmax, detection.ymax],
                "confidence": detection.confidence,
                "class_id": detection.class_id,
                "class_name": detection.class_name,
                "track_id": detection.track_id,
                "center": detection.center,
                "area": detection.area,
                "aspect_ratio": detection.aspect_ratio,
                "color": self._get_detection_color(detection)
            })
        
        # Create visualization data
        result = {
            "image": f"data:image/png;base64,{img_str}",
            "detections": detection_data,
            "timestamp": timestamp or time.time(),
            "metadata": {
                "num_detections": len(detections),
                "config": {
                    "mode": self.config.mode.value,
                    "color_scheme": self.config.color_scheme.value,
                    "opacity": self.config.opacity
                }
            },
            "performance": self.get_performance_stats()
        }
        
        return result
    
    def _get_detection_color(self, detection: BoundingBox, is_top: bool = False, normalize: bool = False) -> Tuple:
        """Get color for a detection based on various factors."""
        # Base color from class_id
        if detection.class_id in self.color_palette:
            color = self.color_palette[detection.class_id]
        else:
            color = self.color_palette[detection.class_id % len(self.color_palette)]
        
        # Adjust for confidence
        if detection.confidence < 0.5:
            # Dim color for low confidence
            factor = detection.confidence * 2  # 0.0-1.0
            color = tuple(int(c * factor) for c in color)
        
        # Highlight top detections
        if is_top:
            # Make color brighter
            color = tuple(min(255, c + 50) for c in color)
        
        # Normalize for matplotlib if needed
        if normalize:
            color = tuple(c / 255.0 for c in color)
        
        return color
    
    def _draw_opencv_bbox(self, image: np.ndarray, detection: BoundingBox, color: Tuple, is_top: bool) -> np.ndarray:
        """Draw bounding box using OpenCV."""
        x1, y1 = int(detection.xmin), int(detection.ymin)
        x2, y2 = int(detection.xmax), int(detection.ymax)
        
        # Draw main rectangle
        thickness = self.config.line_thickness * 2 if is_top else self.config.line_thickness
        
        if self.config.use_rounded_corners:
            # Draw rounded rectangle
            radius = 10
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            
            # Draw rounded corners
            cv2.circle(image, (x1 + radius, y1 + radius), radius, color, -1)
            cv2.circle(image, (x2 - radius, y1 + radius), radius, color, -1)
            cv2.circle(image, (x1 + radius, y2 - radius), radius, color, -1)
            cv2.circle(image, (x2 - radius, y2 - radius), radius, color, -1)
        else:
            # Standard rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # Draw confidence bar
        if self.config.show_confidence:
            bar_height = int((y2 - y1) * detection.confidence)
            bar_y = y2 - bar_height
            cv2.rectangle(image, (x2, bar_y), (x2 + 5, y2), color, -1)
        
        return image
    
    def _draw_opencv_label(self, image: np.ndarray, detection: BoundingBox, color: Tuple) -> np.ndarray:
        """Draw label using OpenCV."""
        # Create label text
        label = self._create_label_text(detection)
        
        if not label:
            return image
        
        # Calculate text size
        font = self.fonts['small']
        font_scale = 0.5
        thickness = 1
        
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )
        
        # Label position (top-left of bbox)
        x, y = int(detection.xmin), int(detection.ymin)
        
        # Draw label background
        bg_x1 = x
        bg_y1 = y - text_height - 10
        bg_x2 = x + text_width
        bg_y2 = y
        
        if bg_y1 < 0:
            bg_y1 = y
            bg_y2 = y + text_height + 10
        
        cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
        
        # Draw label text
        text_y = bg_y1 + text_height + 5 if bg_y1 < y else bg_y1 + text_height - 5
        
        cv2.putText(
            image, label,
            (x, text_y),
            font, font_scale, (255, 255, 255), thickness,
            cv2.LINE_AA
        )
        
        return image
    
    def _draw_opencv_overlays(
        self,
        image: np.ndarray,
        timestamp: Optional[float],
        metadata: Optional[Dict[str, Any]],
        num_detections: int
    ) -> np.ndarray:
        """Draw overlays (timestamp, FPS, etc.) using OpenCV."""
        overlay_texts = []
        
        # Timestamp
        if timestamp is not None and self.config.show_timestamps:
            overlay_texts.append(f"Time: {timestamp:.3f}s")
        
        # FPS
        if self.config.show_fps:
            fps = self.get_current_fps()
            if fps > 0:
                overlay_texts.append(f"FPS: {fps:.1f}")
        
        # Detection count
        overlay_texts.append(f"Detections: {num_detections}")
        
        # Additional metadata
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (int, float)):
                    overlay_texts.append(f"{key}: {value:.2f}")
                else:
                    overlay_texts.append(f"{key}: {value}")
        
        # Draw overlays
        font = self.fonts['small']
        font_scale = 0.6
        thickness = 1
        margin = 10
        
        for i, text in enumerate(overlay_texts):
            position = (margin, margin + (i * 30))
            
            # Text background
            (text_width, text_height), _ = cv2.getTextSize(
                text, font, font_scale, thickness
            )
            
            bg_x1 = position[0] - 5
            bg_y1 = position[1] - text_height - 5
            bg_x2 = position[0] + text_width + 5
            bg_y2 = position[1] + 5
            
            cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
            
            # Text
            cv2.putText(
                image, text,
                position,
                font, font_scale, (255, 255, 255), thickness,
                cv2.LINE_AA
            )
        
        return image
    
    def _draw_matplotlib_bbox(self, ax: plt.Axes, detection: BoundingBox, color: Tuple, is_top: bool):
        """Draw bounding box using Matplotlib."""
        from matplotlib.patches import Rectangle, FancyBBoxPatch
        
        x, y = detection.xmin, detection.ymin
        width = detection.width
        height = detection.height
        
        if self.config.use_rounded_corners:
            # Rounded rectangle
            bbox = FancyBBoxPatch(
                (x, y), width, height,
                boxstyle="round,pad=0.1",
                linewidth=self.config.line_thickness * (2 if is_top else 1),
                edgecolor=color,
                facecolor='none',
                alpha=self.config.opacity
            )
        else:
            # Standard rectangle
            bbox = Rectangle(
                (x, y), width, height,
                linewidth=self.config.line_thickness * (2 if is_top else 1),
                edgecolor=color,
                facecolor='none',
                alpha=self.config.opacity
            )
        
        ax.add_patch(bbox)
        
        # Confidence bar
        if self.config.show_confidence:
            bar_height = height * detection.confidence
            bar = Rectangle(
                (x + width, y + height - bar_height), 5, bar_height,
                linewidth=0,
                edgecolor=color,
                facecolor=color,
                alpha=self.config.opacity * 0.7
            )
            ax.add_patch(bar)
    
    def _draw_matplotlib_label(self, ax: plt.Axes, detection: BoundingBox, color: Tuple):
        """Draw label using Matplotlib."""
        label = self._create_label_text(detection)
        
        if not label:
            return
        
        x, y = detection.xmin, detection.ymin
        
        # Add text with background
        bbox_props = dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8, edgecolor=color)
        
        ax.text(
            x, y - 5, label,
            fontsize=self.config.font_size,
            color='white',
            bbox=bbox_props,
            verticalalignment='bottom',
            horizontalalignment='left'
        )
    
    def _add_matplotlib_legend(self, ax: plt.Axes, detections: List[BoundingBox]):
        """Add legend to Matplotlib plot."""
        # Get unique classes
        unique_classes = {}
        for detection in detections:
            if detection.class_id not in unique_classes:
                unique_classes[detection.class_id] = detection.class_name
        
        # Create legend entries
        legend_elements = []
        for class_id, class_name in unique_classes.items():
            color = self._get_detection_color(
                BoundingBox(0, 0, 0, 0, class_id=class_id, class_name=class_name),
                normalize=True
            )
            
            legend_elements.append(
                Line2D([0], [0], color=color, lw=4, label=f"{class_name} (ID: {class_id})")
            )
        
        # Add count information
        if detections:
            confidence_range = f"{min(d.confidence for d in detections):.2f}-{max(d.confidence for d in detections):.2f}"
            legend_elements.append(
                Line2D([0], [0], color='gray', lw=0, 
                      label=f"Total: {len(detections)}, Confidence: {confidence_range}")
            )
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize='small')
    
    def _draw_pillow_bbox(self, draw: ImageDraw.Draw, detection: BoundingBox, color: Tuple, is_top: bool):
        """Draw bounding box using PIL."""
        x1, y1 = detection.xmin, detection.ymin
        x2, y2 = detection.xmax, detection.ymax
        
        thickness = self.config.line_thickness * 2 if is_top else self.config.line_thickness
        
        # Draw main rectangle
        if self.config.use_rounded_corners:
            # Draw rounded rectangle
            radius = 10
            # Draw four rectangles to create rounded effect
            draw.rectangle([x1, y1 + radius, x2, y2 - radius], fill=None, outline=color, width=thickness)
            draw.rectangle([x1 + radius, y1, x2 - radius, y2], fill=None, outline=color, width=thickness)
            
            # Draw four circles for corners
            draw.ellipse([x1, y1, x1 + 2*radius, y1 + 2*radius], fill=None, outline=color, width=thickness)
            draw.ellipse([x2 - 2*radius, y1, x2, y1 + 2*radius], fill=None, outline=color, width=thickness)
            draw.ellipse([x1, y2 - 2*radius, x1 + 2*radius, y2], fill=None, outline=color, width=thickness)
            draw.ellipse([x2 - 2*radius, y2 - 2*radius, x2, y2], fill=None, outline=color, width=thickness)
        else:
            # Standard rectangle
            draw.rectangle([x1, y1, x2, y2], fill=None, outline=color, width=thickness)
        
        # Draw confidence bar
        if self.config.show_confidence:
            bar_height = (y2 - y1) * detection.confidence
            bar_y = y2 - bar_height
            draw.rectangle([x2, bar_y, x2 + 5, y2], fill=color, outline=color)
    
    def _draw_pillow_label(self, draw: ImageDraw.Draw, detection: BoundingBox, color: Tuple):
        """Draw label using PIL."""
        label = self._create_label_text(detection)
        
        if not label:
            return
        
        font = self.fonts['small']
        
        # Get text size
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Label position
        x, y = detection.xmin, detection.ymin
        
        # Draw label background
        bg_x1 = x
        bg_y1 = y - text_height - 10
        bg_x2 = x + text_width + 10
        bg_y2 = y
        
        if bg_y1 < 0:
            bg_y1 = y
            bg_y2 = y + text_height + 10
        
        draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=color)
        
        # Draw label text
        text_y = bg_y1 + 5 if bg_y1 < y else bg_y1
        
        draw.text(
            (x + 5, text_y),
            label,
            fill=(255, 255, 255),
            font=font
        )
    
    def _draw_pillow_overlays(
        self,
        draw: ImageDraw.Draw,
        image_size: Tuple[int, int],
        timestamp: Optional[float],
        metadata: Optional[Dict[str, Any]],
        num_detections: int
    ):
        """Draw overlays using PIL."""
        overlay_texts = []
        
        # Timestamp
        if timestamp is not None and self.config.show_timestamps:
            overlay_texts.append(f"Time: {timestamp:.3f}s")
        
        # FPS
        if self.config.show_fps:
            fps = self.get_current_fps()
            if fps > 0:
                overlay_texts.append(f"FPS: {fps:.1f}")
        
        # Detection count
        overlay_texts.append(f"Detections: {num_detections}")
        
        # Additional metadata
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (int, float)):
                    overlay_texts.append(f"{key}: {value:.2f}")
                else:
                    overlay_texts.append(f"{key}: {value}")
        
        # Draw overlays
        font = self.fonts['small']
        margin = 10
        
        for i, text in enumerate(overlay_texts):
            position = (margin, margin + (i * 25))
            
            # Text background
            text_bbox = draw.textbbox(position, text, font=font)
            bg_x1 = text_bbox[0] - 5
            bg_y1 = text_bbox[1] - 5
            bg_x2 = text_bbox[2] + 5
            bg_y2 = text_bbox[3] + 5
            
            draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=(0, 0, 0))
            
            # Text
            draw.text(position, text, fill=(255, 255, 255), font=font)
    
    def _create_label_text(self, detection: BoundingBox) -> str:
        """Create label text for detection."""
        parts = []
        
        if self.config.show_class_names:
            parts.append(detection.class_name)
        
        if self.config.show_confidence:
            parts.append(f"{detection.confidence:.2f}")
        
        if self.config.show_track_ids and detection.track_id is not None:
            parts.append(f"ID:{detection.track_id}")
        
        return " ".join(parts) if parts else ""
    
    def _create_hover_text(self, detection: BoundingBox) -> str:
        """Create hover text for interactive visualizations."""
        text = f"""
        Class: {detection.class_name}
        Confidence: {detection.confidence:.2%}
        Position: ({detection.center[0]:.1f}, {detection.center[1]:.1f})
        Size: {detection.width:.1f} × {detection.height:.1f}
        Area: {detection.area:.1f}
        """
        
        if detection.track_id is not None:
            text += f"Track ID: {detection.track_id}\n"
        
        if detection.attributes:
            for key, value in detection.attributes.items():
                if isinstance(value, (int, float)):
                    text += f"{key}: {value:.2f}\n"
                else:
                    text += f"{key}: {value}\n"
        
        return text.strip()
    
    def _generate_cache_key(self, image: np.ndarray, detections: List[BoundingBox], timestamp: Optional[float]) -> str:
        """Generate cache key for current visualization."""
        import hashlib
        
        # Create hashable representation
        key_parts = []
        
        # Image hash (simplified)
        if image is not None:
            # Use downscaled version for hash
            small_image = cv2.resize(image, (32, 32))
            key_parts.append(str(small_image.tobytes()))
        
        # Detections hash
        detections_repr = []
        for det in detections:
            detections_repr.append(f"{det.class_id}:{det.confidence:.2f}:{det.xmin:.1f}:{det.ymin:.1f}:{det.xmax:.1f}:{det.ymax:.1f}")
        
        key_parts.append(",".join(sorted(detections_repr)))
        
        # Config hash
        config_repr = f"{self.config.mode.value}:{self.config.color_scheme.value}:{self.config.opacity}"
        key_parts.append(config_repr)
        
        # Timestamp
        if timestamp is not None:
            key_parts.append(f"{timestamp:.3f}")
        
        # Create hash
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _apply_gradient_overlay(self, image: np.ndarray) -> np.ndarray:
        """Apply gradient overlay for visual appeal."""
        h, w = image.shape[:2]
        
        # Create gradient
        gradient = np.linspace(0, 1, h)
        gradient = np.tile(gradient[:, np.newaxis], (1, w))
        
        if len(image.shape) == 3:
            gradient = np.stack([gradient] * 3, axis=2)
        
        # Apply gradient with low opacity
        overlay_strength = 0.1
        result = image.astype(np.float32) * (1 - overlay_strength * gradient)
        
        return result.astype(np.uint8)
    
    def _apply_shadow_effect(self, image: np.ndarray) -> np.ndarray:
        """Apply shadow effect for depth."""
        # Create drop shadow for image borders
        h, w = image.shape[:2]
        
        # Create shadow mask
        shadow_size = 10
        shadow = np.zeros((h, w), dtype=np.float32)
        
        # Top shadow
        shadow[:shadow_size, :] = np.linspace(0.8, 0, shadow_size)[:, np.newaxis]
        
        # Bottom shadow
        shadow[-shadow_size:, :] = np.linspace(0, 0.8, shadow_size)[:, np.newaxis][::-1]
        
        # Left shadow
        shadow[:, :shadow_size] = np.maximum(shadow[:, :shadow_size], 
                                             np.linspace(0.8, 0, shadow_size)[np.newaxis, :])
        
        # Right shadow
        shadow[:, -shadow_size:] = np.maximum(shadow[:, -shadow_size:],
                                              np.linspace(0, 0.8, shadow_size)[np.newaxis, ::-1])
        
        if len(image.shape) == 3:
            shadow = np.stack([shadow] * 3, axis=2)
        
        # Apply shadow
        result = image.astype(np.float32) * (1 - shadow * 0.3)
        
        return result.astype(np.uint8)
    
    def _save_debug_frame(self, result: Any, timestamp: Optional[float]):
        """Save debug frame to disk."""
        import os
        import json
        
        os.makedirs(self.config.debug_output_path, exist_ok=True)
        
        # Generate filename
        ts = timestamp or time.time()
        filename = f"frame_{int(ts * 1000)}"
        
        # Save based on result type
        if isinstance(result, np.ndarray):
            # OpenCV image
            cv2.imwrite(os.path.join(self.config.debug_output_path, f"{filename}.png"), result)
        
        elif isinstance(result, plt.Figure):
            # Matplotlib figure
            result.savefig(os.path.join(self.config.debug_output_path, f"{filename}.png"),
                          dpi=self.config.output_dpi, bbox_inches='tight')
            plt.close(result)
        
        elif isinstance(result, Image.Image):
            # PIL image
            result.save(os.path.join(self.config.debug_output_path, f"{filename}.png"))
        
        elif isinstance(result, dict):
            # Save JSON data
            with open(os.path.join(self.config.debug_output_path, f"{filename}.json"), 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                serializable_result = self._make_serializable(result)
                json.dump(serializable_result, f, indent=2)
        
        if self.config.debug_mode:
            logger.debug("Saved debug frame: %s", filename)
    
    def _make_serializable(self, obj: Any) -> Any:
        """Make object JSON serializable."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        else:
            return obj
    
    def get_current_fps(self) -> float:
        """Get current FPS based on recent rendering times."""
        if not self.rendering_times:
            return 0.0
        
        # Consider last 30 frames
        recent_times = self.rendering_times[-30:]
        
        if not recent_times:
            return 0.0
        
        avg_time = np.mean(recent_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.rendering_times:
            return {}
        
        return {
            "total_frames_rendered": self.frame_count,
            "avg_rendering_time_ms": np.mean(self.rendering_times) * 1000,
            "std_rendering_time_ms": np.std(self.rendering_times) * 1000,
            "max_rendering_time_ms": np.max(self.rendering_times) * 1000,
            "min_rendering_time_ms": np.min(self.rendering_times) * 1000,
            "current_fps": self.get_current_fps(),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) 
            if (self.cache_hits + self.cache_misses) > 0 else 0.0
        }
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.frame_count = 0
        self.rendering_times = []
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache.clear()
        logger.info("Visualizer statistics reset")


# Factory function for creating visualizers
def create_visualizer(config: Dict[str, Any] = None) -> Visualizer:
    """
    Factory function to create a Visualizer instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured Visualizer instance
    """
    if config is None:
        config = {}
    
    # Convert dict to VisualizationConfig
    config_obj = VisualizationConfig(**config)
    return Visualizer(config_obj)


# Utility functions for common visualization tasks
def draw_bboxes(
    image: np.ndarray,
    bboxes: List[List[float]],
    confidences: Optional[List[float]] = None,
    class_ids: Optional[List[int]] = None,
    class_names: Optional[List[str]] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw multiple bounding boxes on image.
    
    Args:
        image: Input image
        bboxes: List of bounding boxes [x1, y1, x2, y2]
        confidences: Optional confidence scores
        class_ids: Optional class IDs
        class_names: Optional class names
        color: Default color
        thickness: Line thickness
        
    Returns:
        Image with bounding boxes
    """
    if image is None or image.size == 0:
        return image
    
    visualized = image.copy()
    
    if len(visualized.shape) == 2:
        visualized = cv2.cvtColor(visualized, cv2.COLOR_GRAY2BGR)
    
    for i, bbox in enumerate(bboxes):
        if len(bbox) < 4:
            continue
        
        x1, y1, x2, y2 = map(int, bbox[:4])
        
        # Get color for this box
        box_color = color
        if class_ids is not None and i < len(class_ids):
            # Simple color mapping based on class ID
            hue = (class_ids[i] * 50) % 180
            box_color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
        
        # Draw bounding box
        cv2.rectangle(visualized, (x1, y1), (x2, y2), box_color, thickness)
        
        # Create label
        label_parts = []
        
        if class_names is not None and i < len(class_names):
            label_parts.append(class_names[i])
        
        if confidences is not None and i < len(confidences):
            label_parts.append(f"{confidences[i]:.2f}")
        
        if class_ids is not None and i < len(class_ids):
            label_parts.append(f"ID:{class_ids[i]}")
        
        if label_parts:
            label = " ".join(label_parts)
            
            # Label background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, 1
            )
            
            label_bg = ((x1, y1 - text_height - 10),
                       (x1 + text_width, y1))
            
            cv2.rectangle(visualized, label_bg[0], label_bg[1], box_color, -1)
            
            # Label text
            cv2.putText(
                visualized, label,
                (x1, y1 - 5),
                font, font_scale, (255, 255, 255), 1,
                cv2.LINE_AA
            )
    
    return visualized


def draw_keypoints(
    image: np.ndarray,
    keypoints: List[List[float]],
    skeleton: Optional[List[Tuple[int, int]]] = None,
    colors: Optional[List[Tuple[int, int, int]]] = None,
    radius: int = 3,
    thickness: int = 2
) -> np.ndarray:
    """
    Draw keypoints and skeleton on image.
    
    Args:
        image: Input image
        keypoints: List of keypoints [x, y, confidence] for each point
        skeleton: Optional skeleton connections as list of (idx1, idx2) pairs
        colors: Optional colors for each keypoint
        radius: Keypoint radius
        thickness: Line thickness
        
    Returns:
        Image with keypoints
    """
    if image is None or image.size == 0:
        return image
    
    visualized = image.copy()
    
    if len(visualized.shape) == 2:
        visualized = cv2.cvtColor(visualized, cv2.COLOR_GRAY2BGR)
    
    # Draw skeleton connections first
    if skeleton is not None:
        for idx1, idx2 in skeleton:
            if (idx1 < len(keypoints) and idx2 < len(keypoints) and
                len(keypoints[idx1]) >= 3 and len(keypoints[idx2]) >= 3 and
                keypoints[idx1][2] > 0.1 and keypoints[idx2][2] > 0.1):
                
                x1, y1 = int(keypoints[idx1][0]), int(keypoints[idx1][1])
                x2, y2 = int(keypoints[idx2][0]), int(keypoints[idx2][1])
                
                # Get color
                if colors is not None and idx1 < len(colors):
                    color = colors[idx1]
                else:
                    color = (0, 255, 0)
                
                cv2.line(visualized, (x1, y1), (x2, y2), color, thickness)
    
    # Draw keypoints
    for i, kp in enumerate(keypoints):
        if len(kp) >= 3 and kp[2] > 0.1:  # Confidence threshold
            x, y = int(kp[0]), int(kp[1])
            
            # Get color
            if colors is not None and i < len(colors):
                color = colors[i]
            else:
                # Default color based on confidence
                confidence = kp[2]
                intensity = int(255 * confidence)
                color = (0, intensity, 255 - intensity)
            
            # Draw keypoint
            cv2.circle(visualized, (x, y), radius, color, -1)
            
            # Draw confidence ring
            if len(kp) >= 3:
                confidence_radius = int(radius * kp[2])
                cv2.circle(visualized, (x, y), confidence_radius, color, 1)
    
    return visualized


def create_heatmap(
    image: np.ndarray,
    heatmap_data: np.ndarray,
    alpha: float = 0.5,
    colormap: str = 'jet'
) -> np.ndarray:
    """
    Overlay heatmap on image.
    
    Args:
        image: Input image
        heatmap_data: Heatmap data (2D array)
        alpha: Transparency factor
        colormap: Colormap name
        
    Returns:
        Image with heatmap overlay
    """
    if image is None or image.size == 0:
        return image
    
    # Ensure heatmap is same size as image
    if heatmap_data.shape[:2] != image.shape[:2]:
        heatmap_data = cv2.resize(heatmap_data, (image.shape[1], image.shape[0]))
    
    # Normalize heatmap
    heatmap_norm = cv2.normalize(heatmap_data, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_norm = heatmap_norm.astype(np.uint8)
    
    # Apply colormap
    if colormap.lower() == 'hot':
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_HOT)
    elif colormap.lower() == 'cool':
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_COOL)
    elif colormap.lower() == 'spring':
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_SPRING)
    elif colormap.lower() == 'autumn':
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_AUTUMN)
    else:  # Default to jet
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    
    # Blend with image
    if len(image.shape) == 2:
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_color = image.copy()
    
    result = cv2.addWeighted(image_color, 1 - alpha, heatmap_color, alpha, 0)
    
    return result


def save_visualization(
    visualization: Any,
    filepath: str,
    format: str = 'png',
    quality: int = 95
):
    """
    Save visualization to file.
    
    Args:
        visualization: Visualization result
        filepath: Output file path
        format: Output format
        quality: Quality for lossy formats
    """
    if visualization is None:
        logger.warning("Cannot save None visualization")
        return
    
    try:
        if isinstance(visualization, np.ndarray):
            # OpenCV image
            cv2.imwrite(filepath, visualization)
        
        elif isinstance(visualization, plt.Figure):
            # Matplotlib figure
            visualization.savefig(filepath, format=format, dpi=300, bbox_inches='tight')
            plt.close(visualization)
        
        elif isinstance(visualization, Image.Image):
            # PIL image
            visualization.save(filepath, format=format, quality=quality)
        
        elif isinstance(visualization, dict) and 'image' in visualization:
            # Web format
            if visualization['image'].startswith('data:image'):
                # Extract base64 data
                import base64
                data = visualization['image'].split(',')[1]
                image_data = base64.b64decode(data)
                with open(filepath, 'wb') as f:
                    f.write(image_data)
            else:
                logger.warning("Unknown image format in visualization dict")
        
        else:
            logger.warning("Unknown visualization type: %s", type(visualization))
        
        logger.debug("Saved visualization to: %s", filepath)
        
    except Exception as e:
        logger.error("Failed to save visualization: %s", str(e))
