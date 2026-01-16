#!/usr/bin/env python3
"""
Simple YOLO Object Detection Node for ROS 2
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from robot_interfaces.msg import ObjectDetection
from cv_bridge import CvBridge

class YOLODetectorNode(Node):
    def __init__(self):
        super().__init__('yolo_detector_node')
        
        self.get_logger().info('YOLO Detector Node starting...')
        
        # Parameters
        self.declare_parameter('camera_topic', '/camera/camera/image_raw')
        self.declare_parameter('confidence_threshold', 0.6)
        self.declare_parameter('model_path', 'yolov5s')
        
        camera_topic = self.get_parameter('camera_topic').value
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Subscriber
        self.image_sub = self.create_subscription(
            Image,
            camera_topic,
            self.image_callback,
            10
        )
        
        # Publisher
        self.detection_pub = self.create_publisher(
            ObjectDetection,
            '/object_detections',
            10
        )
        
        self.get_logger().info(f'Subscribed to {camera_topic}')
        self.get_logger().info('YOLO Detector ready!')
    
    def image_callback(self, msg):
        # For now, just log that we're receiving images
        # Real YOLO implementation would go here
        self.get_logger().info('Received image', once=True)

def main(args=None):
    rclpy.init(args=args)
    node = YOLODetectorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
