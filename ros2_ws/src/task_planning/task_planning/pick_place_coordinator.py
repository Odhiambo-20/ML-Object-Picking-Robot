#!/usr/bin/env python3
"""
Pick and Place Coordinator Node
"""

import rclpy
from rclpy.node import Node
from robot_interfaces.msg import ObjectDetection

class PickPlaceCoordinator(Node):
    def __init__(self):
        super().__init__('pick_place_coordinator')
        
        self.get_logger().info('Pick-Place Coordinator starting...')
        
        # Parameters
        self.declare_parameter('min_confidence', 0.6)
        self.declare_parameter('target_classes', ['bottle', 'cup', 'box'])
        
        # Subscriber
        self.detection_sub = self.create_subscription(
            ObjectDetection,
            '/object_detections',
            self.detection_callback,
            10
        )
        
        self.get_logger().info('Pick-Place Coordinator ready!')
    
    def detection_callback(self, msg):
        self.get_logger().info(f'Detected: {msg.class_name}')

def main(args=None):
    rclpy.init(args=args)
    node = PickPlaceCoordinator()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
