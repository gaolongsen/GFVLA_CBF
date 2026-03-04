"""
ROS2-based demonstration: Build a 3D tower using dual-arm robots.

This script communicates with the dual-arm robots via ROS2 topics
to build a tower using Jenga blocks.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
import numpy as np
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from std_msgs.msg import Float64MultiArray, Bool, String
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger


class TowerBuilderROS2(Node):
    """
    ROS2 node for building a tower using dual-arm coordination.
    
    Subscribes to:
    - /dual_arm/left_arm/pose: Current left arm pose
    - /dual_arm/right_arm/pose: Current right arm pose
    - /dual_arm/status: System status
    
    Publishes to:
    - /dual_arm/action: Dual-arm action commands
    """
    
    def __init__(self):
        super().__init__('tower_builder_ros2')
        
        # QoS profile
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            depth=10
        )
        
        # Publishers
        self.action_pub = self.create_publisher(
            Float64MultiArray, '/dual_arm/action', qos_profile
        )
        
        # Subscribers
        self.left_pose_sub = self.create_subscription(
            PoseStamped,
            '/dual_arm/left_arm/pose',
            self.left_pose_callback,
            10
        )
        self.right_pose_sub = self.create_subscription(
            PoseStamped,
            '/dual_arm/right_arm/pose',
            self.right_pose_callback,
            10
        )
        self.status_sub = self.create_subscription(
            String,
            '/dual_arm/status',
            self.status_callback,
            10
        )
        
        # Services client
        self.connect_client = self.create_client(Trigger, '/dual_arm/connect')
        self.reset_client = self.create_client(Trigger, '/dual_arm/reset')
        
        # State
        self.left_pose = None
        self.right_pose = None
        self.status = None
        
        # Tower configuration
        self.block_size = 0.015
        self.tower_base_x = 0.5
        self.tower_base_y = 0.0
        self.tower_base_z = 0.1
        self.approach_height = 0.05
        
        # Source block positions
        self.source_positions = [
            [0.3, -0.2, 0.1],
            [0.3, -0.15, 0.1],
            [0.3, -0.1, 0.1],
            [0.3, -0.05, 0.1],
            [0.3, 0.0, 0.1],
            [0.3, 0.05, 0.1],
        ]
        
        self.get_logger().info('Tower builder ROS2 node initialized')
    
    def left_pose_callback(self, msg: PoseStamped):
        """Callback for left arm pose."""
        self.left_pose = msg
    
    def right_pose_callback(self, msg: PoseStamped):
        """Callback for right arm pose."""
        self.right_pose = msg
    
    def status_callback(self, msg: String):
        """Callback for system status."""
        self.status = msg.data
    
    def wait_for_service(self, client, timeout=5.0):
        """Wait for service to be available."""
        if not client.wait_for_service(timeout_sec=timeout):
            self.get_logger().error(f"Service {client.srv_name} not available")
            return False
        return True
    
    def connect(self) -> bool:
        """Connect to hardware via ROS2 service."""
        if not self.wait_for_service(self.connect_client):
            return False
        
        request = Trigger.Request()
        future = self.connect_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result().success:
            self.get_logger().info("Connected to hardware")
            return True
        else:
            self.get_logger().error(f"Connection failed: {future.result().message}")
            return False
    
    def reset(self) -> bool:
        """Reset robots to home position."""
        if not self.wait_for_service(self.reset_client):
            return False
        
        request = Trigger.Request()
        future = self.reset_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        return future.result().success
    
    def send_action(self, action: np.ndarray):
        """Send action command via ROS2."""
        msg = Float64MultiArray()
        msg.data = action.tolist()
        self.action_pub.publish(msg)
        time.sleep(0.1)  # Small delay
    
    def get_place_pose(self, level: int, block_idx: int, approach: bool = True) -> np.ndarray:
        """Get pose for placing a block on the tower."""
        level_height = self.tower_base_z + level * self.block_size
        
        if level % 2 == 0:
            x_offset = (block_idx - 1) * 0.075
            y_offset = 0.0
            yaw = 0.0
        else:
            x_offset = 0.0
            y_offset = (block_idx - 1) * 0.075
            yaw = np.pi / 2
        
        x = self.tower_base_x + x_offset
        y = self.tower_base_y + y_offset
        z = level_height
        
        height = self.approach_height if approach else 0.0
        
        return np.array([x, y, z + height, 0.0, np.pi, yaw])
    
    def build_tower(self, num_levels: int = 3, num_blocks_per_level: int = 3):
        """Build a tower with specified configuration."""
        self.get_logger().info(f"Starting tower construction: {num_levels} levels")
        
        # Reset to home
        self.reset()
        time.sleep(2.0)
        
        block_counter = 0
        
        for level in range(num_levels):
            self.get_logger().info(f"Building level {level + 1}/{num_levels}")
            
            for block_idx in range(num_blocks_per_level):
                if block_counter >= len(self.source_positions):
                    self.get_logger().warn("Not enough source blocks")
                    return
                
                source_pos = self.source_positions[block_counter]
                block_counter += 1
                
                # Pick block with left arm
                pick_approach = np.array([
                    source_pos[0], source_pos[1], source_pos[2] + self.approach_height,
                    0.0, np.pi, 0.0
                ])
                pick_grasp = np.array([
                    source_pos[0], source_pos[1], source_pos[2] + 0.01,
                    0.0, np.pi, 0.0
                ])
                
                # Approach
                action = np.concatenate([pick_approach, [1.0], np.zeros(7)])
                self.send_action(action)
                time.sleep(0.5)
                
                # Grasp
                action = np.concatenate([pick_grasp, [0.0], np.zeros(7)])
                self.send_action(action)
                time.sleep(0.5)
                
                # Lift
                action = np.concatenate([pick_approach, [0.0], np.zeros(7)])
                self.send_action(action)
                time.sleep(0.5)
                
                # Place block
                place_approach = self.get_place_pose(level, block_idx, approach=True)
                place_position = self.get_place_pose(level, block_idx, approach=False)
                
                # Move to tower
                action = np.concatenate([place_approach, [0.0], np.zeros(7)])
                self.send_action(action)
                time.sleep(0.5)
                
                # Place
                action = np.concatenate([place_position, [0.0], np.zeros(7)])
                self.send_action(action)
                time.sleep(0.3)
                
                # Release
                action = np.concatenate([place_position, [1.0], np.zeros(7)])
                self.send_action(action)
                time.sleep(0.3)
                
                # Retract
                action = np.concatenate([place_approach, [1.0], np.zeros(7)])
                self.send_action(action)
                time.sleep(0.5)
            
            time.sleep(1.0)
        
        self.get_logger().info("Tower construction completed")
        self.reset()
        time.sleep(2.0)


def main(args=None):
    """Main function."""
    rclpy.init(args=args)
    node = TowerBuilderROS2()
    
    # Connect to hardware
    if not node.connect():
        node.get_logger().error("Failed to connect to hardware")
        return 1
    
    try:
        # Build tower
        node.build_tower(num_levels=3, num_blocks_per_level=3)
    except KeyboardInterrupt:
        node.get_logger().warn("Interrupted by user")
    finally:
        node.destroy_node()
        rclpy.shutdown()
    
    return 0


if __name__ == '__main__':
    exit(main())

