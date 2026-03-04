"""
ROS2-based demonstration: Build the word "VLA" using dual-arm robots.

This script communicates with the dual-arm robots via ROS2 topics
to arrange Jenga blocks to form the letters "VLA".
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

from std_msgs.msg import Float64MultiArray, String
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger


class VLALetterBuilderROS2(Node):
    """
    ROS2 node for building "VLA" letters using dual-arm coordination.
    """
    
    def __init__(self):
        super().__init__('vla_letter_builder_ros2')
        
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
        
        # Services client
        self.connect_client = self.create_client(Trigger, '/dual_arm/connect')
        self.reset_client = self.create_client(Trigger, '/dual_arm/reset')
        
        # State
        self.left_pose = None
        self.right_pose = None
        
        # Configuration
        self.table_height = 0.1
        self.approach_height = 0.05
        self.letter_start_x = 0.4
        self.letter_start_y = -0.1
        self.letter_spacing = 0.15
        
        # Letter patterns (same as demo_build_vla_2d.py)
        self.letter_patterns = {
            'V': [
                (0.0, 0.08, 0.0), (0.02, 0.04, 0.0), (0.04, 0.0, 0.0),
                (0.06, 0.04, 0.0), (0.08, 0.08, 0.0), (0.01, 0.06, 0.0), (0.07, 0.06, 0.0),
            ],
            'L': [
                (0.0, 0.0, 0.0), (0.0, 0.025, 0.0), (0.0, 0.05, 0.0),
                (0.025, 0.0, 0.0), (0.05, 0.0, 0.0), (0.075, 0.0, 0.0),
            ],
            'A': [
                (0.04, 0.08, 0.0), (0.02, 0.06, 0.0), (0.06, 0.06, 0.0),
                (0.02, 0.04, 0.0), (0.04, 0.04, 0.0), (0.06, 0.04, 0.0), (0.04, 0.0, 0.0),
            ],
        }
        
        # Source positions
        self.source_positions = [
            [0.2, -0.3, self.table_height],
            [0.2, -0.25, self.table_height],
            [0.2, -0.2, self.table_height],
            [0.2, -0.15, self.table_height],
            [0.2, -0.1, self.table_height],
            [0.2, -0.05, self.table_height],
            [0.2, 0.0, self.table_height],
            [0.2, 0.05, self.table_height],
            [0.2, 0.1, self.table_height],
            [0.2, 0.15, self.table_height],
            [0.2, 0.2, self.table_height],
            [0.2, 0.25, self.table_height],
            [0.2, 0.3, self.table_height],
            [0.25, -0.3, self.table_height],
            [0.25, -0.25, self.table_height],
            [0.25, -0.2, self.table_height],
            [0.25, -0.15, self.table_height],
            [0.25, -0.1, self.table_height],
            [0.25, -0.05, self.table_height],
            [0.25, 0.0, self.table_height],
        ]
        
        self.get_logger().info('VLA letter builder ROS2 node initialized')
    
    def left_pose_callback(self, msg: PoseStamped):
        """Callback for left arm pose."""
        self.left_pose = msg
    
    def right_pose_callback(self, msg: PoseStamped):
        """Callback for right arm pose."""
        self.right_pose = msg
    
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
        time.sleep(0.1)
    
    def get_place_pose(self, letter: str, letter_idx: int, block_idx: int, approach: bool = True) -> np.ndarray:
        """Get pose for placing a block to form a letter."""
        pattern = self.letter_patterns[letter]
        rel_x, rel_y, yaw = pattern[block_idx]
        
        letter_x = self.letter_start_x + letter_idx * self.letter_spacing
        x = letter_x + rel_x
        y = self.letter_start_y + rel_y
        z = self.table_height
        
        height = self.approach_height if approach else 0.0
        
        return np.array([x, y, z + height, 0.0, np.pi, yaw])
    
    def build_letter(self, letter: str, letter_idx: int, block_counter: int, use_dual_arm: bool = True) -> int:
        """Build a single letter."""
        self.get_logger().info(f"Building letter '{letter}'")
        
        pattern = self.letter_patterns[letter]
        num_blocks = len(pattern)
        
        if use_dual_arm and num_blocks > 3:
            left_blocks = num_blocks // 2
            right_blocks = num_blocks - left_blocks
        else:
            left_blocks = num_blocks
            right_blocks = 0
        
        # Build with left arm
        for i in range(left_blocks):
            if block_counter >= len(self.source_positions):
                self.get_logger().warn("Not enough blocks")
                return block_counter
            
            source_pos = self.source_positions[block_counter]
            block_counter += 1
            
            # Pick and place sequence (similar to tower builder)
            pick_approach = np.array([source_pos[0], source_pos[1], source_pos[2] + self.approach_height, 0.0, np.pi, 0.0])
            pick_grasp = np.array([source_pos[0], source_pos[1], source_pos[2] + 0.01, 0.0, np.pi, 0.0])
            place_approach = self.get_place_pose(letter, letter_idx, i, approach=True)
            place_position = self.get_place_pose(letter, letter_idx, i, approach=False)
            
            # Execute pick and place
            self.send_action(np.concatenate([pick_approach, [1.0], np.zeros(7)]))
            time.sleep(0.3)
            self.send_action(np.concatenate([pick_grasp, [0.0], np.zeros(7)]))
            time.sleep(0.3)
            self.send_action(np.concatenate([pick_approach, [0.0], np.zeros(7)]))
            time.sleep(0.3)
            self.send_action(np.concatenate([place_approach, [0.0], np.zeros(7)]))
            time.sleep(0.3)
            self.send_action(np.concatenate([place_position, [0.0], np.zeros(7)]))
            time.sleep(0.2)
            self.send_action(np.concatenate([place_position, [1.0], np.zeros(7)]))
            time.sleep(0.2)
            self.send_action(np.concatenate([place_approach, [1.0], np.zeros(7)]))
            time.sleep(0.3)
        
        # Build with right arm (if dual-arm)
        if right_blocks > 0:
            for i in range(left_blocks, num_blocks):
                if block_counter >= len(self.source_positions):
                    return block_counter
                
                source_pos = self.source_positions[block_counter]
                block_counter += 1
                
                # Similar pick and place sequence for right arm
                pick_approach = np.array([source_pos[0], source_pos[1], source_pos[2] + self.approach_height, 0.0, np.pi, 0.0])
                pick_grasp = np.array([source_pos[0], source_pos[1], source_pos[2] + 0.01, 0.0, np.pi, 0.0])
                place_approach = self.get_place_pose(letter, letter_idx, i, approach=True)
                place_position = self.get_place_pose(letter, letter_idx, i, approach=False)
                
                self.send_action(np.concatenate([np.zeros(7), pick_approach, [1.0]]))
                time.sleep(0.3)
                self.send_action(np.concatenate([np.zeros(7), pick_grasp, [0.0]]))
                time.sleep(0.3)
                self.send_action(np.concatenate([np.zeros(7), pick_approach, [0.0]]))
                time.sleep(0.3)
                self.send_action(np.concatenate([np.zeros(7), place_approach, [0.0]]))
                time.sleep(0.3)
                self.send_action(np.concatenate([np.zeros(7), place_position, [0.0]]))
                time.sleep(0.2)
                self.send_action(np.concatenate([np.zeros(7), place_position, [1.0]]))
                time.sleep(0.2)
                self.send_action(np.concatenate([np.zeros(7), place_approach, [1.0]]))
                time.sleep(0.3)
        
        return block_counter
    
    def build_vla(self, use_dual_arm: bool = True):
        """Build the word "VLA"."""
        self.get_logger().info("Starting VLA letter construction")
        
        self.reset()
        time.sleep(2.0)
        
        block_counter = 0
        letters = ['V', 'L', 'A']
        
        for letter_idx, letter in enumerate(letters):
            block_counter = self.build_letter(letter, letter_idx, block_counter, use_dual_arm)
            time.sleep(1.0)
        
        self.get_logger().info("VLA construction completed")
        self.reset()
        time.sleep(2.0)


def main(args=None):
    """Main function."""
    rclpy.init(args=args)
    node = VLALetterBuilderROS2()
    
    if not node.connect():
        node.get_logger().error("Failed to connect to hardware")
        return 1
    
    try:
        node.build_vla(use_dual_arm=True)
    except KeyboardInterrupt:
        node.get_logger().warn("Interrupted by user")
    finally:
        node.destroy_node()
        rclpy.shutdown()
    
    return 0


if __name__ == '__main__':
    exit(main())

