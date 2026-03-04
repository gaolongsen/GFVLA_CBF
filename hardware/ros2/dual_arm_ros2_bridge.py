"""
ROS2 bridge for dual-arm hardware interface.

This module provides ROS2 nodes and interfaces for communicating with
the dual-arm robot system (UR5e + UR10e) via ROS2 topics and services.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import numpy as np
from typing import Optional
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hardware import DualArmHardwareInterface, HardwareConfig
from hardware.vision import CameraInterface

# ROS2 message types
from std_msgs.msg import Float64MultiArray, Bool, String
from geometry_msgs.msg import Pose, PoseStamped, Twist
from sensor_msgs.msg import Image, PointCloud2
from std_srvs.srv import SetBool, Trigger
import cv2
from cv_bridge import CvBridge
from sensor_msgs_py import point_cloud2


class DualArmROS2Bridge(Node):
    """
    ROS2 node that bridges the hardware interface with ROS2 communication.
    
    Publishes:
    - /dual_arm/left_arm/pose: Current pose of UR5e (left arm)
    - /dual_arm/right_arm/pose: Current pose of UR10e (right arm)
    - /dual_arm/left_arm/joint_states: Joint states of UR5e
    - /dual_arm/right_arm/joint_states: Joint states of UR10e
    - /dual_arm/left_arm/gripper_state: Robotiq gripper state
    - /dual_arm/right_arm/gripper_state: Barrett Hand state
    - /dual_arm/camera/rgb: RGB image from camera
    - /dual_arm/camera/depth: Depth image from camera
    - /dual_arm/camera/pointcloud: Point cloud from camera
    - /dual_arm/status: System status
    
    Subscribes:
    - /dual_arm/action: 14-dimensional action command [UR5e pose(6) + gripper(1) + UR10e pose(6) + gripper(1)]
    - /dual_arm/left_arm/command: Command for left arm only
    - /dual_arm/right_arm/command: Command for right arm only
    - /dual_arm/emergency_stop: Emergency stop command
    
    Services:
    - /dual_arm/connect: Connect to hardware
    - /dual_arm/disconnect: Disconnect from hardware
    - /dual_arm/reset: Reset robots to home position
    - /dual_arm/get_observation: Get current observation
    """
    
    def __init__(self):
        super().__init__('dual_arm_ros2_bridge')
        
        # Initialize hardware interface
        self.config = HardwareConfig()
        self.hardware = DualArmHardwareInterface(self.config)
        self.camera = None
        self.bridge = CvBridge()
        
        # State
        self.connected = False
        self.publish_rate = 30.0  # Hz
        
        # QoS profiles
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )
        
        # Publishers
        self.left_pose_pub = self.create_publisher(
            PoseStamped, '/dual_arm/left_arm/pose', qos_profile
        )
        self.right_pose_pub = self.create_publisher(
            PoseStamped, '/dual_arm/right_arm/pose', qos_profile
        )
        self.left_joint_pub = self.create_publisher(
            Float64MultiArray, '/dual_arm/left_arm/joint_states', qos_profile
        )
        self.right_joint_pub = self.create_publisher(
            Float64MultiArray, '/dual_arm/right_arm/joint_states', qos_profile
        )
        self.left_gripper_pub = self.create_publisher(
            Float64MultiArray, '/dual_arm/left_arm/gripper_state', qos_profile
        )
        self.right_gripper_pub = self.create_publisher(
            Float64MultiArray, '/dual_arm/right_arm/gripper_state', qos_profile
        )
        self.rgb_pub = self.create_publisher(
            Image, '/dual_arm/camera/rgb', qos_profile
        )
        self.depth_pub = self.create_publisher(
            Image, '/dual_arm/camera/depth', qos_profile
        )
        self.pointcloud_pub = self.create_publisher(
            PointCloud2, '/dual_arm/camera/pointcloud', qos_profile
        )
        self.status_pub = self.create_publisher(
            String, '/dual_arm/status', qos_profile
        )
        
        # Subscribers
        self.action_sub = self.create_subscription(
            Float64MultiArray,
            '/dual_arm/action',
            self.action_callback,
            10
        )
        self.left_cmd_sub = self.create_subscription(
            Float64MultiArray,
            '/dual_arm/left_arm/command',
            self.left_arm_callback,
            10
        )
        self.right_cmd_sub = self.create_subscription(
            Float64MultiArray,
            '/dual_arm/right_arm/command',
            self.right_arm_callback,
            10
        )
        self.emergency_stop_sub = self.create_subscription(
            Bool,
            '/dual_arm/emergency_stop',
            self.emergency_stop_callback,
            10
        )
        
        # Services
        self.connect_srv = self.create_service(
            Trigger, '/dual_arm/connect', self.connect_service
        )
        self.disconnect_srv = self.create_service(
            Trigger, '/dual_arm/disconnect', self.disconnect_service
        )
        self.reset_srv = self.create_service(
            Trigger, '/dual_arm/reset', self.reset_service
        )
        self.get_obs_srv = self.create_service(
            Trigger, '/dual_arm/get_observation', self.get_observation_service
        )
        
        # Timer for publishing state
        self.timer = self.create_timer(1.0 / self.publish_rate, self.publish_state)
        
        self.get_logger().info('Dual-arm ROS2 bridge node initialized')
    
    def connect_service(self, request, response):
        """Service to connect to hardware."""
        if self.connected:
            response.success = True
            response.message = "Already connected"
            return response
        
        # Connect hardware
        if self.hardware.connect():
            # Connect camera if available
            try:
                self.camera = CameraInterface(camera_type='realsense')
                if self.camera.connect():
                    self.get_logger().info("Camera connected")
            except Exception as e:
                self.get_logger().warn(f"Camera connection failed: {e}")
            
            self.connected = True
            response.success = True
            response.message = "Connected to hardware"
        else:
            response.success = False
            response.message = "Failed to connect to hardware"
        
        return response
    
    def disconnect_service(self, request, response):
        """Service to disconnect from hardware."""
        if not self.connected:
            response.success = True
            response.message = "Already disconnected"
            return response
        
        if self.camera is not None:
            self.camera.disconnect()
        self.hardware.disconnect()
        self.connected = False
        
        response.success = True
        response.message = "Disconnected from hardware"
        return response
    
    def reset_service(self, request, response):
        """Service to reset robots to home position."""
        if not self.connected:
            response.success = False
            response.message = "Not connected to hardware"
            return response
        
        self.hardware.reset()
        response.success = True
        response.message = "Robots reset to home position"
        return response
    
    def get_observation_service(self, request, response):
        """Service to get current observation."""
        if not self.connected:
            response.success = False
            response.message = "Not connected to hardware"
            return response
        
        obs = self.hardware.get_observation()
        # Observation is published via topics, so we just return success
        response.success = True
        response.message = "Observation available via topics"
        return response
    
    def action_callback(self, msg: Float64MultiArray):
        """Callback for dual-arm action command."""
        if not self.connected:
            self.get_logger().warn("Received action but not connected")
            return
        
        action = np.array(msg.data)
        if len(action) != 14:
            self.get_logger().error(f"Invalid action dimension: {len(action)}, expected 14")
            return
        
        success, error = self.hardware.execute_action(action, blocking=False)
        if not success:
            self.get_logger().error(f"Action execution failed: {error}")
    
    def left_arm_callback(self, msg: Float64MultiArray):
        """Callback for left arm only command."""
        if not self.connected:
            return
        
        left_action = np.array(msg.data)
        if len(left_action) != 7:
            self.get_logger().error(f"Invalid left arm action dimension: {len(left_action)}")
            return
        
        # Create full 14-dim action with right arm staying in place
        action = np.concatenate([left_action, np.zeros(7)])
        success, error = self.hardware.execute_action(action, blocking=False)
        if not success:
            self.get_logger().error(f"Left arm action failed: {error}")
    
    def right_arm_callback(self, msg: Float64MultiArray):
        """Callback for right arm only command."""
        if not self.connected:
            return
        
        right_action = np.array(msg.data)
        if len(right_action) != 7:
            self.get_logger().error(f"Invalid right arm action dimension: {len(right_action)}")
            return
        
        # Create full 14-dim action with left arm staying in place
        action = np.concatenate([np.zeros(7), right_action])
        success, error = self.hardware.execute_action(action, blocking=False)
        if not success:
            self.get_logger().error(f"Right arm action failed: {error}")
    
    def emergency_stop_callback(self, msg: Bool):
        """Callback for emergency stop."""
        if msg.data and self.connected:
            self.hardware.emergency_stop()
            self.get_logger().warn("EMERGENCY STOP ACTIVATED")
    
    def publish_state(self):
        """Publish current robot state."""
        if not self.connected:
            return
        
        try:
            # Get observation
            obs = self.hardware.get_observation()
            robot_state = obs.get('robot_state', None)
            
            if robot_state is None:
                return
            
            # Publish left arm pose (UR5e)
            if len(robot_state) >= 7:
                left_pose = self._array_to_pose_stamped(robot_state[:6], 'base_link')
                self.left_pose_pub.publish(left_pose)
                
                # Publish left arm joint states (if available)
                if len(robot_state) >= 13:
                    left_joints = Float64MultiArray()
                    left_joints.data = robot_state[6:12].tolist() if len(robot_state) > 12 else []
                    self.left_joint_pub.publish(left_joints)
                
                # Publish left gripper state
                left_gripper = Float64MultiArray()
                left_gripper.data = [robot_state[6]]  # Robotiq gripper
                self.left_gripper_pub.publish(left_gripper)
            
            # Publish right arm pose (UR10e)
            if len(robot_state) >= 14:
                right_pose = self._array_to_pose_stamped(robot_state[7:13], 'base_link')
                self.right_pose_pub.publish(right_pose)
                
                # Publish right arm joint states (if available)
                if len(robot_state) >= 20:
                    right_joints = Float64MultiArray()
                    right_joints.data = robot_state[13:19].tolist()
                    self.right_joint_pub.publish(right_joints)
                
                # Publish right gripper state
                right_gripper = Float64MultiArray()
                right_gripper.data = [robot_state[13]]  # Barrett Hand
                self.right_gripper_pub.publish(right_gripper)
            
            # Publish camera data
            if self.camera is not None:
                rgb_image, depth_image = self.camera.capture_frame()
                if rgb_image is not None:
                    # Convert BGR to RGB
                    rgb_image_rgb = rgb_image[:, :, ::-1]
                    ros_image = self.bridge.cv2_to_imgmsg(rgb_image_rgb, 'rgb8')
                    self.rgb_pub.publish(ros_image)
                
                if depth_image is not None:
                    ros_depth = self.bridge.cv2_to_imgmsg(depth_image, '32FC1')
                    self.depth_pub.publish(ros_depth)
                    
                    # Publish point cloud
                    try:
                        pointcloud = self._depth_to_pointcloud_msg(depth_image)
                        if pointcloud is not None:
                            self.pointcloud_pub.publish(pointcloud)
                    except Exception as e:
                        self.get_logger().debug(f"Point cloud conversion failed: {e}")
            
            # Publish status
            status_msg = String()
            status_msg.data = "operational"
            self.status_pub.publish(status_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing state: {e}")
    
    def _array_to_pose_stamped(self, pose_array: np.ndarray, frame_id: str) -> PoseStamped:
        """Convert pose array to ROS2 PoseStamped message."""
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = frame_id
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        
        pose = pose_stamped.pose
        pose.position.x = float(pose_array[0])
        pose.position.y = float(pose_array[1])
        pose.position.z = float(pose_array[2])
        
        # Convert roll, pitch, yaw to quaternion
        from scipy.spatial.transform import Rotation as R
        r = R.from_euler('xyz', pose_array[3:6], degrees=False)
        quat = r.as_quat()
        pose.orientation.x = float(quat[0])
        pose.orientation.y = float(quat[1])
        pose.orientation.z = float(quat[2])
        pose.orientation.w = float(quat[3])
        
        return pose_stamped
    
    def _depth_to_pointcloud_msg(self, depth_image: np.ndarray) -> Optional[PointCloud2]:
        """Convert depth image to PointCloud2 message."""
        if self.camera is None:
            return None
        
        intrinsics = self.camera.get_camera_intrinsics()
        height, width = depth_image.shape
        
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
        
        points = []
        for v in range(0, height, 4):  # Downsample
            for u in range(0, width, 4):
                depth = depth_image[v, u]
                if depth > 0 and depth < 2.0:
                    x = (u - cx) * depth / fx
                    y = (v - cy) * depth / fy
                    z = depth
                    points.append([x, y, z])
        
        if len(points) == 0:
            return None
        
        # Create PointCloud2 message
        header = self.get_clock().now().to_msg()
        header.frame_id = 'camera_frame'
        
        return point_cloud2.create_cloud_xyz32(header, points)


def main(args=None):
    """Main function to run ROS2 node."""
    rclpy.init(args=args)
    node = DualArmROS2Bridge()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

