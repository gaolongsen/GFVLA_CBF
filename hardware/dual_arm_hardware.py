"""
Dual-arm hardware interface for GF-VLA.

Main interface for controlling dual-arm robotic system:
- UR5e with Robotiq Gripper (left arm)
- UR10e with Barrett BH282 Hand (right arm)

Action Space: 14 dimensions
- [0:6]: UR5e arm pose (6 DOF: x, y, z, roll, pitch, yaw)
- [6]: Robotiq gripper (0=closed, 1=open)
- [7:13]: UR10e arm pose (6 DOF: x, y, z, roll, pitch, yaw)
- [13]: Barrett Hand gripper (0=closed, 1=open)
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
import logging
from scipy.spatial.transform import Rotation as R

from .ur_robot_interface import UR5eInterface, UR10eInterface
from .robotiq_gripper import RobotiqGripper
from .barrett_hand import BarrettHand
from .safety import SafetyMonitor, SafetyLimits
from .config import HardwareConfig
from .cbf import ControlBarrierFunctionFilter, CBFConfig, Obstacle

logger = logging.getLogger(__name__)


class DualArmHardwareInterface:
    """
    Main interface for dual-arm robotic hardware control.
    
    Hardware Configuration (matching GF-VLA paper):
    - Left Arm: UR5e with Robotiq Gripper
    - Right Arm: UR10e with Barrett BH282 Hand
    
    Action Space: 14 dimensions
    - [0:6]: UR5e arm pose (6 DOF: x, y, z, roll, pitch, yaw)
    - [6]: Robotiq gripper (0=closed, 1=open)
    - [7:13]: UR10e arm pose (6 DOF: x, y, z, roll, pitch, yaw)
    - [13]: Barrett Hand gripper (0=closed, 1=open)
    """
    
    def __init__(self, config: Optional[HardwareConfig] = None):
        """
        Initialize dual-arm hardware interface.
        
        Args:
            config: Hardware configuration (uses defaults if None)
        """
        self.config = config if config is not None else HardwareConfig()
        
        # Initialize robot interfaces
        self.ur5e = UR5eInterface(
            ip_address=self.config.ur5e.ip_address,
            port=self.config.ur5e.port,
            speed=self.config.ur5e.speed,
            acceleration=self.config.ur5e.acceleration,
        )
        
        self.ur10e = UR10eInterface(
            ip_address=self.config.ur10e.ip_address,
            port=self.config.ur10e.port,
            speed=self.config.ur10e.speed,
            acceleration=self.config.ur10e.acceleration,
        )
        
        # Initialize gripper interfaces
        self.robotiq = RobotiqGripper(
            port=self.config.robotiq.port,
            baudrate=self.config.robotiq.baudrate,
            timeout=self.config.robotiq.timeout,
        )
        
        self.barrett = BarrettHand(
            ip_address=self.config.barrett.ip_address,
            port=self.config.barrett.port,
            timeout=self.config.barrett.timeout,
        )
        
        # Initialize safety monitor
        safety_limits = SafetyLimits(
            max_cartesian_velocity=self.config.safety.max_cartesian_velocity,
            max_cartesian_acceleration=self.config.safety.max_cartesian_acceleration,
            workspace_limits=self.config.safety.workspace_limits,
            collision_threshold=self.config.safety.collision_threshold,
        )
        self.safety = SafetyMonitor(limits=safety_limits)
        
        # Initialize CBF filter for obstacle avoidance (if enabled)
        self._use_cbf = getattr(self.config.safety, 'use_cbf_filter', True)
        if self._use_cbf:
            ws = self.config.safety.workspace_limits or {}
            left_ws = ws.get('left', {})
            right_ws = ws.get('right', {})
            cbf_config = CBFConfig(
                obstacle_safety_margin=getattr(self.config.safety, 'cbf_obstacle_margin', 0.08),
                inter_arm_safety_margin=getattr(self.config.safety, 'cbf_inter_arm_margin', 0.12),
                workspace_x=left_ws.get('x', (0.0, 0.8)),
                workspace_y=left_ws.get('y', (-0.5, 0.5)),
                workspace_z=left_ws.get('z', (0.0, 0.6)),
            )
            self.cbf_filter = ControlBarrierFunctionFilter(config=cbf_config)
        else:
            self.cbf_filter = None
        
        self.connected = False
    
    def connect(self) -> bool:
        """
        Connect to all hardware components.
        
        Returns:
            True if all connections successful
        """
        logger.info("Connecting to dual-arm hardware...")
        
        success = True
        
        # Connect robots
        if not self.ur5e.connect():
            logger.error("Failed to connect to UR5e")
            success = False
        
        if not self.ur10e.connect():
            logger.error("Failed to connect to UR10e")
            success = False
        
        # Connect grippers
        if not self.robotiq.connect():
            logger.error("Failed to connect to Robotiq gripper")
            success = False
        
        if not self.barrett.connect():
            logger.error("Failed to connect to Barrett Hand")
            success = False
        
        if success:
            self.connected = True
            logger.info("Successfully connected to all hardware components")
        else:
            logger.error("Some hardware components failed to connect")
            self.disconnect()  # Disconnect any successful connections
        
        return success
    
    def disconnect(self):
        """Disconnect from all hardware components."""
        logger.info("Disconnecting from dual-arm hardware...")
        
        self.ur5e.disconnect()
        self.ur10e.disconnect()
        self.robotiq.disconnect()
        self.barrett.disconnect()
        
        self.connected = False
        logger.info("Disconnected from all hardware components")
    
    def get_observation(self) -> Dict[str, Any]:
        """
        Get current observation from both arms.
        
        Returns:
            Dictionary with observation data:
            - 'robot_state': Combined robot state [14] (7 per arm)
            - 'left_arm_pose': UR5e pose [7] (position + quaternion)
            - 'right_arm_pose': UR10e pose [7] (position + quaternion)
            - 'left_gripper': Robotiq open amount [1]
            - 'right_gripper': Barrett open amount [1]
        """
        if not self.connected:
            raise RuntimeError("Hardware not connected")
        
        # Get arm poses
        left_pos, left_rot = self.ur5e.get_current_pose()
        right_pos, right_rot = self.ur10e.get_current_pose()
        
        # Convert rotation from axis-angle to quaternion
        left_r = R.from_rotvec(left_rot)
        left_quat = left_r.as_quat()  # [x, y, z, w]
        
        right_r = R.from_rotvec(right_rot)
        right_quat = right_r.as_quat()  # [x, y, z, w]
        
        # Get gripper states
        left_gripper = self.robotiq.get_open_amount()
        right_gripper = self.barrett.get_open_amount()
        
        # Construct observation
        left_pose = np.concatenate([left_pos, left_quat])  # [7]
        right_pose = np.concatenate([right_pos, right_quat])  # [7]
        
        robot_state = np.concatenate([
            left_pose,  # [7]
            [left_gripper],  # [1]
            right_pose,  # [7]
            [right_gripper],  # [1]
        ])  # [16] total
        
        return {
            'robot_state': robot_state,
            'left_arm_pose': left_pose,
            'right_arm_pose': right_pose,
            'left_gripper': np.array([left_gripper]),
            'right_gripper': np.array([right_gripper]),
        }
    
    def set_cbf_obstacles(self, obstacles: list):
        """
        Update obstacles for CBF filter (e.g., from vision/point cloud).
        
        Args:
            obstacles: List of Obstacle objects
        """
        if self.cbf_filter is not None:
            self.cbf_filter.set_obstacles(obstacles)
    
    def execute_action(
        self,
        action: np.ndarray,
        blocking: bool = True,
        check_safety: bool = True,
        use_cbf: Optional[bool] = None,
        obstacles: Optional[list] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Execute dual-arm action.
        
        Args:
            action: Action array [14]
                - [0:6]: UR5e arm pose (6 DOF: x, y, z, roll, pitch, yaw)
                - [6]: Robotiq gripper (0=closed, 1=open)
                - [7:13]: UR10e arm pose (6 DOF: x, y, z, roll, pitch, yaw)
                - [13]: Barrett Hand gripper (0=closed, 1=open)
            blocking: Whether to wait for movement completion
            check_safety: Whether to perform safety checks
            use_cbf: Whether to apply CBF filter (default: config setting)
            obstacles: Optional obstacles for CBF (updates filter if provided)
        
        Returns:
            Tuple of (success, error_message)
        """
        if not self.connected:
            return False, "Hardware not connected"
        
        if len(action) != 14:
            return False, f"Action must be 14-dimensional, got {len(action)}"
        
        # Check emergency stop
        if self.safety.emergency_stop:
            return False, "Emergency stop is active"
        
        # Apply CBF filter for obstacle avoidance
        apply_cbf = use_cbf if use_cbf is not None else self._use_cbf
        if apply_cbf and self.cbf_filter is not None:
            if obstacles is not None:
                self.cbf_filter.set_obstacles(obstacles)
            try:
                obs = self.get_observation() if self.connected else {}
                current_state = {
                    'left_arm_pose': obs.get('left_arm_pose'),
                    'right_arm_pose': obs.get('right_arm_pose'),
                }
            except Exception:
                current_state = None
            action, cbf_ok, cbf_err = self.cbf_filter.filter_action(action, current_state)
            if not cbf_ok and cbf_err:
                logger.warning(f"CBF filter: {cbf_err}")
        
        # Split action
        action_left = action[:7]  # [x, y, z, roll, pitch, yaw, gripper]
        action_right = action[7:14]  # [x, y, z, roll, pitch, yaw, gripper]
        
        left_position = action_left[:3]
        left_rotation = action_left[3:6]  # Euler angles (roll, pitch, yaw)
        left_gripper = action_left[6]
        
        right_position = action_right[:3]
        right_rotation = action_right[3:6]  # Euler angles (roll, pitch, yaw)
        right_gripper = action_right[6]
        
        # Safety checks
        if check_safety:
            safe, error = self.safety.check_action(
                left_position=left_position,
                right_position=right_position,
            )
            if not safe:
                return False, error
        
        try:
            # Convert Euler angles to axis-angle for UR robots
            left_r = R.from_euler('xyz', left_rotation, degrees=False)
            left_rotvec = left_r.as_rotvec()
            
            right_r = R.from_euler('xyz', right_rotation, degrees=False)
            right_rotvec = right_r.as_rotvec()
            
            # Execute arm movements (non-blocking for parallel execution)
            left_success = self.ur5e.move_to_pose(
                position=left_position,
                rotation=left_rotvec,
                blocking=False,  # Non-blocking for parallel execution
            )
            
            right_success = self.ur10e.move_to_pose(
                position=right_position,
                rotation=right_rotvec,
                blocking=False,  # Non-blocking for parallel execution
            )
            
            # Execute gripper movements
            robotiq_success = self.robotiq.set_open_amount(left_gripper)
            barrett_success = self.barrett.set_open_amount(right_gripper)
            
            # Wait for completion if blocking
            if blocking:
                # Wait for both arms to complete
                # Note: This is simplified - in practice, you'd poll status
                import time
                time.sleep(0.5)  # Approximate wait time
            
            if not (left_success and right_success and robotiq_success and barrett_success):
                return False, "One or more hardware commands failed"
            
            return True, None
            
        except Exception as e:
            logger.error(f"Failed to execute action: {e}")
            return False, str(e)
    
    def reset(self):
        """Reset both arms to home position."""
        if not self.connected:
            raise RuntimeError("Hardware not connected")
        
        logger.info("Resetting dual-arm system to home position...")
        
        # Move to safe home positions
        # Left arm (UR5e) home position
        left_home_pos = np.array([0.4, 0.0, 0.3])  # Adjust based on setup
        left_home_rot = np.array([0.0, 0.0, 0.0])  # Axis-angle
        
        # Right arm (UR10e) home position
        right_home_pos = np.array([0.4, 0.0, 0.3])  # Adjust based on setup
        right_home_rot = np.array([0.0, 0.0, 0.0])  # Axis-angle
        
        # Move arms
        self.ur5e.move_to_pose(left_home_pos, left_home_rot, blocking=True)
        self.ur10e.move_to_pose(right_home_pos, right_home_rot, blocking=True)
        
        # Open grippers
        self.robotiq.open()
        self.barrett.open()
        
        # Reset safety monitor
        self.safety.reset()
        
        logger.info("Dual-arm system reset complete")
    
    def emergency_stop(self):
        """Emergency stop all hardware."""
        logger.warning("EMERGENCY STOP ACTIVATED")
        
        self.safety.set_emergency_stop(True)
        
        # Stop robots
        self.ur5e.stop()
        self.ur10e.stop()
        
        # Note: Grippers typically maintain position on emergency stop
    
    def clear_emergency_stop(self):
        """Clear emergency stop state."""
        self.safety.set_emergency_stop(False)
        logger.info("Emergency stop cleared")

