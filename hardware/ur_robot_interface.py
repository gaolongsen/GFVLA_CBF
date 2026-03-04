"""
Universal Robots (UR5e/UR10e) interface for real hardware control.

This module provides low-level control interfaces for Universal Robots
using the RTDE (Real-Time Data Exchange) protocol or URX library.
"""

import numpy as np
from typing import Optional, Tuple, List
import time
import logging
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)


class URRobotInterface:
    """Base class for Universal Robots interface."""
    
    def __init__(
        self,
        ip_address: str,
        port: int = 30002,
        speed: float = 0.5,
        acceleration: float = 0.5,
    ):
        """
        Initialize UR robot interface.
        
        Args:
            ip_address: Robot IP address
            port: RTDE port (default: 30002)
            speed: Default speed (0-1)
            acceleration: Default acceleration (0-1)
        """
        self.ip_address = ip_address
        self.port = port
        self.speed = speed
        self.acceleration = acceleration
        self.connected = False
        self._rtde_control = None
        self._rtde_receive = None
        
    def connect(self) -> bool:
        """
        Connect to the robot.
        
        Returns:
            True if connection successful
        """
        try:
            # Try to import ur_rtde (preferred method)
            try:
                import rtde_control
                import rtde_receive
                
                self._rtde_control = rtde_control.RTDEControlInterface(self.ip_address)
                self._rtde_receive = rtde_receive.RTDEReceiveInterface(self.ip_address)
                self.connected = True
                logger.info(f"Connected to UR robot at {self.ip_address} via RTDE")
                return True
            except ImportError:
                # Fallback to urx library
                try:
                    import urx
                    self._robot = urx.Robot(self.ip_address)
                    self.connected = True
                    logger.info(f"Connected to UR robot at {self.ip_address} via urx")
                    return True
                except ImportError:
                    logger.error(
                        "Neither ur_rtde nor urx library found. "
                        "Please install: pip install ur-rtde or pip install urx"
                    )
                    return False
        except Exception as e:
            logger.error(f"Failed to connect to robot: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from the robot."""
        if self._rtde_control is not None:
            self._rtde_control.disconnect()
        if self._rtde_receive is not None:
            self._rtde_receive.disconnect()
        if hasattr(self, '_robot') and self._robot is not None:
            self._robot.close()
        self.connected = False
        logger.info("Disconnected from UR robot")
    
    def get_current_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current end-effector pose.
        
        Returns:
            Tuple of (position [x, y, z], rotation [rx, ry, rz] in axis-angle)
        """
        if not self.connected:
            raise RuntimeError("Robot not connected")
        
        try:
            if self._rtde_receive is not None:
                # RTDE method
                pose = self._rtde_receive.getActualTCPPose()
                position = np.array(pose[:3])
                rotation = np.array(pose[3:])  # Axis-angle representation
                return position, rotation
            elif hasattr(self, '_robot'):
                # urx method
                pose = self._robot.get_pose()
                position = np.array([pose.pos.x, pose.pos.y, pose.pos.z])
                # Convert orientation to axis-angle
                rotation_matrix = np.array(pose.orient.array).reshape(3, 3)
                r = R.from_matrix(rotation_matrix)
                rotation = r.as_rotvec()
                return position, rotation
        except Exception as e:
            logger.error(f"Failed to get current pose: {e}")
            raise
    
    def get_current_joints(self) -> np.ndarray:
        """
        Get current joint positions.
        
        Returns:
            Joint positions in radians [6]
        """
        if not self.connected:
            raise RuntimeError("Robot not connected")
        
        try:
            if self._rtde_receive is not None:
                joints = self._rtde_receive.getActualQ()
                return np.array(joints)
            elif hasattr(self, '_robot'):
                joints = self._robot.get_joint_positions()
                return np.array(joints)
        except Exception as e:
            logger.error(f"Failed to get current joints: {e}")
            raise
    
    def move_to_pose(
        self,
        position: np.ndarray,
        rotation: np.ndarray,
        speed: Optional[float] = None,
        acceleration: Optional[float] = None,
        blocking: bool = True,
    ) -> bool:
        """
        Move end-effector to target pose.
        
        Args:
            position: Target position [x, y, z] in meters
            rotation: Target rotation [rx, ry, rz] in axis-angle (radians)
            speed: Movement speed (0-1), uses default if None
            acceleration: Movement acceleration (0-1), uses default if None
            blocking: Whether to wait for movement completion
        
        Returns:
            True if movement successful
        """
        if not self.connected:
            raise RuntimeError("Robot not connected")
        
        speed = speed if speed is not None else self.speed
        acceleration = acceleration if acceleration is not None else self.acceleration
        
        try:
            if self._rtde_control is not None:
                # RTDE method
                pose = np.concatenate([position, rotation])
                success = self._rtde_control.moveL(
                    pose,
                    speed=speed,
                    acceleration=acceleration,
                    blocking=blocking,
                )
                return success
            elif hasattr(self, '_robot'):
                # urx method
                # Convert axis-angle to rotation matrix
                r = R.from_rotvec(rotation)
                rotation_matrix = r.as_matrix()
                
                # Create pose
                from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper
                pose = self._robot.create_pose(
                    position, rotation_matrix
                )
                
                self._robot.movel(pose, acc=acceleration, vel=speed, wait=blocking)
                return True
        except Exception as e:
            logger.error(f"Failed to move to pose: {e}")
            return False
    
    def move_to_joints(
        self,
        joints: np.ndarray,
        speed: Optional[float] = None,
        acceleration: Optional[float] = None,
        blocking: bool = True,
    ) -> bool:
        """
        Move to target joint positions.
        
        Args:
            joints: Target joint positions [6] in radians
            speed: Movement speed (0-1), uses default if None
            acceleration: Movement acceleration (0-1), uses default if None
            blocking: Whether to wait for movement completion
        
        Returns:
            True if movement successful
        """
        if not self.connected:
            raise RuntimeError("Robot not connected")
        
        speed = speed if speed is not None else self.speed
        acceleration = acceleration if acceleration is not None else self.acceleration
        
        try:
            if self._rtde_control is not None:
                success = self._rtde_control.moveJ(
                    joints.tolist(),
                    speed=speed,
                    acceleration=acceleration,
                    blocking=blocking,
                )
                return success
            elif hasattr(self, '_robot'):
                self._robot.movej(joints.tolist(), acc=acceleration, vel=speed, wait=blocking)
                return True
        except Exception as e:
            logger.error(f"Failed to move to joints: {e}")
            return False
    
    def stop(self):
        """Emergency stop the robot."""
        try:
            if self._rtde_control is not None:
                self._rtde_control.stopL()
                self._rtde_control.stopJ()
            elif hasattr(self, '_robot'):
                self._robot.stop()
        except Exception as e:
            logger.error(f"Failed to stop robot: {e}")


class UR5eInterface(URRobotInterface):
    """Interface for UR5e robot (left arm)."""
    
    def __init__(self, ip_address: str = "192.168.1.101", **kwargs):
        """Initialize UR5e interface."""
        super().__init__(ip_address, **kwargs)


class UR10eInterface(URRobotInterface):
    """Interface for UR10e robot (right arm)."""
    
    def __init__(self, ip_address: str = "192.168.1.102", **kwargs):
        """Initialize UR10e interface."""
        super().__init__(ip_address, **kwargs)

