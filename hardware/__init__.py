"""
Hardware interface for dual-arm robotic manipulation.

This module provides interfaces for controlling real physical robots:
- UR5e with Robotiq Gripper (left arm)
- UR10e with Barrett BH282 Hand (right arm)

Based on GF-VLA framework and paper: Graph-Fused Vision-Language-Action for 
Policy Reasoning in Multi-Arm Robotic Manipulation (arXiv:2509.07957v1)
"""

from .dual_arm_hardware import DualArmHardwareInterface
from .ur_robot_interface import UR5eInterface, UR10eInterface
from .robotiq_gripper import RobotiqGripper
from .barrett_hand import BarrettHand
from .safety import SafetyMonitor
from .config import HardwareConfig
from .vision import BlockDetector, CameraInterface, BlockPose
from .vla_integration import VLAHardwareController
from .cbf import (
    ControlBarrierFunctionFilter,
    CBFConfig,
    Obstacle,
    obstacles_from_point_cloud,
    obstacles_from_blocks,
)

__all__ = [
    'DualArmHardwareInterface',
    'UR5eInterface',
    'UR10eInterface',
    'RobotiqGripper',
    'BarrettHand',
    'SafetyMonitor',
    'HardwareConfig',
    'BlockDetector',
    'CameraInterface',
    'BlockPose',
    'VLAHardwareController',
    'ControlBarrierFunctionFilter',
    'CBFConfig',
    'Obstacle',
    'obstacles_from_point_cloud',
    'obstacles_from_blocks',
]

