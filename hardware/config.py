"""
Hardware configuration for dual-arm robotic system.

Configuration matching GF-VLA paper specifications:
- Left Arm: UR5e with Robotiq Gripper
- Right Arm: UR10e with Barrett BH282 Hand
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class UR5eConfig:
    """Configuration for UR5e robot (left arm)."""
    ip_address: str = "192.168.1.101"  # Default IP for UR5e
    port: int = 30002  # Real-time data port
    speed: float = 0.5  # Default speed (0-1)
    acceleration: float = 0.5  # Default acceleration (0-1)
    base_frame: Optional[np.ndarray] = None  # Base transformation matrix
    joint_limits: Optional[np.ndarray] = None  # Joint limits (radians)
    workspace_limits: Optional[dict] = None  # Cartesian workspace limits


@dataclass
class UR10eConfig:
    """Configuration for UR10e robot (right arm)."""
    ip_address: str = "192.168.1.102"  # Default IP for UR10e
    port: int = 30002  # Real-time data port
    speed: float = 0.5  # Default speed (0-1)
    acceleration: float = 0.5  # Default acceleration (0-1)
    base_frame: Optional[np.ndarray] = None  # Base transformation matrix
    joint_limits: Optional[np.ndarray] = None  # Joint limits (radians)
    workspace_limits: Optional[dict] = None  # Cartesian workspace limits


@dataclass
class RobotiqGripperConfig:
    """Configuration for Robotiq Gripper."""
    port: str = "/dev/ttyUSB0"  # Serial port (Linux) or COM port (Windows)
    baudrate: int = 115200
    timeout: float = 1.0
    min_position: float = 0.0  # Closed position (mm)
    max_position: float = 85.0  # Open position (mm)
    speed: int = 255  # Speed (0-255)
    force: int = 255  # Force (0-255)


@dataclass
class BarrettHandConfig:
    """Configuration for Barrett BH282 Hand."""
    ip_address: str = "192.168.1.103"  # Ethernet IP for Barrett Hand
    port: int = 10000
    timeout: float = 1.0
    finger_positions: dict = None  # Finger position limits
    spread_limits: tuple = (0.0, 2.0)  # Spread joint limits (radians)


@dataclass
class SafetyConfig:
    """Safety configuration for dual-arm system."""
    max_cartesian_velocity: float = 0.5  # m/s
    max_cartesian_acceleration: float = 2.0  # m/s²
    collision_detection: bool = True
    workspace_limits: dict = None
    emergency_stop_enabled: bool = True
    joint_velocity_limits: Optional[np.ndarray] = None
    collision_threshold: float = 0.05  # m (minimum distance between arms)
    # CBF (Control Barrier Function) settings for obstacle avoidance
    use_cbf_filter: bool = True
    cbf_obstacle_margin: float = 0.08  # m
    cbf_inter_arm_margin: float = 0.12  # m


@dataclass
class HardwareConfig:
    """Complete hardware configuration for dual-arm system."""
    ur5e: UR5eConfig = None
    ur10e: UR10eConfig = None
    robotiq: RobotiqGripperConfig = None
    barrett: BarrettHandConfig = None
    safety: SafetyConfig = None
    
    def __post_init__(self):
        """Initialize default configurations if not provided."""
        if self.ur5e is None:
            self.ur5e = UR5eConfig()
        if self.ur10e is None:
            self.ur10e = UR10eConfig()
        if self.robotiq is None:
            self.robotiq = RobotiqGripperConfig()
        if self.barrett is None:
            self.barrett = BarrettHandConfig()
        if self.safety is None:
            self.safety = SafetyConfig()
        # Set default workspace limits if not provided
        if self.safety.workspace_limits is None:
            self.safety.workspace_limits = {
                'left': {'x': (0.0, 0.8), 'y': (-0.5, 0.5), 'z': (0.0, 0.6)},
                'right': {'x': (0.0, 0.8), 'y': (-0.5, 0.5), 'z': (0.0, 0.6)},
            }

