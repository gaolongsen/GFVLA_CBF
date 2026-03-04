"""
Safety monitoring and collision detection for dual-arm system.
"""

import numpy as np
from typing import Optional, Tuple, Dict
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SafetyLimits:
    """Safety limits for robot motion."""
    max_cartesian_velocity: float = 0.5  # m/s
    max_cartesian_acceleration: float = 2.0  # m/s²
    max_joint_velocity: Optional[np.ndarray] = None
    workspace_limits: Optional[Dict[str, Tuple[float, float]]] = None
    collision_threshold: float = 0.05  # m (minimum distance between arms)


class SafetyMonitor:
    """Safety monitor for dual-arm robotic system."""
    
    def __init__(self, limits: Optional[SafetyLimits] = None):
        """
        Initialize safety monitor.
        
        Args:
            limits: Safety limits configuration
        """
        self.limits = limits if limits is not None else SafetyLimits()
        self.last_positions = {
            'left': None,
            'right': None,
        }
        self.last_velocities = {
            'left': None,
            'right': None,
        }
        self.emergency_stop = False
    
    def check_cartesian_velocity(
        self,
        position: np.ndarray,
        arm: str,
        dt: float = 0.1,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if Cartesian velocity is within limits.
        
        Args:
            position: Current position [x, y, z]
            arm: Arm identifier ('left' or 'right')
            dt: Time step in seconds
        
        Returns:
            Tuple of (is_safe, error_message)
        """
        if self.last_positions[arm] is None:
            self.last_positions[arm] = position
            return True, None
        
        # Calculate velocity
        velocity = (position - self.last_positions[arm]) / dt
        velocity_magnitude = np.linalg.norm(velocity)
        
        # Check limit
        if velocity_magnitude > self.limits.max_cartesian_velocity:
            error_msg = (
                f"{arm} arm velocity {velocity_magnitude:.3f} m/s exceeds "
                f"limit {self.limits.max_cartesian_velocity:.3f} m/s"
            )
            logger.warning(error_msg)
            return False, error_msg
        
        # Update last position and velocity
        self.last_positions[arm] = position
        self.last_velocities[arm] = velocity
        
        return True, None
    
    def check_cartesian_acceleration(
        self,
        position: np.ndarray,
        arm: str,
        dt: float = 0.1,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if Cartesian acceleration is within limits.
        
        Args:
            position: Current position [x, y, z]
            arm: Arm identifier ('left' or 'right')
            dt: Time step in seconds
        
        Returns:
            Tuple of (is_safe, error_message)
        """
        if self.last_velocities[arm] is None:
            return True, None
        
        # Calculate velocity
        velocity = (position - self.last_positions[arm]) / dt
        
        # Calculate acceleration
        acceleration = (velocity - self.last_velocities[arm]) / dt
        acceleration_magnitude = np.linalg.norm(acceleration)
        
        # Check limit
        if acceleration_magnitude > self.limits.max_cartesian_acceleration:
            error_msg = (
                f"{arm} arm acceleration {acceleration_magnitude:.3f} m/s² exceeds "
                f"limit {self.limits.max_cartesian_acceleration:.3f} m/s²"
            )
            logger.warning(error_msg)
            return False, error_msg
        
        return True, None
    
    def check_workspace_limits(
        self,
        position: np.ndarray,
        arm: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if position is within workspace limits.
        
        Args:
            position: Current position [x, y, z]
            arm: Arm identifier ('left' or 'right')
        
        Returns:
            Tuple of (is_safe, error_message)
        """
        if self.limits.workspace_limits is None:
            return True, None
        
        limits = self.limits.workspace_limits.get(arm)
        if limits is None:
            return True, None
        
        x_min, x_max = limits.get('x', (-np.inf, np.inf))
        y_min, y_max = limits.get('y', (-np.inf, np.inf))
        z_min, z_max = limits.get('z', (-np.inf, np.inf))
        
        x, y, z = position
        
        if not (x_min <= x <= x_max):
            error_msg = f"{arm} arm x position {x:.3f} outside limits [{x_min:.3f}, {x_max:.3f}]"
            logger.warning(error_msg)
            return False, error_msg
        
        if not (y_min <= y <= y_max):
            error_msg = f"{arm} arm y position {y:.3f} outside limits [{y_min:.3f}, {y_max:.3f}]"
            logger.warning(error_msg)
            return False, error_msg
        
        if not (z_min <= z <= z_max):
            error_msg = f"{arm} arm z position {z:.3f} outside limits [{z_min:.3f}, {z_max:.3f}]"
            logger.warning(error_msg)
            return False, error_msg
        
        return True, None
    
    def check_collision(
        self,
        left_position: np.ndarray,
        right_position: np.ndarray,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check for potential collision between arms.
        
        Args:
            left_position: Left arm position [x, y, z]
            right_position: Right arm position [x, y, z]
        
        Returns:
            Tuple of (is_safe, error_message)
        """
        distance = np.linalg.norm(left_position - right_position)
        
        if distance < self.limits.collision_threshold:
            error_msg = (
                f"Collision risk: arms too close "
                f"({distance:.3f} m < {self.limits.collision_threshold:.3f} m)"
            )
            logger.warning(error_msg)
            return False, error_msg
        
        return True, None
    
    def check_action(
        self,
        left_position: np.ndarray,
        right_position: np.ndarray,
        dt: float = 0.1,
    ) -> Tuple[bool, Optional[str]]:
        """
        Perform all safety checks for a dual-arm action.
        
        Args:
            left_position: Left arm target position [x, y, z]
            right_position: Right arm target position [x, y, z]
            dt: Time step in seconds
        
        Returns:
            Tuple of (is_safe, error_message)
        """
        # Check workspace limits
        safe, error = self.check_workspace_limits(left_position, 'left')
        if not safe:
            return False, error
        
        safe, error = self.check_workspace_limits(right_position, 'right')
        if not safe:
            return False, error
        
        # Check velocities
        safe, error = self.check_cartesian_velocity(left_position, 'left', dt)
        if not safe:
            return False, error
        
        safe, error = self.check_cartesian_velocity(right_position, 'right', dt)
        if not safe:
            return False, error
        
        # Check accelerations
        safe, error = self.check_cartesian_acceleration(left_position, 'left', dt)
        if not safe:
            return False, error
        
        safe, error = self.check_cartesian_acceleration(right_position, 'right', dt)
        if not safe:
            return False, error
        
        # Check collision
        safe, error = self.check_collision(left_position, right_position)
        if not safe:
            return False, error
        
        return True, None
    
    def reset(self):
        """Reset safety monitor state."""
        self.last_positions = {'left': None, 'right': None}
        self.last_velocities = {'left': None, 'right': None}
        self.emergency_stop = False
    
    def set_emergency_stop(self, stop: bool = True):
        """Set emergency stop state."""
        self.emergency_stop = stop
        if stop:
            logger.warning("EMERGENCY STOP ACTIVATED")

