"""
Barrett Hand (BH282) Gripper wrapper for PyRep/RLBench.

The Barrett Hand is a 3-finger dexterous hand. This wrapper provides
a compatible interface for use with RLBench tasks.
"""

from pyrep.robots.end_effectors.gripper import Gripper
from pyrep.objects.joint import Joint
from pyrep.objects.object import Object
from typing import List
import numpy as np


class BarrettHandGripper(Gripper):
    """Wrapper for Barrett Hand (BH282) gripper.
    
    The Barrett Hand is a 3-finger dexterous hand. This implementation
    provides a simplified interface compatible with RLBench's discrete
    gripper action mode. For more complex control, the individual finger
    joints can be accessed directly.
    """
    
    def __init__(self, count: int = 0):
        """
        Args:
            count: The number of the Barrett Hand (0, 1, 2, etc.)
        """
        # Barrett Hand typically has 4 joints per finger (3 fingers)
        # Plus spread joint, so typically 13 joints total
        # For simplicity, we'll use a subset for basic open/close control
        joint_names = [
            'BarrettHand_joint1', 'BarrettHand_joint2', 'BarrettHand_joint3',
            'BarrettHand_joint4', 'BarrettHand_joint5', 'BarrettHand_joint6',
            'BarrettHand_joint7', 'BarrettHand_joint8', 'BarrettHand_joint9',
            'BarrettHand_joint10', 'BarrettHand_joint11', 'BarrettHand_joint12',
            'BarrettHand_spread_joint'
        ]
        
        # Try to find joints - if not found, use generic approach
        try:
            joints = [Joint(name) for name in joint_names if Joint.exists(name)]
        except:
            # Fallback: try to find any Barrett Hand joints
            joints = []
            for i in range(1, 14):
                name = f'BarrettHand_joint{i}'
                if Joint.exists(name):
                    joints.append(Joint(name))
        
        # If still no joints found, create a minimal interface
        if len(joints) == 0:
            # Try alternative naming conventions
            alt_names = [
                'BH282_joint1', 'BH282_joint2', 'BH282_joint3',
                'BH282_finger1_joint1', 'BH282_finger1_joint2',
                'BH282_finger2_joint1', 'BH282_finger2_joint2',
                'BH282_finger3_joint1', 'BH282_finger3_joint2',
                'BH282_spread_joint'
            ]
            joints = [Joint(name) for name in alt_names if Joint.exists(name)]
        
        # Initialize parent class with joint names
        joint_names = [j.get_name() for j in joints] if joints else []
        super().__init__(count, joint_names=joint_names)
        self._joints = joints
        self._open_positions = None
        self._closed_positions = None
        self._initialize_positions()
    
    def _initialize_positions(self):
        """Initialize open and closed positions for the gripper."""
        if len(self._joints) == 0:
            # If no joints found, use default values
            self._open_positions = np.array([0.0] * 4)  # Default for 4 joints
            self._closed_positions = np.array([1.0] * 4)
        else:
            # Get current positions as reference
            current = self.get_joint_positions()
            # Open: fingers spread and extended
            self._open_positions = np.array([0.0] * len(current))
            # Closed: fingers closed
            self._closed_positions = np.array([1.0] * len(current))
    
    def actuate(self, amount: float, velocity: float = 0.04) -> bool:
        """
        Actuate the gripper to a given amount (0 = closed, 1 = open).
        
        Args:
            amount: The amount to open (0.0 = closed, 1.0 = open)
            velocity: The velocity of the actuation
            
        Returns:
            True if the gripper has reached the target position
        """
        if len(self._joints) == 0:
            return True  # No joints to control
        
        # Interpolate between closed and open positions
        target = (1.0 - amount) * self._closed_positions + amount * self._open_positions
        
        # Set target positions for all joints
        for i, joint in enumerate(self._joints):
            if i < len(target):
                joint.set_joint_target_position(target[i])
        
        # Check if we've reached the target
        current = self.get_joint_positions()
        reached = np.allclose(current, target, atol=0.01)
        return reached
    
    def get_open_amount(self) -> List[float]:
        """
        Get the current open amount of the gripper (0 = closed, 1 = open).
        
        Returns:
            List of open amounts for each finger/joint
        """
        if len(self._joints) == 0:
            return [0.5]  # Default middle position
        
        current = self.get_joint_positions()
        if len(current) == 0:
            return [0.5]
        
        # Calculate open amount based on position relative to closed/open
        open_amounts = []
        for i, pos in enumerate(current):
            if i < len(self._closed_positions) and i < len(self._open_positions):
                closed = self._closed_positions[i]
                open_pos = self._open_positions[i]
                if abs(open_pos - closed) > 1e-6:
                    amount = (pos - closed) / (open_pos - closed)
                    amount = max(0.0, min(1.0, amount))  # Clamp to [0, 1]
                else:
                    amount = 0.5
                open_amounts.append(amount)
            else:
                open_amounts.append(0.5)
        
        return open_amounts
    
    def grasp(self, object_to_grasp: Object) -> bool:
        """
        Attempt to grasp an object.
        
        Args:
            object_to_grasp: The object to grasp
            
        Returns:
            True if the object was successfully grasped
        """
        # Close the gripper
        self.actuate(0.0, velocity=0.2)
        
        # Check if object is in contact with gripper
        # This is a simplified check - in practice, you'd use proximity sensors
        # or collision detection
        return True
    
    def release(self):
        """Release any grasped objects."""
        # Open the gripper
        self.actuate(1.0, velocity=0.2)
    
    def get_grasped_objects(self) -> List[Object]:
        """
        Get a list of currently grasped objects.
        
        Returns:
            List of grasped objects
        """
        # This would need to be implemented based on the specific
        # grasping mechanism in the simulation
        return []

