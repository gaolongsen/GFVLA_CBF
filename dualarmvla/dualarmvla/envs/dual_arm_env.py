"""
dual_arm_env.py

Dual-arm environment configuration for GF-VLA.
Implements dual-arm setup with UR5e (Robotiq) and UR10e (Barrett BH282).
"""

from typing import List, Optional, Tuple, Union
import numpy as np
from rlbench import Environment
from rlbench.action_modes.action_mode import ActionMode, MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import (
    EndEffectorPoseViaIK,
    EndEffectorPoseViaPlanning,
)
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import ObservationConfig

from gfvla_env.helpers.common import Logger


class DualArmRLBenchEnv:
    """
    Dual-arm RLBench environment for GF-VLA.
    
    Hardware Configuration (matching paper):
    - Left Arm: UR5e with Robotiq Gripper
    - Right Arm: UR10e with Barrett BH282 Hand
    
    Action Space: 14 dimensions
    - [0:6]: UR5e arm pose (6 DOF: x, y, z, roll, pitch, yaw)
    - [6]: Robotiq gripper (0=closed, 1=open)
    - [7:13]: UR10e arm pose (6 DOF: x, y, z, roll, pitch, yaw)
    - [13]: Barrett Hand gripper (0=closed, 1=open)
    """
    
    def __init__(
        self,
        task_name: str,
        action_mode: Optional[ActionMode] = None,
        obs_config: Optional[ObservationConfig] = None,
        headless: bool = False,
        robot_setup_left: str = 'ur5e',  # UR5e with Robotiq
        robot_setup_right: str = 'ur10e',  # UR10e with Barrett Hand
    ):
        """
        Initialize dual-arm environment.
        
        Args:
            task_name: Name of the RLBench task
            action_mode: Action mode (defaults to EEPose + Gripper)
            obs_config: Observation configuration
            headless: Run in headless mode
            robot_setup_left: Robot configuration for left arm (default: 'ur5e')
            robot_setup_right: Robot configuration for right arm (default: 'ur10e')
        """
        self.task_name = task_name
        self.robot_setup_left = robot_setup_left
        self.robot_setup_right = robot_setup_right
        
        # Default action mode: End-effector pose + discrete gripper
        if action_mode is None:
            action_mode = MoveArmThenGripper(
                arm_action_mode=EndEffectorPoseViaPlanning(absolute_mode=True),
                gripper_action_mode=Discrete(),
            )
        
        self.action_mode = action_mode
        
        # Default observation config: front camera
        if obs_config is None:
            obs_config = ObservationConfig()
            obs_config.set_all_low_dim(True)
            obs_config.set_all_high_dim(False)
            obs_config.front_camera.set_all(True)
            obs_config.front_camera.image_size = (224, 224)
        
        self.obs_config = obs_config
        
        # Note: RLBench currently supports single-arm setups
        # For dual-arm, we'll need to extend RLBench or use a custom implementation
        # For now, we'll create two separate environments and coordinate them
        Logger.log_warning(
            "Dual-arm environment: Using single-arm RLBench with coordination. "
            "Full dual-arm support requires custom RLBench extension."
        )
        
        # Create left arm environment (UR5e)
        self.env_left = Environment(
            action_mode=action_mode,
            obs_config=obs_config,
            headless=headless,
            robot_setup=robot_setup_left,
        )
        
        # Create right arm environment (UR10e)
        # Note: This is a placeholder - actual dual-arm requires scene modification
        self.env_right = None  # Would need custom dual-arm scene
        
        self.action_dim = 14  # 2 × (6 DOF + 1 gripper)
        
    def launch(self):
        """Launch the environment."""
        self.env_left.launch()
        Logger.log_info(
            f"Dual-arm environment launched: "
            f"Left={self.robot_setup_left}, Right={self.robot_setup_right}"
        )
    
    def shutdown(self):
        """Shutdown the environment."""
        if self.env_left is not None:
            self.env_left.shutdown()
        if self.env_right is not None:
            self.env_right.shutdown()
    
    def get_task(self, task_class):
        """Get task instance."""
        # For now, return left arm task
        # Full dual-arm support requires custom task implementation
        return self.env_left.get_task(task_class)
    
    def reset(self):
        """Reset the environment."""
        # Reset both arms
        obs_left = self.env_left.reset()
        
        # For now, return left arm observations
        # Full dual-arm would combine observations from both arms
        return obs_left
    
    def step(self, action: np.ndarray):
        """
        Step the environment with dual-arm action.
        
        Args:
            action: [14] array
                - [0:7]: Left arm (UR5e) action [x, y, z, roll, pitch, yaw, gripper]
                - [7:14]: Right arm (UR10e) action [x, y, z, roll, pitch, yaw, gripper]
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        if len(action) != 14:
            raise ValueError(
                f"Dual-arm action must be 14-dimensional, got {len(action)}"
            )
        
        # Split action into left and right
        action_left = action[:7]  # UR5e + Robotiq
        action_right = action[7:14]  # UR10e + Barrett
        
        # Execute left arm action
        obs_left, reward_left, terminated_left, truncated_left, info_left = \
            self.env_left.step(action_left)
        
        # For now, return left arm results
        # Full dual-arm would coordinate both arms and combine results
        return obs_left, reward_left, terminated_left, truncated_left, info_left
    
    @property
    def action_space(self):
        """Get action space."""
        from gymnasium.spaces import Box
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(14,),
            dtype=np.float32
        )
    
    @property
    def observation_space(self):
        """Get observation space."""
        return self.env_left.observation_space


def create_dual_arm_env(
    task_name: str,
    headless: bool = False,
    robot_setup_left: str = 'ur5e',
    robot_setup_right: str = 'ur10e',
) -> DualArmRLBenchEnv:
    """
    Create a dual-arm environment for GF-VLA.
    
    Args:
        task_name: Name of the RLBench task
        headless: Run in headless mode
        robot_setup_left: Left arm robot setup (default: 'ur5e' with Robotiq)
        robot_setup_right: Right arm robot setup (default: 'ur10e' with Barrett)
    
    Returns:
        DualArmRLBenchEnv instance
    """
    env = DualArmRLBenchEnv(
        task_name=task_name,
        headless=headless,
        robot_setup_left=robot_setup_left,
        robot_setup_right=robot_setup_right,
    )
    return env

