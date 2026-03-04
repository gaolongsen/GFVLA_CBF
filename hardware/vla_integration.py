"""
Integration module connecting GF-VLA models to hardware interface.

This module bridges the gap between the trained VLA models and the actual
hardware, providing a closed-loop system for vision-language-action control.
"""

import numpy as np
import torch
from typing import Optional, Tuple, Dict, Any, List
import logging
from PIL import Image
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from hardware import DualArmHardwareInterface, CameraInterface, BlockDetector
from hardware.vision import BlockPose

logger = logging.getLogger(__name__)


class VLAHardwareController:
    """
    Controller that integrates VLA models with hardware interface.
    
    This class provides a closed-loop system:
    1. Captures observations (images, point clouds, robot state)
    2. Processes them for VLA model input
    3. Calls VLA model to predict actions
    4. Executes actions on hardware
    5. Repeats until task completion
    """
    
    def __init__(
        self,
        model,
        hardware: DualArmHardwareInterface,
        camera: Optional[CameraInterface] = None,
        use_pointcloud: bool = True,
        use_robot_state: bool = True,
        action_dim: int = 14,  # 14 for dual-arm, 7 for single-arm
        predict_mode: str = "diff+ar",
        cfg_scale: float = 1.5,
        ddim_steps: int = 10,
        slow_fast_ratio: int = 4,
    ):
        """
        Initialize VLA hardware controller.
        
        Args:
            model: Loaded VLA model (FiSvla or GFVLA)
            hardware: Dual-arm hardware interface
            camera: Camera interface for capturing images
            use_pointcloud: Whether to use point cloud observations
            use_robot_state: Whether to use current robot state
            action_dim: Action dimension (14 for dual-arm, 7 for single-arm)
            predict_mode: Prediction mode ('diff', 'ar', or 'diff+ar')
            cfg_scale: Classifier-free guidance scale
            ddim_steps: Number of DDIM steps
            slow_fast_ratio: Ratio between slow and fast image updates
        """
        self.model = model
        self.hardware = hardware
        self.camera = camera
        self.use_pointcloud = use_pointcloud
        self.use_robot_state = use_robot_state
        self.action_dim = action_dim
        self.predict_mode = predict_mode
        self.cfg_scale = cfg_scale
        self.ddim_steps = ddim_steps
        self.slow_fast_ratio = slow_fast_ratio
        
        # State tracking
        self.slow_cnt = 0
        self.slow_image = None
        self.slow_latent_embedding = None
        self.input_ids = None
        self.cur_robot_state = None
        
        # Check if model has predict_action method
        if not hasattr(self.model, 'predict_action'):
            logger.warning("Model does not have predict_action method. Using forward pass instead.")
            self.use_predict_action = False
        else:
            self.use_predict_action = True
    
    def capture_observation(self) -> Dict[str, Any]:
        """
        Capture current observation from hardware.
        
        Returns:
            Dictionary containing:
            - 'image': RGB image (PIL Image)
            - 'point_cloud': Point cloud (numpy array) [N, 3] or None
            - 'robot_state': Current robot state (numpy array) or None
        """
        obs = {}
        
        # Capture image
        if self.camera is not None:
            rgb_image, depth_image = self.camera.capture_frame()
            if rgb_image is not None:
                # Convert BGR to RGB and create PIL Image
                rgb_image_rgb = rgb_image[:, :, ::-1]  # BGR to RGB
                obs['image'] = Image.fromarray(rgb_image_rgb)
                
                # Generate point cloud from depth if needed
                if self.use_pointcloud and depth_image is not None:
                    obs['point_cloud'] = self._depth_to_pointcloud(
                        depth_image,
                        self.camera.get_camera_intrinsics()
                    )
                else:
                    obs['point_cloud'] = None
            else:
                obs['image'] = None
                obs['point_cloud'] = None
        else:
            obs['image'] = None
            obs['point_cloud'] = None
        
        # Get robot state
        if self.use_robot_state:
            obs['robot_state'] = self.hardware.get_observation().get('robot_state', None)
        else:
            obs['robot_state'] = None
        
        return obs
    
    def _depth_to_pointcloud(
        self,
        depth_image: np.ndarray,
        intrinsics: np.ndarray,
    ) -> np.ndarray:
        """
        Convert depth image to point cloud.
        
        Args:
            depth_image: Depth image [H, W] in meters
            intrinsics: Camera intrinsic matrix [3, 3]
        
        Returns:
            Point cloud [N, 3] in camera frame
        """
        height, width = depth_image.shape
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
        
        # Create pixel grid
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # Convert to 3D points
        x = (u - cx) * depth_image / fx
        y = (v - cy) * depth_image / fy
        z = depth_image
        
        # Stack and reshape
        points = np.stack([x, y, z], axis=-1)
        points = points.reshape(-1, 3)
        
        # Filter invalid points (zero depth or too far)
        valid_mask = (points[:, 2] > 0) & (points[:, 2] < 2.0)
        points = points[valid_mask]
        
        # Downsample if too many points (keep max 1024 points)
        if len(points) > 1024:
            indices = np.random.choice(len(points), 1024, replace=False)
            points = points[indices]
        
        return points
    
    def predict_action(
        self,
        instruction: str,
        observation: Dict[str, Any],
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """
        Predict action using VLA model.
        
        Args:
            instruction: Language instruction for the task
            observation: Current observation dictionary
        
        Returns:
            Tuple of (action, metadata)
            - action: Predicted action [action_dim] or [T, action_dim]
            - metadata: Optional metadata (e.g., predicted subgoals)
        """
        image = observation.get('image')
        point_cloud = observation.get('point_cloud')
        robot_state = observation.get('robot_state')
        
        if image is None:
            logger.error("No image available for prediction")
            return None, None
        
        # Update slow image periodically
        if self.slow_cnt % self.slow_fast_ratio == 0:
            self.slow_image = image
            if self.predict_mode == 'diff' and self.use_predict_action:
                # Get slow latent embedding
                try:
                    self.input_ids, self.slow_latent_embedding = self.model.slow_system_forward(
                        image_head_slow=self.slow_image,
                        instruction=instruction,
                        unnorm_key='rlbench',
                    )
                except Exception as e:
                    logger.warning(f"Failed to get slow latent embedding: {e}")
                    self.input_ids = None
                    self.slow_latent_embedding = None
        
        # Prepare point cloud
        if self.use_pointcloud:
            if point_cloud is not None and len(point_cloud) > 0:
                # Ensure point cloud is in correct format
                if point_cloud.shape[1] != 3:
                    logger.warning(f"Unexpected point cloud shape: {point_cloud.shape}")
                    point_cloud = None
            else:
                point_cloud = None
        
        # Prepare robot state
        if self.use_robot_state and robot_state is not None:
            # Convert robot state to format expected by model
            # Model expects [x, y, z, roll, pitch, yaw, gripper] for each arm
            if len(robot_state) >= 14:
                # Dual-arm state
                cur_robot_state = robot_state[:14]
            elif len(robot_state) >= 7:
                # Single-arm state, duplicate for dual-arm if needed
                if self.action_dim == 14:
                    cur_robot_state = np.concatenate([robot_state[:7], robot_state[:7]])
                else:
                    cur_robot_state = robot_state[:7]
            else:
                cur_robot_state = None
        else:
            cur_robot_state = None
        
        # Predict action using model
        try:
            if self.use_predict_action:
                # Use model's predict_action method
                if self.predict_mode == 'diff':
                    output = self.model.fast_system_forward(
                        image_head_fast=image,
                        point_cloud=point_cloud,
                        slow_latent_embedding=self.slow_latent_embedding,
                        input_ids=self.input_ids,
                        unnorm_key='rlbench',
                        cur_robot_state=cur_robot_state,
                        cfg_scale=self.cfg_scale,
                        use_ddim=True,
                        num_ddim_steps=self.ddim_steps,
                        action_dim=self.action_dim,
                        predict_mode=self.predict_mode,
                    )
                    action = output
                    metadata = None
                else:
                    output = self.model.predict_action(
                        image_head_slow=self.slow_image,
                        image_head_fast=image,
                        point_cloud=point_cloud,
                        instruction=instruction,
                        unnorm_key='rlbench',
                        cfg_scale=self.cfg_scale,
                        use_ddim=True,
                        num_ddim_steps=self.ddim_steps,
                        cur_robot_state=cur_robot_state,
                        action_dim=self.action_dim,
                        predict_mode=self.predict_mode,
                    )
                    
                    if self.predict_mode == 'ar':
                        action, metadata = output
                    elif self.predict_mode == 'diff+ar':
                        action_diff, action_ar, metadata = output
                        action = action_diff  # Use diffusion output
                    else:
                        action = output
                        metadata = None
            else:
                # Fallback: use forward pass (not recommended)
                logger.warning("Using forward pass instead of predict_action")
                action = None
                metadata = None
        except Exception as e:
            logger.error(f"Error during action prediction: {e}", exc_info=True)
            return None, None
        
        # Convert action to numpy if needed
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        
        # Handle action shape
        if action is not None:
            if len(action.shape) > 1:
                # Multiple actions predicted, take first one
                action = action[0]
            
            # Ensure action has correct dimension
            if len(action) != self.action_dim:
                logger.warning(
                    f"Action dimension mismatch: expected {self.action_dim}, got {len(action)}"
                )
                if len(action) == 7 and self.action_dim == 14:
                    # Duplicate single-arm action for dual-arm
                    action = np.concatenate([action, action])
                elif len(action) == 14 and self.action_dim == 7:
                    # Use only first arm
                    action = action[:7]
        
        self.slow_cnt += 1
        return action, metadata
    
    def execute_action(
        self,
        action: np.ndarray,
        observation: Dict[str, Any],
        apply_relative: bool = True,
    ) -> Tuple[bool, Optional[str]]:
        """
        Execute predicted action on hardware.
        
        Args:
            action: Predicted action [action_dim]
            observation: Current observation (for relative positioning)
            apply_relative: Whether to apply action relative to current state
        
        Returns:
            Tuple of (success, error_message)
        """
        if action is None:
            return False, "Action is None"
        
        # Apply relative positioning if needed
        if apply_relative and self.use_robot_state:
            robot_state = observation.get('robot_state')
            if robot_state is not None:
                if self.action_dim == 14:
                    # Dual-arm: apply relative positioning to both arms
                    if len(robot_state) >= 14:
                        # Left arm (UR5e)
                        action[:3] += robot_state[7:10] if len(robot_state) > 10 else robot_state[:3]
                        # Right arm (UR10e)
                        action[7:10] += robot_state[14:17] if len(robot_state) > 14 else robot_state[7:10]
                    elif len(robot_state) >= 7:
                        # Single-arm state, duplicate for both arms
                        action[:3] += robot_state[:3]
                        action[7:10] += robot_state[:3]
                else:
                    # Single-arm
                    if len(robot_state) >= 7:
                        action[:3] += robot_state[:3]
        
        # Execute action on hardware
        success, error = self.hardware.execute_action(action, blocking=True)
        
        return success, error
    
    def run_episode(
        self,
        instruction: str,
        max_steps: int = 50,
        success_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Run a complete episode using VLA model.
        
        Args:
            instruction: Language instruction for the task
            max_steps: Maximum number of steps
            success_callback: Optional callback to check for success
        
        Returns:
            Episode results dictionary
        """
        logger.info(f"Starting episode with instruction: '{instruction}'")
        
        # Reset hardware
        self.hardware.reset()
        self.slow_cnt = 0
        self.slow_image = None
        self.slow_latent_embedding = None
        self.input_ids = None
        
        results = {
            'success': False,
            'steps': 0,
            'actions': [],
            'observations': [],
        }
        
        for step in range(max_steps):
            logger.info(f"Step {step + 1}/{max_steps}")
            
            # Capture observation
            observation = self.capture_observation()
            results['observations'].append(observation)
            
            # Check for success
            if success_callback is not None:
                if success_callback(observation):
                    logger.info("Task completed successfully!")
                    results['success'] = True
                    break
            
            # Predict action
            action, metadata = self.predict_action(instruction, observation)
            
            if action is None:
                logger.error("Failed to predict action")
                break
            
            results['actions'].append(action.copy())
            
            # Execute action
            success, error = self.execute_action(action, observation)
            
            if not success:
                logger.error(f"Action execution failed: {error}")
                break
            
            # Small delay between steps
            import time
            time.sleep(0.1)
        
        results['steps'] = step + 1
        logger.info(f"Episode completed: success={results['success']}, steps={results['steps']}")
        
        return results

