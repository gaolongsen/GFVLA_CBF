"""
Dual-arm demonstration: Build a 3D tower using Jenga blocks.

This script demonstrates dual-arm coordination to build a vertical tower
by stacking Jenga blocks. The UR5e (left arm) typically retrieves blocks
from a source location, while the UR10e (right arm) stabilizes and places
blocks to build the tower structure.

Based on GF-VLA framework for dual-arm manipulation.
"""

import numpy as np
import time
import logging
import argparse
from typing import List, Tuple, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hardware import DualArmHardwareInterface, HardwareConfig
from hardware.vision import BlockDetector, CameraInterface, BlockPose

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TowerBuilder:
    """Build a 3D tower using dual-arm coordination with vision-based block detection."""
    
    def __init__(
        self,
        hardware: DualArmHardwareInterface,
        camera: Optional[CameraInterface] = None,
        use_vision: bool = True,
    ):
        """
        Initialize tower builder.
        
        Args:
            hardware: Dual-arm hardware interface
            camera: Camera interface for block detection (optional)
            use_vision: Whether to use vision-based detection (default: True)
        """
        self.hardware = hardware
        self.camera = camera
        self.use_vision = use_vision
        
        # Tower configuration
        self.block_size = 0.015  # Jenga block size: 1.5cm x 2.5cm x 7.5cm (typical)
        self.block_width = 0.025  # 2.5cm
        self.block_length = 0.075  # 7.5cm
        
        # Tower base position (center of tower)
        self.tower_base_x = 0.5  # meters
        self.tower_base_y = 0.0  # meters
        self.tower_base_z = 0.1  # meters (table height)
        
        # Initialize block detector
        if self.use_vision and self.camera is not None:
            camera_intrinsics = self.camera.get_camera_intrinsics()
            self.detector = BlockDetector(
                camera_intrinsics=camera_intrinsics,
                block_dimensions=(self.block_length, self.block_width, self.block_size)
            )
            logger.info("Vision-based block detection enabled")
        else:
            self.detector = None
            logger.warning("Vision detection disabled - using fallback positions")
        
        # Fallback source block positions (used if vision fails)
        self.fallback_source_positions = [
            [0.3, -0.2, 0.1],  # Block 1
            [0.3, -0.15, 0.1],  # Block 2
            [0.3, -0.1, 0.1],  # Block 3
            [0.3, -0.05, 0.1],  # Block 4
            [0.3, 0.0, 0.1],  # Block 5
            [0.3, 0.05, 0.1],  # Block 6
        ]
        
        # Detected blocks cache
        self.detected_blocks: List[BlockPose] = []
        self.used_blocks: set = set()  # Track which blocks have been used
        
        # Pick and place approach heights
        self.approach_height = 0.05  # 5cm above target
        self.grasp_height = 0.01  # 1cm above surface for grasping
        
    def detect_available_blocks(self, source_region_center: Optional[np.ndarray] = None) -> List[BlockPose]:
        """
        Detect available blocks in the scene.
        
        Args:
            source_region_center: Center of source region to search (optional)
        
        Returns:
            List of detected block poses
        """
        if not self.use_vision or self.camera is None or self.detector is None:
            logger.warning("Vision not available, using fallback positions")
            return []
        
        logger.info("Detecting blocks in scene...")
        
        # Capture frame
        rgb_image, depth_image = self.camera.capture_frame()
        if rgb_image is None or depth_image is None:
            logger.error("Failed to capture camera frame")
            return []
        
        # Detect blocks
        blocks = self.detector.detect_blocks(rgb_image, depth_image)
        
        # Filter by source region if specified
        if source_region_center is not None:
            blocks = self.detector.filter_blocks_by_region(
                blocks,
                source_region_center,
                region_size=0.3
            )
        
        # Filter out already used blocks
        available_blocks = [
            b for b in blocks
            if b.block_id not in self.used_blocks
        ]
        
        # Sort by confidence (highest first)
        available_blocks.sort(key=lambda x: x.confidence, reverse=True)
        
        logger.info(f"Found {len(available_blocks)} available blocks")
        return available_blocks
    
    def get_pick_pose(self, block_pose: BlockPose, approach: bool = True) -> np.ndarray:
        """
        Get pose for picking a block.
        
        Args:
            block_pose: Detected block pose
            approach: If True, return approach pose (higher), else grasp pose
        
        Returns:
            Pose array [x, y, z, roll, pitch, yaw]
        """
        if self.detector is not None:
            # Use detector to get optimal grasp pose
            approach_pose, grasp_pose = self.detector.select_grasp_pose(
                block_pose,
                approach_height=self.approach_height
            )
            return approach_pose if approach else grasp_pose
        else:
            # Fallback: use block position directly
            position = block_pose.position
            orientation = block_pose.orientation
            height = self.approach_height if approach else self.grasp_height
            
            pose = np.array([
                position[0],
                position[1],
                position[2] + height,
                orientation[0],  # roll
                np.pi,  # pitch (gripper pointing down)
                orientation[2],  # yaw (aligned with block)
            ])
            return pose
    
    def get_place_pose(self, tower_level: int, block_index: int, approach: bool = True) -> np.ndarray:
        """
        Get pose for placing a block on the tower.
        
        Args:
            tower_level: Which level of the tower (0 = base)
            block_index: Which block in the level (0-2 for 3 blocks per level)
            approach: If True, return approach pose (higher), else place pose
        
        Returns:
            Pose array [x, y, z, roll, pitch, yaw]
        """
        # Tower structure: 3 blocks per level, alternating orientation
        level_height = self.tower_base_z + tower_level * self.block_size
        
        # Calculate block position based on level and index
        if tower_level % 2 == 0:
            # Even levels: blocks along X axis
            x_offset = (block_index - 1) * self.block_length
            y_offset = 0.0
            yaw = 0.0  # Blocks aligned with X axis
        else:
            # Odd levels: blocks along Y axis
            x_offset = 0.0
            y_offset = (block_index - 1) * self.block_length
            yaw = np.pi / 2  # Blocks aligned with Y axis
        
        x = self.tower_base_x + x_offset
        y = self.tower_base_y + y_offset
        z = level_height
        
        height = self.approach_height if approach else 0.0
        
        pose = np.array([
            x,
            y,
            z + height,
            0.0,  # roll
            np.pi,  # pitch (gripper pointing down)
            yaw,  # yaw (block orientation)
        ])
        
        return pose
    
    def build_tower(self, num_levels: int = 3, num_blocks_per_level: int = 3):
        """
        Build a tower with specified number of levels.
        
        Args:
            num_levels: Number of levels in the tower
            num_blocks_per_level: Number of blocks per level
        """
        logger.info(f"Starting tower construction: {num_levels} levels, {num_blocks_per_level} blocks per level")
        
        # Reset to home position
        logger.info("Moving to home position...")
        self.hardware.reset()
        time.sleep(2.0)
        
        block_counter = 0
        
        # Source region for block detection
        source_region_center = np.array([0.3, 0.0, 0.1])
        
        for level in range(num_levels):
            logger.info(f"Building level {level + 1}/{num_levels}")
            
            for block_idx in range(num_blocks_per_level):
                # Detect available blocks
                if self.use_vision and self.detector is not None:
                    available_blocks = self.detect_available_blocks(source_region_center)
                    
                    if len(available_blocks) == 0:
                        logger.warning("No available blocks detected! Trying fallback positions.")
                        if block_counter >= len(self.fallback_source_positions):
                            logger.error("Not enough blocks! Stopping construction.")
                            return
                        # Use fallback position
                        source_pos = self.fallback_source_positions[block_counter]
                        block_pose = BlockPose(
                            position=np.array(source_pos),
                            orientation=np.array([0.0, 0.0, 0.0]),
                            block_id=-1
                        )
                        block_counter += 1
                    else:
                        # Use detected block
                        block_pose = available_blocks[0]
                        self.used_blocks.add(block_pose.block_id)
                        logger.info(f"  Selected block ID {block_pose.block_id} at position {block_pose.position}")
                else:
                    # Fallback: use predefined positions
                    if block_counter >= len(self.fallback_source_positions):
                        logger.warning("Not enough source blocks! Stopping construction.")
                        return
                    source_pos = self.fallback_source_positions[block_counter]
                    block_pose = BlockPose(
                        position=np.array(source_pos),
                        orientation=np.array([0.0, 0.0, 0.0]),
                        block_id=block_counter
                    )
                    block_counter += 1
                
                logger.info(f"  Placing block {block_idx + 1}/{num_blocks_per_level} at level {level + 1}")
                
                # Step 1: Left arm (UR5e) picks block from source
                logger.info("    Left arm: Moving to source block...")
                pick_approach = self.get_pick_pose(block_pose, approach=True)
                pick_grasp = self.get_pick_pose(block_pose, approach=False)
                
                # Approach source block
                action_left_approach = np.concatenate([
                    pick_approach,  # [6] pose
                    [1.0],  # [1] gripper open
                    np.zeros(7),  # [7] right arm stay in place
                ])
                success, error = self.hardware.execute_action(action_left_approach, blocking=True)
                if not success:
                    logger.error(f"Failed to approach source: {error}")
                    return
                time.sleep(0.5)
                
                # Grasp block
                logger.info("    Left arm: Grasping block...")
                action_left_grasp = np.concatenate([
                    pick_grasp,  # [6] pose
                    [0.0],  # [1] gripper close
                    np.zeros(7),  # [7] right arm stay in place
                ])
                success, error = self.hardware.execute_action(action_left_grasp, blocking=True)
                if not success:
                    logger.error(f"Failed to grasp block: {error}")
                    return
                time.sleep(0.5)
                
                # Lift block
                lift_pose = pick_approach.copy()
                action_left_lift = np.concatenate([
                    lift_pose,  # [6] pose
                    [0.0],  # [1] gripper closed
                    np.zeros(7),  # [7] right arm stay in place
                ])
                success, error = self.hardware.execute_action(action_left_lift, blocking=True)
                if not success:
                    logger.error(f"Failed to lift block: {error}")
                    return
                time.sleep(0.5)
                
                # Step 2: Right arm (UR10e) moves to stabilize position (if needed)
                if level > 0 and block_idx == 0:
                    logger.info("    Right arm: Moving to stabilization position...")
                    stabilize_pose = self.get_place_pose(level - 1, 1, approach=True)
                    action_right_stabilize = np.concatenate([
                        np.zeros(7),  # [7] left arm hold block
                        stabilize_pose,  # [6] pose
                        [1.0],  # [1] gripper open
                    ])
                    success, error = self.hardware.execute_action(action_right_stabilize, blocking=True)
                    if not success:
                        logger.warning(f"Failed to stabilize: {error}")
                    time.sleep(0.5)
                
                # Step 3: Left arm moves to tower placement position
                logger.info("    Left arm: Moving to tower placement position...")
                place_approach = self.get_place_pose(level, block_idx, approach=True)
                place_position = self.get_place_pose(level, block_idx, approach=False)
                
                # Approach tower
                action_left_to_tower = np.concatenate([
                    place_approach,  # [6] pose
                    [0.0],  # [1] gripper closed
                    np.zeros(7),  # [7] right arm stay
                ])
                success, error = self.hardware.execute_action(action_left_to_tower, blocking=True)
                if not success:
                    logger.error(f"Failed to move to tower: {error}")
                    return
                time.sleep(0.5)
                
                # Place block
                logger.info("    Left arm: Placing block...")
                action_left_place = np.concatenate([
                    place_position,  # [6] pose
                    [0.0],  # [1] gripper closed
                    np.zeros(7),  # [7] right arm stay
                ])
                success, error = self.hardware.execute_action(action_left_place, blocking=True)
                if not success:
                    logger.error(f"Failed to place block: {error}")
                    return
                time.sleep(0.3)
                
                # Release block
                logger.info("    Left arm: Releasing block...")
                action_left_release = np.concatenate([
                    place_position,  # [6] pose
                    [1.0],  # [1] gripper open
                    np.zeros(7),  # [7] right arm stay
                ])
                success, error = self.hardware.execute_action(action_left_release, blocking=True)
                if not success:
                    logger.error(f"Failed to release block: {error}")
                    return
                time.sleep(0.3)
                
                # Retract
                action_left_retract = np.concatenate([
                    place_approach,  # [6] pose
                    [1.0],  # [1] gripper open
                    np.zeros(7),  # [7] right arm stay
                ])
                success, error = self.hardware.execute_action(action_left_retract, blocking=True)
                if not success:
                    logger.warning(f"Failed to retract: {error}")
                time.sleep(0.5)
                
                logger.info(f"  Block {block_idx + 1} placed successfully")
            
            logger.info(f"Level {level + 1} completed")
            time.sleep(1.0)  # Pause between levels
        
        logger.info("Tower construction completed!")
        
        # Return to home position
        logger.info("Returning to home position...")
        self.hardware.reset()
        time.sleep(2.0)


def main():
    """Main function to run tower building demonstration."""
    parser = argparse.ArgumentParser(
        description="Dual-arm demonstration: Build a 3D tower with Jenga blocks"
    )
    parser.add_argument(
        '--num-levels',
        type=int,
        default=3,
        help='Number of levels in the tower (default: 3)'
    )
    parser.add_argument(
        '--blocks-per-level',
        type=int,
        default=3,
        help='Number of blocks per level (default: 3)'
    )
    parser.add_argument(
        '--ur5e-ip',
        type=str,
        default='192.168.1.101',
        help='UR5e robot IP address (default: 192.168.1.101)'
    )
    parser.add_argument(
        '--ur10e-ip',
        type=str,
        default='192.168.1.102',
        help='UR10e robot IP address (default: 192.168.1.102)'
    )
    parser.add_argument(
        '--robotiq-port',
        type=str,
        default='/dev/ttyUSB0',
        help='Robotiq gripper serial port (default: /dev/ttyUSB0)'
    )
    parser.add_argument(
        '--barrett-ip',
        type=str,
        default='192.168.1.103',
        help='Barrett Hand IP address (default: 192.168.1.103)'
    )
    parser.add_argument(
        '--simulation',
        action='store_true',
        help='Run in simulation mode (no actual hardware connection)'
    )
    parser.add_argument(
        '--camera-type',
        type=str,
        default='realsense',
        choices=['realsense', 'kinect', 'simulation'],
        help='Camera type for block detection (default: realsense)'
    )
    parser.add_argument(
        '--no-vision',
        action='store_true',
        help='Disable vision-based detection, use fixed positions'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Dual-Arm Tower Building Demonstration")
    logger.info("=" * 60)
    logger.info(f"Configuration:")
    logger.info(f"  Tower levels: {args.num_levels}")
    logger.info(f"  Blocks per level: {args.blocks_per_level}")
    logger.info(f"  UR5e IP: {args.ur5e_ip}")
    logger.info(f"  UR10e IP: {args.ur10e_ip}")
    logger.info(f"  Robotiq port: {args.robotiq_port}")
    logger.info(f"  Barrett IP: {args.barrett_ip}")
    logger.info(f"  Simulation mode: {args.simulation}")
    logger.info(f"  Camera type: {args.camera_type}")
    logger.info(f"  Vision detection: {not args.no_vision}")
    logger.info("=" * 60)
    
    if args.simulation:
        logger.warning("SIMULATION MODE: No actual hardware will be controlled")
        logger.warning("This is a dry run - actions will be logged but not executed")
    
    # Initialize camera for vision detection
    camera = None
    use_vision = not args.no_vision
    if use_vision:
        camera = CameraInterface(camera_type=args.camera_type)
        if not args.simulation:
            if not camera.connect():
                logger.warning("Failed to connect to camera, disabling vision")
                use_vision = False
        else:
            camera.connect()  # Connect in simulation mode too
    
    # Create hardware configuration
    config = HardwareConfig()
    config.ur5e.ip_address = args.ur5e_ip
    config.ur10e.ip_address = args.ur10e_ip
    config.robotiq.port = args.robotiq_port
    config.barrett.ip_address = args.barrett_ip
    
    # Create hardware interface
    hardware = DualArmHardwareInterface(config)
    
    if not args.simulation:
        # Connect to hardware
        logger.info("Connecting to hardware...")
        if not hardware.connect():
            logger.error("Failed to connect to hardware. Exiting.")
            return 1
        
        logger.info("Hardware connected successfully")
    else:
        logger.info("Simulation mode: Skipping hardware connection")
        hardware.connected = True  # Set flag for simulation
    
    try:
        # Create tower builder
        builder = TowerBuilder(hardware, camera=camera, use_vision=use_vision)
        
        # Build tower
        builder.build_tower(
            num_levels=args.num_levels,
            num_blocks_per_level=args.blocks_per_level
        )
        
        logger.info("Demonstration completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Demonstration interrupted by user")
        if not args.simulation:
            hardware.emergency_stop()
        return 1
    except Exception as e:
        logger.error(f"Error during demonstration: {e}", exc_info=True)
        if not args.simulation:
            hardware.emergency_stop()
        return 1
    finally:
        if camera is not None:
            camera.disconnect()
        if not args.simulation:
            hardware.disconnect()
            logger.info("Hardware disconnected")


if __name__ == '__main__':
    exit(main())

