"""
Dual-arm demonstration: Build the word "VLA" using Jenga blocks on a table.

This script demonstrates dual-arm coordination to arrange Jenga blocks on a
2D plane (table surface) to form the letters "V", "L", and "A". The UR5e
(left arm) and UR10e (right arm) coordinate to efficiently place blocks
in the correct positions to form the letters.

Based on GF-VLA framework for dual-arm manipulation.
"""

import numpy as np
import time
import logging
import argparse
from typing import List, Tuple, Dict, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hardware import (
    DualArmHardwareInterface,
    HardwareConfig,
    obstacles_from_blocks,
    Obstacle,
)
from hardware.vision import BlockDetector, CameraInterface, BlockPose

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VLALetterBuilder:
    """Build the letters "VLA" using Jenga blocks on a 2D plane with vision-based detection."""
    
    def __init__(
        self,
        hardware: DualArmHardwareInterface,
        camera: Optional[CameraInterface] = None,
        use_vision: bool = True,
    ):
        """
        Initialize VLA letter builder.
        
        Args:
            hardware: Dual-arm hardware interface
            camera: Camera interface for block detection (optional)
            use_vision: Whether to use vision-based detection (default: True)
        """
        self.hardware = hardware
        self.camera = camera
        self.use_vision = use_vision
        
        # Block configuration
        self.block_size = 0.015  # Jenga block height: 1.5cm
        self.block_width = 0.025  # 2.5cm
        self.block_length = 0.075  # 7.5cm
        
        # Table surface height
        self.table_height = 0.1  # meters
        
        # Letter configuration
        self.letter_spacing = 0.15  # 15cm spacing between letters
        self.letter_start_x = 0.4  # Starting X position
        self.letter_start_y = -0.1  # Starting Y position
        
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
            [0.2, -0.3, self.table_height],  # Block 1
            [0.2, -0.25, self.table_height],  # Block 2
            [0.2, -0.2, self.table_height],  # Block 3
            [0.2, -0.15, self.table_height],  # Block 4
            [0.2, -0.1, self.table_height],  # Block 5
            [0.2, -0.05, self.table_height],  # Block 6
            [0.2, 0.0, self.table_height],  # Block 7
            [0.2, 0.05, self.table_height],  # Block 8
            [0.2, 0.1, self.table_height],  # Block 9
            [0.2, 0.15, self.table_height],  # Block 10
            [0.2, 0.2, self.table_height],  # Block 11
            [0.2, 0.25, self.table_height],  # Block 12
            [0.2, 0.3, self.table_height],  # Block 13
            [0.25, -0.3, self.table_height],  # Block 14
            [0.25, -0.25, self.table_height],  # Block 15
            [0.25, -0.2, self.table_height],  # Block 16
            [0.25, -0.15, self.table_height],  # Block 17
            [0.25, -0.1, self.table_height],  # Block 18
            [0.25, -0.05, self.table_height],  # Block 19
            [0.25, 0.0, self.table_height],  # Block 20
        ]
        
        # Detected blocks cache
        self.detected_blocks: List[BlockPose] = []
        self.used_blocks: set = set()  # Track which blocks have been used
        
        # Approach height for picking and placing
        self.approach_height = 0.05  # 5cm above target
        self.grasp_height = 0.01  # 1cm above surface for grasping
        
        # Define letter patterns (relative positions for each block)
        self.letter_patterns = self._define_letter_patterns()
        
    def _define_letter_patterns(self) -> Dict[str, List[Tuple[float, float, float]]]:
        """
        Define block positions for each letter.
        
        Returns:
            Dictionary mapping letter to list of (x, y, yaw) positions
        """
        # Letter "V" pattern (7 blocks)
        # V shape: two diagonal lines meeting at bottom
        letter_v = [
            # Left diagonal
            (0.0, 0.08, 0.0),  # Top left
            (0.02, 0.04, 0.0),  # Middle left
            (0.04, 0.0, 0.0),  # Bottom center
            # Right diagonal
            (0.06, 0.04, 0.0),  # Middle right
            (0.08, 0.08, 0.0),  # Top right
            # Additional blocks for thickness
            (0.01, 0.06, 0.0),  # Left side
            (0.07, 0.06, 0.0),  # Right side
        ]
        
        # Letter "L" pattern (6 blocks)
        # L shape: vertical line + horizontal line
        letter_l = [
            # Vertical line
            (0.0, 0.0, 0.0),  # Bottom left
            (0.0, 0.025, 0.0),  # Middle left
            (0.0, 0.05, 0.0),  # Top left
            # Horizontal line
            (0.025, 0.0, 0.0),  # Bottom middle
            (0.05, 0.0, 0.0),  # Bottom right
            (0.075, 0.0, 0.0),  # Bottom far right
        ]
        
        # Letter "A" pattern (7 blocks)
        # A shape: triangle top + horizontal bar + two sides
        letter_a = [
            # Top triangle
            (0.04, 0.08, 0.0),  # Top center
            (0.02, 0.06, 0.0),  # Top left
            (0.06, 0.06, 0.0),  # Top right
            # Horizontal bar
            (0.02, 0.04, 0.0),  # Bar left
            (0.04, 0.04, 0.0),  # Bar center
            (0.06, 0.04, 0.0),  # Bar right
            # Bottom
            (0.04, 0.0, 0.0),  # Bottom center
        ]
        
        return {
            'V': letter_v,
            'L': letter_l,
            'A': letter_a,
        }
    
    def _get_obstacles_for_action(
        self,
        current_block: BlockPose,
        other_blocks: List[BlockPose],
        letter: str,
        letter_index: int,
        block_index: int,
    ) -> Optional[List[Obstacle]]:
        """
        Get obstacles for CBF filter: other blocks + already-placed blocks.
        Excludes the block being picked.
        """
        if not self.use_cbf:
            return None
        obstacles = []
        if other_blocks:
            obstacles.extend(obstacles_from_blocks(other_blocks, safety_margin=0.03))
        # Already-placed blocks (from letter pattern positions)
        pattern = self.letter_patterns.get(letter, [])
        for j in range(min(block_index, len(pattern))):
            rel_x, rel_y, _ = pattern[j]
            letter_x = self.letter_start_x + letter_index * self.letter_spacing
            pos = np.array([
                letter_x + rel_x,
                self.letter_start_y + rel_y,
                self.table_height,
            ])
            obstacles.append(Obstacle(position=pos, radius=0.02))
        return obstacles if obstacles else None
    
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
                region_size=0.4
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
    
    def get_place_pose(
        self,
        letter: str,
        letter_index: int,
        block_index: int,
        approach: bool = True
    ) -> np.ndarray:
        """
        Get pose for placing a block to form a letter.
        
        Args:
            letter: Letter to build ('V', 'L', or 'A')
            letter_index: Which letter (0='V', 1='L', 2='A')
            block_index: Which block in the letter pattern
            approach: If True, return approach pose (higher), else place pose
        
        Returns:
            Pose array [x, y, z, roll, pitch, yaw]
        """
        if letter not in self.letter_patterns:
            raise ValueError(f"Unknown letter: {letter}")
        
        pattern = self.letter_patterns[letter]
        if block_index >= len(pattern):
            raise ValueError(f"Block index {block_index} out of range for letter {letter}")
        
        rel_x, rel_y, yaw = pattern[block_index]
        
        # Calculate absolute position
        letter_x = self.letter_start_x + letter_index * self.letter_spacing
        x = letter_x + rel_x
        y = self.letter_start_y + rel_y
        z = self.table_height
        
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
    
    def build_letter(
        self,
        letter: str,
        letter_index: int,
        block_counter: int,
        use_dual_arm: bool = True,
        source_region_center: Optional[np.ndarray] = None,
    ) -> int:
        """
        Build a single letter.
        
        Args:
            letter: Letter to build ('V', 'L', or 'A')
            letter_index: Index of letter (0='V', 1='L', 2='A')
            block_counter: Current block counter
            use_dual_arm: Whether to use both arms for efficiency
        
        Returns:
            Updated block counter
        """
        logger.info(f"Building letter '{letter}' (index {letter_index})")
        
        pattern = self.letter_patterns[letter]
        num_blocks = len(pattern)
        
        # Determine which arm picks which blocks
        # Left arm (UR5e) picks first half, right arm (UR10e) picks second half
        if use_dual_arm and num_blocks > 3:
            left_blocks = num_blocks // 2
            right_blocks = num_blocks - left_blocks
        else:
            left_blocks = num_blocks
            right_blocks = 0
        
        # Build letter with left arm
        for i in range(left_blocks):
            available_blocks: List[BlockPose] = []
            # Detect available blocks
            if self.use_vision and self.detector is not None:
                available_blocks = self.detect_available_blocks(source_region_center)
                
                if len(available_blocks) == 0:
                    logger.warning("No available blocks detected! Trying fallback positions.")
                    if block_counter >= len(self.fallback_source_positions):
                        logger.error("Not enough blocks! Stopping.")
                        return block_counter
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
                    logger.warning("Not enough source blocks! Stopping.")
                    return block_counter
                source_pos = self.fallback_source_positions[block_counter]
                block_pose = BlockPose(
                    position=np.array(source_pos),
                    orientation=np.array([0.0, 0.0, 0.0]),
                    block_id=block_counter
                )
                block_counter += 1
            
            logger.info(f"  Left arm: Placing block {i + 1}/{left_blocks} for letter '{letter}'")
            
            # Pick block
            pick_approach = self.get_pick_pose(block_pose, approach=True)
            pick_grasp = self.get_pick_pose(block_pose, approach=False)
            
            # Build obstacles for CBF (other blocks in scene, already-placed blocks)
            other_blocks = available_blocks[1:] if len(available_blocks) > 1 else []
            obstacles = self._get_obstacles_for_action(block_pose, other_blocks, letter, letter_index, i)
            
            # Approach and grasp
            action = np.concatenate([
                pick_approach, [1.0],  # Left arm approach, gripper open
                np.zeros(7),  # Right arm stay
            ])
            self.hardware.execute_action(action, blocking=True, obstacles=obstacles)
            time.sleep(0.3)
            
            action = np.concatenate([
                pick_grasp, [0.0],  # Left arm grasp, gripper close
                np.zeros(7),  # Right arm stay
            ])
            self.hardware.execute_action(action, blocking=True, obstacles=obstacles)
            time.sleep(0.3)
            
            # Lift
            action = np.concatenate([
                pick_approach, [0.0],  # Left arm lift, gripper closed
                np.zeros(7),  # Right arm stay
            ])
            self.hardware.execute_action(action, blocking=True, obstacles=obstacles)
            time.sleep(0.3)
            
            # Place block
            place_approach = self.get_place_pose(letter, letter_index, i, approach=True)
            place_position = self.get_place_pose(letter, letter_index, i, approach=False)
            
            action = np.concatenate([
                place_approach, [0.0],  # Left arm approach, gripper closed
                np.zeros(7),  # Right arm stay
            ])
            self.hardware.execute_action(action, blocking=True, obstacles=obstacles)
            time.sleep(0.3)
            
            action = np.concatenate([
                place_position, [0.0],  # Left arm place, gripper closed
                np.zeros(7),  # Right arm stay
            ])
            self.hardware.execute_action(action, blocking=True, obstacles=obstacles)
            time.sleep(0.2)
            
            # Release
            action = np.concatenate([
                place_position, [1.0],  # Left arm release, gripper open
                np.zeros(7),  # Right arm stay
            ])
            self.hardware.execute_action(action, blocking=True, obstacles=obstacles)
            time.sleep(0.2)
            
            # Retract
            action = np.concatenate([
                place_approach, [1.0],  # Left arm retract, gripper open
                np.zeros(7),  # Right arm stay
            ])
            self.hardware.execute_action(action, blocking=True, obstacles=obstacles)
            time.sleep(0.3)
        
        # Build letter with right arm (if dual-arm mode)
        if right_blocks > 0:
            for i in range(left_blocks, num_blocks):
                available_blocks = []
                # Detect available blocks
                if self.use_vision and self.detector is not None:
                    available_blocks = self.detect_available_blocks(source_region_center)
                    
                    if len(available_blocks) == 0:
                        logger.warning("No available blocks detected! Trying fallback positions.")
                        if block_counter >= len(self.fallback_source_positions):
                            logger.error("Not enough blocks! Stopping.")
                            return block_counter
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
                        logger.warning("Not enough source blocks! Stopping.")
                        return block_counter
                    source_pos = self.fallback_source_positions[block_counter]
                    block_pose = BlockPose(
                        position=np.array(source_pos),
                        orientation=np.array([0.0, 0.0, 0.0]),
                        block_id=block_counter
                    )
                    block_counter += 1
                
                logger.info(f"  Right arm: Placing block {i + 1}/{num_blocks} for letter '{letter}'")
                
                # Pick block
                pick_approach = self.get_pick_pose(block_pose, approach=True)
                pick_grasp = self.get_pick_pose(block_pose, approach=False)
                
                # Build obstacles for CBF
                other_blocks_right = available_blocks[1:] if len(available_blocks) > 1 else []
                obstacles_right = self._get_obstacles_for_action(block_pose, other_blocks_right, letter, letter_index, i)
                
                # Approach and grasp
                action = np.concatenate([
                    np.zeros(7),  # Left arm stay
                    pick_approach, [1.0],  # Right arm approach, gripper open
                ])
                self.hardware.execute_action(action, blocking=True, obstacles=obstacles_right)
                time.sleep(0.3)
                
                action = np.concatenate([
                    np.zeros(7),  # Left arm stay
                    pick_grasp, [0.0],  # Right arm grasp, gripper close
                ])
                self.hardware.execute_action(action, blocking=True, obstacles=obstacles_right)
                time.sleep(0.3)
                
                # Lift
                action = np.concatenate([
                    np.zeros(7),  # Left arm stay
                    pick_approach, [0.0],  # Right arm lift, gripper closed
                ])
                self.hardware.execute_action(action, blocking=True, obstacles=obstacles_right)
                time.sleep(0.3)
                
                # Place block
                place_approach = self.get_place_pose(letter, letter_index, i, approach=True)
                place_position = self.get_place_pose(letter, letter_index, i, approach=False)
                
                action = np.concatenate([
                    np.zeros(7),  # Left arm stay
                    place_approach, [0.0],  # Right arm approach, gripper closed
                ])
                self.hardware.execute_action(action, blocking=True, obstacles=obstacles_right)
                time.sleep(0.3)
                
                action = np.concatenate([
                    np.zeros(7),  # Left arm stay
                    place_position, [0.0],  # Right arm place, gripper closed
                ])
                self.hardware.execute_action(action, blocking=True, obstacles=obstacles_right)
                time.sleep(0.2)
                
                # Release
                action = np.concatenate([
                    np.zeros(7),  # Left arm stay
                    place_position, [1.0],  # Right arm release, gripper open
                ])
                self.hardware.execute_action(action, blocking=True, obstacles=obstacles_right)
                time.sleep(0.2)
                
                # Retract
                action = np.concatenate([
                    np.zeros(7),  # Left arm stay
                    place_approach, [1.0],  # Right arm retract, gripper open
                ])
                self.hardware.execute_action(action, blocking=True, obstacles=obstacles_right)
                time.sleep(0.3)
        
        logger.info(f"Letter '{letter}' completed")
        return block_counter
    
    def build_vla(self, use_dual_arm: bool = True):
        """
        Build the word "VLA" using Jenga blocks.
        
        Args:
            use_dual_arm: Whether to use both arms for efficiency
        """
        logger.info("Starting VLA letter construction")
        
        # Reset to home position
        logger.info("Moving to home position...")
        self.hardware.reset()
        time.sleep(2.0)
        
        block_counter = 0
        
        # Source region for block detection
        source_region_center = np.array([0.225, 0.0, self.table_height])
        
        # Build each letter
        letters = ['V', 'L', 'A']
        for letter_idx, letter in enumerate(letters):
            logger.info(f"Building letter {letter_idx + 1}/3: '{letter}'")
            block_counter = self.build_letter(
                letter, letter_idx, block_counter, use_dual_arm, source_region_center
            )
            time.sleep(1.0)  # Pause between letters
        
        logger.info("VLA construction completed!")
        
        # Return to home position
        logger.info("Returning to home position...")
        self.hardware.reset()
        time.sleep(2.0)


def main():
    """Main function to run VLA building demonstration."""
    parser = argparse.ArgumentParser(
        description="Dual-arm demonstration: Build the word 'VLA' with Jenga blocks"
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
        '--single-arm',
        action='store_true',
        help='Use only single arm (left arm) instead of dual-arm coordination'
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
    parser.add_argument(
        '--no-cbf',
        action='store_true',
        help='Disable Control Barrier Function (CBF) obstacle avoidance filter'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Dual-Arm VLA Letter Building Demonstration")
    logger.info("=" * 60)
    logger.info(f"Configuration:")
    logger.info(f"  UR5e IP: {args.ur5e_ip}")
    logger.info(f"  UR10e IP: {args.ur10e_ip}")
    logger.info(f"  Robotiq port: {args.robotiq_port}")
    logger.info(f"  Barrett IP: {args.barrett_ip}")
    logger.info(f"  Dual-arm mode: {not args.single_arm}")
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
    config.safety.use_cbf_filter = not args.no_cbf
    
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
        # Create letter builder
        builder = VLALetterBuilder(hardware, camera=camera, use_vision=use_vision)
        
        # Build VLA
        builder.build_vla(use_dual_arm=not args.single_arm)
        
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

