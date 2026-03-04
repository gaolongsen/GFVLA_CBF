"""
Computer vision module for detecting Jenga blocks and estimating their 6D poses.

Uses RGBD cameras to detect blocks in the scene and estimate their positions
and orientations for robotic grasping.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
import logging
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)


@dataclass
class BlockPose:
    """6D pose of a detected Jenga block."""
    position: np.ndarray  # [x, y, z] in meters
    orientation: np.ndarray  # [roll, pitch, yaw] in radians (Euler angles)
    quaternion: Optional[np.ndarray] = None  # [x, y, z, w] quaternion
    confidence: float = 1.0  # Detection confidence (0-1)
    block_id: Optional[int] = None  # Unique identifier for tracking


class BlockDetector:
    """Detect and estimate poses of Jenga blocks using RGBD cameras."""
    
    def __init__(
        self,
        camera_intrinsics: Optional[np.ndarray] = None,
        block_dimensions: Tuple[float, float, float] = (0.075, 0.025, 0.015),
    ):
        """
        Initialize block detector.
        
        Args:
            camera_intrinsics: Camera intrinsic matrix [3x3] (if None, uses defaults)
            block_dimensions: Jenga block dimensions (length, width, height) in meters
        """
        self.block_length, self.block_width, self.block_height = block_dimensions
        
        # Default camera intrinsics (adjust based on your camera)
        if camera_intrinsics is None:
            # Typical RGBD camera (e.g., RealSense, Azure Kinect)
            self.camera_intrinsics = np.array([
                [525.0, 0.0, 320.0],
                [0.0, 525.0, 240.0],
                [0.0, 0.0, 1.0]
            ])
        else:
            self.camera_intrinsics = camera_intrinsics
        
        # Block detection parameters
        self.min_block_area = 100  # Minimum pixel area for block detection
        self.max_block_area = 5000  # Maximum pixel area
        self.aspect_ratio_range = (2.0, 4.0)  # Expected aspect ratio (length/width)
        
    def detect_blocks(
        self,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        camera_pose: Optional[np.ndarray] = None,
    ) -> List[BlockPose]:
        """
        Detect Jenga blocks in RGBD images and estimate their 6D poses.
        
        Args:
            rgb_image: RGB image [H, W, 3] (BGR format)
            depth_image: Depth image [H, W] in meters
            camera_pose: Camera pose in world frame [4x4] transformation matrix (optional)
        
        Returns:
            List of detected block poses
        """
        if rgb_image is None or depth_image is None:
            logger.warning("Invalid input images")
            return []
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        block_poses = []
        
        for i, contour in enumerate(contours):
            # Filter by area
            area = cv2.contourArea(contour)
            if area < self.min_block_area or area > self.max_block_area:
                continue
            
            # Get bounding rectangle
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            width, height = rect[1]
            if width < height:
                width, height = height, width
            
            # Filter by aspect ratio (Jenga blocks are rectangular)
            aspect_ratio = width / height if height > 0 else 0
            if not (self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]):
                continue
            
            # Get center point
            center_2d = rect[0]  # (x, y) in image coordinates
            center_x, center_y = int(center_2d[0]), int(center_2d[1])
            
            # Get depth at center
            if (0 <= center_y < depth_image.shape[0] and 
                0 <= center_x < depth_image.shape[1]):
                depth = depth_image[center_y, center_x]
                
                if depth > 0 and depth < 2.0:  # Valid depth range (0-2m)
                    # Convert 2D pixel to 3D point
                    position_3d = self._pixel_to_3d(center_x, center_y, depth)
                    
                    # Estimate orientation from bounding box
                    angle = rect[2]  # Rotation angle from minAreaRect
                    orientation = self._estimate_orientation(rect, depth_image)
                    
                    # Create block pose
                    block_pose = BlockPose(
                        position=position_3d,
                        orientation=orientation,
                        confidence=min(area / self.max_block_area, 1.0),
                        block_id=i,
                    )
                    
                    # Transform to world frame if camera pose provided
                    if camera_pose is not None:
                        block_pose = self._transform_to_world(block_pose, camera_pose)
                    
                    block_poses.append(block_pose)
        
        logger.info(f"Detected {len(block_poses)} blocks")
        return block_poses
    
    def _pixel_to_3d(self, u: int, v: int, depth: float) -> np.ndarray:
        """
        Convert pixel coordinates and depth to 3D point.
        
        Args:
            u, v: Pixel coordinates
            depth: Depth value in meters
        
        Returns:
            3D point [x, y, z] in camera frame
        """
        fx = self.camera_intrinsics[0, 0]
        fy = self.camera_intrinsics[1, 1]
        cx = self.camera_intrinsics[0, 2]
        cy = self.camera_intrinsics[1, 2]
        
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth
        
        return np.array([x, y, z])
    
    def _estimate_orientation(
        self,
        rect: Tuple,
        depth_image: np.ndarray,
    ) -> np.ndarray:
        """
        Estimate block orientation from bounding box and depth.
        
        Args:
            rect: Bounding rectangle from cv2.minAreaRect
            depth_image: Depth image for additional information
        
        Returns:
            Orientation [roll, pitch, yaw] in radians
        """
        # Get rotation angle from bounding box
        angle = rect[2]  # Degrees
        
        # Convert to radians and adjust for coordinate system
        yaw = np.deg2rad(angle)
        
        # For blocks on a table, roll and pitch are typically small
        # (assuming blocks are mostly flat on the table)
        roll = 0.0
        pitch = 0.0
        
        # If depth information suggests tilt, adjust pitch
        # This is simplified - in practice, you'd use plane fitting
        
        return np.array([roll, pitch, yaw])
    
    def _transform_to_world(
        self,
        block_pose: BlockPose,
        camera_pose: np.ndarray,
    ) -> BlockPose:
        """
        Transform block pose from camera frame to world frame.
        
        Args:
            block_pose: Block pose in camera frame
            camera_pose: Camera pose transformation matrix [4x4]
        
        Returns:
            Block pose in world frame
        """
        # Transform position
        position_cam = np.append(block_pose.position, 1.0)  # Homogeneous
        position_world = camera_pose @ position_cam
        position_world = position_world[:3]
        
        # Transform orientation
        # Convert Euler to rotation matrix
        r_cam = R.from_euler('xyz', block_pose.orientation, degrees=False)
        R_cam = r_cam.as_matrix()
        
        # Get rotation part of camera pose
        R_world_cam = camera_pose[:3, :3]
        R_world = R_world_cam @ R_cam
        
        # Convert back to Euler
        r_world = R.from_matrix(R_world)
        orientation_world = r_world.as_euler('xyz', degrees=False)
        
        # Create quaternion
        quaternion = r_world.as_quat()  # [x, y, z, w]
        
        return BlockPose(
            position=position_world,
            orientation=orientation_world,
            quaternion=quaternion,
            confidence=block_pose.confidence,
            block_id=block_pose.block_id,
        )
    
    def select_grasp_pose(
        self,
        block_pose: BlockPose,
        approach_height: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select optimal grasp pose for a block.
        
        Args:
            block_pose: Detected block pose
            approach_height: Height above block for approach
        
        Returns:
            Tuple of (approach_pose, grasp_pose) as [x, y, z, roll, pitch, yaw]
        """
        position = block_pose.position
        orientation = block_pose.orientation
        
        # Approach pose: above the block
        approach_pose = np.array([
            position[0],
            position[1],
            position[2] + approach_height,
            orientation[0],  # roll
            np.pi,  # pitch (gripper pointing down)
            orientation[2],  # yaw (aligned with block)
        ])
        
        # Grasp pose: at the block center
        grasp_pose = np.array([
            position[0],
            position[1],
            position[2] + self.block_height / 2,  # Center of block
            orientation[0],  # roll
            np.pi,  # pitch (gripper pointing down)
            orientation[2],  # yaw (aligned with block)
        ])
        
        return approach_pose, grasp_pose
    
    def filter_blocks_by_region(
        self,
        blocks: List[BlockPose],
        region_center: np.ndarray,
        region_size: float = 0.3,
    ) -> List[BlockPose]:
        """
        Filter blocks within a specified region.
        
        Args:
            blocks: List of detected blocks
            region_center: Center of region [x, y, z]
            region_size: Size of region (radius) in meters
        
        Returns:
            Filtered list of blocks within region
        """
        filtered = []
        for block in blocks:
            distance = np.linalg.norm(block.position[:2] - region_center[:2])
            if distance <= region_size:
                filtered.append(block)
        
        return filtered


class CameraInterface:
    """Interface for RGBD cameras (RealSense, Azure Kinect, etc.)."""
    
    def __init__(self, camera_type: str = "realsense"):
        """
        Initialize camera interface.
        
        Args:
            camera_type: Type of camera ('realsense', 'kinect', 'simulation')
        """
        self.camera_type = camera_type
        self.camera = None
        self.connected = False
        
    def connect(self) -> bool:
        """
        Connect to camera.
        
        Returns:
            True if connection successful
        """
        try:
            if self.camera_type == "realsense":
                try:
                    import pyrealsense2 as rs
                    self.camera = rs.pipeline()
                    config = rs.config()
                    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                    self.camera.start(config)
                    self.connected = True
                    logger.info("Connected to RealSense camera")
                    return True
                except ImportError:
                    logger.warning("pyrealsense2 not installed, using simulation mode")
                    self.camera_type = "simulation"
            
            if self.camera_type == "simulation":
                # Simulation mode - return mock images
                self.connected = True
                logger.info("Using simulation camera mode")
                return True
                
        except Exception as e:
            logger.error(f"Failed to connect to camera: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from camera."""
        if self.camera is not None:
            if self.camera_type == "realsense":
                self.camera.stop()
        self.connected = False
    
    def capture_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Capture RGB and depth frames.
        
        Returns:
            Tuple of (rgb_image, depth_image)
        """
        if not self.connected:
            return None, None
        
        try:
            if self.camera_type == "realsense":
                frames = self.camera.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                
                if not depth_frame or not color_frame:
                    return None, None
                
                depth_image = np.asanyarray(depth_frame.get_data())
                rgb_image = np.asanyarray(color_frame.get_data())
                
                # Convert depth from mm to meters
                depth_image = depth_image.astype(np.float32) / 1000.0
                
                return rgb_image, depth_image
            
            elif self.camera_type == "simulation":
                # Return mock images for testing
                rgb_image = np.zeros((480, 640, 3), dtype=np.uint8)
                depth_image = np.ones((480, 640), dtype=np.float32) * 0.5  # 0.5m depth
                return rgb_image, depth_image
                
        except Exception as e:
            logger.error(f"Failed to capture frame: {e}")
            return None, None
    
    def get_camera_intrinsics(self) -> Optional[np.ndarray]:
        """Get camera intrinsic matrix."""
        if self.camera_type == "realsense" and self.camera is not None:
            try:
                frames = self.camera.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
                
                return np.array([
                    [intrinsics.fx, 0, intrinsics.ppx],
                    [0, intrinsics.fy, intrinsics.ppy],
                    [0, 0, 1]
                ])
            except:
                pass
        
        # Default intrinsics
        return np.array([
            [525.0, 0.0, 320.0],
            [0.0, 525.0, 240.0],
            [0.0, 0.0, 1.0]
        ])

