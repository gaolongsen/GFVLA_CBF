"""
Control Barrier Function (CBF) module for dual-arm obstacle avoidance.

Implements CBF-based safety filtering that minimally modifies nominal (VLA-predicted)
actions to ensure collision-free motion during dual-robot manipulation. The CBF-QP
formulation finds the closest safe action to the nominal action while satisfying:
- Obstacle avoidance (point/sphere obstacles)
- Inter-arm collision avoidance
- Workspace boundary constraints

Based on: Ames et al., "Control Barrier Functions: Theory and Applications"
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
import logging
from scipy.optimize import minimize, Bounds

logger = logging.getLogger(__name__)

# Small epsilon for numerical stability
_EPS = 1e-6


@dataclass
class Obstacle:
    """Represents an obstacle for CBF constraint (sphere/point)."""
    position: np.ndarray  # [x, y, z] center in meters
    radius: float = 0.0   # Safety margin around obstacle (meters)
    
    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=np.float64)
        if self.position.shape != (3,):
            raise ValueError("Obstacle position must be [x, y, z]")


@dataclass
class CBFConfig:
    """Configuration for Control Barrier Function filter."""
    # Safety margins (meters)
    obstacle_safety_margin: float = 0.08   # Min distance from obstacles
    inter_arm_safety_margin: float = 0.12  # Min distance between arm end-effectors
    
    # CBF parameters
    alpha: float = 1.0   # CBF class-K function parameter (forward invariance)
    
    # Workspace limits (meters) - [min, max] for each axis
    workspace_x: Tuple[float, float] = (0.0, 0.8)
    workspace_y: Tuple[float, float] = (-0.5, 0.5)
    workspace_z: Tuple[float, float] = (0.0, 0.6)
    
    # QP solver settings
    max_iter: int = 100
    tol: float = 1e-6
    use_slack: bool = False  # Allow small constraint violations (not recommended)


class ControlBarrierFunctionFilter:
    """
    CBF-based safety filter for dual-arm manipulation.
    
    Filters nominal actions (from VLA model) through a CBF-QP to produce
    safe actions that avoid obstacles and inter-arm collisions while
    staying close to the nominal control.
    """
    
    def __init__(
        self,
        config: Optional[CBFConfig] = None,
        obstacles: Optional[List[Obstacle]] = None,
    ):
        """
        Initialize CBF filter.
        
        Args:
            config: CBF configuration (uses defaults if None)
            obstacles: List of obstacles to avoid (can be updated dynamically)
        """
        self.config = config if config is not None else CBFConfig()
        self.obstacles = obstacles if obstacles is not None else []
        
        # Fixed obstacles (e.g., table, walls) - can be set from scene
        self._fixed_obstacles: List[Obstacle] = []
    
    def set_obstacles(self, obstacles: List[Obstacle]):
        """Update dynamic obstacles (e.g., from vision/depth)."""
        self.obstacles = obstacles
    
    def add_obstacle(self, position: np.ndarray, radius: float = 0.0):
        """Add a single obstacle."""
        self.obstacles.append(Obstacle(position=position, radius=radius))
    
    def set_fixed_obstacles(self, obstacles: List[Obstacle]):
        """Set fixed obstacles (table surface, walls, etc.)."""
        self._fixed_obstacles = obstacles
    
    def filter_action(
        self,
        nominal_action: np.ndarray,
        current_state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, bool, Optional[str]]:
        """
        Filter nominal action through CBF-QP to produce safe action.
        
        Args:
            nominal_action: VLA-predicted action [14] 
                [0:6] left arm pose, [6] left gripper, [7:13] right pose, [13] right gripper
            current_state: Optional dict with 'left_arm_pose', 'right_arm_pose' for warm start
        
        Returns:
            Tuple of (safe_action, success, error_message)
        """
        if len(nominal_action) != 14:
            return nominal_action, False, f"Action must be 14-dim, got {len(nominal_action)}"
        
        # Extract positions (only modify x,y,z; keep orientation and gripper)
        left_pos_nom = nominal_action[0:3].copy()
        right_pos_nom = nominal_action[7:10].copy()
        
        # Solve CBF-QP for safe positions
        u_nom = np.concatenate([left_pos_nom, right_pos_nom])  # [6]
        
        try:
            u_safe, success = self._solve_cbf_qp(u_nom, current_state)
            
            if not success:
                # Fallback: use nominal but log warning
                logger.warning("CBF-QP failed to find feasible solution, using nominal action")
                return nominal_action, False, "CBF-QP infeasible"
            
            # Reconstruct full action with safe positions
            safe_action = nominal_action.copy()
            safe_action[0:3] = u_safe[0:3]
            safe_action[7:10] = u_safe[3:6]
            
            return safe_action, True, None
            
        except Exception as e:
            logger.error(f"CBF filter error: {e}", exc_info=True)
            return nominal_action, False, str(e)
    
    def _solve_cbf_qp(
        self,
        u_nom: np.ndarray,
        current_state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, bool]:
        """
        Solve CBF-QP: min ||u - u_nom||^2 s.t. safety constraints.
        
        Uses linearized barrier constraints for real-time performance.
        """
        # Bounds from workspace limits
        x_min, x_max = self.config.workspace_x
        y_min, y_max = self.config.workspace_y
        z_min, z_max = self.config.workspace_z
        
        bounds = Bounds(
            lb=np.array([x_min, y_min, z_min, x_min, y_min, z_min]),
            ub=np.array([x_max, y_max, z_max, x_max, y_max, z_max]),
        )
        
        # Build linear inequality constraints: A_ub @ u <= b_ub
        A_list = []
        b_list = []
        
        # Collect all obstacles (dynamic + fixed)
        all_obstacles = self.obstacles + self._fixed_obstacles
        
        # Obstacle constraints for left arm [0:3]
        for obs in all_obstacles:
            d_safe = self.config.obstacle_safety_margin + obs.radius
            p = obs.position
            
            # Left arm - obstacle
            n_left = self._safe_normal(u_nom[0:3], p)
            if n_left is not None:
                row = np.zeros(6)
                row[0:3] = -n_left
                A_list.append(row)
                b_list.append(-d_safe - np.dot(n_left, p))
            
            # Right arm - obstacle
            n_right = self._safe_normal(u_nom[3:6], p)
            if n_right is not None:
                row = np.zeros(6)
                row[3:6] = -n_right
                A_list.append(row)
                b_list.append(-d_safe - np.dot(n_right, p))
        
        # Inter-arm collision constraint
        d_arm = self.config.inter_arm_safety_margin
        left_pos = u_nom[0:3]
        right_pos = u_nom[3:6]
        n_inter = self._safe_normal(left_pos, right_pos)
        
        if n_inter is not None:
            # Constraint: (left - right) · n >= d_arm
            # => n·left - n·right >= d_arm => -n·left + n·right <= -d_arm
            row = np.zeros(6)
            row[0:3] = -n_inter
            row[3:6] = n_inter
            A_list.append(row)
            b_list.append(-d_arm)
        
        # Solve QP
        if len(A_list) == 0:
            # No constraints - check bounds only
            u_safe = np.clip(u_nom, bounds.lb, bounds.ub)
            return u_safe, True
        
        A_ub = np.array(A_list)
        b_ub = np.array(b_list)
        
        def objective(u):
            return np.sum((u - u_nom) ** 2)
        
        def jac(u):
            return 2 * (u - u_nom)
        
        # Initial guess: clip nominal to bounds
        u0 = np.clip(u_nom, bounds.lb, bounds.ub)
        
        # Check if nominal already satisfies constraints
        if np.all(A_ub @ u0 <= b_ub + 1e-4):
            return u0, True
        
        constraints = {'type': 'ineq', 'fun': lambda u: b_ub - A_ub @ u}
        result = minimize(
            objective,
            u0,
            method='SLSQP',
            jac=jac,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.config.max_iter, 'ftol': self.config.tol},
        )
        
        if result.success:
            u_safe = result.x
            # Verify constraints
            if np.any(A_ub @ u_safe > b_ub + 1e-3):
                logger.warning("CBF solution violates constraints (numerical tolerance)")
            return u_safe, True
        else:
            # Try to return best feasible solution
            if hasattr(result, 'x') and result.x is not None:
                # Check if result.x is feasible
                if np.all(A_ub @ result.x <= b_ub + 1e-2):
                    return result.x, True
            return u_nom, False
    
    def _safe_normal(self, a: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute unit normal from b to a, handling degenerate case.
        
        For constraint: a must stay at least d away from b.
        Normal n = (a - b) / ||a - b|| points from b toward a.
        """
        diff = a - b
        dist = np.linalg.norm(diff)
        if dist < _EPS:
            # a and b coincide - use arbitrary direction
            return np.array([1.0, 0.0, 0.0])
        return diff / dist
    
    def compute_barrier_values(
        self,
        left_pos: np.ndarray,
        right_pos: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute barrier function values for monitoring/debugging.
        
        Returns dict with h values (positive = safe, negative = unsafe).
        """
        values = {}
        
        # Inter-arm barrier
        d_arm = np.linalg.norm(left_pos - right_pos)
        values['inter_arm'] = d_arm - self.config.inter_arm_safety_margin
        
        # Obstacle barriers (minimum over all obstacles)
        all_obstacles = self.obstacles + self._fixed_obstacles
        if all_obstacles:
            h_left = min(
                np.linalg.norm(left_pos - obs.position) - self.config.obstacle_safety_margin - obs.radius
                for obs in all_obstacles
            )
            h_right = min(
                np.linalg.norm(right_pos - obs.position) - self.config.obstacle_safety_margin - obs.radius
                for obs in all_obstacles
            )
            values['left_obstacle'] = h_left
            values['right_obstacle'] = h_right
        else:
            values['left_obstacle'] = float('inf')
            values['right_obstacle'] = float('inf')
        
        return values


def obstacles_from_point_cloud(
    point_cloud: np.ndarray,
    safety_margin: float = 0.05,
    voxel_size: float = 0.03,
    max_obstacles: int = 20,
) -> List[Obstacle]:
    """
    Create obstacle list from point cloud (e.g., from depth camera).
    
    Downsamples and clusters points to create sphere obstacles.
    
    Args:
        point_cloud: [N, 3] array of 3D points in meters
        safety_margin: Radius for each obstacle
        voxel_size: Voxel size for downsampling
        max_obstacles: Maximum number of obstacles to consider
    
    Returns:
        List of Obstacle objects
    """
    if point_cloud is None or len(point_cloud) == 0:
        return []
    
    point_cloud = np.asarray(point_cloud)
    if point_cloud.shape[1] != 3:
        return []
    
    # Simple voxel downsampling: take centroid of each voxel
    voxels = np.floor(point_cloud / voxel_size).astype(int)
    unique_voxels, inverse = np.unique(voxels, axis=0, return_inverse=True)
    
    obstacles = []
    for i in range(min(len(unique_voxels), max_obstacles)):
        mask = inverse == i
        centroid = np.mean(point_cloud[mask], axis=0)
        obstacles.append(Obstacle(position=centroid, radius=safety_margin))
    
    return obstacles


def obstacles_from_blocks(
    block_poses: List[Any],
    exclude_ids: Optional[set] = None,
    safety_margin: float = 0.05,
) -> List[Obstacle]:
    """
    Create obstacles from detected block poses (e.g., BlockPose from vision).
    
    Excludes blocks that are being manipulated (e.g., currently grasped).
    
    Args:
        block_poses: List of BlockPose or similar with .position and .block_id
        exclude_ids: Block IDs to exclude (e.g., grasped block)
        safety_margin: Safety margin around each block
    
    Returns:
        List of Obstacle objects
    """
    obstacles = []
    exclude_ids = exclude_ids or set()
    
    for bp in block_poses:
        if hasattr(bp, 'block_id') and bp.block_id in exclude_ids:
            continue
        pos = bp.position if hasattr(bp, 'position') else np.array(bp[:3])
        obstacles.append(Obstacle(position=np.asarray(pos), radius=safety_margin))
    
    return obstacles
