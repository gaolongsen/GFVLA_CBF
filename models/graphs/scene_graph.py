"""
scene_graph.py

Information-theoretic scene graph extraction module for GF-VLA.
Implements Shannon entropy-based extraction of task-relevant cues from demonstrations.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from collections import defaultdict

from overwatch import initialize_overwatch

overwatch = initialize_overwatch(__name__)


@dataclass
class InteractionNode:
    """Represents a node in the scene graph (hand or object)."""
    node_id: str
    node_type: str  # 'hand' or 'object'
    position: np.ndarray  # 3D position
    features: Optional[np.ndarray] = None
    timestamp: int = 0


@dataclass
class InteractionEdge:
    """Represents an edge in the scene graph (HO or OO interaction)."""
    source_id: str
    target_id: str
    edge_type: str  # 'hand_object' or 'object_object'
    interaction_strength: float
    timestamp: int = 0


@dataclass
class SceneGraph:
    """Temporally ordered scene graph encoding HO/OO interactions."""
    nodes: List[InteractionNode]
    edges: List[InteractionEdge]
    timestamp: int
    graph_id: Optional[str] = None


class InformationTheoreticGraphBuilder:
    """
    Information-theoretic approach to extract task-relevant cues using Shannon entropy.
    
    Based on the paper's equation (1):
    H^X(p) = -ε Σ p(x_i) · ln p(x_i)
    """
    
    def __init__(
        self,
        temporal_window_size: int = 10,
        epsilon: float = 1.0,
        entropy_threshold: float = 0.5,
    ):
        """
        Args:
            temporal_window_size: Size of sliding temporal window φ
            epsilon: Constant value ε ∈ (0, 1) from equation (1)
            entropy_threshold: Threshold for identifying active regions
        """
        self.temporal_window_size = temporal_window_size
        self.epsilon = epsilon
        self.entropy_threshold = entropy_threshold
    
    def compute_shannon_entropy(
        self,
        signal: np.ndarray,
        temporal_window: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute Shannon entropy for a signal over temporal windows.
        
        Args:
            signal: Input signal of shape [T, ...] where T is time steps
            temporal_window: Window size (defaults to self.temporal_window_size)
            
        Returns:
            Entropy values for each time step
        """
        if temporal_window is None:
            temporal_window = self.temporal_window_size
        
        T = signal.shape[0]
        entropy_values = np.zeros(T)
        
        # Sliding window approach
        half_window = temporal_window // 2
        
        for t in range(T):
            # Define window bounds
            window_start = max(0, t - half_window)
            window_end = min(T, t + half_window + 1)
            
            # Extract window data
            window_data = signal[window_start:window_end]
            
            # Flatten spatial dimensions if present
            if window_data.ndim > 1:
                window_data = window_data.reshape(window_data.shape[0], -1)
            
            # Compute probability distribution
            # For continuous signals, we discretize into bins
            n_bins = min(50, len(window_data) // 2)  # Adaptive binning
            if n_bins < 2:
                entropy_values[t] = 0.0
                continue
            
            # Histogram-based probability estimation
            hist, bin_edges = np.histogram(window_data.flatten(), bins=n_bins)
            probs = hist / (hist.sum() + 1e-8)  # Normalize to probabilities
            
            # Remove zero probabilities for entropy calculation
            probs = probs[probs > 0]
            
            # Compute Shannon entropy: H^X(p) = -ε Σ p(x_i) · ln p(x_i)
            entropy = -self.epsilon * np.sum(probs * np.log(probs + 1e-8))
            entropy_values[t] = entropy
        
        return entropy_values
    
    def extract_active_regions(
        self,
        positions: np.ndarray,
        velocities: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Extract active regions (regions with significant dynamical variation).
        
        Args:
            positions: Position data of shape [T, N, 3] where N is number of entities
            velocities: Optional velocity data of shape [T, N, 3]
            
        Returns:
            Binary mask indicating active regions [T, N]
        """
        T, N = positions.shape[:2]
        active_mask = np.zeros((T, N), dtype=bool)
        
        # Compute velocities if not provided
        if velocities is None:
            velocities = np.diff(positions, axis=0, prepend=positions[0:1])
        
        # Compute entropy for each entity
        for n in range(N):
            # Use velocity magnitude as signal
            vel_magnitude = np.linalg.norm(velocities[:, n, :], axis=1)
            entropy = self.compute_shannon_entropy(vel_magnitude[:, None])
            
            # Threshold to identify active regions
            active_mask[:, n] = entropy > self.entropy_threshold
        
        return active_mask
    
    def detect_hand_object_interactions(
        self,
        hand_positions: np.ndarray,
        object_positions: np.ndarray,
        active_mask: Optional[np.ndarray] = None,
        distance_threshold: float = 0.1
    ) -> List[Tuple[int, int, int]]:
        """
        Detect hand-object (HO) interactions based on proximity and activity.
        
        Args:
            hand_positions: Hand positions [T, H, 3] where H is number of hands
            object_positions: Object positions [T, O, 3] where O is number of objects
            active_mask: Optional active region mask
            distance_threshold: Maximum distance for interaction detection
            
        Returns:
            List of (timestamp, hand_idx, object_idx) tuples
        """
        T, H = hand_positions.shape[:2]
        O = object_positions.shape[1]
        
        interactions = []
        
        for t in range(T):
            for h in range(H):
                hand_pos = hand_positions[t, h]
                
                for o in range(O):
                    obj_pos = object_positions[t, o]
                    
                    # Check distance
                    distance = np.linalg.norm(hand_pos - obj_pos)
                    
                    if distance < distance_threshold:
                        # Check if region is active (if mask provided)
                        if active_mask is None or active_mask[t, o]:
                            interactions.append((t, h, o))
        
        return interactions
    
    def detect_object_object_interactions(
        self,
        object_positions: np.ndarray,
        active_mask: Optional[np.ndarray] = None,
        distance_threshold: float = 0.15
    ) -> List[Tuple[int, int, int]]:
        """
        Detect object-object (OO) interactions based on proximity and activity.
        
        Args:
            object_positions: Object positions [T, O, 3]
            active_mask: Optional active region mask
            distance_threshold: Maximum distance for interaction detection
            
        Returns:
            List of (timestamp, obj_idx_1, obj_idx_2) tuples
        """
        T, O = object_positions.shape[:2]
        
        interactions = []
        
        for t in range(T):
            for o1 in range(O):
                for o2 in range(o1 + 1, O):
                    obj1_pos = object_positions[t, o1]
                    obj2_pos = object_positions[t, o2]
                    
                    # Check distance
                    distance = np.linalg.norm(obj1_pos - obj2_pos)
                    
                    if distance < distance_threshold:
                        # Check if at least one region is active
                        if active_mask is None or (active_mask[t, o1] or active_mask[t, o2]):
                            interactions.append((t, o1, o2))
        
        return interactions


class SceneGraphExtractor:
    """
    Main class for extracting temporally ordered scene graphs from demonstrations.
    """
    
    def __init__(
        self,
        temporal_window_size: int = 10,
        epsilon: float = 1.0,
        entropy_threshold: float = 0.5,
        ho_distance_threshold: float = 0.1,
        oo_distance_threshold: float = 0.15,
    ):
        """
        Args:
            temporal_window_size: Size of sliding temporal window
            epsilon: Constant for entropy calculation
            entropy_threshold: Threshold for active region detection
            ho_distance_threshold: Distance threshold for hand-object interactions
            oo_distance_threshold: Distance threshold for object-object interactions
        """
        self.graph_builder = InformationTheoreticGraphBuilder(
            temporal_window_size=temporal_window_size,
            epsilon=epsilon,
            entropy_threshold=entropy_threshold,
        )
        self.ho_distance_threshold = ho_distance_threshold
        self.oo_distance_threshold = oo_distance_threshold
    
    def extract_from_demonstration(
        self,
        hand_positions: np.ndarray,
        object_positions: np.ndarray,
        hand_features: Optional[np.ndarray] = None,
        object_features: Optional[np.ndarray] = None,
    ) -> List[SceneGraph]:
        """
        Extract temporally ordered scene graphs from a demonstration.
        
        Args:
            hand_positions: Hand positions [T, H, 3]
            object_positions: Object positions [T, O, 3]
            hand_features: Optional hand features [T, H, D]
            object_features: Optional object features [T, O, D]
            
        Returns:
            List of SceneGraph objects, one per time step
        """
        T = hand_positions.shape[0]
        H = hand_positions.shape[1]
        O = object_positions.shape[1]
        
        # Extract active regions using information theory
        all_positions = np.concatenate([hand_positions, object_positions], axis=1)
        active_mask = self.graph_builder.extract_active_regions(all_positions)
        
        # Split mask for hands and objects
        hand_active = active_mask[:, :H]
        object_active = active_mask[:, H:]
        
        # Detect interactions
        ho_interactions = self.graph_builder.detect_hand_object_interactions(
            hand_positions,
            object_positions,
            active_mask=object_active,
            distance_threshold=self.ho_distance_threshold
        )
        
        oo_interactions = self.graph_builder.detect_object_object_interactions(
            object_positions,
            active_mask=object_active,
            distance_threshold=self.oo_distance_threshold
        )
        
        # Build scene graphs for each time step
        scene_graphs = []
        
        for t in range(T):
            nodes = []
            edges = []
            
            # Add hand nodes
            for h in range(H):
                node = InteractionNode(
                    node_id=f"hand_{h}",
                    node_type="hand",
                    position=hand_positions[t, h],
                    features=hand_features[t, h] if hand_features is not None else None,
                    timestamp=t
                )
                nodes.append(node)
            
            # Add object nodes
            for o in range(O):
                node = InteractionNode(
                    node_id=f"object_{o}",
                    node_type="object",
                    position=object_positions[t, o],
                    features=object_features[t, o] if object_features is not None else None,
                    timestamp=t
                )
                nodes.append(node)
            
            # Add HO edges
            for timestamp, hand_idx, obj_idx in ho_interactions:
                if timestamp == t:
                    edge = InteractionEdge(
                        source_id=f"hand_{hand_idx}",
                        target_id=f"object_{obj_idx}",
                        edge_type="hand_object",
                        interaction_strength=1.0,  # Could be computed from distance/entropy
                        timestamp=t
                    )
                    edges.append(edge)
            
            # Add OO edges
            for timestamp, obj1_idx, obj2_idx in oo_interactions:
                if timestamp == t:
                    edge = InteractionEdge(
                        source_id=f"object_{obj1_idx}",
                        target_id=f"object_{obj2_idx}",
                        edge_type="object_object",
                        interaction_strength=1.0,
                        timestamp=t
                    )
                    edges.append(edge)
            
            scene_graph = SceneGraph(
                nodes=nodes,
                edges=edges,
                timestamp=t,
                graph_id=f"graph_{t}"
            )
            scene_graphs.append(scene_graph)
        
        return scene_graphs

