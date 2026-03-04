"""
graph_encoder.py

Graph encoding and processing modules for GF-VLA.
Converts scene graphs into embeddings suitable for fusion with VLA features.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple
import numpy as np

from .scene_graph import SceneGraph, InteractionNode, InteractionEdge
from overwatch import initialize_overwatch

overwatch = initialize_overwatch(__name__)


class GraphEncoder(nn.Module):
    """
    Encodes scene graphs into dense feature representations.
    Uses Graph Neural Network (GNN) to process node and edge features.
    """
    
    def __init__(
        self,
        node_feature_dim: int = 128,
        edge_feature_dim: int = 64,
        hidden_dim: int = 256,
        output_dim: int = 512,
        num_layers: int = 3,
    ):
        """
        Args:
            node_feature_dim: Dimension of node features
            edge_feature_dim: Dimension of edge features
            hidden_dim: Hidden dimension for GNN layers
            output_dim: Output dimension of graph embedding
            num_layers: Number of GNN layers
        """
        super().__init__()
        
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Node feature projection (3D position + optional features)
        self.node_encoder = nn.Sequential(
            nn.Linear(3 + node_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Edge feature projection
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
        )
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # Graph-level pooling
        self.graph_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(
        self,
        scene_graph: SceneGraph,
        node_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode a scene graph into a dense feature vector.
        
        Args:
            scene_graph: SceneGraph object
            node_features: Optional pre-computed node features [N, D]
            
        Returns:
            Graph embedding [output_dim]
        """
        num_nodes = len(scene_graph.nodes)
        
        if num_nodes == 0:
            # Return zero embedding for empty graph
            return torch.zeros(self.output_dim, device=next(self.parameters()).device)
        
        # Build node features
        device = next(self.parameters()).device
        if node_features is None:
            node_feats = []
            for node in scene_graph.nodes:
                # Combine position and optional features
                pos_feat = torch.tensor(node.position, dtype=torch.float32, device=device)
                if node.features is not None:
                    feat = torch.tensor(node.features, dtype=torch.float32, device=device)
                    node_feat = torch.cat([pos_feat, feat])
                else:
                    # Use zero features if not provided
                    zero_feat = torch.zeros(self.node_feature_dim, dtype=torch.float32, device=device)
                    node_feat = torch.cat([pos_feat, zero_feat])
                node_feats.append(node_feat)
            
            node_features = torch.stack(node_feats)  # [N, 3 + node_feature_dim]
        
        # Encode nodes
        node_embeddings = self.node_encoder(node_features)  # [N, hidden_dim]
        
        # Build adjacency matrix and edge features
        num_nodes = node_embeddings.shape[0]
        adj_matrix = torch.zeros(num_nodes, num_nodes, device=node_embeddings.device)
        edge_features_list = []
        edge_indices = []
        
        # Create node id to index mapping
        node_id_to_idx = {node.node_id: idx for idx, node in enumerate(scene_graph.nodes)}
        
        for edge in scene_graph.edges:
            src_idx = node_id_to_idx.get(edge.source_id)
            tgt_idx = node_id_to_idx.get(edge.target_id)
            
            if src_idx is not None and tgt_idx is not None:
                adj_matrix[src_idx, tgt_idx] = edge.interaction_strength
                edge_indices.append((src_idx, tgt_idx))
                
                # Create edge feature (could include edge type, strength, etc.)
                edge_feat = torch.tensor([
                    float(edge.edge_type == "hand_object"),
                    float(edge.edge_type == "object_object"),
                    edge.interaction_strength
                ], dtype=torch.float32, device=device)
                # Pad to edge_feature_dim
                if edge_feat.shape[0] < self.edge_feature_dim:
                    padding = torch.zeros(self.edge_feature_dim - edge_feat.shape[0], device=device)
                    edge_feat = torch.cat([edge_feat, padding])
                edge_features_list.append(edge_feat[:self.edge_feature_dim])
        
        if len(edge_features_list) == 0:
            # No edges, use zero edge features
            edge_features = torch.zeros(0, self.edge_feature_dim, device=node_embeddings.device)
        else:
            edge_features = torch.stack(edge_features_list).to(node_embeddings.device)
        
        # Apply GNN layers
        x = node_embeddings
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, adj_matrix, edge_features, edge_indices)
        
        # Graph-level pooling (mean pooling + MLP)
        graph_embedding = x.mean(dim=0)  # [hidden_dim]
        graph_embedding = self.graph_pool(graph_embedding)  # [output_dim]
        
        return graph_embedding


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Network (GAT) layer for processing graph structure.
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        
        self.query = nn.Linear(in_dim, out_dim)
        self.key = nn.Linear(in_dim, out_dim)
        self.value = nn.Linear(in_dim, out_dim)
        self.edge_proj = nn.Linear(64, num_heads)  # Project edge features to attention weights
        
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)
    
    def forward(
        self,
        node_features: torch.Tensor,
        adj_matrix: torch.Tensor,
        edge_features: torch.Tensor,
        edge_indices: List[Tuple[int, int]]
    ) -> torch.Tensor:
        """
        Args:
            node_features: [N, in_dim]
            adj_matrix: [N, N] adjacency matrix
            edge_features: [E, edge_feat_dim] edge features
            edge_indices: List of (src, tgt) tuples
        
        Returns:
            Updated node features [N, out_dim]
        """
        N = node_features.shape[0]
        device = node_features.device
        
        # Multi-head attention
        Q = self.query(node_features)  # [N, out_dim]
        K = self.key(node_features)  # [N, out_dim]
        V = self.value(node_features)  # [N, out_dim]
        
        # Reshape for multi-head attention
        Q = Q.view(N, self.num_heads, self.head_dim)  # [N, H, D_h]
        K = K.view(N, self.num_heads, self.head_dim)  # [N, H, D_h]
        V = V.view(N, self.num_heads, self.head_dim)  # [N, H, D_h]
        
        # Compute attention scores
        scores = torch.einsum('nhd,mhd->nhm', Q, K) / np.sqrt(self.head_dim)  # [N, N, H]
        
        # Add adjacency mask
        adj_mask = adj_matrix.unsqueeze(-1)  # [N, N, 1]
        scores = scores + adj_mask * 10.0  # Large bias for connected nodes
        
        # Add edge feature contributions
        if len(edge_indices) > 0 and edge_features.shape[0] > 0:
            edge_attn = self.edge_proj(edge_features[:, :64])  # [E, H]
            for idx, (src, tgt) in enumerate(edge_indices):
                if idx < edge_attn.shape[0]:
                    scores[src, tgt, :] += edge_attn[idx, :]
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=1)  # [N, N, H]
        
        # Aggregate values
        out = torch.einsum('nhm,mhd->nhd', attn_weights, V)  # [N, H, D_h]
        out = out.reshape(N, self.out_dim)  # [N, out_dim]
        
        # Output projection
        out = self.out_proj(out)
        out = self.layer_norm(out + node_features)  # Residual connection
        
        return out


class TemporalGraphProcessor(nn.Module):
    """
    Processes sequences of scene graphs to extract temporal patterns.
    """
    
    def __init__(
        self,
        graph_embed_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 2,
    ):
        """
        Args:
            graph_embed_dim: Dimension of graph embeddings
            hidden_dim: Hidden dimension for temporal processing
            num_layers: Number of LSTM/transformer layers
        """
        super().__init__()
        
        self.graph_embed_dim = graph_embed_dim
        self.hidden_dim = hidden_dim
        
        # Temporal processing using LSTM
        self.temporal_encoder = nn.LSTM(
            input_size=graph_embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, graph_embed_dim)
    
    def forward(
        self,
        graph_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Process sequence of graph embeddings.
        
        Args:
            graph_embeddings: [T, graph_embed_dim] or [B, T, graph_embed_dim]
            
        Returns:
            Processed embeddings with same shape
        """
        if graph_embeddings.ndim == 2:
            # Add batch dimension
            graph_embeddings = graph_embeddings.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        B, T, D = graph_embeddings.shape
        
        # Process through LSTM
        output, (h_n, c_n) = self.temporal_encoder(graph_embeddings)
        
        # Project output
        output = self.output_proj(output)
        
        if squeeze_output:
            output = output.squeeze(0)
        
        return output

