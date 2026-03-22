"""
graph_encoder.py

Graph encoding and processing modules for GF-VLA.
Converts scene graphs into embeddings suitable for fusion with VLA features.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .scene_graph import SceneGraph
from overwatch import initialize_overwatch

overwatch = initialize_overwatch(__name__)

# Large negative for masked attention (softmax-stable; -inf can cause NaN in edge cases)
_ATTN_MASK_FILL = -1e4


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

        # GNN layers (edge_feat_dim must match GraphEncoder.edge_feature_dim)
        self.gnn_layers = nn.ModuleList(
            [
                GraphAttentionLayer(
                    hidden_dim,
                    hidden_dim,
                    num_heads=4,
                    edge_feat_dim=edge_feature_dim,
                )
                for _ in range(num_layers)
            ]
        )

        # Graph-level pooling
        self.graph_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def _build_node_features_tensor(
        self,
        scene_graph: SceneGraph,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Stack node features on CPU numpy then one GPU transfer (faster than per-node tensor)."""
        nodes = scene_graph.nodes
        n = len(nodes)
        feat_dim = self.node_feature_dim
        pos = np.empty((n, 3), dtype=np.float32)
        extra = np.zeros((n, feat_dim), dtype=np.float32)
        for i, node in enumerate(nodes):
            pos[i] = np.asarray(node.position, dtype=np.float32).reshape(3)
            if node.features is not None:
                f = np.asarray(node.features, dtype=np.float32).ravel()
                if f.size >= feat_dim:
                    extra[i] = f[:feat_dim]
                elif f.size > 0:
                    extra[i, : f.size] = f
        pos_t = torch.from_numpy(pos).to(device=device, dtype=dtype)
        extra_t = torch.from_numpy(extra).to(device=device, dtype=dtype)
        return torch.cat([pos_t, extra_t], dim=-1)

    def _build_adjacency_and_edges(
        self,
        scene_graph: SceneGraph,
        num_nodes: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            adj_matrix [N, N]
            edge_features [E, edge_feature_dim]
            edge_src [E], edge_tgt [E] long tensors for vectorized scatter
        """
        adj_matrix = torch.zeros(num_nodes, num_nodes, device=device, dtype=dtype)
        node_id_to_idx = {node.node_id: idx for idx, node in enumerate(scene_graph.nodes)}
        ef_dim = self.edge_feature_dim

        src_list: List[int] = []
        tgt_list: List[int] = []
        edge_rows: List[np.ndarray] = []

        for edge in scene_graph.edges:
            src_idx = node_id_to_idx.get(edge.source_id)
            tgt_idx = node_id_to_idx.get(edge.target_id)
            if src_idx is None or tgt_idx is None:
                continue
            adj_matrix[src_idx, tgt_idx] = edge.interaction_strength
            src_list.append(src_idx)
            tgt_list.append(tgt_idx)
            row = np.zeros(ef_dim, dtype=np.float32)
            row[0] = float(edge.edge_type == "hand_object")
            row[1] = float(edge.edge_type == "object_object")
            row[2] = float(edge.interaction_strength)
            edge_rows.append(row)

        if not edge_rows:
            empty_ef = torch.zeros(0, ef_dim, device=device, dtype=dtype)
            empty_idx = torch.zeros(0, dtype=torch.long, device=device)
            return adj_matrix, empty_ef, empty_idx, empty_idx

        edge_np = np.stack(edge_rows, axis=0)
        edge_features = torch.from_numpy(edge_np).to(device=device, dtype=dtype)
        edge_src = torch.tensor(src_list, dtype=torch.long, device=device)
        edge_tgt = torch.tensor(tgt_list, dtype=torch.long, device=device)
        return adj_matrix, edge_features, edge_src, edge_tgt

    def forward(
        self,
        scene_graph: SceneGraph,
        node_features: Optional[torch.Tensor] = None,
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
            p = next(self.parameters())
            return torch.zeros(self.output_dim, device=p.device, dtype=p.dtype)

        p = next(self.parameters())
        device, dtype = p.device, p.dtype

        if node_features is None:
            node_features = self._build_node_features_tensor(scene_graph, device, dtype)
        else:
            node_features = node_features.to(device=device, dtype=dtype)

        node_embeddings = self.node_encoder(node_features)
        n = node_embeddings.shape[0]

        adj_matrix, edge_features, edge_src, edge_tgt = self._build_adjacency_and_edges(
            scene_graph, n, device, dtype
        )

        x = node_embeddings
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, adj_matrix, edge_features, edge_src, edge_tgt)

        graph_embedding = x.mean(dim=0)
        graph_embedding = self.graph_pool(graph_embedding)

        return graph_embedding


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Network (GAT) layer with masked neighborhood attention.
    Only attends over self + nodes with positive adjacency (sparse, stable softmax).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        edge_feat_dim: int = 64,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.edge_feat_dim = edge_feat_dim

        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"

        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.query = nn.Linear(in_dim, out_dim)
        self.key = nn.Linear(in_dim, out_dim)
        self.value = nn.Linear(in_dim, out_dim)
        self.edge_proj = nn.Linear(edge_feat_dim, num_heads)
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)
        self.attn_dropout = nn.Dropout(attn_dropout) if attn_dropout > 0 else None

    def forward(
        self,
        node_features: torch.Tensor,
        adj_matrix: torch.Tensor,
        edge_features: torch.Tensor,
        edge_src: torch.Tensor,
        edge_tgt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            node_features: [N, in_dim]
            adj_matrix: [N, N] adjacency (non-zero = edge)
            edge_features: [E, edge_feat_dim]
            edge_src, edge_tgt: [E] indices into nodes

        Returns:
            Updated node features [N, out_dim]
        """
        N = node_features.shape[0]
        device = node_features.device
        dtype = node_features.dtype

        Q = self.query(node_features)
        K = self.key(node_features)
        V = self.value(node_features)

        Q = Q.view(N, self.num_heads, self.head_dim)
        K = K.view(N, self.num_heads, self.head_dim)
        V = V.view(N, self.num_heads, self.head_dim)

        # [N, N, H] — pairwise dot-product logits per head
        scores = torch.einsum("nhd,mhd->nhm", Q, K) * self.scale

        # Bias for edge strength (optional signal)
        scores = scores + adj_matrix.unsqueeze(-1) * 10.0

        # Mask: attend only to self + neighbors (stable vs dense softmax over unrelated nodes)
        self_loops = torch.eye(N, device=device, dtype=torch.bool)
        has_edge = adj_matrix > 1e-8
        allowed = self_loops | has_edge
        scores = scores.masked_fill(~allowed.unsqueeze(-1), _ATTN_MASK_FILL)

        # Vectorized edge feature bias (replaces Python loop over edges)
        if edge_features.numel() > 0 and edge_src.numel() > 0:
            edge_attn = self.edge_proj(edge_features)
            e = min(edge_src.shape[0], edge_attn.shape[0])
            if e > 0:
                scores[edge_src[:e], edge_tgt[:e], :] = scores[
                    edge_src[:e], edge_tgt[:e], :
                ] + edge_attn[:e]

        attn_weights = F.softmax(scores, dim=1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0, posinf=0.0, neginf=0.0)
        if self.attn_dropout is not None and self.training:
            attn_weights = self.attn_dropout(attn_weights)

        out = torch.einsum("nhm,mhd->nhd", attn_weights, V)
        out = out.reshape(N, self.out_dim)

        out = self.out_proj(out)
        out = self.layer_norm(out + node_features)

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
        dropout: float = 0.0,
    ):
        """
        Args:
            graph_embed_dim: Dimension of graph embeddings
            hidden_dim: Hidden dimension for temporal processing
            num_layers: Number of LSTM layers
            dropout: Dropout on LSTM outputs (training stability)
        """
        super().__init__()

        self.graph_embed_dim = graph_embed_dim
        self.hidden_dim = hidden_dim

        self.temporal_encoder = nn.LSTM(
            input_size=graph_embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 and dropout > 0 else 0.0,
        )

        self.output_proj = nn.Linear(hidden_dim, graph_embed_dim)
        self.out_dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(
        self,
        graph_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process sequence of graph embeddings.

        Args:
            graph_embeddings: [T, graph_embed_dim] or [B, T, graph_embed_dim]

        Returns:
            Processed embeddings with same shape
        """
        if graph_embeddings.ndim == 2:
            graph_embeddings = graph_embeddings.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        output, _ = self.temporal_encoder(graph_embeddings)
        if self.out_dropout is not None and self.training:
            output = self.out_dropout(output)
        output = self.output_proj(output)

        if squeeze_output:
            output = output.squeeze(0)

        return output
