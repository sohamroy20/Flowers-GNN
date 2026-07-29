"""Mesh-native SELFWARP block: the core of FLOWERS adapted to irregular meshes.

Faithful adaptation of the SelfWarp block from Muser et al. 2026's Flower model
(grid version) to operate directly on mesh nodes instead of a regular grid.

The mechanism is unchanged:
    1. Predict a per-node displacement rho(x) with a small MLP.
    2. Predict a per-node value v(x) with a small MLP.
    3. For each head, compute a query point q = pos + rho.
    4. Sample the value at q by interpolating from nearby mesh nodes.
    5. Concatenate heads, residual, norm.

The key substantive change from grid FLOWERS is step 4:
    - Grid version:  bilinear grid_sample from surrounding grid cells.
    - Mesh version:  knn_interpolate from surrounding mesh nodes (k=3, IDW).

Everything else (the flow head, value head, multi-head structure, residual
connections, LayerNorms, FFN) mirrors flowers/flowers/models/flower_layers.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Batch
#from torch_geometric.nn import knn_interpolate

def _knn_interpolate(
    x: torch.Tensor,           # [N_src, C] source features
    pos_src: torch.Tensor,     # [N_src, 2] source positions
    pos_qry: torch.Tensor,     # [N_qry, 2] query positions
    batch_idx: torch.Tensor,   # [N] which graph each node belongs to (shared for src and qry since they're the same nodes)
    k: int = 3,
    eps: float = 1e-8,
) -> torch.Tensor:
    """kNN interpolation with inverse-distance weighting.

    For each query point, find k nearest source points *within the same graph*
    and take a weighted average of their features. Weights are 1 / distance.

    Same behavior as torch_geometric.nn.knn_interpolate for the case where
    source and query nodes are identical (which is our situation).

    We do the graph-aware k-NN by masking distances between different graphs to
    infinity before top-k. Simple, correct, differentiable, no external deps.
    """
    # Pairwise distances: [N_qry, N_src]
    dist = torch.cdist(pos_qry, pos_src)                                # [N, N]

    # Mask out cross-graph pairs by setting their distance to +inf.
    # batch_idx[:, None] gives [N, 1] for queries; [None, :] gives [1, N] for sources.
    same_graph = (batch_idx.unsqueeze(1) == batch_idx.unsqueeze(0))     # [N, N] bool
    dist = dist.masked_fill(~same_graph, float("inf"))

    # k nearest neighbors per query
    knn_dist, knn_idx = torch.topk(dist, k=k, dim=-1, largest=False)    # [N, k]

    # Inverse-distance weights (with epsilon to avoid div-by-zero when a query
    # coincides with a source point, e.g. displacement rho=0 at some node).
    w = 1.0 / (knn_dist + eps)                                          # [N, k]
    w = w / w.sum(dim=-1, keepdim=True)                                 # normalize

    # Gather source features at the k neighbors and take weighted sum
    x_neighbors = x[knn_idx]                                            # [N, k, C]
    out = (w.unsqueeze(-1) * x_neighbors).sum(dim=1)                    # [N, C]
    return out


class MLP(nn.Module):
    """2-layer MLP with GELU activation, matching grid-FLOWERS's block MLPs.. MGN and grid-FLOWERS use MLPs of the same general shape"""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SelfWarpMesh(nn.Module):
    """One SELFWARP block, mesh-native.

    Args:
        latent_dim: channel dim of node features throughout the block.
        num_heads:  number of warp heads. Each head sees latent_dim/num_heads
                    channels and predicts its own 2D displacement.
        k_interp:   number of nearest mesh neighbours used to sample value
                    features at each displaced query point. Standard IDW.
    """

    def __init__(
        self,
        latent_dim: int,
        num_heads: int = 8,
        k_interp: int = 3,
    ):
        super().__init__()
        if latent_dim % num_heads != 0:
            raise ValueError(
                f"latent_dim ({latent_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.head_dim = latent_dim // num_heads
        self.k_interp = k_interp

        # Pre-norm on the input (transformer-style)
        self.norm_pre_warp = nn.LayerNorm(latent_dim)

        # Value head: per-node MLP producing v(x)
        self.value_head = MLP(latent_dim, latent_dim, latent_dim)

        # Flow head: per-node MLP producing 2D displacement per head
        # Output shape [N, num_heads * 2] -> reshape to [N, H, 2]
        self.flow_head = MLP(latent_dim, latent_dim, num_heads * 2)

        # Output projection back to latent_dim after concatenating heads
        self.out_proj = nn.Linear(latent_dim, latent_dim)

        # FFN sublayer (transformer-style)
        self.norm_pre_ffn = nn.LayerNorm(latent_dim)
        self.ffn = MLP(latent_dim, latent_dim * 2, latent_dim)

    def forward(
        self,
        h: torch.Tensor,           # [total_N, latent_dim] node features
        pos: torch.Tensor,         # [total_N, 2] node positions
        batch_idx: torch.Tensor,   # [total_N] which graph each node belongs to
    ) -> torch.Tensor:
        # ---- Warp sublayer ----
        residual = h
        x = self.norm_pre_warp(h)

        # Compute value features (one shared v per node, split across heads below)
        v = self.value_head(x)                                # [N, latent_dim]

        # Predict per-head displacement, reshape to [N, H, 2]
        flow = self.flow_head(x).view(-1, self.num_heads, 2)  # [N, H, 2]

        # Split value into per-head slices: [N, H, head_dim]
        v_heads = v.view(-1, self.num_heads, self.head_dim)   # [N, H, head_dim]

        # For each head, compute query points and interpolate.
        # We do heads in a loop for clarity; each head is cheap.
        warped_heads = []
        for h_idx in range(self.num_heads):
            q = pos + flow[:, h_idx]                          # [N, 2] query pts
            # knn_interpolate wants:
            #   x: source features [source_N, C]
            #   pos_x: source positions [source_N, 2]
            #   pos_y: query positions [query_N, 2]
            #   batch_x, batch_y: which graph each src/query belongs to (for
            #     batched disjoint graphs — critical so head-i of graph 0 never
            #     samples from graph 1).
            warped = _knn_interpolate(
                x=v_heads[:, h_idx, :],
                pos_src=pos,
                pos_qry=q,
                batch_idx=batch_idx,
                k=self.k_interp,
            )                                             # [N, head_dim]
            warped_heads.append(warped)

        # Concatenate heads: [N, H, head_dim] -> [N, latent_dim]
        out = torch.stack(warped_heads, dim=1).view(-1, self.latent_dim)
        out = self.out_proj(out)

        # Residual
        h = residual + out

        # ---- FFN sublayer (transformer-style, second residual + norm) ----
        residual = h
        x = self.norm_pre_ffn(h)
        h = residual + self.ffn(x)

        return h
