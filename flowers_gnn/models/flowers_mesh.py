"""Native-mesh FLOWERS:

Same input/output contract as MGN (see mgn.py):
  - Batch in.
  - [total_N, 2] predicted normalized velocity delta out.

Architecture:
  1. Node input features = [velocity(2), one_hot(node_type, 9)] -> 11 dims.
     Same as MGN, for a fair comparison.
  2. Linear lift 11 -> latent_dim.
  3. N SELFWARP-mesh blocks stacked flat (no U-Net, no pooling, one resolution).
     This is the flat-stack variant mirrors MGN's
     structural choice of 15 message-passing steps at one resolution.
  4. Linear projection latent_dim -> 2 = velocity delta.

Key differences from MGN's architecture:
  - No explicit edge features. FLOWERS uses positions (via warp queries), not
    connectivity, to route information. edge_index is not consumed by any block.
  - Non-locality is per-block, unbounded (a warp can point anywhere in the
    domain). MGN needs many message-passing hops to move info the same distance.
  - Multi-head structure. Each block has num_heads independent warps.

Key preservations from grid-FLOWERS:
  - The warp/pullback mechanism itself.
  - Multi-head SELFWARP block structure with FFN sublayer and residuals.
  - Pre-norm transformer-style layout.
The only substitution is grid_sample -> knn_interpolate.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

from flowers_gnn.data.cylinder_dataset import NodeType
from flowers_gnn.models.selfwarp_mesh import SelfWarpMesh


class FlowersMesh(nn.Module):
    """Native-mesh FLOWERS for CylinderFlow."""

    def __init__(
        self,
        latent_dim: int = 192,
        num_blocks: int = 8,
        num_heads: int = 8,
        k_interp: int = 3,
    ):
        super().__init__()

        # Input features match MGN's node inputs: [velocity, one_hot(node_type)]
        self.node_in_dim = 2 + NodeType.SIZE   # velocity(2) + node_type OH(9) = 11
        self.output_size = 2                   # velocity delta

        # Lift 11 -> latent_dim
        self.lift = nn.Linear(self.node_in_dim, latent_dim)

        # Stack of SELFWARP-mesh blocks
        self.blocks = nn.ModuleList([
            SelfWarpMesh(
                latent_dim=latent_dim,
                num_heads=num_heads,
                k_interp=k_interp,
            )
            for _ in range(num_blocks)
        ])

        # Project latent_dim -> 2 (predicted delta)
        self.project = nn.Linear(latent_dim, self.output_size)

    def forward(self, batch: Batch) -> torch.Tensor:
        """Batch in, [total_N, 2] predicted normalized velocity delta out."""
        #Build initial node features
        node_type_oh = F.one_hot(
            batch.node_type.long(), num_classes=NodeType.SIZE
        ).float()
        node_feats = torch.cat([batch.x, node_type_oh], dim=-1)   # [N, 11]

        #Lift
        h = self.lift(node_feats)                                 # [N,latent_dim]

        # Stack of SELFWARP blocks
        # Positions are static per-batch and get passed to every block for
        # query point computation. batch.batch tells knn_interpolate which
        # nodes belong to which graph (critical for disjoint-batched graphs).
        pos = batch.pos                                           # [N, 2]
        batch_idx = batch.batch                                   # [N]
        for block in self.blocks:
            h = block(h, pos, batch_idx)                          # [N, latent_dim]

        # Project to velocity delta
        return self.project(h)                                    # [N, 2]