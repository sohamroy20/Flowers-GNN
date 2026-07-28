"""MeshGraphNets for CylinderFlow — PyTorch port.

Faithful reimplementation of DeepMind's meshgraphnets/cfd_model.py, adapted
to our unified pipeline conventions:
  - Input velocity comes in already normalized (via norm_stats.pt, applied by
    the trainer). MGN's paper normalized internally via an online Normalizer;
    we do it once, statically, upstream. Applied identically to FLOWERS.
  - Target delta is already in batch.y, normalized. MGN's paper built the
    target inside its loss(); we build it in the dataset. Applied identically.
  - The model returns predicted normalized delta directly.

Architecture (unchanged from paper):
  - Node input:  [velocity (2), one_hot(node_type, 9)]           -> 11 dims
  - Edge input:  [rel_pos (2), |rel_pos| (1)]                    ->  3 dims
  - Core:        EncodeProcessDecode(latent=128, mp_steps=15)
  - Output:      2 dims (velocity delta)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

from flowers_gnn.data.cylinder_dataset import NodeType
from flowers_gnn.models.core_gn import EncodeProcessDecode


class MGN(nn.Module):
    """MeshGraphNets for CylinderFlow."""

    def __init__(
        self,
        latent_size: int = 128,
        num_mp_steps: int = 15,
    ):
        super().__init__()

        # Feature dimensions are fixed by the CylinderFlow representation.
        self.node_in_dim = 2 + NodeType.SIZE   # velocity(2) + node_type one-hot(9) = 11
        self.edge_in_dim = 3                   # rel_pos(2) + norm(1) = 3
        self.output_size = 2                   # velocity delta

        self.core = EncodeProcessDecode(
            node_in_dim=self.node_in_dim,
            edge_in_dim=self.edge_in_dim,
            latent_size=latent_size,
            num_mp_steps=num_mp_steps,
            output_size=self.output_size,
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        """Batch in, [total_N, 2] predicted normalized velocity delta out."""
        # ---- Node features ----
        # batch.x is normalized velocity, shape [total_N, 2].
        # node_type is [total_N] long; one-hot to [total_N, NodeType.SIZE].
        node_type_oh = F.one_hot(batch.node_type.long(), num_classes=NodeType.SIZE).float()
        node_feats = torch.cat([batch.x, node_type_oh], dim=-1)   # [total_N, 11]

        # ---- Edge features ----
        # edge_index shape [2, total_E], row 0 senders, row 1 receivers.
        senders, receivers = batch.edge_index[0], batch.edge_index[1]
        rel_pos = batch.pos[senders] - batch.pos[receivers]                       # [total_E, 2]
        rel_norm = torch.norm(rel_pos, dim=-1, keepdim=True)                      # [total_E, 1]
        edge_feats = torch.cat([rel_pos, rel_norm], dim=-1)                       # [total_E, 3]

        # ---- Core graph net ----
        return self.core(node_feats, edge_feats, batch.edge_index)                # [total_N, 2]