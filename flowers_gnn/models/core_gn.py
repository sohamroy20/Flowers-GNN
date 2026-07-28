"""Generic Encode-Process-Decode graph net.

Faithful PyTorch port of DeepMind's meshgraphnets/core_model.py.
Kept task-agnostic: it operates on generic node + edge tensors and knows
nothing about CylinderFlow, velocity, or node types. Feature construction
lives in the task-specific file (mgn.py for CylinderFlow).

Architecture:
  Encoder: two 2-layer LayerNorm-MLPs (one for nodes, one for edges) that
           lift raw features to `latent_size`.
  Processor: N GraphNetBlocks. Each block has ITS OWN edge and node MLPs
             (no weight sharing across blocks) and applies residual updates
             to both edge and node latents. Aggregation is sum, matching MGN.
  Decoder: one 2-layer MLP (no LayerNorm) mapping node latents to `output_size`.

The residual message-passing block (each of N):
  e_ij <- e_ij + MLP_edge^l ( [ v_sender, v_receiver, e_ij ] )
  v_i  <- v_i  + MLP_node^l ( [ v_i, sum_{j: r(j)=i} e_ij ] )
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
#from torch_scatter import scatter_add


def build_mlp(in_dim: int, hidden_dim: int, out_dim: int, layer_norm: bool = True) -> nn.Sequential:
    """2-layer MLP with ReLU activations, optional LayerNorm on the output.

    Matches MGN's `_make_mlp`: two hidden layers of `hidden_dim`, then a
    linear to `out_dim`, then (optionally) LayerNorm. No activation after
    the final linear.
    """
    layers: List[nn.Module] = [
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    ]
    if layer_norm:
        layers.append(nn.LayerNorm(out_dim))
    return nn.Sequential(*layers)


class GraphNetBlock(nn.Module):
    """One message-passing block: edge update then node update, both residual.

    Own MLPs — this block does NOT share weights with any other block.
    """

    def __init__(self, latent_size: int):
        super().__init__()
        # Edge MLP input = [v_sender, v_receiver, e_ij] = 3 * latent_size
        self.edge_mlp = build_mlp(
            in_dim=3 * latent_size,
            hidden_dim=latent_size,
            out_dim=latent_size,
            layer_norm=True,
        )
        # Node MLP input = [v_i, aggregated_edges_i] = 2 * latent_size
        self.node_mlp = build_mlp(
            in_dim=2 * latent_size,
            hidden_dim=latent_size,
            out_dim=latent_size,
            layer_norm=True,
        )

    def forward(
        self,
        node_feats: torch.Tensor,     # [total_N, latent]
        edge_feats: torch.Tensor,     # [total_E, latent]
        edge_index: torch.Tensor,     # [2, total_E], (senders, receivers)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        senders, receivers = edge_index[0], edge_index[1]

        # ---- Edge update ----
        e_input = torch.cat(
            [node_feats[senders], node_feats[receivers], edge_feats],
            dim=-1,
        )
        edge_update = self.edge_mlp(e_input)
        new_edge_feats = edge_feats + edge_update       # residual

        # ---- Node update ----
        # Sum incoming updated edges at each receiver.
        num_nodes = node_feats.shape[0]
        #agg = scatter_add(new_edge_feats, receivers, dim=0, dim_size=num_nodes)
        agg = torch.zeros(num_nodes, new_edge_feats.shape[-1],
                          device=new_edge_feats.device, dtype=new_edge_feats.dtype)
        agg.index_add_(0, receivers, new_edge_feats)
        n_input = torch.cat([node_feats, agg], dim=-1)
        node_update = self.node_mlp(n_input)
        new_node_feats = node_feats + node_update       # residual

        return new_node_feats, new_edge_feats


class EncodeProcessDecode(nn.Module):
    """Full graph net: encode raw features -> N processor blocks -> decode nodes."""

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        latent_size: int = 128,
        num_mp_steps: int = 15,
        output_size: int = 2,
    ):
        super().__init__()
        # Encoders lift raw inputs to latent size
        self.node_encoder = build_mlp(node_in_dim, latent_size, latent_size, layer_norm=True)
        self.edge_encoder = build_mlp(edge_in_dim, latent_size, latent_size, layer_norm=True)

        # Processor: N blocks, each with its own weights
        self.blocks = nn.ModuleList([GraphNetBlock(latent_size) for _ in range(num_mp_steps)])

        # Decoder: no LayerNorm (MGN's convention — the decoder produces the raw output)
        self.decoder = build_mlp(latent_size, latent_size, output_size, layer_norm=False)

    def forward(
        self,
        node_feats: torch.Tensor,    # [total_N, node_in_dim]
        edge_feats: torch.Tensor,    # [total_E, edge_in_dim]
        edge_index: torch.Tensor,    # [2, total_E]
    ) -> torch.Tensor:
        v = self.node_encoder(node_feats)
        e = self.edge_encoder(edge_feats)
        for block in self.blocks:
            v, e = block(v, e, edge_index)
        return self.decoder(v)   # [total_N, output_size]