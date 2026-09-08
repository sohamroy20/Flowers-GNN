"""Mesh-native SELFWARP block: the core of FLOWERS adapted to irregular meshes.

Faithful adaptation of the SelfWarp block from Muser et al. 2026's Flower model
(grid version) to operate directly on mesh nodes instead of a regular grid.

The mechanism is unchanged:
    1. Predict per-node value features v(x) with an MLP.
    2. Predict per-head, per-node displacements rho(x) with an MLP.
    3. For each head, compute query points q = pos + rho, and sample the
       head-specific value features at q via k-NN interpolation.
    4. Concatenate heads, project, residual, norm, FFN.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Batch


def _knn_interpolate_multihead(
    v_heads: torch.Tensor,     # [N, H, head_dim]  per-head source values
    pos: torch.Tensor,         # [N, 2]            source positions
    flow: torch.Tensor,        # [N, H, 2]         per-head displacements
    batch_idx: torch.Tensor,   # [N]               graph id per node
    k: int = 3,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Multi-head kNN interpolation: vectorized over heads, looped over graphs.

    Heads are vectorized (no Python loop over H), but distances are computed
    within each graph only, so we never evaluate the cross-graph pairs that a
    global [N*H, N] cdist would compute and then mask away.

    Neighbour selection is discrete, so it runs under no_grad; the k selected
    distances are then recomputed with an exact norm (torch.cdist uses the
    expanded form and loses precision at mesh-scale separations).
    """
    N, H, head_dim = v_heads.shape
    out = torch.empty(N, H, head_dim, device=v_heads.device, dtype=v_heads.dtype)

    for b in torch.unique(batch_idx):
        idx = (batch_idx == b).nonzero(as_tuple=True)[0]        # [n_b]
        n_b = idx.shape[0]

        pos_b = pos[idx]                                         # [n_b, 2]
        flow_b = flow[idx]                                       # [n_b, H, 2]
        v_b = v_heads[idx]                                       # [n_b, H, head_dim]

        q = (pos_b.unsqueeze(1) + flow_b).reshape(n_b * H, 2)    # [n_b*H, 2]

        with torch.no_grad():
            d = torch.cdist(q, pos_b)                            # [n_b*H, n_b]
            _, knn_idx = torch.topk(d, k=k, dim=-1, largest=False)

        knn_dist = (q.unsqueeze(1) - pos_b[knn_idx]).norm(dim=-1)   # [n_b*H, k]
        w = 1.0 / knn_dist.clamp_min(eps)
        w = w / w.sum(dim=-1, keepdim=True)

        knn_idx_r = knn_idx.view(n_b, H, k)
        head_ar = torch.arange(H, device=pos.device).view(1, H, 1).expand(n_b, H, k)
        neighbour_vals = v_b[knn_idx_r, head_ar]                 # [n_b, H, k, head_dim]

        out[idx] = (w.view(n_b, H, k, 1) * neighbour_vals).sum(dim=2)

    return out


def _knn_interpolate(
    x: torch.Tensor,
    pos_src: torch.Tensor,
    pos_qry: torch.Tensor,
    batch_idx: torch.Tensor,
    k: int = 3,
    eps: float = 1e-6, #earler 1e-8
) -> torch.Tensor:
    """kNN interpolation with inverse-distance weighting, per-graph.

    Memory-efficient: iterates over graphs, computes distances within each graph.
    Peak memory: O(N_graph_max^2) per iteration.
    """
    out = torch.zeros_like(x)
    unique_batches = torch.unique(batch_idx)

    for b in unique_batches:
        mask = (batch_idx == b)
        idx = mask.nonzero(as_tuple=True)[0]

        pos_src_b = pos_src[idx]
        pos_qry_b = pos_qry[idx]
        x_b = x[idx]

       # dist = torch.cdist(pos_qry_b, pos_src_b)

       # knn_dist, knn_idx = torch.topk(dist, k=k, dim=-1, largest=False)

       # w = 1.0 / (knn_dist + eps)
        #
        dist = torch.cdist(pos_qry_b, pos_src_b)

        _, knn_idx = torch.topk(dist, k=k, dim=-1, largest=False)

        # cdist uses the expanded form and loses precision at small distances,
        # so recompute the k selected distances exactly.
        knn_dist = (pos_qry_b.unsqueeze(1) - pos_src_b[knn_idx]).norm(dim=-1)

        w = 1.0 / knn_dist.clamp_min(eps)
        #
        w = w / w.sum(dim=-1, keepdim=True)

        x_neighbors = x_b[knn_idx]
        out[idx] = (w.unsqueeze(-1) * x_neighbors).sum(dim=1)

    return out



class MLP(nn.Module):
    """2-layer MLP with SiLU activation, matching grid-FLOWERS's TwoLayerMLP."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SelfWarpMesh(nn.Module):
    """One SELFWARP block, mesh-native."""

    def __init__(
        self,
        latent_dim: int,
        num_heads: int = 4,
        k_interp: int = 3,
        max_disp: float = 0.5,
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
        self.max_disp = max_disp
        self.norm_pre_warp = nn.LayerNorm(latent_dim)
        self.value_head = MLP(latent_dim, latent_dim, latent_dim)
        self.flow_head = MLP(latent_dim, latent_dim, num_heads * 2)
        self.out_proj = nn.Linear(latent_dim, latent_dim)
        # Start as an identity warp: displacements begin at zero and the model
        # learns to reach outward, rather than starting scattered and having to
        # find its way back.
        nn.init.zeros_(self.flow_head.net[-1].weight)
        nn.init.zeros_(self.flow_head.net[-1].bias)
        self.norm_pre_ffn = nn.LayerNorm(latent_dim)
        self.ffn = MLP(latent_dim, latent_dim * 2, latent_dim)

    def forward(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        batch_idx: torch.Tensor,
    ) -> torch.Tensor:
        residual = h
        x = self.norm_pre_warp(h)

        v = self.value_head(x)
        flow = self.flow_head(x).view(-1, self.num_heads, 2)
        v_heads = v.view(-1, self.num_heads, self.head_dim)

        # Bound displacements to a fraction of the domain. Without this the flow
        # head is unconstrained, and query points that land outside the mesh get
        # no useful gradient (the same distant nodes stay nearest however far you
        # push), so early blocks can settle into a degenerate far-field regime.
        # Scaling by the per-graph extent keeps max_disp dataset-independent.
        with torch.no_grad():
            extent = (pos.max(dim=0).values - pos.min(dim=0).values).max()
        flow = torch.tanh(self.flow_head(x)).view(-1, self.num_heads, 2)
        flow = flow * (self.max_disp * extent)

       # warped_heads = []
       # for h_idx in range(self.num_heads):
       #     q = pos + flow[:, h_idx]
       #     warped = _knn_interpolate(
       #         x=v_heads[:, h_idx, :],
       #         pos_src=pos,
       #         pos_qry=q,
       #         batch_idx=batch_idx,
       #         k=self.k_interp,
       #     )
       #     warped_heads.append(warped)

       # out = torch.stack(warped_heads, dim=1).view(-1, self.latent_dim)
       # out = self.out_proj(out)

        warped = _knn_interpolate_multihead(
            v_heads=v_heads,
            pos=pos,
            flow=flow,
            batch_idx=batch_idx,
            k=self.k_interp,
        )                                                   # [N, H, head_dim]
        out = self.out_proj(warped.reshape(-1, self.latent_dim))
        h = residual + out

        residual = h
        x = self.norm_pre_ffn(h)
        h = residual + self.ffn(x)

        return h
