"""Autoregressive rollout for evaluation.

Given a trained model and a full ground-truth trajectory, produce the model's
own predicted trajectory by feeding its previous prediction back in. Compare
against ground truth at various horizons.

Same rollout logic is used for BOTH MGN and FLOWERS. The model's job is just
'batch in -> normalized delta out'. Everything else — normalization, integration,
boundary handling — is here, so the comparison is apples-to-apples.

This mirrors MGN's cfd_eval.py:_rollout function, with two differences:
  1. We work in PyG Data / Batch objects instead of raw dicts.
  2. Normalization is explicit (MGN did it via online Normalizers inside the model).
"""

from __future__ import annotations
from typing import Dict
import torch
from torch_geometric.data import Batch, Data
from flowers_gnn.data.cylinder_dataset import NodeType
@torch.inference_mode()
def rollout_trajectory(
    model: torch.nn.Module,
    traj: Dict[str, torch.Tensor],
    stats: Dict[str, torch.Tensor],
    static: Data,
    max_steps: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Running the model autoregressively for max_steps.

    Args:
        model: any model satisfying our contract — takes a Batch, returns
               [total_N, 2] predicted normalized velocity delta.
        traj: dict with 'velocity' of shape (T, N, 2). Ground-truth trajectory.
        stats: precomputed normalization stats from norm_stats.pt.
        static: PyG Data with the static per-trajectory fields
                (pos, node_type, edge_index, loss_mask, cells).
        max_steps: how many steps to roll out. Capped at T-1.
        device: cuda/mps/cpu.

    Returns:
        {
          'pred':   (max_steps + 1, N, 2)  full predicted trajectory (includes v_0)
          'target': (max_steps + 1, N, 2)  ground-truth trajectory
        }
    """
    model.eval()

    v_gt = traj["velocity"].to(device)                       # (T, N, 2)
    T = v_gt.shape[0]
    n_steps = min(max_steps, T - 1)

    # Move static fields to device once
    pos = static.pos.to(device)
    node_type = static.node_type.to(device)
    edge_index = static.edge_index.to(device)
    loss_mask = static.loss_mask.to(device)

    # Nodes we update from prediction; others are held to ground truth (boundary
    # conditions are dictated by the simulator, not learned by us).
    update_mask = (
        (node_type == NodeType.NORMAL) | (node_type == NodeType.OUTFLOW)
    ).unsqueeze(-1)  # (N, 1)

    # Normalization stats -> device
    v_mean = stats["velocity_mean"].to(device)
    v_std = stats["velocity_std"].to(device)
    d_mean = stats["delta_mean"].to(device)
    d_std = stats["delta_std"].to(device)

    # Storage for the full predicted trajectory (include v_0 so shapes align with GT)
    pred = torch.empty(n_steps + 1, v_gt.shape[1], 2, device=device)
    pred[0] = v_gt[0]

    cur = v_gt[0].clone()   # (N, 2) in physical units

    for t in range(n_steps):
        # Build a single-graph "batch" for the model. We normalize x here
        # because the model always sees normalized velocity.
        x_norm = (cur - v_mean) / v_std
        data = Data(
            x=x_norm,
            pos=pos,
            node_type=node_type,
            edge_index=edge_index,
            loss_mask=loss_mask,
        )
        batch = Batch.from_data_list([data]).to(device)

        # Model returns predicted NORMALIZED delta
        delta_norm = model(batch)                      # (N, 2)

        # Denormalize the delta into physical units
        delta_phys = delta_norm * d_std + d_mean

        # Integrate
        pred_next = cur + delta_phys #(N,2), all-nodes prediction

        # Boundary handling: hold non-updateable nodes to their ground-truth value
        cur = torch.where(update_mask, pred_next, v_gt[t + 1])

        pred[t + 1] = cur

    target = v_gt[: n_steps + 1]

    return {"pred": pred, "target": target}