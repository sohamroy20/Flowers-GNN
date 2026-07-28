"""PyG dataset for CylinderFlow.

Reads per-trajectory .pt files produced by scripts/convert_tfrecord_to_torch.py.
Each sample is one frame (timestep) of one trajectory, returned as a PyG Data
object that any mesh-native model can consume.

Per-frame bundle:
    x          [N, 2]   velocity at time t           (input)
    y          [N, 2]   velocity(t+1) - velocity(t)  (target delta)
    pos        [N, 2]   mesh_pos                     (static, same per traj)
    node_type  [N]      int, see NodeType enum
    edge_index [2, E]   undirected bidirectional edges from cells
    loss_mask  [N]      bool, True for NORMAL or OUTFLOW
    cells      [M, 3]   triangles, kept for eval/viz
"""

from __future__ import annotations

import enum
from pathlib import Path
from typing import List, Optional

import torch
from torch_geometric.data import Data, Dataset


# MGN node type enum (referred from meshgraphnets/common.py)
class NodeType(enum.IntEnum):
    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    SIZE = 9


def triangles_to_edges(cells: torch.Tensor) -> torch.Tensor:
    """Cells (M, 3) -> bidirectional, deduplicated edge_index (2, E).

    Port of meshgraphnets/common.py:triangles_to_edges. Each undirected edge
    appears in BOTH directions in edge_index so that message passing reaches
    both endpoints.
    """
    # All directed edges from each triangle: (0,1), (1,2), (2,0)
    e = torch.cat([cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]], dim=0)
    # Canonicalize as (min, max) so duplicates within the mesh dedupe correctly
    a = torch.minimum(e[:, 0], e[:, 1])
    b = torch.maximum(e[:, 0], e[:, 1])
    undirected = torch.stack([a, b], dim=1)
    undirected = torch.unique(undirected, dim=0)
    senders = undirected[:, 0]
    receivers = undirected[:, 1]
    # Bidirectional: concat both orientations
    edge_index = torch.stack(
        [torch.cat([senders, receivers]), torch.cat([receivers, senders])],
        dim=0,
    ).long()
    return edge_index


class CylinderFlowDataset(Dataset):
    """One sample = one frame. One trajectory contributes T-1 frames (need t+1 for target).

    Args:
        root:       Path to the dataset folder, e.g. data/cylinder_flow
        split:      'train', 'valid', or 'test'
        in_memory:  If True, load all trajectories into RAM up front. Fast for valid/test.
                    For train this can be ~12 GB of RAM. Default False = lazy load per __getitem__.
        max_trajectories: Optional cap, useful for quick smoke tests.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        in_memory: bool = False,
        max_trajectories: Optional[int] = None,
    ):
        # PyG Dataset is a bit picky — we override what we need and skip the rest.
        super().__init__(root=None, transform=None, pre_transform=None)

        self.root = Path(root)
        self.split = split
        self.in_memory = in_memory

        split_dir = self.root / "torch" / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"{split_dir} not found. Run scripts/convert_tfrecord_to_torch.py first."
            )

        self.traj_paths: List[Path] = sorted(split_dir.glob("traj_*.pt"))
        if max_trajectories is not None:
            self.traj_paths = self.traj_paths[:max_trajectories]
        if not self.traj_paths:
            raise RuntimeError(f"No traj_*.pt files in {split_dir}")

        # Per-trajectory cached static stuff (mesh, edges, masks, length)
        # Always cache static structure — it's tiny. Only optionally cache full velocity.
        self._traj_static: List[dict] = []
        self._traj_velocity: List[Optional[torch.Tensor]] = []

        from tqdm import tqdm
        for p in tqdm(self.traj_paths,
                      desc=f"loading {split} ({'in-memory' if in_memory else 'metadata-only'})",
                      leave=False):

            blob = torch.load(p, weights_only=True)
            mesh_pos = blob["mesh_pos"][0]            # (N, 2)
            cells = blob["cells"][0].long()           # (M, 3)
            node_type = blob["node_type"][0, :, 0].long()  # (N,)
            edge_index = triangles_to_edges(cells)    # (2, E)
            loss_mask = (
                (node_type == NodeType.NORMAL) | (node_type == NodeType.OUTFLOW)
            )

            self._traj_static.append({
                "pos": mesh_pos.float(),
                "cells": cells,
                "node_type": node_type,
                "edge_index": edge_index,
                "loss_mask": loss_mask,
                "T": int(blob["velocity"].shape[0]),
            })
            if in_memory:
                self._traj_velocity.append(blob["velocity"].float())
            else:
                self._traj_velocity.append(None)

        # Number of frames per trajectory is T-1 (need v_{t+1} for the target)
        self._frames_per_traj = [s["T"] - 1 for s in self._traj_static]
        # Cumulative offsets for index -> (traj, frame) lookup
        self._cum_offsets = [0]
        for n in self._frames_per_traj:
            self._cum_offsets.append(self._cum_offsets[-1] + n)
        self._total_frames = self._cum_offsets[-1]

    # ---- PyG Dataset interface (minimal) -------------------------------------------------

    def len(self) -> int:
        return self._total_frames

    def get(self, idx: int) -> Data:
        # Resolve global idx -> (traj_idx, frame_t)
        traj_idx = _bisect(self._cum_offsets, idx)
        frame_t = idx - self._cum_offsets[traj_idx]

        static = self._traj_static[traj_idx]
        velocity = self._traj_velocity[traj_idx]
        if velocity is None:
            # Lazy load this trajectory's full velocity tensor
            blob = torch.load(self.traj_paths[traj_idx], weights_only=True)
            velocity = blob["velocity"].float()
            # NOTE: not caching in self._traj_velocity to keep RAM bounded.

        v_t = velocity[frame_t]          # (N, 2)
        v_tp1 = velocity[frame_t + 1]    # (N, 2)
        y = v_tp1 - v_t

        data = Data(
            x=v_t,
            y=y,
            pos=static["pos"],
            node_type=static["node_type"],
            edge_index=static["edge_index"],
            loss_mask=static["loss_mask"],
            cells=static["cells"],
        )
        return data


def _bisect(cum_offsets: List[int], idx: int) -> int:
    """Return the largest i such that cum_offsets[i] <= idx (one less than bisect_right)."""
    lo, hi = 0, len(cum_offsets) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if cum_offsets[mid] <= idx:
            lo = mid
        else:
            hi = mid - 1
    return lo