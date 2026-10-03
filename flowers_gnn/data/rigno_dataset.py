"""PyG dataset for the RIGNO / PDEGym NetCDF benchmarks.

Files hold two arrays:
    u : (n_traj, n_time, n_nodes, n_fields)   solution field
    x : (1, 1, n_nodes, 2)                    node coordinates, shared by all
                                              trajectories

No connectivity ships with these files, so both datasets are point clouds.
MGN needs edges, so we build a symmetric k-NN graph once from the shared
coordinates and reuse it for every sample. FLOWERS-mesh ignores edge_index
and works from positions alone.

Returns the same Data contract as CylinderFlowDataset (x, y, pos, node_type,
edge_index, loss_mask) so both models run unchanged. node_type is all-NORMAL
and loss_mask all-True: these domains are periodic or closed, with no
per-node boundary tagging in the data.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch_geometric.data import Data, Dataset


def knn_edge_index(pos: torch.Tensor, k: int = 6) -> torch.Tensor:
    """Symmetric k-NN graph, self-loops removed. Returns [2, E] (senders, receivers)."""
    from torch_cluster import knn_graph
    ei = knn_graph(pos, k=k, loop=False)
    # Symmetrise: keep both (i,j) and (j,i), deduplicated.
    both = torch.cat([ei, ei.flip(0)], dim=1)
    key = both[0] * pos.shape[0] + both[1]
    _, keep = torch.unique(key, return_inverse=False, return_counts=False), None
    uniq = torch.unique(key)
    src = uniq // pos.shape[0]
    dst = uniq % pos.shape[0]
    return torch.stack([src, dst], dim=0).long()


class RignoDataset(Dataset):
    """One (frame_t, frame_t+1) pair per item, across all trajectories in a split."""

    def __init__(
        self,
        nc_path: str | Path,
        split: str,
        knn_k: int = 6,
        n_train: int = 1024,
        n_valid: int = 128,
        n_test: int = 128,
        max_trajectories: Optional[int] = None,
        in_memory: bool = True,
    ):
        super().__init__(root=None, transform=None, pre_transform=None)

        import xarray as xr

        self.nc_path = Path(nc_path)
        self.split = split

        ds = xr.open_dataset(self.nc_path)
        u = ds["u"]                                    # (T_traj, T, N, F)
        x = np.asarray(ds["x"]).reshape(-1, 2)         # (N, 2)

        n_traj_total = u.shape[0]
        bounds = {
            "train": (0, n_train),
            "valid": (n_train, n_train + n_valid),
            "test":  (n_train + n_valid, n_train + n_valid + n_test),
        }
        if split not in bounds:
            raise ValueError(f"split must be train/valid/test, got {split}")
        lo, hi = bounds[split]
        hi = min(hi, n_traj_total)
        if max_trajectories is not None:
            hi = min(hi, lo + max_trajectories)
        self.traj_slice = (lo, hi)

        if in_memory:
            self.u = torch.from_numpy(np.asarray(u[lo:hi])).float()
        else:
            raise NotImplementedError("lazy loading not implemented; use in_memory=True")
        ds.close()

        self.n_traj, self.n_time, self.n_nodes, self.n_fields = self.u.shape
        self.pos = torch.from_numpy(x).float()

        # Static graph, built once from the shared coordinates.
        self.edge_index = knn_edge_index(self.pos, k=knn_k)
        self.node_type = torch.zeros(self.n_nodes, dtype=torch.long)
        self.loss_mask = torch.ones(self.n_nodes, dtype=torch.bool)

        self._frames_per_traj = self.n_time - 1
        self._total = self.n_traj * self._frames_per_traj

        print(f"RignoDataset[{split}] {self.nc_path.name}: "
              f"{self.n_traj} traj x {self.n_time} steps x {self.n_nodes} nodes "
              f"x {self.n_fields} fields -> {self._total} samples, "
              f"{self.edge_index.shape[1]} edges (k={knn_k})")

    def len(self) -> int:
        return self._total

    def get(self, idx: int) -> Data:
        traj = idx // self._frames_per_traj
        t = idx % self._frames_per_traj
        return Data(
            x=self.u[traj, t],
            y=self.u[traj, t + 1] - self.u[traj, t],
            pos=self.pos,
            node_type=self.node_type,
            edge_index=self.edge_index,
            loss_mask=self.loss_mask,
        )

    def trajectory(self, i: int) -> dict:
        """Full trajectory i of this split, for rollout evaluation."""
        return {"velocity": self.u[i]}

    def static(self) -> Data:
        return Data(
            pos=self.pos, node_type=self.node_type,
            edge_index=self.edge_index, loss_mask=self.loss_mask,
        )
