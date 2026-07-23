"""Builds train/valid/test DataLoaders from a Hydra data config.

Kept small on purpose. The dataset class does the heavy lifting.
"""

from pathlib import Path
from typing import Optional

import torch
from omegaconf import DictConfig
from torch_geometric.loader import DataLoader

from flowers_gnn.data.cylinder_dataset import CylinderFlowDataset


class CylinderDataModule:
    """Holds datasets and hands out DataLoaders. Also loads normalization stats."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.root = Path(cfg.root)

        # Load precomputed stats
        stats_path = Path(cfg.norm_stats_path)
        if not stats_path.exists():
            raise FileNotFoundError(
                f"norm_stats not found at {stats_path}. "
                f"Run: python scripts/compute_norm_stats.py --dataset cylinder_flow"
            )
        self.stats = torch.load(stats_path, weights_only=True)

    def train_dataset(self) -> CylinderFlowDataset:
        return CylinderFlowDataset(
            root=self.root,
            split="train",
            in_memory=self.cfg.train_in_memory,
            max_trajectories=self.cfg.max_train_trajectories,
        )

    def valid_dataset(self) -> CylinderFlowDataset:
        return CylinderFlowDataset(
            root=self.root,
            split="valid",
            in_memory=self.cfg.valid_in_memory,
            max_trajectories=self.cfg.max_valid_trajectories,
        )

    def test_dataset(self) -> CylinderFlowDataset:
        return CylinderFlowDataset(
            root=self.root,
            split="test",
            in_memory=self.cfg.test_in_memory,
            max_trajectories=self.cfg.max_test_trajectories,
        )

    def train_loader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset(),
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            drop_last=True,
        )

    def valid_loader(self) -> DataLoader:
        return DataLoader(
            self.valid_dataset(),
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
        )