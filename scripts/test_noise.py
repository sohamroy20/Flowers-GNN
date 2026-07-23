"""Verify training noise applies only to NORMAL nodes and respects the std config."""
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

from flowers_gnn.data.cylinder_dataset import NodeType
from flowers_gnn.data.datamodule import CylinderDataModule
from flowers_gnn.training.noise import apply_training_noise


def main():
    torch.manual_seed(0)
    config_dir = str(Path("configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config")

    dm = CylinderDataModule(cfg.data)
    batch = next(iter(dm.valid_loader()))
    x_before = batch.x.clone()
    y_before = batch.y.clone()

    apply_training_noise(batch, std=0.02, gamma=1.0)

    diff = (batch.x - x_before).abs()
    normal_nodes = (batch.node_type == NodeType.NORMAL)
    other_nodes = ~normal_nodes

    print(f"Nodes total:     {batch.x.shape[0]}")
    print(f"NORMAL nodes:    {int(normal_nodes.sum())}")
    print(f"Other nodes:     {int(other_nodes.sum())}")
    print()
    print(f"Max |diff| on NORMAL nodes: {diff[normal_nodes].max().item():.4f}  (expect ~0.05-0.10, i.e. a few std)")
    print(f"Max |diff| on OTHER nodes:  {diff[other_nodes].max().item():.6f}  (expect exactly 0)")
    print(f"Std of diff on NORMAL nodes (vx channel): {diff[normal_nodes, 0].std().item():.4f}  (expect ~0.02)")
    print()
    print(f"Target y unchanged (gamma=1.0): max |y_diff| = {(batch.y - y_before).abs().max().item():.6f}  (expect 0)")


if __name__ == "__main__":
    main()