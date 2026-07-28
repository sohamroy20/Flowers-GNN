"""Verifying MGN instantiates, forward-passes, and produces the expected shapes.

No training. Just: config -> instantiate -> one batch -> forward -> check.
"""
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import hydra

from flowers_gnn.data.datamodule import CylinderDataModule


def main():
    torch.manual_seed(0)

    config_dir = str(Path("configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "model=mgn",
                "data.max_valid_trajectories=1",
                "data.batch_size=2",
                "data.num_workers=0",
            ],
        )

    print(OmegaConf.to_yaml(cfg.model))
    print()

    model = hydra.utils.instantiate(cfg.model.net)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model.__class__.__name__}")
    print(f"Total params: {n_params:,}")
    print()

    dm = CylinderDataModule(cfg.data)
    batch = next(iter(dm.valid_loader()))
    print(f"Batch: {batch.num_graphs} graphs, x shape {tuple(batch.x.shape)}, "
          f"edge_index {tuple(batch.edge_index.shape)}")

    with torch.no_grad():
        pred = model(batch)
    print(f"Prediction shape: {tuple(pred.shape)}   (should match batch.x)")
    print(f"Prediction min/max: {pred.min().item():+.4f} / {pred.max().item():+.4f}   "
          "(random-init, no meaning yet)")


if __name__ == "__main__":
    main()